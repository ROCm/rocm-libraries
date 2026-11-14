// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_,@function
RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 1438) 
s_mul_hi_i32 s1, s16, s11 // (op 1680) 
v_lshrrev_b32 v0, 7, v1 // (op 1495) 
v_and_b32 v3, 15, v1 // (op 1495) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1495) 


v_and_b32 v0, 63, v1 // (op 1495) 
v_lshrrev_b32 v3, 4, v0 // (op 1495) 

v_lshlrev_b32 v0, 5, v3 // (op 1495) 

v_lshl_add_u32 v3, v4, 8, v0 // (op 1495) 


v_mul_lo_u32 v0, v3, 6 // (op 1495) 

v_lshrrev_b32 v3, 3, v0 // (op 1495) 

v_lshrrev_b32 v0, 7, v1 // (op 1606) 
v_and_b32 v4, 15, v1 // (op 1606) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1606) 


v_and_b32 v0, 63, v1 // (op 1606) 
v_lshrrev_b32 v4, 4, v0 // (op 1606) 

v_lshl_add_u32 v0, v5, 3, v4 // (op 1606) 


v_lshrrev_b32 v4, 7, v1 // (op 1706) 
v_and_b32 v5, 15, v1 // (op 1706) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1706) 


v_and_b32 v4, 63, v1 // (op 1706) 
v_lshrrev_b32 v5, 4, v4 // (op 1706) 

v_lshlrev_b32 v4, 5, v5 // (op 1706) 

v_lshl_add_u32 v5, v6, 8, v4 // (op 1706) 


v_mul_lo_u32 v4, v5, 6 // (op 1706) 

v_lshrrev_b32 v5, 3, v4 // (op 1706) 

v_lshrrev_b32 v4, 7, v1 // (op 1759) 
v_and_b32 v6, 15, v1 // (op 1759) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 1759) 


v_and_b32 v4, 63, v1 // (op 1759) 
v_lshrrev_b32 v6, 4, v4 // (op 1759) 

v_lshl_add_u32 v4, v7, 3, v6 // (op 1759) 


v_lshrrev_b32 v6, 7, v1 // (op 1438) 
s_add_i32 s18, s0, s16 // (op 1438) 

v_and_b32 v7, 63, v1 // (op 1438) 
s_ashr_i32 s0, s18, 31 // (op 1438) 
v_lshrrev_b32 v8, 4, v7 // (op 1438) 
s_cmp_eq_i32 s11, 0 // (op 1438) 
s_mov_b32 s19, scc // (op 1438) 
v_lshlrev_b32 v9, 2, v8 // (op 1438) 

s_cmp_eq_i32 s19, 0 // (op 1438) 
s_cselect_b32 s51, 0, -1 // (op 1438) 

v_lshl_add_u32 v8, v6, 4, v9 // (op 1438) 


s_and_b32 s19, s12, 31 // (op 1438) 
v_lshrrev_b32 v6, 2, v7 // (op 1438) 

v_and_b32 v7, 3, v6 // (op 1438) 

v_and_b32 v6, 63, v1 // (op 1553) 
v_and_b32 v9, 48, v1 // (op 1553) 
v_sub_u32 v10, v6, v9 // (op 1553) 

v_lshrrev_b32 v9, 4, v6 // (op 1553) 

v_lshl_add_u32 v6, v9, 5, v10 // (op 1553) 


v_lshrrev_b32 v9, 6, v1 // (op 1553) 
v_and_b32 v10, 1, v9 // (op 1553) 

v_lshlrev_b32 v9, 4, v10 // (op 1553) 

v_lshl_add_u32 v10, v6, 6, v9 // (op 1553) 


v_mul_lo_u32 v6, v10, 6 // (op 1553) 
v_lshrrev_b32 v9, 3, v6 // (op 1553) 

v_lshrrev_b32 v6, 4, v10 // (op 1553) 

v_lshlrev_b32 v10, 2, v6 // (op 1553) 

v_add_u32 v6, v9, v10 // (op 1553) 


v_lshrrev_b32 v9, 6, v1 // (op 1664) 
v_and_b32 v10, 1, v9 // (op 1664) 

v_and_b32 v9, 15, v1 // (op 1664) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 1664) 


v_and_b32 v9, 63, v1 // (op 1664) 
v_lshrrev_b32 v10, 4, v9 // (op 1664) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 1664) 


v_lshrrev_b32 v10, 7, v1 // (op 1680) 
v_and_b32 v11, 63, v1 // (op 1680) 
v_and_b32 v12, 63, v1 // (op 1735) 
v_and_b32 v13, 48, v1 // (op 1735) 
v_sub_u32 v14, v12, v13 // (op 1735) 

v_lshrrev_b32 v13, 4, v12 // (op 1735) 

v_lshl_add_u32 v12, v13, 5, v14 // (op 1735) 


v_lshrrev_b32 v13, 6, v1 // (op 1735) 
v_and_b32 v14, 1, v13 // (op 1735) 

v_lshlrev_b32 v13, 4, v14 // (op 1735) 

v_lshl_add_u32 v14, v12, 6, v13 // (op 1735) 


v_mul_lo_u32 v12, v14, 6 // (op 1735) 
v_lshrrev_b32 v13, 3, v12 // (op 1735) 

v_lshrrev_b32 v12, 4, v14 // (op 1735) 

v_lshlrev_b32 v14, 2, v12 // (op 1735) 

v_add_u32 v12, v13, v14 // (op 1735) 


v_lshrrev_b32 v13, 6, v1 // (op 1788) 
v_and_b32 v14, 1, v13 // (op 1788) 

v_and_b32 v13, 15, v1 // (op 1788) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 1788) 


v_and_b32 v13, 63, v1 // (op 1788) 
v_lshrrev_b32 v14, 4, v13 // (op 1788) 

v_lshl_add_u32 v13, v14, 6, v15 // (op 1788) 


s_lshl_b32 s52, 1, s19 // (op 1438) 
v_and_b32 v14, 3, v1 // (op 1438) 
s_add_i32 s53, s52, s51 // (op 1438) 


s_and_b32 s51, s0, s53 // (op 1438) 


s_add_i32 s0, s1, s16 // (op 1680) 

s_ashr_i32 s1, s0, 31 // (op 1680) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 1438) 


s_add_i32 s52, s18, s51 // (op 1438) 


v_lshrrev_b32 v7, 6, v1 // (op 1438) 
s_ashr_i32 s18, s52, s19 // (op 1438) 


v_and_b32 v14, 1, v7 // (op 1438) 

s_lshl_b32 s19, s12, 24 // (op 1438) 
v_lshl_add_u32 v7, v14, 4, v15 // (op 1438) 


s_ashr_i32 s51, s19, 31 // (op 1438) 

s_xor_b32 s19, s18, s51 // (op 1438) 

s_sub_i32 s18, s19, s51 // (op 1438) 


s_mul_hi_i32 s19, s18, s13 // (op 1438) 
s_add_i32 s51, s19, s18 // (op 1438) 

s_ashr_i32 s19, s51, 31 // (op 1438) 
s_cmp_eq_i32 s13, 0 // (op 1438) 
s_mov_b32 s52, scc // (op 1438) 
s_cmp_eq_i32 s52, 0 // (op 1438) 
s_cselect_b32 s53, 0, -1 // (op 1438) 

s_and_b32 s52, s14, 31 // (op 1438) 
v_lshrrev_b32 v14, 4, v11 // (op 1680) 
s_cmp_eq_i32 s11, 0 // (op 1680) 
s_mov_b32 s54, scc // (op 1680) 
v_lshlrev_b32 v15, 2, v14 // (op 1680) 

s_cmp_eq_i32 s54, 0 // (op 1680) 
s_cselect_b32 s55, 0, -1 // (op 1680) 

s_lshl_b32 s54, 1, s52 // (op 1438) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 1680) 


v_lshrrev_b32 v10, 2, v11 // (op 1680) 

s_add_i32 s56, s54, s53 // (op 1438) 


s_and_b32 s53, s19, s56 // (op 1438) 


s_add_i32 s19, s51, s53 // (op 1438) 


s_ashr_i32 s51, s19, s52 // (op 1438) 


s_lshl_b32 s19, s14, 24 // (op 1438) 
s_ashr_i32 s52, s19, 31 // (op 1438) 

s_xor_b32 s19, s51, s52 // (op 1438) 

s_sub_i32 s51, s19, s52 // (op 1438) 


s_and_b32 s19, s12, 31 // (op 1680) 
s_lshl_b32 s52, 1, s19 // (op 1680) 
v_and_b32 v11, 3, v10 // (op 1680) 

s_add_i32 s53, s52, s55 // (op 1680) 


v_and_b32 v10, 3, v1 // (op 1680) 
s_and_b32 s52, s1, s53 // (op 1680) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 1680) 


s_add_i32 s1, s0, s52 // (op 1680) 


v_lshrrev_b32 v10, 6, v1 // (op 1680) 
s_ashr_i32 s0, s1, s19 // (op 1680) 


v_and_b32 v11, 1, v10 // (op 1680) 

s_lshl_b32 s1, s12, 24 // (op 1680) 
v_lshl_add_u32 v10, v11, 4, v15 // (op 1680) 


s_ashr_i32 s19, s1, 31 // (op 1680) 

s_xor_b32 s1, s0, s19 // (op 1680) 

s_sub_i32 s0, s1, s19 // (op 1680) 


s_mul_hi_i32 s1, s0, s13 // (op 1680) 
s_add_i32 s19, s1, s0 // (op 1680) 

s_ashr_i32 s1, s19, 31 // (op 1680) 
s_cmp_eq_i32 s13, 0 // (op 1680) 
s_mov_b32 s52, scc // (op 1680) 
s_cmp_eq_i32 s52, 0 // (op 1680) 
s_cselect_b32 s53, 0, -1 // (op 1680) 

s_and_b32 s52, s14, 31 // (op 1680) 
s_lshl_b32 s54, 1, s52 // (op 1680) 
s_add_i32 s55, s54, s53 // (op 1680) 


s_and_b32 s53, s1, s55 // (op 1680) 


s_add_i32 s1, s19, s53 // (op 1680) 


s_ashr_i32 s19, s1, s52 // (op 1680) 


s_lshl_b32 s1, s14, 24 // (op 1680) 
s_ashr_i32 s52, s1, 31 // (op 1680) 

s_xor_b32 s1, s19, s52 // (op 1680) 

s_sub_i32 s19, s1, s52 // (op 1680) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 1438) 
s_sub_u32 s52, s16, s1 // (op 1438) 

s_mul_i32 s1, s51, s44 // (op 1438) 
s_add_u32 s1, s1, s52 // (op 1438) 

s_mul_hi_i32 s52, s16, s10 // (op 1438) 
s_add_i32 s53, s52, s16 // (op 1438) 

s_ashr_i32 s52, s53, 31 // (op 1438) 
s_cmp_eq_i32 s10, 0 // (op 1438) 
s_mov_b32 s54, scc // (op 1438) 
s_cmp_eq_i32 s54, 0 // (op 1438) 
s_cselect_b32 s55, 0, -1 // (op 1438) 

s_and_b32 s54, s47, 31 // (op 1438) 
s_lshl_b32 s56, 1, s54 // (op 1438) 
s_add_i32 s57, s56, s55 // (op 1438) 


s_and_b32 s55, s52, s57 // (op 1438) 


s_add_i32 s52, s53, s55 // (op 1438) 


s_ashr_i32 s53, s52, s54 // (op 1438) 


s_lshl_b32 s52, s47, 24 // (op 1438) 
s_ashr_i32 s54, s52, 31 // (op 1438) 

s_xor_b32 s52, s53, s54 // (op 1438) 

s_sub_i32 s53, s52, s54 // (op 1438) 


s_mul_i32 s52, s0, s44 // (op 1680) 
s_sub_u32 s54, s16, s52 // (op 1680) 

s_mul_i32 s52, s19, s44 // (op 1680) 
s_add_u32 s52, s52, s54 // (op 1680) 

s_mul_hi_i32 s54, s16, s10 // (op 1680) 
s_add_i32 s55, s54, s16 // (op 1680) 

s_ashr_i32 s54, s55, 31 // (op 1680) 
s_cmp_eq_i32 s10, 0 // (op 1680) 
s_mov_b32 s56, scc // (op 1680) 
s_cmp_eq_i32 s56, 0 // (op 1680) 
s_cselect_b32 s57, 0, -1 // (op 1680) 

s_and_b32 s56, s47, 31 // (op 1680) 
s_lshl_b32 s58, 1, s56 // (op 1680) 
s_add_i32 s59, s58, s57 // (op 1680) 


s_and_b32 s57, s54, s59 // (op 1680) 


s_add_i32 s54, s55, s57 // (op 1680) 


s_ashr_i32 s55, s54, s56 // (op 1680) 


s_lshl_b32 s54, s47, 24 // (op 1680) 
s_ashr_i32 s56, s54, 31 // (op 1680) 

s_xor_b32 s54, s55, s56 // (op 1680) 

s_sub_i32 s55, s54, s56 // (op 1680) 


s_add_u32 s56, s6, 64 // (op 1438) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1438) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1438) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1438) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1438) 

s_mul_i32 s54, s44, s56 // (op 1438) 
s_mul_i32 s58, s48, s54 // (op 1438) 

s_mul_i32 s54, s53, s58 // (op 1438) 

s_sub_u32 s58, s16, s54 // (op 1438) 

s_mov_b32 s54, s58 // (op 1438) 
s_mul_hi_i32 s59, s54, s15 // (op 1438) 
s_add_i32 s60, s59, s54 // (op 1438) 


s_ashr_i32 s54, s60, 31 // (op 1438) 
s_cmp_eq_i32 s15, 0 // (op 1438) 
s_mov_b32 s59, scc // (op 1438) 
s_cmp_eq_i32 s59, 0 // (op 1438) 
s_cselect_b32 s61, 0, -1 // (op 1438) 

s_and_b32 s59, s46, 31 // (op 1438) 
s_lshl_b32 s62, 1, s59 // (op 1438) 
s_add_i32 s63, s62, s61 // (op 1438) 


s_and_b32 s61, s54, s63 // (op 1438) 


s_add_i32 s54, s60, s61 // (op 1438) 


s_ashr_i32 s60, s54, s59 // (op 1438) 


s_lshl_b32 s54, s46, 24 // (op 1438) 
s_ashr_i32 s59, s54, 31 // (op 1438) 

s_xor_b32 s54, s60, s59 // (op 1438) 

s_sub_i32 s60, s54, s59 // (op 1438) 


s_mul_i32 s54, s45, s44 // (op 1438) 
s_add_u32 s62, s8, 16 // (op 1438) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 1438) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1438) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1438) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1438) 

s_sub_i32 s59, s62, s54 // (op 1438) 

s_mul_i32 s61, s60, s59 // (op 1438) 

s_sub_u32 s59, s58, s61 // (op 1438) 


s_add_u32 s58, s59, s54 // (op 1438) 


s_cmp_eq_i32 s53, 0 // (op 1438) 
s_mov_b32 s54, scc // (op 1438) 

s_cmp_eq_i32 s54, 0 // (op 1438) 
s_cselect_b32 s53, s58, s1 // (op 1438) 


s_mul_i32 s1, s51, s56 // (op 1438) 


v_lshl_add_u32 v11, s53, 4, v8 // (op 1438) 


s_sub_i32 s51, s18, s1 // (op 1438) 


s_cmp_eq_i32 s54, 0 // (op 1438) 
s_cselect_b32 s1, s60, s51 // (op 1438) 



v_lshl_add_u32 v8, s1, 6, v7 // (op 1438) 


v_mul_lo_u32 v7, v8, s2 // (op 1438) 

v_add_lshl_u32 v8, v11, v7, 2 // (op 1438) 


s_add_u32 s56, s6, 64 // (op 1680) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1680) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1680) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1680) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1680) 

s_mul_i32 s1, s44, s56 // (op 1680) 
s_mul_i32 s18, s48, s1 // (op 1680) 

s_mul_i32 s1, s55, s18 // (op 1680) 

s_sub_u32 s18, s16, s1 // (op 1680) 

s_mov_b32 s1, s18 // (op 1680) 
s_mul_hi_i32 s51, s1, s15 // (op 1680) 
s_add_i32 s53, s51, s1 // (op 1680) 


s_ashr_i32 s1, s53, 31 // (op 1680) 
s_cmp_eq_i32 s15, 0 // (op 1680) 
s_mov_b32 s51, scc // (op 1680) 
s_cmp_eq_i32 s51, 0 // (op 1680) 
s_cselect_b32 s54, 0, -1 // (op 1680) 

s_and_b32 s51, s46, 31 // (op 1680) 
s_lshl_b32 s58, 1, s51 // (op 1680) 
s_add_i32 s59, s58, s54 // (op 1680) 


s_and_b32 s54, s1, s59 // (op 1680) 


s_add_i32 s1, s53, s54 // (op 1680) 


s_ashr_i32 s53, s1, s51 // (op 1680) 


s_lshl_b32 s1, s46, 24 // (op 1680) 
s_ashr_i32 s51, s1, 31 // (op 1680) 

s_xor_b32 s1, s53, s51 // (op 1680) 

s_sub_i32 s53, s1, s51 // (op 1680) 


s_mul_i32 s1, s45, s44 // (op 1680) 
s_add_u32 s58, s8, 16 // (op 1680) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 1680) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1680) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1680) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1680) 

s_sub_i32 s51, s58, s1 // (op 1680) 

s_mul_i32 s54, s53, s51 // (op 1680) 

s_sub_u32 s51, s18, s54 // (op 1680) 


s_add_u32 s18, s51, s1 // (op 1680) 


s_cmp_eq_i32 s55, 0 // (op 1680) 
s_mov_b32 s1, scc // (op 1680) 

s_cmp_eq_i32 s1, 0 // (op 1680) 
s_cselect_b32 s51, s18, s52 // (op 1680) 


s_mul_i32 s18, s19, s56 // (op 1680) 


v_lshl_add_u32 v7, s51, 4, v14 // (op 1680) 


s_sub_i32 s19, s0, s18 // (op 1680) 


s_cmp_eq_i32 s1, 0 // (op 1680) 
s_cselect_b32 s0, s53, s19 // (op 1680) 



v_lshl_add_u32 v11, s0, 6, v10 // (op 1680) 


v_mul_lo_u32 v10, v11, s4 // (op 1680) 

v_add_lshl_u32 v11, v7, v10, 2 // (op 1680) 


s_mul_i32 s1, s2, s7 // (op 1440) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1440) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1440) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1440) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1440) most significant: sum
s_add_u32 s1, s1, s19 // (op 1440) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1440) 

s_mul_i32 s1, s4, s7 // (op 1682) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 1682) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1682) least significant
s_mul_hi_u32 s52, s4, s6 // (op 1682) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1682) most significant: sum
s_add_u32 s1, s1, s52 // (op 1682) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1682) 

s_mov_b32 s56, s20 // (op 1440) reg expression
s_mov_b32 s57, s21 // (op 1440) reg expression
s_mov_b32 s58, s18 // (op 1440) reg expression
s_mov_b32 s59, 131072 // (op 1440) call()

s_mov_b32 s60, s22 // (op 1682) reg expression

s_mov_b32 s61, s23 // (op 1682) reg expression
s_mov_b32 s62, s52 // (op 1682) reg expression
s_mov_b32 s63, 131072 // (op 1682) call()


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
s_mul_hi_i32 s0, s16, s11 // (op 1205) 
v_lshrrev_b32 v14, 3, v1 // (op 1205) 
s_add_i32 s1, s0, s16 // (op 1205) 

v_and_b32 v15, -8, v1 // (op 1205) 
s_ashr_i32 s0, s1, 31 // (op 1205) 
v_sub_u32 v16, v1, v15 // (op 1205) 

s_cmp_eq_i32 s11, 0 // (op 1205) 
s_mov_b32 s18, scc // (op 1205) 
v_lshlrev_b32 v15, 5, v16 // (op 1205) 

s_cmp_eq_i32 s18, 0 // (op 1205) 
s_cselect_b32 s19, 0, -1 // (op 1205) 

v_mov_b32 v17, 0 // (op 1205) convert
v_mov_b32 v16, v15 // (op 1205) convert

s_and_b32 s18, s12, 31 // (op 1205) 
s_lshl_b32 s20, 1, s18 // (op 1205) 
s_add_i32 s21, s20, s19 // (op 1205) 


s_and_b32 s19, s0, s21 // (op 1205) 


s_add_i32 s0, s1, s19 // (op 1205) 


s_ashr_i32 s1, s0, s18 // (op 1205) 


s_lshl_b32 s0, s12, 24 // (op 1205) 
s_ashr_i32 s18, s0, 31 // (op 1205) 

s_xor_b32 s0, s1, s18 // (op 1205) 

s_sub_i32 s1, s0, s18 // (op 1205) 


s_mul_hi_i32 s0, s1, s13 // (op 1205) 
s_add_i32 s18, s0, s1 // (op 1205) 

s_ashr_i32 s0, s18, 31 // (op 1205) 
s_cmp_eq_i32 s13, 0 // (op 1205) 
s_mov_b32 s19, scc // (op 1205) 
s_cmp_eq_i32 s19, 0 // (op 1205) 
s_cselect_b32 s20, 0, -1 // (op 1205) 

s_and_b32 s19, s14, 31 // (op 1205) 
s_lshl_b32 s21, 1, s19 // (op 1205) 
s_add_i32 s22, s21, s20 // (op 1205) 


s_and_b32 s23, s0, s22 // (op 1205) 


s_add_i32 s0, s18, s23 // (op 1205) 


s_ashr_i32 s18, s0, s19 // (op 1205) 


s_lshl_b32 s0, s14, 24 // (op 1205) 
s_ashr_i32 s19, s0, 31 // (op 1205) 

s_xor_b32 s0, s18, s19 // (op 1205) 

s_sub_i32 s18, s0, s19 // (op 1205) 


s_mul_i32 s0, s1, s44 // (op 1205) 

s_sub_u32 s1, s16, s0 // (op 1205) 

s_mul_i32 s0, s18, s44 // (op 1205) 
s_add_u32 s0, s0, s1 // (op 1205) 


s_mul_hi_i32 s1, s16, s10 // (op 1205) 
s_add_i32 s18, s1, s16 // (op 1205) 

s_ashr_i32 s1, s18, 31 // (op 1205) 
s_cmp_eq_i32 s10, 0 // (op 1205) 
s_mov_b32 s19, scc // (op 1205) 
s_cmp_eq_i32 s19, 0 // (op 1205) 
s_cselect_b32 s20, 0, -1 // (op 1205) 

s_and_b32 s19, s47, 31 // (op 1205) 
s_lshl_b32 s21, 1, s19 // (op 1205) 
s_add_i32 s22, s21, s20 // (op 1205) 


s_and_b32 s23, s1, s22 // (op 1205) 


s_add_i32 s1, s18, s23 // (op 1205) 


s_ashr_i32 s18, s1, s19 // (op 1205) 


s_lshl_b32 s1, s47, 24 // (op 1205) 
s_ashr_i32 s19, s1, 31 // (op 1205) 

s_xor_b32 s1, s18, s19 // (op 1205) 

s_sub_i32 s18, s1, s19 // (op 1205) 


s_add_u32 s22, s6, 64 // (op 1205) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1205) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 1205) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 1205) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 1205) 

s_mul_i32 s1, s44, s22 // (op 1205) 

s_mul_i32 s19, s48, s1 // (op 1205) 

s_mul_i32 s1, s18, s19 // (op 1205) 

s_sub_u32 s19, s16, s1 // (op 1205) 

s_mov_b32 s1, s19 // (op 1205) 
s_mul_hi_i32 s20, s1, s15 // (op 1205) 
s_add_i32 s21, s20, s1 // (op 1205) 


s_ashr_i32 s1, s21, 31 // (op 1205) 
s_cmp_eq_i32 s15, 0 // (op 1205) 
s_mov_b32 s20, scc // (op 1205) 
s_cmp_eq_i32 s20, 0 // (op 1205) 
s_cselect_b32 s22, 0, -1 // (op 1205) 

s_and_b32 s20, s46, 31 // (op 1205) 
s_lshl_b32 s23, 1, s20 // (op 1205) 
s_add_i32 s49, s23, s22 // (op 1205) 


s_and_b32 s22, s1, s49 // (op 1205) 


s_add_i32 s1, s21, s22 // (op 1205) 


s_ashr_i32 s21, s1, s20 // (op 1205) 


s_lshl_b32 s1, s46, 24 // (op 1205) 
s_ashr_i32 s20, s1, 31 // (op 1205) 

s_xor_b32 s1, s21, s20 // (op 1205) 

s_sub_i32 s21, s1, s20 // (op 1205) 


s_mul_i32 s1, s45, s44 // (op 1205) 
s_add_u32 s22, s8, 16 // (op 1205) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 1205) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 1205) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 1205) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 1205) 

s_sub_i32 s20, s22, s1 // (op 1205) 

s_mul_i32 s22, s21, s20 // (op 1205) 


s_sub_u32 s23, s19, s22 // (op 1205) 


s_add_u32 s19, s23, s1 // (op 1205) 


s_cmp_eq_i32 s18, 0 // (op 1205) 
s_mov_b32 s1, scc // (op 1205) 

s_cmp_eq_i32 s1, 0 // (op 1205) 
s_cselect_b32 s18, s19, s0 // (op 1205) 



v_lshl_add_u32 v15, s18, 4, v14 // (op 1205) 


v_mul_lo_u32 v19, v15, s25 // (op 1205) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 1205) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 1205) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 1205) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1205) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1205) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 1205) most significant: low of high * low
v_mul_hi_u32 v16, v18, 6 // (op 1205) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 1205) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1205) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1205) 

v_mov_b32 v14, v16 // (op 1205) convert

s_mul_i32 s1, s9, 6 // (op 1207) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1207) least significant
s_mul_hi_u32 s18, s8, 6 // (op 1207) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1207) most significant: sum

s_mul_i32 s19, s24, s1 // (op 1207) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 1207) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 1207) least significant
s_mul_hi_u32 s21, s24, s0 // (op 1207) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 1207) most significant: sum
s_add_u32 s19, s19, s21 // (op 1207) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1207) 

s_mov_b32 s20, s26 // (op 1207) reg expression
s_mov_b32 s21, s27 // (op 1207) reg expression
s_mov_b32 s22, s0 // (op 1207) reg expression
s_mov_b32 s23, 131072 // (op 1207) call()

s_mul_hi_i32 s0, s16, s11 // (op 1224) 
v_lshrrev_b32 v15, 3, v1 // (op 1224) 
s_add_i32 s1, s0, s16 // (op 1224) 

v_and_b32 v16, -8, v1 // (op 1224) 
s_ashr_i32 s0, s1, 31 // (op 1224) 
v_sub_u32 v17, v1, v16 // (op 1224) 

s_cmp_eq_i32 s11, 0 // (op 1224) 
s_mov_b32 s18, scc // (op 1224) 
v_lshlrev_b32 v16, 5, v17 // (op 1224) 

s_cmp_eq_i32 s18, 0 // (op 1224) 
s_cselect_b32 s19, 0, -1 // (op 1224) 

v_mov_b32 v19, 0 // (op 1224) convert
v_mov_b32 v18, v16 // (op 1224) convert

s_and_b32 s18, s12, 31 // (op 1224) 
s_lshl_b32 s49, 1, s18 // (op 1224) 
s_add_i32 s50, s49, s19 // (op 1224) 


s_and_b32 s19, s0, s50 // (op 1224) 


s_add_i32 s0, s1, s19 // (op 1224) 


s_ashr_i32 s1, s0, s18 // (op 1224) 


s_lshl_b32 s0, s12, 24 // (op 1224) 
s_ashr_i32 s18, s0, 31 // (op 1224) 

s_xor_b32 s0, s1, s18 // (op 1224) 

s_sub_i32 s1, s0, s18 // (op 1224) 


s_mul_hi_i32 s0, s1, s13 // (op 1224) 
s_add_i32 s18, s0, s1 // (op 1224) 

s_ashr_i32 s0, s18, 31 // (op 1224) 
s_cmp_eq_i32 s13, 0 // (op 1224) 
s_mov_b32 s19, scc // (op 1224) 
s_cmp_eq_i32 s19, 0 // (op 1224) 
s_cselect_b32 s49, 0, -1 // (op 1224) 

s_and_b32 s19, s14, 31 // (op 1224) 
s_lshl_b32 s50, 1, s19 // (op 1224) 
s_add_i32 s51, s50, s49 // (op 1224) 


s_and_b32 s49, s0, s51 // (op 1224) 


s_add_i32 s0, s18, s49 // (op 1224) 


s_ashr_i32 s18, s0, s19 // (op 1224) 


s_lshl_b32 s0, s14, 24 // (op 1224) 
s_ashr_i32 s19, s0, 31 // (op 1224) 

s_xor_b32 s0, s18, s19 // (op 1224) 

s_sub_i32 s18, s0, s19 // (op 1224) 


s_mul_i32 s0, s1, s44 // (op 1224) 

s_sub_u32 s1, s16, s0 // (op 1224) 

s_mul_i32 s0, s18, s44 // (op 1224) 
s_add_u32 s0, s0, s1 // (op 1224) 


s_mul_hi_i32 s1, s16, s10 // (op 1224) 
s_add_i32 s18, s1, s16 // (op 1224) 

s_ashr_i32 s1, s18, 31 // (op 1224) 
s_cmp_eq_i32 s10, 0 // (op 1224) 
s_mov_b32 s19, scc // (op 1224) 
s_cmp_eq_i32 s19, 0 // (op 1224) 
s_cselect_b32 s49, 0, -1 // (op 1224) 

s_and_b32 s19, s47, 31 // (op 1224) 
s_lshl_b32 s50, 1, s19 // (op 1224) 
s_add_i32 s51, s50, s49 // (op 1224) 


s_and_b32 s49, s1, s51 // (op 1224) 


s_add_i32 s1, s18, s49 // (op 1224) 


s_ashr_i32 s18, s1, s19 // (op 1224) 


s_lshl_b32 s1, s47, 24 // (op 1224) 
s_ashr_i32 s19, s1, 31 // (op 1224) 

s_xor_b32 s1, s18, s19 // (op 1224) 

s_sub_i32 s18, s1, s19 // (op 1224) 


s_add_u32 s54, s6, 64 // (op 1224) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 1224) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1224) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1224) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 1224) 

s_mul_i32 s1, s44, s50 // (op 1224) 

s_mul_i32 s19, s48, s1 // (op 1224) 

s_mul_i32 s1, s18, s19 // (op 1224) 

s_sub_u32 s19, s16, s1 // (op 1224) 

s_mov_b32 s1, s19 // (op 1224) 
s_mul_hi_i32 s49, s1, s15 // (op 1224) 
s_add_i32 s50, s49, s1 // (op 1224) 


s_ashr_i32 s1, s50, 31 // (op 1224) 
s_cmp_eq_i32 s15, 0 // (op 1224) 
s_mov_b32 s49, scc // (op 1224) 
s_cmp_eq_i32 s49, 0 // (op 1224) 
s_cselect_b32 s51, 0, -1 // (op 1224) 

s_and_b32 s49, s46, 31 // (op 1224) 
s_lshl_b32 s52, 1, s49 // (op 1224) 
s_add_i32 s53, s52, s51 // (op 1224) 


s_and_b32 s51, s1, s53 // (op 1224) 


s_add_i32 s1, s50, s51 // (op 1224) 


s_ashr_i32 s50, s1, s49 // (op 1224) 


s_lshl_b32 s1, s46, 24 // (op 1224) 
s_ashr_i32 s49, s1, 31 // (op 1224) 

s_xor_b32 s1, s50, s49 // (op 1224) 

s_sub_i32 s50, s1, s49 // (op 1224) 


s_mul_i32 s1, s45, s44 // (op 1224) 
s_add_u32 s54, s8, 16 // (op 1224) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 1224) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1224) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1224) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 1224) 

s_sub_i32 s49, s54, s1 // (op 1224) 

s_mul_i32 s51, s50, s49 // (op 1224) 


s_sub_u32 s49, s19, s51 // (op 1224) 


s_add_u32 s19, s49, s1 // (op 1224) 


s_cmp_eq_i32 s18, 0 // (op 1224) 
s_mov_b32 s1, scc // (op 1224) 

s_cmp_eq_i32 s1, 0 // (op 1224) 
s_cselect_b32 s18, s19, s0 // (op 1224) 



v_lshl_add_u32 v16, s18, 4, v15 // (op 1224) 


v_mul_lo_u32 v21, v16, s25 // (op 1224) most significant: low of low * high
v_mul_hi_u32 v15, v16, s24 // (op 1224) most significant: high of low * low
v_mul_lo_u32 v20, v16, s24 // (op 1224) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1224) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 1224) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 1224) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 1224) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 1224) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 1224) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1224) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 1224) 

v_mov_b32 v15, v16 // (op 1224) convert

s_mul_i32 s1, s9, 6 // (op 1226) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1226) least significant
s_mul_hi_u32 s18, s8, 6 // (op 1226) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1226) most significant: sum

s_mul_i32 s19, s24, s1 // (op 1226) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 1226) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 1226) least significant
s_mul_hi_u32 s50, s24, s0 // (op 1226) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1226) most significant: sum
s_add_u32 s19, s19, s50 // (op 1226) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1226) 

s_mov_b32 s20, s26 // (op 1226) reg expression
s_mov_b32 s21, s27 // (op 1226) reg expression
s_mov_b32 s22, s0 // (op 1226) reg expression
s_mov_b32 s23, 131072 // (op 1226) call()

s_mul_hi_i32 s0, s16, s11 // (op 1269) 
v_lshlrev_b32 v16, 1, v1 // (op 1269) 
s_add_i32 s1, s0, s16 // (op 1269) 

s_ashr_i32 s0, s1, 31 // (op 1269) 
s_cmp_eq_i32 s11, 0 // (op 1269) 
s_mov_b32 s18, scc // (op 1269) 
s_cmp_eq_i32 s18, 0 // (op 1269) 
s_cselect_b32 s19, 0, -1 // (op 1269) 

s_and_b32 s18, s12, 31 // (op 1269) 
s_lshl_b32 s49, 1, s18 // (op 1269) 
s_add_i32 s50, s49, s19 // (op 1269) 


s_and_b32 s19, s0, s50 // (op 1269) 


s_add_i32 s0, s1, s19 // (op 1269) 


s_ashr_i32 s1, s0, s18 // (op 1269) 


s_lshl_b32 s0, s12, 24 // (op 1269) 
s_ashr_i32 s18, s0, 31 // (op 1269) 

s_xor_b32 s0, s1, s18 // (op 1269) 

s_sub_i32 s1, s0, s18 // (op 1269) 


s_mul_hi_i32 s0, s1, s13 // (op 1269) 
s_add_i32 s18, s0, s1 // (op 1269) 

s_ashr_i32 s0, s18, 31 // (op 1269) 
s_cmp_eq_i32 s13, 0 // (op 1269) 
s_mov_b32 s19, scc // (op 1269) 
s_cmp_eq_i32 s19, 0 // (op 1269) 
s_cselect_b32 s49, 0, -1 // (op 1269) 

s_and_b32 s19, s14, 31 // (op 1269) 
s_lshl_b32 s50, 1, s19 // (op 1269) 
s_add_i32 s51, s50, s49 // (op 1269) 


s_and_b32 s49, s0, s51 // (op 1269) 


s_add_i32 s0, s18, s49 // (op 1269) 


s_ashr_i32 s18, s0, s19 // (op 1269) 


s_lshl_b32 s0, s14, 24 // (op 1269) 
s_ashr_i32 s19, s0, 31 // (op 1269) 

s_xor_b32 s0, s18, s19 // (op 1269) 

s_sub_i32 s18, s0, s19 // (op 1269) 


s_add_u32 s54, s6, 64 // (op 1269) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 1269) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1269) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1269) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 1269) 

s_mul_i32 s0, s18, s50 // (op 1269) 

s_sub_i32 s49, s1, s0 // (op 1269) 


s_mul_hi_i32 s0, s16, s10 // (op 1269) 
s_add_i32 s1, s0, s16 // (op 1269) 

s_ashr_i32 s0, s1, 31 // (op 1269) 
s_cmp_eq_i32 s10, 0 // (op 1269) 
s_mov_b32 s18, scc // (op 1269) 
s_cmp_eq_i32 s18, 0 // (op 1269) 
s_cselect_b32 s19, 0, -1 // (op 1269) 

s_and_b32 s18, s47, 31 // (op 1269) 
s_lshl_b32 s52, 1, s18 // (op 1269) 
s_add_i32 s53, s52, s19 // (op 1269) 


s_and_b32 s19, s0, s53 // (op 1269) 


s_add_i32 s0, s1, s19 // (op 1269) 


s_ashr_i32 s1, s0, s18 // (op 1269) 


s_lshl_b32 s0, s47, 24 // (op 1269) 
s_ashr_i32 s18, s0, 31 // (op 1269) 

s_xor_b32 s0, s1, s18 // (op 1269) 

s_sub_i32 s1, s0, s18 // (op 1269) 


s_mul_i32 s0, s44, s50 // (op 1269) 

s_mul_i32 s18, s48, s0 // (op 1269) 

s_mul_i32 s0, s1, s18 // (op 1269) 

s_sub_u32 s18, s16, s0 // (op 1269) 

s_mul_hi_i32 s0, s18, s15 // (op 1269) 
s_add_i32 s19, s0, s18 // (op 1269) 


s_ashr_i32 s0, s19, 31 // (op 1269) 
s_cmp_eq_i32 s15, 0 // (op 1269) 
s_mov_b32 s18, scc // (op 1269) 
s_cmp_eq_i32 s18, 0 // (op 1269) 
s_cselect_b32 s50, 0, -1 // (op 1269) 

s_and_b32 s18, s46, 31 // (op 1269) 
s_lshl_b32 s51, 1, s18 // (op 1269) 
s_add_i32 s52, s51, s50 // (op 1269) 


s_and_b32 s50, s0, s52 // (op 1269) 


s_add_i32 s0, s19, s50 // (op 1269) 


s_ashr_i32 s19, s0, s18 // (op 1269) 


s_lshl_b32 s0, s46, 24 // (op 1269) 
s_ashr_i32 s18, s0, 31 // (op 1269) 

s_xor_b32 s0, s19, s18 // (op 1269) 

s_sub_i32 s19, s0, s18 // (op 1269) 


s_cmp_eq_i32 s1, 0 // (op 1269) 
s_mov_b32 s0, scc // (op 1269) 

s_cmp_eq_i32 s0, 0 // (op 1269) 
s_cselect_b32 s1, s19, s49 // (op 1269) 



s_lshl_b32 s0, s1, 6 // (op 1269) 

s_mov_b32 s19, 0 // (op 1269) convert
s_mov_b32 s18, s0 // (op 1269) convert

v_mul_lo_u32 v19, v16, s29 // (op 1269) most significant: low of low * high
v_mul_hi_u32 v17, v16, s28 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v18, v16, s28 // (op 1269) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 1269) most significant: sum

v_mov_b32 v17, s18 // (op 1269) 
v_mov_b32 v20, s19 // (op 1269) 
v_add_co_u32 v18, s[0:1], v18, v17 // (op 1269) least significant half
v_addc_co_u32 v19, s[0:1], v19, v20, s[0:1] // (op 1269) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 1269) most significant: low of high * low
v_mul_hi_u32 v20, v18, 6 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 1269) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 1269) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 1269) 

v_mov_b32 v16, v18 // (op 1269) convert

s_mul_i32 s1, s33, 6 // (op 1271) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1271) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1271) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1271) most significant: sum

s_mul_i32 s19, s28, s1 // (op 1271) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 1271) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 1271) least significant
s_mul_hi_u32 s50, s28, s0 // (op 1271) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1271) most significant: sum
s_add_u32 s19, s19, s50 // (op 1271) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1271) 

s_mov_b32 s52, s30 // (op 1271) reg expression
s_mov_b32 s53, s31 // (op 1271) reg expression
s_mov_b32 s54, s0 // (op 1271) reg expression
s_mov_b32 s55, 131072 // (op 1271) call()

s_mul_hi_i32 s0, s16, s11 // (op 1288) 
v_lshlrev_b32 v17, 1, v1 // (op 1288) 
s_add_i32 s1, s0, s16 // (op 1288) 

s_ashr_i32 s0, s1, 31 // (op 1288) 
s_cmp_eq_i32 s11, 0 // (op 1288) 
s_mov_b32 s18, scc // (op 1288) 
s_cmp_eq_i32 s18, 0 // (op 1288) 
s_cselect_b32 s19, 0, -1 // (op 1288) 

s_and_b32 s18, s12, 31 // (op 1288) 
s_lshl_b32 s49, 1, s18 // (op 1288) 
s_add_i32 s50, s49, s19 // (op 1288) 


s_and_b32 s19, s0, s50 // (op 1288) 


s_add_i32 s0, s1, s19 // (op 1288) 


s_ashr_i32 s1, s0, s18 // (op 1288) 


s_lshl_b32 s0, s12, 24 // (op 1288) 
s_ashr_i32 s18, s0, 31 // (op 1288) 

s_xor_b32 s0, s1, s18 // (op 1288) 

s_sub_i32 s1, s0, s18 // (op 1288) 


s_mul_hi_i32 s0, s1, s13 // (op 1288) 
s_add_i32 s18, s0, s1 // (op 1288) 

s_ashr_i32 s0, s18, 31 // (op 1288) 
s_cmp_eq_i32 s13, 0 // (op 1288) 
s_mov_b32 s19, scc // (op 1288) 
s_cmp_eq_i32 s19, 0 // (op 1288) 
s_cselect_b32 s49, 0, -1 // (op 1288) 

s_and_b32 s19, s14, 31 // (op 1288) 
s_lshl_b32 s50, 1, s19 // (op 1288) 
s_add_i32 s51, s50, s49 // (op 1288) 


s_and_b32 s49, s0, s51 // (op 1288) 


s_add_i32 s0, s18, s49 // (op 1288) 


s_ashr_i32 s18, s0, s19 // (op 1288) 


s_lshl_b32 s0, s14, 24 // (op 1288) 
s_ashr_i32 s19, s0, 31 // (op 1288) 

s_xor_b32 s0, s18, s19 // (op 1288) 

s_sub_i32 s18, s0, s19 // (op 1288) 


s_add_u32 s50, s6, 64 // (op 1288) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1288) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1288) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1288) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1288) 

s_mul_i32 s0, s18, s50 // (op 1288) 

s_sub_i32 s49, s1, s0 // (op 1288) 


s_mul_hi_i32 s0, s16, s10 // (op 1288) 
s_add_i32 s1, s0, s16 // (op 1288) 

s_ashr_i32 s0, s1, 31 // (op 1288) 
s_cmp_eq_i32 s10, 0 // (op 1288) 
s_mov_b32 s18, scc // (op 1288) 
s_cmp_eq_i32 s18, 0 // (op 1288) 
s_cselect_b32 s19, 0, -1 // (op 1288) 

s_and_b32 s18, s47, 31 // (op 1288) 
s_lshl_b32 s64, 1, s18 // (op 1288) 
s_add_i32 s65, s64, s19 // (op 1288) 


s_and_b32 s19, s0, s65 // (op 1288) 


s_add_i32 s0, s1, s19 // (op 1288) 


s_ashr_i32 s1, s0, s18 // (op 1288) 


s_lshl_b32 s0, s47, 24 // (op 1288) 
s_ashr_i32 s18, s0, 31 // (op 1288) 

s_xor_b32 s0, s1, s18 // (op 1288) 

s_sub_i32 s1, s0, s18 // (op 1288) 


s_mul_i32 s0, s44, s50 // (op 1288) 

s_mul_i32 s18, s48, s0 // (op 1288) 

s_mul_i32 s0, s1, s18 // (op 1288) 

s_sub_u32 s18, s16, s0 // (op 1288) 

s_mul_hi_i32 s0, s18, s15 // (op 1288) 
s_add_i32 s19, s0, s18 // (op 1288) 


s_ashr_i32 s0, s19, 31 // (op 1288) 
s_cmp_eq_i32 s15, 0 // (op 1288) 
s_mov_b32 s18, scc // (op 1288) 
s_cmp_eq_i32 s18, 0 // (op 1288) 
s_cselect_b32 s50, 0, -1 // (op 1288) 

s_and_b32 s18, s46, 31 // (op 1288) 
s_lshl_b32 s51, 1, s18 // (op 1288) 
s_add_i32 s64, s51, s50 // (op 1288) 


s_and_b32 s50, s0, s64 // (op 1288) 


s_add_i32 s0, s19, s50 // (op 1288) 


s_ashr_i32 s19, s0, s18 // (op 1288) 


s_lshl_b32 s0, s46, 24 // (op 1288) 
s_ashr_i32 s18, s0, 31 // (op 1288) 

s_xor_b32 s0, s19, s18 // (op 1288) 

s_sub_i32 s19, s0, s18 // (op 1288) 


s_cmp_eq_i32 s1, 0 // (op 1288) 
s_mov_b32 s0, scc // (op 1288) 

s_cmp_eq_i32 s0, 0 // (op 1288) 
s_cselect_b32 s1, s19, s49 // (op 1288) 



s_lshl_b32 s0, s1, 6 // (op 1288) 

s_mov_b32 s19, 0 // (op 1288) convert
s_mov_b32 s18, s0 // (op 1288) convert

v_mul_lo_u32 v19, v17, s29 // (op 1288) most significant: low of low * high
v_mul_hi_u32 v20, v17, s28 // (op 1288) most significant: high of low * low
v_mul_lo_u32 v18, v17, s28 // (op 1288) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1288) most significant: sum

v_mov_b32 v20, s18 // (op 1288) 
v_mov_b32 v21, s19 // (op 1288) 
v_add_co_u32 v18, s[0:1], v18, v20 // (op 1288) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 1288) most significant half





v_mul_lo_u32 v21, v19, 6 // (op 1288) most significant: low of high * low
v_mul_hi_u32 v17, v18, 6 // (op 1288) most significant: high of low * low
v_mul_lo_u32 v20, v18, 6 // (op 1288) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 1288) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 1288) 

v_mov_b32 v17, v18 // (op 1288) convert

s_mul_i32 s1, s33, 6 // (op 1290) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1290) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1290) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1290) most significant: sum

s_mul_i32 s19, s28, s1 // (op 1290) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 1290) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 1290) least significant
s_mul_hi_u32 s50, s28, s0 // (op 1290) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1290) most significant: sum
s_add_u32 s19, s19, s50 // (op 1290) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1290) 

s_mov_b32 s52, s30 // (op 1290) reg expression
s_mov_b32 s53, s31 // (op 1290) reg expression
s_mov_b32 s54, s0 // (op 1290) reg expression
s_mov_b32 s55, 131072 // (op 1290) call()

s_mul_hi_i32 s0, s16, s11 // (op 1333) 
v_lshrrev_b32 v18, 3, v1 // (op 1333) 
s_add_i32 s1, s0, s16 // (op 1333) 

v_and_b32 v19, -8, v1 // (op 1333) 
s_ashr_i32 s0, s1, 31 // (op 1333) 
v_sub_u32 v20, v1, v19 // (op 1333) 

s_cmp_eq_i32 s11, 0 // (op 1333) 
s_mov_b32 s18, scc // (op 1333) 
s_cmp_eq_i32 s18, 0 // (op 1333) 
s_cselect_b32 s19, 0, -1 // (op 1333) 

s_and_b32 s18, s12, 31 // (op 1333) 
s_lshl_b32 s49, 1, s18 // (op 1333) 
s_add_i32 s50, s49, s19 // (op 1333) 


s_and_b32 s19, s0, s50 // (op 1333) 


s_add_i32 s0, s1, s19 // (op 1333) 


s_ashr_i32 s1, s0, s18 // (op 1333) 


s_lshl_b32 s0, s12, 24 // (op 1333) 
s_ashr_i32 s18, s0, 31 // (op 1333) 

s_xor_b32 s0, s1, s18 // (op 1333) 

s_sub_i32 s1, s0, s18 // (op 1333) 


s_mul_hi_i32 s0, s1, s13 // (op 1333) 
s_add_i32 s18, s0, s1 // (op 1333) 

s_ashr_i32 s0, s18, 31 // (op 1333) 
s_cmp_eq_i32 s13, 0 // (op 1333) 
s_mov_b32 s19, scc // (op 1333) 
s_cmp_eq_i32 s19, 0 // (op 1333) 
s_cselect_b32 s49, 0, -1 // (op 1333) 

s_and_b32 s19, s14, 31 // (op 1333) 
s_lshl_b32 s50, 1, s19 // (op 1333) 
s_add_i32 s51, s50, s49 // (op 1333) 


s_and_b32 s49, s0, s51 // (op 1333) 


s_add_i32 s0, s18, s49 // (op 1333) 


s_ashr_i32 s18, s0, s19 // (op 1333) 


s_lshl_b32 s0, s14, 24 // (op 1333) 
s_ashr_i32 s19, s0, 31 // (op 1333) 

s_xor_b32 s0, s18, s19 // (op 1333) 

s_sub_i32 s18, s0, s19 // (op 1333) 


s_mul_i32 s0, s1, s44 // (op 1333) 

s_sub_u32 s1, s16, s0 // (op 1333) 

s_mul_i32 s0, s18, s44 // (op 1333) 
s_add_u32 s0, s0, s1 // (op 1333) 


s_mul_hi_i32 s1, s16, s10 // (op 1333) 
s_add_i32 s18, s1, s16 // (op 1333) 

s_ashr_i32 s1, s18, 31 // (op 1333) 
s_cmp_eq_i32 s10, 0 // (op 1333) 
s_mov_b32 s19, scc // (op 1333) 
s_cmp_eq_i32 s19, 0 // (op 1333) 
s_cselect_b32 s49, 0, -1 // (op 1333) 

s_and_b32 s19, s47, 31 // (op 1333) 
s_lshl_b32 s50, 1, s19 // (op 1333) 
s_add_i32 s51, s50, s49 // (op 1333) 


s_and_b32 s49, s1, s51 // (op 1333) 


s_add_i32 s1, s18, s49 // (op 1333) 


s_ashr_i32 s18, s1, s19 // (op 1333) 


s_lshl_b32 s1, s47, 24 // (op 1333) 
s_ashr_i32 s19, s1, 31 // (op 1333) 

s_xor_b32 s1, s18, s19 // (op 1333) 

s_sub_i32 s18, s1, s19 // (op 1333) 


s_add_u32 s50, s6, 64 // (op 1333) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1333) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1333) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1333) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1333) 

s_mul_i32 s1, s44, s50 // (op 1333) 

s_mul_i32 s19, s48, s1 // (op 1333) 

s_mul_i32 s1, s18, s19 // (op 1333) 

s_sub_u32 s19, s16, s1 // (op 1333) 

s_mov_b32 s1, s19 // (op 1333) 
s_mul_hi_i32 s49, s1, s15 // (op 1333) 
s_add_i32 s50, s49, s1 // (op 1333) 


s_ashr_i32 s1, s50, 31 // (op 1333) 
s_cmp_eq_i32 s15, 0 // (op 1333) 
s_mov_b32 s49, scc // (op 1333) 
s_cmp_eq_i32 s49, 0 // (op 1333) 
s_cselect_b32 s51, 0, -1 // (op 1333) 

s_and_b32 s49, s46, 31 // (op 1333) 
s_lshl_b32 s64, 1, s49 // (op 1333) 
s_add_i32 s65, s64, s51 // (op 1333) 


s_and_b32 s51, s1, s65 // (op 1333) 


s_add_i32 s1, s50, s51 // (op 1333) 


s_ashr_i32 s50, s1, s49 // (op 1333) 


s_lshl_b32 s1, s46, 24 // (op 1333) 
s_ashr_i32 s49, s1, 31 // (op 1333) 

s_xor_b32 s1, s50, s49 // (op 1333) 

s_sub_i32 s50, s1, s49 // (op 1333) 


s_mul_i32 s1, s45, s44 // (op 1333) 
s_add_u32 s64, s8, 16 // (op 1333) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 1333) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1333) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1333) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 1333) 

s_sub_i32 s49, s64, s1 // (op 1333) 

s_mul_i32 s51, s50, s49 // (op 1333) 


s_sub_u32 s49, s19, s51 // (op 1333) 


s_add_u32 s19, s49, s1 // (op 1333) 


s_cmp_eq_i32 s18, 0 // (op 1333) 
s_mov_b32 s1, scc // (op 1333) 

s_cmp_eq_i32 s1, 0 // (op 1333) 
s_cselect_b32 s18, s19, s0 // (op 1333) 



v_lshl_add_u32 v19, s18, 4, v18 // (op 1333) 


v_mul_lo_u32 v18, v19, s34 // (op 1333) 
v_add_u32 v18, v18, v20 // (op 1333) 


s_mul_i32 s1, s34, s9 // (op 1335) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1335) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1335) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1335) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1335) most significant: sum
s_add_u32 s1, s1, s19 // (op 1335) most significant: sum


s_mov_b32 s64, s36 // (op 1335) reg expression
s_mov_b32 s65, s37 // (op 1335) reg expression
s_mov_b32 s66, s0 // (op 1335) reg expression
s_mov_b32 s67, 131072 // (op 1335) call()

s_mul_hi_i32 s0, s16, s11 // (op 1352) 
v_lshrrev_b32 v19, 3, v1 // (op 1352) 
s_add_i32 s1, s0, s16 // (op 1352) 

v_and_b32 v20, -8, v1 // (op 1352) 
s_ashr_i32 s0, s1, 31 // (op 1352) 
v_sub_u32 v21, v1, v20 // (op 1352) 

s_cmp_eq_i32 s11, 0 // (op 1352) 
s_mov_b32 s18, scc // (op 1352) 
s_cmp_eq_i32 s18, 0 // (op 1352) 
s_cselect_b32 s19, 0, -1 // (op 1352) 

s_and_b32 s18, s12, 31 // (op 1352) 
s_lshl_b32 s49, 1, s18 // (op 1352) 
s_add_i32 s50, s49, s19 // (op 1352) 


s_and_b32 s19, s0, s50 // (op 1352) 


s_add_i32 s0, s1, s19 // (op 1352) 


s_ashr_i32 s1, s0, s18 // (op 1352) 


s_lshl_b32 s0, s12, 24 // (op 1352) 
s_ashr_i32 s18, s0, 31 // (op 1352) 

s_xor_b32 s0, s1, s18 // (op 1352) 

s_sub_i32 s1, s0, s18 // (op 1352) 


s_mul_hi_i32 s0, s1, s13 // (op 1352) 
s_add_i32 s18, s0, s1 // (op 1352) 

s_ashr_i32 s0, s18, 31 // (op 1352) 
s_cmp_eq_i32 s13, 0 // (op 1352) 
s_mov_b32 s19, scc // (op 1352) 
s_cmp_eq_i32 s19, 0 // (op 1352) 
s_cselect_b32 s49, 0, -1 // (op 1352) 

s_and_b32 s19, s14, 31 // (op 1352) 
s_lshl_b32 s50, 1, s19 // (op 1352) 
s_add_i32 s51, s50, s49 // (op 1352) 


s_and_b32 s49, s0, s51 // (op 1352) 


s_add_i32 s0, s18, s49 // (op 1352) 


s_ashr_i32 s18, s0, s19 // (op 1352) 


s_lshl_b32 s0, s14, 24 // (op 1352) 
s_ashr_i32 s19, s0, 31 // (op 1352) 

s_xor_b32 s0, s18, s19 // (op 1352) 

s_sub_i32 s18, s0, s19 // (op 1352) 


s_mul_i32 s0, s1, s44 // (op 1352) 

s_sub_u32 s1, s16, s0 // (op 1352) 

s_mul_i32 s0, s18, s44 // (op 1352) 
s_add_u32 s0, s0, s1 // (op 1352) 


s_mul_hi_i32 s1, s16, s10 // (op 1352) 
s_add_i32 s18, s1, s16 // (op 1352) 

s_ashr_i32 s1, s18, 31 // (op 1352) 
s_cmp_eq_i32 s10, 0 // (op 1352) 
s_mov_b32 s19, scc // (op 1352) 
s_cmp_eq_i32 s19, 0 // (op 1352) 
s_cselect_b32 s49, 0, -1 // (op 1352) 

s_and_b32 s19, s47, 31 // (op 1352) 
s_lshl_b32 s50, 1, s19 // (op 1352) 
s_add_i32 s51, s50, s49 // (op 1352) 


s_and_b32 s49, s1, s51 // (op 1352) 


s_add_i32 s1, s18, s49 // (op 1352) 


s_ashr_i32 s18, s1, s19 // (op 1352) 


s_lshl_b32 s1, s47, 24 // (op 1352) 
s_ashr_i32 s19, s1, 31 // (op 1352) 

s_xor_b32 s1, s18, s19 // (op 1352) 

s_sub_i32 s18, s1, s19 // (op 1352) 


s_add_u32 s50, s6, 64 // (op 1352) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1352) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1352) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1352) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1352) 

s_mul_i32 s1, s44, s50 // (op 1352) 

s_mul_i32 s19, s48, s1 // (op 1352) 

s_mul_i32 s1, s18, s19 // (op 1352) 

s_sub_u32 s19, s16, s1 // (op 1352) 

s_mov_b32 s1, s19 // (op 1352) 
s_mul_hi_i32 s49, s1, s15 // (op 1352) 
s_add_i32 s50, s49, s1 // (op 1352) 


s_ashr_i32 s1, s50, 31 // (op 1352) 
s_cmp_eq_i32 s15, 0 // (op 1352) 
s_mov_b32 s49, scc // (op 1352) 
s_cmp_eq_i32 s49, 0 // (op 1352) 
s_cselect_b32 s51, 0, -1 // (op 1352) 

s_and_b32 s49, s46, 31 // (op 1352) 
s_lshl_b32 s68, 1, s49 // (op 1352) 
s_add_i32 s69, s68, s51 // (op 1352) 


s_and_b32 s51, s1, s69 // (op 1352) 


s_add_i32 s1, s50, s51 // (op 1352) 


s_ashr_i32 s50, s1, s49 // (op 1352) 


s_lshl_b32 s1, s46, 24 // (op 1352) 
s_ashr_i32 s49, s1, 31 // (op 1352) 

s_xor_b32 s1, s50, s49 // (op 1352) 

s_sub_i32 s50, s1, s49 // (op 1352) 


s_mul_i32 s1, s45, s44 // (op 1352) 
s_add_u32 s68, s8, 16 // (op 1352) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 1352) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1352) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1352) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 1352) 

s_sub_i32 s49, s68, s1 // (op 1352) 

s_mul_i32 s51, s50, s49 // (op 1352) 


s_sub_u32 s49, s19, s51 // (op 1352) 


s_add_u32 s19, s49, s1 // (op 1352) 


s_cmp_eq_i32 s18, 0 // (op 1352) 
s_mov_b32 s1, scc // (op 1352) 

s_cmp_eq_i32 s1, 0 // (op 1352) 
s_cselect_b32 s18, s19, s0 // (op 1352) 



v_lshl_add_u32 v20, s18, 4, v19 // (op 1352) 


v_mul_lo_u32 v19, v20, s34 // (op 1352) 
v_add_u32 v19, v19, v21 // (op 1352) 


s_mul_i32 s1, s34, s9 // (op 1354) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1354) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1354) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1354) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1354) most significant: sum
s_add_u32 s1, s1, s19 // (op 1354) most significant: sum


s_mov_b32 s64, s36 // (op 1354) reg expression
s_mov_b32 s65, s37 // (op 1354) reg expression
s_mov_b32 s66, s0 // (op 1354) reg expression
s_mov_b32 s67, 131072 // (op 1354) call()

s_mul_hi_i32 s0, s16, s11 // (op 1397) 
v_and_b32 v20, -16, v1 // (op 1397) 
s_add_i32 s1, s0, s16 // (op 1397) 

v_sub_u32 v21, v1, v20 // (op 1397) 

s_ashr_i32 s0, s1, 31 // (op 1397) 
v_lshlrev_b32 v20, 2, v21 // (op 1397) 

s_cmp_eq_i32 s11, 0 // (op 1397) 
s_mov_b32 s18, scc // (op 1397) 
v_lshrrev_b32 v21, 4, v1 // (op 1397) 
s_cmp_eq_i32 s18, 0 // (op 1397) 
s_cselect_b32 s19, 0, -1 // (op 1397) 

s_and_b32 s18, s12, 31 // (op 1397) 
s_lshl_b32 s49, 1, s18 // (op 1397) 
s_add_i32 s50, s49, s19 // (op 1397) 


s_and_b32 s19, s0, s50 // (op 1397) 


s_add_i32 s0, s1, s19 // (op 1397) 


s_ashr_i32 s1, s0, s18 // (op 1397) 


s_lshl_b32 s0, s12, 24 // (op 1397) 
s_ashr_i32 s18, s0, 31 // (op 1397) 

s_xor_b32 s0, s1, s18 // (op 1397) 

s_sub_i32 s1, s0, s18 // (op 1397) 


s_mul_hi_i32 s0, s1, s13 // (op 1397) 
s_add_i32 s18, s0, s1 // (op 1397) 

s_ashr_i32 s0, s18, 31 // (op 1397) 
s_cmp_eq_i32 s13, 0 // (op 1397) 
s_mov_b32 s19, scc // (op 1397) 
s_cmp_eq_i32 s19, 0 // (op 1397) 
s_cselect_b32 s49, 0, -1 // (op 1397) 

s_and_b32 s19, s14, 31 // (op 1397) 
s_lshl_b32 s50, 1, s19 // (op 1397) 
s_add_i32 s51, s50, s49 // (op 1397) 


s_and_b32 s49, s0, s51 // (op 1397) 


s_add_i32 s0, s18, s49 // (op 1397) 


s_ashr_i32 s18, s0, s19 // (op 1397) 


s_lshl_b32 s0, s14, 24 // (op 1397) 
s_ashr_i32 s19, s0, 31 // (op 1397) 

s_xor_b32 s0, s18, s19 // (op 1397) 

s_sub_i32 s18, s0, s19 // (op 1397) 


s_add_u32 s50, s6, 64 // (op 1397) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1397) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1397) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1397) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1397) 

s_mul_i32 s0, s18, s50 // (op 1397) 

s_sub_i32 s49, s1, s0 // (op 1397) 


s_mul_hi_i32 s0, s16, s10 // (op 1397) 
s_add_i32 s1, s0, s16 // (op 1397) 

s_ashr_i32 s0, s1, 31 // (op 1397) 
s_cmp_eq_i32 s10, 0 // (op 1397) 
s_mov_b32 s18, scc // (op 1397) 
s_cmp_eq_i32 s18, 0 // (op 1397) 
s_cselect_b32 s19, 0, -1 // (op 1397) 

s_and_b32 s18, s47, 31 // (op 1397) 
s_lshl_b32 s68, 1, s18 // (op 1397) 
s_add_i32 s69, s68, s19 // (op 1397) 


s_and_b32 s19, s0, s69 // (op 1397) 


s_add_i32 s0, s1, s19 // (op 1397) 


s_ashr_i32 s1, s0, s18 // (op 1397) 


s_lshl_b32 s0, s47, 24 // (op 1397) 
s_ashr_i32 s18, s0, 31 // (op 1397) 

s_xor_b32 s0, s1, s18 // (op 1397) 

s_sub_i32 s1, s0, s18 // (op 1397) 


s_mul_i32 s0, s44, s50 // (op 1397) 

s_mul_i32 s18, s48, s0 // (op 1397) 

s_mul_i32 s0, s1, s18 // (op 1397) 

s_sub_u32 s18, s16, s0 // (op 1397) 

s_mul_hi_i32 s0, s18, s15 // (op 1397) 
s_add_i32 s19, s0, s18 // (op 1397) 


s_ashr_i32 s0, s19, 31 // (op 1397) 
s_cmp_eq_i32 s15, 0 // (op 1397) 
s_mov_b32 s18, scc // (op 1397) 
s_cmp_eq_i32 s18, 0 // (op 1397) 
s_cselect_b32 s50, 0, -1 // (op 1397) 

s_and_b32 s18, s46, 31 // (op 1397) 
s_lshl_b32 s51, 1, s18 // (op 1397) 
s_add_i32 s68, s51, s50 // (op 1397) 


s_and_b32 s50, s0, s68 // (op 1397) 


s_add_i32 s0, s19, s50 // (op 1397) 


s_ashr_i32 s19, s0, s18 // (op 1397) 


s_lshl_b32 s0, s46, 24 // (op 1397) 
s_ashr_i32 s18, s0, 31 // (op 1397) 

s_xor_b32 s0, s19, s18 // (op 1397) 

s_sub_i32 s19, s0, s18 // (op 1397) 


s_cmp_eq_i32 s1, 0 // (op 1397) 
s_mov_b32 s0, scc // (op 1397) 

s_cmp_eq_i32 s0, 0 // (op 1397) 
s_cselect_b32 s1, s19, s49 // (op 1397) 



v_lshl_add_u32 v22, s1, 6, v20 // (op 1397) 


v_mul_lo_u32 v20, v21, s38 // (op 1397) 
v_add_u32 v20, v20, v22 // (op 1397) 


s_mul_i32 s1, s38, s43 // (op 1399) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1399) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1399) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1399) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1399) most significant: sum
s_add_u32 s1, s1, s19 // (op 1399) most significant: sum


s_mov_b32 s68, s40 // (op 1399) reg expression
s_mov_b32 s69, s41 // (op 1399) reg expression
s_mov_b32 s70, s0 // (op 1399) reg expression
s_mov_b32 s71, 131072 // (op 1399) call()

s_mul_hi_i32 s0, s16, s11 // (op 1416) 
v_and_b32 v21, -16, v1 // (op 1416) 
s_add_i32 s1, s0, s16 // (op 1416) 

v_sub_u32 v22, v1, v21 // (op 1416) 

s_ashr_i32 s0, s1, 31 // (op 1416) 
v_lshlrev_b32 v21, 2, v22 // (op 1416) 

s_cmp_eq_i32 s11, 0 // (op 1416) 
s_mov_b32 s18, scc // (op 1416) 
v_lshrrev_b32 v22, 4, v1 // (op 1416) 
s_cmp_eq_i32 s18, 0 // (op 1416) 
s_cselect_b32 s19, 0, -1 // (op 1416) 

s_and_b32 s18, s12, 31 // (op 1416) 
s_lshl_b32 s49, 1, s18 // (op 1416) 
s_add_i32 s50, s49, s19 // (op 1416) 


s_and_b32 s19, s0, s50 // (op 1416) 


s_add_i32 s0, s1, s19 // (op 1416) 


s_ashr_i32 s1, s0, s18 // (op 1416) 


s_lshl_b32 s0, s12, 24 // (op 1416) 
s_ashr_i32 s18, s0, 31 // (op 1416) 

s_xor_b32 s0, s1, s18 // (op 1416) 

s_sub_i32 s1, s0, s18 // (op 1416) 


s_mul_hi_i32 s0, s1, s13 // (op 1416) 
s_add_i32 s18, s0, s1 // (op 1416) 

s_ashr_i32 s0, s18, 31 // (op 1416) 
s_cmp_eq_i32 s13, 0 // (op 1416) 
s_mov_b32 s19, scc // (op 1416) 
s_cmp_eq_i32 s19, 0 // (op 1416) 
s_cselect_b32 s49, 0, -1 // (op 1416) 

s_and_b32 s19, s14, 31 // (op 1416) 
s_lshl_b32 s50, 1, s19 // (op 1416) 
s_add_i32 s51, s50, s49 // (op 1416) 


s_and_b32 s49, s0, s51 // (op 1416) 


s_add_i32 s0, s18, s49 // (op 1416) 


s_ashr_i32 s18, s0, s19 // (op 1416) 


s_lshl_b32 s0, s14, 24 // (op 1416) 
s_ashr_i32 s19, s0, 31 // (op 1416) 

s_xor_b32 s0, s18, s19 // (op 1416) 

s_sub_i32 s18, s0, s19 // (op 1416) 


s_add_u32 s50, s6, 64 // (op 1416) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1416) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1416) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1416) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1416) 

s_mul_i32 s0, s18, s50 // (op 1416) 

s_sub_i32 s49, s1, s0 // (op 1416) 


s_mul_hi_i32 s0, s16, s10 // (op 1416) 
s_add_i32 s1, s0, s16 // (op 1416) 

s_ashr_i32 s0, s1, 31 // (op 1416) 
s_cmp_eq_i32 s10, 0 // (op 1416) 
s_mov_b32 s18, scc // (op 1416) 
s_cmp_eq_i32 s18, 0 // (op 1416) 
s_cselect_b32 s19, 0, -1 // (op 1416) 

s_and_b32 s18, s47, 31 // (op 1416) 
s_lshl_b32 s72, 1, s18 // (op 1416) 
s_add_i32 s73, s72, s19 // (op 1416) 


s_and_b32 s19, s0, s73 // (op 1416) 


s_add_i32 s0, s1, s19 // (op 1416) 


s_ashr_i32 s1, s0, s18 // (op 1416) 


s_lshl_b32 s0, s47, 24 // (op 1416) 
s_ashr_i32 s18, s0, 31 // (op 1416) 

s_xor_b32 s0, s1, s18 // (op 1416) 

s_sub_i32 s1, s0, s18 // (op 1416) 


s_mul_i32 s0, s44, s50 // (op 1416) 

s_mul_i32 s18, s48, s0 // (op 1416) 

s_mul_i32 s0, s1, s18 // (op 1416) 

s_sub_u32 s18, s16, s0 // (op 1416) 

s_mul_hi_i32 s0, s18, s15 // (op 1416) 
s_add_i32 s19, s0, s18 // (op 1416) 


s_ashr_i32 s0, s19, 31 // (op 1416) 
s_cmp_eq_i32 s15, 0 // (op 1416) 
s_mov_b32 s18, scc // (op 1416) 
s_cmp_eq_i32 s18, 0 // (op 1416) 
s_cselect_b32 s50, 0, -1 // (op 1416) 

s_and_b32 s18, s46, 31 // (op 1416) 
s_lshl_b32 s51, 1, s18 // (op 1416) 
s_add_i32 s72, s51, s50 // (op 1416) 


s_and_b32 s50, s0, s72 // (op 1416) 


s_add_i32 s0, s19, s50 // (op 1416) 


s_ashr_i32 s19, s0, s18 // (op 1416) 


s_lshl_b32 s0, s46, 24 // (op 1416) 
s_ashr_i32 s18, s0, 31 // (op 1416) 

s_xor_b32 s0, s19, s18 // (op 1416) 

s_sub_i32 s19, s0, s18 // (op 1416) 


s_cmp_eq_i32 s1, 0 // (op 1416) 
s_mov_b32 s0, scc // (op 1416) 

s_cmp_eq_i32 s0, 0 // (op 1416) 
s_cselect_b32 s1, s19, s49 // (op 1416) 



v_lshl_add_u32 v23, s1, 6, v21 // (op 1416) 


v_mul_lo_u32 v21, v22, s38 // (op 1416) 
v_add_u32 v21, v21, v23 // (op 1416) 


s_mul_i32 s1, s38, s43 // (op 1418) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1418) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1418) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1418) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1418) most significant: sum
s_add_u32 s1, s1, s19 // (op 1418) most significant: sum


s_mov_b32 s68, s40 // (op 1418) reg expression
s_mov_b32 s69, s41 // (op 1418) reg expression
s_mov_b32 s70, s0 // (op 1418) reg expression
s_mov_b32 s71, 131072 // (op 1418) call()


s_mul_i32 s1, s29, 6 // (op 1029) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1029) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1029) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1029) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 1029) 


v_mov_b32 v24, v16 // (op 1029) reg expression
buffer_load_dwordx3 v[26:28], v24, s[52:55], 0 offen // (op 1029) Load value
buffer_load_dwordx3 v[30:32], v24, s[52:55], 0 offen offset:12 // (op 1029) Load value
buffer_load_dwordx3 v[34:36], v24, s[52:55], 0 offen offset:24 // (op 1029) Load value
buffer_load_dwordx3 v[38:40], v24, s[52:55], 0 offen offset:36 // (op 1029) Load value
v_add_u32 v24, v24, v22 // (op 1029) 
buffer_load_dwordx3 v[42:44], v24, s[52:55], 0 offen // (op 1029) Load value
buffer_load_dwordx3 v[46:48], v24, s[52:55], 0 offen offset:12 // (op 1029) Load value
buffer_load_dwordx3 v[50:52], v24, s[52:55], 0 offen offset:24 // (op 1029) Load value
buffer_load_dwordx3 v[54:56], v24, s[52:55], 0 offen offset:36 // (op 1029) Load value




v_mov_b32 v29, v20 // (op 1032) reg expression

buffer_load_dword v33, v29, s[68:71], 0 offen // (op 1032) Load value




buffer_load_dwordx4 v[58:61], v14, s[20:23], 0 offen // (op 1035) Load value
buffer_load_dwordx2 v[62:63], v14, s[20:23], 0 offen offset:16 // (op 1035) Load value



buffer_load_ubyte v29, v18, s[64:67], 0 offen offset: 0 // (op 1038) Load value

s_mul_i32 s1, s29, 6 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1041) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1041) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 1041) 


s_lshl_b64 s[0:1], s[28:29], 8 // (op 1041) 
s_mul_i32 s19, s1, 6 // (op 1041) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 1041) least significant
s_mul_hi_u32 s49, s0, 6 // (op 1041) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1041) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 1041) 

v_add_u32 v37, v16, s0 // (op 1041) 

buffer_load_dwordx3 v[64:66], v37, s[52:55], 0 offen // (op 1041) Load value
buffer_load_dwordx3 v[68:70], v37, s[52:55], 0 offen offset:12 // (op 1041) Load value
buffer_load_dwordx3 v[72:74], v37, s[52:55], 0 offen offset:24 // (op 1041) Load value
buffer_load_dwordx3 v[76:78], v37, s[52:55], 0 offen offset:36 // (op 1041) Load value
v_add_u32 v37, v37, v22 // (op 1041) 
buffer_load_dwordx3 v[80:82], v37, s[52:55], 0 offen // (op 1041) Load value
buffer_load_dwordx3 v[84:86], v37, s[52:55], 0 offen offset:12 // (op 1041) Load value
buffer_load_dwordx3 v[88:90], v37, s[52:55], 0 offen offset:24 // (op 1041) Load value
buffer_load_dwordx3 v[92:94], v37, s[52:55], 0 offen offset:36 // (op 1041) Load value





s_lshl_b32 s0, s38, 3 // (op 1044) 
v_add_u32 v16, v20, s0 // (op 1044) 


buffer_load_dword v37, v16, s[68:71], 0 offen // (op 1044) Load value





buffer_load_dwordx4 v[96:99], v14, s[20:23], 0 offen offset:192 // (op 1047) Load value
buffer_load_dwordx2 v[100:101], v14, s[20:23], 0 offen offset:208 // (op 1047) Load value




buffer_load_ubyte v14, v18, s[64:67], 0 offen offset:8 // (op 1050) Load value

v_lshlrev_b32 v16, 7, v1 // (op 1536) 
v_mul_lo_u32 v18, v16, 6 // (op 1536) 
v_lshrrev_b32 v20, 3, v18 // (op 1536) 

v_lshrrev_b32 v18, 4, v16 // (op 1536) 

v_lshlrev_b32 v16, 2, v18 // (op 1536) 

v_add_u32 v18, v20, v16 // (op 1536) 





s_waitcnt vmcnt(23)
ds_write_b96 v18, v[26:28] // (op 1052) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v18, v[30:32] offset:16 // (op 1052) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v18, v[34:36] offset:32 // (op 1052) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v18, v[38:40] offset:48 // (op 1052) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v18, v[42:44] offset:64 // (op 1052) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v18, v[46:48] offset:80 // (op 1052) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v18, v[50:52] offset:96 // (op 1052) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v18, v[54:56] offset:112 // (op 1052) Store local data 

v_and_b32 v16, -16, v1 // (op 1647) 
v_sub_u32 v18, v1, v16 // (op 1647) 

v_lshlrev_b32 v16, 2, v18 // (op 1647) 

v_lshrrev_b32 v18, 4, v1 // (op 1647) 
v_lshl_add_u32 v20, v18, 6, v16 // (op 1647) 





s_waitcnt vmcnt(15)
ds_write_b32 v20, v33 offset:16384 // (op 1053) Store local data 

v_and_b32 v16, -8, v1 // (op 1483) 
v_sub_u32 v18, v1, v16 // (op 1483) 

v_lshlrev_b32 v16, 5, v18 // (op 1483) 

v_lshrrev_b32 v18, 3, v1 // (op 1483) 
v_lshl_add_u32 v20, v18, 8, v16 // (op 1483) 


v_mul_lo_u32 v16, v20, 6 // (op 1483) 

v_lshrrev_b32 v18, 3, v16 // (op 1483) 




s_waitcnt vmcnt(14)
ds_write_b128 v18, v[58:61] offset:16896 // (op 1054) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v18, v[62:63] offset:16912 // (op 1054) Store local data 

v_and_b32 v16, -8, v1 // (op 1594) 
v_sub_u32 v18, v1, v16 // (op 1594) 

v_and_b32 v16, 4294967288, v1 // (op 1594) 
v_add_u32 v20, v16, v18 // (op 1594) 





s_waitcnt vmcnt(12)
ds_write_b8 v20, v29 offset:19968 // (op 1055) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1858) 



ds_read_b96_tr_b6 v[102:104], v6 // (op 1073) Transpose load local data 
ds_read_b96_tr_b6 v[105:107], v6 offset:1024 // (op 1073) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1899) 


ds_read_u8 v16, v9 offset:16384 // (op 1086) Load local data 



ds_read_b96_tr_b6 v[108:110], v6 offset:32 // (op 1078) Transpose load local data 
ds_read_b96_tr_b6 v[111:113], v6 offset:1056 // (op 1078) Transpose load local data 


ds_read_u8 v18, v9 offset:16416 // (op 1091) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1839) 



ds_read_b128 v[114:117], v3 offset:16896 // (op 1069) Load local data 
ds_read_b64 v[118:119], v3 offset:16912 // (op 1069) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1879) 



ds_read_u8 v20, v0 offset:19968 // (op 1082) Load local data 
v_and_b32 v41, -8, v1 // (op 1454) 
v_sub_u32 v45, v1, v41 // (op 1454) 

v_lshlrev_b32 v41, 5, v45 // (op 1454) 

v_lshrrev_b32 v45, 3, v1 // (op 1454) 
v_lshl_add_u32 v49, v45, 8, v41 // (op 1454) 


v_mul_lo_u32 v41, v49, 6 // (op 1454) 

v_lshrrev_b32 v45, 3, v41 // (op 1454) 

v_lshlrev_b32 v41, 7, v1 // (op 1507) 
v_mul_lo_u32 v49, v41, 6 // (op 1507) 
v_lshrrev_b32 v53, 3, v49 // (op 1507) 

v_lshrrev_b32 v49, 4, v41 // (op 1507) 

v_lshlrev_b32 v41, 2, v49 // (op 1507) 

v_add_u32 v49, v53, v41 // (op 1507) 


v_and_b32 v41, -8, v1 // (op 1565) 
v_sub_u32 v53, v1, v41 // (op 1565) 

v_and_b32 v41, 4294967288, v1 // (op 1565) 
v_add_u32 v57, v41, v53 // (op 1565) 


v_and_b32 v41, -16, v1 // (op 1618) 
v_sub_u32 v53, v1, v41 // (op 1618) 

v_lshlrev_b32 v41, 2, v53 // (op 1618) 

v_lshrrev_b32 v53, 4, v1 // (op 1618) 
v_lshl_add_u32 v67, v53, 6, v41 // (op 1618) 


v_and_b32 v41, -8, v1 // (op 1694) 
v_sub_u32 v53, v1, v41 // (op 1694) 

v_lshlrev_b32 v41, 5, v53 // (op 1694) 

v_lshrrev_b32 v53, 3, v1 // (op 1694) 
v_lshl_add_u32 v71, v53, 8, v41 // (op 1694) 


v_mul_lo_u32 v41, v71, 6 // (op 1694) 

v_lshrrev_b32 v53, 3, v41 // (op 1694) 

v_lshlrev_b32 v41, 7, v1 // (op 1718) 
v_mul_lo_u32 v71, v41, 6 // (op 1718) 
v_lshrrev_b32 v75, 3, v71 // (op 1718) 

v_lshrrev_b32 v71, 4, v41 // (op 1718) 

v_lshlrev_b32 v41, 2, v71 // (op 1718) 

v_add_u32 v71, v75, v41 // (op 1718) 


v_and_b32 v41, -8, v1 // (op 1747) 
v_sub_u32 v75, v1, v41 // (op 1747) 

v_and_b32 v41, 4294967288, v1 // (op 1747) 
v_add_u32 v79, v41, v75 // (op 1747) 


v_and_b32 v41, -16, v1 // (op 1771) 
v_sub_u32 v75, v1, v41 // (op 1771) 

v_lshlrev_b32 v41, 2, v75 // (op 1771) 

v_lshrrev_b32 v75, 4, v1 // (op 1771) 
v_lshl_add_u32 v83, v75, 6, v41 // (op 1771) 


s_mov_b32 s0, 0 // (op 400) call()
s_ashr_i64 s[18:19], s[32:33], 8 // (op 399) 
s_mov_b32 s1, s18 // (op 399) convert

s_and_b32 s18, s1, 4294967294 // (op 399) 

s_cmp_lt_u32 s0, s18 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399 // (op 399) Condition: Top (jump to RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399 if false)
RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399:

s_add_u32 s0, s0, 2 // (op 401) 
s_mul_i32 s19, s29, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s28, 6 // (op 10) least significant
s_mul_hi_u32 s1, s28, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 10) 


v_mov_b32 v41, v17 // (op 10) 
s_lshl_b64 s[18:19], s[28:29], 8 // (op 10) 
s_mul_i32 s51, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 10) least significant
s_mul_hi_u32 s1, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v75, v41, s1 // (op 10) 

buffer_load_dwordx3 v[26:28], v75, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[30:32], v75, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[34:36], v75, s[52:55], 0 offen offset:24 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[114:119], v[102:107], a[0:3], v20, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 88) 
buffer_load_dwordx3 v[38:40], v75, s[52:55], 0 offen offset:36 // (op 10) Load value
s_barrier  // (op 1848) 
v_add_u32 v75, v75, v22 // (op 10) 



buffer_load_dwordx3 v[42:44], v75, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[46:48], v75, s[52:55], 0 offen offset:12 // (op 10) Load value
ds_read_b128 v[120:123], v3 offset:16992 // (op 91) Load local data 
ds_read_b64 v[124:125], v3 offset:17008 // (op 91) Load local data 
buffer_load_dwordx3 v[50:52], v75, s[52:55], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 1868) 
buffer_load_dwordx3 v[54:56], v75, s[52:55], 0 offen offset:36 // (op 10) Load value







ds_read_b96_tr_b6 v[102:104], v6 offset:8192 // (op 95) Transpose load local data 
ds_read_b96_tr_b6 v[105:107], v6 offset:9216 // (op 95) Transpose load local data 
v_mov_b32 v75, v21 // (op 25) 
s_barrier  // (op 1889) 
s_lshl_b32 s1, s38, 4 // (op 25) 



ds_read_u8 v87, v0 offset:19972 // (op 99) Load local data 
v_add_u32 v91, v75, s1 // (op 25) 


s_barrier  // (op 1909) 
buffer_load_dword v33, v91, s[68:71], 0 offen // (op 25) Load value





v_mov_b32 v91, v15 // (op 4) 

ds_read_u8 v16, v9 offset:16640 // (op 103) Load local data 
buffer_load_dwordx4 v[58:61], v91, s[20:23], 0 offen offset:384 // (op 4) Load value
buffer_load_dwordx2 v[62:63], v91, s[20:23], 0 offen offset:400 // (op 4) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:125], v[102:107], a[0:3], v87, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 105) 


v_mov_b32 v95, v19 // (op 17) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[114:119], v[108:113], a[4:7], v20, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 447) 
buffer_load_ubyte v29, v95, s[64:67], 0 offen offset:16 // (op 17) Load value



s_barrier  // (op 1929) 
ds_read_b96_tr_b6 v[108:110], v6 offset:8224 // (op 451) Transpose load local data 



ds_read_b96_tr_b6 v[111:113], v6 offset:9248 // (op 451) Transpose load local data 
s_waitcnt vmcnt(23)
ds_write_b96 v71, v[64:66] offset:20096 // (op 302) Store local data 


ds_read_u8 v18, v9 offset:16672 // (op 455) Load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v71, v[68:70] offset:20112 // (op 302) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:125], v[108:113], a[4:7], v87, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 456) 
s_waitcnt vmcnt(21)
ds_write_b96 v71, v[72:74] offset:20128 // (op 302) Store local data 


s_waitcnt vmcnt(20)
ds_write_b96 v71, v[76:78] offset:20144 // (op 302) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v71, v[80:82] offset:20160 // (op 302) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v71, v[84:86] offset:20176 // (op 302) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v71, v[88:90] offset:20192 // (op 302) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v71, v[92:94] offset:20208 // (op 302) Store local data 
s_barrier  // (op 1941) 



s_waitcnt vmcnt(15)
ds_write_b32 v83, v37 offset:36480 // (op 304) Store local data 
s_barrier  // (op 1922) 



s_waitcnt vmcnt(14)
ds_write_b128 v53, v[96:99] offset:36992 // (op 283) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v53, v[100:101] offset:37008 // (op 283) Store local data 
s_barrier  // (op 1935) 



s_waitcnt vmcnt(12)
ds_write_b8 v79, v14 offset:40064 // (op 303) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1925) 



ds_read_b96_tr_b6 v[102:104], v12 offset:20096 // (op 287) Transpose load local data 
ds_read_b96_tr_b6 v[105:107], v12 offset:21120 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1938) 


ds_read_u8 v16, v13 offset:36480 // (op 291) Load local data 



ds_read_b96_tr_b6 v[120:122], v12 offset:20128 // (op 470) Transpose load local data 
ds_read_b96_tr_b6 v[123:125], v12 offset:21152 // (op 470) Transpose load local data 


ds_read_u8 v87, v13 offset:36512 // (op 474) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1919) 



ds_read_b128 v[114:117], v5 offset:36992 // (op 285) Load local data 
ds_read_b64 v[118:119], v5 offset:37008 // (op 285) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1932) 



ds_read_u8 v20, v4 offset:40064 // (op 289) Load local data 

s_mul_i32 s19, s29, 6 // (op 305) most significant: low of high * low
s_mul_i32 s18, s28, 6 // (op 305) least significant
s_mul_hi_u32 s1, s28, 6 // (op 305) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 305) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 305) 


s_lshl_b64 s[18:19], s[28:29], 8 // (op 305) 
s_mul_i32 s51, s19, 6 // (op 305) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 305) least significant
s_mul_hi_u32 s1, s18, 6 // (op 305) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 305) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 305) 

s_mul_i32 s1, 3, s18 // (op 305) 

v_add_u32 v24, v41, s1 // (op 305) 

buffer_load_dwordx3 v[64:66], v24, s[52:55], 0 offen // (op 305) Load value
buffer_load_dwordx3 v[68:70], v24, s[52:55], 0 offen offset:12 // (op 305) Load value
buffer_load_dwordx3 v[72:74], v24, s[52:55], 0 offen offset:24 // (op 305) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[114:119], v[102:107], a[0:3], v20, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 292) 
buffer_load_dwordx3 v[76:78], v24, s[52:55], 0 offen offset:36 // (op 305) Load value
v_add_u32 v24, v24, v22 // (op 305) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[114:119], v[120:125], a[4:7], v20, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 457) 
buffer_load_dwordx3 v[80:82], v24, s[52:55], 0 offen // (op 305) Load value


buffer_load_dwordx3 v[84:86], v24, s[52:55], 0 offen offset:12 // (op 305) Load value
ds_read_b128 v[120:123], v5 offset:37088 // (op 294) Load local data 
ds_read_b64 v[124:125], v5 offset:37104 // (op 294) Load local data 
buffer_load_dwordx3 v[88:90], v24, s[52:55], 0 offen offset:24 // (op 305) Load value
buffer_load_dwordx3 v[92:94], v24, s[52:55], 0 offen offset:36 // (op 305) Load value






s_barrier  // (op 1986) 


s_lshl_b32 s1, s38, 3 // (op 307) 
s_mul_i32 s18, 3, s1 // (op 307) 

v_add_u32 v41, v75, s18 // (op 307) 


buffer_load_dword v37, v41, s[68:71], 0 offen // (op 307) Load value

s_lshl_b64 s[18:19], s[28:29], 8 // (op 1280) 

s_mul_i32 s51, s19, 6 // (op 1280) most significant: low of high * low



s_mul_i32 s50, s18, 6 // (op 1280) least significant
s_mul_hi_u32 s1, s18, 6 // (op 1280) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 1280) most significant: sum


s_lshl_b32 s1, s38, 4 // (op 1408) 
v_add_u32 v21, v21, s1 // (op 1408) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 1280) 

s_lshl_b32 s1, s18, 1 // (op 1280) 

v_add_u32 v17, v17, s1 // (op 1280) 

ds_read_b96_tr_b6 v[126:128], v12 offset:28288 // (op 296) Transpose load local data 
ds_read_b96_tr_b6 v[129:131], v12 offset:29312 // (op 296) Transpose load local data 
buffer_load_dwordx4 v[96:99], v91, s[20:23], 0 offen offset:576 // (op 282) Load value



ds_read_u8 v41, v4 offset:40068 // (op 298) Load local data 
buffer_load_dwordx2 v[100:101], v91, s[20:23], 0 offen offset:592 // (op 282) Load value






s_barrier  // (op 1990) 
v_add_u32 v15, 384, v15 // (op 1216) 
ds_read_u8 v75, v13 offset:36736 // (op 300) Load local data 
buffer_load_ubyte v14, v95, s[64:67], 0 offen offset:24 // (op 306) Load value




ds_read_b96_tr_b6 v[132:134], v12 offset:28320 // (op 461) Transpose load local data 
s_barrier  // (op 1861) 
v_add_u32 v19, 16, v19 // (op 1344) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:125], v[126:131], a[0:3], v41, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 301) 
ds_read_b96_tr_b6 v[135:137], v12 offset:29344 // (op 461) Transpose load local data 


s_waitcnt vmcnt(23)
ds_write_b96 v49, v[26:28] // (op 13) Store local data 


s_barrier  // (op 2026) 
ds_read_u8 v75, v13 offset:36768 // (op 465) Load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v49, v[30:32] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v49, v[34:36] offset:32 // (op 13) Store local data 
s_barrier  // (op 2030) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:125], v[132:137], a[4:7], v41, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 466) 
s_waitcnt vmcnt(20)
ds_write_b96 v49, v[38:40] offset:48 // (op 13) Store local data 




s_waitcnt vmcnt(19)
ds_write_b96 v49, v[42:44] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v49, v[46:48] offset:80 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v49, v[50:52] offset:96 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v49, v[54:56] offset:112 // (op 13) Store local data 
s_barrier  // (op 1902) 



s_waitcnt vmcnt(15)
ds_write_b32 v67, v33 offset:16384 // (op 30) Store local data 
s_barrier  // (op 1842) 



s_waitcnt vmcnt(14)
ds_write_b128 v45, v[58:61] offset:16896 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v45, v[62:63] offset:16912 // (op 6) Store local data 
s_barrier  // (op 1882) 



s_waitcnt vmcnt(12)
ds_write_b8 v57, v29 offset:19968 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1871) 



ds_read_b96_tr_b6 v[102:104], v6 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[105:107], v6 offset:1024 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1912) 


ds_read_u8 v16, v9 offset:16384 // (op 85) Load local data 



ds_read_b96_tr_b6 v[108:110], v6 offset:32 // (op 442) Transpose load local data 
ds_read_b96_tr_b6 v[111:113], v6 offset:1056 // (op 442) Transpose load local data 


ds_read_u8 v18, v9 offset:16416 // (op 446) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1851) 



ds_read_b128 v[114:117], v3 offset:16896 // (op 70) Load local data 
ds_read_b64 v[118:119], v3 offset:16912 // (op 70) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1892) 



ds_read_u8 v20, v0 offset:19968 // (op 80) Load local data 
s_ashr_i64 s[18:19], s[32:33], 8 // (op 399) 
s_mov_b32 s1, s18 // (op 399) convert

s_and_b32 s18, s1, 4294967294 // (op 399) 

s_cmp_lt_u32 s0, s18 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399 // (op 399) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoop_399 if true)
RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoop_399:

















s_mul_hi_i32 s0, s16, s11 // (op 1178) 
v_lshrrev_b32 v15, 3, v1 // (op 1178) 
s_add_i32 s1, s0, s16 // (op 1178) 

v_and_b32 v17, -8, v1 // (op 1178) 
s_ashr_i32 s0, s1, 31 // (op 1178) 
v_sub_u32 v19, v1, v17 // (op 1178) 

s_cmp_eq_i32 s11, 0 // (op 1178) 
s_mov_b32 s18, scc // (op 1178) 
v_lshlrev_b32 v17, 5, v19 // (op 1178) 

s_cmp_eq_i32 s18, 0 // (op 1178) 
s_cselect_b32 s19, 0, -1 // (op 1178) 

s_and_b32 s18, s12, 31 // (op 1178) 
s_lshl_b32 s49, 1, s18 // (op 1178) 
s_add_i32 s50, s49, s19 // (op 1178) 


s_and_b32 s19, s0, s50 // (op 1178) 


s_add_i32 s0, s1, s19 // (op 1178) 


s_ashr_i32 s1, s0, s18 // (op 1178) 


s_lshl_b32 s0, s12, 24 // (op 1178) 
s_ashr_i32 s18, s0, 31 // (op 1178) 

s_xor_b32 s0, s1, s18 // (op 1178) 

s_sub_i32 s1, s0, s18 // (op 1178) 


s_mul_hi_i32 s0, s1, s13 // (op 1178) 
s_add_i32 s18, s0, s1 // (op 1178) 

s_ashr_i32 s0, s18, 31 // (op 1178) 
s_cmp_eq_i32 s13, 0 // (op 1178) 
s_mov_b32 s19, scc // (op 1178) 
s_cmp_eq_i32 s19, 0 // (op 1178) 
s_cselect_b32 s49, 0, -1 // (op 1178) 

s_and_b32 s19, s14, 31 // (op 1178) 
s_lshl_b32 s50, 1, s19 // (op 1178) 
s_add_i32 s51, s50, s49 // (op 1178) 


s_and_b32 s49, s0, s51 // (op 1178) 


s_add_i32 s0, s18, s49 // (op 1178) 


s_ashr_i32 s18, s0, s19 // (op 1178) 


s_lshl_b32 s0, s14, 24 // (op 1178) 
s_ashr_i32 s19, s0, 31 // (op 1178) 

s_xor_b32 s0, s18, s19 // (op 1178) 

s_sub_i32 s18, s0, s19 // (op 1178) 


s_mul_i32 s0, s1, s44 // (op 1178) 

s_sub_u32 s1, s16, s0 // (op 1178) 

s_mul_i32 s0, s18, s44 // (op 1178) 
s_add_u32 s0, s0, s1 // (op 1178) 


s_mul_hi_i32 s1, s16, s10 // (op 1178) 
s_add_i32 s18, s1, s16 // (op 1178) 

s_ashr_i32 s1, s18, 31 // (op 1178) 
s_cmp_eq_i32 s10, 0 // (op 1178) 
s_mov_b32 s19, scc // (op 1178) 
s_cmp_eq_i32 s19, 0 // (op 1178) 
s_cselect_b32 s49, 0, -1 // (op 1178) 

s_and_b32 s19, s47, 31 // (op 1178) 
s_lshl_b32 s50, 1, s19 // (op 1178) 
s_add_i32 s51, s50, s49 // (op 1178) 


s_and_b32 s49, s1, s51 // (op 1178) 


s_add_i32 s1, s18, s49 // (op 1178) 


s_ashr_i32 s18, s1, s19 // (op 1178) 


s_lshl_b32 s1, s47, 24 // (op 1178) 
s_ashr_i32 s19, s1, 31 // (op 1178) 

s_xor_b32 s1, s18, s19 // (op 1178) 

s_sub_i32 s18, s1, s19 // (op 1178) 


s_add_u32 s50, s6, 64 // (op 1178) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1178) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1178) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1178) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1178) 

s_mul_i32 s1, s44, s50 // (op 1178) 

s_mul_i32 s19, s48, s1 // (op 1178) 

s_mul_i32 s1, s18, s19 // (op 1178) 

s_sub_u32 s19, s16, s1 // (op 1178) 

s_mov_b32 s1, s19 // (op 1178) 
s_mul_hi_i32 s49, s1, s15 // (op 1178) 
s_add_i32 s50, s49, s1 // (op 1178) 


s_ashr_i32 s1, s50, 31 // (op 1178) 
s_cmp_eq_i32 s15, 0 // (op 1178) 
s_mov_b32 s49, scc // (op 1178) 
s_cmp_eq_i32 s49, 0 // (op 1178) 
s_cselect_b32 s51, 0, -1 // (op 1178) 

s_and_b32 s49, s46, 31 // (op 1178) 
s_lshl_b32 s72, 1, s49 // (op 1178) 
s_add_i32 s73, s72, s51 // (op 1178) 


s_and_b32 s51, s1, s73 // (op 1178) 


s_add_i32 s1, s50, s51 // (op 1178) 


s_ashr_i32 s50, s1, s49 // (op 1178) 


s_lshl_b32 s1, s46, 24 // (op 1178) 
s_ashr_i32 s49, s1, 31 // (op 1178) 

s_xor_b32 s1, s50, s49 // (op 1178) 

s_sub_i32 s50, s1, s49 // (op 1178) 


s_mul_i32 s1, s45, s44 // (op 1178) 
s_add_u32 s72, s8, 16 // (op 1178) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 1178) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1178) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1178) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1178) 

s_sub_i32 s49, s72, s1 // (op 1178) 

s_mul_i32 s51, s50, s49 // (op 1178) 


s_sub_u32 s49, s19, s51 // (op 1178) 


s_add_u32 s19, s49, s1 // (op 1178) 


s_cmp_eq_i32 s18, 0 // (op 1178) 
s_mov_b32 s1, scc // (op 1178) 

s_cmp_eq_i32 s1, 0 // (op 1178) 
s_cselect_b32 s18, s19, s0 // (op 1178) 



v_lshl_add_u32 v19, s18, 4, v15 // (op 1178) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 1178) 
s_lshr_b32 s18, s0, 1 // (op 1178) 

v_lshl_add_u32 v15, s18, 9, v17 // (op 1178) 


v_mov_b32 v39, 0 // (op 1178) convert
v_mov_b32 v38, v15 // (op 1178) convert

v_mul_lo_u32 v41, v19, s25 // (op 1178) most significant: low of low * high
v_mul_hi_u32 v15, v19, s24 // (op 1178) most significant: high of low * low
v_mul_lo_u32 v40, v19, s24 // (op 1178) least significant: low of low * low
v_add_u32 v41, v41, v15 // (op 1178) most significant: sum

v_add_co_u32 v40, s[0:1], v40, v38 // (op 1178) least significant half
v_addc_co_u32 v41, s[0:1], v41, v39, s[0:1] // (op 1178) most significant half



v_mul_lo_u32 v39, v41, 6 // (op 1178) most significant: low of high * low
v_mul_hi_u32 v15, v40, 6 // (op 1178) most significant: high of low * low
v_mul_lo_u32 v38, v40, 6 // (op 1178) least significant: low of low * low
v_add_u32 v39, v39, v15 // (op 1178) most significant: sum


v_lshrrev_b64 v[40:41], 3, v[38:39] // (op 1178) 

v_mov_b32 v15, v40 // (op 1178) convert

s_mul_i32 s1, s9, 6 // (op 1180) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1180) least significant
s_mul_hi_u32 s18, s8, 6 // (op 1180) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1180) most significant: sum

s_mul_i32 s19, s24, s1 // (op 1180) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 1180) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 1180) least significant
s_mul_hi_u32 s50, s24, s0 // (op 1180) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1180) most significant: sum
s_add_u32 s19, s19, s50 // (op 1180) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1180) 

s_mov_b32 s20, s26 // (op 1180) reg expression
s_mov_b32 s21, s27 // (op 1180) reg expression
s_mov_b32 s22, s0 // (op 1180) reg expression
s_mov_b32 s23, 131072 // (op 1180) call()

s_mul_hi_i32 s0, s16, s11 // (op 1246) 
v_lshlrev_b32 v17, 1, v1 // (op 1246) 
s_add_i32 s1, s0, s16 // (op 1246) 

s_ashr_i32 s0, s1, 31 // (op 1246) 
s_cmp_eq_i32 s11, 0 // (op 1246) 
s_mov_b32 s18, scc // (op 1246) 
s_cmp_eq_i32 s18, 0 // (op 1246) 
s_cselect_b32 s19, 0, -1 // (op 1246) 

s_and_b32 s18, s12, 31 // (op 1246) 
s_lshl_b32 s49, 1, s18 // (op 1246) 
s_add_i32 s50, s49, s19 // (op 1246) 


s_and_b32 s19, s0, s50 // (op 1246) 


s_add_i32 s0, s1, s19 // (op 1246) 


s_ashr_i32 s1, s0, s18 // (op 1246) 


s_lshl_b32 s0, s12, 24 // (op 1246) 
s_ashr_i32 s18, s0, 31 // (op 1246) 

s_xor_b32 s0, s1, s18 // (op 1246) 

s_sub_i32 s1, s0, s18 // (op 1246) 


s_mul_hi_i32 s0, s1, s13 // (op 1246) 
s_add_i32 s18, s0, s1 // (op 1246) 

s_ashr_i32 s0, s18, 31 // (op 1246) 
s_cmp_eq_i32 s13, 0 // (op 1246) 
s_mov_b32 s19, scc // (op 1246) 
s_cmp_eq_i32 s19, 0 // (op 1246) 
s_cselect_b32 s49, 0, -1 // (op 1246) 

s_and_b32 s19, s14, 31 // (op 1246) 
s_lshl_b32 s50, 1, s19 // (op 1246) 
s_add_i32 s51, s50, s49 // (op 1246) 


s_and_b32 s49, s0, s51 // (op 1246) 


s_add_i32 s0, s18, s49 // (op 1246) 


s_ashr_i32 s18, s0, s19 // (op 1246) 


s_lshl_b32 s0, s14, 24 // (op 1246) 
s_ashr_i32 s19, s0, 31 // (op 1246) 

s_xor_b32 s0, s18, s19 // (op 1246) 

s_sub_i32 s18, s0, s19 // (op 1246) 


s_add_u32 s50, s6, 64 // (op 1246) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1246) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1246) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1246) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1246) 

s_mul_i32 s0, s18, s50 // (op 1246) 

s_sub_i32 s49, s1, s0 // (op 1246) 


s_mul_hi_i32 s0, s16, s10 // (op 1246) 
s_add_i32 s1, s0, s16 // (op 1246) 

s_ashr_i32 s0, s1, 31 // (op 1246) 
s_cmp_eq_i32 s10, 0 // (op 1246) 
s_mov_b32 s18, scc // (op 1246) 
s_cmp_eq_i32 s18, 0 // (op 1246) 
s_cselect_b32 s19, 0, -1 // (op 1246) 

s_and_b32 s18, s47, 31 // (op 1246) 
s_lshl_b32 s72, 1, s18 // (op 1246) 
s_add_i32 s73, s72, s19 // (op 1246) 


s_and_b32 s19, s0, s73 // (op 1246) 


s_add_i32 s0, s1, s19 // (op 1246) 


s_ashr_i32 s1, s0, s18 // (op 1246) 


s_lshl_b32 s0, s47, 24 // (op 1246) 
s_ashr_i32 s18, s0, 31 // (op 1246) 

s_xor_b32 s0, s1, s18 // (op 1246) 

s_sub_i32 s1, s0, s18 // (op 1246) 


s_mul_i32 s0, s44, s50 // (op 1246) 

s_mul_i32 s18, s48, s0 // (op 1246) 

s_mul_i32 s0, s1, s18 // (op 1246) 

s_sub_u32 s18, s16, s0 // (op 1246) 

s_mul_hi_i32 s0, s18, s15 // (op 1246) 
s_add_i32 s19, s0, s18 // (op 1246) 


s_ashr_i32 s0, s19, 31 // (op 1246) 
s_cmp_eq_i32 s15, 0 // (op 1246) 
s_mov_b32 s18, scc // (op 1246) 
s_cmp_eq_i32 s18, 0 // (op 1246) 
s_cselect_b32 s50, 0, -1 // (op 1246) 

s_and_b32 s18, s46, 31 // (op 1246) 
s_lshl_b32 s51, 1, s18 // (op 1246) 
s_add_i32 s72, s51, s50 // (op 1246) 


s_and_b32 s50, s0, s72 // (op 1246) 


s_add_i32 s0, s19, s50 // (op 1246) 


s_ashr_i32 s19, s0, s18 // (op 1246) 


s_lshl_b32 s0, s46, 24 // (op 1246) 
s_ashr_i32 s18, s0, 31 // (op 1246) 

s_xor_b32 s0, s19, s18 // (op 1246) 

s_sub_i32 s19, s0, s18 // (op 1246) 


s_cmp_eq_i32 s1, 0 // (op 1246) 
s_mov_b32 s0, scc // (op 1246) 

s_cmp_eq_i32 s0, 0 // (op 1246) 
s_cselect_b32 s1, s19, s49 // (op 1246) 



s_lshl_b32 s0, s1, 6 // (op 1246) 

s_mov_b32 s19, 0 // (op 1246) convert
s_mov_b32 s18, s0 // (op 1246) convert

s_ashr_i64 s[0:1], s[32:33], 8 // (op 1246) 
s_lshr_b32 s49, s0, 1 // (op 1246) 

v_lshl_add_u32 v19, s49, 9, v17 // (op 1246) 


v_mul_lo_u32 v39, v19, s29 // (op 1246) most significant: low of low * high
v_mul_hi_u32 v17, v19, s28 // (op 1246) most significant: high of low * low
v_mul_lo_u32 v38, v19, s28 // (op 1246) least significant: low of low * low
v_add_u32 v39, v39, v17 // (op 1246) most significant: sum

v_mov_b32 v17, s18 // (op 1246) 
v_mov_b32 v67, s19 // (op 1246) 
v_add_co_u32 v38, s[0:1], v38, v17 // (op 1246) least significant half
v_addc_co_u32 v39, s[0:1], v39, v67, s[0:1] // (op 1246) most significant half





v_mul_lo_u32 v41, v39, 6 // (op 1246) most significant: low of high * low
v_mul_hi_u32 v17, v38, 6 // (op 1246) most significant: high of low * low
v_mul_lo_u32 v40, v38, 6 // (op 1246) least significant: low of low * low
v_add_u32 v41, v41, v17 // (op 1246) most significant: sum


v_lshrrev_b64 v[38:39], 3, v[40:41] // (op 1246) 

v_mov_b32 v17, v38 // (op 1246) convert

s_mul_i32 s1, s33, 6 // (op 1248) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 1248) least significant
s_mul_hi_u32 s18, s32, 6 // (op 1248) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1248) most significant: sum

s_mul_i32 s19, s28, s1 // (op 1248) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 1248) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 1248) least significant
s_mul_hi_u32 s50, s28, s0 // (op 1248) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1248) most significant: sum
s_add_u32 s19, s19, s50 // (op 1248) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1248) 

s_mov_b32 s52, s30 // (op 1248) reg expression
s_mov_b32 s53, s31 // (op 1248) reg expression
s_mov_b32 s54, s0 // (op 1248) reg expression
s_mov_b32 s55, 131072 // (op 1248) call()

s_mul_hi_i32 s0, s16, s11 // (op 1310) 
v_lshrrev_b32 v19, 3, v1 // (op 1310) 
s_add_i32 s1, s0, s16 // (op 1310) 

v_and_b32 v67, -8, v1 // (op 1310) 
s_ashr_i32 s0, s1, 31 // (op 1310) 
v_sub_u32 v71, v1, v67 // (op 1310) 

s_cmp_eq_i32 s11, 0 // (op 1310) 
s_mov_b32 s18, scc // (op 1310) 
s_cmp_eq_i32 s18, 0 // (op 1310) 
s_cselect_b32 s19, 0, -1 // (op 1310) 

s_and_b32 s18, s12, 31 // (op 1310) 
s_lshl_b32 s49, 1, s18 // (op 1310) 
s_add_i32 s50, s49, s19 // (op 1310) 


s_and_b32 s19, s0, s50 // (op 1310) 


s_add_i32 s0, s1, s19 // (op 1310) 


s_ashr_i32 s1, s0, s18 // (op 1310) 


s_lshl_b32 s0, s12, 24 // (op 1310) 
s_ashr_i32 s18, s0, 31 // (op 1310) 

s_xor_b32 s0, s1, s18 // (op 1310) 

s_sub_i32 s1, s0, s18 // (op 1310) 


s_mul_hi_i32 s0, s1, s13 // (op 1310) 
s_add_i32 s18, s0, s1 // (op 1310) 

s_ashr_i32 s0, s18, 31 // (op 1310) 
s_cmp_eq_i32 s13, 0 // (op 1310) 
s_mov_b32 s19, scc // (op 1310) 
s_cmp_eq_i32 s19, 0 // (op 1310) 
s_cselect_b32 s49, 0, -1 // (op 1310) 

s_and_b32 s19, s14, 31 // (op 1310) 
s_lshl_b32 s50, 1, s19 // (op 1310) 
s_add_i32 s51, s50, s49 // (op 1310) 


s_and_b32 s49, s0, s51 // (op 1310) 


s_add_i32 s0, s18, s49 // (op 1310) 


s_ashr_i32 s18, s0, s19 // (op 1310) 


s_lshl_b32 s0, s14, 24 // (op 1310) 
s_ashr_i32 s19, s0, 31 // (op 1310) 

s_xor_b32 s0, s18, s19 // (op 1310) 

s_sub_i32 s18, s0, s19 // (op 1310) 


s_mul_i32 s0, s1, s44 // (op 1310) 

s_sub_u32 s1, s16, s0 // (op 1310) 

s_mul_i32 s0, s18, s44 // (op 1310) 
s_add_u32 s0, s0, s1 // (op 1310) 


s_mul_hi_i32 s1, s16, s10 // (op 1310) 
s_add_i32 s18, s1, s16 // (op 1310) 

s_ashr_i32 s1, s18, 31 // (op 1310) 
s_cmp_eq_i32 s10, 0 // (op 1310) 
s_mov_b32 s19, scc // (op 1310) 
s_cmp_eq_i32 s19, 0 // (op 1310) 
s_cselect_b32 s49, 0, -1 // (op 1310) 

s_and_b32 s19, s47, 31 // (op 1310) 
s_lshl_b32 s50, 1, s19 // (op 1310) 
s_add_i32 s51, s50, s49 // (op 1310) 


s_and_b32 s49, s1, s51 // (op 1310) 


s_add_i32 s1, s18, s49 // (op 1310) 


s_ashr_i32 s18, s1, s19 // (op 1310) 


s_lshl_b32 s1, s47, 24 // (op 1310) 
s_ashr_i32 s19, s1, 31 // (op 1310) 

s_xor_b32 s1, s18, s19 // (op 1310) 

s_sub_i32 s18, s1, s19 // (op 1310) 


s_add_u32 s50, s6, 64 // (op 1310) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1310) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1310) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1310) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1310) 

s_mul_i32 s1, s44, s50 // (op 1310) 

s_mul_i32 s19, s48, s1 // (op 1310) 

s_mul_i32 s1, s18, s19 // (op 1310) 

s_sub_u32 s19, s16, s1 // (op 1310) 

s_mov_b32 s1, s19 // (op 1310) 
s_mul_hi_i32 s49, s1, s15 // (op 1310) 
s_add_i32 s50, s49, s1 // (op 1310) 


s_ashr_i32 s1, s50, 31 // (op 1310) 
s_cmp_eq_i32 s15, 0 // (op 1310) 
s_mov_b32 s49, scc // (op 1310) 
s_cmp_eq_i32 s49, 0 // (op 1310) 
s_cselect_b32 s51, 0, -1 // (op 1310) 

s_and_b32 s49, s46, 31 // (op 1310) 
s_lshl_b32 s72, 1, s49 // (op 1310) 
s_add_i32 s73, s72, s51 // (op 1310) 


s_and_b32 s51, s1, s73 // (op 1310) 


s_add_i32 s1, s50, s51 // (op 1310) 


s_ashr_i32 s50, s1, s49 // (op 1310) 


s_lshl_b32 s1, s46, 24 // (op 1310) 
s_ashr_i32 s49, s1, 31 // (op 1310) 

s_xor_b32 s1, s50, s49 // (op 1310) 

s_sub_i32 s50, s1, s49 // (op 1310) 


s_mul_i32 s1, s45, s44 // (op 1310) 
s_add_u32 s72, s8, 16 // (op 1310) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 1310) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1310) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1310) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1310) 

s_sub_i32 s49, s72, s1 // (op 1310) 

s_mul_i32 s51, s50, s49 // (op 1310) 


s_sub_u32 s49, s19, s51 // (op 1310) 


s_add_u32 s19, s49, s1 // (op 1310) 


s_cmp_eq_i32 s18, 0 // (op 1310) 
s_mov_b32 s1, scc // (op 1310) 

s_cmp_eq_i32 s1, 0 // (op 1310) 
s_cselect_b32 s18, s19, s0 // (op 1310) 



v_lshl_add_u32 v67, s18, 4, v19 // (op 1310) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 1310) 
s_lshr_b32 s18, s0, 1 // (op 1310) 

v_lshl_add_u32 v19, s18, 4, v71 // (op 1310) 


v_mul_lo_u32 v71, v67, s34 // (op 1310) 
v_add_u32 v71, v71, v19 // (op 1310) 


s_mul_i32 s1, s34, s9 // (op 1312) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1312) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1312) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1312) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1312) most significant: sum
s_add_u32 s1, s1, s19 // (op 1312) most significant: sum


s_mov_b32 s64, s36 // (op 1312) reg expression
s_mov_b32 s65, s37 // (op 1312) reg expression
s_mov_b32 s66, s0 // (op 1312) reg expression
s_mov_b32 s67, 131072 // (op 1312) call()

s_mul_hi_i32 s0, s16, s11 // (op 1374) 
v_and_b32 v19, -16, v1 // (op 1374) 
s_add_i32 s1, s0, s16 // (op 1374) 

v_sub_u32 v67, v1, v19 // (op 1374) 

s_ashr_i32 s0, s1, 31 // (op 1374) 
v_lshlrev_b32 v19, 2, v67 // (op 1374) 

s_cmp_eq_i32 s11, 0 // (op 1374) 
s_mov_b32 s18, scc // (op 1374) 
v_lshrrev_b32 v67, 4, v1 // (op 1374) 
s_cmp_eq_i32 s18, 0 // (op 1374) 
s_cselect_b32 s19, 0, -1 // (op 1374) 

s_and_b32 s18, s12, 31 // (op 1374) 
s_lshl_b32 s49, 1, s18 // (op 1374) 
s_add_i32 s50, s49, s19 // (op 1374) 


s_and_b32 s19, s0, s50 // (op 1374) 


s_add_i32 s0, s1, s19 // (op 1374) 


s_ashr_i32 s1, s0, s18 // (op 1374) 


s_lshl_b32 s0, s12, 24 // (op 1374) 
s_ashr_i32 s18, s0, 31 // (op 1374) 

s_xor_b32 s0, s1, s18 // (op 1374) 

s_sub_i32 s1, s0, s18 // (op 1374) 


s_mul_hi_i32 s0, s1, s13 // (op 1374) 
s_add_i32 s18, s0, s1 // (op 1374) 

s_ashr_i32 s0, s18, 31 // (op 1374) 
s_cmp_eq_i32 s13, 0 // (op 1374) 
s_mov_b32 s19, scc // (op 1374) 
s_cmp_eq_i32 s19, 0 // (op 1374) 
s_cselect_b32 s49, 0, -1 // (op 1374) 

s_and_b32 s19, s14, 31 // (op 1374) 
s_lshl_b32 s50, 1, s19 // (op 1374) 
s_add_i32 s51, s50, s49 // (op 1374) 


s_and_b32 s49, s0, s51 // (op 1374) 


s_add_i32 s0, s18, s49 // (op 1374) 


s_ashr_i32 s18, s0, s19 // (op 1374) 


s_lshl_b32 s0, s14, 24 // (op 1374) 
s_ashr_i32 s19, s0, 31 // (op 1374) 

s_xor_b32 s0, s18, s19 // (op 1374) 

s_sub_i32 s18, s0, s19 // (op 1374) 


s_add_u32 s50, s6, 64 // (op 1374) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1374) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1374) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1374) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1374) 

s_mul_i32 s0, s18, s50 // (op 1374) 

s_sub_i32 s49, s1, s0 // (op 1374) 


s_mul_hi_i32 s0, s16, s10 // (op 1374) 
s_add_i32 s1, s0, s16 // (op 1374) 

s_ashr_i32 s0, s1, 31 // (op 1374) 
s_cmp_eq_i32 s10, 0 // (op 1374) 
s_mov_b32 s18, scc // (op 1374) 
s_cmp_eq_i32 s18, 0 // (op 1374) 
s_cselect_b32 s19, 0, -1 // (op 1374) 

s_and_b32 s18, s47, 31 // (op 1374) 
s_lshl_b32 s72, 1, s18 // (op 1374) 
s_add_i32 s73, s72, s19 // (op 1374) 


s_and_b32 s19, s0, s73 // (op 1374) 


s_add_i32 s0, s1, s19 // (op 1374) 


s_ashr_i32 s1, s0, s18 // (op 1374) 


s_lshl_b32 s0, s47, 24 // (op 1374) 
s_ashr_i32 s18, s0, 31 // (op 1374) 

s_xor_b32 s0, s1, s18 // (op 1374) 

s_sub_i32 s1, s0, s18 // (op 1374) 


s_mul_i32 s0, s44, s50 // (op 1374) 

s_mul_i32 s18, s48, s0 // (op 1374) 

s_mul_i32 s0, s1, s18 // (op 1374) 

s_sub_u32 s18, s16, s0 // (op 1374) 

s_mul_hi_i32 s0, s18, s15 // (op 1374) 
s_add_i32 s19, s0, s18 // (op 1374) 


s_ashr_i32 s0, s19, 31 // (op 1374) 
s_cmp_eq_i32 s15, 0 // (op 1374) 
s_mov_b32 s18, scc // (op 1374) 
s_cmp_eq_i32 s18, 0 // (op 1374) 
s_cselect_b32 s50, 0, -1 // (op 1374) 

s_and_b32 s18, s46, 31 // (op 1374) 
s_lshl_b32 s51, 1, s18 // (op 1374) 
s_add_i32 s72, s51, s50 // (op 1374) 


s_and_b32 s50, s0, s72 // (op 1374) 


s_add_i32 s0, s19, s50 // (op 1374) 


s_ashr_i32 s19, s0, s18 // (op 1374) 


s_lshl_b32 s0, s46, 24 // (op 1374) 
s_ashr_i32 s18, s0, 31 // (op 1374) 

s_xor_b32 s0, s19, s18 // (op 1374) 

s_sub_i32 s19, s0, s18 // (op 1374) 


s_cmp_eq_i32 s1, 0 // (op 1374) 
s_mov_b32 s0, scc // (op 1374) 

s_cmp_eq_i32 s0, 0 // (op 1374) 
s_cselect_b32 s1, s19, s49 // (op 1374) 



v_lshl_add_u32 v75, s1, 6, v19 // (op 1374) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 1374) 
s_lshr_b32 s18, s0, 1 // (op 1374) 

v_lshl_add_u32 v19, s18, 4, v67 // (op 1374) 


v_mul_lo_u32 v67, v19, s38 // (op 1374) 
v_add_u32 v67, v67, v75 // (op 1374) 


s_mul_i32 s1, s38, s43 // (op 1376) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1376) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1376) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1376) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1376) most significant: sum
s_add_u32 s1, s1, s19 // (op 1376) most significant: sum


s_mov_b32 s68, s40 // (op 1376) reg expression
s_mov_b32 s69, s41 // (op 1376) reg expression
s_mov_b32 s70, s0 // (op 1376) reg expression
s_mov_b32 s71, 131072 // (op 1376) call()

v_and_b32 v19, -8, v1 // (op 1467) 
v_sub_u32 v75, v1, v19 // (op 1467) 

v_lshlrev_b32 v19, 5, v75 // (op 1467) 

v_lshrrev_b32 v75, 3, v1 // (op 1467) 
v_lshl_add_u32 v79, v75, 8, v19 // (op 1467) 


v_mul_lo_u32 v19, v79, 6 // (op 1467) 

v_lshrrev_b32 v75, 3, v19 // (op 1467) 

v_lshlrev_b32 v19, 7, v1 // (op 1520) 
v_mul_lo_u32 v79, v19, 6 // (op 1520) 
v_lshrrev_b32 v83, 3, v79 // (op 1520) 

v_lshrrev_b32 v79, 4, v19 // (op 1520) 

v_lshlrev_b32 v19, 2, v79 // (op 1520) 

v_add_u32 v79, v83, v19 // (op 1520) 


v_and_b32 v19, -8, v1 // (op 1578) 
v_sub_u32 v83, v1, v19 // (op 1578) 

v_and_b32 v19, 4294967288, v1 // (op 1578) 
v_add_u32 v87, v19, v83 // (op 1578) 


v_and_b32 v19, -16, v1 // (op 1631) 
v_sub_u32 v83, v1, v19 // (op 1631) 

v_lshlrev_b32 v19, 2, v83 // (op 1631) 

v_lshrrev_b32 v83, 4, v1 // (op 1631) 
v_lshl_add_u32 v91, v83, 6, v19 // (op 1631) 


s_ashr_i64 s[0:1], s[32:33], 8 // (op 405) 
s_and_b32 s18, s0, 4294967294 // (op 405) 

s_ashr_i64 s[0:1], s[32:33], 8 // (op 404) 
s_mov_b32 s19, s0 // (op 404) convert

s_cmp_lt_u32 s18, s19 // (op 404) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404:


s_add_u32 s18, s18, 1 // (op 406) 
v_mov_b32 v14, v15 // (op 198) 

buffer_load_dwordx4 v[18:21], v14, s[20:23], 0 offen // (op 198) Load value
buffer_load_dwordx2 v[22:23], v14, s[20:23], 0 offen offset:16 // (op 198) Load value

v_add_u32 v15, 192, v15 // (op 1168) 

s_barrier  // (op 1845) 
s_mul_i32 s1, s29, 6 // (op 221) most significant: low of high * low



s_mul_i32 s0, s28, 6 // (op 221) least significant
s_mul_hi_u32 s19, s28, 6 // (op 221) most significant: high of low * low
s_add_u32 s1, s1, s19 // (op 221) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 221) 


v_mov_b32 v14, v17 // (op 221) 
v_mov_b32 v16, v14 // (op 221) reg expression
buffer_load_dwordx3 v[26:28], v16, s[52:55], 0 offen // (op 221) Load value
s_waitcnt vmcnt(2)
ds_write_b128 v75, v[18:21] offset:16896 // (op 199) Store local data 
buffer_load_dwordx3 v[30:32], v16, s[52:55], 0 offen offset:12 // (op 221) Load value
buffer_load_dwordx3 v[34:36], v16, s[52:55], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx3 v[38:40], v16, s[52:55], 0 offen offset:36 // (op 221) Load value
v_add_u32 v16, v16, v24 // (op 221) 
s_waitcnt vmcnt(4)
ds_write_b64 v75, v[22:23] offset:16912 // (op 199) Store local data 
buffer_load_dwordx3 v[42:44], v16, s[52:55], 0 offen // (op 221) Load value

buffer_load_dwordx3 v[46:48], v16, s[52:55], 0 offen offset:12 // (op 221) Load value
buffer_load_dwordx3 v[50:52], v16, s[52:55], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx3 v[54:56], v16, s[52:55], 0 offen offset:36 // (op 221) Load value



s_lshl_b64 s[0:1], s[28:29], 8 // (op 1238) 
s_mul_i32 s51, s1, 6 // (op 1238) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 1238) least significant
s_mul_hi_u32 s19, s0, 6 // (op 1238) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 1238) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 1238) 

v_add_u32 v17, v17, s0 // (op 1238) 



s_barrier  // (op 1864) 
v_mov_b32 v14, v71 // (op 222) 




buffer_load_ubyte v16, v14, s[64:67], 0 offen offset: 0 // (op 222) Load value

v_add_u32 v71, 8, v71 // (op 1302) 
s_waitcnt vmcnt(8)
ds_write_b96 v79, v[26:28] // (op 218) Store local data 


v_mov_b32 v14, v67 // (op 223) 
v_mov_b32 v29, v14 // (op 223) reg expression

buffer_load_dword v33, v29, s[68:71], 0 offen // (op 223) Load value


s_lshl_b32 s19, s38, 3 // (op 1366) 
v_add_u32 v67, v67, s19 // (op 1366) 

s_waitcnt vmcnt(8)
ds_write_b96 v79, v[30:32] offset:16 // (op 218) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1854) 
s_waitcnt vmcnt(7)
ds_write_b96 v79, v[34:36] offset:32 // (op 218) Store local data 



ds_read_b128 v[18:21], v3 offset:16896 // (op 201) Load local data 
ds_read_b64 v[22:23], v3 offset:16912 // (op 201) Load local data 
s_waitcnt vmcnt(6)
ds_write_b96 v79, v[38:40] offset:48 // (op 218) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v79, v[42:44] offset:64 // (op 218) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v79, v[46:48] offset:80 // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v79, v[50:52] offset:96 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v79, v[54:56] offset:112 // (op 218) Store local data 

s_barrier  // (op 1885) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1875) 






ds_read_b96_tr_b6 v[24:26], v6 // (op 203) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v6 offset:1024 // (op 203) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v87, v16 offset:19968 // (op 219) Store local data 

s_barrier  // (op 1905) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1895) 






ds_read_u8 v14, v0 offset:19968 // (op 205) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v91, v33 offset:16384 // (op 220) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1915) 


ds_read_u8 v16, v9 offset:16384 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:29], a[0:3], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 208) 





ds_read_b128 v[24:27], v3 offset:16992 // (op 210) Load local data 
ds_read_b64 v[28:29], v3 offset:17008 // (op 210) Load local data 



ds_read_b96_tr_b6 v[30:32], v6 offset:8192 // (op 212) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v6 offset:9216 // (op 212) Transpose load local data 



ds_read_u8 v16, v0 offset:19972 // (op 214) Load local data 


ds_read_u8 v36, v9 offset:16640 // (op 216) Load local data 



ds_read_b96_tr_b6 v[80:82], v6 offset:32 // (op 412) Transpose load local data 
ds_read_b96_tr_b6 v[83:85], v6 offset:1056 // (op 412) Transpose load local data 


ds_read_u8 v86, v9 offset:16416 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:23], v[80:85], a[4:7], v14, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 417) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], a[0:3], v16, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 217) 
ds_read_b96_tr_b6 v[18:20], v6 offset:8224 // (op 421) Transpose load local data 


ds_read_b96_tr_b6 v[21:23], v6 offset:9248 // (op 421) Transpose load local data 


ds_read_u8 v14, v9 offset:16672 // (op 425) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:29], v[18:23], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 426) 




s_ashr_i64 s[0:1], s[32:33], 8 // (op 404) 
s_mov_b32 s19, s0 // (op 404) convert

s_cmp_lt_u32 s18, s19 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopTop_KLoopTail_404 if true)
RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ForLoopBottom_KLoopTail_404:























s_barrier  // (op 2012) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155
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
s_branch RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalFalse_0_beta_155:


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
RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__ConditionalBottom_0_beta_155:










s_endpgm  // End of RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.LRR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end:
.size RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_, .LRR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM__end-RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
  .amdhsa_next_free_vgpr 148
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 40192 // lds bytes
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F6
    DestDataType: s
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
  - .name: RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 40192
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 138
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

