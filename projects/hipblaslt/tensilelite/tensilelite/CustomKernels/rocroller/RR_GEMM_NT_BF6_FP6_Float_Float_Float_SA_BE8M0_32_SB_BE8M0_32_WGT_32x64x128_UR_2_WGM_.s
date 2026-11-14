// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_,@function
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s12 // (op 1127) 
s_mul_hi_i32 s1, s16, s12 // (op 1316) 
v_and_b32 v0, 63, v1 // (op 1184) 
v_and_b32 v3, 48, v1 // (op 1184) 
v_sub_u32 v4, v0, v3 // (op 1184) 

v_lshrrev_b32 v3, 4, v0 // (op 1184) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1184) 


v_lshrrev_b32 v3, 7, v1 // (op 1184) 
v_lshlrev_b32 v4, 4, v3 // (op 1184) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 1184) 


v_mul_lo_u32 v0, v3, 6 // (op 1184) 
v_lshrrev_b32 v4, 3, v0 // (op 1184) 

v_lshrrev_b32 v0, 4, v3 // (op 1184) 

v_lshlrev_b32 v3, 2, v0 // (op 1184) 

v_add_u32 v0, v4, v3 // (op 1184) 


v_and_b32 v3, 63, v1 // (op 1342) 
v_and_b32 v4, 48, v1 // (op 1342) 
v_sub_u32 v5, v3, v4 // (op 1342) 

v_lshrrev_b32 v4, 4, v3 // (op 1342) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 1342) 


v_lshrrev_b32 v4, 7, v1 // (op 1342) 
v_lshlrev_b32 v5, 4, v4 // (op 1342) 

v_lshl_add_u32 v4, v3, 5, v5 // (op 1342) 


v_mul_lo_u32 v3, v4, 6 // (op 1342) 
v_lshrrev_b32 v5, 3, v3 // (op 1342) 

v_lshrrev_b32 v3, 4, v4 // (op 1342) 

v_lshlrev_b32 v4, 2, v3 // (op 1342) 

v_add_u32 v3, v5, v4 // (op 1342) 


v_lshrrev_b32 v4, 7, v1 // (op 1127) 
s_add_i32 s18, s0, s16 // (op 1127) 

v_and_b32 v5, 63, v1 // (op 1127) 
s_ashr_i32 s0, s18, 31 // (op 1127) 
v_lshrrev_b32 v6, 4, v5 // (op 1127) 
s_cmp_eq_i32 s12, 0 // (op 1127) 
s_mov_b32 s19, scc // (op 1127) 
v_lshlrev_b32 v7, 2, v6 // (op 1127) 

s_cmp_eq_i32 s19, 0 // (op 1127) 
s_cselect_b32 s51, 0, -1 // (op 1127) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1127) 


s_and_b32 s19, s15, 31 // (op 1127) 
v_lshrrev_b32 v4, 2, v5 // (op 1127) 

v_and_b32 v5, 3, v4 // (op 1127) 

v_and_b32 v4, 63, v1 // (op 1242) 
v_and_b32 v7, 48, v1 // (op 1242) 
v_sub_u32 v8, v4, v7 // (op 1242) 

v_lshrrev_b32 v7, 4, v4 // (op 1242) 

v_lshl_add_u32 v4, v7, 5, v8 // (op 1242) 


v_lshrrev_b32 v7, 6, v1 // (op 1242) 
v_and_b32 v8, 1, v7 // (op 1242) 

v_lshlrev_b32 v7, 4, v8 // (op 1242) 

v_lshl_add_u32 v8, v4, 6, v7 // (op 1242) 


v_mul_lo_u32 v4, v8, 6 // (op 1242) 
v_lshrrev_b32 v7, 3, v4 // (op 1242) 

v_lshrrev_b32 v4, 4, v8 // (op 1242) 

v_lshlrev_b32 v8, 2, v4 // (op 1242) 

v_add_u32 v4, v7, v8 // (op 1242) 


v_lshrrev_b32 v7, 6, v1 // (op 1300) 
v_and_b32 v8, 1, v7 // (op 1300) 

v_and_b32 v7, 15, v1 // (op 1300) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 1300) 


v_and_b32 v7, 63, v1 // (op 1300) 
v_lshrrev_b32 v8, 4, v7 // (op 1300) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 1300) 


v_lshrrev_b32 v8, 7, v1 // (op 1316) 
v_and_b32 v9, 63, v1 // (op 1316) 
v_and_b32 v10, 63, v1 // (op 1371) 
v_and_b32 v11, 48, v1 // (op 1371) 
v_sub_u32 v12, v10, v11 // (op 1371) 

v_lshrrev_b32 v11, 4, v10 // (op 1371) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 1371) 


v_lshrrev_b32 v11, 6, v1 // (op 1371) 
v_and_b32 v12, 1, v11 // (op 1371) 

v_lshlrev_b32 v11, 4, v12 // (op 1371) 

v_lshl_add_u32 v12, v10, 6, v11 // (op 1371) 


v_mul_lo_u32 v10, v12, 6 // (op 1371) 
v_lshrrev_b32 v11, 3, v10 // (op 1371) 

v_lshrrev_b32 v10, 4, v12 // (op 1371) 

v_lshlrev_b32 v12, 2, v10 // (op 1371) 

v_add_u32 v10, v11, v12 // (op 1371) 


v_lshrrev_b32 v11, 6, v1 // (op 1400) 
v_and_b32 v12, 1, v11 // (op 1400) 

v_and_b32 v11, 15, v1 // (op 1400) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 1400) 


v_and_b32 v11, 63, v1 // (op 1400) 
v_lshrrev_b32 v12, 4, v11 // (op 1400) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 1400) 


s_lshl_b32 s52, 1, s19 // (op 1127) 
v_and_b32 v12, 3, v1 // (op 1127) 
s_add_i32 s53, s52, s51 // (op 1127) 


s_and_b32 s51, s0, s53 // (op 1127) 


s_add_i32 s0, s1, s16 // (op 1316) 

s_ashr_i32 s1, s0, 31 // (op 1316) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1127) 


s_add_i32 s52, s18, s51 // (op 1127) 


v_lshrrev_b32 v5, 6, v1 // (op 1127) 
s_ashr_i32 s18, s52, s19 // (op 1127) 


v_and_b32 v12, 1, v5 // (op 1127) 

s_lshl_b32 s19, s15, 24 // (op 1127) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1127) 


s_ashr_i32 s51, s19, 31 // (op 1127) 

s_xor_b32 s19, s18, s51 // (op 1127) 

s_sub_i32 s18, s19, s51 // (op 1127) 


v_lshrrev_b32 v12, 4, v9 // (op 1316) 
s_cmp_eq_i32 s12, 0 // (op 1316) 
s_mov_b32 s19, scc // (op 1316) 
v_lshlrev_b32 v13, 2, v12 // (op 1316) 

s_cmp_eq_i32 s19, 0 // (op 1316) 
s_cselect_b32 s51, 0, -1 // (op 1316) 

v_lshl_add_u32 v12, v8, 4, v13 // (op 1316) 


s_and_b32 s19, s15, 31 // (op 1316) 
v_lshrrev_b32 v8, 2, v9 // (op 1316) 

s_lshl_b32 s52, 1, s19 // (op 1316) 
v_and_b32 v9, 3, v8 // (op 1316) 

s_add_i32 s53, s52, s51 // (op 1316) 


v_and_b32 v8, 3, v1 // (op 1316) 
s_and_b32 s51, s1, s53 // (op 1316) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1316) 


s_add_i32 s1, s0, s51 // (op 1316) 


v_lshrrev_b32 v8, 6, v1 // (op 1316) 
s_ashr_i32 s0, s1, s19 // (op 1316) 


v_and_b32 v9, 1, v8 // (op 1316) 

s_lshl_b32 s1, s15, 24 // (op 1316) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1316) 


s_ashr_i32 s19, s1, 31 // (op 1316) 

s_xor_b32 s1, s0, s19 // (op 1316) 

s_sub_i32 s0, s1, s19 // (op 1316) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s1, s18, s44 // (op 1127) 
s_add_i32 s19, s1, s18 // (op 1127) 

s_ashr_i32 s1, s19, 31 // (op 1127) 
s_cmp_eq_i32 s44, 0 // (op 1127) 
s_mov_b32 s51, scc // (op 1127) 
s_cmp_eq_i32 s51, 0 // (op 1127) 
s_cselect_b32 s52, 0, -1 // (op 1127) 

s_and_b32 s51, s45, 31 // (op 1127) 
s_lshl_b32 s53, 1, s51 // (op 1127) 
s_add_i32 s54, s53, s52 // (op 1127) 


s_and_b32 s52, s1, s54 // (op 1127) 


s_add_i32 s1, s19, s52 // (op 1127) 


s_ashr_i32 s19, s1, s51 // (op 1127) 


s_lshl_b32 s1, s45, 24 // (op 1127) 
s_ashr_i32 s51, s1, 31 // (op 1127) 

s_xor_b32 s1, s19, s51 // (op 1127) 

s_sub_i32 s19, s1, s51 // (op 1127) 


s_mul_i32 s1, s18, s13 // (op 1127) 
s_sub_u32 s51, s16, s1 // (op 1127) 

s_mul_i32 s1, s19, s13 // (op 1127) 
s_add_u32 s1, s1, s51 // (op 1127) 

s_mul_hi_i32 s51, s16, s11 // (op 1127) 
s_add_i32 s52, s51, s16 // (op 1127) 

s_ashr_i32 s51, s52, 31 // (op 1127) 
s_cmp_eq_i32 s11, 0 // (op 1127) 
s_mov_b32 s53, scc // (op 1127) 
s_cmp_eq_i32 s53, 0 // (op 1127) 
s_cselect_b32 s54, 0, -1 // (op 1127) 

s_and_b32 s53, s47, 31 // (op 1127) 
s_lshl_b32 s55, 1, s53 // (op 1127) 
s_add_i32 s56, s55, s54 // (op 1127) 


s_and_b32 s54, s51, s56 // (op 1127) 


s_add_i32 s51, s52, s54 // (op 1127) 


s_ashr_i32 s52, s51, s53 // (op 1127) 


s_lshl_b32 s51, s47, 24 // (op 1127) 
s_ashr_i32 s53, s51, 31 // (op 1127) 

s_xor_b32 s51, s52, s53 // (op 1127) 

s_sub_i32 s52, s51, s53 // (op 1127) 


s_mul_hi_i32 s51, s0, s44 // (op 1316) 
s_add_i32 s53, s51, s0 // (op 1316) 

s_ashr_i32 s51, s53, 31 // (op 1316) 
s_cmp_eq_i32 s44, 0 // (op 1316) 
s_mov_b32 s54, scc // (op 1316) 
s_cmp_eq_i32 s54, 0 // (op 1316) 
s_cselect_b32 s55, 0, -1 // (op 1316) 

s_and_b32 s54, s45, 31 // (op 1316) 
s_lshl_b32 s56, 1, s54 // (op 1316) 
s_add_i32 s57, s56, s55 // (op 1316) 


s_and_b32 s55, s51, s57 // (op 1316) 


s_add_i32 s51, s53, s55 // (op 1316) 


s_ashr_i32 s53, s51, s54 // (op 1316) 


s_lshl_b32 s51, s45, 24 // (op 1316) 
s_ashr_i32 s54, s51, 31 // (op 1316) 

s_xor_b32 s51, s53, s54 // (op 1316) 

s_sub_i32 s53, s51, s54 // (op 1316) 


s_mul_i32 s51, s0, s13 // (op 1316) 
s_sub_u32 s54, s16, s51 // (op 1316) 

s_mul_i32 s51, s53, s13 // (op 1316) 
s_add_u32 s51, s51, s54 // (op 1316) 

s_mul_hi_i32 s54, s16, s11 // (op 1316) 
s_add_i32 s55, s54, s16 // (op 1316) 

s_ashr_i32 s54, s55, 31 // (op 1316) 
s_cmp_eq_i32 s11, 0 // (op 1316) 
s_mov_b32 s56, scc // (op 1316) 
s_cmp_eq_i32 s56, 0 // (op 1316) 
s_cselect_b32 s57, 0, -1 // (op 1316) 

s_and_b32 s56, s47, 31 // (op 1316) 
s_lshl_b32 s58, 1, s56 // (op 1316) 
s_add_i32 s59, s58, s57 // (op 1316) 


s_and_b32 s57, s54, s59 // (op 1316) 


s_add_i32 s54, s55, s57 // (op 1316) 


s_ashr_i32 s55, s54, s56 // (op 1316) 


s_lshl_b32 s54, s47, 24 // (op 1316) 
s_ashr_i32 s56, s54, 31 // (op 1316) 

s_xor_b32 s54, s55, s56 // (op 1316) 

s_sub_i32 s55, s54, s56 // (op 1316) 


s_add_u32 s56, s8, 64 // (op 1127) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 1127) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1127) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1127) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1127) 

s_mul_i32 s54, s13, s56 // (op 1127) 
s_mul_i32 s58, s10, s54 // (op 1127) 

s_mul_i32 s54, s52, s58 // (op 1127) 

s_sub_u32 s58, s16, s54 // (op 1127) 

s_mov_b32 s54, s58 // (op 1127) 
s_mul_hi_i32 s59, s54, s14 // (op 1127) 
s_add_i32 s60, s59, s54 // (op 1127) 


s_ashr_i32 s54, s60, 31 // (op 1127) 
s_cmp_eq_i32 s14, 0 // (op 1127) 
s_mov_b32 s59, scc // (op 1127) 
s_cmp_eq_i32 s59, 0 // (op 1127) 
s_cselect_b32 s61, 0, -1 // (op 1127) 

s_and_b32 s59, s46, 31 // (op 1127) 
s_lshl_b32 s62, 1, s59 // (op 1127) 
s_add_i32 s63, s62, s61 // (op 1127) 


s_and_b32 s61, s54, s63 // (op 1127) 


s_add_i32 s54, s60, s61 // (op 1127) 


s_ashr_i32 s60, s54, s59 // (op 1127) 


s_lshl_b32 s54, s46, 24 // (op 1127) 
s_ashr_i32 s59, s54, 31 // (op 1127) 

s_xor_b32 s54, s60, s59 // (op 1127) 

s_sub_i32 s60, s54, s59 // (op 1127) 


s_mul_i32 s54, s48, s13 // (op 1127) 
s_add_u32 s62, s6, 32 // (op 1127) Start of Int64 add, locking SCC
s_addc_u32 s63, s7, 0 // (op 1127) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1127) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1127) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 1127) 

s_sub_i32 s59, s62, s54 // (op 1127) 

s_mul_i32 s61, s60, s59 // (op 1127) 

s_sub_u32 s59, s58, s61 // (op 1127) 


s_add_u32 s58, s59, s54 // (op 1127) 


s_cmp_eq_i32 s52, 0 // (op 1127) 
s_mov_b32 s54, scc // (op 1127) 

s_cmp_eq_i32 s54, 0 // (op 1127) 
s_cselect_b32 s52, s58, s1 // (op 1127) 


s_mul_i32 s1, s19, s56 // (op 1127) 


v_lshl_add_u32 v9, s52, 5, v6 // (op 1127) 


s_sub_i32 s19, s18, s1 // (op 1127) 


s_cmp_eq_i32 s54, 0 // (op 1127) 
s_cselect_b32 s1, s60, s19 // (op 1127) 



v_lshl_add_u32 v6, s1, 6, v5 // (op 1127) 


v_mul_lo_u32 v5, v6, s2 // (op 1127) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 1127) 


s_add_u32 s18, s8, 64 // (op 1316) Start of Int64 add, locking SCC
s_addc_u32 s19, s9, 0 // (op 1316) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 1316) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 1316) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 6 // (op 1316) 

s_mul_i32 s1, s13, s18 // (op 1316) 
s_mul_i32 s52, s10, s1 // (op 1316) 

s_mul_i32 s1, s55, s52 // (op 1316) 

s_sub_u32 s52, s16, s1 // (op 1316) 

s_mov_b32 s1, s52 // (op 1316) 
s_mul_hi_i32 s54, s1, s14 // (op 1316) 
s_add_i32 s56, s54, s1 // (op 1316) 


s_ashr_i32 s1, s56, 31 // (op 1316) 
s_cmp_eq_i32 s14, 0 // (op 1316) 
s_mov_b32 s54, scc // (op 1316) 
s_cmp_eq_i32 s54, 0 // (op 1316) 
s_cselect_b32 s57, 0, -1 // (op 1316) 

s_and_b32 s54, s46, 31 // (op 1316) 
s_lshl_b32 s58, 1, s54 // (op 1316) 
s_add_i32 s59, s58, s57 // (op 1316) 


s_and_b32 s57, s1, s59 // (op 1316) 


s_add_i32 s1, s56, s57 // (op 1316) 


s_ashr_i32 s56, s1, s54 // (op 1316) 


s_lshl_b32 s1, s46, 24 // (op 1316) 
s_ashr_i32 s54, s1, 31 // (op 1316) 

s_xor_b32 s1, s56, s54 // (op 1316) 

s_sub_i32 s56, s1, s54 // (op 1316) 


s_mul_i32 s1, s48, s13 // (op 1316) 
s_add_u32 s58, s6, 32 // (op 1316) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1316) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1316) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1316) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 5 // (op 1316) 

s_sub_i32 s54, s58, s1 // (op 1316) 

s_mul_i32 s57, s56, s54 // (op 1316) 

s_sub_u32 s54, s52, s57 // (op 1316) 


s_add_u32 s52, s54, s1 // (op 1316) 


s_cmp_eq_i32 s55, 0 // (op 1316) 
s_mov_b32 s1, scc // (op 1316) 

s_cmp_eq_i32 s1, 0 // (op 1316) 
s_cselect_b32 s54, s52, s51 // (op 1316) 


s_mul_i32 s55, s53, s18 // (op 1316) 


v_lshl_add_u32 v5, s54, 5, v12 // (op 1316) 


s_sub_i32 s18, s0, s55 // (op 1316) 


s_cmp_eq_i32 s1, 0 // (op 1316) 
s_cselect_b32 s0, s56, s18 // (op 1316) 



v_lshl_add_u32 v9, s0, 6, v8 // (op 1316) 


v_mul_lo_u32 v8, v9, s4 // (op 1316) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 1316) 


s_mul_i32 s1, s2, s9 // (op 1129) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1129) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1129) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1129) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1129) most significant: sum
s_add_u32 s1, s1, s19 // (op 1129) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1129) 

s_mul_i32 s1, s4, s9 // (op 1318) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 1318) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1318) least significant
s_mul_hi_u32 s52, s4, s8 // (op 1318) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1318) most significant: sum
s_add_u32 s1, s1, s52 // (op 1318) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1318) 

s_mov_b32 s56, s20 // (op 1129) reg expression
s_mov_b32 s57, s21 // (op 1129) reg expression
s_mov_b32 s58, s18 // (op 1129) reg expression
s_mov_b32 s59, 131072 // (op 1129) call()

s_mov_b32 s60, s22 // (op 1318) reg expression

s_mov_b32 s61, s23 // (op 1318) reg expression
s_mov_b32 s62, s52 // (op 1318) reg expression
s_mov_b32 s63, 131072 // (op 1318) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 306) call()
v_accvgpr_write a5, 0.00000 // (op 306) call()
v_accvgpr_write a6, 0.00000 // (op 306) call()
v_accvgpr_write a7, 0.00000 // (op 306) call()
s_mul_hi_i32 s0, s16, s12 // (op 894) 
v_and_b32 v12, -2, v1 // (op 894) 
s_add_i32 s1, s0, s16 // (op 894) 

v_sub_u32 v13, v1, v12 // (op 894) 

s_ashr_i32 s0, s1, 31 // (op 894) 
v_lshlrev_b32 v12, 4, v13 // (op 894) 

s_cmp_eq_i32 s12, 0 // (op 894) 
s_mov_b32 s18, scc // (op 894) 
v_lshrrev_b32 v13, 1, v1 // (op 894) 
s_cmp_eq_i32 s18, 0 // (op 894) 
s_cselect_b32 s19, 0, -1 // (op 894) 

s_and_b32 s18, s15, 31 // (op 894) 
s_lshl_b32 s20, 1, s18 // (op 894) 
s_add_i32 s21, s20, s19 // (op 894) 


s_and_b32 s19, s0, s21 // (op 894) 


s_add_i32 s0, s1, s19 // (op 894) 


s_ashr_i32 s1, s0, s18 // (op 894) 


s_lshl_b32 s0, s15, 24 // (op 894) 
s_ashr_i32 s18, s0, 31 // (op 894) 

s_xor_b32 s0, s1, s18 // (op 894) 

s_sub_i32 s1, s0, s18 // (op 894) 


s_mul_hi_i32 s0, s1, s44 // (op 894) 
s_add_i32 s18, s0, s1 // (op 894) 

s_ashr_i32 s0, s18, 31 // (op 894) 
s_cmp_eq_i32 s44, 0 // (op 894) 
s_mov_b32 s19, scc // (op 894) 
s_cmp_eq_i32 s19, 0 // (op 894) 
s_cselect_b32 s20, 0, -1 // (op 894) 

s_and_b32 s19, s45, 31 // (op 894) 
s_lshl_b32 s21, 1, s19 // (op 894) 
s_add_i32 s22, s21, s20 // (op 894) 


s_and_b32 s23, s0, s22 // (op 894) 


s_add_i32 s0, s18, s23 // (op 894) 


s_ashr_i32 s18, s0, s19 // (op 894) 


s_lshl_b32 s0, s45, 24 // (op 894) 
s_ashr_i32 s19, s0, 31 // (op 894) 

s_xor_b32 s0, s18, s19 // (op 894) 

s_sub_i32 s18, s0, s19 // (op 894) 


s_mul_i32 s0, s1, s13 // (op 894) 

s_sub_u32 s1, s16, s0 // (op 894) 

s_mul_i32 s0, s18, s13 // (op 894) 
s_add_u32 s0, s0, s1 // (op 894) 


s_mul_hi_i32 s1, s16, s11 // (op 894) 
s_add_i32 s18, s1, s16 // (op 894) 

s_ashr_i32 s1, s18, 31 // (op 894) 
s_cmp_eq_i32 s11, 0 // (op 894) 
s_mov_b32 s19, scc // (op 894) 
s_cmp_eq_i32 s19, 0 // (op 894) 
s_cselect_b32 s20, 0, -1 // (op 894) 

s_and_b32 s19, s47, 31 // (op 894) 
s_lshl_b32 s21, 1, s19 // (op 894) 
s_add_i32 s22, s21, s20 // (op 894) 


s_and_b32 s23, s1, s22 // (op 894) 


s_add_i32 s1, s18, s23 // (op 894) 


s_ashr_i32 s18, s1, s19 // (op 894) 


s_lshl_b32 s1, s47, 24 // (op 894) 
s_ashr_i32 s19, s1, 31 // (op 894) 

s_xor_b32 s1, s18, s19 // (op 894) 

s_sub_i32 s18, s1, s19 // (op 894) 


s_add_u32 s22, s8, 64 // (op 894) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 894) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 894) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 894) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 894) 

s_mul_i32 s1, s13, s22 // (op 894) 

s_mul_i32 s19, s10, s1 // (op 894) 

s_mul_i32 s1, s18, s19 // (op 894) 

s_sub_u32 s19, s16, s1 // (op 894) 

s_mov_b32 s1, s19 // (op 894) 
s_mul_hi_i32 s20, s1, s14 // (op 894) 
s_add_i32 s21, s20, s1 // (op 894) 


s_ashr_i32 s1, s21, 31 // (op 894) 
s_cmp_eq_i32 s14, 0 // (op 894) 
s_mov_b32 s20, scc // (op 894) 
s_cmp_eq_i32 s20, 0 // (op 894) 
s_cselect_b32 s22, 0, -1 // (op 894) 

s_and_b32 s20, s46, 31 // (op 894) 
s_lshl_b32 s23, 1, s20 // (op 894) 
s_add_i32 s49, s23, s22 // (op 894) 


s_and_b32 s22, s1, s49 // (op 894) 


s_add_i32 s1, s21, s22 // (op 894) 


s_ashr_i32 s21, s1, s20 // (op 894) 


s_lshl_b32 s1, s46, 24 // (op 894) 
s_ashr_i32 s20, s1, 31 // (op 894) 

s_xor_b32 s1, s21, s20 // (op 894) 

s_sub_i32 s21, s1, s20 // (op 894) 


s_mul_i32 s1, s48, s13 // (op 894) 
s_add_u32 s22, s6, 32 // (op 894) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 894) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 894) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 894) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 5 // (op 894) 

s_sub_i32 s20, s22, s1 // (op 894) 

s_mul_i32 s22, s21, s20 // (op 894) 


s_sub_u32 s23, s19, s22 // (op 894) 


s_add_u32 s19, s23, s1 // (op 894) 


s_cmp_eq_i32 s18, 0 // (op 894) 
s_mov_b32 s1, scc // (op 894) 

s_cmp_eq_i32 s1, 0 // (op 894) 
s_cselect_b32 s18, s19, s0 // (op 894) 



v_lshl_add_u32 v14, s18, 5, v12 // (op 894) 


v_mov_b32 v17, 0 // (op 894) convert
v_mov_b32 v16, v14 // (op 894) convert

v_mul_lo_u32 v15, v13, s25 // (op 894) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 894) most significant: high of low * low
v_mul_lo_u32 v14, v13, s24 // (op 894) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 894) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 894) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 894) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 894) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 894) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 894) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 894) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 894) 

v_mov_b32 v14, v12 // (op 894) convert

s_mul_i32 s1, s29, 6 // (op 896) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 896) least significant
s_mul_hi_u32 s18, s28, 6 // (op 896) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 896) most significant: sum

s_mul_i32 s19, s24, s1 // (op 896) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 896) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 896) least significant
s_mul_hi_u32 s21, s24, s0 // (op 896) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 896) most significant: sum
s_add_u32 s19, s19, s21 // (op 896) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 896) 

s_mov_b32 s20, s26 // (op 896) reg expression
s_mov_b32 s21, s27 // (op 896) reg expression
s_mov_b32 s22, s0 // (op 896) reg expression
s_mov_b32 s23, 131072 // (op 896) call()

s_mul_hi_i32 s0, s16, s12 // (op 913) 
v_and_b32 v12, -2, v1 // (op 913) 
s_add_i32 s1, s0, s16 // (op 913) 

v_sub_u32 v13, v1, v12 // (op 913) 

s_ashr_i32 s0, s1, 31 // (op 913) 
v_lshlrev_b32 v12, 4, v13 // (op 913) 

s_cmp_eq_i32 s12, 0 // (op 913) 
s_mov_b32 s18, scc // (op 913) 
v_lshrrev_b32 v13, 1, v1 // (op 913) 
s_cmp_eq_i32 s18, 0 // (op 913) 
s_cselect_b32 s19, 0, -1 // (op 913) 

s_and_b32 s18, s15, 31 // (op 913) 
s_lshl_b32 s49, 1, s18 // (op 913) 
s_add_i32 s50, s49, s19 // (op 913) 


s_and_b32 s19, s0, s50 // (op 913) 


s_add_i32 s0, s1, s19 // (op 913) 


s_ashr_i32 s1, s0, s18 // (op 913) 


s_lshl_b32 s0, s15, 24 // (op 913) 
s_ashr_i32 s18, s0, 31 // (op 913) 

s_xor_b32 s0, s1, s18 // (op 913) 

s_sub_i32 s1, s0, s18 // (op 913) 


s_mul_hi_i32 s0, s1, s44 // (op 913) 
s_add_i32 s18, s0, s1 // (op 913) 

s_ashr_i32 s0, s18, 31 // (op 913) 
s_cmp_eq_i32 s44, 0 // (op 913) 
s_mov_b32 s19, scc // (op 913) 
s_cmp_eq_i32 s19, 0 // (op 913) 
s_cselect_b32 s49, 0, -1 // (op 913) 

s_and_b32 s19, s45, 31 // (op 913) 
s_lshl_b32 s50, 1, s19 // (op 913) 
s_add_i32 s51, s50, s49 // (op 913) 


s_and_b32 s49, s0, s51 // (op 913) 


s_add_i32 s0, s18, s49 // (op 913) 


s_ashr_i32 s18, s0, s19 // (op 913) 


s_lshl_b32 s0, s45, 24 // (op 913) 
s_ashr_i32 s19, s0, 31 // (op 913) 

s_xor_b32 s0, s18, s19 // (op 913) 

s_sub_i32 s18, s0, s19 // (op 913) 


s_mul_i32 s0, s1, s13 // (op 913) 

s_sub_u32 s1, s16, s0 // (op 913) 

s_mul_i32 s0, s18, s13 // (op 913) 
s_add_u32 s0, s0, s1 // (op 913) 


s_mul_hi_i32 s1, s16, s11 // (op 913) 
s_add_i32 s18, s1, s16 // (op 913) 

s_ashr_i32 s1, s18, 31 // (op 913) 
s_cmp_eq_i32 s11, 0 // (op 913) 
s_mov_b32 s19, scc // (op 913) 
s_cmp_eq_i32 s19, 0 // (op 913) 
s_cselect_b32 s49, 0, -1 // (op 913) 

s_and_b32 s19, s47, 31 // (op 913) 
s_lshl_b32 s50, 1, s19 // (op 913) 
s_add_i32 s51, s50, s49 // (op 913) 


s_and_b32 s49, s1, s51 // (op 913) 


s_add_i32 s1, s18, s49 // (op 913) 


s_ashr_i32 s18, s1, s19 // (op 913) 


s_lshl_b32 s1, s47, 24 // (op 913) 
s_ashr_i32 s19, s1, 31 // (op 913) 

s_xor_b32 s1, s18, s19 // (op 913) 

s_sub_i32 s18, s1, s19 // (op 913) 


s_add_u32 s54, s8, 64 // (op 913) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 913) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 913) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 913) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 913) 

s_mul_i32 s1, s13, s50 // (op 913) 

s_mul_i32 s19, s10, s1 // (op 913) 

s_mul_i32 s1, s18, s19 // (op 913) 

s_sub_u32 s19, s16, s1 // (op 913) 

s_mov_b32 s1, s19 // (op 913) 
s_mul_hi_i32 s49, s1, s14 // (op 913) 
s_add_i32 s50, s49, s1 // (op 913) 


s_ashr_i32 s1, s50, 31 // (op 913) 
s_cmp_eq_i32 s14, 0 // (op 913) 
s_mov_b32 s49, scc // (op 913) 
s_cmp_eq_i32 s49, 0 // (op 913) 
s_cselect_b32 s51, 0, -1 // (op 913) 

s_and_b32 s49, s46, 31 // (op 913) 
s_lshl_b32 s52, 1, s49 // (op 913) 
s_add_i32 s53, s52, s51 // (op 913) 


s_and_b32 s51, s1, s53 // (op 913) 


s_add_i32 s1, s50, s51 // (op 913) 


s_ashr_i32 s50, s1, s49 // (op 913) 


s_lshl_b32 s1, s46, 24 // (op 913) 
s_ashr_i32 s49, s1, 31 // (op 913) 

s_xor_b32 s1, s50, s49 // (op 913) 

s_sub_i32 s50, s1, s49 // (op 913) 


s_mul_i32 s1, s48, s13 // (op 913) 
s_add_u32 s54, s6, 32 // (op 913) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 913) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 913) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 913) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 5 // (op 913) 

s_sub_i32 s49, s54, s1 // (op 913) 

s_mul_i32 s51, s50, s49 // (op 913) 


s_sub_u32 s49, s19, s51 // (op 913) 


s_add_u32 s19, s49, s1 // (op 913) 


s_cmp_eq_i32 s18, 0 // (op 913) 
s_mov_b32 s1, scc // (op 913) 

s_cmp_eq_i32 s1, 0 // (op 913) 
s_cselect_b32 s18, s19, s0 // (op 913) 



v_lshl_add_u32 v15, s18, 5, v12 // (op 913) 


v_mov_b32 v17, 0 // (op 913) convert
v_mov_b32 v16, v15 // (op 913) convert

v_mul_lo_u32 v19, v13, s25 // (op 913) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 913) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 913) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 913) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 913) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 913) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 913) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 913) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 913) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 913) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 913) 

v_mov_b32 v15, v12 // (op 913) convert

s_mul_i32 s1, s29, 6 // (op 915) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 915) least significant
s_mul_hi_u32 s18, s28, 6 // (op 915) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 915) most significant: sum

s_mul_i32 s19, s24, s1 // (op 915) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 915) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 915) least significant
s_mul_hi_u32 s50, s24, s0 // (op 915) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 915) most significant: sum
s_add_u32 s19, s19, s50 // (op 915) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 915) 

s_mov_b32 s20, s26 // (op 915) reg expression
s_mov_b32 s21, s27 // (op 915) reg expression
s_mov_b32 s22, s0 // (op 915) reg expression
s_mov_b32 s23, 131072 // (op 915) call()

s_mul_hi_i32 s0, s16, s12 // (op 958) 
v_and_b32 v12, -2, v1 // (op 958) 
s_add_i32 s1, s0, s16 // (op 958) 

v_sub_u32 v13, v1, v12 // (op 958) 

s_ashr_i32 s0, s1, 31 // (op 958) 
v_lshlrev_b32 v12, 5, v13 // (op 958) 

s_cmp_eq_i32 s12, 0 // (op 958) 
s_mov_b32 s18, scc // (op 958) 
v_lshrrev_b32 v13, 1, v1 // (op 958) 
s_cmp_eq_i32 s18, 0 // (op 958) 
s_cselect_b32 s19, 0, -1 // (op 958) 

s_and_b32 s18, s15, 31 // (op 958) 
s_lshl_b32 s49, 1, s18 // (op 958) 
s_add_i32 s50, s49, s19 // (op 958) 


s_and_b32 s19, s0, s50 // (op 958) 


s_add_i32 s0, s1, s19 // (op 958) 


s_ashr_i32 s1, s0, s18 // (op 958) 


s_lshl_b32 s0, s15, 24 // (op 958) 
s_ashr_i32 s18, s0, 31 // (op 958) 

s_xor_b32 s0, s1, s18 // (op 958) 

s_sub_i32 s1, s0, s18 // (op 958) 


s_mul_hi_i32 s0, s1, s44 // (op 958) 
s_add_i32 s18, s0, s1 // (op 958) 

s_ashr_i32 s0, s18, 31 // (op 958) 
s_cmp_eq_i32 s44, 0 // (op 958) 
s_mov_b32 s19, scc // (op 958) 
s_cmp_eq_i32 s19, 0 // (op 958) 
s_cselect_b32 s49, 0, -1 // (op 958) 

s_and_b32 s19, s45, 31 // (op 958) 
s_lshl_b32 s50, 1, s19 // (op 958) 
s_add_i32 s51, s50, s49 // (op 958) 


s_and_b32 s49, s0, s51 // (op 958) 


s_add_i32 s0, s18, s49 // (op 958) 


s_ashr_i32 s18, s0, s19 // (op 958) 


s_lshl_b32 s0, s45, 24 // (op 958) 
s_ashr_i32 s19, s0, 31 // (op 958) 

s_xor_b32 s0, s18, s19 // (op 958) 

s_sub_i32 s18, s0, s19 // (op 958) 


s_add_u32 s54, s8, 64 // (op 958) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 958) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 958) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 958) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 958) 

s_mul_i32 s0, s18, s50 // (op 958) 

s_sub_i32 s49, s1, s0 // (op 958) 


s_mul_hi_i32 s0, s16, s11 // (op 958) 
s_add_i32 s1, s0, s16 // (op 958) 

s_ashr_i32 s0, s1, 31 // (op 958) 
s_cmp_eq_i32 s11, 0 // (op 958) 
s_mov_b32 s18, scc // (op 958) 
s_cmp_eq_i32 s18, 0 // (op 958) 
s_cselect_b32 s19, 0, -1 // (op 958) 

s_and_b32 s18, s47, 31 // (op 958) 
s_lshl_b32 s52, 1, s18 // (op 958) 
s_add_i32 s53, s52, s19 // (op 958) 


s_and_b32 s19, s0, s53 // (op 958) 


s_add_i32 s0, s1, s19 // (op 958) 


s_ashr_i32 s1, s0, s18 // (op 958) 


s_lshl_b32 s0, s47, 24 // (op 958) 
s_ashr_i32 s18, s0, 31 // (op 958) 

s_xor_b32 s0, s1, s18 // (op 958) 

s_sub_i32 s1, s0, s18 // (op 958) 


s_mul_i32 s0, s13, s50 // (op 958) 

s_mul_i32 s18, s10, s0 // (op 958) 

s_mul_i32 s0, s1, s18 // (op 958) 

s_sub_u32 s18, s16, s0 // (op 958) 

s_mul_hi_i32 s0, s18, s14 // (op 958) 
s_add_i32 s19, s0, s18 // (op 958) 


s_ashr_i32 s0, s19, 31 // (op 958) 
s_cmp_eq_i32 s14, 0 // (op 958) 
s_mov_b32 s18, scc // (op 958) 
s_cmp_eq_i32 s18, 0 // (op 958) 
s_cselect_b32 s50, 0, -1 // (op 958) 

s_and_b32 s18, s46, 31 // (op 958) 
s_lshl_b32 s51, 1, s18 // (op 958) 
s_add_i32 s52, s51, s50 // (op 958) 


s_and_b32 s50, s0, s52 // (op 958) 


s_add_i32 s0, s19, s50 // (op 958) 


s_ashr_i32 s19, s0, s18 // (op 958) 


s_lshl_b32 s0, s46, 24 // (op 958) 
s_ashr_i32 s18, s0, 31 // (op 958) 

s_xor_b32 s0, s19, s18 // (op 958) 

s_sub_i32 s19, s0, s18 // (op 958) 


s_cmp_eq_i32 s1, 0 // (op 958) 
s_mov_b32 s0, scc // (op 958) 

s_cmp_eq_i32 s0, 0 // (op 958) 
s_cselect_b32 s1, s19, s49 // (op 958) 



v_lshl_add_u32 v16, s1, 6, v12 // (op 958) 


v_mov_b32 v19, 0 // (op 958) convert
v_mov_b32 v18, v16 // (op 958) convert

v_mul_lo_u32 v17, v13, s31 // (op 958) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 958) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 958) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 958) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 958) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 958) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 958) most significant: low of high * low
v_mul_hi_u32 v18, v16, 6 // (op 958) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 958) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 958) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 958) 

v_mov_b32 v12, v16 // (op 958) convert

s_mul_i32 s1, s29, 6 // (op 960) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 960) least significant
s_mul_hi_u32 s18, s28, 6 // (op 960) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 960) most significant: sum

s_mul_i32 s19, s30, s1 // (op 960) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 960) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 960) least significant
s_mul_hi_u32 s50, s30, s0 // (op 960) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 960) most significant: sum
s_add_u32 s19, s19, s50 // (op 960) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 960) 

s_mov_b32 s52, s32 // (op 960) reg expression
s_mov_b32 s53, s33 // (op 960) reg expression
s_mov_b32 s54, s0 // (op 960) reg expression
s_mov_b32 s55, 131072 // (op 960) call()

s_mul_hi_i32 s0, s16, s12 // (op 977) 
v_and_b32 v13, -2, v1 // (op 977) 
s_add_i32 s1, s0, s16 // (op 977) 

v_sub_u32 v16, v1, v13 // (op 977) 

s_ashr_i32 s0, s1, 31 // (op 977) 
v_lshlrev_b32 v13, 5, v16 // (op 977) 

s_cmp_eq_i32 s12, 0 // (op 977) 
s_mov_b32 s18, scc // (op 977) 
v_lshrrev_b32 v16, 1, v1 // (op 977) 
s_cmp_eq_i32 s18, 0 // (op 977) 
s_cselect_b32 s19, 0, -1 // (op 977) 

s_and_b32 s18, s15, 31 // (op 977) 
s_lshl_b32 s49, 1, s18 // (op 977) 
s_add_i32 s50, s49, s19 // (op 977) 


s_and_b32 s19, s0, s50 // (op 977) 


s_add_i32 s0, s1, s19 // (op 977) 


s_ashr_i32 s1, s0, s18 // (op 977) 


s_lshl_b32 s0, s15, 24 // (op 977) 
s_ashr_i32 s18, s0, 31 // (op 977) 

s_xor_b32 s0, s1, s18 // (op 977) 

s_sub_i32 s1, s0, s18 // (op 977) 


s_mul_hi_i32 s0, s1, s44 // (op 977) 
s_add_i32 s18, s0, s1 // (op 977) 

s_ashr_i32 s0, s18, 31 // (op 977) 
s_cmp_eq_i32 s44, 0 // (op 977) 
s_mov_b32 s19, scc // (op 977) 
s_cmp_eq_i32 s19, 0 // (op 977) 
s_cselect_b32 s49, 0, -1 // (op 977) 

s_and_b32 s19, s45, 31 // (op 977) 
s_lshl_b32 s50, 1, s19 // (op 977) 
s_add_i32 s51, s50, s49 // (op 977) 


s_and_b32 s49, s0, s51 // (op 977) 


s_add_i32 s0, s18, s49 // (op 977) 


s_ashr_i32 s18, s0, s19 // (op 977) 


s_lshl_b32 s0, s45, 24 // (op 977) 
s_ashr_i32 s19, s0, 31 // (op 977) 

s_xor_b32 s0, s18, s19 // (op 977) 

s_sub_i32 s18, s0, s19 // (op 977) 


s_add_u32 s50, s8, 64 // (op 977) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 977) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 977) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 977) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 977) 

s_mul_i32 s0, s18, s50 // (op 977) 

s_sub_i32 s49, s1, s0 // (op 977) 


s_mul_hi_i32 s0, s16, s11 // (op 977) 
s_add_i32 s1, s0, s16 // (op 977) 

s_ashr_i32 s0, s1, 31 // (op 977) 
s_cmp_eq_i32 s11, 0 // (op 977) 
s_mov_b32 s18, scc // (op 977) 
s_cmp_eq_i32 s18, 0 // (op 977) 
s_cselect_b32 s19, 0, -1 // (op 977) 

s_and_b32 s18, s47, 31 // (op 977) 
s_lshl_b32 s64, 1, s18 // (op 977) 
s_add_i32 s65, s64, s19 // (op 977) 


s_and_b32 s19, s0, s65 // (op 977) 


s_add_i32 s0, s1, s19 // (op 977) 


s_ashr_i32 s1, s0, s18 // (op 977) 


s_lshl_b32 s0, s47, 24 // (op 977) 
s_ashr_i32 s18, s0, 31 // (op 977) 

s_xor_b32 s0, s1, s18 // (op 977) 

s_sub_i32 s1, s0, s18 // (op 977) 


s_mul_i32 s0, s13, s50 // (op 977) 

s_mul_i32 s18, s10, s0 // (op 977) 

s_mul_i32 s0, s1, s18 // (op 977) 

s_sub_u32 s18, s16, s0 // (op 977) 

s_mul_hi_i32 s0, s18, s14 // (op 977) 
s_add_i32 s19, s0, s18 // (op 977) 


s_ashr_i32 s0, s19, 31 // (op 977) 
s_cmp_eq_i32 s14, 0 // (op 977) 
s_mov_b32 s18, scc // (op 977) 
s_cmp_eq_i32 s18, 0 // (op 977) 
s_cselect_b32 s50, 0, -1 // (op 977) 

s_and_b32 s18, s46, 31 // (op 977) 
s_lshl_b32 s51, 1, s18 // (op 977) 
s_add_i32 s64, s51, s50 // (op 977) 


s_and_b32 s50, s0, s64 // (op 977) 


s_add_i32 s0, s19, s50 // (op 977) 


s_ashr_i32 s19, s0, s18 // (op 977) 


s_lshl_b32 s0, s46, 24 // (op 977) 
s_ashr_i32 s18, s0, 31 // (op 977) 

s_xor_b32 s0, s19, s18 // (op 977) 

s_sub_i32 s19, s0, s18 // (op 977) 


s_cmp_eq_i32 s1, 0 // (op 977) 
s_mov_b32 s0, scc // (op 977) 

s_cmp_eq_i32 s0, 0 // (op 977) 
s_cselect_b32 s1, s19, s49 // (op 977) 



v_lshl_add_u32 v17, s1, 6, v13 // (op 977) 


v_mov_b32 v19, 0 // (op 977) convert
v_mov_b32 v18, v17 // (op 977) convert

v_mul_lo_u32 v21, v16, s31 // (op 977) most significant: low of low * high
v_mul_hi_u32 v13, v16, s30 // (op 977) most significant: high of low * low
v_mul_lo_u32 v20, v16, s30 // (op 977) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 977) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 977) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 977) most significant half



v_mul_lo_u32 v17, v21, 6 // (op 977) most significant: low of high * low
v_mul_hi_u32 v13, v20, 6 // (op 977) most significant: high of low * low
v_mul_lo_u32 v16, v20, 6 // (op 977) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 977) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 977) 

v_mov_b32 v13, v18 // (op 977) convert

s_mul_i32 s1, s29, 6 // (op 979) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 979) least significant
s_mul_hi_u32 s18, s28, 6 // (op 979) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 979) most significant: sum

s_mul_i32 s19, s30, s1 // (op 979) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 979) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 979) least significant
s_mul_hi_u32 s50, s30, s0 // (op 979) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 979) most significant: sum
s_add_u32 s19, s19, s50 // (op 979) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 979) 

s_mov_b32 s52, s32 // (op 979) reg expression
s_mov_b32 s53, s33 // (op 979) reg expression
s_mov_b32 s54, s0 // (op 979) reg expression
s_mov_b32 s55, 131072 // (op 979) call()

s_mul_hi_i32 s0, s16, s12 // (op 1022) 
v_lshrrev_b32 v16, 7, v1 // (op 1022) 
s_add_i32 s1, s0, s16 // (op 1022) 

v_and_b32 v17, 15, v1 // (op 1022) 
s_ashr_i32 s0, s1, 31 // (op 1022) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 1022) 


s_cmp_eq_i32 s12, 0 // (op 1022) 
s_mov_b32 s18, scc // (op 1022) 
v_and_b32 v16, 63, v1 // (op 1022) 
s_cmp_eq_i32 s18, 0 // (op 1022) 
s_cselect_b32 s19, 0, -1 // (op 1022) 

v_lshrrev_b32 v17, 4, v16 // (op 1022) 

s_and_b32 s18, s15, 31 // (op 1022) 
s_lshl_b32 s49, 1, s18 // (op 1022) 
s_add_i32 s50, s49, s19 // (op 1022) 


s_and_b32 s19, s0, s50 // (op 1022) 


s_add_i32 s0, s1, s19 // (op 1022) 


s_ashr_i32 s1, s0, s18 // (op 1022) 


s_lshl_b32 s0, s15, 24 // (op 1022) 
s_ashr_i32 s18, s0, 31 // (op 1022) 

s_xor_b32 s0, s1, s18 // (op 1022) 

s_sub_i32 s1, s0, s18 // (op 1022) 


s_mul_hi_i32 s0, s1, s44 // (op 1022) 
s_add_i32 s18, s0, s1 // (op 1022) 

s_ashr_i32 s0, s18, 31 // (op 1022) 
s_cmp_eq_i32 s44, 0 // (op 1022) 
s_mov_b32 s19, scc // (op 1022) 
s_cmp_eq_i32 s19, 0 // (op 1022) 
s_cselect_b32 s49, 0, -1 // (op 1022) 

s_and_b32 s19, s45, 31 // (op 1022) 
s_lshl_b32 s50, 1, s19 // (op 1022) 
s_add_i32 s51, s50, s49 // (op 1022) 


s_and_b32 s49, s0, s51 // (op 1022) 


s_add_i32 s0, s18, s49 // (op 1022) 


s_ashr_i32 s18, s0, s19 // (op 1022) 


s_lshl_b32 s0, s45, 24 // (op 1022) 
s_ashr_i32 s19, s0, 31 // (op 1022) 

s_xor_b32 s0, s18, s19 // (op 1022) 

s_sub_i32 s18, s0, s19 // (op 1022) 


s_mul_i32 s0, s1, s13 // (op 1022) 

s_sub_u32 s1, s16, s0 // (op 1022) 

s_mul_i32 s0, s18, s13 // (op 1022) 
s_add_u32 s0, s0, s1 // (op 1022) 


s_mul_hi_i32 s1, s16, s11 // (op 1022) 
s_add_i32 s18, s1, s16 // (op 1022) 

s_ashr_i32 s1, s18, 31 // (op 1022) 
s_cmp_eq_i32 s11, 0 // (op 1022) 
s_mov_b32 s19, scc // (op 1022) 
s_cmp_eq_i32 s19, 0 // (op 1022) 
s_cselect_b32 s49, 0, -1 // (op 1022) 

s_and_b32 s19, s47, 31 // (op 1022) 
s_lshl_b32 s50, 1, s19 // (op 1022) 
s_add_i32 s51, s50, s49 // (op 1022) 


s_and_b32 s49, s1, s51 // (op 1022) 


s_add_i32 s1, s18, s49 // (op 1022) 


s_ashr_i32 s18, s1, s19 // (op 1022) 


s_lshl_b32 s1, s47, 24 // (op 1022) 
s_ashr_i32 s19, s1, 31 // (op 1022) 

s_xor_b32 s1, s18, s19 // (op 1022) 

s_sub_i32 s18, s1, s19 // (op 1022) 


s_add_u32 s50, s8, 64 // (op 1022) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1022) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1022) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1022) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1022) 

s_mul_i32 s1, s13, s50 // (op 1022) 

s_mul_i32 s19, s10, s1 // (op 1022) 

s_mul_i32 s1, s18, s19 // (op 1022) 

s_sub_u32 s19, s16, s1 // (op 1022) 

s_mov_b32 s1, s19 // (op 1022) 
s_mul_hi_i32 s49, s1, s14 // (op 1022) 
s_add_i32 s50, s49, s1 // (op 1022) 


s_ashr_i32 s1, s50, 31 // (op 1022) 
s_cmp_eq_i32 s14, 0 // (op 1022) 
s_mov_b32 s49, scc // (op 1022) 
s_cmp_eq_i32 s49, 0 // (op 1022) 
s_cselect_b32 s51, 0, -1 // (op 1022) 

s_and_b32 s49, s46, 31 // (op 1022) 
s_lshl_b32 s64, 1, s49 // (op 1022) 
s_add_i32 s65, s64, s51 // (op 1022) 


s_and_b32 s51, s1, s65 // (op 1022) 


s_add_i32 s1, s50, s51 // (op 1022) 


s_ashr_i32 s50, s1, s49 // (op 1022) 


s_lshl_b32 s1, s46, 24 // (op 1022) 
s_ashr_i32 s49, s1, 31 // (op 1022) 

s_xor_b32 s1, s50, s49 // (op 1022) 

s_sub_i32 s50, s1, s49 // (op 1022) 


s_mul_i32 s1, s48, s13 // (op 1022) 
s_add_u32 s64, s6, 32 // (op 1022) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 1022) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1022) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1022) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 5 // (op 1022) 

s_sub_i32 s49, s64, s1 // (op 1022) 

s_mul_i32 s51, s50, s49 // (op 1022) 


s_sub_u32 s49, s19, s51 // (op 1022) 


s_add_u32 s19, s49, s1 // (op 1022) 


s_cmp_eq_i32 s18, 0 // (op 1022) 
s_mov_b32 s1, scc // (op 1022) 

s_cmp_eq_i32 s1, 0 // (op 1022) 
s_cselect_b32 s18, s19, s0 // (op 1022) 



v_lshl_add_u32 v16, s18, 5, v18 // (op 1022) 


v_mul_lo_u32 v18, v17, s34 // (op 1022) 

v_add_u32 v17, v16, v18 // (op 1022) 


s_mul_i32 s1, s34, s39 // (op 1024) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1024) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1024) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1024) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1024) most significant: sum
s_add_u32 s1, s1, s19 // (op 1024) most significant: sum


s_mov_b32 s64, s36 // (op 1024) reg expression
s_mov_b32 s65, s37 // (op 1024) reg expression
s_mov_b32 s66, s0 // (op 1024) reg expression
s_mov_b32 s67, 131072 // (op 1024) call()

s_mul_hi_i32 s0, s16, s12 // (op 1041) 
v_lshrrev_b32 v16, 7, v1 // (op 1041) 
s_add_i32 s1, s0, s16 // (op 1041) 

v_and_b32 v18, 15, v1 // (op 1041) 
s_ashr_i32 s0, s1, 31 // (op 1041) 
v_lshl_add_u32 v19, v16, 4, v18 // (op 1041) 


s_cmp_eq_i32 s12, 0 // (op 1041) 
s_mov_b32 s18, scc // (op 1041) 
v_and_b32 v16, 63, v1 // (op 1041) 
s_cmp_eq_i32 s18, 0 // (op 1041) 
s_cselect_b32 s19, 0, -1 // (op 1041) 

v_lshrrev_b32 v18, 4, v16 // (op 1041) 

s_and_b32 s18, s15, 31 // (op 1041) 
s_lshl_b32 s49, 1, s18 // (op 1041) 
s_add_i32 s50, s49, s19 // (op 1041) 


s_and_b32 s19, s0, s50 // (op 1041) 


s_add_i32 s0, s1, s19 // (op 1041) 


s_ashr_i32 s1, s0, s18 // (op 1041) 


s_lshl_b32 s0, s15, 24 // (op 1041) 
s_ashr_i32 s18, s0, 31 // (op 1041) 

s_xor_b32 s0, s1, s18 // (op 1041) 

s_sub_i32 s1, s0, s18 // (op 1041) 


s_mul_hi_i32 s0, s1, s44 // (op 1041) 
s_add_i32 s18, s0, s1 // (op 1041) 

s_ashr_i32 s0, s18, 31 // (op 1041) 
s_cmp_eq_i32 s44, 0 // (op 1041) 
s_mov_b32 s19, scc // (op 1041) 
s_cmp_eq_i32 s19, 0 // (op 1041) 
s_cselect_b32 s49, 0, -1 // (op 1041) 

s_and_b32 s19, s45, 31 // (op 1041) 
s_lshl_b32 s50, 1, s19 // (op 1041) 
s_add_i32 s51, s50, s49 // (op 1041) 


s_and_b32 s49, s0, s51 // (op 1041) 


s_add_i32 s0, s18, s49 // (op 1041) 


s_ashr_i32 s18, s0, s19 // (op 1041) 


s_lshl_b32 s0, s45, 24 // (op 1041) 
s_ashr_i32 s19, s0, 31 // (op 1041) 

s_xor_b32 s0, s18, s19 // (op 1041) 

s_sub_i32 s18, s0, s19 // (op 1041) 


s_mul_i32 s0, s1, s13 // (op 1041) 

s_sub_u32 s1, s16, s0 // (op 1041) 

s_mul_i32 s0, s18, s13 // (op 1041) 
s_add_u32 s0, s0, s1 // (op 1041) 


s_mul_hi_i32 s1, s16, s11 // (op 1041) 
s_add_i32 s18, s1, s16 // (op 1041) 

s_ashr_i32 s1, s18, 31 // (op 1041) 
s_cmp_eq_i32 s11, 0 // (op 1041) 
s_mov_b32 s19, scc // (op 1041) 
s_cmp_eq_i32 s19, 0 // (op 1041) 
s_cselect_b32 s49, 0, -1 // (op 1041) 

s_and_b32 s19, s47, 31 // (op 1041) 
s_lshl_b32 s50, 1, s19 // (op 1041) 
s_add_i32 s51, s50, s49 // (op 1041) 


s_and_b32 s49, s1, s51 // (op 1041) 


s_add_i32 s1, s18, s49 // (op 1041) 


s_ashr_i32 s18, s1, s19 // (op 1041) 


s_lshl_b32 s1, s47, 24 // (op 1041) 
s_ashr_i32 s19, s1, 31 // (op 1041) 

s_xor_b32 s1, s18, s19 // (op 1041) 

s_sub_i32 s18, s1, s19 // (op 1041) 


s_add_u32 s50, s8, 64 // (op 1041) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1041) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1041) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1041) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1041) 

s_mul_i32 s1, s13, s50 // (op 1041) 

s_mul_i32 s19, s10, s1 // (op 1041) 

s_mul_i32 s1, s18, s19 // (op 1041) 

s_sub_u32 s19, s16, s1 // (op 1041) 

s_mov_b32 s1, s19 // (op 1041) 
s_mul_hi_i32 s49, s1, s14 // (op 1041) 
s_add_i32 s50, s49, s1 // (op 1041) 


s_ashr_i32 s1, s50, 31 // (op 1041) 
s_cmp_eq_i32 s14, 0 // (op 1041) 
s_mov_b32 s49, scc // (op 1041) 
s_cmp_eq_i32 s49, 0 // (op 1041) 
s_cselect_b32 s51, 0, -1 // (op 1041) 

s_and_b32 s49, s46, 31 // (op 1041) 
s_lshl_b32 s68, 1, s49 // (op 1041) 
s_add_i32 s69, s68, s51 // (op 1041) 


s_and_b32 s51, s1, s69 // (op 1041) 


s_add_i32 s1, s50, s51 // (op 1041) 


s_ashr_i32 s50, s1, s49 // (op 1041) 


s_lshl_b32 s1, s46, 24 // (op 1041) 
s_ashr_i32 s49, s1, 31 // (op 1041) 

s_xor_b32 s1, s50, s49 // (op 1041) 

s_sub_i32 s50, s1, s49 // (op 1041) 


s_mul_i32 s1, s48, s13 // (op 1041) 
s_add_u32 s68, s6, 32 // (op 1041) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 1041) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1041) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1041) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 1041) 

s_sub_i32 s49, s68, s1 // (op 1041) 

s_mul_i32 s51, s50, s49 // (op 1041) 


s_sub_u32 s49, s19, s51 // (op 1041) 


s_add_u32 s19, s49, s1 // (op 1041) 


s_cmp_eq_i32 s18, 0 // (op 1041) 
s_mov_b32 s1, scc // (op 1041) 

s_cmp_eq_i32 s1, 0 // (op 1041) 
s_cselect_b32 s18, s19, s0 // (op 1041) 



v_lshl_add_u32 v16, s18, 5, v19 // (op 1041) 


v_mul_lo_u32 v19, v18, s34 // (op 1041) 

v_add_u32 v18, v16, v19 // (op 1041) 


s_mul_i32 s1, s34, s39 // (op 1043) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1043) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1043) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1043) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1043) most significant: sum
s_add_u32 s1, s1, s19 // (op 1043) most significant: sum


s_mov_b32 s64, s36 // (op 1043) reg expression
s_mov_b32 s65, s37 // (op 1043) reg expression
s_mov_b32 s66, s0 // (op 1043) reg expression
s_mov_b32 s67, 131072 // (op 1043) call()

s_mul_hi_i32 s0, s16, s12 // (op 1086) 
v_and_b32 v16, -64, v1 // (op 1086) 
s_add_i32 s1, s0, s16 // (op 1086) 

v_sub_u32 v19, v1, v16 // (op 1086) 

s_ashr_i32 s0, s1, 31 // (op 1086) 
v_lshrrev_b32 v16, 6, v1 // (op 1086) 
s_cmp_eq_i32 s12, 0 // (op 1086) 
s_mov_b32 s18, scc // (op 1086) 
s_cmp_eq_i32 s18, 0 // (op 1086) 
s_cselect_b32 s19, 0, -1 // (op 1086) 

s_and_b32 s18, s15, 31 // (op 1086) 
s_lshl_b32 s49, 1, s18 // (op 1086) 
s_add_i32 s50, s49, s19 // (op 1086) 


s_and_b32 s19, s0, s50 // (op 1086) 


s_add_i32 s0, s1, s19 // (op 1086) 


s_ashr_i32 s1, s0, s18 // (op 1086) 


s_lshl_b32 s0, s15, 24 // (op 1086) 
s_ashr_i32 s18, s0, 31 // (op 1086) 

s_xor_b32 s0, s1, s18 // (op 1086) 

s_sub_i32 s1, s0, s18 // (op 1086) 


s_mul_hi_i32 s0, s1, s44 // (op 1086) 
s_add_i32 s18, s0, s1 // (op 1086) 

s_ashr_i32 s0, s18, 31 // (op 1086) 
s_cmp_eq_i32 s44, 0 // (op 1086) 
s_mov_b32 s19, scc // (op 1086) 
s_cmp_eq_i32 s19, 0 // (op 1086) 
s_cselect_b32 s49, 0, -1 // (op 1086) 

s_and_b32 s19, s45, 31 // (op 1086) 
s_lshl_b32 s50, 1, s19 // (op 1086) 
s_add_i32 s51, s50, s49 // (op 1086) 


s_and_b32 s49, s0, s51 // (op 1086) 


s_add_i32 s0, s18, s49 // (op 1086) 


s_ashr_i32 s18, s0, s19 // (op 1086) 


s_lshl_b32 s0, s45, 24 // (op 1086) 
s_ashr_i32 s19, s0, 31 // (op 1086) 

s_xor_b32 s0, s18, s19 // (op 1086) 

s_sub_i32 s18, s0, s19 // (op 1086) 


s_add_u32 s50, s8, 64 // (op 1086) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1086) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1086) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1086) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1086) 

s_mul_i32 s0, s18, s50 // (op 1086) 

s_sub_i32 s49, s1, s0 // (op 1086) 


s_mul_hi_i32 s0, s16, s11 // (op 1086) 
s_add_i32 s1, s0, s16 // (op 1086) 

s_ashr_i32 s0, s1, 31 // (op 1086) 
s_cmp_eq_i32 s11, 0 // (op 1086) 
s_mov_b32 s18, scc // (op 1086) 
s_cmp_eq_i32 s18, 0 // (op 1086) 
s_cselect_b32 s19, 0, -1 // (op 1086) 

s_and_b32 s18, s47, 31 // (op 1086) 
s_lshl_b32 s68, 1, s18 // (op 1086) 
s_add_i32 s69, s68, s19 // (op 1086) 


s_and_b32 s19, s0, s69 // (op 1086) 


s_add_i32 s0, s1, s19 // (op 1086) 


s_ashr_i32 s1, s0, s18 // (op 1086) 


s_lshl_b32 s0, s47, 24 // (op 1086) 
s_ashr_i32 s18, s0, 31 // (op 1086) 

s_xor_b32 s0, s1, s18 // (op 1086) 

s_sub_i32 s1, s0, s18 // (op 1086) 


s_mul_i32 s0, s13, s50 // (op 1086) 

s_mul_i32 s18, s10, s0 // (op 1086) 

s_mul_i32 s0, s1, s18 // (op 1086) 

s_sub_u32 s18, s16, s0 // (op 1086) 

s_mul_hi_i32 s0, s18, s14 // (op 1086) 
s_add_i32 s19, s0, s18 // (op 1086) 


s_ashr_i32 s0, s19, 31 // (op 1086) 
s_cmp_eq_i32 s14, 0 // (op 1086) 
s_mov_b32 s18, scc // (op 1086) 
s_cmp_eq_i32 s18, 0 // (op 1086) 
s_cselect_b32 s50, 0, -1 // (op 1086) 

s_and_b32 s18, s46, 31 // (op 1086) 
s_lshl_b32 s51, 1, s18 // (op 1086) 
s_add_i32 s68, s51, s50 // (op 1086) 


s_and_b32 s50, s0, s68 // (op 1086) 


s_add_i32 s0, s19, s50 // (op 1086) 


s_ashr_i32 s19, s0, s18 // (op 1086) 


s_lshl_b32 s0, s46, 24 // (op 1086) 
s_ashr_i32 s18, s0, 31 // (op 1086) 

s_xor_b32 s0, s19, s18 // (op 1086) 

s_sub_i32 s19, s0, s18 // (op 1086) 


s_cmp_eq_i32 s1, 0 // (op 1086) 
s_mov_b32 s0, scc // (op 1086) 

s_cmp_eq_i32 s0, 0 // (op 1086) 
s_cselect_b32 s1, s19, s49 // (op 1086) 



v_lshl_add_u32 v20, s1, 6, v19 // (op 1086) 


v_mul_lo_u32 v19, v16, s40 // (op 1086) 
v_add_u32 v19, v19, v20 // (op 1086) 


s_mul_i32 s1, s40, s39 // (op 1088) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1088) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1088) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1088) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1088) most significant: sum
s_add_u32 s1, s1, s19 // (op 1088) most significant: sum


s_mov_b32 s68, s42 // (op 1088) reg expression
s_mov_b32 s69, s43 // (op 1088) reg expression
s_mov_b32 s70, s0 // (op 1088) reg expression
s_mov_b32 s71, 131072 // (op 1088) call()

s_mul_hi_i32 s0, s16, s12 // (op 1105) 
v_and_b32 v16, -64, v1 // (op 1105) 
s_add_i32 s1, s0, s16 // (op 1105) 

v_sub_u32 v20, v1, v16 // (op 1105) 

s_ashr_i32 s0, s1, 31 // (op 1105) 
v_lshrrev_b32 v16, 6, v1 // (op 1105) 
s_cmp_eq_i32 s12, 0 // (op 1105) 
s_mov_b32 s18, scc // (op 1105) 
s_cmp_eq_i32 s18, 0 // (op 1105) 
s_cselect_b32 s19, 0, -1 // (op 1105) 

s_and_b32 s18, s15, 31 // (op 1105) 
s_lshl_b32 s49, 1, s18 // (op 1105) 
s_add_i32 s50, s49, s19 // (op 1105) 


s_and_b32 s19, s0, s50 // (op 1105) 


s_add_i32 s0, s1, s19 // (op 1105) 


s_ashr_i32 s1, s0, s18 // (op 1105) 


s_lshl_b32 s0, s15, 24 // (op 1105) 
s_ashr_i32 s18, s0, 31 // (op 1105) 

s_xor_b32 s0, s1, s18 // (op 1105) 

s_sub_i32 s1, s0, s18 // (op 1105) 


s_mul_hi_i32 s0, s1, s44 // (op 1105) 
s_add_i32 s18, s0, s1 // (op 1105) 

s_ashr_i32 s0, s18, 31 // (op 1105) 
s_cmp_eq_i32 s44, 0 // (op 1105) 
s_mov_b32 s19, scc // (op 1105) 
s_cmp_eq_i32 s19, 0 // (op 1105) 
s_cselect_b32 s49, 0, -1 // (op 1105) 

s_and_b32 s19, s45, 31 // (op 1105) 
s_lshl_b32 s50, 1, s19 // (op 1105) 
s_add_i32 s51, s50, s49 // (op 1105) 


s_and_b32 s49, s0, s51 // (op 1105) 


s_add_i32 s0, s18, s49 // (op 1105) 


s_ashr_i32 s18, s0, s19 // (op 1105) 


s_lshl_b32 s0, s45, 24 // (op 1105) 
s_ashr_i32 s19, s0, 31 // (op 1105) 

s_xor_b32 s0, s18, s19 // (op 1105) 

s_sub_i32 s18, s0, s19 // (op 1105) 


s_add_u32 s50, s8, 64 // (op 1105) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1105) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1105) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1105) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1105) 

s_mul_i32 s0, s18, s50 // (op 1105) 

s_sub_i32 s49, s1, s0 // (op 1105) 


s_mul_hi_i32 s0, s16, s11 // (op 1105) 
s_add_i32 s1, s0, s16 // (op 1105) 

s_ashr_i32 s0, s1, 31 // (op 1105) 
s_cmp_eq_i32 s11, 0 // (op 1105) 
s_mov_b32 s18, scc // (op 1105) 
s_cmp_eq_i32 s18, 0 // (op 1105) 
s_cselect_b32 s19, 0, -1 // (op 1105) 

s_and_b32 s18, s47, 31 // (op 1105) 
s_lshl_b32 s72, 1, s18 // (op 1105) 
s_add_i32 s73, s72, s19 // (op 1105) 


s_and_b32 s19, s0, s73 // (op 1105) 


s_add_i32 s0, s1, s19 // (op 1105) 


s_ashr_i32 s1, s0, s18 // (op 1105) 


s_lshl_b32 s0, s47, 24 // (op 1105) 
s_ashr_i32 s18, s0, 31 // (op 1105) 

s_xor_b32 s0, s1, s18 // (op 1105) 

s_sub_i32 s1, s0, s18 // (op 1105) 


s_mul_i32 s0, s13, s50 // (op 1105) 

s_mul_i32 s18, s10, s0 // (op 1105) 

s_mul_i32 s0, s1, s18 // (op 1105) 

s_sub_u32 s18, s16, s0 // (op 1105) 

s_mul_hi_i32 s0, s18, s14 // (op 1105) 
s_add_i32 s19, s0, s18 // (op 1105) 


s_ashr_i32 s0, s19, 31 // (op 1105) 
s_cmp_eq_i32 s14, 0 // (op 1105) 
s_mov_b32 s18, scc // (op 1105) 
s_cmp_eq_i32 s18, 0 // (op 1105) 
s_cselect_b32 s50, 0, -1 // (op 1105) 

s_and_b32 s18, s46, 31 // (op 1105) 
s_lshl_b32 s51, 1, s18 // (op 1105) 
s_add_i32 s72, s51, s50 // (op 1105) 


s_and_b32 s50, s0, s72 // (op 1105) 


s_add_i32 s0, s19, s50 // (op 1105) 


s_ashr_i32 s19, s0, s18 // (op 1105) 


s_lshl_b32 s0, s46, 24 // (op 1105) 
s_ashr_i32 s18, s0, 31 // (op 1105) 

s_xor_b32 s0, s19, s18 // (op 1105) 

s_sub_i32 s19, s0, s18 // (op 1105) 


s_cmp_eq_i32 s1, 0 // (op 1105) 
s_mov_b32 s0, scc // (op 1105) 

s_cmp_eq_i32 s0, 0 // (op 1105) 
s_cselect_b32 s1, s19, s49 // (op 1105) 



v_lshl_add_u32 v21, s1, 6, v20 // (op 1105) 


v_mul_lo_u32 v20, v16, s40 // (op 1105) 
v_add_u32 v20, v20, v21 // (op 1105) 


s_mul_i32 s1, s40, s39 // (op 1107) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1107) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1107) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1107) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1107) most significant: sum
s_add_u32 s1, s1, s19 // (op 1107) most significant: sum


s_mov_b32 s68, s42 // (op 1107) reg expression
s_mov_b32 s69, s43 // (op 1107) reg expression
s_mov_b32 s70, s0 // (op 1107) reg expression
s_mov_b32 s71, 131072 // (op 1107) call()


v_mov_b64 v[22:23], s[34:35] // (op 761) convert

v_mov_b32 v16, v17 // (op 761) reg expression
v_mov_b32 v21, v16 // (op 761) 
buffer_load_ubyte v24, v21, s[64:67], 0 offen offset: 0 // (op 761) Load value





v_mov_b32 v16, v12 // (op 765) reg expression

buffer_load_dwordx3 v[26:28], v16, s[52:55], 0 offen // (op 765) Load value
buffer_load_dwordx3 v[30:32], v16, s[52:55], 0 offen offset:12 // (op 765) Load value



v_mov_b32 v16, v19 // (op 768) reg expression

buffer_load_ubyte v25, v16, s[68:71], 0 offen offset: 0 // (op 768) Load value



v_mov_b32 v16, v14 // (op 771) reg expression

buffer_load_dwordx3 v[34:36], v16, s[20:23], 0 offen // (op 771) Load value


v_mov_b64 v[22:23], s[34:35] // (op 775) convert

s_lshl_b32 s0, s34, 2 // (op 775) 
v_add_u32 v16, v17, s0 // (op 775) 

v_mov_b32 v21, v16 // (op 775) 
buffer_load_ubyte v29, v21, s[64:67], 0 offen offset: 0 // (op 775) Load value






s_lshl_b64 s[0:1], s[30:31], 7 // (op 779) 
s_mul_i32 s19, s1, 6 // (op 779) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 779) least significant
s_mul_hi_u32 s49, s0, 6 // (op 779) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 779) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 779) 

v_add_u32 v33, v12, s0 // (op 779) 


buffer_load_dwordx3 v[38:40], v33, s[52:55], 0 offen // (op 779) Load value
buffer_load_dwordx3 v[42:44], v33, s[52:55], 0 offen offset:12 // (op 779) Load value




s_lshl_b32 s0, s40, 2 // (op 782) 
v_add_u32 v12, v19, s0 // (op 782) 


buffer_load_ubyte v33, v12, s[68:71], 0 offen offset: 0 // (op 782) Load value




s_lshl_b64 s[0:1], s[24:25], 7 // (op 785) 
s_mul_i32 s19, s1, 6 // (op 785) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 785) least significant
s_mul_hi_u32 s49, s0, 6 // (op 785) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 785) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 785) 

v_add_u32 v12, v14, s0 // (op 785) 


buffer_load_dwordx3 v[46:48], v12, s[20:23], 0 offen // (op 785) Load value


v_and_b32 v12, -2, v1 // (op 1225) 
v_sub_u32 v14, v1, v12 // (op 1225) 

v_lshlrev_b32 v12, 5, v14 // (op 1225) 

v_lshrrev_b32 v14, 1, v1 // (op 1225) 
v_lshl_add_u32 v19, v14, 6, v12 // (op 1225) 


v_mul_lo_u32 v12, v19, 6 // (op 1225) 
v_lshrrev_b32 v14, 3, v12 // (op 1225) 

v_lshrrev_b32 v12, 4, v19 // (op 1225) 

v_lshlrev_b32 v19, 2, v12 // (op 1225) 

v_add_u32 v12, v14, v19 // (op 1225) 





s_waitcnt vmcnt(8)
ds_write_b96 v12, v[26:28] // (op 787) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v12, v[30:32] offset:16 // (op 787) Store local data 

v_and_b32 v12, -64, v1 // (op 1283) 
v_sub_u32 v14, v1, v12 // (op 1283) 

v_and_b32 v12, 4294967232, v1 // (op 1283) 
v_add_u32 v19, v12, v14 // (op 1283) 





s_waitcnt vmcnt(6)
ds_write_b8 v19, v25 offset:8192 // (op 788) Store local data 

v_and_b32 v12, -2, v1 // (op 1172) 
v_sub_u32 v14, v1, v12 // (op 1172) 

v_lshlrev_b32 v12, 4, v14 // (op 1172) 

v_lshrrev_b32 v14, 1, v1 // (op 1172) 
v_lshl_add_u32 v19, v14, 5, v12 // (op 1172) 


v_mul_lo_u32 v12, v19, 6 // (op 1172) 
v_lshrrev_b32 v14, 3, v12 // (op 1172) 

v_lshrrev_b32 v12, 4, v19 // (op 1172) 

v_lshlrev_b32 v19, 2, v12 // (op 1172) 

v_add_u32 v12, v14, v19 // (op 1172) 





s_waitcnt vmcnt(5)
ds_write_b96 v12, v[34:36] offset:8448 // (op 789) Store local data 

v_and_b32 v12, -2, v1 // (op 1143) 
v_sub_u32 v14, v1, v12 // (op 1143) 

v_lshlrev_b32 v12, 4, v14 // (op 1143) 

v_lshrrev_b32 v14, 1, v1 // (op 1143) 
v_lshl_add_u32 v19, v14, 5, v12 // (op 1143) 


v_mul_lo_u32 v12, v19, 6 // (op 1143) 
v_lshrrev_b32 v14, 3, v12 // (op 1143) 

v_lshrrev_b32 v12, 4, v19 // (op 1143) 

v_lshlrev_b32 v19, 2, v12 // (op 1143) 

v_add_u32 v12, v14, v19 // (op 1143) 


v_and_b32 v14, -2, v1 // (op 1196) 
v_sub_u32 v19, v1, v14 // (op 1196) 

v_lshlrev_b32 v14, 5, v19 // (op 1196) 

v_lshrrev_b32 v19, 1, v1 // (op 1196) 
v_lshl_add_u32 v37, v19, 6, v14 // (op 1196) 


v_mul_lo_u32 v14, v37, 6 // (op 1196) 
v_lshrrev_b32 v19, 3, v14 // (op 1196) 

v_lshrrev_b32 v14, 4, v37 // (op 1196) 

v_lshlrev_b32 v37, 2, v14 // (op 1196) 

v_add_u32 v14, v19, v37 // (op 1196) 


v_and_b32 v19, -64, v1 // (op 1254) 
v_sub_u32 v37, v1, v19 // (op 1254) 

v_and_b32 v19, 4294967232, v1 // (op 1254) 
v_add_u32 v41, v19, v37 // (op 1254) 


v_and_b32 v19, -2, v1 // (op 1330) 
v_sub_u32 v37, v1, v19 // (op 1330) 

v_lshlrev_b32 v19, 4, v37 // (op 1330) 

v_lshrrev_b32 v37, 1, v1 // (op 1330) 
v_lshl_add_u32 v45, v37, 5, v19 // (op 1330) 


v_mul_lo_u32 v19, v45, 6 // (op 1330) 
v_lshrrev_b32 v37, 3, v19 // (op 1330) 

v_lshrrev_b32 v19, 4, v45 // (op 1330) 

v_lshlrev_b32 v45, 2, v19 // (op 1330) 

v_add_u32 v19, v37, v45 // (op 1330) 


v_and_b32 v37, -2, v1 // (op 1354) 
v_sub_u32 v45, v1, v37 // (op 1354) 

v_lshlrev_b32 v37, 5, v45 // (op 1354) 

v_lshrrev_b32 v45, 1, v1 // (op 1354) 
v_lshl_add_u32 v16, v45, 6, v37 // (op 1354) 


v_mul_lo_u32 v17, v16, 6 // (op 1354) 
v_lshrrev_b32 v37, 3, v17 // (op 1354) 

v_lshrrev_b32 v17, 4, v16 // (op 1354) 

v_lshlrev_b32 v16, 2, v17 // (op 1354) 

v_add_u32 v17, v37, v16 // (op 1354) 


v_and_b32 v16, -64, v1 // (op 1383) 
v_sub_u32 v37, v1, v16 // (op 1383) 

v_and_b32 v16, 4294967232, v1 // (op 1383) 
v_add_u32 v45, v16, v37 // (op 1383) 


s_mov_b32 s0, 0 // (op 308) call()
s_ashr_i64 s[18:19], s[28:29], 7 // (op 307) 
s_mov_b32 s1, s18 // (op 307) convert

s_and_b32 s18, s1, 4294967294 // (op 307) 

s_cmp_lt_u32 s0, s18 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_307 // (op 307) Condition: Top (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_307 if false)
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_307:
s_add_u32 s0, s0, 2 // (op 309) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1441) 



ds_read_b96_tr_b6 v[50:52], v0 offset:8448 // (op 65) Transpose load local data 
ds_read_b96_tr_b6 v[53:55], v0 offset:8960 // (op 65) Transpose load local data 
s_barrier  // (op 1454) 



ds_read_b96_tr_b6 v[56:58], v4 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[59:61], v4 offset:1024 // (op 70) Transpose load local data 
s_barrier  // (op 1468) 


ds_read_u8 v16, v7 offset:8192 // (op 79) Load local data 



ds_read_b96_tr_b6 v[62:64], v4 offset:32 // (op 339) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:55], v[56:61], a[0:3], v24, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 82) 
ds_read_b96_tr_b6 v[65:67], v4 offset:1056 // (op 339) Transpose load local data 




ds_read_u8 v16, v7 offset:8224 // (op 342) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:55], v[62:67], a[4:7], v24, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 343) 




v_mov_b64 v[22:23], s[34:35] // (op 75) convert

v_mov_b32 v16, v18 // (op 75) 
s_lshl_b32 s1, s34, 3 // (op 75) 
v_add_u32 v21, v16, s1 // (op 75) 

v_mov_b32 v37, v21 // (op 75) 
buffer_load_ubyte v24, v37, s[64:67], 0 offen offset: 0 // (op 75) Load value





v_mov_b32 v37, v13 // (op 10) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 10) 
s_mul_i32 s51, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 10) least significant
s_mul_hi_u32 s1, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v21, v37, s1 // (op 10) 


buffer_load_dwordx3 v[26:28], v21, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[30:32], v21, s[52:55], 0 offen offset:12 // (op 10) Load value



v_mov_b32 v21, v20 // (op 21) 
s_lshl_b32 s1, s40, 3 // (op 21) 
v_add_u32 v22, v21, s1 // (op 21) 


buffer_load_ubyte v25, v22, s[68:71], 0 offen offset: 0 // (op 21) Load value



v_mov_b32 v22, v15 // (op 4) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 4) 
s_mul_i32 s51, s19, 6 // (op 4) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 4) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v23, v22, s1 // (op 4) 


buffer_load_dwordx3 v[34:36], v23, s[20:23], 0 offen // (op 4) Load value

s_barrier  // (op 1491) 



s_waitcnt vmcnt(8)
ds_write_b96 v17, v[38:40] offset:12544 // (op 235) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v17, v[42:44] offset:12560 // (op 235) Store local data 
s_barrier  // (op 1497) 



s_waitcnt vmcnt(6)
ds_write_b8 v45, v33 offset:20736 // (op 236) Store local data 
s_barrier  // (op 1485) 



s_waitcnt vmcnt(5)
ds_write_b96 v19, v[46:48] offset:20992 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1482) 



ds_read_b96_tr_b6 v[50:52], v3 offset:20992 // (op 227) Transpose load local data 
ds_read_b96_tr_b6 v[53:55], v3 offset:21504 // (op 227) Transpose load local data 
s_barrier  // (op 1488) 
s_barrier  // (op 1534) 



ds_read_b96_tr_b6 v[56:58], v10 offset:12544 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[59:61], v10 offset:13568 // (op 229) Transpose load local data 
s_barrier  // (op 1494) 


ds_read_u8 v23, v11 offset:20736 // (op 233) Load local data 



ds_read_b96_tr_b6 v[62:64], v10 offset:12576 // (op 348) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:55], v[56:61], a[0:3], v29, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 234) 
ds_read_b96_tr_b6 v[65:67], v10 offset:13600 // (op 348) Transpose load local data 




s_barrier  // (op 1568) 
ds_read_u8 v23, v11 offset:20768 // (op 351) Load local data 
s_barrier  // (op 1572) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:55], v[62:67], a[4:7], v29, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 344) 




v_mov_b64 v[50:51], s[34:35] // (op 231) convert

s_lshl_b32 s1, s34, 2 // (op 231) 
s_mul_i32 s18, 3, s1 // (op 231) 

v_add_u32 v23, v16, s18 // (op 231) 

v_mov_b32 v49, v23 // (op 231) 
buffer_load_ubyte v29, v49, s[64:67], 0 offen offset: 0 // (op 231) Load value






s_lshl_b32 s1, s34, 3 // (op 1033) 
v_add_u32 v18, v18, s1 // (op 1033) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 237) 
s_mul_i32 s51, s19, 6 // (op 237) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 237) least significant
s_mul_hi_u32 s1, s18, 6 // (op 237) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 237) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 237) 

s_mul_i32 s1, 3, s18 // (op 237) 

v_add_u32 v16, v37, s1 // (op 237) 


buffer_load_dwordx3 v[38:40], v16, s[52:55], 0 offen // (op 237) Load value
buffer_load_dwordx3 v[42:44], v16, s[52:55], 0 offen offset:12 // (op 237) Load value




s_lshl_b32 s1, s40, 2 // (op 238) 
s_mul_i32 s18, 3, s1 // (op 238) 

v_add_u32 v16, v21, s18 // (op 238) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 969) 
s_mul_i32 s51, s19, 6 // (op 969) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 969) least significant
s_mul_hi_u32 s1, s18, 6 // (op 969) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 969) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 969) 

s_lshl_b32 s1, s18, 1 // (op 969) 

v_add_u32 v13, v13, s1 // (op 969) 

buffer_load_ubyte v33, v16, s[68:71], 0 offen offset: 0 // (op 238) Load value




s_lshl_b32 s1, s40, 3 // (op 1097) 
v_add_u32 v20, v20, s1 // (op 1097) 

s_lshl_b64 s[18:19], s[24:25], 7 // (op 224) 
s_mul_i32 s51, s19, 6 // (op 224) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 224) least significant
s_mul_hi_u32 s1, s18, 6 // (op 224) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 224) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 224) 

s_mul_i32 s1, 3, s18 // (op 224) 

v_add_u32 v16, v22, s1 // (op 224) 


buffer_load_dwordx3 v[46:48], v16, s[20:23], 0 offen // (op 224) Load value


s_barrier  // (op 1457) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 905) 



s_mul_i32 s51, s19, 6 // (op 905) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 905) least significant
s_mul_hi_u32 s1, s18, 6 // (op 905) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 905) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 905) 

s_lshl_b32 s1, s18, 1 // (op 905) 

v_add_u32 v15, v15, s1 // (op 905) 

s_waitcnt vmcnt(8)
ds_write_b96 v14, v[26:28] // (op 13) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v14, v[30:32] offset:16 // (op 13) Store local data 
s_barrier  // (op 1471) 



s_waitcnt vmcnt(6)
ds_write_b8 v41, v25 offset:8192 // (op 26) Store local data 
s_barrier  // (op 1444) 



s_waitcnt vmcnt(5)
ds_write_b96 v12, v[34:36] offset:8448 // (op 6) Store local data 
s_ashr_i64 s[18:19], s[28:29], 7 // (op 307) 
s_mov_b32 s1, s18 // (op 307) convert

s_and_b32 s18, s1, 4294967294 // (op 307) 

s_cmp_lt_u32 s0, s18 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_307 // (op 307) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_307 if true)
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_307:















s_mul_hi_i32 s0, s16, s12 // (op 867) 
v_and_b32 v41, -2, v1 // (op 867) 
s_add_i32 s1, s0, s16 // (op 867) 

v_sub_u32 v45, v1, v41 // (op 867) 

s_ashr_i32 s0, s1, 31 // (op 867) 
v_lshlrev_b32 v41, 4, v45 // (op 867) 

s_cmp_eq_i32 s12, 0 // (op 867) 
s_mov_b32 s18, scc // (op 867) 
v_lshrrev_b32 v45, 1, v1 // (op 867) 
s_cmp_eq_i32 s18, 0 // (op 867) 
s_cselect_b32 s19, 0, -1 // (op 867) 

s_and_b32 s18, s15, 31 // (op 867) 
s_lshl_b32 s49, 1, s18 // (op 867) 
s_add_i32 s50, s49, s19 // (op 867) 


s_and_b32 s19, s0, s50 // (op 867) 


s_add_i32 s0, s1, s19 // (op 867) 


s_ashr_i32 s1, s0, s18 // (op 867) 


s_lshl_b32 s0, s15, 24 // (op 867) 
s_ashr_i32 s18, s0, 31 // (op 867) 

s_xor_b32 s0, s1, s18 // (op 867) 

s_sub_i32 s1, s0, s18 // (op 867) 


s_mul_hi_i32 s0, s1, s44 // (op 867) 
s_add_i32 s18, s0, s1 // (op 867) 

s_ashr_i32 s0, s18, 31 // (op 867) 
s_cmp_eq_i32 s44, 0 // (op 867) 
s_mov_b32 s19, scc // (op 867) 
s_cmp_eq_i32 s19, 0 // (op 867) 
s_cselect_b32 s49, 0, -1 // (op 867) 

s_and_b32 s19, s45, 31 // (op 867) 
s_lshl_b32 s50, 1, s19 // (op 867) 
s_add_i32 s51, s50, s49 // (op 867) 


s_and_b32 s49, s0, s51 // (op 867) 


s_add_i32 s0, s18, s49 // (op 867) 


s_ashr_i32 s18, s0, s19 // (op 867) 


s_lshl_b32 s0, s45, 24 // (op 867) 
s_ashr_i32 s19, s0, 31 // (op 867) 

s_xor_b32 s0, s18, s19 // (op 867) 

s_sub_i32 s18, s0, s19 // (op 867) 


s_mul_i32 s0, s1, s13 // (op 867) 

s_sub_u32 s1, s16, s0 // (op 867) 

s_mul_i32 s0, s18, s13 // (op 867) 
s_add_u32 s0, s0, s1 // (op 867) 


s_mul_hi_i32 s1, s16, s11 // (op 867) 
s_add_i32 s18, s1, s16 // (op 867) 

s_ashr_i32 s1, s18, 31 // (op 867) 
s_cmp_eq_i32 s11, 0 // (op 867) 
s_mov_b32 s19, scc // (op 867) 
s_cmp_eq_i32 s19, 0 // (op 867) 
s_cselect_b32 s49, 0, -1 // (op 867) 

s_and_b32 s19, s47, 31 // (op 867) 
s_lshl_b32 s50, 1, s19 // (op 867) 
s_add_i32 s51, s50, s49 // (op 867) 


s_and_b32 s49, s1, s51 // (op 867) 


s_add_i32 s1, s18, s49 // (op 867) 


s_ashr_i32 s18, s1, s19 // (op 867) 


s_lshl_b32 s1, s47, 24 // (op 867) 
s_ashr_i32 s19, s1, 31 // (op 867) 

s_xor_b32 s1, s18, s19 // (op 867) 

s_sub_i32 s18, s1, s19 // (op 867) 


s_add_u32 s50, s8, 64 // (op 867) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 867) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 867) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 867) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 867) 

s_mul_i32 s1, s13, s50 // (op 867) 

s_mul_i32 s19, s10, s1 // (op 867) 

s_mul_i32 s1, s18, s19 // (op 867) 

s_sub_u32 s19, s16, s1 // (op 867) 

s_mov_b32 s1, s19 // (op 867) 
s_mul_hi_i32 s49, s1, s14 // (op 867) 
s_add_i32 s50, s49, s1 // (op 867) 


s_ashr_i32 s1, s50, 31 // (op 867) 
s_cmp_eq_i32 s14, 0 // (op 867) 
s_mov_b32 s49, scc // (op 867) 
s_cmp_eq_i32 s49, 0 // (op 867) 
s_cselect_b32 s51, 0, -1 // (op 867) 

s_and_b32 s49, s46, 31 // (op 867) 
s_lshl_b32 s72, 1, s49 // (op 867) 
s_add_i32 s73, s72, s51 // (op 867) 


s_and_b32 s51, s1, s73 // (op 867) 


s_add_i32 s1, s50, s51 // (op 867) 


s_ashr_i32 s50, s1, s49 // (op 867) 


s_lshl_b32 s1, s46, 24 // (op 867) 
s_ashr_i32 s49, s1, 31 // (op 867) 

s_xor_b32 s1, s50, s49 // (op 867) 

s_sub_i32 s50, s1, s49 // (op 867) 


s_mul_i32 s1, s48, s13 // (op 867) 
s_add_u32 s72, s6, 32 // (op 867) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 867) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 867) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 867) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 5 // (op 867) 

s_sub_i32 s49, s72, s1 // (op 867) 

s_mul_i32 s51, s50, s49 // (op 867) 


s_sub_u32 s49, s19, s51 // (op 867) 


s_add_u32 s19, s49, s1 // (op 867) 


s_cmp_eq_i32 s18, 0 // (op 867) 
s_mov_b32 s1, scc // (op 867) 

s_cmp_eq_i32 s1, 0 // (op 867) 
s_cselect_b32 s18, s19, s0 // (op 867) 



v_lshl_add_u32 v12, s18, 5, v41 // (op 867) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 867) 
v_mov_b32 v31, 0 // (op 867) convert
v_mov_b32 v30, v12 // (op 867) convert

s_lshr_b32 s18, s0, 1 // (op 867) 

v_lshl_add_u32 v32, s18, 8, v45 // (op 867) 


v_mul_lo_u32 v13, v32, s25 // (op 867) most significant: low of low * high
v_mul_hi_u32 v41, v32, s24 // (op 867) most significant: high of low * low
v_mul_lo_u32 v12, v32, s24 // (op 867) least significant: low of low * low
v_add_u32 v13, v13, v41 // (op 867) most significant: sum


v_add_co_u32 v14, s[0:1], v30, v12 // (op 867) least significant half
v_addc_co_u32 v15, s[0:1], v31, v13, s[0:1] // (op 867) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 867) most significant: low of high * low
v_mul_hi_u32 v41, v14, 6 // (op 867) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 867) least significant: low of low * low
v_add_u32 v13, v13, v41 // (op 867) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 867) 

v_mov_b32 v41, v14 // (op 867) convert

s_mul_i32 s1, s29, 6 // (op 869) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 869) least significant
s_mul_hi_u32 s18, s28, 6 // (op 869) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 869) most significant: sum

s_mul_i32 s19, s24, s1 // (op 869) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 869) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 869) least significant
s_mul_hi_u32 s50, s24, s0 // (op 869) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 869) most significant: sum
s_add_u32 s19, s19, s50 // (op 869) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 869) 

s_mov_b32 s20, s26 // (op 869) reg expression
s_mov_b32 s21, s27 // (op 869) reg expression
s_mov_b32 s22, s0 // (op 869) reg expression
s_mov_b32 s23, 131072 // (op 869) call()

s_mul_hi_i32 s0, s16, s12 // (op 935) 
v_and_b32 v45, -2, v1 // (op 935) 
s_add_i32 s1, s0, s16 // (op 935) 

v_sub_u32 v12, v1, v45 // (op 935) 

s_ashr_i32 s0, s1, 31 // (op 935) 
v_lshlrev_b32 v45, 5, v12 // (op 935) 

s_cmp_eq_i32 s12, 0 // (op 935) 
s_mov_b32 s18, scc // (op 935) 
v_lshrrev_b32 v12, 1, v1 // (op 935) 
s_cmp_eq_i32 s18, 0 // (op 935) 
s_cselect_b32 s19, 0, -1 // (op 935) 

s_and_b32 s18, s15, 31 // (op 935) 
s_lshl_b32 s49, 1, s18 // (op 935) 
s_add_i32 s50, s49, s19 // (op 935) 


s_and_b32 s19, s0, s50 // (op 935) 


s_add_i32 s0, s1, s19 // (op 935) 


s_ashr_i32 s1, s0, s18 // (op 935) 


s_lshl_b32 s0, s15, 24 // (op 935) 
s_ashr_i32 s18, s0, 31 // (op 935) 

s_xor_b32 s0, s1, s18 // (op 935) 

s_sub_i32 s1, s0, s18 // (op 935) 


s_mul_hi_i32 s0, s1, s44 // (op 935) 
s_add_i32 s18, s0, s1 // (op 935) 

s_ashr_i32 s0, s18, 31 // (op 935) 
s_cmp_eq_i32 s44, 0 // (op 935) 
s_mov_b32 s19, scc // (op 935) 
s_cmp_eq_i32 s19, 0 // (op 935) 
s_cselect_b32 s49, 0, -1 // (op 935) 

s_and_b32 s19, s45, 31 // (op 935) 
s_lshl_b32 s50, 1, s19 // (op 935) 
s_add_i32 s51, s50, s49 // (op 935) 


s_and_b32 s49, s0, s51 // (op 935) 


s_add_i32 s0, s18, s49 // (op 935) 


s_ashr_i32 s18, s0, s19 // (op 935) 


s_lshl_b32 s0, s45, 24 // (op 935) 
s_ashr_i32 s19, s0, 31 // (op 935) 

s_xor_b32 s0, s18, s19 // (op 935) 

s_sub_i32 s18, s0, s19 // (op 935) 


s_add_u32 s50, s8, 64 // (op 935) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 935) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 935) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 935) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 935) 

s_mul_i32 s0, s18, s50 // (op 935) 

s_sub_i32 s49, s1, s0 // (op 935) 


s_mul_hi_i32 s0, s16, s11 // (op 935) 
s_add_i32 s1, s0, s16 // (op 935) 

s_ashr_i32 s0, s1, 31 // (op 935) 
s_cmp_eq_i32 s11, 0 // (op 935) 
s_mov_b32 s18, scc // (op 935) 
s_cmp_eq_i32 s18, 0 // (op 935) 
s_cselect_b32 s19, 0, -1 // (op 935) 

s_and_b32 s18, s47, 31 // (op 935) 
s_lshl_b32 s72, 1, s18 // (op 935) 
s_add_i32 s73, s72, s19 // (op 935) 


s_and_b32 s19, s0, s73 // (op 935) 


s_add_i32 s0, s1, s19 // (op 935) 


s_ashr_i32 s1, s0, s18 // (op 935) 


s_lshl_b32 s0, s47, 24 // (op 935) 
s_ashr_i32 s18, s0, 31 // (op 935) 

s_xor_b32 s0, s1, s18 // (op 935) 

s_sub_i32 s1, s0, s18 // (op 935) 


s_mul_i32 s0, s13, s50 // (op 935) 

s_mul_i32 s18, s10, s0 // (op 935) 

s_mul_i32 s0, s1, s18 // (op 935) 

s_sub_u32 s18, s16, s0 // (op 935) 

s_mul_hi_i32 s0, s18, s14 // (op 935) 
s_add_i32 s19, s0, s18 // (op 935) 


s_ashr_i32 s0, s19, 31 // (op 935) 
s_cmp_eq_i32 s14, 0 // (op 935) 
s_mov_b32 s18, scc // (op 935) 
s_cmp_eq_i32 s18, 0 // (op 935) 
s_cselect_b32 s50, 0, -1 // (op 935) 

s_and_b32 s18, s46, 31 // (op 935) 
s_lshl_b32 s51, 1, s18 // (op 935) 
s_add_i32 s72, s51, s50 // (op 935) 


s_and_b32 s50, s0, s72 // (op 935) 


s_add_i32 s0, s19, s50 // (op 935) 


s_ashr_i32 s19, s0, s18 // (op 935) 


s_lshl_b32 s0, s46, 24 // (op 935) 
s_ashr_i32 s18, s0, 31 // (op 935) 

s_xor_b32 s0, s19, s18 // (op 935) 

s_sub_i32 s19, s0, s18 // (op 935) 


s_cmp_eq_i32 s1, 0 // (op 935) 
s_mov_b32 s0, scc // (op 935) 

s_cmp_eq_i32 s0, 0 // (op 935) 
s_cselect_b32 s1, s19, s49 // (op 935) 



v_lshl_add_u32 v13, s1, 6, v45 // (op 935) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 935) 
v_mov_b32 v15, 0 // (op 935) convert
v_mov_b32 v14, v13 // (op 935) convert

s_lshr_b32 s18, s0, 1 // (op 935) 

v_lshl_add_u32 v13, s18, 8, v12 // (op 935) 


v_mul_lo_u32 v17, v13, s31 // (op 935) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 935) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 935) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 935) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 935) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 935) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 935) most significant: low of high * low
v_mul_hi_u32 v45, v16, 6 // (op 935) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 935) least significant: low of low * low
v_add_u32 v13, v13, v45 // (op 935) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 935) 

v_mov_b32 v45, v14 // (op 935) convert

s_mul_i32 s1, s29, 6 // (op 937) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 937) least significant
s_mul_hi_u32 s18, s28, 6 // (op 937) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 937) most significant: sum

s_mul_i32 s19, s30, s1 // (op 937) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 937) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 937) least significant
s_mul_hi_u32 s50, s30, s0 // (op 937) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 937) most significant: sum
s_add_u32 s19, s19, s50 // (op 937) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 937) 

s_mov_b32 s52, s32 // (op 937) reg expression
s_mov_b32 s53, s33 // (op 937) reg expression
s_mov_b32 s54, s0 // (op 937) reg expression
s_mov_b32 s55, 131072 // (op 937) call()

s_mul_hi_i32 s0, s16, s12 // (op 999) 
v_lshrrev_b32 v12, 7, v1 // (op 999) 
s_add_i32 s1, s0, s16 // (op 999) 

v_and_b32 v13, 15, v1 // (op 999) 
s_ashr_i32 s0, s1, 31 // (op 999) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 999) 


s_cmp_eq_i32 s12, 0 // (op 999) 
s_mov_b32 s18, scc // (op 999) 
v_and_b32 v12, 63, v1 // (op 999) 
s_cmp_eq_i32 s18, 0 // (op 999) 
s_cselect_b32 s19, 0, -1 // (op 999) 

v_lshrrev_b32 v13, 4, v12 // (op 999) 

s_and_b32 s18, s15, 31 // (op 999) 
s_lshl_b32 s49, 1, s18 // (op 999) 
s_add_i32 s50, s49, s19 // (op 999) 


s_and_b32 s19, s0, s50 // (op 999) 


s_add_i32 s0, s1, s19 // (op 999) 


s_ashr_i32 s1, s0, s18 // (op 999) 


s_lshl_b32 s0, s15, 24 // (op 999) 
s_ashr_i32 s18, s0, 31 // (op 999) 

s_xor_b32 s0, s1, s18 // (op 999) 

s_sub_i32 s1, s0, s18 // (op 999) 


s_mul_hi_i32 s0, s1, s44 // (op 999) 
s_add_i32 s18, s0, s1 // (op 999) 

s_ashr_i32 s0, s18, 31 // (op 999) 
s_cmp_eq_i32 s44, 0 // (op 999) 
s_mov_b32 s19, scc // (op 999) 
s_cmp_eq_i32 s19, 0 // (op 999) 
s_cselect_b32 s49, 0, -1 // (op 999) 

s_and_b32 s19, s45, 31 // (op 999) 
s_lshl_b32 s50, 1, s19 // (op 999) 
s_add_i32 s51, s50, s49 // (op 999) 


s_and_b32 s49, s0, s51 // (op 999) 


s_add_i32 s0, s18, s49 // (op 999) 


s_ashr_i32 s18, s0, s19 // (op 999) 


s_lshl_b32 s0, s45, 24 // (op 999) 
s_ashr_i32 s19, s0, 31 // (op 999) 

s_xor_b32 s0, s18, s19 // (op 999) 

s_sub_i32 s18, s0, s19 // (op 999) 


s_mul_i32 s0, s1, s13 // (op 999) 

s_sub_u32 s1, s16, s0 // (op 999) 

s_mul_i32 s0, s18, s13 // (op 999) 
s_add_u32 s0, s0, s1 // (op 999) 


s_mul_hi_i32 s1, s16, s11 // (op 999) 
s_add_i32 s18, s1, s16 // (op 999) 

s_ashr_i32 s1, s18, 31 // (op 999) 
s_cmp_eq_i32 s11, 0 // (op 999) 
s_mov_b32 s19, scc // (op 999) 
s_cmp_eq_i32 s19, 0 // (op 999) 
s_cselect_b32 s49, 0, -1 // (op 999) 

s_and_b32 s19, s47, 31 // (op 999) 
s_lshl_b32 s50, 1, s19 // (op 999) 
s_add_i32 s51, s50, s49 // (op 999) 


s_and_b32 s49, s1, s51 // (op 999) 


s_add_i32 s1, s18, s49 // (op 999) 


s_ashr_i32 s18, s1, s19 // (op 999) 


s_lshl_b32 s1, s47, 24 // (op 999) 
s_ashr_i32 s19, s1, 31 // (op 999) 

s_xor_b32 s1, s18, s19 // (op 999) 

s_sub_i32 s18, s1, s19 // (op 999) 


s_add_u32 s50, s8, 64 // (op 999) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 999) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 999) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 999) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 999) 

s_mul_i32 s1, s13, s50 // (op 999) 

s_mul_i32 s19, s10, s1 // (op 999) 

s_mul_i32 s1, s18, s19 // (op 999) 

s_sub_u32 s19, s16, s1 // (op 999) 

s_mov_b32 s1, s19 // (op 999) 
s_mul_hi_i32 s49, s1, s14 // (op 999) 
s_add_i32 s50, s49, s1 // (op 999) 


s_ashr_i32 s1, s50, 31 // (op 999) 
s_cmp_eq_i32 s14, 0 // (op 999) 
s_mov_b32 s49, scc // (op 999) 
s_cmp_eq_i32 s49, 0 // (op 999) 
s_cselect_b32 s51, 0, -1 // (op 999) 

s_and_b32 s49, s46, 31 // (op 999) 
s_lshl_b32 s72, 1, s49 // (op 999) 
s_add_i32 s73, s72, s51 // (op 999) 


s_and_b32 s51, s1, s73 // (op 999) 


s_add_i32 s1, s50, s51 // (op 999) 


s_ashr_i32 s50, s1, s49 // (op 999) 


s_lshl_b32 s1, s46, 24 // (op 999) 
s_ashr_i32 s49, s1, 31 // (op 999) 

s_xor_b32 s1, s50, s49 // (op 999) 

s_sub_i32 s50, s1, s49 // (op 999) 


s_mul_i32 s1, s48, s13 // (op 999) 
s_add_u32 s72, s6, 32 // (op 999) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 999) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 999) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 999) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 5 // (op 999) 

s_sub_i32 s49, s72, s1 // (op 999) 

s_mul_i32 s51, s50, s49 // (op 999) 


s_sub_u32 s49, s19, s51 // (op 999) 


s_add_u32 s19, s49, s1 // (op 999) 


s_cmp_eq_i32 s18, 0 // (op 999) 
s_mov_b32 s1, scc // (op 999) 

s_cmp_eq_i32 s1, 0 // (op 999) 
s_cselect_b32 s18, s19, s0 // (op 999) 



v_lshl_add_u32 v12, s18, 5, v14 // (op 999) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 999) 
s_lshr_b32 s18, s0, 1 // (op 999) 

v_lshl_add_u32 v14, s18, 3, v13 // (op 999) 


v_mul_lo_u32 v13, v14, s34 // (op 999) 

v_add_u32 v14, v12, v13 // (op 999) 


s_mul_i32 s1, s34, s39 // (op 1001) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1001) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1001) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1001) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1001) most significant: sum
s_add_u32 s1, s1, s19 // (op 1001) most significant: sum


s_mov_b32 s64, s36 // (op 1001) reg expression
s_mov_b32 s65, s37 // (op 1001) reg expression
s_mov_b32 s66, s0 // (op 1001) reg expression
s_mov_b32 s67, 131072 // (op 1001) call()

s_mul_hi_i32 s0, s16, s12 // (op 1063) 
v_and_b32 v12, -64, v1 // (op 1063) 
s_add_i32 s1, s0, s16 // (op 1063) 

v_sub_u32 v13, v1, v12 // (op 1063) 

s_ashr_i32 s0, s1, 31 // (op 1063) 
v_lshrrev_b32 v12, 6, v1 // (op 1063) 
s_cmp_eq_i32 s12, 0 // (op 1063) 
s_mov_b32 s18, scc // (op 1063) 
s_cmp_eq_i32 s18, 0 // (op 1063) 
s_cselect_b32 s19, 0, -1 // (op 1063) 

s_and_b32 s18, s15, 31 // (op 1063) 
s_lshl_b32 s49, 1, s18 // (op 1063) 
s_add_i32 s50, s49, s19 // (op 1063) 


s_and_b32 s19, s0, s50 // (op 1063) 


s_add_i32 s0, s1, s19 // (op 1063) 


s_ashr_i32 s1, s0, s18 // (op 1063) 


s_lshl_b32 s0, s15, 24 // (op 1063) 
s_ashr_i32 s18, s0, 31 // (op 1063) 

s_xor_b32 s0, s1, s18 // (op 1063) 

s_sub_i32 s1, s0, s18 // (op 1063) 


s_mul_hi_i32 s0, s1, s44 // (op 1063) 
s_add_i32 s18, s0, s1 // (op 1063) 

s_ashr_i32 s0, s18, 31 // (op 1063) 
s_cmp_eq_i32 s44, 0 // (op 1063) 
s_mov_b32 s19, scc // (op 1063) 
s_cmp_eq_i32 s19, 0 // (op 1063) 
s_cselect_b32 s49, 0, -1 // (op 1063) 

s_and_b32 s19, s45, 31 // (op 1063) 
s_lshl_b32 s50, 1, s19 // (op 1063) 
s_add_i32 s51, s50, s49 // (op 1063) 


s_and_b32 s49, s0, s51 // (op 1063) 


s_add_i32 s0, s18, s49 // (op 1063) 


s_ashr_i32 s18, s0, s19 // (op 1063) 


s_lshl_b32 s0, s45, 24 // (op 1063) 
s_ashr_i32 s19, s0, 31 // (op 1063) 

s_xor_b32 s0, s18, s19 // (op 1063) 

s_sub_i32 s18, s0, s19 // (op 1063) 


s_add_u32 s50, s8, 64 // (op 1063) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1063) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1063) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1063) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1063) 

s_mul_i32 s0, s18, s50 // (op 1063) 

s_sub_i32 s49, s1, s0 // (op 1063) 


s_mul_hi_i32 s0, s16, s11 // (op 1063) 
s_add_i32 s1, s0, s16 // (op 1063) 

s_ashr_i32 s0, s1, 31 // (op 1063) 
s_cmp_eq_i32 s11, 0 // (op 1063) 
s_mov_b32 s18, scc // (op 1063) 
s_cmp_eq_i32 s18, 0 // (op 1063) 
s_cselect_b32 s19, 0, -1 // (op 1063) 

s_and_b32 s18, s47, 31 // (op 1063) 
s_lshl_b32 s72, 1, s18 // (op 1063) 
s_add_i32 s73, s72, s19 // (op 1063) 


s_and_b32 s19, s0, s73 // (op 1063) 


s_add_i32 s0, s1, s19 // (op 1063) 


s_ashr_i32 s1, s0, s18 // (op 1063) 


s_lshl_b32 s0, s47, 24 // (op 1063) 
s_ashr_i32 s18, s0, 31 // (op 1063) 

s_xor_b32 s0, s1, s18 // (op 1063) 

s_sub_i32 s1, s0, s18 // (op 1063) 


s_mul_i32 s0, s13, s50 // (op 1063) 

s_mul_i32 s18, s10, s0 // (op 1063) 

s_mul_i32 s0, s1, s18 // (op 1063) 

s_sub_u32 s18, s16, s0 // (op 1063) 

s_mul_hi_i32 s0, s18, s14 // (op 1063) 
s_add_i32 s19, s0, s18 // (op 1063) 


s_ashr_i32 s0, s19, 31 // (op 1063) 
s_cmp_eq_i32 s14, 0 // (op 1063) 
s_mov_b32 s18, scc // (op 1063) 
s_cmp_eq_i32 s18, 0 // (op 1063) 
s_cselect_b32 s50, 0, -1 // (op 1063) 

s_and_b32 s18, s46, 31 // (op 1063) 
s_lshl_b32 s51, 1, s18 // (op 1063) 
s_add_i32 s72, s51, s50 // (op 1063) 


s_and_b32 s50, s0, s72 // (op 1063) 


s_add_i32 s0, s19, s50 // (op 1063) 


s_ashr_i32 s19, s0, s18 // (op 1063) 


s_lshl_b32 s0, s46, 24 // (op 1063) 
s_ashr_i32 s18, s0, 31 // (op 1063) 

s_xor_b32 s0, s19, s18 // (op 1063) 

s_sub_i32 s19, s0, s18 // (op 1063) 


s_cmp_eq_i32 s1, 0 // (op 1063) 
s_mov_b32 s0, scc // (op 1063) 

s_cmp_eq_i32 s0, 0 // (op 1063) 
s_cselect_b32 s1, s19, s49 // (op 1063) 



v_lshl_add_u32 v15, s1, 6, v13 // (op 1063) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 1063) 
s_lshr_b32 s18, s0, 1 // (op 1063) 

v_lshl_add_u32 v13, s18, 3, v12 // (op 1063) 


v_mul_lo_u32 v12, v13, s40 // (op 1063) 
v_add_u32 v12, v12, v15 // (op 1063) 


s_mul_i32 s1, s40, s39 // (op 1065) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1065) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1065) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1065) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1065) most significant: sum
s_add_u32 s1, s1, s19 // (op 1065) most significant: sum


s_mov_b32 s68, s42 // (op 1065) reg expression
s_mov_b32 s69, s43 // (op 1065) reg expression
s_mov_b32 s70, s0 // (op 1065) reg expression
s_mov_b32 s71, 131072 // (op 1065) call()

v_and_b32 v13, -2, v1 // (op 1156) 
v_sub_u32 v15, v1, v13 // (op 1156) 

v_lshlrev_b32 v13, 4, v15 // (op 1156) 

v_lshrrev_b32 v15, 1, v1 // (op 1156) 
v_lshl_add_u32 v16, v15, 5, v13 // (op 1156) 


v_mul_lo_u32 v13, v16, 6 // (op 1156) 
v_lshrrev_b32 v15, 3, v13 // (op 1156) 

v_lshrrev_b32 v13, 4, v16 // (op 1156) 

v_lshlrev_b32 v16, 2, v13 // (op 1156) 

v_add_u32 v13, v15, v16 // (op 1156) 


v_and_b32 v15, -2, v1 // (op 1209) 
v_sub_u32 v16, v1, v15 // (op 1209) 

v_lshlrev_b32 v15, 5, v16 // (op 1209) 

v_lshrrev_b32 v16, 1, v1 // (op 1209) 
v_lshl_add_u32 v17, v16, 6, v15 // (op 1209) 


v_mul_lo_u32 v15, v17, 6 // (op 1209) 
v_lshrrev_b32 v16, 3, v15 // (op 1209) 

v_lshrrev_b32 v15, 4, v17 // (op 1209) 

v_lshlrev_b32 v17, 2, v15 // (op 1209) 

v_add_u32 v15, v16, v17 // (op 1209) 


v_and_b32 v16, -64, v1 // (op 1267) 
v_sub_u32 v17, v1, v16 // (op 1267) 

v_and_b32 v16, 4294967232, v1 // (op 1267) 
v_add_u32 v18, v16, v17 // (op 1267) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 313) 
s_and_b32 s18, s0, 4294967294 // (op 313) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 312) 
s_mov_b32 s19, s0 // (op 312) convert

s_cmp_lt_u32 s18, s19 // (op 312) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312 // (op 312) Condition: Top (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312:


s_add_u32 s18, s18, 1 // (op 314) 
v_mov_b32 v16, v41 // (op 166) 
v_mov_b32 v17, v16 // (op 166) reg expression

buffer_load_dwordx3 v[20:22], v17, s[20:23], 0 offen // (op 166) Load value


s_lshl_b64 s[0:1], s[24:25], 7 // (op 857) 
s_mul_i32 s51, s1, 6 // (op 857) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 857) least significant
s_mul_hi_u32 s19, s0, 6 // (op 857) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 857) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 857) 

v_add_u32 v41, v41, s0 // (op 857) 



s_barrier  // (op 1447) 
v_mov_b32 v19, v45 // (op 179) 



v_mov_b32 v16, v19 // (op 179) reg expression

buffer_load_dwordx3 v[24:26], v16, s[52:55], 0 offen // (op 179) Load value
buffer_load_dwordx3 v[28:30], v16, s[52:55], 0 offen offset:12 // (op 179) Load value


s_lshl_b64 s[0:1], s[30:31], 7 // (op 927) 
s_mul_i32 s51, s1, 6 // (op 927) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 927) least significant
s_mul_hi_u32 s19, s0, 6 // (op 927) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 927) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 927) 

v_add_u32 v45, v45, s0 // (op 927) 

s_waitcnt vmcnt(2)
ds_write_b96 v13, v[20:22] offset:8448 // (op 167) Store local data 



v_mov_b32 v27, v12 // (op 180) 
s_barrier  // (op 1460) 
v_mov_b32 v16, v27 // (op 180) reg expression




buffer_load_ubyte v17, v16, s[68:71], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s19, s40, 2 // (op 1055) 
v_add_u32 v12, v12, s19 // (op 1055) 

s_waitcnt vmcnt(2)
ds_write_b96 v15, v[24:26] // (op 177) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1450) 
s_waitcnt vmcnt(1)
ds_write_b96 v15, v[28:30] offset:16 // (op 177) Store local data 




ds_read_b96_tr_b6 v[20:22], v0 offset:8448 // (op 169) Transpose load local data 
s_barrier  // (op 1474) 
ds_read_b96_tr_b6 v[23:25], v0 offset:8960 // (op 169) Transpose load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 1464) 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v17 offset:8192 // (op 178) Store local data 




ds_read_b96_tr_b6 v[26:28], v4 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[29:31], v4 offset:1024 // (op 171) Transpose load local data 

v_mov_b64 v[16:17], s[34:35] // (op 173) convert

v_mov_b32 v19, v14 // (op 173) 
v_mov_b32 v32, v19 // (op 173) reg expression
v_mov_b32 v33, v32 // (op 173) 
buffer_load_ubyte v34, v33, s[64:67], 0 offen offset: 0 // (op 173) Load value




s_lshl_b32 s19, s34, 2 // (op 991) 
v_add_u32 v14, v14, s19 // (op 991) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1478) 


ds_read_u8 v19, v7 offset:8192 // (op 175) Load local data 



ds_read_b96_tr_b6 v[46:48], v4 offset:32 // (op 320) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v4 offset:1056 // (op 320) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v34, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 176) 




ds_read_u8 v19, v7 offset:8224 // (op 323) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:25], v[46:51], a[4:7], v34, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 324) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 312) 
s_mov_b32 s19, s0 // (op 312) convert

s_cmp_lt_u32 s18, s19 // (op 312) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312 // (op 312) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312 if true)
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312:






















s_barrier  // (op 1556) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 



v_mov_b32 v0, v9 // (op 52) reg expression

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 52) Store value


v_accvgpr_read v10, a4 // (op 504) 
v_accvgpr_read v11, a5 // (op 504) 
v_accvgpr_read v12, a6 // (op 504) 
v_accvgpr_read v13, a7 // (op 504) 
v_mul_f32 v14, v8, v10 // (op 504) 
v_mul_f32 v15, v8, v11 // (op 504) 
v_mul_f32 v16, v8, v12 // (op 504) 
v_mul_f32 v17, v8, v13 // (op 504) 



s_lshl_b32 s0, s4, 7 // (op 506) 
v_add_u32 v0, v9, s0 // (op 506) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 506) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_123:


v_mov_b32 v0, v6 // (op 135) reg expression

buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 135) Load value

v_accvgpr_read v14, a0 // (op 132) 
v_accvgpr_read v15, a1 // (op 132) 
v_accvgpr_read v16, a2 // (op 132) 
v_accvgpr_read v17, a3 // (op 132) 
v_mul_f32 v18, v8, v14 // (op 132) 
v_mul_f32 v19, v8, v15 // (op 132) 
v_mul_f32 v20, v8, v16 // (op 132) 
v_mul_f32 v21, v8, v17 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 136) 
v_mul_f32 v15, v5, v11 // (op 136) 
v_mul_f32 v16, v5, v12 // (op 136) 
v_mul_f32 v17, v5, v13 // (op 136) 

v_add_f32 v10, v14, v18 // (op 133) 
v_add_f32 v11, v15, v19 // (op 133) 
v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 




v_mov_b32 v0, v9 // (op 134) reg expression

buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 134) Store value




s_lshl_b32 s0, s2, 7 // (op 528) 
v_add_u32 v0, v6, s0 // (op 528) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 528) Load value

v_accvgpr_read v14, a4 // (op 525) 
v_accvgpr_read v15, a5 // (op 525) 
v_accvgpr_read v16, a6 // (op 525) 
v_accvgpr_read v17, a7 // (op 525) 
v_mul_f32 v18, v8, v14 // (op 525) 
v_mul_f32 v19, v8, v15 // (op 525) 
v_mul_f32 v20, v8, v16 // (op 525) 
v_mul_f32 v21, v8, v17 // (op 525) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 529) 
v_mul_f32 v15, v5, v11 // (op 529) 
v_mul_f32 v16, v5, v12 // (op 529) 
v_mul_f32 v17, v5, v13 // (op 529) 

v_add_f32 v10, v14, v18 // (op 526) 
v_add_f32 v11, v15, v19 // (op 526) 
v_add_f32 v12, v16, v20 // (op 526) 
v_add_f32 v13, v17, v21 // (op 526) 




s_lshl_b32 s0, s4, 7 // (op 527) 
v_add_u32 v0, v9, s0 // (op 527) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 527) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.LRR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__end:
.size RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_, .LRR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__end-RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 76
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
  .amdhsa_accum_offset 68
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
    TransposeA: False
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
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
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [32, 64, 128]
    threads: [256, 1, 1]
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
  - .name: RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 68
    .agpr_count: 8
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
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_2_size_1_11
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
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
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicMultiple_28
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
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
      - .name: Subtract_24
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

