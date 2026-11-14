// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 2346) 
s_mul_hi_i32 s1, s16, s10 // (op 2603) 
v_and_b32 v0, 48, v1 // (op 2408) 
v_and_b32 v3, 63, v1 // (op 2408) 
v_sub_u32 v4, v3, v0 // (op 2408) 

v_lshrrev_b32 v3, 1, v4 // (op 2408) 
v_add_lshl_u32 v5, v0, v3, 6 // (op 2408) 


v_and_b32 v0, 1, v4 // (op 2408) 

v_lshlrev_b32 v3, 3, v0 // (op 2408) 

v_lshrrev_b32 v0, 7, v1 // (op 2408) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2408) 


v_add_u32 v0, v5, v4 // (op 2408) 


v_lshrrev_b32 v3, 6, v1 // (op 2466) 
v_and_b32 v4, 1, v3 // (op 2466) 

v_and_b32 v3, 15, v1 // (op 2466) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 2466) 


v_and_b32 v3, 63, v1 // (op 2466) 
v_lshrrev_b32 v4, 4, v3 // (op 2466) 

v_lshlrev_b32 v3, 5, v4 // (op 2466) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 2466) 


v_lshlrev_b32 v3, 2, v4 // (op 2466) 

v_lshrrev_b32 v4, 3, v3 // (op 2466) 

v_lshrrev_b32 v3, 7, v1 // (op 2524) 
v_and_b32 v5, 15, v1 // (op 2524) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 2524) 


v_and_b32 v3, 63, v1 // (op 2524) 
v_lshrrev_b32 v5, 4, v3 // (op 2524) 

v_lshl_add_u32 v3, v5, 6, v6 // (op 2524) 


v_lshrrev_b32 v5, 6, v1 // (op 2582) 
v_and_b32 v6, 1, v5 // (op 2582) 

v_and_b32 v5, 15, v1 // (op 2582) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2582) 


v_and_b32 v5, 63, v1 // (op 2582) 
v_lshrrev_b32 v6, 4, v5 // (op 2582) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 2582) 


v_and_b32 v6, 48, v1 // (op 2634) 
v_and_b32 v7, 63, v1 // (op 2634) 
v_sub_u32 v8, v7, v6 // (op 2634) 

v_lshrrev_b32 v7, 1, v8 // (op 2634) 
v_add_lshl_u32 v9, v6, v7, 6 // (op 2634) 


v_and_b32 v6, 1, v8 // (op 2634) 

v_lshlrev_b32 v7, 3, v6 // (op 2634) 

v_lshrrev_b32 v6, 7, v1 // (op 2634) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 2634) 


v_add_u32 v6, v9, v8 // (op 2634) 


v_lshrrev_b32 v7, 6, v1 // (op 2663) 
v_and_b32 v8, 1, v7 // (op 2663) 

v_and_b32 v7, 15, v1 // (op 2663) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 2663) 


v_and_b32 v7, 63, v1 // (op 2663) 
v_lshrrev_b32 v8, 4, v7 // (op 2663) 

v_lshlrev_b32 v7, 5, v8 // (op 2663) 

v_lshl_add_u32 v8, v9, 7, v7 // (op 2663) 


v_lshlrev_b32 v7, 2, v8 // (op 2663) 

v_lshrrev_b32 v8, 3, v7 // (op 2663) 

v_lshrrev_b32 v7, 7, v1 // (op 2692) 
v_and_b32 v9, 15, v1 // (op 2692) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 2692) 


v_and_b32 v7, 63, v1 // (op 2692) 
v_lshrrev_b32 v9, 4, v7 // (op 2692) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 2692) 


v_lshrrev_b32 v9, 6, v1 // (op 2721) 
v_and_b32 v10, 1, v9 // (op 2721) 

v_and_b32 v9, 15, v1 // (op 2721) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 2721) 


v_and_b32 v9, 63, v1 // (op 2721) 
v_lshrrev_b32 v10, 4, v9 // (op 2721) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 2721) 


v_lshrrev_b32 v10, 7, v1 // (op 2346) 
s_add_i32 s18, s0, s16 // (op 2346) 

v_and_b32 v11, 63, v1 // (op 2346) 
s_ashr_i32 s0, s18, 31 // (op 2346) 
v_lshrrev_b32 v12, 4, v11 // (op 2346) 
s_cmp_eq_i32 s10, 0 // (op 2346) 
s_mov_b32 s19, scc // (op 2346) 
v_lshlrev_b32 v13, 2, v12 // (op 2346) 

s_cmp_eq_i32 s19, 0 // (op 2346) 
s_cselect_b32 s51, 0, -1 // (op 2346) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 2346) 


s_and_b32 s19, s11, 31 // (op 2346) 
v_lshrrev_b32 v10, 2, v11 // (op 2346) 

v_and_b32 v11, 3, v10 // (op 2346) 

v_lshrrev_b32 v10, 7, v1 // (op 2603) 
v_and_b32 v13, 63, v1 // (op 2603) 
s_lshl_b32 s52, 1, s19 // (op 2346) 
v_and_b32 v14, 3, v1 // (op 2346) 
s_add_i32 s53, s52, s51 // (op 2346) 


s_and_b32 s51, s0, s53 // (op 2346) 


s_add_i32 s0, s1, s16 // (op 2603) 

s_ashr_i32 s1, s0, 31 // (op 2603) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 2346) 


s_add_i32 s52, s18, s51 // (op 2346) 


v_lshrrev_b32 v11, 6, v1 // (op 2346) 
s_ashr_i32 s18, s52, s19 // (op 2346) 


v_and_b32 v14, 1, v11 // (op 2346) 

s_lshl_b32 s19, s11, 24 // (op 2346) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 2346) 


s_ashr_i32 s51, s19, 31 // (op 2346) 

s_xor_b32 s19, s18, s51 // (op 2346) 

s_sub_i32 s18, s19, s51 // (op 2346) 


s_mul_hi_i32 s19, s18, s12 // (op 2346) 
s_add_i32 s51, s19, s18 // (op 2346) 

s_ashr_i32 s19, s51, 31 // (op 2346) 
s_cmp_eq_i32 s12, 0 // (op 2346) 
s_mov_b32 s52, scc // (op 2346) 
s_cmp_eq_i32 s52, 0 // (op 2346) 
s_cselect_b32 s53, 0, -1 // (op 2346) 

s_and_b32 s52, s13, 31 // (op 2346) 
v_lshrrev_b32 v14, 4, v13 // (op 2603) 
s_cmp_eq_i32 s10, 0 // (op 2603) 
s_mov_b32 s54, scc // (op 2603) 
v_lshlrev_b32 v15, 2, v14 // (op 2603) 

s_cmp_eq_i32 s54, 0 // (op 2603) 
s_cselect_b32 s55, 0, -1 // (op 2603) 

s_lshl_b32 s54, 1, s52 // (op 2346) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 2603) 


v_lshrrev_b32 v10, 2, v13 // (op 2603) 

s_add_i32 s56, s54, s53 // (op 2346) 


s_and_b32 s53, s19, s56 // (op 2346) 


s_add_i32 s19, s51, s53 // (op 2346) 


s_ashr_i32 s51, s19, s52 // (op 2346) 


s_lshl_b32 s19, s13, 24 // (op 2346) 
s_ashr_i32 s52, s19, 31 // (op 2346) 

s_xor_b32 s19, s51, s52 // (op 2346) 

s_sub_i32 s51, s19, s52 // (op 2346) 


s_mul_i32 s19, s18, s15 // (op 2346) 
s_sub_u32 s52, s16, s19 // (op 2346) 

s_mul_i32 s19, s51, s15 // (op 2346) 
s_add_u32 s19, s19, s52 // (op 2346) 

s_mul_hi_i32 s52, s16, s9 // (op 2346) 
s_add_i32 s53, s52, s16 // (op 2346) 

s_ashr_i32 s52, s53, 31 // (op 2346) 
s_cmp_eq_i32 s9, 0 // (op 2346) 
s_mov_b32 s54, scc // (op 2346) 
s_cmp_eq_i32 s54, 0 // (op 2346) 
s_cselect_b32 s56, 0, -1 // (op 2346) 

s_and_b32 s54, s11, 31 // (op 2603) 
v_and_b32 v13, 3, v10 // (op 2603) 

s_lshl_b32 s57, 1, s54 // (op 2603) 
v_and_b32 v10, 3, v1 // (op 2603) 
s_add_i32 s58, s57, s55 // (op 2603) 


s_and_b32 s55, s1, s58 // (op 2603) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 2603) 


s_add_i32 s1, s0, s55 // (op 2603) 


v_lshrrev_b32 v10, 6, v1 // (op 2603) 
s_ashr_i32 s0, s1, s54 // (op 2603) 


v_and_b32 v13, 1, v10 // (op 2603) 

s_lshl_b32 s1, s11, 24 // (op 2603) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 2603) 


s_ashr_i32 s54, s1, 31 // (op 2603) 

s_xor_b32 s1, s0, s54 // (op 2603) 

s_sub_i32 s0, s1, s54 // (op 2603) 


s_mul_hi_i32 s1, s0, s12 // (op 2603) 
s_add_i32 s54, s1, s0 // (op 2603) 

s_ashr_i32 s1, s54, 31 // (op 2603) 
s_cmp_eq_i32 s12, 0 // (op 2603) 
s_mov_b32 s55, scc // (op 2603) 
s_cmp_eq_i32 s55, 0 // (op 2603) 
s_cselect_b32 s57, 0, -1 // (op 2603) 

s_and_b32 s55, s13, 31 // (op 2603) 
s_lshl_b32 s58, 1, s55 // (op 2603) 
s_add_i32 s59, s58, s57 // (op 2603) 


s_and_b32 s57, s1, s59 // (op 2603) 


s_add_i32 s1, s54, s57 // (op 2603) 


s_ashr_i32 s54, s1, s55 // (op 2603) 


s_lshl_b32 s1, s13, 24 // (op 2603) 
s_ashr_i32 s55, s1, 31 // (op 2603) 

s_xor_b32 s1, s54, s55 // (op 2603) 

s_sub_i32 s54, s1, s55 // (op 2603) 


s_mul_i32 s1, s0, s15 // (op 2603) 
s_sub_u32 s55, s16, s1 // (op 2603) 

s_mul_i32 s1, s54, s15 // (op 2603) 
s_add_u32 s1, s1, s55 // (op 2603) 

s_mul_hi_i32 s55, s16, s9 // (op 2603) 
s_add_i32 s57, s55, s16 // (op 2603) 

s_ashr_i32 s55, s57, 31 // (op 2603) 
s_cmp_eq_i32 s9, 0 // (op 2603) 
s_mov_b32 s58, scc // (op 2603) 
s_cmp_eq_i32 s58, 0 // (op 2603) 
s_cselect_b32 s59, 0, -1 // (op 2603) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 2346) 
s_lshl_b32 s60, 1, s58 // (op 2346) 
s_add_i32 s61, s60, s56 // (op 2346) 


s_and_b32 s56, s52, s61 // (op 2346) 


s_add_i32 s52, s53, s56 // (op 2346) 


s_ashr_i32 s53, s52, s58 // (op 2346) 


s_lshl_b32 s52, s46, 24 // (op 2346) 
s_ashr_i32 s56, s52, 31 // (op 2346) 

s_xor_b32 s52, s53, s56 // (op 2346) 

s_sub_i32 s53, s52, s56 // (op 2346) 


s_and_b32 s52, s46, 31 // (op 2603) 
s_lshl_b32 s56, 1, s52 // (op 2603) 
s_add_i32 s58, s56, s59 // (op 2603) 


s_and_b32 s56, s55, s58 // (op 2603) 


s_add_i32 s55, s57, s56 // (op 2603) 


s_ashr_i32 s56, s55, s52 // (op 2603) 


s_lshl_b32 s52, s46, 24 // (op 2603) 
s_ashr_i32 s55, s52, 31 // (op 2603) 

s_xor_b32 s52, s56, s55 // (op 2603) 

s_sub_i32 s56, s52, s55 // (op 2603) 


s_add_u32 s58, s6, 64 // (op 2346) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 2346) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2346) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2346) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2346) 

s_mul_i32 s52, s15, s58 // (op 2346) 
s_mul_i32 s55, s8, s52 // (op 2346) 

s_mul_i32 s52, s53, s55 // (op 2346) 

s_sub_u32 s55, s16, s52 // (op 2346) 

s_mov_b32 s52, s55 // (op 2346) 
s_mul_hi_i32 s57, s52, s14 // (op 2346) 
s_add_i32 s60, s57, s52 // (op 2346) 


s_ashr_i32 s52, s60, 31 // (op 2346) 
s_cmp_eq_i32 s14, 0 // (op 2346) 
s_mov_b32 s57, scc // (op 2346) 
s_cmp_eq_i32 s57, 0 // (op 2346) 
s_cselect_b32 s61, 0, -1 // (op 2346) 

s_and_b32 s57, s48, 31 // (op 2346) 
s_lshl_b32 s62, 1, s57 // (op 2346) 
s_add_i32 s63, s62, s61 // (op 2346) 


s_and_b32 s61, s52, s63 // (op 2346) 


s_add_i32 s52, s60, s61 // (op 2346) 


s_ashr_i32 s60, s52, s57 // (op 2346) 


s_lshl_b32 s52, s48, 24 // (op 2346) 
s_ashr_i32 s57, s52, 31 // (op 2346) 

s_xor_b32 s52, s60, s57 // (op 2346) 

s_sub_i32 s60, s52, s57 // (op 2346) 


s_mul_i32 s52, s47, s15 // (op 2346) 
s_add_u32 s62, s20, 64 // (op 2346) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 2346) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2346) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2346) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 2346) 

s_sub_i32 s57, s62, s52 // (op 2346) 

s_mul_i32 s61, s60, s57 // (op 2346) 

s_sub_u32 s57, s55, s61 // (op 2346) 


s_add_u32 s55, s57, s52 // (op 2346) 


s_cmp_eq_i32 s53, 0 // (op 2346) 
s_mov_b32 s52, scc // (op 2346) 

s_cmp_eq_i32 s52, 0 // (op 2346) 
s_cselect_b32 s53, s55, s19 // (op 2346) 


s_mul_i32 s19, s51, s58 // (op 2346) 


v_lshl_add_u32 v13, s53, 6, v12 // (op 2346) 


s_sub_i32 s51, s18, s19 // (op 2346) 


s_cmp_eq_i32 s52, 0 // (op 2346) 
s_cselect_b32 s53, s60, s51 // (op 2346) 



v_lshl_add_u32 v12, s53, 6, v11 // (op 2346) 


v_mul_lo_u32 v11, v12, s2 // (op 2346) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 2346) 


s_add_u32 s18, s6, 64 // (op 2603) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 2603) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 2603) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 2603) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 6 // (op 2603) 

s_mul_i32 s55, s15, s18 // (op 2603) 
s_mul_i32 s51, s8, s55 // (op 2603) 

s_mul_i32 s55, s56, s51 // (op 2603) 

s_sub_u32 s51, s16, s55 // (op 2603) 

s_mov_b32 s55, s51 // (op 2603) 
s_mul_hi_i32 s52, s55, s14 // (op 2603) 
s_add_i32 s53, s52, s55 // (op 2603) 


s_ashr_i32 s52, s53, 31 // (op 2603) 
s_cmp_eq_i32 s14, 0 // (op 2603) 
s_mov_b32 s55, scc // (op 2603) 
s_cmp_eq_i32 s55, 0 // (op 2603) 
s_cselect_b32 s57, 0, -1 // (op 2603) 

s_and_b32 s55, s48, 31 // (op 2603) 
s_lshl_b32 s58, 1, s55 // (op 2603) 
s_add_i32 s59, s58, s57 // (op 2603) 


s_and_b32 s57, s52, s59 // (op 2603) 


s_add_i32 s52, s53, s57 // (op 2603) 


s_ashr_i32 s53, s52, s55 // (op 2603) 


s_lshl_b32 s52, s48, 24 // (op 2603) 
s_ashr_i32 s55, s52, 31 // (op 2603) 

s_xor_b32 s52, s53, s55 // (op 2603) 

s_sub_i32 s53, s52, s55 // (op 2603) 


s_mul_i32 s52, s47, s15 // (op 2603) 
s_add_u32 s58, s20, 64 // (op 2603) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 2603) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2603) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2603) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2603) 

s_sub_i32 s55, s58, s52 // (op 2603) 

s_mul_i32 s57, s53, s55 // (op 2603) 

s_sub_u32 s55, s51, s57 // (op 2603) 


s_add_u32 s51, s55, s52 // (op 2603) 


s_cmp_eq_i32 s56, 0 // (op 2603) 
s_mov_b32 s52, scc // (op 2603) 

s_cmp_eq_i32 s52, 0 // (op 2603) 
s_cselect_b32 s55, s51, s1 // (op 2603) 


s_mul_i32 s1, s54, s18 // (op 2603) 


v_lshl_add_u32 v11, s55, 6, v14 // (op 2603) 


s_sub_i32 s51, s0, s1 // (op 2603) 


s_cmp_eq_i32 s52, 0 // (op 2603) 
s_cselect_b32 s0, s53, s51 // (op 2603) 



v_lshl_add_u32 v13, s0, 6, v10 // (op 2603) 


v_mul_lo_u32 v10, v13, s4 // (op 2603) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 2603) 


s_mul_i32 s1, s2, s7 // (op 2348) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 2348) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 2348) least significant
s_mul_hi_u32 s19, s2, s6 // (op 2348) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2348) most significant: sum
s_add_u32 s1, s1, s19 // (op 2348) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2348) 

s_mul_i32 s1, s4, s7 // (op 2605) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 2605) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 2605) least significant
s_mul_hi_u32 s52, s4, s6 // (op 2605) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2605) most significant: sum
s_add_u32 s1, s1, s52 // (op 2605) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 2605) 

s_mov_b32 s56, s22 // (op 2348) reg expression
s_mov_b32 s57, s23 // (op 2348) reg expression
s_mov_b32 s58, s18 // (op 2348) reg expression
s_mov_b32 s59, 131072 // (op 2348) call()

s_mov_b32 s60, s24 // (op 2605) reg expression

s_mov_b32 s61, s25 // (op 2605) reg expression
s_mov_b32 s62, s52 // (op 2605) reg expression
s_mov_b32 s63, 131072 // (op 2605) call()


v_mov_b32 v10, s49 // (op 2) Move value
v_mov_b32 v11, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 336) call()
v_accvgpr_write a5, 0.00000 // (op 336) call()
v_accvgpr_write a6, 0.00000 // (op 336) call()
v_accvgpr_write a7, 0.00000 // (op 336) call()
v_accvgpr_write a8, 0.00000 // (op 556) call()
v_accvgpr_write a9, 0.00000 // (op 556) call()
v_accvgpr_write a10, 0.00000 // (op 556) call()
v_accvgpr_write a11, 0.00000 // (op 556) call()
v_accvgpr_write a12, 0.00000 // (op 711) call()
v_accvgpr_write a13, 0.00000 // (op 711) call()
v_accvgpr_write a14, 0.00000 // (op 711) call()
v_accvgpr_write a15, 0.00000 // (op 711) call()
s_mul_hi_i32 s0, s16, s10 // (op 2108) 
v_and_b32 v14, -4, v1 // (op 2108) 
s_add_i32 s1, s0, s16 // (op 2108) 

v_sub_u32 v15, v1, v14 // (op 2108) 

s_ashr_i32 s0, s1, 31 // (op 2108) 
v_lshlrev_b32 v14, 4, v15 // (op 2108) 

s_cmp_eq_i32 s10, 0 // (op 2108) 
s_mov_b32 s18, scc // (op 2108) 
v_lshrrev_b32 v15, 2, v1 // (op 2108) 
s_cmp_eq_i32 s18, 0 // (op 2108) 
s_cselect_b32 s19, 0, -1 // (op 2108) 

v_lshlrev_b32 v16, 1, v15 // (op 2108) 

s_and_b32 s18, s11, 31 // (op 2108) 
s_lshl_b32 s22, 1, s18 // (op 2108) 
s_add_i32 s23, s22, s19 // (op 2108) 


s_and_b32 s19, s0, s23 // (op 2108) 


s_add_i32 s0, s1, s19 // (op 2108) 


s_ashr_i32 s1, s0, s18 // (op 2108) 


s_lshl_b32 s0, s11, 24 // (op 2108) 
s_ashr_i32 s18, s0, 31 // (op 2108) 

s_xor_b32 s0, s1, s18 // (op 2108) 

s_sub_i32 s1, s0, s18 // (op 2108) 


s_mul_hi_i32 s0, s1, s12 // (op 2108) 
s_add_i32 s18, s0, s1 // (op 2108) 

s_ashr_i32 s0, s18, 31 // (op 2108) 
s_cmp_eq_i32 s12, 0 // (op 2108) 
s_mov_b32 s19, scc // (op 2108) 
s_cmp_eq_i32 s19, 0 // (op 2108) 
s_cselect_b32 s22, 0, -1 // (op 2108) 

s_and_b32 s19, s13, 31 // (op 2108) 
s_lshl_b32 s23, 1, s19 // (op 2108) 
s_add_i32 s24, s23, s22 // (op 2108) 


s_and_b32 s25, s0, s24 // (op 2108) 


s_add_i32 s0, s18, s25 // (op 2108) 


s_ashr_i32 s18, s0, s19 // (op 2108) 


s_lshl_b32 s0, s13, 24 // (op 2108) 
s_ashr_i32 s19, s0, 31 // (op 2108) 

s_xor_b32 s0, s18, s19 // (op 2108) 

s_sub_i32 s18, s0, s19 // (op 2108) 


s_mul_i32 s0, s1, s15 // (op 2108) 

s_sub_u32 s1, s16, s0 // (op 2108) 

s_mul_i32 s0, s18, s15 // (op 2108) 
s_add_u32 s0, s0, s1 // (op 2108) 


s_mul_hi_i32 s1, s16, s9 // (op 2108) 
s_add_i32 s18, s1, s16 // (op 2108) 

s_ashr_i32 s1, s18, 31 // (op 2108) 
s_cmp_eq_i32 s9, 0 // (op 2108) 
s_mov_b32 s19, scc // (op 2108) 
s_cmp_eq_i32 s19, 0 // (op 2108) 
s_cselect_b32 s22, 0, -1 // (op 2108) 

s_and_b32 s19, s46, 31 // (op 2108) 
s_lshl_b32 s23, 1, s19 // (op 2108) 
s_add_i32 s24, s23, s22 // (op 2108) 


s_and_b32 s25, s1, s24 // (op 2108) 


s_add_i32 s1, s18, s25 // (op 2108) 


s_ashr_i32 s18, s1, s19 // (op 2108) 


s_lshl_b32 s1, s46, 24 // (op 2108) 
s_ashr_i32 s19, s1, 31 // (op 2108) 

s_xor_b32 s1, s18, s19 // (op 2108) 

s_sub_i32 s18, s1, s19 // (op 2108) 


s_add_u32 s22, s6, 64 // (op 2108) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2108) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2108) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2108) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2108) 

s_mul_i32 s1, s15, s22 // (op 2108) 

s_mul_i32 s19, s8, s1 // (op 2108) 

s_mul_i32 s1, s18, s19 // (op 2108) 

s_sub_u32 s19, s16, s1 // (op 2108) 

s_mov_b32 s1, s19 // (op 2108) 
s_mul_hi_i32 s22, s1, s14 // (op 2108) 
s_add_i32 s23, s22, s1 // (op 2108) 


s_ashr_i32 s1, s23, 31 // (op 2108) 
s_cmp_eq_i32 s14, 0 // (op 2108) 
s_mov_b32 s22, scc // (op 2108) 
s_cmp_eq_i32 s22, 0 // (op 2108) 
s_cselect_b32 s24, 0, -1 // (op 2108) 

s_and_b32 s22, s48, 31 // (op 2108) 
s_lshl_b32 s25, 1, s22 // (op 2108) 
s_add_i32 s49, s25, s24 // (op 2108) 


s_and_b32 s24, s1, s49 // (op 2108) 


s_add_i32 s1, s23, s24 // (op 2108) 


s_ashr_i32 s23, s1, s22 // (op 2108) 


s_lshl_b32 s1, s48, 24 // (op 2108) 
s_ashr_i32 s22, s1, 31 // (op 2108) 

s_xor_b32 s1, s23, s22 // (op 2108) 

s_sub_i32 s23, s1, s22 // (op 2108) 


s_mul_i32 s1, s47, s15 // (op 2108) 
s_add_u32 s24, s20, 64 // (op 2108) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2108) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 2108) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 2108) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 2108) 

s_sub_i32 s22, s24, s1 // (op 2108) 

s_mul_i32 s24, s23, s22 // (op 2108) 


s_sub_u32 s25, s19, s24 // (op 2108) 


s_add_u32 s19, s25, s1 // (op 2108) 


s_cmp_eq_i32 s18, 0 // (op 2108) 
s_mov_b32 s1, scc // (op 2108) 

s_cmp_eq_i32 s1, 0 // (op 2108) 
s_cselect_b32 s18, s19, s0 // (op 2108) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 2108) 


v_mul_lo_u32 v14, v16, s26 // (op 2108) 

v_add_u32 v16, v15, v14 // (op 2108) 


s_mul_i32 s1, s26, s31 // (op 2110) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2110) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2110) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2110) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2110) most significant: sum
s_add_u32 s1, s1, s19 // (op 2110) most significant: sum


s_mov_b32 s52, s28 // (op 2110) reg expression
s_mov_b32 s53, s29 // (op 2110) reg expression
s_mov_b32 s54, s0 // (op 2110) reg expression
s_mov_b32 s55, 131072 // (op 2110) call()

s_mul_hi_i32 s0, s16, s10 // (op 2127) 
v_and_b32 v14, -4, v1 // (op 2127) 
s_add_i32 s1, s0, s16 // (op 2127) 

v_sub_u32 v15, v1, v14 // (op 2127) 

s_ashr_i32 s0, s1, 31 // (op 2127) 
v_lshlrev_b32 v14, 4, v15 // (op 2127) 

s_cmp_eq_i32 s10, 0 // (op 2127) 
s_mov_b32 s18, scc // (op 2127) 
v_lshrrev_b32 v15, 2, v1 // (op 2127) 
s_cmp_eq_i32 s18, 0 // (op 2127) 
s_cselect_b32 s19, 0, -1 // (op 2127) 

v_lshlrev_b32 v17, 1, v15 // (op 2127) 

s_and_b32 s18, s11, 31 // (op 2127) 
s_lshl_b32 s22, 1, s18 // (op 2127) 
s_add_i32 s23, s22, s19 // (op 2127) 


s_and_b32 s19, s0, s23 // (op 2127) 


s_add_i32 s0, s1, s19 // (op 2127) 


s_ashr_i32 s1, s0, s18 // (op 2127) 


s_lshl_b32 s0, s11, 24 // (op 2127) 
s_ashr_i32 s18, s0, 31 // (op 2127) 

s_xor_b32 s0, s1, s18 // (op 2127) 

s_sub_i32 s1, s0, s18 // (op 2127) 


s_mul_hi_i32 s0, s1, s12 // (op 2127) 
s_add_i32 s18, s0, s1 // (op 2127) 

s_ashr_i32 s0, s18, 31 // (op 2127) 
s_cmp_eq_i32 s12, 0 // (op 2127) 
s_mov_b32 s19, scc // (op 2127) 
s_cmp_eq_i32 s19, 0 // (op 2127) 
s_cselect_b32 s22, 0, -1 // (op 2127) 

s_and_b32 s19, s13, 31 // (op 2127) 
s_lshl_b32 s23, 1, s19 // (op 2127) 
s_add_i32 s24, s23, s22 // (op 2127) 


s_and_b32 s25, s0, s24 // (op 2127) 


s_add_i32 s0, s18, s25 // (op 2127) 


s_ashr_i32 s18, s0, s19 // (op 2127) 


s_lshl_b32 s0, s13, 24 // (op 2127) 
s_ashr_i32 s19, s0, 31 // (op 2127) 

s_xor_b32 s0, s18, s19 // (op 2127) 

s_sub_i32 s18, s0, s19 // (op 2127) 


s_mul_i32 s0, s1, s15 // (op 2127) 

s_sub_u32 s1, s16, s0 // (op 2127) 

s_mul_i32 s0, s18, s15 // (op 2127) 
s_add_u32 s0, s0, s1 // (op 2127) 


s_mul_hi_i32 s1, s16, s9 // (op 2127) 
s_add_i32 s18, s1, s16 // (op 2127) 

s_ashr_i32 s1, s18, 31 // (op 2127) 
s_cmp_eq_i32 s9, 0 // (op 2127) 
s_mov_b32 s19, scc // (op 2127) 
s_cmp_eq_i32 s19, 0 // (op 2127) 
s_cselect_b32 s22, 0, -1 // (op 2127) 

s_and_b32 s19, s46, 31 // (op 2127) 
s_lshl_b32 s23, 1, s19 // (op 2127) 
s_add_i32 s24, s23, s22 // (op 2127) 


s_and_b32 s25, s1, s24 // (op 2127) 


s_add_i32 s1, s18, s25 // (op 2127) 


s_ashr_i32 s18, s1, s19 // (op 2127) 


s_lshl_b32 s1, s46, 24 // (op 2127) 
s_ashr_i32 s19, s1, 31 // (op 2127) 

s_xor_b32 s1, s18, s19 // (op 2127) 

s_sub_i32 s18, s1, s19 // (op 2127) 


s_add_u32 s50, s6, 64 // (op 2127) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2127) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2127) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2127) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2127) 

s_mul_i32 s1, s15, s24 // (op 2127) 

s_mul_i32 s19, s8, s1 // (op 2127) 

s_mul_i32 s1, s18, s19 // (op 2127) 

s_sub_u32 s19, s16, s1 // (op 2127) 

s_mov_b32 s1, s19 // (op 2127) 
s_mul_hi_i32 s22, s1, s14 // (op 2127) 
s_add_i32 s23, s22, s1 // (op 2127) 


s_ashr_i32 s1, s23, 31 // (op 2127) 
s_cmp_eq_i32 s14, 0 // (op 2127) 
s_mov_b32 s22, scc // (op 2127) 
s_cmp_eq_i32 s22, 0 // (op 2127) 
s_cselect_b32 s24, 0, -1 // (op 2127) 

s_and_b32 s22, s48, 31 // (op 2127) 
s_lshl_b32 s25, 1, s22 // (op 2127) 
s_add_i32 s49, s25, s24 // (op 2127) 


s_and_b32 s24, s1, s49 // (op 2127) 


s_add_i32 s1, s23, s24 // (op 2127) 


s_ashr_i32 s23, s1, s22 // (op 2127) 


s_lshl_b32 s1, s48, 24 // (op 2127) 
s_ashr_i32 s22, s1, 31 // (op 2127) 

s_xor_b32 s1, s23, s22 // (op 2127) 

s_sub_i32 s23, s1, s22 // (op 2127) 


s_mul_i32 s1, s47, s15 // (op 2127) 
s_add_u32 s24, s20, 64 // (op 2127) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2127) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2127) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2127) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2127) 

s_sub_i32 s22, s24, s1 // (op 2127) 

s_mul_i32 s24, s23, s22 // (op 2127) 


s_sub_u32 s25, s19, s24 // (op 2127) 


s_add_u32 s19, s25, s1 // (op 2127) 


s_cmp_eq_i32 s18, 0 // (op 2127) 
s_mov_b32 s1, scc // (op 2127) 

s_cmp_eq_i32 s1, 0 // (op 2127) 
s_cselect_b32 s18, s19, s0 // (op 2127) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 2127) 


v_mul_lo_u32 v14, v17, s26 // (op 2127) 

v_add_u32 v17, v15, v14 // (op 2127) 


s_mul_i32 s1, s26, s31 // (op 2129) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2129) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2129) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2129) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2129) most significant: sum
s_add_u32 s1, s1, s19 // (op 2129) most significant: sum


s_mov_b32 s52, s28 // (op 2129) reg expression
s_mov_b32 s53, s29 // (op 2129) reg expression
s_mov_b32 s54, s0 // (op 2129) reg expression
s_mov_b32 s55, 131072 // (op 2129) call()

s_mul_hi_i32 s0, s16, s10 // (op 2172) 
v_lshrrev_b32 v14, 2, v1 // (op 2172) 
s_add_i32 s1, s0, s16 // (op 2172) 

v_and_b32 v15, -4, v1 // (op 2172) 
s_ashr_i32 s0, s1, 31 // (op 2172) 
v_sub_u32 v18, v1, v15 // (op 2172) 

s_cmp_eq_i32 s10, 0 // (op 2172) 
s_mov_b32 s18, scc // (op 2172) 
v_lshlrev_b32 v15, 5, v18 // (op 2172) 

s_cmp_eq_i32 s18, 0 // (op 2172) 
s_cselect_b32 s19, 0, -1 // (op 2172) 

v_mov_b32 v19, 0 // (op 2172) convert
v_mov_b32 v18, v15 // (op 2172) convert

s_and_b32 s18, s11, 31 // (op 2172) 
s_lshl_b32 s22, 1, s18 // (op 2172) 
s_add_i32 s23, s22, s19 // (op 2172) 


s_and_b32 s19, s0, s23 // (op 2172) 


s_add_i32 s0, s1, s19 // (op 2172) 


s_ashr_i32 s1, s0, s18 // (op 2172) 


s_lshl_b32 s0, s11, 24 // (op 2172) 
s_ashr_i32 s18, s0, 31 // (op 2172) 

s_xor_b32 s0, s1, s18 // (op 2172) 

s_sub_i32 s1, s0, s18 // (op 2172) 


s_mul_hi_i32 s0, s1, s12 // (op 2172) 
s_add_i32 s18, s0, s1 // (op 2172) 

s_ashr_i32 s0, s18, 31 // (op 2172) 
s_cmp_eq_i32 s12, 0 // (op 2172) 
s_mov_b32 s19, scc // (op 2172) 
s_cmp_eq_i32 s19, 0 // (op 2172) 
s_cselect_b32 s22, 0, -1 // (op 2172) 

s_and_b32 s19, s13, 31 // (op 2172) 
s_lshl_b32 s23, 1, s19 // (op 2172) 
s_add_i32 s24, s23, s22 // (op 2172) 


s_and_b32 s25, s0, s24 // (op 2172) 


s_add_i32 s0, s18, s25 // (op 2172) 


s_ashr_i32 s18, s0, s19 // (op 2172) 


s_lshl_b32 s0, s13, 24 // (op 2172) 
s_ashr_i32 s19, s0, 31 // (op 2172) 

s_xor_b32 s0, s18, s19 // (op 2172) 

s_sub_i32 s18, s0, s19 // (op 2172) 


s_add_u32 s50, s6, 64 // (op 2172) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2172) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2172) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2172) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2172) 

s_mul_i32 s0, s18, s24 // (op 2172) 

s_sub_i32 s18, s1, s0 // (op 2172) 


s_mul_hi_i32 s19, s16, s9 // (op 2172) 
s_add_i32 s0, s19, s16 // (op 2172) 

s_ashr_i32 s1, s0, 31 // (op 2172) 
s_cmp_eq_i32 s9, 0 // (op 2172) 
s_mov_b32 s19, scc // (op 2172) 
s_cmp_eq_i32 s19, 0 // (op 2172) 
s_cselect_b32 s22, 0, -1 // (op 2172) 

s_and_b32 s19, s46, 31 // (op 2172) 
s_lshl_b32 s23, 1, s19 // (op 2172) 
s_add_i32 s49, s23, s22 // (op 2172) 


s_and_b32 s22, s1, s49 // (op 2172) 


s_add_i32 s1, s0, s22 // (op 2172) 


s_ashr_i32 s0, s1, s19 // (op 2172) 


s_lshl_b32 s1, s46, 24 // (op 2172) 
s_ashr_i32 s19, s1, 31 // (op 2172) 

s_xor_b32 s1, s0, s19 // (op 2172) 

s_sub_i32 s0, s1, s19 // (op 2172) 


s_mul_i32 s1, s15, s24 // (op 2172) 

s_mul_i32 s19, s8, s1 // (op 2172) 

s_mul_i32 s1, s0, s19 // (op 2172) 

s_sub_u32 s19, s16, s1 // (op 2172) 

s_mul_hi_i32 s1, s19, s14 // (op 2172) 
s_add_i32 s22, s1, s19 // (op 2172) 


s_ashr_i32 s1, s22, 31 // (op 2172) 
s_cmp_eq_i32 s14, 0 // (op 2172) 
s_mov_b32 s19, scc // (op 2172) 
s_cmp_eq_i32 s19, 0 // (op 2172) 
s_cselect_b32 s23, 0, -1 // (op 2172) 

s_and_b32 s19, s48, 31 // (op 2172) 
s_lshl_b32 s24, 1, s19 // (op 2172) 
s_add_i32 s25, s24, s23 // (op 2172) 


s_and_b32 s23, s1, s25 // (op 2172) 


s_add_i32 s1, s22, s23 // (op 2172) 


s_ashr_i32 s22, s1, s19 // (op 2172) 


s_lshl_b32 s1, s48, 24 // (op 2172) 
s_ashr_i32 s19, s1, 31 // (op 2172) 

s_xor_b32 s1, s22, s19 // (op 2172) 

s_sub_i32 s22, s1, s19 // (op 2172) 


s_cmp_eq_i32 s0, 0 // (op 2172) 
s_mov_b32 s1, scc // (op 2172) 

s_cmp_eq_i32 s1, 0 // (op 2172) 
s_cselect_b32 s0, s22, s18 // (op 2172) 



v_lshl_add_u32 v15, s0, 6, v14 // (op 2172) 


v_mul_lo_u32 v21, v15, s33 // (op 2172) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 2172) most significant: high of low * low
v_mul_lo_u32 v20, v15, s32 // (op 2172) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 2172) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 2172) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 2172) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 2172) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 2172) 

v_mov_b32 v14, v18 // (op 2172) convert

s_mul_i32 s1, s32, s7 // (op 2174) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2174) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2174) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2174) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2174) most significant: sum
s_add_u32 s1, s1, s19 // (op 2174) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2174) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2174) 

s_mov_b32 s64, s34 // (op 2174) reg expression
s_mov_b32 s65, s35 // (op 2174) reg expression
s_mov_b32 s66, s0 // (op 2174) reg expression
s_mov_b32 s67, 131072 // (op 2174) call()

s_mul_hi_i32 s0, s16, s10 // (op 2191) 
v_lshrrev_b32 v15, 2, v1 // (op 2191) 
s_add_i32 s1, s0, s16 // (op 2191) 

v_and_b32 v18, -4, v1 // (op 2191) 
s_ashr_i32 s0, s1, 31 // (op 2191) 
v_sub_u32 v19, v1, v18 // (op 2191) 

s_cmp_eq_i32 s10, 0 // (op 2191) 
s_mov_b32 s18, scc // (op 2191) 
v_lshlrev_b32 v18, 5, v19 // (op 2191) 

s_cmp_eq_i32 s18, 0 // (op 2191) 
s_cselect_b32 s19, 0, -1 // (op 2191) 

v_mov_b32 v21, 0 // (op 2191) convert
v_mov_b32 v20, v18 // (op 2191) convert

s_and_b32 s18, s11, 31 // (op 2191) 
s_lshl_b32 s22, 1, s18 // (op 2191) 
s_add_i32 s23, s22, s19 // (op 2191) 


s_and_b32 s19, s0, s23 // (op 2191) 


s_add_i32 s0, s1, s19 // (op 2191) 


s_ashr_i32 s1, s0, s18 // (op 2191) 


s_lshl_b32 s0, s11, 24 // (op 2191) 
s_ashr_i32 s18, s0, 31 // (op 2191) 

s_xor_b32 s0, s1, s18 // (op 2191) 

s_sub_i32 s1, s0, s18 // (op 2191) 


s_mul_hi_i32 s0, s1, s12 // (op 2191) 
s_add_i32 s18, s0, s1 // (op 2191) 

s_ashr_i32 s0, s18, 31 // (op 2191) 
s_cmp_eq_i32 s12, 0 // (op 2191) 
s_mov_b32 s19, scc // (op 2191) 
s_cmp_eq_i32 s19, 0 // (op 2191) 
s_cselect_b32 s22, 0, -1 // (op 2191) 

s_and_b32 s19, s13, 31 // (op 2191) 
s_lshl_b32 s23, 1, s19 // (op 2191) 
s_add_i32 s24, s23, s22 // (op 2191) 


s_and_b32 s25, s0, s24 // (op 2191) 


s_add_i32 s0, s18, s25 // (op 2191) 


s_ashr_i32 s18, s0, s19 // (op 2191) 


s_lshl_b32 s0, s13, 24 // (op 2191) 
s_ashr_i32 s19, s0, 31 // (op 2191) 

s_xor_b32 s0, s18, s19 // (op 2191) 

s_sub_i32 s18, s0, s19 // (op 2191) 


s_add_u32 s50, s6, 64 // (op 2191) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2191) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2191) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2191) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2191) 

s_mul_i32 s0, s18, s24 // (op 2191) 

s_sub_i32 s18, s1, s0 // (op 2191) 


s_mul_hi_i32 s19, s16, s9 // (op 2191) 
s_add_i32 s0, s19, s16 // (op 2191) 

s_ashr_i32 s1, s0, 31 // (op 2191) 
s_cmp_eq_i32 s9, 0 // (op 2191) 
s_mov_b32 s19, scc // (op 2191) 
s_cmp_eq_i32 s19, 0 // (op 2191) 
s_cselect_b32 s22, 0, -1 // (op 2191) 

s_and_b32 s19, s46, 31 // (op 2191) 
s_lshl_b32 s23, 1, s19 // (op 2191) 
s_add_i32 s49, s23, s22 // (op 2191) 


s_and_b32 s22, s1, s49 // (op 2191) 


s_add_i32 s1, s0, s22 // (op 2191) 


s_ashr_i32 s0, s1, s19 // (op 2191) 


s_lshl_b32 s1, s46, 24 // (op 2191) 
s_ashr_i32 s19, s1, 31 // (op 2191) 

s_xor_b32 s1, s0, s19 // (op 2191) 

s_sub_i32 s0, s1, s19 // (op 2191) 


s_mul_i32 s1, s15, s24 // (op 2191) 

s_mul_i32 s19, s8, s1 // (op 2191) 

s_mul_i32 s1, s0, s19 // (op 2191) 

s_sub_u32 s19, s16, s1 // (op 2191) 

s_mul_hi_i32 s1, s19, s14 // (op 2191) 
s_add_i32 s22, s1, s19 // (op 2191) 


s_ashr_i32 s1, s22, 31 // (op 2191) 
s_cmp_eq_i32 s14, 0 // (op 2191) 
s_mov_b32 s19, scc // (op 2191) 
s_cmp_eq_i32 s19, 0 // (op 2191) 
s_cselect_b32 s23, 0, -1 // (op 2191) 

s_and_b32 s19, s48, 31 // (op 2191) 
s_lshl_b32 s24, 1, s19 // (op 2191) 
s_add_i32 s25, s24, s23 // (op 2191) 


s_and_b32 s23, s1, s25 // (op 2191) 


s_add_i32 s1, s22, s23 // (op 2191) 


s_ashr_i32 s22, s1, s19 // (op 2191) 


s_lshl_b32 s1, s48, 24 // (op 2191) 
s_ashr_i32 s19, s1, 31 // (op 2191) 

s_xor_b32 s1, s22, s19 // (op 2191) 

s_sub_i32 s22, s1, s19 // (op 2191) 


s_cmp_eq_i32 s0, 0 // (op 2191) 
s_mov_b32 s1, scc // (op 2191) 

s_cmp_eq_i32 s1, 0 // (op 2191) 
s_cselect_b32 s0, s22, s18 // (op 2191) 



v_lshl_add_u32 v18, s0, 6, v15 // (op 2191) 


v_mul_lo_u32 v23, v18, s33 // (op 2191) most significant: low of low * high
v_mul_hi_u32 v15, v18, s32 // (op 2191) most significant: high of low * low
v_mul_lo_u32 v22, v18, s32 // (op 2191) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 2191) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v22 // (op 2191) least significant half
v_addc_co_u32 v19, s[0:1], v21, v23, s[0:1] // (op 2191) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 2191) 


v_lshrrev_b64 v[20:21], 3, v[18:19] // (op 2191) 

v_mov_b32 v15, v20 // (op 2191) convert

s_mul_i32 s1, s32, s7 // (op 2193) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2193) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2193) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2193) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2193) most significant: sum
s_add_u32 s1, s1, s19 // (op 2193) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2193) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2193) 

s_mov_b32 s64, s34 // (op 2193) reg expression
s_mov_b32 s65, s35 // (op 2193) reg expression
s_mov_b32 s66, s0 // (op 2193) reg expression
s_mov_b32 s67, 131072 // (op 2193) call()

s_mul_hi_i32 s0, s16, s10 // (op 2236) 
v_and_b32 v18, -64, v1 // (op 2236) 
s_add_i32 s1, s0, s16 // (op 2236) 

v_sub_u32 v19, v1, v18 // (op 2236) 

s_ashr_i32 s0, s1, 31 // (op 2236) 
v_lshrrev_b32 v18, 6, v1 // (op 2236) 
s_cmp_eq_i32 s10, 0 // (op 2236) 
s_mov_b32 s18, scc // (op 2236) 
s_cmp_eq_i32 s18, 0 // (op 2236) 
s_cselect_b32 s19, 0, -1 // (op 2236) 

s_and_b32 s18, s11, 31 // (op 2236) 
s_lshl_b32 s22, 1, s18 // (op 2236) 
s_add_i32 s23, s22, s19 // (op 2236) 


s_and_b32 s19, s0, s23 // (op 2236) 


s_add_i32 s0, s1, s19 // (op 2236) 


s_ashr_i32 s1, s0, s18 // (op 2236) 


s_lshl_b32 s0, s11, 24 // (op 2236) 
s_ashr_i32 s18, s0, 31 // (op 2236) 

s_xor_b32 s0, s1, s18 // (op 2236) 

s_sub_i32 s1, s0, s18 // (op 2236) 


s_mul_hi_i32 s0, s1, s12 // (op 2236) 
s_add_i32 s18, s0, s1 // (op 2236) 

s_ashr_i32 s0, s18, 31 // (op 2236) 
s_cmp_eq_i32 s12, 0 // (op 2236) 
s_mov_b32 s19, scc // (op 2236) 
s_cmp_eq_i32 s19, 0 // (op 2236) 
s_cselect_b32 s22, 0, -1 // (op 2236) 

s_and_b32 s19, s13, 31 // (op 2236) 
s_lshl_b32 s23, 1, s19 // (op 2236) 
s_add_i32 s24, s23, s22 // (op 2236) 


s_and_b32 s25, s0, s24 // (op 2236) 


s_add_i32 s0, s18, s25 // (op 2236) 


s_ashr_i32 s18, s0, s19 // (op 2236) 


s_lshl_b32 s0, s13, 24 // (op 2236) 
s_ashr_i32 s19, s0, 31 // (op 2236) 

s_xor_b32 s0, s18, s19 // (op 2236) 

s_sub_i32 s18, s0, s19 // (op 2236) 


s_mul_i32 s0, s1, s15 // (op 2236) 

s_sub_u32 s1, s16, s0 // (op 2236) 

s_mul_i32 s0, s18, s15 // (op 2236) 
s_add_u32 s0, s0, s1 // (op 2236) 


s_mul_hi_i32 s1, s16, s9 // (op 2236) 
s_add_i32 s18, s1, s16 // (op 2236) 

s_ashr_i32 s1, s18, 31 // (op 2236) 
s_cmp_eq_i32 s9, 0 // (op 2236) 
s_mov_b32 s19, scc // (op 2236) 
s_cmp_eq_i32 s19, 0 // (op 2236) 
s_cselect_b32 s22, 0, -1 // (op 2236) 

s_and_b32 s19, s46, 31 // (op 2236) 
s_lshl_b32 s23, 1, s19 // (op 2236) 
s_add_i32 s24, s23, s22 // (op 2236) 


s_and_b32 s25, s1, s24 // (op 2236) 


s_add_i32 s1, s18, s25 // (op 2236) 


s_ashr_i32 s18, s1, s19 // (op 2236) 


s_lshl_b32 s1, s46, 24 // (op 2236) 
s_ashr_i32 s19, s1, 31 // (op 2236) 

s_xor_b32 s1, s18, s19 // (op 2236) 

s_sub_i32 s18, s1, s19 // (op 2236) 


s_add_u32 s50, s6, 64 // (op 2236) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2236) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2236) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2236) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2236) 

s_mul_i32 s1, s15, s24 // (op 2236) 

s_mul_i32 s19, s8, s1 // (op 2236) 

s_mul_i32 s1, s18, s19 // (op 2236) 

s_sub_u32 s19, s16, s1 // (op 2236) 

s_mov_b32 s1, s19 // (op 2236) 
s_mul_hi_i32 s22, s1, s14 // (op 2236) 
s_add_i32 s23, s22, s1 // (op 2236) 


s_ashr_i32 s1, s23, 31 // (op 2236) 
s_cmp_eq_i32 s14, 0 // (op 2236) 
s_mov_b32 s22, scc // (op 2236) 
s_cmp_eq_i32 s22, 0 // (op 2236) 
s_cselect_b32 s24, 0, -1 // (op 2236) 

s_and_b32 s22, s48, 31 // (op 2236) 
s_lshl_b32 s25, 1, s22 // (op 2236) 
s_add_i32 s49, s25, s24 // (op 2236) 


s_and_b32 s24, s1, s49 // (op 2236) 


s_add_i32 s1, s23, s24 // (op 2236) 


s_ashr_i32 s23, s1, s22 // (op 2236) 


s_lshl_b32 s1, s48, 24 // (op 2236) 
s_ashr_i32 s22, s1, 31 // (op 2236) 

s_xor_b32 s1, s23, s22 // (op 2236) 

s_sub_i32 s23, s1, s22 // (op 2236) 


s_mul_i32 s1, s47, s15 // (op 2236) 
s_add_u32 s24, s20, 64 // (op 2236) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2236) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2236) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2236) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2236) 

s_sub_i32 s22, s24, s1 // (op 2236) 

s_mul_i32 s24, s23, s22 // (op 2236) 


s_sub_u32 s25, s19, s24 // (op 2236) 


s_add_u32 s19, s25, s1 // (op 2236) 


s_cmp_eq_i32 s18, 0 // (op 2236) 
s_mov_b32 s1, scc // (op 2236) 

s_cmp_eq_i32 s1, 0 // (op 2236) 
s_cselect_b32 s18, s19, s0 // (op 2236) 



v_lshl_add_u32 v20, s18, 6, v19 // (op 2236) 


v_mul_lo_u32 v19, v18, s36 // (op 2236) 

v_add_u32 v18, v20, v19 // (op 2236) 


s_mul_i32 s1, s36, s41 // (op 2238) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2238) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2238) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2238) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2238) most significant: sum
s_add_u32 s1, s1, s19 // (op 2238) most significant: sum


s_mov_b32 s68, s38 // (op 2238) reg expression
s_mov_b32 s69, s39 // (op 2238) reg expression
s_mov_b32 s70, s0 // (op 2238) reg expression
s_mov_b32 s71, 131072 // (op 2238) call()

s_mul_hi_i32 s0, s16, s10 // (op 2255) 
v_and_b32 v19, -64, v1 // (op 2255) 
s_add_i32 s1, s0, s16 // (op 2255) 

v_sub_u32 v20, v1, v19 // (op 2255) 

s_ashr_i32 s0, s1, 31 // (op 2255) 
v_lshrrev_b32 v19, 6, v1 // (op 2255) 
s_cmp_eq_i32 s10, 0 // (op 2255) 
s_mov_b32 s18, scc // (op 2255) 
s_cmp_eq_i32 s18, 0 // (op 2255) 
s_cselect_b32 s19, 0, -1 // (op 2255) 

s_and_b32 s18, s11, 31 // (op 2255) 
s_lshl_b32 s22, 1, s18 // (op 2255) 
s_add_i32 s23, s22, s19 // (op 2255) 


s_and_b32 s19, s0, s23 // (op 2255) 


s_add_i32 s0, s1, s19 // (op 2255) 


s_ashr_i32 s1, s0, s18 // (op 2255) 


s_lshl_b32 s0, s11, 24 // (op 2255) 
s_ashr_i32 s18, s0, 31 // (op 2255) 

s_xor_b32 s0, s1, s18 // (op 2255) 

s_sub_i32 s1, s0, s18 // (op 2255) 


s_mul_hi_i32 s0, s1, s12 // (op 2255) 
s_add_i32 s18, s0, s1 // (op 2255) 

s_ashr_i32 s0, s18, 31 // (op 2255) 
s_cmp_eq_i32 s12, 0 // (op 2255) 
s_mov_b32 s19, scc // (op 2255) 
s_cmp_eq_i32 s19, 0 // (op 2255) 
s_cselect_b32 s22, 0, -1 // (op 2255) 

s_and_b32 s19, s13, 31 // (op 2255) 
s_lshl_b32 s23, 1, s19 // (op 2255) 
s_add_i32 s24, s23, s22 // (op 2255) 


s_and_b32 s25, s0, s24 // (op 2255) 


s_add_i32 s0, s18, s25 // (op 2255) 


s_ashr_i32 s18, s0, s19 // (op 2255) 


s_lshl_b32 s0, s13, 24 // (op 2255) 
s_ashr_i32 s19, s0, 31 // (op 2255) 

s_xor_b32 s0, s18, s19 // (op 2255) 

s_sub_i32 s18, s0, s19 // (op 2255) 


s_mul_i32 s0, s1, s15 // (op 2255) 

s_sub_u32 s1, s16, s0 // (op 2255) 

s_mul_i32 s0, s18, s15 // (op 2255) 
s_add_u32 s0, s0, s1 // (op 2255) 


s_mul_hi_i32 s1, s16, s9 // (op 2255) 
s_add_i32 s18, s1, s16 // (op 2255) 

s_ashr_i32 s1, s18, 31 // (op 2255) 
s_cmp_eq_i32 s9, 0 // (op 2255) 
s_mov_b32 s19, scc // (op 2255) 
s_cmp_eq_i32 s19, 0 // (op 2255) 
s_cselect_b32 s22, 0, -1 // (op 2255) 

s_and_b32 s19, s46, 31 // (op 2255) 
s_lshl_b32 s23, 1, s19 // (op 2255) 
s_add_i32 s24, s23, s22 // (op 2255) 


s_and_b32 s25, s1, s24 // (op 2255) 


s_add_i32 s1, s18, s25 // (op 2255) 


s_ashr_i32 s18, s1, s19 // (op 2255) 


s_lshl_b32 s1, s46, 24 // (op 2255) 
s_ashr_i32 s19, s1, 31 // (op 2255) 

s_xor_b32 s1, s18, s19 // (op 2255) 

s_sub_i32 s18, s1, s19 // (op 2255) 


s_add_u32 s50, s6, 64 // (op 2255) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2255) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2255) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2255) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2255) 

s_mul_i32 s1, s15, s24 // (op 2255) 

s_mul_i32 s19, s8, s1 // (op 2255) 

s_mul_i32 s1, s18, s19 // (op 2255) 

s_sub_u32 s19, s16, s1 // (op 2255) 

s_mov_b32 s1, s19 // (op 2255) 
s_mul_hi_i32 s22, s1, s14 // (op 2255) 
s_add_i32 s23, s22, s1 // (op 2255) 


s_ashr_i32 s1, s23, 31 // (op 2255) 
s_cmp_eq_i32 s14, 0 // (op 2255) 
s_mov_b32 s22, scc // (op 2255) 
s_cmp_eq_i32 s22, 0 // (op 2255) 
s_cselect_b32 s24, 0, -1 // (op 2255) 

s_and_b32 s22, s48, 31 // (op 2255) 
s_lshl_b32 s25, 1, s22 // (op 2255) 
s_add_i32 s49, s25, s24 // (op 2255) 


s_and_b32 s24, s1, s49 // (op 2255) 


s_add_i32 s1, s23, s24 // (op 2255) 


s_ashr_i32 s23, s1, s22 // (op 2255) 


s_lshl_b32 s1, s48, 24 // (op 2255) 
s_ashr_i32 s22, s1, 31 // (op 2255) 

s_xor_b32 s1, s23, s22 // (op 2255) 

s_sub_i32 s23, s1, s22 // (op 2255) 


s_mul_i32 s1, s47, s15 // (op 2255) 
s_add_u32 s24, s20, 64 // (op 2255) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2255) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2255) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2255) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2255) 

s_sub_i32 s22, s24, s1 // (op 2255) 

s_mul_i32 s24, s23, s22 // (op 2255) 


s_sub_u32 s25, s19, s24 // (op 2255) 


s_add_u32 s19, s25, s1 // (op 2255) 


s_cmp_eq_i32 s18, 0 // (op 2255) 
s_mov_b32 s1, scc // (op 2255) 

s_cmp_eq_i32 s1, 0 // (op 2255) 
s_cselect_b32 s18, s19, s0 // (op 2255) 



v_lshl_add_u32 v21, s18, 6, v20 // (op 2255) 


v_mul_lo_u32 v20, v19, s36 // (op 2255) 

v_add_u32 v19, v21, v20 // (op 2255) 


s_mul_i32 s1, s36, s41 // (op 2257) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2257) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2257) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2257) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2257) most significant: sum
s_add_u32 s1, s1, s19 // (op 2257) most significant: sum


s_mov_b32 s68, s38 // (op 2257) reg expression
s_mov_b32 s69, s39 // (op 2257) reg expression
s_mov_b32 s70, s0 // (op 2257) reg expression
s_mov_b32 s71, 131072 // (op 2257) call()

s_mul_hi_i32 s0, s16, s10 // (op 2300) 
v_lshrrev_b32 v20, 2, v1 // (op 2300) 
s_add_i32 s1, s0, s16 // (op 2300) 

v_and_b32 v21, -4, v1 // (op 2300) 
s_ashr_i32 s0, s1, 31 // (op 2300) 
v_sub_u32 v22, v1, v21 // (op 2300) 

s_cmp_eq_i32 s10, 0 // (op 2300) 
s_mov_b32 s18, scc // (op 2300) 
s_cmp_eq_i32 s18, 0 // (op 2300) 
s_cselect_b32 s19, 0, -1 // (op 2300) 

s_and_b32 s18, s11, 31 // (op 2300) 
s_lshl_b32 s22, 1, s18 // (op 2300) 
s_add_i32 s23, s22, s19 // (op 2300) 


s_and_b32 s19, s0, s23 // (op 2300) 


s_add_i32 s0, s1, s19 // (op 2300) 


s_ashr_i32 s1, s0, s18 // (op 2300) 


s_lshl_b32 s0, s11, 24 // (op 2300) 
s_ashr_i32 s18, s0, 31 // (op 2300) 

s_xor_b32 s0, s1, s18 // (op 2300) 

s_sub_i32 s1, s0, s18 // (op 2300) 


s_mul_hi_i32 s0, s1, s12 // (op 2300) 
s_add_i32 s18, s0, s1 // (op 2300) 

s_ashr_i32 s0, s18, 31 // (op 2300) 
s_cmp_eq_i32 s12, 0 // (op 2300) 
s_mov_b32 s19, scc // (op 2300) 
s_cmp_eq_i32 s19, 0 // (op 2300) 
s_cselect_b32 s22, 0, -1 // (op 2300) 

s_and_b32 s19, s13, 31 // (op 2300) 
s_lshl_b32 s23, 1, s19 // (op 2300) 
s_add_i32 s24, s23, s22 // (op 2300) 


s_and_b32 s25, s0, s24 // (op 2300) 


s_add_i32 s0, s18, s25 // (op 2300) 


s_ashr_i32 s18, s0, s19 // (op 2300) 


s_lshl_b32 s0, s13, 24 // (op 2300) 
s_ashr_i32 s19, s0, 31 // (op 2300) 

s_xor_b32 s0, s18, s19 // (op 2300) 

s_sub_i32 s18, s0, s19 // (op 2300) 


s_add_u32 s50, s6, 64 // (op 2300) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2300) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2300) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2300) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2300) 

s_mul_i32 s0, s18, s24 // (op 2300) 

s_sub_i32 s18, s1, s0 // (op 2300) 


s_mul_hi_i32 s19, s16, s9 // (op 2300) 
s_add_i32 s0, s19, s16 // (op 2300) 

s_ashr_i32 s1, s0, 31 // (op 2300) 
s_cmp_eq_i32 s9, 0 // (op 2300) 
s_mov_b32 s19, scc // (op 2300) 
s_cmp_eq_i32 s19, 0 // (op 2300) 
s_cselect_b32 s22, 0, -1 // (op 2300) 

s_and_b32 s19, s46, 31 // (op 2300) 
s_lshl_b32 s23, 1, s19 // (op 2300) 
s_add_i32 s49, s23, s22 // (op 2300) 


s_and_b32 s22, s1, s49 // (op 2300) 


s_add_i32 s1, s0, s22 // (op 2300) 


s_ashr_i32 s0, s1, s19 // (op 2300) 


s_lshl_b32 s1, s46, 24 // (op 2300) 
s_ashr_i32 s19, s1, 31 // (op 2300) 

s_xor_b32 s1, s0, s19 // (op 2300) 

s_sub_i32 s0, s1, s19 // (op 2300) 


s_mul_i32 s1, s15, s24 // (op 2300) 

s_mul_i32 s19, s8, s1 // (op 2300) 

s_mul_i32 s1, s0, s19 // (op 2300) 

s_sub_u32 s19, s16, s1 // (op 2300) 

s_mul_hi_i32 s1, s19, s14 // (op 2300) 
s_add_i32 s22, s1, s19 // (op 2300) 


s_ashr_i32 s1, s22, 31 // (op 2300) 
s_cmp_eq_i32 s14, 0 // (op 2300) 
s_mov_b32 s19, scc // (op 2300) 
s_cmp_eq_i32 s19, 0 // (op 2300) 
s_cselect_b32 s23, 0, -1 // (op 2300) 

s_and_b32 s19, s48, 31 // (op 2300) 
s_lshl_b32 s24, 1, s19 // (op 2300) 
s_add_i32 s25, s24, s23 // (op 2300) 


s_and_b32 s23, s1, s25 // (op 2300) 


s_add_i32 s1, s22, s23 // (op 2300) 


s_ashr_i32 s22, s1, s19 // (op 2300) 


s_lshl_b32 s1, s48, 24 // (op 2300) 
s_ashr_i32 s19, s1, 31 // (op 2300) 

s_xor_b32 s1, s22, s19 // (op 2300) 

s_sub_i32 s22, s1, s19 // (op 2300) 


s_cmp_eq_i32 s0, 0 // (op 2300) 
s_mov_b32 s1, scc // (op 2300) 

s_cmp_eq_i32 s1, 0 // (op 2300) 
s_cselect_b32 s0, s22, s18 // (op 2300) 



v_lshl_add_u32 v21, s0, 6, v20 // (op 2300) 


v_mul_lo_u32 v20, v21, s42 // (op 2300) 

v_add_u32 v21, v22, v20 // (op 2300) 


s_mul_i32 s1, s42, s7 // (op 2302) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2302) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2302) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2302) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2302) most significant: sum
s_add_u32 s1, s1, s19 // (op 2302) most significant: sum


s_mov_b32 s72, s44 // (op 2302) reg expression
s_mov_b32 s73, s45 // (op 2302) reg expression
s_mov_b32 s74, s0 // (op 2302) reg expression
s_mov_b32 s75, 131072 // (op 2302) call()

s_mul_hi_i32 s0, s16, s10 // (op 2319) 
v_lshrrev_b32 v20, 2, v1 // (op 2319) 
s_add_i32 s1, s0, s16 // (op 2319) 

v_and_b32 v22, -4, v1 // (op 2319) 
s_ashr_i32 s0, s1, 31 // (op 2319) 
v_sub_u32 v23, v1, v22 // (op 2319) 

s_cmp_eq_i32 s10, 0 // (op 2319) 
s_mov_b32 s18, scc // (op 2319) 
s_cmp_eq_i32 s18, 0 // (op 2319) 
s_cselect_b32 s19, 0, -1 // (op 2319) 

s_and_b32 s18, s11, 31 // (op 2319) 
s_lshl_b32 s22, 1, s18 // (op 2319) 
s_add_i32 s23, s22, s19 // (op 2319) 


s_and_b32 s19, s0, s23 // (op 2319) 


s_add_i32 s0, s1, s19 // (op 2319) 


s_ashr_i32 s1, s0, s18 // (op 2319) 


s_lshl_b32 s0, s11, 24 // (op 2319) 
s_ashr_i32 s18, s0, 31 // (op 2319) 

s_xor_b32 s0, s1, s18 // (op 2319) 

s_sub_i32 s1, s0, s18 // (op 2319) 


s_mul_hi_i32 s0, s1, s12 // (op 2319) 
s_add_i32 s18, s0, s1 // (op 2319) 

s_ashr_i32 s0, s18, 31 // (op 2319) 
s_cmp_eq_i32 s12, 0 // (op 2319) 
s_mov_b32 s19, scc // (op 2319) 
s_cmp_eq_i32 s19, 0 // (op 2319) 
s_cselect_b32 s22, 0, -1 // (op 2319) 

s_and_b32 s19, s13, 31 // (op 2319) 
s_lshl_b32 s23, 1, s19 // (op 2319) 
s_add_i32 s24, s23, s22 // (op 2319) 


s_and_b32 s25, s0, s24 // (op 2319) 


s_add_i32 s0, s18, s25 // (op 2319) 


s_ashr_i32 s18, s0, s19 // (op 2319) 


s_lshl_b32 s0, s13, 24 // (op 2319) 
s_ashr_i32 s19, s0, 31 // (op 2319) 

s_xor_b32 s0, s18, s19 // (op 2319) 

s_sub_i32 s18, s0, s19 // (op 2319) 


s_add_u32 s50, s6, 64 // (op 2319) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2319) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2319) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2319) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2319) 

s_mul_i32 s0, s18, s24 // (op 2319) 

s_sub_i32 s18, s1, s0 // (op 2319) 


s_mul_hi_i32 s19, s16, s9 // (op 2319) 
s_add_i32 s0, s19, s16 // (op 2319) 

s_ashr_i32 s1, s0, 31 // (op 2319) 
s_cmp_eq_i32 s9, 0 // (op 2319) 
s_mov_b32 s19, scc // (op 2319) 
s_cmp_eq_i32 s19, 0 // (op 2319) 
s_cselect_b32 s22, 0, -1 // (op 2319) 

s_and_b32 s19, s46, 31 // (op 2319) 
s_lshl_b32 s23, 1, s19 // (op 2319) 
s_add_i32 s49, s23, s22 // (op 2319) 


s_and_b32 s22, s1, s49 // (op 2319) 


s_add_i32 s1, s0, s22 // (op 2319) 


s_ashr_i32 s0, s1, s19 // (op 2319) 


s_lshl_b32 s1, s46, 24 // (op 2319) 
s_ashr_i32 s19, s1, 31 // (op 2319) 

s_xor_b32 s1, s0, s19 // (op 2319) 

s_sub_i32 s0, s1, s19 // (op 2319) 


s_mul_i32 s1, s15, s24 // (op 2319) 

s_mul_i32 s19, s8, s1 // (op 2319) 

s_mul_i32 s1, s0, s19 // (op 2319) 

s_sub_u32 s19, s16, s1 // (op 2319) 

s_mul_hi_i32 s1, s19, s14 // (op 2319) 
s_add_i32 s22, s1, s19 // (op 2319) 


s_ashr_i32 s1, s22, 31 // (op 2319) 
s_cmp_eq_i32 s14, 0 // (op 2319) 
s_mov_b32 s19, scc // (op 2319) 
s_cmp_eq_i32 s19, 0 // (op 2319) 
s_cselect_b32 s23, 0, -1 // (op 2319) 

s_and_b32 s19, s48, 31 // (op 2319) 
s_lshl_b32 s24, 1, s19 // (op 2319) 
s_add_i32 s25, s24, s23 // (op 2319) 


s_and_b32 s23, s1, s25 // (op 2319) 


s_add_i32 s1, s22, s23 // (op 2319) 


s_ashr_i32 s22, s1, s19 // (op 2319) 


s_lshl_b32 s1, s48, 24 // (op 2319) 
s_ashr_i32 s19, s1, 31 // (op 2319) 

s_xor_b32 s1, s22, s19 // (op 2319) 

s_sub_i32 s22, s1, s19 // (op 2319) 


s_cmp_eq_i32 s0, 0 // (op 2319) 
s_mov_b32 s1, scc // (op 2319) 

s_cmp_eq_i32 s1, 0 // (op 2319) 
s_cselect_b32 s0, s22, s18 // (op 2319) 



v_lshl_add_u32 v22, s0, 6, v20 // (op 2319) 


v_mul_lo_u32 v20, v22, s42 // (op 2319) 

v_add_u32 v22, v23, v20 // (op 2319) 


s_mul_i32 s1, s42, s7 // (op 2321) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2321) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2321) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2321) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2321) most significant: sum
s_add_u32 s1, s1, s19 // (op 2321) most significant: sum


s_mov_b32 s72, s44 // (op 2321) reg expression
s_mov_b32 s73, s45 // (op 2321) reg expression
s_mov_b32 s74, s0 // (op 2321) reg expression
s_mov_b32 s75, 131072 // (op 2321) call()




buffer_load_dwordx4 v[24:27], v14, s[64:67], 0 offen // (op 1940) Load value



buffer_load_ubyte v20, v21, s[72:75], 0 offen offset: 0 // (op 1943) Load value

v_mov_b64 v[28:29], s[26:27] // (op 1946) convert

v_mov_b32 v23, v16 // (op 1946) reg expression
buffer_load_dwordx4 v[30:33], v23, s[52:55], 0 offen // (op 1946) Load value
v_add_u32 v23, v23, v28 // (op 1946) 
buffer_load_dwordx4 v[34:37], v23, s[52:55], 0 offen // (op 1946) Load value




v_mov_b32 v23, v18 // (op 1949) reg expression

buffer_load_ubyte v28, v23, s[68:71], 0 offen offset: 0 // (op 1949) Load value




buffer_load_dwordx4 v[38:41], v14, s[64:67], 0 offen offset:64 // (op 1952) Load value




buffer_load_ubyte v14, v21, s[72:75], 0 offen offset:4 // (op 1955) Load value


v_mov_b64 v[42:43], s[26:27] // (op 1958) convert

s_lshl_b32 s0, s26, 7 // (op 1958) 
v_add_u32 v21, v16, s0 // (op 1958) 

buffer_load_dwordx4 v[44:47], v21, s[52:55], 0 offen // (op 1958) Load value
v_add_u32 v21, v21, v42 // (op 1958) 
buffer_load_dwordx4 v[48:51], v21, s[52:55], 0 offen // (op 1958) Load value





s_lshl_b32 s0, s36, 2 // (op 1961) 
v_add_u32 v16, v18, s0 // (op 1961) 


buffer_load_ubyte v21, v16, s[68:71], 0 offen offset: 0 // (op 1961) Load value


v_and_b32 v16, -4, v1 // (op 2449) 
v_sub_u32 v18, v1, v16 // (op 2449) 

v_lshlrev_b32 v16, 5, v18 // (op 2449) 

v_lshrrev_b32 v18, 2, v1 // (op 2449) 
v_lshl_add_u32 v23, v18, 7, v16 // (op 2449) 


v_lshlrev_b32 v16, 2, v23 // (op 2449) 

v_lshrrev_b32 v18, 3, v16 // (op 2449) 




s_waitcnt vmcnt(9)
ds_write_b128 v18, v[24:27] // (op 1963) Store local data 

v_and_b32 v16, -4, v1 // (op 2565) 
v_sub_u32 v18, v1, v16 // (op 2565) 

v_and_b32 v16, 4294967292, v1 // (op 2565) 
v_add_u32 v23, v16, v18 // (op 2565) 





s_waitcnt vmcnt(8)
ds_write_b8 v23, v20 offset:4096 // (op 1964) Store local data 

v_and_b32 v16, -4, v1 // (op 2391) 
v_sub_u32 v18, v1, v16 // (op 2391) 

v_lshlrev_b32 v16, 4, v18 // (op 2391) 

v_lshrrev_b32 v18, 2, v1 // (op 2391) 
v_lshl_add_u32 v23, v18, 7, v16 // (op 2391) 





s_waitcnt vmcnt(7)
ds_write_b128 v23, v[30:33] offset:4352 // (op 1965) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v23, v[34:37] offset:4416 // (op 1965) Store local data 

v_and_b32 v16, -64, v1 // (op 2507) 
v_sub_u32 v18, v1, v16 // (op 2507) 

v_and_b32 v16, 4294967232, v1 // (op 2507) 
v_add_u32 v23, v16, v18 // (op 2507) 





s_waitcnt vmcnt(5)
ds_write_b8 v23, v28 offset:12544 // (op 1966) Store local data 

v_and_b32 v16, -4, v1 // (op 2362) 
v_sub_u32 v18, v1, v16 // (op 2362) 

v_lshlrev_b32 v16, 4, v18 // (op 2362) 

v_lshrrev_b32 v18, 2, v1 // (op 2362) 
v_lshl_add_u32 v23, v18, 7, v16 // (op 2362) 


v_and_b32 v16, -4, v1 // (op 2420) 
v_sub_u32 v18, v1, v16 // (op 2420) 

v_lshlrev_b32 v16, 5, v18 // (op 2420) 

v_lshrrev_b32 v18, 2, v1 // (op 2420) 
v_lshl_add_u32 v29, v18, 7, v16 // (op 2420) 


v_lshlrev_b32 v16, 2, v29 // (op 2420) 

v_lshrrev_b32 v18, 3, v16 // (op 2420) 

v_and_b32 v16, -64, v1 // (op 2478) 
v_sub_u32 v29, v1, v16 // (op 2478) 

v_and_b32 v16, 4294967232, v1 // (op 2478) 
v_add_u32 v42, v16, v29 // (op 2478) 


v_and_b32 v16, -4, v1 // (op 2536) 
v_sub_u32 v29, v1, v16 // (op 2536) 

v_and_b32 v16, 4294967292, v1 // (op 2536) 
v_add_u32 v43, v16, v29 // (op 2536) 


v_and_b32 v16, -4, v1 // (op 2617) 
v_sub_u32 v29, v1, v16 // (op 2617) 

v_lshlrev_b32 v16, 4, v29 // (op 2617) 

v_lshrrev_b32 v29, 2, v1 // (op 2617) 
v_lshl_add_u32 v52, v29, 7, v16 // (op 2617) 


v_and_b32 v16, -4, v1 // (op 2646) 
v_sub_u32 v29, v1, v16 // (op 2646) 

v_lshlrev_b32 v16, 5, v29 // (op 2646) 

v_lshrrev_b32 v29, 2, v1 // (op 2646) 
v_lshl_add_u32 v53, v29, 7, v16 // (op 2646) 


v_lshlrev_b32 v16, 2, v53 // (op 2646) 

v_lshrrev_b32 v29, 3, v16 // (op 2646) 

v_and_b32 v16, -64, v1 // (op 2675) 
v_sub_u32 v53, v1, v16 // (op 2675) 

v_and_b32 v16, 4294967232, v1 // (op 2675) 
v_add_u32 v54, v16, v53 // (op 2675) 


v_and_b32 v16, -4, v1 // (op 2704) 
v_sub_u32 v53, v1, v16 // (op 2704) 

v_and_b32 v16, 4294967292, v1 // (op 2704) 
v_add_u32 v55, v16, v53 // (op 2704) 


s_mov_b32 s0, 0 // (op 589) call()
s_ashr_i64 s[18:19], s[30:31], 7 // (op 588) 
s_mov_b32 s1, s18 // (op 588) convert

s_and_b32 s18, s1, 4294967294 // (op 588) 

s_cmp_lt_u32 s0, s18 // (op 588) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_588 // (op 588) Condition: Top (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_588 if false)
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_588:


s_barrier  // (op 2772) 
s_add_u32 s0, s0, 2 // (op 590) 



v_mov_b32 v16, v15 // (op 10) 

buffer_load_dwordx4 v[24:27], v16, s[64:67], 0 offen offset:128 // (op 10) Load value
ds_read_b64_tr_b8 v[56:57], v0 offset:4352 // (op 70) Transpose load local data 


ds_read_b64_tr_b8 v[58:59], v0 offset:4864 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:8448 // (op 70) Transpose load local data 
v_mov_b32 v53, v22 // (op 25) 

buffer_load_ubyte v20, v53, s[72:75], 0 offen offset:8 // (op 25) Load value
ds_read_b64_tr_b8 v[62:63], v0 offset:8960 // (op 70) Transpose load local data 

s_barrier  // (op 2785) 
v_mov_b64 v[64:65], s[26:27] // (op 4) convert




v_mov_b32 v66, v17 // (op 4) 
s_lshl_b32 s1, s26, 8 // (op 4) 
v_add_u32 v67, v66, s1 // (op 4) 

ds_read_b128 v[68:71], v4 // (op 75) Load local data 
buffer_load_dwordx4 v[30:33], v67, s[52:55], 0 offen // (op 4) Load value
s_barrier  // (op 2799) 
v_add_u32 v67, v67, v64 // (op 4) 


ds_read_u8 v72, v3 offset:12544 // (op 80) Load local data 
buffer_load_dwordx4 v[34:37], v67, s[52:55], 0 offen // (op 4) Load value




s_barrier  // (op 2813) 
v_mov_b32 v67, v19 // (op 17) 



ds_read_u8 v64, v5 offset:4096 // (op 85) Load local data 
s_lshl_b32 s1, s36, 3 // (op 17) 



v_add_u32 v65, v67, s1 // (op 17) 


buffer_load_ubyte v28, v65, s[68:71], 0 offen offset: 0 // (op 17) Load value

ds_read_b128 v[74:77], v4 offset:2048 // (op 371) Load local data 



s_barrier  // (op 2836) 
ds_read_u8 v65, v5 offset:4224 // (op 375) Load local data 






s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[68:71], a[0:3], v72, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 88) 
ds_read_b64_tr_b8 v[78:79], v0 offset:4384 // (op 622) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v0 offset:4896 // (op 622) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v0 offset:8480 // (op 622) Transpose load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[74:77], a[4:7], v72, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 376) 
ds_read_b64_tr_b8 v[84:85], v0 offset:8992 // (op 622) Transpose load local data 




ds_read_u8 v56, v3 offset:12576 // (op 625) Load local data 
s_waitcnt vmcnt(9)
ds_write_b128 v29, v[38:41] offset:12800 // (op 253) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:85], v[68:71], a[8:11], v56, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 673) 


s_barrier  // (op 2848) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:85], v[74:77], a[12:15], v56, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 627) 
s_waitcnt vmcnt(8)
ds_write_b8 v55, v14 offset:16896 // (op 255) Store local data 




s_barrier  // (op 2830) 



s_waitcnt vmcnt(7)
ds_write_b128 v52, v[44:47] offset:17152 // (op 243) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v52, v[48:51] offset:17216 // (op 243) Store local data 
s_barrier  // (op 2842) 



s_waitcnt vmcnt(5)
ds_write_b8 v54, v21 offset:25344 // (op 254) Store local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 2827) 
buffer_load_dwordx4 v[38:41], v16, s[64:67], 0 offen offset:192 // (op 256) Load value




ds_read_b64_tr_b8 v[56:57], v6 offset:17152 // (op 245) Transpose load local data 



ds_read_b64_tr_b8 v[58:59], v6 offset:17664 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v6 offset:21248 // (op 245) Transpose load local data 
v_add_u32 v15, 128, v15 // (op 2183) 
ds_read_b64_tr_b8 v[62:63], v6 offset:21760 // (op 245) Transpose load local data 
buffer_load_ubyte v14, v53, s[72:75], 0 offen offset:12 // (op 258) Load value

s_barrier  // (op 2833) 




v_add_u32 v22, 8, v22 // (op 2311) 
v_mov_b64 v[64:65], s[26:27] // (op 242) convert

s_lshl_b32 s1, s26, 7 // (op 242) 
s_mul_i32 s18, 3, s1 // (op 242) 

v_add_u32 v16, v66, s18 // (op 242) 

buffer_load_dwordx4 v[44:47], v16, s[52:55], 0 offen // (op 242) Load value
v_add_u32 v16, v16, v64 // (op 242) 
ds_read_b128 v[68:71], v8 offset:12800 // (op 247) Load local data 
buffer_load_dwordx4 v[48:51], v16, s[52:55], 0 offen // (op 242) Load value



s_waitcnt lgkmcnt(5)
s_barrier  // (op 2839) 




ds_read_u8 v16, v7 offset:25344 // (op 249) Load local data 
s_lshl_b32 s1, s36, 2 // (op 257) 
s_barrier  // (op 2845) 
s_mul_i32 s18, 3, s1 // (op 257) 




ds_read_u8 v53, v9 offset:16896 // (op 251) Load local data 
s_lshl_b32 s1, s26, 8 // (op 2119) 



v_add_u32 v17, v17, s1 // (op 2119) 

v_add_u32 v64, v67, s18 // (op 257) 


ds_read_b128 v[72:75], v8 offset:14848 // (op 381) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[68:71], a[0:3], v16, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 252) 
buffer_load_ubyte v21, v64, s[68:71], 0 offen offset: 0 // (op 257) Load value





s_barrier  // (op 2905) 
s_barrier  // (op 2788) 
ds_read_u8 v64, v9 offset:17024 // (op 385) Load local data 



s_lshl_b32 s1, s36, 3 // (op 2247) 



s_barrier  // (op 2909) 
v_add_u32 v19, v19, s1 // (op 2247) 

ds_read_b64_tr_b8 v[76:77], v6 offset:17184 // (op 630) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v6 offset:17696 // (op 630) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[72:75], a[4:7], v16, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 377) 
ds_read_b64_tr_b8 v[80:81], v6 offset:21280 // (op 630) Transpose load local data 


ds_read_b64_tr_b8 v[82:83], v6 offset:21792 // (op 630) Transpose load local data 
s_waitcnt vmcnt(9)
ds_write_b128 v18, v[24:27] // (op 13) Store local data 


s_barrier  // (op 2929) 
s_barrier  // (op 2816) 
ds_read_u8 v16, v7 offset:25376 // (op 633) Load local data 



s_waitcnt vmcnt(8)
ds_write_b8 v43, v20 offset:4096 // (op 30) Store local data 
s_barrier  // (op 2933) 
s_barrier  // (op 2775) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:83], v[68:71], a[8:11], v16, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 674) 





s_waitcnt vmcnt(7)
ds_write_b128 v23, v[30:33] offset:4352 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:83], v[72:75], a[12:15], v16, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 628) 
s_waitcnt vmcnt(6)
ds_write_b128 v23, v[34:37] offset:4416 // (op 6) Store local data 




s_barrier  // (op 2802) 



s_waitcnt vmcnt(5)
ds_write_b8 v42, v28 offset:12544 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 588) 
s_mov_b32 s1, s18 // (op 588) convert

s_and_b32 s18, s1, 4294967294 // (op 588) 

s_cmp_lt_u32 s0, s18 // (op 588) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_588 // (op 588) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_588 if true)
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_588:

















s_mul_hi_i32 s0, s16, s10 // (op 2079) 
v_and_b32 v15, -4, v1 // (op 2079) 
s_add_i32 s1, s0, s16 // (op 2079) 

v_sub_u32 v16, v1, v15 // (op 2079) 

s_ashr_i32 s0, s1, 31 // (op 2079) 
v_lshlrev_b32 v15, 4, v16 // (op 2079) 

s_cmp_eq_i32 s10, 0 // (op 2079) 
s_mov_b32 s18, scc // (op 2079) 
v_lshrrev_b32 v16, 2, v1 // (op 2079) 
s_cmp_eq_i32 s18, 0 // (op 2079) 
s_cselect_b32 s19, 0, -1 // (op 2079) 

v_lshlrev_b32 v17, 1, v16 // (op 2079) 

s_and_b32 s18, s11, 31 // (op 2079) 
s_lshl_b32 s22, 1, s18 // (op 2079) 
s_add_i32 s23, s22, s19 // (op 2079) 


s_and_b32 s19, s0, s23 // (op 2079) 


s_add_i32 s0, s1, s19 // (op 2079) 


s_ashr_i32 s1, s0, s18 // (op 2079) 


s_lshl_b32 s0, s11, 24 // (op 2079) 
s_ashr_i32 s18, s0, 31 // (op 2079) 

s_xor_b32 s0, s1, s18 // (op 2079) 

s_sub_i32 s1, s0, s18 // (op 2079) 


s_mul_hi_i32 s0, s1, s12 // (op 2079) 
s_add_i32 s18, s0, s1 // (op 2079) 

s_ashr_i32 s0, s18, 31 // (op 2079) 
s_cmp_eq_i32 s12, 0 // (op 2079) 
s_mov_b32 s19, scc // (op 2079) 
s_cmp_eq_i32 s19, 0 // (op 2079) 
s_cselect_b32 s22, 0, -1 // (op 2079) 

s_and_b32 s19, s13, 31 // (op 2079) 
s_lshl_b32 s23, 1, s19 // (op 2079) 
s_add_i32 s24, s23, s22 // (op 2079) 


s_and_b32 s25, s0, s24 // (op 2079) 


s_add_i32 s0, s18, s25 // (op 2079) 


s_ashr_i32 s18, s0, s19 // (op 2079) 


s_lshl_b32 s0, s13, 24 // (op 2079) 
s_ashr_i32 s19, s0, 31 // (op 2079) 

s_xor_b32 s0, s18, s19 // (op 2079) 

s_sub_i32 s18, s0, s19 // (op 2079) 


s_mul_i32 s0, s1, s15 // (op 2079) 

s_sub_u32 s1, s16, s0 // (op 2079) 

s_mul_i32 s0, s18, s15 // (op 2079) 
s_add_u32 s0, s0, s1 // (op 2079) 


s_mul_hi_i32 s1, s16, s9 // (op 2079) 
s_add_i32 s18, s1, s16 // (op 2079) 

s_ashr_i32 s1, s18, 31 // (op 2079) 
s_cmp_eq_i32 s9, 0 // (op 2079) 
s_mov_b32 s19, scc // (op 2079) 
s_cmp_eq_i32 s19, 0 // (op 2079) 
s_cselect_b32 s22, 0, -1 // (op 2079) 

s_and_b32 s19, s46, 31 // (op 2079) 
s_lshl_b32 s23, 1, s19 // (op 2079) 
s_add_i32 s24, s23, s22 // (op 2079) 


s_and_b32 s25, s1, s24 // (op 2079) 


s_add_i32 s1, s18, s25 // (op 2079) 


s_ashr_i32 s18, s1, s19 // (op 2079) 


s_lshl_b32 s1, s46, 24 // (op 2079) 
s_ashr_i32 s19, s1, 31 // (op 2079) 

s_xor_b32 s1, s18, s19 // (op 2079) 

s_sub_i32 s18, s1, s19 // (op 2079) 


s_add_u32 s50, s6, 64 // (op 2079) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2079) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2079) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2079) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2079) 

s_mul_i32 s1, s15, s24 // (op 2079) 

s_mul_i32 s19, s8, s1 // (op 2079) 

s_mul_i32 s1, s18, s19 // (op 2079) 

s_sub_u32 s19, s16, s1 // (op 2079) 

s_mov_b32 s1, s19 // (op 2079) 
s_mul_hi_i32 s22, s1, s14 // (op 2079) 
s_add_i32 s23, s22, s1 // (op 2079) 


s_ashr_i32 s1, s23, 31 // (op 2079) 
s_cmp_eq_i32 s14, 0 // (op 2079) 
s_mov_b32 s22, scc // (op 2079) 
s_cmp_eq_i32 s22, 0 // (op 2079) 
s_cselect_b32 s24, 0, -1 // (op 2079) 

s_and_b32 s22, s48, 31 // (op 2079) 
s_lshl_b32 s25, 1, s22 // (op 2079) 
s_add_i32 s49, s25, s24 // (op 2079) 


s_and_b32 s24, s1, s49 // (op 2079) 


s_add_i32 s1, s23, s24 // (op 2079) 


s_ashr_i32 s23, s1, s22 // (op 2079) 


s_lshl_b32 s1, s48, 24 // (op 2079) 
s_ashr_i32 s22, s1, 31 // (op 2079) 

s_xor_b32 s1, s23, s22 // (op 2079) 

s_sub_i32 s23, s1, s22 // (op 2079) 


s_mul_i32 s1, s47, s15 // (op 2079) 
s_add_u32 s24, s20, 64 // (op 2079) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2079) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2079) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2079) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2079) 

s_sub_i32 s22, s24, s1 // (op 2079) 

s_mul_i32 s24, s23, s22 // (op 2079) 


s_sub_u32 s25, s19, s24 // (op 2079) 


s_add_u32 s19, s25, s1 // (op 2079) 


s_cmp_eq_i32 s18, 0 // (op 2079) 
s_mov_b32 s1, scc // (op 2079) 

s_cmp_eq_i32 s1, 0 // (op 2079) 
s_cselect_b32 s18, s19, s0 // (op 2079) 



v_lshl_add_u32 v16, s18, 6, v15 // (op 2079) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2079) 
s_lshr_b32 s18, s0, 1 // (op 2079) 

v_lshl_add_u32 v15, s18, 8, v17 // (op 2079) 


v_mul_lo_u32 v17, v15, s26 // (op 2079) 

v_add_u32 v15, v16, v17 // (op 2079) 


s_mul_i32 s1, s26, s31 // (op 2081) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2081) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2081) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2081) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2081) most significant: sum
s_add_u32 s1, s1, s19 // (op 2081) most significant: sum


s_mov_b32 s52, s28 // (op 2081) reg expression
s_mov_b32 s53, s29 // (op 2081) reg expression
s_mov_b32 s54, s0 // (op 2081) reg expression
s_mov_b32 s55, 131072 // (op 2081) call()

s_mul_hi_i32 s0, s16, s10 // (op 2149) 
v_lshrrev_b32 v16, 2, v1 // (op 2149) 
s_add_i32 s1, s0, s16 // (op 2149) 

v_and_b32 v17, -4, v1 // (op 2149) 
s_ashr_i32 s0, s1, 31 // (op 2149) 
v_sub_u32 v18, v1, v17 // (op 2149) 

s_cmp_eq_i32 s10, 0 // (op 2149) 
s_mov_b32 s18, scc // (op 2149) 
v_lshlrev_b32 v17, 5, v18 // (op 2149) 

s_cmp_eq_i32 s18, 0 // (op 2149) 
s_cselect_b32 s19, 0, -1 // (op 2149) 

s_and_b32 s18, s11, 31 // (op 2149) 
s_lshl_b32 s22, 1, s18 // (op 2149) 
s_add_i32 s23, s22, s19 // (op 2149) 


s_and_b32 s19, s0, s23 // (op 2149) 


s_add_i32 s0, s1, s19 // (op 2149) 


s_ashr_i32 s1, s0, s18 // (op 2149) 


s_lshl_b32 s0, s11, 24 // (op 2149) 
s_ashr_i32 s18, s0, 31 // (op 2149) 

s_xor_b32 s0, s1, s18 // (op 2149) 

s_sub_i32 s1, s0, s18 // (op 2149) 


s_mul_hi_i32 s0, s1, s12 // (op 2149) 
s_add_i32 s18, s0, s1 // (op 2149) 

s_ashr_i32 s0, s18, 31 // (op 2149) 
s_cmp_eq_i32 s12, 0 // (op 2149) 
s_mov_b32 s19, scc // (op 2149) 
s_cmp_eq_i32 s19, 0 // (op 2149) 
s_cselect_b32 s22, 0, -1 // (op 2149) 

s_and_b32 s19, s13, 31 // (op 2149) 
s_lshl_b32 s23, 1, s19 // (op 2149) 
s_add_i32 s24, s23, s22 // (op 2149) 


s_and_b32 s25, s0, s24 // (op 2149) 


s_add_i32 s0, s18, s25 // (op 2149) 


s_ashr_i32 s18, s0, s19 // (op 2149) 


s_lshl_b32 s0, s13, 24 // (op 2149) 
s_ashr_i32 s19, s0, 31 // (op 2149) 

s_xor_b32 s0, s18, s19 // (op 2149) 

s_sub_i32 s18, s0, s19 // (op 2149) 


s_add_u32 s50, s6, 64 // (op 2149) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2149) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2149) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2149) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2149) 

s_mul_i32 s0, s18, s24 // (op 2149) 

s_sub_i32 s18, s1, s0 // (op 2149) 


s_mul_hi_i32 s19, s16, s9 // (op 2149) 
s_add_i32 s0, s19, s16 // (op 2149) 

s_ashr_i32 s1, s0, 31 // (op 2149) 
s_cmp_eq_i32 s9, 0 // (op 2149) 
s_mov_b32 s19, scc // (op 2149) 
s_cmp_eq_i32 s19, 0 // (op 2149) 
s_cselect_b32 s22, 0, -1 // (op 2149) 

s_and_b32 s19, s46, 31 // (op 2149) 
s_lshl_b32 s23, 1, s19 // (op 2149) 
s_add_i32 s49, s23, s22 // (op 2149) 


s_and_b32 s22, s1, s49 // (op 2149) 


s_add_i32 s1, s0, s22 // (op 2149) 


s_ashr_i32 s0, s1, s19 // (op 2149) 


s_lshl_b32 s1, s46, 24 // (op 2149) 
s_ashr_i32 s19, s1, 31 // (op 2149) 

s_xor_b32 s1, s0, s19 // (op 2149) 

s_sub_i32 s0, s1, s19 // (op 2149) 


s_mul_i32 s1, s15, s24 // (op 2149) 

s_mul_i32 s19, s8, s1 // (op 2149) 

s_mul_i32 s1, s0, s19 // (op 2149) 

s_sub_u32 s19, s16, s1 // (op 2149) 

s_mul_hi_i32 s1, s19, s14 // (op 2149) 
s_add_i32 s22, s1, s19 // (op 2149) 


s_ashr_i32 s1, s22, 31 // (op 2149) 
s_cmp_eq_i32 s14, 0 // (op 2149) 
s_mov_b32 s19, scc // (op 2149) 
s_cmp_eq_i32 s19, 0 // (op 2149) 
s_cselect_b32 s23, 0, -1 // (op 2149) 

s_and_b32 s19, s48, 31 // (op 2149) 
s_lshl_b32 s24, 1, s19 // (op 2149) 
s_add_i32 s25, s24, s23 // (op 2149) 


s_and_b32 s23, s1, s25 // (op 2149) 


s_add_i32 s1, s22, s23 // (op 2149) 


s_ashr_i32 s22, s1, s19 // (op 2149) 


s_lshl_b32 s1, s48, 24 // (op 2149) 
s_ashr_i32 s19, s1, 31 // (op 2149) 

s_xor_b32 s1, s22, s19 // (op 2149) 

s_sub_i32 s22, s1, s19 // (op 2149) 


s_cmp_eq_i32 s0, 0 // (op 2149) 
s_mov_b32 s1, scc // (op 2149) 

s_cmp_eq_i32 s1, 0 // (op 2149) 
s_cselect_b32 s0, s22, s18 // (op 2149) 



v_lshl_add_u32 v18, s0, 6, v16 // (op 2149) 


v_mul_lo_u32 v43, v18, s33 // (op 2149) most significant: low of low * high
v_mul_hi_u32 v16, v18, s32 // (op 2149) most significant: high of low * low
v_mul_lo_u32 v42, v18, s32 // (op 2149) least significant: low of low * low
v_add_u32 v43, v43, v16 // (op 2149) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2149) 
s_lshr_b32 s18, s0, 1 // (op 2149) 

v_lshl_add_u32 v16, s18, 8, v17 // (op 2149) 


v_mov_b32 v23, 0 // (op 2149) convert
v_mov_b32 v22, v16 // (op 2149) convert

v_add_co_u32 v16, s[0:1], v22, v42 // (op 2149) least significant half
v_addc_co_u32 v17, s[0:1], v23, v43, s[0:1] // (op 2149) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 2149) 


v_lshrrev_b64 v[42:43], 3, v[16:17] // (op 2149) 

v_mov_b32 v16, v42 // (op 2149) convert

s_mul_i32 s1, s32, s7 // (op 2151) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2151) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2151) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2151) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2151) most significant: sum
s_add_u32 s1, s1, s19 // (op 2151) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2151) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2151) 

s_mov_b32 s64, s34 // (op 2151) reg expression
s_mov_b32 s65, s35 // (op 2151) reg expression
s_mov_b32 s66, s0 // (op 2151) reg expression
s_mov_b32 s67, 131072 // (op 2151) call()

s_mul_hi_i32 s0, s16, s10 // (op 2213) 
v_and_b32 v17, -64, v1 // (op 2213) 
s_add_i32 s1, s0, s16 // (op 2213) 

v_sub_u32 v18, v1, v17 // (op 2213) 

s_ashr_i32 s0, s1, 31 // (op 2213) 
v_lshrrev_b32 v17, 6, v1 // (op 2213) 
s_cmp_eq_i32 s10, 0 // (op 2213) 
s_mov_b32 s18, scc // (op 2213) 
s_cmp_eq_i32 s18, 0 // (op 2213) 
s_cselect_b32 s19, 0, -1 // (op 2213) 

s_and_b32 s18, s11, 31 // (op 2213) 
s_lshl_b32 s22, 1, s18 // (op 2213) 
s_add_i32 s23, s22, s19 // (op 2213) 


s_and_b32 s19, s0, s23 // (op 2213) 


s_add_i32 s0, s1, s19 // (op 2213) 


s_ashr_i32 s1, s0, s18 // (op 2213) 


s_lshl_b32 s0, s11, 24 // (op 2213) 
s_ashr_i32 s18, s0, 31 // (op 2213) 

s_xor_b32 s0, s1, s18 // (op 2213) 

s_sub_i32 s1, s0, s18 // (op 2213) 


s_mul_hi_i32 s0, s1, s12 // (op 2213) 
s_add_i32 s18, s0, s1 // (op 2213) 

s_ashr_i32 s0, s18, 31 // (op 2213) 
s_cmp_eq_i32 s12, 0 // (op 2213) 
s_mov_b32 s19, scc // (op 2213) 
s_cmp_eq_i32 s19, 0 // (op 2213) 
s_cselect_b32 s22, 0, -1 // (op 2213) 

s_and_b32 s19, s13, 31 // (op 2213) 
s_lshl_b32 s23, 1, s19 // (op 2213) 
s_add_i32 s24, s23, s22 // (op 2213) 


s_and_b32 s25, s0, s24 // (op 2213) 


s_add_i32 s0, s18, s25 // (op 2213) 


s_ashr_i32 s18, s0, s19 // (op 2213) 


s_lshl_b32 s0, s13, 24 // (op 2213) 
s_ashr_i32 s19, s0, 31 // (op 2213) 

s_xor_b32 s0, s18, s19 // (op 2213) 

s_sub_i32 s18, s0, s19 // (op 2213) 


s_mul_i32 s0, s1, s15 // (op 2213) 

s_sub_u32 s1, s16, s0 // (op 2213) 

s_mul_i32 s0, s18, s15 // (op 2213) 
s_add_u32 s0, s0, s1 // (op 2213) 


s_mul_hi_i32 s1, s16, s9 // (op 2213) 
s_add_i32 s18, s1, s16 // (op 2213) 

s_ashr_i32 s1, s18, 31 // (op 2213) 
s_cmp_eq_i32 s9, 0 // (op 2213) 
s_mov_b32 s19, scc // (op 2213) 
s_cmp_eq_i32 s19, 0 // (op 2213) 
s_cselect_b32 s22, 0, -1 // (op 2213) 

s_and_b32 s19, s46, 31 // (op 2213) 
s_lshl_b32 s23, 1, s19 // (op 2213) 
s_add_i32 s24, s23, s22 // (op 2213) 


s_and_b32 s25, s1, s24 // (op 2213) 


s_add_i32 s1, s18, s25 // (op 2213) 


s_ashr_i32 s18, s1, s19 // (op 2213) 


s_lshl_b32 s1, s46, 24 // (op 2213) 
s_ashr_i32 s19, s1, 31 // (op 2213) 

s_xor_b32 s1, s18, s19 // (op 2213) 

s_sub_i32 s18, s1, s19 // (op 2213) 


s_add_u32 s50, s6, 64 // (op 2213) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2213) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2213) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2213) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2213) 

s_mul_i32 s1, s15, s24 // (op 2213) 

s_mul_i32 s19, s8, s1 // (op 2213) 

s_mul_i32 s1, s18, s19 // (op 2213) 

s_sub_u32 s19, s16, s1 // (op 2213) 

s_mov_b32 s1, s19 // (op 2213) 
s_mul_hi_i32 s22, s1, s14 // (op 2213) 
s_add_i32 s23, s22, s1 // (op 2213) 


s_ashr_i32 s1, s23, 31 // (op 2213) 
s_cmp_eq_i32 s14, 0 // (op 2213) 
s_mov_b32 s22, scc // (op 2213) 
s_cmp_eq_i32 s22, 0 // (op 2213) 
s_cselect_b32 s24, 0, -1 // (op 2213) 

s_and_b32 s22, s48, 31 // (op 2213) 
s_lshl_b32 s25, 1, s22 // (op 2213) 
s_add_i32 s49, s25, s24 // (op 2213) 


s_and_b32 s24, s1, s49 // (op 2213) 


s_add_i32 s1, s23, s24 // (op 2213) 


s_ashr_i32 s23, s1, s22 // (op 2213) 


s_lshl_b32 s1, s48, 24 // (op 2213) 
s_ashr_i32 s22, s1, 31 // (op 2213) 

s_xor_b32 s1, s23, s22 // (op 2213) 

s_sub_i32 s23, s1, s22 // (op 2213) 


s_mul_i32 s1, s47, s15 // (op 2213) 
s_add_u32 s24, s20, 64 // (op 2213) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2213) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2213) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2213) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2213) 

s_sub_i32 s22, s24, s1 // (op 2213) 

s_mul_i32 s24, s23, s22 // (op 2213) 


s_sub_u32 s25, s19, s24 // (op 2213) 


s_add_u32 s19, s25, s1 // (op 2213) 


s_cmp_eq_i32 s18, 0 // (op 2213) 
s_mov_b32 s1, scc // (op 2213) 

s_cmp_eq_i32 s1, 0 // (op 2213) 
s_cselect_b32 s18, s19, s0 // (op 2213) 



v_lshl_add_u32 v19, s18, 6, v18 // (op 2213) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2213) 
s_lshr_b32 s18, s0, 1 // (op 2213) 

v_lshl_add_u32 v18, s18, 3, v17 // (op 2213) 


v_mul_lo_u32 v17, v18, s36 // (op 2213) 

v_add_u32 v18, v19, v17 // (op 2213) 


s_mul_i32 s1, s36, s41 // (op 2215) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2215) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2215) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2215) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2215) most significant: sum
s_add_u32 s1, s1, s19 // (op 2215) most significant: sum


s_mov_b32 s68, s38 // (op 2215) reg expression
s_mov_b32 s69, s39 // (op 2215) reg expression
s_mov_b32 s70, s0 // (op 2215) reg expression
s_mov_b32 s71, 131072 // (op 2215) call()

s_mul_hi_i32 s0, s16, s10 // (op 2277) 
v_lshrrev_b32 v17, 2, v1 // (op 2277) 
s_add_i32 s1, s0, s16 // (op 2277) 

v_and_b32 v19, -4, v1 // (op 2277) 
s_ashr_i32 s0, s1, 31 // (op 2277) 
v_sub_u32 v20, v1, v19 // (op 2277) 

s_cmp_eq_i32 s10, 0 // (op 2277) 
s_mov_b32 s18, scc // (op 2277) 
s_cmp_eq_i32 s18, 0 // (op 2277) 
s_cselect_b32 s19, 0, -1 // (op 2277) 

s_and_b32 s18, s11, 31 // (op 2277) 
s_lshl_b32 s22, 1, s18 // (op 2277) 
s_add_i32 s23, s22, s19 // (op 2277) 


s_and_b32 s19, s0, s23 // (op 2277) 


s_add_i32 s0, s1, s19 // (op 2277) 


s_ashr_i32 s1, s0, s18 // (op 2277) 


s_lshl_b32 s0, s11, 24 // (op 2277) 
s_ashr_i32 s18, s0, 31 // (op 2277) 

s_xor_b32 s0, s1, s18 // (op 2277) 

s_sub_i32 s1, s0, s18 // (op 2277) 


s_mul_hi_i32 s0, s1, s12 // (op 2277) 
s_add_i32 s18, s0, s1 // (op 2277) 

s_ashr_i32 s0, s18, 31 // (op 2277) 
s_cmp_eq_i32 s12, 0 // (op 2277) 
s_mov_b32 s19, scc // (op 2277) 
s_cmp_eq_i32 s19, 0 // (op 2277) 
s_cselect_b32 s22, 0, -1 // (op 2277) 

s_and_b32 s19, s13, 31 // (op 2277) 
s_lshl_b32 s23, 1, s19 // (op 2277) 
s_add_i32 s24, s23, s22 // (op 2277) 


s_and_b32 s25, s0, s24 // (op 2277) 


s_add_i32 s0, s18, s25 // (op 2277) 


s_ashr_i32 s18, s0, s19 // (op 2277) 


s_lshl_b32 s0, s13, 24 // (op 2277) 
s_ashr_i32 s19, s0, 31 // (op 2277) 

s_xor_b32 s0, s18, s19 // (op 2277) 

s_sub_i32 s18, s0, s19 // (op 2277) 


s_add_u32 s50, s6, 64 // (op 2277) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2277) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2277) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2277) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2277) 

s_mul_i32 s0, s18, s24 // (op 2277) 

s_sub_i32 s18, s1, s0 // (op 2277) 


s_mul_hi_i32 s19, s16, s9 // (op 2277) 
s_add_i32 s0, s19, s16 // (op 2277) 

s_ashr_i32 s1, s0, 31 // (op 2277) 
s_cmp_eq_i32 s9, 0 // (op 2277) 
s_mov_b32 s19, scc // (op 2277) 
s_cmp_eq_i32 s19, 0 // (op 2277) 
s_cselect_b32 s22, 0, -1 // (op 2277) 

s_and_b32 s19, s46, 31 // (op 2277) 
s_lshl_b32 s23, 1, s19 // (op 2277) 
s_add_i32 s49, s23, s22 // (op 2277) 


s_and_b32 s22, s1, s49 // (op 2277) 


s_add_i32 s1, s0, s22 // (op 2277) 


s_ashr_i32 s0, s1, s19 // (op 2277) 


s_lshl_b32 s1, s46, 24 // (op 2277) 
s_ashr_i32 s19, s1, 31 // (op 2277) 

s_xor_b32 s1, s0, s19 // (op 2277) 

s_sub_i32 s0, s1, s19 // (op 2277) 


s_mul_i32 s1, s15, s24 // (op 2277) 

s_mul_i32 s19, s8, s1 // (op 2277) 

s_mul_i32 s1, s0, s19 // (op 2277) 

s_sub_u32 s19, s16, s1 // (op 2277) 

s_mul_hi_i32 s1, s19, s14 // (op 2277) 
s_add_i32 s22, s1, s19 // (op 2277) 


s_ashr_i32 s1, s22, 31 // (op 2277) 
s_cmp_eq_i32 s14, 0 // (op 2277) 
s_mov_b32 s19, scc // (op 2277) 
s_cmp_eq_i32 s19, 0 // (op 2277) 
s_cselect_b32 s23, 0, -1 // (op 2277) 

s_and_b32 s19, s48, 31 // (op 2277) 
s_lshl_b32 s24, 1, s19 // (op 2277) 
s_add_i32 s25, s24, s23 // (op 2277) 


s_and_b32 s23, s1, s25 // (op 2277) 


s_add_i32 s1, s22, s23 // (op 2277) 


s_ashr_i32 s22, s1, s19 // (op 2277) 


s_lshl_b32 s1, s48, 24 // (op 2277) 
s_ashr_i32 s19, s1, 31 // (op 2277) 

s_xor_b32 s1, s22, s19 // (op 2277) 

s_sub_i32 s22, s1, s19 // (op 2277) 


s_cmp_eq_i32 s0, 0 // (op 2277) 
s_mov_b32 s1, scc // (op 2277) 

s_cmp_eq_i32 s1, 0 // (op 2277) 
s_cselect_b32 s0, s22, s18 // (op 2277) 



v_lshl_add_u32 v19, s0, 6, v17 // (op 2277) 


v_mul_lo_u32 v17, v19, s42 // (op 2277) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 2277) 
s_lshr_b32 s18, s0, 1 // (op 2277) 

v_lshl_add_u32 v19, s18, 3, v20 // (op 2277) 


v_add_u32 v20, v19, v17 // (op 2277) 


s_mul_i32 s1, s42, s7 // (op 2279) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2279) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2279) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2279) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2279) most significant: sum
s_add_u32 s1, s1, s19 // (op 2279) most significant: sum


s_mov_b32 s72, s44 // (op 2279) reg expression
s_mov_b32 s73, s45 // (op 2279) reg expression
s_mov_b32 s74, s0 // (op 2279) reg expression
s_mov_b32 s75, 131072 // (op 2279) call()

v_and_b32 v17, -4, v1 // (op 2375) 
v_sub_u32 v19, v1, v17 // (op 2375) 

v_lshlrev_b32 v17, 4, v19 // (op 2375) 

v_lshrrev_b32 v19, 2, v1 // (op 2375) 
v_lshl_add_u32 v22, v19, 7, v17 // (op 2375) 


v_and_b32 v17, -4, v1 // (op 2433) 
v_sub_u32 v19, v1, v17 // (op 2433) 

v_lshlrev_b32 v17, 5, v19 // (op 2433) 

v_lshrrev_b32 v19, 2, v1 // (op 2433) 
v_lshl_add_u32 v23, v19, 7, v17 // (op 2433) 


v_lshlrev_b32 v17, 2, v23 // (op 2433) 

v_lshrrev_b32 v19, 3, v17 // (op 2433) 

v_and_b32 v17, -64, v1 // (op 2491) 
v_sub_u32 v23, v1, v17 // (op 2491) 

v_and_b32 v17, 4294967232, v1 // (op 2491) 
v_add_u32 v24, v17, v23 // (op 2491) 


v_and_b32 v17, -4, v1 // (op 2549) 
v_sub_u32 v23, v1, v17 // (op 2549) 

v_and_b32 v17, 4294967292, v1 // (op 2549) 
v_add_u32 v25, v17, v23 // (op 2549) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 594) 
s_and_b32 s18, s0, 4294967294 // (op 594) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 593) 
s_mov_b32 s19, s0 // (op 593) convert

s_cmp_lt_u32 s18, s19 // (op 593) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_593 // (op 593) Condition: Top (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_593 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_593:

s_add_u32 s18, s18, 1 // (op 595) 
v_mov_b64 v[26:27], s[26:27] // (op 176) convert

v_mov_b32 v14, v15 // (op 176) 
v_mov_b32 v17, v14 // (op 176) reg expression
buffer_load_dwordx4 v[28:31], v17, s[52:55], 0 offen // (op 176) Load value
v_add_u32 v17, v17, v26 // (op 176) 
buffer_load_dwordx4 v[32:35], v17, s[52:55], 0 offen // (op 176) Load value



s_lshl_b32 s19, s26, 7 // (op 2069) 
v_add_u32 v15, v15, s19 // (op 2069) 



s_barrier  // (op 2778) 
v_mov_b32 v14, v16 // (op 190) 




buffer_load_dwordx4 v[36:39], v14, s[64:67], 0 offen // (op 190) Load value

v_add_u32 v16, 64, v16 // (op 2141) 
s_waitcnt vmcnt(2)
ds_write_b128 v22, v[28:31] offset:4352 // (op 177) Store local data 


v_mov_b32 v14, v18 // (op 191) 
v_mov_b32 v17, v14 // (op 191) reg expression

buffer_load_ubyte v21, v17, s[68:71], 0 offen offset: 0 // (op 191) Load value


s_lshl_b32 s19, s36, 2 // (op 2205) 
v_add_u32 v18, v18, s19 // (op 2205) 

s_waitcnt vmcnt(2)
ds_write_b128 v22, v[32:35] offset:4416 // (op 177) Store local data 



v_mov_b32 v14, v20 // (op 192) 

s_barrier  // (op 2791) 
buffer_load_ubyte v17, v14, s[72:75], 0 offen offset: 0 // (op 192) Load value




v_add_u32 v20, 4, v20 // (op 2269) 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[36:39] // (op 187) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 2781) 



s_barrier  // (op 2805) 
ds_read_b64_tr_b8 v[26:27], v0 offset:4352 // (op 179) Transpose load local data 



ds_read_b64_tr_b8 v[28:29], v0 offset:4864 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:8448 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:8960 // (op 179) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v21 offset:12544 // (op 188) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 2795) 



s_barrier  // (op 2819) 
ds_read_b128 v[34:37], v4 // (op 181) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 2809) 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v17 offset:4096 // (op 189) Store local data 



ds_read_u8 v14, v3 offset:12544 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 2823) 



ds_read_u8 v17, v5 offset:4096 // (op 185) Load local data 



ds_read_b128 v[38:41], v4 offset:2048 // (op 350) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:37], a[0:3], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 186) 



ds_read_u8 v21, v5 offset:4224 // (op 354) Load local data 



ds_read_b64_tr_b8 v[42:43], v0 offset:4384 // (op 568) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:4896 // (op 568) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[38:41], a[4:7], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 355) 
ds_read_b64_tr_b8 v[46:47], v0 offset:8480 // (op 568) Transpose load local data 


ds_read_b64_tr_b8 v[48:49], v0 offset:8992 // (op 568) Transpose load local data 


ds_read_u8 v14, v3 offset:12576 // (op 571) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:49], v[34:37], a[8:11], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 573) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[38:41], a[12:15], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 604) 




s_ashr_i64 s[0:1], s[30:31], 7 // (op 593) 
s_mov_b32 s19, s0 // (op 593) convert

s_cmp_lt_u32 s18, s19 // (op 593) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_593 // (op 593) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_593 if true)
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_593:























s_barrier  // (op 2917) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 



v_mov_b32 v0, v13 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 56) Store value


v_accvgpr_read v6, a4 // (op 1017) 
v_accvgpr_read v7, a5 // (op 1017) 
v_accvgpr_read v8, a6 // (op 1017) 
v_accvgpr_read v9, a7 // (op 1017) 
v_mul_f32 v14, v11, v6 // (op 1017) 
v_mul_f32 v15, v11, v7 // (op 1017) 
v_mul_f32 v16, v11, v8 // (op 1017) 
v_mul_f32 v17, v11, v9 // (op 1017) 



s_lshl_b32 s0, s4, 7 // (op 1019) 
v_add_u32 v0, v13, s0 // (op 1019) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1019) Store value


v_accvgpr_read v6, a8 // (op 1041) 
v_accvgpr_read v7, a9 // (op 1041) 
v_accvgpr_read v8, a10 // (op 1041) 
v_accvgpr_read v9, a11 // (op 1041) 
v_mul_f32 v14, v11, v6 // (op 1041) 
v_mul_f32 v15, v11, v7 // (op 1041) 
v_mul_f32 v16, v11, v8 // (op 1041) 
v_mul_f32 v17, v11, v9 // (op 1041) 



v_add_u32 v0, 128, v13 // (op 1044) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1044) Store value


v_accvgpr_read v6, a12 // (op 1050) 
v_accvgpr_read v7, a13 // (op 1050) 
v_accvgpr_read v8, a14 // (op 1050) 
v_accvgpr_read v9, a15 // (op 1050) 
v_mul_f32 v14, v11, v6 // (op 1050) 
v_mul_f32 v15, v11, v7 // (op 1050) 
v_mul_f32 v16, v11, v8 // (op 1050) 
v_mul_f32 v17, v11, v9 // (op 1050) 



s_lshl_b32 s0, s4, 7 // (op 1046) 
s_add_u32 s0, s0, 128 // (op 1046) 
v_add_u32 v0, v13, s0 // (op 1046) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1046) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v12 // (op 1130) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 1130) Load value

v_accvgpr_read v14, a8 // (op 1122) 
v_accvgpr_read v15, a9 // (op 1122) 
v_accvgpr_read v16, a10 // (op 1122) 
v_accvgpr_read v17, a11 // (op 1122) 
v_mul_f32 v18, v11, v14 // (op 1122) 
v_mul_f32 v19, v11, v15 // (op 1122) 
v_mul_f32 v20, v11, v16 // (op 1122) 
v_mul_f32 v21, v11, v17 // (op 1122) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 1129) 
v_mul_f32 v15, v10, v7 // (op 1129) 
v_mul_f32 v16, v10, v8 // (op 1129) 
v_mul_f32 v17, v10, v9 // (op 1129) 

v_add_f32 v6, v14, v18 // (op 1123) 
v_add_f32 v7, v15, v19 // (op 1123) 
v_add_f32 v8, v16, v20 // (op 1123) 
v_add_f32 v9, v17, v21 // (op 1123) 




v_add_u32 v0, 128, v13 // (op 1125) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 1125) Store value




s_lshl_b32 s0, s2, 7 // (op 1134) 
s_add_u32 s0, s0, 128 // (op 1134) 
v_add_u32 v0, v12, s0 // (op 1134) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 1134) Load value

v_accvgpr_read v14, a12 // (op 1136) 
v_accvgpr_read v15, a13 // (op 1136) 
v_accvgpr_read v16, a14 // (op 1136) 
v_accvgpr_read v17, a15 // (op 1136) 
v_mul_f32 v18, v11, v14 // (op 1136) 
v_mul_f32 v19, v11, v15 // (op 1136) 
v_mul_f32 v20, v11, v16 // (op 1136) 
v_mul_f32 v21, v11, v17 // (op 1136) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 1132) 
v_mul_f32 v15, v10, v7 // (op 1132) 
v_mul_f32 v16, v10, v8 // (op 1132) 
v_mul_f32 v17, v10, v9 // (op 1132) 

v_add_f32 v6, v14, v18 // (op 1133) 
v_add_f32 v7, v15, v19 // (op 1133) 
v_add_f32 v8, v16, v20 // (op 1133) 
v_add_f32 v9, v17, v21 // (op 1133) 




s_lshl_b32 s0, s4, 7 // (op 1127) 
s_add_u32 s0, s0, 128 // (op 1127) 
v_add_u32 v0, v13, s0 // (op 1127) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 1127) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 145) Load value

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

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 1092) 
v_add_u32 v0, v12, s0 // (op 1092) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 1092) Load value

v_accvgpr_read v14, a4 // (op 1089) 
v_accvgpr_read v15, a5 // (op 1089) 
v_accvgpr_read v16, a6 // (op 1089) 
v_accvgpr_read v17, a7 // (op 1089) 
v_mul_f32 v18, v11, v14 // (op 1089) 
v_mul_f32 v19, v11, v15 // (op 1089) 
v_mul_f32 v20, v11, v16 // (op 1089) 
v_mul_f32 v21, v11, v17 // (op 1089) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 1093) 
v_mul_f32 v15, v10, v7 // (op 1093) 
v_mul_f32 v16, v10, v8 // (op 1093) 
v_mul_f32 v17, v10, v9 // (op 1093) 

v_add_f32 v6, v14, v18 // (op 1090) 
v_add_f32 v7, v15, v19 // (op 1090) 
v_add_f32 v8, v16, v20 // (op 1090) 
v_add_f32 v9, v17, v21 // (op 1090) 




s_lshl_b32 s0, s4, 7 // (op 1091) 
v_add_u32 v0, v13, s0 // (op 1091) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 1091) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133:












s_endpgm  // End of RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC
  .amdhsa_next_free_vgpr 104
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25600 // lds bytes
  .amdhsa_accum_offset 88
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: F4
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
    macrotile: [64, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 25600
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 86
    .agpr_count: 16
    .max_flat_workgroup_size: 256
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

