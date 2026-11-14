// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_,@function
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s12 // (op 2405) 
s_mul_hi_i32 s1, s16, s12 // (op 2660) 
v_and_b32 v0, 48, v1 // (op 2466) 
v_and_b32 v3, 63, v1 // (op 2466) 
v_sub_u32 v4, v3, v0 // (op 2466) 

v_lshrrev_b32 v3, 1, v4 // (op 2466) 
v_add_lshl_u32 v5, v0, v3, 6 // (op 2466) 


v_and_b32 v0, 1, v4 // (op 2466) 

v_lshlrev_b32 v3, 3, v0 // (op 2466) 

v_lshrrev_b32 v0, 7, v1 // (op 2466) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2466) 


v_add_u32 v0, v5, v4 // (op 2466) 


v_and_b32 v3, 48, v1 // (op 2523) 
v_and_b32 v4, 63, v1 // (op 2523) 
v_sub_u32 v5, v4, v3 // (op 2523) 

v_lshrrev_b32 v4, 1, v5 // (op 2523) 
v_add_lshl_u32 v6, v3, v4, 6 // (op 2523) 


v_and_b32 v3, 1, v5 // (op 2523) 

v_lshlrev_b32 v4, 3, v3 // (op 2523) 

v_lshrrev_b32 v3, 6, v1 // (op 2523) 
v_and_b32 v5, 1, v3 // (op 2523) 

v_lshl_add_u32 v3, v5, 4, v4 // (op 2523) 


v_add_u32 v4, v6, v3 // (op 2523) 


v_lshrrev_b32 v3, 7, v1 // (op 2581) 
v_and_b32 v5, 15, v1 // (op 2581) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 2581) 


v_and_b32 v3, 63, v1 // (op 2581) 
v_lshrrev_b32 v5, 4, v3 // (op 2581) 

v_lshl_add_u32 v3, v5, 6, v6 // (op 2581) 


v_lshrrev_b32 v5, 6, v1 // (op 2639) 
v_and_b32 v6, 1, v5 // (op 2639) 

v_and_b32 v5, 15, v1 // (op 2639) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2639) 


v_and_b32 v5, 63, v1 // (op 2639) 
v_lshrrev_b32 v6, 4, v5 // (op 2639) 

v_lshl_add_u32 v5, v6, 6, v7 // (op 2639) 


v_and_b32 v6, 48, v1 // (op 2706) 
v_and_b32 v7, 63, v1 // (op 2706) 
v_sub_u32 v8, v7, v6 // (op 2706) 

v_lshrrev_b32 v7, 1, v8 // (op 2706) 
v_add_lshl_u32 v9, v6, v7, 6 // (op 2706) 


v_and_b32 v6, 1, v8 // (op 2706) 

v_lshlrev_b32 v7, 3, v6 // (op 2706) 

v_lshrrev_b32 v6, 7, v1 // (op 2706) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 2706) 


v_add_u32 v6, v9, v8 // (op 2706) 


v_and_b32 v7, 48, v1 // (op 2750) 
v_and_b32 v8, 63, v1 // (op 2750) 
v_sub_u32 v9, v8, v7 // (op 2750) 

v_lshrrev_b32 v8, 1, v9 // (op 2750) 
v_add_lshl_u32 v10, v7, v8, 6 // (op 2750) 


v_and_b32 v7, 1, v9 // (op 2750) 

v_lshlrev_b32 v8, 3, v7 // (op 2750) 

v_lshrrev_b32 v7, 6, v1 // (op 2750) 
v_and_b32 v9, 1, v7 // (op 2750) 

v_lshl_add_u32 v7, v9, 4, v8 // (op 2750) 


v_add_u32 v8, v10, v7 // (op 2750) 


v_lshrrev_b32 v7, 7, v1 // (op 2779) 
v_and_b32 v9, 15, v1 // (op 2779) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 2779) 


v_and_b32 v7, 63, v1 // (op 2779) 
v_lshrrev_b32 v9, 4, v7 // (op 2779) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 2779) 


v_lshrrev_b32 v9, 6, v1 // (op 2808) 
v_and_b32 v10, 1, v9 // (op 2808) 

v_and_b32 v9, 15, v1 // (op 2808) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 2808) 


v_and_b32 v9, 63, v1 // (op 2808) 
v_lshrrev_b32 v10, 4, v9 // (op 2808) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 2808) 


v_lshrrev_b32 v10, 7, v1 // (op 2405) 
s_add_i32 s18, s0, s16 // (op 2405) 

v_and_b32 v11, 63, v1 // (op 2405) 
s_ashr_i32 s0, s18, 31 // (op 2405) 
v_lshrrev_b32 v12, 4, v11 // (op 2405) 
s_cmp_eq_i32 s12, 0 // (op 2405) 
s_mov_b32 s19, scc // (op 2405) 
v_lshlrev_b32 v13, 2, v12 // (op 2405) 

s_cmp_eq_i32 s19, 0 // (op 2405) 
s_cselect_b32 s51, 0, -1 // (op 2405) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 2405) 


s_and_b32 s19, s15, 31 // (op 2405) 
v_lshrrev_b32 v10, 2, v11 // (op 2405) 

v_and_b32 v11, 3, v10 // (op 2405) 

v_lshrrev_b32 v10, 7, v1 // (op 2660) 
v_and_b32 v13, 63, v1 // (op 2660) 
s_lshl_b32 s52, 1, s19 // (op 2405) 
v_and_b32 v14, 3, v1 // (op 2405) 
s_add_i32 s53, s52, s51 // (op 2405) 


s_and_b32 s51, s0, s53 // (op 2405) 


s_add_i32 s0, s1, s16 // (op 2660) 

s_ashr_i32 s1, s0, 31 // (op 2660) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 2405) 


s_add_i32 s52, s18, s51 // (op 2405) 


v_lshrrev_b32 v11, 6, v1 // (op 2405) 
s_ashr_i32 s18, s52, s19 // (op 2405) 


v_and_b32 v14, 1, v11 // (op 2405) 

s_lshl_b32 s19, s15, 24 // (op 2405) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 2405) 


s_ashr_i32 s51, s19, 31 // (op 2405) 

s_xor_b32 s19, s18, s51 // (op 2405) 

s_sub_i32 s18, s19, s51 // (op 2405) 


v_lshrrev_b32 v14, 4, v13 // (op 2660) 
s_cmp_eq_i32 s12, 0 // (op 2660) 
s_mov_b32 s19, scc // (op 2660) 
v_lshlrev_b32 v15, 2, v14 // (op 2660) 

s_cmp_eq_i32 s19, 0 // (op 2660) 
s_cselect_b32 s51, 0, -1 // (op 2660) 

v_lshl_add_u32 v14, v10, 4, v15 // (op 2660) 


s_and_b32 s19, s15, 31 // (op 2660) 
v_lshrrev_b32 v10, 2, v13 // (op 2660) 

s_lshl_b32 s52, 1, s19 // (op 2660) 
v_and_b32 v13, 3, v10 // (op 2660) 

s_add_i32 s53, s52, s51 // (op 2660) 


v_and_b32 v10, 3, v1 // (op 2660) 
s_and_b32 s51, s1, s53 // (op 2660) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 2660) 


s_add_i32 s1, s0, s51 // (op 2660) 


v_lshrrev_b32 v10, 6, v1 // (op 2660) 
s_ashr_i32 s0, s1, s19 // (op 2660) 


v_and_b32 v13, 1, v10 // (op 2660) 

s_lshl_b32 s1, s15, 24 // (op 2660) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 2660) 


s_ashr_i32 s19, s1, 31 // (op 2660) 

s_xor_b32 s1, s0, s19 // (op 2660) 

s_sub_i32 s0, s1, s19 // (op 2660) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s1, s18, s44 // (op 2405) 
s_add_i32 s19, s1, s18 // (op 2405) 

s_ashr_i32 s1, s19, 31 // (op 2405) 
s_cmp_eq_i32 s44, 0 // (op 2405) 
s_mov_b32 s51, scc // (op 2405) 
s_cmp_eq_i32 s51, 0 // (op 2405) 
s_cselect_b32 s52, 0, -1 // (op 2405) 

s_and_b32 s51, s45, 31 // (op 2405) 
s_lshl_b32 s53, 1, s51 // (op 2405) 
s_add_i32 s54, s53, s52 // (op 2405) 


s_and_b32 s52, s1, s54 // (op 2405) 


s_add_i32 s1, s19, s52 // (op 2405) 


s_ashr_i32 s19, s1, s51 // (op 2405) 


s_lshl_b32 s1, s45, 24 // (op 2405) 
s_ashr_i32 s51, s1, 31 // (op 2405) 

s_xor_b32 s1, s19, s51 // (op 2405) 

s_sub_i32 s19, s1, s51 // (op 2405) 


s_mul_i32 s1, s18, s13 // (op 2405) 
s_sub_u32 s51, s16, s1 // (op 2405) 

s_mul_i32 s1, s19, s13 // (op 2405) 
s_add_u32 s1, s1, s51 // (op 2405) 

s_mul_hi_i32 s51, s16, s11 // (op 2405) 
s_add_i32 s52, s51, s16 // (op 2405) 

s_ashr_i32 s51, s52, 31 // (op 2405) 
s_cmp_eq_i32 s11, 0 // (op 2405) 
s_mov_b32 s53, scc // (op 2405) 
s_cmp_eq_i32 s53, 0 // (op 2405) 
s_cselect_b32 s54, 0, -1 // (op 2405) 

s_and_b32 s53, s47, 31 // (op 2405) 
s_lshl_b32 s55, 1, s53 // (op 2405) 
s_add_i32 s56, s55, s54 // (op 2405) 


s_and_b32 s54, s51, s56 // (op 2405) 


s_add_i32 s51, s52, s54 // (op 2405) 


s_ashr_i32 s52, s51, s53 // (op 2405) 


s_lshl_b32 s51, s47, 24 // (op 2405) 
s_ashr_i32 s53, s51, 31 // (op 2405) 

s_xor_b32 s51, s52, s53 // (op 2405) 

s_sub_i32 s52, s51, s53 // (op 2405) 


s_mul_hi_i32 s51, s0, s44 // (op 2660) 
s_add_i32 s53, s51, s0 // (op 2660) 

s_ashr_i32 s51, s53, 31 // (op 2660) 
s_cmp_eq_i32 s44, 0 // (op 2660) 
s_mov_b32 s54, scc // (op 2660) 
s_cmp_eq_i32 s54, 0 // (op 2660) 
s_cselect_b32 s55, 0, -1 // (op 2660) 

s_and_b32 s54, s45, 31 // (op 2660) 
s_lshl_b32 s56, 1, s54 // (op 2660) 
s_add_i32 s57, s56, s55 // (op 2660) 


s_and_b32 s55, s51, s57 // (op 2660) 


s_add_i32 s51, s53, s55 // (op 2660) 


s_ashr_i32 s53, s51, s54 // (op 2660) 


s_lshl_b32 s51, s45, 24 // (op 2660) 
s_ashr_i32 s54, s51, 31 // (op 2660) 

s_xor_b32 s51, s53, s54 // (op 2660) 

s_sub_i32 s53, s51, s54 // (op 2660) 


s_mul_i32 s51, s0, s13 // (op 2660) 
s_sub_u32 s54, s16, s51 // (op 2660) 

s_mul_i32 s51, s53, s13 // (op 2660) 
s_add_u32 s51, s51, s54 // (op 2660) 

s_mul_hi_i32 s54, s16, s11 // (op 2660) 
s_add_i32 s55, s54, s16 // (op 2660) 

s_ashr_i32 s54, s55, 31 // (op 2660) 
s_cmp_eq_i32 s11, 0 // (op 2660) 
s_mov_b32 s56, scc // (op 2660) 
s_cmp_eq_i32 s56, 0 // (op 2660) 
s_cselect_b32 s57, 0, -1 // (op 2660) 

s_and_b32 s56, s47, 31 // (op 2660) 
s_lshl_b32 s58, 1, s56 // (op 2660) 
s_add_i32 s59, s58, s57 // (op 2660) 


s_and_b32 s57, s54, s59 // (op 2660) 


s_add_i32 s54, s55, s57 // (op 2660) 


s_ashr_i32 s55, s54, s56 // (op 2660) 


s_lshl_b32 s54, s47, 24 // (op 2660) 
s_ashr_i32 s56, s54, 31 // (op 2660) 

s_xor_b32 s54, s55, s56 // (op 2660) 

s_sub_i32 s55, s54, s56 // (op 2660) 


s_add_u32 s56, s8, 64 // (op 2405) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 2405) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2405) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2405) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 2405) 

s_mul_i32 s54, s13, s56 // (op 2405) 
s_mul_i32 s58, s10, s54 // (op 2405) 

s_mul_i32 s54, s52, s58 // (op 2405) 

s_sub_u32 s58, s16, s54 // (op 2405) 

s_mov_b32 s54, s58 // (op 2405) 
s_mul_hi_i32 s59, s54, s14 // (op 2405) 
s_add_i32 s60, s59, s54 // (op 2405) 


s_ashr_i32 s54, s60, 31 // (op 2405) 
s_cmp_eq_i32 s14, 0 // (op 2405) 
s_mov_b32 s59, scc // (op 2405) 
s_cmp_eq_i32 s59, 0 // (op 2405) 
s_cselect_b32 s61, 0, -1 // (op 2405) 

s_and_b32 s59, s46, 31 // (op 2405) 
s_lshl_b32 s62, 1, s59 // (op 2405) 
s_add_i32 s63, s62, s61 // (op 2405) 


s_and_b32 s61, s54, s63 // (op 2405) 


s_add_i32 s54, s60, s61 // (op 2405) 


s_ashr_i32 s60, s54, s59 // (op 2405) 


s_lshl_b32 s54, s46, 24 // (op 2405) 
s_ashr_i32 s59, s54, 31 // (op 2405) 

s_xor_b32 s54, s60, s59 // (op 2405) 

s_sub_i32 s60, s54, s59 // (op 2405) 


s_mul_i32 s54, s48, s13 // (op 2405) 
s_add_u32 s62, s6, 64 // (op 2405) Start of Int64 add, locking SCC
s_addc_u32 s63, s7, 0 // (op 2405) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2405) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2405) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 2405) 

s_sub_i32 s59, s62, s54 // (op 2405) 

s_mul_i32 s61, s60, s59 // (op 2405) 

s_sub_u32 s59, s58, s61 // (op 2405) 


s_add_u32 s58, s59, s54 // (op 2405) 


s_cmp_eq_i32 s52, 0 // (op 2405) 
s_mov_b32 s54, scc // (op 2405) 

s_cmp_eq_i32 s54, 0 // (op 2405) 
s_cselect_b32 s52, s58, s1 // (op 2405) 


s_mul_i32 s1, s19, s56 // (op 2405) 


v_lshl_add_u32 v13, s52, 6, v12 // (op 2405) 


s_sub_i32 s19, s18, s1 // (op 2405) 


s_cmp_eq_i32 s54, 0 // (op 2405) 
s_cselect_b32 s1, s60, s19 // (op 2405) 



v_lshl_add_u32 v12, s1, 6, v11 // (op 2405) 


v_mul_lo_u32 v11, v12, s2 // (op 2405) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 2405) 


s_add_u32 s18, s8, 64 // (op 2660) Start of Int64 add, locking SCC
s_addc_u32 s19, s9, 0 // (op 2660) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 2660) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 2660) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 6 // (op 2660) 

s_mul_i32 s1, s13, s18 // (op 2660) 
s_mul_i32 s52, s10, s1 // (op 2660) 

s_mul_i32 s1, s55, s52 // (op 2660) 

s_sub_u32 s52, s16, s1 // (op 2660) 

s_mov_b32 s1, s52 // (op 2660) 
s_mul_hi_i32 s54, s1, s14 // (op 2660) 
s_add_i32 s56, s54, s1 // (op 2660) 


s_ashr_i32 s1, s56, 31 // (op 2660) 
s_cmp_eq_i32 s14, 0 // (op 2660) 
s_mov_b32 s54, scc // (op 2660) 
s_cmp_eq_i32 s54, 0 // (op 2660) 
s_cselect_b32 s57, 0, -1 // (op 2660) 

s_and_b32 s54, s46, 31 // (op 2660) 
s_lshl_b32 s58, 1, s54 // (op 2660) 
s_add_i32 s59, s58, s57 // (op 2660) 


s_and_b32 s57, s1, s59 // (op 2660) 


s_add_i32 s1, s56, s57 // (op 2660) 


s_ashr_i32 s56, s1, s54 // (op 2660) 


s_lshl_b32 s1, s46, 24 // (op 2660) 
s_ashr_i32 s54, s1, 31 // (op 2660) 

s_xor_b32 s1, s56, s54 // (op 2660) 

s_sub_i32 s56, s1, s54 // (op 2660) 


s_mul_i32 s1, s48, s13 // (op 2660) 
s_add_u32 s58, s6, 64 // (op 2660) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 2660) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2660) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2660) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2660) 

s_sub_i32 s54, s58, s1 // (op 2660) 

s_mul_i32 s57, s56, s54 // (op 2660) 

s_sub_u32 s54, s52, s57 // (op 2660) 


s_add_u32 s52, s54, s1 // (op 2660) 


s_cmp_eq_i32 s55, 0 // (op 2660) 
s_mov_b32 s1, scc // (op 2660) 

s_cmp_eq_i32 s1, 0 // (op 2660) 
s_cselect_b32 s54, s52, s51 // (op 2660) 


s_mul_i32 s55, s53, s18 // (op 2660) 


v_lshl_add_u32 v11, s54, 6, v14 // (op 2660) 


s_sub_i32 s18, s0, s55 // (op 2660) 


s_cmp_eq_i32 s1, 0 // (op 2660) 
s_cselect_b32 s0, s56, s18 // (op 2660) 



v_lshl_add_u32 v13, s0, 6, v10 // (op 2660) 


v_mul_lo_u32 v10, v13, s4 // (op 2660) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 2660) 


s_mul_i32 s1, s2, s9 // (op 2407) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2407) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2407) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2407) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2407) most significant: sum
s_add_u32 s1, s1, s19 // (op 2407) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2407) 

s_mul_i32 s1, s4, s9 // (op 2662) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 2662) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2662) least significant
s_mul_hi_u32 s52, s4, s8 // (op 2662) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2662) most significant: sum
s_add_u32 s1, s1, s52 // (op 2662) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 2662) 

s_mov_b32 s56, s20 // (op 2407) reg expression
s_mov_b32 s57, s21 // (op 2407) reg expression
s_mov_b32 s58, s18 // (op 2407) reg expression
s_mov_b32 s59, 131072 // (op 2407) call()

s_mov_b32 s60, s22 // (op 2662) reg expression

s_mov_b32 s61, s23 // (op 2662) reg expression
s_mov_b32 s62, s52 // (op 2662) reg expression
s_mov_b32 s63, 131072 // (op 2662) call()


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
s_mul_hi_i32 s0, s16, s12 // (op 2167) 
v_and_b32 v14, -4, v1 // (op 2167) 
s_add_i32 s1, s0, s16 // (op 2167) 

v_sub_u32 v15, v1, v14 // (op 2167) 

s_ashr_i32 s0, s1, 31 // (op 2167) 
v_lshlrev_b32 v14, 4, v15 // (op 2167) 

s_cmp_eq_i32 s12, 0 // (op 2167) 
s_mov_b32 s18, scc // (op 2167) 
v_lshrrev_b32 v15, 2, v1 // (op 2167) 
s_cmp_eq_i32 s18, 0 // (op 2167) 
s_cselect_b32 s19, 0, -1 // (op 2167) 

s_and_b32 s18, s15, 31 // (op 2167) 
s_lshl_b32 s20, 1, s18 // (op 2167) 
s_add_i32 s21, s20, s19 // (op 2167) 


s_and_b32 s19, s0, s21 // (op 2167) 


s_add_i32 s0, s1, s19 // (op 2167) 


s_ashr_i32 s1, s0, s18 // (op 2167) 


s_lshl_b32 s0, s15, 24 // (op 2167) 
s_ashr_i32 s18, s0, 31 // (op 2167) 

s_xor_b32 s0, s1, s18 // (op 2167) 

s_sub_i32 s1, s0, s18 // (op 2167) 


s_mul_hi_i32 s0, s1, s44 // (op 2167) 
s_add_i32 s18, s0, s1 // (op 2167) 

s_ashr_i32 s0, s18, 31 // (op 2167) 
s_cmp_eq_i32 s44, 0 // (op 2167) 
s_mov_b32 s19, scc // (op 2167) 
s_cmp_eq_i32 s19, 0 // (op 2167) 
s_cselect_b32 s20, 0, -1 // (op 2167) 

s_and_b32 s19, s45, 31 // (op 2167) 
s_lshl_b32 s21, 1, s19 // (op 2167) 
s_add_i32 s22, s21, s20 // (op 2167) 


s_and_b32 s23, s0, s22 // (op 2167) 


s_add_i32 s0, s18, s23 // (op 2167) 


s_ashr_i32 s18, s0, s19 // (op 2167) 


s_lshl_b32 s0, s45, 24 // (op 2167) 
s_ashr_i32 s19, s0, 31 // (op 2167) 

s_xor_b32 s0, s18, s19 // (op 2167) 

s_sub_i32 s18, s0, s19 // (op 2167) 


s_mul_i32 s0, s1, s13 // (op 2167) 

s_sub_u32 s1, s16, s0 // (op 2167) 

s_mul_i32 s0, s18, s13 // (op 2167) 
s_add_u32 s0, s0, s1 // (op 2167) 


s_mul_hi_i32 s1, s16, s11 // (op 2167) 
s_add_i32 s18, s1, s16 // (op 2167) 

s_ashr_i32 s1, s18, 31 // (op 2167) 
s_cmp_eq_i32 s11, 0 // (op 2167) 
s_mov_b32 s19, scc // (op 2167) 
s_cmp_eq_i32 s19, 0 // (op 2167) 
s_cselect_b32 s20, 0, -1 // (op 2167) 

s_and_b32 s19, s47, 31 // (op 2167) 
s_lshl_b32 s21, 1, s19 // (op 2167) 
s_add_i32 s22, s21, s20 // (op 2167) 


s_and_b32 s23, s1, s22 // (op 2167) 


s_add_i32 s1, s18, s23 // (op 2167) 


s_ashr_i32 s18, s1, s19 // (op 2167) 


s_lshl_b32 s1, s47, 24 // (op 2167) 
s_ashr_i32 s19, s1, 31 // (op 2167) 

s_xor_b32 s1, s18, s19 // (op 2167) 

s_sub_i32 s18, s1, s19 // (op 2167) 


s_add_u32 s22, s8, 64 // (op 2167) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 2167) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 2167) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 2167) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 2167) 

s_mul_i32 s1, s13, s22 // (op 2167) 

s_mul_i32 s19, s10, s1 // (op 2167) 

s_mul_i32 s1, s18, s19 // (op 2167) 

s_sub_u32 s19, s16, s1 // (op 2167) 

s_mov_b32 s1, s19 // (op 2167) 
s_mul_hi_i32 s20, s1, s14 // (op 2167) 
s_add_i32 s21, s20, s1 // (op 2167) 


s_ashr_i32 s1, s21, 31 // (op 2167) 
s_cmp_eq_i32 s14, 0 // (op 2167) 
s_mov_b32 s20, scc // (op 2167) 
s_cmp_eq_i32 s20, 0 // (op 2167) 
s_cselect_b32 s22, 0, -1 // (op 2167) 

s_and_b32 s20, s46, 31 // (op 2167) 
s_lshl_b32 s23, 1, s20 // (op 2167) 
s_add_i32 s49, s23, s22 // (op 2167) 


s_and_b32 s22, s1, s49 // (op 2167) 


s_add_i32 s1, s21, s22 // (op 2167) 


s_ashr_i32 s21, s1, s20 // (op 2167) 


s_lshl_b32 s1, s46, 24 // (op 2167) 
s_ashr_i32 s20, s1, 31 // (op 2167) 

s_xor_b32 s1, s21, s20 // (op 2167) 

s_sub_i32 s21, s1, s20 // (op 2167) 


s_mul_i32 s1, s48, s13 // (op 2167) 
s_add_u32 s22, s6, 64 // (op 2167) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2167) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 2167) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 2167) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 6 // (op 2167) 

s_sub_i32 s20, s22, s1 // (op 2167) 

s_mul_i32 s22, s21, s20 // (op 2167) 


s_sub_u32 s23, s19, s22 // (op 2167) 


s_add_u32 s19, s23, s1 // (op 2167) 


s_cmp_eq_i32 s18, 0 // (op 2167) 
s_mov_b32 s1, scc // (op 2167) 

s_cmp_eq_i32 s1, 0 // (op 2167) 
s_cselect_b32 s18, s19, s0 // (op 2167) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 2167) 


v_mov_b32 v19, 0 // (op 2167) convert
v_mov_b32 v18, v16 // (op 2167) convert

v_mul_lo_u32 v17, v15, s25 // (op 2167) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 2167) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 2167) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 2167) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 2167) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 2167) most significant half



s_mul_i32 s1, s24, s29 // (op 2169) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 2169) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 2169) least significant
s_mul_hi_u32 s19, s24, s28 // (op 2169) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2169) most significant: sum
s_add_u32 s1, s1, s19 // (op 2169) most significant: sum


s_mov_b32 s20, s26 // (op 2169) reg expression
s_mov_b32 s21, s27 // (op 2169) reg expression
s_mov_b32 s22, s0 // (op 2169) reg expression
s_mov_b32 s23, 131072 // (op 2169) call()

s_mul_hi_i32 s0, s16, s12 // (op 2186) 
v_and_b32 v16, -4, v1 // (op 2186) 
s_add_i32 s1, s0, s16 // (op 2186) 

v_sub_u32 v17, v1, v16 // (op 2186) 

s_ashr_i32 s0, s1, 31 // (op 2186) 
v_lshlrev_b32 v16, 4, v17 // (op 2186) 

s_cmp_eq_i32 s12, 0 // (op 2186) 
s_mov_b32 s18, scc // (op 2186) 
v_lshrrev_b32 v17, 2, v1 // (op 2186) 
s_cmp_eq_i32 s18, 0 // (op 2186) 
s_cselect_b32 s19, 0, -1 // (op 2186) 

s_and_b32 s18, s15, 31 // (op 2186) 
s_lshl_b32 s49, 1, s18 // (op 2186) 
s_add_i32 s50, s49, s19 // (op 2186) 


s_and_b32 s19, s0, s50 // (op 2186) 


s_add_i32 s0, s1, s19 // (op 2186) 


s_ashr_i32 s1, s0, s18 // (op 2186) 


s_lshl_b32 s0, s15, 24 // (op 2186) 
s_ashr_i32 s18, s0, 31 // (op 2186) 

s_xor_b32 s0, s1, s18 // (op 2186) 

s_sub_i32 s1, s0, s18 // (op 2186) 


s_mul_hi_i32 s0, s1, s44 // (op 2186) 
s_add_i32 s18, s0, s1 // (op 2186) 

s_ashr_i32 s0, s18, 31 // (op 2186) 
s_cmp_eq_i32 s44, 0 // (op 2186) 
s_mov_b32 s19, scc // (op 2186) 
s_cmp_eq_i32 s19, 0 // (op 2186) 
s_cselect_b32 s49, 0, -1 // (op 2186) 

s_and_b32 s19, s45, 31 // (op 2186) 
s_lshl_b32 s50, 1, s19 // (op 2186) 
s_add_i32 s51, s50, s49 // (op 2186) 


s_and_b32 s49, s0, s51 // (op 2186) 


s_add_i32 s0, s18, s49 // (op 2186) 


s_ashr_i32 s18, s0, s19 // (op 2186) 


s_lshl_b32 s0, s45, 24 // (op 2186) 
s_ashr_i32 s19, s0, 31 // (op 2186) 

s_xor_b32 s0, s18, s19 // (op 2186) 

s_sub_i32 s18, s0, s19 // (op 2186) 


s_mul_i32 s0, s1, s13 // (op 2186) 

s_sub_u32 s1, s16, s0 // (op 2186) 

s_mul_i32 s0, s18, s13 // (op 2186) 
s_add_u32 s0, s0, s1 // (op 2186) 


s_mul_hi_i32 s1, s16, s11 // (op 2186) 
s_add_i32 s18, s1, s16 // (op 2186) 

s_ashr_i32 s1, s18, 31 // (op 2186) 
s_cmp_eq_i32 s11, 0 // (op 2186) 
s_mov_b32 s19, scc // (op 2186) 
s_cmp_eq_i32 s19, 0 // (op 2186) 
s_cselect_b32 s49, 0, -1 // (op 2186) 

s_and_b32 s19, s47, 31 // (op 2186) 
s_lshl_b32 s50, 1, s19 // (op 2186) 
s_add_i32 s51, s50, s49 // (op 2186) 


s_and_b32 s49, s1, s51 // (op 2186) 


s_add_i32 s1, s18, s49 // (op 2186) 


s_ashr_i32 s18, s1, s19 // (op 2186) 


s_lshl_b32 s1, s47, 24 // (op 2186) 
s_ashr_i32 s19, s1, 31 // (op 2186) 

s_xor_b32 s1, s18, s19 // (op 2186) 

s_sub_i32 s18, s1, s19 // (op 2186) 


s_add_u32 s54, s8, 64 // (op 2186) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 2186) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 2186) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 2186) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 2186) 

s_mul_i32 s1, s13, s50 // (op 2186) 

s_mul_i32 s19, s10, s1 // (op 2186) 

s_mul_i32 s1, s18, s19 // (op 2186) 

s_sub_u32 s19, s16, s1 // (op 2186) 

s_mov_b32 s1, s19 // (op 2186) 
s_mul_hi_i32 s49, s1, s14 // (op 2186) 
s_add_i32 s50, s49, s1 // (op 2186) 


s_ashr_i32 s1, s50, 31 // (op 2186) 
s_cmp_eq_i32 s14, 0 // (op 2186) 
s_mov_b32 s49, scc // (op 2186) 
s_cmp_eq_i32 s49, 0 // (op 2186) 
s_cselect_b32 s51, 0, -1 // (op 2186) 

s_and_b32 s49, s46, 31 // (op 2186) 
s_lshl_b32 s52, 1, s49 // (op 2186) 
s_add_i32 s53, s52, s51 // (op 2186) 


s_and_b32 s51, s1, s53 // (op 2186) 


s_add_i32 s1, s50, s51 // (op 2186) 


s_ashr_i32 s50, s1, s49 // (op 2186) 


s_lshl_b32 s1, s46, 24 // (op 2186) 
s_ashr_i32 s49, s1, 31 // (op 2186) 

s_xor_b32 s1, s50, s49 // (op 2186) 

s_sub_i32 s50, s1, s49 // (op 2186) 


s_mul_i32 s1, s48, s13 // (op 2186) 
s_add_u32 s54, s6, 64 // (op 2186) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 2186) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 2186) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 2186) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 6 // (op 2186) 

s_sub_i32 s49, s54, s1 // (op 2186) 

s_mul_i32 s51, s50, s49 // (op 2186) 


s_sub_u32 s49, s19, s51 // (op 2186) 


s_add_u32 s19, s49, s1 // (op 2186) 


s_cmp_eq_i32 s18, 0 // (op 2186) 
s_mov_b32 s1, scc // (op 2186) 

s_cmp_eq_i32 s1, 0 // (op 2186) 
s_cselect_b32 s18, s19, s0 // (op 2186) 



v_lshl_add_u32 v18, s18, 6, v16 // (op 2186) 


v_mov_b32 v21, 0 // (op 2186) convert
v_mov_b32 v20, v18 // (op 2186) convert

v_mul_lo_u32 v19, v17, s25 // (op 2186) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 2186) most significant: high of low * low
v_mul_lo_u32 v18, v17, s24 // (op 2186) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 2186) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v18 // (op 2186) least significant half
v_addc_co_u32 v17, s[0:1], v21, v19, s[0:1] // (op 2186) most significant half



s_mul_i32 s1, s24, s29 // (op 2188) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 2188) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 2188) least significant
s_mul_hi_u32 s19, s24, s28 // (op 2188) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2188) most significant: sum
s_add_u32 s1, s1, s19 // (op 2188) most significant: sum


s_mov_b32 s20, s26 // (op 2188) reg expression
s_mov_b32 s21, s27 // (op 2188) reg expression
s_mov_b32 s22, s0 // (op 2188) reg expression
s_mov_b32 s23, 131072 // (op 2188) call()

s_mul_hi_i32 s0, s16, s12 // (op 2231) 
v_and_b32 v18, -4, v1 // (op 2231) 
s_add_i32 s1, s0, s16 // (op 2231) 

v_sub_u32 v19, v1, v18 // (op 2231) 

s_ashr_i32 s0, s1, 31 // (op 2231) 
v_lshlrev_b32 v18, 4, v19 // (op 2231) 

s_cmp_eq_i32 s12, 0 // (op 2231) 
s_mov_b32 s18, scc // (op 2231) 
v_lshrrev_b32 v19, 2, v1 // (op 2231) 
s_cmp_eq_i32 s18, 0 // (op 2231) 
s_cselect_b32 s19, 0, -1 // (op 2231) 

s_and_b32 s18, s15, 31 // (op 2231) 
s_lshl_b32 s49, 1, s18 // (op 2231) 
s_add_i32 s50, s49, s19 // (op 2231) 


s_and_b32 s19, s0, s50 // (op 2231) 


s_add_i32 s0, s1, s19 // (op 2231) 


s_ashr_i32 s1, s0, s18 // (op 2231) 


s_lshl_b32 s0, s15, 24 // (op 2231) 
s_ashr_i32 s18, s0, 31 // (op 2231) 

s_xor_b32 s0, s1, s18 // (op 2231) 

s_sub_i32 s1, s0, s18 // (op 2231) 


s_mul_hi_i32 s0, s1, s44 // (op 2231) 
s_add_i32 s18, s0, s1 // (op 2231) 

s_ashr_i32 s0, s18, 31 // (op 2231) 
s_cmp_eq_i32 s44, 0 // (op 2231) 
s_mov_b32 s19, scc // (op 2231) 
s_cmp_eq_i32 s19, 0 // (op 2231) 
s_cselect_b32 s49, 0, -1 // (op 2231) 

s_and_b32 s19, s45, 31 // (op 2231) 
s_lshl_b32 s50, 1, s19 // (op 2231) 
s_add_i32 s51, s50, s49 // (op 2231) 


s_and_b32 s49, s0, s51 // (op 2231) 


s_add_i32 s0, s18, s49 // (op 2231) 


s_ashr_i32 s18, s0, s19 // (op 2231) 


s_lshl_b32 s0, s45, 24 // (op 2231) 
s_ashr_i32 s19, s0, 31 // (op 2231) 

s_xor_b32 s0, s18, s19 // (op 2231) 

s_sub_i32 s18, s0, s19 // (op 2231) 


s_add_u32 s54, s8, 64 // (op 2231) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 2231) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 2231) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 2231) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 2231) 

s_mul_i32 s0, s18, s50 // (op 2231) 

s_sub_i32 s49, s1, s0 // (op 2231) 


s_mul_hi_i32 s0, s16, s11 // (op 2231) 
s_add_i32 s1, s0, s16 // (op 2231) 

s_ashr_i32 s0, s1, 31 // (op 2231) 
s_cmp_eq_i32 s11, 0 // (op 2231) 
s_mov_b32 s18, scc // (op 2231) 
s_cmp_eq_i32 s18, 0 // (op 2231) 
s_cselect_b32 s19, 0, -1 // (op 2231) 

s_and_b32 s18, s47, 31 // (op 2231) 
s_lshl_b32 s52, 1, s18 // (op 2231) 
s_add_i32 s53, s52, s19 // (op 2231) 


s_and_b32 s19, s0, s53 // (op 2231) 


s_add_i32 s0, s1, s19 // (op 2231) 


s_ashr_i32 s1, s0, s18 // (op 2231) 


s_lshl_b32 s0, s47, 24 // (op 2231) 
s_ashr_i32 s18, s0, 31 // (op 2231) 

s_xor_b32 s0, s1, s18 // (op 2231) 

s_sub_i32 s1, s0, s18 // (op 2231) 


s_mul_i32 s0, s13, s50 // (op 2231) 

s_mul_i32 s18, s10, s0 // (op 2231) 

s_mul_i32 s0, s1, s18 // (op 2231) 

s_sub_u32 s18, s16, s0 // (op 2231) 

s_mul_hi_i32 s0, s18, s14 // (op 2231) 
s_add_i32 s19, s0, s18 // (op 2231) 


s_ashr_i32 s0, s19, 31 // (op 2231) 
s_cmp_eq_i32 s14, 0 // (op 2231) 
s_mov_b32 s18, scc // (op 2231) 
s_cmp_eq_i32 s18, 0 // (op 2231) 
s_cselect_b32 s50, 0, -1 // (op 2231) 

s_and_b32 s18, s46, 31 // (op 2231) 
s_lshl_b32 s51, 1, s18 // (op 2231) 
s_add_i32 s52, s51, s50 // (op 2231) 


s_and_b32 s50, s0, s52 // (op 2231) 


s_add_i32 s0, s19, s50 // (op 2231) 


s_ashr_i32 s19, s0, s18 // (op 2231) 


s_lshl_b32 s0, s46, 24 // (op 2231) 
s_ashr_i32 s18, s0, 31 // (op 2231) 

s_xor_b32 s0, s19, s18 // (op 2231) 

s_sub_i32 s19, s0, s18 // (op 2231) 


s_cmp_eq_i32 s1, 0 // (op 2231) 
s_mov_b32 s0, scc // (op 2231) 

s_cmp_eq_i32 s0, 0 // (op 2231) 
s_cselect_b32 s1, s19, s49 // (op 2231) 



v_lshl_add_u32 v20, s1, 6, v18 // (op 2231) 


v_mov_b32 v23, 0 // (op 2231) convert
v_mov_b32 v22, v20 // (op 2231) convert

v_mul_lo_u32 v21, v19, s31 // (op 2231) most significant: low of low * high
v_mul_hi_u32 v18, v19, s30 // (op 2231) most significant: high of low * low
v_mul_lo_u32 v20, v19, s30 // (op 2231) least significant: low of low * low
v_add_u32 v21, v21, v18 // (op 2231) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v22 // (op 2231) least significant half
v_addc_co_u32 v21, s[0:1], v21, v23, s[0:1] // (op 2231) most significant half



s_mul_i32 s1, s30, s29 // (op 2233) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 2233) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 2233) least significant
s_mul_hi_u32 s19, s30, s28 // (op 2233) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2233) most significant: sum
s_add_u32 s1, s1, s19 // (op 2233) most significant: sum


s_mov_b32 s52, s32 // (op 2233) reg expression
s_mov_b32 s53, s33 // (op 2233) reg expression
s_mov_b32 s54, s0 // (op 2233) reg expression
s_mov_b32 s55, 131072 // (op 2233) call()

s_mul_hi_i32 s0, s16, s12 // (op 2250) 
v_and_b32 v18, -4, v1 // (op 2250) 
s_add_i32 s1, s0, s16 // (op 2250) 

v_sub_u32 v19, v1, v18 // (op 2250) 

s_ashr_i32 s0, s1, 31 // (op 2250) 
v_lshlrev_b32 v18, 4, v19 // (op 2250) 

s_cmp_eq_i32 s12, 0 // (op 2250) 
s_mov_b32 s18, scc // (op 2250) 
v_lshrrev_b32 v19, 2, v1 // (op 2250) 
s_cmp_eq_i32 s18, 0 // (op 2250) 
s_cselect_b32 s19, 0, -1 // (op 2250) 

s_and_b32 s18, s15, 31 // (op 2250) 
s_lshl_b32 s49, 1, s18 // (op 2250) 
s_add_i32 s50, s49, s19 // (op 2250) 


s_and_b32 s19, s0, s50 // (op 2250) 


s_add_i32 s0, s1, s19 // (op 2250) 


s_ashr_i32 s1, s0, s18 // (op 2250) 


s_lshl_b32 s0, s15, 24 // (op 2250) 
s_ashr_i32 s18, s0, 31 // (op 2250) 

s_xor_b32 s0, s1, s18 // (op 2250) 

s_sub_i32 s1, s0, s18 // (op 2250) 


s_mul_hi_i32 s0, s1, s44 // (op 2250) 
s_add_i32 s18, s0, s1 // (op 2250) 

s_ashr_i32 s0, s18, 31 // (op 2250) 
s_cmp_eq_i32 s44, 0 // (op 2250) 
s_mov_b32 s19, scc // (op 2250) 
s_cmp_eq_i32 s19, 0 // (op 2250) 
s_cselect_b32 s49, 0, -1 // (op 2250) 

s_and_b32 s19, s45, 31 // (op 2250) 
s_lshl_b32 s50, 1, s19 // (op 2250) 
s_add_i32 s51, s50, s49 // (op 2250) 


s_and_b32 s49, s0, s51 // (op 2250) 


s_add_i32 s0, s18, s49 // (op 2250) 


s_ashr_i32 s18, s0, s19 // (op 2250) 


s_lshl_b32 s0, s45, 24 // (op 2250) 
s_ashr_i32 s19, s0, 31 // (op 2250) 

s_xor_b32 s0, s18, s19 // (op 2250) 

s_sub_i32 s18, s0, s19 // (op 2250) 


s_add_u32 s50, s8, 64 // (op 2250) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2250) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 2250) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 2250) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 2250) 

s_mul_i32 s0, s18, s50 // (op 2250) 

s_sub_i32 s49, s1, s0 // (op 2250) 


s_mul_hi_i32 s0, s16, s11 // (op 2250) 
s_add_i32 s1, s0, s16 // (op 2250) 

s_ashr_i32 s0, s1, 31 // (op 2250) 
s_cmp_eq_i32 s11, 0 // (op 2250) 
s_mov_b32 s18, scc // (op 2250) 
s_cmp_eq_i32 s18, 0 // (op 2250) 
s_cselect_b32 s19, 0, -1 // (op 2250) 

s_and_b32 s18, s47, 31 // (op 2250) 
s_lshl_b32 s64, 1, s18 // (op 2250) 
s_add_i32 s65, s64, s19 // (op 2250) 


s_and_b32 s19, s0, s65 // (op 2250) 


s_add_i32 s0, s1, s19 // (op 2250) 


s_ashr_i32 s1, s0, s18 // (op 2250) 


s_lshl_b32 s0, s47, 24 // (op 2250) 
s_ashr_i32 s18, s0, 31 // (op 2250) 

s_xor_b32 s0, s1, s18 // (op 2250) 

s_sub_i32 s1, s0, s18 // (op 2250) 


s_mul_i32 s0, s13, s50 // (op 2250) 

s_mul_i32 s18, s10, s0 // (op 2250) 

s_mul_i32 s0, s1, s18 // (op 2250) 

s_sub_u32 s18, s16, s0 // (op 2250) 

s_mul_hi_i32 s0, s18, s14 // (op 2250) 
s_add_i32 s19, s0, s18 // (op 2250) 


s_ashr_i32 s0, s19, 31 // (op 2250) 
s_cmp_eq_i32 s14, 0 // (op 2250) 
s_mov_b32 s18, scc // (op 2250) 
s_cmp_eq_i32 s18, 0 // (op 2250) 
s_cselect_b32 s50, 0, -1 // (op 2250) 

s_and_b32 s18, s46, 31 // (op 2250) 
s_lshl_b32 s51, 1, s18 // (op 2250) 
s_add_i32 s64, s51, s50 // (op 2250) 


s_and_b32 s50, s0, s64 // (op 2250) 


s_add_i32 s0, s19, s50 // (op 2250) 


s_ashr_i32 s19, s0, s18 // (op 2250) 


s_lshl_b32 s0, s46, 24 // (op 2250) 
s_ashr_i32 s18, s0, 31 // (op 2250) 

s_xor_b32 s0, s19, s18 // (op 2250) 

s_sub_i32 s19, s0, s18 // (op 2250) 


s_cmp_eq_i32 s1, 0 // (op 2250) 
s_mov_b32 s0, scc // (op 2250) 

s_cmp_eq_i32 s0, 0 // (op 2250) 
s_cselect_b32 s1, s19, s49 // (op 2250) 



v_lshl_add_u32 v22, s1, 6, v18 // (op 2250) 


v_mov_b32 v25, 0 // (op 2250) convert
v_mov_b32 v24, v22 // (op 2250) convert

v_mul_lo_u32 v23, v19, s31 // (op 2250) most significant: low of low * high
v_mul_hi_u32 v18, v19, s30 // (op 2250) most significant: high of low * low
v_mul_lo_u32 v22, v19, s30 // (op 2250) least significant: low of low * low
v_add_u32 v23, v23, v18 // (op 2250) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v24 // (op 2250) least significant half
v_addc_co_u32 v23, s[0:1], v23, v25, s[0:1] // (op 2250) most significant half



s_mul_i32 s1, s30, s29 // (op 2252) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 2252) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 2252) least significant
s_mul_hi_u32 s19, s30, s28 // (op 2252) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2252) most significant: sum
s_add_u32 s1, s1, s19 // (op 2252) most significant: sum


s_mov_b32 s52, s32 // (op 2252) reg expression
s_mov_b32 s53, s33 // (op 2252) reg expression
s_mov_b32 s54, s0 // (op 2252) reg expression
s_mov_b32 s55, 131072 // (op 2252) call()

s_mul_hi_i32 s0, s16, s12 // (op 2295) 
v_and_b32 v18, -64, v1 // (op 2295) 
s_add_i32 s1, s0, s16 // (op 2295) 

v_sub_u32 v19, v1, v18 // (op 2295) 

s_ashr_i32 s0, s1, 31 // (op 2295) 
v_lshrrev_b32 v18, 6, v1 // (op 2295) 
s_cmp_eq_i32 s12, 0 // (op 2295) 
s_mov_b32 s18, scc // (op 2295) 
s_cmp_eq_i32 s18, 0 // (op 2295) 
s_cselect_b32 s19, 0, -1 // (op 2295) 

s_and_b32 s18, s15, 31 // (op 2295) 
s_lshl_b32 s49, 1, s18 // (op 2295) 
s_add_i32 s50, s49, s19 // (op 2295) 


s_and_b32 s19, s0, s50 // (op 2295) 


s_add_i32 s0, s1, s19 // (op 2295) 


s_ashr_i32 s1, s0, s18 // (op 2295) 


s_lshl_b32 s0, s15, 24 // (op 2295) 
s_ashr_i32 s18, s0, 31 // (op 2295) 

s_xor_b32 s0, s1, s18 // (op 2295) 

s_sub_i32 s1, s0, s18 // (op 2295) 


s_mul_hi_i32 s0, s1, s44 // (op 2295) 
s_add_i32 s18, s0, s1 // (op 2295) 

s_ashr_i32 s0, s18, 31 // (op 2295) 
s_cmp_eq_i32 s44, 0 // (op 2295) 
s_mov_b32 s19, scc // (op 2295) 
s_cmp_eq_i32 s19, 0 // (op 2295) 
s_cselect_b32 s49, 0, -1 // (op 2295) 

s_and_b32 s19, s45, 31 // (op 2295) 
s_lshl_b32 s50, 1, s19 // (op 2295) 
s_add_i32 s51, s50, s49 // (op 2295) 


s_and_b32 s49, s0, s51 // (op 2295) 


s_add_i32 s0, s18, s49 // (op 2295) 


s_ashr_i32 s18, s0, s19 // (op 2295) 


s_lshl_b32 s0, s45, 24 // (op 2295) 
s_ashr_i32 s19, s0, 31 // (op 2295) 

s_xor_b32 s0, s18, s19 // (op 2295) 

s_sub_i32 s18, s0, s19 // (op 2295) 


s_mul_i32 s0, s1, s13 // (op 2295) 

s_sub_u32 s1, s16, s0 // (op 2295) 

s_mul_i32 s0, s18, s13 // (op 2295) 
s_add_u32 s0, s0, s1 // (op 2295) 


s_mul_hi_i32 s1, s16, s11 // (op 2295) 
s_add_i32 s18, s1, s16 // (op 2295) 

s_ashr_i32 s1, s18, 31 // (op 2295) 
s_cmp_eq_i32 s11, 0 // (op 2295) 
s_mov_b32 s19, scc // (op 2295) 
s_cmp_eq_i32 s19, 0 // (op 2295) 
s_cselect_b32 s49, 0, -1 // (op 2295) 

s_and_b32 s19, s47, 31 // (op 2295) 
s_lshl_b32 s50, 1, s19 // (op 2295) 
s_add_i32 s51, s50, s49 // (op 2295) 


s_and_b32 s49, s1, s51 // (op 2295) 


s_add_i32 s1, s18, s49 // (op 2295) 


s_ashr_i32 s18, s1, s19 // (op 2295) 


s_lshl_b32 s1, s47, 24 // (op 2295) 
s_ashr_i32 s19, s1, 31 // (op 2295) 

s_xor_b32 s1, s18, s19 // (op 2295) 

s_sub_i32 s18, s1, s19 // (op 2295) 


s_add_u32 s50, s8, 64 // (op 2295) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2295) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 2295) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 2295) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 2295) 

s_mul_i32 s1, s13, s50 // (op 2295) 

s_mul_i32 s19, s10, s1 // (op 2295) 

s_mul_i32 s1, s18, s19 // (op 2295) 

s_sub_u32 s19, s16, s1 // (op 2295) 

s_mov_b32 s1, s19 // (op 2295) 
s_mul_hi_i32 s49, s1, s14 // (op 2295) 
s_add_i32 s50, s49, s1 // (op 2295) 


s_ashr_i32 s1, s50, 31 // (op 2295) 
s_cmp_eq_i32 s14, 0 // (op 2295) 
s_mov_b32 s49, scc // (op 2295) 
s_cmp_eq_i32 s49, 0 // (op 2295) 
s_cselect_b32 s51, 0, -1 // (op 2295) 

s_and_b32 s49, s46, 31 // (op 2295) 
s_lshl_b32 s64, 1, s49 // (op 2295) 
s_add_i32 s65, s64, s51 // (op 2295) 


s_and_b32 s51, s1, s65 // (op 2295) 


s_add_i32 s1, s50, s51 // (op 2295) 


s_ashr_i32 s50, s1, s49 // (op 2295) 


s_lshl_b32 s1, s46, 24 // (op 2295) 
s_ashr_i32 s49, s1, 31 // (op 2295) 

s_xor_b32 s1, s50, s49 // (op 2295) 

s_sub_i32 s50, s1, s49 // (op 2295) 


s_mul_i32 s1, s48, s13 // (op 2295) 
s_add_u32 s64, s6, 64 // (op 2295) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 2295) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 2295) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 2295) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 6 // (op 2295) 

s_sub_i32 s49, s64, s1 // (op 2295) 

s_mul_i32 s51, s50, s49 // (op 2295) 


s_sub_u32 s49, s19, s51 // (op 2295) 


s_add_u32 s19, s49, s1 // (op 2295) 


s_cmp_eq_i32 s18, 0 // (op 2295) 
s_mov_b32 s1, scc // (op 2295) 

s_cmp_eq_i32 s1, 0 // (op 2295) 
s_cselect_b32 s18, s19, s0 // (op 2295) 



v_lshl_add_u32 v24, s18, 6, v19 // (op 2295) 


v_mul_lo_u32 v19, v18, s34 // (op 2295) 

v_add_u32 v18, v24, v19 // (op 2295) 


s_mul_i32 s1, s34, s39 // (op 2297) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 2297) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2297) least significant
s_mul_hi_u32 s19, s34, s38 // (op 2297) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2297) most significant: sum
s_add_u32 s1, s1, s19 // (op 2297) most significant: sum


s_mov_b32 s64, s36 // (op 2297) reg expression
s_mov_b32 s65, s37 // (op 2297) reg expression
s_mov_b32 s66, s0 // (op 2297) reg expression
s_mov_b32 s67, 131072 // (op 2297) call()

s_mul_hi_i32 s0, s16, s12 // (op 2314) 
v_and_b32 v19, -64, v1 // (op 2314) 
s_add_i32 s1, s0, s16 // (op 2314) 

v_sub_u32 v24, v1, v19 // (op 2314) 

s_ashr_i32 s0, s1, 31 // (op 2314) 
v_lshrrev_b32 v19, 6, v1 // (op 2314) 
s_cmp_eq_i32 s12, 0 // (op 2314) 
s_mov_b32 s18, scc // (op 2314) 
s_cmp_eq_i32 s18, 0 // (op 2314) 
s_cselect_b32 s19, 0, -1 // (op 2314) 

s_and_b32 s18, s15, 31 // (op 2314) 
s_lshl_b32 s49, 1, s18 // (op 2314) 
s_add_i32 s50, s49, s19 // (op 2314) 


s_and_b32 s19, s0, s50 // (op 2314) 


s_add_i32 s0, s1, s19 // (op 2314) 


s_ashr_i32 s1, s0, s18 // (op 2314) 


s_lshl_b32 s0, s15, 24 // (op 2314) 
s_ashr_i32 s18, s0, 31 // (op 2314) 

s_xor_b32 s0, s1, s18 // (op 2314) 

s_sub_i32 s1, s0, s18 // (op 2314) 


s_mul_hi_i32 s0, s1, s44 // (op 2314) 
s_add_i32 s18, s0, s1 // (op 2314) 

s_ashr_i32 s0, s18, 31 // (op 2314) 
s_cmp_eq_i32 s44, 0 // (op 2314) 
s_mov_b32 s19, scc // (op 2314) 
s_cmp_eq_i32 s19, 0 // (op 2314) 
s_cselect_b32 s49, 0, -1 // (op 2314) 

s_and_b32 s19, s45, 31 // (op 2314) 
s_lshl_b32 s50, 1, s19 // (op 2314) 
s_add_i32 s51, s50, s49 // (op 2314) 


s_and_b32 s49, s0, s51 // (op 2314) 


s_add_i32 s0, s18, s49 // (op 2314) 


s_ashr_i32 s18, s0, s19 // (op 2314) 


s_lshl_b32 s0, s45, 24 // (op 2314) 
s_ashr_i32 s19, s0, 31 // (op 2314) 

s_xor_b32 s0, s18, s19 // (op 2314) 

s_sub_i32 s18, s0, s19 // (op 2314) 


s_mul_i32 s0, s1, s13 // (op 2314) 

s_sub_u32 s1, s16, s0 // (op 2314) 

s_mul_i32 s0, s18, s13 // (op 2314) 
s_add_u32 s0, s0, s1 // (op 2314) 


s_mul_hi_i32 s1, s16, s11 // (op 2314) 
s_add_i32 s18, s1, s16 // (op 2314) 

s_ashr_i32 s1, s18, 31 // (op 2314) 
s_cmp_eq_i32 s11, 0 // (op 2314) 
s_mov_b32 s19, scc // (op 2314) 
s_cmp_eq_i32 s19, 0 // (op 2314) 
s_cselect_b32 s49, 0, -1 // (op 2314) 

s_and_b32 s19, s47, 31 // (op 2314) 
s_lshl_b32 s50, 1, s19 // (op 2314) 
s_add_i32 s51, s50, s49 // (op 2314) 


s_and_b32 s49, s1, s51 // (op 2314) 


s_add_i32 s1, s18, s49 // (op 2314) 


s_ashr_i32 s18, s1, s19 // (op 2314) 


s_lshl_b32 s1, s47, 24 // (op 2314) 
s_ashr_i32 s19, s1, 31 // (op 2314) 

s_xor_b32 s1, s18, s19 // (op 2314) 

s_sub_i32 s18, s1, s19 // (op 2314) 


s_add_u32 s50, s8, 64 // (op 2314) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2314) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 2314) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 2314) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 2314) 

s_mul_i32 s1, s13, s50 // (op 2314) 

s_mul_i32 s19, s10, s1 // (op 2314) 

s_mul_i32 s1, s18, s19 // (op 2314) 

s_sub_u32 s19, s16, s1 // (op 2314) 

s_mov_b32 s1, s19 // (op 2314) 
s_mul_hi_i32 s49, s1, s14 // (op 2314) 
s_add_i32 s50, s49, s1 // (op 2314) 


s_ashr_i32 s1, s50, 31 // (op 2314) 
s_cmp_eq_i32 s14, 0 // (op 2314) 
s_mov_b32 s49, scc // (op 2314) 
s_cmp_eq_i32 s49, 0 // (op 2314) 
s_cselect_b32 s51, 0, -1 // (op 2314) 

s_and_b32 s49, s46, 31 // (op 2314) 
s_lshl_b32 s68, 1, s49 // (op 2314) 
s_add_i32 s69, s68, s51 // (op 2314) 


s_and_b32 s51, s1, s69 // (op 2314) 


s_add_i32 s1, s50, s51 // (op 2314) 


s_ashr_i32 s50, s1, s49 // (op 2314) 


s_lshl_b32 s1, s46, 24 // (op 2314) 
s_ashr_i32 s49, s1, 31 // (op 2314) 

s_xor_b32 s1, s50, s49 // (op 2314) 

s_sub_i32 s50, s1, s49 // (op 2314) 


s_mul_i32 s1, s48, s13 // (op 2314) 
s_add_u32 s68, s6, 64 // (op 2314) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 2314) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 2314) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 2314) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 6 // (op 2314) 

s_sub_i32 s49, s68, s1 // (op 2314) 

s_mul_i32 s51, s50, s49 // (op 2314) 


s_sub_u32 s49, s19, s51 // (op 2314) 


s_add_u32 s19, s49, s1 // (op 2314) 


s_cmp_eq_i32 s18, 0 // (op 2314) 
s_mov_b32 s1, scc // (op 2314) 

s_cmp_eq_i32 s1, 0 // (op 2314) 
s_cselect_b32 s18, s19, s0 // (op 2314) 



v_lshl_add_u32 v25, s18, 6, v24 // (op 2314) 


v_mul_lo_u32 v24, v19, s34 // (op 2314) 

v_add_u32 v19, v25, v24 // (op 2314) 


s_mul_i32 s1, s34, s39 // (op 2316) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 2316) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2316) least significant
s_mul_hi_u32 s19, s34, s38 // (op 2316) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2316) most significant: sum
s_add_u32 s1, s1, s19 // (op 2316) most significant: sum


s_mov_b32 s64, s36 // (op 2316) reg expression
s_mov_b32 s65, s37 // (op 2316) reg expression
s_mov_b32 s66, s0 // (op 2316) reg expression
s_mov_b32 s67, 131072 // (op 2316) call()

s_mul_hi_i32 s0, s16, s12 // (op 2359) 
v_and_b32 v24, -64, v1 // (op 2359) 
s_add_i32 s1, s0, s16 // (op 2359) 

v_sub_u32 v25, v1, v24 // (op 2359) 

s_ashr_i32 s0, s1, 31 // (op 2359) 
v_lshrrev_b32 v24, 6, v1 // (op 2359) 
s_cmp_eq_i32 s12, 0 // (op 2359) 
s_mov_b32 s18, scc // (op 2359) 
s_cmp_eq_i32 s18, 0 // (op 2359) 
s_cselect_b32 s19, 0, -1 // (op 2359) 

s_and_b32 s18, s15, 31 // (op 2359) 
s_lshl_b32 s49, 1, s18 // (op 2359) 
s_add_i32 s50, s49, s19 // (op 2359) 


s_and_b32 s19, s0, s50 // (op 2359) 


s_add_i32 s0, s1, s19 // (op 2359) 


s_ashr_i32 s1, s0, s18 // (op 2359) 


s_lshl_b32 s0, s15, 24 // (op 2359) 
s_ashr_i32 s18, s0, 31 // (op 2359) 

s_xor_b32 s0, s1, s18 // (op 2359) 

s_sub_i32 s1, s0, s18 // (op 2359) 


s_mul_hi_i32 s0, s1, s44 // (op 2359) 
s_add_i32 s18, s0, s1 // (op 2359) 

s_ashr_i32 s0, s18, 31 // (op 2359) 
s_cmp_eq_i32 s44, 0 // (op 2359) 
s_mov_b32 s19, scc // (op 2359) 
s_cmp_eq_i32 s19, 0 // (op 2359) 
s_cselect_b32 s49, 0, -1 // (op 2359) 

s_and_b32 s19, s45, 31 // (op 2359) 
s_lshl_b32 s50, 1, s19 // (op 2359) 
s_add_i32 s51, s50, s49 // (op 2359) 


s_and_b32 s49, s0, s51 // (op 2359) 


s_add_i32 s0, s18, s49 // (op 2359) 


s_ashr_i32 s18, s0, s19 // (op 2359) 


s_lshl_b32 s0, s45, 24 // (op 2359) 
s_ashr_i32 s19, s0, 31 // (op 2359) 

s_xor_b32 s0, s18, s19 // (op 2359) 

s_sub_i32 s18, s0, s19 // (op 2359) 


s_add_u32 s50, s8, 64 // (op 2359) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2359) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 2359) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 2359) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 2359) 

s_mul_i32 s0, s18, s50 // (op 2359) 

s_sub_i32 s49, s1, s0 // (op 2359) 


s_mul_hi_i32 s0, s16, s11 // (op 2359) 
s_add_i32 s1, s0, s16 // (op 2359) 

s_ashr_i32 s0, s1, 31 // (op 2359) 
s_cmp_eq_i32 s11, 0 // (op 2359) 
s_mov_b32 s18, scc // (op 2359) 
s_cmp_eq_i32 s18, 0 // (op 2359) 
s_cselect_b32 s19, 0, -1 // (op 2359) 

s_and_b32 s18, s47, 31 // (op 2359) 
s_lshl_b32 s68, 1, s18 // (op 2359) 
s_add_i32 s69, s68, s19 // (op 2359) 


s_and_b32 s19, s0, s69 // (op 2359) 


s_add_i32 s0, s1, s19 // (op 2359) 


s_ashr_i32 s1, s0, s18 // (op 2359) 


s_lshl_b32 s0, s47, 24 // (op 2359) 
s_ashr_i32 s18, s0, 31 // (op 2359) 

s_xor_b32 s0, s1, s18 // (op 2359) 

s_sub_i32 s1, s0, s18 // (op 2359) 


s_mul_i32 s0, s13, s50 // (op 2359) 

s_mul_i32 s18, s10, s0 // (op 2359) 

s_mul_i32 s0, s1, s18 // (op 2359) 

s_sub_u32 s18, s16, s0 // (op 2359) 

s_mul_hi_i32 s0, s18, s14 // (op 2359) 
s_add_i32 s19, s0, s18 // (op 2359) 


s_ashr_i32 s0, s19, 31 // (op 2359) 
s_cmp_eq_i32 s14, 0 // (op 2359) 
s_mov_b32 s18, scc // (op 2359) 
s_cmp_eq_i32 s18, 0 // (op 2359) 
s_cselect_b32 s50, 0, -1 // (op 2359) 

s_and_b32 s18, s46, 31 // (op 2359) 
s_lshl_b32 s51, 1, s18 // (op 2359) 
s_add_i32 s68, s51, s50 // (op 2359) 


s_and_b32 s50, s0, s68 // (op 2359) 


s_add_i32 s0, s19, s50 // (op 2359) 


s_ashr_i32 s19, s0, s18 // (op 2359) 


s_lshl_b32 s0, s46, 24 // (op 2359) 
s_ashr_i32 s18, s0, 31 // (op 2359) 

s_xor_b32 s0, s19, s18 // (op 2359) 

s_sub_i32 s19, s0, s18 // (op 2359) 


s_cmp_eq_i32 s1, 0 // (op 2359) 
s_mov_b32 s0, scc // (op 2359) 

s_cmp_eq_i32 s0, 0 // (op 2359) 
s_cselect_b32 s1, s19, s49 // (op 2359) 



v_lshl_add_u32 v26, s1, 6, v25 // (op 2359) 


v_mul_lo_u32 v25, v24, s40 // (op 2359) 
v_add_u32 v25, v25, v26 // (op 2359) 


s_mul_i32 s1, s40, s39 // (op 2361) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 2361) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 2361) least significant
s_mul_hi_u32 s19, s40, s38 // (op 2361) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2361) most significant: sum
s_add_u32 s1, s1, s19 // (op 2361) most significant: sum


s_mov_b32 s68, s42 // (op 2361) reg expression
s_mov_b32 s69, s43 // (op 2361) reg expression
s_mov_b32 s70, s0 // (op 2361) reg expression
s_mov_b32 s71, 131072 // (op 2361) call()

s_mul_hi_i32 s0, s16, s12 // (op 2378) 
v_and_b32 v24, -64, v1 // (op 2378) 
s_add_i32 s1, s0, s16 // (op 2378) 

v_sub_u32 v26, v1, v24 // (op 2378) 

s_ashr_i32 s0, s1, 31 // (op 2378) 
v_lshrrev_b32 v24, 6, v1 // (op 2378) 
s_cmp_eq_i32 s12, 0 // (op 2378) 
s_mov_b32 s18, scc // (op 2378) 
s_cmp_eq_i32 s18, 0 // (op 2378) 
s_cselect_b32 s19, 0, -1 // (op 2378) 

s_and_b32 s18, s15, 31 // (op 2378) 
s_lshl_b32 s49, 1, s18 // (op 2378) 
s_add_i32 s50, s49, s19 // (op 2378) 


s_and_b32 s19, s0, s50 // (op 2378) 


s_add_i32 s0, s1, s19 // (op 2378) 


s_ashr_i32 s1, s0, s18 // (op 2378) 


s_lshl_b32 s0, s15, 24 // (op 2378) 
s_ashr_i32 s18, s0, 31 // (op 2378) 

s_xor_b32 s0, s1, s18 // (op 2378) 

s_sub_i32 s1, s0, s18 // (op 2378) 


s_mul_hi_i32 s0, s1, s44 // (op 2378) 
s_add_i32 s18, s0, s1 // (op 2378) 

s_ashr_i32 s0, s18, 31 // (op 2378) 
s_cmp_eq_i32 s44, 0 // (op 2378) 
s_mov_b32 s19, scc // (op 2378) 
s_cmp_eq_i32 s19, 0 // (op 2378) 
s_cselect_b32 s49, 0, -1 // (op 2378) 

s_and_b32 s19, s45, 31 // (op 2378) 
s_lshl_b32 s50, 1, s19 // (op 2378) 
s_add_i32 s51, s50, s49 // (op 2378) 


s_and_b32 s49, s0, s51 // (op 2378) 


s_add_i32 s0, s18, s49 // (op 2378) 


s_ashr_i32 s18, s0, s19 // (op 2378) 


s_lshl_b32 s0, s45, 24 // (op 2378) 
s_ashr_i32 s19, s0, 31 // (op 2378) 

s_xor_b32 s0, s18, s19 // (op 2378) 

s_sub_i32 s18, s0, s19 // (op 2378) 


s_add_u32 s50, s8, 64 // (op 2378) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2378) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2378) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2378) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 2378) 

s_mul_i32 s0, s18, s50 // (op 2378) 

s_sub_i32 s49, s1, s0 // (op 2378) 


s_mul_hi_i32 s0, s16, s11 // (op 2378) 
s_add_i32 s1, s0, s16 // (op 2378) 

s_ashr_i32 s0, s1, 31 // (op 2378) 
s_cmp_eq_i32 s11, 0 // (op 2378) 
s_mov_b32 s18, scc // (op 2378) 
s_cmp_eq_i32 s18, 0 // (op 2378) 
s_cselect_b32 s19, 0, -1 // (op 2378) 

s_and_b32 s18, s47, 31 // (op 2378) 
s_lshl_b32 s72, 1, s18 // (op 2378) 
s_add_i32 s73, s72, s19 // (op 2378) 


s_and_b32 s19, s0, s73 // (op 2378) 


s_add_i32 s0, s1, s19 // (op 2378) 


s_ashr_i32 s1, s0, s18 // (op 2378) 


s_lshl_b32 s0, s47, 24 // (op 2378) 
s_ashr_i32 s18, s0, 31 // (op 2378) 

s_xor_b32 s0, s1, s18 // (op 2378) 

s_sub_i32 s1, s0, s18 // (op 2378) 


s_mul_i32 s0, s13, s50 // (op 2378) 

s_mul_i32 s18, s10, s0 // (op 2378) 

s_mul_i32 s0, s1, s18 // (op 2378) 

s_sub_u32 s18, s16, s0 // (op 2378) 

s_mul_hi_i32 s0, s18, s14 // (op 2378) 
s_add_i32 s19, s0, s18 // (op 2378) 


s_ashr_i32 s0, s19, 31 // (op 2378) 
s_cmp_eq_i32 s14, 0 // (op 2378) 
s_mov_b32 s18, scc // (op 2378) 
s_cmp_eq_i32 s18, 0 // (op 2378) 
s_cselect_b32 s50, 0, -1 // (op 2378) 

s_and_b32 s18, s46, 31 // (op 2378) 
s_lshl_b32 s51, 1, s18 // (op 2378) 
s_add_i32 s72, s51, s50 // (op 2378) 


s_and_b32 s50, s0, s72 // (op 2378) 


s_add_i32 s0, s19, s50 // (op 2378) 


s_ashr_i32 s19, s0, s18 // (op 2378) 


s_lshl_b32 s0, s46, 24 // (op 2378) 
s_ashr_i32 s18, s0, 31 // (op 2378) 

s_xor_b32 s0, s19, s18 // (op 2378) 

s_sub_i32 s19, s0, s18 // (op 2378) 


s_cmp_eq_i32 s1, 0 // (op 2378) 
s_mov_b32 s0, scc // (op 2378) 

s_cmp_eq_i32 s0, 0 // (op 2378) 
s_cselect_b32 s1, s19, s49 // (op 2378) 



v_lshl_add_u32 v27, s1, 6, v26 // (op 2378) 


v_mul_lo_u32 v26, v24, s40 // (op 2378) 
v_add_u32 v26, v26, v27 // (op 2378) 


s_mul_i32 s1, s40, s39 // (op 2380) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 2380) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 2380) least significant
s_mul_hi_u32 s19, s40, s38 // (op 2380) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2380) most significant: sum
s_add_u32 s1, s1, s19 // (op 2380) most significant: sum


s_mov_b32 s68, s42 // (op 2380) reg expression
s_mov_b32 s69, s43 // (op 2380) reg expression
s_mov_b32 s70, s0 // (op 2380) reg expression
s_mov_b32 s71, 131072 // (op 2380) call()

v_and_b32 v24, -4, v1 // (op 2506) 
v_sub_u32 v27, v1, v24 // (op 2506) 

v_lshlrev_b32 v24, 4, v27 // (op 2506) 

v_lshrrev_b32 v27, 2, v1 // (op 2506) 
v_lshl_add_u32 v28, v27, 6, v24 // (op 2506) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 2506) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[30:31], 6 // (op 2113) 
v_mov_b64 v[28:29], s[18:19] // (op 2113) convert


v_mov_b64 v[30:31], v[20:21] // (op 2113) reg expression
s_mov_b32 m0, s0 // (op 2113) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 2113) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 2113) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2113) most significant half

s_add_u32 m0, m0, 4096 // (op 2113) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 2113) Load value direct to lds





v_mov_b32 v24, v25 // (op 1943) reg expression

buffer_load_ubyte v27, v24, s[68:71], 0 offen offset: 0 // (op 1943) Load value

v_and_b32 v24, -4, v1 // (op 2449) 
v_sub_u32 v28, v1, v24 // (op 2449) 

v_lshlrev_b32 v24, 4, v28 // (op 2449) 

v_lshrrev_b32 v28, 2, v1 // (op 2449) 
v_lshl_add_u32 v29, v28, 6, v24 // (op 2449) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 2449) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 2118) 
v_mov_b64 v[28:29], s[18:19] // (op 2118) convert


v_mov_b64 v[30:31], v[14:15] // (op 2118) reg expression
s_add_u32 s1, 8192, s0 // (op 2118) 
s_mov_b32 m0, s1 // (op 2118) reg expression
buffer_load_dwordx4 v30, s[20:23], 0 offen offset: 0 lds // (op 2118) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 2118) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2118) most significant half

s_add_u32 m0, m0, 4096 // (op 2118) 
buffer_load_dwordx4 v30, s[20:23], 0 offen offset: 0 lds // (op 2118) Load value direct to lds






v_mov_b32 v24, v18 // (op 1949) reg expression

buffer_load_ubyte v28, v24, s[64:67], 0 offen offset: 0 // (op 1949) Load value

v_and_b32 v24, -4, v1 // (op 2733) 
v_sub_u32 v29, v1, v24 // (op 2733) 

v_lshlrev_b32 v24, 4, v29 // (op 2733) 

v_lshrrev_b32 v29, 2, v1 // (op 2733) 
v_lshl_add_u32 v30, v29, 6, v24 // (op 2733) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 2733) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[30:31], 6 // (op 2123) 
v_mov_b64 v[30:31], s[18:19] // (op 2123) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 2123) 
v_mov_b32 v24, s18 // (op 2123) 
v_mov_b32 v29, s19 // (op 2123) 
v_add_co_u32 v32, s[50:51], v20, v24 // (op 2123) least significant half
v_addc_co_u32 v33, s[50:51], v21, v29, s[50:51] // (op 2123) most significant half




s_add_u32 s1, 16384, s0 // (op 2123) 
s_mov_b32 m0, s1 // (op 2123) reg expression
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 2123) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 2123) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 2123) most significant half

s_add_u32 m0, m0, 4096 // (op 2123) 
buffer_load_dwordx4 v32, s[52:55], 0 offen offset: 0 lds // (op 2123) Load value direct to lds







s_lshl_b32 s0, s40, 2 // (op 1955) 
v_add_u32 v24, v25, s0 // (op 1955) 


buffer_load_ubyte v20, v24, s[68:71], 0 offen offset: 0 // (op 1955) Load value


v_and_b32 v21, -4, v1 // (op 2689) 
v_sub_u32 v24, v1, v21 // (op 2689) 

v_lshlrev_b32 v21, 4, v24 // (op 2689) 

v_lshrrev_b32 v24, 2, v1 // (op 2689) 
v_lshl_add_u32 v25, v24, 6, v21 // (op 2689) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 2689) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 2125) 
v_mov_b64 v[24:25], s[18:19] // (op 2125) convert


s_lshl_b64 s[18:19], s[24:25], 7 // (op 2125) 
v_mov_b32 v21, s18 // (op 2125) 
v_mov_b32 v29, s19 // (op 2125) 
v_add_co_u32 v30, s[50:51], v14, v21 // (op 2125) least significant half
v_addc_co_u32 v31, s[50:51], v15, v29, s[50:51] // (op 2125) most significant half




s_add_u32 s1, 24576, s0 // (op 2125) 
s_mov_b32 m0, s1 // (op 2125) reg expression
buffer_load_dwordx4 v30, s[20:23], 0 offen offset: 0 lds // (op 2125) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 2125) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2125) most significant half

s_add_u32 m0, m0, 4096 // (op 2125) 
buffer_load_dwordx4 v30, s[20:23], 0 offen offset: 0 lds // (op 2125) Load value direct to lds







s_lshl_b32 s0, s34, 2 // (op 1961) 
v_add_u32 v21, v18, s0 // (op 1961) 


buffer_load_ubyte v14, v21, s[64:67], 0 offen offset: 0 // (op 1961) Load value


v_and_b32 v15, -64, v1 // (op 2622) 
v_sub_u32 v18, v1, v15 // (op 2622) 

v_and_b32 v15, 4294967232, v1 // (op 2622) 
v_add_u32 v21, v15, v18 // (op 2622) 





s_waitcnt vmcnt(9)
ds_write_b8 v21, v27 offset:32768 // (op 1964) Store local data 

v_and_b32 v15, -64, v1 // (op 2564) 
v_sub_u32 v18, v1, v15 // (op 2564) 

v_and_b32 v15, 4294967232, v1 // (op 2564) 
v_add_u32 v21, v15, v18 // (op 2564) 





s_waitcnt vmcnt(6)
ds_write_b8 v21, v28 offset:33024 // (op 1966) Store local data 

s_barrier  // (op 2889) 



ds_read_b64_tr_b8 v[30:31], v4 // (op 1991) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:512 // (op 1991) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:4096 // (op 1991) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:4608 // (op 1991) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2924) 


ds_read_u8 v15, v5 offset:32768 // (op 2011) Load local data 



ds_read_b64_tr_b8 v[38:39], v4 offset:32 // (op 1996) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:544 // (op 1996) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:4128 // (op 1996) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:4640 // (op 1996) Transpose load local data 


ds_read_u8 v18, v5 offset:32800 // (op 2016) Load local data 
s_barrier  // (op 2873) 



ds_read_b64_tr_b8 v[46:47], v0 offset:8192 // (op 1981) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:8704 // (op 1981) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:12288 // (op 1981) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:12800 // (op 1981) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 2906) 


ds_read_u8 v21, v3 offset:33024 // (op 2001) Load local data 



ds_read_b64_tr_b8 v[54:55], v0 offset:8224 // (op 1986) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:8736 // (op 1986) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:12320 // (op 1986) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:12832 // (op 1986) Transpose load local data 


ds_read_u8 v29, v3 offset:33056 // (op 2006) Load local data 
v_and_b32 v24, -4, v1 // (op 2421) 
v_sub_u32 v25, v1, v24 // (op 2421) 

v_lshlrev_b32 v24, 4, v25 // (op 2421) 

v_lshrrev_b32 v25, 2, v1 // (op 2421) 
v_lshl_add_u32 v62, v25, 6, v24 // (op 2421) 


s_nop 0
v_readfirstlane_b32 s0, v62 // (op 2421) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 2478) 
v_sub_u32 v25, v1, v24 // (op 2478) 

v_lshlrev_b32 v24, 4, v25 // (op 2478) 

v_lshrrev_b32 v25, 2, v1 // (op 2478) 
v_lshl_add_u32 v62, v25, 6, v24 // (op 2478) 


s_nop 0
v_readfirstlane_b32 s1, v62 // (op 2478) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -64, v1 // (op 2535) 
v_sub_u32 v25, v1, v24 // (op 2535) 

v_and_b32 v24, 4294967232, v1 // (op 2535) 
v_add_u32 v62, v24, v25 // (op 2535) 


v_and_b32 v24, -64, v1 // (op 2593) 
v_sub_u32 v25, v1, v24 // (op 2593) 

v_and_b32 v24, 4294967232, v1 // (op 2593) 
v_add_u32 v63, v24, v25 // (op 2593) 


v_and_b32 v24, -4, v1 // (op 2674) 
v_sub_u32 v25, v1, v24 // (op 2674) 

v_lshlrev_b32 v24, 4, v25 // (op 2674) 

v_lshrrev_b32 v25, 2, v1 // (op 2674) 
v_lshl_add_u32 v64, v25, 6, v24 // (op 2674) 


s_nop 0
v_readfirstlane_b32 s18, v64 // (op 2674) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 2718) 
v_sub_u32 v25, v1, v24 // (op 2718) 

v_lshlrev_b32 v24, 4, v25 // (op 2718) 

v_lshrrev_b32 v25, 2, v1 // (op 2718) 
v_lshl_add_u32 v64, v25, 6, v24 // (op 2718) 


s_nop 0
v_readfirstlane_b32 s19, v64 // (op 2718) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -64, v1 // (op 2762) 
v_sub_u32 v25, v1, v24 // (op 2762) 

v_and_b32 v24, 4294967232, v1 // (op 2762) 
v_add_u32 v64, v24, v25 // (op 2762) 


v_and_b32 v24, -64, v1 // (op 2791) 
v_sub_u32 v25, v1, v24 // (op 2791) 

v_and_b32 v24, 4294967232, v1 // (op 2791) 
v_add_u32 v65, v24, v25 // (op 2791) 


s_mov_b32 s49, 0 // (op 589) call()
s_ashr_i64 s[50:51], s[28:29], 7 // (op 588) 
s_mov_b32 s72, s50 // (op 588) convert

s_and_b32 s50, s72, 4294967294 // (op 588) 

s_cmp_lt_u32 s49, s50 // (op 588) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoop_588 // (op 588) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoop_588 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoop_588:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[30:37], a[0:3], v21, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 88) 
s_barrier  // (op 2892) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[38:45], a[4:7], v21, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 376) 
s_add_u32 s49, s49, 2 // (op 590) 
s_lshl_b64 s[50:51], s[30:31], 6 // (op 2085) 
v_mov_b64 v[24:25], s[50:51] // (op 2085) convert


v_mov_b64 v[66:67], v[22:23] // (op 2085) 
s_lshl_b64 s[50:51], s[30:31], 7 // (op 2085) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 2085) 

v_mov_b32 v68, s72 // (op 2085) 
v_mov_b32 v69, s73 // (op 2085) 
v_add_co_u32 v70, s[50:51], v66, v68 // (op 2085) least significant half
v_addc_co_u32 v71, s[50:51], v67, v69, s[50:51] // (op 2085) most significant half




s_mov_b32 m0, s1 // (op 2085) reg expression
buffer_load_dwordx4 v70, s[52:55], 0 offen offset: 0 lds // (op 2085) Load value direct to lds
v_add_co_u32 v70, s[50:51], v70, v24 // (op 2085) least significant half
v_addc_co_u32 v71, s[50:51], 0, v71, s[50:51] // (op 2085) most significant half

s_add_u32 m0, m0, 4096 // (op 2085) 
buffer_load_dwordx4 v70, s[52:55], 0 offen offset: 0 lds // (op 2085) Load value direct to lds


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[30:37], a[8:11], v29, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 673) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:61], v[38:45], a[12:15], v29, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 627) 
v_mov_b32 v24, v26 // (op 25) 
s_lshl_b32 s50, s40, 3 // (op 25) 
v_add_u32 v25, v24, s50 // (op 25) 


buffer_load_ubyte v27, v25, s[68:71], 0 offen offset: 0 // (op 25) Load value

s_barrier  // (op 2876) 

s_lshl_b64 s[50:51], s[24:25], 6 // (op 2080) 
v_mov_b64 v[68:69], s[50:51] // (op 2080) convert


v_mov_b64 v[70:71], v[16:17] // (op 2080) 
s_lshl_b64 s[50:51], s[24:25], 7 // (op 2080) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 2080) 

v_mov_b32 v25, s72 // (op 2080) 
v_mov_b32 v72, s73 // (op 2080) 
v_add_co_u32 v74, s[50:51], v70, v25 // (op 2080) least significant half
v_addc_co_u32 v75, s[50:51], v71, v72, s[50:51] // (op 2080) most significant half




s_add_u32 s50, 8192, s0 // (op 2080) 
s_nop 0
s_mov_b32 m0, s50 // (op 2080) reg expression
buffer_load_dwordx4 v74, s[20:23], 0 offen offset: 0 lds // (op 2080) Load value direct to lds
v_add_co_u32 v74, s[72:73], v74, v68 // (op 2080) least significant half
v_addc_co_u32 v75, s[72:73], 0, v75, s[72:73] // (op 2080) most significant half

s_add_u32 m0, m0, 4096 // (op 2080) 
buffer_load_dwordx4 v74, s[20:23], 0 offen offset: 0 lds // (op 2080) Load value direct to lds





v_mov_b32 v25, v19 // (op 17) 
s_lshl_b32 s50, s34, 3 // (op 17) 
v_add_u32 v68, v25, s50 // (op 17) 


buffer_load_ubyte v28, v68, s[64:67], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 2964) 



s_waitcnt vmcnt(9)
ds_write_b8 v65, v20 offset:33280 // (op 255) Store local data 
s_barrier  // (op 2958) 



s_waitcnt vmcnt(6)
ds_write_b8 v64, v14 offset:33536 // (op 254) Store local data 
s_barrier  // (op 2947) 



ds_read_b64_tr_b8 v[72:73], v8 offset:16384 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v8 offset:16896 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v8 offset:20480 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v8 offset:20992 // (op 247) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2961) 


ds_read_u8 v68, v9 offset:33280 // (op 251) Load local data 



ds_read_b64_tr_b8 v[80:81], v8 offset:16416 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v8 offset:16928 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v8 offset:20512 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v8 offset:21024 // (op 381) Transpose load local data 


ds_read_u8 v69, v9 offset:33312 // (op 385) Load local data 
s_barrier  // (op 2941) 
s_barrier  // (op 3005) 



ds_read_b64_tr_b8 v[46:47], v6 offset:24576 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v6 offset:25088 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v6 offset:28672 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v6 offset:29184 // (op 245) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 2954) 


ds_read_u8 v21, v7 offset:33536 // (op 249) Load local data 



ds_read_b64_tr_b8 v[54:55], v6 offset:24608 // (op 630) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v6 offset:25120 // (op 630) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v6 offset:28704 // (op 630) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v6 offset:29216 // (op 630) Transpose load local data 


ds_read_u8 v29, v7 offset:33568 // (op 633) Load local data 
s_barrier  // (op 3027) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[72:79], a[0:3], v21, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 252) 
s_barrier  // (op 2951) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[80:87], a[4:7], v21, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 377) 
s_lshl_b64 s[50:51], s[30:31], 6 // (op 2108) 
v_mov_b64 v[88:89], s[50:51] // (op 2108) convert


s_lshl_b64 s[50:51], s[30:31], 7 // (op 2108) 
s_mul_i32 s73, 3, s51 // (op 2108) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 2108) least significant
s_mul_hi_u32 s74, 3, s50 // (op 2108) most significant: high of low * low
s_add_u32 s73, s73, s74 // (op 2108) most significant: sum


v_mov_b32 v90, s72 // (op 2108) 
v_mov_b32 v91, s73 // (op 2108) 
v_add_co_u32 v92, s[50:51], v66, v90 // (op 2108) least significant half
v_addc_co_u32 v93, s[50:51], v67, v91, s[50:51] // (op 2108) most significant half




s_add_u32 s50, 16384, s19 // (op 2108) 
s_nop 0
s_mov_b32 m0, s50 // (op 2108) reg expression
buffer_load_dwordx4 v92, s[52:55], 0 offen offset: 0 lds // (op 2108) Load value direct to lds
v_add_co_u32 v92, s[72:73], v92, v88 // (op 2108) least significant half
v_addc_co_u32 v93, s[72:73], 0, v93, s[72:73] // (op 2108) most significant half

s_add_u32 m0, m0, 4096 // (op 2108) 
buffer_load_dwordx4 v92, s[52:55], 0 offen offset: 0 lds // (op 2108) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[72:79], a[8:11], v29, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 674) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:61], v[80:87], a[12:15], v29, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 628) 
s_lshl_b32 s50, s40, 2 // (op 258) 


s_mul_i32 s51, 3, s50 // (op 258) 

v_add_u32 v66, v24, s51 // (op 258) 


buffer_load_ubyte v20, v66, s[68:71], 0 offen offset: 0 // (op 258) Load value

s_lshl_b64 s[50:51], s[30:31], 7 // (op 2242) 

s_lshl_b64 s[72:73], s[50:51], 1 // (op 2242) 

s_barrier  // (op 2944) 
s_lshl_b32 s50, s40, 3 // (op 2370) 

v_add_u32 v26, v26, s50 // (op 2370) 

v_mov_b32 v24, s72 // (op 2242) 
v_mov_b32 v66, s73 // (op 2242) 
s_lshl_b64 s[50:51], s[24:25], 6 // (op 2103) 
v_mov_b64 v[68:69], s[50:51] // (op 2103) convert


s_lshl_b64 s[50:51], s[24:25], 7 // (op 2103) 
s_mul_i32 s75, 3, s51 // (op 2103) most significant: low of low * high
s_mul_i32 s74, 3, s50 // (op 2103) least significant
s_mul_hi_u32 s76, 3, s50 // (op 2103) most significant: high of low * low
s_add_u32 s75, s75, s76 // (op 2103) most significant: sum


v_mov_b32 v67, s74 // (op 2103) 
v_mov_b32 v72, s75 // (op 2103) 
v_add_co_u32 v22, s[50:51], v22, v24 // (op 2242) least significant half
v_addc_co_u32 v23, s[50:51], v23, v66, s[50:51] // (op 2242) most significant half




v_add_co_u32 v74, s[50:51], v70, v67 // (op 2103) least significant half
v_addc_co_u32 v75, s[50:51], v71, v72, s[50:51] // (op 2103) most significant half




s_add_u32 s50, 24576, s18 // (op 2103) 
s_nop 0
s_mov_b32 m0, s50 // (op 2103) reg expression
buffer_load_dwordx4 v74, s[20:23], 0 offen offset: 0 lds // (op 2103) Load value direct to lds
v_add_co_u32 v74, s[72:73], v74, v68 // (op 2103) least significant half
v_addc_co_u32 v75, s[72:73], 0, v75, s[72:73] // (op 2103) most significant half

s_add_u32 m0, m0, 4096 // (op 2103) 
buffer_load_dwordx4 v74, s[20:23], 0 offen offset: 0 lds // (op 2103) Load value direct to lds






s_lshl_b32 s50, s34, 2 // (op 257) 
s_mul_i32 s51, 3, s50 // (op 257) 

v_add_u32 v24, v25, s51 // (op 257) 


s_lshl_b64 s[50:51], s[24:25], 7 // (op 2178) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 2178) 

v_mov_b32 v66, s72 // (op 2178) 
v_mov_b32 v67, s73 // (op 2178) 
v_add_co_u32 v16, s[50:51], v16, v66 // (op 2178) least significant half
v_addc_co_u32 v17, s[50:51], v17, v67, s[50:51] // (op 2178) most significant half




buffer_load_ubyte v14, v24, s[64:67], 0 offen offset: 0 // (op 257) Load value


s_barrier  // (op 2927) 
s_lshl_b32 s50, s34, 3 // (op 2306) 



v_add_u32 v19, v19, s50 // (op 2306) 

s_waitcnt vmcnt(9)
ds_write_b8 v63, v27 offset:32768 // (op 30) Store local data 
s_barrier  // (op 2909) 



s_waitcnt vmcnt(6)
ds_write_b8 v62, v28 offset:33024 // (op 21) Store local data 
s_barrier  // (op 2898) 



ds_read_b64_tr_b8 v[30:31], v4 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:512 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:4096 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:4608 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2934) 


ds_read_u8 v15, v5 offset:32768 // (op 85) Load local data 



ds_read_b64_tr_b8 v[38:39], v4 offset:32 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:544 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:4128 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:4640 // (op 371) Transpose load local data 


ds_read_u8 v18, v5 offset:32800 // (op 375) Load local data 
s_barrier  // (op 2882) 



ds_read_b64_tr_b8 v[46:47], v0 offset:8192 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:8704 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:12288 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:12800 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 2916) 


ds_read_u8 v21, v3 offset:33024 // (op 80) Load local data 



ds_read_b64_tr_b8 v[54:55], v0 offset:8224 // (op 622) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:8736 // (op 622) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:12320 // (op 622) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:12832 // (op 622) Transpose load local data 


ds_read_u8 v29, v3 offset:33056 // (op 625) Load local data 
s_ashr_i64 s[50:51], s[28:29], 7 // (op 588) 
s_mov_b32 s72, s50 // (op 588) convert

s_and_b32 s50, s72, 4294967294 // (op 588) 

s_cmp_lt_u32 s49, s50 // (op 588) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoop_588 // (op 588) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoop_588 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoop_588:

















s_barrier  // (op 3011) 
s_mul_hi_i32 s0, s16, s12 // (op 2138) 
v_and_b32 v19, -4, v1 // (op 2138) 
s_add_i32 s1, s0, s16 // (op 2138) 

v_sub_u32 v15, v1, v19 // (op 2138) 

s_ashr_i32 s0, s1, 31 // (op 2138) 
v_lshlrev_b32 v19, 4, v15 // (op 2138) 

s_cmp_eq_i32 s12, 0 // (op 2138) 
s_mov_b32 s18, scc // (op 2138) 
v_lshrrev_b32 v15, 2, v1 // (op 2138) 
s_cmp_eq_i32 s18, 0 // (op 2138) 
s_cselect_b32 s19, 0, -1 // (op 2138) 

s_and_b32 s18, s15, 31 // (op 2138) 
s_lshl_b32 s49, 1, s18 // (op 2138) 
s_add_i32 s50, s49, s19 // (op 2138) 


s_and_b32 s19, s0, s50 // (op 2138) 


s_add_i32 s0, s1, s19 // (op 2138) 


s_ashr_i32 s1, s0, s18 // (op 2138) 


s_lshl_b32 s0, s15, 24 // (op 2138) 
s_ashr_i32 s18, s0, 31 // (op 2138) 

s_xor_b32 s0, s1, s18 // (op 2138) 

s_sub_i32 s1, s0, s18 // (op 2138) 


s_mul_hi_i32 s0, s1, s44 // (op 2138) 
s_add_i32 s18, s0, s1 // (op 2138) 

s_ashr_i32 s0, s18, 31 // (op 2138) 
s_cmp_eq_i32 s44, 0 // (op 2138) 
s_mov_b32 s19, scc // (op 2138) 
s_cmp_eq_i32 s19, 0 // (op 2138) 
s_cselect_b32 s49, 0, -1 // (op 2138) 

s_and_b32 s19, s45, 31 // (op 2138) 
s_lshl_b32 s50, 1, s19 // (op 2138) 
s_add_i32 s51, s50, s49 // (op 2138) 


s_and_b32 s49, s0, s51 // (op 2138) 


s_add_i32 s0, s18, s49 // (op 2138) 


s_ashr_i32 s18, s0, s19 // (op 2138) 


s_lshl_b32 s0, s45, 24 // (op 2138) 
s_ashr_i32 s19, s0, 31 // (op 2138) 

s_xor_b32 s0, s18, s19 // (op 2138) 

s_sub_i32 s18, s0, s19 // (op 2138) 


s_mul_i32 s0, s1, s13 // (op 2138) 

s_sub_u32 s1, s16, s0 // (op 2138) 

s_mul_i32 s0, s18, s13 // (op 2138) 
s_add_u32 s0, s0, s1 // (op 2138) 


s_mul_hi_i32 s1, s16, s11 // (op 2138) 
s_add_i32 s18, s1, s16 // (op 2138) 

s_ashr_i32 s1, s18, 31 // (op 2138) 
s_cmp_eq_i32 s11, 0 // (op 2138) 
s_mov_b32 s19, scc // (op 2138) 
s_cmp_eq_i32 s19, 0 // (op 2138) 
s_cselect_b32 s49, 0, -1 // (op 2138) 

s_and_b32 s19, s47, 31 // (op 2138) 
s_lshl_b32 s50, 1, s19 // (op 2138) 
s_add_i32 s51, s50, s49 // (op 2138) 


s_and_b32 s49, s1, s51 // (op 2138) 


s_add_i32 s1, s18, s49 // (op 2138) 


s_ashr_i32 s18, s1, s19 // (op 2138) 


s_lshl_b32 s1, s47, 24 // (op 2138) 
s_ashr_i32 s19, s1, 31 // (op 2138) 

s_xor_b32 s1, s18, s19 // (op 2138) 

s_sub_i32 s18, s1, s19 // (op 2138) 


s_add_u32 s50, s8, 64 // (op 2138) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2138) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2138) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2138) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 2138) 

s_mul_i32 s1, s13, s50 // (op 2138) 

s_mul_i32 s19, s10, s1 // (op 2138) 

s_mul_i32 s1, s18, s19 // (op 2138) 

s_sub_u32 s19, s16, s1 // (op 2138) 

s_mov_b32 s1, s19 // (op 2138) 
s_mul_hi_i32 s49, s1, s14 // (op 2138) 
s_add_i32 s50, s49, s1 // (op 2138) 


s_ashr_i32 s1, s50, 31 // (op 2138) 
s_cmp_eq_i32 s14, 0 // (op 2138) 
s_mov_b32 s49, scc // (op 2138) 
s_cmp_eq_i32 s49, 0 // (op 2138) 
s_cselect_b32 s51, 0, -1 // (op 2138) 

s_and_b32 s49, s46, 31 // (op 2138) 
s_lshl_b32 s72, 1, s49 // (op 2138) 
s_add_i32 s73, s72, s51 // (op 2138) 


s_and_b32 s51, s1, s73 // (op 2138) 


s_add_i32 s1, s50, s51 // (op 2138) 


s_ashr_i32 s50, s1, s49 // (op 2138) 


s_lshl_b32 s1, s46, 24 // (op 2138) 
s_ashr_i32 s49, s1, 31 // (op 2138) 

s_xor_b32 s1, s50, s49 // (op 2138) 

s_sub_i32 s50, s1, s49 // (op 2138) 


s_mul_i32 s1, s48, s13 // (op 2138) 
s_add_u32 s72, s6, 64 // (op 2138) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 2138) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 2138) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 2138) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 2138) 

s_sub_i32 s49, s72, s1 // (op 2138) 

s_mul_i32 s51, s50, s49 // (op 2138) 


s_sub_u32 s49, s19, s51 // (op 2138) 


s_add_u32 s19, s49, s1 // (op 2138) 


s_cmp_eq_i32 s18, 0 // (op 2138) 
s_mov_b32 s1, scc // (op 2138) 

s_cmp_eq_i32 s1, 0 // (op 2138) 
s_cselect_b32 s18, s19, s0 // (op 2138) 



v_lshl_add_u32 v16, s18, 6, v19 // (op 2138) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2138) 
v_mov_b32 v23, 0 // (op 2138) convert
v_mov_b32 v22, v16 // (op 2138) convert

s_lshr_b32 s18, s0, 1 // (op 2138) 

v_lshl_add_u32 v19, s18, 8, v15 // (op 2138) 


v_mul_lo_u32 v17, v19, s25 // (op 2138) most significant: low of low * high
v_mul_hi_u32 v15, v19, s24 // (op 2138) most significant: high of low * low
v_mul_lo_u32 v16, v19, s24 // (op 2138) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 2138) most significant: sum


v_add_co_u32 v24, s[0:1], v22, v16 // (op 2138) least significant half
v_addc_co_u32 v25, s[0:1], v23, v17, s[0:1] // (op 2138) most significant half



s_mul_i32 s1, s24, s29 // (op 2140) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 2140) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 2140) least significant
s_mul_hi_u32 s19, s24, s28 // (op 2140) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2140) most significant: sum
s_add_u32 s1, s1, s19 // (op 2140) most significant: sum


s_mov_b32 s20, s26 // (op 2140) reg expression
s_mov_b32 s21, s27 // (op 2140) reg expression
s_mov_b32 s22, s0 // (op 2140) reg expression
s_mov_b32 s23, 131072 // (op 2140) call()

s_mul_hi_i32 s0, s16, s12 // (op 2208) 
v_and_b32 v19, -4, v1 // (op 2208) 
s_add_i32 s1, s0, s16 // (op 2208) 

v_sub_u32 v15, v1, v19 // (op 2208) 

s_ashr_i32 s0, s1, 31 // (op 2208) 
v_lshlrev_b32 v19, 4, v15 // (op 2208) 

s_cmp_eq_i32 s12, 0 // (op 2208) 
s_mov_b32 s18, scc // (op 2208) 
v_lshrrev_b32 v15, 2, v1 // (op 2208) 
s_cmp_eq_i32 s18, 0 // (op 2208) 
s_cselect_b32 s19, 0, -1 // (op 2208) 

s_and_b32 s18, s15, 31 // (op 2208) 
s_lshl_b32 s49, 1, s18 // (op 2208) 
s_add_i32 s50, s49, s19 // (op 2208) 


s_and_b32 s19, s0, s50 // (op 2208) 


s_add_i32 s0, s1, s19 // (op 2208) 


s_ashr_i32 s1, s0, s18 // (op 2208) 


s_lshl_b32 s0, s15, 24 // (op 2208) 
s_ashr_i32 s18, s0, 31 // (op 2208) 

s_xor_b32 s0, s1, s18 // (op 2208) 

s_sub_i32 s1, s0, s18 // (op 2208) 


s_mul_hi_i32 s0, s1, s44 // (op 2208) 
s_add_i32 s18, s0, s1 // (op 2208) 

s_ashr_i32 s0, s18, 31 // (op 2208) 
s_cmp_eq_i32 s44, 0 // (op 2208) 
s_mov_b32 s19, scc // (op 2208) 
s_cmp_eq_i32 s19, 0 // (op 2208) 
s_cselect_b32 s49, 0, -1 // (op 2208) 

s_and_b32 s19, s45, 31 // (op 2208) 
s_lshl_b32 s50, 1, s19 // (op 2208) 
s_add_i32 s51, s50, s49 // (op 2208) 


s_and_b32 s49, s0, s51 // (op 2208) 


s_add_i32 s0, s18, s49 // (op 2208) 


s_ashr_i32 s18, s0, s19 // (op 2208) 


s_lshl_b32 s0, s45, 24 // (op 2208) 
s_ashr_i32 s19, s0, 31 // (op 2208) 

s_xor_b32 s0, s18, s19 // (op 2208) 

s_sub_i32 s18, s0, s19 // (op 2208) 


s_add_u32 s50, s8, 64 // (op 2208) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2208) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2208) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2208) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 2208) 

s_mul_i32 s0, s18, s50 // (op 2208) 

s_sub_i32 s49, s1, s0 // (op 2208) 


s_mul_hi_i32 s0, s16, s11 // (op 2208) 
s_add_i32 s1, s0, s16 // (op 2208) 

s_ashr_i32 s0, s1, 31 // (op 2208) 
s_cmp_eq_i32 s11, 0 // (op 2208) 
s_mov_b32 s18, scc // (op 2208) 
s_cmp_eq_i32 s18, 0 // (op 2208) 
s_cselect_b32 s19, 0, -1 // (op 2208) 

s_and_b32 s18, s47, 31 // (op 2208) 
s_lshl_b32 s72, 1, s18 // (op 2208) 
s_add_i32 s73, s72, s19 // (op 2208) 


s_and_b32 s19, s0, s73 // (op 2208) 


s_add_i32 s0, s1, s19 // (op 2208) 


s_ashr_i32 s1, s0, s18 // (op 2208) 


s_lshl_b32 s0, s47, 24 // (op 2208) 
s_ashr_i32 s18, s0, 31 // (op 2208) 

s_xor_b32 s0, s1, s18 // (op 2208) 

s_sub_i32 s1, s0, s18 // (op 2208) 


s_mul_i32 s0, s13, s50 // (op 2208) 

s_mul_i32 s18, s10, s0 // (op 2208) 

s_mul_i32 s0, s1, s18 // (op 2208) 

s_sub_u32 s18, s16, s0 // (op 2208) 

s_mul_hi_i32 s0, s18, s14 // (op 2208) 
s_add_i32 s19, s0, s18 // (op 2208) 


s_ashr_i32 s0, s19, 31 // (op 2208) 
s_cmp_eq_i32 s14, 0 // (op 2208) 
s_mov_b32 s18, scc // (op 2208) 
s_cmp_eq_i32 s18, 0 // (op 2208) 
s_cselect_b32 s50, 0, -1 // (op 2208) 

s_and_b32 s18, s46, 31 // (op 2208) 
s_lshl_b32 s51, 1, s18 // (op 2208) 
s_add_i32 s72, s51, s50 // (op 2208) 


s_and_b32 s50, s0, s72 // (op 2208) 


s_add_i32 s0, s19, s50 // (op 2208) 


s_ashr_i32 s19, s0, s18 // (op 2208) 


s_lshl_b32 s0, s46, 24 // (op 2208) 
s_ashr_i32 s18, s0, 31 // (op 2208) 

s_xor_b32 s0, s19, s18 // (op 2208) 

s_sub_i32 s19, s0, s18 // (op 2208) 


s_cmp_eq_i32 s1, 0 // (op 2208) 
s_mov_b32 s0, scc // (op 2208) 

s_cmp_eq_i32 s0, 0 // (op 2208) 
s_cselect_b32 s1, s19, s49 // (op 2208) 



v_lshl_add_u32 v16, s1, 6, v19 // (op 2208) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2208) 
v_mov_b32 v23, 0 // (op 2208) convert
v_mov_b32 v22, v16 // (op 2208) convert

s_lshr_b32 s18, s0, 1 // (op 2208) 

v_lshl_add_u32 v19, s18, 8, v15 // (op 2208) 


v_mul_lo_u32 v17, v19, s31 // (op 2208) most significant: low of low * high
v_mul_hi_u32 v15, v19, s30 // (op 2208) most significant: high of low * low
v_mul_lo_u32 v16, v19, s30 // (op 2208) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 2208) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v22 // (op 2208) least significant half
v_addc_co_u32 v17, s[0:1], v17, v23, s[0:1] // (op 2208) most significant half



s_mul_i32 s1, s30, s29 // (op 2210) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 2210) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 2210) least significant
s_mul_hi_u32 s19, s30, s28 // (op 2210) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2210) most significant: sum
s_add_u32 s1, s1, s19 // (op 2210) most significant: sum


s_mov_b32 s52, s32 // (op 2210) reg expression
s_mov_b32 s53, s33 // (op 2210) reg expression
s_mov_b32 s54, s0 // (op 2210) reg expression
s_mov_b32 s55, 131072 // (op 2210) call()

s_mul_hi_i32 s0, s16, s12 // (op 2272) 
v_and_b32 v15, -64, v1 // (op 2272) 
s_add_i32 s1, s0, s16 // (op 2272) 

v_sub_u32 v19, v1, v15 // (op 2272) 

s_ashr_i32 s0, s1, 31 // (op 2272) 
v_lshrrev_b32 v15, 6, v1 // (op 2272) 
s_cmp_eq_i32 s12, 0 // (op 2272) 
s_mov_b32 s18, scc // (op 2272) 
s_cmp_eq_i32 s18, 0 // (op 2272) 
s_cselect_b32 s19, 0, -1 // (op 2272) 

s_and_b32 s18, s15, 31 // (op 2272) 
s_lshl_b32 s49, 1, s18 // (op 2272) 
s_add_i32 s50, s49, s19 // (op 2272) 


s_and_b32 s19, s0, s50 // (op 2272) 


s_add_i32 s0, s1, s19 // (op 2272) 


s_ashr_i32 s1, s0, s18 // (op 2272) 


s_lshl_b32 s0, s15, 24 // (op 2272) 
s_ashr_i32 s18, s0, 31 // (op 2272) 

s_xor_b32 s0, s1, s18 // (op 2272) 

s_sub_i32 s1, s0, s18 // (op 2272) 


s_mul_hi_i32 s0, s1, s44 // (op 2272) 
s_add_i32 s18, s0, s1 // (op 2272) 

s_ashr_i32 s0, s18, 31 // (op 2272) 
s_cmp_eq_i32 s44, 0 // (op 2272) 
s_mov_b32 s19, scc // (op 2272) 
s_cmp_eq_i32 s19, 0 // (op 2272) 
s_cselect_b32 s49, 0, -1 // (op 2272) 

s_and_b32 s19, s45, 31 // (op 2272) 
s_lshl_b32 s50, 1, s19 // (op 2272) 
s_add_i32 s51, s50, s49 // (op 2272) 


s_and_b32 s49, s0, s51 // (op 2272) 


s_add_i32 s0, s18, s49 // (op 2272) 


s_ashr_i32 s18, s0, s19 // (op 2272) 


s_lshl_b32 s0, s45, 24 // (op 2272) 
s_ashr_i32 s19, s0, 31 // (op 2272) 

s_xor_b32 s0, s18, s19 // (op 2272) 

s_sub_i32 s18, s0, s19 // (op 2272) 


s_mul_i32 s0, s1, s13 // (op 2272) 

s_sub_u32 s1, s16, s0 // (op 2272) 

s_mul_i32 s0, s18, s13 // (op 2272) 
s_add_u32 s0, s0, s1 // (op 2272) 


s_mul_hi_i32 s1, s16, s11 // (op 2272) 
s_add_i32 s18, s1, s16 // (op 2272) 

s_ashr_i32 s1, s18, 31 // (op 2272) 
s_cmp_eq_i32 s11, 0 // (op 2272) 
s_mov_b32 s19, scc // (op 2272) 
s_cmp_eq_i32 s19, 0 // (op 2272) 
s_cselect_b32 s49, 0, -1 // (op 2272) 

s_and_b32 s19, s47, 31 // (op 2272) 
s_lshl_b32 s50, 1, s19 // (op 2272) 
s_add_i32 s51, s50, s49 // (op 2272) 


s_and_b32 s49, s1, s51 // (op 2272) 


s_add_i32 s1, s18, s49 // (op 2272) 


s_ashr_i32 s18, s1, s19 // (op 2272) 


s_lshl_b32 s1, s47, 24 // (op 2272) 
s_ashr_i32 s19, s1, 31 // (op 2272) 

s_xor_b32 s1, s18, s19 // (op 2272) 

s_sub_i32 s18, s1, s19 // (op 2272) 


s_add_u32 s50, s8, 64 // (op 2272) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2272) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2272) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2272) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 2272) 

s_mul_i32 s1, s13, s50 // (op 2272) 

s_mul_i32 s19, s10, s1 // (op 2272) 

s_mul_i32 s1, s18, s19 // (op 2272) 

s_sub_u32 s19, s16, s1 // (op 2272) 

s_mov_b32 s1, s19 // (op 2272) 
s_mul_hi_i32 s49, s1, s14 // (op 2272) 
s_add_i32 s50, s49, s1 // (op 2272) 


s_ashr_i32 s1, s50, 31 // (op 2272) 
s_cmp_eq_i32 s14, 0 // (op 2272) 
s_mov_b32 s49, scc // (op 2272) 
s_cmp_eq_i32 s49, 0 // (op 2272) 
s_cselect_b32 s51, 0, -1 // (op 2272) 

s_and_b32 s49, s46, 31 // (op 2272) 
s_lshl_b32 s72, 1, s49 // (op 2272) 
s_add_i32 s73, s72, s51 // (op 2272) 


s_and_b32 s51, s1, s73 // (op 2272) 


s_add_i32 s1, s50, s51 // (op 2272) 


s_ashr_i32 s50, s1, s49 // (op 2272) 


s_lshl_b32 s1, s46, 24 // (op 2272) 
s_ashr_i32 s49, s1, 31 // (op 2272) 

s_xor_b32 s1, s50, s49 // (op 2272) 

s_sub_i32 s50, s1, s49 // (op 2272) 


s_mul_i32 s1, s48, s13 // (op 2272) 
s_add_u32 s72, s6, 64 // (op 2272) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 2272) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 2272) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 2272) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 2272) 

s_sub_i32 s49, s72, s1 // (op 2272) 

s_mul_i32 s51, s50, s49 // (op 2272) 


s_sub_u32 s49, s19, s51 // (op 2272) 


s_add_u32 s19, s49, s1 // (op 2272) 


s_cmp_eq_i32 s18, 0 // (op 2272) 
s_mov_b32 s1, scc // (op 2272) 

s_cmp_eq_i32 s1, 0 // (op 2272) 
s_cselect_b32 s18, s19, s0 // (op 2272) 



v_lshl_add_u32 v22, s18, 6, v19 // (op 2272) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2272) 
s_lshr_b32 s18, s0, 1 // (op 2272) 

v_lshl_add_u32 v19, s18, 3, v15 // (op 2272) 


v_mul_lo_u32 v15, v19, s34 // (op 2272) 

v_add_u32 v19, v22, v15 // (op 2272) 


s_mul_i32 s1, s34, s39 // (op 2274) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 2274) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2274) least significant
s_mul_hi_u32 s19, s34, s38 // (op 2274) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2274) most significant: sum
s_add_u32 s1, s1, s19 // (op 2274) most significant: sum


s_mov_b32 s64, s36 // (op 2274) reg expression
s_mov_b32 s65, s37 // (op 2274) reg expression
s_mov_b32 s66, s0 // (op 2274) reg expression
s_mov_b32 s67, 131072 // (op 2274) call()

s_mul_hi_i32 s0, s16, s12 // (op 2336) 
v_and_b32 v15, -64, v1 // (op 2336) 
s_add_i32 s1, s0, s16 // (op 2336) 

v_sub_u32 v22, v1, v15 // (op 2336) 

s_ashr_i32 s0, s1, 31 // (op 2336) 
v_lshrrev_b32 v15, 6, v1 // (op 2336) 
s_cmp_eq_i32 s12, 0 // (op 2336) 
s_mov_b32 s18, scc // (op 2336) 
s_cmp_eq_i32 s18, 0 // (op 2336) 
s_cselect_b32 s19, 0, -1 // (op 2336) 

s_and_b32 s18, s15, 31 // (op 2336) 
s_lshl_b32 s49, 1, s18 // (op 2336) 
s_add_i32 s50, s49, s19 // (op 2336) 


s_and_b32 s19, s0, s50 // (op 2336) 


s_add_i32 s0, s1, s19 // (op 2336) 


s_ashr_i32 s1, s0, s18 // (op 2336) 


s_lshl_b32 s0, s15, 24 // (op 2336) 
s_ashr_i32 s18, s0, 31 // (op 2336) 

s_xor_b32 s0, s1, s18 // (op 2336) 

s_sub_i32 s1, s0, s18 // (op 2336) 


s_mul_hi_i32 s0, s1, s44 // (op 2336) 
s_add_i32 s18, s0, s1 // (op 2336) 

s_ashr_i32 s0, s18, 31 // (op 2336) 
s_cmp_eq_i32 s44, 0 // (op 2336) 
s_mov_b32 s19, scc // (op 2336) 
s_cmp_eq_i32 s19, 0 // (op 2336) 
s_cselect_b32 s49, 0, -1 // (op 2336) 

s_and_b32 s19, s45, 31 // (op 2336) 
s_lshl_b32 s50, 1, s19 // (op 2336) 
s_add_i32 s51, s50, s49 // (op 2336) 


s_and_b32 s49, s0, s51 // (op 2336) 


s_add_i32 s0, s18, s49 // (op 2336) 


s_ashr_i32 s18, s0, s19 // (op 2336) 


s_lshl_b32 s0, s45, 24 // (op 2336) 
s_ashr_i32 s19, s0, 31 // (op 2336) 

s_xor_b32 s0, s18, s19 // (op 2336) 

s_sub_i32 s18, s0, s19 // (op 2336) 


s_add_u32 s50, s8, 64 // (op 2336) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2336) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2336) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2336) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 2336) 

s_mul_i32 s0, s18, s50 // (op 2336) 

s_sub_i32 s49, s1, s0 // (op 2336) 


s_mul_hi_i32 s0, s16, s11 // (op 2336) 
s_add_i32 s1, s0, s16 // (op 2336) 

s_ashr_i32 s0, s1, 31 // (op 2336) 
s_cmp_eq_i32 s11, 0 // (op 2336) 
s_mov_b32 s18, scc // (op 2336) 
s_cmp_eq_i32 s18, 0 // (op 2336) 
s_cselect_b32 s19, 0, -1 // (op 2336) 

s_and_b32 s18, s47, 31 // (op 2336) 
s_lshl_b32 s72, 1, s18 // (op 2336) 
s_add_i32 s73, s72, s19 // (op 2336) 


s_and_b32 s19, s0, s73 // (op 2336) 


s_add_i32 s0, s1, s19 // (op 2336) 


s_ashr_i32 s1, s0, s18 // (op 2336) 


s_lshl_b32 s0, s47, 24 // (op 2336) 
s_ashr_i32 s18, s0, 31 // (op 2336) 

s_xor_b32 s0, s1, s18 // (op 2336) 

s_sub_i32 s1, s0, s18 // (op 2336) 


s_mul_i32 s0, s13, s50 // (op 2336) 

s_mul_i32 s18, s10, s0 // (op 2336) 

s_mul_i32 s0, s1, s18 // (op 2336) 

s_sub_u32 s18, s16, s0 // (op 2336) 

s_mul_hi_i32 s0, s18, s14 // (op 2336) 
s_add_i32 s19, s0, s18 // (op 2336) 


s_ashr_i32 s0, s19, 31 // (op 2336) 
s_cmp_eq_i32 s14, 0 // (op 2336) 
s_mov_b32 s18, scc // (op 2336) 
s_cmp_eq_i32 s18, 0 // (op 2336) 
s_cselect_b32 s50, 0, -1 // (op 2336) 

s_and_b32 s18, s46, 31 // (op 2336) 
s_lshl_b32 s51, 1, s18 // (op 2336) 
s_add_i32 s72, s51, s50 // (op 2336) 


s_and_b32 s50, s0, s72 // (op 2336) 


s_add_i32 s0, s19, s50 // (op 2336) 


s_ashr_i32 s19, s0, s18 // (op 2336) 


s_lshl_b32 s0, s46, 24 // (op 2336) 
s_ashr_i32 s18, s0, 31 // (op 2336) 

s_xor_b32 s0, s19, s18 // (op 2336) 

s_sub_i32 s19, s0, s18 // (op 2336) 


s_cmp_eq_i32 s1, 0 // (op 2336) 
s_mov_b32 s0, scc // (op 2336) 

s_cmp_eq_i32 s0, 0 // (op 2336) 
s_cselect_b32 s1, s19, s49 // (op 2336) 



v_lshl_add_u32 v23, s1, 6, v22 // (op 2336) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2336) 
s_lshr_b32 s18, s0, 1 // (op 2336) 

v_lshl_add_u32 v22, s18, 3, v15 // (op 2336) 


v_mul_lo_u32 v15, v22, s40 // (op 2336) 
v_add_u32 v15, v15, v23 // (op 2336) 


s_mul_i32 s1, s40, s39 // (op 2338) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 2338) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 2338) least significant
s_mul_hi_u32 s19, s40, s38 // (op 2338) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2338) most significant: sum
s_add_u32 s1, s1, s19 // (op 2338) most significant: sum


s_mov_b32 s68, s42 // (op 2338) reg expression
s_mov_b32 s69, s43 // (op 2338) reg expression
s_mov_b32 s70, s0 // (op 2338) reg expression
s_mov_b32 s71, 131072 // (op 2338) call()

v_and_b32 v22, -4, v1 // (op 2434) 
v_sub_u32 v23, v1, v22 // (op 2434) 

v_lshlrev_b32 v22, 4, v23 // (op 2434) 

v_lshrrev_b32 v23, 2, v1 // (op 2434) 
v_lshl_add_u32 v26, v23, 6, v22 // (op 2434) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 2434) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 2491) 
v_sub_u32 v23, v1, v22 // (op 2491) 

v_lshlrev_b32 v22, 4, v23 // (op 2491) 

v_lshrrev_b32 v23, 2, v1 // (op 2491) 
v_lshl_add_u32 v26, v23, 6, v22 // (op 2491) 


s_nop 0
v_readfirstlane_b32 s1, v26 // (op 2491) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 2548) 
v_sub_u32 v23, v1, v22 // (op 2548) 

v_and_b32 v22, 4294967232, v1 // (op 2548) 
v_add_u32 v26, v22, v23 // (op 2548) 


v_and_b32 v22, -64, v1 // (op 2606) 
v_sub_u32 v23, v1, v22 // (op 2606) 

v_and_b32 v22, 4294967232, v1 // (op 2606) 
v_add_u32 v27, v22, v23 // (op 2606) 


s_ashr_i64 s[18:19], s[28:29], 7 // (op 594) 
s_and_b32 s49, s18, 4294967294 // (op 594) 

s_ashr_i64 s[18:19], s[28:29], 7 // (op 593) 
s_mov_b32 s50, s18 // (op 593) convert

s_cmp_lt_u32 s49, s50 // (op 593) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_593 // (op 593) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_593 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoopTail_593:
s_barrier  // (op 2879) 
s_add_u32 s49, s49, 1 // (op 595) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 2090) 
v_mov_b64 v[20:21], s[18:19] // (op 2090) convert


v_mov_b64 v[22:23], v[24:25] // (op 2090) 
v_mov_b64 v[28:29], v[22:23] // (op 2090) reg expression
s_add_u32 s18, 8192, s0 // (op 2090) 
s_mov_b32 m0, s18 // (op 2090) reg expression
buffer_load_dwordx4 v28, s[20:23], 0 offen offset: 0 lds // (op 2090) Load value direct to lds
v_add_co_u32 v28, s[50:51], v28, v20 // (op 2090) least significant half
v_addc_co_u32 v29, s[50:51], 0, v29, s[50:51] // (op 2090) most significant half

s_add_u32 m0, m0, 4096 // (op 2090) 
buffer_load_dwordx4 v28, s[20:23], 0 offen offset: 0 lds // (op 2090) Load value direct to lds




s_lshl_b64 s[18:19], s[24:25], 7 // (op 2128) 
v_mov_b32 v14, s18 // (op 2128) 
v_mov_b32 v18, s19 // (op 2128) 
v_add_co_u32 v24, s[50:51], v24, v14 // (op 2128) least significant half
v_addc_co_u32 v25, s[50:51], v25, v18, s[50:51] // (op 2128) most significant half




s_barrier  // (op 2895) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 2096) 
v_mov_b64 v[20:21], s[18:19] // (op 2096) convert


v_mov_b64 v[22:23], v[16:17] // (op 2096) 
v_mov_b64 v[28:29], v[22:23] // (op 2096) reg expression
s_mov_b32 m0, s1 // (op 2096) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 2096) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v20 // (op 2096) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 2096) most significant half

s_add_u32 m0, m0, 4096 // (op 2096) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 2096) Load value direct to lds



s_lshl_b64 s[18:19], s[30:31], 7 // (op 2200) 
v_mov_b32 v14, s18 // (op 2200) 
v_mov_b32 v18, s19 // (op 2200) 
v_add_co_u32 v16, s[50:51], v16, v14 // (op 2200) least significant half
v_addc_co_u32 v17, s[50:51], v17, v18, s[50:51] // (op 2200) most significant half






v_mov_b32 v14, v19 // (op 191) 
v_mov_b32 v18, v14 // (op 191) reg expression

buffer_load_ubyte v20, v18, s[64:67], 0 offen offset: 0 // (op 191) Load value


s_lshl_b32 s18, s34, 2 // (op 2264) 
v_add_u32 v19, v19, s18 // (op 2264) 



s_barrier  // (op 2912) 
v_mov_b32 v14, v15 // (op 192) 



v_mov_b32 v18, v14 // (op 192) reg expression

buffer_load_ubyte v21, v18, s[68:71], 0 offen offset: 0 // (op 192) Load value


s_lshl_b32 s18, s40, 2 // (op 2328) 
v_add_u32 v15, v15, s18 // (op 2328) 

s_waitcnt vmcnt(1)
ds_write_b8 v26, v20 offset:33024 // (op 188) Store local data 

s_barrier  // (op 2885) 



s_barrier  // (op 2930) 
ds_read_b64_tr_b8 v[28:29], v0 offset:8192 // (op 179) Transpose load local data 



ds_read_b64_tr_b8 v[30:31], v0 offset:8704 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:12288 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:12800 // (op 179) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v27, v21 offset:32768 // (op 189) Store local data 

s_barrier  // (op 2902) 



ds_read_b64_tr_b8 v[36:37], v4 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:512 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:4096 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:4608 // (op 181) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 2920) 


ds_read_u8 v14, v3 offset:33024 // (op 183) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2937) 


ds_read_u8 v18, v5 offset:32768 // (op 185) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:43], a[0:3], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 186) 
ds_read_b64_tr_b8 v[44:45], v4 offset:32 // (op 350) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:544 // (op 350) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v4 offset:4128 // (op 350) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v4 offset:4640 // (op 350) Transpose load local data 


ds_read_u8 v20, v5 offset:32800 // (op 354) Load local data 



ds_read_b64_tr_b8 v[52:53], v0 offset:8224 // (op 568) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:35], v[44:51], a[4:7], v14, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 355) 
ds_read_b64_tr_b8 v[54:55], v0 offset:8736 // (op 568) Transpose load local data 


ds_read_b64_tr_b8 v[56:57], v0 offset:12320 // (op 568) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:12832 // (op 568) Transpose load local data 


ds_read_u8 v14, v3 offset:33056 // (op 571) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:59], v[36:43], a[8:11], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 573) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:59], v[44:51], a[12:15], v14, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 604) 




s_ashr_i64 s[18:19], s[28:29], 7 // (op 593) 
s_mov_b32 s50, s18 // (op 593) convert

s_cmp_lt_u32 s49, s50 // (op 593) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoopTail_593 // (op 593) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoopTail_593 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_593:























s_barrier  // (op 3015) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalFalse_0_beta_133
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
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalBottom_0_beta_133
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalFalse_0_beta_133:


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
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalBottom_0_beta_133:












s_endpgm  // End of RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
.LRR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__end:
.size RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_, .LRR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__end-RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 112
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 33792 // lds bytes
  .amdhsa_accum_offset 96
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
    DataTypeB: B8
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
  - .name: RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 33792
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 77
    .vgpr_count: 94
    .agpr_count: 16
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

