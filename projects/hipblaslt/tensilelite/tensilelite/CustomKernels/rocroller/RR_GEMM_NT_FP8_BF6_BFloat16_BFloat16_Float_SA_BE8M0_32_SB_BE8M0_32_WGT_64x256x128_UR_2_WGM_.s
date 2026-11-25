// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_,@function
RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s12 // (op 2687) 
s_mul_hi_i32 s1, s16, s12 // (op 2928) 
v_and_b32 v0, 63, v1 // (op 2743) 
v_lshrrev_b32 v3, 4, v0 // (op 2743) 
v_and_b32 v4, -2, v3 // (op 2743) 
v_sub_u32 v5, v3, v4 // (op 2743) 


v_and_b32 v3, 48, v1 // (op 2743) 
v_sub_u32 v4, v0, v3 // (op 2743) 

v_and_b32 v3, 1, v4 // (op 2743) 
v_lshl_add_u32 v6, v5, 1, v3 // (op 2743) 


v_lshlrev_b32 v3, 3, v6 // (op 2743) 

v_lshrrev_b32 v5, 7, v1 // (op 2743) 
v_lshl_add_u32 v6, v5, 5, v3 // (op 2743) 


v_lshrrev_b32 v3, 1, v4 // (op 2743) 

v_lshrrev_b32 v4, 5, v0 // (op 2743) 

v_lshl_add_u32 v0, v4, 4, v3 // (op 2743) 


v_lshl_add_u32 v3, v0, 6, v6 // (op 2743) 


v_lshrrev_b32 v0, 7, v1 // (op 2854) 
v_and_b32 v4, 31, v1 // (op 2854) 
v_lshl_add_u32 v5, v0, 5, v4 // (op 2854) 


v_and_b32 v0, 63, v1 // (op 2854) 
v_lshrrev_b32 v4, 5, v0 // (op 2854) 

v_lshl_add_u32 v0, v4, 6, v5 // (op 2854) 


v_and_b32 v4, 63, v1 // (op 2969) 
v_lshrrev_b32 v5, 4, v4 // (op 2969) 
v_and_b32 v6, -2, v5 // (op 2969) 
v_sub_u32 v7, v5, v6 // (op 2969) 


v_and_b32 v5, 48, v1 // (op 2969) 
v_sub_u32 v6, v4, v5 // (op 2969) 

v_and_b32 v5, 1, v6 // (op 2969) 
v_lshl_add_u32 v8, v7, 1, v5 // (op 2969) 


v_lshlrev_b32 v5, 3, v8 // (op 2969) 

v_lshrrev_b32 v7, 7, v1 // (op 2969) 
v_lshl_add_u32 v8, v7, 5, v5 // (op 2969) 


v_lshrrev_b32 v5, 1, v6 // (op 2969) 

v_lshrrev_b32 v6, 5, v4 // (op 2969) 

v_lshl_add_u32 v4, v6, 4, v5 // (op 2969) 


v_lshl_add_u32 v5, v4, 6, v8 // (op 2969) 


v_lshrrev_b32 v4, 7, v1 // (op 3022) 
v_and_b32 v6, 31, v1 // (op 3022) 
v_lshl_add_u32 v7, v4, 5, v6 // (op 3022) 


v_and_b32 v4, 63, v1 // (op 3022) 
v_lshrrev_b32 v6, 5, v4 // (op 3022) 

v_lshl_add_u32 v4, v6, 6, v7 // (op 3022) 


v_lshrrev_b32 v6, 7, v1 // (op 2687) 
s_add_i32 s18, s0, s16 // (op 2687) 

v_and_b32 v7, 63, v1 // (op 2687) 
s_ashr_i32 s0, s18, 31 // (op 2687) 
v_lshrrev_b32 v8, 5, v7 // (op 2687) 
s_cmp_eq_i32 s12, 0 // (op 2687) 
s_mov_b32 s19, scc // (op 2687) 
v_lshlrev_b32 v9, 2, v8 // (op 2687) 

s_cmp_eq_i32 s19, 0 // (op 2687) 
s_cselect_b32 s51, 0, -1 // (op 2687) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2687) 


s_and_b32 s19, s15, 31 // (op 2687) 
v_lshrrev_b32 v6, 2, v7 // (op 2687) 

v_and_b32 v7, 7, v6 // (op 2687) 

v_and_b32 v6, 63, v1 // (op 2801) 
v_and_b32 v9, 48, v1 // (op 2801) 
v_sub_u32 v10, v6, v9 // (op 2801) 

v_lshrrev_b32 v9, 5, v6 // (op 2801) 
v_lshl_add_u32 v11, v9, 5, v10 // (op 2801) 


v_lshrrev_b32 v9, 4, v6 // (op 2801) 

v_and_b32 v6, -2, v9 // (op 2801) 
v_sub_u32 v10, v9, v6 // (op 2801) 


v_lshlrev_b32 v6, 4, v10 // (op 2801) 

v_lshrrev_b32 v9, 6, v1 // (op 2801) 
v_and_b32 v10, 1, v9 // (op 2801) 

v_lshl_add_u32 v9, v10, 5, v6 // (op 2801) 


v_lshl_add_u32 v6, v11, 8, v9 // (op 2801) 


v_mul_lo_u32 v9, v6, 6 // (op 2801) 
v_lshrrev_b32 v10, 3, v9 // (op 2801) 

v_lshrrev_b32 v9, 4, v6 // (op 2801) 

v_lshlrev_b32 v6, 2, v9 // (op 2801) 

v_add_u32 v9, v10, v6 // (op 2801) 


v_lshrrev_b32 v6, 6, v1 // (op 2912) 
v_and_b32 v10, 1, v6 // (op 2912) 

v_and_b32 v6, 31, v1 // (op 2912) 
v_lshl_add_u32 v11, v10, 5, v6 // (op 2912) 


v_and_b32 v6, 63, v1 // (op 2912) 
v_lshrrev_b32 v10, 5, v6 // (op 2912) 

v_lshl_add_u32 v6, v10, 8, v11 // (op 2912) 


v_lshrrev_b32 v10, 7, v1 // (op 2928) 
v_and_b32 v11, 63, v1 // (op 2928) 
v_and_b32 v12, 63, v1 // (op 2998) 
v_and_b32 v13, 48, v1 // (op 2998) 
v_sub_u32 v14, v12, v13 // (op 2998) 

v_lshrrev_b32 v13, 5, v12 // (op 2998) 
v_lshl_add_u32 v15, v13, 5, v14 // (op 2998) 


v_lshrrev_b32 v13, 4, v12 // (op 2998) 

v_and_b32 v12, -2, v13 // (op 2998) 
v_sub_u32 v14, v13, v12 // (op 2998) 


v_lshlrev_b32 v12, 4, v14 // (op 2998) 

v_lshrrev_b32 v13, 6, v1 // (op 2998) 
v_and_b32 v14, 1, v13 // (op 2998) 

v_lshl_add_u32 v13, v14, 5, v12 // (op 2998) 


v_lshl_add_u32 v12, v15, 8, v13 // (op 2998) 


v_mul_lo_u32 v13, v12, 6 // (op 2998) 
v_lshrrev_b32 v14, 3, v13 // (op 2998) 

v_lshrrev_b32 v13, 4, v12 // (op 2998) 

v_lshlrev_b32 v12, 2, v13 // (op 2998) 

v_add_u32 v13, v14, v12 // (op 2998) 


v_lshrrev_b32 v12, 6, v1 // (op 3051) 
v_and_b32 v14, 1, v12 // (op 3051) 

v_and_b32 v12, 31, v1 // (op 3051) 
v_lshl_add_u32 v15, v14, 5, v12 // (op 3051) 


v_and_b32 v12, 63, v1 // (op 3051) 
v_lshrrev_b32 v14, 5, v12 // (op 3051) 

v_lshl_add_u32 v12, v14, 8, v15 // (op 3051) 


s_lshl_b32 s52, 1, s19 // (op 2687) 
v_and_b32 v14, 3, v1 // (op 2687) 
s_add_i32 s53, s52, s51 // (op 2687) 


s_and_b32 s51, s0, s53 // (op 2687) 


s_add_i32 s0, s1, s16 // (op 2928) 

s_ashr_i32 s1, s0, 31 // (op 2928) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 2687) 


s_add_i32 s52, s18, s51 // (op 2687) 


v_lshrrev_b32 v7, 6, v1 // (op 2687) 
s_ashr_i32 s18, s52, s19 // (op 2687) 


v_and_b32 v14, 1, v7 // (op 2687) 

s_lshl_b32 s19, s15, 24 // (op 2687) 
v_lshl_add_u32 v7, v14, 5, v15 // (op 2687) 


s_ashr_i32 s51, s19, 31 // (op 2687) 

s_xor_b32 s19, s18, s51 // (op 2687) 

s_sub_i32 s18, s19, s51 // (op 2687) 


v_lshrrev_b32 v14, 5, v11 // (op 2928) 
s_cmp_eq_i32 s12, 0 // (op 2928) 
s_mov_b32 s19, scc // (op 2928) 
v_lshlrev_b32 v15, 2, v14 // (op 2928) 

s_cmp_eq_i32 s19, 0 // (op 2928) 
s_cselect_b32 s51, 0, -1 // (op 2928) 

v_lshl_add_u32 v14, v10, 5, v15 // (op 2928) 


s_and_b32 s19, s15, 31 // (op 2928) 
v_lshrrev_b32 v10, 2, v11 // (op 2928) 

s_lshl_b32 s52, 1, s19 // (op 2928) 
v_and_b32 v11, 7, v10 // (op 2928) 

s_add_i32 s53, s52, s51 // (op 2928) 


v_and_b32 v10, 3, v1 // (op 2928) 
s_and_b32 s51, s1, s53 // (op 2928) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 2928) 


s_add_i32 s1, s0, s51 // (op 2928) 


v_lshrrev_b32 v10, 6, v1 // (op 2928) 
s_ashr_i32 s0, s1, s19 // (op 2928) 


v_and_b32 v11, 1, v10 // (op 2928) 

s_lshl_b32 s1, s15, 24 // (op 2928) 
v_lshl_add_u32 v10, v11, 5, v15 // (op 2928) 


s_ashr_i32 s19, s1, 31 // (op 2928) 

s_xor_b32 s1, s0, s19 // (op 2928) 

s_sub_i32 s0, s1, s19 // (op 2928) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s1, s18, s44 // (op 2687) 
s_add_i32 s19, s1, s18 // (op 2687) 

s_ashr_i32 s1, s19, 31 // (op 2687) 
s_cmp_eq_i32 s44, 0 // (op 2687) 
s_mov_b32 s51, scc // (op 2687) 
s_cmp_eq_i32 s51, 0 // (op 2687) 
s_cselect_b32 s52, 0, -1 // (op 2687) 

s_and_b32 s51, s45, 31 // (op 2687) 
s_lshl_b32 s53, 1, s51 // (op 2687) 
s_add_i32 s54, s53, s52 // (op 2687) 


s_and_b32 s52, s1, s54 // (op 2687) 


s_add_i32 s1, s19, s52 // (op 2687) 


s_ashr_i32 s19, s1, s51 // (op 2687) 


s_lshl_b32 s1, s45, 24 // (op 2687) 
s_ashr_i32 s51, s1, 31 // (op 2687) 

s_xor_b32 s1, s19, s51 // (op 2687) 

s_sub_i32 s19, s1, s51 // (op 2687) 


s_mul_i32 s1, s18, s13 // (op 2687) 
s_sub_u32 s51, s16, s1 // (op 2687) 

s_mul_i32 s1, s19, s13 // (op 2687) 
s_add_u32 s1, s1, s51 // (op 2687) 

s_mul_hi_i32 s51, s16, s11 // (op 2687) 
s_add_i32 s52, s51, s16 // (op 2687) 

s_ashr_i32 s51, s52, 31 // (op 2687) 
s_cmp_eq_i32 s11, 0 // (op 2687) 
s_mov_b32 s53, scc // (op 2687) 
s_cmp_eq_i32 s53, 0 // (op 2687) 
s_cselect_b32 s54, 0, -1 // (op 2687) 

s_and_b32 s53, s47, 31 // (op 2687) 
s_lshl_b32 s55, 1, s53 // (op 2687) 
s_add_i32 s56, s55, s54 // (op 2687) 


s_and_b32 s54, s51, s56 // (op 2687) 


s_add_i32 s51, s52, s54 // (op 2687) 


s_ashr_i32 s52, s51, s53 // (op 2687) 


s_lshl_b32 s51, s47, 24 // (op 2687) 
s_ashr_i32 s53, s51, 31 // (op 2687) 

s_xor_b32 s51, s52, s53 // (op 2687) 

s_sub_i32 s52, s51, s53 // (op 2687) 


s_mul_hi_i32 s51, s0, s44 // (op 2928) 
s_add_i32 s53, s51, s0 // (op 2928) 

s_ashr_i32 s51, s53, 31 // (op 2928) 
s_cmp_eq_i32 s44, 0 // (op 2928) 
s_mov_b32 s54, scc // (op 2928) 
s_cmp_eq_i32 s54, 0 // (op 2928) 
s_cselect_b32 s55, 0, -1 // (op 2928) 

s_and_b32 s54, s45, 31 // (op 2928) 
s_lshl_b32 s56, 1, s54 // (op 2928) 
s_add_i32 s57, s56, s55 // (op 2928) 


s_and_b32 s55, s51, s57 // (op 2928) 


s_add_i32 s51, s53, s55 // (op 2928) 


s_ashr_i32 s53, s51, s54 // (op 2928) 


s_lshl_b32 s51, s45, 24 // (op 2928) 
s_ashr_i32 s54, s51, 31 // (op 2928) 

s_xor_b32 s51, s53, s54 // (op 2928) 

s_sub_i32 s53, s51, s54 // (op 2928) 


s_mul_i32 s51, s0, s13 // (op 2928) 
s_sub_u32 s54, s16, s51 // (op 2928) 

s_mul_i32 s51, s53, s13 // (op 2928) 
s_add_u32 s51, s51, s54 // (op 2928) 

s_mul_hi_i32 s54, s16, s11 // (op 2928) 
s_add_i32 s55, s54, s16 // (op 2928) 

s_ashr_i32 s54, s55, 31 // (op 2928) 
s_cmp_eq_i32 s11, 0 // (op 2928) 
s_mov_b32 s56, scc // (op 2928) 
s_cmp_eq_i32 s56, 0 // (op 2928) 
s_cselect_b32 s57, 0, -1 // (op 2928) 

s_and_b32 s56, s47, 31 // (op 2928) 
s_lshl_b32 s58, 1, s56 // (op 2928) 
s_add_i32 s59, s58, s57 // (op 2928) 


s_and_b32 s57, s54, s59 // (op 2928) 


s_add_i32 s54, s55, s57 // (op 2928) 


s_ashr_i32 s55, s54, s56 // (op 2928) 


s_lshl_b32 s54, s47, 24 // (op 2928) 
s_ashr_i32 s56, s54, 31 // (op 2928) 

s_xor_b32 s54, s55, s56 // (op 2928) 

s_sub_i32 s55, s54, s56 // (op 2928) 


s_add_u32 s56, s8, 256 // (op 2687) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 2687) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2687) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2687) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 2687) 

s_mul_i32 s54, s13, s56 // (op 2687) 
s_mul_i32 s58, s10, s54 // (op 2687) 

s_mul_i32 s54, s52, s58 // (op 2687) 

s_sub_u32 s58, s16, s54 // (op 2687) 

s_mov_b32 s54, s58 // (op 2687) 
s_mul_hi_i32 s59, s54, s14 // (op 2687) 
s_add_i32 s60, s59, s54 // (op 2687) 


s_ashr_i32 s54, s60, 31 // (op 2687) 
s_cmp_eq_i32 s14, 0 // (op 2687) 
s_mov_b32 s59, scc // (op 2687) 
s_cmp_eq_i32 s59, 0 // (op 2687) 
s_cselect_b32 s61, 0, -1 // (op 2687) 

s_and_b32 s59, s46, 31 // (op 2687) 
s_lshl_b32 s62, 1, s59 // (op 2687) 
s_add_i32 s63, s62, s61 // (op 2687) 


s_and_b32 s61, s54, s63 // (op 2687) 


s_add_i32 s54, s60, s61 // (op 2687) 


s_ashr_i32 s60, s54, s59 // (op 2687) 


s_lshl_b32 s54, s46, 24 // (op 2687) 
s_ashr_i32 s59, s54, 31 // (op 2687) 

s_xor_b32 s54, s60, s59 // (op 2687) 

s_sub_i32 s60, s54, s59 // (op 2687) 


s_mul_i32 s54, s48, s13 // (op 2687) 
s_add_u32 s62, s6, 64 // (op 2687) Start of Int64 add, locking SCC
s_addc_u32 s63, s7, 0 // (op 2687) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2687) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2687) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 2687) 

s_sub_i32 s59, s62, s54 // (op 2687) 

s_mul_i32 s61, s60, s59 // (op 2687) 

s_sub_u32 s59, s58, s61 // (op 2687) 


s_add_u32 s58, s59, s54 // (op 2687) 


s_cmp_eq_i32 s52, 0 // (op 2687) 
s_mov_b32 s54, scc // (op 2687) 

s_cmp_eq_i32 s54, 0 // (op 2687) 
s_cselect_b32 s52, s58, s1 // (op 2687) 


s_mul_i32 s1, s19, s56 // (op 2687) 


v_lshl_add_u32 v11, s52, 6, v8 // (op 2687) 


s_sub_i32 s19, s18, s1 // (op 2687) 


s_cmp_eq_i32 s54, 0 // (op 2687) 
s_cselect_b32 s1, s60, s19 // (op 2687) 



v_lshl_add_u32 v8, s1, 8, v7 // (op 2687) 


v_mul_lo_u32 v7, v8, s2 // (op 2687) 

v_add_lshl_u32 v8, v11, v7, 1 // (op 2687) 


s_add_u32 s18, s8, 256 // (op 2928) Start of Int64 add, locking SCC
s_addc_u32 s19, s9, 0 // (op 2928) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 2928) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 2928) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 8 // (op 2928) 

s_mul_i32 s1, s13, s18 // (op 2928) 
s_mul_i32 s52, s10, s1 // (op 2928) 

s_mul_i32 s1, s55, s52 // (op 2928) 

s_sub_u32 s52, s16, s1 // (op 2928) 

s_mov_b32 s1, s52 // (op 2928) 
s_mul_hi_i32 s54, s1, s14 // (op 2928) 
s_add_i32 s56, s54, s1 // (op 2928) 


s_ashr_i32 s1, s56, 31 // (op 2928) 
s_cmp_eq_i32 s14, 0 // (op 2928) 
s_mov_b32 s54, scc // (op 2928) 
s_cmp_eq_i32 s54, 0 // (op 2928) 
s_cselect_b32 s57, 0, -1 // (op 2928) 

s_and_b32 s54, s46, 31 // (op 2928) 
s_lshl_b32 s58, 1, s54 // (op 2928) 
s_add_i32 s59, s58, s57 // (op 2928) 


s_and_b32 s57, s1, s59 // (op 2928) 


s_add_i32 s1, s56, s57 // (op 2928) 


s_ashr_i32 s56, s1, s54 // (op 2928) 


s_lshl_b32 s1, s46, 24 // (op 2928) 
s_ashr_i32 s54, s1, 31 // (op 2928) 

s_xor_b32 s1, s56, s54 // (op 2928) 

s_sub_i32 s56, s1, s54 // (op 2928) 


s_mul_i32 s1, s48, s13 // (op 2928) 
s_add_u32 s58, s6, 64 // (op 2928) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 2928) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2928) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2928) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2928) 

s_sub_i32 s54, s58, s1 // (op 2928) 

s_mul_i32 s57, s56, s54 // (op 2928) 

s_sub_u32 s54, s52, s57 // (op 2928) 


s_add_u32 s52, s54, s1 // (op 2928) 


s_cmp_eq_i32 s55, 0 // (op 2928) 
s_mov_b32 s1, scc // (op 2928) 

s_cmp_eq_i32 s1, 0 // (op 2928) 
s_cselect_b32 s54, s52, s51 // (op 2928) 


s_mul_i32 s55, s53, s18 // (op 2928) 


v_lshl_add_u32 v7, s54, 6, v14 // (op 2928) 


s_sub_i32 s18, s0, s55 // (op 2928) 


s_cmp_eq_i32 s1, 0 // (op 2928) 
s_cselect_b32 s0, s56, s18 // (op 2928) 



v_lshl_add_u32 v11, s0, 8, v10 // (op 2928) 


v_mul_lo_u32 v10, v11, s4 // (op 2928) 

v_add_lshl_u32 v11, v7, v10, 1 // (op 2928) 


s_mul_i32 s1, s2, s9 // (op 2689) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2689) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2689) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2689) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2689) most significant: sum
s_add_u32 s1, s1, s19 // (op 2689) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2689) 

s_mul_i32 s1, s4, s9 // (op 2930) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 2930) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2930) least significant
s_mul_hi_u32 s52, s4, s8 // (op 2930) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2930) most significant: sum
s_add_u32 s1, s1, s52 // (op 2930) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 2930) 

s_mov_b32 s56, s20 // (op 2689) reg expression
s_mov_b32 s57, s21 // (op 2689) reg expression
s_mov_b32 s58, s18 // (op 2689) reg expression
s_mov_b32 s59, 131072 // (op 2689) call()

s_mov_b32 s60, s22 // (op 2930) reg expression

s_mov_b32 s61, s23 // (op 2930) reg expression
s_mov_b32 s62, s52 // (op 2930) reg expression
s_mov_b32 s63, 131072 // (op 2930) call()


v_mov_b32 v7, s49 // (op 2) Move value
v_mov_b32 v10, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 65) call()
v_accvgpr_write a5, 0.00000 // (op 65) call()
v_accvgpr_write a6, 0.00000 // (op 65) call()
v_accvgpr_write a7, 0.00000 // (op 65) call()
v_accvgpr_write a8, 0.00000 // (op 65) call()
v_accvgpr_write a9, 0.00000 // (op 65) call()
v_accvgpr_write a10, 0.00000 // (op 65) call()
v_accvgpr_write a11, 0.00000 // (op 65) call()
v_accvgpr_write a12, 0.00000 // (op 65) call()
v_accvgpr_write a13, 0.00000 // (op 65) call()
v_accvgpr_write a14, 0.00000 // (op 65) call()
v_accvgpr_write a15, 0.00000 // (op 65) call()
v_accvgpr_write a16, 0.00000 // (op 402) call()
v_accvgpr_write a17, 0.00000 // (op 402) call()
v_accvgpr_write a18, 0.00000 // (op 402) call()
v_accvgpr_write a19, 0.00000 // (op 402) call()
v_accvgpr_write a20, 0.00000 // (op 402) call()
v_accvgpr_write a21, 0.00000 // (op 402) call()
v_accvgpr_write a22, 0.00000 // (op 402) call()
v_accvgpr_write a23, 0.00000 // (op 402) call()
v_accvgpr_write a24, 0.00000 // (op 402) call()
v_accvgpr_write a25, 0.00000 // (op 402) call()
v_accvgpr_write a26, 0.00000 // (op 402) call()
v_accvgpr_write a27, 0.00000 // (op 402) call()
v_accvgpr_write a28, 0.00000 // (op 402) call()
v_accvgpr_write a29, 0.00000 // (op 402) call()
v_accvgpr_write a30, 0.00000 // (op 402) call()
v_accvgpr_write a31, 0.00000 // (op 402) call()
v_accvgpr_write a32, 0.00000 // (op 636) call()
v_accvgpr_write a33, 0.00000 // (op 636) call()
v_accvgpr_write a34, 0.00000 // (op 636) call()
v_accvgpr_write a35, 0.00000 // (op 636) call()
v_accvgpr_write a36, 0.00000 // (op 636) call()
v_accvgpr_write a37, 0.00000 // (op 636) call()
v_accvgpr_write a38, 0.00000 // (op 636) call()
v_accvgpr_write a39, 0.00000 // (op 636) call()
v_accvgpr_write a40, 0.00000 // (op 636) call()
v_accvgpr_write a41, 0.00000 // (op 636) call()
v_accvgpr_write a42, 0.00000 // (op 636) call()
v_accvgpr_write a43, 0.00000 // (op 636) call()
v_accvgpr_write a44, 0.00000 // (op 636) call()
v_accvgpr_write a45, 0.00000 // (op 636) call()
v_accvgpr_write a46, 0.00000 // (op 636) call()
v_accvgpr_write a47, 0.00000 // (op 636) call()
v_accvgpr_write a48, 0.00000 // (op 870) call()
v_accvgpr_write a49, 0.00000 // (op 870) call()
v_accvgpr_write a50, 0.00000 // (op 870) call()
v_accvgpr_write a51, 0.00000 // (op 870) call()
v_accvgpr_write a52, 0.00000 // (op 870) call()
v_accvgpr_write a53, 0.00000 // (op 870) call()
v_accvgpr_write a54, 0.00000 // (op 870) call()
v_accvgpr_write a55, 0.00000 // (op 870) call()
v_accvgpr_write a56, 0.00000 // (op 870) call()
v_accvgpr_write a57, 0.00000 // (op 870) call()
v_accvgpr_write a58, 0.00000 // (op 870) call()
v_accvgpr_write a59, 0.00000 // (op 870) call()
v_accvgpr_write a60, 0.00000 // (op 870) call()
v_accvgpr_write a61, 0.00000 // (op 870) call()
v_accvgpr_write a62, 0.00000 // (op 870) call()
v_accvgpr_write a63, 0.00000 // (op 870) call()
s_mul_hi_i32 s0, s16, s12 // (op 2454) 
v_and_b32 v14, -4, v1 // (op 2454) 
s_add_i32 s1, s0, s16 // (op 2454) 

v_sub_u32 v15, v1, v14 // (op 2454) 

s_ashr_i32 s0, s1, 31 // (op 2454) 
v_lshlrev_b32 v14, 4, v15 // (op 2454) 

s_cmp_eq_i32 s12, 0 // (op 2454) 
s_mov_b32 s18, scc // (op 2454) 
v_lshrrev_b32 v15, 2, v1 // (op 2454) 
s_cmp_eq_i32 s18, 0 // (op 2454) 
s_cselect_b32 s19, 0, -1 // (op 2454) 

s_and_b32 s18, s15, 31 // (op 2454) 
s_lshl_b32 s20, 1, s18 // (op 2454) 
s_add_i32 s21, s20, s19 // (op 2454) 


s_and_b32 s19, s0, s21 // (op 2454) 


s_add_i32 s0, s1, s19 // (op 2454) 


s_ashr_i32 s1, s0, s18 // (op 2454) 


s_lshl_b32 s0, s15, 24 // (op 2454) 
s_ashr_i32 s18, s0, 31 // (op 2454) 

s_xor_b32 s0, s1, s18 // (op 2454) 

s_sub_i32 s1, s0, s18 // (op 2454) 


s_mul_hi_i32 s0, s1, s44 // (op 2454) 
s_add_i32 s18, s0, s1 // (op 2454) 

s_ashr_i32 s0, s18, 31 // (op 2454) 
s_cmp_eq_i32 s44, 0 // (op 2454) 
s_mov_b32 s19, scc // (op 2454) 
s_cmp_eq_i32 s19, 0 // (op 2454) 
s_cselect_b32 s20, 0, -1 // (op 2454) 

s_and_b32 s19, s45, 31 // (op 2454) 
s_lshl_b32 s21, 1, s19 // (op 2454) 
s_add_i32 s22, s21, s20 // (op 2454) 


s_and_b32 s23, s0, s22 // (op 2454) 


s_add_i32 s0, s18, s23 // (op 2454) 


s_ashr_i32 s18, s0, s19 // (op 2454) 


s_lshl_b32 s0, s45, 24 // (op 2454) 
s_ashr_i32 s19, s0, 31 // (op 2454) 

s_xor_b32 s0, s18, s19 // (op 2454) 

s_sub_i32 s18, s0, s19 // (op 2454) 


s_mul_i32 s0, s1, s13 // (op 2454) 

s_sub_u32 s1, s16, s0 // (op 2454) 

s_mul_i32 s0, s18, s13 // (op 2454) 
s_add_u32 s0, s0, s1 // (op 2454) 


s_mul_hi_i32 s1, s16, s11 // (op 2454) 
s_add_i32 s18, s1, s16 // (op 2454) 

s_ashr_i32 s1, s18, 31 // (op 2454) 
s_cmp_eq_i32 s11, 0 // (op 2454) 
s_mov_b32 s19, scc // (op 2454) 
s_cmp_eq_i32 s19, 0 // (op 2454) 
s_cselect_b32 s20, 0, -1 // (op 2454) 

s_and_b32 s19, s47, 31 // (op 2454) 
s_lshl_b32 s21, 1, s19 // (op 2454) 
s_add_i32 s22, s21, s20 // (op 2454) 


s_and_b32 s23, s1, s22 // (op 2454) 


s_add_i32 s1, s18, s23 // (op 2454) 


s_ashr_i32 s18, s1, s19 // (op 2454) 


s_lshl_b32 s1, s47, 24 // (op 2454) 
s_ashr_i32 s19, s1, 31 // (op 2454) 

s_xor_b32 s1, s18, s19 // (op 2454) 

s_sub_i32 s18, s1, s19 // (op 2454) 


s_add_u32 s22, s8, 256 // (op 2454) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 2454) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 2454) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 2454) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 8 // (op 2454) 

s_mul_i32 s1, s13, s22 // (op 2454) 

s_mul_i32 s19, s10, s1 // (op 2454) 

s_mul_i32 s1, s18, s19 // (op 2454) 

s_sub_u32 s19, s16, s1 // (op 2454) 

s_mov_b32 s1, s19 // (op 2454) 
s_mul_hi_i32 s20, s1, s14 // (op 2454) 
s_add_i32 s21, s20, s1 // (op 2454) 


s_ashr_i32 s1, s21, 31 // (op 2454) 
s_cmp_eq_i32 s14, 0 // (op 2454) 
s_mov_b32 s20, scc // (op 2454) 
s_cmp_eq_i32 s20, 0 // (op 2454) 
s_cselect_b32 s22, 0, -1 // (op 2454) 

s_and_b32 s20, s46, 31 // (op 2454) 
s_lshl_b32 s23, 1, s20 // (op 2454) 
s_add_i32 s49, s23, s22 // (op 2454) 


s_and_b32 s22, s1, s49 // (op 2454) 


s_add_i32 s1, s21, s22 // (op 2454) 


s_ashr_i32 s21, s1, s20 // (op 2454) 


s_lshl_b32 s1, s46, 24 // (op 2454) 
s_ashr_i32 s20, s1, 31 // (op 2454) 

s_xor_b32 s1, s21, s20 // (op 2454) 

s_sub_i32 s21, s1, s20 // (op 2454) 


s_mul_i32 s1, s48, s13 // (op 2454) 
s_add_u32 s22, s6, 64 // (op 2454) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2454) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 2454) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 2454) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 6 // (op 2454) 

s_sub_i32 s20, s22, s1 // (op 2454) 

s_mul_i32 s22, s21, s20 // (op 2454) 


s_sub_u32 s23, s19, s22 // (op 2454) 


s_add_u32 s19, s23, s1 // (op 2454) 


s_cmp_eq_i32 s18, 0 // (op 2454) 
s_mov_b32 s1, scc // (op 2454) 

s_cmp_eq_i32 s1, 0 // (op 2454) 
s_cselect_b32 s18, s19, s0 // (op 2454) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 2454) 


v_mov_b32 v19, 0 // (op 2454) convert
v_mov_b32 v18, v16 // (op 2454) convert

v_mul_lo_u32 v17, v15, s25 // (op 2454) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 2454) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 2454) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 2454) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 2454) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 2454) most significant half



s_mul_i32 s1, s24, s29 // (op 2456) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 2456) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 2456) least significant
s_mul_hi_u32 s19, s24, s28 // (op 2456) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2456) most significant: sum
s_add_u32 s1, s1, s19 // (op 2456) most significant: sum


s_mov_b32 s20, s26 // (op 2456) reg expression
s_mov_b32 s21, s27 // (op 2456) reg expression
s_mov_b32 s22, s0 // (op 2456) reg expression
s_mov_b32 s23, 131072 // (op 2456) call()

s_mul_hi_i32 s0, s16, s12 // (op 2473) 
v_and_b32 v16, -4, v1 // (op 2473) 
s_add_i32 s1, s0, s16 // (op 2473) 

v_sub_u32 v17, v1, v16 // (op 2473) 

s_ashr_i32 s0, s1, 31 // (op 2473) 
v_lshlrev_b32 v16, 4, v17 // (op 2473) 

s_cmp_eq_i32 s12, 0 // (op 2473) 
s_mov_b32 s18, scc // (op 2473) 
v_lshrrev_b32 v17, 2, v1 // (op 2473) 
s_cmp_eq_i32 s18, 0 // (op 2473) 
s_cselect_b32 s19, 0, -1 // (op 2473) 

s_and_b32 s18, s15, 31 // (op 2473) 
s_lshl_b32 s49, 1, s18 // (op 2473) 
s_add_i32 s50, s49, s19 // (op 2473) 


s_and_b32 s19, s0, s50 // (op 2473) 


s_add_i32 s0, s1, s19 // (op 2473) 


s_ashr_i32 s1, s0, s18 // (op 2473) 


s_lshl_b32 s0, s15, 24 // (op 2473) 
s_ashr_i32 s18, s0, 31 // (op 2473) 

s_xor_b32 s0, s1, s18 // (op 2473) 

s_sub_i32 s1, s0, s18 // (op 2473) 


s_mul_hi_i32 s0, s1, s44 // (op 2473) 
s_add_i32 s18, s0, s1 // (op 2473) 

s_ashr_i32 s0, s18, 31 // (op 2473) 
s_cmp_eq_i32 s44, 0 // (op 2473) 
s_mov_b32 s19, scc // (op 2473) 
s_cmp_eq_i32 s19, 0 // (op 2473) 
s_cselect_b32 s49, 0, -1 // (op 2473) 

s_and_b32 s19, s45, 31 // (op 2473) 
s_lshl_b32 s50, 1, s19 // (op 2473) 
s_add_i32 s51, s50, s49 // (op 2473) 


s_and_b32 s49, s0, s51 // (op 2473) 


s_add_i32 s0, s18, s49 // (op 2473) 


s_ashr_i32 s18, s0, s19 // (op 2473) 


s_lshl_b32 s0, s45, 24 // (op 2473) 
s_ashr_i32 s19, s0, 31 // (op 2473) 

s_xor_b32 s0, s18, s19 // (op 2473) 

s_sub_i32 s18, s0, s19 // (op 2473) 


s_mul_i32 s0, s1, s13 // (op 2473) 

s_sub_u32 s1, s16, s0 // (op 2473) 

s_mul_i32 s0, s18, s13 // (op 2473) 
s_add_u32 s0, s0, s1 // (op 2473) 


s_mul_hi_i32 s1, s16, s11 // (op 2473) 
s_add_i32 s18, s1, s16 // (op 2473) 

s_ashr_i32 s1, s18, 31 // (op 2473) 
s_cmp_eq_i32 s11, 0 // (op 2473) 
s_mov_b32 s19, scc // (op 2473) 
s_cmp_eq_i32 s19, 0 // (op 2473) 
s_cselect_b32 s49, 0, -1 // (op 2473) 

s_and_b32 s19, s47, 31 // (op 2473) 
s_lshl_b32 s50, 1, s19 // (op 2473) 
s_add_i32 s51, s50, s49 // (op 2473) 


s_and_b32 s49, s1, s51 // (op 2473) 


s_add_i32 s1, s18, s49 // (op 2473) 


s_ashr_i32 s18, s1, s19 // (op 2473) 


s_lshl_b32 s1, s47, 24 // (op 2473) 
s_ashr_i32 s19, s1, 31 // (op 2473) 

s_xor_b32 s1, s18, s19 // (op 2473) 

s_sub_i32 s18, s1, s19 // (op 2473) 


s_add_u32 s54, s8, 256 // (op 2473) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 2473) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 2473) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 2473) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 2473) 

s_mul_i32 s1, s13, s50 // (op 2473) 

s_mul_i32 s19, s10, s1 // (op 2473) 

s_mul_i32 s1, s18, s19 // (op 2473) 

s_sub_u32 s19, s16, s1 // (op 2473) 

s_mov_b32 s1, s19 // (op 2473) 
s_mul_hi_i32 s49, s1, s14 // (op 2473) 
s_add_i32 s50, s49, s1 // (op 2473) 


s_ashr_i32 s1, s50, 31 // (op 2473) 
s_cmp_eq_i32 s14, 0 // (op 2473) 
s_mov_b32 s49, scc // (op 2473) 
s_cmp_eq_i32 s49, 0 // (op 2473) 
s_cselect_b32 s51, 0, -1 // (op 2473) 

s_and_b32 s49, s46, 31 // (op 2473) 
s_lshl_b32 s52, 1, s49 // (op 2473) 
s_add_i32 s53, s52, s51 // (op 2473) 


s_and_b32 s51, s1, s53 // (op 2473) 


s_add_i32 s1, s50, s51 // (op 2473) 


s_ashr_i32 s50, s1, s49 // (op 2473) 


s_lshl_b32 s1, s46, 24 // (op 2473) 
s_ashr_i32 s49, s1, 31 // (op 2473) 

s_xor_b32 s1, s50, s49 // (op 2473) 

s_sub_i32 s50, s1, s49 // (op 2473) 


s_mul_i32 s1, s48, s13 // (op 2473) 
s_add_u32 s54, s6, 64 // (op 2473) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 2473) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 2473) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 2473) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 6 // (op 2473) 

s_sub_i32 s49, s54, s1 // (op 2473) 

s_mul_i32 s51, s50, s49 // (op 2473) 


s_sub_u32 s49, s19, s51 // (op 2473) 


s_add_u32 s19, s49, s1 // (op 2473) 


s_cmp_eq_i32 s18, 0 // (op 2473) 
s_mov_b32 s1, scc // (op 2473) 

s_cmp_eq_i32 s1, 0 // (op 2473) 
s_cselect_b32 s18, s19, s0 // (op 2473) 



v_lshl_add_u32 v18, s18, 6, v16 // (op 2473) 


v_mov_b32 v21, 0 // (op 2473) convert
v_mov_b32 v20, v18 // (op 2473) convert

v_mul_lo_u32 v19, v17, s25 // (op 2473) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 2473) most significant: high of low * low
v_mul_lo_u32 v18, v17, s24 // (op 2473) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 2473) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v18 // (op 2473) least significant half
v_addc_co_u32 v17, s[0:1], v21, v19, s[0:1] // (op 2473) most significant half



s_mul_i32 s1, s24, s29 // (op 2475) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 2475) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 2475) least significant
s_mul_hi_u32 s19, s24, s28 // (op 2475) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2475) most significant: sum
s_add_u32 s1, s1, s19 // (op 2475) most significant: sum


s_mov_b32 s20, s26 // (op 2475) reg expression
s_mov_b32 s21, s27 // (op 2475) reg expression
s_mov_b32 s22, s0 // (op 2475) reg expression
s_mov_b32 s23, 131072 // (op 2475) call()

s_mul_hi_i32 s0, s16, s12 // (op 2518) 
v_and_b32 v18, -4, v1 // (op 2518) 
s_add_i32 s1, s0, s16 // (op 2518) 

v_sub_u32 v19, v1, v18 // (op 2518) 

s_ashr_i32 s0, s1, 31 // (op 2518) 
v_lshlrev_b32 v18, 6, v19 // (op 2518) 

s_cmp_eq_i32 s12, 0 // (op 2518) 
s_mov_b32 s18, scc // (op 2518) 
v_lshrrev_b32 v19, 2, v1 // (op 2518) 
s_cmp_eq_i32 s18, 0 // (op 2518) 
s_cselect_b32 s19, 0, -1 // (op 2518) 

v_lshlrev_b32 v20, 1, v19 // (op 2518) 

s_and_b32 s18, s15, 31 // (op 2518) 
s_lshl_b32 s49, 1, s18 // (op 2518) 
s_add_i32 s50, s49, s19 // (op 2518) 


s_and_b32 s19, s0, s50 // (op 2518) 


s_add_i32 s0, s1, s19 // (op 2518) 


s_ashr_i32 s1, s0, s18 // (op 2518) 


s_lshl_b32 s0, s15, 24 // (op 2518) 
s_ashr_i32 s18, s0, 31 // (op 2518) 

s_xor_b32 s0, s1, s18 // (op 2518) 

s_sub_i32 s1, s0, s18 // (op 2518) 


s_mul_hi_i32 s0, s1, s44 // (op 2518) 
s_add_i32 s18, s0, s1 // (op 2518) 

s_ashr_i32 s0, s18, 31 // (op 2518) 
s_cmp_eq_i32 s44, 0 // (op 2518) 
s_mov_b32 s19, scc // (op 2518) 
s_cmp_eq_i32 s19, 0 // (op 2518) 
s_cselect_b32 s49, 0, -1 // (op 2518) 

s_and_b32 s19, s45, 31 // (op 2518) 
s_lshl_b32 s50, 1, s19 // (op 2518) 
s_add_i32 s51, s50, s49 // (op 2518) 


s_and_b32 s49, s0, s51 // (op 2518) 


s_add_i32 s0, s18, s49 // (op 2518) 


s_ashr_i32 s18, s0, s19 // (op 2518) 


s_lshl_b32 s0, s45, 24 // (op 2518) 
s_ashr_i32 s19, s0, 31 // (op 2518) 

s_xor_b32 s0, s18, s19 // (op 2518) 

s_sub_i32 s18, s0, s19 // (op 2518) 


s_add_u32 s54, s8, 256 // (op 2518) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 2518) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 2518) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 2518) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 2518) 

s_mul_i32 s0, s18, s50 // (op 2518) 

s_sub_i32 s49, s1, s0 // (op 2518) 


s_mul_hi_i32 s0, s16, s11 // (op 2518) 
s_add_i32 s1, s0, s16 // (op 2518) 

s_ashr_i32 s0, s1, 31 // (op 2518) 
s_cmp_eq_i32 s11, 0 // (op 2518) 
s_mov_b32 s18, scc // (op 2518) 
s_cmp_eq_i32 s18, 0 // (op 2518) 
s_cselect_b32 s19, 0, -1 // (op 2518) 

s_and_b32 s18, s47, 31 // (op 2518) 
s_lshl_b32 s52, 1, s18 // (op 2518) 
s_add_i32 s53, s52, s19 // (op 2518) 


s_and_b32 s19, s0, s53 // (op 2518) 


s_add_i32 s0, s1, s19 // (op 2518) 


s_ashr_i32 s1, s0, s18 // (op 2518) 


s_lshl_b32 s0, s47, 24 // (op 2518) 
s_ashr_i32 s18, s0, 31 // (op 2518) 

s_xor_b32 s0, s1, s18 // (op 2518) 

s_sub_i32 s1, s0, s18 // (op 2518) 


s_mul_i32 s0, s13, s50 // (op 2518) 

s_mul_i32 s18, s10, s0 // (op 2518) 

s_mul_i32 s0, s1, s18 // (op 2518) 

s_sub_u32 s18, s16, s0 // (op 2518) 

s_mul_hi_i32 s0, s18, s14 // (op 2518) 
s_add_i32 s19, s0, s18 // (op 2518) 


s_ashr_i32 s0, s19, 31 // (op 2518) 
s_cmp_eq_i32 s14, 0 // (op 2518) 
s_mov_b32 s18, scc // (op 2518) 
s_cmp_eq_i32 s18, 0 // (op 2518) 
s_cselect_b32 s50, 0, -1 // (op 2518) 

s_and_b32 s18, s46, 31 // (op 2518) 
s_lshl_b32 s51, 1, s18 // (op 2518) 
s_add_i32 s52, s51, s50 // (op 2518) 


s_and_b32 s50, s0, s52 // (op 2518) 


s_add_i32 s0, s19, s50 // (op 2518) 


s_ashr_i32 s19, s0, s18 // (op 2518) 


s_lshl_b32 s0, s46, 24 // (op 2518) 
s_ashr_i32 s18, s0, 31 // (op 2518) 

s_xor_b32 s0, s19, s18 // (op 2518) 

s_sub_i32 s19, s0, s18 // (op 2518) 


s_cmp_eq_i32 s1, 0 // (op 2518) 
s_mov_b32 s0, scc // (op 2518) 

s_cmp_eq_i32 s0, 0 // (op 2518) 
s_cselect_b32 s1, s19, s49 // (op 2518) 



v_lshl_add_u32 v19, s1, 8, v18 // (op 2518) 


v_mov_b32 v23, 0 // (op 2518) convert
v_mov_b32 v22, v19 // (op 2518) convert

v_mul_lo_u32 v19, v20, s31 // (op 2518) most significant: low of low * high
v_mul_hi_u32 v21, v20, s30 // (op 2518) most significant: high of low * low
v_mul_lo_u32 v18, v20, s30 // (op 2518) least significant: low of low * low
v_add_u32 v19, v19, v21 // (op 2518) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v22 // (op 2518) least significant half
v_addc_co_u32 v19, s[0:1], v19, v23, s[0:1] // (op 2518) most significant half



v_mul_lo_u32 v21, v19, 6 // (op 2518) most significant: low of high * low
v_mul_hi_u32 v22, v18, 6 // (op 2518) most significant: high of low * low
v_mul_lo_u32 v20, v18, 6 // (op 2518) least significant: low of low * low
v_add_u32 v21, v21, v22 // (op 2518) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 2518) 

v_mov_b32 v20, v18 // (op 2518) convert

s_mul_i32 s1, s29, 6 // (op 2520) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 2520) least significant
s_mul_hi_u32 s18, s28, 6 // (op 2520) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2520) most significant: sum

s_mul_i32 s19, s30, s1 // (op 2520) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 2520) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 2520) least significant
s_mul_hi_u32 s50, s30, s0 // (op 2520) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 2520) most significant: sum
s_add_u32 s19, s19, s50 // (op 2520) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2520) 

s_mov_b32 s52, s32 // (op 2520) reg expression
s_mov_b32 s53, s33 // (op 2520) reg expression
s_mov_b32 s54, s0 // (op 2520) reg expression
s_mov_b32 s55, 131072 // (op 2520) call()

s_mul_hi_i32 s0, s16, s12 // (op 2537) 
v_and_b32 v18, -4, v1 // (op 2537) 
s_add_i32 s1, s0, s16 // (op 2537) 

v_sub_u32 v19, v1, v18 // (op 2537) 

s_ashr_i32 s0, s1, 31 // (op 2537) 
v_lshlrev_b32 v18, 6, v19 // (op 2537) 

s_cmp_eq_i32 s12, 0 // (op 2537) 
s_mov_b32 s18, scc // (op 2537) 
v_lshrrev_b32 v19, 2, v1 // (op 2537) 
s_cmp_eq_i32 s18, 0 // (op 2537) 
s_cselect_b32 s19, 0, -1 // (op 2537) 

v_lshlrev_b32 v21, 1, v19 // (op 2537) 

s_and_b32 s18, s15, 31 // (op 2537) 
s_lshl_b32 s49, 1, s18 // (op 2537) 
s_add_i32 s50, s49, s19 // (op 2537) 


s_and_b32 s19, s0, s50 // (op 2537) 


s_add_i32 s0, s1, s19 // (op 2537) 


s_ashr_i32 s1, s0, s18 // (op 2537) 


s_lshl_b32 s0, s15, 24 // (op 2537) 
s_ashr_i32 s18, s0, 31 // (op 2537) 

s_xor_b32 s0, s1, s18 // (op 2537) 

s_sub_i32 s1, s0, s18 // (op 2537) 


s_mul_hi_i32 s0, s1, s44 // (op 2537) 
s_add_i32 s18, s0, s1 // (op 2537) 

s_ashr_i32 s0, s18, 31 // (op 2537) 
s_cmp_eq_i32 s44, 0 // (op 2537) 
s_mov_b32 s19, scc // (op 2537) 
s_cmp_eq_i32 s19, 0 // (op 2537) 
s_cselect_b32 s49, 0, -1 // (op 2537) 

s_and_b32 s19, s45, 31 // (op 2537) 
s_lshl_b32 s50, 1, s19 // (op 2537) 
s_add_i32 s51, s50, s49 // (op 2537) 


s_and_b32 s49, s0, s51 // (op 2537) 


s_add_i32 s0, s18, s49 // (op 2537) 


s_ashr_i32 s18, s0, s19 // (op 2537) 


s_lshl_b32 s0, s45, 24 // (op 2537) 
s_ashr_i32 s19, s0, 31 // (op 2537) 

s_xor_b32 s0, s18, s19 // (op 2537) 

s_sub_i32 s18, s0, s19 // (op 2537) 


s_add_u32 s50, s8, 256 // (op 2537) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2537) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 2537) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 2537) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 2537) 

s_mul_i32 s0, s18, s50 // (op 2537) 

s_sub_i32 s49, s1, s0 // (op 2537) 


s_mul_hi_i32 s0, s16, s11 // (op 2537) 
s_add_i32 s1, s0, s16 // (op 2537) 

s_ashr_i32 s0, s1, 31 // (op 2537) 
s_cmp_eq_i32 s11, 0 // (op 2537) 
s_mov_b32 s18, scc // (op 2537) 
s_cmp_eq_i32 s18, 0 // (op 2537) 
s_cselect_b32 s19, 0, -1 // (op 2537) 

s_and_b32 s18, s47, 31 // (op 2537) 
s_lshl_b32 s64, 1, s18 // (op 2537) 
s_add_i32 s65, s64, s19 // (op 2537) 


s_and_b32 s19, s0, s65 // (op 2537) 


s_add_i32 s0, s1, s19 // (op 2537) 


s_ashr_i32 s1, s0, s18 // (op 2537) 


s_lshl_b32 s0, s47, 24 // (op 2537) 
s_ashr_i32 s18, s0, 31 // (op 2537) 

s_xor_b32 s0, s1, s18 // (op 2537) 

s_sub_i32 s1, s0, s18 // (op 2537) 


s_mul_i32 s0, s13, s50 // (op 2537) 

s_mul_i32 s18, s10, s0 // (op 2537) 

s_mul_i32 s0, s1, s18 // (op 2537) 

s_sub_u32 s18, s16, s0 // (op 2537) 

s_mul_hi_i32 s0, s18, s14 // (op 2537) 
s_add_i32 s19, s0, s18 // (op 2537) 


s_ashr_i32 s0, s19, 31 // (op 2537) 
s_cmp_eq_i32 s14, 0 // (op 2537) 
s_mov_b32 s18, scc // (op 2537) 
s_cmp_eq_i32 s18, 0 // (op 2537) 
s_cselect_b32 s50, 0, -1 // (op 2537) 

s_and_b32 s18, s46, 31 // (op 2537) 
s_lshl_b32 s51, 1, s18 // (op 2537) 
s_add_i32 s64, s51, s50 // (op 2537) 


s_and_b32 s50, s0, s64 // (op 2537) 


s_add_i32 s0, s19, s50 // (op 2537) 


s_ashr_i32 s19, s0, s18 // (op 2537) 


s_lshl_b32 s0, s46, 24 // (op 2537) 
s_ashr_i32 s18, s0, 31 // (op 2537) 

s_xor_b32 s0, s19, s18 // (op 2537) 

s_sub_i32 s19, s0, s18 // (op 2537) 


s_cmp_eq_i32 s1, 0 // (op 2537) 
s_mov_b32 s0, scc // (op 2537) 

s_cmp_eq_i32 s0, 0 // (op 2537) 
s_cselect_b32 s1, s19, s49 // (op 2537) 



v_lshl_add_u32 v19, s1, 8, v18 // (op 2537) 


v_mov_b32 v23, 0 // (op 2537) convert
v_mov_b32 v22, v19 // (op 2537) convert

v_mul_lo_u32 v19, v21, s31 // (op 2537) most significant: low of low * high
v_mul_hi_u32 v24, v21, s30 // (op 2537) most significant: high of low * low
v_mul_lo_u32 v18, v21, s30 // (op 2537) least significant: low of low * low
v_add_u32 v19, v19, v24 // (op 2537) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v22 // (op 2537) least significant half
v_addc_co_u32 v19, s[0:1], v19, v23, s[0:1] // (op 2537) most significant half



v_mul_lo_u32 v23, v19, 6 // (op 2537) most significant: low of high * low
v_mul_hi_u32 v21, v18, 6 // (op 2537) most significant: high of low * low
v_mul_lo_u32 v22, v18, 6 // (op 2537) least significant: low of low * low
v_add_u32 v23, v23, v21 // (op 2537) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[22:23] // (op 2537) 

v_mov_b32 v21, v18 // (op 2537) convert

s_mul_i32 s1, s29, 6 // (op 2539) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 2539) least significant
s_mul_hi_u32 s18, s28, 6 // (op 2539) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2539) most significant: sum

s_mul_i32 s19, s30, s1 // (op 2539) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 2539) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 2539) least significant
s_mul_hi_u32 s50, s30, s0 // (op 2539) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 2539) most significant: sum
s_add_u32 s19, s19, s50 // (op 2539) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2539) 

s_mov_b32 s52, s32 // (op 2539) reg expression
s_mov_b32 s53, s33 // (op 2539) reg expression
s_mov_b32 s54, s0 // (op 2539) reg expression
s_mov_b32 s55, 131072 // (op 2539) call()

s_mul_hi_i32 s0, s16, s12 // (op 2582) 
v_and_b32 v18, -64, v1 // (op 2582) 
s_add_i32 s1, s0, s16 // (op 2582) 

v_sub_u32 v19, v1, v18 // (op 2582) 

s_ashr_i32 s0, s1, 31 // (op 2582) 
v_lshrrev_b32 v18, 6, v1 // (op 2582) 
s_cmp_eq_i32 s12, 0 // (op 2582) 
s_mov_b32 s18, scc // (op 2582) 
s_cmp_eq_i32 s18, 0 // (op 2582) 
s_cselect_b32 s19, 0, -1 // (op 2582) 

s_and_b32 s18, s15, 31 // (op 2582) 
s_lshl_b32 s49, 1, s18 // (op 2582) 
s_add_i32 s50, s49, s19 // (op 2582) 


s_and_b32 s19, s0, s50 // (op 2582) 


s_add_i32 s0, s1, s19 // (op 2582) 


s_ashr_i32 s1, s0, s18 // (op 2582) 


s_lshl_b32 s0, s15, 24 // (op 2582) 
s_ashr_i32 s18, s0, 31 // (op 2582) 

s_xor_b32 s0, s1, s18 // (op 2582) 

s_sub_i32 s1, s0, s18 // (op 2582) 


s_mul_hi_i32 s0, s1, s44 // (op 2582) 
s_add_i32 s18, s0, s1 // (op 2582) 

s_ashr_i32 s0, s18, 31 // (op 2582) 
s_cmp_eq_i32 s44, 0 // (op 2582) 
s_mov_b32 s19, scc // (op 2582) 
s_cmp_eq_i32 s19, 0 // (op 2582) 
s_cselect_b32 s49, 0, -1 // (op 2582) 

s_and_b32 s19, s45, 31 // (op 2582) 
s_lshl_b32 s50, 1, s19 // (op 2582) 
s_add_i32 s51, s50, s49 // (op 2582) 


s_and_b32 s49, s0, s51 // (op 2582) 


s_add_i32 s0, s18, s49 // (op 2582) 


s_ashr_i32 s18, s0, s19 // (op 2582) 


s_lshl_b32 s0, s45, 24 // (op 2582) 
s_ashr_i32 s19, s0, 31 // (op 2582) 

s_xor_b32 s0, s18, s19 // (op 2582) 

s_sub_i32 s18, s0, s19 // (op 2582) 


s_mul_i32 s0, s1, s13 // (op 2582) 

s_sub_u32 s1, s16, s0 // (op 2582) 

s_mul_i32 s0, s18, s13 // (op 2582) 
s_add_u32 s0, s0, s1 // (op 2582) 


s_mul_hi_i32 s1, s16, s11 // (op 2582) 
s_add_i32 s18, s1, s16 // (op 2582) 

s_ashr_i32 s1, s18, 31 // (op 2582) 
s_cmp_eq_i32 s11, 0 // (op 2582) 
s_mov_b32 s19, scc // (op 2582) 
s_cmp_eq_i32 s19, 0 // (op 2582) 
s_cselect_b32 s49, 0, -1 // (op 2582) 

s_and_b32 s19, s47, 31 // (op 2582) 
s_lshl_b32 s50, 1, s19 // (op 2582) 
s_add_i32 s51, s50, s49 // (op 2582) 


s_and_b32 s49, s1, s51 // (op 2582) 


s_add_i32 s1, s18, s49 // (op 2582) 


s_ashr_i32 s18, s1, s19 // (op 2582) 


s_lshl_b32 s1, s47, 24 // (op 2582) 
s_ashr_i32 s19, s1, 31 // (op 2582) 

s_xor_b32 s1, s18, s19 // (op 2582) 

s_sub_i32 s18, s1, s19 // (op 2582) 


s_add_u32 s50, s8, 256 // (op 2582) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2582) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 2582) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 2582) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 2582) 

s_mul_i32 s1, s13, s50 // (op 2582) 

s_mul_i32 s19, s10, s1 // (op 2582) 

s_mul_i32 s1, s18, s19 // (op 2582) 

s_sub_u32 s19, s16, s1 // (op 2582) 

s_mov_b32 s1, s19 // (op 2582) 
s_mul_hi_i32 s49, s1, s14 // (op 2582) 
s_add_i32 s50, s49, s1 // (op 2582) 


s_ashr_i32 s1, s50, 31 // (op 2582) 
s_cmp_eq_i32 s14, 0 // (op 2582) 
s_mov_b32 s49, scc // (op 2582) 
s_cmp_eq_i32 s49, 0 // (op 2582) 
s_cselect_b32 s51, 0, -1 // (op 2582) 

s_and_b32 s49, s46, 31 // (op 2582) 
s_lshl_b32 s64, 1, s49 // (op 2582) 
s_add_i32 s65, s64, s51 // (op 2582) 


s_and_b32 s51, s1, s65 // (op 2582) 


s_add_i32 s1, s50, s51 // (op 2582) 


s_ashr_i32 s50, s1, s49 // (op 2582) 


s_lshl_b32 s1, s46, 24 // (op 2582) 
s_ashr_i32 s49, s1, 31 // (op 2582) 

s_xor_b32 s1, s50, s49 // (op 2582) 

s_sub_i32 s50, s1, s49 // (op 2582) 


s_mul_i32 s1, s48, s13 // (op 2582) 
s_add_u32 s64, s6, 64 // (op 2582) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 2582) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 2582) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 2582) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 6 // (op 2582) 

s_sub_i32 s49, s64, s1 // (op 2582) 

s_mul_i32 s51, s50, s49 // (op 2582) 


s_sub_u32 s49, s19, s51 // (op 2582) 


s_add_u32 s19, s49, s1 // (op 2582) 


s_cmp_eq_i32 s18, 0 // (op 2582) 
s_mov_b32 s1, scc // (op 2582) 

s_cmp_eq_i32 s1, 0 // (op 2582) 
s_cselect_b32 s18, s19, s0 // (op 2582) 



v_lshl_add_u32 v22, s18, 6, v19 // (op 2582) 


v_mul_lo_u32 v19, v18, s34 // (op 2582) 

v_add_u32 v18, v22, v19 // (op 2582) 


s_mul_i32 s1, s34, s39 // (op 2584) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 2584) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2584) least significant
s_mul_hi_u32 s19, s34, s38 // (op 2584) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2584) most significant: sum
s_add_u32 s1, s1, s19 // (op 2584) most significant: sum


s_mov_b32 s64, s36 // (op 2584) reg expression
s_mov_b32 s65, s37 // (op 2584) reg expression
s_mov_b32 s66, s0 // (op 2584) reg expression
s_mov_b32 s67, 131072 // (op 2584) call()

s_mul_hi_i32 s0, s16, s12 // (op 2601) 
v_and_b32 v19, -64, v1 // (op 2601) 
s_add_i32 s1, s0, s16 // (op 2601) 

v_sub_u32 v22, v1, v19 // (op 2601) 

s_ashr_i32 s0, s1, 31 // (op 2601) 
v_lshrrev_b32 v19, 6, v1 // (op 2601) 
s_cmp_eq_i32 s12, 0 // (op 2601) 
s_mov_b32 s18, scc // (op 2601) 
s_cmp_eq_i32 s18, 0 // (op 2601) 
s_cselect_b32 s19, 0, -1 // (op 2601) 

s_and_b32 s18, s15, 31 // (op 2601) 
s_lshl_b32 s49, 1, s18 // (op 2601) 
s_add_i32 s50, s49, s19 // (op 2601) 


s_and_b32 s19, s0, s50 // (op 2601) 


s_add_i32 s0, s1, s19 // (op 2601) 


s_ashr_i32 s1, s0, s18 // (op 2601) 


s_lshl_b32 s0, s15, 24 // (op 2601) 
s_ashr_i32 s18, s0, 31 // (op 2601) 

s_xor_b32 s0, s1, s18 // (op 2601) 

s_sub_i32 s1, s0, s18 // (op 2601) 


s_mul_hi_i32 s0, s1, s44 // (op 2601) 
s_add_i32 s18, s0, s1 // (op 2601) 

s_ashr_i32 s0, s18, 31 // (op 2601) 
s_cmp_eq_i32 s44, 0 // (op 2601) 
s_mov_b32 s19, scc // (op 2601) 
s_cmp_eq_i32 s19, 0 // (op 2601) 
s_cselect_b32 s49, 0, -1 // (op 2601) 

s_and_b32 s19, s45, 31 // (op 2601) 
s_lshl_b32 s50, 1, s19 // (op 2601) 
s_add_i32 s51, s50, s49 // (op 2601) 


s_and_b32 s49, s0, s51 // (op 2601) 


s_add_i32 s0, s18, s49 // (op 2601) 


s_ashr_i32 s18, s0, s19 // (op 2601) 


s_lshl_b32 s0, s45, 24 // (op 2601) 
s_ashr_i32 s19, s0, 31 // (op 2601) 

s_xor_b32 s0, s18, s19 // (op 2601) 

s_sub_i32 s18, s0, s19 // (op 2601) 


s_mul_i32 s0, s1, s13 // (op 2601) 

s_sub_u32 s1, s16, s0 // (op 2601) 

s_mul_i32 s0, s18, s13 // (op 2601) 
s_add_u32 s0, s0, s1 // (op 2601) 


s_mul_hi_i32 s1, s16, s11 // (op 2601) 
s_add_i32 s18, s1, s16 // (op 2601) 

s_ashr_i32 s1, s18, 31 // (op 2601) 
s_cmp_eq_i32 s11, 0 // (op 2601) 
s_mov_b32 s19, scc // (op 2601) 
s_cmp_eq_i32 s19, 0 // (op 2601) 
s_cselect_b32 s49, 0, -1 // (op 2601) 

s_and_b32 s19, s47, 31 // (op 2601) 
s_lshl_b32 s50, 1, s19 // (op 2601) 
s_add_i32 s51, s50, s49 // (op 2601) 


s_and_b32 s49, s1, s51 // (op 2601) 


s_add_i32 s1, s18, s49 // (op 2601) 


s_ashr_i32 s18, s1, s19 // (op 2601) 


s_lshl_b32 s1, s47, 24 // (op 2601) 
s_ashr_i32 s19, s1, 31 // (op 2601) 

s_xor_b32 s1, s18, s19 // (op 2601) 

s_sub_i32 s18, s1, s19 // (op 2601) 


s_add_u32 s50, s8, 256 // (op 2601) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2601) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 2601) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 2601) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 2601) 

s_mul_i32 s1, s13, s50 // (op 2601) 

s_mul_i32 s19, s10, s1 // (op 2601) 

s_mul_i32 s1, s18, s19 // (op 2601) 

s_sub_u32 s19, s16, s1 // (op 2601) 

s_mov_b32 s1, s19 // (op 2601) 
s_mul_hi_i32 s49, s1, s14 // (op 2601) 
s_add_i32 s50, s49, s1 // (op 2601) 


s_ashr_i32 s1, s50, 31 // (op 2601) 
s_cmp_eq_i32 s14, 0 // (op 2601) 
s_mov_b32 s49, scc // (op 2601) 
s_cmp_eq_i32 s49, 0 // (op 2601) 
s_cselect_b32 s51, 0, -1 // (op 2601) 

s_and_b32 s49, s46, 31 // (op 2601) 
s_lshl_b32 s68, 1, s49 // (op 2601) 
s_add_i32 s69, s68, s51 // (op 2601) 


s_and_b32 s51, s1, s69 // (op 2601) 


s_add_i32 s1, s50, s51 // (op 2601) 


s_ashr_i32 s50, s1, s49 // (op 2601) 


s_lshl_b32 s1, s46, 24 // (op 2601) 
s_ashr_i32 s49, s1, 31 // (op 2601) 

s_xor_b32 s1, s50, s49 // (op 2601) 

s_sub_i32 s50, s1, s49 // (op 2601) 


s_mul_i32 s1, s48, s13 // (op 2601) 
s_add_u32 s68, s6, 64 // (op 2601) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 2601) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 2601) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 2601) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 6 // (op 2601) 

s_sub_i32 s49, s68, s1 // (op 2601) 

s_mul_i32 s51, s50, s49 // (op 2601) 


s_sub_u32 s49, s19, s51 // (op 2601) 


s_add_u32 s19, s49, s1 // (op 2601) 


s_cmp_eq_i32 s18, 0 // (op 2601) 
s_mov_b32 s1, scc // (op 2601) 

s_cmp_eq_i32 s1, 0 // (op 2601) 
s_cselect_b32 s18, s19, s0 // (op 2601) 



v_lshl_add_u32 v23, s18, 6, v22 // (op 2601) 


v_mul_lo_u32 v22, v19, s34 // (op 2601) 

v_add_u32 v19, v23, v22 // (op 2601) 


s_mul_i32 s1, s34, s39 // (op 2603) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 2603) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2603) least significant
s_mul_hi_u32 s19, s34, s38 // (op 2603) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2603) most significant: sum
s_add_u32 s1, s1, s19 // (op 2603) most significant: sum


s_mov_b32 s64, s36 // (op 2603) reg expression
s_mov_b32 s65, s37 // (op 2603) reg expression
s_mov_b32 s66, s0 // (op 2603) reg expression
s_mov_b32 s67, 131072 // (op 2603) call()

s_mul_hi_i32 s0, s16, s12 // (op 2646) 
v_and_b32 v22, -64, v1 // (op 2646) 
s_add_i32 s1, s0, s16 // (op 2646) 

v_sub_u32 v23, v1, v22 // (op 2646) 

s_ashr_i32 s0, s1, 31 // (op 2646) 
v_lshlrev_b32 v22, 2, v23 // (op 2646) 

s_cmp_eq_i32 s12, 0 // (op 2646) 
s_mov_b32 s18, scc // (op 2646) 
v_lshrrev_b32 v23, 6, v1 // (op 2646) 
s_cmp_eq_i32 s18, 0 // (op 2646) 
s_cselect_b32 s19, 0, -1 // (op 2646) 

s_and_b32 s18, s15, 31 // (op 2646) 
s_lshl_b32 s49, 1, s18 // (op 2646) 
s_add_i32 s50, s49, s19 // (op 2646) 


s_and_b32 s19, s0, s50 // (op 2646) 


s_add_i32 s0, s1, s19 // (op 2646) 


s_ashr_i32 s1, s0, s18 // (op 2646) 


s_lshl_b32 s0, s15, 24 // (op 2646) 
s_ashr_i32 s18, s0, 31 // (op 2646) 

s_xor_b32 s0, s1, s18 // (op 2646) 

s_sub_i32 s1, s0, s18 // (op 2646) 


s_mul_hi_i32 s0, s1, s44 // (op 2646) 
s_add_i32 s18, s0, s1 // (op 2646) 

s_ashr_i32 s0, s18, 31 // (op 2646) 
s_cmp_eq_i32 s44, 0 // (op 2646) 
s_mov_b32 s19, scc // (op 2646) 
s_cmp_eq_i32 s19, 0 // (op 2646) 
s_cselect_b32 s49, 0, -1 // (op 2646) 

s_and_b32 s19, s45, 31 // (op 2646) 
s_lshl_b32 s50, 1, s19 // (op 2646) 
s_add_i32 s51, s50, s49 // (op 2646) 


s_and_b32 s49, s0, s51 // (op 2646) 


s_add_i32 s0, s18, s49 // (op 2646) 


s_ashr_i32 s18, s0, s19 // (op 2646) 


s_lshl_b32 s0, s45, 24 // (op 2646) 
s_ashr_i32 s19, s0, 31 // (op 2646) 

s_xor_b32 s0, s18, s19 // (op 2646) 

s_sub_i32 s18, s0, s19 // (op 2646) 


s_add_u32 s50, s8, 256 // (op 2646) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2646) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 2646) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 2646) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 2646) 

s_mul_i32 s0, s18, s50 // (op 2646) 

s_sub_i32 s49, s1, s0 // (op 2646) 


s_mul_hi_i32 s0, s16, s11 // (op 2646) 
s_add_i32 s1, s0, s16 // (op 2646) 

s_ashr_i32 s0, s1, 31 // (op 2646) 
s_cmp_eq_i32 s11, 0 // (op 2646) 
s_mov_b32 s18, scc // (op 2646) 
s_cmp_eq_i32 s18, 0 // (op 2646) 
s_cselect_b32 s19, 0, -1 // (op 2646) 

s_and_b32 s18, s47, 31 // (op 2646) 
s_lshl_b32 s68, 1, s18 // (op 2646) 
s_add_i32 s69, s68, s19 // (op 2646) 


s_and_b32 s19, s0, s69 // (op 2646) 


s_add_i32 s0, s1, s19 // (op 2646) 


s_ashr_i32 s1, s0, s18 // (op 2646) 


s_lshl_b32 s0, s47, 24 // (op 2646) 
s_ashr_i32 s18, s0, 31 // (op 2646) 

s_xor_b32 s0, s1, s18 // (op 2646) 

s_sub_i32 s1, s0, s18 // (op 2646) 


s_mul_i32 s0, s13, s50 // (op 2646) 

s_mul_i32 s18, s10, s0 // (op 2646) 

s_mul_i32 s0, s1, s18 // (op 2646) 

s_sub_u32 s18, s16, s0 // (op 2646) 

s_mul_hi_i32 s0, s18, s14 // (op 2646) 
s_add_i32 s19, s0, s18 // (op 2646) 


s_ashr_i32 s0, s19, 31 // (op 2646) 
s_cmp_eq_i32 s14, 0 // (op 2646) 
s_mov_b32 s18, scc // (op 2646) 
s_cmp_eq_i32 s18, 0 // (op 2646) 
s_cselect_b32 s50, 0, -1 // (op 2646) 

s_and_b32 s18, s46, 31 // (op 2646) 
s_lshl_b32 s51, 1, s18 // (op 2646) 
s_add_i32 s68, s51, s50 // (op 2646) 


s_and_b32 s50, s0, s68 // (op 2646) 


s_add_i32 s0, s19, s50 // (op 2646) 


s_ashr_i32 s19, s0, s18 // (op 2646) 


s_lshl_b32 s0, s46, 24 // (op 2646) 
s_ashr_i32 s18, s0, 31 // (op 2646) 

s_xor_b32 s0, s19, s18 // (op 2646) 

s_sub_i32 s19, s0, s18 // (op 2646) 


s_cmp_eq_i32 s1, 0 // (op 2646) 
s_mov_b32 s0, scc // (op 2646) 

s_cmp_eq_i32 s0, 0 // (op 2646) 
s_cselect_b32 s1, s19, s49 // (op 2646) 



v_lshl_add_u32 v24, s1, 8, v22 // (op 2646) 


v_mul_lo_u32 v22, v23, s40 // (op 2646) 
v_add_u32 v22, v22, v24 // (op 2646) 


s_mul_i32 s1, s40, s39 // (op 2648) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 2648) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 2648) least significant
s_mul_hi_u32 s19, s40, s38 // (op 2648) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2648) most significant: sum
s_add_u32 s1, s1, s19 // (op 2648) most significant: sum


s_mov_b32 s68, s42 // (op 2648) reg expression
s_mov_b32 s69, s43 // (op 2648) reg expression
s_mov_b32 s70, s0 // (op 2648) reg expression
s_mov_b32 s71, 131072 // (op 2648) call()

s_mul_hi_i32 s0, s16, s12 // (op 2665) 
v_and_b32 v23, -64, v1 // (op 2665) 
s_add_i32 s1, s0, s16 // (op 2665) 

v_sub_u32 v24, v1, v23 // (op 2665) 

s_ashr_i32 s0, s1, 31 // (op 2665) 
v_lshlrev_b32 v23, 2, v24 // (op 2665) 

s_cmp_eq_i32 s12, 0 // (op 2665) 
s_mov_b32 s18, scc // (op 2665) 
v_lshrrev_b32 v24, 6, v1 // (op 2665) 
s_cmp_eq_i32 s18, 0 // (op 2665) 
s_cselect_b32 s19, 0, -1 // (op 2665) 

s_and_b32 s18, s15, 31 // (op 2665) 
s_lshl_b32 s49, 1, s18 // (op 2665) 
s_add_i32 s50, s49, s19 // (op 2665) 


s_and_b32 s19, s0, s50 // (op 2665) 


s_add_i32 s0, s1, s19 // (op 2665) 


s_ashr_i32 s1, s0, s18 // (op 2665) 


s_lshl_b32 s0, s15, 24 // (op 2665) 
s_ashr_i32 s18, s0, 31 // (op 2665) 

s_xor_b32 s0, s1, s18 // (op 2665) 

s_sub_i32 s1, s0, s18 // (op 2665) 


s_mul_hi_i32 s0, s1, s44 // (op 2665) 
s_add_i32 s18, s0, s1 // (op 2665) 

s_ashr_i32 s0, s18, 31 // (op 2665) 
s_cmp_eq_i32 s44, 0 // (op 2665) 
s_mov_b32 s19, scc // (op 2665) 
s_cmp_eq_i32 s19, 0 // (op 2665) 
s_cselect_b32 s49, 0, -1 // (op 2665) 

s_and_b32 s19, s45, 31 // (op 2665) 
s_lshl_b32 s50, 1, s19 // (op 2665) 
s_add_i32 s51, s50, s49 // (op 2665) 


s_and_b32 s49, s0, s51 // (op 2665) 


s_add_i32 s0, s18, s49 // (op 2665) 


s_ashr_i32 s18, s0, s19 // (op 2665) 


s_lshl_b32 s0, s45, 24 // (op 2665) 
s_ashr_i32 s19, s0, 31 // (op 2665) 

s_xor_b32 s0, s18, s19 // (op 2665) 

s_sub_i32 s18, s0, s19 // (op 2665) 


s_add_u32 s50, s8, 256 // (op 2665) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2665) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2665) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2665) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 2665) 

s_mul_i32 s0, s18, s50 // (op 2665) 

s_sub_i32 s49, s1, s0 // (op 2665) 


s_mul_hi_i32 s0, s16, s11 // (op 2665) 
s_add_i32 s1, s0, s16 // (op 2665) 

s_ashr_i32 s0, s1, 31 // (op 2665) 
s_cmp_eq_i32 s11, 0 // (op 2665) 
s_mov_b32 s18, scc // (op 2665) 
s_cmp_eq_i32 s18, 0 // (op 2665) 
s_cselect_b32 s19, 0, -1 // (op 2665) 

s_and_b32 s18, s47, 31 // (op 2665) 
s_lshl_b32 s72, 1, s18 // (op 2665) 
s_add_i32 s73, s72, s19 // (op 2665) 


s_and_b32 s19, s0, s73 // (op 2665) 


s_add_i32 s0, s1, s19 // (op 2665) 


s_ashr_i32 s1, s0, s18 // (op 2665) 


s_lshl_b32 s0, s47, 24 // (op 2665) 
s_ashr_i32 s18, s0, 31 // (op 2665) 

s_xor_b32 s0, s1, s18 // (op 2665) 

s_sub_i32 s1, s0, s18 // (op 2665) 


s_mul_i32 s0, s13, s50 // (op 2665) 

s_mul_i32 s18, s10, s0 // (op 2665) 

s_mul_i32 s0, s1, s18 // (op 2665) 

s_sub_u32 s18, s16, s0 // (op 2665) 

s_mul_hi_i32 s0, s18, s14 // (op 2665) 
s_add_i32 s19, s0, s18 // (op 2665) 


s_ashr_i32 s0, s19, 31 // (op 2665) 
s_cmp_eq_i32 s14, 0 // (op 2665) 
s_mov_b32 s18, scc // (op 2665) 
s_cmp_eq_i32 s18, 0 // (op 2665) 
s_cselect_b32 s50, 0, -1 // (op 2665) 

s_and_b32 s18, s46, 31 // (op 2665) 
s_lshl_b32 s51, 1, s18 // (op 2665) 
s_add_i32 s72, s51, s50 // (op 2665) 


s_and_b32 s50, s0, s72 // (op 2665) 


s_add_i32 s0, s19, s50 // (op 2665) 


s_ashr_i32 s19, s0, s18 // (op 2665) 


s_lshl_b32 s0, s46, 24 // (op 2665) 
s_ashr_i32 s18, s0, 31 // (op 2665) 

s_xor_b32 s0, s19, s18 // (op 2665) 

s_sub_i32 s19, s0, s18 // (op 2665) 


s_cmp_eq_i32 s1, 0 // (op 2665) 
s_mov_b32 s0, scc // (op 2665) 

s_cmp_eq_i32 s0, 0 // (op 2665) 
s_cselect_b32 s1, s19, s49 // (op 2665) 



v_lshl_add_u32 v25, s1, 8, v23 // (op 2665) 


v_mul_lo_u32 v23, v24, s40 // (op 2665) 
v_add_u32 v23, v23, v25 // (op 2665) 


s_mul_i32 s1, s40, s39 // (op 2667) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 2667) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 2667) least significant
s_mul_hi_u32 s19, s40, s38 // (op 2667) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2667) most significant: sum
s_add_u32 s1, s1, s19 // (op 2667) most significant: sum


s_mov_b32 s68, s42 // (op 2667) reg expression
s_mov_b32 s69, s43 // (op 2667) reg expression
s_mov_b32 s70, s0 // (op 2667) reg expression
s_mov_b32 s71, 131072 // (op 2667) call()


s_mul_i32 s1, s31, 6 // (op 2169) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 2169) least significant
s_mul_hi_u32 s18, s30, 6 // (op 2169) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2169) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2169) 


v_mov_b32 v26, v20 // (op 2169) reg expression
buffer_load_dwordx3 v[28:30], v26, s[52:55], 0 offen // (op 2169) Load value
buffer_load_dwordx3 v[32:34], v26, s[52:55], 0 offen offset:12 // (op 2169) Load value
buffer_load_dwordx3 v[36:38], v26, s[52:55], 0 offen offset:24 // (op 2169) Load value
buffer_load_dwordx3 v[40:42], v26, s[52:55], 0 offen offset:36 // (op 2169) Load value
v_add_u32 v26, v26, v24 // (op 2169) 
buffer_load_dwordx3 v[44:46], v26, s[52:55], 0 offen // (op 2169) Load value
buffer_load_dwordx3 v[48:50], v26, s[52:55], 0 offen offset:12 // (op 2169) Load value
buffer_load_dwordx3 v[52:54], v26, s[52:55], 0 offen offset:24 // (op 2169) Load value
buffer_load_dwordx3 v[56:58], v26, s[52:55], 0 offen offset:36 // (op 2169) Load value




v_mov_b32 v31, v22 // (op 2172) reg expression

buffer_load_dword v35, v31, s[68:71], 0 offen // (op 2172) Load value

v_and_b32 v31, -4, v1 // (op 2731) 
v_sub_u32 v39, v1, v31 // (op 2731) 

v_lshlrev_b32 v31, 4, v39 // (op 2731) 

v_lshrrev_b32 v39, 2, v1 // (op 2731) 
v_lshl_add_u32 v43, v39, 6, v31 // (op 2731) 


s_nop 0
v_readfirstlane_b32 s0, v43 // (op 2731) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 2407) 
v_mov_b64 v[24:25], s[18:19] // (op 2407) convert


v_mov_b64 v[26:27], v[14:15] // (op 2407) reg expression
s_mov_b32 m0, s0 // (op 2407) reg expression
buffer_load_dwordx4 v26, s[20:23], 0 offen offset: 0 lds // (op 2407) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 2407) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 2407) most significant half

s_add_u32 m0, m0, 4096 // (op 2407) 
buffer_load_dwordx4 v26, s[20:23], 0 offen offset: 0 lds // (op 2407) Load value direct to lds





v_mov_b32 v31, v18 // (op 2178) reg expression

buffer_load_ubyte v39, v31, s[64:67], 0 offen offset: 0 // (op 2178) Load value


s_mul_i32 s1, s31, 6 // (op 2181) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 2181) least significant
s_mul_hi_u32 s18, s30, 6 // (op 2181) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2181) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2181) 


s_lshl_b64 s[0:1], s[30:31], 7 // (op 2181) 
s_mul_i32 s19, s1, 6 // (op 2181) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 2181) least significant
s_mul_hi_u32 s49, s0, 6 // (op 2181) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 2181) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 2181) 

v_add_u32 v31, v20, s0 // (op 2181) 

buffer_load_dwordx3 v[60:62], v31, s[52:55], 0 offen // (op 2181) Load value
buffer_load_dwordx3 v[64:66], v31, s[52:55], 0 offen offset:12 // (op 2181) Load value
buffer_load_dwordx3 v[68:70], v31, s[52:55], 0 offen offset:24 // (op 2181) Load value
buffer_load_dwordx3 v[72:74], v31, s[52:55], 0 offen offset:36 // (op 2181) Load value
v_add_u32 v31, v31, v24 // (op 2181) 
buffer_load_dwordx3 v[76:78], v31, s[52:55], 0 offen // (op 2181) Load value
buffer_load_dwordx3 v[80:82], v31, s[52:55], 0 offen offset:12 // (op 2181) Load value
buffer_load_dwordx3 v[84:86], v31, s[52:55], 0 offen offset:24 // (op 2181) Load value
buffer_load_dwordx3 v[88:90], v31, s[52:55], 0 offen offset:36 // (op 2181) Load value





s_lshl_b32 s0, s40, 2 // (op 2184) 
v_add_u32 v20, v22, s0 // (op 2184) 


buffer_load_dword v31, v20, s[68:71], 0 offen // (op 2184) Load value


v_and_b32 v20, -4, v1 // (op 2957) 
v_sub_u32 v22, v1, v20 // (op 2957) 

v_lshlrev_b32 v20, 4, v22 // (op 2957) 

v_lshrrev_b32 v22, 2, v1 // (op 2957) 
v_lshl_add_u32 v43, v22, 6, v20 // (op 2957) 


s_nop 0
v_readfirstlane_b32 s0, v43 // (op 2957) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 2412) 
v_mov_b64 v[24:25], s[18:19] // (op 2412) convert


s_lshl_b64 s[18:19], s[24:25], 7 // (op 2412) 
v_mov_b32 v20, s18 // (op 2412) 
v_mov_b32 v22, s19 // (op 2412) 
v_add_co_u32 v26, s[50:51], v14, v20 // (op 2412) least significant half
v_addc_co_u32 v27, s[50:51], v15, v22, s[50:51] // (op 2412) most significant half




s_add_u32 s1, 8192, s0 // (op 2412) 
s_mov_b32 m0, s1 // (op 2412) reg expression
buffer_load_dwordx4 v26, s[20:23], 0 offen offset: 0 lds // (op 2412) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 2412) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 2412) most significant half

s_add_u32 m0, m0, 4096 // (op 2412) 
buffer_load_dwordx4 v26, s[20:23], 0 offen offset: 0 lds // (op 2412) Load value direct to lds







s_lshl_b32 s0, s34, 2 // (op 2190) 
v_add_u32 v20, v18, s0 // (op 2190) 


buffer_load_ubyte v22, v20, s[64:67], 0 offen offset: 0 // (op 2190) Load value


v_and_b32 v18, -4, v1 // (op 2784) 
v_sub_u32 v20, v1, v18 // (op 2784) 

v_lshlrev_b32 v18, 6, v20 // (op 2784) 

v_lshrrev_b32 v20, 2, v1 // (op 2784) 
v_lshl_add_u32 v43, v20, 9, v18 // (op 2784) 


v_mul_lo_u32 v18, v43, 6 // (op 2784) 
v_lshrrev_b32 v20, 3, v18 // (op 2784) 

v_lshrrev_b32 v18, 4, v43 // (op 2784) 

v_lshlrev_b32 v43, 2, v18 // (op 2784) 

v_add_u32 v18, v20, v43 // (op 2784) 





s_waitcnt vmcnt(23)
ds_write_b96 v18, v[28:30] offset:16384 // (op 2192) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v18, v[32:34] offset:16400 // (op 2192) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v18, v[36:38] offset:16416 // (op 2192) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v18, v[40:42] offset:16432 // (op 2192) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v18, v[44:46] offset:16640 // (op 2192) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v18, v[48:50] offset:16656 // (op 2192) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v18, v[52:54] offset:16672 // (op 2192) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v18, v[56:58] offset:16688 // (op 2192) Store local data 

v_and_b32 v18, -64, v1 // (op 2895) 
v_sub_u32 v20, v1, v18 // (op 2895) 

v_lshlrev_b32 v18, 2, v20 // (op 2895) 

v_lshrrev_b32 v20, 6, v1 // (op 2895) 
v_lshl_add_u32 v43, v20, 8, v18 // (op 2895) 





s_waitcnt vmcnt(15)
ds_write_b32 v43, v35 offset:49152 // (op 2193) Store local data 

v_and_b32 v18, -64, v1 // (op 2842) 
v_sub_u32 v20, v1, v18 // (op 2842) 

v_and_b32 v18, 4294967232, v1 // (op 2842) 
v_add_u32 v43, v18, v20 // (op 2842) 





s_waitcnt vmcnt(12)
ds_write_b8 v43, v39 offset:50176 // (op 2195) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 3133) 



ds_read_b96_tr_b6 v[92:94], v9 offset:16384 // (op 2213) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:20480 // (op 2213) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3174) 


ds_read_u8 v18, v6 offset:49152 // (op 2236) Load local data 



ds_read_b96_tr_b6 v[98:100], v9 offset:16448 // (op 2218) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v9 offset:20544 // (op 2218) Transpose load local data 


ds_read_u8 v20, v6 offset:49216 // (op 2241) Load local data 



ds_read_b96_tr_b6 v[104:106], v9 offset:16512 // (op 2223) Transpose load local data 
ds_read_b96_tr_b6 v[107:109], v9 offset:20608 // (op 2223) Transpose load local data 


ds_read_u8 v43, v6 offset:49280 // (op 2246) Load local data 



ds_read_b96_tr_b6 v[110:112], v9 offset:16576 // (op 2228) Transpose load local data 
ds_read_b96_tr_b6 v[113:115], v9 offset:20672 // (op 2228) Transpose load local data 


ds_read_u8 v47, v6 offset:49344 // (op 2251) Load local data 
s_barrier  // (op 3114) 



ds_read_b64_tr_b8 v[116:117], v3 // (op 2209) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v3 offset:512 // (op 2209) Transpose load local data 
ds_read_b64_tr_b8 v[120:121], v3 offset:2048 // (op 2209) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v3 offset:2560 // (op 2209) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 3154) 


ds_read_u8 v51, v0 offset:50176 // (op 2232) Load local data 
v_and_b32 v55, -4, v1 // (op 2703) 
v_sub_u32 v59, v1, v55 // (op 2703) 

v_lshlrev_b32 v55, 4, v59 // (op 2703) 

v_lshrrev_b32 v59, 2, v1 // (op 2703) 
v_lshl_add_u32 v63, v59, 6, v55 // (op 2703) 


s_nop 0
v_readfirstlane_b32 s0, v63 // (op 2703) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v55, -4, v1 // (op 2755) 
v_sub_u32 v59, v1, v55 // (op 2755) 

v_lshlrev_b32 v55, 6, v59 // (op 2755) 

v_lshrrev_b32 v59, 2, v1 // (op 2755) 
v_lshl_add_u32 v63, v59, 9, v55 // (op 2755) 


v_mul_lo_u32 v55, v63, 6 // (op 2755) 
v_lshrrev_b32 v59, 3, v55 // (op 2755) 

v_lshrrev_b32 v55, 4, v63 // (op 2755) 

v_lshlrev_b32 v63, 2, v55 // (op 2755) 

v_add_u32 v55, v59, v63 // (op 2755) 


v_and_b32 v59, -64, v1 // (op 2813) 
v_sub_u32 v63, v1, v59 // (op 2813) 

v_and_b32 v59, 4294967232, v1 // (op 2813) 
v_add_u32 v67, v59, v63 // (op 2813) 


v_and_b32 v59, -64, v1 // (op 2866) 
v_sub_u32 v63, v1, v59 // (op 2866) 

v_lshlrev_b32 v59, 2, v63 // (op 2866) 

v_lshrrev_b32 v63, 6, v1 // (op 2866) 
v_lshl_add_u32 v71, v63, 8, v59 // (op 2866) 


v_and_b32 v59, -4, v1 // (op 2942) 
v_sub_u32 v63, v1, v59 // (op 2942) 

v_lshlrev_b32 v59, 4, v63 // (op 2942) 

v_lshrrev_b32 v63, 2, v1 // (op 2942) 
v_lshl_add_u32 v75, v63, 6, v59 // (op 2942) 


s_nop 0
v_readfirstlane_b32 s1, v75 // (op 2942) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v59, -4, v1 // (op 2981) 
v_sub_u32 v63, v1, v59 // (op 2981) 

v_lshlrev_b32 v59, 6, v63 // (op 2981) 

v_lshrrev_b32 v63, 2, v1 // (op 2981) 
v_lshl_add_u32 v75, v63, 9, v59 // (op 2981) 


v_mul_lo_u32 v59, v75, 6 // (op 2981) 
v_lshrrev_b32 v63, 3, v59 // (op 2981) 

v_lshrrev_b32 v59, 4, v75 // (op 2981) 

v_lshlrev_b32 v75, 2, v59 // (op 2981) 

v_add_u32 v59, v63, v75 // (op 2981) 


v_and_b32 v63, -64, v1 // (op 3010) 
v_sub_u32 v75, v1, v63 // (op 3010) 

v_and_b32 v63, 4294967232, v1 // (op 3010) 
v_add_u32 v79, v63, v75 // (op 3010) 


v_and_b32 v63, -64, v1 // (op 3034) 
v_sub_u32 v75, v1, v63 // (op 3034) 

v_lshlrev_b32 v63, 2, v75 // (op 3034) 

v_lshrrev_b32 v75, 6, v1 // (op 3034) 
v_lshl_add_u32 v83, v75, 8, v63 // (op 3034) 


s_mov_b32 s18, 0 // (op 872) call()
s_ashr_i64 s[50:51], s[28:29], 7 // (op 871) 
s_mov_b32 s19, s50 // (op 871) convert

s_and_b32 s49, s19, 4294967294 // (op 871) 

s_cmp_lt_u32 s18, s49 // (op 871) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_871 // (op 871) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_871 if false)
RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_871:

s_add_u32 s18, s18, 2 // (op 873) 
s_mul_i32 s51, s31, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s30, 6 // (op 10) least significant
s_mul_hi_u32 s19, s30, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 10) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[50:51] // (op 10) 


v_mov_b32 v63, v21 // (op 10) 
s_lshl_b64 s[50:51], s[30:31], 7 // (op 10) 
s_mul_i32 s73, s51, 6 // (op 10) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 10) least significant
s_mul_hi_u32 s19, s50, 6 // (op 10) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 10) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 10) 

s_lshl_b32 s19, s50, 1 // (op 10) 

v_add_u32 v75, v63, s19 // (op 10) 

buffer_load_dwordx3 v[28:30], v75, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[32:34], v75, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v75, s[52:55], 0 offen offset:24 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[116:123], v[92:97], a[0:15], v51, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 90) 
buffer_load_dwordx3 v[40:42], v75, s[52:55], 0 offen offset:36 // (op 10) Load value
s_barrier  // (op 3123) 
v_add_u32 v75, v75, v14 // (op 10) 



buffer_load_dwordx3 v[44:46], v75, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[48:50], v75, s[52:55], 0 offen offset:12 // (op 10) Load value
ds_read_b64_tr_b8 v[124:125], v3 offset:4096 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[126:127], v3 offset:4608 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[128:129], v3 offset:6144 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[130:131], v3 offset:6656 // (op 93) Transpose load local data 
buffer_load_dwordx3 v[52:54], v75, s[52:55], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 3143) 
buffer_load_dwordx3 v[56:58], v75, s[52:55], 0 offen offset:36 // (op 10) Load value







ds_read_b96_tr_b6 v[92:94], v9 offset:32768 // (op 97) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:36864 // (op 97) Transpose load local data 
v_mov_b32 v75, v23 // (op 25) 
s_barrier  // (op 3164) 
s_lshl_b32 s19, s40, 3 // (op 25) 


ds_read_u8 v87, v0 offset:50304 // (op 101) Load local data 
v_add_u32 v91, v75, s19 // (op 25) 


s_barrier  // (op 3184) 
buffer_load_dword v35, v91, s[68:71], 0 offen // (op 25) Load value



s_waitcnt lgkmcnt(3)
s_barrier  // (op 3117) 
ds_read_u8 v18, v6 offset:49664 // (op 105) Load local data 

s_lshl_b64 s[50:51], s[24:25], 6 // (op 2391) 
v_mov_b64 v[14:15], s[50:51] // (op 2391) convert


v_mov_b64 v[24:25], v[16:17] // (op 2391) 
s_lshl_b64 s[50:51], s[24:25], 7 // (op 2391) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 2391) 

v_mov_b32 v91, s72 // (op 2391) 
v_mov_b32 v26, s73 // (op 2391) 
v_add_co_u32 v132, s[50:51], v24, v91 // (op 2391) least significant half
v_addc_co_u32 v133, s[50:51], v25, v26, s[50:51] // (op 2391) most significant half




s_mov_b32 m0, s0 // (op 2391) reg expression
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[124:131], v[92:97], a[0:15], v87, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 107) 
buffer_load_dwordx4 v132, s[20:23], 0 offen offset: 0 lds // (op 2391) Load value direct to lds
v_add_co_u32 v132, s[50:51], v132, v14 // (op 2391) least significant half
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[116:123], v[98:103], a[16:31], v51, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 451) 
v_addc_co_u32 v133, s[50:51], 0, v133, s[50:51] // (op 2391) most significant half




s_add_u32 m0, m0, 4096 // (op 2391) 
ds_read_b96_tr_b6 v[92:94], v9 offset:32832 // (op 455) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:36928 // (op 455) Transpose load local data 
buffer_load_dwordx4 v132, s[20:23], 0 offen offset: 0 lds // (op 2391) Load value direct to lds






ds_read_u8 v18, v6 offset:49728 // (op 459) Load local data 
v_mov_b32 v91, v19 // (op 17) 
s_lshl_b32 s19, s34, 3 // (op 17) 
v_add_u32 v14, v91, s19 // (op 17) 


buffer_load_ubyte v39, v14, s[64:67], 0 offen offset: 0 // (op 17) Load value

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[124:131], v[92:97], a[16:31], v87, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 460) 
s_barrier  // (op 3204) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[116:123], v[104:109], a[32:47], v51, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 685) 
s_waitcnt vmcnt(23)
ds_write_b96 v59, v[60:62] offset:50432 // (op 306) Store local data 



ds_read_b96_tr_b6 v[92:94], v9 offset:32896 // (op 689) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:36992 // (op 689) Transpose load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v59, v[64:66] offset:50448 // (op 306) Store local data 


ds_read_u8 v18, v6 offset:49792 // (op 693) Load local data 
s_waitcnt vmcnt(21)
ds_write_b96 v59, v[68:70] offset:50464 // (op 306) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[124:131], v[92:97], a[32:47], v87, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 694) 
s_waitcnt vmcnt(20)
ds_write_b96 v59, v[72:74] offset:50480 // (op 306) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v59, v[76:78] offset:50688 // (op 306) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[116:123], v[110:115], a[48:63], v51, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 919) 
s_waitcnt vmcnt(18)
ds_write_b96 v59, v[80:82] offset:50704 // (op 306) Store local data 



ds_read_b96_tr_b6 v[110:112], v9 offset:32960 // (op 923) Transpose load local data 
ds_read_b96_tr_b6 v[113:115], v9 offset:37056 // (op 923) Transpose load local data 
s_waitcnt vmcnt(17)
ds_write_b96 v59, v[84:86] offset:50720 // (op 306) Store local data 


ds_read_u8 v47, v6 offset:49856 // (op 927) Load local data 
s_waitcnt vmcnt(16)
ds_write_b96 v59, v[88:90] offset:50736 // (op 306) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[124:131], v[110:115], a[48:63], v87, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 928) 


s_barrier  // (op 3216) 



v_add_u32 v87, 83200, v83 // (op 308) 
s_waitcnt vmcnt(15)
ds_write_b32 v87, v31 // (op 308) Store local data 

s_barrier  // (op 3210) 



v_add_u32 v87, 84224, v79 // (op 307) 
s_waitcnt vmcnt(12)
ds_write_b8 v87, v22 // (op 307) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 3200) 



ds_read_b96_tr_b6 v[92:94], v13 offset:50432 // (op 291) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v13 offset:54528 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3213) 


v_add_u32 v87, 83200, v12 // (op 295) 
ds_read_u8 v18, v87 // (op 295) Load local data 




ds_read_b96_tr_b6 v[98:100], v13 offset:50496 // (op 474) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v13 offset:54592 // (op 474) Transpose load local data 


v_add_u32 v87, 83264, v12 // (op 478) 
ds_read_u8 v20, v87 // (op 478) Load local data 




ds_read_b96_tr_b6 v[104:106], v13 offset:50560 // (op 708) Transpose load local data 
ds_read_b96_tr_b6 v[107:109], v13 offset:54656 // (op 708) Transpose load local data 


v_add_u32 v87, 83328, v12 // (op 712) 
ds_read_u8 v43, v87 // (op 712) Load local data 




ds_read_b96_tr_b6 v[124:126], v13 offset:50624 // (op 942) Transpose load local data 
ds_read_b96_tr_b6 v[127:129], v13 offset:54720 // (op 942) Transpose load local data 


v_add_u32 v87, 83392, v12 // (op 946) 
ds_read_u8 v14, v87 // (op 946) Load local data 

s_barrier  // (op 3194) 



ds_read_b64_tr_b8 v[116:117], v5 offset:8192 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v5 offset:8704 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[120:121], v5 offset:10240 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v5 offset:10752 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 3207) 


v_add_u32 v15, 84224, v4 // (op 293) 
ds_read_u8 v51, v15 // (op 293) Load local data 


s_mul_i32 s51, s31, 6 // (op 309) most significant: low of high * low
s_mul_i32 s50, s30, 6 // (op 309) least significant
s_mul_hi_u32 s19, s30, 6 // (op 309) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 309) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[50:51] // (op 309) 


s_lshl_b64 s[50:51], s[30:31], 7 // (op 309) 
s_mul_i32 s73, s51, 6 // (op 309) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 309) least significant
s_mul_hi_u32 s19, s50, 6 // (op 309) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 309) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 309) 

s_mul_i32 s19, 3, s50 // (op 309) 

v_add_u32 v15, v63, s19 // (op 309) 

buffer_load_dwordx3 v[60:62], v15, s[52:55], 0 offen // (op 309) Load value
buffer_load_dwordx3 v[64:66], v15, s[52:55], 0 offen offset:12 // (op 309) Load value
buffer_load_dwordx3 v[68:70], v15, s[52:55], 0 offen offset:24 // (op 309) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[116:123], v[92:97], a[0:15], v51, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 296) 
buffer_load_dwordx3 v[72:74], v15, s[52:55], 0 offen offset:36 // (op 309) Load value
v_add_u32 v15, v15, v26 // (op 309) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[116:123], v[98:103], a[16:31], v51, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 461) 
buffer_load_dwordx3 v[76:78], v15, s[52:55], 0 offen // (op 309) Load value
buffer_load_dwordx3 v[80:82], v15, s[52:55], 0 offen offset:12 // (op 309) Load value
ds_read_b64_tr_b8 v[130:131], v5 offset:12288 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[132:133], v5 offset:12800 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[134:135], v5 offset:14336 // (op 298) Transpose load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[116:123], v[104:109], a[32:47], v51, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 695) 
ds_read_b64_tr_b8 v[136:137], v5 offset:14848 // (op 298) Transpose load local data 



v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[116:123], v[124:129], a[48:63], v51, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 929) 
v_add_u32 v87, 66816, v13 // (op 300) 


ds_read_b96_tr_b6 v[124:126], v87 // (op 300) Transpose load local data 

v_add_u32 v14, 70912, v13 // (op 300) 
ds_read_b96_tr_b6 v[127:129], v14 // (op 300) Transpose load local data 

buffer_load_dwordx3 v[84:86], v15, s[52:55], 0 offen offset:24 // (op 309) Load value


v_add_u32 v14, 84352, v4 // (op 302) 
ds_read_u8 v87, v14 // (op 302) Load local data 

buffer_load_dwordx3 v[88:90], v15, s[52:55], 0 offen offset:36 // (op 309) Load value







s_barrier  // (op 3259) 
s_lshl_b32 s19, s40, 2 // (op 311) 
s_mul_i32 s49, 3, s19 // (op 311) 

v_add_u32 v63, 83712, v12 // (op 304) 
ds_read_u8 v14, v63 // (op 304) Load local data 

v_add_u32 v15, v75, s49 // (op 311) 





buffer_load_dword v31, v15, s[68:71], 0 offen // (op 311) Load value

v_add_u32 v15, 66880, v13 // (op 465) 

s_lshl_b64 s[50:51], s[30:31], 7 // (op 2529) 
s_lshl_b32 s19, s40, 3 // (op 2657) 
v_add_u32 v23, v23, s19 // (op 2657) 

s_mul_i32 s73, s51, 6 // (op 2529) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 2529) least significant
s_mul_hi_u32 s19, s50, 6 // (op 2529) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 2529) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 2529) 

s_lshl_b32 s19, s50, 1 // (op 2529) 

v_add_u32 v21, v21, s19 // (op 2529) 

ds_read_b96_tr_b6 v[138:140], v15 // (op 465) Transpose load local data 

v_add_u32 v15, 70976, v13 // (op 465) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[130:137], v[124:129], a[0:15], v87, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 305) 
s_barrier  // (op 3197) 



ds_read_b96_tr_b6 v[141:143], v15 // (op 465) Transpose load local data 

s_lshl_b64 s[50:51], s[24:25], 6 // (op 2402) 


v_add_u32 v63, 83776, v12 // (op 469) 
ds_read_u8 v75, v63 // (op 469) Load local data 

v_mov_b64 v[14:15], s[50:51] // (op 2402) convert





v_add_u32 v63, 66944, v13 // (op 699) 
s_lshl_b64 s[50:51], s[24:25], 7 // (op 2402) 
s_mul_i32 s73, 3, s51 // (op 2402) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 2402) least significant
s_mul_hi_u32 s19, 3, s50 // (op 2402) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 2402) most significant: sum


v_mov_b32 v26, s72 // (op 2402) 
v_mov_b32 v27, s73 // (op 2402) 
ds_read_b96_tr_b6 v[124:126], v63 // (op 699) Transpose load local data 

v_add_u32 v63, 71040, v13 // (op 699) 
v_add_co_u32 v144, s[50:51], v24, v26 // (op 2402) least significant half
v_addc_co_u32 v145, s[50:51], v25, v27, s[50:51] // (op 2402) most significant half




s_add_u32 s19, 8192, s1 // (op 2402) 
s_mov_b32 m0, s19 // (op 2402) reg expression
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[130:137], v[138:143], a[16:31], v87, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 470) 
ds_read_b96_tr_b6 v[127:129], v63 // (op 699) Transpose load local data 





v_add_u32 v63, 83840, v12 // (op 703) 
ds_read_u8 v75, v63 // (op 703) Load local data 

buffer_load_dwordx4 v144, s[20:23], 0 offen offset: 0 lds // (op 2402) Load value direct to lds



v_add_u32 v63, 67008, v13 // (op 933) 
v_add_co_u32 v144, s[50:51], v144, v14 // (op 2402) least significant half
v_addc_co_u32 v145, s[50:51], 0, v145, s[50:51] // (op 2402) most significant half

s_add_u32 m0, m0, 4096 // (op 2402) 
buffer_load_dwordx4 v144, s[20:23], 0 offen offset: 0 lds // (op 2402) Load value direct to lds



ds_read_b96_tr_b6 v[138:140], v63 // (op 933) Transpose load local data 


v_add_u32 v63, 71104, v13 // (op 933) 


ds_read_b96_tr_b6 v[141:143], v63 // (op 933) Transpose load local data 

s_lshl_b32 s19, s34, 2 // (op 310) 
s_mul_i32 s49, 3, s19 // (op 310) 



s_barrier  // (op 3317) 
v_add_u32 v63, v91, s49 // (op 310) 


buffer_load_ubyte v22, v63, s[64:67], 0 offen offset: 0 // (op 310) Load value

v_add_u32 v63, 83904, v12 // (op 937) 

ds_read_u8 v91, v63 // (op 937) Load local data 

s_barrier  // (op 3136) 



s_barrier  // (op 3321) 
s_lshl_b32 s19, s34, 3 // (op 2593) 
v_add_u32 v19, v19, s19 // (op 2593) 

s_lshl_b64 s[50:51], s[24:25], 7 // (op 2465) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 2465) 

v_mov_b32 v63, s72 // (op 2465) 
v_mov_b32 v14, s73 // (op 2465) 
v_add_co_u32 v16, s[50:51], v16, v63 // (op 2465) least significant half
v_addc_co_u32 v17, s[50:51], v17, v14, s[50:51] // (op 2465) most significant half




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[130:137], v[124:129], a[32:47], v87, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 704) 
s_waitcnt vmcnt(23)
ds_write_b96 v55, v[28:30] offset:16384 // (op 13) Store local data 


s_waitcnt vmcnt(22)
ds_write_b96 v55, v[32:34] offset:16400 // (op 13) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[130:137], v[138:143], a[48:63], v87, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 938) 
s_waitcnt vmcnt(21)
ds_write_b96 v55, v[36:38] offset:16416 // (op 13) Store local data 




s_waitcnt vmcnt(20)
ds_write_b96 v55, v[40:42] offset:16432 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v55, v[44:46] offset:16640 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v55, v[48:50] offset:16656 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v55, v[52:54] offset:16672 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v55, v[56:58] offset:16688 // (op 13) Store local data 
s_barrier  // (op 3177) 



s_waitcnt vmcnt(15)
ds_write_b32 v71, v35 offset:49152 // (op 30) Store local data 
s_barrier  // (op 3157) 



s_waitcnt vmcnt(12)
ds_write_b8 v67, v39 offset:50176 // (op 21) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3146) 



ds_read_b96_tr_b6 v[92:94], v9 offset:16384 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v9 offset:20480 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3187) 


ds_read_u8 v18, v6 offset:49152 // (op 87) Load local data 



ds_read_b96_tr_b6 v[98:100], v9 offset:16448 // (op 446) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v9 offset:20544 // (op 446) Transpose load local data 


ds_read_u8 v20, v6 offset:49216 // (op 450) Load local data 



ds_read_b96_tr_b6 v[104:106], v9 offset:16512 // (op 680) Transpose load local data 
ds_read_b96_tr_b6 v[107:109], v9 offset:20608 // (op 680) Transpose load local data 


ds_read_u8 v43, v6 offset:49280 // (op 684) Load local data 



ds_read_b96_tr_b6 v[110:112], v9 offset:16576 // (op 914) Transpose load local data 
ds_read_b96_tr_b6 v[113:115], v9 offset:20672 // (op 914) Transpose load local data 


ds_read_u8 v47, v6 offset:49344 // (op 918) Load local data 
s_barrier  // (op 3126) 



ds_read_b64_tr_b8 v[116:117], v3 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v3 offset:512 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[120:121], v3 offset:2048 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v3 offset:2560 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 3167) 


ds_read_u8 v51, v0 offset:50176 // (op 82) Load local data 
s_ashr_i64 s[50:51], s[28:29], 7 // (op 871) 
s_mov_b32 s19, s50 // (op 871) convert

s_and_b32 s49, s19, 4294967294 // (op 871) 

s_cmp_lt_u32 s18, s49 // (op 871) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_871 // (op 871) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_871 if true)
RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_871:

















s_barrier  // (op 3299) 
s_mul_hi_i32 s0, s16, s12 // (op 2425) 
v_and_b32 v19, -4, v1 // (op 2425) 
s_add_i32 s1, s0, s16 // (op 2425) 

v_sub_u32 v21, v1, v19 // (op 2425) 

s_ashr_i32 s0, s1, 31 // (op 2425) 
v_lshlrev_b32 v19, 4, v21 // (op 2425) 

s_cmp_eq_i32 s12, 0 // (op 2425) 
s_mov_b32 s18, scc // (op 2425) 
v_lshrrev_b32 v21, 2, v1 // (op 2425) 
s_cmp_eq_i32 s18, 0 // (op 2425) 
s_cselect_b32 s19, 0, -1 // (op 2425) 

s_and_b32 s18, s15, 31 // (op 2425) 
s_lshl_b32 s49, 1, s18 // (op 2425) 
s_add_i32 s50, s49, s19 // (op 2425) 


s_and_b32 s19, s0, s50 // (op 2425) 


s_add_i32 s0, s1, s19 // (op 2425) 


s_ashr_i32 s1, s0, s18 // (op 2425) 


s_lshl_b32 s0, s15, 24 // (op 2425) 
s_ashr_i32 s18, s0, 31 // (op 2425) 

s_xor_b32 s0, s1, s18 // (op 2425) 

s_sub_i32 s1, s0, s18 // (op 2425) 


s_mul_hi_i32 s0, s1, s44 // (op 2425) 
s_add_i32 s18, s0, s1 // (op 2425) 

s_ashr_i32 s0, s18, 31 // (op 2425) 
s_cmp_eq_i32 s44, 0 // (op 2425) 
s_mov_b32 s19, scc // (op 2425) 
s_cmp_eq_i32 s19, 0 // (op 2425) 
s_cselect_b32 s49, 0, -1 // (op 2425) 

s_and_b32 s19, s45, 31 // (op 2425) 
s_lshl_b32 s50, 1, s19 // (op 2425) 
s_add_i32 s51, s50, s49 // (op 2425) 


s_and_b32 s49, s0, s51 // (op 2425) 


s_add_i32 s0, s18, s49 // (op 2425) 


s_ashr_i32 s18, s0, s19 // (op 2425) 


s_lshl_b32 s0, s45, 24 // (op 2425) 
s_ashr_i32 s19, s0, 31 // (op 2425) 

s_xor_b32 s0, s18, s19 // (op 2425) 

s_sub_i32 s18, s0, s19 // (op 2425) 


s_mul_i32 s0, s1, s13 // (op 2425) 

s_sub_u32 s1, s16, s0 // (op 2425) 

s_mul_i32 s0, s18, s13 // (op 2425) 
s_add_u32 s0, s0, s1 // (op 2425) 


s_mul_hi_i32 s1, s16, s11 // (op 2425) 
s_add_i32 s18, s1, s16 // (op 2425) 

s_ashr_i32 s1, s18, 31 // (op 2425) 
s_cmp_eq_i32 s11, 0 // (op 2425) 
s_mov_b32 s19, scc // (op 2425) 
s_cmp_eq_i32 s19, 0 // (op 2425) 
s_cselect_b32 s49, 0, -1 // (op 2425) 

s_and_b32 s19, s47, 31 // (op 2425) 
s_lshl_b32 s50, 1, s19 // (op 2425) 
s_add_i32 s51, s50, s49 // (op 2425) 


s_and_b32 s49, s1, s51 // (op 2425) 


s_add_i32 s1, s18, s49 // (op 2425) 


s_ashr_i32 s18, s1, s19 // (op 2425) 


s_lshl_b32 s1, s47, 24 // (op 2425) 
s_ashr_i32 s19, s1, 31 // (op 2425) 

s_xor_b32 s1, s18, s19 // (op 2425) 

s_sub_i32 s18, s1, s19 // (op 2425) 


s_add_u32 s50, s8, 256 // (op 2425) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2425) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2425) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2425) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 2425) 

s_mul_i32 s1, s13, s50 // (op 2425) 

s_mul_i32 s19, s10, s1 // (op 2425) 

s_mul_i32 s1, s18, s19 // (op 2425) 

s_sub_u32 s19, s16, s1 // (op 2425) 

s_mov_b32 s1, s19 // (op 2425) 
s_mul_hi_i32 s49, s1, s14 // (op 2425) 
s_add_i32 s50, s49, s1 // (op 2425) 


s_ashr_i32 s1, s50, 31 // (op 2425) 
s_cmp_eq_i32 s14, 0 // (op 2425) 
s_mov_b32 s49, scc // (op 2425) 
s_cmp_eq_i32 s49, 0 // (op 2425) 
s_cselect_b32 s51, 0, -1 // (op 2425) 

s_and_b32 s49, s46, 31 // (op 2425) 
s_lshl_b32 s72, 1, s49 // (op 2425) 
s_add_i32 s73, s72, s51 // (op 2425) 


s_and_b32 s51, s1, s73 // (op 2425) 


s_add_i32 s1, s50, s51 // (op 2425) 


s_ashr_i32 s50, s1, s49 // (op 2425) 


s_lshl_b32 s1, s46, 24 // (op 2425) 
s_ashr_i32 s49, s1, 31 // (op 2425) 

s_xor_b32 s1, s50, s49 // (op 2425) 

s_sub_i32 s50, s1, s49 // (op 2425) 


s_mul_i32 s1, s48, s13 // (op 2425) 
s_add_u32 s72, s6, 64 // (op 2425) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 2425) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 2425) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 2425) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 2425) 

s_sub_i32 s49, s72, s1 // (op 2425) 

s_mul_i32 s51, s50, s49 // (op 2425) 


s_sub_u32 s49, s19, s51 // (op 2425) 


s_add_u32 s19, s49, s1 // (op 2425) 


s_cmp_eq_i32 s18, 0 // (op 2425) 
s_mov_b32 s1, scc // (op 2425) 

s_cmp_eq_i32 s1, 0 // (op 2425) 
s_cselect_b32 s18, s19, s0 // (op 2425) 



v_lshl_add_u32 v63, s18, 6, v19 // (op 2425) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2425) 
v_mov_b32 v15, 0 // (op 2425) convert
v_mov_b32 v14, v63 // (op 2425) convert

s_lshr_b32 s18, s0, 1 // (op 2425) 

v_lshl_add_u32 v19, s18, 8, v21 // (op 2425) 


v_mul_lo_u32 v17, v19, s25 // (op 2425) most significant: low of low * high
v_mul_hi_u32 v21, v19, s24 // (op 2425) most significant: high of low * low
v_mul_lo_u32 v16, v19, s24 // (op 2425) least significant: low of low * low
v_add_u32 v17, v17, v21 // (op 2425) most significant: sum


v_add_co_u32 v32, s[0:1], v14, v16 // (op 2425) least significant half
v_addc_co_u32 v33, s[0:1], v15, v17, s[0:1] // (op 2425) most significant half



s_mul_i32 s1, s24, s29 // (op 2427) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 2427) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 2427) least significant
s_mul_hi_u32 s19, s24, s28 // (op 2427) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2427) most significant: sum
s_add_u32 s1, s1, s19 // (op 2427) most significant: sum


s_mov_b32 s20, s26 // (op 2427) reg expression
s_mov_b32 s21, s27 // (op 2427) reg expression
s_mov_b32 s22, s0 // (op 2427) reg expression
s_mov_b32 s23, 131072 // (op 2427) call()

s_mul_hi_i32 s0, s16, s12 // (op 2495) 
v_and_b32 v19, -4, v1 // (op 2495) 
s_add_i32 s1, s0, s16 // (op 2495) 

v_sub_u32 v21, v1, v19 // (op 2495) 

s_ashr_i32 s0, s1, 31 // (op 2495) 
v_lshlrev_b32 v19, 6, v21 // (op 2495) 

s_cmp_eq_i32 s12, 0 // (op 2495) 
s_mov_b32 s18, scc // (op 2495) 
v_lshrrev_b32 v21, 2, v1 // (op 2495) 
s_cmp_eq_i32 s18, 0 // (op 2495) 
s_cselect_b32 s19, 0, -1 // (op 2495) 

v_lshlrev_b32 v63, 1, v21 // (op 2495) 

s_and_b32 s18, s15, 31 // (op 2495) 
s_lshl_b32 s49, 1, s18 // (op 2495) 
s_add_i32 s50, s49, s19 // (op 2495) 


s_and_b32 s19, s0, s50 // (op 2495) 


s_add_i32 s0, s1, s19 // (op 2495) 


s_ashr_i32 s1, s0, s18 // (op 2495) 


s_lshl_b32 s0, s15, 24 // (op 2495) 
s_ashr_i32 s18, s0, 31 // (op 2495) 

s_xor_b32 s0, s1, s18 // (op 2495) 

s_sub_i32 s1, s0, s18 // (op 2495) 


s_mul_hi_i32 s0, s1, s44 // (op 2495) 
s_add_i32 s18, s0, s1 // (op 2495) 

s_ashr_i32 s0, s18, 31 // (op 2495) 
s_cmp_eq_i32 s44, 0 // (op 2495) 
s_mov_b32 s19, scc // (op 2495) 
s_cmp_eq_i32 s19, 0 // (op 2495) 
s_cselect_b32 s49, 0, -1 // (op 2495) 

s_and_b32 s19, s45, 31 // (op 2495) 
s_lshl_b32 s50, 1, s19 // (op 2495) 
s_add_i32 s51, s50, s49 // (op 2495) 


s_and_b32 s49, s0, s51 // (op 2495) 


s_add_i32 s0, s18, s49 // (op 2495) 


s_ashr_i32 s18, s0, s19 // (op 2495) 


s_lshl_b32 s0, s45, 24 // (op 2495) 
s_ashr_i32 s19, s0, 31 // (op 2495) 

s_xor_b32 s0, s18, s19 // (op 2495) 

s_sub_i32 s18, s0, s19 // (op 2495) 


s_add_u32 s50, s8, 256 // (op 2495) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2495) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2495) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2495) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 2495) 

s_mul_i32 s0, s18, s50 // (op 2495) 

s_sub_i32 s49, s1, s0 // (op 2495) 


s_mul_hi_i32 s0, s16, s11 // (op 2495) 
s_add_i32 s1, s0, s16 // (op 2495) 

s_ashr_i32 s0, s1, 31 // (op 2495) 
s_cmp_eq_i32 s11, 0 // (op 2495) 
s_mov_b32 s18, scc // (op 2495) 
s_cmp_eq_i32 s18, 0 // (op 2495) 
s_cselect_b32 s19, 0, -1 // (op 2495) 

s_and_b32 s18, s47, 31 // (op 2495) 
s_lshl_b32 s72, 1, s18 // (op 2495) 
s_add_i32 s73, s72, s19 // (op 2495) 


s_and_b32 s19, s0, s73 // (op 2495) 


s_add_i32 s0, s1, s19 // (op 2495) 


s_ashr_i32 s1, s0, s18 // (op 2495) 


s_lshl_b32 s0, s47, 24 // (op 2495) 
s_ashr_i32 s18, s0, 31 // (op 2495) 

s_xor_b32 s0, s1, s18 // (op 2495) 

s_sub_i32 s1, s0, s18 // (op 2495) 


s_mul_i32 s0, s13, s50 // (op 2495) 

s_mul_i32 s18, s10, s0 // (op 2495) 

s_mul_i32 s0, s1, s18 // (op 2495) 

s_sub_u32 s18, s16, s0 // (op 2495) 

s_mul_hi_i32 s0, s18, s14 // (op 2495) 
s_add_i32 s19, s0, s18 // (op 2495) 


s_ashr_i32 s0, s19, 31 // (op 2495) 
s_cmp_eq_i32 s14, 0 // (op 2495) 
s_mov_b32 s18, scc // (op 2495) 
s_cmp_eq_i32 s18, 0 // (op 2495) 
s_cselect_b32 s50, 0, -1 // (op 2495) 

s_and_b32 s18, s46, 31 // (op 2495) 
s_lshl_b32 s51, 1, s18 // (op 2495) 
s_add_i32 s72, s51, s50 // (op 2495) 


s_and_b32 s50, s0, s72 // (op 2495) 


s_add_i32 s0, s19, s50 // (op 2495) 


s_ashr_i32 s19, s0, s18 // (op 2495) 


s_lshl_b32 s0, s46, 24 // (op 2495) 
s_ashr_i32 s18, s0, 31 // (op 2495) 

s_xor_b32 s0, s19, s18 // (op 2495) 

s_sub_i32 s19, s0, s18 // (op 2495) 


s_cmp_eq_i32 s1, 0 // (op 2495) 
s_mov_b32 s0, scc // (op 2495) 

s_cmp_eq_i32 s0, 0 // (op 2495) 
s_cselect_b32 s1, s19, s49 // (op 2495) 



v_lshl_add_u32 v21, s1, 8, v19 // (op 2495) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2495) 
v_mov_b32 v15, 0 // (op 2495) convert
v_mov_b32 v14, v21 // (op 2495) convert

s_lshr_b32 s18, s0, 1 // (op 2495) 

v_lshl_add_u32 v19, s18, 8, v63 // (op 2495) 


v_mul_lo_u32 v17, v19, s31 // (op 2495) most significant: low of low * high
v_mul_hi_u32 v21, v19, s30 // (op 2495) most significant: high of low * low
v_mul_lo_u32 v16, v19, s30 // (op 2495) least significant: low of low * low
v_add_u32 v17, v17, v21 // (op 2495) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 2495) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 2495) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 2495) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 2495) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 2495) least significant: low of low * low
v_add_u32 v15, v15, v19 // (op 2495) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 2495) 

v_mov_b32 v19, v16 // (op 2495) convert

s_mul_i32 s1, s29, 6 // (op 2497) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 2497) least significant
s_mul_hi_u32 s18, s28, 6 // (op 2497) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2497) most significant: sum

s_mul_i32 s19, s30, s1 // (op 2497) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 2497) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 2497) least significant
s_mul_hi_u32 s50, s30, s0 // (op 2497) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 2497) most significant: sum
s_add_u32 s19, s19, s50 // (op 2497) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2497) 

s_mov_b32 s52, s32 // (op 2497) reg expression
s_mov_b32 s53, s33 // (op 2497) reg expression
s_mov_b32 s54, s0 // (op 2497) reg expression
s_mov_b32 s55, 131072 // (op 2497) call()

s_mul_hi_i32 s0, s16, s12 // (op 2559) 
v_and_b32 v21, -64, v1 // (op 2559) 
s_add_i32 s1, s0, s16 // (op 2559) 

v_sub_u32 v63, v1, v21 // (op 2559) 

s_ashr_i32 s0, s1, 31 // (op 2559) 
v_lshrrev_b32 v21, 6, v1 // (op 2559) 
s_cmp_eq_i32 s12, 0 // (op 2559) 
s_mov_b32 s18, scc // (op 2559) 
s_cmp_eq_i32 s18, 0 // (op 2559) 
s_cselect_b32 s19, 0, -1 // (op 2559) 

s_and_b32 s18, s15, 31 // (op 2559) 
s_lshl_b32 s49, 1, s18 // (op 2559) 
s_add_i32 s50, s49, s19 // (op 2559) 


s_and_b32 s19, s0, s50 // (op 2559) 


s_add_i32 s0, s1, s19 // (op 2559) 


s_ashr_i32 s1, s0, s18 // (op 2559) 


s_lshl_b32 s0, s15, 24 // (op 2559) 
s_ashr_i32 s18, s0, 31 // (op 2559) 

s_xor_b32 s0, s1, s18 // (op 2559) 

s_sub_i32 s1, s0, s18 // (op 2559) 


s_mul_hi_i32 s0, s1, s44 // (op 2559) 
s_add_i32 s18, s0, s1 // (op 2559) 

s_ashr_i32 s0, s18, 31 // (op 2559) 
s_cmp_eq_i32 s44, 0 // (op 2559) 
s_mov_b32 s19, scc // (op 2559) 
s_cmp_eq_i32 s19, 0 // (op 2559) 
s_cselect_b32 s49, 0, -1 // (op 2559) 

s_and_b32 s19, s45, 31 // (op 2559) 
s_lshl_b32 s50, 1, s19 // (op 2559) 
s_add_i32 s51, s50, s49 // (op 2559) 


s_and_b32 s49, s0, s51 // (op 2559) 


s_add_i32 s0, s18, s49 // (op 2559) 


s_ashr_i32 s18, s0, s19 // (op 2559) 


s_lshl_b32 s0, s45, 24 // (op 2559) 
s_ashr_i32 s19, s0, 31 // (op 2559) 

s_xor_b32 s0, s18, s19 // (op 2559) 

s_sub_i32 s18, s0, s19 // (op 2559) 


s_mul_i32 s0, s1, s13 // (op 2559) 

s_sub_u32 s1, s16, s0 // (op 2559) 

s_mul_i32 s0, s18, s13 // (op 2559) 
s_add_u32 s0, s0, s1 // (op 2559) 


s_mul_hi_i32 s1, s16, s11 // (op 2559) 
s_add_i32 s18, s1, s16 // (op 2559) 

s_ashr_i32 s1, s18, 31 // (op 2559) 
s_cmp_eq_i32 s11, 0 // (op 2559) 
s_mov_b32 s19, scc // (op 2559) 
s_cmp_eq_i32 s19, 0 // (op 2559) 
s_cselect_b32 s49, 0, -1 // (op 2559) 

s_and_b32 s19, s47, 31 // (op 2559) 
s_lshl_b32 s50, 1, s19 // (op 2559) 
s_add_i32 s51, s50, s49 // (op 2559) 


s_and_b32 s49, s1, s51 // (op 2559) 


s_add_i32 s1, s18, s49 // (op 2559) 


s_ashr_i32 s18, s1, s19 // (op 2559) 


s_lshl_b32 s1, s47, 24 // (op 2559) 
s_ashr_i32 s19, s1, 31 // (op 2559) 

s_xor_b32 s1, s18, s19 // (op 2559) 

s_sub_i32 s18, s1, s19 // (op 2559) 


s_add_u32 s50, s8, 256 // (op 2559) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2559) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2559) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2559) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 2559) 

s_mul_i32 s1, s13, s50 // (op 2559) 

s_mul_i32 s19, s10, s1 // (op 2559) 

s_mul_i32 s1, s18, s19 // (op 2559) 

s_sub_u32 s19, s16, s1 // (op 2559) 

s_mov_b32 s1, s19 // (op 2559) 
s_mul_hi_i32 s49, s1, s14 // (op 2559) 
s_add_i32 s50, s49, s1 // (op 2559) 


s_ashr_i32 s1, s50, 31 // (op 2559) 
s_cmp_eq_i32 s14, 0 // (op 2559) 
s_mov_b32 s49, scc // (op 2559) 
s_cmp_eq_i32 s49, 0 // (op 2559) 
s_cselect_b32 s51, 0, -1 // (op 2559) 

s_and_b32 s49, s46, 31 // (op 2559) 
s_lshl_b32 s72, 1, s49 // (op 2559) 
s_add_i32 s73, s72, s51 // (op 2559) 


s_and_b32 s51, s1, s73 // (op 2559) 


s_add_i32 s1, s50, s51 // (op 2559) 


s_ashr_i32 s50, s1, s49 // (op 2559) 


s_lshl_b32 s1, s46, 24 // (op 2559) 
s_ashr_i32 s49, s1, 31 // (op 2559) 

s_xor_b32 s1, s50, s49 // (op 2559) 

s_sub_i32 s50, s1, s49 // (op 2559) 


s_mul_i32 s1, s48, s13 // (op 2559) 
s_add_u32 s72, s6, 64 // (op 2559) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 2559) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 2559) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 2559) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 2559) 

s_sub_i32 s49, s72, s1 // (op 2559) 

s_mul_i32 s51, s50, s49 // (op 2559) 


s_sub_u32 s49, s19, s51 // (op 2559) 


s_add_u32 s19, s49, s1 // (op 2559) 


s_cmp_eq_i32 s18, 0 // (op 2559) 
s_mov_b32 s1, scc // (op 2559) 

s_cmp_eq_i32 s1, 0 // (op 2559) 
s_cselect_b32 s18, s19, s0 // (op 2559) 



v_lshl_add_u32 v67, s18, 6, v63 // (op 2559) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2559) 
s_lshr_b32 s18, s0, 1 // (op 2559) 

v_lshl_add_u32 v63, s18, 3, v21 // (op 2559) 


v_mul_lo_u32 v21, v63, s34 // (op 2559) 

v_add_u32 v63, v67, v21 // (op 2559) 


s_mul_i32 s1, s34, s39 // (op 2561) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 2561) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2561) least significant
s_mul_hi_u32 s19, s34, s38 // (op 2561) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2561) most significant: sum
s_add_u32 s1, s1, s19 // (op 2561) most significant: sum


s_mov_b32 s64, s36 // (op 2561) reg expression
s_mov_b32 s65, s37 // (op 2561) reg expression
s_mov_b32 s66, s0 // (op 2561) reg expression
s_mov_b32 s67, 131072 // (op 2561) call()

s_mul_hi_i32 s0, s16, s12 // (op 2623) 
v_and_b32 v21, -64, v1 // (op 2623) 
s_add_i32 s1, s0, s16 // (op 2623) 

v_sub_u32 v67, v1, v21 // (op 2623) 

s_ashr_i32 s0, s1, 31 // (op 2623) 
v_lshlrev_b32 v21, 2, v67 // (op 2623) 

s_cmp_eq_i32 s12, 0 // (op 2623) 
s_mov_b32 s18, scc // (op 2623) 
v_lshrrev_b32 v67, 6, v1 // (op 2623) 
s_cmp_eq_i32 s18, 0 // (op 2623) 
s_cselect_b32 s19, 0, -1 // (op 2623) 

s_and_b32 s18, s15, 31 // (op 2623) 
s_lshl_b32 s49, 1, s18 // (op 2623) 
s_add_i32 s50, s49, s19 // (op 2623) 


s_and_b32 s19, s0, s50 // (op 2623) 


s_add_i32 s0, s1, s19 // (op 2623) 


s_ashr_i32 s1, s0, s18 // (op 2623) 


s_lshl_b32 s0, s15, 24 // (op 2623) 
s_ashr_i32 s18, s0, 31 // (op 2623) 

s_xor_b32 s0, s1, s18 // (op 2623) 

s_sub_i32 s1, s0, s18 // (op 2623) 


s_mul_hi_i32 s0, s1, s44 // (op 2623) 
s_add_i32 s18, s0, s1 // (op 2623) 

s_ashr_i32 s0, s18, 31 // (op 2623) 
s_cmp_eq_i32 s44, 0 // (op 2623) 
s_mov_b32 s19, scc // (op 2623) 
s_cmp_eq_i32 s19, 0 // (op 2623) 
s_cselect_b32 s49, 0, -1 // (op 2623) 

s_and_b32 s19, s45, 31 // (op 2623) 
s_lshl_b32 s50, 1, s19 // (op 2623) 
s_add_i32 s51, s50, s49 // (op 2623) 


s_and_b32 s49, s0, s51 // (op 2623) 


s_add_i32 s0, s18, s49 // (op 2623) 


s_ashr_i32 s18, s0, s19 // (op 2623) 


s_lshl_b32 s0, s45, 24 // (op 2623) 
s_ashr_i32 s19, s0, 31 // (op 2623) 

s_xor_b32 s0, s18, s19 // (op 2623) 

s_sub_i32 s18, s0, s19 // (op 2623) 


s_add_u32 s50, s8, 256 // (op 2623) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 2623) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 2623) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 2623) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 2623) 

s_mul_i32 s0, s18, s50 // (op 2623) 

s_sub_i32 s49, s1, s0 // (op 2623) 


s_mul_hi_i32 s0, s16, s11 // (op 2623) 
s_add_i32 s1, s0, s16 // (op 2623) 

s_ashr_i32 s0, s1, 31 // (op 2623) 
s_cmp_eq_i32 s11, 0 // (op 2623) 
s_mov_b32 s18, scc // (op 2623) 
s_cmp_eq_i32 s18, 0 // (op 2623) 
s_cselect_b32 s19, 0, -1 // (op 2623) 

s_and_b32 s18, s47, 31 // (op 2623) 
s_lshl_b32 s72, 1, s18 // (op 2623) 
s_add_i32 s73, s72, s19 // (op 2623) 


s_and_b32 s19, s0, s73 // (op 2623) 


s_add_i32 s0, s1, s19 // (op 2623) 


s_ashr_i32 s1, s0, s18 // (op 2623) 


s_lshl_b32 s0, s47, 24 // (op 2623) 
s_ashr_i32 s18, s0, 31 // (op 2623) 

s_xor_b32 s0, s1, s18 // (op 2623) 

s_sub_i32 s1, s0, s18 // (op 2623) 


s_mul_i32 s0, s13, s50 // (op 2623) 

s_mul_i32 s18, s10, s0 // (op 2623) 

s_mul_i32 s0, s1, s18 // (op 2623) 

s_sub_u32 s18, s16, s0 // (op 2623) 

s_mul_hi_i32 s0, s18, s14 // (op 2623) 
s_add_i32 s19, s0, s18 // (op 2623) 


s_ashr_i32 s0, s19, 31 // (op 2623) 
s_cmp_eq_i32 s14, 0 // (op 2623) 
s_mov_b32 s18, scc // (op 2623) 
s_cmp_eq_i32 s18, 0 // (op 2623) 
s_cselect_b32 s50, 0, -1 // (op 2623) 

s_and_b32 s18, s46, 31 // (op 2623) 
s_lshl_b32 s51, 1, s18 // (op 2623) 
s_add_i32 s72, s51, s50 // (op 2623) 


s_and_b32 s50, s0, s72 // (op 2623) 


s_add_i32 s0, s19, s50 // (op 2623) 


s_ashr_i32 s19, s0, s18 // (op 2623) 


s_lshl_b32 s0, s46, 24 // (op 2623) 
s_ashr_i32 s18, s0, 31 // (op 2623) 

s_xor_b32 s0, s19, s18 // (op 2623) 

s_sub_i32 s19, s0, s18 // (op 2623) 


s_cmp_eq_i32 s1, 0 // (op 2623) 
s_mov_b32 s0, scc // (op 2623) 

s_cmp_eq_i32 s0, 0 // (op 2623) 
s_cselect_b32 s1, s19, s49 // (op 2623) 



v_lshl_add_u32 v71, s1, 8, v21 // (op 2623) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2623) 
s_lshr_b32 s18, s0, 1 // (op 2623) 

v_lshl_add_u32 v21, s18, 3, v67 // (op 2623) 


v_mul_lo_u32 v67, v21, s40 // (op 2623) 
v_add_u32 v67, v67, v71 // (op 2623) 


s_mul_i32 s1, s40, s39 // (op 2625) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 2625) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 2625) least significant
s_mul_hi_u32 s19, s40, s38 // (op 2625) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2625) most significant: sum
s_add_u32 s1, s1, s19 // (op 2625) most significant: sum


s_mov_b32 s68, s42 // (op 2625) reg expression
s_mov_b32 s69, s43 // (op 2625) reg expression
s_mov_b32 s70, s0 // (op 2625) reg expression
s_mov_b32 s71, 131072 // (op 2625) call()

v_and_b32 v21, -4, v1 // (op 2716) 
v_sub_u32 v71, v1, v21 // (op 2716) 

v_lshlrev_b32 v21, 4, v71 // (op 2716) 

v_lshrrev_b32 v71, 2, v1 // (op 2716) 
v_lshl_add_u32 v75, v71, 6, v21 // (op 2716) 


s_nop 0
v_readfirstlane_b32 s0, v75 // (op 2716) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -4, v1 // (op 2768) 
v_sub_u32 v71, v1, v21 // (op 2768) 

v_lshlrev_b32 v21, 6, v71 // (op 2768) 

v_lshrrev_b32 v71, 2, v1 // (op 2768) 
v_lshl_add_u32 v75, v71, 9, v21 // (op 2768) 


v_mul_lo_u32 v21, v75, 6 // (op 2768) 
v_lshrrev_b32 v71, 3, v21 // (op 2768) 

v_lshrrev_b32 v21, 4, v75 // (op 2768) 

v_lshlrev_b32 v75, 2, v21 // (op 2768) 

v_add_u32 v21, v71, v75 // (op 2768) 


v_and_b32 v71, -64, v1 // (op 2826) 
v_sub_u32 v75, v1, v71 // (op 2826) 

v_and_b32 v71, 4294967232, v1 // (op 2826) 
v_add_u32 v79, v71, v75 // (op 2826) 


v_and_b32 v71, -64, v1 // (op 2879) 
v_sub_u32 v75, v1, v71 // (op 2879) 

v_lshlrev_b32 v71, 2, v75 // (op 2879) 

v_lshrrev_b32 v75, 6, v1 // (op 2879) 
v_lshl_add_u32 v83, v75, 8, v71 // (op 2879) 


s_ashr_i64 s[18:19], s[28:29], 7 // (op 877) 
s_and_b32 s1, s18, 4294967294 // (op 877) 

s_ashr_i64 s[18:19], s[28:29], 7 // (op 876) 
s_mov_b32 s49, s18 // (op 876) convert

s_cmp_lt_u32 s1, s49 // (op 876) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_876 // (op 876) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_876 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_876:
s_barrier  // (op 3120) 
s_add_u32 s1, s1, 1 // (op 878) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 2396) 
v_mov_b64 v[14:15], s[18:19] // (op 2396) convert


v_mov_b64 v[16:17], v[32:33] // (op 2396) 
v_mov_b64 v[22:23], v[16:17] // (op 2396) reg expression
s_mov_b32 m0, s0 // (op 2396) reg expression
buffer_load_dwordx4 v22, s[20:23], 0 offen offset: 0 lds // (op 2396) Load value direct to lds
v_add_co_u32 v22, s[18:19], v22, v14 // (op 2396) least significant half
v_addc_co_u32 v23, s[18:19], 0, v23, s[18:19] // (op 2396) most significant half

s_add_u32 m0, m0, 4096 // (op 2396) 
buffer_load_dwordx4 v22, s[20:23], 0 offen offset: 0 lds // (op 2396) Load value direct to lds



s_lshl_b64 s[18:19], s[24:25], 7 // (op 2415) 
v_mov_b32 v20, s18 // (op 2415) 
v_mov_b32 v14, s19 // (op 2415) 
v_add_co_u32 v32, s[50:51], v32, v20 // (op 2415) least significant half
v_addc_co_u32 v33, s[50:51], v33, v14, s[50:51] // (op 2415) most significant half





s_mul_i32 s19, s31, 6 // (op 225) most significant: low of high * low
s_mul_i32 s18, s30, 6 // (op 225) least significant
s_mul_hi_u32 s49, s30, 6 // (op 225) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 225) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[18:19] // (op 225) 


v_mov_b32 v20, v19 // (op 225) 
v_mov_b32 v16, v20 // (op 225) reg expression
buffer_load_dwordx3 v[22:24], v16, s[52:55], 0 offen // (op 225) Load value
buffer_load_dwordx3 v[26:28], v16, s[52:55], 0 offen offset:12 // (op 225) Load value
buffer_load_dwordx3 v[34:36], v16, s[52:55], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx3 v[38:40], v16, s[52:55], 0 offen offset:36 // (op 225) Load value
v_add_u32 v16, v16, v14 // (op 225) 
buffer_load_dwordx3 v[42:44], v16, s[52:55], 0 offen // (op 225) Load value
buffer_load_dwordx3 v[46:48], v16, s[52:55], 0 offen offset:12 // (op 225) Load value
buffer_load_dwordx3 v[50:52], v16, s[52:55], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx3 v[54:56], v16, s[52:55], 0 offen offset:36 // (op 225) Load value



s_lshl_b64 s[18:19], s[30:31], 7 // (op 2487) 
s_mul_i32 s51, s19, 6 // (op 2487) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 2487) least significant
s_mul_hi_u32 s49, s18, 6 // (op 2487) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 2487) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 2487) 

v_add_u32 v19, v19, s18 // (op 2487) 



s_barrier  // (op 3139) 
v_mov_b32 v20, v63 // (op 226) 



v_mov_b32 v25, v20 // (op 226) reg expression

buffer_load_ubyte v37, v25, s[64:67], 0 offen offset: 0 // (op 226) Load value


s_lshl_b32 s18, s34, 2 // (op 2551) 
v_add_u32 v63, v63, s18 // (op 2551) 

s_waitcnt vmcnt(8)
ds_write_b96 v21, v[22:24] offset:16384 // (op 222) Store local data 


v_mov_b32 v20, v67 // (op 227) 
v_mov_b32 v25, v20 // (op 227) reg expression

buffer_load_dword v41, v25, s[68:71], 0 offen // (op 227) Load value


s_lshl_b32 s18, s40, 2 // (op 2615) 
v_add_u32 v67, v67, s18 // (op 2615) 

s_waitcnt vmcnt(8)
ds_write_b96 v21, v[26:28] offset:16400 // (op 222) Store local data 
s_barrier  // (op 3129) 
s_waitcnt vmcnt(7)
ds_write_b96 v21, v[34:36] offset:16416 // (op 222) Store local data 



ds_read_b64_tr_b8 v[68:69], v3 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v3 offset:512 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v3 offset:2048 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v3 offset:2560 // (op 205) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b96 v21, v[38:40] offset:16432 // (op 222) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v21, v[42:44] offset:16640 // (op 222) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v21, v[46:48] offset:16656 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v21, v[50:52] offset:16672 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v21, v[54:56] offset:16688 // (op 222) Store local data 

s_barrier  // (op 3160) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3150) 






ds_read_b96_tr_b6 v[22:24], v9 offset:16384 // (op 207) Transpose load local data 
ds_read_b96_tr_b6 v[25:27], v9 offset:20480 // (op 207) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v79, v37 offset:50176 // (op 223) Store local data 

s_barrier  // (op 3180) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3170) 





ds_read_u8 v20, v0 offset:50176 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v83, v41 offset:49152 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3190) 


ds_read_u8 v14, v6 offset:49152 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[68:75], v[22:27], a[0:15], v20, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 212) 





ds_read_b64_tr_b8 v[22:23], v3 offset:4096 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v3 offset:4608 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v3 offset:6144 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v3 offset:6656 // (op 214) Transpose load local data 



ds_read_b96_tr_b6 v[34:36], v9 offset:32768 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v9 offset:36864 // (op 216) Transpose load local data 


ds_read_u8 v14, v0 offset:50304 // (op 218) Load local data 


ds_read_u8 v15, v6 offset:49664 // (op 220) Load local data 



ds_read_b96_tr_b6 v[40:42], v9 offset:16448 // (op 416) Transpose load local data 
ds_read_b96_tr_b6 v[43:45], v9 offset:20544 // (op 416) Transpose load local data 


ds_read_u8 v16, v6 offset:49216 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[68:75], v[40:45], a[16:31], v20, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 421) 





ds_read_b96_tr_b6 v[40:42], v9 offset:32832 // (op 425) Transpose load local data 
ds_read_b96_tr_b6 v[43:45], v9 offset:36928 // (op 425) Transpose load local data 


ds_read_u8 v16, v6 offset:49728 // (op 429) Load local data 



ds_read_b96_tr_b6 v[46:48], v9 offset:16512 // (op 650) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v9 offset:20608 // (op 650) Transpose load local data 


ds_read_u8 v17, v6 offset:49280 // (op 654) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[68:75], v[46:51], a[32:47], v20, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 655) 





ds_read_b96_tr_b6 v[46:48], v9 offset:32896 // (op 659) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v9 offset:36992 // (op 659) Transpose load local data 


ds_read_u8 v17, v6 offset:49792 // (op 663) Load local data 



ds_read_b96_tr_b6 v[52:54], v9 offset:16576 // (op 884) Transpose load local data 
ds_read_b96_tr_b6 v[55:57], v9 offset:20672 // (op 884) Transpose load local data 


ds_read_u8 v18, v6 offset:49344 // (op 888) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[68:75], v[52:57], a[48:63], v20, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 889) 







ds_read_b96_tr_b6 v[52:54], v9 offset:32960 // (op 893) Transpose load local data 
ds_read_b96_tr_b6 v[55:57], v9 offset:37056 // (op 893) Transpose load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[22:29], v[34:39], a[0:15], v14, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 221) 




ds_read_u8 v15, v6 offset:49856 // (op 897) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[22:29], v[40:45], a[16:31], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 430) 


v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[22:29], v[46:51], a[32:47], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 664) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[22:29], v[52:57], a[48:63], v14, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 898) 




s_ashr_i64 s[18:19], s[28:29], 7 // (op 876) 
s_mov_b32 s49, s18 // (op 876) convert

s_cmp_lt_u32 s1, s49 // (op 876) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_876 // (op 876) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_876 if true)
RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_876:























s_barrier  // (op 3303) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_accvgpr_read v16, a4 // (op 50) 
v_accvgpr_read v17, a5 // (op 50) 
v_accvgpr_read v18, a6 // (op 50) 
v_accvgpr_read v19, a7 // (op 50) 
v_accvgpr_read v20, a8 // (op 50) 
v_accvgpr_read v21, a9 // (op 50) 
v_accvgpr_read v22, a10 // (op 50) 
v_accvgpr_read v23, a11 // (op 50) 
v_accvgpr_read v24, a12 // (op 50) 
v_accvgpr_read v25, a13 // (op 50) 
v_accvgpr_read v26, a14 // (op 50) 
v_accvgpr_read v27, a15 // (op 50) 
v_mul_f32 v28, v10, v12 // (op 50) 
v_mul_f32 v29, v10, v13 // (op 50) 
v_mul_f32 v30, v10, v14 // (op 50) 
v_mul_f32 v31, v10, v15 // (op 50) 
v_mul_f32 v32, v10, v16 // (op 50) 
v_mul_f32 v33, v10, v17 // (op 50) 
v_mul_f32 v34, v10, v18 // (op 50) 
v_mul_f32 v35, v10, v19 // (op 50) 
v_mul_f32 v36, v10, v20 // (op 50) 
v_mul_f32 v37, v10, v21 // (op 50) 
v_mul_f32 v38, v10, v22 // (op 50) 
v_mul_f32 v39, v10, v23 // (op 50) 
v_mul_f32 v40, v10, v24 // (op 50) 
v_mul_f32 v41, v10, v25 // (op 50) 
v_mul_f32 v42, v10, v26 // (op 50) 
v_mul_f32 v43, v10, v27 // (op 50) 

v_lshrrev_b32 v12, 16, v28 // (op 56) 
v_lshrrev_b32 v13, 16, v29 // (op 56) 
v_lshrrev_b32 v14, 16, v30 // (op 56) 
v_lshrrev_b32 v15, 16, v31 // (op 56) 
v_lshrrev_b32 v16, 16, v32 // (op 56) 
v_lshrrev_b32 v17, 16, v33 // (op 56) 
v_lshrrev_b32 v18, 16, v34 // (op 56) 
v_lshrrev_b32 v19, 16, v35 // (op 56) 
v_lshrrev_b32 v20, 16, v36 // (op 56) 
v_lshrrev_b32 v21, 16, v37 // (op 56) 
v_lshrrev_b32 v22, 16, v38 // (op 56) 
v_lshrrev_b32 v23, 16, v39 // (op 56) 
v_lshrrev_b32 v24, 16, v40 // (op 56) 
v_lshrrev_b32 v25, 16, v41 // (op 56) 
v_lshrrev_b32 v26, 16, v42 // (op 56) 
v_lshrrev_b32 v27, 16, v43 // (op 56) 


v_pack_b32_f16 v28, v12, v13 // (op 58) 
v_pack_b32_f16 v29, v14, v15 // (op 58) 
v_pack_b32_f16 v30, v16, v17 // (op 58) 
v_pack_b32_f16 v31, v18, v19 // (op 58) 
v_pack_b32_f16 v32, v20, v21 // (op 58) 
v_pack_b32_f16 v33, v22, v23 // (op 58) 
v_pack_b32_f16 v34, v24, v25 // (op 58) 
v_pack_b32_f16 v35, v26, v27 // (op 58) 

v_mov_b32 v0, v11 // (op 58) reg expression

buffer_store_dwordx2 v[28:29], v0, s[60:63], 0 offen // (op 58) Store value
buffer_store_dwordx2 v[30:31], v0, s[60:63], 0 offen offset:16 // (op 58) Store value
buffer_store_dwordx2 v[32:33], v0, s[60:63], 0 offen offset:32 // (op 58) Store value
buffer_store_dwordx2 v[34:35], v0, s[60:63], 0 offen offset:48 // (op 58) Store value



v_accvgpr_read v12, a16 // (op 1262) 
v_accvgpr_read v13, a17 // (op 1262) 
v_accvgpr_read v14, a18 // (op 1262) 
v_accvgpr_read v15, a19 // (op 1262) 
v_accvgpr_read v16, a20 // (op 1262) 
v_accvgpr_read v17, a21 // (op 1262) 
v_accvgpr_read v18, a22 // (op 1262) 
v_accvgpr_read v19, a23 // (op 1262) 
v_accvgpr_read v20, a24 // (op 1262) 
v_accvgpr_read v21, a25 // (op 1262) 
v_accvgpr_read v22, a26 // (op 1262) 
v_accvgpr_read v23, a27 // (op 1262) 
v_accvgpr_read v24, a28 // (op 1262) 
v_accvgpr_read v25, a29 // (op 1262) 
v_accvgpr_read v26, a30 // (op 1262) 
v_accvgpr_read v27, a31 // (op 1262) 
v_mul_f32 v28, v10, v12 // (op 1262) 
v_mul_f32 v29, v10, v13 // (op 1262) 
v_mul_f32 v30, v10, v14 // (op 1262) 
v_mul_f32 v31, v10, v15 // (op 1262) 
v_mul_f32 v32, v10, v16 // (op 1262) 
v_mul_f32 v33, v10, v17 // (op 1262) 
v_mul_f32 v34, v10, v18 // (op 1262) 
v_mul_f32 v35, v10, v19 // (op 1262) 
v_mul_f32 v36, v10, v20 // (op 1262) 
v_mul_f32 v37, v10, v21 // (op 1262) 
v_mul_f32 v38, v10, v22 // (op 1262) 
v_mul_f32 v39, v10, v23 // (op 1262) 
v_mul_f32 v40, v10, v24 // (op 1262) 
v_mul_f32 v41, v10, v25 // (op 1262) 
v_mul_f32 v42, v10, v26 // (op 1262) 
v_mul_f32 v43, v10, v27 // (op 1262) 

v_lshrrev_b32 v12, 16, v28 // (op 1264) 
v_lshrrev_b32 v13, 16, v29 // (op 1264) 
v_lshrrev_b32 v14, 16, v30 // (op 1264) 
v_lshrrev_b32 v15, 16, v31 // (op 1264) 
v_lshrrev_b32 v16, 16, v32 // (op 1264) 
v_lshrrev_b32 v17, 16, v33 // (op 1264) 
v_lshrrev_b32 v18, 16, v34 // (op 1264) 
v_lshrrev_b32 v19, 16, v35 // (op 1264) 
v_lshrrev_b32 v20, 16, v36 // (op 1264) 
v_lshrrev_b32 v21, 16, v37 // (op 1264) 
v_lshrrev_b32 v22, 16, v38 // (op 1264) 
v_lshrrev_b32 v23, 16, v39 // (op 1264) 
v_lshrrev_b32 v24, 16, v40 // (op 1264) 
v_lshrrev_b32 v25, 16, v41 // (op 1264) 
v_lshrrev_b32 v26, 16, v42 // (op 1264) 
v_lshrrev_b32 v27, 16, v43 // (op 1264) 


v_pack_b32_f16 v28, v12, v13 // (op 1265) 
v_pack_b32_f16 v29, v14, v15 // (op 1265) 
v_pack_b32_f16 v30, v16, v17 // (op 1265) 
v_pack_b32_f16 v31, v18, v19 // (op 1265) 
v_pack_b32_f16 v32, v20, v21 // (op 1265) 
v_pack_b32_f16 v33, v22, v23 // (op 1265) 
v_pack_b32_f16 v34, v24, v25 // (op 1265) 
v_pack_b32_f16 v35, v26, v27 // (op 1265) 

s_lshl_b32 s0, s4, 7 // (op 1265) 
v_add_u32 v0, v11, s0 // (op 1265) 


buffer_store_dwordx2 v[28:29], v0, s[60:63], 0 offen // (op 1265) Store value
buffer_store_dwordx2 v[30:31], v0, s[60:63], 0 offen offset:16 // (op 1265) Store value
buffer_store_dwordx2 v[32:33], v0, s[60:63], 0 offen offset:32 // (op 1265) Store value
buffer_store_dwordx2 v[34:35], v0, s[60:63], 0 offen offset:48 // (op 1265) Store value



v_accvgpr_read v12, a32 // (op 1276) 
v_accvgpr_read v13, a33 // (op 1276) 
v_accvgpr_read v14, a34 // (op 1276) 
v_accvgpr_read v15, a35 // (op 1276) 
v_accvgpr_read v16, a36 // (op 1276) 
v_accvgpr_read v17, a37 // (op 1276) 
v_accvgpr_read v18, a38 // (op 1276) 
v_accvgpr_read v19, a39 // (op 1276) 
v_accvgpr_read v20, a40 // (op 1276) 
v_accvgpr_read v21, a41 // (op 1276) 
v_accvgpr_read v22, a42 // (op 1276) 
v_accvgpr_read v23, a43 // (op 1276) 
v_accvgpr_read v24, a44 // (op 1276) 
v_accvgpr_read v25, a45 // (op 1276) 
v_accvgpr_read v26, a46 // (op 1276) 
v_accvgpr_read v27, a47 // (op 1276) 
v_mul_f32 v28, v10, v12 // (op 1276) 
v_mul_f32 v29, v10, v13 // (op 1276) 
v_mul_f32 v30, v10, v14 // (op 1276) 
v_mul_f32 v31, v10, v15 // (op 1276) 
v_mul_f32 v32, v10, v16 // (op 1276) 
v_mul_f32 v33, v10, v17 // (op 1276) 
v_mul_f32 v34, v10, v18 // (op 1276) 
v_mul_f32 v35, v10, v19 // (op 1276) 
v_mul_f32 v36, v10, v20 // (op 1276) 
v_mul_f32 v37, v10, v21 // (op 1276) 
v_mul_f32 v38, v10, v22 // (op 1276) 
v_mul_f32 v39, v10, v23 // (op 1276) 
v_mul_f32 v40, v10, v24 // (op 1276) 
v_mul_f32 v41, v10, v25 // (op 1276) 
v_mul_f32 v42, v10, v26 // (op 1276) 
v_mul_f32 v43, v10, v27 // (op 1276) 

v_lshrrev_b32 v12, 16, v28 // (op 1278) 
v_lshrrev_b32 v13, 16, v29 // (op 1278) 
v_lshrrev_b32 v14, 16, v30 // (op 1278) 
v_lshrrev_b32 v15, 16, v31 // (op 1278) 
v_lshrrev_b32 v16, 16, v32 // (op 1278) 
v_lshrrev_b32 v17, 16, v33 // (op 1278) 
v_lshrrev_b32 v18, 16, v34 // (op 1278) 
v_lshrrev_b32 v19, 16, v35 // (op 1278) 
v_lshrrev_b32 v20, 16, v36 // (op 1278) 
v_lshrrev_b32 v21, 16, v37 // (op 1278) 
v_lshrrev_b32 v22, 16, v38 // (op 1278) 
v_lshrrev_b32 v23, 16, v39 // (op 1278) 
v_lshrrev_b32 v24, 16, v40 // (op 1278) 
v_lshrrev_b32 v25, 16, v41 // (op 1278) 
v_lshrrev_b32 v26, 16, v42 // (op 1278) 
v_lshrrev_b32 v27, 16, v43 // (op 1278) 


v_pack_b32_f16 v28, v12, v13 // (op 1279) 
v_pack_b32_f16 v29, v14, v15 // (op 1279) 
v_pack_b32_f16 v30, v16, v17 // (op 1279) 
v_pack_b32_f16 v31, v18, v19 // (op 1279) 
v_pack_b32_f16 v32, v20, v21 // (op 1279) 
v_pack_b32_f16 v33, v22, v23 // (op 1279) 
v_pack_b32_f16 v34, v24, v25 // (op 1279) 
v_pack_b32_f16 v35, v26, v27 // (op 1279) 

s_lshl_b32 s0, s4, 8 // (op 1279) 
v_add_u32 v0, v11, s0 // (op 1279) 


buffer_store_dwordx2 v[28:29], v0, s[60:63], 0 offen // (op 1279) Store value
buffer_store_dwordx2 v[30:31], v0, s[60:63], 0 offen offset:16 // (op 1279) Store value
buffer_store_dwordx2 v[32:33], v0, s[60:63], 0 offen offset:32 // (op 1279) Store value
buffer_store_dwordx2 v[34:35], v0, s[60:63], 0 offen offset:48 // (op 1279) Store value



v_accvgpr_read v12, a48 // (op 1290) 
v_accvgpr_read v13, a49 // (op 1290) 
v_accvgpr_read v14, a50 // (op 1290) 
v_accvgpr_read v15, a51 // (op 1290) 
v_accvgpr_read v16, a52 // (op 1290) 
v_accvgpr_read v17, a53 // (op 1290) 
v_accvgpr_read v18, a54 // (op 1290) 
v_accvgpr_read v19, a55 // (op 1290) 
v_accvgpr_read v20, a56 // (op 1290) 
v_accvgpr_read v21, a57 // (op 1290) 
v_accvgpr_read v22, a58 // (op 1290) 
v_accvgpr_read v23, a59 // (op 1290) 
v_accvgpr_read v24, a60 // (op 1290) 
v_accvgpr_read v25, a61 // (op 1290) 
v_accvgpr_read v26, a62 // (op 1290) 
v_accvgpr_read v27, a63 // (op 1290) 
v_mul_f32 v28, v10, v12 // (op 1290) 
v_mul_f32 v29, v10, v13 // (op 1290) 
v_mul_f32 v30, v10, v14 // (op 1290) 
v_mul_f32 v31, v10, v15 // (op 1290) 
v_mul_f32 v32, v10, v16 // (op 1290) 
v_mul_f32 v33, v10, v17 // (op 1290) 
v_mul_f32 v34, v10, v18 // (op 1290) 
v_mul_f32 v35, v10, v19 // (op 1290) 
v_mul_f32 v36, v10, v20 // (op 1290) 
v_mul_f32 v37, v10, v21 // (op 1290) 
v_mul_f32 v38, v10, v22 // (op 1290) 
v_mul_f32 v39, v10, v23 // (op 1290) 
v_mul_f32 v40, v10, v24 // (op 1290) 
v_mul_f32 v41, v10, v25 // (op 1290) 
v_mul_f32 v42, v10, v26 // (op 1290) 
v_mul_f32 v43, v10, v27 // (op 1290) 

v_lshrrev_b32 v12, 16, v28 // (op 1292) 
v_lshrrev_b32 v13, 16, v29 // (op 1292) 
v_lshrrev_b32 v14, 16, v30 // (op 1292) 
v_lshrrev_b32 v15, 16, v31 // (op 1292) 
v_lshrrev_b32 v16, 16, v32 // (op 1292) 
v_lshrrev_b32 v17, 16, v33 // (op 1292) 
v_lshrrev_b32 v18, 16, v34 // (op 1292) 
v_lshrrev_b32 v19, 16, v35 // (op 1292) 
v_lshrrev_b32 v20, 16, v36 // (op 1292) 
v_lshrrev_b32 v21, 16, v37 // (op 1292) 
v_lshrrev_b32 v22, 16, v38 // (op 1292) 
v_lshrrev_b32 v23, 16, v39 // (op 1292) 
v_lshrrev_b32 v24, 16, v40 // (op 1292) 
v_lshrrev_b32 v25, 16, v41 // (op 1292) 
v_lshrrev_b32 v26, 16, v42 // (op 1292) 
v_lshrrev_b32 v27, 16, v43 // (op 1292) 


v_pack_b32_f16 v28, v12, v13 // (op 1293) 
v_pack_b32_f16 v29, v14, v15 // (op 1293) 
v_pack_b32_f16 v30, v16, v17 // (op 1293) 
v_pack_b32_f16 v31, v18, v19 // (op 1293) 
v_pack_b32_f16 v32, v20, v21 // (op 1293) 
v_pack_b32_f16 v33, v22, v23 // (op 1293) 
v_pack_b32_f16 v34, v24, v25 // (op 1293) 
v_pack_b32_f16 v35, v26, v27 // (op 1293) 

s_lshl_b32 s0, s4, 7 // (op 1293) 
s_mul_i32 s1, 3, s0 // (op 1293) 

v_add_u32 v0, v11, s1 // (op 1293) 


buffer_store_dwordx2 v[28:29], v0, s[60:63], 0 offen // (op 1293) Store value
buffer_store_dwordx2 v[30:31], v0, s[60:63], 0 offen offset:16 // (op 1293) Store value
buffer_store_dwordx2 v[32:33], v0, s[60:63], 0 offen offset:32 // (op 1293) Store value
buffer_store_dwordx2 v[34:35], v0, s[60:63], 0 offen offset:48 // (op 1293) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_157:


v_mov_b32 v0, v8 // (op 170) reg expression

buffer_load_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 170) Load value
buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen offset:16 // (op 170) Load value
buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen offset:32 // (op 170) Load value
v_accvgpr_read v20, a0 // (op 166) 
v_accvgpr_read v21, a1 // (op 166) 
v_accvgpr_read v22, a2 // (op 166) 
v_accvgpr_read v23, a3 // (op 166) 
v_accvgpr_read v24, a4 // (op 166) 
v_accvgpr_read v25, a5 // (op 166) 
v_accvgpr_read v26, a6 // (op 166) 
v_accvgpr_read v27, a7 // (op 166) 
v_accvgpr_read v28, a8 // (op 166) 
v_accvgpr_read v29, a9 // (op 166) 
v_accvgpr_read v30, a10 // (op 166) 
v_accvgpr_read v31, a11 // (op 166) 
v_accvgpr_read v32, a12 // (op 166) 
v_accvgpr_read v33, a13 // (op 166) 
v_accvgpr_read v34, a14 // (op 166) 
v_accvgpr_read v35, a15 // (op 166) 
v_mul_f32 v36, v10, v20 // (op 166) 
v_mul_f32 v37, v10, v21 // (op 166) 
v_mul_f32 v38, v10, v22 // (op 166) 
v_mul_f32 v39, v10, v23 // (op 166) 
v_mul_f32 v40, v10, v24 // (op 166) 
v_mul_f32 v41, v10, v25 // (op 166) 
v_mul_f32 v42, v10, v26 // (op 166) 
v_mul_f32 v43, v10, v27 // (op 166) 
v_mul_f32 v44, v10, v28 // (op 166) 
v_mul_f32 v45, v10, v29 // (op 166) 
v_mul_f32 v46, v10, v30 // (op 166) 
v_mul_f32 v47, v10, v31 // (op 166) 
v_mul_f32 v48, v10, v32 // (op 166) 
v_mul_f32 v49, v10, v33 // (op 166) 
v_mul_f32 v50, v10, v34 // (op 166) 
v_mul_f32 v51, v10, v35 // (op 166) 

buffer_load_dwordx2 v[18:19], v0, s[56:59], 0 offen offset:48 // (op 170) Load value

s_waitcnt vmcnt(3)
v_and_b32 v20, 65535, v12 // (op 171) 
v_lshrrev_b32 v21, 16, v12 // (op 171) 
v_lshlrev_b32 v20, 16, v20 // (op 171) 
v_lshlrev_b32 v21, 16, v21 // (op 171) 
v_mul_f32 v52, v7, v20 // (op 171) 
v_mul_f32 v53, v7, v21 // (op 171) 
v_and_b32 v20, 65535, v13 // (op 171) 
v_lshrrev_b32 v21, 16, v13 // (op 171) 
v_lshlrev_b32 v20, 16, v20 // (op 171) 
v_lshlrev_b32 v21, 16, v21 // (op 171) 
v_mul_f32 v54, v7, v20 // (op 171) 
v_mul_f32 v55, v7, v21 // (op 171) 
s_waitcnt vmcnt(2)
v_and_b32 v20, 65535, v14 // (op 171) 
v_lshrrev_b32 v21, 16, v14 // (op 171) 
v_lshlrev_b32 v20, 16, v20 // (op 171) 
v_lshlrev_b32 v21, 16, v21 // (op 171) 
v_mul_f32 v56, v7, v20 // (op 171) 
v_mul_f32 v57, v7, v21 // (op 171) 
v_and_b32 v20, 65535, v15 // (op 171) 
v_lshrrev_b32 v21, 16, v15 // (op 171) 
v_lshlrev_b32 v20, 16, v20 // (op 171) 
v_lshlrev_b32 v21, 16, v21 // (op 171) 
v_mul_f32 v58, v7, v20 // (op 171) 
v_mul_f32 v59, v7, v21 // (op 171) 
s_waitcnt vmcnt(1)
v_and_b32 v20, 65535, v16 // (op 171) 
v_lshrrev_b32 v21, 16, v16 // (op 171) 
v_lshlrev_b32 v20, 16, v20 // (op 171) 
v_lshlrev_b32 v21, 16, v21 // (op 171) 
v_mul_f32 v60, v7, v20 // (op 171) 
v_mul_f32 v61, v7, v21 // (op 171) 
v_and_b32 v20, 65535, v17 // (op 171) 
v_lshrrev_b32 v21, 16, v17 // (op 171) 
v_lshlrev_b32 v20, 16, v20 // (op 171) 
v_lshlrev_b32 v21, 16, v21 // (op 171) 
v_mul_f32 v62, v7, v20 // (op 171) 
v_mul_f32 v63, v7, v21 // (op 171) 
s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v18 // (op 171) 
v_lshrrev_b32 v21, 16, v18 // (op 171) 
v_lshlrev_b32 v20, 16, v20 // (op 171) 
v_lshlrev_b32 v21, 16, v21 // (op 171) 
v_mul_f32 v64, v7, v20 // (op 171) 
v_mul_f32 v65, v7, v21 // (op 171) 
v_and_b32 v20, 65535, v19 // (op 171) 
v_lshrrev_b32 v21, 16, v19 // (op 171) 
v_lshlrev_b32 v20, 16, v20 // (op 171) 
v_lshlrev_b32 v21, 16, v21 // (op 171) 
v_mul_f32 v66, v7, v20 // (op 171) 
v_mul_f32 v67, v7, v21 // (op 171) 


v_add_f32 v12, v52, v36 // (op 167) 
v_add_f32 v13, v53, v37 // (op 167) 
v_add_f32 v14, v54, v38 // (op 167) 
v_add_f32 v15, v55, v39 // (op 167) 
v_add_f32 v16, v56, v40 // (op 167) 
v_add_f32 v17, v57, v41 // (op 167) 
v_add_f32 v18, v58, v42 // (op 167) 
v_add_f32 v19, v59, v43 // (op 167) 
v_add_f32 v20, v60, v44 // (op 167) 
v_add_f32 v21, v61, v45 // (op 167) 
v_add_f32 v22, v62, v46 // (op 167) 
v_add_f32 v23, v63, v47 // (op 167) 
v_add_f32 v24, v64, v48 // (op 167) 
v_add_f32 v25, v65, v49 // (op 167) 
v_add_f32 v26, v66, v50 // (op 167) 
v_add_f32 v27, v67, v51 // (op 167) 


v_lshrrev_b32 v28, 16, v12 // (op 168) 
v_lshrrev_b32 v29, 16, v13 // (op 168) 
v_lshrrev_b32 v30, 16, v14 // (op 168) 
v_lshrrev_b32 v31, 16, v15 // (op 168) 
v_lshrrev_b32 v32, 16, v16 // (op 168) 
v_lshrrev_b32 v33, 16, v17 // (op 168) 
v_lshrrev_b32 v34, 16, v18 // (op 168) 
v_lshrrev_b32 v35, 16, v19 // (op 168) 
v_lshrrev_b32 v36, 16, v20 // (op 168) 
v_lshrrev_b32 v37, 16, v21 // (op 168) 
v_lshrrev_b32 v38, 16, v22 // (op 168) 
v_lshrrev_b32 v39, 16, v23 // (op 168) 
v_lshrrev_b32 v40, 16, v24 // (op 168) 
v_lshrrev_b32 v41, 16, v25 // (op 168) 
v_lshrrev_b32 v42, 16, v26 // (op 168) 
v_lshrrev_b32 v43, 16, v27 // (op 168) 


v_pack_b32_f16 v12, v28, v29 // (op 169) 
v_pack_b32_f16 v13, v30, v31 // (op 169) 
v_pack_b32_f16 v14, v32, v33 // (op 169) 
v_pack_b32_f16 v15, v34, v35 // (op 169) 
v_pack_b32_f16 v16, v36, v37 // (op 169) 
v_pack_b32_f16 v17, v38, v39 // (op 169) 
v_pack_b32_f16 v18, v40, v41 // (op 169) 
v_pack_b32_f16 v19, v42, v43 // (op 169) 

v_mov_b32 v0, v11 // (op 169) reg expression

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 169) Store value
buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen offset:16 // (op 169) Store value
buffer_store_dwordx2 v[16:17], v0, s[60:63], 0 offen offset:32 // (op 169) Store value
buffer_store_dwordx2 v[18:19], v0, s[60:63], 0 offen offset:48 // (op 169) Store value





s_lshl_b32 s0, s2, 7 // (op 1327) 
v_add_u32 v0, v8, s0 // (op 1327) 


buffer_load_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 1327) Load value
buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen offset:16 // (op 1327) Load value
v_accvgpr_read v20, a16 // (op 1323) 
v_accvgpr_read v21, a17 // (op 1323) 
v_accvgpr_read v22, a18 // (op 1323) 
v_accvgpr_read v23, a19 // (op 1323) 
v_accvgpr_read v24, a20 // (op 1323) 
v_accvgpr_read v25, a21 // (op 1323) 
v_accvgpr_read v26, a22 // (op 1323) 
v_accvgpr_read v27, a23 // (op 1323) 
v_accvgpr_read v28, a24 // (op 1323) 
v_accvgpr_read v29, a25 // (op 1323) 
v_accvgpr_read v30, a26 // (op 1323) 
v_accvgpr_read v31, a27 // (op 1323) 
v_accvgpr_read v32, a28 // (op 1323) 
v_accvgpr_read v33, a29 // (op 1323) 
v_accvgpr_read v34, a30 // (op 1323) 
v_accvgpr_read v35, a31 // (op 1323) 
v_mul_f32 v36, v10, v20 // (op 1323) 
v_mul_f32 v37, v10, v21 // (op 1323) 
v_mul_f32 v38, v10, v22 // (op 1323) 
v_mul_f32 v39, v10, v23 // (op 1323) 
v_mul_f32 v40, v10, v24 // (op 1323) 
v_mul_f32 v41, v10, v25 // (op 1323) 
v_mul_f32 v42, v10, v26 // (op 1323) 
v_mul_f32 v43, v10, v27 // (op 1323) 
v_mul_f32 v44, v10, v28 // (op 1323) 
v_mul_f32 v45, v10, v29 // (op 1323) 
v_mul_f32 v46, v10, v30 // (op 1323) 
v_mul_f32 v47, v10, v31 // (op 1323) 
v_mul_f32 v48, v10, v32 // (op 1323) 
v_mul_f32 v49, v10, v33 // (op 1323) 
v_mul_f32 v50, v10, v34 // (op 1323) 
v_mul_f32 v51, v10, v35 // (op 1323) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen offset:32 // (op 1327) Load value
buffer_load_dwordx2 v[18:19], v0, s[56:59], 0 offen offset:48 // (op 1327) Load value

s_waitcnt vmcnt(3)
v_and_b32 v20, 65535, v12 // (op 1328) 
v_lshrrev_b32 v21, 16, v12 // (op 1328) 
v_lshlrev_b32 v20, 16, v20 // (op 1328) 
v_lshlrev_b32 v21, 16, v21 // (op 1328) 
v_mul_f32 v52, v7, v20 // (op 1328) 
v_mul_f32 v53, v7, v21 // (op 1328) 
v_and_b32 v20, 65535, v13 // (op 1328) 
v_lshrrev_b32 v21, 16, v13 // (op 1328) 
v_lshlrev_b32 v20, 16, v20 // (op 1328) 
v_lshlrev_b32 v21, 16, v21 // (op 1328) 
v_mul_f32 v54, v7, v20 // (op 1328) 
v_mul_f32 v55, v7, v21 // (op 1328) 
s_waitcnt vmcnt(2)
v_and_b32 v20, 65535, v14 // (op 1328) 
v_lshrrev_b32 v21, 16, v14 // (op 1328) 
v_lshlrev_b32 v20, 16, v20 // (op 1328) 
v_lshlrev_b32 v21, 16, v21 // (op 1328) 
v_mul_f32 v56, v7, v20 // (op 1328) 
v_mul_f32 v57, v7, v21 // (op 1328) 
v_and_b32 v20, 65535, v15 // (op 1328) 
v_lshrrev_b32 v21, 16, v15 // (op 1328) 
v_lshlrev_b32 v20, 16, v20 // (op 1328) 
v_lshlrev_b32 v21, 16, v21 // (op 1328) 
v_mul_f32 v58, v7, v20 // (op 1328) 
v_mul_f32 v59, v7, v21 // (op 1328) 
s_waitcnt vmcnt(1)
v_and_b32 v20, 65535, v16 // (op 1328) 
v_lshrrev_b32 v21, 16, v16 // (op 1328) 
v_lshlrev_b32 v20, 16, v20 // (op 1328) 
v_lshlrev_b32 v21, 16, v21 // (op 1328) 
v_mul_f32 v60, v7, v20 // (op 1328) 
v_mul_f32 v61, v7, v21 // (op 1328) 
v_and_b32 v20, 65535, v17 // (op 1328) 
v_lshrrev_b32 v21, 16, v17 // (op 1328) 
v_lshlrev_b32 v20, 16, v20 // (op 1328) 
v_lshlrev_b32 v21, 16, v21 // (op 1328) 
v_mul_f32 v62, v7, v20 // (op 1328) 
v_mul_f32 v63, v7, v21 // (op 1328) 
s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v18 // (op 1328) 
v_lshrrev_b32 v21, 16, v18 // (op 1328) 
v_lshlrev_b32 v20, 16, v20 // (op 1328) 
v_lshlrev_b32 v21, 16, v21 // (op 1328) 
v_mul_f32 v64, v7, v20 // (op 1328) 
v_mul_f32 v65, v7, v21 // (op 1328) 
v_and_b32 v20, 65535, v19 // (op 1328) 
v_lshrrev_b32 v21, 16, v19 // (op 1328) 
v_lshlrev_b32 v20, 16, v20 // (op 1328) 
v_lshlrev_b32 v21, 16, v21 // (op 1328) 
v_mul_f32 v66, v7, v20 // (op 1328) 
v_mul_f32 v67, v7, v21 // (op 1328) 


v_add_f32 v12, v52, v36 // (op 1324) 
v_add_f32 v13, v53, v37 // (op 1324) 
v_add_f32 v14, v54, v38 // (op 1324) 
v_add_f32 v15, v55, v39 // (op 1324) 
v_add_f32 v16, v56, v40 // (op 1324) 
v_add_f32 v17, v57, v41 // (op 1324) 
v_add_f32 v18, v58, v42 // (op 1324) 
v_add_f32 v19, v59, v43 // (op 1324) 
v_add_f32 v20, v60, v44 // (op 1324) 
v_add_f32 v21, v61, v45 // (op 1324) 
v_add_f32 v22, v62, v46 // (op 1324) 
v_add_f32 v23, v63, v47 // (op 1324) 
v_add_f32 v24, v64, v48 // (op 1324) 
v_add_f32 v25, v65, v49 // (op 1324) 
v_add_f32 v26, v66, v50 // (op 1324) 
v_add_f32 v27, v67, v51 // (op 1324) 


v_lshrrev_b32 v28, 16, v12 // (op 1325) 
v_lshrrev_b32 v29, 16, v13 // (op 1325) 
v_lshrrev_b32 v30, 16, v14 // (op 1325) 
v_lshrrev_b32 v31, 16, v15 // (op 1325) 
v_lshrrev_b32 v32, 16, v16 // (op 1325) 
v_lshrrev_b32 v33, 16, v17 // (op 1325) 
v_lshrrev_b32 v34, 16, v18 // (op 1325) 
v_lshrrev_b32 v35, 16, v19 // (op 1325) 
v_lshrrev_b32 v36, 16, v20 // (op 1325) 
v_lshrrev_b32 v37, 16, v21 // (op 1325) 
v_lshrrev_b32 v38, 16, v22 // (op 1325) 
v_lshrrev_b32 v39, 16, v23 // (op 1325) 
v_lshrrev_b32 v40, 16, v24 // (op 1325) 
v_lshrrev_b32 v41, 16, v25 // (op 1325) 
v_lshrrev_b32 v42, 16, v26 // (op 1325) 
v_lshrrev_b32 v43, 16, v27 // (op 1325) 


v_pack_b32_f16 v12, v28, v29 // (op 1326) 
v_pack_b32_f16 v13, v30, v31 // (op 1326) 
v_pack_b32_f16 v14, v32, v33 // (op 1326) 
v_pack_b32_f16 v15, v34, v35 // (op 1326) 
v_pack_b32_f16 v16, v36, v37 // (op 1326) 
v_pack_b32_f16 v17, v38, v39 // (op 1326) 
v_pack_b32_f16 v18, v40, v41 // (op 1326) 
v_pack_b32_f16 v19, v42, v43 // (op 1326) 

s_lshl_b32 s0, s4, 7 // (op 1326) 
v_add_u32 v0, v11, s0 // (op 1326) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 1326) Store value
buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen offset:16 // (op 1326) Store value
buffer_store_dwordx2 v[16:17], v0, s[60:63], 0 offen offset:32 // (op 1326) Store value
buffer_store_dwordx2 v[18:19], v0, s[60:63], 0 offen offset:48 // (op 1326) Store value





s_lshl_b32 s0, s2, 8 // (op 1341) 
v_add_u32 v0, v8, s0 // (op 1341) 


buffer_load_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 1341) Load value
buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen offset:16 // (op 1341) Load value
v_accvgpr_read v20, a32 // (op 1337) 
v_accvgpr_read v21, a33 // (op 1337) 
v_accvgpr_read v22, a34 // (op 1337) 
v_accvgpr_read v23, a35 // (op 1337) 
v_accvgpr_read v24, a36 // (op 1337) 
v_accvgpr_read v25, a37 // (op 1337) 
v_accvgpr_read v26, a38 // (op 1337) 
v_accvgpr_read v27, a39 // (op 1337) 
v_accvgpr_read v28, a40 // (op 1337) 
v_accvgpr_read v29, a41 // (op 1337) 
v_accvgpr_read v30, a42 // (op 1337) 
v_accvgpr_read v31, a43 // (op 1337) 
v_accvgpr_read v32, a44 // (op 1337) 
v_accvgpr_read v33, a45 // (op 1337) 
v_accvgpr_read v34, a46 // (op 1337) 
v_accvgpr_read v35, a47 // (op 1337) 
v_mul_f32 v36, v10, v20 // (op 1337) 
v_mul_f32 v37, v10, v21 // (op 1337) 
v_mul_f32 v38, v10, v22 // (op 1337) 
v_mul_f32 v39, v10, v23 // (op 1337) 
v_mul_f32 v40, v10, v24 // (op 1337) 
v_mul_f32 v41, v10, v25 // (op 1337) 
v_mul_f32 v42, v10, v26 // (op 1337) 
v_mul_f32 v43, v10, v27 // (op 1337) 
v_mul_f32 v44, v10, v28 // (op 1337) 
v_mul_f32 v45, v10, v29 // (op 1337) 
v_mul_f32 v46, v10, v30 // (op 1337) 
v_mul_f32 v47, v10, v31 // (op 1337) 
v_mul_f32 v48, v10, v32 // (op 1337) 
v_mul_f32 v49, v10, v33 // (op 1337) 
v_mul_f32 v50, v10, v34 // (op 1337) 
v_mul_f32 v51, v10, v35 // (op 1337) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen offset:32 // (op 1341) Load value
buffer_load_dwordx2 v[18:19], v0, s[56:59], 0 offen offset:48 // (op 1341) Load value

s_waitcnt vmcnt(3)
v_and_b32 v20, 65535, v12 // (op 1342) 
v_lshrrev_b32 v21, 16, v12 // (op 1342) 
v_lshlrev_b32 v20, 16, v20 // (op 1342) 
v_lshlrev_b32 v21, 16, v21 // (op 1342) 
v_mul_f32 v52, v7, v20 // (op 1342) 
v_mul_f32 v53, v7, v21 // (op 1342) 
v_and_b32 v20, 65535, v13 // (op 1342) 
v_lshrrev_b32 v21, 16, v13 // (op 1342) 
v_lshlrev_b32 v20, 16, v20 // (op 1342) 
v_lshlrev_b32 v21, 16, v21 // (op 1342) 
v_mul_f32 v54, v7, v20 // (op 1342) 
v_mul_f32 v55, v7, v21 // (op 1342) 
s_waitcnt vmcnt(2)
v_and_b32 v20, 65535, v14 // (op 1342) 
v_lshrrev_b32 v21, 16, v14 // (op 1342) 
v_lshlrev_b32 v20, 16, v20 // (op 1342) 
v_lshlrev_b32 v21, 16, v21 // (op 1342) 
v_mul_f32 v56, v7, v20 // (op 1342) 
v_mul_f32 v57, v7, v21 // (op 1342) 
v_and_b32 v20, 65535, v15 // (op 1342) 
v_lshrrev_b32 v21, 16, v15 // (op 1342) 
v_lshlrev_b32 v20, 16, v20 // (op 1342) 
v_lshlrev_b32 v21, 16, v21 // (op 1342) 
v_mul_f32 v58, v7, v20 // (op 1342) 
v_mul_f32 v59, v7, v21 // (op 1342) 
s_waitcnt vmcnt(1)
v_and_b32 v20, 65535, v16 // (op 1342) 
v_lshrrev_b32 v21, 16, v16 // (op 1342) 
v_lshlrev_b32 v20, 16, v20 // (op 1342) 
v_lshlrev_b32 v21, 16, v21 // (op 1342) 
v_mul_f32 v60, v7, v20 // (op 1342) 
v_mul_f32 v61, v7, v21 // (op 1342) 
v_and_b32 v20, 65535, v17 // (op 1342) 
v_lshrrev_b32 v21, 16, v17 // (op 1342) 
v_lshlrev_b32 v20, 16, v20 // (op 1342) 
v_lshlrev_b32 v21, 16, v21 // (op 1342) 
v_mul_f32 v62, v7, v20 // (op 1342) 
v_mul_f32 v63, v7, v21 // (op 1342) 
s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v18 // (op 1342) 
v_lshrrev_b32 v21, 16, v18 // (op 1342) 
v_lshlrev_b32 v20, 16, v20 // (op 1342) 
v_lshlrev_b32 v21, 16, v21 // (op 1342) 
v_mul_f32 v64, v7, v20 // (op 1342) 
v_mul_f32 v65, v7, v21 // (op 1342) 
v_and_b32 v20, 65535, v19 // (op 1342) 
v_lshrrev_b32 v21, 16, v19 // (op 1342) 
v_lshlrev_b32 v20, 16, v20 // (op 1342) 
v_lshlrev_b32 v21, 16, v21 // (op 1342) 
v_mul_f32 v66, v7, v20 // (op 1342) 
v_mul_f32 v67, v7, v21 // (op 1342) 


v_add_f32 v12, v52, v36 // (op 1338) 
v_add_f32 v13, v53, v37 // (op 1338) 
v_add_f32 v14, v54, v38 // (op 1338) 
v_add_f32 v15, v55, v39 // (op 1338) 
v_add_f32 v16, v56, v40 // (op 1338) 
v_add_f32 v17, v57, v41 // (op 1338) 
v_add_f32 v18, v58, v42 // (op 1338) 
v_add_f32 v19, v59, v43 // (op 1338) 
v_add_f32 v20, v60, v44 // (op 1338) 
v_add_f32 v21, v61, v45 // (op 1338) 
v_add_f32 v22, v62, v46 // (op 1338) 
v_add_f32 v23, v63, v47 // (op 1338) 
v_add_f32 v24, v64, v48 // (op 1338) 
v_add_f32 v25, v65, v49 // (op 1338) 
v_add_f32 v26, v66, v50 // (op 1338) 
v_add_f32 v27, v67, v51 // (op 1338) 


v_lshrrev_b32 v28, 16, v12 // (op 1339) 
v_lshrrev_b32 v29, 16, v13 // (op 1339) 
v_lshrrev_b32 v30, 16, v14 // (op 1339) 
v_lshrrev_b32 v31, 16, v15 // (op 1339) 
v_lshrrev_b32 v32, 16, v16 // (op 1339) 
v_lshrrev_b32 v33, 16, v17 // (op 1339) 
v_lshrrev_b32 v34, 16, v18 // (op 1339) 
v_lshrrev_b32 v35, 16, v19 // (op 1339) 
v_lshrrev_b32 v36, 16, v20 // (op 1339) 
v_lshrrev_b32 v37, 16, v21 // (op 1339) 
v_lshrrev_b32 v38, 16, v22 // (op 1339) 
v_lshrrev_b32 v39, 16, v23 // (op 1339) 
v_lshrrev_b32 v40, 16, v24 // (op 1339) 
v_lshrrev_b32 v41, 16, v25 // (op 1339) 
v_lshrrev_b32 v42, 16, v26 // (op 1339) 
v_lshrrev_b32 v43, 16, v27 // (op 1339) 


v_pack_b32_f16 v12, v28, v29 // (op 1340) 
v_pack_b32_f16 v13, v30, v31 // (op 1340) 
v_pack_b32_f16 v14, v32, v33 // (op 1340) 
v_pack_b32_f16 v15, v34, v35 // (op 1340) 
v_pack_b32_f16 v16, v36, v37 // (op 1340) 
v_pack_b32_f16 v17, v38, v39 // (op 1340) 
v_pack_b32_f16 v18, v40, v41 // (op 1340) 
v_pack_b32_f16 v19, v42, v43 // (op 1340) 

s_lshl_b32 s0, s4, 8 // (op 1340) 
v_add_u32 v0, v11, s0 // (op 1340) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 1340) Store value
buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen offset:16 // (op 1340) Store value
buffer_store_dwordx2 v[16:17], v0, s[60:63], 0 offen offset:32 // (op 1340) Store value
buffer_store_dwordx2 v[18:19], v0, s[60:63], 0 offen offset:48 // (op 1340) Store value





s_lshl_b32 s0, s2, 7 // (op 1355) 
s_mul_i32 s1, 3, s0 // (op 1355) 

v_add_u32 v0, v8, s1 // (op 1355) 


buffer_load_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 1355) Load value
buffer_load_dwordx2 v[14:15], v0, s[56:59], 0 offen offset:16 // (op 1355) Load value
v_accvgpr_read v20, a48 // (op 1351) 
v_accvgpr_read v21, a49 // (op 1351) 
v_accvgpr_read v22, a50 // (op 1351) 
v_accvgpr_read v23, a51 // (op 1351) 
v_accvgpr_read v24, a52 // (op 1351) 
v_accvgpr_read v25, a53 // (op 1351) 
v_accvgpr_read v26, a54 // (op 1351) 
v_accvgpr_read v27, a55 // (op 1351) 
v_accvgpr_read v28, a56 // (op 1351) 
v_accvgpr_read v29, a57 // (op 1351) 
v_accvgpr_read v30, a58 // (op 1351) 
v_accvgpr_read v31, a59 // (op 1351) 
v_accvgpr_read v32, a60 // (op 1351) 
v_accvgpr_read v33, a61 // (op 1351) 
v_accvgpr_read v34, a62 // (op 1351) 
v_accvgpr_read v35, a63 // (op 1351) 
v_mul_f32 v36, v10, v20 // (op 1351) 
v_mul_f32 v37, v10, v21 // (op 1351) 
v_mul_f32 v38, v10, v22 // (op 1351) 
v_mul_f32 v39, v10, v23 // (op 1351) 
v_mul_f32 v40, v10, v24 // (op 1351) 
v_mul_f32 v41, v10, v25 // (op 1351) 
v_mul_f32 v42, v10, v26 // (op 1351) 
v_mul_f32 v43, v10, v27 // (op 1351) 
v_mul_f32 v44, v10, v28 // (op 1351) 
v_mul_f32 v45, v10, v29 // (op 1351) 
v_mul_f32 v46, v10, v30 // (op 1351) 
v_mul_f32 v47, v10, v31 // (op 1351) 
v_mul_f32 v48, v10, v32 // (op 1351) 
v_mul_f32 v49, v10, v33 // (op 1351) 
v_mul_f32 v50, v10, v34 // (op 1351) 
v_mul_f32 v51, v10, v35 // (op 1351) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen offset:32 // (op 1355) Load value
buffer_load_dwordx2 v[18:19], v0, s[56:59], 0 offen offset:48 // (op 1355) Load value

s_waitcnt vmcnt(3)
v_and_b32 v20, 65535, v12 // (op 1356) 
v_lshrrev_b32 v21, 16, v12 // (op 1356) 
v_lshlrev_b32 v20, 16, v20 // (op 1356) 
v_lshlrev_b32 v21, 16, v21 // (op 1356) 
v_mul_f32 v52, v7, v20 // (op 1356) 
v_mul_f32 v53, v7, v21 // (op 1356) 
v_and_b32 v20, 65535, v13 // (op 1356) 
v_lshrrev_b32 v21, 16, v13 // (op 1356) 
v_lshlrev_b32 v20, 16, v20 // (op 1356) 
v_lshlrev_b32 v21, 16, v21 // (op 1356) 
v_mul_f32 v54, v7, v20 // (op 1356) 
v_mul_f32 v55, v7, v21 // (op 1356) 
s_waitcnt vmcnt(2)
v_and_b32 v20, 65535, v14 // (op 1356) 
v_lshrrev_b32 v21, 16, v14 // (op 1356) 
v_lshlrev_b32 v20, 16, v20 // (op 1356) 
v_lshlrev_b32 v21, 16, v21 // (op 1356) 
v_mul_f32 v56, v7, v20 // (op 1356) 
v_mul_f32 v57, v7, v21 // (op 1356) 
v_and_b32 v20, 65535, v15 // (op 1356) 
v_lshrrev_b32 v21, 16, v15 // (op 1356) 
v_lshlrev_b32 v20, 16, v20 // (op 1356) 
v_lshlrev_b32 v21, 16, v21 // (op 1356) 
v_mul_f32 v58, v7, v20 // (op 1356) 
v_mul_f32 v59, v7, v21 // (op 1356) 
s_waitcnt vmcnt(1)
v_and_b32 v20, 65535, v16 // (op 1356) 
v_lshrrev_b32 v21, 16, v16 // (op 1356) 
v_lshlrev_b32 v20, 16, v20 // (op 1356) 
v_lshlrev_b32 v21, 16, v21 // (op 1356) 
v_mul_f32 v60, v7, v20 // (op 1356) 
v_mul_f32 v61, v7, v21 // (op 1356) 
v_and_b32 v20, 65535, v17 // (op 1356) 
v_lshrrev_b32 v21, 16, v17 // (op 1356) 
v_lshlrev_b32 v20, 16, v20 // (op 1356) 
v_lshlrev_b32 v21, 16, v21 // (op 1356) 
v_mul_f32 v62, v7, v20 // (op 1356) 
v_mul_f32 v63, v7, v21 // (op 1356) 
s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v18 // (op 1356) 
v_lshrrev_b32 v21, 16, v18 // (op 1356) 
v_lshlrev_b32 v20, 16, v20 // (op 1356) 
v_lshlrev_b32 v21, 16, v21 // (op 1356) 
v_mul_f32 v64, v7, v20 // (op 1356) 
v_mul_f32 v65, v7, v21 // (op 1356) 
v_and_b32 v20, 65535, v19 // (op 1356) 
v_lshrrev_b32 v21, 16, v19 // (op 1356) 
v_lshlrev_b32 v20, 16, v20 // (op 1356) 
v_lshlrev_b32 v21, 16, v21 // (op 1356) 
v_mul_f32 v66, v7, v20 // (op 1356) 
v_mul_f32 v67, v7, v21 // (op 1356) 


v_add_f32 v12, v52, v36 // (op 1352) 
v_add_f32 v13, v53, v37 // (op 1352) 
v_add_f32 v14, v54, v38 // (op 1352) 
v_add_f32 v15, v55, v39 // (op 1352) 
v_add_f32 v16, v56, v40 // (op 1352) 
v_add_f32 v17, v57, v41 // (op 1352) 
v_add_f32 v18, v58, v42 // (op 1352) 
v_add_f32 v19, v59, v43 // (op 1352) 
v_add_f32 v20, v60, v44 // (op 1352) 
v_add_f32 v21, v61, v45 // (op 1352) 
v_add_f32 v22, v62, v46 // (op 1352) 
v_add_f32 v23, v63, v47 // (op 1352) 
v_add_f32 v24, v64, v48 // (op 1352) 
v_add_f32 v25, v65, v49 // (op 1352) 
v_add_f32 v26, v66, v50 // (op 1352) 
v_add_f32 v27, v67, v51 // (op 1352) 


v_lshrrev_b32 v28, 16, v12 // (op 1353) 
v_lshrrev_b32 v29, 16, v13 // (op 1353) 
v_lshrrev_b32 v30, 16, v14 // (op 1353) 
v_lshrrev_b32 v31, 16, v15 // (op 1353) 
v_lshrrev_b32 v32, 16, v16 // (op 1353) 
v_lshrrev_b32 v33, 16, v17 // (op 1353) 
v_lshrrev_b32 v34, 16, v18 // (op 1353) 
v_lshrrev_b32 v35, 16, v19 // (op 1353) 
v_lshrrev_b32 v36, 16, v20 // (op 1353) 
v_lshrrev_b32 v37, 16, v21 // (op 1353) 
v_lshrrev_b32 v38, 16, v22 // (op 1353) 
v_lshrrev_b32 v39, 16, v23 // (op 1353) 
v_lshrrev_b32 v40, 16, v24 // (op 1353) 
v_lshrrev_b32 v41, 16, v25 // (op 1353) 
v_lshrrev_b32 v42, 16, v26 // (op 1353) 
v_lshrrev_b32 v43, 16, v27 // (op 1353) 


v_pack_b32_f16 v12, v28, v29 // (op 1354) 
v_pack_b32_f16 v13, v30, v31 // (op 1354) 
v_pack_b32_f16 v14, v32, v33 // (op 1354) 
v_pack_b32_f16 v15, v34, v35 // (op 1354) 
v_pack_b32_f16 v16, v36, v37 // (op 1354) 
v_pack_b32_f16 v17, v38, v39 // (op 1354) 
v_pack_b32_f16 v18, v40, v41 // (op 1354) 
v_pack_b32_f16 v19, v42, v43 // (op 1354) 

s_lshl_b32 s0, s4, 7 // (op 1354) 
s_mul_i32 s1, 3, s0 // (op 1354) 

v_add_u32 v0, v11, s1 // (op 1354) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 1354) Store value
buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen offset:16 // (op 1354) Store value
buffer_store_dwordx2 v[16:17], v0, s[60:63], 0 offen offset:32 // (op 1354) Store value
buffer_store_dwordx2 v[18:19], v0, s[60:63], 0 offen offset:48 // (op 1354) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_157:












s_endpgm  // End of RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.LRR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end:
.size RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_, .LRR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end-RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 212
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 84480 // lds bytes
  .amdhsa_accum_offset 148
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
    DataTypeB: B6
    DestDataType: b
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
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 84480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 146
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

