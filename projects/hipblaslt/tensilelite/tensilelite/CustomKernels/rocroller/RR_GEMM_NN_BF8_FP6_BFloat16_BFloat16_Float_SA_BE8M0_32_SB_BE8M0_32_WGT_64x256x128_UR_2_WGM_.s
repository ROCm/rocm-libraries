// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_,@function
RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 2687) 
s_mul_hi_i32 s1, s16, s10 // (op 2928) 
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
s_cmp_eq_i32 s10, 0 // (op 2687) 
s_mov_b32 s19, scc // (op 2687) 
v_lshlrev_b32 v9, 2, v8 // (op 2687) 

s_cmp_eq_i32 s19, 0 // (op 2687) 
s_cselect_b32 s51, 0, -1 // (op 2687) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2687) 


s_and_b32 s19, s11, 31 // (op 2687) 
v_lshrrev_b32 v6, 2, v7 // (op 2687) 

v_and_b32 v7, 7, v6 // (op 2687) 

v_and_b32 v6, 63, v1 // (op 2801) 
v_lshrrev_b32 v9, 4, v6 // (op 2801) 

v_and_b32 v6, 1, v9 // (op 2801) 

v_and_b32 v9, 15, v1 // (op 2801) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 2801) 


v_lshrrev_b32 v6, 6, v1 // (op 2801) 
v_and_b32 v9, 1, v6 // (op 2801) 

v_lshl_add_u32 v6, v9, 5, v10 // (op 2801) 


v_and_b32 v9, 32, v1 // (op 2801) 
v_lshl_add_u32 v10, v6, 7, v9 // (op 2801) 


v_mul_lo_u32 v6, v10, 6 // (op 2801) 

v_lshrrev_b32 v9, 3, v6 // (op 2801) 

v_lshrrev_b32 v6, 6, v1 // (op 2912) 
v_and_b32 v10, 1, v6 // (op 2912) 

v_and_b32 v6, 31, v1 // (op 2912) 
v_lshl_add_u32 v11, v10, 5, v6 // (op 2912) 


v_and_b32 v6, 63, v1 // (op 2912) 
v_lshrrev_b32 v10, 5, v6 // (op 2912) 

v_lshl_add_u32 v6, v11, 2, v10 // (op 2912) 


v_lshrrev_b32 v10, 7, v1 // (op 2928) 
v_and_b32 v11, 63, v1 // (op 2928) 
v_and_b32 v12, 63, v1 // (op 2998) 
v_lshrrev_b32 v13, 4, v12 // (op 2998) 

v_and_b32 v12, 1, v13 // (op 2998) 

v_and_b32 v13, 15, v1 // (op 2998) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2998) 


v_lshrrev_b32 v12, 6, v1 // (op 2998) 
v_and_b32 v13, 1, v12 // (op 2998) 

v_lshl_add_u32 v12, v13, 5, v14 // (op 2998) 


v_and_b32 v13, 32, v1 // (op 2998) 
v_lshl_add_u32 v14, v12, 7, v13 // (op 2998) 


v_mul_lo_u32 v12, v14, 6 // (op 2998) 

v_lshrrev_b32 v13, 3, v12 // (op 2998) 

v_lshrrev_b32 v12, 6, v1 // (op 3051) 
v_and_b32 v14, 1, v12 // (op 3051) 

v_and_b32 v12, 31, v1 // (op 3051) 
v_lshl_add_u32 v15, v14, 5, v12 // (op 3051) 


v_and_b32 v12, 63, v1 // (op 3051) 
v_lshrrev_b32 v14, 5, v12 // (op 3051) 

v_lshl_add_u32 v12, v15, 2, v14 // (op 3051) 


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

s_lshl_b32 s19, s11, 24 // (op 2687) 
v_lshl_add_u32 v7, v14, 5, v15 // (op 2687) 


s_ashr_i32 s51, s19, 31 // (op 2687) 

s_xor_b32 s19, s18, s51 // (op 2687) 

s_sub_i32 s18, s19, s51 // (op 2687) 


s_mul_hi_i32 s19, s18, s12 // (op 2687) 
s_add_i32 s51, s19, s18 // (op 2687) 

s_ashr_i32 s19, s51, 31 // (op 2687) 
s_cmp_eq_i32 s12, 0 // (op 2687) 
s_mov_b32 s52, scc // (op 2687) 
s_cmp_eq_i32 s52, 0 // (op 2687) 
s_cselect_b32 s53, 0, -1 // (op 2687) 

s_and_b32 s52, s13, 31 // (op 2687) 
v_lshrrev_b32 v14, 5, v11 // (op 2928) 
s_cmp_eq_i32 s10, 0 // (op 2928) 
s_mov_b32 s54, scc // (op 2928) 
v_lshlrev_b32 v15, 2, v14 // (op 2928) 

s_cmp_eq_i32 s54, 0 // (op 2928) 
s_cselect_b32 s55, 0, -1 // (op 2928) 

s_lshl_b32 s54, 1, s52 // (op 2687) 
v_lshl_add_u32 v14, v10, 5, v15 // (op 2928) 


v_lshrrev_b32 v10, 2, v11 // (op 2928) 

s_add_i32 s56, s54, s53 // (op 2687) 


s_and_b32 s53, s19, s56 // (op 2687) 


s_add_i32 s19, s51, s53 // (op 2687) 


s_ashr_i32 s51, s19, s52 // (op 2687) 


s_lshl_b32 s19, s13, 24 // (op 2687) 
s_ashr_i32 s52, s19, 31 // (op 2687) 

s_xor_b32 s19, s51, s52 // (op 2687) 

s_sub_i32 s51, s19, s52 // (op 2687) 


s_mul_i32 s19, s18, s15 // (op 2687) 
s_sub_u32 s52, s16, s19 // (op 2687) 

s_mul_i32 s19, s51, s15 // (op 2687) 
s_add_u32 s19, s19, s52 // (op 2687) 

s_mul_hi_i32 s52, s16, s9 // (op 2687) 
s_add_i32 s53, s52, s16 // (op 2687) 

s_ashr_i32 s52, s53, 31 // (op 2687) 
s_cmp_eq_i32 s9, 0 // (op 2687) 
s_mov_b32 s54, scc // (op 2687) 
s_cmp_eq_i32 s54, 0 // (op 2687) 
s_cselect_b32 s56, 0, -1 // (op 2687) 

s_and_b32 s54, s11, 31 // (op 2928) 
v_and_b32 v11, 7, v10 // (op 2928) 

s_lshl_b32 s57, 1, s54 // (op 2928) 
v_and_b32 v10, 3, v1 // (op 2928) 
s_add_i32 s58, s57, s55 // (op 2928) 


s_and_b32 s55, s1, s58 // (op 2928) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 2928) 


s_add_i32 s1, s0, s55 // (op 2928) 


v_lshrrev_b32 v10, 6, v1 // (op 2928) 
s_ashr_i32 s0, s1, s54 // (op 2928) 


v_and_b32 v11, 1, v10 // (op 2928) 

s_lshl_b32 s1, s11, 24 // (op 2928) 
v_lshl_add_u32 v10, v11, 5, v15 // (op 2928) 


s_ashr_i32 s54, s1, 31 // (op 2928) 

s_xor_b32 s1, s0, s54 // (op 2928) 

s_sub_i32 s0, s1, s54 // (op 2928) 


s_mul_hi_i32 s1, s0, s12 // (op 2928) 
s_add_i32 s54, s1, s0 // (op 2928) 

s_ashr_i32 s1, s54, 31 // (op 2928) 
s_cmp_eq_i32 s12, 0 // (op 2928) 
s_mov_b32 s55, scc // (op 2928) 
s_cmp_eq_i32 s55, 0 // (op 2928) 
s_cselect_b32 s57, 0, -1 // (op 2928) 

s_and_b32 s55, s13, 31 // (op 2928) 
s_lshl_b32 s58, 1, s55 // (op 2928) 
s_add_i32 s59, s58, s57 // (op 2928) 


s_and_b32 s57, s1, s59 // (op 2928) 


s_add_i32 s1, s54, s57 // (op 2928) 


s_ashr_i32 s54, s1, s55 // (op 2928) 


s_lshl_b32 s1, s13, 24 // (op 2928) 
s_ashr_i32 s55, s1, 31 // (op 2928) 

s_xor_b32 s1, s54, s55 // (op 2928) 

s_sub_i32 s54, s1, s55 // (op 2928) 


s_mul_i32 s1, s0, s15 // (op 2928) 
s_sub_u32 s55, s16, s1 // (op 2928) 

s_mul_i32 s1, s54, s15 // (op 2928) 
s_add_u32 s1, s1, s55 // (op 2928) 

s_mul_hi_i32 s55, s16, s9 // (op 2928) 
s_add_i32 s57, s55, s16 // (op 2928) 

s_ashr_i32 s55, s57, 31 // (op 2928) 
s_cmp_eq_i32 s9, 0 // (op 2928) 
s_mov_b32 s58, scc // (op 2928) 
s_cmp_eq_i32 s58, 0 // (op 2928) 
s_cselect_b32 s59, 0, -1 // (op 2928) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 2687) 
s_lshl_b32 s60, 1, s58 // (op 2687) 
s_add_i32 s61, s60, s56 // (op 2687) 


s_and_b32 s56, s52, s61 // (op 2687) 


s_add_i32 s52, s53, s56 // (op 2687) 


s_ashr_i32 s53, s52, s58 // (op 2687) 


s_lshl_b32 s52, s46, 24 // (op 2687) 
s_ashr_i32 s56, s52, 31 // (op 2687) 

s_xor_b32 s52, s53, s56 // (op 2687) 

s_sub_i32 s53, s52, s56 // (op 2687) 


s_and_b32 s52, s46, 31 // (op 2928) 
s_lshl_b32 s56, 1, s52 // (op 2928) 
s_add_i32 s58, s56, s59 // (op 2928) 


s_and_b32 s56, s55, s58 // (op 2928) 


s_add_i32 s55, s57, s56 // (op 2928) 


s_ashr_i32 s56, s55, s52 // (op 2928) 


s_lshl_b32 s52, s46, 24 // (op 2928) 
s_ashr_i32 s55, s52, 31 // (op 2928) 

s_xor_b32 s52, s56, s55 // (op 2928) 

s_sub_i32 s56, s52, s55 // (op 2928) 


s_add_u32 s58, s6, 256 // (op 2687) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 2687) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2687) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2687) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 8 // (op 2687) 

s_mul_i32 s52, s15, s58 // (op 2687) 
s_mul_i32 s55, s8, s52 // (op 2687) 

s_mul_i32 s52, s53, s55 // (op 2687) 

s_sub_u32 s55, s16, s52 // (op 2687) 

s_mov_b32 s52, s55 // (op 2687) 
s_mul_hi_i32 s57, s52, s14 // (op 2687) 
s_add_i32 s60, s57, s52 // (op 2687) 


s_ashr_i32 s52, s60, 31 // (op 2687) 
s_cmp_eq_i32 s14, 0 // (op 2687) 
s_mov_b32 s57, scc // (op 2687) 
s_cmp_eq_i32 s57, 0 // (op 2687) 
s_cselect_b32 s61, 0, -1 // (op 2687) 

s_and_b32 s57, s48, 31 // (op 2687) 
s_lshl_b32 s62, 1, s57 // (op 2687) 
s_add_i32 s63, s62, s61 // (op 2687) 


s_and_b32 s61, s52, s63 // (op 2687) 


s_add_i32 s52, s60, s61 // (op 2687) 


s_ashr_i32 s60, s52, s57 // (op 2687) 


s_lshl_b32 s52, s48, 24 // (op 2687) 
s_ashr_i32 s57, s52, 31 // (op 2687) 

s_xor_b32 s52, s60, s57 // (op 2687) 

s_sub_i32 s60, s52, s57 // (op 2687) 


s_mul_i32 s52, s47, s15 // (op 2687) 
s_add_u32 s62, s20, 64 // (op 2687) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 2687) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2687) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2687) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 2687) 

s_sub_i32 s57, s62, s52 // (op 2687) 

s_mul_i32 s61, s60, s57 // (op 2687) 

s_sub_u32 s57, s55, s61 // (op 2687) 


s_add_u32 s55, s57, s52 // (op 2687) 


s_cmp_eq_i32 s53, 0 // (op 2687) 
s_mov_b32 s52, scc // (op 2687) 

s_cmp_eq_i32 s52, 0 // (op 2687) 
s_cselect_b32 s53, s55, s19 // (op 2687) 


s_mul_i32 s19, s51, s58 // (op 2687) 


v_lshl_add_u32 v11, s53, 6, v8 // (op 2687) 


s_sub_i32 s51, s18, s19 // (op 2687) 


s_cmp_eq_i32 s52, 0 // (op 2687) 
s_cselect_b32 s53, s60, s51 // (op 2687) 



v_lshl_add_u32 v8, s53, 8, v7 // (op 2687) 


v_mul_lo_u32 v7, v8, s2 // (op 2687) 

v_add_lshl_u32 v8, v11, v7, 1 // (op 2687) 


s_add_u32 s18, s6, 256 // (op 2928) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 2928) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 2928) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 2928) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 8 // (op 2928) 

s_mul_i32 s55, s15, s18 // (op 2928) 
s_mul_i32 s51, s8, s55 // (op 2928) 

s_mul_i32 s55, s56, s51 // (op 2928) 

s_sub_u32 s51, s16, s55 // (op 2928) 

s_mov_b32 s55, s51 // (op 2928) 
s_mul_hi_i32 s52, s55, s14 // (op 2928) 
s_add_i32 s53, s52, s55 // (op 2928) 


s_ashr_i32 s52, s53, 31 // (op 2928) 
s_cmp_eq_i32 s14, 0 // (op 2928) 
s_mov_b32 s55, scc // (op 2928) 
s_cmp_eq_i32 s55, 0 // (op 2928) 
s_cselect_b32 s57, 0, -1 // (op 2928) 

s_and_b32 s55, s48, 31 // (op 2928) 
s_lshl_b32 s58, 1, s55 // (op 2928) 
s_add_i32 s59, s58, s57 // (op 2928) 


s_and_b32 s57, s52, s59 // (op 2928) 


s_add_i32 s52, s53, s57 // (op 2928) 


s_ashr_i32 s53, s52, s55 // (op 2928) 


s_lshl_b32 s52, s48, 24 // (op 2928) 
s_ashr_i32 s55, s52, 31 // (op 2928) 

s_xor_b32 s52, s53, s55 // (op 2928) 

s_sub_i32 s53, s52, s55 // (op 2928) 


s_mul_i32 s52, s47, s15 // (op 2928) 
s_add_u32 s58, s20, 64 // (op 2928) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 2928) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2928) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2928) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2928) 

s_sub_i32 s55, s58, s52 // (op 2928) 

s_mul_i32 s57, s53, s55 // (op 2928) 

s_sub_u32 s55, s51, s57 // (op 2928) 


s_add_u32 s51, s55, s52 // (op 2928) 


s_cmp_eq_i32 s56, 0 // (op 2928) 
s_mov_b32 s52, scc // (op 2928) 

s_cmp_eq_i32 s52, 0 // (op 2928) 
s_cselect_b32 s55, s51, s1 // (op 2928) 


s_mul_i32 s1, s54, s18 // (op 2928) 


v_lshl_add_u32 v7, s55, 6, v14 // (op 2928) 


s_sub_i32 s51, s0, s1 // (op 2928) 


s_cmp_eq_i32 s52, 0 // (op 2928) 
s_cselect_b32 s0, s53, s51 // (op 2928) 



v_lshl_add_u32 v11, s0, 8, v10 // (op 2928) 


v_mul_lo_u32 v10, v11, s4 // (op 2928) 

v_add_lshl_u32 v11, v7, v10, 1 // (op 2928) 


s_mul_i32 s1, s2, s7 // (op 2689) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 2689) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 2689) least significant
s_mul_hi_u32 s19, s2, s6 // (op 2689) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2689) most significant: sum
s_add_u32 s1, s1, s19 // (op 2689) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2689) 

s_mul_i32 s1, s4, s7 // (op 2930) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 2930) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 2930) least significant
s_mul_hi_u32 s52, s4, s6 // (op 2930) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2930) most significant: sum
s_add_u32 s1, s1, s52 // (op 2930) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 2930) 

s_mov_b32 s56, s22 // (op 2689) reg expression
s_mov_b32 s57, s23 // (op 2689) reg expression
s_mov_b32 s58, s18 // (op 2689) reg expression
s_mov_b32 s59, 131072 // (op 2689) call()

s_mov_b32 s60, s24 // (op 2930) reg expression

s_mov_b32 s61, s25 // (op 2930) reg expression
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
s_mul_hi_i32 s0, s16, s10 // (op 2454) 
v_and_b32 v14, -4, v1 // (op 2454) 
s_add_i32 s1, s0, s16 // (op 2454) 

v_sub_u32 v15, v1, v14 // (op 2454) 

s_ashr_i32 s0, s1, 31 // (op 2454) 
v_lshlrev_b32 v14, 4, v15 // (op 2454) 

s_cmp_eq_i32 s10, 0 // (op 2454) 
s_mov_b32 s18, scc // (op 2454) 
v_lshrrev_b32 v15, 2, v1 // (op 2454) 
s_cmp_eq_i32 s18, 0 // (op 2454) 
s_cselect_b32 s19, 0, -1 // (op 2454) 

s_and_b32 s18, s11, 31 // (op 2454) 
s_lshl_b32 s22, 1, s18 // (op 2454) 
s_add_i32 s23, s22, s19 // (op 2454) 


s_and_b32 s19, s0, s23 // (op 2454) 


s_add_i32 s0, s1, s19 // (op 2454) 


s_ashr_i32 s1, s0, s18 // (op 2454) 


s_lshl_b32 s0, s11, 24 // (op 2454) 
s_ashr_i32 s18, s0, 31 // (op 2454) 

s_xor_b32 s0, s1, s18 // (op 2454) 

s_sub_i32 s1, s0, s18 // (op 2454) 


s_mul_hi_i32 s0, s1, s12 // (op 2454) 
s_add_i32 s18, s0, s1 // (op 2454) 

s_ashr_i32 s0, s18, 31 // (op 2454) 
s_cmp_eq_i32 s12, 0 // (op 2454) 
s_mov_b32 s19, scc // (op 2454) 
s_cmp_eq_i32 s19, 0 // (op 2454) 
s_cselect_b32 s22, 0, -1 // (op 2454) 

s_and_b32 s19, s13, 31 // (op 2454) 
s_lshl_b32 s23, 1, s19 // (op 2454) 
s_add_i32 s24, s23, s22 // (op 2454) 


s_and_b32 s25, s0, s24 // (op 2454) 


s_add_i32 s0, s18, s25 // (op 2454) 


s_ashr_i32 s18, s0, s19 // (op 2454) 


s_lshl_b32 s0, s13, 24 // (op 2454) 
s_ashr_i32 s19, s0, 31 // (op 2454) 

s_xor_b32 s0, s18, s19 // (op 2454) 

s_sub_i32 s18, s0, s19 // (op 2454) 


s_mul_i32 s0, s1, s15 // (op 2454) 

s_sub_u32 s1, s16, s0 // (op 2454) 

s_mul_i32 s0, s18, s15 // (op 2454) 
s_add_u32 s0, s0, s1 // (op 2454) 


s_mul_hi_i32 s1, s16, s9 // (op 2454) 
s_add_i32 s18, s1, s16 // (op 2454) 

s_ashr_i32 s1, s18, 31 // (op 2454) 
s_cmp_eq_i32 s9, 0 // (op 2454) 
s_mov_b32 s19, scc // (op 2454) 
s_cmp_eq_i32 s19, 0 // (op 2454) 
s_cselect_b32 s22, 0, -1 // (op 2454) 

s_and_b32 s19, s46, 31 // (op 2454) 
s_lshl_b32 s23, 1, s19 // (op 2454) 
s_add_i32 s24, s23, s22 // (op 2454) 


s_and_b32 s25, s1, s24 // (op 2454) 


s_add_i32 s1, s18, s25 // (op 2454) 


s_ashr_i32 s18, s1, s19 // (op 2454) 


s_lshl_b32 s1, s46, 24 // (op 2454) 
s_ashr_i32 s19, s1, 31 // (op 2454) 

s_xor_b32 s1, s18, s19 // (op 2454) 

s_sub_i32 s18, s1, s19 // (op 2454) 


s_add_u32 s22, s6, 256 // (op 2454) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2454) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2454) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2454) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2454) 

s_mul_i32 s1, s15, s22 // (op 2454) 

s_mul_i32 s19, s8, s1 // (op 2454) 

s_mul_i32 s1, s18, s19 // (op 2454) 

s_sub_u32 s19, s16, s1 // (op 2454) 

s_mov_b32 s1, s19 // (op 2454) 
s_mul_hi_i32 s22, s1, s14 // (op 2454) 
s_add_i32 s23, s22, s1 // (op 2454) 


s_ashr_i32 s1, s23, 31 // (op 2454) 
s_cmp_eq_i32 s14, 0 // (op 2454) 
s_mov_b32 s22, scc // (op 2454) 
s_cmp_eq_i32 s22, 0 // (op 2454) 
s_cselect_b32 s24, 0, -1 // (op 2454) 

s_and_b32 s22, s48, 31 // (op 2454) 
s_lshl_b32 s25, 1, s22 // (op 2454) 
s_add_i32 s49, s25, s24 // (op 2454) 


s_and_b32 s24, s1, s49 // (op 2454) 


s_add_i32 s1, s23, s24 // (op 2454) 


s_ashr_i32 s23, s1, s22 // (op 2454) 


s_lshl_b32 s1, s48, 24 // (op 2454) 
s_ashr_i32 s22, s1, 31 // (op 2454) 

s_xor_b32 s1, s23, s22 // (op 2454) 

s_sub_i32 s23, s1, s22 // (op 2454) 


s_mul_i32 s1, s47, s15 // (op 2454) 
s_add_u32 s24, s20, 64 // (op 2454) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2454) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 2454) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 2454) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 2454) 

s_sub_i32 s22, s24, s1 // (op 2454) 

s_mul_i32 s24, s23, s22 // (op 2454) 


s_sub_u32 s25, s19, s24 // (op 2454) 


s_add_u32 s19, s25, s1 // (op 2454) 


s_cmp_eq_i32 s18, 0 // (op 2454) 
s_mov_b32 s1, scc // (op 2454) 

s_cmp_eq_i32 s1, 0 // (op 2454) 
s_cselect_b32 s18, s19, s0 // (op 2454) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 2454) 


v_mov_b32 v19, 0 // (op 2454) convert
v_mov_b32 v18, v16 // (op 2454) convert

v_mul_lo_u32 v17, v15, s27 // (op 2454) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 2454) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 2454) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 2454) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 2454) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 2454) most significant half



s_mul_i32 s1, s26, s31 // (op 2456) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2456) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2456) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2456) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2456) most significant: sum
s_add_u32 s1, s1, s19 // (op 2456) most significant: sum


s_mov_b32 s52, s28 // (op 2456) reg expression
s_mov_b32 s53, s29 // (op 2456) reg expression
s_mov_b32 s54, s0 // (op 2456) reg expression
s_mov_b32 s55, 131072 // (op 2456) call()

s_mul_hi_i32 s0, s16, s10 // (op 2473) 
v_and_b32 v16, -4, v1 // (op 2473) 
s_add_i32 s1, s0, s16 // (op 2473) 

v_sub_u32 v17, v1, v16 // (op 2473) 

s_ashr_i32 s0, s1, 31 // (op 2473) 
v_lshlrev_b32 v16, 4, v17 // (op 2473) 

s_cmp_eq_i32 s10, 0 // (op 2473) 
s_mov_b32 s18, scc // (op 2473) 
v_lshrrev_b32 v17, 2, v1 // (op 2473) 
s_cmp_eq_i32 s18, 0 // (op 2473) 
s_cselect_b32 s19, 0, -1 // (op 2473) 

s_and_b32 s18, s11, 31 // (op 2473) 
s_lshl_b32 s22, 1, s18 // (op 2473) 
s_add_i32 s23, s22, s19 // (op 2473) 


s_and_b32 s19, s0, s23 // (op 2473) 


s_add_i32 s0, s1, s19 // (op 2473) 


s_ashr_i32 s1, s0, s18 // (op 2473) 


s_lshl_b32 s0, s11, 24 // (op 2473) 
s_ashr_i32 s18, s0, 31 // (op 2473) 

s_xor_b32 s0, s1, s18 // (op 2473) 

s_sub_i32 s1, s0, s18 // (op 2473) 


s_mul_hi_i32 s0, s1, s12 // (op 2473) 
s_add_i32 s18, s0, s1 // (op 2473) 

s_ashr_i32 s0, s18, 31 // (op 2473) 
s_cmp_eq_i32 s12, 0 // (op 2473) 
s_mov_b32 s19, scc // (op 2473) 
s_cmp_eq_i32 s19, 0 // (op 2473) 
s_cselect_b32 s22, 0, -1 // (op 2473) 

s_and_b32 s19, s13, 31 // (op 2473) 
s_lshl_b32 s23, 1, s19 // (op 2473) 
s_add_i32 s24, s23, s22 // (op 2473) 


s_and_b32 s25, s0, s24 // (op 2473) 


s_add_i32 s0, s18, s25 // (op 2473) 


s_ashr_i32 s18, s0, s19 // (op 2473) 


s_lshl_b32 s0, s13, 24 // (op 2473) 
s_ashr_i32 s19, s0, 31 // (op 2473) 

s_xor_b32 s0, s18, s19 // (op 2473) 

s_sub_i32 s18, s0, s19 // (op 2473) 


s_mul_i32 s0, s1, s15 // (op 2473) 

s_sub_u32 s1, s16, s0 // (op 2473) 

s_mul_i32 s0, s18, s15 // (op 2473) 
s_add_u32 s0, s0, s1 // (op 2473) 


s_mul_hi_i32 s1, s16, s9 // (op 2473) 
s_add_i32 s18, s1, s16 // (op 2473) 

s_ashr_i32 s1, s18, 31 // (op 2473) 
s_cmp_eq_i32 s9, 0 // (op 2473) 
s_mov_b32 s19, scc // (op 2473) 
s_cmp_eq_i32 s19, 0 // (op 2473) 
s_cselect_b32 s22, 0, -1 // (op 2473) 

s_and_b32 s19, s46, 31 // (op 2473) 
s_lshl_b32 s23, 1, s19 // (op 2473) 
s_add_i32 s24, s23, s22 // (op 2473) 


s_and_b32 s25, s1, s24 // (op 2473) 


s_add_i32 s1, s18, s25 // (op 2473) 


s_ashr_i32 s18, s1, s19 // (op 2473) 


s_lshl_b32 s1, s46, 24 // (op 2473) 
s_ashr_i32 s19, s1, 31 // (op 2473) 

s_xor_b32 s1, s18, s19 // (op 2473) 

s_sub_i32 s18, s1, s19 // (op 2473) 


s_add_u32 s50, s6, 256 // (op 2473) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2473) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2473) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2473) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2473) 

s_mul_i32 s1, s15, s24 // (op 2473) 

s_mul_i32 s19, s8, s1 // (op 2473) 

s_mul_i32 s1, s18, s19 // (op 2473) 

s_sub_u32 s19, s16, s1 // (op 2473) 

s_mov_b32 s1, s19 // (op 2473) 
s_mul_hi_i32 s22, s1, s14 // (op 2473) 
s_add_i32 s23, s22, s1 // (op 2473) 


s_ashr_i32 s1, s23, 31 // (op 2473) 
s_cmp_eq_i32 s14, 0 // (op 2473) 
s_mov_b32 s22, scc // (op 2473) 
s_cmp_eq_i32 s22, 0 // (op 2473) 
s_cselect_b32 s24, 0, -1 // (op 2473) 

s_and_b32 s22, s48, 31 // (op 2473) 
s_lshl_b32 s25, 1, s22 // (op 2473) 
s_add_i32 s49, s25, s24 // (op 2473) 


s_and_b32 s24, s1, s49 // (op 2473) 


s_add_i32 s1, s23, s24 // (op 2473) 


s_ashr_i32 s23, s1, s22 // (op 2473) 


s_lshl_b32 s1, s48, 24 // (op 2473) 
s_ashr_i32 s22, s1, 31 // (op 2473) 

s_xor_b32 s1, s23, s22 // (op 2473) 

s_sub_i32 s23, s1, s22 // (op 2473) 


s_mul_i32 s1, s47, s15 // (op 2473) 
s_add_u32 s24, s20, 64 // (op 2473) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2473) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2473) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2473) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2473) 

s_sub_i32 s22, s24, s1 // (op 2473) 

s_mul_i32 s24, s23, s22 // (op 2473) 


s_sub_u32 s25, s19, s24 // (op 2473) 


s_add_u32 s19, s25, s1 // (op 2473) 


s_cmp_eq_i32 s18, 0 // (op 2473) 
s_mov_b32 s1, scc // (op 2473) 

s_cmp_eq_i32 s1, 0 // (op 2473) 
s_cselect_b32 s18, s19, s0 // (op 2473) 



v_lshl_add_u32 v18, s18, 6, v16 // (op 2473) 


v_mov_b32 v21, 0 // (op 2473) convert
v_mov_b32 v20, v18 // (op 2473) convert

v_mul_lo_u32 v19, v17, s27 // (op 2473) most significant: low of low * high
v_mul_hi_u32 v16, v17, s26 // (op 2473) most significant: high of low * low
v_mul_lo_u32 v18, v17, s26 // (op 2473) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 2473) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v18 // (op 2473) least significant half
v_addc_co_u32 v17, s[0:1], v21, v19, s[0:1] // (op 2473) most significant half



s_mul_i32 s1, s26, s31 // (op 2475) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2475) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2475) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2475) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2475) most significant: sum
s_add_u32 s1, s1, s19 // (op 2475) most significant: sum


s_mov_b32 s52, s28 // (op 2475) reg expression
s_mov_b32 s53, s29 // (op 2475) reg expression
s_mov_b32 s54, s0 // (op 2475) reg expression
s_mov_b32 s55, 131072 // (op 2475) call()

s_mul_hi_i32 s0, s16, s10 // (op 2518) 
v_and_b32 v18, 4294967294, v1 // (op 2518) 
s_add_i32 s1, s0, s16 // (op 2518) 

v_and_b32 v19, -2, v1 // (op 2518) 
s_ashr_i32 s0, s1, 31 // (op 2518) 
v_sub_u32 v20, v1, v19 // (op 2518) 

s_cmp_eq_i32 s10, 0 // (op 2518) 
s_mov_b32 s18, scc // (op 2518) 
v_lshlrev_b32 v19, 6, v20 // (op 2518) 

s_cmp_eq_i32 s18, 0 // (op 2518) 
s_cselect_b32 s19, 0, -1 // (op 2518) 

v_mov_b32 v21, 0 // (op 2518) convert
v_mov_b32 v20, v19 // (op 2518) convert

s_and_b32 s18, s11, 31 // (op 2518) 
s_lshl_b32 s22, 1, s18 // (op 2518) 
s_add_i32 s23, s22, s19 // (op 2518) 


s_and_b32 s19, s0, s23 // (op 2518) 


s_add_i32 s0, s1, s19 // (op 2518) 


s_ashr_i32 s1, s0, s18 // (op 2518) 


s_lshl_b32 s0, s11, 24 // (op 2518) 
s_ashr_i32 s18, s0, 31 // (op 2518) 

s_xor_b32 s0, s1, s18 // (op 2518) 

s_sub_i32 s1, s0, s18 // (op 2518) 


s_mul_hi_i32 s0, s1, s12 // (op 2518) 
s_add_i32 s18, s0, s1 // (op 2518) 

s_ashr_i32 s0, s18, 31 // (op 2518) 
s_cmp_eq_i32 s12, 0 // (op 2518) 
s_mov_b32 s19, scc // (op 2518) 
s_cmp_eq_i32 s19, 0 // (op 2518) 
s_cselect_b32 s22, 0, -1 // (op 2518) 

s_and_b32 s19, s13, 31 // (op 2518) 
s_lshl_b32 s23, 1, s19 // (op 2518) 
s_add_i32 s24, s23, s22 // (op 2518) 


s_and_b32 s25, s0, s24 // (op 2518) 


s_add_i32 s0, s18, s25 // (op 2518) 


s_ashr_i32 s18, s0, s19 // (op 2518) 


s_lshl_b32 s0, s13, 24 // (op 2518) 
s_ashr_i32 s19, s0, 31 // (op 2518) 

s_xor_b32 s0, s18, s19 // (op 2518) 

s_sub_i32 s18, s0, s19 // (op 2518) 


s_add_u32 s50, s6, 256 // (op 2518) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2518) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2518) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2518) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2518) 

s_mul_i32 s0, s18, s24 // (op 2518) 

s_sub_i32 s18, s1, s0 // (op 2518) 


s_mul_hi_i32 s19, s16, s9 // (op 2518) 
s_add_i32 s0, s19, s16 // (op 2518) 

s_ashr_i32 s1, s0, 31 // (op 2518) 
s_cmp_eq_i32 s9, 0 // (op 2518) 
s_mov_b32 s19, scc // (op 2518) 
s_cmp_eq_i32 s19, 0 // (op 2518) 
s_cselect_b32 s22, 0, -1 // (op 2518) 

s_and_b32 s19, s46, 31 // (op 2518) 
s_lshl_b32 s23, 1, s19 // (op 2518) 
s_add_i32 s49, s23, s22 // (op 2518) 


s_and_b32 s22, s1, s49 // (op 2518) 


s_add_i32 s1, s0, s22 // (op 2518) 


s_ashr_i32 s0, s1, s19 // (op 2518) 


s_lshl_b32 s1, s46, 24 // (op 2518) 
s_ashr_i32 s19, s1, 31 // (op 2518) 

s_xor_b32 s1, s0, s19 // (op 2518) 

s_sub_i32 s0, s1, s19 // (op 2518) 


s_mul_i32 s1, s15, s24 // (op 2518) 

s_mul_i32 s19, s8, s1 // (op 2518) 

s_mul_i32 s1, s0, s19 // (op 2518) 

s_sub_u32 s19, s16, s1 // (op 2518) 

s_mul_hi_i32 s1, s19, s14 // (op 2518) 
s_add_i32 s22, s1, s19 // (op 2518) 


s_ashr_i32 s1, s22, 31 // (op 2518) 
s_cmp_eq_i32 s14, 0 // (op 2518) 
s_mov_b32 s19, scc // (op 2518) 
s_cmp_eq_i32 s19, 0 // (op 2518) 
s_cselect_b32 s23, 0, -1 // (op 2518) 

s_and_b32 s19, s48, 31 // (op 2518) 
s_lshl_b32 s24, 1, s19 // (op 2518) 
s_add_i32 s25, s24, s23 // (op 2518) 


s_and_b32 s23, s1, s25 // (op 2518) 


s_add_i32 s1, s22, s23 // (op 2518) 


s_ashr_i32 s22, s1, s19 // (op 2518) 


s_lshl_b32 s1, s48, 24 // (op 2518) 
s_ashr_i32 s19, s1, 31 // (op 2518) 

s_xor_b32 s1, s22, s19 // (op 2518) 

s_sub_i32 s22, s1, s19 // (op 2518) 


s_cmp_eq_i32 s0, 0 // (op 2518) 
s_mov_b32 s1, scc // (op 2518) 

s_cmp_eq_i32 s1, 0 // (op 2518) 
s_cselect_b32 s0, s22, s18 // (op 2518) 



v_lshl_add_u32 v19, s0, 8, v18 // (op 2518) 


v_mul_lo_u32 v23, v19, s33 // (op 2518) most significant: low of low * high
v_mul_hi_u32 v18, v19, s32 // (op 2518) most significant: high of low * low
v_mul_lo_u32 v22, v19, s32 // (op 2518) least significant: low of low * low
v_add_u32 v23, v23, v18 // (op 2518) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v22 // (op 2518) least significant half
v_addc_co_u32 v19, s[0:1], v21, v23, s[0:1] // (op 2518) most significant half



v_mul_lo_u32 v21, v19, 6 // (op 2518) most significant: low of high * low
v_mul_hi_u32 v22, v18, 6 // (op 2518) most significant: high of low * low
v_mul_lo_u32 v20, v18, 6 // (op 2518) least significant: low of low * low
v_add_u32 v21, v21, v22 // (op 2518) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 2518) 

v_mov_b32 v20, v18 // (op 2518) convert

s_mul_i32 s1, s7, 6 // (op 2520) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 2520) least significant
s_mul_hi_u32 s18, s6, 6 // (op 2520) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2520) most significant: sum

s_mul_i32 s19, s32, s1 // (op 2520) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 2520) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 2520) least significant
s_mul_hi_u32 s23, s32, s0 // (op 2520) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 2520) most significant: sum
s_add_u32 s19, s19, s23 // (op 2520) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2520) 

s_mov_b32 s64, s34 // (op 2520) reg expression
s_mov_b32 s65, s35 // (op 2520) reg expression
s_mov_b32 s66, s0 // (op 2520) reg expression
s_mov_b32 s67, 131072 // (op 2520) call()

s_mul_hi_i32 s0, s16, s10 // (op 2537) 
v_and_b32 v18, 4294967294, v1 // (op 2537) 
s_add_i32 s1, s0, s16 // (op 2537) 

v_and_b32 v19, -2, v1 // (op 2537) 
s_ashr_i32 s0, s1, 31 // (op 2537) 
v_sub_u32 v21, v1, v19 // (op 2537) 

s_cmp_eq_i32 s10, 0 // (op 2537) 
s_mov_b32 s18, scc // (op 2537) 
v_lshlrev_b32 v19, 6, v21 // (op 2537) 

s_cmp_eq_i32 s18, 0 // (op 2537) 
s_cselect_b32 s19, 0, -1 // (op 2537) 

v_mov_b32 v23, 0 // (op 2537) convert
v_mov_b32 v22, v19 // (op 2537) convert

s_and_b32 s18, s11, 31 // (op 2537) 
s_lshl_b32 s22, 1, s18 // (op 2537) 
s_add_i32 s23, s22, s19 // (op 2537) 


s_and_b32 s19, s0, s23 // (op 2537) 


s_add_i32 s0, s1, s19 // (op 2537) 


s_ashr_i32 s1, s0, s18 // (op 2537) 


s_lshl_b32 s0, s11, 24 // (op 2537) 
s_ashr_i32 s18, s0, 31 // (op 2537) 

s_xor_b32 s0, s1, s18 // (op 2537) 

s_sub_i32 s1, s0, s18 // (op 2537) 


s_mul_hi_i32 s0, s1, s12 // (op 2537) 
s_add_i32 s18, s0, s1 // (op 2537) 

s_ashr_i32 s0, s18, 31 // (op 2537) 
s_cmp_eq_i32 s12, 0 // (op 2537) 
s_mov_b32 s19, scc // (op 2537) 
s_cmp_eq_i32 s19, 0 // (op 2537) 
s_cselect_b32 s22, 0, -1 // (op 2537) 

s_and_b32 s19, s13, 31 // (op 2537) 
s_lshl_b32 s23, 1, s19 // (op 2537) 
s_add_i32 s24, s23, s22 // (op 2537) 


s_and_b32 s25, s0, s24 // (op 2537) 


s_add_i32 s0, s18, s25 // (op 2537) 


s_ashr_i32 s18, s0, s19 // (op 2537) 


s_lshl_b32 s0, s13, 24 // (op 2537) 
s_ashr_i32 s19, s0, 31 // (op 2537) 

s_xor_b32 s0, s18, s19 // (op 2537) 

s_sub_i32 s18, s0, s19 // (op 2537) 


s_add_u32 s50, s6, 256 // (op 2537) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2537) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2537) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2537) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2537) 

s_mul_i32 s0, s18, s24 // (op 2537) 

s_sub_i32 s18, s1, s0 // (op 2537) 


s_mul_hi_i32 s19, s16, s9 // (op 2537) 
s_add_i32 s0, s19, s16 // (op 2537) 

s_ashr_i32 s1, s0, 31 // (op 2537) 
s_cmp_eq_i32 s9, 0 // (op 2537) 
s_mov_b32 s19, scc // (op 2537) 
s_cmp_eq_i32 s19, 0 // (op 2537) 
s_cselect_b32 s22, 0, -1 // (op 2537) 

s_and_b32 s19, s46, 31 // (op 2537) 
s_lshl_b32 s23, 1, s19 // (op 2537) 
s_add_i32 s49, s23, s22 // (op 2537) 


s_and_b32 s22, s1, s49 // (op 2537) 


s_add_i32 s1, s0, s22 // (op 2537) 


s_ashr_i32 s0, s1, s19 // (op 2537) 


s_lshl_b32 s1, s46, 24 // (op 2537) 
s_ashr_i32 s19, s1, 31 // (op 2537) 

s_xor_b32 s1, s0, s19 // (op 2537) 

s_sub_i32 s0, s1, s19 // (op 2537) 


s_mul_i32 s1, s15, s24 // (op 2537) 

s_mul_i32 s19, s8, s1 // (op 2537) 

s_mul_i32 s1, s0, s19 // (op 2537) 

s_sub_u32 s19, s16, s1 // (op 2537) 

s_mul_hi_i32 s1, s19, s14 // (op 2537) 
s_add_i32 s22, s1, s19 // (op 2537) 


s_ashr_i32 s1, s22, 31 // (op 2537) 
s_cmp_eq_i32 s14, 0 // (op 2537) 
s_mov_b32 s19, scc // (op 2537) 
s_cmp_eq_i32 s19, 0 // (op 2537) 
s_cselect_b32 s23, 0, -1 // (op 2537) 

s_and_b32 s19, s48, 31 // (op 2537) 
s_lshl_b32 s24, 1, s19 // (op 2537) 
s_add_i32 s25, s24, s23 // (op 2537) 


s_and_b32 s23, s1, s25 // (op 2537) 


s_add_i32 s1, s22, s23 // (op 2537) 


s_ashr_i32 s22, s1, s19 // (op 2537) 


s_lshl_b32 s1, s48, 24 // (op 2537) 
s_ashr_i32 s19, s1, 31 // (op 2537) 

s_xor_b32 s1, s22, s19 // (op 2537) 

s_sub_i32 s22, s1, s19 // (op 2537) 


s_cmp_eq_i32 s0, 0 // (op 2537) 
s_mov_b32 s1, scc // (op 2537) 

s_cmp_eq_i32 s1, 0 // (op 2537) 
s_cselect_b32 s0, s22, s18 // (op 2537) 



v_lshl_add_u32 v19, s0, 8, v18 // (op 2537) 


v_mul_lo_u32 v25, v19, s33 // (op 2537) most significant: low of low * high
v_mul_hi_u32 v18, v19, s32 // (op 2537) most significant: high of low * low
v_mul_lo_u32 v24, v19, s32 // (op 2537) least significant: low of low * low
v_add_u32 v25, v25, v18 // (op 2537) most significant: sum


v_add_co_u32 v18, s[0:1], v22, v24 // (op 2537) least significant half
v_addc_co_u32 v19, s[0:1], v23, v25, s[0:1] // (op 2537) most significant half



v_mul_lo_u32 v23, v19, 6 // (op 2537) most significant: low of high * low
v_mul_hi_u32 v21, v18, 6 // (op 2537) most significant: high of low * low
v_mul_lo_u32 v22, v18, 6 // (op 2537) least significant: low of low * low
v_add_u32 v23, v23, v21 // (op 2537) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[22:23] // (op 2537) 

v_mov_b32 v21, v18 // (op 2537) convert

s_mul_i32 s1, s7, 6 // (op 2539) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 2539) least significant
s_mul_hi_u32 s18, s6, 6 // (op 2539) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2539) most significant: sum

s_mul_i32 s19, s32, s1 // (op 2539) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 2539) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 2539) least significant
s_mul_hi_u32 s23, s32, s0 // (op 2539) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 2539) most significant: sum
s_add_u32 s19, s19, s23 // (op 2539) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2539) 

s_mov_b32 s64, s34 // (op 2539) reg expression
s_mov_b32 s65, s35 // (op 2539) reg expression
s_mov_b32 s66, s0 // (op 2539) reg expression
s_mov_b32 s67, 131072 // (op 2539) call()

s_mul_hi_i32 s0, s16, s10 // (op 2582) 
v_and_b32 v18, -64, v1 // (op 2582) 
s_add_i32 s1, s0, s16 // (op 2582) 

v_sub_u32 v19, v1, v18 // (op 2582) 

s_ashr_i32 s0, s1, 31 // (op 2582) 
v_lshrrev_b32 v18, 6, v1 // (op 2582) 
s_cmp_eq_i32 s10, 0 // (op 2582) 
s_mov_b32 s18, scc // (op 2582) 
s_cmp_eq_i32 s18, 0 // (op 2582) 
s_cselect_b32 s19, 0, -1 // (op 2582) 

s_and_b32 s18, s11, 31 // (op 2582) 
s_lshl_b32 s22, 1, s18 // (op 2582) 
s_add_i32 s23, s22, s19 // (op 2582) 


s_and_b32 s19, s0, s23 // (op 2582) 


s_add_i32 s0, s1, s19 // (op 2582) 


s_ashr_i32 s1, s0, s18 // (op 2582) 


s_lshl_b32 s0, s11, 24 // (op 2582) 
s_ashr_i32 s18, s0, 31 // (op 2582) 

s_xor_b32 s0, s1, s18 // (op 2582) 

s_sub_i32 s1, s0, s18 // (op 2582) 


s_mul_hi_i32 s0, s1, s12 // (op 2582) 
s_add_i32 s18, s0, s1 // (op 2582) 

s_ashr_i32 s0, s18, 31 // (op 2582) 
s_cmp_eq_i32 s12, 0 // (op 2582) 
s_mov_b32 s19, scc // (op 2582) 
s_cmp_eq_i32 s19, 0 // (op 2582) 
s_cselect_b32 s22, 0, -1 // (op 2582) 

s_and_b32 s19, s13, 31 // (op 2582) 
s_lshl_b32 s23, 1, s19 // (op 2582) 
s_add_i32 s24, s23, s22 // (op 2582) 


s_and_b32 s25, s0, s24 // (op 2582) 


s_add_i32 s0, s18, s25 // (op 2582) 


s_ashr_i32 s18, s0, s19 // (op 2582) 


s_lshl_b32 s0, s13, 24 // (op 2582) 
s_ashr_i32 s19, s0, 31 // (op 2582) 

s_xor_b32 s0, s18, s19 // (op 2582) 

s_sub_i32 s18, s0, s19 // (op 2582) 


s_mul_i32 s0, s1, s15 // (op 2582) 

s_sub_u32 s1, s16, s0 // (op 2582) 

s_mul_i32 s0, s18, s15 // (op 2582) 
s_add_u32 s0, s0, s1 // (op 2582) 


s_mul_hi_i32 s1, s16, s9 // (op 2582) 
s_add_i32 s18, s1, s16 // (op 2582) 

s_ashr_i32 s1, s18, 31 // (op 2582) 
s_cmp_eq_i32 s9, 0 // (op 2582) 
s_mov_b32 s19, scc // (op 2582) 
s_cmp_eq_i32 s19, 0 // (op 2582) 
s_cselect_b32 s22, 0, -1 // (op 2582) 

s_and_b32 s19, s46, 31 // (op 2582) 
s_lshl_b32 s23, 1, s19 // (op 2582) 
s_add_i32 s24, s23, s22 // (op 2582) 


s_and_b32 s25, s1, s24 // (op 2582) 


s_add_i32 s1, s18, s25 // (op 2582) 


s_ashr_i32 s18, s1, s19 // (op 2582) 


s_lshl_b32 s1, s46, 24 // (op 2582) 
s_ashr_i32 s19, s1, 31 // (op 2582) 

s_xor_b32 s1, s18, s19 // (op 2582) 

s_sub_i32 s18, s1, s19 // (op 2582) 


s_add_u32 s50, s6, 256 // (op 2582) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2582) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2582) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2582) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2582) 

s_mul_i32 s1, s15, s24 // (op 2582) 

s_mul_i32 s19, s8, s1 // (op 2582) 

s_mul_i32 s1, s18, s19 // (op 2582) 

s_sub_u32 s19, s16, s1 // (op 2582) 

s_mov_b32 s1, s19 // (op 2582) 
s_mul_hi_i32 s22, s1, s14 // (op 2582) 
s_add_i32 s23, s22, s1 // (op 2582) 


s_ashr_i32 s1, s23, 31 // (op 2582) 
s_cmp_eq_i32 s14, 0 // (op 2582) 
s_mov_b32 s22, scc // (op 2582) 
s_cmp_eq_i32 s22, 0 // (op 2582) 
s_cselect_b32 s24, 0, -1 // (op 2582) 

s_and_b32 s22, s48, 31 // (op 2582) 
s_lshl_b32 s25, 1, s22 // (op 2582) 
s_add_i32 s49, s25, s24 // (op 2582) 


s_and_b32 s24, s1, s49 // (op 2582) 


s_add_i32 s1, s23, s24 // (op 2582) 


s_ashr_i32 s23, s1, s22 // (op 2582) 


s_lshl_b32 s1, s48, 24 // (op 2582) 
s_ashr_i32 s22, s1, 31 // (op 2582) 

s_xor_b32 s1, s23, s22 // (op 2582) 

s_sub_i32 s23, s1, s22 // (op 2582) 


s_mul_i32 s1, s47, s15 // (op 2582) 
s_add_u32 s24, s20, 64 // (op 2582) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2582) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2582) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2582) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2582) 

s_sub_i32 s22, s24, s1 // (op 2582) 

s_mul_i32 s24, s23, s22 // (op 2582) 


s_sub_u32 s25, s19, s24 // (op 2582) 


s_add_u32 s19, s25, s1 // (op 2582) 


s_cmp_eq_i32 s18, 0 // (op 2582) 
s_mov_b32 s1, scc // (op 2582) 

s_cmp_eq_i32 s1, 0 // (op 2582) 
s_cselect_b32 s18, s19, s0 // (op 2582) 



v_lshl_add_u32 v22, s18, 6, v19 // (op 2582) 


v_mul_lo_u32 v19, v18, s36 // (op 2582) 

v_add_u32 v18, v22, v19 // (op 2582) 


s_mul_i32 s1, s36, s41 // (op 2584) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2584) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2584) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2584) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2584) most significant: sum
s_add_u32 s1, s1, s19 // (op 2584) most significant: sum


s_mov_b32 s68, s38 // (op 2584) reg expression
s_mov_b32 s69, s39 // (op 2584) reg expression
s_mov_b32 s70, s0 // (op 2584) reg expression
s_mov_b32 s71, 131072 // (op 2584) call()

s_mul_hi_i32 s0, s16, s10 // (op 2601) 
v_and_b32 v19, -64, v1 // (op 2601) 
s_add_i32 s1, s0, s16 // (op 2601) 

v_sub_u32 v22, v1, v19 // (op 2601) 

s_ashr_i32 s0, s1, 31 // (op 2601) 
v_lshrrev_b32 v19, 6, v1 // (op 2601) 
s_cmp_eq_i32 s10, 0 // (op 2601) 
s_mov_b32 s18, scc // (op 2601) 
s_cmp_eq_i32 s18, 0 // (op 2601) 
s_cselect_b32 s19, 0, -1 // (op 2601) 

s_and_b32 s18, s11, 31 // (op 2601) 
s_lshl_b32 s22, 1, s18 // (op 2601) 
s_add_i32 s23, s22, s19 // (op 2601) 


s_and_b32 s19, s0, s23 // (op 2601) 


s_add_i32 s0, s1, s19 // (op 2601) 


s_ashr_i32 s1, s0, s18 // (op 2601) 


s_lshl_b32 s0, s11, 24 // (op 2601) 
s_ashr_i32 s18, s0, 31 // (op 2601) 

s_xor_b32 s0, s1, s18 // (op 2601) 

s_sub_i32 s1, s0, s18 // (op 2601) 


s_mul_hi_i32 s0, s1, s12 // (op 2601) 
s_add_i32 s18, s0, s1 // (op 2601) 

s_ashr_i32 s0, s18, 31 // (op 2601) 
s_cmp_eq_i32 s12, 0 // (op 2601) 
s_mov_b32 s19, scc // (op 2601) 
s_cmp_eq_i32 s19, 0 // (op 2601) 
s_cselect_b32 s22, 0, -1 // (op 2601) 

s_and_b32 s19, s13, 31 // (op 2601) 
s_lshl_b32 s23, 1, s19 // (op 2601) 
s_add_i32 s24, s23, s22 // (op 2601) 


s_and_b32 s25, s0, s24 // (op 2601) 


s_add_i32 s0, s18, s25 // (op 2601) 


s_ashr_i32 s18, s0, s19 // (op 2601) 


s_lshl_b32 s0, s13, 24 // (op 2601) 
s_ashr_i32 s19, s0, 31 // (op 2601) 

s_xor_b32 s0, s18, s19 // (op 2601) 

s_sub_i32 s18, s0, s19 // (op 2601) 


s_mul_i32 s0, s1, s15 // (op 2601) 

s_sub_u32 s1, s16, s0 // (op 2601) 

s_mul_i32 s0, s18, s15 // (op 2601) 
s_add_u32 s0, s0, s1 // (op 2601) 


s_mul_hi_i32 s1, s16, s9 // (op 2601) 
s_add_i32 s18, s1, s16 // (op 2601) 

s_ashr_i32 s1, s18, 31 // (op 2601) 
s_cmp_eq_i32 s9, 0 // (op 2601) 
s_mov_b32 s19, scc // (op 2601) 
s_cmp_eq_i32 s19, 0 // (op 2601) 
s_cselect_b32 s22, 0, -1 // (op 2601) 

s_and_b32 s19, s46, 31 // (op 2601) 
s_lshl_b32 s23, 1, s19 // (op 2601) 
s_add_i32 s24, s23, s22 // (op 2601) 


s_and_b32 s25, s1, s24 // (op 2601) 


s_add_i32 s1, s18, s25 // (op 2601) 


s_ashr_i32 s18, s1, s19 // (op 2601) 


s_lshl_b32 s1, s46, 24 // (op 2601) 
s_ashr_i32 s19, s1, 31 // (op 2601) 

s_xor_b32 s1, s18, s19 // (op 2601) 

s_sub_i32 s18, s1, s19 // (op 2601) 


s_add_u32 s50, s6, 256 // (op 2601) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2601) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2601) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2601) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2601) 

s_mul_i32 s1, s15, s24 // (op 2601) 

s_mul_i32 s19, s8, s1 // (op 2601) 

s_mul_i32 s1, s18, s19 // (op 2601) 

s_sub_u32 s19, s16, s1 // (op 2601) 

s_mov_b32 s1, s19 // (op 2601) 
s_mul_hi_i32 s22, s1, s14 // (op 2601) 
s_add_i32 s23, s22, s1 // (op 2601) 


s_ashr_i32 s1, s23, 31 // (op 2601) 
s_cmp_eq_i32 s14, 0 // (op 2601) 
s_mov_b32 s22, scc // (op 2601) 
s_cmp_eq_i32 s22, 0 // (op 2601) 
s_cselect_b32 s24, 0, -1 // (op 2601) 

s_and_b32 s22, s48, 31 // (op 2601) 
s_lshl_b32 s25, 1, s22 // (op 2601) 
s_add_i32 s49, s25, s24 // (op 2601) 


s_and_b32 s24, s1, s49 // (op 2601) 


s_add_i32 s1, s23, s24 // (op 2601) 


s_ashr_i32 s23, s1, s22 // (op 2601) 


s_lshl_b32 s1, s48, 24 // (op 2601) 
s_ashr_i32 s22, s1, 31 // (op 2601) 

s_xor_b32 s1, s23, s22 // (op 2601) 

s_sub_i32 s23, s1, s22 // (op 2601) 


s_mul_i32 s1, s47, s15 // (op 2601) 
s_add_u32 s24, s20, 64 // (op 2601) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2601) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2601) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2601) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2601) 

s_sub_i32 s22, s24, s1 // (op 2601) 

s_mul_i32 s24, s23, s22 // (op 2601) 


s_sub_u32 s25, s19, s24 // (op 2601) 


s_add_u32 s19, s25, s1 // (op 2601) 


s_cmp_eq_i32 s18, 0 // (op 2601) 
s_mov_b32 s1, scc // (op 2601) 

s_cmp_eq_i32 s1, 0 // (op 2601) 
s_cselect_b32 s18, s19, s0 // (op 2601) 



v_lshl_add_u32 v23, s18, 6, v22 // (op 2601) 


v_mul_lo_u32 v22, v19, s36 // (op 2601) 

v_add_u32 v19, v23, v22 // (op 2601) 


s_mul_i32 s1, s36, s41 // (op 2603) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2603) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2603) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2603) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2603) most significant: sum
s_add_u32 s1, s1, s19 // (op 2603) most significant: sum


s_mov_b32 s68, s38 // (op 2603) reg expression
s_mov_b32 s69, s39 // (op 2603) reg expression
s_mov_b32 s70, s0 // (op 2603) reg expression
s_mov_b32 s71, 131072 // (op 2603) call()

s_mul_hi_i32 s0, s16, s10 // (op 2646) 
s_add_i32 s1, s0, s16 // (op 2646) 

s_ashr_i32 s0, s1, 31 // (op 2646) 
s_cmp_eq_i32 s10, 0 // (op 2646) 
s_mov_b32 s18, scc // (op 2646) 
s_cmp_eq_i32 s18, 0 // (op 2646) 
s_cselect_b32 s19, 0, -1 // (op 2646) 

s_and_b32 s18, s11, 31 // (op 2646) 
s_lshl_b32 s22, 1, s18 // (op 2646) 
s_add_i32 s23, s22, s19 // (op 2646) 


s_and_b32 s19, s0, s23 // (op 2646) 


s_add_i32 s0, s1, s19 // (op 2646) 


s_ashr_i32 s1, s0, s18 // (op 2646) 


s_lshl_b32 s0, s11, 24 // (op 2646) 
s_ashr_i32 s18, s0, 31 // (op 2646) 

s_xor_b32 s0, s1, s18 // (op 2646) 

s_sub_i32 s1, s0, s18 // (op 2646) 


s_mul_hi_i32 s0, s1, s12 // (op 2646) 
s_add_i32 s18, s0, s1 // (op 2646) 

s_ashr_i32 s0, s18, 31 // (op 2646) 
s_cmp_eq_i32 s12, 0 // (op 2646) 
s_mov_b32 s19, scc // (op 2646) 
s_cmp_eq_i32 s19, 0 // (op 2646) 
s_cselect_b32 s22, 0, -1 // (op 2646) 

s_and_b32 s19, s13, 31 // (op 2646) 
s_lshl_b32 s23, 1, s19 // (op 2646) 
s_add_i32 s24, s23, s22 // (op 2646) 


s_and_b32 s25, s0, s24 // (op 2646) 


s_add_i32 s0, s18, s25 // (op 2646) 


s_ashr_i32 s18, s0, s19 // (op 2646) 


s_lshl_b32 s0, s13, 24 // (op 2646) 
s_ashr_i32 s19, s0, 31 // (op 2646) 

s_xor_b32 s0, s18, s19 // (op 2646) 

s_sub_i32 s18, s0, s19 // (op 2646) 


s_add_u32 s50, s6, 256 // (op 2646) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2646) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2646) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2646) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2646) 

s_mul_i32 s0, s18, s24 // (op 2646) 

s_sub_i32 s18, s1, s0 // (op 2646) 


s_mul_hi_i32 s19, s16, s9 // (op 2646) 
s_add_i32 s0, s19, s16 // (op 2646) 

s_ashr_i32 s1, s0, 31 // (op 2646) 
s_cmp_eq_i32 s9, 0 // (op 2646) 
s_mov_b32 s19, scc // (op 2646) 
s_cmp_eq_i32 s19, 0 // (op 2646) 
s_cselect_b32 s22, 0, -1 // (op 2646) 

s_and_b32 s19, s46, 31 // (op 2646) 
s_lshl_b32 s23, 1, s19 // (op 2646) 
s_add_i32 s49, s23, s22 // (op 2646) 


s_and_b32 s22, s1, s49 // (op 2646) 


s_add_i32 s1, s0, s22 // (op 2646) 


s_ashr_i32 s0, s1, s19 // (op 2646) 


s_lshl_b32 s1, s46, 24 // (op 2646) 
s_ashr_i32 s19, s1, 31 // (op 2646) 

s_xor_b32 s1, s0, s19 // (op 2646) 

s_sub_i32 s0, s1, s19 // (op 2646) 


s_mul_i32 s1, s15, s24 // (op 2646) 

s_mul_i32 s19, s8, s1 // (op 2646) 

s_mul_i32 s1, s0, s19 // (op 2646) 

s_sub_u32 s19, s16, s1 // (op 2646) 

s_mul_hi_i32 s1, s19, s14 // (op 2646) 
s_add_i32 s22, s1, s19 // (op 2646) 


s_ashr_i32 s1, s22, 31 // (op 2646) 
s_cmp_eq_i32 s14, 0 // (op 2646) 
s_mov_b32 s19, scc // (op 2646) 
s_cmp_eq_i32 s19, 0 // (op 2646) 
s_cselect_b32 s23, 0, -1 // (op 2646) 

s_and_b32 s19, s48, 31 // (op 2646) 
s_lshl_b32 s24, 1, s19 // (op 2646) 
s_add_i32 s25, s24, s23 // (op 2646) 


s_and_b32 s23, s1, s25 // (op 2646) 


s_add_i32 s1, s22, s23 // (op 2646) 


s_ashr_i32 s22, s1, s19 // (op 2646) 


s_lshl_b32 s1, s48, 24 // (op 2646) 
s_ashr_i32 s19, s1, 31 // (op 2646) 

s_xor_b32 s1, s22, s19 // (op 2646) 

s_sub_i32 s22, s1, s19 // (op 2646) 


s_cmp_eq_i32 s0, 0 // (op 2646) 
s_mov_b32 s1, scc // (op 2646) 

s_cmp_eq_i32 s1, 0 // (op 2646) 
s_cselect_b32 s0, s22, s18 // (op 2646) 



v_lshl_add_u32 v22, s0, 8, v1 // (op 2646) 

v_mul_lo_u32 v23, v22, s42 // (op 2646) 

s_mul_i32 s1, s42, s7 // (op 2648) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2648) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2648) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2648) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2648) most significant: sum
s_add_u32 s1, s1, s19 // (op 2648) most significant: sum


s_mov_b32 s72, s44 // (op 2648) reg expression
s_mov_b32 s73, s45 // (op 2648) reg expression
s_mov_b32 s74, s0 // (op 2648) reg expression
s_mov_b32 s75, 131072 // (op 2648) call()

s_mul_hi_i32 s0, s16, s10 // (op 2665) 
s_add_i32 s1, s0, s16 // (op 2665) 

s_ashr_i32 s0, s1, 31 // (op 2665) 
s_cmp_eq_i32 s10, 0 // (op 2665) 
s_mov_b32 s18, scc // (op 2665) 
s_cmp_eq_i32 s18, 0 // (op 2665) 
s_cselect_b32 s19, 0, -1 // (op 2665) 

s_and_b32 s18, s11, 31 // (op 2665) 
s_lshl_b32 s22, 1, s18 // (op 2665) 
s_add_i32 s23, s22, s19 // (op 2665) 


s_and_b32 s19, s0, s23 // (op 2665) 


s_add_i32 s0, s1, s19 // (op 2665) 


s_ashr_i32 s1, s0, s18 // (op 2665) 


s_lshl_b32 s0, s11, 24 // (op 2665) 
s_ashr_i32 s18, s0, 31 // (op 2665) 

s_xor_b32 s0, s1, s18 // (op 2665) 

s_sub_i32 s1, s0, s18 // (op 2665) 


s_mul_hi_i32 s0, s1, s12 // (op 2665) 
s_add_i32 s18, s0, s1 // (op 2665) 

s_ashr_i32 s0, s18, 31 // (op 2665) 
s_cmp_eq_i32 s12, 0 // (op 2665) 
s_mov_b32 s19, scc // (op 2665) 
s_cmp_eq_i32 s19, 0 // (op 2665) 
s_cselect_b32 s22, 0, -1 // (op 2665) 

s_and_b32 s19, s13, 31 // (op 2665) 
s_lshl_b32 s23, 1, s19 // (op 2665) 
s_add_i32 s24, s23, s22 // (op 2665) 


s_and_b32 s25, s0, s24 // (op 2665) 


s_add_i32 s0, s18, s25 // (op 2665) 


s_ashr_i32 s18, s0, s19 // (op 2665) 


s_lshl_b32 s0, s13, 24 // (op 2665) 
s_ashr_i32 s19, s0, 31 // (op 2665) 

s_xor_b32 s0, s18, s19 // (op 2665) 

s_sub_i32 s18, s0, s19 // (op 2665) 


s_add_u32 s50, s6, 256 // (op 2665) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2665) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2665) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2665) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2665) 

s_mul_i32 s0, s18, s24 // (op 2665) 

s_sub_i32 s18, s1, s0 // (op 2665) 


s_mul_hi_i32 s19, s16, s9 // (op 2665) 
s_add_i32 s0, s19, s16 // (op 2665) 

s_ashr_i32 s1, s0, 31 // (op 2665) 
s_cmp_eq_i32 s9, 0 // (op 2665) 
s_mov_b32 s19, scc // (op 2665) 
s_cmp_eq_i32 s19, 0 // (op 2665) 
s_cselect_b32 s22, 0, -1 // (op 2665) 

s_and_b32 s19, s46, 31 // (op 2665) 
s_lshl_b32 s23, 1, s19 // (op 2665) 
s_add_i32 s49, s23, s22 // (op 2665) 


s_and_b32 s22, s1, s49 // (op 2665) 


s_add_i32 s1, s0, s22 // (op 2665) 


s_ashr_i32 s0, s1, s19 // (op 2665) 


s_lshl_b32 s1, s46, 24 // (op 2665) 
s_ashr_i32 s19, s1, 31 // (op 2665) 

s_xor_b32 s1, s0, s19 // (op 2665) 

s_sub_i32 s0, s1, s19 // (op 2665) 


s_mul_i32 s1, s15, s24 // (op 2665) 

s_mul_i32 s19, s8, s1 // (op 2665) 

s_mul_i32 s1, s0, s19 // (op 2665) 

s_sub_u32 s19, s16, s1 // (op 2665) 

s_mul_hi_i32 s1, s19, s14 // (op 2665) 
s_add_i32 s22, s1, s19 // (op 2665) 


s_ashr_i32 s1, s22, 31 // (op 2665) 
s_cmp_eq_i32 s14, 0 // (op 2665) 
s_mov_b32 s19, scc // (op 2665) 
s_cmp_eq_i32 s19, 0 // (op 2665) 
s_cselect_b32 s23, 0, -1 // (op 2665) 

s_and_b32 s19, s48, 31 // (op 2665) 
s_lshl_b32 s24, 1, s19 // (op 2665) 
s_add_i32 s25, s24, s23 // (op 2665) 


s_and_b32 s23, s1, s25 // (op 2665) 


s_add_i32 s1, s22, s23 // (op 2665) 


s_ashr_i32 s22, s1, s19 // (op 2665) 


s_lshl_b32 s1, s48, 24 // (op 2665) 
s_ashr_i32 s19, s1, 31 // (op 2665) 

s_xor_b32 s1, s22, s19 // (op 2665) 

s_sub_i32 s22, s1, s19 // (op 2665) 


s_cmp_eq_i32 s0, 0 // (op 2665) 
s_mov_b32 s1, scc // (op 2665) 

s_cmp_eq_i32 s1, 0 // (op 2665) 
s_cselect_b32 s0, s22, s18 // (op 2665) 



v_lshl_add_u32 v22, s0, 8, v1 // (op 2665) 

v_mul_lo_u32 v24, v22, s42 // (op 2665) 

s_mul_i32 s1, s42, s7 // (op 2667) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2667) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2667) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2667) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2667) most significant: sum
s_add_u32 s1, s1, s19 // (op 2667) most significant: sum


s_mov_b32 s72, s44 // (op 2667) reg expression
s_mov_b32 s73, s45 // (op 2667) reg expression
s_mov_b32 s74, s0 // (op 2667) reg expression
s_mov_b32 s75, 131072 // (op 2667) call()


s_mul_i32 s1, s33, 6 // (op 2169) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 2169) least significant
s_mul_hi_u32 s18, s32, 6 // (op 2169) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2169) most significant: sum

v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 2169) 


v_mov_b32 v22, v20 // (op 2169) 
buffer_load_dwordx4 v[28:31], v22, s[64:67], 0 offen // (op 2169) Load value
buffer_load_dwordx2 v[32:33], v22, s[64:67], 0 offen offset:16 // (op 2169) Load value
buffer_load_dwordx4 v[34:37], v22, s[64:67], 0 offen offset:24 // (op 2169) Load value
buffer_load_dwordx2 v[38:39], v22, s[64:67], 0 offen offset:40 // (op 2169) Load value
v_add_u32 v22, v22, v26 // (op 2169) 
buffer_load_dwordx4 v[40:43], v22, s[64:67], 0 offen // (op 2169) Load value
buffer_load_dwordx2 v[44:45], v22, s[64:67], 0 offen offset:16 // (op 2169) Load value
buffer_load_dwordx4 v[46:49], v22, s[64:67], 0 offen offset:24 // (op 2169) Load value
buffer_load_dwordx2 v[50:51], v22, s[64:67], 0 offen offset:40 // (op 2169) Load value





buffer_load_dword v22, v23, s[72:75], 0 offen // (op 2172) Load value
v_and_b32 v25, -4, v1 // (op 2731) 
v_sub_u32 v26, v1, v25 // (op 2731) 

v_lshlrev_b32 v25, 4, v26 // (op 2731) 

v_lshrrev_b32 v26, 2, v1 // (op 2731) 
v_lshl_add_u32 v27, v26, 6, v25 // (op 2731) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2731) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 6 // (op 2407) 
v_mov_b64 v[26:27], s[18:19] // (op 2407) convert


v_mov_b64 v[52:53], v[14:15] // (op 2407) reg expression
s_mov_b32 m0, s0 // (op 2407) reg expression
buffer_load_dwordx4 v52, s[52:55], 0 offen offset: 0 lds // (op 2407) Load value direct to lds
v_add_co_u32 v52, s[18:19], v52, v26 // (op 2407) least significant half
v_addc_co_u32 v53, s[18:19], 0, v53, s[18:19] // (op 2407) most significant half

s_add_u32 m0, m0, 4096 // (op 2407) 
buffer_load_dwordx4 v52, s[52:55], 0 offen offset: 0 lds // (op 2407) Load value direct to lds





v_mov_b32 v25, v18 // (op 2178) reg expression

buffer_load_ubyte v26, v25, s[68:71], 0 offen offset: 0 // (op 2178) Load value


s_mul_i32 s1, s33, 6 // (op 2181) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 2181) least significant
s_mul_hi_u32 s18, s32, 6 // (op 2181) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2181) most significant: sum

v_lshrrev_b64 v[52:53], 3, s[0:1] // (op 2181) 


v_mov_b32 v25, v20 // (op 2181) 
buffer_load_dwordx4 v[54:57], v25, s[64:67], 0 offen offset:96 // (op 2181) Load value
buffer_load_dwordx2 v[58:59], v25, s[64:67], 0 offen offset:112 // (op 2181) Load value
buffer_load_dwordx4 v[60:63], v25, s[64:67], 0 offen offset:120 // (op 2181) Load value
buffer_load_dwordx2 v[64:65], v25, s[64:67], 0 offen offset:136 // (op 2181) Load value
v_add_u32 v25, v25, v52 // (op 2181) 
buffer_load_dwordx4 v[66:69], v25, s[64:67], 0 offen offset:96 // (op 2181) Load value
buffer_load_dwordx2 v[70:71], v25, s[64:67], 0 offen offset:112 // (op 2181) Load value
buffer_load_dwordx4 v[72:75], v25, s[64:67], 0 offen offset:120 // (op 2181) Load value
buffer_load_dwordx2 v[76:77], v25, s[64:67], 0 offen offset:136 // (op 2181) Load value






buffer_load_dword v20, v23, s[72:75], 0 offen offset:4 // (op 2184) Load value

v_and_b32 v23, -4, v1 // (op 2957) 
v_sub_u32 v25, v1, v23 // (op 2957) 

v_lshlrev_b32 v23, 4, v25 // (op 2957) 

v_lshrrev_b32 v25, 2, v1 // (op 2957) 
v_lshl_add_u32 v27, v25, 6, v23 // (op 2957) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 2957) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 6 // (op 2412) 
v_mov_b64 v[52:53], s[18:19] // (op 2412) convert


s_lshl_b64 s[18:19], s[26:27], 7 // (op 2412) 
v_mov_b32 v23, s18 // (op 2412) 
v_mov_b32 v25, s19 // (op 2412) 
v_add_co_u32 v78, s[50:51], v14, v23 // (op 2412) least significant half
v_addc_co_u32 v79, s[50:51], v15, v25, s[50:51] // (op 2412) most significant half




s_add_u32 s1, 8192, s0 // (op 2412) 
s_mov_b32 m0, s1 // (op 2412) reg expression
buffer_load_dwordx4 v78, s[52:55], 0 offen offset: 0 lds // (op 2412) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v52 // (op 2412) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 2412) most significant half

s_add_u32 m0, m0, 4096 // (op 2412) 
buffer_load_dwordx4 v78, s[52:55], 0 offen offset: 0 lds // (op 2412) Load value direct to lds







s_lshl_b32 s0, s36, 2 // (op 2190) 
v_add_u32 v23, v18, s0 // (op 2190) 


buffer_load_ubyte v25, v23, s[68:71], 0 offen offset: 0 // (op 2190) Load value


v_and_b32 v18, -2, v1 // (op 2784) 
v_sub_u32 v23, v1, v18 // (op 2784) 

v_lshlrev_b32 v18, 6, v23 // (op 2784) 

v_lshrrev_b32 v23, 1, v1 // (op 2784) 
v_lshl_add_u32 v27, v23, 8, v18 // (op 2784) 


v_mul_lo_u32 v18, v27, 6 // (op 2784) 

v_lshrrev_b32 v23, 3, v18 // (op 2784) 




s_waitcnt vmcnt(23)
ds_write_b128 v23, v[28:31] offset:16384 // (op 2192) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v23, v[32:33] offset:16400 // (op 2192) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v23, v[34:37] offset:16408 // (op 2192) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v23, v[38:39] offset:16424 // (op 2192) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v23, v[40:43] offset:16480 // (op 2192) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v23, v[44:45] offset:16496 // (op 2192) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v23, v[46:49] offset:16504 // (op 2192) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v23, v[50:51] offset:16520 // (op 2192) Store local data 

v_lshlrev_b32 v18, 2, v1 // (op 2895) 



s_waitcnt vmcnt(15)
ds_write_b32 v18, v22 offset:40960 // (op 2193) Store local data 

v_and_b32 v18, -64, v1 // (op 2842) 
v_sub_u32 v23, v1, v18 // (op 2842) 

v_and_b32 v18, 4294967232, v1 // (op 2842) 
v_add_u32 v27, v18, v23 // (op 2842) 





s_waitcnt vmcnt(12)
ds_write_b8 v27, v26 offset:41984 // (op 2195) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 3133) 



ds_read_b128 v[78:81], v9 offset:16384 // (op 2213) Load local data 
ds_read_b64 v[82:83], v9 offset:16400 // (op 2213) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3174) 



ds_read_u8 v18, v6 offset:40960 // (op 2236) Load local data 



ds_read_b128 v[84:87], v9 offset:22528 // (op 2218) Load local data 
ds_read_b64 v[88:89], v9 offset:22544 // (op 2218) Load local data 



ds_read_u8 v23, v6 offset:41216 // (op 2241) Load local data 



ds_read_b128 v[90:93], v9 offset:28672 // (op 2223) Load local data 
ds_read_b64 v[94:95], v9 offset:28688 // (op 2223) Load local data 



ds_read_u8 v27, v6 offset:41472 // (op 2246) Load local data 



ds_read_b128 v[96:99], v9 offset:34816 // (op 2228) Load local data 
ds_read_b64 v[100:101], v9 offset:34832 // (op 2228) Load local data 



ds_read_u8 v14, v6 offset:41728 // (op 2251) Load local data 
s_barrier  // (op 3114) 



ds_read_b64_tr_b8 v[102:103], v3 // (op 2209) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v3 offset:512 // (op 2209) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v3 offset:2048 // (op 2209) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v3 offset:2560 // (op 2209) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 3154) 


ds_read_u8 v15, v0 offset:41984 // (op 2232) Load local data 
v_and_b32 v52, -4, v1 // (op 2703) 
v_sub_u32 v53, v1, v52 // (op 2703) 

v_lshlrev_b32 v52, 4, v53 // (op 2703) 

v_lshrrev_b32 v53, 2, v1 // (op 2703) 
v_lshl_add_u32 v110, v53, 6, v52 // (op 2703) 


s_nop 0
v_readfirstlane_b32 s0, v110 // (op 2703) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v52, -2, v1 // (op 2755) 
v_sub_u32 v53, v1, v52 // (op 2755) 

v_lshlrev_b32 v52, 6, v53 // (op 2755) 

v_lshrrev_b32 v53, 1, v1 // (op 2755) 
v_lshl_add_u32 v110, v53, 8, v52 // (op 2755) 


v_mul_lo_u32 v52, v110, 6 // (op 2755) 

v_lshrrev_b32 v53, 3, v52 // (op 2755) 

v_and_b32 v52, -64, v1 // (op 2813) 
v_sub_u32 v110, v1, v52 // (op 2813) 

v_and_b32 v52, 4294967232, v1 // (op 2813) 
v_add_u32 v111, v52, v110 // (op 2813) 


v_lshlrev_b32 v52, 2, v1 // (op 2866) 
v_and_b32 v110, -4, v1 // (op 2942) 
v_sub_u32 v112, v1, v110 // (op 2942) 

v_lshlrev_b32 v110, 4, v112 // (op 2942) 

v_lshrrev_b32 v112, 2, v1 // (op 2942) 
v_lshl_add_u32 v113, v112, 6, v110 // (op 2942) 


s_nop 0
v_readfirstlane_b32 s1, v113 // (op 2942) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v110, -2, v1 // (op 2981) 
v_sub_u32 v112, v1, v110 // (op 2981) 

v_lshlrev_b32 v110, 6, v112 // (op 2981) 

v_lshrrev_b32 v112, 1, v1 // (op 2981) 
v_lshl_add_u32 v113, v112, 8, v110 // (op 2981) 


v_mul_lo_u32 v110, v113, 6 // (op 2981) 

v_lshrrev_b32 v112, 3, v110 // (op 2981) 

v_and_b32 v110, -64, v1 // (op 3010) 
v_sub_u32 v113, v1, v110 // (op 3010) 

v_and_b32 v110, 4294967232, v1 // (op 3010) 
v_add_u32 v114, v110, v113 // (op 3010) 


v_lshlrev_b32 v110, 2, v1 // (op 3034) 
s_mov_b32 s18, 0 // (op 872) call()
s_ashr_i64 s[50:51], s[30:31], 7 // (op 871) 
s_mov_b32 s19, s50 // (op 871) convert

s_and_b32 s22, s19, 4294967294 // (op 871) 

s_cmp_lt_u32 s18, s22 // (op 871) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_871 // (op 871) Condition: Top (jump to RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_871 if false)
RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_871:

s_add_u32 s18, s18, 2 // (op 873) 
s_mul_i32 s51, s33, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s32, 6 // (op 10) least significant
s_mul_hi_u32 s19, s32, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 10) most significant: sum

v_lshrrev_b64 v[116:117], 3, s[50:51] // (op 10) 


v_mov_b32 v113, v21 // (op 10) 
v_mov_b32 v115, v113 // (op 10) 
buffer_load_dwordx4 v[28:31], v115, s[64:67], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[32:33], v115, s[64:67], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[34:37], v115, s[64:67], 0 offen offset:216 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[102:109], v[78:83], a[0:15], v15, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 90) 
buffer_load_dwordx2 v[38:39], v115, s[64:67], 0 offen offset:232 // (op 10) Load value
s_barrier  // (op 3123) 
v_add_u32 v115, v115, v116 // (op 10) 



buffer_load_dwordx4 v[40:43], v115, s[64:67], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[44:45], v115, s[64:67], 0 offen offset:208 // (op 10) Load value
ds_read_b64_tr_b8 v[118:119], v3 offset:4096 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[120:121], v3 offset:4608 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v3 offset:6144 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[124:125], v3 offset:6656 // (op 93) Transpose load local data 
buffer_load_dwordx4 v[46:49], v115, s[64:67], 0 offen offset:216 // (op 10) Load value
s_barrier  // (op 3143) 
buffer_load_dwordx2 v[50:51], v115, s[64:67], 0 offen offset:232 // (op 10) Load value







ds_read_b128 v[78:81], v9 offset:16432 // (op 97) Load local data 
ds_read_b64 v[82:83], v9 offset:16448 // (op 97) Load local data 
v_mov_b32 v115, v24 // (op 25) 

s_barrier  // (op 3164) 
buffer_load_dword v22, v115, s[72:75], 0 offen offset:8 // (op 25) Load value


ds_read_u8 v116, v0 offset:42112 // (op 101) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3117) 

s_barrier  // (op 3184) 
s_lshl_b64 s[50:51], s[26:27], 6 // (op 2391) 



v_mov_b64 v[126:127], s[50:51] // (op 2391) convert


v_mov_b64 v[128:129], v[16:17] // (op 2391) 
s_lshl_b64 s[50:51], s[26:27], 7 // (op 2391) 
s_lshl_b64 s[22:23], s[50:51], 1 // (op 2391) 

v_mov_b32 v117, s22 // (op 2391) 
v_mov_b32 v130, s23 // (op 2391) 
v_add_co_u32 v132, s[24:25], v128, v117 // (op 2391) least significant half
v_addc_co_u32 v133, s[24:25], v129, v130, s[24:25] // (op 2391) most significant half




s_mov_b32 m0, s0 // (op 2391) reg expression
ds_read_u8 v18, v6 offset:40962 // (op 105) Load local data 
buffer_load_dwordx4 v132, s[52:55], 0 offen offset: 0 lds // (op 2391) Load value direct to lds
v_add_co_u32 v132, s[50:51], v132, v126 // (op 2391) least significant half
v_addc_co_u32 v133, s[50:51], 0, v133, s[50:51] // (op 2391) most significant half

s_add_u32 m0, m0, 4096 // (op 2391) 
buffer_load_dwordx4 v132, s[52:55], 0 offen offset: 0 lds // (op 2391) Load value direct to lds


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[118:125], v[78:83], a[0:15], v116, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 107) 


v_mov_b32 v117, v19 // (op 17) 
s_lshl_b32 s19, s36, 3 // (op 17) 
v_add_u32 v126, v117, s19 // (op 17) 


buffer_load_ubyte v26, v126, s[68:71], 0 offen offset: 0 // (op 17) Load value

v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[102:109], v[84:89], a[16:31], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 451) 



s_barrier  // (op 3204) 
ds_read_b128 v[78:81], v9 offset:22576 // (op 455) Load local data 



ds_read_b64 v[82:83], v9 offset:22592 // (op 455) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v112, v[54:57] offset:42240 // (op 306) Store local data 



ds_read_u8 v18, v6 offset:41218 // (op 459) Load local data 
s_waitcnt vmcnt(22)
ds_write_b64 v112, v[58:59] offset:42256 // (op 306) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[118:125], v[78:83], a[16:31], v116, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 460) 
s_waitcnt vmcnt(21)
ds_write_b128 v112, v[60:63] offset:42264 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v112, v[64:65] offset:42280 // (op 306) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[102:109], v[90:95], a[32:47], v15, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 685) 
s_waitcnt vmcnt(19)
ds_write_b128 v112, v[66:69] offset:42336 // (op 306) Store local data 



ds_read_b128 v[78:81], v9 offset:28720 // (op 689) Load local data 
ds_read_b64 v[82:83], v9 offset:28736 // (op 689) Load local data 
s_waitcnt vmcnt(18)
ds_write_b64 v112, v[70:71] offset:42352 // (op 306) Store local data 



ds_read_u8 v18, v6 offset:41474 // (op 693) Load local data 
s_waitcnt vmcnt(17)
ds_write_b128 v112, v[72:75] offset:42360 // (op 306) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[118:125], v[78:83], a[32:47], v116, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 694) 
s_waitcnt vmcnt(16)
ds_write_b64 v112, v[76:77] offset:42376 // (op 306) Store local data 
s_barrier  // (op 3216) 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[102:109], v[96:101], a[48:63], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 919) 






ds_read_b128 v[96:99], v9 offset:34864 // (op 923) Load local data 
ds_read_b64 v[100:101], v9 offset:34880 // (op 923) Load local data 
v_add_u32 v126, 66816, v110 // (op 308) 



ds_read_u8 v14, v6 offset:41730 // (op 927) Load local data 
s_waitcnt vmcnt(15)
ds_write_b32 v126, v20 // (op 308) Store local data 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[118:125], v[96:101], a[48:63], v116, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 928) 


s_barrier  // (op 3210) 



v_add_u32 v116, 67840, v114 // (op 307) 
s_waitcnt vmcnt(12)
ds_write_b8 v116, v25 // (op 307) Store local data 

s_barrier  // (op 3200) 



ds_read_b128 v[78:81], v13 offset:42240 // (op 291) Load local data 
ds_read_b64 v[82:83], v13 offset:42256 // (op 291) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3213) 



v_add_u32 v116, 66816, v12 // (op 295) 
ds_read_u8 v18, v116 // (op 295) Load local data 




ds_read_b128 v[84:87], v13 offset:48384 // (op 474) Load local data 
ds_read_b64 v[88:89], v13 offset:48400 // (op 474) Load local data 



v_add_u32 v116, 67072, v12 // (op 478) 
ds_read_u8 v23, v116 // (op 478) Load local data 




ds_read_b128 v[90:93], v13 offset:54528 // (op 708) Load local data 
ds_read_b64 v[94:95], v13 offset:54544 // (op 708) Load local data 



v_add_u32 v116, 67328, v12 // (op 712) 
ds_read_u8 v27, v116 // (op 712) Load local data 




ds_read_b128 v[118:121], v13 offset:60672 // (op 942) Load local data 
ds_read_b64 v[122:123], v13 offset:60688 // (op 942) Load local data 



v_add_u32 v116, 67584, v12 // (op 946) 
ds_read_u8 v124, v116 // (op 946) Load local data 

s_barrier  // (op 3194) 



ds_read_b64_tr_b8 v[102:103], v5 offset:8192 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v5 offset:8704 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v5 offset:10240 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v5 offset:10752 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 3207) 


v_add_u32 v116, 67840, v4 // (op 293) 
ds_read_u8 v15, v116 // (op 293) Load local data 


s_mul_i32 s51, s33, 6 // (op 309) most significant: low of high * low
s_mul_i32 s50, s32, 6 // (op 309) least significant
s_mul_hi_u32 s19, s32, 6 // (op 309) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 309) most significant: sum

v_lshrrev_b64 v[126:127], 3, s[50:51] // (op 309) 


v_mov_b32 v116, v113 // (op 309) 
buffer_load_dwordx4 v[54:57], v116, s[64:67], 0 offen offset:288 // (op 309) Load value
buffer_load_dwordx2 v[58:59], v116, s[64:67], 0 offen offset:304 // (op 309) Load value
buffer_load_dwordx4 v[60:63], v116, s[64:67], 0 offen offset:312 // (op 309) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[102:109], v[78:83], a[0:15], v15, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 296) 
buffer_load_dwordx2 v[64:65], v116, s[64:67], 0 offen offset:328 // (op 309) Load value
v_add_u32 v116, v116, v126 // (op 309) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[102:109], v[84:89], a[16:31], v15, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 461) 
buffer_load_dwordx4 v[66:69], v116, s[64:67], 0 offen offset:288 // (op 309) Load value
buffer_load_dwordx2 v[70:71], v116, s[64:67], 0 offen offset:304 // (op 309) Load value
ds_read_b64_tr_b8 v[130:131], v5 offset:12288 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[132:133], v5 offset:12800 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[134:135], v5 offset:14336 // (op 298) Transpose load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[102:109], v[90:95], a[32:47], v15, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 695) 
ds_read_b64_tr_b8 v[136:137], v5 offset:14848 // (op 298) Transpose load local data 



v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[102:109], v[118:123], a[48:63], v15, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 929) 
ds_read_b128 v[138:141], v13 offset:42288 // (op 300) Load local data 


ds_read_b64 v[142:143], v13 offset:42304 // (op 300) Load local data 
buffer_load_dwordx4 v[72:75], v116, s[64:67], 0 offen offset:312 // (op 309) Load value


v_add_u32 v118, 67968, v4 // (op 302) 
ds_read_u8 v119, v118 // (op 302) Load local data 

buffer_load_dwordx2 v[76:77], v116, s[64:67], 0 offen offset:328 // (op 309) Load value









s_barrier  // (op 3259) 
buffer_load_dword v20, v115, s[72:75], 0 offen offset:12 // (op 311) Load value
v_add_u32 v21, 192, v21 // (op 2529) 

v_add_u32 v113, 66818, v12 // (op 304) 
v_add_u32 v24, 8, v24 // (op 2657) 
ds_read_u8 v118, v113 // (op 304) Load local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 3197) 




s_lshl_b64 s[50:51], s[26:27], 6 // (op 2402) 
v_mov_b64 v[120:121], s[50:51] // (op 2402) convert


s_lshl_b64 s[50:51], s[26:27], 7 // (op 2402) 
s_mul_i32 s23, 3, s51 // (op 2402) most significant: low of low * high
s_mul_i32 s22, 3, s50 // (op 2402) least significant
s_mul_hi_u32 s19, 3, s50 // (op 2402) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 2402) most significant: sum


v_mov_b32 v113, s22 // (op 2402) 
v_mov_b32 v115, s23 // (op 2402) 
ds_read_b128 v[122:125], v13 offset:48432 // (op 465) Load local data 
v_add_co_u32 v144, s[24:25], v128, v113 // (op 2402) least significant half
v_addc_co_u32 v145, s[24:25], v129, v115, s[24:25] // (op 2402) most significant half




s_add_u32 s19, 8192, s1 // (op 2402) 
s_mov_b32 m0, s19 // (op 2402) reg expression
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[130:137], v[138:143], a[0:15], v119, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 305) 
ds_read_b64 v[126:127], v13 offset:48448 // (op 465) Load local data 





v_add_u32 v113, 67074, v12 // (op 469) 
ds_read_u8 v118, v113 // (op 469) Load local data 

buffer_load_dwordx4 v144, s[52:55], 0 offen offset: 0 lds // (op 2402) Load value direct to lds



v_add_co_u32 v144, s[50:51], v144, v120 // (op 2402) least significant half
v_addc_co_u32 v145, s[50:51], 0, v145, s[50:51] // (op 2402) most significant half

s_add_u32 m0, m0, 4096 // (op 2402) 
buffer_load_dwordx4 v144, s[52:55], 0 offen offset: 0 lds // (op 2402) Load value direct to lds



ds_read_b128 v[138:141], v13 offset:54576 // (op 699) Load local data 

ds_read_b64 v[142:143], v13 offset:54592 // (op 699) Load local data 





s_lshl_b32 s19, s36, 2 // (op 310) 
s_mul_i32 s22, 3, s19 // (op 310) 

v_add_u32 v113, v117, s22 // (op 310) 


buffer_load_ubyte v25, v113, s[68:71], 0 offen offset: 0 // (op 310) Load value

v_add_u32 v113, 67330, v12 // (op 703) 

ds_read_u8 v115, v113 // (op 703) Load local data 




s_barrier  // (op 3136) 
s_lshl_b32 s19, s36, 3 // (op 2593) 



v_add_u32 v19, v19, s19 // (op 2593) 

s_lshl_b64 s[50:51], s[26:27], 7 // (op 2465) 
s_lshl_b64 s[22:23], s[50:51], 1 // (op 2465) 

v_mov_b32 v113, s22 // (op 2465) 
v_mov_b32 v116, s23 // (op 2465) 
v_add_co_u32 v16, s[24:25], v16, v113 // (op 2465) least significant half
v_addc_co_u32 v17, s[24:25], v17, v116, s[24:25] // (op 2465) most significant half




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[130:137], v[122:127], a[16:31], v119, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 470) 
ds_read_b128 v[144:147], v13 offset:60720 // (op 933) Load local data 


ds_read_b64 v[148:149], v13 offset:60736 // (op 933) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[130:137], v[138:143], a[32:47], v119, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 704) 





s_barrier  // (op 3317) 
v_add_u32 v113, 67586, v12 // (op 937) 
ds_read_u8 v115, v113 // (op 937) Load local data 

s_waitcnt vmcnt(23)
ds_write_b128 v53, v[28:31] offset:16384 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v53, v[32:33] offset:16400 // (op 13) Store local data 
s_barrier  // (op 3321) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[130:137], v[144:149], a[48:63], v119, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 938) 
s_waitcnt vmcnt(21)
ds_write_b128 v53, v[34:37] offset:16408 // (op 13) Store local data 




s_waitcnt vmcnt(20)
ds_write_b64 v53, v[38:39] offset:16424 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v53, v[40:43] offset:16480 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v53, v[44:45] offset:16496 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v53, v[46:49] offset:16504 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v53, v[50:51] offset:16520 // (op 13) Store local data 
s_barrier  // (op 3177) 



s_waitcnt vmcnt(15)
ds_write_b32 v52, v22 offset:40960 // (op 30) Store local data 
s_barrier  // (op 3157) 



s_waitcnt vmcnt(12)
ds_write_b8 v111, v26 offset:41984 // (op 21) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3146) 



ds_read_b128 v[78:81], v9 offset:16384 // (op 77) Load local data 
ds_read_b64 v[82:83], v9 offset:16400 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3187) 



ds_read_u8 v18, v6 offset:40960 // (op 87) Load local data 



ds_read_b128 v[84:87], v9 offset:22528 // (op 446) Load local data 
ds_read_b64 v[88:89], v9 offset:22544 // (op 446) Load local data 



ds_read_u8 v23, v6 offset:41216 // (op 450) Load local data 



ds_read_b128 v[90:93], v9 offset:28672 // (op 680) Load local data 
ds_read_b64 v[94:95], v9 offset:28688 // (op 680) Load local data 



ds_read_u8 v27, v6 offset:41472 // (op 684) Load local data 



ds_read_b128 v[96:99], v9 offset:34816 // (op 914) Load local data 
ds_read_b64 v[100:101], v9 offset:34832 // (op 914) Load local data 



ds_read_u8 v14, v6 offset:41728 // (op 918) Load local data 
s_barrier  // (op 3126) 



ds_read_b64_tr_b8 v[102:103], v3 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v3 offset:512 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v3 offset:2048 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v3 offset:2560 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 3167) 


ds_read_u8 v15, v0 offset:41984 // (op 82) Load local data 
s_ashr_i64 s[50:51], s[30:31], 7 // (op 871) 
s_mov_b32 s19, s50 // (op 871) convert

s_and_b32 s22, s19, 4294967294 // (op 871) 

s_cmp_lt_u32 s18, s22 // (op 871) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_871 // (op 871) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_871 if true)
RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_871:

















s_barrier  // (op 3299) 
s_mul_hi_i32 s0, s16, s10 // (op 2425) 
v_and_b32 v19, -4, v1 // (op 2425) 
s_add_i32 s1, s0, s16 // (op 2425) 

v_sub_u32 v24, v1, v19 // (op 2425) 

s_ashr_i32 s0, s1, 31 // (op 2425) 
v_lshlrev_b32 v19, 4, v24 // (op 2425) 

s_cmp_eq_i32 s10, 0 // (op 2425) 
s_mov_b32 s18, scc // (op 2425) 
v_lshrrev_b32 v24, 2, v1 // (op 2425) 
s_cmp_eq_i32 s18, 0 // (op 2425) 
s_cselect_b32 s19, 0, -1 // (op 2425) 

s_and_b32 s18, s11, 31 // (op 2425) 
s_lshl_b32 s22, 1, s18 // (op 2425) 
s_add_i32 s23, s22, s19 // (op 2425) 


s_and_b32 s19, s0, s23 // (op 2425) 


s_add_i32 s0, s1, s19 // (op 2425) 


s_ashr_i32 s1, s0, s18 // (op 2425) 


s_lshl_b32 s0, s11, 24 // (op 2425) 
s_ashr_i32 s18, s0, 31 // (op 2425) 

s_xor_b32 s0, s1, s18 // (op 2425) 

s_sub_i32 s1, s0, s18 // (op 2425) 


s_mul_hi_i32 s0, s1, s12 // (op 2425) 
s_add_i32 s18, s0, s1 // (op 2425) 

s_ashr_i32 s0, s18, 31 // (op 2425) 
s_cmp_eq_i32 s12, 0 // (op 2425) 
s_mov_b32 s19, scc // (op 2425) 
s_cmp_eq_i32 s19, 0 // (op 2425) 
s_cselect_b32 s22, 0, -1 // (op 2425) 

s_and_b32 s19, s13, 31 // (op 2425) 
s_lshl_b32 s23, 1, s19 // (op 2425) 
s_add_i32 s24, s23, s22 // (op 2425) 


s_and_b32 s25, s0, s24 // (op 2425) 


s_add_i32 s0, s18, s25 // (op 2425) 


s_ashr_i32 s18, s0, s19 // (op 2425) 


s_lshl_b32 s0, s13, 24 // (op 2425) 
s_ashr_i32 s19, s0, 31 // (op 2425) 

s_xor_b32 s0, s18, s19 // (op 2425) 

s_sub_i32 s18, s0, s19 // (op 2425) 


s_mul_i32 s0, s1, s15 // (op 2425) 

s_sub_u32 s1, s16, s0 // (op 2425) 

s_mul_i32 s0, s18, s15 // (op 2425) 
s_add_u32 s0, s0, s1 // (op 2425) 


s_mul_hi_i32 s1, s16, s9 // (op 2425) 
s_add_i32 s18, s1, s16 // (op 2425) 

s_ashr_i32 s1, s18, 31 // (op 2425) 
s_cmp_eq_i32 s9, 0 // (op 2425) 
s_mov_b32 s19, scc // (op 2425) 
s_cmp_eq_i32 s19, 0 // (op 2425) 
s_cselect_b32 s22, 0, -1 // (op 2425) 

s_and_b32 s19, s46, 31 // (op 2425) 
s_lshl_b32 s23, 1, s19 // (op 2425) 
s_add_i32 s24, s23, s22 // (op 2425) 


s_and_b32 s25, s1, s24 // (op 2425) 


s_add_i32 s1, s18, s25 // (op 2425) 


s_ashr_i32 s18, s1, s19 // (op 2425) 


s_lshl_b32 s1, s46, 24 // (op 2425) 
s_ashr_i32 s19, s1, 31 // (op 2425) 

s_xor_b32 s1, s18, s19 // (op 2425) 

s_sub_i32 s18, s1, s19 // (op 2425) 


s_add_u32 s50, s6, 256 // (op 2425) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2425) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2425) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2425) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2425) 

s_mul_i32 s1, s15, s24 // (op 2425) 

s_mul_i32 s19, s8, s1 // (op 2425) 

s_mul_i32 s1, s18, s19 // (op 2425) 

s_sub_u32 s19, s16, s1 // (op 2425) 

s_mov_b32 s1, s19 // (op 2425) 
s_mul_hi_i32 s22, s1, s14 // (op 2425) 
s_add_i32 s23, s22, s1 // (op 2425) 


s_ashr_i32 s1, s23, 31 // (op 2425) 
s_cmp_eq_i32 s14, 0 // (op 2425) 
s_mov_b32 s22, scc // (op 2425) 
s_cmp_eq_i32 s22, 0 // (op 2425) 
s_cselect_b32 s24, 0, -1 // (op 2425) 

s_and_b32 s22, s48, 31 // (op 2425) 
s_lshl_b32 s25, 1, s22 // (op 2425) 
s_add_i32 s49, s25, s24 // (op 2425) 


s_and_b32 s24, s1, s49 // (op 2425) 


s_add_i32 s1, s23, s24 // (op 2425) 


s_ashr_i32 s23, s1, s22 // (op 2425) 


s_lshl_b32 s1, s48, 24 // (op 2425) 
s_ashr_i32 s22, s1, 31 // (op 2425) 

s_xor_b32 s1, s23, s22 // (op 2425) 

s_sub_i32 s23, s1, s22 // (op 2425) 


s_mul_i32 s1, s47, s15 // (op 2425) 
s_add_u32 s24, s20, 64 // (op 2425) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2425) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2425) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2425) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2425) 

s_sub_i32 s22, s24, s1 // (op 2425) 

s_mul_i32 s24, s23, s22 // (op 2425) 


s_sub_u32 s25, s19, s24 // (op 2425) 


s_add_u32 s19, s25, s1 // (op 2425) 


s_cmp_eq_i32 s18, 0 // (op 2425) 
s_mov_b32 s1, scc // (op 2425) 

s_cmp_eq_i32 s1, 0 // (op 2425) 
s_cselect_b32 s18, s19, s0 // (op 2425) 



v_lshl_add_u32 v26, s18, 6, v19 // (op 2425) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2425) 
v_mov_b32 v17, 0 // (op 2425) convert
v_mov_b32 v16, v26 // (op 2425) convert

s_lshr_b32 s18, s0, 1 // (op 2425) 

v_lshl_add_u32 v19, s18, 8, v24 // (op 2425) 


v_mul_lo_u32 v29, v19, s27 // (op 2425) most significant: low of low * high
v_mul_hi_u32 v24, v19, s26 // (op 2425) most significant: high of low * low
v_mul_lo_u32 v28, v19, s26 // (op 2425) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 2425) most significant: sum


v_add_co_u32 v30, s[0:1], v16, v28 // (op 2425) least significant half
v_addc_co_u32 v31, s[0:1], v17, v29, s[0:1] // (op 2425) most significant half



s_mul_i32 s1, s26, s31 // (op 2427) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2427) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2427) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2427) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2427) most significant: sum
s_add_u32 s1, s1, s19 // (op 2427) most significant: sum


s_mov_b32 s52, s28 // (op 2427) reg expression
s_mov_b32 s53, s29 // (op 2427) reg expression
s_mov_b32 s54, s0 // (op 2427) reg expression
s_mov_b32 s55, 131072 // (op 2427) call()

s_mul_hi_i32 s0, s16, s10 // (op 2495) 
v_and_b32 v19, 4294967294, v1 // (op 2495) 
s_add_i32 s1, s0, s16 // (op 2495) 

v_and_b32 v24, -2, v1 // (op 2495) 
s_ashr_i32 s0, s1, 31 // (op 2495) 
v_sub_u32 v26, v1, v24 // (op 2495) 

s_cmp_eq_i32 s10, 0 // (op 2495) 
s_mov_b32 s18, scc // (op 2495) 
v_lshlrev_b32 v24, 6, v26 // (op 2495) 

s_cmp_eq_i32 s18, 0 // (op 2495) 
s_cselect_b32 s19, 0, -1 // (op 2495) 

s_and_b32 s18, s11, 31 // (op 2495) 
s_lshl_b32 s22, 1, s18 // (op 2495) 
s_add_i32 s23, s22, s19 // (op 2495) 


s_and_b32 s19, s0, s23 // (op 2495) 


s_add_i32 s0, s1, s19 // (op 2495) 


s_ashr_i32 s1, s0, s18 // (op 2495) 


s_lshl_b32 s0, s11, 24 // (op 2495) 
s_ashr_i32 s18, s0, 31 // (op 2495) 

s_xor_b32 s0, s1, s18 // (op 2495) 

s_sub_i32 s1, s0, s18 // (op 2495) 


s_mul_hi_i32 s0, s1, s12 // (op 2495) 
s_add_i32 s18, s0, s1 // (op 2495) 

s_ashr_i32 s0, s18, 31 // (op 2495) 
s_cmp_eq_i32 s12, 0 // (op 2495) 
s_mov_b32 s19, scc // (op 2495) 
s_cmp_eq_i32 s19, 0 // (op 2495) 
s_cselect_b32 s22, 0, -1 // (op 2495) 

s_and_b32 s19, s13, 31 // (op 2495) 
s_lshl_b32 s23, 1, s19 // (op 2495) 
s_add_i32 s24, s23, s22 // (op 2495) 


s_and_b32 s25, s0, s24 // (op 2495) 


s_add_i32 s0, s18, s25 // (op 2495) 


s_ashr_i32 s18, s0, s19 // (op 2495) 


s_lshl_b32 s0, s13, 24 // (op 2495) 
s_ashr_i32 s19, s0, 31 // (op 2495) 

s_xor_b32 s0, s18, s19 // (op 2495) 

s_sub_i32 s18, s0, s19 // (op 2495) 


s_add_u32 s50, s6, 256 // (op 2495) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2495) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2495) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2495) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2495) 

s_mul_i32 s0, s18, s24 // (op 2495) 

s_sub_i32 s18, s1, s0 // (op 2495) 


s_mul_hi_i32 s19, s16, s9 // (op 2495) 
s_add_i32 s0, s19, s16 // (op 2495) 

s_ashr_i32 s1, s0, 31 // (op 2495) 
s_cmp_eq_i32 s9, 0 // (op 2495) 
s_mov_b32 s19, scc // (op 2495) 
s_cmp_eq_i32 s19, 0 // (op 2495) 
s_cselect_b32 s22, 0, -1 // (op 2495) 

s_and_b32 s19, s46, 31 // (op 2495) 
s_lshl_b32 s23, 1, s19 // (op 2495) 
s_add_i32 s49, s23, s22 // (op 2495) 


s_and_b32 s22, s1, s49 // (op 2495) 


s_add_i32 s1, s0, s22 // (op 2495) 


s_ashr_i32 s0, s1, s19 // (op 2495) 


s_lshl_b32 s1, s46, 24 // (op 2495) 
s_ashr_i32 s19, s1, 31 // (op 2495) 

s_xor_b32 s1, s0, s19 // (op 2495) 

s_sub_i32 s0, s1, s19 // (op 2495) 


s_mul_i32 s1, s15, s24 // (op 2495) 

s_mul_i32 s19, s8, s1 // (op 2495) 

s_mul_i32 s1, s0, s19 // (op 2495) 

s_sub_u32 s19, s16, s1 // (op 2495) 

s_mul_hi_i32 s1, s19, s14 // (op 2495) 
s_add_i32 s22, s1, s19 // (op 2495) 


s_ashr_i32 s1, s22, 31 // (op 2495) 
s_cmp_eq_i32 s14, 0 // (op 2495) 
s_mov_b32 s19, scc // (op 2495) 
s_cmp_eq_i32 s19, 0 // (op 2495) 
s_cselect_b32 s23, 0, -1 // (op 2495) 

s_and_b32 s19, s48, 31 // (op 2495) 
s_lshl_b32 s24, 1, s19 // (op 2495) 
s_add_i32 s25, s24, s23 // (op 2495) 


s_and_b32 s23, s1, s25 // (op 2495) 


s_add_i32 s1, s22, s23 // (op 2495) 


s_ashr_i32 s22, s1, s19 // (op 2495) 


s_lshl_b32 s1, s48, 24 // (op 2495) 
s_ashr_i32 s19, s1, 31 // (op 2495) 

s_xor_b32 s1, s22, s19 // (op 2495) 

s_sub_i32 s22, s1, s19 // (op 2495) 


s_cmp_eq_i32 s0, 0 // (op 2495) 
s_mov_b32 s1, scc // (op 2495) 

s_cmp_eq_i32 s1, 0 // (op 2495) 
s_cselect_b32 s0, s22, s18 // (op 2495) 



v_lshl_add_u32 v26, s0, 8, v19 // (op 2495) 


v_mul_lo_u32 v17, v26, s33 // (op 2495) most significant: low of low * high
v_mul_hi_u32 v19, v26, s32 // (op 2495) most significant: high of low * low
v_mul_lo_u32 v16, v26, s32 // (op 2495) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 2495) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2495) 
s_lshr_b32 s18, s0, 1 // (op 2495) 

v_lshl_add_u32 v19, s18, 8, v24 // (op 2495) 


v_mov_b32 v29, 0 // (op 2495) convert
v_mov_b32 v28, v19 // (op 2495) convert

v_add_co_u32 v32, s[0:1], v28, v16 // (op 2495) least significant half
v_addc_co_u32 v33, s[0:1], v29, v17, s[0:1] // (op 2495) most significant half



v_mul_lo_u32 v17, v33, 6 // (op 2495) most significant: low of high * low
v_mul_hi_u32 v19, v32, 6 // (op 2495) most significant: high of low * low
v_mul_lo_u32 v16, v32, 6 // (op 2495) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 2495) most significant: sum


v_lshrrev_b64 v[28:29], 3, v[16:17] // (op 2495) 

v_mov_b32 v19, v28 // (op 2495) convert

s_mul_i32 s1, s7, 6 // (op 2497) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 2497) least significant
s_mul_hi_u32 s18, s6, 6 // (op 2497) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2497) most significant: sum

s_mul_i32 s19, s32, s1 // (op 2497) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 2497) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 2497) least significant
s_mul_hi_u32 s23, s32, s0 // (op 2497) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 2497) most significant: sum
s_add_u32 s19, s19, s23 // (op 2497) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2497) 

s_mov_b32 s64, s34 // (op 2497) reg expression
s_mov_b32 s65, s35 // (op 2497) reg expression
s_mov_b32 s66, s0 // (op 2497) reg expression
s_mov_b32 s67, 131072 // (op 2497) call()

s_mul_hi_i32 s0, s16, s10 // (op 2559) 
v_and_b32 v24, -64, v1 // (op 2559) 
s_add_i32 s1, s0, s16 // (op 2559) 

v_sub_u32 v26, v1, v24 // (op 2559) 

s_ashr_i32 s0, s1, 31 // (op 2559) 
v_lshrrev_b32 v24, 6, v1 // (op 2559) 
s_cmp_eq_i32 s10, 0 // (op 2559) 
s_mov_b32 s18, scc // (op 2559) 
s_cmp_eq_i32 s18, 0 // (op 2559) 
s_cselect_b32 s19, 0, -1 // (op 2559) 

s_and_b32 s18, s11, 31 // (op 2559) 
s_lshl_b32 s22, 1, s18 // (op 2559) 
s_add_i32 s23, s22, s19 // (op 2559) 


s_and_b32 s19, s0, s23 // (op 2559) 


s_add_i32 s0, s1, s19 // (op 2559) 


s_ashr_i32 s1, s0, s18 // (op 2559) 


s_lshl_b32 s0, s11, 24 // (op 2559) 
s_ashr_i32 s18, s0, 31 // (op 2559) 

s_xor_b32 s0, s1, s18 // (op 2559) 

s_sub_i32 s1, s0, s18 // (op 2559) 


s_mul_hi_i32 s0, s1, s12 // (op 2559) 
s_add_i32 s18, s0, s1 // (op 2559) 

s_ashr_i32 s0, s18, 31 // (op 2559) 
s_cmp_eq_i32 s12, 0 // (op 2559) 
s_mov_b32 s19, scc // (op 2559) 
s_cmp_eq_i32 s19, 0 // (op 2559) 
s_cselect_b32 s22, 0, -1 // (op 2559) 

s_and_b32 s19, s13, 31 // (op 2559) 
s_lshl_b32 s23, 1, s19 // (op 2559) 
s_add_i32 s24, s23, s22 // (op 2559) 


s_and_b32 s25, s0, s24 // (op 2559) 


s_add_i32 s0, s18, s25 // (op 2559) 


s_ashr_i32 s18, s0, s19 // (op 2559) 


s_lshl_b32 s0, s13, 24 // (op 2559) 
s_ashr_i32 s19, s0, 31 // (op 2559) 

s_xor_b32 s0, s18, s19 // (op 2559) 

s_sub_i32 s18, s0, s19 // (op 2559) 


s_mul_i32 s0, s1, s15 // (op 2559) 

s_sub_u32 s1, s16, s0 // (op 2559) 

s_mul_i32 s0, s18, s15 // (op 2559) 
s_add_u32 s0, s0, s1 // (op 2559) 


s_mul_hi_i32 s1, s16, s9 // (op 2559) 
s_add_i32 s18, s1, s16 // (op 2559) 

s_ashr_i32 s1, s18, 31 // (op 2559) 
s_cmp_eq_i32 s9, 0 // (op 2559) 
s_mov_b32 s19, scc // (op 2559) 
s_cmp_eq_i32 s19, 0 // (op 2559) 
s_cselect_b32 s22, 0, -1 // (op 2559) 

s_and_b32 s19, s46, 31 // (op 2559) 
s_lshl_b32 s23, 1, s19 // (op 2559) 
s_add_i32 s24, s23, s22 // (op 2559) 


s_and_b32 s25, s1, s24 // (op 2559) 


s_add_i32 s1, s18, s25 // (op 2559) 


s_ashr_i32 s18, s1, s19 // (op 2559) 


s_lshl_b32 s1, s46, 24 // (op 2559) 
s_ashr_i32 s19, s1, 31 // (op 2559) 

s_xor_b32 s1, s18, s19 // (op 2559) 

s_sub_i32 s18, s1, s19 // (op 2559) 


s_add_u32 s50, s6, 256 // (op 2559) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2559) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2559) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2559) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2559) 

s_mul_i32 s1, s15, s24 // (op 2559) 

s_mul_i32 s19, s8, s1 // (op 2559) 

s_mul_i32 s1, s18, s19 // (op 2559) 

s_sub_u32 s19, s16, s1 // (op 2559) 

s_mov_b32 s1, s19 // (op 2559) 
s_mul_hi_i32 s22, s1, s14 // (op 2559) 
s_add_i32 s23, s22, s1 // (op 2559) 


s_ashr_i32 s1, s23, 31 // (op 2559) 
s_cmp_eq_i32 s14, 0 // (op 2559) 
s_mov_b32 s22, scc // (op 2559) 
s_cmp_eq_i32 s22, 0 // (op 2559) 
s_cselect_b32 s24, 0, -1 // (op 2559) 

s_and_b32 s22, s48, 31 // (op 2559) 
s_lshl_b32 s25, 1, s22 // (op 2559) 
s_add_i32 s49, s25, s24 // (op 2559) 


s_and_b32 s24, s1, s49 // (op 2559) 


s_add_i32 s1, s23, s24 // (op 2559) 


s_ashr_i32 s23, s1, s22 // (op 2559) 


s_lshl_b32 s1, s48, 24 // (op 2559) 
s_ashr_i32 s22, s1, 31 // (op 2559) 

s_xor_b32 s1, s23, s22 // (op 2559) 

s_sub_i32 s23, s1, s22 // (op 2559) 


s_mul_i32 s1, s47, s15 // (op 2559) 
s_add_u32 s24, s20, 64 // (op 2559) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2559) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2559) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2559) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2559) 

s_sub_i32 s22, s24, s1 // (op 2559) 

s_mul_i32 s24, s23, s22 // (op 2559) 


s_sub_u32 s25, s19, s24 // (op 2559) 


s_add_u32 s19, s25, s1 // (op 2559) 


s_cmp_eq_i32 s18, 0 // (op 2559) 
s_mov_b32 s1, scc // (op 2559) 

s_cmp_eq_i32 s1, 0 // (op 2559) 
s_cselect_b32 s18, s19, s0 // (op 2559) 



v_lshl_add_u32 v16, s18, 6, v26 // (op 2559) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2559) 
s_lshr_b32 s18, s0, 1 // (op 2559) 

v_lshl_add_u32 v17, s18, 3, v24 // (op 2559) 


v_mul_lo_u32 v24, v17, s36 // (op 2559) 

v_add_u32 v17, v16, v24 // (op 2559) 


s_mul_i32 s1, s36, s41 // (op 2561) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2561) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2561) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2561) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2561) most significant: sum
s_add_u32 s1, s1, s19 // (op 2561) most significant: sum


s_mov_b32 s68, s38 // (op 2561) reg expression
s_mov_b32 s69, s39 // (op 2561) reg expression
s_mov_b32 s70, s0 // (op 2561) reg expression
s_mov_b32 s71, 131072 // (op 2561) call()

s_mul_hi_i32 s0, s16, s10 // (op 2623) 
s_add_i32 s1, s0, s16 // (op 2623) 

s_ashr_i32 s0, s1, 31 // (op 2623) 
s_cmp_eq_i32 s10, 0 // (op 2623) 
s_mov_b32 s18, scc // (op 2623) 
s_cmp_eq_i32 s18, 0 // (op 2623) 
s_cselect_b32 s19, 0, -1 // (op 2623) 

s_and_b32 s18, s11, 31 // (op 2623) 
s_lshl_b32 s22, 1, s18 // (op 2623) 
s_add_i32 s23, s22, s19 // (op 2623) 


s_and_b32 s19, s0, s23 // (op 2623) 


s_add_i32 s0, s1, s19 // (op 2623) 


s_ashr_i32 s1, s0, s18 // (op 2623) 


s_lshl_b32 s0, s11, 24 // (op 2623) 
s_ashr_i32 s18, s0, 31 // (op 2623) 

s_xor_b32 s0, s1, s18 // (op 2623) 

s_sub_i32 s1, s0, s18 // (op 2623) 


s_mul_hi_i32 s0, s1, s12 // (op 2623) 
s_add_i32 s18, s0, s1 // (op 2623) 

s_ashr_i32 s0, s18, 31 // (op 2623) 
s_cmp_eq_i32 s12, 0 // (op 2623) 
s_mov_b32 s19, scc // (op 2623) 
s_cmp_eq_i32 s19, 0 // (op 2623) 
s_cselect_b32 s22, 0, -1 // (op 2623) 

s_and_b32 s19, s13, 31 // (op 2623) 
s_lshl_b32 s23, 1, s19 // (op 2623) 
s_add_i32 s24, s23, s22 // (op 2623) 


s_and_b32 s25, s0, s24 // (op 2623) 


s_add_i32 s0, s18, s25 // (op 2623) 


s_ashr_i32 s18, s0, s19 // (op 2623) 


s_lshl_b32 s0, s13, 24 // (op 2623) 
s_ashr_i32 s19, s0, 31 // (op 2623) 

s_xor_b32 s0, s18, s19 // (op 2623) 

s_sub_i32 s18, s0, s19 // (op 2623) 


s_add_u32 s50, s6, 256 // (op 2623) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2623) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2623) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2623) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 2623) 

s_mul_i32 s0, s18, s24 // (op 2623) 

s_sub_i32 s18, s1, s0 // (op 2623) 


s_mul_hi_i32 s19, s16, s9 // (op 2623) 
s_add_i32 s0, s19, s16 // (op 2623) 

s_ashr_i32 s1, s0, 31 // (op 2623) 
s_cmp_eq_i32 s9, 0 // (op 2623) 
s_mov_b32 s19, scc // (op 2623) 
s_cmp_eq_i32 s19, 0 // (op 2623) 
s_cselect_b32 s22, 0, -1 // (op 2623) 

s_and_b32 s19, s46, 31 // (op 2623) 
s_lshl_b32 s23, 1, s19 // (op 2623) 
s_add_i32 s49, s23, s22 // (op 2623) 


s_and_b32 s22, s1, s49 // (op 2623) 


s_add_i32 s1, s0, s22 // (op 2623) 


s_ashr_i32 s0, s1, s19 // (op 2623) 


s_lshl_b32 s1, s46, 24 // (op 2623) 
s_ashr_i32 s19, s1, 31 // (op 2623) 

s_xor_b32 s1, s0, s19 // (op 2623) 

s_sub_i32 s0, s1, s19 // (op 2623) 


s_mul_i32 s1, s15, s24 // (op 2623) 

s_mul_i32 s19, s8, s1 // (op 2623) 

s_mul_i32 s1, s0, s19 // (op 2623) 

s_sub_u32 s19, s16, s1 // (op 2623) 

s_mul_hi_i32 s1, s19, s14 // (op 2623) 
s_add_i32 s22, s1, s19 // (op 2623) 


s_ashr_i32 s1, s22, 31 // (op 2623) 
s_cmp_eq_i32 s14, 0 // (op 2623) 
s_mov_b32 s19, scc // (op 2623) 
s_cmp_eq_i32 s19, 0 // (op 2623) 
s_cselect_b32 s23, 0, -1 // (op 2623) 

s_and_b32 s19, s48, 31 // (op 2623) 
s_lshl_b32 s24, 1, s19 // (op 2623) 
s_add_i32 s25, s24, s23 // (op 2623) 


s_and_b32 s23, s1, s25 // (op 2623) 


s_add_i32 s1, s22, s23 // (op 2623) 


s_ashr_i32 s22, s1, s19 // (op 2623) 


s_lshl_b32 s1, s48, 24 // (op 2623) 
s_ashr_i32 s19, s1, 31 // (op 2623) 

s_xor_b32 s1, s22, s19 // (op 2623) 

s_sub_i32 s22, s1, s19 // (op 2623) 


s_cmp_eq_i32 s0, 0 // (op 2623) 
s_mov_b32 s1, scc // (op 2623) 

s_cmp_eq_i32 s1, 0 // (op 2623) 
s_cselect_b32 s0, s22, s18 // (op 2623) 



v_lshl_add_u32 v16, s0, 8, v1 // (op 2623) 

v_mul_lo_u32 v24, v16, s42 // (op 2623) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 2623) 
s_lshr_b32 s18, s0, 1 // (op 2623) 

v_lshl_add_u32 v16, s18, 3, v24 // (op 2623) 


s_mul_i32 s1, s42, s7 // (op 2625) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2625) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2625) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2625) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2625) most significant: sum
s_add_u32 s1, s1, s19 // (op 2625) most significant: sum


s_mov_b32 s72, s44 // (op 2625) reg expression
s_mov_b32 s73, s45 // (op 2625) reg expression
s_mov_b32 s74, s0 // (op 2625) reg expression
s_mov_b32 s75, 131072 // (op 2625) call()

v_and_b32 v24, -4, v1 // (op 2716) 
v_sub_u32 v26, v1, v24 // (op 2716) 

v_lshlrev_b32 v24, 4, v26 // (op 2716) 

v_lshrrev_b32 v26, 2, v1 // (op 2716) 
v_lshl_add_u32 v21, v26, 6, v24 // (op 2716) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 2716) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -2, v1 // (op 2768) 
v_sub_u32 v26, v1, v24 // (op 2768) 

v_lshlrev_b32 v24, 6, v26 // (op 2768) 

v_lshrrev_b32 v26, 1, v1 // (op 2768) 
v_lshl_add_u32 v21, v26, 8, v24 // (op 2768) 


v_mul_lo_u32 v22, v21, 6 // (op 2768) 

v_lshrrev_b32 v21, 3, v22 // (op 2768) 

v_and_b32 v22, -64, v1 // (op 2826) 
v_sub_u32 v24, v1, v22 // (op 2826) 

v_and_b32 v22, 4294967232, v1 // (op 2826) 
v_add_u32 v26, v22, v24 // (op 2826) 


v_lshlrev_b32 v22, 2, v1 // (op 2879) 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 877) 
s_and_b32 s1, s18, 4294967294 // (op 877) 

s_ashr_i64 s[18:19], s[30:31], 7 // (op 876) 
s_mov_b32 s22, s18 // (op 876) convert

s_cmp_lt_u32 s1, s22 // (op 876) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_876 // (op 876) Condition: Top (jump to RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_876 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_876:
s_barrier  // (op 3120) 
s_add_u32 s1, s1, 1 // (op 878) 

s_lshl_b64 s[18:19], s[26:27], 6 // (op 2396) 
v_mov_b64 v[14:15], s[18:19] // (op 2396) convert


v_mov_b64 v[24:25], v[30:31] // (op 2396) 
v_mov_b64 v[28:29], v[24:25] // (op 2396) reg expression
s_mov_b32 m0, s0 // (op 2396) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 2396) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 2396) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 2396) most significant half

s_add_u32 m0, m0, 4096 // (op 2396) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 2396) Load value direct to lds



s_lshl_b64 s[18:19], s[26:27], 7 // (op 2415) 
v_mov_b32 v18, s18 // (op 2415) 
v_mov_b32 v20, s19 // (op 2415) 
v_add_co_u32 v30, s[50:51], v30, v18 // (op 2415) least significant half
v_addc_co_u32 v31, s[50:51], v31, v20, s[50:51] // (op 2415) most significant half





s_mul_i32 s19, s33, 6 // (op 225) most significant: low of high * low
s_mul_i32 s18, s32, 6 // (op 225) least significant
s_mul_hi_u32 s22, s32, 6 // (op 225) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 225) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[18:19] // (op 225) 


v_mov_b32 v18, v19 // (op 225) 
v_mov_b32 v20, v18 // (op 225) 
buffer_load_dwordx4 v[32:35], v20, s[64:67], 0 offen // (op 225) Load value
buffer_load_dwordx2 v[36:37], v20, s[64:67], 0 offen offset:16 // (op 225) Load value
buffer_load_dwordx4 v[38:41], v20, s[64:67], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx2 v[42:43], v20, s[64:67], 0 offen offset:40 // (op 225) Load value
v_add_u32 v20, v20, v14 // (op 225) 
buffer_load_dwordx4 v[44:47], v20, s[64:67], 0 offen // (op 225) Load value
buffer_load_dwordx2 v[48:49], v20, s[64:67], 0 offen offset:16 // (op 225) Load value
buffer_load_dwordx4 v[50:53], v20, s[64:67], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx2 v[54:55], v20, s[64:67], 0 offen offset:40 // (op 225) Load value



v_add_u32 v19, 96, v19 // (op 2487) 


s_barrier  // (op 3139) 
v_mov_b32 v18, v17 // (op 226) 



v_mov_b32 v20, v18 // (op 226) reg expression

buffer_load_ubyte v14, v20, s[68:71], 0 offen offset: 0 // (op 226) Load value


s_lshl_b32 s18, s36, 2 // (op 2551) 
v_add_u32 v17, v17, s18 // (op 2551) 

s_waitcnt vmcnt(8)
ds_write_b128 v21, v[32:35] offset:16384 // (op 222) Store local data 


v_mov_b32 v15, v16 // (op 227) 

buffer_load_dword v18, v15, s[72:75], 0 offen // (op 227) Load value

v_add_u32 v16, 4, v16 // (op 2615) 
s_waitcnt vmcnt(8)
ds_write_b64 v21, v[36:37] offset:16400 // (op 222) Store local data 
s_barrier  // (op 3129) 
s_waitcnt vmcnt(7)
ds_write_b128 v21, v[38:41] offset:16408 // (op 222) Store local data 



ds_read_b64_tr_b8 v[56:57], v3 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v3 offset:512 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v3 offset:2048 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v3 offset:2560 // (op 205) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b64 v21, v[42:43] offset:16424 // (op 222) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v21, v[44:47] offset:16480 // (op 222) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v21, v[48:49] offset:16496 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v21, v[50:53] offset:16504 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v21, v[54:55] offset:16520 // (op 222) Store local data 

s_barrier  // (op 3160) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3150) 






ds_read_b128 v[32:35], v9 offset:16384 // (op 207) Load local data 
ds_read_b64 v[36:37], v9 offset:16400 // (op 207) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v26, v14 offset:41984 // (op 223) Store local data 

s_barrier  // (op 3180) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3170) 





ds_read_u8 v20, v0 offset:41984 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v22, v18 offset:40960 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3190) 



ds_read_u8 v18, v6 offset:40960 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[56:63], v[32:37], a[0:15], v20, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 212) 





ds_read_b64_tr_b8 v[32:33], v3 offset:4096 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:4608 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:6144 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:6656 // (op 214) Transpose load local data 



ds_read_b128 v[40:43], v9 offset:16432 // (op 216) Load local data 
ds_read_b64 v[44:45], v9 offset:16448 // (op 216) Load local data 


ds_read_u8 v18, v0 offset:42112 // (op 218) Load local data 



ds_read_u8 v14, v6 offset:40962 // (op 220) Load local data 



ds_read_b128 v[46:49], v9 offset:22528 // (op 416) Load local data 
ds_read_b64 v[50:51], v9 offset:22544 // (op 416) Load local data 



ds_read_u8 v15, v6 offset:41216 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[56:63], v[46:51], a[16:31], v20, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 421) 





ds_read_b128 v[46:49], v9 offset:22576 // (op 425) Load local data 
ds_read_b64 v[50:51], v9 offset:22592 // (op 425) Load local data 



ds_read_u8 v15, v6 offset:41218 // (op 429) Load local data 



ds_read_b128 v[64:67], v9 offset:28672 // (op 650) Load local data 
ds_read_b64 v[68:69], v9 offset:28688 // (op 650) Load local data 



ds_read_u8 v23, v6 offset:41472 // (op 654) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[56:63], v[64:69], a[32:47], v20, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 655) 





ds_read_b128 v[64:67], v9 offset:28720 // (op 659) Load local data 
ds_read_b64 v[68:69], v9 offset:28736 // (op 659) Load local data 



ds_read_u8 v23, v6 offset:41474 // (op 663) Load local data 



ds_read_b128 v[70:73], v9 offset:34816 // (op 884) Load local data 
ds_read_b64 v[74:75], v9 offset:34832 // (op 884) Load local data 



ds_read_u8 v24, v6 offset:41728 // (op 888) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[56:63], v[70:75], a[48:63], v20, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 889) 







ds_read_b128 v[52:55], v9 offset:34864 // (op 893) Load local data 
ds_read_b64 v[56:57], v9 offset:34880 // (op 893) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:39], v[40:45], a[0:15], v18, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 221) 





ds_read_u8 v14, v6 offset:41730 // (op 897) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:39], v[46:51], a[16:31], v18, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 430) 


v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[32:39], v[64:69], a[32:47], v18, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 664) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:39], v[52:57], a[48:63], v18, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 898) 




s_ashr_i64 s[18:19], s[30:31], 7 // (op 876) 
s_mov_b32 s22, s18 // (op 876) convert

s_cmp_lt_u32 s1, s22 // (op 876) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_876 // (op 876) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_876 if true)
RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_876:























s_barrier  // (op 3303) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_157
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
s_branch RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_157:


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
RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_157:












s_endpgm  // End of RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.LRR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end:
.size RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_, .LRR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end-RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 216
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 68096 // lds bytes
  .amdhsa_accum_offset 152
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
  - .name: RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 68096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 150
    .agpr_count: 64
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

