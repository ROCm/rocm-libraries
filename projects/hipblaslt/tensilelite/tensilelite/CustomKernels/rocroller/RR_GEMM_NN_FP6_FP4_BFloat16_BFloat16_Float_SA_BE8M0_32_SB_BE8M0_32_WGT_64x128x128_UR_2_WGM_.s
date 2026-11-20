// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_,@function
RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 4512) 
s_mul_hi_i32 s1, s16, s10 // (op 4769) 
v_and_b32 v0, 63, v1 // (op 4574) 
v_and_b32 v3, 48, v1 // (op 4574) 
v_sub_u32 v4, v0, v3 // (op 4574) 

v_lshrrev_b32 v3, 4, v0 // (op 4574) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 4574) 


v_lshrrev_b32 v3, 7, v1 // (op 4574) 
v_lshlrev_b32 v4, 4, v3 // (op 4574) 

v_lshl_add_u32 v3, v0, 6, v4 // (op 4574) 


v_mul_lo_u32 v0, v3, 6 // (op 4574) 
v_lshrrev_b32 v4, 3, v0 // (op 4574) 

v_lshrrev_b32 v0, 4, v3 // (op 4574) 

v_lshlrev_b32 v3, 2, v0 // (op 4574) 

v_add_u32 v0, v4, v3 // (op 4574) 


v_lshrrev_b32 v3, 6, v1 // (op 4632) 
v_and_b32 v4, 1, v3 // (op 4632) 

v_and_b32 v3, 15, v1 // (op 4632) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 4632) 


v_and_b32 v3, 63, v1 // (op 4632) 
v_lshrrev_b32 v4, 4, v3 // (op 4632) 

v_lshlrev_b32 v3, 5, v4 // (op 4632) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 4632) 


v_lshlrev_b32 v3, 2, v4 // (op 4632) 

v_lshrrev_b32 v4, 3, v3 // (op 4632) 

v_lshrrev_b32 v3, 7, v1 // (op 4690) 
v_and_b32 v5, 15, v1 // (op 4690) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 4690) 


v_and_b32 v3, 63, v1 // (op 4690) 
v_lshrrev_b32 v5, 4, v3 // (op 4690) 

v_lshl_add_u32 v3, v5, 6, v6 // (op 4690) 


v_lshrrev_b32 v5, 6, v1 // (op 4748) 
v_and_b32 v6, 1, v5 // (op 4748) 

v_and_b32 v5, 15, v1 // (op 4748) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4748) 


v_and_b32 v5, 63, v1 // (op 4748) 
v_lshrrev_b32 v6, 4, v5 // (op 4748) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 4748) 


v_and_b32 v6, 63, v1 // (op 4800) 
v_and_b32 v7, 48, v1 // (op 4800) 
v_sub_u32 v8, v6, v7 // (op 4800) 

v_lshrrev_b32 v7, 4, v6 // (op 4800) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 4800) 


v_lshrrev_b32 v7, 7, v1 // (op 4800) 
v_lshlrev_b32 v8, 4, v7 // (op 4800) 

v_lshl_add_u32 v7, v6, 6, v8 // (op 4800) 


v_mul_lo_u32 v6, v7, 6 // (op 4800) 
v_lshrrev_b32 v8, 3, v6 // (op 4800) 

v_lshrrev_b32 v6, 4, v7 // (op 4800) 

v_lshlrev_b32 v7, 2, v6 // (op 4800) 

v_add_u32 v6, v8, v7 // (op 4800) 


v_lshrrev_b32 v7, 6, v1 // (op 4829) 
v_and_b32 v8, 1, v7 // (op 4829) 

v_and_b32 v7, 15, v1 // (op 4829) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 4829) 


v_and_b32 v7, 63, v1 // (op 4829) 
v_lshrrev_b32 v8, 4, v7 // (op 4829) 

v_lshlrev_b32 v7, 5, v8 // (op 4829) 

v_lshl_add_u32 v8, v9, 7, v7 // (op 4829) 


v_lshlrev_b32 v7, 2, v8 // (op 4829) 

v_lshrrev_b32 v8, 3, v7 // (op 4829) 

v_lshrrev_b32 v7, 7, v1 // (op 4858) 
v_and_b32 v9, 15, v1 // (op 4858) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 4858) 


v_and_b32 v7, 63, v1 // (op 4858) 
v_lshrrev_b32 v9, 4, v7 // (op 4858) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 4858) 


v_lshrrev_b32 v9, 6, v1 // (op 4887) 
v_and_b32 v10, 1, v9 // (op 4887) 

v_and_b32 v9, 15, v1 // (op 4887) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4887) 


v_and_b32 v9, 63, v1 // (op 4887) 
v_lshrrev_b32 v10, 4, v9 // (op 4887) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 4887) 


v_lshrrev_b32 v10, 7, v1 // (op 4512) 
s_add_i32 s18, s0, s16 // (op 4512) 

v_and_b32 v11, 63, v1 // (op 4512) 
s_ashr_i32 s0, s18, 31 // (op 4512) 
v_lshrrev_b32 v12, 4, v11 // (op 4512) 
s_cmp_eq_i32 s10, 0 // (op 4512) 
s_mov_b32 s19, scc // (op 4512) 
v_lshlrev_b32 v13, 2, v12 // (op 4512) 

s_cmp_eq_i32 s19, 0 // (op 4512) 
s_cselect_b32 s51, 0, -1 // (op 4512) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4512) 


s_and_b32 s19, s11, 31 // (op 4512) 
v_lshrrev_b32 v10, 2, v11 // (op 4512) 

v_and_b32 v11, 3, v10 // (op 4512) 

v_lshrrev_b32 v10, 7, v1 // (op 4769) 
v_and_b32 v13, 63, v1 // (op 4769) 
s_lshl_b32 s52, 1, s19 // (op 4512) 
v_and_b32 v14, 3, v1 // (op 4512) 
s_add_i32 s53, s52, s51 // (op 4512) 


s_and_b32 s51, s0, s53 // (op 4512) 


s_add_i32 s0, s1, s16 // (op 4769) 

s_ashr_i32 s1, s0, 31 // (op 4769) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 4512) 


s_add_i32 s52, s18, s51 // (op 4512) 


v_lshrrev_b32 v11, 6, v1 // (op 4512) 
s_ashr_i32 s18, s52, s19 // (op 4512) 


v_and_b32 v14, 1, v11 // (op 4512) 

s_lshl_b32 s19, s11, 24 // (op 4512) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 4512) 


s_ashr_i32 s51, s19, 31 // (op 4512) 

s_xor_b32 s19, s18, s51 // (op 4512) 

s_sub_i32 s18, s19, s51 // (op 4512) 


s_mul_hi_i32 s19, s18, s12 // (op 4512) 
s_add_i32 s51, s19, s18 // (op 4512) 

s_ashr_i32 s19, s51, 31 // (op 4512) 
s_cmp_eq_i32 s12, 0 // (op 4512) 
s_mov_b32 s52, scc // (op 4512) 
s_cmp_eq_i32 s52, 0 // (op 4512) 
s_cselect_b32 s53, 0, -1 // (op 4512) 

s_and_b32 s52, s13, 31 // (op 4512) 
v_lshrrev_b32 v14, 4, v13 // (op 4769) 
s_cmp_eq_i32 s10, 0 // (op 4769) 
s_mov_b32 s54, scc // (op 4769) 
v_lshlrev_b32 v15, 2, v14 // (op 4769) 

s_cmp_eq_i32 s54, 0 // (op 4769) 
s_cselect_b32 s55, 0, -1 // (op 4769) 

s_lshl_b32 s54, 1, s52 // (op 4512) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 4769) 


v_lshrrev_b32 v10, 2, v13 // (op 4769) 

s_add_i32 s56, s54, s53 // (op 4512) 


s_and_b32 s53, s19, s56 // (op 4512) 


s_add_i32 s19, s51, s53 // (op 4512) 


s_ashr_i32 s51, s19, s52 // (op 4512) 


s_lshl_b32 s19, s13, 24 // (op 4512) 
s_ashr_i32 s52, s19, 31 // (op 4512) 

s_xor_b32 s19, s51, s52 // (op 4512) 

s_sub_i32 s51, s19, s52 // (op 4512) 


s_mul_i32 s19, s18, s15 // (op 4512) 
s_sub_u32 s52, s16, s19 // (op 4512) 

s_mul_i32 s19, s51, s15 // (op 4512) 
s_add_u32 s19, s19, s52 // (op 4512) 

s_mul_hi_i32 s52, s16, s9 // (op 4512) 
s_add_i32 s53, s52, s16 // (op 4512) 

s_ashr_i32 s52, s53, 31 // (op 4512) 
s_cmp_eq_i32 s9, 0 // (op 4512) 
s_mov_b32 s54, scc // (op 4512) 
s_cmp_eq_i32 s54, 0 // (op 4512) 
s_cselect_b32 s56, 0, -1 // (op 4512) 

s_and_b32 s54, s11, 31 // (op 4769) 
v_and_b32 v13, 3, v10 // (op 4769) 

s_lshl_b32 s57, 1, s54 // (op 4769) 
v_and_b32 v10, 3, v1 // (op 4769) 
s_add_i32 s58, s57, s55 // (op 4769) 


s_and_b32 s55, s1, s58 // (op 4769) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 4769) 


s_add_i32 s1, s0, s55 // (op 4769) 


v_lshrrev_b32 v10, 6, v1 // (op 4769) 
s_ashr_i32 s0, s1, s54 // (op 4769) 


v_and_b32 v13, 1, v10 // (op 4769) 

s_lshl_b32 s1, s11, 24 // (op 4769) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 4769) 


s_ashr_i32 s54, s1, 31 // (op 4769) 

s_xor_b32 s1, s0, s54 // (op 4769) 

s_sub_i32 s0, s1, s54 // (op 4769) 


s_mul_hi_i32 s1, s0, s12 // (op 4769) 
s_add_i32 s54, s1, s0 // (op 4769) 

s_ashr_i32 s1, s54, 31 // (op 4769) 
s_cmp_eq_i32 s12, 0 // (op 4769) 
s_mov_b32 s55, scc // (op 4769) 
s_cmp_eq_i32 s55, 0 // (op 4769) 
s_cselect_b32 s57, 0, -1 // (op 4769) 

s_and_b32 s55, s13, 31 // (op 4769) 
s_lshl_b32 s58, 1, s55 // (op 4769) 
s_add_i32 s59, s58, s57 // (op 4769) 


s_and_b32 s57, s1, s59 // (op 4769) 


s_add_i32 s1, s54, s57 // (op 4769) 


s_ashr_i32 s54, s1, s55 // (op 4769) 


s_lshl_b32 s1, s13, 24 // (op 4769) 
s_ashr_i32 s55, s1, 31 // (op 4769) 

s_xor_b32 s1, s54, s55 // (op 4769) 

s_sub_i32 s54, s1, s55 // (op 4769) 


s_mul_i32 s1, s0, s15 // (op 4769) 
s_sub_u32 s55, s16, s1 // (op 4769) 

s_mul_i32 s1, s54, s15 // (op 4769) 
s_add_u32 s1, s1, s55 // (op 4769) 

s_mul_hi_i32 s55, s16, s9 // (op 4769) 
s_add_i32 s57, s55, s16 // (op 4769) 

s_ashr_i32 s55, s57, 31 // (op 4769) 
s_cmp_eq_i32 s9, 0 // (op 4769) 
s_mov_b32 s58, scc // (op 4769) 
s_cmp_eq_i32 s58, 0 // (op 4769) 
s_cselect_b32 s59, 0, -1 // (op 4769) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 4512) 
s_lshl_b32 s60, 1, s58 // (op 4512) 
s_add_i32 s61, s60, s56 // (op 4512) 


s_and_b32 s56, s52, s61 // (op 4512) 


s_add_i32 s52, s53, s56 // (op 4512) 


s_ashr_i32 s53, s52, s58 // (op 4512) 


s_lshl_b32 s52, s46, 24 // (op 4512) 
s_ashr_i32 s56, s52, 31 // (op 4512) 

s_xor_b32 s52, s53, s56 // (op 4512) 

s_sub_i32 s53, s52, s56 // (op 4512) 


s_and_b32 s52, s46, 31 // (op 4769) 
s_lshl_b32 s56, 1, s52 // (op 4769) 
s_add_i32 s58, s56, s59 // (op 4769) 


s_and_b32 s56, s55, s58 // (op 4769) 


s_add_i32 s55, s57, s56 // (op 4769) 


s_ashr_i32 s56, s55, s52 // (op 4769) 


s_lshl_b32 s52, s46, 24 // (op 4769) 
s_ashr_i32 s55, s52, 31 // (op 4769) 

s_xor_b32 s52, s56, s55 // (op 4769) 

s_sub_i32 s56, s52, s55 // (op 4769) 


s_add_u32 s58, s6, 128 // (op 4512) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 4512) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 4512) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 4512) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 7 // (op 4512) 

s_mul_i32 s52, s15, s58 // (op 4512) 
s_mul_i32 s55, s8, s52 // (op 4512) 

s_mul_i32 s52, s53, s55 // (op 4512) 

s_sub_u32 s55, s16, s52 // (op 4512) 

s_mov_b32 s52, s55 // (op 4512) 
s_mul_hi_i32 s57, s52, s14 // (op 4512) 
s_add_i32 s60, s57, s52 // (op 4512) 


s_ashr_i32 s52, s60, 31 // (op 4512) 
s_cmp_eq_i32 s14, 0 // (op 4512) 
s_mov_b32 s57, scc // (op 4512) 
s_cmp_eq_i32 s57, 0 // (op 4512) 
s_cselect_b32 s61, 0, -1 // (op 4512) 

s_and_b32 s57, s48, 31 // (op 4512) 
s_lshl_b32 s62, 1, s57 // (op 4512) 
s_add_i32 s63, s62, s61 // (op 4512) 


s_and_b32 s61, s52, s63 // (op 4512) 


s_add_i32 s52, s60, s61 // (op 4512) 


s_ashr_i32 s60, s52, s57 // (op 4512) 


s_lshl_b32 s52, s48, 24 // (op 4512) 
s_ashr_i32 s57, s52, 31 // (op 4512) 

s_xor_b32 s52, s60, s57 // (op 4512) 

s_sub_i32 s60, s52, s57 // (op 4512) 


s_mul_i32 s52, s47, s15 // (op 4512) 
s_add_u32 s62, s20, 64 // (op 4512) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 4512) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 4512) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 4512) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 4512) 

s_sub_i32 s57, s62, s52 // (op 4512) 

s_mul_i32 s61, s60, s57 // (op 4512) 

s_sub_u32 s57, s55, s61 // (op 4512) 


s_add_u32 s55, s57, s52 // (op 4512) 


s_cmp_eq_i32 s53, 0 // (op 4512) 
s_mov_b32 s52, scc // (op 4512) 

s_cmp_eq_i32 s52, 0 // (op 4512) 
s_cselect_b32 s53, s55, s19 // (op 4512) 


s_mul_i32 s19, s51, s58 // (op 4512) 


v_lshl_add_u32 v13, s53, 6, v12 // (op 4512) 


s_sub_i32 s51, s18, s19 // (op 4512) 


s_cmp_eq_i32 s52, 0 // (op 4512) 
s_cselect_b32 s53, s60, s51 // (op 4512) 



v_lshl_add_u32 v12, s53, 7, v11 // (op 4512) 


v_mul_lo_u32 v11, v12, s2 // (op 4512) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 4512) 


s_add_u32 s18, s6, 128 // (op 4769) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 4769) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 4769) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 4769) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 7 // (op 4769) 

s_mul_i32 s55, s15, s18 // (op 4769) 
s_mul_i32 s51, s8, s55 // (op 4769) 

s_mul_i32 s55, s56, s51 // (op 4769) 

s_sub_u32 s51, s16, s55 // (op 4769) 

s_mov_b32 s55, s51 // (op 4769) 
s_mul_hi_i32 s52, s55, s14 // (op 4769) 
s_add_i32 s53, s52, s55 // (op 4769) 


s_ashr_i32 s52, s53, 31 // (op 4769) 
s_cmp_eq_i32 s14, 0 // (op 4769) 
s_mov_b32 s55, scc // (op 4769) 
s_cmp_eq_i32 s55, 0 // (op 4769) 
s_cselect_b32 s57, 0, -1 // (op 4769) 

s_and_b32 s55, s48, 31 // (op 4769) 
s_lshl_b32 s58, 1, s55 // (op 4769) 
s_add_i32 s59, s58, s57 // (op 4769) 


s_and_b32 s57, s52, s59 // (op 4769) 


s_add_i32 s52, s53, s57 // (op 4769) 


s_ashr_i32 s53, s52, s55 // (op 4769) 


s_lshl_b32 s52, s48, 24 // (op 4769) 
s_ashr_i32 s55, s52, 31 // (op 4769) 

s_xor_b32 s52, s53, s55 // (op 4769) 

s_sub_i32 s53, s52, s55 // (op 4769) 


s_mul_i32 s52, s47, s15 // (op 4769) 
s_add_u32 s58, s20, 64 // (op 4769) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 4769) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 4769) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 4769) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 4769) 

s_sub_i32 s55, s58, s52 // (op 4769) 

s_mul_i32 s57, s53, s55 // (op 4769) 

s_sub_u32 s55, s51, s57 // (op 4769) 


s_add_u32 s51, s55, s52 // (op 4769) 


s_cmp_eq_i32 s56, 0 // (op 4769) 
s_mov_b32 s52, scc // (op 4769) 

s_cmp_eq_i32 s52, 0 // (op 4769) 
s_cselect_b32 s55, s51, s1 // (op 4769) 


s_mul_i32 s1, s54, s18 // (op 4769) 


v_lshl_add_u32 v11, s55, 6, v14 // (op 4769) 


s_sub_i32 s51, s0, s1 // (op 4769) 


s_cmp_eq_i32 s52, 0 // (op 4769) 
s_cselect_b32 s0, s53, s51 // (op 4769) 



v_lshl_add_u32 v13, s0, 7, v10 // (op 4769) 


v_mul_lo_u32 v10, v13, s4 // (op 4769) 

v_add_lshl_u32 v13, v11, v10, 1 // (op 4769) 


s_mul_i32 s1, s2, s7 // (op 4514) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 4514) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 4514) least significant
s_mul_hi_u32 s19, s2, s6 // (op 4514) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4514) most significant: sum
s_add_u32 s1, s1, s19 // (op 4514) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 4514) 

s_mul_i32 s1, s4, s7 // (op 4771) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 4771) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 4771) least significant
s_mul_hi_u32 s52, s4, s6 // (op 4771) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 4771) most significant: sum
s_add_u32 s1, s1, s52 // (op 4771) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 4771) 

s_mov_b32 s56, s22 // (op 4514) reg expression
s_mov_b32 s57, s23 // (op 4514) reg expression
s_mov_b32 s58, s18 // (op 4514) reg expression
s_mov_b32 s59, 131072 // (op 4514) call()

s_mov_b32 s60, s24 // (op 4771) reg expression

s_mov_b32 s61, s25 // (op 4771) reg expression
s_mov_b32 s62, s52 // (op 4771) reg expression
s_mov_b32 s63, 131072 // (op 4771) call()


v_mov_b32 v10, s49 // (op 2) Move value
v_mov_b32 v11, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 340) call()
v_accvgpr_write a5, 0.00000 // (op 340) call()
v_accvgpr_write a6, 0.00000 // (op 340) call()
v_accvgpr_write a7, 0.00000 // (op 340) call()
v_accvgpr_write a8, 0.00000 // (op 516) call()
v_accvgpr_write a9, 0.00000 // (op 516) call()
v_accvgpr_write a10, 0.00000 // (op 516) call()
v_accvgpr_write a11, 0.00000 // (op 516) call()
v_accvgpr_write a12, 0.00000 // (op 692) call()
v_accvgpr_write a13, 0.00000 // (op 692) call()
v_accvgpr_write a14, 0.00000 // (op 692) call()
v_accvgpr_write a15, 0.00000 // (op 692) call()
v_accvgpr_write a16, 0.00000 // (op 956) call()
v_accvgpr_write a17, 0.00000 // (op 956) call()
v_accvgpr_write a18, 0.00000 // (op 956) call()
v_accvgpr_write a19, 0.00000 // (op 956) call()
v_accvgpr_write a20, 0.00000 // (op 1265) call()
v_accvgpr_write a21, 0.00000 // (op 1265) call()
v_accvgpr_write a22, 0.00000 // (op 1265) call()
v_accvgpr_write a23, 0.00000 // (op 1265) call()
v_accvgpr_write a24, 0.00000 // (op 1266) call()
v_accvgpr_write a25, 0.00000 // (op 1266) call()
v_accvgpr_write a26, 0.00000 // (op 1266) call()
v_accvgpr_write a27, 0.00000 // (op 1266) call()
v_accvgpr_write a28, 0.00000 // (op 1267) call()
v_accvgpr_write a29, 0.00000 // (op 1267) call()
v_accvgpr_write a30, 0.00000 // (op 1267) call()
v_accvgpr_write a31, 0.00000 // (op 1267) call()
s_mul_hi_i32 s0, s16, s10 // (op 4274) 
v_and_b32 v14, -2, v1 // (op 4274) 
s_add_i32 s1, s0, s16 // (op 4274) 

v_sub_u32 v15, v1, v14 // (op 4274) 

s_ashr_i32 s0, s1, 31 // (op 4274) 
v_lshlrev_b32 v14, 5, v15 // (op 4274) 

s_cmp_eq_i32 s10, 0 // (op 4274) 
s_mov_b32 s18, scc // (op 4274) 
v_lshrrev_b32 v15, 1, v1 // (op 4274) 
s_cmp_eq_i32 s18, 0 // (op 4274) 
s_cselect_b32 s19, 0, -1 // (op 4274) 

s_and_b32 s18, s11, 31 // (op 4274) 
s_lshl_b32 s22, 1, s18 // (op 4274) 
s_add_i32 s23, s22, s19 // (op 4274) 


s_and_b32 s19, s0, s23 // (op 4274) 


s_add_i32 s0, s1, s19 // (op 4274) 


s_ashr_i32 s1, s0, s18 // (op 4274) 


s_lshl_b32 s0, s11, 24 // (op 4274) 
s_ashr_i32 s18, s0, 31 // (op 4274) 

s_xor_b32 s0, s1, s18 // (op 4274) 

s_sub_i32 s1, s0, s18 // (op 4274) 


s_mul_hi_i32 s0, s1, s12 // (op 4274) 
s_add_i32 s18, s0, s1 // (op 4274) 

s_ashr_i32 s0, s18, 31 // (op 4274) 
s_cmp_eq_i32 s12, 0 // (op 4274) 
s_mov_b32 s19, scc // (op 4274) 
s_cmp_eq_i32 s19, 0 // (op 4274) 
s_cselect_b32 s22, 0, -1 // (op 4274) 

s_and_b32 s19, s13, 31 // (op 4274) 
s_lshl_b32 s23, 1, s19 // (op 4274) 
s_add_i32 s24, s23, s22 // (op 4274) 


s_and_b32 s25, s0, s24 // (op 4274) 


s_add_i32 s0, s18, s25 // (op 4274) 


s_ashr_i32 s18, s0, s19 // (op 4274) 


s_lshl_b32 s0, s13, 24 // (op 4274) 
s_ashr_i32 s19, s0, 31 // (op 4274) 

s_xor_b32 s0, s18, s19 // (op 4274) 

s_sub_i32 s18, s0, s19 // (op 4274) 


s_mul_i32 s0, s1, s15 // (op 4274) 

s_sub_u32 s1, s16, s0 // (op 4274) 

s_mul_i32 s0, s18, s15 // (op 4274) 
s_add_u32 s0, s0, s1 // (op 4274) 


s_mul_hi_i32 s1, s16, s9 // (op 4274) 
s_add_i32 s18, s1, s16 // (op 4274) 

s_ashr_i32 s1, s18, 31 // (op 4274) 
s_cmp_eq_i32 s9, 0 // (op 4274) 
s_mov_b32 s19, scc // (op 4274) 
s_cmp_eq_i32 s19, 0 // (op 4274) 
s_cselect_b32 s22, 0, -1 // (op 4274) 

s_and_b32 s19, s46, 31 // (op 4274) 
s_lshl_b32 s23, 1, s19 // (op 4274) 
s_add_i32 s24, s23, s22 // (op 4274) 


s_and_b32 s25, s1, s24 // (op 4274) 


s_add_i32 s1, s18, s25 // (op 4274) 


s_ashr_i32 s18, s1, s19 // (op 4274) 


s_lshl_b32 s1, s46, 24 // (op 4274) 
s_ashr_i32 s19, s1, 31 // (op 4274) 

s_xor_b32 s1, s18, s19 // (op 4274) 

s_sub_i32 s18, s1, s19 // (op 4274) 


s_add_u32 s22, s6, 128 // (op 4274) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4274) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4274) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4274) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4274) 

s_mul_i32 s1, s15, s22 // (op 4274) 

s_mul_i32 s19, s8, s1 // (op 4274) 

s_mul_i32 s1, s18, s19 // (op 4274) 

s_sub_u32 s19, s16, s1 // (op 4274) 

s_mov_b32 s1, s19 // (op 4274) 
s_mul_hi_i32 s22, s1, s14 // (op 4274) 
s_add_i32 s23, s22, s1 // (op 4274) 


s_ashr_i32 s1, s23, 31 // (op 4274) 
s_cmp_eq_i32 s14, 0 // (op 4274) 
s_mov_b32 s22, scc // (op 4274) 
s_cmp_eq_i32 s22, 0 // (op 4274) 
s_cselect_b32 s24, 0, -1 // (op 4274) 

s_and_b32 s22, s48, 31 // (op 4274) 
s_lshl_b32 s25, 1, s22 // (op 4274) 
s_add_i32 s49, s25, s24 // (op 4274) 


s_and_b32 s24, s1, s49 // (op 4274) 


s_add_i32 s1, s23, s24 // (op 4274) 


s_ashr_i32 s23, s1, s22 // (op 4274) 


s_lshl_b32 s1, s48, 24 // (op 4274) 
s_ashr_i32 s22, s1, 31 // (op 4274) 

s_xor_b32 s1, s23, s22 // (op 4274) 

s_sub_i32 s23, s1, s22 // (op 4274) 


s_mul_i32 s1, s47, s15 // (op 4274) 
s_add_u32 s24, s20, 64 // (op 4274) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4274) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 4274) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 4274) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 4274) 

s_sub_i32 s22, s24, s1 // (op 4274) 

s_mul_i32 s24, s23, s22 // (op 4274) 


s_sub_u32 s25, s19, s24 // (op 4274) 


s_add_u32 s19, s25, s1 // (op 4274) 


s_cmp_eq_i32 s18, 0 // (op 4274) 
s_mov_b32 s1, scc // (op 4274) 

s_cmp_eq_i32 s1, 0 // (op 4274) 
s_cselect_b32 s18, s19, s0 // (op 4274) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 4274) 


v_mov_b32 v19, 0 // (op 4274) convert
v_mov_b32 v18, v16 // (op 4274) convert

v_mul_lo_u32 v17, v15, s27 // (op 4274) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 4274) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 4274) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 4274) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 4274) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 4274) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 4274) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 4274) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 4274) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 4274) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 4274) 

v_mov_b32 v16, v14 // (op 4274) convert

s_mul_i32 s1, s31, 6 // (op 4276) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 4276) least significant
s_mul_hi_u32 s18, s30, 6 // (op 4276) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4276) most significant: sum

s_mul_i32 s19, s26, s1 // (op 4276) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 4276) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 4276) least significant
s_mul_hi_u32 s23, s26, s0 // (op 4276) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 4276) most significant: sum
s_add_u32 s19, s19, s23 // (op 4276) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4276) 

s_mov_b32 s52, s28 // (op 4276) reg expression
s_mov_b32 s53, s29 // (op 4276) reg expression
s_mov_b32 s54, s0 // (op 4276) reg expression
s_mov_b32 s55, 131072 // (op 4276) call()

s_mul_hi_i32 s0, s16, s10 // (op 4293) 
v_and_b32 v14, -2, v1 // (op 4293) 
s_add_i32 s1, s0, s16 // (op 4293) 

v_sub_u32 v15, v1, v14 // (op 4293) 

s_ashr_i32 s0, s1, 31 // (op 4293) 
v_lshlrev_b32 v14, 5, v15 // (op 4293) 

s_cmp_eq_i32 s10, 0 // (op 4293) 
s_mov_b32 s18, scc // (op 4293) 
v_lshrrev_b32 v15, 1, v1 // (op 4293) 
s_cmp_eq_i32 s18, 0 // (op 4293) 
s_cselect_b32 s19, 0, -1 // (op 4293) 

s_and_b32 s18, s11, 31 // (op 4293) 
s_lshl_b32 s22, 1, s18 // (op 4293) 
s_add_i32 s23, s22, s19 // (op 4293) 


s_and_b32 s19, s0, s23 // (op 4293) 


s_add_i32 s0, s1, s19 // (op 4293) 


s_ashr_i32 s1, s0, s18 // (op 4293) 


s_lshl_b32 s0, s11, 24 // (op 4293) 
s_ashr_i32 s18, s0, 31 // (op 4293) 

s_xor_b32 s0, s1, s18 // (op 4293) 

s_sub_i32 s1, s0, s18 // (op 4293) 


s_mul_hi_i32 s0, s1, s12 // (op 4293) 
s_add_i32 s18, s0, s1 // (op 4293) 

s_ashr_i32 s0, s18, 31 // (op 4293) 
s_cmp_eq_i32 s12, 0 // (op 4293) 
s_mov_b32 s19, scc // (op 4293) 
s_cmp_eq_i32 s19, 0 // (op 4293) 
s_cselect_b32 s22, 0, -1 // (op 4293) 

s_and_b32 s19, s13, 31 // (op 4293) 
s_lshl_b32 s23, 1, s19 // (op 4293) 
s_add_i32 s24, s23, s22 // (op 4293) 


s_and_b32 s25, s0, s24 // (op 4293) 


s_add_i32 s0, s18, s25 // (op 4293) 


s_ashr_i32 s18, s0, s19 // (op 4293) 


s_lshl_b32 s0, s13, 24 // (op 4293) 
s_ashr_i32 s19, s0, 31 // (op 4293) 

s_xor_b32 s0, s18, s19 // (op 4293) 

s_sub_i32 s18, s0, s19 // (op 4293) 


s_mul_i32 s0, s1, s15 // (op 4293) 

s_sub_u32 s1, s16, s0 // (op 4293) 

s_mul_i32 s0, s18, s15 // (op 4293) 
s_add_u32 s0, s0, s1 // (op 4293) 


s_mul_hi_i32 s1, s16, s9 // (op 4293) 
s_add_i32 s18, s1, s16 // (op 4293) 

s_ashr_i32 s1, s18, 31 // (op 4293) 
s_cmp_eq_i32 s9, 0 // (op 4293) 
s_mov_b32 s19, scc // (op 4293) 
s_cmp_eq_i32 s19, 0 // (op 4293) 
s_cselect_b32 s22, 0, -1 // (op 4293) 

s_and_b32 s19, s46, 31 // (op 4293) 
s_lshl_b32 s23, 1, s19 // (op 4293) 
s_add_i32 s24, s23, s22 // (op 4293) 


s_and_b32 s25, s1, s24 // (op 4293) 


s_add_i32 s1, s18, s25 // (op 4293) 


s_ashr_i32 s18, s1, s19 // (op 4293) 


s_lshl_b32 s1, s46, 24 // (op 4293) 
s_ashr_i32 s19, s1, 31 // (op 4293) 

s_xor_b32 s1, s18, s19 // (op 4293) 

s_sub_i32 s18, s1, s19 // (op 4293) 


s_add_u32 s50, s6, 128 // (op 4293) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4293) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4293) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4293) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4293) 

s_mul_i32 s1, s15, s24 // (op 4293) 

s_mul_i32 s19, s8, s1 // (op 4293) 

s_mul_i32 s1, s18, s19 // (op 4293) 

s_sub_u32 s19, s16, s1 // (op 4293) 

s_mov_b32 s1, s19 // (op 4293) 
s_mul_hi_i32 s22, s1, s14 // (op 4293) 
s_add_i32 s23, s22, s1 // (op 4293) 


s_ashr_i32 s1, s23, 31 // (op 4293) 
s_cmp_eq_i32 s14, 0 // (op 4293) 
s_mov_b32 s22, scc // (op 4293) 
s_cmp_eq_i32 s22, 0 // (op 4293) 
s_cselect_b32 s24, 0, -1 // (op 4293) 

s_and_b32 s22, s48, 31 // (op 4293) 
s_lshl_b32 s25, 1, s22 // (op 4293) 
s_add_i32 s49, s25, s24 // (op 4293) 


s_and_b32 s24, s1, s49 // (op 4293) 


s_add_i32 s1, s23, s24 // (op 4293) 


s_ashr_i32 s23, s1, s22 // (op 4293) 


s_lshl_b32 s1, s48, 24 // (op 4293) 
s_ashr_i32 s22, s1, 31 // (op 4293) 

s_xor_b32 s1, s23, s22 // (op 4293) 

s_sub_i32 s23, s1, s22 // (op 4293) 


s_mul_i32 s1, s47, s15 // (op 4293) 
s_add_u32 s24, s20, 64 // (op 4293) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4293) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 4293) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 4293) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 4293) 

s_sub_i32 s22, s24, s1 // (op 4293) 

s_mul_i32 s24, s23, s22 // (op 4293) 


s_sub_u32 s25, s19, s24 // (op 4293) 


s_add_u32 s19, s25, s1 // (op 4293) 


s_cmp_eq_i32 s18, 0 // (op 4293) 
s_mov_b32 s1, scc // (op 4293) 

s_cmp_eq_i32 s1, 0 // (op 4293) 
s_cselect_b32 s18, s19, s0 // (op 4293) 



v_lshl_add_u32 v17, s18, 6, v14 // (op 4293) 


v_mov_b32 v19, 0 // (op 4293) convert
v_mov_b32 v18, v17 // (op 4293) convert

v_mul_lo_u32 v21, v15, s27 // (op 4293) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 4293) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 4293) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 4293) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 4293) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 4293) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 4293) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 4293) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 4293) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 4293) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 4293) 

v_mov_b32 v17, v14 // (op 4293) convert

s_mul_i32 s1, s31, 6 // (op 4295) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 4295) least significant
s_mul_hi_u32 s18, s30, 6 // (op 4295) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4295) most significant: sum

s_mul_i32 s19, s26, s1 // (op 4295) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 4295) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 4295) least significant
s_mul_hi_u32 s23, s26, s0 // (op 4295) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 4295) most significant: sum
s_add_u32 s19, s19, s23 // (op 4295) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4295) 

s_mov_b32 s52, s28 // (op 4295) reg expression
s_mov_b32 s53, s29 // (op 4295) reg expression
s_mov_b32 s54, s0 // (op 4295) reg expression
s_mov_b32 s55, 131072 // (op 4295) call()

s_mul_hi_i32 s0, s16, s10 // (op 4338) 
v_lshrrev_b32 v14, 2, v1 // (op 4338) 
s_add_i32 s1, s0, s16 // (op 4338) 

v_lshlrev_b32 v15, 1, v14 // (op 4338) 

s_ashr_i32 s0, s1, 31 // (op 4338) 
v_and_b32 v14, -4, v1 // (op 4338) 
s_cmp_eq_i32 s10, 0 // (op 4338) 
s_mov_b32 s18, scc // (op 4338) 
v_sub_u32 v18, v1, v14 // (op 4338) 

s_cmp_eq_i32 s18, 0 // (op 4338) 
s_cselect_b32 s19, 0, -1 // (op 4338) 

v_lshlrev_b32 v14, 5, v18 // (op 4338) 

v_mov_b32 v19, 0 // (op 4338) convert
v_mov_b32 v18, v14 // (op 4338) convert

s_and_b32 s18, s11, 31 // (op 4338) 
s_lshl_b32 s22, 1, s18 // (op 4338) 
s_add_i32 s23, s22, s19 // (op 4338) 


s_and_b32 s19, s0, s23 // (op 4338) 


s_add_i32 s0, s1, s19 // (op 4338) 


s_ashr_i32 s1, s0, s18 // (op 4338) 


s_lshl_b32 s0, s11, 24 // (op 4338) 
s_ashr_i32 s18, s0, 31 // (op 4338) 

s_xor_b32 s0, s1, s18 // (op 4338) 

s_sub_i32 s1, s0, s18 // (op 4338) 


s_mul_hi_i32 s0, s1, s12 // (op 4338) 
s_add_i32 s18, s0, s1 // (op 4338) 

s_ashr_i32 s0, s18, 31 // (op 4338) 
s_cmp_eq_i32 s12, 0 // (op 4338) 
s_mov_b32 s19, scc // (op 4338) 
s_cmp_eq_i32 s19, 0 // (op 4338) 
s_cselect_b32 s22, 0, -1 // (op 4338) 

s_and_b32 s19, s13, 31 // (op 4338) 
s_lshl_b32 s23, 1, s19 // (op 4338) 
s_add_i32 s24, s23, s22 // (op 4338) 


s_and_b32 s25, s0, s24 // (op 4338) 


s_add_i32 s0, s18, s25 // (op 4338) 


s_ashr_i32 s18, s0, s19 // (op 4338) 


s_lshl_b32 s0, s13, 24 // (op 4338) 
s_ashr_i32 s19, s0, 31 // (op 4338) 

s_xor_b32 s0, s18, s19 // (op 4338) 

s_sub_i32 s18, s0, s19 // (op 4338) 


s_add_u32 s50, s6, 128 // (op 4338) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4338) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4338) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4338) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4338) 

s_mul_i32 s0, s18, s24 // (op 4338) 

s_sub_i32 s18, s1, s0 // (op 4338) 


s_mul_hi_i32 s19, s16, s9 // (op 4338) 
s_add_i32 s0, s19, s16 // (op 4338) 

s_ashr_i32 s1, s0, 31 // (op 4338) 
s_cmp_eq_i32 s9, 0 // (op 4338) 
s_mov_b32 s19, scc // (op 4338) 
s_cmp_eq_i32 s19, 0 // (op 4338) 
s_cselect_b32 s22, 0, -1 // (op 4338) 

s_and_b32 s19, s46, 31 // (op 4338) 
s_lshl_b32 s23, 1, s19 // (op 4338) 
s_add_i32 s49, s23, s22 // (op 4338) 


s_and_b32 s22, s1, s49 // (op 4338) 


s_add_i32 s1, s0, s22 // (op 4338) 


s_ashr_i32 s0, s1, s19 // (op 4338) 


s_lshl_b32 s1, s46, 24 // (op 4338) 
s_ashr_i32 s19, s1, 31 // (op 4338) 

s_xor_b32 s1, s0, s19 // (op 4338) 

s_sub_i32 s0, s1, s19 // (op 4338) 


s_mul_i32 s1, s15, s24 // (op 4338) 

s_mul_i32 s19, s8, s1 // (op 4338) 

s_mul_i32 s1, s0, s19 // (op 4338) 

s_sub_u32 s19, s16, s1 // (op 4338) 

s_mul_hi_i32 s1, s19, s14 // (op 4338) 
s_add_i32 s22, s1, s19 // (op 4338) 


s_ashr_i32 s1, s22, 31 // (op 4338) 
s_cmp_eq_i32 s14, 0 // (op 4338) 
s_mov_b32 s19, scc // (op 4338) 
s_cmp_eq_i32 s19, 0 // (op 4338) 
s_cselect_b32 s23, 0, -1 // (op 4338) 

s_and_b32 s19, s48, 31 // (op 4338) 
s_lshl_b32 s24, 1, s19 // (op 4338) 
s_add_i32 s25, s24, s23 // (op 4338) 


s_and_b32 s23, s1, s25 // (op 4338) 


s_add_i32 s1, s22, s23 // (op 4338) 


s_ashr_i32 s22, s1, s19 // (op 4338) 


s_lshl_b32 s1, s48, 24 // (op 4338) 
s_ashr_i32 s19, s1, 31 // (op 4338) 

s_xor_b32 s1, s22, s19 // (op 4338) 

s_sub_i32 s22, s1, s19 // (op 4338) 


s_cmp_eq_i32 s0, 0 // (op 4338) 
s_mov_b32 s1, scc // (op 4338) 

s_cmp_eq_i32 s1, 0 // (op 4338) 
s_cselect_b32 s0, s22, s18 // (op 4338) 



v_lshl_add_u32 v14, s0, 7, v15 // (op 4338) 


v_mul_lo_u32 v21, v14, s33 // (op 4338) most significant: low of low * high
v_mul_hi_u32 v15, v14, s32 // (op 4338) most significant: high of low * low
v_mul_lo_u32 v20, v14, s32 // (op 4338) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 4338) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 4338) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 4338) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 4338) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 4338) 

v_mov_b32 v14, v18 // (op 4338) convert

s_mul_i32 s1, s32, s7 // (op 4340) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 4340) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 4340) least significant
s_mul_hi_u32 s19, s32, s6 // (op 4340) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4340) most significant: sum
s_add_u32 s1, s1, s19 // (op 4340) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4340) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 4340) 

s_mov_b32 s64, s34 // (op 4340) reg expression
s_mov_b32 s65, s35 // (op 4340) reg expression
s_mov_b32 s66, s0 // (op 4340) reg expression
s_mov_b32 s67, 131072 // (op 4340) call()

s_mul_hi_i32 s0, s16, s10 // (op 4357) 
v_lshrrev_b32 v15, 2, v1 // (op 4357) 
s_add_i32 s1, s0, s16 // (op 4357) 

v_lshlrev_b32 v18, 1, v15 // (op 4357) 

s_ashr_i32 s0, s1, 31 // (op 4357) 
v_and_b32 v15, -4, v1 // (op 4357) 
s_cmp_eq_i32 s10, 0 // (op 4357) 
s_mov_b32 s18, scc // (op 4357) 
v_sub_u32 v19, v1, v15 // (op 4357) 

s_cmp_eq_i32 s18, 0 // (op 4357) 
s_cselect_b32 s19, 0, -1 // (op 4357) 

v_lshlrev_b32 v15, 5, v19 // (op 4357) 

v_mov_b32 v21, 0 // (op 4357) convert
v_mov_b32 v20, v15 // (op 4357) convert

s_and_b32 s18, s11, 31 // (op 4357) 
s_lshl_b32 s22, 1, s18 // (op 4357) 
s_add_i32 s23, s22, s19 // (op 4357) 


s_and_b32 s19, s0, s23 // (op 4357) 


s_add_i32 s0, s1, s19 // (op 4357) 


s_ashr_i32 s1, s0, s18 // (op 4357) 


s_lshl_b32 s0, s11, 24 // (op 4357) 
s_ashr_i32 s18, s0, 31 // (op 4357) 

s_xor_b32 s0, s1, s18 // (op 4357) 

s_sub_i32 s1, s0, s18 // (op 4357) 


s_mul_hi_i32 s0, s1, s12 // (op 4357) 
s_add_i32 s18, s0, s1 // (op 4357) 

s_ashr_i32 s0, s18, 31 // (op 4357) 
s_cmp_eq_i32 s12, 0 // (op 4357) 
s_mov_b32 s19, scc // (op 4357) 
s_cmp_eq_i32 s19, 0 // (op 4357) 
s_cselect_b32 s22, 0, -1 // (op 4357) 

s_and_b32 s19, s13, 31 // (op 4357) 
s_lshl_b32 s23, 1, s19 // (op 4357) 
s_add_i32 s24, s23, s22 // (op 4357) 


s_and_b32 s25, s0, s24 // (op 4357) 


s_add_i32 s0, s18, s25 // (op 4357) 


s_ashr_i32 s18, s0, s19 // (op 4357) 


s_lshl_b32 s0, s13, 24 // (op 4357) 
s_ashr_i32 s19, s0, 31 // (op 4357) 

s_xor_b32 s0, s18, s19 // (op 4357) 

s_sub_i32 s18, s0, s19 // (op 4357) 


s_add_u32 s50, s6, 128 // (op 4357) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4357) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4357) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4357) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4357) 

s_mul_i32 s0, s18, s24 // (op 4357) 

s_sub_i32 s18, s1, s0 // (op 4357) 


s_mul_hi_i32 s19, s16, s9 // (op 4357) 
s_add_i32 s0, s19, s16 // (op 4357) 

s_ashr_i32 s1, s0, 31 // (op 4357) 
s_cmp_eq_i32 s9, 0 // (op 4357) 
s_mov_b32 s19, scc // (op 4357) 
s_cmp_eq_i32 s19, 0 // (op 4357) 
s_cselect_b32 s22, 0, -1 // (op 4357) 

s_and_b32 s19, s46, 31 // (op 4357) 
s_lshl_b32 s23, 1, s19 // (op 4357) 
s_add_i32 s49, s23, s22 // (op 4357) 


s_and_b32 s22, s1, s49 // (op 4357) 


s_add_i32 s1, s0, s22 // (op 4357) 


s_ashr_i32 s0, s1, s19 // (op 4357) 


s_lshl_b32 s1, s46, 24 // (op 4357) 
s_ashr_i32 s19, s1, 31 // (op 4357) 

s_xor_b32 s1, s0, s19 // (op 4357) 

s_sub_i32 s0, s1, s19 // (op 4357) 


s_mul_i32 s1, s15, s24 // (op 4357) 

s_mul_i32 s19, s8, s1 // (op 4357) 

s_mul_i32 s1, s0, s19 // (op 4357) 

s_sub_u32 s19, s16, s1 // (op 4357) 

s_mul_hi_i32 s1, s19, s14 // (op 4357) 
s_add_i32 s22, s1, s19 // (op 4357) 


s_ashr_i32 s1, s22, 31 // (op 4357) 
s_cmp_eq_i32 s14, 0 // (op 4357) 
s_mov_b32 s19, scc // (op 4357) 
s_cmp_eq_i32 s19, 0 // (op 4357) 
s_cselect_b32 s23, 0, -1 // (op 4357) 

s_and_b32 s19, s48, 31 // (op 4357) 
s_lshl_b32 s24, 1, s19 // (op 4357) 
s_add_i32 s25, s24, s23 // (op 4357) 


s_and_b32 s23, s1, s25 // (op 4357) 


s_add_i32 s1, s22, s23 // (op 4357) 


s_ashr_i32 s22, s1, s19 // (op 4357) 


s_lshl_b32 s1, s48, 24 // (op 4357) 
s_ashr_i32 s19, s1, 31 // (op 4357) 

s_xor_b32 s1, s22, s19 // (op 4357) 

s_sub_i32 s22, s1, s19 // (op 4357) 


s_cmp_eq_i32 s0, 0 // (op 4357) 
s_mov_b32 s1, scc // (op 4357) 

s_cmp_eq_i32 s1, 0 // (op 4357) 
s_cselect_b32 s0, s22, s18 // (op 4357) 



v_lshl_add_u32 v15, s0, 7, v18 // (op 4357) 


v_mul_lo_u32 v19, v15, s33 // (op 4357) most significant: low of low * high
v_mul_hi_u32 v22, v15, s32 // (op 4357) most significant: high of low * low
v_mul_lo_u32 v18, v15, s32 // (op 4357) least significant: low of low * low
v_add_u32 v19, v19, v22 // (op 4357) most significant: sum


v_add_co_u32 v22, s[0:1], v20, v18 // (op 4357) least significant half
v_addc_co_u32 v23, s[0:1], v21, v19, s[0:1] // (op 4357) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 4357) 


v_lshrrev_b64 v[18:19], 3, v[22:23] // (op 4357) 

v_mov_b32 v15, v18 // (op 4357) convert

s_mul_i32 s1, s32, s7 // (op 4359) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 4359) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 4359) least significant
s_mul_hi_u32 s19, s32, s6 // (op 4359) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4359) most significant: sum
s_add_u32 s1, s1, s19 // (op 4359) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4359) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 4359) 

s_mov_b32 s64, s34 // (op 4359) reg expression
s_mov_b32 s65, s35 // (op 4359) reg expression
s_mov_b32 s66, s0 // (op 4359) reg expression
s_mov_b32 s67, 131072 // (op 4359) call()

s_mul_hi_i32 s0, s16, s10 // (op 4402) 
v_and_b32 v18, -64, v1 // (op 4402) 
s_add_i32 s1, s0, s16 // (op 4402) 

v_sub_u32 v19, v1, v18 // (op 4402) 

s_ashr_i32 s0, s1, 31 // (op 4402) 
v_lshrrev_b32 v18, 6, v1 // (op 4402) 
s_cmp_eq_i32 s10, 0 // (op 4402) 
s_mov_b32 s18, scc // (op 4402) 
s_cmp_eq_i32 s18, 0 // (op 4402) 
s_cselect_b32 s19, 0, -1 // (op 4402) 

s_and_b32 s18, s11, 31 // (op 4402) 
s_lshl_b32 s22, 1, s18 // (op 4402) 
s_add_i32 s23, s22, s19 // (op 4402) 


s_and_b32 s19, s0, s23 // (op 4402) 


s_add_i32 s0, s1, s19 // (op 4402) 


s_ashr_i32 s1, s0, s18 // (op 4402) 


s_lshl_b32 s0, s11, 24 // (op 4402) 
s_ashr_i32 s18, s0, 31 // (op 4402) 

s_xor_b32 s0, s1, s18 // (op 4402) 

s_sub_i32 s1, s0, s18 // (op 4402) 


s_mul_hi_i32 s0, s1, s12 // (op 4402) 
s_add_i32 s18, s0, s1 // (op 4402) 

s_ashr_i32 s0, s18, 31 // (op 4402) 
s_cmp_eq_i32 s12, 0 // (op 4402) 
s_mov_b32 s19, scc // (op 4402) 
s_cmp_eq_i32 s19, 0 // (op 4402) 
s_cselect_b32 s22, 0, -1 // (op 4402) 

s_and_b32 s19, s13, 31 // (op 4402) 
s_lshl_b32 s23, 1, s19 // (op 4402) 
s_add_i32 s24, s23, s22 // (op 4402) 


s_and_b32 s25, s0, s24 // (op 4402) 


s_add_i32 s0, s18, s25 // (op 4402) 


s_ashr_i32 s18, s0, s19 // (op 4402) 


s_lshl_b32 s0, s13, 24 // (op 4402) 
s_ashr_i32 s19, s0, 31 // (op 4402) 

s_xor_b32 s0, s18, s19 // (op 4402) 

s_sub_i32 s18, s0, s19 // (op 4402) 


s_mul_i32 s0, s1, s15 // (op 4402) 

s_sub_u32 s1, s16, s0 // (op 4402) 

s_mul_i32 s0, s18, s15 // (op 4402) 
s_add_u32 s0, s0, s1 // (op 4402) 


s_mul_hi_i32 s1, s16, s9 // (op 4402) 
s_add_i32 s18, s1, s16 // (op 4402) 

s_ashr_i32 s1, s18, 31 // (op 4402) 
s_cmp_eq_i32 s9, 0 // (op 4402) 
s_mov_b32 s19, scc // (op 4402) 
s_cmp_eq_i32 s19, 0 // (op 4402) 
s_cselect_b32 s22, 0, -1 // (op 4402) 

s_and_b32 s19, s46, 31 // (op 4402) 
s_lshl_b32 s23, 1, s19 // (op 4402) 
s_add_i32 s24, s23, s22 // (op 4402) 


s_and_b32 s25, s1, s24 // (op 4402) 


s_add_i32 s1, s18, s25 // (op 4402) 


s_ashr_i32 s18, s1, s19 // (op 4402) 


s_lshl_b32 s1, s46, 24 // (op 4402) 
s_ashr_i32 s19, s1, 31 // (op 4402) 

s_xor_b32 s1, s18, s19 // (op 4402) 

s_sub_i32 s18, s1, s19 // (op 4402) 


s_add_u32 s50, s6, 128 // (op 4402) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4402) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4402) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4402) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4402) 

s_mul_i32 s1, s15, s24 // (op 4402) 

s_mul_i32 s19, s8, s1 // (op 4402) 

s_mul_i32 s1, s18, s19 // (op 4402) 

s_sub_u32 s19, s16, s1 // (op 4402) 

s_mov_b32 s1, s19 // (op 4402) 
s_mul_hi_i32 s22, s1, s14 // (op 4402) 
s_add_i32 s23, s22, s1 // (op 4402) 


s_ashr_i32 s1, s23, 31 // (op 4402) 
s_cmp_eq_i32 s14, 0 // (op 4402) 
s_mov_b32 s22, scc // (op 4402) 
s_cmp_eq_i32 s22, 0 // (op 4402) 
s_cselect_b32 s24, 0, -1 // (op 4402) 

s_and_b32 s22, s48, 31 // (op 4402) 
s_lshl_b32 s25, 1, s22 // (op 4402) 
s_add_i32 s49, s25, s24 // (op 4402) 


s_and_b32 s24, s1, s49 // (op 4402) 


s_add_i32 s1, s23, s24 // (op 4402) 


s_ashr_i32 s23, s1, s22 // (op 4402) 


s_lshl_b32 s1, s48, 24 // (op 4402) 
s_ashr_i32 s22, s1, 31 // (op 4402) 

s_xor_b32 s1, s23, s22 // (op 4402) 

s_sub_i32 s23, s1, s22 // (op 4402) 


s_mul_i32 s1, s47, s15 // (op 4402) 
s_add_u32 s24, s20, 64 // (op 4402) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4402) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 4402) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 4402) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 4402) 

s_sub_i32 s22, s24, s1 // (op 4402) 

s_mul_i32 s24, s23, s22 // (op 4402) 


s_sub_u32 s25, s19, s24 // (op 4402) 


s_add_u32 s19, s25, s1 // (op 4402) 


s_cmp_eq_i32 s18, 0 // (op 4402) 
s_mov_b32 s1, scc // (op 4402) 

s_cmp_eq_i32 s1, 0 // (op 4402) 
s_cselect_b32 s18, s19, s0 // (op 4402) 



v_lshl_add_u32 v20, s18, 6, v19 // (op 4402) 


v_mul_lo_u32 v19, v18, s36 // (op 4402) 

v_add_u32 v18, v20, v19 // (op 4402) 


s_mul_i32 s1, s36, s41 // (op 4404) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 4404) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 4404) least significant
s_mul_hi_u32 s19, s36, s40 // (op 4404) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4404) most significant: sum
s_add_u32 s1, s1, s19 // (op 4404) most significant: sum


s_mov_b32 s68, s38 // (op 4404) reg expression
s_mov_b32 s69, s39 // (op 4404) reg expression
s_mov_b32 s70, s0 // (op 4404) reg expression
s_mov_b32 s71, 131072 // (op 4404) call()

s_mul_hi_i32 s0, s16, s10 // (op 4421) 
v_and_b32 v19, -64, v1 // (op 4421) 
s_add_i32 s1, s0, s16 // (op 4421) 

v_sub_u32 v20, v1, v19 // (op 4421) 

s_ashr_i32 s0, s1, 31 // (op 4421) 
v_lshrrev_b32 v19, 6, v1 // (op 4421) 
s_cmp_eq_i32 s10, 0 // (op 4421) 
s_mov_b32 s18, scc // (op 4421) 
s_cmp_eq_i32 s18, 0 // (op 4421) 
s_cselect_b32 s19, 0, -1 // (op 4421) 

s_and_b32 s18, s11, 31 // (op 4421) 
s_lshl_b32 s22, 1, s18 // (op 4421) 
s_add_i32 s23, s22, s19 // (op 4421) 


s_and_b32 s19, s0, s23 // (op 4421) 


s_add_i32 s0, s1, s19 // (op 4421) 


s_ashr_i32 s1, s0, s18 // (op 4421) 


s_lshl_b32 s0, s11, 24 // (op 4421) 
s_ashr_i32 s18, s0, 31 // (op 4421) 

s_xor_b32 s0, s1, s18 // (op 4421) 

s_sub_i32 s1, s0, s18 // (op 4421) 


s_mul_hi_i32 s0, s1, s12 // (op 4421) 
s_add_i32 s18, s0, s1 // (op 4421) 

s_ashr_i32 s0, s18, 31 // (op 4421) 
s_cmp_eq_i32 s12, 0 // (op 4421) 
s_mov_b32 s19, scc // (op 4421) 
s_cmp_eq_i32 s19, 0 // (op 4421) 
s_cselect_b32 s22, 0, -1 // (op 4421) 

s_and_b32 s19, s13, 31 // (op 4421) 
s_lshl_b32 s23, 1, s19 // (op 4421) 
s_add_i32 s24, s23, s22 // (op 4421) 


s_and_b32 s25, s0, s24 // (op 4421) 


s_add_i32 s0, s18, s25 // (op 4421) 


s_ashr_i32 s18, s0, s19 // (op 4421) 


s_lshl_b32 s0, s13, 24 // (op 4421) 
s_ashr_i32 s19, s0, 31 // (op 4421) 

s_xor_b32 s0, s18, s19 // (op 4421) 

s_sub_i32 s18, s0, s19 // (op 4421) 


s_mul_i32 s0, s1, s15 // (op 4421) 

s_sub_u32 s1, s16, s0 // (op 4421) 

s_mul_i32 s0, s18, s15 // (op 4421) 
s_add_u32 s0, s0, s1 // (op 4421) 


s_mul_hi_i32 s1, s16, s9 // (op 4421) 
s_add_i32 s18, s1, s16 // (op 4421) 

s_ashr_i32 s1, s18, 31 // (op 4421) 
s_cmp_eq_i32 s9, 0 // (op 4421) 
s_mov_b32 s19, scc // (op 4421) 
s_cmp_eq_i32 s19, 0 // (op 4421) 
s_cselect_b32 s22, 0, -1 // (op 4421) 

s_and_b32 s19, s46, 31 // (op 4421) 
s_lshl_b32 s23, 1, s19 // (op 4421) 
s_add_i32 s24, s23, s22 // (op 4421) 


s_and_b32 s25, s1, s24 // (op 4421) 


s_add_i32 s1, s18, s25 // (op 4421) 


s_ashr_i32 s18, s1, s19 // (op 4421) 


s_lshl_b32 s1, s46, 24 // (op 4421) 
s_ashr_i32 s19, s1, 31 // (op 4421) 

s_xor_b32 s1, s18, s19 // (op 4421) 

s_sub_i32 s18, s1, s19 // (op 4421) 


s_add_u32 s50, s6, 128 // (op 4421) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4421) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4421) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4421) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4421) 

s_mul_i32 s1, s15, s24 // (op 4421) 

s_mul_i32 s19, s8, s1 // (op 4421) 

s_mul_i32 s1, s18, s19 // (op 4421) 

s_sub_u32 s19, s16, s1 // (op 4421) 

s_mov_b32 s1, s19 // (op 4421) 
s_mul_hi_i32 s22, s1, s14 // (op 4421) 
s_add_i32 s23, s22, s1 // (op 4421) 


s_ashr_i32 s1, s23, 31 // (op 4421) 
s_cmp_eq_i32 s14, 0 // (op 4421) 
s_mov_b32 s22, scc // (op 4421) 
s_cmp_eq_i32 s22, 0 // (op 4421) 
s_cselect_b32 s24, 0, -1 // (op 4421) 

s_and_b32 s22, s48, 31 // (op 4421) 
s_lshl_b32 s25, 1, s22 // (op 4421) 
s_add_i32 s49, s25, s24 // (op 4421) 


s_and_b32 s24, s1, s49 // (op 4421) 


s_add_i32 s1, s23, s24 // (op 4421) 


s_ashr_i32 s23, s1, s22 // (op 4421) 


s_lshl_b32 s1, s48, 24 // (op 4421) 
s_ashr_i32 s22, s1, 31 // (op 4421) 

s_xor_b32 s1, s23, s22 // (op 4421) 

s_sub_i32 s23, s1, s22 // (op 4421) 


s_mul_i32 s1, s47, s15 // (op 4421) 
s_add_u32 s24, s20, 64 // (op 4421) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4421) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 4421) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 4421) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 4421) 

s_sub_i32 s22, s24, s1 // (op 4421) 

s_mul_i32 s24, s23, s22 // (op 4421) 


s_sub_u32 s25, s19, s24 // (op 4421) 


s_add_u32 s19, s25, s1 // (op 4421) 


s_cmp_eq_i32 s18, 0 // (op 4421) 
s_mov_b32 s1, scc // (op 4421) 

s_cmp_eq_i32 s1, 0 // (op 4421) 
s_cselect_b32 s18, s19, s0 // (op 4421) 



v_lshl_add_u32 v21, s18, 6, v20 // (op 4421) 


v_mul_lo_u32 v20, v19, s36 // (op 4421) 

v_add_u32 v19, v21, v20 // (op 4421) 


s_mul_i32 s1, s36, s41 // (op 4423) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 4423) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 4423) least significant
s_mul_hi_u32 s19, s36, s40 // (op 4423) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4423) most significant: sum
s_add_u32 s1, s1, s19 // (op 4423) most significant: sum


s_mov_b32 s68, s38 // (op 4423) reg expression
s_mov_b32 s69, s39 // (op 4423) reg expression
s_mov_b32 s70, s0 // (op 4423) reg expression
s_mov_b32 s71, 131072 // (op 4423) call()

s_mul_hi_i32 s0, s16, s10 // (op 4466) 
v_lshrrev_b32 v20, 2, v1 // (op 4466) 
s_add_i32 s1, s0, s16 // (op 4466) 

v_lshlrev_b32 v21, 1, v20 // (op 4466) 

s_ashr_i32 s0, s1, 31 // (op 4466) 
v_and_b32 v20, -4, v1 // (op 4466) 
s_cmp_eq_i32 s10, 0 // (op 4466) 
s_mov_b32 s18, scc // (op 4466) 
v_sub_u32 v22, v1, v20 // (op 4466) 

s_cmp_eq_i32 s18, 0 // (op 4466) 
s_cselect_b32 s19, 0, -1 // (op 4466) 

s_and_b32 s18, s11, 31 // (op 4466) 
s_lshl_b32 s22, 1, s18 // (op 4466) 
s_add_i32 s23, s22, s19 // (op 4466) 


s_and_b32 s19, s0, s23 // (op 4466) 


s_add_i32 s0, s1, s19 // (op 4466) 


s_ashr_i32 s1, s0, s18 // (op 4466) 


s_lshl_b32 s0, s11, 24 // (op 4466) 
s_ashr_i32 s18, s0, 31 // (op 4466) 

s_xor_b32 s0, s1, s18 // (op 4466) 

s_sub_i32 s1, s0, s18 // (op 4466) 


s_mul_hi_i32 s0, s1, s12 // (op 4466) 
s_add_i32 s18, s0, s1 // (op 4466) 

s_ashr_i32 s0, s18, 31 // (op 4466) 
s_cmp_eq_i32 s12, 0 // (op 4466) 
s_mov_b32 s19, scc // (op 4466) 
s_cmp_eq_i32 s19, 0 // (op 4466) 
s_cselect_b32 s22, 0, -1 // (op 4466) 

s_and_b32 s19, s13, 31 // (op 4466) 
s_lshl_b32 s23, 1, s19 // (op 4466) 
s_add_i32 s24, s23, s22 // (op 4466) 


s_and_b32 s25, s0, s24 // (op 4466) 


s_add_i32 s0, s18, s25 // (op 4466) 


s_ashr_i32 s18, s0, s19 // (op 4466) 


s_lshl_b32 s0, s13, 24 // (op 4466) 
s_ashr_i32 s19, s0, 31 // (op 4466) 

s_xor_b32 s0, s18, s19 // (op 4466) 

s_sub_i32 s18, s0, s19 // (op 4466) 


s_add_u32 s50, s6, 128 // (op 4466) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4466) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4466) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4466) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4466) 

s_mul_i32 s0, s18, s24 // (op 4466) 

s_sub_i32 s18, s1, s0 // (op 4466) 


s_mul_hi_i32 s19, s16, s9 // (op 4466) 
s_add_i32 s0, s19, s16 // (op 4466) 

s_ashr_i32 s1, s0, 31 // (op 4466) 
s_cmp_eq_i32 s9, 0 // (op 4466) 
s_mov_b32 s19, scc // (op 4466) 
s_cmp_eq_i32 s19, 0 // (op 4466) 
s_cselect_b32 s22, 0, -1 // (op 4466) 

s_and_b32 s19, s46, 31 // (op 4466) 
s_lshl_b32 s23, 1, s19 // (op 4466) 
s_add_i32 s49, s23, s22 // (op 4466) 


s_and_b32 s22, s1, s49 // (op 4466) 


s_add_i32 s1, s0, s22 // (op 4466) 


s_ashr_i32 s0, s1, s19 // (op 4466) 


s_lshl_b32 s1, s46, 24 // (op 4466) 
s_ashr_i32 s19, s1, 31 // (op 4466) 

s_xor_b32 s1, s0, s19 // (op 4466) 

s_sub_i32 s0, s1, s19 // (op 4466) 


s_mul_i32 s1, s15, s24 // (op 4466) 

s_mul_i32 s19, s8, s1 // (op 4466) 

s_mul_i32 s1, s0, s19 // (op 4466) 

s_sub_u32 s19, s16, s1 // (op 4466) 

s_mul_hi_i32 s1, s19, s14 // (op 4466) 
s_add_i32 s22, s1, s19 // (op 4466) 


s_ashr_i32 s1, s22, 31 // (op 4466) 
s_cmp_eq_i32 s14, 0 // (op 4466) 
s_mov_b32 s19, scc // (op 4466) 
s_cmp_eq_i32 s19, 0 // (op 4466) 
s_cselect_b32 s23, 0, -1 // (op 4466) 

s_and_b32 s19, s48, 31 // (op 4466) 
s_lshl_b32 s24, 1, s19 // (op 4466) 
s_add_i32 s25, s24, s23 // (op 4466) 


s_and_b32 s23, s1, s25 // (op 4466) 


s_add_i32 s1, s22, s23 // (op 4466) 


s_ashr_i32 s22, s1, s19 // (op 4466) 


s_lshl_b32 s1, s48, 24 // (op 4466) 
s_ashr_i32 s19, s1, 31 // (op 4466) 

s_xor_b32 s1, s22, s19 // (op 4466) 

s_sub_i32 s22, s1, s19 // (op 4466) 


s_cmp_eq_i32 s0, 0 // (op 4466) 
s_mov_b32 s1, scc // (op 4466) 

s_cmp_eq_i32 s1, 0 // (op 4466) 
s_cselect_b32 s0, s22, s18 // (op 4466) 



v_lshl_add_u32 v20, s0, 7, v21 // (op 4466) 


v_mul_lo_u32 v21, v20, s42 // (op 4466) 

v_add_u32 v20, v22, v21 // (op 4466) 


s_mul_i32 s1, s42, s7 // (op 4468) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 4468) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 4468) least significant
s_mul_hi_u32 s19, s42, s6 // (op 4468) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4468) most significant: sum
s_add_u32 s1, s1, s19 // (op 4468) most significant: sum


s_mov_b32 s72, s44 // (op 4468) reg expression
s_mov_b32 s73, s45 // (op 4468) reg expression
s_mov_b32 s74, s0 // (op 4468) reg expression
s_mov_b32 s75, 131072 // (op 4468) call()

s_mul_hi_i32 s0, s16, s10 // (op 4485) 
v_lshrrev_b32 v21, 2, v1 // (op 4485) 
s_add_i32 s1, s0, s16 // (op 4485) 

v_lshlrev_b32 v22, 1, v21 // (op 4485) 

s_ashr_i32 s0, s1, 31 // (op 4485) 
v_and_b32 v21, -4, v1 // (op 4485) 
s_cmp_eq_i32 s10, 0 // (op 4485) 
s_mov_b32 s18, scc // (op 4485) 
v_sub_u32 v23, v1, v21 // (op 4485) 

s_cmp_eq_i32 s18, 0 // (op 4485) 
s_cselect_b32 s19, 0, -1 // (op 4485) 

s_and_b32 s18, s11, 31 // (op 4485) 
s_lshl_b32 s22, 1, s18 // (op 4485) 
s_add_i32 s23, s22, s19 // (op 4485) 


s_and_b32 s19, s0, s23 // (op 4485) 


s_add_i32 s0, s1, s19 // (op 4485) 


s_ashr_i32 s1, s0, s18 // (op 4485) 


s_lshl_b32 s0, s11, 24 // (op 4485) 
s_ashr_i32 s18, s0, 31 // (op 4485) 

s_xor_b32 s0, s1, s18 // (op 4485) 

s_sub_i32 s1, s0, s18 // (op 4485) 


s_mul_hi_i32 s0, s1, s12 // (op 4485) 
s_add_i32 s18, s0, s1 // (op 4485) 

s_ashr_i32 s0, s18, 31 // (op 4485) 
s_cmp_eq_i32 s12, 0 // (op 4485) 
s_mov_b32 s19, scc // (op 4485) 
s_cmp_eq_i32 s19, 0 // (op 4485) 
s_cselect_b32 s22, 0, -1 // (op 4485) 

s_and_b32 s19, s13, 31 // (op 4485) 
s_lshl_b32 s23, 1, s19 // (op 4485) 
s_add_i32 s24, s23, s22 // (op 4485) 


s_and_b32 s25, s0, s24 // (op 4485) 


s_add_i32 s0, s18, s25 // (op 4485) 


s_ashr_i32 s18, s0, s19 // (op 4485) 


s_lshl_b32 s0, s13, 24 // (op 4485) 
s_ashr_i32 s19, s0, 31 // (op 4485) 

s_xor_b32 s0, s18, s19 // (op 4485) 

s_sub_i32 s18, s0, s19 // (op 4485) 


s_add_u32 s50, s6, 128 // (op 4485) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4485) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4485) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4485) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4485) 

s_mul_i32 s0, s18, s24 // (op 4485) 

s_sub_i32 s18, s1, s0 // (op 4485) 


s_mul_hi_i32 s19, s16, s9 // (op 4485) 
s_add_i32 s0, s19, s16 // (op 4485) 

s_ashr_i32 s1, s0, 31 // (op 4485) 
s_cmp_eq_i32 s9, 0 // (op 4485) 
s_mov_b32 s19, scc // (op 4485) 
s_cmp_eq_i32 s19, 0 // (op 4485) 
s_cselect_b32 s22, 0, -1 // (op 4485) 

s_and_b32 s19, s46, 31 // (op 4485) 
s_lshl_b32 s23, 1, s19 // (op 4485) 
s_add_i32 s49, s23, s22 // (op 4485) 


s_and_b32 s22, s1, s49 // (op 4485) 


s_add_i32 s1, s0, s22 // (op 4485) 


s_ashr_i32 s0, s1, s19 // (op 4485) 


s_lshl_b32 s1, s46, 24 // (op 4485) 
s_ashr_i32 s19, s1, 31 // (op 4485) 

s_xor_b32 s1, s0, s19 // (op 4485) 

s_sub_i32 s0, s1, s19 // (op 4485) 


s_mul_i32 s1, s15, s24 // (op 4485) 

s_mul_i32 s19, s8, s1 // (op 4485) 

s_mul_i32 s1, s0, s19 // (op 4485) 

s_sub_u32 s19, s16, s1 // (op 4485) 

s_mul_hi_i32 s1, s19, s14 // (op 4485) 
s_add_i32 s22, s1, s19 // (op 4485) 


s_ashr_i32 s1, s22, 31 // (op 4485) 
s_cmp_eq_i32 s14, 0 // (op 4485) 
s_mov_b32 s19, scc // (op 4485) 
s_cmp_eq_i32 s19, 0 // (op 4485) 
s_cselect_b32 s23, 0, -1 // (op 4485) 

s_and_b32 s19, s48, 31 // (op 4485) 
s_lshl_b32 s24, 1, s19 // (op 4485) 
s_add_i32 s25, s24, s23 // (op 4485) 


s_and_b32 s23, s1, s25 // (op 4485) 


s_add_i32 s1, s22, s23 // (op 4485) 


s_ashr_i32 s22, s1, s19 // (op 4485) 


s_lshl_b32 s1, s48, 24 // (op 4485) 
s_ashr_i32 s19, s1, 31 // (op 4485) 

s_xor_b32 s1, s22, s19 // (op 4485) 

s_sub_i32 s22, s1, s19 // (op 4485) 


s_cmp_eq_i32 s0, 0 // (op 4485) 
s_mov_b32 s1, scc // (op 4485) 

s_cmp_eq_i32 s1, 0 // (op 4485) 
s_cselect_b32 s0, s22, s18 // (op 4485) 



v_lshl_add_u32 v21, s0, 7, v22 // (op 4485) 


v_mul_lo_u32 v22, v21, s42 // (op 4485) 

v_add_u32 v21, v23, v22 // (op 4485) 


s_mul_i32 s1, s42, s7 // (op 4487) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 4487) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 4487) least significant
s_mul_hi_u32 s19, s42, s6 // (op 4487) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4487) most significant: sum
s_add_u32 s1, s1, s19 // (op 4487) most significant: sum


s_mov_b32 s72, s44 // (op 4487) reg expression
s_mov_b32 s73, s45 // (op 4487) reg expression
s_mov_b32 s74, s0 // (op 4487) reg expression
s_mov_b32 s75, 131072 // (op 4487) call()


s_lshl_b64 s[0:1], s[32:33], 2 // (op 4018) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 4018) 


v_mov_b32 v24, v14 // (op 4018) 
buffer_load_dwordx4 v[26:29], v24, s[64:67], 0 offen // (op 4018) Load value
v_add_u32 v24, v24, v22 // (op 4018) 
buffer_load_dwordx4 v[30:33], v24, s[64:67], 0 offen // (op 4018) Load value



v_mov_b64 v[22:23], s[42:43] // (op 4021) convert

v_mov_b32 v24, v20 // (op 4021) 
buffer_load_ubyte v34, v24, s[72:75], 0 offen offset: 0 // (op 4021) Load value
v_add_u32 v24, v24, v22 // (op 4021) 
buffer_load_ubyte v35, v24, s[72:75], 0 offen offset: 0 // (op 4021) Load value




v_mov_b32 v22, v16 // (op 4024) reg expression

buffer_load_dwordx3 v[36:38], v22, s[52:55], 0 offen // (op 4024) Load value
buffer_load_dwordx3 v[40:42], v22, s[52:55], 0 offen offset:12 // (op 4024) Load value



v_mov_b32 v39, v18 // (op 4027) reg expression

buffer_load_ubyte v22, v39, s[68:71], 0 offen offset: 0 // (op 4027) Load value


s_lshl_b64 s[0:1], s[32:33], 2 // (op 4030) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 4030) 


v_mov_b32 v23, v14 // (op 4030) 
buffer_load_dwordx4 v[44:47], v23, s[64:67], 0 offen offset:64 // (op 4030) Load value
v_add_u32 v23, v23, v24 // (op 4030) 
buffer_load_dwordx4 v[48:51], v23, s[64:67], 0 offen offset:64 // (op 4030) Load value




v_mov_b64 v[24:25], s[42:43] // (op 4033) convert

v_mov_b32 v14, v20 // (op 4033) 
buffer_load_ubyte v52, v14, s[72:75], 0 offen offset:4 // (op 4033) Load value
v_add_u32 v14, v14, v24 // (op 4033) 
buffer_load_ubyte v53, v14, s[72:75], 0 offen offset:4 // (op 4033) Load value





s_lshl_b64 s[0:1], s[26:27], 7 // (op 4036) 
s_mul_i32 s19, s1, 6 // (op 4036) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 4036) least significant
s_mul_hi_u32 s22, s0, 6 // (op 4036) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 4036) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 4036) 

v_add_u32 v14, v16, s0 // (op 4036) 


buffer_load_dwordx3 v[54:56], v14, s[52:55], 0 offen // (op 4036) Load value
buffer_load_dwordx3 v[58:60], v14, s[52:55], 0 offen offset:12 // (op 4036) Load value




s_lshl_b32 s0, s36, 2 // (op 4039) 
v_add_u32 v14, v18, s0 // (op 4039) 


buffer_load_ubyte v16, v14, s[68:71], 0 offen offset: 0 // (op 4039) Load value


v_and_b32 v14, -4, v1 // (op 4615) 
v_sub_u32 v18, v1, v14 // (op 4615) 

v_lshlrev_b32 v14, 5, v18 // (op 4615) 

v_lshrrev_b32 v18, 2, v1 // (op 4615) 
v_lshl_add_u32 v20, v18, 8, v14 // (op 4615) 


v_lshlrev_b32 v14, 2, v20 // (op 4615) 

v_lshrrev_b32 v18, 3, v14 // (op 4615) 




s_waitcnt vmcnt(13)
ds_write_b128 v18, v[26:29] // (op 4041) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[30:33] offset:64 // (op 4041) Store local data 

v_and_b32 v14, -4, v1 // (op 4731) 
v_sub_u32 v18, v1, v14 // (op 4731) 

v_lshrrev_b32 v14, 2, v1 // (op 4731) 
v_lshl_add_u32 v20, v14, 3, v18 // (op 4731) 





s_waitcnt vmcnt(11)
ds_write_b8 v20, v34 offset:8192 // (op 4042) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v20, v35 offset:8196 // (op 4042) Store local data 

v_and_b32 v14, -2, v1 // (op 4557) 
v_sub_u32 v18, v1, v14 // (op 4557) 

v_lshlrev_b32 v14, 5, v18 // (op 4557) 

v_lshrrev_b32 v18, 1, v1 // (op 4557) 
v_lshl_add_u32 v20, v18, 6, v14 // (op 4557) 


v_mul_lo_u32 v14, v20, 6 // (op 4557) 
v_lshrrev_b32 v18, 3, v14 // (op 4557) 

v_lshrrev_b32 v14, 4, v20 // (op 4557) 

v_lshlrev_b32 v20, 2, v14 // (op 4557) 

v_add_u32 v14, v18, v20 // (op 4557) 





s_waitcnt vmcnt(9)
ds_write_b96 v14, v[36:38] offset:8704 // (op 4043) Store local data 
s_waitcnt vmcnt(8)
ds_write_b96 v14, v[40:42] offset:8720 // (op 4043) Store local data 

v_and_b32 v14, -64, v1 // (op 4673) 
v_sub_u32 v18, v1, v14 // (op 4673) 

v_and_b32 v14, 4294967232, v1 // (op 4673) 
v_add_u32 v20, v14, v18 // (op 4673) 





s_waitcnt vmcnt(7)
ds_write_b8 v20, v22 offset:16896 // (op 4044) Store local data 

v_and_b32 v14, -2, v1 // (op 4541) 
v_sub_u32 v18, v1, v14 // (op 4541) 

v_lshlrev_b32 v14, 5, v18 // (op 4541) 

v_lshrrev_b32 v18, 1, v1 // (op 4541) 
v_lshl_add_u32 v20, v18, 6, v14 // (op 4541) 


v_mul_lo_u32 v14, v20, 6 // (op 4541) 
v_lshrrev_b32 v18, 3, v14 // (op 4541) 

v_lshrrev_b32 v14, 4, v20 // (op 4541) 

v_lshlrev_b32 v20, 2, v14 // (op 4541) 

v_add_u32 v14, v18, v20 // (op 4541) 


v_and_b32 v18, -4, v1 // (op 4599) 
v_sub_u32 v20, v1, v18 // (op 4599) 

v_lshlrev_b32 v18, 5, v20 // (op 4599) 

v_lshrrev_b32 v20, 2, v1 // (op 4599) 
v_lshl_add_u32 v39, v20, 8, v18 // (op 4599) 


v_lshlrev_b32 v18, 2, v39 // (op 4599) 

v_lshrrev_b32 v20, 3, v18 // (op 4599) 

v_and_b32 v18, -64, v1 // (op 4657) 
v_sub_u32 v39, v1, v18 // (op 4657) 

v_and_b32 v18, 4294967232, v1 // (op 4657) 
v_add_u32 v43, v18, v39 // (op 4657) 


v_and_b32 v18, -4, v1 // (op 4715) 
v_sub_u32 v39, v1, v18 // (op 4715) 

v_lshrrev_b32 v18, 2, v1 // (op 4715) 
v_lshl_add_u32 v57, v18, 3, v39 // (op 4715) 


v_and_b32 v18, -2, v1 // (op 4783) 
v_sub_u32 v39, v1, v18 // (op 4783) 

v_lshlrev_b32 v18, 5, v39 // (op 4783) 

v_lshrrev_b32 v39, 1, v1 // (op 4783) 
v_lshl_add_u32 v23, v39, 6, v18 // (op 4783) 


v_mul_lo_u32 v18, v23, 6 // (op 4783) 
v_lshrrev_b32 v39, 3, v18 // (op 4783) 

v_lshrrev_b32 v18, 4, v23 // (op 4783) 

v_lshlrev_b32 v23, 2, v18 // (op 4783) 

v_add_u32 v18, v39, v23 // (op 4783) 


v_and_b32 v39, -4, v1 // (op 4812) 
v_sub_u32 v23, v1, v39 // (op 4812) 

v_lshlrev_b32 v39, 5, v23 // (op 4812) 

v_lshrrev_b32 v23, 2, v1 // (op 4812) 
v_lshl_add_u32 v24, v23, 8, v39 // (op 4812) 


v_lshlrev_b32 v23, 2, v24 // (op 4812) 

v_lshrrev_b32 v39, 3, v23 // (op 4812) 

v_and_b32 v23, -64, v1 // (op 4841) 
v_sub_u32 v24, v1, v23 // (op 4841) 

v_and_b32 v23, 4294967232, v1 // (op 4841) 
v_add_u32 v25, v23, v24 // (op 4841) 


v_and_b32 v23, -4, v1 // (op 4870) 
v_sub_u32 v24, v1, v23 // (op 4870) 

v_lshrrev_b32 v23, 2, v1 // (op 4870) 
v_lshl_add_u32 v61, v23, 3, v24 // (op 4870) 


s_mov_b32 s0, 0 // (op 1051) call()
s_ashr_i64 s[18:19], s[30:31], 7 // (op 1050) 
s_mov_b32 s1, s18 // (op 1050) convert

s_and_b32 s18, s1, 4294967294 // (op 1050) 

s_cmp_lt_u32 s0, s18 // (op 1050) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoop_1050 // (op 1050) Condition: Top (jump to RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoop_1050 if false)
RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoop_1050:

s_add_u32 s0, s0, 2 // (op 1052) 
s_lshl_b64 s[18:19], s[32:33], 2 // (op 10) 
v_lshrrev_b64 v[62:63], 3, s[18:19] // (op 10) 


v_mov_b32 v23, v15 // (op 10) 
v_mov_b32 v24, v23 // (op 10) 
buffer_load_dwordx4 v[26:29], v24, s[64:67], 0 offen offset:128 // (op 10) Load value
v_add_u32 v24, v24, v62 // (op 10) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4950) 
buffer_load_dwordx4 v[30:33], v24, s[64:67], 0 offen offset:128 // (op 10) Load value






v_mov_b64 v[62:63], s[42:43] // (op 25) convert

v_mov_b32 v24, v21 // (op 25) 
v_mov_b32 v64, v24 // (op 25) 
buffer_load_ubyte v34, v64, s[72:75], 0 offen offset:8 // (op 25) Load value
v_add_u32 v64, v64, v62 // (op 25) 
ds_read_b96_tr_b6 v[66:68], v0 offset:8704 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[69:71], v0 offset:9728 // (op 72) Transpose load local data 
buffer_load_ubyte v35, v64, s[72:75], 0 offen offset:8 // (op 25) Load value




s_barrier  // (op 4963) 
v_mov_b32 v62, v17 // (op 4) 



s_lshl_b64 s[18:19], s[26:27], 7 // (op 4) 
s_mul_i32 s51, s19, 6 // (op 4) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 4) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v63, v62, s1 // (op 4) 


ds_read_b128 v[72:75], v4 // (op 77) Load local data 
buffer_load_dwordx3 v[36:38], v63, s[52:55], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[40:42], v63, s[52:55], 0 offen offset:12 // (op 4) Load value

s_waitcnt lgkmcnt(3)
s_barrier  // (op 4977) 




ds_read_u8 v63, v3 offset:16896 // (op 82) Load local data 
v_mov_b32 v64, v19 // (op 17) 
s_barrier  // (op 4991) 
s_lshl_b32 s1, s36, 3 // (op 17) 



ds_read_u8 v65, v5 offset:8192 // (op 87) Load local data 
v_add_u32 v76, v64, s1 // (op 17) 





ds_read_b128 v[78:81], v4 offset:2048 // (op 375) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:71], v[72:75], a[0:3], v63, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 90) 



ds_read_u8 v77, v5 offset:8320 // (op 379) Load local data 
buffer_load_ubyte v22, v76, s[68:71], 0 offen offset: 0 // (op 17) Load value




s_barrier  // (op 5014) 
ds_read_b128 v[82:85], v4 offset:4096 // (op 551) Load local data 






ds_read_u8 v76, v5 offset:8448 // (op 555) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:71], v[78:81], a[4:7], v63, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 380) 



ds_read_b128 v[86:89], v4 offset:6144 // (op 727) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:71], v[82:85], a[8:11], v63, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 556) 



ds_read_u8 v90, v5 offset:8576 // (op 731) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v39, v[44:47] offset:17152 // (op 257) Store local data 



ds_read_b96_tr_b6 v[92:94], v0 offset:8736 // (op 1084) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:71], v[86:89], a[12:15], v63, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 732) 
ds_read_b96_tr_b6 v[95:97], v0 offset:9760 // (op 1084) Transpose load local data 




ds_read_u8 v63, v3 offset:16928 // (op 1087) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v39, v[48:51] offset:17216 // (op 257) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:97], v[72:75], a[16:19], v63, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1227) 


s_barrier  // (op 5026) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:97], v[78:81], a[20:23], v63, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1181) 
s_waitcnt vmcnt(11)
ds_write_b8 v61, v52 offset:25344 // (op 259) Store local data 


s_waitcnt vmcnt(10)
ds_write_b8 v61, v53 offset:25348 // (op 259) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:97], v[82:85], a[24:27], v63, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1135) 


s_barrier  // (op 5008) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:97], v[86:89], a[28:31], v63, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1089) 
s_waitcnt vmcnt(9)
ds_write_b96 v18, v[54:56] offset:25856 // (op 247) Store local data 




s_waitcnt vmcnt(8)
ds_write_b96 v18, v[58:60] offset:25872 // (op 247) Store local data 
s_barrier  // (op 5020) 



s_waitcnt vmcnt(7)
ds_write_b8 v25, v16 offset:34048 // (op 258) Store local data 

s_lshl_b64 s[18:19], s[32:33], 2 // (op 260) 
v_lshrrev_b64 v[66:67], 3, s[18:19] // (op 260) 


v_mov_b32 v63, v23 // (op 260) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 5005) 
buffer_load_dwordx4 v[44:47], v63, s[64:67], 0 offen offset:192 // (op 260) Load value



v_add_u32 v63, v63, v66 // (op 260) 
ds_read_b96_tr_b6 v[68:70], v6 offset:25856 // (op 249) Transpose load local data 
ds_read_b96_tr_b6 v[71:73], v6 offset:26880 // (op 249) Transpose load local data 
buffer_load_dwordx4 v[48:51], v63, s[64:67], 0 offen offset:192 // (op 260) Load value



s_barrier  // (op 5011) 




v_add_u32 v15, 128, v15 // (op 4349) 
v_mov_b64 v[66:67], s[42:43] // (op 262) convert

v_mov_b32 v23, v24 // (op 262) 
ds_read_b128 v[74:77], v8 offset:17152 // (op 251) Load local data 
buffer_load_ubyte v52, v23, s[72:75], 0 offen offset:12 // (op 262) Load value
v_add_u32 v23, v23, v66 // (op 262) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5017) 
buffer_load_ubyte v53, v23, s[72:75], 0 offen offset:12 // (op 262) Load value





ds_read_u8 v63, v7 offset:34048 // (op 253) Load local data 


s_barrier  // (op 5023) 
v_add_u32 v21, 8, v21 // (op 4477) 



ds_read_u8 v23, v9 offset:25344 // (op 255) Load local data 
s_lshl_b64 s[18:19], s[26:27], 7 // (op 246) 



s_mul_i32 s51, s19, 6 // (op 246) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 246) least significant
s_mul_hi_u32 s1, s18, 6 // (op 246) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 246) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 246) 

s_mul_i32 s1, 3, s18 // (op 246) 

v_add_u32 v24, v62, s1 // (op 246) 


buffer_load_dwordx3 v[54:56], v24, s[52:55], 0 offen // (op 246) Load value
ds_read_b128 v[78:81], v8 offset:19200 // (op 385) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:73], v[74:77], a[0:3], v63, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 256) 



ds_read_u8 v65, v9 offset:25472 // (op 389) Load local data 
buffer_load_dwordx3 v[58:60], v24, s[52:55], 0 offen offset:12 // (op 246) Load value





ds_read_b128 v[82:85], v8 offset:21248 // (op 561) Load local data 





ds_read_u8 v24, v9 offset:25600 // (op 565) Load local data 
s_lshl_b32 s1, s36, 2 // (op 261) 



s_mul_i32 s18, 3, s1 // (op 261) 

v_add_u32 v62, v64, s18 // (op 261) 


s_lshl_b64 s[18:19], s[26:27], 7 // (op 4285) 
s_mul_i32 s51, s19, 6 // (op 4285) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 4285) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4285) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4285) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 4285) 

s_lshl_b32 s1, s18, 1 // (op 4285) 

v_add_u32 v17, v17, s1 // (op 4285) 

ds_read_b128 v[86:89], v8 offset:23296 // (op 737) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:73], v[78:81], a[4:7], v63, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 381) 



s_barrier  // (op 5087) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:73], v[82:85], a[8:11], v63, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 557) 
ds_read_u8 v66, v9 offset:25728 // (op 741) Load local data 
buffer_load_ubyte v16, v62, s[68:71], 0 offen offset: 0 // (op 261) Load value





s_barrier  // (op 5091) 
s_barrier  // (op 4966) 
s_lshl_b32 s1, s36, 3 // (op 4413) 



v_add_u32 v19, v19, s1 // (op 4413) 

ds_read_b96_tr_b6 v[90:92], v6 offset:25888 // (op 1092) Transpose load local data 
ds_read_b96_tr_b6 v[93:95], v6 offset:26912 // (op 1092) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:73], v[86:89], a[12:15], v63, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 733) 




s_barrier  // (op 5115) 
ds_read_u8 v62, v7 offset:34080 // (op 1095) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v20, v[26:29] // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v20, v[30:33] offset:64 // (op 13) Store local data 
s_barrier  // (op 5119) 
s_barrier  // (op 4994) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:95], v[74:77], a[16:19], v62, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1228) 





s_waitcnt vmcnt(11)
ds_write_b8 v57, v34 offset:8192 // (op 30) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:95], v[78:81], a[20:23], v62, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1182) 
s_waitcnt vmcnt(10)
ds_write_b8 v57, v35 offset:8196 // (op 30) Store local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[82:85], a[24:27], v62, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1136) 
s_barrier  // (op 4953) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[86:89], a[28:31], v62, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1090) 
s_waitcnt vmcnt(9)
ds_write_b96 v14, v[36:38] offset:8704 // (op 6) Store local data 




s_waitcnt vmcnt(8)
ds_write_b96 v14, v[40:42] offset:8720 // (op 6) Store local data 
s_barrier  // (op 4980) 



s_waitcnt vmcnt(7)
ds_write_b8 v43, v22 offset:16896 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 1050) 
s_mov_b32 s1, s18 // (op 1050) convert

s_and_b32 s18, s1, 4294967294 // (op 1050) 

s_cmp_lt_u32 s0, s18 // (op 1050) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoop_1050 // (op 1050) Condition: Bottom (jump to RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoop_1050 if true)
RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoop_1050:

















s_mul_hi_i32 s0, s16, s10 // (op 4241) 
v_and_b32 v57, -2, v1 // (op 4241) 
s_add_i32 s1, s0, s16 // (op 4241) 

v_sub_u32 v14, v1, v57 // (op 4241) 

s_ashr_i32 s0, s1, 31 // (op 4241) 
v_lshlrev_b32 v15, 5, v14 // (op 4241) 

s_cmp_eq_i32 s10, 0 // (op 4241) 
s_mov_b32 s18, scc // (op 4241) 
v_lshrrev_b32 v14, 1, v1 // (op 4241) 
s_cmp_eq_i32 s18, 0 // (op 4241) 
s_cselect_b32 s19, 0, -1 // (op 4241) 

s_and_b32 s18, s11, 31 // (op 4241) 
s_lshl_b32 s22, 1, s18 // (op 4241) 
s_add_i32 s23, s22, s19 // (op 4241) 


s_and_b32 s19, s0, s23 // (op 4241) 


s_add_i32 s0, s1, s19 // (op 4241) 


s_ashr_i32 s1, s0, s18 // (op 4241) 


s_lshl_b32 s0, s11, 24 // (op 4241) 
s_ashr_i32 s18, s0, 31 // (op 4241) 

s_xor_b32 s0, s1, s18 // (op 4241) 

s_sub_i32 s1, s0, s18 // (op 4241) 


s_mul_hi_i32 s0, s1, s12 // (op 4241) 
s_add_i32 s18, s0, s1 // (op 4241) 

s_ashr_i32 s0, s18, 31 // (op 4241) 
s_cmp_eq_i32 s12, 0 // (op 4241) 
s_mov_b32 s19, scc // (op 4241) 
s_cmp_eq_i32 s19, 0 // (op 4241) 
s_cselect_b32 s22, 0, -1 // (op 4241) 

s_and_b32 s19, s13, 31 // (op 4241) 
s_lshl_b32 s23, 1, s19 // (op 4241) 
s_add_i32 s24, s23, s22 // (op 4241) 


s_and_b32 s25, s0, s24 // (op 4241) 


s_add_i32 s0, s18, s25 // (op 4241) 


s_ashr_i32 s18, s0, s19 // (op 4241) 


s_lshl_b32 s0, s13, 24 // (op 4241) 
s_ashr_i32 s19, s0, 31 // (op 4241) 

s_xor_b32 s0, s18, s19 // (op 4241) 

s_sub_i32 s18, s0, s19 // (op 4241) 


s_mul_i32 s0, s1, s15 // (op 4241) 

s_sub_u32 s1, s16, s0 // (op 4241) 

s_mul_i32 s0, s18, s15 // (op 4241) 
s_add_u32 s0, s0, s1 // (op 4241) 


s_mul_hi_i32 s1, s16, s9 // (op 4241) 
s_add_i32 s18, s1, s16 // (op 4241) 

s_ashr_i32 s1, s18, 31 // (op 4241) 
s_cmp_eq_i32 s9, 0 // (op 4241) 
s_mov_b32 s19, scc // (op 4241) 
s_cmp_eq_i32 s19, 0 // (op 4241) 
s_cselect_b32 s22, 0, -1 // (op 4241) 

s_and_b32 s19, s46, 31 // (op 4241) 
s_lshl_b32 s23, 1, s19 // (op 4241) 
s_add_i32 s24, s23, s22 // (op 4241) 


s_and_b32 s25, s1, s24 // (op 4241) 


s_add_i32 s1, s18, s25 // (op 4241) 


s_ashr_i32 s18, s1, s19 // (op 4241) 


s_lshl_b32 s1, s46, 24 // (op 4241) 
s_ashr_i32 s19, s1, 31 // (op 4241) 

s_xor_b32 s1, s18, s19 // (op 4241) 

s_sub_i32 s18, s1, s19 // (op 4241) 


s_add_u32 s50, s6, 128 // (op 4241) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4241) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4241) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4241) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4241) 

s_mul_i32 s1, s15, s24 // (op 4241) 

s_mul_i32 s19, s8, s1 // (op 4241) 

s_mul_i32 s1, s18, s19 // (op 4241) 

s_sub_u32 s19, s16, s1 // (op 4241) 

s_mov_b32 s1, s19 // (op 4241) 
s_mul_hi_i32 s22, s1, s14 // (op 4241) 
s_add_i32 s23, s22, s1 // (op 4241) 


s_ashr_i32 s1, s23, 31 // (op 4241) 
s_cmp_eq_i32 s14, 0 // (op 4241) 
s_mov_b32 s22, scc // (op 4241) 
s_cmp_eq_i32 s22, 0 // (op 4241) 
s_cselect_b32 s24, 0, -1 // (op 4241) 

s_and_b32 s22, s48, 31 // (op 4241) 
s_lshl_b32 s25, 1, s22 // (op 4241) 
s_add_i32 s49, s25, s24 // (op 4241) 


s_and_b32 s24, s1, s49 // (op 4241) 


s_add_i32 s1, s23, s24 // (op 4241) 


s_ashr_i32 s23, s1, s22 // (op 4241) 


s_lshl_b32 s1, s48, 24 // (op 4241) 
s_ashr_i32 s22, s1, 31 // (op 4241) 

s_xor_b32 s1, s23, s22 // (op 4241) 

s_sub_i32 s23, s1, s22 // (op 4241) 


s_mul_i32 s1, s47, s15 // (op 4241) 
s_add_u32 s24, s20, 64 // (op 4241) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4241) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 4241) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 4241) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 4241) 

s_sub_i32 s22, s24, s1 // (op 4241) 

s_mul_i32 s24, s23, s22 // (op 4241) 


s_sub_u32 s25, s19, s24 // (op 4241) 


s_add_u32 s19, s25, s1 // (op 4241) 


s_cmp_eq_i32 s18, 0 // (op 4241) 
s_mov_b32 s1, scc // (op 4241) 

s_cmp_eq_i32 s1, 0 // (op 4241) 
s_cselect_b32 s18, s19, s0 // (op 4241) 



v_lshl_add_u32 v57, s18, 6, v15 // (op 4241) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 4241) 
v_mov_b32 v43, 0 // (op 4241) convert
v_mov_b32 v42, v57 // (op 4241) convert

s_lshr_b32 s18, s0, 1 // (op 4241) 

v_lshl_add_u32 v15, s18, 8, v14 // (op 4241) 


v_mul_lo_u32 v41, v15, s27 // (op 4241) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 4241) most significant: high of low * low
v_mul_lo_u32 v40, v15, s26 // (op 4241) least significant: low of low * low
v_add_u32 v41, v41, v14 // (op 4241) most significant: sum


v_add_co_u32 v14, s[0:1], v42, v40 // (op 4241) least significant half
v_addc_co_u32 v15, s[0:1], v43, v41, s[0:1] // (op 4241) most significant half



v_mul_lo_u32 v43, v15, 6 // (op 4241) most significant: low of high * low
v_mul_hi_u32 v57, v14, 6 // (op 4241) most significant: high of low * low
v_mul_lo_u32 v42, v14, 6 // (op 4241) least significant: low of low * low
v_add_u32 v43, v43, v57 // (op 4241) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[42:43] // (op 4241) 

v_mov_b32 v57, v14 // (op 4241) convert

s_mul_i32 s1, s31, 6 // (op 4243) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 4243) least significant
s_mul_hi_u32 s18, s30, 6 // (op 4243) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4243) most significant: sum

s_mul_i32 s19, s26, s1 // (op 4243) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 4243) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 4243) least significant
s_mul_hi_u32 s23, s26, s0 // (op 4243) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 4243) most significant: sum
s_add_u32 s19, s19, s23 // (op 4243) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4243) 

s_mov_b32 s52, s28 // (op 4243) reg expression
s_mov_b32 s53, s29 // (op 4243) reg expression
s_mov_b32 s54, s0 // (op 4243) reg expression
s_mov_b32 s55, 131072 // (op 4243) call()

s_mul_hi_i32 s0, s16, s10 // (op 4315) 
v_lshrrev_b32 v14, 2, v1 // (op 4315) 
s_add_i32 s1, s0, s16 // (op 4315) 

v_lshlrev_b32 v15, 1, v14 // (op 4315) 

s_ashr_i32 s0, s1, 31 // (op 4315) 
v_and_b32 v14, -4, v1 // (op 4315) 
s_cmp_eq_i32 s10, 0 // (op 4315) 
s_mov_b32 s18, scc // (op 4315) 
v_sub_u32 v17, v1, v14 // (op 4315) 

s_cmp_eq_i32 s18, 0 // (op 4315) 
s_cselect_b32 s19, 0, -1 // (op 4315) 

v_lshlrev_b32 v14, 5, v17 // (op 4315) 

s_and_b32 s18, s11, 31 // (op 4315) 
s_lshl_b32 s22, 1, s18 // (op 4315) 
s_add_i32 s23, s22, s19 // (op 4315) 


s_and_b32 s19, s0, s23 // (op 4315) 


s_add_i32 s0, s1, s19 // (op 4315) 


s_ashr_i32 s1, s0, s18 // (op 4315) 


s_lshl_b32 s0, s11, 24 // (op 4315) 
s_ashr_i32 s18, s0, 31 // (op 4315) 

s_xor_b32 s0, s1, s18 // (op 4315) 

s_sub_i32 s1, s0, s18 // (op 4315) 


s_mul_hi_i32 s0, s1, s12 // (op 4315) 
s_add_i32 s18, s0, s1 // (op 4315) 

s_ashr_i32 s0, s18, 31 // (op 4315) 
s_cmp_eq_i32 s12, 0 // (op 4315) 
s_mov_b32 s19, scc // (op 4315) 
s_cmp_eq_i32 s19, 0 // (op 4315) 
s_cselect_b32 s22, 0, -1 // (op 4315) 

s_and_b32 s19, s13, 31 // (op 4315) 
s_lshl_b32 s23, 1, s19 // (op 4315) 
s_add_i32 s24, s23, s22 // (op 4315) 


s_and_b32 s25, s0, s24 // (op 4315) 


s_add_i32 s0, s18, s25 // (op 4315) 


s_ashr_i32 s18, s0, s19 // (op 4315) 


s_lshl_b32 s0, s13, 24 // (op 4315) 
s_ashr_i32 s19, s0, 31 // (op 4315) 

s_xor_b32 s0, s18, s19 // (op 4315) 

s_sub_i32 s18, s0, s19 // (op 4315) 


s_add_u32 s50, s6, 128 // (op 4315) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4315) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4315) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4315) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4315) 

s_mul_i32 s0, s18, s24 // (op 4315) 

s_sub_i32 s18, s1, s0 // (op 4315) 


s_mul_hi_i32 s19, s16, s9 // (op 4315) 
s_add_i32 s0, s19, s16 // (op 4315) 

s_ashr_i32 s1, s0, 31 // (op 4315) 
s_cmp_eq_i32 s9, 0 // (op 4315) 
s_mov_b32 s19, scc // (op 4315) 
s_cmp_eq_i32 s19, 0 // (op 4315) 
s_cselect_b32 s22, 0, -1 // (op 4315) 

s_and_b32 s19, s46, 31 // (op 4315) 
s_lshl_b32 s23, 1, s19 // (op 4315) 
s_add_i32 s49, s23, s22 // (op 4315) 


s_and_b32 s22, s1, s49 // (op 4315) 


s_add_i32 s1, s0, s22 // (op 4315) 


s_ashr_i32 s0, s1, s19 // (op 4315) 


s_lshl_b32 s1, s46, 24 // (op 4315) 
s_ashr_i32 s19, s1, 31 // (op 4315) 

s_xor_b32 s1, s0, s19 // (op 4315) 

s_sub_i32 s0, s1, s19 // (op 4315) 


s_mul_i32 s1, s15, s24 // (op 4315) 

s_mul_i32 s19, s8, s1 // (op 4315) 

s_mul_i32 s1, s0, s19 // (op 4315) 

s_sub_u32 s19, s16, s1 // (op 4315) 

s_mul_hi_i32 s1, s19, s14 // (op 4315) 
s_add_i32 s22, s1, s19 // (op 4315) 


s_ashr_i32 s1, s22, 31 // (op 4315) 
s_cmp_eq_i32 s14, 0 // (op 4315) 
s_mov_b32 s19, scc // (op 4315) 
s_cmp_eq_i32 s19, 0 // (op 4315) 
s_cselect_b32 s23, 0, -1 // (op 4315) 

s_and_b32 s19, s48, 31 // (op 4315) 
s_lshl_b32 s24, 1, s19 // (op 4315) 
s_add_i32 s25, s24, s23 // (op 4315) 


s_and_b32 s23, s1, s25 // (op 4315) 


s_add_i32 s1, s22, s23 // (op 4315) 


s_ashr_i32 s22, s1, s19 // (op 4315) 


s_lshl_b32 s1, s48, 24 // (op 4315) 
s_ashr_i32 s19, s1, 31 // (op 4315) 

s_xor_b32 s1, s22, s19 // (op 4315) 

s_sub_i32 s22, s1, s19 // (op 4315) 


s_cmp_eq_i32 s0, 0 // (op 4315) 
s_mov_b32 s1, scc // (op 4315) 

s_cmp_eq_i32 s1, 0 // (op 4315) 
s_cselect_b32 s0, s22, s18 // (op 4315) 



v_lshl_add_u32 v17, s0, 7, v15 // (op 4315) 


v_mul_lo_u32 v19, v17, s33 // (op 4315) most significant: low of low * high
v_mul_hi_u32 v15, v17, s32 // (op 4315) most significant: high of low * low
v_mul_lo_u32 v18, v17, s32 // (op 4315) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 4315) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 4315) 
s_lshr_b32 s18, s0, 1 // (op 4315) 

v_lshl_add_u32 v15, s18, 8, v14 // (op 4315) 


v_mov_b32 v21, 0 // (op 4315) convert
v_mov_b32 v20, v15 // (op 4315) convert

v_add_co_u32 v14, s[0:1], v20, v18 // (op 4315) least significant half
v_addc_co_u32 v15, s[0:1], v21, v19, s[0:1] // (op 4315) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 4315) 


v_lshrrev_b64 v[42:43], 3, v[14:15] // (op 4315) 

v_mov_b32 v14, v42 // (op 4315) convert

s_mul_i32 s1, s32, s7 // (op 4317) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 4317) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 4317) least significant
s_mul_hi_u32 s19, s32, s6 // (op 4317) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4317) most significant: sum
s_add_u32 s1, s1, s19 // (op 4317) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4317) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 4317) 

s_mov_b32 s64, s34 // (op 4317) reg expression
s_mov_b32 s65, s35 // (op 4317) reg expression
s_mov_b32 s66, s0 // (op 4317) reg expression
s_mov_b32 s67, 131072 // (op 4317) call()

s_mul_hi_i32 s0, s16, s10 // (op 4379) 
v_and_b32 v15, -64, v1 // (op 4379) 
s_add_i32 s1, s0, s16 // (op 4379) 

v_sub_u32 v17, v1, v15 // (op 4379) 

s_ashr_i32 s0, s1, 31 // (op 4379) 
v_lshrrev_b32 v15, 6, v1 // (op 4379) 
s_cmp_eq_i32 s10, 0 // (op 4379) 
s_mov_b32 s18, scc // (op 4379) 
s_cmp_eq_i32 s18, 0 // (op 4379) 
s_cselect_b32 s19, 0, -1 // (op 4379) 

s_and_b32 s18, s11, 31 // (op 4379) 
s_lshl_b32 s22, 1, s18 // (op 4379) 
s_add_i32 s23, s22, s19 // (op 4379) 


s_and_b32 s19, s0, s23 // (op 4379) 


s_add_i32 s0, s1, s19 // (op 4379) 


s_ashr_i32 s1, s0, s18 // (op 4379) 


s_lshl_b32 s0, s11, 24 // (op 4379) 
s_ashr_i32 s18, s0, 31 // (op 4379) 

s_xor_b32 s0, s1, s18 // (op 4379) 

s_sub_i32 s1, s0, s18 // (op 4379) 


s_mul_hi_i32 s0, s1, s12 // (op 4379) 
s_add_i32 s18, s0, s1 // (op 4379) 

s_ashr_i32 s0, s18, 31 // (op 4379) 
s_cmp_eq_i32 s12, 0 // (op 4379) 
s_mov_b32 s19, scc // (op 4379) 
s_cmp_eq_i32 s19, 0 // (op 4379) 
s_cselect_b32 s22, 0, -1 // (op 4379) 

s_and_b32 s19, s13, 31 // (op 4379) 
s_lshl_b32 s23, 1, s19 // (op 4379) 
s_add_i32 s24, s23, s22 // (op 4379) 


s_and_b32 s25, s0, s24 // (op 4379) 


s_add_i32 s0, s18, s25 // (op 4379) 


s_ashr_i32 s18, s0, s19 // (op 4379) 


s_lshl_b32 s0, s13, 24 // (op 4379) 
s_ashr_i32 s19, s0, 31 // (op 4379) 

s_xor_b32 s0, s18, s19 // (op 4379) 

s_sub_i32 s18, s0, s19 // (op 4379) 


s_mul_i32 s0, s1, s15 // (op 4379) 

s_sub_u32 s1, s16, s0 // (op 4379) 

s_mul_i32 s0, s18, s15 // (op 4379) 
s_add_u32 s0, s0, s1 // (op 4379) 


s_mul_hi_i32 s1, s16, s9 // (op 4379) 
s_add_i32 s18, s1, s16 // (op 4379) 

s_ashr_i32 s1, s18, 31 // (op 4379) 
s_cmp_eq_i32 s9, 0 // (op 4379) 
s_mov_b32 s19, scc // (op 4379) 
s_cmp_eq_i32 s19, 0 // (op 4379) 
s_cselect_b32 s22, 0, -1 // (op 4379) 

s_and_b32 s19, s46, 31 // (op 4379) 
s_lshl_b32 s23, 1, s19 // (op 4379) 
s_add_i32 s24, s23, s22 // (op 4379) 


s_and_b32 s25, s1, s24 // (op 4379) 


s_add_i32 s1, s18, s25 // (op 4379) 


s_ashr_i32 s18, s1, s19 // (op 4379) 


s_lshl_b32 s1, s46, 24 // (op 4379) 
s_ashr_i32 s19, s1, 31 // (op 4379) 

s_xor_b32 s1, s18, s19 // (op 4379) 

s_sub_i32 s18, s1, s19 // (op 4379) 


s_add_u32 s50, s6, 128 // (op 4379) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4379) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4379) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4379) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4379) 

s_mul_i32 s1, s15, s24 // (op 4379) 

s_mul_i32 s19, s8, s1 // (op 4379) 

s_mul_i32 s1, s18, s19 // (op 4379) 

s_sub_u32 s19, s16, s1 // (op 4379) 

s_mov_b32 s1, s19 // (op 4379) 
s_mul_hi_i32 s22, s1, s14 // (op 4379) 
s_add_i32 s23, s22, s1 // (op 4379) 


s_ashr_i32 s1, s23, 31 // (op 4379) 
s_cmp_eq_i32 s14, 0 // (op 4379) 
s_mov_b32 s22, scc // (op 4379) 
s_cmp_eq_i32 s22, 0 // (op 4379) 
s_cselect_b32 s24, 0, -1 // (op 4379) 

s_and_b32 s22, s48, 31 // (op 4379) 
s_lshl_b32 s25, 1, s22 // (op 4379) 
s_add_i32 s49, s25, s24 // (op 4379) 


s_and_b32 s24, s1, s49 // (op 4379) 


s_add_i32 s1, s23, s24 // (op 4379) 


s_ashr_i32 s23, s1, s22 // (op 4379) 


s_lshl_b32 s1, s48, 24 // (op 4379) 
s_ashr_i32 s22, s1, 31 // (op 4379) 

s_xor_b32 s1, s23, s22 // (op 4379) 

s_sub_i32 s23, s1, s22 // (op 4379) 


s_mul_i32 s1, s47, s15 // (op 4379) 
s_add_u32 s24, s20, 64 // (op 4379) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4379) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 4379) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 4379) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 4379) 

s_sub_i32 s22, s24, s1 // (op 4379) 

s_mul_i32 s24, s23, s22 // (op 4379) 


s_sub_u32 s25, s19, s24 // (op 4379) 


s_add_u32 s19, s25, s1 // (op 4379) 


s_cmp_eq_i32 s18, 0 // (op 4379) 
s_mov_b32 s1, scc // (op 4379) 

s_cmp_eq_i32 s1, 0 // (op 4379) 
s_cselect_b32 s18, s19, s0 // (op 4379) 



v_lshl_add_u32 v18, s18, 6, v17 // (op 4379) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 4379) 
s_lshr_b32 s18, s0, 1 // (op 4379) 

v_lshl_add_u32 v17, s18, 3, v15 // (op 4379) 


v_mul_lo_u32 v15, v17, s36 // (op 4379) 

v_add_u32 v17, v18, v15 // (op 4379) 


s_mul_i32 s1, s36, s41 // (op 4381) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 4381) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 4381) least significant
s_mul_hi_u32 s19, s36, s40 // (op 4381) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4381) most significant: sum
s_add_u32 s1, s1, s19 // (op 4381) most significant: sum


s_mov_b32 s68, s38 // (op 4381) reg expression
s_mov_b32 s69, s39 // (op 4381) reg expression
s_mov_b32 s70, s0 // (op 4381) reg expression
s_mov_b32 s71, 131072 // (op 4381) call()

s_mul_hi_i32 s0, s16, s10 // (op 4443) 
v_lshrrev_b32 v15, 2, v1 // (op 4443) 
s_add_i32 s1, s0, s16 // (op 4443) 

v_lshlrev_b32 v18, 1, v15 // (op 4443) 

s_ashr_i32 s0, s1, 31 // (op 4443) 
v_and_b32 v15, -4, v1 // (op 4443) 
s_cmp_eq_i32 s10, 0 // (op 4443) 
s_mov_b32 s18, scc // (op 4443) 
v_sub_u32 v19, v1, v15 // (op 4443) 

s_cmp_eq_i32 s18, 0 // (op 4443) 
s_cselect_b32 s19, 0, -1 // (op 4443) 

s_and_b32 s18, s11, 31 // (op 4443) 
s_lshl_b32 s22, 1, s18 // (op 4443) 
s_add_i32 s23, s22, s19 // (op 4443) 


s_and_b32 s19, s0, s23 // (op 4443) 


s_add_i32 s0, s1, s19 // (op 4443) 


s_ashr_i32 s1, s0, s18 // (op 4443) 


s_lshl_b32 s0, s11, 24 // (op 4443) 
s_ashr_i32 s18, s0, 31 // (op 4443) 

s_xor_b32 s0, s1, s18 // (op 4443) 

s_sub_i32 s1, s0, s18 // (op 4443) 


s_mul_hi_i32 s0, s1, s12 // (op 4443) 
s_add_i32 s18, s0, s1 // (op 4443) 

s_ashr_i32 s0, s18, 31 // (op 4443) 
s_cmp_eq_i32 s12, 0 // (op 4443) 
s_mov_b32 s19, scc // (op 4443) 
s_cmp_eq_i32 s19, 0 // (op 4443) 
s_cselect_b32 s22, 0, -1 // (op 4443) 

s_and_b32 s19, s13, 31 // (op 4443) 
s_lshl_b32 s23, 1, s19 // (op 4443) 
s_add_i32 s24, s23, s22 // (op 4443) 


s_and_b32 s25, s0, s24 // (op 4443) 


s_add_i32 s0, s18, s25 // (op 4443) 


s_ashr_i32 s18, s0, s19 // (op 4443) 


s_lshl_b32 s0, s13, 24 // (op 4443) 
s_ashr_i32 s19, s0, 31 // (op 4443) 

s_xor_b32 s0, s18, s19 // (op 4443) 

s_sub_i32 s18, s0, s19 // (op 4443) 


s_add_u32 s50, s6, 128 // (op 4443) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4443) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 4443) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 4443) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 7 // (op 4443) 

s_mul_i32 s0, s18, s24 // (op 4443) 

s_sub_i32 s18, s1, s0 // (op 4443) 


s_mul_hi_i32 s19, s16, s9 // (op 4443) 
s_add_i32 s0, s19, s16 // (op 4443) 

s_ashr_i32 s1, s0, 31 // (op 4443) 
s_cmp_eq_i32 s9, 0 // (op 4443) 
s_mov_b32 s19, scc // (op 4443) 
s_cmp_eq_i32 s19, 0 // (op 4443) 
s_cselect_b32 s22, 0, -1 // (op 4443) 

s_and_b32 s19, s46, 31 // (op 4443) 
s_lshl_b32 s23, 1, s19 // (op 4443) 
s_add_i32 s49, s23, s22 // (op 4443) 


s_and_b32 s22, s1, s49 // (op 4443) 


s_add_i32 s1, s0, s22 // (op 4443) 


s_ashr_i32 s0, s1, s19 // (op 4443) 


s_lshl_b32 s1, s46, 24 // (op 4443) 
s_ashr_i32 s19, s1, 31 // (op 4443) 

s_xor_b32 s1, s0, s19 // (op 4443) 

s_sub_i32 s0, s1, s19 // (op 4443) 


s_mul_i32 s1, s15, s24 // (op 4443) 

s_mul_i32 s19, s8, s1 // (op 4443) 

s_mul_i32 s1, s0, s19 // (op 4443) 

s_sub_u32 s19, s16, s1 // (op 4443) 

s_mul_hi_i32 s1, s19, s14 // (op 4443) 
s_add_i32 s22, s1, s19 // (op 4443) 


s_ashr_i32 s1, s22, 31 // (op 4443) 
s_cmp_eq_i32 s14, 0 // (op 4443) 
s_mov_b32 s19, scc // (op 4443) 
s_cmp_eq_i32 s19, 0 // (op 4443) 
s_cselect_b32 s23, 0, -1 // (op 4443) 

s_and_b32 s19, s48, 31 // (op 4443) 
s_lshl_b32 s24, 1, s19 // (op 4443) 
s_add_i32 s25, s24, s23 // (op 4443) 


s_and_b32 s23, s1, s25 // (op 4443) 


s_add_i32 s1, s22, s23 // (op 4443) 


s_ashr_i32 s22, s1, s19 // (op 4443) 


s_lshl_b32 s1, s48, 24 // (op 4443) 
s_ashr_i32 s19, s1, 31 // (op 4443) 

s_xor_b32 s1, s22, s19 // (op 4443) 

s_sub_i32 s22, s1, s19 // (op 4443) 


s_cmp_eq_i32 s0, 0 // (op 4443) 
s_mov_b32 s1, scc // (op 4443) 

s_cmp_eq_i32 s1, 0 // (op 4443) 
s_cselect_b32 s0, s22, s18 // (op 4443) 



v_lshl_add_u32 v15, s0, 7, v18 // (op 4443) 


v_mul_lo_u32 v18, v15, s42 // (op 4443) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 4443) 
s_lshr_b32 s18, s0, 1 // (op 4443) 

v_lshl_add_u32 v15, s18, 3, v19 // (op 4443) 


v_add_u32 v19, v15, v18 // (op 4443) 


s_mul_i32 s1, s42, s7 // (op 4445) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 4445) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 4445) least significant
s_mul_hi_u32 s19, s42, s6 // (op 4445) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4445) most significant: sum
s_add_u32 s1, s1, s19 // (op 4445) most significant: sum


s_mov_b32 s72, s44 // (op 4445) reg expression
s_mov_b32 s73, s45 // (op 4445) reg expression
s_mov_b32 s74, s0 // (op 4445) reg expression
s_mov_b32 s75, 131072 // (op 4445) call()

v_and_b32 v15, -2, v1 // (op 4526) 
v_sub_u32 v18, v1, v15 // (op 4526) 

v_lshlrev_b32 v15, 5, v18 // (op 4526) 

v_lshrrev_b32 v18, 1, v1 // (op 4526) 
v_lshl_add_u32 v20, v18, 6, v15 // (op 4526) 


v_mul_lo_u32 v15, v20, 6 // (op 4526) 
v_lshrrev_b32 v18, 3, v15 // (op 4526) 

v_lshrrev_b32 v15, 4, v20 // (op 4526) 

v_lshlrev_b32 v20, 2, v15 // (op 4526) 

v_add_u32 v15, v18, v20 // (op 4526) 


v_and_b32 v18, -4, v1 // (op 4586) 
v_sub_u32 v20, v1, v18 // (op 4586) 

v_lshlrev_b32 v18, 5, v20 // (op 4586) 

v_lshrrev_b32 v20, 2, v1 // (op 4586) 
v_lshl_add_u32 v21, v20, 8, v18 // (op 4586) 


v_lshlrev_b32 v18, 2, v21 // (op 4586) 

v_lshrrev_b32 v20, 3, v18 // (op 4586) 

v_and_b32 v18, -64, v1 // (op 4644) 
v_sub_u32 v21, v1, v18 // (op 4644) 

v_and_b32 v18, 4294967232, v1 // (op 4644) 
v_add_u32 v22, v18, v21 // (op 4644) 


v_and_b32 v18, -4, v1 // (op 4702) 
v_sub_u32 v21, v1, v18 // (op 4702) 

v_lshrrev_b32 v18, 2, v1 // (op 4702) 
v_lshl_add_u32 v23, v18, 3, v21 // (op 4702) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1025) 
s_and_b32 s18, s0, 4294967294 // (op 1025) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 1024) 
s_mov_b32 s19, s0 // (op 1024) convert

s_cmp_lt_u32 s18, s19 // (op 1024) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_1024 // (op 1024) Condition: Top (jump to RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_1024 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoopTail_1024:


s_add_u32 s18, s18, 1 // (op 1026) 
v_mov_b32 v16, v57 // (op 180) 
v_mov_b32 v18, v16 // (op 180) reg expression

buffer_load_dwordx3 v[24:26], v18, s[52:55], 0 offen // (op 180) Load value
buffer_load_dwordx3 v[28:30], v18, s[52:55], 0 offen offset:12 // (op 180) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 4231) 
s_mul_i32 s51, s1, 6 // (op 4231) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 4231) least significant
s_mul_hi_u32 s19, s0, 6 // (op 4231) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 4231) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 4231) 

v_add_u32 v57, v57, s0 // (op 4231) 


s_barrier  // (op 4956) 
s_lshl_b64 s[0:1], s[32:33], 2 // (op 194) 



v_lshrrev_b64 v[32:33], 3, s[0:1] // (op 194) 


v_mov_b32 v16, v14 // (op 194) 
v_mov_b32 v18, v16 // (op 194) 
buffer_load_dwordx4 v[34:37], v18, s[64:67], 0 offen // (op 194) Load value
v_add_u32 v18, v18, v32 // (op 194) 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[24:26] offset:8704 // (op 181) Store local data 
buffer_load_dwordx4 v[38:41], v18, s[64:67], 0 offen // (op 194) Load value



v_add_u32 v14, 64, v14 // (op 4307) 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[28:30] offset:8720 // (op 181) Store local data 



v_mov_b32 v16, v17 // (op 195) 
s_barrier  // (op 4969) 
v_mov_b32 v18, v16 // (op 195) reg expression




buffer_load_ubyte v21, v18, s[68:71], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s19, s36, 2 // (op 4371) 
v_add_u32 v17, v17, s19 // (op 4371) 

s_waitcnt vmcnt(2)
ds_write_b128 v20, v[34:37] // (op 191) Store local data 

v_mov_b64 v[24:25], s[42:43] // (op 196) convert

v_mov_b32 v16, v19 // (op 196) 
v_mov_b32 v18, v16 // (op 196) 
buffer_load_ubyte v26, v18, s[72:75], 0 offen offset: 0 // (op 196) Load value
v_add_u32 v18, v18, v24 // (op 196) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[38:41] offset:64 // (op 191) Store local data 
buffer_load_ubyte v27, v18, s[72:75], 0 offen offset: 0 // (op 196) Load value




v_add_u32 v19, 4, v19 // (op 4435) 
s_barrier  // (op 4983) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4959) 






ds_read_b96_tr_b6 v[28:30], v0 offset:8704 // (op 183) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v0 offset:9728 // (op 183) Transpose load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v22, v21 offset:16896 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 4973) 



s_barrier  // (op 4997) 
ds_read_b128 v[34:37], v4 // (op 185) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 4987) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v26 offset:8192 // (op 193) Store local data 


ds_read_u8 v16, v3 offset:16896 // (op 187) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v27 offset:8196 // (op 193) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 5001) 



ds_read_u8 v18, v5 offset:8192 // (op 189) Load local data 



ds_read_b128 v[24:27], v4 offset:2048 // (op 354) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:33], v[34:37], a[0:3], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 190) 



ds_read_u8 v21, v5 offset:8320 // (op 358) Load local data 



ds_read_b128 v[38:41], v4 offset:4096 // (op 530) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:33], v[24:27], a[4:7], v16, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 359) 



ds_read_u8 v42, v5 offset:8448 // (op 534) Load local data 



ds_read_b128 v[44:47], v4 offset:6144 // (op 706) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:33], v[38:41], a[8:11], v16, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 535) 



ds_read_u8 v43, v5 offset:8576 // (op 710) Load local data 



ds_read_b96_tr_b6 v[48:50], v0 offset:8736 // (op 968) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v0 offset:9760 // (op 968) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[28:33], v[44:47], a[12:15], v16, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 711) 




ds_read_u8 v16, v3 offset:16928 // (op 971) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:53], v[34:37], a[16:19], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 973) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:53], v[24:27], a[20:23], v16, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1004) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:53], v[38:41], a[24:27], v16, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1035) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[44:47], a[28:31], v16, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1066) 




s_ashr_i64 s[0:1], s[30:31], 7 // (op 1024) 
s_mov_b32 s19, s0 // (op 1024) convert

s_cmp_lt_u32 s18, s19 // (op 1024) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoopTail_1024 // (op 1024) Condition: Bottom (jump to RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoopTail_1024 if true)
RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_1024:























s_barrier  // (op 5099) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_lshrrev_b32 v6, 16, v14 // (op 56) 
v_lshrrev_b32 v7, 16, v15 // (op 56) 
v_lshrrev_b32 v8, 16, v16 // (op 56) 
v_lshrrev_b32 v9, 16, v17 // (op 56) 


v_pack_b32_f16 v4, v6, v7 // (op 58) 
v_pack_b32_f16 v5, v8, v9 // (op 58) 

v_mov_b32 v0, v13 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 1873) 
v_accvgpr_read v7, a5 // (op 1873) 
v_accvgpr_read v8, a6 // (op 1873) 
v_accvgpr_read v9, a7 // (op 1873) 
v_mul_f32 v14, v11, v6 // (op 1873) 
v_mul_f32 v15, v11, v7 // (op 1873) 
v_mul_f32 v16, v11, v8 // (op 1873) 
v_mul_f32 v17, v11, v9 // (op 1873) 

v_lshrrev_b32 v6, 16, v14 // (op 1875) 
v_lshrrev_b32 v7, 16, v15 // (op 1875) 
v_lshrrev_b32 v8, 16, v16 // (op 1875) 
v_lshrrev_b32 v9, 16, v17 // (op 1875) 


v_pack_b32_f16 v4, v6, v7 // (op 1876) 
v_pack_b32_f16 v5, v8, v9 // (op 1876) 

s_lshl_b32 s0, s4, 6 // (op 1876) 
v_add_u32 v0, v13, s0 // (op 1876) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1876) Store value



v_accvgpr_read v6, a8 // (op 1887) 
v_accvgpr_read v7, a9 // (op 1887) 
v_accvgpr_read v8, a10 // (op 1887) 
v_accvgpr_read v9, a11 // (op 1887) 
v_mul_f32 v14, v11, v6 // (op 1887) 
v_mul_f32 v15, v11, v7 // (op 1887) 
v_mul_f32 v16, v11, v8 // (op 1887) 
v_mul_f32 v17, v11, v9 // (op 1887) 

v_lshrrev_b32 v6, 16, v14 // (op 1889) 
v_lshrrev_b32 v7, 16, v15 // (op 1889) 
v_lshrrev_b32 v8, 16, v16 // (op 1889) 
v_lshrrev_b32 v9, 16, v17 // (op 1889) 


v_pack_b32_f16 v4, v6, v7 // (op 1890) 
v_pack_b32_f16 v5, v8, v9 // (op 1890) 

s_lshl_b32 s0, s4, 7 // (op 1890) 
v_add_u32 v0, v13, s0 // (op 1890) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1890) Store value



v_accvgpr_read v6, a12 // (op 1901) 
v_accvgpr_read v7, a13 // (op 1901) 
v_accvgpr_read v8, a14 // (op 1901) 
v_accvgpr_read v9, a15 // (op 1901) 
v_mul_f32 v14, v11, v6 // (op 1901) 
v_mul_f32 v15, v11, v7 // (op 1901) 
v_mul_f32 v16, v11, v8 // (op 1901) 
v_mul_f32 v17, v11, v9 // (op 1901) 

v_lshrrev_b32 v6, 16, v14 // (op 1903) 
v_lshrrev_b32 v7, 16, v15 // (op 1903) 
v_lshrrev_b32 v8, 16, v16 // (op 1903) 
v_lshrrev_b32 v9, 16, v17 // (op 1903) 


v_pack_b32_f16 v4, v6, v7 // (op 1904) 
v_pack_b32_f16 v5, v8, v9 // (op 1904) 

s_lshl_b32 s0, s4, 6 // (op 1904) 
s_mul_i32 s1, 3, s0 // (op 1904) 

v_add_u32 v0, v13, s1 // (op 1904) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1904) Store value



v_accvgpr_read v6, a16 // (op 1937) 
v_accvgpr_read v7, a17 // (op 1937) 
v_accvgpr_read v8, a18 // (op 1937) 
v_accvgpr_read v9, a19 // (op 1937) 
v_mul_f32 v14, v11, v6 // (op 1937) 
v_mul_f32 v15, v11, v7 // (op 1937) 
v_mul_f32 v16, v11, v8 // (op 1937) 
v_mul_f32 v17, v11, v9 // (op 1937) 

v_lshrrev_b32 v6, 16, v14 // (op 1939) 
v_lshrrev_b32 v7, 16, v15 // (op 1939) 
v_lshrrev_b32 v8, 16, v16 // (op 1939) 
v_lshrrev_b32 v9, 16, v17 // (op 1939) 


v_pack_b32_f16 v4, v6, v7 // (op 1941) 
v_pack_b32_f16 v5, v8, v9 // (op 1941) 

v_add_u32 v0, 64, v13 // (op 1941) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1941) Store value



v_accvgpr_read v6, a20 // (op 1951) 
v_accvgpr_read v7, a21 // (op 1951) 
v_accvgpr_read v8, a22 // (op 1951) 
v_accvgpr_read v9, a23 // (op 1951) 
v_mul_f32 v14, v11, v6 // (op 1951) 
v_mul_f32 v15, v11, v7 // (op 1951) 
v_mul_f32 v16, v11, v8 // (op 1951) 
v_mul_f32 v17, v11, v9 // (op 1951) 

v_lshrrev_b32 v6, 16, v14 // (op 1953) 
v_lshrrev_b32 v7, 16, v15 // (op 1953) 
v_lshrrev_b32 v8, 16, v16 // (op 1953) 
v_lshrrev_b32 v9, 16, v17 // (op 1953) 


v_pack_b32_f16 v4, v6, v7 // (op 1943) 
v_pack_b32_f16 v5, v8, v9 // (op 1943) 

s_lshl_b32 s0, s4, 6 // (op 1943) 
s_add_u32 s0, s0, 64 // (op 1943) 
v_add_u32 v0, v13, s0 // (op 1943) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1943) Store value



v_accvgpr_read v6, a24 // (op 1957) 
v_accvgpr_read v7, a25 // (op 1957) 
v_accvgpr_read v8, a26 // (op 1957) 
v_accvgpr_read v9, a27 // (op 1957) 
v_mul_f32 v14, v11, v6 // (op 1957) 
v_mul_f32 v15, v11, v7 // (op 1957) 
v_mul_f32 v16, v11, v8 // (op 1957) 
v_mul_f32 v17, v11, v9 // (op 1957) 

v_lshrrev_b32 v6, 16, v14 // (op 1959) 
v_lshrrev_b32 v7, 16, v15 // (op 1959) 
v_lshrrev_b32 v8, 16, v16 // (op 1959) 
v_lshrrev_b32 v9, 16, v17 // (op 1959) 


v_pack_b32_f16 v4, v6, v7 // (op 1945) 
v_pack_b32_f16 v5, v8, v9 // (op 1945) 

s_lshl_b32 s0, s4, 7 // (op 1945) 
s_add_u32 s0, s0, 64 // (op 1945) 
v_add_u32 v0, v13, s0 // (op 1945) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1945) Store value



v_accvgpr_read v6, a28 // (op 1963) 
v_accvgpr_read v7, a29 // (op 1963) 
v_accvgpr_read v8, a30 // (op 1963) 
v_accvgpr_read v9, a31 // (op 1963) 
v_mul_f32 v14, v11, v6 // (op 1963) 
v_mul_f32 v15, v11, v7 // (op 1963) 
v_mul_f32 v16, v11, v8 // (op 1963) 
v_mul_f32 v17, v11, v9 // (op 1963) 

v_lshrrev_b32 v6, 16, v14 // (op 1965) 
v_lshrrev_b32 v7, 16, v15 // (op 1965) 
v_lshrrev_b32 v8, 16, v16 // (op 1965) 
v_lshrrev_b32 v9, 16, v17 // (op 1965) 


v_pack_b32_f16 v4, v6, v7 // (op 1947) 
v_pack_b32_f16 v5, v8, v9 // (op 1947) 

s_lshl_b32 s0, s4, 6 // (op 1947) 
s_mul_i32 s1, 3, s0 // (op 1947) 
s_add_u32 s1, s1, 64 // (op 1947) 

v_add_u32 v0, v13, s1 // (op 1947) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1947) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalBottom_0_beta_135
RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalFalse_0_beta_135:


v_add_u32 v0, 64, v12 // (op 2133) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2133) Load value

v_accvgpr_read v4, a16 // (op 2120) 
v_accvgpr_read v5, a17 // (op 2120) 
v_accvgpr_read v6, a18 // (op 2120) 
v_accvgpr_read v7, a19 // (op 2120) 
v_mul_f32 v14, v11, v4 // (op 2120) 
v_mul_f32 v15, v11, v5 // (op 2120) 
v_mul_f32 v16, v11, v6 // (op 2120) 
v_mul_f32 v17, v11, v7 // (op 2120) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2132) 
v_lshrrev_b32 v7, 16, v8 // (op 2132) 
v_lshlrev_b32 v6, 16, v6 // (op 2132) 
v_lshlrev_b32 v7, 16, v7 // (op 2132) 
v_mul_f32 v18, v10, v6 // (op 2132) 
v_mul_f32 v19, v10, v7 // (op 2132) 
v_and_b32 v6, 65535, v9 // (op 2132) 
v_lshrrev_b32 v7, 16, v9 // (op 2132) 
v_lshlrev_b32 v6, 16, v6 // (op 2132) 
v_lshlrev_b32 v7, 16, v7 // (op 2132) 
v_mul_f32 v20, v10, v6 // (op 2132) 
v_mul_f32 v21, v10, v7 // (op 2132) 


v_add_f32 v6, v18, v14 // (op 2121) 
v_add_f32 v7, v19, v15 // (op 2121) 
v_add_f32 v8, v20, v16 // (op 2121) 
v_add_f32 v9, v21, v17 // (op 2121) 


v_lshrrev_b32 v14, 16, v6 // (op 2122) 
v_lshrrev_b32 v15, 16, v7 // (op 2122) 
v_lshrrev_b32 v16, 16, v8 // (op 2122) 
v_lshrrev_b32 v17, 16, v9 // (op 2122) 


v_pack_b32_f16 v8, v14, v15 // (op 2124) 
v_pack_b32_f16 v9, v16, v17 // (op 2124) 

v_add_u32 v0, 64, v13 // (op 2124) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 2124) Store value





s_lshl_b32 s0, s2, 6 // (op 2138) 
s_add_u32 s0, s0, 64 // (op 2138) 
v_add_u32 v0, v12, s0 // (op 2138) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2138) Load value

v_accvgpr_read v4, a20 // (op 2150) 
v_accvgpr_read v5, a21 // (op 2150) 
v_accvgpr_read v6, a22 // (op 2150) 
v_accvgpr_read v7, a23 // (op 2150) 
v_mul_f32 v14, v11, v4 // (op 2150) 
v_mul_f32 v15, v11, v5 // (op 2150) 
v_mul_f32 v16, v11, v6 // (op 2150) 
v_mul_f32 v17, v11, v7 // (op 2150) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2135) 
v_lshrrev_b32 v7, 16, v8 // (op 2135) 
v_lshlrev_b32 v6, 16, v6 // (op 2135) 
v_lshlrev_b32 v7, 16, v7 // (op 2135) 
v_mul_f32 v18, v10, v6 // (op 2135) 
v_mul_f32 v19, v10, v7 // (op 2135) 
v_and_b32 v6, 65535, v9 // (op 2135) 
v_lshrrev_b32 v7, 16, v9 // (op 2135) 
v_lshlrev_b32 v6, 16, v6 // (op 2135) 
v_lshlrev_b32 v7, 16, v7 // (op 2135) 
v_mul_f32 v20, v10, v6 // (op 2135) 
v_mul_f32 v21, v10, v7 // (op 2135) 


v_add_f32 v6, v18, v14 // (op 2136) 
v_add_f32 v7, v19, v15 // (op 2136) 
v_add_f32 v8, v20, v16 // (op 2136) 
v_add_f32 v9, v21, v17 // (op 2136) 


v_lshrrev_b32 v14, 16, v6 // (op 2137) 
v_lshrrev_b32 v15, 16, v7 // (op 2137) 
v_lshrrev_b32 v16, 16, v8 // (op 2137) 
v_lshrrev_b32 v17, 16, v9 // (op 2137) 


v_pack_b32_f16 v8, v14, v15 // (op 2126) 
v_pack_b32_f16 v9, v16, v17 // (op 2126) 

s_lshl_b32 s0, s4, 6 // (op 2126) 
s_add_u32 s0, s0, 64 // (op 2126) 
v_add_u32 v0, v13, s0 // (op 2126) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 2126) Store value





s_lshl_b32 s0, s2, 7 // (op 2143) 
s_add_u32 s0, s0, 64 // (op 2143) 
v_add_u32 v0, v12, s0 // (op 2143) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2143) Load value

v_accvgpr_read v4, a24 // (op 2152) 
v_accvgpr_read v5, a25 // (op 2152) 
v_accvgpr_read v6, a26 // (op 2152) 
v_accvgpr_read v7, a27 // (op 2152) 
v_mul_f32 v14, v11, v4 // (op 2152) 
v_mul_f32 v15, v11, v5 // (op 2152) 
v_mul_f32 v16, v11, v6 // (op 2152) 
v_mul_f32 v17, v11, v7 // (op 2152) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2140) 
v_lshrrev_b32 v7, 16, v8 // (op 2140) 
v_lshlrev_b32 v6, 16, v6 // (op 2140) 
v_lshlrev_b32 v7, 16, v7 // (op 2140) 
v_mul_f32 v18, v10, v6 // (op 2140) 
v_mul_f32 v19, v10, v7 // (op 2140) 
v_and_b32 v6, 65535, v9 // (op 2140) 
v_lshrrev_b32 v7, 16, v9 // (op 2140) 
v_lshlrev_b32 v6, 16, v6 // (op 2140) 
v_lshlrev_b32 v7, 16, v7 // (op 2140) 
v_mul_f32 v20, v10, v6 // (op 2140) 
v_mul_f32 v21, v10, v7 // (op 2140) 


v_add_f32 v6, v18, v14 // (op 2141) 
v_add_f32 v7, v19, v15 // (op 2141) 
v_add_f32 v8, v20, v16 // (op 2141) 
v_add_f32 v9, v21, v17 // (op 2141) 


v_lshrrev_b32 v14, 16, v6 // (op 2142) 
v_lshrrev_b32 v15, 16, v7 // (op 2142) 
v_lshrrev_b32 v16, 16, v8 // (op 2142) 
v_lshrrev_b32 v17, 16, v9 // (op 2142) 


v_pack_b32_f16 v8, v14, v15 // (op 2128) 
v_pack_b32_f16 v9, v16, v17 // (op 2128) 

s_lshl_b32 s0, s4, 7 // (op 2128) 
s_add_u32 s0, s0, 64 // (op 2128) 
v_add_u32 v0, v13, s0 // (op 2128) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 2128) Store value





s_lshl_b32 s0, s2, 6 // (op 2148) 
s_mul_i32 s1, 3, s0 // (op 2148) 
s_add_u32 s1, s1, 64 // (op 2148) 

v_add_u32 v0, v12, s1 // (op 2148) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2148) Load value

v_accvgpr_read v4, a28 // (op 2154) 
v_accvgpr_read v5, a29 // (op 2154) 
v_accvgpr_read v6, a30 // (op 2154) 
v_accvgpr_read v7, a31 // (op 2154) 
v_mul_f32 v14, v11, v4 // (op 2154) 
v_mul_f32 v15, v11, v5 // (op 2154) 
v_mul_f32 v16, v11, v6 // (op 2154) 
v_mul_f32 v17, v11, v7 // (op 2154) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2145) 
v_lshrrev_b32 v7, 16, v8 // (op 2145) 
v_lshlrev_b32 v6, 16, v6 // (op 2145) 
v_lshlrev_b32 v7, 16, v7 // (op 2145) 
v_mul_f32 v18, v10, v6 // (op 2145) 
v_mul_f32 v19, v10, v7 // (op 2145) 
v_and_b32 v6, 65535, v9 // (op 2145) 
v_lshrrev_b32 v7, 16, v9 // (op 2145) 
v_lshlrev_b32 v6, 16, v6 // (op 2145) 
v_lshlrev_b32 v7, 16, v7 // (op 2145) 
v_mul_f32 v20, v10, v6 // (op 2145) 
v_mul_f32 v21, v10, v7 // (op 2145) 


v_add_f32 v6, v18, v14 // (op 2146) 
v_add_f32 v7, v19, v15 // (op 2146) 
v_add_f32 v8, v20, v16 // (op 2146) 
v_add_f32 v9, v21, v17 // (op 2146) 


v_lshrrev_b32 v14, 16, v6 // (op 2147) 
v_lshrrev_b32 v15, 16, v7 // (op 2147) 
v_lshrrev_b32 v16, 16, v8 // (op 2147) 
v_lshrrev_b32 v17, 16, v9 // (op 2147) 


v_pack_b32_f16 v8, v14, v15 // (op 2130) 
v_pack_b32_f16 v9, v16, v17 // (op 2130) 

s_lshl_b32 s0, s4, 6 // (op 2130) 
s_mul_i32 s1, 3, s0 // (op 2130) 
s_add_u32 s1, s1, 64 // (op 2130) 

v_add_u32 v0, v13, s1 // (op 2130) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 2130) Store value





v_mov_b32 v0, v12 // (op 148) reg expression

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 148) Load value

v_accvgpr_read v4, a0 // (op 144) 
v_accvgpr_read v5, a1 // (op 144) 
v_accvgpr_read v6, a2 // (op 144) 
v_accvgpr_read v7, a3 // (op 144) 
v_mul_f32 v14, v11, v4 // (op 144) 
v_mul_f32 v15, v11, v5 // (op 144) 
v_mul_f32 v16, v11, v6 // (op 144) 
v_mul_f32 v17, v11, v7 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_and_b32 v6, 65535, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_lshrrev_b32 v14, 16, v6 // (op 146) 
v_lshrrev_b32 v15, 16, v7 // (op 146) 
v_lshrrev_b32 v16, 16, v8 // (op 146) 
v_lshrrev_b32 v17, 16, v9 // (op 146) 


v_pack_b32_f16 v8, v14, v15 // (op 147) 
v_pack_b32_f16 v9, v16, v17 // (op 147) 

v_mov_b32 v0, v13 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 2041) 
v_add_u32 v0, v12, s0 // (op 2041) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2041) Load value

v_accvgpr_read v4, a4 // (op 2037) 
v_accvgpr_read v5, a5 // (op 2037) 
v_accvgpr_read v6, a6 // (op 2037) 
v_accvgpr_read v7, a7 // (op 2037) 
v_mul_f32 v14, v11, v4 // (op 2037) 
v_mul_f32 v15, v11, v5 // (op 2037) 
v_mul_f32 v16, v11, v6 // (op 2037) 
v_mul_f32 v17, v11, v7 // (op 2037) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2042) 
v_lshrrev_b32 v7, 16, v8 // (op 2042) 
v_lshlrev_b32 v6, 16, v6 // (op 2042) 
v_lshlrev_b32 v7, 16, v7 // (op 2042) 
v_mul_f32 v18, v10, v6 // (op 2042) 
v_mul_f32 v19, v10, v7 // (op 2042) 
v_and_b32 v6, 65535, v9 // (op 2042) 
v_lshrrev_b32 v7, 16, v9 // (op 2042) 
v_lshlrev_b32 v6, 16, v6 // (op 2042) 
v_lshlrev_b32 v7, 16, v7 // (op 2042) 
v_mul_f32 v20, v10, v6 // (op 2042) 
v_mul_f32 v21, v10, v7 // (op 2042) 


v_add_f32 v6, v18, v14 // (op 2038) 
v_add_f32 v7, v19, v15 // (op 2038) 
v_add_f32 v8, v20, v16 // (op 2038) 
v_add_f32 v9, v21, v17 // (op 2038) 


v_lshrrev_b32 v14, 16, v6 // (op 2039) 
v_lshrrev_b32 v15, 16, v7 // (op 2039) 
v_lshrrev_b32 v16, 16, v8 // (op 2039) 
v_lshrrev_b32 v17, 16, v9 // (op 2039) 


v_pack_b32_f16 v8, v14, v15 // (op 2040) 
v_pack_b32_f16 v9, v16, v17 // (op 2040) 

s_lshl_b32 s0, s4, 6 // (op 2040) 
v_add_u32 v0, v13, s0 // (op 2040) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 2040) Store value





s_lshl_b32 s0, s2, 7 // (op 2055) 
v_add_u32 v0, v12, s0 // (op 2055) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2055) Load value

v_accvgpr_read v4, a8 // (op 2051) 
v_accvgpr_read v5, a9 // (op 2051) 
v_accvgpr_read v6, a10 // (op 2051) 
v_accvgpr_read v7, a11 // (op 2051) 
v_mul_f32 v14, v11, v4 // (op 2051) 
v_mul_f32 v15, v11, v5 // (op 2051) 
v_mul_f32 v16, v11, v6 // (op 2051) 
v_mul_f32 v17, v11, v7 // (op 2051) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2056) 
v_lshrrev_b32 v7, 16, v8 // (op 2056) 
v_lshlrev_b32 v6, 16, v6 // (op 2056) 
v_lshlrev_b32 v7, 16, v7 // (op 2056) 
v_mul_f32 v18, v10, v6 // (op 2056) 
v_mul_f32 v19, v10, v7 // (op 2056) 
v_and_b32 v6, 65535, v9 // (op 2056) 
v_lshrrev_b32 v7, 16, v9 // (op 2056) 
v_lshlrev_b32 v6, 16, v6 // (op 2056) 
v_lshlrev_b32 v7, 16, v7 // (op 2056) 
v_mul_f32 v20, v10, v6 // (op 2056) 
v_mul_f32 v21, v10, v7 // (op 2056) 


v_add_f32 v6, v18, v14 // (op 2052) 
v_add_f32 v7, v19, v15 // (op 2052) 
v_add_f32 v8, v20, v16 // (op 2052) 
v_add_f32 v9, v21, v17 // (op 2052) 


v_lshrrev_b32 v14, 16, v6 // (op 2053) 
v_lshrrev_b32 v15, 16, v7 // (op 2053) 
v_lshrrev_b32 v16, 16, v8 // (op 2053) 
v_lshrrev_b32 v17, 16, v9 // (op 2053) 


v_pack_b32_f16 v8, v14, v15 // (op 2054) 
v_pack_b32_f16 v9, v16, v17 // (op 2054) 

s_lshl_b32 s0, s4, 7 // (op 2054) 
v_add_u32 v0, v13, s0 // (op 2054) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 2054) Store value





s_lshl_b32 s0, s2, 6 // (op 2069) 
s_mul_i32 s1, 3, s0 // (op 2069) 

v_add_u32 v0, v12, s1 // (op 2069) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2069) Load value

v_accvgpr_read v4, a12 // (op 2065) 
v_accvgpr_read v5, a13 // (op 2065) 
v_accvgpr_read v6, a14 // (op 2065) 
v_accvgpr_read v7, a15 // (op 2065) 
v_mul_f32 v14, v11, v4 // (op 2065) 
v_mul_f32 v15, v11, v5 // (op 2065) 
v_mul_f32 v16, v11, v6 // (op 2065) 
v_mul_f32 v17, v11, v7 // (op 2065) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 2070) 
v_lshrrev_b32 v7, 16, v8 // (op 2070) 
v_lshlrev_b32 v6, 16, v6 // (op 2070) 
v_lshlrev_b32 v7, 16, v7 // (op 2070) 
v_mul_f32 v18, v10, v6 // (op 2070) 
v_mul_f32 v19, v10, v7 // (op 2070) 
v_and_b32 v6, 65535, v9 // (op 2070) 
v_lshrrev_b32 v7, 16, v9 // (op 2070) 
v_lshlrev_b32 v6, 16, v6 // (op 2070) 
v_lshlrev_b32 v7, 16, v7 // (op 2070) 
v_mul_f32 v20, v10, v6 // (op 2070) 
v_mul_f32 v21, v10, v7 // (op 2070) 


v_add_f32 v6, v18, v14 // (op 2066) 
v_add_f32 v7, v19, v15 // (op 2066) 
v_add_f32 v8, v20, v16 // (op 2066) 
v_add_f32 v9, v21, v17 // (op 2066) 


v_lshrrev_b32 v14, 16, v6 // (op 2067) 
v_lshrrev_b32 v15, 16, v7 // (op 2067) 
v_lshrrev_b32 v16, 16, v8 // (op 2067) 
v_lshrrev_b32 v17, 16, v9 // (op 2067) 


v_pack_b32_f16 v8, v14, v15 // (op 2068) 
v_pack_b32_f16 v9, v16, v17 // (op 2068) 

s_lshl_b32 s0, s4, 6 // (op 2068) 
s_mul_i32 s1, 3, s0 // (op 2068) 

v_add_u32 v0, v13, s1 // (op 2068) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 2068) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_
.LRR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__end:
.size RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_, .LRR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__end-RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 132
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 34304 // lds bytes
  .amdhsa_accum_offset 100
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F4
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
    macrotile: [64, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP6_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 34304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 98
    .agpr_count: 32
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

