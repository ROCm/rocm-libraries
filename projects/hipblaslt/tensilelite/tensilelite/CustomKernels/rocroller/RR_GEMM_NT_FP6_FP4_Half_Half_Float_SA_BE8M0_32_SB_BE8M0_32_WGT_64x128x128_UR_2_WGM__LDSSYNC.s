// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s12 // (op 4512) 
s_mul_hi_i32 s1, s16, s12 // (op 4769) 
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


v_and_b32 v3, 63, v1 // (op 4632) 
v_and_b32 v4, 48, v1 // (op 4632) 
v_sub_u32 v5, v3, v4 // (op 4632) 

v_lshrrev_b32 v4, 4, v3 // (op 4632) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 4632) 


v_lshrrev_b32 v4, 6, v1 // (op 4632) 
v_and_b32 v5, 1, v4 // (op 4632) 

v_lshlrev_b32 v4, 4, v5 // (op 4632) 

v_lshl_add_u32 v5, v3, 7, v4 // (op 4632) 


v_lshlrev_b32 v3, 2, v5 // (op 4632) 

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

v_lshl_add_u32 v5, v6, 7, v7 // (op 4748) 


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


v_and_b32 v7, 63, v1 // (op 4829) 
v_and_b32 v8, 48, v1 // (op 4829) 
v_sub_u32 v9, v7, v8 // (op 4829) 

v_lshrrev_b32 v8, 4, v7 // (op 4829) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 4829) 


v_lshrrev_b32 v8, 6, v1 // (op 4829) 
v_and_b32 v9, 1, v8 // (op 4829) 

v_lshlrev_b32 v8, 4, v9 // (op 4829) 

v_lshl_add_u32 v9, v7, 7, v8 // (op 4829) 


v_lshlrev_b32 v7, 2, v9 // (op 4829) 

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

v_lshl_add_u32 v9, v10, 7, v11 // (op 4887) 


v_lshrrev_b32 v10, 7, v1 // (op 4512) 
s_add_i32 s18, s0, s16 // (op 4512) 

v_and_b32 v11, 63, v1 // (op 4512) 
s_ashr_i32 s0, s18, 31 // (op 4512) 
v_lshrrev_b32 v12, 4, v11 // (op 4512) 
s_cmp_eq_i32 s12, 0 // (op 4512) 
s_mov_b32 s19, scc // (op 4512) 
v_lshlrev_b32 v13, 2, v12 // (op 4512) 

s_cmp_eq_i32 s19, 0 // (op 4512) 
s_cselect_b32 s51, 0, -1 // (op 4512) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4512) 


s_and_b32 s19, s15, 31 // (op 4512) 
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

s_lshl_b32 s19, s15, 24 // (op 4512) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 4512) 


s_ashr_i32 s51, s19, 31 // (op 4512) 

s_xor_b32 s19, s18, s51 // (op 4512) 

s_sub_i32 s18, s19, s51 // (op 4512) 


v_lshrrev_b32 v14, 4, v13 // (op 4769) 
s_cmp_eq_i32 s12, 0 // (op 4769) 
s_mov_b32 s19, scc // (op 4769) 
v_lshlrev_b32 v15, 2, v14 // (op 4769) 

s_cmp_eq_i32 s19, 0 // (op 4769) 
s_cselect_b32 s51, 0, -1 // (op 4769) 

v_lshl_add_u32 v14, v10, 4, v15 // (op 4769) 


s_and_b32 s19, s15, 31 // (op 4769) 
v_lshrrev_b32 v10, 2, v13 // (op 4769) 

s_lshl_b32 s52, 1, s19 // (op 4769) 
v_and_b32 v13, 3, v10 // (op 4769) 

s_add_i32 s53, s52, s51 // (op 4769) 


v_and_b32 v10, 3, v1 // (op 4769) 
s_and_b32 s51, s1, s53 // (op 4769) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 4769) 


s_add_i32 s1, s0, s51 // (op 4769) 


v_lshrrev_b32 v10, 6, v1 // (op 4769) 
s_ashr_i32 s0, s1, s19 // (op 4769) 


v_and_b32 v13, 1, v10 // (op 4769) 

s_lshl_b32 s1, s15, 24 // (op 4769) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 4769) 


s_ashr_i32 s19, s1, 31 // (op 4769) 

s_xor_b32 s1, s0, s19 // (op 4769) 

s_sub_i32 s0, s1, s19 // (op 4769) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s1, s18, s44 // (op 4512) 
s_add_i32 s19, s1, s18 // (op 4512) 

s_ashr_i32 s1, s19, 31 // (op 4512) 
s_cmp_eq_i32 s44, 0 // (op 4512) 
s_mov_b32 s51, scc // (op 4512) 
s_cmp_eq_i32 s51, 0 // (op 4512) 
s_cselect_b32 s52, 0, -1 // (op 4512) 

s_and_b32 s51, s45, 31 // (op 4512) 
s_lshl_b32 s53, 1, s51 // (op 4512) 
s_add_i32 s54, s53, s52 // (op 4512) 


s_and_b32 s52, s1, s54 // (op 4512) 


s_add_i32 s1, s19, s52 // (op 4512) 


s_ashr_i32 s19, s1, s51 // (op 4512) 


s_lshl_b32 s1, s45, 24 // (op 4512) 
s_ashr_i32 s51, s1, 31 // (op 4512) 

s_xor_b32 s1, s19, s51 // (op 4512) 

s_sub_i32 s19, s1, s51 // (op 4512) 


s_mul_i32 s1, s18, s13 // (op 4512) 
s_sub_u32 s51, s16, s1 // (op 4512) 

s_mul_i32 s1, s19, s13 // (op 4512) 
s_add_u32 s1, s1, s51 // (op 4512) 

s_mul_hi_i32 s51, s16, s11 // (op 4512) 
s_add_i32 s52, s51, s16 // (op 4512) 

s_ashr_i32 s51, s52, 31 // (op 4512) 
s_cmp_eq_i32 s11, 0 // (op 4512) 
s_mov_b32 s53, scc // (op 4512) 
s_cmp_eq_i32 s53, 0 // (op 4512) 
s_cselect_b32 s54, 0, -1 // (op 4512) 

s_and_b32 s53, s47, 31 // (op 4512) 
s_lshl_b32 s55, 1, s53 // (op 4512) 
s_add_i32 s56, s55, s54 // (op 4512) 


s_and_b32 s54, s51, s56 // (op 4512) 


s_add_i32 s51, s52, s54 // (op 4512) 


s_ashr_i32 s52, s51, s53 // (op 4512) 


s_lshl_b32 s51, s47, 24 // (op 4512) 
s_ashr_i32 s53, s51, 31 // (op 4512) 

s_xor_b32 s51, s52, s53 // (op 4512) 

s_sub_i32 s52, s51, s53 // (op 4512) 


s_mul_hi_i32 s51, s0, s44 // (op 4769) 
s_add_i32 s53, s51, s0 // (op 4769) 

s_ashr_i32 s51, s53, 31 // (op 4769) 
s_cmp_eq_i32 s44, 0 // (op 4769) 
s_mov_b32 s54, scc // (op 4769) 
s_cmp_eq_i32 s54, 0 // (op 4769) 
s_cselect_b32 s55, 0, -1 // (op 4769) 

s_and_b32 s54, s45, 31 // (op 4769) 
s_lshl_b32 s56, 1, s54 // (op 4769) 
s_add_i32 s57, s56, s55 // (op 4769) 


s_and_b32 s55, s51, s57 // (op 4769) 


s_add_i32 s51, s53, s55 // (op 4769) 


s_ashr_i32 s53, s51, s54 // (op 4769) 


s_lshl_b32 s51, s45, 24 // (op 4769) 
s_ashr_i32 s54, s51, 31 // (op 4769) 

s_xor_b32 s51, s53, s54 // (op 4769) 

s_sub_i32 s53, s51, s54 // (op 4769) 


s_mul_i32 s51, s0, s13 // (op 4769) 
s_sub_u32 s54, s16, s51 // (op 4769) 

s_mul_i32 s51, s53, s13 // (op 4769) 
s_add_u32 s51, s51, s54 // (op 4769) 

s_mul_hi_i32 s54, s16, s11 // (op 4769) 
s_add_i32 s55, s54, s16 // (op 4769) 

s_ashr_i32 s54, s55, 31 // (op 4769) 
s_cmp_eq_i32 s11, 0 // (op 4769) 
s_mov_b32 s56, scc // (op 4769) 
s_cmp_eq_i32 s56, 0 // (op 4769) 
s_cselect_b32 s57, 0, -1 // (op 4769) 

s_and_b32 s56, s47, 31 // (op 4769) 
s_lshl_b32 s58, 1, s56 // (op 4769) 
s_add_i32 s59, s58, s57 // (op 4769) 


s_and_b32 s57, s54, s59 // (op 4769) 


s_add_i32 s54, s55, s57 // (op 4769) 


s_ashr_i32 s55, s54, s56 // (op 4769) 


s_lshl_b32 s54, s47, 24 // (op 4769) 
s_ashr_i32 s56, s54, 31 // (op 4769) 

s_xor_b32 s54, s55, s56 // (op 4769) 

s_sub_i32 s55, s54, s56 // (op 4769) 


s_add_u32 s56, s8, 128 // (op 4512) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 4512) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4512) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4512) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 4512) 

s_mul_i32 s54, s13, s56 // (op 4512) 
s_mul_i32 s58, s10, s54 // (op 4512) 

s_mul_i32 s54, s52, s58 // (op 4512) 

s_sub_u32 s58, s16, s54 // (op 4512) 

s_mov_b32 s54, s58 // (op 4512) 
s_mul_hi_i32 s59, s54, s14 // (op 4512) 
s_add_i32 s60, s59, s54 // (op 4512) 


s_ashr_i32 s54, s60, 31 // (op 4512) 
s_cmp_eq_i32 s14, 0 // (op 4512) 
s_mov_b32 s59, scc // (op 4512) 
s_cmp_eq_i32 s59, 0 // (op 4512) 
s_cselect_b32 s61, 0, -1 // (op 4512) 

s_and_b32 s59, s46, 31 // (op 4512) 
s_lshl_b32 s62, 1, s59 // (op 4512) 
s_add_i32 s63, s62, s61 // (op 4512) 


s_and_b32 s61, s54, s63 // (op 4512) 


s_add_i32 s54, s60, s61 // (op 4512) 


s_ashr_i32 s60, s54, s59 // (op 4512) 


s_lshl_b32 s54, s46, 24 // (op 4512) 
s_ashr_i32 s59, s54, 31 // (op 4512) 

s_xor_b32 s54, s60, s59 // (op 4512) 

s_sub_i32 s60, s54, s59 // (op 4512) 


s_mul_i32 s54, s48, s13 // (op 4512) 
s_add_u32 s62, s6, 64 // (op 4512) Start of Int64 add, locking SCC
s_addc_u32 s63, s7, 0 // (op 4512) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 4512) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 4512) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 4512) 

s_sub_i32 s59, s62, s54 // (op 4512) 

s_mul_i32 s61, s60, s59 // (op 4512) 

s_sub_u32 s59, s58, s61 // (op 4512) 


s_add_u32 s58, s59, s54 // (op 4512) 


s_cmp_eq_i32 s52, 0 // (op 4512) 
s_mov_b32 s54, scc // (op 4512) 

s_cmp_eq_i32 s54, 0 // (op 4512) 
s_cselect_b32 s52, s58, s1 // (op 4512) 


s_mul_i32 s1, s19, s56 // (op 4512) 


v_lshl_add_u32 v13, s52, 6, v12 // (op 4512) 


s_sub_i32 s19, s18, s1 // (op 4512) 


s_cmp_eq_i32 s54, 0 // (op 4512) 
s_cselect_b32 s1, s60, s19 // (op 4512) 



v_lshl_add_u32 v12, s1, 7, v11 // (op 4512) 


v_mul_lo_u32 v11, v12, s2 // (op 4512) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 4512) 


s_add_u32 s18, s8, 128 // (op 4769) Start of Int64 add, locking SCC
s_addc_u32 s19, s9, 0 // (op 4769) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 4769) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 4769) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 7 // (op 4769) 

s_mul_i32 s1, s13, s18 // (op 4769) 
s_mul_i32 s52, s10, s1 // (op 4769) 

s_mul_i32 s1, s55, s52 // (op 4769) 

s_sub_u32 s52, s16, s1 // (op 4769) 

s_mov_b32 s1, s52 // (op 4769) 
s_mul_hi_i32 s54, s1, s14 // (op 4769) 
s_add_i32 s56, s54, s1 // (op 4769) 


s_ashr_i32 s1, s56, 31 // (op 4769) 
s_cmp_eq_i32 s14, 0 // (op 4769) 
s_mov_b32 s54, scc // (op 4769) 
s_cmp_eq_i32 s54, 0 // (op 4769) 
s_cselect_b32 s57, 0, -1 // (op 4769) 

s_and_b32 s54, s46, 31 // (op 4769) 
s_lshl_b32 s58, 1, s54 // (op 4769) 
s_add_i32 s59, s58, s57 // (op 4769) 


s_and_b32 s57, s1, s59 // (op 4769) 


s_add_i32 s1, s56, s57 // (op 4769) 


s_ashr_i32 s56, s1, s54 // (op 4769) 


s_lshl_b32 s1, s46, 24 // (op 4769) 
s_ashr_i32 s54, s1, 31 // (op 4769) 

s_xor_b32 s1, s56, s54 // (op 4769) 

s_sub_i32 s56, s1, s54 // (op 4769) 


s_mul_i32 s1, s48, s13 // (op 4769) 
s_add_u32 s58, s6, 64 // (op 4769) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 4769) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 4769) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 4769) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 4769) 

s_sub_i32 s54, s58, s1 // (op 4769) 

s_mul_i32 s57, s56, s54 // (op 4769) 

s_sub_u32 s54, s52, s57 // (op 4769) 


s_add_u32 s52, s54, s1 // (op 4769) 


s_cmp_eq_i32 s55, 0 // (op 4769) 
s_mov_b32 s1, scc // (op 4769) 

s_cmp_eq_i32 s1, 0 // (op 4769) 
s_cselect_b32 s54, s52, s51 // (op 4769) 


s_mul_i32 s55, s53, s18 // (op 4769) 


v_lshl_add_u32 v11, s54, 6, v14 // (op 4769) 


s_sub_i32 s18, s0, s55 // (op 4769) 


s_cmp_eq_i32 s1, 0 // (op 4769) 
s_cselect_b32 s0, s56, s18 // (op 4769) 



v_lshl_add_u32 v13, s0, 7, v10 // (op 4769) 


v_mul_lo_u32 v10, v13, s4 // (op 4769) 

v_add_lshl_u32 v13, v11, v10, 1 // (op 4769) 


s_mul_i32 s1, s2, s9 // (op 4514) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 4514) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 4514) least significant
s_mul_hi_u32 s19, s2, s8 // (op 4514) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4514) most significant: sum
s_add_u32 s1, s1, s19 // (op 4514) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 4514) 

s_mul_i32 s1, s4, s9 // (op 4771) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 4771) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 4771) least significant
s_mul_hi_u32 s52, s4, s8 // (op 4771) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 4771) most significant: sum
s_add_u32 s1, s1, s52 // (op 4771) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 4771) 

s_mov_b32 s56, s20 // (op 4514) reg expression
s_mov_b32 s57, s21 // (op 4514) reg expression
s_mov_b32 s58, s18 // (op 4514) reg expression
s_mov_b32 s59, 131072 // (op 4514) call()

s_mov_b32 s60, s22 // (op 4771) reg expression

s_mov_b32 s61, s23 // (op 4771) reg expression
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
s_mul_hi_i32 s0, s16, s12 // (op 4274) 
v_and_b32 v14, -2, v1 // (op 4274) 
s_add_i32 s1, s0, s16 // (op 4274) 

v_sub_u32 v15, v1, v14 // (op 4274) 

s_ashr_i32 s0, s1, 31 // (op 4274) 
v_lshlrev_b32 v14, 5, v15 // (op 4274) 

s_cmp_eq_i32 s12, 0 // (op 4274) 
s_mov_b32 s18, scc // (op 4274) 
v_lshrrev_b32 v15, 1, v1 // (op 4274) 
s_cmp_eq_i32 s18, 0 // (op 4274) 
s_cselect_b32 s19, 0, -1 // (op 4274) 

s_and_b32 s18, s15, 31 // (op 4274) 
s_lshl_b32 s20, 1, s18 // (op 4274) 
s_add_i32 s21, s20, s19 // (op 4274) 


s_and_b32 s19, s0, s21 // (op 4274) 


s_add_i32 s0, s1, s19 // (op 4274) 


s_ashr_i32 s1, s0, s18 // (op 4274) 


s_lshl_b32 s0, s15, 24 // (op 4274) 
s_ashr_i32 s18, s0, 31 // (op 4274) 

s_xor_b32 s0, s1, s18 // (op 4274) 

s_sub_i32 s1, s0, s18 // (op 4274) 


s_mul_hi_i32 s0, s1, s44 // (op 4274) 
s_add_i32 s18, s0, s1 // (op 4274) 

s_ashr_i32 s0, s18, 31 // (op 4274) 
s_cmp_eq_i32 s44, 0 // (op 4274) 
s_mov_b32 s19, scc // (op 4274) 
s_cmp_eq_i32 s19, 0 // (op 4274) 
s_cselect_b32 s20, 0, -1 // (op 4274) 

s_and_b32 s19, s45, 31 // (op 4274) 
s_lshl_b32 s21, 1, s19 // (op 4274) 
s_add_i32 s22, s21, s20 // (op 4274) 


s_and_b32 s23, s0, s22 // (op 4274) 


s_add_i32 s0, s18, s23 // (op 4274) 


s_ashr_i32 s18, s0, s19 // (op 4274) 


s_lshl_b32 s0, s45, 24 // (op 4274) 
s_ashr_i32 s19, s0, 31 // (op 4274) 

s_xor_b32 s0, s18, s19 // (op 4274) 

s_sub_i32 s18, s0, s19 // (op 4274) 


s_mul_i32 s0, s1, s13 // (op 4274) 

s_sub_u32 s1, s16, s0 // (op 4274) 

s_mul_i32 s0, s18, s13 // (op 4274) 
s_add_u32 s0, s0, s1 // (op 4274) 


s_mul_hi_i32 s1, s16, s11 // (op 4274) 
s_add_i32 s18, s1, s16 // (op 4274) 

s_ashr_i32 s1, s18, 31 // (op 4274) 
s_cmp_eq_i32 s11, 0 // (op 4274) 
s_mov_b32 s19, scc // (op 4274) 
s_cmp_eq_i32 s19, 0 // (op 4274) 
s_cselect_b32 s20, 0, -1 // (op 4274) 

s_and_b32 s19, s47, 31 // (op 4274) 
s_lshl_b32 s21, 1, s19 // (op 4274) 
s_add_i32 s22, s21, s20 // (op 4274) 


s_and_b32 s23, s1, s22 // (op 4274) 


s_add_i32 s1, s18, s23 // (op 4274) 


s_ashr_i32 s18, s1, s19 // (op 4274) 


s_lshl_b32 s1, s47, 24 // (op 4274) 
s_ashr_i32 s19, s1, 31 // (op 4274) 

s_xor_b32 s1, s18, s19 // (op 4274) 

s_sub_i32 s18, s1, s19 // (op 4274) 


s_add_u32 s22, s8, 128 // (op 4274) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 4274) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 4274) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 4274) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 7 // (op 4274) 

s_mul_i32 s1, s13, s22 // (op 4274) 

s_mul_i32 s19, s10, s1 // (op 4274) 

s_mul_i32 s1, s18, s19 // (op 4274) 

s_sub_u32 s19, s16, s1 // (op 4274) 

s_mov_b32 s1, s19 // (op 4274) 
s_mul_hi_i32 s20, s1, s14 // (op 4274) 
s_add_i32 s21, s20, s1 // (op 4274) 


s_ashr_i32 s1, s21, 31 // (op 4274) 
s_cmp_eq_i32 s14, 0 // (op 4274) 
s_mov_b32 s20, scc // (op 4274) 
s_cmp_eq_i32 s20, 0 // (op 4274) 
s_cselect_b32 s22, 0, -1 // (op 4274) 

s_and_b32 s20, s46, 31 // (op 4274) 
s_lshl_b32 s23, 1, s20 // (op 4274) 
s_add_i32 s49, s23, s22 // (op 4274) 


s_and_b32 s22, s1, s49 // (op 4274) 


s_add_i32 s1, s21, s22 // (op 4274) 


s_ashr_i32 s21, s1, s20 // (op 4274) 


s_lshl_b32 s1, s46, 24 // (op 4274) 
s_ashr_i32 s20, s1, 31 // (op 4274) 

s_xor_b32 s1, s21, s20 // (op 4274) 

s_sub_i32 s21, s1, s20 // (op 4274) 


s_mul_i32 s1, s48, s13 // (op 4274) 
s_add_u32 s22, s6, 64 // (op 4274) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4274) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 4274) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 4274) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 6 // (op 4274) 

s_sub_i32 s20, s22, s1 // (op 4274) 

s_mul_i32 s22, s21, s20 // (op 4274) 


s_sub_u32 s23, s19, s22 // (op 4274) 


s_add_u32 s19, s23, s1 // (op 4274) 


s_cmp_eq_i32 s18, 0 // (op 4274) 
s_mov_b32 s1, scc // (op 4274) 

s_cmp_eq_i32 s1, 0 // (op 4274) 
s_cselect_b32 s18, s19, s0 // (op 4274) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 4274) 


v_mov_b32 v19, 0 // (op 4274) convert
v_mov_b32 v18, v16 // (op 4274) convert

v_mul_lo_u32 v17, v15, s25 // (op 4274) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 4274) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 4274) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 4274) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 4274) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 4274) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 4274) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 4274) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 4274) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 4274) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 4274) 

v_mov_b32 v16, v14 // (op 4274) convert

s_mul_i32 s1, s29, 6 // (op 4276) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 4276) least significant
s_mul_hi_u32 s18, s28, 6 // (op 4276) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4276) most significant: sum

s_mul_i32 s19, s24, s1 // (op 4276) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 4276) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 4276) least significant
s_mul_hi_u32 s21, s24, s0 // (op 4276) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 4276) most significant: sum
s_add_u32 s19, s19, s21 // (op 4276) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4276) 

s_mov_b32 s20, s26 // (op 4276) reg expression
s_mov_b32 s21, s27 // (op 4276) reg expression
s_mov_b32 s22, s0 // (op 4276) reg expression
s_mov_b32 s23, 131072 // (op 4276) call()

s_mul_hi_i32 s0, s16, s12 // (op 4293) 
v_and_b32 v14, -2, v1 // (op 4293) 
s_add_i32 s1, s0, s16 // (op 4293) 

v_sub_u32 v15, v1, v14 // (op 4293) 

s_ashr_i32 s0, s1, 31 // (op 4293) 
v_lshlrev_b32 v14, 5, v15 // (op 4293) 

s_cmp_eq_i32 s12, 0 // (op 4293) 
s_mov_b32 s18, scc // (op 4293) 
v_lshrrev_b32 v15, 1, v1 // (op 4293) 
s_cmp_eq_i32 s18, 0 // (op 4293) 
s_cselect_b32 s19, 0, -1 // (op 4293) 

s_and_b32 s18, s15, 31 // (op 4293) 
s_lshl_b32 s49, 1, s18 // (op 4293) 
s_add_i32 s50, s49, s19 // (op 4293) 


s_and_b32 s19, s0, s50 // (op 4293) 


s_add_i32 s0, s1, s19 // (op 4293) 


s_ashr_i32 s1, s0, s18 // (op 4293) 


s_lshl_b32 s0, s15, 24 // (op 4293) 
s_ashr_i32 s18, s0, 31 // (op 4293) 

s_xor_b32 s0, s1, s18 // (op 4293) 

s_sub_i32 s1, s0, s18 // (op 4293) 


s_mul_hi_i32 s0, s1, s44 // (op 4293) 
s_add_i32 s18, s0, s1 // (op 4293) 

s_ashr_i32 s0, s18, 31 // (op 4293) 
s_cmp_eq_i32 s44, 0 // (op 4293) 
s_mov_b32 s19, scc // (op 4293) 
s_cmp_eq_i32 s19, 0 // (op 4293) 
s_cselect_b32 s49, 0, -1 // (op 4293) 

s_and_b32 s19, s45, 31 // (op 4293) 
s_lshl_b32 s50, 1, s19 // (op 4293) 
s_add_i32 s51, s50, s49 // (op 4293) 


s_and_b32 s49, s0, s51 // (op 4293) 


s_add_i32 s0, s18, s49 // (op 4293) 


s_ashr_i32 s18, s0, s19 // (op 4293) 


s_lshl_b32 s0, s45, 24 // (op 4293) 
s_ashr_i32 s19, s0, 31 // (op 4293) 

s_xor_b32 s0, s18, s19 // (op 4293) 

s_sub_i32 s18, s0, s19 // (op 4293) 


s_mul_i32 s0, s1, s13 // (op 4293) 

s_sub_u32 s1, s16, s0 // (op 4293) 

s_mul_i32 s0, s18, s13 // (op 4293) 
s_add_u32 s0, s0, s1 // (op 4293) 


s_mul_hi_i32 s1, s16, s11 // (op 4293) 
s_add_i32 s18, s1, s16 // (op 4293) 

s_ashr_i32 s1, s18, 31 // (op 4293) 
s_cmp_eq_i32 s11, 0 // (op 4293) 
s_mov_b32 s19, scc // (op 4293) 
s_cmp_eq_i32 s19, 0 // (op 4293) 
s_cselect_b32 s49, 0, -1 // (op 4293) 

s_and_b32 s19, s47, 31 // (op 4293) 
s_lshl_b32 s50, 1, s19 // (op 4293) 
s_add_i32 s51, s50, s49 // (op 4293) 


s_and_b32 s49, s1, s51 // (op 4293) 


s_add_i32 s1, s18, s49 // (op 4293) 


s_ashr_i32 s18, s1, s19 // (op 4293) 


s_lshl_b32 s1, s47, 24 // (op 4293) 
s_ashr_i32 s19, s1, 31 // (op 4293) 

s_xor_b32 s1, s18, s19 // (op 4293) 

s_sub_i32 s18, s1, s19 // (op 4293) 


s_add_u32 s54, s8, 128 // (op 4293) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 4293) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 4293) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 4293) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 4293) 

s_mul_i32 s1, s13, s50 // (op 4293) 

s_mul_i32 s19, s10, s1 // (op 4293) 

s_mul_i32 s1, s18, s19 // (op 4293) 

s_sub_u32 s19, s16, s1 // (op 4293) 

s_mov_b32 s1, s19 // (op 4293) 
s_mul_hi_i32 s49, s1, s14 // (op 4293) 
s_add_i32 s50, s49, s1 // (op 4293) 


s_ashr_i32 s1, s50, 31 // (op 4293) 
s_cmp_eq_i32 s14, 0 // (op 4293) 
s_mov_b32 s49, scc // (op 4293) 
s_cmp_eq_i32 s49, 0 // (op 4293) 
s_cselect_b32 s51, 0, -1 // (op 4293) 

s_and_b32 s49, s46, 31 // (op 4293) 
s_lshl_b32 s52, 1, s49 // (op 4293) 
s_add_i32 s53, s52, s51 // (op 4293) 


s_and_b32 s51, s1, s53 // (op 4293) 


s_add_i32 s1, s50, s51 // (op 4293) 


s_ashr_i32 s50, s1, s49 // (op 4293) 


s_lshl_b32 s1, s46, 24 // (op 4293) 
s_ashr_i32 s49, s1, 31 // (op 4293) 

s_xor_b32 s1, s50, s49 // (op 4293) 

s_sub_i32 s50, s1, s49 // (op 4293) 


s_mul_i32 s1, s48, s13 // (op 4293) 
s_add_u32 s54, s6, 64 // (op 4293) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 4293) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 4293) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 4293) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 6 // (op 4293) 

s_sub_i32 s49, s54, s1 // (op 4293) 

s_mul_i32 s51, s50, s49 // (op 4293) 


s_sub_u32 s49, s19, s51 // (op 4293) 


s_add_u32 s19, s49, s1 // (op 4293) 


s_cmp_eq_i32 s18, 0 // (op 4293) 
s_mov_b32 s1, scc // (op 4293) 

s_cmp_eq_i32 s1, 0 // (op 4293) 
s_cselect_b32 s18, s19, s0 // (op 4293) 



v_lshl_add_u32 v17, s18, 6, v14 // (op 4293) 


v_mov_b32 v19, 0 // (op 4293) convert
v_mov_b32 v18, v17 // (op 4293) convert

v_mul_lo_u32 v21, v15, s25 // (op 4293) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 4293) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 4293) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 4293) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 4293) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 4293) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 4293) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 4293) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 4293) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 4293) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 4293) 

v_mov_b32 v17, v14 // (op 4293) convert

s_mul_i32 s1, s29, 6 // (op 4295) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 4295) least significant
s_mul_hi_u32 s18, s28, 6 // (op 4295) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4295) most significant: sum

s_mul_i32 s19, s24, s1 // (op 4295) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 4295) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 4295) least significant
s_mul_hi_u32 s50, s24, s0 // (op 4295) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4295) most significant: sum
s_add_u32 s19, s19, s50 // (op 4295) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4295) 

s_mov_b32 s20, s26 // (op 4295) reg expression
s_mov_b32 s21, s27 // (op 4295) reg expression
s_mov_b32 s22, s0 // (op 4295) reg expression
s_mov_b32 s23, 131072 // (op 4295) call()

s_mul_hi_i32 s0, s16, s12 // (op 4338) 
v_and_b32 v14, -4, v1 // (op 4338) 
s_add_i32 s1, s0, s16 // (op 4338) 

v_sub_u32 v15, v1, v14 // (op 4338) 

s_ashr_i32 s0, s1, 31 // (op 4338) 
v_lshlrev_b32 v14, 5, v15 // (op 4338) 

s_cmp_eq_i32 s12, 0 // (op 4338) 
s_mov_b32 s18, scc // (op 4338) 
v_lshrrev_b32 v15, 2, v1 // (op 4338) 
s_cmp_eq_i32 s18, 0 // (op 4338) 
s_cselect_b32 s19, 0, -1 // (op 4338) 

v_lshlrev_b32 v18, 1, v15 // (op 4338) 

s_and_b32 s18, s15, 31 // (op 4338) 
s_lshl_b32 s49, 1, s18 // (op 4338) 
s_add_i32 s50, s49, s19 // (op 4338) 


s_and_b32 s19, s0, s50 // (op 4338) 


s_add_i32 s0, s1, s19 // (op 4338) 


s_ashr_i32 s1, s0, s18 // (op 4338) 


s_lshl_b32 s0, s15, 24 // (op 4338) 
s_ashr_i32 s18, s0, 31 // (op 4338) 

s_xor_b32 s0, s1, s18 // (op 4338) 

s_sub_i32 s1, s0, s18 // (op 4338) 


s_mul_hi_i32 s0, s1, s44 // (op 4338) 
s_add_i32 s18, s0, s1 // (op 4338) 

s_ashr_i32 s0, s18, 31 // (op 4338) 
s_cmp_eq_i32 s44, 0 // (op 4338) 
s_mov_b32 s19, scc // (op 4338) 
s_cmp_eq_i32 s19, 0 // (op 4338) 
s_cselect_b32 s49, 0, -1 // (op 4338) 

s_and_b32 s19, s45, 31 // (op 4338) 
s_lshl_b32 s50, 1, s19 // (op 4338) 
s_add_i32 s51, s50, s49 // (op 4338) 


s_and_b32 s49, s0, s51 // (op 4338) 


s_add_i32 s0, s18, s49 // (op 4338) 


s_ashr_i32 s18, s0, s19 // (op 4338) 


s_lshl_b32 s0, s45, 24 // (op 4338) 
s_ashr_i32 s19, s0, 31 // (op 4338) 

s_xor_b32 s0, s18, s19 // (op 4338) 

s_sub_i32 s18, s0, s19 // (op 4338) 


s_add_u32 s54, s8, 128 // (op 4338) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 4338) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 4338) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 4338) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 4338) 

s_mul_i32 s0, s18, s50 // (op 4338) 

s_sub_i32 s49, s1, s0 // (op 4338) 


s_mul_hi_i32 s0, s16, s11 // (op 4338) 
s_add_i32 s1, s0, s16 // (op 4338) 

s_ashr_i32 s0, s1, 31 // (op 4338) 
s_cmp_eq_i32 s11, 0 // (op 4338) 
s_mov_b32 s18, scc // (op 4338) 
s_cmp_eq_i32 s18, 0 // (op 4338) 
s_cselect_b32 s19, 0, -1 // (op 4338) 

s_and_b32 s18, s47, 31 // (op 4338) 
s_lshl_b32 s52, 1, s18 // (op 4338) 
s_add_i32 s53, s52, s19 // (op 4338) 


s_and_b32 s19, s0, s53 // (op 4338) 


s_add_i32 s0, s1, s19 // (op 4338) 


s_ashr_i32 s1, s0, s18 // (op 4338) 


s_lshl_b32 s0, s47, 24 // (op 4338) 
s_ashr_i32 s18, s0, 31 // (op 4338) 

s_xor_b32 s0, s1, s18 // (op 4338) 

s_sub_i32 s1, s0, s18 // (op 4338) 


s_mul_i32 s0, s13, s50 // (op 4338) 

s_mul_i32 s18, s10, s0 // (op 4338) 

s_mul_i32 s0, s1, s18 // (op 4338) 

s_sub_u32 s18, s16, s0 // (op 4338) 

s_mul_hi_i32 s0, s18, s14 // (op 4338) 
s_add_i32 s19, s0, s18 // (op 4338) 


s_ashr_i32 s0, s19, 31 // (op 4338) 
s_cmp_eq_i32 s14, 0 // (op 4338) 
s_mov_b32 s18, scc // (op 4338) 
s_cmp_eq_i32 s18, 0 // (op 4338) 
s_cselect_b32 s50, 0, -1 // (op 4338) 

s_and_b32 s18, s46, 31 // (op 4338) 
s_lshl_b32 s51, 1, s18 // (op 4338) 
s_add_i32 s52, s51, s50 // (op 4338) 


s_and_b32 s50, s0, s52 // (op 4338) 


s_add_i32 s0, s19, s50 // (op 4338) 


s_ashr_i32 s19, s0, s18 // (op 4338) 


s_lshl_b32 s0, s46, 24 // (op 4338) 
s_ashr_i32 s18, s0, 31 // (op 4338) 

s_xor_b32 s0, s19, s18 // (op 4338) 

s_sub_i32 s19, s0, s18 // (op 4338) 


s_cmp_eq_i32 s1, 0 // (op 4338) 
s_mov_b32 s0, scc // (op 4338) 

s_cmp_eq_i32 s0, 0 // (op 4338) 
s_cselect_b32 s1, s19, s49 // (op 4338) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 4338) 


v_mov_b32 v21, 0 // (op 4338) convert
v_mov_b32 v20, v15 // (op 4338) convert

v_mul_lo_u32 v15, v18, s31 // (op 4338) most significant: low of low * high
v_mul_hi_u32 v19, v18, s30 // (op 4338) most significant: high of low * low
v_mul_lo_u32 v14, v18, s30 // (op 4338) least significant: low of low * low
v_add_u32 v15, v15, v19 // (op 4338) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v20 // (op 4338) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 4338) most significant half



v_lshlrev_b64 v[18:19], 2, v[14:15] // (op 4338) 

v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 4338) 

v_mov_b32 v18, v14 // (op 4338) convert

s_mul_i32 s1, s30, s29 // (op 4340) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 4340) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 4340) least significant
s_mul_hi_u32 s19, s30, s28 // (op 4340) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4340) most significant: sum
s_add_u32 s1, s1, s19 // (op 4340) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4340) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 4340) 

s_mov_b32 s52, s32 // (op 4340) reg expression
s_mov_b32 s53, s33 // (op 4340) reg expression
s_mov_b32 s54, s0 // (op 4340) reg expression
s_mov_b32 s55, 131072 // (op 4340) call()

s_mul_hi_i32 s0, s16, s12 // (op 4357) 
v_and_b32 v14, -4, v1 // (op 4357) 
s_add_i32 s1, s0, s16 // (op 4357) 

v_sub_u32 v15, v1, v14 // (op 4357) 

s_ashr_i32 s0, s1, 31 // (op 4357) 
v_lshlrev_b32 v14, 5, v15 // (op 4357) 

s_cmp_eq_i32 s12, 0 // (op 4357) 
s_mov_b32 s18, scc // (op 4357) 
v_lshrrev_b32 v15, 2, v1 // (op 4357) 
s_cmp_eq_i32 s18, 0 // (op 4357) 
s_cselect_b32 s19, 0, -1 // (op 4357) 

v_lshlrev_b32 v19, 1, v15 // (op 4357) 

s_and_b32 s18, s15, 31 // (op 4357) 
s_lshl_b32 s49, 1, s18 // (op 4357) 
s_add_i32 s50, s49, s19 // (op 4357) 


s_and_b32 s19, s0, s50 // (op 4357) 


s_add_i32 s0, s1, s19 // (op 4357) 


s_ashr_i32 s1, s0, s18 // (op 4357) 


s_lshl_b32 s0, s15, 24 // (op 4357) 
s_ashr_i32 s18, s0, 31 // (op 4357) 

s_xor_b32 s0, s1, s18 // (op 4357) 

s_sub_i32 s1, s0, s18 // (op 4357) 


s_mul_hi_i32 s0, s1, s44 // (op 4357) 
s_add_i32 s18, s0, s1 // (op 4357) 

s_ashr_i32 s0, s18, 31 // (op 4357) 
s_cmp_eq_i32 s44, 0 // (op 4357) 
s_mov_b32 s19, scc // (op 4357) 
s_cmp_eq_i32 s19, 0 // (op 4357) 
s_cselect_b32 s49, 0, -1 // (op 4357) 

s_and_b32 s19, s45, 31 // (op 4357) 
s_lshl_b32 s50, 1, s19 // (op 4357) 
s_add_i32 s51, s50, s49 // (op 4357) 


s_and_b32 s49, s0, s51 // (op 4357) 


s_add_i32 s0, s18, s49 // (op 4357) 


s_ashr_i32 s18, s0, s19 // (op 4357) 


s_lshl_b32 s0, s45, 24 // (op 4357) 
s_ashr_i32 s19, s0, 31 // (op 4357) 

s_xor_b32 s0, s18, s19 // (op 4357) 

s_sub_i32 s18, s0, s19 // (op 4357) 


s_add_u32 s50, s8, 128 // (op 4357) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4357) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 4357) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 4357) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 4357) 

s_mul_i32 s0, s18, s50 // (op 4357) 

s_sub_i32 s49, s1, s0 // (op 4357) 


s_mul_hi_i32 s0, s16, s11 // (op 4357) 
s_add_i32 s1, s0, s16 // (op 4357) 

s_ashr_i32 s0, s1, 31 // (op 4357) 
s_cmp_eq_i32 s11, 0 // (op 4357) 
s_mov_b32 s18, scc // (op 4357) 
s_cmp_eq_i32 s18, 0 // (op 4357) 
s_cselect_b32 s19, 0, -1 // (op 4357) 

s_and_b32 s18, s47, 31 // (op 4357) 
s_lshl_b32 s64, 1, s18 // (op 4357) 
s_add_i32 s65, s64, s19 // (op 4357) 


s_and_b32 s19, s0, s65 // (op 4357) 


s_add_i32 s0, s1, s19 // (op 4357) 


s_ashr_i32 s1, s0, s18 // (op 4357) 


s_lshl_b32 s0, s47, 24 // (op 4357) 
s_ashr_i32 s18, s0, 31 // (op 4357) 

s_xor_b32 s0, s1, s18 // (op 4357) 

s_sub_i32 s1, s0, s18 // (op 4357) 


s_mul_i32 s0, s13, s50 // (op 4357) 

s_mul_i32 s18, s10, s0 // (op 4357) 

s_mul_i32 s0, s1, s18 // (op 4357) 

s_sub_u32 s18, s16, s0 // (op 4357) 

s_mul_hi_i32 s0, s18, s14 // (op 4357) 
s_add_i32 s19, s0, s18 // (op 4357) 


s_ashr_i32 s0, s19, 31 // (op 4357) 
s_cmp_eq_i32 s14, 0 // (op 4357) 
s_mov_b32 s18, scc // (op 4357) 
s_cmp_eq_i32 s18, 0 // (op 4357) 
s_cselect_b32 s50, 0, -1 // (op 4357) 

s_and_b32 s18, s46, 31 // (op 4357) 
s_lshl_b32 s51, 1, s18 // (op 4357) 
s_add_i32 s64, s51, s50 // (op 4357) 


s_and_b32 s50, s0, s64 // (op 4357) 


s_add_i32 s0, s19, s50 // (op 4357) 


s_ashr_i32 s19, s0, s18 // (op 4357) 


s_lshl_b32 s0, s46, 24 // (op 4357) 
s_ashr_i32 s18, s0, 31 // (op 4357) 

s_xor_b32 s0, s19, s18 // (op 4357) 

s_sub_i32 s19, s0, s18 // (op 4357) 


s_cmp_eq_i32 s1, 0 // (op 4357) 
s_mov_b32 s0, scc // (op 4357) 

s_cmp_eq_i32 s0, 0 // (op 4357) 
s_cselect_b32 s1, s19, s49 // (op 4357) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 4357) 


v_mov_b32 v21, 0 // (op 4357) convert
v_mov_b32 v20, v15 // (op 4357) convert

v_mul_lo_u32 v15, v19, s31 // (op 4357) most significant: low of low * high
v_mul_hi_u32 v22, v19, s30 // (op 4357) most significant: high of low * low
v_mul_lo_u32 v14, v19, s30 // (op 4357) least significant: low of low * low
v_add_u32 v15, v15, v22 // (op 4357) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v20 // (op 4357) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 4357) most significant half



v_lshlrev_b64 v[20:21], 2, v[14:15] // (op 4357) 

v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 4357) 

v_mov_b32 v19, v14 // (op 4357) convert

s_mul_i32 s1, s30, s29 // (op 4359) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 4359) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 4359) least significant
s_mul_hi_u32 s19, s30, s28 // (op 4359) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4359) most significant: sum
s_add_u32 s1, s1, s19 // (op 4359) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4359) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 4359) 

s_mov_b32 s52, s32 // (op 4359) reg expression
s_mov_b32 s53, s33 // (op 4359) reg expression
s_mov_b32 s54, s0 // (op 4359) reg expression
s_mov_b32 s55, 131072 // (op 4359) call()

s_mul_hi_i32 s0, s16, s12 // (op 4402) 
v_and_b32 v14, -64, v1 // (op 4402) 
s_add_i32 s1, s0, s16 // (op 4402) 

v_sub_u32 v15, v1, v14 // (op 4402) 

s_ashr_i32 s0, s1, 31 // (op 4402) 
v_lshrrev_b32 v14, 6, v1 // (op 4402) 
s_cmp_eq_i32 s12, 0 // (op 4402) 
s_mov_b32 s18, scc // (op 4402) 
s_cmp_eq_i32 s18, 0 // (op 4402) 
s_cselect_b32 s19, 0, -1 // (op 4402) 

s_and_b32 s18, s15, 31 // (op 4402) 
s_lshl_b32 s49, 1, s18 // (op 4402) 
s_add_i32 s50, s49, s19 // (op 4402) 


s_and_b32 s19, s0, s50 // (op 4402) 


s_add_i32 s0, s1, s19 // (op 4402) 


s_ashr_i32 s1, s0, s18 // (op 4402) 


s_lshl_b32 s0, s15, 24 // (op 4402) 
s_ashr_i32 s18, s0, 31 // (op 4402) 

s_xor_b32 s0, s1, s18 // (op 4402) 

s_sub_i32 s1, s0, s18 // (op 4402) 


s_mul_hi_i32 s0, s1, s44 // (op 4402) 
s_add_i32 s18, s0, s1 // (op 4402) 

s_ashr_i32 s0, s18, 31 // (op 4402) 
s_cmp_eq_i32 s44, 0 // (op 4402) 
s_mov_b32 s19, scc // (op 4402) 
s_cmp_eq_i32 s19, 0 // (op 4402) 
s_cselect_b32 s49, 0, -1 // (op 4402) 

s_and_b32 s19, s45, 31 // (op 4402) 
s_lshl_b32 s50, 1, s19 // (op 4402) 
s_add_i32 s51, s50, s49 // (op 4402) 


s_and_b32 s49, s0, s51 // (op 4402) 


s_add_i32 s0, s18, s49 // (op 4402) 


s_ashr_i32 s18, s0, s19 // (op 4402) 


s_lshl_b32 s0, s45, 24 // (op 4402) 
s_ashr_i32 s19, s0, 31 // (op 4402) 

s_xor_b32 s0, s18, s19 // (op 4402) 

s_sub_i32 s18, s0, s19 // (op 4402) 


s_mul_i32 s0, s1, s13 // (op 4402) 

s_sub_u32 s1, s16, s0 // (op 4402) 

s_mul_i32 s0, s18, s13 // (op 4402) 
s_add_u32 s0, s0, s1 // (op 4402) 


s_mul_hi_i32 s1, s16, s11 // (op 4402) 
s_add_i32 s18, s1, s16 // (op 4402) 

s_ashr_i32 s1, s18, 31 // (op 4402) 
s_cmp_eq_i32 s11, 0 // (op 4402) 
s_mov_b32 s19, scc // (op 4402) 
s_cmp_eq_i32 s19, 0 // (op 4402) 
s_cselect_b32 s49, 0, -1 // (op 4402) 

s_and_b32 s19, s47, 31 // (op 4402) 
s_lshl_b32 s50, 1, s19 // (op 4402) 
s_add_i32 s51, s50, s49 // (op 4402) 


s_and_b32 s49, s1, s51 // (op 4402) 


s_add_i32 s1, s18, s49 // (op 4402) 


s_ashr_i32 s18, s1, s19 // (op 4402) 


s_lshl_b32 s1, s47, 24 // (op 4402) 
s_ashr_i32 s19, s1, 31 // (op 4402) 

s_xor_b32 s1, s18, s19 // (op 4402) 

s_sub_i32 s18, s1, s19 // (op 4402) 


s_add_u32 s50, s8, 128 // (op 4402) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4402) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 4402) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 4402) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 4402) 

s_mul_i32 s1, s13, s50 // (op 4402) 

s_mul_i32 s19, s10, s1 // (op 4402) 

s_mul_i32 s1, s18, s19 // (op 4402) 

s_sub_u32 s19, s16, s1 // (op 4402) 

s_mov_b32 s1, s19 // (op 4402) 
s_mul_hi_i32 s49, s1, s14 // (op 4402) 
s_add_i32 s50, s49, s1 // (op 4402) 


s_ashr_i32 s1, s50, 31 // (op 4402) 
s_cmp_eq_i32 s14, 0 // (op 4402) 
s_mov_b32 s49, scc // (op 4402) 
s_cmp_eq_i32 s49, 0 // (op 4402) 
s_cselect_b32 s51, 0, -1 // (op 4402) 

s_and_b32 s49, s46, 31 // (op 4402) 
s_lshl_b32 s64, 1, s49 // (op 4402) 
s_add_i32 s65, s64, s51 // (op 4402) 


s_and_b32 s51, s1, s65 // (op 4402) 


s_add_i32 s1, s50, s51 // (op 4402) 


s_ashr_i32 s50, s1, s49 // (op 4402) 


s_lshl_b32 s1, s46, 24 // (op 4402) 
s_ashr_i32 s49, s1, 31 // (op 4402) 

s_xor_b32 s1, s50, s49 // (op 4402) 

s_sub_i32 s50, s1, s49 // (op 4402) 


s_mul_i32 s1, s48, s13 // (op 4402) 
s_add_u32 s64, s6, 64 // (op 4402) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 4402) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 4402) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 4402) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 6 // (op 4402) 

s_sub_i32 s49, s64, s1 // (op 4402) 

s_mul_i32 s51, s50, s49 // (op 4402) 


s_sub_u32 s49, s19, s51 // (op 4402) 


s_add_u32 s19, s49, s1 // (op 4402) 


s_cmp_eq_i32 s18, 0 // (op 4402) 
s_mov_b32 s1, scc // (op 4402) 

s_cmp_eq_i32 s1, 0 // (op 4402) 
s_cselect_b32 s18, s19, s0 // (op 4402) 



v_lshl_add_u32 v20, s18, 6, v15 // (op 4402) 


v_mul_lo_u32 v15, v14, s34 // (op 4402) 

v_add_u32 v14, v20, v15 // (op 4402) 


s_mul_i32 s1, s34, s39 // (op 4404) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 4404) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 4404) least significant
s_mul_hi_u32 s19, s34, s38 // (op 4404) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4404) most significant: sum
s_add_u32 s1, s1, s19 // (op 4404) most significant: sum


s_mov_b32 s64, s36 // (op 4404) reg expression
s_mov_b32 s65, s37 // (op 4404) reg expression
s_mov_b32 s66, s0 // (op 4404) reg expression
s_mov_b32 s67, 131072 // (op 4404) call()

s_mul_hi_i32 s0, s16, s12 // (op 4421) 
v_and_b32 v15, -64, v1 // (op 4421) 
s_add_i32 s1, s0, s16 // (op 4421) 

v_sub_u32 v20, v1, v15 // (op 4421) 

s_ashr_i32 s0, s1, 31 // (op 4421) 
v_lshrrev_b32 v15, 6, v1 // (op 4421) 
s_cmp_eq_i32 s12, 0 // (op 4421) 
s_mov_b32 s18, scc // (op 4421) 
s_cmp_eq_i32 s18, 0 // (op 4421) 
s_cselect_b32 s19, 0, -1 // (op 4421) 

s_and_b32 s18, s15, 31 // (op 4421) 
s_lshl_b32 s49, 1, s18 // (op 4421) 
s_add_i32 s50, s49, s19 // (op 4421) 


s_and_b32 s19, s0, s50 // (op 4421) 


s_add_i32 s0, s1, s19 // (op 4421) 


s_ashr_i32 s1, s0, s18 // (op 4421) 


s_lshl_b32 s0, s15, 24 // (op 4421) 
s_ashr_i32 s18, s0, 31 // (op 4421) 

s_xor_b32 s0, s1, s18 // (op 4421) 

s_sub_i32 s1, s0, s18 // (op 4421) 


s_mul_hi_i32 s0, s1, s44 // (op 4421) 
s_add_i32 s18, s0, s1 // (op 4421) 

s_ashr_i32 s0, s18, 31 // (op 4421) 
s_cmp_eq_i32 s44, 0 // (op 4421) 
s_mov_b32 s19, scc // (op 4421) 
s_cmp_eq_i32 s19, 0 // (op 4421) 
s_cselect_b32 s49, 0, -1 // (op 4421) 

s_and_b32 s19, s45, 31 // (op 4421) 
s_lshl_b32 s50, 1, s19 // (op 4421) 
s_add_i32 s51, s50, s49 // (op 4421) 


s_and_b32 s49, s0, s51 // (op 4421) 


s_add_i32 s0, s18, s49 // (op 4421) 


s_ashr_i32 s18, s0, s19 // (op 4421) 


s_lshl_b32 s0, s45, 24 // (op 4421) 
s_ashr_i32 s19, s0, 31 // (op 4421) 

s_xor_b32 s0, s18, s19 // (op 4421) 

s_sub_i32 s18, s0, s19 // (op 4421) 


s_mul_i32 s0, s1, s13 // (op 4421) 

s_sub_u32 s1, s16, s0 // (op 4421) 

s_mul_i32 s0, s18, s13 // (op 4421) 
s_add_u32 s0, s0, s1 // (op 4421) 


s_mul_hi_i32 s1, s16, s11 // (op 4421) 
s_add_i32 s18, s1, s16 // (op 4421) 

s_ashr_i32 s1, s18, 31 // (op 4421) 
s_cmp_eq_i32 s11, 0 // (op 4421) 
s_mov_b32 s19, scc // (op 4421) 
s_cmp_eq_i32 s19, 0 // (op 4421) 
s_cselect_b32 s49, 0, -1 // (op 4421) 

s_and_b32 s19, s47, 31 // (op 4421) 
s_lshl_b32 s50, 1, s19 // (op 4421) 
s_add_i32 s51, s50, s49 // (op 4421) 


s_and_b32 s49, s1, s51 // (op 4421) 


s_add_i32 s1, s18, s49 // (op 4421) 


s_ashr_i32 s18, s1, s19 // (op 4421) 


s_lshl_b32 s1, s47, 24 // (op 4421) 
s_ashr_i32 s19, s1, 31 // (op 4421) 

s_xor_b32 s1, s18, s19 // (op 4421) 

s_sub_i32 s18, s1, s19 // (op 4421) 


s_add_u32 s50, s8, 128 // (op 4421) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4421) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 4421) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 4421) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 4421) 

s_mul_i32 s1, s13, s50 // (op 4421) 

s_mul_i32 s19, s10, s1 // (op 4421) 

s_mul_i32 s1, s18, s19 // (op 4421) 

s_sub_u32 s19, s16, s1 // (op 4421) 

s_mov_b32 s1, s19 // (op 4421) 
s_mul_hi_i32 s49, s1, s14 // (op 4421) 
s_add_i32 s50, s49, s1 // (op 4421) 


s_ashr_i32 s1, s50, 31 // (op 4421) 
s_cmp_eq_i32 s14, 0 // (op 4421) 
s_mov_b32 s49, scc // (op 4421) 
s_cmp_eq_i32 s49, 0 // (op 4421) 
s_cselect_b32 s51, 0, -1 // (op 4421) 

s_and_b32 s49, s46, 31 // (op 4421) 
s_lshl_b32 s68, 1, s49 // (op 4421) 
s_add_i32 s69, s68, s51 // (op 4421) 


s_and_b32 s51, s1, s69 // (op 4421) 


s_add_i32 s1, s50, s51 // (op 4421) 


s_ashr_i32 s50, s1, s49 // (op 4421) 


s_lshl_b32 s1, s46, 24 // (op 4421) 
s_ashr_i32 s49, s1, 31 // (op 4421) 

s_xor_b32 s1, s50, s49 // (op 4421) 

s_sub_i32 s50, s1, s49 // (op 4421) 


s_mul_i32 s1, s48, s13 // (op 4421) 
s_add_u32 s68, s6, 64 // (op 4421) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 4421) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 4421) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 4421) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 6 // (op 4421) 

s_sub_i32 s49, s68, s1 // (op 4421) 

s_mul_i32 s51, s50, s49 // (op 4421) 


s_sub_u32 s49, s19, s51 // (op 4421) 


s_add_u32 s19, s49, s1 // (op 4421) 


s_cmp_eq_i32 s18, 0 // (op 4421) 
s_mov_b32 s1, scc // (op 4421) 

s_cmp_eq_i32 s1, 0 // (op 4421) 
s_cselect_b32 s18, s19, s0 // (op 4421) 



v_lshl_add_u32 v21, s18, 6, v20 // (op 4421) 


v_mul_lo_u32 v20, v15, s34 // (op 4421) 

v_add_u32 v15, v21, v20 // (op 4421) 


s_mul_i32 s1, s34, s39 // (op 4423) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 4423) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 4423) least significant
s_mul_hi_u32 s19, s34, s38 // (op 4423) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4423) most significant: sum
s_add_u32 s1, s1, s19 // (op 4423) most significant: sum


s_mov_b32 s64, s36 // (op 4423) reg expression
s_mov_b32 s65, s37 // (op 4423) reg expression
s_mov_b32 s66, s0 // (op 4423) reg expression
s_mov_b32 s67, 131072 // (op 4423) call()

s_mul_hi_i32 s0, s16, s12 // (op 4466) 
v_and_b32 v20, -128, v1 // (op 4466) 
s_add_i32 s1, s0, s16 // (op 4466) 

v_sub_u32 v21, v1, v20 // (op 4466) 

s_ashr_i32 s0, s1, 31 // (op 4466) 
v_lshrrev_b32 v20, 7, v1 // (op 4466) 
s_cmp_eq_i32 s12, 0 // (op 4466) 
s_mov_b32 s18, scc // (op 4466) 
v_lshlrev_b32 v22, 1, v20 // (op 4466) 

s_cmp_eq_i32 s18, 0 // (op 4466) 
s_cselect_b32 s19, 0, -1 // (op 4466) 

s_and_b32 s18, s15, 31 // (op 4466) 
s_lshl_b32 s49, 1, s18 // (op 4466) 
s_add_i32 s50, s49, s19 // (op 4466) 


s_and_b32 s19, s0, s50 // (op 4466) 


s_add_i32 s0, s1, s19 // (op 4466) 


s_ashr_i32 s1, s0, s18 // (op 4466) 


s_lshl_b32 s0, s15, 24 // (op 4466) 
s_ashr_i32 s18, s0, 31 // (op 4466) 

s_xor_b32 s0, s1, s18 // (op 4466) 

s_sub_i32 s1, s0, s18 // (op 4466) 


s_mul_hi_i32 s0, s1, s44 // (op 4466) 
s_add_i32 s18, s0, s1 // (op 4466) 

s_ashr_i32 s0, s18, 31 // (op 4466) 
s_cmp_eq_i32 s44, 0 // (op 4466) 
s_mov_b32 s19, scc // (op 4466) 
s_cmp_eq_i32 s19, 0 // (op 4466) 
s_cselect_b32 s49, 0, -1 // (op 4466) 

s_and_b32 s19, s45, 31 // (op 4466) 
s_lshl_b32 s50, 1, s19 // (op 4466) 
s_add_i32 s51, s50, s49 // (op 4466) 


s_and_b32 s49, s0, s51 // (op 4466) 


s_add_i32 s0, s18, s49 // (op 4466) 


s_ashr_i32 s18, s0, s19 // (op 4466) 


s_lshl_b32 s0, s45, 24 // (op 4466) 
s_ashr_i32 s19, s0, 31 // (op 4466) 

s_xor_b32 s0, s18, s19 // (op 4466) 

s_sub_i32 s18, s0, s19 // (op 4466) 


s_add_u32 s50, s8, 128 // (op 4466) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4466) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 4466) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 4466) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 4466) 

s_mul_i32 s0, s18, s50 // (op 4466) 

s_sub_i32 s49, s1, s0 // (op 4466) 


s_mul_hi_i32 s0, s16, s11 // (op 4466) 
s_add_i32 s1, s0, s16 // (op 4466) 

s_ashr_i32 s0, s1, 31 // (op 4466) 
s_cmp_eq_i32 s11, 0 // (op 4466) 
s_mov_b32 s18, scc // (op 4466) 
s_cmp_eq_i32 s18, 0 // (op 4466) 
s_cselect_b32 s19, 0, -1 // (op 4466) 

s_and_b32 s18, s47, 31 // (op 4466) 
s_lshl_b32 s68, 1, s18 // (op 4466) 
s_add_i32 s69, s68, s19 // (op 4466) 


s_and_b32 s19, s0, s69 // (op 4466) 


s_add_i32 s0, s1, s19 // (op 4466) 


s_ashr_i32 s1, s0, s18 // (op 4466) 


s_lshl_b32 s0, s47, 24 // (op 4466) 
s_ashr_i32 s18, s0, 31 // (op 4466) 

s_xor_b32 s0, s1, s18 // (op 4466) 

s_sub_i32 s1, s0, s18 // (op 4466) 


s_mul_i32 s0, s13, s50 // (op 4466) 

s_mul_i32 s18, s10, s0 // (op 4466) 

s_mul_i32 s0, s1, s18 // (op 4466) 

s_sub_u32 s18, s16, s0 // (op 4466) 

s_mul_hi_i32 s0, s18, s14 // (op 4466) 
s_add_i32 s19, s0, s18 // (op 4466) 


s_ashr_i32 s0, s19, 31 // (op 4466) 
s_cmp_eq_i32 s14, 0 // (op 4466) 
s_mov_b32 s18, scc // (op 4466) 
s_cmp_eq_i32 s18, 0 // (op 4466) 
s_cselect_b32 s50, 0, -1 // (op 4466) 

s_and_b32 s18, s46, 31 // (op 4466) 
s_lshl_b32 s51, 1, s18 // (op 4466) 
s_add_i32 s68, s51, s50 // (op 4466) 


s_and_b32 s50, s0, s68 // (op 4466) 


s_add_i32 s0, s19, s50 // (op 4466) 


s_ashr_i32 s19, s0, s18 // (op 4466) 


s_lshl_b32 s0, s46, 24 // (op 4466) 
s_ashr_i32 s18, s0, 31 // (op 4466) 

s_xor_b32 s0, s19, s18 // (op 4466) 

s_sub_i32 s19, s0, s18 // (op 4466) 


s_cmp_eq_i32 s1, 0 // (op 4466) 
s_mov_b32 s0, scc // (op 4466) 

s_cmp_eq_i32 s0, 0 // (op 4466) 
s_cselect_b32 s1, s19, s49 // (op 4466) 



v_lshl_add_u32 v20, s1, 7, v21 // (op 4466) 


v_mul_lo_u32 v21, v22, s40 // (op 4466) 
v_add_u32 v21, v21, v20 // (op 4466) 


s_mul_i32 s1, s40, s39 // (op 4468) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 4468) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 4468) least significant
s_mul_hi_u32 s19, s40, s38 // (op 4468) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4468) most significant: sum
s_add_u32 s1, s1, s19 // (op 4468) most significant: sum


s_mov_b32 s68, s42 // (op 4468) reg expression
s_mov_b32 s69, s43 // (op 4468) reg expression
s_mov_b32 s70, s0 // (op 4468) reg expression
s_mov_b32 s71, 131072 // (op 4468) call()

s_mul_hi_i32 s0, s16, s12 // (op 4485) 
v_and_b32 v20, -128, v1 // (op 4485) 
s_add_i32 s1, s0, s16 // (op 4485) 

v_sub_u32 v22, v1, v20 // (op 4485) 

s_ashr_i32 s0, s1, 31 // (op 4485) 
v_lshrrev_b32 v20, 7, v1 // (op 4485) 
s_cmp_eq_i32 s12, 0 // (op 4485) 
s_mov_b32 s18, scc // (op 4485) 
v_lshlrev_b32 v23, 1, v20 // (op 4485) 

s_cmp_eq_i32 s18, 0 // (op 4485) 
s_cselect_b32 s19, 0, -1 // (op 4485) 

s_and_b32 s18, s15, 31 // (op 4485) 
s_lshl_b32 s49, 1, s18 // (op 4485) 
s_add_i32 s50, s49, s19 // (op 4485) 


s_and_b32 s19, s0, s50 // (op 4485) 


s_add_i32 s0, s1, s19 // (op 4485) 


s_ashr_i32 s1, s0, s18 // (op 4485) 


s_lshl_b32 s0, s15, 24 // (op 4485) 
s_ashr_i32 s18, s0, 31 // (op 4485) 

s_xor_b32 s0, s1, s18 // (op 4485) 

s_sub_i32 s1, s0, s18 // (op 4485) 


s_mul_hi_i32 s0, s1, s44 // (op 4485) 
s_add_i32 s18, s0, s1 // (op 4485) 

s_ashr_i32 s0, s18, 31 // (op 4485) 
s_cmp_eq_i32 s44, 0 // (op 4485) 
s_mov_b32 s19, scc // (op 4485) 
s_cmp_eq_i32 s19, 0 // (op 4485) 
s_cselect_b32 s49, 0, -1 // (op 4485) 

s_and_b32 s19, s45, 31 // (op 4485) 
s_lshl_b32 s50, 1, s19 // (op 4485) 
s_add_i32 s51, s50, s49 // (op 4485) 


s_and_b32 s49, s0, s51 // (op 4485) 


s_add_i32 s0, s18, s49 // (op 4485) 


s_ashr_i32 s18, s0, s19 // (op 4485) 


s_lshl_b32 s0, s45, 24 // (op 4485) 
s_ashr_i32 s19, s0, 31 // (op 4485) 

s_xor_b32 s0, s18, s19 // (op 4485) 

s_sub_i32 s18, s0, s19 // (op 4485) 


s_add_u32 s50, s8, 128 // (op 4485) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4485) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4485) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4485) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 4485) 

s_mul_i32 s0, s18, s50 // (op 4485) 

s_sub_i32 s49, s1, s0 // (op 4485) 


s_mul_hi_i32 s0, s16, s11 // (op 4485) 
s_add_i32 s1, s0, s16 // (op 4485) 

s_ashr_i32 s0, s1, 31 // (op 4485) 
s_cmp_eq_i32 s11, 0 // (op 4485) 
s_mov_b32 s18, scc // (op 4485) 
s_cmp_eq_i32 s18, 0 // (op 4485) 
s_cselect_b32 s19, 0, -1 // (op 4485) 

s_and_b32 s18, s47, 31 // (op 4485) 
s_lshl_b32 s72, 1, s18 // (op 4485) 
s_add_i32 s73, s72, s19 // (op 4485) 


s_and_b32 s19, s0, s73 // (op 4485) 


s_add_i32 s0, s1, s19 // (op 4485) 


s_ashr_i32 s1, s0, s18 // (op 4485) 


s_lshl_b32 s0, s47, 24 // (op 4485) 
s_ashr_i32 s18, s0, 31 // (op 4485) 

s_xor_b32 s0, s1, s18 // (op 4485) 

s_sub_i32 s1, s0, s18 // (op 4485) 


s_mul_i32 s0, s13, s50 // (op 4485) 

s_mul_i32 s18, s10, s0 // (op 4485) 

s_mul_i32 s0, s1, s18 // (op 4485) 

s_sub_u32 s18, s16, s0 // (op 4485) 

s_mul_hi_i32 s0, s18, s14 // (op 4485) 
s_add_i32 s19, s0, s18 // (op 4485) 


s_ashr_i32 s0, s19, 31 // (op 4485) 
s_cmp_eq_i32 s14, 0 // (op 4485) 
s_mov_b32 s18, scc // (op 4485) 
s_cmp_eq_i32 s18, 0 // (op 4485) 
s_cselect_b32 s50, 0, -1 // (op 4485) 

s_and_b32 s18, s46, 31 // (op 4485) 
s_lshl_b32 s51, 1, s18 // (op 4485) 
s_add_i32 s72, s51, s50 // (op 4485) 


s_and_b32 s50, s0, s72 // (op 4485) 


s_add_i32 s0, s19, s50 // (op 4485) 


s_ashr_i32 s19, s0, s18 // (op 4485) 


s_lshl_b32 s0, s46, 24 // (op 4485) 
s_ashr_i32 s18, s0, 31 // (op 4485) 

s_xor_b32 s0, s19, s18 // (op 4485) 

s_sub_i32 s19, s0, s18 // (op 4485) 


s_cmp_eq_i32 s1, 0 // (op 4485) 
s_mov_b32 s0, scc // (op 4485) 

s_cmp_eq_i32 s0, 0 // (op 4485) 
s_cselect_b32 s1, s19, s49 // (op 4485) 



v_lshl_add_u32 v20, s1, 7, v22 // (op 4485) 


v_mul_lo_u32 v22, v23, s40 // (op 4485) 
v_add_u32 v22, v22, v20 // (op 4485) 


s_mul_i32 s1, s40, s39 // (op 4487) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 4487) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 4487) least significant
s_mul_hi_u32 s19, s40, s38 // (op 4487) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4487) most significant: sum
s_add_u32 s1, s1, s19 // (op 4487) most significant: sum


s_mov_b32 s68, s42 // (op 4487) reg expression
s_mov_b32 s69, s43 // (op 4487) reg expression
s_mov_b32 s70, s0 // (op 4487) reg expression
s_mov_b32 s71, 131072 // (op 4487) call()


s_lshl_b64 s[0:1], s[30:31], 2 // (op 4018) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 4018) 


v_mov_b32 v20, v18 // (op 4018) reg expression
buffer_load_dwordx4 v[26:29], v20, s[52:55], 0 offen // (op 4018) Load value
v_add_u32 v20, v20, v24 // (op 4018) 
buffer_load_dwordx4 v[30:33], v20, s[52:55], 0 offen // (op 4018) Load value



v_mov_b64 v[24:25], s[40:41] // (op 4021) convert

v_mov_b32 v20, v21 // (op 4021) reg expression
buffer_load_ubyte v34, v20, s[68:71], 0 offen offset: 0 // (op 4021) Load value
v_add_u32 v20, v20, v24 // (op 4021) 
buffer_load_ubyte v35, v20, s[68:71], 0 offen offset: 0 // (op 4021) Load value




v_mov_b32 v20, v16 // (op 4024) reg expression

buffer_load_dwordx3 v[36:38], v20, s[20:23], 0 offen // (op 4024) Load value
buffer_load_dwordx3 v[40:42], v20, s[20:23], 0 offen offset:12 // (op 4024) Load value



v_mov_b32 v20, v14 // (op 4027) reg expression

buffer_load_ubyte v39, v20, s[64:67], 0 offen offset: 0 // (op 4027) Load value


s_lshl_b64 s[0:1], s[30:31], 2 // (op 4030) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 4030) 


s_lshl_b64 s[0:1], s[30:31], 7 // (op 4030) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 4030) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 4030) 

v_add_u32 v20, v18, s0 // (op 4030) 

buffer_load_dwordx4 v[44:47], v20, s[52:55], 0 offen // (op 4030) Load value
v_add_u32 v20, v20, v24 // (op 4030) 
buffer_load_dwordx4 v[48:51], v20, s[52:55], 0 offen // (op 4030) Load value




v_mov_b64 v[24:25], s[40:41] // (op 4033) convert

s_lshl_b32 s0, s40, 2 // (op 4033) 
v_add_u32 v18, v21, s0 // (op 4033) 

buffer_load_ubyte v52, v18, s[68:71], 0 offen offset: 0 // (op 4033) Load value
v_add_u32 v18, v18, v24 // (op 4033) 
buffer_load_ubyte v53, v18, s[68:71], 0 offen offset: 0 // (op 4033) Load value





s_lshl_b64 s[0:1], s[24:25], 7 // (op 4036) 
s_mul_i32 s19, s1, 6 // (op 4036) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 4036) least significant
s_mul_hi_u32 s49, s0, 6 // (op 4036) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4036) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 4036) 

v_add_u32 v18, v16, s0 // (op 4036) 


buffer_load_dwordx3 v[54:56], v18, s[20:23], 0 offen // (op 4036) Load value
buffer_load_dwordx3 v[58:60], v18, s[20:23], 0 offen offset:12 // (op 4036) Load value




s_lshl_b32 s0, s34, 2 // (op 4039) 
v_add_u32 v16, v14, s0 // (op 4039) 


buffer_load_ubyte v18, v16, s[64:67], 0 offen offset: 0 // (op 4039) Load value


v_and_b32 v14, -4, v1 // (op 4615) 
v_sub_u32 v16, v1, v14 // (op 4615) 

v_lshlrev_b32 v14, 5, v16 // (op 4615) 

v_lshrrev_b32 v16, 2, v1 // (op 4615) 
v_lshl_add_u32 v43, v16, 8, v14 // (op 4615) 


v_lshlrev_b32 v14, 2, v43 // (op 4615) 

v_lshrrev_b32 v16, 3, v14 // (op 4615) 




s_waitcnt vmcnt(13)
ds_write_b128 v16, v[26:29] // (op 4041) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v16, v[30:33] offset:64 // (op 4041) Store local data 

v_and_b32 v14, -128, v1 // (op 4731) 
v_sub_u32 v16, v1, v14 // (op 4731) 

v_lshrrev_b32 v14, 7, v1 // (op 4731) 
v_lshl_add_u32 v43, v14, 8, v16 // (op 4731) 





s_waitcnt vmcnt(11)
ds_write_b8 v43, v34 offset:8192 // (op 4042) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v43, v35 offset:8320 // (op 4042) Store local data 

v_and_b32 v14, -2, v1 // (op 4557) 
v_sub_u32 v16, v1, v14 // (op 4557) 

v_lshlrev_b32 v14, 5, v16 // (op 4557) 

v_lshrrev_b32 v16, 1, v1 // (op 4557) 
v_lshl_add_u32 v43, v16, 6, v14 // (op 4557) 


v_mul_lo_u32 v14, v43, 6 // (op 4557) 
v_lshrrev_b32 v16, 3, v14 // (op 4557) 

v_lshrrev_b32 v14, 4, v43 // (op 4557) 

v_lshlrev_b32 v43, 2, v14 // (op 4557) 

v_add_u32 v14, v16, v43 // (op 4557) 





s_waitcnt vmcnt(9)
ds_write_b96 v14, v[36:38] offset:8704 // (op 4043) Store local data 
s_waitcnt vmcnt(8)
ds_write_b96 v14, v[40:42] offset:8720 // (op 4043) Store local data 

v_and_b32 v14, -64, v1 // (op 4673) 
v_sub_u32 v16, v1, v14 // (op 4673) 

v_and_b32 v14, 4294967232, v1 // (op 4673) 
v_add_u32 v43, v14, v16 // (op 4673) 





s_waitcnt vmcnt(7)
ds_write_b8 v43, v39 offset:16896 // (op 4044) Store local data 

v_and_b32 v14, -2, v1 // (op 4541) 
v_sub_u32 v16, v1, v14 // (op 4541) 

v_lshlrev_b32 v14, 5, v16 // (op 4541) 

v_lshrrev_b32 v16, 1, v1 // (op 4541) 
v_lshl_add_u32 v43, v16, 6, v14 // (op 4541) 


v_mul_lo_u32 v14, v43, 6 // (op 4541) 
v_lshrrev_b32 v16, 3, v14 // (op 4541) 

v_lshrrev_b32 v14, 4, v43 // (op 4541) 

v_lshlrev_b32 v43, 2, v14 // (op 4541) 

v_add_u32 v14, v16, v43 // (op 4541) 


v_and_b32 v16, -4, v1 // (op 4599) 
v_sub_u32 v43, v1, v16 // (op 4599) 

v_lshlrev_b32 v16, 5, v43 // (op 4599) 

v_lshrrev_b32 v43, 2, v1 // (op 4599) 
v_lshl_add_u32 v57, v43, 8, v16 // (op 4599) 


v_lshlrev_b32 v16, 2, v57 // (op 4599) 

v_lshrrev_b32 v43, 3, v16 // (op 4599) 

v_and_b32 v16, -64, v1 // (op 4657) 
v_sub_u32 v57, v1, v16 // (op 4657) 

v_and_b32 v16, 4294967232, v1 // (op 4657) 
v_add_u32 v20, v16, v57 // (op 4657) 


v_and_b32 v16, -128, v1 // (op 4715) 
v_sub_u32 v21, v1, v16 // (op 4715) 

v_lshrrev_b32 v16, 7, v1 // (op 4715) 
v_lshl_add_u32 v57, v16, 8, v21 // (op 4715) 


v_and_b32 v16, -2, v1 // (op 4783) 
v_sub_u32 v21, v1, v16 // (op 4783) 

v_lshlrev_b32 v16, 5, v21 // (op 4783) 

v_lshrrev_b32 v21, 1, v1 // (op 4783) 
v_lshl_add_u32 v23, v21, 6, v16 // (op 4783) 


v_mul_lo_u32 v16, v23, 6 // (op 4783) 
v_lshrrev_b32 v21, 3, v16 // (op 4783) 

v_lshrrev_b32 v16, 4, v23 // (op 4783) 

v_lshlrev_b32 v23, 2, v16 // (op 4783) 

v_add_u32 v16, v21, v23 // (op 4783) 


v_and_b32 v21, -4, v1 // (op 4812) 
v_sub_u32 v23, v1, v21 // (op 4812) 

v_lshlrev_b32 v21, 5, v23 // (op 4812) 

v_lshrrev_b32 v23, 2, v1 // (op 4812) 
v_lshl_add_u32 v24, v23, 8, v21 // (op 4812) 


v_lshlrev_b32 v21, 2, v24 // (op 4812) 

v_lshrrev_b32 v23, 3, v21 // (op 4812) 

v_and_b32 v21, -64, v1 // (op 4841) 
v_sub_u32 v24, v1, v21 // (op 4841) 

v_and_b32 v21, 4294967232, v1 // (op 4841) 
v_add_u32 v25, v21, v24 // (op 4841) 


v_and_b32 v21, -128, v1 // (op 4870) 
v_sub_u32 v24, v1, v21 // (op 4870) 

v_lshrrev_b32 v21, 7, v1 // (op 4870) 
v_lshl_add_u32 v61, v21, 8, v24 // (op 4870) 


s_mov_b32 s0, 0 // (op 1051) call()
s_ashr_i64 s[18:19], s[28:29], 7 // (op 1050) 
s_mov_b32 s1, s18 // (op 1050) convert

s_and_b32 s18, s1, 4294967294 // (op 1050) 

s_cmp_lt_u32 s0, s18 // (op 1050) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1050 // (op 1050) Condition: Top (jump to RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1050 if false)
RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1050:

s_barrier  // (op 4950) 
s_add_u32 s0, s0, 2 // (op 1052) 



s_lshl_b64 s[18:19], s[30:31], 2 // (op 10) 
v_lshrrev_b64 v[62:63], 3, s[18:19] // (op 10) 


v_mov_b32 v21, v19 // (op 10) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 10) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 10) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v24, v21, s1 // (op 10) 

buffer_load_dwordx4 v[26:29], v24, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v24, v24, v62 // (op 10) 
ds_read_b96_tr_b6 v[64:66], v0 offset:8704 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v0 offset:9728 // (op 72) Transpose load local data 
buffer_load_dwordx4 v[30:33], v24, s[52:55], 0 offen // (op 10) Load value



s_barrier  // (op 4963) 
v_mov_b64 v[62:63], s[40:41] // (op 25) convert




v_mov_b32 v24, v22 // (op 25) 
s_lshl_b32 s1, s40, 3 // (op 25) 
v_add_u32 v70, v24, s1 // (op 25) 

ds_read_b64_tr_b4 v[72:73], v4 // (op 77) Transpose load local data 
ds_read_b64_tr_b4 v[74:75], v4 offset:1024 // (op 77) Transpose load local data 
buffer_load_ubyte v34, v70, s[68:71], 0 offen offset: 0 // (op 25) Load value
s_barrier  // (op 4977) 
v_add_u32 v70, v70, v62 // (op 25) 


ds_read_u8 v71, v3 offset:16896 // (op 82) Load local data 
buffer_load_ubyte v35, v70, s[68:71], 0 offen offset: 0 // (op 25) Load value




s_barrier  // (op 4991) 
v_mov_b32 v70, v17 // (op 4) 


ds_read_u8 v62, v5 offset:8192 // (op 87) Load local data 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 4) 



s_mul_i32 s51, s19, 6 // (op 4) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 4) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v63, v70, s1 // (op 4) 


ds_read_b64_tr_b4 v[76:77], v4 offset:16 // (op 375) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v4 offset:1040 // (op 375) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[72:75], a[0:3], v71, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 90) 


ds_read_u8 v80, v5 offset:8224 // (op 379) Load local data 
buffer_load_dwordx3 v[36:38], v63, s[20:23], 0 offen // (op 4) Load value



ds_read_b64_tr_b4 v[82:83], v4 offset:32 // (op 551) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v4 offset:1056 // (op 551) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:69], v[76:79], a[4:7], v71, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 380) 


ds_read_u8 v81, v5 offset:8256 // (op 555) Load local data 
buffer_load_dwordx3 v[40:42], v63, s[20:23], 0 offen offset:12 // (op 4) Load value






v_mov_b32 v63, v15 // (op 17) 
s_lshl_b32 s1, s34, 3 // (op 17) 
v_add_u32 v86, v63, s1 // (op 17) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:69], v[82:85], a[8:11], v71, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 556) 
ds_read_b64_tr_b4 v[88:89], v4 offset:48 // (op 727) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v4 offset:1072 // (op 727) Transpose load local data 
buffer_load_ubyte v39, v86, s[64:67], 0 offen offset: 0 // (op 17) Load value



s_barrier  // (op 5014) 
ds_read_u8 v86, v5 offset:8288 // (op 731) Load local data 






ds_read_b96_tr_b6 v[92:94], v0 offset:8736 // (op 1084) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v0 offset:9760 // (op 1084) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:69], v[88:91], a[12:15], v71, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 732) 




ds_read_u8 v71, v3 offset:16928 // (op 1087) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v23, v[44:47] offset:17152 // (op 257) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:97], v[72:75], a[16:19], v71, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1227) 
s_waitcnt vmcnt(12)
ds_write_b128 v23, v[48:51] offset:17216 // (op 257) Store local data 


s_barrier  // (op 5026) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:97], v[76:79], a[20:23], v71, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1181) 





s_waitcnt vmcnt(11)
ds_write_b8 v61, v52 offset:25344 // (op 259) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:97], v[82:85], a[24:27], v71, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1135) 
s_waitcnt vmcnt(10)
ds_write_b8 v61, v53 offset:25472 // (op 259) Store local data 


s_barrier  // (op 5008) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:97], v[88:91], a[28:31], v71, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1089) 







s_waitcnt vmcnt(9)
ds_write_b96 v16, v[54:56] offset:25856 // (op 247) Store local data 
s_waitcnt vmcnt(8)
ds_write_b96 v16, v[58:60] offset:25872 // (op 247) Store local data 
s_barrier  // (op 5020) 



s_waitcnt vmcnt(7)
ds_write_b8 v25, v18 offset:34048 // (op 258) Store local data 

s_lshl_b64 s[18:19], s[30:31], 2 // (op 260) 
v_lshrrev_b64 v[64:65], 3, s[18:19] // (op 260) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 260) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 260) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 260) 

s_mul_i32 s1, 3, s18 // (op 260) 

v_add_u32 v62, v21, s1 // (op 260) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 5005) 
buffer_load_dwordx4 v[44:47], v62, s[52:55], 0 offen // (op 260) Load value



v_add_u32 v62, v62, v64 // (op 260) 
ds_read_b96_tr_b6 v[72:74], v6 offset:25856 // (op 249) Transpose load local data 
ds_read_b96_tr_b6 v[75:77], v6 offset:26880 // (op 249) Transpose load local data 
buffer_load_dwordx4 v[48:51], v62, s[52:55], 0 offen // (op 260) Load value



s_barrier  // (op 5011) 




v_mov_b64 v[64:65], s[40:41] // (op 262) convert

s_lshl_b32 s1, s40, 2 // (op 262) 
s_mul_i32 s18, 3, s1 // (op 262) 

v_add_u32 v21, v24, s18 // (op 262) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 4349) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 4349) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 4349) 

s_lshl_b32 s1, s18, 1 // (op 4349) 

v_add_u32 v19, v19, s1 // (op 4349) 

ds_read_b64_tr_b4 v[66:67], v8 offset:17152 // (op 251) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v8 offset:18176 // (op 251) Transpose load local data 
buffer_load_ubyte v52, v21, s[68:71], 0 offen offset: 0 // (op 262) Load value
v_add_u32 v21, v21, v64 // (op 262) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5017) 
buffer_load_ubyte v53, v21, s[68:71], 0 offen offset: 0 // (op 262) Load value





ds_read_u8 v21, v7 offset:34048 // (op 253) Load local data 


s_barrier  // (op 5023) 
s_lshl_b32 s1, s40, 3 // (op 4477) 


v_add_u32 v22, v22, s1 // (op 4477) 

ds_read_u8 v24, v9 offset:25344 // (op 255) Load local data 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 246) 



s_mul_i32 s51, s19, 6 // (op 246) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 246) least significant
s_mul_hi_u32 s1, s18, 6 // (op 246) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 246) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 246) 

s_mul_i32 s1, 3, s18 // (op 246) 

v_add_u32 v62, v70, s1 // (op 246) 


buffer_load_dwordx3 v[54:56], v62, s[20:23], 0 offen // (op 246) Load value
ds_read_b64_tr_b4 v[78:79], v8 offset:17168 // (op 385) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:77], v[66:69], a[0:3], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 256) 
ds_read_b64_tr_b4 v[80:81], v8 offset:18192 // (op 385) Transpose load local data 
buffer_load_dwordx3 v[58:60], v62, s[20:23], 0 offen offset:12 // (op 246) Load value




ds_read_u8 v62, v9 offset:25376 // (op 389) Load local data 





s_lshl_b32 s1, s34, 2 // (op 261) 
s_mul_i32 s18, 3, s1 // (op 261) 

v_add_u32 v64, v63, s18 // (op 261) 


s_lshl_b64 s[18:19], s[24:25], 7 // (op 4285) 
s_mul_i32 s51, s19, 6 // (op 4285) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 4285) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4285) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4285) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 4285) 

s_lshl_b32 s1, s18, 1 // (op 4285) 

v_add_u32 v17, v17, s1 // (op 4285) 

ds_read_b64_tr_b4 v[82:83], v8 offset:17184 // (op 561) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v8 offset:18208 // (op 561) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[78:81], a[4:7], v21, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 381) 


ds_read_u8 v65, v9 offset:25408 // (op 565) Load local data 
buffer_load_ubyte v18, v64, s[64:67], 0 offen offset: 0 // (op 261) Load value





ds_read_b64_tr_b4 v[86:87], v8 offset:17200 // (op 737) Transpose load local data 
s_barrier  // (op 4966) 
ds_read_b64_tr_b4 v[88:89], v8 offset:18224 // (op 737) Transpose load local data 



s_lshl_b32 s1, s34, 3 // (op 4413) 


s_barrier  // (op 5087) 
v_add_u32 v15, v15, s1 // (op 4413) 

ds_read_u8 v63, v9 offset:25440 // (op 741) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:77], v[82:85], a[8:11], v21, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 557) 
s_waitcnt vmcnt(13)
ds_write_b128 v43, v[26:29] // (op 13) Store local data 



s_barrier  // (op 5091) 
ds_read_b96_tr_b6 v[90:92], v6 offset:25888 // (op 1092) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[86:89], a[12:15], v21, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 733) 
ds_read_b96_tr_b6 v[93:95], v6 offset:26912 // (op 1092) Transpose load local data 


s_waitcnt vmcnt(12)
ds_write_b128 v43, v[30:33] offset:64 // (op 13) Store local data 


s_barrier  // (op 5115) 
s_barrier  // (op 4994) 
ds_read_u8 v21, v7 offset:34080 // (op 1095) Load local data 



s_waitcnt vmcnt(11)
ds_write_b8 v57, v34 offset:8192 // (op 30) Store local data 
s_barrier  // (op 5119) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:95], v[66:69], a[16:19], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1228) 
s_waitcnt vmcnt(10)
ds_write_b8 v57, v35 offset:8320 // (op 30) Store local data 


s_barrier  // (op 4953) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:95], v[78:81], a[20:23], v21, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1182) 





s_waitcnt vmcnt(9)
ds_write_b96 v14, v[36:38] offset:8704 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[82:85], a[24:27], v21, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1136) 
s_waitcnt vmcnt(8)
ds_write_b96 v14, v[40:42] offset:8720 // (op 6) Store local data 


s_barrier  // (op 4980) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[86:89], a[28:31], v21, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1090) 







s_waitcnt vmcnt(7)
ds_write_b8 v20, v39 offset:16896 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[28:29], 7 // (op 1050) 
s_mov_b32 s1, s18 // (op 1050) convert

s_and_b32 s18, s1, 4294967294 // (op 1050) 

s_cmp_lt_u32 s0, s18 // (op 1050) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1050 // (op 1050) Condition: Bottom (jump to RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1050 if true)
RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1050:

















s_mul_hi_i32 s0, s16, s12 // (op 4241) 
v_and_b32 v57, -2, v1 // (op 4241) 
s_add_i32 s1, s0, s16 // (op 4241) 

v_sub_u32 v14, v1, v57 // (op 4241) 

s_ashr_i32 s0, s1, 31 // (op 4241) 
v_lshlrev_b32 v57, 5, v14 // (op 4241) 

s_cmp_eq_i32 s12, 0 // (op 4241) 
s_mov_b32 s18, scc // (op 4241) 
v_lshrrev_b32 v14, 1, v1 // (op 4241) 
s_cmp_eq_i32 s18, 0 // (op 4241) 
s_cselect_b32 s19, 0, -1 // (op 4241) 

s_and_b32 s18, s15, 31 // (op 4241) 
s_lshl_b32 s49, 1, s18 // (op 4241) 
s_add_i32 s50, s49, s19 // (op 4241) 


s_and_b32 s19, s0, s50 // (op 4241) 


s_add_i32 s0, s1, s19 // (op 4241) 


s_ashr_i32 s1, s0, s18 // (op 4241) 


s_lshl_b32 s0, s15, 24 // (op 4241) 
s_ashr_i32 s18, s0, 31 // (op 4241) 

s_xor_b32 s0, s1, s18 // (op 4241) 

s_sub_i32 s1, s0, s18 // (op 4241) 


s_mul_hi_i32 s0, s1, s44 // (op 4241) 
s_add_i32 s18, s0, s1 // (op 4241) 

s_ashr_i32 s0, s18, 31 // (op 4241) 
s_cmp_eq_i32 s44, 0 // (op 4241) 
s_mov_b32 s19, scc // (op 4241) 
s_cmp_eq_i32 s19, 0 // (op 4241) 
s_cselect_b32 s49, 0, -1 // (op 4241) 

s_and_b32 s19, s45, 31 // (op 4241) 
s_lshl_b32 s50, 1, s19 // (op 4241) 
s_add_i32 s51, s50, s49 // (op 4241) 


s_and_b32 s49, s0, s51 // (op 4241) 


s_add_i32 s0, s18, s49 // (op 4241) 


s_ashr_i32 s18, s0, s19 // (op 4241) 


s_lshl_b32 s0, s45, 24 // (op 4241) 
s_ashr_i32 s19, s0, 31 // (op 4241) 

s_xor_b32 s0, s18, s19 // (op 4241) 

s_sub_i32 s18, s0, s19 // (op 4241) 


s_mul_i32 s0, s1, s13 // (op 4241) 

s_sub_u32 s1, s16, s0 // (op 4241) 

s_mul_i32 s0, s18, s13 // (op 4241) 
s_add_u32 s0, s0, s1 // (op 4241) 


s_mul_hi_i32 s1, s16, s11 // (op 4241) 
s_add_i32 s18, s1, s16 // (op 4241) 

s_ashr_i32 s1, s18, 31 // (op 4241) 
s_cmp_eq_i32 s11, 0 // (op 4241) 
s_mov_b32 s19, scc // (op 4241) 
s_cmp_eq_i32 s19, 0 // (op 4241) 
s_cselect_b32 s49, 0, -1 // (op 4241) 

s_and_b32 s19, s47, 31 // (op 4241) 
s_lshl_b32 s50, 1, s19 // (op 4241) 
s_add_i32 s51, s50, s49 // (op 4241) 


s_and_b32 s49, s1, s51 // (op 4241) 


s_add_i32 s1, s18, s49 // (op 4241) 


s_ashr_i32 s18, s1, s19 // (op 4241) 


s_lshl_b32 s1, s47, 24 // (op 4241) 
s_ashr_i32 s19, s1, 31 // (op 4241) 

s_xor_b32 s1, s18, s19 // (op 4241) 

s_sub_i32 s18, s1, s19 // (op 4241) 


s_add_u32 s50, s8, 128 // (op 4241) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4241) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4241) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4241) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 4241) 

s_mul_i32 s1, s13, s50 // (op 4241) 

s_mul_i32 s19, s10, s1 // (op 4241) 

s_mul_i32 s1, s18, s19 // (op 4241) 

s_sub_u32 s19, s16, s1 // (op 4241) 

s_mov_b32 s1, s19 // (op 4241) 
s_mul_hi_i32 s49, s1, s14 // (op 4241) 
s_add_i32 s50, s49, s1 // (op 4241) 


s_ashr_i32 s1, s50, 31 // (op 4241) 
s_cmp_eq_i32 s14, 0 // (op 4241) 
s_mov_b32 s49, scc // (op 4241) 
s_cmp_eq_i32 s49, 0 // (op 4241) 
s_cselect_b32 s51, 0, -1 // (op 4241) 

s_and_b32 s49, s46, 31 // (op 4241) 
s_lshl_b32 s72, 1, s49 // (op 4241) 
s_add_i32 s73, s72, s51 // (op 4241) 


s_and_b32 s51, s1, s73 // (op 4241) 


s_add_i32 s1, s50, s51 // (op 4241) 


s_ashr_i32 s50, s1, s49 // (op 4241) 


s_lshl_b32 s1, s46, 24 // (op 4241) 
s_ashr_i32 s49, s1, 31 // (op 4241) 

s_xor_b32 s1, s50, s49 // (op 4241) 

s_sub_i32 s50, s1, s49 // (op 4241) 


s_mul_i32 s1, s48, s13 // (op 4241) 
s_add_u32 s72, s6, 64 // (op 4241) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 4241) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 4241) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 4241) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 4241) 

s_sub_i32 s49, s72, s1 // (op 4241) 

s_mul_i32 s51, s50, s49 // (op 4241) 


s_sub_u32 s49, s19, s51 // (op 4241) 


s_add_u32 s19, s49, s1 // (op 4241) 


s_cmp_eq_i32 s18, 0 // (op 4241) 
s_mov_b32 s1, scc // (op 4241) 

s_cmp_eq_i32 s1, 0 // (op 4241) 
s_cselect_b32 s18, s19, s0 // (op 4241) 



v_lshl_add_u32 v15, s18, 6, v57 // (op 4241) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 4241) 
v_mov_b32 v17, 0 // (op 4241) convert
v_mov_b32 v16, v15 // (op 4241) convert

s_lshr_b32 s18, s0, 1 // (op 4241) 

v_lshl_add_u32 v15, s18, 8, v14 // (op 4241) 


v_mul_lo_u32 v43, v15, s25 // (op 4241) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 4241) most significant: high of low * low
v_mul_lo_u32 v42, v15, s24 // (op 4241) least significant: low of low * low
v_add_u32 v43, v43, v14 // (op 4241) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v42 // (op 4241) least significant half
v_addc_co_u32 v15, s[0:1], v17, v43, s[0:1] // (op 4241) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 4241) most significant: low of high * low
v_mul_hi_u32 v57, v14, 6 // (op 4241) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 4241) least significant: low of low * low
v_add_u32 v17, v17, v57 // (op 4241) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 4241) 

v_mov_b32 v57, v14 // (op 4241) convert

s_mul_i32 s1, s29, 6 // (op 4243) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 4243) least significant
s_mul_hi_u32 s18, s28, 6 // (op 4243) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4243) most significant: sum

s_mul_i32 s19, s24, s1 // (op 4243) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 4243) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 4243) least significant
s_mul_hi_u32 s50, s24, s0 // (op 4243) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4243) most significant: sum
s_add_u32 s19, s19, s50 // (op 4243) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4243) 

s_mov_b32 s20, s26 // (op 4243) reg expression
s_mov_b32 s21, s27 // (op 4243) reg expression
s_mov_b32 s22, s0 // (op 4243) reg expression
s_mov_b32 s23, 131072 // (op 4243) call()

s_mul_hi_i32 s0, s16, s12 // (op 4315) 
v_and_b32 v14, -4, v1 // (op 4315) 
s_add_i32 s1, s0, s16 // (op 4315) 

v_sub_u32 v15, v1, v14 // (op 4315) 

s_ashr_i32 s0, s1, 31 // (op 4315) 
v_lshlrev_b32 v14, 5, v15 // (op 4315) 

s_cmp_eq_i32 s12, 0 // (op 4315) 
s_mov_b32 s18, scc // (op 4315) 
v_lshrrev_b32 v15, 2, v1 // (op 4315) 
s_cmp_eq_i32 s18, 0 // (op 4315) 
s_cselect_b32 s19, 0, -1 // (op 4315) 

v_lshlrev_b32 v16, 1, v15 // (op 4315) 

s_and_b32 s18, s15, 31 // (op 4315) 
s_lshl_b32 s49, 1, s18 // (op 4315) 
s_add_i32 s50, s49, s19 // (op 4315) 


s_and_b32 s19, s0, s50 // (op 4315) 


s_add_i32 s0, s1, s19 // (op 4315) 


s_ashr_i32 s1, s0, s18 // (op 4315) 


s_lshl_b32 s0, s15, 24 // (op 4315) 
s_ashr_i32 s18, s0, 31 // (op 4315) 

s_xor_b32 s0, s1, s18 // (op 4315) 

s_sub_i32 s1, s0, s18 // (op 4315) 


s_mul_hi_i32 s0, s1, s44 // (op 4315) 
s_add_i32 s18, s0, s1 // (op 4315) 

s_ashr_i32 s0, s18, 31 // (op 4315) 
s_cmp_eq_i32 s44, 0 // (op 4315) 
s_mov_b32 s19, scc // (op 4315) 
s_cmp_eq_i32 s19, 0 // (op 4315) 
s_cselect_b32 s49, 0, -1 // (op 4315) 

s_and_b32 s19, s45, 31 // (op 4315) 
s_lshl_b32 s50, 1, s19 // (op 4315) 
s_add_i32 s51, s50, s49 // (op 4315) 


s_and_b32 s49, s0, s51 // (op 4315) 


s_add_i32 s0, s18, s49 // (op 4315) 


s_ashr_i32 s18, s0, s19 // (op 4315) 


s_lshl_b32 s0, s45, 24 // (op 4315) 
s_ashr_i32 s19, s0, 31 // (op 4315) 

s_xor_b32 s0, s18, s19 // (op 4315) 

s_sub_i32 s18, s0, s19 // (op 4315) 


s_add_u32 s50, s8, 128 // (op 4315) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4315) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4315) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4315) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 4315) 

s_mul_i32 s0, s18, s50 // (op 4315) 

s_sub_i32 s49, s1, s0 // (op 4315) 


s_mul_hi_i32 s0, s16, s11 // (op 4315) 
s_add_i32 s1, s0, s16 // (op 4315) 

s_ashr_i32 s0, s1, 31 // (op 4315) 
s_cmp_eq_i32 s11, 0 // (op 4315) 
s_mov_b32 s18, scc // (op 4315) 
s_cmp_eq_i32 s18, 0 // (op 4315) 
s_cselect_b32 s19, 0, -1 // (op 4315) 

s_and_b32 s18, s47, 31 // (op 4315) 
s_lshl_b32 s72, 1, s18 // (op 4315) 
s_add_i32 s73, s72, s19 // (op 4315) 


s_and_b32 s19, s0, s73 // (op 4315) 


s_add_i32 s0, s1, s19 // (op 4315) 


s_ashr_i32 s1, s0, s18 // (op 4315) 


s_lshl_b32 s0, s47, 24 // (op 4315) 
s_ashr_i32 s18, s0, 31 // (op 4315) 

s_xor_b32 s0, s1, s18 // (op 4315) 

s_sub_i32 s1, s0, s18 // (op 4315) 


s_mul_i32 s0, s13, s50 // (op 4315) 

s_mul_i32 s18, s10, s0 // (op 4315) 

s_mul_i32 s0, s1, s18 // (op 4315) 

s_sub_u32 s18, s16, s0 // (op 4315) 

s_mul_hi_i32 s0, s18, s14 // (op 4315) 
s_add_i32 s19, s0, s18 // (op 4315) 


s_ashr_i32 s0, s19, 31 // (op 4315) 
s_cmp_eq_i32 s14, 0 // (op 4315) 
s_mov_b32 s18, scc // (op 4315) 
s_cmp_eq_i32 s18, 0 // (op 4315) 
s_cselect_b32 s50, 0, -1 // (op 4315) 

s_and_b32 s18, s46, 31 // (op 4315) 
s_lshl_b32 s51, 1, s18 // (op 4315) 
s_add_i32 s72, s51, s50 // (op 4315) 


s_and_b32 s50, s0, s72 // (op 4315) 


s_add_i32 s0, s19, s50 // (op 4315) 


s_ashr_i32 s19, s0, s18 // (op 4315) 


s_lshl_b32 s0, s46, 24 // (op 4315) 
s_ashr_i32 s18, s0, 31 // (op 4315) 

s_xor_b32 s0, s19, s18 // (op 4315) 

s_sub_i32 s19, s0, s18 // (op 4315) 


s_cmp_eq_i32 s1, 0 // (op 4315) 
s_mov_b32 s0, scc // (op 4315) 

s_cmp_eq_i32 s0, 0 // (op 4315) 
s_cselect_b32 s1, s19, s49 // (op 4315) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 4315) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 4315) 
v_mov_b32 v43, 0 // (op 4315) convert
v_mov_b32 v42, v15 // (op 4315) convert

s_lshr_b32 s18, s0, 1 // (op 4315) 

v_lshl_add_u32 v17, s18, 8, v16 // (op 4315) 


v_mul_lo_u32 v15, v17, s31 // (op 4315) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 4315) most significant: high of low * low
v_mul_lo_u32 v14, v17, s30 // (op 4315) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 4315) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v42 // (op 4315) least significant half
v_addc_co_u32 v15, s[0:1], v15, v43, s[0:1] // (op 4315) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 4315) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 4315) 

v_mov_b32 v16, v14 // (op 4315) convert

s_mul_i32 s1, s30, s29 // (op 4317) most significant: low of low * high
s_mul_i32 s18, s31, s28 // (op 4317) most significant: low of high * low
s_mul_i32 s0, s30, s28 // (op 4317) least significant
s_mul_hi_u32 s19, s30, s28 // (op 4317) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4317) most significant: sum
s_add_u32 s1, s1, s19 // (op 4317) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4317) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 4317) 

s_mov_b32 s52, s32 // (op 4317) reg expression
s_mov_b32 s53, s33 // (op 4317) reg expression
s_mov_b32 s54, s0 // (op 4317) reg expression
s_mov_b32 s55, 131072 // (op 4317) call()

s_mul_hi_i32 s0, s16, s12 // (op 4379) 
v_and_b32 v17, -64, v1 // (op 4379) 
s_add_i32 s1, s0, s16 // (op 4379) 

v_sub_u32 v14, v1, v17 // (op 4379) 

s_ashr_i32 s0, s1, 31 // (op 4379) 
v_lshrrev_b32 v15, 6, v1 // (op 4379) 
s_cmp_eq_i32 s12, 0 // (op 4379) 
s_mov_b32 s18, scc // (op 4379) 
s_cmp_eq_i32 s18, 0 // (op 4379) 
s_cselect_b32 s19, 0, -1 // (op 4379) 

s_and_b32 s18, s15, 31 // (op 4379) 
s_lshl_b32 s49, 1, s18 // (op 4379) 
s_add_i32 s50, s49, s19 // (op 4379) 


s_and_b32 s19, s0, s50 // (op 4379) 


s_add_i32 s0, s1, s19 // (op 4379) 


s_ashr_i32 s1, s0, s18 // (op 4379) 


s_lshl_b32 s0, s15, 24 // (op 4379) 
s_ashr_i32 s18, s0, 31 // (op 4379) 

s_xor_b32 s0, s1, s18 // (op 4379) 

s_sub_i32 s1, s0, s18 // (op 4379) 


s_mul_hi_i32 s0, s1, s44 // (op 4379) 
s_add_i32 s18, s0, s1 // (op 4379) 

s_ashr_i32 s0, s18, 31 // (op 4379) 
s_cmp_eq_i32 s44, 0 // (op 4379) 
s_mov_b32 s19, scc // (op 4379) 
s_cmp_eq_i32 s19, 0 // (op 4379) 
s_cselect_b32 s49, 0, -1 // (op 4379) 

s_and_b32 s19, s45, 31 // (op 4379) 
s_lshl_b32 s50, 1, s19 // (op 4379) 
s_add_i32 s51, s50, s49 // (op 4379) 


s_and_b32 s49, s0, s51 // (op 4379) 


s_add_i32 s0, s18, s49 // (op 4379) 


s_ashr_i32 s18, s0, s19 // (op 4379) 


s_lshl_b32 s0, s45, 24 // (op 4379) 
s_ashr_i32 s19, s0, 31 // (op 4379) 

s_xor_b32 s0, s18, s19 // (op 4379) 

s_sub_i32 s18, s0, s19 // (op 4379) 


s_mul_i32 s0, s1, s13 // (op 4379) 

s_sub_u32 s1, s16, s0 // (op 4379) 

s_mul_i32 s0, s18, s13 // (op 4379) 
s_add_u32 s0, s0, s1 // (op 4379) 


s_mul_hi_i32 s1, s16, s11 // (op 4379) 
s_add_i32 s18, s1, s16 // (op 4379) 

s_ashr_i32 s1, s18, 31 // (op 4379) 
s_cmp_eq_i32 s11, 0 // (op 4379) 
s_mov_b32 s19, scc // (op 4379) 
s_cmp_eq_i32 s19, 0 // (op 4379) 
s_cselect_b32 s49, 0, -1 // (op 4379) 

s_and_b32 s19, s47, 31 // (op 4379) 
s_lshl_b32 s50, 1, s19 // (op 4379) 
s_add_i32 s51, s50, s49 // (op 4379) 


s_and_b32 s49, s1, s51 // (op 4379) 


s_add_i32 s1, s18, s49 // (op 4379) 


s_ashr_i32 s18, s1, s19 // (op 4379) 


s_lshl_b32 s1, s47, 24 // (op 4379) 
s_ashr_i32 s19, s1, 31 // (op 4379) 

s_xor_b32 s1, s18, s19 // (op 4379) 

s_sub_i32 s18, s1, s19 // (op 4379) 


s_add_u32 s50, s8, 128 // (op 4379) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4379) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4379) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4379) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 4379) 

s_mul_i32 s1, s13, s50 // (op 4379) 

s_mul_i32 s19, s10, s1 // (op 4379) 

s_mul_i32 s1, s18, s19 // (op 4379) 

s_sub_u32 s19, s16, s1 // (op 4379) 

s_mov_b32 s1, s19 // (op 4379) 
s_mul_hi_i32 s49, s1, s14 // (op 4379) 
s_add_i32 s50, s49, s1 // (op 4379) 


s_ashr_i32 s1, s50, 31 // (op 4379) 
s_cmp_eq_i32 s14, 0 // (op 4379) 
s_mov_b32 s49, scc // (op 4379) 
s_cmp_eq_i32 s49, 0 // (op 4379) 
s_cselect_b32 s51, 0, -1 // (op 4379) 

s_and_b32 s49, s46, 31 // (op 4379) 
s_lshl_b32 s72, 1, s49 // (op 4379) 
s_add_i32 s73, s72, s51 // (op 4379) 


s_and_b32 s51, s1, s73 // (op 4379) 


s_add_i32 s1, s50, s51 // (op 4379) 


s_ashr_i32 s50, s1, s49 // (op 4379) 


s_lshl_b32 s1, s46, 24 // (op 4379) 
s_ashr_i32 s49, s1, 31 // (op 4379) 

s_xor_b32 s1, s50, s49 // (op 4379) 

s_sub_i32 s50, s1, s49 // (op 4379) 


s_mul_i32 s1, s48, s13 // (op 4379) 
s_add_u32 s72, s6, 64 // (op 4379) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 4379) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 4379) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 4379) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 6 // (op 4379) 

s_sub_i32 s49, s72, s1 // (op 4379) 

s_mul_i32 s51, s50, s49 // (op 4379) 


s_sub_u32 s49, s19, s51 // (op 4379) 


s_add_u32 s19, s49, s1 // (op 4379) 


s_cmp_eq_i32 s18, 0 // (op 4379) 
s_mov_b32 s1, scc // (op 4379) 

s_cmp_eq_i32 s1, 0 // (op 4379) 
s_cselect_b32 s18, s19, s0 // (op 4379) 



v_lshl_add_u32 v17, s18, 6, v14 // (op 4379) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 4379) 
s_lshr_b32 s18, s0, 1 // (op 4379) 

v_lshl_add_u32 v14, s18, 3, v15 // (op 4379) 


v_mul_lo_u32 v15, v14, s34 // (op 4379) 

v_add_u32 v14, v17, v15 // (op 4379) 


s_mul_i32 s1, s34, s39 // (op 4381) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 4381) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 4381) least significant
s_mul_hi_u32 s19, s34, s38 // (op 4381) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4381) most significant: sum
s_add_u32 s1, s1, s19 // (op 4381) most significant: sum


s_mov_b32 s64, s36 // (op 4381) reg expression
s_mov_b32 s65, s37 // (op 4381) reg expression
s_mov_b32 s66, s0 // (op 4381) reg expression
s_mov_b32 s67, 131072 // (op 4381) call()

s_mul_hi_i32 s0, s16, s12 // (op 4443) 
v_and_b32 v15, -128, v1 // (op 4443) 
s_add_i32 s1, s0, s16 // (op 4443) 

v_sub_u32 v17, v1, v15 // (op 4443) 

s_ashr_i32 s0, s1, 31 // (op 4443) 
v_lshrrev_b32 v15, 7, v1 // (op 4443) 
s_cmp_eq_i32 s12, 0 // (op 4443) 
s_mov_b32 s18, scc // (op 4443) 
v_lshlrev_b32 v19, 1, v15 // (op 4443) 

s_cmp_eq_i32 s18, 0 // (op 4443) 
s_cselect_b32 s19, 0, -1 // (op 4443) 

s_and_b32 s18, s15, 31 // (op 4443) 
s_lshl_b32 s49, 1, s18 // (op 4443) 
s_add_i32 s50, s49, s19 // (op 4443) 


s_and_b32 s19, s0, s50 // (op 4443) 


s_add_i32 s0, s1, s19 // (op 4443) 


s_ashr_i32 s1, s0, s18 // (op 4443) 


s_lshl_b32 s0, s15, 24 // (op 4443) 
s_ashr_i32 s18, s0, 31 // (op 4443) 

s_xor_b32 s0, s1, s18 // (op 4443) 

s_sub_i32 s1, s0, s18 // (op 4443) 


s_mul_hi_i32 s0, s1, s44 // (op 4443) 
s_add_i32 s18, s0, s1 // (op 4443) 

s_ashr_i32 s0, s18, 31 // (op 4443) 
s_cmp_eq_i32 s44, 0 // (op 4443) 
s_mov_b32 s19, scc // (op 4443) 
s_cmp_eq_i32 s19, 0 // (op 4443) 
s_cselect_b32 s49, 0, -1 // (op 4443) 

s_and_b32 s19, s45, 31 // (op 4443) 
s_lshl_b32 s50, 1, s19 // (op 4443) 
s_add_i32 s51, s50, s49 // (op 4443) 


s_and_b32 s49, s0, s51 // (op 4443) 


s_add_i32 s0, s18, s49 // (op 4443) 


s_ashr_i32 s18, s0, s19 // (op 4443) 


s_lshl_b32 s0, s45, 24 // (op 4443) 
s_ashr_i32 s19, s0, 31 // (op 4443) 

s_xor_b32 s0, s18, s19 // (op 4443) 

s_sub_i32 s18, s0, s19 // (op 4443) 


s_add_u32 s50, s8, 128 // (op 4443) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 4443) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4443) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4443) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 4443) 

s_mul_i32 s0, s18, s50 // (op 4443) 

s_sub_i32 s49, s1, s0 // (op 4443) 


s_mul_hi_i32 s0, s16, s11 // (op 4443) 
s_add_i32 s1, s0, s16 // (op 4443) 

s_ashr_i32 s0, s1, 31 // (op 4443) 
s_cmp_eq_i32 s11, 0 // (op 4443) 
s_mov_b32 s18, scc // (op 4443) 
s_cmp_eq_i32 s18, 0 // (op 4443) 
s_cselect_b32 s19, 0, -1 // (op 4443) 

s_and_b32 s18, s47, 31 // (op 4443) 
s_lshl_b32 s72, 1, s18 // (op 4443) 
s_add_i32 s73, s72, s19 // (op 4443) 


s_and_b32 s19, s0, s73 // (op 4443) 


s_add_i32 s0, s1, s19 // (op 4443) 


s_ashr_i32 s1, s0, s18 // (op 4443) 


s_lshl_b32 s0, s47, 24 // (op 4443) 
s_ashr_i32 s18, s0, 31 // (op 4443) 

s_xor_b32 s0, s1, s18 // (op 4443) 

s_sub_i32 s1, s0, s18 // (op 4443) 


s_mul_i32 s0, s13, s50 // (op 4443) 

s_mul_i32 s18, s10, s0 // (op 4443) 

s_mul_i32 s0, s1, s18 // (op 4443) 

s_sub_u32 s18, s16, s0 // (op 4443) 

s_mul_hi_i32 s0, s18, s14 // (op 4443) 
s_add_i32 s19, s0, s18 // (op 4443) 


s_ashr_i32 s0, s19, 31 // (op 4443) 
s_cmp_eq_i32 s14, 0 // (op 4443) 
s_mov_b32 s18, scc // (op 4443) 
s_cmp_eq_i32 s18, 0 // (op 4443) 
s_cselect_b32 s50, 0, -1 // (op 4443) 

s_and_b32 s18, s46, 31 // (op 4443) 
s_lshl_b32 s51, 1, s18 // (op 4443) 
s_add_i32 s72, s51, s50 // (op 4443) 


s_and_b32 s50, s0, s72 // (op 4443) 


s_add_i32 s0, s19, s50 // (op 4443) 


s_ashr_i32 s19, s0, s18 // (op 4443) 


s_lshl_b32 s0, s46, 24 // (op 4443) 
s_ashr_i32 s18, s0, 31 // (op 4443) 

s_xor_b32 s0, s19, s18 // (op 4443) 

s_sub_i32 s19, s0, s18 // (op 4443) 


s_cmp_eq_i32 s1, 0 // (op 4443) 
s_mov_b32 s0, scc // (op 4443) 

s_cmp_eq_i32 s0, 0 // (op 4443) 
s_cselect_b32 s1, s19, s49 // (op 4443) 



v_lshl_add_u32 v15, s1, 7, v17 // (op 4443) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 4443) 
s_lshr_b32 s18, s0, 1 // (op 4443) 

v_lshl_add_u32 v17, s18, 3, v19 // (op 4443) 


v_mul_lo_u32 v19, v17, s40 // (op 4443) 
v_add_u32 v19, v19, v15 // (op 4443) 


s_mul_i32 s1, s40, s39 // (op 4445) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 4445) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 4445) least significant
s_mul_hi_u32 s19, s40, s38 // (op 4445) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4445) most significant: sum
s_add_u32 s1, s1, s19 // (op 4445) most significant: sum


s_mov_b32 s68, s42 // (op 4445) reg expression
s_mov_b32 s69, s43 // (op 4445) reg expression
s_mov_b32 s70, s0 // (op 4445) reg expression
s_mov_b32 s71, 131072 // (op 4445) call()

v_and_b32 v15, -2, v1 // (op 4526) 
v_sub_u32 v17, v1, v15 // (op 4526) 

v_lshlrev_b32 v15, 5, v17 // (op 4526) 

v_lshrrev_b32 v17, 1, v1 // (op 4526) 
v_lshl_add_u32 v20, v17, 6, v15 // (op 4526) 


v_mul_lo_u32 v15, v20, 6 // (op 4526) 
v_lshrrev_b32 v17, 3, v15 // (op 4526) 

v_lshrrev_b32 v15, 4, v20 // (op 4526) 

v_lshlrev_b32 v20, 2, v15 // (op 4526) 

v_add_u32 v15, v17, v20 // (op 4526) 


v_and_b32 v17, -4, v1 // (op 4586) 
v_sub_u32 v20, v1, v17 // (op 4586) 

v_lshlrev_b32 v17, 5, v20 // (op 4586) 

v_lshrrev_b32 v20, 2, v1 // (op 4586) 
v_lshl_add_u32 v21, v20, 8, v17 // (op 4586) 


v_lshlrev_b32 v17, 2, v21 // (op 4586) 

v_lshrrev_b32 v20, 3, v17 // (op 4586) 

v_and_b32 v17, -64, v1 // (op 4644) 
v_sub_u32 v21, v1, v17 // (op 4644) 

v_and_b32 v17, 4294967232, v1 // (op 4644) 
v_add_u32 v22, v17, v21 // (op 4644) 


v_and_b32 v17, -128, v1 // (op 4702) 
v_sub_u32 v21, v1, v17 // (op 4702) 

v_lshrrev_b32 v17, 7, v1 // (op 4702) 
v_lshl_add_u32 v23, v17, 8, v21 // (op 4702) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 1025) 
s_and_b32 s18, s0, 4294967294 // (op 1025) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 1024) 
s_mov_b32 s19, s0 // (op 1024) convert

s_cmp_lt_u32 s18, s19 // (op 1024) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1024 // (op 1024) Condition: Top (jump to RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1024 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1024:


s_add_u32 s18, s18, 1 // (op 1026) 
v_mov_b32 v21, v57 // (op 180) 
v_mov_b32 v17, v21 // (op 180) reg expression

buffer_load_dwordx3 v[24:26], v17, s[20:23], 0 offen // (op 180) Load value
buffer_load_dwordx3 v[28:30], v17, s[20:23], 0 offen offset:12 // (op 180) Load value


s_lshl_b64 s[0:1], s[24:25], 7 // (op 4231) 
s_mul_i32 s51, s1, 6 // (op 4231) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 4231) least significant
s_mul_hi_u32 s19, s0, 6 // (op 4231) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 4231) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 4231) 

v_add_u32 v57, v57, s0 // (op 4231) 


s_barrier  // (op 4956) 
s_lshl_b64 s[0:1], s[30:31], 2 // (op 194) 



v_lshrrev_b64 v[32:33], 3, s[0:1] // (op 194) 


v_mov_b32 v21, v16 // (op 194) 
v_mov_b32 v27, v21 // (op 194) reg expression
buffer_load_dwordx4 v[34:37], v27, s[52:55], 0 offen // (op 194) Load value
v_add_u32 v27, v27, v32 // (op 194) 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[24:26] offset:8704 // (op 181) Store local data 
buffer_load_dwordx4 v[38:41], v27, s[52:55], 0 offen // (op 194) Load value



s_lshl_b64 s[0:1], s[30:31], 7 // (op 4307) 
s_lshl_b64 s[50:51], s[0:1], 2 // (op 4307) 

s_lshr_b64 s[0:1], s[50:51], 3 // (op 4307) 

v_add_u32 v16, v16, s0 // (op 4307) 

s_waitcnt vmcnt(2)
ds_write_b96 v15, v[28:30] offset:8720 // (op 181) Store local data 



v_mov_b32 v21, v14 // (op 195) 
s_barrier  // (op 4969) 
v_mov_b32 v17, v21 // (op 195) reg expression




buffer_load_ubyte v18, v17, s[64:67], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s19, s34, 2 // (op 4371) 
v_add_u32 v14, v14, s19 // (op 4371) 

s_waitcnt vmcnt(2)
ds_write_b128 v20, v[34:37] // (op 191) Store local data 

v_mov_b64 v[24:25], s[40:41] // (op 196) convert

v_mov_b32 v17, v19 // (op 196) 
v_mov_b32 v21, v17 // (op 196) reg expression
buffer_load_ubyte v26, v21, s[68:71], 0 offen offset: 0 // (op 196) Load value
v_add_u32 v21, v21, v24 // (op 196) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[38:41] offset:64 // (op 191) Store local data 
buffer_load_ubyte v27, v21, s[68:71], 0 offen offset: 0 // (op 196) Load value




s_lshl_b32 s19, s40, 2 // (op 4435) 
s_barrier  // (op 4983) 
v_add_u32 v19, v19, s19 // (op 4435) 




s_waitcnt lgkmcnt(2)
s_barrier  // (op 4959) 
s_waitcnt vmcnt(2)
ds_write_b8 v22, v18 offset:16896 // (op 192) Store local data 




ds_read_b96_tr_b6 v[28:30], v0 offset:8704 // (op 183) Transpose load local data 
s_barrier  // (op 4997) 
ds_read_b96_tr_b6 v[31:33], v0 offset:9728 // (op 183) Transpose load local data 



s_waitcnt lgkmcnt(3)
s_barrier  // (op 4973) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v26 offset:8192 // (op 193) Store local data 



ds_read_b64_tr_b4 v[34:35], v4 // (op 185) Transpose load local data 
ds_read_b64_tr_b4 v[36:37], v4 offset:1024 // (op 185) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v27 offset:8320 // (op 193) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 4987) 


ds_read_u8 v21, v3 offset:16896 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 5001) 


ds_read_u8 v17, v5 offset:8192 // (op 189) Load local data 



ds_read_b64_tr_b4 v[24:25], v4 offset:16 // (op 354) Transpose load local data 
ds_read_b64_tr_b4 v[26:27], v4 offset:1040 // (op 354) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:33], v[34:37], a[0:3], v21, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 190) 


ds_read_u8 v18, v5 offset:8224 // (op 358) Load local data 



ds_read_b64_tr_b4 v[38:39], v4 offset:32 // (op 530) Transpose load local data 
ds_read_b64_tr_b4 v[40:41], v4 offset:1056 // (op 530) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:33], v[24:27], a[4:7], v21, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 359) 


ds_read_u8 v42, v5 offset:8256 // (op 534) Load local data 



ds_read_b64_tr_b4 v[44:45], v4 offset:48 // (op 706) Transpose load local data 
ds_read_b64_tr_b4 v[46:47], v4 offset:1072 // (op 706) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:33], v[38:41], a[8:11], v21, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 535) 


ds_read_u8 v43, v5 offset:8288 // (op 710) Load local data 



ds_read_b96_tr_b6 v[48:50], v0 offset:8736 // (op 968) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v0 offset:9760 // (op 968) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[28:33], v[44:47], a[12:15], v21, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 711) 




ds_read_u8 v21, v3 offset:16928 // (op 971) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:53], v[34:37], a[16:19], v21, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 973) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:53], v[24:27], a[20:23], v21, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1004) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:53], v[38:41], a[24:27], v21, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1035) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:53], v[44:47], a[28:31], v21, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b100 // (op 1066) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 1024) 
s_mov_b32 s19, s0 // (op 1024) convert

s_cmp_lt_u32 s18, s19 // (op 1024) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1024 // (op 1024) Condition: Bottom (jump to RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1024 if true)
RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1024:























s_barrier  // (op 5099) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 
v_cvt_f16_f32 v8, v16 // (op 56) 
v_cvt_f16_f32 v9, v17 // (op 56) 


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

v_cvt_f16_f32 v6, v14 // (op 1875) 
v_cvt_f16_f32 v7, v15 // (op 1875) 
v_cvt_f16_f32 v8, v16 // (op 1875) 
v_cvt_f16_f32 v9, v17 // (op 1875) 


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

v_cvt_f16_f32 v6, v14 // (op 1889) 
v_cvt_f16_f32 v7, v15 // (op 1889) 
v_cvt_f16_f32 v8, v16 // (op 1889) 
v_cvt_f16_f32 v9, v17 // (op 1889) 


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

v_cvt_f16_f32 v6, v14 // (op 1903) 
v_cvt_f16_f32 v7, v15 // (op 1903) 
v_cvt_f16_f32 v8, v16 // (op 1903) 
v_cvt_f16_f32 v9, v17 // (op 1903) 


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

v_cvt_f16_f32 v6, v14 // (op 1939) 
v_cvt_f16_f32 v7, v15 // (op 1939) 
v_cvt_f16_f32 v8, v16 // (op 1939) 
v_cvt_f16_f32 v9, v17 // (op 1939) 


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

v_cvt_f16_f32 v6, v14 // (op 1953) 
v_cvt_f16_f32 v7, v15 // (op 1953) 
v_cvt_f16_f32 v8, v16 // (op 1953) 
v_cvt_f16_f32 v9, v17 // (op 1953) 


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

v_cvt_f16_f32 v6, v14 // (op 1959) 
v_cvt_f16_f32 v7, v15 // (op 1959) 
v_cvt_f16_f32 v8, v16 // (op 1959) 
v_cvt_f16_f32 v9, v17 // (op 1959) 


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

v_cvt_f16_f32 v6, v14 // (op 1965) 
v_cvt_f16_f32 v7, v15 // (op 1965) 
v_cvt_f16_f32 v8, v16 // (op 1965) 
v_cvt_f16_f32 v9, v17 // (op 1965) 


v_pack_b32_f16 v4, v6, v7 // (op 1947) 
v_pack_b32_f16 v5, v8, v9 // (op 1947) 

s_lshl_b32 s0, s4, 6 // (op 1947) 
s_mul_i32 s1, 3, s0 // (op 1947) 
s_add_u32 s1, s1, 64 // (op 1947) 

v_add_u32 v0, v13, s1 // (op 1947) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1947) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135:


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
v_cvt_f32_f16 v6, v8 // (op 2132) 
v_lshrrev_b32 v7, 16, v8 // (op 2132) 
v_cvt_f32_f16 v7, v7 // (op 2132) 
v_mul_f32 v18, v10, v6 // (op 2132) 
v_mul_f32 v19, v10, v7 // (op 2132) 
v_cvt_f32_f16 v6, v9 // (op 2132) 
v_lshrrev_b32 v7, 16, v9 // (op 2132) 
v_cvt_f32_f16 v7, v7 // (op 2132) 
v_mul_f32 v20, v10, v6 // (op 2132) 
v_mul_f32 v21, v10, v7 // (op 2132) 


v_add_f32 v6, v18, v14 // (op 2121) 
v_add_f32 v7, v19, v15 // (op 2121) 
v_add_f32 v8, v20, v16 // (op 2121) 
v_add_f32 v9, v21, v17 // (op 2121) 


v_cvt_f16_f32 v14, v6 // (op 2122) 
v_cvt_f16_f32 v15, v7 // (op 2122) 
v_cvt_f16_f32 v16, v8 // (op 2122) 
v_cvt_f16_f32 v17, v9 // (op 2122) 


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
v_cvt_f32_f16 v6, v8 // (op 2135) 
v_lshrrev_b32 v7, 16, v8 // (op 2135) 
v_cvt_f32_f16 v7, v7 // (op 2135) 
v_mul_f32 v18, v10, v6 // (op 2135) 
v_mul_f32 v19, v10, v7 // (op 2135) 
v_cvt_f32_f16 v6, v9 // (op 2135) 
v_lshrrev_b32 v7, 16, v9 // (op 2135) 
v_cvt_f32_f16 v7, v7 // (op 2135) 
v_mul_f32 v20, v10, v6 // (op 2135) 
v_mul_f32 v21, v10, v7 // (op 2135) 


v_add_f32 v6, v18, v14 // (op 2136) 
v_add_f32 v7, v19, v15 // (op 2136) 
v_add_f32 v8, v20, v16 // (op 2136) 
v_add_f32 v9, v21, v17 // (op 2136) 


v_cvt_f16_f32 v14, v6 // (op 2137) 
v_cvt_f16_f32 v15, v7 // (op 2137) 
v_cvt_f16_f32 v16, v8 // (op 2137) 
v_cvt_f16_f32 v17, v9 // (op 2137) 


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
v_cvt_f32_f16 v6, v8 // (op 2140) 
v_lshrrev_b32 v7, 16, v8 // (op 2140) 
v_cvt_f32_f16 v7, v7 // (op 2140) 
v_mul_f32 v18, v10, v6 // (op 2140) 
v_mul_f32 v19, v10, v7 // (op 2140) 
v_cvt_f32_f16 v6, v9 // (op 2140) 
v_lshrrev_b32 v7, 16, v9 // (op 2140) 
v_cvt_f32_f16 v7, v7 // (op 2140) 
v_mul_f32 v20, v10, v6 // (op 2140) 
v_mul_f32 v21, v10, v7 // (op 2140) 


v_add_f32 v6, v18, v14 // (op 2141) 
v_add_f32 v7, v19, v15 // (op 2141) 
v_add_f32 v8, v20, v16 // (op 2141) 
v_add_f32 v9, v21, v17 // (op 2141) 


v_cvt_f16_f32 v14, v6 // (op 2142) 
v_cvt_f16_f32 v15, v7 // (op 2142) 
v_cvt_f16_f32 v16, v8 // (op 2142) 
v_cvt_f16_f32 v17, v9 // (op 2142) 


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
v_cvt_f32_f16 v6, v8 // (op 2145) 
v_lshrrev_b32 v7, 16, v8 // (op 2145) 
v_cvt_f32_f16 v7, v7 // (op 2145) 
v_mul_f32 v18, v10, v6 // (op 2145) 
v_mul_f32 v19, v10, v7 // (op 2145) 
v_cvt_f32_f16 v6, v9 // (op 2145) 
v_lshrrev_b32 v7, 16, v9 // (op 2145) 
v_cvt_f32_f16 v7, v7 // (op 2145) 
v_mul_f32 v20, v10, v6 // (op 2145) 
v_mul_f32 v21, v10, v7 // (op 2145) 


v_add_f32 v6, v18, v14 // (op 2146) 
v_add_f32 v7, v19, v15 // (op 2146) 
v_add_f32 v8, v20, v16 // (op 2146) 
v_add_f32 v9, v21, v17 // (op 2146) 


v_cvt_f16_f32 v14, v6 // (op 2147) 
v_cvt_f16_f32 v15, v7 // (op 2147) 
v_cvt_f16_f32 v16, v8 // (op 2147) 
v_cvt_f16_f32 v17, v9 // (op 2147) 


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
v_cvt_f32_f16 v6, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_cvt_f32_f16 v6, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_cvt_f16_f32 v14, v6 // (op 146) 
v_cvt_f16_f32 v15, v7 // (op 146) 
v_cvt_f16_f32 v16, v8 // (op 146) 
v_cvt_f16_f32 v17, v9 // (op 146) 


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
v_cvt_f32_f16 v6, v8 // (op 2042) 
v_lshrrev_b32 v7, 16, v8 // (op 2042) 
v_cvt_f32_f16 v7, v7 // (op 2042) 
v_mul_f32 v18, v10, v6 // (op 2042) 
v_mul_f32 v19, v10, v7 // (op 2042) 
v_cvt_f32_f16 v6, v9 // (op 2042) 
v_lshrrev_b32 v7, 16, v9 // (op 2042) 
v_cvt_f32_f16 v7, v7 // (op 2042) 
v_mul_f32 v20, v10, v6 // (op 2042) 
v_mul_f32 v21, v10, v7 // (op 2042) 


v_add_f32 v6, v18, v14 // (op 2038) 
v_add_f32 v7, v19, v15 // (op 2038) 
v_add_f32 v8, v20, v16 // (op 2038) 
v_add_f32 v9, v21, v17 // (op 2038) 


v_cvt_f16_f32 v14, v6 // (op 2039) 
v_cvt_f16_f32 v15, v7 // (op 2039) 
v_cvt_f16_f32 v16, v8 // (op 2039) 
v_cvt_f16_f32 v17, v9 // (op 2039) 


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
v_cvt_f32_f16 v6, v8 // (op 2056) 
v_lshrrev_b32 v7, 16, v8 // (op 2056) 
v_cvt_f32_f16 v7, v7 // (op 2056) 
v_mul_f32 v18, v10, v6 // (op 2056) 
v_mul_f32 v19, v10, v7 // (op 2056) 
v_cvt_f32_f16 v6, v9 // (op 2056) 
v_lshrrev_b32 v7, 16, v9 // (op 2056) 
v_cvt_f32_f16 v7, v7 // (op 2056) 
v_mul_f32 v20, v10, v6 // (op 2056) 
v_mul_f32 v21, v10, v7 // (op 2056) 


v_add_f32 v6, v18, v14 // (op 2052) 
v_add_f32 v7, v19, v15 // (op 2052) 
v_add_f32 v8, v20, v16 // (op 2052) 
v_add_f32 v9, v21, v17 // (op 2052) 


v_cvt_f16_f32 v14, v6 // (op 2053) 
v_cvt_f16_f32 v15, v7 // (op 2053) 
v_cvt_f16_f32 v16, v8 // (op 2053) 
v_cvt_f16_f32 v17, v9 // (op 2053) 


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
v_cvt_f32_f16 v6, v8 // (op 2070) 
v_lshrrev_b32 v7, 16, v8 // (op 2070) 
v_cvt_f32_f16 v7, v7 // (op 2070) 
v_mul_f32 v18, v10, v6 // (op 2070) 
v_mul_f32 v19, v10, v7 // (op 2070) 
v_cvt_f32_f16 v6, v9 // (op 2070) 
v_lshrrev_b32 v7, 16, v9 // (op 2070) 
v_cvt_f32_f16 v7, v7 // (op 2070) 
v_mul_f32 v20, v10, v6 // (op 2070) 
v_mul_f32 v21, v10, v7 // (op 2070) 


v_add_f32 v6, v18, v14 // (op 2066) 
v_add_f32 v7, v19, v15 // (op 2066) 
v_add_f32 v8, v20, v16 // (op 2066) 
v_add_f32 v9, v21, v17 // (op 2066) 


v_cvt_f16_f32 v14, v6 // (op 2067) 
v_cvt_f16_f32 v15, v7 // (op 2067) 
v_cvt_f16_f32 v16, v8 // (op 2067) 
v_cvt_f16_f32 v17, v9 // (op 2067) 


v_pack_b32_f16 v8, v14, v15 // (op 2068) 
v_pack_b32_f16 v9, v16, v17 // (op 2068) 

s_lshl_b32 s0, s4, 6 // (op 2068) 
s_mul_i32 s1, 3, s0 // (op 2068) 

v_add_u32 v0, v13, s1 // (op 2068) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 2068) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC
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
    DestDataType: h
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
  - .name: RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_NT_FP6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__LDSSYNC.kd
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

