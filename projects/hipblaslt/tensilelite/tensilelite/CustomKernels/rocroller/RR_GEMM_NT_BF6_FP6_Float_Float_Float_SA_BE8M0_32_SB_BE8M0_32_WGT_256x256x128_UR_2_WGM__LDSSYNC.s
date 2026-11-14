// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s12 // (op 35662) 
s_mul_hi_i32 s1, s16, s12 // (op 35919) 
v_and_b32 v0, 63, v1 // (op 35724) 
v_and_b32 v3, 48, v1 // (op 35724) 
v_sub_u32 v4, v0, v3 // (op 35724) 

v_lshrrev_b32 v3, 4, v0 // (op 35724) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 35724) 


v_lshrrev_b32 v3, 7, v1 // (op 35724) 
v_lshlrev_b32 v4, 4, v3 // (op 35724) 

v_lshl_add_u32 v3, v0, 8, v4 // (op 35724) 


v_mul_lo_u32 v0, v3, 6 // (op 35724) 
v_lshrrev_b32 v4, 3, v0 // (op 35724) 

v_lshrrev_b32 v0, 4, v3 // (op 35724) 

v_lshlrev_b32 v3, 2, v0 // (op 35724) 

v_add_u32 v0, v4, v3 // (op 35724) 


v_and_b32 v3, 63, v1 // (op 35782) 
v_and_b32 v4, 48, v1 // (op 35782) 
v_sub_u32 v5, v3, v4 // (op 35782) 

v_lshrrev_b32 v4, 4, v3 // (op 35782) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 35782) 


v_lshrrev_b32 v4, 6, v1 // (op 35782) 
v_and_b32 v5, 1, v4 // (op 35782) 

v_lshlrev_b32 v4, 4, v5 // (op 35782) 

v_lshl_add_u32 v5, v3, 8, v4 // (op 35782) 


v_mul_lo_u32 v3, v5, 6 // (op 35782) 
v_lshrrev_b32 v4, 3, v3 // (op 35782) 

v_lshrrev_b32 v3, 4, v5 // (op 35782) 

v_lshlrev_b32 v5, 2, v3 // (op 35782) 

v_add_u32 v3, v4, v5 // (op 35782) 


v_lshrrev_b32 v4, 7, v1 // (op 35840) 
v_and_b32 v5, 15, v1 // (op 35840) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 35840) 


v_and_b32 v4, 63, v1 // (op 35840) 
v_lshrrev_b32 v5, 4, v4 // (op 35840) 

v_lshl_add_u32 v4, v5, 8, v6 // (op 35840) 


v_lshrrev_b32 v5, 6, v1 // (op 35898) 
v_and_b32 v6, 1, v5 // (op 35898) 

v_and_b32 v5, 15, v1 // (op 35898) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 35898) 


v_and_b32 v5, 63, v1 // (op 35898) 
v_lshrrev_b32 v6, 4, v5 // (op 35898) 

v_lshl_add_u32 v5, v6, 8, v7 // (op 35898) 


v_and_b32 v6, 63, v1 // (op 35950) 
v_and_b32 v7, 48, v1 // (op 35950) 
v_sub_u32 v8, v6, v7 // (op 35950) 

v_lshrrev_b32 v7, 4, v6 // (op 35950) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 35950) 


v_lshrrev_b32 v7, 7, v1 // (op 35950) 
v_lshlrev_b32 v8, 4, v7 // (op 35950) 

v_lshl_add_u32 v7, v6, 8, v8 // (op 35950) 


v_mul_lo_u32 v6, v7, 6 // (op 35950) 
v_lshrrev_b32 v8, 3, v6 // (op 35950) 

v_lshrrev_b32 v6, 4, v7 // (op 35950) 

v_lshlrev_b32 v7, 2, v6 // (op 35950) 

v_add_u32 v6, v8, v7 // (op 35950) 


v_and_b32 v7, 63, v1 // (op 35979) 
v_and_b32 v8, 48, v1 // (op 35979) 
v_sub_u32 v9, v7, v8 // (op 35979) 

v_lshrrev_b32 v8, 4, v7 // (op 35979) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 35979) 


v_lshrrev_b32 v8, 6, v1 // (op 35979) 
v_and_b32 v9, 1, v8 // (op 35979) 

v_lshlrev_b32 v8, 4, v9 // (op 35979) 

v_lshl_add_u32 v9, v7, 8, v8 // (op 35979) 


v_mul_lo_u32 v7, v9, 6 // (op 35979) 
v_lshrrev_b32 v8, 3, v7 // (op 35979) 

v_lshrrev_b32 v7, 4, v9 // (op 35979) 

v_lshlrev_b32 v9, 2, v7 // (op 35979) 

v_add_u32 v7, v8, v9 // (op 35979) 


v_lshrrev_b32 v8, 7, v1 // (op 36008) 
v_and_b32 v9, 15, v1 // (op 36008) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 36008) 


v_and_b32 v8, 63, v1 // (op 36008) 
v_lshrrev_b32 v9, 4, v8 // (op 36008) 

v_lshl_add_u32 v8, v9, 8, v10 // (op 36008) 


v_lshrrev_b32 v9, 6, v1 // (op 36037) 
v_and_b32 v10, 1, v9 // (op 36037) 

v_and_b32 v9, 15, v1 // (op 36037) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 36037) 


v_and_b32 v9, 63, v1 // (op 36037) 
v_lshrrev_b32 v10, 4, v9 // (op 36037) 

v_lshl_add_u32 v9, v10, 8, v11 // (op 36037) 


v_lshrrev_b32 v10, 7, v1 // (op 35662) 
s_add_i32 s18, s0, s16 // (op 35662) 

v_and_b32 v11, 63, v1 // (op 35662) 
s_ashr_i32 s0, s18, 31 // (op 35662) 
v_lshrrev_b32 v12, 4, v11 // (op 35662) 
s_cmp_eq_i32 s12, 0 // (op 35662) 
s_mov_b32 s19, scc // (op 35662) 
v_lshlrev_b32 v13, 2, v12 // (op 35662) 

s_cmp_eq_i32 s19, 0 // (op 35662) 
s_cselect_b32 s51, 0, -1 // (op 35662) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 35662) 


s_and_b32 s19, s15, 31 // (op 35662) 
v_lshrrev_b32 v10, 2, v11 // (op 35662) 

v_and_b32 v11, 3, v10 // (op 35662) 

v_lshrrev_b32 v10, 7, v1 // (op 35919) 
v_and_b32 v13, 63, v1 // (op 35919) 
s_lshl_b32 s52, 1, s19 // (op 35662) 
v_and_b32 v14, 3, v1 // (op 35662) 
s_add_i32 s53, s52, s51 // (op 35662) 


s_and_b32 s51, s0, s53 // (op 35662) 


s_add_i32 s0, s1, s16 // (op 35919) 

s_ashr_i32 s1, s0, 31 // (op 35919) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 35662) 


s_add_i32 s52, s18, s51 // (op 35662) 


v_lshrrev_b32 v11, 6, v1 // (op 35662) 
s_ashr_i32 s18, s52, s19 // (op 35662) 


v_and_b32 v14, 1, v11 // (op 35662) 

s_lshl_b32 s19, s15, 24 // (op 35662) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 35662) 


s_ashr_i32 s51, s19, 31 // (op 35662) 

s_xor_b32 s19, s18, s51 // (op 35662) 

s_sub_i32 s18, s19, s51 // (op 35662) 


v_lshrrev_b32 v14, 4, v13 // (op 35919) 
s_cmp_eq_i32 s12, 0 // (op 35919) 
s_mov_b32 s19, scc // (op 35919) 
v_lshlrev_b32 v15, 2, v14 // (op 35919) 

s_cmp_eq_i32 s19, 0 // (op 35919) 
s_cselect_b32 s51, 0, -1 // (op 35919) 

v_lshl_add_u32 v14, v10, 4, v15 // (op 35919) 


s_and_b32 s19, s15, 31 // (op 35919) 
v_lshrrev_b32 v10, 2, v13 // (op 35919) 

s_lshl_b32 s52, 1, s19 // (op 35919) 
v_and_b32 v13, 3, v10 // (op 35919) 

s_add_i32 s53, s52, s51 // (op 35919) 


v_and_b32 v10, 3, v1 // (op 35919) 
s_and_b32 s51, s1, s53 // (op 35919) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 35919) 


s_add_i32 s1, s0, s51 // (op 35919) 


v_lshrrev_b32 v10, 6, v1 // (op 35919) 
s_ashr_i32 s0, s1, s19 // (op 35919) 


v_and_b32 v13, 1, v10 // (op 35919) 

s_lshl_b32 s1, s15, 24 // (op 35919) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 35919) 


s_ashr_i32 s19, s1, 31 // (op 35919) 

s_xor_b32 s1, s0, s19 // (op 35919) 

s_sub_i32 s0, s1, s19 // (op 35919) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s1, s18, s44 // (op 35662) 
s_add_i32 s19, s1, s18 // (op 35662) 

s_ashr_i32 s1, s19, 31 // (op 35662) 
s_cmp_eq_i32 s44, 0 // (op 35662) 
s_mov_b32 s51, scc // (op 35662) 
s_cmp_eq_i32 s51, 0 // (op 35662) 
s_cselect_b32 s52, 0, -1 // (op 35662) 

s_and_b32 s51, s45, 31 // (op 35662) 
s_lshl_b32 s53, 1, s51 // (op 35662) 
s_add_i32 s54, s53, s52 // (op 35662) 


s_and_b32 s52, s1, s54 // (op 35662) 


s_add_i32 s1, s19, s52 // (op 35662) 


s_ashr_i32 s19, s1, s51 // (op 35662) 


s_lshl_b32 s1, s45, 24 // (op 35662) 
s_ashr_i32 s51, s1, 31 // (op 35662) 

s_xor_b32 s1, s19, s51 // (op 35662) 

s_sub_i32 s19, s1, s51 // (op 35662) 


s_mul_i32 s1, s18, s13 // (op 35662) 
s_sub_u32 s51, s16, s1 // (op 35662) 

s_mul_i32 s1, s19, s13 // (op 35662) 
s_add_u32 s1, s1, s51 // (op 35662) 

s_mul_hi_i32 s51, s16, s11 // (op 35662) 
s_add_i32 s52, s51, s16 // (op 35662) 

s_ashr_i32 s51, s52, 31 // (op 35662) 
s_cmp_eq_i32 s11, 0 // (op 35662) 
s_mov_b32 s53, scc // (op 35662) 
s_cmp_eq_i32 s53, 0 // (op 35662) 
s_cselect_b32 s54, 0, -1 // (op 35662) 

s_and_b32 s53, s47, 31 // (op 35662) 
s_lshl_b32 s55, 1, s53 // (op 35662) 
s_add_i32 s56, s55, s54 // (op 35662) 


s_and_b32 s54, s51, s56 // (op 35662) 


s_add_i32 s51, s52, s54 // (op 35662) 


s_ashr_i32 s52, s51, s53 // (op 35662) 


s_lshl_b32 s51, s47, 24 // (op 35662) 
s_ashr_i32 s53, s51, 31 // (op 35662) 

s_xor_b32 s51, s52, s53 // (op 35662) 

s_sub_i32 s52, s51, s53 // (op 35662) 


s_mul_hi_i32 s51, s0, s44 // (op 35919) 
s_add_i32 s53, s51, s0 // (op 35919) 

s_ashr_i32 s51, s53, 31 // (op 35919) 
s_cmp_eq_i32 s44, 0 // (op 35919) 
s_mov_b32 s54, scc // (op 35919) 
s_cmp_eq_i32 s54, 0 // (op 35919) 
s_cselect_b32 s55, 0, -1 // (op 35919) 

s_and_b32 s54, s45, 31 // (op 35919) 
s_lshl_b32 s56, 1, s54 // (op 35919) 
s_add_i32 s57, s56, s55 // (op 35919) 


s_and_b32 s55, s51, s57 // (op 35919) 


s_add_i32 s51, s53, s55 // (op 35919) 


s_ashr_i32 s53, s51, s54 // (op 35919) 


s_lshl_b32 s51, s45, 24 // (op 35919) 
s_ashr_i32 s54, s51, 31 // (op 35919) 

s_xor_b32 s51, s53, s54 // (op 35919) 

s_sub_i32 s53, s51, s54 // (op 35919) 


s_mul_i32 s51, s0, s13 // (op 35919) 
s_sub_u32 s54, s16, s51 // (op 35919) 

s_mul_i32 s51, s53, s13 // (op 35919) 
s_add_u32 s51, s51, s54 // (op 35919) 

s_mul_hi_i32 s54, s16, s11 // (op 35919) 
s_add_i32 s55, s54, s16 // (op 35919) 

s_ashr_i32 s54, s55, 31 // (op 35919) 
s_cmp_eq_i32 s11, 0 // (op 35919) 
s_mov_b32 s56, scc // (op 35919) 
s_cmp_eq_i32 s56, 0 // (op 35919) 
s_cselect_b32 s57, 0, -1 // (op 35919) 

s_and_b32 s56, s47, 31 // (op 35919) 
s_lshl_b32 s58, 1, s56 // (op 35919) 
s_add_i32 s59, s58, s57 // (op 35919) 


s_and_b32 s57, s54, s59 // (op 35919) 


s_add_i32 s54, s55, s57 // (op 35919) 


s_ashr_i32 s55, s54, s56 // (op 35919) 


s_lshl_b32 s54, s47, 24 // (op 35919) 
s_ashr_i32 s56, s54, 31 // (op 35919) 

s_xor_b32 s54, s55, s56 // (op 35919) 

s_sub_i32 s55, s54, s56 // (op 35919) 


s_add_u32 s56, s8, 256 // (op 35662) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 35662) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 35662) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 35662) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 35662) 

s_mul_i32 s54, s13, s56 // (op 35662) 
s_mul_i32 s58, s10, s54 // (op 35662) 

s_mul_i32 s54, s52, s58 // (op 35662) 

s_sub_u32 s58, s16, s54 // (op 35662) 

s_mov_b32 s54, s58 // (op 35662) 
s_mul_hi_i32 s59, s54, s14 // (op 35662) 
s_add_i32 s60, s59, s54 // (op 35662) 


s_ashr_i32 s54, s60, 31 // (op 35662) 
s_cmp_eq_i32 s14, 0 // (op 35662) 
s_mov_b32 s59, scc // (op 35662) 
s_cmp_eq_i32 s59, 0 // (op 35662) 
s_cselect_b32 s61, 0, -1 // (op 35662) 

s_and_b32 s59, s46, 31 // (op 35662) 
s_lshl_b32 s62, 1, s59 // (op 35662) 
s_add_i32 s63, s62, s61 // (op 35662) 


s_and_b32 s61, s54, s63 // (op 35662) 


s_add_i32 s54, s60, s61 // (op 35662) 


s_ashr_i32 s60, s54, s59 // (op 35662) 


s_lshl_b32 s54, s46, 24 // (op 35662) 
s_ashr_i32 s59, s54, 31 // (op 35662) 

s_xor_b32 s54, s60, s59 // (op 35662) 

s_sub_i32 s60, s54, s59 // (op 35662) 


s_mul_i32 s54, s48, s13 // (op 35662) 
s_add_u32 s62, s6, 256 // (op 35662) Start of Int64 add, locking SCC
s_addc_u32 s63, s7, 0 // (op 35662) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 35662) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 35662) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 8 // (op 35662) 

s_sub_i32 s59, s62, s54 // (op 35662) 

s_mul_i32 s61, s60, s59 // (op 35662) 

s_sub_u32 s59, s58, s61 // (op 35662) 


s_add_u32 s58, s59, s54 // (op 35662) 


s_cmp_eq_i32 s52, 0 // (op 35662) 
s_mov_b32 s54, scc // (op 35662) 

s_cmp_eq_i32 s54, 0 // (op 35662) 
s_cselect_b32 s52, s58, s1 // (op 35662) 


s_mul_i32 s1, s19, s56 // (op 35662) 


v_lshl_add_u32 v13, s52, 8, v12 // (op 35662) 


s_sub_i32 s19, s18, s1 // (op 35662) 


s_cmp_eq_i32 s54, 0 // (op 35662) 
s_cselect_b32 s1, s60, s19 // (op 35662) 



v_lshl_add_u32 v12, s1, 8, v11 // (op 35662) 


v_mul_lo_u32 v11, v12, s2 // (op 35662) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 35662) 


s_add_u32 s18, s8, 256 // (op 35919) Start of Int64 add, locking SCC
s_addc_u32 s19, s9, 0 // (op 35919) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 35919) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 35919) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 8 // (op 35919) 

s_mul_i32 s1, s13, s18 // (op 35919) 
s_mul_i32 s52, s10, s1 // (op 35919) 

s_mul_i32 s1, s55, s52 // (op 35919) 

s_sub_u32 s52, s16, s1 // (op 35919) 

s_mov_b32 s1, s52 // (op 35919) 
s_mul_hi_i32 s54, s1, s14 // (op 35919) 
s_add_i32 s56, s54, s1 // (op 35919) 


s_ashr_i32 s1, s56, 31 // (op 35919) 
s_cmp_eq_i32 s14, 0 // (op 35919) 
s_mov_b32 s54, scc // (op 35919) 
s_cmp_eq_i32 s54, 0 // (op 35919) 
s_cselect_b32 s57, 0, -1 // (op 35919) 

s_and_b32 s54, s46, 31 // (op 35919) 
s_lshl_b32 s58, 1, s54 // (op 35919) 
s_add_i32 s59, s58, s57 // (op 35919) 


s_and_b32 s57, s1, s59 // (op 35919) 


s_add_i32 s1, s56, s57 // (op 35919) 


s_ashr_i32 s56, s1, s54 // (op 35919) 


s_lshl_b32 s1, s46, 24 // (op 35919) 
s_ashr_i32 s54, s1, 31 // (op 35919) 

s_xor_b32 s1, s56, s54 // (op 35919) 

s_sub_i32 s56, s1, s54 // (op 35919) 


s_mul_i32 s1, s48, s13 // (op 35919) 
s_add_u32 s58, s6, 256 // (op 35919) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 35919) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 35919) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 35919) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 8 // (op 35919) 

s_sub_i32 s54, s58, s1 // (op 35919) 

s_mul_i32 s57, s56, s54 // (op 35919) 

s_sub_u32 s54, s52, s57 // (op 35919) 


s_add_u32 s52, s54, s1 // (op 35919) 


s_cmp_eq_i32 s55, 0 // (op 35919) 
s_mov_b32 s1, scc // (op 35919) 

s_cmp_eq_i32 s1, 0 // (op 35919) 
s_cselect_b32 s54, s52, s51 // (op 35919) 


s_mul_i32 s55, s53, s18 // (op 35919) 


v_lshl_add_u32 v11, s54, 8, v14 // (op 35919) 


s_sub_i32 s18, s0, s55 // (op 35919) 


s_cmp_eq_i32 s1, 0 // (op 35919) 
s_cselect_b32 s0, s56, s18 // (op 35919) 



v_lshl_add_u32 v13, s0, 8, v10 // (op 35919) 


v_mul_lo_u32 v10, v13, s4 // (op 35919) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 35919) 


s_mul_i32 s1, s2, s9 // (op 35664) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 35664) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 35664) least significant
s_mul_hi_u32 s19, s2, s8 // (op 35664) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35664) most significant: sum
s_add_u32 s1, s1, s19 // (op 35664) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 35664) 

s_mul_i32 s1, s4, s9 // (op 35921) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 35921) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 35921) least significant
s_mul_hi_u32 s52, s4, s8 // (op 35921) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 35921) most significant: sum
s_add_u32 s1, s1, s52 // (op 35921) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 35921) 

s_mov_b32 s56, s20 // (op 35664) reg expression
s_mov_b32 s57, s21 // (op 35664) reg expression
s_mov_b32 s58, s18 // (op 35664) reg expression
s_mov_b32 s59, 131072 // (op 35664) call()

s_mov_b32 s60, s22 // (op 35921) reg expression

s_mov_b32 s61, s23 // (op 35921) reg expression
s_mov_b32 s62, s52 // (op 35921) reg expression
s_mov_b32 s63, 131072 // (op 35921) call()


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
v_accvgpr_write a8, 0.00000 // (op 512) call()
v_accvgpr_write a9, 0.00000 // (op 512) call()
v_accvgpr_write a10, 0.00000 // (op 512) call()
v_accvgpr_write a11, 0.00000 // (op 512) call()
v_accvgpr_write a12, 0.00000 // (op 688) call()
v_accvgpr_write a13, 0.00000 // (op 688) call()
v_accvgpr_write a14, 0.00000 // (op 688) call()
v_accvgpr_write a15, 0.00000 // (op 688) call()
v_accvgpr_write a16, 0.00000 // (op 864) call()
v_accvgpr_write a17, 0.00000 // (op 864) call()
v_accvgpr_write a18, 0.00000 // (op 864) call()
v_accvgpr_write a19, 0.00000 // (op 864) call()
v_accvgpr_write a20, 0.00000 // (op 1040) call()
v_accvgpr_write a21, 0.00000 // (op 1040) call()
v_accvgpr_write a22, 0.00000 // (op 1040) call()
v_accvgpr_write a23, 0.00000 // (op 1040) call()
v_accvgpr_write a24, 0.00000 // (op 1216) call()
v_accvgpr_write a25, 0.00000 // (op 1216) call()
v_accvgpr_write a26, 0.00000 // (op 1216) call()
v_accvgpr_write a27, 0.00000 // (op 1216) call()
v_accvgpr_write a28, 0.00000 // (op 1392) call()
v_accvgpr_write a29, 0.00000 // (op 1392) call()
v_accvgpr_write a30, 0.00000 // (op 1392) call()
v_accvgpr_write a31, 0.00000 // (op 1392) call()
v_accvgpr_write a32, 0.00000 // (op 1744) call()
v_accvgpr_write a33, 0.00000 // (op 1744) call()
v_accvgpr_write a34, 0.00000 // (op 1744) call()
v_accvgpr_write a35, 0.00000 // (op 1744) call()
v_accvgpr_write a36, 0.00000 // (op 2361) call()
v_accvgpr_write a37, 0.00000 // (op 2361) call()
v_accvgpr_write a38, 0.00000 // (op 2361) call()
v_accvgpr_write a39, 0.00000 // (op 2361) call()
v_accvgpr_write a40, 0.00000 // (op 2362) call()
v_accvgpr_write a41, 0.00000 // (op 2362) call()
v_accvgpr_write a42, 0.00000 // (op 2362) call()
v_accvgpr_write a43, 0.00000 // (op 2362) call()
v_accvgpr_write a44, 0.00000 // (op 2363) call()
v_accvgpr_write a45, 0.00000 // (op 2363) call()
v_accvgpr_write a46, 0.00000 // (op 2363) call()
v_accvgpr_write a47, 0.00000 // (op 2363) call()
v_accvgpr_write a48, 0.00000 // (op 2364) call()
v_accvgpr_write a49, 0.00000 // (op 2364) call()
v_accvgpr_write a50, 0.00000 // (op 2364) call()
v_accvgpr_write a51, 0.00000 // (op 2364) call()
v_accvgpr_write a52, 0.00000 // (op 2365) call()
v_accvgpr_write a53, 0.00000 // (op 2365) call()
v_accvgpr_write a54, 0.00000 // (op 2365) call()
v_accvgpr_write a55, 0.00000 // (op 2365) call()
v_accvgpr_write a56, 0.00000 // (op 2366) call()
v_accvgpr_write a57, 0.00000 // (op 2366) call()
v_accvgpr_write a58, 0.00000 // (op 2366) call()
v_accvgpr_write a59, 0.00000 // (op 2366) call()
v_accvgpr_write a60, 0.00000 // (op 2367) call()
v_accvgpr_write a61, 0.00000 // (op 2367) call()
v_accvgpr_write a62, 0.00000 // (op 2367) call()
v_accvgpr_write a63, 0.00000 // (op 2367) call()
v_accvgpr_write a64, 0.00000 // (op 3282) call()
v_accvgpr_write a65, 0.00000 // (op 3282) call()
v_accvgpr_write a66, 0.00000 // (op 3282) call()
v_accvgpr_write a67, 0.00000 // (op 3282) call()
v_accvgpr_write a68, 0.00000 // (op 3899) call()
v_accvgpr_write a69, 0.00000 // (op 3899) call()
v_accvgpr_write a70, 0.00000 // (op 3899) call()
v_accvgpr_write a71, 0.00000 // (op 3899) call()
v_accvgpr_write a72, 0.00000 // (op 3900) call()
v_accvgpr_write a73, 0.00000 // (op 3900) call()
v_accvgpr_write a74, 0.00000 // (op 3900) call()
v_accvgpr_write a75, 0.00000 // (op 3900) call()
v_accvgpr_write a76, 0.00000 // (op 3901) call()
v_accvgpr_write a77, 0.00000 // (op 3901) call()
v_accvgpr_write a78, 0.00000 // (op 3901) call()
v_accvgpr_write a79, 0.00000 // (op 3901) call()
v_accvgpr_write a80, 0.00000 // (op 3902) call()
v_accvgpr_write a81, 0.00000 // (op 3902) call()
v_accvgpr_write a82, 0.00000 // (op 3902) call()
v_accvgpr_write a83, 0.00000 // (op 3902) call()
v_accvgpr_write a84, 0.00000 // (op 3903) call()
v_accvgpr_write a85, 0.00000 // (op 3903) call()
v_accvgpr_write a86, 0.00000 // (op 3903) call()
v_accvgpr_write a87, 0.00000 // (op 3903) call()
v_accvgpr_write a88, 0.00000 // (op 3904) call()
v_accvgpr_write a89, 0.00000 // (op 3904) call()
v_accvgpr_write a90, 0.00000 // (op 3904) call()
v_accvgpr_write a91, 0.00000 // (op 3904) call()
v_accvgpr_write a92, 0.00000 // (op 3905) call()
v_accvgpr_write a93, 0.00000 // (op 3905) call()
v_accvgpr_write a94, 0.00000 // (op 3905) call()
v_accvgpr_write a95, 0.00000 // (op 3905) call()
v_accvgpr_write a96, 0.00000 // (op 4820) call()
v_accvgpr_write a97, 0.00000 // (op 4820) call()
v_accvgpr_write a98, 0.00000 // (op 4820) call()
v_accvgpr_write a99, 0.00000 // (op 4820) call()
v_accvgpr_write a100, 0.00000 // (op 5437) call()
v_accvgpr_write a101, 0.00000 // (op 5437) call()
v_accvgpr_write a102, 0.00000 // (op 5437) call()
v_accvgpr_write a103, 0.00000 // (op 5437) call()
v_accvgpr_write a104, 0.00000 // (op 5438) call()
v_accvgpr_write a105, 0.00000 // (op 5438) call()
v_accvgpr_write a106, 0.00000 // (op 5438) call()
v_accvgpr_write a107, 0.00000 // (op 5438) call()
v_accvgpr_write a108, 0.00000 // (op 5439) call()
v_accvgpr_write a109, 0.00000 // (op 5439) call()
v_accvgpr_write a110, 0.00000 // (op 5439) call()
v_accvgpr_write a111, 0.00000 // (op 5439) call()
v_accvgpr_write a112, 0.00000 // (op 5440) call()
v_accvgpr_write a113, 0.00000 // (op 5440) call()
v_accvgpr_write a114, 0.00000 // (op 5440) call()
v_accvgpr_write a115, 0.00000 // (op 5440) call()
v_accvgpr_write a116, 0.00000 // (op 5441) call()
v_accvgpr_write a117, 0.00000 // (op 5441) call()
v_accvgpr_write a118, 0.00000 // (op 5441) call()
v_accvgpr_write a119, 0.00000 // (op 5441) call()
v_accvgpr_write a120, 0.00000 // (op 5442) call()
v_accvgpr_write a121, 0.00000 // (op 5442) call()
v_accvgpr_write a122, 0.00000 // (op 5442) call()
v_accvgpr_write a123, 0.00000 // (op 5442) call()
v_accvgpr_write a124, 0.00000 // (op 5443) call()
v_accvgpr_write a125, 0.00000 // (op 5443) call()
v_accvgpr_write a126, 0.00000 // (op 5443) call()
v_accvgpr_write a127, 0.00000 // (op 5443) call()
v_accvgpr_write a128, 0.00000 // (op 6358) call()
v_accvgpr_write a129, 0.00000 // (op 6358) call()
v_accvgpr_write a130, 0.00000 // (op 6358) call()
v_accvgpr_write a131, 0.00000 // (op 6358) call()
v_accvgpr_write a132, 0.00000 // (op 6975) call()
v_accvgpr_write a133, 0.00000 // (op 6975) call()
v_accvgpr_write a134, 0.00000 // (op 6975) call()
v_accvgpr_write a135, 0.00000 // (op 6975) call()
v_accvgpr_write a136, 0.00000 // (op 6976) call()
v_accvgpr_write a137, 0.00000 // (op 6976) call()
v_accvgpr_write a138, 0.00000 // (op 6976) call()
v_accvgpr_write a139, 0.00000 // (op 6976) call()
v_accvgpr_write a140, 0.00000 // (op 6977) call()
v_accvgpr_write a141, 0.00000 // (op 6977) call()
v_accvgpr_write a142, 0.00000 // (op 6977) call()
v_accvgpr_write a143, 0.00000 // (op 6977) call()
v_accvgpr_write a144, 0.00000 // (op 6978) call()
v_accvgpr_write a145, 0.00000 // (op 6978) call()
v_accvgpr_write a146, 0.00000 // (op 6978) call()
v_accvgpr_write a147, 0.00000 // (op 6978) call()
v_accvgpr_write a148, 0.00000 // (op 6979) call()
v_accvgpr_write a149, 0.00000 // (op 6979) call()
v_accvgpr_write a150, 0.00000 // (op 6979) call()
v_accvgpr_write a151, 0.00000 // (op 6979) call()
v_accvgpr_write a152, 0.00000 // (op 6980) call()
v_accvgpr_write a153, 0.00000 // (op 6980) call()
v_accvgpr_write a154, 0.00000 // (op 6980) call()
v_accvgpr_write a155, 0.00000 // (op 6980) call()
v_accvgpr_write a156, 0.00000 // (op 6981) call()
v_accvgpr_write a157, 0.00000 // (op 6981) call()
v_accvgpr_write a158, 0.00000 // (op 6981) call()
v_accvgpr_write a159, 0.00000 // (op 6981) call()
v_accvgpr_write a160, 0.00000 // (op 7896) call()
v_accvgpr_write a161, 0.00000 // (op 7896) call()
v_accvgpr_write a162, 0.00000 // (op 7896) call()
v_accvgpr_write a163, 0.00000 // (op 7896) call()
v_accvgpr_write a164, 0.00000 // (op 8513) call()
v_accvgpr_write a165, 0.00000 // (op 8513) call()
v_accvgpr_write a166, 0.00000 // (op 8513) call()
v_accvgpr_write a167, 0.00000 // (op 8513) call()
v_accvgpr_write a168, 0.00000 // (op 8514) call()
v_accvgpr_write a169, 0.00000 // (op 8514) call()
v_accvgpr_write a170, 0.00000 // (op 8514) call()
v_accvgpr_write a171, 0.00000 // (op 8514) call()
v_accvgpr_write a172, 0.00000 // (op 8515) call()
v_accvgpr_write a173, 0.00000 // (op 8515) call()
v_accvgpr_write a174, 0.00000 // (op 8515) call()
v_accvgpr_write a175, 0.00000 // (op 8515) call()
v_accvgpr_write a176, 0.00000 // (op 8516) call()
v_accvgpr_write a177, 0.00000 // (op 8516) call()
v_accvgpr_write a178, 0.00000 // (op 8516) call()
v_accvgpr_write a179, 0.00000 // (op 8516) call()
v_accvgpr_write a180, 0.00000 // (op 8517) call()
v_accvgpr_write a181, 0.00000 // (op 8517) call()
v_accvgpr_write a182, 0.00000 // (op 8517) call()
v_accvgpr_write a183, 0.00000 // (op 8517) call()
v_accvgpr_write a184, 0.00000 // (op 8518) call()
v_accvgpr_write a185, 0.00000 // (op 8518) call()
v_accvgpr_write a186, 0.00000 // (op 8518) call()
v_accvgpr_write a187, 0.00000 // (op 8518) call()
v_accvgpr_write a188, 0.00000 // (op 8519) call()
v_accvgpr_write a189, 0.00000 // (op 8519) call()
v_accvgpr_write a190, 0.00000 // (op 8519) call()
v_accvgpr_write a191, 0.00000 // (op 8519) call()
v_accvgpr_write a192, 0.00000 // (op 9434) call()
v_accvgpr_write a193, 0.00000 // (op 9434) call()
v_accvgpr_write a194, 0.00000 // (op 9434) call()
v_accvgpr_write a195, 0.00000 // (op 9434) call()
v_accvgpr_write a196, 0.00000 // (op 10051) call()
v_accvgpr_write a197, 0.00000 // (op 10051) call()
v_accvgpr_write a198, 0.00000 // (op 10051) call()
v_accvgpr_write a199, 0.00000 // (op 10051) call()
v_accvgpr_write a200, 0.00000 // (op 10052) call()
v_accvgpr_write a201, 0.00000 // (op 10052) call()
v_accvgpr_write a202, 0.00000 // (op 10052) call()
v_accvgpr_write a203, 0.00000 // (op 10052) call()
v_accvgpr_write a204, 0.00000 // (op 10053) call()
v_accvgpr_write a205, 0.00000 // (op 10053) call()
v_accvgpr_write a206, 0.00000 // (op 10053) call()
v_accvgpr_write a207, 0.00000 // (op 10053) call()
v_accvgpr_write a208, 0.00000 // (op 10054) call()
v_accvgpr_write a209, 0.00000 // (op 10054) call()
v_accvgpr_write a210, 0.00000 // (op 10054) call()
v_accvgpr_write a211, 0.00000 // (op 10054) call()
v_accvgpr_write a212, 0.00000 // (op 10055) call()
v_accvgpr_write a213, 0.00000 // (op 10055) call()
v_accvgpr_write a214, 0.00000 // (op 10055) call()
v_accvgpr_write a215, 0.00000 // (op 10055) call()
v_accvgpr_write a216, 0.00000 // (op 10056) call()
v_accvgpr_write a217, 0.00000 // (op 10056) call()
v_accvgpr_write a218, 0.00000 // (op 10056) call()
v_accvgpr_write a219, 0.00000 // (op 10056) call()
v_accvgpr_write a220, 0.00000 // (op 10057) call()
v_accvgpr_write a221, 0.00000 // (op 10057) call()
v_accvgpr_write a222, 0.00000 // (op 10057) call()
v_accvgpr_write a223, 0.00000 // (op 10057) call()
v_accvgpr_write a224, 0.00000 // (op 10972) call()
v_accvgpr_write a225, 0.00000 // (op 10972) call()
v_accvgpr_write a226, 0.00000 // (op 10972) call()
v_accvgpr_write a227, 0.00000 // (op 10972) call()
v_accvgpr_write a228, 0.00000 // (op 11589) call()
v_accvgpr_write a229, 0.00000 // (op 11589) call()
v_accvgpr_write a230, 0.00000 // (op 11589) call()
v_accvgpr_write a231, 0.00000 // (op 11589) call()
v_accvgpr_write a232, 0.00000 // (op 11590) call()
v_accvgpr_write a233, 0.00000 // (op 11590) call()
v_accvgpr_write a234, 0.00000 // (op 11590) call()
v_accvgpr_write a235, 0.00000 // (op 11590) call()
v_accvgpr_write a236, 0.00000 // (op 11591) call()
v_accvgpr_write a237, 0.00000 // (op 11591) call()
v_accvgpr_write a238, 0.00000 // (op 11591) call()
v_accvgpr_write a239, 0.00000 // (op 11591) call()
v_accvgpr_write a240, 0.00000 // (op 11592) call()
v_accvgpr_write a241, 0.00000 // (op 11592) call()
v_accvgpr_write a242, 0.00000 // (op 11592) call()
v_accvgpr_write a243, 0.00000 // (op 11592) call()
v_accvgpr_write a244, 0.00000 // (op 11593) call()
v_accvgpr_write a245, 0.00000 // (op 11593) call()
v_accvgpr_write a246, 0.00000 // (op 11593) call()
v_accvgpr_write a247, 0.00000 // (op 11593) call()
v_accvgpr_write a248, 0.00000 // (op 11594) call()
v_accvgpr_write a249, 0.00000 // (op 11594) call()
v_accvgpr_write a250, 0.00000 // (op 11594) call()
v_accvgpr_write a251, 0.00000 // (op 11594) call()
v_accvgpr_write a252, 0.00000 // (op 11595) call()
v_accvgpr_write a253, 0.00000 // (op 11595) call()
v_accvgpr_write a254, 0.00000 // (op 11595) call()
v_accvgpr_write a255, 0.00000 // (op 11595) call()
s_mul_hi_i32 s0, s16, s12 // (op 35424) 
v_and_b32 v14, -4, v1 // (op 35424) 
s_add_i32 s1, s0, s16 // (op 35424) 

v_sub_u32 v15, v1, v14 // (op 35424) 

s_ashr_i32 s0, s1, 31 // (op 35424) 
v_lshlrev_b32 v14, 6, v15 // (op 35424) 

s_cmp_eq_i32 s12, 0 // (op 35424) 
s_mov_b32 s18, scc // (op 35424) 
v_lshrrev_b32 v15, 2, v1 // (op 35424) 
s_cmp_eq_i32 s18, 0 // (op 35424) 
s_cselect_b32 s19, 0, -1 // (op 35424) 

v_lshlrev_b32 v16, 1, v15 // (op 35424) 

s_and_b32 s18, s15, 31 // (op 35424) 
s_lshl_b32 s20, 1, s18 // (op 35424) 
s_add_i32 s21, s20, s19 // (op 35424) 


s_and_b32 s19, s0, s21 // (op 35424) 


s_add_i32 s0, s1, s19 // (op 35424) 


s_ashr_i32 s1, s0, s18 // (op 35424) 


s_lshl_b32 s0, s15, 24 // (op 35424) 
s_ashr_i32 s18, s0, 31 // (op 35424) 

s_xor_b32 s0, s1, s18 // (op 35424) 

s_sub_i32 s1, s0, s18 // (op 35424) 


s_mul_hi_i32 s0, s1, s44 // (op 35424) 
s_add_i32 s18, s0, s1 // (op 35424) 

s_ashr_i32 s0, s18, 31 // (op 35424) 
s_cmp_eq_i32 s44, 0 // (op 35424) 
s_mov_b32 s19, scc // (op 35424) 
s_cmp_eq_i32 s19, 0 // (op 35424) 
s_cselect_b32 s20, 0, -1 // (op 35424) 

s_and_b32 s19, s45, 31 // (op 35424) 
s_lshl_b32 s21, 1, s19 // (op 35424) 
s_add_i32 s22, s21, s20 // (op 35424) 


s_and_b32 s23, s0, s22 // (op 35424) 


s_add_i32 s0, s18, s23 // (op 35424) 


s_ashr_i32 s18, s0, s19 // (op 35424) 


s_lshl_b32 s0, s45, 24 // (op 35424) 
s_ashr_i32 s19, s0, 31 // (op 35424) 

s_xor_b32 s0, s18, s19 // (op 35424) 

s_sub_i32 s18, s0, s19 // (op 35424) 


s_mul_i32 s0, s1, s13 // (op 35424) 

s_sub_u32 s1, s16, s0 // (op 35424) 

s_mul_i32 s0, s18, s13 // (op 35424) 
s_add_u32 s0, s0, s1 // (op 35424) 


s_mul_hi_i32 s1, s16, s11 // (op 35424) 
s_add_i32 s18, s1, s16 // (op 35424) 

s_ashr_i32 s1, s18, 31 // (op 35424) 
s_cmp_eq_i32 s11, 0 // (op 35424) 
s_mov_b32 s19, scc // (op 35424) 
s_cmp_eq_i32 s19, 0 // (op 35424) 
s_cselect_b32 s20, 0, -1 // (op 35424) 

s_and_b32 s19, s47, 31 // (op 35424) 
s_lshl_b32 s21, 1, s19 // (op 35424) 
s_add_i32 s22, s21, s20 // (op 35424) 


s_and_b32 s23, s1, s22 // (op 35424) 


s_add_i32 s1, s18, s23 // (op 35424) 


s_ashr_i32 s18, s1, s19 // (op 35424) 


s_lshl_b32 s1, s47, 24 // (op 35424) 
s_ashr_i32 s19, s1, 31 // (op 35424) 

s_xor_b32 s1, s18, s19 // (op 35424) 

s_sub_i32 s18, s1, s19 // (op 35424) 


s_add_u32 s22, s8, 256 // (op 35424) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 35424) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 35424) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 35424) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 8 // (op 35424) 

s_mul_i32 s1, s13, s22 // (op 35424) 

s_mul_i32 s19, s10, s1 // (op 35424) 

s_mul_i32 s1, s18, s19 // (op 35424) 

s_sub_u32 s19, s16, s1 // (op 35424) 

s_mov_b32 s1, s19 // (op 35424) 
s_mul_hi_i32 s20, s1, s14 // (op 35424) 
s_add_i32 s21, s20, s1 // (op 35424) 


s_ashr_i32 s1, s21, 31 // (op 35424) 
s_cmp_eq_i32 s14, 0 // (op 35424) 
s_mov_b32 s20, scc // (op 35424) 
s_cmp_eq_i32 s20, 0 // (op 35424) 
s_cselect_b32 s22, 0, -1 // (op 35424) 

s_and_b32 s20, s46, 31 // (op 35424) 
s_lshl_b32 s23, 1, s20 // (op 35424) 
s_add_i32 s49, s23, s22 // (op 35424) 


s_and_b32 s22, s1, s49 // (op 35424) 


s_add_i32 s1, s21, s22 // (op 35424) 


s_ashr_i32 s21, s1, s20 // (op 35424) 


s_lshl_b32 s1, s46, 24 // (op 35424) 
s_ashr_i32 s20, s1, 31 // (op 35424) 

s_xor_b32 s1, s21, s20 // (op 35424) 

s_sub_i32 s21, s1, s20 // (op 35424) 


s_mul_i32 s1, s48, s13 // (op 35424) 
s_add_u32 s22, s6, 256 // (op 35424) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 35424) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 35424) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 35424) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 8 // (op 35424) 

s_sub_i32 s20, s22, s1 // (op 35424) 

s_mul_i32 s22, s21, s20 // (op 35424) 


s_sub_u32 s23, s19, s22 // (op 35424) 


s_add_u32 s19, s23, s1 // (op 35424) 


s_cmp_eq_i32 s18, 0 // (op 35424) 
s_mov_b32 s1, scc // (op 35424) 

s_cmp_eq_i32 s1, 0 // (op 35424) 
s_cselect_b32 s18, s19, s0 // (op 35424) 



v_lshl_add_u32 v15, s18, 8, v14 // (op 35424) 


v_mov_b32 v19, 0 // (op 35424) convert
v_mov_b32 v18, v15 // (op 35424) convert

v_mul_lo_u32 v15, v16, s25 // (op 35424) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 35424) most significant: high of low * low
v_mul_lo_u32 v14, v16, s24 // (op 35424) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 35424) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v14 // (op 35424) least significant half
v_addc_co_u32 v17, s[0:1], v19, v15, s[0:1] // (op 35424) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 35424) most significant: low of high * low
v_mul_hi_u32 v18, v16, 6 // (op 35424) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 35424) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 35424) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 35424) 

v_mov_b32 v14, v16 // (op 35424) convert

s_mul_i32 s1, s29, 6 // (op 35426) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 35426) least significant
s_mul_hi_u32 s18, s28, 6 // (op 35426) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35426) most significant: sum

s_mul_i32 s19, s24, s1 // (op 35426) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 35426) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 35426) least significant
s_mul_hi_u32 s21, s24, s0 // (op 35426) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 35426) most significant: sum
s_add_u32 s19, s19, s21 // (op 35426) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 35426) 

s_mov_b32 s20, s26 // (op 35426) reg expression
s_mov_b32 s21, s27 // (op 35426) reg expression
s_mov_b32 s22, s0 // (op 35426) reg expression
s_mov_b32 s23, 131072 // (op 35426) call()

s_mul_hi_i32 s0, s16, s12 // (op 35443) 
v_and_b32 v15, -4, v1 // (op 35443) 
s_add_i32 s1, s0, s16 // (op 35443) 

v_sub_u32 v16, v1, v15 // (op 35443) 

s_ashr_i32 s0, s1, 31 // (op 35443) 
v_lshlrev_b32 v15, 6, v16 // (op 35443) 

s_cmp_eq_i32 s12, 0 // (op 35443) 
s_mov_b32 s18, scc // (op 35443) 
v_lshrrev_b32 v16, 2, v1 // (op 35443) 
s_cmp_eq_i32 s18, 0 // (op 35443) 
s_cselect_b32 s19, 0, -1 // (op 35443) 

v_lshlrev_b32 v17, 1, v16 // (op 35443) 

s_and_b32 s18, s15, 31 // (op 35443) 
s_lshl_b32 s49, 1, s18 // (op 35443) 
s_add_i32 s50, s49, s19 // (op 35443) 


s_and_b32 s19, s0, s50 // (op 35443) 


s_add_i32 s0, s1, s19 // (op 35443) 


s_ashr_i32 s1, s0, s18 // (op 35443) 


s_lshl_b32 s0, s15, 24 // (op 35443) 
s_ashr_i32 s18, s0, 31 // (op 35443) 

s_xor_b32 s0, s1, s18 // (op 35443) 

s_sub_i32 s1, s0, s18 // (op 35443) 


s_mul_hi_i32 s0, s1, s44 // (op 35443) 
s_add_i32 s18, s0, s1 // (op 35443) 

s_ashr_i32 s0, s18, 31 // (op 35443) 
s_cmp_eq_i32 s44, 0 // (op 35443) 
s_mov_b32 s19, scc // (op 35443) 
s_cmp_eq_i32 s19, 0 // (op 35443) 
s_cselect_b32 s49, 0, -1 // (op 35443) 

s_and_b32 s19, s45, 31 // (op 35443) 
s_lshl_b32 s50, 1, s19 // (op 35443) 
s_add_i32 s51, s50, s49 // (op 35443) 


s_and_b32 s49, s0, s51 // (op 35443) 


s_add_i32 s0, s18, s49 // (op 35443) 


s_ashr_i32 s18, s0, s19 // (op 35443) 


s_lshl_b32 s0, s45, 24 // (op 35443) 
s_ashr_i32 s19, s0, 31 // (op 35443) 

s_xor_b32 s0, s18, s19 // (op 35443) 

s_sub_i32 s18, s0, s19 // (op 35443) 


s_mul_i32 s0, s1, s13 // (op 35443) 

s_sub_u32 s1, s16, s0 // (op 35443) 

s_mul_i32 s0, s18, s13 // (op 35443) 
s_add_u32 s0, s0, s1 // (op 35443) 


s_mul_hi_i32 s1, s16, s11 // (op 35443) 
s_add_i32 s18, s1, s16 // (op 35443) 

s_ashr_i32 s1, s18, 31 // (op 35443) 
s_cmp_eq_i32 s11, 0 // (op 35443) 
s_mov_b32 s19, scc // (op 35443) 
s_cmp_eq_i32 s19, 0 // (op 35443) 
s_cselect_b32 s49, 0, -1 // (op 35443) 

s_and_b32 s19, s47, 31 // (op 35443) 
s_lshl_b32 s50, 1, s19 // (op 35443) 
s_add_i32 s51, s50, s49 // (op 35443) 


s_and_b32 s49, s1, s51 // (op 35443) 


s_add_i32 s1, s18, s49 // (op 35443) 


s_ashr_i32 s18, s1, s19 // (op 35443) 


s_lshl_b32 s1, s47, 24 // (op 35443) 
s_ashr_i32 s19, s1, 31 // (op 35443) 

s_xor_b32 s1, s18, s19 // (op 35443) 

s_sub_i32 s18, s1, s19 // (op 35443) 


s_add_u32 s54, s8, 256 // (op 35443) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 35443) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 35443) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 35443) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 35443) 

s_mul_i32 s1, s13, s50 // (op 35443) 

s_mul_i32 s19, s10, s1 // (op 35443) 

s_mul_i32 s1, s18, s19 // (op 35443) 

s_sub_u32 s19, s16, s1 // (op 35443) 

s_mov_b32 s1, s19 // (op 35443) 
s_mul_hi_i32 s49, s1, s14 // (op 35443) 
s_add_i32 s50, s49, s1 // (op 35443) 


s_ashr_i32 s1, s50, 31 // (op 35443) 
s_cmp_eq_i32 s14, 0 // (op 35443) 
s_mov_b32 s49, scc // (op 35443) 
s_cmp_eq_i32 s49, 0 // (op 35443) 
s_cselect_b32 s51, 0, -1 // (op 35443) 

s_and_b32 s49, s46, 31 // (op 35443) 
s_lshl_b32 s52, 1, s49 // (op 35443) 
s_add_i32 s53, s52, s51 // (op 35443) 


s_and_b32 s51, s1, s53 // (op 35443) 


s_add_i32 s1, s50, s51 // (op 35443) 


s_ashr_i32 s50, s1, s49 // (op 35443) 


s_lshl_b32 s1, s46, 24 // (op 35443) 
s_ashr_i32 s49, s1, 31 // (op 35443) 

s_xor_b32 s1, s50, s49 // (op 35443) 

s_sub_i32 s50, s1, s49 // (op 35443) 


s_mul_i32 s1, s48, s13 // (op 35443) 
s_add_u32 s54, s6, 256 // (op 35443) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 35443) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 35443) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 35443) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 8 // (op 35443) 

s_sub_i32 s49, s54, s1 // (op 35443) 

s_mul_i32 s51, s50, s49 // (op 35443) 


s_sub_u32 s49, s19, s51 // (op 35443) 


s_add_u32 s19, s49, s1 // (op 35443) 


s_cmp_eq_i32 s18, 0 // (op 35443) 
s_mov_b32 s1, scc // (op 35443) 

s_cmp_eq_i32 s1, 0 // (op 35443) 
s_cselect_b32 s18, s19, s0 // (op 35443) 



v_lshl_add_u32 v16, s18, 8, v15 // (op 35443) 


v_mov_b32 v19, 0 // (op 35443) convert
v_mov_b32 v18, v16 // (op 35443) convert

v_mul_lo_u32 v21, v17, s25 // (op 35443) most significant: low of low * high
v_mul_hi_u32 v15, v17, s24 // (op 35443) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 35443) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 35443) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 35443) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 35443) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 35443) most significant: low of high * low
v_mul_hi_u32 v15, v16, 6 // (op 35443) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 35443) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 35443) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 35443) 

v_mov_b32 v15, v16 // (op 35443) convert

s_mul_i32 s1, s29, 6 // (op 35445) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 35445) least significant
s_mul_hi_u32 s18, s28, 6 // (op 35445) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35445) most significant: sum

s_mul_i32 s19, s24, s1 // (op 35445) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 35445) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 35445) least significant
s_mul_hi_u32 s50, s24, s0 // (op 35445) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 35445) most significant: sum
s_add_u32 s19, s19, s50 // (op 35445) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 35445) 

s_mov_b32 s20, s26 // (op 35445) reg expression
s_mov_b32 s21, s27 // (op 35445) reg expression
s_mov_b32 s22, s0 // (op 35445) reg expression
s_mov_b32 s23, 131072 // (op 35445) call()

s_mul_hi_i32 s0, s16, s12 // (op 35488) 
v_and_b32 v16, -4, v1 // (op 35488) 
s_add_i32 s1, s0, s16 // (op 35488) 

v_sub_u32 v17, v1, v16 // (op 35488) 

s_ashr_i32 s0, s1, 31 // (op 35488) 
v_lshlrev_b32 v16, 6, v17 // (op 35488) 

s_cmp_eq_i32 s12, 0 // (op 35488) 
s_mov_b32 s18, scc // (op 35488) 
v_lshrrev_b32 v17, 2, v1 // (op 35488) 
s_cmp_eq_i32 s18, 0 // (op 35488) 
s_cselect_b32 s19, 0, -1 // (op 35488) 

v_lshlrev_b32 v18, 1, v17 // (op 35488) 

s_and_b32 s18, s15, 31 // (op 35488) 
s_lshl_b32 s49, 1, s18 // (op 35488) 
s_add_i32 s50, s49, s19 // (op 35488) 


s_and_b32 s19, s0, s50 // (op 35488) 


s_add_i32 s0, s1, s19 // (op 35488) 


s_ashr_i32 s1, s0, s18 // (op 35488) 


s_lshl_b32 s0, s15, 24 // (op 35488) 
s_ashr_i32 s18, s0, 31 // (op 35488) 

s_xor_b32 s0, s1, s18 // (op 35488) 

s_sub_i32 s1, s0, s18 // (op 35488) 


s_mul_hi_i32 s0, s1, s44 // (op 35488) 
s_add_i32 s18, s0, s1 // (op 35488) 

s_ashr_i32 s0, s18, 31 // (op 35488) 
s_cmp_eq_i32 s44, 0 // (op 35488) 
s_mov_b32 s19, scc // (op 35488) 
s_cmp_eq_i32 s19, 0 // (op 35488) 
s_cselect_b32 s49, 0, -1 // (op 35488) 

s_and_b32 s19, s45, 31 // (op 35488) 
s_lshl_b32 s50, 1, s19 // (op 35488) 
s_add_i32 s51, s50, s49 // (op 35488) 


s_and_b32 s49, s0, s51 // (op 35488) 


s_add_i32 s0, s18, s49 // (op 35488) 


s_ashr_i32 s18, s0, s19 // (op 35488) 


s_lshl_b32 s0, s45, 24 // (op 35488) 
s_ashr_i32 s19, s0, 31 // (op 35488) 

s_xor_b32 s0, s18, s19 // (op 35488) 

s_sub_i32 s18, s0, s19 // (op 35488) 


s_add_u32 s54, s8, 256 // (op 35488) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 35488) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 35488) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 35488) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 35488) 

s_mul_i32 s0, s18, s50 // (op 35488) 

s_sub_i32 s49, s1, s0 // (op 35488) 


s_mul_hi_i32 s0, s16, s11 // (op 35488) 
s_add_i32 s1, s0, s16 // (op 35488) 

s_ashr_i32 s0, s1, 31 // (op 35488) 
s_cmp_eq_i32 s11, 0 // (op 35488) 
s_mov_b32 s18, scc // (op 35488) 
s_cmp_eq_i32 s18, 0 // (op 35488) 
s_cselect_b32 s19, 0, -1 // (op 35488) 

s_and_b32 s18, s47, 31 // (op 35488) 
s_lshl_b32 s52, 1, s18 // (op 35488) 
s_add_i32 s53, s52, s19 // (op 35488) 


s_and_b32 s19, s0, s53 // (op 35488) 


s_add_i32 s0, s1, s19 // (op 35488) 


s_ashr_i32 s1, s0, s18 // (op 35488) 


s_lshl_b32 s0, s47, 24 // (op 35488) 
s_ashr_i32 s18, s0, 31 // (op 35488) 

s_xor_b32 s0, s1, s18 // (op 35488) 

s_sub_i32 s1, s0, s18 // (op 35488) 


s_mul_i32 s0, s13, s50 // (op 35488) 

s_mul_i32 s18, s10, s0 // (op 35488) 

s_mul_i32 s0, s1, s18 // (op 35488) 

s_sub_u32 s18, s16, s0 // (op 35488) 

s_mul_hi_i32 s0, s18, s14 // (op 35488) 
s_add_i32 s19, s0, s18 // (op 35488) 


s_ashr_i32 s0, s19, 31 // (op 35488) 
s_cmp_eq_i32 s14, 0 // (op 35488) 
s_mov_b32 s18, scc // (op 35488) 
s_cmp_eq_i32 s18, 0 // (op 35488) 
s_cselect_b32 s50, 0, -1 // (op 35488) 

s_and_b32 s18, s46, 31 // (op 35488) 
s_lshl_b32 s51, 1, s18 // (op 35488) 
s_add_i32 s52, s51, s50 // (op 35488) 


s_and_b32 s50, s0, s52 // (op 35488) 


s_add_i32 s0, s19, s50 // (op 35488) 


s_ashr_i32 s19, s0, s18 // (op 35488) 


s_lshl_b32 s0, s46, 24 // (op 35488) 
s_ashr_i32 s18, s0, 31 // (op 35488) 

s_xor_b32 s0, s19, s18 // (op 35488) 

s_sub_i32 s19, s0, s18 // (op 35488) 


s_cmp_eq_i32 s1, 0 // (op 35488) 
s_mov_b32 s0, scc // (op 35488) 

s_cmp_eq_i32 s0, 0 // (op 35488) 
s_cselect_b32 s1, s19, s49 // (op 35488) 



v_lshl_add_u32 v17, s1, 8, v16 // (op 35488) 


v_mov_b32 v21, 0 // (op 35488) convert
v_mov_b32 v20, v17 // (op 35488) convert

v_mul_lo_u32 v17, v18, s31 // (op 35488) most significant: low of low * high
v_mul_hi_u32 v19, v18, s30 // (op 35488) most significant: high of low * low
v_mul_lo_u32 v16, v18, s30 // (op 35488) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 35488) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v20 // (op 35488) least significant half
v_addc_co_u32 v17, s[0:1], v17, v21, s[0:1] // (op 35488) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 35488) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 35488) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 35488) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 35488) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 35488) 

v_mov_b32 v18, v16 // (op 35488) convert

s_mul_i32 s1, s29, 6 // (op 35490) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 35490) least significant
s_mul_hi_u32 s18, s28, 6 // (op 35490) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35490) most significant: sum

s_mul_i32 s19, s30, s1 // (op 35490) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 35490) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 35490) least significant
s_mul_hi_u32 s50, s30, s0 // (op 35490) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 35490) most significant: sum
s_add_u32 s19, s19, s50 // (op 35490) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 35490) 

s_mov_b32 s52, s32 // (op 35490) reg expression
s_mov_b32 s53, s33 // (op 35490) reg expression
s_mov_b32 s54, s0 // (op 35490) reg expression
s_mov_b32 s55, 131072 // (op 35490) call()

s_mul_hi_i32 s0, s16, s12 // (op 35507) 
v_and_b32 v16, -4, v1 // (op 35507) 
s_add_i32 s1, s0, s16 // (op 35507) 

v_sub_u32 v17, v1, v16 // (op 35507) 

s_ashr_i32 s0, s1, 31 // (op 35507) 
v_lshlrev_b32 v16, 6, v17 // (op 35507) 

s_cmp_eq_i32 s12, 0 // (op 35507) 
s_mov_b32 s18, scc // (op 35507) 
v_lshrrev_b32 v17, 2, v1 // (op 35507) 
s_cmp_eq_i32 s18, 0 // (op 35507) 
s_cselect_b32 s19, 0, -1 // (op 35507) 

v_lshlrev_b32 v19, 1, v17 // (op 35507) 

s_and_b32 s18, s15, 31 // (op 35507) 
s_lshl_b32 s49, 1, s18 // (op 35507) 
s_add_i32 s50, s49, s19 // (op 35507) 


s_and_b32 s19, s0, s50 // (op 35507) 


s_add_i32 s0, s1, s19 // (op 35507) 


s_ashr_i32 s1, s0, s18 // (op 35507) 


s_lshl_b32 s0, s15, 24 // (op 35507) 
s_ashr_i32 s18, s0, 31 // (op 35507) 

s_xor_b32 s0, s1, s18 // (op 35507) 

s_sub_i32 s1, s0, s18 // (op 35507) 


s_mul_hi_i32 s0, s1, s44 // (op 35507) 
s_add_i32 s18, s0, s1 // (op 35507) 

s_ashr_i32 s0, s18, 31 // (op 35507) 
s_cmp_eq_i32 s44, 0 // (op 35507) 
s_mov_b32 s19, scc // (op 35507) 
s_cmp_eq_i32 s19, 0 // (op 35507) 
s_cselect_b32 s49, 0, -1 // (op 35507) 

s_and_b32 s19, s45, 31 // (op 35507) 
s_lshl_b32 s50, 1, s19 // (op 35507) 
s_add_i32 s51, s50, s49 // (op 35507) 


s_and_b32 s49, s0, s51 // (op 35507) 


s_add_i32 s0, s18, s49 // (op 35507) 


s_ashr_i32 s18, s0, s19 // (op 35507) 


s_lshl_b32 s0, s45, 24 // (op 35507) 
s_ashr_i32 s19, s0, 31 // (op 35507) 

s_xor_b32 s0, s18, s19 // (op 35507) 

s_sub_i32 s18, s0, s19 // (op 35507) 


s_add_u32 s50, s8, 256 // (op 35507) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 35507) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 35507) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 35507) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 35507) 

s_mul_i32 s0, s18, s50 // (op 35507) 

s_sub_i32 s49, s1, s0 // (op 35507) 


s_mul_hi_i32 s0, s16, s11 // (op 35507) 
s_add_i32 s1, s0, s16 // (op 35507) 

s_ashr_i32 s0, s1, 31 // (op 35507) 
s_cmp_eq_i32 s11, 0 // (op 35507) 
s_mov_b32 s18, scc // (op 35507) 
s_cmp_eq_i32 s18, 0 // (op 35507) 
s_cselect_b32 s19, 0, -1 // (op 35507) 

s_and_b32 s18, s47, 31 // (op 35507) 
s_lshl_b32 s64, 1, s18 // (op 35507) 
s_add_i32 s65, s64, s19 // (op 35507) 


s_and_b32 s19, s0, s65 // (op 35507) 


s_add_i32 s0, s1, s19 // (op 35507) 


s_ashr_i32 s1, s0, s18 // (op 35507) 


s_lshl_b32 s0, s47, 24 // (op 35507) 
s_ashr_i32 s18, s0, 31 // (op 35507) 

s_xor_b32 s0, s1, s18 // (op 35507) 

s_sub_i32 s1, s0, s18 // (op 35507) 


s_mul_i32 s0, s13, s50 // (op 35507) 

s_mul_i32 s18, s10, s0 // (op 35507) 

s_mul_i32 s0, s1, s18 // (op 35507) 

s_sub_u32 s18, s16, s0 // (op 35507) 

s_mul_hi_i32 s0, s18, s14 // (op 35507) 
s_add_i32 s19, s0, s18 // (op 35507) 


s_ashr_i32 s0, s19, 31 // (op 35507) 
s_cmp_eq_i32 s14, 0 // (op 35507) 
s_mov_b32 s18, scc // (op 35507) 
s_cmp_eq_i32 s18, 0 // (op 35507) 
s_cselect_b32 s50, 0, -1 // (op 35507) 

s_and_b32 s18, s46, 31 // (op 35507) 
s_lshl_b32 s51, 1, s18 // (op 35507) 
s_add_i32 s64, s51, s50 // (op 35507) 


s_and_b32 s50, s0, s64 // (op 35507) 


s_add_i32 s0, s19, s50 // (op 35507) 


s_ashr_i32 s19, s0, s18 // (op 35507) 


s_lshl_b32 s0, s46, 24 // (op 35507) 
s_ashr_i32 s18, s0, 31 // (op 35507) 

s_xor_b32 s0, s19, s18 // (op 35507) 

s_sub_i32 s19, s0, s18 // (op 35507) 


s_cmp_eq_i32 s1, 0 // (op 35507) 
s_mov_b32 s0, scc // (op 35507) 

s_cmp_eq_i32 s0, 0 // (op 35507) 
s_cselect_b32 s1, s19, s49 // (op 35507) 



v_lshl_add_u32 v17, s1, 8, v16 // (op 35507) 


v_mov_b32 v21, 0 // (op 35507) convert
v_mov_b32 v20, v17 // (op 35507) convert

v_mul_lo_u32 v17, v19, s31 // (op 35507) most significant: low of low * high
v_mul_hi_u32 v22, v19, s30 // (op 35507) most significant: high of low * low
v_mul_lo_u32 v16, v19, s30 // (op 35507) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 35507) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v20 // (op 35507) least significant half
v_addc_co_u32 v17, s[0:1], v17, v21, s[0:1] // (op 35507) most significant half



v_mul_lo_u32 v21, v17, 6 // (op 35507) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 35507) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 35507) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 35507) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 35507) 

v_mov_b32 v19, v16 // (op 35507) convert

s_mul_i32 s1, s29, 6 // (op 35509) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 35509) least significant
s_mul_hi_u32 s18, s28, 6 // (op 35509) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35509) most significant: sum

s_mul_i32 s19, s30, s1 // (op 35509) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 35509) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 35509) least significant
s_mul_hi_u32 s50, s30, s0 // (op 35509) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 35509) most significant: sum
s_add_u32 s19, s19, s50 // (op 35509) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 35509) 

s_mov_b32 s52, s32 // (op 35509) reg expression
s_mov_b32 s53, s33 // (op 35509) reg expression
s_mov_b32 s54, s0 // (op 35509) reg expression
s_mov_b32 s55, 131072 // (op 35509) call()

s_mul_hi_i32 s0, s16, s12 // (op 35552) 
v_and_b32 v16, -64, v1 // (op 35552) 
s_add_i32 s1, s0, s16 // (op 35552) 

v_sub_u32 v17, v1, v16 // (op 35552) 

s_ashr_i32 s0, s1, 31 // (op 35552) 
v_lshlrev_b32 v16, 2, v17 // (op 35552) 

s_cmp_eq_i32 s12, 0 // (op 35552) 
s_mov_b32 s18, scc // (op 35552) 
v_lshrrev_b32 v17, 6, v1 // (op 35552) 
s_cmp_eq_i32 s18, 0 // (op 35552) 
s_cselect_b32 s19, 0, -1 // (op 35552) 

s_and_b32 s18, s15, 31 // (op 35552) 
s_lshl_b32 s49, 1, s18 // (op 35552) 
s_add_i32 s50, s49, s19 // (op 35552) 


s_and_b32 s19, s0, s50 // (op 35552) 


s_add_i32 s0, s1, s19 // (op 35552) 


s_ashr_i32 s1, s0, s18 // (op 35552) 


s_lshl_b32 s0, s15, 24 // (op 35552) 
s_ashr_i32 s18, s0, 31 // (op 35552) 

s_xor_b32 s0, s1, s18 // (op 35552) 

s_sub_i32 s1, s0, s18 // (op 35552) 


s_mul_hi_i32 s0, s1, s44 // (op 35552) 
s_add_i32 s18, s0, s1 // (op 35552) 

s_ashr_i32 s0, s18, 31 // (op 35552) 
s_cmp_eq_i32 s44, 0 // (op 35552) 
s_mov_b32 s19, scc // (op 35552) 
s_cmp_eq_i32 s19, 0 // (op 35552) 
s_cselect_b32 s49, 0, -1 // (op 35552) 

s_and_b32 s19, s45, 31 // (op 35552) 
s_lshl_b32 s50, 1, s19 // (op 35552) 
s_add_i32 s51, s50, s49 // (op 35552) 


s_and_b32 s49, s0, s51 // (op 35552) 


s_add_i32 s0, s18, s49 // (op 35552) 


s_ashr_i32 s18, s0, s19 // (op 35552) 


s_lshl_b32 s0, s45, 24 // (op 35552) 
s_ashr_i32 s19, s0, 31 // (op 35552) 

s_xor_b32 s0, s18, s19 // (op 35552) 

s_sub_i32 s18, s0, s19 // (op 35552) 


s_mul_i32 s0, s1, s13 // (op 35552) 

s_sub_u32 s1, s16, s0 // (op 35552) 

s_mul_i32 s0, s18, s13 // (op 35552) 
s_add_u32 s0, s0, s1 // (op 35552) 


s_mul_hi_i32 s1, s16, s11 // (op 35552) 
s_add_i32 s18, s1, s16 // (op 35552) 

s_ashr_i32 s1, s18, 31 // (op 35552) 
s_cmp_eq_i32 s11, 0 // (op 35552) 
s_mov_b32 s19, scc // (op 35552) 
s_cmp_eq_i32 s19, 0 // (op 35552) 
s_cselect_b32 s49, 0, -1 // (op 35552) 

s_and_b32 s19, s47, 31 // (op 35552) 
s_lshl_b32 s50, 1, s19 // (op 35552) 
s_add_i32 s51, s50, s49 // (op 35552) 


s_and_b32 s49, s1, s51 // (op 35552) 


s_add_i32 s1, s18, s49 // (op 35552) 


s_ashr_i32 s18, s1, s19 // (op 35552) 


s_lshl_b32 s1, s47, 24 // (op 35552) 
s_ashr_i32 s19, s1, 31 // (op 35552) 

s_xor_b32 s1, s18, s19 // (op 35552) 

s_sub_i32 s18, s1, s19 // (op 35552) 


s_add_u32 s50, s8, 256 // (op 35552) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 35552) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 35552) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 35552) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 35552) 

s_mul_i32 s1, s13, s50 // (op 35552) 

s_mul_i32 s19, s10, s1 // (op 35552) 

s_mul_i32 s1, s18, s19 // (op 35552) 

s_sub_u32 s19, s16, s1 // (op 35552) 

s_mov_b32 s1, s19 // (op 35552) 
s_mul_hi_i32 s49, s1, s14 // (op 35552) 
s_add_i32 s50, s49, s1 // (op 35552) 


s_ashr_i32 s1, s50, 31 // (op 35552) 
s_cmp_eq_i32 s14, 0 // (op 35552) 
s_mov_b32 s49, scc // (op 35552) 
s_cmp_eq_i32 s49, 0 // (op 35552) 
s_cselect_b32 s51, 0, -1 // (op 35552) 

s_and_b32 s49, s46, 31 // (op 35552) 
s_lshl_b32 s64, 1, s49 // (op 35552) 
s_add_i32 s65, s64, s51 // (op 35552) 


s_and_b32 s51, s1, s65 // (op 35552) 


s_add_i32 s1, s50, s51 // (op 35552) 


s_ashr_i32 s50, s1, s49 // (op 35552) 


s_lshl_b32 s1, s46, 24 // (op 35552) 
s_ashr_i32 s49, s1, 31 // (op 35552) 

s_xor_b32 s1, s50, s49 // (op 35552) 

s_sub_i32 s50, s1, s49 // (op 35552) 


s_mul_i32 s1, s48, s13 // (op 35552) 
s_add_u32 s64, s6, 256 // (op 35552) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 35552) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 35552) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 35552) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 8 // (op 35552) 

s_sub_i32 s49, s64, s1 // (op 35552) 

s_mul_i32 s51, s50, s49 // (op 35552) 


s_sub_u32 s49, s19, s51 // (op 35552) 


s_add_u32 s19, s49, s1 // (op 35552) 


s_cmp_eq_i32 s18, 0 // (op 35552) 
s_mov_b32 s1, scc // (op 35552) 

s_cmp_eq_i32 s1, 0 // (op 35552) 
s_cselect_b32 s18, s19, s0 // (op 35552) 



v_lshl_add_u32 v20, s18, 8, v16 // (op 35552) 


v_mul_lo_u32 v16, v17, s34 // (op 35552) 

v_add_u32 v17, v20, v16 // (op 35552) 


s_mul_i32 s1, s34, s39 // (op 35554) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 35554) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 35554) least significant
s_mul_hi_u32 s19, s34, s38 // (op 35554) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35554) most significant: sum
s_add_u32 s1, s1, s19 // (op 35554) most significant: sum


s_mov_b32 s64, s36 // (op 35554) reg expression
s_mov_b32 s65, s37 // (op 35554) reg expression
s_mov_b32 s66, s0 // (op 35554) reg expression
s_mov_b32 s67, 131072 // (op 35554) call()

s_mul_hi_i32 s0, s16, s12 // (op 35571) 
v_and_b32 v16, -64, v1 // (op 35571) 
s_add_i32 s1, s0, s16 // (op 35571) 

v_sub_u32 v20, v1, v16 // (op 35571) 

s_ashr_i32 s0, s1, 31 // (op 35571) 
v_lshlrev_b32 v16, 2, v20 // (op 35571) 

s_cmp_eq_i32 s12, 0 // (op 35571) 
s_mov_b32 s18, scc // (op 35571) 
v_lshrrev_b32 v20, 6, v1 // (op 35571) 
s_cmp_eq_i32 s18, 0 // (op 35571) 
s_cselect_b32 s19, 0, -1 // (op 35571) 

s_and_b32 s18, s15, 31 // (op 35571) 
s_lshl_b32 s49, 1, s18 // (op 35571) 
s_add_i32 s50, s49, s19 // (op 35571) 


s_and_b32 s19, s0, s50 // (op 35571) 


s_add_i32 s0, s1, s19 // (op 35571) 


s_ashr_i32 s1, s0, s18 // (op 35571) 


s_lshl_b32 s0, s15, 24 // (op 35571) 
s_ashr_i32 s18, s0, 31 // (op 35571) 

s_xor_b32 s0, s1, s18 // (op 35571) 

s_sub_i32 s1, s0, s18 // (op 35571) 


s_mul_hi_i32 s0, s1, s44 // (op 35571) 
s_add_i32 s18, s0, s1 // (op 35571) 

s_ashr_i32 s0, s18, 31 // (op 35571) 
s_cmp_eq_i32 s44, 0 // (op 35571) 
s_mov_b32 s19, scc // (op 35571) 
s_cmp_eq_i32 s19, 0 // (op 35571) 
s_cselect_b32 s49, 0, -1 // (op 35571) 

s_and_b32 s19, s45, 31 // (op 35571) 
s_lshl_b32 s50, 1, s19 // (op 35571) 
s_add_i32 s51, s50, s49 // (op 35571) 


s_and_b32 s49, s0, s51 // (op 35571) 


s_add_i32 s0, s18, s49 // (op 35571) 


s_ashr_i32 s18, s0, s19 // (op 35571) 


s_lshl_b32 s0, s45, 24 // (op 35571) 
s_ashr_i32 s19, s0, 31 // (op 35571) 

s_xor_b32 s0, s18, s19 // (op 35571) 

s_sub_i32 s18, s0, s19 // (op 35571) 


s_mul_i32 s0, s1, s13 // (op 35571) 

s_sub_u32 s1, s16, s0 // (op 35571) 

s_mul_i32 s0, s18, s13 // (op 35571) 
s_add_u32 s0, s0, s1 // (op 35571) 


s_mul_hi_i32 s1, s16, s11 // (op 35571) 
s_add_i32 s18, s1, s16 // (op 35571) 

s_ashr_i32 s1, s18, 31 // (op 35571) 
s_cmp_eq_i32 s11, 0 // (op 35571) 
s_mov_b32 s19, scc // (op 35571) 
s_cmp_eq_i32 s19, 0 // (op 35571) 
s_cselect_b32 s49, 0, -1 // (op 35571) 

s_and_b32 s19, s47, 31 // (op 35571) 
s_lshl_b32 s50, 1, s19 // (op 35571) 
s_add_i32 s51, s50, s49 // (op 35571) 


s_and_b32 s49, s1, s51 // (op 35571) 


s_add_i32 s1, s18, s49 // (op 35571) 


s_ashr_i32 s18, s1, s19 // (op 35571) 


s_lshl_b32 s1, s47, 24 // (op 35571) 
s_ashr_i32 s19, s1, 31 // (op 35571) 

s_xor_b32 s1, s18, s19 // (op 35571) 

s_sub_i32 s18, s1, s19 // (op 35571) 


s_add_u32 s50, s8, 256 // (op 35571) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 35571) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 35571) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 35571) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 35571) 

s_mul_i32 s1, s13, s50 // (op 35571) 

s_mul_i32 s19, s10, s1 // (op 35571) 

s_mul_i32 s1, s18, s19 // (op 35571) 

s_sub_u32 s19, s16, s1 // (op 35571) 

s_mov_b32 s1, s19 // (op 35571) 
s_mul_hi_i32 s49, s1, s14 // (op 35571) 
s_add_i32 s50, s49, s1 // (op 35571) 


s_ashr_i32 s1, s50, 31 // (op 35571) 
s_cmp_eq_i32 s14, 0 // (op 35571) 
s_mov_b32 s49, scc // (op 35571) 
s_cmp_eq_i32 s49, 0 // (op 35571) 
s_cselect_b32 s51, 0, -1 // (op 35571) 

s_and_b32 s49, s46, 31 // (op 35571) 
s_lshl_b32 s68, 1, s49 // (op 35571) 
s_add_i32 s69, s68, s51 // (op 35571) 


s_and_b32 s51, s1, s69 // (op 35571) 


s_add_i32 s1, s50, s51 // (op 35571) 


s_ashr_i32 s50, s1, s49 // (op 35571) 


s_lshl_b32 s1, s46, 24 // (op 35571) 
s_ashr_i32 s49, s1, 31 // (op 35571) 

s_xor_b32 s1, s50, s49 // (op 35571) 

s_sub_i32 s50, s1, s49 // (op 35571) 


s_mul_i32 s1, s48, s13 // (op 35571) 
s_add_u32 s68, s6, 256 // (op 35571) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 35571) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 35571) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 35571) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 8 // (op 35571) 

s_sub_i32 s49, s68, s1 // (op 35571) 

s_mul_i32 s51, s50, s49 // (op 35571) 


s_sub_u32 s49, s19, s51 // (op 35571) 


s_add_u32 s19, s49, s1 // (op 35571) 


s_cmp_eq_i32 s18, 0 // (op 35571) 
s_mov_b32 s1, scc // (op 35571) 

s_cmp_eq_i32 s1, 0 // (op 35571) 
s_cselect_b32 s18, s19, s0 // (op 35571) 



v_lshl_add_u32 v21, s18, 8, v16 // (op 35571) 


v_mul_lo_u32 v16, v20, s34 // (op 35571) 

v_add_u32 v20, v21, v16 // (op 35571) 


s_mul_i32 s1, s34, s39 // (op 35573) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 35573) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 35573) least significant
s_mul_hi_u32 s19, s34, s38 // (op 35573) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35573) most significant: sum
s_add_u32 s1, s1, s19 // (op 35573) most significant: sum


s_mov_b32 s64, s36 // (op 35573) reg expression
s_mov_b32 s65, s37 // (op 35573) reg expression
s_mov_b32 s66, s0 // (op 35573) reg expression
s_mov_b32 s67, 131072 // (op 35573) call()

s_mul_hi_i32 s0, s16, s12 // (op 35616) 
v_and_b32 v16, -64, v1 // (op 35616) 
s_add_i32 s1, s0, s16 // (op 35616) 

v_sub_u32 v21, v1, v16 // (op 35616) 

s_ashr_i32 s0, s1, 31 // (op 35616) 
v_lshlrev_b32 v16, 2, v21 // (op 35616) 

s_cmp_eq_i32 s12, 0 // (op 35616) 
s_mov_b32 s18, scc // (op 35616) 
v_lshrrev_b32 v21, 6, v1 // (op 35616) 
s_cmp_eq_i32 s18, 0 // (op 35616) 
s_cselect_b32 s19, 0, -1 // (op 35616) 

s_and_b32 s18, s15, 31 // (op 35616) 
s_lshl_b32 s49, 1, s18 // (op 35616) 
s_add_i32 s50, s49, s19 // (op 35616) 


s_and_b32 s19, s0, s50 // (op 35616) 


s_add_i32 s0, s1, s19 // (op 35616) 


s_ashr_i32 s1, s0, s18 // (op 35616) 


s_lshl_b32 s0, s15, 24 // (op 35616) 
s_ashr_i32 s18, s0, 31 // (op 35616) 

s_xor_b32 s0, s1, s18 // (op 35616) 

s_sub_i32 s1, s0, s18 // (op 35616) 


s_mul_hi_i32 s0, s1, s44 // (op 35616) 
s_add_i32 s18, s0, s1 // (op 35616) 

s_ashr_i32 s0, s18, 31 // (op 35616) 
s_cmp_eq_i32 s44, 0 // (op 35616) 
s_mov_b32 s19, scc // (op 35616) 
s_cmp_eq_i32 s19, 0 // (op 35616) 
s_cselect_b32 s49, 0, -1 // (op 35616) 

s_and_b32 s19, s45, 31 // (op 35616) 
s_lshl_b32 s50, 1, s19 // (op 35616) 
s_add_i32 s51, s50, s49 // (op 35616) 


s_and_b32 s49, s0, s51 // (op 35616) 


s_add_i32 s0, s18, s49 // (op 35616) 


s_ashr_i32 s18, s0, s19 // (op 35616) 


s_lshl_b32 s0, s45, 24 // (op 35616) 
s_ashr_i32 s19, s0, 31 // (op 35616) 

s_xor_b32 s0, s18, s19 // (op 35616) 

s_sub_i32 s18, s0, s19 // (op 35616) 


s_add_u32 s50, s8, 256 // (op 35616) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 35616) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 35616) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 35616) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 35616) 

s_mul_i32 s0, s18, s50 // (op 35616) 

s_sub_i32 s49, s1, s0 // (op 35616) 


s_mul_hi_i32 s0, s16, s11 // (op 35616) 
s_add_i32 s1, s0, s16 // (op 35616) 

s_ashr_i32 s0, s1, 31 // (op 35616) 
s_cmp_eq_i32 s11, 0 // (op 35616) 
s_mov_b32 s18, scc // (op 35616) 
s_cmp_eq_i32 s18, 0 // (op 35616) 
s_cselect_b32 s19, 0, -1 // (op 35616) 

s_and_b32 s18, s47, 31 // (op 35616) 
s_lshl_b32 s68, 1, s18 // (op 35616) 
s_add_i32 s69, s68, s19 // (op 35616) 


s_and_b32 s19, s0, s69 // (op 35616) 


s_add_i32 s0, s1, s19 // (op 35616) 


s_ashr_i32 s1, s0, s18 // (op 35616) 


s_lshl_b32 s0, s47, 24 // (op 35616) 
s_ashr_i32 s18, s0, 31 // (op 35616) 

s_xor_b32 s0, s1, s18 // (op 35616) 

s_sub_i32 s1, s0, s18 // (op 35616) 


s_mul_i32 s0, s13, s50 // (op 35616) 

s_mul_i32 s18, s10, s0 // (op 35616) 

s_mul_i32 s0, s1, s18 // (op 35616) 

s_sub_u32 s18, s16, s0 // (op 35616) 

s_mul_hi_i32 s0, s18, s14 // (op 35616) 
s_add_i32 s19, s0, s18 // (op 35616) 


s_ashr_i32 s0, s19, 31 // (op 35616) 
s_cmp_eq_i32 s14, 0 // (op 35616) 
s_mov_b32 s18, scc // (op 35616) 
s_cmp_eq_i32 s18, 0 // (op 35616) 
s_cselect_b32 s50, 0, -1 // (op 35616) 

s_and_b32 s18, s46, 31 // (op 35616) 
s_lshl_b32 s51, 1, s18 // (op 35616) 
s_add_i32 s68, s51, s50 // (op 35616) 


s_and_b32 s50, s0, s68 // (op 35616) 


s_add_i32 s0, s19, s50 // (op 35616) 


s_ashr_i32 s19, s0, s18 // (op 35616) 


s_lshl_b32 s0, s46, 24 // (op 35616) 
s_ashr_i32 s18, s0, 31 // (op 35616) 

s_xor_b32 s0, s19, s18 // (op 35616) 

s_sub_i32 s19, s0, s18 // (op 35616) 


s_cmp_eq_i32 s1, 0 // (op 35616) 
s_mov_b32 s0, scc // (op 35616) 

s_cmp_eq_i32 s0, 0 // (op 35616) 
s_cselect_b32 s1, s19, s49 // (op 35616) 



v_lshl_add_u32 v22, s1, 8, v16 // (op 35616) 


v_mul_lo_u32 v16, v21, s40 // (op 35616) 
v_add_u32 v16, v16, v22 // (op 35616) 


s_mul_i32 s1, s40, s39 // (op 35618) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 35618) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 35618) least significant
s_mul_hi_u32 s19, s40, s38 // (op 35618) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35618) most significant: sum
s_add_u32 s1, s1, s19 // (op 35618) most significant: sum


s_mov_b32 s68, s42 // (op 35618) reg expression
s_mov_b32 s69, s43 // (op 35618) reg expression
s_mov_b32 s70, s0 // (op 35618) reg expression
s_mov_b32 s71, 131072 // (op 35618) call()

s_mul_hi_i32 s0, s16, s12 // (op 35635) 
v_and_b32 v21, -64, v1 // (op 35635) 
s_add_i32 s1, s0, s16 // (op 35635) 

v_sub_u32 v22, v1, v21 // (op 35635) 

s_ashr_i32 s0, s1, 31 // (op 35635) 
v_lshlrev_b32 v21, 2, v22 // (op 35635) 

s_cmp_eq_i32 s12, 0 // (op 35635) 
s_mov_b32 s18, scc // (op 35635) 
v_lshrrev_b32 v22, 6, v1 // (op 35635) 
s_cmp_eq_i32 s18, 0 // (op 35635) 
s_cselect_b32 s19, 0, -1 // (op 35635) 

s_and_b32 s18, s15, 31 // (op 35635) 
s_lshl_b32 s49, 1, s18 // (op 35635) 
s_add_i32 s50, s49, s19 // (op 35635) 


s_and_b32 s19, s0, s50 // (op 35635) 


s_add_i32 s0, s1, s19 // (op 35635) 


s_ashr_i32 s1, s0, s18 // (op 35635) 


s_lshl_b32 s0, s15, 24 // (op 35635) 
s_ashr_i32 s18, s0, 31 // (op 35635) 

s_xor_b32 s0, s1, s18 // (op 35635) 

s_sub_i32 s1, s0, s18 // (op 35635) 


s_mul_hi_i32 s0, s1, s44 // (op 35635) 
s_add_i32 s18, s0, s1 // (op 35635) 

s_ashr_i32 s0, s18, 31 // (op 35635) 
s_cmp_eq_i32 s44, 0 // (op 35635) 
s_mov_b32 s19, scc // (op 35635) 
s_cmp_eq_i32 s19, 0 // (op 35635) 
s_cselect_b32 s49, 0, -1 // (op 35635) 

s_and_b32 s19, s45, 31 // (op 35635) 
s_lshl_b32 s50, 1, s19 // (op 35635) 
s_add_i32 s51, s50, s49 // (op 35635) 


s_and_b32 s49, s0, s51 // (op 35635) 


s_add_i32 s0, s18, s49 // (op 35635) 


s_ashr_i32 s18, s0, s19 // (op 35635) 


s_lshl_b32 s0, s45, 24 // (op 35635) 
s_ashr_i32 s19, s0, 31 // (op 35635) 

s_xor_b32 s0, s18, s19 // (op 35635) 

s_sub_i32 s18, s0, s19 // (op 35635) 


s_add_u32 s50, s8, 256 // (op 35635) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 35635) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 35635) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 35635) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 35635) 

s_mul_i32 s0, s18, s50 // (op 35635) 

s_sub_i32 s49, s1, s0 // (op 35635) 


s_mul_hi_i32 s0, s16, s11 // (op 35635) 
s_add_i32 s1, s0, s16 // (op 35635) 

s_ashr_i32 s0, s1, 31 // (op 35635) 
s_cmp_eq_i32 s11, 0 // (op 35635) 
s_mov_b32 s18, scc // (op 35635) 
s_cmp_eq_i32 s18, 0 // (op 35635) 
s_cselect_b32 s19, 0, -1 // (op 35635) 

s_and_b32 s18, s47, 31 // (op 35635) 
s_lshl_b32 s72, 1, s18 // (op 35635) 
s_add_i32 s73, s72, s19 // (op 35635) 


s_and_b32 s19, s0, s73 // (op 35635) 


s_add_i32 s0, s1, s19 // (op 35635) 


s_ashr_i32 s1, s0, s18 // (op 35635) 


s_lshl_b32 s0, s47, 24 // (op 35635) 
s_ashr_i32 s18, s0, 31 // (op 35635) 

s_xor_b32 s0, s1, s18 // (op 35635) 

s_sub_i32 s1, s0, s18 // (op 35635) 


s_mul_i32 s0, s13, s50 // (op 35635) 

s_mul_i32 s18, s10, s0 // (op 35635) 

s_mul_i32 s0, s1, s18 // (op 35635) 

s_sub_u32 s18, s16, s0 // (op 35635) 

s_mul_hi_i32 s0, s18, s14 // (op 35635) 
s_add_i32 s19, s0, s18 // (op 35635) 


s_ashr_i32 s0, s19, 31 // (op 35635) 
s_cmp_eq_i32 s14, 0 // (op 35635) 
s_mov_b32 s18, scc // (op 35635) 
s_cmp_eq_i32 s18, 0 // (op 35635) 
s_cselect_b32 s50, 0, -1 // (op 35635) 

s_and_b32 s18, s46, 31 // (op 35635) 
s_lshl_b32 s51, 1, s18 // (op 35635) 
s_add_i32 s72, s51, s50 // (op 35635) 


s_and_b32 s50, s0, s72 // (op 35635) 


s_add_i32 s0, s19, s50 // (op 35635) 


s_ashr_i32 s19, s0, s18 // (op 35635) 


s_lshl_b32 s0, s46, 24 // (op 35635) 
s_ashr_i32 s18, s0, 31 // (op 35635) 

s_xor_b32 s0, s19, s18 // (op 35635) 

s_sub_i32 s19, s0, s18 // (op 35635) 


s_cmp_eq_i32 s1, 0 // (op 35635) 
s_mov_b32 s0, scc // (op 35635) 

s_cmp_eq_i32 s0, 0 // (op 35635) 
s_cselect_b32 s1, s19, s49 // (op 35635) 



v_lshl_add_u32 v23, s1, 8, v21 // (op 35635) 


v_mul_lo_u32 v21, v22, s40 // (op 35635) 
v_add_u32 v21, v21, v23 // (op 35635) 


s_mul_i32 s1, s40, s39 // (op 35637) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 35637) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 35637) least significant
s_mul_hi_u32 s19, s40, s38 // (op 35637) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35637) most significant: sum
s_add_u32 s1, s1, s19 // (op 35637) most significant: sum


s_mov_b32 s68, s42 // (op 35637) reg expression
s_mov_b32 s69, s43 // (op 35637) reg expression
s_mov_b32 s70, s0 // (op 35637) reg expression
s_mov_b32 s71, 131072 // (op 35637) call()


s_mul_i32 s1, s31, 6 // (op 34020) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 34020) least significant
s_mul_hi_u32 s18, s30, 6 // (op 34020) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34020) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 34020) 


v_mov_b32 v24, v18 // (op 34020) reg expression
buffer_load_dwordx3 v[26:28], v24, s[52:55], 0 offen // (op 34020) Load value
buffer_load_dwordx3 v[30:32], v24, s[52:55], 0 offen offset:12 // (op 34020) Load value
buffer_load_dwordx3 v[34:36], v24, s[52:55], 0 offen offset:24 // (op 34020) Load value
buffer_load_dwordx3 v[38:40], v24, s[52:55], 0 offen offset:36 // (op 34020) Load value
v_add_u32 v24, v24, v22 // (op 34020) 
buffer_load_dwordx3 v[42:44], v24, s[52:55], 0 offen // (op 34020) Load value
buffer_load_dwordx3 v[46:48], v24, s[52:55], 0 offen offset:12 // (op 34020) Load value
buffer_load_dwordx3 v[50:52], v24, s[52:55], 0 offen offset:24 // (op 34020) Load value
buffer_load_dwordx3 v[54:56], v24, s[52:55], 0 offen offset:36 // (op 34020) Load value




v_mov_b32 v29, v16 // (op 34023) reg expression

buffer_load_dword v33, v29, s[68:71], 0 offen // (op 34023) Load value


s_mul_i32 s1, s25, 6 // (op 34026) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 34026) least significant
s_mul_hi_u32 s18, s24, 6 // (op 34026) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34026) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 34026) 


v_mov_b32 v29, v14 // (op 34026) reg expression
buffer_load_dwordx3 v[58:60], v29, s[20:23], 0 offen // (op 34026) Load value
buffer_load_dwordx3 v[62:64], v29, s[20:23], 0 offen offset:12 // (op 34026) Load value
buffer_load_dwordx3 v[66:68], v29, s[20:23], 0 offen offset:24 // (op 34026) Load value
buffer_load_dwordx3 v[70:72], v29, s[20:23], 0 offen offset:36 // (op 34026) Load value
v_add_u32 v29, v29, v22 // (op 34026) 
buffer_load_dwordx3 v[74:76], v29, s[20:23], 0 offen // (op 34026) Load value
buffer_load_dwordx3 v[78:80], v29, s[20:23], 0 offen offset:12 // (op 34026) Load value
buffer_load_dwordx3 v[82:84], v29, s[20:23], 0 offen offset:24 // (op 34026) Load value
buffer_load_dwordx3 v[86:88], v29, s[20:23], 0 offen offset:36 // (op 34026) Load value




v_mov_b32 v29, v17 // (op 34029) reg expression

buffer_load_dword v37, v29, s[64:67], 0 offen // (op 34029) Load value


s_mul_i32 s1, s31, 6 // (op 34032) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 34032) least significant
s_mul_hi_u32 s18, s30, 6 // (op 34032) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34032) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 34032) 


s_lshl_b64 s[0:1], s[30:31], 7 // (op 34032) 
s_mul_i32 s19, s1, 6 // (op 34032) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 34032) least significant
s_mul_hi_u32 s49, s0, 6 // (op 34032) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 34032) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 34032) 

v_add_u32 v29, v18, s0 // (op 34032) 

buffer_load_dwordx3 v[90:92], v29, s[52:55], 0 offen // (op 34032) Load value
buffer_load_dwordx3 v[94:96], v29, s[52:55], 0 offen offset:12 // (op 34032) Load value
buffer_load_dwordx3 v[98:100], v29, s[52:55], 0 offen offset:24 // (op 34032) Load value
buffer_load_dwordx3 v[102:104], v29, s[52:55], 0 offen offset:36 // (op 34032) Load value
v_add_u32 v29, v29, v22 // (op 34032) 
buffer_load_dwordx3 v[106:108], v29, s[52:55], 0 offen // (op 34032) Load value
buffer_load_dwordx3 v[110:112], v29, s[52:55], 0 offen offset:12 // (op 34032) Load value
buffer_load_dwordx3 v[114:116], v29, s[52:55], 0 offen offset:24 // (op 34032) Load value
buffer_load_dwordx3 v[118:120], v29, s[52:55], 0 offen offset:36 // (op 34032) Load value





s_lshl_b32 s0, s40, 2 // (op 34035) 
v_add_u32 v18, v16, s0 // (op 34035) 


buffer_load_dword v29, v18, s[68:71], 0 offen // (op 34035) Load value



s_mul_i32 s1, s25, 6 // (op 34038) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 34038) least significant
s_mul_hi_u32 s18, s24, 6 // (op 34038) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34038) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 34038) 


s_lshl_b64 s[0:1], s[24:25], 7 // (op 34038) 
s_mul_i32 s19, s1, 6 // (op 34038) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 34038) least significant
s_mul_hi_u32 s49, s0, 6 // (op 34038) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 34038) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 34038) 

v_add_u32 v16, v14, s0 // (op 34038) 

buffer_load_dwordx3 v[122:124], v16, s[20:23], 0 offen // (op 34038) Load value
buffer_load_dwordx3 v[126:128], v16, s[20:23], 0 offen offset:12 // (op 34038) Load value
buffer_load_dwordx3 v[130:132], v16, s[20:23], 0 offen offset:24 // (op 34038) Load value
buffer_load_dwordx3 v[134:136], v16, s[20:23], 0 offen offset:36 // (op 34038) Load value
v_add_u32 v16, v16, v22 // (op 34038) 
buffer_load_dwordx3 v[138:140], v16, s[20:23], 0 offen // (op 34038) Load value
buffer_load_dwordx3 v[142:144], v16, s[20:23], 0 offen offset:12 // (op 34038) Load value
buffer_load_dwordx3 v[146:148], v16, s[20:23], 0 offen offset:24 // (op 34038) Load value
buffer_load_dwordx3 v[150:152], v16, s[20:23], 0 offen offset:36 // (op 34038) Load value





s_lshl_b32 s0, s34, 2 // (op 34041) 
v_add_u32 v14, v17, s0 // (op 34041) 


buffer_load_dword v16, v14, s[64:67], 0 offen // (op 34041) Load value


v_and_b32 v14, -4, v1 // (op 35765) 
v_sub_u32 v41, v1, v14 // (op 35765) 

v_lshlrev_b32 v14, 6, v41 // (op 35765) 

v_lshrrev_b32 v41, 2, v1 // (op 35765) 
v_lshl_add_u32 v45, v41, 9, v14 // (op 35765) 


v_mul_lo_u32 v14, v45, 6 // (op 35765) 
v_lshrrev_b32 v41, 3, v14 // (op 35765) 

v_lshrrev_b32 v14, 4, v45 // (op 35765) 

v_lshlrev_b32 v45, 2, v14 // (op 35765) 

v_add_u32 v14, v41, v45 // (op 35765) 





s_waitcnt vmcnt(35)
ds_write_b96 v14, v[26:28] // (op 34043) Store local data 
s_waitcnt vmcnt(34)
ds_write_b96 v14, v[30:32] offset:16 // (op 34043) Store local data 
s_waitcnt vmcnt(33)
ds_write_b96 v14, v[34:36] offset:32 // (op 34043) Store local data 
s_waitcnt vmcnt(32)
ds_write_b96 v14, v[38:40] offset:48 // (op 34043) Store local data 
s_waitcnt vmcnt(31)
ds_write_b96 v14, v[42:44] offset:256 // (op 34043) Store local data 
s_waitcnt vmcnt(30)
ds_write_b96 v14, v[46:48] offset:272 // (op 34043) Store local data 
s_waitcnt vmcnt(29)
ds_write_b96 v14, v[50:52] offset:288 // (op 34043) Store local data 
s_waitcnt vmcnt(28)
ds_write_b96 v14, v[54:56] offset:304 // (op 34043) Store local data 

v_and_b32 v14, -64, v1 // (op 35881) 
v_sub_u32 v41, v1, v14 // (op 35881) 

v_lshlrev_b32 v14, 2, v41 // (op 35881) 

v_lshrrev_b32 v41, 6, v1 // (op 35881) 
v_lshl_add_u32 v45, v41, 8, v14 // (op 35881) 





s_waitcnt vmcnt(27)
ds_write_b32 v45, v33 offset:32768 // (op 34044) Store local data 

v_and_b32 v14, -4, v1 // (op 35707) 
v_sub_u32 v41, v1, v14 // (op 35707) 

v_lshlrev_b32 v14, 6, v41 // (op 35707) 

v_lshrrev_b32 v41, 2, v1 // (op 35707) 
v_lshl_add_u32 v45, v41, 9, v14 // (op 35707) 


v_mul_lo_u32 v14, v45, 6 // (op 35707) 
v_lshrrev_b32 v41, 3, v14 // (op 35707) 

v_lshrrev_b32 v14, 4, v45 // (op 35707) 

v_lshlrev_b32 v45, 2, v14 // (op 35707) 

v_add_u32 v14, v41, v45 // (op 35707) 





s_waitcnt vmcnt(26)
ds_write_b96 v14, v[58:60] offset:33792 // (op 34045) Store local data 
s_waitcnt vmcnt(25)
ds_write_b96 v14, v[62:64] offset:33808 // (op 34045) Store local data 
s_waitcnt vmcnt(24)
ds_write_b96 v14, v[66:68] offset:33824 // (op 34045) Store local data 
s_waitcnt vmcnt(23)
ds_write_b96 v14, v[70:72] offset:33840 // (op 34045) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v14, v[74:76] offset:34048 // (op 34045) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v14, v[78:80] offset:34064 // (op 34045) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v14, v[82:84] offset:34080 // (op 34045) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v14, v[86:88] offset:34096 // (op 34045) Store local data 

v_and_b32 v14, -64, v1 // (op 35823) 
v_sub_u32 v41, v1, v14 // (op 35823) 

v_lshlrev_b32 v14, 2, v41 // (op 35823) 

v_lshrrev_b32 v41, 6, v1 // (op 35823) 
v_lshl_add_u32 v45, v41, 8, v14 // (op 35823) 





v_add_u32 v14, 66560, v45 // (op 34046) 
s_waitcnt vmcnt(18)
ds_write_b32 v14, v37 // (op 34046) Store local data 


v_and_b32 v14, -4, v1 // (op 35678) 
v_sub_u32 v41, v1, v14 // (op 35678) 

v_lshlrev_b32 v14, 6, v41 // (op 35678) 

v_lshrrev_b32 v41, 2, v1 // (op 35678) 
v_lshl_add_u32 v45, v41, 9, v14 // (op 35678) 


v_mul_lo_u32 v14, v45, 6 // (op 35678) 
v_lshrrev_b32 v41, 3, v14 // (op 35678) 

v_lshrrev_b32 v14, 4, v45 // (op 35678) 

v_lshlrev_b32 v45, 2, v14 // (op 35678) 

v_add_u32 v14, v41, v45 // (op 35678) 


v_and_b32 v41, -4, v1 // (op 35736) 
v_sub_u32 v45, v1, v41 // (op 35736) 

v_lshlrev_b32 v41, 6, v45 // (op 35736) 

v_lshrrev_b32 v45, 2, v1 // (op 35736) 
v_lshl_add_u32 v49, v45, 9, v41 // (op 35736) 


v_mul_lo_u32 v41, v49, 6 // (op 35736) 
v_lshrrev_b32 v45, 3, v41 // (op 35736) 

v_lshrrev_b32 v41, 4, v49 // (op 35736) 

v_lshlrev_b32 v49, 2, v41 // (op 35736) 

v_add_u32 v41, v45, v49 // (op 35736) 


v_and_b32 v45, -64, v1 // (op 35794) 
v_sub_u32 v49, v1, v45 // (op 35794) 

v_lshlrev_b32 v45, 2, v49 // (op 35794) 

v_lshrrev_b32 v49, 6, v1 // (op 35794) 
v_lshl_add_u32 v53, v49, 8, v45 // (op 35794) 


v_and_b32 v45, -64, v1 // (op 35852) 
v_sub_u32 v49, v1, v45 // (op 35852) 

v_lshlrev_b32 v45, 2, v49 // (op 35852) 

v_lshrrev_b32 v49, 6, v1 // (op 35852) 
v_lshl_add_u32 v57, v49, 8, v45 // (op 35852) 


v_and_b32 v45, -4, v1 // (op 35933) 
v_sub_u32 v49, v1, v45 // (op 35933) 

v_lshlrev_b32 v45, 6, v49 // (op 35933) 

v_lshrrev_b32 v49, 2, v1 // (op 35933) 
v_lshl_add_u32 v61, v49, 9, v45 // (op 35933) 


v_mul_lo_u32 v45, v61, 6 // (op 35933) 
v_lshrrev_b32 v49, 3, v45 // (op 35933) 

v_lshrrev_b32 v45, 4, v61 // (op 35933) 

v_lshlrev_b32 v61, 2, v45 // (op 35933) 

v_add_u32 v45, v49, v61 // (op 35933) 


v_and_b32 v49, -4, v1 // (op 35962) 
v_sub_u32 v61, v1, v49 // (op 35962) 

v_lshlrev_b32 v49, 6, v61 // (op 35962) 

v_lshrrev_b32 v61, 2, v1 // (op 35962) 
v_lshl_add_u32 v65, v61, 9, v49 // (op 35962) 


v_mul_lo_u32 v49, v65, 6 // (op 35962) 
v_lshrrev_b32 v61, 3, v49 // (op 35962) 

v_lshrrev_b32 v49, 4, v65 // (op 35962) 

v_lshlrev_b32 v65, 2, v49 // (op 35962) 

v_add_u32 v49, v61, v65 // (op 35962) 


v_and_b32 v61, -64, v1 // (op 35991) 
v_sub_u32 v65, v1, v61 // (op 35991) 

v_lshlrev_b32 v61, 2, v65 // (op 35991) 

v_lshrrev_b32 v65, 6, v1 // (op 35991) 
v_lshl_add_u32 v69, v65, 8, v61 // (op 35991) 


v_and_b32 v61, -64, v1 // (op 36020) 
v_sub_u32 v65, v1, v61 // (op 36020) 

v_lshlrev_b32 v61, 2, v65 // (op 36020) 

v_lshrrev_b32 v65, 6, v1 // (op 36020) 
v_lshl_add_u32 v73, v65, 8, v61 // (op 36020) 


s_mov_b32 s0, 0 // (op 9653) call()
s_ashr_i64 s[18:19], s[28:29], 7 // (op 9652) 
s_mov_b32 s1, s18 // (op 9652) convert

s_and_b32 s18, s1, 4294967294 // (op 9652) 

s_cmp_lt_u32 s0, s18 // (op 9652) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_9652 // (op 9652) Condition: Top (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_9652 if false)
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_9652:

s_barrier  // (op 36268) 
s_add_u32 s0, s0, 2 // (op 9654) 



s_mul_i32 s19, s31, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s30, 6 // (op 10) least significant
s_mul_hi_u32 s1, s30, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 10) 


v_mov_b32 v61, v19 // (op 10) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 10) 
s_mul_i32 s51, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 10) least significant
s_mul_hi_u32 s1, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v65, v61, s1 // (op 10) 

buffer_load_dwordx3 v[26:28], v65, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[30:32], v65, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[34:36], v65, s[52:55], 0 offen offset:24 // (op 10) Load value
ds_read_b96_tr_b6 v[154:156], v0 offset:33792 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[157:159], v0 offset:37888 // (op 70) Transpose load local data 
buffer_load_dwordx3 v[38:40], v65, s[52:55], 0 offen offset:36 // (op 10) Load value
s_barrier  // (op 36281) 
v_add_u32 v65, v65, v22 // (op 10) 



buffer_load_dwordx3 v[42:44], v65, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[46:48], v65, s[52:55], 0 offen offset:12 // (op 10) Load value
ds_read_b96_tr_b6 v[160:162], v3 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[163:165], v3 offset:4096 // (op 75) Transpose load local data 
buffer_load_dwordx3 v[50:52], v65, s[52:55], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 36295) 
buffer_load_dwordx3 v[54:56], v65, s[52:55], 0 offen offset:36 // (op 10) Load value






v_add_u32 v65, 66560, v4 // (op 80) 
ds_read_u8 v77, v65 // (op 80) Load local data 

v_mov_b32 v65, v21 // (op 25) 
s_barrier  // (op 36309) 
s_lshl_b32 s1, s40, 3 // (op 25) 


ds_read_u8 v81, v5 offset:32768 // (op 85) Load local data 
v_add_u32 v85, v65, s1 // (op 25) 





buffer_load_dword v33, v85, s[68:71], 0 offen // (op 25) Load value

ds_read_b96_tr_b6 v[166:168], v3 offset:32 // (op 371) Transpose load local data 

s_mul_i32 s19, s25, 6 // (op 4) most significant: low of high * low
s_mul_i32 s18, s24, 6 // (op 4) least significant
s_mul_hi_u32 s1, s24, 6 // (op 4) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 4) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 4) 


v_mov_b32 v85, v15 // (op 4) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 4) 
s_mul_i32 s51, s19, 6 // (op 4) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 4) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v89, v85, s1 // (op 4) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[154:159], v[160:165], a[0:3], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 88) 
ds_read_b96_tr_b6 v[169:171], v3 offset:4128 // (op 371) Transpose load local data 
buffer_load_dwordx3 v[58:60], v89, s[20:23], 0 offen // (op 4) Load value


ds_read_u8 v93, v5 offset:32800 // (op 375) Load local data 
buffer_load_dwordx3 v[62:64], v89, s[20:23], 0 offen offset:12 // (op 4) Load value



ds_read_b96_tr_b6 v[172:174], v3 offset:64 // (op 547) Transpose load local data 
ds_read_b96_tr_b6 v[175:177], v3 offset:4160 // (op 547) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[154:159], v[166:171], a[4:7], v77, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 376) 


ds_read_u8 v97, v5 offset:32832 // (op 551) Load local data 
buffer_load_dwordx3 v[66:68], v89, s[20:23], 0 offen offset:24 // (op 4) Load value



ds_read_b96_tr_b6 v[178:180], v3 offset:96 // (op 723) Transpose load local data 
ds_read_b96_tr_b6 v[181:183], v3 offset:4192 // (op 723) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[154:159], v[172:177], a[8:11], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 552) 


ds_read_u8 v101, v5 offset:32864 // (op 727) Load local data 
buffer_load_dwordx3 v[70:72], v89, s[20:23], 0 offen offset:36 // (op 4) Load value



v_add_u32 v89, v89, v22 // (op 4) 
ds_read_b96_tr_b6 v[184:186], v3 offset:128 // (op 899) Transpose load local data 
ds_read_b96_tr_b6 v[187:189], v3 offset:4224 // (op 899) Transpose load local data 
buffer_load_dwordx3 v[74:76], v89, s[20:23], 0 offen // (op 4) Load value


ds_read_u8 v105, v5 offset:32896 // (op 903) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[154:159], v[178:183], a[12:15], v77, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 728) 



buffer_load_dwordx3 v[78:80], v89, s[20:23], 0 offen offset:12 // (op 4) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[154:159], v[184:189], a[16:19], v77, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 904) 
ds_read_b96_tr_b6 v[190:192], v3 offset:160 // (op 1075) Transpose load local data 
ds_read_b96_tr_b6 v[193:195], v3 offset:4256 // (op 1075) Transpose load local data 
buffer_load_dwordx3 v[82:84], v89, s[20:23], 0 offen offset:24 // (op 4) Load value


ds_read_u8 v109, v5 offset:32928 // (op 1079) Load local data 
buffer_load_dwordx3 v[86:88], v89, s[20:23], 0 offen offset:36 // (op 4) Load value







v_mov_b32 v89, v20 // (op 17) 
s_lshl_b32 s1, s34, 3 // (op 17) 
v_add_u32 v113, v89, s1 // (op 17) 


ds_read_b96_tr_b6 v[196:198], v3 offset:192 // (op 1251) Transpose load local data 
ds_read_b96_tr_b6 v[199:201], v3 offset:4288 // (op 1251) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[154:159], v[190:195], a[20:23], v77, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1080) 


ds_read_u8 v117, v5 offset:32960 // (op 1255) Load local data 
buffer_load_dword v37, v113, s[64:67], 0 offen // (op 17) Load value




s_barrier  // (op 36332) 
ds_read_b96_tr_b6 v[202:204], v3 offset:224 // (op 1427) Transpose load local data 



ds_read_b96_tr_b6 v[205:207], v3 offset:4320 // (op 1427) Transpose load local data 
v_add_u32 v113, 67584, v49 // (op 253) 


s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[154:159], v[196:201], a[24:27], v77, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1256) 
ds_read_u8 v121, v5 offset:32992 // (op 1431) Load local data 
s_waitcnt vmcnt(35)
ds_write_b96 v113, v[90:92] // (op 253) Store local data 




v_add_u32 v113, 67600, v49 // (op 253) 
ds_read_b96_tr_b6 v[208:210], v0 offset:33824 // (op 1996) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[154:159], v[202:207], a[28:31], v77, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1432) 
ds_read_b96_tr_b6 v[211:213], v0 offset:37920 // (op 1996) Transpose load local data 




v_add_u32 v77, 66592, v4 // (op 1999) 
ds_read_u8 v125, v77 // (op 1999) Load local data 

s_waitcnt vmcnt(34)
ds_write_b96 v113, v[94:96] // (op 253) Store local data 




v_add_u32 v77, 67616, v49 // (op 253) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[208:213], v[160:165], a[32:35], v125, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2323) 
ds_read_b96_tr_b6 v[154:156], v0 offset:33856 // (op 3534) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[208:213], v[166:171], a[36:39], v125, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2277) 
ds_read_b96_tr_b6 v[157:159], v0 offset:37952 // (op 3534) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[208:213], v[172:177], a[40:43], v125, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2231) 
v_add_u32 v113, 66624, v4 // (op 3537) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[208:213], v[178:183], a[44:47], v125, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2185) 
ds_read_u8 v129, v113 // (op 3537) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[208:213], v[184:189], a[48:51], v125, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2139) 
ds_read_b96_tr_b6 v[214:216], v0 offset:33888 // (op 5072) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[208:213], v[190:195], a[52:55], v125, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2093) 
ds_read_b96_tr_b6 v[217:219], v0 offset:37984 // (op 5072) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[208:213], v[196:201], a[56:59], v125, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2047) 
v_add_u32 v113, 66656, v4 // (op 5075) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[208:213], v[202:207], a[60:63], v125, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2001) 
ds_read_u8 v133, v113 // (op 5075) Load local data 






s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[154:159], v[160:165], a[64:67], v129, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3861) 
ds_read_b96_tr_b6 v[208:210], v0 offset:33920 // (op 6610) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[154:159], v[166:171], a[68:71], v129, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3815) 
ds_read_b96_tr_b6 v[211:213], v0 offset:38016 // (op 6610) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[154:159], v[172:177], a[72:75], v129, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3769) 
v_add_u32 v113, 66688, v4 // (op 6613) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[178:183], a[76:79], v129, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3723) 
ds_read_u8 v125, v113 // (op 6613) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[154:159], v[184:189], a[80:83], v129, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3677) 
ds_read_b96_tr_b6 v[220:222], v0 offset:33952 // (op 8148) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[154:159], v[190:195], a[84:87], v129, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3631) 
ds_read_b96_tr_b6 v[223:225], v0 offset:38048 // (op 8148) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[154:159], v[196:201], a[88:91], v129, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3585) 
v_add_u32 v113, 66720, v4 // (op 8151) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[154:159], v[202:207], a[92:95], v129, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3539) 
ds_read_u8 v137, v113 // (op 8151) Load local data 






s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[214:219], v[160:165], a[96:99], v133, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5399) 
ds_read_b96_tr_b6 v[154:156], v0 offset:33984 // (op 9686) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[214:219], v[166:171], a[100:103], v133, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5353) 
ds_read_b96_tr_b6 v[157:159], v0 offset:38080 // (op 9686) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[214:219], v[172:177], a[104:107], v133, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5307) 
v_add_u32 v113, 66752, v4 // (op 9689) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[214:219], v[178:183], a[108:111], v133, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5261) 
ds_read_u8 v129, v113 // (op 9689) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[214:219], v[184:189], a[112:115], v133, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5215) 
ds_read_b96_tr_b6 v[226:228], v0 offset:34016 // (op 11224) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[214:219], v[190:195], a[116:119], v133, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5169) 
ds_read_b96_tr_b6 v[229:231], v0 offset:38112 // (op 11224) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[214:219], v[196:201], a[120:123], v133, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5123) 
v_add_u32 v113, 66784, v4 // (op 11227) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[214:219], v[202:207], a[124:127], v133, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5077) 
ds_read_u8 v141, v113 // (op 11227) Load local data 



s_waitcnt vmcnt(33)
ds_write_b96 v77, v[98:100] // (op 253) Store local data 

s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[208:213], v[160:165], a[128:131], v125, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6937) 
v_add_u32 v77, 67632, v49 // (op 253) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[208:213], v[166:171], a[132:135], v125, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6891) 
s_waitcnt vmcnt(32)
ds_write_b96 v77, v[102:104] // (op 253) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[208:213], v[172:177], a[136:139], v125, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6845) 
v_add_u32 v77, 67840, v49 // (op 253) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[208:213], v[178:183], a[140:143], v125, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6799) 
s_waitcnt vmcnt(31)
ds_write_b96 v77, v[106:108] // (op 253) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[208:213], v[184:189], a[144:147], v125, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6753) 
v_add_u32 v77, 67856, v49 // (op 253) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[208:213], v[190:195], a[148:151], v125, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6707) 
s_waitcnt vmcnt(30)
ds_write_b96 v77, v[110:112] // (op 253) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[208:213], v[196:201], a[152:155], v125, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6661) 
v_add_u32 v77, 67872, v49 // (op 253) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[208:213], v[202:207], a[156:159], v125, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6615) 
s_waitcnt vmcnt(29)
ds_write_b96 v77, v[114:116] // (op 253) Store local data 



v_add_u32 v77, 67888, v49 // (op 253) 
s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[220:225], v[160:165], a[160:163], v137, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8475) 
s_waitcnt vmcnt(28)
ds_write_b96 v77, v[118:120] // (op 253) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[220:225], v[166:171], a[164:167], v137, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8429) 
s_barrier  // (op 36344) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[220:225], v[172:177], a[168:171], v137, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8383) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[220:225], v[178:183], a[172:175], v137, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8337) 
v_add_u32 v77, 100352, v73 // (op 255) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[220:225], v[184:189], a[176:179], v137, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8291) 
s_waitcnt vmcnt(27)
ds_write_b32 v77, v29 // (op 255) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[220:225], v[190:195], a[180:183], v137, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8245) 
s_barrier  // (op 36326) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[220:225], v[196:201], a[184:187], v137, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8199) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[220:225], v[202:207], a[188:191], v137, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8153) 
v_add_u32 v77, 101376, v45 // (op 243) 


s_waitcnt vmcnt(26)
ds_write_b96 v77, v[122:124] // (op 243) Store local data 

s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[154:159], v[160:165], a[192:195], v129, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 10013) 
v_add_u32 v77, 101392, v45 // (op 243) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[154:159], v[166:171], a[196:199], v129, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9967) 
s_waitcnt vmcnt(25)
ds_write_b96 v77, v[126:128] // (op 243) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[154:159], v[172:177], a[200:203], v129, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9921) 
v_add_u32 v77, 101408, v45 // (op 243) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[154:159], v[178:183], a[204:207], v129, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9875) 
s_waitcnt vmcnt(24)
ds_write_b96 v77, v[130:132] // (op 243) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[154:159], v[184:189], a[208:211], v129, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9829) 
v_add_u32 v77, 101424, v45 // (op 243) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[154:159], v[190:195], a[212:215], v129, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9783) 
s_waitcnt vmcnt(23)
ds_write_b96 v77, v[134:136] // (op 243) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[154:159], v[196:201], a[216:219], v129, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9737) 
v_add_u32 v77, 101632, v45 // (op 243) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[154:159], v[202:207], a[220:223], v129, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9691) 
s_waitcnt vmcnt(22)
ds_write_b96 v77, v[138:140] // (op 243) Store local data 



v_add_u32 v77, 101648, v45 // (op 243) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[226:231], v[160:165], a[224:227], v141, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11551) 
s_waitcnt vmcnt(21)
ds_write_b96 v77, v[142:144] // (op 243) Store local data 



v_add_u32 v77, 101664, v45 // (op 243) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[226:231], v[166:171], a[228:231], v141, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11505) 
s_waitcnt vmcnt(20)
ds_write_b96 v77, v[146:148] // (op 243) Store local data 



v_add_u32 v77, 101680, v45 // (op 243) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[226:231], v[172:177], a[232:235], v141, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11459) 
s_waitcnt vmcnt(19)
ds_write_b96 v77, v[150:152] // (op 243) Store local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[226:231], v[178:183], a[236:239], v141, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11413) 
s_barrier  // (op 36338) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[226:231], v[184:189], a[240:243], v141, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11367) 
v_add_u32 v77, 134144, v69 // (op 254) 


s_waitcnt vmcnt(18)
ds_write_b32 v77, v16 // (op 254) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[226:231], v[190:195], a[244:247], v141, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11321) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[226:231], v[196:201], a[248:251], v141, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11275) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[226:231], v[202:207], a[252:255], v141, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11229) 





s_mul_i32 s19, s31, 6 // (op 256) most significant: low of high * low
s_mul_i32 s18, s30, 6 // (op 256) least significant
s_mul_hi_u32 s1, s30, 6 // (op 256) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 256) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 256) 


s_lshl_b64 s[18:19], s[30:31], 7 // (op 256) 
s_mul_i32 s51, s19, 6 // (op 256) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 256) least significant
s_mul_hi_u32 s1, s18, 6 // (op 256) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 256) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 256) 

s_mul_i32 s1, 3, s18 // (op 256) 

v_add_u32 v77, v61, s1 // (op 256) 

buffer_load_dwordx3 v[90:92], v77, s[52:55], 0 offen // (op 256) Load value
buffer_load_dwordx3 v[94:96], v77, s[52:55], 0 offen offset:12 // (op 256) Load value
buffer_load_dwordx3 v[98:100], v77, s[52:55], 0 offen offset:24 // (op 256) Load value
s_waitcnt lgkmcnt(1)
s_barrier  // (op 36323) 
buffer_load_dwordx3 v[102:104], v77, s[52:55], 0 offen offset:36 // (op 256) Load value



v_add_u32 v77, v77, v22 // (op 256) 
buffer_load_dwordx3 v[106:108], v77, s[52:55], 0 offen // (op 256) Load value
buffer_load_dwordx3 v[110:112], v77, s[52:55], 0 offen offset:12 // (op 256) Load value
v_add_u32 v81, 101376, v6 // (op 245) 
ds_read_b96_tr_b6 v[154:156], v81 // (op 245) Transpose load local data 

v_add_u32 v81, 105472, v6 // (op 245) 
ds_read_b96_tr_b6 v[157:159], v81 // (op 245) Transpose load local data 

buffer_load_dwordx3 v[114:116], v77, s[52:55], 0 offen offset:24 // (op 256) Load value
s_barrier  // (op 36329) 
buffer_load_dwordx3 v[118:120], v77, s[52:55], 0 offen offset:36 // (op 256) Load value






v_add_u32 v61, 67584, v7 // (op 247) 


s_lshl_b32 s1, s40, 2 // (op 258) 
s_mul_i32 s18, 3, s1 // (op 258) 

v_add_u32 v77, v65, s18 // (op 258) 


buffer_load_dword v29, v77, s[68:71], 0 offen // (op 258) Load value

s_lshl_b64 s[18:19], s[30:31], 7 // (op 35499) 

s_mul_i32 s51, s19, 6 // (op 35499) most significant: low of high * low

s_mul_i32 s50, s18, 6 // (op 35499) least significant
s_mul_hi_u32 s1, s18, 6 // (op 35499) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 35499) most significant: sum


s_lshl_b32 s1, s40, 3 // (op 35627) 
v_add_u32 v21, v21, s1 // (op 35627) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 35499) 

s_lshl_b32 s1, s18, 1 // (op 35499) 

v_add_u32 v19, v19, s1 // (op 35499) 

s_mul_i32 s19, s25, 6 // (op 242) most significant: low of high * low
s_mul_i32 s18, s24, 6 // (op 242) least significant
s_mul_hi_u32 s1, s24, 6 // (op 242) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 242) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 242) 


s_lshl_b64 s[18:19], s[24:25], 7 // (op 242) 
s_mul_i32 s51, s19, 6 // (op 242) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 242) least significant
s_mul_hi_u32 s1, s18, 6 // (op 242) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 242) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 242) 

s_mul_i32 s1, 3, s18 // (op 242) 

v_add_u32 v65, v85, s1 // (op 242) 

ds_read_b96_tr_b6 v[160:162], v61 // (op 247) Transpose load local data 

v_add_u32 v61, 71680, v7 // (op 247) 
ds_read_b96_tr_b6 v[163:165], v61 // (op 247) Transpose load local data 

buffer_load_dwordx3 v[122:124], v65, s[20:23], 0 offen // (op 242) Load value
buffer_load_dwordx3 v[126:128], v65, s[20:23], 0 offen offset:12 // (op 242) Load value
buffer_load_dwordx3 v[130:132], v65, s[20:23], 0 offen offset:24 // (op 242) Load value
s_waitcnt lgkmcnt(4)
s_barrier  // (op 36335) 
buffer_load_dwordx3 v[134:136], v65, s[20:23], 0 offen offset:36 // (op 242) Load value


v_add_u32 v65, v65, v22 // (op 242) 
buffer_load_dwordx3 v[138:140], v65, s[20:23], 0 offen // (op 242) Load value
v_add_u32 v61, 134144, v8 // (op 249) 
ds_read_u8 v77, v61 // (op 249) Load local data 

buffer_load_dwordx3 v[142:144], v65, s[20:23], 0 offen offset:12 // (op 242) Load value
s_barrier  // (op 36341) 
buffer_load_dwordx3 v[146:148], v65, s[20:23], 0 offen offset:24 // (op 242) Load value


v_add_u32 v61, 100352, v9 // (op 251) 
buffer_load_dwordx3 v[150:152], v65, s[20:23], 0 offen offset:36 // (op 242) Load value


ds_read_u8 v65, v61 // (op 251) Load local data 







s_lshl_b32 s1, s34, 2 // (op 257) 
s_mul_i32 s18, 3, s1 // (op 257) 

v_add_u32 v61, 67616, v7 // (op 381) 
v_add_u32 v81, v89, s18 // (op 257) 


s_lshl_b64 s[18:19], s[24:25], 7 // (op 35435) 
s_mul_i32 s51, s19, 6 // (op 35435) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 35435) least significant
s_mul_hi_u32 s1, s18, 6 // (op 35435) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 35435) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 35435) 

s_lshl_b32 s1, s18, 1 // (op 35435) 

v_add_u32 v15, v15, s1 // (op 35435) 

ds_read_b96_tr_b6 v[166:168], v61 // (op 381) Transpose load local data 

v_add_u32 v61, 71712, v7 // (op 381) 
ds_read_b96_tr_b6 v[169:171], v61 // (op 381) Transpose load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[154:159], v[160:165], a[0:3], v77, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 252) 


v_add_u32 v61, 100384, v9 // (op 385) 
ds_read_u8 v85, v61 // (op 385) Load local data 

buffer_load_dword v16, v81, s[64:67], 0 offen // (op 257) Load value





v_add_u32 v61, 67648, v7 // (op 557) 
s_barrier  // (op 36284) 
s_lshl_b32 s1, s34, 3 // (op 35563) 



v_add_u32 v20, v20, s1 // (op 35563) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[154:159], v[166:171], a[4:7], v77, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 377) 
ds_read_b96_tr_b6 v[172:174], v61 // (op 557) Transpose load local data 

v_add_u32 v61, 71744, v7 // (op 557) 
ds_read_b96_tr_b6 v[175:177], v61 // (op 557) Transpose load local data 

s_waitcnt vmcnt(35)
ds_write_b96 v41, v[26:28] // (op 13) Store local data 


v_add_u32 v61, 100416, v9 // (op 561) 
ds_read_u8 v81, v61 // (op 561) Load local data 

s_waitcnt vmcnt(34)
ds_write_b96 v41, v[30:32] offset:16 // (op 13) Store local data 



v_add_u32 v61, 67680, v7 // (op 733) 
ds_read_b96_tr_b6 v[178:180], v61 // (op 733) Transpose load local data 

v_add_u32 v61, 71776, v7 // (op 733) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[154:159], v[172:177], a[8:11], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 553) 
ds_read_b96_tr_b6 v[181:183], v61 // (op 733) Transpose load local data 

s_waitcnt vmcnt(33)
ds_write_b96 v41, v[34:36] offset:32 // (op 13) Store local data 


v_add_u32 v61, 100448, v9 // (op 737) 
ds_read_u8 v89, v61 // (op 737) Load local data 

s_waitcnt vmcnt(32)
ds_write_b96 v41, v[38:40] offset:48 // (op 13) Store local data 



v_add_u32 v61, 67712, v7 // (op 909) 
ds_read_b96_tr_b6 v[184:186], v61 // (op 909) Transpose load local data 

v_add_u32 v61, 71808, v7 // (op 909) 
ds_read_b96_tr_b6 v[187:189], v61 // (op 909) Transpose load local data 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[154:159], v[178:183], a[12:15], v77, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 729) 


v_add_u32 v61, 100480, v9 // (op 913) 
ds_read_u8 v93, v61 // (op 913) Load local data 

s_waitcnt vmcnt(31)
ds_write_b96 v41, v[42:44] offset:256 // (op 13) Store local data 



v_add_u32 v61, 67744, v7 // (op 1085) 
ds_read_b96_tr_b6 v[190:192], v61 // (op 1085) Transpose load local data 

v_add_u32 v61, 71840, v7 // (op 1085) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[154:159], v[184:189], a[16:19], v77, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 905) 
ds_read_b96_tr_b6 v[193:195], v61 // (op 1085) Transpose load local data 

s_waitcnt vmcnt(30)
ds_write_b96 v41, v[46:48] offset:272 // (op 13) Store local data 


v_add_u32 v61, 100512, v9 // (op 1089) 
ds_read_u8 v97, v61 // (op 1089) Load local data 

s_waitcnt vmcnt(29)
ds_write_b96 v41, v[50:52] offset:288 // (op 13) Store local data 



v_add_u32 v61, 67776, v7 // (op 1261) 
ds_read_b96_tr_b6 v[196:198], v61 // (op 1261) Transpose load local data 

v_add_u32 v61, 71872, v7 // (op 1261) 
ds_read_b96_tr_b6 v[199:201], v61 // (op 1261) Transpose load local data 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[154:159], v[190:195], a[20:23], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1081) 


v_add_u32 v61, 100544, v9 // (op 1265) 
ds_read_u8 v101, v61 // (op 1265) Load local data 

s_waitcnt vmcnt(28)
ds_write_b96 v41, v[54:56] offset:304 // (op 13) Store local data 



v_add_u32 v61, 67808, v7 // (op 1437) 
s_barrier  // (op 36312) 
ds_read_b96_tr_b6 v[202:204], v61 // (op 1437) Transpose load local data 




v_add_u32 v61, 71904, v7 // (op 1437) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[154:159], v[196:201], a[24:27], v77, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1257) 
ds_read_b96_tr_b6 v[205:207], v61 // (op 1437) Transpose load local data 

s_waitcnt vmcnt(27)
ds_write_b32 v57, v33 offset:32768 // (op 30) Store local data 


s_barrier  // (op 36401) 
s_barrier  // (op 36271) 
v_add_u32 v61, 100576, v9 // (op 1441) 



ds_read_u8 v105, v61 // (op 1441) Load local data 

s_waitcnt vmcnt(26)
ds_write_b96 v14, v[58:60] offset:33792 // (op 6) Store local data 
s_waitcnt vmcnt(25)
ds_write_b96 v14, v[62:64] offset:33808 // (op 6) Store local data 



s_barrier  // (op 36405) 
v_add_u32 v61, 101408, v6 // (op 2004) 
ds_read_b96_tr_b6 v[208:210], v61 // (op 2004) Transpose load local data 

v_add_u32 v61, 105504, v6 // (op 2004) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[154:159], v[202:207], a[28:31], v77, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1433) 
ds_read_b96_tr_b6 v[211:213], v61 // (op 2004) Transpose load local data 





v_add_u32 v61, 134176, v8 // (op 2007) 
ds_read_u8 v77, v61 // (op 2007) Load local data 

s_waitcnt vmcnt(24)
ds_write_b96 v14, v[66:68] offset:33824 // (op 6) Store local data 



v_add_u32 v61, 101440, v6 // (op 3542) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[208:213], v[160:165], a[32:35], v77, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2324) 
ds_read_b96_tr_b6 v[154:156], v61 // (op 3542) Transpose load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[208:213], v[166:171], a[36:39], v77, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2278) 
v_add_u32 v61, 105536, v6 // (op 3542) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[208:213], v[172:177], a[40:43], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2232) 
ds_read_b96_tr_b6 v[157:159], v61 // (op 3542) Transpose load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[208:213], v[178:183], a[44:47], v77, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2186) 
v_add_u32 v61, 134208, v8 // (op 3545) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[208:213], v[184:189], a[48:51], v77, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2140) 
ds_read_u8 v109, v61 // (op 3545) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[208:213], v[190:195], a[52:55], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2094) 
v_add_u32 v61, 101472, v6 // (op 5080) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[208:213], v[196:201], a[56:59], v77, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2048) 
ds_read_b96_tr_b6 v[214:216], v61 // (op 5080) Transpose load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[208:213], v[202:207], a[60:63], v77, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 2002) 
v_add_u32 v61, 105568, v6 // (op 5080) 


ds_read_b96_tr_b6 v[217:219], v61 // (op 5080) Transpose load local data 



v_add_u32 v61, 134240, v8 // (op 5083) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[154:159], v[160:165], a[64:67], v109, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3862) 
ds_read_u8 v77, v61 // (op 5083) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[154:159], v[166:171], a[68:71], v109, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3816) 
v_add_u32 v61, 101504, v6 // (op 6618) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[154:159], v[172:177], a[72:75], v109, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3770) 
ds_read_b96_tr_b6 v[208:210], v61 // (op 6618) Transpose load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[178:183], a[76:79], v109, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3724) 
v_add_u32 v61, 105600, v6 // (op 6618) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[154:159], v[184:189], a[80:83], v109, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3678) 
ds_read_b96_tr_b6 v[211:213], v61 // (op 6618) Transpose load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[154:159], v[190:195], a[84:87], v109, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3632) 
v_add_u32 v61, 134272, v8 // (op 6621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[154:159], v[196:201], a[88:91], v109, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3586) 
ds_read_u8 v113, v61 // (op 6621) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[154:159], v[202:207], a[92:95], v109, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3540) 
v_add_u32 v61, 101536, v6 // (op 8156) 


ds_read_b96_tr_b6 v[154:156], v61 // (op 8156) Transpose load local data 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[214:219], v[160:165], a[96:99], v77, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5400) 
v_add_u32 v61, 105632, v6 // (op 8156) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[214:219], v[166:171], a[100:103], v77, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5354) 
ds_read_b96_tr_b6 v[157:159], v61 // (op 8156) Transpose load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[214:219], v[172:177], a[104:107], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5308) 
v_add_u32 v61, 134304, v8 // (op 8159) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[214:219], v[178:183], a[108:111], v77, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5262) 
ds_read_u8 v109, v61 // (op 8159) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[214:219], v[184:189], a[112:115], v77, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5216) 
v_add_u32 v61, 101568, v6 // (op 9694) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[214:219], v[190:195], a[116:119], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5170) 
ds_read_b96_tr_b6 v[220:222], v61 // (op 9694) Transpose load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[214:219], v[196:201], a[120:123], v77, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5124) 
v_add_u32 v61, 105664, v6 // (op 9694) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[214:219], v[202:207], a[124:127], v77, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5078) 
ds_read_b96_tr_b6 v[223:225], v61 // (op 9694) Transpose load local data 





s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[208:213], v[160:165], a[128:131], v113, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6938) 
v_add_u32 v61, 134336, v8 // (op 9697) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[208:213], v[166:171], a[132:135], v113, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6892) 
ds_read_u8 v77, v61 // (op 9697) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[208:213], v[172:177], a[136:139], v113, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6846) 
v_add_u32 v61, 101600, v6 // (op 11232) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[208:213], v[178:183], a[140:143], v113, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6800) 
ds_read_b96_tr_b6 v[214:216], v61 // (op 11232) Transpose load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[208:213], v[184:189], a[144:147], v113, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6754) 
v_add_u32 v61, 105696, v6 // (op 11232) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[208:213], v[190:195], a[148:151], v113, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6708) 
ds_read_b96_tr_b6 v[217:219], v61 // (op 11232) Transpose load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[208:213], v[196:201], a[152:155], v113, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6662) 


s_barrier  // (op 36473) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[208:213], v[202:207], a[156:159], v113, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6616) 
v_add_u32 v61, 134368, v8 // (op 11235) 


ds_read_u8 v113, v61 // (op 11235) Load local data 

s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[154:159], v[160:165], a[160:163], v109, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8476) 
s_barrier  // (op 36477) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[154:159], v[166:171], a[164:167], v109, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8430) 
s_waitcnt vmcnt(23)
ds_write_b96 v14, v[70:72] offset:33840 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[154:159], v[172:177], a[168:171], v109, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8384) 
s_waitcnt vmcnt(22)
ds_write_b96 v14, v[74:76] offset:34048 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[154:159], v[178:183], a[172:175], v109, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8338) 
s_waitcnt vmcnt(21)
ds_write_b96 v14, v[78:80] offset:34064 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:159], v[184:189], a[176:179], v109, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8292) 
s_waitcnt vmcnt(20)
ds_write_b96 v14, v[82:84] offset:34080 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:159], v[190:195], a[180:183], v109, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8246) 
s_waitcnt vmcnt(19)
ds_write_b96 v14, v[86:88] offset:34096 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:159], v[196:201], a[184:187], v109, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8200) 
s_barrier  // (op 36298) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:159], v[202:207], a[188:191], v109, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8154) 





v_add_u32 v61, 66560, v53 // (op 21) 
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[220:225], v[160:165], a[192:195], v77, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 10014) 
s_waitcnt vmcnt(18)
ds_write_b32 v61, v37 // (op 21) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[220:225], v[166:171], a[196:199], v77, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9968) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[220:225], v[172:177], a[200:203], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9922) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[220:225], v[178:183], a[204:207], v77, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9876) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[220:225], v[184:189], a[208:211], v77, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9830) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[220:225], v[190:195], a[212:215], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9784) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[220:225], v[196:201], a[216:219], v77, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9738) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[220:225], v[202:207], a[220:223], v77, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9692) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[214:219], v[160:165], a[224:227], v113, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11552) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[214:219], v[166:171], a[228:231], v113, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11506) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[214:219], v[172:177], a[232:235], v113, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11460) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[214:219], v[178:183], a[236:239], v113, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11414) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[214:219], v[184:189], a[240:243], v113, v93 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11368) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[214:219], v[190:195], a[244:247], v113, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11322) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[214:219], v[196:201], a[248:251], v113, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11276) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[214:219], v[202:207], a[252:255], v113, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11230) 




s_ashr_i64 s[18:19], s[28:29], 7 // (op 9652) 
s_mov_b32 s1, s18 // (op 9652) convert

s_and_b32 s18, s1, 4294967294 // (op 9652) 

s_cmp_lt_u32 s0, s18 // (op 9652) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_9652 // (op 9652) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_9652 if true)
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_9652:

















s_mul_hi_i32 s0, s16, s12 // (op 35335) 
v_and_b32 v93, -4, v1 // (op 35335) 
s_add_i32 s1, s0, s16 // (op 35335) 

v_sub_u32 v97, v1, v93 // (op 35335) 

s_ashr_i32 s0, s1, 31 // (op 35335) 
v_lshlrev_b32 v93, 6, v97 // (op 35335) 

s_cmp_eq_i32 s12, 0 // (op 35335) 
s_mov_b32 s18, scc // (op 35335) 
v_lshrrev_b32 v97, 2, v1 // (op 35335) 
s_cmp_eq_i32 s18, 0 // (op 35335) 
s_cselect_b32 s19, 0, -1 // (op 35335) 

v_lshlrev_b32 v101, 1, v97 // (op 35335) 

s_and_b32 s18, s15, 31 // (op 35335) 
s_lshl_b32 s49, 1, s18 // (op 35335) 
s_add_i32 s50, s49, s19 // (op 35335) 


s_and_b32 s19, s0, s50 // (op 35335) 


s_add_i32 s0, s1, s19 // (op 35335) 


s_ashr_i32 s1, s0, s18 // (op 35335) 


s_lshl_b32 s0, s15, 24 // (op 35335) 
s_ashr_i32 s18, s0, 31 // (op 35335) 

s_xor_b32 s0, s1, s18 // (op 35335) 

s_sub_i32 s1, s0, s18 // (op 35335) 


s_mul_hi_i32 s0, s1, s44 // (op 35335) 
s_add_i32 s18, s0, s1 // (op 35335) 

s_ashr_i32 s0, s18, 31 // (op 35335) 
s_cmp_eq_i32 s44, 0 // (op 35335) 
s_mov_b32 s19, scc // (op 35335) 
s_cmp_eq_i32 s19, 0 // (op 35335) 
s_cselect_b32 s49, 0, -1 // (op 35335) 

s_and_b32 s19, s45, 31 // (op 35335) 
s_lshl_b32 s50, 1, s19 // (op 35335) 
s_add_i32 s51, s50, s49 // (op 35335) 


s_and_b32 s49, s0, s51 // (op 35335) 


s_add_i32 s0, s18, s49 // (op 35335) 


s_ashr_i32 s18, s0, s19 // (op 35335) 


s_lshl_b32 s0, s45, 24 // (op 35335) 
s_ashr_i32 s19, s0, 31 // (op 35335) 

s_xor_b32 s0, s18, s19 // (op 35335) 

s_sub_i32 s18, s0, s19 // (op 35335) 


s_mul_i32 s0, s1, s13 // (op 35335) 

s_sub_u32 s1, s16, s0 // (op 35335) 

s_mul_i32 s0, s18, s13 // (op 35335) 
s_add_u32 s0, s0, s1 // (op 35335) 


s_mul_hi_i32 s1, s16, s11 // (op 35335) 
s_add_i32 s18, s1, s16 // (op 35335) 

s_ashr_i32 s1, s18, 31 // (op 35335) 
s_cmp_eq_i32 s11, 0 // (op 35335) 
s_mov_b32 s19, scc // (op 35335) 
s_cmp_eq_i32 s19, 0 // (op 35335) 
s_cselect_b32 s49, 0, -1 // (op 35335) 

s_and_b32 s19, s47, 31 // (op 35335) 
s_lshl_b32 s50, 1, s19 // (op 35335) 
s_add_i32 s51, s50, s49 // (op 35335) 


s_and_b32 s49, s1, s51 // (op 35335) 


s_add_i32 s1, s18, s49 // (op 35335) 


s_ashr_i32 s18, s1, s19 // (op 35335) 


s_lshl_b32 s1, s47, 24 // (op 35335) 
s_ashr_i32 s19, s1, 31 // (op 35335) 

s_xor_b32 s1, s18, s19 // (op 35335) 

s_sub_i32 s18, s1, s19 // (op 35335) 


s_add_u32 s50, s8, 256 // (op 35335) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 35335) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 35335) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 35335) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 35335) 

s_mul_i32 s1, s13, s50 // (op 35335) 

s_mul_i32 s19, s10, s1 // (op 35335) 

s_mul_i32 s1, s18, s19 // (op 35335) 

s_sub_u32 s19, s16, s1 // (op 35335) 

s_mov_b32 s1, s19 // (op 35335) 
s_mul_hi_i32 s49, s1, s14 // (op 35335) 
s_add_i32 s50, s49, s1 // (op 35335) 


s_ashr_i32 s1, s50, 31 // (op 35335) 
s_cmp_eq_i32 s14, 0 // (op 35335) 
s_mov_b32 s49, scc // (op 35335) 
s_cmp_eq_i32 s49, 0 // (op 35335) 
s_cselect_b32 s51, 0, -1 // (op 35335) 

s_and_b32 s49, s46, 31 // (op 35335) 
s_lshl_b32 s72, 1, s49 // (op 35335) 
s_add_i32 s73, s72, s51 // (op 35335) 


s_and_b32 s51, s1, s73 // (op 35335) 


s_add_i32 s1, s50, s51 // (op 35335) 


s_ashr_i32 s50, s1, s49 // (op 35335) 


s_lshl_b32 s1, s46, 24 // (op 35335) 
s_ashr_i32 s49, s1, 31 // (op 35335) 

s_xor_b32 s1, s50, s49 // (op 35335) 

s_sub_i32 s50, s1, s49 // (op 35335) 


s_mul_i32 s1, s48, s13 // (op 35335) 
s_add_u32 s72, s6, 256 // (op 35335) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 35335) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 35335) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 35335) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 8 // (op 35335) 

s_sub_i32 s49, s72, s1 // (op 35335) 

s_mul_i32 s51, s50, s49 // (op 35335) 


s_sub_u32 s49, s19, s51 // (op 35335) 


s_add_u32 s19, s49, s1 // (op 35335) 


s_cmp_eq_i32 s18, 0 // (op 35335) 
s_mov_b32 s1, scc // (op 35335) 

s_cmp_eq_i32 s1, 0 // (op 35335) 
s_cselect_b32 s18, s19, s0 // (op 35335) 



v_lshl_add_u32 v97, s18, 8, v93 // (op 35335) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 35335) 
v_mov_b32 v15, 0 // (op 35335) convert
v_mov_b32 v14, v97 // (op 35335) convert

s_lshr_b32 s18, s0, 1 // (op 35335) 

v_lshl_add_u32 v93, s18, 8, v101 // (op 35335) 


v_mul_lo_u32 v31, v93, s25 // (op 35335) most significant: low of low * high
v_mul_hi_u32 v97, v93, s24 // (op 35335) most significant: high of low * low
v_mul_lo_u32 v30, v93, s24 // (op 35335) least significant: low of low * low
v_add_u32 v31, v31, v97 // (op 35335) most significant: sum


v_add_co_u32 v32, s[0:1], v14, v30 // (op 35335) least significant half
v_addc_co_u32 v33, s[0:1], v15, v31, s[0:1] // (op 35335) most significant half



v_mul_lo_u32 v15, v33, 6 // (op 35335) most significant: low of high * low
v_mul_hi_u32 v93, v32, 6 // (op 35335) most significant: high of low * low
v_mul_lo_u32 v14, v32, 6 // (op 35335) least significant: low of low * low
v_add_u32 v15, v15, v93 // (op 35335) most significant: sum


v_lshrrev_b64 v[30:31], 3, v[14:15] // (op 35335) 

v_mov_b32 v93, v30 // (op 35335) convert

s_mul_i32 s1, s29, 6 // (op 35337) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 35337) least significant
s_mul_hi_u32 s18, s28, 6 // (op 35337) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35337) most significant: sum

s_mul_i32 s19, s24, s1 // (op 35337) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 35337) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 35337) least significant
s_mul_hi_u32 s50, s24, s0 // (op 35337) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 35337) most significant: sum
s_add_u32 s19, s19, s50 // (op 35337) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 35337) 

s_mov_b32 s20, s26 // (op 35337) reg expression
s_mov_b32 s21, s27 // (op 35337) reg expression
s_mov_b32 s22, s0 // (op 35337) reg expression
s_mov_b32 s23, 131072 // (op 35337) call()

s_mul_hi_i32 s0, s16, s12 // (op 35465) 
v_and_b32 v97, -4, v1 // (op 35465) 
s_add_i32 s1, s0, s16 // (op 35465) 

v_sub_u32 v101, v1, v97 // (op 35465) 

s_ashr_i32 s0, s1, 31 // (op 35465) 
v_lshlrev_b32 v97, 6, v101 // (op 35465) 

s_cmp_eq_i32 s12, 0 // (op 35465) 
s_mov_b32 s18, scc // (op 35465) 
v_lshrrev_b32 v101, 2, v1 // (op 35465) 
s_cmp_eq_i32 s18, 0 // (op 35465) 
s_cselect_b32 s19, 0, -1 // (op 35465) 

v_lshlrev_b32 v105, 1, v101 // (op 35465) 

s_and_b32 s18, s15, 31 // (op 35465) 
s_lshl_b32 s49, 1, s18 // (op 35465) 
s_add_i32 s50, s49, s19 // (op 35465) 


s_and_b32 s19, s0, s50 // (op 35465) 


s_add_i32 s0, s1, s19 // (op 35465) 


s_ashr_i32 s1, s0, s18 // (op 35465) 


s_lshl_b32 s0, s15, 24 // (op 35465) 
s_ashr_i32 s18, s0, 31 // (op 35465) 

s_xor_b32 s0, s1, s18 // (op 35465) 

s_sub_i32 s1, s0, s18 // (op 35465) 


s_mul_hi_i32 s0, s1, s44 // (op 35465) 
s_add_i32 s18, s0, s1 // (op 35465) 

s_ashr_i32 s0, s18, 31 // (op 35465) 
s_cmp_eq_i32 s44, 0 // (op 35465) 
s_mov_b32 s19, scc // (op 35465) 
s_cmp_eq_i32 s19, 0 // (op 35465) 
s_cselect_b32 s49, 0, -1 // (op 35465) 

s_and_b32 s19, s45, 31 // (op 35465) 
s_lshl_b32 s50, 1, s19 // (op 35465) 
s_add_i32 s51, s50, s49 // (op 35465) 


s_and_b32 s49, s0, s51 // (op 35465) 


s_add_i32 s0, s18, s49 // (op 35465) 


s_ashr_i32 s18, s0, s19 // (op 35465) 


s_lshl_b32 s0, s45, 24 // (op 35465) 
s_ashr_i32 s19, s0, 31 // (op 35465) 

s_xor_b32 s0, s18, s19 // (op 35465) 

s_sub_i32 s18, s0, s19 // (op 35465) 


s_add_u32 s50, s8, 256 // (op 35465) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 35465) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 35465) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 35465) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 35465) 

s_mul_i32 s0, s18, s50 // (op 35465) 

s_sub_i32 s49, s1, s0 // (op 35465) 


s_mul_hi_i32 s0, s16, s11 // (op 35465) 
s_add_i32 s1, s0, s16 // (op 35465) 

s_ashr_i32 s0, s1, 31 // (op 35465) 
s_cmp_eq_i32 s11, 0 // (op 35465) 
s_mov_b32 s18, scc // (op 35465) 
s_cmp_eq_i32 s18, 0 // (op 35465) 
s_cselect_b32 s19, 0, -1 // (op 35465) 

s_and_b32 s18, s47, 31 // (op 35465) 
s_lshl_b32 s72, 1, s18 // (op 35465) 
s_add_i32 s73, s72, s19 // (op 35465) 


s_and_b32 s19, s0, s73 // (op 35465) 


s_add_i32 s0, s1, s19 // (op 35465) 


s_ashr_i32 s1, s0, s18 // (op 35465) 


s_lshl_b32 s0, s47, 24 // (op 35465) 
s_ashr_i32 s18, s0, 31 // (op 35465) 

s_xor_b32 s0, s1, s18 // (op 35465) 

s_sub_i32 s1, s0, s18 // (op 35465) 


s_mul_i32 s0, s13, s50 // (op 35465) 

s_mul_i32 s18, s10, s0 // (op 35465) 

s_mul_i32 s0, s1, s18 // (op 35465) 

s_sub_u32 s18, s16, s0 // (op 35465) 

s_mul_hi_i32 s0, s18, s14 // (op 35465) 
s_add_i32 s19, s0, s18 // (op 35465) 


s_ashr_i32 s0, s19, 31 // (op 35465) 
s_cmp_eq_i32 s14, 0 // (op 35465) 
s_mov_b32 s18, scc // (op 35465) 
s_cmp_eq_i32 s18, 0 // (op 35465) 
s_cselect_b32 s50, 0, -1 // (op 35465) 

s_and_b32 s18, s46, 31 // (op 35465) 
s_lshl_b32 s51, 1, s18 // (op 35465) 
s_add_i32 s72, s51, s50 // (op 35465) 


s_and_b32 s50, s0, s72 // (op 35465) 


s_add_i32 s0, s19, s50 // (op 35465) 


s_ashr_i32 s19, s0, s18 // (op 35465) 


s_lshl_b32 s0, s46, 24 // (op 35465) 
s_ashr_i32 s18, s0, 31 // (op 35465) 

s_xor_b32 s0, s19, s18 // (op 35465) 

s_sub_i32 s19, s0, s18 // (op 35465) 


s_cmp_eq_i32 s1, 0 // (op 35465) 
s_mov_b32 s0, scc // (op 35465) 

s_cmp_eq_i32 s0, 0 // (op 35465) 
s_cselect_b32 s1, s19, s49 // (op 35465) 



v_lshl_add_u32 v101, s1, 8, v97 // (op 35465) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 35465) 
v_mov_b32 v15, 0 // (op 35465) convert
v_mov_b32 v14, v101 // (op 35465) convert

s_lshr_b32 s18, s0, 1 // (op 35465) 

v_lshl_add_u32 v97, s18, 8, v105 // (op 35465) 


v_mul_lo_u32 v31, v97, s31 // (op 35465) most significant: low of low * high
v_mul_hi_u32 v101, v97, s30 // (op 35465) most significant: high of low * low
v_mul_lo_u32 v30, v97, s30 // (op 35465) least significant: low of low * low
v_add_u32 v31, v31, v101 // (op 35465) most significant: sum

v_add_co_u32 v30, s[0:1], v30, v14 // (op 35465) least significant half
v_addc_co_u32 v31, s[0:1], v31, v15, s[0:1] // (op 35465) most significant half



v_mul_lo_u32 v15, v31, 6 // (op 35465) most significant: low of high * low
v_mul_hi_u32 v97, v30, 6 // (op 35465) most significant: high of low * low
v_mul_lo_u32 v14, v30, 6 // (op 35465) least significant: low of low * low
v_add_u32 v15, v15, v97 // (op 35465) most significant: sum


v_lshrrev_b64 v[30:31], 3, v[14:15] // (op 35465) 

v_mov_b32 v97, v30 // (op 35465) convert

s_mul_i32 s1, s29, 6 // (op 35467) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 35467) least significant
s_mul_hi_u32 s18, s28, 6 // (op 35467) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35467) most significant: sum

s_mul_i32 s19, s30, s1 // (op 35467) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 35467) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 35467) least significant
s_mul_hi_u32 s50, s30, s0 // (op 35467) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 35467) most significant: sum
s_add_u32 s19, s19, s50 // (op 35467) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 35467) 

s_mov_b32 s52, s32 // (op 35467) reg expression
s_mov_b32 s53, s33 // (op 35467) reg expression
s_mov_b32 s54, s0 // (op 35467) reg expression
s_mov_b32 s55, 131072 // (op 35467) call()

s_mul_hi_i32 s0, s16, s12 // (op 35529) 
v_and_b32 v101, -64, v1 // (op 35529) 
s_add_i32 s1, s0, s16 // (op 35529) 

v_sub_u32 v105, v1, v101 // (op 35529) 

s_ashr_i32 s0, s1, 31 // (op 35529) 
v_lshlrev_b32 v101, 2, v105 // (op 35529) 

s_cmp_eq_i32 s12, 0 // (op 35529) 
s_mov_b32 s18, scc // (op 35529) 
v_lshrrev_b32 v105, 6, v1 // (op 35529) 
s_cmp_eq_i32 s18, 0 // (op 35529) 
s_cselect_b32 s19, 0, -1 // (op 35529) 

s_and_b32 s18, s15, 31 // (op 35529) 
s_lshl_b32 s49, 1, s18 // (op 35529) 
s_add_i32 s50, s49, s19 // (op 35529) 


s_and_b32 s19, s0, s50 // (op 35529) 


s_add_i32 s0, s1, s19 // (op 35529) 


s_ashr_i32 s1, s0, s18 // (op 35529) 


s_lshl_b32 s0, s15, 24 // (op 35529) 
s_ashr_i32 s18, s0, 31 // (op 35529) 

s_xor_b32 s0, s1, s18 // (op 35529) 

s_sub_i32 s1, s0, s18 // (op 35529) 


s_mul_hi_i32 s0, s1, s44 // (op 35529) 
s_add_i32 s18, s0, s1 // (op 35529) 

s_ashr_i32 s0, s18, 31 // (op 35529) 
s_cmp_eq_i32 s44, 0 // (op 35529) 
s_mov_b32 s19, scc // (op 35529) 
s_cmp_eq_i32 s19, 0 // (op 35529) 
s_cselect_b32 s49, 0, -1 // (op 35529) 

s_and_b32 s19, s45, 31 // (op 35529) 
s_lshl_b32 s50, 1, s19 // (op 35529) 
s_add_i32 s51, s50, s49 // (op 35529) 


s_and_b32 s49, s0, s51 // (op 35529) 


s_add_i32 s0, s18, s49 // (op 35529) 


s_ashr_i32 s18, s0, s19 // (op 35529) 


s_lshl_b32 s0, s45, 24 // (op 35529) 
s_ashr_i32 s19, s0, 31 // (op 35529) 

s_xor_b32 s0, s18, s19 // (op 35529) 

s_sub_i32 s18, s0, s19 // (op 35529) 


s_mul_i32 s0, s1, s13 // (op 35529) 

s_sub_u32 s1, s16, s0 // (op 35529) 

s_mul_i32 s0, s18, s13 // (op 35529) 
s_add_u32 s0, s0, s1 // (op 35529) 


s_mul_hi_i32 s1, s16, s11 // (op 35529) 
s_add_i32 s18, s1, s16 // (op 35529) 

s_ashr_i32 s1, s18, 31 // (op 35529) 
s_cmp_eq_i32 s11, 0 // (op 35529) 
s_mov_b32 s19, scc // (op 35529) 
s_cmp_eq_i32 s19, 0 // (op 35529) 
s_cselect_b32 s49, 0, -1 // (op 35529) 

s_and_b32 s19, s47, 31 // (op 35529) 
s_lshl_b32 s50, 1, s19 // (op 35529) 
s_add_i32 s51, s50, s49 // (op 35529) 


s_and_b32 s49, s1, s51 // (op 35529) 


s_add_i32 s1, s18, s49 // (op 35529) 


s_ashr_i32 s18, s1, s19 // (op 35529) 


s_lshl_b32 s1, s47, 24 // (op 35529) 
s_ashr_i32 s19, s1, 31 // (op 35529) 

s_xor_b32 s1, s18, s19 // (op 35529) 

s_sub_i32 s18, s1, s19 // (op 35529) 


s_add_u32 s50, s8, 256 // (op 35529) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 35529) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 35529) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 35529) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 35529) 

s_mul_i32 s1, s13, s50 // (op 35529) 

s_mul_i32 s19, s10, s1 // (op 35529) 

s_mul_i32 s1, s18, s19 // (op 35529) 

s_sub_u32 s19, s16, s1 // (op 35529) 

s_mov_b32 s1, s19 // (op 35529) 
s_mul_hi_i32 s49, s1, s14 // (op 35529) 
s_add_i32 s50, s49, s1 // (op 35529) 


s_ashr_i32 s1, s50, 31 // (op 35529) 
s_cmp_eq_i32 s14, 0 // (op 35529) 
s_mov_b32 s49, scc // (op 35529) 
s_cmp_eq_i32 s49, 0 // (op 35529) 
s_cselect_b32 s51, 0, -1 // (op 35529) 

s_and_b32 s49, s46, 31 // (op 35529) 
s_lshl_b32 s72, 1, s49 // (op 35529) 
s_add_i32 s73, s72, s51 // (op 35529) 


s_and_b32 s51, s1, s73 // (op 35529) 


s_add_i32 s1, s50, s51 // (op 35529) 


s_ashr_i32 s50, s1, s49 // (op 35529) 


s_lshl_b32 s1, s46, 24 // (op 35529) 
s_ashr_i32 s49, s1, 31 // (op 35529) 

s_xor_b32 s1, s50, s49 // (op 35529) 

s_sub_i32 s50, s1, s49 // (op 35529) 


s_mul_i32 s1, s48, s13 // (op 35529) 
s_add_u32 s72, s6, 256 // (op 35529) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 35529) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 35529) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 35529) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 8 // (op 35529) 

s_sub_i32 s49, s72, s1 // (op 35529) 

s_mul_i32 s51, s50, s49 // (op 35529) 


s_sub_u32 s49, s19, s51 // (op 35529) 


s_add_u32 s19, s49, s1 // (op 35529) 


s_cmp_eq_i32 s18, 0 // (op 35529) 
s_mov_b32 s1, scc // (op 35529) 

s_cmp_eq_i32 s1, 0 // (op 35529) 
s_cselect_b32 s18, s19, s0 // (op 35529) 



v_lshl_add_u32 v109, s18, 8, v101 // (op 35529) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 35529) 
s_lshr_b32 s18, s0, 1 // (op 35529) 

v_lshl_add_u32 v101, s18, 3, v105 // (op 35529) 


v_mul_lo_u32 v105, v101, s34 // (op 35529) 

v_add_u32 v101, v109, v105 // (op 35529) 


s_mul_i32 s1, s34, s39 // (op 35531) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 35531) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 35531) least significant
s_mul_hi_u32 s19, s34, s38 // (op 35531) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35531) most significant: sum
s_add_u32 s1, s1, s19 // (op 35531) most significant: sum


s_mov_b32 s64, s36 // (op 35531) reg expression
s_mov_b32 s65, s37 // (op 35531) reg expression
s_mov_b32 s66, s0 // (op 35531) reg expression
s_mov_b32 s67, 131072 // (op 35531) call()

s_mul_hi_i32 s0, s16, s12 // (op 35593) 
v_and_b32 v105, -64, v1 // (op 35593) 
s_add_i32 s1, s0, s16 // (op 35593) 

v_sub_u32 v109, v1, v105 // (op 35593) 

s_ashr_i32 s0, s1, 31 // (op 35593) 
v_lshlrev_b32 v105, 2, v109 // (op 35593) 

s_cmp_eq_i32 s12, 0 // (op 35593) 
s_mov_b32 s18, scc // (op 35593) 
v_lshrrev_b32 v109, 6, v1 // (op 35593) 
s_cmp_eq_i32 s18, 0 // (op 35593) 
s_cselect_b32 s19, 0, -1 // (op 35593) 

s_and_b32 s18, s15, 31 // (op 35593) 
s_lshl_b32 s49, 1, s18 // (op 35593) 
s_add_i32 s50, s49, s19 // (op 35593) 


s_and_b32 s19, s0, s50 // (op 35593) 


s_add_i32 s0, s1, s19 // (op 35593) 


s_ashr_i32 s1, s0, s18 // (op 35593) 


s_lshl_b32 s0, s15, 24 // (op 35593) 
s_ashr_i32 s18, s0, 31 // (op 35593) 

s_xor_b32 s0, s1, s18 // (op 35593) 

s_sub_i32 s1, s0, s18 // (op 35593) 


s_mul_hi_i32 s0, s1, s44 // (op 35593) 
s_add_i32 s18, s0, s1 // (op 35593) 

s_ashr_i32 s0, s18, 31 // (op 35593) 
s_cmp_eq_i32 s44, 0 // (op 35593) 
s_mov_b32 s19, scc // (op 35593) 
s_cmp_eq_i32 s19, 0 // (op 35593) 
s_cselect_b32 s49, 0, -1 // (op 35593) 

s_and_b32 s19, s45, 31 // (op 35593) 
s_lshl_b32 s50, 1, s19 // (op 35593) 
s_add_i32 s51, s50, s49 // (op 35593) 


s_and_b32 s49, s0, s51 // (op 35593) 


s_add_i32 s0, s18, s49 // (op 35593) 


s_ashr_i32 s18, s0, s19 // (op 35593) 


s_lshl_b32 s0, s45, 24 // (op 35593) 
s_ashr_i32 s19, s0, 31 // (op 35593) 

s_xor_b32 s0, s18, s19 // (op 35593) 

s_sub_i32 s18, s0, s19 // (op 35593) 


s_add_u32 s50, s8, 256 // (op 35593) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 35593) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 35593) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 35593) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 35593) 

s_mul_i32 s0, s18, s50 // (op 35593) 

s_sub_i32 s49, s1, s0 // (op 35593) 


s_mul_hi_i32 s0, s16, s11 // (op 35593) 
s_add_i32 s1, s0, s16 // (op 35593) 

s_ashr_i32 s0, s1, 31 // (op 35593) 
s_cmp_eq_i32 s11, 0 // (op 35593) 
s_mov_b32 s18, scc // (op 35593) 
s_cmp_eq_i32 s18, 0 // (op 35593) 
s_cselect_b32 s19, 0, -1 // (op 35593) 

s_and_b32 s18, s47, 31 // (op 35593) 
s_lshl_b32 s72, 1, s18 // (op 35593) 
s_add_i32 s73, s72, s19 // (op 35593) 


s_and_b32 s19, s0, s73 // (op 35593) 


s_add_i32 s0, s1, s19 // (op 35593) 


s_ashr_i32 s1, s0, s18 // (op 35593) 


s_lshl_b32 s0, s47, 24 // (op 35593) 
s_ashr_i32 s18, s0, 31 // (op 35593) 

s_xor_b32 s0, s1, s18 // (op 35593) 

s_sub_i32 s1, s0, s18 // (op 35593) 


s_mul_i32 s0, s13, s50 // (op 35593) 

s_mul_i32 s18, s10, s0 // (op 35593) 

s_mul_i32 s0, s1, s18 // (op 35593) 

s_sub_u32 s18, s16, s0 // (op 35593) 

s_mul_hi_i32 s0, s18, s14 // (op 35593) 
s_add_i32 s19, s0, s18 // (op 35593) 


s_ashr_i32 s0, s19, 31 // (op 35593) 
s_cmp_eq_i32 s14, 0 // (op 35593) 
s_mov_b32 s18, scc // (op 35593) 
s_cmp_eq_i32 s18, 0 // (op 35593) 
s_cselect_b32 s50, 0, -1 // (op 35593) 

s_and_b32 s18, s46, 31 // (op 35593) 
s_lshl_b32 s51, 1, s18 // (op 35593) 
s_add_i32 s72, s51, s50 // (op 35593) 


s_and_b32 s50, s0, s72 // (op 35593) 


s_add_i32 s0, s19, s50 // (op 35593) 


s_ashr_i32 s19, s0, s18 // (op 35593) 


s_lshl_b32 s0, s46, 24 // (op 35593) 
s_ashr_i32 s18, s0, 31 // (op 35593) 

s_xor_b32 s0, s19, s18 // (op 35593) 

s_sub_i32 s19, s0, s18 // (op 35593) 


s_cmp_eq_i32 s1, 0 // (op 35593) 
s_mov_b32 s0, scc // (op 35593) 

s_cmp_eq_i32 s0, 0 // (op 35593) 
s_cselect_b32 s1, s19, s49 // (op 35593) 



v_lshl_add_u32 v113, s1, 8, v105 // (op 35593) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 35593) 
s_lshr_b32 s18, s0, 1 // (op 35593) 

v_lshl_add_u32 v105, s18, 3, v109 // (op 35593) 


v_mul_lo_u32 v109, v105, s40 // (op 35593) 
v_add_u32 v109, v109, v113 // (op 35593) 


s_mul_i32 s1, s40, s39 // (op 35595) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 35595) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 35595) least significant
s_mul_hi_u32 s19, s40, s38 // (op 35595) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 35595) most significant: sum
s_add_u32 s1, s1, s19 // (op 35595) most significant: sum


s_mov_b32 s68, s42 // (op 35595) reg expression
s_mov_b32 s69, s43 // (op 35595) reg expression
s_mov_b32 s70, s0 // (op 35595) reg expression
s_mov_b32 s71, 131072 // (op 35595) call()

v_and_b32 v105, -4, v1 // (op 35691) 
v_sub_u32 v113, v1, v105 // (op 35691) 

v_lshlrev_b32 v105, 6, v113 // (op 35691) 

v_lshrrev_b32 v113, 2, v1 // (op 35691) 
v_lshl_add_u32 v117, v113, 9, v105 // (op 35691) 


v_mul_lo_u32 v105, v117, 6 // (op 35691) 
v_lshrrev_b32 v113, 3, v105 // (op 35691) 

v_lshrrev_b32 v105, 4, v117 // (op 35691) 

v_lshlrev_b32 v117, 2, v105 // (op 35691) 

v_add_u32 v105, v113, v117 // (op 35691) 


v_and_b32 v113, -4, v1 // (op 35749) 
v_sub_u32 v117, v1, v113 // (op 35749) 

v_lshlrev_b32 v113, 6, v117 // (op 35749) 

v_lshrrev_b32 v117, 2, v1 // (op 35749) 
v_lshl_add_u32 v121, v117, 9, v113 // (op 35749) 


v_mul_lo_u32 v113, v121, 6 // (op 35749) 
v_lshrrev_b32 v117, 3, v113 // (op 35749) 

v_lshrrev_b32 v113, 4, v121 // (op 35749) 

v_lshlrev_b32 v121, 2, v113 // (op 35749) 

v_add_u32 v113, v117, v121 // (op 35749) 


v_and_b32 v117, -64, v1 // (op 35807) 
v_sub_u32 v121, v1, v117 // (op 35807) 

v_lshlrev_b32 v117, 2, v121 // (op 35807) 

v_lshrrev_b32 v121, 6, v1 // (op 35807) 
v_lshl_add_u32 v125, v121, 8, v117 // (op 35807) 


v_and_b32 v117, -64, v1 // (op 35865) 
v_sub_u32 v121, v1, v117 // (op 35865) 

v_lshlrev_b32 v117, 2, v121 // (op 35865) 

v_lshrrev_b32 v121, 6, v1 // (op 35865) 
v_lshl_add_u32 v129, v121, 8, v117 // (op 35865) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 11196) 
s_and_b32 s18, s0, 4294967294 // (op 11196) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 11195) 
s_mov_b32 s19, s0 // (op 11195) convert

s_cmp_lt_u32 s18, s19 // (op 11195) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_11195 // (op 11195) Condition: Top (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_11195 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_11195:

s_add_u32 s18, s18, 1 // (op 11197) 
s_mul_i32 s1, s25, 6 // (op 176) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 176) least significant
s_mul_hi_u32 s19, s24, 6 // (op 176) most significant: high of low * low
s_add_u32 s1, s1, s19 // (op 176) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[0:1] // (op 176) 


v_mov_b32 v16, v93 // (op 176) 
v_mov_b32 v17, v16 // (op 176) reg expression
buffer_load_dwordx3 v[18:20], v17, s[20:23], 0 offen // (op 176) Load value
buffer_load_dwordx3 v[22:24], v17, s[20:23], 0 offen offset:12 // (op 176) Load value
buffer_load_dwordx3 v[26:28], v17, s[20:23], 0 offen offset:24 // (op 176) Load value
buffer_load_dwordx3 v[30:32], v17, s[20:23], 0 offen offset:36 // (op 176) Load value
v_add_u32 v17, v17, v14 // (op 176) 
buffer_load_dwordx3 v[34:36], v17, s[20:23], 0 offen // (op 176) Load value
buffer_load_dwordx3 v[38:40], v17, s[20:23], 0 offen offset:12 // (op 176) Load value
buffer_load_dwordx3 v[42:44], v17, s[20:23], 0 offen offset:24 // (op 176) Load value
buffer_load_dwordx3 v[46:48], v17, s[20:23], 0 offen offset:36 // (op 176) Load value



s_lshl_b64 s[0:1], s[24:25], 7 // (op 35325) 
s_mul_i32 s51, s1, 6 // (op 35325) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 35325) least significant
s_mul_hi_u32 s19, s0, 6 // (op 35325) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 35325) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 35325) 

v_add_u32 v93, v93, s0 // (op 35325) 


s_barrier  // (op 36274) 
s_mul_i32 s1, s31, 6 // (op 190) most significant: low of high * low



s_mul_i32 s0, s30, 6 // (op 190) least significant
s_mul_hi_u32 s19, s30, 6 // (op 190) most significant: high of low * low
s_add_u32 s1, s1, s19 // (op 190) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[0:1] // (op 190) 


v_mov_b32 v21, v97 // (op 190) 
v_mov_b32 v25, v21 // (op 190) reg expression
buffer_load_dwordx3 v[50:52], v25, s[52:55], 0 offen // (op 190) Load value
s_waitcnt vmcnt(8)
ds_write_b96 v105, v[18:20] offset:33792 // (op 177) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v105, v[22:24] offset:33808 // (op 177) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v105, v[26:28] offset:33824 // (op 177) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v105, v[30:32] offset:33840 // (op 177) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v105, v[34:36] offset:34048 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v105, v[38:40] offset:34064 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v105, v[42:44] offset:34080 // (op 177) Store local data 
buffer_load_dwordx3 v[54:56], v25, s[52:55], 0 offen offset:12 // (op 190) Load value
buffer_load_dwordx3 v[58:60], v25, s[52:55], 0 offen offset:24 // (op 190) Load value
buffer_load_dwordx3 v[62:64], v25, s[52:55], 0 offen offset:36 // (op 190) Load value
v_add_u32 v25, v25, v14 // (op 190) 
s_waitcnt vmcnt(4)
ds_write_b96 v105, v[46:48] offset:34096 // (op 177) Store local data 
buffer_load_dwordx3 v[66:68], v25, s[52:55], 0 offen // (op 190) Load value

buffer_load_dwordx3 v[70:72], v25, s[52:55], 0 offen offset:12 // (op 190) Load value
buffer_load_dwordx3 v[74:76], v25, s[52:55], 0 offen offset:24 // (op 190) Load value
buffer_load_dwordx3 v[78:80], v25, s[52:55], 0 offen offset:36 // (op 190) Load value



s_lshl_b64 s[0:1], s[30:31], 7 // (op 35457) 
s_mul_i32 s51, s1, 6 // (op 35457) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 35457) least significant
s_mul_hi_u32 s19, s0, 6 // (op 35457) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 35457) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 35457) 

v_add_u32 v97, v97, s0 // (op 35457) 



s_barrier  // (op 36287) 
v_mov_b32 v53, v101 // (op 191) 



v_mov_b32 v57, v53 // (op 191) reg expression

buffer_load_dword v61, v57, s[64:67], 0 offen // (op 191) Load value


s_lshl_b32 s19, s34, 2 // (op 35521) 
v_add_u32 v101, v101, s19 // (op 35521) 

s_waitcnt vmcnt(8)
ds_write_b96 v113, v[50:52] // (op 187) Store local data 


v_mov_b32 v53, v109 // (op 192) 
v_mov_b32 v57, v53 // (op 192) reg expression

buffer_load_dword v65, v57, s[68:71], 0 offen // (op 192) Load value


s_lshl_b32 s19, s40, 2 // (op 35585) 
v_add_u32 v109, v109, s19 // (op 35585) 

s_waitcnt vmcnt(8)
ds_write_b96 v113, v[54:56] offset:16 // (op 187) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 36277) 
s_waitcnt vmcnt(7)
ds_write_b96 v113, v[58:60] offset:32 // (op 187) Store local data 



ds_read_b96_tr_b6 v[14:16], v0 offset:33792 // (op 179) Transpose load local data 
ds_read_b96_tr_b6 v[17:19], v0 offset:37888 // (op 179) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b96 v113, v[62:64] offset:48 // (op 187) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v113, v[66:68] offset:256 // (op 187) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v113, v[70:72] offset:272 // (op 187) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v113, v[74:76] offset:288 // (op 187) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v113, v[78:80] offset:304 // (op 187) Store local data 

s_barrier  // (op 36301) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 36291) 






v_add_u32 v20, 66560, v125 // (op 188) 
ds_read_b96_tr_b6 v[66:68], v3 // (op 181) Transpose load local data 
ds_read_b96_tr_b6 v[69:71], v3 offset:4096 // (op 181) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b32 v20, v61 // (op 188) Store local data 


s_barrier  // (op 36315) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 36305) 





v_add_u32 v20, 66560, v4 // (op 183) 
ds_read_u8 v21, v20 // (op 183) Load local data 

s_waitcnt vmcnt(0)
ds_write_b32 v129, v65 offset:32768 // (op 189) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 36319) 


ds_read_u8 v20, v5 offset:32768 // (op 185) Load local data 



ds_read_b96_tr_b6 v[22:24], v3 offset:32 // (op 350) Transpose load local data 
ds_read_b96_tr_b6 v[25:27], v3 offset:4128 // (op 350) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[14:19], v[66:71], a[0:3], v21, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 186) 


ds_read_u8 v28, v5 offset:32800 // (op 354) Load local data 



ds_read_b96_tr_b6 v[60:62], v3 offset:64 // (op 526) Transpose load local data 
ds_read_b96_tr_b6 v[63:65], v3 offset:4160 // (op 526) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[14:19], v[22:27], a[4:7], v21, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 355) 


ds_read_u8 v29, v5 offset:32832 // (op 530) Load local data 



ds_read_b96_tr_b6 v[30:32], v3 offset:96 // (op 702) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v3 offset:4192 // (op 702) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[14:19], v[60:65], a[8:11], v21, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 531) 


ds_read_u8 v36, v5 offset:32864 // (op 706) Load local data 



ds_read_b96_tr_b6 v[54:56], v3 offset:128 // (op 878) Transpose load local data 
ds_read_b96_tr_b6 v[57:59], v3 offset:4224 // (op 878) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[14:19], v[30:35], a[12:15], v21, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 707) 


ds_read_u8 v37, v5 offset:32896 // (op 882) Load local data 



ds_read_b96_tr_b6 v[38:40], v3 offset:160 // (op 1054) Transpose load local data 
ds_read_b96_tr_b6 v[41:43], v3 offset:4256 // (op 1054) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:19], v[54:59], a[16:19], v21, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 883) 


ds_read_u8 v44, v5 offset:32928 // (op 1058) Load local data 



ds_read_b96_tr_b6 v[48:50], v3 offset:192 // (op 1230) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v3 offset:4288 // (op 1230) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[14:19], v[38:43], a[20:23], v21, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1059) 


ds_read_u8 v45, v5 offset:32960 // (op 1234) Load local data 



ds_read_b96_tr_b6 v[72:74], v3 offset:224 // (op 1406) Transpose load local data 
ds_read_b96_tr_b6 v[75:77], v3 offset:4320 // (op 1406) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[14:19], v[48:53], a[24:27], v21, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1235) 


ds_read_u8 v46, v5 offset:32992 // (op 1410) Load local data 



ds_read_b96_tr_b6 v[78:80], v0 offset:33824 // (op 1756) Transpose load local data 
ds_read_b96_tr_b6 v[81:83], v0 offset:37920 // (op 1756) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[14:19], v[72:77], a[28:31], v21, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1411) 




v_add_u32 v21, 66592, v4 // (op 1759) 
ds_read_u8 v47, v21 // (op 1759) Load local data 




ds_read_b96_tr_b6 v[14:16], v0 offset:33856 // (op 3294) Transpose load local data 
ds_read_b96_tr_b6 v[17:19], v0 offset:37952 // (op 3294) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:83], v[66:71], a[32:35], v47, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1761) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:83], v[22:27], a[36:39], v47, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1792) 
v_add_u32 v21, 66624, v4 // (op 3297) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:83], v[60:65], a[40:43], v47, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1823) 
ds_read_u8 v84, v21 // (op 3297) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:83], v[30:35], a[44:47], v47, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1854) 
ds_read_b96_tr_b6 v[114:116], v0 offset:33888 // (op 4832) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[78:83], v[54:59], a[48:51], v47, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1885) 
ds_read_b96_tr_b6 v[117:119], v0 offset:37984 // (op 4832) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[78:83], v[38:43], a[52:55], v47, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1916) 
v_add_u32 v21, 66656, v4 // (op 4835) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[78:83], v[48:53], a[56:59], v47, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1947) 
ds_read_u8 v85, v21 // (op 4835) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[78:83], v[72:77], a[60:63], v47, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 1978) 
ds_read_b96_tr_b6 v[86:88], v0 offset:33920 // (op 6370) Transpose load local data 


ds_read_b96_tr_b6 v[89:91], v0 offset:38016 // (op 6370) Transpose load local data 


s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[14:19], v[66:71], a[64:67], v84, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3299) 
v_add_u32 v21, 66688, v4 // (op 6373) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[14:19], v[22:27], a[68:71], v84, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3330) 
ds_read_u8 v47, v21 // (op 6373) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:19], v[60:65], a[72:75], v84, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3361) 
ds_read_b96_tr_b6 v[78:80], v0 offset:33952 // (op 7908) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:19], v[30:35], a[76:79], v84, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3392) 
ds_read_b96_tr_b6 v[81:83], v0 offset:38048 // (op 7908) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:19], v[54:59], a[80:83], v84, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3423) 
v_add_u32 v21, 66720, v4 // (op 7911) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:19], v[38:43], a[84:87], v84, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3454) 
ds_read_u8 v92, v21 // (op 7911) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:19], v[48:53], a[88:91], v84, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3485) 
ds_read_b96_tr_b6 v[130:132], v0 offset:33984 // (op 9446) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[14:19], v[72:77], a[92:95], v84, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 3516) 
ds_read_b96_tr_b6 v[133:135], v0 offset:38080 // (op 9446) Transpose load local data 




s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[114:119], v[66:71], a[96:99], v85, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 4837) 
v_add_u32 v21, 66752, v4 // (op 9449) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[114:119], v[22:27], a[100:103], v85, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 4868) 
ds_read_u8 v84, v21 // (op 9449) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[114:119], v[60:65], a[104:107], v85, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 4899) 
ds_read_b96_tr_b6 v[14:16], v0 offset:34016 // (op 10984) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:119], v[30:35], a[108:111], v85, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 4930) 
ds_read_b96_tr_b6 v[17:19], v0 offset:38112 // (op 10984) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[114:119], v[54:59], a[112:115], v85, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 4961) 
v_add_u32 v21, 66784, v4 // (op 10987) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[114:119], v[38:43], a[116:119], v85, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 4992) 
ds_read_u8 v94, v21 // (op 10987) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[114:119], v[48:53], a[120:123], v85, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5023) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[114:119], v[72:77], a[124:127], v85, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 5054) 


s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:91], v[66:71], a[128:131], v47, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6375) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:91], v[22:27], a[132:135], v47, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6406) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:91], v[60:65], a[136:139], v47, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6437) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:91], v[30:35], a[140:143], v47, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6468) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:91], v[54:59], a[144:147], v47, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6499) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:91], v[38:43], a[148:151], v47, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6530) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:91], v[48:53], a[152:155], v47, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6561) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:91], v[72:77], a[156:159], v47, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 6592) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:83], v[66:71], a[160:163], v92, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 7913) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[78:83], v[22:27], a[164:167], v92, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 7944) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[78:83], v[60:65], a[168:171], v92, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 7975) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:83], v[30:35], a[172:175], v92, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8006) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[78:83], v[54:59], a[176:179], v92, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8037) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[78:83], v[38:43], a[180:183], v92, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8068) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:83], v[48:53], a[184:187], v92, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8099) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:83], v[72:77], a[188:191], v92, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 8130) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:135], v[66:71], a[192:195], v84, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9451) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:135], v[22:27], a[196:199], v84, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9482) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:135], v[60:65], a[200:203], v84, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9513) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:135], v[30:35], a[204:207], v84, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9544) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[130:135], v[54:59], a[208:211], v84, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9575) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:135], v[38:43], a[212:215], v84, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9606) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[130:135], v[48:53], a[216:219], v84, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9637) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[130:135], v[72:77], a[220:223], v84, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 9668) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[14:19], v[66:71], a[224:227], v94, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 10989) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[14:19], v[22:27], a[228:231], v94, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11020) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[14:19], v[60:65], a[232:235], v94, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11051) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[14:19], v[30:35], a[236:239], v94, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11082) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[14:19], v[54:59], a[240:243], v94, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11113) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:19], v[38:43], a[244:247], v94, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11144) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:19], v[48:53], a[248:251], v94, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11175) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:19], v[72:77], a[252:255], v94, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 11206) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 11195) 
s_mov_b32 s19, s0 // (op 11195) convert

s_cmp_lt_u32 s18, s19 // (op 11195) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_11195 // (op 11195) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_11195 if true)
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_11195:























s_barrier  // (op 36461) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133
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


v_accvgpr_read v6, a4 // (op 13689) 
v_accvgpr_read v7, a5 // (op 13689) 
v_accvgpr_read v8, a6 // (op 13689) 
v_accvgpr_read v9, a7 // (op 13689) 
v_mul_f32 v14, v11, v6 // (op 13689) 
v_mul_f32 v15, v11, v7 // (op 13689) 
v_mul_f32 v16, v11, v8 // (op 13689) 
v_mul_f32 v17, v11, v9 // (op 13689) 



s_lshl_b32 s0, s4, 7 // (op 13691) 
v_add_u32 v0, v13, s0 // (op 13691) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13691) Store value


v_accvgpr_read v6, a8 // (op 13701) 
v_accvgpr_read v7, a9 // (op 13701) 
v_accvgpr_read v8, a10 // (op 13701) 
v_accvgpr_read v9, a11 // (op 13701) 
v_mul_f32 v14, v11, v6 // (op 13701) 
v_mul_f32 v15, v11, v7 // (op 13701) 
v_mul_f32 v16, v11, v8 // (op 13701) 
v_mul_f32 v17, v11, v9 // (op 13701) 



s_lshl_b32 s0, s4, 8 // (op 13703) 
v_add_u32 v0, v13, s0 // (op 13703) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13703) Store value


v_accvgpr_read v6, a12 // (op 13713) 
v_accvgpr_read v7, a13 // (op 13713) 
v_accvgpr_read v8, a14 // (op 13713) 
v_accvgpr_read v9, a15 // (op 13713) 
v_mul_f32 v14, v11, v6 // (op 13713) 
v_mul_f32 v15, v11, v7 // (op 13713) 
v_mul_f32 v16, v11, v8 // (op 13713) 
v_mul_f32 v17, v11, v9 // (op 13713) 



s_lshl_b32 s0, s4, 7 // (op 13715) 
s_mul_i32 s1, 3, s0 // (op 13715) 

v_add_u32 v0, v13, s1 // (op 13715) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13715) Store value


v_accvgpr_read v6, a16 // (op 13725) 
v_accvgpr_read v7, a17 // (op 13725) 
v_accvgpr_read v8, a18 // (op 13725) 
v_accvgpr_read v9, a19 // (op 13725) 
v_mul_f32 v14, v11, v6 // (op 13725) 
v_mul_f32 v15, v11, v7 // (op 13725) 
v_mul_f32 v16, v11, v8 // (op 13725) 
v_mul_f32 v17, v11, v9 // (op 13725) 



s_lshl_b32 s0, s4, 9 // (op 13727) 
v_add_u32 v0, v13, s0 // (op 13727) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13727) Store value


v_accvgpr_read v6, a20 // (op 13737) 
v_accvgpr_read v7, a21 // (op 13737) 
v_accvgpr_read v8, a22 // (op 13737) 
v_accvgpr_read v9, a23 // (op 13737) 
v_mul_f32 v14, v11, v6 // (op 13737) 
v_mul_f32 v15, v11, v7 // (op 13737) 
v_mul_f32 v16, v11, v8 // (op 13737) 
v_mul_f32 v17, v11, v9 // (op 13737) 



s_lshl_b32 s0, s4, 7 // (op 13739) 
s_mul_i32 s1, 5, s0 // (op 13739) 

v_add_u32 v0, v13, s1 // (op 13739) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13739) Store value


v_accvgpr_read v6, a24 // (op 13749) 
v_accvgpr_read v7, a25 // (op 13749) 
v_accvgpr_read v8, a26 // (op 13749) 
v_accvgpr_read v9, a27 // (op 13749) 
v_mul_f32 v14, v11, v6 // (op 13749) 
v_mul_f32 v15, v11, v7 // (op 13749) 
v_mul_f32 v16, v11, v8 // (op 13749) 
v_mul_f32 v17, v11, v9 // (op 13749) 



s_lshl_b32 s0, s4, 7 // (op 13751) 
s_mul_i32 s1, 6, s0 // (op 13751) 

v_add_u32 v0, v13, s1 // (op 13751) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13751) Store value


v_accvgpr_read v6, a28 // (op 13761) 
v_accvgpr_read v7, a29 // (op 13761) 
v_accvgpr_read v8, a30 // (op 13761) 
v_accvgpr_read v9, a31 // (op 13761) 
v_mul_f32 v14, v11, v6 // (op 13761) 
v_mul_f32 v15, v11, v7 // (op 13761) 
v_mul_f32 v16, v11, v8 // (op 13761) 
v_mul_f32 v17, v11, v9 // (op 13761) 



s_lshl_b32 s0, s4, 7 // (op 13763) 
s_mul_i32 s1, 7, s0 // (op 13763) 

v_add_u32 v0, v13, s1 // (op 13763) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13763) Store value


v_accvgpr_read v6, a32 // (op 13815) 
v_accvgpr_read v7, a33 // (op 13815) 
v_accvgpr_read v8, a34 // (op 13815) 
v_accvgpr_read v9, a35 // (op 13815) 
v_mul_f32 v14, v11, v6 // (op 13815) 
v_mul_f32 v15, v11, v7 // (op 13815) 
v_mul_f32 v16, v11, v8 // (op 13815) 
v_mul_f32 v17, v11, v9 // (op 13815) 



v_add_u32 v0, 128, v13 // (op 13818) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13818) Store value


v_accvgpr_read v6, a36 // (op 13836) 
v_accvgpr_read v7, a37 // (op 13836) 
v_accvgpr_read v8, a38 // (op 13836) 
v_accvgpr_read v9, a39 // (op 13836) 
v_mul_f32 v14, v11, v6 // (op 13836) 
v_mul_f32 v15, v11, v7 // (op 13836) 
v_mul_f32 v16, v11, v8 // (op 13836) 
v_mul_f32 v17, v11, v9 // (op 13836) 



s_lshl_b32 s0, s4, 7 // (op 13820) 
s_add_u32 s0, s0, 128 // (op 13820) 
v_add_u32 v0, v13, s0 // (op 13820) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13820) Store value


v_accvgpr_read v6, a40 // (op 13841) 
v_accvgpr_read v7, a41 // (op 13841) 
v_accvgpr_read v8, a42 // (op 13841) 
v_accvgpr_read v9, a43 // (op 13841) 
v_mul_f32 v14, v11, v6 // (op 13841) 
v_mul_f32 v15, v11, v7 // (op 13841) 
v_mul_f32 v16, v11, v8 // (op 13841) 
v_mul_f32 v17, v11, v9 // (op 13841) 



s_lshl_b32 s0, s4, 8 // (op 13822) 
s_add_u32 s0, s0, 128 // (op 13822) 
v_add_u32 v0, v13, s0 // (op 13822) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13822) Store value


v_accvgpr_read v6, a44 // (op 13846) 
v_accvgpr_read v7, a45 // (op 13846) 
v_accvgpr_read v8, a46 // (op 13846) 
v_accvgpr_read v9, a47 // (op 13846) 
v_mul_f32 v14, v11, v6 // (op 13846) 
v_mul_f32 v15, v11, v7 // (op 13846) 
v_mul_f32 v16, v11, v8 // (op 13846) 
v_mul_f32 v17, v11, v9 // (op 13846) 



s_lshl_b32 s0, s4, 7 // (op 13824) 
s_mul_i32 s1, 3, s0 // (op 13824) 
s_add_u32 s1, s1, 128 // (op 13824) 

v_add_u32 v0, v13, s1 // (op 13824) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13824) Store value


v_accvgpr_read v6, a48 // (op 13851) 
v_accvgpr_read v7, a49 // (op 13851) 
v_accvgpr_read v8, a50 // (op 13851) 
v_accvgpr_read v9, a51 // (op 13851) 
v_mul_f32 v14, v11, v6 // (op 13851) 
v_mul_f32 v15, v11, v7 // (op 13851) 
v_mul_f32 v16, v11, v8 // (op 13851) 
v_mul_f32 v17, v11, v9 // (op 13851) 



s_lshl_b32 s0, s4, 9 // (op 13826) 
s_add_u32 s0, s0, 128 // (op 13826) 
v_add_u32 v0, v13, s0 // (op 13826) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13826) Store value


v_accvgpr_read v6, a52 // (op 13856) 
v_accvgpr_read v7, a53 // (op 13856) 
v_accvgpr_read v8, a54 // (op 13856) 
v_accvgpr_read v9, a55 // (op 13856) 
v_mul_f32 v14, v11, v6 // (op 13856) 
v_mul_f32 v15, v11, v7 // (op 13856) 
v_mul_f32 v16, v11, v8 // (op 13856) 
v_mul_f32 v17, v11, v9 // (op 13856) 



s_lshl_b32 s0, s4, 7 // (op 13828) 
s_mul_i32 s1, 5, s0 // (op 13828) 
s_add_u32 s1, s1, 128 // (op 13828) 

v_add_u32 v0, v13, s1 // (op 13828) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13828) Store value


v_accvgpr_read v6, a56 // (op 13861) 
v_accvgpr_read v7, a57 // (op 13861) 
v_accvgpr_read v8, a58 // (op 13861) 
v_accvgpr_read v9, a59 // (op 13861) 
v_mul_f32 v14, v11, v6 // (op 13861) 
v_mul_f32 v15, v11, v7 // (op 13861) 
v_mul_f32 v16, v11, v8 // (op 13861) 
v_mul_f32 v17, v11, v9 // (op 13861) 



s_lshl_b32 s0, s4, 7 // (op 13830) 
s_mul_i32 s1, 6, s0 // (op 13830) 
s_add_u32 s1, s1, 128 // (op 13830) 

v_add_u32 v0, v13, s1 // (op 13830) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13830) Store value


v_accvgpr_read v6, a60 // (op 13866) 
v_accvgpr_read v7, a61 // (op 13866) 
v_accvgpr_read v8, a62 // (op 13866) 
v_accvgpr_read v9, a63 // (op 13866) 
v_mul_f32 v14, v11, v6 // (op 13866) 
v_mul_f32 v15, v11, v7 // (op 13866) 
v_mul_f32 v16, v11, v8 // (op 13866) 
v_mul_f32 v17, v11, v9 // (op 13866) 



s_lshl_b32 s0, s4, 7 // (op 13832) 
s_mul_i32 s1, 7, s0 // (op 13832) 
s_add_u32 s1, s1, 128 // (op 13832) 

v_add_u32 v0, v13, s1 // (op 13832) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13832) Store value


v_accvgpr_read v6, a64 // (op 13947) 
v_accvgpr_read v7, a65 // (op 13947) 
v_accvgpr_read v8, a66 // (op 13947) 
v_accvgpr_read v9, a67 // (op 13947) 
v_mul_f32 v14, v11, v6 // (op 13947) 
v_mul_f32 v15, v11, v7 // (op 13947) 
v_mul_f32 v16, v11, v8 // (op 13947) 
v_mul_f32 v17, v11, v9 // (op 13947) 



v_add_u32 v0, 256, v13 // (op 13950) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13950) Store value


v_accvgpr_read v6, a68 // (op 13968) 
v_accvgpr_read v7, a69 // (op 13968) 
v_accvgpr_read v8, a70 // (op 13968) 
v_accvgpr_read v9, a71 // (op 13968) 
v_mul_f32 v14, v11, v6 // (op 13968) 
v_mul_f32 v15, v11, v7 // (op 13968) 
v_mul_f32 v16, v11, v8 // (op 13968) 
v_mul_f32 v17, v11, v9 // (op 13968) 



s_lshl_b32 s0, s4, 7 // (op 13952) 
s_add_u32 s0, s0, 256 // (op 13952) 
v_add_u32 v0, v13, s0 // (op 13952) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13952) Store value


v_accvgpr_read v6, a72 // (op 13973) 
v_accvgpr_read v7, a73 // (op 13973) 
v_accvgpr_read v8, a74 // (op 13973) 
v_accvgpr_read v9, a75 // (op 13973) 
v_mul_f32 v14, v11, v6 // (op 13973) 
v_mul_f32 v15, v11, v7 // (op 13973) 
v_mul_f32 v16, v11, v8 // (op 13973) 
v_mul_f32 v17, v11, v9 // (op 13973) 



s_lshl_b32 s0, s4, 8 // (op 13954) 
s_add_u32 s0, s0, 256 // (op 13954) 
v_add_u32 v0, v13, s0 // (op 13954) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13954) Store value


v_accvgpr_read v6, a76 // (op 13978) 
v_accvgpr_read v7, a77 // (op 13978) 
v_accvgpr_read v8, a78 // (op 13978) 
v_accvgpr_read v9, a79 // (op 13978) 
v_mul_f32 v14, v11, v6 // (op 13978) 
v_mul_f32 v15, v11, v7 // (op 13978) 
v_mul_f32 v16, v11, v8 // (op 13978) 
v_mul_f32 v17, v11, v9 // (op 13978) 



s_lshl_b32 s0, s4, 7 // (op 13956) 
s_mul_i32 s1, 3, s0 // (op 13956) 
s_add_u32 s1, s1, 256 // (op 13956) 

v_add_u32 v0, v13, s1 // (op 13956) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13956) Store value


v_accvgpr_read v6, a80 // (op 13983) 
v_accvgpr_read v7, a81 // (op 13983) 
v_accvgpr_read v8, a82 // (op 13983) 
v_accvgpr_read v9, a83 // (op 13983) 
v_mul_f32 v14, v11, v6 // (op 13983) 
v_mul_f32 v15, v11, v7 // (op 13983) 
v_mul_f32 v16, v11, v8 // (op 13983) 
v_mul_f32 v17, v11, v9 // (op 13983) 



s_lshl_b32 s0, s4, 9 // (op 13958) 
s_add_u32 s0, s0, 256 // (op 13958) 
v_add_u32 v0, v13, s0 // (op 13958) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13958) Store value


v_accvgpr_read v6, a84 // (op 13988) 
v_accvgpr_read v7, a85 // (op 13988) 
v_accvgpr_read v8, a86 // (op 13988) 
v_accvgpr_read v9, a87 // (op 13988) 
v_mul_f32 v14, v11, v6 // (op 13988) 
v_mul_f32 v15, v11, v7 // (op 13988) 
v_mul_f32 v16, v11, v8 // (op 13988) 
v_mul_f32 v17, v11, v9 // (op 13988) 



s_lshl_b32 s0, s4, 7 // (op 13960) 
s_mul_i32 s1, 5, s0 // (op 13960) 
s_add_u32 s1, s1, 256 // (op 13960) 

v_add_u32 v0, v13, s1 // (op 13960) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13960) Store value


v_accvgpr_read v6, a88 // (op 13993) 
v_accvgpr_read v7, a89 // (op 13993) 
v_accvgpr_read v8, a90 // (op 13993) 
v_accvgpr_read v9, a91 // (op 13993) 
v_mul_f32 v14, v11, v6 // (op 13993) 
v_mul_f32 v15, v11, v7 // (op 13993) 
v_mul_f32 v16, v11, v8 // (op 13993) 
v_mul_f32 v17, v11, v9 // (op 13993) 



s_lshl_b32 s0, s4, 7 // (op 13962) 
s_mul_i32 s1, 6, s0 // (op 13962) 
s_add_u32 s1, s1, 256 // (op 13962) 

v_add_u32 v0, v13, s1 // (op 13962) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13962) Store value


v_accvgpr_read v6, a92 // (op 13998) 
v_accvgpr_read v7, a93 // (op 13998) 
v_accvgpr_read v8, a94 // (op 13998) 
v_accvgpr_read v9, a95 // (op 13998) 
v_mul_f32 v14, v11, v6 // (op 13998) 
v_mul_f32 v15, v11, v7 // (op 13998) 
v_mul_f32 v16, v11, v8 // (op 13998) 
v_mul_f32 v17, v11, v9 // (op 13998) 



s_lshl_b32 s0, s4, 7 // (op 13964) 
s_mul_i32 s1, 7, s0 // (op 13964) 
s_add_u32 s1, s1, 256 // (op 13964) 

v_add_u32 v0, v13, s1 // (op 13964) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 13964) Store value


v_accvgpr_read v6, a96 // (op 14079) 
v_accvgpr_read v7, a97 // (op 14079) 
v_accvgpr_read v8, a98 // (op 14079) 
v_accvgpr_read v9, a99 // (op 14079) 
v_mul_f32 v14, v11, v6 // (op 14079) 
v_mul_f32 v15, v11, v7 // (op 14079) 
v_mul_f32 v16, v11, v8 // (op 14079) 
v_mul_f32 v17, v11, v9 // (op 14079) 



v_add_u32 v0, 384, v13 // (op 14082) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14082) Store value


v_accvgpr_read v6, a100 // (op 14100) 
v_accvgpr_read v7, a101 // (op 14100) 
v_accvgpr_read v8, a102 // (op 14100) 
v_accvgpr_read v9, a103 // (op 14100) 
v_mul_f32 v14, v11, v6 // (op 14100) 
v_mul_f32 v15, v11, v7 // (op 14100) 
v_mul_f32 v16, v11, v8 // (op 14100) 
v_mul_f32 v17, v11, v9 // (op 14100) 



s_lshl_b32 s0, s4, 7 // (op 14084) 
s_add_u32 s0, s0, 384 // (op 14084) 
v_add_u32 v0, v13, s0 // (op 14084) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14084) Store value


v_accvgpr_read v6, a104 // (op 14105) 
v_accvgpr_read v7, a105 // (op 14105) 
v_accvgpr_read v8, a106 // (op 14105) 
v_accvgpr_read v9, a107 // (op 14105) 
v_mul_f32 v14, v11, v6 // (op 14105) 
v_mul_f32 v15, v11, v7 // (op 14105) 
v_mul_f32 v16, v11, v8 // (op 14105) 
v_mul_f32 v17, v11, v9 // (op 14105) 



s_lshl_b32 s0, s4, 8 // (op 14086) 
s_add_u32 s0, s0, 384 // (op 14086) 
v_add_u32 v0, v13, s0 // (op 14086) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14086) Store value


v_accvgpr_read v6, a108 // (op 14110) 
v_accvgpr_read v7, a109 // (op 14110) 
v_accvgpr_read v8, a110 // (op 14110) 
v_accvgpr_read v9, a111 // (op 14110) 
v_mul_f32 v14, v11, v6 // (op 14110) 
v_mul_f32 v15, v11, v7 // (op 14110) 
v_mul_f32 v16, v11, v8 // (op 14110) 
v_mul_f32 v17, v11, v9 // (op 14110) 



s_lshl_b32 s0, s4, 7 // (op 14088) 
s_mul_i32 s1, 3, s0 // (op 14088) 
s_add_u32 s1, s1, 384 // (op 14088) 

v_add_u32 v0, v13, s1 // (op 14088) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14088) Store value


v_accvgpr_read v6, a112 // (op 14115) 
v_accvgpr_read v7, a113 // (op 14115) 
v_accvgpr_read v8, a114 // (op 14115) 
v_accvgpr_read v9, a115 // (op 14115) 
v_mul_f32 v14, v11, v6 // (op 14115) 
v_mul_f32 v15, v11, v7 // (op 14115) 
v_mul_f32 v16, v11, v8 // (op 14115) 
v_mul_f32 v17, v11, v9 // (op 14115) 



s_lshl_b32 s0, s4, 9 // (op 14090) 
s_add_u32 s0, s0, 384 // (op 14090) 
v_add_u32 v0, v13, s0 // (op 14090) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14090) Store value


v_accvgpr_read v6, a116 // (op 14120) 
v_accvgpr_read v7, a117 // (op 14120) 
v_accvgpr_read v8, a118 // (op 14120) 
v_accvgpr_read v9, a119 // (op 14120) 
v_mul_f32 v14, v11, v6 // (op 14120) 
v_mul_f32 v15, v11, v7 // (op 14120) 
v_mul_f32 v16, v11, v8 // (op 14120) 
v_mul_f32 v17, v11, v9 // (op 14120) 



s_lshl_b32 s0, s4, 7 // (op 14092) 
s_mul_i32 s1, 5, s0 // (op 14092) 
s_add_u32 s1, s1, 384 // (op 14092) 

v_add_u32 v0, v13, s1 // (op 14092) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14092) Store value


v_accvgpr_read v6, a120 // (op 14125) 
v_accvgpr_read v7, a121 // (op 14125) 
v_accvgpr_read v8, a122 // (op 14125) 
v_accvgpr_read v9, a123 // (op 14125) 
v_mul_f32 v14, v11, v6 // (op 14125) 
v_mul_f32 v15, v11, v7 // (op 14125) 
v_mul_f32 v16, v11, v8 // (op 14125) 
v_mul_f32 v17, v11, v9 // (op 14125) 



s_lshl_b32 s0, s4, 7 // (op 14094) 
s_mul_i32 s1, 6, s0 // (op 14094) 
s_add_u32 s1, s1, 384 // (op 14094) 

v_add_u32 v0, v13, s1 // (op 14094) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14094) Store value


v_accvgpr_read v6, a124 // (op 14130) 
v_accvgpr_read v7, a125 // (op 14130) 
v_accvgpr_read v8, a126 // (op 14130) 
v_accvgpr_read v9, a127 // (op 14130) 
v_mul_f32 v14, v11, v6 // (op 14130) 
v_mul_f32 v15, v11, v7 // (op 14130) 
v_mul_f32 v16, v11, v8 // (op 14130) 
v_mul_f32 v17, v11, v9 // (op 14130) 



s_lshl_b32 s0, s4, 7 // (op 14096) 
s_mul_i32 s1, 7, s0 // (op 14096) 
s_add_u32 s1, s1, 384 // (op 14096) 

v_add_u32 v0, v13, s1 // (op 14096) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14096) Store value


v_accvgpr_read v6, a128 // (op 14211) 
v_accvgpr_read v7, a129 // (op 14211) 
v_accvgpr_read v8, a130 // (op 14211) 
v_accvgpr_read v9, a131 // (op 14211) 
v_mul_f32 v14, v11, v6 // (op 14211) 
v_mul_f32 v15, v11, v7 // (op 14211) 
v_mul_f32 v16, v11, v8 // (op 14211) 
v_mul_f32 v17, v11, v9 // (op 14211) 



v_add_u32 v0, 512, v13 // (op 14214) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14214) Store value


v_accvgpr_read v6, a132 // (op 14232) 
v_accvgpr_read v7, a133 // (op 14232) 
v_accvgpr_read v8, a134 // (op 14232) 
v_accvgpr_read v9, a135 // (op 14232) 
v_mul_f32 v14, v11, v6 // (op 14232) 
v_mul_f32 v15, v11, v7 // (op 14232) 
v_mul_f32 v16, v11, v8 // (op 14232) 
v_mul_f32 v17, v11, v9 // (op 14232) 



s_lshl_b32 s0, s4, 7 // (op 14216) 
s_add_u32 s0, s0, 512 // (op 14216) 
v_add_u32 v0, v13, s0 // (op 14216) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14216) Store value


v_accvgpr_read v6, a136 // (op 14237) 
v_accvgpr_read v7, a137 // (op 14237) 
v_accvgpr_read v8, a138 // (op 14237) 
v_accvgpr_read v9, a139 // (op 14237) 
v_mul_f32 v14, v11, v6 // (op 14237) 
v_mul_f32 v15, v11, v7 // (op 14237) 
v_mul_f32 v16, v11, v8 // (op 14237) 
v_mul_f32 v17, v11, v9 // (op 14237) 



s_lshl_b32 s0, s4, 8 // (op 14218) 
s_add_u32 s0, s0, 512 // (op 14218) 
v_add_u32 v0, v13, s0 // (op 14218) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14218) Store value


v_accvgpr_read v6, a140 // (op 14242) 
v_accvgpr_read v7, a141 // (op 14242) 
v_accvgpr_read v8, a142 // (op 14242) 
v_accvgpr_read v9, a143 // (op 14242) 
v_mul_f32 v14, v11, v6 // (op 14242) 
v_mul_f32 v15, v11, v7 // (op 14242) 
v_mul_f32 v16, v11, v8 // (op 14242) 
v_mul_f32 v17, v11, v9 // (op 14242) 



s_lshl_b32 s0, s4, 7 // (op 14220) 
s_mul_i32 s1, 3, s0 // (op 14220) 
s_add_u32 s1, s1, 512 // (op 14220) 

v_add_u32 v0, v13, s1 // (op 14220) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14220) Store value


v_accvgpr_read v6, a144 // (op 14247) 
v_accvgpr_read v7, a145 // (op 14247) 
v_accvgpr_read v8, a146 // (op 14247) 
v_accvgpr_read v9, a147 // (op 14247) 
v_mul_f32 v14, v11, v6 // (op 14247) 
v_mul_f32 v15, v11, v7 // (op 14247) 
v_mul_f32 v16, v11, v8 // (op 14247) 
v_mul_f32 v17, v11, v9 // (op 14247) 



s_lshl_b32 s0, s4, 9 // (op 14222) 
s_add_u32 s0, s0, 512 // (op 14222) 
v_add_u32 v0, v13, s0 // (op 14222) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14222) Store value


v_accvgpr_read v6, a148 // (op 14252) 
v_accvgpr_read v7, a149 // (op 14252) 
v_accvgpr_read v8, a150 // (op 14252) 
v_accvgpr_read v9, a151 // (op 14252) 
v_mul_f32 v14, v11, v6 // (op 14252) 
v_mul_f32 v15, v11, v7 // (op 14252) 
v_mul_f32 v16, v11, v8 // (op 14252) 
v_mul_f32 v17, v11, v9 // (op 14252) 



s_lshl_b32 s0, s4, 7 // (op 14224) 
s_mul_i32 s1, 5, s0 // (op 14224) 
s_add_u32 s1, s1, 512 // (op 14224) 

v_add_u32 v0, v13, s1 // (op 14224) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14224) Store value


v_accvgpr_read v6, a152 // (op 14257) 
v_accvgpr_read v7, a153 // (op 14257) 
v_accvgpr_read v8, a154 // (op 14257) 
v_accvgpr_read v9, a155 // (op 14257) 
v_mul_f32 v14, v11, v6 // (op 14257) 
v_mul_f32 v15, v11, v7 // (op 14257) 
v_mul_f32 v16, v11, v8 // (op 14257) 
v_mul_f32 v17, v11, v9 // (op 14257) 



s_lshl_b32 s0, s4, 7 // (op 14226) 
s_mul_i32 s1, 6, s0 // (op 14226) 
s_add_u32 s1, s1, 512 // (op 14226) 

v_add_u32 v0, v13, s1 // (op 14226) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14226) Store value


v_accvgpr_read v6, a156 // (op 14262) 
v_accvgpr_read v7, a157 // (op 14262) 
v_accvgpr_read v8, a158 // (op 14262) 
v_accvgpr_read v9, a159 // (op 14262) 
v_mul_f32 v14, v11, v6 // (op 14262) 
v_mul_f32 v15, v11, v7 // (op 14262) 
v_mul_f32 v16, v11, v8 // (op 14262) 
v_mul_f32 v17, v11, v9 // (op 14262) 



s_lshl_b32 s0, s4, 7 // (op 14228) 
s_mul_i32 s1, 7, s0 // (op 14228) 
s_add_u32 s1, s1, 512 // (op 14228) 

v_add_u32 v0, v13, s1 // (op 14228) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14228) Store value


v_accvgpr_read v6, a160 // (op 14343) 
v_accvgpr_read v7, a161 // (op 14343) 
v_accvgpr_read v8, a162 // (op 14343) 
v_accvgpr_read v9, a163 // (op 14343) 
v_mul_f32 v14, v11, v6 // (op 14343) 
v_mul_f32 v15, v11, v7 // (op 14343) 
v_mul_f32 v16, v11, v8 // (op 14343) 
v_mul_f32 v17, v11, v9 // (op 14343) 



v_add_u32 v0, 640, v13 // (op 14346) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14346) Store value


v_accvgpr_read v6, a164 // (op 14364) 
v_accvgpr_read v7, a165 // (op 14364) 
v_accvgpr_read v8, a166 // (op 14364) 
v_accvgpr_read v9, a167 // (op 14364) 
v_mul_f32 v14, v11, v6 // (op 14364) 
v_mul_f32 v15, v11, v7 // (op 14364) 
v_mul_f32 v16, v11, v8 // (op 14364) 
v_mul_f32 v17, v11, v9 // (op 14364) 



s_lshl_b32 s0, s4, 7 // (op 14348) 
s_add_u32 s0, s0, 640 // (op 14348) 
v_add_u32 v0, v13, s0 // (op 14348) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14348) Store value


v_accvgpr_read v6, a168 // (op 14369) 
v_accvgpr_read v7, a169 // (op 14369) 
v_accvgpr_read v8, a170 // (op 14369) 
v_accvgpr_read v9, a171 // (op 14369) 
v_mul_f32 v14, v11, v6 // (op 14369) 
v_mul_f32 v15, v11, v7 // (op 14369) 
v_mul_f32 v16, v11, v8 // (op 14369) 
v_mul_f32 v17, v11, v9 // (op 14369) 



s_lshl_b32 s0, s4, 8 // (op 14350) 
s_add_u32 s0, s0, 640 // (op 14350) 
v_add_u32 v0, v13, s0 // (op 14350) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14350) Store value


v_accvgpr_read v6, a172 // (op 14374) 
v_accvgpr_read v7, a173 // (op 14374) 
v_accvgpr_read v8, a174 // (op 14374) 
v_accvgpr_read v9, a175 // (op 14374) 
v_mul_f32 v14, v11, v6 // (op 14374) 
v_mul_f32 v15, v11, v7 // (op 14374) 
v_mul_f32 v16, v11, v8 // (op 14374) 
v_mul_f32 v17, v11, v9 // (op 14374) 



s_lshl_b32 s0, s4, 7 // (op 14352) 
s_mul_i32 s1, 3, s0 // (op 14352) 
s_add_u32 s1, s1, 640 // (op 14352) 

v_add_u32 v0, v13, s1 // (op 14352) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14352) Store value


v_accvgpr_read v6, a176 // (op 14379) 
v_accvgpr_read v7, a177 // (op 14379) 
v_accvgpr_read v8, a178 // (op 14379) 
v_accvgpr_read v9, a179 // (op 14379) 
v_mul_f32 v14, v11, v6 // (op 14379) 
v_mul_f32 v15, v11, v7 // (op 14379) 
v_mul_f32 v16, v11, v8 // (op 14379) 
v_mul_f32 v17, v11, v9 // (op 14379) 



s_lshl_b32 s0, s4, 9 // (op 14354) 
s_add_u32 s0, s0, 640 // (op 14354) 
v_add_u32 v0, v13, s0 // (op 14354) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14354) Store value


v_accvgpr_read v6, a180 // (op 14384) 
v_accvgpr_read v7, a181 // (op 14384) 
v_accvgpr_read v8, a182 // (op 14384) 
v_accvgpr_read v9, a183 // (op 14384) 
v_mul_f32 v14, v11, v6 // (op 14384) 
v_mul_f32 v15, v11, v7 // (op 14384) 
v_mul_f32 v16, v11, v8 // (op 14384) 
v_mul_f32 v17, v11, v9 // (op 14384) 



s_lshl_b32 s0, s4, 7 // (op 14356) 
s_mul_i32 s1, 5, s0 // (op 14356) 
s_add_u32 s1, s1, 640 // (op 14356) 

v_add_u32 v0, v13, s1 // (op 14356) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14356) Store value


v_accvgpr_read v6, a184 // (op 14389) 
v_accvgpr_read v7, a185 // (op 14389) 
v_accvgpr_read v8, a186 // (op 14389) 
v_accvgpr_read v9, a187 // (op 14389) 
v_mul_f32 v14, v11, v6 // (op 14389) 
v_mul_f32 v15, v11, v7 // (op 14389) 
v_mul_f32 v16, v11, v8 // (op 14389) 
v_mul_f32 v17, v11, v9 // (op 14389) 



s_lshl_b32 s0, s4, 7 // (op 14358) 
s_mul_i32 s1, 6, s0 // (op 14358) 
s_add_u32 s1, s1, 640 // (op 14358) 

v_add_u32 v0, v13, s1 // (op 14358) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14358) Store value


v_accvgpr_read v6, a188 // (op 14394) 
v_accvgpr_read v7, a189 // (op 14394) 
v_accvgpr_read v8, a190 // (op 14394) 
v_accvgpr_read v9, a191 // (op 14394) 
v_mul_f32 v14, v11, v6 // (op 14394) 
v_mul_f32 v15, v11, v7 // (op 14394) 
v_mul_f32 v16, v11, v8 // (op 14394) 
v_mul_f32 v17, v11, v9 // (op 14394) 



s_lshl_b32 s0, s4, 7 // (op 14360) 
s_mul_i32 s1, 7, s0 // (op 14360) 
s_add_u32 s1, s1, 640 // (op 14360) 

v_add_u32 v0, v13, s1 // (op 14360) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14360) Store value


v_accvgpr_read v6, a192 // (op 14475) 
v_accvgpr_read v7, a193 // (op 14475) 
v_accvgpr_read v8, a194 // (op 14475) 
v_accvgpr_read v9, a195 // (op 14475) 
v_mul_f32 v14, v11, v6 // (op 14475) 
v_mul_f32 v15, v11, v7 // (op 14475) 
v_mul_f32 v16, v11, v8 // (op 14475) 
v_mul_f32 v17, v11, v9 // (op 14475) 



v_add_u32 v0, 768, v13 // (op 14478) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14478) Store value


v_accvgpr_read v6, a196 // (op 14496) 
v_accvgpr_read v7, a197 // (op 14496) 
v_accvgpr_read v8, a198 // (op 14496) 
v_accvgpr_read v9, a199 // (op 14496) 
v_mul_f32 v14, v11, v6 // (op 14496) 
v_mul_f32 v15, v11, v7 // (op 14496) 
v_mul_f32 v16, v11, v8 // (op 14496) 
v_mul_f32 v17, v11, v9 // (op 14496) 



s_lshl_b32 s0, s4, 7 // (op 14480) 
s_add_u32 s0, s0, 768 // (op 14480) 
v_add_u32 v0, v13, s0 // (op 14480) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14480) Store value


v_accvgpr_read v6, a200 // (op 14501) 
v_accvgpr_read v7, a201 // (op 14501) 
v_accvgpr_read v8, a202 // (op 14501) 
v_accvgpr_read v9, a203 // (op 14501) 
v_mul_f32 v14, v11, v6 // (op 14501) 
v_mul_f32 v15, v11, v7 // (op 14501) 
v_mul_f32 v16, v11, v8 // (op 14501) 
v_mul_f32 v17, v11, v9 // (op 14501) 



s_lshl_b32 s0, s4, 8 // (op 14482) 
s_add_u32 s0, s0, 768 // (op 14482) 
v_add_u32 v0, v13, s0 // (op 14482) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14482) Store value


v_accvgpr_read v6, a204 // (op 14506) 
v_accvgpr_read v7, a205 // (op 14506) 
v_accvgpr_read v8, a206 // (op 14506) 
v_accvgpr_read v9, a207 // (op 14506) 
v_mul_f32 v14, v11, v6 // (op 14506) 
v_mul_f32 v15, v11, v7 // (op 14506) 
v_mul_f32 v16, v11, v8 // (op 14506) 
v_mul_f32 v17, v11, v9 // (op 14506) 



s_lshl_b32 s0, s4, 7 // (op 14484) 
s_mul_i32 s1, 3, s0 // (op 14484) 
s_add_u32 s1, s1, 768 // (op 14484) 

v_add_u32 v0, v13, s1 // (op 14484) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14484) Store value


v_accvgpr_read v6, a208 // (op 14511) 
v_accvgpr_read v7, a209 // (op 14511) 
v_accvgpr_read v8, a210 // (op 14511) 
v_accvgpr_read v9, a211 // (op 14511) 
v_mul_f32 v14, v11, v6 // (op 14511) 
v_mul_f32 v15, v11, v7 // (op 14511) 
v_mul_f32 v16, v11, v8 // (op 14511) 
v_mul_f32 v17, v11, v9 // (op 14511) 



s_lshl_b32 s0, s4, 9 // (op 14486) 
s_add_u32 s0, s0, 768 // (op 14486) 
v_add_u32 v0, v13, s0 // (op 14486) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14486) Store value


v_accvgpr_read v6, a212 // (op 14516) 
v_accvgpr_read v7, a213 // (op 14516) 
v_accvgpr_read v8, a214 // (op 14516) 
v_accvgpr_read v9, a215 // (op 14516) 
v_mul_f32 v14, v11, v6 // (op 14516) 
v_mul_f32 v15, v11, v7 // (op 14516) 
v_mul_f32 v16, v11, v8 // (op 14516) 
v_mul_f32 v17, v11, v9 // (op 14516) 



s_lshl_b32 s0, s4, 7 // (op 14488) 
s_mul_i32 s1, 5, s0 // (op 14488) 
s_add_u32 s1, s1, 768 // (op 14488) 

v_add_u32 v0, v13, s1 // (op 14488) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14488) Store value


v_accvgpr_read v6, a216 // (op 14521) 
v_accvgpr_read v7, a217 // (op 14521) 
v_accvgpr_read v8, a218 // (op 14521) 
v_accvgpr_read v9, a219 // (op 14521) 
v_mul_f32 v14, v11, v6 // (op 14521) 
v_mul_f32 v15, v11, v7 // (op 14521) 
v_mul_f32 v16, v11, v8 // (op 14521) 
v_mul_f32 v17, v11, v9 // (op 14521) 



s_lshl_b32 s0, s4, 7 // (op 14490) 
s_mul_i32 s1, 6, s0 // (op 14490) 
s_add_u32 s1, s1, 768 // (op 14490) 

v_add_u32 v0, v13, s1 // (op 14490) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14490) Store value


v_accvgpr_read v6, a220 // (op 14526) 
v_accvgpr_read v7, a221 // (op 14526) 
v_accvgpr_read v8, a222 // (op 14526) 
v_accvgpr_read v9, a223 // (op 14526) 
v_mul_f32 v14, v11, v6 // (op 14526) 
v_mul_f32 v15, v11, v7 // (op 14526) 
v_mul_f32 v16, v11, v8 // (op 14526) 
v_mul_f32 v17, v11, v9 // (op 14526) 



s_lshl_b32 s0, s4, 7 // (op 14492) 
s_mul_i32 s1, 7, s0 // (op 14492) 
s_add_u32 s1, s1, 768 // (op 14492) 

v_add_u32 v0, v13, s1 // (op 14492) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14492) Store value


v_accvgpr_read v6, a224 // (op 14607) 
v_accvgpr_read v7, a225 // (op 14607) 
v_accvgpr_read v8, a226 // (op 14607) 
v_accvgpr_read v9, a227 // (op 14607) 
v_mul_f32 v14, v11, v6 // (op 14607) 
v_mul_f32 v15, v11, v7 // (op 14607) 
v_mul_f32 v16, v11, v8 // (op 14607) 
v_mul_f32 v17, v11, v9 // (op 14607) 



v_add_u32 v0, 896, v13 // (op 14610) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14610) Store value


v_accvgpr_read v6, a228 // (op 14628) 
v_accvgpr_read v7, a229 // (op 14628) 
v_accvgpr_read v8, a230 // (op 14628) 
v_accvgpr_read v9, a231 // (op 14628) 
v_mul_f32 v14, v11, v6 // (op 14628) 
v_mul_f32 v15, v11, v7 // (op 14628) 
v_mul_f32 v16, v11, v8 // (op 14628) 
v_mul_f32 v17, v11, v9 // (op 14628) 



s_lshl_b32 s0, s4, 7 // (op 14612) 
s_add_u32 s0, s0, 896 // (op 14612) 
v_add_u32 v0, v13, s0 // (op 14612) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14612) Store value


v_accvgpr_read v6, a232 // (op 14633) 
v_accvgpr_read v7, a233 // (op 14633) 
v_accvgpr_read v8, a234 // (op 14633) 
v_accvgpr_read v9, a235 // (op 14633) 
v_mul_f32 v14, v11, v6 // (op 14633) 
v_mul_f32 v15, v11, v7 // (op 14633) 
v_mul_f32 v16, v11, v8 // (op 14633) 
v_mul_f32 v17, v11, v9 // (op 14633) 



s_lshl_b32 s0, s4, 8 // (op 14614) 
s_add_u32 s0, s0, 896 // (op 14614) 
v_add_u32 v0, v13, s0 // (op 14614) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14614) Store value


v_accvgpr_read v6, a236 // (op 14638) 
v_accvgpr_read v7, a237 // (op 14638) 
v_accvgpr_read v8, a238 // (op 14638) 
v_accvgpr_read v9, a239 // (op 14638) 
v_mul_f32 v14, v11, v6 // (op 14638) 
v_mul_f32 v15, v11, v7 // (op 14638) 
v_mul_f32 v16, v11, v8 // (op 14638) 
v_mul_f32 v17, v11, v9 // (op 14638) 



s_lshl_b32 s0, s4, 7 // (op 14616) 
s_mul_i32 s1, 3, s0 // (op 14616) 
s_add_u32 s1, s1, 896 // (op 14616) 

v_add_u32 v0, v13, s1 // (op 14616) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14616) Store value


v_accvgpr_read v6, a240 // (op 14643) 
v_accvgpr_read v7, a241 // (op 14643) 
v_accvgpr_read v8, a242 // (op 14643) 
v_accvgpr_read v9, a243 // (op 14643) 
v_mul_f32 v14, v11, v6 // (op 14643) 
v_mul_f32 v15, v11, v7 // (op 14643) 
v_mul_f32 v16, v11, v8 // (op 14643) 
v_mul_f32 v17, v11, v9 // (op 14643) 



s_lshl_b32 s0, s4, 9 // (op 14618) 
s_add_u32 s0, s0, 896 // (op 14618) 
v_add_u32 v0, v13, s0 // (op 14618) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14618) Store value


v_accvgpr_read v6, a244 // (op 14648) 
v_accvgpr_read v7, a245 // (op 14648) 
v_accvgpr_read v8, a246 // (op 14648) 
v_accvgpr_read v9, a247 // (op 14648) 
v_mul_f32 v14, v11, v6 // (op 14648) 
v_mul_f32 v15, v11, v7 // (op 14648) 
v_mul_f32 v16, v11, v8 // (op 14648) 
v_mul_f32 v17, v11, v9 // (op 14648) 



s_lshl_b32 s0, s4, 7 // (op 14620) 
s_mul_i32 s1, 5, s0 // (op 14620) 
s_add_u32 s1, s1, 896 // (op 14620) 

v_add_u32 v0, v13, s1 // (op 14620) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14620) Store value


v_accvgpr_read v6, a248 // (op 14653) 
v_accvgpr_read v7, a249 // (op 14653) 
v_accvgpr_read v8, a250 // (op 14653) 
v_accvgpr_read v9, a251 // (op 14653) 
v_mul_f32 v14, v11, v6 // (op 14653) 
v_mul_f32 v15, v11, v7 // (op 14653) 
v_mul_f32 v16, v11, v8 // (op 14653) 
v_mul_f32 v17, v11, v9 // (op 14653) 



s_lshl_b32 s0, s4, 7 // (op 14622) 
s_mul_i32 s1, 6, s0 // (op 14622) 
s_add_u32 s1, s1, 896 // (op 14622) 

v_add_u32 v0, v13, s1 // (op 14622) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14622) Store value


v_accvgpr_read v6, a252 // (op 14658) 
v_accvgpr_read v7, a253 // (op 14658) 
v_accvgpr_read v8, a254 // (op 14658) 
v_accvgpr_read v9, a255 // (op 14658) 
v_mul_f32 v14, v11, v6 // (op 14658) 
v_mul_f32 v15, v11, v7 // (op 14658) 
v_mul_f32 v16, v11, v8 // (op 14658) 
v_mul_f32 v17, v11, v9 // (op 14658) 



s_lshl_b32 s0, s4, 7 // (op 14624) 
s_mul_i32 s1, 7, s0 // (op 14624) 
s_add_u32 s1, s1, 896 // (op 14624) 

v_add_u32 v0, v13, s1 // (op 14624) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 14624) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 896, v12 // (op 16058) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 16058) Load value

v_accvgpr_read v14, a224 // (op 16038) 
v_accvgpr_read v15, a225 // (op 16038) 
v_accvgpr_read v16, a226 // (op 16038) 
v_accvgpr_read v17, a227 // (op 16038) 
v_mul_f32 v18, v11, v14 // (op 16038) 
v_mul_f32 v19, v11, v15 // (op 16038) 
v_mul_f32 v20, v11, v16 // (op 16038) 
v_mul_f32 v21, v11, v17 // (op 16038) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 16057) 
v_mul_f32 v15, v10, v7 // (op 16057) 
v_mul_f32 v16, v10, v8 // (op 16057) 
v_mul_f32 v17, v10, v9 // (op 16057) 

v_add_f32 v6, v14, v18 // (op 16039) 
v_add_f32 v7, v15, v19 // (op 16039) 
v_add_f32 v8, v16, v20 // (op 16039) 
v_add_f32 v9, v17, v21 // (op 16039) 




v_add_u32 v0, 896, v13 // (op 16041) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 16041) Store value




s_lshl_b32 s0, s2, 7 // (op 16062) 
s_add_u32 s0, s0, 896 // (op 16062) 
v_add_u32 v0, v12, s0 // (op 16062) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 16062) Load value

v_accvgpr_read v14, a228 // (op 16088) 
v_accvgpr_read v15, a229 // (op 16088) 
v_accvgpr_read v16, a230 // (op 16088) 
v_accvgpr_read v17, a231 // (op 16088) 
v_mul_f32 v18, v11, v14 // (op 16088) 
v_mul_f32 v19, v11, v15 // (op 16088) 
v_mul_f32 v20, v11, v16 // (op 16088) 
v_mul_f32 v21, v11, v17 // (op 16088) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 16060) 
v_mul_f32 v15, v10, v7 // (op 16060) 
v_mul_f32 v16, v10, v8 // (op 16060) 
v_mul_f32 v17, v10, v9 // (op 16060) 

v_add_f32 v6, v14, v18 // (op 16061) 
v_add_f32 v7, v15, v19 // (op 16061) 
v_add_f32 v8, v16, v20 // (op 16061) 
v_add_f32 v9, v17, v21 // (op 16061) 




s_lshl_b32 s0, s4, 7 // (op 16043) 
s_add_u32 s0, s0, 896 // (op 16043) 
v_add_u32 v0, v13, s0 // (op 16043) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 16043) Store value




s_lshl_b32 s0, s2, 8 // (op 16066) 
s_add_u32 s0, s0, 896 // (op 16066) 
v_add_u32 v0, v12, s0 // (op 16066) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 16066) Load value

v_accvgpr_read v14, a232 // (op 16090) 
v_accvgpr_read v15, a233 // (op 16090) 
v_accvgpr_read v16, a234 // (op 16090) 
v_accvgpr_read v17, a235 // (op 16090) 
v_mul_f32 v18, v11, v14 // (op 16090) 
v_mul_f32 v19, v11, v15 // (op 16090) 
v_mul_f32 v20, v11, v16 // (op 16090) 
v_mul_f32 v21, v11, v17 // (op 16090) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 16064) 
v_mul_f32 v15, v10, v7 // (op 16064) 
v_mul_f32 v16, v10, v8 // (op 16064) 
v_mul_f32 v17, v10, v9 // (op 16064) 

v_add_f32 v6, v14, v18 // (op 16065) 
v_add_f32 v7, v15, v19 // (op 16065) 
v_add_f32 v8, v16, v20 // (op 16065) 
v_add_f32 v9, v17, v21 // (op 16065) 




s_lshl_b32 s0, s4, 8 // (op 16045) 
s_add_u32 s0, s0, 896 // (op 16045) 
v_add_u32 v0, v13, s0 // (op 16045) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 16045) Store value




s_lshl_b32 s0, s2, 7 // (op 16070) 
s_mul_i32 s1, 3, s0 // (op 16070) 
s_add_u32 s1, s1, 896 // (op 16070) 

v_add_u32 v0, v12, s1 // (op 16070) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 16070) Load value

v_accvgpr_read v14, a236 // (op 16092) 
v_accvgpr_read v15, a237 // (op 16092) 
v_accvgpr_read v16, a238 // (op 16092) 
v_accvgpr_read v17, a239 // (op 16092) 
v_mul_f32 v18, v11, v14 // (op 16092) 
v_mul_f32 v19, v11, v15 // (op 16092) 
v_mul_f32 v20, v11, v16 // (op 16092) 
v_mul_f32 v21, v11, v17 // (op 16092) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 16068) 
v_mul_f32 v15, v10, v7 // (op 16068) 
v_mul_f32 v16, v10, v8 // (op 16068) 
v_mul_f32 v17, v10, v9 // (op 16068) 

v_add_f32 v6, v14, v18 // (op 16069) 
v_add_f32 v7, v15, v19 // (op 16069) 
v_add_f32 v8, v16, v20 // (op 16069) 
v_add_f32 v9, v17, v21 // (op 16069) 




s_lshl_b32 s0, s4, 7 // (op 16047) 
s_mul_i32 s1, 3, s0 // (op 16047) 
s_add_u32 s1, s1, 896 // (op 16047) 

v_add_u32 v0, v13, s1 // (op 16047) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 16047) Store value




s_lshl_b32 s0, s2, 9 // (op 16074) 
s_add_u32 s0, s0, 896 // (op 16074) 
v_add_u32 v0, v12, s0 // (op 16074) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 16074) Load value

v_accvgpr_read v14, a240 // (op 16094) 
v_accvgpr_read v15, a241 // (op 16094) 
v_accvgpr_read v16, a242 // (op 16094) 
v_accvgpr_read v17, a243 // (op 16094) 
v_mul_f32 v18, v11, v14 // (op 16094) 
v_mul_f32 v19, v11, v15 // (op 16094) 
v_mul_f32 v20, v11, v16 // (op 16094) 
v_mul_f32 v21, v11, v17 // (op 16094) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 16072) 
v_mul_f32 v15, v10, v7 // (op 16072) 
v_mul_f32 v16, v10, v8 // (op 16072) 
v_mul_f32 v17, v10, v9 // (op 16072) 

v_add_f32 v6, v14, v18 // (op 16073) 
v_add_f32 v7, v15, v19 // (op 16073) 
v_add_f32 v8, v16, v20 // (op 16073) 
v_add_f32 v9, v17, v21 // (op 16073) 




s_lshl_b32 s0, s4, 9 // (op 16049) 
s_add_u32 s0, s0, 896 // (op 16049) 
v_add_u32 v0, v13, s0 // (op 16049) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 16049) Store value




s_lshl_b32 s0, s2, 7 // (op 16078) 
s_mul_i32 s1, 5, s0 // (op 16078) 
s_add_u32 s1, s1, 896 // (op 16078) 

v_add_u32 v0, v12, s1 // (op 16078) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 16078) Load value

v_accvgpr_read v14, a244 // (op 16096) 
v_accvgpr_read v15, a245 // (op 16096) 
v_accvgpr_read v16, a246 // (op 16096) 
v_accvgpr_read v17, a247 // (op 16096) 
v_mul_f32 v18, v11, v14 // (op 16096) 
v_mul_f32 v19, v11, v15 // (op 16096) 
v_mul_f32 v20, v11, v16 // (op 16096) 
v_mul_f32 v21, v11, v17 // (op 16096) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 16076) 
v_mul_f32 v15, v10, v7 // (op 16076) 
v_mul_f32 v16, v10, v8 // (op 16076) 
v_mul_f32 v17, v10, v9 // (op 16076) 

v_add_f32 v6, v14, v18 // (op 16077) 
v_add_f32 v7, v15, v19 // (op 16077) 
v_add_f32 v8, v16, v20 // (op 16077) 
v_add_f32 v9, v17, v21 // (op 16077) 




s_lshl_b32 s0, s4, 7 // (op 16051) 
s_mul_i32 s1, 5, s0 // (op 16051) 
s_add_u32 s1, s1, 896 // (op 16051) 

v_add_u32 v0, v13, s1 // (op 16051) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 16051) Store value




s_lshl_b32 s0, s2, 7 // (op 16082) 
s_mul_i32 s1, 6, s0 // (op 16082) 
s_add_u32 s1, s1, 896 // (op 16082) 

v_add_u32 v0, v12, s1 // (op 16082) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 16082) Load value

v_accvgpr_read v14, a248 // (op 16098) 
v_accvgpr_read v15, a249 // (op 16098) 
v_accvgpr_read v16, a250 // (op 16098) 
v_accvgpr_read v17, a251 // (op 16098) 
v_mul_f32 v18, v11, v14 // (op 16098) 
v_mul_f32 v19, v11, v15 // (op 16098) 
v_mul_f32 v20, v11, v16 // (op 16098) 
v_mul_f32 v21, v11, v17 // (op 16098) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 16080) 
v_mul_f32 v15, v10, v7 // (op 16080) 
v_mul_f32 v16, v10, v8 // (op 16080) 
v_mul_f32 v17, v10, v9 // (op 16080) 

v_add_f32 v6, v14, v18 // (op 16081) 
v_add_f32 v7, v15, v19 // (op 16081) 
v_add_f32 v8, v16, v20 // (op 16081) 
v_add_f32 v9, v17, v21 // (op 16081) 




s_lshl_b32 s0, s4, 7 // (op 16053) 
s_mul_i32 s1, 6, s0 // (op 16053) 
s_add_u32 s1, s1, 896 // (op 16053) 

v_add_u32 v0, v13, s1 // (op 16053) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 16053) Store value




s_lshl_b32 s0, s2, 7 // (op 16086) 
s_mul_i32 s1, 7, s0 // (op 16086) 
s_add_u32 s1, s1, 896 // (op 16086) 

v_add_u32 v0, v12, s1 // (op 16086) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 16086) Load value

v_accvgpr_read v14, a252 // (op 16100) 
v_accvgpr_read v15, a253 // (op 16100) 
v_accvgpr_read v16, a254 // (op 16100) 
v_accvgpr_read v17, a255 // (op 16100) 
v_mul_f32 v18, v11, v14 // (op 16100) 
v_mul_f32 v19, v11, v15 // (op 16100) 
v_mul_f32 v20, v11, v16 // (op 16100) 
v_mul_f32 v21, v11, v17 // (op 16100) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 16084) 
v_mul_f32 v15, v10, v7 // (op 16084) 
v_mul_f32 v16, v10, v8 // (op 16084) 
v_mul_f32 v17, v10, v9 // (op 16084) 

v_add_f32 v6, v14, v18 // (op 16085) 
v_add_f32 v7, v15, v19 // (op 16085) 
v_add_f32 v8, v16, v20 // (op 16085) 
v_add_f32 v9, v17, v21 // (op 16085) 




s_lshl_b32 s0, s4, 7 // (op 16055) 
s_mul_i32 s1, 7, s0 // (op 16055) 
s_add_u32 s1, s1, 896 // (op 16055) 

v_add_u32 v0, v13, s1 // (op 16055) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 16055) Store value




v_add_u32 v0, 768, v12 // (op 15903) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15903) Load value

v_accvgpr_read v14, a192 // (op 15883) 
v_accvgpr_read v15, a193 // (op 15883) 
v_accvgpr_read v16, a194 // (op 15883) 
v_accvgpr_read v17, a195 // (op 15883) 
v_mul_f32 v18, v11, v14 // (op 15883) 
v_mul_f32 v19, v11, v15 // (op 15883) 
v_mul_f32 v20, v11, v16 // (op 15883) 
v_mul_f32 v21, v11, v17 // (op 15883) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15902) 
v_mul_f32 v15, v10, v7 // (op 15902) 
v_mul_f32 v16, v10, v8 // (op 15902) 
v_mul_f32 v17, v10, v9 // (op 15902) 

v_add_f32 v6, v14, v18 // (op 15884) 
v_add_f32 v7, v15, v19 // (op 15884) 
v_add_f32 v8, v16, v20 // (op 15884) 
v_add_f32 v9, v17, v21 // (op 15884) 




v_add_u32 v0, 768, v13 // (op 15886) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15886) Store value




s_lshl_b32 s0, s2, 7 // (op 15907) 
s_add_u32 s0, s0, 768 // (op 15907) 
v_add_u32 v0, v12, s0 // (op 15907) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15907) Load value

v_accvgpr_read v14, a196 // (op 15933) 
v_accvgpr_read v15, a197 // (op 15933) 
v_accvgpr_read v16, a198 // (op 15933) 
v_accvgpr_read v17, a199 // (op 15933) 
v_mul_f32 v18, v11, v14 // (op 15933) 
v_mul_f32 v19, v11, v15 // (op 15933) 
v_mul_f32 v20, v11, v16 // (op 15933) 
v_mul_f32 v21, v11, v17 // (op 15933) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15905) 
v_mul_f32 v15, v10, v7 // (op 15905) 
v_mul_f32 v16, v10, v8 // (op 15905) 
v_mul_f32 v17, v10, v9 // (op 15905) 

v_add_f32 v6, v14, v18 // (op 15906) 
v_add_f32 v7, v15, v19 // (op 15906) 
v_add_f32 v8, v16, v20 // (op 15906) 
v_add_f32 v9, v17, v21 // (op 15906) 




s_lshl_b32 s0, s4, 7 // (op 15888) 
s_add_u32 s0, s0, 768 // (op 15888) 
v_add_u32 v0, v13, s0 // (op 15888) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15888) Store value




s_lshl_b32 s0, s2, 8 // (op 15911) 
s_add_u32 s0, s0, 768 // (op 15911) 
v_add_u32 v0, v12, s0 // (op 15911) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15911) Load value

v_accvgpr_read v14, a200 // (op 15935) 
v_accvgpr_read v15, a201 // (op 15935) 
v_accvgpr_read v16, a202 // (op 15935) 
v_accvgpr_read v17, a203 // (op 15935) 
v_mul_f32 v18, v11, v14 // (op 15935) 
v_mul_f32 v19, v11, v15 // (op 15935) 
v_mul_f32 v20, v11, v16 // (op 15935) 
v_mul_f32 v21, v11, v17 // (op 15935) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15909) 
v_mul_f32 v15, v10, v7 // (op 15909) 
v_mul_f32 v16, v10, v8 // (op 15909) 
v_mul_f32 v17, v10, v9 // (op 15909) 

v_add_f32 v6, v14, v18 // (op 15910) 
v_add_f32 v7, v15, v19 // (op 15910) 
v_add_f32 v8, v16, v20 // (op 15910) 
v_add_f32 v9, v17, v21 // (op 15910) 




s_lshl_b32 s0, s4, 8 // (op 15890) 
s_add_u32 s0, s0, 768 // (op 15890) 
v_add_u32 v0, v13, s0 // (op 15890) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15890) Store value




s_lshl_b32 s0, s2, 7 // (op 15915) 
s_mul_i32 s1, 3, s0 // (op 15915) 
s_add_u32 s1, s1, 768 // (op 15915) 

v_add_u32 v0, v12, s1 // (op 15915) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15915) Load value

v_accvgpr_read v14, a204 // (op 15937) 
v_accvgpr_read v15, a205 // (op 15937) 
v_accvgpr_read v16, a206 // (op 15937) 
v_accvgpr_read v17, a207 // (op 15937) 
v_mul_f32 v18, v11, v14 // (op 15937) 
v_mul_f32 v19, v11, v15 // (op 15937) 
v_mul_f32 v20, v11, v16 // (op 15937) 
v_mul_f32 v21, v11, v17 // (op 15937) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15913) 
v_mul_f32 v15, v10, v7 // (op 15913) 
v_mul_f32 v16, v10, v8 // (op 15913) 
v_mul_f32 v17, v10, v9 // (op 15913) 

v_add_f32 v6, v14, v18 // (op 15914) 
v_add_f32 v7, v15, v19 // (op 15914) 
v_add_f32 v8, v16, v20 // (op 15914) 
v_add_f32 v9, v17, v21 // (op 15914) 




s_lshl_b32 s0, s4, 7 // (op 15892) 
s_mul_i32 s1, 3, s0 // (op 15892) 
s_add_u32 s1, s1, 768 // (op 15892) 

v_add_u32 v0, v13, s1 // (op 15892) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15892) Store value




s_lshl_b32 s0, s2, 9 // (op 15919) 
s_add_u32 s0, s0, 768 // (op 15919) 
v_add_u32 v0, v12, s0 // (op 15919) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15919) Load value

v_accvgpr_read v14, a208 // (op 15939) 
v_accvgpr_read v15, a209 // (op 15939) 
v_accvgpr_read v16, a210 // (op 15939) 
v_accvgpr_read v17, a211 // (op 15939) 
v_mul_f32 v18, v11, v14 // (op 15939) 
v_mul_f32 v19, v11, v15 // (op 15939) 
v_mul_f32 v20, v11, v16 // (op 15939) 
v_mul_f32 v21, v11, v17 // (op 15939) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15917) 
v_mul_f32 v15, v10, v7 // (op 15917) 
v_mul_f32 v16, v10, v8 // (op 15917) 
v_mul_f32 v17, v10, v9 // (op 15917) 

v_add_f32 v6, v14, v18 // (op 15918) 
v_add_f32 v7, v15, v19 // (op 15918) 
v_add_f32 v8, v16, v20 // (op 15918) 
v_add_f32 v9, v17, v21 // (op 15918) 




s_lshl_b32 s0, s4, 9 // (op 15894) 
s_add_u32 s0, s0, 768 // (op 15894) 
v_add_u32 v0, v13, s0 // (op 15894) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15894) Store value




s_lshl_b32 s0, s2, 7 // (op 15923) 
s_mul_i32 s1, 5, s0 // (op 15923) 
s_add_u32 s1, s1, 768 // (op 15923) 

v_add_u32 v0, v12, s1 // (op 15923) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15923) Load value

v_accvgpr_read v14, a212 // (op 15941) 
v_accvgpr_read v15, a213 // (op 15941) 
v_accvgpr_read v16, a214 // (op 15941) 
v_accvgpr_read v17, a215 // (op 15941) 
v_mul_f32 v18, v11, v14 // (op 15941) 
v_mul_f32 v19, v11, v15 // (op 15941) 
v_mul_f32 v20, v11, v16 // (op 15941) 
v_mul_f32 v21, v11, v17 // (op 15941) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15921) 
v_mul_f32 v15, v10, v7 // (op 15921) 
v_mul_f32 v16, v10, v8 // (op 15921) 
v_mul_f32 v17, v10, v9 // (op 15921) 

v_add_f32 v6, v14, v18 // (op 15922) 
v_add_f32 v7, v15, v19 // (op 15922) 
v_add_f32 v8, v16, v20 // (op 15922) 
v_add_f32 v9, v17, v21 // (op 15922) 




s_lshl_b32 s0, s4, 7 // (op 15896) 
s_mul_i32 s1, 5, s0 // (op 15896) 
s_add_u32 s1, s1, 768 // (op 15896) 

v_add_u32 v0, v13, s1 // (op 15896) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15896) Store value




s_lshl_b32 s0, s2, 7 // (op 15927) 
s_mul_i32 s1, 6, s0 // (op 15927) 
s_add_u32 s1, s1, 768 // (op 15927) 

v_add_u32 v0, v12, s1 // (op 15927) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15927) Load value

v_accvgpr_read v14, a216 // (op 15943) 
v_accvgpr_read v15, a217 // (op 15943) 
v_accvgpr_read v16, a218 // (op 15943) 
v_accvgpr_read v17, a219 // (op 15943) 
v_mul_f32 v18, v11, v14 // (op 15943) 
v_mul_f32 v19, v11, v15 // (op 15943) 
v_mul_f32 v20, v11, v16 // (op 15943) 
v_mul_f32 v21, v11, v17 // (op 15943) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15925) 
v_mul_f32 v15, v10, v7 // (op 15925) 
v_mul_f32 v16, v10, v8 // (op 15925) 
v_mul_f32 v17, v10, v9 // (op 15925) 

v_add_f32 v6, v14, v18 // (op 15926) 
v_add_f32 v7, v15, v19 // (op 15926) 
v_add_f32 v8, v16, v20 // (op 15926) 
v_add_f32 v9, v17, v21 // (op 15926) 




s_lshl_b32 s0, s4, 7 // (op 15898) 
s_mul_i32 s1, 6, s0 // (op 15898) 
s_add_u32 s1, s1, 768 // (op 15898) 

v_add_u32 v0, v13, s1 // (op 15898) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15898) Store value




s_lshl_b32 s0, s2, 7 // (op 15931) 
s_mul_i32 s1, 7, s0 // (op 15931) 
s_add_u32 s1, s1, 768 // (op 15931) 

v_add_u32 v0, v12, s1 // (op 15931) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15931) Load value

v_accvgpr_read v14, a220 // (op 15945) 
v_accvgpr_read v15, a221 // (op 15945) 
v_accvgpr_read v16, a222 // (op 15945) 
v_accvgpr_read v17, a223 // (op 15945) 
v_mul_f32 v18, v11, v14 // (op 15945) 
v_mul_f32 v19, v11, v15 // (op 15945) 
v_mul_f32 v20, v11, v16 // (op 15945) 
v_mul_f32 v21, v11, v17 // (op 15945) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15929) 
v_mul_f32 v15, v10, v7 // (op 15929) 
v_mul_f32 v16, v10, v8 // (op 15929) 
v_mul_f32 v17, v10, v9 // (op 15929) 

v_add_f32 v6, v14, v18 // (op 15930) 
v_add_f32 v7, v15, v19 // (op 15930) 
v_add_f32 v8, v16, v20 // (op 15930) 
v_add_f32 v9, v17, v21 // (op 15930) 




s_lshl_b32 s0, s4, 7 // (op 15900) 
s_mul_i32 s1, 7, s0 // (op 15900) 
s_add_u32 s1, s1, 768 // (op 15900) 

v_add_u32 v0, v13, s1 // (op 15900) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15900) Store value




v_add_u32 v0, 640, v12 // (op 15748) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15748) Load value

v_accvgpr_read v14, a160 // (op 15728) 
v_accvgpr_read v15, a161 // (op 15728) 
v_accvgpr_read v16, a162 // (op 15728) 
v_accvgpr_read v17, a163 // (op 15728) 
v_mul_f32 v18, v11, v14 // (op 15728) 
v_mul_f32 v19, v11, v15 // (op 15728) 
v_mul_f32 v20, v11, v16 // (op 15728) 
v_mul_f32 v21, v11, v17 // (op 15728) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15747) 
v_mul_f32 v15, v10, v7 // (op 15747) 
v_mul_f32 v16, v10, v8 // (op 15747) 
v_mul_f32 v17, v10, v9 // (op 15747) 

v_add_f32 v6, v14, v18 // (op 15729) 
v_add_f32 v7, v15, v19 // (op 15729) 
v_add_f32 v8, v16, v20 // (op 15729) 
v_add_f32 v9, v17, v21 // (op 15729) 




v_add_u32 v0, 640, v13 // (op 15731) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15731) Store value




s_lshl_b32 s0, s2, 7 // (op 15752) 
s_add_u32 s0, s0, 640 // (op 15752) 
v_add_u32 v0, v12, s0 // (op 15752) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15752) Load value

v_accvgpr_read v14, a164 // (op 15778) 
v_accvgpr_read v15, a165 // (op 15778) 
v_accvgpr_read v16, a166 // (op 15778) 
v_accvgpr_read v17, a167 // (op 15778) 
v_mul_f32 v18, v11, v14 // (op 15778) 
v_mul_f32 v19, v11, v15 // (op 15778) 
v_mul_f32 v20, v11, v16 // (op 15778) 
v_mul_f32 v21, v11, v17 // (op 15778) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15750) 
v_mul_f32 v15, v10, v7 // (op 15750) 
v_mul_f32 v16, v10, v8 // (op 15750) 
v_mul_f32 v17, v10, v9 // (op 15750) 

v_add_f32 v6, v14, v18 // (op 15751) 
v_add_f32 v7, v15, v19 // (op 15751) 
v_add_f32 v8, v16, v20 // (op 15751) 
v_add_f32 v9, v17, v21 // (op 15751) 




s_lshl_b32 s0, s4, 7 // (op 15733) 
s_add_u32 s0, s0, 640 // (op 15733) 
v_add_u32 v0, v13, s0 // (op 15733) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15733) Store value




s_lshl_b32 s0, s2, 8 // (op 15756) 
s_add_u32 s0, s0, 640 // (op 15756) 
v_add_u32 v0, v12, s0 // (op 15756) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15756) Load value

v_accvgpr_read v14, a168 // (op 15780) 
v_accvgpr_read v15, a169 // (op 15780) 
v_accvgpr_read v16, a170 // (op 15780) 
v_accvgpr_read v17, a171 // (op 15780) 
v_mul_f32 v18, v11, v14 // (op 15780) 
v_mul_f32 v19, v11, v15 // (op 15780) 
v_mul_f32 v20, v11, v16 // (op 15780) 
v_mul_f32 v21, v11, v17 // (op 15780) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15754) 
v_mul_f32 v15, v10, v7 // (op 15754) 
v_mul_f32 v16, v10, v8 // (op 15754) 
v_mul_f32 v17, v10, v9 // (op 15754) 

v_add_f32 v6, v14, v18 // (op 15755) 
v_add_f32 v7, v15, v19 // (op 15755) 
v_add_f32 v8, v16, v20 // (op 15755) 
v_add_f32 v9, v17, v21 // (op 15755) 




s_lshl_b32 s0, s4, 8 // (op 15735) 
s_add_u32 s0, s0, 640 // (op 15735) 
v_add_u32 v0, v13, s0 // (op 15735) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15735) Store value




s_lshl_b32 s0, s2, 7 // (op 15760) 
s_mul_i32 s1, 3, s0 // (op 15760) 
s_add_u32 s1, s1, 640 // (op 15760) 

v_add_u32 v0, v12, s1 // (op 15760) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15760) Load value

v_accvgpr_read v14, a172 // (op 15782) 
v_accvgpr_read v15, a173 // (op 15782) 
v_accvgpr_read v16, a174 // (op 15782) 
v_accvgpr_read v17, a175 // (op 15782) 
v_mul_f32 v18, v11, v14 // (op 15782) 
v_mul_f32 v19, v11, v15 // (op 15782) 
v_mul_f32 v20, v11, v16 // (op 15782) 
v_mul_f32 v21, v11, v17 // (op 15782) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15758) 
v_mul_f32 v15, v10, v7 // (op 15758) 
v_mul_f32 v16, v10, v8 // (op 15758) 
v_mul_f32 v17, v10, v9 // (op 15758) 

v_add_f32 v6, v14, v18 // (op 15759) 
v_add_f32 v7, v15, v19 // (op 15759) 
v_add_f32 v8, v16, v20 // (op 15759) 
v_add_f32 v9, v17, v21 // (op 15759) 




s_lshl_b32 s0, s4, 7 // (op 15737) 
s_mul_i32 s1, 3, s0 // (op 15737) 
s_add_u32 s1, s1, 640 // (op 15737) 

v_add_u32 v0, v13, s1 // (op 15737) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15737) Store value




s_lshl_b32 s0, s2, 9 // (op 15764) 
s_add_u32 s0, s0, 640 // (op 15764) 
v_add_u32 v0, v12, s0 // (op 15764) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15764) Load value

v_accvgpr_read v14, a176 // (op 15784) 
v_accvgpr_read v15, a177 // (op 15784) 
v_accvgpr_read v16, a178 // (op 15784) 
v_accvgpr_read v17, a179 // (op 15784) 
v_mul_f32 v18, v11, v14 // (op 15784) 
v_mul_f32 v19, v11, v15 // (op 15784) 
v_mul_f32 v20, v11, v16 // (op 15784) 
v_mul_f32 v21, v11, v17 // (op 15784) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15762) 
v_mul_f32 v15, v10, v7 // (op 15762) 
v_mul_f32 v16, v10, v8 // (op 15762) 
v_mul_f32 v17, v10, v9 // (op 15762) 

v_add_f32 v6, v14, v18 // (op 15763) 
v_add_f32 v7, v15, v19 // (op 15763) 
v_add_f32 v8, v16, v20 // (op 15763) 
v_add_f32 v9, v17, v21 // (op 15763) 




s_lshl_b32 s0, s4, 9 // (op 15739) 
s_add_u32 s0, s0, 640 // (op 15739) 
v_add_u32 v0, v13, s0 // (op 15739) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15739) Store value




s_lshl_b32 s0, s2, 7 // (op 15768) 
s_mul_i32 s1, 5, s0 // (op 15768) 
s_add_u32 s1, s1, 640 // (op 15768) 

v_add_u32 v0, v12, s1 // (op 15768) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15768) Load value

v_accvgpr_read v14, a180 // (op 15786) 
v_accvgpr_read v15, a181 // (op 15786) 
v_accvgpr_read v16, a182 // (op 15786) 
v_accvgpr_read v17, a183 // (op 15786) 
v_mul_f32 v18, v11, v14 // (op 15786) 
v_mul_f32 v19, v11, v15 // (op 15786) 
v_mul_f32 v20, v11, v16 // (op 15786) 
v_mul_f32 v21, v11, v17 // (op 15786) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15766) 
v_mul_f32 v15, v10, v7 // (op 15766) 
v_mul_f32 v16, v10, v8 // (op 15766) 
v_mul_f32 v17, v10, v9 // (op 15766) 

v_add_f32 v6, v14, v18 // (op 15767) 
v_add_f32 v7, v15, v19 // (op 15767) 
v_add_f32 v8, v16, v20 // (op 15767) 
v_add_f32 v9, v17, v21 // (op 15767) 




s_lshl_b32 s0, s4, 7 // (op 15741) 
s_mul_i32 s1, 5, s0 // (op 15741) 
s_add_u32 s1, s1, 640 // (op 15741) 

v_add_u32 v0, v13, s1 // (op 15741) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15741) Store value




s_lshl_b32 s0, s2, 7 // (op 15772) 
s_mul_i32 s1, 6, s0 // (op 15772) 
s_add_u32 s1, s1, 640 // (op 15772) 

v_add_u32 v0, v12, s1 // (op 15772) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15772) Load value

v_accvgpr_read v14, a184 // (op 15788) 
v_accvgpr_read v15, a185 // (op 15788) 
v_accvgpr_read v16, a186 // (op 15788) 
v_accvgpr_read v17, a187 // (op 15788) 
v_mul_f32 v18, v11, v14 // (op 15788) 
v_mul_f32 v19, v11, v15 // (op 15788) 
v_mul_f32 v20, v11, v16 // (op 15788) 
v_mul_f32 v21, v11, v17 // (op 15788) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15770) 
v_mul_f32 v15, v10, v7 // (op 15770) 
v_mul_f32 v16, v10, v8 // (op 15770) 
v_mul_f32 v17, v10, v9 // (op 15770) 

v_add_f32 v6, v14, v18 // (op 15771) 
v_add_f32 v7, v15, v19 // (op 15771) 
v_add_f32 v8, v16, v20 // (op 15771) 
v_add_f32 v9, v17, v21 // (op 15771) 




s_lshl_b32 s0, s4, 7 // (op 15743) 
s_mul_i32 s1, 6, s0 // (op 15743) 
s_add_u32 s1, s1, 640 // (op 15743) 

v_add_u32 v0, v13, s1 // (op 15743) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15743) Store value




s_lshl_b32 s0, s2, 7 // (op 15776) 
s_mul_i32 s1, 7, s0 // (op 15776) 
s_add_u32 s1, s1, 640 // (op 15776) 

v_add_u32 v0, v12, s1 // (op 15776) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15776) Load value

v_accvgpr_read v14, a188 // (op 15790) 
v_accvgpr_read v15, a189 // (op 15790) 
v_accvgpr_read v16, a190 // (op 15790) 
v_accvgpr_read v17, a191 // (op 15790) 
v_mul_f32 v18, v11, v14 // (op 15790) 
v_mul_f32 v19, v11, v15 // (op 15790) 
v_mul_f32 v20, v11, v16 // (op 15790) 
v_mul_f32 v21, v11, v17 // (op 15790) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15774) 
v_mul_f32 v15, v10, v7 // (op 15774) 
v_mul_f32 v16, v10, v8 // (op 15774) 
v_mul_f32 v17, v10, v9 // (op 15774) 

v_add_f32 v6, v14, v18 // (op 15775) 
v_add_f32 v7, v15, v19 // (op 15775) 
v_add_f32 v8, v16, v20 // (op 15775) 
v_add_f32 v9, v17, v21 // (op 15775) 




s_lshl_b32 s0, s4, 7 // (op 15745) 
s_mul_i32 s1, 7, s0 // (op 15745) 
s_add_u32 s1, s1, 640 // (op 15745) 

v_add_u32 v0, v13, s1 // (op 15745) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15745) Store value




v_add_u32 v0, 384, v12 // (op 15438) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15438) Load value

v_accvgpr_read v14, a96 // (op 15418) 
v_accvgpr_read v15, a97 // (op 15418) 
v_accvgpr_read v16, a98 // (op 15418) 
v_accvgpr_read v17, a99 // (op 15418) 
v_mul_f32 v18, v11, v14 // (op 15418) 
v_mul_f32 v19, v11, v15 // (op 15418) 
v_mul_f32 v20, v11, v16 // (op 15418) 
v_mul_f32 v21, v11, v17 // (op 15418) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15437) 
v_mul_f32 v15, v10, v7 // (op 15437) 
v_mul_f32 v16, v10, v8 // (op 15437) 
v_mul_f32 v17, v10, v9 // (op 15437) 

v_add_f32 v6, v14, v18 // (op 15419) 
v_add_f32 v7, v15, v19 // (op 15419) 
v_add_f32 v8, v16, v20 // (op 15419) 
v_add_f32 v9, v17, v21 // (op 15419) 




v_add_u32 v0, 384, v13 // (op 15421) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15421) Store value




s_lshl_b32 s0, s2, 7 // (op 15442) 
s_add_u32 s0, s0, 384 // (op 15442) 
v_add_u32 v0, v12, s0 // (op 15442) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15442) Load value

v_accvgpr_read v14, a100 // (op 15468) 
v_accvgpr_read v15, a101 // (op 15468) 
v_accvgpr_read v16, a102 // (op 15468) 
v_accvgpr_read v17, a103 // (op 15468) 
v_mul_f32 v18, v11, v14 // (op 15468) 
v_mul_f32 v19, v11, v15 // (op 15468) 
v_mul_f32 v20, v11, v16 // (op 15468) 
v_mul_f32 v21, v11, v17 // (op 15468) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15440) 
v_mul_f32 v15, v10, v7 // (op 15440) 
v_mul_f32 v16, v10, v8 // (op 15440) 
v_mul_f32 v17, v10, v9 // (op 15440) 

v_add_f32 v6, v14, v18 // (op 15441) 
v_add_f32 v7, v15, v19 // (op 15441) 
v_add_f32 v8, v16, v20 // (op 15441) 
v_add_f32 v9, v17, v21 // (op 15441) 




s_lshl_b32 s0, s4, 7 // (op 15423) 
s_add_u32 s0, s0, 384 // (op 15423) 
v_add_u32 v0, v13, s0 // (op 15423) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15423) Store value




s_lshl_b32 s0, s2, 8 // (op 15446) 
s_add_u32 s0, s0, 384 // (op 15446) 
v_add_u32 v0, v12, s0 // (op 15446) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15446) Load value

v_accvgpr_read v14, a104 // (op 15470) 
v_accvgpr_read v15, a105 // (op 15470) 
v_accvgpr_read v16, a106 // (op 15470) 
v_accvgpr_read v17, a107 // (op 15470) 
v_mul_f32 v18, v11, v14 // (op 15470) 
v_mul_f32 v19, v11, v15 // (op 15470) 
v_mul_f32 v20, v11, v16 // (op 15470) 
v_mul_f32 v21, v11, v17 // (op 15470) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15444) 
v_mul_f32 v15, v10, v7 // (op 15444) 
v_mul_f32 v16, v10, v8 // (op 15444) 
v_mul_f32 v17, v10, v9 // (op 15444) 

v_add_f32 v6, v14, v18 // (op 15445) 
v_add_f32 v7, v15, v19 // (op 15445) 
v_add_f32 v8, v16, v20 // (op 15445) 
v_add_f32 v9, v17, v21 // (op 15445) 




s_lshl_b32 s0, s4, 8 // (op 15425) 
s_add_u32 s0, s0, 384 // (op 15425) 
v_add_u32 v0, v13, s0 // (op 15425) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15425) Store value




s_lshl_b32 s0, s2, 7 // (op 15450) 
s_mul_i32 s1, 3, s0 // (op 15450) 
s_add_u32 s1, s1, 384 // (op 15450) 

v_add_u32 v0, v12, s1 // (op 15450) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15450) Load value

v_accvgpr_read v14, a108 // (op 15472) 
v_accvgpr_read v15, a109 // (op 15472) 
v_accvgpr_read v16, a110 // (op 15472) 
v_accvgpr_read v17, a111 // (op 15472) 
v_mul_f32 v18, v11, v14 // (op 15472) 
v_mul_f32 v19, v11, v15 // (op 15472) 
v_mul_f32 v20, v11, v16 // (op 15472) 
v_mul_f32 v21, v11, v17 // (op 15472) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15448) 
v_mul_f32 v15, v10, v7 // (op 15448) 
v_mul_f32 v16, v10, v8 // (op 15448) 
v_mul_f32 v17, v10, v9 // (op 15448) 

v_add_f32 v6, v14, v18 // (op 15449) 
v_add_f32 v7, v15, v19 // (op 15449) 
v_add_f32 v8, v16, v20 // (op 15449) 
v_add_f32 v9, v17, v21 // (op 15449) 




s_lshl_b32 s0, s4, 7 // (op 15427) 
s_mul_i32 s1, 3, s0 // (op 15427) 
s_add_u32 s1, s1, 384 // (op 15427) 

v_add_u32 v0, v13, s1 // (op 15427) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15427) Store value




s_lshl_b32 s0, s2, 9 // (op 15454) 
s_add_u32 s0, s0, 384 // (op 15454) 
v_add_u32 v0, v12, s0 // (op 15454) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15454) Load value

v_accvgpr_read v14, a112 // (op 15474) 
v_accvgpr_read v15, a113 // (op 15474) 
v_accvgpr_read v16, a114 // (op 15474) 
v_accvgpr_read v17, a115 // (op 15474) 
v_mul_f32 v18, v11, v14 // (op 15474) 
v_mul_f32 v19, v11, v15 // (op 15474) 
v_mul_f32 v20, v11, v16 // (op 15474) 
v_mul_f32 v21, v11, v17 // (op 15474) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15452) 
v_mul_f32 v15, v10, v7 // (op 15452) 
v_mul_f32 v16, v10, v8 // (op 15452) 
v_mul_f32 v17, v10, v9 // (op 15452) 

v_add_f32 v6, v14, v18 // (op 15453) 
v_add_f32 v7, v15, v19 // (op 15453) 
v_add_f32 v8, v16, v20 // (op 15453) 
v_add_f32 v9, v17, v21 // (op 15453) 




s_lshl_b32 s0, s4, 9 // (op 15429) 
s_add_u32 s0, s0, 384 // (op 15429) 
v_add_u32 v0, v13, s0 // (op 15429) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15429) Store value




s_lshl_b32 s0, s2, 7 // (op 15458) 
s_mul_i32 s1, 5, s0 // (op 15458) 
s_add_u32 s1, s1, 384 // (op 15458) 

v_add_u32 v0, v12, s1 // (op 15458) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15458) Load value

v_accvgpr_read v14, a116 // (op 15476) 
v_accvgpr_read v15, a117 // (op 15476) 
v_accvgpr_read v16, a118 // (op 15476) 
v_accvgpr_read v17, a119 // (op 15476) 
v_mul_f32 v18, v11, v14 // (op 15476) 
v_mul_f32 v19, v11, v15 // (op 15476) 
v_mul_f32 v20, v11, v16 // (op 15476) 
v_mul_f32 v21, v11, v17 // (op 15476) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15456) 
v_mul_f32 v15, v10, v7 // (op 15456) 
v_mul_f32 v16, v10, v8 // (op 15456) 
v_mul_f32 v17, v10, v9 // (op 15456) 

v_add_f32 v6, v14, v18 // (op 15457) 
v_add_f32 v7, v15, v19 // (op 15457) 
v_add_f32 v8, v16, v20 // (op 15457) 
v_add_f32 v9, v17, v21 // (op 15457) 




s_lshl_b32 s0, s4, 7 // (op 15431) 
s_mul_i32 s1, 5, s0 // (op 15431) 
s_add_u32 s1, s1, 384 // (op 15431) 

v_add_u32 v0, v13, s1 // (op 15431) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15431) Store value




s_lshl_b32 s0, s2, 7 // (op 15462) 
s_mul_i32 s1, 6, s0 // (op 15462) 
s_add_u32 s1, s1, 384 // (op 15462) 

v_add_u32 v0, v12, s1 // (op 15462) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15462) Load value

v_accvgpr_read v14, a120 // (op 15478) 
v_accvgpr_read v15, a121 // (op 15478) 
v_accvgpr_read v16, a122 // (op 15478) 
v_accvgpr_read v17, a123 // (op 15478) 
v_mul_f32 v18, v11, v14 // (op 15478) 
v_mul_f32 v19, v11, v15 // (op 15478) 
v_mul_f32 v20, v11, v16 // (op 15478) 
v_mul_f32 v21, v11, v17 // (op 15478) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15460) 
v_mul_f32 v15, v10, v7 // (op 15460) 
v_mul_f32 v16, v10, v8 // (op 15460) 
v_mul_f32 v17, v10, v9 // (op 15460) 

v_add_f32 v6, v14, v18 // (op 15461) 
v_add_f32 v7, v15, v19 // (op 15461) 
v_add_f32 v8, v16, v20 // (op 15461) 
v_add_f32 v9, v17, v21 // (op 15461) 




s_lshl_b32 s0, s4, 7 // (op 15433) 
s_mul_i32 s1, 6, s0 // (op 15433) 
s_add_u32 s1, s1, 384 // (op 15433) 

v_add_u32 v0, v13, s1 // (op 15433) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15433) Store value




s_lshl_b32 s0, s2, 7 // (op 15466) 
s_mul_i32 s1, 7, s0 // (op 15466) 
s_add_u32 s1, s1, 384 // (op 15466) 

v_add_u32 v0, v12, s1 // (op 15466) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15466) Load value

v_accvgpr_read v14, a124 // (op 15480) 
v_accvgpr_read v15, a125 // (op 15480) 
v_accvgpr_read v16, a126 // (op 15480) 
v_accvgpr_read v17, a127 // (op 15480) 
v_mul_f32 v18, v11, v14 // (op 15480) 
v_mul_f32 v19, v11, v15 // (op 15480) 
v_mul_f32 v20, v11, v16 // (op 15480) 
v_mul_f32 v21, v11, v17 // (op 15480) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15464) 
v_mul_f32 v15, v10, v7 // (op 15464) 
v_mul_f32 v16, v10, v8 // (op 15464) 
v_mul_f32 v17, v10, v9 // (op 15464) 

v_add_f32 v6, v14, v18 // (op 15465) 
v_add_f32 v7, v15, v19 // (op 15465) 
v_add_f32 v8, v16, v20 // (op 15465) 
v_add_f32 v9, v17, v21 // (op 15465) 




s_lshl_b32 s0, s4, 7 // (op 15435) 
s_mul_i32 s1, 7, s0 // (op 15435) 
s_add_u32 s1, s1, 384 // (op 15435) 

v_add_u32 v0, v13, s1 // (op 15435) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15435) Store value




v_add_u32 v0, 256, v12 // (op 15283) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15283) Load value

v_accvgpr_read v14, a64 // (op 15263) 
v_accvgpr_read v15, a65 // (op 15263) 
v_accvgpr_read v16, a66 // (op 15263) 
v_accvgpr_read v17, a67 // (op 15263) 
v_mul_f32 v18, v11, v14 // (op 15263) 
v_mul_f32 v19, v11, v15 // (op 15263) 
v_mul_f32 v20, v11, v16 // (op 15263) 
v_mul_f32 v21, v11, v17 // (op 15263) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15282) 
v_mul_f32 v15, v10, v7 // (op 15282) 
v_mul_f32 v16, v10, v8 // (op 15282) 
v_mul_f32 v17, v10, v9 // (op 15282) 

v_add_f32 v6, v14, v18 // (op 15264) 
v_add_f32 v7, v15, v19 // (op 15264) 
v_add_f32 v8, v16, v20 // (op 15264) 
v_add_f32 v9, v17, v21 // (op 15264) 




v_add_u32 v0, 256, v13 // (op 15266) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15266) Store value




s_lshl_b32 s0, s2, 7 // (op 15287) 
s_add_u32 s0, s0, 256 // (op 15287) 
v_add_u32 v0, v12, s0 // (op 15287) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15287) Load value

v_accvgpr_read v14, a68 // (op 15313) 
v_accvgpr_read v15, a69 // (op 15313) 
v_accvgpr_read v16, a70 // (op 15313) 
v_accvgpr_read v17, a71 // (op 15313) 
v_mul_f32 v18, v11, v14 // (op 15313) 
v_mul_f32 v19, v11, v15 // (op 15313) 
v_mul_f32 v20, v11, v16 // (op 15313) 
v_mul_f32 v21, v11, v17 // (op 15313) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15285) 
v_mul_f32 v15, v10, v7 // (op 15285) 
v_mul_f32 v16, v10, v8 // (op 15285) 
v_mul_f32 v17, v10, v9 // (op 15285) 

v_add_f32 v6, v14, v18 // (op 15286) 
v_add_f32 v7, v15, v19 // (op 15286) 
v_add_f32 v8, v16, v20 // (op 15286) 
v_add_f32 v9, v17, v21 // (op 15286) 




s_lshl_b32 s0, s4, 7 // (op 15268) 
s_add_u32 s0, s0, 256 // (op 15268) 
v_add_u32 v0, v13, s0 // (op 15268) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15268) Store value




s_lshl_b32 s0, s2, 8 // (op 15291) 
s_add_u32 s0, s0, 256 // (op 15291) 
v_add_u32 v0, v12, s0 // (op 15291) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15291) Load value

v_accvgpr_read v14, a72 // (op 15315) 
v_accvgpr_read v15, a73 // (op 15315) 
v_accvgpr_read v16, a74 // (op 15315) 
v_accvgpr_read v17, a75 // (op 15315) 
v_mul_f32 v18, v11, v14 // (op 15315) 
v_mul_f32 v19, v11, v15 // (op 15315) 
v_mul_f32 v20, v11, v16 // (op 15315) 
v_mul_f32 v21, v11, v17 // (op 15315) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 15289) 
v_mul_f32 v15, v10, v7 // (op 15289) 
v_mul_f32 v16, v10, v8 // (op 15289) 
v_mul_f32 v17, v10, v9 // (op 15289) 

v_add_f32 v6, v14, v18 // (op 15290) 
v_add_f32 v7, v15, v19 // (op 15290) 
v_add_f32 v8, v16, v20 // (op 15290) 
v_add_f32 v9, v17, v21 // (op 15290) 




s_lshl_b32 s0, s4, 8 // (op 15270) 
s_add_u32 s0, s0, 256 // (op 15270) 
v_add_u32 v0, v13, s0 // (op 15270) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15270) Store value




s_lshl_b32 s0, s2, 7 // (op 15295) 
s_mul_i32 s1, 3, s0 // (op 15295) 
s_add_u32 s1, s1, 256 // (op 15295) 

v_add_u32 v0, v12, s1 // (op 15295) 


v_accvgpr_read v6, a76 // (op 15317) 
v_accvgpr_read v7, a77 // (op 15317) 
v_accvgpr_read v8, a78 // (op 15317) 
v_accvgpr_read v9, a79 // (op 15317) 
v_mul_f32 v14, v11, v6 // (op 15317) 
v_mul_f32 v15, v11, v7 // (op 15317) 
v_mul_f32 v16, v11, v8 // (op 15317) 
v_mul_f32 v17, v11, v9 // (op 15317) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15295) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15293) 
v_mul_f32 v19, v10, v7 // (op 15293) 
v_mul_f32 v20, v10, v8 // (op 15293) 
v_mul_f32 v21, v10, v9 // (op 15293) 

v_add_f32 v6, v18, v14 // (op 15294) 
v_add_f32 v7, v19, v15 // (op 15294) 
v_add_f32 v8, v20, v16 // (op 15294) 
v_add_f32 v9, v21, v17 // (op 15294) 




s_lshl_b32 s0, s4, 7 // (op 15272) 
s_mul_i32 s1, 3, s0 // (op 15272) 
s_add_u32 s1, s1, 256 // (op 15272) 

v_add_u32 v0, v13, s1 // (op 15272) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15272) Store value




s_lshl_b32 s0, s2, 9 // (op 15299) 
s_add_u32 s0, s0, 256 // (op 15299) 
v_add_u32 v0, v12, s0 // (op 15299) 


v_accvgpr_read v6, a80 // (op 15319) 
v_accvgpr_read v7, a81 // (op 15319) 
v_accvgpr_read v8, a82 // (op 15319) 
v_accvgpr_read v9, a83 // (op 15319) 
v_mul_f32 v14, v11, v6 // (op 15319) 
v_mul_f32 v15, v11, v7 // (op 15319) 
v_mul_f32 v16, v11, v8 // (op 15319) 
v_mul_f32 v17, v11, v9 // (op 15319) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15299) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15297) 
v_mul_f32 v19, v10, v7 // (op 15297) 
v_mul_f32 v20, v10, v8 // (op 15297) 
v_mul_f32 v21, v10, v9 // (op 15297) 

v_add_f32 v6, v18, v14 // (op 15298) 
v_add_f32 v7, v19, v15 // (op 15298) 
v_add_f32 v8, v20, v16 // (op 15298) 
v_add_f32 v9, v21, v17 // (op 15298) 




s_lshl_b32 s0, s4, 9 // (op 15274) 
s_add_u32 s0, s0, 256 // (op 15274) 
v_add_u32 v0, v13, s0 // (op 15274) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15274) Store value




s_lshl_b32 s0, s2, 7 // (op 15303) 
s_mul_i32 s1, 5, s0 // (op 15303) 
s_add_u32 s1, s1, 256 // (op 15303) 

v_add_u32 v0, v12, s1 // (op 15303) 


v_accvgpr_read v6, a84 // (op 15321) 
v_accvgpr_read v7, a85 // (op 15321) 
v_accvgpr_read v8, a86 // (op 15321) 
v_accvgpr_read v9, a87 // (op 15321) 
v_mul_f32 v14, v11, v6 // (op 15321) 
v_mul_f32 v15, v11, v7 // (op 15321) 
v_mul_f32 v16, v11, v8 // (op 15321) 
v_mul_f32 v17, v11, v9 // (op 15321) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15303) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15301) 
v_mul_f32 v19, v10, v7 // (op 15301) 
v_mul_f32 v20, v10, v8 // (op 15301) 
v_mul_f32 v21, v10, v9 // (op 15301) 

v_add_f32 v6, v18, v14 // (op 15302) 
v_add_f32 v7, v19, v15 // (op 15302) 
v_add_f32 v8, v20, v16 // (op 15302) 
v_add_f32 v9, v21, v17 // (op 15302) 




s_lshl_b32 s0, s4, 7 // (op 15276) 
s_mul_i32 s1, 5, s0 // (op 15276) 
s_add_u32 s1, s1, 256 // (op 15276) 

v_add_u32 v0, v13, s1 // (op 15276) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15276) Store value




s_lshl_b32 s0, s2, 7 // (op 15307) 
s_mul_i32 s1, 6, s0 // (op 15307) 
s_add_u32 s1, s1, 256 // (op 15307) 

v_add_u32 v0, v12, s1 // (op 15307) 


v_accvgpr_read v6, a88 // (op 15323) 
v_accvgpr_read v7, a89 // (op 15323) 
v_accvgpr_read v8, a90 // (op 15323) 
v_accvgpr_read v9, a91 // (op 15323) 
v_mul_f32 v14, v11, v6 // (op 15323) 
v_mul_f32 v15, v11, v7 // (op 15323) 
v_mul_f32 v16, v11, v8 // (op 15323) 
v_mul_f32 v17, v11, v9 // (op 15323) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15307) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15305) 
v_mul_f32 v19, v10, v7 // (op 15305) 
v_mul_f32 v20, v10, v8 // (op 15305) 
v_mul_f32 v21, v10, v9 // (op 15305) 

v_add_f32 v6, v18, v14 // (op 15306) 
v_add_f32 v7, v19, v15 // (op 15306) 
v_add_f32 v8, v20, v16 // (op 15306) 
v_add_f32 v9, v21, v17 // (op 15306) 




s_lshl_b32 s0, s4, 7 // (op 15278) 
s_mul_i32 s1, 6, s0 // (op 15278) 
s_add_u32 s1, s1, 256 // (op 15278) 

v_add_u32 v0, v13, s1 // (op 15278) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15278) Store value




s_lshl_b32 s0, s2, 7 // (op 15311) 
s_mul_i32 s1, 7, s0 // (op 15311) 
s_add_u32 s1, s1, 256 // (op 15311) 

v_add_u32 v0, v12, s1 // (op 15311) 


v_accvgpr_read v6, a92 // (op 15325) 
v_accvgpr_read v7, a93 // (op 15325) 
v_accvgpr_read v8, a94 // (op 15325) 
v_accvgpr_read v9, a95 // (op 15325) 
v_mul_f32 v14, v11, v6 // (op 15325) 
v_mul_f32 v15, v11, v7 // (op 15325) 
v_mul_f32 v16, v11, v8 // (op 15325) 
v_mul_f32 v17, v11, v9 // (op 15325) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15311) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15309) 
v_mul_f32 v19, v10, v7 // (op 15309) 
v_mul_f32 v20, v10, v8 // (op 15309) 
v_mul_f32 v21, v10, v9 // (op 15309) 

v_add_f32 v6, v18, v14 // (op 15310) 
v_add_f32 v7, v19, v15 // (op 15310) 
v_add_f32 v8, v20, v16 // (op 15310) 
v_add_f32 v9, v21, v17 // (op 15310) 




s_lshl_b32 s0, s4, 7 // (op 15280) 
s_mul_i32 s1, 7, s0 // (op 15280) 
s_add_u32 s1, s1, 256 // (op 15280) 

v_add_u32 v0, v13, s1 // (op 15280) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15280) Store value




v_add_u32 v0, 128, v12 // (op 15128) 

s_nop 0
v_accvgpr_read v6, a32 // (op 15108) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v7, a33 // (op 15108) 
v_accvgpr_read v8, a34 // (op 15108) 
v_accvgpr_read v9, a35 // (op 15108) 
v_mul_f32 v14, v11, v6 // (op 15108) 
v_mul_f32 v15, v11, v7 // (op 15108) 
v_mul_f32 v16, v11, v8 // (op 15108) 
v_mul_f32 v17, v11, v9 // (op 15108) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15128) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15127) 
v_mul_f32 v19, v10, v7 // (op 15127) 
v_mul_f32 v20, v10, v8 // (op 15127) 
v_mul_f32 v21, v10, v9 // (op 15127) 

v_add_f32 v6, v18, v14 // (op 15109) 
v_add_f32 v7, v19, v15 // (op 15109) 
v_add_f32 v8, v20, v16 // (op 15109) 
v_add_f32 v9, v21, v17 // (op 15109) 




v_add_u32 v0, 128, v13 // (op 15111) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15111) Store value




s_lshl_b32 s0, s2, 7 // (op 15132) 
s_add_u32 s0, s0, 128 // (op 15132) 
v_add_u32 v0, v12, s0 // (op 15132) 


v_accvgpr_read v6, a36 // (op 15158) 
v_accvgpr_read v7, a37 // (op 15158) 
v_accvgpr_read v8, a38 // (op 15158) 
v_accvgpr_read v9, a39 // (op 15158) 
v_mul_f32 v14, v11, v6 // (op 15158) 
v_mul_f32 v15, v11, v7 // (op 15158) 
v_mul_f32 v16, v11, v8 // (op 15158) 
v_mul_f32 v17, v11, v9 // (op 15158) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15132) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15130) 
v_mul_f32 v19, v10, v7 // (op 15130) 
v_mul_f32 v20, v10, v8 // (op 15130) 
v_mul_f32 v21, v10, v9 // (op 15130) 

v_add_f32 v6, v18, v14 // (op 15131) 
v_add_f32 v7, v19, v15 // (op 15131) 
v_add_f32 v8, v20, v16 // (op 15131) 
v_add_f32 v9, v21, v17 // (op 15131) 




s_lshl_b32 s0, s4, 7 // (op 15113) 
s_add_u32 s0, s0, 128 // (op 15113) 
v_add_u32 v0, v13, s0 // (op 15113) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15113) Store value




s_lshl_b32 s0, s2, 8 // (op 15136) 
s_add_u32 s0, s0, 128 // (op 15136) 
v_add_u32 v0, v12, s0 // (op 15136) 


v_accvgpr_read v6, a40 // (op 15160) 
v_accvgpr_read v7, a41 // (op 15160) 
v_accvgpr_read v8, a42 // (op 15160) 
v_accvgpr_read v9, a43 // (op 15160) 
v_mul_f32 v14, v11, v6 // (op 15160) 
v_mul_f32 v15, v11, v7 // (op 15160) 
v_mul_f32 v16, v11, v8 // (op 15160) 
v_mul_f32 v17, v11, v9 // (op 15160) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15136) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15134) 
v_mul_f32 v19, v10, v7 // (op 15134) 
v_mul_f32 v20, v10, v8 // (op 15134) 
v_mul_f32 v21, v10, v9 // (op 15134) 

v_add_f32 v6, v18, v14 // (op 15135) 
v_add_f32 v7, v19, v15 // (op 15135) 
v_add_f32 v8, v20, v16 // (op 15135) 
v_add_f32 v9, v21, v17 // (op 15135) 




s_lshl_b32 s0, s4, 8 // (op 15115) 
s_add_u32 s0, s0, 128 // (op 15115) 
v_add_u32 v0, v13, s0 // (op 15115) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15115) Store value




s_lshl_b32 s0, s2, 7 // (op 15140) 
s_mul_i32 s1, 3, s0 // (op 15140) 
s_add_u32 s1, s1, 128 // (op 15140) 

v_add_u32 v0, v12, s1 // (op 15140) 


v_accvgpr_read v6, a44 // (op 15162) 
v_accvgpr_read v7, a45 // (op 15162) 
v_accvgpr_read v8, a46 // (op 15162) 
v_accvgpr_read v9, a47 // (op 15162) 
v_mul_f32 v14, v11, v6 // (op 15162) 
v_mul_f32 v15, v11, v7 // (op 15162) 
v_mul_f32 v16, v11, v8 // (op 15162) 
v_mul_f32 v17, v11, v9 // (op 15162) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15140) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15138) 
v_mul_f32 v19, v10, v7 // (op 15138) 
v_mul_f32 v20, v10, v8 // (op 15138) 
v_mul_f32 v21, v10, v9 // (op 15138) 

v_add_f32 v6, v18, v14 // (op 15139) 
v_add_f32 v7, v19, v15 // (op 15139) 
v_add_f32 v8, v20, v16 // (op 15139) 
v_add_f32 v9, v21, v17 // (op 15139) 




s_lshl_b32 s0, s4, 7 // (op 15117) 
s_mul_i32 s1, 3, s0 // (op 15117) 
s_add_u32 s1, s1, 128 // (op 15117) 

v_add_u32 v0, v13, s1 // (op 15117) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15117) Store value




s_lshl_b32 s0, s2, 9 // (op 15144) 
s_add_u32 s0, s0, 128 // (op 15144) 
v_add_u32 v0, v12, s0 // (op 15144) 


v_accvgpr_read v6, a48 // (op 15164) 
v_accvgpr_read v7, a49 // (op 15164) 
v_accvgpr_read v8, a50 // (op 15164) 
v_accvgpr_read v9, a51 // (op 15164) 
v_mul_f32 v14, v11, v6 // (op 15164) 
v_mul_f32 v15, v11, v7 // (op 15164) 
v_mul_f32 v16, v11, v8 // (op 15164) 
v_mul_f32 v17, v11, v9 // (op 15164) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15144) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15142) 
v_mul_f32 v19, v10, v7 // (op 15142) 
v_mul_f32 v20, v10, v8 // (op 15142) 
v_mul_f32 v21, v10, v9 // (op 15142) 

v_add_f32 v6, v18, v14 // (op 15143) 
v_add_f32 v7, v19, v15 // (op 15143) 
v_add_f32 v8, v20, v16 // (op 15143) 
v_add_f32 v9, v21, v17 // (op 15143) 




s_lshl_b32 s0, s4, 9 // (op 15119) 
s_add_u32 s0, s0, 128 // (op 15119) 
v_add_u32 v0, v13, s0 // (op 15119) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15119) Store value




s_lshl_b32 s0, s2, 7 // (op 15148) 
s_mul_i32 s1, 5, s0 // (op 15148) 
s_add_u32 s1, s1, 128 // (op 15148) 

v_add_u32 v0, v12, s1 // (op 15148) 


v_accvgpr_read v6, a52 // (op 15166) 
v_accvgpr_read v7, a53 // (op 15166) 
v_accvgpr_read v8, a54 // (op 15166) 
v_accvgpr_read v9, a55 // (op 15166) 
v_mul_f32 v14, v11, v6 // (op 15166) 
v_mul_f32 v15, v11, v7 // (op 15166) 
v_mul_f32 v16, v11, v8 // (op 15166) 
v_mul_f32 v17, v11, v9 // (op 15166) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15148) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15146) 
v_mul_f32 v19, v10, v7 // (op 15146) 
v_mul_f32 v20, v10, v8 // (op 15146) 
v_mul_f32 v21, v10, v9 // (op 15146) 

v_add_f32 v6, v18, v14 // (op 15147) 
v_add_f32 v7, v19, v15 // (op 15147) 
v_add_f32 v8, v20, v16 // (op 15147) 
v_add_f32 v9, v21, v17 // (op 15147) 




s_lshl_b32 s0, s4, 7 // (op 15121) 
s_mul_i32 s1, 5, s0 // (op 15121) 
s_add_u32 s1, s1, 128 // (op 15121) 

v_add_u32 v0, v13, s1 // (op 15121) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15121) Store value




s_lshl_b32 s0, s2, 7 // (op 15152) 
s_mul_i32 s1, 6, s0 // (op 15152) 
s_add_u32 s1, s1, 128 // (op 15152) 

v_add_u32 v0, v12, s1 // (op 15152) 


v_accvgpr_read v6, a56 // (op 15168) 
v_accvgpr_read v7, a57 // (op 15168) 
v_accvgpr_read v8, a58 // (op 15168) 
v_accvgpr_read v9, a59 // (op 15168) 
v_mul_f32 v14, v11, v6 // (op 15168) 
v_mul_f32 v15, v11, v7 // (op 15168) 
v_mul_f32 v16, v11, v8 // (op 15168) 
v_mul_f32 v17, v11, v9 // (op 15168) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15152) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15150) 
v_mul_f32 v19, v10, v7 // (op 15150) 
v_mul_f32 v20, v10, v8 // (op 15150) 
v_mul_f32 v21, v10, v9 // (op 15150) 

v_add_f32 v6, v18, v14 // (op 15151) 
v_add_f32 v7, v19, v15 // (op 15151) 
v_add_f32 v8, v20, v16 // (op 15151) 
v_add_f32 v9, v21, v17 // (op 15151) 




s_lshl_b32 s0, s4, 7 // (op 15123) 
s_mul_i32 s1, 6, s0 // (op 15123) 
s_add_u32 s1, s1, 128 // (op 15123) 

v_add_u32 v0, v13, s1 // (op 15123) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15123) Store value




s_lshl_b32 s0, s2, 7 // (op 15156) 
s_mul_i32 s1, 7, s0 // (op 15156) 
s_add_u32 s1, s1, 128 // (op 15156) 

v_add_u32 v0, v12, s1 // (op 15156) 


v_accvgpr_read v6, a60 // (op 15170) 
v_accvgpr_read v7, a61 // (op 15170) 
v_accvgpr_read v8, a62 // (op 15170) 
v_accvgpr_read v9, a63 // (op 15170) 
v_mul_f32 v14, v11, v6 // (op 15170) 
v_mul_f32 v15, v11, v7 // (op 15170) 
v_mul_f32 v16, v11, v8 // (op 15170) 
v_mul_f32 v17, v11, v9 // (op 15170) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15156) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15154) 
v_mul_f32 v19, v10, v7 // (op 15154) 
v_mul_f32 v20, v10, v8 // (op 15154) 
v_mul_f32 v21, v10, v9 // (op 15154) 

v_add_f32 v6, v18, v14 // (op 15155) 
v_add_f32 v7, v19, v15 // (op 15155) 
v_add_f32 v8, v20, v16 // (op 15155) 
v_add_f32 v9, v21, v17 // (op 15155) 




s_lshl_b32 s0, s4, 7 // (op 15125) 
s_mul_i32 s1, 7, s0 // (op 15125) 
s_add_u32 s1, s1, 128 // (op 15125) 

v_add_u32 v0, v13, s1 // (op 15125) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15125) Store value




v_add_u32 v0, 512, v12 // (op 15593) 

s_nop 0
v_accvgpr_read v6, a128 // (op 15573) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v7, a129 // (op 15573) 
v_accvgpr_read v8, a130 // (op 15573) 
v_accvgpr_read v9, a131 // (op 15573) 
v_mul_f32 v14, v11, v6 // (op 15573) 
v_mul_f32 v15, v11, v7 // (op 15573) 
v_mul_f32 v16, v11, v8 // (op 15573) 
v_mul_f32 v17, v11, v9 // (op 15573) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15593) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15592) 
v_mul_f32 v19, v10, v7 // (op 15592) 
v_mul_f32 v20, v10, v8 // (op 15592) 
v_mul_f32 v21, v10, v9 // (op 15592) 

v_add_f32 v6, v18, v14 // (op 15574) 
v_add_f32 v7, v19, v15 // (op 15574) 
v_add_f32 v8, v20, v16 // (op 15574) 
v_add_f32 v9, v21, v17 // (op 15574) 




v_add_u32 v0, 512, v13 // (op 15576) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15576) Store value




s_lshl_b32 s0, s2, 7 // (op 15597) 
s_add_u32 s0, s0, 512 // (op 15597) 
v_add_u32 v0, v12, s0 // (op 15597) 


v_accvgpr_read v6, a132 // (op 15623) 
v_accvgpr_read v7, a133 // (op 15623) 
v_accvgpr_read v8, a134 // (op 15623) 
v_accvgpr_read v9, a135 // (op 15623) 
v_mul_f32 v14, v11, v6 // (op 15623) 
v_mul_f32 v15, v11, v7 // (op 15623) 
v_mul_f32 v16, v11, v8 // (op 15623) 
v_mul_f32 v17, v11, v9 // (op 15623) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15597) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15595) 
v_mul_f32 v19, v10, v7 // (op 15595) 
v_mul_f32 v20, v10, v8 // (op 15595) 
v_mul_f32 v21, v10, v9 // (op 15595) 

v_add_f32 v6, v18, v14 // (op 15596) 
v_add_f32 v7, v19, v15 // (op 15596) 
v_add_f32 v8, v20, v16 // (op 15596) 
v_add_f32 v9, v21, v17 // (op 15596) 




s_lshl_b32 s0, s4, 7 // (op 15578) 
s_add_u32 s0, s0, 512 // (op 15578) 
v_add_u32 v0, v13, s0 // (op 15578) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15578) Store value




s_lshl_b32 s0, s2, 8 // (op 15601) 
s_add_u32 s0, s0, 512 // (op 15601) 
v_add_u32 v0, v12, s0 // (op 15601) 


v_accvgpr_read v6, a136 // (op 15625) 
v_accvgpr_read v7, a137 // (op 15625) 
v_accvgpr_read v8, a138 // (op 15625) 
v_accvgpr_read v9, a139 // (op 15625) 
v_mul_f32 v14, v11, v6 // (op 15625) 
v_mul_f32 v15, v11, v7 // (op 15625) 
v_mul_f32 v16, v11, v8 // (op 15625) 
v_mul_f32 v17, v11, v9 // (op 15625) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15601) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15599) 
v_mul_f32 v19, v10, v7 // (op 15599) 
v_mul_f32 v20, v10, v8 // (op 15599) 
v_mul_f32 v21, v10, v9 // (op 15599) 

v_add_f32 v6, v18, v14 // (op 15600) 
v_add_f32 v7, v19, v15 // (op 15600) 
v_add_f32 v8, v20, v16 // (op 15600) 
v_add_f32 v9, v21, v17 // (op 15600) 




s_lshl_b32 s0, s4, 8 // (op 15580) 
s_add_u32 s0, s0, 512 // (op 15580) 
v_add_u32 v0, v13, s0 // (op 15580) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15580) Store value




s_lshl_b32 s0, s2, 7 // (op 15605) 
s_mul_i32 s1, 3, s0 // (op 15605) 
s_add_u32 s1, s1, 512 // (op 15605) 

v_add_u32 v0, v12, s1 // (op 15605) 


v_accvgpr_read v6, a140 // (op 15627) 
v_accvgpr_read v7, a141 // (op 15627) 
v_accvgpr_read v8, a142 // (op 15627) 
v_accvgpr_read v9, a143 // (op 15627) 
v_mul_f32 v14, v11, v6 // (op 15627) 
v_mul_f32 v15, v11, v7 // (op 15627) 
v_mul_f32 v16, v11, v8 // (op 15627) 
v_mul_f32 v17, v11, v9 // (op 15627) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15605) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15603) 
v_mul_f32 v19, v10, v7 // (op 15603) 
v_mul_f32 v20, v10, v8 // (op 15603) 
v_mul_f32 v21, v10, v9 // (op 15603) 

v_add_f32 v6, v18, v14 // (op 15604) 
v_add_f32 v7, v19, v15 // (op 15604) 
v_add_f32 v8, v20, v16 // (op 15604) 
v_add_f32 v9, v21, v17 // (op 15604) 




s_lshl_b32 s0, s4, 7 // (op 15582) 
s_mul_i32 s1, 3, s0 // (op 15582) 
s_add_u32 s1, s1, 512 // (op 15582) 

v_add_u32 v0, v13, s1 // (op 15582) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15582) Store value




s_lshl_b32 s0, s2, 9 // (op 15609) 
s_add_u32 s0, s0, 512 // (op 15609) 
v_add_u32 v0, v12, s0 // (op 15609) 


v_accvgpr_read v6, a144 // (op 15629) 
v_accvgpr_read v7, a145 // (op 15629) 
v_accvgpr_read v8, a146 // (op 15629) 
v_accvgpr_read v9, a147 // (op 15629) 
v_mul_f32 v14, v11, v6 // (op 15629) 
v_mul_f32 v15, v11, v7 // (op 15629) 
v_mul_f32 v16, v11, v8 // (op 15629) 
v_mul_f32 v17, v11, v9 // (op 15629) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15609) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15607) 
v_mul_f32 v19, v10, v7 // (op 15607) 
v_mul_f32 v20, v10, v8 // (op 15607) 
v_mul_f32 v21, v10, v9 // (op 15607) 

v_add_f32 v6, v18, v14 // (op 15608) 
v_add_f32 v7, v19, v15 // (op 15608) 
v_add_f32 v8, v20, v16 // (op 15608) 
v_add_f32 v9, v21, v17 // (op 15608) 




s_lshl_b32 s0, s4, 9 // (op 15584) 
s_add_u32 s0, s0, 512 // (op 15584) 
v_add_u32 v0, v13, s0 // (op 15584) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15584) Store value




s_lshl_b32 s0, s2, 7 // (op 15613) 
s_mul_i32 s1, 5, s0 // (op 15613) 
s_add_u32 s1, s1, 512 // (op 15613) 

v_add_u32 v0, v12, s1 // (op 15613) 


v_accvgpr_read v6, a148 // (op 15631) 
v_accvgpr_read v7, a149 // (op 15631) 
v_accvgpr_read v8, a150 // (op 15631) 
v_accvgpr_read v9, a151 // (op 15631) 
v_mul_f32 v14, v11, v6 // (op 15631) 
v_mul_f32 v15, v11, v7 // (op 15631) 
v_mul_f32 v16, v11, v8 // (op 15631) 
v_mul_f32 v17, v11, v9 // (op 15631) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15613) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15611) 
v_mul_f32 v19, v10, v7 // (op 15611) 
v_mul_f32 v20, v10, v8 // (op 15611) 
v_mul_f32 v21, v10, v9 // (op 15611) 

v_add_f32 v6, v18, v14 // (op 15612) 
v_add_f32 v7, v19, v15 // (op 15612) 
v_add_f32 v8, v20, v16 // (op 15612) 
v_add_f32 v9, v21, v17 // (op 15612) 




s_lshl_b32 s0, s4, 7 // (op 15586) 
s_mul_i32 s1, 5, s0 // (op 15586) 
s_add_u32 s1, s1, 512 // (op 15586) 

v_add_u32 v0, v13, s1 // (op 15586) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15586) Store value




s_lshl_b32 s0, s2, 7 // (op 15617) 
s_mul_i32 s1, 6, s0 // (op 15617) 
s_add_u32 s1, s1, 512 // (op 15617) 

v_add_u32 v0, v12, s1 // (op 15617) 


v_accvgpr_read v6, a152 // (op 15633) 
v_accvgpr_read v7, a153 // (op 15633) 
v_accvgpr_read v8, a154 // (op 15633) 
v_accvgpr_read v9, a155 // (op 15633) 
v_mul_f32 v14, v11, v6 // (op 15633) 
v_mul_f32 v15, v11, v7 // (op 15633) 
v_mul_f32 v16, v11, v8 // (op 15633) 
v_mul_f32 v17, v11, v9 // (op 15633) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15617) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15615) 
v_mul_f32 v19, v10, v7 // (op 15615) 
v_mul_f32 v20, v10, v8 // (op 15615) 
v_mul_f32 v21, v10, v9 // (op 15615) 

v_add_f32 v6, v18, v14 // (op 15616) 
v_add_f32 v7, v19, v15 // (op 15616) 
v_add_f32 v8, v20, v16 // (op 15616) 
v_add_f32 v9, v21, v17 // (op 15616) 




s_lshl_b32 s0, s4, 7 // (op 15588) 
s_mul_i32 s1, 6, s0 // (op 15588) 
s_add_u32 s1, s1, 512 // (op 15588) 

v_add_u32 v0, v13, s1 // (op 15588) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15588) Store value




s_lshl_b32 s0, s2, 7 // (op 15621) 
s_mul_i32 s1, 7, s0 // (op 15621) 
s_add_u32 s1, s1, 512 // (op 15621) 

v_add_u32 v0, v12, s1 // (op 15621) 


v_accvgpr_read v6, a156 // (op 15635) 
v_accvgpr_read v7, a157 // (op 15635) 
v_accvgpr_read v8, a158 // (op 15635) 
v_accvgpr_read v9, a159 // (op 15635) 
v_mul_f32 v14, v11, v6 // (op 15635) 
v_mul_f32 v15, v11, v7 // (op 15635) 
v_mul_f32 v16, v11, v8 // (op 15635) 
v_mul_f32 v17, v11, v9 // (op 15635) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15621) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15619) 
v_mul_f32 v19, v10, v7 // (op 15619) 
v_mul_f32 v20, v10, v8 // (op 15619) 
v_mul_f32 v21, v10, v9 // (op 15619) 

v_add_f32 v6, v18, v14 // (op 15620) 
v_add_f32 v7, v19, v15 // (op 15620) 
v_add_f32 v8, v20, v16 // (op 15620) 
v_add_f32 v9, v21, v17 // (op 15620) 




s_lshl_b32 s0, s4, 7 // (op 15590) 
s_mul_i32 s1, 7, s0 // (op 15590) 
s_add_u32 s1, s1, 512 // (op 15590) 

v_add_u32 v0, v13, s1 // (op 15590) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15590) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
v_accvgpr_read v6, a0 // (op 142) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v7, a1 // (op 142) 
v_accvgpr_read v8, a2 // (op 142) 
v_accvgpr_read v9, a3 // (op 142) 
v_mul_f32 v14, v11, v6 // (op 142) 
v_mul_f32 v15, v11, v7 // (op 142) 
v_mul_f32 v16, v11, v8 // (op 142) 
v_mul_f32 v17, v11, v9 // (op 142) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 145) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 146) 
v_mul_f32 v19, v10, v7 // (op 146) 
v_mul_f32 v20, v10, v8 // (op 146) 
v_mul_f32 v21, v10, v9 // (op 146) 

v_add_f32 v6, v18, v14 // (op 143) 
v_add_f32 v7, v19, v15 // (op 143) 
v_add_f32 v8, v20, v16 // (op 143) 
v_add_f32 v9, v21, v17 // (op 143) 




v_mov_b32 v0, v13 // (op 144) reg expression

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 14946) 
v_add_u32 v0, v12, s0 // (op 14946) 


v_accvgpr_read v6, a4 // (op 14943) 
v_accvgpr_read v7, a5 // (op 14943) 
v_accvgpr_read v8, a6 // (op 14943) 
v_accvgpr_read v9, a7 // (op 14943) 
v_mul_f32 v14, v11, v6 // (op 14943) 
v_mul_f32 v15, v11, v7 // (op 14943) 
v_mul_f32 v16, v11, v8 // (op 14943) 
v_mul_f32 v17, v11, v9 // (op 14943) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 14946) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 14947) 
v_mul_f32 v19, v10, v7 // (op 14947) 
v_mul_f32 v20, v10, v8 // (op 14947) 
v_mul_f32 v21, v10, v9 // (op 14947) 

v_add_f32 v6, v18, v14 // (op 14944) 
v_add_f32 v7, v19, v15 // (op 14944) 
v_add_f32 v8, v20, v16 // (op 14944) 
v_add_f32 v9, v21, v17 // (op 14944) 




s_lshl_b32 s0, s4, 7 // (op 14945) 
v_add_u32 v0, v13, s0 // (op 14945) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 14945) Store value




s_lshl_b32 s0, s2, 8 // (op 14958) 
v_add_u32 v0, v12, s0 // (op 14958) 


v_accvgpr_read v6, a8 // (op 14955) 
v_accvgpr_read v7, a9 // (op 14955) 
v_accvgpr_read v8, a10 // (op 14955) 
v_accvgpr_read v9, a11 // (op 14955) 
v_mul_f32 v14, v11, v6 // (op 14955) 
v_mul_f32 v15, v11, v7 // (op 14955) 
v_mul_f32 v16, v11, v8 // (op 14955) 
v_mul_f32 v17, v11, v9 // (op 14955) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 14958) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 14959) 
v_mul_f32 v19, v10, v7 // (op 14959) 
v_mul_f32 v20, v10, v8 // (op 14959) 
v_mul_f32 v21, v10, v9 // (op 14959) 

v_add_f32 v6, v18, v14 // (op 14956) 
v_add_f32 v7, v19, v15 // (op 14956) 
v_add_f32 v8, v20, v16 // (op 14956) 
v_add_f32 v9, v21, v17 // (op 14956) 




s_lshl_b32 s0, s4, 8 // (op 14957) 
v_add_u32 v0, v13, s0 // (op 14957) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 14957) Store value




s_lshl_b32 s0, s2, 7 // (op 14970) 
s_mul_i32 s1, 3, s0 // (op 14970) 

v_add_u32 v0, v12, s1 // (op 14970) 


v_accvgpr_read v6, a12 // (op 14967) 
v_accvgpr_read v7, a13 // (op 14967) 
v_accvgpr_read v8, a14 // (op 14967) 
v_accvgpr_read v9, a15 // (op 14967) 
v_mul_f32 v14, v11, v6 // (op 14967) 
v_mul_f32 v15, v11, v7 // (op 14967) 
v_mul_f32 v16, v11, v8 // (op 14967) 
v_mul_f32 v17, v11, v9 // (op 14967) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 14970) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 14971) 
v_mul_f32 v19, v10, v7 // (op 14971) 
v_mul_f32 v20, v10, v8 // (op 14971) 
v_mul_f32 v21, v10, v9 // (op 14971) 

v_add_f32 v6, v18, v14 // (op 14968) 
v_add_f32 v7, v19, v15 // (op 14968) 
v_add_f32 v8, v20, v16 // (op 14968) 
v_add_f32 v9, v21, v17 // (op 14968) 




s_lshl_b32 s0, s4, 7 // (op 14969) 
s_mul_i32 s1, 3, s0 // (op 14969) 

v_add_u32 v0, v13, s1 // (op 14969) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 14969) Store value




s_lshl_b32 s0, s2, 9 // (op 14982) 
v_add_u32 v0, v12, s0 // (op 14982) 


v_accvgpr_read v6, a16 // (op 14979) 
v_accvgpr_read v7, a17 // (op 14979) 
v_accvgpr_read v8, a18 // (op 14979) 
v_accvgpr_read v9, a19 // (op 14979) 
v_mul_f32 v14, v11, v6 // (op 14979) 
v_mul_f32 v15, v11, v7 // (op 14979) 
v_mul_f32 v16, v11, v8 // (op 14979) 
v_mul_f32 v17, v11, v9 // (op 14979) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 14982) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 14983) 
v_mul_f32 v19, v10, v7 // (op 14983) 
v_mul_f32 v20, v10, v8 // (op 14983) 
v_mul_f32 v21, v10, v9 // (op 14983) 

v_add_f32 v6, v18, v14 // (op 14980) 
v_add_f32 v7, v19, v15 // (op 14980) 
v_add_f32 v8, v20, v16 // (op 14980) 
v_add_f32 v9, v21, v17 // (op 14980) 




s_lshl_b32 s0, s4, 9 // (op 14981) 
v_add_u32 v0, v13, s0 // (op 14981) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 14981) Store value




s_lshl_b32 s0, s2, 7 // (op 14994) 
s_mul_i32 s1, 5, s0 // (op 14994) 

v_add_u32 v0, v12, s1 // (op 14994) 


v_accvgpr_read v6, a20 // (op 14991) 
v_accvgpr_read v7, a21 // (op 14991) 
v_accvgpr_read v8, a22 // (op 14991) 
v_accvgpr_read v9, a23 // (op 14991) 
v_mul_f32 v14, v11, v6 // (op 14991) 
v_mul_f32 v15, v11, v7 // (op 14991) 
v_mul_f32 v16, v11, v8 // (op 14991) 
v_mul_f32 v17, v11, v9 // (op 14991) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 14994) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 14995) 
v_mul_f32 v19, v10, v7 // (op 14995) 
v_mul_f32 v20, v10, v8 // (op 14995) 
v_mul_f32 v21, v10, v9 // (op 14995) 

v_add_f32 v6, v18, v14 // (op 14992) 
v_add_f32 v7, v19, v15 // (op 14992) 
v_add_f32 v8, v20, v16 // (op 14992) 
v_add_f32 v9, v21, v17 // (op 14992) 




s_lshl_b32 s0, s4, 7 // (op 14993) 
s_mul_i32 s1, 5, s0 // (op 14993) 

v_add_u32 v0, v13, s1 // (op 14993) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 14993) Store value




s_lshl_b32 s0, s2, 7 // (op 15006) 
s_mul_i32 s1, 6, s0 // (op 15006) 

v_add_u32 v0, v12, s1 // (op 15006) 


v_accvgpr_read v6, a24 // (op 15003) 
v_accvgpr_read v7, a25 // (op 15003) 
v_accvgpr_read v8, a26 // (op 15003) 
v_accvgpr_read v9, a27 // (op 15003) 
v_mul_f32 v14, v11, v6 // (op 15003) 
v_mul_f32 v15, v11, v7 // (op 15003) 
v_mul_f32 v16, v11, v8 // (op 15003) 
v_mul_f32 v17, v11, v9 // (op 15003) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15006) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15007) 
v_mul_f32 v19, v10, v7 // (op 15007) 
v_mul_f32 v20, v10, v8 // (op 15007) 
v_mul_f32 v21, v10, v9 // (op 15007) 

v_add_f32 v6, v18, v14 // (op 15004) 
v_add_f32 v7, v19, v15 // (op 15004) 
v_add_f32 v8, v20, v16 // (op 15004) 
v_add_f32 v9, v21, v17 // (op 15004) 




s_lshl_b32 s0, s4, 7 // (op 15005) 
s_mul_i32 s1, 6, s0 // (op 15005) 

v_add_u32 v0, v13, s1 // (op 15005) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15005) Store value




s_lshl_b32 s0, s2, 7 // (op 15018) 
s_mul_i32 s1, 7, s0 // (op 15018) 

v_add_u32 v0, v12, s1 // (op 15018) 


v_accvgpr_read v6, a28 // (op 15015) 
v_accvgpr_read v7, a29 // (op 15015) 
v_accvgpr_read v8, a30 // (op 15015) 
v_accvgpr_read v9, a31 // (op 15015) 
v_mul_f32 v14, v11, v6 // (op 15015) 
v_mul_f32 v15, v11, v7 // (op 15015) 
v_mul_f32 v16, v11, v8 // (op 15015) 
v_mul_f32 v17, v11, v9 // (op 15015) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 15018) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v10, v6 // (op 15019) 
v_mul_f32 v19, v10, v7 // (op 15019) 
v_mul_f32 v20, v10, v8 // (op 15019) 
v_mul_f32 v21, v10, v9 // (op 15019) 

v_add_f32 v6, v18, v14 // (op 15016) 
v_add_f32 v7, v19, v15 // (op 15016) 
v_add_f32 v8, v20, v16 // (op 15016) 
v_add_f32 v9, v21, v17 // (op 15016) 




s_lshl_b32 s0, s4, 7 // (op 15017) 
s_mul_i32 s1, 7, s0 // (op 15017) 

v_add_u32 v0, v13, s1 // (op 15017) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 15017) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133:








































































s_endpgm  // End of RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC
  .amdhsa_next_free_vgpr 488
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 135168 // lds bytes
  .amdhsa_accum_offset 232
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
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__LDSSYNC.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 135168
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 232
    .agpr_count: 256
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

