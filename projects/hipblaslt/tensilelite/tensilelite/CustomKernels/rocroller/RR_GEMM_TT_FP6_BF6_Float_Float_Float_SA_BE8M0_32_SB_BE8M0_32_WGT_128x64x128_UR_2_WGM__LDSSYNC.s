// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 4518) 
s_mul_hi_i32 s1, s16, s11 // (op 4775) 
v_lshrrev_b32 v0, 7, v1 // (op 4580) 
v_and_b32 v3, 15, v1 // (op 4580) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4580) 


v_and_b32 v0, 63, v1 // (op 4580) 
v_lshrrev_b32 v3, 4, v0 // (op 4580) 

v_lshlrev_b32 v0, 5, v3 // (op 4580) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 4580) 


v_mul_lo_u32 v0, v3, 6 // (op 4580) 

v_lshrrev_b32 v3, 3, v0 // (op 4580) 

v_and_b32 v0, 63, v1 // (op 4638) 
v_and_b32 v4, 48, v1 // (op 4638) 
v_sub_u32 v5, v0, v4 // (op 4638) 

v_lshrrev_b32 v4, 4, v0 // (op 4638) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 4638) 


v_lshrrev_b32 v4, 6, v1 // (op 4638) 
v_and_b32 v5, 1, v4 // (op 4638) 

v_lshlrev_b32 v4, 4, v5 // (op 4638) 

v_lshl_add_u32 v5, v0, 6, v4 // (op 4638) 


v_mul_lo_u32 v0, v5, 6 // (op 4638) 
v_lshrrev_b32 v4, 3, v0 // (op 4638) 

v_lshrrev_b32 v0, 4, v5 // (op 4638) 

v_lshlrev_b32 v5, 2, v0 // (op 4638) 

v_add_u32 v0, v4, v5 // (op 4638) 


v_lshrrev_b32 v4, 7, v1 // (op 4696) 
v_and_b32 v5, 15, v1 // (op 4696) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 4696) 


v_and_b32 v4, 63, v1 // (op 4696) 
v_lshrrev_b32 v5, 4, v4 // (op 4696) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 4696) 


v_lshrrev_b32 v5, 6, v1 // (op 4754) 
v_and_b32 v6, 1, v5 // (op 4754) 

v_and_b32 v5, 15, v1 // (op 4754) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4754) 


v_and_b32 v5, 63, v1 // (op 4754) 
v_lshrrev_b32 v6, 4, v5 // (op 4754) 

v_lshl_add_u32 v5, v6, 6, v7 // (op 4754) 


v_lshrrev_b32 v6, 7, v1 // (op 4806) 
v_and_b32 v7, 15, v1 // (op 4806) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 4806) 


v_and_b32 v6, 63, v1 // (op 4806) 
v_lshrrev_b32 v7, 4, v6 // (op 4806) 

v_lshlrev_b32 v6, 5, v7 // (op 4806) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 4806) 


v_mul_lo_u32 v6, v7, 6 // (op 4806) 

v_lshrrev_b32 v7, 3, v6 // (op 4806) 

v_and_b32 v6, 63, v1 // (op 4835) 
v_and_b32 v8, 48, v1 // (op 4835) 
v_sub_u32 v9, v6, v8 // (op 4835) 

v_lshrrev_b32 v8, 4, v6 // (op 4835) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 4835) 


v_lshrrev_b32 v8, 6, v1 // (op 4835) 
v_and_b32 v9, 1, v8 // (op 4835) 

v_lshlrev_b32 v8, 4, v9 // (op 4835) 

v_lshl_add_u32 v9, v6, 6, v8 // (op 4835) 


v_mul_lo_u32 v6, v9, 6 // (op 4835) 
v_lshrrev_b32 v8, 3, v6 // (op 4835) 

v_lshrrev_b32 v6, 4, v9 // (op 4835) 

v_lshlrev_b32 v9, 2, v6 // (op 4835) 

v_add_u32 v6, v8, v9 // (op 4835) 


v_lshrrev_b32 v8, 7, v1 // (op 4864) 
v_and_b32 v9, 15, v1 // (op 4864) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 4864) 


v_and_b32 v8, 63, v1 // (op 4864) 
v_lshrrev_b32 v9, 4, v8 // (op 4864) 

v_lshl_add_u32 v8, v10, 2, v9 // (op 4864) 


v_lshrrev_b32 v9, 6, v1 // (op 4893) 
v_and_b32 v10, 1, v9 // (op 4893) 

v_and_b32 v9, 15, v1 // (op 4893) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4893) 


v_and_b32 v9, 63, v1 // (op 4893) 
v_lshrrev_b32 v10, 4, v9 // (op 4893) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 4893) 


v_lshrrev_b32 v10, 7, v1 // (op 4518) 
s_add_i32 s18, s0, s16 // (op 4518) 

v_and_b32 v11, 63, v1 // (op 4518) 
s_ashr_i32 s0, s18, 31 // (op 4518) 
v_lshrrev_b32 v12, 4, v11 // (op 4518) 
s_cmp_eq_i32 s11, 0 // (op 4518) 
s_mov_b32 s19, scc // (op 4518) 
v_lshlrev_b32 v13, 2, v12 // (op 4518) 

s_cmp_eq_i32 s19, 0 // (op 4518) 
s_cselect_b32 s51, 0, -1 // (op 4518) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4518) 


s_and_b32 s19, s12, 31 // (op 4518) 
v_lshrrev_b32 v10, 2, v11 // (op 4518) 

v_and_b32 v11, 3, v10 // (op 4518) 

v_lshrrev_b32 v10, 7, v1 // (op 4775) 
v_and_b32 v13, 63, v1 // (op 4775) 
s_lshl_b32 s52, 1, s19 // (op 4518) 
v_and_b32 v14, 3, v1 // (op 4518) 
s_add_i32 s53, s52, s51 // (op 4518) 


s_and_b32 s51, s0, s53 // (op 4518) 


s_add_i32 s0, s1, s16 // (op 4775) 

s_ashr_i32 s1, s0, 31 // (op 4775) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 4518) 


s_add_i32 s52, s18, s51 // (op 4518) 


v_lshrrev_b32 v11, 6, v1 // (op 4518) 
s_ashr_i32 s18, s52, s19 // (op 4518) 


v_and_b32 v14, 1, v11 // (op 4518) 

s_lshl_b32 s19, s12, 24 // (op 4518) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 4518) 


s_ashr_i32 s51, s19, 31 // (op 4518) 

s_xor_b32 s19, s18, s51 // (op 4518) 

s_sub_i32 s18, s19, s51 // (op 4518) 


s_mul_hi_i32 s19, s18, s13 // (op 4518) 
s_add_i32 s51, s19, s18 // (op 4518) 

s_ashr_i32 s19, s51, 31 // (op 4518) 
s_cmp_eq_i32 s13, 0 // (op 4518) 
s_mov_b32 s52, scc // (op 4518) 
s_cmp_eq_i32 s52, 0 // (op 4518) 
s_cselect_b32 s53, 0, -1 // (op 4518) 

s_and_b32 s52, s14, 31 // (op 4518) 
v_lshrrev_b32 v14, 4, v13 // (op 4775) 
s_cmp_eq_i32 s11, 0 // (op 4775) 
s_mov_b32 s54, scc // (op 4775) 
v_lshlrev_b32 v15, 2, v14 // (op 4775) 

s_cmp_eq_i32 s54, 0 // (op 4775) 
s_cselect_b32 s55, 0, -1 // (op 4775) 

s_lshl_b32 s54, 1, s52 // (op 4518) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 4775) 


v_lshrrev_b32 v10, 2, v13 // (op 4775) 

s_add_i32 s56, s54, s53 // (op 4518) 


s_and_b32 s53, s19, s56 // (op 4518) 


s_add_i32 s19, s51, s53 // (op 4518) 


s_ashr_i32 s51, s19, s52 // (op 4518) 


s_lshl_b32 s19, s14, 24 // (op 4518) 
s_ashr_i32 s52, s19, 31 // (op 4518) 

s_xor_b32 s19, s51, s52 // (op 4518) 

s_sub_i32 s51, s19, s52 // (op 4518) 


s_and_b32 s19, s12, 31 // (op 4775) 
s_lshl_b32 s52, 1, s19 // (op 4775) 
v_and_b32 v13, 3, v10 // (op 4775) 

s_add_i32 s53, s52, s55 // (op 4775) 


v_and_b32 v10, 3, v1 // (op 4775) 
s_and_b32 s52, s1, s53 // (op 4775) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 4775) 


s_add_i32 s1, s0, s52 // (op 4775) 


v_lshrrev_b32 v10, 6, v1 // (op 4775) 
s_ashr_i32 s0, s1, s19 // (op 4775) 


v_and_b32 v13, 1, v10 // (op 4775) 

s_lshl_b32 s1, s12, 24 // (op 4775) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 4775) 


s_ashr_i32 s19, s1, 31 // (op 4775) 

s_xor_b32 s1, s0, s19 // (op 4775) 

s_sub_i32 s0, s1, s19 // (op 4775) 


s_mul_hi_i32 s1, s0, s13 // (op 4775) 
s_add_i32 s19, s1, s0 // (op 4775) 

s_ashr_i32 s1, s19, 31 // (op 4775) 
s_cmp_eq_i32 s13, 0 // (op 4775) 
s_mov_b32 s52, scc // (op 4775) 
s_cmp_eq_i32 s52, 0 // (op 4775) 
s_cselect_b32 s53, 0, -1 // (op 4775) 

s_and_b32 s52, s14, 31 // (op 4775) 
s_lshl_b32 s54, 1, s52 // (op 4775) 
s_add_i32 s55, s54, s53 // (op 4775) 


s_and_b32 s53, s1, s55 // (op 4775) 


s_add_i32 s1, s19, s53 // (op 4775) 


s_ashr_i32 s19, s1, s52 // (op 4775) 


s_lshl_b32 s1, s14, 24 // (op 4775) 
s_ashr_i32 s52, s1, 31 // (op 4775) 

s_xor_b32 s1, s19, s52 // (op 4775) 

s_sub_i32 s19, s1, s52 // (op 4775) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 4518) 
s_sub_u32 s52, s16, s1 // (op 4518) 

s_mul_i32 s1, s51, s44 // (op 4518) 
s_add_u32 s1, s1, s52 // (op 4518) 

s_mul_hi_i32 s52, s16, s10 // (op 4518) 
s_add_i32 s53, s52, s16 // (op 4518) 

s_ashr_i32 s52, s53, 31 // (op 4518) 
s_cmp_eq_i32 s10, 0 // (op 4518) 
s_mov_b32 s54, scc // (op 4518) 
s_cmp_eq_i32 s54, 0 // (op 4518) 
s_cselect_b32 s55, 0, -1 // (op 4518) 

s_and_b32 s54, s47, 31 // (op 4518) 
s_lshl_b32 s56, 1, s54 // (op 4518) 
s_add_i32 s57, s56, s55 // (op 4518) 


s_and_b32 s55, s52, s57 // (op 4518) 


s_add_i32 s52, s53, s55 // (op 4518) 


s_ashr_i32 s53, s52, s54 // (op 4518) 


s_lshl_b32 s52, s47, 24 // (op 4518) 
s_ashr_i32 s54, s52, 31 // (op 4518) 

s_xor_b32 s52, s53, s54 // (op 4518) 

s_sub_i32 s53, s52, s54 // (op 4518) 


s_mul_i32 s52, s0, s44 // (op 4775) 
s_sub_u32 s54, s16, s52 // (op 4775) 

s_mul_i32 s52, s19, s44 // (op 4775) 
s_add_u32 s52, s52, s54 // (op 4775) 

s_mul_hi_i32 s54, s16, s10 // (op 4775) 
s_add_i32 s55, s54, s16 // (op 4775) 

s_ashr_i32 s54, s55, 31 // (op 4775) 
s_cmp_eq_i32 s10, 0 // (op 4775) 
s_mov_b32 s56, scc // (op 4775) 
s_cmp_eq_i32 s56, 0 // (op 4775) 
s_cselect_b32 s57, 0, -1 // (op 4775) 

s_and_b32 s56, s47, 31 // (op 4775) 
s_lshl_b32 s58, 1, s56 // (op 4775) 
s_add_i32 s59, s58, s57 // (op 4775) 


s_and_b32 s57, s54, s59 // (op 4775) 


s_add_i32 s54, s55, s57 // (op 4775) 


s_ashr_i32 s55, s54, s56 // (op 4775) 


s_lshl_b32 s54, s47, 24 // (op 4775) 
s_ashr_i32 s56, s54, 31 // (op 4775) 

s_xor_b32 s54, s55, s56 // (op 4775) 

s_sub_i32 s55, s54, s56 // (op 4775) 


s_add_u32 s56, s6, 64 // (op 4518) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 4518) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4518) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4518) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 4518) 

s_mul_i32 s54, s44, s56 // (op 4518) 
s_mul_i32 s58, s48, s54 // (op 4518) 

s_mul_i32 s54, s53, s58 // (op 4518) 

s_sub_u32 s58, s16, s54 // (op 4518) 

s_mov_b32 s54, s58 // (op 4518) 
s_mul_hi_i32 s59, s54, s15 // (op 4518) 
s_add_i32 s60, s59, s54 // (op 4518) 


s_ashr_i32 s54, s60, 31 // (op 4518) 
s_cmp_eq_i32 s15, 0 // (op 4518) 
s_mov_b32 s59, scc // (op 4518) 
s_cmp_eq_i32 s59, 0 // (op 4518) 
s_cselect_b32 s61, 0, -1 // (op 4518) 

s_and_b32 s59, s46, 31 // (op 4518) 
s_lshl_b32 s62, 1, s59 // (op 4518) 
s_add_i32 s63, s62, s61 // (op 4518) 


s_and_b32 s61, s54, s63 // (op 4518) 


s_add_i32 s54, s60, s61 // (op 4518) 


s_ashr_i32 s60, s54, s59 // (op 4518) 


s_lshl_b32 s54, s46, 24 // (op 4518) 
s_ashr_i32 s59, s54, 31 // (op 4518) 

s_xor_b32 s54, s60, s59 // (op 4518) 

s_sub_i32 s60, s54, s59 // (op 4518) 


s_mul_i32 s54, s45, s44 // (op 4518) 
s_add_u32 s62, s8, 128 // (op 4518) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 4518) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 4518) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 4518) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 7 // (op 4518) 

s_sub_i32 s59, s62, s54 // (op 4518) 

s_mul_i32 s61, s60, s59 // (op 4518) 

s_sub_u32 s59, s58, s61 // (op 4518) 


s_add_u32 s58, s59, s54 // (op 4518) 


s_cmp_eq_i32 s53, 0 // (op 4518) 
s_mov_b32 s54, scc // (op 4518) 

s_cmp_eq_i32 s54, 0 // (op 4518) 
s_cselect_b32 s53, s58, s1 // (op 4518) 


s_mul_i32 s1, s51, s56 // (op 4518) 


v_lshl_add_u32 v13, s53, 7, v12 // (op 4518) 


s_sub_i32 s51, s18, s1 // (op 4518) 


s_cmp_eq_i32 s54, 0 // (op 4518) 
s_cselect_b32 s1, s60, s51 // (op 4518) 



v_lshl_add_u32 v12, s1, 6, v11 // (op 4518) 


v_mul_lo_u32 v11, v12, s2 // (op 4518) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 4518) 


s_add_u32 s56, s6, 64 // (op 4775) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 4775) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4775) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4775) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 4775) 

s_mul_i32 s1, s44, s56 // (op 4775) 
s_mul_i32 s18, s48, s1 // (op 4775) 

s_mul_i32 s1, s55, s18 // (op 4775) 

s_sub_u32 s18, s16, s1 // (op 4775) 

s_mov_b32 s1, s18 // (op 4775) 
s_mul_hi_i32 s51, s1, s15 // (op 4775) 
s_add_i32 s53, s51, s1 // (op 4775) 


s_ashr_i32 s1, s53, 31 // (op 4775) 
s_cmp_eq_i32 s15, 0 // (op 4775) 
s_mov_b32 s51, scc // (op 4775) 
s_cmp_eq_i32 s51, 0 // (op 4775) 
s_cselect_b32 s54, 0, -1 // (op 4775) 

s_and_b32 s51, s46, 31 // (op 4775) 
s_lshl_b32 s58, 1, s51 // (op 4775) 
s_add_i32 s59, s58, s54 // (op 4775) 


s_and_b32 s54, s1, s59 // (op 4775) 


s_add_i32 s1, s53, s54 // (op 4775) 


s_ashr_i32 s53, s1, s51 // (op 4775) 


s_lshl_b32 s1, s46, 24 // (op 4775) 
s_ashr_i32 s51, s1, 31 // (op 4775) 

s_xor_b32 s1, s53, s51 // (op 4775) 

s_sub_i32 s53, s1, s51 // (op 4775) 


s_mul_i32 s1, s45, s44 // (op 4775) 
s_add_u32 s58, s8, 128 // (op 4775) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 4775) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 4775) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 4775) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 7 // (op 4775) 

s_sub_i32 s51, s58, s1 // (op 4775) 

s_mul_i32 s54, s53, s51 // (op 4775) 

s_sub_u32 s51, s18, s54 // (op 4775) 


s_add_u32 s18, s51, s1 // (op 4775) 


s_cmp_eq_i32 s55, 0 // (op 4775) 
s_mov_b32 s1, scc // (op 4775) 

s_cmp_eq_i32 s1, 0 // (op 4775) 
s_cselect_b32 s51, s18, s52 // (op 4775) 


s_mul_i32 s18, s19, s56 // (op 4775) 


v_lshl_add_u32 v11, s51, 7, v14 // (op 4775) 


s_sub_i32 s19, s0, s18 // (op 4775) 


s_cmp_eq_i32 s1, 0 // (op 4775) 
s_cselect_b32 s0, s53, s19 // (op 4775) 



v_lshl_add_u32 v13, s0, 6, v10 // (op 4775) 


v_mul_lo_u32 v10, v13, s4 // (op 4775) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 4775) 


s_mul_i32 s1, s2, s7 // (op 4520) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 4520) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 4520) least significant
s_mul_hi_u32 s19, s2, s6 // (op 4520) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4520) most significant: sum
s_add_u32 s1, s1, s19 // (op 4520) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4520) 

s_mul_i32 s1, s4, s7 // (op 4777) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 4777) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 4777) least significant
s_mul_hi_u32 s52, s4, s6 // (op 4777) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 4777) most significant: sum
s_add_u32 s1, s1, s52 // (op 4777) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 4777) 

s_mov_b32 s56, s20 // (op 4520) reg expression
s_mov_b32 s57, s21 // (op 4520) reg expression
s_mov_b32 s58, s18 // (op 4520) reg expression
s_mov_b32 s59, 131072 // (op 4520) call()

s_mov_b32 s60, s22 // (op 4777) reg expression

s_mov_b32 s61, s23 // (op 4777) reg expression
s_mov_b32 s62, s52 // (op 4777) reg expression
s_mov_b32 s63, 131072 // (op 4777) call()


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
v_accvgpr_write a16, 0.00000 // (op 942) call()
v_accvgpr_write a17, 0.00000 // (op 942) call()
v_accvgpr_write a18, 0.00000 // (op 942) call()
v_accvgpr_write a19, 0.00000 // (op 942) call()
v_accvgpr_write a20, 0.00000 // (op 1097) call()
v_accvgpr_write a21, 0.00000 // (op 1097) call()
v_accvgpr_write a22, 0.00000 // (op 1097) call()
v_accvgpr_write a23, 0.00000 // (op 1097) call()
v_accvgpr_write a24, 0.00000 // (op 1328) call()
v_accvgpr_write a25, 0.00000 // (op 1328) call()
v_accvgpr_write a26, 0.00000 // (op 1328) call()
v_accvgpr_write a27, 0.00000 // (op 1328) call()
v_accvgpr_write a28, 0.00000 // (op 1483) call()
v_accvgpr_write a29, 0.00000 // (op 1483) call()
v_accvgpr_write a30, 0.00000 // (op 1483) call()
v_accvgpr_write a31, 0.00000 // (op 1483) call()
s_mul_hi_i32 s0, s16, s11 // (op 4280) 
v_lshrrev_b32 v14, 1, v1 // (op 4280) 
s_add_i32 s1, s0, s16 // (op 4280) 

v_and_b32 v15, -2, v1 // (op 4280) 
s_ashr_i32 s0, s1, 31 // (op 4280) 
v_sub_u32 v16, v1, v15 // (op 4280) 

s_cmp_eq_i32 s11, 0 // (op 4280) 
s_mov_b32 s18, scc // (op 4280) 
v_lshlrev_b32 v15, 6, v16 // (op 4280) 

s_cmp_eq_i32 s18, 0 // (op 4280) 
s_cselect_b32 s19, 0, -1 // (op 4280) 

v_mov_b32 v17, 0 // (op 4280) convert
v_mov_b32 v16, v15 // (op 4280) convert

s_and_b32 s18, s12, 31 // (op 4280) 
s_lshl_b32 s20, 1, s18 // (op 4280) 
s_add_i32 s21, s20, s19 // (op 4280) 


s_and_b32 s19, s0, s21 // (op 4280) 


s_add_i32 s0, s1, s19 // (op 4280) 


s_ashr_i32 s1, s0, s18 // (op 4280) 


s_lshl_b32 s0, s12, 24 // (op 4280) 
s_ashr_i32 s18, s0, 31 // (op 4280) 

s_xor_b32 s0, s1, s18 // (op 4280) 

s_sub_i32 s1, s0, s18 // (op 4280) 


s_mul_hi_i32 s0, s1, s13 // (op 4280) 
s_add_i32 s18, s0, s1 // (op 4280) 

s_ashr_i32 s0, s18, 31 // (op 4280) 
s_cmp_eq_i32 s13, 0 // (op 4280) 
s_mov_b32 s19, scc // (op 4280) 
s_cmp_eq_i32 s19, 0 // (op 4280) 
s_cselect_b32 s20, 0, -1 // (op 4280) 

s_and_b32 s19, s14, 31 // (op 4280) 
s_lshl_b32 s21, 1, s19 // (op 4280) 
s_add_i32 s22, s21, s20 // (op 4280) 


s_and_b32 s23, s0, s22 // (op 4280) 


s_add_i32 s0, s18, s23 // (op 4280) 


s_ashr_i32 s18, s0, s19 // (op 4280) 


s_lshl_b32 s0, s14, 24 // (op 4280) 
s_ashr_i32 s19, s0, 31 // (op 4280) 

s_xor_b32 s0, s18, s19 // (op 4280) 

s_sub_i32 s18, s0, s19 // (op 4280) 


s_mul_i32 s0, s1, s44 // (op 4280) 

s_sub_u32 s1, s16, s0 // (op 4280) 

s_mul_i32 s0, s18, s44 // (op 4280) 
s_add_u32 s0, s0, s1 // (op 4280) 


s_mul_hi_i32 s1, s16, s10 // (op 4280) 
s_add_i32 s18, s1, s16 // (op 4280) 

s_ashr_i32 s1, s18, 31 // (op 4280) 
s_cmp_eq_i32 s10, 0 // (op 4280) 
s_mov_b32 s19, scc // (op 4280) 
s_cmp_eq_i32 s19, 0 // (op 4280) 
s_cselect_b32 s20, 0, -1 // (op 4280) 

s_and_b32 s19, s47, 31 // (op 4280) 
s_lshl_b32 s21, 1, s19 // (op 4280) 
s_add_i32 s22, s21, s20 // (op 4280) 


s_and_b32 s23, s1, s22 // (op 4280) 


s_add_i32 s1, s18, s23 // (op 4280) 


s_ashr_i32 s18, s1, s19 // (op 4280) 


s_lshl_b32 s1, s47, 24 // (op 4280) 
s_ashr_i32 s19, s1, 31 // (op 4280) 

s_xor_b32 s1, s18, s19 // (op 4280) 

s_sub_i32 s18, s1, s19 // (op 4280) 


s_add_u32 s22, s6, 64 // (op 4280) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4280) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 4280) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 4280) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 4280) 

s_mul_i32 s1, s44, s22 // (op 4280) 

s_mul_i32 s19, s48, s1 // (op 4280) 

s_mul_i32 s1, s18, s19 // (op 4280) 

s_sub_u32 s19, s16, s1 // (op 4280) 

s_mov_b32 s1, s19 // (op 4280) 
s_mul_hi_i32 s20, s1, s15 // (op 4280) 
s_add_i32 s21, s20, s1 // (op 4280) 


s_ashr_i32 s1, s21, 31 // (op 4280) 
s_cmp_eq_i32 s15, 0 // (op 4280) 
s_mov_b32 s20, scc // (op 4280) 
s_cmp_eq_i32 s20, 0 // (op 4280) 
s_cselect_b32 s22, 0, -1 // (op 4280) 

s_and_b32 s20, s46, 31 // (op 4280) 
s_lshl_b32 s23, 1, s20 // (op 4280) 
s_add_i32 s49, s23, s22 // (op 4280) 


s_and_b32 s22, s1, s49 // (op 4280) 


s_add_i32 s1, s21, s22 // (op 4280) 


s_ashr_i32 s21, s1, s20 // (op 4280) 


s_lshl_b32 s1, s46, 24 // (op 4280) 
s_ashr_i32 s20, s1, 31 // (op 4280) 

s_xor_b32 s1, s21, s20 // (op 4280) 

s_sub_i32 s21, s1, s20 // (op 4280) 


s_mul_i32 s1, s45, s44 // (op 4280) 
s_add_u32 s22, s8, 128 // (op 4280) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 4280) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 4280) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 4280) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 7 // (op 4280) 

s_sub_i32 s20, s22, s1 // (op 4280) 

s_mul_i32 s22, s21, s20 // (op 4280) 


s_sub_u32 s23, s19, s22 // (op 4280) 


s_add_u32 s19, s23, s1 // (op 4280) 


s_cmp_eq_i32 s18, 0 // (op 4280) 
s_mov_b32 s1, scc // (op 4280) 

s_cmp_eq_i32 s1, 0 // (op 4280) 
s_cselect_b32 s18, s19, s0 // (op 4280) 



v_lshl_add_u32 v15, s18, 7, v14 // (op 4280) 


v_mul_lo_u32 v19, v15, s25 // (op 4280) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 4280) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 4280) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 4280) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 4280) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 4280) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 4280) most significant: low of high * low
v_mul_hi_u32 v16, v18, 6 // (op 4280) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 4280) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 4280) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 4280) 

v_mov_b32 v14, v16 // (op 4280) convert

s_mul_i32 s1, s9, 6 // (op 4282) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 4282) least significant
s_mul_hi_u32 s18, s8, 6 // (op 4282) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4282) most significant: sum

s_mul_i32 s19, s24, s1 // (op 4282) most significant: low of low * high
s_mul_i32 s20, s25, s0 // (op 4282) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 4282) least significant
s_mul_hi_u32 s21, s24, s0 // (op 4282) most significant: high of low * low
s_add_u32 s19, s19, s20 // (op 4282) most significant: sum
s_add_u32 s19, s19, s21 // (op 4282) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4282) 

s_mov_b32 s20, s26 // (op 4282) reg expression
s_mov_b32 s21, s27 // (op 4282) reg expression
s_mov_b32 s22, s0 // (op 4282) reg expression
s_mov_b32 s23, 131072 // (op 4282) call()

s_mul_hi_i32 s0, s16, s11 // (op 4299) 
v_lshrrev_b32 v15, 1, v1 // (op 4299) 
s_add_i32 s1, s0, s16 // (op 4299) 

v_and_b32 v16, -2, v1 // (op 4299) 
s_ashr_i32 s0, s1, 31 // (op 4299) 
v_sub_u32 v17, v1, v16 // (op 4299) 

s_cmp_eq_i32 s11, 0 // (op 4299) 
s_mov_b32 s18, scc // (op 4299) 
v_lshlrev_b32 v16, 6, v17 // (op 4299) 

s_cmp_eq_i32 s18, 0 // (op 4299) 
s_cselect_b32 s19, 0, -1 // (op 4299) 

v_mov_b32 v19, 0 // (op 4299) convert
v_mov_b32 v18, v16 // (op 4299) convert

s_and_b32 s18, s12, 31 // (op 4299) 
s_lshl_b32 s49, 1, s18 // (op 4299) 
s_add_i32 s50, s49, s19 // (op 4299) 


s_and_b32 s19, s0, s50 // (op 4299) 


s_add_i32 s0, s1, s19 // (op 4299) 


s_ashr_i32 s1, s0, s18 // (op 4299) 


s_lshl_b32 s0, s12, 24 // (op 4299) 
s_ashr_i32 s18, s0, 31 // (op 4299) 

s_xor_b32 s0, s1, s18 // (op 4299) 

s_sub_i32 s1, s0, s18 // (op 4299) 


s_mul_hi_i32 s0, s1, s13 // (op 4299) 
s_add_i32 s18, s0, s1 // (op 4299) 

s_ashr_i32 s0, s18, 31 // (op 4299) 
s_cmp_eq_i32 s13, 0 // (op 4299) 
s_mov_b32 s19, scc // (op 4299) 
s_cmp_eq_i32 s19, 0 // (op 4299) 
s_cselect_b32 s49, 0, -1 // (op 4299) 

s_and_b32 s19, s14, 31 // (op 4299) 
s_lshl_b32 s50, 1, s19 // (op 4299) 
s_add_i32 s51, s50, s49 // (op 4299) 


s_and_b32 s49, s0, s51 // (op 4299) 


s_add_i32 s0, s18, s49 // (op 4299) 


s_ashr_i32 s18, s0, s19 // (op 4299) 


s_lshl_b32 s0, s14, 24 // (op 4299) 
s_ashr_i32 s19, s0, 31 // (op 4299) 

s_xor_b32 s0, s18, s19 // (op 4299) 

s_sub_i32 s18, s0, s19 // (op 4299) 


s_mul_i32 s0, s1, s44 // (op 4299) 

s_sub_u32 s1, s16, s0 // (op 4299) 

s_mul_i32 s0, s18, s44 // (op 4299) 
s_add_u32 s0, s0, s1 // (op 4299) 


s_mul_hi_i32 s1, s16, s10 // (op 4299) 
s_add_i32 s18, s1, s16 // (op 4299) 

s_ashr_i32 s1, s18, 31 // (op 4299) 
s_cmp_eq_i32 s10, 0 // (op 4299) 
s_mov_b32 s19, scc // (op 4299) 
s_cmp_eq_i32 s19, 0 // (op 4299) 
s_cselect_b32 s49, 0, -1 // (op 4299) 

s_and_b32 s19, s47, 31 // (op 4299) 
s_lshl_b32 s50, 1, s19 // (op 4299) 
s_add_i32 s51, s50, s49 // (op 4299) 


s_and_b32 s49, s1, s51 // (op 4299) 


s_add_i32 s1, s18, s49 // (op 4299) 


s_ashr_i32 s18, s1, s19 // (op 4299) 


s_lshl_b32 s1, s47, 24 // (op 4299) 
s_ashr_i32 s19, s1, 31 // (op 4299) 

s_xor_b32 s1, s18, s19 // (op 4299) 

s_sub_i32 s18, s1, s19 // (op 4299) 


s_add_u32 s54, s6, 64 // (op 4299) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 4299) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 4299) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 4299) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 4299) 

s_mul_i32 s1, s44, s50 // (op 4299) 

s_mul_i32 s19, s48, s1 // (op 4299) 

s_mul_i32 s1, s18, s19 // (op 4299) 

s_sub_u32 s19, s16, s1 // (op 4299) 

s_mov_b32 s1, s19 // (op 4299) 
s_mul_hi_i32 s49, s1, s15 // (op 4299) 
s_add_i32 s50, s49, s1 // (op 4299) 


s_ashr_i32 s1, s50, 31 // (op 4299) 
s_cmp_eq_i32 s15, 0 // (op 4299) 
s_mov_b32 s49, scc // (op 4299) 
s_cmp_eq_i32 s49, 0 // (op 4299) 
s_cselect_b32 s51, 0, -1 // (op 4299) 

s_and_b32 s49, s46, 31 // (op 4299) 
s_lshl_b32 s52, 1, s49 // (op 4299) 
s_add_i32 s53, s52, s51 // (op 4299) 


s_and_b32 s51, s1, s53 // (op 4299) 


s_add_i32 s1, s50, s51 // (op 4299) 


s_ashr_i32 s50, s1, s49 // (op 4299) 


s_lshl_b32 s1, s46, 24 // (op 4299) 
s_ashr_i32 s49, s1, 31 // (op 4299) 

s_xor_b32 s1, s50, s49 // (op 4299) 

s_sub_i32 s50, s1, s49 // (op 4299) 


s_mul_i32 s1, s45, s44 // (op 4299) 
s_add_u32 s54, s8, 128 // (op 4299) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 4299) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 4299) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 4299) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 7 // (op 4299) 

s_sub_i32 s49, s54, s1 // (op 4299) 

s_mul_i32 s51, s50, s49 // (op 4299) 


s_sub_u32 s49, s19, s51 // (op 4299) 


s_add_u32 s19, s49, s1 // (op 4299) 


s_cmp_eq_i32 s18, 0 // (op 4299) 
s_mov_b32 s1, scc // (op 4299) 

s_cmp_eq_i32 s1, 0 // (op 4299) 
s_cselect_b32 s18, s19, s0 // (op 4299) 



v_lshl_add_u32 v16, s18, 7, v15 // (op 4299) 


v_mul_lo_u32 v21, v16, s25 // (op 4299) most significant: low of low * high
v_mul_hi_u32 v15, v16, s24 // (op 4299) most significant: high of low * low
v_mul_lo_u32 v20, v16, s24 // (op 4299) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 4299) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 4299) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 4299) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 4299) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 4299) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 4299) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 4299) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 4299) 

v_mov_b32 v15, v16 // (op 4299) convert

s_mul_i32 s1, s9, 6 // (op 4301) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 4301) least significant
s_mul_hi_u32 s18, s8, 6 // (op 4301) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4301) most significant: sum

s_mul_i32 s19, s24, s1 // (op 4301) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 4301) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 4301) least significant
s_mul_hi_u32 s50, s24, s0 // (op 4301) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4301) most significant: sum
s_add_u32 s19, s19, s50 // (op 4301) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4301) 

s_mov_b32 s20, s26 // (op 4301) reg expression
s_mov_b32 s21, s27 // (op 4301) reg expression
s_mov_b32 s22, s0 // (op 4301) reg expression
s_mov_b32 s23, 131072 // (op 4301) call()

s_mul_hi_i32 s0, s16, s11 // (op 4344) 
v_and_b32 v16, -2, v1 // (op 4344) 
s_add_i32 s1, s0, s16 // (op 4344) 

v_sub_u32 v17, v1, v16 // (op 4344) 

s_ashr_i32 s0, s1, 31 // (op 4344) 
v_lshlrev_b32 v16, 5, v17 // (op 4344) 

s_cmp_eq_i32 s11, 0 // (op 4344) 
s_mov_b32 s18, scc // (op 4344) 
v_lshrrev_b32 v17, 1, v1 // (op 4344) 
s_cmp_eq_i32 s18, 0 // (op 4344) 
s_cselect_b32 s19, 0, -1 // (op 4344) 

s_and_b32 s18, s12, 31 // (op 4344) 
s_lshl_b32 s49, 1, s18 // (op 4344) 
s_add_i32 s50, s49, s19 // (op 4344) 


s_and_b32 s19, s0, s50 // (op 4344) 


s_add_i32 s0, s1, s19 // (op 4344) 


s_ashr_i32 s1, s0, s18 // (op 4344) 


s_lshl_b32 s0, s12, 24 // (op 4344) 
s_ashr_i32 s18, s0, 31 // (op 4344) 

s_xor_b32 s0, s1, s18 // (op 4344) 

s_sub_i32 s1, s0, s18 // (op 4344) 


s_mul_hi_i32 s0, s1, s13 // (op 4344) 
s_add_i32 s18, s0, s1 // (op 4344) 

s_ashr_i32 s0, s18, 31 // (op 4344) 
s_cmp_eq_i32 s13, 0 // (op 4344) 
s_mov_b32 s19, scc // (op 4344) 
s_cmp_eq_i32 s19, 0 // (op 4344) 
s_cselect_b32 s49, 0, -1 // (op 4344) 

s_and_b32 s19, s14, 31 // (op 4344) 
s_lshl_b32 s50, 1, s19 // (op 4344) 
s_add_i32 s51, s50, s49 // (op 4344) 


s_and_b32 s49, s0, s51 // (op 4344) 


s_add_i32 s0, s18, s49 // (op 4344) 


s_ashr_i32 s18, s0, s19 // (op 4344) 


s_lshl_b32 s0, s14, 24 // (op 4344) 
s_ashr_i32 s19, s0, 31 // (op 4344) 

s_xor_b32 s0, s18, s19 // (op 4344) 

s_sub_i32 s18, s0, s19 // (op 4344) 


s_add_u32 s54, s6, 64 // (op 4344) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 4344) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 4344) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 4344) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 4344) 

s_mul_i32 s0, s18, s50 // (op 4344) 

s_sub_i32 s49, s1, s0 // (op 4344) 


s_mul_hi_i32 s0, s16, s10 // (op 4344) 
s_add_i32 s1, s0, s16 // (op 4344) 

s_ashr_i32 s0, s1, 31 // (op 4344) 
s_cmp_eq_i32 s10, 0 // (op 4344) 
s_mov_b32 s18, scc // (op 4344) 
s_cmp_eq_i32 s18, 0 // (op 4344) 
s_cselect_b32 s19, 0, -1 // (op 4344) 

s_and_b32 s18, s47, 31 // (op 4344) 
s_lshl_b32 s52, 1, s18 // (op 4344) 
s_add_i32 s53, s52, s19 // (op 4344) 


s_and_b32 s19, s0, s53 // (op 4344) 


s_add_i32 s0, s1, s19 // (op 4344) 


s_ashr_i32 s1, s0, s18 // (op 4344) 


s_lshl_b32 s0, s47, 24 // (op 4344) 
s_ashr_i32 s18, s0, 31 // (op 4344) 

s_xor_b32 s0, s1, s18 // (op 4344) 

s_sub_i32 s1, s0, s18 // (op 4344) 


s_mul_i32 s0, s44, s50 // (op 4344) 

s_mul_i32 s18, s48, s0 // (op 4344) 

s_mul_i32 s0, s1, s18 // (op 4344) 

s_sub_u32 s18, s16, s0 // (op 4344) 

s_mul_hi_i32 s0, s18, s15 // (op 4344) 
s_add_i32 s19, s0, s18 // (op 4344) 


s_ashr_i32 s0, s19, 31 // (op 4344) 
s_cmp_eq_i32 s15, 0 // (op 4344) 
s_mov_b32 s18, scc // (op 4344) 
s_cmp_eq_i32 s18, 0 // (op 4344) 
s_cselect_b32 s50, 0, -1 // (op 4344) 

s_and_b32 s18, s46, 31 // (op 4344) 
s_lshl_b32 s51, 1, s18 // (op 4344) 
s_add_i32 s52, s51, s50 // (op 4344) 


s_and_b32 s50, s0, s52 // (op 4344) 


s_add_i32 s0, s19, s50 // (op 4344) 


s_ashr_i32 s19, s0, s18 // (op 4344) 


s_lshl_b32 s0, s46, 24 // (op 4344) 
s_ashr_i32 s18, s0, 31 // (op 4344) 

s_xor_b32 s0, s19, s18 // (op 4344) 

s_sub_i32 s19, s0, s18 // (op 4344) 


s_cmp_eq_i32 s1, 0 // (op 4344) 
s_mov_b32 s0, scc // (op 4344) 

s_cmp_eq_i32 s0, 0 // (op 4344) 
s_cselect_b32 s1, s19, s49 // (op 4344) 



v_lshl_add_u32 v18, s1, 6, v16 // (op 4344) 


v_mov_b32 v21, 0 // (op 4344) convert
v_mov_b32 v20, v18 // (op 4344) convert

v_mul_lo_u32 v19, v17, s29 // (op 4344) most significant: low of low * high
v_mul_hi_u32 v16, v17, s28 // (op 4344) most significant: high of low * low
v_mul_lo_u32 v18, v17, s28 // (op 4344) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 4344) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 4344) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 4344) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 4344) most significant: low of high * low
v_mul_hi_u32 v20, v18, 6 // (op 4344) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 4344) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 4344) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 4344) 

v_mov_b32 v16, v18 // (op 4344) convert

s_mul_i32 s1, s33, 6 // (op 4346) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 4346) least significant
s_mul_hi_u32 s18, s32, 6 // (op 4346) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4346) most significant: sum

s_mul_i32 s19, s28, s1 // (op 4346) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 4346) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 4346) least significant
s_mul_hi_u32 s50, s28, s0 // (op 4346) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4346) most significant: sum
s_add_u32 s19, s19, s50 // (op 4346) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4346) 

s_mov_b32 s52, s30 // (op 4346) reg expression
s_mov_b32 s53, s31 // (op 4346) reg expression
s_mov_b32 s54, s0 // (op 4346) reg expression
s_mov_b32 s55, 131072 // (op 4346) call()

s_mul_hi_i32 s0, s16, s11 // (op 4363) 
v_and_b32 v17, -2, v1 // (op 4363) 
s_add_i32 s1, s0, s16 // (op 4363) 

v_sub_u32 v18, v1, v17 // (op 4363) 

s_ashr_i32 s0, s1, 31 // (op 4363) 
v_lshlrev_b32 v17, 5, v18 // (op 4363) 

s_cmp_eq_i32 s11, 0 // (op 4363) 
s_mov_b32 s18, scc // (op 4363) 
v_lshrrev_b32 v18, 1, v1 // (op 4363) 
s_cmp_eq_i32 s18, 0 // (op 4363) 
s_cselect_b32 s19, 0, -1 // (op 4363) 

s_and_b32 s18, s12, 31 // (op 4363) 
s_lshl_b32 s49, 1, s18 // (op 4363) 
s_add_i32 s50, s49, s19 // (op 4363) 


s_and_b32 s19, s0, s50 // (op 4363) 


s_add_i32 s0, s1, s19 // (op 4363) 


s_ashr_i32 s1, s0, s18 // (op 4363) 


s_lshl_b32 s0, s12, 24 // (op 4363) 
s_ashr_i32 s18, s0, 31 // (op 4363) 

s_xor_b32 s0, s1, s18 // (op 4363) 

s_sub_i32 s1, s0, s18 // (op 4363) 


s_mul_hi_i32 s0, s1, s13 // (op 4363) 
s_add_i32 s18, s0, s1 // (op 4363) 

s_ashr_i32 s0, s18, 31 // (op 4363) 
s_cmp_eq_i32 s13, 0 // (op 4363) 
s_mov_b32 s19, scc // (op 4363) 
s_cmp_eq_i32 s19, 0 // (op 4363) 
s_cselect_b32 s49, 0, -1 // (op 4363) 

s_and_b32 s19, s14, 31 // (op 4363) 
s_lshl_b32 s50, 1, s19 // (op 4363) 
s_add_i32 s51, s50, s49 // (op 4363) 


s_and_b32 s49, s0, s51 // (op 4363) 


s_add_i32 s0, s18, s49 // (op 4363) 


s_ashr_i32 s18, s0, s19 // (op 4363) 


s_lshl_b32 s0, s14, 24 // (op 4363) 
s_ashr_i32 s19, s0, 31 // (op 4363) 

s_xor_b32 s0, s18, s19 // (op 4363) 

s_sub_i32 s18, s0, s19 // (op 4363) 


s_add_u32 s50, s6, 64 // (op 4363) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4363) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 4363) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 4363) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 4363) 

s_mul_i32 s0, s18, s50 // (op 4363) 

s_sub_i32 s49, s1, s0 // (op 4363) 


s_mul_hi_i32 s0, s16, s10 // (op 4363) 
s_add_i32 s1, s0, s16 // (op 4363) 

s_ashr_i32 s0, s1, 31 // (op 4363) 
s_cmp_eq_i32 s10, 0 // (op 4363) 
s_mov_b32 s18, scc // (op 4363) 
s_cmp_eq_i32 s18, 0 // (op 4363) 
s_cselect_b32 s19, 0, -1 // (op 4363) 

s_and_b32 s18, s47, 31 // (op 4363) 
s_lshl_b32 s64, 1, s18 // (op 4363) 
s_add_i32 s65, s64, s19 // (op 4363) 


s_and_b32 s19, s0, s65 // (op 4363) 


s_add_i32 s0, s1, s19 // (op 4363) 


s_ashr_i32 s1, s0, s18 // (op 4363) 


s_lshl_b32 s0, s47, 24 // (op 4363) 
s_ashr_i32 s18, s0, 31 // (op 4363) 

s_xor_b32 s0, s1, s18 // (op 4363) 

s_sub_i32 s1, s0, s18 // (op 4363) 


s_mul_i32 s0, s44, s50 // (op 4363) 

s_mul_i32 s18, s48, s0 // (op 4363) 

s_mul_i32 s0, s1, s18 // (op 4363) 

s_sub_u32 s18, s16, s0 // (op 4363) 

s_mul_hi_i32 s0, s18, s15 // (op 4363) 
s_add_i32 s19, s0, s18 // (op 4363) 


s_ashr_i32 s0, s19, 31 // (op 4363) 
s_cmp_eq_i32 s15, 0 // (op 4363) 
s_mov_b32 s18, scc // (op 4363) 
s_cmp_eq_i32 s18, 0 // (op 4363) 
s_cselect_b32 s50, 0, -1 // (op 4363) 

s_and_b32 s18, s46, 31 // (op 4363) 
s_lshl_b32 s51, 1, s18 // (op 4363) 
s_add_i32 s64, s51, s50 // (op 4363) 


s_and_b32 s50, s0, s64 // (op 4363) 


s_add_i32 s0, s19, s50 // (op 4363) 


s_ashr_i32 s19, s0, s18 // (op 4363) 


s_lshl_b32 s0, s46, 24 // (op 4363) 
s_ashr_i32 s18, s0, 31 // (op 4363) 

s_xor_b32 s0, s19, s18 // (op 4363) 

s_sub_i32 s19, s0, s18 // (op 4363) 


s_cmp_eq_i32 s1, 0 // (op 4363) 
s_mov_b32 s0, scc // (op 4363) 

s_cmp_eq_i32 s0, 0 // (op 4363) 
s_cselect_b32 s1, s19, s49 // (op 4363) 



v_lshl_add_u32 v19, s1, 6, v17 // (op 4363) 


v_mov_b32 v21, 0 // (op 4363) convert
v_mov_b32 v20, v19 // (op 4363) convert

v_mul_lo_u32 v23, v18, s29 // (op 4363) most significant: low of low * high
v_mul_hi_u32 v17, v18, s28 // (op 4363) most significant: high of low * low
v_mul_lo_u32 v22, v18, s28 // (op 4363) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 4363) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v20 // (op 4363) least significant half
v_addc_co_u32 v23, s[0:1], v23, v21, s[0:1] // (op 4363) most significant half



v_mul_lo_u32 v21, v23, 6 // (op 4363) most significant: low of high * low
v_mul_hi_u32 v17, v22, 6 // (op 4363) most significant: high of low * low
v_mul_lo_u32 v20, v22, 6 // (op 4363) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 4363) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 4363) 

v_mov_b32 v17, v18 // (op 4363) convert

s_mul_i32 s1, s33, 6 // (op 4365) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 4365) least significant
s_mul_hi_u32 s18, s32, 6 // (op 4365) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4365) most significant: sum

s_mul_i32 s19, s28, s1 // (op 4365) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 4365) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 4365) least significant
s_mul_hi_u32 s50, s28, s0 // (op 4365) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4365) most significant: sum
s_add_u32 s19, s19, s50 // (op 4365) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4365) 

s_mov_b32 s52, s30 // (op 4365) reg expression
s_mov_b32 s53, s31 // (op 4365) reg expression
s_mov_b32 s54, s0 // (op 4365) reg expression
s_mov_b32 s55, 131072 // (op 4365) call()

s_mul_hi_i32 s0, s16, s11 // (op 4408) 
v_lshrrev_b32 v18, 2, v1 // (op 4408) 
s_add_i32 s1, s0, s16 // (op 4408) 

v_lshlrev_b32 v19, 1, v18 // (op 4408) 

s_ashr_i32 s0, s1, 31 // (op 4408) 
v_and_b32 v18, -4, v1 // (op 4408) 
s_cmp_eq_i32 s11, 0 // (op 4408) 
s_mov_b32 s18, scc // (op 4408) 
v_sub_u32 v20, v1, v18 // (op 4408) 

s_cmp_eq_i32 s18, 0 // (op 4408) 
s_cselect_b32 s19, 0, -1 // (op 4408) 

s_and_b32 s18, s12, 31 // (op 4408) 
s_lshl_b32 s49, 1, s18 // (op 4408) 
s_add_i32 s50, s49, s19 // (op 4408) 


s_and_b32 s19, s0, s50 // (op 4408) 


s_add_i32 s0, s1, s19 // (op 4408) 


s_ashr_i32 s1, s0, s18 // (op 4408) 


s_lshl_b32 s0, s12, 24 // (op 4408) 
s_ashr_i32 s18, s0, 31 // (op 4408) 

s_xor_b32 s0, s1, s18 // (op 4408) 

s_sub_i32 s1, s0, s18 // (op 4408) 


s_mul_hi_i32 s0, s1, s13 // (op 4408) 
s_add_i32 s18, s0, s1 // (op 4408) 

s_ashr_i32 s0, s18, 31 // (op 4408) 
s_cmp_eq_i32 s13, 0 // (op 4408) 
s_mov_b32 s19, scc // (op 4408) 
s_cmp_eq_i32 s19, 0 // (op 4408) 
s_cselect_b32 s49, 0, -1 // (op 4408) 

s_and_b32 s19, s14, 31 // (op 4408) 
s_lshl_b32 s50, 1, s19 // (op 4408) 
s_add_i32 s51, s50, s49 // (op 4408) 


s_and_b32 s49, s0, s51 // (op 4408) 


s_add_i32 s0, s18, s49 // (op 4408) 


s_ashr_i32 s18, s0, s19 // (op 4408) 


s_lshl_b32 s0, s14, 24 // (op 4408) 
s_ashr_i32 s19, s0, 31 // (op 4408) 

s_xor_b32 s0, s18, s19 // (op 4408) 

s_sub_i32 s18, s0, s19 // (op 4408) 


s_mul_i32 s0, s1, s44 // (op 4408) 

s_sub_u32 s1, s16, s0 // (op 4408) 

s_mul_i32 s0, s18, s44 // (op 4408) 
s_add_u32 s0, s0, s1 // (op 4408) 


s_mul_hi_i32 s1, s16, s10 // (op 4408) 
s_add_i32 s18, s1, s16 // (op 4408) 

s_ashr_i32 s1, s18, 31 // (op 4408) 
s_cmp_eq_i32 s10, 0 // (op 4408) 
s_mov_b32 s19, scc // (op 4408) 
s_cmp_eq_i32 s19, 0 // (op 4408) 
s_cselect_b32 s49, 0, -1 // (op 4408) 

s_and_b32 s19, s47, 31 // (op 4408) 
s_lshl_b32 s50, 1, s19 // (op 4408) 
s_add_i32 s51, s50, s49 // (op 4408) 


s_and_b32 s49, s1, s51 // (op 4408) 


s_add_i32 s1, s18, s49 // (op 4408) 


s_ashr_i32 s18, s1, s19 // (op 4408) 


s_lshl_b32 s1, s47, 24 // (op 4408) 
s_ashr_i32 s19, s1, 31 // (op 4408) 

s_xor_b32 s1, s18, s19 // (op 4408) 

s_sub_i32 s18, s1, s19 // (op 4408) 


s_add_u32 s50, s6, 64 // (op 4408) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4408) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 4408) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 4408) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 4408) 

s_mul_i32 s1, s44, s50 // (op 4408) 

s_mul_i32 s19, s48, s1 // (op 4408) 

s_mul_i32 s1, s18, s19 // (op 4408) 

s_sub_u32 s19, s16, s1 // (op 4408) 

s_mov_b32 s1, s19 // (op 4408) 
s_mul_hi_i32 s49, s1, s15 // (op 4408) 
s_add_i32 s50, s49, s1 // (op 4408) 


s_ashr_i32 s1, s50, 31 // (op 4408) 
s_cmp_eq_i32 s15, 0 // (op 4408) 
s_mov_b32 s49, scc // (op 4408) 
s_cmp_eq_i32 s49, 0 // (op 4408) 
s_cselect_b32 s51, 0, -1 // (op 4408) 

s_and_b32 s49, s46, 31 // (op 4408) 
s_lshl_b32 s64, 1, s49 // (op 4408) 
s_add_i32 s65, s64, s51 // (op 4408) 


s_and_b32 s51, s1, s65 // (op 4408) 


s_add_i32 s1, s50, s51 // (op 4408) 


s_ashr_i32 s50, s1, s49 // (op 4408) 


s_lshl_b32 s1, s46, 24 // (op 4408) 
s_ashr_i32 s49, s1, 31 // (op 4408) 

s_xor_b32 s1, s50, s49 // (op 4408) 

s_sub_i32 s50, s1, s49 // (op 4408) 


s_mul_i32 s1, s45, s44 // (op 4408) 
s_add_u32 s64, s8, 128 // (op 4408) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 4408) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 4408) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 4408) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 4408) 

s_sub_i32 s49, s64, s1 // (op 4408) 

s_mul_i32 s51, s50, s49 // (op 4408) 


s_sub_u32 s49, s19, s51 // (op 4408) 


s_add_u32 s19, s49, s1 // (op 4408) 


s_cmp_eq_i32 s18, 0 // (op 4408) 
s_mov_b32 s1, scc // (op 4408) 

s_cmp_eq_i32 s1, 0 // (op 4408) 
s_cselect_b32 s18, s19, s0 // (op 4408) 



v_lshl_add_u32 v18, s18, 7, v19 // (op 4408) 


v_mul_lo_u32 v19, v18, s34 // (op 4408) 
v_add_u32 v19, v19, v20 // (op 4408) 


s_mul_i32 s1, s34, s9 // (op 4410) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 4410) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 4410) least significant
s_mul_hi_u32 s19, s34, s8 // (op 4410) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4410) most significant: sum
s_add_u32 s1, s1, s19 // (op 4410) most significant: sum


s_mov_b32 s64, s36 // (op 4410) reg expression
s_mov_b32 s65, s37 // (op 4410) reg expression
s_mov_b32 s66, s0 // (op 4410) reg expression
s_mov_b32 s67, 131072 // (op 4410) call()

s_mul_hi_i32 s0, s16, s11 // (op 4427) 
v_lshrrev_b32 v18, 2, v1 // (op 4427) 
s_add_i32 s1, s0, s16 // (op 4427) 

v_lshlrev_b32 v20, 1, v18 // (op 4427) 

s_ashr_i32 s0, s1, 31 // (op 4427) 
v_and_b32 v18, -4, v1 // (op 4427) 
s_cmp_eq_i32 s11, 0 // (op 4427) 
s_mov_b32 s18, scc // (op 4427) 
v_sub_u32 v21, v1, v18 // (op 4427) 

s_cmp_eq_i32 s18, 0 // (op 4427) 
s_cselect_b32 s19, 0, -1 // (op 4427) 

s_and_b32 s18, s12, 31 // (op 4427) 
s_lshl_b32 s49, 1, s18 // (op 4427) 
s_add_i32 s50, s49, s19 // (op 4427) 


s_and_b32 s19, s0, s50 // (op 4427) 


s_add_i32 s0, s1, s19 // (op 4427) 


s_ashr_i32 s1, s0, s18 // (op 4427) 


s_lshl_b32 s0, s12, 24 // (op 4427) 
s_ashr_i32 s18, s0, 31 // (op 4427) 

s_xor_b32 s0, s1, s18 // (op 4427) 

s_sub_i32 s1, s0, s18 // (op 4427) 


s_mul_hi_i32 s0, s1, s13 // (op 4427) 
s_add_i32 s18, s0, s1 // (op 4427) 

s_ashr_i32 s0, s18, 31 // (op 4427) 
s_cmp_eq_i32 s13, 0 // (op 4427) 
s_mov_b32 s19, scc // (op 4427) 
s_cmp_eq_i32 s19, 0 // (op 4427) 
s_cselect_b32 s49, 0, -1 // (op 4427) 

s_and_b32 s19, s14, 31 // (op 4427) 
s_lshl_b32 s50, 1, s19 // (op 4427) 
s_add_i32 s51, s50, s49 // (op 4427) 


s_and_b32 s49, s0, s51 // (op 4427) 


s_add_i32 s0, s18, s49 // (op 4427) 


s_ashr_i32 s18, s0, s19 // (op 4427) 


s_lshl_b32 s0, s14, 24 // (op 4427) 
s_ashr_i32 s19, s0, 31 // (op 4427) 

s_xor_b32 s0, s18, s19 // (op 4427) 

s_sub_i32 s18, s0, s19 // (op 4427) 


s_mul_i32 s0, s1, s44 // (op 4427) 

s_sub_u32 s1, s16, s0 // (op 4427) 

s_mul_i32 s0, s18, s44 // (op 4427) 
s_add_u32 s0, s0, s1 // (op 4427) 


s_mul_hi_i32 s1, s16, s10 // (op 4427) 
s_add_i32 s18, s1, s16 // (op 4427) 

s_ashr_i32 s1, s18, 31 // (op 4427) 
s_cmp_eq_i32 s10, 0 // (op 4427) 
s_mov_b32 s19, scc // (op 4427) 
s_cmp_eq_i32 s19, 0 // (op 4427) 
s_cselect_b32 s49, 0, -1 // (op 4427) 

s_and_b32 s19, s47, 31 // (op 4427) 
s_lshl_b32 s50, 1, s19 // (op 4427) 
s_add_i32 s51, s50, s49 // (op 4427) 


s_and_b32 s49, s1, s51 // (op 4427) 


s_add_i32 s1, s18, s49 // (op 4427) 


s_ashr_i32 s18, s1, s19 // (op 4427) 


s_lshl_b32 s1, s47, 24 // (op 4427) 
s_ashr_i32 s19, s1, 31 // (op 4427) 

s_xor_b32 s1, s18, s19 // (op 4427) 

s_sub_i32 s18, s1, s19 // (op 4427) 


s_add_u32 s50, s6, 64 // (op 4427) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4427) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 4427) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 4427) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 4427) 

s_mul_i32 s1, s44, s50 // (op 4427) 

s_mul_i32 s19, s48, s1 // (op 4427) 

s_mul_i32 s1, s18, s19 // (op 4427) 

s_sub_u32 s19, s16, s1 // (op 4427) 

s_mov_b32 s1, s19 // (op 4427) 
s_mul_hi_i32 s49, s1, s15 // (op 4427) 
s_add_i32 s50, s49, s1 // (op 4427) 


s_ashr_i32 s1, s50, 31 // (op 4427) 
s_cmp_eq_i32 s15, 0 // (op 4427) 
s_mov_b32 s49, scc // (op 4427) 
s_cmp_eq_i32 s49, 0 // (op 4427) 
s_cselect_b32 s51, 0, -1 // (op 4427) 

s_and_b32 s49, s46, 31 // (op 4427) 
s_lshl_b32 s68, 1, s49 // (op 4427) 
s_add_i32 s69, s68, s51 // (op 4427) 


s_and_b32 s51, s1, s69 // (op 4427) 


s_add_i32 s1, s50, s51 // (op 4427) 


s_ashr_i32 s50, s1, s49 // (op 4427) 


s_lshl_b32 s1, s46, 24 // (op 4427) 
s_ashr_i32 s49, s1, 31 // (op 4427) 

s_xor_b32 s1, s50, s49 // (op 4427) 

s_sub_i32 s50, s1, s49 // (op 4427) 


s_mul_i32 s1, s45, s44 // (op 4427) 
s_add_u32 s68, s8, 128 // (op 4427) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 4427) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 4427) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 4427) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 7 // (op 4427) 

s_sub_i32 s49, s68, s1 // (op 4427) 

s_mul_i32 s51, s50, s49 // (op 4427) 


s_sub_u32 s49, s19, s51 // (op 4427) 


s_add_u32 s19, s49, s1 // (op 4427) 


s_cmp_eq_i32 s18, 0 // (op 4427) 
s_mov_b32 s1, scc // (op 4427) 

s_cmp_eq_i32 s1, 0 // (op 4427) 
s_cselect_b32 s18, s19, s0 // (op 4427) 



v_lshl_add_u32 v18, s18, 7, v20 // (op 4427) 


v_mul_lo_u32 v20, v18, s34 // (op 4427) 
v_add_u32 v20, v20, v21 // (op 4427) 


s_mul_i32 s1, s34, s9 // (op 4429) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 4429) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 4429) least significant
s_mul_hi_u32 s19, s34, s8 // (op 4429) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4429) most significant: sum
s_add_u32 s1, s1, s19 // (op 4429) most significant: sum


s_mov_b32 s64, s36 // (op 4429) reg expression
s_mov_b32 s65, s37 // (op 4429) reg expression
s_mov_b32 s66, s0 // (op 4429) reg expression
s_mov_b32 s67, 131072 // (op 4429) call()

s_mul_hi_i32 s0, s16, s11 // (op 4472) 
v_and_b32 v18, -64, v1 // (op 4472) 
s_add_i32 s1, s0, s16 // (op 4472) 

v_sub_u32 v21, v1, v18 // (op 4472) 

s_ashr_i32 s0, s1, 31 // (op 4472) 
v_lshrrev_b32 v18, 6, v1 // (op 4472) 
s_cmp_eq_i32 s11, 0 // (op 4472) 
s_mov_b32 s18, scc // (op 4472) 
s_cmp_eq_i32 s18, 0 // (op 4472) 
s_cselect_b32 s19, 0, -1 // (op 4472) 

s_and_b32 s18, s12, 31 // (op 4472) 
s_lshl_b32 s49, 1, s18 // (op 4472) 
s_add_i32 s50, s49, s19 // (op 4472) 


s_and_b32 s19, s0, s50 // (op 4472) 


s_add_i32 s0, s1, s19 // (op 4472) 


s_ashr_i32 s1, s0, s18 // (op 4472) 


s_lshl_b32 s0, s12, 24 // (op 4472) 
s_ashr_i32 s18, s0, 31 // (op 4472) 

s_xor_b32 s0, s1, s18 // (op 4472) 

s_sub_i32 s1, s0, s18 // (op 4472) 


s_mul_hi_i32 s0, s1, s13 // (op 4472) 
s_add_i32 s18, s0, s1 // (op 4472) 

s_ashr_i32 s0, s18, 31 // (op 4472) 
s_cmp_eq_i32 s13, 0 // (op 4472) 
s_mov_b32 s19, scc // (op 4472) 
s_cmp_eq_i32 s19, 0 // (op 4472) 
s_cselect_b32 s49, 0, -1 // (op 4472) 

s_and_b32 s19, s14, 31 // (op 4472) 
s_lshl_b32 s50, 1, s19 // (op 4472) 
s_add_i32 s51, s50, s49 // (op 4472) 


s_and_b32 s49, s0, s51 // (op 4472) 


s_add_i32 s0, s18, s49 // (op 4472) 


s_ashr_i32 s18, s0, s19 // (op 4472) 


s_lshl_b32 s0, s14, 24 // (op 4472) 
s_ashr_i32 s19, s0, 31 // (op 4472) 

s_xor_b32 s0, s18, s19 // (op 4472) 

s_sub_i32 s18, s0, s19 // (op 4472) 


s_add_u32 s50, s6, 64 // (op 4472) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4472) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 4472) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 4472) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 4472) 

s_mul_i32 s0, s18, s50 // (op 4472) 

s_sub_i32 s49, s1, s0 // (op 4472) 


s_mul_hi_i32 s0, s16, s10 // (op 4472) 
s_add_i32 s1, s0, s16 // (op 4472) 

s_ashr_i32 s0, s1, 31 // (op 4472) 
s_cmp_eq_i32 s10, 0 // (op 4472) 
s_mov_b32 s18, scc // (op 4472) 
s_cmp_eq_i32 s18, 0 // (op 4472) 
s_cselect_b32 s19, 0, -1 // (op 4472) 

s_and_b32 s18, s47, 31 // (op 4472) 
s_lshl_b32 s68, 1, s18 // (op 4472) 
s_add_i32 s69, s68, s19 // (op 4472) 


s_and_b32 s19, s0, s69 // (op 4472) 


s_add_i32 s0, s1, s19 // (op 4472) 


s_ashr_i32 s1, s0, s18 // (op 4472) 


s_lshl_b32 s0, s47, 24 // (op 4472) 
s_ashr_i32 s18, s0, 31 // (op 4472) 

s_xor_b32 s0, s1, s18 // (op 4472) 

s_sub_i32 s1, s0, s18 // (op 4472) 


s_mul_i32 s0, s44, s50 // (op 4472) 

s_mul_i32 s18, s48, s0 // (op 4472) 

s_mul_i32 s0, s1, s18 // (op 4472) 

s_sub_u32 s18, s16, s0 // (op 4472) 

s_mul_hi_i32 s0, s18, s15 // (op 4472) 
s_add_i32 s19, s0, s18 // (op 4472) 


s_ashr_i32 s0, s19, 31 // (op 4472) 
s_cmp_eq_i32 s15, 0 // (op 4472) 
s_mov_b32 s18, scc // (op 4472) 
s_cmp_eq_i32 s18, 0 // (op 4472) 
s_cselect_b32 s50, 0, -1 // (op 4472) 

s_and_b32 s18, s46, 31 // (op 4472) 
s_lshl_b32 s51, 1, s18 // (op 4472) 
s_add_i32 s68, s51, s50 // (op 4472) 


s_and_b32 s50, s0, s68 // (op 4472) 


s_add_i32 s0, s19, s50 // (op 4472) 


s_ashr_i32 s19, s0, s18 // (op 4472) 


s_lshl_b32 s0, s46, 24 // (op 4472) 
s_ashr_i32 s18, s0, 31 // (op 4472) 

s_xor_b32 s0, s19, s18 // (op 4472) 

s_sub_i32 s19, s0, s18 // (op 4472) 


s_cmp_eq_i32 s1, 0 // (op 4472) 
s_mov_b32 s0, scc // (op 4472) 

s_cmp_eq_i32 s0, 0 // (op 4472) 
s_cselect_b32 s1, s19, s49 // (op 4472) 



v_lshl_add_u32 v22, s1, 6, v21 // (op 4472) 


v_mul_lo_u32 v21, v18, s38 // (op 4472) 
v_add_u32 v21, v21, v22 // (op 4472) 


s_mul_i32 s1, s38, s43 // (op 4474) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 4474) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 4474) least significant
s_mul_hi_u32 s19, s38, s42 // (op 4474) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4474) most significant: sum
s_add_u32 s1, s1, s19 // (op 4474) most significant: sum


s_mov_b32 s68, s40 // (op 4474) reg expression
s_mov_b32 s69, s41 // (op 4474) reg expression
s_mov_b32 s70, s0 // (op 4474) reg expression
s_mov_b32 s71, 131072 // (op 4474) call()

s_mul_hi_i32 s0, s16, s11 // (op 4491) 
v_and_b32 v18, -64, v1 // (op 4491) 
s_add_i32 s1, s0, s16 // (op 4491) 

v_sub_u32 v22, v1, v18 // (op 4491) 

s_ashr_i32 s0, s1, 31 // (op 4491) 
v_lshrrev_b32 v18, 6, v1 // (op 4491) 
s_cmp_eq_i32 s11, 0 // (op 4491) 
s_mov_b32 s18, scc // (op 4491) 
s_cmp_eq_i32 s18, 0 // (op 4491) 
s_cselect_b32 s19, 0, -1 // (op 4491) 

s_and_b32 s18, s12, 31 // (op 4491) 
s_lshl_b32 s49, 1, s18 // (op 4491) 
s_add_i32 s50, s49, s19 // (op 4491) 


s_and_b32 s19, s0, s50 // (op 4491) 


s_add_i32 s0, s1, s19 // (op 4491) 


s_ashr_i32 s1, s0, s18 // (op 4491) 


s_lshl_b32 s0, s12, 24 // (op 4491) 
s_ashr_i32 s18, s0, 31 // (op 4491) 

s_xor_b32 s0, s1, s18 // (op 4491) 

s_sub_i32 s1, s0, s18 // (op 4491) 


s_mul_hi_i32 s0, s1, s13 // (op 4491) 
s_add_i32 s18, s0, s1 // (op 4491) 

s_ashr_i32 s0, s18, 31 // (op 4491) 
s_cmp_eq_i32 s13, 0 // (op 4491) 
s_mov_b32 s19, scc // (op 4491) 
s_cmp_eq_i32 s19, 0 // (op 4491) 
s_cselect_b32 s49, 0, -1 // (op 4491) 

s_and_b32 s19, s14, 31 // (op 4491) 
s_lshl_b32 s50, 1, s19 // (op 4491) 
s_add_i32 s51, s50, s49 // (op 4491) 


s_and_b32 s49, s0, s51 // (op 4491) 


s_add_i32 s0, s18, s49 // (op 4491) 


s_ashr_i32 s18, s0, s19 // (op 4491) 


s_lshl_b32 s0, s14, 24 // (op 4491) 
s_ashr_i32 s19, s0, 31 // (op 4491) 

s_xor_b32 s0, s18, s19 // (op 4491) 

s_sub_i32 s18, s0, s19 // (op 4491) 


s_add_u32 s50, s6, 64 // (op 4491) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4491) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4491) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4491) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 4491) 

s_mul_i32 s0, s18, s50 // (op 4491) 

s_sub_i32 s49, s1, s0 // (op 4491) 


s_mul_hi_i32 s0, s16, s10 // (op 4491) 
s_add_i32 s1, s0, s16 // (op 4491) 

s_ashr_i32 s0, s1, 31 // (op 4491) 
s_cmp_eq_i32 s10, 0 // (op 4491) 
s_mov_b32 s18, scc // (op 4491) 
s_cmp_eq_i32 s18, 0 // (op 4491) 
s_cselect_b32 s19, 0, -1 // (op 4491) 

s_and_b32 s18, s47, 31 // (op 4491) 
s_lshl_b32 s72, 1, s18 // (op 4491) 
s_add_i32 s73, s72, s19 // (op 4491) 


s_and_b32 s19, s0, s73 // (op 4491) 


s_add_i32 s0, s1, s19 // (op 4491) 


s_ashr_i32 s1, s0, s18 // (op 4491) 


s_lshl_b32 s0, s47, 24 // (op 4491) 
s_ashr_i32 s18, s0, 31 // (op 4491) 

s_xor_b32 s0, s1, s18 // (op 4491) 

s_sub_i32 s1, s0, s18 // (op 4491) 


s_mul_i32 s0, s44, s50 // (op 4491) 

s_mul_i32 s18, s48, s0 // (op 4491) 

s_mul_i32 s0, s1, s18 // (op 4491) 

s_sub_u32 s18, s16, s0 // (op 4491) 

s_mul_hi_i32 s0, s18, s15 // (op 4491) 
s_add_i32 s19, s0, s18 // (op 4491) 


s_ashr_i32 s0, s19, 31 // (op 4491) 
s_cmp_eq_i32 s15, 0 // (op 4491) 
s_mov_b32 s18, scc // (op 4491) 
s_cmp_eq_i32 s18, 0 // (op 4491) 
s_cselect_b32 s50, 0, -1 // (op 4491) 

s_and_b32 s18, s46, 31 // (op 4491) 
s_lshl_b32 s51, 1, s18 // (op 4491) 
s_add_i32 s72, s51, s50 // (op 4491) 


s_and_b32 s50, s0, s72 // (op 4491) 


s_add_i32 s0, s19, s50 // (op 4491) 


s_ashr_i32 s19, s0, s18 // (op 4491) 


s_lshl_b32 s0, s46, 24 // (op 4491) 
s_ashr_i32 s18, s0, 31 // (op 4491) 

s_xor_b32 s0, s19, s18 // (op 4491) 

s_sub_i32 s19, s0, s18 // (op 4491) 


s_cmp_eq_i32 s1, 0 // (op 4491) 
s_mov_b32 s0, scc // (op 4491) 

s_cmp_eq_i32 s0, 0 // (op 4491) 
s_cselect_b32 s1, s19, s49 // (op 4491) 



v_lshl_add_u32 v23, s1, 6, v22 // (op 4491) 


v_mul_lo_u32 v22, v18, s38 // (op 4491) 
v_add_u32 v22, v22, v23 // (op 4491) 


s_mul_i32 s1, s38, s43 // (op 4493) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 4493) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 4493) least significant
s_mul_hi_u32 s19, s38, s42 // (op 4493) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4493) most significant: sum
s_add_u32 s1, s1, s19 // (op 4493) most significant: sum


s_mov_b32 s68, s40 // (op 4493) reg expression
s_mov_b32 s69, s41 // (op 4493) reg expression
s_mov_b32 s70, s0 // (op 4493) reg expression
s_mov_b32 s71, 131072 // (op 4493) call()



v_mov_b32 v18, v16 // (op 4024) reg expression

buffer_load_dwordx3 v[24:26], v18, s[52:55], 0 offen // (op 4024) Load value
buffer_load_dwordx3 v[28:30], v18, s[52:55], 0 offen offset:12 // (op 4024) Load value



v_mov_b32 v18, v21 // (op 4027) reg expression

buffer_load_ubyte v23, v18, s[68:71], 0 offen offset: 0 // (op 4027) Load value




buffer_load_dwordx4 v[32:35], v14, s[20:23], 0 offen // (op 4030) Load value
buffer_load_dwordx2 v[36:37], v14, s[20:23], 0 offen offset:16 // (op 4030) Load value
buffer_load_dwordx4 v[38:41], v14, s[20:23], 0 offen offset:24 // (op 4030) Load value
buffer_load_dwordx2 v[42:43], v14, s[20:23], 0 offen offset:40 // (op 4030) Load value

v_mov_b64 v[44:45], s[34:35] // (op 4033) convert

v_mov_b32 v18, v19 // (op 4033) 
buffer_load_ubyte v46, v18, s[64:67], 0 offen offset: 0 // (op 4033) Load value
v_add_u32 v18, v18, v44 // (op 4033) 
buffer_load_ubyte v47, v18, s[64:67], 0 offen offset: 0 // (op 4033) Load value




s_lshl_b64 s[0:1], s[28:29], 7 // (op 4036) 
s_mul_i32 s19, s1, 6 // (op 4036) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 4036) least significant
s_mul_hi_u32 s49, s0, 6 // (op 4036) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4036) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 4036) 

v_add_u32 v18, v16, s0 // (op 4036) 


buffer_load_dwordx3 v[48:50], v18, s[52:55], 0 offen // (op 4036) Load value
buffer_load_dwordx3 v[52:54], v18, s[52:55], 0 offen offset:12 // (op 4036) Load value




s_lshl_b32 s0, s38, 2 // (op 4039) 
v_add_u32 v16, v21, s0 // (op 4039) 


buffer_load_ubyte v18, v16, s[68:71], 0 offen offset: 0 // (op 4039) Load value





buffer_load_dwordx4 v[56:59], v14, s[20:23], 0 offen offset:96 // (op 4042) Load value
buffer_load_dwordx2 v[60:61], v14, s[20:23], 0 offen offset:112 // (op 4042) Load value
buffer_load_dwordx4 v[62:65], v14, s[20:23], 0 offen offset:120 // (op 4042) Load value
buffer_load_dwordx2 v[66:67], v14, s[20:23], 0 offen offset:136 // (op 4042) Load value


v_mov_b64 v[44:45], s[34:35] // (op 4045) convert

v_mov_b32 v14, v19 // (op 4045) 
buffer_load_ubyte v68, v14, s[64:67], 0 offen offset:4 // (op 4045) Load value
v_add_u32 v14, v14, v44 // (op 4045) 
buffer_load_ubyte v69, v14, s[64:67], 0 offen offset:4 // (op 4045) Load value



v_and_b32 v14, -2, v1 // (op 4621) 
v_sub_u32 v16, v1, v14 // (op 4621) 

v_lshlrev_b32 v14, 5, v16 // (op 4621) 

v_lshrrev_b32 v16, 1, v1 // (op 4621) 
v_lshl_add_u32 v19, v16, 6, v14 // (op 4621) 


v_mul_lo_u32 v14, v19, 6 // (op 4621) 
v_lshrrev_b32 v16, 3, v14 // (op 4621) 

v_lshrrev_b32 v14, 4, v19 // (op 4621) 

v_lshlrev_b32 v19, 2, v14 // (op 4621) 

v_add_u32 v14, v16, v19 // (op 4621) 





s_waitcnt vmcnt(17)
ds_write_b96 v14, v[24:26] // (op 4047) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v14, v[28:30] offset:16 // (op 4047) Store local data 

v_and_b32 v14, -64, v1 // (op 4737) 
v_sub_u32 v16, v1, v14 // (op 4737) 

v_and_b32 v14, 4294967232, v1 // (op 4737) 
v_add_u32 v19, v14, v16 // (op 4737) 





s_waitcnt vmcnt(15)
ds_write_b8 v19, v23 offset:8192 // (op 4048) Store local data 

v_and_b32 v14, -2, v1 // (op 4563) 
v_sub_u32 v16, v1, v14 // (op 4563) 

v_lshlrev_b32 v14, 6, v16 // (op 4563) 

v_lshrrev_b32 v16, 1, v1 // (op 4563) 
v_lshl_add_u32 v19, v16, 7, v14 // (op 4563) 


v_mul_lo_u32 v14, v19, 6 // (op 4563) 

v_lshrrev_b32 v16, 3, v14 // (op 4563) 




s_waitcnt vmcnt(14)
ds_write_b128 v16, v[32:35] offset:8448 // (op 4049) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v16, v[36:37] offset:8464 // (op 4049) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v16, v[38:41] offset:8472 // (op 4049) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v16, v[42:43] offset:8488 // (op 4049) Store local data 

v_and_b32 v14, -4, v1 // (op 4679) 
v_sub_u32 v16, v1, v14 // (op 4679) 

v_lshrrev_b32 v14, 2, v1 // (op 4679) 
v_lshl_add_u32 v19, v14, 3, v16 // (op 4679) 





s_waitcnt vmcnt(10)
ds_write_b8 v19, v46 offset:20736 // (op 4050) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v19, v47 offset:20740 // (op 4050) Store local data 

v_and_b32 v14, -2, v1 // (op 4534) 
v_sub_u32 v16, v1, v14 // (op 4534) 

v_lshlrev_b32 v14, 6, v16 // (op 4534) 

v_lshrrev_b32 v16, 1, v1 // (op 4534) 
v_lshl_add_u32 v19, v16, 7, v14 // (op 4534) 


v_mul_lo_u32 v14, v19, 6 // (op 4534) 

v_lshrrev_b32 v16, 3, v14 // (op 4534) 

v_and_b32 v14, -2, v1 // (op 4592) 
v_sub_u32 v19, v1, v14 // (op 4592) 

v_lshlrev_b32 v14, 5, v19 // (op 4592) 

v_lshrrev_b32 v19, 1, v1 // (op 4592) 
v_lshl_add_u32 v21, v19, 6, v14 // (op 4592) 


v_mul_lo_u32 v14, v21, 6 // (op 4592) 
v_lshrrev_b32 v19, 3, v14 // (op 4592) 

v_lshrrev_b32 v14, 4, v21 // (op 4592) 

v_lshlrev_b32 v21, 2, v14 // (op 4592) 

v_add_u32 v14, v19, v21 // (op 4592) 


v_and_b32 v19, -4, v1 // (op 4650) 
v_sub_u32 v21, v1, v19 // (op 4650) 

v_lshrrev_b32 v19, 2, v1 // (op 4650) 
v_lshl_add_u32 v27, v19, 3, v21 // (op 4650) 


v_and_b32 v19, -64, v1 // (op 4708) 
v_sub_u32 v21, v1, v19 // (op 4708) 

v_and_b32 v19, 4294967232, v1 // (op 4708) 
v_add_u32 v31, v19, v21 // (op 4708) 


v_and_b32 v19, -2, v1 // (op 4789) 
v_sub_u32 v21, v1, v19 // (op 4789) 

v_lshlrev_b32 v19, 6, v21 // (op 4789) 

v_lshrrev_b32 v21, 1, v1 // (op 4789) 
v_lshl_add_u32 v51, v21, 7, v19 // (op 4789) 


v_mul_lo_u32 v19, v51, 6 // (op 4789) 

v_lshrrev_b32 v21, 3, v19 // (op 4789) 

v_and_b32 v19, -2, v1 // (op 4818) 
v_sub_u32 v51, v1, v19 // (op 4818) 

v_lshlrev_b32 v19, 5, v51 // (op 4818) 

v_lshrrev_b32 v51, 1, v1 // (op 4818) 
v_lshl_add_u32 v55, v51, 6, v19 // (op 4818) 


v_mul_lo_u32 v19, v55, 6 // (op 4818) 
v_lshrrev_b32 v51, 3, v19 // (op 4818) 

v_lshrrev_b32 v19, 4, v55 // (op 4818) 

v_lshlrev_b32 v55, 2, v19 // (op 4818) 

v_add_u32 v19, v51, v55 // (op 4818) 


v_and_b32 v51, -4, v1 // (op 4847) 
v_sub_u32 v55, v1, v51 // (op 4847) 

v_lshrrev_b32 v51, 2, v1 // (op 4847) 
v_lshl_add_u32 v44, v51, 3, v55 // (op 4847) 


v_and_b32 v45, -64, v1 // (op 4876) 
v_sub_u32 v51, v1, v45 // (op 4876) 

v_and_b32 v45, 4294967232, v1 // (op 4876) 
v_add_u32 v55, v45, v51 // (op 4876) 


s_mov_b32 s0, 0 // (op 1361) call()
s_ashr_i64 s[18:19], s[32:33], 7 // (op 1360) 
s_mov_b32 s1, s18 // (op 1360) convert

s_and_b32 s18, s1, 4294967294 // (op 1360) 

s_cmp_lt_u32 s0, s18 // (op 1360) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1360 // (op 1360) Condition: Top (jump to RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1360 if false)
RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1360:


s_barrier  // (op 4956) 
s_add_u32 s0, s0, 2 // (op 1362) 



v_mov_b32 v45, v17 // (op 10) 
s_lshl_b64 s[18:19], s[28:29], 7 // (op 10) 
s_mul_i32 s51, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 10) least significant
s_mul_hi_u32 s1, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v51, v45, s1 // (op 10) 


ds_read_b128 v[70:73], v3 offset:8448 // (op 70) Load local data 
ds_read_b64 v[74:75], v3 offset:8464 // (op 70) Load local data 
buffer_load_dwordx3 v[24:26], v51, s[52:55], 0 offen // (op 10) Load value
s_barrier  // (op 4969) 
buffer_load_dwordx3 v[28:30], v51, s[52:55], 0 offen offset:12 // (op 10) Load value






v_mov_b32 v51, v22 // (op 25) 
s_lshl_b32 s1, s38, 3 // (op 25) 
v_add_u32 v76, v51, s1 // (op 25) 


buffer_load_ubyte v23, v76, s[68:71], 0 offen offset: 0 // (op 25) Load value

ds_read_b96_tr_b6 v[76:78], v0 // (op 75) Transpose load local data 


ds_read_b96_tr_b6 v[79:81], v0 offset:1024 // (op 75) Transpose load local data 
v_mov_b32 v82, v15 // (op 4) 

s_barrier  // (op 4983) 
buffer_load_dwordx4 v[32:35], v82, s[20:23], 0 offen offset:192 // (op 4) Load value



ds_read_u8 v83, v4 offset:20736 // (op 80) Load local data 
buffer_load_dwordx2 v[36:37], v82, s[20:23], 0 offen offset:208 // (op 4) Load value
s_barrier  // (op 4997) 
buffer_load_dwordx4 v[38:41], v82, s[20:23], 0 offen offset:216 // (op 4) Load value


ds_read_u8 v84, v5 offset:8192 // (op 85) Load local data 
buffer_load_dwordx2 v[42:43], v82, s[20:23], 0 offen offset:232 // (op 4) Load value




v_mov_b64 v[86:87], s[34:35] // (op 17) convert

v_mov_b32 v85, v20 // (op 17) 
v_mov_b32 v88, v85 // (op 17) 
buffer_load_ubyte v46, v88, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v88, v88, v86 // (op 17) 
buffer_load_ubyte v47, v88, s[64:67], 0 offen offset:8 // (op 17) Load value


ds_read_b96_tr_b6 v[86:88], v0 offset:32 // (op 371) Transpose load local data 
s_barrier  // (op 5020) 
ds_read_b96_tr_b6 v[89:91], v0 offset:1056 // (op 371) Transpose load local data 





ds_read_u8 v92, v5 offset:8224 // (op 375) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:75], v[76:81], a[0:3], v83, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 88) 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:75], v[86:91], a[4:7], v83, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 376) 
ds_read_b128 v[94:97], v3 offset:11520 // (op 622) Load local data 


ds_read_b64 v[98:99], v3 offset:11536 // (op 622) Load local data 
s_waitcnt vmcnt(17)
ds_write_b96 v19, v[48:50] offset:21248 // (op 253) Store local data 



ds_read_u8 v83, v4 offset:20864 // (op 625) Load local data 
s_waitcnt vmcnt(16)
ds_write_b96 v19, v[52:54] offset:21264 // (op 253) Store local data 



ds_read_b128 v[70:73], v3 offset:14592 // (op 1008) Load local data 
s_barrier  // (op 5032) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[94:99], v[76:81], a[8:11], v83, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 673) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:99], v[86:91], a[12:15], v83, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 627) 
ds_read_b64 v[74:75], v3 offset:14608 // (op 1008) Load local data 





ds_read_u8 v83, v4 offset:20992 // (op 1011) Load local data 
s_waitcnt vmcnt(15)
ds_write_b8 v55, v18 offset:29440 // (op 255) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:75], v[76:81], a[16:19], v83, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1059) 
s_barrier  // (op 5014) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:75], v[86:91], a[20:23], v83, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1013) 





ds_read_b128 v[70:73], v3 offset:17664 // (op 1394) Load local data 
ds_read_b64 v[74:75], v3 offset:17680 // (op 1394) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v21, v[56:59] offset:29696 // (op 243) Store local data 



ds_read_u8 v83, v4 offset:21120 // (op 1397) Load local data 
s_waitcnt vmcnt(13)
ds_write_b64 v21, v[60:61] offset:29712 // (op 243) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:75], v[76:81], a[24:27], v83, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1445) 
s_waitcnt vmcnt(12)
ds_write_b128 v21, v[62:65] offset:29720 // (op 243) Store local data 


s_waitcnt vmcnt(11)
ds_write_b64 v21, v[66:67] offset:29736 // (op 243) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:75], v[86:91], a[28:31], v83, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1399) 




s_barrier  // (op 5026) 



s_waitcnt vmcnt(10)
ds_write_b8 v44, v68 offset:41984 // (op 254) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v44, v69 offset:41988 // (op 254) Store local data 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 256) 
s_mul_i32 s51, s19, 6 // (op 256) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 256) least significant
s_mul_hi_u32 s1, s18, 6 // (op 256) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 256) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 256) 

s_mul_i32 s1, 3, s18 // (op 256) 

v_add_u32 v70, v45, s1 // (op 256) 


s_waitcnt lgkmcnt(2)
s_barrier  // (op 5011) 
buffer_load_dwordx3 v[48:50], v70, s[52:55], 0 offen // (op 256) Load value



ds_read_b128 v[76:79], v7 offset:29696 // (op 245) Load local data 
ds_read_b64 v[80:81], v7 offset:29712 // (op 245) Load local data 
buffer_load_dwordx3 v[52:54], v70, s[52:55], 0 offen offset:12 // (op 256) Load value


s_barrier  // (op 5017) 





s_lshl_b32 s1, s38, 2 // (op 258) 
s_mul_i32 s18, 3, s1 // (op 258) 

v_add_u32 v45, v51, s18 // (op 258) 


buffer_load_ubyte v18, v45, s[68:71], 0 offen offset: 0 // (op 258) Load value

s_lshl_b64 s[18:19], s[28:29], 7 // (op 4355) 

s_mul_i32 s51, s19, 6 // (op 4355) most significant: low of high * low



s_mul_i32 s50, s18, 6 // (op 4355) least significant
s_mul_hi_u32 s1, s18, 6 // (op 4355) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 4355) most significant: sum


s_lshl_b32 s1, s38, 3 // (op 4483) 
v_add_u32 v22, v22, s1 // (op 4483) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 4355) 

s_lshl_b32 s1, s18, 1 // (op 4355) 

v_add_u32 v17, v17, s1 // (op 4355) 

ds_read_b96_tr_b6 v[70:72], v6 offset:21248 // (op 247) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v6 offset:22272 // (op 247) Transpose load local data 
buffer_load_dwordx4 v[56:59], v82, s[20:23], 0 offen offset:288 // (op 242) Load value
buffer_load_dwordx2 v[60:61], v82, s[20:23], 0 offen offset:304 // (op 242) Load value
buffer_load_dwordx4 v[62:65], v82, s[20:23], 0 offen offset:312 // (op 242) Load value
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5023) 
buffer_load_dwordx2 v[66:67], v82, s[20:23], 0 offen offset:328 // (op 242) Load value




ds_read_u8 v45, v8 offset:41984 // (op 249) Load local data 

s_barrier  // (op 5029) 
v_add_u32 v15, 192, v15 // (op 4291) 


ds_read_u8 v51, v9 offset:29440 // (op 251) Load local data 
v_mov_b64 v[82:83], s[34:35] // (op 257) convert




v_mov_b32 v84, v85 // (op 257) 
buffer_load_ubyte v68, v84, s[64:67], 0 offen offset:12 // (op 257) Load value
v_add_u32 v84, v84, v82 // (op 257) 
buffer_load_ubyte v69, v84, s[64:67], 0 offen offset:12 // (op 257) Load value


ds_read_b96_tr_b6 v[86:88], v6 offset:21280 // (op 381) Transpose load local data 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:81], v[70:75], a[0:3], v45, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 252) 
s_barrier  // (op 4972) 
ds_read_b96_tr_b6 v[89:91], v6 offset:22304 // (op 381) Transpose load local data 



v_add_u32 v20, 8, v20 // (op 4419) 


s_barrier  // (op 5089) 
ds_read_u8 v82, v9 offset:29472 // (op 385) Load local data 
s_waitcnt vmcnt(17)
ds_write_b96 v14, v[24:26] // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v14, v[28:30] offset:16 // (op 13) Store local data 



s_barrier  // (op 5093) 
s_barrier  // (op 5000) 
ds_read_b128 v[92:95], v7 offset:32768 // (op 630) Load local data 



ds_read_b64 v[96:97], v7 offset:32784 // (op 630) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:81], v[86:91], a[4:7], v45, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 377) 





ds_read_u8 v45, v8 offset:42112 // (op 633) Load local data 
s_waitcnt vmcnt(15)
ds_write_b8 v31, v23 offset:8192 // (op 30) Store local data 



ds_read_b128 v[76:79], v7 offset:35840 // (op 1016) Load local data 
s_barrier  // (op 4959) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:97], v[70:75], a[8:11], v45, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 674) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:97], v[86:91], a[12:15], v45, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 628) 
ds_read_b64 v[80:81], v7 offset:35856 // (op 1016) Load local data 





ds_read_u8 v45, v8 offset:42240 // (op 1019) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v16, v[32:35] offset:8448 // (op 6) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:81], v[70:75], a[16:19], v45, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1060) 
ds_read_b128 v[92:95], v7 offset:38912 // (op 1402) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:81], v[86:91], a[20:23], v45, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1014) 
ds_read_b64 v[96:97], v7 offset:38928 // (op 1402) Load local data 


s_waitcnt vmcnt(13)
ds_write_b64 v16, v[36:37] offset:8464 // (op 6) Store local data 



s_barrier  // (op 5125) 
ds_read_u8 v45, v8 offset:42368 // (op 1405) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v16, v[38:41] offset:8472 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b64 v16, v[42:43] offset:8488 // (op 6) Store local data 
s_barrier  // (op 5129) 
s_barrier  // (op 4986) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:97], v[70:75], a[24:27], v45, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1446) 





s_waitcnt vmcnt(10)
ds_write_b8 v27, v46 offset:20736 // (op 21) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:97], v[86:91], a[28:31], v45, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1400) 
s_waitcnt vmcnt(9)
ds_write_b8 v27, v47 offset:20740 // (op 21) Store local data 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 1360) 
s_mov_b32 s1, s18 // (op 1360) convert

s_and_b32 s18, s1, 4294967294 // (op 1360) 

s_cmp_lt_u32 s0, s18 // (op 1360) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1360 // (op 1360) Condition: Bottom (jump to RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_1360 if true)
RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_1360:

















s_mul_hi_i32 s0, s16, s11 // (op 4247) 
v_lshrrev_b32 v51, 1, v1 // (op 4247) 
s_add_i32 s1, s0, s16 // (op 4247) 

v_and_b32 v55, -2, v1 // (op 4247) 
s_ashr_i32 s0, s1, 31 // (op 4247) 
v_sub_u32 v14, v1, v55 // (op 4247) 

s_cmp_eq_i32 s11, 0 // (op 4247) 
s_mov_b32 s18, scc // (op 4247) 
v_lshlrev_b32 v55, 6, v14 // (op 4247) 

s_cmp_eq_i32 s18, 0 // (op 4247) 
s_cselect_b32 s19, 0, -1 // (op 4247) 

s_and_b32 s18, s12, 31 // (op 4247) 
s_lshl_b32 s49, 1, s18 // (op 4247) 
s_add_i32 s50, s49, s19 // (op 4247) 


s_and_b32 s19, s0, s50 // (op 4247) 


s_add_i32 s0, s1, s19 // (op 4247) 


s_ashr_i32 s1, s0, s18 // (op 4247) 


s_lshl_b32 s0, s12, 24 // (op 4247) 
s_ashr_i32 s18, s0, 31 // (op 4247) 

s_xor_b32 s0, s1, s18 // (op 4247) 

s_sub_i32 s1, s0, s18 // (op 4247) 


s_mul_hi_i32 s0, s1, s13 // (op 4247) 
s_add_i32 s18, s0, s1 // (op 4247) 

s_ashr_i32 s0, s18, 31 // (op 4247) 
s_cmp_eq_i32 s13, 0 // (op 4247) 
s_mov_b32 s19, scc // (op 4247) 
s_cmp_eq_i32 s19, 0 // (op 4247) 
s_cselect_b32 s49, 0, -1 // (op 4247) 

s_and_b32 s19, s14, 31 // (op 4247) 
s_lshl_b32 s50, 1, s19 // (op 4247) 
s_add_i32 s51, s50, s49 // (op 4247) 


s_and_b32 s49, s0, s51 // (op 4247) 


s_add_i32 s0, s18, s49 // (op 4247) 


s_ashr_i32 s18, s0, s19 // (op 4247) 


s_lshl_b32 s0, s14, 24 // (op 4247) 
s_ashr_i32 s19, s0, 31 // (op 4247) 

s_xor_b32 s0, s18, s19 // (op 4247) 

s_sub_i32 s18, s0, s19 // (op 4247) 


s_mul_i32 s0, s1, s44 // (op 4247) 

s_sub_u32 s1, s16, s0 // (op 4247) 

s_mul_i32 s0, s18, s44 // (op 4247) 
s_add_u32 s0, s0, s1 // (op 4247) 


s_mul_hi_i32 s1, s16, s10 // (op 4247) 
s_add_i32 s18, s1, s16 // (op 4247) 

s_ashr_i32 s1, s18, 31 // (op 4247) 
s_cmp_eq_i32 s10, 0 // (op 4247) 
s_mov_b32 s19, scc // (op 4247) 
s_cmp_eq_i32 s19, 0 // (op 4247) 
s_cselect_b32 s49, 0, -1 // (op 4247) 

s_and_b32 s19, s47, 31 // (op 4247) 
s_lshl_b32 s50, 1, s19 // (op 4247) 
s_add_i32 s51, s50, s49 // (op 4247) 


s_and_b32 s49, s1, s51 // (op 4247) 


s_add_i32 s1, s18, s49 // (op 4247) 


s_ashr_i32 s18, s1, s19 // (op 4247) 


s_lshl_b32 s1, s47, 24 // (op 4247) 
s_ashr_i32 s19, s1, 31 // (op 4247) 

s_xor_b32 s1, s18, s19 // (op 4247) 

s_sub_i32 s18, s1, s19 // (op 4247) 


s_add_u32 s50, s6, 64 // (op 4247) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4247) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4247) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4247) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 4247) 

s_mul_i32 s1, s44, s50 // (op 4247) 

s_mul_i32 s19, s48, s1 // (op 4247) 

s_mul_i32 s1, s18, s19 // (op 4247) 

s_sub_u32 s19, s16, s1 // (op 4247) 

s_mov_b32 s1, s19 // (op 4247) 
s_mul_hi_i32 s49, s1, s15 // (op 4247) 
s_add_i32 s50, s49, s1 // (op 4247) 


s_ashr_i32 s1, s50, 31 // (op 4247) 
s_cmp_eq_i32 s15, 0 // (op 4247) 
s_mov_b32 s49, scc // (op 4247) 
s_cmp_eq_i32 s49, 0 // (op 4247) 
s_cselect_b32 s51, 0, -1 // (op 4247) 

s_and_b32 s49, s46, 31 // (op 4247) 
s_lshl_b32 s72, 1, s49 // (op 4247) 
s_add_i32 s73, s72, s51 // (op 4247) 


s_and_b32 s51, s1, s73 // (op 4247) 


s_add_i32 s1, s50, s51 // (op 4247) 


s_ashr_i32 s50, s1, s49 // (op 4247) 


s_lshl_b32 s1, s46, 24 // (op 4247) 
s_ashr_i32 s49, s1, 31 // (op 4247) 

s_xor_b32 s1, s50, s49 // (op 4247) 

s_sub_i32 s50, s1, s49 // (op 4247) 


s_mul_i32 s1, s45, s44 // (op 4247) 
s_add_u32 s72, s8, 128 // (op 4247) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 4247) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 4247) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 4247) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 4247) 

s_sub_i32 s49, s72, s1 // (op 4247) 

s_mul_i32 s51, s50, s49 // (op 4247) 


s_sub_u32 s49, s19, s51 // (op 4247) 


s_add_u32 s19, s49, s1 // (op 4247) 


s_cmp_eq_i32 s18, 0 // (op 4247) 
s_mov_b32 s1, scc // (op 4247) 

s_cmp_eq_i32 s1, 0 // (op 4247) 
s_cselect_b32 s18, s19, s0 // (op 4247) 



v_lshl_add_u32 v14, s18, 7, v51 // (op 4247) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 4247) 
s_lshr_b32 s18, s0, 1 // (op 4247) 

v_lshl_add_u32 v51, s18, 8, v55 // (op 4247) 


v_mov_b32 v17, 0 // (op 4247) convert
v_mov_b32 v16, v51 // (op 4247) convert

v_mul_lo_u32 v47, v14, s25 // (op 4247) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 4247) most significant: high of low * low
v_mul_lo_u32 v46, v14, s24 // (op 4247) least significant: low of low * low
v_add_u32 v47, v47, v15 // (op 4247) most significant: sum

v_add_co_u32 v46, s[0:1], v46, v16 // (op 4247) least significant half
v_addc_co_u32 v47, s[0:1], v47, v17, s[0:1] // (op 4247) most significant half



v_mul_lo_u32 v15, v47, 6 // (op 4247) most significant: low of high * low
v_mul_hi_u32 v51, v46, 6 // (op 4247) most significant: high of low * low
v_mul_lo_u32 v14, v46, 6 // (op 4247) least significant: low of low * low
v_add_u32 v15, v15, v51 // (op 4247) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 4247) 

v_mov_b32 v51, v16 // (op 4247) convert

s_mul_i32 s1, s9, 6 // (op 4249) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 4249) least significant
s_mul_hi_u32 s18, s8, 6 // (op 4249) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4249) most significant: sum

s_mul_i32 s19, s24, s1 // (op 4249) most significant: low of low * high
s_mul_i32 s49, s25, s0 // (op 4249) most significant: low of high * low
s_mul_i32 s18, s24, s0 // (op 4249) least significant
s_mul_hi_u32 s50, s24, s0 // (op 4249) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4249) most significant: sum
s_add_u32 s19, s19, s50 // (op 4249) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4249) 

s_mov_b32 s20, s26 // (op 4249) reg expression
s_mov_b32 s21, s27 // (op 4249) reg expression
s_mov_b32 s22, s0 // (op 4249) reg expression
s_mov_b32 s23, 131072 // (op 4249) call()

s_mul_hi_i32 s0, s16, s11 // (op 4321) 
v_and_b32 v55, -2, v1 // (op 4321) 
s_add_i32 s1, s0, s16 // (op 4321) 

v_sub_u32 v14, v1, v55 // (op 4321) 

s_ashr_i32 s0, s1, 31 // (op 4321) 
v_lshlrev_b32 v55, 5, v14 // (op 4321) 

s_cmp_eq_i32 s11, 0 // (op 4321) 
s_mov_b32 s18, scc // (op 4321) 
v_lshrrev_b32 v14, 1, v1 // (op 4321) 
s_cmp_eq_i32 s18, 0 // (op 4321) 
s_cselect_b32 s19, 0, -1 // (op 4321) 

s_and_b32 s18, s12, 31 // (op 4321) 
s_lshl_b32 s49, 1, s18 // (op 4321) 
s_add_i32 s50, s49, s19 // (op 4321) 


s_and_b32 s19, s0, s50 // (op 4321) 


s_add_i32 s0, s1, s19 // (op 4321) 


s_ashr_i32 s1, s0, s18 // (op 4321) 


s_lshl_b32 s0, s12, 24 // (op 4321) 
s_ashr_i32 s18, s0, 31 // (op 4321) 

s_xor_b32 s0, s1, s18 // (op 4321) 

s_sub_i32 s1, s0, s18 // (op 4321) 


s_mul_hi_i32 s0, s1, s13 // (op 4321) 
s_add_i32 s18, s0, s1 // (op 4321) 

s_ashr_i32 s0, s18, 31 // (op 4321) 
s_cmp_eq_i32 s13, 0 // (op 4321) 
s_mov_b32 s19, scc // (op 4321) 
s_cmp_eq_i32 s19, 0 // (op 4321) 
s_cselect_b32 s49, 0, -1 // (op 4321) 

s_and_b32 s19, s14, 31 // (op 4321) 
s_lshl_b32 s50, 1, s19 // (op 4321) 
s_add_i32 s51, s50, s49 // (op 4321) 


s_and_b32 s49, s0, s51 // (op 4321) 


s_add_i32 s0, s18, s49 // (op 4321) 


s_ashr_i32 s18, s0, s19 // (op 4321) 


s_lshl_b32 s0, s14, 24 // (op 4321) 
s_ashr_i32 s19, s0, 31 // (op 4321) 

s_xor_b32 s0, s18, s19 // (op 4321) 

s_sub_i32 s18, s0, s19 // (op 4321) 


s_add_u32 s50, s6, 64 // (op 4321) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4321) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4321) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4321) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 4321) 

s_mul_i32 s0, s18, s50 // (op 4321) 

s_sub_i32 s49, s1, s0 // (op 4321) 


s_mul_hi_i32 s0, s16, s10 // (op 4321) 
s_add_i32 s1, s0, s16 // (op 4321) 

s_ashr_i32 s0, s1, 31 // (op 4321) 
s_cmp_eq_i32 s10, 0 // (op 4321) 
s_mov_b32 s18, scc // (op 4321) 
s_cmp_eq_i32 s18, 0 // (op 4321) 
s_cselect_b32 s19, 0, -1 // (op 4321) 

s_and_b32 s18, s47, 31 // (op 4321) 
s_lshl_b32 s72, 1, s18 // (op 4321) 
s_add_i32 s73, s72, s19 // (op 4321) 


s_and_b32 s19, s0, s73 // (op 4321) 


s_add_i32 s0, s1, s19 // (op 4321) 


s_ashr_i32 s1, s0, s18 // (op 4321) 


s_lshl_b32 s0, s47, 24 // (op 4321) 
s_ashr_i32 s18, s0, 31 // (op 4321) 

s_xor_b32 s0, s1, s18 // (op 4321) 

s_sub_i32 s1, s0, s18 // (op 4321) 


s_mul_i32 s0, s44, s50 // (op 4321) 

s_mul_i32 s18, s48, s0 // (op 4321) 

s_mul_i32 s0, s1, s18 // (op 4321) 

s_sub_u32 s18, s16, s0 // (op 4321) 

s_mul_hi_i32 s0, s18, s15 // (op 4321) 
s_add_i32 s19, s0, s18 // (op 4321) 


s_ashr_i32 s0, s19, 31 // (op 4321) 
s_cmp_eq_i32 s15, 0 // (op 4321) 
s_mov_b32 s18, scc // (op 4321) 
s_cmp_eq_i32 s18, 0 // (op 4321) 
s_cselect_b32 s50, 0, -1 // (op 4321) 

s_and_b32 s18, s46, 31 // (op 4321) 
s_lshl_b32 s51, 1, s18 // (op 4321) 
s_add_i32 s72, s51, s50 // (op 4321) 


s_and_b32 s50, s0, s72 // (op 4321) 


s_add_i32 s0, s19, s50 // (op 4321) 


s_ashr_i32 s19, s0, s18 // (op 4321) 


s_lshl_b32 s0, s46, 24 // (op 4321) 
s_ashr_i32 s18, s0, 31 // (op 4321) 

s_xor_b32 s0, s19, s18 // (op 4321) 

s_sub_i32 s19, s0, s18 // (op 4321) 


s_cmp_eq_i32 s1, 0 // (op 4321) 
s_mov_b32 s0, scc // (op 4321) 

s_cmp_eq_i32 s0, 0 // (op 4321) 
s_cselect_b32 s1, s19, s49 // (op 4321) 



v_lshl_add_u32 v15, s1, 6, v55 // (op 4321) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 4321) 
v_mov_b32 v17, 0 // (op 4321) convert
v_mov_b32 v16, v15 // (op 4321) convert

s_lshr_b32 s18, s0, 1 // (op 4321) 

v_lshl_add_u32 v15, s18, 8, v14 // (op 4321) 


v_mul_lo_u32 v47, v15, s29 // (op 4321) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 4321) most significant: high of low * low
v_mul_lo_u32 v46, v15, s28 // (op 4321) least significant: low of low * low
v_add_u32 v47, v47, v14 // (op 4321) most significant: sum

v_add_co_u32 v46, s[0:1], v46, v16 // (op 4321) least significant half
v_addc_co_u32 v47, s[0:1], v47, v17, s[0:1] // (op 4321) most significant half



v_mul_lo_u32 v15, v47, 6 // (op 4321) most significant: low of high * low
v_mul_hi_u32 v55, v46, 6 // (op 4321) most significant: high of low * low
v_mul_lo_u32 v14, v46, 6 // (op 4321) least significant: low of low * low
v_add_u32 v15, v15, v55 // (op 4321) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 4321) 

v_mov_b32 v55, v16 // (op 4321) convert

s_mul_i32 s1, s33, 6 // (op 4323) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 4323) least significant
s_mul_hi_u32 s18, s32, 6 // (op 4323) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4323) most significant: sum

s_mul_i32 s19, s28, s1 // (op 4323) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 4323) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 4323) least significant
s_mul_hi_u32 s50, s28, s0 // (op 4323) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 4323) most significant: sum
s_add_u32 s19, s19, s50 // (op 4323) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 4323) 

s_mov_b32 s52, s30 // (op 4323) reg expression
s_mov_b32 s53, s31 // (op 4323) reg expression
s_mov_b32 s54, s0 // (op 4323) reg expression
s_mov_b32 s55, 131072 // (op 4323) call()

s_mul_hi_i32 s0, s16, s11 // (op 4385) 
v_lshrrev_b32 v14, 2, v1 // (op 4385) 
s_add_i32 s1, s0, s16 // (op 4385) 

v_lshlrev_b32 v15, 1, v14 // (op 4385) 

s_ashr_i32 s0, s1, 31 // (op 4385) 
v_and_b32 v14, -4, v1 // (op 4385) 
s_cmp_eq_i32 s11, 0 // (op 4385) 
s_mov_b32 s18, scc // (op 4385) 
v_sub_u32 v16, v1, v14 // (op 4385) 

s_cmp_eq_i32 s18, 0 // (op 4385) 
s_cselect_b32 s19, 0, -1 // (op 4385) 

s_and_b32 s18, s12, 31 // (op 4385) 
s_lshl_b32 s49, 1, s18 // (op 4385) 
s_add_i32 s50, s49, s19 // (op 4385) 


s_and_b32 s19, s0, s50 // (op 4385) 


s_add_i32 s0, s1, s19 // (op 4385) 


s_ashr_i32 s1, s0, s18 // (op 4385) 


s_lshl_b32 s0, s12, 24 // (op 4385) 
s_ashr_i32 s18, s0, 31 // (op 4385) 

s_xor_b32 s0, s1, s18 // (op 4385) 

s_sub_i32 s1, s0, s18 // (op 4385) 


s_mul_hi_i32 s0, s1, s13 // (op 4385) 
s_add_i32 s18, s0, s1 // (op 4385) 

s_ashr_i32 s0, s18, 31 // (op 4385) 
s_cmp_eq_i32 s13, 0 // (op 4385) 
s_mov_b32 s19, scc // (op 4385) 
s_cmp_eq_i32 s19, 0 // (op 4385) 
s_cselect_b32 s49, 0, -1 // (op 4385) 

s_and_b32 s19, s14, 31 // (op 4385) 
s_lshl_b32 s50, 1, s19 // (op 4385) 
s_add_i32 s51, s50, s49 // (op 4385) 


s_and_b32 s49, s0, s51 // (op 4385) 


s_add_i32 s0, s18, s49 // (op 4385) 


s_ashr_i32 s18, s0, s19 // (op 4385) 


s_lshl_b32 s0, s14, 24 // (op 4385) 
s_ashr_i32 s19, s0, 31 // (op 4385) 

s_xor_b32 s0, s18, s19 // (op 4385) 

s_sub_i32 s18, s0, s19 // (op 4385) 


s_mul_i32 s0, s1, s44 // (op 4385) 

s_sub_u32 s1, s16, s0 // (op 4385) 

s_mul_i32 s0, s18, s44 // (op 4385) 
s_add_u32 s0, s0, s1 // (op 4385) 


s_mul_hi_i32 s1, s16, s10 // (op 4385) 
s_add_i32 s18, s1, s16 // (op 4385) 

s_ashr_i32 s1, s18, 31 // (op 4385) 
s_cmp_eq_i32 s10, 0 // (op 4385) 
s_mov_b32 s19, scc // (op 4385) 
s_cmp_eq_i32 s19, 0 // (op 4385) 
s_cselect_b32 s49, 0, -1 // (op 4385) 

s_and_b32 s19, s47, 31 // (op 4385) 
s_lshl_b32 s50, 1, s19 // (op 4385) 
s_add_i32 s51, s50, s49 // (op 4385) 


s_and_b32 s49, s1, s51 // (op 4385) 


s_add_i32 s1, s18, s49 // (op 4385) 


s_ashr_i32 s18, s1, s19 // (op 4385) 


s_lshl_b32 s1, s47, 24 // (op 4385) 
s_ashr_i32 s19, s1, 31 // (op 4385) 

s_xor_b32 s1, s18, s19 // (op 4385) 

s_sub_i32 s18, s1, s19 // (op 4385) 


s_add_u32 s50, s6, 64 // (op 4385) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4385) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4385) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4385) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 4385) 

s_mul_i32 s1, s44, s50 // (op 4385) 

s_mul_i32 s19, s48, s1 // (op 4385) 

s_mul_i32 s1, s18, s19 // (op 4385) 

s_sub_u32 s19, s16, s1 // (op 4385) 

s_mov_b32 s1, s19 // (op 4385) 
s_mul_hi_i32 s49, s1, s15 // (op 4385) 
s_add_i32 s50, s49, s1 // (op 4385) 


s_ashr_i32 s1, s50, 31 // (op 4385) 
s_cmp_eq_i32 s15, 0 // (op 4385) 
s_mov_b32 s49, scc // (op 4385) 
s_cmp_eq_i32 s49, 0 // (op 4385) 
s_cselect_b32 s51, 0, -1 // (op 4385) 

s_and_b32 s49, s46, 31 // (op 4385) 
s_lshl_b32 s72, 1, s49 // (op 4385) 
s_add_i32 s73, s72, s51 // (op 4385) 


s_and_b32 s51, s1, s73 // (op 4385) 


s_add_i32 s1, s50, s51 // (op 4385) 


s_ashr_i32 s50, s1, s49 // (op 4385) 


s_lshl_b32 s1, s46, 24 // (op 4385) 
s_ashr_i32 s49, s1, 31 // (op 4385) 

s_xor_b32 s1, s50, s49 // (op 4385) 

s_sub_i32 s50, s1, s49 // (op 4385) 


s_mul_i32 s1, s45, s44 // (op 4385) 
s_add_u32 s72, s8, 128 // (op 4385) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 4385) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 4385) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 4385) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 4385) 

s_sub_i32 s49, s72, s1 // (op 4385) 

s_mul_i32 s51, s50, s49 // (op 4385) 


s_sub_u32 s49, s19, s51 // (op 4385) 


s_add_u32 s19, s49, s1 // (op 4385) 


s_cmp_eq_i32 s18, 0 // (op 4385) 
s_mov_b32 s1, scc // (op 4385) 

s_cmp_eq_i32 s1, 0 // (op 4385) 
s_cselect_b32 s18, s19, s0 // (op 4385) 



v_lshl_add_u32 v14, s18, 7, v15 // (op 4385) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 4385) 
s_lshr_b32 s18, s0, 1 // (op 4385) 

v_lshl_add_u32 v15, s18, 3, v16 // (op 4385) 


v_mul_lo_u32 v16, v14, s34 // (op 4385) 
v_add_u32 v16, v16, v15 // (op 4385) 


s_mul_i32 s1, s34, s9 // (op 4387) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 4387) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 4387) least significant
s_mul_hi_u32 s19, s34, s8 // (op 4387) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4387) most significant: sum
s_add_u32 s1, s1, s19 // (op 4387) most significant: sum


s_mov_b32 s64, s36 // (op 4387) reg expression
s_mov_b32 s65, s37 // (op 4387) reg expression
s_mov_b32 s66, s0 // (op 4387) reg expression
s_mov_b32 s67, 131072 // (op 4387) call()

s_mul_hi_i32 s0, s16, s11 // (op 4449) 
v_and_b32 v17, -64, v1 // (op 4449) 
s_add_i32 s1, s0, s16 // (op 4449) 

v_sub_u32 v14, v1, v17 // (op 4449) 

s_ashr_i32 s0, s1, 31 // (op 4449) 
v_lshrrev_b32 v15, 6, v1 // (op 4449) 
s_cmp_eq_i32 s11, 0 // (op 4449) 
s_mov_b32 s18, scc // (op 4449) 
s_cmp_eq_i32 s18, 0 // (op 4449) 
s_cselect_b32 s19, 0, -1 // (op 4449) 

s_and_b32 s18, s12, 31 // (op 4449) 
s_lshl_b32 s49, 1, s18 // (op 4449) 
s_add_i32 s50, s49, s19 // (op 4449) 


s_and_b32 s19, s0, s50 // (op 4449) 


s_add_i32 s0, s1, s19 // (op 4449) 


s_ashr_i32 s1, s0, s18 // (op 4449) 


s_lshl_b32 s0, s12, 24 // (op 4449) 
s_ashr_i32 s18, s0, 31 // (op 4449) 

s_xor_b32 s0, s1, s18 // (op 4449) 

s_sub_i32 s1, s0, s18 // (op 4449) 


s_mul_hi_i32 s0, s1, s13 // (op 4449) 
s_add_i32 s18, s0, s1 // (op 4449) 

s_ashr_i32 s0, s18, 31 // (op 4449) 
s_cmp_eq_i32 s13, 0 // (op 4449) 
s_mov_b32 s19, scc // (op 4449) 
s_cmp_eq_i32 s19, 0 // (op 4449) 
s_cselect_b32 s49, 0, -1 // (op 4449) 

s_and_b32 s19, s14, 31 // (op 4449) 
s_lshl_b32 s50, 1, s19 // (op 4449) 
s_add_i32 s51, s50, s49 // (op 4449) 


s_and_b32 s49, s0, s51 // (op 4449) 


s_add_i32 s0, s18, s49 // (op 4449) 


s_ashr_i32 s18, s0, s19 // (op 4449) 


s_lshl_b32 s0, s14, 24 // (op 4449) 
s_ashr_i32 s19, s0, 31 // (op 4449) 

s_xor_b32 s0, s18, s19 // (op 4449) 

s_sub_i32 s18, s0, s19 // (op 4449) 


s_add_u32 s50, s6, 64 // (op 4449) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 4449) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 4449) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 4449) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 4449) 

s_mul_i32 s0, s18, s50 // (op 4449) 

s_sub_i32 s49, s1, s0 // (op 4449) 


s_mul_hi_i32 s0, s16, s10 // (op 4449) 
s_add_i32 s1, s0, s16 // (op 4449) 

s_ashr_i32 s0, s1, 31 // (op 4449) 
s_cmp_eq_i32 s10, 0 // (op 4449) 
s_mov_b32 s18, scc // (op 4449) 
s_cmp_eq_i32 s18, 0 // (op 4449) 
s_cselect_b32 s19, 0, -1 // (op 4449) 

s_and_b32 s18, s47, 31 // (op 4449) 
s_lshl_b32 s72, 1, s18 // (op 4449) 
s_add_i32 s73, s72, s19 // (op 4449) 


s_and_b32 s19, s0, s73 // (op 4449) 


s_add_i32 s0, s1, s19 // (op 4449) 


s_ashr_i32 s1, s0, s18 // (op 4449) 


s_lshl_b32 s0, s47, 24 // (op 4449) 
s_ashr_i32 s18, s0, 31 // (op 4449) 

s_xor_b32 s0, s1, s18 // (op 4449) 

s_sub_i32 s1, s0, s18 // (op 4449) 


s_mul_i32 s0, s44, s50 // (op 4449) 

s_mul_i32 s18, s48, s0 // (op 4449) 

s_mul_i32 s0, s1, s18 // (op 4449) 

s_sub_u32 s18, s16, s0 // (op 4449) 

s_mul_hi_i32 s0, s18, s15 // (op 4449) 
s_add_i32 s19, s0, s18 // (op 4449) 


s_ashr_i32 s0, s19, 31 // (op 4449) 
s_cmp_eq_i32 s15, 0 // (op 4449) 
s_mov_b32 s18, scc // (op 4449) 
s_cmp_eq_i32 s18, 0 // (op 4449) 
s_cselect_b32 s50, 0, -1 // (op 4449) 

s_and_b32 s18, s46, 31 // (op 4449) 
s_lshl_b32 s51, 1, s18 // (op 4449) 
s_add_i32 s72, s51, s50 // (op 4449) 


s_and_b32 s50, s0, s72 // (op 4449) 


s_add_i32 s0, s19, s50 // (op 4449) 


s_ashr_i32 s19, s0, s18 // (op 4449) 


s_lshl_b32 s0, s46, 24 // (op 4449) 
s_ashr_i32 s18, s0, 31 // (op 4449) 

s_xor_b32 s0, s19, s18 // (op 4449) 

s_sub_i32 s19, s0, s18 // (op 4449) 


s_cmp_eq_i32 s1, 0 // (op 4449) 
s_mov_b32 s0, scc // (op 4449) 

s_cmp_eq_i32 s0, 0 // (op 4449) 
s_cselect_b32 s1, s19, s49 // (op 4449) 



v_lshl_add_u32 v17, s1, 6, v14 // (op 4449) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 4449) 
s_lshr_b32 s18, s0, 1 // (op 4449) 

v_lshl_add_u32 v14, s18, 3, v15 // (op 4449) 


v_mul_lo_u32 v15, v14, s38 // (op 4449) 
v_add_u32 v15, v15, v17 // (op 4449) 


s_mul_i32 s1, s38, s43 // (op 4451) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 4451) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 4451) least significant
s_mul_hi_u32 s19, s38, s42 // (op 4451) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4451) most significant: sum
s_add_u32 s1, s1, s19 // (op 4451) most significant: sum


s_mov_b32 s68, s40 // (op 4451) reg expression
s_mov_b32 s69, s41 // (op 4451) reg expression
s_mov_b32 s70, s0 // (op 4451) reg expression
s_mov_b32 s71, 131072 // (op 4451) call()

v_and_b32 v14, -2, v1 // (op 4547) 
v_sub_u32 v17, v1, v14 // (op 4547) 

v_lshlrev_b32 v14, 6, v17 // (op 4547) 

v_lshrrev_b32 v17, 1, v1 // (op 4547) 
v_lshl_add_u32 v19, v17, 7, v14 // (op 4547) 


v_mul_lo_u32 v14, v19, 6 // (op 4547) 

v_lshrrev_b32 v17, 3, v14 // (op 4547) 

v_and_b32 v14, -2, v1 // (op 4605) 
v_sub_u32 v19, v1, v14 // (op 4605) 

v_lshlrev_b32 v14, 5, v19 // (op 4605) 

v_lshrrev_b32 v19, 1, v1 // (op 4605) 
v_lshl_add_u32 v20, v19, 6, v14 // (op 4605) 


v_mul_lo_u32 v14, v20, 6 // (op 4605) 
v_lshrrev_b32 v19, 3, v14 // (op 4605) 

v_lshrrev_b32 v14, 4, v20 // (op 4605) 

v_lshlrev_b32 v20, 2, v14 // (op 4605) 

v_add_u32 v14, v19, v20 // (op 4605) 


v_and_b32 v19, -4, v1 // (op 4663) 
v_sub_u32 v20, v1, v19 // (op 4663) 

v_lshrrev_b32 v19, 2, v1 // (op 4663) 
v_lshl_add_u32 v21, v19, 3, v20 // (op 4663) 


v_and_b32 v19, -64, v1 // (op 4721) 
v_sub_u32 v20, v1, v19 // (op 4721) 

v_and_b32 v19, 4294967232, v1 // (op 4721) 
v_add_u32 v22, v19, v20 // (op 4721) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1366) 
s_and_b32 s18, s0, 4294967294 // (op 1366) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 1365) 
s_mov_b32 s19, s0 // (op 1365) convert

s_cmp_lt_u32 s18, s19 // (op 1365) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1365 // (op 1365) Condition: Top (jump to RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1365 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1365:


s_add_u32 s18, s18, 1 // (op 1367) 
v_mov_b32 v18, v51 // (op 176) 

buffer_load_dwordx4 v[24:27], v18, s[20:23], 0 offen // (op 176) Load value
buffer_load_dwordx2 v[28:29], v18, s[20:23], 0 offen offset:16 // (op 176) Load value
buffer_load_dwordx4 v[30:33], v18, s[20:23], 0 offen offset:24 // (op 176) Load value
buffer_load_dwordx2 v[34:35], v18, s[20:23], 0 offen offset:40 // (op 176) Load value

v_add_u32 v51, 96, v51 // (op 4237) 


s_barrier  // (op 4962) 
v_mov_b32 v23, v55 // (op 190) 



v_mov_b32 v18, v23 // (op 190) reg expression

buffer_load_dwordx3 v[36:38], v18, s[52:55], 0 offen // (op 190) Load value
buffer_load_dwordx3 v[40:42], v18, s[52:55], 0 offen offset:12 // (op 190) Load value


s_lshl_b64 s[0:1], s[28:29], 7 // (op 4313) 
s_mul_i32 s51, s1, 6 // (op 4313) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 4313) least significant
s_mul_hi_u32 s19, s0, 6 // (op 4313) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 4313) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 4313) 

v_add_u32 v55, v55, s0 // (op 4313) 

s_waitcnt vmcnt(5)
ds_write_b128 v17, v[24:27] offset:8448 // (op 177) Store local data 

v_mov_b64 v[18:19], s[34:35] // (op 191) convert

v_mov_b32 v20, v16 // (op 191) 
v_mov_b32 v23, v20 // (op 191) 
s_waitcnt vmcnt(4)
ds_write_b64 v17, v[28:29] offset:8464 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v17, v[30:33] offset:8472 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[34:35] offset:8488 // (op 177) Store local data 
buffer_load_ubyte v52, v23, s[64:67], 0 offen offset: 0 // (op 191) Load value

v_add_u32 v23, v23, v18 // (op 191) 
s_barrier  // (op 4975) 
buffer_load_ubyte v53, v23, s[64:67], 0 offen offset: 0 // (op 191) Load value






v_add_u32 v16, 4, v16 // (op 4377) 
s_waitcnt vmcnt(3)
ds_write_b96 v14, v[36:38] // (op 187) Store local data 


v_mov_b32 v39, v15 // (op 192) 
v_mov_b32 v54, v39 // (op 192) reg expression

buffer_load_ubyte v18, v54, s[68:71], 0 offen offset: 0 // (op 192) Load value


s_lshl_b32 s19, s38, 2 // (op 4441) 
v_add_u32 v15, v15, s19 // (op 4441) 

s_waitcnt vmcnt(3)
ds_write_b96 v14, v[40:42] offset:16 // (op 187) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 4965) 



s_barrier  // (op 4989) 
ds_read_b128 v[24:27], v3 offset:8448 // (op 179) Load local data 



ds_read_b64 v[28:29], v3 offset:8464 // (op 179) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v21, v52 offset:20736 // (op 188) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4979) 
s_waitcnt vmcnt(1)
ds_write_b8 v21, v53 offset:20740 // (op 188) Store local data 




ds_read_b96_tr_b6 v[30:32], v0 // (op 181) Transpose load local data 
s_barrier  // (op 5003) 
ds_read_b96_tr_b6 v[33:35], v0 offset:1024 // (op 181) Transpose load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 4993) 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v18 offset:8192 // (op 189) Store local data 




ds_read_u8 v23, v4 offset:20736 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 5007) 


ds_read_u8 v18, v5 offset:8192 // (op 185) Load local data 



ds_read_b96_tr_b6 v[36:38], v0 offset:32 // (op 350) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v0 offset:1056 // (op 350) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], a[0:3], v23, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 186) 


ds_read_u8 v19, v5 offset:8224 // (op 354) Load local data 



ds_read_b128 v[42:45], v3 offset:11520 // (op 568) Load local data 
ds_read_b64 v[46:47], v3 offset:11536 // (op 568) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:29], v[36:41], a[4:7], v23, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 355) 





ds_read_u8 v20, v4 offset:20864 // (op 571) Load local data 



ds_read_b128 v[24:27], v3 offset:14592 // (op 954) Load local data 
ds_read_b64 v[28:29], v3 offset:14608 // (op 954) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:47], v[30:35], a[8:11], v20, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 573) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:47], v[36:41], a[12:15], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 604) 
ds_read_u8 v23, v4 offset:20992 // (op 957) Load local data 





ds_read_b128 v[42:45], v3 offset:17664 // (op 1340) Load local data 
ds_read_b64 v[46:47], v3 offset:17680 // (op 1340) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:29], v[30:35], a[16:19], v23, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 959) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:29], v[36:41], a[20:23], v23, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 990) 
ds_read_u8 v20, v4 offset:21120 // (op 1343) Load local data 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:47], v[30:35], a[24:27], v20, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1345) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:47], v[36:41], a[28:31], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b011 // (op 1376) 




s_ashr_i64 s[0:1], s[32:33], 7 // (op 1365) 
s_mov_b32 s19, s0 // (op 1365) convert

s_cmp_lt_u32 s18, s19 // (op 1365) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1365 // (op 1365) Condition: Bottom (jump to RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_1365 if true)
RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_1365:























s_barrier  // (op 5113) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133
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


v_accvgpr_read v6, a4 // (op 1869) 
v_accvgpr_read v7, a5 // (op 1869) 
v_accvgpr_read v8, a6 // (op 1869) 
v_accvgpr_read v9, a7 // (op 1869) 
v_mul_f32 v14, v11, v6 // (op 1869) 
v_mul_f32 v15, v11, v7 // (op 1869) 
v_mul_f32 v16, v11, v8 // (op 1869) 
v_mul_f32 v17, v11, v9 // (op 1869) 



s_lshl_b32 s0, s4, 7 // (op 1871) 
v_add_u32 v0, v13, s0 // (op 1871) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1871) Store value


v_accvgpr_read v6, a8 // (op 1893) 
v_accvgpr_read v7, a9 // (op 1893) 
v_accvgpr_read v8, a10 // (op 1893) 
v_accvgpr_read v9, a11 // (op 1893) 
v_mul_f32 v14, v11, v6 // (op 1893) 
v_mul_f32 v15, v11, v7 // (op 1893) 
v_mul_f32 v16, v11, v8 // (op 1893) 
v_mul_f32 v17, v11, v9 // (op 1893) 



v_add_u32 v0, 128, v13 // (op 1896) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1896) Store value


v_accvgpr_read v6, a12 // (op 1902) 
v_accvgpr_read v7, a13 // (op 1902) 
v_accvgpr_read v8, a14 // (op 1902) 
v_accvgpr_read v9, a15 // (op 1902) 
v_mul_f32 v14, v11, v6 // (op 1902) 
v_mul_f32 v15, v11, v7 // (op 1902) 
v_mul_f32 v16, v11, v8 // (op 1902) 
v_mul_f32 v17, v11, v9 // (op 1902) 



s_lshl_b32 s0, s4, 7 // (op 1898) 
s_add_u32 s0, s0, 128 // (op 1898) 
v_add_u32 v0, v13, s0 // (op 1898) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1898) Store value


v_accvgpr_read v6, a16 // (op 1929) 
v_accvgpr_read v7, a17 // (op 1929) 
v_accvgpr_read v8, a18 // (op 1929) 
v_accvgpr_read v9, a19 // (op 1929) 
v_mul_f32 v14, v11, v6 // (op 1929) 
v_mul_f32 v15, v11, v7 // (op 1929) 
v_mul_f32 v16, v11, v8 // (op 1929) 
v_mul_f32 v17, v11, v9 // (op 1929) 



v_add_u32 v0, 256, v13 // (op 1932) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1932) Store value


v_accvgpr_read v6, a20 // (op 1938) 
v_accvgpr_read v7, a21 // (op 1938) 
v_accvgpr_read v8, a22 // (op 1938) 
v_accvgpr_read v9, a23 // (op 1938) 
v_mul_f32 v14, v11, v6 // (op 1938) 
v_mul_f32 v15, v11, v7 // (op 1938) 
v_mul_f32 v16, v11, v8 // (op 1938) 
v_mul_f32 v17, v11, v9 // (op 1938) 



s_lshl_b32 s0, s4, 7 // (op 1934) 
s_add_u32 s0, s0, 256 // (op 1934) 
v_add_u32 v0, v13, s0 // (op 1934) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1934) Store value


v_accvgpr_read v6, a24 // (op 1965) 
v_accvgpr_read v7, a25 // (op 1965) 
v_accvgpr_read v8, a26 // (op 1965) 
v_accvgpr_read v9, a27 // (op 1965) 
v_mul_f32 v14, v11, v6 // (op 1965) 
v_mul_f32 v15, v11, v7 // (op 1965) 
v_mul_f32 v16, v11, v8 // (op 1965) 
v_mul_f32 v17, v11, v9 // (op 1965) 



v_add_u32 v0, 384, v13 // (op 1968) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1968) Store value


v_accvgpr_read v6, a28 // (op 1974) 
v_accvgpr_read v7, a29 // (op 1974) 
v_accvgpr_read v8, a30 // (op 1974) 
v_accvgpr_read v9, a31 // (op 1974) 
v_mul_f32 v14, v11, v6 // (op 1974) 
v_mul_f32 v15, v11, v7 // (op 1974) 
v_mul_f32 v16, v11, v8 // (op 1974) 
v_mul_f32 v17, v11, v9 // (op 1974) 



s_lshl_b32 s0, s4, 7 // (op 1970) 
s_add_u32 s0, s0, 384 // (op 1970) 
v_add_u32 v0, v13, s0 // (op 1970) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1970) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 256, v12 // (op 2111) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2111) Load value

v_accvgpr_read v14, a16 // (op 2103) 
v_accvgpr_read v15, a17 // (op 2103) 
v_accvgpr_read v16, a18 // (op 2103) 
v_accvgpr_read v17, a19 // (op 2103) 
v_mul_f32 v18, v11, v14 // (op 2103) 
v_mul_f32 v19, v11, v15 // (op 2103) 
v_mul_f32 v20, v11, v16 // (op 2103) 
v_mul_f32 v21, v11, v17 // (op 2103) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2110) 
v_mul_f32 v15, v10, v7 // (op 2110) 
v_mul_f32 v16, v10, v8 // (op 2110) 
v_mul_f32 v17, v10, v9 // (op 2110) 

v_add_f32 v6, v14, v18 // (op 2104) 
v_add_f32 v7, v15, v19 // (op 2104) 
v_add_f32 v8, v16, v20 // (op 2104) 
v_add_f32 v9, v17, v21 // (op 2104) 




v_add_u32 v0, 256, v13 // (op 2106) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2106) Store value




s_lshl_b32 s0, s2, 7 // (op 2115) 
s_add_u32 s0, s0, 256 // (op 2115) 
v_add_u32 v0, v12, s0 // (op 2115) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2115) Load value

v_accvgpr_read v14, a20 // (op 2117) 
v_accvgpr_read v15, a21 // (op 2117) 
v_accvgpr_read v16, a22 // (op 2117) 
v_accvgpr_read v17, a23 // (op 2117) 
v_mul_f32 v18, v11, v14 // (op 2117) 
v_mul_f32 v19, v11, v15 // (op 2117) 
v_mul_f32 v20, v11, v16 // (op 2117) 
v_mul_f32 v21, v11, v17 // (op 2117) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2113) 
v_mul_f32 v15, v10, v7 // (op 2113) 
v_mul_f32 v16, v10, v8 // (op 2113) 
v_mul_f32 v17, v10, v9 // (op 2113) 

v_add_f32 v6, v14, v18 // (op 2114) 
v_add_f32 v7, v15, v19 // (op 2114) 
v_add_f32 v8, v16, v20 // (op 2114) 
v_add_f32 v9, v17, v21 // (op 2114) 




s_lshl_b32 s0, s4, 7 // (op 2108) 
s_add_u32 s0, s0, 256 // (op 2108) 
v_add_u32 v0, v13, s0 // (op 2108) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2108) Store value




v_add_u32 v0, 128, v12 // (op 2070) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2070) Load value

v_accvgpr_read v14, a8 // (op 2062) 
v_accvgpr_read v15, a9 // (op 2062) 
v_accvgpr_read v16, a10 // (op 2062) 
v_accvgpr_read v17, a11 // (op 2062) 
v_mul_f32 v18, v11, v14 // (op 2062) 
v_mul_f32 v19, v11, v15 // (op 2062) 
v_mul_f32 v20, v11, v16 // (op 2062) 
v_mul_f32 v21, v11, v17 // (op 2062) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2069) 
v_mul_f32 v15, v10, v7 // (op 2069) 
v_mul_f32 v16, v10, v8 // (op 2069) 
v_mul_f32 v17, v10, v9 // (op 2069) 

v_add_f32 v6, v14, v18 // (op 2063) 
v_add_f32 v7, v15, v19 // (op 2063) 
v_add_f32 v8, v16, v20 // (op 2063) 
v_add_f32 v9, v17, v21 // (op 2063) 




v_add_u32 v0, 128, v13 // (op 2065) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2065) Store value




s_lshl_b32 s0, s2, 7 // (op 2074) 
s_add_u32 s0, s0, 128 // (op 2074) 
v_add_u32 v0, v12, s0 // (op 2074) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2074) Load value

v_accvgpr_read v14, a12 // (op 2076) 
v_accvgpr_read v15, a13 // (op 2076) 
v_accvgpr_read v16, a14 // (op 2076) 
v_accvgpr_read v17, a15 // (op 2076) 
v_mul_f32 v18, v11, v14 // (op 2076) 
v_mul_f32 v19, v11, v15 // (op 2076) 
v_mul_f32 v20, v11, v16 // (op 2076) 
v_mul_f32 v21, v11, v17 // (op 2076) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2072) 
v_mul_f32 v15, v10, v7 // (op 2072) 
v_mul_f32 v16, v10, v8 // (op 2072) 
v_mul_f32 v17, v10, v9 // (op 2072) 

v_add_f32 v6, v14, v18 // (op 2073) 
v_add_f32 v7, v15, v19 // (op 2073) 
v_add_f32 v8, v16, v20 // (op 2073) 
v_add_f32 v9, v17, v21 // (op 2073) 




s_lshl_b32 s0, s4, 7 // (op 2067) 
s_add_u32 s0, s0, 128 // (op 2067) 
v_add_u32 v0, v13, s0 // (op 2067) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2067) Store value




v_add_u32 v0, 384, v12 // (op 2152) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2152) Load value

v_accvgpr_read v14, a24 // (op 2144) 
v_accvgpr_read v15, a25 // (op 2144) 
v_accvgpr_read v16, a26 // (op 2144) 
v_accvgpr_read v17, a27 // (op 2144) 
v_mul_f32 v18, v11, v14 // (op 2144) 
v_mul_f32 v19, v11, v15 // (op 2144) 
v_mul_f32 v20, v11, v16 // (op 2144) 
v_mul_f32 v21, v11, v17 // (op 2144) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2151) 
v_mul_f32 v15, v10, v7 // (op 2151) 
v_mul_f32 v16, v10, v8 // (op 2151) 
v_mul_f32 v17, v10, v9 // (op 2151) 

v_add_f32 v6, v14, v18 // (op 2145) 
v_add_f32 v7, v15, v19 // (op 2145) 
v_add_f32 v8, v16, v20 // (op 2145) 
v_add_f32 v9, v17, v21 // (op 2145) 




v_add_u32 v0, 384, v13 // (op 2147) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2147) Store value




s_lshl_b32 s0, s2, 7 // (op 2156) 
s_add_u32 s0, s0, 384 // (op 2156) 
v_add_u32 v0, v12, s0 // (op 2156) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2156) Load value

v_accvgpr_read v14, a28 // (op 2158) 
v_accvgpr_read v15, a29 // (op 2158) 
v_accvgpr_read v16, a30 // (op 2158) 
v_accvgpr_read v17, a31 // (op 2158) 
v_mul_f32 v18, v11, v14 // (op 2158) 
v_mul_f32 v19, v11, v15 // (op 2158) 
v_mul_f32 v20, v11, v16 // (op 2158) 
v_mul_f32 v21, v11, v17 // (op 2158) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2154) 
v_mul_f32 v15, v10, v7 // (op 2154) 
v_mul_f32 v16, v10, v8 // (op 2154) 
v_mul_f32 v17, v10, v9 // (op 2154) 

v_add_f32 v6, v14, v18 // (op 2155) 
v_add_f32 v7, v15, v19 // (op 2155) 
v_add_f32 v8, v16, v20 // (op 2155) 
v_add_f32 v9, v17, v21 // (op 2155) 




s_lshl_b32 s0, s4, 7 // (op 2149) 
s_add_u32 s0, s0, 384 // (op 2149) 
v_add_u32 v0, v13, s0 // (op 2149) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2149) Store value




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




s_lshl_b32 s0, s2, 7 // (op 2032) 
v_add_u32 v0, v12, s0 // (op 2032) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2032) Load value

v_accvgpr_read v14, a4 // (op 2029) 
v_accvgpr_read v15, a5 // (op 2029) 
v_accvgpr_read v16, a6 // (op 2029) 
v_accvgpr_read v17, a7 // (op 2029) 
v_mul_f32 v18, v11, v14 // (op 2029) 
v_mul_f32 v19, v11, v15 // (op 2029) 
v_mul_f32 v20, v11, v16 // (op 2029) 
v_mul_f32 v21, v11, v17 // (op 2029) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2033) 
v_mul_f32 v15, v10, v7 // (op 2033) 
v_mul_f32 v16, v10, v8 // (op 2033) 
v_mul_f32 v17, v10, v9 // (op 2033) 

v_add_f32 v6, v14, v18 // (op 2030) 
v_add_f32 v7, v15, v19 // (op 2030) 
v_add_f32 v8, v16, v20 // (op 2030) 
v_add_f32 v9, v17, v21 // (op 2030) 




s_lshl_b32 s0, s4, 7 // (op 2031) 
v_add_u32 v0, v13, s0 // (op 2031) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 2031) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC
  .amdhsa_next_free_vgpr 132
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 42496 // lds bytes
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
    DataTypeB: B6
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
    macrotile: [128, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_TT_FP6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__LDSSYNC.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 42496
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 100
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

