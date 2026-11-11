// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_,@function
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s12 // (op 1158) 
s_mul_hi_i32 s1, s16, s12 // (op 1346) 
v_and_b32 v0, 48, v1 // (op 1214) 
v_and_b32 v3, 63, v1 // (op 1214) 
v_sub_u32 v4, v3, v0 // (op 1214) 

v_lshrrev_b32 v3, 1, v4 // (op 1214) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1214) 


v_and_b32 v0, 1, v4 // (op 1214) 

v_lshlrev_b32 v3, 3, v0 // (op 1214) 

v_lshrrev_b32 v0, 7, v1 // (op 1214) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1214) 


v_add_u32 v0, v5, v4 // (op 1214) 


v_and_b32 v3, 48, v1 // (op 1387) 
v_and_b32 v4, 63, v1 // (op 1387) 
v_sub_u32 v5, v4, v3 // (op 1387) 

v_lshrrev_b32 v4, 1, v5 // (op 1387) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 1387) 


v_and_b32 v3, 1, v5 // (op 1387) 

v_lshlrev_b32 v4, 3, v3 // (op 1387) 

v_lshrrev_b32 v3, 7, v1 // (op 1387) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1387) 


v_add_u32 v3, v6, v5 // (op 1387) 


v_lshrrev_b32 v4, 7, v1 // (op 1158) 
s_add_i32 s18, s0, s16 // (op 1158) 

v_and_b32 v5, 63, v1 // (op 1158) 
s_ashr_i32 s0, s18, 31 // (op 1158) 
v_lshrrev_b32 v6, 4, v5 // (op 1158) 
s_cmp_eq_i32 s12, 0 // (op 1158) 
s_mov_b32 s19, scc // (op 1158) 
v_lshlrev_b32 v7, 2, v6 // (op 1158) 

s_cmp_eq_i32 s19, 0 // (op 1158) 
s_cselect_b32 s51, 0, -1 // (op 1158) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1158) 


s_and_b32 s19, s15, 31 // (op 1158) 
v_lshrrev_b32 v4, 2, v5 // (op 1158) 

v_and_b32 v5, 3, v4 // (op 1158) 

v_and_b32 v4, 63, v1 // (op 1272) 
v_and_b32 v7, 48, v1 // (op 1272) 
v_sub_u32 v8, v4, v7 // (op 1272) 

v_lshrrev_b32 v7, 4, v4 // (op 1272) 

v_lshl_add_u32 v4, v7, 5, v8 // (op 1272) 


v_lshrrev_b32 v7, 6, v1 // (op 1272) 
v_and_b32 v8, 1, v7 // (op 1272) 

v_lshlrev_b32 v7, 4, v8 // (op 1272) 

v_lshl_add_u32 v8, v4, 6, v7 // (op 1272) 


v_mul_lo_u32 v4, v8, 6 // (op 1272) 
v_lshrrev_b32 v7, 3, v4 // (op 1272) 

v_lshrrev_b32 v4, 4, v8 // (op 1272) 

v_lshlrev_b32 v8, 2, v4 // (op 1272) 

v_add_u32 v4, v7, v8 // (op 1272) 


v_lshrrev_b32 v7, 6, v1 // (op 1330) 
v_and_b32 v8, 1, v7 // (op 1330) 

v_and_b32 v7, 15, v1 // (op 1330) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 1330) 


v_and_b32 v7, 63, v1 // (op 1330) 
v_lshrrev_b32 v8, 4, v7 // (op 1330) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 1330) 


v_lshrrev_b32 v8, 7, v1 // (op 1346) 
v_and_b32 v9, 63, v1 // (op 1346) 
v_and_b32 v10, 63, v1 // (op 1416) 
v_and_b32 v11, 48, v1 // (op 1416) 
v_sub_u32 v12, v10, v11 // (op 1416) 

v_lshrrev_b32 v11, 4, v10 // (op 1416) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 1416) 


v_lshrrev_b32 v11, 6, v1 // (op 1416) 
v_and_b32 v12, 1, v11 // (op 1416) 

v_lshlrev_b32 v11, 4, v12 // (op 1416) 

v_lshl_add_u32 v12, v10, 6, v11 // (op 1416) 


v_mul_lo_u32 v10, v12, 6 // (op 1416) 
v_lshrrev_b32 v11, 3, v10 // (op 1416) 

v_lshrrev_b32 v10, 4, v12 // (op 1416) 

v_lshlrev_b32 v12, 2, v10 // (op 1416) 

v_add_u32 v10, v11, v12 // (op 1416) 


v_lshrrev_b32 v11, 6, v1 // (op 1445) 
v_and_b32 v12, 1, v11 // (op 1445) 

v_and_b32 v11, 15, v1 // (op 1445) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 1445) 


v_and_b32 v11, 63, v1 // (op 1445) 
v_lshrrev_b32 v12, 4, v11 // (op 1445) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 1445) 


s_lshl_b32 s52, 1, s19 // (op 1158) 
v_and_b32 v12, 3, v1 // (op 1158) 
s_add_i32 s53, s52, s51 // (op 1158) 


s_and_b32 s51, s0, s53 // (op 1158) 


s_add_i32 s0, s1, s16 // (op 1346) 

s_ashr_i32 s1, s0, 31 // (op 1346) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1158) 


s_add_i32 s52, s18, s51 // (op 1158) 


v_lshrrev_b32 v5, 6, v1 // (op 1158) 
s_ashr_i32 s18, s52, s19 // (op 1158) 


v_and_b32 v12, 1, v5 // (op 1158) 

s_lshl_b32 s19, s15, 24 // (op 1158) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1158) 


s_ashr_i32 s51, s19, 31 // (op 1158) 

s_xor_b32 s19, s18, s51 // (op 1158) 

s_sub_i32 s18, s19, s51 // (op 1158) 


v_lshrrev_b32 v12, 4, v9 // (op 1346) 
s_cmp_eq_i32 s12, 0 // (op 1346) 
s_mov_b32 s19, scc // (op 1346) 
v_lshlrev_b32 v13, 2, v12 // (op 1346) 

s_cmp_eq_i32 s19, 0 // (op 1346) 
s_cselect_b32 s51, 0, -1 // (op 1346) 

v_lshl_add_u32 v12, v8, 4, v13 // (op 1346) 


s_and_b32 s19, s15, 31 // (op 1346) 
v_lshrrev_b32 v8, 2, v9 // (op 1346) 

s_lshl_b32 s52, 1, s19 // (op 1346) 
v_and_b32 v9, 3, v8 // (op 1346) 

s_add_i32 s53, s52, s51 // (op 1346) 


v_and_b32 v8, 3, v1 // (op 1346) 
s_and_b32 s51, s1, s53 // (op 1346) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1346) 


s_add_i32 s1, s0, s51 // (op 1346) 


v_lshrrev_b32 v8, 6, v1 // (op 1346) 
s_ashr_i32 s0, s1, s19 // (op 1346) 


v_and_b32 v9, 1, v8 // (op 1346) 

s_lshl_b32 s1, s15, 24 // (op 1346) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1346) 


s_ashr_i32 s19, s1, 31 // (op 1346) 

s_xor_b32 s1, s0, s19 // (op 1346) 

s_sub_i32 s0, s1, s19 // (op 1346) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s1, s18, s44 // (op 1158) 
s_add_i32 s19, s1, s18 // (op 1158) 

s_ashr_i32 s1, s19, 31 // (op 1158) 
s_cmp_eq_i32 s44, 0 // (op 1158) 
s_mov_b32 s51, scc // (op 1158) 
s_cmp_eq_i32 s51, 0 // (op 1158) 
s_cselect_b32 s52, 0, -1 // (op 1158) 

s_and_b32 s51, s45, 31 // (op 1158) 
s_lshl_b32 s53, 1, s51 // (op 1158) 
s_add_i32 s54, s53, s52 // (op 1158) 


s_and_b32 s52, s1, s54 // (op 1158) 


s_add_i32 s1, s19, s52 // (op 1158) 


s_ashr_i32 s19, s1, s51 // (op 1158) 


s_lshl_b32 s1, s45, 24 // (op 1158) 
s_ashr_i32 s51, s1, 31 // (op 1158) 

s_xor_b32 s1, s19, s51 // (op 1158) 

s_sub_i32 s19, s1, s51 // (op 1158) 


s_mul_i32 s1, s18, s13 // (op 1158) 
s_sub_u32 s51, s16, s1 // (op 1158) 

s_mul_i32 s1, s19, s13 // (op 1158) 
s_add_u32 s1, s1, s51 // (op 1158) 

s_mul_hi_i32 s51, s16, s11 // (op 1158) 
s_add_i32 s52, s51, s16 // (op 1158) 

s_ashr_i32 s51, s52, 31 // (op 1158) 
s_cmp_eq_i32 s11, 0 // (op 1158) 
s_mov_b32 s53, scc // (op 1158) 
s_cmp_eq_i32 s53, 0 // (op 1158) 
s_cselect_b32 s54, 0, -1 // (op 1158) 

s_and_b32 s53, s47, 31 // (op 1158) 
s_lshl_b32 s55, 1, s53 // (op 1158) 
s_add_i32 s56, s55, s54 // (op 1158) 


s_and_b32 s54, s51, s56 // (op 1158) 


s_add_i32 s51, s52, s54 // (op 1158) 


s_ashr_i32 s52, s51, s53 // (op 1158) 


s_lshl_b32 s51, s47, 24 // (op 1158) 
s_ashr_i32 s53, s51, 31 // (op 1158) 

s_xor_b32 s51, s52, s53 // (op 1158) 

s_sub_i32 s52, s51, s53 // (op 1158) 


s_mul_hi_i32 s51, s0, s44 // (op 1346) 
s_add_i32 s53, s51, s0 // (op 1346) 

s_ashr_i32 s51, s53, 31 // (op 1346) 
s_cmp_eq_i32 s44, 0 // (op 1346) 
s_mov_b32 s54, scc // (op 1346) 
s_cmp_eq_i32 s54, 0 // (op 1346) 
s_cselect_b32 s55, 0, -1 // (op 1346) 

s_and_b32 s54, s45, 31 // (op 1346) 
s_lshl_b32 s56, 1, s54 // (op 1346) 
s_add_i32 s57, s56, s55 // (op 1346) 


s_and_b32 s55, s51, s57 // (op 1346) 


s_add_i32 s51, s53, s55 // (op 1346) 


s_ashr_i32 s53, s51, s54 // (op 1346) 


s_lshl_b32 s51, s45, 24 // (op 1346) 
s_ashr_i32 s54, s51, 31 // (op 1346) 

s_xor_b32 s51, s53, s54 // (op 1346) 

s_sub_i32 s53, s51, s54 // (op 1346) 


s_mul_i32 s51, s0, s13 // (op 1346) 
s_sub_u32 s54, s16, s51 // (op 1346) 

s_mul_i32 s51, s53, s13 // (op 1346) 
s_add_u32 s51, s51, s54 // (op 1346) 

s_mul_hi_i32 s54, s16, s11 // (op 1346) 
s_add_i32 s55, s54, s16 // (op 1346) 

s_ashr_i32 s54, s55, 31 // (op 1346) 
s_cmp_eq_i32 s11, 0 // (op 1346) 
s_mov_b32 s56, scc // (op 1346) 
s_cmp_eq_i32 s56, 0 // (op 1346) 
s_cselect_b32 s57, 0, -1 // (op 1346) 

s_and_b32 s56, s47, 31 // (op 1346) 
s_lshl_b32 s58, 1, s56 // (op 1346) 
s_add_i32 s59, s58, s57 // (op 1346) 


s_and_b32 s57, s54, s59 // (op 1346) 


s_add_i32 s54, s55, s57 // (op 1346) 


s_ashr_i32 s55, s54, s56 // (op 1346) 


s_lshl_b32 s54, s47, 24 // (op 1346) 
s_ashr_i32 s56, s54, 31 // (op 1346) 

s_xor_b32 s54, s55, s56 // (op 1346) 

s_sub_i32 s55, s54, s56 // (op 1346) 


s_add_u32 s56, s8, 64 // (op 1158) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 1158) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1158) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1158) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1158) 

s_mul_i32 s54, s13, s56 // (op 1158) 
s_mul_i32 s58, s10, s54 // (op 1158) 

s_mul_i32 s54, s52, s58 // (op 1158) 

s_sub_u32 s58, s16, s54 // (op 1158) 

s_mov_b32 s54, s58 // (op 1158) 
s_mul_hi_i32 s59, s54, s14 // (op 1158) 
s_add_i32 s60, s59, s54 // (op 1158) 


s_ashr_i32 s54, s60, 31 // (op 1158) 
s_cmp_eq_i32 s14, 0 // (op 1158) 
s_mov_b32 s59, scc // (op 1158) 
s_cmp_eq_i32 s59, 0 // (op 1158) 
s_cselect_b32 s61, 0, -1 // (op 1158) 

s_and_b32 s59, s46, 31 // (op 1158) 
s_lshl_b32 s62, 1, s59 // (op 1158) 
s_add_i32 s63, s62, s61 // (op 1158) 


s_and_b32 s61, s54, s63 // (op 1158) 


s_add_i32 s54, s60, s61 // (op 1158) 


s_ashr_i32 s60, s54, s59 // (op 1158) 


s_lshl_b32 s54, s46, 24 // (op 1158) 
s_ashr_i32 s59, s54, 31 // (op 1158) 

s_xor_b32 s54, s60, s59 // (op 1158) 

s_sub_i32 s60, s54, s59 // (op 1158) 


s_mul_i32 s54, s48, s13 // (op 1158) 
s_add_u32 s62, s6, 16 // (op 1158) Start of Int64 add, locking SCC
s_addc_u32 s63, s7, 0 // (op 1158) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1158) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1158) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 1158) 

s_sub_i32 s59, s62, s54 // (op 1158) 

s_mul_i32 s61, s60, s59 // (op 1158) 

s_sub_u32 s59, s58, s61 // (op 1158) 


s_add_u32 s58, s59, s54 // (op 1158) 


s_cmp_eq_i32 s52, 0 // (op 1158) 
s_mov_b32 s54, scc // (op 1158) 

s_cmp_eq_i32 s54, 0 // (op 1158) 
s_cselect_b32 s52, s58, s1 // (op 1158) 


s_mul_i32 s1, s19, s56 // (op 1158) 


v_lshl_add_u32 v9, s52, 4, v6 // (op 1158) 


s_sub_i32 s19, s18, s1 // (op 1158) 


s_cmp_eq_i32 s54, 0 // (op 1158) 
s_cselect_b32 s1, s60, s19 // (op 1158) 



v_lshl_add_u32 v6, s1, 6, v5 // (op 1158) 


v_mul_lo_u32 v5, v6, s2 // (op 1158) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1158) 


s_add_u32 s18, s8, 64 // (op 1346) Start of Int64 add, locking SCC
s_addc_u32 s19, s9, 0 // (op 1346) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 1346) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 1346) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 6 // (op 1346) 

s_mul_i32 s1, s13, s18 // (op 1346) 
s_mul_i32 s52, s10, s1 // (op 1346) 

s_mul_i32 s1, s55, s52 // (op 1346) 

s_sub_u32 s52, s16, s1 // (op 1346) 

s_mov_b32 s1, s52 // (op 1346) 
s_mul_hi_i32 s54, s1, s14 // (op 1346) 
s_add_i32 s56, s54, s1 // (op 1346) 


s_ashr_i32 s1, s56, 31 // (op 1346) 
s_cmp_eq_i32 s14, 0 // (op 1346) 
s_mov_b32 s54, scc // (op 1346) 
s_cmp_eq_i32 s54, 0 // (op 1346) 
s_cselect_b32 s57, 0, -1 // (op 1346) 

s_and_b32 s54, s46, 31 // (op 1346) 
s_lshl_b32 s58, 1, s54 // (op 1346) 
s_add_i32 s59, s58, s57 // (op 1346) 


s_and_b32 s57, s1, s59 // (op 1346) 


s_add_i32 s1, s56, s57 // (op 1346) 


s_ashr_i32 s56, s1, s54 // (op 1346) 


s_lshl_b32 s1, s46, 24 // (op 1346) 
s_ashr_i32 s54, s1, 31 // (op 1346) 

s_xor_b32 s1, s56, s54 // (op 1346) 

s_sub_i32 s56, s1, s54 // (op 1346) 


s_mul_i32 s1, s48, s13 // (op 1346) 
s_add_u32 s58, s6, 16 // (op 1346) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1346) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1346) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1346) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1346) 

s_sub_i32 s54, s58, s1 // (op 1346) 

s_mul_i32 s57, s56, s54 // (op 1346) 

s_sub_u32 s54, s52, s57 // (op 1346) 


s_add_u32 s52, s54, s1 // (op 1346) 


s_cmp_eq_i32 s55, 0 // (op 1346) 
s_mov_b32 s1, scc // (op 1346) 

s_cmp_eq_i32 s1, 0 // (op 1346) 
s_cselect_b32 s54, s52, s51 // (op 1346) 


s_mul_i32 s55, s53, s18 // (op 1346) 


v_lshl_add_u32 v5, s54, 4, v12 // (op 1346) 


s_sub_i32 s18, s0, s55 // (op 1346) 


s_cmp_eq_i32 s1, 0 // (op 1346) 
s_cselect_b32 s0, s56, s18 // (op 1346) 



v_lshl_add_u32 v9, s0, 6, v8 // (op 1346) 


v_mul_lo_u32 v8, v9, s4 // (op 1346) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 1346) 


s_mul_i32 s1, s2, s9 // (op 1160) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1160) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1160) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1160) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1160) most significant: sum
s_add_u32 s1, s1, s19 // (op 1160) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1160) 

s_mul_i32 s1, s4, s9 // (op 1348) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 1348) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1348) least significant
s_mul_hi_u32 s52, s4, s8 // (op 1348) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1348) most significant: sum
s_add_u32 s1, s1, s52 // (op 1348) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1348) 

s_mov_b32 s56, s20 // (op 1160) reg expression
s_mov_b32 s57, s21 // (op 1160) reg expression
s_mov_b32 s58, s18 // (op 1160) reg expression
s_mov_b32 s59, 131072 // (op 1160) call()

s_mov_b32 s60, s22 // (op 1348) reg expression

s_mov_b32 s61, s23 // (op 1348) reg expression
s_mov_b32 s62, s52 // (op 1348) reg expression
s_mov_b32 s63, 131072 // (op 1348) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 310) call()
v_accvgpr_write a5, 0.00000 // (op 310) call()
v_accvgpr_write a6, 0.00000 // (op 310) call()
v_accvgpr_write a7, 0.00000 // (op 310) call()
s_mul_hi_i32 s0, s16, s12 // (op 925) 
s_add_i32 s1, s0, s16 // (op 925) 

s_ashr_i32 s0, s1, 31 // (op 925) 
s_cmp_eq_i32 s12, 0 // (op 925) 
s_mov_b32 s18, scc // (op 925) 
s_cmp_eq_i32 s18, 0 // (op 925) 
s_cselect_b32 s19, 0, -1 // (op 925) 

s_and_b32 s18, s15, 31 // (op 925) 
s_lshl_b32 s20, 1, s18 // (op 925) 
s_add_i32 s21, s20, s19 // (op 925) 


s_and_b32 s19, s0, s21 // (op 925) 


s_add_i32 s0, s1, s19 // (op 925) 


s_ashr_i32 s1, s0, s18 // (op 925) 


s_lshl_b32 s0, s15, 24 // (op 925) 
s_ashr_i32 s18, s0, 31 // (op 925) 

s_xor_b32 s0, s1, s18 // (op 925) 

s_sub_i32 s1, s0, s18 // (op 925) 


s_mul_hi_i32 s0, s1, s44 // (op 925) 
s_add_i32 s18, s0, s1 // (op 925) 

s_ashr_i32 s0, s18, 31 // (op 925) 
s_cmp_eq_i32 s44, 0 // (op 925) 
s_mov_b32 s19, scc // (op 925) 
s_cmp_eq_i32 s19, 0 // (op 925) 
s_cselect_b32 s20, 0, -1 // (op 925) 

s_and_b32 s19, s45, 31 // (op 925) 
s_lshl_b32 s21, 1, s19 // (op 925) 
s_add_i32 s22, s21, s20 // (op 925) 


s_and_b32 s23, s0, s22 // (op 925) 


s_add_i32 s0, s18, s23 // (op 925) 


s_ashr_i32 s18, s0, s19 // (op 925) 


s_lshl_b32 s0, s45, 24 // (op 925) 
s_ashr_i32 s19, s0, 31 // (op 925) 

s_xor_b32 s0, s18, s19 // (op 925) 

s_sub_i32 s18, s0, s19 // (op 925) 


s_mul_i32 s0, s1, s13 // (op 925) 

s_sub_u32 s1, s16, s0 // (op 925) 

s_mul_i32 s0, s18, s13 // (op 925) 
s_add_u32 s0, s0, s1 // (op 925) 


s_mul_hi_i32 s1, s16, s11 // (op 925) 
s_add_i32 s18, s1, s16 // (op 925) 

s_ashr_i32 s1, s18, 31 // (op 925) 
s_cmp_eq_i32 s11, 0 // (op 925) 
s_mov_b32 s19, scc // (op 925) 
s_cmp_eq_i32 s19, 0 // (op 925) 
s_cselect_b32 s20, 0, -1 // (op 925) 

s_and_b32 s19, s47, 31 // (op 925) 
s_lshl_b32 s21, 1, s19 // (op 925) 
s_add_i32 s22, s21, s20 // (op 925) 


s_and_b32 s23, s1, s22 // (op 925) 


s_add_i32 s1, s18, s23 // (op 925) 


s_ashr_i32 s18, s1, s19 // (op 925) 


s_lshl_b32 s1, s47, 24 // (op 925) 
s_ashr_i32 s19, s1, 31 // (op 925) 

s_xor_b32 s1, s18, s19 // (op 925) 

s_sub_i32 s18, s1, s19 // (op 925) 


s_add_u32 s22, s8, 64 // (op 925) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 925) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 925) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 925) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 925) 

s_mul_i32 s1, s13, s22 // (op 925) 

s_mul_i32 s19, s10, s1 // (op 925) 

s_mul_i32 s1, s18, s19 // (op 925) 

s_sub_u32 s19, s16, s1 // (op 925) 

s_mov_b32 s1, s19 // (op 925) 
s_mul_hi_i32 s20, s1, s14 // (op 925) 
s_add_i32 s21, s20, s1 // (op 925) 


s_ashr_i32 s1, s21, 31 // (op 925) 
s_cmp_eq_i32 s14, 0 // (op 925) 
s_mov_b32 s20, scc // (op 925) 
s_cmp_eq_i32 s20, 0 // (op 925) 
s_cselect_b32 s22, 0, -1 // (op 925) 

s_and_b32 s20, s46, 31 // (op 925) 
s_lshl_b32 s23, 1, s20 // (op 925) 
s_add_i32 s49, s23, s22 // (op 925) 


s_and_b32 s22, s1, s49 // (op 925) 


s_add_i32 s1, s21, s22 // (op 925) 


s_ashr_i32 s21, s1, s20 // (op 925) 


s_lshl_b32 s1, s46, 24 // (op 925) 
s_ashr_i32 s20, s1, 31 // (op 925) 

s_xor_b32 s1, s21, s20 // (op 925) 

s_sub_i32 s21, s1, s20 // (op 925) 


s_mul_i32 s1, s48, s13 // (op 925) 
s_add_u32 s22, s6, 16 // (op 925) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 925) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 925) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 925) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 925) 

s_sub_i32 s20, s22, s1 // (op 925) 

s_mul_i32 s22, s21, s20 // (op 925) 


s_sub_u32 s23, s19, s22 // (op 925) 


s_add_u32 s19, s23, s1 // (op 925) 


s_cmp_eq_i32 s18, 0 // (op 925) 
s_mov_b32 s1, scc // (op 925) 

s_cmp_eq_i32 s1, 0 // (op 925) 
s_cselect_b32 s18, s19, s0 // (op 925) 



s_lshl_b32 s0, s18, 4 // (op 925) 

s_mov_b32 s19, 0 // (op 925) convert
s_mov_b32 s18, s0 // (op 925) convert

v_mul_lo_u32 v13, v1, s25 // (op 925) most significant: low of low * high
v_mul_hi_u32 v14, v1, s24 // (op 925) most significant: high of low * low
v_mul_lo_u32 v12, v1, s24 // (op 925) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 925) most significant: sum

v_mov_b32 v14, s18 // (op 925) 
v_mov_b32 v15, s19 // (op 925) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 925) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 925) most significant half





s_mul_i32 s1, s24, s29 // (op 927) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 927) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 927) least significant
s_mul_hi_u32 s19, s24, s28 // (op 927) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 927) most significant: sum
s_add_u32 s1, s1, s19 // (op 927) most significant: sum


s_mov_b32 s20, s26 // (op 927) reg expression
s_mov_b32 s21, s27 // (op 927) reg expression
s_mov_b32 s22, s0 // (op 927) reg expression
s_mov_b32 s23, 131072 // (op 927) call()

s_mul_hi_i32 s0, s16, s12 // (op 944) 
s_add_i32 s1, s0, s16 // (op 944) 

s_ashr_i32 s0, s1, 31 // (op 944) 
s_cmp_eq_i32 s12, 0 // (op 944) 
s_mov_b32 s18, scc // (op 944) 
s_cmp_eq_i32 s18, 0 // (op 944) 
s_cselect_b32 s19, 0, -1 // (op 944) 

s_and_b32 s18, s15, 31 // (op 944) 
s_lshl_b32 s49, 1, s18 // (op 944) 
s_add_i32 s50, s49, s19 // (op 944) 


s_and_b32 s19, s0, s50 // (op 944) 


s_add_i32 s0, s1, s19 // (op 944) 


s_ashr_i32 s1, s0, s18 // (op 944) 


s_lshl_b32 s0, s15, 24 // (op 944) 
s_ashr_i32 s18, s0, 31 // (op 944) 

s_xor_b32 s0, s1, s18 // (op 944) 

s_sub_i32 s1, s0, s18 // (op 944) 


s_mul_hi_i32 s0, s1, s44 // (op 944) 
s_add_i32 s18, s0, s1 // (op 944) 

s_ashr_i32 s0, s18, 31 // (op 944) 
s_cmp_eq_i32 s44, 0 // (op 944) 
s_mov_b32 s19, scc // (op 944) 
s_cmp_eq_i32 s19, 0 // (op 944) 
s_cselect_b32 s49, 0, -1 // (op 944) 

s_and_b32 s19, s45, 31 // (op 944) 
s_lshl_b32 s50, 1, s19 // (op 944) 
s_add_i32 s51, s50, s49 // (op 944) 


s_and_b32 s49, s0, s51 // (op 944) 


s_add_i32 s0, s18, s49 // (op 944) 


s_ashr_i32 s18, s0, s19 // (op 944) 


s_lshl_b32 s0, s45, 24 // (op 944) 
s_ashr_i32 s19, s0, 31 // (op 944) 

s_xor_b32 s0, s18, s19 // (op 944) 

s_sub_i32 s18, s0, s19 // (op 944) 


s_mul_i32 s0, s1, s13 // (op 944) 

s_sub_u32 s1, s16, s0 // (op 944) 

s_mul_i32 s0, s18, s13 // (op 944) 
s_add_u32 s0, s0, s1 // (op 944) 


s_mul_hi_i32 s1, s16, s11 // (op 944) 
s_add_i32 s18, s1, s16 // (op 944) 

s_ashr_i32 s1, s18, 31 // (op 944) 
s_cmp_eq_i32 s11, 0 // (op 944) 
s_mov_b32 s19, scc // (op 944) 
s_cmp_eq_i32 s19, 0 // (op 944) 
s_cselect_b32 s49, 0, -1 // (op 944) 

s_and_b32 s19, s47, 31 // (op 944) 
s_lshl_b32 s50, 1, s19 // (op 944) 
s_add_i32 s51, s50, s49 // (op 944) 


s_and_b32 s49, s1, s51 // (op 944) 


s_add_i32 s1, s18, s49 // (op 944) 


s_ashr_i32 s18, s1, s19 // (op 944) 


s_lshl_b32 s1, s47, 24 // (op 944) 
s_ashr_i32 s19, s1, 31 // (op 944) 

s_xor_b32 s1, s18, s19 // (op 944) 

s_sub_i32 s18, s1, s19 // (op 944) 


s_add_u32 s54, s8, 64 // (op 944) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 944) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 944) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 944) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 944) 

s_mul_i32 s1, s13, s50 // (op 944) 

s_mul_i32 s19, s10, s1 // (op 944) 

s_mul_i32 s1, s18, s19 // (op 944) 

s_sub_u32 s19, s16, s1 // (op 944) 

s_mov_b32 s1, s19 // (op 944) 
s_mul_hi_i32 s49, s1, s14 // (op 944) 
s_add_i32 s50, s49, s1 // (op 944) 


s_ashr_i32 s1, s50, 31 // (op 944) 
s_cmp_eq_i32 s14, 0 // (op 944) 
s_mov_b32 s49, scc // (op 944) 
s_cmp_eq_i32 s49, 0 // (op 944) 
s_cselect_b32 s51, 0, -1 // (op 944) 

s_and_b32 s49, s46, 31 // (op 944) 
s_lshl_b32 s52, 1, s49 // (op 944) 
s_add_i32 s53, s52, s51 // (op 944) 


s_and_b32 s51, s1, s53 // (op 944) 


s_add_i32 s1, s50, s51 // (op 944) 


s_ashr_i32 s50, s1, s49 // (op 944) 


s_lshl_b32 s1, s46, 24 // (op 944) 
s_ashr_i32 s49, s1, 31 // (op 944) 

s_xor_b32 s1, s50, s49 // (op 944) 

s_sub_i32 s50, s1, s49 // (op 944) 


s_mul_i32 s1, s48, s13 // (op 944) 
s_add_u32 s54, s6, 16 // (op 944) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 944) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 944) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 944) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 944) 

s_sub_i32 s49, s54, s1 // (op 944) 

s_mul_i32 s51, s50, s49 // (op 944) 


s_sub_u32 s49, s19, s51 // (op 944) 


s_add_u32 s19, s49, s1 // (op 944) 


s_cmp_eq_i32 s18, 0 // (op 944) 
s_mov_b32 s1, scc // (op 944) 

s_cmp_eq_i32 s1, 0 // (op 944) 
s_cselect_b32 s18, s19, s0 // (op 944) 



s_lshl_b32 s19, s18, 4 // (op 944) 

s_mov_b32 s1, 0 // (op 944) convert
s_mov_b32 s0, s19 // (op 944) convert

v_mul_lo_u32 v13, v1, s25 // (op 944) most significant: low of low * high
v_mul_hi_u32 v14, v1, s24 // (op 944) most significant: high of low * low
v_mul_lo_u32 v12, v1, s24 // (op 944) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 944) most significant: sum

v_mov_b32 v14, s0 // (op 944) 
v_mov_b32 v15, s1 // (op 944) 
v_add_co_u32 v18, s[18:19], v14, v12 // (op 944) least significant half
v_addc_co_u32 v19, s[18:19], v15, v13, s[18:19] // (op 944) most significant half





s_mul_i32 s1, s24, s29 // (op 946) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 946) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 946) least significant
s_mul_hi_u32 s19, s24, s28 // (op 946) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 946) most significant: sum
s_add_u32 s1, s1, s19 // (op 946) most significant: sum


s_mov_b32 s20, s26 // (op 946) reg expression
s_mov_b32 s21, s27 // (op 946) reg expression
s_mov_b32 s22, s0 // (op 946) reg expression
s_mov_b32 s23, 131072 // (op 946) call()

s_mul_hi_i32 s0, s16, s12 // (op 989) 
s_add_i32 s1, s0, s16 // (op 989) 

s_ashr_i32 s0, s1, 31 // (op 989) 
s_cmp_eq_i32 s12, 0 // (op 989) 
s_mov_b32 s18, scc // (op 989) 
s_cmp_eq_i32 s18, 0 // (op 989) 
s_cselect_b32 s19, 0, -1 // (op 989) 

s_and_b32 s18, s15, 31 // (op 989) 
s_lshl_b32 s49, 1, s18 // (op 989) 
s_add_i32 s50, s49, s19 // (op 989) 


s_and_b32 s19, s0, s50 // (op 989) 


s_add_i32 s0, s1, s19 // (op 989) 


s_ashr_i32 s1, s0, s18 // (op 989) 


s_lshl_b32 s0, s15, 24 // (op 989) 
s_ashr_i32 s18, s0, 31 // (op 989) 

s_xor_b32 s0, s1, s18 // (op 989) 

s_sub_i32 s1, s0, s18 // (op 989) 


s_mul_hi_i32 s0, s1, s44 // (op 989) 
s_add_i32 s18, s0, s1 // (op 989) 

s_ashr_i32 s0, s18, 31 // (op 989) 
s_cmp_eq_i32 s44, 0 // (op 989) 
s_mov_b32 s19, scc // (op 989) 
s_cmp_eq_i32 s19, 0 // (op 989) 
s_cselect_b32 s49, 0, -1 // (op 989) 

s_and_b32 s19, s45, 31 // (op 989) 
s_lshl_b32 s50, 1, s19 // (op 989) 
s_add_i32 s51, s50, s49 // (op 989) 


s_and_b32 s49, s0, s51 // (op 989) 


s_add_i32 s0, s18, s49 // (op 989) 


s_ashr_i32 s18, s0, s19 // (op 989) 


s_lshl_b32 s0, s45, 24 // (op 989) 
s_ashr_i32 s19, s0, 31 // (op 989) 

s_xor_b32 s0, s18, s19 // (op 989) 

s_sub_i32 s18, s0, s19 // (op 989) 


s_add_u32 s54, s8, 64 // (op 989) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 989) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 989) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 989) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 989) 

s_mul_i32 s0, s18, s50 // (op 989) 

s_sub_i32 s49, s1, s0 // (op 989) 


s_mul_hi_i32 s0, s16, s11 // (op 989) 
s_add_i32 s1, s0, s16 // (op 989) 

s_ashr_i32 s0, s1, 31 // (op 989) 
s_cmp_eq_i32 s11, 0 // (op 989) 
s_mov_b32 s18, scc // (op 989) 
s_cmp_eq_i32 s18, 0 // (op 989) 
s_cselect_b32 s19, 0, -1 // (op 989) 

s_and_b32 s18, s47, 31 // (op 989) 
s_lshl_b32 s52, 1, s18 // (op 989) 
s_add_i32 s53, s52, s19 // (op 989) 


s_and_b32 s19, s0, s53 // (op 989) 


s_add_i32 s0, s1, s19 // (op 989) 


s_ashr_i32 s1, s0, s18 // (op 989) 


s_lshl_b32 s0, s47, 24 // (op 989) 
s_ashr_i32 s18, s0, 31 // (op 989) 

s_xor_b32 s0, s1, s18 // (op 989) 

s_sub_i32 s1, s0, s18 // (op 989) 


s_mul_i32 s0, s13, s50 // (op 989) 

s_mul_i32 s18, s10, s0 // (op 989) 

s_mul_i32 s0, s1, s18 // (op 989) 

s_sub_u32 s18, s16, s0 // (op 989) 

s_mul_hi_i32 s0, s18, s14 // (op 989) 
s_add_i32 s19, s0, s18 // (op 989) 


s_ashr_i32 s0, s19, 31 // (op 989) 
s_cmp_eq_i32 s14, 0 // (op 989) 
s_mov_b32 s18, scc // (op 989) 
s_cmp_eq_i32 s18, 0 // (op 989) 
s_cselect_b32 s50, 0, -1 // (op 989) 

s_and_b32 s18, s46, 31 // (op 989) 
s_lshl_b32 s51, 1, s18 // (op 989) 
s_add_i32 s52, s51, s50 // (op 989) 


s_and_b32 s50, s0, s52 // (op 989) 


s_add_i32 s0, s19, s50 // (op 989) 


s_ashr_i32 s19, s0, s18 // (op 989) 


s_lshl_b32 s0, s46, 24 // (op 989) 
s_ashr_i32 s18, s0, 31 // (op 989) 

s_xor_b32 s0, s19, s18 // (op 989) 

s_sub_i32 s19, s0, s18 // (op 989) 


s_cmp_eq_i32 s1, 0 // (op 989) 
s_mov_b32 s0, scc // (op 989) 

s_cmp_eq_i32 s0, 0 // (op 989) 
s_cselect_b32 s1, s19, s49 // (op 989) 



s_lshl_b32 s0, s1, 6 // (op 989) 

s_mov_b32 s19, 0 // (op 989) convert
s_mov_b32 s18, s0 // (op 989) convert

v_mul_lo_u32 v13, v1, s31 // (op 989) most significant: low of low * high
v_mul_hi_u32 v14, v1, s30 // (op 989) most significant: high of low * low
v_mul_lo_u32 v12, v1, s30 // (op 989) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 989) most significant: sum

v_mov_b32 v14, s18 // (op 989) 
v_mov_b32 v15, s19 // (op 989) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 989) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 989) most significant half




v_mul_lo_u32 v15, v13, 6 // (op 989) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 989) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 989) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 989) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 989) 

v_mov_b32 v14, v12 // (op 989) convert

s_mul_i32 s1, s29, 6 // (op 991) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 991) least significant
s_mul_hi_u32 s18, s28, 6 // (op 991) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 991) most significant: sum

s_mul_i32 s19, s30, s1 // (op 991) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 991) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 991) least significant
s_mul_hi_u32 s50, s30, s0 // (op 991) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 991) most significant: sum
s_add_u32 s19, s19, s50 // (op 991) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 991) 

s_mov_b32 s52, s32 // (op 991) reg expression
s_mov_b32 s53, s33 // (op 991) reg expression
s_mov_b32 s54, s0 // (op 991) reg expression
s_mov_b32 s55, 131072 // (op 991) call()

s_mul_hi_i32 s0, s16, s12 // (op 1008) 
s_add_i32 s1, s0, s16 // (op 1008) 

s_ashr_i32 s0, s1, 31 // (op 1008) 
s_cmp_eq_i32 s12, 0 // (op 1008) 
s_mov_b32 s18, scc // (op 1008) 
s_cmp_eq_i32 s18, 0 // (op 1008) 
s_cselect_b32 s19, 0, -1 // (op 1008) 

s_and_b32 s18, s15, 31 // (op 1008) 
s_lshl_b32 s49, 1, s18 // (op 1008) 
s_add_i32 s50, s49, s19 // (op 1008) 


s_and_b32 s19, s0, s50 // (op 1008) 


s_add_i32 s0, s1, s19 // (op 1008) 


s_ashr_i32 s1, s0, s18 // (op 1008) 


s_lshl_b32 s0, s15, 24 // (op 1008) 
s_ashr_i32 s18, s0, 31 // (op 1008) 

s_xor_b32 s0, s1, s18 // (op 1008) 

s_sub_i32 s1, s0, s18 // (op 1008) 


s_mul_hi_i32 s0, s1, s44 // (op 1008) 
s_add_i32 s18, s0, s1 // (op 1008) 

s_ashr_i32 s0, s18, 31 // (op 1008) 
s_cmp_eq_i32 s44, 0 // (op 1008) 
s_mov_b32 s19, scc // (op 1008) 
s_cmp_eq_i32 s19, 0 // (op 1008) 
s_cselect_b32 s49, 0, -1 // (op 1008) 

s_and_b32 s19, s45, 31 // (op 1008) 
s_lshl_b32 s50, 1, s19 // (op 1008) 
s_add_i32 s51, s50, s49 // (op 1008) 


s_and_b32 s49, s0, s51 // (op 1008) 


s_add_i32 s0, s18, s49 // (op 1008) 


s_ashr_i32 s18, s0, s19 // (op 1008) 


s_lshl_b32 s0, s45, 24 // (op 1008) 
s_ashr_i32 s19, s0, 31 // (op 1008) 

s_xor_b32 s0, s18, s19 // (op 1008) 

s_sub_i32 s18, s0, s19 // (op 1008) 


s_add_u32 s50, s8, 64 // (op 1008) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1008) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1008) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1008) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1008) 

s_mul_i32 s0, s18, s50 // (op 1008) 

s_sub_i32 s49, s1, s0 // (op 1008) 


s_mul_hi_i32 s0, s16, s11 // (op 1008) 
s_add_i32 s1, s0, s16 // (op 1008) 

s_ashr_i32 s0, s1, 31 // (op 1008) 
s_cmp_eq_i32 s11, 0 // (op 1008) 
s_mov_b32 s18, scc // (op 1008) 
s_cmp_eq_i32 s18, 0 // (op 1008) 
s_cselect_b32 s19, 0, -1 // (op 1008) 

s_and_b32 s18, s47, 31 // (op 1008) 
s_lshl_b32 s64, 1, s18 // (op 1008) 
s_add_i32 s65, s64, s19 // (op 1008) 


s_and_b32 s19, s0, s65 // (op 1008) 


s_add_i32 s0, s1, s19 // (op 1008) 


s_ashr_i32 s1, s0, s18 // (op 1008) 


s_lshl_b32 s0, s47, 24 // (op 1008) 
s_ashr_i32 s18, s0, 31 // (op 1008) 

s_xor_b32 s0, s1, s18 // (op 1008) 

s_sub_i32 s1, s0, s18 // (op 1008) 


s_mul_i32 s0, s13, s50 // (op 1008) 

s_mul_i32 s18, s10, s0 // (op 1008) 

s_mul_i32 s0, s1, s18 // (op 1008) 

s_sub_u32 s18, s16, s0 // (op 1008) 

s_mul_hi_i32 s0, s18, s14 // (op 1008) 
s_add_i32 s19, s0, s18 // (op 1008) 


s_ashr_i32 s0, s19, 31 // (op 1008) 
s_cmp_eq_i32 s14, 0 // (op 1008) 
s_mov_b32 s18, scc // (op 1008) 
s_cmp_eq_i32 s18, 0 // (op 1008) 
s_cselect_b32 s50, 0, -1 // (op 1008) 

s_and_b32 s18, s46, 31 // (op 1008) 
s_lshl_b32 s51, 1, s18 // (op 1008) 
s_add_i32 s64, s51, s50 // (op 1008) 


s_and_b32 s50, s0, s64 // (op 1008) 


s_add_i32 s0, s19, s50 // (op 1008) 


s_ashr_i32 s19, s0, s18 // (op 1008) 


s_lshl_b32 s0, s46, 24 // (op 1008) 
s_ashr_i32 s18, s0, 31 // (op 1008) 

s_xor_b32 s0, s19, s18 // (op 1008) 

s_sub_i32 s19, s0, s18 // (op 1008) 


s_cmp_eq_i32 s1, 0 // (op 1008) 
s_mov_b32 s0, scc // (op 1008) 

s_cmp_eq_i32 s0, 0 // (op 1008) 
s_cselect_b32 s1, s19, s49 // (op 1008) 



s_lshl_b32 s0, s1, 6 // (op 1008) 

s_mov_b32 s19, 0 // (op 1008) convert
s_mov_b32 s18, s0 // (op 1008) convert

v_mul_lo_u32 v13, v1, s31 // (op 1008) most significant: low of low * high
v_mul_hi_u32 v15, v1, s30 // (op 1008) most significant: high of low * low
v_mul_lo_u32 v12, v1, s30 // (op 1008) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 1008) most significant: sum

v_mov_b32 v15, s18 // (op 1008) 
v_mov_b32 v20, s19 // (op 1008) 
v_add_co_u32 v12, s[0:1], v12, v15 // (op 1008) least significant half
v_addc_co_u32 v13, s[0:1], v13, v20, s[0:1] // (op 1008) most significant half




v_mul_lo_u32 v21, v13, 6 // (op 1008) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 1008) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 1008) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1008) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 1008) 

v_mov_b32 v15, v12 // (op 1008) convert

s_mul_i32 s1, s29, 6 // (op 1010) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 1010) least significant
s_mul_hi_u32 s18, s28, 6 // (op 1010) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1010) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1010) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 1010) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1010) least significant
s_mul_hi_u32 s50, s30, s0 // (op 1010) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 1010) most significant: sum
s_add_u32 s19, s19, s50 // (op 1010) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1010) 

s_mov_b32 s52, s32 // (op 1010) reg expression
s_mov_b32 s53, s33 // (op 1010) reg expression
s_mov_b32 s54, s0 // (op 1010) reg expression
s_mov_b32 s55, 131072 // (op 1010) call()

s_mul_hi_i32 s0, s16, s12 // (op 1053) 
v_lshrrev_b32 v12, 7, v1 // (op 1053) 
s_add_i32 s1, s0, s16 // (op 1053) 

v_and_b32 v13, 15, v1 // (op 1053) 
s_ashr_i32 s0, s1, 31 // (op 1053) 
v_lshl_add_u32 v20, v12, 4, v13 // (op 1053) 


s_cmp_eq_i32 s12, 0 // (op 1053) 
s_mov_b32 s18, scc // (op 1053) 
v_and_b32 v12, 63, v1 // (op 1053) 
s_cmp_eq_i32 s18, 0 // (op 1053) 
s_cselect_b32 s19, 0, -1 // (op 1053) 

v_lshrrev_b32 v13, 4, v12 // (op 1053) 

s_and_b32 s18, s15, 31 // (op 1053) 
s_lshl_b32 s49, 1, s18 // (op 1053) 
s_add_i32 s50, s49, s19 // (op 1053) 


s_and_b32 s19, s0, s50 // (op 1053) 


s_add_i32 s0, s1, s19 // (op 1053) 


s_ashr_i32 s1, s0, s18 // (op 1053) 


s_lshl_b32 s0, s15, 24 // (op 1053) 
s_ashr_i32 s18, s0, 31 // (op 1053) 

s_xor_b32 s0, s1, s18 // (op 1053) 

s_sub_i32 s1, s0, s18 // (op 1053) 


s_mul_hi_i32 s0, s1, s44 // (op 1053) 
s_add_i32 s18, s0, s1 // (op 1053) 

s_ashr_i32 s0, s18, 31 // (op 1053) 
s_cmp_eq_i32 s44, 0 // (op 1053) 
s_mov_b32 s19, scc // (op 1053) 
s_cmp_eq_i32 s19, 0 // (op 1053) 
s_cselect_b32 s49, 0, -1 // (op 1053) 

s_and_b32 s19, s45, 31 // (op 1053) 
s_lshl_b32 s50, 1, s19 // (op 1053) 
s_add_i32 s51, s50, s49 // (op 1053) 


s_and_b32 s49, s0, s51 // (op 1053) 


s_add_i32 s0, s18, s49 // (op 1053) 


s_ashr_i32 s18, s0, s19 // (op 1053) 


s_lshl_b32 s0, s45, 24 // (op 1053) 
s_ashr_i32 s19, s0, 31 // (op 1053) 

s_xor_b32 s0, s18, s19 // (op 1053) 

s_sub_i32 s18, s0, s19 // (op 1053) 


s_mul_i32 s0, s1, s13 // (op 1053) 

s_sub_u32 s1, s16, s0 // (op 1053) 

s_mul_i32 s0, s18, s13 // (op 1053) 
s_add_u32 s0, s0, s1 // (op 1053) 


s_mul_hi_i32 s1, s16, s11 // (op 1053) 
s_add_i32 s18, s1, s16 // (op 1053) 

s_ashr_i32 s1, s18, 31 // (op 1053) 
s_cmp_eq_i32 s11, 0 // (op 1053) 
s_mov_b32 s19, scc // (op 1053) 
s_cmp_eq_i32 s19, 0 // (op 1053) 
s_cselect_b32 s49, 0, -1 // (op 1053) 

s_and_b32 s19, s47, 31 // (op 1053) 
s_lshl_b32 s50, 1, s19 // (op 1053) 
s_add_i32 s51, s50, s49 // (op 1053) 


s_and_b32 s49, s1, s51 // (op 1053) 


s_add_i32 s1, s18, s49 // (op 1053) 


s_ashr_i32 s18, s1, s19 // (op 1053) 


s_lshl_b32 s1, s47, 24 // (op 1053) 
s_ashr_i32 s19, s1, 31 // (op 1053) 

s_xor_b32 s1, s18, s19 // (op 1053) 

s_sub_i32 s18, s1, s19 // (op 1053) 


s_add_u32 s50, s8, 64 // (op 1053) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1053) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1053) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1053) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1053) 

s_mul_i32 s1, s13, s50 // (op 1053) 

s_mul_i32 s19, s10, s1 // (op 1053) 

s_mul_i32 s1, s18, s19 // (op 1053) 

s_sub_u32 s19, s16, s1 // (op 1053) 

s_mov_b32 s1, s19 // (op 1053) 
s_mul_hi_i32 s49, s1, s14 // (op 1053) 
s_add_i32 s50, s49, s1 // (op 1053) 


s_ashr_i32 s1, s50, 31 // (op 1053) 
s_cmp_eq_i32 s14, 0 // (op 1053) 
s_mov_b32 s49, scc // (op 1053) 
s_cmp_eq_i32 s49, 0 // (op 1053) 
s_cselect_b32 s51, 0, -1 // (op 1053) 

s_and_b32 s49, s46, 31 // (op 1053) 
s_lshl_b32 s64, 1, s49 // (op 1053) 
s_add_i32 s65, s64, s51 // (op 1053) 


s_and_b32 s51, s1, s65 // (op 1053) 


s_add_i32 s1, s50, s51 // (op 1053) 


s_ashr_i32 s50, s1, s49 // (op 1053) 


s_lshl_b32 s1, s46, 24 // (op 1053) 
s_ashr_i32 s49, s1, 31 // (op 1053) 

s_xor_b32 s1, s50, s49 // (op 1053) 

s_sub_i32 s50, s1, s49 // (op 1053) 


s_mul_i32 s1, s48, s13 // (op 1053) 
s_add_u32 s64, s6, 16 // (op 1053) Start of Int64 add, locking SCC
s_addc_u32 s65, s7, 0 // (op 1053) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1053) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1053) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 1053) 

s_sub_i32 s49, s64, s1 // (op 1053) 

s_mul_i32 s51, s50, s49 // (op 1053) 


s_sub_u32 s49, s19, s51 // (op 1053) 


s_add_u32 s19, s49, s1 // (op 1053) 


s_cmp_eq_i32 s18, 0 // (op 1053) 
s_mov_b32 s1, scc // (op 1053) 

s_cmp_eq_i32 s1, 0 // (op 1053) 
s_cselect_b32 s18, s19, s0 // (op 1053) 



v_lshl_add_u32 v12, s18, 4, v20 // (op 1053) 


v_mul_lo_u32 v20, v13, s34 // (op 1053) 

v_add_u32 v13, v12, v20 // (op 1053) 


s_mul_i32 s1, s34, s39 // (op 1055) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1055) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1055) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1055) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1055) most significant: sum
s_add_u32 s1, s1, s19 // (op 1055) most significant: sum


s_mov_b32 s64, s36 // (op 1055) reg expression
s_mov_b32 s65, s37 // (op 1055) reg expression
s_mov_b32 s66, s0 // (op 1055) reg expression
s_mov_b32 s67, 131072 // (op 1055) call()

s_mul_hi_i32 s0, s16, s12 // (op 1072) 
v_lshrrev_b32 v12, 7, v1 // (op 1072) 
s_add_i32 s1, s0, s16 // (op 1072) 

v_and_b32 v20, 15, v1 // (op 1072) 
s_ashr_i32 s0, s1, 31 // (op 1072) 
v_lshl_add_u32 v21, v12, 4, v20 // (op 1072) 


s_cmp_eq_i32 s12, 0 // (op 1072) 
s_mov_b32 s18, scc // (op 1072) 
v_and_b32 v12, 63, v1 // (op 1072) 
s_cmp_eq_i32 s18, 0 // (op 1072) 
s_cselect_b32 s19, 0, -1 // (op 1072) 

v_lshrrev_b32 v20, 4, v12 // (op 1072) 

s_and_b32 s18, s15, 31 // (op 1072) 
s_lshl_b32 s49, 1, s18 // (op 1072) 
s_add_i32 s50, s49, s19 // (op 1072) 


s_and_b32 s19, s0, s50 // (op 1072) 


s_add_i32 s0, s1, s19 // (op 1072) 


s_ashr_i32 s1, s0, s18 // (op 1072) 


s_lshl_b32 s0, s15, 24 // (op 1072) 
s_ashr_i32 s18, s0, 31 // (op 1072) 

s_xor_b32 s0, s1, s18 // (op 1072) 

s_sub_i32 s1, s0, s18 // (op 1072) 


s_mul_hi_i32 s0, s1, s44 // (op 1072) 
s_add_i32 s18, s0, s1 // (op 1072) 

s_ashr_i32 s0, s18, 31 // (op 1072) 
s_cmp_eq_i32 s44, 0 // (op 1072) 
s_mov_b32 s19, scc // (op 1072) 
s_cmp_eq_i32 s19, 0 // (op 1072) 
s_cselect_b32 s49, 0, -1 // (op 1072) 

s_and_b32 s19, s45, 31 // (op 1072) 
s_lshl_b32 s50, 1, s19 // (op 1072) 
s_add_i32 s51, s50, s49 // (op 1072) 


s_and_b32 s49, s0, s51 // (op 1072) 


s_add_i32 s0, s18, s49 // (op 1072) 


s_ashr_i32 s18, s0, s19 // (op 1072) 


s_lshl_b32 s0, s45, 24 // (op 1072) 
s_ashr_i32 s19, s0, 31 // (op 1072) 

s_xor_b32 s0, s18, s19 // (op 1072) 

s_sub_i32 s18, s0, s19 // (op 1072) 


s_mul_i32 s0, s1, s13 // (op 1072) 

s_sub_u32 s1, s16, s0 // (op 1072) 

s_mul_i32 s0, s18, s13 // (op 1072) 
s_add_u32 s0, s0, s1 // (op 1072) 


s_mul_hi_i32 s1, s16, s11 // (op 1072) 
s_add_i32 s18, s1, s16 // (op 1072) 

s_ashr_i32 s1, s18, 31 // (op 1072) 
s_cmp_eq_i32 s11, 0 // (op 1072) 
s_mov_b32 s19, scc // (op 1072) 
s_cmp_eq_i32 s19, 0 // (op 1072) 
s_cselect_b32 s49, 0, -1 // (op 1072) 

s_and_b32 s19, s47, 31 // (op 1072) 
s_lshl_b32 s50, 1, s19 // (op 1072) 
s_add_i32 s51, s50, s49 // (op 1072) 


s_and_b32 s49, s1, s51 // (op 1072) 


s_add_i32 s1, s18, s49 // (op 1072) 


s_ashr_i32 s18, s1, s19 // (op 1072) 


s_lshl_b32 s1, s47, 24 // (op 1072) 
s_ashr_i32 s19, s1, 31 // (op 1072) 

s_xor_b32 s1, s18, s19 // (op 1072) 

s_sub_i32 s18, s1, s19 // (op 1072) 


s_add_u32 s50, s8, 64 // (op 1072) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1072) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1072) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1072) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1072) 

s_mul_i32 s1, s13, s50 // (op 1072) 

s_mul_i32 s19, s10, s1 // (op 1072) 

s_mul_i32 s1, s18, s19 // (op 1072) 

s_sub_u32 s19, s16, s1 // (op 1072) 

s_mov_b32 s1, s19 // (op 1072) 
s_mul_hi_i32 s49, s1, s14 // (op 1072) 
s_add_i32 s50, s49, s1 // (op 1072) 


s_ashr_i32 s1, s50, 31 // (op 1072) 
s_cmp_eq_i32 s14, 0 // (op 1072) 
s_mov_b32 s49, scc // (op 1072) 
s_cmp_eq_i32 s49, 0 // (op 1072) 
s_cselect_b32 s51, 0, -1 // (op 1072) 

s_and_b32 s49, s46, 31 // (op 1072) 
s_lshl_b32 s68, 1, s49 // (op 1072) 
s_add_i32 s69, s68, s51 // (op 1072) 


s_and_b32 s51, s1, s69 // (op 1072) 


s_add_i32 s1, s50, s51 // (op 1072) 


s_ashr_i32 s50, s1, s49 // (op 1072) 


s_lshl_b32 s1, s46, 24 // (op 1072) 
s_ashr_i32 s49, s1, 31 // (op 1072) 

s_xor_b32 s1, s50, s49 // (op 1072) 

s_sub_i32 s50, s1, s49 // (op 1072) 


s_mul_i32 s1, s48, s13 // (op 1072) 
s_add_u32 s68, s6, 16 // (op 1072) Start of Int64 add, locking SCC
s_addc_u32 s69, s7, 0 // (op 1072) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1072) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1072) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 1072) 

s_sub_i32 s49, s68, s1 // (op 1072) 

s_mul_i32 s51, s50, s49 // (op 1072) 


s_sub_u32 s49, s19, s51 // (op 1072) 


s_add_u32 s19, s49, s1 // (op 1072) 


s_cmp_eq_i32 s18, 0 // (op 1072) 
s_mov_b32 s1, scc // (op 1072) 

s_cmp_eq_i32 s1, 0 // (op 1072) 
s_cselect_b32 s18, s19, s0 // (op 1072) 



v_lshl_add_u32 v12, s18, 4, v21 // (op 1072) 


v_mul_lo_u32 v21, v20, s34 // (op 1072) 

v_add_u32 v20, v12, v21 // (op 1072) 


s_mul_i32 s1, s34, s39 // (op 1074) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1074) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1074) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1074) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1074) most significant: sum
s_add_u32 s1, s1, s19 // (op 1074) most significant: sum


s_mov_b32 s64, s36 // (op 1074) reg expression
s_mov_b32 s65, s37 // (op 1074) reg expression
s_mov_b32 s66, s0 // (op 1074) reg expression
s_mov_b32 s67, 131072 // (op 1074) call()

s_mul_hi_i32 s0, s16, s12 // (op 1117) 
v_and_b32 v12, -64, v1 // (op 1117) 
s_add_i32 s1, s0, s16 // (op 1117) 

v_sub_u32 v21, v1, v12 // (op 1117) 

s_ashr_i32 s0, s1, 31 // (op 1117) 
v_lshrrev_b32 v12, 6, v1 // (op 1117) 
s_cmp_eq_i32 s12, 0 // (op 1117) 
s_mov_b32 s18, scc // (op 1117) 
v_lshlrev_b32 v22, 1, v12 // (op 1117) 

s_cmp_eq_i32 s18, 0 // (op 1117) 
s_cselect_b32 s19, 0, -1 // (op 1117) 

s_and_b32 s18, s15, 31 // (op 1117) 
s_lshl_b32 s49, 1, s18 // (op 1117) 
s_add_i32 s50, s49, s19 // (op 1117) 


s_and_b32 s19, s0, s50 // (op 1117) 


s_add_i32 s0, s1, s19 // (op 1117) 


s_ashr_i32 s1, s0, s18 // (op 1117) 


s_lshl_b32 s0, s15, 24 // (op 1117) 
s_ashr_i32 s18, s0, 31 // (op 1117) 

s_xor_b32 s0, s1, s18 // (op 1117) 

s_sub_i32 s1, s0, s18 // (op 1117) 


s_mul_hi_i32 s0, s1, s44 // (op 1117) 
s_add_i32 s18, s0, s1 // (op 1117) 

s_ashr_i32 s0, s18, 31 // (op 1117) 
s_cmp_eq_i32 s44, 0 // (op 1117) 
s_mov_b32 s19, scc // (op 1117) 
s_cmp_eq_i32 s19, 0 // (op 1117) 
s_cselect_b32 s49, 0, -1 // (op 1117) 

s_and_b32 s19, s45, 31 // (op 1117) 
s_lshl_b32 s50, 1, s19 // (op 1117) 
s_add_i32 s51, s50, s49 // (op 1117) 


s_and_b32 s49, s0, s51 // (op 1117) 


s_add_i32 s0, s18, s49 // (op 1117) 


s_ashr_i32 s18, s0, s19 // (op 1117) 


s_lshl_b32 s0, s45, 24 // (op 1117) 
s_ashr_i32 s19, s0, 31 // (op 1117) 

s_xor_b32 s0, s18, s19 // (op 1117) 

s_sub_i32 s18, s0, s19 // (op 1117) 


s_add_u32 s50, s8, 64 // (op 1117) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1117) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1117) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1117) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1117) 

s_mul_i32 s0, s18, s50 // (op 1117) 

s_sub_i32 s49, s1, s0 // (op 1117) 


s_mul_hi_i32 s0, s16, s11 // (op 1117) 
s_add_i32 s1, s0, s16 // (op 1117) 

s_ashr_i32 s0, s1, 31 // (op 1117) 
s_cmp_eq_i32 s11, 0 // (op 1117) 
s_mov_b32 s18, scc // (op 1117) 
s_cmp_eq_i32 s18, 0 // (op 1117) 
s_cselect_b32 s19, 0, -1 // (op 1117) 

s_and_b32 s18, s47, 31 // (op 1117) 
s_lshl_b32 s68, 1, s18 // (op 1117) 
s_add_i32 s69, s68, s19 // (op 1117) 


s_and_b32 s19, s0, s69 // (op 1117) 


s_add_i32 s0, s1, s19 // (op 1117) 


s_ashr_i32 s1, s0, s18 // (op 1117) 


s_lshl_b32 s0, s47, 24 // (op 1117) 
s_ashr_i32 s18, s0, 31 // (op 1117) 

s_xor_b32 s0, s1, s18 // (op 1117) 

s_sub_i32 s1, s0, s18 // (op 1117) 


s_mul_i32 s0, s13, s50 // (op 1117) 

s_mul_i32 s18, s10, s0 // (op 1117) 

s_mul_i32 s0, s1, s18 // (op 1117) 

s_sub_u32 s18, s16, s0 // (op 1117) 

s_mul_hi_i32 s0, s18, s14 // (op 1117) 
s_add_i32 s19, s0, s18 // (op 1117) 


s_ashr_i32 s0, s19, 31 // (op 1117) 
s_cmp_eq_i32 s14, 0 // (op 1117) 
s_mov_b32 s18, scc // (op 1117) 
s_cmp_eq_i32 s18, 0 // (op 1117) 
s_cselect_b32 s50, 0, -1 // (op 1117) 

s_and_b32 s18, s46, 31 // (op 1117) 
s_lshl_b32 s51, 1, s18 // (op 1117) 
s_add_i32 s68, s51, s50 // (op 1117) 


s_and_b32 s50, s0, s68 // (op 1117) 


s_add_i32 s0, s19, s50 // (op 1117) 


s_ashr_i32 s19, s0, s18 // (op 1117) 


s_lshl_b32 s0, s46, 24 // (op 1117) 
s_ashr_i32 s18, s0, 31 // (op 1117) 

s_xor_b32 s0, s19, s18 // (op 1117) 

s_sub_i32 s19, s0, s18 // (op 1117) 


s_cmp_eq_i32 s1, 0 // (op 1117) 
s_mov_b32 s0, scc // (op 1117) 

s_cmp_eq_i32 s0, 0 // (op 1117) 
s_cselect_b32 s1, s19, s49 // (op 1117) 



v_lshl_add_u32 v12, s1, 6, v21 // (op 1117) 


v_mul_lo_u32 v21, v22, s40 // (op 1117) 
v_add_u32 v21, v21, v12 // (op 1117) 


s_mul_i32 s1, s40, s39 // (op 1119) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1119) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1119) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1119) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1119) most significant: sum
s_add_u32 s1, s1, s19 // (op 1119) most significant: sum


s_mov_b32 s68, s42 // (op 1119) reg expression
s_mov_b32 s69, s43 // (op 1119) reg expression
s_mov_b32 s70, s0 // (op 1119) reg expression
s_mov_b32 s71, 131072 // (op 1119) call()

s_mul_hi_i32 s0, s16, s12 // (op 1136) 
v_and_b32 v12, -64, v1 // (op 1136) 
s_add_i32 s1, s0, s16 // (op 1136) 

v_sub_u32 v22, v1, v12 // (op 1136) 

s_ashr_i32 s0, s1, 31 // (op 1136) 
v_lshrrev_b32 v12, 6, v1 // (op 1136) 
s_cmp_eq_i32 s12, 0 // (op 1136) 
s_mov_b32 s18, scc // (op 1136) 
v_lshlrev_b32 v23, 1, v12 // (op 1136) 

s_cmp_eq_i32 s18, 0 // (op 1136) 
s_cselect_b32 s19, 0, -1 // (op 1136) 

s_and_b32 s18, s15, 31 // (op 1136) 
s_lshl_b32 s49, 1, s18 // (op 1136) 
s_add_i32 s50, s49, s19 // (op 1136) 


s_and_b32 s19, s0, s50 // (op 1136) 


s_add_i32 s0, s1, s19 // (op 1136) 


s_ashr_i32 s1, s0, s18 // (op 1136) 


s_lshl_b32 s0, s15, 24 // (op 1136) 
s_ashr_i32 s18, s0, 31 // (op 1136) 

s_xor_b32 s0, s1, s18 // (op 1136) 

s_sub_i32 s1, s0, s18 // (op 1136) 


s_mul_hi_i32 s0, s1, s44 // (op 1136) 
s_add_i32 s18, s0, s1 // (op 1136) 

s_ashr_i32 s0, s18, 31 // (op 1136) 
s_cmp_eq_i32 s44, 0 // (op 1136) 
s_mov_b32 s19, scc // (op 1136) 
s_cmp_eq_i32 s19, 0 // (op 1136) 
s_cselect_b32 s49, 0, -1 // (op 1136) 

s_and_b32 s19, s45, 31 // (op 1136) 
s_lshl_b32 s50, 1, s19 // (op 1136) 
s_add_i32 s51, s50, s49 // (op 1136) 


s_and_b32 s49, s0, s51 // (op 1136) 


s_add_i32 s0, s18, s49 // (op 1136) 


s_ashr_i32 s18, s0, s19 // (op 1136) 


s_lshl_b32 s0, s45, 24 // (op 1136) 
s_ashr_i32 s19, s0, 31 // (op 1136) 

s_xor_b32 s0, s18, s19 // (op 1136) 

s_sub_i32 s18, s0, s19 // (op 1136) 


s_add_u32 s50, s8, 64 // (op 1136) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1136) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1136) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1136) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1136) 

s_mul_i32 s0, s18, s50 // (op 1136) 

s_sub_i32 s49, s1, s0 // (op 1136) 


s_mul_hi_i32 s0, s16, s11 // (op 1136) 
s_add_i32 s1, s0, s16 // (op 1136) 

s_ashr_i32 s0, s1, 31 // (op 1136) 
s_cmp_eq_i32 s11, 0 // (op 1136) 
s_mov_b32 s18, scc // (op 1136) 
s_cmp_eq_i32 s18, 0 // (op 1136) 
s_cselect_b32 s19, 0, -1 // (op 1136) 

s_and_b32 s18, s47, 31 // (op 1136) 
s_lshl_b32 s72, 1, s18 // (op 1136) 
s_add_i32 s73, s72, s19 // (op 1136) 


s_and_b32 s19, s0, s73 // (op 1136) 


s_add_i32 s0, s1, s19 // (op 1136) 


s_ashr_i32 s1, s0, s18 // (op 1136) 


s_lshl_b32 s0, s47, 24 // (op 1136) 
s_ashr_i32 s18, s0, 31 // (op 1136) 

s_xor_b32 s0, s1, s18 // (op 1136) 

s_sub_i32 s1, s0, s18 // (op 1136) 


s_mul_i32 s0, s13, s50 // (op 1136) 

s_mul_i32 s18, s10, s0 // (op 1136) 

s_mul_i32 s0, s1, s18 // (op 1136) 

s_sub_u32 s18, s16, s0 // (op 1136) 

s_mul_hi_i32 s0, s18, s14 // (op 1136) 
s_add_i32 s19, s0, s18 // (op 1136) 


s_ashr_i32 s0, s19, 31 // (op 1136) 
s_cmp_eq_i32 s14, 0 // (op 1136) 
s_mov_b32 s18, scc // (op 1136) 
s_cmp_eq_i32 s18, 0 // (op 1136) 
s_cselect_b32 s50, 0, -1 // (op 1136) 

s_and_b32 s18, s46, 31 // (op 1136) 
s_lshl_b32 s51, 1, s18 // (op 1136) 
s_add_i32 s72, s51, s50 // (op 1136) 


s_and_b32 s50, s0, s72 // (op 1136) 


s_add_i32 s0, s19, s50 // (op 1136) 


s_ashr_i32 s19, s0, s18 // (op 1136) 


s_lshl_b32 s0, s46, 24 // (op 1136) 
s_ashr_i32 s18, s0, 31 // (op 1136) 

s_xor_b32 s0, s19, s18 // (op 1136) 

s_sub_i32 s19, s0, s18 // (op 1136) 


s_cmp_eq_i32 s1, 0 // (op 1136) 
s_mov_b32 s0, scc // (op 1136) 

s_cmp_eq_i32 s0, 0 // (op 1136) 
s_cselect_b32 s1, s19, s49 // (op 1136) 



v_lshl_add_u32 v12, s1, 6, v22 // (op 1136) 


v_mul_lo_u32 v22, v23, s40 // (op 1136) 
v_add_u32 v22, v22, v12 // (op 1136) 


s_mul_i32 s1, s40, s39 // (op 1138) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1138) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1138) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1138) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1138) most significant: sum
s_add_u32 s1, s1, s19 // (op 1138) most significant: sum


s_mov_b32 s68, s42 // (op 1138) reg expression
s_mov_b32 s69, s43 // (op 1138) reg expression
s_mov_b32 s70, s0 // (op 1138) reg expression
s_mov_b32 s71, 131072 // (op 1138) call()


v_mov_b64 v[24:25], s[34:35] // (op 769) convert

v_mov_b32 v12, v13 // (op 769) reg expression
v_mov_b32 v23, v12 // (op 769) 
buffer_load_ubyte v26, v23, s[64:67], 0 offen offset: 0 // (op 769) Load value





v_mov_b32 v12, v14 // (op 773) reg expression

buffer_load_dwordx3 v[28:30], v12, s[52:55], 0 offen // (op 773) Load value
buffer_load_dwordx3 v[32:34], v12, s[52:55], 0 offen offset:12 // (op 773) Load value
buffer_load_dwordx3 v[36:38], v12, s[52:55], 0 offen offset:24 // (op 773) Load value
buffer_load_dwordx3 v[40:42], v12, s[52:55], 0 offen offset:36 // (op 773) Load value


v_mov_b64 v[24:25], s[40:41] // (op 776) convert

v_mov_b32 v12, v21 // (op 776) reg expression
buffer_load_ubyte v44, v12, s[68:71], 0 offen offset: 0 // (op 776) Load value
v_add_u32 v12, v12, v24 // (op 776) 
buffer_load_ubyte v45, v12, s[68:71], 0 offen offset: 0 // (op 776) Load value


v_lshlrev_b32 v12, 4, v1 // (op 1202) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1202) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[24:25], v[16:17] // (op 880) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 880) reg expression
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 880) Load value direct to lds



v_mov_b64 v[24:25], s[34:35] // (op 783) convert

s_lshl_b32 s0, s34, 2 // (op 783) 
v_add_u32 v12, v13, s0 // (op 783) 

v_mov_b32 v23, v12 // (op 783) 
buffer_load_ubyte v27, v23, s[64:67], 0 offen offset: 0 // (op 783) Load value






s_lshl_b64 s[0:1], s[30:31], 7 // (op 787) 
s_mul_i32 s19, s1, 6 // (op 787) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 787) least significant
s_mul_hi_u32 s49, s0, 6 // (op 787) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 787) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 787) 

v_add_u32 v31, v14, s0 // (op 787) 


buffer_load_dwordx3 v[46:48], v31, s[52:55], 0 offen // (op 787) Load value
buffer_load_dwordx3 v[50:52], v31, s[52:55], 0 offen offset:12 // (op 787) Load value
buffer_load_dwordx3 v[54:56], v31, s[52:55], 0 offen offset:24 // (op 787) Load value
buffer_load_dwordx3 v[58:60], v31, s[52:55], 0 offen offset:36 // (op 787) Load value



v_mov_b64 v[24:25], s[40:41] // (op 790) convert

s_lshl_b32 s0, s40, 2 // (op 790) 
v_add_u32 v23, v21, s0 // (op 790) 

buffer_load_ubyte v12, v23, s[68:71], 0 offen offset: 0 // (op 790) Load value
v_add_u32 v23, v23, v24 // (op 790) 
buffer_load_ubyte v13, v23, s[68:71], 0 offen offset: 0 // (op 790) Load value



v_lshlrev_b32 v14, 4, v1 // (op 1375) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 1375) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[18:19], s[24:25], 7 // (op 885) 
v_mov_b32 v14, s18 // (op 885) 
v_mov_b32 v21, s19 // (op 885) 
v_add_co_u32 v24, s[50:51], v16, v14 // (op 885) least significant half
v_addc_co_u32 v25, s[50:51], v17, v21, s[50:51] // (op 885) most significant half




s_add_u32 s1, 2048, s0 // (op 885) 

s_mov_b32 m0, s1 // (op 885) reg expression
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 885) Load value direct to lds




v_lshlrev_b32 v14, 6, v1 // (op 1255) 
v_mul_lo_u32 v21, v14, 6 // (op 1255) 
v_lshrrev_b32 v31, 3, v21 // (op 1255) 

v_lshrrev_b32 v21, 4, v14 // (op 1255) 

v_lshlrev_b32 v14, 2, v21 // (op 1255) 

v_add_u32 v21, v31, v14 // (op 1255) 





s_waitcnt vmcnt(14)
ds_write_b96 v21, v[28:30] offset:4096 // (op 795) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v21, v[32:34] offset:4112 // (op 795) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v21, v[36:38] offset:4128 // (op 795) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v21, v[40:42] offset:4144 // (op 795) Store local data 

v_and_b32 v14, -64, v1 // (op 1313) 
v_sub_u32 v21, v1, v14 // (op 1313) 

v_lshrrev_b32 v14, 6, v1 // (op 1313) 
v_lshl_add_u32 v31, v14, 7, v21 // (op 1313) 





s_waitcnt vmcnt(10)
ds_write_b8 v31, v44 offset:12288 // (op 796) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v31, v45 offset:12352 // (op 796) Store local data 

v_lshlrev_b32 v14, 4, v1 // (op 1174) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 1174) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 6, v1 // (op 1226) 
v_mul_lo_u32 v21, v14, 6 // (op 1226) 
v_lshrrev_b32 v31, 3, v21 // (op 1226) 

v_lshrrev_b32 v21, 4, v14 // (op 1226) 

v_lshlrev_b32 v14, 2, v21 // (op 1226) 

v_add_u32 v21, v31, v14 // (op 1226) 


v_and_b32 v14, -64, v1 // (op 1284) 
v_sub_u32 v31, v1, v14 // (op 1284) 

v_lshrrev_b32 v14, 6, v1 // (op 1284) 
v_lshl_add_u32 v35, v14, 7, v31 // (op 1284) 


v_lshlrev_b32 v14, 4, v1 // (op 1360) 
s_nop 0
v_readfirstlane_b32 s1, v14 // (op 1360) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 6, v1 // (op 1399) 
v_mul_lo_u32 v31, v14, 6 // (op 1399) 
v_lshrrev_b32 v39, 3, v31 // (op 1399) 

v_lshrrev_b32 v31, 4, v14 // (op 1399) 

v_lshlrev_b32 v14, 2, v31 // (op 1399) 

v_add_u32 v31, v39, v14 // (op 1399) 


v_and_b32 v14, -64, v1 // (op 1428) 
v_sub_u32 v39, v1, v14 // (op 1428) 

v_lshrrev_b32 v14, 6, v1 // (op 1428) 
v_lshl_add_u32 v43, v14, 7, v39 // (op 1428) 


s_mov_b32 s18, 0 // (op 312) call()
s_ashr_i64 s[50:51], s[28:29], 7 // (op 311) 
s_mov_b32 s19, s50 // (op 311) convert

s_and_b32 s49, s19, 4294967294 // (op 311) 

s_cmp_lt_u32 s18, s49 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 if false)
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_311:
s_add_u32 s18, s18, 2 // (op 313) 
s_waitcnt vmcnt(8)
s_barrier  // (op 1486) 



ds_read_b64_tr_b8 v[62:63], v0 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:128 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v0 offset:1024 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:1152 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1499) 



ds_read_b96_tr_b6 v[70:72], v4 offset:4096 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v4 offset:5120 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1513) 


ds_read_u8 v14, v7 offset:12288 // (op 81) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[70:75], a[0:3], v26, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 84) 
ds_read_b96_tr_b6 v[76:78], v4 offset:4128 // (op 343) Transpose load local data 


ds_read_b96_tr_b6 v[79:81], v4 offset:5152 // (op 343) Transpose load local data 


ds_read_u8 v14, v7 offset:12320 // (op 346) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[76:81], a[4:7], v26, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 347) 




v_mov_b64 v[16:17], s[34:35] // (op 77) convert

v_mov_b32 v14, v20 // (op 77) 
s_lshl_b32 s19, s34, 3 // (op 77) 
v_add_u32 v39, v14, s19 // (op 77) 

v_mov_b32 v49, v39 // (op 77) 
buffer_load_ubyte v26, v49, s[64:67], 0 offen offset: 0 // (op 77) Load value





v_mov_b32 v39, v15 // (op 10) 
s_lshl_b64 s[50:51], s[30:31], 7 // (op 10) 
s_mul_i32 s73, s51, 6 // (op 10) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 10) least significant
s_mul_hi_u32 s19, s50, 6 // (op 10) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 10) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 10) 

s_lshl_b32 s19, s50, 1 // (op 10) 

v_add_u32 v49, v39, s19 // (op 10) 


buffer_load_dwordx3 v[28:30], v49, s[52:55], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[32:34], v49, s[52:55], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v49, s[52:55], 0 offen offset:24 // (op 10) Load value
buffer_load_dwordx3 v[40:42], v49, s[52:55], 0 offen offset:36 // (op 10) Load value


v_mov_b64 v[16:17], s[40:41] // (op 21) convert

v_mov_b32 v49, v22 // (op 21) 
s_lshl_b32 s19, s40, 3 // (op 21) 
v_add_u32 v53, v49, s19 // (op 21) 

buffer_load_ubyte v44, v53, s[68:71], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v53, v53, v16 // (op 21) 
buffer_load_ubyte v45, v53, s[68:71], 0 offen offset: 0 // (op 21) Load value


s_barrier  // (op 1489) 


v_mov_b64 v[16:17], v[18:19] // (op 864) 
s_lshl_b64 s[50:51], s[24:25], 7 // (op 864) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 864) 

v_mov_b32 v53, s72 // (op 864) 
v_mov_b32 v57, s73 // (op 864) 
v_add_co_u32 v24, s[50:51], v16, v53 // (op 864) least significant half
v_addc_co_u32 v25, s[50:51], v17, v57, s[50:51] // (op 864) most significant half





s_mov_b32 m0, s0 // (op 864) reg expression
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 864) Load value direct to lds

s_barrier  // (op 1536) 



s_waitcnt vmcnt(14)
ds_write_b96 v31, v[46:48] offset:12544 // (op 239) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v31, v[50:52] offset:12560 // (op 239) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v31, v[54:56] offset:12576 // (op 239) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v31, v[58:60] offset:12592 // (op 239) Store local data 
s_barrier  // (op 1542) 



s_waitcnt vmcnt(10)
ds_write_b8 v43, v12 offset:20736 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v43, v13 offset:20800 // (op 240) Store local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 1527) 



ds_read_b64_tr_b8 v[62:63], v3 offset:2048 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v3 offset:2176 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v3 offset:3072 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v3 offset:3200 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1533) 



ds_read_b96_tr_b6 v[70:72], v10 offset:12544 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v10 offset:13568 // (op 233) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1539) 


ds_read_u8 v53, v11 offset:20736 // (op 237) Load local data 



ds_read_b96_tr_b6 v[76:78], v10 offset:12576 // (op 352) Transpose load local data 
ds_read_b96_tr_b6 v[79:81], v10 offset:13600 // (op 352) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:69], v[70:75], a[0:3], v27, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 238) 




s_barrier  // (op 1607) 
ds_read_u8 v53, v11 offset:20768 // (op 355) Load local data 
s_barrier  // (op 1611) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:69], v[76:81], a[4:7], v27, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 348) 




v_mov_b64 v[24:25], s[34:35] // (op 235) convert

s_lshl_b32 s19, s34, 2 // (op 235) 
s_mul_i32 s49, 3, s19 // (op 235) 

v_add_u32 v23, v14, s49 // (op 235) 

v_mov_b32 v53, v23 // (op 235) 
buffer_load_ubyte v27, v53, s[64:67], 0 offen offset: 0 // (op 235) Load value






s_lshl_b32 s19, s34, 3 // (op 1064) 
v_add_u32 v20, v20, s19 // (op 1064) 

s_lshl_b64 s[50:51], s[30:31], 7 // (op 241) 
s_mul_i32 s73, s51, 6 // (op 241) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 241) least significant
s_mul_hi_u32 s19, s50, 6 // (op 241) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 241) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 241) 

s_mul_i32 s19, 3, s50 // (op 241) 

v_add_u32 v14, v39, s19 // (op 241) 


buffer_load_dwordx3 v[46:48], v14, s[52:55], 0 offen // (op 241) Load value
buffer_load_dwordx3 v[50:52], v14, s[52:55], 0 offen offset:12 // (op 241) Load value
buffer_load_dwordx3 v[54:56], v14, s[52:55], 0 offen offset:24 // (op 241) Load value
buffer_load_dwordx3 v[58:60], v14, s[52:55], 0 offen offset:36 // (op 241) Load value



v_mov_b64 v[24:25], s[40:41] // (op 242) convert

s_lshl_b32 s19, s40, 2 // (op 242) 
s_mul_i32 s49, 3, s19 // (op 242) 

v_add_u32 v14, v49, s49 // (op 242) 

s_lshl_b64 s[50:51], s[30:31], 7 // (op 1000) 
s_mul_i32 s73, s51, 6 // (op 1000) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 1000) least significant
s_mul_hi_u32 s19, s50, 6 // (op 1000) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 1000) most significant: sum


buffer_load_ubyte v12, v14, s[68:71], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v14, v14, v24 // (op 242) 
s_lshr_b64 s[50:51], s[72:73], 3 // (op 1000) 

s_lshl_b32 s19, s50, 1 // (op 1000) 

v_add_u32 v15, v15, s19 // (op 1000) 

buffer_load_ubyte v13, v14, s[68:71], 0 offen offset: 0 // (op 242) Load value



s_barrier  // (op 1530) 
s_lshl_b32 s19, s40, 3 // (op 1128) 


v_add_u32 v22, v22, s19 // (op 1128) 

s_lshl_b64 s[50:51], s[24:25], 7 // (op 875) 
s_mul_i32 s73, 3, s51 // (op 875) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 875) least significant
s_mul_hi_u32 s19, 3, s50 // (op 875) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 875) most significant: sum


v_mov_b32 v14, s72 // (op 875) 
v_mov_b32 v39, s73 // (op 875) 
v_add_co_u32 v24, s[50:51], v16, v14 // (op 875) least significant half
v_addc_co_u32 v25, s[50:51], v17, v39, s[50:51] // (op 875) most significant half




s_add_u32 s19, 2048, s1 // (op 875) 

s_mov_b32 m0, s19 // (op 875) reg expression
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 875) Load value direct to lds



s_barrier  // (op 1502) 
s_lshl_b64 s[50:51], s[24:25], 7 // (op 936) 



s_lshl_b64 s[72:73], s[50:51], 1 // (op 936) 

v_mov_b32 v14, s72 // (op 936) 
v_mov_b32 v39, s73 // (op 936) 
v_add_co_u32 v18, s[50:51], v18, v14 // (op 936) least significant half
v_addc_co_u32 v19, s[50:51], v19, v39, s[50:51] // (op 936) most significant half




s_waitcnt vmcnt(14)
ds_write_b96 v21, v[28:30] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v21, v[32:34] offset:4112 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v21, v[36:38] offset:4128 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v21, v[40:42] offset:4144 // (op 13) Store local data 
s_barrier  // (op 1516) 



s_waitcnt vmcnt(10)
ds_write_b8 v35, v44 offset:12288 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v35, v45 offset:12352 // (op 26) Store local data 
s_ashr_i64 s[50:51], s[28:29], 7 // (op 311) 
s_mov_b32 s19, s50 // (op 311) convert

s_and_b32 s49, s19, 4294967294 // (op 311) 

s_cmp_lt_u32 s18, s49 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoop_311 if true)
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoop_311:














s_barrier  // (op 1591) 
s_mul_hi_i32 s0, s16, s12 // (op 898) 
s_add_i32 s1, s0, s16 // (op 898) 

s_ashr_i32 s0, s1, 31 // (op 898) 
s_cmp_eq_i32 s12, 0 // (op 898) 
s_mov_b32 s18, scc // (op 898) 
s_cmp_eq_i32 s18, 0 // (op 898) 
s_cselect_b32 s19, 0, -1 // (op 898) 

s_and_b32 s18, s15, 31 // (op 898) 
s_lshl_b32 s49, 1, s18 // (op 898) 
s_add_i32 s50, s49, s19 // (op 898) 


s_and_b32 s19, s0, s50 // (op 898) 


s_add_i32 s0, s1, s19 // (op 898) 


s_ashr_i32 s1, s0, s18 // (op 898) 


s_lshl_b32 s0, s15, 24 // (op 898) 
s_ashr_i32 s18, s0, 31 // (op 898) 

s_xor_b32 s0, s1, s18 // (op 898) 

s_sub_i32 s1, s0, s18 // (op 898) 


s_mul_hi_i32 s0, s1, s44 // (op 898) 
s_add_i32 s18, s0, s1 // (op 898) 

s_ashr_i32 s0, s18, 31 // (op 898) 
s_cmp_eq_i32 s44, 0 // (op 898) 
s_mov_b32 s19, scc // (op 898) 
s_cmp_eq_i32 s19, 0 // (op 898) 
s_cselect_b32 s49, 0, -1 // (op 898) 

s_and_b32 s19, s45, 31 // (op 898) 
s_lshl_b32 s50, 1, s19 // (op 898) 
s_add_i32 s51, s50, s49 // (op 898) 


s_and_b32 s49, s0, s51 // (op 898) 


s_add_i32 s0, s18, s49 // (op 898) 


s_ashr_i32 s18, s0, s19 // (op 898) 


s_lshl_b32 s0, s45, 24 // (op 898) 
s_ashr_i32 s19, s0, 31 // (op 898) 

s_xor_b32 s0, s18, s19 // (op 898) 

s_sub_i32 s18, s0, s19 // (op 898) 


s_mul_i32 s0, s1, s13 // (op 898) 

s_sub_u32 s1, s16, s0 // (op 898) 

s_mul_i32 s0, s18, s13 // (op 898) 
s_add_u32 s0, s0, s1 // (op 898) 


s_mul_hi_i32 s1, s16, s11 // (op 898) 
s_add_i32 s18, s1, s16 // (op 898) 

s_ashr_i32 s1, s18, 31 // (op 898) 
s_cmp_eq_i32 s11, 0 // (op 898) 
s_mov_b32 s19, scc // (op 898) 
s_cmp_eq_i32 s19, 0 // (op 898) 
s_cselect_b32 s49, 0, -1 // (op 898) 

s_and_b32 s19, s47, 31 // (op 898) 
s_lshl_b32 s50, 1, s19 // (op 898) 
s_add_i32 s51, s50, s49 // (op 898) 


s_and_b32 s49, s1, s51 // (op 898) 


s_add_i32 s1, s18, s49 // (op 898) 


s_ashr_i32 s18, s1, s19 // (op 898) 


s_lshl_b32 s1, s47, 24 // (op 898) 
s_ashr_i32 s19, s1, 31 // (op 898) 

s_xor_b32 s1, s18, s19 // (op 898) 

s_sub_i32 s18, s1, s19 // (op 898) 


s_add_u32 s50, s8, 64 // (op 898) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 898) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 898) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 898) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 898) 

s_mul_i32 s1, s13, s50 // (op 898) 

s_mul_i32 s19, s10, s1 // (op 898) 

s_mul_i32 s1, s18, s19 // (op 898) 

s_sub_u32 s19, s16, s1 // (op 898) 

s_mov_b32 s1, s19 // (op 898) 
s_mul_hi_i32 s49, s1, s14 // (op 898) 
s_add_i32 s50, s49, s1 // (op 898) 


s_ashr_i32 s1, s50, 31 // (op 898) 
s_cmp_eq_i32 s14, 0 // (op 898) 
s_mov_b32 s49, scc // (op 898) 
s_cmp_eq_i32 s49, 0 // (op 898) 
s_cselect_b32 s51, 0, -1 // (op 898) 

s_and_b32 s49, s46, 31 // (op 898) 
s_lshl_b32 s72, 1, s49 // (op 898) 
s_add_i32 s73, s72, s51 // (op 898) 


s_and_b32 s51, s1, s73 // (op 898) 


s_add_i32 s1, s50, s51 // (op 898) 


s_ashr_i32 s50, s1, s49 // (op 898) 


s_lshl_b32 s1, s46, 24 // (op 898) 
s_ashr_i32 s49, s1, 31 // (op 898) 

s_xor_b32 s1, s50, s49 // (op 898) 

s_sub_i32 s50, s1, s49 // (op 898) 


s_mul_i32 s1, s48, s13 // (op 898) 
s_add_u32 s72, s6, 16 // (op 898) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 898) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 898) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 898) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 898) 

s_sub_i32 s49, s72, s1 // (op 898) 

s_mul_i32 s51, s50, s49 // (op 898) 


s_sub_u32 s49, s19, s51 // (op 898) 


s_add_u32 s19, s49, s1 // (op 898) 


s_cmp_eq_i32 s18, 0 // (op 898) 
s_mov_b32 s1, scc // (op 898) 

s_cmp_eq_i32 s1, 0 // (op 898) 
s_cselect_b32 s18, s19, s0 // (op 898) 



s_lshl_b32 s19, s18, 4 // (op 898) 

s_mov_b32 s1, 0 // (op 898) convert
s_mov_b32 s0, s19 // (op 898) convert

s_ashr_i64 s[18:19], s[28:29], 7 // (op 898) 
s_lshr_b32 s49, s18, 1 // (op 898) 

v_lshl_add_u32 v49, s49, 8, v1 // (op 898) 

v_mul_lo_u32 v15, v49, s25 // (op 898) most significant: low of low * high
v_mul_hi_u32 v53, v49, s24 // (op 898) most significant: high of low * low
v_mul_lo_u32 v14, v49, s24 // (op 898) least significant: low of low * low
v_add_u32 v15, v15, v53 // (op 898) most significant: sum


v_mov_b32 v49, s0 // (op 898) 
v_mov_b32 v53, s1 // (op 898) 
v_add_co_u32 v16, s[18:19], v49, v14 // (op 898) least significant half
v_addc_co_u32 v17, s[18:19], v53, v15, s[18:19] // (op 898) most significant half





s_mul_i32 s1, s24, s29 // (op 900) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 900) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 900) least significant
s_mul_hi_u32 s19, s24, s28 // (op 900) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 900) most significant: sum
s_add_u32 s1, s1, s19 // (op 900) most significant: sum


s_mov_b32 s20, s26 // (op 900) reg expression
s_mov_b32 s21, s27 // (op 900) reg expression
s_mov_b32 s22, s0 // (op 900) reg expression
s_mov_b32 s23, 131072 // (op 900) call()

s_mul_hi_i32 s0, s16, s12 // (op 966) 
s_add_i32 s1, s0, s16 // (op 966) 

s_ashr_i32 s0, s1, 31 // (op 966) 
s_cmp_eq_i32 s12, 0 // (op 966) 
s_mov_b32 s18, scc // (op 966) 
s_cmp_eq_i32 s18, 0 // (op 966) 
s_cselect_b32 s19, 0, -1 // (op 966) 

s_and_b32 s18, s15, 31 // (op 966) 
s_lshl_b32 s49, 1, s18 // (op 966) 
s_add_i32 s50, s49, s19 // (op 966) 


s_and_b32 s19, s0, s50 // (op 966) 


s_add_i32 s0, s1, s19 // (op 966) 


s_ashr_i32 s1, s0, s18 // (op 966) 


s_lshl_b32 s0, s15, 24 // (op 966) 
s_ashr_i32 s18, s0, 31 // (op 966) 

s_xor_b32 s0, s1, s18 // (op 966) 

s_sub_i32 s1, s0, s18 // (op 966) 


s_mul_hi_i32 s0, s1, s44 // (op 966) 
s_add_i32 s18, s0, s1 // (op 966) 

s_ashr_i32 s0, s18, 31 // (op 966) 
s_cmp_eq_i32 s44, 0 // (op 966) 
s_mov_b32 s19, scc // (op 966) 
s_cmp_eq_i32 s19, 0 // (op 966) 
s_cselect_b32 s49, 0, -1 // (op 966) 

s_and_b32 s19, s45, 31 // (op 966) 
s_lshl_b32 s50, 1, s19 // (op 966) 
s_add_i32 s51, s50, s49 // (op 966) 


s_and_b32 s49, s0, s51 // (op 966) 


s_add_i32 s0, s18, s49 // (op 966) 


s_ashr_i32 s18, s0, s19 // (op 966) 


s_lshl_b32 s0, s45, 24 // (op 966) 
s_ashr_i32 s19, s0, 31 // (op 966) 

s_xor_b32 s0, s18, s19 // (op 966) 

s_sub_i32 s18, s0, s19 // (op 966) 


s_add_u32 s50, s8, 64 // (op 966) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 966) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 966) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 966) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 966) 

s_mul_i32 s0, s18, s50 // (op 966) 

s_sub_i32 s49, s1, s0 // (op 966) 


s_mul_hi_i32 s0, s16, s11 // (op 966) 
s_add_i32 s1, s0, s16 // (op 966) 

s_ashr_i32 s0, s1, 31 // (op 966) 
s_cmp_eq_i32 s11, 0 // (op 966) 
s_mov_b32 s18, scc // (op 966) 
s_cmp_eq_i32 s18, 0 // (op 966) 
s_cselect_b32 s19, 0, -1 // (op 966) 

s_and_b32 s18, s47, 31 // (op 966) 
s_lshl_b32 s72, 1, s18 // (op 966) 
s_add_i32 s73, s72, s19 // (op 966) 


s_and_b32 s19, s0, s73 // (op 966) 


s_add_i32 s0, s1, s19 // (op 966) 


s_ashr_i32 s1, s0, s18 // (op 966) 


s_lshl_b32 s0, s47, 24 // (op 966) 
s_ashr_i32 s18, s0, 31 // (op 966) 

s_xor_b32 s0, s1, s18 // (op 966) 

s_sub_i32 s1, s0, s18 // (op 966) 


s_mul_i32 s0, s13, s50 // (op 966) 

s_mul_i32 s18, s10, s0 // (op 966) 

s_mul_i32 s0, s1, s18 // (op 966) 

s_sub_u32 s18, s16, s0 // (op 966) 

s_mul_hi_i32 s0, s18, s14 // (op 966) 
s_add_i32 s19, s0, s18 // (op 966) 


s_ashr_i32 s0, s19, 31 // (op 966) 
s_cmp_eq_i32 s14, 0 // (op 966) 
s_mov_b32 s18, scc // (op 966) 
s_cmp_eq_i32 s18, 0 // (op 966) 
s_cselect_b32 s50, 0, -1 // (op 966) 

s_and_b32 s18, s46, 31 // (op 966) 
s_lshl_b32 s51, 1, s18 // (op 966) 
s_add_i32 s72, s51, s50 // (op 966) 


s_and_b32 s50, s0, s72 // (op 966) 


s_add_i32 s0, s19, s50 // (op 966) 


s_ashr_i32 s19, s0, s18 // (op 966) 


s_lshl_b32 s0, s46, 24 // (op 966) 
s_ashr_i32 s18, s0, 31 // (op 966) 

s_xor_b32 s0, s19, s18 // (op 966) 

s_sub_i32 s19, s0, s18 // (op 966) 


s_cmp_eq_i32 s1, 0 // (op 966) 
s_mov_b32 s0, scc // (op 966) 

s_cmp_eq_i32 s0, 0 // (op 966) 
s_cselect_b32 s1, s19, s49 // (op 966) 



s_lshl_b32 s0, s1, 6 // (op 966) 

s_mov_b32 s19, 0 // (op 966) convert
s_mov_b32 s18, s0 // (op 966) convert

s_ashr_i64 s[0:1], s[28:29], 7 // (op 966) 
s_lshr_b32 s49, s0, 1 // (op 966) 

v_lshl_add_u32 v49, s49, 8, v1 // (op 966) 

v_mul_lo_u32 v15, v49, s31 // (op 966) most significant: low of low * high
v_mul_hi_u32 v53, v49, s30 // (op 966) most significant: high of low * low
v_mul_lo_u32 v14, v49, s30 // (op 966) least significant: low of low * low
v_add_u32 v15, v15, v53 // (op 966) most significant: sum

v_mov_b32 v53, s18 // (op 966) 
v_mov_b32 v57, s19 // (op 966) 
v_add_co_u32 v14, s[0:1], v14, v53 // (op 966) least significant half
v_addc_co_u32 v15, s[0:1], v15, v57, s[0:1] // (op 966) most significant half





v_mul_lo_u32 v19, v15, 6 // (op 966) most significant: low of high * low
v_mul_hi_u32 v49, v14, 6 // (op 966) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 966) least significant: low of low * low
v_add_u32 v19, v19, v49 // (op 966) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 966) 

v_mov_b32 v49, v14 // (op 966) convert

s_mul_i32 s1, s29, 6 // (op 968) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 968) least significant
s_mul_hi_u32 s18, s28, 6 // (op 968) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 968) most significant: sum

s_mul_i32 s19, s30, s1 // (op 968) most significant: low of low * high
s_mul_i32 s49, s31, s0 // (op 968) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 968) least significant
s_mul_hi_u32 s50, s30, s0 // (op 968) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 968) most significant: sum
s_add_u32 s19, s19, s50 // (op 968) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 968) 

s_mov_b32 s52, s32 // (op 968) reg expression
s_mov_b32 s53, s33 // (op 968) reg expression
s_mov_b32 s54, s0 // (op 968) reg expression
s_mov_b32 s55, 131072 // (op 968) call()

s_mul_hi_i32 s0, s16, s12 // (op 1030) 
v_lshrrev_b32 v53, 7, v1 // (op 1030) 
s_add_i32 s1, s0, s16 // (op 1030) 

v_and_b32 v57, 15, v1 // (op 1030) 
s_ashr_i32 s0, s1, 31 // (op 1030) 
v_lshl_add_u32 v14, v53, 4, v57 // (op 1030) 


s_cmp_eq_i32 s12, 0 // (op 1030) 
s_mov_b32 s18, scc // (op 1030) 
v_and_b32 v15, 63, v1 // (op 1030) 
s_cmp_eq_i32 s18, 0 // (op 1030) 
s_cselect_b32 s19, 0, -1 // (op 1030) 

v_lshrrev_b32 v53, 4, v15 // (op 1030) 

s_and_b32 s18, s15, 31 // (op 1030) 
s_lshl_b32 s49, 1, s18 // (op 1030) 
s_add_i32 s50, s49, s19 // (op 1030) 


s_and_b32 s19, s0, s50 // (op 1030) 


s_add_i32 s0, s1, s19 // (op 1030) 


s_ashr_i32 s1, s0, s18 // (op 1030) 


s_lshl_b32 s0, s15, 24 // (op 1030) 
s_ashr_i32 s18, s0, 31 // (op 1030) 

s_xor_b32 s0, s1, s18 // (op 1030) 

s_sub_i32 s1, s0, s18 // (op 1030) 


s_mul_hi_i32 s0, s1, s44 // (op 1030) 
s_add_i32 s18, s0, s1 // (op 1030) 

s_ashr_i32 s0, s18, 31 // (op 1030) 
s_cmp_eq_i32 s44, 0 // (op 1030) 
s_mov_b32 s19, scc // (op 1030) 
s_cmp_eq_i32 s19, 0 // (op 1030) 
s_cselect_b32 s49, 0, -1 // (op 1030) 

s_and_b32 s19, s45, 31 // (op 1030) 
s_lshl_b32 s50, 1, s19 // (op 1030) 
s_add_i32 s51, s50, s49 // (op 1030) 


s_and_b32 s49, s0, s51 // (op 1030) 


s_add_i32 s0, s18, s49 // (op 1030) 


s_ashr_i32 s18, s0, s19 // (op 1030) 


s_lshl_b32 s0, s45, 24 // (op 1030) 
s_ashr_i32 s19, s0, 31 // (op 1030) 

s_xor_b32 s0, s18, s19 // (op 1030) 

s_sub_i32 s18, s0, s19 // (op 1030) 


s_mul_i32 s0, s1, s13 // (op 1030) 

s_sub_u32 s1, s16, s0 // (op 1030) 

s_mul_i32 s0, s18, s13 // (op 1030) 
s_add_u32 s0, s0, s1 // (op 1030) 


s_mul_hi_i32 s1, s16, s11 // (op 1030) 
s_add_i32 s18, s1, s16 // (op 1030) 

s_ashr_i32 s1, s18, 31 // (op 1030) 
s_cmp_eq_i32 s11, 0 // (op 1030) 
s_mov_b32 s19, scc // (op 1030) 
s_cmp_eq_i32 s19, 0 // (op 1030) 
s_cselect_b32 s49, 0, -1 // (op 1030) 

s_and_b32 s19, s47, 31 // (op 1030) 
s_lshl_b32 s50, 1, s19 // (op 1030) 
s_add_i32 s51, s50, s49 // (op 1030) 


s_and_b32 s49, s1, s51 // (op 1030) 


s_add_i32 s1, s18, s49 // (op 1030) 


s_ashr_i32 s18, s1, s19 // (op 1030) 


s_lshl_b32 s1, s47, 24 // (op 1030) 
s_ashr_i32 s19, s1, 31 // (op 1030) 

s_xor_b32 s1, s18, s19 // (op 1030) 

s_sub_i32 s18, s1, s19 // (op 1030) 


s_add_u32 s50, s8, 64 // (op 1030) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1030) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1030) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1030) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1030) 

s_mul_i32 s1, s13, s50 // (op 1030) 

s_mul_i32 s19, s10, s1 // (op 1030) 

s_mul_i32 s1, s18, s19 // (op 1030) 

s_sub_u32 s19, s16, s1 // (op 1030) 

s_mov_b32 s1, s19 // (op 1030) 
s_mul_hi_i32 s49, s1, s14 // (op 1030) 
s_add_i32 s50, s49, s1 // (op 1030) 


s_ashr_i32 s1, s50, 31 // (op 1030) 
s_cmp_eq_i32 s14, 0 // (op 1030) 
s_mov_b32 s49, scc // (op 1030) 
s_cmp_eq_i32 s49, 0 // (op 1030) 
s_cselect_b32 s51, 0, -1 // (op 1030) 

s_and_b32 s49, s46, 31 // (op 1030) 
s_lshl_b32 s72, 1, s49 // (op 1030) 
s_add_i32 s73, s72, s51 // (op 1030) 


s_and_b32 s51, s1, s73 // (op 1030) 


s_add_i32 s1, s50, s51 // (op 1030) 


s_ashr_i32 s50, s1, s49 // (op 1030) 


s_lshl_b32 s1, s46, 24 // (op 1030) 
s_ashr_i32 s49, s1, 31 // (op 1030) 

s_xor_b32 s1, s50, s49 // (op 1030) 

s_sub_i32 s50, s1, s49 // (op 1030) 


s_mul_i32 s1, s48, s13 // (op 1030) 
s_add_u32 s72, s6, 16 // (op 1030) Start of Int64 add, locking SCC
s_addc_u32 s73, s7, 0 // (op 1030) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1030) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1030) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 1030) 

s_sub_i32 s49, s72, s1 // (op 1030) 

s_mul_i32 s51, s50, s49 // (op 1030) 


s_sub_u32 s49, s19, s51 // (op 1030) 


s_add_u32 s19, s49, s1 // (op 1030) 


s_cmp_eq_i32 s18, 0 // (op 1030) 
s_mov_b32 s1, scc // (op 1030) 

s_cmp_eq_i32 s1, 0 // (op 1030) 
s_cselect_b32 s18, s19, s0 // (op 1030) 



v_lshl_add_u32 v15, s18, 4, v14 // (op 1030) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 1030) 
s_lshr_b32 s18, s0, 1 // (op 1030) 

v_lshl_add_u32 v14, s18, 3, v53 // (op 1030) 


v_mul_lo_u32 v53, v14, s34 // (op 1030) 

v_add_u32 v14, v15, v53 // (op 1030) 


s_mul_i32 s1, s34, s39 // (op 1032) most significant: low of low * high
s_mul_i32 s18, s35, s38 // (op 1032) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 1032) least significant
s_mul_hi_u32 s19, s34, s38 // (op 1032) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1032) most significant: sum
s_add_u32 s1, s1, s19 // (op 1032) most significant: sum


s_mov_b32 s64, s36 // (op 1032) reg expression
s_mov_b32 s65, s37 // (op 1032) reg expression
s_mov_b32 s66, s0 // (op 1032) reg expression
s_mov_b32 s67, 131072 // (op 1032) call()

s_mul_hi_i32 s0, s16, s12 // (op 1094) 
v_and_b32 v15, -64, v1 // (op 1094) 
s_add_i32 s1, s0, s16 // (op 1094) 

v_sub_u32 v53, v1, v15 // (op 1094) 

s_ashr_i32 s0, s1, 31 // (op 1094) 
v_lshrrev_b32 v15, 6, v1 // (op 1094) 
s_cmp_eq_i32 s12, 0 // (op 1094) 
s_mov_b32 s18, scc // (op 1094) 
v_lshlrev_b32 v57, 1, v15 // (op 1094) 

s_cmp_eq_i32 s18, 0 // (op 1094) 
s_cselect_b32 s19, 0, -1 // (op 1094) 

s_and_b32 s18, s15, 31 // (op 1094) 
s_lshl_b32 s49, 1, s18 // (op 1094) 
s_add_i32 s50, s49, s19 // (op 1094) 


s_and_b32 s19, s0, s50 // (op 1094) 


s_add_i32 s0, s1, s19 // (op 1094) 


s_ashr_i32 s1, s0, s18 // (op 1094) 


s_lshl_b32 s0, s15, 24 // (op 1094) 
s_ashr_i32 s18, s0, 31 // (op 1094) 

s_xor_b32 s0, s1, s18 // (op 1094) 

s_sub_i32 s1, s0, s18 // (op 1094) 


s_mul_hi_i32 s0, s1, s44 // (op 1094) 
s_add_i32 s18, s0, s1 // (op 1094) 

s_ashr_i32 s0, s18, 31 // (op 1094) 
s_cmp_eq_i32 s44, 0 // (op 1094) 
s_mov_b32 s19, scc // (op 1094) 
s_cmp_eq_i32 s19, 0 // (op 1094) 
s_cselect_b32 s49, 0, -1 // (op 1094) 

s_and_b32 s19, s45, 31 // (op 1094) 
s_lshl_b32 s50, 1, s19 // (op 1094) 
s_add_i32 s51, s50, s49 // (op 1094) 


s_and_b32 s49, s0, s51 // (op 1094) 


s_add_i32 s0, s18, s49 // (op 1094) 


s_ashr_i32 s18, s0, s19 // (op 1094) 


s_lshl_b32 s0, s45, 24 // (op 1094) 
s_ashr_i32 s19, s0, 31 // (op 1094) 

s_xor_b32 s0, s18, s19 // (op 1094) 

s_sub_i32 s18, s0, s19 // (op 1094) 


s_add_u32 s50, s8, 64 // (op 1094) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 1094) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1094) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1094) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1094) 

s_mul_i32 s0, s18, s50 // (op 1094) 

s_sub_i32 s49, s1, s0 // (op 1094) 


s_mul_hi_i32 s0, s16, s11 // (op 1094) 
s_add_i32 s1, s0, s16 // (op 1094) 

s_ashr_i32 s0, s1, 31 // (op 1094) 
s_cmp_eq_i32 s11, 0 // (op 1094) 
s_mov_b32 s18, scc // (op 1094) 
s_cmp_eq_i32 s18, 0 // (op 1094) 
s_cselect_b32 s19, 0, -1 // (op 1094) 

s_and_b32 s18, s47, 31 // (op 1094) 
s_lshl_b32 s72, 1, s18 // (op 1094) 
s_add_i32 s73, s72, s19 // (op 1094) 


s_and_b32 s19, s0, s73 // (op 1094) 


s_add_i32 s0, s1, s19 // (op 1094) 


s_ashr_i32 s1, s0, s18 // (op 1094) 


s_lshl_b32 s0, s47, 24 // (op 1094) 
s_ashr_i32 s18, s0, 31 // (op 1094) 

s_xor_b32 s0, s1, s18 // (op 1094) 

s_sub_i32 s1, s0, s18 // (op 1094) 


s_mul_i32 s0, s13, s50 // (op 1094) 

s_mul_i32 s18, s10, s0 // (op 1094) 

s_mul_i32 s0, s1, s18 // (op 1094) 

s_sub_u32 s18, s16, s0 // (op 1094) 

s_mul_hi_i32 s0, s18, s14 // (op 1094) 
s_add_i32 s19, s0, s18 // (op 1094) 


s_ashr_i32 s0, s19, 31 // (op 1094) 
s_cmp_eq_i32 s14, 0 // (op 1094) 
s_mov_b32 s18, scc // (op 1094) 
s_cmp_eq_i32 s18, 0 // (op 1094) 
s_cselect_b32 s50, 0, -1 // (op 1094) 

s_and_b32 s18, s46, 31 // (op 1094) 
s_lshl_b32 s51, 1, s18 // (op 1094) 
s_add_i32 s72, s51, s50 // (op 1094) 


s_and_b32 s50, s0, s72 // (op 1094) 


s_add_i32 s0, s19, s50 // (op 1094) 


s_ashr_i32 s19, s0, s18 // (op 1094) 


s_lshl_b32 s0, s46, 24 // (op 1094) 
s_ashr_i32 s18, s0, 31 // (op 1094) 

s_xor_b32 s0, s19, s18 // (op 1094) 

s_sub_i32 s19, s0, s18 // (op 1094) 


s_cmp_eq_i32 s1, 0 // (op 1094) 
s_mov_b32 s0, scc // (op 1094) 

s_cmp_eq_i32 s0, 0 // (op 1094) 
s_cselect_b32 s1, s19, s49 // (op 1094) 



v_lshl_add_u32 v15, s1, 6, v53 // (op 1094) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 1094) 
s_lshr_b32 s18, s0, 1 // (op 1094) 

v_lshl_add_u32 v53, s18, 3, v57 // (op 1094) 


v_mul_lo_u32 v57, v53, s40 // (op 1094) 
v_add_u32 v57, v57, v15 // (op 1094) 


s_mul_i32 s1, s40, s39 // (op 1096) most significant: low of low * high
s_mul_i32 s18, s41, s38 // (op 1096) most significant: low of high * low
s_mul_i32 s0, s40, s38 // (op 1096) least significant
s_mul_hi_u32 s19, s40, s38 // (op 1096) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1096) most significant: sum
s_add_u32 s1, s1, s19 // (op 1096) most significant: sum


s_mov_b32 s68, s42 // (op 1096) reg expression
s_mov_b32 s69, s43 // (op 1096) reg expression
s_mov_b32 s70, s0 // (op 1096) reg expression
s_mov_b32 s71, 131072 // (op 1096) call()

v_lshlrev_b32 v15, 4, v1 // (op 1187) 
s_nop 0
v_readfirstlane_b32 s0, v15 // (op 1187) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v15, 6, v1 // (op 1239) 
v_mul_lo_u32 v53, v15, 6 // (op 1239) 
v_lshrrev_b32 v18, 3, v53 // (op 1239) 

v_lshrrev_b32 v53, 4, v15 // (op 1239) 

v_lshlrev_b32 v15, 2, v53 // (op 1239) 

v_add_u32 v53, v18, v15 // (op 1239) 


v_and_b32 v15, -64, v1 // (op 1297) 
v_sub_u32 v18, v1, v15 // (op 1297) 

v_lshrrev_b32 v15, 6, v1 // (op 1297) 
v_lshl_add_u32 v19, v15, 7, v18 // (op 1297) 


s_ashr_i64 s[18:19], s[28:29], 7 // (op 317) 
s_and_b32 s1, s18, 4294967294 // (op 317) 

s_ashr_i64 s[18:19], s[28:29], 7 // (op 316) 
s_mov_b32 s49, s18 // (op 316) convert

s_cmp_lt_u32 s1, s49 // (op 316) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316:
s_barrier  // (op 1492) 
s_add_u32 s1, s1, 1 // (op 318) 


v_mov_b64 v[12:13], v[16:17] // (op 869) 
v_mov_b64 v[20:21], v[12:13] // (op 869) reg expression

s_mov_b32 m0, s0 // (op 869) reg expression
buffer_load_dwordx4 v20, s[20:23], 0 offen offset: 0 lds // (op 869) Load value direct to lds


s_lshl_b64 s[18:19], s[24:25], 7 // (op 888) 
v_mov_b32 v15, s18 // (op 888) 
v_mov_b32 v18, s19 // (op 888) 
v_add_co_u32 v16, s[50:51], v16, v15 // (op 888) least significant half
v_addc_co_u32 v17, s[50:51], v17, v18, s[50:51] // (op 888) most significant half






v_mov_b32 v15, v49 // (op 183) 
v_mov_b32 v18, v15 // (op 183) reg expression

buffer_load_dwordx3 v[20:22], v18, s[52:55], 0 offen // (op 183) Load value
buffer_load_dwordx3 v[24:26], v18, s[52:55], 0 offen offset:12 // (op 183) Load value
buffer_load_dwordx3 v[28:30], v18, s[52:55], 0 offen offset:24 // (op 183) Load value
buffer_load_dwordx3 v[32:34], v18, s[52:55], 0 offen offset:36 // (op 183) Load value


s_lshl_b64 s[18:19], s[30:31], 7 // (op 958) 
s_mul_i32 s51, s19, 6 // (op 958) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 958) least significant
s_mul_hi_u32 s49, s18, 6 // (op 958) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 958) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 958) 

v_add_u32 v49, v49, s18 // (op 958) 


s_barrier  // (op 1505) 
v_mov_b64 v[12:13], s[40:41] // (op 184) convert




v_mov_b32 v15, v57 // (op 184) 
v_mov_b32 v18, v15 // (op 184) reg expression
buffer_load_ubyte v50, v18, s[68:71], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v18, v18, v12 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b96 v53, v[20:22] offset:4096 // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v53, v[24:26] offset:4112 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v53, v[28:30] offset:4128 // (op 181) Store local data 
buffer_load_ubyte v51, v18, s[68:71], 0 offen offset: 0 // (op 184) Load value



s_lshl_b32 s18, s40, 2 // (op 1086) 
v_add_u32 v57, v57, s18 // (op 1086) 

s_waitcnt vmcnt(2)
ds_write_b96 v53, v[32:34] offset:4144 // (op 181) Store local data 

s_barrier  // (op 1495) 



s_barrier  // (op 1519) 
ds_read_b64_tr_b8 v[20:21], v0 // (op 173) Transpose load local data 



ds_read_b64_tr_b8 v[22:23], v0 offset:128 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:1024 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:1152 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v50 offset:12288 // (op 182) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1509) 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v51 offset:12352 // (op 182) Store local data 




ds_read_b96_tr_b6 v[28:30], v4 offset:4096 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v4 offset:5120 // (op 175) Transpose load local data 

v_mov_b64 v[12:13], s[34:35] // (op 177) convert

v_mov_b32 v15, v14 // (op 177) 
v_mov_b32 v18, v15 // (op 177) reg expression
v_mov_b32 v34, v18 // (op 177) 
buffer_load_ubyte v35, v34, s[64:67], 0 offen offset: 0 // (op 177) Load value




s_lshl_b32 s18, s34, 2 // (op 1022) 
v_add_u32 v14, v14, s18 // (op 1022) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1523) 


ds_read_u8 v15, v7 offset:12288 // (op 179) Load local data 



ds_read_b96_tr_b6 v[36:38], v4 offset:4128 // (op 324) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v4 offset:5152 // (op 324) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:33], a[0:3], v35, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 180) 




ds_read_u8 v15, v7 offset:12320 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:27], v[36:41], a[4:7], v35, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 328) 




s_ashr_i64 s[18:19], s[28:29], 7 // (op 316) 
s_mov_b32 s49, s18 // (op 316) convert

s_cmp_lt_u32 s1, s49 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 if true)
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316:






















s_barrier  // (op 1595) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 

v_cvt_f16_f32 v10, v14 // (op 52) 
v_cvt_f16_f32 v11, v15 // (op 52) 
v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 


v_pack_b32_f16 v14, v10, v11 // (op 54) 
v_pack_b32_f16 v15, v12, v13 // (op 54) 

v_mov_b32 v0, v9 // (op 54) reg expression

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 54) Store value



v_accvgpr_read v10, a4 // (op 508) 
v_accvgpr_read v11, a5 // (op 508) 
v_accvgpr_read v12, a6 // (op 508) 
v_accvgpr_read v13, a7 // (op 508) 
v_mul_f32 v14, v8, v10 // (op 508) 
v_mul_f32 v15, v8, v11 // (op 508) 
v_mul_f32 v16, v8, v12 // (op 508) 
v_mul_f32 v17, v8, v13 // (op 508) 

v_cvt_f16_f32 v10, v14 // (op 510) 
v_cvt_f16_f32 v11, v15 // (op 510) 
v_cvt_f16_f32 v12, v16 // (op 510) 
v_cvt_f16_f32 v13, v17 // (op 510) 


v_pack_b32_f16 v14, v10, v11 // (op 511) 
v_pack_b32_f16 v15, v12, v13 // (op 511) 

s_lshl_b32 s0, s4, 6 // (op 511) 
v_add_u32 v0, v9, s0 // (op 511) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 511) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalFalse_0_beta_125:


v_mov_b32 v0, v6 // (op 138) reg expression

buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 138) Load value

v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v8, v12 // (op 134) 
v_mul_f32 v17, v8, v13 // (op 134) 
v_mul_f32 v18, v8, v14 // (op 134) 
v_mul_f32 v19, v8, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 139) 
v_lshrrev_b32 v13, 16, v10 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v5, v12 // (op 139) 
v_mul_f32 v21, v5, v13 // (op 139) 
v_cvt_f32_f16 v12, v11 // (op 139) 
v_lshrrev_b32 v13, 16, v11 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v22, v5, v12 // (op 139) 
v_mul_f32 v23, v5, v13 // (op 139) 


v_add_f32 v10, v20, v16 // (op 135) 
v_add_f32 v11, v21, v17 // (op 135) 
v_add_f32 v12, v22, v18 // (op 135) 
v_add_f32 v13, v23, v19 // (op 135) 


v_cvt_f16_f32 v14, v10 // (op 136) 
v_cvt_f16_f32 v15, v11 // (op 136) 
v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 


v_pack_b32_f16 v10, v14, v15 // (op 137) 
v_pack_b32_f16 v11, v16, v17 // (op 137) 

v_mov_b32 v0, v9 // (op 137) reg expression

buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 535) 
v_add_u32 v0, v6, s0 // (op 535) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 535) Load value

v_accvgpr_read v12, a4 // (op 531) 
v_accvgpr_read v13, a5 // (op 531) 
v_accvgpr_read v14, a6 // (op 531) 
v_accvgpr_read v15, a7 // (op 531) 
v_mul_f32 v16, v8, v12 // (op 531) 
v_mul_f32 v17, v8, v13 // (op 531) 
v_mul_f32 v18, v8, v14 // (op 531) 
v_mul_f32 v19, v8, v15 // (op 531) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 536) 
v_lshrrev_b32 v13, 16, v10 // (op 536) 
v_cvt_f32_f16 v13, v13 // (op 536) 
v_mul_f32 v20, v5, v12 // (op 536) 
v_mul_f32 v21, v5, v13 // (op 536) 
v_cvt_f32_f16 v12, v11 // (op 536) 
v_lshrrev_b32 v13, 16, v11 // (op 536) 
v_cvt_f32_f16 v13, v13 // (op 536) 
v_mul_f32 v22, v5, v12 // (op 536) 
v_mul_f32 v23, v5, v13 // (op 536) 


v_add_f32 v10, v20, v16 // (op 532) 
v_add_f32 v11, v21, v17 // (op 532) 
v_add_f32 v12, v22, v18 // (op 532) 
v_add_f32 v13, v23, v19 // (op 532) 


v_cvt_f16_f32 v14, v10 // (op 533) 
v_cvt_f16_f32 v15, v11 // (op 533) 
v_cvt_f16_f32 v16, v12 // (op 533) 
v_cvt_f16_f32 v17, v13 // (op 533) 


v_pack_b32_f16 v10, v14, v15 // (op 534) 
v_pack_b32_f16 v11, v16, v17 // (op 534) 

s_lshl_b32 s0, s4, 6 // (op 534) 
v_add_u32 v0, v9, s0 // (op 534) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 534) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.LRR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__end:
.size RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_, .LRR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM__end-RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 92
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 84
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
    DataTypeB: B6
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
    macrotile: [16, 64, 128]
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
  - .name: RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 82
    .agpr_count: 8
    .max_flat_workgroup_size: 128
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

