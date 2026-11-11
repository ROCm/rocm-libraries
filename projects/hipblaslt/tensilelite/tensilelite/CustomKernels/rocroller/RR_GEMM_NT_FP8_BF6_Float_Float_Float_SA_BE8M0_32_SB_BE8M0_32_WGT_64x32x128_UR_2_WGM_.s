// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_,@function
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1159) 
s_mul_hi_i32 s1, s16, s10 // (op 1347) 
v_and_b32 v0, 63, v1 // (op 1273) 
v_and_b32 v3, 48, v1 // (op 1273) 
v_sub_u32 v4, v0, v3 // (op 1273) 

v_lshrrev_b32 v3, 4, v0 // (op 1273) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1273) 


v_lshrrev_b32 v3, 6, v1 // (op 1273) 
v_and_b32 v4, 1, v3 // (op 1273) 

v_lshlrev_b32 v3, 4, v4 // (op 1273) 

v_lshl_add_u32 v4, v0, 5, v3 // (op 1273) 


v_mul_lo_u32 v0, v4, 6 // (op 1273) 
v_lshrrev_b32 v3, 3, v0 // (op 1273) 

v_lshrrev_b32 v0, 4, v4 // (op 1273) 

v_lshlrev_b32 v4, 2, v0 // (op 1273) 

v_add_u32 v0, v3, v4 // (op 1273) 


v_and_b32 v3, 63, v1 // (op 1417) 
v_and_b32 v4, 48, v1 // (op 1417) 
v_sub_u32 v5, v3, v4 // (op 1417) 

v_lshrrev_b32 v4, 4, v3 // (op 1417) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 1417) 


v_lshrrev_b32 v4, 6, v1 // (op 1417) 
v_and_b32 v5, 1, v4 // (op 1417) 

v_lshlrev_b32 v4, 4, v5 // (op 1417) 

v_lshl_add_u32 v5, v3, 5, v4 // (op 1417) 


v_mul_lo_u32 v3, v5, 6 // (op 1417) 
v_lshrrev_b32 v4, 3, v3 // (op 1417) 

v_lshrrev_b32 v3, 4, v5 // (op 1417) 

v_lshlrev_b32 v5, 2, v3 // (op 1417) 

v_add_u32 v3, v4, v5 // (op 1417) 


v_lshrrev_b32 v4, 7, v1 // (op 1159) 
s_add_i32 s18, s0, s16 // (op 1159) 

v_and_b32 v5, 63, v1 // (op 1159) 
s_ashr_i32 s0, s18, 31 // (op 1159) 
v_lshrrev_b32 v6, 4, v5 // (op 1159) 
s_cmp_eq_i32 s10, 0 // (op 1159) 
s_mov_b32 s19, scc // (op 1159) 
v_lshlrev_b32 v7, 2, v6 // (op 1159) 

s_cmp_eq_i32 s19, 0 // (op 1159) 
s_cselect_b32 s51, 0, -1 // (op 1159) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1159) 


s_and_b32 s19, s13, 31 // (op 1159) 
v_lshrrev_b32 v4, 2, v5 // (op 1159) 

v_and_b32 v5, 3, v4 // (op 1159) 

v_and_b32 v4, 48, v1 // (op 1220) 
v_and_b32 v7, 63, v1 // (op 1220) 
v_sub_u32 v8, v7, v4 // (op 1220) 

v_lshrrev_b32 v7, 1, v8 // (op 1220) 
v_add_lshl_u32 v9, v4, v7, 6 // (op 1220) 


v_and_b32 v4, 1, v8 // (op 1220) 

v_lshlrev_b32 v7, 3, v4 // (op 1220) 

v_lshrrev_b32 v4, 7, v1 // (op 1220) 
v_lshl_add_u32 v8, v4, 4, v7 // (op 1220) 


v_add_u32 v4, v9, v8 // (op 1220) 


v_lshrrev_b32 v7, 7, v1 // (op 1331) 
v_and_b32 v8, 15, v1 // (op 1331) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 1331) 


v_and_b32 v7, 63, v1 // (op 1331) 
v_lshrrev_b32 v8, 4, v7 // (op 1331) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 1331) 


v_lshrrev_b32 v8, 7, v1 // (op 1347) 
v_and_b32 v9, 63, v1 // (op 1347) 
v_and_b32 v10, 48, v1 // (op 1393) 
v_and_b32 v11, 63, v1 // (op 1393) 
v_sub_u32 v12, v11, v10 // (op 1393) 

v_lshrrev_b32 v11, 1, v12 // (op 1393) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 1393) 


v_and_b32 v10, 1, v12 // (op 1393) 

v_lshlrev_b32 v11, 3, v10 // (op 1393) 

v_lshrrev_b32 v10, 7, v1 // (op 1393) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 1393) 


v_add_u32 v10, v13, v12 // (op 1393) 


v_lshrrev_b32 v11, 7, v1 // (op 1446) 
v_and_b32 v12, 15, v1 // (op 1446) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1446) 


v_and_b32 v11, 63, v1 // (op 1446) 
v_lshrrev_b32 v12, 4, v11 // (op 1446) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 1446) 


s_lshl_b32 s52, 1, s19 // (op 1159) 
v_and_b32 v12, 3, v1 // (op 1159) 
s_add_i32 s53, s52, s51 // (op 1159) 


s_and_b32 s51, s0, s53 // (op 1159) 


s_add_i32 s0, s1, s16 // (op 1347) 

s_ashr_i32 s1, s0, 31 // (op 1347) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1159) 


s_add_i32 s52, s18, s51 // (op 1159) 


v_lshrrev_b32 v5, 6, v1 // (op 1159) 
s_ashr_i32 s18, s52, s19 // (op 1159) 


v_and_b32 v12, 1, v5 // (op 1159) 

s_lshl_b32 s19, s13, 24 // (op 1159) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1159) 


s_ashr_i32 s51, s19, 31 // (op 1159) 

s_xor_b32 s19, s18, s51 // (op 1159) 

s_sub_i32 s18, s19, s51 // (op 1159) 


s_mul_hi_i32 s19, s18, s14 // (op 1159) 
s_add_i32 s51, s19, s18 // (op 1159) 

s_ashr_i32 s19, s51, 31 // (op 1159) 
s_cmp_eq_i32 s14, 0 // (op 1159) 
s_mov_b32 s52, scc // (op 1159) 
s_cmp_eq_i32 s52, 0 // (op 1159) 
s_cselect_b32 s53, 0, -1 // (op 1159) 

s_and_b32 s52, s15, 31 // (op 1159) 
v_lshrrev_b32 v12, 4, v9 // (op 1347) 
s_cmp_eq_i32 s10, 0 // (op 1347) 
s_mov_b32 s54, scc // (op 1347) 
v_lshlrev_b32 v13, 2, v12 // (op 1347) 

s_cmp_eq_i32 s54, 0 // (op 1347) 
s_cselect_b32 s55, 0, -1 // (op 1347) 

s_lshl_b32 s54, 1, s52 // (op 1159) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1347) 


v_lshrrev_b32 v8, 2, v9 // (op 1347) 

s_add_i32 s56, s54, s53 // (op 1159) 


s_and_b32 s53, s19, s56 // (op 1159) 


s_add_i32 s19, s51, s53 // (op 1159) 


s_ashr_i32 s51, s19, s52 // (op 1159) 


s_lshl_b32 s19, s15, 24 // (op 1159) 
s_ashr_i32 s52, s19, 31 // (op 1159) 

s_xor_b32 s19, s51, s52 // (op 1159) 

s_sub_i32 s51, s19, s52 // (op 1159) 


s_mul_i32 s19, s18, s11 // (op 1159) 
s_sub_u32 s52, s16, s19 // (op 1159) 

s_mul_i32 s19, s51, s11 // (op 1159) 
s_add_u32 s19, s19, s52 // (op 1159) 

s_mul_hi_i32 s52, s16, s9 // (op 1159) 
s_add_i32 s53, s52, s16 // (op 1159) 

s_ashr_i32 s52, s53, 31 // (op 1159) 
s_cmp_eq_i32 s9, 0 // (op 1159) 
s_mov_b32 s54, scc // (op 1159) 
s_cmp_eq_i32 s54, 0 // (op 1159) 
s_cselect_b32 s56, 0, -1 // (op 1159) 

s_and_b32 s54, s13, 31 // (op 1347) 
v_and_b32 v9, 3, v8 // (op 1347) 

s_lshl_b32 s57, 1, s54 // (op 1347) 
v_and_b32 v8, 3, v1 // (op 1347) 
s_add_i32 s58, s57, s55 // (op 1347) 


s_and_b32 s55, s1, s58 // (op 1347) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1347) 


s_add_i32 s1, s0, s55 // (op 1347) 


v_lshrrev_b32 v8, 6, v1 // (op 1347) 
s_ashr_i32 s0, s1, s54 // (op 1347) 


v_and_b32 v9, 1, v8 // (op 1347) 

s_lshl_b32 s1, s13, 24 // (op 1347) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1347) 


s_ashr_i32 s54, s1, 31 // (op 1347) 

s_xor_b32 s1, s0, s54 // (op 1347) 

s_sub_i32 s0, s1, s54 // (op 1347) 


s_mul_hi_i32 s1, s0, s14 // (op 1347) 
s_add_i32 s54, s1, s0 // (op 1347) 

s_ashr_i32 s1, s54, 31 // (op 1347) 
s_cmp_eq_i32 s14, 0 // (op 1347) 
s_mov_b32 s55, scc // (op 1347) 
s_cmp_eq_i32 s55, 0 // (op 1347) 
s_cselect_b32 s57, 0, -1 // (op 1347) 

s_and_b32 s55, s15, 31 // (op 1347) 
s_lshl_b32 s58, 1, s55 // (op 1347) 
s_add_i32 s59, s58, s57 // (op 1347) 


s_and_b32 s57, s1, s59 // (op 1347) 


s_add_i32 s1, s54, s57 // (op 1347) 


s_ashr_i32 s54, s1, s55 // (op 1347) 


s_lshl_b32 s1, s15, 24 // (op 1347) 
s_ashr_i32 s55, s1, 31 // (op 1347) 

s_xor_b32 s1, s54, s55 // (op 1347) 

s_sub_i32 s54, s1, s55 // (op 1347) 


s_mul_i32 s1, s0, s11 // (op 1347) 
s_sub_u32 s55, s16, s1 // (op 1347) 

s_mul_i32 s1, s54, s11 // (op 1347) 
s_add_u32 s1, s1, s55 // (op 1347) 

s_mul_hi_i32 s55, s16, s9 // (op 1347) 
s_add_i32 s57, s55, s16 // (op 1347) 

s_ashr_i32 s55, s57, 31 // (op 1347) 
s_cmp_eq_i32 s9, 0 // (op 1347) 
s_mov_b32 s58, scc // (op 1347) 
s_cmp_eq_i32 s58, 0 // (op 1347) 
s_cselect_b32 s59, 0, -1 // (op 1347) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s47, 31 // (op 1159) 
s_lshl_b32 s60, 1, s58 // (op 1159) 
s_add_i32 s61, s60, s56 // (op 1159) 


s_and_b32 s56, s52, s61 // (op 1159) 


s_add_i32 s52, s53, s56 // (op 1159) 


s_ashr_i32 s53, s52, s58 // (op 1159) 


s_lshl_b32 s52, s47, 24 // (op 1159) 
s_ashr_i32 s56, s52, 31 // (op 1159) 

s_xor_b32 s52, s53, s56 // (op 1159) 

s_sub_i32 s53, s52, s56 // (op 1159) 


s_and_b32 s52, s47, 31 // (op 1347) 
s_lshl_b32 s56, 1, s52 // (op 1347) 
s_add_i32 s58, s56, s59 // (op 1347) 


s_and_b32 s56, s55, s58 // (op 1347) 


s_add_i32 s55, s57, s56 // (op 1347) 


s_ashr_i32 s56, s55, s52 // (op 1347) 


s_lshl_b32 s52, s47, 24 // (op 1347) 
s_ashr_i32 s55, s52, 31 // (op 1347) 

s_xor_b32 s52, s56, s55 // (op 1347) 

s_sub_i32 s56, s52, s55 // (op 1347) 


s_add_u32 s58, s6, 32 // (op 1159) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1159) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1159) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1159) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 5 // (op 1159) 

s_mul_i32 s52, s11, s58 // (op 1159) 
s_mul_i32 s55, s8, s52 // (op 1159) 

s_mul_i32 s52, s53, s55 // (op 1159) 

s_sub_u32 s55, s16, s52 // (op 1159) 

s_mov_b32 s52, s55 // (op 1159) 
s_mul_hi_i32 s57, s52, s12 // (op 1159) 
s_add_i32 s60, s57, s52 // (op 1159) 


s_ashr_i32 s52, s60, 31 // (op 1159) 
s_cmp_eq_i32 s12, 0 // (op 1159) 
s_mov_b32 s57, scc // (op 1159) 
s_cmp_eq_i32 s57, 0 // (op 1159) 
s_cselect_b32 s61, 0, -1 // (op 1159) 

s_and_b32 s57, s46, 31 // (op 1159) 
s_lshl_b32 s62, 1, s57 // (op 1159) 
s_add_i32 s63, s62, s61 // (op 1159) 


s_and_b32 s61, s52, s63 // (op 1159) 


s_add_i32 s52, s60, s61 // (op 1159) 


s_ashr_i32 s60, s52, s57 // (op 1159) 


s_lshl_b32 s52, s46, 24 // (op 1159) 
s_ashr_i32 s57, s52, 31 // (op 1159) 

s_xor_b32 s52, s60, s57 // (op 1159) 

s_sub_i32 s60, s52, s57 // (op 1159) 


s_mul_i32 s52, s48, s11 // (op 1159) 
s_add_u32 s62, s4, 64 // (op 1159) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 1159) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1159) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1159) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 1159) 

s_sub_i32 s57, s62, s52 // (op 1159) 

s_mul_i32 s61, s60, s57 // (op 1159) 

s_sub_u32 s57, s55, s61 // (op 1159) 


s_add_u32 s55, s57, s52 // (op 1159) 


s_cmp_eq_i32 s53, 0 // (op 1159) 
s_mov_b32 s52, scc // (op 1159) 

s_cmp_eq_i32 s52, 0 // (op 1159) 
s_cselect_b32 s53, s55, s19 // (op 1159) 


s_mul_i32 s19, s51, s58 // (op 1159) 


v_lshl_add_u32 v9, s53, 6, v6 // (op 1159) 


s_sub_i32 s51, s18, s19 // (op 1159) 


s_cmp_eq_i32 s52, 0 // (op 1159) 
s_cselect_b32 s53, s60, s51 // (op 1159) 



v_lshl_add_u32 v6, s53, 5, v5 // (op 1159) 


v_mul_lo_u32 v5, v6, s2 // (op 1159) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 1159) 


s_add_u32 s18, s6, 32 // (op 1347) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1347) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 1347) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 1347) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 1347) 

s_mul_i32 s55, s11, s18 // (op 1347) 
s_mul_i32 s51, s8, s55 // (op 1347) 

s_mul_i32 s55, s56, s51 // (op 1347) 

s_sub_u32 s51, s16, s55 // (op 1347) 

s_mov_b32 s55, s51 // (op 1347) 
s_mul_hi_i32 s52, s55, s12 // (op 1347) 
s_add_i32 s53, s52, s55 // (op 1347) 


s_ashr_i32 s52, s53, 31 // (op 1347) 
s_cmp_eq_i32 s12, 0 // (op 1347) 
s_mov_b32 s55, scc // (op 1347) 
s_cmp_eq_i32 s55, 0 // (op 1347) 
s_cselect_b32 s57, 0, -1 // (op 1347) 

s_and_b32 s55, s46, 31 // (op 1347) 
s_lshl_b32 s58, 1, s55 // (op 1347) 
s_add_i32 s59, s58, s57 // (op 1347) 


s_and_b32 s57, s52, s59 // (op 1347) 


s_add_i32 s52, s53, s57 // (op 1347) 


s_ashr_i32 s53, s52, s55 // (op 1347) 


s_lshl_b32 s52, s46, 24 // (op 1347) 
s_ashr_i32 s55, s52, 31 // (op 1347) 

s_xor_b32 s52, s53, s55 // (op 1347) 

s_sub_i32 s53, s52, s55 // (op 1347) 


s_mul_i32 s52, s48, s11 // (op 1347) 
s_add_u32 s58, s4, 64 // (op 1347) Start of Int64 add, locking SCC
s_addc_u32 s59, s5, 0 // (op 1347) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1347) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1347) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 1347) 

s_sub_i32 s55, s58, s52 // (op 1347) 

s_mul_i32 s57, s53, s55 // (op 1347) 

s_sub_u32 s55, s51, s57 // (op 1347) 


s_add_u32 s51, s55, s52 // (op 1347) 


s_cmp_eq_i32 s56, 0 // (op 1347) 
s_mov_b32 s52, scc // (op 1347) 

s_cmp_eq_i32 s52, 0 // (op 1347) 
s_cselect_b32 s55, s51, s1 // (op 1347) 


s_mul_i32 s1, s54, s18 // (op 1347) 


v_lshl_add_u32 v5, s55, 6, v12 // (op 1347) 


s_sub_i32 s51, s0, s1 // (op 1347) 


s_cmp_eq_i32 s52, 0 // (op 1347) 
s_cselect_b32 s0, s53, s51 // (op 1347) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 1347) 


v_mul_lo_u32 v8, v9, s20 // (op 1347) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 1347) 


s_mul_i32 s1, s2, s7 // (op 1161) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1161) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1161) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1161) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1161) most significant: sum
s_add_u32 s1, s1, s19 // (op 1161) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1161) 

s_mul_i32 s1, s20, s7 // (op 1349) most significant: low of low * high
s_mul_i32 s51, s21, s6 // (op 1349) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 1349) least significant
s_mul_hi_u32 s52, s20, s6 // (op 1349) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1349) most significant: sum
s_add_u32 s1, s1, s52 // (op 1349) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1349) 

s_mov_b32 s56, s22 // (op 1161) reg expression
s_mov_b32 s57, s23 // (op 1161) reg expression
s_mov_b32 s58, s18 // (op 1161) reg expression
s_mov_b32 s59, 131072 // (op 1161) call()

s_mov_b32 s60, s24 // (op 1349) reg expression

s_mov_b32 s61, s25 // (op 1349) reg expression
s_mov_b32 s62, s52 // (op 1349) reg expression
s_mov_b32 s63, 131072 // (op 1349) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 309) call()
v_accvgpr_write a5, 0.00000 // (op 309) call()
v_accvgpr_write a6, 0.00000 // (op 309) call()
v_accvgpr_write a7, 0.00000 // (op 309) call()
s_mul_hi_i32 s0, s16, s10 // (op 926) 
v_and_b32 v12, -4, v1 // (op 926) 
s_add_i32 s1, s0, s16 // (op 926) 

v_sub_u32 v13, v1, v12 // (op 926) 

s_ashr_i32 s0, s1, 31 // (op 926) 
v_lshlrev_b32 v12, 4, v13 // (op 926) 

s_cmp_eq_i32 s10, 0 // (op 926) 
s_mov_b32 s18, scc // (op 926) 
v_lshrrev_b32 v13, 2, v1 // (op 926) 
s_cmp_eq_i32 s18, 0 // (op 926) 
s_cselect_b32 s19, 0, -1 // (op 926) 

s_and_b32 s18, s13, 31 // (op 926) 
s_lshl_b32 s22, 1, s18 // (op 926) 
s_add_i32 s23, s22, s19 // (op 926) 


s_and_b32 s19, s0, s23 // (op 926) 


s_add_i32 s0, s1, s19 // (op 926) 


s_ashr_i32 s1, s0, s18 // (op 926) 


s_lshl_b32 s0, s13, 24 // (op 926) 
s_ashr_i32 s18, s0, 31 // (op 926) 

s_xor_b32 s0, s1, s18 // (op 926) 

s_sub_i32 s1, s0, s18 // (op 926) 


s_mul_hi_i32 s0, s1, s14 // (op 926) 
s_add_i32 s18, s0, s1 // (op 926) 

s_ashr_i32 s0, s18, 31 // (op 926) 
s_cmp_eq_i32 s14, 0 // (op 926) 
s_mov_b32 s19, scc // (op 926) 
s_cmp_eq_i32 s19, 0 // (op 926) 
s_cselect_b32 s22, 0, -1 // (op 926) 

s_and_b32 s19, s15, 31 // (op 926) 
s_lshl_b32 s23, 1, s19 // (op 926) 
s_add_i32 s24, s23, s22 // (op 926) 


s_and_b32 s25, s0, s24 // (op 926) 


s_add_i32 s0, s18, s25 // (op 926) 


s_ashr_i32 s18, s0, s19 // (op 926) 


s_lshl_b32 s0, s15, 24 // (op 926) 
s_ashr_i32 s19, s0, 31 // (op 926) 

s_xor_b32 s0, s18, s19 // (op 926) 

s_sub_i32 s18, s0, s19 // (op 926) 


s_mul_i32 s0, s1, s11 // (op 926) 

s_sub_u32 s1, s16, s0 // (op 926) 

s_mul_i32 s0, s18, s11 // (op 926) 
s_add_u32 s0, s0, s1 // (op 926) 


s_mul_hi_i32 s1, s16, s9 // (op 926) 
s_add_i32 s18, s1, s16 // (op 926) 

s_ashr_i32 s1, s18, 31 // (op 926) 
s_cmp_eq_i32 s9, 0 // (op 926) 
s_mov_b32 s19, scc // (op 926) 
s_cmp_eq_i32 s19, 0 // (op 926) 
s_cselect_b32 s22, 0, -1 // (op 926) 

s_and_b32 s19, s47, 31 // (op 926) 
s_lshl_b32 s23, 1, s19 // (op 926) 
s_add_i32 s24, s23, s22 // (op 926) 


s_and_b32 s25, s1, s24 // (op 926) 


s_add_i32 s1, s18, s25 // (op 926) 


s_ashr_i32 s18, s1, s19 // (op 926) 


s_lshl_b32 s1, s47, 24 // (op 926) 
s_ashr_i32 s19, s1, 31 // (op 926) 

s_xor_b32 s1, s18, s19 // (op 926) 

s_sub_i32 s18, s1, s19 // (op 926) 


s_add_u32 s22, s6, 32 // (op 926) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 926) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 926) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 926) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 926) 

s_mul_i32 s1, s11, s22 // (op 926) 

s_mul_i32 s19, s8, s1 // (op 926) 

s_mul_i32 s1, s18, s19 // (op 926) 

s_sub_u32 s19, s16, s1 // (op 926) 

s_mov_b32 s1, s19 // (op 926) 
s_mul_hi_i32 s22, s1, s12 // (op 926) 
s_add_i32 s23, s22, s1 // (op 926) 


s_ashr_i32 s1, s23, 31 // (op 926) 
s_cmp_eq_i32 s12, 0 // (op 926) 
s_mov_b32 s22, scc // (op 926) 
s_cmp_eq_i32 s22, 0 // (op 926) 
s_cselect_b32 s24, 0, -1 // (op 926) 

s_and_b32 s22, s46, 31 // (op 926) 
s_lshl_b32 s25, 1, s22 // (op 926) 
s_add_i32 s49, s25, s24 // (op 926) 


s_and_b32 s24, s1, s49 // (op 926) 


s_add_i32 s1, s23, s24 // (op 926) 


s_ashr_i32 s23, s1, s22 // (op 926) 


s_lshl_b32 s1, s46, 24 // (op 926) 
s_ashr_i32 s22, s1, 31 // (op 926) 

s_xor_b32 s1, s23, s22 // (op 926) 

s_sub_i32 s23, s1, s22 // (op 926) 


s_mul_i32 s1, s48, s11 // (op 926) 
s_add_u32 s24, s4, 64 // (op 926) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 926) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 926) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 926) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 926) 

s_sub_i32 s22, s24, s1 // (op 926) 

s_mul_i32 s24, s23, s22 // (op 926) 


s_sub_u32 s25, s19, s24 // (op 926) 


s_add_u32 s19, s25, s1 // (op 926) 


s_cmp_eq_i32 s18, 0 // (op 926) 
s_mov_b32 s1, scc // (op 926) 

s_cmp_eq_i32 s1, 0 // (op 926) 
s_cselect_b32 s18, s19, s0 // (op 926) 



v_lshl_add_u32 v14, s18, 6, v12 // (op 926) 


v_mov_b32 v17, 0 // (op 926) convert
v_mov_b32 v16, v14 // (op 926) convert

v_mul_lo_u32 v15, v13, s27 // (op 926) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 926) most significant: high of low * low
v_mul_lo_u32 v14, v13, s26 // (op 926) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 926) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 926) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 926) most significant half



s_mul_i32 s1, s26, s31 // (op 928) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 928) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 928) least significant
s_mul_hi_u32 s19, s26, s30 // (op 928) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 928) most significant: sum
s_add_u32 s1, s1, s19 // (op 928) most significant: sum


s_mov_b32 s52, s28 // (op 928) reg expression
s_mov_b32 s53, s29 // (op 928) reg expression
s_mov_b32 s54, s0 // (op 928) reg expression
s_mov_b32 s55, 131072 // (op 928) call()

s_mul_hi_i32 s0, s16, s10 // (op 945) 
v_and_b32 v14, -4, v1 // (op 945) 
s_add_i32 s1, s0, s16 // (op 945) 

v_sub_u32 v15, v1, v14 // (op 945) 

s_ashr_i32 s0, s1, 31 // (op 945) 
v_lshlrev_b32 v14, 4, v15 // (op 945) 

s_cmp_eq_i32 s10, 0 // (op 945) 
s_mov_b32 s18, scc // (op 945) 
v_lshrrev_b32 v15, 2, v1 // (op 945) 
s_cmp_eq_i32 s18, 0 // (op 945) 
s_cselect_b32 s19, 0, -1 // (op 945) 

s_and_b32 s18, s13, 31 // (op 945) 
s_lshl_b32 s22, 1, s18 // (op 945) 
s_add_i32 s23, s22, s19 // (op 945) 


s_and_b32 s19, s0, s23 // (op 945) 


s_add_i32 s0, s1, s19 // (op 945) 


s_ashr_i32 s1, s0, s18 // (op 945) 


s_lshl_b32 s0, s13, 24 // (op 945) 
s_ashr_i32 s18, s0, 31 // (op 945) 

s_xor_b32 s0, s1, s18 // (op 945) 

s_sub_i32 s1, s0, s18 // (op 945) 


s_mul_hi_i32 s0, s1, s14 // (op 945) 
s_add_i32 s18, s0, s1 // (op 945) 

s_ashr_i32 s0, s18, 31 // (op 945) 
s_cmp_eq_i32 s14, 0 // (op 945) 
s_mov_b32 s19, scc // (op 945) 
s_cmp_eq_i32 s19, 0 // (op 945) 
s_cselect_b32 s22, 0, -1 // (op 945) 

s_and_b32 s19, s15, 31 // (op 945) 
s_lshl_b32 s23, 1, s19 // (op 945) 
s_add_i32 s24, s23, s22 // (op 945) 


s_and_b32 s25, s0, s24 // (op 945) 


s_add_i32 s0, s18, s25 // (op 945) 


s_ashr_i32 s18, s0, s19 // (op 945) 


s_lshl_b32 s0, s15, 24 // (op 945) 
s_ashr_i32 s19, s0, 31 // (op 945) 

s_xor_b32 s0, s18, s19 // (op 945) 

s_sub_i32 s18, s0, s19 // (op 945) 


s_mul_i32 s0, s1, s11 // (op 945) 

s_sub_u32 s1, s16, s0 // (op 945) 

s_mul_i32 s0, s18, s11 // (op 945) 
s_add_u32 s0, s0, s1 // (op 945) 


s_mul_hi_i32 s1, s16, s9 // (op 945) 
s_add_i32 s18, s1, s16 // (op 945) 

s_ashr_i32 s1, s18, 31 // (op 945) 
s_cmp_eq_i32 s9, 0 // (op 945) 
s_mov_b32 s19, scc // (op 945) 
s_cmp_eq_i32 s19, 0 // (op 945) 
s_cselect_b32 s22, 0, -1 // (op 945) 

s_and_b32 s19, s47, 31 // (op 945) 
s_lshl_b32 s23, 1, s19 // (op 945) 
s_add_i32 s24, s23, s22 // (op 945) 


s_and_b32 s25, s1, s24 // (op 945) 


s_add_i32 s1, s18, s25 // (op 945) 


s_ashr_i32 s18, s1, s19 // (op 945) 


s_lshl_b32 s1, s47, 24 // (op 945) 
s_ashr_i32 s19, s1, 31 // (op 945) 

s_xor_b32 s1, s18, s19 // (op 945) 

s_sub_i32 s18, s1, s19 // (op 945) 


s_add_u32 s50, s6, 32 // (op 945) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 945) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 945) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 945) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 945) 

s_mul_i32 s1, s11, s24 // (op 945) 

s_mul_i32 s19, s8, s1 // (op 945) 

s_mul_i32 s1, s18, s19 // (op 945) 

s_sub_u32 s19, s16, s1 // (op 945) 

s_mov_b32 s1, s19 // (op 945) 
s_mul_hi_i32 s22, s1, s12 // (op 945) 
s_add_i32 s23, s22, s1 // (op 945) 


s_ashr_i32 s1, s23, 31 // (op 945) 
s_cmp_eq_i32 s12, 0 // (op 945) 
s_mov_b32 s22, scc // (op 945) 
s_cmp_eq_i32 s22, 0 // (op 945) 
s_cselect_b32 s24, 0, -1 // (op 945) 

s_and_b32 s22, s46, 31 // (op 945) 
s_lshl_b32 s25, 1, s22 // (op 945) 
s_add_i32 s49, s25, s24 // (op 945) 


s_and_b32 s24, s1, s49 // (op 945) 


s_add_i32 s1, s23, s24 // (op 945) 


s_ashr_i32 s23, s1, s22 // (op 945) 


s_lshl_b32 s1, s46, 24 // (op 945) 
s_ashr_i32 s22, s1, 31 // (op 945) 

s_xor_b32 s1, s23, s22 // (op 945) 

s_sub_i32 s23, s1, s22 // (op 945) 


s_mul_i32 s1, s48, s11 // (op 945) 
s_add_u32 s24, s4, 64 // (op 945) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 945) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 945) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 945) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 945) 

s_sub_i32 s22, s24, s1 // (op 945) 

s_mul_i32 s24, s23, s22 // (op 945) 


s_sub_u32 s25, s19, s24 // (op 945) 


s_add_u32 s19, s25, s1 // (op 945) 


s_cmp_eq_i32 s18, 0 // (op 945) 
s_mov_b32 s1, scc // (op 945) 

s_cmp_eq_i32 s1, 0 // (op 945) 
s_cselect_b32 s18, s19, s0 // (op 945) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 945) 


v_mov_b32 v19, 0 // (op 945) convert
v_mov_b32 v18, v16 // (op 945) convert

v_mul_lo_u32 v17, v15, s27 // (op 945) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 945) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 945) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 945) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 945) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 945) most significant half



s_mul_i32 s1, s26, s31 // (op 947) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 947) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 947) least significant
s_mul_hi_u32 s19, s26, s30 // (op 947) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 947) most significant: sum
s_add_u32 s1, s1, s19 // (op 947) most significant: sum


s_mov_b32 s52, s28 // (op 947) reg expression
s_mov_b32 s53, s29 // (op 947) reg expression
s_mov_b32 s54, s0 // (op 947) reg expression
s_mov_b32 s55, 131072 // (op 947) call()

s_mul_hi_i32 s0, s16, s10 // (op 990) 
v_and_b32 v16, -2, v1 // (op 990) 
s_add_i32 s1, s0, s16 // (op 990) 

v_sub_u32 v17, v1, v16 // (op 990) 

s_ashr_i32 s0, s1, 31 // (op 990) 
v_lshlrev_b32 v16, 4, v17 // (op 990) 

s_cmp_eq_i32 s10, 0 // (op 990) 
s_mov_b32 s18, scc // (op 990) 
v_lshrrev_b32 v17, 1, v1 // (op 990) 
s_cmp_eq_i32 s18, 0 // (op 990) 
s_cselect_b32 s19, 0, -1 // (op 990) 

s_and_b32 s18, s13, 31 // (op 990) 
s_lshl_b32 s22, 1, s18 // (op 990) 
s_add_i32 s23, s22, s19 // (op 990) 


s_and_b32 s19, s0, s23 // (op 990) 


s_add_i32 s0, s1, s19 // (op 990) 


s_ashr_i32 s1, s0, s18 // (op 990) 


s_lshl_b32 s0, s13, 24 // (op 990) 
s_ashr_i32 s18, s0, 31 // (op 990) 

s_xor_b32 s0, s1, s18 // (op 990) 

s_sub_i32 s1, s0, s18 // (op 990) 


s_mul_hi_i32 s0, s1, s14 // (op 990) 
s_add_i32 s18, s0, s1 // (op 990) 

s_ashr_i32 s0, s18, 31 // (op 990) 
s_cmp_eq_i32 s14, 0 // (op 990) 
s_mov_b32 s19, scc // (op 990) 
s_cmp_eq_i32 s19, 0 // (op 990) 
s_cselect_b32 s22, 0, -1 // (op 990) 

s_and_b32 s19, s15, 31 // (op 990) 
s_lshl_b32 s23, 1, s19 // (op 990) 
s_add_i32 s24, s23, s22 // (op 990) 


s_and_b32 s25, s0, s24 // (op 990) 


s_add_i32 s0, s18, s25 // (op 990) 


s_ashr_i32 s18, s0, s19 // (op 990) 


s_lshl_b32 s0, s15, 24 // (op 990) 
s_ashr_i32 s19, s0, 31 // (op 990) 

s_xor_b32 s0, s18, s19 // (op 990) 

s_sub_i32 s18, s0, s19 // (op 990) 


s_add_u32 s50, s6, 32 // (op 990) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 990) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 990) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 990) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 990) 

s_mul_i32 s0, s18, s24 // (op 990) 

s_sub_i32 s18, s1, s0 // (op 990) 


s_mul_hi_i32 s19, s16, s9 // (op 990) 
s_add_i32 s0, s19, s16 // (op 990) 

s_ashr_i32 s1, s0, 31 // (op 990) 
s_cmp_eq_i32 s9, 0 // (op 990) 
s_mov_b32 s19, scc // (op 990) 
s_cmp_eq_i32 s19, 0 // (op 990) 
s_cselect_b32 s22, 0, -1 // (op 990) 

s_and_b32 s19, s47, 31 // (op 990) 
s_lshl_b32 s23, 1, s19 // (op 990) 
s_add_i32 s49, s23, s22 // (op 990) 


s_and_b32 s22, s1, s49 // (op 990) 


s_add_i32 s1, s0, s22 // (op 990) 


s_ashr_i32 s0, s1, s19 // (op 990) 


s_lshl_b32 s1, s47, 24 // (op 990) 
s_ashr_i32 s19, s1, 31 // (op 990) 

s_xor_b32 s1, s0, s19 // (op 990) 

s_sub_i32 s0, s1, s19 // (op 990) 


s_mul_i32 s1, s11, s24 // (op 990) 

s_mul_i32 s19, s8, s1 // (op 990) 

s_mul_i32 s1, s0, s19 // (op 990) 

s_sub_u32 s19, s16, s1 // (op 990) 

s_mul_hi_i32 s1, s19, s12 // (op 990) 
s_add_i32 s22, s1, s19 // (op 990) 


s_ashr_i32 s1, s22, 31 // (op 990) 
s_cmp_eq_i32 s12, 0 // (op 990) 
s_mov_b32 s19, scc // (op 990) 
s_cmp_eq_i32 s19, 0 // (op 990) 
s_cselect_b32 s23, 0, -1 // (op 990) 

s_and_b32 s19, s46, 31 // (op 990) 
s_lshl_b32 s24, 1, s19 // (op 990) 
s_add_i32 s25, s24, s23 // (op 990) 


s_and_b32 s23, s1, s25 // (op 990) 


s_add_i32 s1, s22, s23 // (op 990) 


s_ashr_i32 s22, s1, s19 // (op 990) 


s_lshl_b32 s1, s46, 24 // (op 990) 
s_ashr_i32 s19, s1, 31 // (op 990) 

s_xor_b32 s1, s22, s19 // (op 990) 

s_sub_i32 s22, s1, s19 // (op 990) 


s_cmp_eq_i32 s0, 0 // (op 990) 
s_mov_b32 s1, scc // (op 990) 

s_cmp_eq_i32 s1, 0 // (op 990) 
s_cselect_b32 s0, s22, s18 // (op 990) 



v_lshl_add_u32 v18, s0, 5, v16 // (op 990) 


v_mov_b32 v21, 0 // (op 990) convert
v_mov_b32 v20, v18 // (op 990) convert

v_mul_lo_u32 v19, v17, s33 // (op 990) most significant: low of low * high
v_mul_hi_u32 v16, v17, s32 // (op 990) most significant: high of low * low
v_mul_lo_u32 v18, v17, s32 // (op 990) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 990) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 990) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 990) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 990) most significant: low of high * low
v_mul_hi_u32 v20, v18, 6 // (op 990) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 990) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 990) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 990) 

v_mov_b32 v16, v18 // (op 990) convert

s_mul_i32 s1, s31, 6 // (op 992) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 992) least significant
s_mul_hi_u32 s18, s30, 6 // (op 992) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 992) most significant: sum

s_mul_i32 s19, s32, s1 // (op 992) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 992) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 992) least significant
s_mul_hi_u32 s23, s32, s0 // (op 992) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 992) most significant: sum
s_add_u32 s19, s19, s23 // (op 992) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 992) 

s_mov_b32 s64, s34 // (op 992) reg expression
s_mov_b32 s65, s35 // (op 992) reg expression
s_mov_b32 s66, s0 // (op 992) reg expression
s_mov_b32 s67, 131072 // (op 992) call()

s_mul_hi_i32 s0, s16, s10 // (op 1009) 
v_and_b32 v17, -2, v1 // (op 1009) 
s_add_i32 s1, s0, s16 // (op 1009) 

v_sub_u32 v18, v1, v17 // (op 1009) 

s_ashr_i32 s0, s1, 31 // (op 1009) 
v_lshlrev_b32 v17, 4, v18 // (op 1009) 

s_cmp_eq_i32 s10, 0 // (op 1009) 
s_mov_b32 s18, scc // (op 1009) 
v_lshrrev_b32 v18, 1, v1 // (op 1009) 
s_cmp_eq_i32 s18, 0 // (op 1009) 
s_cselect_b32 s19, 0, -1 // (op 1009) 

s_and_b32 s18, s13, 31 // (op 1009) 
s_lshl_b32 s22, 1, s18 // (op 1009) 
s_add_i32 s23, s22, s19 // (op 1009) 


s_and_b32 s19, s0, s23 // (op 1009) 


s_add_i32 s0, s1, s19 // (op 1009) 


s_ashr_i32 s1, s0, s18 // (op 1009) 


s_lshl_b32 s0, s13, 24 // (op 1009) 
s_ashr_i32 s18, s0, 31 // (op 1009) 

s_xor_b32 s0, s1, s18 // (op 1009) 

s_sub_i32 s1, s0, s18 // (op 1009) 


s_mul_hi_i32 s0, s1, s14 // (op 1009) 
s_add_i32 s18, s0, s1 // (op 1009) 

s_ashr_i32 s0, s18, 31 // (op 1009) 
s_cmp_eq_i32 s14, 0 // (op 1009) 
s_mov_b32 s19, scc // (op 1009) 
s_cmp_eq_i32 s19, 0 // (op 1009) 
s_cselect_b32 s22, 0, -1 // (op 1009) 

s_and_b32 s19, s15, 31 // (op 1009) 
s_lshl_b32 s23, 1, s19 // (op 1009) 
s_add_i32 s24, s23, s22 // (op 1009) 


s_and_b32 s25, s0, s24 // (op 1009) 


s_add_i32 s0, s18, s25 // (op 1009) 


s_ashr_i32 s18, s0, s19 // (op 1009) 


s_lshl_b32 s0, s15, 24 // (op 1009) 
s_ashr_i32 s19, s0, 31 // (op 1009) 

s_xor_b32 s0, s18, s19 // (op 1009) 

s_sub_i32 s18, s0, s19 // (op 1009) 


s_add_u32 s50, s6, 32 // (op 1009) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1009) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1009) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1009) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1009) 

s_mul_i32 s0, s18, s24 // (op 1009) 

s_sub_i32 s18, s1, s0 // (op 1009) 


s_mul_hi_i32 s19, s16, s9 // (op 1009) 
s_add_i32 s0, s19, s16 // (op 1009) 

s_ashr_i32 s1, s0, 31 // (op 1009) 
s_cmp_eq_i32 s9, 0 // (op 1009) 
s_mov_b32 s19, scc // (op 1009) 
s_cmp_eq_i32 s19, 0 // (op 1009) 
s_cselect_b32 s22, 0, -1 // (op 1009) 

s_and_b32 s19, s47, 31 // (op 1009) 
s_lshl_b32 s23, 1, s19 // (op 1009) 
s_add_i32 s49, s23, s22 // (op 1009) 


s_and_b32 s22, s1, s49 // (op 1009) 


s_add_i32 s1, s0, s22 // (op 1009) 


s_ashr_i32 s0, s1, s19 // (op 1009) 


s_lshl_b32 s1, s47, 24 // (op 1009) 
s_ashr_i32 s19, s1, 31 // (op 1009) 

s_xor_b32 s1, s0, s19 // (op 1009) 

s_sub_i32 s0, s1, s19 // (op 1009) 


s_mul_i32 s1, s11, s24 // (op 1009) 

s_mul_i32 s19, s8, s1 // (op 1009) 

s_mul_i32 s1, s0, s19 // (op 1009) 

s_sub_u32 s19, s16, s1 // (op 1009) 

s_mul_hi_i32 s1, s19, s12 // (op 1009) 
s_add_i32 s22, s1, s19 // (op 1009) 


s_ashr_i32 s1, s22, 31 // (op 1009) 
s_cmp_eq_i32 s12, 0 // (op 1009) 
s_mov_b32 s19, scc // (op 1009) 
s_cmp_eq_i32 s19, 0 // (op 1009) 
s_cselect_b32 s23, 0, -1 // (op 1009) 

s_and_b32 s19, s46, 31 // (op 1009) 
s_lshl_b32 s24, 1, s19 // (op 1009) 
s_add_i32 s25, s24, s23 // (op 1009) 


s_and_b32 s23, s1, s25 // (op 1009) 


s_add_i32 s1, s22, s23 // (op 1009) 


s_ashr_i32 s22, s1, s19 // (op 1009) 


s_lshl_b32 s1, s46, 24 // (op 1009) 
s_ashr_i32 s19, s1, 31 // (op 1009) 

s_xor_b32 s1, s22, s19 // (op 1009) 

s_sub_i32 s22, s1, s19 // (op 1009) 


s_cmp_eq_i32 s0, 0 // (op 1009) 
s_mov_b32 s1, scc // (op 1009) 

s_cmp_eq_i32 s1, 0 // (op 1009) 
s_cselect_b32 s0, s22, s18 // (op 1009) 



v_lshl_add_u32 v19, s0, 5, v17 // (op 1009) 


v_mov_b32 v21, 0 // (op 1009) convert
v_mov_b32 v20, v19 // (op 1009) convert

v_mul_lo_u32 v23, v18, s33 // (op 1009) most significant: low of low * high
v_mul_hi_u32 v17, v18, s32 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v22, v18, s32 // (op 1009) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 1009) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v20 // (op 1009) least significant half
v_addc_co_u32 v23, s[0:1], v23, v21, s[0:1] // (op 1009) most significant half



v_mul_lo_u32 v21, v23, 6 // (op 1009) most significant: low of high * low
v_mul_hi_u32 v17, v22, 6 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v20, v22, 6 // (op 1009) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 1009) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 1009) 

v_mov_b32 v17, v18 // (op 1009) convert

s_mul_i32 s1, s31, 6 // (op 1011) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 1011) least significant
s_mul_hi_u32 s18, s30, 6 // (op 1011) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1011) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1011) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1011) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1011) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1011) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1011) most significant: sum
s_add_u32 s19, s19, s23 // (op 1011) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1011) 

s_mov_b32 s64, s34 // (op 1011) reg expression
s_mov_b32 s65, s35 // (op 1011) reg expression
s_mov_b32 s66, s0 // (op 1011) reg expression
s_mov_b32 s67, 131072 // (op 1011) call()

s_mul_hi_i32 s0, s16, s10 // (op 1054) 
v_and_b32 v18, -64, v1 // (op 1054) 
s_add_i32 s1, s0, s16 // (op 1054) 

v_sub_u32 v19, v1, v18 // (op 1054) 

s_ashr_i32 s0, s1, 31 // (op 1054) 
v_lshrrev_b32 v18, 6, v1 // (op 1054) 
s_cmp_eq_i32 s10, 0 // (op 1054) 
s_mov_b32 s18, scc // (op 1054) 
s_cmp_eq_i32 s18, 0 // (op 1054) 
s_cselect_b32 s19, 0, -1 // (op 1054) 

s_and_b32 s18, s13, 31 // (op 1054) 
s_lshl_b32 s22, 1, s18 // (op 1054) 
s_add_i32 s23, s22, s19 // (op 1054) 


s_and_b32 s19, s0, s23 // (op 1054) 


s_add_i32 s0, s1, s19 // (op 1054) 


s_ashr_i32 s1, s0, s18 // (op 1054) 


s_lshl_b32 s0, s13, 24 // (op 1054) 
s_ashr_i32 s18, s0, 31 // (op 1054) 

s_xor_b32 s0, s1, s18 // (op 1054) 

s_sub_i32 s1, s0, s18 // (op 1054) 


s_mul_hi_i32 s0, s1, s14 // (op 1054) 
s_add_i32 s18, s0, s1 // (op 1054) 

s_ashr_i32 s0, s18, 31 // (op 1054) 
s_cmp_eq_i32 s14, 0 // (op 1054) 
s_mov_b32 s19, scc // (op 1054) 
s_cmp_eq_i32 s19, 0 // (op 1054) 
s_cselect_b32 s22, 0, -1 // (op 1054) 

s_and_b32 s19, s15, 31 // (op 1054) 
s_lshl_b32 s23, 1, s19 // (op 1054) 
s_add_i32 s24, s23, s22 // (op 1054) 


s_and_b32 s25, s0, s24 // (op 1054) 


s_add_i32 s0, s18, s25 // (op 1054) 


s_ashr_i32 s18, s0, s19 // (op 1054) 


s_lshl_b32 s0, s15, 24 // (op 1054) 
s_ashr_i32 s19, s0, 31 // (op 1054) 

s_xor_b32 s0, s18, s19 // (op 1054) 

s_sub_i32 s18, s0, s19 // (op 1054) 


s_mul_i32 s0, s1, s11 // (op 1054) 

s_sub_u32 s1, s16, s0 // (op 1054) 

s_mul_i32 s0, s18, s11 // (op 1054) 
s_add_u32 s0, s0, s1 // (op 1054) 


s_mul_hi_i32 s1, s16, s9 // (op 1054) 
s_add_i32 s18, s1, s16 // (op 1054) 

s_ashr_i32 s1, s18, 31 // (op 1054) 
s_cmp_eq_i32 s9, 0 // (op 1054) 
s_mov_b32 s19, scc // (op 1054) 
s_cmp_eq_i32 s19, 0 // (op 1054) 
s_cselect_b32 s22, 0, -1 // (op 1054) 

s_and_b32 s19, s47, 31 // (op 1054) 
s_lshl_b32 s23, 1, s19 // (op 1054) 
s_add_i32 s24, s23, s22 // (op 1054) 


s_and_b32 s25, s1, s24 // (op 1054) 


s_add_i32 s1, s18, s25 // (op 1054) 


s_ashr_i32 s18, s1, s19 // (op 1054) 


s_lshl_b32 s1, s47, 24 // (op 1054) 
s_ashr_i32 s19, s1, 31 // (op 1054) 

s_xor_b32 s1, s18, s19 // (op 1054) 

s_sub_i32 s18, s1, s19 // (op 1054) 


s_add_u32 s50, s6, 32 // (op 1054) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1054) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1054) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1054) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1054) 

s_mul_i32 s1, s11, s24 // (op 1054) 

s_mul_i32 s19, s8, s1 // (op 1054) 

s_mul_i32 s1, s18, s19 // (op 1054) 

s_sub_u32 s19, s16, s1 // (op 1054) 

s_mov_b32 s1, s19 // (op 1054) 
s_mul_hi_i32 s22, s1, s12 // (op 1054) 
s_add_i32 s23, s22, s1 // (op 1054) 


s_ashr_i32 s1, s23, 31 // (op 1054) 
s_cmp_eq_i32 s12, 0 // (op 1054) 
s_mov_b32 s22, scc // (op 1054) 
s_cmp_eq_i32 s22, 0 // (op 1054) 
s_cselect_b32 s24, 0, -1 // (op 1054) 

s_and_b32 s22, s46, 31 // (op 1054) 
s_lshl_b32 s25, 1, s22 // (op 1054) 
s_add_i32 s49, s25, s24 // (op 1054) 


s_and_b32 s24, s1, s49 // (op 1054) 


s_add_i32 s1, s23, s24 // (op 1054) 


s_ashr_i32 s23, s1, s22 // (op 1054) 


s_lshl_b32 s1, s46, 24 // (op 1054) 
s_ashr_i32 s22, s1, 31 // (op 1054) 

s_xor_b32 s1, s23, s22 // (op 1054) 

s_sub_i32 s23, s1, s22 // (op 1054) 


s_mul_i32 s1, s48, s11 // (op 1054) 
s_add_u32 s24, s4, 64 // (op 1054) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1054) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1054) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1054) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1054) 

s_sub_i32 s22, s24, s1 // (op 1054) 

s_mul_i32 s24, s23, s22 // (op 1054) 


s_sub_u32 s25, s19, s24 // (op 1054) 


s_add_u32 s19, s25, s1 // (op 1054) 


s_cmp_eq_i32 s18, 0 // (op 1054) 
s_mov_b32 s1, scc // (op 1054) 

s_cmp_eq_i32 s1, 0 // (op 1054) 
s_cselect_b32 s18, s19, s0 // (op 1054) 



v_lshl_add_u32 v20, s18, 6, v19 // (op 1054) 


v_mul_lo_u32 v19, v18, s36 // (op 1054) 

v_add_u32 v18, v20, v19 // (op 1054) 


s_mul_i32 s1, s36, s41 // (op 1056) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1056) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1056) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1056) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1056) most significant: sum
s_add_u32 s1, s1, s19 // (op 1056) most significant: sum


s_mov_b32 s68, s38 // (op 1056) reg expression
s_mov_b32 s69, s39 // (op 1056) reg expression
s_mov_b32 s70, s0 // (op 1056) reg expression
s_mov_b32 s71, 131072 // (op 1056) call()

s_mul_hi_i32 s0, s16, s10 // (op 1073) 
v_and_b32 v19, -64, v1 // (op 1073) 
s_add_i32 s1, s0, s16 // (op 1073) 

v_sub_u32 v20, v1, v19 // (op 1073) 

s_ashr_i32 s0, s1, 31 // (op 1073) 
v_lshrrev_b32 v19, 6, v1 // (op 1073) 
s_cmp_eq_i32 s10, 0 // (op 1073) 
s_mov_b32 s18, scc // (op 1073) 
s_cmp_eq_i32 s18, 0 // (op 1073) 
s_cselect_b32 s19, 0, -1 // (op 1073) 

s_and_b32 s18, s13, 31 // (op 1073) 
s_lshl_b32 s22, 1, s18 // (op 1073) 
s_add_i32 s23, s22, s19 // (op 1073) 


s_and_b32 s19, s0, s23 // (op 1073) 


s_add_i32 s0, s1, s19 // (op 1073) 


s_ashr_i32 s1, s0, s18 // (op 1073) 


s_lshl_b32 s0, s13, 24 // (op 1073) 
s_ashr_i32 s18, s0, 31 // (op 1073) 

s_xor_b32 s0, s1, s18 // (op 1073) 

s_sub_i32 s1, s0, s18 // (op 1073) 


s_mul_hi_i32 s0, s1, s14 // (op 1073) 
s_add_i32 s18, s0, s1 // (op 1073) 

s_ashr_i32 s0, s18, 31 // (op 1073) 
s_cmp_eq_i32 s14, 0 // (op 1073) 
s_mov_b32 s19, scc // (op 1073) 
s_cmp_eq_i32 s19, 0 // (op 1073) 
s_cselect_b32 s22, 0, -1 // (op 1073) 

s_and_b32 s19, s15, 31 // (op 1073) 
s_lshl_b32 s23, 1, s19 // (op 1073) 
s_add_i32 s24, s23, s22 // (op 1073) 


s_and_b32 s25, s0, s24 // (op 1073) 


s_add_i32 s0, s18, s25 // (op 1073) 


s_ashr_i32 s18, s0, s19 // (op 1073) 


s_lshl_b32 s0, s15, 24 // (op 1073) 
s_ashr_i32 s19, s0, 31 // (op 1073) 

s_xor_b32 s0, s18, s19 // (op 1073) 

s_sub_i32 s18, s0, s19 // (op 1073) 


s_mul_i32 s0, s1, s11 // (op 1073) 

s_sub_u32 s1, s16, s0 // (op 1073) 

s_mul_i32 s0, s18, s11 // (op 1073) 
s_add_u32 s0, s0, s1 // (op 1073) 


s_mul_hi_i32 s1, s16, s9 // (op 1073) 
s_add_i32 s18, s1, s16 // (op 1073) 

s_ashr_i32 s1, s18, 31 // (op 1073) 
s_cmp_eq_i32 s9, 0 // (op 1073) 
s_mov_b32 s19, scc // (op 1073) 
s_cmp_eq_i32 s19, 0 // (op 1073) 
s_cselect_b32 s22, 0, -1 // (op 1073) 

s_and_b32 s19, s47, 31 // (op 1073) 
s_lshl_b32 s23, 1, s19 // (op 1073) 
s_add_i32 s24, s23, s22 // (op 1073) 


s_and_b32 s25, s1, s24 // (op 1073) 


s_add_i32 s1, s18, s25 // (op 1073) 


s_ashr_i32 s18, s1, s19 // (op 1073) 


s_lshl_b32 s1, s47, 24 // (op 1073) 
s_ashr_i32 s19, s1, 31 // (op 1073) 

s_xor_b32 s1, s18, s19 // (op 1073) 

s_sub_i32 s18, s1, s19 // (op 1073) 


s_add_u32 s50, s6, 32 // (op 1073) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1073) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1073) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1073) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1073) 

s_mul_i32 s1, s11, s24 // (op 1073) 

s_mul_i32 s19, s8, s1 // (op 1073) 

s_mul_i32 s1, s18, s19 // (op 1073) 

s_sub_u32 s19, s16, s1 // (op 1073) 

s_mov_b32 s1, s19 // (op 1073) 
s_mul_hi_i32 s22, s1, s12 // (op 1073) 
s_add_i32 s23, s22, s1 // (op 1073) 


s_ashr_i32 s1, s23, 31 // (op 1073) 
s_cmp_eq_i32 s12, 0 // (op 1073) 
s_mov_b32 s22, scc // (op 1073) 
s_cmp_eq_i32 s22, 0 // (op 1073) 
s_cselect_b32 s24, 0, -1 // (op 1073) 

s_and_b32 s22, s46, 31 // (op 1073) 
s_lshl_b32 s25, 1, s22 // (op 1073) 
s_add_i32 s49, s25, s24 // (op 1073) 


s_and_b32 s24, s1, s49 // (op 1073) 


s_add_i32 s1, s23, s24 // (op 1073) 


s_ashr_i32 s23, s1, s22 // (op 1073) 


s_lshl_b32 s1, s46, 24 // (op 1073) 
s_ashr_i32 s22, s1, 31 // (op 1073) 

s_xor_b32 s1, s23, s22 // (op 1073) 

s_sub_i32 s23, s1, s22 // (op 1073) 


s_mul_i32 s1, s48, s11 // (op 1073) 
s_add_u32 s24, s4, 64 // (op 1073) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1073) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1073) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1073) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1073) 

s_sub_i32 s22, s24, s1 // (op 1073) 

s_mul_i32 s24, s23, s22 // (op 1073) 


s_sub_u32 s25, s19, s24 // (op 1073) 


s_add_u32 s19, s25, s1 // (op 1073) 


s_cmp_eq_i32 s18, 0 // (op 1073) 
s_mov_b32 s1, scc // (op 1073) 

s_cmp_eq_i32 s1, 0 // (op 1073) 
s_cselect_b32 s18, s19, s0 // (op 1073) 



v_lshl_add_u32 v21, s18, 6, v20 // (op 1073) 


v_mul_lo_u32 v20, v19, s36 // (op 1073) 

v_add_u32 v19, v21, v20 // (op 1073) 


s_mul_i32 s1, s36, s41 // (op 1075) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1075) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1075) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1075) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1075) most significant: sum
s_add_u32 s1, s1, s19 // (op 1075) most significant: sum


s_mov_b32 s68, s38 // (op 1075) reg expression
s_mov_b32 s69, s39 // (op 1075) reg expression
s_mov_b32 s70, s0 // (op 1075) reg expression
s_mov_b32 s71, 131072 // (op 1075) call()

s_mul_hi_i32 s0, s16, s10 // (op 1118) 
v_lshrrev_b32 v20, 6, v1 // (op 1118) 
s_add_i32 s1, s0, s16 // (op 1118) 

v_and_b32 v21, 1, v20 // (op 1118) 

s_ashr_i32 s0, s1, 31 // (op 1118) 
v_and_b32 v20, 15, v1 // (op 1118) 
s_cmp_eq_i32 s10, 0 // (op 1118) 
s_mov_b32 s18, scc // (op 1118) 
v_lshl_add_u32 v22, v21, 4, v20 // (op 1118) 


s_cmp_eq_i32 s18, 0 // (op 1118) 
s_cselect_b32 s19, 0, -1 // (op 1118) 

v_and_b32 v20, 63, v1 // (op 1118) 
s_and_b32 s18, s13, 31 // (op 1118) 
v_lshrrev_b32 v21, 4, v20 // (op 1118) 

s_lshl_b32 s22, 1, s18 // (op 1118) 
s_add_i32 s23, s22, s19 // (op 1118) 


s_and_b32 s19, s0, s23 // (op 1118) 


s_add_i32 s0, s1, s19 // (op 1118) 


s_ashr_i32 s1, s0, s18 // (op 1118) 


s_lshl_b32 s0, s13, 24 // (op 1118) 
s_ashr_i32 s18, s0, 31 // (op 1118) 

s_xor_b32 s0, s1, s18 // (op 1118) 

s_sub_i32 s1, s0, s18 // (op 1118) 


s_mul_hi_i32 s0, s1, s14 // (op 1118) 
s_add_i32 s18, s0, s1 // (op 1118) 

s_ashr_i32 s0, s18, 31 // (op 1118) 
s_cmp_eq_i32 s14, 0 // (op 1118) 
s_mov_b32 s19, scc // (op 1118) 
s_cmp_eq_i32 s19, 0 // (op 1118) 
s_cselect_b32 s22, 0, -1 // (op 1118) 

s_and_b32 s19, s15, 31 // (op 1118) 
s_lshl_b32 s23, 1, s19 // (op 1118) 
s_add_i32 s24, s23, s22 // (op 1118) 


s_and_b32 s25, s0, s24 // (op 1118) 


s_add_i32 s0, s18, s25 // (op 1118) 


s_ashr_i32 s18, s0, s19 // (op 1118) 


s_lshl_b32 s0, s15, 24 // (op 1118) 
s_ashr_i32 s19, s0, 31 // (op 1118) 

s_xor_b32 s0, s18, s19 // (op 1118) 

s_sub_i32 s18, s0, s19 // (op 1118) 


s_add_u32 s50, s6, 32 // (op 1118) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1118) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1118) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1118) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1118) 

s_mul_i32 s0, s18, s24 // (op 1118) 

s_sub_i32 s18, s1, s0 // (op 1118) 


s_mul_hi_i32 s19, s16, s9 // (op 1118) 
s_add_i32 s0, s19, s16 // (op 1118) 

s_ashr_i32 s1, s0, 31 // (op 1118) 
s_cmp_eq_i32 s9, 0 // (op 1118) 
s_mov_b32 s19, scc // (op 1118) 
s_cmp_eq_i32 s19, 0 // (op 1118) 
s_cselect_b32 s22, 0, -1 // (op 1118) 

s_and_b32 s19, s47, 31 // (op 1118) 
s_lshl_b32 s23, 1, s19 // (op 1118) 
s_add_i32 s49, s23, s22 // (op 1118) 


s_and_b32 s22, s1, s49 // (op 1118) 


s_add_i32 s1, s0, s22 // (op 1118) 


s_ashr_i32 s0, s1, s19 // (op 1118) 


s_lshl_b32 s1, s47, 24 // (op 1118) 
s_ashr_i32 s19, s1, 31 // (op 1118) 

s_xor_b32 s1, s0, s19 // (op 1118) 

s_sub_i32 s0, s1, s19 // (op 1118) 


s_mul_i32 s1, s11, s24 // (op 1118) 

s_mul_i32 s19, s8, s1 // (op 1118) 

s_mul_i32 s1, s0, s19 // (op 1118) 

s_sub_u32 s19, s16, s1 // (op 1118) 

s_mul_hi_i32 s1, s19, s12 // (op 1118) 
s_add_i32 s22, s1, s19 // (op 1118) 


s_ashr_i32 s1, s22, 31 // (op 1118) 
s_cmp_eq_i32 s12, 0 // (op 1118) 
s_mov_b32 s19, scc // (op 1118) 
s_cmp_eq_i32 s19, 0 // (op 1118) 
s_cselect_b32 s23, 0, -1 // (op 1118) 

s_and_b32 s19, s46, 31 // (op 1118) 
s_lshl_b32 s24, 1, s19 // (op 1118) 
s_add_i32 s25, s24, s23 // (op 1118) 


s_and_b32 s23, s1, s25 // (op 1118) 


s_add_i32 s1, s22, s23 // (op 1118) 


s_ashr_i32 s22, s1, s19 // (op 1118) 


s_lshl_b32 s1, s46, 24 // (op 1118) 
s_ashr_i32 s19, s1, 31 // (op 1118) 

s_xor_b32 s1, s22, s19 // (op 1118) 

s_sub_i32 s22, s1, s19 // (op 1118) 


s_cmp_eq_i32 s0, 0 // (op 1118) 
s_mov_b32 s1, scc // (op 1118) 

s_cmp_eq_i32 s1, 0 // (op 1118) 
s_cselect_b32 s0, s22, s18 // (op 1118) 



v_lshl_add_u32 v20, s0, 5, v22 // (op 1118) 


v_mul_lo_u32 v22, v21, s42 // (op 1118) 
v_add_u32 v22, v22, v20 // (op 1118) 


s_mul_i32 s1, s42, s41 // (op 1120) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1120) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1120) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1120) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1120) most significant: sum
s_add_u32 s1, s1, s19 // (op 1120) most significant: sum


s_mov_b32 s72, s44 // (op 1120) reg expression
s_mov_b32 s73, s45 // (op 1120) reg expression
s_mov_b32 s74, s0 // (op 1120) reg expression
s_mov_b32 s75, 131072 // (op 1120) call()

s_mul_hi_i32 s0, s16, s10 // (op 1137) 
v_lshrrev_b32 v20, 6, v1 // (op 1137) 
s_add_i32 s1, s0, s16 // (op 1137) 

v_and_b32 v21, 1, v20 // (op 1137) 

s_ashr_i32 s0, s1, 31 // (op 1137) 
v_and_b32 v20, 15, v1 // (op 1137) 
s_cmp_eq_i32 s10, 0 // (op 1137) 
s_mov_b32 s18, scc // (op 1137) 
v_lshl_add_u32 v23, v21, 4, v20 // (op 1137) 


s_cmp_eq_i32 s18, 0 // (op 1137) 
s_cselect_b32 s19, 0, -1 // (op 1137) 

v_and_b32 v20, 63, v1 // (op 1137) 
s_and_b32 s18, s13, 31 // (op 1137) 
v_lshrrev_b32 v21, 4, v20 // (op 1137) 

s_lshl_b32 s22, 1, s18 // (op 1137) 
s_add_i32 s23, s22, s19 // (op 1137) 


s_and_b32 s19, s0, s23 // (op 1137) 


s_add_i32 s0, s1, s19 // (op 1137) 


s_ashr_i32 s1, s0, s18 // (op 1137) 


s_lshl_b32 s0, s13, 24 // (op 1137) 
s_ashr_i32 s18, s0, 31 // (op 1137) 

s_xor_b32 s0, s1, s18 // (op 1137) 

s_sub_i32 s1, s0, s18 // (op 1137) 


s_mul_hi_i32 s0, s1, s14 // (op 1137) 
s_add_i32 s18, s0, s1 // (op 1137) 

s_ashr_i32 s0, s18, 31 // (op 1137) 
s_cmp_eq_i32 s14, 0 // (op 1137) 
s_mov_b32 s19, scc // (op 1137) 
s_cmp_eq_i32 s19, 0 // (op 1137) 
s_cselect_b32 s22, 0, -1 // (op 1137) 

s_and_b32 s19, s15, 31 // (op 1137) 
s_lshl_b32 s23, 1, s19 // (op 1137) 
s_add_i32 s24, s23, s22 // (op 1137) 


s_and_b32 s25, s0, s24 // (op 1137) 


s_add_i32 s0, s18, s25 // (op 1137) 


s_ashr_i32 s18, s0, s19 // (op 1137) 


s_lshl_b32 s0, s15, 24 // (op 1137) 
s_ashr_i32 s19, s0, 31 // (op 1137) 

s_xor_b32 s0, s18, s19 // (op 1137) 

s_sub_i32 s18, s0, s19 // (op 1137) 


s_add_u32 s50, s6, 32 // (op 1137) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1137) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1137) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1137) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1137) 

s_mul_i32 s0, s18, s24 // (op 1137) 

s_sub_i32 s18, s1, s0 // (op 1137) 


s_mul_hi_i32 s19, s16, s9 // (op 1137) 
s_add_i32 s0, s19, s16 // (op 1137) 

s_ashr_i32 s1, s0, 31 // (op 1137) 
s_cmp_eq_i32 s9, 0 // (op 1137) 
s_mov_b32 s19, scc // (op 1137) 
s_cmp_eq_i32 s19, 0 // (op 1137) 
s_cselect_b32 s22, 0, -1 // (op 1137) 

s_and_b32 s19, s47, 31 // (op 1137) 
s_lshl_b32 s23, 1, s19 // (op 1137) 
s_add_i32 s49, s23, s22 // (op 1137) 


s_and_b32 s22, s1, s49 // (op 1137) 


s_add_i32 s1, s0, s22 // (op 1137) 


s_ashr_i32 s0, s1, s19 // (op 1137) 


s_lshl_b32 s1, s47, 24 // (op 1137) 
s_ashr_i32 s19, s1, 31 // (op 1137) 

s_xor_b32 s1, s0, s19 // (op 1137) 

s_sub_i32 s0, s1, s19 // (op 1137) 


s_mul_i32 s1, s11, s24 // (op 1137) 

s_mul_i32 s19, s8, s1 // (op 1137) 

s_mul_i32 s1, s0, s19 // (op 1137) 

s_sub_u32 s19, s16, s1 // (op 1137) 

s_mul_hi_i32 s1, s19, s12 // (op 1137) 
s_add_i32 s22, s1, s19 // (op 1137) 


s_ashr_i32 s1, s22, 31 // (op 1137) 
s_cmp_eq_i32 s12, 0 // (op 1137) 
s_mov_b32 s19, scc // (op 1137) 
s_cmp_eq_i32 s19, 0 // (op 1137) 
s_cselect_b32 s23, 0, -1 // (op 1137) 

s_and_b32 s19, s46, 31 // (op 1137) 
s_lshl_b32 s24, 1, s19 // (op 1137) 
s_add_i32 s25, s24, s23 // (op 1137) 


s_and_b32 s23, s1, s25 // (op 1137) 


s_add_i32 s1, s22, s23 // (op 1137) 


s_ashr_i32 s22, s1, s19 // (op 1137) 


s_lshl_b32 s1, s46, 24 // (op 1137) 
s_ashr_i32 s19, s1, 31 // (op 1137) 

s_xor_b32 s1, s22, s19 // (op 1137) 

s_sub_i32 s22, s1, s19 // (op 1137) 


s_cmp_eq_i32 s0, 0 // (op 1137) 
s_mov_b32 s1, scc // (op 1137) 

s_cmp_eq_i32 s1, 0 // (op 1137) 
s_cselect_b32 s0, s22, s18 // (op 1137) 



v_lshl_add_u32 v20, s0, 5, v23 // (op 1137) 


v_mul_lo_u32 v23, v21, s42 // (op 1137) 
v_add_u32 v23, v23, v20 // (op 1137) 


s_mul_i32 s1, s42, s41 // (op 1139) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1139) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1139) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1139) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1139) most significant: sum
s_add_u32 s1, s1, s19 // (op 1139) most significant: sum


s_mov_b32 s72, s44 // (op 1139) reg expression
s_mov_b32 s73, s45 // (op 1139) reg expression
s_mov_b32 s74, s0 // (op 1139) reg expression
s_mov_b32 s75, 131072 // (op 1139) call()


v_mov_b64 v[20:21], s[42:43] // (op 774) convert

v_mov_b32 v24, v22 // (op 774) reg expression
v_mov_b32 v25, v24 // (op 774) 
buffer_load_ubyte v26, v25, s[72:75], 0 offen offset: 0 // (op 774) Load value





v_mov_b32 v20, v16 // (op 778) reg expression

buffer_load_dwordx3 v[28:30], v20, s[64:67], 0 offen // (op 778) Load value

v_and_b32 v27, -4, v1 // (op 1203) 
v_sub_u32 v20, v1, v27 // (op 1203) 

v_lshlrev_b32 v21, 4, v20 // (op 1203) 

v_lshrrev_b32 v20, 2, v1 // (op 1203) 
v_lshl_add_u32 v27, v20, 6, v21 // (op 1203) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1203) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 6 // (op 881) 
v_mov_b64 v[20:21], s[18:19] // (op 881) convert


v_mov_b64 v[24:25], v[12:13] // (op 881) reg expression
s_mov_b32 m0, s0 // (op 881) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 881) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 881) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 881) most significant half

s_add_u32 m0, m0, 4096 // (op 881) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 881) Load value direct to lds





v_mov_b32 v27, v18 // (op 784) reg expression

buffer_load_ubyte v20, v27, s[68:71], 0 offen offset: 0 // (op 784) Load value


v_mov_b64 v[24:25], s[42:43] // (op 788) convert

s_lshl_b32 s0, s42, 2 // (op 788) 
v_add_u32 v21, v22, s0 // (op 788) 

v_mov_b32 v27, v21 // (op 788) 
buffer_load_ubyte v31, v27, s[72:75], 0 offen offset: 0 // (op 788) Load value






s_lshl_b64 s[0:1], s[32:33], 7 // (op 792) 
s_mul_i32 s19, s1, 6 // (op 792) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 792) least significant
s_mul_hi_u32 s22, s0, 6 // (op 792) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 792) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 792) 

v_add_u32 v27, v16, s0 // (op 792) 


buffer_load_dwordx3 v[32:34], v27, s[64:67], 0 offen // (op 792) Load value


v_and_b32 v16, -4, v1 // (op 1376) 
v_sub_u32 v27, v1, v16 // (op 1376) 

v_lshlrev_b32 v16, 4, v27 // (op 1376) 

v_lshrrev_b32 v27, 2, v1 // (op 1376) 
v_lshl_add_u32 v21, v27, 6, v16 // (op 1376) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 1376) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 6 // (op 886) 
v_mov_b64 v[24:25], s[18:19] // (op 886) convert


s_lshl_b64 s[18:19], s[26:27], 7 // (op 886) 
v_mov_b32 v16, s18 // (op 886) 
v_mov_b32 v27, s19 // (op 886) 
v_add_co_u32 v36, s[50:51], v12, v16 // (op 886) least significant half
v_addc_co_u32 v37, s[50:51], v13, v27, s[50:51] // (op 886) most significant half




s_add_u32 s1, 8192, s0 // (op 886) 
s_mov_b32 m0, s1 // (op 886) reg expression
buffer_load_dwordx4 v36, s[52:55], 0 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v24 // (op 886) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 886) most significant half

s_add_u32 m0, m0, 4096 // (op 886) 
buffer_load_dwordx4 v36, s[52:55], 0 offen offset: 0 lds // (op 886) Load value direct to lds







s_lshl_b32 s0, s36, 2 // (op 798) 
v_add_u32 v16, v18, s0 // (op 798) 


buffer_load_ubyte v27, v16, s[68:71], 0 offen offset: 0 // (op 798) Load value


v_and_b32 v16, -2, v1 // (op 1261) 
v_sub_u32 v18, v1, v16 // (op 1261) 

v_lshlrev_b32 v16, 4, v18 // (op 1261) 

v_lshrrev_b32 v18, 1, v1 // (op 1261) 
v_lshl_add_u32 v12, v18, 5, v16 // (op 1261) 


v_mul_lo_u32 v13, v12, 6 // (op 1261) 
v_lshrrev_b32 v16, 3, v13 // (op 1261) 

v_lshrrev_b32 v13, 4, v12 // (op 1261) 

v_lshlrev_b32 v12, 2, v13 // (op 1261) 

v_add_u32 v13, v16, v12 // (op 1261) 





s_waitcnt vmcnt(8)
ds_write_b96 v13, v[28:30] offset:16384 // (op 800) Store local data 

v_and_b32 v16, -64, v1 // (op 1314) 
v_sub_u32 v18, v1, v16 // (op 1314) 

v_and_b32 v16, 4294967232, v1 // (op 1314) 
v_add_u32 v12, v16, v18 // (op 1314) 





s_waitcnt vmcnt(5)
ds_write_b8 v12, v20 offset:20480 // (op 802) Store local data 

v_and_b32 v16, -4, v1 // (op 1175) 
v_sub_u32 v18, v1, v16 // (op 1175) 

v_lshlrev_b32 v16, 4, v18 // (op 1175) 

v_lshrrev_b32 v18, 2, v1 // (op 1175) 
v_lshl_add_u32 v12, v18, 6, v16 // (op 1175) 


s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1175) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -2, v1 // (op 1232) 
v_sub_u32 v18, v1, v16 // (op 1232) 

v_lshlrev_b32 v16, 4, v18 // (op 1232) 

v_lshrrev_b32 v18, 1, v1 // (op 1232) 
v_lshl_add_u32 v12, v18, 5, v16 // (op 1232) 


v_mul_lo_u32 v13, v12, 6 // (op 1232) 
v_lshrrev_b32 v16, 3, v13 // (op 1232) 

v_lshrrev_b32 v13, 4, v12 // (op 1232) 

v_lshlrev_b32 v12, 2, v13 // (op 1232) 

v_add_u32 v13, v16, v12 // (op 1232) 


v_and_b32 v12, -64, v1 // (op 1285) 
v_sub_u32 v16, v1, v12 // (op 1285) 

v_and_b32 v12, 4294967232, v1 // (op 1285) 
v_add_u32 v18, v12, v16 // (op 1285) 


v_and_b32 v12, -4, v1 // (op 1361) 
v_sub_u32 v16, v1, v12 // (op 1361) 

v_lshlrev_b32 v12, 4, v16 // (op 1361) 

v_lshrrev_b32 v16, 2, v1 // (op 1361) 
v_lshl_add_u32 v21, v16, 6, v12 // (op 1361) 


s_nop 0
v_readfirstlane_b32 s1, v21 // (op 1361) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -2, v1 // (op 1405) 
v_sub_u32 v16, v1, v12 // (op 1405) 

v_lshlrev_b32 v12, 4, v16 // (op 1405) 

v_lshrrev_b32 v16, 1, v1 // (op 1405) 
v_lshl_add_u32 v21, v16, 5, v12 // (op 1405) 


v_mul_lo_u32 v12, v21, 6 // (op 1405) 
v_lshrrev_b32 v16, 3, v12 // (op 1405) 

v_lshrrev_b32 v12, 4, v21 // (op 1405) 

v_lshlrev_b32 v21, 2, v12 // (op 1405) 

v_add_u32 v12, v16, v21 // (op 1405) 


v_and_b32 v16, -64, v1 // (op 1429) 
v_sub_u32 v21, v1, v16 // (op 1429) 

v_and_b32 v16, 4294967232, v1 // (op 1429) 
v_add_u32 v22, v16, v21 // (op 1429) 


s_mov_b32 s18, 0 // (op 311) call()
s_ashr_i64 s[50:51], s[30:31], 7 // (op 310) 
s_mov_b32 s19, s50 // (op 310) convert

s_and_b32 s22, s19, 4294967294 // (op 310) 

s_cmp_lt_u32 s18, s22 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_310 if false)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_310:
s_barrier  // (op 1487) 
s_add_u32 s18, s18, 2 // (op 312) 



ds_read_b64_tr_b8 v[36:37], v4 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:512 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:4096 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:4608 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1500) 



ds_read_b96_tr_b6 v[44:46], v0 offset:16384 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[47:49], v0 offset:16896 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1514) 


ds_read_u8 v16, v7 offset:20480 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:49], a[0:3], v16, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 82) 
ds_read_b64_tr_b8 v[50:51], v4 offset:32 // (op 340) Transpose load local data 


ds_read_b64_tr_b8 v[52:53], v4 offset:544 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:4128 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:4640 // (op 340) Transpose load local data 


ds_read_u8 v16, v7 offset:20512 // (op 344) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[44:49], a[4:7], v16, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 346) 




v_mov_b64 v[24:25], s[42:43] // (op 80) convert

v_mov_b32 v16, v23 // (op 80) 
s_lshl_b32 s19, s42, 3 // (op 80) 
v_add_u32 v21, v16, s19 // (op 80) 

v_mov_b32 v35, v21 // (op 80) 
buffer_load_ubyte v26, v35, s[72:75], 0 offen offset: 0 // (op 80) Load value





v_mov_b32 v21, v17 // (op 10) 
s_lshl_b64 s[50:51], s[32:33], 7 // (op 10) 
s_mul_i32 s23, s51, 6 // (op 10) most significant: low of high * low
s_mul_i32 s22, s50, 6 // (op 10) least significant
s_mul_hi_u32 s19, s50, 6 // (op 10) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 10) most significant: sum


s_lshr_b64 s[24:25], s[22:23], 3 // (op 10) 

s_lshl_b32 s19, s24, 1 // (op 10) 

v_add_u32 v24, v21, s19 // (op 10) 


buffer_load_dwordx3 v[28:30], v24, s[64:67], 0 offen // (op 10) Load value

s_barrier  // (op 1490) 

s_lshl_b64 s[50:51], s[26:27], 6 // (op 865) 
v_mov_b64 v[24:25], s[50:51] // (op 865) convert


v_mov_b64 v[36:37], v[14:15] // (op 865) 
s_lshl_b64 s[50:51], s[26:27], 7 // (op 865) 
s_lshl_b64 s[22:23], s[50:51], 1 // (op 865) 

v_mov_b32 v35, s22 // (op 865) 
v_mov_b32 v38, s23 // (op 865) 
v_add_co_u32 v40, s[24:25], v36, v35 // (op 865) least significant half
v_addc_co_u32 v41, s[24:25], v37, v38, s[24:25] // (op 865) most significant half




s_mov_b32 m0, s0 // (op 865) reg expression
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 865) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v24 // (op 865) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 865) most significant half

s_add_u32 m0, m0, 4096 // (op 865) 
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 865) Load value direct to lds




v_mov_b32 v35, v19 // (op 17) 
s_lshl_b32 s19, s36, 3 // (op 17) 
v_add_u32 v24, v35, s19 // (op 17) 


buffer_load_ubyte v20, v24, s[68:71], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1537) 



s_waitcnt vmcnt(8)
ds_write_b96 v12, v[32:34] offset:20736 // (op 235) Store local data 
s_barrier  // (op 1543) 



s_waitcnt vmcnt(5)
ds_write_b8 v22, v27 offset:24832 // (op 236) Store local data 
s_barrier  // (op 1528) 



ds_read_b64_tr_b8 v[38:39], v10 offset:8192 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v10 offset:8704 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v10 offset:12288 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v10 offset:12800 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1534) 



ds_read_b96_tr_b6 v[46:48], v3 offset:20736 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v3 offset:21248 // (op 229) Transpose load local data 
s_barrier  // (op 1572) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1540) 


ds_read_u8 v24, v11 offset:24832 // (op 231) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:51], a[0:3], v24, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 234) 
ds_read_b64_tr_b8 v[52:53], v10 offset:8224 // (op 353) Transpose load local data 


ds_read_b64_tr_b8 v[54:55], v10 offset:8736 // (op 353) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v10 offset:12320 // (op 353) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v10 offset:12832 // (op 353) Transpose load local data 


ds_read_u8 v24, v11 offset:24864 // (op 357) Load local data 
s_barrier  // (op 1610) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[46:51], a[4:7], v24, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 347) 




v_mov_b64 v[24:25], s[42:43] // (op 233) convert

s_lshl_b32 s19, s42, 2 // (op 233) 
s_mul_i32 s22, 3, s19 // (op 233) 

v_add_u32 v38, v16, s22 // (op 233) 

v_mov_b32 v39, v38 // (op 233) 
buffer_load_ubyte v31, v39, s[72:75], 0 offen offset: 0 // (op 233) Load value






s_lshl_b32 s19, s42, 3 // (op 1129) 
v_add_u32 v23, v23, s19 // (op 1129) 

s_lshl_b64 s[50:51], s[32:33], 7 // (op 237) 
s_mul_i32 s23, s51, 6 // (op 237) most significant: low of high * low
s_mul_i32 s22, s50, 6 // (op 237) least significant
s_mul_hi_u32 s19, s50, 6 // (op 237) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 237) most significant: sum


s_lshr_b64 s[24:25], s[22:23], 3 // (op 237) 

s_mul_i32 s19, 3, s24 // (op 237) 

v_add_u32 v16, v21, s19 // (op 237) 


buffer_load_dwordx3 v[32:34], v16, s[64:67], 0 offen // (op 237) Load value


s_barrier  // (op 1531) 
s_lshl_b64 s[50:51], s[32:33], 7 // (op 1001) 

s_mul_i32 s23, s51, 6 // (op 1001) most significant: low of high * low
s_mul_i32 s22, s50, 6 // (op 1001) least significant
s_mul_hi_u32 s19, s50, 6 // (op 1001) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 1001) most significant: sum


s_lshr_b64 s[24:25], s[22:23], 3 // (op 1001) 

s_lshl_b32 s19, s24, 1 // (op 1001) 

v_add_u32 v17, v17, s19 // (op 1001) 

s_lshl_b64 s[50:51], s[26:27], 6 // (op 876) 
v_mov_b64 v[24:25], s[50:51] // (op 876) convert


s_lshl_b64 s[50:51], s[26:27], 7 // (op 876) 
s_mul_i32 s23, 3, s51 // (op 876) most significant: low of low * high
s_mul_i32 s22, 3, s50 // (op 876) least significant
s_mul_hi_u32 s19, 3, s50 // (op 876) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 876) most significant: sum


v_mov_b32 v16, s22 // (op 876) 
v_mov_b32 v21, s23 // (op 876) 
v_add_co_u32 v38, s[24:25], v36, v16 // (op 876) least significant half
v_addc_co_u32 v39, s[24:25], v37, v21, s[24:25] // (op 876) most significant half




s_add_u32 s19, 8192, s1 // (op 876) 
s_mov_b32 m0, s19 // (op 876) reg expression
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 876) Load value direct to lds
v_add_co_u32 v38, s[50:51], v38, v24 // (op 876) least significant half
v_addc_co_u32 v39, s[50:51], 0, v39, s[50:51] // (op 876) most significant half

s_add_u32 m0, m0, 4096 // (op 876) 
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 876) Load value direct to lds






s_lshl_b32 s19, s36, 2 // (op 238) 
s_mul_i32 s22, 3, s19 // (op 238) 

v_add_u32 v16, v35, s22 // (op 238) 


s_lshl_b64 s[50:51], s[26:27], 7 // (op 937) 
s_lshl_b64 s[22:23], s[50:51], 1 // (op 937) 

v_mov_b32 v21, s22 // (op 937) 
v_mov_b32 v24, s23 // (op 937) 
v_add_co_u32 v14, s[24:25], v14, v21 // (op 937) least significant half
v_addc_co_u32 v15, s[24:25], v15, v24, s[24:25] // (op 937) most significant half




buffer_load_ubyte v27, v16, s[68:71], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1503) 
s_lshl_b32 s19, s36, 3 // (op 1065) 



v_add_u32 v19, v19, s19 // (op 1065) 

s_waitcnt vmcnt(8)
ds_write_b96 v13, v[28:30] offset:16384 // (op 13) Store local data 
s_barrier  // (op 1517) 



s_waitcnt vmcnt(5)
ds_write_b8 v18, v20 offset:20480 // (op 21) Store local data 
s_ashr_i64 s[50:51], s[30:31], 7 // (op 310) 
s_mov_b32 s19, s50 // (op 310) convert

s_and_b32 s22, s19, 4294967294 // (op 310) 

s_cmp_lt_u32 s18, s22 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_310 if true)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_310:














s_barrier  // (op 1592) 
s_mul_hi_i32 s0, s16, s10 // (op 899) 
v_and_b32 v12, -4, v1 // (op 899) 
s_add_i32 s1, s0, s16 // (op 899) 

v_sub_u32 v13, v1, v12 // (op 899) 

s_ashr_i32 s0, s1, 31 // (op 899) 
v_lshlrev_b32 v12, 4, v13 // (op 899) 

s_cmp_eq_i32 s10, 0 // (op 899) 
s_mov_b32 s18, scc // (op 899) 
v_lshrrev_b32 v13, 2, v1 // (op 899) 
s_cmp_eq_i32 s18, 0 // (op 899) 
s_cselect_b32 s19, 0, -1 // (op 899) 

s_and_b32 s18, s13, 31 // (op 899) 
s_lshl_b32 s22, 1, s18 // (op 899) 
s_add_i32 s23, s22, s19 // (op 899) 


s_and_b32 s19, s0, s23 // (op 899) 


s_add_i32 s0, s1, s19 // (op 899) 


s_ashr_i32 s1, s0, s18 // (op 899) 


s_lshl_b32 s0, s13, 24 // (op 899) 
s_ashr_i32 s18, s0, 31 // (op 899) 

s_xor_b32 s0, s1, s18 // (op 899) 

s_sub_i32 s1, s0, s18 // (op 899) 


s_mul_hi_i32 s0, s1, s14 // (op 899) 
s_add_i32 s18, s0, s1 // (op 899) 

s_ashr_i32 s0, s18, 31 // (op 899) 
s_cmp_eq_i32 s14, 0 // (op 899) 
s_mov_b32 s19, scc // (op 899) 
s_cmp_eq_i32 s19, 0 // (op 899) 
s_cselect_b32 s22, 0, -1 // (op 899) 

s_and_b32 s19, s15, 31 // (op 899) 
s_lshl_b32 s23, 1, s19 // (op 899) 
s_add_i32 s24, s23, s22 // (op 899) 


s_and_b32 s25, s0, s24 // (op 899) 


s_add_i32 s0, s18, s25 // (op 899) 


s_ashr_i32 s18, s0, s19 // (op 899) 


s_lshl_b32 s0, s15, 24 // (op 899) 
s_ashr_i32 s19, s0, 31 // (op 899) 

s_xor_b32 s0, s18, s19 // (op 899) 

s_sub_i32 s18, s0, s19 // (op 899) 


s_mul_i32 s0, s1, s11 // (op 899) 

s_sub_u32 s1, s16, s0 // (op 899) 

s_mul_i32 s0, s18, s11 // (op 899) 
s_add_u32 s0, s0, s1 // (op 899) 


s_mul_hi_i32 s1, s16, s9 // (op 899) 
s_add_i32 s18, s1, s16 // (op 899) 

s_ashr_i32 s1, s18, 31 // (op 899) 
s_cmp_eq_i32 s9, 0 // (op 899) 
s_mov_b32 s19, scc // (op 899) 
s_cmp_eq_i32 s19, 0 // (op 899) 
s_cselect_b32 s22, 0, -1 // (op 899) 

s_and_b32 s19, s47, 31 // (op 899) 
s_lshl_b32 s23, 1, s19 // (op 899) 
s_add_i32 s24, s23, s22 // (op 899) 


s_and_b32 s25, s1, s24 // (op 899) 


s_add_i32 s1, s18, s25 // (op 899) 


s_ashr_i32 s18, s1, s19 // (op 899) 


s_lshl_b32 s1, s47, 24 // (op 899) 
s_ashr_i32 s19, s1, 31 // (op 899) 

s_xor_b32 s1, s18, s19 // (op 899) 

s_sub_i32 s18, s1, s19 // (op 899) 


s_add_u32 s50, s6, 32 // (op 899) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 899) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 899) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 899) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 899) 

s_mul_i32 s1, s11, s24 // (op 899) 

s_mul_i32 s19, s8, s1 // (op 899) 

s_mul_i32 s1, s18, s19 // (op 899) 

s_sub_u32 s19, s16, s1 // (op 899) 

s_mov_b32 s1, s19 // (op 899) 
s_mul_hi_i32 s22, s1, s12 // (op 899) 
s_add_i32 s23, s22, s1 // (op 899) 


s_ashr_i32 s1, s23, 31 // (op 899) 
s_cmp_eq_i32 s12, 0 // (op 899) 
s_mov_b32 s22, scc // (op 899) 
s_cmp_eq_i32 s22, 0 // (op 899) 
s_cselect_b32 s24, 0, -1 // (op 899) 

s_and_b32 s22, s46, 31 // (op 899) 
s_lshl_b32 s25, 1, s22 // (op 899) 
s_add_i32 s49, s25, s24 // (op 899) 


s_and_b32 s24, s1, s49 // (op 899) 


s_add_i32 s1, s23, s24 // (op 899) 


s_ashr_i32 s23, s1, s22 // (op 899) 


s_lshl_b32 s1, s46, 24 // (op 899) 
s_ashr_i32 s22, s1, 31 // (op 899) 

s_xor_b32 s1, s23, s22 // (op 899) 

s_sub_i32 s23, s1, s22 // (op 899) 


s_mul_i32 s1, s48, s11 // (op 899) 
s_add_u32 s24, s4, 64 // (op 899) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 899) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 899) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 899) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 899) 

s_sub_i32 s22, s24, s1 // (op 899) 

s_mul_i32 s24, s23, s22 // (op 899) 


s_sub_u32 s25, s19, s24 // (op 899) 


s_add_u32 s19, s25, s1 // (op 899) 


s_cmp_eq_i32 s18, 0 // (op 899) 
s_mov_b32 s1, scc // (op 899) 

s_cmp_eq_i32 s1, 0 // (op 899) 
s_cselect_b32 s18, s19, s0 // (op 899) 



v_lshl_add_u32 v14, s18, 6, v12 // (op 899) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 899) 
v_mov_b32 v29, 0 // (op 899) convert
v_mov_b32 v28, v14 // (op 899) convert

s_lshr_b32 s18, s0, 1 // (op 899) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 899) 


v_mul_lo_u32 v15, v12, s27 // (op 899) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 899) most significant: high of low * low
v_mul_lo_u32 v14, v12, s26 // (op 899) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 899) most significant: sum


v_add_co_u32 v12, s[0:1], v28, v14 // (op 899) least significant half
v_addc_co_u32 v13, s[0:1], v29, v15, s[0:1] // (op 899) most significant half



s_mul_i32 s1, s26, s31 // (op 901) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 901) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 901) least significant
s_mul_hi_u32 s19, s26, s30 // (op 901) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 901) most significant: sum
s_add_u32 s1, s1, s19 // (op 901) most significant: sum


s_mov_b32 s52, s28 // (op 901) reg expression
s_mov_b32 s53, s29 // (op 901) reg expression
s_mov_b32 s54, s0 // (op 901) reg expression
s_mov_b32 s55, 131072 // (op 901) call()

s_mul_hi_i32 s0, s16, s10 // (op 967) 
v_and_b32 v14, -2, v1 // (op 967) 
s_add_i32 s1, s0, s16 // (op 967) 

v_sub_u32 v15, v1, v14 // (op 967) 

s_ashr_i32 s0, s1, 31 // (op 967) 
v_lshlrev_b32 v14, 4, v15 // (op 967) 

s_cmp_eq_i32 s10, 0 // (op 967) 
s_mov_b32 s18, scc // (op 967) 
v_lshrrev_b32 v15, 1, v1 // (op 967) 
s_cmp_eq_i32 s18, 0 // (op 967) 
s_cselect_b32 s19, 0, -1 // (op 967) 

s_and_b32 s18, s13, 31 // (op 967) 
s_lshl_b32 s22, 1, s18 // (op 967) 
s_add_i32 s23, s22, s19 // (op 967) 


s_and_b32 s19, s0, s23 // (op 967) 


s_add_i32 s0, s1, s19 // (op 967) 


s_ashr_i32 s1, s0, s18 // (op 967) 


s_lshl_b32 s0, s13, 24 // (op 967) 
s_ashr_i32 s18, s0, 31 // (op 967) 

s_xor_b32 s0, s1, s18 // (op 967) 

s_sub_i32 s1, s0, s18 // (op 967) 


s_mul_hi_i32 s0, s1, s14 // (op 967) 
s_add_i32 s18, s0, s1 // (op 967) 

s_ashr_i32 s0, s18, 31 // (op 967) 
s_cmp_eq_i32 s14, 0 // (op 967) 
s_mov_b32 s19, scc // (op 967) 
s_cmp_eq_i32 s19, 0 // (op 967) 
s_cselect_b32 s22, 0, -1 // (op 967) 

s_and_b32 s19, s15, 31 // (op 967) 
s_lshl_b32 s23, 1, s19 // (op 967) 
s_add_i32 s24, s23, s22 // (op 967) 


s_and_b32 s25, s0, s24 // (op 967) 


s_add_i32 s0, s18, s25 // (op 967) 


s_ashr_i32 s18, s0, s19 // (op 967) 


s_lshl_b32 s0, s15, 24 // (op 967) 
s_ashr_i32 s19, s0, 31 // (op 967) 

s_xor_b32 s0, s18, s19 // (op 967) 

s_sub_i32 s18, s0, s19 // (op 967) 


s_add_u32 s50, s6, 32 // (op 967) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 967) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 967) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 967) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 967) 

s_mul_i32 s0, s18, s24 // (op 967) 

s_sub_i32 s18, s1, s0 // (op 967) 


s_mul_hi_i32 s19, s16, s9 // (op 967) 
s_add_i32 s0, s19, s16 // (op 967) 

s_ashr_i32 s1, s0, 31 // (op 967) 
s_cmp_eq_i32 s9, 0 // (op 967) 
s_mov_b32 s19, scc // (op 967) 
s_cmp_eq_i32 s19, 0 // (op 967) 
s_cselect_b32 s22, 0, -1 // (op 967) 

s_and_b32 s19, s47, 31 // (op 967) 
s_lshl_b32 s23, 1, s19 // (op 967) 
s_add_i32 s49, s23, s22 // (op 967) 


s_and_b32 s22, s1, s49 // (op 967) 


s_add_i32 s1, s0, s22 // (op 967) 


s_ashr_i32 s0, s1, s19 // (op 967) 


s_lshl_b32 s1, s47, 24 // (op 967) 
s_ashr_i32 s19, s1, 31 // (op 967) 

s_xor_b32 s1, s0, s19 // (op 967) 

s_sub_i32 s0, s1, s19 // (op 967) 


s_mul_i32 s1, s11, s24 // (op 967) 

s_mul_i32 s19, s8, s1 // (op 967) 

s_mul_i32 s1, s0, s19 // (op 967) 

s_sub_u32 s19, s16, s1 // (op 967) 

s_mul_hi_i32 s1, s19, s12 // (op 967) 
s_add_i32 s22, s1, s19 // (op 967) 


s_ashr_i32 s1, s22, 31 // (op 967) 
s_cmp_eq_i32 s12, 0 // (op 967) 
s_mov_b32 s19, scc // (op 967) 
s_cmp_eq_i32 s19, 0 // (op 967) 
s_cselect_b32 s23, 0, -1 // (op 967) 

s_and_b32 s19, s46, 31 // (op 967) 
s_lshl_b32 s24, 1, s19 // (op 967) 
s_add_i32 s25, s24, s23 // (op 967) 


s_and_b32 s23, s1, s25 // (op 967) 


s_add_i32 s1, s22, s23 // (op 967) 


s_ashr_i32 s22, s1, s19 // (op 967) 


s_lshl_b32 s1, s46, 24 // (op 967) 
s_ashr_i32 s19, s1, 31 // (op 967) 

s_xor_b32 s1, s22, s19 // (op 967) 

s_sub_i32 s22, s1, s19 // (op 967) 


s_cmp_eq_i32 s0, 0 // (op 967) 
s_mov_b32 s1, scc // (op 967) 

s_cmp_eq_i32 s1, 0 // (op 967) 
s_cselect_b32 s0, s22, s18 // (op 967) 



v_lshl_add_u32 v16, s0, 5, v14 // (op 967) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 967) 
v_mov_b32 v29, 0 // (op 967) convert
v_mov_b32 v28, v16 // (op 967) convert

s_lshr_b32 s18, s0, 1 // (op 967) 

v_lshl_add_u32 v14, s18, 8, v15 // (op 967) 


v_mul_lo_u32 v17, v14, s33 // (op 967) most significant: low of low * high
v_mul_hi_u32 v15, v14, s32 // (op 967) most significant: high of low * low
v_mul_lo_u32 v16, v14, s32 // (op 967) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 967) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v28 // (op 967) least significant half
v_addc_co_u32 v17, s[0:1], v17, v29, s[0:1] // (op 967) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 967) most significant: low of high * low
v_mul_hi_u32 v18, v16, 6 // (op 967) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 967) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 967) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 967) 

v_mov_b32 v14, v16 // (op 967) convert

s_mul_i32 s1, s31, 6 // (op 969) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 969) least significant
s_mul_hi_u32 s18, s30, 6 // (op 969) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 969) most significant: sum

s_mul_i32 s19, s32, s1 // (op 969) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 969) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 969) least significant
s_mul_hi_u32 s23, s32, s0 // (op 969) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 969) most significant: sum
s_add_u32 s19, s19, s23 // (op 969) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 969) 

s_mov_b32 s64, s34 // (op 969) reg expression
s_mov_b32 s65, s35 // (op 969) reg expression
s_mov_b32 s66, s0 // (op 969) reg expression
s_mov_b32 s67, 131072 // (op 969) call()

s_mul_hi_i32 s0, s16, s10 // (op 1031) 
v_and_b32 v15, -64, v1 // (op 1031) 
s_add_i32 s1, s0, s16 // (op 1031) 

v_sub_u32 v16, v1, v15 // (op 1031) 

s_ashr_i32 s0, s1, 31 // (op 1031) 
v_lshrrev_b32 v15, 6, v1 // (op 1031) 
s_cmp_eq_i32 s10, 0 // (op 1031) 
s_mov_b32 s18, scc // (op 1031) 
s_cmp_eq_i32 s18, 0 // (op 1031) 
s_cselect_b32 s19, 0, -1 // (op 1031) 

s_and_b32 s18, s13, 31 // (op 1031) 
s_lshl_b32 s22, 1, s18 // (op 1031) 
s_add_i32 s23, s22, s19 // (op 1031) 


s_and_b32 s19, s0, s23 // (op 1031) 


s_add_i32 s0, s1, s19 // (op 1031) 


s_ashr_i32 s1, s0, s18 // (op 1031) 


s_lshl_b32 s0, s13, 24 // (op 1031) 
s_ashr_i32 s18, s0, 31 // (op 1031) 

s_xor_b32 s0, s1, s18 // (op 1031) 

s_sub_i32 s1, s0, s18 // (op 1031) 


s_mul_hi_i32 s0, s1, s14 // (op 1031) 
s_add_i32 s18, s0, s1 // (op 1031) 

s_ashr_i32 s0, s18, 31 // (op 1031) 
s_cmp_eq_i32 s14, 0 // (op 1031) 
s_mov_b32 s19, scc // (op 1031) 
s_cmp_eq_i32 s19, 0 // (op 1031) 
s_cselect_b32 s22, 0, -1 // (op 1031) 

s_and_b32 s19, s15, 31 // (op 1031) 
s_lshl_b32 s23, 1, s19 // (op 1031) 
s_add_i32 s24, s23, s22 // (op 1031) 


s_and_b32 s25, s0, s24 // (op 1031) 


s_add_i32 s0, s18, s25 // (op 1031) 


s_ashr_i32 s18, s0, s19 // (op 1031) 


s_lshl_b32 s0, s15, 24 // (op 1031) 
s_ashr_i32 s19, s0, 31 // (op 1031) 

s_xor_b32 s0, s18, s19 // (op 1031) 

s_sub_i32 s18, s0, s19 // (op 1031) 


s_mul_i32 s0, s1, s11 // (op 1031) 

s_sub_u32 s1, s16, s0 // (op 1031) 

s_mul_i32 s0, s18, s11 // (op 1031) 
s_add_u32 s0, s0, s1 // (op 1031) 


s_mul_hi_i32 s1, s16, s9 // (op 1031) 
s_add_i32 s18, s1, s16 // (op 1031) 

s_ashr_i32 s1, s18, 31 // (op 1031) 
s_cmp_eq_i32 s9, 0 // (op 1031) 
s_mov_b32 s19, scc // (op 1031) 
s_cmp_eq_i32 s19, 0 // (op 1031) 
s_cselect_b32 s22, 0, -1 // (op 1031) 

s_and_b32 s19, s47, 31 // (op 1031) 
s_lshl_b32 s23, 1, s19 // (op 1031) 
s_add_i32 s24, s23, s22 // (op 1031) 


s_and_b32 s25, s1, s24 // (op 1031) 


s_add_i32 s1, s18, s25 // (op 1031) 


s_ashr_i32 s18, s1, s19 // (op 1031) 


s_lshl_b32 s1, s47, 24 // (op 1031) 
s_ashr_i32 s19, s1, 31 // (op 1031) 

s_xor_b32 s1, s18, s19 // (op 1031) 

s_sub_i32 s18, s1, s19 // (op 1031) 


s_add_u32 s50, s6, 32 // (op 1031) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1031) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1031) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1031) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1031) 

s_mul_i32 s1, s11, s24 // (op 1031) 

s_mul_i32 s19, s8, s1 // (op 1031) 

s_mul_i32 s1, s18, s19 // (op 1031) 

s_sub_u32 s19, s16, s1 // (op 1031) 

s_mov_b32 s1, s19 // (op 1031) 
s_mul_hi_i32 s22, s1, s12 // (op 1031) 
s_add_i32 s23, s22, s1 // (op 1031) 


s_ashr_i32 s1, s23, 31 // (op 1031) 
s_cmp_eq_i32 s12, 0 // (op 1031) 
s_mov_b32 s22, scc // (op 1031) 
s_cmp_eq_i32 s22, 0 // (op 1031) 
s_cselect_b32 s24, 0, -1 // (op 1031) 

s_and_b32 s22, s46, 31 // (op 1031) 
s_lshl_b32 s25, 1, s22 // (op 1031) 
s_add_i32 s49, s25, s24 // (op 1031) 


s_and_b32 s24, s1, s49 // (op 1031) 


s_add_i32 s1, s23, s24 // (op 1031) 


s_ashr_i32 s23, s1, s22 // (op 1031) 


s_lshl_b32 s1, s46, 24 // (op 1031) 
s_ashr_i32 s22, s1, 31 // (op 1031) 

s_xor_b32 s1, s23, s22 // (op 1031) 

s_sub_i32 s23, s1, s22 // (op 1031) 


s_mul_i32 s1, s48, s11 // (op 1031) 
s_add_u32 s24, s4, 64 // (op 1031) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1031) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1031) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1031) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1031) 

s_sub_i32 s22, s24, s1 // (op 1031) 

s_mul_i32 s24, s23, s22 // (op 1031) 


s_sub_u32 s25, s19, s24 // (op 1031) 


s_add_u32 s19, s25, s1 // (op 1031) 


s_cmp_eq_i32 s18, 0 // (op 1031) 
s_mov_b32 s1, scc // (op 1031) 

s_cmp_eq_i32 s1, 0 // (op 1031) 
s_cselect_b32 s18, s19, s0 // (op 1031) 



v_lshl_add_u32 v17, s18, 6, v16 // (op 1031) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1031) 
s_lshr_b32 s18, s0, 1 // (op 1031) 

v_lshl_add_u32 v16, s18, 3, v15 // (op 1031) 


v_mul_lo_u32 v15, v16, s36 // (op 1031) 

v_add_u32 v16, v17, v15 // (op 1031) 


s_mul_i32 s1, s36, s41 // (op 1033) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1033) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1033) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1033) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1033) most significant: sum
s_add_u32 s1, s1, s19 // (op 1033) most significant: sum


s_mov_b32 s68, s38 // (op 1033) reg expression
s_mov_b32 s69, s39 // (op 1033) reg expression
s_mov_b32 s70, s0 // (op 1033) reg expression
s_mov_b32 s71, 131072 // (op 1033) call()

s_mul_hi_i32 s0, s16, s10 // (op 1095) 
v_lshrrev_b32 v15, 6, v1 // (op 1095) 
s_add_i32 s1, s0, s16 // (op 1095) 

v_and_b32 v17, 1, v15 // (op 1095) 

s_ashr_i32 s0, s1, 31 // (op 1095) 
v_and_b32 v15, 15, v1 // (op 1095) 
s_cmp_eq_i32 s10, 0 // (op 1095) 
s_mov_b32 s18, scc // (op 1095) 
v_lshl_add_u32 v18, v17, 4, v15 // (op 1095) 


s_cmp_eq_i32 s18, 0 // (op 1095) 
s_cselect_b32 s19, 0, -1 // (op 1095) 

v_and_b32 v15, 63, v1 // (op 1095) 
s_and_b32 s18, s13, 31 // (op 1095) 
v_lshrrev_b32 v17, 4, v15 // (op 1095) 

s_lshl_b32 s22, 1, s18 // (op 1095) 
s_add_i32 s23, s22, s19 // (op 1095) 


s_and_b32 s19, s0, s23 // (op 1095) 


s_add_i32 s0, s1, s19 // (op 1095) 


s_ashr_i32 s1, s0, s18 // (op 1095) 


s_lshl_b32 s0, s13, 24 // (op 1095) 
s_ashr_i32 s18, s0, 31 // (op 1095) 

s_xor_b32 s0, s1, s18 // (op 1095) 

s_sub_i32 s1, s0, s18 // (op 1095) 


s_mul_hi_i32 s0, s1, s14 // (op 1095) 
s_add_i32 s18, s0, s1 // (op 1095) 

s_ashr_i32 s0, s18, 31 // (op 1095) 
s_cmp_eq_i32 s14, 0 // (op 1095) 
s_mov_b32 s19, scc // (op 1095) 
s_cmp_eq_i32 s19, 0 // (op 1095) 
s_cselect_b32 s22, 0, -1 // (op 1095) 

s_and_b32 s19, s15, 31 // (op 1095) 
s_lshl_b32 s23, 1, s19 // (op 1095) 
s_add_i32 s24, s23, s22 // (op 1095) 


s_and_b32 s25, s0, s24 // (op 1095) 


s_add_i32 s0, s18, s25 // (op 1095) 


s_ashr_i32 s18, s0, s19 // (op 1095) 


s_lshl_b32 s0, s15, 24 // (op 1095) 
s_ashr_i32 s19, s0, 31 // (op 1095) 

s_xor_b32 s0, s18, s19 // (op 1095) 

s_sub_i32 s18, s0, s19 // (op 1095) 


s_add_u32 s50, s6, 32 // (op 1095) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1095) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1095) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1095) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1095) 

s_mul_i32 s0, s18, s24 // (op 1095) 

s_sub_i32 s18, s1, s0 // (op 1095) 


s_mul_hi_i32 s19, s16, s9 // (op 1095) 
s_add_i32 s0, s19, s16 // (op 1095) 

s_ashr_i32 s1, s0, 31 // (op 1095) 
s_cmp_eq_i32 s9, 0 // (op 1095) 
s_mov_b32 s19, scc // (op 1095) 
s_cmp_eq_i32 s19, 0 // (op 1095) 
s_cselect_b32 s22, 0, -1 // (op 1095) 

s_and_b32 s19, s47, 31 // (op 1095) 
s_lshl_b32 s23, 1, s19 // (op 1095) 
s_add_i32 s49, s23, s22 // (op 1095) 


s_and_b32 s22, s1, s49 // (op 1095) 


s_add_i32 s1, s0, s22 // (op 1095) 


s_ashr_i32 s0, s1, s19 // (op 1095) 


s_lshl_b32 s1, s47, 24 // (op 1095) 
s_ashr_i32 s19, s1, 31 // (op 1095) 

s_xor_b32 s1, s0, s19 // (op 1095) 

s_sub_i32 s0, s1, s19 // (op 1095) 


s_mul_i32 s1, s11, s24 // (op 1095) 

s_mul_i32 s19, s8, s1 // (op 1095) 

s_mul_i32 s1, s0, s19 // (op 1095) 

s_sub_u32 s19, s16, s1 // (op 1095) 

s_mul_hi_i32 s1, s19, s12 // (op 1095) 
s_add_i32 s22, s1, s19 // (op 1095) 


s_ashr_i32 s1, s22, 31 // (op 1095) 
s_cmp_eq_i32 s12, 0 // (op 1095) 
s_mov_b32 s19, scc // (op 1095) 
s_cmp_eq_i32 s19, 0 // (op 1095) 
s_cselect_b32 s23, 0, -1 // (op 1095) 

s_and_b32 s19, s46, 31 // (op 1095) 
s_lshl_b32 s24, 1, s19 // (op 1095) 
s_add_i32 s25, s24, s23 // (op 1095) 


s_and_b32 s23, s1, s25 // (op 1095) 


s_add_i32 s1, s22, s23 // (op 1095) 


s_ashr_i32 s22, s1, s19 // (op 1095) 


s_lshl_b32 s1, s46, 24 // (op 1095) 
s_ashr_i32 s19, s1, 31 // (op 1095) 

s_xor_b32 s1, s22, s19 // (op 1095) 

s_sub_i32 s22, s1, s19 // (op 1095) 


s_cmp_eq_i32 s0, 0 // (op 1095) 
s_mov_b32 s1, scc // (op 1095) 

s_cmp_eq_i32 s1, 0 // (op 1095) 
s_cselect_b32 s0, s22, s18 // (op 1095) 



v_lshl_add_u32 v15, s0, 5, v18 // (op 1095) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1095) 
s_lshr_b32 s18, s0, 1 // (op 1095) 

v_lshl_add_u32 v18, s18, 3, v17 // (op 1095) 


v_mul_lo_u32 v17, v18, s42 // (op 1095) 
v_add_u32 v17, v17, v15 // (op 1095) 


s_mul_i32 s1, s42, s41 // (op 1097) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1097) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1097) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1097) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1097) most significant: sum
s_add_u32 s1, s1, s19 // (op 1097) most significant: sum


s_mov_b32 s72, s44 // (op 1097) reg expression
s_mov_b32 s73, s45 // (op 1097) reg expression
s_mov_b32 s74, s0 // (op 1097) reg expression
s_mov_b32 s75, 131072 // (op 1097) call()

v_and_b32 v15, -4, v1 // (op 1188) 
v_sub_u32 v18, v1, v15 // (op 1188) 

v_lshlrev_b32 v15, 4, v18 // (op 1188) 

v_lshrrev_b32 v18, 2, v1 // (op 1188) 
v_lshl_add_u32 v19, v18, 6, v15 // (op 1188) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 1188) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -2, v1 // (op 1245) 
v_sub_u32 v18, v1, v15 // (op 1245) 

v_lshlrev_b32 v15, 4, v18 // (op 1245) 

v_lshrrev_b32 v18, 1, v1 // (op 1245) 
v_lshl_add_u32 v19, v18, 5, v15 // (op 1245) 


v_mul_lo_u32 v15, v19, 6 // (op 1245) 
v_lshrrev_b32 v18, 3, v15 // (op 1245) 

v_lshrrev_b32 v15, 4, v19 // (op 1245) 

v_lshlrev_b32 v19, 2, v15 // (op 1245) 

v_add_u32 v15, v18, v19 // (op 1245) 


v_and_b32 v18, -64, v1 // (op 1298) 
v_sub_u32 v19, v1, v18 // (op 1298) 

v_and_b32 v18, 4294967232, v1 // (op 1298) 
v_add_u32 v20, v18, v19 // (op 1298) 


s_ashr_i64 s[18:19], s[30:31], 7 // (op 316) 
s_and_b32 s1, s18, 4294967294 // (op 316) 

s_ashr_i64 s[18:19], s[30:31], 7 // (op 315) 
s_mov_b32 s22, s18 // (op 315) convert

s_cmp_lt_u32 s1, s22 // (op 315) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_315:
s_barrier  // (op 1493) 
s_add_u32 s1, s1, 1 // (op 317) 

s_lshl_b64 s[18:19], s[26:27], 6 // (op 870) 
v_mov_b64 v[18:19], s[18:19] // (op 870) convert


v_mov_b64 v[22:23], v[12:13] // (op 870) 
v_mov_b64 v[24:25], v[22:23] // (op 870) reg expression
s_mov_b32 m0, s0 // (op 870) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 870) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v18 // (op 870) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 870) most significant half

s_add_u32 m0, m0, 4096 // (op 870) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 870) Load value direct to lds



s_lshl_b64 s[18:19], s[26:27], 7 // (op 889) 
v_mov_b32 v18, s18 // (op 889) 
v_mov_b32 v19, s19 // (op 889) 
v_add_co_u32 v12, s[50:51], v12, v18 // (op 889) least significant half
v_addc_co_u32 v13, s[50:51], v13, v19, s[50:51] // (op 889) most significant half






v_mov_b32 v18, v14 // (op 179) 
v_mov_b32 v19, v18 // (op 179) reg expression

buffer_load_dwordx3 v[22:24], v19, s[64:67], 0 offen // (op 179) Load value


s_lshl_b64 s[18:19], s[32:33], 7 // (op 959) 
s_mul_i32 s51, s19, 6 // (op 959) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 959) least significant
s_mul_hi_u32 s49, s18, 6 // (op 959) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 959) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 959) 

v_add_u32 v14, v14, s18 // (op 959) 



s_barrier  // (op 1506) 
v_mov_b32 v21, v16 // (op 180) 



v_mov_b32 v18, v21 // (op 180) reg expression

s_waitcnt vmcnt(0)
ds_write_b96 v15, v[22:24] offset:16384 // (op 177) Store local data 
buffer_load_ubyte v19, v18, s[68:71], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s18, s36, 2 // (op 1023) 
v_add_u32 v16, v16, s18 // (op 1023) 

s_barrier  // (op 1496) 
s_barrier  // (op 1520) 






ds_read_b64_tr_b8 v[22:23], v4 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v4 offset:512 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v4 offset:4096 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v4 offset:4608 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v19 offset:20480 // (op 178) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1510) 



ds_read_b96_tr_b6 v[30:32], v0 offset:16384 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v0 offset:16896 // (op 171) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1524) 


ds_read_u8 v21, v7 offset:20480 // (op 173) Load local data 

v_mov_b64 v[18:19], s[42:43] // (op 175) convert

v_mov_b32 v36, v17 // (op 175) 
v_mov_b32 v37, v36 // (op 175) reg expression
v_mov_b32 v38, v37 // (op 175) 
buffer_load_ubyte v39, v38, s[72:75], 0 offen offset: 0 // (op 175) Load value




s_lshl_b32 s18, s42, 2 // (op 1087) 
v_add_u32 v17, v17, s18 // (op 1087) 




s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:35], a[0:3], v21, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 176) 
ds_read_b64_tr_b8 v[40:41], v4 offset:32 // (op 321) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v4 offset:544 // (op 321) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:4128 // (op 321) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:4640 // (op 321) Transpose load local data 


ds_read_u8 v18, v7 offset:20512 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:47], v[30:35], a[4:7], v18, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 327) 




s_ashr_i64 s[18:19], s[30:31], 7 // (op 315) 
s_mov_b32 s22, s18 // (op 315) convert

s_cmp_lt_u32 s1, s22 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_315 if true)
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_315:






















s_barrier  // (op 1596) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen // (op 52) Store value

v_accvgpr_read v10, a4 // (op 513) 
v_accvgpr_read v11, a5 // (op 513) 
v_accvgpr_read v12, a6 // (op 513) 
v_accvgpr_read v13, a7 // (op 513) 
v_mul_f32 v14, v8, v10 // (op 513) 
v_mul_f32 v15, v8, v11 // (op 513) 
v_mul_f32 v16, v8, v12 // (op 513) 
v_mul_f32 v17, v8, v13 // (op 513) 




buffer_store_dwordx4 v[14:17], v9, s[60:63], 0 offen offset:128 // (op 515) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:128 // (op 543) Load value
v_accvgpr_read v14, a4 // (op 540) 
v_accvgpr_read v15, a5 // (op 540) 
v_accvgpr_read v16, a6 // (op 540) 
v_accvgpr_read v17, a7 // (op 540) 
v_mul_f32 v18, v8, v14 // (op 540) 
v_mul_f32 v19, v8, v15 // (op 540) 
v_mul_f32 v20, v8, v16 // (op 540) 
v_mul_f32 v21, v8, v17 // (op 540) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 544) 
v_mul_f32 v15, v5, v11 // (op 544) 
v_mul_f32 v16, v5, v12 // (op 544) 
v_mul_f32 v17, v5, v13 // (op 544) 

v_add_f32 v10, v14, v18 // (op 541) 
v_add_f32 v11, v15, v19 // (op 541) 
v_add_f32 v12, v16, v20 // (op 541) 
v_add_f32 v13, v17, v21 // (op 541) 





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen offset:128 // (op 542) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen // (op 135) Load value
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





buffer_store_dwordx4 v[10:13], v9, s[60:63], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.LRR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end:
.size RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_, .LRR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end-RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 68
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
  .amdhsa_accum_offset 60
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int64, semantic: StrideD0 },
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
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [64, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 60
    .agpr_count: 8
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_2_size_1_11
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
      - .name: WGM_18
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
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
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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

