// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_,@function
RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1136) 
s_mul_hi_i32 s1, s16, s10 // (op 1325) 
v_and_b32 v0, 63, v1 // (op 1251) 
v_and_b32 v3, 48, v1 // (op 1251) 
v_sub_u32 v4, v0, v3 // (op 1251) 

v_lshrrev_b32 v3, 4, v0 // (op 1251) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1251) 


v_lshrrev_b32 v3, 6, v1 // (op 1251) 
v_and_b32 v4, 1, v3 // (op 1251) 

v_lshlrev_b32 v3, 4, v4 // (op 1251) 

v_lshl_add_u32 v4, v0, 5, v3 // (op 1251) 


v_mul_lo_u32 v0, v4, 6 // (op 1251) 
v_lshrrev_b32 v3, 3, v0 // (op 1251) 

v_lshrrev_b32 v0, 4, v4 // (op 1251) 

v_lshlrev_b32 v4, 2, v0 // (op 1251) 

v_add_u32 v0, v3, v4 // (op 1251) 


v_and_b32 v3, 63, v1 // (op 1380) 
v_and_b32 v4, 48, v1 // (op 1380) 
v_sub_u32 v5, v3, v4 // (op 1380) 

v_lshrrev_b32 v4, 4, v3 // (op 1380) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 1380) 


v_lshrrev_b32 v4, 6, v1 // (op 1380) 
v_and_b32 v5, 1, v4 // (op 1380) 

v_lshlrev_b32 v4, 4, v5 // (op 1380) 

v_lshl_add_u32 v5, v3, 5, v4 // (op 1380) 


v_mul_lo_u32 v3, v5, 6 // (op 1380) 
v_lshrrev_b32 v4, 3, v3 // (op 1380) 

v_lshrrev_b32 v3, 4, v5 // (op 1380) 

v_lshlrev_b32 v5, 2, v3 // (op 1380) 

v_add_u32 v3, v4, v5 // (op 1380) 


v_lshrrev_b32 v4, 7, v1 // (op 1136) 
s_add_i32 s18, s0, s16 // (op 1136) 

v_and_b32 v5, 63, v1 // (op 1136) 
s_ashr_i32 s0, s18, 31 // (op 1136) 
v_lshrrev_b32 v6, 4, v5 // (op 1136) 
s_cmp_eq_i32 s10, 0 // (op 1136) 
s_mov_b32 s19, scc // (op 1136) 
v_lshlrev_b32 v7, 2, v6 // (op 1136) 

s_cmp_eq_i32 s19, 0 // (op 1136) 
s_cselect_b32 s51, 0, -1 // (op 1136) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1136) 


s_and_b32 s19, s13, 31 // (op 1136) 
v_lshrrev_b32 v4, 2, v5 // (op 1136) 

v_and_b32 v5, 3, v4 // (op 1136) 

v_and_b32 v4, 63, v1 // (op 1198) 
v_and_b32 v7, 48, v1 // (op 1198) 
v_sub_u32 v8, v4, v7 // (op 1198) 

v_lshrrev_b32 v7, 4, v4 // (op 1198) 

v_lshl_add_u32 v4, v7, 5, v8 // (op 1198) 


v_lshrrev_b32 v7, 7, v1 // (op 1198) 
v_lshlrev_b32 v8, 4, v7 // (op 1198) 

v_lshl_add_u32 v7, v4, 6, v8 // (op 1198) 


v_lshlrev_b32 v4, 2, v7 // (op 1198) 

v_lshrrev_b32 v7, 3, v4 // (op 1198) 

v_lshrrev_b32 v4, 7, v1 // (op 1309) 
v_and_b32 v8, 15, v1 // (op 1309) 
v_lshl_add_u32 v9, v4, 4, v8 // (op 1309) 


v_and_b32 v4, 63, v1 // (op 1309) 
v_lshrrev_b32 v8, 4, v4 // (op 1309) 

v_lshl_add_u32 v4, v8, 6, v9 // (op 1309) 


v_lshrrev_b32 v8, 7, v1 // (op 1325) 
v_and_b32 v9, 63, v1 // (op 1325) 
v_and_b32 v10, 63, v1 // (op 1356) 
v_and_b32 v11, 48, v1 // (op 1356) 
v_sub_u32 v12, v10, v11 // (op 1356) 

v_lshrrev_b32 v11, 4, v10 // (op 1356) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 1356) 


v_lshrrev_b32 v11, 7, v1 // (op 1356) 
v_lshlrev_b32 v12, 4, v11 // (op 1356) 

v_lshl_add_u32 v11, v10, 6, v12 // (op 1356) 


v_lshlrev_b32 v10, 2, v11 // (op 1356) 

v_lshrrev_b32 v11, 3, v10 // (op 1356) 

v_lshrrev_b32 v10, 7, v1 // (op 1409) 
v_and_b32 v12, 15, v1 // (op 1409) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 1409) 


v_and_b32 v10, 63, v1 // (op 1409) 
v_lshrrev_b32 v12, 4, v10 // (op 1409) 

v_lshl_add_u32 v10, v12, 6, v13 // (op 1409) 


s_lshl_b32 s52, 1, s19 // (op 1136) 
v_and_b32 v12, 3, v1 // (op 1136) 
s_add_i32 s53, s52, s51 // (op 1136) 


s_and_b32 s51, s0, s53 // (op 1136) 


s_add_i32 s0, s1, s16 // (op 1325) 

s_ashr_i32 s1, s0, 31 // (op 1325) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1136) 


s_add_i32 s52, s18, s51 // (op 1136) 


v_lshrrev_b32 v5, 6, v1 // (op 1136) 
s_ashr_i32 s18, s52, s19 // (op 1136) 


v_and_b32 v12, 1, v5 // (op 1136) 

s_lshl_b32 s19, s13, 24 // (op 1136) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1136) 


s_ashr_i32 s51, s19, 31 // (op 1136) 

s_xor_b32 s19, s18, s51 // (op 1136) 

s_sub_i32 s18, s19, s51 // (op 1136) 


s_mul_hi_i32 s19, s18, s14 // (op 1136) 
s_add_i32 s51, s19, s18 // (op 1136) 

s_ashr_i32 s19, s51, 31 // (op 1136) 
s_cmp_eq_i32 s14, 0 // (op 1136) 
s_mov_b32 s52, scc // (op 1136) 
s_cmp_eq_i32 s52, 0 // (op 1136) 
s_cselect_b32 s53, 0, -1 // (op 1136) 

s_and_b32 s52, s15, 31 // (op 1136) 
v_lshrrev_b32 v12, 4, v9 // (op 1325) 
s_cmp_eq_i32 s10, 0 // (op 1325) 
s_mov_b32 s54, scc // (op 1325) 
v_lshlrev_b32 v13, 2, v12 // (op 1325) 

s_cmp_eq_i32 s54, 0 // (op 1325) 
s_cselect_b32 s55, 0, -1 // (op 1325) 

s_lshl_b32 s54, 1, s52 // (op 1136) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1325) 


v_lshrrev_b32 v8, 2, v9 // (op 1325) 

s_add_i32 s56, s54, s53 // (op 1136) 


s_and_b32 s53, s19, s56 // (op 1136) 


s_add_i32 s19, s51, s53 // (op 1136) 


s_ashr_i32 s51, s19, s52 // (op 1136) 


s_lshl_b32 s19, s15, 24 // (op 1136) 
s_ashr_i32 s52, s19, 31 // (op 1136) 

s_xor_b32 s19, s51, s52 // (op 1136) 

s_sub_i32 s51, s19, s52 // (op 1136) 


s_mul_i32 s19, s18, s11 // (op 1136) 
s_sub_u32 s52, s16, s19 // (op 1136) 

s_mul_i32 s19, s51, s11 // (op 1136) 
s_add_u32 s19, s19, s52 // (op 1136) 

s_mul_hi_i32 s52, s16, s9 // (op 1136) 
s_add_i32 s53, s52, s16 // (op 1136) 

s_ashr_i32 s52, s53, 31 // (op 1136) 
s_cmp_eq_i32 s9, 0 // (op 1136) 
s_mov_b32 s54, scc // (op 1136) 
s_cmp_eq_i32 s54, 0 // (op 1136) 
s_cselect_b32 s56, 0, -1 // (op 1136) 

s_and_b32 s54, s13, 31 // (op 1325) 
v_and_b32 v9, 3, v8 // (op 1325) 

s_lshl_b32 s57, 1, s54 // (op 1325) 
v_and_b32 v8, 3, v1 // (op 1325) 
s_add_i32 s58, s57, s55 // (op 1325) 


s_and_b32 s55, s1, s58 // (op 1325) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1325) 


s_add_i32 s1, s0, s55 // (op 1325) 


v_lshrrev_b32 v8, 6, v1 // (op 1325) 
s_ashr_i32 s0, s1, s54 // (op 1325) 


v_and_b32 v9, 1, v8 // (op 1325) 

s_lshl_b32 s1, s13, 24 // (op 1325) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1325) 


s_ashr_i32 s54, s1, 31 // (op 1325) 

s_xor_b32 s1, s0, s54 // (op 1325) 

s_sub_i32 s0, s1, s54 // (op 1325) 


s_mul_hi_i32 s1, s0, s14 // (op 1325) 
s_add_i32 s54, s1, s0 // (op 1325) 

s_ashr_i32 s1, s54, 31 // (op 1325) 
s_cmp_eq_i32 s14, 0 // (op 1325) 
s_mov_b32 s55, scc // (op 1325) 
s_cmp_eq_i32 s55, 0 // (op 1325) 
s_cselect_b32 s57, 0, -1 // (op 1325) 

s_and_b32 s55, s15, 31 // (op 1325) 
s_lshl_b32 s58, 1, s55 // (op 1325) 
s_add_i32 s59, s58, s57 // (op 1325) 


s_and_b32 s57, s1, s59 // (op 1325) 


s_add_i32 s1, s54, s57 // (op 1325) 


s_ashr_i32 s54, s1, s55 // (op 1325) 


s_lshl_b32 s1, s15, 24 // (op 1325) 
s_ashr_i32 s55, s1, 31 // (op 1325) 

s_xor_b32 s1, s54, s55 // (op 1325) 

s_sub_i32 s54, s1, s55 // (op 1325) 


s_mul_i32 s1, s0, s11 // (op 1325) 
s_sub_u32 s55, s16, s1 // (op 1325) 

s_mul_i32 s1, s54, s11 // (op 1325) 
s_add_u32 s1, s1, s55 // (op 1325) 

s_mul_hi_i32 s55, s16, s9 // (op 1325) 
s_add_i32 s57, s55, s16 // (op 1325) 

s_ashr_i32 s55, s57, 31 // (op 1325) 
s_cmp_eq_i32 s9, 0 // (op 1325) 
s_mov_b32 s58, scc // (op 1325) 
s_cmp_eq_i32 s58, 0 // (op 1325) 
s_cselect_b32 s59, 0, -1 // (op 1325) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s47, 31 // (op 1136) 
s_lshl_b32 s60, 1, s58 // (op 1136) 
s_add_i32 s61, s60, s56 // (op 1136) 


s_and_b32 s56, s52, s61 // (op 1136) 


s_add_i32 s52, s53, s56 // (op 1136) 


s_ashr_i32 s53, s52, s58 // (op 1136) 


s_lshl_b32 s52, s47, 24 // (op 1136) 
s_ashr_i32 s56, s52, 31 // (op 1136) 

s_xor_b32 s52, s53, s56 // (op 1136) 

s_sub_i32 s53, s52, s56 // (op 1136) 


s_and_b32 s52, s47, 31 // (op 1325) 
s_lshl_b32 s56, 1, s52 // (op 1325) 
s_add_i32 s58, s56, s59 // (op 1325) 


s_and_b32 s56, s55, s58 // (op 1325) 


s_add_i32 s55, s57, s56 // (op 1325) 


s_ashr_i32 s56, s55, s52 // (op 1325) 


s_lshl_b32 s52, s47, 24 // (op 1325) 
s_ashr_i32 s55, s52, 31 // (op 1325) 

s_xor_b32 s52, s56, s55 // (op 1325) 

s_sub_i32 s56, s52, s55 // (op 1325) 


s_add_u32 s58, s6, 32 // (op 1136) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1136) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1136) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1136) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 5 // (op 1136) 

s_mul_i32 s52, s11, s58 // (op 1136) 
s_mul_i32 s55, s8, s52 // (op 1136) 

s_mul_i32 s52, s53, s55 // (op 1136) 

s_sub_u32 s55, s16, s52 // (op 1136) 

s_mov_b32 s52, s55 // (op 1136) 
s_mul_hi_i32 s57, s52, s12 // (op 1136) 
s_add_i32 s60, s57, s52 // (op 1136) 


s_ashr_i32 s52, s60, 31 // (op 1136) 
s_cmp_eq_i32 s12, 0 // (op 1136) 
s_mov_b32 s57, scc // (op 1136) 
s_cmp_eq_i32 s57, 0 // (op 1136) 
s_cselect_b32 s61, 0, -1 // (op 1136) 

s_and_b32 s57, s46, 31 // (op 1136) 
s_lshl_b32 s62, 1, s57 // (op 1136) 
s_add_i32 s63, s62, s61 // (op 1136) 


s_and_b32 s61, s52, s63 // (op 1136) 


s_add_i32 s52, s60, s61 // (op 1136) 


s_ashr_i32 s60, s52, s57 // (op 1136) 


s_lshl_b32 s52, s46, 24 // (op 1136) 
s_ashr_i32 s57, s52, 31 // (op 1136) 

s_xor_b32 s52, s60, s57 // (op 1136) 

s_sub_i32 s60, s52, s57 // (op 1136) 


s_mul_i32 s52, s48, s11 // (op 1136) 
s_add_u32 s62, s4, 64 // (op 1136) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 1136) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1136) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1136) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 1136) 

s_sub_i32 s57, s62, s52 // (op 1136) 

s_mul_i32 s61, s60, s57 // (op 1136) 

s_sub_u32 s57, s55, s61 // (op 1136) 


s_add_u32 s55, s57, s52 // (op 1136) 


s_cmp_eq_i32 s53, 0 // (op 1136) 
s_mov_b32 s52, scc // (op 1136) 

s_cmp_eq_i32 s52, 0 // (op 1136) 
s_cselect_b32 s53, s55, s19 // (op 1136) 


s_mul_i32 s19, s51, s58 // (op 1136) 


v_lshl_add_u32 v9, s53, 6, v6 // (op 1136) 


s_sub_i32 s51, s18, s19 // (op 1136) 


s_cmp_eq_i32 s52, 0 // (op 1136) 
s_cselect_b32 s53, s60, s51 // (op 1136) 



v_lshl_add_u32 v6, s53, 5, v5 // (op 1136) 


v_mul_lo_u32 v5, v6, s2 // (op 1136) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 1136) 


s_add_u32 s18, s6, 32 // (op 1325) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1325) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 1325) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 1325) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 1325) 

s_mul_i32 s55, s11, s18 // (op 1325) 
s_mul_i32 s51, s8, s55 // (op 1325) 

s_mul_i32 s55, s56, s51 // (op 1325) 

s_sub_u32 s51, s16, s55 // (op 1325) 

s_mov_b32 s55, s51 // (op 1325) 
s_mul_hi_i32 s52, s55, s12 // (op 1325) 
s_add_i32 s53, s52, s55 // (op 1325) 


s_ashr_i32 s52, s53, 31 // (op 1325) 
s_cmp_eq_i32 s12, 0 // (op 1325) 
s_mov_b32 s55, scc // (op 1325) 
s_cmp_eq_i32 s55, 0 // (op 1325) 
s_cselect_b32 s57, 0, -1 // (op 1325) 

s_and_b32 s55, s46, 31 // (op 1325) 
s_lshl_b32 s58, 1, s55 // (op 1325) 
s_add_i32 s59, s58, s57 // (op 1325) 


s_and_b32 s57, s52, s59 // (op 1325) 


s_add_i32 s52, s53, s57 // (op 1325) 


s_ashr_i32 s53, s52, s55 // (op 1325) 


s_lshl_b32 s52, s46, 24 // (op 1325) 
s_ashr_i32 s55, s52, 31 // (op 1325) 

s_xor_b32 s52, s53, s55 // (op 1325) 

s_sub_i32 s53, s52, s55 // (op 1325) 


s_mul_i32 s52, s48, s11 // (op 1325) 
s_add_u32 s58, s4, 64 // (op 1325) Start of Int64 add, locking SCC
s_addc_u32 s59, s5, 0 // (op 1325) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1325) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1325) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 1325) 

s_sub_i32 s55, s58, s52 // (op 1325) 

s_mul_i32 s57, s53, s55 // (op 1325) 

s_sub_u32 s55, s51, s57 // (op 1325) 


s_add_u32 s51, s55, s52 // (op 1325) 


s_cmp_eq_i32 s56, 0 // (op 1325) 
s_mov_b32 s52, scc // (op 1325) 

s_cmp_eq_i32 s52, 0 // (op 1325) 
s_cselect_b32 s55, s51, s1 // (op 1325) 


s_mul_i32 s1, s54, s18 // (op 1325) 


v_lshl_add_u32 v5, s55, 6, v12 // (op 1325) 


s_sub_i32 s51, s0, s1 // (op 1325) 


s_cmp_eq_i32 s52, 0 // (op 1325) 
s_cselect_b32 s0, s53, s51 // (op 1325) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 1325) 


v_mul_lo_u32 v8, v9, s20 // (op 1325) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 1325) 


s_mul_i32 s1, s2, s7 // (op 1138) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1138) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1138) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1138) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1138) most significant: sum
s_add_u32 s1, s1, s19 // (op 1138) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1138) 

s_mul_i32 s1, s20, s7 // (op 1327) most significant: low of low * high
s_mul_i32 s51, s21, s6 // (op 1327) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 1327) least significant
s_mul_hi_u32 s52, s20, s6 // (op 1327) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1327) most significant: sum
s_add_u32 s1, s1, s52 // (op 1327) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1327) 

s_mov_b32 s56, s22 // (op 1138) reg expression
s_mov_b32 s57, s23 // (op 1138) reg expression
s_mov_b32 s58, s18 // (op 1138) reg expression
s_mov_b32 s59, 131072 // (op 1138) call()

s_mov_b32 s60, s24 // (op 1327) reg expression

s_mov_b32 s61, s25 // (op 1327) reg expression
s_mov_b32 s62, s52 // (op 1327) reg expression
s_mov_b32 s63, 131072 // (op 1327) call()


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
s_mul_hi_i32 s0, s16, s10 // (op 903) 
v_and_b32 v12, -2, v1 // (op 903) 
s_add_i32 s1, s0, s16 // (op 903) 

v_sub_u32 v13, v1, v12 // (op 903) 

s_ashr_i32 s0, s1, 31 // (op 903) 
v_lshlrev_b32 v12, 5, v13 // (op 903) 

s_cmp_eq_i32 s10, 0 // (op 903) 
s_mov_b32 s18, scc // (op 903) 
v_lshrrev_b32 v13, 1, v1 // (op 903) 
s_cmp_eq_i32 s18, 0 // (op 903) 
s_cselect_b32 s19, 0, -1 // (op 903) 

s_and_b32 s18, s13, 31 // (op 903) 
s_lshl_b32 s22, 1, s18 // (op 903) 
s_add_i32 s23, s22, s19 // (op 903) 


s_and_b32 s19, s0, s23 // (op 903) 


s_add_i32 s0, s1, s19 // (op 903) 


s_ashr_i32 s1, s0, s18 // (op 903) 


s_lshl_b32 s0, s13, 24 // (op 903) 
s_ashr_i32 s18, s0, 31 // (op 903) 

s_xor_b32 s0, s1, s18 // (op 903) 

s_sub_i32 s1, s0, s18 // (op 903) 


s_mul_hi_i32 s0, s1, s14 // (op 903) 
s_add_i32 s18, s0, s1 // (op 903) 

s_ashr_i32 s0, s18, 31 // (op 903) 
s_cmp_eq_i32 s14, 0 // (op 903) 
s_mov_b32 s19, scc // (op 903) 
s_cmp_eq_i32 s19, 0 // (op 903) 
s_cselect_b32 s22, 0, -1 // (op 903) 

s_and_b32 s19, s15, 31 // (op 903) 
s_lshl_b32 s23, 1, s19 // (op 903) 
s_add_i32 s24, s23, s22 // (op 903) 


s_and_b32 s25, s0, s24 // (op 903) 


s_add_i32 s0, s18, s25 // (op 903) 


s_ashr_i32 s18, s0, s19 // (op 903) 


s_lshl_b32 s0, s15, 24 // (op 903) 
s_ashr_i32 s19, s0, 31 // (op 903) 

s_xor_b32 s0, s18, s19 // (op 903) 

s_sub_i32 s18, s0, s19 // (op 903) 


s_mul_i32 s0, s1, s11 // (op 903) 

s_sub_u32 s1, s16, s0 // (op 903) 

s_mul_i32 s0, s18, s11 // (op 903) 
s_add_u32 s0, s0, s1 // (op 903) 


s_mul_hi_i32 s1, s16, s9 // (op 903) 
s_add_i32 s18, s1, s16 // (op 903) 

s_ashr_i32 s1, s18, 31 // (op 903) 
s_cmp_eq_i32 s9, 0 // (op 903) 
s_mov_b32 s19, scc // (op 903) 
s_cmp_eq_i32 s19, 0 // (op 903) 
s_cselect_b32 s22, 0, -1 // (op 903) 

s_and_b32 s19, s47, 31 // (op 903) 
s_lshl_b32 s23, 1, s19 // (op 903) 
s_add_i32 s24, s23, s22 // (op 903) 


s_and_b32 s25, s1, s24 // (op 903) 


s_add_i32 s1, s18, s25 // (op 903) 


s_ashr_i32 s18, s1, s19 // (op 903) 


s_lshl_b32 s1, s47, 24 // (op 903) 
s_ashr_i32 s19, s1, 31 // (op 903) 

s_xor_b32 s1, s18, s19 // (op 903) 

s_sub_i32 s18, s1, s19 // (op 903) 


s_add_u32 s22, s6, 32 // (op 903) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 903) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 903) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 903) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 903) 

s_mul_i32 s1, s11, s22 // (op 903) 

s_mul_i32 s19, s8, s1 // (op 903) 

s_mul_i32 s1, s18, s19 // (op 903) 

s_sub_u32 s19, s16, s1 // (op 903) 

s_mov_b32 s1, s19 // (op 903) 
s_mul_hi_i32 s22, s1, s12 // (op 903) 
s_add_i32 s23, s22, s1 // (op 903) 


s_ashr_i32 s1, s23, 31 // (op 903) 
s_cmp_eq_i32 s12, 0 // (op 903) 
s_mov_b32 s22, scc // (op 903) 
s_cmp_eq_i32 s22, 0 // (op 903) 
s_cselect_b32 s24, 0, -1 // (op 903) 

s_and_b32 s22, s46, 31 // (op 903) 
s_lshl_b32 s25, 1, s22 // (op 903) 
s_add_i32 s49, s25, s24 // (op 903) 


s_and_b32 s24, s1, s49 // (op 903) 


s_add_i32 s1, s23, s24 // (op 903) 


s_ashr_i32 s23, s1, s22 // (op 903) 


s_lshl_b32 s1, s46, 24 // (op 903) 
s_ashr_i32 s22, s1, 31 // (op 903) 

s_xor_b32 s1, s23, s22 // (op 903) 

s_sub_i32 s23, s1, s22 // (op 903) 


s_mul_i32 s1, s48, s11 // (op 903) 
s_add_u32 s24, s4, 64 // (op 903) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 903) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 903) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 903) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 903) 

s_sub_i32 s22, s24, s1 // (op 903) 

s_mul_i32 s24, s23, s22 // (op 903) 


s_sub_u32 s25, s19, s24 // (op 903) 


s_add_u32 s19, s25, s1 // (op 903) 


s_cmp_eq_i32 s18, 0 // (op 903) 
s_mov_b32 s1, scc // (op 903) 

s_cmp_eq_i32 s1, 0 // (op 903) 
s_cselect_b32 s18, s19, s0 // (op 903) 



v_lshl_add_u32 v14, s18, 6, v12 // (op 903) 


v_mov_b32 v17, 0 // (op 903) convert
v_mov_b32 v16, v14 // (op 903) convert

v_mul_lo_u32 v15, v13, s27 // (op 903) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 903) most significant: high of low * low
v_mul_lo_u32 v14, v13, s26 // (op 903) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 903) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 903) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 903) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 903) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 903) 

v_mov_b32 v12, v14 // (op 903) convert

s_mul_i32 s1, s26, s31 // (op 905) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 905) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 905) least significant
s_mul_hi_u32 s19, s26, s30 // (op 905) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 905) most significant: sum
s_add_u32 s1, s1, s19 // (op 905) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 905) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 905) 

s_mov_b32 s52, s28 // (op 905) reg expression
s_mov_b32 s53, s29 // (op 905) reg expression
s_mov_b32 s54, s0 // (op 905) reg expression
s_mov_b32 s55, 131072 // (op 905) call()

s_mul_hi_i32 s0, s16, s10 // (op 922) 
v_and_b32 v13, -2, v1 // (op 922) 
s_add_i32 s1, s0, s16 // (op 922) 

v_sub_u32 v14, v1, v13 // (op 922) 

s_ashr_i32 s0, s1, 31 // (op 922) 
v_lshlrev_b32 v13, 5, v14 // (op 922) 

s_cmp_eq_i32 s10, 0 // (op 922) 
s_mov_b32 s18, scc // (op 922) 
v_lshrrev_b32 v14, 1, v1 // (op 922) 
s_cmp_eq_i32 s18, 0 // (op 922) 
s_cselect_b32 s19, 0, -1 // (op 922) 

s_and_b32 s18, s13, 31 // (op 922) 
s_lshl_b32 s22, 1, s18 // (op 922) 
s_add_i32 s23, s22, s19 // (op 922) 


s_and_b32 s19, s0, s23 // (op 922) 


s_add_i32 s0, s1, s19 // (op 922) 


s_ashr_i32 s1, s0, s18 // (op 922) 


s_lshl_b32 s0, s13, 24 // (op 922) 
s_ashr_i32 s18, s0, 31 // (op 922) 

s_xor_b32 s0, s1, s18 // (op 922) 

s_sub_i32 s1, s0, s18 // (op 922) 


s_mul_hi_i32 s0, s1, s14 // (op 922) 
s_add_i32 s18, s0, s1 // (op 922) 

s_ashr_i32 s0, s18, 31 // (op 922) 
s_cmp_eq_i32 s14, 0 // (op 922) 
s_mov_b32 s19, scc // (op 922) 
s_cmp_eq_i32 s19, 0 // (op 922) 
s_cselect_b32 s22, 0, -1 // (op 922) 

s_and_b32 s19, s15, 31 // (op 922) 
s_lshl_b32 s23, 1, s19 // (op 922) 
s_add_i32 s24, s23, s22 // (op 922) 


s_and_b32 s25, s0, s24 // (op 922) 


s_add_i32 s0, s18, s25 // (op 922) 


s_ashr_i32 s18, s0, s19 // (op 922) 


s_lshl_b32 s0, s15, 24 // (op 922) 
s_ashr_i32 s19, s0, 31 // (op 922) 

s_xor_b32 s0, s18, s19 // (op 922) 

s_sub_i32 s18, s0, s19 // (op 922) 


s_mul_i32 s0, s1, s11 // (op 922) 

s_sub_u32 s1, s16, s0 // (op 922) 

s_mul_i32 s0, s18, s11 // (op 922) 
s_add_u32 s0, s0, s1 // (op 922) 


s_mul_hi_i32 s1, s16, s9 // (op 922) 
s_add_i32 s18, s1, s16 // (op 922) 

s_ashr_i32 s1, s18, 31 // (op 922) 
s_cmp_eq_i32 s9, 0 // (op 922) 
s_mov_b32 s19, scc // (op 922) 
s_cmp_eq_i32 s19, 0 // (op 922) 
s_cselect_b32 s22, 0, -1 // (op 922) 

s_and_b32 s19, s47, 31 // (op 922) 
s_lshl_b32 s23, 1, s19 // (op 922) 
s_add_i32 s24, s23, s22 // (op 922) 


s_and_b32 s25, s1, s24 // (op 922) 


s_add_i32 s1, s18, s25 // (op 922) 


s_ashr_i32 s18, s1, s19 // (op 922) 


s_lshl_b32 s1, s47, 24 // (op 922) 
s_ashr_i32 s19, s1, 31 // (op 922) 

s_xor_b32 s1, s18, s19 // (op 922) 

s_sub_i32 s18, s1, s19 // (op 922) 


s_add_u32 s50, s6, 32 // (op 922) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 922) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 922) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 922) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 922) 

s_mul_i32 s1, s11, s24 // (op 922) 

s_mul_i32 s19, s8, s1 // (op 922) 

s_mul_i32 s1, s18, s19 // (op 922) 

s_sub_u32 s19, s16, s1 // (op 922) 

s_mov_b32 s1, s19 // (op 922) 
s_mul_hi_i32 s22, s1, s12 // (op 922) 
s_add_i32 s23, s22, s1 // (op 922) 


s_ashr_i32 s1, s23, 31 // (op 922) 
s_cmp_eq_i32 s12, 0 // (op 922) 
s_mov_b32 s22, scc // (op 922) 
s_cmp_eq_i32 s22, 0 // (op 922) 
s_cselect_b32 s24, 0, -1 // (op 922) 

s_and_b32 s22, s46, 31 // (op 922) 
s_lshl_b32 s25, 1, s22 // (op 922) 
s_add_i32 s49, s25, s24 // (op 922) 


s_and_b32 s24, s1, s49 // (op 922) 


s_add_i32 s1, s23, s24 // (op 922) 


s_ashr_i32 s23, s1, s22 // (op 922) 


s_lshl_b32 s1, s46, 24 // (op 922) 
s_ashr_i32 s22, s1, 31 // (op 922) 

s_xor_b32 s1, s23, s22 // (op 922) 

s_sub_i32 s23, s1, s22 // (op 922) 


s_mul_i32 s1, s48, s11 // (op 922) 
s_add_u32 s24, s4, 64 // (op 922) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 922) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 922) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 922) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 922) 

s_sub_i32 s22, s24, s1 // (op 922) 

s_mul_i32 s24, s23, s22 // (op 922) 


s_sub_u32 s25, s19, s24 // (op 922) 


s_add_u32 s19, s25, s1 // (op 922) 


s_cmp_eq_i32 s18, 0 // (op 922) 
s_mov_b32 s1, scc // (op 922) 

s_cmp_eq_i32 s1, 0 // (op 922) 
s_cselect_b32 s18, s19, s0 // (op 922) 



v_lshl_add_u32 v15, s18, 6, v13 // (op 922) 


v_mov_b32 v17, 0 // (op 922) convert
v_mov_b32 v16, v15 // (op 922) convert

v_mul_lo_u32 v19, v14, s27 // (op 922) most significant: low of low * high
v_mul_hi_u32 v13, v14, s26 // (op 922) most significant: high of low * low
v_mul_lo_u32 v18, v14, s26 // (op 922) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 922) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 922) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 922) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 922) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 922) 

v_mov_b32 v13, v16 // (op 922) convert

s_mul_i32 s1, s26, s31 // (op 924) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 924) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 924) least significant
s_mul_hi_u32 s19, s26, s30 // (op 924) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 924) most significant: sum
s_add_u32 s1, s1, s19 // (op 924) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 924) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 924) 

s_mov_b32 s52, s28 // (op 924) reg expression
s_mov_b32 s53, s29 // (op 924) reg expression
s_mov_b32 s54, s0 // (op 924) reg expression
s_mov_b32 s55, 131072 // (op 924) call()

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


v_mov_b32 v19, 0 // (op 967) convert
v_mov_b32 v18, v16 // (op 967) convert

v_mul_lo_u32 v17, v15, s33 // (op 967) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 967) most significant: high of low * low
v_mul_lo_u32 v16, v15, s32 // (op 967) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 967) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 967) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 967) most significant half



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

s_mul_hi_i32 s0, s16, s10 // (op 986) 
v_and_b32 v15, -2, v1 // (op 986) 
s_add_i32 s1, s0, s16 // (op 986) 

v_sub_u32 v16, v1, v15 // (op 986) 

s_ashr_i32 s0, s1, 31 // (op 986) 
v_lshlrev_b32 v15, 4, v16 // (op 986) 

s_cmp_eq_i32 s10, 0 // (op 986) 
s_mov_b32 s18, scc // (op 986) 
v_lshrrev_b32 v16, 1, v1 // (op 986) 
s_cmp_eq_i32 s18, 0 // (op 986) 
s_cselect_b32 s19, 0, -1 // (op 986) 

s_and_b32 s18, s13, 31 // (op 986) 
s_lshl_b32 s22, 1, s18 // (op 986) 
s_add_i32 s23, s22, s19 // (op 986) 


s_and_b32 s19, s0, s23 // (op 986) 


s_add_i32 s0, s1, s19 // (op 986) 


s_ashr_i32 s1, s0, s18 // (op 986) 


s_lshl_b32 s0, s13, 24 // (op 986) 
s_ashr_i32 s18, s0, 31 // (op 986) 

s_xor_b32 s0, s1, s18 // (op 986) 

s_sub_i32 s1, s0, s18 // (op 986) 


s_mul_hi_i32 s0, s1, s14 // (op 986) 
s_add_i32 s18, s0, s1 // (op 986) 

s_ashr_i32 s0, s18, 31 // (op 986) 
s_cmp_eq_i32 s14, 0 // (op 986) 
s_mov_b32 s19, scc // (op 986) 
s_cmp_eq_i32 s19, 0 // (op 986) 
s_cselect_b32 s22, 0, -1 // (op 986) 

s_and_b32 s19, s15, 31 // (op 986) 
s_lshl_b32 s23, 1, s19 // (op 986) 
s_add_i32 s24, s23, s22 // (op 986) 


s_and_b32 s25, s0, s24 // (op 986) 


s_add_i32 s0, s18, s25 // (op 986) 


s_ashr_i32 s18, s0, s19 // (op 986) 


s_lshl_b32 s0, s15, 24 // (op 986) 
s_ashr_i32 s19, s0, 31 // (op 986) 

s_xor_b32 s0, s18, s19 // (op 986) 

s_sub_i32 s18, s0, s19 // (op 986) 


s_add_u32 s50, s6, 32 // (op 986) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 986) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 986) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 986) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 986) 

s_mul_i32 s0, s18, s24 // (op 986) 

s_sub_i32 s18, s1, s0 // (op 986) 


s_mul_hi_i32 s19, s16, s9 // (op 986) 
s_add_i32 s0, s19, s16 // (op 986) 

s_ashr_i32 s1, s0, 31 // (op 986) 
s_cmp_eq_i32 s9, 0 // (op 986) 
s_mov_b32 s19, scc // (op 986) 
s_cmp_eq_i32 s19, 0 // (op 986) 
s_cselect_b32 s22, 0, -1 // (op 986) 

s_and_b32 s19, s47, 31 // (op 986) 
s_lshl_b32 s23, 1, s19 // (op 986) 
s_add_i32 s49, s23, s22 // (op 986) 


s_and_b32 s22, s1, s49 // (op 986) 


s_add_i32 s1, s0, s22 // (op 986) 


s_ashr_i32 s0, s1, s19 // (op 986) 


s_lshl_b32 s1, s47, 24 // (op 986) 
s_ashr_i32 s19, s1, 31 // (op 986) 

s_xor_b32 s1, s0, s19 // (op 986) 

s_sub_i32 s0, s1, s19 // (op 986) 


s_mul_i32 s1, s11, s24 // (op 986) 

s_mul_i32 s19, s8, s1 // (op 986) 

s_mul_i32 s1, s0, s19 // (op 986) 

s_sub_u32 s19, s16, s1 // (op 986) 

s_mul_hi_i32 s1, s19, s12 // (op 986) 
s_add_i32 s22, s1, s19 // (op 986) 


s_ashr_i32 s1, s22, 31 // (op 986) 
s_cmp_eq_i32 s12, 0 // (op 986) 
s_mov_b32 s19, scc // (op 986) 
s_cmp_eq_i32 s19, 0 // (op 986) 
s_cselect_b32 s23, 0, -1 // (op 986) 

s_and_b32 s19, s46, 31 // (op 986) 
s_lshl_b32 s24, 1, s19 // (op 986) 
s_add_i32 s25, s24, s23 // (op 986) 


s_and_b32 s23, s1, s25 // (op 986) 


s_add_i32 s1, s22, s23 // (op 986) 


s_ashr_i32 s22, s1, s19 // (op 986) 


s_lshl_b32 s1, s46, 24 // (op 986) 
s_ashr_i32 s19, s1, 31 // (op 986) 

s_xor_b32 s1, s22, s19 // (op 986) 

s_sub_i32 s22, s1, s19 // (op 986) 


s_cmp_eq_i32 s0, 0 // (op 986) 
s_mov_b32 s1, scc // (op 986) 

s_cmp_eq_i32 s1, 0 // (op 986) 
s_cselect_b32 s0, s22, s18 // (op 986) 



v_lshl_add_u32 v17, s0, 5, v15 // (op 986) 


v_mov_b32 v19, 0 // (op 986) convert
v_mov_b32 v18, v17 // (op 986) convert

v_mul_lo_u32 v21, v16, s33 // (op 986) most significant: low of low * high
v_mul_hi_u32 v15, v16, s32 // (op 986) most significant: high of low * low
v_mul_lo_u32 v20, v16, s32 // (op 986) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 986) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 986) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 986) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 986) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 986) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 986) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 986) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 986) 

v_mov_b32 v15, v16 // (op 986) convert

s_mul_i32 s1, s31, 6 // (op 988) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 988) least significant
s_mul_hi_u32 s18, s30, 6 // (op 988) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 988) most significant: sum

s_mul_i32 s19, s32, s1 // (op 988) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 988) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 988) least significant
s_mul_hi_u32 s23, s32, s0 // (op 988) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 988) most significant: sum
s_add_u32 s19, s19, s23 // (op 988) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 988) 

s_mov_b32 s64, s34 // (op 988) reg expression
s_mov_b32 s65, s35 // (op 988) reg expression
s_mov_b32 s66, s0 // (op 988) reg expression
s_mov_b32 s67, 131072 // (op 988) call()

s_mul_hi_i32 s0, s16, s10 // (op 1031) 
v_and_b32 v16, -64, v1 // (op 1031) 
s_add_i32 s1, s0, s16 // (op 1031) 

v_sub_u32 v17, v1, v16 // (op 1031) 

s_ashr_i32 s0, s1, 31 // (op 1031) 
v_lshrrev_b32 v16, 6, v1 // (op 1031) 
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



v_lshl_add_u32 v18, s18, 6, v17 // (op 1031) 


v_mul_lo_u32 v17, v16, s36 // (op 1031) 

v_add_u32 v16, v18, v17 // (op 1031) 


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

s_mul_hi_i32 s0, s16, s10 // (op 1050) 
v_and_b32 v17, -64, v1 // (op 1050) 
s_add_i32 s1, s0, s16 // (op 1050) 

v_sub_u32 v18, v1, v17 // (op 1050) 

s_ashr_i32 s0, s1, 31 // (op 1050) 
v_lshrrev_b32 v17, 6, v1 // (op 1050) 
s_cmp_eq_i32 s10, 0 // (op 1050) 
s_mov_b32 s18, scc // (op 1050) 
s_cmp_eq_i32 s18, 0 // (op 1050) 
s_cselect_b32 s19, 0, -1 // (op 1050) 

s_and_b32 s18, s13, 31 // (op 1050) 
s_lshl_b32 s22, 1, s18 // (op 1050) 
s_add_i32 s23, s22, s19 // (op 1050) 


s_and_b32 s19, s0, s23 // (op 1050) 


s_add_i32 s0, s1, s19 // (op 1050) 


s_ashr_i32 s1, s0, s18 // (op 1050) 


s_lshl_b32 s0, s13, 24 // (op 1050) 
s_ashr_i32 s18, s0, 31 // (op 1050) 

s_xor_b32 s0, s1, s18 // (op 1050) 

s_sub_i32 s1, s0, s18 // (op 1050) 


s_mul_hi_i32 s0, s1, s14 // (op 1050) 
s_add_i32 s18, s0, s1 // (op 1050) 

s_ashr_i32 s0, s18, 31 // (op 1050) 
s_cmp_eq_i32 s14, 0 // (op 1050) 
s_mov_b32 s19, scc // (op 1050) 
s_cmp_eq_i32 s19, 0 // (op 1050) 
s_cselect_b32 s22, 0, -1 // (op 1050) 

s_and_b32 s19, s15, 31 // (op 1050) 
s_lshl_b32 s23, 1, s19 // (op 1050) 
s_add_i32 s24, s23, s22 // (op 1050) 


s_and_b32 s25, s0, s24 // (op 1050) 


s_add_i32 s0, s18, s25 // (op 1050) 


s_ashr_i32 s18, s0, s19 // (op 1050) 


s_lshl_b32 s0, s15, 24 // (op 1050) 
s_ashr_i32 s19, s0, 31 // (op 1050) 

s_xor_b32 s0, s18, s19 // (op 1050) 

s_sub_i32 s18, s0, s19 // (op 1050) 


s_mul_i32 s0, s1, s11 // (op 1050) 

s_sub_u32 s1, s16, s0 // (op 1050) 

s_mul_i32 s0, s18, s11 // (op 1050) 
s_add_u32 s0, s0, s1 // (op 1050) 


s_mul_hi_i32 s1, s16, s9 // (op 1050) 
s_add_i32 s18, s1, s16 // (op 1050) 

s_ashr_i32 s1, s18, 31 // (op 1050) 
s_cmp_eq_i32 s9, 0 // (op 1050) 
s_mov_b32 s19, scc // (op 1050) 
s_cmp_eq_i32 s19, 0 // (op 1050) 
s_cselect_b32 s22, 0, -1 // (op 1050) 

s_and_b32 s19, s47, 31 // (op 1050) 
s_lshl_b32 s23, 1, s19 // (op 1050) 
s_add_i32 s24, s23, s22 // (op 1050) 


s_and_b32 s25, s1, s24 // (op 1050) 


s_add_i32 s1, s18, s25 // (op 1050) 


s_ashr_i32 s18, s1, s19 // (op 1050) 


s_lshl_b32 s1, s47, 24 // (op 1050) 
s_ashr_i32 s19, s1, 31 // (op 1050) 

s_xor_b32 s1, s18, s19 // (op 1050) 

s_sub_i32 s18, s1, s19 // (op 1050) 


s_add_u32 s50, s6, 32 // (op 1050) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1050) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1050) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1050) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1050) 

s_mul_i32 s1, s11, s24 // (op 1050) 

s_mul_i32 s19, s8, s1 // (op 1050) 

s_mul_i32 s1, s18, s19 // (op 1050) 

s_sub_u32 s19, s16, s1 // (op 1050) 

s_mov_b32 s1, s19 // (op 1050) 
s_mul_hi_i32 s22, s1, s12 // (op 1050) 
s_add_i32 s23, s22, s1 // (op 1050) 


s_ashr_i32 s1, s23, 31 // (op 1050) 
s_cmp_eq_i32 s12, 0 // (op 1050) 
s_mov_b32 s22, scc // (op 1050) 
s_cmp_eq_i32 s22, 0 // (op 1050) 
s_cselect_b32 s24, 0, -1 // (op 1050) 

s_and_b32 s22, s46, 31 // (op 1050) 
s_lshl_b32 s25, 1, s22 // (op 1050) 
s_add_i32 s49, s25, s24 // (op 1050) 


s_and_b32 s24, s1, s49 // (op 1050) 


s_add_i32 s1, s23, s24 // (op 1050) 


s_ashr_i32 s23, s1, s22 // (op 1050) 


s_lshl_b32 s1, s46, 24 // (op 1050) 
s_ashr_i32 s22, s1, 31 // (op 1050) 

s_xor_b32 s1, s23, s22 // (op 1050) 

s_sub_i32 s23, s1, s22 // (op 1050) 


s_mul_i32 s1, s48, s11 // (op 1050) 
s_add_u32 s24, s4, 64 // (op 1050) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1050) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1050) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1050) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1050) 

s_sub_i32 s22, s24, s1 // (op 1050) 

s_mul_i32 s24, s23, s22 // (op 1050) 


s_sub_u32 s25, s19, s24 // (op 1050) 


s_add_u32 s19, s25, s1 // (op 1050) 


s_cmp_eq_i32 s18, 0 // (op 1050) 
s_mov_b32 s1, scc // (op 1050) 

s_cmp_eq_i32 s1, 0 // (op 1050) 
s_cselect_b32 s18, s19, s0 // (op 1050) 



v_lshl_add_u32 v19, s18, 6, v18 // (op 1050) 


v_mul_lo_u32 v18, v17, s36 // (op 1050) 

v_add_u32 v17, v19, v18 // (op 1050) 


s_mul_i32 s1, s36, s41 // (op 1052) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1052) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1052) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1052) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1052) most significant: sum
s_add_u32 s1, s1, s19 // (op 1052) most significant: sum


s_mov_b32 s68, s38 // (op 1052) reg expression
s_mov_b32 s69, s39 // (op 1052) reg expression
s_mov_b32 s70, s0 // (op 1052) reg expression
s_mov_b32 s71, 131072 // (op 1052) call()

s_mul_hi_i32 s0, s16, s10 // (op 1095) 
v_lshrrev_b32 v18, 6, v1 // (op 1095) 
s_add_i32 s1, s0, s16 // (op 1095) 

v_and_b32 v19, 1, v18 // (op 1095) 

s_ashr_i32 s0, s1, 31 // (op 1095) 
v_and_b32 v18, 15, v1 // (op 1095) 
s_cmp_eq_i32 s10, 0 // (op 1095) 
s_mov_b32 s18, scc // (op 1095) 
v_lshl_add_u32 v20, v19, 4, v18 // (op 1095) 


s_cmp_eq_i32 s18, 0 // (op 1095) 
s_cselect_b32 s19, 0, -1 // (op 1095) 

v_and_b32 v18, 63, v1 // (op 1095) 
s_and_b32 s18, s13, 31 // (op 1095) 
v_lshrrev_b32 v19, 4, v18 // (op 1095) 

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



v_lshl_add_u32 v18, s0, 5, v20 // (op 1095) 


v_mul_lo_u32 v20, v19, s42 // (op 1095) 
v_add_u32 v20, v20, v18 // (op 1095) 


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

s_mul_hi_i32 s0, s16, s10 // (op 1114) 
v_lshrrev_b32 v18, 6, v1 // (op 1114) 
s_add_i32 s1, s0, s16 // (op 1114) 

v_and_b32 v19, 1, v18 // (op 1114) 

s_ashr_i32 s0, s1, 31 // (op 1114) 
v_and_b32 v18, 15, v1 // (op 1114) 
s_cmp_eq_i32 s10, 0 // (op 1114) 
s_mov_b32 s18, scc // (op 1114) 
v_lshl_add_u32 v21, v19, 4, v18 // (op 1114) 


s_cmp_eq_i32 s18, 0 // (op 1114) 
s_cselect_b32 s19, 0, -1 // (op 1114) 

v_and_b32 v18, 63, v1 // (op 1114) 
s_and_b32 s18, s13, 31 // (op 1114) 
v_lshrrev_b32 v19, 4, v18 // (op 1114) 

s_lshl_b32 s22, 1, s18 // (op 1114) 
s_add_i32 s23, s22, s19 // (op 1114) 


s_and_b32 s19, s0, s23 // (op 1114) 


s_add_i32 s0, s1, s19 // (op 1114) 


s_ashr_i32 s1, s0, s18 // (op 1114) 


s_lshl_b32 s0, s13, 24 // (op 1114) 
s_ashr_i32 s18, s0, 31 // (op 1114) 

s_xor_b32 s0, s1, s18 // (op 1114) 

s_sub_i32 s1, s0, s18 // (op 1114) 


s_mul_hi_i32 s0, s1, s14 // (op 1114) 
s_add_i32 s18, s0, s1 // (op 1114) 

s_ashr_i32 s0, s18, 31 // (op 1114) 
s_cmp_eq_i32 s14, 0 // (op 1114) 
s_mov_b32 s19, scc // (op 1114) 
s_cmp_eq_i32 s19, 0 // (op 1114) 
s_cselect_b32 s22, 0, -1 // (op 1114) 

s_and_b32 s19, s15, 31 // (op 1114) 
s_lshl_b32 s23, 1, s19 // (op 1114) 
s_add_i32 s24, s23, s22 // (op 1114) 


s_and_b32 s25, s0, s24 // (op 1114) 


s_add_i32 s0, s18, s25 // (op 1114) 


s_ashr_i32 s18, s0, s19 // (op 1114) 


s_lshl_b32 s0, s15, 24 // (op 1114) 
s_ashr_i32 s19, s0, 31 // (op 1114) 

s_xor_b32 s0, s18, s19 // (op 1114) 

s_sub_i32 s18, s0, s19 // (op 1114) 


s_add_u32 s50, s6, 32 // (op 1114) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1114) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1114) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1114) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1114) 

s_mul_i32 s0, s18, s24 // (op 1114) 

s_sub_i32 s18, s1, s0 // (op 1114) 


s_mul_hi_i32 s19, s16, s9 // (op 1114) 
s_add_i32 s0, s19, s16 // (op 1114) 

s_ashr_i32 s1, s0, 31 // (op 1114) 
s_cmp_eq_i32 s9, 0 // (op 1114) 
s_mov_b32 s19, scc // (op 1114) 
s_cmp_eq_i32 s19, 0 // (op 1114) 
s_cselect_b32 s22, 0, -1 // (op 1114) 

s_and_b32 s19, s47, 31 // (op 1114) 
s_lshl_b32 s23, 1, s19 // (op 1114) 
s_add_i32 s49, s23, s22 // (op 1114) 


s_and_b32 s22, s1, s49 // (op 1114) 


s_add_i32 s1, s0, s22 // (op 1114) 


s_ashr_i32 s0, s1, s19 // (op 1114) 


s_lshl_b32 s1, s47, 24 // (op 1114) 
s_ashr_i32 s19, s1, 31 // (op 1114) 

s_xor_b32 s1, s0, s19 // (op 1114) 

s_sub_i32 s0, s1, s19 // (op 1114) 


s_mul_i32 s1, s11, s24 // (op 1114) 

s_mul_i32 s19, s8, s1 // (op 1114) 

s_mul_i32 s1, s0, s19 // (op 1114) 

s_sub_u32 s19, s16, s1 // (op 1114) 

s_mul_hi_i32 s1, s19, s12 // (op 1114) 
s_add_i32 s22, s1, s19 // (op 1114) 


s_ashr_i32 s1, s22, 31 // (op 1114) 
s_cmp_eq_i32 s12, 0 // (op 1114) 
s_mov_b32 s19, scc // (op 1114) 
s_cmp_eq_i32 s19, 0 // (op 1114) 
s_cselect_b32 s23, 0, -1 // (op 1114) 

s_and_b32 s19, s46, 31 // (op 1114) 
s_lshl_b32 s24, 1, s19 // (op 1114) 
s_add_i32 s25, s24, s23 // (op 1114) 


s_and_b32 s23, s1, s25 // (op 1114) 


s_add_i32 s1, s22, s23 // (op 1114) 


s_ashr_i32 s22, s1, s19 // (op 1114) 


s_lshl_b32 s1, s46, 24 // (op 1114) 
s_ashr_i32 s19, s1, 31 // (op 1114) 

s_xor_b32 s1, s22, s19 // (op 1114) 

s_sub_i32 s22, s1, s19 // (op 1114) 


s_cmp_eq_i32 s0, 0 // (op 1114) 
s_mov_b32 s1, scc // (op 1114) 

s_cmp_eq_i32 s1, 0 // (op 1114) 
s_cselect_b32 s0, s22, s18 // (op 1114) 



v_lshl_add_u32 v18, s0, 5, v21 // (op 1114) 


v_mul_lo_u32 v21, v19, s42 // (op 1114) 
v_add_u32 v21, v21, v18 // (op 1114) 


s_mul_i32 s1, s42, s41 // (op 1116) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1116) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1116) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1116) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1116) most significant: sum
s_add_u32 s1, s1, s19 // (op 1116) most significant: sum


s_mov_b32 s72, s44 // (op 1116) reg expression
s_mov_b32 s73, s45 // (op 1116) reg expression
s_mov_b32 s74, s0 // (op 1116) reg expression
s_mov_b32 s75, 131072 // (op 1116) call()


v_mov_b64 v[18:19], s[42:43] // (op 774) convert

v_mov_b32 v22, v20 // (op 774) reg expression
v_mov_b32 v23, v22 // (op 774) 
buffer_load_ubyte v24, v23, s[72:75], 0 offen offset: 0 // (op 774) Load value





v_mov_b32 v18, v14 // (op 778) reg expression

buffer_load_dwordx3 v[26:28], v18, s[64:67], 0 offen // (op 778) Load value



v_mov_b32 v25, v12 // (op 781) reg expression

buffer_load_dwordx4 v[30:33], v25, s[52:55], 0 offen // (op 781) Load value



v_mov_b32 v25, v16 // (op 784) reg expression

buffer_load_ubyte v29, v25, s[68:71], 0 offen offset: 0 // (op 784) Load value


v_mov_b64 v[18:19], s[42:43] // (op 788) convert

s_lshl_b32 s0, s42, 2 // (op 788) 
v_add_u32 v25, v20, s0 // (op 788) 

v_mov_b32 v22, v25 // (op 788) 
buffer_load_ubyte v23, v22, s[72:75], 0 offen offset: 0 // (op 788) Load value






s_lshl_b64 s[0:1], s[32:33], 7 // (op 792) 
s_mul_i32 s19, s1, 6 // (op 792) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 792) least significant
s_mul_hi_u32 s22, s0, 6 // (op 792) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 792) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 792) 

v_add_u32 v22, v14, s0 // (op 792) 


buffer_load_dwordx3 v[18:20], v22, s[64:67], 0 offen // (op 792) Load value




s_lshl_b64 s[0:1], s[26:27], 7 // (op 795) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 795) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 795) 

v_add_u32 v14, v12, s0 // (op 795) 


buffer_load_dwordx4 v[34:37], v14, s[52:55], 0 offen // (op 795) Load value




s_lshl_b32 s0, s36, 2 // (op 798) 
v_add_u32 v12, v16, s0 // (op 798) 


buffer_load_ubyte v14, v12, s[68:71], 0 offen offset: 0 // (op 798) Load value


v_and_b32 v12, -2, v1 // (op 1239) 
v_sub_u32 v16, v1, v12 // (op 1239) 

v_lshlrev_b32 v12, 4, v16 // (op 1239) 

v_lshrrev_b32 v16, 1, v1 // (op 1239) 
v_lshl_add_u32 v22, v16, 5, v12 // (op 1239) 


v_mul_lo_u32 v12, v22, 6 // (op 1239) 
v_lshrrev_b32 v16, 3, v12 // (op 1239) 

v_lshrrev_b32 v12, 4, v22 // (op 1239) 

v_lshlrev_b32 v22, 2, v12 // (op 1239) 

v_add_u32 v12, v16, v22 // (op 1239) 





s_waitcnt vmcnt(6)
ds_write_b96 v12, v[26:28] // (op 800) Store local data 

v_and_b32 v12, -2, v1 // (op 1181) 
v_sub_u32 v16, v1, v12 // (op 1181) 

v_lshlrev_b32 v12, 5, v16 // (op 1181) 

v_lshrrev_b32 v16, 1, v1 // (op 1181) 
v_lshl_add_u32 v22, v16, 6, v12 // (op 1181) 


v_lshlrev_b32 v12, 2, v22 // (op 1181) 

v_lshrrev_b32 v16, 3, v12 // (op 1181) 




s_waitcnt vmcnt(5)
ds_write_b128 v16, v[30:33] offset:4096 // (op 801) Store local data 

v_and_b32 v12, -64, v1 // (op 1292) 
v_sub_u32 v16, v1, v12 // (op 1292) 

v_and_b32 v12, 4294967232, v1 // (op 1292) 
v_add_u32 v22, v12, v16 // (op 1292) 





s_waitcnt vmcnt(4)
ds_write_b8 v22, v29 offset:8192 // (op 802) Store local data 

v_and_b32 v12, -2, v1 // (op 1152) 
v_sub_u32 v16, v1, v12 // (op 1152) 

v_lshlrev_b32 v12, 5, v16 // (op 1152) 

v_lshrrev_b32 v16, 1, v1 // (op 1152) 
v_lshl_add_u32 v22, v16, 6, v12 // (op 1152) 


v_lshlrev_b32 v12, 2, v22 // (op 1152) 

v_lshrrev_b32 v16, 3, v12 // (op 1152) 

v_and_b32 v12, -2, v1 // (op 1210) 
v_sub_u32 v22, v1, v12 // (op 1210) 

v_lshlrev_b32 v12, 4, v22 // (op 1210) 

v_lshrrev_b32 v22, 1, v1 // (op 1210) 
v_lshl_add_u32 v25, v22, 5, v12 // (op 1210) 


v_mul_lo_u32 v12, v25, 6 // (op 1210) 
v_lshrrev_b32 v22, 3, v12 // (op 1210) 

v_lshrrev_b32 v12, 4, v25 // (op 1210) 

v_lshlrev_b32 v25, 2, v12 // (op 1210) 

v_add_u32 v12, v22, v25 // (op 1210) 


v_and_b32 v22, -64, v1 // (op 1263) 
v_sub_u32 v25, v1, v22 // (op 1263) 

v_and_b32 v22, 4294967232, v1 // (op 1263) 
v_add_u32 v38, v22, v25 // (op 1263) 


v_and_b32 v22, -2, v1 // (op 1339) 
v_sub_u32 v25, v1, v22 // (op 1339) 

v_lshlrev_b32 v22, 5, v25 // (op 1339) 

v_lshrrev_b32 v25, 1, v1 // (op 1339) 
v_lshl_add_u32 v39, v25, 6, v22 // (op 1339) 


v_lshlrev_b32 v22, 2, v39 // (op 1339) 

v_lshrrev_b32 v25, 3, v22 // (op 1339) 

v_and_b32 v22, -2, v1 // (op 1368) 
v_sub_u32 v39, v1, v22 // (op 1368) 

v_lshlrev_b32 v22, 4, v39 // (op 1368) 

v_lshrrev_b32 v39, 1, v1 // (op 1368) 
v_lshl_add_u32 v40, v39, 5, v22 // (op 1368) 


v_mul_lo_u32 v22, v40, 6 // (op 1368) 
v_lshrrev_b32 v39, 3, v22 // (op 1368) 

v_lshrrev_b32 v22, 4, v40 // (op 1368) 

v_lshlrev_b32 v40, 2, v22 // (op 1368) 

v_add_u32 v22, v39, v40 // (op 1368) 


v_and_b32 v39, -64, v1 // (op 1392) 
v_sub_u32 v40, v1, v39 // (op 1392) 

v_and_b32 v39, 4294967232, v1 // (op 1392) 
v_add_u32 v41, v39, v40 // (op 1392) 


s_mov_b32 s0, 0 // (op 311) call()
s_ashr_i64 s[18:19], s[30:31], 7 // (op 310) 
s_mov_b32 s1, s18 // (op 310) convert

s_and_b32 s18, s1, 4294967294 // (op 310) 

s_cmp_lt_u32 s0, s18 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_310 if false)
RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_310:
s_add_u32 s0, s0, 2 // (op 312) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1450) 



ds_read_b64_tr_b4 v[42:43], v7 offset:4096 // (op 65) Transpose load local data 
ds_read_b64_tr_b4 v[44:45], v7 offset:4608 // (op 65) Transpose load local data 
s_barrier  // (op 1463) 



ds_read_b96_tr_b6 v[46:48], v0 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v0 offset:512 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1477) 


ds_read_u8 v39, v4 offset:8192 // (op 75) Load local data 



ds_read_b64_tr_b4 v[52:53], v7 offset:4112 // (op 340) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[46:51], a[0:3], v39, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 82) 
ds_read_b64_tr_b4 v[54:55], v7 offset:4624 // (op 340) Transpose load local data 




ds_read_u8 v39, v4 offset:8224 // (op 344) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[46:51], a[4:7], v39, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 346) 




v_mov_b64 v[42:43], s[42:43] // (op 80) convert

v_mov_b32 v39, v21 // (op 80) 
s_lshl_b32 s1, s42, 3 // (op 80) 
v_add_u32 v40, v39, s1 // (op 80) 

v_mov_b32 v44, v40 // (op 80) 
buffer_load_ubyte v24, v44, s[72:75], 0 offen offset: 0 // (op 80) Load value





v_mov_b32 v40, v15 // (op 10) 
s_lshl_b64 s[18:19], s[32:33], 7 // (op 10) 
s_mul_i32 s51, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 10) least significant
s_mul_hi_u32 s1, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v42, v40, s1 // (op 10) 


buffer_load_dwordx3 v[26:28], v42, s[64:67], 0 offen // (op 10) Load value



v_mov_b32 v42, v13 // (op 4) 
s_lshl_b64 s[18:19], s[26:27], 7 // (op 4) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 4) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 4) 

s_lshl_b32 s1, s18, 1 // (op 4) 

v_add_u32 v43, v42, s1 // (op 4) 


buffer_load_dwordx4 v[30:33], v43, s[52:55], 0 offen // (op 4) Load value



v_mov_b32 v43, v17 // (op 17) 
s_lshl_b32 s1, s36, 3 // (op 17) 
v_add_u32 v44, v43, s1 // (op 17) 


buffer_load_ubyte v29, v44, s[68:71], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1500) 



s_waitcnt vmcnt(6)
ds_write_b96 v22, v[18:20] offset:8448 // (op 235) Store local data 
s_barrier  // (op 1494) 



s_waitcnt vmcnt(5)
ds_write_b128 v25, v[34:37] offset:12544 // (op 225) Store local data 
s_barrier  // (op 1506) 



s_waitcnt vmcnt(4)
ds_write_b8 v41, v14 offset:16640 // (op 236) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1491) 



ds_read_b64_tr_b4 v[44:45], v11 offset:12544 // (op 227) Transpose load local data 
ds_read_b64_tr_b4 v[46:47], v11 offset:13056 // (op 227) Transpose load local data 
s_barrier  // (op 1497) 



ds_read_b96_tr_b6 v[48:50], v3 offset:8448 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v3 offset:8960 // (op 229) Transpose load local data 
s_barrier  // (op 1543) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1503) 


ds_read_u8 v54, v10 offset:16640 // (op 231) Load local data 



ds_read_b64_tr_b4 v[56:57], v11 offset:12560 // (op 353) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:47], v[48:53], a[0:3], v54, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 234) 
ds_read_b64_tr_b4 v[58:59], v11 offset:13072 // (op 353) Transpose load local data 




s_barrier  // (op 1577) 
ds_read_u8 v44, v10 offset:16672 // (op 357) Load local data 
s_barrier  // (op 1581) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[48:53], a[4:7], v44, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 347) 




v_mov_b64 v[44:45], s[42:43] // (op 233) convert

s_lshl_b32 s1, s42, 2 // (op 233) 
s_mul_i32 s18, 3, s1 // (op 233) 

v_add_u32 v46, v39, s18 // (op 233) 

v_mov_b32 v47, v46 // (op 233) 
buffer_load_ubyte v23, v47, s[72:75], 0 offen offset: 0 // (op 233) Load value






s_lshl_b32 s1, s42, 3 // (op 1106) 
v_add_u32 v21, v21, s1 // (op 1106) 

s_lshl_b64 s[18:19], s[32:33], 7 // (op 237) 
s_mul_i32 s51, s19, 6 // (op 237) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 237) least significant
s_mul_hi_u32 s1, s18, 6 // (op 237) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 237) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 237) 

s_mul_i32 s1, 3, s18 // (op 237) 

v_add_u32 v39, v40, s1 // (op 237) 


buffer_load_dwordx3 v[18:20], v39, s[64:67], 0 offen // (op 237) Load value




s_lshl_b64 s[18:19], s[26:27], 7 // (op 224) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 224) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 224) 

s_mul_i32 s1, 3, s18 // (op 224) 

v_add_u32 v39, v42, s1 // (op 224) 


s_lshl_b64 s[18:19], s[32:33], 7 // (op 978) 
s_mul_i32 s51, s19, 6 // (op 978) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 978) least significant
s_mul_hi_u32 s1, s18, 6 // (op 978) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 978) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 978) 

s_lshl_b32 s1, s18, 1 // (op 978) 

v_add_u32 v15, v15, s1 // (op 978) 

buffer_load_dwordx4 v[34:37], v39, s[52:55], 0 offen // (op 224) Load value




s_lshl_b32 s1, s36, 2 // (op 238) 
s_mul_i32 s18, 3, s1 // (op 238) 

v_add_u32 v42, v43, s18 // (op 238) 


s_lshl_b64 s[18:19], s[26:27], 7 // (op 914) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 914) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 914) 

s_lshl_b32 s1, s18, 1 // (op 914) 

v_add_u32 v13, v13, s1 // (op 914) 

buffer_load_ubyte v14, v42, s[68:71], 0 offen offset: 0 // (op 238) Load value


s_barrier  // (op 1466) 
s_lshl_b32 s1, s36, 3 // (op 1042) 



v_add_u32 v17, v17, s1 // (op 1042) 

s_waitcnt vmcnt(6)
ds_write_b96 v12, v[26:28] // (op 13) Store local data 
s_barrier  // (op 1453) 



s_waitcnt vmcnt(5)
ds_write_b128 v16, v[30:33] offset:4096 // (op 6) Store local data 
s_barrier  // (op 1480) 



s_waitcnt vmcnt(4)
ds_write_b8 v38, v29 offset:8192 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 310) 
s_mov_b32 s1, s18 // (op 310) convert

s_and_b32 s18, s1, 4294967294 // (op 310) 

s_cmp_lt_u32 s0, s18 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_310 if true)
RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_310:















s_mul_hi_i32 s0, s16, s10 // (op 876) 
v_and_b32 v12, -2, v1 // (op 876) 
s_add_i32 s1, s0, s16 // (op 876) 

v_sub_u32 v13, v1, v12 // (op 876) 

s_ashr_i32 s0, s1, 31 // (op 876) 
v_lshlrev_b32 v12, 5, v13 // (op 876) 

s_cmp_eq_i32 s10, 0 // (op 876) 
s_mov_b32 s18, scc // (op 876) 
v_lshrrev_b32 v13, 1, v1 // (op 876) 
s_cmp_eq_i32 s18, 0 // (op 876) 
s_cselect_b32 s19, 0, -1 // (op 876) 

s_and_b32 s18, s13, 31 // (op 876) 
s_lshl_b32 s22, 1, s18 // (op 876) 
s_add_i32 s23, s22, s19 // (op 876) 


s_and_b32 s19, s0, s23 // (op 876) 


s_add_i32 s0, s1, s19 // (op 876) 


s_ashr_i32 s1, s0, s18 // (op 876) 


s_lshl_b32 s0, s13, 24 // (op 876) 
s_ashr_i32 s18, s0, 31 // (op 876) 

s_xor_b32 s0, s1, s18 // (op 876) 

s_sub_i32 s1, s0, s18 // (op 876) 


s_mul_hi_i32 s0, s1, s14 // (op 876) 
s_add_i32 s18, s0, s1 // (op 876) 

s_ashr_i32 s0, s18, 31 // (op 876) 
s_cmp_eq_i32 s14, 0 // (op 876) 
s_mov_b32 s19, scc // (op 876) 
s_cmp_eq_i32 s19, 0 // (op 876) 
s_cselect_b32 s22, 0, -1 // (op 876) 

s_and_b32 s19, s15, 31 // (op 876) 
s_lshl_b32 s23, 1, s19 // (op 876) 
s_add_i32 s24, s23, s22 // (op 876) 


s_and_b32 s25, s0, s24 // (op 876) 


s_add_i32 s0, s18, s25 // (op 876) 


s_ashr_i32 s18, s0, s19 // (op 876) 


s_lshl_b32 s0, s15, 24 // (op 876) 
s_ashr_i32 s19, s0, 31 // (op 876) 

s_xor_b32 s0, s18, s19 // (op 876) 

s_sub_i32 s18, s0, s19 // (op 876) 


s_mul_i32 s0, s1, s11 // (op 876) 

s_sub_u32 s1, s16, s0 // (op 876) 

s_mul_i32 s0, s18, s11 // (op 876) 
s_add_u32 s0, s0, s1 // (op 876) 


s_mul_hi_i32 s1, s16, s9 // (op 876) 
s_add_i32 s18, s1, s16 // (op 876) 

s_ashr_i32 s1, s18, 31 // (op 876) 
s_cmp_eq_i32 s9, 0 // (op 876) 
s_mov_b32 s19, scc // (op 876) 
s_cmp_eq_i32 s19, 0 // (op 876) 
s_cselect_b32 s22, 0, -1 // (op 876) 

s_and_b32 s19, s47, 31 // (op 876) 
s_lshl_b32 s23, 1, s19 // (op 876) 
s_add_i32 s24, s23, s22 // (op 876) 


s_and_b32 s25, s1, s24 // (op 876) 


s_add_i32 s1, s18, s25 // (op 876) 


s_ashr_i32 s18, s1, s19 // (op 876) 


s_lshl_b32 s1, s47, 24 // (op 876) 
s_ashr_i32 s19, s1, 31 // (op 876) 

s_xor_b32 s1, s18, s19 // (op 876) 

s_sub_i32 s18, s1, s19 // (op 876) 


s_add_u32 s50, s6, 32 // (op 876) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 876) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 876) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 876) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 876) 

s_mul_i32 s1, s11, s24 // (op 876) 

s_mul_i32 s19, s8, s1 // (op 876) 

s_mul_i32 s1, s18, s19 // (op 876) 

s_sub_u32 s19, s16, s1 // (op 876) 

s_mov_b32 s1, s19 // (op 876) 
s_mul_hi_i32 s22, s1, s12 // (op 876) 
s_add_i32 s23, s22, s1 // (op 876) 


s_ashr_i32 s1, s23, 31 // (op 876) 
s_cmp_eq_i32 s12, 0 // (op 876) 
s_mov_b32 s22, scc // (op 876) 
s_cmp_eq_i32 s22, 0 // (op 876) 
s_cselect_b32 s24, 0, -1 // (op 876) 

s_and_b32 s22, s46, 31 // (op 876) 
s_lshl_b32 s25, 1, s22 // (op 876) 
s_add_i32 s49, s25, s24 // (op 876) 


s_and_b32 s24, s1, s49 // (op 876) 


s_add_i32 s1, s23, s24 // (op 876) 


s_ashr_i32 s23, s1, s22 // (op 876) 


s_lshl_b32 s1, s46, 24 // (op 876) 
s_ashr_i32 s22, s1, 31 // (op 876) 

s_xor_b32 s1, s23, s22 // (op 876) 

s_sub_i32 s23, s1, s22 // (op 876) 


s_mul_i32 s1, s48, s11 // (op 876) 
s_add_u32 s24, s4, 64 // (op 876) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 876) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 876) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 876) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 876) 

s_sub_i32 s22, s24, s1 // (op 876) 

s_mul_i32 s24, s23, s22 // (op 876) 


s_sub_u32 s25, s19, s24 // (op 876) 


s_add_u32 s19, s25, s1 // (op 876) 


s_cmp_eq_i32 s18, 0 // (op 876) 
s_mov_b32 s1, scc // (op 876) 

s_cmp_eq_i32 s1, 0 // (op 876) 
s_cselect_b32 s18, s19, s0 // (op 876) 



v_lshl_add_u32 v15, s18, 6, v12 // (op 876) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 876) 
v_mov_b32 v17, 0 // (op 876) convert
v_mov_b32 v16, v15 // (op 876) convert

s_lshr_b32 s18, s0, 1 // (op 876) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 876) 


v_mul_lo_u32 v25, v12, s27 // (op 876) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 876) most significant: high of low * low
v_mul_lo_u32 v24, v12, s26 // (op 876) least significant: low of low * low
v_add_u32 v25, v25, v13 // (op 876) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v24 // (op 876) least significant half
v_addc_co_u32 v13, s[0:1], v17, v25, s[0:1] // (op 876) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 876) 


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 876) 

v_mov_b32 v15, v16 // (op 876) convert

s_mul_i32 s1, s26, s31 // (op 878) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 878) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 878) least significant
s_mul_hi_u32 s19, s26, s30 // (op 878) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 878) most significant: sum
s_add_u32 s1, s1, s19 // (op 878) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 878) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 878) 

s_mov_b32 s52, s28 // (op 878) reg expression
s_mov_b32 s53, s29 // (op 878) reg expression
s_mov_b32 s54, s0 // (op 878) reg expression
s_mov_b32 s55, 131072 // (op 878) call()

s_mul_hi_i32 s0, s16, s10 // (op 944) 
v_and_b32 v12, -2, v1 // (op 944) 
s_add_i32 s1, s0, s16 // (op 944) 

v_sub_u32 v13, v1, v12 // (op 944) 

s_ashr_i32 s0, s1, 31 // (op 944) 
v_lshlrev_b32 v12, 4, v13 // (op 944) 

s_cmp_eq_i32 s10, 0 // (op 944) 
s_mov_b32 s18, scc // (op 944) 
v_lshrrev_b32 v13, 1, v1 // (op 944) 
s_cmp_eq_i32 s18, 0 // (op 944) 
s_cselect_b32 s19, 0, -1 // (op 944) 

s_and_b32 s18, s13, 31 // (op 944) 
s_lshl_b32 s22, 1, s18 // (op 944) 
s_add_i32 s23, s22, s19 // (op 944) 


s_and_b32 s19, s0, s23 // (op 944) 


s_add_i32 s0, s1, s19 // (op 944) 


s_ashr_i32 s1, s0, s18 // (op 944) 


s_lshl_b32 s0, s13, 24 // (op 944) 
s_ashr_i32 s18, s0, 31 // (op 944) 

s_xor_b32 s0, s1, s18 // (op 944) 

s_sub_i32 s1, s0, s18 // (op 944) 


s_mul_hi_i32 s0, s1, s14 // (op 944) 
s_add_i32 s18, s0, s1 // (op 944) 

s_ashr_i32 s0, s18, 31 // (op 944) 
s_cmp_eq_i32 s14, 0 // (op 944) 
s_mov_b32 s19, scc // (op 944) 
s_cmp_eq_i32 s19, 0 // (op 944) 
s_cselect_b32 s22, 0, -1 // (op 944) 

s_and_b32 s19, s15, 31 // (op 944) 
s_lshl_b32 s23, 1, s19 // (op 944) 
s_add_i32 s24, s23, s22 // (op 944) 


s_and_b32 s25, s0, s24 // (op 944) 


s_add_i32 s0, s18, s25 // (op 944) 


s_ashr_i32 s18, s0, s19 // (op 944) 


s_lshl_b32 s0, s15, 24 // (op 944) 
s_ashr_i32 s19, s0, 31 // (op 944) 

s_xor_b32 s0, s18, s19 // (op 944) 

s_sub_i32 s18, s0, s19 // (op 944) 


s_add_u32 s50, s6, 32 // (op 944) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 944) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 944) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 944) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 944) 

s_mul_i32 s0, s18, s24 // (op 944) 

s_sub_i32 s18, s1, s0 // (op 944) 


s_mul_hi_i32 s19, s16, s9 // (op 944) 
s_add_i32 s0, s19, s16 // (op 944) 

s_ashr_i32 s1, s0, 31 // (op 944) 
s_cmp_eq_i32 s9, 0 // (op 944) 
s_mov_b32 s19, scc // (op 944) 
s_cmp_eq_i32 s19, 0 // (op 944) 
s_cselect_b32 s22, 0, -1 // (op 944) 

s_and_b32 s19, s47, 31 // (op 944) 
s_lshl_b32 s23, 1, s19 // (op 944) 
s_add_i32 s49, s23, s22 // (op 944) 


s_and_b32 s22, s1, s49 // (op 944) 


s_add_i32 s1, s0, s22 // (op 944) 


s_ashr_i32 s0, s1, s19 // (op 944) 


s_lshl_b32 s1, s47, 24 // (op 944) 
s_ashr_i32 s19, s1, 31 // (op 944) 

s_xor_b32 s1, s0, s19 // (op 944) 

s_sub_i32 s0, s1, s19 // (op 944) 


s_mul_i32 s1, s11, s24 // (op 944) 

s_mul_i32 s19, s8, s1 // (op 944) 

s_mul_i32 s1, s0, s19 // (op 944) 

s_sub_u32 s19, s16, s1 // (op 944) 

s_mul_hi_i32 s1, s19, s12 // (op 944) 
s_add_i32 s22, s1, s19 // (op 944) 


s_ashr_i32 s1, s22, 31 // (op 944) 
s_cmp_eq_i32 s12, 0 // (op 944) 
s_mov_b32 s19, scc // (op 944) 
s_cmp_eq_i32 s19, 0 // (op 944) 
s_cselect_b32 s23, 0, -1 // (op 944) 

s_and_b32 s19, s46, 31 // (op 944) 
s_lshl_b32 s24, 1, s19 // (op 944) 
s_add_i32 s25, s24, s23 // (op 944) 


s_and_b32 s23, s1, s25 // (op 944) 


s_add_i32 s1, s22, s23 // (op 944) 


s_ashr_i32 s22, s1, s19 // (op 944) 


s_lshl_b32 s1, s46, 24 // (op 944) 
s_ashr_i32 s19, s1, 31 // (op 944) 

s_xor_b32 s1, s22, s19 // (op 944) 

s_sub_i32 s22, s1, s19 // (op 944) 


s_cmp_eq_i32 s0, 0 // (op 944) 
s_mov_b32 s1, scc // (op 944) 

s_cmp_eq_i32 s1, 0 // (op 944) 
s_cselect_b32 s0, s22, s18 // (op 944) 



v_lshl_add_u32 v16, s0, 5, v12 // (op 944) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 944) 
v_mov_b32 v25, 0 // (op 944) convert
v_mov_b32 v24, v16 // (op 944) convert

s_lshr_b32 s18, s0, 1 // (op 944) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 944) 


v_mul_lo_u32 v17, v12, s33 // (op 944) most significant: low of low * high
v_mul_hi_u32 v13, v12, s32 // (op 944) most significant: high of low * low
v_mul_lo_u32 v16, v12, s32 // (op 944) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 944) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v24 // (op 944) least significant half
v_addc_co_u32 v17, s[0:1], v17, v25, s[0:1] // (op 944) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 944) most significant: low of high * low
v_mul_hi_u32 v21, v16, 6 // (op 944) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 944) least significant: low of low * low
v_add_u32 v13, v13, v21 // (op 944) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 944) 

v_mov_b32 v12, v16 // (op 944) convert

s_mul_i32 s1, s31, 6 // (op 946) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 946) least significant
s_mul_hi_u32 s18, s30, 6 // (op 946) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 946) most significant: sum

s_mul_i32 s19, s32, s1 // (op 946) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 946) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 946) least significant
s_mul_hi_u32 s23, s32, s0 // (op 946) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 946) most significant: sum
s_add_u32 s19, s19, s23 // (op 946) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 946) 

s_mov_b32 s64, s34 // (op 946) reg expression
s_mov_b32 s65, s35 // (op 946) reg expression
s_mov_b32 s66, s0 // (op 946) reg expression
s_mov_b32 s67, 131072 // (op 946) call()

s_mul_hi_i32 s0, s16, s10 // (op 1008) 
v_and_b32 v13, -64, v1 // (op 1008) 
s_add_i32 s1, s0, s16 // (op 1008) 

v_sub_u32 v16, v1, v13 // (op 1008) 

s_ashr_i32 s0, s1, 31 // (op 1008) 
v_lshrrev_b32 v13, 6, v1 // (op 1008) 
s_cmp_eq_i32 s10, 0 // (op 1008) 
s_mov_b32 s18, scc // (op 1008) 
s_cmp_eq_i32 s18, 0 // (op 1008) 
s_cselect_b32 s19, 0, -1 // (op 1008) 

s_and_b32 s18, s13, 31 // (op 1008) 
s_lshl_b32 s22, 1, s18 // (op 1008) 
s_add_i32 s23, s22, s19 // (op 1008) 


s_and_b32 s19, s0, s23 // (op 1008) 


s_add_i32 s0, s1, s19 // (op 1008) 


s_ashr_i32 s1, s0, s18 // (op 1008) 


s_lshl_b32 s0, s13, 24 // (op 1008) 
s_ashr_i32 s18, s0, 31 // (op 1008) 

s_xor_b32 s0, s1, s18 // (op 1008) 

s_sub_i32 s1, s0, s18 // (op 1008) 


s_mul_hi_i32 s0, s1, s14 // (op 1008) 
s_add_i32 s18, s0, s1 // (op 1008) 

s_ashr_i32 s0, s18, 31 // (op 1008) 
s_cmp_eq_i32 s14, 0 // (op 1008) 
s_mov_b32 s19, scc // (op 1008) 
s_cmp_eq_i32 s19, 0 // (op 1008) 
s_cselect_b32 s22, 0, -1 // (op 1008) 

s_and_b32 s19, s15, 31 // (op 1008) 
s_lshl_b32 s23, 1, s19 // (op 1008) 
s_add_i32 s24, s23, s22 // (op 1008) 


s_and_b32 s25, s0, s24 // (op 1008) 


s_add_i32 s0, s18, s25 // (op 1008) 


s_ashr_i32 s18, s0, s19 // (op 1008) 


s_lshl_b32 s0, s15, 24 // (op 1008) 
s_ashr_i32 s19, s0, 31 // (op 1008) 

s_xor_b32 s0, s18, s19 // (op 1008) 

s_sub_i32 s18, s0, s19 // (op 1008) 


s_mul_i32 s0, s1, s11 // (op 1008) 

s_sub_u32 s1, s16, s0 // (op 1008) 

s_mul_i32 s0, s18, s11 // (op 1008) 
s_add_u32 s0, s0, s1 // (op 1008) 


s_mul_hi_i32 s1, s16, s9 // (op 1008) 
s_add_i32 s18, s1, s16 // (op 1008) 

s_ashr_i32 s1, s18, 31 // (op 1008) 
s_cmp_eq_i32 s9, 0 // (op 1008) 
s_mov_b32 s19, scc // (op 1008) 
s_cmp_eq_i32 s19, 0 // (op 1008) 
s_cselect_b32 s22, 0, -1 // (op 1008) 

s_and_b32 s19, s47, 31 // (op 1008) 
s_lshl_b32 s23, 1, s19 // (op 1008) 
s_add_i32 s24, s23, s22 // (op 1008) 


s_and_b32 s25, s1, s24 // (op 1008) 


s_add_i32 s1, s18, s25 // (op 1008) 


s_ashr_i32 s18, s1, s19 // (op 1008) 


s_lshl_b32 s1, s47, 24 // (op 1008) 
s_ashr_i32 s19, s1, 31 // (op 1008) 

s_xor_b32 s1, s18, s19 // (op 1008) 

s_sub_i32 s18, s1, s19 // (op 1008) 


s_add_u32 s50, s6, 32 // (op 1008) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1008) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1008) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1008) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1008) 

s_mul_i32 s1, s11, s24 // (op 1008) 

s_mul_i32 s19, s8, s1 // (op 1008) 

s_mul_i32 s1, s18, s19 // (op 1008) 

s_sub_u32 s19, s16, s1 // (op 1008) 

s_mov_b32 s1, s19 // (op 1008) 
s_mul_hi_i32 s22, s1, s12 // (op 1008) 
s_add_i32 s23, s22, s1 // (op 1008) 


s_ashr_i32 s1, s23, 31 // (op 1008) 
s_cmp_eq_i32 s12, 0 // (op 1008) 
s_mov_b32 s22, scc // (op 1008) 
s_cmp_eq_i32 s22, 0 // (op 1008) 
s_cselect_b32 s24, 0, -1 // (op 1008) 

s_and_b32 s22, s46, 31 // (op 1008) 
s_lshl_b32 s25, 1, s22 // (op 1008) 
s_add_i32 s49, s25, s24 // (op 1008) 


s_and_b32 s24, s1, s49 // (op 1008) 


s_add_i32 s1, s23, s24 // (op 1008) 


s_ashr_i32 s23, s1, s22 // (op 1008) 


s_lshl_b32 s1, s46, 24 // (op 1008) 
s_ashr_i32 s22, s1, 31 // (op 1008) 

s_xor_b32 s1, s23, s22 // (op 1008) 

s_sub_i32 s23, s1, s22 // (op 1008) 


s_mul_i32 s1, s48, s11 // (op 1008) 
s_add_u32 s24, s4, 64 // (op 1008) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1008) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1008) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1008) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1008) 

s_sub_i32 s22, s24, s1 // (op 1008) 

s_mul_i32 s24, s23, s22 // (op 1008) 


s_sub_u32 s25, s19, s24 // (op 1008) 


s_add_u32 s19, s25, s1 // (op 1008) 


s_cmp_eq_i32 s18, 0 // (op 1008) 
s_mov_b32 s1, scc // (op 1008) 

s_cmp_eq_i32 s1, 0 // (op 1008) 
s_cselect_b32 s18, s19, s0 // (op 1008) 



v_lshl_add_u32 v17, s18, 6, v16 // (op 1008) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1008) 
s_lshr_b32 s18, s0, 1 // (op 1008) 

v_lshl_add_u32 v16, s18, 3, v13 // (op 1008) 


v_mul_lo_u32 v13, v16, s36 // (op 1008) 

v_add_u32 v16, v17, v13 // (op 1008) 


s_mul_i32 s1, s36, s41 // (op 1010) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1010) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1010) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1010) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1010) most significant: sum
s_add_u32 s1, s1, s19 // (op 1010) most significant: sum


s_mov_b32 s68, s38 // (op 1010) reg expression
s_mov_b32 s69, s39 // (op 1010) reg expression
s_mov_b32 s70, s0 // (op 1010) reg expression
s_mov_b32 s71, 131072 // (op 1010) call()

s_mul_hi_i32 s0, s16, s10 // (op 1072) 
v_lshrrev_b32 v13, 6, v1 // (op 1072) 
s_add_i32 s1, s0, s16 // (op 1072) 

v_and_b32 v17, 1, v13 // (op 1072) 

s_ashr_i32 s0, s1, 31 // (op 1072) 
v_and_b32 v13, 15, v1 // (op 1072) 
s_cmp_eq_i32 s10, 0 // (op 1072) 
s_mov_b32 s18, scc // (op 1072) 
v_lshl_add_u32 v21, v17, 4, v13 // (op 1072) 


s_cmp_eq_i32 s18, 0 // (op 1072) 
s_cselect_b32 s19, 0, -1 // (op 1072) 

v_and_b32 v13, 63, v1 // (op 1072) 
s_and_b32 s18, s13, 31 // (op 1072) 
v_lshrrev_b32 v17, 4, v13 // (op 1072) 

s_lshl_b32 s22, 1, s18 // (op 1072) 
s_add_i32 s23, s22, s19 // (op 1072) 


s_and_b32 s19, s0, s23 // (op 1072) 


s_add_i32 s0, s1, s19 // (op 1072) 


s_ashr_i32 s1, s0, s18 // (op 1072) 


s_lshl_b32 s0, s13, 24 // (op 1072) 
s_ashr_i32 s18, s0, 31 // (op 1072) 

s_xor_b32 s0, s1, s18 // (op 1072) 

s_sub_i32 s1, s0, s18 // (op 1072) 


s_mul_hi_i32 s0, s1, s14 // (op 1072) 
s_add_i32 s18, s0, s1 // (op 1072) 

s_ashr_i32 s0, s18, 31 // (op 1072) 
s_cmp_eq_i32 s14, 0 // (op 1072) 
s_mov_b32 s19, scc // (op 1072) 
s_cmp_eq_i32 s19, 0 // (op 1072) 
s_cselect_b32 s22, 0, -1 // (op 1072) 

s_and_b32 s19, s15, 31 // (op 1072) 
s_lshl_b32 s23, 1, s19 // (op 1072) 
s_add_i32 s24, s23, s22 // (op 1072) 


s_and_b32 s25, s0, s24 // (op 1072) 


s_add_i32 s0, s18, s25 // (op 1072) 


s_ashr_i32 s18, s0, s19 // (op 1072) 


s_lshl_b32 s0, s15, 24 // (op 1072) 
s_ashr_i32 s19, s0, 31 // (op 1072) 

s_xor_b32 s0, s18, s19 // (op 1072) 

s_sub_i32 s18, s0, s19 // (op 1072) 


s_add_u32 s50, s6, 32 // (op 1072) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1072) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1072) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1072) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1072) 

s_mul_i32 s0, s18, s24 // (op 1072) 

s_sub_i32 s18, s1, s0 // (op 1072) 


s_mul_hi_i32 s19, s16, s9 // (op 1072) 
s_add_i32 s0, s19, s16 // (op 1072) 

s_ashr_i32 s1, s0, 31 // (op 1072) 
s_cmp_eq_i32 s9, 0 // (op 1072) 
s_mov_b32 s19, scc // (op 1072) 
s_cmp_eq_i32 s19, 0 // (op 1072) 
s_cselect_b32 s22, 0, -1 // (op 1072) 

s_and_b32 s19, s47, 31 // (op 1072) 
s_lshl_b32 s23, 1, s19 // (op 1072) 
s_add_i32 s49, s23, s22 // (op 1072) 


s_and_b32 s22, s1, s49 // (op 1072) 


s_add_i32 s1, s0, s22 // (op 1072) 


s_ashr_i32 s0, s1, s19 // (op 1072) 


s_lshl_b32 s1, s47, 24 // (op 1072) 
s_ashr_i32 s19, s1, 31 // (op 1072) 

s_xor_b32 s1, s0, s19 // (op 1072) 

s_sub_i32 s0, s1, s19 // (op 1072) 


s_mul_i32 s1, s11, s24 // (op 1072) 

s_mul_i32 s19, s8, s1 // (op 1072) 

s_mul_i32 s1, s0, s19 // (op 1072) 

s_sub_u32 s19, s16, s1 // (op 1072) 

s_mul_hi_i32 s1, s19, s12 // (op 1072) 
s_add_i32 s22, s1, s19 // (op 1072) 


s_ashr_i32 s1, s22, 31 // (op 1072) 
s_cmp_eq_i32 s12, 0 // (op 1072) 
s_mov_b32 s19, scc // (op 1072) 
s_cmp_eq_i32 s19, 0 // (op 1072) 
s_cselect_b32 s23, 0, -1 // (op 1072) 

s_and_b32 s19, s46, 31 // (op 1072) 
s_lshl_b32 s24, 1, s19 // (op 1072) 
s_add_i32 s25, s24, s23 // (op 1072) 


s_and_b32 s23, s1, s25 // (op 1072) 


s_add_i32 s1, s22, s23 // (op 1072) 


s_ashr_i32 s22, s1, s19 // (op 1072) 


s_lshl_b32 s1, s46, 24 // (op 1072) 
s_ashr_i32 s19, s1, 31 // (op 1072) 

s_xor_b32 s1, s22, s19 // (op 1072) 

s_sub_i32 s22, s1, s19 // (op 1072) 


s_cmp_eq_i32 s0, 0 // (op 1072) 
s_mov_b32 s1, scc // (op 1072) 

s_cmp_eq_i32 s1, 0 // (op 1072) 
s_cselect_b32 s0, s22, s18 // (op 1072) 



v_lshl_add_u32 v13, s0, 5, v21 // (op 1072) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1072) 
s_lshr_b32 s18, s0, 1 // (op 1072) 

v_lshl_add_u32 v21, s18, 3, v17 // (op 1072) 


v_mul_lo_u32 v17, v21, s42 // (op 1072) 
v_add_u32 v17, v17, v13 // (op 1072) 


s_mul_i32 s1, s42, s41 // (op 1074) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1074) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1074) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1074) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1074) most significant: sum
s_add_u32 s1, s1, s19 // (op 1074) most significant: sum


s_mov_b32 s72, s44 // (op 1074) reg expression
s_mov_b32 s73, s45 // (op 1074) reg expression
s_mov_b32 s74, s0 // (op 1074) reg expression
s_mov_b32 s75, 131072 // (op 1074) call()

v_and_b32 v13, -2, v1 // (op 1165) 
v_sub_u32 v21, v1, v13 // (op 1165) 

v_lshlrev_b32 v13, 5, v21 // (op 1165) 

v_lshrrev_b32 v21, 1, v1 // (op 1165) 
v_lshl_add_u32 v22, v21, 6, v13 // (op 1165) 


v_lshlrev_b32 v13, 2, v22 // (op 1165) 

v_lshrrev_b32 v21, 3, v13 // (op 1165) 

v_and_b32 v13, -2, v1 // (op 1223) 
v_sub_u32 v22, v1, v13 // (op 1223) 

v_lshlrev_b32 v13, 4, v22 // (op 1223) 

v_lshrrev_b32 v22, 1, v1 // (op 1223) 
v_lshl_add_u32 v24, v22, 5, v13 // (op 1223) 


v_mul_lo_u32 v13, v24, 6 // (op 1223) 
v_lshrrev_b32 v22, 3, v13 // (op 1223) 

v_lshrrev_b32 v13, 4, v24 // (op 1223) 

v_lshlrev_b32 v24, 2, v13 // (op 1223) 

v_add_u32 v13, v22, v24 // (op 1223) 


v_and_b32 v22, -64, v1 // (op 1276) 
v_sub_u32 v24, v1, v22 // (op 1276) 

v_and_b32 v22, 4294967232, v1 // (op 1276) 
v_add_u32 v25, v22, v24 // (op 1276) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 316) 
s_and_b32 s18, s0, 4294967294 // (op 316) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 315) 
s_mov_b32 s19, s0 // (op 315) convert

s_cmp_lt_u32 s18, s19 // (op 315) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_315:


s_add_u32 s18, s18, 1 // (op 317) 
v_mov_b32 v14, v15 // (op 166) 
v_mov_b32 v18, v14 // (op 166) reg expression

buffer_load_dwordx4 v[26:29], v18, s[52:55], 0 offen // (op 166) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 866) 
s_lshl_b64 s[50:51], s[0:1], 2 // (op 866) 

s_lshr_b64 s[0:1], s[50:51], 3 // (op 866) 

v_add_u32 v15, v15, s0 // (op 866) 



s_barrier  // (op 1456) 
v_mov_b32 v14, v12 // (op 179) 



v_mov_b32 v18, v14 // (op 179) reg expression

buffer_load_dwordx3 v[22:24], v18, s[64:67], 0 offen // (op 179) Load value


s_lshl_b64 s[0:1], s[32:33], 7 // (op 936) 
s_mul_i32 s51, s1, 6 // (op 936) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 936) least significant
s_mul_hi_u32 s19, s0, 6 // (op 936) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 936) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 936) 

v_add_u32 v12, v12, s0 // (op 936) 

s_waitcnt vmcnt(1)
ds_write_b128 v21, v[26:29] offset:4096 // (op 167) Store local data 



v_mov_b32 v14, v16 // (op 180) 
s_barrier  // (op 1469) 
v_mov_b32 v18, v14 // (op 180) reg expression




buffer_load_ubyte v19, v18, s[68:71], 0 offen offset: 0 // (op 180) Load value


s_lshl_b32 s19, s36, 2 // (op 1000) 
v_add_u32 v16, v16, s19 // (op 1000) 

s_waitcnt vmcnt(1)
ds_write_b96 v13, v[22:24] // (op 177) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1459) 



s_barrier  // (op 1483) 
ds_read_b64_tr_b4 v[26:27], v7 offset:4096 // (op 169) Transpose load local data 



ds_read_b64_tr_b4 v[28:29], v7 offset:4608 // (op 169) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v19 offset:8192 // (op 178) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1473) 



ds_read_b96_tr_b6 v[30:32], v0 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v0 offset:512 // (op 171) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1487) 


ds_read_u8 v14, v4 offset:8192 // (op 173) Load local data 

v_mov_b64 v[18:19], s[42:43] // (op 175) convert

v_mov_b32 v20, v17 // (op 175) 
v_mov_b32 v22, v20 // (op 175) reg expression
v_mov_b32 v23, v22 // (op 175) 
buffer_load_ubyte v24, v23, s[72:75], 0 offen offset: 0 // (op 175) Load value




s_lshl_b32 s19, s42, 2 // (op 1064) 
v_add_u32 v17, v17, s19 // (op 1064) 




s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:35], a[0:3], v14, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 176) 
ds_read_b64_tr_b4 v[36:37], v7 offset:4112 // (op 321) Transpose load local data 


ds_read_b64_tr_b4 v[38:39], v7 offset:4624 // (op 321) Transpose load local data 


ds_read_u8 v14, v4 offset:8224 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:39], v[30:35], a[4:7], v14, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 327) 




s_ashr_i64 s[0:1], s[30:31], 7 // (op 315) 
s_mov_b32 s19, s0 // (op 315) convert

s_cmp_lt_u32 s18, s19 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_315 if true)
RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_315:






















s_barrier  // (op 1565) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_123:



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
RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.LRR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end:
.size RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_, .LRR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end-RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 68
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
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
    DataType: F4
    DataTypeA: F4
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
  - .name: RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 16896
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

