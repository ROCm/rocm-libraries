// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_,@function
RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1160) 
s_mul_hi_i32 s1, s16, s10 // (op 1348) 
v_and_b32 v0, 48, v1 // (op 1274) 
v_and_b32 v3, 63, v1 // (op 1274) 
v_sub_u32 v4, v3, v0 // (op 1274) 

v_lshrrev_b32 v3, 1, v4 // (op 1274) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1274) 


v_and_b32 v0, 1, v4 // (op 1274) 

v_lshlrev_b32 v3, 3, v0 // (op 1274) 

v_add_u32 v0, v5, v3 // (op 1274) 


v_and_b32 v3, 48, v1 // (op 1418) 
v_and_b32 v4, 63, v1 // (op 1418) 
v_sub_u32 v5, v4, v3 // (op 1418) 

v_lshrrev_b32 v4, 1, v5 // (op 1418) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 1418) 


v_and_b32 v3, 1, v5 // (op 1418) 

v_lshlrev_b32 v4, 3, v3 // (op 1418) 

v_add_u32 v3, v6, v4 // (op 1418) 


v_lshrrev_b32 v4, 6, v1 // (op 1160) 
s_add_i32 s18, s0, s16 // (op 1160) 

v_and_b32 v5, 63, v1 // (op 1160) 
s_ashr_i32 s0, s18, 31 // (op 1160) 
v_lshrrev_b32 v6, 4, v5 // (op 1160) 
s_cmp_eq_i32 s10, 0 // (op 1160) 
s_mov_b32 s19, scc // (op 1160) 
v_lshlrev_b32 v7, 2, v6 // (op 1160) 

s_cmp_eq_i32 s19, 0 // (op 1160) 
s_cselect_b32 s51, 0, -1 // (op 1160) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1160) 


s_and_b32 s19, s13, 31 // (op 1160) 
v_lshrrev_b32 v4, 2, v5 // (op 1160) 

v_and_b32 v5, 3, v4 // (op 1160) 

v_and_b32 v4, 63, v1 // (op 1222) 
v_and_b32 v7, 48, v1 // (op 1222) 
v_sub_u32 v8, v4, v7 // (op 1222) 

v_lshrrev_b32 v7, 4, v4 // (op 1222) 

v_lshl_add_u32 v4, v7, 5, v8 // (op 1222) 


v_lshrrev_b32 v7, 6, v1 // (op 1222) 
v_lshlrev_b32 v8, 4, v7 // (op 1222) 

v_lshl_add_u32 v7, v4, 6, v8 // (op 1222) 


v_mul_lo_u32 v4, v7, 6 // (op 1222) 
v_lshrrev_b32 v8, 3, v4 // (op 1222) 

v_lshrrev_b32 v4, 4, v7 // (op 1222) 

v_lshlrev_b32 v7, 2, v4 // (op 1222) 

v_add_u32 v4, v8, v7 // (op 1222) 


v_lshrrev_b32 v7, 6, v1 // (op 1332) 
v_and_b32 v8, 15, v1 // (op 1332) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 1332) 


v_and_b32 v7, 63, v1 // (op 1332) 
v_lshrrev_b32 v8, 4, v7 // (op 1332) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 1332) 


v_lshrrev_b32 v8, 6, v1 // (op 1348) 
v_and_b32 v9, 63, v1 // (op 1348) 
v_and_b32 v10, 63, v1 // (op 1379) 
v_and_b32 v11, 48, v1 // (op 1379) 
v_sub_u32 v12, v10, v11 // (op 1379) 

v_lshrrev_b32 v11, 4, v10 // (op 1379) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 1379) 


v_lshrrev_b32 v11, 6, v1 // (op 1379) 
v_lshlrev_b32 v12, 4, v11 // (op 1379) 

v_lshl_add_u32 v11, v10, 6, v12 // (op 1379) 


v_mul_lo_u32 v10, v11, 6 // (op 1379) 
v_lshrrev_b32 v12, 3, v10 // (op 1379) 

v_lshrrev_b32 v10, 4, v11 // (op 1379) 

v_lshlrev_b32 v11, 2, v10 // (op 1379) 

v_add_u32 v10, v12, v11 // (op 1379) 


v_lshrrev_b32 v11, 6, v1 // (op 1447) 
v_and_b32 v12, 15, v1 // (op 1447) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1447) 


v_and_b32 v11, 63, v1 // (op 1447) 
v_lshrrev_b32 v12, 4, v11 // (op 1447) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 1447) 


s_lshl_b32 s52, 1, s19 // (op 1160) 
v_and_b32 v12, 3, v1 // (op 1160) 
s_add_i32 s53, s52, s51 // (op 1160) 


s_and_b32 s51, s0, s53 // (op 1160) 


s_add_i32 s0, s1, s16 // (op 1348) 

s_ashr_i32 s1, s0, 31 // (op 1348) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1160) 


s_add_i32 s52, s18, s51 // (op 1160) 


s_ashr_i32 s18, s52, s19 // (op 1160) 


s_lshl_b32 s19, s13, 24 // (op 1160) 
s_ashr_i32 s51, s19, 31 // (op 1160) 

s_xor_b32 s19, s18, s51 // (op 1160) 

s_sub_i32 s18, s19, s51 // (op 1160) 


s_mul_hi_i32 s19, s18, s14 // (op 1160) 
s_add_i32 s51, s19, s18 // (op 1160) 

s_ashr_i32 s19, s51, 31 // (op 1160) 
s_cmp_eq_i32 s14, 0 // (op 1160) 
s_mov_b32 s52, scc // (op 1160) 
s_cmp_eq_i32 s52, 0 // (op 1160) 
s_cselect_b32 s53, 0, -1 // (op 1160) 

s_and_b32 s52, s15, 31 // (op 1160) 
v_lshrrev_b32 v5, 4, v9 // (op 1348) 
s_cmp_eq_i32 s10, 0 // (op 1348) 
s_mov_b32 s54, scc // (op 1348) 
v_lshlrev_b32 v12, 2, v5 // (op 1348) 

s_cmp_eq_i32 s54, 0 // (op 1348) 
s_cselect_b32 s55, 0, -1 // (op 1348) 

s_lshl_b32 s54, 1, s52 // (op 1160) 
v_lshl_add_u32 v5, v8, 4, v12 // (op 1348) 


v_lshrrev_b32 v8, 2, v9 // (op 1348) 

s_add_i32 s56, s54, s53 // (op 1160) 


s_and_b32 s53, s19, s56 // (op 1160) 


s_add_i32 s19, s51, s53 // (op 1160) 


s_ashr_i32 s51, s19, s52 // (op 1160) 


s_lshl_b32 s19, s15, 24 // (op 1160) 
s_ashr_i32 s52, s19, 31 // (op 1160) 

s_xor_b32 s19, s51, s52 // (op 1160) 

s_sub_i32 s51, s19, s52 // (op 1160) 


s_mul_i32 s19, s18, s11 // (op 1160) 
s_sub_u32 s52, s16, s19 // (op 1160) 

s_mul_i32 s19, s51, s11 // (op 1160) 
s_add_u32 s19, s19, s52 // (op 1160) 

s_mul_hi_i32 s52, s16, s9 // (op 1160) 
s_add_i32 s53, s52, s16 // (op 1160) 

s_ashr_i32 s52, s53, 31 // (op 1160) 
s_cmp_eq_i32 s9, 0 // (op 1160) 
s_mov_b32 s54, scc // (op 1160) 
s_cmp_eq_i32 s54, 0 // (op 1160) 
s_cselect_b32 s56, 0, -1 // (op 1160) 

s_and_b32 s54, s13, 31 // (op 1348) 
v_and_b32 v9, 3, v8 // (op 1348) 

s_lshl_b32 s57, 1, s54 // (op 1348) 
v_and_b32 v8, 3, v1 // (op 1348) 
s_add_i32 s58, s57, s55 // (op 1348) 


s_and_b32 s55, s1, s58 // (op 1348) 


v_lshl_add_u32 v12, v9, 2, v8 // (op 1348) 


s_add_i32 s1, s0, s55 // (op 1348) 


s_ashr_i32 s0, s1, s54 // (op 1348) 


s_lshl_b32 s1, s13, 24 // (op 1348) 
s_ashr_i32 s54, s1, 31 // (op 1348) 

s_xor_b32 s1, s0, s54 // (op 1348) 

s_sub_i32 s0, s1, s54 // (op 1348) 


s_mul_hi_i32 s1, s0, s14 // (op 1348) 
s_add_i32 s54, s1, s0 // (op 1348) 

s_ashr_i32 s1, s54, 31 // (op 1348) 
s_cmp_eq_i32 s14, 0 // (op 1348) 
s_mov_b32 s55, scc // (op 1348) 
s_cmp_eq_i32 s55, 0 // (op 1348) 
s_cselect_b32 s57, 0, -1 // (op 1348) 

s_and_b32 s55, s15, 31 // (op 1348) 
s_lshl_b32 s58, 1, s55 // (op 1348) 
s_add_i32 s59, s58, s57 // (op 1348) 


s_and_b32 s57, s1, s59 // (op 1348) 


s_add_i32 s1, s54, s57 // (op 1348) 


s_ashr_i32 s54, s1, s55 // (op 1348) 


s_lshl_b32 s1, s15, 24 // (op 1348) 
s_ashr_i32 s55, s1, 31 // (op 1348) 

s_xor_b32 s1, s54, s55 // (op 1348) 

s_sub_i32 s54, s1, s55 // (op 1348) 


s_mul_i32 s1, s0, s11 // (op 1348) 
s_sub_u32 s55, s16, s1 // (op 1348) 

s_mul_i32 s1, s54, s11 // (op 1348) 
s_add_u32 s1, s1, s55 // (op 1348) 

s_mul_hi_i32 s55, s16, s9 // (op 1348) 
s_add_i32 s57, s55, s16 // (op 1348) 

s_ashr_i32 s55, s57, 31 // (op 1348) 
s_cmp_eq_i32 s9, 0 // (op 1348) 
s_mov_b32 s58, scc // (op 1348) 
s_cmp_eq_i32 s58, 0 // (op 1348) 
s_cselect_b32 s59, 0, -1 // (op 1348) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s47, 31 // (op 1160) 
s_lshl_b32 s60, 1, s58 // (op 1160) 
s_add_i32 s61, s60, s56 // (op 1160) 


s_and_b32 s56, s52, s61 // (op 1160) 


s_add_i32 s52, s53, s56 // (op 1160) 


s_ashr_i32 s53, s52, s58 // (op 1160) 


s_lshl_b32 s52, s47, 24 // (op 1160) 
s_ashr_i32 s56, s52, 31 // (op 1160) 

s_xor_b32 s52, s53, s56 // (op 1160) 

s_sub_i32 s53, s52, s56 // (op 1160) 


s_and_b32 s52, s47, 31 // (op 1348) 
s_lshl_b32 s56, 1, s52 // (op 1348) 
s_add_i32 s58, s56, s59 // (op 1348) 


s_and_b32 s56, s55, s58 // (op 1348) 


s_add_i32 s55, s57, s56 // (op 1348) 


s_ashr_i32 s56, s55, s52 // (op 1348) 


s_lshl_b32 s52, s47, 24 // (op 1348) 
s_ashr_i32 s55, s52, 31 // (op 1348) 

s_xor_b32 s52, s56, s55 // (op 1348) 

s_sub_i32 s56, s52, s55 // (op 1348) 


s_add_u32 s58, s6, 16 // (op 1160) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1160) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1160) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1160) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1160) 

s_mul_i32 s52, s11, s58 // (op 1160) 
s_mul_i32 s55, s8, s52 // (op 1160) 

s_mul_i32 s52, s53, s55 // (op 1160) 

s_sub_u32 s55, s16, s52 // (op 1160) 

s_mov_b32 s52, s55 // (op 1160) 
s_mul_hi_i32 s57, s52, s12 // (op 1160) 
s_add_i32 s60, s57, s52 // (op 1160) 


s_ashr_i32 s52, s60, 31 // (op 1160) 
s_cmp_eq_i32 s12, 0 // (op 1160) 
s_mov_b32 s57, scc // (op 1160) 
s_cmp_eq_i32 s57, 0 // (op 1160) 
s_cselect_b32 s61, 0, -1 // (op 1160) 

s_and_b32 s57, s46, 31 // (op 1160) 
s_lshl_b32 s62, 1, s57 // (op 1160) 
s_add_i32 s63, s62, s61 // (op 1160) 


s_and_b32 s61, s52, s63 // (op 1160) 


s_add_i32 s52, s60, s61 // (op 1160) 


s_ashr_i32 s60, s52, s57 // (op 1160) 


s_lshl_b32 s52, s46, 24 // (op 1160) 
s_ashr_i32 s57, s52, 31 // (op 1160) 

s_xor_b32 s52, s60, s57 // (op 1160) 

s_sub_i32 s60, s52, s57 // (op 1160) 


s_mul_i32 s52, s48, s11 // (op 1160) 
s_add_u32 s62, s4, 64 // (op 1160) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 1160) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1160) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1160) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 1160) 

s_sub_i32 s57, s62, s52 // (op 1160) 

s_mul_i32 s61, s60, s57 // (op 1160) 

s_sub_u32 s57, s55, s61 // (op 1160) 


s_add_u32 s55, s57, s52 // (op 1160) 


s_cmp_eq_i32 s53, 0 // (op 1160) 
s_mov_b32 s52, scc // (op 1160) 

s_cmp_eq_i32 s52, 0 // (op 1160) 
s_cselect_b32 s53, s55, s19 // (op 1160) 


s_mul_i32 s19, s51, s58 // (op 1160) 


v_lshl_add_u32 v8, s53, 6, v6 // (op 1160) 


s_sub_i32 s51, s18, s19 // (op 1160) 


s_cmp_eq_i32 s52, 0 // (op 1160) 
s_cselect_b32 s53, s60, s51 // (op 1160) 



v_lshl_add_u32 v6, s53, 4, v13 // (op 1160) 


v_mul_lo_u32 v9, v6, s2 // (op 1160) 

v_add_lshl_u32 v6, v8, v9, 2 // (op 1160) 


s_add_u32 s18, s6, 16 // (op 1348) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1348) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 1348) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 1348) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 1348) 

s_mul_i32 s55, s11, s18 // (op 1348) 
s_mul_i32 s51, s8, s55 // (op 1348) 

s_mul_i32 s55, s56, s51 // (op 1348) 

s_sub_u32 s51, s16, s55 // (op 1348) 

s_mov_b32 s55, s51 // (op 1348) 
s_mul_hi_i32 s52, s55, s12 // (op 1348) 
s_add_i32 s53, s52, s55 // (op 1348) 


s_ashr_i32 s52, s53, 31 // (op 1348) 
s_cmp_eq_i32 s12, 0 // (op 1348) 
s_mov_b32 s55, scc // (op 1348) 
s_cmp_eq_i32 s55, 0 // (op 1348) 
s_cselect_b32 s57, 0, -1 // (op 1348) 

s_and_b32 s55, s46, 31 // (op 1348) 
s_lshl_b32 s58, 1, s55 // (op 1348) 
s_add_i32 s59, s58, s57 // (op 1348) 


s_and_b32 s57, s52, s59 // (op 1348) 


s_add_i32 s52, s53, s57 // (op 1348) 


s_ashr_i32 s53, s52, s55 // (op 1348) 


s_lshl_b32 s52, s46, 24 // (op 1348) 
s_ashr_i32 s55, s52, 31 // (op 1348) 

s_xor_b32 s52, s53, s55 // (op 1348) 

s_sub_i32 s53, s52, s55 // (op 1348) 


s_mul_i32 s52, s48, s11 // (op 1348) 
s_add_u32 s58, s4, 64 // (op 1348) Start of Int64 add, locking SCC
s_addc_u32 s59, s5, 0 // (op 1348) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1348) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1348) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 1348) 

s_sub_i32 s55, s58, s52 // (op 1348) 

s_mul_i32 s57, s53, s55 // (op 1348) 

s_sub_u32 s55, s51, s57 // (op 1348) 


s_add_u32 s51, s55, s52 // (op 1348) 


s_cmp_eq_i32 s56, 0 // (op 1348) 
s_mov_b32 s52, scc // (op 1348) 

s_cmp_eq_i32 s52, 0 // (op 1348) 
s_cselect_b32 s55, s51, s1 // (op 1348) 


s_mul_i32 s1, s54, s18 // (op 1348) 


v_lshl_add_u32 v8, s55, 6, v5 // (op 1348) 


s_sub_i32 s51, s0, s1 // (op 1348) 


s_cmp_eq_i32 s52, 0 // (op 1348) 
s_cselect_b32 s0, s53, s51 // (op 1348) 



v_lshl_add_u32 v5, s0, 4, v12 // (op 1348) 


v_mul_lo_u32 v9, v5, s20 // (op 1348) 

v_add_lshl_u32 v5, v8, v9, 2 // (op 1348) 


s_mul_i32 s1, s2, s7 // (op 1162) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1162) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1162) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1162) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1162) most significant: sum
s_add_u32 s1, s1, s19 // (op 1162) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1162) 

s_mul_i32 s1, s20, s7 // (op 1350) most significant: low of low * high
s_mul_i32 s51, s21, s6 // (op 1350) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 1350) least significant
s_mul_hi_u32 s52, s20, s6 // (op 1350) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1350) most significant: sum
s_add_u32 s1, s1, s52 // (op 1350) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 1350) 

s_mov_b32 s56, s22 // (op 1162) reg expression
s_mov_b32 s57, s23 // (op 1162) reg expression
s_mov_b32 s58, s18 // (op 1162) reg expression
s_mov_b32 s59, 131072 // (op 1162) call()

s_mov_b32 s60, s24 // (op 1350) reg expression

s_mov_b32 s61, s25 // (op 1350) reg expression
s_mov_b32 s62, s52 // (op 1350) reg expression
s_mov_b32 s63, 131072 // (op 1350) call()


v_mov_b32 v8, s49 // (op 2) Move value
v_mov_b32 v9, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 309) call()
v_accvgpr_write a5, 0.00000 // (op 309) call()
v_accvgpr_write a6, 0.00000 // (op 309) call()
v_accvgpr_write a7, 0.00000 // (op 309) call()
s_mul_hi_i32 s0, s16, s10 // (op 927) 
s_add_i32 s1, s0, s16 // (op 927) 

s_ashr_i32 s0, s1, 31 // (op 927) 
s_cmp_eq_i32 s10, 0 // (op 927) 
s_mov_b32 s18, scc // (op 927) 
s_cmp_eq_i32 s18, 0 // (op 927) 
s_cselect_b32 s19, 0, -1 // (op 927) 

s_and_b32 s18, s13, 31 // (op 927) 
s_lshl_b32 s22, 1, s18 // (op 927) 
s_add_i32 s23, s22, s19 // (op 927) 


s_and_b32 s19, s0, s23 // (op 927) 


s_add_i32 s0, s1, s19 // (op 927) 


s_ashr_i32 s1, s0, s18 // (op 927) 


s_lshl_b32 s0, s13, 24 // (op 927) 
s_ashr_i32 s18, s0, 31 // (op 927) 

s_xor_b32 s0, s1, s18 // (op 927) 

s_sub_i32 s1, s0, s18 // (op 927) 


s_mul_hi_i32 s0, s1, s14 // (op 927) 
s_add_i32 s18, s0, s1 // (op 927) 

s_ashr_i32 s0, s18, 31 // (op 927) 
s_cmp_eq_i32 s14, 0 // (op 927) 
s_mov_b32 s19, scc // (op 927) 
s_cmp_eq_i32 s19, 0 // (op 927) 
s_cselect_b32 s22, 0, -1 // (op 927) 

s_and_b32 s19, s15, 31 // (op 927) 
s_lshl_b32 s23, 1, s19 // (op 927) 
s_add_i32 s24, s23, s22 // (op 927) 


s_and_b32 s25, s0, s24 // (op 927) 


s_add_i32 s0, s18, s25 // (op 927) 


s_ashr_i32 s18, s0, s19 // (op 927) 


s_lshl_b32 s0, s15, 24 // (op 927) 
s_ashr_i32 s19, s0, 31 // (op 927) 

s_xor_b32 s0, s18, s19 // (op 927) 

s_sub_i32 s18, s0, s19 // (op 927) 


s_mul_i32 s0, s1, s11 // (op 927) 

s_sub_u32 s1, s16, s0 // (op 927) 

s_mul_i32 s0, s18, s11 // (op 927) 
s_add_u32 s0, s0, s1 // (op 927) 


s_mul_hi_i32 s1, s16, s9 // (op 927) 
s_add_i32 s18, s1, s16 // (op 927) 

s_ashr_i32 s1, s18, 31 // (op 927) 
s_cmp_eq_i32 s9, 0 // (op 927) 
s_mov_b32 s19, scc // (op 927) 
s_cmp_eq_i32 s19, 0 // (op 927) 
s_cselect_b32 s22, 0, -1 // (op 927) 

s_and_b32 s19, s47, 31 // (op 927) 
s_lshl_b32 s23, 1, s19 // (op 927) 
s_add_i32 s24, s23, s22 // (op 927) 


s_and_b32 s25, s1, s24 // (op 927) 


s_add_i32 s1, s18, s25 // (op 927) 


s_ashr_i32 s18, s1, s19 // (op 927) 


s_lshl_b32 s1, s47, 24 // (op 927) 
s_ashr_i32 s19, s1, 31 // (op 927) 

s_xor_b32 s1, s18, s19 // (op 927) 

s_sub_i32 s18, s1, s19 // (op 927) 


s_add_u32 s22, s6, 16 // (op 927) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 927) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 927) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 927) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 927) 

s_mul_i32 s1, s11, s22 // (op 927) 

s_mul_i32 s19, s8, s1 // (op 927) 

s_mul_i32 s1, s18, s19 // (op 927) 

s_sub_u32 s19, s16, s1 // (op 927) 

s_mov_b32 s1, s19 // (op 927) 
s_mul_hi_i32 s22, s1, s12 // (op 927) 
s_add_i32 s23, s22, s1 // (op 927) 


s_ashr_i32 s1, s23, 31 // (op 927) 
s_cmp_eq_i32 s12, 0 // (op 927) 
s_mov_b32 s22, scc // (op 927) 
s_cmp_eq_i32 s22, 0 // (op 927) 
s_cselect_b32 s24, 0, -1 // (op 927) 

s_and_b32 s22, s46, 31 // (op 927) 
s_lshl_b32 s25, 1, s22 // (op 927) 
s_add_i32 s49, s25, s24 // (op 927) 


s_and_b32 s24, s1, s49 // (op 927) 


s_add_i32 s1, s23, s24 // (op 927) 


s_ashr_i32 s23, s1, s22 // (op 927) 


s_lshl_b32 s1, s46, 24 // (op 927) 
s_ashr_i32 s22, s1, 31 // (op 927) 

s_xor_b32 s1, s23, s22 // (op 927) 

s_sub_i32 s23, s1, s22 // (op 927) 


s_mul_i32 s1, s48, s11 // (op 927) 
s_add_u32 s24, s4, 64 // (op 927) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 927) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 927) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 927) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 927) 

s_sub_i32 s22, s24, s1 // (op 927) 

s_mul_i32 s24, s23, s22 // (op 927) 


s_sub_u32 s25, s19, s24 // (op 927) 


s_add_u32 s19, s25, s1 // (op 927) 


s_cmp_eq_i32 s18, 0 // (op 927) 
s_mov_b32 s1, scc // (op 927) 

s_cmp_eq_i32 s1, 0 // (op 927) 
s_cselect_b32 s18, s19, s0 // (op 927) 



s_lshl_b32 s19, s18, 6 // (op 927) 

s_mov_b32 s1, 0 // (op 927) convert
s_mov_b32 s0, s19 // (op 927) convert

v_mul_lo_u32 v13, v1, s27 // (op 927) most significant: low of low * high
v_mul_hi_u32 v14, v1, s26 // (op 927) most significant: high of low * low
v_mul_lo_u32 v12, v1, s26 // (op 927) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 927) most significant: sum

v_mov_b32 v14, s0 // (op 927) 
v_mov_b32 v15, s1 // (op 927) 
v_add_co_u32 v16, s[18:19], v14, v12 // (op 927) least significant half
v_addc_co_u32 v17, s[18:19], v15, v13, s[18:19] // (op 927) most significant half





v_mul_lo_u32 v13, v17, 6 // (op 927) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 927) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 927) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 927) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 927) 

v_mov_b32 v12, v14 // (op 927) convert

s_mul_i32 s1, s31, 6 // (op 929) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 929) least significant
s_mul_hi_u32 s18, s30, 6 // (op 929) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 929) most significant: sum

s_mul_i32 s19, s26, s1 // (op 929) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 929) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 929) least significant
s_mul_hi_u32 s23, s26, s0 // (op 929) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 929) most significant: sum
s_add_u32 s19, s19, s23 // (op 929) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 929) 

s_mov_b32 s52, s28 // (op 929) reg expression
s_mov_b32 s53, s29 // (op 929) reg expression
s_mov_b32 s54, s0 // (op 929) reg expression
s_mov_b32 s55, 131072 // (op 929) call()

s_mul_hi_i32 s0, s16, s10 // (op 946) 
s_add_i32 s1, s0, s16 // (op 946) 

s_ashr_i32 s0, s1, 31 // (op 946) 
s_cmp_eq_i32 s10, 0 // (op 946) 
s_mov_b32 s18, scc // (op 946) 
s_cmp_eq_i32 s18, 0 // (op 946) 
s_cselect_b32 s19, 0, -1 // (op 946) 

s_and_b32 s18, s13, 31 // (op 946) 
s_lshl_b32 s22, 1, s18 // (op 946) 
s_add_i32 s23, s22, s19 // (op 946) 


s_and_b32 s19, s0, s23 // (op 946) 


s_add_i32 s0, s1, s19 // (op 946) 


s_ashr_i32 s1, s0, s18 // (op 946) 


s_lshl_b32 s0, s13, 24 // (op 946) 
s_ashr_i32 s18, s0, 31 // (op 946) 

s_xor_b32 s0, s1, s18 // (op 946) 

s_sub_i32 s1, s0, s18 // (op 946) 


s_mul_hi_i32 s0, s1, s14 // (op 946) 
s_add_i32 s18, s0, s1 // (op 946) 

s_ashr_i32 s0, s18, 31 // (op 946) 
s_cmp_eq_i32 s14, 0 // (op 946) 
s_mov_b32 s19, scc // (op 946) 
s_cmp_eq_i32 s19, 0 // (op 946) 
s_cselect_b32 s22, 0, -1 // (op 946) 

s_and_b32 s19, s15, 31 // (op 946) 
s_lshl_b32 s23, 1, s19 // (op 946) 
s_add_i32 s24, s23, s22 // (op 946) 


s_and_b32 s25, s0, s24 // (op 946) 


s_add_i32 s0, s18, s25 // (op 946) 


s_ashr_i32 s18, s0, s19 // (op 946) 


s_lshl_b32 s0, s15, 24 // (op 946) 
s_ashr_i32 s19, s0, 31 // (op 946) 

s_xor_b32 s0, s18, s19 // (op 946) 

s_sub_i32 s18, s0, s19 // (op 946) 


s_mul_i32 s0, s1, s11 // (op 946) 

s_sub_u32 s1, s16, s0 // (op 946) 

s_mul_i32 s0, s18, s11 // (op 946) 
s_add_u32 s0, s0, s1 // (op 946) 


s_mul_hi_i32 s1, s16, s9 // (op 946) 
s_add_i32 s18, s1, s16 // (op 946) 

s_ashr_i32 s1, s18, 31 // (op 946) 
s_cmp_eq_i32 s9, 0 // (op 946) 
s_mov_b32 s19, scc // (op 946) 
s_cmp_eq_i32 s19, 0 // (op 946) 
s_cselect_b32 s22, 0, -1 // (op 946) 

s_and_b32 s19, s47, 31 // (op 946) 
s_lshl_b32 s23, 1, s19 // (op 946) 
s_add_i32 s24, s23, s22 // (op 946) 


s_and_b32 s25, s1, s24 // (op 946) 


s_add_i32 s1, s18, s25 // (op 946) 


s_ashr_i32 s18, s1, s19 // (op 946) 


s_lshl_b32 s1, s47, 24 // (op 946) 
s_ashr_i32 s19, s1, 31 // (op 946) 

s_xor_b32 s1, s18, s19 // (op 946) 

s_sub_i32 s18, s1, s19 // (op 946) 


s_add_u32 s50, s6, 16 // (op 946) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 946) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 946) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 946) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 946) 

s_mul_i32 s1, s11, s24 // (op 946) 

s_mul_i32 s19, s8, s1 // (op 946) 

s_mul_i32 s1, s18, s19 // (op 946) 

s_sub_u32 s19, s16, s1 // (op 946) 

s_mov_b32 s1, s19 // (op 946) 
s_mul_hi_i32 s22, s1, s12 // (op 946) 
s_add_i32 s23, s22, s1 // (op 946) 


s_ashr_i32 s1, s23, 31 // (op 946) 
s_cmp_eq_i32 s12, 0 // (op 946) 
s_mov_b32 s22, scc // (op 946) 
s_cmp_eq_i32 s22, 0 // (op 946) 
s_cselect_b32 s24, 0, -1 // (op 946) 

s_and_b32 s22, s46, 31 // (op 946) 
s_lshl_b32 s25, 1, s22 // (op 946) 
s_add_i32 s49, s25, s24 // (op 946) 


s_and_b32 s24, s1, s49 // (op 946) 


s_add_i32 s1, s23, s24 // (op 946) 


s_ashr_i32 s23, s1, s22 // (op 946) 


s_lshl_b32 s1, s46, 24 // (op 946) 
s_ashr_i32 s22, s1, 31 // (op 946) 

s_xor_b32 s1, s23, s22 // (op 946) 

s_sub_i32 s23, s1, s22 // (op 946) 


s_mul_i32 s1, s48, s11 // (op 946) 
s_add_u32 s24, s4, 64 // (op 946) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 946) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 946) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 946) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 946) 

s_sub_i32 s22, s24, s1 // (op 946) 

s_mul_i32 s24, s23, s22 // (op 946) 


s_sub_u32 s25, s19, s24 // (op 946) 


s_add_u32 s19, s25, s1 // (op 946) 


s_cmp_eq_i32 s18, 0 // (op 946) 
s_mov_b32 s1, scc // (op 946) 

s_cmp_eq_i32 s1, 0 // (op 946) 
s_cselect_b32 s18, s19, s0 // (op 946) 



s_lshl_b32 s19, s18, 6 // (op 946) 

s_mov_b32 s1, 0 // (op 946) convert
s_mov_b32 s0, s19 // (op 946) convert

v_mul_lo_u32 v15, v1, s27 // (op 946) most significant: low of low * high
v_mul_hi_u32 v13, v1, s26 // (op 946) most significant: high of low * low
v_mul_lo_u32 v14, v1, s26 // (op 946) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 946) most significant: sum

v_mov_b32 v13, s0 // (op 946) 
v_mov_b32 v16, s1 // (op 946) 
v_add_co_u32 v18, s[18:19], v13, v14 // (op 946) least significant half
v_addc_co_u32 v19, s[18:19], v16, v15, s[18:19] // (op 946) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 946) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 946) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 946) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 946) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 946) 

v_mov_b32 v13, v14 // (op 946) convert

s_mul_i32 s1, s31, 6 // (op 948) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 948) least significant
s_mul_hi_u32 s18, s30, 6 // (op 948) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 948) most significant: sum

s_mul_i32 s19, s26, s1 // (op 948) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 948) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 948) least significant
s_mul_hi_u32 s23, s26, s0 // (op 948) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 948) most significant: sum
s_add_u32 s19, s19, s23 // (op 948) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 948) 

s_mov_b32 s52, s28 // (op 948) reg expression
s_mov_b32 s53, s29 // (op 948) reg expression
s_mov_b32 s54, s0 // (op 948) reg expression
s_mov_b32 s55, 131072 // (op 948) call()

s_mul_hi_i32 s0, s16, s10 // (op 991) 
s_add_i32 s1, s0, s16 // (op 991) 

s_ashr_i32 s0, s1, 31 // (op 991) 
s_cmp_eq_i32 s10, 0 // (op 991) 
s_mov_b32 s18, scc // (op 991) 
s_cmp_eq_i32 s18, 0 // (op 991) 
s_cselect_b32 s19, 0, -1 // (op 991) 

s_and_b32 s18, s13, 31 // (op 991) 
s_lshl_b32 s22, 1, s18 // (op 991) 
s_add_i32 s23, s22, s19 // (op 991) 


s_and_b32 s19, s0, s23 // (op 991) 


s_add_i32 s0, s1, s19 // (op 991) 


s_ashr_i32 s1, s0, s18 // (op 991) 


s_lshl_b32 s0, s13, 24 // (op 991) 
s_ashr_i32 s18, s0, 31 // (op 991) 

s_xor_b32 s0, s1, s18 // (op 991) 

s_sub_i32 s1, s0, s18 // (op 991) 


s_mul_hi_i32 s0, s1, s14 // (op 991) 
s_add_i32 s18, s0, s1 // (op 991) 

s_ashr_i32 s0, s18, 31 // (op 991) 
s_cmp_eq_i32 s14, 0 // (op 991) 
s_mov_b32 s19, scc // (op 991) 
s_cmp_eq_i32 s19, 0 // (op 991) 
s_cselect_b32 s22, 0, -1 // (op 991) 

s_and_b32 s19, s15, 31 // (op 991) 
s_lshl_b32 s23, 1, s19 // (op 991) 
s_add_i32 s24, s23, s22 // (op 991) 


s_and_b32 s25, s0, s24 // (op 991) 


s_add_i32 s0, s18, s25 // (op 991) 


s_ashr_i32 s18, s0, s19 // (op 991) 


s_lshl_b32 s0, s15, 24 // (op 991) 
s_ashr_i32 s19, s0, 31 // (op 991) 

s_xor_b32 s0, s18, s19 // (op 991) 

s_sub_i32 s18, s0, s19 // (op 991) 


s_add_u32 s50, s6, 16 // (op 991) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 991) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 991) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 991) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 991) 

s_mul_i32 s0, s18, s24 // (op 991) 

s_sub_i32 s18, s1, s0 // (op 991) 


s_mul_hi_i32 s19, s16, s9 // (op 991) 
s_add_i32 s0, s19, s16 // (op 991) 

s_ashr_i32 s1, s0, 31 // (op 991) 
s_cmp_eq_i32 s9, 0 // (op 991) 
s_mov_b32 s19, scc // (op 991) 
s_cmp_eq_i32 s19, 0 // (op 991) 
s_cselect_b32 s22, 0, -1 // (op 991) 

s_and_b32 s19, s47, 31 // (op 991) 
s_lshl_b32 s23, 1, s19 // (op 991) 
s_add_i32 s49, s23, s22 // (op 991) 


s_and_b32 s22, s1, s49 // (op 991) 


s_add_i32 s1, s0, s22 // (op 991) 


s_ashr_i32 s0, s1, s19 // (op 991) 


s_lshl_b32 s1, s47, 24 // (op 991) 
s_ashr_i32 s19, s1, 31 // (op 991) 

s_xor_b32 s1, s0, s19 // (op 991) 

s_sub_i32 s0, s1, s19 // (op 991) 


s_mul_i32 s1, s11, s24 // (op 991) 

s_mul_i32 s19, s8, s1 // (op 991) 

s_mul_i32 s1, s0, s19 // (op 991) 

s_sub_u32 s19, s16, s1 // (op 991) 

s_mul_hi_i32 s1, s19, s12 // (op 991) 
s_add_i32 s22, s1, s19 // (op 991) 


s_ashr_i32 s1, s22, 31 // (op 991) 
s_cmp_eq_i32 s12, 0 // (op 991) 
s_mov_b32 s19, scc // (op 991) 
s_cmp_eq_i32 s19, 0 // (op 991) 
s_cselect_b32 s23, 0, -1 // (op 991) 

s_and_b32 s19, s46, 31 // (op 991) 
s_lshl_b32 s24, 1, s19 // (op 991) 
s_add_i32 s25, s24, s23 // (op 991) 


s_and_b32 s23, s1, s25 // (op 991) 


s_add_i32 s1, s22, s23 // (op 991) 


s_ashr_i32 s22, s1, s19 // (op 991) 


s_lshl_b32 s1, s46, 24 // (op 991) 
s_ashr_i32 s19, s1, 31 // (op 991) 

s_xor_b32 s1, s22, s19 // (op 991) 

s_sub_i32 s22, s1, s19 // (op 991) 


s_cmp_eq_i32 s0, 0 // (op 991) 
s_mov_b32 s1, scc // (op 991) 

s_cmp_eq_i32 s1, 0 // (op 991) 
s_cselect_b32 s0, s22, s18 // (op 991) 



s_lshl_b32 s1, s0, 4 // (op 991) 

s_mov_b32 s19, 0 // (op 991) convert
s_mov_b32 s18, s1 // (op 991) convert

v_mul_lo_u32 v15, v1, s33 // (op 991) most significant: low of low * high
v_mul_hi_u32 v16, v1, s32 // (op 991) most significant: high of low * low
v_mul_lo_u32 v14, v1, s32 // (op 991) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 991) most significant: sum

v_mov_b32 v16, s18 // (op 991) 
v_mov_b32 v17, s19 // (op 991) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 991) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 991) most significant half




s_mul_i32 s1, s32, s31 // (op 993) most significant: low of low * high
s_mul_i32 s18, s33, s30 // (op 993) most significant: low of high * low
s_mul_i32 s0, s32, s30 // (op 993) least significant
s_mul_hi_u32 s19, s32, s30 // (op 993) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 993) most significant: sum
s_add_u32 s1, s1, s19 // (op 993) most significant: sum


s_mov_b32 s64, s34 // (op 993) reg expression
s_mov_b32 s65, s35 // (op 993) reg expression
s_mov_b32 s66, s0 // (op 993) reg expression
s_mov_b32 s67, 131072 // (op 993) call()

s_mul_hi_i32 s0, s16, s10 // (op 1010) 
s_add_i32 s1, s0, s16 // (op 1010) 

s_ashr_i32 s0, s1, 31 // (op 1010) 
s_cmp_eq_i32 s10, 0 // (op 1010) 
s_mov_b32 s18, scc // (op 1010) 
s_cmp_eq_i32 s18, 0 // (op 1010) 
s_cselect_b32 s19, 0, -1 // (op 1010) 

s_and_b32 s18, s13, 31 // (op 1010) 
s_lshl_b32 s22, 1, s18 // (op 1010) 
s_add_i32 s23, s22, s19 // (op 1010) 


s_and_b32 s19, s0, s23 // (op 1010) 


s_add_i32 s0, s1, s19 // (op 1010) 


s_ashr_i32 s1, s0, s18 // (op 1010) 


s_lshl_b32 s0, s13, 24 // (op 1010) 
s_ashr_i32 s18, s0, 31 // (op 1010) 

s_xor_b32 s0, s1, s18 // (op 1010) 

s_sub_i32 s1, s0, s18 // (op 1010) 


s_mul_hi_i32 s0, s1, s14 // (op 1010) 
s_add_i32 s18, s0, s1 // (op 1010) 

s_ashr_i32 s0, s18, 31 // (op 1010) 
s_cmp_eq_i32 s14, 0 // (op 1010) 
s_mov_b32 s19, scc // (op 1010) 
s_cmp_eq_i32 s19, 0 // (op 1010) 
s_cselect_b32 s22, 0, -1 // (op 1010) 

s_and_b32 s19, s15, 31 // (op 1010) 
s_lshl_b32 s23, 1, s19 // (op 1010) 
s_add_i32 s24, s23, s22 // (op 1010) 


s_and_b32 s25, s0, s24 // (op 1010) 


s_add_i32 s0, s18, s25 // (op 1010) 


s_ashr_i32 s18, s0, s19 // (op 1010) 


s_lshl_b32 s0, s15, 24 // (op 1010) 
s_ashr_i32 s19, s0, 31 // (op 1010) 

s_xor_b32 s0, s18, s19 // (op 1010) 

s_sub_i32 s18, s0, s19 // (op 1010) 


s_add_u32 s50, s6, 16 // (op 1010) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1010) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1010) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1010) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1010) 

s_mul_i32 s0, s18, s24 // (op 1010) 

s_sub_i32 s18, s1, s0 // (op 1010) 


s_mul_hi_i32 s19, s16, s9 // (op 1010) 
s_add_i32 s0, s19, s16 // (op 1010) 

s_ashr_i32 s1, s0, 31 // (op 1010) 
s_cmp_eq_i32 s9, 0 // (op 1010) 
s_mov_b32 s19, scc // (op 1010) 
s_cmp_eq_i32 s19, 0 // (op 1010) 
s_cselect_b32 s22, 0, -1 // (op 1010) 

s_and_b32 s19, s47, 31 // (op 1010) 
s_lshl_b32 s23, 1, s19 // (op 1010) 
s_add_i32 s49, s23, s22 // (op 1010) 


s_and_b32 s22, s1, s49 // (op 1010) 


s_add_i32 s1, s0, s22 // (op 1010) 


s_ashr_i32 s0, s1, s19 // (op 1010) 


s_lshl_b32 s1, s47, 24 // (op 1010) 
s_ashr_i32 s19, s1, 31 // (op 1010) 

s_xor_b32 s1, s0, s19 // (op 1010) 

s_sub_i32 s0, s1, s19 // (op 1010) 


s_mul_i32 s1, s11, s24 // (op 1010) 

s_mul_i32 s19, s8, s1 // (op 1010) 

s_mul_i32 s1, s0, s19 // (op 1010) 

s_sub_u32 s19, s16, s1 // (op 1010) 

s_mul_hi_i32 s1, s19, s12 // (op 1010) 
s_add_i32 s22, s1, s19 // (op 1010) 


s_ashr_i32 s1, s22, 31 // (op 1010) 
s_cmp_eq_i32 s12, 0 // (op 1010) 
s_mov_b32 s19, scc // (op 1010) 
s_cmp_eq_i32 s19, 0 // (op 1010) 
s_cselect_b32 s23, 0, -1 // (op 1010) 

s_and_b32 s19, s46, 31 // (op 1010) 
s_lshl_b32 s24, 1, s19 // (op 1010) 
s_add_i32 s25, s24, s23 // (op 1010) 


s_and_b32 s23, s1, s25 // (op 1010) 


s_add_i32 s1, s22, s23 // (op 1010) 


s_ashr_i32 s22, s1, s19 // (op 1010) 


s_lshl_b32 s1, s46, 24 // (op 1010) 
s_ashr_i32 s19, s1, 31 // (op 1010) 

s_xor_b32 s1, s22, s19 // (op 1010) 

s_sub_i32 s22, s1, s19 // (op 1010) 


s_cmp_eq_i32 s0, 0 // (op 1010) 
s_mov_b32 s1, scc // (op 1010) 

s_cmp_eq_i32 s1, 0 // (op 1010) 
s_cselect_b32 s0, s22, s18 // (op 1010) 



s_lshl_b32 s1, s0, 4 // (op 1010) 

s_mov_b32 s19, 0 // (op 1010) convert
s_mov_b32 s18, s1 // (op 1010) convert

v_mul_lo_u32 v17, v1, s33 // (op 1010) most significant: low of low * high
v_mul_hi_u32 v18, v1, s32 // (op 1010) most significant: high of low * low
v_mul_lo_u32 v16, v1, s32 // (op 1010) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1010) most significant: sum

v_mov_b32 v18, s18 // (op 1010) 
v_mov_b32 v19, s19 // (op 1010) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 1010) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 1010) most significant half




s_mul_i32 s1, s32, s31 // (op 1012) most significant: low of low * high
s_mul_i32 s18, s33, s30 // (op 1012) most significant: low of high * low
s_mul_i32 s0, s32, s30 // (op 1012) least significant
s_mul_hi_u32 s19, s32, s30 // (op 1012) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1012) most significant: sum
s_add_u32 s1, s1, s19 // (op 1012) most significant: sum


s_mov_b32 s64, s34 // (op 1012) reg expression
s_mov_b32 s65, s35 // (op 1012) reg expression
s_mov_b32 s66, s0 // (op 1012) reg expression
s_mov_b32 s67, 131072 // (op 1012) call()

s_mul_hi_i32 s0, s16, s10 // (op 1055) 
v_and_b32 v18, -64, v1 // (op 1055) 
s_add_i32 s1, s0, s16 // (op 1055) 

v_sub_u32 v19, v1, v18 // (op 1055) 

s_ashr_i32 s0, s1, 31 // (op 1055) 
v_lshrrev_b32 v18, 6, v1 // (op 1055) 
s_cmp_eq_i32 s10, 0 // (op 1055) 
s_mov_b32 s18, scc // (op 1055) 
v_lshlrev_b32 v20, 1, v18 // (op 1055) 

s_cmp_eq_i32 s18, 0 // (op 1055) 
s_cselect_b32 s19, 0, -1 // (op 1055) 

s_and_b32 s18, s13, 31 // (op 1055) 
s_lshl_b32 s22, 1, s18 // (op 1055) 
s_add_i32 s23, s22, s19 // (op 1055) 


s_and_b32 s19, s0, s23 // (op 1055) 


s_add_i32 s0, s1, s19 // (op 1055) 


s_ashr_i32 s1, s0, s18 // (op 1055) 


s_lshl_b32 s0, s13, 24 // (op 1055) 
s_ashr_i32 s18, s0, 31 // (op 1055) 

s_xor_b32 s0, s1, s18 // (op 1055) 

s_sub_i32 s1, s0, s18 // (op 1055) 


s_mul_hi_i32 s0, s1, s14 // (op 1055) 
s_add_i32 s18, s0, s1 // (op 1055) 

s_ashr_i32 s0, s18, 31 // (op 1055) 
s_cmp_eq_i32 s14, 0 // (op 1055) 
s_mov_b32 s19, scc // (op 1055) 
s_cmp_eq_i32 s19, 0 // (op 1055) 
s_cselect_b32 s22, 0, -1 // (op 1055) 

s_and_b32 s19, s15, 31 // (op 1055) 
s_lshl_b32 s23, 1, s19 // (op 1055) 
s_add_i32 s24, s23, s22 // (op 1055) 


s_and_b32 s25, s0, s24 // (op 1055) 


s_add_i32 s0, s18, s25 // (op 1055) 


s_ashr_i32 s18, s0, s19 // (op 1055) 


s_lshl_b32 s0, s15, 24 // (op 1055) 
s_ashr_i32 s19, s0, 31 // (op 1055) 

s_xor_b32 s0, s18, s19 // (op 1055) 

s_sub_i32 s18, s0, s19 // (op 1055) 


s_mul_i32 s0, s1, s11 // (op 1055) 

s_sub_u32 s1, s16, s0 // (op 1055) 

s_mul_i32 s0, s18, s11 // (op 1055) 
s_add_u32 s0, s0, s1 // (op 1055) 


s_mul_hi_i32 s1, s16, s9 // (op 1055) 
s_add_i32 s18, s1, s16 // (op 1055) 

s_ashr_i32 s1, s18, 31 // (op 1055) 
s_cmp_eq_i32 s9, 0 // (op 1055) 
s_mov_b32 s19, scc // (op 1055) 
s_cmp_eq_i32 s19, 0 // (op 1055) 
s_cselect_b32 s22, 0, -1 // (op 1055) 

s_and_b32 s19, s47, 31 // (op 1055) 
s_lshl_b32 s23, 1, s19 // (op 1055) 
s_add_i32 s24, s23, s22 // (op 1055) 


s_and_b32 s25, s1, s24 // (op 1055) 


s_add_i32 s1, s18, s25 // (op 1055) 


s_ashr_i32 s18, s1, s19 // (op 1055) 


s_lshl_b32 s1, s47, 24 // (op 1055) 
s_ashr_i32 s19, s1, 31 // (op 1055) 

s_xor_b32 s1, s18, s19 // (op 1055) 

s_sub_i32 s18, s1, s19 // (op 1055) 


s_add_u32 s50, s6, 16 // (op 1055) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1055) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1055) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1055) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1055) 

s_mul_i32 s1, s11, s24 // (op 1055) 

s_mul_i32 s19, s8, s1 // (op 1055) 

s_mul_i32 s1, s18, s19 // (op 1055) 

s_sub_u32 s19, s16, s1 // (op 1055) 

s_mov_b32 s1, s19 // (op 1055) 
s_mul_hi_i32 s22, s1, s12 // (op 1055) 
s_add_i32 s23, s22, s1 // (op 1055) 


s_ashr_i32 s1, s23, 31 // (op 1055) 
s_cmp_eq_i32 s12, 0 // (op 1055) 
s_mov_b32 s22, scc // (op 1055) 
s_cmp_eq_i32 s22, 0 // (op 1055) 
s_cselect_b32 s24, 0, -1 // (op 1055) 

s_and_b32 s22, s46, 31 // (op 1055) 
s_lshl_b32 s25, 1, s22 // (op 1055) 
s_add_i32 s49, s25, s24 // (op 1055) 


s_and_b32 s24, s1, s49 // (op 1055) 


s_add_i32 s1, s23, s24 // (op 1055) 


s_ashr_i32 s23, s1, s22 // (op 1055) 


s_lshl_b32 s1, s46, 24 // (op 1055) 
s_ashr_i32 s22, s1, 31 // (op 1055) 

s_xor_b32 s1, s23, s22 // (op 1055) 

s_sub_i32 s23, s1, s22 // (op 1055) 


s_mul_i32 s1, s48, s11 // (op 1055) 
s_add_u32 s24, s4, 64 // (op 1055) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1055) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1055) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1055) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1055) 

s_sub_i32 s22, s24, s1 // (op 1055) 

s_mul_i32 s24, s23, s22 // (op 1055) 


s_sub_u32 s25, s19, s24 // (op 1055) 


s_add_u32 s19, s25, s1 // (op 1055) 


s_cmp_eq_i32 s18, 0 // (op 1055) 
s_mov_b32 s1, scc // (op 1055) 

s_cmp_eq_i32 s1, 0 // (op 1055) 
s_cselect_b32 s18, s19, s0 // (op 1055) 



v_lshl_add_u32 v18, s18, 6, v19 // (op 1055) 


v_mul_lo_u32 v19, v20, s36 // (op 1055) 

v_add_u32 v20, v18, v19 // (op 1055) 


s_mul_i32 s1, s36, s41 // (op 1057) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1057) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1057) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1057) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1057) most significant: sum
s_add_u32 s1, s1, s19 // (op 1057) most significant: sum


s_mov_b32 s68, s38 // (op 1057) reg expression
s_mov_b32 s69, s39 // (op 1057) reg expression
s_mov_b32 s70, s0 // (op 1057) reg expression
s_mov_b32 s71, 131072 // (op 1057) call()

s_mul_hi_i32 s0, s16, s10 // (op 1074) 
v_and_b32 v18, -64, v1 // (op 1074) 
s_add_i32 s1, s0, s16 // (op 1074) 

v_sub_u32 v19, v1, v18 // (op 1074) 

s_ashr_i32 s0, s1, 31 // (op 1074) 
v_lshrrev_b32 v18, 6, v1 // (op 1074) 
s_cmp_eq_i32 s10, 0 // (op 1074) 
s_mov_b32 s18, scc // (op 1074) 
v_lshlrev_b32 v21, 1, v18 // (op 1074) 

s_cmp_eq_i32 s18, 0 // (op 1074) 
s_cselect_b32 s19, 0, -1 // (op 1074) 

s_and_b32 s18, s13, 31 // (op 1074) 
s_lshl_b32 s22, 1, s18 // (op 1074) 
s_add_i32 s23, s22, s19 // (op 1074) 


s_and_b32 s19, s0, s23 // (op 1074) 


s_add_i32 s0, s1, s19 // (op 1074) 


s_ashr_i32 s1, s0, s18 // (op 1074) 


s_lshl_b32 s0, s13, 24 // (op 1074) 
s_ashr_i32 s18, s0, 31 // (op 1074) 

s_xor_b32 s0, s1, s18 // (op 1074) 

s_sub_i32 s1, s0, s18 // (op 1074) 


s_mul_hi_i32 s0, s1, s14 // (op 1074) 
s_add_i32 s18, s0, s1 // (op 1074) 

s_ashr_i32 s0, s18, 31 // (op 1074) 
s_cmp_eq_i32 s14, 0 // (op 1074) 
s_mov_b32 s19, scc // (op 1074) 
s_cmp_eq_i32 s19, 0 // (op 1074) 
s_cselect_b32 s22, 0, -1 // (op 1074) 

s_and_b32 s19, s15, 31 // (op 1074) 
s_lshl_b32 s23, 1, s19 // (op 1074) 
s_add_i32 s24, s23, s22 // (op 1074) 


s_and_b32 s25, s0, s24 // (op 1074) 


s_add_i32 s0, s18, s25 // (op 1074) 


s_ashr_i32 s18, s0, s19 // (op 1074) 


s_lshl_b32 s0, s15, 24 // (op 1074) 
s_ashr_i32 s19, s0, 31 // (op 1074) 

s_xor_b32 s0, s18, s19 // (op 1074) 

s_sub_i32 s18, s0, s19 // (op 1074) 


s_mul_i32 s0, s1, s11 // (op 1074) 

s_sub_u32 s1, s16, s0 // (op 1074) 

s_mul_i32 s0, s18, s11 // (op 1074) 
s_add_u32 s0, s0, s1 // (op 1074) 


s_mul_hi_i32 s1, s16, s9 // (op 1074) 
s_add_i32 s18, s1, s16 // (op 1074) 

s_ashr_i32 s1, s18, 31 // (op 1074) 
s_cmp_eq_i32 s9, 0 // (op 1074) 
s_mov_b32 s19, scc // (op 1074) 
s_cmp_eq_i32 s19, 0 // (op 1074) 
s_cselect_b32 s22, 0, -1 // (op 1074) 

s_and_b32 s19, s47, 31 // (op 1074) 
s_lshl_b32 s23, 1, s19 // (op 1074) 
s_add_i32 s24, s23, s22 // (op 1074) 


s_and_b32 s25, s1, s24 // (op 1074) 


s_add_i32 s1, s18, s25 // (op 1074) 


s_ashr_i32 s18, s1, s19 // (op 1074) 


s_lshl_b32 s1, s47, 24 // (op 1074) 
s_ashr_i32 s19, s1, 31 // (op 1074) 

s_xor_b32 s1, s18, s19 // (op 1074) 

s_sub_i32 s18, s1, s19 // (op 1074) 


s_add_u32 s50, s6, 16 // (op 1074) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1074) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1074) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1074) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1074) 

s_mul_i32 s1, s11, s24 // (op 1074) 

s_mul_i32 s19, s8, s1 // (op 1074) 

s_mul_i32 s1, s18, s19 // (op 1074) 

s_sub_u32 s19, s16, s1 // (op 1074) 

s_mov_b32 s1, s19 // (op 1074) 
s_mul_hi_i32 s22, s1, s12 // (op 1074) 
s_add_i32 s23, s22, s1 // (op 1074) 


s_ashr_i32 s1, s23, 31 // (op 1074) 
s_cmp_eq_i32 s12, 0 // (op 1074) 
s_mov_b32 s22, scc // (op 1074) 
s_cmp_eq_i32 s22, 0 // (op 1074) 
s_cselect_b32 s24, 0, -1 // (op 1074) 

s_and_b32 s22, s46, 31 // (op 1074) 
s_lshl_b32 s25, 1, s22 // (op 1074) 
s_add_i32 s49, s25, s24 // (op 1074) 


s_and_b32 s24, s1, s49 // (op 1074) 


s_add_i32 s1, s23, s24 // (op 1074) 


s_ashr_i32 s23, s1, s22 // (op 1074) 


s_lshl_b32 s1, s46, 24 // (op 1074) 
s_ashr_i32 s22, s1, 31 // (op 1074) 

s_xor_b32 s1, s23, s22 // (op 1074) 

s_sub_i32 s23, s1, s22 // (op 1074) 


s_mul_i32 s1, s48, s11 // (op 1074) 
s_add_u32 s24, s4, 64 // (op 1074) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1074) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1074) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1074) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1074) 

s_sub_i32 s22, s24, s1 // (op 1074) 

s_mul_i32 s24, s23, s22 // (op 1074) 


s_sub_u32 s25, s19, s24 // (op 1074) 


s_add_u32 s19, s25, s1 // (op 1074) 


s_cmp_eq_i32 s18, 0 // (op 1074) 
s_mov_b32 s1, scc // (op 1074) 

s_cmp_eq_i32 s1, 0 // (op 1074) 
s_cselect_b32 s18, s19, s0 // (op 1074) 



v_lshl_add_u32 v18, s18, 6, v19 // (op 1074) 


v_mul_lo_u32 v19, v21, s36 // (op 1074) 

v_add_u32 v21, v18, v19 // (op 1074) 


s_mul_i32 s1, s36, s41 // (op 1076) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1076) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1076) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1076) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1076) most significant: sum
s_add_u32 s1, s1, s19 // (op 1076) most significant: sum


s_mov_b32 s68, s38 // (op 1076) reg expression
s_mov_b32 s69, s39 // (op 1076) reg expression
s_mov_b32 s70, s0 // (op 1076) reg expression
s_mov_b32 s71, 131072 // (op 1076) call()

s_mul_hi_i32 s0, s16, s10 // (op 1119) 
v_and_b32 v18, 15, v1 // (op 1119) 
s_add_i32 s1, s0, s16 // (op 1119) 

v_and_b32 v19, 63, v1 // (op 1119) 
s_ashr_i32 s0, s1, 31 // (op 1119) 
v_lshrrev_b32 v22, 4, v19 // (op 1119) 

s_cmp_eq_i32 s10, 0 // (op 1119) 
s_mov_b32 s18, scc // (op 1119) 
s_cmp_eq_i32 s18, 0 // (op 1119) 
s_cselect_b32 s19, 0, -1 // (op 1119) 

s_and_b32 s18, s13, 31 // (op 1119) 
s_lshl_b32 s22, 1, s18 // (op 1119) 
s_add_i32 s23, s22, s19 // (op 1119) 


s_and_b32 s19, s0, s23 // (op 1119) 


s_add_i32 s0, s1, s19 // (op 1119) 


s_ashr_i32 s1, s0, s18 // (op 1119) 


s_lshl_b32 s0, s13, 24 // (op 1119) 
s_ashr_i32 s18, s0, 31 // (op 1119) 

s_xor_b32 s0, s1, s18 // (op 1119) 

s_sub_i32 s1, s0, s18 // (op 1119) 


s_mul_hi_i32 s0, s1, s14 // (op 1119) 
s_add_i32 s18, s0, s1 // (op 1119) 

s_ashr_i32 s0, s18, 31 // (op 1119) 
s_cmp_eq_i32 s14, 0 // (op 1119) 
s_mov_b32 s19, scc // (op 1119) 
s_cmp_eq_i32 s19, 0 // (op 1119) 
s_cselect_b32 s22, 0, -1 // (op 1119) 

s_and_b32 s19, s15, 31 // (op 1119) 
s_lshl_b32 s23, 1, s19 // (op 1119) 
s_add_i32 s24, s23, s22 // (op 1119) 


s_and_b32 s25, s0, s24 // (op 1119) 


s_add_i32 s0, s18, s25 // (op 1119) 


s_ashr_i32 s18, s0, s19 // (op 1119) 


s_lshl_b32 s0, s15, 24 // (op 1119) 
s_ashr_i32 s19, s0, 31 // (op 1119) 

s_xor_b32 s0, s18, s19 // (op 1119) 

s_sub_i32 s18, s0, s19 // (op 1119) 


s_add_u32 s50, s6, 16 // (op 1119) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1119) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1119) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1119) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1119) 

s_mul_i32 s0, s18, s24 // (op 1119) 

s_sub_i32 s18, s1, s0 // (op 1119) 


s_mul_hi_i32 s19, s16, s9 // (op 1119) 
s_add_i32 s0, s19, s16 // (op 1119) 

s_ashr_i32 s1, s0, 31 // (op 1119) 
s_cmp_eq_i32 s9, 0 // (op 1119) 
s_mov_b32 s19, scc // (op 1119) 
s_cmp_eq_i32 s19, 0 // (op 1119) 
s_cselect_b32 s22, 0, -1 // (op 1119) 

s_and_b32 s19, s47, 31 // (op 1119) 
s_lshl_b32 s23, 1, s19 // (op 1119) 
s_add_i32 s49, s23, s22 // (op 1119) 


s_and_b32 s22, s1, s49 // (op 1119) 


s_add_i32 s1, s0, s22 // (op 1119) 


s_ashr_i32 s0, s1, s19 // (op 1119) 


s_lshl_b32 s1, s47, 24 // (op 1119) 
s_ashr_i32 s19, s1, 31 // (op 1119) 

s_xor_b32 s1, s0, s19 // (op 1119) 

s_sub_i32 s0, s1, s19 // (op 1119) 


s_mul_i32 s1, s11, s24 // (op 1119) 

s_mul_i32 s19, s8, s1 // (op 1119) 

s_mul_i32 s1, s0, s19 // (op 1119) 

s_sub_u32 s19, s16, s1 // (op 1119) 

s_mul_hi_i32 s1, s19, s12 // (op 1119) 
s_add_i32 s22, s1, s19 // (op 1119) 


s_ashr_i32 s1, s22, 31 // (op 1119) 
s_cmp_eq_i32 s12, 0 // (op 1119) 
s_mov_b32 s19, scc // (op 1119) 
s_cmp_eq_i32 s19, 0 // (op 1119) 
s_cselect_b32 s23, 0, -1 // (op 1119) 

s_and_b32 s19, s46, 31 // (op 1119) 
s_lshl_b32 s24, 1, s19 // (op 1119) 
s_add_i32 s25, s24, s23 // (op 1119) 


s_and_b32 s23, s1, s25 // (op 1119) 


s_add_i32 s1, s22, s23 // (op 1119) 


s_ashr_i32 s22, s1, s19 // (op 1119) 


s_lshl_b32 s1, s46, 24 // (op 1119) 
s_ashr_i32 s19, s1, 31 // (op 1119) 

s_xor_b32 s1, s22, s19 // (op 1119) 

s_sub_i32 s22, s1, s19 // (op 1119) 


s_cmp_eq_i32 s0, 0 // (op 1119) 
s_mov_b32 s1, scc // (op 1119) 

s_cmp_eq_i32 s1, 0 // (op 1119) 
s_cselect_b32 s0, s22, s18 // (op 1119) 



v_lshl_add_u32 v19, s0, 4, v18 // (op 1119) 


v_mul_lo_u32 v18, v22, s42 // (op 1119) 
v_add_u32 v18, v18, v19 // (op 1119) 


s_mul_i32 s1, s42, s41 // (op 1121) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1121) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1121) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1121) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1121) most significant: sum
s_add_u32 s1, s1, s19 // (op 1121) most significant: sum


s_mov_b32 s72, s44 // (op 1121) reg expression
s_mov_b32 s73, s45 // (op 1121) reg expression
s_mov_b32 s74, s0 // (op 1121) reg expression
s_mov_b32 s75, 131072 // (op 1121) call()

s_mul_hi_i32 s0, s16, s10 // (op 1138) 
v_and_b32 v19, 15, v1 // (op 1138) 
s_add_i32 s1, s0, s16 // (op 1138) 

v_and_b32 v22, 63, v1 // (op 1138) 
s_ashr_i32 s0, s1, 31 // (op 1138) 
v_lshrrev_b32 v23, 4, v22 // (op 1138) 

s_cmp_eq_i32 s10, 0 // (op 1138) 
s_mov_b32 s18, scc // (op 1138) 
s_cmp_eq_i32 s18, 0 // (op 1138) 
s_cselect_b32 s19, 0, -1 // (op 1138) 

s_and_b32 s18, s13, 31 // (op 1138) 
s_lshl_b32 s22, 1, s18 // (op 1138) 
s_add_i32 s23, s22, s19 // (op 1138) 


s_and_b32 s19, s0, s23 // (op 1138) 


s_add_i32 s0, s1, s19 // (op 1138) 


s_ashr_i32 s1, s0, s18 // (op 1138) 


s_lshl_b32 s0, s13, 24 // (op 1138) 
s_ashr_i32 s18, s0, 31 // (op 1138) 

s_xor_b32 s0, s1, s18 // (op 1138) 

s_sub_i32 s1, s0, s18 // (op 1138) 


s_mul_hi_i32 s0, s1, s14 // (op 1138) 
s_add_i32 s18, s0, s1 // (op 1138) 

s_ashr_i32 s0, s18, 31 // (op 1138) 
s_cmp_eq_i32 s14, 0 // (op 1138) 
s_mov_b32 s19, scc // (op 1138) 
s_cmp_eq_i32 s19, 0 // (op 1138) 
s_cselect_b32 s22, 0, -1 // (op 1138) 

s_and_b32 s19, s15, 31 // (op 1138) 
s_lshl_b32 s23, 1, s19 // (op 1138) 
s_add_i32 s24, s23, s22 // (op 1138) 


s_and_b32 s25, s0, s24 // (op 1138) 


s_add_i32 s0, s18, s25 // (op 1138) 


s_ashr_i32 s18, s0, s19 // (op 1138) 


s_lshl_b32 s0, s15, 24 // (op 1138) 
s_ashr_i32 s19, s0, 31 // (op 1138) 

s_xor_b32 s0, s18, s19 // (op 1138) 

s_sub_i32 s18, s0, s19 // (op 1138) 


s_add_u32 s50, s6, 16 // (op 1138) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1138) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1138) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1138) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1138) 

s_mul_i32 s0, s18, s24 // (op 1138) 

s_sub_i32 s18, s1, s0 // (op 1138) 


s_mul_hi_i32 s19, s16, s9 // (op 1138) 
s_add_i32 s0, s19, s16 // (op 1138) 

s_ashr_i32 s1, s0, 31 // (op 1138) 
s_cmp_eq_i32 s9, 0 // (op 1138) 
s_mov_b32 s19, scc // (op 1138) 
s_cmp_eq_i32 s19, 0 // (op 1138) 
s_cselect_b32 s22, 0, -1 // (op 1138) 

s_and_b32 s19, s47, 31 // (op 1138) 
s_lshl_b32 s23, 1, s19 // (op 1138) 
s_add_i32 s49, s23, s22 // (op 1138) 


s_and_b32 s22, s1, s49 // (op 1138) 


s_add_i32 s1, s0, s22 // (op 1138) 


s_ashr_i32 s0, s1, s19 // (op 1138) 


s_lshl_b32 s1, s47, 24 // (op 1138) 
s_ashr_i32 s19, s1, 31 // (op 1138) 

s_xor_b32 s1, s0, s19 // (op 1138) 

s_sub_i32 s0, s1, s19 // (op 1138) 


s_mul_i32 s1, s11, s24 // (op 1138) 

s_mul_i32 s19, s8, s1 // (op 1138) 

s_mul_i32 s1, s0, s19 // (op 1138) 

s_sub_u32 s19, s16, s1 // (op 1138) 

s_mul_hi_i32 s1, s19, s12 // (op 1138) 
s_add_i32 s22, s1, s19 // (op 1138) 


s_ashr_i32 s1, s22, 31 // (op 1138) 
s_cmp_eq_i32 s12, 0 // (op 1138) 
s_mov_b32 s19, scc // (op 1138) 
s_cmp_eq_i32 s19, 0 // (op 1138) 
s_cselect_b32 s23, 0, -1 // (op 1138) 

s_and_b32 s19, s46, 31 // (op 1138) 
s_lshl_b32 s24, 1, s19 // (op 1138) 
s_add_i32 s25, s24, s23 // (op 1138) 


s_and_b32 s23, s1, s25 // (op 1138) 


s_add_i32 s1, s22, s23 // (op 1138) 


s_ashr_i32 s22, s1, s19 // (op 1138) 


s_lshl_b32 s1, s46, 24 // (op 1138) 
s_ashr_i32 s19, s1, 31 // (op 1138) 

s_xor_b32 s1, s22, s19 // (op 1138) 

s_sub_i32 s22, s1, s19 // (op 1138) 


s_cmp_eq_i32 s0, 0 // (op 1138) 
s_mov_b32 s1, scc // (op 1138) 

s_cmp_eq_i32 s1, 0 // (op 1138) 
s_cselect_b32 s0, s22, s18 // (op 1138) 



v_lshl_add_u32 v22, s0, 4, v19 // (op 1138) 


v_mul_lo_u32 v19, v23, s42 // (op 1138) 
v_add_u32 v19, v19, v22 // (op 1138) 


s_mul_i32 s1, s42, s41 // (op 1140) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1140) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1140) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1140) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1140) most significant: sum
s_add_u32 s1, s1, s19 // (op 1140) most significant: sum


s_mov_b32 s72, s44 // (op 1140) reg expression
s_mov_b32 s73, s45 // (op 1140) reg expression
s_mov_b32 s74, s0 // (op 1140) reg expression
s_mov_b32 s75, 131072 // (op 1140) call()


v_mov_b64 v[22:23], s[42:43] // (op 774) convert

v_mov_b32 v24, v18 // (op 774) reg expression
v_mov_b32 v25, v24 // (op 774) 
buffer_load_ubyte v26, v25, s[72:75], 0 offen offset: 0 // (op 774) Load value



v_lshlrev_b32 v22, 4, v1 // (op 1262) 
s_nop 0
v_readfirstlane_b32 s0, v22 // (op 1262) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[22:23], v[14:15] // (op 882) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 882) reg expression
buffer_load_dwordx4 v22, s[64:67], 0 offen offset: 0 lds // (op 882) Load value direct to lds




v_mov_b32 v22, v12 // (op 781) reg expression

buffer_load_dwordx3 v[28:30], v22, s[52:55], 0 offen // (op 781) Load value
buffer_load_dwordx3 v[32:34], v22, s[52:55], 0 offen offset:12 // (op 781) Load value
buffer_load_dwordx3 v[36:38], v22, s[52:55], 0 offen offset:24 // (op 781) Load value
buffer_load_dwordx3 v[40:42], v22, s[52:55], 0 offen offset:36 // (op 781) Load value


v_mov_b64 v[22:23], s[36:37] // (op 784) convert

v_mov_b32 v27, v20 // (op 784) reg expression
buffer_load_ubyte v24, v27, s[68:71], 0 offen offset: 0 // (op 784) Load value
v_add_u32 v27, v27, v22 // (op 784) 
buffer_load_ubyte v25, v27, s[68:71], 0 offen offset: 0 // (op 784) Load value



v_mov_b64 v[22:23], s[42:43] // (op 788) convert

s_lshl_b32 s0, s42, 2 // (op 788) 
v_add_u32 v27, v18, s0 // (op 788) 

v_mov_b32 v31, v27 // (op 788) 
buffer_load_ubyte v35, v31, s[72:75], 0 offen offset: 0 // (op 788) Load value




v_lshlrev_b32 v18, 4, v1 // (op 1406) 
s_nop 0
v_readfirstlane_b32 s0, v18 // (op 1406) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[18:19], s[32:33], 7 // (op 887) 
v_mov_b32 v18, s18 // (op 887) 
v_mov_b32 v27, s19 // (op 887) 
v_add_co_u32 v22, s[50:51], v14, v18 // (op 887) least significant half
v_addc_co_u32 v23, s[50:51], v15, v27, s[50:51] // (op 887) most significant half




s_add_u32 s1, 2048, s0 // (op 887) 

s_mov_b32 m0, s1 // (op 887) reg expression
buffer_load_dwordx4 v22, s[64:67], 0 offen offset: 0 lds // (op 887) Load value direct to lds






s_lshl_b64 s[0:1], s[26:27], 7 // (op 795) 
s_mul_i32 s19, s1, 6 // (op 795) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 795) least significant
s_mul_hi_u32 s22, s0, 6 // (op 795) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 795) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 795) 

v_add_u32 v18, v12, s0 // (op 795) 


buffer_load_dwordx3 v[44:46], v18, s[52:55], 0 offen // (op 795) Load value
buffer_load_dwordx3 v[48:50], v18, s[52:55], 0 offen offset:12 // (op 795) Load value
buffer_load_dwordx3 v[52:54], v18, s[52:55], 0 offen offset:24 // (op 795) Load value
buffer_load_dwordx3 v[56:58], v18, s[52:55], 0 offen offset:36 // (op 795) Load value



v_mov_b64 v[14:15], s[36:37] // (op 798) convert

s_lshl_b32 s0, s36, 2 // (op 798) 
v_add_u32 v12, v20, s0 // (op 798) 

buffer_load_ubyte v22, v12, s[68:71], 0 offen offset: 0 // (op 798) Load value
v_add_u32 v12, v12, v14 // (op 798) 
buffer_load_ubyte v23, v12, s[68:71], 0 offen offset: 0 // (op 798) Load value



v_lshlrev_b32 v12, 6, v1 // (op 1205) 
v_mul_lo_u32 v18, v12, 6 // (op 1205) 
v_lshrrev_b32 v20, 3, v18 // (op 1205) 

v_lshrrev_b32 v18, 4, v12 // (op 1205) 

v_lshlrev_b32 v12, 2, v18 // (op 1205) 

v_add_u32 v18, v20, v12 // (op 1205) 





s_waitcnt vmcnt(13)
ds_write_b96 v18, v[28:30] offset:4096 // (op 801) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v18, v[32:34] offset:4112 // (op 801) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v18, v[36:38] offset:4128 // (op 801) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v18, v[40:42] offset:4144 // (op 801) Store local data 

v_and_b32 v12, -64, v1 // (op 1315) 
v_sub_u32 v18, v1, v12 // (op 1315) 

v_lshrrev_b32 v12, 6, v1 // (op 1315) 
v_lshl_add_u32 v20, v12, 7, v18 // (op 1315) 





s_waitcnt vmcnt(9)
ds_write_b8 v20, v24 offset:12288 // (op 802) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v25 offset:12352 // (op 802) Store local data 

v_lshlrev_b32 v12, 6, v1 // (op 1176) 
v_mul_lo_u32 v18, v12, 6 // (op 1176) 
v_lshrrev_b32 v20, 3, v18 // (op 1176) 

v_lshrrev_b32 v18, 4, v12 // (op 1176) 

v_lshlrev_b32 v12, 2, v18 // (op 1176) 

v_add_u32 v18, v20, v12 // (op 1176) 


v_lshlrev_b32 v12, 4, v1 // (op 1234) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1234) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1286) 
v_sub_u32 v20, v1, v12 // (op 1286) 

v_lshrrev_b32 v12, 6, v1 // (op 1286) 
v_lshl_add_u32 v27, v12, 7, v20 // (op 1286) 


v_lshlrev_b32 v12, 6, v1 // (op 1362) 
v_mul_lo_u32 v20, v12, 6 // (op 1362) 
v_lshrrev_b32 v31, 3, v20 // (op 1362) 

v_lshrrev_b32 v20, 4, v12 // (op 1362) 

v_lshlrev_b32 v12, 2, v20 // (op 1362) 

v_add_u32 v20, v31, v12 // (op 1362) 


v_lshlrev_b32 v12, 4, v1 // (op 1391) 
s_nop 0
v_readfirstlane_b32 s1, v12 // (op 1391) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1430) 
v_sub_u32 v31, v1, v12 // (op 1430) 

v_lshrrev_b32 v12, 6, v1 // (op 1430) 
v_lshl_add_u32 v39, v12, 7, v31 // (op 1430) 


s_mov_b32 s18, 0 // (op 311) call()
s_ashr_i64 s[50:51], s[30:31], 7 // (op 310) 
s_mov_b32 s19, s50 // (op 310) convert

s_and_b32 s22, s19, 4294967294 // (op 310) 

s_cmp_lt_u32 s18, s22 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_310 if false)
RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_310:
s_add_u32 s18, s18, 2 // (op 312) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1488) 



ds_read_b96_tr_b6 v[60:62], v4 offset:4096 // (op 65) Transpose load local data 
ds_read_b96_tr_b6 v[63:65], v4 offset:5120 // (op 65) Transpose load local data 
s_barrier  // (op 1501) 



ds_read_b64_tr_b8 v[66:67], v0 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:128 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:1024 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:1152 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1514) 


ds_read_u8 v12, v7 offset:12288 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:73], a[0:3], v12, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 82) 
ds_read_b96_tr_b6 v[74:76], v4 offset:4128 // (op 340) Transpose load local data 


ds_read_b96_tr_b6 v[77:79], v4 offset:5152 // (op 340) Transpose load local data 


ds_read_u8 v12, v7 offset:12320 // (op 344) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[66:73], a[4:7], v12, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 346) 




v_mov_b64 v[14:15], s[42:43] // (op 80) convert

v_mov_b32 v12, v19 // (op 80) 
s_lshl_b32 s19, s42, 3 // (op 80) 
v_add_u32 v31, v12, s19 // (op 80) 

v_mov_b32 v43, v31 // (op 80) 
buffer_load_ubyte v26, v43, s[72:75], 0 offen offset: 0 // (op 80) Load value



s_barrier  // (op 1504) 


v_mov_b64 v[14:15], v[16:17] // (op 865) 
s_lshl_b64 s[50:51], s[32:33], 7 // (op 865) 
s_lshl_b64 s[22:23], s[50:51], 1 // (op 865) 

v_mov_b32 v31, s22 // (op 865) 
v_mov_b32 v43, s23 // (op 865) 
v_add_co_u32 v60, s[24:25], v14, v31 // (op 865) least significant half
v_addc_co_u32 v61, s[24:25], v15, v43, s[24:25] // (op 865) most significant half





s_mov_b32 m0, s0 // (op 865) reg expression
buffer_load_dwordx4 v60, s[64:67], 0 offen offset: 0 lds // (op 865) Load value direct to lds



v_mov_b32 v31, v13 // (op 4) 
s_lshl_b64 s[50:51], s[26:27], 7 // (op 4) 
s_mul_i32 s23, s51, 6 // (op 4) most significant: low of high * low
s_mul_i32 s22, s50, 6 // (op 4) least significant
s_mul_hi_u32 s19, s50, 6 // (op 4) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 4) most significant: sum


s_lshr_b64 s[24:25], s[22:23], 3 // (op 4) 

s_lshl_b32 s19, s24, 1 // (op 4) 

v_add_u32 v43, v31, s19 // (op 4) 


buffer_load_dwordx3 v[28:30], v43, s[52:55], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[32:34], v43, s[52:55], 0 offen offset:12 // (op 4) Load value
buffer_load_dwordx3 v[36:38], v43, s[52:55], 0 offen offset:24 // (op 4) Load value
buffer_load_dwordx3 v[40:42], v43, s[52:55], 0 offen offset:36 // (op 4) Load value


v_mov_b64 v[60:61], s[36:37] // (op 17) convert

v_mov_b32 v43, v21 // (op 17) 
s_lshl_b32 s19, s36, 3 // (op 17) 
v_add_u32 v47, v43, s19 // (op 17) 

buffer_load_ubyte v24, v47, s[68:71], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v47, v47, v60 // (op 17) 
buffer_load_ubyte v25, v47, s[68:71], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1531) 



s_waitcnt vmcnt(13)
ds_write_b96 v20, v[44:46] offset:12544 // (op 225) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v20, v[48:50] offset:12560 // (op 225) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v20, v[52:54] offset:12576 // (op 225) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v20, v[56:58] offset:12592 // (op 225) Store local data 
s_barrier  // (op 1543) 



s_waitcnt vmcnt(9)
ds_write_b8 v39, v22 offset:20736 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v39, v23 offset:20800 // (op 236) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1528) 



ds_read_b96_tr_b6 v[60:62], v10 offset:12544 // (op 227) Transpose load local data 
ds_read_b96_tr_b6 v[63:65], v10 offset:13568 // (op 227) Transpose load local data 
s_barrier  // (op 1534) 



ds_read_b64_tr_b8 v[66:67], v3 offset:2048 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v3 offset:2176 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v3 offset:3072 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v3 offset:3200 // (op 229) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1540) 


ds_read_u8 v47, v11 offset:20736 // (op 231) Load local data 



ds_read_b96_tr_b6 v[74:76], v10 offset:12576 // (op 353) Transpose load local data 
ds_read_b96_tr_b6 v[77:79], v10 offset:13600 // (op 353) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:73], a[0:3], v47, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 234) 




s_barrier  // (op 1604) 
ds_read_u8 v47, v11 offset:20768 // (op 357) Load local data 
s_barrier  // (op 1608) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[66:73], a[4:7], v47, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 347) 




v_mov_b64 v[60:61], s[42:43] // (op 233) convert

s_lshl_b32 s19, s42, 2 // (op 233) 
s_mul_i32 s22, 3, s19 // (op 233) 

v_add_u32 v47, v12, s22 // (op 233) 

v_mov_b32 v51, v47 // (op 233) 
buffer_load_ubyte v35, v51, s[72:75], 0 offen offset: 0 // (op 233) Load value




s_barrier  // (op 1537) 
s_lshl_b32 s19, s42, 3 // (op 1130) 


v_add_u32 v19, v19, s19 // (op 1130) 

s_lshl_b64 s[50:51], s[32:33], 7 // (op 877) 
s_mul_i32 s23, 3, s51 // (op 877) most significant: low of low * high
s_mul_i32 s22, 3, s50 // (op 877) least significant
s_mul_hi_u32 s19, 3, s50 // (op 877) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 877) most significant: sum


v_mov_b32 v12, s22 // (op 877) 
v_mov_b32 v47, s23 // (op 877) 
v_add_co_u32 v60, s[24:25], v14, v12 // (op 877) least significant half
v_addc_co_u32 v61, s[24:25], v15, v47, s[24:25] // (op 877) most significant half




s_add_u32 s19, 2048, s1 // (op 877) 

s_mov_b32 m0, s19 // (op 877) reg expression
buffer_load_dwordx4 v60, s[64:67], 0 offen offset: 0 lds // (op 877) Load value direct to lds





s_lshl_b64 s[50:51], s[26:27], 7 // (op 224) 
s_mul_i32 s23, s51, 6 // (op 224) most significant: low of high * low
s_mul_i32 s22, s50, 6 // (op 224) least significant
s_mul_hi_u32 s19, s50, 6 // (op 224) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 224) most significant: sum


s_lshr_b64 s[24:25], s[22:23], 3 // (op 224) 

s_mul_i32 s19, 3, s24 // (op 224) 

v_add_u32 v12, v31, s19 // (op 224) 


s_lshl_b64 s[50:51], s[32:33], 7 // (op 1002) 
buffer_load_dwordx3 v[44:46], v12, s[52:55], 0 offen // (op 224) Load value
s_lshl_b64 s[22:23], s[50:51], 1 // (op 1002) 

v_mov_b32 v47, s22 // (op 1002) 
v_mov_b32 v51, s23 // (op 1002) 
v_add_co_u32 v16, s[24:25], v16, v47 // (op 1002) least significant half
v_addc_co_u32 v17, s[24:25], v17, v51, s[24:25] // (op 1002) most significant half




buffer_load_dwordx3 v[48:50], v12, s[52:55], 0 offen offset:12 // (op 224) Load value
buffer_load_dwordx3 v[52:54], v12, s[52:55], 0 offen offset:24 // (op 224) Load value
buffer_load_dwordx3 v[56:58], v12, s[52:55], 0 offen offset:36 // (op 224) Load value



v_mov_b64 v[14:15], s[36:37] // (op 238) convert

s_lshl_b32 s19, s36, 2 // (op 238) 
s_mul_i32 s22, 3, s19 // (op 238) 

v_add_u32 v12, v43, s22 // (op 238) 

s_lshl_b64 s[50:51], s[26:27], 7 // (op 938) 
s_mul_i32 s23, s51, 6 // (op 938) most significant: low of high * low
s_mul_i32 s22, s50, 6 // (op 938) least significant
s_mul_hi_u32 s19, s50, 6 // (op 938) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 938) most significant: sum


s_lshr_b64 s[24:25], s[22:23], 3 // (op 938) 

s_lshl_b32 s19, s24, 1 // (op 938) 

v_add_u32 v13, v13, s19 // (op 938) 

buffer_load_ubyte v22, v12, s[68:71], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v12, v12, v14 // (op 238) 
buffer_load_ubyte v23, v12, s[68:71], 0 offen offset: 0 // (op 238) Load value



s_barrier  // (op 1491) 
s_lshl_b32 s19, s36, 3 // (op 1066) 



v_add_u32 v21, v21, s19 // (op 1066) 

s_waitcnt vmcnt(13)
ds_write_b96 v18, v[28:30] offset:4096 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v18, v[32:34] offset:4112 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v18, v[36:38] offset:4128 // (op 6) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v18, v[40:42] offset:4144 // (op 6) Store local data 
s_barrier  // (op 1517) 



s_waitcnt vmcnt(9)
ds_write_b8 v27, v24 offset:12288 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v27, v25 offset:12352 // (op 21) Store local data 
s_ashr_i64 s[50:51], s[30:31], 7 // (op 310) 
s_mov_b32 s19, s50 // (op 310) convert

s_and_b32 s22, s19, 4294967294 // (op 310) 

s_cmp_lt_u32 s18, s22 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_310 if true)
RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_310:














s_barrier  // (op 1588) 
s_mul_hi_i32 s0, s16, s10 // (op 900) 
s_add_i32 s1, s0, s16 // (op 900) 

s_ashr_i32 s0, s1, 31 // (op 900) 
s_cmp_eq_i32 s10, 0 // (op 900) 
s_mov_b32 s18, scc // (op 900) 
s_cmp_eq_i32 s18, 0 // (op 900) 
s_cselect_b32 s19, 0, -1 // (op 900) 

s_and_b32 s18, s13, 31 // (op 900) 
s_lshl_b32 s22, 1, s18 // (op 900) 
s_add_i32 s23, s22, s19 // (op 900) 


s_and_b32 s19, s0, s23 // (op 900) 


s_add_i32 s0, s1, s19 // (op 900) 


s_ashr_i32 s1, s0, s18 // (op 900) 


s_lshl_b32 s0, s13, 24 // (op 900) 
s_ashr_i32 s18, s0, 31 // (op 900) 

s_xor_b32 s0, s1, s18 // (op 900) 

s_sub_i32 s1, s0, s18 // (op 900) 


s_mul_hi_i32 s0, s1, s14 // (op 900) 
s_add_i32 s18, s0, s1 // (op 900) 

s_ashr_i32 s0, s18, 31 // (op 900) 
s_cmp_eq_i32 s14, 0 // (op 900) 
s_mov_b32 s19, scc // (op 900) 
s_cmp_eq_i32 s19, 0 // (op 900) 
s_cselect_b32 s22, 0, -1 // (op 900) 

s_and_b32 s19, s15, 31 // (op 900) 
s_lshl_b32 s23, 1, s19 // (op 900) 
s_add_i32 s24, s23, s22 // (op 900) 


s_and_b32 s25, s0, s24 // (op 900) 


s_add_i32 s0, s18, s25 // (op 900) 


s_ashr_i32 s18, s0, s19 // (op 900) 


s_lshl_b32 s0, s15, 24 // (op 900) 
s_ashr_i32 s19, s0, 31 // (op 900) 

s_xor_b32 s0, s18, s19 // (op 900) 

s_sub_i32 s18, s0, s19 // (op 900) 


s_mul_i32 s0, s1, s11 // (op 900) 

s_sub_u32 s1, s16, s0 // (op 900) 

s_mul_i32 s0, s18, s11 // (op 900) 
s_add_u32 s0, s0, s1 // (op 900) 


s_mul_hi_i32 s1, s16, s9 // (op 900) 
s_add_i32 s18, s1, s16 // (op 900) 

s_ashr_i32 s1, s18, 31 // (op 900) 
s_cmp_eq_i32 s9, 0 // (op 900) 
s_mov_b32 s19, scc // (op 900) 
s_cmp_eq_i32 s19, 0 // (op 900) 
s_cselect_b32 s22, 0, -1 // (op 900) 

s_and_b32 s19, s47, 31 // (op 900) 
s_lshl_b32 s23, 1, s19 // (op 900) 
s_add_i32 s24, s23, s22 // (op 900) 


s_and_b32 s25, s1, s24 // (op 900) 


s_add_i32 s1, s18, s25 // (op 900) 


s_ashr_i32 s18, s1, s19 // (op 900) 


s_lshl_b32 s1, s47, 24 // (op 900) 
s_ashr_i32 s19, s1, 31 // (op 900) 

s_xor_b32 s1, s18, s19 // (op 900) 

s_sub_i32 s18, s1, s19 // (op 900) 


s_add_u32 s50, s6, 16 // (op 900) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 900) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 900) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 900) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 900) 

s_mul_i32 s1, s11, s24 // (op 900) 

s_mul_i32 s19, s8, s1 // (op 900) 

s_mul_i32 s1, s18, s19 // (op 900) 

s_sub_u32 s19, s16, s1 // (op 900) 

s_mov_b32 s1, s19 // (op 900) 
s_mul_hi_i32 s22, s1, s12 // (op 900) 
s_add_i32 s23, s22, s1 // (op 900) 


s_ashr_i32 s1, s23, 31 // (op 900) 
s_cmp_eq_i32 s12, 0 // (op 900) 
s_mov_b32 s22, scc // (op 900) 
s_cmp_eq_i32 s22, 0 // (op 900) 
s_cselect_b32 s24, 0, -1 // (op 900) 

s_and_b32 s22, s46, 31 // (op 900) 
s_lshl_b32 s25, 1, s22 // (op 900) 
s_add_i32 s49, s25, s24 // (op 900) 


s_and_b32 s24, s1, s49 // (op 900) 


s_add_i32 s1, s23, s24 // (op 900) 


s_ashr_i32 s23, s1, s22 // (op 900) 


s_lshl_b32 s1, s46, 24 // (op 900) 
s_ashr_i32 s22, s1, 31 // (op 900) 

s_xor_b32 s1, s23, s22 // (op 900) 

s_sub_i32 s23, s1, s22 // (op 900) 


s_mul_i32 s1, s48, s11 // (op 900) 
s_add_u32 s24, s4, 64 // (op 900) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 900) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 900) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 900) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 900) 

s_sub_i32 s22, s24, s1 // (op 900) 

s_mul_i32 s24, s23, s22 // (op 900) 


s_sub_u32 s25, s19, s24 // (op 900) 


s_add_u32 s19, s25, s1 // (op 900) 


s_cmp_eq_i32 s18, 0 // (op 900) 
s_mov_b32 s1, scc // (op 900) 

s_cmp_eq_i32 s1, 0 // (op 900) 
s_cselect_b32 s18, s19, s0 // (op 900) 



s_lshl_b32 s19, s18, 6 // (op 900) 

s_mov_b32 s1, 0 // (op 900) convert
s_mov_b32 s0, s19 // (op 900) convert

s_ashr_i64 s[18:19], s[30:31], 7 // (op 900) 
s_lshr_b32 s22, s18, 1 // (op 900) 

v_lshl_add_u32 v47, s22, 8, v1 // (op 900) 

v_mul_lo_u32 v13, v47, s27 // (op 900) most significant: low of low * high
v_mul_hi_u32 v51, v47, s26 // (op 900) most significant: high of low * low
v_mul_lo_u32 v12, v47, s26 // (op 900) least significant: low of low * low
v_add_u32 v13, v13, v51 // (op 900) most significant: sum


v_mov_b32 v47, s0 // (op 900) 
v_mov_b32 v51, s1 // (op 900) 
v_add_co_u32 v14, s[18:19], v47, v12 // (op 900) least significant half
v_addc_co_u32 v15, s[18:19], v51, v13, s[18:19] // (op 900) most significant half





v_mul_lo_u32 v13, v15, 6 // (op 900) most significant: low of high * low
v_mul_hi_u32 v47, v14, 6 // (op 900) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 900) least significant: low of low * low
v_add_u32 v13, v13, v47 // (op 900) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 900) 

v_mov_b32 v47, v14 // (op 900) convert

s_mul_i32 s1, s31, 6 // (op 902) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 902) least significant
s_mul_hi_u32 s18, s30, 6 // (op 902) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 902) most significant: sum

s_mul_i32 s19, s26, s1 // (op 902) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 902) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 902) least significant
s_mul_hi_u32 s23, s26, s0 // (op 902) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 902) most significant: sum
s_add_u32 s19, s19, s23 // (op 902) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 902) 

s_mov_b32 s52, s28 // (op 902) reg expression
s_mov_b32 s53, s29 // (op 902) reg expression
s_mov_b32 s54, s0 // (op 902) reg expression
s_mov_b32 s55, 131072 // (op 902) call()

s_mul_hi_i32 s0, s16, s10 // (op 968) 
s_add_i32 s1, s0, s16 // (op 968) 

s_ashr_i32 s0, s1, 31 // (op 968) 
s_cmp_eq_i32 s10, 0 // (op 968) 
s_mov_b32 s18, scc // (op 968) 
s_cmp_eq_i32 s18, 0 // (op 968) 
s_cselect_b32 s19, 0, -1 // (op 968) 

s_and_b32 s18, s13, 31 // (op 968) 
s_lshl_b32 s22, 1, s18 // (op 968) 
s_add_i32 s23, s22, s19 // (op 968) 


s_and_b32 s19, s0, s23 // (op 968) 


s_add_i32 s0, s1, s19 // (op 968) 


s_ashr_i32 s1, s0, s18 // (op 968) 


s_lshl_b32 s0, s13, 24 // (op 968) 
s_ashr_i32 s18, s0, 31 // (op 968) 

s_xor_b32 s0, s1, s18 // (op 968) 

s_sub_i32 s1, s0, s18 // (op 968) 


s_mul_hi_i32 s0, s1, s14 // (op 968) 
s_add_i32 s18, s0, s1 // (op 968) 

s_ashr_i32 s0, s18, 31 // (op 968) 
s_cmp_eq_i32 s14, 0 // (op 968) 
s_mov_b32 s19, scc // (op 968) 
s_cmp_eq_i32 s19, 0 // (op 968) 
s_cselect_b32 s22, 0, -1 // (op 968) 

s_and_b32 s19, s15, 31 // (op 968) 
s_lshl_b32 s23, 1, s19 // (op 968) 
s_add_i32 s24, s23, s22 // (op 968) 


s_and_b32 s25, s0, s24 // (op 968) 


s_add_i32 s0, s18, s25 // (op 968) 


s_ashr_i32 s18, s0, s19 // (op 968) 


s_lshl_b32 s0, s15, 24 // (op 968) 
s_ashr_i32 s19, s0, 31 // (op 968) 

s_xor_b32 s0, s18, s19 // (op 968) 

s_sub_i32 s18, s0, s19 // (op 968) 


s_add_u32 s50, s6, 16 // (op 968) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 968) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 968) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 968) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 968) 

s_mul_i32 s0, s18, s24 // (op 968) 

s_sub_i32 s18, s1, s0 // (op 968) 


s_mul_hi_i32 s19, s16, s9 // (op 968) 
s_add_i32 s0, s19, s16 // (op 968) 

s_ashr_i32 s1, s0, 31 // (op 968) 
s_cmp_eq_i32 s9, 0 // (op 968) 
s_mov_b32 s19, scc // (op 968) 
s_cmp_eq_i32 s19, 0 // (op 968) 
s_cselect_b32 s22, 0, -1 // (op 968) 

s_and_b32 s19, s47, 31 // (op 968) 
s_lshl_b32 s23, 1, s19 // (op 968) 
s_add_i32 s49, s23, s22 // (op 968) 


s_and_b32 s22, s1, s49 // (op 968) 


s_add_i32 s1, s0, s22 // (op 968) 


s_ashr_i32 s0, s1, s19 // (op 968) 


s_lshl_b32 s1, s47, 24 // (op 968) 
s_ashr_i32 s19, s1, 31 // (op 968) 

s_xor_b32 s1, s0, s19 // (op 968) 

s_sub_i32 s0, s1, s19 // (op 968) 


s_mul_i32 s1, s11, s24 // (op 968) 

s_mul_i32 s19, s8, s1 // (op 968) 

s_mul_i32 s1, s0, s19 // (op 968) 

s_sub_u32 s19, s16, s1 // (op 968) 

s_mul_hi_i32 s1, s19, s12 // (op 968) 
s_add_i32 s22, s1, s19 // (op 968) 


s_ashr_i32 s1, s22, 31 // (op 968) 
s_cmp_eq_i32 s12, 0 // (op 968) 
s_mov_b32 s19, scc // (op 968) 
s_cmp_eq_i32 s19, 0 // (op 968) 
s_cselect_b32 s23, 0, -1 // (op 968) 

s_and_b32 s19, s46, 31 // (op 968) 
s_lshl_b32 s24, 1, s19 // (op 968) 
s_add_i32 s25, s24, s23 // (op 968) 


s_and_b32 s23, s1, s25 // (op 968) 


s_add_i32 s1, s22, s23 // (op 968) 


s_ashr_i32 s22, s1, s19 // (op 968) 


s_lshl_b32 s1, s46, 24 // (op 968) 
s_ashr_i32 s19, s1, 31 // (op 968) 

s_xor_b32 s1, s22, s19 // (op 968) 

s_sub_i32 s22, s1, s19 // (op 968) 


s_cmp_eq_i32 s0, 0 // (op 968) 
s_mov_b32 s1, scc // (op 968) 

s_cmp_eq_i32 s1, 0 // (op 968) 
s_cselect_b32 s0, s22, s18 // (op 968) 



s_lshl_b32 s1, s0, 4 // (op 968) 

s_mov_b32 s19, 0 // (op 968) convert
s_mov_b32 s18, s1 // (op 968) convert

s_ashr_i64 s[0:1], s[30:31], 7 // (op 968) 
s_lshr_b32 s22, s0, 1 // (op 968) 

v_lshl_add_u32 v51, s22, 8, v1 // (op 968) 

v_mul_lo_u32 v13, v51, s33 // (op 968) most significant: low of low * high
v_mul_hi_u32 v55, v51, s32 // (op 968) most significant: high of low * low
v_mul_lo_u32 v12, v51, s32 // (op 968) least significant: low of low * low
v_add_u32 v13, v13, v55 // (op 968) most significant: sum

v_mov_b32 v55, s18 // (op 968) 
v_mov_b32 v14, s19 // (op 968) 
v_add_co_u32 v12, s[0:1], v12, v55 // (op 968) least significant half
v_addc_co_u32 v13, s[0:1], v13, v14, s[0:1] // (op 968) most significant half





s_mul_i32 s1, s32, s31 // (op 970) most significant: low of low * high
s_mul_i32 s18, s33, s30 // (op 970) most significant: low of high * low
s_mul_i32 s0, s32, s30 // (op 970) least significant
s_mul_hi_u32 s19, s32, s30 // (op 970) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 970) most significant: sum
s_add_u32 s1, s1, s19 // (op 970) most significant: sum


s_mov_b32 s64, s34 // (op 970) reg expression
s_mov_b32 s65, s35 // (op 970) reg expression
s_mov_b32 s66, s0 // (op 970) reg expression
s_mov_b32 s67, 131072 // (op 970) call()

s_mul_hi_i32 s0, s16, s10 // (op 1032) 
v_and_b32 v51, -64, v1 // (op 1032) 
s_add_i32 s1, s0, s16 // (op 1032) 

v_sub_u32 v55, v1, v51 // (op 1032) 

s_ashr_i32 s0, s1, 31 // (op 1032) 
v_lshrrev_b32 v51, 6, v1 // (op 1032) 
s_cmp_eq_i32 s10, 0 // (op 1032) 
s_mov_b32 s18, scc // (op 1032) 
v_lshlrev_b32 v14, 1, v51 // (op 1032) 

s_cmp_eq_i32 s18, 0 // (op 1032) 
s_cselect_b32 s19, 0, -1 // (op 1032) 

s_and_b32 s18, s13, 31 // (op 1032) 
s_lshl_b32 s22, 1, s18 // (op 1032) 
s_add_i32 s23, s22, s19 // (op 1032) 


s_and_b32 s19, s0, s23 // (op 1032) 


s_add_i32 s0, s1, s19 // (op 1032) 


s_ashr_i32 s1, s0, s18 // (op 1032) 


s_lshl_b32 s0, s13, 24 // (op 1032) 
s_ashr_i32 s18, s0, 31 // (op 1032) 

s_xor_b32 s0, s1, s18 // (op 1032) 

s_sub_i32 s1, s0, s18 // (op 1032) 


s_mul_hi_i32 s0, s1, s14 // (op 1032) 
s_add_i32 s18, s0, s1 // (op 1032) 

s_ashr_i32 s0, s18, 31 // (op 1032) 
s_cmp_eq_i32 s14, 0 // (op 1032) 
s_mov_b32 s19, scc // (op 1032) 
s_cmp_eq_i32 s19, 0 // (op 1032) 
s_cselect_b32 s22, 0, -1 // (op 1032) 

s_and_b32 s19, s15, 31 // (op 1032) 
s_lshl_b32 s23, 1, s19 // (op 1032) 
s_add_i32 s24, s23, s22 // (op 1032) 


s_and_b32 s25, s0, s24 // (op 1032) 


s_add_i32 s0, s18, s25 // (op 1032) 


s_ashr_i32 s18, s0, s19 // (op 1032) 


s_lshl_b32 s0, s15, 24 // (op 1032) 
s_ashr_i32 s19, s0, 31 // (op 1032) 

s_xor_b32 s0, s18, s19 // (op 1032) 

s_sub_i32 s18, s0, s19 // (op 1032) 


s_mul_i32 s0, s1, s11 // (op 1032) 

s_sub_u32 s1, s16, s0 // (op 1032) 

s_mul_i32 s0, s18, s11 // (op 1032) 
s_add_u32 s0, s0, s1 // (op 1032) 


s_mul_hi_i32 s1, s16, s9 // (op 1032) 
s_add_i32 s18, s1, s16 // (op 1032) 

s_ashr_i32 s1, s18, 31 // (op 1032) 
s_cmp_eq_i32 s9, 0 // (op 1032) 
s_mov_b32 s19, scc // (op 1032) 
s_cmp_eq_i32 s19, 0 // (op 1032) 
s_cselect_b32 s22, 0, -1 // (op 1032) 

s_and_b32 s19, s47, 31 // (op 1032) 
s_lshl_b32 s23, 1, s19 // (op 1032) 
s_add_i32 s24, s23, s22 // (op 1032) 


s_and_b32 s25, s1, s24 // (op 1032) 


s_add_i32 s1, s18, s25 // (op 1032) 


s_ashr_i32 s18, s1, s19 // (op 1032) 


s_lshl_b32 s1, s47, 24 // (op 1032) 
s_ashr_i32 s19, s1, 31 // (op 1032) 

s_xor_b32 s1, s18, s19 // (op 1032) 

s_sub_i32 s18, s1, s19 // (op 1032) 


s_add_u32 s50, s6, 16 // (op 1032) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1032) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1032) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1032) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1032) 

s_mul_i32 s1, s11, s24 // (op 1032) 

s_mul_i32 s19, s8, s1 // (op 1032) 

s_mul_i32 s1, s18, s19 // (op 1032) 

s_sub_u32 s19, s16, s1 // (op 1032) 

s_mov_b32 s1, s19 // (op 1032) 
s_mul_hi_i32 s22, s1, s12 // (op 1032) 
s_add_i32 s23, s22, s1 // (op 1032) 


s_ashr_i32 s1, s23, 31 // (op 1032) 
s_cmp_eq_i32 s12, 0 // (op 1032) 
s_mov_b32 s22, scc // (op 1032) 
s_cmp_eq_i32 s22, 0 // (op 1032) 
s_cselect_b32 s24, 0, -1 // (op 1032) 

s_and_b32 s22, s46, 31 // (op 1032) 
s_lshl_b32 s25, 1, s22 // (op 1032) 
s_add_i32 s49, s25, s24 // (op 1032) 


s_and_b32 s24, s1, s49 // (op 1032) 


s_add_i32 s1, s23, s24 // (op 1032) 


s_ashr_i32 s23, s1, s22 // (op 1032) 


s_lshl_b32 s1, s46, 24 // (op 1032) 
s_ashr_i32 s22, s1, 31 // (op 1032) 

s_xor_b32 s1, s23, s22 // (op 1032) 

s_sub_i32 s23, s1, s22 // (op 1032) 


s_mul_i32 s1, s48, s11 // (op 1032) 
s_add_u32 s24, s4, 64 // (op 1032) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1032) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1032) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1032) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1032) 

s_sub_i32 s22, s24, s1 // (op 1032) 

s_mul_i32 s24, s23, s22 // (op 1032) 


s_sub_u32 s25, s19, s24 // (op 1032) 


s_add_u32 s19, s25, s1 // (op 1032) 


s_cmp_eq_i32 s18, 0 // (op 1032) 
s_mov_b32 s1, scc // (op 1032) 

s_cmp_eq_i32 s1, 0 // (op 1032) 
s_cselect_b32 s18, s19, s0 // (op 1032) 



v_lshl_add_u32 v51, s18, 6, v55 // (op 1032) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1032) 
s_lshr_b32 s18, s0, 1 // (op 1032) 

v_lshl_add_u32 v55, s18, 3, v14 // (op 1032) 


v_mul_lo_u32 v14, v55, s36 // (op 1032) 

v_add_u32 v55, v51, v14 // (op 1032) 


s_mul_i32 s1, s36, s41 // (op 1034) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1034) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1034) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1034) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1034) most significant: sum
s_add_u32 s1, s1, s19 // (op 1034) most significant: sum


s_mov_b32 s68, s38 // (op 1034) reg expression
s_mov_b32 s69, s39 // (op 1034) reg expression
s_mov_b32 s70, s0 // (op 1034) reg expression
s_mov_b32 s71, 131072 // (op 1034) call()

s_mul_hi_i32 s0, s16, s10 // (op 1096) 
v_and_b32 v51, 15, v1 // (op 1096) 
s_add_i32 s1, s0, s16 // (op 1096) 

v_and_b32 v14, 63, v1 // (op 1096) 
s_ashr_i32 s0, s1, 31 // (op 1096) 
v_lshrrev_b32 v15, 4, v14 // (op 1096) 

s_cmp_eq_i32 s10, 0 // (op 1096) 
s_mov_b32 s18, scc // (op 1096) 
s_cmp_eq_i32 s18, 0 // (op 1096) 
s_cselect_b32 s19, 0, -1 // (op 1096) 

s_and_b32 s18, s13, 31 // (op 1096) 
s_lshl_b32 s22, 1, s18 // (op 1096) 
s_add_i32 s23, s22, s19 // (op 1096) 


s_and_b32 s19, s0, s23 // (op 1096) 


s_add_i32 s0, s1, s19 // (op 1096) 


s_ashr_i32 s1, s0, s18 // (op 1096) 


s_lshl_b32 s0, s13, 24 // (op 1096) 
s_ashr_i32 s18, s0, 31 // (op 1096) 

s_xor_b32 s0, s1, s18 // (op 1096) 

s_sub_i32 s1, s0, s18 // (op 1096) 


s_mul_hi_i32 s0, s1, s14 // (op 1096) 
s_add_i32 s18, s0, s1 // (op 1096) 

s_ashr_i32 s0, s18, 31 // (op 1096) 
s_cmp_eq_i32 s14, 0 // (op 1096) 
s_mov_b32 s19, scc // (op 1096) 
s_cmp_eq_i32 s19, 0 // (op 1096) 
s_cselect_b32 s22, 0, -1 // (op 1096) 

s_and_b32 s19, s15, 31 // (op 1096) 
s_lshl_b32 s23, 1, s19 // (op 1096) 
s_add_i32 s24, s23, s22 // (op 1096) 


s_and_b32 s25, s0, s24 // (op 1096) 


s_add_i32 s0, s18, s25 // (op 1096) 


s_ashr_i32 s18, s0, s19 // (op 1096) 


s_lshl_b32 s0, s15, 24 // (op 1096) 
s_ashr_i32 s19, s0, 31 // (op 1096) 

s_xor_b32 s0, s18, s19 // (op 1096) 

s_sub_i32 s18, s0, s19 // (op 1096) 


s_add_u32 s50, s6, 16 // (op 1096) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1096) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1096) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1096) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1096) 

s_mul_i32 s0, s18, s24 // (op 1096) 

s_sub_i32 s18, s1, s0 // (op 1096) 


s_mul_hi_i32 s19, s16, s9 // (op 1096) 
s_add_i32 s0, s19, s16 // (op 1096) 

s_ashr_i32 s1, s0, 31 // (op 1096) 
s_cmp_eq_i32 s9, 0 // (op 1096) 
s_mov_b32 s19, scc // (op 1096) 
s_cmp_eq_i32 s19, 0 // (op 1096) 
s_cselect_b32 s22, 0, -1 // (op 1096) 

s_and_b32 s19, s47, 31 // (op 1096) 
s_lshl_b32 s23, 1, s19 // (op 1096) 
s_add_i32 s49, s23, s22 // (op 1096) 


s_and_b32 s22, s1, s49 // (op 1096) 


s_add_i32 s1, s0, s22 // (op 1096) 


s_ashr_i32 s0, s1, s19 // (op 1096) 


s_lshl_b32 s1, s47, 24 // (op 1096) 
s_ashr_i32 s19, s1, 31 // (op 1096) 

s_xor_b32 s1, s0, s19 // (op 1096) 

s_sub_i32 s0, s1, s19 // (op 1096) 


s_mul_i32 s1, s11, s24 // (op 1096) 

s_mul_i32 s19, s8, s1 // (op 1096) 

s_mul_i32 s1, s0, s19 // (op 1096) 

s_sub_u32 s19, s16, s1 // (op 1096) 

s_mul_hi_i32 s1, s19, s12 // (op 1096) 
s_add_i32 s22, s1, s19 // (op 1096) 


s_ashr_i32 s1, s22, 31 // (op 1096) 
s_cmp_eq_i32 s12, 0 // (op 1096) 
s_mov_b32 s19, scc // (op 1096) 
s_cmp_eq_i32 s19, 0 // (op 1096) 
s_cselect_b32 s23, 0, -1 // (op 1096) 

s_and_b32 s19, s46, 31 // (op 1096) 
s_lshl_b32 s24, 1, s19 // (op 1096) 
s_add_i32 s25, s24, s23 // (op 1096) 


s_and_b32 s23, s1, s25 // (op 1096) 


s_add_i32 s1, s22, s23 // (op 1096) 


s_ashr_i32 s22, s1, s19 // (op 1096) 


s_lshl_b32 s1, s46, 24 // (op 1096) 
s_ashr_i32 s19, s1, 31 // (op 1096) 

s_xor_b32 s1, s22, s19 // (op 1096) 

s_sub_i32 s22, s1, s19 // (op 1096) 


s_cmp_eq_i32 s0, 0 // (op 1096) 
s_mov_b32 s1, scc // (op 1096) 

s_cmp_eq_i32 s1, 0 // (op 1096) 
s_cselect_b32 s0, s22, s18 // (op 1096) 



v_lshl_add_u32 v14, s0, 4, v51 // (op 1096) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1096) 
s_lshr_b32 s18, s0, 1 // (op 1096) 

v_lshl_add_u32 v51, s18, 3, v15 // (op 1096) 


v_mul_lo_u32 v15, v51, s42 // (op 1096) 
v_add_u32 v15, v15, v14 // (op 1096) 


s_mul_i32 s1, s42, s41 // (op 1098) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1098) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1098) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1098) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1098) most significant: sum
s_add_u32 s1, s1, s19 // (op 1098) most significant: sum


s_mov_b32 s72, s44 // (op 1098) reg expression
s_mov_b32 s73, s45 // (op 1098) reg expression
s_mov_b32 s74, s0 // (op 1098) reg expression
s_mov_b32 s75, 131072 // (op 1098) call()

v_lshlrev_b32 v14, 6, v1 // (op 1189) 
v_mul_lo_u32 v51, v14, 6 // (op 1189) 
v_lshrrev_b32 v16, 3, v51 // (op 1189) 

v_lshrrev_b32 v51, 4, v14 // (op 1189) 

v_lshlrev_b32 v14, 2, v51 // (op 1189) 

v_add_u32 v51, v16, v14 // (op 1189) 


v_lshlrev_b32 v14, 4, v1 // (op 1247) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 1247) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -64, v1 // (op 1299) 
v_sub_u32 v16, v1, v14 // (op 1299) 

v_lshrrev_b32 v14, 6, v1 // (op 1299) 
v_lshl_add_u32 v17, v14, 7, v16 // (op 1299) 


s_ashr_i64 s[18:19], s[30:31], 7 // (op 316) 
s_and_b32 s1, s18, 4294967294 // (op 316) 

s_ashr_i64 s[18:19], s[30:31], 7 // (op 315) 
s_mov_b32 s22, s18 // (op 315) convert

s_cmp_lt_u32 s1, s22 // (op 315) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_315:


s_add_u32 s1, s1, 1 // (op 317) 
v_mov_b32 v14, v47 // (op 166) 
v_mov_b32 v16, v14 // (op 166) reg expression

buffer_load_dwordx3 v[18:20], v16, s[52:55], 0 offen // (op 166) Load value
buffer_load_dwordx3 v[22:24], v16, s[52:55], 0 offen offset:12 // (op 166) Load value
buffer_load_dwordx3 v[26:28], v16, s[52:55], 0 offen offset:24 // (op 166) Load value
buffer_load_dwordx3 v[30:32], v16, s[52:55], 0 offen offset:36 // (op 166) Load value


s_lshl_b64 s[18:19], s[26:27], 7 // (op 890) 
s_mul_i32 s51, s19, 6 // (op 890) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 890) least significant
s_mul_hi_u32 s49, s18, 6 // (op 890) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 890) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 890) 

v_add_u32 v47, v47, s18 // (op 890) 

s_barrier  // (op 1494) 
s_barrier  // (op 1507) 





v_mov_b64 v[48:49], v[12:13] // (op 870) 
v_mov_b64 v[52:53], v[48:49] // (op 870) reg expression

s_mov_b32 m0, s0 // (op 870) reg expression
buffer_load_dwordx4 v52, s[64:67], 0 offen offset: 0 lds // (op 870) Load value direct to lds


s_lshl_b64 s[18:19], s[32:33], 7 // (op 960) 
v_mov_b32 v14, s18 // (op 960) 
v_mov_b32 v16, s19 // (op 960) 
v_add_co_u32 v12, s[50:51], v12, v14 // (op 960) least significant half
v_addc_co_u32 v13, s[50:51], v13, v16, s[50:51] // (op 960) most significant half




s_waitcnt vmcnt(4)
ds_write_b96 v51, v[18:20] offset:4096 // (op 167) Store local data 

v_mov_b64 v[48:49], s[36:37] // (op 180) convert

v_mov_b32 v14, v55 // (op 180) 
v_mov_b32 v16, v14 // (op 180) reg expression
buffer_load_ubyte v52, v16, s[68:71], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v16, v16, v48 // (op 180) 
s_waitcnt vmcnt(4)
ds_write_b96 v51, v[22:24] offset:4112 // (op 167) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v51, v[26:28] offset:4128 // (op 167) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v51, v[30:32] offset:4144 // (op 167) Store local data 
buffer_load_ubyte v53, v16, s[68:71], 0 offen offset: 0 // (op 180) Load value




s_lshl_b32 s18, s36, 2 // (op 1024) 
v_add_u32 v55, v55, s18 // (op 1024) 

s_barrier  // (op 1520) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1497) 






ds_read_b96_tr_b6 v[18:20], v4 offset:4096 // (op 169) Transpose load local data 
ds_read_b96_tr_b6 v[21:23], v4 offset:5120 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v17, v52 offset:12288 // (op 178) Store local data 
s_barrier  // (op 1510) 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v53 offset:12352 // (op 178) Store local data 




ds_read_b64_tr_b8 v[24:25], v0 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:128 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:1024 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:1152 // (op 171) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1524) 


ds_read_u8 v14, v7 offset:12288 // (op 173) Load local data 

v_mov_b64 v[32:33], s[42:43] // (op 175) convert

v_mov_b32 v16, v15 // (op 175) 
v_mov_b32 v34, v16 // (op 175) reg expression
v_mov_b32 v35, v34 // (op 175) 
buffer_load_ubyte v36, v35, s[72:75], 0 offen offset: 0 // (op 175) Load value




s_lshl_b32 s18, s42, 2 // (op 1088) 
v_add_u32 v15, v15, s18 // (op 1088) 




s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:31], a[0:3], v14, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 176) 
ds_read_b96_tr_b6 v[38:40], v4 offset:4128 // (op 321) Transpose load local data 


ds_read_b96_tr_b6 v[41:43], v4 offset:5152 // (op 321) Transpose load local data 


ds_read_u8 v14, v7 offset:12320 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:43], v[24:31], a[4:7], v14, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 327) 




s_ashr_i64 s[18:19], s[30:31], 7 // (op 315) 
s_mov_b32 s22, s18 // (op 315) convert

s_cmp_lt_u32 s1, s22 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_315 if true)
RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_315:






















s_barrier  // (op 1592) 



















v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 




buffer_store_dwordx4 v[14:17], v5, s[60:63], 0 offen // (op 52) Store value

v_accvgpr_read v10, a4 // (op 513) 
v_accvgpr_read v11, a5 // (op 513) 
v_accvgpr_read v12, a6 // (op 513) 
v_accvgpr_read v13, a7 // (op 513) 
v_mul_f32 v14, v9, v10 // (op 513) 
v_mul_f32 v15, v9, v11 // (op 513) 
v_mul_f32 v16, v9, v12 // (op 513) 
v_mul_f32 v17, v9, v13 // (op 513) 




buffer_store_dwordx4 v[14:17], v5, s[60:63], 0 offen offset:128 // (op 515) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:128 // (op 543) Load value
v_accvgpr_read v14, a4 // (op 540) 
v_accvgpr_read v15, a5 // (op 540) 
v_accvgpr_read v16, a6 // (op 540) 
v_accvgpr_read v17, a7 // (op 540) 
v_mul_f32 v18, v9, v14 // (op 540) 
v_mul_f32 v19, v9, v15 // (op 540) 
v_mul_f32 v20, v9, v16 // (op 540) 
v_mul_f32 v21, v9, v17 // (op 540) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 544) 
v_mul_f32 v15, v8, v11 // (op 544) 
v_mul_f32 v16, v8, v12 // (op 544) 
v_mul_f32 v17, v8, v13 // (op 544) 

v_add_f32 v10, v14, v18 // (op 541) 
v_add_f32 v11, v15, v19 // (op 541) 
v_add_f32 v12, v16, v20 // (op 541) 
v_add_f32 v13, v17, v21 // (op 541) 





buffer_store_dwordx4 v[10:13], v5, s[60:63], 0 offen offset:128 // (op 542) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen // (op 135) Load value
v_accvgpr_read v14, a0 // (op 132) 
v_accvgpr_read v15, a1 // (op 132) 
v_accvgpr_read v16, a2 // (op 132) 
v_accvgpr_read v17, a3 // (op 132) 
v_mul_f32 v18, v9, v14 // (op 132) 
v_mul_f32 v19, v9, v15 // (op 132) 
v_mul_f32 v20, v9, v16 // (op 132) 
v_mul_f32 v21, v9, v17 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 136) 
v_mul_f32 v15, v8, v11 // (op 136) 
v_mul_f32 v16, v8, v12 // (op 136) 
v_mul_f32 v17, v8, v13 // (op 136) 

v_add_f32 v10, v14, v18 // (op 133) 
v_add_f32 v11, v15, v19 // (op 133) 
v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 





buffer_store_dwordx4 v[10:13], v5, s[60:63], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.LRR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__end:
.size RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_, .LRR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__end-RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 88
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 80
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
    DataTypeB: F8
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
    macrotile: [64, 16, 128]
    threads: [128, 1, 1]
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
  - .name: RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 80
    .agpr_count: 8
    .max_flat_workgroup_size: 128
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

