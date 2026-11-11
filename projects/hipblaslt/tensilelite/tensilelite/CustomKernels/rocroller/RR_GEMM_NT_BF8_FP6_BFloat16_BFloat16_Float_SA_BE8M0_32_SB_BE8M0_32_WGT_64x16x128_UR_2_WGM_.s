// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_,@function
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1167) 
s_mul_hi_i32 s1, s16, s10 // (op 1355) 
v_and_b32 v0, 63, v1 // (op 1281) 
v_and_b32 v3, 48, v1 // (op 1281) 
v_sub_u32 v4, v0, v3 // (op 1281) 

v_lshrrev_b32 v3, 4, v0 // (op 1281) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1281) 


v_lshlrev_b32 v3, 4, v0 // (op 1281) 
v_mul_lo_u32 v4, v3, 6 // (op 1281) 

v_lshrrev_b32 v3, 3, v4 // (op 1281) 

v_and_b32 v4, 268435455, v0 // (op 1281) 

v_lshlrev_b32 v0, 2, v4 // (op 1281) 

v_add_u32 v4, v3, v0 // (op 1281) 


v_and_b32 v0, 63, v1 // (op 1425) 
v_and_b32 v3, 48, v1 // (op 1425) 
v_sub_u32 v5, v0, v3 // (op 1425) 

v_lshrrev_b32 v3, 4, v0 // (op 1425) 

v_lshl_add_u32 v0, v3, 5, v5 // (op 1425) 


v_lshlrev_b32 v3, 4, v0 // (op 1425) 
v_mul_lo_u32 v5, v3, 6 // (op 1425) 

v_lshrrev_b32 v3, 3, v5 // (op 1425) 

v_and_b32 v5, 268435455, v0 // (op 1425) 

v_lshlrev_b32 v0, 2, v5 // (op 1425) 

v_add_u32 v5, v3, v0 // (op 1425) 


v_lshrrev_b32 v0, 6, v1 // (op 1167) 
s_add_i32 s18, s0, s16 // (op 1167) 

v_and_b32 v3, 63, v1 // (op 1167) 
s_ashr_i32 s0, s18, 31 // (op 1167) 
v_lshrrev_b32 v6, 4, v3 // (op 1167) 
s_cmp_eq_i32 s10, 0 // (op 1167) 
s_mov_b32 s19, scc // (op 1167) 
v_lshlrev_b32 v7, 2, v6 // (op 1167) 

s_cmp_eq_i32 s19, 0 // (op 1167) 
s_cselect_b32 s51, 0, -1 // (op 1167) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1167) 


s_and_b32 s19, s13, 31 // (op 1167) 
v_lshrrev_b32 v0, 2, v3 // (op 1167) 

v_and_b32 v3, 3, v0 // (op 1167) 

v_and_b32 v0, 48, v1 // (op 1228) 
v_and_b32 v7, 63, v1 // (op 1228) 
v_sub_u32 v8, v7, v0 // (op 1228) 

v_lshrrev_b32 v7, 1, v8 // (op 1228) 
v_add_lshl_u32 v9, v0, v7, 6 // (op 1228) 


v_and_b32 v0, 1, v8 // (op 1228) 

v_lshlrev_b32 v7, 3, v0 // (op 1228) 

v_lshrrev_b32 v0, 6, v1 // (op 1228) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 1228) 


v_add_u32 v0, v9, v8 // (op 1228) 


v_lshrrev_b32 v7, 6, v1 // (op 1339) 
v_and_b32 v8, 15, v1 // (op 1339) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 1339) 


v_and_b32 v7, 63, v1 // (op 1339) 
v_lshrrev_b32 v8, 4, v7 // (op 1339) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 1339) 


v_lshrrev_b32 v8, 6, v1 // (op 1355) 
v_and_b32 v9, 63, v1 // (op 1355) 
v_and_b32 v10, 48, v1 // (op 1401) 
v_and_b32 v11, 63, v1 // (op 1401) 
v_sub_u32 v12, v11, v10 // (op 1401) 

v_lshrrev_b32 v11, 1, v12 // (op 1401) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 1401) 


v_and_b32 v10, 1, v12 // (op 1401) 

v_lshlrev_b32 v11, 3, v10 // (op 1401) 

v_lshrrev_b32 v10, 6, v1 // (op 1401) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 1401) 


v_add_u32 v10, v13, v12 // (op 1401) 


v_lshrrev_b32 v11, 6, v1 // (op 1454) 
v_and_b32 v12, 15, v1 // (op 1454) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1454) 


v_and_b32 v11, 63, v1 // (op 1454) 
v_lshrrev_b32 v12, 4, v11 // (op 1454) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 1454) 


s_lshl_b32 s52, 1, s19 // (op 1167) 
v_and_b32 v12, 3, v1 // (op 1167) 
s_add_i32 s53, s52, s51 // (op 1167) 


s_and_b32 s51, s0, s53 // (op 1167) 


s_add_i32 s0, s1, s16 // (op 1355) 

s_ashr_i32 s1, s0, 31 // (op 1355) 
v_lshl_add_u32 v13, v3, 2, v12 // (op 1167) 


s_add_i32 s52, s18, s51 // (op 1167) 


s_ashr_i32 s18, s52, s19 // (op 1167) 


s_lshl_b32 s19, s13, 24 // (op 1167) 
s_ashr_i32 s51, s19, 31 // (op 1167) 

s_xor_b32 s19, s18, s51 // (op 1167) 

s_sub_i32 s18, s19, s51 // (op 1167) 


s_mul_hi_i32 s19, s18, s14 // (op 1167) 
s_add_i32 s51, s19, s18 // (op 1167) 

s_ashr_i32 s19, s51, 31 // (op 1167) 
s_cmp_eq_i32 s14, 0 // (op 1167) 
s_mov_b32 s52, scc // (op 1167) 
s_cmp_eq_i32 s52, 0 // (op 1167) 
s_cselect_b32 s53, 0, -1 // (op 1167) 

s_and_b32 s52, s15, 31 // (op 1167) 
v_lshrrev_b32 v3, 4, v9 // (op 1355) 
s_cmp_eq_i32 s10, 0 // (op 1355) 
s_mov_b32 s54, scc // (op 1355) 
v_lshlrev_b32 v12, 2, v3 // (op 1355) 

s_cmp_eq_i32 s54, 0 // (op 1355) 
s_cselect_b32 s55, 0, -1 // (op 1355) 

s_lshl_b32 s54, 1, s52 // (op 1167) 
v_lshl_add_u32 v3, v8, 4, v12 // (op 1355) 


v_lshrrev_b32 v8, 2, v9 // (op 1355) 

s_add_i32 s56, s54, s53 // (op 1167) 


s_and_b32 s53, s19, s56 // (op 1167) 


s_add_i32 s19, s51, s53 // (op 1167) 


s_ashr_i32 s51, s19, s52 // (op 1167) 


s_lshl_b32 s19, s15, 24 // (op 1167) 
s_ashr_i32 s52, s19, 31 // (op 1167) 

s_xor_b32 s19, s51, s52 // (op 1167) 

s_sub_i32 s51, s19, s52 // (op 1167) 


s_mul_i32 s19, s18, s11 // (op 1167) 
s_sub_u32 s52, s16, s19 // (op 1167) 

s_mul_i32 s19, s51, s11 // (op 1167) 
s_add_u32 s19, s19, s52 // (op 1167) 

s_mul_hi_i32 s52, s16, s9 // (op 1167) 
s_add_i32 s53, s52, s16 // (op 1167) 

s_ashr_i32 s52, s53, 31 // (op 1167) 
s_cmp_eq_i32 s9, 0 // (op 1167) 
s_mov_b32 s54, scc // (op 1167) 
s_cmp_eq_i32 s54, 0 // (op 1167) 
s_cselect_b32 s56, 0, -1 // (op 1167) 

s_and_b32 s54, s13, 31 // (op 1355) 
v_and_b32 v9, 3, v8 // (op 1355) 

s_lshl_b32 s57, 1, s54 // (op 1355) 
v_and_b32 v8, 3, v1 // (op 1355) 
s_add_i32 s58, s57, s55 // (op 1355) 


s_and_b32 s55, s1, s58 // (op 1355) 


v_lshl_add_u32 v12, v9, 2, v8 // (op 1355) 


s_add_i32 s1, s0, s55 // (op 1355) 


s_ashr_i32 s0, s1, s54 // (op 1355) 


s_lshl_b32 s1, s13, 24 // (op 1355) 
s_ashr_i32 s54, s1, 31 // (op 1355) 

s_xor_b32 s1, s0, s54 // (op 1355) 

s_sub_i32 s0, s1, s54 // (op 1355) 


s_mul_hi_i32 s1, s0, s14 // (op 1355) 
s_add_i32 s54, s1, s0 // (op 1355) 

s_ashr_i32 s1, s54, 31 // (op 1355) 
s_cmp_eq_i32 s14, 0 // (op 1355) 
s_mov_b32 s55, scc // (op 1355) 
s_cmp_eq_i32 s55, 0 // (op 1355) 
s_cselect_b32 s57, 0, -1 // (op 1355) 

s_and_b32 s55, s15, 31 // (op 1355) 
s_lshl_b32 s58, 1, s55 // (op 1355) 
s_add_i32 s59, s58, s57 // (op 1355) 


s_and_b32 s57, s1, s59 // (op 1355) 


s_add_i32 s1, s54, s57 // (op 1355) 


s_ashr_i32 s54, s1, s55 // (op 1355) 


s_lshl_b32 s1, s15, 24 // (op 1355) 
s_ashr_i32 s55, s1, 31 // (op 1355) 

s_xor_b32 s1, s54, s55 // (op 1355) 

s_sub_i32 s54, s1, s55 // (op 1355) 


s_mul_i32 s1, s0, s11 // (op 1355) 
s_sub_u32 s55, s16, s1 // (op 1355) 

s_mul_i32 s1, s54, s11 // (op 1355) 
s_add_u32 s1, s1, s55 // (op 1355) 

s_mul_hi_i32 s55, s16, s9 // (op 1355) 
s_add_i32 s57, s55, s16 // (op 1355) 

s_ashr_i32 s55, s57, 31 // (op 1355) 
s_cmp_eq_i32 s9, 0 // (op 1355) 
s_mov_b32 s58, scc // (op 1355) 
s_cmp_eq_i32 s58, 0 // (op 1355) 
s_cselect_b32 s59, 0, -1 // (op 1355) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s47, 31 // (op 1167) 
s_lshl_b32 s60, 1, s58 // (op 1167) 
s_add_i32 s61, s60, s56 // (op 1167) 


s_and_b32 s56, s52, s61 // (op 1167) 


s_add_i32 s52, s53, s56 // (op 1167) 


s_ashr_i32 s53, s52, s58 // (op 1167) 


s_lshl_b32 s52, s47, 24 // (op 1167) 
s_ashr_i32 s56, s52, 31 // (op 1167) 

s_xor_b32 s52, s53, s56 // (op 1167) 

s_sub_i32 s53, s52, s56 // (op 1167) 


s_and_b32 s52, s47, 31 // (op 1355) 
s_lshl_b32 s56, 1, s52 // (op 1355) 
s_add_i32 s58, s56, s59 // (op 1355) 


s_and_b32 s56, s55, s58 // (op 1355) 


s_add_i32 s55, s57, s56 // (op 1355) 


s_ashr_i32 s56, s55, s52 // (op 1355) 


s_lshl_b32 s52, s47, 24 // (op 1355) 
s_ashr_i32 s55, s52, 31 // (op 1355) 

s_xor_b32 s52, s56, s55 // (op 1355) 

s_sub_i32 s56, s52, s55 // (op 1355) 


s_add_u32 s58, s6, 16 // (op 1167) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1167) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1167) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1167) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 1167) 

s_mul_i32 s52, s11, s58 // (op 1167) 
s_mul_i32 s55, s8, s52 // (op 1167) 

s_mul_i32 s52, s53, s55 // (op 1167) 

s_sub_u32 s55, s16, s52 // (op 1167) 

s_mov_b32 s52, s55 // (op 1167) 
s_mul_hi_i32 s57, s52, s12 // (op 1167) 
s_add_i32 s60, s57, s52 // (op 1167) 


s_ashr_i32 s52, s60, 31 // (op 1167) 
s_cmp_eq_i32 s12, 0 // (op 1167) 
s_mov_b32 s57, scc // (op 1167) 
s_cmp_eq_i32 s57, 0 // (op 1167) 
s_cselect_b32 s61, 0, -1 // (op 1167) 

s_and_b32 s57, s46, 31 // (op 1167) 
s_lshl_b32 s62, 1, s57 // (op 1167) 
s_add_i32 s63, s62, s61 // (op 1167) 


s_and_b32 s61, s52, s63 // (op 1167) 


s_add_i32 s52, s60, s61 // (op 1167) 


s_ashr_i32 s60, s52, s57 // (op 1167) 


s_lshl_b32 s52, s46, 24 // (op 1167) 
s_ashr_i32 s57, s52, 31 // (op 1167) 

s_xor_b32 s52, s60, s57 // (op 1167) 

s_sub_i32 s60, s52, s57 // (op 1167) 


s_mul_i32 s52, s48, s11 // (op 1167) 
s_add_u32 s62, s4, 64 // (op 1167) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 1167) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1167) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1167) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 1167) 

s_sub_i32 s57, s62, s52 // (op 1167) 

s_mul_i32 s61, s60, s57 // (op 1167) 

s_sub_u32 s57, s55, s61 // (op 1167) 


s_add_u32 s55, s57, s52 // (op 1167) 


s_cmp_eq_i32 s53, 0 // (op 1167) 
s_mov_b32 s52, scc // (op 1167) 

s_cmp_eq_i32 s52, 0 // (op 1167) 
s_cselect_b32 s53, s55, s19 // (op 1167) 


s_mul_i32 s19, s51, s58 // (op 1167) 


v_lshl_add_u32 v8, s53, 6, v6 // (op 1167) 


s_sub_i32 s51, s18, s19 // (op 1167) 


s_cmp_eq_i32 s52, 0 // (op 1167) 
s_cselect_b32 s53, s60, s51 // (op 1167) 



v_lshl_add_u32 v6, s53, 4, v13 // (op 1167) 


v_mul_lo_u32 v9, v6, s2 // (op 1167) 

v_add_lshl_u32 v6, v8, v9, 1 // (op 1167) 


s_add_u32 s18, s6, 16 // (op 1355) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1355) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 1355) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 1355) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 1355) 

s_mul_i32 s55, s11, s18 // (op 1355) 
s_mul_i32 s51, s8, s55 // (op 1355) 

s_mul_i32 s55, s56, s51 // (op 1355) 

s_sub_u32 s51, s16, s55 // (op 1355) 

s_mov_b32 s55, s51 // (op 1355) 
s_mul_hi_i32 s52, s55, s12 // (op 1355) 
s_add_i32 s53, s52, s55 // (op 1355) 


s_ashr_i32 s52, s53, 31 // (op 1355) 
s_cmp_eq_i32 s12, 0 // (op 1355) 
s_mov_b32 s55, scc // (op 1355) 
s_cmp_eq_i32 s55, 0 // (op 1355) 
s_cselect_b32 s57, 0, -1 // (op 1355) 

s_and_b32 s55, s46, 31 // (op 1355) 
s_lshl_b32 s58, 1, s55 // (op 1355) 
s_add_i32 s59, s58, s57 // (op 1355) 


s_and_b32 s57, s52, s59 // (op 1355) 


s_add_i32 s52, s53, s57 // (op 1355) 


s_ashr_i32 s53, s52, s55 // (op 1355) 


s_lshl_b32 s52, s46, 24 // (op 1355) 
s_ashr_i32 s55, s52, 31 // (op 1355) 

s_xor_b32 s52, s53, s55 // (op 1355) 

s_sub_i32 s53, s52, s55 // (op 1355) 


s_mul_i32 s52, s48, s11 // (op 1355) 
s_add_u32 s58, s4, 64 // (op 1355) Start of Int64 add, locking SCC
s_addc_u32 s59, s5, 0 // (op 1355) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1355) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1355) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 1355) 

s_sub_i32 s55, s58, s52 // (op 1355) 

s_mul_i32 s57, s53, s55 // (op 1355) 

s_sub_u32 s55, s51, s57 // (op 1355) 


s_add_u32 s51, s55, s52 // (op 1355) 


s_cmp_eq_i32 s56, 0 // (op 1355) 
s_mov_b32 s52, scc // (op 1355) 

s_cmp_eq_i32 s52, 0 // (op 1355) 
s_cselect_b32 s55, s51, s1 // (op 1355) 


s_mul_i32 s1, s54, s18 // (op 1355) 


v_lshl_add_u32 v8, s55, 6, v3 // (op 1355) 


s_sub_i32 s51, s0, s1 // (op 1355) 


s_cmp_eq_i32 s52, 0 // (op 1355) 
s_cselect_b32 s0, s53, s51 // (op 1355) 



v_lshl_add_u32 v3, s0, 4, v12 // (op 1355) 


v_mul_lo_u32 v9, v3, s20 // (op 1355) 

v_add_lshl_u32 v3, v8, v9, 1 // (op 1355) 


s_mul_i32 s1, s2, s7 // (op 1169) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1169) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1169) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1169) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1169) most significant: sum
s_add_u32 s1, s1, s19 // (op 1169) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1169) 

s_mul_i32 s1, s20, s7 // (op 1357) most significant: low of low * high
s_mul_i32 s51, s21, s6 // (op 1357) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 1357) least significant
s_mul_hi_u32 s52, s20, s6 // (op 1357) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1357) most significant: sum
s_add_u32 s1, s1, s52 // (op 1357) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1357) 

s_mov_b32 s56, s22 // (op 1169) reg expression
s_mov_b32 s57, s23 // (op 1169) reg expression
s_mov_b32 s58, s18 // (op 1169) reg expression
s_mov_b32 s59, 131072 // (op 1169) call()

s_mov_b32 s60, s24 // (op 1357) reg expression

s_mov_b32 s61, s25 // (op 1357) reg expression
s_mov_b32 s62, s52 // (op 1357) reg expression
s_mov_b32 s63, 131072 // (op 1357) call()


v_mov_b32 v8, s49 // (op 2) Move value
v_mov_b32 v9, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
s_mul_hi_i32 s0, s16, s10 // (op 934) 
v_and_b32 v12, -4, v1 // (op 934) 
s_add_i32 s1, s0, s16 // (op 934) 

v_sub_u32 v13, v1, v12 // (op 934) 

s_ashr_i32 s0, s1, 31 // (op 934) 
v_lshlrev_b32 v12, 4, v13 // (op 934) 

s_cmp_eq_i32 s10, 0 // (op 934) 
s_mov_b32 s18, scc // (op 934) 
v_lshrrev_b32 v13, 2, v1 // (op 934) 
s_cmp_eq_i32 s18, 0 // (op 934) 
s_cselect_b32 s19, 0, -1 // (op 934) 

s_and_b32 s18, s13, 31 // (op 934) 
s_lshl_b32 s22, 1, s18 // (op 934) 
s_add_i32 s23, s22, s19 // (op 934) 


s_and_b32 s19, s0, s23 // (op 934) 


s_add_i32 s0, s1, s19 // (op 934) 


s_ashr_i32 s1, s0, s18 // (op 934) 


s_lshl_b32 s0, s13, 24 // (op 934) 
s_ashr_i32 s18, s0, 31 // (op 934) 

s_xor_b32 s0, s1, s18 // (op 934) 

s_sub_i32 s1, s0, s18 // (op 934) 


s_mul_hi_i32 s0, s1, s14 // (op 934) 
s_add_i32 s18, s0, s1 // (op 934) 

s_ashr_i32 s0, s18, 31 // (op 934) 
s_cmp_eq_i32 s14, 0 // (op 934) 
s_mov_b32 s19, scc // (op 934) 
s_cmp_eq_i32 s19, 0 // (op 934) 
s_cselect_b32 s22, 0, -1 // (op 934) 

s_and_b32 s19, s15, 31 // (op 934) 
s_lshl_b32 s23, 1, s19 // (op 934) 
s_add_i32 s24, s23, s22 // (op 934) 


s_and_b32 s25, s0, s24 // (op 934) 


s_add_i32 s0, s18, s25 // (op 934) 


s_ashr_i32 s18, s0, s19 // (op 934) 


s_lshl_b32 s0, s15, 24 // (op 934) 
s_ashr_i32 s19, s0, 31 // (op 934) 

s_xor_b32 s0, s18, s19 // (op 934) 

s_sub_i32 s18, s0, s19 // (op 934) 


s_mul_i32 s0, s1, s11 // (op 934) 

s_sub_u32 s1, s16, s0 // (op 934) 

s_mul_i32 s0, s18, s11 // (op 934) 
s_add_u32 s0, s0, s1 // (op 934) 


s_mul_hi_i32 s1, s16, s9 // (op 934) 
s_add_i32 s18, s1, s16 // (op 934) 

s_ashr_i32 s1, s18, 31 // (op 934) 
s_cmp_eq_i32 s9, 0 // (op 934) 
s_mov_b32 s19, scc // (op 934) 
s_cmp_eq_i32 s19, 0 // (op 934) 
s_cselect_b32 s22, 0, -1 // (op 934) 

s_and_b32 s19, s47, 31 // (op 934) 
s_lshl_b32 s23, 1, s19 // (op 934) 
s_add_i32 s24, s23, s22 // (op 934) 


s_and_b32 s25, s1, s24 // (op 934) 


s_add_i32 s1, s18, s25 // (op 934) 


s_ashr_i32 s18, s1, s19 // (op 934) 


s_lshl_b32 s1, s47, 24 // (op 934) 
s_ashr_i32 s19, s1, 31 // (op 934) 

s_xor_b32 s1, s18, s19 // (op 934) 

s_sub_i32 s18, s1, s19 // (op 934) 


s_add_u32 s22, s6, 16 // (op 934) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 934) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 934) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 934) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 934) 

s_mul_i32 s1, s11, s22 // (op 934) 

s_mul_i32 s19, s8, s1 // (op 934) 

s_mul_i32 s1, s18, s19 // (op 934) 

s_sub_u32 s19, s16, s1 // (op 934) 

s_mov_b32 s1, s19 // (op 934) 
s_mul_hi_i32 s22, s1, s12 // (op 934) 
s_add_i32 s23, s22, s1 // (op 934) 


s_ashr_i32 s1, s23, 31 // (op 934) 
s_cmp_eq_i32 s12, 0 // (op 934) 
s_mov_b32 s22, scc // (op 934) 
s_cmp_eq_i32 s22, 0 // (op 934) 
s_cselect_b32 s24, 0, -1 // (op 934) 

s_and_b32 s22, s46, 31 // (op 934) 
s_lshl_b32 s25, 1, s22 // (op 934) 
s_add_i32 s49, s25, s24 // (op 934) 


s_and_b32 s24, s1, s49 // (op 934) 


s_add_i32 s1, s23, s24 // (op 934) 


s_ashr_i32 s23, s1, s22 // (op 934) 


s_lshl_b32 s1, s46, 24 // (op 934) 
s_ashr_i32 s22, s1, 31 // (op 934) 

s_xor_b32 s1, s23, s22 // (op 934) 

s_sub_i32 s23, s1, s22 // (op 934) 


s_mul_i32 s1, s48, s11 // (op 934) 
s_add_u32 s24, s4, 64 // (op 934) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 934) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 934) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 934) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 934) 

s_sub_i32 s22, s24, s1 // (op 934) 

s_mul_i32 s24, s23, s22 // (op 934) 


s_sub_u32 s25, s19, s24 // (op 934) 


s_add_u32 s19, s25, s1 // (op 934) 


s_cmp_eq_i32 s18, 0 // (op 934) 
s_mov_b32 s1, scc // (op 934) 

s_cmp_eq_i32 s1, 0 // (op 934) 
s_cselect_b32 s18, s19, s0 // (op 934) 



v_lshl_add_u32 v14, s18, 6, v12 // (op 934) 


v_mov_b32 v17, 0 // (op 934) convert
v_mov_b32 v16, v14 // (op 934) convert

v_mul_lo_u32 v15, v13, s27 // (op 934) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 934) most significant: high of low * low
v_mul_lo_u32 v14, v13, s26 // (op 934) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 934) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 934) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 934) most significant half



s_mul_i32 s1, s26, s31 // (op 936) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 936) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 936) least significant
s_mul_hi_u32 s19, s26, s30 // (op 936) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 936) most significant: sum
s_add_u32 s1, s1, s19 // (op 936) most significant: sum


s_mov_b32 s52, s28 // (op 936) reg expression
s_mov_b32 s53, s29 // (op 936) reg expression
s_mov_b32 s54, s0 // (op 936) reg expression
s_mov_b32 s55, 131072 // (op 936) call()

s_mul_hi_i32 s0, s16, s10 // (op 953) 
v_and_b32 v14, -4, v1 // (op 953) 
s_add_i32 s1, s0, s16 // (op 953) 

v_sub_u32 v15, v1, v14 // (op 953) 

s_ashr_i32 s0, s1, 31 // (op 953) 
v_lshlrev_b32 v14, 4, v15 // (op 953) 

s_cmp_eq_i32 s10, 0 // (op 953) 
s_mov_b32 s18, scc // (op 953) 
v_lshrrev_b32 v15, 2, v1 // (op 953) 
s_cmp_eq_i32 s18, 0 // (op 953) 
s_cselect_b32 s19, 0, -1 // (op 953) 

s_and_b32 s18, s13, 31 // (op 953) 
s_lshl_b32 s22, 1, s18 // (op 953) 
s_add_i32 s23, s22, s19 // (op 953) 


s_and_b32 s19, s0, s23 // (op 953) 


s_add_i32 s0, s1, s19 // (op 953) 


s_ashr_i32 s1, s0, s18 // (op 953) 


s_lshl_b32 s0, s13, 24 // (op 953) 
s_ashr_i32 s18, s0, 31 // (op 953) 

s_xor_b32 s0, s1, s18 // (op 953) 

s_sub_i32 s1, s0, s18 // (op 953) 


s_mul_hi_i32 s0, s1, s14 // (op 953) 
s_add_i32 s18, s0, s1 // (op 953) 

s_ashr_i32 s0, s18, 31 // (op 953) 
s_cmp_eq_i32 s14, 0 // (op 953) 
s_mov_b32 s19, scc // (op 953) 
s_cmp_eq_i32 s19, 0 // (op 953) 
s_cselect_b32 s22, 0, -1 // (op 953) 

s_and_b32 s19, s15, 31 // (op 953) 
s_lshl_b32 s23, 1, s19 // (op 953) 
s_add_i32 s24, s23, s22 // (op 953) 


s_and_b32 s25, s0, s24 // (op 953) 


s_add_i32 s0, s18, s25 // (op 953) 


s_ashr_i32 s18, s0, s19 // (op 953) 


s_lshl_b32 s0, s15, 24 // (op 953) 
s_ashr_i32 s19, s0, 31 // (op 953) 

s_xor_b32 s0, s18, s19 // (op 953) 

s_sub_i32 s18, s0, s19 // (op 953) 


s_mul_i32 s0, s1, s11 // (op 953) 

s_sub_u32 s1, s16, s0 // (op 953) 

s_mul_i32 s0, s18, s11 // (op 953) 
s_add_u32 s0, s0, s1 // (op 953) 


s_mul_hi_i32 s1, s16, s9 // (op 953) 
s_add_i32 s18, s1, s16 // (op 953) 

s_ashr_i32 s1, s18, 31 // (op 953) 
s_cmp_eq_i32 s9, 0 // (op 953) 
s_mov_b32 s19, scc // (op 953) 
s_cmp_eq_i32 s19, 0 // (op 953) 
s_cselect_b32 s22, 0, -1 // (op 953) 

s_and_b32 s19, s47, 31 // (op 953) 
s_lshl_b32 s23, 1, s19 // (op 953) 
s_add_i32 s24, s23, s22 // (op 953) 


s_and_b32 s25, s1, s24 // (op 953) 


s_add_i32 s1, s18, s25 // (op 953) 


s_ashr_i32 s18, s1, s19 // (op 953) 


s_lshl_b32 s1, s47, 24 // (op 953) 
s_ashr_i32 s19, s1, 31 // (op 953) 

s_xor_b32 s1, s18, s19 // (op 953) 

s_sub_i32 s18, s1, s19 // (op 953) 


s_add_u32 s50, s6, 16 // (op 953) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 953) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 953) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 953) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 953) 

s_mul_i32 s1, s11, s24 // (op 953) 

s_mul_i32 s19, s8, s1 // (op 953) 

s_mul_i32 s1, s18, s19 // (op 953) 

s_sub_u32 s19, s16, s1 // (op 953) 

s_mov_b32 s1, s19 // (op 953) 
s_mul_hi_i32 s22, s1, s12 // (op 953) 
s_add_i32 s23, s22, s1 // (op 953) 


s_ashr_i32 s1, s23, 31 // (op 953) 
s_cmp_eq_i32 s12, 0 // (op 953) 
s_mov_b32 s22, scc // (op 953) 
s_cmp_eq_i32 s22, 0 // (op 953) 
s_cselect_b32 s24, 0, -1 // (op 953) 

s_and_b32 s22, s46, 31 // (op 953) 
s_lshl_b32 s25, 1, s22 // (op 953) 
s_add_i32 s49, s25, s24 // (op 953) 


s_and_b32 s24, s1, s49 // (op 953) 


s_add_i32 s1, s23, s24 // (op 953) 


s_ashr_i32 s23, s1, s22 // (op 953) 


s_lshl_b32 s1, s46, 24 // (op 953) 
s_ashr_i32 s22, s1, 31 // (op 953) 

s_xor_b32 s1, s23, s22 // (op 953) 

s_sub_i32 s23, s1, s22 // (op 953) 


s_mul_i32 s1, s48, s11 // (op 953) 
s_add_u32 s24, s4, 64 // (op 953) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 953) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 953) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 953) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 953) 

s_sub_i32 s22, s24, s1 // (op 953) 

s_mul_i32 s24, s23, s22 // (op 953) 


s_sub_u32 s25, s19, s24 // (op 953) 


s_add_u32 s19, s25, s1 // (op 953) 


s_cmp_eq_i32 s18, 0 // (op 953) 
s_mov_b32 s1, scc // (op 953) 

s_cmp_eq_i32 s1, 0 // (op 953) 
s_cselect_b32 s18, s19, s0 // (op 953) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 953) 


v_mov_b32 v19, 0 // (op 953) convert
v_mov_b32 v18, v16 // (op 953) convert

v_mul_lo_u32 v17, v15, s27 // (op 953) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 953) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 953) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 953) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 953) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 953) most significant half



s_mul_i32 s1, s26, s31 // (op 955) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 955) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 955) least significant
s_mul_hi_u32 s19, s26, s30 // (op 955) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 955) most significant: sum
s_add_u32 s1, s1, s19 // (op 955) most significant: sum


s_mov_b32 s52, s28 // (op 955) reg expression
s_mov_b32 s53, s29 // (op 955) reg expression
s_mov_b32 s54, s0 // (op 955) reg expression
s_mov_b32 s55, 131072 // (op 955) call()

s_mul_hi_i32 s0, s16, s10 // (op 998) 
s_add_i32 s1, s0, s16 // (op 998) 

s_ashr_i32 s0, s1, 31 // (op 998) 
s_cmp_eq_i32 s10, 0 // (op 998) 
s_mov_b32 s18, scc // (op 998) 
s_cmp_eq_i32 s18, 0 // (op 998) 
s_cselect_b32 s19, 0, -1 // (op 998) 

s_and_b32 s18, s13, 31 // (op 998) 
s_lshl_b32 s22, 1, s18 // (op 998) 
s_add_i32 s23, s22, s19 // (op 998) 


s_and_b32 s19, s0, s23 // (op 998) 


s_add_i32 s0, s1, s19 // (op 998) 


s_ashr_i32 s1, s0, s18 // (op 998) 


s_lshl_b32 s0, s13, 24 // (op 998) 
s_ashr_i32 s18, s0, 31 // (op 998) 

s_xor_b32 s0, s1, s18 // (op 998) 

s_sub_i32 s1, s0, s18 // (op 998) 


s_mul_hi_i32 s0, s1, s14 // (op 998) 
s_add_i32 s18, s0, s1 // (op 998) 

s_ashr_i32 s0, s18, 31 // (op 998) 
s_cmp_eq_i32 s14, 0 // (op 998) 
s_mov_b32 s19, scc // (op 998) 
s_cmp_eq_i32 s19, 0 // (op 998) 
s_cselect_b32 s22, 0, -1 // (op 998) 

s_and_b32 s19, s15, 31 // (op 998) 
s_lshl_b32 s23, 1, s19 // (op 998) 
s_add_i32 s24, s23, s22 // (op 998) 


s_and_b32 s25, s0, s24 // (op 998) 


s_add_i32 s0, s18, s25 // (op 998) 


s_ashr_i32 s18, s0, s19 // (op 998) 


s_lshl_b32 s0, s15, 24 // (op 998) 
s_ashr_i32 s19, s0, 31 // (op 998) 

s_xor_b32 s0, s18, s19 // (op 998) 

s_sub_i32 s18, s0, s19 // (op 998) 


s_add_u32 s50, s6, 16 // (op 998) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 998) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 998) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 998) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 998) 

s_mul_i32 s0, s18, s24 // (op 998) 

s_sub_i32 s18, s1, s0 // (op 998) 


s_mul_hi_i32 s19, s16, s9 // (op 998) 
s_add_i32 s0, s19, s16 // (op 998) 

s_ashr_i32 s1, s0, 31 // (op 998) 
s_cmp_eq_i32 s9, 0 // (op 998) 
s_mov_b32 s19, scc // (op 998) 
s_cmp_eq_i32 s19, 0 // (op 998) 
s_cselect_b32 s22, 0, -1 // (op 998) 

s_and_b32 s19, s47, 31 // (op 998) 
s_lshl_b32 s23, 1, s19 // (op 998) 
s_add_i32 s49, s23, s22 // (op 998) 


s_and_b32 s22, s1, s49 // (op 998) 


s_add_i32 s1, s0, s22 // (op 998) 


s_ashr_i32 s0, s1, s19 // (op 998) 


s_lshl_b32 s1, s47, 24 // (op 998) 
s_ashr_i32 s19, s1, 31 // (op 998) 

s_xor_b32 s1, s0, s19 // (op 998) 

s_sub_i32 s0, s1, s19 // (op 998) 


s_mul_i32 s1, s11, s24 // (op 998) 

s_mul_i32 s19, s8, s1 // (op 998) 

s_mul_i32 s1, s0, s19 // (op 998) 

s_sub_u32 s19, s16, s1 // (op 998) 

s_mul_hi_i32 s1, s19, s12 // (op 998) 
s_add_i32 s22, s1, s19 // (op 998) 


s_ashr_i32 s1, s22, 31 // (op 998) 
s_cmp_eq_i32 s12, 0 // (op 998) 
s_mov_b32 s19, scc // (op 998) 
s_cmp_eq_i32 s19, 0 // (op 998) 
s_cselect_b32 s23, 0, -1 // (op 998) 

s_and_b32 s19, s46, 31 // (op 998) 
s_lshl_b32 s24, 1, s19 // (op 998) 
s_add_i32 s25, s24, s23 // (op 998) 


s_and_b32 s23, s1, s25 // (op 998) 


s_add_i32 s1, s22, s23 // (op 998) 


s_ashr_i32 s22, s1, s19 // (op 998) 


s_lshl_b32 s1, s46, 24 // (op 998) 
s_ashr_i32 s19, s1, 31 // (op 998) 

s_xor_b32 s1, s22, s19 // (op 998) 

s_sub_i32 s22, s1, s19 // (op 998) 


s_cmp_eq_i32 s0, 0 // (op 998) 
s_mov_b32 s1, scc // (op 998) 

s_cmp_eq_i32 s1, 0 // (op 998) 
s_cselect_b32 s0, s22, s18 // (op 998) 



s_lshl_b32 s1, s0, 4 // (op 998) 

s_mov_b32 s19, 0 // (op 998) convert
s_mov_b32 s18, s1 // (op 998) convert

v_mul_lo_u32 v17, v1, s33 // (op 998) most significant: low of low * high
v_mul_hi_u32 v18, v1, s32 // (op 998) most significant: high of low * low
v_mul_lo_u32 v16, v1, s32 // (op 998) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 998) most significant: sum

v_mov_b32 v18, s18 // (op 998) 
v_mov_b32 v19, s19 // (op 998) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 998) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 998) most significant half




v_mul_lo_u32 v19, v17, 6 // (op 998) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 998) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 998) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 998) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 998) 

v_mov_b32 v18, v16 // (op 998) convert

s_mul_i32 s1, s31, 6 // (op 1000) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 1000) least significant
s_mul_hi_u32 s18, s30, 6 // (op 1000) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1000) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1000) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1000) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1000) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1000) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1000) most significant: sum
s_add_u32 s19, s19, s23 // (op 1000) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1000) 

s_mov_b32 s64, s34 // (op 1000) reg expression
s_mov_b32 s65, s35 // (op 1000) reg expression
s_mov_b32 s66, s0 // (op 1000) reg expression
s_mov_b32 s67, 131072 // (op 1000) call()

s_mul_hi_i32 s0, s16, s10 // (op 1017) 
s_add_i32 s1, s0, s16 // (op 1017) 

s_ashr_i32 s0, s1, 31 // (op 1017) 
s_cmp_eq_i32 s10, 0 // (op 1017) 
s_mov_b32 s18, scc // (op 1017) 
s_cmp_eq_i32 s18, 0 // (op 1017) 
s_cselect_b32 s19, 0, -1 // (op 1017) 

s_and_b32 s18, s13, 31 // (op 1017) 
s_lshl_b32 s22, 1, s18 // (op 1017) 
s_add_i32 s23, s22, s19 // (op 1017) 


s_and_b32 s19, s0, s23 // (op 1017) 


s_add_i32 s0, s1, s19 // (op 1017) 


s_ashr_i32 s1, s0, s18 // (op 1017) 


s_lshl_b32 s0, s13, 24 // (op 1017) 
s_ashr_i32 s18, s0, 31 // (op 1017) 

s_xor_b32 s0, s1, s18 // (op 1017) 

s_sub_i32 s1, s0, s18 // (op 1017) 


s_mul_hi_i32 s0, s1, s14 // (op 1017) 
s_add_i32 s18, s0, s1 // (op 1017) 

s_ashr_i32 s0, s18, 31 // (op 1017) 
s_cmp_eq_i32 s14, 0 // (op 1017) 
s_mov_b32 s19, scc // (op 1017) 
s_cmp_eq_i32 s19, 0 // (op 1017) 
s_cselect_b32 s22, 0, -1 // (op 1017) 

s_and_b32 s19, s15, 31 // (op 1017) 
s_lshl_b32 s23, 1, s19 // (op 1017) 
s_add_i32 s24, s23, s22 // (op 1017) 


s_and_b32 s25, s0, s24 // (op 1017) 


s_add_i32 s0, s18, s25 // (op 1017) 


s_ashr_i32 s18, s0, s19 // (op 1017) 


s_lshl_b32 s0, s15, 24 // (op 1017) 
s_ashr_i32 s19, s0, 31 // (op 1017) 

s_xor_b32 s0, s18, s19 // (op 1017) 

s_sub_i32 s18, s0, s19 // (op 1017) 


s_add_u32 s50, s6, 16 // (op 1017) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1017) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1017) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1017) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1017) 

s_mul_i32 s0, s18, s24 // (op 1017) 

s_sub_i32 s18, s1, s0 // (op 1017) 


s_mul_hi_i32 s19, s16, s9 // (op 1017) 
s_add_i32 s0, s19, s16 // (op 1017) 

s_ashr_i32 s1, s0, 31 // (op 1017) 
s_cmp_eq_i32 s9, 0 // (op 1017) 
s_mov_b32 s19, scc // (op 1017) 
s_cmp_eq_i32 s19, 0 // (op 1017) 
s_cselect_b32 s22, 0, -1 // (op 1017) 

s_and_b32 s19, s47, 31 // (op 1017) 
s_lshl_b32 s23, 1, s19 // (op 1017) 
s_add_i32 s49, s23, s22 // (op 1017) 


s_and_b32 s22, s1, s49 // (op 1017) 


s_add_i32 s1, s0, s22 // (op 1017) 


s_ashr_i32 s0, s1, s19 // (op 1017) 


s_lshl_b32 s1, s47, 24 // (op 1017) 
s_ashr_i32 s19, s1, 31 // (op 1017) 

s_xor_b32 s1, s0, s19 // (op 1017) 

s_sub_i32 s0, s1, s19 // (op 1017) 


s_mul_i32 s1, s11, s24 // (op 1017) 

s_mul_i32 s19, s8, s1 // (op 1017) 

s_mul_i32 s1, s0, s19 // (op 1017) 

s_sub_u32 s19, s16, s1 // (op 1017) 

s_mul_hi_i32 s1, s19, s12 // (op 1017) 
s_add_i32 s22, s1, s19 // (op 1017) 


s_ashr_i32 s1, s22, 31 // (op 1017) 
s_cmp_eq_i32 s12, 0 // (op 1017) 
s_mov_b32 s19, scc // (op 1017) 
s_cmp_eq_i32 s19, 0 // (op 1017) 
s_cselect_b32 s23, 0, -1 // (op 1017) 

s_and_b32 s19, s46, 31 // (op 1017) 
s_lshl_b32 s24, 1, s19 // (op 1017) 
s_add_i32 s25, s24, s23 // (op 1017) 


s_and_b32 s23, s1, s25 // (op 1017) 


s_add_i32 s1, s22, s23 // (op 1017) 


s_ashr_i32 s22, s1, s19 // (op 1017) 


s_lshl_b32 s1, s46, 24 // (op 1017) 
s_ashr_i32 s19, s1, 31 // (op 1017) 

s_xor_b32 s1, s22, s19 // (op 1017) 

s_sub_i32 s22, s1, s19 // (op 1017) 


s_cmp_eq_i32 s0, 0 // (op 1017) 
s_mov_b32 s1, scc // (op 1017) 

s_cmp_eq_i32 s1, 0 // (op 1017) 
s_cselect_b32 s0, s22, s18 // (op 1017) 



s_lshl_b32 s1, s0, 4 // (op 1017) 

s_mov_b32 s19, 0 // (op 1017) convert
s_mov_b32 s18, s1 // (op 1017) convert

v_mul_lo_u32 v17, v1, s33 // (op 1017) most significant: low of low * high
v_mul_hi_u32 v19, v1, s32 // (op 1017) most significant: high of low * low
v_mul_lo_u32 v16, v1, s32 // (op 1017) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 1017) most significant: sum

v_mov_b32 v19, s18 // (op 1017) 
v_mov_b32 v20, s19 // (op 1017) 
v_add_co_u32 v16, s[0:1], v16, v19 // (op 1017) least significant half
v_addc_co_u32 v17, s[0:1], v17, v20, s[0:1] // (op 1017) most significant half




v_mul_lo_u32 v21, v17, 6 // (op 1017) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 1017) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 1017) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 1017) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 1017) 

v_mov_b32 v19, v16 // (op 1017) convert

s_mul_i32 s1, s31, 6 // (op 1019) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 1019) least significant
s_mul_hi_u32 s18, s30, 6 // (op 1019) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1019) most significant: sum

s_mul_i32 s19, s32, s1 // (op 1019) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 1019) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 1019) least significant
s_mul_hi_u32 s23, s32, s0 // (op 1019) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 1019) most significant: sum
s_add_u32 s19, s19, s23 // (op 1019) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1019) 

s_mov_b32 s64, s34 // (op 1019) reg expression
s_mov_b32 s65, s35 // (op 1019) reg expression
s_mov_b32 s66, s0 // (op 1019) reg expression
s_mov_b32 s67, 131072 // (op 1019) call()

s_mul_hi_i32 s0, s16, s10 // (op 1062) 
v_and_b32 v16, -64, v1 // (op 1062) 
s_add_i32 s1, s0, s16 // (op 1062) 

v_sub_u32 v17, v1, v16 // (op 1062) 

s_ashr_i32 s0, s1, 31 // (op 1062) 
v_lshrrev_b32 v16, 6, v1 // (op 1062) 
s_cmp_eq_i32 s10, 0 // (op 1062) 
s_mov_b32 s18, scc // (op 1062) 
v_lshlrev_b32 v20, 1, v16 // (op 1062) 

s_cmp_eq_i32 s18, 0 // (op 1062) 
s_cselect_b32 s19, 0, -1 // (op 1062) 

s_and_b32 s18, s13, 31 // (op 1062) 
s_lshl_b32 s22, 1, s18 // (op 1062) 
s_add_i32 s23, s22, s19 // (op 1062) 


s_and_b32 s19, s0, s23 // (op 1062) 


s_add_i32 s0, s1, s19 // (op 1062) 


s_ashr_i32 s1, s0, s18 // (op 1062) 


s_lshl_b32 s0, s13, 24 // (op 1062) 
s_ashr_i32 s18, s0, 31 // (op 1062) 

s_xor_b32 s0, s1, s18 // (op 1062) 

s_sub_i32 s1, s0, s18 // (op 1062) 


s_mul_hi_i32 s0, s1, s14 // (op 1062) 
s_add_i32 s18, s0, s1 // (op 1062) 

s_ashr_i32 s0, s18, 31 // (op 1062) 
s_cmp_eq_i32 s14, 0 // (op 1062) 
s_mov_b32 s19, scc // (op 1062) 
s_cmp_eq_i32 s19, 0 // (op 1062) 
s_cselect_b32 s22, 0, -1 // (op 1062) 

s_and_b32 s19, s15, 31 // (op 1062) 
s_lshl_b32 s23, 1, s19 // (op 1062) 
s_add_i32 s24, s23, s22 // (op 1062) 


s_and_b32 s25, s0, s24 // (op 1062) 


s_add_i32 s0, s18, s25 // (op 1062) 


s_ashr_i32 s18, s0, s19 // (op 1062) 


s_lshl_b32 s0, s15, 24 // (op 1062) 
s_ashr_i32 s19, s0, 31 // (op 1062) 

s_xor_b32 s0, s18, s19 // (op 1062) 

s_sub_i32 s18, s0, s19 // (op 1062) 


s_mul_i32 s0, s1, s11 // (op 1062) 

s_sub_u32 s1, s16, s0 // (op 1062) 

s_mul_i32 s0, s18, s11 // (op 1062) 
s_add_u32 s0, s0, s1 // (op 1062) 


s_mul_hi_i32 s1, s16, s9 // (op 1062) 
s_add_i32 s18, s1, s16 // (op 1062) 

s_ashr_i32 s1, s18, 31 // (op 1062) 
s_cmp_eq_i32 s9, 0 // (op 1062) 
s_mov_b32 s19, scc // (op 1062) 
s_cmp_eq_i32 s19, 0 // (op 1062) 
s_cselect_b32 s22, 0, -1 // (op 1062) 

s_and_b32 s19, s47, 31 // (op 1062) 
s_lshl_b32 s23, 1, s19 // (op 1062) 
s_add_i32 s24, s23, s22 // (op 1062) 


s_and_b32 s25, s1, s24 // (op 1062) 


s_add_i32 s1, s18, s25 // (op 1062) 


s_ashr_i32 s18, s1, s19 // (op 1062) 


s_lshl_b32 s1, s47, 24 // (op 1062) 
s_ashr_i32 s19, s1, 31 // (op 1062) 

s_xor_b32 s1, s18, s19 // (op 1062) 

s_sub_i32 s18, s1, s19 // (op 1062) 


s_add_u32 s50, s6, 16 // (op 1062) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1062) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1062) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1062) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1062) 

s_mul_i32 s1, s11, s24 // (op 1062) 

s_mul_i32 s19, s8, s1 // (op 1062) 

s_mul_i32 s1, s18, s19 // (op 1062) 

s_sub_u32 s19, s16, s1 // (op 1062) 

s_mov_b32 s1, s19 // (op 1062) 
s_mul_hi_i32 s22, s1, s12 // (op 1062) 
s_add_i32 s23, s22, s1 // (op 1062) 


s_ashr_i32 s1, s23, 31 // (op 1062) 
s_cmp_eq_i32 s12, 0 // (op 1062) 
s_mov_b32 s22, scc // (op 1062) 
s_cmp_eq_i32 s22, 0 // (op 1062) 
s_cselect_b32 s24, 0, -1 // (op 1062) 

s_and_b32 s22, s46, 31 // (op 1062) 
s_lshl_b32 s25, 1, s22 // (op 1062) 
s_add_i32 s49, s25, s24 // (op 1062) 


s_and_b32 s24, s1, s49 // (op 1062) 


s_add_i32 s1, s23, s24 // (op 1062) 


s_ashr_i32 s23, s1, s22 // (op 1062) 


s_lshl_b32 s1, s46, 24 // (op 1062) 
s_ashr_i32 s22, s1, 31 // (op 1062) 

s_xor_b32 s1, s23, s22 // (op 1062) 

s_sub_i32 s23, s1, s22 // (op 1062) 


s_mul_i32 s1, s48, s11 // (op 1062) 
s_add_u32 s24, s4, 64 // (op 1062) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1062) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1062) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1062) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1062) 

s_sub_i32 s22, s24, s1 // (op 1062) 

s_mul_i32 s24, s23, s22 // (op 1062) 


s_sub_u32 s25, s19, s24 // (op 1062) 


s_add_u32 s19, s25, s1 // (op 1062) 


s_cmp_eq_i32 s18, 0 // (op 1062) 
s_mov_b32 s1, scc // (op 1062) 

s_cmp_eq_i32 s1, 0 // (op 1062) 
s_cselect_b32 s18, s19, s0 // (op 1062) 



v_lshl_add_u32 v16, s18, 6, v17 // (op 1062) 


v_mul_lo_u32 v17, v20, s36 // (op 1062) 

v_add_u32 v20, v16, v17 // (op 1062) 


s_mul_i32 s1, s36, s41 // (op 1064) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1064) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1064) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1064) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1064) most significant: sum
s_add_u32 s1, s1, s19 // (op 1064) most significant: sum


s_mov_b32 s68, s38 // (op 1064) reg expression
s_mov_b32 s69, s39 // (op 1064) reg expression
s_mov_b32 s70, s0 // (op 1064) reg expression
s_mov_b32 s71, 131072 // (op 1064) call()

s_mul_hi_i32 s0, s16, s10 // (op 1081) 
v_and_b32 v16, -64, v1 // (op 1081) 
s_add_i32 s1, s0, s16 // (op 1081) 

v_sub_u32 v17, v1, v16 // (op 1081) 

s_ashr_i32 s0, s1, 31 // (op 1081) 
v_lshrrev_b32 v16, 6, v1 // (op 1081) 
s_cmp_eq_i32 s10, 0 // (op 1081) 
s_mov_b32 s18, scc // (op 1081) 
v_lshlrev_b32 v21, 1, v16 // (op 1081) 

s_cmp_eq_i32 s18, 0 // (op 1081) 
s_cselect_b32 s19, 0, -1 // (op 1081) 

s_and_b32 s18, s13, 31 // (op 1081) 
s_lshl_b32 s22, 1, s18 // (op 1081) 
s_add_i32 s23, s22, s19 // (op 1081) 


s_and_b32 s19, s0, s23 // (op 1081) 


s_add_i32 s0, s1, s19 // (op 1081) 


s_ashr_i32 s1, s0, s18 // (op 1081) 


s_lshl_b32 s0, s13, 24 // (op 1081) 
s_ashr_i32 s18, s0, 31 // (op 1081) 

s_xor_b32 s0, s1, s18 // (op 1081) 

s_sub_i32 s1, s0, s18 // (op 1081) 


s_mul_hi_i32 s0, s1, s14 // (op 1081) 
s_add_i32 s18, s0, s1 // (op 1081) 

s_ashr_i32 s0, s18, 31 // (op 1081) 
s_cmp_eq_i32 s14, 0 // (op 1081) 
s_mov_b32 s19, scc // (op 1081) 
s_cmp_eq_i32 s19, 0 // (op 1081) 
s_cselect_b32 s22, 0, -1 // (op 1081) 

s_and_b32 s19, s15, 31 // (op 1081) 
s_lshl_b32 s23, 1, s19 // (op 1081) 
s_add_i32 s24, s23, s22 // (op 1081) 


s_and_b32 s25, s0, s24 // (op 1081) 


s_add_i32 s0, s18, s25 // (op 1081) 


s_ashr_i32 s18, s0, s19 // (op 1081) 


s_lshl_b32 s0, s15, 24 // (op 1081) 
s_ashr_i32 s19, s0, 31 // (op 1081) 

s_xor_b32 s0, s18, s19 // (op 1081) 

s_sub_i32 s18, s0, s19 // (op 1081) 


s_mul_i32 s0, s1, s11 // (op 1081) 

s_sub_u32 s1, s16, s0 // (op 1081) 

s_mul_i32 s0, s18, s11 // (op 1081) 
s_add_u32 s0, s0, s1 // (op 1081) 


s_mul_hi_i32 s1, s16, s9 // (op 1081) 
s_add_i32 s18, s1, s16 // (op 1081) 

s_ashr_i32 s1, s18, 31 // (op 1081) 
s_cmp_eq_i32 s9, 0 // (op 1081) 
s_mov_b32 s19, scc // (op 1081) 
s_cmp_eq_i32 s19, 0 // (op 1081) 
s_cselect_b32 s22, 0, -1 // (op 1081) 

s_and_b32 s19, s47, 31 // (op 1081) 
s_lshl_b32 s23, 1, s19 // (op 1081) 
s_add_i32 s24, s23, s22 // (op 1081) 


s_and_b32 s25, s1, s24 // (op 1081) 


s_add_i32 s1, s18, s25 // (op 1081) 


s_ashr_i32 s18, s1, s19 // (op 1081) 


s_lshl_b32 s1, s47, 24 // (op 1081) 
s_ashr_i32 s19, s1, 31 // (op 1081) 

s_xor_b32 s1, s18, s19 // (op 1081) 

s_sub_i32 s18, s1, s19 // (op 1081) 


s_add_u32 s50, s6, 16 // (op 1081) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1081) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1081) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1081) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1081) 

s_mul_i32 s1, s11, s24 // (op 1081) 

s_mul_i32 s19, s8, s1 // (op 1081) 

s_mul_i32 s1, s18, s19 // (op 1081) 

s_sub_u32 s19, s16, s1 // (op 1081) 

s_mov_b32 s1, s19 // (op 1081) 
s_mul_hi_i32 s22, s1, s12 // (op 1081) 
s_add_i32 s23, s22, s1 // (op 1081) 


s_ashr_i32 s1, s23, 31 // (op 1081) 
s_cmp_eq_i32 s12, 0 // (op 1081) 
s_mov_b32 s22, scc // (op 1081) 
s_cmp_eq_i32 s22, 0 // (op 1081) 
s_cselect_b32 s24, 0, -1 // (op 1081) 

s_and_b32 s22, s46, 31 // (op 1081) 
s_lshl_b32 s25, 1, s22 // (op 1081) 
s_add_i32 s49, s25, s24 // (op 1081) 


s_and_b32 s24, s1, s49 // (op 1081) 


s_add_i32 s1, s23, s24 // (op 1081) 


s_ashr_i32 s23, s1, s22 // (op 1081) 


s_lshl_b32 s1, s46, 24 // (op 1081) 
s_ashr_i32 s22, s1, 31 // (op 1081) 

s_xor_b32 s1, s23, s22 // (op 1081) 

s_sub_i32 s23, s1, s22 // (op 1081) 


s_mul_i32 s1, s48, s11 // (op 1081) 
s_add_u32 s24, s4, 64 // (op 1081) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1081) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1081) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1081) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1081) 

s_sub_i32 s22, s24, s1 // (op 1081) 

s_mul_i32 s24, s23, s22 // (op 1081) 


s_sub_u32 s25, s19, s24 // (op 1081) 


s_add_u32 s19, s25, s1 // (op 1081) 


s_cmp_eq_i32 s18, 0 // (op 1081) 
s_mov_b32 s1, scc // (op 1081) 

s_cmp_eq_i32 s1, 0 // (op 1081) 
s_cselect_b32 s18, s19, s0 // (op 1081) 



v_lshl_add_u32 v16, s18, 6, v17 // (op 1081) 


v_mul_lo_u32 v17, v21, s36 // (op 1081) 

v_add_u32 v21, v16, v17 // (op 1081) 


s_mul_i32 s1, s36, s41 // (op 1083) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1083) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1083) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1083) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1083) most significant: sum
s_add_u32 s1, s1, s19 // (op 1083) most significant: sum


s_mov_b32 s68, s38 // (op 1083) reg expression
s_mov_b32 s69, s39 // (op 1083) reg expression
s_mov_b32 s70, s0 // (op 1083) reg expression
s_mov_b32 s71, 131072 // (op 1083) call()

s_mul_hi_i32 s0, s16, s10 // (op 1126) 
v_and_b32 v16, 15, v1 // (op 1126) 
s_add_i32 s1, s0, s16 // (op 1126) 

v_and_b32 v17, 63, v1 // (op 1126) 
s_ashr_i32 s0, s1, 31 // (op 1126) 
v_lshrrev_b32 v22, 4, v17 // (op 1126) 

s_cmp_eq_i32 s10, 0 // (op 1126) 
s_mov_b32 s18, scc // (op 1126) 
s_cmp_eq_i32 s18, 0 // (op 1126) 
s_cselect_b32 s19, 0, -1 // (op 1126) 

s_and_b32 s18, s13, 31 // (op 1126) 
s_lshl_b32 s22, 1, s18 // (op 1126) 
s_add_i32 s23, s22, s19 // (op 1126) 


s_and_b32 s19, s0, s23 // (op 1126) 


s_add_i32 s0, s1, s19 // (op 1126) 


s_ashr_i32 s1, s0, s18 // (op 1126) 


s_lshl_b32 s0, s13, 24 // (op 1126) 
s_ashr_i32 s18, s0, 31 // (op 1126) 

s_xor_b32 s0, s1, s18 // (op 1126) 

s_sub_i32 s1, s0, s18 // (op 1126) 


s_mul_hi_i32 s0, s1, s14 // (op 1126) 
s_add_i32 s18, s0, s1 // (op 1126) 

s_ashr_i32 s0, s18, 31 // (op 1126) 
s_cmp_eq_i32 s14, 0 // (op 1126) 
s_mov_b32 s19, scc // (op 1126) 
s_cmp_eq_i32 s19, 0 // (op 1126) 
s_cselect_b32 s22, 0, -1 // (op 1126) 

s_and_b32 s19, s15, 31 // (op 1126) 
s_lshl_b32 s23, 1, s19 // (op 1126) 
s_add_i32 s24, s23, s22 // (op 1126) 


s_and_b32 s25, s0, s24 // (op 1126) 


s_add_i32 s0, s18, s25 // (op 1126) 


s_ashr_i32 s18, s0, s19 // (op 1126) 


s_lshl_b32 s0, s15, 24 // (op 1126) 
s_ashr_i32 s19, s0, 31 // (op 1126) 

s_xor_b32 s0, s18, s19 // (op 1126) 

s_sub_i32 s18, s0, s19 // (op 1126) 


s_add_u32 s50, s6, 16 // (op 1126) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1126) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1126) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1126) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1126) 

s_mul_i32 s0, s18, s24 // (op 1126) 

s_sub_i32 s18, s1, s0 // (op 1126) 


s_mul_hi_i32 s19, s16, s9 // (op 1126) 
s_add_i32 s0, s19, s16 // (op 1126) 

s_ashr_i32 s1, s0, 31 // (op 1126) 
s_cmp_eq_i32 s9, 0 // (op 1126) 
s_mov_b32 s19, scc // (op 1126) 
s_cmp_eq_i32 s19, 0 // (op 1126) 
s_cselect_b32 s22, 0, -1 // (op 1126) 

s_and_b32 s19, s47, 31 // (op 1126) 
s_lshl_b32 s23, 1, s19 // (op 1126) 
s_add_i32 s49, s23, s22 // (op 1126) 


s_and_b32 s22, s1, s49 // (op 1126) 


s_add_i32 s1, s0, s22 // (op 1126) 


s_ashr_i32 s0, s1, s19 // (op 1126) 


s_lshl_b32 s1, s47, 24 // (op 1126) 
s_ashr_i32 s19, s1, 31 // (op 1126) 

s_xor_b32 s1, s0, s19 // (op 1126) 

s_sub_i32 s0, s1, s19 // (op 1126) 


s_mul_i32 s1, s11, s24 // (op 1126) 

s_mul_i32 s19, s8, s1 // (op 1126) 

s_mul_i32 s1, s0, s19 // (op 1126) 

s_sub_u32 s19, s16, s1 // (op 1126) 

s_mul_hi_i32 s1, s19, s12 // (op 1126) 
s_add_i32 s22, s1, s19 // (op 1126) 


s_ashr_i32 s1, s22, 31 // (op 1126) 
s_cmp_eq_i32 s12, 0 // (op 1126) 
s_mov_b32 s19, scc // (op 1126) 
s_cmp_eq_i32 s19, 0 // (op 1126) 
s_cselect_b32 s23, 0, -1 // (op 1126) 

s_and_b32 s19, s46, 31 // (op 1126) 
s_lshl_b32 s24, 1, s19 // (op 1126) 
s_add_i32 s25, s24, s23 // (op 1126) 


s_and_b32 s23, s1, s25 // (op 1126) 


s_add_i32 s1, s22, s23 // (op 1126) 


s_ashr_i32 s22, s1, s19 // (op 1126) 


s_lshl_b32 s1, s46, 24 // (op 1126) 
s_ashr_i32 s19, s1, 31 // (op 1126) 

s_xor_b32 s1, s22, s19 // (op 1126) 

s_sub_i32 s22, s1, s19 // (op 1126) 


s_cmp_eq_i32 s0, 0 // (op 1126) 
s_mov_b32 s1, scc // (op 1126) 

s_cmp_eq_i32 s1, 0 // (op 1126) 
s_cselect_b32 s0, s22, s18 // (op 1126) 



v_lshl_add_u32 v17, s0, 4, v16 // (op 1126) 


v_mul_lo_u32 v16, v22, s42 // (op 1126) 
v_add_u32 v16, v16, v17 // (op 1126) 


s_mul_i32 s1, s42, s41 // (op 1128) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1128) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1128) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1128) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1128) most significant: sum
s_add_u32 s1, s1, s19 // (op 1128) most significant: sum


s_mov_b32 s72, s44 // (op 1128) reg expression
s_mov_b32 s73, s45 // (op 1128) reg expression
s_mov_b32 s74, s0 // (op 1128) reg expression
s_mov_b32 s75, 131072 // (op 1128) call()

s_mul_hi_i32 s0, s16, s10 // (op 1145) 
v_and_b32 v17, 15, v1 // (op 1145) 
s_add_i32 s1, s0, s16 // (op 1145) 

v_and_b32 v22, 63, v1 // (op 1145) 
s_ashr_i32 s0, s1, 31 // (op 1145) 
v_lshrrev_b32 v23, 4, v22 // (op 1145) 

s_cmp_eq_i32 s10, 0 // (op 1145) 
s_mov_b32 s18, scc // (op 1145) 
s_cmp_eq_i32 s18, 0 // (op 1145) 
s_cselect_b32 s19, 0, -1 // (op 1145) 

s_and_b32 s18, s13, 31 // (op 1145) 
s_lshl_b32 s22, 1, s18 // (op 1145) 
s_add_i32 s23, s22, s19 // (op 1145) 


s_and_b32 s19, s0, s23 // (op 1145) 


s_add_i32 s0, s1, s19 // (op 1145) 


s_ashr_i32 s1, s0, s18 // (op 1145) 


s_lshl_b32 s0, s13, 24 // (op 1145) 
s_ashr_i32 s18, s0, 31 // (op 1145) 

s_xor_b32 s0, s1, s18 // (op 1145) 

s_sub_i32 s1, s0, s18 // (op 1145) 


s_mul_hi_i32 s0, s1, s14 // (op 1145) 
s_add_i32 s18, s0, s1 // (op 1145) 

s_ashr_i32 s0, s18, 31 // (op 1145) 
s_cmp_eq_i32 s14, 0 // (op 1145) 
s_mov_b32 s19, scc // (op 1145) 
s_cmp_eq_i32 s19, 0 // (op 1145) 
s_cselect_b32 s22, 0, -1 // (op 1145) 

s_and_b32 s19, s15, 31 // (op 1145) 
s_lshl_b32 s23, 1, s19 // (op 1145) 
s_add_i32 s24, s23, s22 // (op 1145) 


s_and_b32 s25, s0, s24 // (op 1145) 


s_add_i32 s0, s18, s25 // (op 1145) 


s_ashr_i32 s18, s0, s19 // (op 1145) 


s_lshl_b32 s0, s15, 24 // (op 1145) 
s_ashr_i32 s19, s0, 31 // (op 1145) 

s_xor_b32 s0, s18, s19 // (op 1145) 

s_sub_i32 s18, s0, s19 // (op 1145) 


s_add_u32 s50, s6, 16 // (op 1145) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1145) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1145) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1145) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1145) 

s_mul_i32 s0, s18, s24 // (op 1145) 

s_sub_i32 s18, s1, s0 // (op 1145) 


s_mul_hi_i32 s19, s16, s9 // (op 1145) 
s_add_i32 s0, s19, s16 // (op 1145) 

s_ashr_i32 s1, s0, 31 // (op 1145) 
s_cmp_eq_i32 s9, 0 // (op 1145) 
s_mov_b32 s19, scc // (op 1145) 
s_cmp_eq_i32 s19, 0 // (op 1145) 
s_cselect_b32 s22, 0, -1 // (op 1145) 

s_and_b32 s19, s47, 31 // (op 1145) 
s_lshl_b32 s23, 1, s19 // (op 1145) 
s_add_i32 s49, s23, s22 // (op 1145) 


s_and_b32 s22, s1, s49 // (op 1145) 


s_add_i32 s1, s0, s22 // (op 1145) 


s_ashr_i32 s0, s1, s19 // (op 1145) 


s_lshl_b32 s1, s47, 24 // (op 1145) 
s_ashr_i32 s19, s1, 31 // (op 1145) 

s_xor_b32 s1, s0, s19 // (op 1145) 

s_sub_i32 s0, s1, s19 // (op 1145) 


s_mul_i32 s1, s11, s24 // (op 1145) 

s_mul_i32 s19, s8, s1 // (op 1145) 

s_mul_i32 s1, s0, s19 // (op 1145) 

s_sub_u32 s19, s16, s1 // (op 1145) 

s_mul_hi_i32 s1, s19, s12 // (op 1145) 
s_add_i32 s22, s1, s19 // (op 1145) 


s_ashr_i32 s1, s22, 31 // (op 1145) 
s_cmp_eq_i32 s12, 0 // (op 1145) 
s_mov_b32 s19, scc // (op 1145) 
s_cmp_eq_i32 s19, 0 // (op 1145) 
s_cselect_b32 s23, 0, -1 // (op 1145) 

s_and_b32 s19, s46, 31 // (op 1145) 
s_lshl_b32 s24, 1, s19 // (op 1145) 
s_add_i32 s25, s24, s23 // (op 1145) 


s_and_b32 s23, s1, s25 // (op 1145) 


s_add_i32 s1, s22, s23 // (op 1145) 


s_ashr_i32 s22, s1, s19 // (op 1145) 


s_lshl_b32 s1, s46, 24 // (op 1145) 
s_ashr_i32 s19, s1, 31 // (op 1145) 

s_xor_b32 s1, s22, s19 // (op 1145) 

s_sub_i32 s22, s1, s19 // (op 1145) 


s_cmp_eq_i32 s0, 0 // (op 1145) 
s_mov_b32 s1, scc // (op 1145) 

s_cmp_eq_i32 s1, 0 // (op 1145) 
s_cselect_b32 s0, s22, s18 // (op 1145) 



v_lshl_add_u32 v22, s0, 4, v17 // (op 1145) 


v_mul_lo_u32 v17, v23, s42 // (op 1145) 
v_add_u32 v17, v17, v22 // (op 1145) 


s_mul_i32 s1, s42, s41 // (op 1147) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1147) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1147) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1147) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1147) most significant: sum
s_add_u32 s1, s1, s19 // (op 1147) most significant: sum


s_mov_b32 s72, s44 // (op 1147) reg expression
s_mov_b32 s73, s45 // (op 1147) reg expression
s_mov_b32 s74, s0 // (op 1147) reg expression
s_mov_b32 s75, 131072 // (op 1147) call()


v_mov_b64 v[22:23], s[42:43] // (op 782) convert

v_mov_b32 v24, v16 // (op 782) reg expression
v_mov_b32 v25, v24 // (op 782) 
buffer_load_ubyte v26, v25, s[72:75], 0 offen offset: 0 // (op 782) Load value





v_mov_b32 v22, v18 // (op 786) reg expression

buffer_load_dwordx3 v[28:30], v22, s[64:67], 0 offen // (op 786) Load value

v_and_b32 v27, -4, v1 // (op 1211) 
v_sub_u32 v22, v1, v27 // (op 1211) 

v_lshlrev_b32 v27, 4, v22 // (op 1211) 

v_lshrrev_b32 v22, 2, v1 // (op 1211) 
v_lshl_add_u32 v23, v22, 6, v27 // (op 1211) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 1211) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 5 // (op 889) 
v_mov_b64 v[22:23], s[18:19] // (op 889) convert


v_mov_b64 v[24:25], v[12:13] // (op 889) reg expression
s_mov_b32 m0, s0 // (op 889) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 889) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 889) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 889) most significant half

s_add_u32 m0, m0, 2048 // (op 889) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 889) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 889) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 889) most significant half

s_add_u32 m0, m0, 2048 // (op 889) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 889) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 889) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 889) most significant half

s_add_u32 m0, m0, 2048 // (op 889) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 889) Load value direct to lds




v_mov_b64 v[22:23], s[36:37] // (op 792) convert

v_mov_b32 v27, v20 // (op 792) reg expression
buffer_load_ubyte v24, v27, s[68:71], 0 offen offset: 0 // (op 792) Load value
v_add_u32 v27, v27, v22 // (op 792) 
buffer_load_ubyte v25, v27, s[68:71], 0 offen offset: 0 // (op 792) Load value



v_mov_b64 v[22:23], s[42:43] // (op 796) convert

s_lshl_b32 s0, s42, 2 // (op 796) 
v_add_u32 v27, v16, s0 // (op 796) 

v_mov_b32 v31, v27 // (op 796) 
buffer_load_ubyte v32, v31, s[72:75], 0 offen offset: 0 // (op 796) Load value






s_lshl_b64 s[0:1], s[32:33], 7 // (op 800) 
s_mul_i32 s19, s1, 6 // (op 800) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 800) least significant
s_mul_hi_u32 s22, s0, 6 // (op 800) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 800) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 800) 

v_add_u32 v16, v18, s0 // (op 800) 


buffer_load_dwordx3 v[34:36], v16, s[64:67], 0 offen // (op 800) Load value


v_and_b32 v16, -4, v1 // (op 1384) 
v_sub_u32 v18, v1, v16 // (op 1384) 

v_lshlrev_b32 v16, 4, v18 // (op 1384) 

v_lshrrev_b32 v18, 2, v1 // (op 1384) 
v_lshl_add_u32 v27, v18, 6, v16 // (op 1384) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1384) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 5 // (op 894) 
v_mov_b64 v[22:23], s[18:19] // (op 894) convert


s_lshl_b64 s[18:19], s[26:27], 7 // (op 894) 
v_mov_b32 v16, s18 // (op 894) 
v_mov_b32 v18, s19 // (op 894) 
v_add_co_u32 v38, s[50:51], v12, v16 // (op 894) least significant half
v_addc_co_u32 v39, s[50:51], v13, v18, s[50:51] // (op 894) most significant half




s_add_u32 s1, 8192, s0 // (op 894) 
s_mov_b32 m0, s1 // (op 894) reg expression
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 894) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v22 // (op 894) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 894) most significant half

s_add_u32 m0, m0, 2048 // (op 894) 
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 894) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v22 // (op 894) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 894) most significant half

s_add_u32 m0, m0, 2048 // (op 894) 
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 894) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v22 // (op 894) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 894) most significant half

s_add_u32 m0, m0, 2048 // (op 894) 
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 894) Load value direct to lds






v_mov_b64 v[12:13], s[36:37] // (op 806) convert

s_lshl_b32 s0, s36, 2 // (op 806) 
v_add_u32 v16, v20, s0 // (op 806) 

buffer_load_ubyte v22, v16, s[68:71], 0 offen offset: 0 // (op 806) Load value
v_add_u32 v16, v16, v12 // (op 806) 
buffer_load_ubyte v23, v16, s[68:71], 0 offen offset: 0 // (op 806) Load value



v_lshlrev_b32 v16, 4, v1 // (op 1269) 
v_mul_lo_u32 v18, v16, 6 // (op 1269) 

v_lshrrev_b32 v16, 3, v18 // (op 1269) 

v_and_b32 v18, 268435455, v1 // (op 1269) 
v_lshlrev_b32 v20, 2, v18 // (op 1269) 

v_add_u32 v18, v16, v20 // (op 1269) 





s_waitcnt vmcnt(14)
ds_write_b96 v18, v[28:30] offset:16384 // (op 808) Store local data 

v_and_b32 v16, -64, v1 // (op 1322) 
v_sub_u32 v18, v1, v16 // (op 1322) 

v_lshrrev_b32 v16, 6, v1 // (op 1322) 
v_lshl_add_u32 v20, v16, 7, v18 // (op 1322) 





s_waitcnt vmcnt(9)
ds_write_b8 v20, v24 offset:18432 // (op 810) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v25 offset:18496 // (op 810) Store local data 

v_and_b32 v16, -4, v1 // (op 1183) 
v_sub_u32 v18, v1, v16 // (op 1183) 

v_lshlrev_b32 v16, 4, v18 // (op 1183) 

v_lshrrev_b32 v18, 2, v1 // (op 1183) 
v_lshl_add_u32 v20, v18, 6, v16 // (op 1183) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1183) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v16, 4, v1 // (op 1240) 
v_mul_lo_u32 v18, v16, 6 // (op 1240) 

v_lshrrev_b32 v16, 3, v18 // (op 1240) 

v_and_b32 v18, 268435455, v1 // (op 1240) 
v_lshlrev_b32 v20, 2, v18 // (op 1240) 

v_add_u32 v18, v16, v20 // (op 1240) 


v_and_b32 v16, -64, v1 // (op 1293) 
v_sub_u32 v20, v1, v16 // (op 1293) 

v_lshrrev_b32 v16, 6, v1 // (op 1293) 
v_lshl_add_u32 v27, v16, 7, v20 // (op 1293) 


v_and_b32 v16, -4, v1 // (op 1369) 
v_sub_u32 v20, v1, v16 // (op 1369) 

v_lshlrev_b32 v16, 4, v20 // (op 1369) 

v_lshrrev_b32 v20, 2, v1 // (op 1369) 
v_lshl_add_u32 v31, v20, 6, v16 // (op 1369) 


s_nop 0
v_readfirstlane_b32 s1, v31 // (op 1369) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v16, 4, v1 // (op 1413) 
v_mul_lo_u32 v20, v16, 6 // (op 1413) 

v_lshrrev_b32 v16, 3, v20 // (op 1413) 

v_and_b32 v20, 268435455, v1 // (op 1413) 
v_lshlrev_b32 v31, 2, v20 // (op 1413) 

v_add_u32 v20, v16, v31 // (op 1413) 


v_and_b32 v16, -64, v1 // (op 1437) 
v_sub_u32 v31, v1, v16 // (op 1437) 

v_lshrrev_b32 v16, 6, v1 // (op 1437) 
v_lshl_add_u32 v33, v16, 7, v31 // (op 1437) 


s_mov_b32 s18, 0 // (op 315) call()
s_ashr_i64 s[50:51], s[30:31], 7 // (op 314) 
s_mov_b32 s19, s50 // (op 314) convert

s_and_b32 s22, s19, 4294967294 // (op 314) 

s_cmp_lt_u32 s18, s22 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_314 if false)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_314:
s_barrier  // (op 1495) 
s_add_u32 s18, s18, 2 // (op 316) 



ds_read_b64_tr_b8 v[38:39], v0 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:512 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:4096 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:4608 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1508) 



ds_read_b96_tr_b6 v[46:48], v4 offset:16384 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v4 offset:16640 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1522) 


ds_read_u8 v16, v7 offset:18432 // (op 77) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:51], a[0:3], v16, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 84) 
ds_read_b64_tr_b8 v[52:53], v0 offset:32 // (op 344) Transpose load local data 


ds_read_b64_tr_b8 v[54:55], v0 offset:544 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:4128 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:4640 // (op 344) Transpose load local data 


ds_read_u8 v16, v7 offset:18464 // (op 348) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[46:51], a[4:7], v16, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 350) 




v_mov_b64 v[12:13], s[42:43] // (op 82) convert

v_mov_b32 v16, v17 // (op 82) 
s_lshl_b32 s19, s42, 3 // (op 82) 
v_add_u32 v31, v16, s19 // (op 82) 

v_mov_b32 v37, v31 // (op 82) 
buffer_load_ubyte v26, v37, s[72:75], 0 offen offset: 0 // (op 82) Load value





v_mov_b32 v31, v19 // (op 10) 
s_lshl_b64 s[50:51], s[32:33], 7 // (op 10) 
s_mul_i32 s23, s51, 6 // (op 10) most significant: low of high * low
s_mul_i32 s22, s50, 6 // (op 10) least significant
s_mul_hi_u32 s19, s50, 6 // (op 10) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 10) most significant: sum


s_lshr_b64 s[24:25], s[22:23], 3 // (op 10) 

s_lshl_b32 s19, s24, 1 // (op 10) 

v_add_u32 v12, v31, s19 // (op 10) 


buffer_load_dwordx3 v[28:30], v12, s[64:67], 0 offen // (op 10) Load value

s_barrier  // (op 1498) 

s_lshl_b64 s[50:51], s[26:27], 5 // (op 873) 
v_mov_b64 v[12:13], s[50:51] // (op 873) convert


v_mov_b64 v[38:39], v[14:15] // (op 873) 
s_lshl_b64 s[50:51], s[26:27], 7 // (op 873) 
s_lshl_b64 s[22:23], s[50:51], 1 // (op 873) 

v_mov_b32 v37, s22 // (op 873) 
v_mov_b32 v40, s23 // (op 873) 
v_add_co_u32 v42, s[24:25], v38, v37 // (op 873) least significant half
v_addc_co_u32 v43, s[24:25], v39, v40, s[24:25] // (op 873) most significant half




s_mov_b32 m0, s0 // (op 873) reg expression
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v12 // (op 873) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v12 // (op 873) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v42, s[50:51], v42, v12 // (op 873) least significant half
v_addc_co_u32 v43, s[50:51], 0, v43, s[50:51] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 873) Load value direct to lds



v_mov_b64 v[12:13], s[36:37] // (op 17) convert

v_mov_b32 v37, v21 // (op 17) 
s_lshl_b32 s19, s36, 3 // (op 17) 
v_add_u32 v40, v37, s19 // (op 17) 

buffer_load_ubyte v24, v40, s[68:71], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v40, v40, v12 // (op 17) 
buffer_load_ubyte v25, v40, s[68:71], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1545) 



s_waitcnt vmcnt(14)
ds_write_b96 v20, v[34:36] offset:18688 // (op 239) Store local data 
s_barrier  // (op 1551) 



s_waitcnt vmcnt(9)
ds_write_b8 v33, v22 offset:20736 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v33, v23 offset:20800 // (op 240) Store local data 
s_barrier  // (op 1536) 



ds_read_b64_tr_b8 v[40:41], v10 offset:8192 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v10 offset:8704 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v10 offset:12288 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v10 offset:12800 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1542) 



ds_read_b96_tr_b6 v[48:50], v5 offset:18688 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v5 offset:18944 // (op 233) Transpose load local data 
s_barrier  // (op 1584) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1548) 


ds_read_u8 v12, v11 offset:20736 // (op 235) Load local data 



ds_read_b64_tr_b8 v[54:55], v10 offset:8224 // (op 357) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v10 offset:8736 // (op 357) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:53], a[0:3], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 238) 
ds_read_b64_tr_b8 v[58:59], v10 offset:12320 // (op 357) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v10 offset:12832 // (op 357) Transpose load local data 


ds_read_u8 v12, v11 offset:20768 // (op 361) Load local data 
s_barrier  // (op 1622) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[48:53], a[4:7], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 351) 




v_mov_b64 v[12:13], s[42:43] // (op 237) convert

s_lshl_b32 s19, s42, 2 // (op 237) 
s_mul_i32 s22, 3, s19 // (op 237) 

v_add_u32 v40, v16, s22 // (op 237) 

v_mov_b32 v41, v40 // (op 237) 
buffer_load_ubyte v32, v41, s[72:75], 0 offen offset: 0 // (op 237) Load value






s_lshl_b32 s19, s42, 3 // (op 1137) 
v_add_u32 v17, v17, s19 // (op 1137) 

s_lshl_b64 s[50:51], s[32:33], 7 // (op 241) 
s_mul_i32 s23, s51, 6 // (op 241) most significant: low of high * low
s_mul_i32 s22, s50, 6 // (op 241) least significant
s_mul_hi_u32 s19, s50, 6 // (op 241) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 241) most significant: sum


s_lshr_b64 s[24:25], s[22:23], 3 // (op 241) 

s_mul_i32 s19, 3, s24 // (op 241) 

v_add_u32 v16, v31, s19 // (op 241) 


buffer_load_dwordx3 v[34:36], v16, s[64:67], 0 offen // (op 241) Load value


s_barrier  // (op 1539) 
s_lshl_b64 s[50:51], s[32:33], 7 // (op 1009) 

s_mul_i32 s23, s51, 6 // (op 1009) most significant: low of high * low
s_mul_i32 s22, s50, 6 // (op 1009) least significant
s_mul_hi_u32 s19, s50, 6 // (op 1009) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 1009) most significant: sum


s_lshr_b64 s[24:25], s[22:23], 3 // (op 1009) 

s_lshl_b32 s19, s24, 1 // (op 1009) 

v_add_u32 v19, v19, s19 // (op 1009) 

s_lshl_b64 s[50:51], s[26:27], 5 // (op 884) 
v_mov_b64 v[12:13], s[50:51] // (op 884) convert


s_lshl_b64 s[50:51], s[26:27], 7 // (op 884) 
s_mul_i32 s23, 3, s51 // (op 884) most significant: low of low * high
s_mul_i32 s22, 3, s50 // (op 884) least significant
s_mul_hi_u32 s19, 3, s50 // (op 884) most significant: high of low * low
s_add_u32 s23, s23, s19 // (op 884) most significant: sum


v_mov_b32 v16, s22 // (op 884) 
v_mov_b32 v31, s23 // (op 884) 
v_add_co_u32 v40, s[24:25], v38, v16 // (op 884) least significant half
v_addc_co_u32 v41, s[24:25], v39, v31, s[24:25] // (op 884) most significant half




s_add_u32 s19, 8192, s1 // (op 884) 
s_mov_b32 m0, s19 // (op 884) reg expression
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 884) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v12 // (op 884) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 884) most significant half

s_add_u32 m0, m0, 2048 // (op 884) 
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 884) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v12 // (op 884) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 884) most significant half

s_add_u32 m0, m0, 2048 // (op 884) 
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 884) Load value direct to lds
v_add_co_u32 v40, s[50:51], v40, v12 // (op 884) least significant half
v_addc_co_u32 v41, s[50:51], 0, v41, s[50:51] // (op 884) most significant half

s_add_u32 m0, m0, 2048 // (op 884) 
buffer_load_dwordx4 v40, s[52:55], 0 offen offset: 0 lds // (op 884) Load value direct to lds





v_mov_b64 v[12:13], s[36:37] // (op 242) convert

s_lshl_b32 s19, s36, 2 // (op 242) 
s_mul_i32 s22, 3, s19 // (op 242) 

v_add_u32 v16, v37, s22 // (op 242) 

s_lshl_b64 s[50:51], s[26:27], 7 // (op 945) 
s_lshl_b64 s[22:23], s[50:51], 1 // (op 945) 

v_mov_b32 v31, s22 // (op 945) 
v_mov_b32 v38, s23 // (op 945) 
v_add_co_u32 v14, s[24:25], v14, v31 // (op 945) least significant half
v_addc_co_u32 v15, s[24:25], v15, v38, s[24:25] // (op 945) most significant half




buffer_load_ubyte v22, v16, s[68:71], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v16, v16, v12 // (op 242) 
buffer_load_ubyte v23, v16, s[68:71], 0 offen offset: 0 // (op 242) Load value



s_barrier  // (op 1511) 
s_lshl_b32 s19, s36, 3 // (op 1073) 



v_add_u32 v21, v21, s19 // (op 1073) 

s_waitcnt vmcnt(14)
ds_write_b96 v18, v[28:30] offset:16384 // (op 13) Store local data 
s_barrier  // (op 1525) 



s_waitcnt vmcnt(9)
ds_write_b8 v27, v24 offset:18432 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v27, v25 offset:18496 // (op 21) Store local data 
s_ashr_i64 s[50:51], s[30:31], 7 // (op 314) 
s_mov_b32 s19, s50 // (op 314) convert

s_and_b32 s22, s19, 4294967294 // (op 314) 

s_cmp_lt_u32 s18, s22 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_314 if true)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_314:














s_barrier  // (op 1604) 
s_mul_hi_i32 s0, s16, s10 // (op 907) 
v_and_b32 v33, -4, v1 // (op 907) 
s_add_i32 s1, s0, s16 // (op 907) 

v_sub_u32 v12, v1, v33 // (op 907) 

s_ashr_i32 s0, s1, 31 // (op 907) 
v_lshlrev_b32 v33, 4, v12 // (op 907) 

s_cmp_eq_i32 s10, 0 // (op 907) 
s_mov_b32 s18, scc // (op 907) 
v_lshrrev_b32 v12, 2, v1 // (op 907) 
s_cmp_eq_i32 s18, 0 // (op 907) 
s_cselect_b32 s19, 0, -1 // (op 907) 

s_and_b32 s18, s13, 31 // (op 907) 
s_lshl_b32 s22, 1, s18 // (op 907) 
s_add_i32 s23, s22, s19 // (op 907) 


s_and_b32 s19, s0, s23 // (op 907) 


s_add_i32 s0, s1, s19 // (op 907) 


s_ashr_i32 s1, s0, s18 // (op 907) 


s_lshl_b32 s0, s13, 24 // (op 907) 
s_ashr_i32 s18, s0, 31 // (op 907) 

s_xor_b32 s0, s1, s18 // (op 907) 

s_sub_i32 s1, s0, s18 // (op 907) 


s_mul_hi_i32 s0, s1, s14 // (op 907) 
s_add_i32 s18, s0, s1 // (op 907) 

s_ashr_i32 s0, s18, 31 // (op 907) 
s_cmp_eq_i32 s14, 0 // (op 907) 
s_mov_b32 s19, scc // (op 907) 
s_cmp_eq_i32 s19, 0 // (op 907) 
s_cselect_b32 s22, 0, -1 // (op 907) 

s_and_b32 s19, s15, 31 // (op 907) 
s_lshl_b32 s23, 1, s19 // (op 907) 
s_add_i32 s24, s23, s22 // (op 907) 


s_and_b32 s25, s0, s24 // (op 907) 


s_add_i32 s0, s18, s25 // (op 907) 


s_ashr_i32 s18, s0, s19 // (op 907) 


s_lshl_b32 s0, s15, 24 // (op 907) 
s_ashr_i32 s19, s0, 31 // (op 907) 

s_xor_b32 s0, s18, s19 // (op 907) 

s_sub_i32 s18, s0, s19 // (op 907) 


s_mul_i32 s0, s1, s11 // (op 907) 

s_sub_u32 s1, s16, s0 // (op 907) 

s_mul_i32 s0, s18, s11 // (op 907) 
s_add_u32 s0, s0, s1 // (op 907) 


s_mul_hi_i32 s1, s16, s9 // (op 907) 
s_add_i32 s18, s1, s16 // (op 907) 

s_ashr_i32 s1, s18, 31 // (op 907) 
s_cmp_eq_i32 s9, 0 // (op 907) 
s_mov_b32 s19, scc // (op 907) 
s_cmp_eq_i32 s19, 0 // (op 907) 
s_cselect_b32 s22, 0, -1 // (op 907) 

s_and_b32 s19, s47, 31 // (op 907) 
s_lshl_b32 s23, 1, s19 // (op 907) 
s_add_i32 s24, s23, s22 // (op 907) 


s_and_b32 s25, s1, s24 // (op 907) 


s_add_i32 s1, s18, s25 // (op 907) 


s_ashr_i32 s18, s1, s19 // (op 907) 


s_lshl_b32 s1, s47, 24 // (op 907) 
s_ashr_i32 s19, s1, 31 // (op 907) 

s_xor_b32 s1, s18, s19 // (op 907) 

s_sub_i32 s18, s1, s19 // (op 907) 


s_add_u32 s50, s6, 16 // (op 907) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 907) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 907) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 907) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 907) 

s_mul_i32 s1, s11, s24 // (op 907) 

s_mul_i32 s19, s8, s1 // (op 907) 

s_mul_i32 s1, s18, s19 // (op 907) 

s_sub_u32 s19, s16, s1 // (op 907) 

s_mov_b32 s1, s19 // (op 907) 
s_mul_hi_i32 s22, s1, s12 // (op 907) 
s_add_i32 s23, s22, s1 // (op 907) 


s_ashr_i32 s1, s23, 31 // (op 907) 
s_cmp_eq_i32 s12, 0 // (op 907) 
s_mov_b32 s22, scc // (op 907) 
s_cmp_eq_i32 s22, 0 // (op 907) 
s_cselect_b32 s24, 0, -1 // (op 907) 

s_and_b32 s22, s46, 31 // (op 907) 
s_lshl_b32 s25, 1, s22 // (op 907) 
s_add_i32 s49, s25, s24 // (op 907) 


s_and_b32 s24, s1, s49 // (op 907) 


s_add_i32 s1, s23, s24 // (op 907) 


s_ashr_i32 s23, s1, s22 // (op 907) 


s_lshl_b32 s1, s46, 24 // (op 907) 
s_ashr_i32 s22, s1, 31 // (op 907) 

s_xor_b32 s1, s23, s22 // (op 907) 

s_sub_i32 s23, s1, s22 // (op 907) 


s_mul_i32 s1, s48, s11 // (op 907) 
s_add_u32 s24, s4, 64 // (op 907) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 907) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 907) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 907) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 907) 

s_sub_i32 s22, s24, s1 // (op 907) 

s_mul_i32 s24, s23, s22 // (op 907) 


s_sub_u32 s25, s19, s24 // (op 907) 


s_add_u32 s19, s25, s1 // (op 907) 


s_cmp_eq_i32 s18, 0 // (op 907) 
s_mov_b32 s1, scc // (op 907) 

s_cmp_eq_i32 s1, 0 // (op 907) 
s_cselect_b32 s18, s19, s0 // (op 907) 



v_lshl_add_u32 v13, s18, 6, v33 // (op 907) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 907) 
v_mov_b32 v15, 0 // (op 907) convert
v_mov_b32 v14, v13 // (op 907) convert

s_lshr_b32 s18, s0, 1 // (op 907) 

v_lshl_add_u32 v13, s18, 8, v12 // (op 907) 


v_mul_lo_u32 v17, v13, s27 // (op 907) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 907) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 907) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 907) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 907) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 907) most significant half



s_mul_i32 s1, s26, s31 // (op 909) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 909) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 909) least significant
s_mul_hi_u32 s19, s26, s30 // (op 909) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 909) most significant: sum
s_add_u32 s1, s1, s19 // (op 909) most significant: sum


s_mov_b32 s52, s28 // (op 909) reg expression
s_mov_b32 s53, s29 // (op 909) reg expression
s_mov_b32 s54, s0 // (op 909) reg expression
s_mov_b32 s55, 131072 // (op 909) call()

s_mul_hi_i32 s0, s16, s10 // (op 975) 
s_add_i32 s1, s0, s16 // (op 975) 

s_ashr_i32 s0, s1, 31 // (op 975) 
s_cmp_eq_i32 s10, 0 // (op 975) 
s_mov_b32 s18, scc // (op 975) 
s_cmp_eq_i32 s18, 0 // (op 975) 
s_cselect_b32 s19, 0, -1 // (op 975) 

s_and_b32 s18, s13, 31 // (op 975) 
s_lshl_b32 s22, 1, s18 // (op 975) 
s_add_i32 s23, s22, s19 // (op 975) 


s_and_b32 s19, s0, s23 // (op 975) 


s_add_i32 s0, s1, s19 // (op 975) 


s_ashr_i32 s1, s0, s18 // (op 975) 


s_lshl_b32 s0, s13, 24 // (op 975) 
s_ashr_i32 s18, s0, 31 // (op 975) 

s_xor_b32 s0, s1, s18 // (op 975) 

s_sub_i32 s1, s0, s18 // (op 975) 


s_mul_hi_i32 s0, s1, s14 // (op 975) 
s_add_i32 s18, s0, s1 // (op 975) 

s_ashr_i32 s0, s18, 31 // (op 975) 
s_cmp_eq_i32 s14, 0 // (op 975) 
s_mov_b32 s19, scc // (op 975) 
s_cmp_eq_i32 s19, 0 // (op 975) 
s_cselect_b32 s22, 0, -1 // (op 975) 

s_and_b32 s19, s15, 31 // (op 975) 
s_lshl_b32 s23, 1, s19 // (op 975) 
s_add_i32 s24, s23, s22 // (op 975) 


s_and_b32 s25, s0, s24 // (op 975) 


s_add_i32 s0, s18, s25 // (op 975) 


s_ashr_i32 s18, s0, s19 // (op 975) 


s_lshl_b32 s0, s15, 24 // (op 975) 
s_ashr_i32 s19, s0, 31 // (op 975) 

s_xor_b32 s0, s18, s19 // (op 975) 

s_sub_i32 s18, s0, s19 // (op 975) 


s_add_u32 s50, s6, 16 // (op 975) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 975) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 975) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 975) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 975) 

s_mul_i32 s0, s18, s24 // (op 975) 

s_sub_i32 s18, s1, s0 // (op 975) 


s_mul_hi_i32 s19, s16, s9 // (op 975) 
s_add_i32 s0, s19, s16 // (op 975) 

s_ashr_i32 s1, s0, 31 // (op 975) 
s_cmp_eq_i32 s9, 0 // (op 975) 
s_mov_b32 s19, scc // (op 975) 
s_cmp_eq_i32 s19, 0 // (op 975) 
s_cselect_b32 s22, 0, -1 // (op 975) 

s_and_b32 s19, s47, 31 // (op 975) 
s_lshl_b32 s23, 1, s19 // (op 975) 
s_add_i32 s49, s23, s22 // (op 975) 


s_and_b32 s22, s1, s49 // (op 975) 


s_add_i32 s1, s0, s22 // (op 975) 


s_ashr_i32 s0, s1, s19 // (op 975) 


s_lshl_b32 s1, s47, 24 // (op 975) 
s_ashr_i32 s19, s1, 31 // (op 975) 

s_xor_b32 s1, s0, s19 // (op 975) 

s_sub_i32 s0, s1, s19 // (op 975) 


s_mul_i32 s1, s11, s24 // (op 975) 

s_mul_i32 s19, s8, s1 // (op 975) 

s_mul_i32 s1, s0, s19 // (op 975) 

s_sub_u32 s19, s16, s1 // (op 975) 

s_mul_hi_i32 s1, s19, s12 // (op 975) 
s_add_i32 s22, s1, s19 // (op 975) 


s_ashr_i32 s1, s22, 31 // (op 975) 
s_cmp_eq_i32 s12, 0 // (op 975) 
s_mov_b32 s19, scc // (op 975) 
s_cmp_eq_i32 s19, 0 // (op 975) 
s_cselect_b32 s23, 0, -1 // (op 975) 

s_and_b32 s19, s46, 31 // (op 975) 
s_lshl_b32 s24, 1, s19 // (op 975) 
s_add_i32 s25, s24, s23 // (op 975) 


s_and_b32 s23, s1, s25 // (op 975) 


s_add_i32 s1, s22, s23 // (op 975) 


s_ashr_i32 s22, s1, s19 // (op 975) 


s_lshl_b32 s1, s46, 24 // (op 975) 
s_ashr_i32 s19, s1, 31 // (op 975) 

s_xor_b32 s1, s22, s19 // (op 975) 

s_sub_i32 s22, s1, s19 // (op 975) 


s_cmp_eq_i32 s0, 0 // (op 975) 
s_mov_b32 s1, scc // (op 975) 

s_cmp_eq_i32 s1, 0 // (op 975) 
s_cselect_b32 s0, s22, s18 // (op 975) 



s_lshl_b32 s1, s0, 4 // (op 975) 

s_mov_b32 s19, 0 // (op 975) convert
s_mov_b32 s18, s1 // (op 975) convert

s_ashr_i64 s[0:1], s[30:31], 7 // (op 975) 
s_lshr_b32 s22, s0, 1 // (op 975) 

v_lshl_add_u32 v33, s22, 8, v1 // (op 975) 

v_mul_lo_u32 v15, v33, s33 // (op 975) most significant: low of low * high
v_mul_hi_u32 v16, v33, s32 // (op 975) most significant: high of low * low
v_mul_lo_u32 v14, v33, s32 // (op 975) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 975) most significant: sum

v_mov_b32 v16, s18 // (op 975) 
v_mov_b32 v17, s19 // (op 975) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 975) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 975) most significant half





v_mul_lo_u32 v17, v15, 6 // (op 975) most significant: low of high * low
v_mul_hi_u32 v33, v14, 6 // (op 975) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 975) least significant: low of low * low
v_add_u32 v17, v17, v33 // (op 975) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 975) 

v_mov_b32 v33, v14 // (op 975) convert

s_mul_i32 s1, s31, 6 // (op 977) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 977) least significant
s_mul_hi_u32 s18, s30, 6 // (op 977) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 977) most significant: sum

s_mul_i32 s19, s32, s1 // (op 977) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 977) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 977) least significant
s_mul_hi_u32 s23, s32, s0 // (op 977) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 977) most significant: sum
s_add_u32 s19, s19, s23 // (op 977) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 977) 

s_mov_b32 s64, s34 // (op 977) reg expression
s_mov_b32 s65, s35 // (op 977) reg expression
s_mov_b32 s66, s0 // (op 977) reg expression
s_mov_b32 s67, 131072 // (op 977) call()

s_mul_hi_i32 s0, s16, s10 // (op 1039) 
v_and_b32 v14, -64, v1 // (op 1039) 
s_add_i32 s1, s0, s16 // (op 1039) 

v_sub_u32 v15, v1, v14 // (op 1039) 

s_ashr_i32 s0, s1, 31 // (op 1039) 
v_lshrrev_b32 v14, 6, v1 // (op 1039) 
s_cmp_eq_i32 s10, 0 // (op 1039) 
s_mov_b32 s18, scc // (op 1039) 
v_lshlrev_b32 v16, 1, v14 // (op 1039) 

s_cmp_eq_i32 s18, 0 // (op 1039) 
s_cselect_b32 s19, 0, -1 // (op 1039) 

s_and_b32 s18, s13, 31 // (op 1039) 
s_lshl_b32 s22, 1, s18 // (op 1039) 
s_add_i32 s23, s22, s19 // (op 1039) 


s_and_b32 s19, s0, s23 // (op 1039) 


s_add_i32 s0, s1, s19 // (op 1039) 


s_ashr_i32 s1, s0, s18 // (op 1039) 


s_lshl_b32 s0, s13, 24 // (op 1039) 
s_ashr_i32 s18, s0, 31 // (op 1039) 

s_xor_b32 s0, s1, s18 // (op 1039) 

s_sub_i32 s1, s0, s18 // (op 1039) 


s_mul_hi_i32 s0, s1, s14 // (op 1039) 
s_add_i32 s18, s0, s1 // (op 1039) 

s_ashr_i32 s0, s18, 31 // (op 1039) 
s_cmp_eq_i32 s14, 0 // (op 1039) 
s_mov_b32 s19, scc // (op 1039) 
s_cmp_eq_i32 s19, 0 // (op 1039) 
s_cselect_b32 s22, 0, -1 // (op 1039) 

s_and_b32 s19, s15, 31 // (op 1039) 
s_lshl_b32 s23, 1, s19 // (op 1039) 
s_add_i32 s24, s23, s22 // (op 1039) 


s_and_b32 s25, s0, s24 // (op 1039) 


s_add_i32 s0, s18, s25 // (op 1039) 


s_ashr_i32 s18, s0, s19 // (op 1039) 


s_lshl_b32 s0, s15, 24 // (op 1039) 
s_ashr_i32 s19, s0, 31 // (op 1039) 

s_xor_b32 s0, s18, s19 // (op 1039) 

s_sub_i32 s18, s0, s19 // (op 1039) 


s_mul_i32 s0, s1, s11 // (op 1039) 

s_sub_u32 s1, s16, s0 // (op 1039) 

s_mul_i32 s0, s18, s11 // (op 1039) 
s_add_u32 s0, s0, s1 // (op 1039) 


s_mul_hi_i32 s1, s16, s9 // (op 1039) 
s_add_i32 s18, s1, s16 // (op 1039) 

s_ashr_i32 s1, s18, 31 // (op 1039) 
s_cmp_eq_i32 s9, 0 // (op 1039) 
s_mov_b32 s19, scc // (op 1039) 
s_cmp_eq_i32 s19, 0 // (op 1039) 
s_cselect_b32 s22, 0, -1 // (op 1039) 

s_and_b32 s19, s47, 31 // (op 1039) 
s_lshl_b32 s23, 1, s19 // (op 1039) 
s_add_i32 s24, s23, s22 // (op 1039) 


s_and_b32 s25, s1, s24 // (op 1039) 


s_add_i32 s1, s18, s25 // (op 1039) 


s_ashr_i32 s18, s1, s19 // (op 1039) 


s_lshl_b32 s1, s47, 24 // (op 1039) 
s_ashr_i32 s19, s1, 31 // (op 1039) 

s_xor_b32 s1, s18, s19 // (op 1039) 

s_sub_i32 s18, s1, s19 // (op 1039) 


s_add_u32 s50, s6, 16 // (op 1039) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1039) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1039) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1039) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1039) 

s_mul_i32 s1, s11, s24 // (op 1039) 

s_mul_i32 s19, s8, s1 // (op 1039) 

s_mul_i32 s1, s18, s19 // (op 1039) 

s_sub_u32 s19, s16, s1 // (op 1039) 

s_mov_b32 s1, s19 // (op 1039) 
s_mul_hi_i32 s22, s1, s12 // (op 1039) 
s_add_i32 s23, s22, s1 // (op 1039) 


s_ashr_i32 s1, s23, 31 // (op 1039) 
s_cmp_eq_i32 s12, 0 // (op 1039) 
s_mov_b32 s22, scc // (op 1039) 
s_cmp_eq_i32 s22, 0 // (op 1039) 
s_cselect_b32 s24, 0, -1 // (op 1039) 

s_and_b32 s22, s46, 31 // (op 1039) 
s_lshl_b32 s25, 1, s22 // (op 1039) 
s_add_i32 s49, s25, s24 // (op 1039) 


s_and_b32 s24, s1, s49 // (op 1039) 


s_add_i32 s1, s23, s24 // (op 1039) 


s_ashr_i32 s23, s1, s22 // (op 1039) 


s_lshl_b32 s1, s46, 24 // (op 1039) 
s_ashr_i32 s22, s1, 31 // (op 1039) 

s_xor_b32 s1, s23, s22 // (op 1039) 

s_sub_i32 s23, s1, s22 // (op 1039) 


s_mul_i32 s1, s48, s11 // (op 1039) 
s_add_u32 s24, s4, 64 // (op 1039) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1039) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1039) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1039) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1039) 

s_sub_i32 s22, s24, s1 // (op 1039) 

s_mul_i32 s24, s23, s22 // (op 1039) 


s_sub_u32 s25, s19, s24 // (op 1039) 


s_add_u32 s19, s25, s1 // (op 1039) 


s_cmp_eq_i32 s18, 0 // (op 1039) 
s_mov_b32 s1, scc // (op 1039) 

s_cmp_eq_i32 s1, 0 // (op 1039) 
s_cselect_b32 s18, s19, s0 // (op 1039) 



v_lshl_add_u32 v14, s18, 6, v15 // (op 1039) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1039) 
s_lshr_b32 s18, s0, 1 // (op 1039) 

v_lshl_add_u32 v15, s18, 3, v16 // (op 1039) 


v_mul_lo_u32 v16, v15, s36 // (op 1039) 

v_add_u32 v15, v14, v16 // (op 1039) 


s_mul_i32 s1, s36, s41 // (op 1041) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1041) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1041) most significant: sum
s_add_u32 s1, s1, s19 // (op 1041) most significant: sum


s_mov_b32 s68, s38 // (op 1041) reg expression
s_mov_b32 s69, s39 // (op 1041) reg expression
s_mov_b32 s70, s0 // (op 1041) reg expression
s_mov_b32 s71, 131072 // (op 1041) call()

s_mul_hi_i32 s0, s16, s10 // (op 1103) 
v_and_b32 v14, 15, v1 // (op 1103) 
s_add_i32 s1, s0, s16 // (op 1103) 

v_and_b32 v16, 63, v1 // (op 1103) 
s_ashr_i32 s0, s1, 31 // (op 1103) 
v_lshrrev_b32 v17, 4, v16 // (op 1103) 

s_cmp_eq_i32 s10, 0 // (op 1103) 
s_mov_b32 s18, scc // (op 1103) 
s_cmp_eq_i32 s18, 0 // (op 1103) 
s_cselect_b32 s19, 0, -1 // (op 1103) 

s_and_b32 s18, s13, 31 // (op 1103) 
s_lshl_b32 s22, 1, s18 // (op 1103) 
s_add_i32 s23, s22, s19 // (op 1103) 


s_and_b32 s19, s0, s23 // (op 1103) 


s_add_i32 s0, s1, s19 // (op 1103) 


s_ashr_i32 s1, s0, s18 // (op 1103) 


s_lshl_b32 s0, s13, 24 // (op 1103) 
s_ashr_i32 s18, s0, 31 // (op 1103) 

s_xor_b32 s0, s1, s18 // (op 1103) 

s_sub_i32 s1, s0, s18 // (op 1103) 


s_mul_hi_i32 s0, s1, s14 // (op 1103) 
s_add_i32 s18, s0, s1 // (op 1103) 

s_ashr_i32 s0, s18, 31 // (op 1103) 
s_cmp_eq_i32 s14, 0 // (op 1103) 
s_mov_b32 s19, scc // (op 1103) 
s_cmp_eq_i32 s19, 0 // (op 1103) 
s_cselect_b32 s22, 0, -1 // (op 1103) 

s_and_b32 s19, s15, 31 // (op 1103) 
s_lshl_b32 s23, 1, s19 // (op 1103) 
s_add_i32 s24, s23, s22 // (op 1103) 


s_and_b32 s25, s0, s24 // (op 1103) 


s_add_i32 s0, s18, s25 // (op 1103) 


s_ashr_i32 s18, s0, s19 // (op 1103) 


s_lshl_b32 s0, s15, 24 // (op 1103) 
s_ashr_i32 s19, s0, 31 // (op 1103) 

s_xor_b32 s0, s18, s19 // (op 1103) 

s_sub_i32 s18, s0, s19 // (op 1103) 


s_add_u32 s50, s6, 16 // (op 1103) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1103) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1103) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1103) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 4 // (op 1103) 

s_mul_i32 s0, s18, s24 // (op 1103) 

s_sub_i32 s18, s1, s0 // (op 1103) 


s_mul_hi_i32 s19, s16, s9 // (op 1103) 
s_add_i32 s0, s19, s16 // (op 1103) 

s_ashr_i32 s1, s0, 31 // (op 1103) 
s_cmp_eq_i32 s9, 0 // (op 1103) 
s_mov_b32 s19, scc // (op 1103) 
s_cmp_eq_i32 s19, 0 // (op 1103) 
s_cselect_b32 s22, 0, -1 // (op 1103) 

s_and_b32 s19, s47, 31 // (op 1103) 
s_lshl_b32 s23, 1, s19 // (op 1103) 
s_add_i32 s49, s23, s22 // (op 1103) 


s_and_b32 s22, s1, s49 // (op 1103) 


s_add_i32 s1, s0, s22 // (op 1103) 


s_ashr_i32 s0, s1, s19 // (op 1103) 


s_lshl_b32 s1, s47, 24 // (op 1103) 
s_ashr_i32 s19, s1, 31 // (op 1103) 

s_xor_b32 s1, s0, s19 // (op 1103) 

s_sub_i32 s0, s1, s19 // (op 1103) 


s_mul_i32 s1, s11, s24 // (op 1103) 

s_mul_i32 s19, s8, s1 // (op 1103) 

s_mul_i32 s1, s0, s19 // (op 1103) 

s_sub_u32 s19, s16, s1 // (op 1103) 

s_mul_hi_i32 s1, s19, s12 // (op 1103) 
s_add_i32 s22, s1, s19 // (op 1103) 


s_ashr_i32 s1, s22, 31 // (op 1103) 
s_cmp_eq_i32 s12, 0 // (op 1103) 
s_mov_b32 s19, scc // (op 1103) 
s_cmp_eq_i32 s19, 0 // (op 1103) 
s_cselect_b32 s23, 0, -1 // (op 1103) 

s_and_b32 s19, s46, 31 // (op 1103) 
s_lshl_b32 s24, 1, s19 // (op 1103) 
s_add_i32 s25, s24, s23 // (op 1103) 


s_and_b32 s23, s1, s25 // (op 1103) 


s_add_i32 s1, s22, s23 // (op 1103) 


s_ashr_i32 s22, s1, s19 // (op 1103) 


s_lshl_b32 s1, s46, 24 // (op 1103) 
s_ashr_i32 s19, s1, 31 // (op 1103) 

s_xor_b32 s1, s22, s19 // (op 1103) 

s_sub_i32 s22, s1, s19 // (op 1103) 


s_cmp_eq_i32 s0, 0 // (op 1103) 
s_mov_b32 s1, scc // (op 1103) 

s_cmp_eq_i32 s1, 0 // (op 1103) 
s_cselect_b32 s0, s22, s18 // (op 1103) 



v_lshl_add_u32 v16, s0, 4, v14 // (op 1103) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1103) 
s_lshr_b32 s18, s0, 1 // (op 1103) 

v_lshl_add_u32 v14, s18, 3, v17 // (op 1103) 


v_mul_lo_u32 v17, v14, s42 // (op 1103) 
v_add_u32 v17, v17, v16 // (op 1103) 


s_mul_i32 s1, s42, s41 // (op 1105) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1105) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1105) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1105) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1105) most significant: sum
s_add_u32 s1, s1, s19 // (op 1105) most significant: sum


s_mov_b32 s72, s44 // (op 1105) reg expression
s_mov_b32 s73, s45 // (op 1105) reg expression
s_mov_b32 s74, s0 // (op 1105) reg expression
s_mov_b32 s75, 131072 // (op 1105) call()

v_and_b32 v14, -4, v1 // (op 1196) 
v_sub_u32 v16, v1, v14 // (op 1196) 

v_lshlrev_b32 v14, 4, v16 // (op 1196) 

v_lshrrev_b32 v16, 2, v1 // (op 1196) 
v_lshl_add_u32 v18, v16, 6, v14 // (op 1196) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 1196) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 4, v1 // (op 1253) 
v_mul_lo_u32 v16, v14, 6 // (op 1253) 

v_lshrrev_b32 v14, 3, v16 // (op 1253) 

v_and_b32 v16, 268435455, v1 // (op 1253) 
v_lshlrev_b32 v18, 2, v16 // (op 1253) 

v_add_u32 v16, v14, v18 // (op 1253) 


v_and_b32 v14, -64, v1 // (op 1306) 
v_sub_u32 v18, v1, v14 // (op 1306) 

v_lshrrev_b32 v14, 6, v1 // (op 1306) 
v_lshl_add_u32 v19, v14, 7, v18 // (op 1306) 


s_ashr_i64 s[18:19], s[30:31], 7 // (op 320) 
s_and_b32 s1, s18, 4294967294 // (op 320) 

s_ashr_i64 s[18:19], s[30:31], 7 // (op 319) 
s_mov_b32 s22, s18 // (op 319) convert

s_cmp_lt_u32 s1, s22 // (op 319) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_319:
s_barrier  // (op 1501) 
s_add_u32 s1, s1, 1 // (op 321) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 878) 
v_mov_b64 v[20:21], s[18:19] // (op 878) convert


v_mov_b64 v[22:23], v[12:13] // (op 878) 
v_mov_b64 v[24:25], v[22:23] // (op 878) reg expression
s_mov_b32 m0, s0 // (op 878) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 878) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 878) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 878) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 878) Load value direct to lds



s_lshl_b64 s[18:19], s[26:27], 7 // (op 897) 
v_mov_b32 v14, s18 // (op 897) 
v_mov_b32 v18, s19 // (op 897) 
v_add_co_u32 v12, s[50:51], v12, v14 // (op 897) least significant half
v_addc_co_u32 v13, s[50:51], v13, v18, s[50:51] // (op 897) most significant half






v_mov_b32 v14, v33 // (op 183) 
v_mov_b32 v18, v14 // (op 183) reg expression

buffer_load_dwordx3 v[20:22], v18, s[64:67], 0 offen // (op 183) Load value


s_lshl_b64 s[18:19], s[32:33], 7 // (op 967) 
s_mul_i32 s51, s19, 6 // (op 967) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 967) least significant
s_mul_hi_u32 s49, s18, 6 // (op 967) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 967) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 967) 

v_add_u32 v33, v33, s18 // (op 967) 


s_barrier  // (op 1514) 
v_mov_b64 v[24:25], s[36:37] // (op 184) convert




v_mov_b32 v14, v15 // (op 184) 
v_mov_b32 v18, v14 // (op 184) reg expression
buffer_load_ubyte v26, v18, s[68:71], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v18, v18, v24 // (op 184) 
s_waitcnt vmcnt(1)
ds_write_b96 v16, v[20:22] offset:16384 // (op 181) Store local data 
buffer_load_ubyte v27, v18, s[68:71], 0 offen offset: 0 // (op 184) Load value




s_lshl_b32 s18, s36, 2 // (op 1031) 
v_add_u32 v15, v15, s18 // (op 1031) 

s_barrier  // (op 1504) 
s_barrier  // (op 1528) 






ds_read_b64_tr_b8 v[34:35], v0 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:512 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:4096 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:4608 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v26 offset:18432 // (op 182) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1518) 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v27 offset:18496 // (op 182) Store local data 




ds_read_b96_tr_b6 v[20:22], v4 offset:16384 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v4 offset:16640 // (op 175) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1532) 


ds_read_u8 v14, v7 offset:18432 // (op 177) Load local data 

v_mov_b64 v[26:27], s[42:43] // (op 179) convert

v_mov_b32 v18, v17 // (op 179) 
v_mov_b32 v28, v18 // (op 179) reg expression
v_mov_b32 v29, v28 // (op 179) 
buffer_load_ubyte v30, v29, s[72:75], 0 offen offset: 0 // (op 179) Load value




s_lshl_b32 s18, s42, 2 // (op 1095) 
v_add_u32 v17, v17, s18 // (op 1095) 




ds_read_b64_tr_b8 v[42:43], v0 offset:32 // (op 325) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:544 // (op 325) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[20:25], a[0:3], v14, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 180) 
ds_read_b64_tr_b8 v[46:47], v0 offset:4128 // (op 325) Transpose load local data 


ds_read_b64_tr_b8 v[48:49], v0 offset:4640 // (op 325) Transpose load local data 


ds_read_u8 v14, v7 offset:18464 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[20:25], a[4:7], v14, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 331) 




s_ashr_i64 s[18:19], s[30:31], 7 // (op 319) 
s_mov_b32 s22, s18 // (op 319) convert

s_cmp_lt_u32 s1, s22 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_319 if true)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_319:






















s_barrier  // (op 1608) 



















v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 

v_lshrrev_b32 v10, 16, v14 // (op 52) 
v_lshrrev_b32 v11, 16, v15 // (op 52) 
v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 


v_pack_b32_f16 v4, v10, v11 // (op 54) 
v_pack_b32_f16 v5, v12, v13 // (op 54) 


buffer_store_dwordx2 v[4:5], v3, s[60:63], 0 offen // (op 54) Store value


v_accvgpr_read v10, a4 // (op 517) 
v_accvgpr_read v11, a5 // (op 517) 
v_accvgpr_read v12, a6 // (op 517) 
v_accvgpr_read v13, a7 // (op 517) 
v_mul_f32 v14, v9, v10 // (op 517) 
v_mul_f32 v15, v9, v11 // (op 517) 
v_mul_f32 v16, v9, v12 // (op 517) 
v_mul_f32 v17, v9, v13 // (op 517) 

v_lshrrev_b32 v10, 16, v14 // (op 519) 
v_lshrrev_b32 v11, 16, v15 // (op 519) 
v_lshrrev_b32 v12, 16, v16 // (op 519) 
v_lshrrev_b32 v13, 16, v17 // (op 519) 


v_pack_b32_f16 v4, v10, v11 // (op 520) 
v_pack_b32_f16 v5, v12, v13 // (op 520) 


buffer_store_dwordx2 v[4:5], v3, s[60:63], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[4:5], v6, s[56:59], 0 offen offset:64 // (op 550) Load value
v_accvgpr_read v10, a4 // (op 546) 
v_accvgpr_read v11, a5 // (op 546) 
v_accvgpr_read v12, a6 // (op 546) 
v_accvgpr_read v13, a7 // (op 546) 
v_mul_f32 v14, v9, v10 // (op 546) 
v_mul_f32 v15, v9, v11 // (op 546) 
v_mul_f32 v16, v9, v12 // (op 546) 
v_mul_f32 v17, v9, v13 // (op 546) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 551) 
v_lshrrev_b32 v11, 16, v4 // (op 551) 
v_lshlrev_b32 v10, 16, v10 // (op 551) 
v_lshlrev_b32 v11, 16, v11 // (op 551) 
v_mul_f32 v18, v8, v10 // (op 551) 
v_mul_f32 v19, v8, v11 // (op 551) 
v_and_b32 v10, 65535, v5 // (op 551) 
v_lshrrev_b32 v11, 16, v5 // (op 551) 
v_lshlrev_b32 v10, 16, v10 // (op 551) 
v_lshlrev_b32 v11, 16, v11 // (op 551) 
v_mul_f32 v20, v8, v10 // (op 551) 
v_mul_f32 v21, v8, v11 // (op 551) 


v_add_f32 v10, v18, v14 // (op 547) 
v_add_f32 v11, v19, v15 // (op 547) 
v_add_f32 v12, v20, v16 // (op 547) 
v_add_f32 v13, v21, v17 // (op 547) 


v_lshrrev_b32 v14, 16, v10 // (op 548) 
v_lshrrev_b32 v15, 16, v11 // (op 548) 
v_lshrrev_b32 v16, 16, v12 // (op 548) 
v_lshrrev_b32 v17, 16, v13 // (op 548) 


v_pack_b32_f16 v4, v14, v15 // (op 549) 
v_pack_b32_f16 v5, v16, v17 // (op 549) 


buffer_store_dwordx2 v[4:5], v3, s[60:63], 0 offen offset:64 // (op 549) Store value





buffer_load_dwordx2 v[4:5], v6, s[56:59], 0 offen // (op 138) Load value
v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 139) 
v_lshrrev_b32 v11, 16, v4 // (op 139) 
v_lshlrev_b32 v10, 16, v10 // (op 139) 
v_lshlrev_b32 v11, 16, v11 // (op 139) 
v_mul_f32 v18, v8, v10 // (op 139) 
v_mul_f32 v19, v8, v11 // (op 139) 
v_and_b32 v10, 65535, v5 // (op 139) 
v_lshrrev_b32 v11, 16, v5 // (op 139) 
v_lshlrev_b32 v10, 16, v10 // (op 139) 
v_lshlrev_b32 v11, 16, v11 // (op 139) 
v_mul_f32 v20, v8, v10 // (op 139) 
v_mul_f32 v21, v8, v11 // (op 139) 


v_add_f32 v10, v18, v14 // (op 135) 
v_add_f32 v11, v19, v15 // (op 135) 
v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 


v_lshrrev_b32 v14, 16, v10 // (op 136) 
v_lshrrev_b32 v15, 16, v11 // (op 136) 
v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 


v_pack_b32_f16 v4, v14, v15 // (op 137) 
v_pack_b32_f16 v5, v16, v17 // (op 137) 


buffer_store_dwordx2 v[4:5], v3, s[60:63], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.LRR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__end:
.size RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_, .LRR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__end-RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 64
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
  - .name: RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 62
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

