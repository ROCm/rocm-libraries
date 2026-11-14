// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_,@function
RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1144) 
s_mul_hi_i32 s1, s16, s10 // (op 1333) 
v_and_b32 v0, 63, v1 // (op 1259) 
v_and_b32 v3, 48, v1 // (op 1259) 
v_sub_u32 v4, v0, v3 // (op 1259) 

v_lshrrev_b32 v3, 4, v0 // (op 1259) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1259) 


v_lshrrev_b32 v3, 6, v1 // (op 1259) 
v_and_b32 v4, 1, v3 // (op 1259) 

v_lshlrev_b32 v3, 4, v4 // (op 1259) 

v_lshl_add_u32 v4, v0, 5, v3 // (op 1259) 


v_mul_lo_u32 v0, v4, 6 // (op 1259) 
v_lshrrev_b32 v3, 3, v0 // (op 1259) 

v_lshrrev_b32 v0, 4, v4 // (op 1259) 

v_lshlrev_b32 v4, 2, v0 // (op 1259) 

v_add_u32 v0, v3, v4 // (op 1259) 


v_and_b32 v3, 63, v1 // (op 1388) 
v_and_b32 v4, 48, v1 // (op 1388) 
v_sub_u32 v5, v3, v4 // (op 1388) 

v_lshrrev_b32 v4, 4, v3 // (op 1388) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 1388) 


v_lshrrev_b32 v4, 6, v1 // (op 1388) 
v_and_b32 v5, 1, v4 // (op 1388) 

v_lshlrev_b32 v4, 4, v5 // (op 1388) 

v_lshl_add_u32 v5, v3, 5, v4 // (op 1388) 


v_mul_lo_u32 v3, v5, 6 // (op 1388) 
v_lshrrev_b32 v4, 3, v3 // (op 1388) 

v_lshrrev_b32 v3, 4, v5 // (op 1388) 

v_lshlrev_b32 v5, 2, v3 // (op 1388) 

v_add_u32 v3, v4, v5 // (op 1388) 


v_lshrrev_b32 v4, 7, v1 // (op 1144) 
s_add_i32 s18, s0, s16 // (op 1144) 

v_and_b32 v5, 63, v1 // (op 1144) 
s_ashr_i32 s0, s18, 31 // (op 1144) 
v_lshrrev_b32 v6, 4, v5 // (op 1144) 
s_cmp_eq_i32 s10, 0 // (op 1144) 
s_mov_b32 s19, scc // (op 1144) 
v_lshlrev_b32 v7, 2, v6 // (op 1144) 

s_cmp_eq_i32 s19, 0 // (op 1144) 
s_cselect_b32 s51, 0, -1 // (op 1144) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1144) 


s_and_b32 s19, s13, 31 // (op 1144) 
v_lshrrev_b32 v4, 2, v5 // (op 1144) 

v_and_b32 v5, 3, v4 // (op 1144) 

v_and_b32 v4, 63, v1 // (op 1206) 
v_and_b32 v7, 48, v1 // (op 1206) 
v_sub_u32 v8, v4, v7 // (op 1206) 

v_lshrrev_b32 v7, 4, v4 // (op 1206) 

v_lshl_add_u32 v4, v7, 5, v8 // (op 1206) 


v_lshrrev_b32 v7, 7, v1 // (op 1206) 
v_lshlrev_b32 v8, 4, v7 // (op 1206) 

v_lshl_add_u32 v7, v4, 6, v8 // (op 1206) 


v_lshlrev_b32 v4, 2, v7 // (op 1206) 

v_lshrrev_b32 v7, 3, v4 // (op 1206) 

v_lshrrev_b32 v4, 7, v1 // (op 1317) 
v_and_b32 v8, 15, v1 // (op 1317) 
v_lshl_add_u32 v9, v4, 4, v8 // (op 1317) 


v_and_b32 v4, 63, v1 // (op 1317) 
v_lshrrev_b32 v8, 4, v4 // (op 1317) 

v_lshl_add_u32 v4, v8, 6, v9 // (op 1317) 


v_lshrrev_b32 v8, 7, v1 // (op 1333) 
v_and_b32 v9, 63, v1 // (op 1333) 
v_and_b32 v10, 63, v1 // (op 1364) 
v_and_b32 v11, 48, v1 // (op 1364) 
v_sub_u32 v12, v10, v11 // (op 1364) 

v_lshrrev_b32 v11, 4, v10 // (op 1364) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 1364) 


v_lshrrev_b32 v11, 7, v1 // (op 1364) 
v_lshlrev_b32 v12, 4, v11 // (op 1364) 

v_lshl_add_u32 v11, v10, 6, v12 // (op 1364) 


v_lshlrev_b32 v10, 2, v11 // (op 1364) 

v_lshrrev_b32 v11, 3, v10 // (op 1364) 

v_lshrrev_b32 v10, 7, v1 // (op 1417) 
v_and_b32 v12, 15, v1 // (op 1417) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 1417) 


v_and_b32 v10, 63, v1 // (op 1417) 
v_lshrrev_b32 v12, 4, v10 // (op 1417) 

v_lshl_add_u32 v10, v12, 6, v13 // (op 1417) 


s_lshl_b32 s52, 1, s19 // (op 1144) 
v_and_b32 v12, 3, v1 // (op 1144) 
s_add_i32 s53, s52, s51 // (op 1144) 


s_and_b32 s51, s0, s53 // (op 1144) 


s_add_i32 s0, s1, s16 // (op 1333) 

s_ashr_i32 s1, s0, 31 // (op 1333) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1144) 


s_add_i32 s52, s18, s51 // (op 1144) 


v_lshrrev_b32 v5, 6, v1 // (op 1144) 
s_ashr_i32 s18, s52, s19 // (op 1144) 


v_and_b32 v12, 1, v5 // (op 1144) 

s_lshl_b32 s19, s13, 24 // (op 1144) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1144) 


s_ashr_i32 s51, s19, 31 // (op 1144) 

s_xor_b32 s19, s18, s51 // (op 1144) 

s_sub_i32 s18, s19, s51 // (op 1144) 


s_mul_hi_i32 s19, s18, s14 // (op 1144) 
s_add_i32 s51, s19, s18 // (op 1144) 

s_ashr_i32 s19, s51, 31 // (op 1144) 
s_cmp_eq_i32 s14, 0 // (op 1144) 
s_mov_b32 s52, scc // (op 1144) 
s_cmp_eq_i32 s52, 0 // (op 1144) 
s_cselect_b32 s53, 0, -1 // (op 1144) 

s_and_b32 s52, s15, 31 // (op 1144) 
v_lshrrev_b32 v12, 4, v9 // (op 1333) 
s_cmp_eq_i32 s10, 0 // (op 1333) 
s_mov_b32 s54, scc // (op 1333) 
v_lshlrev_b32 v13, 2, v12 // (op 1333) 

s_cmp_eq_i32 s54, 0 // (op 1333) 
s_cselect_b32 s55, 0, -1 // (op 1333) 

s_lshl_b32 s54, 1, s52 // (op 1144) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1333) 


v_lshrrev_b32 v8, 2, v9 // (op 1333) 

s_add_i32 s56, s54, s53 // (op 1144) 


s_and_b32 s53, s19, s56 // (op 1144) 


s_add_i32 s19, s51, s53 // (op 1144) 


s_ashr_i32 s51, s19, s52 // (op 1144) 


s_lshl_b32 s19, s15, 24 // (op 1144) 
s_ashr_i32 s52, s19, 31 // (op 1144) 

s_xor_b32 s19, s51, s52 // (op 1144) 

s_sub_i32 s51, s19, s52 // (op 1144) 


s_mul_i32 s19, s18, s11 // (op 1144) 
s_sub_u32 s52, s16, s19 // (op 1144) 

s_mul_i32 s19, s51, s11 // (op 1144) 
s_add_u32 s19, s19, s52 // (op 1144) 

s_mul_hi_i32 s52, s16, s9 // (op 1144) 
s_add_i32 s53, s52, s16 // (op 1144) 

s_ashr_i32 s52, s53, 31 // (op 1144) 
s_cmp_eq_i32 s9, 0 // (op 1144) 
s_mov_b32 s54, scc // (op 1144) 
s_cmp_eq_i32 s54, 0 // (op 1144) 
s_cselect_b32 s56, 0, -1 // (op 1144) 

s_and_b32 s54, s13, 31 // (op 1333) 
v_and_b32 v9, 3, v8 // (op 1333) 

s_lshl_b32 s57, 1, s54 // (op 1333) 
v_and_b32 v8, 3, v1 // (op 1333) 
s_add_i32 s58, s57, s55 // (op 1333) 


s_and_b32 s55, s1, s58 // (op 1333) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1333) 


s_add_i32 s1, s0, s55 // (op 1333) 


v_lshrrev_b32 v8, 6, v1 // (op 1333) 
s_ashr_i32 s0, s1, s54 // (op 1333) 


v_and_b32 v9, 1, v8 // (op 1333) 

s_lshl_b32 s1, s13, 24 // (op 1333) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1333) 


s_ashr_i32 s54, s1, 31 // (op 1333) 

s_xor_b32 s1, s0, s54 // (op 1333) 

s_sub_i32 s0, s1, s54 // (op 1333) 


s_mul_hi_i32 s1, s0, s14 // (op 1333) 
s_add_i32 s54, s1, s0 // (op 1333) 

s_ashr_i32 s1, s54, 31 // (op 1333) 
s_cmp_eq_i32 s14, 0 // (op 1333) 
s_mov_b32 s55, scc // (op 1333) 
s_cmp_eq_i32 s55, 0 // (op 1333) 
s_cselect_b32 s57, 0, -1 // (op 1333) 

s_and_b32 s55, s15, 31 // (op 1333) 
s_lshl_b32 s58, 1, s55 // (op 1333) 
s_add_i32 s59, s58, s57 // (op 1333) 


s_and_b32 s57, s1, s59 // (op 1333) 


s_add_i32 s1, s54, s57 // (op 1333) 


s_ashr_i32 s54, s1, s55 // (op 1333) 


s_lshl_b32 s1, s15, 24 // (op 1333) 
s_ashr_i32 s55, s1, 31 // (op 1333) 

s_xor_b32 s1, s54, s55 // (op 1333) 

s_sub_i32 s54, s1, s55 // (op 1333) 


s_mul_i32 s1, s0, s11 // (op 1333) 
s_sub_u32 s55, s16, s1 // (op 1333) 

s_mul_i32 s1, s54, s11 // (op 1333) 
s_add_u32 s1, s1, s55 // (op 1333) 

s_mul_hi_i32 s55, s16, s9 // (op 1333) 
s_add_i32 s57, s55, s16 // (op 1333) 

s_ashr_i32 s55, s57, 31 // (op 1333) 
s_cmp_eq_i32 s9, 0 // (op 1333) 
s_mov_b32 s58, scc // (op 1333) 
s_cmp_eq_i32 s58, 0 // (op 1333) 
s_cselect_b32 s59, 0, -1 // (op 1333) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s47, 31 // (op 1144) 
s_lshl_b32 s60, 1, s58 // (op 1144) 
s_add_i32 s61, s60, s56 // (op 1144) 


s_and_b32 s56, s52, s61 // (op 1144) 


s_add_i32 s52, s53, s56 // (op 1144) 


s_ashr_i32 s53, s52, s58 // (op 1144) 


s_lshl_b32 s52, s47, 24 // (op 1144) 
s_ashr_i32 s56, s52, 31 // (op 1144) 

s_xor_b32 s52, s53, s56 // (op 1144) 

s_sub_i32 s53, s52, s56 // (op 1144) 


s_and_b32 s52, s47, 31 // (op 1333) 
s_lshl_b32 s56, 1, s52 // (op 1333) 
s_add_i32 s58, s56, s59 // (op 1333) 


s_and_b32 s56, s55, s58 // (op 1333) 


s_add_i32 s55, s57, s56 // (op 1333) 


s_ashr_i32 s56, s55, s52 // (op 1333) 


s_lshl_b32 s52, s47, 24 // (op 1333) 
s_ashr_i32 s55, s52, 31 // (op 1333) 

s_xor_b32 s52, s56, s55 // (op 1333) 

s_sub_i32 s56, s52, s55 // (op 1333) 


s_add_u32 s58, s6, 32 // (op 1144) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 1144) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1144) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1144) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 5 // (op 1144) 

s_mul_i32 s52, s11, s58 // (op 1144) 
s_mul_i32 s55, s8, s52 // (op 1144) 

s_mul_i32 s52, s53, s55 // (op 1144) 

s_sub_u32 s55, s16, s52 // (op 1144) 

s_mov_b32 s52, s55 // (op 1144) 
s_mul_hi_i32 s57, s52, s12 // (op 1144) 
s_add_i32 s60, s57, s52 // (op 1144) 


s_ashr_i32 s52, s60, 31 // (op 1144) 
s_cmp_eq_i32 s12, 0 // (op 1144) 
s_mov_b32 s57, scc // (op 1144) 
s_cmp_eq_i32 s57, 0 // (op 1144) 
s_cselect_b32 s61, 0, -1 // (op 1144) 

s_and_b32 s57, s46, 31 // (op 1144) 
s_lshl_b32 s62, 1, s57 // (op 1144) 
s_add_i32 s63, s62, s61 // (op 1144) 


s_and_b32 s61, s52, s63 // (op 1144) 


s_add_i32 s52, s60, s61 // (op 1144) 


s_ashr_i32 s60, s52, s57 // (op 1144) 


s_lshl_b32 s52, s46, 24 // (op 1144) 
s_ashr_i32 s57, s52, 31 // (op 1144) 

s_xor_b32 s52, s60, s57 // (op 1144) 

s_sub_i32 s60, s52, s57 // (op 1144) 


s_mul_i32 s52, s48, s11 // (op 1144) 
s_add_u32 s62, s4, 64 // (op 1144) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 1144) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1144) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1144) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 1144) 

s_sub_i32 s57, s62, s52 // (op 1144) 

s_mul_i32 s61, s60, s57 // (op 1144) 

s_sub_u32 s57, s55, s61 // (op 1144) 


s_add_u32 s55, s57, s52 // (op 1144) 


s_cmp_eq_i32 s53, 0 // (op 1144) 
s_mov_b32 s52, scc // (op 1144) 

s_cmp_eq_i32 s52, 0 // (op 1144) 
s_cselect_b32 s53, s55, s19 // (op 1144) 


s_mul_i32 s19, s51, s58 // (op 1144) 


v_lshl_add_u32 v9, s53, 6, v6 // (op 1144) 


s_sub_i32 s51, s18, s19 // (op 1144) 


s_cmp_eq_i32 s52, 0 // (op 1144) 
s_cselect_b32 s53, s60, s51 // (op 1144) 



v_lshl_add_u32 v6, s53, 5, v5 // (op 1144) 


v_mul_lo_u32 v5, v6, s2 // (op 1144) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1144) 


s_add_u32 s18, s6, 32 // (op 1333) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1333) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 1333) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 1333) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 1333) 

s_mul_i32 s55, s11, s18 // (op 1333) 
s_mul_i32 s51, s8, s55 // (op 1333) 

s_mul_i32 s55, s56, s51 // (op 1333) 

s_sub_u32 s51, s16, s55 // (op 1333) 

s_mov_b32 s55, s51 // (op 1333) 
s_mul_hi_i32 s52, s55, s12 // (op 1333) 
s_add_i32 s53, s52, s55 // (op 1333) 


s_ashr_i32 s52, s53, 31 // (op 1333) 
s_cmp_eq_i32 s12, 0 // (op 1333) 
s_mov_b32 s55, scc // (op 1333) 
s_cmp_eq_i32 s55, 0 // (op 1333) 
s_cselect_b32 s57, 0, -1 // (op 1333) 

s_and_b32 s55, s46, 31 // (op 1333) 
s_lshl_b32 s58, 1, s55 // (op 1333) 
s_add_i32 s59, s58, s57 // (op 1333) 


s_and_b32 s57, s52, s59 // (op 1333) 


s_add_i32 s52, s53, s57 // (op 1333) 


s_ashr_i32 s53, s52, s55 // (op 1333) 


s_lshl_b32 s52, s46, 24 // (op 1333) 
s_ashr_i32 s55, s52, 31 // (op 1333) 

s_xor_b32 s52, s53, s55 // (op 1333) 

s_sub_i32 s53, s52, s55 // (op 1333) 


s_mul_i32 s52, s48, s11 // (op 1333) 
s_add_u32 s58, s4, 64 // (op 1333) Start of Int64 add, locking SCC
s_addc_u32 s59, s5, 0 // (op 1333) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1333) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1333) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 1333) 

s_sub_i32 s55, s58, s52 // (op 1333) 

s_mul_i32 s57, s53, s55 // (op 1333) 

s_sub_u32 s55, s51, s57 // (op 1333) 


s_add_u32 s51, s55, s52 // (op 1333) 


s_cmp_eq_i32 s56, 0 // (op 1333) 
s_mov_b32 s52, scc // (op 1333) 

s_cmp_eq_i32 s52, 0 // (op 1333) 
s_cselect_b32 s55, s51, s1 // (op 1333) 


s_mul_i32 s1, s54, s18 // (op 1333) 


v_lshl_add_u32 v5, s55, 6, v12 // (op 1333) 


s_sub_i32 s51, s0, s1 // (op 1333) 


s_cmp_eq_i32 s52, 0 // (op 1333) 
s_cselect_b32 s0, s53, s51 // (op 1333) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 1333) 


v_mul_lo_u32 v8, v9, s20 // (op 1333) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 1333) 


s_mul_i32 s1, s2, s7 // (op 1146) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1146) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1146) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1146) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1146) most significant: sum
s_add_u32 s1, s1, s19 // (op 1146) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1146) 

s_mul_i32 s1, s20, s7 // (op 1335) most significant: low of low * high
s_mul_i32 s51, s21, s6 // (op 1335) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 1335) least significant
s_mul_hi_u32 s52, s20, s6 // (op 1335) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1335) most significant: sum
s_add_u32 s1, s1, s52 // (op 1335) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1335) 

s_mov_b32 s56, s22 // (op 1146) reg expression
s_mov_b32 s57, s23 // (op 1146) reg expression
s_mov_b32 s58, s18 // (op 1146) reg expression
s_mov_b32 s59, 131072 // (op 1146) call()

s_mov_b32 s60, s24 // (op 1335) reg expression

s_mov_b32 s61, s25 // (op 1335) reg expression
s_mov_b32 s62, s52 // (op 1335) reg expression
s_mov_b32 s63, 131072 // (op 1335) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
s_mul_hi_i32 s0, s16, s10 // (op 911) 
v_and_b32 v12, -2, v1 // (op 911) 
s_add_i32 s1, s0, s16 // (op 911) 

v_sub_u32 v13, v1, v12 // (op 911) 

s_ashr_i32 s0, s1, 31 // (op 911) 
v_lshlrev_b32 v12, 5, v13 // (op 911) 

s_cmp_eq_i32 s10, 0 // (op 911) 
s_mov_b32 s18, scc // (op 911) 
v_lshrrev_b32 v13, 1, v1 // (op 911) 
s_cmp_eq_i32 s18, 0 // (op 911) 
s_cselect_b32 s19, 0, -1 // (op 911) 

s_and_b32 s18, s13, 31 // (op 911) 
s_lshl_b32 s22, 1, s18 // (op 911) 
s_add_i32 s23, s22, s19 // (op 911) 


s_and_b32 s19, s0, s23 // (op 911) 


s_add_i32 s0, s1, s19 // (op 911) 


s_ashr_i32 s1, s0, s18 // (op 911) 


s_lshl_b32 s0, s13, 24 // (op 911) 
s_ashr_i32 s18, s0, 31 // (op 911) 

s_xor_b32 s0, s1, s18 // (op 911) 

s_sub_i32 s1, s0, s18 // (op 911) 


s_mul_hi_i32 s0, s1, s14 // (op 911) 
s_add_i32 s18, s0, s1 // (op 911) 

s_ashr_i32 s0, s18, 31 // (op 911) 
s_cmp_eq_i32 s14, 0 // (op 911) 
s_mov_b32 s19, scc // (op 911) 
s_cmp_eq_i32 s19, 0 // (op 911) 
s_cselect_b32 s22, 0, -1 // (op 911) 

s_and_b32 s19, s15, 31 // (op 911) 
s_lshl_b32 s23, 1, s19 // (op 911) 
s_add_i32 s24, s23, s22 // (op 911) 


s_and_b32 s25, s0, s24 // (op 911) 


s_add_i32 s0, s18, s25 // (op 911) 


s_ashr_i32 s18, s0, s19 // (op 911) 


s_lshl_b32 s0, s15, 24 // (op 911) 
s_ashr_i32 s19, s0, 31 // (op 911) 

s_xor_b32 s0, s18, s19 // (op 911) 

s_sub_i32 s18, s0, s19 // (op 911) 


s_mul_i32 s0, s1, s11 // (op 911) 

s_sub_u32 s1, s16, s0 // (op 911) 

s_mul_i32 s0, s18, s11 // (op 911) 
s_add_u32 s0, s0, s1 // (op 911) 


s_mul_hi_i32 s1, s16, s9 // (op 911) 
s_add_i32 s18, s1, s16 // (op 911) 

s_ashr_i32 s1, s18, 31 // (op 911) 
s_cmp_eq_i32 s9, 0 // (op 911) 
s_mov_b32 s19, scc // (op 911) 
s_cmp_eq_i32 s19, 0 // (op 911) 
s_cselect_b32 s22, 0, -1 // (op 911) 

s_and_b32 s19, s47, 31 // (op 911) 
s_lshl_b32 s23, 1, s19 // (op 911) 
s_add_i32 s24, s23, s22 // (op 911) 


s_and_b32 s25, s1, s24 // (op 911) 


s_add_i32 s1, s18, s25 // (op 911) 


s_ashr_i32 s18, s1, s19 // (op 911) 


s_lshl_b32 s1, s47, 24 // (op 911) 
s_ashr_i32 s19, s1, 31 // (op 911) 

s_xor_b32 s1, s18, s19 // (op 911) 

s_sub_i32 s18, s1, s19 // (op 911) 


s_add_u32 s22, s6, 32 // (op 911) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 911) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 911) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 911) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 911) 

s_mul_i32 s1, s11, s22 // (op 911) 

s_mul_i32 s19, s8, s1 // (op 911) 

s_mul_i32 s1, s18, s19 // (op 911) 

s_sub_u32 s19, s16, s1 // (op 911) 

s_mov_b32 s1, s19 // (op 911) 
s_mul_hi_i32 s22, s1, s12 // (op 911) 
s_add_i32 s23, s22, s1 // (op 911) 


s_ashr_i32 s1, s23, 31 // (op 911) 
s_cmp_eq_i32 s12, 0 // (op 911) 
s_mov_b32 s22, scc // (op 911) 
s_cmp_eq_i32 s22, 0 // (op 911) 
s_cselect_b32 s24, 0, -1 // (op 911) 

s_and_b32 s22, s46, 31 // (op 911) 
s_lshl_b32 s25, 1, s22 // (op 911) 
s_add_i32 s49, s25, s24 // (op 911) 


s_and_b32 s24, s1, s49 // (op 911) 


s_add_i32 s1, s23, s24 // (op 911) 


s_ashr_i32 s23, s1, s22 // (op 911) 


s_lshl_b32 s1, s46, 24 // (op 911) 
s_ashr_i32 s22, s1, 31 // (op 911) 

s_xor_b32 s1, s23, s22 // (op 911) 

s_sub_i32 s23, s1, s22 // (op 911) 


s_mul_i32 s1, s48, s11 // (op 911) 
s_add_u32 s24, s4, 64 // (op 911) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 911) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 911) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 911) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 911) 

s_sub_i32 s22, s24, s1 // (op 911) 

s_mul_i32 s24, s23, s22 // (op 911) 


s_sub_u32 s25, s19, s24 // (op 911) 


s_add_u32 s19, s25, s1 // (op 911) 


s_cmp_eq_i32 s18, 0 // (op 911) 
s_mov_b32 s1, scc // (op 911) 

s_cmp_eq_i32 s1, 0 // (op 911) 
s_cselect_b32 s18, s19, s0 // (op 911) 



v_lshl_add_u32 v14, s18, 6, v12 // (op 911) 


v_mov_b32 v17, 0 // (op 911) convert
v_mov_b32 v16, v14 // (op 911) convert

v_mul_lo_u32 v15, v13, s27 // (op 911) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 911) most significant: high of low * low
v_mul_lo_u32 v14, v13, s26 // (op 911) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 911) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 911) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 911) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 911) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 911) 

v_mov_b32 v12, v14 // (op 911) convert

s_mul_i32 s1, s26, s31 // (op 913) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 913) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 913) least significant
s_mul_hi_u32 s19, s26, s30 // (op 913) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 913) most significant: sum
s_add_u32 s1, s1, s19 // (op 913) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 913) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 913) 

s_mov_b32 s52, s28 // (op 913) reg expression
s_mov_b32 s53, s29 // (op 913) reg expression
s_mov_b32 s54, s0 // (op 913) reg expression
s_mov_b32 s55, 131072 // (op 913) call()

s_mul_hi_i32 s0, s16, s10 // (op 930) 
v_and_b32 v13, -2, v1 // (op 930) 
s_add_i32 s1, s0, s16 // (op 930) 

v_sub_u32 v14, v1, v13 // (op 930) 

s_ashr_i32 s0, s1, 31 // (op 930) 
v_lshlrev_b32 v13, 5, v14 // (op 930) 

s_cmp_eq_i32 s10, 0 // (op 930) 
s_mov_b32 s18, scc // (op 930) 
v_lshrrev_b32 v14, 1, v1 // (op 930) 
s_cmp_eq_i32 s18, 0 // (op 930) 
s_cselect_b32 s19, 0, -1 // (op 930) 

s_and_b32 s18, s13, 31 // (op 930) 
s_lshl_b32 s22, 1, s18 // (op 930) 
s_add_i32 s23, s22, s19 // (op 930) 


s_and_b32 s19, s0, s23 // (op 930) 


s_add_i32 s0, s1, s19 // (op 930) 


s_ashr_i32 s1, s0, s18 // (op 930) 


s_lshl_b32 s0, s13, 24 // (op 930) 
s_ashr_i32 s18, s0, 31 // (op 930) 

s_xor_b32 s0, s1, s18 // (op 930) 

s_sub_i32 s1, s0, s18 // (op 930) 


s_mul_hi_i32 s0, s1, s14 // (op 930) 
s_add_i32 s18, s0, s1 // (op 930) 

s_ashr_i32 s0, s18, 31 // (op 930) 
s_cmp_eq_i32 s14, 0 // (op 930) 
s_mov_b32 s19, scc // (op 930) 
s_cmp_eq_i32 s19, 0 // (op 930) 
s_cselect_b32 s22, 0, -1 // (op 930) 

s_and_b32 s19, s15, 31 // (op 930) 
s_lshl_b32 s23, 1, s19 // (op 930) 
s_add_i32 s24, s23, s22 // (op 930) 


s_and_b32 s25, s0, s24 // (op 930) 


s_add_i32 s0, s18, s25 // (op 930) 


s_ashr_i32 s18, s0, s19 // (op 930) 


s_lshl_b32 s0, s15, 24 // (op 930) 
s_ashr_i32 s19, s0, 31 // (op 930) 

s_xor_b32 s0, s18, s19 // (op 930) 

s_sub_i32 s18, s0, s19 // (op 930) 


s_mul_i32 s0, s1, s11 // (op 930) 

s_sub_u32 s1, s16, s0 // (op 930) 

s_mul_i32 s0, s18, s11 // (op 930) 
s_add_u32 s0, s0, s1 // (op 930) 


s_mul_hi_i32 s1, s16, s9 // (op 930) 
s_add_i32 s18, s1, s16 // (op 930) 

s_ashr_i32 s1, s18, 31 // (op 930) 
s_cmp_eq_i32 s9, 0 // (op 930) 
s_mov_b32 s19, scc // (op 930) 
s_cmp_eq_i32 s19, 0 // (op 930) 
s_cselect_b32 s22, 0, -1 // (op 930) 

s_and_b32 s19, s47, 31 // (op 930) 
s_lshl_b32 s23, 1, s19 // (op 930) 
s_add_i32 s24, s23, s22 // (op 930) 


s_and_b32 s25, s1, s24 // (op 930) 


s_add_i32 s1, s18, s25 // (op 930) 


s_ashr_i32 s18, s1, s19 // (op 930) 


s_lshl_b32 s1, s47, 24 // (op 930) 
s_ashr_i32 s19, s1, 31 // (op 930) 

s_xor_b32 s1, s18, s19 // (op 930) 

s_sub_i32 s18, s1, s19 // (op 930) 


s_add_u32 s50, s6, 32 // (op 930) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 930) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 930) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 930) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 930) 

s_mul_i32 s1, s11, s24 // (op 930) 

s_mul_i32 s19, s8, s1 // (op 930) 

s_mul_i32 s1, s18, s19 // (op 930) 

s_sub_u32 s19, s16, s1 // (op 930) 

s_mov_b32 s1, s19 // (op 930) 
s_mul_hi_i32 s22, s1, s12 // (op 930) 
s_add_i32 s23, s22, s1 // (op 930) 


s_ashr_i32 s1, s23, 31 // (op 930) 
s_cmp_eq_i32 s12, 0 // (op 930) 
s_mov_b32 s22, scc // (op 930) 
s_cmp_eq_i32 s22, 0 // (op 930) 
s_cselect_b32 s24, 0, -1 // (op 930) 

s_and_b32 s22, s46, 31 // (op 930) 
s_lshl_b32 s25, 1, s22 // (op 930) 
s_add_i32 s49, s25, s24 // (op 930) 


s_and_b32 s24, s1, s49 // (op 930) 


s_add_i32 s1, s23, s24 // (op 930) 


s_ashr_i32 s23, s1, s22 // (op 930) 


s_lshl_b32 s1, s46, 24 // (op 930) 
s_ashr_i32 s22, s1, 31 // (op 930) 

s_xor_b32 s1, s23, s22 // (op 930) 

s_sub_i32 s23, s1, s22 // (op 930) 


s_mul_i32 s1, s48, s11 // (op 930) 
s_add_u32 s24, s4, 64 // (op 930) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 930) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 930) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 930) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 930) 

s_sub_i32 s22, s24, s1 // (op 930) 

s_mul_i32 s24, s23, s22 // (op 930) 


s_sub_u32 s25, s19, s24 // (op 930) 


s_add_u32 s19, s25, s1 // (op 930) 


s_cmp_eq_i32 s18, 0 // (op 930) 
s_mov_b32 s1, scc // (op 930) 

s_cmp_eq_i32 s1, 0 // (op 930) 
s_cselect_b32 s18, s19, s0 // (op 930) 



v_lshl_add_u32 v15, s18, 6, v13 // (op 930) 


v_mov_b32 v17, 0 // (op 930) convert
v_mov_b32 v16, v15 // (op 930) convert

v_mul_lo_u32 v19, v14, s27 // (op 930) most significant: low of low * high
v_mul_hi_u32 v13, v14, s26 // (op 930) most significant: high of low * low
v_mul_lo_u32 v18, v14, s26 // (op 930) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 930) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 930) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 930) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 930) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 930) 

v_mov_b32 v13, v16 // (op 930) convert

s_mul_i32 s1, s26, s31 // (op 932) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 932) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 932) least significant
s_mul_hi_u32 s19, s26, s30 // (op 932) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 932) most significant: sum
s_add_u32 s1, s1, s19 // (op 932) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 932) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 932) 

s_mov_b32 s52, s28 // (op 932) reg expression
s_mov_b32 s53, s29 // (op 932) reg expression
s_mov_b32 s54, s0 // (op 932) reg expression
s_mov_b32 s55, 131072 // (op 932) call()

s_mul_hi_i32 s0, s16, s10 // (op 975) 
v_and_b32 v14, -2, v1 // (op 975) 
s_add_i32 s1, s0, s16 // (op 975) 

v_sub_u32 v15, v1, v14 // (op 975) 

s_ashr_i32 s0, s1, 31 // (op 975) 
v_lshlrev_b32 v14, 4, v15 // (op 975) 

s_cmp_eq_i32 s10, 0 // (op 975) 
s_mov_b32 s18, scc // (op 975) 
v_lshrrev_b32 v15, 1, v1 // (op 975) 
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


s_add_u32 s50, s6, 32 // (op 975) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 975) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 975) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 975) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 975) 

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



v_lshl_add_u32 v16, s0, 5, v14 // (op 975) 


v_mov_b32 v19, 0 // (op 975) convert
v_mov_b32 v18, v16 // (op 975) convert

v_mul_lo_u32 v17, v15, s33 // (op 975) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 975) most significant: high of low * low
v_mul_lo_u32 v16, v15, s32 // (op 975) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 975) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 975) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 975) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 975) most significant: low of high * low
v_mul_hi_u32 v18, v16, 6 // (op 975) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 975) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 975) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 975) 

v_mov_b32 v14, v16 // (op 975) convert

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

s_mul_hi_i32 s0, s16, s10 // (op 994) 
v_and_b32 v15, -2, v1 // (op 994) 
s_add_i32 s1, s0, s16 // (op 994) 

v_sub_u32 v16, v1, v15 // (op 994) 

s_ashr_i32 s0, s1, 31 // (op 994) 
v_lshlrev_b32 v15, 4, v16 // (op 994) 

s_cmp_eq_i32 s10, 0 // (op 994) 
s_mov_b32 s18, scc // (op 994) 
v_lshrrev_b32 v16, 1, v1 // (op 994) 
s_cmp_eq_i32 s18, 0 // (op 994) 
s_cselect_b32 s19, 0, -1 // (op 994) 

s_and_b32 s18, s13, 31 // (op 994) 
s_lshl_b32 s22, 1, s18 // (op 994) 
s_add_i32 s23, s22, s19 // (op 994) 


s_and_b32 s19, s0, s23 // (op 994) 


s_add_i32 s0, s1, s19 // (op 994) 


s_ashr_i32 s1, s0, s18 // (op 994) 


s_lshl_b32 s0, s13, 24 // (op 994) 
s_ashr_i32 s18, s0, 31 // (op 994) 

s_xor_b32 s0, s1, s18 // (op 994) 

s_sub_i32 s1, s0, s18 // (op 994) 


s_mul_hi_i32 s0, s1, s14 // (op 994) 
s_add_i32 s18, s0, s1 // (op 994) 

s_ashr_i32 s0, s18, 31 // (op 994) 
s_cmp_eq_i32 s14, 0 // (op 994) 
s_mov_b32 s19, scc // (op 994) 
s_cmp_eq_i32 s19, 0 // (op 994) 
s_cselect_b32 s22, 0, -1 // (op 994) 

s_and_b32 s19, s15, 31 // (op 994) 
s_lshl_b32 s23, 1, s19 // (op 994) 
s_add_i32 s24, s23, s22 // (op 994) 


s_and_b32 s25, s0, s24 // (op 994) 


s_add_i32 s0, s18, s25 // (op 994) 


s_ashr_i32 s18, s0, s19 // (op 994) 


s_lshl_b32 s0, s15, 24 // (op 994) 
s_ashr_i32 s19, s0, 31 // (op 994) 

s_xor_b32 s0, s18, s19 // (op 994) 

s_sub_i32 s18, s0, s19 // (op 994) 


s_add_u32 s50, s6, 32 // (op 994) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 994) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 994) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 994) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 994) 

s_mul_i32 s0, s18, s24 // (op 994) 

s_sub_i32 s18, s1, s0 // (op 994) 


s_mul_hi_i32 s19, s16, s9 // (op 994) 
s_add_i32 s0, s19, s16 // (op 994) 

s_ashr_i32 s1, s0, 31 // (op 994) 
s_cmp_eq_i32 s9, 0 // (op 994) 
s_mov_b32 s19, scc // (op 994) 
s_cmp_eq_i32 s19, 0 // (op 994) 
s_cselect_b32 s22, 0, -1 // (op 994) 

s_and_b32 s19, s47, 31 // (op 994) 
s_lshl_b32 s23, 1, s19 // (op 994) 
s_add_i32 s49, s23, s22 // (op 994) 


s_and_b32 s22, s1, s49 // (op 994) 


s_add_i32 s1, s0, s22 // (op 994) 


s_ashr_i32 s0, s1, s19 // (op 994) 


s_lshl_b32 s1, s47, 24 // (op 994) 
s_ashr_i32 s19, s1, 31 // (op 994) 

s_xor_b32 s1, s0, s19 // (op 994) 

s_sub_i32 s0, s1, s19 // (op 994) 


s_mul_i32 s1, s11, s24 // (op 994) 

s_mul_i32 s19, s8, s1 // (op 994) 

s_mul_i32 s1, s0, s19 // (op 994) 

s_sub_u32 s19, s16, s1 // (op 994) 

s_mul_hi_i32 s1, s19, s12 // (op 994) 
s_add_i32 s22, s1, s19 // (op 994) 


s_ashr_i32 s1, s22, 31 // (op 994) 
s_cmp_eq_i32 s12, 0 // (op 994) 
s_mov_b32 s19, scc // (op 994) 
s_cmp_eq_i32 s19, 0 // (op 994) 
s_cselect_b32 s23, 0, -1 // (op 994) 

s_and_b32 s19, s46, 31 // (op 994) 
s_lshl_b32 s24, 1, s19 // (op 994) 
s_add_i32 s25, s24, s23 // (op 994) 


s_and_b32 s23, s1, s25 // (op 994) 


s_add_i32 s1, s22, s23 // (op 994) 


s_ashr_i32 s22, s1, s19 // (op 994) 


s_lshl_b32 s1, s46, 24 // (op 994) 
s_ashr_i32 s19, s1, 31 // (op 994) 

s_xor_b32 s1, s22, s19 // (op 994) 

s_sub_i32 s22, s1, s19 // (op 994) 


s_cmp_eq_i32 s0, 0 // (op 994) 
s_mov_b32 s1, scc // (op 994) 

s_cmp_eq_i32 s1, 0 // (op 994) 
s_cselect_b32 s0, s22, s18 // (op 994) 



v_lshl_add_u32 v17, s0, 5, v15 // (op 994) 


v_mov_b32 v19, 0 // (op 994) convert
v_mov_b32 v18, v17 // (op 994) convert

v_mul_lo_u32 v21, v16, s33 // (op 994) most significant: low of low * high
v_mul_hi_u32 v15, v16, s32 // (op 994) most significant: high of low * low
v_mul_lo_u32 v20, v16, s32 // (op 994) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 994) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 994) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 994) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 994) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 994) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 994) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 994) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 994) 

v_mov_b32 v15, v16 // (op 994) convert

s_mul_i32 s1, s31, 6 // (op 996) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 996) least significant
s_mul_hi_u32 s18, s30, 6 // (op 996) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 996) most significant: sum

s_mul_i32 s19, s32, s1 // (op 996) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 996) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 996) least significant
s_mul_hi_u32 s23, s32, s0 // (op 996) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 996) most significant: sum
s_add_u32 s19, s19, s23 // (op 996) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 996) 

s_mov_b32 s64, s34 // (op 996) reg expression
s_mov_b32 s65, s35 // (op 996) reg expression
s_mov_b32 s66, s0 // (op 996) reg expression
s_mov_b32 s67, 131072 // (op 996) call()

s_mul_hi_i32 s0, s16, s10 // (op 1039) 
v_and_b32 v16, -64, v1 // (op 1039) 
s_add_i32 s1, s0, s16 // (op 1039) 

v_sub_u32 v17, v1, v16 // (op 1039) 

s_ashr_i32 s0, s1, 31 // (op 1039) 
v_lshrrev_b32 v16, 6, v1 // (op 1039) 
s_cmp_eq_i32 s10, 0 // (op 1039) 
s_mov_b32 s18, scc // (op 1039) 
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


s_add_u32 s50, s6, 32 // (op 1039) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1039) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1039) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1039) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1039) 

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



v_lshl_add_u32 v18, s18, 6, v17 // (op 1039) 


v_mul_lo_u32 v17, v16, s36 // (op 1039) 

v_add_u32 v16, v18, v17 // (op 1039) 


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

s_mul_hi_i32 s0, s16, s10 // (op 1058) 
v_and_b32 v17, -64, v1 // (op 1058) 
s_add_i32 s1, s0, s16 // (op 1058) 

v_sub_u32 v18, v1, v17 // (op 1058) 

s_ashr_i32 s0, s1, 31 // (op 1058) 
v_lshrrev_b32 v17, 6, v1 // (op 1058) 
s_cmp_eq_i32 s10, 0 // (op 1058) 
s_mov_b32 s18, scc // (op 1058) 
s_cmp_eq_i32 s18, 0 // (op 1058) 
s_cselect_b32 s19, 0, -1 // (op 1058) 

s_and_b32 s18, s13, 31 // (op 1058) 
s_lshl_b32 s22, 1, s18 // (op 1058) 
s_add_i32 s23, s22, s19 // (op 1058) 


s_and_b32 s19, s0, s23 // (op 1058) 


s_add_i32 s0, s1, s19 // (op 1058) 


s_ashr_i32 s1, s0, s18 // (op 1058) 


s_lshl_b32 s0, s13, 24 // (op 1058) 
s_ashr_i32 s18, s0, 31 // (op 1058) 

s_xor_b32 s0, s1, s18 // (op 1058) 

s_sub_i32 s1, s0, s18 // (op 1058) 


s_mul_hi_i32 s0, s1, s14 // (op 1058) 
s_add_i32 s18, s0, s1 // (op 1058) 

s_ashr_i32 s0, s18, 31 // (op 1058) 
s_cmp_eq_i32 s14, 0 // (op 1058) 
s_mov_b32 s19, scc // (op 1058) 
s_cmp_eq_i32 s19, 0 // (op 1058) 
s_cselect_b32 s22, 0, -1 // (op 1058) 

s_and_b32 s19, s15, 31 // (op 1058) 
s_lshl_b32 s23, 1, s19 // (op 1058) 
s_add_i32 s24, s23, s22 // (op 1058) 


s_and_b32 s25, s0, s24 // (op 1058) 


s_add_i32 s0, s18, s25 // (op 1058) 


s_ashr_i32 s18, s0, s19 // (op 1058) 


s_lshl_b32 s0, s15, 24 // (op 1058) 
s_ashr_i32 s19, s0, 31 // (op 1058) 

s_xor_b32 s0, s18, s19 // (op 1058) 

s_sub_i32 s18, s0, s19 // (op 1058) 


s_mul_i32 s0, s1, s11 // (op 1058) 

s_sub_u32 s1, s16, s0 // (op 1058) 

s_mul_i32 s0, s18, s11 // (op 1058) 
s_add_u32 s0, s0, s1 // (op 1058) 


s_mul_hi_i32 s1, s16, s9 // (op 1058) 
s_add_i32 s18, s1, s16 // (op 1058) 

s_ashr_i32 s1, s18, 31 // (op 1058) 
s_cmp_eq_i32 s9, 0 // (op 1058) 
s_mov_b32 s19, scc // (op 1058) 
s_cmp_eq_i32 s19, 0 // (op 1058) 
s_cselect_b32 s22, 0, -1 // (op 1058) 

s_and_b32 s19, s47, 31 // (op 1058) 
s_lshl_b32 s23, 1, s19 // (op 1058) 
s_add_i32 s24, s23, s22 // (op 1058) 


s_and_b32 s25, s1, s24 // (op 1058) 


s_add_i32 s1, s18, s25 // (op 1058) 


s_ashr_i32 s18, s1, s19 // (op 1058) 


s_lshl_b32 s1, s47, 24 // (op 1058) 
s_ashr_i32 s19, s1, 31 // (op 1058) 

s_xor_b32 s1, s18, s19 // (op 1058) 

s_sub_i32 s18, s1, s19 // (op 1058) 


s_add_u32 s50, s6, 32 // (op 1058) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1058) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1058) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1058) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1058) 

s_mul_i32 s1, s11, s24 // (op 1058) 

s_mul_i32 s19, s8, s1 // (op 1058) 

s_mul_i32 s1, s18, s19 // (op 1058) 

s_sub_u32 s19, s16, s1 // (op 1058) 

s_mov_b32 s1, s19 // (op 1058) 
s_mul_hi_i32 s22, s1, s12 // (op 1058) 
s_add_i32 s23, s22, s1 // (op 1058) 


s_ashr_i32 s1, s23, 31 // (op 1058) 
s_cmp_eq_i32 s12, 0 // (op 1058) 
s_mov_b32 s22, scc // (op 1058) 
s_cmp_eq_i32 s22, 0 // (op 1058) 
s_cselect_b32 s24, 0, -1 // (op 1058) 

s_and_b32 s22, s46, 31 // (op 1058) 
s_lshl_b32 s25, 1, s22 // (op 1058) 
s_add_i32 s49, s25, s24 // (op 1058) 


s_and_b32 s24, s1, s49 // (op 1058) 


s_add_i32 s1, s23, s24 // (op 1058) 


s_ashr_i32 s23, s1, s22 // (op 1058) 


s_lshl_b32 s1, s46, 24 // (op 1058) 
s_ashr_i32 s22, s1, 31 // (op 1058) 

s_xor_b32 s1, s23, s22 // (op 1058) 

s_sub_i32 s23, s1, s22 // (op 1058) 


s_mul_i32 s1, s48, s11 // (op 1058) 
s_add_u32 s24, s4, 64 // (op 1058) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1058) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1058) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1058) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1058) 

s_sub_i32 s22, s24, s1 // (op 1058) 

s_mul_i32 s24, s23, s22 // (op 1058) 


s_sub_u32 s25, s19, s24 // (op 1058) 


s_add_u32 s19, s25, s1 // (op 1058) 


s_cmp_eq_i32 s18, 0 // (op 1058) 
s_mov_b32 s1, scc // (op 1058) 

s_cmp_eq_i32 s1, 0 // (op 1058) 
s_cselect_b32 s18, s19, s0 // (op 1058) 



v_lshl_add_u32 v19, s18, 6, v18 // (op 1058) 


v_mul_lo_u32 v18, v17, s36 // (op 1058) 

v_add_u32 v17, v19, v18 // (op 1058) 


s_mul_i32 s1, s36, s41 // (op 1060) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1060) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1060) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1060) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1060) most significant: sum
s_add_u32 s1, s1, s19 // (op 1060) most significant: sum


s_mov_b32 s68, s38 // (op 1060) reg expression
s_mov_b32 s69, s39 // (op 1060) reg expression
s_mov_b32 s70, s0 // (op 1060) reg expression
s_mov_b32 s71, 131072 // (op 1060) call()

s_mul_hi_i32 s0, s16, s10 // (op 1103) 
v_lshrrev_b32 v18, 6, v1 // (op 1103) 
s_add_i32 s1, s0, s16 // (op 1103) 

v_and_b32 v19, 1, v18 // (op 1103) 

s_ashr_i32 s0, s1, 31 // (op 1103) 
v_and_b32 v18, 15, v1 // (op 1103) 
s_cmp_eq_i32 s10, 0 // (op 1103) 
s_mov_b32 s18, scc // (op 1103) 
v_lshl_add_u32 v20, v19, 4, v18 // (op 1103) 


s_cmp_eq_i32 s18, 0 // (op 1103) 
s_cselect_b32 s19, 0, -1 // (op 1103) 

v_and_b32 v18, 63, v1 // (op 1103) 
s_and_b32 s18, s13, 31 // (op 1103) 
v_lshrrev_b32 v19, 4, v18 // (op 1103) 

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


s_add_u32 s50, s6, 32 // (op 1103) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1103) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1103) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1103) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1103) 

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



v_lshl_add_u32 v18, s0, 5, v20 // (op 1103) 


v_mul_lo_u32 v20, v19, s42 // (op 1103) 
v_add_u32 v20, v20, v18 // (op 1103) 


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

s_mul_hi_i32 s0, s16, s10 // (op 1122) 
v_lshrrev_b32 v18, 6, v1 // (op 1122) 
s_add_i32 s1, s0, s16 // (op 1122) 

v_and_b32 v19, 1, v18 // (op 1122) 

s_ashr_i32 s0, s1, 31 // (op 1122) 
v_and_b32 v18, 15, v1 // (op 1122) 
s_cmp_eq_i32 s10, 0 // (op 1122) 
s_mov_b32 s18, scc // (op 1122) 
v_lshl_add_u32 v21, v19, 4, v18 // (op 1122) 


s_cmp_eq_i32 s18, 0 // (op 1122) 
s_cselect_b32 s19, 0, -1 // (op 1122) 

v_and_b32 v18, 63, v1 // (op 1122) 
s_and_b32 s18, s13, 31 // (op 1122) 
v_lshrrev_b32 v19, 4, v18 // (op 1122) 

s_lshl_b32 s22, 1, s18 // (op 1122) 
s_add_i32 s23, s22, s19 // (op 1122) 


s_and_b32 s19, s0, s23 // (op 1122) 


s_add_i32 s0, s1, s19 // (op 1122) 


s_ashr_i32 s1, s0, s18 // (op 1122) 


s_lshl_b32 s0, s13, 24 // (op 1122) 
s_ashr_i32 s18, s0, 31 // (op 1122) 

s_xor_b32 s0, s1, s18 // (op 1122) 

s_sub_i32 s1, s0, s18 // (op 1122) 


s_mul_hi_i32 s0, s1, s14 // (op 1122) 
s_add_i32 s18, s0, s1 // (op 1122) 

s_ashr_i32 s0, s18, 31 // (op 1122) 
s_cmp_eq_i32 s14, 0 // (op 1122) 
s_mov_b32 s19, scc // (op 1122) 
s_cmp_eq_i32 s19, 0 // (op 1122) 
s_cselect_b32 s22, 0, -1 // (op 1122) 

s_and_b32 s19, s15, 31 // (op 1122) 
s_lshl_b32 s23, 1, s19 // (op 1122) 
s_add_i32 s24, s23, s22 // (op 1122) 


s_and_b32 s25, s0, s24 // (op 1122) 


s_add_i32 s0, s18, s25 // (op 1122) 


s_ashr_i32 s18, s0, s19 // (op 1122) 


s_lshl_b32 s0, s15, 24 // (op 1122) 
s_ashr_i32 s19, s0, 31 // (op 1122) 

s_xor_b32 s0, s18, s19 // (op 1122) 

s_sub_i32 s18, s0, s19 // (op 1122) 


s_add_u32 s50, s6, 32 // (op 1122) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1122) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1122) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1122) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1122) 

s_mul_i32 s0, s18, s24 // (op 1122) 

s_sub_i32 s18, s1, s0 // (op 1122) 


s_mul_hi_i32 s19, s16, s9 // (op 1122) 
s_add_i32 s0, s19, s16 // (op 1122) 

s_ashr_i32 s1, s0, 31 // (op 1122) 
s_cmp_eq_i32 s9, 0 // (op 1122) 
s_mov_b32 s19, scc // (op 1122) 
s_cmp_eq_i32 s19, 0 // (op 1122) 
s_cselect_b32 s22, 0, -1 // (op 1122) 

s_and_b32 s19, s47, 31 // (op 1122) 
s_lshl_b32 s23, 1, s19 // (op 1122) 
s_add_i32 s49, s23, s22 // (op 1122) 


s_and_b32 s22, s1, s49 // (op 1122) 


s_add_i32 s1, s0, s22 // (op 1122) 


s_ashr_i32 s0, s1, s19 // (op 1122) 


s_lshl_b32 s1, s47, 24 // (op 1122) 
s_ashr_i32 s19, s1, 31 // (op 1122) 

s_xor_b32 s1, s0, s19 // (op 1122) 

s_sub_i32 s0, s1, s19 // (op 1122) 


s_mul_i32 s1, s11, s24 // (op 1122) 

s_mul_i32 s19, s8, s1 // (op 1122) 

s_mul_i32 s1, s0, s19 // (op 1122) 

s_sub_u32 s19, s16, s1 // (op 1122) 

s_mul_hi_i32 s1, s19, s12 // (op 1122) 
s_add_i32 s22, s1, s19 // (op 1122) 


s_ashr_i32 s1, s22, 31 // (op 1122) 
s_cmp_eq_i32 s12, 0 // (op 1122) 
s_mov_b32 s19, scc // (op 1122) 
s_cmp_eq_i32 s19, 0 // (op 1122) 
s_cselect_b32 s23, 0, -1 // (op 1122) 

s_and_b32 s19, s46, 31 // (op 1122) 
s_lshl_b32 s24, 1, s19 // (op 1122) 
s_add_i32 s25, s24, s23 // (op 1122) 


s_and_b32 s23, s1, s25 // (op 1122) 


s_add_i32 s1, s22, s23 // (op 1122) 


s_ashr_i32 s22, s1, s19 // (op 1122) 


s_lshl_b32 s1, s46, 24 // (op 1122) 
s_ashr_i32 s19, s1, 31 // (op 1122) 

s_xor_b32 s1, s22, s19 // (op 1122) 

s_sub_i32 s22, s1, s19 // (op 1122) 


s_cmp_eq_i32 s0, 0 // (op 1122) 
s_mov_b32 s1, scc // (op 1122) 

s_cmp_eq_i32 s1, 0 // (op 1122) 
s_cselect_b32 s0, s22, s18 // (op 1122) 



v_lshl_add_u32 v18, s0, 5, v21 // (op 1122) 


v_mul_lo_u32 v21, v19, s42 // (op 1122) 
v_add_u32 v21, v21, v18 // (op 1122) 


s_mul_i32 s1, s42, s41 // (op 1124) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1124) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1124) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1124) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1124) most significant: sum
s_add_u32 s1, s1, s19 // (op 1124) most significant: sum


s_mov_b32 s72, s44 // (op 1124) reg expression
s_mov_b32 s73, s45 // (op 1124) reg expression
s_mov_b32 s74, s0 // (op 1124) reg expression
s_mov_b32 s75, 131072 // (op 1124) call()


v_mov_b64 v[18:19], s[42:43] // (op 782) convert

v_mov_b32 v22, v20 // (op 782) reg expression
v_mov_b32 v23, v22 // (op 782) 
buffer_load_ubyte v24, v23, s[72:75], 0 offen offset: 0 // (op 782) Load value





v_mov_b32 v18, v14 // (op 786) reg expression

buffer_load_dwordx3 v[26:28], v18, s[64:67], 0 offen // (op 786) Load value



v_mov_b32 v25, v12 // (op 789) reg expression

buffer_load_dwordx4 v[30:33], v25, s[52:55], 0 offen // (op 789) Load value



v_mov_b32 v25, v16 // (op 792) reg expression

buffer_load_ubyte v29, v25, s[68:71], 0 offen offset: 0 // (op 792) Load value


v_mov_b64 v[18:19], s[42:43] // (op 796) convert

s_lshl_b32 s0, s42, 2 // (op 796) 
v_add_u32 v25, v20, s0 // (op 796) 

v_mov_b32 v22, v25 // (op 796) 
buffer_load_ubyte v23, v22, s[72:75], 0 offen offset: 0 // (op 796) Load value






s_lshl_b64 s[0:1], s[32:33], 7 // (op 800) 
s_mul_i32 s19, s1, 6 // (op 800) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 800) least significant
s_mul_hi_u32 s22, s0, 6 // (op 800) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 800) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 800) 

v_add_u32 v22, v14, s0 // (op 800) 


buffer_load_dwordx3 v[18:20], v22, s[64:67], 0 offen // (op 800) Load value




s_lshl_b64 s[0:1], s[26:27], 7 // (op 803) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 803) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 803) 

v_add_u32 v14, v12, s0 // (op 803) 


buffer_load_dwordx4 v[34:37], v14, s[52:55], 0 offen // (op 803) Load value




s_lshl_b32 s0, s36, 2 // (op 806) 
v_add_u32 v12, v16, s0 // (op 806) 


buffer_load_ubyte v14, v12, s[68:71], 0 offen offset: 0 // (op 806) Load value


v_and_b32 v12, -2, v1 // (op 1247) 
v_sub_u32 v16, v1, v12 // (op 1247) 

v_lshlrev_b32 v12, 4, v16 // (op 1247) 

v_lshrrev_b32 v16, 1, v1 // (op 1247) 
v_lshl_add_u32 v22, v16, 5, v12 // (op 1247) 


v_mul_lo_u32 v12, v22, 6 // (op 1247) 
v_lshrrev_b32 v16, 3, v12 // (op 1247) 

v_lshrrev_b32 v12, 4, v22 // (op 1247) 

v_lshlrev_b32 v22, 2, v12 // (op 1247) 

v_add_u32 v12, v16, v22 // (op 1247) 





s_waitcnt vmcnt(6)
ds_write_b96 v12, v[26:28] // (op 808) Store local data 

v_and_b32 v12, -2, v1 // (op 1189) 
v_sub_u32 v16, v1, v12 // (op 1189) 

v_lshlrev_b32 v12, 5, v16 // (op 1189) 

v_lshrrev_b32 v16, 1, v1 // (op 1189) 
v_lshl_add_u32 v22, v16, 6, v12 // (op 1189) 


v_lshlrev_b32 v12, 2, v22 // (op 1189) 

v_lshrrev_b32 v16, 3, v12 // (op 1189) 




s_waitcnt vmcnt(5)
ds_write_b128 v16, v[30:33] offset:4096 // (op 809) Store local data 

v_and_b32 v12, -64, v1 // (op 1300) 
v_sub_u32 v16, v1, v12 // (op 1300) 

v_and_b32 v12, 4294967232, v1 // (op 1300) 
v_add_u32 v22, v12, v16 // (op 1300) 





s_waitcnt vmcnt(4)
ds_write_b8 v22, v29 offset:8192 // (op 810) Store local data 

v_and_b32 v12, -2, v1 // (op 1160) 
v_sub_u32 v16, v1, v12 // (op 1160) 

v_lshlrev_b32 v12, 5, v16 // (op 1160) 

v_lshrrev_b32 v16, 1, v1 // (op 1160) 
v_lshl_add_u32 v22, v16, 6, v12 // (op 1160) 


v_lshlrev_b32 v12, 2, v22 // (op 1160) 

v_lshrrev_b32 v16, 3, v12 // (op 1160) 

v_and_b32 v12, -2, v1 // (op 1218) 
v_sub_u32 v22, v1, v12 // (op 1218) 

v_lshlrev_b32 v12, 4, v22 // (op 1218) 

v_lshrrev_b32 v22, 1, v1 // (op 1218) 
v_lshl_add_u32 v25, v22, 5, v12 // (op 1218) 


v_mul_lo_u32 v12, v25, 6 // (op 1218) 
v_lshrrev_b32 v22, 3, v12 // (op 1218) 

v_lshrrev_b32 v12, 4, v25 // (op 1218) 

v_lshlrev_b32 v25, 2, v12 // (op 1218) 

v_add_u32 v12, v22, v25 // (op 1218) 


v_and_b32 v22, -64, v1 // (op 1271) 
v_sub_u32 v25, v1, v22 // (op 1271) 

v_and_b32 v22, 4294967232, v1 // (op 1271) 
v_add_u32 v38, v22, v25 // (op 1271) 


v_and_b32 v22, -2, v1 // (op 1347) 
v_sub_u32 v25, v1, v22 // (op 1347) 

v_lshlrev_b32 v22, 5, v25 // (op 1347) 

v_lshrrev_b32 v25, 1, v1 // (op 1347) 
v_lshl_add_u32 v39, v25, 6, v22 // (op 1347) 


v_lshlrev_b32 v22, 2, v39 // (op 1347) 

v_lshrrev_b32 v25, 3, v22 // (op 1347) 

v_and_b32 v22, -2, v1 // (op 1376) 
v_sub_u32 v39, v1, v22 // (op 1376) 

v_lshlrev_b32 v22, 4, v39 // (op 1376) 

v_lshrrev_b32 v39, 1, v1 // (op 1376) 
v_lshl_add_u32 v40, v39, 5, v22 // (op 1376) 


v_mul_lo_u32 v22, v40, 6 // (op 1376) 
v_lshrrev_b32 v39, 3, v22 // (op 1376) 

v_lshrrev_b32 v22, 4, v40 // (op 1376) 

v_lshlrev_b32 v40, 2, v22 // (op 1376) 

v_add_u32 v22, v39, v40 // (op 1376) 


v_and_b32 v39, -64, v1 // (op 1400) 
v_sub_u32 v40, v1, v39 // (op 1400) 

v_and_b32 v39, 4294967232, v1 // (op 1400) 
v_add_u32 v41, v39, v40 // (op 1400) 


s_mov_b32 s0, 0 // (op 315) call()
s_ashr_i64 s[18:19], s[30:31], 7 // (op 314) 
s_mov_b32 s1, s18 // (op 314) convert

s_and_b32 s18, s1, 4294967294 // (op 314) 

s_cmp_lt_u32 s0, s18 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314 if false)
RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314:
s_add_u32 s0, s0, 2 // (op 316) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1458) 



ds_read_b64_tr_b4 v[42:43], v7 offset:4096 // (op 67) Transpose load local data 
ds_read_b64_tr_b4 v[44:45], v7 offset:4608 // (op 67) Transpose load local data 
s_barrier  // (op 1471) 



ds_read_b96_tr_b6 v[46:48], v0 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[49:51], v0 offset:512 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1485) 


ds_read_u8 v39, v4 offset:8192 // (op 77) Load local data 



ds_read_b64_tr_b4 v[52:53], v7 offset:4112 // (op 344) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[46:51], a[0:3], v39, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 84) 
ds_read_b64_tr_b4 v[54:55], v7 offset:4624 // (op 344) Transpose load local data 




ds_read_u8 v39, v4 offset:8224 // (op 348) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[46:51], a[4:7], v39, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 350) 




v_mov_b64 v[42:43], s[42:43] // (op 82) convert

v_mov_b32 v39, v21 // (op 82) 
s_lshl_b32 s1, s42, 3 // (op 82) 
v_add_u32 v40, v39, s1 // (op 82) 

v_mov_b32 v44, v40 // (op 82) 
buffer_load_ubyte v24, v44, s[72:75], 0 offen offset: 0 // (op 82) Load value





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

s_barrier  // (op 1508) 



s_waitcnt vmcnt(6)
ds_write_b96 v22, v[18:20] offset:8448 // (op 239) Store local data 
s_barrier  // (op 1502) 



s_waitcnt vmcnt(5)
ds_write_b128 v25, v[34:37] offset:12544 // (op 229) Store local data 
s_barrier  // (op 1514) 



s_waitcnt vmcnt(4)
ds_write_b8 v41, v14 offset:16640 // (op 240) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1499) 



ds_read_b64_tr_b4 v[44:45], v11 offset:12544 // (op 231) Transpose load local data 
ds_read_b64_tr_b4 v[46:47], v11 offset:13056 // (op 231) Transpose load local data 
s_barrier  // (op 1505) 



ds_read_b96_tr_b6 v[48:50], v3 offset:8448 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v3 offset:8960 // (op 233) Transpose load local data 
s_barrier  // (op 1555) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1511) 


ds_read_u8 v54, v10 offset:16640 // (op 235) Load local data 



ds_read_b64_tr_b4 v[56:57], v11 offset:12560 // (op 357) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:47], v[48:53], a[0:3], v54, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 238) 
ds_read_b64_tr_b4 v[58:59], v11 offset:13072 // (op 357) Transpose load local data 




s_barrier  // (op 1589) 
ds_read_u8 v44, v10 offset:16672 // (op 361) Load local data 
s_barrier  // (op 1593) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[48:53], a[4:7], v44, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 351) 




v_mov_b64 v[44:45], s[42:43] // (op 237) convert

s_lshl_b32 s1, s42, 2 // (op 237) 
s_mul_i32 s18, 3, s1 // (op 237) 

v_add_u32 v46, v39, s18 // (op 237) 

v_mov_b32 v47, v46 // (op 237) 
buffer_load_ubyte v23, v47, s[72:75], 0 offen offset: 0 // (op 237) Load value






s_lshl_b32 s1, s42, 3 // (op 1114) 
v_add_u32 v21, v21, s1 // (op 1114) 

s_lshl_b64 s[18:19], s[32:33], 7 // (op 241) 
s_mul_i32 s51, s19, 6 // (op 241) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 241) least significant
s_mul_hi_u32 s1, s18, 6 // (op 241) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 241) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 241) 

s_mul_i32 s1, 3, s18 // (op 241) 

v_add_u32 v39, v40, s1 // (op 241) 


buffer_load_dwordx3 v[18:20], v39, s[64:67], 0 offen // (op 241) Load value




s_lshl_b64 s[18:19], s[26:27], 7 // (op 228) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 228) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 228) 

s_mul_i32 s1, 3, s18 // (op 228) 

v_add_u32 v39, v42, s1 // (op 228) 


s_lshl_b64 s[18:19], s[32:33], 7 // (op 986) 
s_mul_i32 s51, s19, 6 // (op 986) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 986) least significant
s_mul_hi_u32 s1, s18, 6 // (op 986) most significant: high of low * low
s_add_u32 s51, s51, s1 // (op 986) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 986) 

s_lshl_b32 s1, s18, 1 // (op 986) 

v_add_u32 v15, v15, s1 // (op 986) 

buffer_load_dwordx4 v[34:37], v39, s[52:55], 0 offen // (op 228) Load value




s_lshl_b32 s1, s36, 2 // (op 242) 
s_mul_i32 s18, 3, s1 // (op 242) 

v_add_u32 v42, v43, s18 // (op 242) 


s_lshl_b64 s[18:19], s[26:27], 7 // (op 922) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 922) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 922) 

s_lshl_b32 s1, s18, 1 // (op 922) 

v_add_u32 v13, v13, s1 // (op 922) 

buffer_load_ubyte v14, v42, s[68:71], 0 offen offset: 0 // (op 242) Load value


s_barrier  // (op 1474) 
s_lshl_b32 s1, s36, 3 // (op 1050) 



v_add_u32 v17, v17, s1 // (op 1050) 

s_waitcnt vmcnt(6)
ds_write_b96 v12, v[26:28] // (op 13) Store local data 
s_barrier  // (op 1461) 



s_waitcnt vmcnt(5)
ds_write_b128 v16, v[30:33] offset:4096 // (op 6) Store local data 
s_barrier  // (op 1488) 



s_waitcnt vmcnt(4)
ds_write_b8 v38, v29 offset:8192 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 314) 
s_mov_b32 s1, s18 // (op 314) convert

s_and_b32 s18, s1, 4294967294 // (op 314) 

s_cmp_lt_u32 s0, s18 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314 if true)
RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314:















s_mul_hi_i32 s0, s16, s10 // (op 884) 
v_and_b32 v12, -2, v1 // (op 884) 
s_add_i32 s1, s0, s16 // (op 884) 

v_sub_u32 v13, v1, v12 // (op 884) 

s_ashr_i32 s0, s1, 31 // (op 884) 
v_lshlrev_b32 v12, 5, v13 // (op 884) 

s_cmp_eq_i32 s10, 0 // (op 884) 
s_mov_b32 s18, scc // (op 884) 
v_lshrrev_b32 v13, 1, v1 // (op 884) 
s_cmp_eq_i32 s18, 0 // (op 884) 
s_cselect_b32 s19, 0, -1 // (op 884) 

s_and_b32 s18, s13, 31 // (op 884) 
s_lshl_b32 s22, 1, s18 // (op 884) 
s_add_i32 s23, s22, s19 // (op 884) 


s_and_b32 s19, s0, s23 // (op 884) 


s_add_i32 s0, s1, s19 // (op 884) 


s_ashr_i32 s1, s0, s18 // (op 884) 


s_lshl_b32 s0, s13, 24 // (op 884) 
s_ashr_i32 s18, s0, 31 // (op 884) 

s_xor_b32 s0, s1, s18 // (op 884) 

s_sub_i32 s1, s0, s18 // (op 884) 


s_mul_hi_i32 s0, s1, s14 // (op 884) 
s_add_i32 s18, s0, s1 // (op 884) 

s_ashr_i32 s0, s18, 31 // (op 884) 
s_cmp_eq_i32 s14, 0 // (op 884) 
s_mov_b32 s19, scc // (op 884) 
s_cmp_eq_i32 s19, 0 // (op 884) 
s_cselect_b32 s22, 0, -1 // (op 884) 

s_and_b32 s19, s15, 31 // (op 884) 
s_lshl_b32 s23, 1, s19 // (op 884) 
s_add_i32 s24, s23, s22 // (op 884) 


s_and_b32 s25, s0, s24 // (op 884) 


s_add_i32 s0, s18, s25 // (op 884) 


s_ashr_i32 s18, s0, s19 // (op 884) 


s_lshl_b32 s0, s15, 24 // (op 884) 
s_ashr_i32 s19, s0, 31 // (op 884) 

s_xor_b32 s0, s18, s19 // (op 884) 

s_sub_i32 s18, s0, s19 // (op 884) 


s_mul_i32 s0, s1, s11 // (op 884) 

s_sub_u32 s1, s16, s0 // (op 884) 

s_mul_i32 s0, s18, s11 // (op 884) 
s_add_u32 s0, s0, s1 // (op 884) 


s_mul_hi_i32 s1, s16, s9 // (op 884) 
s_add_i32 s18, s1, s16 // (op 884) 

s_ashr_i32 s1, s18, 31 // (op 884) 
s_cmp_eq_i32 s9, 0 // (op 884) 
s_mov_b32 s19, scc // (op 884) 
s_cmp_eq_i32 s19, 0 // (op 884) 
s_cselect_b32 s22, 0, -1 // (op 884) 

s_and_b32 s19, s47, 31 // (op 884) 
s_lshl_b32 s23, 1, s19 // (op 884) 
s_add_i32 s24, s23, s22 // (op 884) 


s_and_b32 s25, s1, s24 // (op 884) 


s_add_i32 s1, s18, s25 // (op 884) 


s_ashr_i32 s18, s1, s19 // (op 884) 


s_lshl_b32 s1, s47, 24 // (op 884) 
s_ashr_i32 s19, s1, 31 // (op 884) 

s_xor_b32 s1, s18, s19 // (op 884) 

s_sub_i32 s18, s1, s19 // (op 884) 


s_add_u32 s50, s6, 32 // (op 884) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 884) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 884) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 884) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 884) 

s_mul_i32 s1, s11, s24 // (op 884) 

s_mul_i32 s19, s8, s1 // (op 884) 

s_mul_i32 s1, s18, s19 // (op 884) 

s_sub_u32 s19, s16, s1 // (op 884) 

s_mov_b32 s1, s19 // (op 884) 
s_mul_hi_i32 s22, s1, s12 // (op 884) 
s_add_i32 s23, s22, s1 // (op 884) 


s_ashr_i32 s1, s23, 31 // (op 884) 
s_cmp_eq_i32 s12, 0 // (op 884) 
s_mov_b32 s22, scc // (op 884) 
s_cmp_eq_i32 s22, 0 // (op 884) 
s_cselect_b32 s24, 0, -1 // (op 884) 

s_and_b32 s22, s46, 31 // (op 884) 
s_lshl_b32 s25, 1, s22 // (op 884) 
s_add_i32 s49, s25, s24 // (op 884) 


s_and_b32 s24, s1, s49 // (op 884) 


s_add_i32 s1, s23, s24 // (op 884) 


s_ashr_i32 s23, s1, s22 // (op 884) 


s_lshl_b32 s1, s46, 24 // (op 884) 
s_ashr_i32 s22, s1, 31 // (op 884) 

s_xor_b32 s1, s23, s22 // (op 884) 

s_sub_i32 s23, s1, s22 // (op 884) 


s_mul_i32 s1, s48, s11 // (op 884) 
s_add_u32 s24, s4, 64 // (op 884) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 884) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 884) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 884) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 884) 

s_sub_i32 s22, s24, s1 // (op 884) 

s_mul_i32 s24, s23, s22 // (op 884) 


s_sub_u32 s25, s19, s24 // (op 884) 


s_add_u32 s19, s25, s1 // (op 884) 


s_cmp_eq_i32 s18, 0 // (op 884) 
s_mov_b32 s1, scc // (op 884) 

s_cmp_eq_i32 s1, 0 // (op 884) 
s_cselect_b32 s18, s19, s0 // (op 884) 



v_lshl_add_u32 v15, s18, 6, v12 // (op 884) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 884) 
v_mov_b32 v17, 0 // (op 884) convert
v_mov_b32 v16, v15 // (op 884) convert

s_lshr_b32 s18, s0, 1 // (op 884) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 884) 


v_mul_lo_u32 v25, v12, s27 // (op 884) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 884) most significant: high of low * low
v_mul_lo_u32 v24, v12, s26 // (op 884) least significant: low of low * low
v_add_u32 v25, v25, v13 // (op 884) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v24 // (op 884) least significant half
v_addc_co_u32 v13, s[0:1], v17, v25, s[0:1] // (op 884) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 884) 


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 884) 

v_mov_b32 v15, v16 // (op 884) convert

s_mul_i32 s1, s26, s31 // (op 886) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 886) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 886) least significant
s_mul_hi_u32 s19, s26, s30 // (op 886) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 886) most significant: sum
s_add_u32 s1, s1, s19 // (op 886) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 886) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 886) 

s_mov_b32 s52, s28 // (op 886) reg expression
s_mov_b32 s53, s29 // (op 886) reg expression
s_mov_b32 s54, s0 // (op 886) reg expression
s_mov_b32 s55, 131072 // (op 886) call()

s_mul_hi_i32 s0, s16, s10 // (op 952) 
v_and_b32 v12, -2, v1 // (op 952) 
s_add_i32 s1, s0, s16 // (op 952) 

v_sub_u32 v13, v1, v12 // (op 952) 

s_ashr_i32 s0, s1, 31 // (op 952) 
v_lshlrev_b32 v12, 4, v13 // (op 952) 

s_cmp_eq_i32 s10, 0 // (op 952) 
s_mov_b32 s18, scc // (op 952) 
v_lshrrev_b32 v13, 1, v1 // (op 952) 
s_cmp_eq_i32 s18, 0 // (op 952) 
s_cselect_b32 s19, 0, -1 // (op 952) 

s_and_b32 s18, s13, 31 // (op 952) 
s_lshl_b32 s22, 1, s18 // (op 952) 
s_add_i32 s23, s22, s19 // (op 952) 


s_and_b32 s19, s0, s23 // (op 952) 


s_add_i32 s0, s1, s19 // (op 952) 


s_ashr_i32 s1, s0, s18 // (op 952) 


s_lshl_b32 s0, s13, 24 // (op 952) 
s_ashr_i32 s18, s0, 31 // (op 952) 

s_xor_b32 s0, s1, s18 // (op 952) 

s_sub_i32 s1, s0, s18 // (op 952) 


s_mul_hi_i32 s0, s1, s14 // (op 952) 
s_add_i32 s18, s0, s1 // (op 952) 

s_ashr_i32 s0, s18, 31 // (op 952) 
s_cmp_eq_i32 s14, 0 // (op 952) 
s_mov_b32 s19, scc // (op 952) 
s_cmp_eq_i32 s19, 0 // (op 952) 
s_cselect_b32 s22, 0, -1 // (op 952) 

s_and_b32 s19, s15, 31 // (op 952) 
s_lshl_b32 s23, 1, s19 // (op 952) 
s_add_i32 s24, s23, s22 // (op 952) 


s_and_b32 s25, s0, s24 // (op 952) 


s_add_i32 s0, s18, s25 // (op 952) 


s_ashr_i32 s18, s0, s19 // (op 952) 


s_lshl_b32 s0, s15, 24 // (op 952) 
s_ashr_i32 s19, s0, 31 // (op 952) 

s_xor_b32 s0, s18, s19 // (op 952) 

s_sub_i32 s18, s0, s19 // (op 952) 


s_add_u32 s50, s6, 32 // (op 952) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 952) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 952) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 952) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 952) 

s_mul_i32 s0, s18, s24 // (op 952) 

s_sub_i32 s18, s1, s0 // (op 952) 


s_mul_hi_i32 s19, s16, s9 // (op 952) 
s_add_i32 s0, s19, s16 // (op 952) 

s_ashr_i32 s1, s0, 31 // (op 952) 
s_cmp_eq_i32 s9, 0 // (op 952) 
s_mov_b32 s19, scc // (op 952) 
s_cmp_eq_i32 s19, 0 // (op 952) 
s_cselect_b32 s22, 0, -1 // (op 952) 

s_and_b32 s19, s47, 31 // (op 952) 
s_lshl_b32 s23, 1, s19 // (op 952) 
s_add_i32 s49, s23, s22 // (op 952) 


s_and_b32 s22, s1, s49 // (op 952) 


s_add_i32 s1, s0, s22 // (op 952) 


s_ashr_i32 s0, s1, s19 // (op 952) 


s_lshl_b32 s1, s47, 24 // (op 952) 
s_ashr_i32 s19, s1, 31 // (op 952) 

s_xor_b32 s1, s0, s19 // (op 952) 

s_sub_i32 s0, s1, s19 // (op 952) 


s_mul_i32 s1, s11, s24 // (op 952) 

s_mul_i32 s19, s8, s1 // (op 952) 

s_mul_i32 s1, s0, s19 // (op 952) 

s_sub_u32 s19, s16, s1 // (op 952) 

s_mul_hi_i32 s1, s19, s12 // (op 952) 
s_add_i32 s22, s1, s19 // (op 952) 


s_ashr_i32 s1, s22, 31 // (op 952) 
s_cmp_eq_i32 s12, 0 // (op 952) 
s_mov_b32 s19, scc // (op 952) 
s_cmp_eq_i32 s19, 0 // (op 952) 
s_cselect_b32 s23, 0, -1 // (op 952) 

s_and_b32 s19, s46, 31 // (op 952) 
s_lshl_b32 s24, 1, s19 // (op 952) 
s_add_i32 s25, s24, s23 // (op 952) 


s_and_b32 s23, s1, s25 // (op 952) 


s_add_i32 s1, s22, s23 // (op 952) 


s_ashr_i32 s22, s1, s19 // (op 952) 


s_lshl_b32 s1, s46, 24 // (op 952) 
s_ashr_i32 s19, s1, 31 // (op 952) 

s_xor_b32 s1, s22, s19 // (op 952) 

s_sub_i32 s22, s1, s19 // (op 952) 


s_cmp_eq_i32 s0, 0 // (op 952) 
s_mov_b32 s1, scc // (op 952) 

s_cmp_eq_i32 s1, 0 // (op 952) 
s_cselect_b32 s0, s22, s18 // (op 952) 



v_lshl_add_u32 v16, s0, 5, v12 // (op 952) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 952) 
v_mov_b32 v25, 0 // (op 952) convert
v_mov_b32 v24, v16 // (op 952) convert

s_lshr_b32 s18, s0, 1 // (op 952) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 952) 


v_mul_lo_u32 v17, v12, s33 // (op 952) most significant: low of low * high
v_mul_hi_u32 v13, v12, s32 // (op 952) most significant: high of low * low
v_mul_lo_u32 v16, v12, s32 // (op 952) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 952) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v24 // (op 952) least significant half
v_addc_co_u32 v17, s[0:1], v17, v25, s[0:1] // (op 952) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 952) most significant: low of high * low
v_mul_hi_u32 v21, v16, 6 // (op 952) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 952) least significant: low of low * low
v_add_u32 v13, v13, v21 // (op 952) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 952) 

v_mov_b32 v12, v16 // (op 952) convert

s_mul_i32 s1, s31, 6 // (op 954) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 954) least significant
s_mul_hi_u32 s18, s30, 6 // (op 954) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 954) most significant: sum

s_mul_i32 s19, s32, s1 // (op 954) most significant: low of low * high
s_mul_i32 s22, s33, s0 // (op 954) most significant: low of high * low
s_mul_i32 s18, s32, s0 // (op 954) least significant
s_mul_hi_u32 s23, s32, s0 // (op 954) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 954) most significant: sum
s_add_u32 s19, s19, s23 // (op 954) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 954) 

s_mov_b32 s64, s34 // (op 954) reg expression
s_mov_b32 s65, s35 // (op 954) reg expression
s_mov_b32 s66, s0 // (op 954) reg expression
s_mov_b32 s67, 131072 // (op 954) call()

s_mul_hi_i32 s0, s16, s10 // (op 1016) 
v_and_b32 v13, -64, v1 // (op 1016) 
s_add_i32 s1, s0, s16 // (op 1016) 

v_sub_u32 v16, v1, v13 // (op 1016) 

s_ashr_i32 s0, s1, 31 // (op 1016) 
v_lshrrev_b32 v13, 6, v1 // (op 1016) 
s_cmp_eq_i32 s10, 0 // (op 1016) 
s_mov_b32 s18, scc // (op 1016) 
s_cmp_eq_i32 s18, 0 // (op 1016) 
s_cselect_b32 s19, 0, -1 // (op 1016) 

s_and_b32 s18, s13, 31 // (op 1016) 
s_lshl_b32 s22, 1, s18 // (op 1016) 
s_add_i32 s23, s22, s19 // (op 1016) 


s_and_b32 s19, s0, s23 // (op 1016) 


s_add_i32 s0, s1, s19 // (op 1016) 


s_ashr_i32 s1, s0, s18 // (op 1016) 


s_lshl_b32 s0, s13, 24 // (op 1016) 
s_ashr_i32 s18, s0, 31 // (op 1016) 

s_xor_b32 s0, s1, s18 // (op 1016) 

s_sub_i32 s1, s0, s18 // (op 1016) 


s_mul_hi_i32 s0, s1, s14 // (op 1016) 
s_add_i32 s18, s0, s1 // (op 1016) 

s_ashr_i32 s0, s18, 31 // (op 1016) 
s_cmp_eq_i32 s14, 0 // (op 1016) 
s_mov_b32 s19, scc // (op 1016) 
s_cmp_eq_i32 s19, 0 // (op 1016) 
s_cselect_b32 s22, 0, -1 // (op 1016) 

s_and_b32 s19, s15, 31 // (op 1016) 
s_lshl_b32 s23, 1, s19 // (op 1016) 
s_add_i32 s24, s23, s22 // (op 1016) 


s_and_b32 s25, s0, s24 // (op 1016) 


s_add_i32 s0, s18, s25 // (op 1016) 


s_ashr_i32 s18, s0, s19 // (op 1016) 


s_lshl_b32 s0, s15, 24 // (op 1016) 
s_ashr_i32 s19, s0, 31 // (op 1016) 

s_xor_b32 s0, s18, s19 // (op 1016) 

s_sub_i32 s18, s0, s19 // (op 1016) 


s_mul_i32 s0, s1, s11 // (op 1016) 

s_sub_u32 s1, s16, s0 // (op 1016) 

s_mul_i32 s0, s18, s11 // (op 1016) 
s_add_u32 s0, s0, s1 // (op 1016) 


s_mul_hi_i32 s1, s16, s9 // (op 1016) 
s_add_i32 s18, s1, s16 // (op 1016) 

s_ashr_i32 s1, s18, 31 // (op 1016) 
s_cmp_eq_i32 s9, 0 // (op 1016) 
s_mov_b32 s19, scc // (op 1016) 
s_cmp_eq_i32 s19, 0 // (op 1016) 
s_cselect_b32 s22, 0, -1 // (op 1016) 

s_and_b32 s19, s47, 31 // (op 1016) 
s_lshl_b32 s23, 1, s19 // (op 1016) 
s_add_i32 s24, s23, s22 // (op 1016) 


s_and_b32 s25, s1, s24 // (op 1016) 


s_add_i32 s1, s18, s25 // (op 1016) 


s_ashr_i32 s18, s1, s19 // (op 1016) 


s_lshl_b32 s1, s47, 24 // (op 1016) 
s_ashr_i32 s19, s1, 31 // (op 1016) 

s_xor_b32 s1, s18, s19 // (op 1016) 

s_sub_i32 s18, s1, s19 // (op 1016) 


s_add_u32 s50, s6, 32 // (op 1016) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1016) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1016) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1016) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1016) 

s_mul_i32 s1, s11, s24 // (op 1016) 

s_mul_i32 s19, s8, s1 // (op 1016) 

s_mul_i32 s1, s18, s19 // (op 1016) 

s_sub_u32 s19, s16, s1 // (op 1016) 

s_mov_b32 s1, s19 // (op 1016) 
s_mul_hi_i32 s22, s1, s12 // (op 1016) 
s_add_i32 s23, s22, s1 // (op 1016) 


s_ashr_i32 s1, s23, 31 // (op 1016) 
s_cmp_eq_i32 s12, 0 // (op 1016) 
s_mov_b32 s22, scc // (op 1016) 
s_cmp_eq_i32 s22, 0 // (op 1016) 
s_cselect_b32 s24, 0, -1 // (op 1016) 

s_and_b32 s22, s46, 31 // (op 1016) 
s_lshl_b32 s25, 1, s22 // (op 1016) 
s_add_i32 s49, s25, s24 // (op 1016) 


s_and_b32 s24, s1, s49 // (op 1016) 


s_add_i32 s1, s23, s24 // (op 1016) 


s_ashr_i32 s23, s1, s22 // (op 1016) 


s_lshl_b32 s1, s46, 24 // (op 1016) 
s_ashr_i32 s22, s1, 31 // (op 1016) 

s_xor_b32 s1, s23, s22 // (op 1016) 

s_sub_i32 s23, s1, s22 // (op 1016) 


s_mul_i32 s1, s48, s11 // (op 1016) 
s_add_u32 s24, s4, 64 // (op 1016) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 1016) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 1016) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 1016) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 1016) 

s_sub_i32 s22, s24, s1 // (op 1016) 

s_mul_i32 s24, s23, s22 // (op 1016) 


s_sub_u32 s25, s19, s24 // (op 1016) 


s_add_u32 s19, s25, s1 // (op 1016) 


s_cmp_eq_i32 s18, 0 // (op 1016) 
s_mov_b32 s1, scc // (op 1016) 

s_cmp_eq_i32 s1, 0 // (op 1016) 
s_cselect_b32 s18, s19, s0 // (op 1016) 



v_lshl_add_u32 v17, s18, 6, v16 // (op 1016) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1016) 
s_lshr_b32 s18, s0, 1 // (op 1016) 

v_lshl_add_u32 v16, s18, 3, v13 // (op 1016) 


v_mul_lo_u32 v13, v16, s36 // (op 1016) 

v_add_u32 v16, v17, v13 // (op 1016) 


s_mul_i32 s1, s36, s41 // (op 1018) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 1018) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 1018) least significant
s_mul_hi_u32 s19, s36, s40 // (op 1018) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1018) most significant: sum
s_add_u32 s1, s1, s19 // (op 1018) most significant: sum


s_mov_b32 s68, s38 // (op 1018) reg expression
s_mov_b32 s69, s39 // (op 1018) reg expression
s_mov_b32 s70, s0 // (op 1018) reg expression
s_mov_b32 s71, 131072 // (op 1018) call()

s_mul_hi_i32 s0, s16, s10 // (op 1080) 
v_lshrrev_b32 v13, 6, v1 // (op 1080) 
s_add_i32 s1, s0, s16 // (op 1080) 

v_and_b32 v17, 1, v13 // (op 1080) 

s_ashr_i32 s0, s1, 31 // (op 1080) 
v_and_b32 v13, 15, v1 // (op 1080) 
s_cmp_eq_i32 s10, 0 // (op 1080) 
s_mov_b32 s18, scc // (op 1080) 
v_lshl_add_u32 v21, v17, 4, v13 // (op 1080) 


s_cmp_eq_i32 s18, 0 // (op 1080) 
s_cselect_b32 s19, 0, -1 // (op 1080) 

v_and_b32 v13, 63, v1 // (op 1080) 
s_and_b32 s18, s13, 31 // (op 1080) 
v_lshrrev_b32 v17, 4, v13 // (op 1080) 

s_lshl_b32 s22, 1, s18 // (op 1080) 
s_add_i32 s23, s22, s19 // (op 1080) 


s_and_b32 s19, s0, s23 // (op 1080) 


s_add_i32 s0, s1, s19 // (op 1080) 


s_ashr_i32 s1, s0, s18 // (op 1080) 


s_lshl_b32 s0, s13, 24 // (op 1080) 
s_ashr_i32 s18, s0, 31 // (op 1080) 

s_xor_b32 s0, s1, s18 // (op 1080) 

s_sub_i32 s1, s0, s18 // (op 1080) 


s_mul_hi_i32 s0, s1, s14 // (op 1080) 
s_add_i32 s18, s0, s1 // (op 1080) 

s_ashr_i32 s0, s18, 31 // (op 1080) 
s_cmp_eq_i32 s14, 0 // (op 1080) 
s_mov_b32 s19, scc // (op 1080) 
s_cmp_eq_i32 s19, 0 // (op 1080) 
s_cselect_b32 s22, 0, -1 // (op 1080) 

s_and_b32 s19, s15, 31 // (op 1080) 
s_lshl_b32 s23, 1, s19 // (op 1080) 
s_add_i32 s24, s23, s22 // (op 1080) 


s_and_b32 s25, s0, s24 // (op 1080) 


s_add_i32 s0, s18, s25 // (op 1080) 


s_ashr_i32 s18, s0, s19 // (op 1080) 


s_lshl_b32 s0, s15, 24 // (op 1080) 
s_ashr_i32 s19, s0, 31 // (op 1080) 

s_xor_b32 s0, s18, s19 // (op 1080) 

s_sub_i32 s18, s0, s19 // (op 1080) 


s_add_u32 s50, s6, 32 // (op 1080) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1080) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 1080) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 1080) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 1080) 

s_mul_i32 s0, s18, s24 // (op 1080) 

s_sub_i32 s18, s1, s0 // (op 1080) 


s_mul_hi_i32 s19, s16, s9 // (op 1080) 
s_add_i32 s0, s19, s16 // (op 1080) 

s_ashr_i32 s1, s0, 31 // (op 1080) 
s_cmp_eq_i32 s9, 0 // (op 1080) 
s_mov_b32 s19, scc // (op 1080) 
s_cmp_eq_i32 s19, 0 // (op 1080) 
s_cselect_b32 s22, 0, -1 // (op 1080) 

s_and_b32 s19, s47, 31 // (op 1080) 
s_lshl_b32 s23, 1, s19 // (op 1080) 
s_add_i32 s49, s23, s22 // (op 1080) 


s_and_b32 s22, s1, s49 // (op 1080) 


s_add_i32 s1, s0, s22 // (op 1080) 


s_ashr_i32 s0, s1, s19 // (op 1080) 


s_lshl_b32 s1, s47, 24 // (op 1080) 
s_ashr_i32 s19, s1, 31 // (op 1080) 

s_xor_b32 s1, s0, s19 // (op 1080) 

s_sub_i32 s0, s1, s19 // (op 1080) 


s_mul_i32 s1, s11, s24 // (op 1080) 

s_mul_i32 s19, s8, s1 // (op 1080) 

s_mul_i32 s1, s0, s19 // (op 1080) 

s_sub_u32 s19, s16, s1 // (op 1080) 

s_mul_hi_i32 s1, s19, s12 // (op 1080) 
s_add_i32 s22, s1, s19 // (op 1080) 


s_ashr_i32 s1, s22, 31 // (op 1080) 
s_cmp_eq_i32 s12, 0 // (op 1080) 
s_mov_b32 s19, scc // (op 1080) 
s_cmp_eq_i32 s19, 0 // (op 1080) 
s_cselect_b32 s23, 0, -1 // (op 1080) 

s_and_b32 s19, s46, 31 // (op 1080) 
s_lshl_b32 s24, 1, s19 // (op 1080) 
s_add_i32 s25, s24, s23 // (op 1080) 


s_and_b32 s23, s1, s25 // (op 1080) 


s_add_i32 s1, s22, s23 // (op 1080) 


s_ashr_i32 s22, s1, s19 // (op 1080) 


s_lshl_b32 s1, s46, 24 // (op 1080) 
s_ashr_i32 s19, s1, 31 // (op 1080) 

s_xor_b32 s1, s22, s19 // (op 1080) 

s_sub_i32 s22, s1, s19 // (op 1080) 


s_cmp_eq_i32 s0, 0 // (op 1080) 
s_mov_b32 s1, scc // (op 1080) 

s_cmp_eq_i32 s1, 0 // (op 1080) 
s_cselect_b32 s0, s22, s18 // (op 1080) 



v_lshl_add_u32 v13, s0, 5, v21 // (op 1080) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 1080) 
s_lshr_b32 s18, s0, 1 // (op 1080) 

v_lshl_add_u32 v21, s18, 3, v17 // (op 1080) 


v_mul_lo_u32 v17, v21, s42 // (op 1080) 
v_add_u32 v17, v17, v13 // (op 1080) 


s_mul_i32 s1, s42, s41 // (op 1082) most significant: low of low * high
s_mul_i32 s18, s43, s40 // (op 1082) most significant: low of high * low
s_mul_i32 s0, s42, s40 // (op 1082) least significant
s_mul_hi_u32 s19, s42, s40 // (op 1082) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1082) most significant: sum
s_add_u32 s1, s1, s19 // (op 1082) most significant: sum


s_mov_b32 s72, s44 // (op 1082) reg expression
s_mov_b32 s73, s45 // (op 1082) reg expression
s_mov_b32 s74, s0 // (op 1082) reg expression
s_mov_b32 s75, 131072 // (op 1082) call()

v_and_b32 v13, -2, v1 // (op 1173) 
v_sub_u32 v21, v1, v13 // (op 1173) 

v_lshlrev_b32 v13, 5, v21 // (op 1173) 

v_lshrrev_b32 v21, 1, v1 // (op 1173) 
v_lshl_add_u32 v22, v21, 6, v13 // (op 1173) 


v_lshlrev_b32 v13, 2, v22 // (op 1173) 

v_lshrrev_b32 v21, 3, v13 // (op 1173) 

v_and_b32 v13, -2, v1 // (op 1231) 
v_sub_u32 v22, v1, v13 // (op 1231) 

v_lshlrev_b32 v13, 4, v22 // (op 1231) 

v_lshrrev_b32 v22, 1, v1 // (op 1231) 
v_lshl_add_u32 v24, v22, 5, v13 // (op 1231) 


v_mul_lo_u32 v13, v24, 6 // (op 1231) 
v_lshrrev_b32 v22, 3, v13 // (op 1231) 

v_lshrrev_b32 v13, 4, v24 // (op 1231) 

v_lshlrev_b32 v24, 2, v13 // (op 1231) 

v_add_u32 v13, v22, v24 // (op 1231) 


v_and_b32 v22, -64, v1 // (op 1284) 
v_sub_u32 v24, v1, v22 // (op 1284) 

v_and_b32 v22, 4294967232, v1 // (op 1284) 
v_add_u32 v25, v22, v24 // (op 1284) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 320) 
s_and_b32 s18, s0, 4294967294 // (op 320) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 319) 
s_mov_b32 s19, s0 // (op 319) convert

s_cmp_lt_u32 s18, s19 // (op 319) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319:


s_add_u32 s18, s18, 1 // (op 321) 
v_mov_b32 v14, v15 // (op 170) 
v_mov_b32 v18, v14 // (op 170) reg expression

buffer_load_dwordx4 v[26:29], v18, s[52:55], 0 offen // (op 170) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 874) 
s_lshl_b64 s[50:51], s[0:1], 2 // (op 874) 

s_lshr_b64 s[0:1], s[50:51], 3 // (op 874) 

v_add_u32 v15, v15, s0 // (op 874) 



s_barrier  // (op 1464) 
v_mov_b32 v14, v12 // (op 183) 



v_mov_b32 v18, v14 // (op 183) reg expression

buffer_load_dwordx3 v[22:24], v18, s[64:67], 0 offen // (op 183) Load value


s_lshl_b64 s[0:1], s[32:33], 7 // (op 944) 
s_mul_i32 s51, s1, 6 // (op 944) most significant: low of high * low
s_mul_i32 s50, s0, 6 // (op 944) least significant
s_mul_hi_u32 s19, s0, 6 // (op 944) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 944) most significant: sum


s_lshr_b64 s[0:1], s[50:51], 3 // (op 944) 

v_add_u32 v12, v12, s0 // (op 944) 

s_waitcnt vmcnt(1)
ds_write_b128 v21, v[26:29] offset:4096 // (op 171) Store local data 



v_mov_b32 v14, v16 // (op 184) 
s_barrier  // (op 1477) 
v_mov_b32 v18, v14 // (op 184) reg expression




buffer_load_ubyte v19, v18, s[68:71], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s19, s36, 2 // (op 1008) 
v_add_u32 v16, v16, s19 // (op 1008) 

s_waitcnt vmcnt(1)
ds_write_b96 v13, v[22:24] // (op 181) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1467) 



s_barrier  // (op 1491) 
ds_read_b64_tr_b4 v[26:27], v7 offset:4096 // (op 173) Transpose load local data 



ds_read_b64_tr_b4 v[28:29], v7 offset:4608 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v19 offset:8192 // (op 182) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1481) 



ds_read_b96_tr_b6 v[30:32], v0 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v0 offset:512 // (op 175) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1495) 


ds_read_u8 v14, v4 offset:8192 // (op 177) Load local data 

v_mov_b64 v[18:19], s[42:43] // (op 179) convert

v_mov_b32 v20, v17 // (op 179) 
v_mov_b32 v22, v20 // (op 179) reg expression
v_mov_b32 v23, v22 // (op 179) 
buffer_load_ubyte v24, v23, s[72:75], 0 offen offset: 0 // (op 179) Load value




s_lshl_b32 s19, s42, 2 // (op 1072) 
v_add_u32 v17, v17, s19 // (op 1072) 




s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:35], a[0:3], v14, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 180) 
ds_read_b64_tr_b4 v[36:37], v7 offset:4112 // (op 325) Transpose load local data 


ds_read_b64_tr_b4 v[38:39], v7 offset:4624 // (op 325) Transpose load local data 


ds_read_u8 v14, v4 offset:8224 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:39], v[30:35], a[4:7], v14, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 331) 




s_ashr_i64 s[0:1], s[30:31], 7 // (op 319) 
s_mov_b32 s19, s0 // (op 319) convert

s_cmp_lt_u32 s18, s19 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319 if true)
RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319:






















s_barrier  // (op 1577) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 

v_lshrrev_b32 v10, 16, v14 // (op 52) 
v_lshrrev_b32 v11, 16, v15 // (op 52) 
v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 


v_pack_b32_f16 v14, v10, v11 // (op 54) 
v_pack_b32_f16 v15, v12, v13 // (op 54) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen // (op 54) Store value


v_accvgpr_read v10, a4 // (op 517) 
v_accvgpr_read v11, a5 // (op 517) 
v_accvgpr_read v12, a6 // (op 517) 
v_accvgpr_read v13, a7 // (op 517) 
v_mul_f32 v14, v8, v10 // (op 517) 
v_mul_f32 v15, v8, v11 // (op 517) 
v_mul_f32 v16, v8, v12 // (op 517) 
v_mul_f32 v17, v8, v13 // (op 517) 

v_lshrrev_b32 v10, 16, v14 // (op 519) 
v_lshrrev_b32 v11, 16, v15 // (op 519) 
v_lshrrev_b32 v12, 16, v16 // (op 519) 
v_lshrrev_b32 v13, 16, v17 // (op 519) 


v_pack_b32_f16 v14, v10, v11 // (op 520) 
v_pack_b32_f16 v15, v12, v13 // (op 520) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:64 // (op 550) Load value
v_accvgpr_read v12, a4 // (op 546) 
v_accvgpr_read v13, a5 // (op 546) 
v_accvgpr_read v14, a6 // (op 546) 
v_accvgpr_read v15, a7 // (op 546) 
v_mul_f32 v16, v8, v12 // (op 546) 
v_mul_f32 v17, v8, v13 // (op 546) 
v_mul_f32 v18, v8, v14 // (op 546) 
v_mul_f32 v19, v8, v15 // (op 546) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 551) 
v_lshrrev_b32 v13, 16, v10 // (op 551) 
v_lshlrev_b32 v12, 16, v12 // (op 551) 
v_lshlrev_b32 v13, 16, v13 // (op 551) 
v_mul_f32 v20, v5, v12 // (op 551) 
v_mul_f32 v21, v5, v13 // (op 551) 
v_and_b32 v12, 65535, v11 // (op 551) 
v_lshrrev_b32 v13, 16, v11 // (op 551) 
v_lshlrev_b32 v12, 16, v12 // (op 551) 
v_lshlrev_b32 v13, 16, v13 // (op 551) 
v_mul_f32 v22, v5, v12 // (op 551) 
v_mul_f32 v23, v5, v13 // (op 551) 


v_add_f32 v10, v20, v16 // (op 547) 
v_add_f32 v11, v21, v17 // (op 547) 
v_add_f32 v12, v22, v18 // (op 547) 
v_add_f32 v13, v23, v19 // (op 547) 


v_lshrrev_b32 v14, 16, v10 // (op 548) 
v_lshrrev_b32 v15, 16, v11 // (op 548) 
v_lshrrev_b32 v16, 16, v12 // (op 548) 
v_lshrrev_b32 v17, 16, v13 // (op 548) 


v_pack_b32_f16 v10, v14, v15 // (op 549) 
v_pack_b32_f16 v11, v16, v17 // (op 549) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:64 // (op 549) Store value





buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen // (op 138) Load value
v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v8, v12 // (op 134) 
v_mul_f32 v17, v8, v13 // (op 134) 
v_mul_f32 v18, v8, v14 // (op 134) 
v_mul_f32 v19, v8, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 139) 
v_lshrrev_b32 v13, 16, v10 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v20, v5, v12 // (op 139) 
v_mul_f32 v21, v5, v13 // (op 139) 
v_and_b32 v12, 65535, v11 // (op 139) 
v_lshrrev_b32 v13, 16, v11 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v22, v5, v12 // (op 139) 
v_mul_f32 v23, v5, v13 // (op 139) 


v_add_f32 v10, v20, v16 // (op 135) 
v_add_f32 v11, v21, v17 // (op 135) 
v_add_f32 v12, v22, v18 // (op 135) 
v_add_f32 v13, v23, v19 // (op 135) 


v_lshrrev_b32 v14, 16, v10 // (op 136) 
v_lshrrev_b32 v15, 16, v11 // (op 136) 
v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 


v_pack_b32_f16 v10, v14, v15 // (op 137) 
v_pack_b32_f16 v11, v16, v17 // (op 137) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.LRR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end:
.size RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_, .LRR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end-RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
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
    DataTypeB: B6
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
  - .name: RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP4_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_.kd
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

