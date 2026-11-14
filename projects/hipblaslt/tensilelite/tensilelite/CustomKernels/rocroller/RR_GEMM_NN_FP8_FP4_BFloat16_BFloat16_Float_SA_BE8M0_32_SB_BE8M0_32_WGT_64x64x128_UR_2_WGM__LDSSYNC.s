// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC
.p2align 8
.type RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC,@function
RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 2362) 
s_mul_hi_i32 s1, s16, s10 // (op 2619) 
v_and_b32 v0, 48, v1 // (op 2424) 
v_and_b32 v3, 63, v1 // (op 2424) 
v_sub_u32 v4, v3, v0 // (op 2424) 

v_lshrrev_b32 v3, 1, v4 // (op 2424) 
v_add_lshl_u32 v5, v0, v3, 6 // (op 2424) 


v_and_b32 v0, 1, v4 // (op 2424) 

v_lshlrev_b32 v3, 3, v0 // (op 2424) 

v_lshrrev_b32 v0, 7, v1 // (op 2424) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2424) 


v_add_u32 v0, v5, v4 // (op 2424) 


v_lshrrev_b32 v3, 6, v1 // (op 2482) 
v_and_b32 v4, 1, v3 // (op 2482) 

v_and_b32 v3, 15, v1 // (op 2482) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 2482) 


v_and_b32 v3, 63, v1 // (op 2482) 
v_lshrrev_b32 v4, 4, v3 // (op 2482) 

v_lshlrev_b32 v3, 5, v4 // (op 2482) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 2482) 


v_lshlrev_b32 v3, 2, v4 // (op 2482) 

v_lshrrev_b32 v4, 3, v3 // (op 2482) 

v_lshrrev_b32 v3, 7, v1 // (op 2540) 
v_and_b32 v5, 15, v1 // (op 2540) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 2540) 


v_and_b32 v3, 63, v1 // (op 2540) 
v_lshrrev_b32 v5, 4, v3 // (op 2540) 

v_lshl_add_u32 v3, v5, 6, v6 // (op 2540) 


v_lshrrev_b32 v5, 6, v1 // (op 2598) 
v_and_b32 v6, 1, v5 // (op 2598) 

v_and_b32 v5, 15, v1 // (op 2598) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2598) 


v_and_b32 v5, 63, v1 // (op 2598) 
v_lshrrev_b32 v6, 4, v5 // (op 2598) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 2598) 


v_and_b32 v6, 48, v1 // (op 2650) 
v_and_b32 v7, 63, v1 // (op 2650) 
v_sub_u32 v8, v7, v6 // (op 2650) 

v_lshrrev_b32 v7, 1, v8 // (op 2650) 
v_add_lshl_u32 v9, v6, v7, 6 // (op 2650) 


v_and_b32 v6, 1, v8 // (op 2650) 

v_lshlrev_b32 v7, 3, v6 // (op 2650) 

v_lshrrev_b32 v6, 7, v1 // (op 2650) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 2650) 


v_add_u32 v6, v9, v8 // (op 2650) 


v_lshrrev_b32 v7, 6, v1 // (op 2679) 
v_and_b32 v8, 1, v7 // (op 2679) 

v_and_b32 v7, 15, v1 // (op 2679) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 2679) 


v_and_b32 v7, 63, v1 // (op 2679) 
v_lshrrev_b32 v8, 4, v7 // (op 2679) 

v_lshlrev_b32 v7, 5, v8 // (op 2679) 

v_lshl_add_u32 v8, v9, 7, v7 // (op 2679) 


v_lshlrev_b32 v7, 2, v8 // (op 2679) 

v_lshrrev_b32 v8, 3, v7 // (op 2679) 

v_lshrrev_b32 v7, 7, v1 // (op 2708) 
v_and_b32 v9, 15, v1 // (op 2708) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 2708) 


v_and_b32 v7, 63, v1 // (op 2708) 
v_lshrrev_b32 v9, 4, v7 // (op 2708) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 2708) 


v_lshrrev_b32 v9, 6, v1 // (op 2737) 
v_and_b32 v10, 1, v9 // (op 2737) 

v_and_b32 v9, 15, v1 // (op 2737) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 2737) 


v_and_b32 v9, 63, v1 // (op 2737) 
v_lshrrev_b32 v10, 4, v9 // (op 2737) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 2737) 


v_lshrrev_b32 v10, 7, v1 // (op 2362) 
s_add_i32 s18, s0, s16 // (op 2362) 

v_and_b32 v11, 63, v1 // (op 2362) 
s_ashr_i32 s0, s18, 31 // (op 2362) 
v_lshrrev_b32 v12, 4, v11 // (op 2362) 
s_cmp_eq_i32 s10, 0 // (op 2362) 
s_mov_b32 s19, scc // (op 2362) 
v_lshlrev_b32 v13, 2, v12 // (op 2362) 

s_cmp_eq_i32 s19, 0 // (op 2362) 
s_cselect_b32 s51, 0, -1 // (op 2362) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 2362) 


s_and_b32 s19, s11, 31 // (op 2362) 
v_lshrrev_b32 v10, 2, v11 // (op 2362) 

v_and_b32 v11, 3, v10 // (op 2362) 

v_lshrrev_b32 v10, 7, v1 // (op 2619) 
v_and_b32 v13, 63, v1 // (op 2619) 
s_lshl_b32 s52, 1, s19 // (op 2362) 
v_and_b32 v14, 3, v1 // (op 2362) 
s_add_i32 s53, s52, s51 // (op 2362) 


s_and_b32 s51, s0, s53 // (op 2362) 


s_add_i32 s0, s1, s16 // (op 2619) 

s_ashr_i32 s1, s0, 31 // (op 2619) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 2362) 


s_add_i32 s52, s18, s51 // (op 2362) 


v_lshrrev_b32 v11, 6, v1 // (op 2362) 
s_ashr_i32 s18, s52, s19 // (op 2362) 


v_and_b32 v14, 1, v11 // (op 2362) 

s_lshl_b32 s19, s11, 24 // (op 2362) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 2362) 


s_ashr_i32 s51, s19, 31 // (op 2362) 

s_xor_b32 s19, s18, s51 // (op 2362) 

s_sub_i32 s18, s19, s51 // (op 2362) 


s_mul_hi_i32 s19, s18, s12 // (op 2362) 
s_add_i32 s51, s19, s18 // (op 2362) 

s_ashr_i32 s19, s51, 31 // (op 2362) 
s_cmp_eq_i32 s12, 0 // (op 2362) 
s_mov_b32 s52, scc // (op 2362) 
s_cmp_eq_i32 s52, 0 // (op 2362) 
s_cselect_b32 s53, 0, -1 // (op 2362) 

s_and_b32 s52, s13, 31 // (op 2362) 
v_lshrrev_b32 v14, 4, v13 // (op 2619) 
s_cmp_eq_i32 s10, 0 // (op 2619) 
s_mov_b32 s54, scc // (op 2619) 
v_lshlrev_b32 v15, 2, v14 // (op 2619) 

s_cmp_eq_i32 s54, 0 // (op 2619) 
s_cselect_b32 s55, 0, -1 // (op 2619) 

s_lshl_b32 s54, 1, s52 // (op 2362) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 2619) 


v_lshrrev_b32 v10, 2, v13 // (op 2619) 

s_add_i32 s56, s54, s53 // (op 2362) 


s_and_b32 s53, s19, s56 // (op 2362) 


s_add_i32 s19, s51, s53 // (op 2362) 


s_ashr_i32 s51, s19, s52 // (op 2362) 


s_lshl_b32 s19, s13, 24 // (op 2362) 
s_ashr_i32 s52, s19, 31 // (op 2362) 

s_xor_b32 s19, s51, s52 // (op 2362) 

s_sub_i32 s51, s19, s52 // (op 2362) 


s_mul_i32 s19, s18, s15 // (op 2362) 
s_sub_u32 s52, s16, s19 // (op 2362) 

s_mul_i32 s19, s51, s15 // (op 2362) 
s_add_u32 s19, s19, s52 // (op 2362) 

s_mul_hi_i32 s52, s16, s9 // (op 2362) 
s_add_i32 s53, s52, s16 // (op 2362) 

s_ashr_i32 s52, s53, 31 // (op 2362) 
s_cmp_eq_i32 s9, 0 // (op 2362) 
s_mov_b32 s54, scc // (op 2362) 
s_cmp_eq_i32 s54, 0 // (op 2362) 
s_cselect_b32 s56, 0, -1 // (op 2362) 

s_and_b32 s54, s11, 31 // (op 2619) 
v_and_b32 v13, 3, v10 // (op 2619) 

s_lshl_b32 s57, 1, s54 // (op 2619) 
v_and_b32 v10, 3, v1 // (op 2619) 
s_add_i32 s58, s57, s55 // (op 2619) 


s_and_b32 s55, s1, s58 // (op 2619) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 2619) 


s_add_i32 s1, s0, s55 // (op 2619) 


v_lshrrev_b32 v10, 6, v1 // (op 2619) 
s_ashr_i32 s0, s1, s54 // (op 2619) 


v_and_b32 v13, 1, v10 // (op 2619) 

s_lshl_b32 s1, s11, 24 // (op 2619) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 2619) 


s_ashr_i32 s54, s1, 31 // (op 2619) 

s_xor_b32 s1, s0, s54 // (op 2619) 

s_sub_i32 s0, s1, s54 // (op 2619) 


s_mul_hi_i32 s1, s0, s12 // (op 2619) 
s_add_i32 s54, s1, s0 // (op 2619) 

s_ashr_i32 s1, s54, 31 // (op 2619) 
s_cmp_eq_i32 s12, 0 // (op 2619) 
s_mov_b32 s55, scc // (op 2619) 
s_cmp_eq_i32 s55, 0 // (op 2619) 
s_cselect_b32 s57, 0, -1 // (op 2619) 

s_and_b32 s55, s13, 31 // (op 2619) 
s_lshl_b32 s58, 1, s55 // (op 2619) 
s_add_i32 s59, s58, s57 // (op 2619) 


s_and_b32 s57, s1, s59 // (op 2619) 


s_add_i32 s1, s54, s57 // (op 2619) 


s_ashr_i32 s54, s1, s55 // (op 2619) 


s_lshl_b32 s1, s13, 24 // (op 2619) 
s_ashr_i32 s55, s1, 31 // (op 2619) 

s_xor_b32 s1, s54, s55 // (op 2619) 

s_sub_i32 s54, s1, s55 // (op 2619) 


s_mul_i32 s1, s0, s15 // (op 2619) 
s_sub_u32 s55, s16, s1 // (op 2619) 

s_mul_i32 s1, s54, s15 // (op 2619) 
s_add_u32 s1, s1, s55 // (op 2619) 

s_mul_hi_i32 s55, s16, s9 // (op 2619) 
s_add_i32 s57, s55, s16 // (op 2619) 

s_ashr_i32 s55, s57, 31 // (op 2619) 
s_cmp_eq_i32 s9, 0 // (op 2619) 
s_mov_b32 s58, scc // (op 2619) 
s_cmp_eq_i32 s58, 0 // (op 2619) 
s_cselect_b32 s59, 0, -1 // (op 2619) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 2362) 
s_lshl_b32 s60, 1, s58 // (op 2362) 
s_add_i32 s61, s60, s56 // (op 2362) 


s_and_b32 s56, s52, s61 // (op 2362) 


s_add_i32 s52, s53, s56 // (op 2362) 


s_ashr_i32 s53, s52, s58 // (op 2362) 


s_lshl_b32 s52, s46, 24 // (op 2362) 
s_ashr_i32 s56, s52, 31 // (op 2362) 

s_xor_b32 s52, s53, s56 // (op 2362) 

s_sub_i32 s53, s52, s56 // (op 2362) 


s_and_b32 s52, s46, 31 // (op 2619) 
s_lshl_b32 s56, 1, s52 // (op 2619) 
s_add_i32 s58, s56, s59 // (op 2619) 


s_and_b32 s56, s55, s58 // (op 2619) 


s_add_i32 s55, s57, s56 // (op 2619) 


s_ashr_i32 s56, s55, s52 // (op 2619) 


s_lshl_b32 s52, s46, 24 // (op 2619) 
s_ashr_i32 s55, s52, 31 // (op 2619) 

s_xor_b32 s52, s56, s55 // (op 2619) 

s_sub_i32 s56, s52, s55 // (op 2619) 


s_add_u32 s58, s6, 64 // (op 2362) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 2362) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2362) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2362) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2362) 

s_mul_i32 s52, s15, s58 // (op 2362) 
s_mul_i32 s55, s8, s52 // (op 2362) 

s_mul_i32 s52, s53, s55 // (op 2362) 

s_sub_u32 s55, s16, s52 // (op 2362) 

s_mov_b32 s52, s55 // (op 2362) 
s_mul_hi_i32 s57, s52, s14 // (op 2362) 
s_add_i32 s60, s57, s52 // (op 2362) 


s_ashr_i32 s52, s60, 31 // (op 2362) 
s_cmp_eq_i32 s14, 0 // (op 2362) 
s_mov_b32 s57, scc // (op 2362) 
s_cmp_eq_i32 s57, 0 // (op 2362) 
s_cselect_b32 s61, 0, -1 // (op 2362) 

s_and_b32 s57, s48, 31 // (op 2362) 
s_lshl_b32 s62, 1, s57 // (op 2362) 
s_add_i32 s63, s62, s61 // (op 2362) 


s_and_b32 s61, s52, s63 // (op 2362) 


s_add_i32 s52, s60, s61 // (op 2362) 


s_ashr_i32 s60, s52, s57 // (op 2362) 


s_lshl_b32 s52, s48, 24 // (op 2362) 
s_ashr_i32 s57, s52, 31 // (op 2362) 

s_xor_b32 s52, s60, s57 // (op 2362) 

s_sub_i32 s60, s52, s57 // (op 2362) 


s_mul_i32 s52, s47, s15 // (op 2362) 
s_add_u32 s62, s20, 64 // (op 2362) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 2362) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2362) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2362) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 2362) 

s_sub_i32 s57, s62, s52 // (op 2362) 

s_mul_i32 s61, s60, s57 // (op 2362) 

s_sub_u32 s57, s55, s61 // (op 2362) 


s_add_u32 s55, s57, s52 // (op 2362) 


s_cmp_eq_i32 s53, 0 // (op 2362) 
s_mov_b32 s52, scc // (op 2362) 

s_cmp_eq_i32 s52, 0 // (op 2362) 
s_cselect_b32 s53, s55, s19 // (op 2362) 


s_mul_i32 s19, s51, s58 // (op 2362) 


v_lshl_add_u32 v13, s53, 6, v12 // (op 2362) 


s_sub_i32 s51, s18, s19 // (op 2362) 


s_cmp_eq_i32 s52, 0 // (op 2362) 
s_cselect_b32 s53, s60, s51 // (op 2362) 



v_lshl_add_u32 v12, s53, 6, v11 // (op 2362) 


v_mul_lo_u32 v11, v12, s2 // (op 2362) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 2362) 


s_add_u32 s18, s6, 64 // (op 2619) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 2619) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 2619) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 2619) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 6 // (op 2619) 

s_mul_i32 s55, s15, s18 // (op 2619) 
s_mul_i32 s51, s8, s55 // (op 2619) 

s_mul_i32 s55, s56, s51 // (op 2619) 

s_sub_u32 s51, s16, s55 // (op 2619) 

s_mov_b32 s55, s51 // (op 2619) 
s_mul_hi_i32 s52, s55, s14 // (op 2619) 
s_add_i32 s53, s52, s55 // (op 2619) 


s_ashr_i32 s52, s53, 31 // (op 2619) 
s_cmp_eq_i32 s14, 0 // (op 2619) 
s_mov_b32 s55, scc // (op 2619) 
s_cmp_eq_i32 s55, 0 // (op 2619) 
s_cselect_b32 s57, 0, -1 // (op 2619) 

s_and_b32 s55, s48, 31 // (op 2619) 
s_lshl_b32 s58, 1, s55 // (op 2619) 
s_add_i32 s59, s58, s57 // (op 2619) 


s_and_b32 s57, s52, s59 // (op 2619) 


s_add_i32 s52, s53, s57 // (op 2619) 


s_ashr_i32 s53, s52, s55 // (op 2619) 


s_lshl_b32 s52, s48, 24 // (op 2619) 
s_ashr_i32 s55, s52, 31 // (op 2619) 

s_xor_b32 s52, s53, s55 // (op 2619) 

s_sub_i32 s53, s52, s55 // (op 2619) 


s_mul_i32 s52, s47, s15 // (op 2619) 
s_add_u32 s58, s20, 64 // (op 2619) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 2619) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2619) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2619) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2619) 

s_sub_i32 s55, s58, s52 // (op 2619) 

s_mul_i32 s57, s53, s55 // (op 2619) 

s_sub_u32 s55, s51, s57 // (op 2619) 


s_add_u32 s51, s55, s52 // (op 2619) 


s_cmp_eq_i32 s56, 0 // (op 2619) 
s_mov_b32 s52, scc // (op 2619) 

s_cmp_eq_i32 s52, 0 // (op 2619) 
s_cselect_b32 s55, s51, s1 // (op 2619) 


s_mul_i32 s1, s54, s18 // (op 2619) 


v_lshl_add_u32 v11, s55, 6, v14 // (op 2619) 


s_sub_i32 s51, s0, s1 // (op 2619) 


s_cmp_eq_i32 s52, 0 // (op 2619) 
s_cselect_b32 s0, s53, s51 // (op 2619) 



v_lshl_add_u32 v13, s0, 6, v10 // (op 2619) 


v_mul_lo_u32 v10, v13, s4 // (op 2619) 

v_add_lshl_u32 v13, v11, v10, 1 // (op 2619) 


s_mul_i32 s1, s2, s7 // (op 2364) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 2364) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 2364) least significant
s_mul_hi_u32 s19, s2, s6 // (op 2364) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2364) most significant: sum
s_add_u32 s1, s1, s19 // (op 2364) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2364) 

s_mul_i32 s1, s4, s7 // (op 2621) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 2621) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 2621) least significant
s_mul_hi_u32 s52, s4, s6 // (op 2621) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2621) most significant: sum
s_add_u32 s1, s1, s52 // (op 2621) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 2621) 

s_mov_b32 s56, s22 // (op 2364) reg expression
s_mov_b32 s57, s23 // (op 2364) reg expression
s_mov_b32 s58, s18 // (op 2364) reg expression
s_mov_b32 s59, 131072 // (op 2364) call()

s_mov_b32 s60, s24 // (op 2621) reg expression

s_mov_b32 s61, s25 // (op 2621) reg expression
s_mov_b32 s62, s52 // (op 2621) reg expression
s_mov_b32 s63, 131072 // (op 2621) call()


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
v_accvgpr_write a8, 0.00000 // (op 560) call()
v_accvgpr_write a9, 0.00000 // (op 560) call()
v_accvgpr_write a10, 0.00000 // (op 560) call()
v_accvgpr_write a11, 0.00000 // (op 560) call()
v_accvgpr_write a12, 0.00000 // (op 715) call()
v_accvgpr_write a13, 0.00000 // (op 715) call()
v_accvgpr_write a14, 0.00000 // (op 715) call()
v_accvgpr_write a15, 0.00000 // (op 715) call()
s_mul_hi_i32 s0, s16, s10 // (op 2124) 
v_and_b32 v14, -4, v1 // (op 2124) 
s_add_i32 s1, s0, s16 // (op 2124) 

v_sub_u32 v15, v1, v14 // (op 2124) 

s_ashr_i32 s0, s1, 31 // (op 2124) 
v_lshlrev_b32 v14, 4, v15 // (op 2124) 

s_cmp_eq_i32 s10, 0 // (op 2124) 
s_mov_b32 s18, scc // (op 2124) 
v_lshrrev_b32 v15, 2, v1 // (op 2124) 
s_cmp_eq_i32 s18, 0 // (op 2124) 
s_cselect_b32 s19, 0, -1 // (op 2124) 

v_lshlrev_b32 v16, 1, v15 // (op 2124) 

s_and_b32 s18, s11, 31 // (op 2124) 
s_lshl_b32 s22, 1, s18 // (op 2124) 
s_add_i32 s23, s22, s19 // (op 2124) 


s_and_b32 s19, s0, s23 // (op 2124) 


s_add_i32 s0, s1, s19 // (op 2124) 


s_ashr_i32 s1, s0, s18 // (op 2124) 


s_lshl_b32 s0, s11, 24 // (op 2124) 
s_ashr_i32 s18, s0, 31 // (op 2124) 

s_xor_b32 s0, s1, s18 // (op 2124) 

s_sub_i32 s1, s0, s18 // (op 2124) 


s_mul_hi_i32 s0, s1, s12 // (op 2124) 
s_add_i32 s18, s0, s1 // (op 2124) 

s_ashr_i32 s0, s18, 31 // (op 2124) 
s_cmp_eq_i32 s12, 0 // (op 2124) 
s_mov_b32 s19, scc // (op 2124) 
s_cmp_eq_i32 s19, 0 // (op 2124) 
s_cselect_b32 s22, 0, -1 // (op 2124) 

s_and_b32 s19, s13, 31 // (op 2124) 
s_lshl_b32 s23, 1, s19 // (op 2124) 
s_add_i32 s24, s23, s22 // (op 2124) 


s_and_b32 s25, s0, s24 // (op 2124) 


s_add_i32 s0, s18, s25 // (op 2124) 


s_ashr_i32 s18, s0, s19 // (op 2124) 


s_lshl_b32 s0, s13, 24 // (op 2124) 
s_ashr_i32 s19, s0, 31 // (op 2124) 

s_xor_b32 s0, s18, s19 // (op 2124) 

s_sub_i32 s18, s0, s19 // (op 2124) 


s_mul_i32 s0, s1, s15 // (op 2124) 

s_sub_u32 s1, s16, s0 // (op 2124) 

s_mul_i32 s0, s18, s15 // (op 2124) 
s_add_u32 s0, s0, s1 // (op 2124) 


s_mul_hi_i32 s1, s16, s9 // (op 2124) 
s_add_i32 s18, s1, s16 // (op 2124) 

s_ashr_i32 s1, s18, 31 // (op 2124) 
s_cmp_eq_i32 s9, 0 // (op 2124) 
s_mov_b32 s19, scc // (op 2124) 
s_cmp_eq_i32 s19, 0 // (op 2124) 
s_cselect_b32 s22, 0, -1 // (op 2124) 

s_and_b32 s19, s46, 31 // (op 2124) 
s_lshl_b32 s23, 1, s19 // (op 2124) 
s_add_i32 s24, s23, s22 // (op 2124) 


s_and_b32 s25, s1, s24 // (op 2124) 


s_add_i32 s1, s18, s25 // (op 2124) 


s_ashr_i32 s18, s1, s19 // (op 2124) 


s_lshl_b32 s1, s46, 24 // (op 2124) 
s_ashr_i32 s19, s1, 31 // (op 2124) 

s_xor_b32 s1, s18, s19 // (op 2124) 

s_sub_i32 s18, s1, s19 // (op 2124) 


s_add_u32 s22, s6, 64 // (op 2124) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2124) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2124) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2124) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2124) 

s_mul_i32 s1, s15, s22 // (op 2124) 

s_mul_i32 s19, s8, s1 // (op 2124) 

s_mul_i32 s1, s18, s19 // (op 2124) 

s_sub_u32 s19, s16, s1 // (op 2124) 

s_mov_b32 s1, s19 // (op 2124) 
s_mul_hi_i32 s22, s1, s14 // (op 2124) 
s_add_i32 s23, s22, s1 // (op 2124) 


s_ashr_i32 s1, s23, 31 // (op 2124) 
s_cmp_eq_i32 s14, 0 // (op 2124) 
s_mov_b32 s22, scc // (op 2124) 
s_cmp_eq_i32 s22, 0 // (op 2124) 
s_cselect_b32 s24, 0, -1 // (op 2124) 

s_and_b32 s22, s48, 31 // (op 2124) 
s_lshl_b32 s25, 1, s22 // (op 2124) 
s_add_i32 s49, s25, s24 // (op 2124) 


s_and_b32 s24, s1, s49 // (op 2124) 


s_add_i32 s1, s23, s24 // (op 2124) 


s_ashr_i32 s23, s1, s22 // (op 2124) 


s_lshl_b32 s1, s48, 24 // (op 2124) 
s_ashr_i32 s22, s1, 31 // (op 2124) 

s_xor_b32 s1, s23, s22 // (op 2124) 

s_sub_i32 s23, s1, s22 // (op 2124) 


s_mul_i32 s1, s47, s15 // (op 2124) 
s_add_u32 s24, s20, 64 // (op 2124) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2124) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 2124) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 2124) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 2124) 

s_sub_i32 s22, s24, s1 // (op 2124) 

s_mul_i32 s24, s23, s22 // (op 2124) 


s_sub_u32 s25, s19, s24 // (op 2124) 


s_add_u32 s19, s25, s1 // (op 2124) 


s_cmp_eq_i32 s18, 0 // (op 2124) 
s_mov_b32 s1, scc // (op 2124) 

s_cmp_eq_i32 s1, 0 // (op 2124) 
s_cselect_b32 s18, s19, s0 // (op 2124) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 2124) 


v_mul_lo_u32 v14, v16, s26 // (op 2124) 

v_add_u32 v16, v15, v14 // (op 2124) 


s_mul_i32 s1, s26, s31 // (op 2126) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2126) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2126) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2126) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2126) most significant: sum
s_add_u32 s1, s1, s19 // (op 2126) most significant: sum


s_mov_b32 s52, s28 // (op 2126) reg expression
s_mov_b32 s53, s29 // (op 2126) reg expression
s_mov_b32 s54, s0 // (op 2126) reg expression
s_mov_b32 s55, 131072 // (op 2126) call()

s_mul_hi_i32 s0, s16, s10 // (op 2143) 
v_and_b32 v14, -4, v1 // (op 2143) 
s_add_i32 s1, s0, s16 // (op 2143) 

v_sub_u32 v15, v1, v14 // (op 2143) 

s_ashr_i32 s0, s1, 31 // (op 2143) 
v_lshlrev_b32 v14, 4, v15 // (op 2143) 

s_cmp_eq_i32 s10, 0 // (op 2143) 
s_mov_b32 s18, scc // (op 2143) 
v_lshrrev_b32 v15, 2, v1 // (op 2143) 
s_cmp_eq_i32 s18, 0 // (op 2143) 
s_cselect_b32 s19, 0, -1 // (op 2143) 

v_lshlrev_b32 v17, 1, v15 // (op 2143) 

s_and_b32 s18, s11, 31 // (op 2143) 
s_lshl_b32 s22, 1, s18 // (op 2143) 
s_add_i32 s23, s22, s19 // (op 2143) 


s_and_b32 s19, s0, s23 // (op 2143) 


s_add_i32 s0, s1, s19 // (op 2143) 


s_ashr_i32 s1, s0, s18 // (op 2143) 


s_lshl_b32 s0, s11, 24 // (op 2143) 
s_ashr_i32 s18, s0, 31 // (op 2143) 

s_xor_b32 s0, s1, s18 // (op 2143) 

s_sub_i32 s1, s0, s18 // (op 2143) 


s_mul_hi_i32 s0, s1, s12 // (op 2143) 
s_add_i32 s18, s0, s1 // (op 2143) 

s_ashr_i32 s0, s18, 31 // (op 2143) 
s_cmp_eq_i32 s12, 0 // (op 2143) 
s_mov_b32 s19, scc // (op 2143) 
s_cmp_eq_i32 s19, 0 // (op 2143) 
s_cselect_b32 s22, 0, -1 // (op 2143) 

s_and_b32 s19, s13, 31 // (op 2143) 
s_lshl_b32 s23, 1, s19 // (op 2143) 
s_add_i32 s24, s23, s22 // (op 2143) 


s_and_b32 s25, s0, s24 // (op 2143) 


s_add_i32 s0, s18, s25 // (op 2143) 


s_ashr_i32 s18, s0, s19 // (op 2143) 


s_lshl_b32 s0, s13, 24 // (op 2143) 
s_ashr_i32 s19, s0, 31 // (op 2143) 

s_xor_b32 s0, s18, s19 // (op 2143) 

s_sub_i32 s18, s0, s19 // (op 2143) 


s_mul_i32 s0, s1, s15 // (op 2143) 

s_sub_u32 s1, s16, s0 // (op 2143) 

s_mul_i32 s0, s18, s15 // (op 2143) 
s_add_u32 s0, s0, s1 // (op 2143) 


s_mul_hi_i32 s1, s16, s9 // (op 2143) 
s_add_i32 s18, s1, s16 // (op 2143) 

s_ashr_i32 s1, s18, 31 // (op 2143) 
s_cmp_eq_i32 s9, 0 // (op 2143) 
s_mov_b32 s19, scc // (op 2143) 
s_cmp_eq_i32 s19, 0 // (op 2143) 
s_cselect_b32 s22, 0, -1 // (op 2143) 

s_and_b32 s19, s46, 31 // (op 2143) 
s_lshl_b32 s23, 1, s19 // (op 2143) 
s_add_i32 s24, s23, s22 // (op 2143) 


s_and_b32 s25, s1, s24 // (op 2143) 


s_add_i32 s1, s18, s25 // (op 2143) 


s_ashr_i32 s18, s1, s19 // (op 2143) 


s_lshl_b32 s1, s46, 24 // (op 2143) 
s_ashr_i32 s19, s1, 31 // (op 2143) 

s_xor_b32 s1, s18, s19 // (op 2143) 

s_sub_i32 s18, s1, s19 // (op 2143) 


s_add_u32 s50, s6, 64 // (op 2143) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2143) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2143) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2143) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2143) 

s_mul_i32 s1, s15, s24 // (op 2143) 

s_mul_i32 s19, s8, s1 // (op 2143) 

s_mul_i32 s1, s18, s19 // (op 2143) 

s_sub_u32 s19, s16, s1 // (op 2143) 

s_mov_b32 s1, s19 // (op 2143) 
s_mul_hi_i32 s22, s1, s14 // (op 2143) 
s_add_i32 s23, s22, s1 // (op 2143) 


s_ashr_i32 s1, s23, 31 // (op 2143) 
s_cmp_eq_i32 s14, 0 // (op 2143) 
s_mov_b32 s22, scc // (op 2143) 
s_cmp_eq_i32 s22, 0 // (op 2143) 
s_cselect_b32 s24, 0, -1 // (op 2143) 

s_and_b32 s22, s48, 31 // (op 2143) 
s_lshl_b32 s25, 1, s22 // (op 2143) 
s_add_i32 s49, s25, s24 // (op 2143) 


s_and_b32 s24, s1, s49 // (op 2143) 


s_add_i32 s1, s23, s24 // (op 2143) 


s_ashr_i32 s23, s1, s22 // (op 2143) 


s_lshl_b32 s1, s48, 24 // (op 2143) 
s_ashr_i32 s22, s1, 31 // (op 2143) 

s_xor_b32 s1, s23, s22 // (op 2143) 

s_sub_i32 s23, s1, s22 // (op 2143) 


s_mul_i32 s1, s47, s15 // (op 2143) 
s_add_u32 s24, s20, 64 // (op 2143) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2143) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2143) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2143) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2143) 

s_sub_i32 s22, s24, s1 // (op 2143) 

s_mul_i32 s24, s23, s22 // (op 2143) 


s_sub_u32 s25, s19, s24 // (op 2143) 


s_add_u32 s19, s25, s1 // (op 2143) 


s_cmp_eq_i32 s18, 0 // (op 2143) 
s_mov_b32 s1, scc // (op 2143) 

s_cmp_eq_i32 s1, 0 // (op 2143) 
s_cselect_b32 s18, s19, s0 // (op 2143) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 2143) 


v_mul_lo_u32 v14, v17, s26 // (op 2143) 

v_add_u32 v17, v15, v14 // (op 2143) 


s_mul_i32 s1, s26, s31 // (op 2145) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2145) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2145) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2145) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2145) most significant: sum
s_add_u32 s1, s1, s19 // (op 2145) most significant: sum


s_mov_b32 s52, s28 // (op 2145) reg expression
s_mov_b32 s53, s29 // (op 2145) reg expression
s_mov_b32 s54, s0 // (op 2145) reg expression
s_mov_b32 s55, 131072 // (op 2145) call()

s_mul_hi_i32 s0, s16, s10 // (op 2188) 
v_lshrrev_b32 v14, 2, v1 // (op 2188) 
s_add_i32 s1, s0, s16 // (op 2188) 

v_and_b32 v15, -4, v1 // (op 2188) 
s_ashr_i32 s0, s1, 31 // (op 2188) 
v_sub_u32 v18, v1, v15 // (op 2188) 

s_cmp_eq_i32 s10, 0 // (op 2188) 
s_mov_b32 s18, scc // (op 2188) 
v_lshlrev_b32 v15, 5, v18 // (op 2188) 

s_cmp_eq_i32 s18, 0 // (op 2188) 
s_cselect_b32 s19, 0, -1 // (op 2188) 

v_mov_b32 v19, 0 // (op 2188) convert
v_mov_b32 v18, v15 // (op 2188) convert

s_and_b32 s18, s11, 31 // (op 2188) 
s_lshl_b32 s22, 1, s18 // (op 2188) 
s_add_i32 s23, s22, s19 // (op 2188) 


s_and_b32 s19, s0, s23 // (op 2188) 


s_add_i32 s0, s1, s19 // (op 2188) 


s_ashr_i32 s1, s0, s18 // (op 2188) 


s_lshl_b32 s0, s11, 24 // (op 2188) 
s_ashr_i32 s18, s0, 31 // (op 2188) 

s_xor_b32 s0, s1, s18 // (op 2188) 

s_sub_i32 s1, s0, s18 // (op 2188) 


s_mul_hi_i32 s0, s1, s12 // (op 2188) 
s_add_i32 s18, s0, s1 // (op 2188) 

s_ashr_i32 s0, s18, 31 // (op 2188) 
s_cmp_eq_i32 s12, 0 // (op 2188) 
s_mov_b32 s19, scc // (op 2188) 
s_cmp_eq_i32 s19, 0 // (op 2188) 
s_cselect_b32 s22, 0, -1 // (op 2188) 

s_and_b32 s19, s13, 31 // (op 2188) 
s_lshl_b32 s23, 1, s19 // (op 2188) 
s_add_i32 s24, s23, s22 // (op 2188) 


s_and_b32 s25, s0, s24 // (op 2188) 


s_add_i32 s0, s18, s25 // (op 2188) 


s_ashr_i32 s18, s0, s19 // (op 2188) 


s_lshl_b32 s0, s13, 24 // (op 2188) 
s_ashr_i32 s19, s0, 31 // (op 2188) 

s_xor_b32 s0, s18, s19 // (op 2188) 

s_sub_i32 s18, s0, s19 // (op 2188) 


s_add_u32 s50, s6, 64 // (op 2188) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2188) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2188) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2188) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2188) 

s_mul_i32 s0, s18, s24 // (op 2188) 

s_sub_i32 s18, s1, s0 // (op 2188) 


s_mul_hi_i32 s19, s16, s9 // (op 2188) 
s_add_i32 s0, s19, s16 // (op 2188) 

s_ashr_i32 s1, s0, 31 // (op 2188) 
s_cmp_eq_i32 s9, 0 // (op 2188) 
s_mov_b32 s19, scc // (op 2188) 
s_cmp_eq_i32 s19, 0 // (op 2188) 
s_cselect_b32 s22, 0, -1 // (op 2188) 

s_and_b32 s19, s46, 31 // (op 2188) 
s_lshl_b32 s23, 1, s19 // (op 2188) 
s_add_i32 s49, s23, s22 // (op 2188) 


s_and_b32 s22, s1, s49 // (op 2188) 


s_add_i32 s1, s0, s22 // (op 2188) 


s_ashr_i32 s0, s1, s19 // (op 2188) 


s_lshl_b32 s1, s46, 24 // (op 2188) 
s_ashr_i32 s19, s1, 31 // (op 2188) 

s_xor_b32 s1, s0, s19 // (op 2188) 

s_sub_i32 s0, s1, s19 // (op 2188) 


s_mul_i32 s1, s15, s24 // (op 2188) 

s_mul_i32 s19, s8, s1 // (op 2188) 

s_mul_i32 s1, s0, s19 // (op 2188) 

s_sub_u32 s19, s16, s1 // (op 2188) 

s_mul_hi_i32 s1, s19, s14 // (op 2188) 
s_add_i32 s22, s1, s19 // (op 2188) 


s_ashr_i32 s1, s22, 31 // (op 2188) 
s_cmp_eq_i32 s14, 0 // (op 2188) 
s_mov_b32 s19, scc // (op 2188) 
s_cmp_eq_i32 s19, 0 // (op 2188) 
s_cselect_b32 s23, 0, -1 // (op 2188) 

s_and_b32 s19, s48, 31 // (op 2188) 
s_lshl_b32 s24, 1, s19 // (op 2188) 
s_add_i32 s25, s24, s23 // (op 2188) 


s_and_b32 s23, s1, s25 // (op 2188) 


s_add_i32 s1, s22, s23 // (op 2188) 


s_ashr_i32 s22, s1, s19 // (op 2188) 


s_lshl_b32 s1, s48, 24 // (op 2188) 
s_ashr_i32 s19, s1, 31 // (op 2188) 

s_xor_b32 s1, s22, s19 // (op 2188) 

s_sub_i32 s22, s1, s19 // (op 2188) 


s_cmp_eq_i32 s0, 0 // (op 2188) 
s_mov_b32 s1, scc // (op 2188) 

s_cmp_eq_i32 s1, 0 // (op 2188) 
s_cselect_b32 s0, s22, s18 // (op 2188) 



v_lshl_add_u32 v15, s0, 6, v14 // (op 2188) 


v_mul_lo_u32 v21, v15, s33 // (op 2188) most significant: low of low * high
v_mul_hi_u32 v14, v15, s32 // (op 2188) most significant: high of low * low
v_mul_lo_u32 v20, v15, s32 // (op 2188) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 2188) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 2188) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 2188) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 2188) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 2188) 

v_mov_b32 v14, v18 // (op 2188) convert

s_mul_i32 s1, s32, s7 // (op 2190) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2190) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2190) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2190) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2190) most significant: sum
s_add_u32 s1, s1, s19 // (op 2190) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2190) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2190) 

s_mov_b32 s64, s34 // (op 2190) reg expression
s_mov_b32 s65, s35 // (op 2190) reg expression
s_mov_b32 s66, s0 // (op 2190) reg expression
s_mov_b32 s67, 131072 // (op 2190) call()

s_mul_hi_i32 s0, s16, s10 // (op 2207) 
v_lshrrev_b32 v15, 2, v1 // (op 2207) 
s_add_i32 s1, s0, s16 // (op 2207) 

v_and_b32 v18, -4, v1 // (op 2207) 
s_ashr_i32 s0, s1, 31 // (op 2207) 
v_sub_u32 v19, v1, v18 // (op 2207) 

s_cmp_eq_i32 s10, 0 // (op 2207) 
s_mov_b32 s18, scc // (op 2207) 
v_lshlrev_b32 v18, 5, v19 // (op 2207) 

s_cmp_eq_i32 s18, 0 // (op 2207) 
s_cselect_b32 s19, 0, -1 // (op 2207) 

v_mov_b32 v21, 0 // (op 2207) convert
v_mov_b32 v20, v18 // (op 2207) convert

s_and_b32 s18, s11, 31 // (op 2207) 
s_lshl_b32 s22, 1, s18 // (op 2207) 
s_add_i32 s23, s22, s19 // (op 2207) 


s_and_b32 s19, s0, s23 // (op 2207) 


s_add_i32 s0, s1, s19 // (op 2207) 


s_ashr_i32 s1, s0, s18 // (op 2207) 


s_lshl_b32 s0, s11, 24 // (op 2207) 
s_ashr_i32 s18, s0, 31 // (op 2207) 

s_xor_b32 s0, s1, s18 // (op 2207) 

s_sub_i32 s1, s0, s18 // (op 2207) 


s_mul_hi_i32 s0, s1, s12 // (op 2207) 
s_add_i32 s18, s0, s1 // (op 2207) 

s_ashr_i32 s0, s18, 31 // (op 2207) 
s_cmp_eq_i32 s12, 0 // (op 2207) 
s_mov_b32 s19, scc // (op 2207) 
s_cmp_eq_i32 s19, 0 // (op 2207) 
s_cselect_b32 s22, 0, -1 // (op 2207) 

s_and_b32 s19, s13, 31 // (op 2207) 
s_lshl_b32 s23, 1, s19 // (op 2207) 
s_add_i32 s24, s23, s22 // (op 2207) 


s_and_b32 s25, s0, s24 // (op 2207) 


s_add_i32 s0, s18, s25 // (op 2207) 


s_ashr_i32 s18, s0, s19 // (op 2207) 


s_lshl_b32 s0, s13, 24 // (op 2207) 
s_ashr_i32 s19, s0, 31 // (op 2207) 

s_xor_b32 s0, s18, s19 // (op 2207) 

s_sub_i32 s18, s0, s19 // (op 2207) 


s_add_u32 s50, s6, 64 // (op 2207) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2207) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2207) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2207) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2207) 

s_mul_i32 s0, s18, s24 // (op 2207) 

s_sub_i32 s18, s1, s0 // (op 2207) 


s_mul_hi_i32 s19, s16, s9 // (op 2207) 
s_add_i32 s0, s19, s16 // (op 2207) 

s_ashr_i32 s1, s0, 31 // (op 2207) 
s_cmp_eq_i32 s9, 0 // (op 2207) 
s_mov_b32 s19, scc // (op 2207) 
s_cmp_eq_i32 s19, 0 // (op 2207) 
s_cselect_b32 s22, 0, -1 // (op 2207) 

s_and_b32 s19, s46, 31 // (op 2207) 
s_lshl_b32 s23, 1, s19 // (op 2207) 
s_add_i32 s49, s23, s22 // (op 2207) 


s_and_b32 s22, s1, s49 // (op 2207) 


s_add_i32 s1, s0, s22 // (op 2207) 


s_ashr_i32 s0, s1, s19 // (op 2207) 


s_lshl_b32 s1, s46, 24 // (op 2207) 
s_ashr_i32 s19, s1, 31 // (op 2207) 

s_xor_b32 s1, s0, s19 // (op 2207) 

s_sub_i32 s0, s1, s19 // (op 2207) 


s_mul_i32 s1, s15, s24 // (op 2207) 

s_mul_i32 s19, s8, s1 // (op 2207) 

s_mul_i32 s1, s0, s19 // (op 2207) 

s_sub_u32 s19, s16, s1 // (op 2207) 

s_mul_hi_i32 s1, s19, s14 // (op 2207) 
s_add_i32 s22, s1, s19 // (op 2207) 


s_ashr_i32 s1, s22, 31 // (op 2207) 
s_cmp_eq_i32 s14, 0 // (op 2207) 
s_mov_b32 s19, scc // (op 2207) 
s_cmp_eq_i32 s19, 0 // (op 2207) 
s_cselect_b32 s23, 0, -1 // (op 2207) 

s_and_b32 s19, s48, 31 // (op 2207) 
s_lshl_b32 s24, 1, s19 // (op 2207) 
s_add_i32 s25, s24, s23 // (op 2207) 


s_and_b32 s23, s1, s25 // (op 2207) 


s_add_i32 s1, s22, s23 // (op 2207) 


s_ashr_i32 s22, s1, s19 // (op 2207) 


s_lshl_b32 s1, s48, 24 // (op 2207) 
s_ashr_i32 s19, s1, 31 // (op 2207) 

s_xor_b32 s1, s22, s19 // (op 2207) 

s_sub_i32 s22, s1, s19 // (op 2207) 


s_cmp_eq_i32 s0, 0 // (op 2207) 
s_mov_b32 s1, scc // (op 2207) 

s_cmp_eq_i32 s1, 0 // (op 2207) 
s_cselect_b32 s0, s22, s18 // (op 2207) 



v_lshl_add_u32 v18, s0, 6, v15 // (op 2207) 


v_mul_lo_u32 v23, v18, s33 // (op 2207) most significant: low of low * high
v_mul_hi_u32 v15, v18, s32 // (op 2207) most significant: high of low * low
v_mul_lo_u32 v22, v18, s32 // (op 2207) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 2207) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v22 // (op 2207) least significant half
v_addc_co_u32 v19, s[0:1], v21, v23, s[0:1] // (op 2207) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 2207) 


v_lshrrev_b64 v[20:21], 3, v[18:19] // (op 2207) 

v_mov_b32 v15, v20 // (op 2207) convert

s_mul_i32 s1, s32, s7 // (op 2209) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2209) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2209) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2209) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2209) most significant: sum
s_add_u32 s1, s1, s19 // (op 2209) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2209) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2209) 

s_mov_b32 s64, s34 // (op 2209) reg expression
s_mov_b32 s65, s35 // (op 2209) reg expression
s_mov_b32 s66, s0 // (op 2209) reg expression
s_mov_b32 s67, 131072 // (op 2209) call()

s_mul_hi_i32 s0, s16, s10 // (op 2252) 
v_and_b32 v18, -64, v1 // (op 2252) 
s_add_i32 s1, s0, s16 // (op 2252) 

v_sub_u32 v19, v1, v18 // (op 2252) 

s_ashr_i32 s0, s1, 31 // (op 2252) 
v_lshrrev_b32 v18, 6, v1 // (op 2252) 
s_cmp_eq_i32 s10, 0 // (op 2252) 
s_mov_b32 s18, scc // (op 2252) 
s_cmp_eq_i32 s18, 0 // (op 2252) 
s_cselect_b32 s19, 0, -1 // (op 2252) 

s_and_b32 s18, s11, 31 // (op 2252) 
s_lshl_b32 s22, 1, s18 // (op 2252) 
s_add_i32 s23, s22, s19 // (op 2252) 


s_and_b32 s19, s0, s23 // (op 2252) 


s_add_i32 s0, s1, s19 // (op 2252) 


s_ashr_i32 s1, s0, s18 // (op 2252) 


s_lshl_b32 s0, s11, 24 // (op 2252) 
s_ashr_i32 s18, s0, 31 // (op 2252) 

s_xor_b32 s0, s1, s18 // (op 2252) 

s_sub_i32 s1, s0, s18 // (op 2252) 


s_mul_hi_i32 s0, s1, s12 // (op 2252) 
s_add_i32 s18, s0, s1 // (op 2252) 

s_ashr_i32 s0, s18, 31 // (op 2252) 
s_cmp_eq_i32 s12, 0 // (op 2252) 
s_mov_b32 s19, scc // (op 2252) 
s_cmp_eq_i32 s19, 0 // (op 2252) 
s_cselect_b32 s22, 0, -1 // (op 2252) 

s_and_b32 s19, s13, 31 // (op 2252) 
s_lshl_b32 s23, 1, s19 // (op 2252) 
s_add_i32 s24, s23, s22 // (op 2252) 


s_and_b32 s25, s0, s24 // (op 2252) 


s_add_i32 s0, s18, s25 // (op 2252) 


s_ashr_i32 s18, s0, s19 // (op 2252) 


s_lshl_b32 s0, s13, 24 // (op 2252) 
s_ashr_i32 s19, s0, 31 // (op 2252) 

s_xor_b32 s0, s18, s19 // (op 2252) 

s_sub_i32 s18, s0, s19 // (op 2252) 


s_mul_i32 s0, s1, s15 // (op 2252) 

s_sub_u32 s1, s16, s0 // (op 2252) 

s_mul_i32 s0, s18, s15 // (op 2252) 
s_add_u32 s0, s0, s1 // (op 2252) 


s_mul_hi_i32 s1, s16, s9 // (op 2252) 
s_add_i32 s18, s1, s16 // (op 2252) 

s_ashr_i32 s1, s18, 31 // (op 2252) 
s_cmp_eq_i32 s9, 0 // (op 2252) 
s_mov_b32 s19, scc // (op 2252) 
s_cmp_eq_i32 s19, 0 // (op 2252) 
s_cselect_b32 s22, 0, -1 // (op 2252) 

s_and_b32 s19, s46, 31 // (op 2252) 
s_lshl_b32 s23, 1, s19 // (op 2252) 
s_add_i32 s24, s23, s22 // (op 2252) 


s_and_b32 s25, s1, s24 // (op 2252) 


s_add_i32 s1, s18, s25 // (op 2252) 


s_ashr_i32 s18, s1, s19 // (op 2252) 


s_lshl_b32 s1, s46, 24 // (op 2252) 
s_ashr_i32 s19, s1, 31 // (op 2252) 

s_xor_b32 s1, s18, s19 // (op 2252) 

s_sub_i32 s18, s1, s19 // (op 2252) 


s_add_u32 s50, s6, 64 // (op 2252) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2252) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2252) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2252) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2252) 

s_mul_i32 s1, s15, s24 // (op 2252) 

s_mul_i32 s19, s8, s1 // (op 2252) 

s_mul_i32 s1, s18, s19 // (op 2252) 

s_sub_u32 s19, s16, s1 // (op 2252) 

s_mov_b32 s1, s19 // (op 2252) 
s_mul_hi_i32 s22, s1, s14 // (op 2252) 
s_add_i32 s23, s22, s1 // (op 2252) 


s_ashr_i32 s1, s23, 31 // (op 2252) 
s_cmp_eq_i32 s14, 0 // (op 2252) 
s_mov_b32 s22, scc // (op 2252) 
s_cmp_eq_i32 s22, 0 // (op 2252) 
s_cselect_b32 s24, 0, -1 // (op 2252) 

s_and_b32 s22, s48, 31 // (op 2252) 
s_lshl_b32 s25, 1, s22 // (op 2252) 
s_add_i32 s49, s25, s24 // (op 2252) 


s_and_b32 s24, s1, s49 // (op 2252) 


s_add_i32 s1, s23, s24 // (op 2252) 


s_ashr_i32 s23, s1, s22 // (op 2252) 


s_lshl_b32 s1, s48, 24 // (op 2252) 
s_ashr_i32 s22, s1, 31 // (op 2252) 

s_xor_b32 s1, s23, s22 // (op 2252) 

s_sub_i32 s23, s1, s22 // (op 2252) 


s_mul_i32 s1, s47, s15 // (op 2252) 
s_add_u32 s24, s20, 64 // (op 2252) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2252) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2252) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2252) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2252) 

s_sub_i32 s22, s24, s1 // (op 2252) 

s_mul_i32 s24, s23, s22 // (op 2252) 


s_sub_u32 s25, s19, s24 // (op 2252) 


s_add_u32 s19, s25, s1 // (op 2252) 


s_cmp_eq_i32 s18, 0 // (op 2252) 
s_mov_b32 s1, scc // (op 2252) 

s_cmp_eq_i32 s1, 0 // (op 2252) 
s_cselect_b32 s18, s19, s0 // (op 2252) 



v_lshl_add_u32 v20, s18, 6, v19 // (op 2252) 


v_mul_lo_u32 v19, v18, s36 // (op 2252) 

v_add_u32 v18, v20, v19 // (op 2252) 


s_mul_i32 s1, s36, s41 // (op 2254) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2254) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2254) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2254) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2254) most significant: sum
s_add_u32 s1, s1, s19 // (op 2254) most significant: sum


s_mov_b32 s68, s38 // (op 2254) reg expression
s_mov_b32 s69, s39 // (op 2254) reg expression
s_mov_b32 s70, s0 // (op 2254) reg expression
s_mov_b32 s71, 131072 // (op 2254) call()

s_mul_hi_i32 s0, s16, s10 // (op 2271) 
v_and_b32 v19, -64, v1 // (op 2271) 
s_add_i32 s1, s0, s16 // (op 2271) 

v_sub_u32 v20, v1, v19 // (op 2271) 

s_ashr_i32 s0, s1, 31 // (op 2271) 
v_lshrrev_b32 v19, 6, v1 // (op 2271) 
s_cmp_eq_i32 s10, 0 // (op 2271) 
s_mov_b32 s18, scc // (op 2271) 
s_cmp_eq_i32 s18, 0 // (op 2271) 
s_cselect_b32 s19, 0, -1 // (op 2271) 

s_and_b32 s18, s11, 31 // (op 2271) 
s_lshl_b32 s22, 1, s18 // (op 2271) 
s_add_i32 s23, s22, s19 // (op 2271) 


s_and_b32 s19, s0, s23 // (op 2271) 


s_add_i32 s0, s1, s19 // (op 2271) 


s_ashr_i32 s1, s0, s18 // (op 2271) 


s_lshl_b32 s0, s11, 24 // (op 2271) 
s_ashr_i32 s18, s0, 31 // (op 2271) 

s_xor_b32 s0, s1, s18 // (op 2271) 

s_sub_i32 s1, s0, s18 // (op 2271) 


s_mul_hi_i32 s0, s1, s12 // (op 2271) 
s_add_i32 s18, s0, s1 // (op 2271) 

s_ashr_i32 s0, s18, 31 // (op 2271) 
s_cmp_eq_i32 s12, 0 // (op 2271) 
s_mov_b32 s19, scc // (op 2271) 
s_cmp_eq_i32 s19, 0 // (op 2271) 
s_cselect_b32 s22, 0, -1 // (op 2271) 

s_and_b32 s19, s13, 31 // (op 2271) 
s_lshl_b32 s23, 1, s19 // (op 2271) 
s_add_i32 s24, s23, s22 // (op 2271) 


s_and_b32 s25, s0, s24 // (op 2271) 


s_add_i32 s0, s18, s25 // (op 2271) 


s_ashr_i32 s18, s0, s19 // (op 2271) 


s_lshl_b32 s0, s13, 24 // (op 2271) 
s_ashr_i32 s19, s0, 31 // (op 2271) 

s_xor_b32 s0, s18, s19 // (op 2271) 

s_sub_i32 s18, s0, s19 // (op 2271) 


s_mul_i32 s0, s1, s15 // (op 2271) 

s_sub_u32 s1, s16, s0 // (op 2271) 

s_mul_i32 s0, s18, s15 // (op 2271) 
s_add_u32 s0, s0, s1 // (op 2271) 


s_mul_hi_i32 s1, s16, s9 // (op 2271) 
s_add_i32 s18, s1, s16 // (op 2271) 

s_ashr_i32 s1, s18, 31 // (op 2271) 
s_cmp_eq_i32 s9, 0 // (op 2271) 
s_mov_b32 s19, scc // (op 2271) 
s_cmp_eq_i32 s19, 0 // (op 2271) 
s_cselect_b32 s22, 0, -1 // (op 2271) 

s_and_b32 s19, s46, 31 // (op 2271) 
s_lshl_b32 s23, 1, s19 // (op 2271) 
s_add_i32 s24, s23, s22 // (op 2271) 


s_and_b32 s25, s1, s24 // (op 2271) 


s_add_i32 s1, s18, s25 // (op 2271) 


s_ashr_i32 s18, s1, s19 // (op 2271) 


s_lshl_b32 s1, s46, 24 // (op 2271) 
s_ashr_i32 s19, s1, 31 // (op 2271) 

s_xor_b32 s1, s18, s19 // (op 2271) 

s_sub_i32 s18, s1, s19 // (op 2271) 


s_add_u32 s50, s6, 64 // (op 2271) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2271) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2271) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2271) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2271) 

s_mul_i32 s1, s15, s24 // (op 2271) 

s_mul_i32 s19, s8, s1 // (op 2271) 

s_mul_i32 s1, s18, s19 // (op 2271) 

s_sub_u32 s19, s16, s1 // (op 2271) 

s_mov_b32 s1, s19 // (op 2271) 
s_mul_hi_i32 s22, s1, s14 // (op 2271) 
s_add_i32 s23, s22, s1 // (op 2271) 


s_ashr_i32 s1, s23, 31 // (op 2271) 
s_cmp_eq_i32 s14, 0 // (op 2271) 
s_mov_b32 s22, scc // (op 2271) 
s_cmp_eq_i32 s22, 0 // (op 2271) 
s_cselect_b32 s24, 0, -1 // (op 2271) 

s_and_b32 s22, s48, 31 // (op 2271) 
s_lshl_b32 s25, 1, s22 // (op 2271) 
s_add_i32 s49, s25, s24 // (op 2271) 


s_and_b32 s24, s1, s49 // (op 2271) 


s_add_i32 s1, s23, s24 // (op 2271) 


s_ashr_i32 s23, s1, s22 // (op 2271) 


s_lshl_b32 s1, s48, 24 // (op 2271) 
s_ashr_i32 s22, s1, 31 // (op 2271) 

s_xor_b32 s1, s23, s22 // (op 2271) 

s_sub_i32 s23, s1, s22 // (op 2271) 


s_mul_i32 s1, s47, s15 // (op 2271) 
s_add_u32 s24, s20, 64 // (op 2271) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2271) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2271) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2271) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2271) 

s_sub_i32 s22, s24, s1 // (op 2271) 

s_mul_i32 s24, s23, s22 // (op 2271) 


s_sub_u32 s25, s19, s24 // (op 2271) 


s_add_u32 s19, s25, s1 // (op 2271) 


s_cmp_eq_i32 s18, 0 // (op 2271) 
s_mov_b32 s1, scc // (op 2271) 

s_cmp_eq_i32 s1, 0 // (op 2271) 
s_cselect_b32 s18, s19, s0 // (op 2271) 



v_lshl_add_u32 v21, s18, 6, v20 // (op 2271) 


v_mul_lo_u32 v20, v19, s36 // (op 2271) 

v_add_u32 v19, v21, v20 // (op 2271) 


s_mul_i32 s1, s36, s41 // (op 2273) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2273) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2273) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2273) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2273) most significant: sum
s_add_u32 s1, s1, s19 // (op 2273) most significant: sum


s_mov_b32 s68, s38 // (op 2273) reg expression
s_mov_b32 s69, s39 // (op 2273) reg expression
s_mov_b32 s70, s0 // (op 2273) reg expression
s_mov_b32 s71, 131072 // (op 2273) call()

s_mul_hi_i32 s0, s16, s10 // (op 2316) 
v_lshrrev_b32 v20, 2, v1 // (op 2316) 
s_add_i32 s1, s0, s16 // (op 2316) 

v_and_b32 v21, -4, v1 // (op 2316) 
s_ashr_i32 s0, s1, 31 // (op 2316) 
v_sub_u32 v22, v1, v21 // (op 2316) 

s_cmp_eq_i32 s10, 0 // (op 2316) 
s_mov_b32 s18, scc // (op 2316) 
s_cmp_eq_i32 s18, 0 // (op 2316) 
s_cselect_b32 s19, 0, -1 // (op 2316) 

s_and_b32 s18, s11, 31 // (op 2316) 
s_lshl_b32 s22, 1, s18 // (op 2316) 
s_add_i32 s23, s22, s19 // (op 2316) 


s_and_b32 s19, s0, s23 // (op 2316) 


s_add_i32 s0, s1, s19 // (op 2316) 


s_ashr_i32 s1, s0, s18 // (op 2316) 


s_lshl_b32 s0, s11, 24 // (op 2316) 
s_ashr_i32 s18, s0, 31 // (op 2316) 

s_xor_b32 s0, s1, s18 // (op 2316) 

s_sub_i32 s1, s0, s18 // (op 2316) 


s_mul_hi_i32 s0, s1, s12 // (op 2316) 
s_add_i32 s18, s0, s1 // (op 2316) 

s_ashr_i32 s0, s18, 31 // (op 2316) 
s_cmp_eq_i32 s12, 0 // (op 2316) 
s_mov_b32 s19, scc // (op 2316) 
s_cmp_eq_i32 s19, 0 // (op 2316) 
s_cselect_b32 s22, 0, -1 // (op 2316) 

s_and_b32 s19, s13, 31 // (op 2316) 
s_lshl_b32 s23, 1, s19 // (op 2316) 
s_add_i32 s24, s23, s22 // (op 2316) 


s_and_b32 s25, s0, s24 // (op 2316) 


s_add_i32 s0, s18, s25 // (op 2316) 


s_ashr_i32 s18, s0, s19 // (op 2316) 


s_lshl_b32 s0, s13, 24 // (op 2316) 
s_ashr_i32 s19, s0, 31 // (op 2316) 

s_xor_b32 s0, s18, s19 // (op 2316) 

s_sub_i32 s18, s0, s19 // (op 2316) 


s_add_u32 s50, s6, 64 // (op 2316) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2316) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2316) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2316) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2316) 

s_mul_i32 s0, s18, s24 // (op 2316) 

s_sub_i32 s18, s1, s0 // (op 2316) 


s_mul_hi_i32 s19, s16, s9 // (op 2316) 
s_add_i32 s0, s19, s16 // (op 2316) 

s_ashr_i32 s1, s0, 31 // (op 2316) 
s_cmp_eq_i32 s9, 0 // (op 2316) 
s_mov_b32 s19, scc // (op 2316) 
s_cmp_eq_i32 s19, 0 // (op 2316) 
s_cselect_b32 s22, 0, -1 // (op 2316) 

s_and_b32 s19, s46, 31 // (op 2316) 
s_lshl_b32 s23, 1, s19 // (op 2316) 
s_add_i32 s49, s23, s22 // (op 2316) 


s_and_b32 s22, s1, s49 // (op 2316) 


s_add_i32 s1, s0, s22 // (op 2316) 


s_ashr_i32 s0, s1, s19 // (op 2316) 


s_lshl_b32 s1, s46, 24 // (op 2316) 
s_ashr_i32 s19, s1, 31 // (op 2316) 

s_xor_b32 s1, s0, s19 // (op 2316) 

s_sub_i32 s0, s1, s19 // (op 2316) 


s_mul_i32 s1, s15, s24 // (op 2316) 

s_mul_i32 s19, s8, s1 // (op 2316) 

s_mul_i32 s1, s0, s19 // (op 2316) 

s_sub_u32 s19, s16, s1 // (op 2316) 

s_mul_hi_i32 s1, s19, s14 // (op 2316) 
s_add_i32 s22, s1, s19 // (op 2316) 


s_ashr_i32 s1, s22, 31 // (op 2316) 
s_cmp_eq_i32 s14, 0 // (op 2316) 
s_mov_b32 s19, scc // (op 2316) 
s_cmp_eq_i32 s19, 0 // (op 2316) 
s_cselect_b32 s23, 0, -1 // (op 2316) 

s_and_b32 s19, s48, 31 // (op 2316) 
s_lshl_b32 s24, 1, s19 // (op 2316) 
s_add_i32 s25, s24, s23 // (op 2316) 


s_and_b32 s23, s1, s25 // (op 2316) 


s_add_i32 s1, s22, s23 // (op 2316) 


s_ashr_i32 s22, s1, s19 // (op 2316) 


s_lshl_b32 s1, s48, 24 // (op 2316) 
s_ashr_i32 s19, s1, 31 // (op 2316) 

s_xor_b32 s1, s22, s19 // (op 2316) 

s_sub_i32 s22, s1, s19 // (op 2316) 


s_cmp_eq_i32 s0, 0 // (op 2316) 
s_mov_b32 s1, scc // (op 2316) 

s_cmp_eq_i32 s1, 0 // (op 2316) 
s_cselect_b32 s0, s22, s18 // (op 2316) 



v_lshl_add_u32 v21, s0, 6, v20 // (op 2316) 


v_mul_lo_u32 v20, v21, s42 // (op 2316) 

v_add_u32 v21, v22, v20 // (op 2316) 


s_mul_i32 s1, s42, s7 // (op 2318) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2318) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2318) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2318) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2318) most significant: sum
s_add_u32 s1, s1, s19 // (op 2318) most significant: sum


s_mov_b32 s72, s44 // (op 2318) reg expression
s_mov_b32 s73, s45 // (op 2318) reg expression
s_mov_b32 s74, s0 // (op 2318) reg expression
s_mov_b32 s75, 131072 // (op 2318) call()

s_mul_hi_i32 s0, s16, s10 // (op 2335) 
v_lshrrev_b32 v20, 2, v1 // (op 2335) 
s_add_i32 s1, s0, s16 // (op 2335) 

v_and_b32 v22, -4, v1 // (op 2335) 
s_ashr_i32 s0, s1, 31 // (op 2335) 
v_sub_u32 v23, v1, v22 // (op 2335) 

s_cmp_eq_i32 s10, 0 // (op 2335) 
s_mov_b32 s18, scc // (op 2335) 
s_cmp_eq_i32 s18, 0 // (op 2335) 
s_cselect_b32 s19, 0, -1 // (op 2335) 

s_and_b32 s18, s11, 31 // (op 2335) 
s_lshl_b32 s22, 1, s18 // (op 2335) 
s_add_i32 s23, s22, s19 // (op 2335) 


s_and_b32 s19, s0, s23 // (op 2335) 


s_add_i32 s0, s1, s19 // (op 2335) 


s_ashr_i32 s1, s0, s18 // (op 2335) 


s_lshl_b32 s0, s11, 24 // (op 2335) 
s_ashr_i32 s18, s0, 31 // (op 2335) 

s_xor_b32 s0, s1, s18 // (op 2335) 

s_sub_i32 s1, s0, s18 // (op 2335) 


s_mul_hi_i32 s0, s1, s12 // (op 2335) 
s_add_i32 s18, s0, s1 // (op 2335) 

s_ashr_i32 s0, s18, 31 // (op 2335) 
s_cmp_eq_i32 s12, 0 // (op 2335) 
s_mov_b32 s19, scc // (op 2335) 
s_cmp_eq_i32 s19, 0 // (op 2335) 
s_cselect_b32 s22, 0, -1 // (op 2335) 

s_and_b32 s19, s13, 31 // (op 2335) 
s_lshl_b32 s23, 1, s19 // (op 2335) 
s_add_i32 s24, s23, s22 // (op 2335) 


s_and_b32 s25, s0, s24 // (op 2335) 


s_add_i32 s0, s18, s25 // (op 2335) 


s_ashr_i32 s18, s0, s19 // (op 2335) 


s_lshl_b32 s0, s13, 24 // (op 2335) 
s_ashr_i32 s19, s0, 31 // (op 2335) 

s_xor_b32 s0, s18, s19 // (op 2335) 

s_sub_i32 s18, s0, s19 // (op 2335) 


s_add_u32 s50, s6, 64 // (op 2335) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2335) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2335) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2335) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2335) 

s_mul_i32 s0, s18, s24 // (op 2335) 

s_sub_i32 s18, s1, s0 // (op 2335) 


s_mul_hi_i32 s19, s16, s9 // (op 2335) 
s_add_i32 s0, s19, s16 // (op 2335) 

s_ashr_i32 s1, s0, 31 // (op 2335) 
s_cmp_eq_i32 s9, 0 // (op 2335) 
s_mov_b32 s19, scc // (op 2335) 
s_cmp_eq_i32 s19, 0 // (op 2335) 
s_cselect_b32 s22, 0, -1 // (op 2335) 

s_and_b32 s19, s46, 31 // (op 2335) 
s_lshl_b32 s23, 1, s19 // (op 2335) 
s_add_i32 s49, s23, s22 // (op 2335) 


s_and_b32 s22, s1, s49 // (op 2335) 


s_add_i32 s1, s0, s22 // (op 2335) 


s_ashr_i32 s0, s1, s19 // (op 2335) 


s_lshl_b32 s1, s46, 24 // (op 2335) 
s_ashr_i32 s19, s1, 31 // (op 2335) 

s_xor_b32 s1, s0, s19 // (op 2335) 

s_sub_i32 s0, s1, s19 // (op 2335) 


s_mul_i32 s1, s15, s24 // (op 2335) 

s_mul_i32 s19, s8, s1 // (op 2335) 

s_mul_i32 s1, s0, s19 // (op 2335) 

s_sub_u32 s19, s16, s1 // (op 2335) 

s_mul_hi_i32 s1, s19, s14 // (op 2335) 
s_add_i32 s22, s1, s19 // (op 2335) 


s_ashr_i32 s1, s22, 31 // (op 2335) 
s_cmp_eq_i32 s14, 0 // (op 2335) 
s_mov_b32 s19, scc // (op 2335) 
s_cmp_eq_i32 s19, 0 // (op 2335) 
s_cselect_b32 s23, 0, -1 // (op 2335) 

s_and_b32 s19, s48, 31 // (op 2335) 
s_lshl_b32 s24, 1, s19 // (op 2335) 
s_add_i32 s25, s24, s23 // (op 2335) 


s_and_b32 s23, s1, s25 // (op 2335) 


s_add_i32 s1, s22, s23 // (op 2335) 


s_ashr_i32 s22, s1, s19 // (op 2335) 


s_lshl_b32 s1, s48, 24 // (op 2335) 
s_ashr_i32 s19, s1, 31 // (op 2335) 

s_xor_b32 s1, s22, s19 // (op 2335) 

s_sub_i32 s22, s1, s19 // (op 2335) 


s_cmp_eq_i32 s0, 0 // (op 2335) 
s_mov_b32 s1, scc // (op 2335) 

s_cmp_eq_i32 s1, 0 // (op 2335) 
s_cselect_b32 s0, s22, s18 // (op 2335) 



v_lshl_add_u32 v22, s0, 6, v20 // (op 2335) 


v_mul_lo_u32 v20, v22, s42 // (op 2335) 

v_add_u32 v22, v23, v20 // (op 2335) 


s_mul_i32 s1, s42, s7 // (op 2337) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2337) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2337) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2337) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2337) most significant: sum
s_add_u32 s1, s1, s19 // (op 2337) most significant: sum


s_mov_b32 s72, s44 // (op 2337) reg expression
s_mov_b32 s73, s45 // (op 2337) reg expression
s_mov_b32 s74, s0 // (op 2337) reg expression
s_mov_b32 s75, 131072 // (op 2337) call()




buffer_load_dwordx4 v[24:27], v14, s[64:67], 0 offen // (op 1956) Load value



buffer_load_ubyte v20, v21, s[72:75], 0 offen offset: 0 // (op 1959) Load value

v_mov_b64 v[28:29], s[26:27] // (op 1962) convert

v_mov_b32 v23, v16 // (op 1962) reg expression
buffer_load_dwordx4 v[30:33], v23, s[52:55], 0 offen // (op 1962) Load value
v_add_u32 v23, v23, v28 // (op 1962) 
buffer_load_dwordx4 v[34:37], v23, s[52:55], 0 offen // (op 1962) Load value




v_mov_b32 v23, v18 // (op 1965) reg expression

buffer_load_ubyte v28, v23, s[68:71], 0 offen offset: 0 // (op 1965) Load value




buffer_load_dwordx4 v[38:41], v14, s[64:67], 0 offen offset:64 // (op 1968) Load value




buffer_load_ubyte v14, v21, s[72:75], 0 offen offset:4 // (op 1971) Load value


v_mov_b64 v[42:43], s[26:27] // (op 1974) convert

s_lshl_b32 s0, s26, 7 // (op 1974) 
v_add_u32 v21, v16, s0 // (op 1974) 

buffer_load_dwordx4 v[44:47], v21, s[52:55], 0 offen // (op 1974) Load value
v_add_u32 v21, v21, v42 // (op 1974) 
buffer_load_dwordx4 v[48:51], v21, s[52:55], 0 offen // (op 1974) Load value





s_lshl_b32 s0, s36, 2 // (op 1977) 
v_add_u32 v16, v18, s0 // (op 1977) 


buffer_load_ubyte v21, v16, s[68:71], 0 offen offset: 0 // (op 1977) Load value


v_and_b32 v16, -4, v1 // (op 2465) 
v_sub_u32 v18, v1, v16 // (op 2465) 

v_lshlrev_b32 v16, 5, v18 // (op 2465) 

v_lshrrev_b32 v18, 2, v1 // (op 2465) 
v_lshl_add_u32 v23, v18, 7, v16 // (op 2465) 


v_lshlrev_b32 v16, 2, v23 // (op 2465) 

v_lshrrev_b32 v18, 3, v16 // (op 2465) 




s_waitcnt vmcnt(9)
ds_write_b128 v18, v[24:27] // (op 1979) Store local data 

v_and_b32 v16, -4, v1 // (op 2581) 
v_sub_u32 v18, v1, v16 // (op 2581) 

v_and_b32 v16, 4294967292, v1 // (op 2581) 
v_add_u32 v23, v16, v18 // (op 2581) 





s_waitcnt vmcnt(8)
ds_write_b8 v23, v20 offset:4096 // (op 1980) Store local data 

v_and_b32 v16, -4, v1 // (op 2407) 
v_sub_u32 v18, v1, v16 // (op 2407) 

v_lshlrev_b32 v16, 4, v18 // (op 2407) 

v_lshrrev_b32 v18, 2, v1 // (op 2407) 
v_lshl_add_u32 v23, v18, 7, v16 // (op 2407) 





s_waitcnt vmcnt(7)
ds_write_b128 v23, v[30:33] offset:4352 // (op 1981) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v23, v[34:37] offset:4416 // (op 1981) Store local data 

v_and_b32 v16, -64, v1 // (op 2523) 
v_sub_u32 v18, v1, v16 // (op 2523) 

v_and_b32 v16, 4294967232, v1 // (op 2523) 
v_add_u32 v23, v16, v18 // (op 2523) 





s_waitcnt vmcnt(5)
ds_write_b8 v23, v28 offset:12544 // (op 1982) Store local data 

v_and_b32 v16, -4, v1 // (op 2378) 
v_sub_u32 v18, v1, v16 // (op 2378) 

v_lshlrev_b32 v16, 4, v18 // (op 2378) 

v_lshrrev_b32 v18, 2, v1 // (op 2378) 
v_lshl_add_u32 v23, v18, 7, v16 // (op 2378) 


v_and_b32 v16, -4, v1 // (op 2436) 
v_sub_u32 v18, v1, v16 // (op 2436) 

v_lshlrev_b32 v16, 5, v18 // (op 2436) 

v_lshrrev_b32 v18, 2, v1 // (op 2436) 
v_lshl_add_u32 v29, v18, 7, v16 // (op 2436) 


v_lshlrev_b32 v16, 2, v29 // (op 2436) 

v_lshrrev_b32 v18, 3, v16 // (op 2436) 

v_and_b32 v16, -64, v1 // (op 2494) 
v_sub_u32 v29, v1, v16 // (op 2494) 

v_and_b32 v16, 4294967232, v1 // (op 2494) 
v_add_u32 v42, v16, v29 // (op 2494) 


v_and_b32 v16, -4, v1 // (op 2552) 
v_sub_u32 v29, v1, v16 // (op 2552) 

v_and_b32 v16, 4294967292, v1 // (op 2552) 
v_add_u32 v43, v16, v29 // (op 2552) 


v_and_b32 v16, -4, v1 // (op 2633) 
v_sub_u32 v29, v1, v16 // (op 2633) 

v_lshlrev_b32 v16, 4, v29 // (op 2633) 

v_lshrrev_b32 v29, 2, v1 // (op 2633) 
v_lshl_add_u32 v52, v29, 7, v16 // (op 2633) 


v_and_b32 v16, -4, v1 // (op 2662) 
v_sub_u32 v29, v1, v16 // (op 2662) 

v_lshlrev_b32 v16, 5, v29 // (op 2662) 

v_lshrrev_b32 v29, 2, v1 // (op 2662) 
v_lshl_add_u32 v53, v29, 7, v16 // (op 2662) 


v_lshlrev_b32 v16, 2, v53 // (op 2662) 

v_lshrrev_b32 v29, 3, v16 // (op 2662) 

v_and_b32 v16, -64, v1 // (op 2691) 
v_sub_u32 v53, v1, v16 // (op 2691) 

v_and_b32 v16, 4294967232, v1 // (op 2691) 
v_add_u32 v54, v16, v53 // (op 2691) 


v_and_b32 v16, -4, v1 // (op 2720) 
v_sub_u32 v53, v1, v16 // (op 2720) 

v_and_b32 v16, 4294967292, v1 // (op 2720) 
v_add_u32 v55, v16, v53 // (op 2720) 


s_mov_b32 s0, 0 // (op 593) call()
s_ashr_i64 s[18:19], s[30:31], 7 // (op 592) 
s_mov_b32 s1, s18 // (op 592) convert

s_and_b32 s18, s1, 4294967294 // (op 592) 

s_cmp_lt_u32 s0, s18 // (op 592) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_592 // (op 592) Condition: Top (jump to RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_592 if false)
RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_592:


s_barrier  // (op 2788) 
s_add_u32 s0, s0, 2 // (op 594) 



v_mov_b32 v16, v15 // (op 10) 

buffer_load_dwordx4 v[24:27], v16, s[64:67], 0 offen offset:128 // (op 10) Load value
ds_read_b64_tr_b8 v[56:57], v0 offset:4352 // (op 72) Transpose load local data 


ds_read_b64_tr_b8 v[58:59], v0 offset:4864 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:8448 // (op 72) Transpose load local data 
v_mov_b32 v53, v22 // (op 25) 

buffer_load_ubyte v20, v53, s[72:75], 0 offen offset:8 // (op 25) Load value
ds_read_b64_tr_b8 v[62:63], v0 offset:8960 // (op 72) Transpose load local data 

s_barrier  // (op 2801) 
v_mov_b64 v[64:65], s[26:27] // (op 4) convert




v_mov_b32 v66, v17 // (op 4) 
s_lshl_b32 s1, s26, 8 // (op 4) 
v_add_u32 v67, v66, s1 // (op 4) 

ds_read_b128 v[68:71], v4 // (op 77) Load local data 
buffer_load_dwordx4 v[30:33], v67, s[52:55], 0 offen // (op 4) Load value
s_barrier  // (op 2815) 
v_add_u32 v67, v67, v64 // (op 4) 


ds_read_u8 v72, v3 offset:12544 // (op 82) Load local data 
buffer_load_dwordx4 v[34:37], v67, s[52:55], 0 offen // (op 4) Load value




s_barrier  // (op 2829) 
v_mov_b32 v67, v19 // (op 17) 



ds_read_u8 v64, v5 offset:4096 // (op 87) Load local data 
s_lshl_b32 s1, s36, 3 // (op 17) 



v_add_u32 v65, v67, s1 // (op 17) 


buffer_load_ubyte v28, v65, s[68:71], 0 offen offset: 0 // (op 17) Load value

ds_read_b128 v[74:77], v4 offset:2048 // (op 375) Load local data 



s_barrier  // (op 2852) 
ds_read_u8 v65, v5 offset:4224 // (op 379) Load local data 






s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[68:71], a[0:3], v72, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 90) 
ds_read_b64_tr_b8 v[78:79], v0 offset:4384 // (op 626) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v0 offset:4896 // (op 626) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v0 offset:8480 // (op 626) Transpose load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[74:77], a[4:7], v72, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 380) 
ds_read_b64_tr_b8 v[84:85], v0 offset:8992 // (op 626) Transpose load local data 




ds_read_u8 v56, v3 offset:12576 // (op 629) Load local data 
s_waitcnt vmcnt(9)
ds_write_b128 v29, v[38:41] offset:12800 // (op 257) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:85], v[68:71], a[8:11], v56, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 677) 


s_barrier  // (op 2864) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:85], v[74:77], a[12:15], v56, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 631) 
s_waitcnt vmcnt(8)
ds_write_b8 v55, v14 offset:16896 // (op 259) Store local data 




s_barrier  // (op 2846) 



s_waitcnt vmcnt(7)
ds_write_b128 v52, v[44:47] offset:17152 // (op 247) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v52, v[48:51] offset:17216 // (op 247) Store local data 
s_barrier  // (op 2858) 



s_waitcnt vmcnt(5)
ds_write_b8 v54, v21 offset:25344 // (op 258) Store local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 2843) 
buffer_load_dwordx4 v[38:41], v16, s[64:67], 0 offen offset:192 // (op 260) Load value




ds_read_b64_tr_b8 v[56:57], v6 offset:17152 // (op 249) Transpose load local data 



ds_read_b64_tr_b8 v[58:59], v6 offset:17664 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v6 offset:21248 // (op 249) Transpose load local data 
v_add_u32 v15, 128, v15 // (op 2199) 
ds_read_b64_tr_b8 v[62:63], v6 offset:21760 // (op 249) Transpose load local data 
buffer_load_ubyte v14, v53, s[72:75], 0 offen offset:12 // (op 262) Load value

s_barrier  // (op 2849) 




v_add_u32 v22, 8, v22 // (op 2327) 
v_mov_b64 v[64:65], s[26:27] // (op 246) convert

s_lshl_b32 s1, s26, 7 // (op 246) 
s_mul_i32 s18, 3, s1 // (op 246) 

v_add_u32 v16, v66, s18 // (op 246) 

buffer_load_dwordx4 v[44:47], v16, s[52:55], 0 offen // (op 246) Load value
v_add_u32 v16, v16, v64 // (op 246) 
ds_read_b128 v[68:71], v8 offset:12800 // (op 251) Load local data 
buffer_load_dwordx4 v[48:51], v16, s[52:55], 0 offen // (op 246) Load value



s_waitcnt lgkmcnt(5)
s_barrier  // (op 2855) 




ds_read_u8 v16, v7 offset:25344 // (op 253) Load local data 
s_lshl_b32 s1, s36, 2 // (op 261) 
s_barrier  // (op 2861) 
s_mul_i32 s18, 3, s1 // (op 261) 




ds_read_u8 v53, v9 offset:16896 // (op 255) Load local data 
s_lshl_b32 s1, s26, 8 // (op 2135) 



v_add_u32 v17, v17, s1 // (op 2135) 

v_add_u32 v64, v67, s18 // (op 261) 


ds_read_b128 v[72:75], v8 offset:14848 // (op 385) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[68:71], a[0:3], v16, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 256) 
buffer_load_ubyte v21, v64, s[68:71], 0 offen offset: 0 // (op 261) Load value





s_barrier  // (op 2925) 
s_barrier  // (op 2804) 
ds_read_u8 v64, v9 offset:17024 // (op 389) Load local data 



s_lshl_b32 s1, s36, 3 // (op 2263) 



s_barrier  // (op 2929) 
v_add_u32 v19, v19, s1 // (op 2263) 

ds_read_b64_tr_b8 v[76:77], v6 offset:17184 // (op 634) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v6 offset:17696 // (op 634) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[72:75], a[4:7], v16, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 381) 
ds_read_b64_tr_b8 v[80:81], v6 offset:21280 // (op 634) Transpose load local data 


ds_read_b64_tr_b8 v[82:83], v6 offset:21792 // (op 634) Transpose load local data 
s_waitcnt vmcnt(9)
ds_write_b128 v18, v[24:27] // (op 13) Store local data 


s_barrier  // (op 2949) 
s_barrier  // (op 2832) 
ds_read_u8 v16, v7 offset:25376 // (op 637) Load local data 



s_waitcnt vmcnt(8)
ds_write_b8 v43, v20 offset:4096 // (op 30) Store local data 
s_barrier  // (op 2953) 
s_barrier  // (op 2791) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:83], v[68:71], a[8:11], v16, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 678) 





s_waitcnt vmcnt(7)
ds_write_b128 v23, v[30:33] offset:4352 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:83], v[72:75], a[12:15], v16, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 632) 
s_waitcnt vmcnt(6)
ds_write_b128 v23, v[34:37] offset:4416 // (op 6) Store local data 




s_barrier  // (op 2818) 



s_waitcnt vmcnt(5)
ds_write_b8 v42, v28 offset:12544 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 592) 
s_mov_b32 s1, s18 // (op 592) convert

s_and_b32 s18, s1, 4294967294 // (op 592) 

s_cmp_lt_u32 s0, s18 // (op 592) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_592 // (op 592) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoop_592 if true)
RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoop_592:

















s_mul_hi_i32 s0, s16, s10 // (op 2095) 
v_and_b32 v15, -4, v1 // (op 2095) 
s_add_i32 s1, s0, s16 // (op 2095) 

v_sub_u32 v16, v1, v15 // (op 2095) 

s_ashr_i32 s0, s1, 31 // (op 2095) 
v_lshlrev_b32 v15, 4, v16 // (op 2095) 

s_cmp_eq_i32 s10, 0 // (op 2095) 
s_mov_b32 s18, scc // (op 2095) 
v_lshrrev_b32 v16, 2, v1 // (op 2095) 
s_cmp_eq_i32 s18, 0 // (op 2095) 
s_cselect_b32 s19, 0, -1 // (op 2095) 

v_lshlrev_b32 v17, 1, v16 // (op 2095) 

s_and_b32 s18, s11, 31 // (op 2095) 
s_lshl_b32 s22, 1, s18 // (op 2095) 
s_add_i32 s23, s22, s19 // (op 2095) 


s_and_b32 s19, s0, s23 // (op 2095) 


s_add_i32 s0, s1, s19 // (op 2095) 


s_ashr_i32 s1, s0, s18 // (op 2095) 


s_lshl_b32 s0, s11, 24 // (op 2095) 
s_ashr_i32 s18, s0, 31 // (op 2095) 

s_xor_b32 s0, s1, s18 // (op 2095) 

s_sub_i32 s1, s0, s18 // (op 2095) 


s_mul_hi_i32 s0, s1, s12 // (op 2095) 
s_add_i32 s18, s0, s1 // (op 2095) 

s_ashr_i32 s0, s18, 31 // (op 2095) 
s_cmp_eq_i32 s12, 0 // (op 2095) 
s_mov_b32 s19, scc // (op 2095) 
s_cmp_eq_i32 s19, 0 // (op 2095) 
s_cselect_b32 s22, 0, -1 // (op 2095) 

s_and_b32 s19, s13, 31 // (op 2095) 
s_lshl_b32 s23, 1, s19 // (op 2095) 
s_add_i32 s24, s23, s22 // (op 2095) 


s_and_b32 s25, s0, s24 // (op 2095) 


s_add_i32 s0, s18, s25 // (op 2095) 


s_ashr_i32 s18, s0, s19 // (op 2095) 


s_lshl_b32 s0, s13, 24 // (op 2095) 
s_ashr_i32 s19, s0, 31 // (op 2095) 

s_xor_b32 s0, s18, s19 // (op 2095) 

s_sub_i32 s18, s0, s19 // (op 2095) 


s_mul_i32 s0, s1, s15 // (op 2095) 

s_sub_u32 s1, s16, s0 // (op 2095) 

s_mul_i32 s0, s18, s15 // (op 2095) 
s_add_u32 s0, s0, s1 // (op 2095) 


s_mul_hi_i32 s1, s16, s9 // (op 2095) 
s_add_i32 s18, s1, s16 // (op 2095) 

s_ashr_i32 s1, s18, 31 // (op 2095) 
s_cmp_eq_i32 s9, 0 // (op 2095) 
s_mov_b32 s19, scc // (op 2095) 
s_cmp_eq_i32 s19, 0 // (op 2095) 
s_cselect_b32 s22, 0, -1 // (op 2095) 

s_and_b32 s19, s46, 31 // (op 2095) 
s_lshl_b32 s23, 1, s19 // (op 2095) 
s_add_i32 s24, s23, s22 // (op 2095) 


s_and_b32 s25, s1, s24 // (op 2095) 


s_add_i32 s1, s18, s25 // (op 2095) 


s_ashr_i32 s18, s1, s19 // (op 2095) 


s_lshl_b32 s1, s46, 24 // (op 2095) 
s_ashr_i32 s19, s1, 31 // (op 2095) 

s_xor_b32 s1, s18, s19 // (op 2095) 

s_sub_i32 s18, s1, s19 // (op 2095) 


s_add_u32 s50, s6, 64 // (op 2095) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2095) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2095) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2095) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2095) 

s_mul_i32 s1, s15, s24 // (op 2095) 

s_mul_i32 s19, s8, s1 // (op 2095) 

s_mul_i32 s1, s18, s19 // (op 2095) 

s_sub_u32 s19, s16, s1 // (op 2095) 

s_mov_b32 s1, s19 // (op 2095) 
s_mul_hi_i32 s22, s1, s14 // (op 2095) 
s_add_i32 s23, s22, s1 // (op 2095) 


s_ashr_i32 s1, s23, 31 // (op 2095) 
s_cmp_eq_i32 s14, 0 // (op 2095) 
s_mov_b32 s22, scc // (op 2095) 
s_cmp_eq_i32 s22, 0 // (op 2095) 
s_cselect_b32 s24, 0, -1 // (op 2095) 

s_and_b32 s22, s48, 31 // (op 2095) 
s_lshl_b32 s25, 1, s22 // (op 2095) 
s_add_i32 s49, s25, s24 // (op 2095) 


s_and_b32 s24, s1, s49 // (op 2095) 


s_add_i32 s1, s23, s24 // (op 2095) 


s_ashr_i32 s23, s1, s22 // (op 2095) 


s_lshl_b32 s1, s48, 24 // (op 2095) 
s_ashr_i32 s22, s1, 31 // (op 2095) 

s_xor_b32 s1, s23, s22 // (op 2095) 

s_sub_i32 s23, s1, s22 // (op 2095) 


s_mul_i32 s1, s47, s15 // (op 2095) 
s_add_u32 s24, s20, 64 // (op 2095) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2095) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2095) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2095) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2095) 

s_sub_i32 s22, s24, s1 // (op 2095) 

s_mul_i32 s24, s23, s22 // (op 2095) 


s_sub_u32 s25, s19, s24 // (op 2095) 


s_add_u32 s19, s25, s1 // (op 2095) 


s_cmp_eq_i32 s18, 0 // (op 2095) 
s_mov_b32 s1, scc // (op 2095) 

s_cmp_eq_i32 s1, 0 // (op 2095) 
s_cselect_b32 s18, s19, s0 // (op 2095) 



v_lshl_add_u32 v16, s18, 6, v15 // (op 2095) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2095) 
s_lshr_b32 s18, s0, 1 // (op 2095) 

v_lshl_add_u32 v15, s18, 8, v17 // (op 2095) 


v_mul_lo_u32 v17, v15, s26 // (op 2095) 

v_add_u32 v15, v16, v17 // (op 2095) 


s_mul_i32 s1, s26, s31 // (op 2097) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2097) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2097) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2097) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2097) most significant: sum
s_add_u32 s1, s1, s19 // (op 2097) most significant: sum


s_mov_b32 s52, s28 // (op 2097) reg expression
s_mov_b32 s53, s29 // (op 2097) reg expression
s_mov_b32 s54, s0 // (op 2097) reg expression
s_mov_b32 s55, 131072 // (op 2097) call()

s_mul_hi_i32 s0, s16, s10 // (op 2165) 
v_lshrrev_b32 v16, 2, v1 // (op 2165) 
s_add_i32 s1, s0, s16 // (op 2165) 

v_and_b32 v17, -4, v1 // (op 2165) 
s_ashr_i32 s0, s1, 31 // (op 2165) 
v_sub_u32 v18, v1, v17 // (op 2165) 

s_cmp_eq_i32 s10, 0 // (op 2165) 
s_mov_b32 s18, scc // (op 2165) 
v_lshlrev_b32 v17, 5, v18 // (op 2165) 

s_cmp_eq_i32 s18, 0 // (op 2165) 
s_cselect_b32 s19, 0, -1 // (op 2165) 

s_and_b32 s18, s11, 31 // (op 2165) 
s_lshl_b32 s22, 1, s18 // (op 2165) 
s_add_i32 s23, s22, s19 // (op 2165) 


s_and_b32 s19, s0, s23 // (op 2165) 


s_add_i32 s0, s1, s19 // (op 2165) 


s_ashr_i32 s1, s0, s18 // (op 2165) 


s_lshl_b32 s0, s11, 24 // (op 2165) 
s_ashr_i32 s18, s0, 31 // (op 2165) 

s_xor_b32 s0, s1, s18 // (op 2165) 

s_sub_i32 s1, s0, s18 // (op 2165) 


s_mul_hi_i32 s0, s1, s12 // (op 2165) 
s_add_i32 s18, s0, s1 // (op 2165) 

s_ashr_i32 s0, s18, 31 // (op 2165) 
s_cmp_eq_i32 s12, 0 // (op 2165) 
s_mov_b32 s19, scc // (op 2165) 
s_cmp_eq_i32 s19, 0 // (op 2165) 
s_cselect_b32 s22, 0, -1 // (op 2165) 

s_and_b32 s19, s13, 31 // (op 2165) 
s_lshl_b32 s23, 1, s19 // (op 2165) 
s_add_i32 s24, s23, s22 // (op 2165) 


s_and_b32 s25, s0, s24 // (op 2165) 


s_add_i32 s0, s18, s25 // (op 2165) 


s_ashr_i32 s18, s0, s19 // (op 2165) 


s_lshl_b32 s0, s13, 24 // (op 2165) 
s_ashr_i32 s19, s0, 31 // (op 2165) 

s_xor_b32 s0, s18, s19 // (op 2165) 

s_sub_i32 s18, s0, s19 // (op 2165) 


s_add_u32 s50, s6, 64 // (op 2165) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2165) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2165) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2165) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2165) 

s_mul_i32 s0, s18, s24 // (op 2165) 

s_sub_i32 s18, s1, s0 // (op 2165) 


s_mul_hi_i32 s19, s16, s9 // (op 2165) 
s_add_i32 s0, s19, s16 // (op 2165) 

s_ashr_i32 s1, s0, 31 // (op 2165) 
s_cmp_eq_i32 s9, 0 // (op 2165) 
s_mov_b32 s19, scc // (op 2165) 
s_cmp_eq_i32 s19, 0 // (op 2165) 
s_cselect_b32 s22, 0, -1 // (op 2165) 

s_and_b32 s19, s46, 31 // (op 2165) 
s_lshl_b32 s23, 1, s19 // (op 2165) 
s_add_i32 s49, s23, s22 // (op 2165) 


s_and_b32 s22, s1, s49 // (op 2165) 


s_add_i32 s1, s0, s22 // (op 2165) 


s_ashr_i32 s0, s1, s19 // (op 2165) 


s_lshl_b32 s1, s46, 24 // (op 2165) 
s_ashr_i32 s19, s1, 31 // (op 2165) 

s_xor_b32 s1, s0, s19 // (op 2165) 

s_sub_i32 s0, s1, s19 // (op 2165) 


s_mul_i32 s1, s15, s24 // (op 2165) 

s_mul_i32 s19, s8, s1 // (op 2165) 

s_mul_i32 s1, s0, s19 // (op 2165) 

s_sub_u32 s19, s16, s1 // (op 2165) 

s_mul_hi_i32 s1, s19, s14 // (op 2165) 
s_add_i32 s22, s1, s19 // (op 2165) 


s_ashr_i32 s1, s22, 31 // (op 2165) 
s_cmp_eq_i32 s14, 0 // (op 2165) 
s_mov_b32 s19, scc // (op 2165) 
s_cmp_eq_i32 s19, 0 // (op 2165) 
s_cselect_b32 s23, 0, -1 // (op 2165) 

s_and_b32 s19, s48, 31 // (op 2165) 
s_lshl_b32 s24, 1, s19 // (op 2165) 
s_add_i32 s25, s24, s23 // (op 2165) 


s_and_b32 s23, s1, s25 // (op 2165) 


s_add_i32 s1, s22, s23 // (op 2165) 


s_ashr_i32 s22, s1, s19 // (op 2165) 


s_lshl_b32 s1, s48, 24 // (op 2165) 
s_ashr_i32 s19, s1, 31 // (op 2165) 

s_xor_b32 s1, s22, s19 // (op 2165) 

s_sub_i32 s22, s1, s19 // (op 2165) 


s_cmp_eq_i32 s0, 0 // (op 2165) 
s_mov_b32 s1, scc // (op 2165) 

s_cmp_eq_i32 s1, 0 // (op 2165) 
s_cselect_b32 s0, s22, s18 // (op 2165) 



v_lshl_add_u32 v18, s0, 6, v16 // (op 2165) 


v_mul_lo_u32 v43, v18, s33 // (op 2165) most significant: low of low * high
v_mul_hi_u32 v16, v18, s32 // (op 2165) most significant: high of low * low
v_mul_lo_u32 v42, v18, s32 // (op 2165) least significant: low of low * low
v_add_u32 v43, v43, v16 // (op 2165) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2165) 
s_lshr_b32 s18, s0, 1 // (op 2165) 

v_lshl_add_u32 v16, s18, 8, v17 // (op 2165) 


v_mov_b32 v23, 0 // (op 2165) convert
v_mov_b32 v22, v16 // (op 2165) convert

v_add_co_u32 v16, s[0:1], v22, v42 // (op 2165) least significant half
v_addc_co_u32 v17, s[0:1], v23, v43, s[0:1] // (op 2165) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 2165) 


v_lshrrev_b64 v[42:43], 3, v[16:17] // (op 2165) 

v_mov_b32 v16, v42 // (op 2165) convert

s_mul_i32 s1, s32, s7 // (op 2167) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2167) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2167) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2167) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2167) most significant: sum
s_add_u32 s1, s1, s19 // (op 2167) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2167) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2167) 

s_mov_b32 s64, s34 // (op 2167) reg expression
s_mov_b32 s65, s35 // (op 2167) reg expression
s_mov_b32 s66, s0 // (op 2167) reg expression
s_mov_b32 s67, 131072 // (op 2167) call()

s_mul_hi_i32 s0, s16, s10 // (op 2229) 
v_and_b32 v17, -64, v1 // (op 2229) 
s_add_i32 s1, s0, s16 // (op 2229) 

v_sub_u32 v18, v1, v17 // (op 2229) 

s_ashr_i32 s0, s1, 31 // (op 2229) 
v_lshrrev_b32 v17, 6, v1 // (op 2229) 
s_cmp_eq_i32 s10, 0 // (op 2229) 
s_mov_b32 s18, scc // (op 2229) 
s_cmp_eq_i32 s18, 0 // (op 2229) 
s_cselect_b32 s19, 0, -1 // (op 2229) 

s_and_b32 s18, s11, 31 // (op 2229) 
s_lshl_b32 s22, 1, s18 // (op 2229) 
s_add_i32 s23, s22, s19 // (op 2229) 


s_and_b32 s19, s0, s23 // (op 2229) 


s_add_i32 s0, s1, s19 // (op 2229) 


s_ashr_i32 s1, s0, s18 // (op 2229) 


s_lshl_b32 s0, s11, 24 // (op 2229) 
s_ashr_i32 s18, s0, 31 // (op 2229) 

s_xor_b32 s0, s1, s18 // (op 2229) 

s_sub_i32 s1, s0, s18 // (op 2229) 


s_mul_hi_i32 s0, s1, s12 // (op 2229) 
s_add_i32 s18, s0, s1 // (op 2229) 

s_ashr_i32 s0, s18, 31 // (op 2229) 
s_cmp_eq_i32 s12, 0 // (op 2229) 
s_mov_b32 s19, scc // (op 2229) 
s_cmp_eq_i32 s19, 0 // (op 2229) 
s_cselect_b32 s22, 0, -1 // (op 2229) 

s_and_b32 s19, s13, 31 // (op 2229) 
s_lshl_b32 s23, 1, s19 // (op 2229) 
s_add_i32 s24, s23, s22 // (op 2229) 


s_and_b32 s25, s0, s24 // (op 2229) 


s_add_i32 s0, s18, s25 // (op 2229) 


s_ashr_i32 s18, s0, s19 // (op 2229) 


s_lshl_b32 s0, s13, 24 // (op 2229) 
s_ashr_i32 s19, s0, 31 // (op 2229) 

s_xor_b32 s0, s18, s19 // (op 2229) 

s_sub_i32 s18, s0, s19 // (op 2229) 


s_mul_i32 s0, s1, s15 // (op 2229) 

s_sub_u32 s1, s16, s0 // (op 2229) 

s_mul_i32 s0, s18, s15 // (op 2229) 
s_add_u32 s0, s0, s1 // (op 2229) 


s_mul_hi_i32 s1, s16, s9 // (op 2229) 
s_add_i32 s18, s1, s16 // (op 2229) 

s_ashr_i32 s1, s18, 31 // (op 2229) 
s_cmp_eq_i32 s9, 0 // (op 2229) 
s_mov_b32 s19, scc // (op 2229) 
s_cmp_eq_i32 s19, 0 // (op 2229) 
s_cselect_b32 s22, 0, -1 // (op 2229) 

s_and_b32 s19, s46, 31 // (op 2229) 
s_lshl_b32 s23, 1, s19 // (op 2229) 
s_add_i32 s24, s23, s22 // (op 2229) 


s_and_b32 s25, s1, s24 // (op 2229) 


s_add_i32 s1, s18, s25 // (op 2229) 


s_ashr_i32 s18, s1, s19 // (op 2229) 


s_lshl_b32 s1, s46, 24 // (op 2229) 
s_ashr_i32 s19, s1, 31 // (op 2229) 

s_xor_b32 s1, s18, s19 // (op 2229) 

s_sub_i32 s18, s1, s19 // (op 2229) 


s_add_u32 s50, s6, 64 // (op 2229) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2229) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2229) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2229) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2229) 

s_mul_i32 s1, s15, s24 // (op 2229) 

s_mul_i32 s19, s8, s1 // (op 2229) 

s_mul_i32 s1, s18, s19 // (op 2229) 

s_sub_u32 s19, s16, s1 // (op 2229) 

s_mov_b32 s1, s19 // (op 2229) 
s_mul_hi_i32 s22, s1, s14 // (op 2229) 
s_add_i32 s23, s22, s1 // (op 2229) 


s_ashr_i32 s1, s23, 31 // (op 2229) 
s_cmp_eq_i32 s14, 0 // (op 2229) 
s_mov_b32 s22, scc // (op 2229) 
s_cmp_eq_i32 s22, 0 // (op 2229) 
s_cselect_b32 s24, 0, -1 // (op 2229) 

s_and_b32 s22, s48, 31 // (op 2229) 
s_lshl_b32 s25, 1, s22 // (op 2229) 
s_add_i32 s49, s25, s24 // (op 2229) 


s_and_b32 s24, s1, s49 // (op 2229) 


s_add_i32 s1, s23, s24 // (op 2229) 


s_ashr_i32 s23, s1, s22 // (op 2229) 


s_lshl_b32 s1, s48, 24 // (op 2229) 
s_ashr_i32 s22, s1, 31 // (op 2229) 

s_xor_b32 s1, s23, s22 // (op 2229) 

s_sub_i32 s23, s1, s22 // (op 2229) 


s_mul_i32 s1, s47, s15 // (op 2229) 
s_add_u32 s24, s20, 64 // (op 2229) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2229) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2229) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2229) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2229) 

s_sub_i32 s22, s24, s1 // (op 2229) 

s_mul_i32 s24, s23, s22 // (op 2229) 


s_sub_u32 s25, s19, s24 // (op 2229) 


s_add_u32 s19, s25, s1 // (op 2229) 


s_cmp_eq_i32 s18, 0 // (op 2229) 
s_mov_b32 s1, scc // (op 2229) 

s_cmp_eq_i32 s1, 0 // (op 2229) 
s_cselect_b32 s18, s19, s0 // (op 2229) 



v_lshl_add_u32 v19, s18, 6, v18 // (op 2229) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2229) 
s_lshr_b32 s18, s0, 1 // (op 2229) 

v_lshl_add_u32 v18, s18, 3, v17 // (op 2229) 


v_mul_lo_u32 v17, v18, s36 // (op 2229) 

v_add_u32 v18, v19, v17 // (op 2229) 


s_mul_i32 s1, s36, s41 // (op 2231) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2231) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2231) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2231) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2231) most significant: sum
s_add_u32 s1, s1, s19 // (op 2231) most significant: sum


s_mov_b32 s68, s38 // (op 2231) reg expression
s_mov_b32 s69, s39 // (op 2231) reg expression
s_mov_b32 s70, s0 // (op 2231) reg expression
s_mov_b32 s71, 131072 // (op 2231) call()

s_mul_hi_i32 s0, s16, s10 // (op 2293) 
v_lshrrev_b32 v17, 2, v1 // (op 2293) 
s_add_i32 s1, s0, s16 // (op 2293) 

v_and_b32 v19, -4, v1 // (op 2293) 
s_ashr_i32 s0, s1, 31 // (op 2293) 
v_sub_u32 v20, v1, v19 // (op 2293) 

s_cmp_eq_i32 s10, 0 // (op 2293) 
s_mov_b32 s18, scc // (op 2293) 
s_cmp_eq_i32 s18, 0 // (op 2293) 
s_cselect_b32 s19, 0, -1 // (op 2293) 

s_and_b32 s18, s11, 31 // (op 2293) 
s_lshl_b32 s22, 1, s18 // (op 2293) 
s_add_i32 s23, s22, s19 // (op 2293) 


s_and_b32 s19, s0, s23 // (op 2293) 


s_add_i32 s0, s1, s19 // (op 2293) 


s_ashr_i32 s1, s0, s18 // (op 2293) 


s_lshl_b32 s0, s11, 24 // (op 2293) 
s_ashr_i32 s18, s0, 31 // (op 2293) 

s_xor_b32 s0, s1, s18 // (op 2293) 

s_sub_i32 s1, s0, s18 // (op 2293) 


s_mul_hi_i32 s0, s1, s12 // (op 2293) 
s_add_i32 s18, s0, s1 // (op 2293) 

s_ashr_i32 s0, s18, 31 // (op 2293) 
s_cmp_eq_i32 s12, 0 // (op 2293) 
s_mov_b32 s19, scc // (op 2293) 
s_cmp_eq_i32 s19, 0 // (op 2293) 
s_cselect_b32 s22, 0, -1 // (op 2293) 

s_and_b32 s19, s13, 31 // (op 2293) 
s_lshl_b32 s23, 1, s19 // (op 2293) 
s_add_i32 s24, s23, s22 // (op 2293) 


s_and_b32 s25, s0, s24 // (op 2293) 


s_add_i32 s0, s18, s25 // (op 2293) 


s_ashr_i32 s18, s0, s19 // (op 2293) 


s_lshl_b32 s0, s13, 24 // (op 2293) 
s_ashr_i32 s19, s0, 31 // (op 2293) 

s_xor_b32 s0, s18, s19 // (op 2293) 

s_sub_i32 s18, s0, s19 // (op 2293) 


s_add_u32 s50, s6, 64 // (op 2293) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2293) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2293) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2293) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2293) 

s_mul_i32 s0, s18, s24 // (op 2293) 

s_sub_i32 s18, s1, s0 // (op 2293) 


s_mul_hi_i32 s19, s16, s9 // (op 2293) 
s_add_i32 s0, s19, s16 // (op 2293) 

s_ashr_i32 s1, s0, 31 // (op 2293) 
s_cmp_eq_i32 s9, 0 // (op 2293) 
s_mov_b32 s19, scc // (op 2293) 
s_cmp_eq_i32 s19, 0 // (op 2293) 
s_cselect_b32 s22, 0, -1 // (op 2293) 

s_and_b32 s19, s46, 31 // (op 2293) 
s_lshl_b32 s23, 1, s19 // (op 2293) 
s_add_i32 s49, s23, s22 // (op 2293) 


s_and_b32 s22, s1, s49 // (op 2293) 


s_add_i32 s1, s0, s22 // (op 2293) 


s_ashr_i32 s0, s1, s19 // (op 2293) 


s_lshl_b32 s1, s46, 24 // (op 2293) 
s_ashr_i32 s19, s1, 31 // (op 2293) 

s_xor_b32 s1, s0, s19 // (op 2293) 

s_sub_i32 s0, s1, s19 // (op 2293) 


s_mul_i32 s1, s15, s24 // (op 2293) 

s_mul_i32 s19, s8, s1 // (op 2293) 

s_mul_i32 s1, s0, s19 // (op 2293) 

s_sub_u32 s19, s16, s1 // (op 2293) 

s_mul_hi_i32 s1, s19, s14 // (op 2293) 
s_add_i32 s22, s1, s19 // (op 2293) 


s_ashr_i32 s1, s22, 31 // (op 2293) 
s_cmp_eq_i32 s14, 0 // (op 2293) 
s_mov_b32 s19, scc // (op 2293) 
s_cmp_eq_i32 s19, 0 // (op 2293) 
s_cselect_b32 s23, 0, -1 // (op 2293) 

s_and_b32 s19, s48, 31 // (op 2293) 
s_lshl_b32 s24, 1, s19 // (op 2293) 
s_add_i32 s25, s24, s23 // (op 2293) 


s_and_b32 s23, s1, s25 // (op 2293) 


s_add_i32 s1, s22, s23 // (op 2293) 


s_ashr_i32 s22, s1, s19 // (op 2293) 


s_lshl_b32 s1, s48, 24 // (op 2293) 
s_ashr_i32 s19, s1, 31 // (op 2293) 

s_xor_b32 s1, s22, s19 // (op 2293) 

s_sub_i32 s22, s1, s19 // (op 2293) 


s_cmp_eq_i32 s0, 0 // (op 2293) 
s_mov_b32 s1, scc // (op 2293) 

s_cmp_eq_i32 s1, 0 // (op 2293) 
s_cselect_b32 s0, s22, s18 // (op 2293) 



v_lshl_add_u32 v19, s0, 6, v17 // (op 2293) 


v_mul_lo_u32 v17, v19, s42 // (op 2293) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 2293) 
s_lshr_b32 s18, s0, 1 // (op 2293) 

v_lshl_add_u32 v19, s18, 3, v20 // (op 2293) 


v_add_u32 v20, v19, v17 // (op 2293) 


s_mul_i32 s1, s42, s7 // (op 2295) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2295) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2295) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2295) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2295) most significant: sum
s_add_u32 s1, s1, s19 // (op 2295) most significant: sum


s_mov_b32 s72, s44 // (op 2295) reg expression
s_mov_b32 s73, s45 // (op 2295) reg expression
s_mov_b32 s74, s0 // (op 2295) reg expression
s_mov_b32 s75, 131072 // (op 2295) call()

v_and_b32 v17, -4, v1 // (op 2391) 
v_sub_u32 v19, v1, v17 // (op 2391) 

v_lshlrev_b32 v17, 4, v19 // (op 2391) 

v_lshrrev_b32 v19, 2, v1 // (op 2391) 
v_lshl_add_u32 v22, v19, 7, v17 // (op 2391) 


v_and_b32 v17, -4, v1 // (op 2449) 
v_sub_u32 v19, v1, v17 // (op 2449) 

v_lshlrev_b32 v17, 5, v19 // (op 2449) 

v_lshrrev_b32 v19, 2, v1 // (op 2449) 
v_lshl_add_u32 v23, v19, 7, v17 // (op 2449) 


v_lshlrev_b32 v17, 2, v23 // (op 2449) 

v_lshrrev_b32 v19, 3, v17 // (op 2449) 

v_and_b32 v17, -64, v1 // (op 2507) 
v_sub_u32 v23, v1, v17 // (op 2507) 

v_and_b32 v17, 4294967232, v1 // (op 2507) 
v_add_u32 v24, v17, v23 // (op 2507) 


v_and_b32 v17, -4, v1 // (op 2565) 
v_sub_u32 v23, v1, v17 // (op 2565) 

v_and_b32 v17, 4294967292, v1 // (op 2565) 
v_add_u32 v25, v17, v23 // (op 2565) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 598) 
s_and_b32 s18, s0, 4294967294 // (op 598) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 597) 
s_mov_b32 s19, s0 // (op 597) convert

s_cmp_lt_u32 s18, s19 // (op 597) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_597 // (op 597) Condition: Top (jump to RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_597 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_597:

s_add_u32 s18, s18, 1 // (op 599) 
v_mov_b64 v[26:27], s[26:27] // (op 180) convert

v_mov_b32 v14, v15 // (op 180) 
v_mov_b32 v17, v14 // (op 180) reg expression
buffer_load_dwordx4 v[28:31], v17, s[52:55], 0 offen // (op 180) Load value
v_add_u32 v17, v17, v26 // (op 180) 
buffer_load_dwordx4 v[32:35], v17, s[52:55], 0 offen // (op 180) Load value



s_lshl_b32 s19, s26, 7 // (op 2085) 
v_add_u32 v15, v15, s19 // (op 2085) 



s_barrier  // (op 2794) 
v_mov_b32 v14, v16 // (op 194) 




buffer_load_dwordx4 v[36:39], v14, s[64:67], 0 offen // (op 194) Load value

v_add_u32 v16, 64, v16 // (op 2157) 
s_waitcnt vmcnt(2)
ds_write_b128 v22, v[28:31] offset:4352 // (op 181) Store local data 


v_mov_b32 v14, v18 // (op 195) 
v_mov_b32 v17, v14 // (op 195) reg expression

buffer_load_ubyte v21, v17, s[68:71], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s19, s36, 2 // (op 2221) 
v_add_u32 v18, v18, s19 // (op 2221) 

s_waitcnt vmcnt(2)
ds_write_b128 v22, v[32:35] offset:4416 // (op 181) Store local data 



v_mov_b32 v14, v20 // (op 196) 

s_barrier  // (op 2807) 
buffer_load_ubyte v17, v14, s[72:75], 0 offen offset: 0 // (op 196) Load value




v_add_u32 v20, 4, v20 // (op 2285) 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[36:39] // (op 191) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 2797) 



s_barrier  // (op 2821) 
ds_read_b64_tr_b8 v[26:27], v0 offset:4352 // (op 183) Transpose load local data 



ds_read_b64_tr_b8 v[28:29], v0 offset:4864 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:8448 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:8960 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v21 offset:12544 // (op 192) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 2811) 



s_barrier  // (op 2835) 
ds_read_b128 v[34:37], v4 // (op 185) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 2825) 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v17 offset:4096 // (op 193) Store local data 



ds_read_u8 v14, v3 offset:12544 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 2839) 



ds_read_u8 v17, v5 offset:4096 // (op 189) Load local data 



ds_read_b128 v[38:41], v4 offset:2048 // (op 354) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:37], a[0:3], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 190) 



ds_read_u8 v21, v5 offset:4224 // (op 358) Load local data 



ds_read_b64_tr_b8 v[42:43], v0 offset:4384 // (op 572) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:4896 // (op 572) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[38:41], a[4:7], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 359) 
ds_read_b64_tr_b8 v[46:47], v0 offset:8480 // (op 572) Transpose load local data 


ds_read_b64_tr_b8 v[48:49], v0 offset:8992 // (op 572) Transpose load local data 


ds_read_u8 v14, v3 offset:12576 // (op 575) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:49], v[34:37], a[8:11], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 577) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[38:41], a[12:15], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 608) 




s_ashr_i64 s[0:1], s[30:31], 7 // (op 597) 
s_mov_b32 s19, s0 // (op 597) convert

s_cmp_lt_u32 s18, s19 // (op 597) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_597 // (op 597) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopTop_KLoopTail_597 if true)
RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ForLoopBottom_KLoopTail_597:























s_barrier  // (op 2937) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135
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



v_accvgpr_read v6, a4 // (op 1021) 
v_accvgpr_read v7, a5 // (op 1021) 
v_accvgpr_read v8, a6 // (op 1021) 
v_accvgpr_read v9, a7 // (op 1021) 
v_mul_f32 v14, v11, v6 // (op 1021) 
v_mul_f32 v15, v11, v7 // (op 1021) 
v_mul_f32 v16, v11, v8 // (op 1021) 
v_mul_f32 v17, v11, v9 // (op 1021) 

v_lshrrev_b32 v6, 16, v14 // (op 1023) 
v_lshrrev_b32 v7, 16, v15 // (op 1023) 
v_lshrrev_b32 v8, 16, v16 // (op 1023) 
v_lshrrev_b32 v9, 16, v17 // (op 1023) 


v_pack_b32_f16 v4, v6, v7 // (op 1024) 
v_pack_b32_f16 v5, v8, v9 // (op 1024) 

s_lshl_b32 s0, s4, 6 // (op 1024) 
v_add_u32 v0, v13, s0 // (op 1024) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1024) Store value



v_accvgpr_read v6, a8 // (op 1047) 
v_accvgpr_read v7, a9 // (op 1047) 
v_accvgpr_read v8, a10 // (op 1047) 
v_accvgpr_read v9, a11 // (op 1047) 
v_mul_f32 v14, v11, v6 // (op 1047) 
v_mul_f32 v15, v11, v7 // (op 1047) 
v_mul_f32 v16, v11, v8 // (op 1047) 
v_mul_f32 v17, v11, v9 // (op 1047) 

v_lshrrev_b32 v6, 16, v14 // (op 1049) 
v_lshrrev_b32 v7, 16, v15 // (op 1049) 
v_lshrrev_b32 v8, 16, v16 // (op 1049) 
v_lshrrev_b32 v9, 16, v17 // (op 1049) 


v_pack_b32_f16 v4, v6, v7 // (op 1051) 
v_pack_b32_f16 v5, v8, v9 // (op 1051) 

v_add_u32 v0, 64, v13 // (op 1051) 

buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1051) Store value



v_accvgpr_read v6, a12 // (op 1057) 
v_accvgpr_read v7, a13 // (op 1057) 
v_accvgpr_read v8, a14 // (op 1057) 
v_accvgpr_read v9, a15 // (op 1057) 
v_mul_f32 v14, v11, v6 // (op 1057) 
v_mul_f32 v15, v11, v7 // (op 1057) 
v_mul_f32 v16, v11, v8 // (op 1057) 
v_mul_f32 v17, v11, v9 // (op 1057) 

v_lshrrev_b32 v6, 16, v14 // (op 1059) 
v_lshrrev_b32 v7, 16, v15 // (op 1059) 
v_lshrrev_b32 v8, 16, v16 // (op 1059) 
v_lshrrev_b32 v9, 16, v17 // (op 1059) 


v_pack_b32_f16 v4, v6, v7 // (op 1053) 
v_pack_b32_f16 v5, v8, v9 // (op 1053) 

s_lshl_b32 s0, s4, 6 // (op 1053) 
s_add_u32 s0, s0, 64 // (op 1053) 
v_add_u32 v0, v13, s0 // (op 1053) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1053) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 64, v12 // (op 1143) 

buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 1143) Load value

v_accvgpr_read v4, a8 // (op 1134) 
v_accvgpr_read v5, a9 // (op 1134) 
v_accvgpr_read v6, a10 // (op 1134) 
v_accvgpr_read v7, a11 // (op 1134) 
v_mul_f32 v14, v11, v4 // (op 1134) 
v_mul_f32 v15, v11, v5 // (op 1134) 
v_mul_f32 v16, v11, v6 // (op 1134) 
v_mul_f32 v17, v11, v7 // (op 1134) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 1142) 
v_lshrrev_b32 v7, 16, v8 // (op 1142) 
v_lshlrev_b32 v6, 16, v6 // (op 1142) 
v_lshlrev_b32 v7, 16, v7 // (op 1142) 
v_mul_f32 v18, v10, v6 // (op 1142) 
v_mul_f32 v19, v10, v7 // (op 1142) 
v_and_b32 v6, 65535, v9 // (op 1142) 
v_lshrrev_b32 v7, 16, v9 // (op 1142) 
v_lshlrev_b32 v6, 16, v6 // (op 1142) 
v_lshlrev_b32 v7, 16, v7 // (op 1142) 
v_mul_f32 v20, v10, v6 // (op 1142) 
v_mul_f32 v21, v10, v7 // (op 1142) 


v_add_f32 v6, v18, v14 // (op 1135) 
v_add_f32 v7, v19, v15 // (op 1135) 
v_add_f32 v8, v20, v16 // (op 1135) 
v_add_f32 v9, v21, v17 // (op 1135) 


v_lshrrev_b32 v14, 16, v6 // (op 1136) 
v_lshrrev_b32 v15, 16, v7 // (op 1136) 
v_lshrrev_b32 v16, 16, v8 // (op 1136) 
v_lshrrev_b32 v17, 16, v9 // (op 1136) 


v_pack_b32_f16 v8, v14, v15 // (op 1138) 
v_pack_b32_f16 v9, v16, v17 // (op 1138) 

v_add_u32 v0, 64, v13 // (op 1138) 

buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 1138) Store value





s_lshl_b32 s0, s2, 6 // (op 1148) 
s_add_u32 s0, s0, 64 // (op 1148) 
v_add_u32 v0, v12, s0 // (op 1148) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 1148) Load value

v_accvgpr_read v4, a12 // (op 1150) 
v_accvgpr_read v5, a13 // (op 1150) 
v_accvgpr_read v6, a14 // (op 1150) 
v_accvgpr_read v7, a15 // (op 1150) 
v_mul_f32 v14, v11, v4 // (op 1150) 
v_mul_f32 v15, v11, v5 // (op 1150) 
v_mul_f32 v16, v11, v6 // (op 1150) 
v_mul_f32 v17, v11, v7 // (op 1150) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 1145) 
v_lshrrev_b32 v7, 16, v8 // (op 1145) 
v_lshlrev_b32 v6, 16, v6 // (op 1145) 
v_lshlrev_b32 v7, 16, v7 // (op 1145) 
v_mul_f32 v18, v10, v6 // (op 1145) 
v_mul_f32 v19, v10, v7 // (op 1145) 
v_and_b32 v6, 65535, v9 // (op 1145) 
v_lshrrev_b32 v7, 16, v9 // (op 1145) 
v_lshlrev_b32 v6, 16, v6 // (op 1145) 
v_lshlrev_b32 v7, 16, v7 // (op 1145) 
v_mul_f32 v20, v10, v6 // (op 1145) 
v_mul_f32 v21, v10, v7 // (op 1145) 


v_add_f32 v6, v18, v14 // (op 1146) 
v_add_f32 v7, v19, v15 // (op 1146) 
v_add_f32 v8, v20, v16 // (op 1146) 
v_add_f32 v9, v21, v17 // (op 1146) 


v_lshrrev_b32 v14, 16, v6 // (op 1147) 
v_lshrrev_b32 v15, 16, v7 // (op 1147) 
v_lshrrev_b32 v16, 16, v8 // (op 1147) 
v_lshrrev_b32 v17, 16, v9 // (op 1147) 


v_pack_b32_f16 v8, v14, v15 // (op 1140) 
v_pack_b32_f16 v9, v16, v17 // (op 1140) 

s_lshl_b32 s0, s4, 6 // (op 1140) 
s_add_u32 s0, s0, 64 // (op 1140) 
v_add_u32 v0, v13, s0 // (op 1140) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 1140) Store value





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





s_lshl_b32 s0, s2, 6 // (op 1103) 
v_add_u32 v0, v12, s0 // (op 1103) 


buffer_load_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 1103) Load value

v_accvgpr_read v4, a4 // (op 1099) 
v_accvgpr_read v5, a5 // (op 1099) 
v_accvgpr_read v6, a6 // (op 1099) 
v_accvgpr_read v7, a7 // (op 1099) 
v_mul_f32 v14, v11, v4 // (op 1099) 
v_mul_f32 v15, v11, v5 // (op 1099) 
v_mul_f32 v16, v11, v6 // (op 1099) 
v_mul_f32 v17, v11, v7 // (op 1099) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 1104) 
v_lshrrev_b32 v7, 16, v8 // (op 1104) 
v_lshlrev_b32 v6, 16, v6 // (op 1104) 
v_lshlrev_b32 v7, 16, v7 // (op 1104) 
v_mul_f32 v18, v10, v6 // (op 1104) 
v_mul_f32 v19, v10, v7 // (op 1104) 
v_and_b32 v6, 65535, v9 // (op 1104) 
v_lshrrev_b32 v7, 16, v9 // (op 1104) 
v_lshlrev_b32 v6, 16, v6 // (op 1104) 
v_lshlrev_b32 v7, 16, v7 // (op 1104) 
v_mul_f32 v20, v10, v6 // (op 1104) 
v_mul_f32 v21, v10, v7 // (op 1104) 


v_add_f32 v6, v18, v14 // (op 1100) 
v_add_f32 v7, v19, v15 // (op 1100) 
v_add_f32 v8, v20, v16 // (op 1100) 
v_add_f32 v9, v21, v17 // (op 1100) 


v_lshrrev_b32 v14, 16, v6 // (op 1101) 
v_lshrrev_b32 v15, 16, v7 // (op 1101) 
v_lshrrev_b32 v16, 16, v8 // (op 1101) 
v_lshrrev_b32 v17, 16, v9 // (op 1101) 


v_pack_b32_f16 v8, v14, v15 // (op 1102) 
v_pack_b32_f16 v9, v16, v17 // (op 1102) 

s_lshl_b32 s0, s4, 6 // (op 1102) 
v_add_u32 v0, v13, s0 // (op 1102) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 1102) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_ConditionalBottom_0_beta_135:












s_endpgm  // End of RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC
.LRR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_end:
.size RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC, .LRR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC_end-RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC
  .amdhsa_next_free_vgpr 104
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25600 // lds bytes
  .amdhsa_accum_offset 88
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
    macrotile: [64, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC
    .symbol: RR_GEMM_NN_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__LDSSYNC.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 25600
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 86
    .agpr_count: 16
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

