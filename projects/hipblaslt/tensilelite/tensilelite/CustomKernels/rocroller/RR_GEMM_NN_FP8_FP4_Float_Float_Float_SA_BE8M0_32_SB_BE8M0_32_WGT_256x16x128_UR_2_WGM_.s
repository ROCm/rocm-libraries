// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_,@function
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s8 // (op 3800) 
s_mul_hi_i32 s1, s16, s8 // (op 3989) 
v_and_b32 v0, 15, v1 // (op 3915) 
v_and_b32 v3, 63, v1 // (op 3915) 
v_lshrrev_b32 v4, 4, v3 // (op 3915) 

v_lshlrev_b32 v3, 5, v4 // (op 3915) 

v_lshl_add_u32 v4, v0, 7, v3 // (op 3915) 


v_lshlrev_b32 v0, 2, v4 // (op 3915) 

v_lshrrev_b32 v3, 3, v0 // (op 3915) 

v_and_b32 v0, 15, v1 // (op 4044) 
v_and_b32 v4, 63, v1 // (op 4044) 
v_lshrrev_b32 v5, 4, v4 // (op 4044) 

v_lshlrev_b32 v4, 5, v5 // (op 4044) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 4044) 


v_lshlrev_b32 v0, 2, v5 // (op 4044) 

v_lshrrev_b32 v4, 3, v0 // (op 4044) 

v_lshrrev_b32 v0, 6, v1 // (op 3800) 
s_add_i32 s18, s0, s16 // (op 3800) 

v_and_b32 v5, 63, v1 // (op 3800) 
s_ashr_i32 s0, s18, 31 // (op 3800) 
v_lshrrev_b32 v6, 4, v5 // (op 3800) 
s_cmp_eq_i32 s8, 0 // (op 3800) 
s_mov_b32 s19, scc // (op 3800) 
v_lshlrev_b32 v7, 2, v6 // (op 3800) 

s_cmp_eq_i32 s19, 0 // (op 3800) 
s_cselect_b32 s51, 0, -1 // (op 3800) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3800) 


s_and_b32 s19, s9, 31 // (op 3800) 
v_lshrrev_b32 v0, 2, v5 // (op 3800) 

v_and_b32 v5, 3, v0 // (op 3800) 

v_and_b32 v0, 48, v1 // (op 3862) 
v_and_b32 v7, 63, v1 // (op 3862) 
v_sub_u32 v8, v7, v0 // (op 3862) 

v_lshrrev_b32 v7, 1, v8 // (op 3862) 
v_add_lshl_u32 v9, v0, v7, 8 // (op 3862) 


v_and_b32 v0, 1, v8 // (op 3862) 

v_lshlrev_b32 v7, 3, v0 // (op 3862) 

v_lshrrev_b32 v0, 6, v1 // (op 3862) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 3862) 


v_add_u32 v0, v9, v8 // (op 3862) 


v_lshrrev_b32 v7, 6, v1 // (op 3973) 
v_and_b32 v8, 15, v1 // (op 3973) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 3973) 


v_and_b32 v7, 63, v1 // (op 3973) 
v_lshrrev_b32 v8, 4, v7 // (op 3973) 

v_lshl_add_u32 v7, v8, 8, v9 // (op 3973) 


v_lshrrev_b32 v8, 6, v1 // (op 3989) 
v_and_b32 v9, 63, v1 // (op 3989) 
v_and_b32 v10, 48, v1 // (op 4020) 
v_and_b32 v11, 63, v1 // (op 4020) 
v_sub_u32 v12, v11, v10 // (op 4020) 

v_lshrrev_b32 v11, 1, v12 // (op 4020) 
v_add_lshl_u32 v13, v10, v11, 8 // (op 4020) 


v_and_b32 v10, 1, v12 // (op 4020) 

v_lshlrev_b32 v11, 3, v10 // (op 4020) 

v_lshrrev_b32 v10, 6, v1 // (op 4020) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 4020) 


v_add_u32 v10, v13, v12 // (op 4020) 


v_lshrrev_b32 v11, 6, v1 // (op 4073) 
v_and_b32 v12, 15, v1 // (op 4073) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 4073) 


v_and_b32 v11, 63, v1 // (op 4073) 
v_lshrrev_b32 v12, 4, v11 // (op 4073) 

v_lshl_add_u32 v11, v12, 8, v13 // (op 4073) 


s_lshl_b32 s52, 1, s19 // (op 3800) 
v_and_b32 v12, 3, v1 // (op 3800) 
s_add_i32 s53, s52, s51 // (op 3800) 


s_and_b32 s51, s0, s53 // (op 3800) 


s_add_i32 s0, s1, s16 // (op 3989) 

s_ashr_i32 s1, s0, 31 // (op 3989) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 3800) 


s_add_i32 s52, s18, s51 // (op 3800) 


s_ashr_i32 s18, s52, s19 // (op 3800) 


s_lshl_b32 s19, s9, 24 // (op 3800) 
s_ashr_i32 s51, s19, 31 // (op 3800) 

s_xor_b32 s19, s18, s51 // (op 3800) 

s_sub_i32 s18, s19, s51 // (op 3800) 


s_mul_hi_i32 s19, s18, s10 // (op 3800) 
s_add_i32 s51, s19, s18 // (op 3800) 

s_ashr_i32 s19, s51, 31 // (op 3800) 
s_cmp_eq_i32 s10, 0 // (op 3800) 
s_mov_b32 s52, scc // (op 3800) 
s_cmp_eq_i32 s52, 0 // (op 3800) 
s_cselect_b32 s53, 0, -1 // (op 3800) 

s_and_b32 s52, s11, 31 // (op 3800) 
v_lshrrev_b32 v5, 4, v9 // (op 3989) 
s_cmp_eq_i32 s8, 0 // (op 3989) 
s_mov_b32 s54, scc // (op 3989) 
v_lshlrev_b32 v12, 2, v5 // (op 3989) 

s_cmp_eq_i32 s54, 0 // (op 3989) 
s_cselect_b32 s55, 0, -1 // (op 3989) 

s_lshl_b32 s54, 1, s52 // (op 3800) 
v_lshl_add_u32 v5, v8, 4, v12 // (op 3989) 


v_lshrrev_b32 v8, 2, v9 // (op 3989) 

s_add_i32 s56, s54, s53 // (op 3800) 


s_and_b32 s53, s19, s56 // (op 3800) 


s_add_i32 s19, s51, s53 // (op 3800) 


s_ashr_i32 s51, s19, s52 // (op 3800) 


s_lshl_b32 s19, s11, 24 // (op 3800) 
s_ashr_i32 s52, s19, 31 // (op 3800) 

s_xor_b32 s19, s51, s52 // (op 3800) 

s_sub_i32 s51, s19, s52 // (op 3800) 


s_mul_i32 s19, s18, s13 // (op 3800) 
s_sub_u32 s52, s16, s19 // (op 3800) 

s_mul_i32 s19, s51, s13 // (op 3800) 
s_add_u32 s19, s19, s52 // (op 3800) 

s_mul_hi_i32 s52, s16, s7 // (op 3800) 
s_add_i32 s53, s52, s16 // (op 3800) 

s_ashr_i32 s52, s53, 31 // (op 3800) 
s_cmp_eq_i32 s7, 0 // (op 3800) 
s_mov_b32 s54, scc // (op 3800) 
s_cmp_eq_i32 s54, 0 // (op 3800) 
s_cselect_b32 s56, 0, -1 // (op 3800) 

s_and_b32 s54, s14, 31 // (op 3800) 
s_lshl_b32 s57, 1, s54 // (op 3800) 
s_add_i32 s58, s57, s56 // (op 3800) 


s_and_b32 s56, s52, s58 // (op 3800) 


s_add_i32 s52, s53, s56 // (op 3800) 


s_ashr_i32 s53, s52, s54 // (op 3800) 


s_lshl_b32 s52, s14, 24 // (op 3800) 
s_ashr_i32 s54, s52, 31 // (op 3800) 

s_xor_b32 s52, s53, s54 // (op 3800) 

s_sub_i32 s53, s52, s54 // (op 3800) 


s_and_b32 s52, s9, 31 // (op 3989) 
s_lshl_b32 s54, 1, s52 // (op 3989) 
v_and_b32 v9, 3, v8 // (op 3989) 

s_add_i32 s56, s54, s55 // (op 3989) 


v_and_b32 v8, 3, v1 // (op 3989) 
s_and_b32 s54, s1, s56 // (op 3989) 


v_lshl_add_u32 v12, v9, 2, v8 // (op 3989) 


s_add_i32 s1, s0, s54 // (op 3989) 


s_ashr_i32 s0, s1, s52 // (op 3989) 


s_lshl_b32 s1, s9, 24 // (op 3989) 
s_ashr_i32 s52, s1, 31 // (op 3989) 

s_xor_b32 s1, s0, s52 // (op 3989) 

s_sub_i32 s0, s1, s52 // (op 3989) 


s_mul_hi_i32 s1, s0, s10 // (op 3989) 
s_add_i32 s52, s1, s0 // (op 3989) 

s_ashr_i32 s1, s52, 31 // (op 3989) 
s_cmp_eq_i32 s10, 0 // (op 3989) 
s_mov_b32 s54, scc // (op 3989) 
s_cmp_eq_i32 s54, 0 // (op 3989) 
s_cselect_b32 s55, 0, -1 // (op 3989) 

s_and_b32 s54, s11, 31 // (op 3989) 
s_lshl_b32 s56, 1, s54 // (op 3989) 
s_add_i32 s57, s56, s55 // (op 3989) 


s_and_b32 s55, s1, s57 // (op 3989) 


s_add_i32 s1, s52, s55 // (op 3989) 


s_ashr_i32 s52, s1, s54 // (op 3989) 


s_lshl_b32 s1, s11, 24 // (op 3989) 
s_ashr_i32 s54, s1, 31 // (op 3989) 

s_xor_b32 s1, s52, s54 // (op 3989) 

s_sub_i32 s52, s1, s54 // (op 3989) 


s_mul_i32 s1, s0, s13 // (op 3989) 
s_sub_u32 s54, s16, s1 // (op 3989) 

s_mul_i32 s1, s52, s13 // (op 3989) 
s_add_u32 s1, s1, s54 // (op 3989) 

s_mul_hi_i32 s54, s16, s7 // (op 3989) 
s_add_i32 s55, s54, s16 // (op 3989) 

s_ashr_i32 s54, s55, 31 // (op 3989) 
s_cmp_eq_i32 s7, 0 // (op 3989) 
s_mov_b32 s56, scc // (op 3989) 
s_cmp_eq_i32 s56, 0 // (op 3989) 
s_cselect_b32 s57, 0, -1 // (op 3989) 

s_and_b32 s56, s14, 31 // (op 3989) 
s_lshl_b32 s58, 1, s56 // (op 3989) 
s_add_i32 s59, s58, s57 // (op 3989) 


s_and_b32 s57, s54, s59 // (op 3989) 


s_add_i32 s54, s55, s57 // (op 3989) 


s_ashr_i32 s55, s54, s56 // (op 3989) 


s_lshl_b32 s54, s14, 24 // (op 3989) 
s_ashr_i32 s56, s54, 31 // (op 3989) 

s_xor_b32 s54, s55, s56 // (op 3989) 

s_sub_i32 s55, s54, s56 // (op 3989) 


s_add_u32 s56, s4, 16 // (op 3800) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 3800) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3800) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3800) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 3800) 

s_mul_i32 s54, s13, s56 // (op 3800) 
s_mul_i32 s58, s6, s54 // (op 3800) 

s_mul_i32 s54, s53, s58 // (op 3800) 

s_sub_u32 s58, s16, s54 // (op 3800) 

s_mov_b32 s54, s58 // (op 3800) 
s_mul_hi_i32 s59, s54, s12 // (op 3800) 
s_add_i32 s60, s59, s54 // (op 3800) 


s_ashr_i32 s54, s60, 31 // (op 3800) 
s_cmp_eq_i32 s12, 0 // (op 3800) 
s_mov_b32 s59, scc // (op 3800) 
s_cmp_eq_i32 s59, 0 // (op 3800) 
s_cselect_b32 s61, 0, -1 // (op 3800) 

s_add_u32 s62, s4, 16 // (op 3989) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 3989) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 3989) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 3989) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 3989) 

s_mul_i32 s59, s13, s62 // (op 3989) 
s_mul_i32 s64, s6, s59 // (op 3989) 

s_mul_i32 s59, s55, s64 // (op 3989) 

s_sub_u32 s64, s16, s59 // (op 3989) 

s_mov_b32 s59, s64 // (op 3989) 
s_mul_hi_i32 s65, s59, s12 // (op 3989) 
s_add_i32 s66, s65, s59 // (op 3989) 


s_ashr_i32 s59, s66, 31 // (op 3989) 
s_cmp_eq_i32 s12, 0 // (op 3989) 
s_mov_b32 s65, scc // (op 3989) 
s_cmp_eq_i32 s65, 0 // (op 3989) 
s_cselect_b32 s67, 0, -1 // (op 3989) 


s_waitcnt lgkmcnt(0)
s_and_b32 s65, s48, 31 // (op 3800) 
s_lshl_b32 s68, 1, s65 // (op 3800) 
s_add_i32 s69, s68, s61 // (op 3800) 


s_and_b32 s61, s54, s69 // (op 3800) 


s_add_i32 s54, s60, s61 // (op 3800) 


s_ashr_i32 s60, s54, s65 // (op 3800) 


s_lshl_b32 s54, s48, 24 // (op 3800) 
s_ashr_i32 s61, s54, 31 // (op 3800) 

s_xor_b32 s54, s60, s61 // (op 3800) 

s_sub_i32 s60, s54, s61 // (op 3800) 


s_mul_i32 s54, s15, s13 // (op 3800) 
s_and_b32 s61, s48, 31 // (op 3989) 
s_lshl_b32 s65, 1, s61 // (op 3989) 
s_add_i32 s68, s65, s67 // (op 3989) 


s_and_b32 s65, s59, s68 // (op 3989) 


s_add_i32 s59, s66, s65 // (op 3989) 


s_ashr_i32 s65, s59, s61 // (op 3989) 


s_lshl_b32 s59, s48, 24 // (op 3989) 
s_ashr_i32 s61, s59, 31 // (op 3989) 

s_xor_b32 s59, s65, s61 // (op 3989) 

s_sub_i32 s65, s59, s61 // (op 3989) 


s_mul_i32 s59, s15, s13 // (op 3989) 
s_add_u32 s66, s20, 256 // (op 3800) Start of Int64 add, locking SCC
s_addc_u32 s67, s21, 0 // (op 3800) End of Int64 add, unlocking SCC
s_sub_u32 s68, s66, 1 // (op 3800) least significant half; sets scc
s_subb_u32 s69, s67, 0 // (op 3800) most significant half; uses scc

s_ashr_i64 s[66:67], s[68:69], 8 // (op 3800) 

s_sub_i32 s61, s66, s54 // (op 3800) 

s_mul_i32 s66, s60, s61 // (op 3800) 

s_sub_u32 s61, s58, s66 // (op 3800) 


s_add_u32 s58, s61, s54 // (op 3800) 


s_cmp_eq_i32 s53, 0 // (op 3800) 
s_mov_b32 s54, scc // (op 3800) 

s_cmp_eq_i32 s54, 0 // (op 3800) 
s_cselect_b32 s53, s58, s19 // (op 3800) 


s_mul_i32 s19, s51, s56 // (op 3800) 


v_lshl_add_u32 v8, s53, 8, v6 // (op 3800) 


s_sub_i32 s51, s18, s19 // (op 3800) 


s_cmp_eq_i32 s54, 0 // (op 3800) 
s_cselect_b32 s53, s60, s51 // (op 3800) 



v_lshl_add_u32 v6, s53, 4, v13 // (op 3800) 


v_mul_lo_u32 v9, v6, s2 // (op 3800) 

v_add_lshl_u32 v6, v8, v9, 2 // (op 3800) 


s_add_u32 s18, s20, 256 // (op 3989) Start of Int64 add, locking SCC
s_addc_u32 s19, s21, 0 // (op 3989) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 3989) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 3989) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 8 // (op 3989) 

s_sub_i32 s51, s18, s59 // (op 3989) 

s_mul_i32 s18, s65, s51 // (op 3989) 

s_sub_u32 s19, s64, s18 // (op 3989) 


s_add_u32 s18, s19, s59 // (op 3989) 


s_cmp_eq_i32 s55, 0 // (op 3989) 
s_mov_b32 s19, scc // (op 3989) 

s_cmp_eq_i32 s19, 0 // (op 3989) 
s_cselect_b32 s51, s18, s1 // (op 3989) 


s_mul_i32 s1, s52, s62 // (op 3989) 


v_lshl_add_u32 v8, s51, 8, v5 // (op 3989) 


s_sub_i32 s18, s0, s1 // (op 3989) 


s_cmp_eq_i32 s19, 0 // (op 3989) 
s_cselect_b32 s0, s65, s18 // (op 3989) 



v_lshl_add_u32 v5, s0, 4, v12 // (op 3989) 


v_mul_lo_u32 v9, v5, s22 // (op 3989) 

v_add_lshl_u32 v5, v8, v9, 2 // (op 3989) 


s_mul_i32 s1, s2, s5 // (op 3802) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 3802) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 3802) least significant
s_mul_hi_u32 s19, s2, s4 // (op 3802) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3802) most significant: sum
s_add_u32 s1, s1, s19 // (op 3802) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3802) 

s_mul_i32 s1, s22, s5 // (op 3991) most significant: low of low * high
s_mul_i32 s51, s23, s4 // (op 3991) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 3991) least significant
s_mul_hi_u32 s52, s22, s4 // (op 3991) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 3991) most significant: sum
s_add_u32 s1, s1, s52 // (op 3991) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 3991) 

s_mov_b32 s56, s24 // (op 3802) reg expression
s_mov_b32 s57, s25 // (op 3802) reg expression
s_mov_b32 s58, s18 // (op 3802) reg expression
s_mov_b32 s59, 131072 // (op 3802) call()

s_mov_b32 s60, s26 // (op 3991) reg expression

s_mov_b32 s61, s27 // (op 3991) reg expression
s_mov_b32 s62, s52 // (op 3991) reg expression
s_mov_b32 s63, 131072 // (op 3991) call()


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
v_accvgpr_write a8, 0.00000 // (op 471) call()
v_accvgpr_write a9, 0.00000 // (op 471) call()
v_accvgpr_write a10, 0.00000 // (op 471) call()
v_accvgpr_write a11, 0.00000 // (op 471) call()
v_accvgpr_write a12, 0.00000 // (op 633) call()
v_accvgpr_write a13, 0.00000 // (op 633) call()
v_accvgpr_write a14, 0.00000 // (op 633) call()
v_accvgpr_write a15, 0.00000 // (op 633) call()
v_accvgpr_write a16, 0.00000 // (op 795) call()
v_accvgpr_write a17, 0.00000 // (op 795) call()
v_accvgpr_write a18, 0.00000 // (op 795) call()
v_accvgpr_write a19, 0.00000 // (op 795) call()
v_accvgpr_write a20, 0.00000 // (op 957) call()
v_accvgpr_write a21, 0.00000 // (op 957) call()
v_accvgpr_write a22, 0.00000 // (op 957) call()
v_accvgpr_write a23, 0.00000 // (op 957) call()
v_accvgpr_write a24, 0.00000 // (op 1119) call()
v_accvgpr_write a25, 0.00000 // (op 1119) call()
v_accvgpr_write a26, 0.00000 // (op 1119) call()
v_accvgpr_write a27, 0.00000 // (op 1119) call()
v_accvgpr_write a28, 0.00000 // (op 1281) call()
v_accvgpr_write a29, 0.00000 // (op 1281) call()
v_accvgpr_write a30, 0.00000 // (op 1281) call()
v_accvgpr_write a31, 0.00000 // (op 1281) call()
s_mul_hi_i32 s0, s16, s8 // (op 3567) 
v_and_b32 v12, -16, v1 // (op 3567) 
s_add_i32 s1, s0, s16 // (op 3567) 

v_sub_u32 v13, v1, v12 // (op 3567) 

s_ashr_i32 s0, s1, 31 // (op 3567) 
v_lshlrev_b32 v12, 4, v13 // (op 3567) 

s_cmp_eq_i32 s8, 0 // (op 3567) 
s_mov_b32 s18, scc // (op 3567) 
v_and_b32 v13, 4294967280, v1 // (op 3567) 
s_cmp_eq_i32 s18, 0 // (op 3567) 
s_cselect_b32 s19, 0, -1 // (op 3567) 

s_and_b32 s18, s9, 31 // (op 3567) 
s_lshl_b32 s24, 1, s18 // (op 3567) 
s_add_i32 s25, s24, s19 // (op 3567) 


s_and_b32 s19, s0, s25 // (op 3567) 


s_add_i32 s0, s1, s19 // (op 3567) 


s_ashr_i32 s1, s0, s18 // (op 3567) 


s_lshl_b32 s0, s9, 24 // (op 3567) 
s_ashr_i32 s18, s0, 31 // (op 3567) 

s_xor_b32 s0, s1, s18 // (op 3567) 

s_sub_i32 s1, s0, s18 // (op 3567) 


s_mul_hi_i32 s0, s1, s10 // (op 3567) 
s_add_i32 s18, s0, s1 // (op 3567) 

s_ashr_i32 s0, s18, 31 // (op 3567) 
s_cmp_eq_i32 s10, 0 // (op 3567) 
s_mov_b32 s19, scc // (op 3567) 
s_cmp_eq_i32 s19, 0 // (op 3567) 
s_cselect_b32 s24, 0, -1 // (op 3567) 

s_and_b32 s19, s11, 31 // (op 3567) 
s_lshl_b32 s25, 1, s19 // (op 3567) 
s_add_i32 s26, s25, s24 // (op 3567) 


s_and_b32 s27, s0, s26 // (op 3567) 


s_add_i32 s0, s18, s27 // (op 3567) 


s_ashr_i32 s18, s0, s19 // (op 3567) 


s_lshl_b32 s0, s11, 24 // (op 3567) 
s_ashr_i32 s19, s0, 31 // (op 3567) 

s_xor_b32 s0, s18, s19 // (op 3567) 

s_sub_i32 s18, s0, s19 // (op 3567) 


s_mul_i32 s0, s1, s13 // (op 3567) 

s_sub_u32 s1, s16, s0 // (op 3567) 

s_mul_i32 s0, s18, s13 // (op 3567) 
s_add_u32 s0, s0, s1 // (op 3567) 


s_mul_hi_i32 s1, s16, s7 // (op 3567) 
s_add_i32 s18, s1, s16 // (op 3567) 

s_ashr_i32 s1, s18, 31 // (op 3567) 
s_cmp_eq_i32 s7, 0 // (op 3567) 
s_mov_b32 s19, scc // (op 3567) 
s_cmp_eq_i32 s19, 0 // (op 3567) 
s_cselect_b32 s24, 0, -1 // (op 3567) 

s_and_b32 s19, s14, 31 // (op 3567) 
s_lshl_b32 s25, 1, s19 // (op 3567) 
s_add_i32 s26, s25, s24 // (op 3567) 


s_and_b32 s27, s1, s26 // (op 3567) 


s_add_i32 s1, s18, s27 // (op 3567) 


s_ashr_i32 s18, s1, s19 // (op 3567) 


s_lshl_b32 s1, s14, 24 // (op 3567) 
s_ashr_i32 s19, s1, 31 // (op 3567) 

s_xor_b32 s1, s18, s19 // (op 3567) 

s_sub_i32 s18, s1, s19 // (op 3567) 


s_add_u32 s24, s4, 16 // (op 3567) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 3567) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 3567) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 3567) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 4 // (op 3567) 

s_mul_i32 s1, s13, s24 // (op 3567) 

s_mul_i32 s19, s6, s1 // (op 3567) 

s_mul_i32 s1, s18, s19 // (op 3567) 

s_sub_u32 s19, s16, s1 // (op 3567) 

s_mov_b32 s1, s19 // (op 3567) 
s_mul_hi_i32 s24, s1, s12 // (op 3567) 
s_add_i32 s25, s24, s1 // (op 3567) 


s_ashr_i32 s1, s25, 31 // (op 3567) 
s_cmp_eq_i32 s12, 0 // (op 3567) 
s_mov_b32 s24, scc // (op 3567) 
s_cmp_eq_i32 s24, 0 // (op 3567) 
s_cselect_b32 s26, 0, -1 // (op 3567) 

s_and_b32 s24, s48, 31 // (op 3567) 
s_lshl_b32 s27, 1, s24 // (op 3567) 
s_add_i32 s49, s27, s26 // (op 3567) 


s_and_b32 s26, s1, s49 // (op 3567) 


s_add_i32 s1, s25, s26 // (op 3567) 


s_ashr_i32 s25, s1, s24 // (op 3567) 


s_lshl_b32 s1, s48, 24 // (op 3567) 
s_ashr_i32 s24, s1, 31 // (op 3567) 

s_xor_b32 s1, s25, s24 // (op 3567) 

s_sub_i32 s25, s1, s24 // (op 3567) 


s_mul_i32 s1, s15, s13 // (op 3567) 
s_add_u32 s26, s20, 256 // (op 3567) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 3567) End of Int64 add, unlocking SCC
s_sub_u32 s54, s26, 1 // (op 3567) least significant half; sets scc
s_subb_u32 s55, s27, 0 // (op 3567) most significant half; uses scc

s_ashr_i64 s[26:27], s[54:55], 8 // (op 3567) 

s_sub_i32 s24, s26, s1 // (op 3567) 

s_mul_i32 s26, s25, s24 // (op 3567) 


s_sub_u32 s27, s19, s26 // (op 3567) 


s_add_u32 s19, s27, s1 // (op 3567) 


s_cmp_eq_i32 s18, 0 // (op 3567) 
s_mov_b32 s1, scc // (op 3567) 

s_cmp_eq_i32 s1, 0 // (op 3567) 
s_cselect_b32 s18, s19, s0 // (op 3567) 



v_lshl_add_u32 v14, s18, 8, v12 // (op 3567) 


v_mul_lo_u32 v12, v13, s28 // (op 3567) 

v_add_u32 v13, v14, v12 // (op 3567) 


s_mul_i32 s1, s28, s33 // (op 3569) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 3569) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3569) least significant
s_mul_hi_u32 s19, s28, s32 // (op 3569) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3569) most significant: sum
s_add_u32 s1, s1, s19 // (op 3569) most significant: sum


s_mov_b32 s24, s30 // (op 3569) reg expression
s_mov_b32 s25, s31 // (op 3569) reg expression
s_mov_b32 s26, s0 // (op 3569) reg expression
s_mov_b32 s27, 131072 // (op 3569) call()

s_mul_hi_i32 s0, s16, s8 // (op 3586) 
v_and_b32 v12, -16, v1 // (op 3586) 
s_add_i32 s1, s0, s16 // (op 3586) 

v_sub_u32 v14, v1, v12 // (op 3586) 

s_ashr_i32 s0, s1, 31 // (op 3586) 
v_lshlrev_b32 v12, 4, v14 // (op 3586) 

s_cmp_eq_i32 s8, 0 // (op 3586) 
s_mov_b32 s18, scc // (op 3586) 
v_and_b32 v14, 4294967280, v1 // (op 3586) 
s_cmp_eq_i32 s18, 0 // (op 3586) 
s_cselect_b32 s19, 0, -1 // (op 3586) 

s_and_b32 s18, s9, 31 // (op 3586) 
s_lshl_b32 s49, 1, s18 // (op 3586) 
s_add_i32 s50, s49, s19 // (op 3586) 


s_and_b32 s19, s0, s50 // (op 3586) 


s_add_i32 s0, s1, s19 // (op 3586) 


s_ashr_i32 s1, s0, s18 // (op 3586) 


s_lshl_b32 s0, s9, 24 // (op 3586) 
s_ashr_i32 s18, s0, 31 // (op 3586) 

s_xor_b32 s0, s1, s18 // (op 3586) 

s_sub_i32 s1, s0, s18 // (op 3586) 


s_mul_hi_i32 s0, s1, s10 // (op 3586) 
s_add_i32 s18, s0, s1 // (op 3586) 

s_ashr_i32 s0, s18, 31 // (op 3586) 
s_cmp_eq_i32 s10, 0 // (op 3586) 
s_mov_b32 s19, scc // (op 3586) 
s_cmp_eq_i32 s19, 0 // (op 3586) 
s_cselect_b32 s49, 0, -1 // (op 3586) 

s_and_b32 s19, s11, 31 // (op 3586) 
s_lshl_b32 s50, 1, s19 // (op 3586) 
s_add_i32 s51, s50, s49 // (op 3586) 


s_and_b32 s49, s0, s51 // (op 3586) 


s_add_i32 s0, s18, s49 // (op 3586) 


s_ashr_i32 s18, s0, s19 // (op 3586) 


s_lshl_b32 s0, s11, 24 // (op 3586) 
s_ashr_i32 s19, s0, 31 // (op 3586) 

s_xor_b32 s0, s18, s19 // (op 3586) 

s_sub_i32 s18, s0, s19 // (op 3586) 


s_mul_i32 s0, s1, s13 // (op 3586) 

s_sub_u32 s1, s16, s0 // (op 3586) 

s_mul_i32 s0, s18, s13 // (op 3586) 
s_add_u32 s0, s0, s1 // (op 3586) 


s_mul_hi_i32 s1, s16, s7 // (op 3586) 
s_add_i32 s18, s1, s16 // (op 3586) 

s_ashr_i32 s1, s18, 31 // (op 3586) 
s_cmp_eq_i32 s7, 0 // (op 3586) 
s_mov_b32 s19, scc // (op 3586) 
s_cmp_eq_i32 s19, 0 // (op 3586) 
s_cselect_b32 s49, 0, -1 // (op 3586) 

s_and_b32 s19, s14, 31 // (op 3586) 
s_lshl_b32 s50, 1, s19 // (op 3586) 
s_add_i32 s51, s50, s49 // (op 3586) 


s_and_b32 s49, s1, s51 // (op 3586) 


s_add_i32 s1, s18, s49 // (op 3586) 


s_ashr_i32 s18, s1, s19 // (op 3586) 


s_lshl_b32 s1, s14, 24 // (op 3586) 
s_ashr_i32 s19, s1, 31 // (op 3586) 

s_xor_b32 s1, s18, s19 // (op 3586) 

s_sub_i32 s18, s1, s19 // (op 3586) 


s_add_u32 s54, s4, 16 // (op 3586) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 3586) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3586) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3586) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 3586) 

s_mul_i32 s1, s13, s50 // (op 3586) 

s_mul_i32 s19, s6, s1 // (op 3586) 

s_mul_i32 s1, s18, s19 // (op 3586) 

s_sub_u32 s19, s16, s1 // (op 3586) 

s_mov_b32 s1, s19 // (op 3586) 
s_mul_hi_i32 s49, s1, s12 // (op 3586) 
s_add_i32 s50, s49, s1 // (op 3586) 


s_ashr_i32 s1, s50, 31 // (op 3586) 
s_cmp_eq_i32 s12, 0 // (op 3586) 
s_mov_b32 s49, scc // (op 3586) 
s_cmp_eq_i32 s49, 0 // (op 3586) 
s_cselect_b32 s51, 0, -1 // (op 3586) 

s_and_b32 s49, s48, 31 // (op 3586) 
s_lshl_b32 s52, 1, s49 // (op 3586) 
s_add_i32 s53, s52, s51 // (op 3586) 


s_and_b32 s51, s1, s53 // (op 3586) 


s_add_i32 s1, s50, s51 // (op 3586) 


s_ashr_i32 s50, s1, s49 // (op 3586) 


s_lshl_b32 s1, s48, 24 // (op 3586) 
s_ashr_i32 s49, s1, 31 // (op 3586) 

s_xor_b32 s1, s50, s49 // (op 3586) 

s_sub_i32 s50, s1, s49 // (op 3586) 


s_mul_i32 s1, s15, s13 // (op 3586) 
s_add_u32 s54, s20, 256 // (op 3586) Start of Int64 add, locking SCC
s_addc_u32 s55, s21, 0 // (op 3586) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3586) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3586) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 8 // (op 3586) 

s_sub_i32 s49, s54, s1 // (op 3586) 

s_mul_i32 s51, s50, s49 // (op 3586) 


s_sub_u32 s49, s19, s51 // (op 3586) 


s_add_u32 s19, s49, s1 // (op 3586) 


s_cmp_eq_i32 s18, 0 // (op 3586) 
s_mov_b32 s1, scc // (op 3586) 

s_cmp_eq_i32 s1, 0 // (op 3586) 
s_cselect_b32 s18, s19, s0 // (op 3586) 



v_lshl_add_u32 v15, s18, 8, v12 // (op 3586) 


v_mul_lo_u32 v12, v14, s28 // (op 3586) 

v_add_u32 v14, v15, v12 // (op 3586) 


s_mul_i32 s1, s28, s33 // (op 3588) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 3588) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3588) least significant
s_mul_hi_u32 s19, s28, s32 // (op 3588) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3588) most significant: sum
s_add_u32 s1, s1, s19 // (op 3588) most significant: sum


s_mov_b32 s24, s30 // (op 3588) reg expression
s_mov_b32 s25, s31 // (op 3588) reg expression
s_mov_b32 s26, s0 // (op 3588) reg expression
s_mov_b32 s27, 131072 // (op 3588) call()

s_mul_hi_i32 s0, s16, s8 // (op 3631) 
v_lshrrev_b32 v12, 3, v1 // (op 3631) 
s_add_i32 s1, s0, s16 // (op 3631) 

v_and_b32 v15, -8, v1 // (op 3631) 
s_ashr_i32 s0, s1, 31 // (op 3631) 
v_sub_u32 v16, v1, v15 // (op 3631) 

s_cmp_eq_i32 s8, 0 // (op 3631) 
s_mov_b32 s18, scc // (op 3631) 
v_lshlrev_b32 v15, 4, v16 // (op 3631) 

s_cmp_eq_i32 s18, 0 // (op 3631) 
s_cselect_b32 s19, 0, -1 // (op 3631) 

v_mov_b32 v17, 0 // (op 3631) convert
v_mov_b32 v16, v15 // (op 3631) convert

s_and_b32 s18, s9, 31 // (op 3631) 
s_lshl_b32 s49, 1, s18 // (op 3631) 
s_add_i32 s50, s49, s19 // (op 3631) 


s_and_b32 s19, s0, s50 // (op 3631) 


s_add_i32 s0, s1, s19 // (op 3631) 


s_ashr_i32 s1, s0, s18 // (op 3631) 


s_lshl_b32 s0, s9, 24 // (op 3631) 
s_ashr_i32 s18, s0, 31 // (op 3631) 

s_xor_b32 s0, s1, s18 // (op 3631) 

s_sub_i32 s1, s0, s18 // (op 3631) 


s_mul_hi_i32 s0, s1, s10 // (op 3631) 
s_add_i32 s18, s0, s1 // (op 3631) 

s_ashr_i32 s0, s18, 31 // (op 3631) 
s_cmp_eq_i32 s10, 0 // (op 3631) 
s_mov_b32 s19, scc // (op 3631) 
s_cmp_eq_i32 s19, 0 // (op 3631) 
s_cselect_b32 s49, 0, -1 // (op 3631) 

s_and_b32 s19, s11, 31 // (op 3631) 
s_lshl_b32 s50, 1, s19 // (op 3631) 
s_add_i32 s51, s50, s49 // (op 3631) 


s_and_b32 s49, s0, s51 // (op 3631) 


s_add_i32 s0, s18, s49 // (op 3631) 


s_ashr_i32 s18, s0, s19 // (op 3631) 


s_lshl_b32 s0, s11, 24 // (op 3631) 
s_ashr_i32 s19, s0, 31 // (op 3631) 

s_xor_b32 s0, s18, s19 // (op 3631) 

s_sub_i32 s18, s0, s19 // (op 3631) 


s_add_u32 s54, s4, 16 // (op 3631) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 3631) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3631) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3631) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 3631) 

s_mul_i32 s0, s18, s50 // (op 3631) 

s_sub_i32 s49, s1, s0 // (op 3631) 


s_mul_hi_i32 s0, s16, s7 // (op 3631) 
s_add_i32 s1, s0, s16 // (op 3631) 

s_ashr_i32 s0, s1, 31 // (op 3631) 
s_cmp_eq_i32 s7, 0 // (op 3631) 
s_mov_b32 s18, scc // (op 3631) 
s_cmp_eq_i32 s18, 0 // (op 3631) 
s_cselect_b32 s19, 0, -1 // (op 3631) 

s_and_b32 s18, s14, 31 // (op 3631) 
s_lshl_b32 s52, 1, s18 // (op 3631) 
s_add_i32 s53, s52, s19 // (op 3631) 


s_and_b32 s19, s0, s53 // (op 3631) 


s_add_i32 s0, s1, s19 // (op 3631) 


s_ashr_i32 s1, s0, s18 // (op 3631) 


s_lshl_b32 s0, s14, 24 // (op 3631) 
s_ashr_i32 s18, s0, 31 // (op 3631) 

s_xor_b32 s0, s1, s18 // (op 3631) 

s_sub_i32 s1, s0, s18 // (op 3631) 


s_mul_i32 s0, s13, s50 // (op 3631) 

s_mul_i32 s18, s6, s0 // (op 3631) 

s_mul_i32 s0, s1, s18 // (op 3631) 

s_sub_u32 s18, s16, s0 // (op 3631) 

s_mul_hi_i32 s0, s18, s12 // (op 3631) 
s_add_i32 s19, s0, s18 // (op 3631) 


s_ashr_i32 s0, s19, 31 // (op 3631) 
s_cmp_eq_i32 s12, 0 // (op 3631) 
s_mov_b32 s18, scc // (op 3631) 
s_cmp_eq_i32 s18, 0 // (op 3631) 
s_cselect_b32 s50, 0, -1 // (op 3631) 

s_and_b32 s18, s48, 31 // (op 3631) 
s_lshl_b32 s51, 1, s18 // (op 3631) 
s_add_i32 s52, s51, s50 // (op 3631) 


s_and_b32 s50, s0, s52 // (op 3631) 


s_add_i32 s0, s19, s50 // (op 3631) 


s_ashr_i32 s19, s0, s18 // (op 3631) 


s_lshl_b32 s0, s48, 24 // (op 3631) 
s_ashr_i32 s18, s0, 31 // (op 3631) 

s_xor_b32 s0, s19, s18 // (op 3631) 

s_sub_i32 s19, s0, s18 // (op 3631) 


s_cmp_eq_i32 s1, 0 // (op 3631) 
s_mov_b32 s0, scc // (op 3631) 

s_cmp_eq_i32 s0, 0 // (op 3631) 
s_cselect_b32 s1, s19, s49 // (op 3631) 



v_lshl_add_u32 v15, s1, 4, v12 // (op 3631) 


v_mul_lo_u32 v19, v15, s35 // (op 3631) most significant: low of low * high
v_mul_hi_u32 v12, v15, s34 // (op 3631) most significant: high of low * low
v_mul_lo_u32 v18, v15, s34 // (op 3631) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3631) most significant: sum


v_add_co_u32 v20, s[0:1], v16, v18 // (op 3631) least significant half
v_addc_co_u32 v21, s[0:1], v17, v19, s[0:1] // (op 3631) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 3631) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 3631) 

v_mov_b32 v12, v18 // (op 3631) convert

s_mul_i32 s1, s34, s5 // (op 3633) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 3633) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 3633) least significant
s_mul_hi_u32 s19, s34, s4 // (op 3633) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3633) most significant: sum
s_add_u32 s1, s1, s19 // (op 3633) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3633) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3633) 

s_mov_b32 s52, s36 // (op 3633) reg expression
s_mov_b32 s53, s37 // (op 3633) reg expression
s_mov_b32 s54, s0 // (op 3633) reg expression
s_mov_b32 s55, 131072 // (op 3633) call()

s_mul_hi_i32 s0, s16, s8 // (op 3650) 
v_lshrrev_b32 v15, 3, v1 // (op 3650) 
s_add_i32 s1, s0, s16 // (op 3650) 

v_and_b32 v16, -8, v1 // (op 3650) 
s_ashr_i32 s0, s1, 31 // (op 3650) 
v_sub_u32 v17, v1, v16 // (op 3650) 

s_cmp_eq_i32 s8, 0 // (op 3650) 
s_mov_b32 s18, scc // (op 3650) 
v_lshlrev_b32 v16, 4, v17 // (op 3650) 

s_cmp_eq_i32 s18, 0 // (op 3650) 
s_cselect_b32 s19, 0, -1 // (op 3650) 

v_mov_b32 v19, 0 // (op 3650) convert
v_mov_b32 v18, v16 // (op 3650) convert

s_and_b32 s18, s9, 31 // (op 3650) 
s_lshl_b32 s49, 1, s18 // (op 3650) 
s_add_i32 s50, s49, s19 // (op 3650) 


s_and_b32 s19, s0, s50 // (op 3650) 


s_add_i32 s0, s1, s19 // (op 3650) 


s_ashr_i32 s1, s0, s18 // (op 3650) 


s_lshl_b32 s0, s9, 24 // (op 3650) 
s_ashr_i32 s18, s0, 31 // (op 3650) 

s_xor_b32 s0, s1, s18 // (op 3650) 

s_sub_i32 s1, s0, s18 // (op 3650) 


s_mul_hi_i32 s0, s1, s10 // (op 3650) 
s_add_i32 s18, s0, s1 // (op 3650) 

s_ashr_i32 s0, s18, 31 // (op 3650) 
s_cmp_eq_i32 s10, 0 // (op 3650) 
s_mov_b32 s19, scc // (op 3650) 
s_cmp_eq_i32 s19, 0 // (op 3650) 
s_cselect_b32 s49, 0, -1 // (op 3650) 

s_and_b32 s19, s11, 31 // (op 3650) 
s_lshl_b32 s50, 1, s19 // (op 3650) 
s_add_i32 s51, s50, s49 // (op 3650) 


s_and_b32 s49, s0, s51 // (op 3650) 


s_add_i32 s0, s18, s49 // (op 3650) 


s_ashr_i32 s18, s0, s19 // (op 3650) 


s_lshl_b32 s0, s11, 24 // (op 3650) 
s_ashr_i32 s19, s0, 31 // (op 3650) 

s_xor_b32 s0, s18, s19 // (op 3650) 

s_sub_i32 s18, s0, s19 // (op 3650) 


s_add_u32 s50, s4, 16 // (op 3650) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3650) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3650) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3650) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 3650) 

s_mul_i32 s0, s18, s50 // (op 3650) 

s_sub_i32 s49, s1, s0 // (op 3650) 


s_mul_hi_i32 s0, s16, s7 // (op 3650) 
s_add_i32 s1, s0, s16 // (op 3650) 

s_ashr_i32 s0, s1, 31 // (op 3650) 
s_cmp_eq_i32 s7, 0 // (op 3650) 
s_mov_b32 s18, scc // (op 3650) 
s_cmp_eq_i32 s18, 0 // (op 3650) 
s_cselect_b32 s19, 0, -1 // (op 3650) 

s_and_b32 s18, s14, 31 // (op 3650) 
s_lshl_b32 s64, 1, s18 // (op 3650) 
s_add_i32 s65, s64, s19 // (op 3650) 


s_and_b32 s19, s0, s65 // (op 3650) 


s_add_i32 s0, s1, s19 // (op 3650) 


s_ashr_i32 s1, s0, s18 // (op 3650) 


s_lshl_b32 s0, s14, 24 // (op 3650) 
s_ashr_i32 s18, s0, 31 // (op 3650) 

s_xor_b32 s0, s1, s18 // (op 3650) 

s_sub_i32 s1, s0, s18 // (op 3650) 


s_mul_i32 s0, s13, s50 // (op 3650) 

s_mul_i32 s18, s6, s0 // (op 3650) 

s_mul_i32 s0, s1, s18 // (op 3650) 

s_sub_u32 s18, s16, s0 // (op 3650) 

s_mul_hi_i32 s0, s18, s12 // (op 3650) 
s_add_i32 s19, s0, s18 // (op 3650) 


s_ashr_i32 s0, s19, 31 // (op 3650) 
s_cmp_eq_i32 s12, 0 // (op 3650) 
s_mov_b32 s18, scc // (op 3650) 
s_cmp_eq_i32 s18, 0 // (op 3650) 
s_cselect_b32 s50, 0, -1 // (op 3650) 

s_and_b32 s18, s48, 31 // (op 3650) 
s_lshl_b32 s51, 1, s18 // (op 3650) 
s_add_i32 s64, s51, s50 // (op 3650) 


s_and_b32 s50, s0, s64 // (op 3650) 


s_add_i32 s0, s19, s50 // (op 3650) 


s_ashr_i32 s19, s0, s18 // (op 3650) 


s_lshl_b32 s0, s48, 24 // (op 3650) 
s_ashr_i32 s18, s0, 31 // (op 3650) 

s_xor_b32 s0, s19, s18 // (op 3650) 

s_sub_i32 s19, s0, s18 // (op 3650) 


s_cmp_eq_i32 s1, 0 // (op 3650) 
s_mov_b32 s0, scc // (op 3650) 

s_cmp_eq_i32 s0, 0 // (op 3650) 
s_cselect_b32 s1, s19, s49 // (op 3650) 



v_lshl_add_u32 v16, s1, 4, v15 // (op 3650) 


v_mul_lo_u32 v21, v16, s35 // (op 3650) most significant: low of low * high
v_mul_hi_u32 v15, v16, s34 // (op 3650) most significant: high of low * low
v_mul_lo_u32 v20, v16, s34 // (op 3650) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 3650) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 3650) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 3650) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 3650) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 3650) 

v_mov_b32 v15, v18 // (op 3650) convert

s_mul_i32 s1, s34, s5 // (op 3652) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 3652) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 3652) least significant
s_mul_hi_u32 s19, s34, s4 // (op 3652) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3652) most significant: sum
s_add_u32 s1, s1, s19 // (op 3652) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3652) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3652) 

s_mov_b32 s52, s36 // (op 3652) reg expression
s_mov_b32 s53, s37 // (op 3652) reg expression
s_mov_b32 s54, s0 // (op 3652) reg expression
s_mov_b32 s55, 131072 // (op 3652) call()

s_mul_hi_i32 s0, s16, s8 // (op 3695) 
v_and_b32 v16, -32, v1 // (op 3695) 
s_add_i32 s1, s0, s16 // (op 3695) 

v_sub_u32 v17, v1, v16 // (op 3695) 

s_ashr_i32 s0, s1, 31 // (op 3695) 
v_lshlrev_b32 v16, 3, v17 // (op 3695) 

s_cmp_eq_i32 s8, 0 // (op 3695) 
s_mov_b32 s18, scc // (op 3695) 
v_lshrrev_b32 v17, 5, v1 // (op 3695) 
s_cmp_eq_i32 s18, 0 // (op 3695) 
s_cselect_b32 s19, 0, -1 // (op 3695) 

s_and_b32 s18, s9, 31 // (op 3695) 
s_lshl_b32 s49, 1, s18 // (op 3695) 
s_add_i32 s50, s49, s19 // (op 3695) 


s_and_b32 s19, s0, s50 // (op 3695) 


s_add_i32 s0, s1, s19 // (op 3695) 


s_ashr_i32 s1, s0, s18 // (op 3695) 


s_lshl_b32 s0, s9, 24 // (op 3695) 
s_ashr_i32 s18, s0, 31 // (op 3695) 

s_xor_b32 s0, s1, s18 // (op 3695) 

s_sub_i32 s1, s0, s18 // (op 3695) 


s_mul_hi_i32 s0, s1, s10 // (op 3695) 
s_add_i32 s18, s0, s1 // (op 3695) 

s_ashr_i32 s0, s18, 31 // (op 3695) 
s_cmp_eq_i32 s10, 0 // (op 3695) 
s_mov_b32 s19, scc // (op 3695) 
s_cmp_eq_i32 s19, 0 // (op 3695) 
s_cselect_b32 s49, 0, -1 // (op 3695) 

s_and_b32 s19, s11, 31 // (op 3695) 
s_lshl_b32 s50, 1, s19 // (op 3695) 
s_add_i32 s51, s50, s49 // (op 3695) 


s_and_b32 s49, s0, s51 // (op 3695) 


s_add_i32 s0, s18, s49 // (op 3695) 


s_ashr_i32 s18, s0, s19 // (op 3695) 


s_lshl_b32 s0, s11, 24 // (op 3695) 
s_ashr_i32 s19, s0, 31 // (op 3695) 

s_xor_b32 s0, s18, s19 // (op 3695) 

s_sub_i32 s18, s0, s19 // (op 3695) 


s_mul_i32 s0, s1, s13 // (op 3695) 

s_sub_u32 s1, s16, s0 // (op 3695) 

s_mul_i32 s0, s18, s13 // (op 3695) 
s_add_u32 s0, s0, s1 // (op 3695) 


s_mul_hi_i32 s1, s16, s7 // (op 3695) 
s_add_i32 s18, s1, s16 // (op 3695) 

s_ashr_i32 s1, s18, 31 // (op 3695) 
s_cmp_eq_i32 s7, 0 // (op 3695) 
s_mov_b32 s19, scc // (op 3695) 
s_cmp_eq_i32 s19, 0 // (op 3695) 
s_cselect_b32 s49, 0, -1 // (op 3695) 

s_and_b32 s19, s14, 31 // (op 3695) 
s_lshl_b32 s50, 1, s19 // (op 3695) 
s_add_i32 s51, s50, s49 // (op 3695) 


s_and_b32 s49, s1, s51 // (op 3695) 


s_add_i32 s1, s18, s49 // (op 3695) 


s_ashr_i32 s18, s1, s19 // (op 3695) 


s_lshl_b32 s1, s14, 24 // (op 3695) 
s_ashr_i32 s19, s1, 31 // (op 3695) 

s_xor_b32 s1, s18, s19 // (op 3695) 

s_sub_i32 s18, s1, s19 // (op 3695) 


s_add_u32 s50, s4, 16 // (op 3695) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3695) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3695) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3695) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 3695) 

s_mul_i32 s1, s13, s50 // (op 3695) 

s_mul_i32 s19, s6, s1 // (op 3695) 

s_mul_i32 s1, s18, s19 // (op 3695) 

s_sub_u32 s19, s16, s1 // (op 3695) 

s_mov_b32 s1, s19 // (op 3695) 
s_mul_hi_i32 s49, s1, s12 // (op 3695) 
s_add_i32 s50, s49, s1 // (op 3695) 


s_ashr_i32 s1, s50, 31 // (op 3695) 
s_cmp_eq_i32 s12, 0 // (op 3695) 
s_mov_b32 s49, scc // (op 3695) 
s_cmp_eq_i32 s49, 0 // (op 3695) 
s_cselect_b32 s51, 0, -1 // (op 3695) 

s_and_b32 s49, s48, 31 // (op 3695) 
s_lshl_b32 s64, 1, s49 // (op 3695) 
s_add_i32 s65, s64, s51 // (op 3695) 


s_and_b32 s51, s1, s65 // (op 3695) 


s_add_i32 s1, s50, s51 // (op 3695) 


s_ashr_i32 s50, s1, s49 // (op 3695) 


s_lshl_b32 s1, s48, 24 // (op 3695) 
s_ashr_i32 s49, s1, 31 // (op 3695) 

s_xor_b32 s1, s50, s49 // (op 3695) 

s_sub_i32 s50, s1, s49 // (op 3695) 


s_mul_i32 s1, s15, s13 // (op 3695) 
s_add_u32 s64, s20, 256 // (op 3695) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 3695) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 3695) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 3695) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 8 // (op 3695) 

s_sub_i32 s49, s64, s1 // (op 3695) 

s_mul_i32 s51, s50, s49 // (op 3695) 


s_sub_u32 s49, s19, s51 // (op 3695) 


s_add_u32 s19, s49, s1 // (op 3695) 


s_cmp_eq_i32 s18, 0 // (op 3695) 
s_mov_b32 s1, scc // (op 3695) 

s_cmp_eq_i32 s1, 0 // (op 3695) 
s_cselect_b32 s18, s19, s0 // (op 3695) 



v_lshl_add_u32 v18, s18, 8, v16 // (op 3695) 


v_mul_lo_u32 v16, v17, s38 // (op 3695) 

v_add_u32 v17, v18, v16 // (op 3695) 


s_mul_i32 s1, s38, s43 // (op 3697) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3697) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3697) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3697) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3697) most significant: sum
s_add_u32 s1, s1, s19 // (op 3697) most significant: sum


s_mov_b32 s64, s40 // (op 3697) reg expression
s_mov_b32 s65, s41 // (op 3697) reg expression
s_mov_b32 s66, s0 // (op 3697) reg expression
s_mov_b32 s67, 131072 // (op 3697) call()

s_mul_hi_i32 s0, s16, s8 // (op 3714) 
v_and_b32 v16, -32, v1 // (op 3714) 
s_add_i32 s1, s0, s16 // (op 3714) 

v_sub_u32 v18, v1, v16 // (op 3714) 

s_ashr_i32 s0, s1, 31 // (op 3714) 
v_lshlrev_b32 v16, 3, v18 // (op 3714) 

s_cmp_eq_i32 s8, 0 // (op 3714) 
s_mov_b32 s18, scc // (op 3714) 
v_lshrrev_b32 v18, 5, v1 // (op 3714) 
s_cmp_eq_i32 s18, 0 // (op 3714) 
s_cselect_b32 s19, 0, -1 // (op 3714) 

s_and_b32 s18, s9, 31 // (op 3714) 
s_lshl_b32 s49, 1, s18 // (op 3714) 
s_add_i32 s50, s49, s19 // (op 3714) 


s_and_b32 s19, s0, s50 // (op 3714) 


s_add_i32 s0, s1, s19 // (op 3714) 


s_ashr_i32 s1, s0, s18 // (op 3714) 


s_lshl_b32 s0, s9, 24 // (op 3714) 
s_ashr_i32 s18, s0, 31 // (op 3714) 

s_xor_b32 s0, s1, s18 // (op 3714) 

s_sub_i32 s1, s0, s18 // (op 3714) 


s_mul_hi_i32 s0, s1, s10 // (op 3714) 
s_add_i32 s18, s0, s1 // (op 3714) 

s_ashr_i32 s0, s18, 31 // (op 3714) 
s_cmp_eq_i32 s10, 0 // (op 3714) 
s_mov_b32 s19, scc // (op 3714) 
s_cmp_eq_i32 s19, 0 // (op 3714) 
s_cselect_b32 s49, 0, -1 // (op 3714) 

s_and_b32 s19, s11, 31 // (op 3714) 
s_lshl_b32 s50, 1, s19 // (op 3714) 
s_add_i32 s51, s50, s49 // (op 3714) 


s_and_b32 s49, s0, s51 // (op 3714) 


s_add_i32 s0, s18, s49 // (op 3714) 


s_ashr_i32 s18, s0, s19 // (op 3714) 


s_lshl_b32 s0, s11, 24 // (op 3714) 
s_ashr_i32 s19, s0, 31 // (op 3714) 

s_xor_b32 s0, s18, s19 // (op 3714) 

s_sub_i32 s18, s0, s19 // (op 3714) 


s_mul_i32 s0, s1, s13 // (op 3714) 

s_sub_u32 s1, s16, s0 // (op 3714) 

s_mul_i32 s0, s18, s13 // (op 3714) 
s_add_u32 s0, s0, s1 // (op 3714) 


s_mul_hi_i32 s1, s16, s7 // (op 3714) 
s_add_i32 s18, s1, s16 // (op 3714) 

s_ashr_i32 s1, s18, 31 // (op 3714) 
s_cmp_eq_i32 s7, 0 // (op 3714) 
s_mov_b32 s19, scc // (op 3714) 
s_cmp_eq_i32 s19, 0 // (op 3714) 
s_cselect_b32 s49, 0, -1 // (op 3714) 

s_and_b32 s19, s14, 31 // (op 3714) 
s_lshl_b32 s50, 1, s19 // (op 3714) 
s_add_i32 s51, s50, s49 // (op 3714) 


s_and_b32 s49, s1, s51 // (op 3714) 


s_add_i32 s1, s18, s49 // (op 3714) 


s_ashr_i32 s18, s1, s19 // (op 3714) 


s_lshl_b32 s1, s14, 24 // (op 3714) 
s_ashr_i32 s19, s1, 31 // (op 3714) 

s_xor_b32 s1, s18, s19 // (op 3714) 

s_sub_i32 s18, s1, s19 // (op 3714) 


s_add_u32 s50, s4, 16 // (op 3714) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3714) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3714) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3714) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 3714) 

s_mul_i32 s1, s13, s50 // (op 3714) 

s_mul_i32 s19, s6, s1 // (op 3714) 

s_mul_i32 s1, s18, s19 // (op 3714) 

s_sub_u32 s19, s16, s1 // (op 3714) 

s_mov_b32 s1, s19 // (op 3714) 
s_mul_hi_i32 s49, s1, s12 // (op 3714) 
s_add_i32 s50, s49, s1 // (op 3714) 


s_ashr_i32 s1, s50, 31 // (op 3714) 
s_cmp_eq_i32 s12, 0 // (op 3714) 
s_mov_b32 s49, scc // (op 3714) 
s_cmp_eq_i32 s49, 0 // (op 3714) 
s_cselect_b32 s51, 0, -1 // (op 3714) 

s_and_b32 s49, s48, 31 // (op 3714) 
s_lshl_b32 s68, 1, s49 // (op 3714) 
s_add_i32 s69, s68, s51 // (op 3714) 


s_and_b32 s51, s1, s69 // (op 3714) 


s_add_i32 s1, s50, s51 // (op 3714) 


s_ashr_i32 s50, s1, s49 // (op 3714) 


s_lshl_b32 s1, s48, 24 // (op 3714) 
s_ashr_i32 s49, s1, 31 // (op 3714) 

s_xor_b32 s1, s50, s49 // (op 3714) 

s_sub_i32 s50, s1, s49 // (op 3714) 


s_mul_i32 s1, s15, s13 // (op 3714) 
s_add_u32 s68, s20, 256 // (op 3714) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 3714) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 3714) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 3714) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 8 // (op 3714) 

s_sub_i32 s49, s68, s1 // (op 3714) 

s_mul_i32 s51, s50, s49 // (op 3714) 


s_sub_u32 s49, s19, s51 // (op 3714) 


s_add_u32 s19, s49, s1 // (op 3714) 


s_cmp_eq_i32 s18, 0 // (op 3714) 
s_mov_b32 s1, scc // (op 3714) 

s_cmp_eq_i32 s1, 0 // (op 3714) 
s_cselect_b32 s18, s19, s0 // (op 3714) 



v_lshl_add_u32 v19, s18, 8, v16 // (op 3714) 


v_mul_lo_u32 v16, v18, s38 // (op 3714) 

v_add_u32 v18, v19, v16 // (op 3714) 


s_mul_i32 s1, s38, s43 // (op 3716) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3716) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3716) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3716) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3716) most significant: sum
s_add_u32 s1, s1, s19 // (op 3716) most significant: sum


s_mov_b32 s64, s40 // (op 3716) reg expression
s_mov_b32 s65, s41 // (op 3716) reg expression
s_mov_b32 s66, s0 // (op 3716) reg expression
s_mov_b32 s67, 131072 // (op 3716) call()

s_mul_hi_i32 s0, s16, s8 // (op 3759) 
v_and_b32 v16, 15, v1 // (op 3759) 
s_add_i32 s1, s0, s16 // (op 3759) 

v_and_b32 v19, 63, v1 // (op 3759) 
s_ashr_i32 s0, s1, 31 // (op 3759) 
v_lshrrev_b32 v20, 4, v19 // (op 3759) 

s_cmp_eq_i32 s8, 0 // (op 3759) 
s_mov_b32 s18, scc // (op 3759) 
s_cmp_eq_i32 s18, 0 // (op 3759) 
s_cselect_b32 s19, 0, -1 // (op 3759) 

s_and_b32 s18, s9, 31 // (op 3759) 
s_lshl_b32 s49, 1, s18 // (op 3759) 
s_add_i32 s50, s49, s19 // (op 3759) 


s_and_b32 s19, s0, s50 // (op 3759) 


s_add_i32 s0, s1, s19 // (op 3759) 


s_ashr_i32 s1, s0, s18 // (op 3759) 


s_lshl_b32 s0, s9, 24 // (op 3759) 
s_ashr_i32 s18, s0, 31 // (op 3759) 

s_xor_b32 s0, s1, s18 // (op 3759) 

s_sub_i32 s1, s0, s18 // (op 3759) 


s_mul_hi_i32 s0, s1, s10 // (op 3759) 
s_add_i32 s18, s0, s1 // (op 3759) 

s_ashr_i32 s0, s18, 31 // (op 3759) 
s_cmp_eq_i32 s10, 0 // (op 3759) 
s_mov_b32 s19, scc // (op 3759) 
s_cmp_eq_i32 s19, 0 // (op 3759) 
s_cselect_b32 s49, 0, -1 // (op 3759) 

s_and_b32 s19, s11, 31 // (op 3759) 
s_lshl_b32 s50, 1, s19 // (op 3759) 
s_add_i32 s51, s50, s49 // (op 3759) 


s_and_b32 s49, s0, s51 // (op 3759) 


s_add_i32 s0, s18, s49 // (op 3759) 


s_ashr_i32 s18, s0, s19 // (op 3759) 


s_lshl_b32 s0, s11, 24 // (op 3759) 
s_ashr_i32 s19, s0, 31 // (op 3759) 

s_xor_b32 s0, s18, s19 // (op 3759) 

s_sub_i32 s18, s0, s19 // (op 3759) 


s_add_u32 s50, s4, 16 // (op 3759) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3759) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3759) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3759) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 3759) 

s_mul_i32 s0, s18, s50 // (op 3759) 

s_sub_i32 s49, s1, s0 // (op 3759) 


s_mul_hi_i32 s0, s16, s7 // (op 3759) 
s_add_i32 s1, s0, s16 // (op 3759) 

s_ashr_i32 s0, s1, 31 // (op 3759) 
s_cmp_eq_i32 s7, 0 // (op 3759) 
s_mov_b32 s18, scc // (op 3759) 
s_cmp_eq_i32 s18, 0 // (op 3759) 
s_cselect_b32 s19, 0, -1 // (op 3759) 

s_and_b32 s18, s14, 31 // (op 3759) 
s_lshl_b32 s68, 1, s18 // (op 3759) 
s_add_i32 s69, s68, s19 // (op 3759) 


s_and_b32 s19, s0, s69 // (op 3759) 


s_add_i32 s0, s1, s19 // (op 3759) 


s_ashr_i32 s1, s0, s18 // (op 3759) 


s_lshl_b32 s0, s14, 24 // (op 3759) 
s_ashr_i32 s18, s0, 31 // (op 3759) 

s_xor_b32 s0, s1, s18 // (op 3759) 

s_sub_i32 s1, s0, s18 // (op 3759) 


s_mul_i32 s0, s13, s50 // (op 3759) 

s_mul_i32 s18, s6, s0 // (op 3759) 

s_mul_i32 s0, s1, s18 // (op 3759) 

s_sub_u32 s18, s16, s0 // (op 3759) 

s_mul_hi_i32 s0, s18, s12 // (op 3759) 
s_add_i32 s19, s0, s18 // (op 3759) 


s_ashr_i32 s0, s19, 31 // (op 3759) 
s_cmp_eq_i32 s12, 0 // (op 3759) 
s_mov_b32 s18, scc // (op 3759) 
s_cmp_eq_i32 s18, 0 // (op 3759) 
s_cselect_b32 s50, 0, -1 // (op 3759) 

s_and_b32 s18, s48, 31 // (op 3759) 
s_lshl_b32 s51, 1, s18 // (op 3759) 
s_add_i32 s68, s51, s50 // (op 3759) 


s_and_b32 s50, s0, s68 // (op 3759) 


s_add_i32 s0, s19, s50 // (op 3759) 


s_ashr_i32 s19, s0, s18 // (op 3759) 


s_lshl_b32 s0, s48, 24 // (op 3759) 
s_ashr_i32 s18, s0, 31 // (op 3759) 

s_xor_b32 s0, s19, s18 // (op 3759) 

s_sub_i32 s19, s0, s18 // (op 3759) 


s_cmp_eq_i32 s1, 0 // (op 3759) 
s_mov_b32 s0, scc // (op 3759) 

s_cmp_eq_i32 s0, 0 // (op 3759) 
s_cselect_b32 s1, s19, s49 // (op 3759) 



v_lshl_add_u32 v19, s1, 4, v16 // (op 3759) 


v_mul_lo_u32 v16, v19, s44 // (op 3759) 

v_add_u32 v19, v20, v16 // (op 3759) 


s_mul_i32 s1, s44, s5 // (op 3761) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 3761) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 3761) least significant
s_mul_hi_u32 s19, s44, s4 // (op 3761) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3761) most significant: sum
s_add_u32 s1, s1, s19 // (op 3761) most significant: sum


s_mov_b32 s68, s46 // (op 3761) reg expression
s_mov_b32 s69, s47 // (op 3761) reg expression
s_mov_b32 s70, s0 // (op 3761) reg expression
s_mov_b32 s71, 131072 // (op 3761) call()

s_mul_hi_i32 s0, s16, s8 // (op 3778) 
v_and_b32 v16, 15, v1 // (op 3778) 
s_add_i32 s1, s0, s16 // (op 3778) 

v_and_b32 v20, 63, v1 // (op 3778) 
s_ashr_i32 s0, s1, 31 // (op 3778) 
v_lshrrev_b32 v21, 4, v20 // (op 3778) 

s_cmp_eq_i32 s8, 0 // (op 3778) 
s_mov_b32 s18, scc // (op 3778) 
s_cmp_eq_i32 s18, 0 // (op 3778) 
s_cselect_b32 s19, 0, -1 // (op 3778) 

s_and_b32 s18, s9, 31 // (op 3778) 
s_lshl_b32 s49, 1, s18 // (op 3778) 
s_add_i32 s50, s49, s19 // (op 3778) 


s_and_b32 s19, s0, s50 // (op 3778) 


s_add_i32 s0, s1, s19 // (op 3778) 


s_ashr_i32 s1, s0, s18 // (op 3778) 


s_lshl_b32 s0, s9, 24 // (op 3778) 
s_ashr_i32 s18, s0, 31 // (op 3778) 

s_xor_b32 s0, s1, s18 // (op 3778) 

s_sub_i32 s1, s0, s18 // (op 3778) 


s_mul_hi_i32 s0, s1, s10 // (op 3778) 
s_add_i32 s18, s0, s1 // (op 3778) 

s_ashr_i32 s0, s18, 31 // (op 3778) 
s_cmp_eq_i32 s10, 0 // (op 3778) 
s_mov_b32 s19, scc // (op 3778) 
s_cmp_eq_i32 s19, 0 // (op 3778) 
s_cselect_b32 s49, 0, -1 // (op 3778) 

s_and_b32 s19, s11, 31 // (op 3778) 
s_lshl_b32 s50, 1, s19 // (op 3778) 
s_add_i32 s51, s50, s49 // (op 3778) 


s_and_b32 s49, s0, s51 // (op 3778) 


s_add_i32 s0, s18, s49 // (op 3778) 


s_ashr_i32 s18, s0, s19 // (op 3778) 


s_lshl_b32 s0, s11, 24 // (op 3778) 
s_ashr_i32 s19, s0, 31 // (op 3778) 

s_xor_b32 s0, s18, s19 // (op 3778) 

s_sub_i32 s18, s0, s19 // (op 3778) 


s_add_u32 s50, s4, 16 // (op 3778) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3778) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3778) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3778) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 4 // (op 3778) 

s_mul_i32 s0, s18, s50 // (op 3778) 

s_sub_i32 s49, s1, s0 // (op 3778) 


s_mul_hi_i32 s0, s16, s7 // (op 3778) 
s_add_i32 s1, s0, s16 // (op 3778) 

s_ashr_i32 s0, s1, 31 // (op 3778) 
s_cmp_eq_i32 s7, 0 // (op 3778) 
s_mov_b32 s18, scc // (op 3778) 
s_cmp_eq_i32 s18, 0 // (op 3778) 
s_cselect_b32 s19, 0, -1 // (op 3778) 

s_and_b32 s18, s14, 31 // (op 3778) 
s_lshl_b32 s72, 1, s18 // (op 3778) 
s_add_i32 s73, s72, s19 // (op 3778) 


s_and_b32 s19, s0, s73 // (op 3778) 


s_add_i32 s0, s1, s19 // (op 3778) 


s_ashr_i32 s1, s0, s18 // (op 3778) 


s_lshl_b32 s0, s14, 24 // (op 3778) 
s_ashr_i32 s18, s0, 31 // (op 3778) 

s_xor_b32 s0, s1, s18 // (op 3778) 

s_sub_i32 s1, s0, s18 // (op 3778) 


s_mul_i32 s0, s13, s50 // (op 3778) 

s_mul_i32 s18, s6, s0 // (op 3778) 

s_mul_i32 s0, s1, s18 // (op 3778) 

s_sub_u32 s18, s16, s0 // (op 3778) 

s_mul_hi_i32 s0, s18, s12 // (op 3778) 
s_add_i32 s19, s0, s18 // (op 3778) 


s_ashr_i32 s0, s19, 31 // (op 3778) 
s_cmp_eq_i32 s12, 0 // (op 3778) 
s_mov_b32 s18, scc // (op 3778) 
s_cmp_eq_i32 s18, 0 // (op 3778) 
s_cselect_b32 s50, 0, -1 // (op 3778) 

s_and_b32 s18, s48, 31 // (op 3778) 
s_lshl_b32 s51, 1, s18 // (op 3778) 
s_add_i32 s72, s51, s50 // (op 3778) 


s_and_b32 s50, s0, s72 // (op 3778) 


s_add_i32 s0, s19, s50 // (op 3778) 


s_ashr_i32 s19, s0, s18 // (op 3778) 


s_lshl_b32 s0, s48, 24 // (op 3778) 
s_ashr_i32 s18, s0, 31 // (op 3778) 

s_xor_b32 s0, s19, s18 // (op 3778) 

s_sub_i32 s19, s0, s18 // (op 3778) 


s_cmp_eq_i32 s1, 0 // (op 3778) 
s_mov_b32 s0, scc // (op 3778) 

s_cmp_eq_i32 s0, 0 // (op 3778) 
s_cselect_b32 s1, s19, s49 // (op 3778) 



v_lshl_add_u32 v20, s1, 4, v16 // (op 3778) 


v_mul_lo_u32 v16, v20, s44 // (op 3778) 

v_add_u32 v20, v21, v16 // (op 3778) 


s_mul_i32 s1, s44, s5 // (op 3780) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 3780) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 3780) least significant
s_mul_hi_u32 s19, s44, s4 // (op 3780) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3780) most significant: sum
s_add_u32 s1, s1, s19 // (op 3780) most significant: sum


s_mov_b32 s68, s46 // (op 3780) reg expression
s_mov_b32 s69, s47 // (op 3780) reg expression
s_mov_b32 s70, s0 // (op 3780) reg expression
s_mov_b32 s71, 131072 // (op 3780) call()




buffer_load_ubyte v16, v19, s[68:71], 0 offen offset: 0 // (op 3252) Load value



buffer_load_dwordx2 v[22:23], v12, s[52:55], 0 offen // (op 3256) Load value

v_mov_b64 v[24:25], s[28:29] // (op 3259) convert

v_mov_b32 v21, v13 // (op 3259) reg expression
buffer_load_dwordx4 v[26:29], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[30:33], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[34:37], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[38:41], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[42:45], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[46:49], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[50:53], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[54:57], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[58:61], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[62:65], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[66:69], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[70:73], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[74:77], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[78:81], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[82:85], v21, s[24:27], 0 offen // (op 3259) Load value
v_add_u32 v21, v21, v24 // (op 3259) 
buffer_load_dwordx4 v[86:89], v21, s[24:27], 0 offen // (op 3259) Load value




v_mov_b32 v21, v17 // (op 3262) reg expression

buffer_load_dwordx2 v[24:25], v21, s[64:67], 0 offen // (op 3262) Load value




buffer_load_ubyte v21, v19, s[68:71], 0 offen offset:4 // (op 3266) Load value




buffer_load_dwordx2 v[90:91], v12, s[52:55], 0 offen offset:64 // (op 3270) Load value


v_mov_b64 v[92:93], s[28:29] // (op 3273) convert

s_lshl_b32 s0, s28, 7 // (op 3273) 
v_add_u32 v12, v13, s0 // (op 3273) 

buffer_load_dwordx4 v[94:97], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[98:101], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[102:105], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[106:109], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[110:113], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[114:117], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[118:121], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[122:125], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[126:129], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[130:133], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[134:137], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[138:141], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[142:145], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[146:149], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[150:153], v12, s[24:27], 0 offen // (op 3273) Load value
v_add_u32 v12, v12, v92 // (op 3273) 
buffer_load_dwordx4 v[154:157], v12, s[24:27], 0 offen // (op 3273) Load value





s_lshl_b32 s0, s38, 2 // (op 3276) 
v_add_u32 v19, v17, s0 // (op 3276) 


buffer_load_dwordx2 v[12:13], v19, s[64:67], 0 offen // (op 3276) Load value


v_and_b32 v17, -8, v1 // (op 3903) 
v_sub_u32 v19, v1, v17 // (op 3903) 

v_lshlrev_b32 v17, 4, v19 // (op 3903) 

v_lshrrev_b32 v19, 3, v1 // (op 3903) 
v_lshl_add_u32 v92, v19, 7, v17 // (op 3903) 


v_lshlrev_b32 v17, 2, v92 // (op 3903) 

v_lshrrev_b32 v19, 3, v17 // (op 3903) 




s_waitcnt vmcnt(36)
ds_write_b64 v19, v[22:23] // (op 3278) Store local data 

v_and_b32 v17, -16, v1 // (op 3845) 
v_sub_u32 v19, v1, v17 // (op 3845) 

v_lshlrev_b32 v17, 4, v19 // (op 3845) 

v_lshrrev_b32 v19, 4, v1 // (op 3845) 
v_lshl_add_u32 v92, v19, 12, v17 // (op 3845) 





s_waitcnt vmcnt(35)
ds_write_b128 v92, v[26:29] offset:1024 // (op 3279) Store local data 
s_waitcnt vmcnt(34)
ds_write_b128 v92, v[30:33] offset:1280 // (op 3279) Store local data 
s_waitcnt vmcnt(33)
ds_write_b128 v92, v[34:37] offset:1536 // (op 3279) Store local data 
s_waitcnt vmcnt(32)
ds_write_b128 v92, v[38:41] offset:1792 // (op 3279) Store local data 
s_waitcnt vmcnt(31)
ds_write_b128 v92, v[42:45] offset:2048 // (op 3279) Store local data 
s_waitcnt vmcnt(30)
ds_write_b128 v92, v[46:49] offset:2304 // (op 3279) Store local data 
s_waitcnt vmcnt(29)
ds_write_b128 v92, v[50:53] offset:2560 // (op 3279) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v92, v[54:57] offset:2816 // (op 3279) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v92, v[58:61] offset:3072 // (op 3279) Store local data 
s_waitcnt vmcnt(26)
ds_write_b128 v92, v[62:65] offset:3328 // (op 3279) Store local data 
s_waitcnt vmcnt(25)
ds_write_b128 v92, v[66:69] offset:3584 // (op 3279) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v92, v[70:73] offset:3840 // (op 3279) Store local data 
s_waitcnt vmcnt(23)
ds_write_b128 v92, v[74:77] offset:4096 // (op 3279) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v92, v[78:81] offset:4352 // (op 3279) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v92, v[82:85] offset:4608 // (op 3279) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v92, v[86:89] offset:4864 // (op 3279) Store local data 

v_and_b32 v17, -32, v1 // (op 3956) 
v_sub_u32 v19, v1, v17 // (op 3956) 

v_lshlrev_b32 v17, 3, v19 // (op 3956) 

v_lshrrev_b32 v19, 5, v1 // (op 3956) 
v_lshl_add_u32 v92, v19, 8, v17 // (op 3956) 





s_waitcnt vmcnt(19)
ds_write_b64 v92, v[24:25] offset:33792 // (op 3280) Store local data 

v_and_b32 v17, -16, v1 // (op 3816) 
v_sub_u32 v19, v1, v17 // (op 3816) 

v_lshlrev_b32 v17, 4, v19 // (op 3816) 

v_lshrrev_b32 v19, 4, v1 // (op 3816) 
v_lshl_add_u32 v92, v19, 12, v17 // (op 3816) 


v_and_b32 v17, -8, v1 // (op 3874) 
v_sub_u32 v19, v1, v17 // (op 3874) 

v_lshlrev_b32 v17, 4, v19 // (op 3874) 

v_lshrrev_b32 v19, 3, v1 // (op 3874) 
v_lshl_add_u32 v93, v19, 7, v17 // (op 3874) 


v_lshlrev_b32 v17, 2, v93 // (op 3874) 

v_lshrrev_b32 v19, 3, v17 // (op 3874) 

v_and_b32 v17, -32, v1 // (op 3927) 
v_sub_u32 v93, v1, v17 // (op 3927) 

v_lshlrev_b32 v17, 3, v93 // (op 3927) 

v_lshrrev_b32 v93, 5, v1 // (op 3927) 
v_lshl_add_u32 v158, v93, 8, v17 // (op 3927) 


v_and_b32 v17, -16, v1 // (op 4003) 
v_sub_u32 v93, v1, v17 // (op 4003) 

v_lshlrev_b32 v17, 4, v93 // (op 4003) 

v_lshrrev_b32 v93, 4, v1 // (op 4003) 
v_lshl_add_u32 v159, v93, 12, v17 // (op 4003) 


v_and_b32 v17, -8, v1 // (op 4032) 
v_sub_u32 v93, v1, v17 // (op 4032) 

v_lshlrev_b32 v17, 4, v93 // (op 4032) 

v_lshrrev_b32 v93, 3, v1 // (op 4032) 
v_lshl_add_u32 v160, v93, 7, v17 // (op 4032) 


v_lshlrev_b32 v17, 2, v160 // (op 4032) 

v_lshrrev_b32 v93, 3, v17 // (op 4032) 

v_and_b32 v17, -32, v1 // (op 4056) 
v_sub_u32 v160, v1, v17 // (op 4056) 

v_lshlrev_b32 v17, 3, v160 // (op 4056) 

v_lshrrev_b32 v160, 5, v1 // (op 4056) 
v_lshl_add_u32 v161, v160, 8, v17 // (op 4056) 


s_mov_b32 s0, 0 // (op 1283) call()
s_ashr_i64 s[18:19], s[32:33], 7 // (op 1282) 
s_mov_b32 s1, s18 // (op 1282) convert

s_and_b32 s18, s1, 4294967294 // (op 1282) 

s_cmp_lt_u32 s0, s18 // (op 1282) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoop_1282 // (op 1282) Condition: Top (jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoop_1282 if false)
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoop_1282:
s_add_u32 s0, s0, 2 // (op 1284) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4132) 



ds_read_b64_tr_b8 v[162:163], v0 offset:1024 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[164:165], v0 offset:3072 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[166:167], v0 offset:17408 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[168:169], v0 offset:19456 // (op 65) Transpose load local data 
s_barrier  // (op 4145) 



ds_read_b128 v[170:173], v3 // (op 70) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 4159) 


ds_read_u8 v17, v7 offset:33792 // (op 75) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[162:169], v[170:173], a[0:3], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 82) 
ds_read_b64_tr_b8 v[174:175], v0 offset:1056 // (op 340) Transpose load local data 


ds_read_b64_tr_b8 v[176:177], v0 offset:3104 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[178:179], v0 offset:17440 // (op 340) Transpose load local data 
ds_read_b64_tr_b8 v[180:181], v0 offset:19488 // (op 340) Transpose load local data 


ds_read_u8 v17, v7 offset:33824 // (op 344) Load local data 



ds_read_b64_tr_b8 v[162:163], v0 offset:1088 // (op 502) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[174:181], v[170:173], a[4:7], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 346) 
ds_read_b64_tr_b8 v[164:165], v0 offset:3136 // (op 502) Transpose load local data 


ds_read_b64_tr_b8 v[166:167], v0 offset:17472 // (op 502) Transpose load local data 
ds_read_b64_tr_b8 v[168:169], v0 offset:19520 // (op 502) Transpose load local data 


ds_read_u8 v17, v7 offset:33856 // (op 506) Load local data 



ds_read_b64_tr_b8 v[174:175], v0 offset:1120 // (op 664) Transpose load local data 
ds_read_b64_tr_b8 v[176:177], v0 offset:3168 // (op 664) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[162:169], v[170:173], a[8:11], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 508) 
ds_read_b64_tr_b8 v[178:179], v0 offset:17504 // (op 664) Transpose load local data 


ds_read_b64_tr_b8 v[180:181], v0 offset:19552 // (op 664) Transpose load local data 


ds_read_u8 v17, v7 offset:33888 // (op 668) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[174:181], v[170:173], a[12:15], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 670) 
ds_read_b64_tr_b8 v[162:163], v0 offset:1152 // (op 826) Transpose load local data 


ds_read_b64_tr_b8 v[164:165], v0 offset:3200 // (op 826) Transpose load local data 
ds_read_b64_tr_b8 v[166:167], v0 offset:17536 // (op 826) Transpose load local data 
ds_read_b64_tr_b8 v[168:169], v0 offset:19584 // (op 826) Transpose load local data 


ds_read_u8 v17, v7 offset:33920 // (op 830) Load local data 



ds_read_b64_tr_b8 v[174:175], v0 offset:1184 // (op 988) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[162:169], v[170:173], a[16:19], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 832) 
ds_read_b64_tr_b8 v[176:177], v0 offset:3232 // (op 988) Transpose load local data 


ds_read_b64_tr_b8 v[178:179], v0 offset:17568 // (op 988) Transpose load local data 
ds_read_b64_tr_b8 v[180:181], v0 offset:19616 // (op 988) Transpose load local data 


ds_read_u8 v17, v7 offset:33952 // (op 992) Load local data 



ds_read_b64_tr_b8 v[162:163], v0 offset:1216 // (op 1150) Transpose load local data 
ds_read_b64_tr_b8 v[164:165], v0 offset:3264 // (op 1150) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[174:181], v[170:173], a[20:23], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 994) 
ds_read_b64_tr_b8 v[166:167], v0 offset:17600 // (op 1150) Transpose load local data 


ds_read_b64_tr_b8 v[168:169], v0 offset:19648 // (op 1150) Transpose load local data 


ds_read_u8 v17, v7 offset:33984 // (op 1154) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[162:169], v[170:173], a[24:27], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1156) 
ds_read_b64_tr_b8 v[174:175], v0 offset:1248 // (op 1312) Transpose load local data 


ds_read_b64_tr_b8 v[176:177], v0 offset:3296 // (op 1312) Transpose load local data 
ds_read_b64_tr_b8 v[178:179], v0 offset:17632 // (op 1312) Transpose load local data 
ds_read_b64_tr_b8 v[180:181], v0 offset:19680 // (op 1312) Transpose load local data 


ds_read_u8 v17, v7 offset:34016 // (op 1316) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[174:181], v[170:173], a[28:31], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1318) 





v_mov_b32 v17, v20 // (op 80) 

buffer_load_ubyte v16, v17, s[68:71], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v160, v15 // (op 10) 

buffer_load_dwordx2 v[22:23], v160, s[52:55], 0 offen offset:128 // (op 10) Load value

v_mov_b64 v[162:163], s[28:29] // (op 4) convert

v_mov_b32 v164, v14 // (op 4) 
s_lshl_b32 s1, s28, 8 // (op 4) 
v_add_u32 v165, v164, s1 // (op 4) 

buffer_load_dwordx4 v[26:29], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[30:33], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[34:37], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[38:41], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[42:45], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[46:49], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[50:53], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[54:57], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[58:61], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[62:65], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[66:69], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[70:73], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[74:77], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[78:81], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[82:85], v165, s[24:27], 0 offen // (op 4) Load value
v_add_u32 v165, v165, v162 // (op 4) 
buffer_load_dwordx4 v[86:89], v165, s[24:27], 0 offen // (op 4) Load value




v_mov_b32 v162, v18 // (op 17) 
s_lshl_b32 s1, s38, 3 // (op 17) 
v_add_u32 v163, v162, s1 // (op 17) 


buffer_load_dwordx2 v[24:25], v163, s[64:67], 0 offen // (op 17) Load value

s_barrier  // (op 4182) 



s_waitcnt vmcnt(36)
ds_write_b64 v93, v[90:91] offset:34816 // (op 235) Store local data 
s_barrier  // (op 4176) 



s_waitcnt vmcnt(35)
ds_write_b128 v159, v[94:97] offset:35840 // (op 225) Store local data 
s_waitcnt vmcnt(34)
ds_write_b128 v159, v[98:101] offset:36096 // (op 225) Store local data 
s_waitcnt vmcnt(33)
ds_write_b128 v159, v[102:105] offset:36352 // (op 225) Store local data 
s_waitcnt vmcnt(32)
ds_write_b128 v159, v[106:109] offset:36608 // (op 225) Store local data 
s_waitcnt vmcnt(31)
ds_write_b128 v159, v[110:113] offset:36864 // (op 225) Store local data 
s_waitcnt vmcnt(30)
ds_write_b128 v159, v[114:117] offset:37120 // (op 225) Store local data 
s_waitcnt vmcnt(29)
ds_write_b128 v159, v[118:121] offset:37376 // (op 225) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v159, v[122:125] offset:37632 // (op 225) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v159, v[126:129] offset:37888 // (op 225) Store local data 
s_waitcnt vmcnt(26)
ds_write_b128 v159, v[130:133] offset:38144 // (op 225) Store local data 
s_waitcnt vmcnt(25)
ds_write_b128 v159, v[134:137] offset:38400 // (op 225) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v159, v[138:141] offset:38656 // (op 225) Store local data 
s_waitcnt vmcnt(23)
ds_write_b128 v159, v[142:145] offset:38912 // (op 225) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v159, v[146:149] offset:39168 // (op 225) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v159, v[150:153] offset:39424 // (op 225) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v159, v[154:157] offset:39680 // (op 225) Store local data 
s_barrier  // (op 4188) 



v_add_u32 v163, 68608, v161 // (op 236) 
s_waitcnt vmcnt(19)
ds_write_b64 v163, v[12:13] // (op 236) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 4173) 



ds_read_b64_tr_b8 v[166:167], v10 offset:35840 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[168:169], v10 offset:37888 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[170:171], v10 offset:52224 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[172:173], v10 offset:54272 // (op 227) Transpose load local data 
s_barrier  // (op 4179) 



ds_read_b128 v[174:177], v4 offset:34816 // (op 229) Load local data 
s_barrier  // (op 4225) 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 4185) 


v_add_u32 v163, 68608, v11 // (op 231) 
ds_read_u8 v165, v163 // (op 231) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[166:173], v[174:177], a[0:3], v165, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 234) 
ds_read_b64_tr_b8 v[178:179], v10 offset:35872 // (op 353) Transpose load local data 


ds_read_b64_tr_b8 v[180:181], v10 offset:37920 // (op 353) Transpose load local data 
ds_read_b64_tr_b8 v[182:183], v10 offset:52256 // (op 353) Transpose load local data 
ds_read_b64_tr_b8 v[184:185], v10 offset:54304 // (op 353) Transpose load local data 


v_add_u32 v163, 68640, v11 // (op 357) 
ds_read_u8 v165, v163 // (op 357) Load local data 




ds_read_b64_tr_b8 v[166:167], v10 offset:35904 // (op 515) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[178:185], v[174:177], a[4:7], v165, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 347) 
ds_read_b64_tr_b8 v[168:169], v10 offset:37952 // (op 515) Transpose load local data 


ds_read_b64_tr_b8 v[170:171], v10 offset:52288 // (op 515) Transpose load local data 
ds_read_b64_tr_b8 v[172:173], v10 offset:54336 // (op 515) Transpose load local data 


v_add_u32 v163, 68672, v11 // (op 519) 
ds_read_u8 v165, v163 // (op 519) Load local data 




ds_read_b64_tr_b8 v[178:179], v10 offset:35936 // (op 677) Transpose load local data 
ds_read_b64_tr_b8 v[180:181], v10 offset:37984 // (op 677) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[166:173], v[174:177], a[8:11], v165, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 509) 
ds_read_b64_tr_b8 v[182:183], v10 offset:52320 // (op 677) Transpose load local data 


ds_read_b64_tr_b8 v[184:185], v10 offset:54368 // (op 677) Transpose load local data 


v_add_u32 v163, 68704, v11 // (op 681) 
ds_read_u8 v165, v163 // (op 681) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[178:185], v[174:177], a[12:15], v165, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 671) 
ds_read_b64_tr_b8 v[166:167], v10 offset:35968 // (op 839) Transpose load local data 


ds_read_b64_tr_b8 v[168:169], v10 offset:38016 // (op 839) Transpose load local data 
ds_read_b64_tr_b8 v[170:171], v10 offset:52352 // (op 839) Transpose load local data 
ds_read_b64_tr_b8 v[172:173], v10 offset:54400 // (op 839) Transpose load local data 


v_add_u32 v163, 68736, v11 // (op 843) 
ds_read_u8 v165, v163 // (op 843) Load local data 




ds_read_b64_tr_b8 v[178:179], v10 offset:36000 // (op 1001) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[166:173], v[174:177], a[16:19], v165, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 833) 
ds_read_b64_tr_b8 v[180:181], v10 offset:38048 // (op 1001) Transpose load local data 


ds_read_b64_tr_b8 v[182:183], v10 offset:52384 // (op 1001) Transpose load local data 
ds_read_b64_tr_b8 v[184:185], v10 offset:54432 // (op 1001) Transpose load local data 


v_add_u32 v163, 68768, v11 // (op 1005) 
ds_read_u8 v165, v163 // (op 1005) Load local data 




ds_read_b64_tr_b8 v[166:167], v10 offset:36032 // (op 1163) Transpose load local data 
ds_read_b64_tr_b8 v[168:169], v10 offset:38080 // (op 1163) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[178:185], v[174:177], a[20:23], v165, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 995) 
ds_read_b64_tr_b8 v[170:171], v10 offset:52416 // (op 1163) Transpose load local data 


ds_read_b64_tr_b8 v[172:173], v10 offset:54464 // (op 1163) Transpose load local data 


v_add_u32 v163, 68800, v11 // (op 1167) 
ds_read_u8 v165, v163 // (op 1167) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[166:173], v[174:177], a[24:27], v165, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1157) 
ds_read_b64_tr_b8 v[178:179], v10 offset:36064 // (op 1325) Transpose load local data 


ds_read_b64_tr_b8 v[180:181], v10 offset:38112 // (op 1325) Transpose load local data 
ds_read_b64_tr_b8 v[182:183], v10 offset:52448 // (op 1325) Transpose load local data 
ds_read_b64_tr_b8 v[184:185], v10 offset:54496 // (op 1325) Transpose load local data 


s_barrier  // (op 4295) 
v_add_u32 v163, 68832, v11 // (op 1329) 
ds_read_u8 v165, v163 // (op 1329) Load local data 

s_barrier  // (op 4299) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[178:185], v[174:177], a[28:31], v165, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1319) 






buffer_load_ubyte v21, v17, s[68:71], 0 offen offset:12 // (op 233) Load value




buffer_load_dwordx2 v[90:91], v160, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v20, 8, v20 // (op 3770) 


v_add_u32 v15, 128, v15 // (op 3642) 
v_mov_b64 v[166:167], s[28:29] // (op 224) convert

s_lshl_b32 s1, s28, 7 // (op 224) 
s_mul_i32 s18, 3, s1 // (op 224) 

v_add_u32 v17, v164, s18 // (op 224) 

buffer_load_dwordx4 v[94:97], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[98:101], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[102:105], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[106:109], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[110:113], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[114:117], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[118:121], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[122:125], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[126:129], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[130:133], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[134:137], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[138:141], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[142:145], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[146:149], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[150:153], v17, s[24:27], 0 offen // (op 224) Load value
v_add_u32 v17, v17, v166 // (op 224) 
buffer_load_dwordx4 v[154:157], v17, s[24:27], 0 offen // (op 224) Load value





s_lshl_b32 s1, s38, 2 // (op 238) 
s_mul_i32 s18, 3, s1 // (op 238) 

s_lshl_b32 s1, s28, 8 // (op 3578) 
v_add_u32 v14, v14, s1 // (op 3578) 

v_add_u32 v17, v162, s18 // (op 238) 


buffer_load_dwordx2 v[12:13], v17, s[64:67], 0 offen // (op 238) Load value


s_barrier  // (op 4148) 
s_lshl_b32 s1, s38, 3 // (op 3706) 



v_add_u32 v18, v18, s1 // (op 3706) 

s_waitcnt vmcnt(36)
ds_write_b64 v19, v[22:23] // (op 13) Store local data 
s_barrier  // (op 4135) 



s_waitcnt vmcnt(35)
ds_write_b128 v92, v[26:29] offset:1024 // (op 6) Store local data 
s_waitcnt vmcnt(34)
ds_write_b128 v92, v[30:33] offset:1280 // (op 6) Store local data 
s_waitcnt vmcnt(33)
ds_write_b128 v92, v[34:37] offset:1536 // (op 6) Store local data 
s_waitcnt vmcnt(32)
ds_write_b128 v92, v[38:41] offset:1792 // (op 6) Store local data 
s_waitcnt vmcnt(31)
ds_write_b128 v92, v[42:45] offset:2048 // (op 6) Store local data 
s_waitcnt vmcnt(30)
ds_write_b128 v92, v[46:49] offset:2304 // (op 6) Store local data 
s_waitcnt vmcnt(29)
ds_write_b128 v92, v[50:53] offset:2560 // (op 6) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v92, v[54:57] offset:2816 // (op 6) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v92, v[58:61] offset:3072 // (op 6) Store local data 
s_waitcnt vmcnt(26)
ds_write_b128 v92, v[62:65] offset:3328 // (op 6) Store local data 
s_waitcnt vmcnt(25)
ds_write_b128 v92, v[66:69] offset:3584 // (op 6) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v92, v[70:73] offset:3840 // (op 6) Store local data 
s_waitcnt vmcnt(23)
ds_write_b128 v92, v[74:77] offset:4096 // (op 6) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v92, v[78:81] offset:4352 // (op 6) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v92, v[82:85] offset:4608 // (op 6) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v92, v[86:89] offset:4864 // (op 6) Store local data 
s_barrier  // (op 4162) 



s_waitcnt vmcnt(19)
ds_write_b64 v158, v[24:25] offset:33792 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[32:33], 7 // (op 1282) 
s_mov_b32 s1, s18 // (op 1282) convert

s_and_b32 s18, s1, 4294967294 // (op 1282) 

s_cmp_lt_u32 s0, s18 // (op 1282) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoop_1282 // (op 1282) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoop_1282 if true)
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoop_1282:















s_mul_hi_i32 s0, s16, s8 // (op 3534) 
v_and_b32 v14, -16, v1 // (op 3534) 
s_add_i32 s1, s0, s16 // (op 3534) 

v_sub_u32 v15, v1, v14 // (op 3534) 

s_ashr_i32 s0, s1, 31 // (op 3534) 
v_lshlrev_b32 v14, 4, v15 // (op 3534) 

s_cmp_eq_i32 s8, 0 // (op 3534) 
s_mov_b32 s18, scc // (op 3534) 
v_and_b32 v15, 4294967280, v1 // (op 3534) 
s_cmp_eq_i32 s18, 0 // (op 3534) 
s_cselect_b32 s19, 0, -1 // (op 3534) 

s_and_b32 s18, s9, 31 // (op 3534) 
s_lshl_b32 s49, 1, s18 // (op 3534) 
s_add_i32 s50, s49, s19 // (op 3534) 


s_and_b32 s19, s0, s50 // (op 3534) 


s_add_i32 s0, s1, s19 // (op 3534) 


s_ashr_i32 s1, s0, s18 // (op 3534) 


s_lshl_b32 s0, s9, 24 // (op 3534) 
s_ashr_i32 s18, s0, 31 // (op 3534) 

s_xor_b32 s0, s1, s18 // (op 3534) 

s_sub_i32 s1, s0, s18 // (op 3534) 


s_mul_hi_i32 s0, s1, s10 // (op 3534) 
s_add_i32 s18, s0, s1 // (op 3534) 

s_ashr_i32 s0, s18, 31 // (op 3534) 
s_cmp_eq_i32 s10, 0 // (op 3534) 
s_mov_b32 s19, scc // (op 3534) 
s_cmp_eq_i32 s19, 0 // (op 3534) 
s_cselect_b32 s49, 0, -1 // (op 3534) 

s_and_b32 s19, s11, 31 // (op 3534) 
s_lshl_b32 s50, 1, s19 // (op 3534) 
s_add_i32 s51, s50, s49 // (op 3534) 


s_and_b32 s49, s0, s51 // (op 3534) 


s_add_i32 s0, s18, s49 // (op 3534) 


s_ashr_i32 s18, s0, s19 // (op 3534) 


s_lshl_b32 s0, s11, 24 // (op 3534) 
s_ashr_i32 s19, s0, 31 // (op 3534) 

s_xor_b32 s0, s18, s19 // (op 3534) 

s_sub_i32 s18, s0, s19 // (op 3534) 


s_mul_i32 s0, s1, s13 // (op 3534) 

s_sub_u32 s1, s16, s0 // (op 3534) 

s_mul_i32 s0, s18, s13 // (op 3534) 
s_add_u32 s0, s0, s1 // (op 3534) 


s_mul_hi_i32 s1, s16, s7 // (op 3534) 
s_add_i32 s18, s1, s16 // (op 3534) 

s_ashr_i32 s1, s18, 31 // (op 3534) 
s_cmp_eq_i32 s7, 0 // (op 3534) 
s_mov_b32 s19, scc // (op 3534) 
s_cmp_eq_i32 s19, 0 // (op 3534) 
s_cselect_b32 s49, 0, -1 // (op 3534) 

s_and_b32 s19, s14, 31 // (op 3534) 
s_lshl_b32 s50, 1, s19 // (op 3534) 
s_add_i32 s51, s50, s49 // (op 3534) 


s_and_b32 s49, s1, s51 // (op 3534) 


s_add_i32 s1, s18, s49 // (op 3534) 


s_ashr_i32 s18, s1, s19 // (op 3534) 


s_lshl_b32 s1, s14, 24 // (op 3534) 
s_ashr_i32 s19, s1, 31 // (op 3534) 

s_xor_b32 s1, s18, s19 // (op 3534) 

s_sub_i32 s18, s1, s19 // (op 3534) 


s_add_u32 s50, s4, 16 // (op 3534) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3534) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3534) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3534) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 4 // (op 3534) 

s_mul_i32 s1, s13, s50 // (op 3534) 

s_mul_i32 s19, s6, s1 // (op 3534) 

s_mul_i32 s1, s18, s19 // (op 3534) 

s_sub_u32 s19, s16, s1 // (op 3534) 

s_mov_b32 s1, s19 // (op 3534) 
s_mul_hi_i32 s49, s1, s12 // (op 3534) 
s_add_i32 s50, s49, s1 // (op 3534) 


s_ashr_i32 s1, s50, 31 // (op 3534) 
s_cmp_eq_i32 s12, 0 // (op 3534) 
s_mov_b32 s49, scc // (op 3534) 
s_cmp_eq_i32 s49, 0 // (op 3534) 
s_cselect_b32 s51, 0, -1 // (op 3534) 

s_and_b32 s49, s48, 31 // (op 3534) 
s_lshl_b32 s72, 1, s49 // (op 3534) 
s_add_i32 s73, s72, s51 // (op 3534) 


s_and_b32 s51, s1, s73 // (op 3534) 


s_add_i32 s1, s50, s51 // (op 3534) 


s_ashr_i32 s50, s1, s49 // (op 3534) 


s_lshl_b32 s1, s48, 24 // (op 3534) 
s_ashr_i32 s49, s1, 31 // (op 3534) 

s_xor_b32 s1, s50, s49 // (op 3534) 

s_sub_i32 s50, s1, s49 // (op 3534) 


s_mul_i32 s1, s15, s13 // (op 3534) 
s_add_u32 s72, s20, 256 // (op 3534) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 3534) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3534) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3534) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 8 // (op 3534) 

s_sub_i32 s49, s72, s1 // (op 3534) 

s_mul_i32 s51, s50, s49 // (op 3534) 


s_sub_u32 s49, s19, s51 // (op 3534) 


s_add_u32 s19, s49, s1 // (op 3534) 


s_cmp_eq_i32 s18, 0 // (op 3534) 
s_mov_b32 s1, scc // (op 3534) 

s_cmp_eq_i32 s1, 0 // (op 3534) 
s_cselect_b32 s18, s19, s0 // (op 3534) 



v_lshl_add_u32 v16, s18, 8, v14 // (op 3534) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3534) 
s_lshr_b32 s18, s0, 1 // (op 3534) 

v_lshl_add_u32 v14, s18, 8, v15 // (op 3534) 


v_mul_lo_u32 v15, v14, s28 // (op 3534) 

v_add_u32 v14, v16, v15 // (op 3534) 


s_mul_i32 s1, s28, s33 // (op 3536) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 3536) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3536) least significant
s_mul_hi_u32 s19, s28, s32 // (op 3536) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3536) most significant: sum
s_add_u32 s1, s1, s19 // (op 3536) most significant: sum


s_mov_b32 s24, s30 // (op 3536) reg expression
s_mov_b32 s25, s31 // (op 3536) reg expression
s_mov_b32 s26, s0 // (op 3536) reg expression
s_mov_b32 s27, 131072 // (op 3536) call()

s_mul_hi_i32 s0, s16, s8 // (op 3608) 
v_lshrrev_b32 v15, 3, v1 // (op 3608) 
s_add_i32 s1, s0, s16 // (op 3608) 

v_and_b32 v16, -8, v1 // (op 3608) 
s_ashr_i32 s0, s1, 31 // (op 3608) 
v_sub_u32 v17, v1, v16 // (op 3608) 

s_cmp_eq_i32 s8, 0 // (op 3608) 
s_mov_b32 s18, scc // (op 3608) 
v_lshlrev_b32 v16, 4, v17 // (op 3608) 

s_cmp_eq_i32 s18, 0 // (op 3608) 
s_cselect_b32 s19, 0, -1 // (op 3608) 

s_and_b32 s18, s9, 31 // (op 3608) 
s_lshl_b32 s49, 1, s18 // (op 3608) 
s_add_i32 s50, s49, s19 // (op 3608) 


s_and_b32 s19, s0, s50 // (op 3608) 


s_add_i32 s0, s1, s19 // (op 3608) 


s_ashr_i32 s1, s0, s18 // (op 3608) 


s_lshl_b32 s0, s9, 24 // (op 3608) 
s_ashr_i32 s18, s0, 31 // (op 3608) 

s_xor_b32 s0, s1, s18 // (op 3608) 

s_sub_i32 s1, s0, s18 // (op 3608) 


s_mul_hi_i32 s0, s1, s10 // (op 3608) 
s_add_i32 s18, s0, s1 // (op 3608) 

s_ashr_i32 s0, s18, 31 // (op 3608) 
s_cmp_eq_i32 s10, 0 // (op 3608) 
s_mov_b32 s19, scc // (op 3608) 
s_cmp_eq_i32 s19, 0 // (op 3608) 
s_cselect_b32 s49, 0, -1 // (op 3608) 

s_and_b32 s19, s11, 31 // (op 3608) 
s_lshl_b32 s50, 1, s19 // (op 3608) 
s_add_i32 s51, s50, s49 // (op 3608) 


s_and_b32 s49, s0, s51 // (op 3608) 


s_add_i32 s0, s18, s49 // (op 3608) 


s_ashr_i32 s18, s0, s19 // (op 3608) 


s_lshl_b32 s0, s11, 24 // (op 3608) 
s_ashr_i32 s19, s0, 31 // (op 3608) 

s_xor_b32 s0, s18, s19 // (op 3608) 

s_sub_i32 s18, s0, s19 // (op 3608) 


s_add_u32 s50, s4, 16 // (op 3608) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3608) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3608) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3608) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 4 // (op 3608) 

s_mul_i32 s0, s18, s50 // (op 3608) 

s_sub_i32 s49, s1, s0 // (op 3608) 


s_mul_hi_i32 s0, s16, s7 // (op 3608) 
s_add_i32 s1, s0, s16 // (op 3608) 

s_ashr_i32 s0, s1, 31 // (op 3608) 
s_cmp_eq_i32 s7, 0 // (op 3608) 
s_mov_b32 s18, scc // (op 3608) 
s_cmp_eq_i32 s18, 0 // (op 3608) 
s_cselect_b32 s19, 0, -1 // (op 3608) 

s_and_b32 s18, s14, 31 // (op 3608) 
s_lshl_b32 s72, 1, s18 // (op 3608) 
s_add_i32 s73, s72, s19 // (op 3608) 


s_and_b32 s19, s0, s73 // (op 3608) 


s_add_i32 s0, s1, s19 // (op 3608) 


s_ashr_i32 s1, s0, s18 // (op 3608) 


s_lshl_b32 s0, s14, 24 // (op 3608) 
s_ashr_i32 s18, s0, 31 // (op 3608) 

s_xor_b32 s0, s1, s18 // (op 3608) 

s_sub_i32 s1, s0, s18 // (op 3608) 


s_mul_i32 s0, s13, s50 // (op 3608) 

s_mul_i32 s18, s6, s0 // (op 3608) 

s_mul_i32 s0, s1, s18 // (op 3608) 

s_sub_u32 s18, s16, s0 // (op 3608) 

s_mul_hi_i32 s0, s18, s12 // (op 3608) 
s_add_i32 s19, s0, s18 // (op 3608) 


s_ashr_i32 s0, s19, 31 // (op 3608) 
s_cmp_eq_i32 s12, 0 // (op 3608) 
s_mov_b32 s18, scc // (op 3608) 
s_cmp_eq_i32 s18, 0 // (op 3608) 
s_cselect_b32 s50, 0, -1 // (op 3608) 

s_and_b32 s18, s48, 31 // (op 3608) 
s_lshl_b32 s51, 1, s18 // (op 3608) 
s_add_i32 s72, s51, s50 // (op 3608) 


s_and_b32 s50, s0, s72 // (op 3608) 


s_add_i32 s0, s19, s50 // (op 3608) 


s_ashr_i32 s19, s0, s18 // (op 3608) 


s_lshl_b32 s0, s48, 24 // (op 3608) 
s_ashr_i32 s18, s0, 31 // (op 3608) 

s_xor_b32 s0, s19, s18 // (op 3608) 

s_sub_i32 s19, s0, s18 // (op 3608) 


s_cmp_eq_i32 s1, 0 // (op 3608) 
s_mov_b32 s0, scc // (op 3608) 

s_cmp_eq_i32 s0, 0 // (op 3608) 
s_cselect_b32 s1, s19, s49 // (op 3608) 



v_lshl_add_u32 v17, s1, 4, v15 // (op 3608) 


v_mul_lo_u32 v93, v17, s35 // (op 3608) most significant: low of low * high
v_mul_hi_u32 v15, v17, s34 // (op 3608) most significant: high of low * low
v_mul_lo_u32 v92, v17, s34 // (op 3608) least significant: low of low * low
v_add_u32 v93, v93, v15 // (op 3608) most significant: sum


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3608) 
s_lshr_b32 s18, s0, 1 // (op 3608) 

v_lshl_add_u32 v15, s18, 8, v16 // (op 3608) 


v_mov_b32 v17, 0 // (op 3608) convert
v_mov_b32 v16, v15 // (op 3608) convert

v_add_co_u32 v18, s[0:1], v16, v92 // (op 3608) least significant half
v_addc_co_u32 v19, s[0:1], v17, v93, s[0:1] // (op 3608) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 3608) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 3608) 

v_mov_b32 v15, v16 // (op 3608) convert

s_mul_i32 s1, s34, s5 // (op 3610) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 3610) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 3610) least significant
s_mul_hi_u32 s19, s34, s4 // (op 3610) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3610) most significant: sum
s_add_u32 s1, s1, s19 // (op 3610) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3610) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3610) 

s_mov_b32 s52, s36 // (op 3610) reg expression
s_mov_b32 s53, s37 // (op 3610) reg expression
s_mov_b32 s54, s0 // (op 3610) reg expression
s_mov_b32 s55, 131072 // (op 3610) call()

s_mul_hi_i32 s0, s16, s8 // (op 3672) 
v_and_b32 v16, -32, v1 // (op 3672) 
s_add_i32 s1, s0, s16 // (op 3672) 

v_sub_u32 v17, v1, v16 // (op 3672) 

s_ashr_i32 s0, s1, 31 // (op 3672) 
v_lshlrev_b32 v16, 3, v17 // (op 3672) 

s_cmp_eq_i32 s8, 0 // (op 3672) 
s_mov_b32 s18, scc // (op 3672) 
v_lshrrev_b32 v17, 5, v1 // (op 3672) 
s_cmp_eq_i32 s18, 0 // (op 3672) 
s_cselect_b32 s19, 0, -1 // (op 3672) 

s_and_b32 s18, s9, 31 // (op 3672) 
s_lshl_b32 s49, 1, s18 // (op 3672) 
s_add_i32 s50, s49, s19 // (op 3672) 


s_and_b32 s19, s0, s50 // (op 3672) 


s_add_i32 s0, s1, s19 // (op 3672) 


s_ashr_i32 s1, s0, s18 // (op 3672) 


s_lshl_b32 s0, s9, 24 // (op 3672) 
s_ashr_i32 s18, s0, 31 // (op 3672) 

s_xor_b32 s0, s1, s18 // (op 3672) 

s_sub_i32 s1, s0, s18 // (op 3672) 


s_mul_hi_i32 s0, s1, s10 // (op 3672) 
s_add_i32 s18, s0, s1 // (op 3672) 

s_ashr_i32 s0, s18, 31 // (op 3672) 
s_cmp_eq_i32 s10, 0 // (op 3672) 
s_mov_b32 s19, scc // (op 3672) 
s_cmp_eq_i32 s19, 0 // (op 3672) 
s_cselect_b32 s49, 0, -1 // (op 3672) 

s_and_b32 s19, s11, 31 // (op 3672) 
s_lshl_b32 s50, 1, s19 // (op 3672) 
s_add_i32 s51, s50, s49 // (op 3672) 


s_and_b32 s49, s0, s51 // (op 3672) 


s_add_i32 s0, s18, s49 // (op 3672) 


s_ashr_i32 s18, s0, s19 // (op 3672) 


s_lshl_b32 s0, s11, 24 // (op 3672) 
s_ashr_i32 s19, s0, 31 // (op 3672) 

s_xor_b32 s0, s18, s19 // (op 3672) 

s_sub_i32 s18, s0, s19 // (op 3672) 


s_mul_i32 s0, s1, s13 // (op 3672) 

s_sub_u32 s1, s16, s0 // (op 3672) 

s_mul_i32 s0, s18, s13 // (op 3672) 
s_add_u32 s0, s0, s1 // (op 3672) 


s_mul_hi_i32 s1, s16, s7 // (op 3672) 
s_add_i32 s18, s1, s16 // (op 3672) 

s_ashr_i32 s1, s18, 31 // (op 3672) 
s_cmp_eq_i32 s7, 0 // (op 3672) 
s_mov_b32 s19, scc // (op 3672) 
s_cmp_eq_i32 s19, 0 // (op 3672) 
s_cselect_b32 s49, 0, -1 // (op 3672) 

s_and_b32 s19, s14, 31 // (op 3672) 
s_lshl_b32 s50, 1, s19 // (op 3672) 
s_add_i32 s51, s50, s49 // (op 3672) 


s_and_b32 s49, s1, s51 // (op 3672) 


s_add_i32 s1, s18, s49 // (op 3672) 


s_ashr_i32 s18, s1, s19 // (op 3672) 


s_lshl_b32 s1, s14, 24 // (op 3672) 
s_ashr_i32 s19, s1, 31 // (op 3672) 

s_xor_b32 s1, s18, s19 // (op 3672) 

s_sub_i32 s18, s1, s19 // (op 3672) 


s_add_u32 s50, s4, 16 // (op 3672) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3672) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3672) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3672) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 4 // (op 3672) 

s_mul_i32 s1, s13, s50 // (op 3672) 

s_mul_i32 s19, s6, s1 // (op 3672) 

s_mul_i32 s1, s18, s19 // (op 3672) 

s_sub_u32 s19, s16, s1 // (op 3672) 

s_mov_b32 s1, s19 // (op 3672) 
s_mul_hi_i32 s49, s1, s12 // (op 3672) 
s_add_i32 s50, s49, s1 // (op 3672) 


s_ashr_i32 s1, s50, 31 // (op 3672) 
s_cmp_eq_i32 s12, 0 // (op 3672) 
s_mov_b32 s49, scc // (op 3672) 
s_cmp_eq_i32 s49, 0 // (op 3672) 
s_cselect_b32 s51, 0, -1 // (op 3672) 

s_and_b32 s49, s48, 31 // (op 3672) 
s_lshl_b32 s72, 1, s49 // (op 3672) 
s_add_i32 s73, s72, s51 // (op 3672) 


s_and_b32 s51, s1, s73 // (op 3672) 


s_add_i32 s1, s50, s51 // (op 3672) 


s_ashr_i32 s50, s1, s49 // (op 3672) 


s_lshl_b32 s1, s48, 24 // (op 3672) 
s_ashr_i32 s49, s1, 31 // (op 3672) 

s_xor_b32 s1, s50, s49 // (op 3672) 

s_sub_i32 s50, s1, s49 // (op 3672) 


s_mul_i32 s1, s15, s13 // (op 3672) 
s_add_u32 s72, s20, 256 // (op 3672) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 3672) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3672) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3672) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 8 // (op 3672) 

s_sub_i32 s49, s72, s1 // (op 3672) 

s_mul_i32 s51, s50, s49 // (op 3672) 


s_sub_u32 s49, s19, s51 // (op 3672) 


s_add_u32 s19, s49, s1 // (op 3672) 


s_cmp_eq_i32 s18, 0 // (op 3672) 
s_mov_b32 s1, scc // (op 3672) 

s_cmp_eq_i32 s1, 0 // (op 3672) 
s_cselect_b32 s18, s19, s0 // (op 3672) 



v_lshl_add_u32 v18, s18, 8, v16 // (op 3672) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3672) 
s_lshr_b32 s18, s0, 1 // (op 3672) 

v_lshl_add_u32 v16, s18, 3, v17 // (op 3672) 


v_mul_lo_u32 v17, v16, s38 // (op 3672) 

v_add_u32 v16, v18, v17 // (op 3672) 


s_mul_i32 s1, s38, s43 // (op 3674) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3674) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3674) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3674) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3674) most significant: sum
s_add_u32 s1, s1, s19 // (op 3674) most significant: sum


s_mov_b32 s64, s40 // (op 3674) reg expression
s_mov_b32 s65, s41 // (op 3674) reg expression
s_mov_b32 s66, s0 // (op 3674) reg expression
s_mov_b32 s67, 131072 // (op 3674) call()

s_mul_hi_i32 s0, s16, s8 // (op 3736) 
v_and_b32 v17, 15, v1 // (op 3736) 
s_add_i32 s1, s0, s16 // (op 3736) 

v_and_b32 v18, 63, v1 // (op 3736) 
s_ashr_i32 s0, s1, 31 // (op 3736) 
v_lshrrev_b32 v19, 4, v18 // (op 3736) 

s_cmp_eq_i32 s8, 0 // (op 3736) 
s_mov_b32 s18, scc // (op 3736) 
s_cmp_eq_i32 s18, 0 // (op 3736) 
s_cselect_b32 s19, 0, -1 // (op 3736) 

s_and_b32 s18, s9, 31 // (op 3736) 
s_lshl_b32 s49, 1, s18 // (op 3736) 
s_add_i32 s50, s49, s19 // (op 3736) 


s_and_b32 s19, s0, s50 // (op 3736) 


s_add_i32 s0, s1, s19 // (op 3736) 


s_ashr_i32 s1, s0, s18 // (op 3736) 


s_lshl_b32 s0, s9, 24 // (op 3736) 
s_ashr_i32 s18, s0, 31 // (op 3736) 

s_xor_b32 s0, s1, s18 // (op 3736) 

s_sub_i32 s1, s0, s18 // (op 3736) 


s_mul_hi_i32 s0, s1, s10 // (op 3736) 
s_add_i32 s18, s0, s1 // (op 3736) 

s_ashr_i32 s0, s18, 31 // (op 3736) 
s_cmp_eq_i32 s10, 0 // (op 3736) 
s_mov_b32 s19, scc // (op 3736) 
s_cmp_eq_i32 s19, 0 // (op 3736) 
s_cselect_b32 s49, 0, -1 // (op 3736) 

s_and_b32 s19, s11, 31 // (op 3736) 
s_lshl_b32 s50, 1, s19 // (op 3736) 
s_add_i32 s51, s50, s49 // (op 3736) 


s_and_b32 s49, s0, s51 // (op 3736) 


s_add_i32 s0, s18, s49 // (op 3736) 


s_ashr_i32 s18, s0, s19 // (op 3736) 


s_lshl_b32 s0, s11, 24 // (op 3736) 
s_ashr_i32 s19, s0, 31 // (op 3736) 

s_xor_b32 s0, s18, s19 // (op 3736) 

s_sub_i32 s18, s0, s19 // (op 3736) 


s_add_u32 s50, s4, 16 // (op 3736) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3736) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3736) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3736) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 4 // (op 3736) 

s_mul_i32 s0, s18, s50 // (op 3736) 

s_sub_i32 s49, s1, s0 // (op 3736) 


s_mul_hi_i32 s0, s16, s7 // (op 3736) 
s_add_i32 s1, s0, s16 // (op 3736) 

s_ashr_i32 s0, s1, 31 // (op 3736) 
s_cmp_eq_i32 s7, 0 // (op 3736) 
s_mov_b32 s18, scc // (op 3736) 
s_cmp_eq_i32 s18, 0 // (op 3736) 
s_cselect_b32 s19, 0, -1 // (op 3736) 

s_and_b32 s18, s14, 31 // (op 3736) 
s_lshl_b32 s72, 1, s18 // (op 3736) 
s_add_i32 s73, s72, s19 // (op 3736) 


s_and_b32 s19, s0, s73 // (op 3736) 


s_add_i32 s0, s1, s19 // (op 3736) 


s_ashr_i32 s1, s0, s18 // (op 3736) 


s_lshl_b32 s0, s14, 24 // (op 3736) 
s_ashr_i32 s18, s0, 31 // (op 3736) 

s_xor_b32 s0, s1, s18 // (op 3736) 

s_sub_i32 s1, s0, s18 // (op 3736) 


s_mul_i32 s0, s13, s50 // (op 3736) 

s_mul_i32 s18, s6, s0 // (op 3736) 

s_mul_i32 s0, s1, s18 // (op 3736) 

s_sub_u32 s18, s16, s0 // (op 3736) 

s_mul_hi_i32 s0, s18, s12 // (op 3736) 
s_add_i32 s19, s0, s18 // (op 3736) 


s_ashr_i32 s0, s19, 31 // (op 3736) 
s_cmp_eq_i32 s12, 0 // (op 3736) 
s_mov_b32 s18, scc // (op 3736) 
s_cmp_eq_i32 s18, 0 // (op 3736) 
s_cselect_b32 s50, 0, -1 // (op 3736) 

s_and_b32 s18, s48, 31 // (op 3736) 
s_lshl_b32 s51, 1, s18 // (op 3736) 
s_add_i32 s72, s51, s50 // (op 3736) 


s_and_b32 s50, s0, s72 // (op 3736) 


s_add_i32 s0, s19, s50 // (op 3736) 


s_ashr_i32 s19, s0, s18 // (op 3736) 


s_lshl_b32 s0, s48, 24 // (op 3736) 
s_ashr_i32 s18, s0, 31 // (op 3736) 

s_xor_b32 s0, s19, s18 // (op 3736) 

s_sub_i32 s19, s0, s18 // (op 3736) 


s_cmp_eq_i32 s1, 0 // (op 3736) 
s_mov_b32 s0, scc // (op 3736) 

s_cmp_eq_i32 s0, 0 // (op 3736) 
s_cselect_b32 s1, s19, s49 // (op 3736) 



v_lshl_add_u32 v18, s1, 4, v17 // (op 3736) 


v_mul_lo_u32 v17, v18, s44 // (op 3736) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 3736) 
s_lshr_b32 s18, s0, 1 // (op 3736) 

v_lshl_add_u32 v18, s18, 3, v19 // (op 3736) 


v_add_u32 v19, v18, v17 // (op 3736) 


s_mul_i32 s1, s44, s5 // (op 3738) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 3738) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 3738) least significant
s_mul_hi_u32 s19, s44, s4 // (op 3738) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3738) most significant: sum
s_add_u32 s1, s1, s19 // (op 3738) most significant: sum


s_mov_b32 s68, s46 // (op 3738) reg expression
s_mov_b32 s69, s47 // (op 3738) reg expression
s_mov_b32 s70, s0 // (op 3738) reg expression
s_mov_b32 s71, 131072 // (op 3738) call()

v_and_b32 v20, -16, v1 // (op 3829) 
v_sub_u32 v17, v1, v20 // (op 3829) 

v_lshlrev_b32 v18, 4, v17 // (op 3829) 

v_lshrrev_b32 v17, 4, v1 // (op 3829) 
v_lshl_add_u32 v20, v17, 12, v18 // (op 3829) 


v_and_b32 v17, -8, v1 // (op 3887) 
v_sub_u32 v18, v1, v17 // (op 3887) 

v_lshlrev_b32 v17, 4, v18 // (op 3887) 

v_lshrrev_b32 v18, 3, v1 // (op 3887) 
v_lshl_add_u32 v22, v18, 7, v17 // (op 3887) 


v_lshlrev_b32 v17, 2, v22 // (op 3887) 

v_lshrrev_b32 v18, 3, v17 // (op 3887) 

v_and_b32 v17, -32, v1 // (op 3940) 
v_sub_u32 v22, v1, v17 // (op 3940) 

v_lshlrev_b32 v17, 3, v22 // (op 3940) 

v_lshrrev_b32 v22, 5, v1 // (op 3940) 
v_lshl_add_u32 v23, v22, 8, v17 // (op 3940) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1288) 
s_and_b32 s18, s0, 4294967294 // (op 1288) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 1287) 
s_mov_b32 s19, s0 // (op 1287) convert

s_cmp_lt_u32 s18, s19 // (op 1287) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_1287 // (op 1287) Condition: Top (jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_1287 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoopTail_1287:

s_add_u32 s18, s18, 1 // (op 1289) 
v_mov_b64 v[12:13], s[28:29] // (op 166) convert

v_mov_b32 v17, v14 // (op 166) 
v_mov_b32 v21, v17 // (op 166) reg expression
buffer_load_dwordx4 v[24:27], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[28:31], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[32:35], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[36:39], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[40:43], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[44:47], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[48:51], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[52:55], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[56:59], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[60:63], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[64:67], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[68:71], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[72:75], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[76:79], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[80:83], v21, s[24:27], 0 offen // (op 166) Load value
v_add_u32 v21, v21, v12 // (op 166) 
buffer_load_dwordx4 v[84:87], v21, s[24:27], 0 offen // (op 166) Load value



s_lshl_b32 s19, s28, 7 // (op 3524) 
v_add_u32 v14, v14, s19 // (op 3524) 



s_barrier  // (op 4138) 
v_mov_b32 v17, v15 // (op 179) 




buffer_load_dwordx2 v[12:13], v17, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v15, 64, v15 // (op 3600) 
s_waitcnt vmcnt(16)
ds_write_b128 v20, v[24:27] offset:1024 // (op 167) Store local data 


v_mov_b32 v17, v16 // (op 180) 
v_mov_b32 v21, v17 // (op 180) reg expression

buffer_load_dwordx2 v[88:89], v21, s[64:67], 0 offen // (op 180) Load value


s_lshl_b32 s19, s38, 2 // (op 3664) 
v_add_u32 v16, v16, s19 // (op 3664) 

s_waitcnt vmcnt(16)
ds_write_b128 v20, v[28:31] offset:1280 // (op 167) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v20, v[32:35] offset:1536 // (op 167) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v20, v[36:39] offset:1792 // (op 167) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v20, v[40:43] offset:2048 // (op 167) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v20, v[44:47] offset:2304 // (op 167) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v20, v[48:51] offset:2560 // (op 167) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v20, v[52:55] offset:2816 // (op 167) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v20, v[56:59] offset:3072 // (op 167) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v20, v[60:63] offset:3328 // (op 167) Store local data 
s_waitcnt vmcnt(7)
ds_write_b128 v20, v[64:67] offset:3584 // (op 167) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v20, v[68:71] offset:3840 // (op 167) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v20, v[72:75] offset:4096 // (op 167) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v20, v[76:79] offset:4352 // (op 167) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v20, v[80:83] offset:4608 // (op 167) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[84:87] offset:4864 // (op 167) Store local data 

s_barrier  // (op 4151) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4141) 






ds_read_b64_tr_b8 v[24:25], v0 offset:1024 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:3072 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:17408 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:19456 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b64 v18, v[12:13] // (op 177) Store local data 

s_barrier  // (op 4165) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4155) 






ds_read_b128 v[32:35], v3 // (op 171) Load local data 
s_waitcnt vmcnt(0)
ds_write_b64 v23, v[88:89] offset:33792 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 4169) 


ds_read_u8 v17, v7 offset:33792 // (op 173) Load local data 


v_mov_b32 v12, v19 // (op 175) 

buffer_load_ubyte v13, v12, s[68:71], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v19, 4, v19 // (op 3728) 



ds_read_b64_tr_b8 v[36:37], v0 offset:1056 // (op 321) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:3104 // (op 321) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:35], a[0:3], v17, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 176) 
ds_read_b64_tr_b8 v[40:41], v0 offset:17440 // (op 321) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v0 offset:19488 // (op 321) Transpose load local data 


ds_read_u8 v12, v7 offset:33824 // (op 325) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:43], v[32:35], a[4:7], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 327) 
ds_read_b64_tr_b8 v[24:25], v0 offset:1088 // (op 483) Transpose load local data 


ds_read_b64_tr_b8 v[26:27], v0 offset:3136 // (op 483) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:17472 // (op 483) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:19520 // (op 483) Transpose load local data 


ds_read_u8 v12, v7 offset:33856 // (op 487) Load local data 



ds_read_b64_tr_b8 v[36:37], v0 offset:1120 // (op 645) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:31], v[32:35], a[8:11], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 489) 
ds_read_b64_tr_b8 v[38:39], v0 offset:3168 // (op 645) Transpose load local data 


ds_read_b64_tr_b8 v[40:41], v0 offset:17504 // (op 645) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:19552 // (op 645) Transpose load local data 


ds_read_u8 v12, v7 offset:33888 // (op 649) Load local data 



ds_read_b64_tr_b8 v[24:25], v0 offset:1152 // (op 807) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:3200 // (op 807) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:43], v[32:35], a[12:15], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 651) 
ds_read_b64_tr_b8 v[28:29], v0 offset:17536 // (op 807) Transpose load local data 


ds_read_b64_tr_b8 v[30:31], v0 offset:19584 // (op 807) Transpose load local data 


ds_read_u8 v12, v7 offset:33920 // (op 811) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:31], v[32:35], a[16:19], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 813) 
ds_read_b64_tr_b8 v[36:37], v0 offset:1184 // (op 969) Transpose load local data 


ds_read_b64_tr_b8 v[38:39], v0 offset:3232 // (op 969) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:17568 // (op 969) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:19616 // (op 969) Transpose load local data 


ds_read_u8 v12, v7 offset:33952 // (op 973) Load local data 



ds_read_b64_tr_b8 v[24:25], v0 offset:1216 // (op 1131) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:43], v[32:35], a[20:23], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 975) 
ds_read_b64_tr_b8 v[26:27], v0 offset:3264 // (op 1131) Transpose load local data 


ds_read_b64_tr_b8 v[28:29], v0 offset:17600 // (op 1131) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:19648 // (op 1131) Transpose load local data 


ds_read_u8 v12, v7 offset:33984 // (op 1135) Load local data 



ds_read_b64_tr_b8 v[36:37], v0 offset:1248 // (op 1293) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:3296 // (op 1293) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[24:31], v[32:35], a[24:27], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1137) 
ds_read_b64_tr_b8 v[40:41], v0 offset:17632 // (op 1293) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v0 offset:19680 // (op 1293) Transpose load local data 


ds_read_u8 v12, v7 offset:34016 // (op 1297) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[36:43], v[32:35], a[28:31], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1299) 




s_ashr_i64 s[0:1], s[32:33], 7 // (op 1287) 
s_mov_b32 s19, s0 // (op 1287) convert

s_cmp_lt_u32 s18, s19 // (op 1287) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoopTail_1287 // (op 1287) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoopTail_1287 if true)
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_1287:






















s_barrier  // (op 4283) 



















v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 




buffer_store_dwordx4 v[14:17], v5, s[60:63], 0 offen // (op 52) Store value

v_accvgpr_read v10, a4 // (op 1617) 
v_accvgpr_read v11, a5 // (op 1617) 
v_accvgpr_read v12, a6 // (op 1617) 
v_accvgpr_read v13, a7 // (op 1617) 
v_mul_f32 v14, v9, v10 // (op 1617) 
v_mul_f32 v15, v9, v11 // (op 1617) 
v_mul_f32 v16, v9, v12 // (op 1617) 
v_mul_f32 v17, v9, v13 // (op 1617) 




buffer_store_dwordx4 v[14:17], v5, s[60:63], 0 offen offset:128 // (op 1619) Store value

v_accvgpr_read v10, a8 // (op 1635) 
v_accvgpr_read v11, a9 // (op 1635) 
v_accvgpr_read v12, a10 // (op 1635) 
v_accvgpr_read v13, a11 // (op 1635) 
v_mul_f32 v14, v9, v10 // (op 1635) 
v_mul_f32 v15, v9, v11 // (op 1635) 
v_mul_f32 v16, v9, v12 // (op 1635) 
v_mul_f32 v17, v9, v13 // (op 1635) 




buffer_store_dwordx4 v[14:17], v5, s[60:63], 0 offen offset:256 // (op 1637) Store value

v_accvgpr_read v10, a12 // (op 1653) 
v_accvgpr_read v11, a13 // (op 1653) 
v_accvgpr_read v12, a14 // (op 1653) 
v_accvgpr_read v13, a15 // (op 1653) 
v_mul_f32 v14, v9, v10 // (op 1653) 
v_mul_f32 v15, v9, v11 // (op 1653) 
v_mul_f32 v16, v9, v12 // (op 1653) 
v_mul_f32 v17, v9, v13 // (op 1653) 




buffer_store_dwordx4 v[14:17], v5, s[60:63], 0 offen offset:384 // (op 1655) Store value

v_accvgpr_read v10, a16 // (op 1671) 
v_accvgpr_read v11, a17 // (op 1671) 
v_accvgpr_read v12, a18 // (op 1671) 
v_accvgpr_read v13, a19 // (op 1671) 
v_mul_f32 v14, v9, v10 // (op 1671) 
v_mul_f32 v15, v9, v11 // (op 1671) 
v_mul_f32 v16, v9, v12 // (op 1671) 
v_mul_f32 v17, v9, v13 // (op 1671) 




buffer_store_dwordx4 v[14:17], v5, s[60:63], 0 offen offset:512 // (op 1673) Store value

v_accvgpr_read v10, a20 // (op 1689) 
v_accvgpr_read v11, a21 // (op 1689) 
v_accvgpr_read v12, a22 // (op 1689) 
v_accvgpr_read v13, a23 // (op 1689) 
v_mul_f32 v14, v9, v10 // (op 1689) 
v_mul_f32 v15, v9, v11 // (op 1689) 
v_mul_f32 v16, v9, v12 // (op 1689) 
v_mul_f32 v17, v9, v13 // (op 1689) 




buffer_store_dwordx4 v[14:17], v5, s[60:63], 0 offen offset:640 // (op 1691) Store value

v_accvgpr_read v10, a24 // (op 1707) 
v_accvgpr_read v11, a25 // (op 1707) 
v_accvgpr_read v12, a26 // (op 1707) 
v_accvgpr_read v13, a27 // (op 1707) 
v_mul_f32 v14, v9, v10 // (op 1707) 
v_mul_f32 v15, v9, v11 // (op 1707) 
v_mul_f32 v16, v9, v12 // (op 1707) 
v_mul_f32 v17, v9, v13 // (op 1707) 




buffer_store_dwordx4 v[14:17], v5, s[60:63], 0 offen offset:768 // (op 1709) Store value

v_accvgpr_read v10, a28 // (op 1725) 
v_accvgpr_read v11, a29 // (op 1725) 
v_accvgpr_read v12, a30 // (op 1725) 
v_accvgpr_read v13, a31 // (op 1725) 
v_mul_f32 v14, v9, v10 // (op 1725) 
v_mul_f32 v15, v9, v11 // (op 1725) 
v_mul_f32 v16, v9, v12 // (op 1725) 
v_mul_f32 v17, v9, v13 // (op 1725) 




buffer_store_dwordx4 v[14:17], v5, s[60:63], 0 offen offset:896 // (op 1727) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:896 // (op 1893) Load value
v_accvgpr_read v14, a28 // (op 1890) 
v_accvgpr_read v15, a29 // (op 1890) 
v_accvgpr_read v16, a30 // (op 1890) 
v_accvgpr_read v17, a31 // (op 1890) 
v_mul_f32 v18, v9, v14 // (op 1890) 
v_mul_f32 v19, v9, v15 // (op 1890) 
v_mul_f32 v20, v9, v16 // (op 1890) 
v_mul_f32 v21, v9, v17 // (op 1890) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1894) 
v_mul_f32 v15, v8, v11 // (op 1894) 
v_mul_f32 v16, v8, v12 // (op 1894) 
v_mul_f32 v17, v8, v13 // (op 1894) 

v_add_f32 v10, v14, v18 // (op 1891) 
v_add_f32 v11, v15, v19 // (op 1891) 
v_add_f32 v12, v16, v20 // (op 1891) 
v_add_f32 v13, v17, v21 // (op 1891) 





buffer_store_dwordx4 v[10:13], v5, s[60:63], 0 offen offset:896 // (op 1892) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:768 // (op 1875) Load value
v_accvgpr_read v14, a24 // (op 1872) 
v_accvgpr_read v15, a25 // (op 1872) 
v_accvgpr_read v16, a26 // (op 1872) 
v_accvgpr_read v17, a27 // (op 1872) 
v_mul_f32 v18, v9, v14 // (op 1872) 
v_mul_f32 v19, v9, v15 // (op 1872) 
v_mul_f32 v20, v9, v16 // (op 1872) 
v_mul_f32 v21, v9, v17 // (op 1872) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1876) 
v_mul_f32 v15, v8, v11 // (op 1876) 
v_mul_f32 v16, v8, v12 // (op 1876) 
v_mul_f32 v17, v8, v13 // (op 1876) 

v_add_f32 v10, v14, v18 // (op 1873) 
v_add_f32 v11, v15, v19 // (op 1873) 
v_add_f32 v12, v16, v20 // (op 1873) 
v_add_f32 v13, v17, v21 // (op 1873) 





buffer_store_dwordx4 v[10:13], v5, s[60:63], 0 offen offset:768 // (op 1874) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:640 // (op 1857) Load value
v_accvgpr_read v14, a20 // (op 1854) 
v_accvgpr_read v15, a21 // (op 1854) 
v_accvgpr_read v16, a22 // (op 1854) 
v_accvgpr_read v17, a23 // (op 1854) 
v_mul_f32 v18, v9, v14 // (op 1854) 
v_mul_f32 v19, v9, v15 // (op 1854) 
v_mul_f32 v20, v9, v16 // (op 1854) 
v_mul_f32 v21, v9, v17 // (op 1854) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1858) 
v_mul_f32 v15, v8, v11 // (op 1858) 
v_mul_f32 v16, v8, v12 // (op 1858) 
v_mul_f32 v17, v8, v13 // (op 1858) 

v_add_f32 v10, v14, v18 // (op 1855) 
v_add_f32 v11, v15, v19 // (op 1855) 
v_add_f32 v12, v16, v20 // (op 1855) 
v_add_f32 v13, v17, v21 // (op 1855) 





buffer_store_dwordx4 v[10:13], v5, s[60:63], 0 offen offset:640 // (op 1856) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:512 // (op 1839) Load value
v_accvgpr_read v14, a16 // (op 1836) 
v_accvgpr_read v15, a17 // (op 1836) 
v_accvgpr_read v16, a18 // (op 1836) 
v_accvgpr_read v17, a19 // (op 1836) 
v_mul_f32 v18, v9, v14 // (op 1836) 
v_mul_f32 v19, v9, v15 // (op 1836) 
v_mul_f32 v20, v9, v16 // (op 1836) 
v_mul_f32 v21, v9, v17 // (op 1836) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1840) 
v_mul_f32 v15, v8, v11 // (op 1840) 
v_mul_f32 v16, v8, v12 // (op 1840) 
v_mul_f32 v17, v8, v13 // (op 1840) 

v_add_f32 v10, v14, v18 // (op 1837) 
v_add_f32 v11, v15, v19 // (op 1837) 
v_add_f32 v12, v16, v20 // (op 1837) 
v_add_f32 v13, v17, v21 // (op 1837) 





buffer_store_dwordx4 v[10:13], v5, s[60:63], 0 offen offset:512 // (op 1838) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:384 // (op 1821) Load value
v_accvgpr_read v14, a12 // (op 1818) 
v_accvgpr_read v15, a13 // (op 1818) 
v_accvgpr_read v16, a14 // (op 1818) 
v_accvgpr_read v17, a15 // (op 1818) 
v_mul_f32 v18, v9, v14 // (op 1818) 
v_mul_f32 v19, v9, v15 // (op 1818) 
v_mul_f32 v20, v9, v16 // (op 1818) 
v_mul_f32 v21, v9, v17 // (op 1818) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1822) 
v_mul_f32 v15, v8, v11 // (op 1822) 
v_mul_f32 v16, v8, v12 // (op 1822) 
v_mul_f32 v17, v8, v13 // (op 1822) 

v_add_f32 v10, v14, v18 // (op 1819) 
v_add_f32 v11, v15, v19 // (op 1819) 
v_add_f32 v12, v16, v20 // (op 1819) 
v_add_f32 v13, v17, v21 // (op 1819) 





buffer_store_dwordx4 v[10:13], v5, s[60:63], 0 offen offset:384 // (op 1820) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:256 // (op 1803) Load value
v_accvgpr_read v14, a8 // (op 1800) 
v_accvgpr_read v15, a9 // (op 1800) 
v_accvgpr_read v16, a10 // (op 1800) 
v_accvgpr_read v17, a11 // (op 1800) 
v_mul_f32 v18, v9, v14 // (op 1800) 
v_mul_f32 v19, v9, v15 // (op 1800) 
v_mul_f32 v20, v9, v16 // (op 1800) 
v_mul_f32 v21, v9, v17 // (op 1800) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1804) 
v_mul_f32 v15, v8, v11 // (op 1804) 
v_mul_f32 v16, v8, v12 // (op 1804) 
v_mul_f32 v17, v8, v13 // (op 1804) 

v_add_f32 v10, v14, v18 // (op 1801) 
v_add_f32 v11, v15, v19 // (op 1801) 
v_add_f32 v12, v16, v20 // (op 1801) 
v_add_f32 v13, v17, v21 // (op 1801) 





buffer_store_dwordx4 v[10:13], v5, s[60:63], 0 offen offset:256 // (op 1802) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[56:59], 0 offen offset:128 // (op 1785) Load value
v_accvgpr_read v14, a4 // (op 1782) 
v_accvgpr_read v15, a5 // (op 1782) 
v_accvgpr_read v16, a6 // (op 1782) 
v_accvgpr_read v17, a7 // (op 1782) 
v_mul_f32 v18, v9, v14 // (op 1782) 
v_mul_f32 v19, v9, v15 // (op 1782) 
v_mul_f32 v20, v9, v16 // (op 1782) 
v_mul_f32 v21, v9, v17 // (op 1782) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1786) 
v_mul_f32 v15, v8, v11 // (op 1786) 
v_mul_f32 v16, v8, v12 // (op 1786) 
v_mul_f32 v17, v8, v13 // (op 1786) 

v_add_f32 v10, v14, v18 // (op 1783) 
v_add_f32 v11, v15, v19 // (op 1783) 
v_add_f32 v12, v16, v20 // (op 1783) 
v_add_f32 v13, v17, v21 // (op 1783) 





buffer_store_dwordx4 v[10:13], v5, s[60:63], 0 offen offset:128 // (op 1784) Store value




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
RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
.LRR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__end:
.size RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_, .LRR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__end-RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 220
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 69632 // lds bytes
  .amdhsa_accum_offset 188
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
    DestDataType: s
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
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: int64, semantic: SizeFree0 },
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
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [256, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 69632
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 186
    .agpr_count: 32
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 16
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 20
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 72
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 152
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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

