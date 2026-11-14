// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_,@function
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 34977) 
s_mul_hi_i32 s1, s16, s10 // (op 35232) 
v_and_b32 v0, 48, v1 // (op 35038) 
v_and_b32 v3, 63, v1 // (op 35038) 
v_sub_u32 v4, v3, v0 // (op 35038) 

v_lshrrev_b32 v3, 1, v4 // (op 35038) 
v_add_lshl_u32 v5, v0, v3, 8 // (op 35038) 


v_and_b32 v0, 1, v4 // (op 35038) 

v_lshlrev_b32 v3, 3, v0 // (op 35038) 

v_lshrrev_b32 v0, 7, v1 // (op 35038) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 35038) 


v_add_u32 v0, v5, v4 // (op 35038) 


v_lshrrev_b32 v3, 6, v1 // (op 35095) 
v_and_b32 v4, 1, v3 // (op 35095) 

v_and_b32 v3, 15, v1 // (op 35095) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 35095) 


v_and_b32 v3, 48, v1 // (op 35095) 
v_lshl_add_u32 v4, v5, 7, v3 // (op 35095) 


v_lshrrev_b32 v3, 7, v1 // (op 35153) 
v_and_b32 v5, 15, v1 // (op 35153) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 35153) 


v_and_b32 v3, 63, v1 // (op 35153) 
v_lshrrev_b32 v5, 4, v3 // (op 35153) 

v_lshl_add_u32 v3, v5, 8, v6 // (op 35153) 


v_lshrrev_b32 v5, 6, v1 // (op 35211) 
v_and_b32 v6, 1, v5 // (op 35211) 

v_and_b32 v5, 15, v1 // (op 35211) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 35211) 


v_and_b32 v5, 63, v1 // (op 35211) 
v_lshrrev_b32 v6, 4, v5 // (op 35211) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 35211) 


v_and_b32 v6, 48, v1 // (op 35278) 
v_and_b32 v7, 63, v1 // (op 35278) 
v_sub_u32 v8, v7, v6 // (op 35278) 

v_lshrrev_b32 v7, 1, v8 // (op 35278) 
v_add_lshl_u32 v9, v6, v7, 8 // (op 35278) 


v_and_b32 v6, 1, v8 // (op 35278) 

v_lshlrev_b32 v7, 3, v6 // (op 35278) 

v_lshrrev_b32 v6, 7, v1 // (op 35278) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 35278) 


v_add_u32 v6, v9, v8 // (op 35278) 


v_lshrrev_b32 v7, 6, v1 // (op 35322) 
v_and_b32 v8, 1, v7 // (op 35322) 

v_and_b32 v7, 15, v1 // (op 35322) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 35322) 


v_and_b32 v7, 48, v1 // (op 35322) 
v_lshl_add_u32 v8, v9, 7, v7 // (op 35322) 


v_lshrrev_b32 v7, 7, v1 // (op 35351) 
v_and_b32 v9, 15, v1 // (op 35351) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 35351) 


v_and_b32 v7, 63, v1 // (op 35351) 
v_lshrrev_b32 v9, 4, v7 // (op 35351) 

v_lshl_add_u32 v7, v9, 8, v10 // (op 35351) 


v_lshrrev_b32 v9, 6, v1 // (op 35380) 
v_and_b32 v10, 1, v9 // (op 35380) 

v_and_b32 v9, 15, v1 // (op 35380) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 35380) 


v_and_b32 v9, 63, v1 // (op 35380) 
v_lshrrev_b32 v10, 4, v9 // (op 35380) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 35380) 


v_lshrrev_b32 v10, 7, v1 // (op 34977) 
s_add_i32 s18, s0, s16 // (op 34977) 

v_and_b32 v11, 63, v1 // (op 34977) 
s_ashr_i32 s0, s18, 31 // (op 34977) 
v_lshrrev_b32 v12, 4, v11 // (op 34977) 
s_cmp_eq_i32 s10, 0 // (op 34977) 
s_mov_b32 s19, scc // (op 34977) 
v_lshlrev_b32 v13, 2, v12 // (op 34977) 

s_cmp_eq_i32 s19, 0 // (op 34977) 
s_cselect_b32 s51, 0, -1 // (op 34977) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 34977) 


s_and_b32 s19, s11, 31 // (op 34977) 
v_lshrrev_b32 v10, 2, v11 // (op 34977) 

v_and_b32 v11, 3, v10 // (op 34977) 

v_lshrrev_b32 v10, 7, v1 // (op 35232) 
v_and_b32 v13, 63, v1 // (op 35232) 
s_lshl_b32 s52, 1, s19 // (op 34977) 
v_and_b32 v14, 3, v1 // (op 34977) 
s_add_i32 s53, s52, s51 // (op 34977) 


s_and_b32 s51, s0, s53 // (op 34977) 


s_add_i32 s0, s1, s16 // (op 35232) 

s_ashr_i32 s1, s0, 31 // (op 35232) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 34977) 


s_add_i32 s52, s18, s51 // (op 34977) 


v_lshrrev_b32 v11, 6, v1 // (op 34977) 
s_ashr_i32 s18, s52, s19 // (op 34977) 


v_and_b32 v14, 1, v11 // (op 34977) 

s_lshl_b32 s19, s11, 24 // (op 34977) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 34977) 


s_ashr_i32 s51, s19, 31 // (op 34977) 

s_xor_b32 s19, s18, s51 // (op 34977) 

s_sub_i32 s18, s19, s51 // (op 34977) 


s_mul_hi_i32 s19, s18, s12 // (op 34977) 
s_add_i32 s51, s19, s18 // (op 34977) 

s_ashr_i32 s19, s51, 31 // (op 34977) 
s_cmp_eq_i32 s12, 0 // (op 34977) 
s_mov_b32 s52, scc // (op 34977) 
s_cmp_eq_i32 s52, 0 // (op 34977) 
s_cselect_b32 s53, 0, -1 // (op 34977) 

s_and_b32 s52, s13, 31 // (op 34977) 
v_lshrrev_b32 v14, 4, v13 // (op 35232) 
s_cmp_eq_i32 s10, 0 // (op 35232) 
s_mov_b32 s54, scc // (op 35232) 
v_lshlrev_b32 v15, 2, v14 // (op 35232) 

s_cmp_eq_i32 s54, 0 // (op 35232) 
s_cselect_b32 s55, 0, -1 // (op 35232) 

s_lshl_b32 s54, 1, s52 // (op 34977) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 35232) 


v_lshrrev_b32 v10, 2, v13 // (op 35232) 

s_add_i32 s56, s54, s53 // (op 34977) 


s_and_b32 s53, s19, s56 // (op 34977) 


s_add_i32 s19, s51, s53 // (op 34977) 


s_ashr_i32 s51, s19, s52 // (op 34977) 


s_lshl_b32 s19, s13, 24 // (op 34977) 
s_ashr_i32 s52, s19, 31 // (op 34977) 

s_xor_b32 s19, s51, s52 // (op 34977) 

s_sub_i32 s51, s19, s52 // (op 34977) 


s_mul_i32 s19, s18, s15 // (op 34977) 
s_sub_u32 s52, s16, s19 // (op 34977) 

s_mul_i32 s19, s51, s15 // (op 34977) 
s_add_u32 s19, s19, s52 // (op 34977) 

s_mul_hi_i32 s52, s16, s9 // (op 34977) 
s_add_i32 s53, s52, s16 // (op 34977) 

s_ashr_i32 s52, s53, 31 // (op 34977) 
s_cmp_eq_i32 s9, 0 // (op 34977) 
s_mov_b32 s54, scc // (op 34977) 
s_cmp_eq_i32 s54, 0 // (op 34977) 
s_cselect_b32 s56, 0, -1 // (op 34977) 

s_and_b32 s54, s11, 31 // (op 35232) 
v_and_b32 v13, 3, v10 // (op 35232) 

s_lshl_b32 s57, 1, s54 // (op 35232) 
v_and_b32 v10, 3, v1 // (op 35232) 
s_add_i32 s58, s57, s55 // (op 35232) 


s_and_b32 s55, s1, s58 // (op 35232) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 35232) 


s_add_i32 s1, s0, s55 // (op 35232) 


v_lshrrev_b32 v10, 6, v1 // (op 35232) 
s_ashr_i32 s0, s1, s54 // (op 35232) 


v_and_b32 v13, 1, v10 // (op 35232) 

s_lshl_b32 s1, s11, 24 // (op 35232) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 35232) 


s_ashr_i32 s54, s1, 31 // (op 35232) 

s_xor_b32 s1, s0, s54 // (op 35232) 

s_sub_i32 s0, s1, s54 // (op 35232) 


s_mul_hi_i32 s1, s0, s12 // (op 35232) 
s_add_i32 s54, s1, s0 // (op 35232) 

s_ashr_i32 s1, s54, 31 // (op 35232) 
s_cmp_eq_i32 s12, 0 // (op 35232) 
s_mov_b32 s55, scc // (op 35232) 
s_cmp_eq_i32 s55, 0 // (op 35232) 
s_cselect_b32 s57, 0, -1 // (op 35232) 

s_and_b32 s55, s13, 31 // (op 35232) 
s_lshl_b32 s58, 1, s55 // (op 35232) 
s_add_i32 s59, s58, s57 // (op 35232) 


s_and_b32 s57, s1, s59 // (op 35232) 


s_add_i32 s1, s54, s57 // (op 35232) 


s_ashr_i32 s54, s1, s55 // (op 35232) 


s_lshl_b32 s1, s13, 24 // (op 35232) 
s_ashr_i32 s55, s1, 31 // (op 35232) 

s_xor_b32 s1, s54, s55 // (op 35232) 

s_sub_i32 s54, s1, s55 // (op 35232) 


s_mul_i32 s1, s0, s15 // (op 35232) 
s_sub_u32 s55, s16, s1 // (op 35232) 

s_mul_i32 s1, s54, s15 // (op 35232) 
s_add_u32 s1, s1, s55 // (op 35232) 

s_mul_hi_i32 s55, s16, s9 // (op 35232) 
s_add_i32 s57, s55, s16 // (op 35232) 

s_ashr_i32 s55, s57, 31 // (op 35232) 
s_cmp_eq_i32 s9, 0 // (op 35232) 
s_mov_b32 s58, scc // (op 35232) 
s_cmp_eq_i32 s58, 0 // (op 35232) 
s_cselect_b32 s59, 0, -1 // (op 35232) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 34977) 
s_lshl_b32 s60, 1, s58 // (op 34977) 
s_add_i32 s61, s60, s56 // (op 34977) 


s_and_b32 s56, s52, s61 // (op 34977) 


s_add_i32 s52, s53, s56 // (op 34977) 


s_ashr_i32 s53, s52, s58 // (op 34977) 


s_lshl_b32 s52, s46, 24 // (op 34977) 
s_ashr_i32 s56, s52, 31 // (op 34977) 

s_xor_b32 s52, s53, s56 // (op 34977) 

s_sub_i32 s53, s52, s56 // (op 34977) 


s_and_b32 s52, s46, 31 // (op 35232) 
s_lshl_b32 s56, 1, s52 // (op 35232) 
s_add_i32 s58, s56, s59 // (op 35232) 


s_and_b32 s56, s55, s58 // (op 35232) 


s_add_i32 s55, s57, s56 // (op 35232) 


s_ashr_i32 s56, s55, s52 // (op 35232) 


s_lshl_b32 s52, s46, 24 // (op 35232) 
s_ashr_i32 s55, s52, 31 // (op 35232) 

s_xor_b32 s52, s56, s55 // (op 35232) 

s_sub_i32 s56, s52, s55 // (op 35232) 


s_add_u32 s58, s6, 256 // (op 34977) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 34977) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 34977) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 34977) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 8 // (op 34977) 

s_mul_i32 s52, s15, s58 // (op 34977) 
s_mul_i32 s55, s8, s52 // (op 34977) 

s_mul_i32 s52, s53, s55 // (op 34977) 

s_sub_u32 s55, s16, s52 // (op 34977) 

s_mov_b32 s52, s55 // (op 34977) 
s_mul_hi_i32 s57, s52, s14 // (op 34977) 
s_add_i32 s60, s57, s52 // (op 34977) 


s_ashr_i32 s52, s60, 31 // (op 34977) 
s_cmp_eq_i32 s14, 0 // (op 34977) 
s_mov_b32 s57, scc // (op 34977) 
s_cmp_eq_i32 s57, 0 // (op 34977) 
s_cselect_b32 s61, 0, -1 // (op 34977) 

s_and_b32 s57, s48, 31 // (op 34977) 
s_lshl_b32 s62, 1, s57 // (op 34977) 
s_add_i32 s63, s62, s61 // (op 34977) 


s_and_b32 s61, s52, s63 // (op 34977) 


s_add_i32 s52, s60, s61 // (op 34977) 


s_ashr_i32 s60, s52, s57 // (op 34977) 


s_lshl_b32 s52, s48, 24 // (op 34977) 
s_ashr_i32 s57, s52, 31 // (op 34977) 

s_xor_b32 s52, s60, s57 // (op 34977) 

s_sub_i32 s60, s52, s57 // (op 34977) 


s_mul_i32 s52, s47, s15 // (op 34977) 
s_add_u32 s62, s20, 256 // (op 34977) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 34977) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 34977) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 34977) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 8 // (op 34977) 

s_sub_i32 s57, s62, s52 // (op 34977) 

s_mul_i32 s61, s60, s57 // (op 34977) 

s_sub_u32 s57, s55, s61 // (op 34977) 


s_add_u32 s55, s57, s52 // (op 34977) 


s_cmp_eq_i32 s53, 0 // (op 34977) 
s_mov_b32 s52, scc // (op 34977) 

s_cmp_eq_i32 s52, 0 // (op 34977) 
s_cselect_b32 s53, s55, s19 // (op 34977) 


s_mul_i32 s19, s51, s58 // (op 34977) 


v_lshl_add_u32 v13, s53, 8, v12 // (op 34977) 


s_sub_i32 s51, s18, s19 // (op 34977) 


s_cmp_eq_i32 s52, 0 // (op 34977) 
s_cselect_b32 s53, s60, s51 // (op 34977) 



v_lshl_add_u32 v12, s53, 8, v11 // (op 34977) 


v_mul_lo_u32 v11, v12, s2 // (op 34977) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 34977) 


s_add_u32 s18, s6, 256 // (op 35232) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 35232) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 35232) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 35232) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 8 // (op 35232) 

s_mul_i32 s55, s15, s18 // (op 35232) 
s_mul_i32 s51, s8, s55 // (op 35232) 

s_mul_i32 s55, s56, s51 // (op 35232) 

s_sub_u32 s51, s16, s55 // (op 35232) 

s_mov_b32 s55, s51 // (op 35232) 
s_mul_hi_i32 s52, s55, s14 // (op 35232) 
s_add_i32 s53, s52, s55 // (op 35232) 


s_ashr_i32 s52, s53, 31 // (op 35232) 
s_cmp_eq_i32 s14, 0 // (op 35232) 
s_mov_b32 s55, scc // (op 35232) 
s_cmp_eq_i32 s55, 0 // (op 35232) 
s_cselect_b32 s57, 0, -1 // (op 35232) 

s_and_b32 s55, s48, 31 // (op 35232) 
s_lshl_b32 s58, 1, s55 // (op 35232) 
s_add_i32 s59, s58, s57 // (op 35232) 


s_and_b32 s57, s52, s59 // (op 35232) 


s_add_i32 s52, s53, s57 // (op 35232) 


s_ashr_i32 s53, s52, s55 // (op 35232) 


s_lshl_b32 s52, s48, 24 // (op 35232) 
s_ashr_i32 s55, s52, 31 // (op 35232) 

s_xor_b32 s52, s53, s55 // (op 35232) 

s_sub_i32 s53, s52, s55 // (op 35232) 


s_mul_i32 s52, s47, s15 // (op 35232) 
s_add_u32 s58, s20, 256 // (op 35232) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 35232) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 35232) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 35232) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 8 // (op 35232) 

s_sub_i32 s55, s58, s52 // (op 35232) 

s_mul_i32 s57, s53, s55 // (op 35232) 

s_sub_u32 s55, s51, s57 // (op 35232) 


s_add_u32 s51, s55, s52 // (op 35232) 


s_cmp_eq_i32 s56, 0 // (op 35232) 
s_mov_b32 s52, scc // (op 35232) 

s_cmp_eq_i32 s52, 0 // (op 35232) 
s_cselect_b32 s55, s51, s1 // (op 35232) 


s_mul_i32 s1, s54, s18 // (op 35232) 


v_lshl_add_u32 v11, s55, 8, v14 // (op 35232) 


s_sub_i32 s51, s0, s1 // (op 35232) 


s_cmp_eq_i32 s52, 0 // (op 35232) 
s_cselect_b32 s0, s53, s51 // (op 35232) 



v_lshl_add_u32 v13, s0, 8, v10 // (op 35232) 


v_mul_lo_u32 v10, v13, s4 // (op 35232) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 35232) 


s_mul_i32 s1, s2, s7 // (op 34979) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 34979) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 34979) least significant
s_mul_hi_u32 s19, s2, s6 // (op 34979) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34979) most significant: sum
s_add_u32 s1, s1, s19 // (op 34979) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 34979) 

s_mul_i32 s1, s4, s7 // (op 35234) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 35234) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 35234) least significant
s_mul_hi_u32 s52, s4, s6 // (op 35234) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 35234) most significant: sum
s_add_u32 s1, s1, s52 // (op 35234) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 35234) 

s_mov_b32 s56, s22 // (op 34979) reg expression
s_mov_b32 s57, s23 // (op 34979) reg expression
s_mov_b32 s58, s18 // (op 34979) reg expression
s_mov_b32 s59, 131072 // (op 34979) call()

s_mov_b32 s60, s24 // (op 35234) reg expression

s_mov_b32 s61, s25 // (op 35234) reg expression
s_mov_b32 s62, s52 // (op 35234) reg expression
s_mov_b32 s63, 131072 // (op 35234) call()


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
s_mul_hi_i32 s0, s16, s10 // (op 34739) 
v_and_b32 v14, -16, v1 // (op 34739) 
s_add_i32 s1, s0, s16 // (op 34739) 

v_sub_u32 v15, v1, v14 // (op 34739) 

s_ashr_i32 s0, s1, 31 // (op 34739) 
v_lshlrev_b32 v14, 4, v15 // (op 34739) 

s_cmp_eq_i32 s10, 0 // (op 34739) 
s_mov_b32 s18, scc // (op 34739) 
v_lshrrev_b32 v15, 4, v1 // (op 34739) 
s_cmp_eq_i32 s18, 0 // (op 34739) 
s_cselect_b32 s19, 0, -1 // (op 34739) 

s_and_b32 s18, s11, 31 // (op 34739) 
s_lshl_b32 s22, 1, s18 // (op 34739) 
s_add_i32 s23, s22, s19 // (op 34739) 


s_and_b32 s19, s0, s23 // (op 34739) 


s_add_i32 s0, s1, s19 // (op 34739) 


s_ashr_i32 s1, s0, s18 // (op 34739) 


s_lshl_b32 s0, s11, 24 // (op 34739) 
s_ashr_i32 s18, s0, 31 // (op 34739) 

s_xor_b32 s0, s1, s18 // (op 34739) 

s_sub_i32 s1, s0, s18 // (op 34739) 


s_mul_hi_i32 s0, s1, s12 // (op 34739) 
s_add_i32 s18, s0, s1 // (op 34739) 

s_ashr_i32 s0, s18, 31 // (op 34739) 
s_cmp_eq_i32 s12, 0 // (op 34739) 
s_mov_b32 s19, scc // (op 34739) 
s_cmp_eq_i32 s19, 0 // (op 34739) 
s_cselect_b32 s22, 0, -1 // (op 34739) 

s_and_b32 s19, s13, 31 // (op 34739) 
s_lshl_b32 s23, 1, s19 // (op 34739) 
s_add_i32 s24, s23, s22 // (op 34739) 


s_and_b32 s25, s0, s24 // (op 34739) 


s_add_i32 s0, s18, s25 // (op 34739) 


s_ashr_i32 s18, s0, s19 // (op 34739) 


s_lshl_b32 s0, s13, 24 // (op 34739) 
s_ashr_i32 s19, s0, 31 // (op 34739) 

s_xor_b32 s0, s18, s19 // (op 34739) 

s_sub_i32 s18, s0, s19 // (op 34739) 


s_mul_i32 s0, s1, s15 // (op 34739) 

s_sub_u32 s1, s16, s0 // (op 34739) 

s_mul_i32 s0, s18, s15 // (op 34739) 
s_add_u32 s0, s0, s1 // (op 34739) 


s_mul_hi_i32 s1, s16, s9 // (op 34739) 
s_add_i32 s18, s1, s16 // (op 34739) 

s_ashr_i32 s1, s18, 31 // (op 34739) 
s_cmp_eq_i32 s9, 0 // (op 34739) 
s_mov_b32 s19, scc // (op 34739) 
s_cmp_eq_i32 s19, 0 // (op 34739) 
s_cselect_b32 s22, 0, -1 // (op 34739) 

s_and_b32 s19, s46, 31 // (op 34739) 
s_lshl_b32 s23, 1, s19 // (op 34739) 
s_add_i32 s24, s23, s22 // (op 34739) 


s_and_b32 s25, s1, s24 // (op 34739) 


s_add_i32 s1, s18, s25 // (op 34739) 


s_ashr_i32 s18, s1, s19 // (op 34739) 


s_lshl_b32 s1, s46, 24 // (op 34739) 
s_ashr_i32 s19, s1, 31 // (op 34739) 

s_xor_b32 s1, s18, s19 // (op 34739) 

s_sub_i32 s18, s1, s19 // (op 34739) 


s_add_u32 s22, s6, 256 // (op 34739) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 34739) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 34739) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 34739) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 34739) 

s_mul_i32 s1, s15, s22 // (op 34739) 

s_mul_i32 s19, s8, s1 // (op 34739) 

s_mul_i32 s1, s18, s19 // (op 34739) 

s_sub_u32 s19, s16, s1 // (op 34739) 

s_mov_b32 s1, s19 // (op 34739) 
s_mul_hi_i32 s22, s1, s14 // (op 34739) 
s_add_i32 s23, s22, s1 // (op 34739) 


s_ashr_i32 s1, s23, 31 // (op 34739) 
s_cmp_eq_i32 s14, 0 // (op 34739) 
s_mov_b32 s22, scc // (op 34739) 
s_cmp_eq_i32 s22, 0 // (op 34739) 
s_cselect_b32 s24, 0, -1 // (op 34739) 

s_and_b32 s22, s48, 31 // (op 34739) 
s_lshl_b32 s25, 1, s22 // (op 34739) 
s_add_i32 s49, s25, s24 // (op 34739) 


s_and_b32 s24, s1, s49 // (op 34739) 


s_add_i32 s1, s23, s24 // (op 34739) 


s_ashr_i32 s23, s1, s22 // (op 34739) 


s_lshl_b32 s1, s48, 24 // (op 34739) 
s_ashr_i32 s22, s1, 31 // (op 34739) 

s_xor_b32 s1, s23, s22 // (op 34739) 

s_sub_i32 s23, s1, s22 // (op 34739) 


s_mul_i32 s1, s47, s15 // (op 34739) 
s_add_u32 s24, s20, 256 // (op 34739) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 34739) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 34739) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 34739) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 8 // (op 34739) 

s_sub_i32 s22, s24, s1 // (op 34739) 

s_mul_i32 s24, s23, s22 // (op 34739) 


s_sub_u32 s25, s19, s24 // (op 34739) 


s_add_u32 s19, s25, s1 // (op 34739) 


s_cmp_eq_i32 s18, 0 // (op 34739) 
s_mov_b32 s1, scc // (op 34739) 

s_cmp_eq_i32 s1, 0 // (op 34739) 
s_cselect_b32 s18, s19, s0 // (op 34739) 



v_lshl_add_u32 v16, s18, 8, v14 // (op 34739) 


v_mov_b32 v19, 0 // (op 34739) convert
v_mov_b32 v18, v16 // (op 34739) convert

v_mul_lo_u32 v17, v15, s27 // (op 34739) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 34739) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 34739) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 34739) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 34739) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 34739) most significant half



s_mul_i32 s1, s26, s31 // (op 34741) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 34741) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 34741) least significant
s_mul_hi_u32 s19, s26, s30 // (op 34741) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34741) most significant: sum
s_add_u32 s1, s1, s19 // (op 34741) most significant: sum


s_mov_b32 s52, s28 // (op 34741) reg expression
s_mov_b32 s53, s29 // (op 34741) reg expression
s_mov_b32 s54, s0 // (op 34741) reg expression
s_mov_b32 s55, 131072 // (op 34741) call()

s_mul_hi_i32 s0, s16, s10 // (op 34758) 
v_and_b32 v16, -16, v1 // (op 34758) 
s_add_i32 s1, s0, s16 // (op 34758) 

v_sub_u32 v17, v1, v16 // (op 34758) 

s_ashr_i32 s0, s1, 31 // (op 34758) 
v_lshlrev_b32 v16, 4, v17 // (op 34758) 

s_cmp_eq_i32 s10, 0 // (op 34758) 
s_mov_b32 s18, scc // (op 34758) 
v_lshrrev_b32 v17, 4, v1 // (op 34758) 
s_cmp_eq_i32 s18, 0 // (op 34758) 
s_cselect_b32 s19, 0, -1 // (op 34758) 

s_and_b32 s18, s11, 31 // (op 34758) 
s_lshl_b32 s22, 1, s18 // (op 34758) 
s_add_i32 s23, s22, s19 // (op 34758) 


s_and_b32 s19, s0, s23 // (op 34758) 


s_add_i32 s0, s1, s19 // (op 34758) 


s_ashr_i32 s1, s0, s18 // (op 34758) 


s_lshl_b32 s0, s11, 24 // (op 34758) 
s_ashr_i32 s18, s0, 31 // (op 34758) 

s_xor_b32 s0, s1, s18 // (op 34758) 

s_sub_i32 s1, s0, s18 // (op 34758) 


s_mul_hi_i32 s0, s1, s12 // (op 34758) 
s_add_i32 s18, s0, s1 // (op 34758) 

s_ashr_i32 s0, s18, 31 // (op 34758) 
s_cmp_eq_i32 s12, 0 // (op 34758) 
s_mov_b32 s19, scc // (op 34758) 
s_cmp_eq_i32 s19, 0 // (op 34758) 
s_cselect_b32 s22, 0, -1 // (op 34758) 

s_and_b32 s19, s13, 31 // (op 34758) 
s_lshl_b32 s23, 1, s19 // (op 34758) 
s_add_i32 s24, s23, s22 // (op 34758) 


s_and_b32 s25, s0, s24 // (op 34758) 


s_add_i32 s0, s18, s25 // (op 34758) 


s_ashr_i32 s18, s0, s19 // (op 34758) 


s_lshl_b32 s0, s13, 24 // (op 34758) 
s_ashr_i32 s19, s0, 31 // (op 34758) 

s_xor_b32 s0, s18, s19 // (op 34758) 

s_sub_i32 s18, s0, s19 // (op 34758) 


s_mul_i32 s0, s1, s15 // (op 34758) 

s_sub_u32 s1, s16, s0 // (op 34758) 

s_mul_i32 s0, s18, s15 // (op 34758) 
s_add_u32 s0, s0, s1 // (op 34758) 


s_mul_hi_i32 s1, s16, s9 // (op 34758) 
s_add_i32 s18, s1, s16 // (op 34758) 

s_ashr_i32 s1, s18, 31 // (op 34758) 
s_cmp_eq_i32 s9, 0 // (op 34758) 
s_mov_b32 s19, scc // (op 34758) 
s_cmp_eq_i32 s19, 0 // (op 34758) 
s_cselect_b32 s22, 0, -1 // (op 34758) 

s_and_b32 s19, s46, 31 // (op 34758) 
s_lshl_b32 s23, 1, s19 // (op 34758) 
s_add_i32 s24, s23, s22 // (op 34758) 


s_and_b32 s25, s1, s24 // (op 34758) 


s_add_i32 s1, s18, s25 // (op 34758) 


s_ashr_i32 s18, s1, s19 // (op 34758) 


s_lshl_b32 s1, s46, 24 // (op 34758) 
s_ashr_i32 s19, s1, 31 // (op 34758) 

s_xor_b32 s1, s18, s19 // (op 34758) 

s_sub_i32 s18, s1, s19 // (op 34758) 


s_add_u32 s50, s6, 256 // (op 34758) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34758) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34758) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34758) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34758) 

s_mul_i32 s1, s15, s24 // (op 34758) 

s_mul_i32 s19, s8, s1 // (op 34758) 

s_mul_i32 s1, s18, s19 // (op 34758) 

s_sub_u32 s19, s16, s1 // (op 34758) 

s_mov_b32 s1, s19 // (op 34758) 
s_mul_hi_i32 s22, s1, s14 // (op 34758) 
s_add_i32 s23, s22, s1 // (op 34758) 


s_ashr_i32 s1, s23, 31 // (op 34758) 
s_cmp_eq_i32 s14, 0 // (op 34758) 
s_mov_b32 s22, scc // (op 34758) 
s_cmp_eq_i32 s22, 0 // (op 34758) 
s_cselect_b32 s24, 0, -1 // (op 34758) 

s_and_b32 s22, s48, 31 // (op 34758) 
s_lshl_b32 s25, 1, s22 // (op 34758) 
s_add_i32 s49, s25, s24 // (op 34758) 


s_and_b32 s24, s1, s49 // (op 34758) 


s_add_i32 s1, s23, s24 // (op 34758) 


s_ashr_i32 s23, s1, s22 // (op 34758) 


s_lshl_b32 s1, s48, 24 // (op 34758) 
s_ashr_i32 s22, s1, 31 // (op 34758) 

s_xor_b32 s1, s23, s22 // (op 34758) 

s_sub_i32 s23, s1, s22 // (op 34758) 


s_mul_i32 s1, s47, s15 // (op 34758) 
s_add_u32 s24, s20, 256 // (op 34758) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 34758) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 34758) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 34758) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 34758) 

s_sub_i32 s22, s24, s1 // (op 34758) 

s_mul_i32 s24, s23, s22 // (op 34758) 


s_sub_u32 s25, s19, s24 // (op 34758) 


s_add_u32 s19, s25, s1 // (op 34758) 


s_cmp_eq_i32 s18, 0 // (op 34758) 
s_mov_b32 s1, scc // (op 34758) 

s_cmp_eq_i32 s1, 0 // (op 34758) 
s_cselect_b32 s18, s19, s0 // (op 34758) 



v_lshl_add_u32 v18, s18, 8, v16 // (op 34758) 


v_mov_b32 v21, 0 // (op 34758) convert
v_mov_b32 v20, v18 // (op 34758) convert

v_mul_lo_u32 v19, v17, s27 // (op 34758) most significant: low of low * high
v_mul_hi_u32 v16, v17, s26 // (op 34758) most significant: high of low * low
v_mul_lo_u32 v18, v17, s26 // (op 34758) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 34758) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v18 // (op 34758) least significant half
v_addc_co_u32 v17, s[0:1], v21, v19, s[0:1] // (op 34758) most significant half



s_mul_i32 s1, s26, s31 // (op 34760) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 34760) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 34760) least significant
s_mul_hi_u32 s19, s26, s30 // (op 34760) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34760) most significant: sum
s_add_u32 s1, s1, s19 // (op 34760) most significant: sum


s_mov_b32 s52, s28 // (op 34760) reg expression
s_mov_b32 s53, s29 // (op 34760) reg expression
s_mov_b32 s54, s0 // (op 34760) reg expression
s_mov_b32 s55, 131072 // (op 34760) call()

s_mul_hi_i32 s0, s16, s10 // (op 34803) 
v_lshrrev_b32 v18, 3, v1 // (op 34803) 
s_add_i32 s1, s0, s16 // (op 34803) 

v_and_b32 v19, -8, v1 // (op 34803) 
s_ashr_i32 s0, s1, 31 // (op 34803) 
v_sub_u32 v20, v1, v19 // (op 34803) 

s_cmp_eq_i32 s10, 0 // (op 34803) 
s_mov_b32 s18, scc // (op 34803) 
v_lshlrev_b32 v19, 4, v20 // (op 34803) 

s_cmp_eq_i32 s18, 0 // (op 34803) 
s_cselect_b32 s19, 0, -1 // (op 34803) 

v_mov_b32 v21, 0 // (op 34803) convert
v_mov_b32 v20, v19 // (op 34803) convert

s_and_b32 s18, s11, 31 // (op 34803) 
s_lshl_b32 s22, 1, s18 // (op 34803) 
s_add_i32 s23, s22, s19 // (op 34803) 


s_and_b32 s19, s0, s23 // (op 34803) 


s_add_i32 s0, s1, s19 // (op 34803) 


s_ashr_i32 s1, s0, s18 // (op 34803) 


s_lshl_b32 s0, s11, 24 // (op 34803) 
s_ashr_i32 s18, s0, 31 // (op 34803) 

s_xor_b32 s0, s1, s18 // (op 34803) 

s_sub_i32 s1, s0, s18 // (op 34803) 


s_mul_hi_i32 s0, s1, s12 // (op 34803) 
s_add_i32 s18, s0, s1 // (op 34803) 

s_ashr_i32 s0, s18, 31 // (op 34803) 
s_cmp_eq_i32 s12, 0 // (op 34803) 
s_mov_b32 s19, scc // (op 34803) 
s_cmp_eq_i32 s19, 0 // (op 34803) 
s_cselect_b32 s22, 0, -1 // (op 34803) 

s_and_b32 s19, s13, 31 // (op 34803) 
s_lshl_b32 s23, 1, s19 // (op 34803) 
s_add_i32 s24, s23, s22 // (op 34803) 


s_and_b32 s25, s0, s24 // (op 34803) 


s_add_i32 s0, s18, s25 // (op 34803) 


s_ashr_i32 s18, s0, s19 // (op 34803) 


s_lshl_b32 s0, s13, 24 // (op 34803) 
s_ashr_i32 s19, s0, 31 // (op 34803) 

s_xor_b32 s0, s18, s19 // (op 34803) 

s_sub_i32 s18, s0, s19 // (op 34803) 


s_add_u32 s50, s6, 256 // (op 34803) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34803) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34803) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34803) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34803) 

s_mul_i32 s0, s18, s24 // (op 34803) 

s_sub_i32 s18, s1, s0 // (op 34803) 


s_mul_hi_i32 s19, s16, s9 // (op 34803) 
s_add_i32 s0, s19, s16 // (op 34803) 

s_ashr_i32 s1, s0, 31 // (op 34803) 
s_cmp_eq_i32 s9, 0 // (op 34803) 
s_mov_b32 s19, scc // (op 34803) 
s_cmp_eq_i32 s19, 0 // (op 34803) 
s_cselect_b32 s22, 0, -1 // (op 34803) 

s_and_b32 s19, s46, 31 // (op 34803) 
s_lshl_b32 s23, 1, s19 // (op 34803) 
s_add_i32 s49, s23, s22 // (op 34803) 


s_and_b32 s22, s1, s49 // (op 34803) 


s_add_i32 s1, s0, s22 // (op 34803) 


s_ashr_i32 s0, s1, s19 // (op 34803) 


s_lshl_b32 s1, s46, 24 // (op 34803) 
s_ashr_i32 s19, s1, 31 // (op 34803) 

s_xor_b32 s1, s0, s19 // (op 34803) 

s_sub_i32 s0, s1, s19 // (op 34803) 


s_mul_i32 s1, s15, s24 // (op 34803) 

s_mul_i32 s19, s8, s1 // (op 34803) 

s_mul_i32 s1, s0, s19 // (op 34803) 

s_sub_u32 s19, s16, s1 // (op 34803) 

s_mul_hi_i32 s1, s19, s14 // (op 34803) 
s_add_i32 s22, s1, s19 // (op 34803) 


s_ashr_i32 s1, s22, 31 // (op 34803) 
s_cmp_eq_i32 s14, 0 // (op 34803) 
s_mov_b32 s19, scc // (op 34803) 
s_cmp_eq_i32 s19, 0 // (op 34803) 
s_cselect_b32 s23, 0, -1 // (op 34803) 

s_and_b32 s19, s48, 31 // (op 34803) 
s_lshl_b32 s24, 1, s19 // (op 34803) 
s_add_i32 s25, s24, s23 // (op 34803) 


s_and_b32 s23, s1, s25 // (op 34803) 


s_add_i32 s1, s22, s23 // (op 34803) 


s_ashr_i32 s22, s1, s19 // (op 34803) 


s_lshl_b32 s1, s48, 24 // (op 34803) 
s_ashr_i32 s19, s1, 31 // (op 34803) 

s_xor_b32 s1, s22, s19 // (op 34803) 

s_sub_i32 s22, s1, s19 // (op 34803) 


s_cmp_eq_i32 s0, 0 // (op 34803) 
s_mov_b32 s1, scc // (op 34803) 

s_cmp_eq_i32 s1, 0 // (op 34803) 
s_cselect_b32 s0, s22, s18 // (op 34803) 



v_lshl_add_u32 v19, s0, 8, v18 // (op 34803) 


v_mul_lo_u32 v23, v19, s33 // (op 34803) most significant: low of low * high
v_mul_hi_u32 v18, v19, s32 // (op 34803) most significant: high of low * low
v_mul_lo_u32 v22, v19, s32 // (op 34803) least significant: low of low * low
v_add_u32 v23, v23, v18 // (op 34803) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v22 // (op 34803) least significant half
v_addc_co_u32 v19, s[0:1], v21, v23, s[0:1] // (op 34803) most significant half



s_mul_i32 s1, s32, s7 // (op 34805) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 34805) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 34805) least significant
s_mul_hi_u32 s19, s32, s6 // (op 34805) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34805) most significant: sum
s_add_u32 s1, s1, s19 // (op 34805) most significant: sum


s_mov_b32 s64, s34 // (op 34805) reg expression
s_mov_b32 s65, s35 // (op 34805) reg expression
s_mov_b32 s66, s0 // (op 34805) reg expression
s_mov_b32 s67, 131072 // (op 34805) call()

s_mul_hi_i32 s0, s16, s10 // (op 34822) 
v_lshrrev_b32 v20, 3, v1 // (op 34822) 
s_add_i32 s1, s0, s16 // (op 34822) 

v_and_b32 v21, -8, v1 // (op 34822) 
s_ashr_i32 s0, s1, 31 // (op 34822) 
v_sub_u32 v22, v1, v21 // (op 34822) 

s_cmp_eq_i32 s10, 0 // (op 34822) 
s_mov_b32 s18, scc // (op 34822) 
v_lshlrev_b32 v21, 4, v22 // (op 34822) 

s_cmp_eq_i32 s18, 0 // (op 34822) 
s_cselect_b32 s19, 0, -1 // (op 34822) 

v_mov_b32 v23, 0 // (op 34822) convert
v_mov_b32 v22, v21 // (op 34822) convert

s_and_b32 s18, s11, 31 // (op 34822) 
s_lshl_b32 s22, 1, s18 // (op 34822) 
s_add_i32 s23, s22, s19 // (op 34822) 


s_and_b32 s19, s0, s23 // (op 34822) 


s_add_i32 s0, s1, s19 // (op 34822) 


s_ashr_i32 s1, s0, s18 // (op 34822) 


s_lshl_b32 s0, s11, 24 // (op 34822) 
s_ashr_i32 s18, s0, 31 // (op 34822) 

s_xor_b32 s0, s1, s18 // (op 34822) 

s_sub_i32 s1, s0, s18 // (op 34822) 


s_mul_hi_i32 s0, s1, s12 // (op 34822) 
s_add_i32 s18, s0, s1 // (op 34822) 

s_ashr_i32 s0, s18, 31 // (op 34822) 
s_cmp_eq_i32 s12, 0 // (op 34822) 
s_mov_b32 s19, scc // (op 34822) 
s_cmp_eq_i32 s19, 0 // (op 34822) 
s_cselect_b32 s22, 0, -1 // (op 34822) 

s_and_b32 s19, s13, 31 // (op 34822) 
s_lshl_b32 s23, 1, s19 // (op 34822) 
s_add_i32 s24, s23, s22 // (op 34822) 


s_and_b32 s25, s0, s24 // (op 34822) 


s_add_i32 s0, s18, s25 // (op 34822) 


s_ashr_i32 s18, s0, s19 // (op 34822) 


s_lshl_b32 s0, s13, 24 // (op 34822) 
s_ashr_i32 s19, s0, 31 // (op 34822) 

s_xor_b32 s0, s18, s19 // (op 34822) 

s_sub_i32 s18, s0, s19 // (op 34822) 


s_add_u32 s50, s6, 256 // (op 34822) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34822) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34822) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34822) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34822) 

s_mul_i32 s0, s18, s24 // (op 34822) 

s_sub_i32 s18, s1, s0 // (op 34822) 


s_mul_hi_i32 s19, s16, s9 // (op 34822) 
s_add_i32 s0, s19, s16 // (op 34822) 

s_ashr_i32 s1, s0, 31 // (op 34822) 
s_cmp_eq_i32 s9, 0 // (op 34822) 
s_mov_b32 s19, scc // (op 34822) 
s_cmp_eq_i32 s19, 0 // (op 34822) 
s_cselect_b32 s22, 0, -1 // (op 34822) 

s_and_b32 s19, s46, 31 // (op 34822) 
s_lshl_b32 s23, 1, s19 // (op 34822) 
s_add_i32 s49, s23, s22 // (op 34822) 


s_and_b32 s22, s1, s49 // (op 34822) 


s_add_i32 s1, s0, s22 // (op 34822) 


s_ashr_i32 s0, s1, s19 // (op 34822) 


s_lshl_b32 s1, s46, 24 // (op 34822) 
s_ashr_i32 s19, s1, 31 // (op 34822) 

s_xor_b32 s1, s0, s19 // (op 34822) 

s_sub_i32 s0, s1, s19 // (op 34822) 


s_mul_i32 s1, s15, s24 // (op 34822) 

s_mul_i32 s19, s8, s1 // (op 34822) 

s_mul_i32 s1, s0, s19 // (op 34822) 

s_sub_u32 s19, s16, s1 // (op 34822) 

s_mul_hi_i32 s1, s19, s14 // (op 34822) 
s_add_i32 s22, s1, s19 // (op 34822) 


s_ashr_i32 s1, s22, 31 // (op 34822) 
s_cmp_eq_i32 s14, 0 // (op 34822) 
s_mov_b32 s19, scc // (op 34822) 
s_cmp_eq_i32 s19, 0 // (op 34822) 
s_cselect_b32 s23, 0, -1 // (op 34822) 

s_and_b32 s19, s48, 31 // (op 34822) 
s_lshl_b32 s24, 1, s19 // (op 34822) 
s_add_i32 s25, s24, s23 // (op 34822) 


s_and_b32 s23, s1, s25 // (op 34822) 


s_add_i32 s1, s22, s23 // (op 34822) 


s_ashr_i32 s22, s1, s19 // (op 34822) 


s_lshl_b32 s1, s48, 24 // (op 34822) 
s_ashr_i32 s19, s1, 31 // (op 34822) 

s_xor_b32 s1, s22, s19 // (op 34822) 

s_sub_i32 s22, s1, s19 // (op 34822) 


s_cmp_eq_i32 s0, 0 // (op 34822) 
s_mov_b32 s1, scc // (op 34822) 

s_cmp_eq_i32 s1, 0 // (op 34822) 
s_cselect_b32 s0, s22, s18 // (op 34822) 



v_lshl_add_u32 v21, s0, 8, v20 // (op 34822) 


v_mul_lo_u32 v25, v21, s33 // (op 34822) most significant: low of low * high
v_mul_hi_u32 v20, v21, s32 // (op 34822) most significant: high of low * low
v_mul_lo_u32 v24, v21, s32 // (op 34822) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 34822) most significant: sum


v_add_co_u32 v20, s[0:1], v22, v24 // (op 34822) least significant half
v_addc_co_u32 v21, s[0:1], v23, v25, s[0:1] // (op 34822) most significant half



s_mul_i32 s1, s32, s7 // (op 34824) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 34824) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 34824) least significant
s_mul_hi_u32 s19, s32, s6 // (op 34824) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34824) most significant: sum
s_add_u32 s1, s1, s19 // (op 34824) most significant: sum


s_mov_b32 s64, s34 // (op 34824) reg expression
s_mov_b32 s65, s35 // (op 34824) reg expression
s_mov_b32 s66, s0 // (op 34824) reg expression
s_mov_b32 s67, 131072 // (op 34824) call()

s_mul_hi_i32 s0, s16, s10 // (op 34867) 
v_and_b32 v22, -64, v1 // (op 34867) 
s_add_i32 s1, s0, s16 // (op 34867) 

v_sub_u32 v23, v1, v22 // (op 34867) 

s_ashr_i32 s0, s1, 31 // (op 34867) 
v_lshlrev_b32 v22, 2, v23 // (op 34867) 

s_cmp_eq_i32 s10, 0 // (op 34867) 
s_mov_b32 s18, scc // (op 34867) 
v_lshrrev_b32 v23, 6, v1 // (op 34867) 
s_cmp_eq_i32 s18, 0 // (op 34867) 
s_cselect_b32 s19, 0, -1 // (op 34867) 

s_and_b32 s18, s11, 31 // (op 34867) 
s_lshl_b32 s22, 1, s18 // (op 34867) 
s_add_i32 s23, s22, s19 // (op 34867) 


s_and_b32 s19, s0, s23 // (op 34867) 


s_add_i32 s0, s1, s19 // (op 34867) 


s_ashr_i32 s1, s0, s18 // (op 34867) 


s_lshl_b32 s0, s11, 24 // (op 34867) 
s_ashr_i32 s18, s0, 31 // (op 34867) 

s_xor_b32 s0, s1, s18 // (op 34867) 

s_sub_i32 s1, s0, s18 // (op 34867) 


s_mul_hi_i32 s0, s1, s12 // (op 34867) 
s_add_i32 s18, s0, s1 // (op 34867) 

s_ashr_i32 s0, s18, 31 // (op 34867) 
s_cmp_eq_i32 s12, 0 // (op 34867) 
s_mov_b32 s19, scc // (op 34867) 
s_cmp_eq_i32 s19, 0 // (op 34867) 
s_cselect_b32 s22, 0, -1 // (op 34867) 

s_and_b32 s19, s13, 31 // (op 34867) 
s_lshl_b32 s23, 1, s19 // (op 34867) 
s_add_i32 s24, s23, s22 // (op 34867) 


s_and_b32 s25, s0, s24 // (op 34867) 


s_add_i32 s0, s18, s25 // (op 34867) 


s_ashr_i32 s18, s0, s19 // (op 34867) 


s_lshl_b32 s0, s13, 24 // (op 34867) 
s_ashr_i32 s19, s0, 31 // (op 34867) 

s_xor_b32 s0, s18, s19 // (op 34867) 

s_sub_i32 s18, s0, s19 // (op 34867) 


s_mul_i32 s0, s1, s15 // (op 34867) 

s_sub_u32 s1, s16, s0 // (op 34867) 

s_mul_i32 s0, s18, s15 // (op 34867) 
s_add_u32 s0, s0, s1 // (op 34867) 


s_mul_hi_i32 s1, s16, s9 // (op 34867) 
s_add_i32 s18, s1, s16 // (op 34867) 

s_ashr_i32 s1, s18, 31 // (op 34867) 
s_cmp_eq_i32 s9, 0 // (op 34867) 
s_mov_b32 s19, scc // (op 34867) 
s_cmp_eq_i32 s19, 0 // (op 34867) 
s_cselect_b32 s22, 0, -1 // (op 34867) 

s_and_b32 s19, s46, 31 // (op 34867) 
s_lshl_b32 s23, 1, s19 // (op 34867) 
s_add_i32 s24, s23, s22 // (op 34867) 


s_and_b32 s25, s1, s24 // (op 34867) 


s_add_i32 s1, s18, s25 // (op 34867) 


s_ashr_i32 s18, s1, s19 // (op 34867) 


s_lshl_b32 s1, s46, 24 // (op 34867) 
s_ashr_i32 s19, s1, 31 // (op 34867) 

s_xor_b32 s1, s18, s19 // (op 34867) 

s_sub_i32 s18, s1, s19 // (op 34867) 


s_add_u32 s50, s6, 256 // (op 34867) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34867) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34867) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34867) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34867) 

s_mul_i32 s1, s15, s24 // (op 34867) 

s_mul_i32 s19, s8, s1 // (op 34867) 

s_mul_i32 s1, s18, s19 // (op 34867) 

s_sub_u32 s19, s16, s1 // (op 34867) 

s_mov_b32 s1, s19 // (op 34867) 
s_mul_hi_i32 s22, s1, s14 // (op 34867) 
s_add_i32 s23, s22, s1 // (op 34867) 


s_ashr_i32 s1, s23, 31 // (op 34867) 
s_cmp_eq_i32 s14, 0 // (op 34867) 
s_mov_b32 s22, scc // (op 34867) 
s_cmp_eq_i32 s22, 0 // (op 34867) 
s_cselect_b32 s24, 0, -1 // (op 34867) 

s_and_b32 s22, s48, 31 // (op 34867) 
s_lshl_b32 s25, 1, s22 // (op 34867) 
s_add_i32 s49, s25, s24 // (op 34867) 


s_and_b32 s24, s1, s49 // (op 34867) 


s_add_i32 s1, s23, s24 // (op 34867) 


s_ashr_i32 s23, s1, s22 // (op 34867) 


s_lshl_b32 s1, s48, 24 // (op 34867) 
s_ashr_i32 s22, s1, 31 // (op 34867) 

s_xor_b32 s1, s23, s22 // (op 34867) 

s_sub_i32 s23, s1, s22 // (op 34867) 


s_mul_i32 s1, s47, s15 // (op 34867) 
s_add_u32 s24, s20, 256 // (op 34867) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 34867) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 34867) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 34867) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 34867) 

s_sub_i32 s22, s24, s1 // (op 34867) 

s_mul_i32 s24, s23, s22 // (op 34867) 


s_sub_u32 s25, s19, s24 // (op 34867) 


s_add_u32 s19, s25, s1 // (op 34867) 


s_cmp_eq_i32 s18, 0 // (op 34867) 
s_mov_b32 s1, scc // (op 34867) 

s_cmp_eq_i32 s1, 0 // (op 34867) 
s_cselect_b32 s18, s19, s0 // (op 34867) 



v_lshl_add_u32 v24, s18, 8, v22 // (op 34867) 


v_mul_lo_u32 v22, v23, s36 // (op 34867) 

v_add_u32 v23, v24, v22 // (op 34867) 


s_mul_i32 s1, s36, s41 // (op 34869) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 34869) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 34869) least significant
s_mul_hi_u32 s19, s36, s40 // (op 34869) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34869) most significant: sum
s_add_u32 s1, s1, s19 // (op 34869) most significant: sum


s_mov_b32 s68, s38 // (op 34869) reg expression
s_mov_b32 s69, s39 // (op 34869) reg expression
s_mov_b32 s70, s0 // (op 34869) reg expression
s_mov_b32 s71, 131072 // (op 34869) call()

s_mul_hi_i32 s0, s16, s10 // (op 34886) 
v_and_b32 v22, -64, v1 // (op 34886) 
s_add_i32 s1, s0, s16 // (op 34886) 

v_sub_u32 v24, v1, v22 // (op 34886) 

s_ashr_i32 s0, s1, 31 // (op 34886) 
v_lshlrev_b32 v22, 2, v24 // (op 34886) 

s_cmp_eq_i32 s10, 0 // (op 34886) 
s_mov_b32 s18, scc // (op 34886) 
v_lshrrev_b32 v24, 6, v1 // (op 34886) 
s_cmp_eq_i32 s18, 0 // (op 34886) 
s_cselect_b32 s19, 0, -1 // (op 34886) 

s_and_b32 s18, s11, 31 // (op 34886) 
s_lshl_b32 s22, 1, s18 // (op 34886) 
s_add_i32 s23, s22, s19 // (op 34886) 


s_and_b32 s19, s0, s23 // (op 34886) 


s_add_i32 s0, s1, s19 // (op 34886) 


s_ashr_i32 s1, s0, s18 // (op 34886) 


s_lshl_b32 s0, s11, 24 // (op 34886) 
s_ashr_i32 s18, s0, 31 // (op 34886) 

s_xor_b32 s0, s1, s18 // (op 34886) 

s_sub_i32 s1, s0, s18 // (op 34886) 


s_mul_hi_i32 s0, s1, s12 // (op 34886) 
s_add_i32 s18, s0, s1 // (op 34886) 

s_ashr_i32 s0, s18, 31 // (op 34886) 
s_cmp_eq_i32 s12, 0 // (op 34886) 
s_mov_b32 s19, scc // (op 34886) 
s_cmp_eq_i32 s19, 0 // (op 34886) 
s_cselect_b32 s22, 0, -1 // (op 34886) 

s_and_b32 s19, s13, 31 // (op 34886) 
s_lshl_b32 s23, 1, s19 // (op 34886) 
s_add_i32 s24, s23, s22 // (op 34886) 


s_and_b32 s25, s0, s24 // (op 34886) 


s_add_i32 s0, s18, s25 // (op 34886) 


s_ashr_i32 s18, s0, s19 // (op 34886) 


s_lshl_b32 s0, s13, 24 // (op 34886) 
s_ashr_i32 s19, s0, 31 // (op 34886) 

s_xor_b32 s0, s18, s19 // (op 34886) 

s_sub_i32 s18, s0, s19 // (op 34886) 


s_mul_i32 s0, s1, s15 // (op 34886) 

s_sub_u32 s1, s16, s0 // (op 34886) 

s_mul_i32 s0, s18, s15 // (op 34886) 
s_add_u32 s0, s0, s1 // (op 34886) 


s_mul_hi_i32 s1, s16, s9 // (op 34886) 
s_add_i32 s18, s1, s16 // (op 34886) 

s_ashr_i32 s1, s18, 31 // (op 34886) 
s_cmp_eq_i32 s9, 0 // (op 34886) 
s_mov_b32 s19, scc // (op 34886) 
s_cmp_eq_i32 s19, 0 // (op 34886) 
s_cselect_b32 s22, 0, -1 // (op 34886) 

s_and_b32 s19, s46, 31 // (op 34886) 
s_lshl_b32 s23, 1, s19 // (op 34886) 
s_add_i32 s24, s23, s22 // (op 34886) 


s_and_b32 s25, s1, s24 // (op 34886) 


s_add_i32 s1, s18, s25 // (op 34886) 


s_ashr_i32 s18, s1, s19 // (op 34886) 


s_lshl_b32 s1, s46, 24 // (op 34886) 
s_ashr_i32 s19, s1, 31 // (op 34886) 

s_xor_b32 s1, s18, s19 // (op 34886) 

s_sub_i32 s18, s1, s19 // (op 34886) 


s_add_u32 s50, s6, 256 // (op 34886) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34886) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34886) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34886) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34886) 

s_mul_i32 s1, s15, s24 // (op 34886) 

s_mul_i32 s19, s8, s1 // (op 34886) 

s_mul_i32 s1, s18, s19 // (op 34886) 

s_sub_u32 s19, s16, s1 // (op 34886) 

s_mov_b32 s1, s19 // (op 34886) 
s_mul_hi_i32 s22, s1, s14 // (op 34886) 
s_add_i32 s23, s22, s1 // (op 34886) 


s_ashr_i32 s1, s23, 31 // (op 34886) 
s_cmp_eq_i32 s14, 0 // (op 34886) 
s_mov_b32 s22, scc // (op 34886) 
s_cmp_eq_i32 s22, 0 // (op 34886) 
s_cselect_b32 s24, 0, -1 // (op 34886) 

s_and_b32 s22, s48, 31 // (op 34886) 
s_lshl_b32 s25, 1, s22 // (op 34886) 
s_add_i32 s49, s25, s24 // (op 34886) 


s_and_b32 s24, s1, s49 // (op 34886) 


s_add_i32 s1, s23, s24 // (op 34886) 


s_ashr_i32 s23, s1, s22 // (op 34886) 


s_lshl_b32 s1, s48, 24 // (op 34886) 
s_ashr_i32 s22, s1, 31 // (op 34886) 

s_xor_b32 s1, s23, s22 // (op 34886) 

s_sub_i32 s23, s1, s22 // (op 34886) 


s_mul_i32 s1, s47, s15 // (op 34886) 
s_add_u32 s24, s20, 256 // (op 34886) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 34886) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 34886) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 34886) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 34886) 

s_sub_i32 s22, s24, s1 // (op 34886) 

s_mul_i32 s24, s23, s22 // (op 34886) 


s_sub_u32 s25, s19, s24 // (op 34886) 


s_add_u32 s19, s25, s1 // (op 34886) 


s_cmp_eq_i32 s18, 0 // (op 34886) 
s_mov_b32 s1, scc // (op 34886) 

s_cmp_eq_i32 s1, 0 // (op 34886) 
s_cselect_b32 s18, s19, s0 // (op 34886) 



v_lshl_add_u32 v25, s18, 8, v22 // (op 34886) 


v_mul_lo_u32 v22, v24, s36 // (op 34886) 

v_add_u32 v24, v25, v22 // (op 34886) 


s_mul_i32 s1, s36, s41 // (op 34888) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 34888) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 34888) least significant
s_mul_hi_u32 s19, s36, s40 // (op 34888) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34888) most significant: sum
s_add_u32 s1, s1, s19 // (op 34888) most significant: sum


s_mov_b32 s68, s38 // (op 34888) reg expression
s_mov_b32 s69, s39 // (op 34888) reg expression
s_mov_b32 s70, s0 // (op 34888) reg expression
s_mov_b32 s71, 131072 // (op 34888) call()

s_mul_hi_i32 s0, s16, s10 // (op 34931) 
s_add_i32 s1, s0, s16 // (op 34931) 

s_ashr_i32 s0, s1, 31 // (op 34931) 
s_cmp_eq_i32 s10, 0 // (op 34931) 
s_mov_b32 s18, scc // (op 34931) 
s_cmp_eq_i32 s18, 0 // (op 34931) 
s_cselect_b32 s19, 0, -1 // (op 34931) 

s_and_b32 s18, s11, 31 // (op 34931) 
s_lshl_b32 s22, 1, s18 // (op 34931) 
s_add_i32 s23, s22, s19 // (op 34931) 


s_and_b32 s19, s0, s23 // (op 34931) 


s_add_i32 s0, s1, s19 // (op 34931) 


s_ashr_i32 s1, s0, s18 // (op 34931) 


s_lshl_b32 s0, s11, 24 // (op 34931) 
s_ashr_i32 s18, s0, 31 // (op 34931) 

s_xor_b32 s0, s1, s18 // (op 34931) 

s_sub_i32 s1, s0, s18 // (op 34931) 


s_mul_hi_i32 s0, s1, s12 // (op 34931) 
s_add_i32 s18, s0, s1 // (op 34931) 

s_ashr_i32 s0, s18, 31 // (op 34931) 
s_cmp_eq_i32 s12, 0 // (op 34931) 
s_mov_b32 s19, scc // (op 34931) 
s_cmp_eq_i32 s19, 0 // (op 34931) 
s_cselect_b32 s22, 0, -1 // (op 34931) 

s_and_b32 s19, s13, 31 // (op 34931) 
s_lshl_b32 s23, 1, s19 // (op 34931) 
s_add_i32 s24, s23, s22 // (op 34931) 


s_and_b32 s25, s0, s24 // (op 34931) 


s_add_i32 s0, s18, s25 // (op 34931) 


s_ashr_i32 s18, s0, s19 // (op 34931) 


s_lshl_b32 s0, s13, 24 // (op 34931) 
s_ashr_i32 s19, s0, 31 // (op 34931) 

s_xor_b32 s0, s18, s19 // (op 34931) 

s_sub_i32 s18, s0, s19 // (op 34931) 


s_add_u32 s50, s6, 256 // (op 34931) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34931) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34931) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34931) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34931) 

s_mul_i32 s0, s18, s24 // (op 34931) 

s_sub_i32 s18, s1, s0 // (op 34931) 


s_mul_hi_i32 s19, s16, s9 // (op 34931) 
s_add_i32 s0, s19, s16 // (op 34931) 

s_ashr_i32 s1, s0, 31 // (op 34931) 
s_cmp_eq_i32 s9, 0 // (op 34931) 
s_mov_b32 s19, scc // (op 34931) 
s_cmp_eq_i32 s19, 0 // (op 34931) 
s_cselect_b32 s22, 0, -1 // (op 34931) 

s_and_b32 s19, s46, 31 // (op 34931) 
s_lshl_b32 s23, 1, s19 // (op 34931) 
s_add_i32 s49, s23, s22 // (op 34931) 


s_and_b32 s22, s1, s49 // (op 34931) 


s_add_i32 s1, s0, s22 // (op 34931) 


s_ashr_i32 s0, s1, s19 // (op 34931) 


s_lshl_b32 s1, s46, 24 // (op 34931) 
s_ashr_i32 s19, s1, 31 // (op 34931) 

s_xor_b32 s1, s0, s19 // (op 34931) 

s_sub_i32 s0, s1, s19 // (op 34931) 


s_mul_i32 s1, s15, s24 // (op 34931) 

s_mul_i32 s19, s8, s1 // (op 34931) 

s_mul_i32 s1, s0, s19 // (op 34931) 

s_sub_u32 s19, s16, s1 // (op 34931) 

s_mul_hi_i32 s1, s19, s14 // (op 34931) 
s_add_i32 s22, s1, s19 // (op 34931) 


s_ashr_i32 s1, s22, 31 // (op 34931) 
s_cmp_eq_i32 s14, 0 // (op 34931) 
s_mov_b32 s19, scc // (op 34931) 
s_cmp_eq_i32 s19, 0 // (op 34931) 
s_cselect_b32 s23, 0, -1 // (op 34931) 

s_and_b32 s19, s48, 31 // (op 34931) 
s_lshl_b32 s24, 1, s19 // (op 34931) 
s_add_i32 s25, s24, s23 // (op 34931) 


s_and_b32 s23, s1, s25 // (op 34931) 


s_add_i32 s1, s22, s23 // (op 34931) 


s_ashr_i32 s22, s1, s19 // (op 34931) 


s_lshl_b32 s1, s48, 24 // (op 34931) 
s_ashr_i32 s19, s1, 31 // (op 34931) 

s_xor_b32 s1, s22, s19 // (op 34931) 

s_sub_i32 s22, s1, s19 // (op 34931) 


s_cmp_eq_i32 s0, 0 // (op 34931) 
s_mov_b32 s1, scc // (op 34931) 

s_cmp_eq_i32 s1, 0 // (op 34931) 
s_cselect_b32 s0, s22, s18 // (op 34931) 



v_lshl_add_u32 v22, s0, 8, v1 // (op 34931) 

v_mul_lo_u32 v25, v22, s42 // (op 34931) 

s_mul_i32 s1, s42, s7 // (op 34933) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 34933) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 34933) least significant
s_mul_hi_u32 s19, s42, s6 // (op 34933) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34933) most significant: sum
s_add_u32 s1, s1, s19 // (op 34933) most significant: sum


s_mov_b32 s72, s44 // (op 34933) reg expression
s_mov_b32 s73, s45 // (op 34933) reg expression
s_mov_b32 s74, s0 // (op 34933) reg expression
s_mov_b32 s75, 131072 // (op 34933) call()

s_mul_hi_i32 s0, s16, s10 // (op 34950) 
s_add_i32 s1, s0, s16 // (op 34950) 

s_ashr_i32 s0, s1, 31 // (op 34950) 
s_cmp_eq_i32 s10, 0 // (op 34950) 
s_mov_b32 s18, scc // (op 34950) 
s_cmp_eq_i32 s18, 0 // (op 34950) 
s_cselect_b32 s19, 0, -1 // (op 34950) 

s_and_b32 s18, s11, 31 // (op 34950) 
s_lshl_b32 s22, 1, s18 // (op 34950) 
s_add_i32 s23, s22, s19 // (op 34950) 


s_and_b32 s19, s0, s23 // (op 34950) 


s_add_i32 s0, s1, s19 // (op 34950) 


s_ashr_i32 s1, s0, s18 // (op 34950) 


s_lshl_b32 s0, s11, 24 // (op 34950) 
s_ashr_i32 s18, s0, 31 // (op 34950) 

s_xor_b32 s0, s1, s18 // (op 34950) 

s_sub_i32 s1, s0, s18 // (op 34950) 


s_mul_hi_i32 s0, s1, s12 // (op 34950) 
s_add_i32 s18, s0, s1 // (op 34950) 

s_ashr_i32 s0, s18, 31 // (op 34950) 
s_cmp_eq_i32 s12, 0 // (op 34950) 
s_mov_b32 s19, scc // (op 34950) 
s_cmp_eq_i32 s19, 0 // (op 34950) 
s_cselect_b32 s22, 0, -1 // (op 34950) 

s_and_b32 s19, s13, 31 // (op 34950) 
s_lshl_b32 s23, 1, s19 // (op 34950) 
s_add_i32 s24, s23, s22 // (op 34950) 


s_and_b32 s25, s0, s24 // (op 34950) 


s_add_i32 s0, s18, s25 // (op 34950) 


s_ashr_i32 s18, s0, s19 // (op 34950) 


s_lshl_b32 s0, s13, 24 // (op 34950) 
s_ashr_i32 s19, s0, 31 // (op 34950) 

s_xor_b32 s0, s18, s19 // (op 34950) 

s_sub_i32 s18, s0, s19 // (op 34950) 


s_add_u32 s50, s6, 256 // (op 34950) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34950) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34950) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34950) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34950) 

s_mul_i32 s0, s18, s24 // (op 34950) 

s_sub_i32 s18, s1, s0 // (op 34950) 


s_mul_hi_i32 s19, s16, s9 // (op 34950) 
s_add_i32 s0, s19, s16 // (op 34950) 

s_ashr_i32 s1, s0, 31 // (op 34950) 
s_cmp_eq_i32 s9, 0 // (op 34950) 
s_mov_b32 s19, scc // (op 34950) 
s_cmp_eq_i32 s19, 0 // (op 34950) 
s_cselect_b32 s22, 0, -1 // (op 34950) 

s_and_b32 s19, s46, 31 // (op 34950) 
s_lshl_b32 s23, 1, s19 // (op 34950) 
s_add_i32 s49, s23, s22 // (op 34950) 


s_and_b32 s22, s1, s49 // (op 34950) 


s_add_i32 s1, s0, s22 // (op 34950) 


s_ashr_i32 s0, s1, s19 // (op 34950) 


s_lshl_b32 s1, s46, 24 // (op 34950) 
s_ashr_i32 s19, s1, 31 // (op 34950) 

s_xor_b32 s1, s0, s19 // (op 34950) 

s_sub_i32 s0, s1, s19 // (op 34950) 


s_mul_i32 s1, s15, s24 // (op 34950) 

s_mul_i32 s19, s8, s1 // (op 34950) 

s_mul_i32 s1, s0, s19 // (op 34950) 

s_sub_u32 s19, s16, s1 // (op 34950) 

s_mul_hi_i32 s1, s19, s14 // (op 34950) 
s_add_i32 s22, s1, s19 // (op 34950) 


s_ashr_i32 s1, s22, 31 // (op 34950) 
s_cmp_eq_i32 s14, 0 // (op 34950) 
s_mov_b32 s19, scc // (op 34950) 
s_cmp_eq_i32 s19, 0 // (op 34950) 
s_cselect_b32 s23, 0, -1 // (op 34950) 

s_and_b32 s19, s48, 31 // (op 34950) 
s_lshl_b32 s24, 1, s19 // (op 34950) 
s_add_i32 s25, s24, s23 // (op 34950) 


s_and_b32 s23, s1, s25 // (op 34950) 


s_add_i32 s1, s22, s23 // (op 34950) 


s_ashr_i32 s22, s1, s19 // (op 34950) 


s_lshl_b32 s1, s48, 24 // (op 34950) 
s_ashr_i32 s19, s1, 31 // (op 34950) 

s_xor_b32 s1, s22, s19 // (op 34950) 

s_sub_i32 s22, s1, s19 // (op 34950) 


s_cmp_eq_i32 s0, 0 // (op 34950) 
s_mov_b32 s1, scc // (op 34950) 

s_cmp_eq_i32 s1, 0 // (op 34950) 
s_cselect_b32 s0, s22, s18 // (op 34950) 



v_lshl_add_u32 v22, s0, 8, v1 // (op 34950) 

v_mul_lo_u32 v26, v22, s42 // (op 34950) 

s_mul_i32 s1, s42, s7 // (op 34952) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 34952) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 34952) least significant
s_mul_hi_u32 s19, s42, s6 // (op 34952) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34952) most significant: sum
s_add_u32 s1, s1, s19 // (op 34952) most significant: sum


s_mov_b32 s72, s44 // (op 34952) reg expression
s_mov_b32 s73, s45 // (op 34952) reg expression
s_mov_b32 s74, s0 // (op 34952) reg expression
s_mov_b32 s75, 131072 // (op 34952) call()

v_and_b32 v22, -8, v1 // (op 35078) 
v_sub_u32 v27, v1, v22 // (op 35078) 

v_lshlrev_b32 v22, 4, v27 // (op 35078) 

v_lshrrev_b32 v27, 3, v1 // (op 35078) 
v_lshl_add_u32 v28, v27, 7, v22 // (op 35078) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 35078) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[32:33], 5 // (op 34625) 
v_mov_b64 v[28:29], s[18:19] // (op 34625) convert


v_mov_b64 v[30:31], v[18:19] // (op 34625) 
s_mov_b32 m0, s0 // (op 34625) reg expression
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34625) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34625) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34625) most significant half

s_add_u32 m0, m0, 4096 // (op 34625) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34625) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34625) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34625) most significant half

s_add_u32 m0, m0, 4096 // (op 34625) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34625) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34625) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34625) most significant half

s_add_u32 m0, m0, 4096 // (op 34625) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34625) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34625) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34625) most significant half

s_add_u32 m0, m0, 4096 // (op 34625) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34625) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34625) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34625) most significant half

s_add_u32 m0, m0, 4096 // (op 34625) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34625) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34625) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34625) most significant half

s_add_u32 m0, m0, 4096 // (op 34625) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34625) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34625) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34625) most significant half

s_add_u32 m0, m0, 4096 // (op 34625) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 34625) Load value direct to lds






buffer_load_dword v22, v25, s[72:75], 0 offen // (op 34023) Load value
v_and_b32 v27, -16, v1 // (op 35021) 
v_sub_u32 v28, v1, v27 // (op 35021) 

v_lshlrev_b32 v27, 4, v28 // (op 35021) 

v_lshrrev_b32 v28, 4, v1 // (op 35021) 
v_lshl_add_u32 v29, v28, 8, v27 // (op 35021) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 35021) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 4 // (op 34630) 
v_mov_b64 v[28:29], s[18:19] // (op 34630) convert


v_mov_b64 v[30:31], v[14:15] // (op 34630) reg expression
s_add_u32 s1, 32768, s0 // (op 34630) 
s_mov_b32 m0, s1 // (op 34630) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34630) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34630) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34630) most significant half

s_add_u32 m0, m0, 4096 // (op 34630) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34630) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34630) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34630) most significant half

s_add_u32 m0, m0, 4096 // (op 34630) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34630) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34630) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34630) most significant half

s_add_u32 m0, m0, 4096 // (op 34630) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34630) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34630) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34630) most significant half

s_add_u32 m0, m0, 4096 // (op 34630) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34630) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34630) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34630) most significant half

s_add_u32 m0, m0, 4096 // (op 34630) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34630) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34630) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34630) most significant half

s_add_u32 m0, m0, 4096 // (op 34630) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34630) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 34630) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34630) most significant half

s_add_u32 m0, m0, 4096 // (op 34630) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34630) Load value direct to lds






v_mov_b32 v27, v23 // (op 34029) reg expression

buffer_load_dword v28, v27, s[68:71], 0 offen // (op 34029) Load value

v_and_b32 v27, -8, v1 // (op 35305) 
v_sub_u32 v29, v1, v27 // (op 35305) 

v_lshlrev_b32 v27, 4, v29 // (op 35305) 

v_lshrrev_b32 v29, 3, v1 // (op 35305) 
v_lshl_add_u32 v30, v29, 7, v27 // (op 35305) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 35305) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[32:33], 5 // (op 34635) 
v_mov_b64 v[30:31], s[18:19] // (op 34635) convert


v_mov_b64 v[32:33], v[18:19] // (op 34635) 
s_add_u32 s1, 65536, s0 // (op 34635) 
s_mov_b32 m0, s1 // (op 34635) reg expression
s_mov_b32 s18, 128 // (op 34635) call()
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34635) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34635) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34635) most significant half

s_add_u32 m0, m0, 4096 // (op 34635) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34635) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34635) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34635) most significant half

s_add_u32 m0, m0, 4096 // (op 34635) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34635) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34635) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34635) most significant half

s_add_u32 m0, m0, 4096 // (op 34635) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34635) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34635) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34635) most significant half

s_add_u32 m0, m0, 4096 // (op 34635) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34635) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34635) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34635) most significant half

s_add_u32 m0, m0, 4096 // (op 34635) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34635) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34635) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34635) most significant half

s_add_u32 m0, m0, 4096 // (op 34635) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34635) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 34635) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 34635) most significant half

s_add_u32 m0, m0, 4096 // (op 34635) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 34635) Load value direct to lds









buffer_load_dword v27, v25, s[72:75], 0 offen offset:4 // (op 34035) Load value

v_and_b32 v25, -16, v1 // (op 35261) 
v_sub_u32 v18, v1, v25 // (op 35261) 

v_lshlrev_b32 v19, 4, v18 // (op 35261) 

v_lshrrev_b32 v18, 4, v1 // (op 35261) 
v_lshl_add_u32 v25, v18, 8, v19 // (op 35261) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 35261) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 4 // (op 34637) 
v_mov_b64 v[18:19], s[18:19] // (op 34637) convert


s_lshl_b64 s[18:19], s[26:27], 7 // (op 34637) 
v_mov_b32 v25, s18 // (op 34637) 
v_mov_b32 v29, s19 // (op 34637) 
v_add_co_u32 v30, s[50:51], v14, v25 // (op 34637) least significant half
v_addc_co_u32 v31, s[50:51], v15, v29, s[50:51] // (op 34637) most significant half




s_add_u32 s1, 98304, s0 // (op 34637) 
s_mov_b32 m0, s1 // (op 34637) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34637) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 34637) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34637) most significant half

s_add_u32 m0, m0, 4096 // (op 34637) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34637) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 34637) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34637) most significant half

s_add_u32 m0, m0, 4096 // (op 34637) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34637) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 34637) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34637) most significant half

s_add_u32 m0, m0, 4096 // (op 34637) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34637) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 34637) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34637) most significant half

s_add_u32 m0, m0, 4096 // (op 34637) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34637) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 34637) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34637) most significant half

s_add_u32 m0, m0, 4096 // (op 34637) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34637) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 34637) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34637) most significant half

s_add_u32 m0, m0, 4096 // (op 34637) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34637) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 34637) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 34637) most significant half

s_add_u32 m0, m0, 4096 // (op 34637) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 34637) Load value direct to lds







s_lshl_b32 s0, s36, 2 // (op 34041) 
v_add_u32 v25, v23, s0 // (op 34041) 


buffer_load_dword v14, v25, s[68:71], 0 offen // (op 34041) Load value


v_lshlrev_b32 v15, 2, v1 // (op 35194) 



v_add_u32 v23, 131072, v15 // (op 34044) 
s_waitcnt vmcnt(27)
ds_write_b32 v23, v22 // (op 34044) Store local data 


v_and_b32 v15, -64, v1 // (op 35136) 
v_sub_u32 v23, v1, v15 // (op 35136) 

v_lshlrev_b32 v15, 2, v23 // (op 35136) 

v_lshrrev_b32 v23, 6, v1 // (op 35136) 
v_lshl_add_u32 v25, v23, 8, v15 // (op 35136) 





v_add_u32 v15, 132096, v25 // (op 34046) 
s_waitcnt vmcnt(18)
ds_write_b32 v15, v28 // (op 34046) Store local data 


s_barrier  // (op 35689) 



ds_read_b128 v[30:33], v4 // (op 34101) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 34101) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 35723) 



v_add_u32 v15, 131072, v5 // (op 34181) 
ds_read_u8 v23, v15 // (op 34181) Load local data 




ds_read_b128 v[38:41], v4 offset:4096 // (op 34106) Load local data 
ds_read_b128 v[42:45], v4 offset:4160 // (op 34106) Load local data 



v_add_u32 v15, 131200, v5 // (op 34186) 
ds_read_u8 v25, v15 // (op 34186) Load local data 




ds_read_b128 v[46:49], v4 offset:8192 // (op 34111) Load local data 
ds_read_b128 v[50:53], v4 offset:8256 // (op 34111) Load local data 



v_add_u32 v15, 131328, v5 // (op 34191) 
ds_read_u8 v29, v15 // (op 34191) Load local data 




ds_read_b128 v[54:57], v4 offset:12288 // (op 34116) Load local data 
ds_read_b128 v[58:61], v4 offset:12352 // (op 34116) Load local data 



v_add_u32 v15, 131456, v5 // (op 34196) 
ds_read_u8 v18, v15 // (op 34196) Load local data 




ds_read_b128 v[62:65], v4 offset:16384 // (op 34121) Load local data 
ds_read_b128 v[66:69], v4 offset:16448 // (op 34121) Load local data 



v_add_u32 v15, 131584, v5 // (op 34201) 
ds_read_u8 v19, v15 // (op 34201) Load local data 




ds_read_b128 v[70:73], v4 offset:20480 // (op 34126) Load local data 
ds_read_b128 v[74:77], v4 offset:20544 // (op 34126) Load local data 



v_add_u32 v15, 131712, v5 // (op 34206) 
ds_read_u8 v78, v15 // (op 34206) Load local data 




ds_read_b128 v[80:83], v4 offset:24576 // (op 34131) Load local data 
ds_read_b128 v[84:87], v4 offset:24640 // (op 34131) Load local data 



v_add_u32 v15, 131840, v5 // (op 34211) 
ds_read_u8 v79, v15 // (op 34211) Load local data 




ds_read_b128 v[88:91], v4 offset:28672 // (op 34136) Load local data 
ds_read_b128 v[92:95], v4 offset:28736 // (op 34136) Load local data 



v_add_u32 v15, 131968, v5 // (op 34216) 
ds_read_u8 v96, v15 // (op 34216) Load local data 

s_barrier  // (op 35673) 



ds_read_b64_tr_b8 v[98:99], v0 offset:32768 // (op 34061) Transpose load local data 
ds_read_b64_tr_b8 v[100:101], v0 offset:34816 // (op 34061) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v0 offset:49152 // (op 34061) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v0 offset:51200 // (op 34061) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 35706) 


v_add_u32 v15, 132096, v3 // (op 34141) 
ds_read_u8 v97, v15 // (op 34141) Load local data 




ds_read_b64_tr_b8 v[106:107], v0 offset:32800 // (op 34066) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v0 offset:34848 // (op 34066) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v0 offset:49184 // (op 34066) Transpose load local data 
ds_read_b64_tr_b8 v[112:113], v0 offset:51232 // (op 34066) Transpose load local data 


v_add_u32 v15, 132128, v3 // (op 34146) 
ds_read_u8 v114, v15 // (op 34146) Load local data 




ds_read_b64_tr_b8 v[116:117], v0 offset:32832 // (op 34071) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v0 offset:34880 // (op 34071) Transpose load local data 
ds_read_b64_tr_b8 v[120:121], v0 offset:49216 // (op 34071) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v0 offset:51264 // (op 34071) Transpose load local data 


v_add_u32 v15, 132160, v3 // (op 34151) 
ds_read_u8 v115, v15 // (op 34151) Load local data 




ds_read_b64_tr_b8 v[124:125], v0 offset:32864 // (op 34076) Transpose load local data 
ds_read_b64_tr_b8 v[126:127], v0 offset:34912 // (op 34076) Transpose load local data 
ds_read_b64_tr_b8 v[128:129], v0 offset:49248 // (op 34076) Transpose load local data 
ds_read_b64_tr_b8 v[130:131], v0 offset:51296 // (op 34076) Transpose load local data 


v_add_u32 v15, 132192, v3 // (op 34156) 
ds_read_u8 v132, v15 // (op 34156) Load local data 




ds_read_b64_tr_b8 v[134:135], v0 offset:32896 // (op 34081) Transpose load local data 
ds_read_b64_tr_b8 v[136:137], v0 offset:34944 // (op 34081) Transpose load local data 
ds_read_b64_tr_b8 v[138:139], v0 offset:49280 // (op 34081) Transpose load local data 
ds_read_b64_tr_b8 v[140:141], v0 offset:51328 // (op 34081) Transpose load local data 


v_add_u32 v15, 132224, v3 // (op 34161) 
ds_read_u8 v133, v15 // (op 34161) Load local data 




ds_read_b64_tr_b8 v[142:143], v0 offset:32928 // (op 34086) Transpose load local data 
ds_read_b64_tr_b8 v[144:145], v0 offset:34976 // (op 34086) Transpose load local data 
ds_read_b64_tr_b8 v[146:147], v0 offset:49312 // (op 34086) Transpose load local data 
ds_read_b64_tr_b8 v[148:149], v0 offset:51360 // (op 34086) Transpose load local data 


v_add_u32 v15, 132256, v3 // (op 34166) 
ds_read_u8 v150, v15 // (op 34166) Load local data 




ds_read_b64_tr_b8 v[152:153], v0 offset:32960 // (op 34091) Transpose load local data 
ds_read_b64_tr_b8 v[154:155], v0 offset:35008 // (op 34091) Transpose load local data 
ds_read_b64_tr_b8 v[156:157], v0 offset:49344 // (op 34091) Transpose load local data 
ds_read_b64_tr_b8 v[158:159], v0 offset:51392 // (op 34091) Transpose load local data 


v_add_u32 v15, 132288, v3 // (op 34171) 
ds_read_u8 v151, v15 // (op 34171) Load local data 




ds_read_b64_tr_b8 v[160:161], v0 offset:32992 // (op 34096) Transpose load local data 
ds_read_b64_tr_b8 v[162:163], v0 offset:35040 // (op 34096) Transpose load local data 
ds_read_b64_tr_b8 v[164:165], v0 offset:49376 // (op 34096) Transpose load local data 
ds_read_b64_tr_b8 v[166:167], v0 offset:51424 // (op 34096) Transpose load local data 


v_add_u32 v15, 132320, v3 // (op 34176) 
ds_read_u8 v168, v15 // (op 34176) Load local data 

v_and_b32 v15, -16, v1 // (op 34993) 
v_sub_u32 v169, v1, v15 // (op 34993) 

v_lshlrev_b32 v15, 4, v169 // (op 34993) 

v_lshrrev_b32 v169, 4, v1 // (op 34993) 
v_lshl_add_u32 v170, v169, 8, v15 // (op 34993) 


s_nop 0
v_readfirstlane_b32 s0, v170 // (op 34993) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -8, v1 // (op 35050) 
v_sub_u32 v169, v1, v15 // (op 35050) 

v_lshlrev_b32 v15, 4, v169 // (op 35050) 

v_lshrrev_b32 v169, 3, v1 // (op 35050) 
v_lshl_add_u32 v170, v169, 7, v15 // (op 35050) 


s_nop 0
v_readfirstlane_b32 s1, v170 // (op 35050) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -64, v1 // (op 35107) 
v_sub_u32 v169, v1, v15 // (op 35107) 

v_lshlrev_b32 v15, 2, v169 // (op 35107) 

v_lshrrev_b32 v169, 6, v1 // (op 35107) 
v_lshl_add_u32 v170, v169, 8, v15 // (op 35107) 


v_lshlrev_b32 v15, 2, v1 // (op 35165) 
v_and_b32 v169, -16, v1 // (op 35246) 
v_sub_u32 v171, v1, v169 // (op 35246) 

v_lshlrev_b32 v169, 4, v171 // (op 35246) 

v_lshrrev_b32 v171, 4, v1 // (op 35246) 
v_lshl_add_u32 v172, v171, 8, v169 // (op 35246) 


s_nop 0
v_readfirstlane_b32 s18, v172 // (op 35246) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v169, -8, v1 // (op 35290) 
v_sub_u32 v171, v1, v169 // (op 35290) 

v_lshlrev_b32 v169, 4, v171 // (op 35290) 

v_lshrrev_b32 v171, 3, v1 // (op 35290) 
v_lshl_add_u32 v172, v171, 7, v169 // (op 35290) 


s_nop 0
v_readfirstlane_b32 s19, v172 // (op 35290) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v169, -64, v1 // (op 35334) 
v_sub_u32 v171, v1, v169 // (op 35334) 

v_lshlrev_b32 v169, 2, v171 // (op 35334) 

v_lshrrev_b32 v171, 6, v1 // (op 35334) 
v_lshl_add_u32 v172, v171, 8, v169 // (op 35334) 


v_lshlrev_b32 v169, 2, v1 // (op 35363) 
s_mov_b32 s22, 0 // (op 9653) call()
s_ashr_i64 s[24:25], s[30:31], 7 // (op 9652) 
s_mov_b32 s23, s24 // (op 9652) convert

s_and_b32 s24, s23, 4294967294 // (op 9652) 

s_cmp_lt_u32 s22, s24 // (op 9652) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_9652 // (op 9652) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_9652 if false)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_9652:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:105], v[30:37], a[0:3], v97, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 88) 
s_barrier  // (op 35692) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:105], v[38:45], a[4:7], v97, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 376) 
s_add_u32 s22, s22, 2 // (op 9654) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[46:53], a[8:11], v97, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 552) 
s_lshl_b64 s[24:25], s[32:33], 5 // (op 34597) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[54:61], a[12:15], v97, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 728) 
v_mov_b64 v[174:175], s[24:25] // (op 34597) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[62:69], a[16:19], v97, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 904) 
v_mov_b64 v[176:177], v[20:21] // (op 34597) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[70:77], a[20:23], v97, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1080) 
v_mov_b64 v[178:179], v[176:177] // (op 34597) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[80:87], a[24:27], v97, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1256) 
s_mov_b32 m0, s1 // (op 34597) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[88:95], a[28:31], v97, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1432) 
s_mov_b32 s23, 256 // (op 34597) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[106:113], v[30:37], a[32:35], v114, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2323) 
buffer_load_dwordx4 v178, s[64:67], s23 offen offset: 0 lds // (op 34597) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[106:113], v[38:45], a[36:39], v114, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2277) 
v_add_co_u32 v178, s[24:25], v178, v174 // (op 34597) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[106:113], v[46:53], a[40:43], v114, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2231) 
v_addc_co_u32 v179, s[24:25], 0, v179, s[24:25] // (op 34597) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[106:113], v[54:61], a[44:47], v114, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2185) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[106:113], v[62:69], a[48:51], v114, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2139) 
s_add_u32 m0, m0, 4096 // (op 34597) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[106:113], v[70:77], a[52:55], v114, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2093) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[80:87], a[56:59], v114, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2047) 
buffer_load_dwordx4 v178, s[64:67], s23 offen offset: 0 lds // (op 34597) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[88:95], a[60:63], v114, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2001) 
v_add_co_u32 v178, s[24:25], v178, v174 // (op 34597) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[116:123], v[30:37], a[64:67], v115, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3861) 
v_addc_co_u32 v179, s[24:25], 0, v179, s[24:25] // (op 34597) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:123], v[38:45], a[68:71], v115, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3815) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[116:123], v[46:53], a[72:75], v115, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3769) 
s_add_u32 m0, m0, 4096 // (op 34597) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:123], v[54:61], a[76:79], v115, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3723) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:123], v[62:69], a[80:83], v115, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3677) 
buffer_load_dwordx4 v178, s[64:67], s23 offen offset: 0 lds // (op 34597) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[116:123], v[70:77], a[84:87], v115, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3631) 
v_add_co_u32 v178, s[24:25], v178, v174 // (op 34597) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[116:123], v[80:87], a[88:91], v115, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3585) 
v_addc_co_u32 v179, s[24:25], 0, v179, s[24:25] // (op 34597) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:123], v[88:95], a[92:95], v115, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3539) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:131], v[30:37], a[96:99], v132, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5399) 
s_add_u32 m0, m0, 4096 // (op 34597) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[124:131], v[38:45], a[100:103], v132, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5353) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[124:131], v[46:53], a[104:107], v132, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5307) 
buffer_load_dwordx4 v178, s[64:67], s23 offen offset: 0 lds // (op 34597) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[124:131], v[54:61], a[108:111], v132, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5261) 
v_add_co_u32 v178, s[24:25], v178, v174 // (op 34597) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:131], v[62:69], a[112:115], v132, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5215) 
v_addc_co_u32 v179, s[24:25], 0, v179, s[24:25] // (op 34597) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:131], v[70:77], a[116:119], v132, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5169) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:131], v[80:87], a[120:123], v132, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5123) 
s_add_u32 m0, m0, 4096 // (op 34597) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:131], v[88:95], a[124:127], v132, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5077) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[134:141], v[30:37], a[128:131], v133, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6937) 
buffer_load_dwordx4 v178, s[64:67], s23 offen offset: 0 lds // (op 34597) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[134:141], v[38:45], a[132:135], v133, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6891) 
v_add_co_u32 v178, s[24:25], v178, v174 // (op 34597) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[134:141], v[46:53], a[136:139], v133, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6845) 
v_addc_co_u32 v179, s[24:25], 0, v179, s[24:25] // (op 34597) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[134:141], v[54:61], a[140:143], v133, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6799) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:141], v[62:69], a[144:147], v133, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6753) 
s_add_u32 m0, m0, 4096 // (op 34597) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:141], v[70:77], a[148:151], v133, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6707) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:141], v[80:87], a[152:155], v133, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6661) 
buffer_load_dwordx4 v178, s[64:67], s23 offen offset: 0 lds // (op 34597) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:141], v[88:95], a[156:159], v133, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6615) 
v_add_co_u32 v178, s[24:25], v178, v174 // (op 34597) least significant half
v_addc_co_u32 v179, s[24:25], 0, v179, s[24:25] // (op 34597) most significant half

s_add_u32 m0, m0, 4096 // (op 34597) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:149], v[30:37], a[160:163], v150, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8475) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:149], v[38:45], a[164:167], v150, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8429) 
buffer_load_dwordx4 v178, s[64:67], s23 offen offset: 0 lds // (op 34597) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:149], v[46:53], a[168:171], v150, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8383) 
v_add_co_u32 v178, s[24:25], v178, v174 // (op 34597) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:149], v[54:61], a[172:175], v150, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8337) 
v_addc_co_u32 v179, s[24:25], 0, v179, s[24:25] // (op 34597) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[142:149], v[62:69], a[176:179], v150, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8291) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[142:149], v[70:77], a[180:183], v150, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8245) 
s_add_u32 m0, m0, 4096 // (op 34597) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[142:149], v[80:87], a[184:187], v150, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8199) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[142:149], v[88:95], a[188:191], v150, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8153) 
buffer_load_dwordx4 v178, s[64:67], s23 offen offset: 0 lds // (op 34597) Load value direct to lds





s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[152:159], v[30:37], a[192:195], v151, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 10013) 
v_mov_b32 v171, v26 // (op 25) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[152:159], v[38:45], a[196:199], v151, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9967) 
buffer_load_dword v22, v171, s[72:75], 0 offen offset:8 // (op 25) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[152:159], v[46:53], a[200:203], v151, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9921) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 35676) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[152:159], v[54:61], a[204:207], v151, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9875) 
s_lshl_b64 s[24:25], s[26:27], 4 // (op 34592) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[152:159], v[62:69], a[208:211], v151, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9829) 
v_mov_b64 v[174:175], s[24:25] // (op 34592) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[152:159], v[70:77], a[212:215], v151, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9783) 
v_mov_b64 v[178:179], v[16:17] // (op 34592) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[152:159], v[80:87], a[216:219], v151, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9737) 
s_lshl_b64 s[24:25], s[26:27], 7 // (op 34592) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[152:159], v[88:95], a[220:223], v151, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9691) 
s_lshl_b64 s[50:51], s[24:25], 1 // (op 34592) 

v_mov_b32 v173, s50 // (op 34592) 
v_mov_b32 v180, s51 // (op 34592) 
v_add_co_u32 v182, s[24:25], v178, v173 // (op 34592) least significant half
v_addc_co_u32 v183, s[24:25], v179, v180, s[24:25] // (op 34592) most significant half




s_add_u32 s23, 32768, s0 // (op 34592) 
s_mov_b32 m0, s23 // (op 34592) reg expression
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[160:167], v[30:37], a[224:227], v168, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11551) 
buffer_load_dwordx4 v182, s[52:55], 0 offen offset: 0 lds // (op 34592) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[160:167], v[38:45], a[228:231], v168, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11505) 
v_add_co_u32 v182, s[24:25], v182, v174 // (op 34592) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[160:167], v[46:53], a[232:235], v168, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11459) 
v_addc_co_u32 v183, s[24:25], 0, v183, s[24:25] // (op 34592) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[160:167], v[54:61], a[236:239], v168, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11413) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:167], v[62:69], a[240:243], v168, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11367) 
s_add_u32 m0, m0, 4096 // (op 34592) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[160:167], v[70:77], a[244:247], v168, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11321) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[160:167], v[80:87], a[248:251], v168, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11275) 
buffer_load_dwordx4 v182, s[52:55], 0 offen offset: 0 lds // (op 34592) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[160:167], v[88:95], a[252:255], v168, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11229) 
v_add_co_u32 v182, s[24:25], v182, v174 // (op 34592) least significant half
v_addc_co_u32 v183, s[24:25], 0, v183, s[24:25] // (op 34592) most significant half

s_add_u32 m0, m0, 4096 // (op 34592) 
buffer_load_dwordx4 v182, s[52:55], 0 offen offset: 0 lds // (op 34592) Load value direct to lds
v_add_co_u32 v182, s[24:25], v182, v174 // (op 34592) least significant half
v_addc_co_u32 v183, s[24:25], 0, v183, s[24:25] // (op 34592) most significant half

s_add_u32 m0, m0, 4096 // (op 34592) 
buffer_load_dwordx4 v182, s[52:55], 0 offen offset: 0 lds // (op 34592) Load value direct to lds
v_add_co_u32 v182, s[24:25], v182, v174 // (op 34592) least significant half
v_addc_co_u32 v183, s[24:25], 0, v183, s[24:25] // (op 34592) most significant half

s_add_u32 m0, m0, 4096 // (op 34592) 
buffer_load_dwordx4 v182, s[52:55], 0 offen offset: 0 lds // (op 34592) Load value direct to lds
v_add_co_u32 v182, s[24:25], v182, v174 // (op 34592) least significant half
v_addc_co_u32 v183, s[24:25], 0, v183, s[24:25] // (op 34592) most significant half

s_add_u32 m0, m0, 4096 // (op 34592) 
buffer_load_dwordx4 v182, s[52:55], 0 offen offset: 0 lds // (op 34592) Load value direct to lds
v_add_co_u32 v182, s[24:25], v182, v174 // (op 34592) least significant half
v_addc_co_u32 v183, s[24:25], 0, v183, s[24:25] // (op 34592) most significant half

s_add_u32 m0, m0, 4096 // (op 34592) 
buffer_load_dwordx4 v182, s[52:55], 0 offen offset: 0 lds // (op 34592) Load value direct to lds
v_add_co_u32 v182, s[24:25], v182, v174 // (op 34592) least significant half
v_addc_co_u32 v183, s[24:25], 0, v183, s[24:25] // (op 34592) most significant half

s_add_u32 m0, m0, 4096 // (op 34592) 
buffer_load_dwordx4 v182, s[52:55], 0 offen offset: 0 lds // (op 34592) Load value direct to lds





v_mov_b32 v173, v24 // (op 17) 
s_lshl_b32 s23, s36, 3 // (op 17) 
v_add_u32 v174, v173, s23 // (op 17) 


buffer_load_dword v28, v174, s[68:71], 0 offen // (op 17) Load value

s_barrier  // (op 35762) 



v_add_u32 v174, 133120, v169 // (op 255) 
s_waitcnt vmcnt(27)
ds_write_b32 v174, v27 // (op 255) Store local data 

s_barrier  // (op 35756) 



v_add_u32 v174, 134144, v172 // (op 254) 
s_waitcnt vmcnt(18)
ds_write_b32 v174, v14 // (op 254) Store local data 

s_barrier  // (op 35746) 



v_add_u32 v174, 65536, v8 // (op 247) 
ds_read_b128 v[180:183], v174 // (op 247) Load local data 

v_add_u32 v174, 65600, v8 // (op 247) 
ds_read_b128 v[184:187], v174 // (op 247) Load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 35759) 



v_add_u32 v174, 133120, v9 // (op 251) 
ds_read_u8 v175, v174 // (op 251) Load local data 




v_add_u32 v174, 69632, v8 // (op 381) 
ds_read_b128 v[188:191], v174 // (op 381) Load local data 

v_add_u32 v174, 69696, v8 // (op 381) 
ds_read_b128 v[192:195], v174 // (op 381) Load local data 




v_add_u32 v174, 133248, v9 // (op 385) 
ds_read_u8 v114, v174 // (op 385) Load local data 




v_add_u32 v174, 73728, v8 // (op 557) 
ds_read_b128 v[196:199], v174 // (op 557) Load local data 

v_add_u32 v174, 73792, v8 // (op 557) 
ds_read_b128 v[200:203], v174 // (op 557) Load local data 




v_add_u32 v174, 133376, v9 // (op 561) 
ds_read_u8 v115, v174 // (op 561) Load local data 




v_add_u32 v174, 77824, v8 // (op 733) 
ds_read_b128 v[204:207], v174 // (op 733) Load local data 

v_add_u32 v174, 77888, v8 // (op 733) 
ds_read_b128 v[208:211], v174 // (op 733) Load local data 




v_add_u32 v174, 133504, v9 // (op 737) 
ds_read_u8 v132, v174 // (op 737) Load local data 




v_add_u32 v174, 81920, v8 // (op 909) 
ds_read_b128 v[212:215], v174 // (op 909) Load local data 

v_add_u32 v174, 81984, v8 // (op 909) 
ds_read_b128 v[216:219], v174 // (op 909) Load local data 




v_add_u32 v174, 133632, v9 // (op 913) 
ds_read_u8 v133, v174 // (op 913) Load local data 




v_add_u32 v174, 86016, v8 // (op 1085) 
ds_read_b128 v[220:223], v174 // (op 1085) Load local data 

v_add_u32 v174, 86080, v8 // (op 1085) 
ds_read_b128 v[224:227], v174 // (op 1085) Load local data 




v_add_u32 v174, 133760, v9 // (op 1089) 
ds_read_u8 v150, v174 // (op 1089) Load local data 




v_add_u32 v174, 90112, v8 // (op 1261) 
ds_read_b128 v[228:231], v174 // (op 1261) Load local data 

v_add_u32 v174, 90176, v8 // (op 1261) 
ds_read_b128 v[232:235], v174 // (op 1261) Load local data 




v_add_u32 v174, 133888, v9 // (op 1265) 
ds_read_u8 v151, v174 // (op 1265) Load local data 




v_add_u32 v174, 94208, v8 // (op 1437) 
ds_read_b128 v[236:239], v174 // (op 1437) Load local data 

v_add_u32 v174, 94272, v8 // (op 1437) 
ds_read_b128 v[240:243], v174 // (op 1437) Load local data 




v_add_u32 v174, 134016, v9 // (op 1441) 
ds_read_u8 v244, v174 // (op 1441) Load local data 

s_barrier  // (op 35740) 
s_barrier  // (op 35803) 



v_add_u32 v174, 98304, v6 // (op 245) 
ds_read_b64_tr_b8 v[98:99], v174 // (op 245) Transpose load local data 

v_add_u32 v174, 100352, v6 // (op 245) 
ds_read_b64_tr_b8 v[100:101], v174 // (op 245) Transpose load local data 

v_add_u32 v174, 114688, v6 // (op 245) 
ds_read_b64_tr_b8 v[102:103], v174 // (op 245) Transpose load local data 

v_add_u32 v174, 116736, v6 // (op 245) 
ds_read_b64_tr_b8 v[104:105], v174 // (op 245) Transpose load local data 

s_waitcnt lgkmcnt(15)
s_barrier  // (op 35753) 


v_add_u32 v174, 134144, v7 // (op 249) 
ds_read_u8 v97, v174 // (op 249) Load local data 




v_add_u32 v174, 98336, v6 // (op 2004) 
ds_read_b64_tr_b8 v[106:107], v174 // (op 2004) Transpose load local data 

v_add_u32 v174, 100384, v6 // (op 2004) 
ds_read_b64_tr_b8 v[108:109], v174 // (op 2004) Transpose load local data 

v_add_u32 v174, 114720, v6 // (op 2004) 
ds_read_b64_tr_b8 v[110:111], v174 // (op 2004) Transpose load local data 

v_add_u32 v174, 116768, v6 // (op 2004) 
ds_read_b64_tr_b8 v[112:113], v174 // (op 2004) Transpose load local data 



v_add_u32 v174, 134176, v7 // (op 2007) 
ds_read_u8 v25, v174 // (op 2007) Load local data 




v_add_u32 v174, 98368, v6 // (op 3542) 
ds_read_b64_tr_b8 v[116:117], v174 // (op 3542) Transpose load local data 

v_add_u32 v174, 100416, v6 // (op 3542) 
ds_read_b64_tr_b8 v[118:119], v174 // (op 3542) Transpose load local data 

v_add_u32 v174, 114752, v6 // (op 3542) 
ds_read_b64_tr_b8 v[120:121], v174 // (op 3542) Transpose load local data 

v_add_u32 v174, 116800, v6 // (op 3542) 
ds_read_b64_tr_b8 v[122:123], v174 // (op 3542) Transpose load local data 



v_add_u32 v174, 134208, v7 // (op 3545) 
ds_read_u8 v29, v174 // (op 3545) Load local data 




v_add_u32 v174, 98400, v6 // (op 5080) 
ds_read_b64_tr_b8 v[124:125], v174 // (op 5080) Transpose load local data 

v_add_u32 v174, 100448, v6 // (op 5080) 
ds_read_b64_tr_b8 v[126:127], v174 // (op 5080) Transpose load local data 

v_add_u32 v174, 114784, v6 // (op 5080) 
ds_read_b64_tr_b8 v[128:129], v174 // (op 5080) Transpose load local data 

v_add_u32 v174, 116832, v6 // (op 5080) 
ds_read_b64_tr_b8 v[130:131], v174 // (op 5080) Transpose load local data 



v_add_u32 v174, 134240, v7 // (op 5083) 
ds_read_u8 v18, v174 // (op 5083) Load local data 




v_add_u32 v174, 98432, v6 // (op 6618) 
ds_read_b64_tr_b8 v[134:135], v174 // (op 6618) Transpose load local data 

v_add_u32 v174, 100480, v6 // (op 6618) 
ds_read_b64_tr_b8 v[136:137], v174 // (op 6618) Transpose load local data 

v_add_u32 v174, 114816, v6 // (op 6618) 
ds_read_b64_tr_b8 v[138:139], v174 // (op 6618) Transpose load local data 

v_add_u32 v174, 116864, v6 // (op 6618) 
ds_read_b64_tr_b8 v[140:141], v174 // (op 6618) Transpose load local data 



v_add_u32 v174, 134272, v7 // (op 6621) 
ds_read_u8 v19, v174 // (op 6621) Load local data 




v_add_u32 v174, 98464, v6 // (op 8156) 
ds_read_b64_tr_b8 v[142:143], v174 // (op 8156) Transpose load local data 

v_add_u32 v174, 100512, v6 // (op 8156) 
ds_read_b64_tr_b8 v[144:145], v174 // (op 8156) Transpose load local data 

v_add_u32 v174, 114848, v6 // (op 8156) 
ds_read_b64_tr_b8 v[146:147], v174 // (op 8156) Transpose load local data 

v_add_u32 v174, 116896, v6 // (op 8156) 
ds_read_b64_tr_b8 v[148:149], v174 // (op 8156) Transpose load local data 



v_add_u32 v174, 134304, v7 // (op 8159) 
ds_read_u8 v78, v174 // (op 8159) Load local data 




v_add_u32 v174, 98496, v6 // (op 9694) 
ds_read_b64_tr_b8 v[152:153], v174 // (op 9694) Transpose load local data 

v_add_u32 v174, 100544, v6 // (op 9694) 
ds_read_b64_tr_b8 v[154:155], v174 // (op 9694) Transpose load local data 

v_add_u32 v174, 114880, v6 // (op 9694) 
ds_read_b64_tr_b8 v[156:157], v174 // (op 9694) Transpose load local data 

v_add_u32 v174, 116928, v6 // (op 9694) 
ds_read_b64_tr_b8 v[158:159], v174 // (op 9694) Transpose load local data 



v_add_u32 v174, 134336, v7 // (op 9697) 
ds_read_u8 v79, v174 // (op 9697) Load local data 




v_add_u32 v174, 98528, v6 // (op 11232) 
ds_read_b64_tr_b8 v[160:161], v174 // (op 11232) Transpose load local data 

v_add_u32 v174, 100576, v6 // (op 11232) 
ds_read_b64_tr_b8 v[162:163], v174 // (op 11232) Transpose load local data 

v_add_u32 v174, 114912, v6 // (op 11232) 
ds_read_b64_tr_b8 v[164:165], v174 // (op 11232) Transpose load local data 

v_add_u32 v174, 116960, v6 // (op 11232) 
ds_read_b64_tr_b8 v[166:167], v174 // (op 11232) Transpose load local data 



v_add_u32 v174, 134368, v7 // (op 11235) 
ds_read_u8 v168, v174 // (op 11235) Load local data 

s_barrier  // (op 35861) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:105], v[180:187], a[0:3], v97, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 252) 
s_barrier  // (op 35750) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:105], v[188:195], a[4:7], v97, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 377) 
s_lshl_b64 s[24:25], s[32:33], 5 // (op 34620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[196:203], a[8:11], v97, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 553) 
v_mov_b64 v[246:247], s[24:25] // (op 34620) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[204:211], a[12:15], v97, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 729) 
v_mov_b64 v[248:249], v[176:177] // (op 34620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[212:219], a[16:19], v97, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 905) 
s_add_u32 s23, 65536, s19 // (op 34620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[220:227], a[20:23], v97, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1081) 
s_mov_b32 m0, s23 // (op 34620) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[228:235], a[24:27], v97, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1257) 
s_mov_b32 s24, 384 // (op 34620) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[236:243], a[28:31], v97, v244 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1433) 
buffer_load_dwordx4 v248, s[64:67], s24 offen offset: 0 lds // (op 34620) Load value direct to lds
v_add_co_u32 v248, s[50:51], v248, v246 // (op 34620) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[106:113], v[180:187], a[32:35], v25, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2324) 
v_addc_co_u32 v249, s[50:51], 0, v249, s[50:51] // (op 34620) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[106:113], v[188:195], a[36:39], v25, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2278) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[106:113], v[196:203], a[40:43], v25, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2232) 
s_add_u32 m0, m0, 4096 // (op 34620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[106:113], v[204:211], a[44:47], v25, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2186) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[106:113], v[212:219], a[48:51], v25, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2140) 
buffer_load_dwordx4 v248, s[64:67], s24 offen offset: 0 lds // (op 34620) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[106:113], v[220:227], a[52:55], v25, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2094) 
v_add_co_u32 v248, s[50:51], v248, v246 // (op 34620) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[228:235], a[56:59], v25, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2048) 
v_addc_co_u32 v249, s[50:51], 0, v249, s[50:51] // (op 34620) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[236:243], a[60:63], v25, v244 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 2002) 
s_add_u32 m0, m0, 4096 // (op 34620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[116:123], v[180:187], a[64:67], v29, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3862) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:123], v[188:195], a[68:71], v29, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3816) 
buffer_load_dwordx4 v248, s[64:67], s24 offen offset: 0 lds // (op 34620) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[116:123], v[196:203], a[72:75], v29, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3770) 
v_add_co_u32 v248, s[50:51], v248, v246 // (op 34620) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:123], v[204:211], a[76:79], v29, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3724) 
v_addc_co_u32 v249, s[50:51], 0, v249, s[50:51] // (op 34620) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:123], v[212:219], a[80:83], v29, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3678) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[116:123], v[220:227], a[84:87], v29, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3632) 
s_add_u32 m0, m0, 4096 // (op 34620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[116:123], v[228:235], a[88:91], v29, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3586) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:123], v[236:243], a[92:95], v29, v244 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3540) 
buffer_load_dwordx4 v248, s[64:67], s24 offen offset: 0 lds // (op 34620) Load value direct to lds
v_add_co_u32 v248, s[50:51], v248, v246 // (op 34620) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:131], v[180:187], a[96:99], v18, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5400) 
v_addc_co_u32 v249, s[50:51], 0, v249, s[50:51] // (op 34620) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[124:131], v[188:195], a[100:103], v18, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5354) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[124:131], v[196:203], a[104:107], v18, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5308) 
s_add_u32 m0, m0, 4096 // (op 34620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[124:131], v[204:211], a[108:111], v18, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5262) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:131], v[212:219], a[112:115], v18, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5216) 
buffer_load_dwordx4 v248, s[64:67], s24 offen offset: 0 lds // (op 34620) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:131], v[220:227], a[116:119], v18, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5170) 
v_add_co_u32 v248, s[50:51], v248, v246 // (op 34620) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:131], v[228:235], a[120:123], v18, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5124) 
v_addc_co_u32 v249, s[50:51], 0, v249, s[50:51] // (op 34620) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:131], v[236:243], a[124:127], v18, v244 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5078) 
s_add_u32 m0, m0, 4096 // (op 34620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[134:141], v[180:187], a[128:131], v19, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6938) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[134:141], v[188:195], a[132:135], v19, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6892) 
buffer_load_dwordx4 v248, s[64:67], s24 offen offset: 0 lds // (op 34620) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[134:141], v[196:203], a[136:139], v19, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6846) 
v_add_co_u32 v248, s[50:51], v248, v246 // (op 34620) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[134:141], v[204:211], a[140:143], v19, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6800) 
v_addc_co_u32 v249, s[50:51], 0, v249, s[50:51] // (op 34620) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:141], v[212:219], a[144:147], v19, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6754) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:141], v[220:227], a[148:151], v19, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6708) 
s_add_u32 m0, m0, 4096 // (op 34620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:141], v[228:235], a[152:155], v19, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6662) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:141], v[236:243], a[156:159], v19, v244 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6616) 
buffer_load_dwordx4 v248, s[64:67], s24 offen offset: 0 lds // (op 34620) Load value direct to lds
v_add_co_u32 v248, s[50:51], v248, v246 // (op 34620) least significant half
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:149], v[180:187], a[160:163], v78, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8476) 
v_addc_co_u32 v249, s[50:51], 0, v249, s[50:51] // (op 34620) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:149], v[188:195], a[164:167], v78, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8430) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:149], v[196:203], a[168:171], v78, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8384) 
s_add_u32 m0, m0, 4096 // (op 34620) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:149], v[204:211], a[172:175], v78, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8338) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[142:149], v[212:219], a[176:179], v78, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8292) 
buffer_load_dwordx4 v248, s[64:67], s24 offen offset: 0 lds // (op 34620) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[142:149], v[220:227], a[180:183], v78, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8246) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[142:149], v[228:235], a[184:187], v78, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8200) 
v_add_co_u32 v20, vcc, 256, v20 // (op 34814) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[142:149], v[236:243], a[188:191], v78, v244 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8154) 
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 34814) most significant half
buffer_load_dword v27, v171, s[72:75], 0 offen offset:12 // (op 258) Load value

s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[152:159], v[180:187], a[192:195], v79, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 10014) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[152:159], v[188:195], a[196:199], v79, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9968) 
v_add_u32 v26, 8, v26 // (op 34942) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[152:159], v[196:203], a[200:203], v79, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9922) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 35743) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[152:159], v[204:211], a[204:207], v79, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9876) 
s_lshl_b64 s[24:25], s[26:27], 4 // (op 34615) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[152:159], v[212:219], a[208:211], v79, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9830) 
v_mov_b64 v[176:177], s[24:25] // (op 34615) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[152:159], v[220:227], a[212:215], v79, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9784) 
s_lshl_b64 s[24:25], s[26:27], 7 // (op 34615) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[152:159], v[228:235], a[216:219], v79, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9738) 
s_mul_i32 s51, 3, s25 // (op 34615) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[152:159], v[236:243], a[220:223], v79, v244 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9692) 
s_mul_i32 s50, 3, s24 // (op 34615) least significant
s_mul_hi_u32 s23, 3, s24 // (op 34615) most significant: high of low * low
s_add_u32 s51, s51, s23 // (op 34615) most significant: sum


v_mov_b32 v171, s50 // (op 34615) 
v_mov_b32 v174, s51 // (op 34615) 
v_add_co_u32 v246, s[24:25], v178, v171 // (op 34615) least significant half
v_addc_co_u32 v247, s[24:25], v179, v174, s[24:25] // (op 34615) most significant half




s_add_u32 s23, 98304, s18 // (op 34615) 
s_mov_b32 m0, s23 // (op 34615) reg expression
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[160:167], v[180:187], a[224:227], v168, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11552) 
buffer_load_dwordx4 v246, s[52:55], 0 offen offset: 0 lds // (op 34615) Load value direct to lds


v_add_co_u32 v246, s[24:25], v246, v176 // (op 34615) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[160:167], v[188:195], a[228:231], v168, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11506) 
v_addc_co_u32 v247, s[24:25], 0, v247, s[24:25] // (op 34615) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[160:167], v[196:203], a[232:235], v168, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11460) 
s_add_u32 m0, m0, 4096 // (op 34615) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[160:167], v[204:211], a[236:239], v168, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11414) 
buffer_load_dwordx4 v246, s[52:55], 0 offen offset: 0 lds // (op 34615) Load value direct to lds

v_add_co_u32 v246, s[24:25], v246, v176 // (op 34615) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:167], v[212:219], a[240:243], v168, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11368) 
v_addc_co_u32 v247, s[24:25], 0, v247, s[24:25] // (op 34615) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[160:167], v[220:227], a[244:247], v168, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11322) 
s_add_u32 m0, m0, 4096 // (op 34615) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[160:167], v[228:235], a[248:251], v168, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11276) 
buffer_load_dwordx4 v246, s[52:55], 0 offen offset: 0 lds // (op 34615) Load value direct to lds

v_add_co_u32 v246, s[24:25], v246, v176 // (op 34615) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[160:167], v[236:243], a[252:255], v168, v244 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11230) 
v_addc_co_u32 v247, s[24:25], 0, v247, s[24:25] // (op 34615) most significant half



s_add_u32 m0, m0, 4096 // (op 34615) 
buffer_load_dwordx4 v246, s[52:55], 0 offen offset: 0 lds // (op 34615) Load value direct to lds
v_add_co_u32 v246, s[24:25], v246, v176 // (op 34615) least significant half
v_addc_co_u32 v247, s[24:25], 0, v247, s[24:25] // (op 34615) most significant half

s_add_u32 m0, m0, 4096 // (op 34615) 
buffer_load_dwordx4 v246, s[52:55], 0 offen offset: 0 lds // (op 34615) Load value direct to lds
v_add_co_u32 v246, s[24:25], v246, v176 // (op 34615) least significant half
v_addc_co_u32 v247, s[24:25], 0, v247, s[24:25] // (op 34615) most significant half

s_add_u32 m0, m0, 4096 // (op 34615) 
buffer_load_dwordx4 v246, s[52:55], 0 offen offset: 0 lds // (op 34615) Load value direct to lds
v_add_co_u32 v246, s[24:25], v246, v176 // (op 34615) least significant half
v_addc_co_u32 v247, s[24:25], 0, v247, s[24:25] // (op 34615) most significant half

s_add_u32 m0, m0, 4096 // (op 34615) 
buffer_load_dwordx4 v246, s[52:55], 0 offen offset: 0 lds // (op 34615) Load value direct to lds
v_add_co_u32 v246, s[24:25], v246, v176 // (op 34615) least significant half
v_addc_co_u32 v247, s[24:25], 0, v247, s[24:25] // (op 34615) most significant half

s_add_u32 m0, m0, 4096 // (op 34615) 
buffer_load_dwordx4 v246, s[52:55], 0 offen offset: 0 lds // (op 34615) Load value direct to lds






s_lshl_b32 s23, s36, 2 // (op 257) 
s_mul_i32 s24, 3, s23 // (op 257) 

v_add_u32 v171, v173, s24 // (op 257) 


s_lshl_b64 s[24:25], s[26:27], 7 // (op 34750) 
s_lshl_b64 s[50:51], s[24:25], 1 // (op 34750) 

v_mov_b32 v174, s50 // (op 34750) 
v_mov_b32 v175, s51 // (op 34750) 
v_add_co_u32 v16, s[24:25], v16, v174 // (op 34750) least significant half
v_addc_co_u32 v17, s[24:25], v17, v175, s[24:25] // (op 34750) most significant half




buffer_load_dword v14, v171, s[68:71], 0 offen // (op 257) Load value


s_barrier  // (op 35726) 
s_lshl_b32 s23, s36, 3 // (op 34878) 



v_add_u32 v24, v24, s23 // (op 34878) 

v_add_u32 v171, 131072, v15 // (op 30) 
s_waitcnt vmcnt(27)
ds_write_b32 v171, v22 // (op 30) Store local data 

s_barrier  // (op 35709) 



v_add_u32 v171, 132096, v170 // (op 21) 
s_waitcnt vmcnt(18)
ds_write_b32 v171, v28 // (op 21) Store local data 

s_barrier  // (op 35698) 



ds_read_b128 v[30:33], v4 // (op 75) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 35733) 



v_add_u32 v171, 131072, v5 // (op 85) 
ds_read_u8 v23, v171 // (op 85) Load local data 




ds_read_b128 v[38:41], v4 offset:4096 // (op 371) Load local data 
ds_read_b128 v[42:45], v4 offset:4160 // (op 371) Load local data 



v_add_u32 v171, 131200, v5 // (op 375) 
ds_read_u8 v25, v171 // (op 375) Load local data 




ds_read_b128 v[46:49], v4 offset:8192 // (op 547) Load local data 
ds_read_b128 v[50:53], v4 offset:8256 // (op 547) Load local data 



v_add_u32 v171, 131328, v5 // (op 551) 
ds_read_u8 v29, v171 // (op 551) Load local data 




ds_read_b128 v[54:57], v4 offset:12288 // (op 723) Load local data 
ds_read_b128 v[58:61], v4 offset:12352 // (op 723) Load local data 



v_add_u32 v171, 131456, v5 // (op 727) 
ds_read_u8 v18, v171 // (op 727) Load local data 




ds_read_b128 v[62:65], v4 offset:16384 // (op 899) Load local data 
ds_read_b128 v[66:69], v4 offset:16448 // (op 899) Load local data 



v_add_u32 v171, 131584, v5 // (op 903) 
ds_read_u8 v19, v171 // (op 903) Load local data 




ds_read_b128 v[70:73], v4 offset:20480 // (op 1075) Load local data 
ds_read_b128 v[74:77], v4 offset:20544 // (op 1075) Load local data 



v_add_u32 v171, 131712, v5 // (op 1079) 
ds_read_u8 v78, v171 // (op 1079) Load local data 




ds_read_b128 v[80:83], v4 offset:24576 // (op 1251) Load local data 
ds_read_b128 v[84:87], v4 offset:24640 // (op 1251) Load local data 



v_add_u32 v171, 131840, v5 // (op 1255) 
ds_read_u8 v79, v171 // (op 1255) Load local data 




ds_read_b128 v[88:91], v4 offset:28672 // (op 1427) Load local data 
ds_read_b128 v[92:95], v4 offset:28736 // (op 1427) Load local data 



v_add_u32 v171, 131968, v5 // (op 1431) 
ds_read_u8 v96, v171 // (op 1431) Load local data 

s_barrier  // (op 35682) 



ds_read_b64_tr_b8 v[98:99], v0 offset:32768 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[100:101], v0 offset:34816 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v0 offset:49152 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v0 offset:51200 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 35716) 


v_add_u32 v171, 132096, v3 // (op 80) 
ds_read_u8 v97, v171 // (op 80) Load local data 




ds_read_b64_tr_b8 v[106:107], v0 offset:32800 // (op 1996) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v0 offset:34848 // (op 1996) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v0 offset:49184 // (op 1996) Transpose load local data 
ds_read_b64_tr_b8 v[112:113], v0 offset:51232 // (op 1996) Transpose load local data 


v_add_u32 v171, 132128, v3 // (op 1999) 
ds_read_u8 v114, v171 // (op 1999) Load local data 




ds_read_b64_tr_b8 v[116:117], v0 offset:32832 // (op 3534) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v0 offset:34880 // (op 3534) Transpose load local data 
ds_read_b64_tr_b8 v[120:121], v0 offset:49216 // (op 3534) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v0 offset:51264 // (op 3534) Transpose load local data 


v_add_u32 v171, 132160, v3 // (op 3537) 
ds_read_u8 v115, v171 // (op 3537) Load local data 




ds_read_b64_tr_b8 v[124:125], v0 offset:32864 // (op 5072) Transpose load local data 
ds_read_b64_tr_b8 v[126:127], v0 offset:34912 // (op 5072) Transpose load local data 
ds_read_b64_tr_b8 v[128:129], v0 offset:49248 // (op 5072) Transpose load local data 
ds_read_b64_tr_b8 v[130:131], v0 offset:51296 // (op 5072) Transpose load local data 


v_add_u32 v171, 132192, v3 // (op 5075) 
ds_read_u8 v132, v171 // (op 5075) Load local data 




ds_read_b64_tr_b8 v[134:135], v0 offset:32896 // (op 6610) Transpose load local data 
ds_read_b64_tr_b8 v[136:137], v0 offset:34944 // (op 6610) Transpose load local data 
ds_read_b64_tr_b8 v[138:139], v0 offset:49280 // (op 6610) Transpose load local data 
ds_read_b64_tr_b8 v[140:141], v0 offset:51328 // (op 6610) Transpose load local data 


v_add_u32 v171, 132224, v3 // (op 6613) 
ds_read_u8 v133, v171 // (op 6613) Load local data 




ds_read_b64_tr_b8 v[142:143], v0 offset:32928 // (op 8148) Transpose load local data 
ds_read_b64_tr_b8 v[144:145], v0 offset:34976 // (op 8148) Transpose load local data 
ds_read_b64_tr_b8 v[146:147], v0 offset:49312 // (op 8148) Transpose load local data 
ds_read_b64_tr_b8 v[148:149], v0 offset:51360 // (op 8148) Transpose load local data 


v_add_u32 v171, 132256, v3 // (op 8151) 
ds_read_u8 v150, v171 // (op 8151) Load local data 




ds_read_b64_tr_b8 v[152:153], v0 offset:32960 // (op 9686) Transpose load local data 
ds_read_b64_tr_b8 v[154:155], v0 offset:35008 // (op 9686) Transpose load local data 
ds_read_b64_tr_b8 v[156:157], v0 offset:49344 // (op 9686) Transpose load local data 
ds_read_b64_tr_b8 v[158:159], v0 offset:51392 // (op 9686) Transpose load local data 


v_add_u32 v171, 132288, v3 // (op 9689) 
ds_read_u8 v151, v171 // (op 9689) Load local data 




ds_read_b64_tr_b8 v[160:161], v0 offset:32992 // (op 11224) Transpose load local data 
ds_read_b64_tr_b8 v[162:163], v0 offset:35040 // (op 11224) Transpose load local data 
ds_read_b64_tr_b8 v[164:165], v0 offset:49376 // (op 11224) Transpose load local data 
ds_read_b64_tr_b8 v[166:167], v0 offset:51424 // (op 11224) Transpose load local data 


v_add_u32 v171, 132320, v3 // (op 11227) 
ds_read_u8 v168, v171 // (op 11227) Load local data 

s_ashr_i64 s[24:25], s[30:31], 7 // (op 9652) 
s_mov_b32 s23, s24 // (op 9652) convert

s_and_b32 s24, s23, 4294967294 // (op 9652) 

s_cmp_lt_u32 s22, s24 // (op 9652) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_9652 // (op 9652) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_9652 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_9652:









































s_barrier  // (op 35827) 
s_mul_hi_i32 s0, s16, s10 // (op 34650) 
v_and_b32 v15, -16, v1 // (op 34650) 
s_add_i32 s1, s0, s16 // (op 34650) 

v_sub_u32 v16, v1, v15 // (op 34650) 

s_ashr_i32 s0, s1, 31 // (op 34650) 
v_lshlrev_b32 v15, 4, v16 // (op 34650) 

s_cmp_eq_i32 s10, 0 // (op 34650) 
s_mov_b32 s18, scc // (op 34650) 
v_lshrrev_b32 v16, 4, v1 // (op 34650) 
s_cmp_eq_i32 s18, 0 // (op 34650) 
s_cselect_b32 s19, 0, -1 // (op 34650) 

s_and_b32 s18, s11, 31 // (op 34650) 
s_lshl_b32 s22, 1, s18 // (op 34650) 
s_add_i32 s23, s22, s19 // (op 34650) 


s_and_b32 s19, s0, s23 // (op 34650) 


s_add_i32 s0, s1, s19 // (op 34650) 


s_ashr_i32 s1, s0, s18 // (op 34650) 


s_lshl_b32 s0, s11, 24 // (op 34650) 
s_ashr_i32 s18, s0, 31 // (op 34650) 

s_xor_b32 s0, s1, s18 // (op 34650) 

s_sub_i32 s1, s0, s18 // (op 34650) 


s_mul_hi_i32 s0, s1, s12 // (op 34650) 
s_add_i32 s18, s0, s1 // (op 34650) 

s_ashr_i32 s0, s18, 31 // (op 34650) 
s_cmp_eq_i32 s12, 0 // (op 34650) 
s_mov_b32 s19, scc // (op 34650) 
s_cmp_eq_i32 s19, 0 // (op 34650) 
s_cselect_b32 s22, 0, -1 // (op 34650) 

s_and_b32 s19, s13, 31 // (op 34650) 
s_lshl_b32 s23, 1, s19 // (op 34650) 
s_add_i32 s24, s23, s22 // (op 34650) 


s_and_b32 s25, s0, s24 // (op 34650) 


s_add_i32 s0, s18, s25 // (op 34650) 


s_ashr_i32 s18, s0, s19 // (op 34650) 


s_lshl_b32 s0, s13, 24 // (op 34650) 
s_ashr_i32 s19, s0, 31 // (op 34650) 

s_xor_b32 s0, s18, s19 // (op 34650) 

s_sub_i32 s18, s0, s19 // (op 34650) 


s_mul_i32 s0, s1, s15 // (op 34650) 

s_sub_u32 s1, s16, s0 // (op 34650) 

s_mul_i32 s0, s18, s15 // (op 34650) 
s_add_u32 s0, s0, s1 // (op 34650) 


s_mul_hi_i32 s1, s16, s9 // (op 34650) 
s_add_i32 s18, s1, s16 // (op 34650) 

s_ashr_i32 s1, s18, 31 // (op 34650) 
s_cmp_eq_i32 s9, 0 // (op 34650) 
s_mov_b32 s19, scc // (op 34650) 
s_cmp_eq_i32 s19, 0 // (op 34650) 
s_cselect_b32 s22, 0, -1 // (op 34650) 

s_and_b32 s19, s46, 31 // (op 34650) 
s_lshl_b32 s23, 1, s19 // (op 34650) 
s_add_i32 s24, s23, s22 // (op 34650) 


s_and_b32 s25, s1, s24 // (op 34650) 


s_add_i32 s1, s18, s25 // (op 34650) 


s_ashr_i32 s18, s1, s19 // (op 34650) 


s_lshl_b32 s1, s46, 24 // (op 34650) 
s_ashr_i32 s19, s1, 31 // (op 34650) 

s_xor_b32 s1, s18, s19 // (op 34650) 

s_sub_i32 s18, s1, s19 // (op 34650) 


s_add_u32 s50, s6, 256 // (op 34650) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34650) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34650) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34650) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34650) 

s_mul_i32 s1, s15, s24 // (op 34650) 

s_mul_i32 s19, s8, s1 // (op 34650) 

s_mul_i32 s1, s18, s19 // (op 34650) 

s_sub_u32 s19, s16, s1 // (op 34650) 

s_mov_b32 s1, s19 // (op 34650) 
s_mul_hi_i32 s22, s1, s14 // (op 34650) 
s_add_i32 s23, s22, s1 // (op 34650) 


s_ashr_i32 s1, s23, 31 // (op 34650) 
s_cmp_eq_i32 s14, 0 // (op 34650) 
s_mov_b32 s22, scc // (op 34650) 
s_cmp_eq_i32 s22, 0 // (op 34650) 
s_cselect_b32 s24, 0, -1 // (op 34650) 

s_and_b32 s22, s48, 31 // (op 34650) 
s_lshl_b32 s25, 1, s22 // (op 34650) 
s_add_i32 s49, s25, s24 // (op 34650) 


s_and_b32 s24, s1, s49 // (op 34650) 


s_add_i32 s1, s23, s24 // (op 34650) 


s_ashr_i32 s23, s1, s22 // (op 34650) 


s_lshl_b32 s1, s48, 24 // (op 34650) 
s_ashr_i32 s22, s1, 31 // (op 34650) 

s_xor_b32 s1, s23, s22 // (op 34650) 

s_sub_i32 s23, s1, s22 // (op 34650) 


s_mul_i32 s1, s47, s15 // (op 34650) 
s_add_u32 s24, s20, 256 // (op 34650) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 34650) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 34650) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 34650) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 34650) 

s_sub_i32 s22, s24, s1 // (op 34650) 

s_mul_i32 s24, s23, s22 // (op 34650) 


s_sub_u32 s25, s19, s24 // (op 34650) 


s_add_u32 s19, s25, s1 // (op 34650) 


s_cmp_eq_i32 s18, 0 // (op 34650) 
s_mov_b32 s1, scc // (op 34650) 

s_cmp_eq_i32 s1, 0 // (op 34650) 
s_cselect_b32 s18, s19, s0 // (op 34650) 



v_lshl_add_u32 v17, s18, 8, v15 // (op 34650) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 34650) 
v_mov_b32 v19, 0 // (op 34650) convert
v_mov_b32 v18, v17 // (op 34650) convert

s_lshr_b32 s18, s0, 1 // (op 34650) 

v_lshl_add_u32 v15, s18, 8, v16 // (op 34650) 


v_mul_lo_u32 v17, v15, s27 // (op 34650) most significant: low of low * high
v_mul_hi_u32 v20, v15, s26 // (op 34650) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 34650) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 34650) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v16 // (op 34650) least significant half
v_addc_co_u32 v21, s[0:1], v19, v17, s[0:1] // (op 34650) most significant half



s_mul_i32 s1, s26, s31 // (op 34652) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 34652) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 34652) least significant
s_mul_hi_u32 s19, s26, s30 // (op 34652) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34652) most significant: sum
s_add_u32 s1, s1, s19 // (op 34652) most significant: sum


s_mov_b32 s52, s28 // (op 34652) reg expression
s_mov_b32 s53, s29 // (op 34652) reg expression
s_mov_b32 s54, s0 // (op 34652) reg expression
s_mov_b32 s55, 131072 // (op 34652) call()

s_mul_hi_i32 s0, s16, s10 // (op 34780) 
v_lshrrev_b32 v15, 3, v1 // (op 34780) 
s_add_i32 s1, s0, s16 // (op 34780) 

v_and_b32 v16, -8, v1 // (op 34780) 
s_ashr_i32 s0, s1, 31 // (op 34780) 
v_sub_u32 v17, v1, v16 // (op 34780) 

s_cmp_eq_i32 s10, 0 // (op 34780) 
s_mov_b32 s18, scc // (op 34780) 
v_lshlrev_b32 v16, 4, v17 // (op 34780) 

s_cmp_eq_i32 s18, 0 // (op 34780) 
s_cselect_b32 s19, 0, -1 // (op 34780) 

s_and_b32 s18, s11, 31 // (op 34780) 
s_lshl_b32 s22, 1, s18 // (op 34780) 
s_add_i32 s23, s22, s19 // (op 34780) 


s_and_b32 s19, s0, s23 // (op 34780) 


s_add_i32 s0, s1, s19 // (op 34780) 


s_ashr_i32 s1, s0, s18 // (op 34780) 


s_lshl_b32 s0, s11, 24 // (op 34780) 
s_ashr_i32 s18, s0, 31 // (op 34780) 

s_xor_b32 s0, s1, s18 // (op 34780) 

s_sub_i32 s1, s0, s18 // (op 34780) 


s_mul_hi_i32 s0, s1, s12 // (op 34780) 
s_add_i32 s18, s0, s1 // (op 34780) 

s_ashr_i32 s0, s18, 31 // (op 34780) 
s_cmp_eq_i32 s12, 0 // (op 34780) 
s_mov_b32 s19, scc // (op 34780) 
s_cmp_eq_i32 s19, 0 // (op 34780) 
s_cselect_b32 s22, 0, -1 // (op 34780) 

s_and_b32 s19, s13, 31 // (op 34780) 
s_lshl_b32 s23, 1, s19 // (op 34780) 
s_add_i32 s24, s23, s22 // (op 34780) 


s_and_b32 s25, s0, s24 // (op 34780) 


s_add_i32 s0, s18, s25 // (op 34780) 


s_ashr_i32 s18, s0, s19 // (op 34780) 


s_lshl_b32 s0, s13, 24 // (op 34780) 
s_ashr_i32 s19, s0, 31 // (op 34780) 

s_xor_b32 s0, s18, s19 // (op 34780) 

s_sub_i32 s18, s0, s19 // (op 34780) 


s_add_u32 s50, s6, 256 // (op 34780) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34780) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34780) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34780) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34780) 

s_mul_i32 s0, s18, s24 // (op 34780) 

s_sub_i32 s18, s1, s0 // (op 34780) 


s_mul_hi_i32 s19, s16, s9 // (op 34780) 
s_add_i32 s0, s19, s16 // (op 34780) 

s_ashr_i32 s1, s0, 31 // (op 34780) 
s_cmp_eq_i32 s9, 0 // (op 34780) 
s_mov_b32 s19, scc // (op 34780) 
s_cmp_eq_i32 s19, 0 // (op 34780) 
s_cselect_b32 s22, 0, -1 // (op 34780) 

s_and_b32 s19, s46, 31 // (op 34780) 
s_lshl_b32 s23, 1, s19 // (op 34780) 
s_add_i32 s49, s23, s22 // (op 34780) 


s_and_b32 s22, s1, s49 // (op 34780) 


s_add_i32 s1, s0, s22 // (op 34780) 


s_ashr_i32 s0, s1, s19 // (op 34780) 


s_lshl_b32 s1, s46, 24 // (op 34780) 
s_ashr_i32 s19, s1, 31 // (op 34780) 

s_xor_b32 s1, s0, s19 // (op 34780) 

s_sub_i32 s0, s1, s19 // (op 34780) 


s_mul_i32 s1, s15, s24 // (op 34780) 

s_mul_i32 s19, s8, s1 // (op 34780) 

s_mul_i32 s1, s0, s19 // (op 34780) 

s_sub_u32 s19, s16, s1 // (op 34780) 

s_mul_hi_i32 s1, s19, s14 // (op 34780) 
s_add_i32 s22, s1, s19 // (op 34780) 


s_ashr_i32 s1, s22, 31 // (op 34780) 
s_cmp_eq_i32 s14, 0 // (op 34780) 
s_mov_b32 s19, scc // (op 34780) 
s_cmp_eq_i32 s19, 0 // (op 34780) 
s_cselect_b32 s23, 0, -1 // (op 34780) 

s_and_b32 s19, s48, 31 // (op 34780) 
s_lshl_b32 s24, 1, s19 // (op 34780) 
s_add_i32 s25, s24, s23 // (op 34780) 


s_and_b32 s23, s1, s25 // (op 34780) 


s_add_i32 s1, s22, s23 // (op 34780) 


s_ashr_i32 s22, s1, s19 // (op 34780) 


s_lshl_b32 s1, s48, 24 // (op 34780) 
s_ashr_i32 s19, s1, 31 // (op 34780) 

s_xor_b32 s1, s22, s19 // (op 34780) 

s_sub_i32 s22, s1, s19 // (op 34780) 


s_cmp_eq_i32 s0, 0 // (op 34780) 
s_mov_b32 s1, scc // (op 34780) 

s_cmp_eq_i32 s1, 0 // (op 34780) 
s_cselect_b32 s0, s22, s18 // (op 34780) 



v_lshl_add_u32 v17, s0, 8, v15 // (op 34780) 


v_mul_lo_u32 v19, v17, s33 // (op 34780) most significant: low of low * high
v_mul_hi_u32 v15, v17, s32 // (op 34780) most significant: high of low * low
v_mul_lo_u32 v18, v17, s32 // (op 34780) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 34780) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 34780) 
s_lshr_b32 s18, s0, 1 // (op 34780) 

v_lshl_add_u32 v15, s18, 8, v16 // (op 34780) 


v_mov_b32 v17, 0 // (op 34780) convert
v_mov_b32 v16, v15 // (op 34780) convert

v_add_co_u32 v22, s[0:1], v16, v18 // (op 34780) least significant half
v_addc_co_u32 v23, s[0:1], v17, v19, s[0:1] // (op 34780) most significant half



s_mul_i32 s1, s32, s7 // (op 34782) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 34782) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 34782) least significant
s_mul_hi_u32 s19, s32, s6 // (op 34782) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34782) most significant: sum
s_add_u32 s1, s1, s19 // (op 34782) most significant: sum


s_mov_b32 s64, s34 // (op 34782) reg expression
s_mov_b32 s65, s35 // (op 34782) reg expression
s_mov_b32 s66, s0 // (op 34782) reg expression
s_mov_b32 s67, 131072 // (op 34782) call()

s_mul_hi_i32 s0, s16, s10 // (op 34844) 
v_and_b32 v15, -64, v1 // (op 34844) 
s_add_i32 s1, s0, s16 // (op 34844) 

v_sub_u32 v16, v1, v15 // (op 34844) 

s_ashr_i32 s0, s1, 31 // (op 34844) 
v_lshlrev_b32 v15, 2, v16 // (op 34844) 

s_cmp_eq_i32 s10, 0 // (op 34844) 
s_mov_b32 s18, scc // (op 34844) 
v_lshrrev_b32 v16, 6, v1 // (op 34844) 
s_cmp_eq_i32 s18, 0 // (op 34844) 
s_cselect_b32 s19, 0, -1 // (op 34844) 

s_and_b32 s18, s11, 31 // (op 34844) 
s_lshl_b32 s22, 1, s18 // (op 34844) 
s_add_i32 s23, s22, s19 // (op 34844) 


s_and_b32 s19, s0, s23 // (op 34844) 


s_add_i32 s0, s1, s19 // (op 34844) 


s_ashr_i32 s1, s0, s18 // (op 34844) 


s_lshl_b32 s0, s11, 24 // (op 34844) 
s_ashr_i32 s18, s0, 31 // (op 34844) 

s_xor_b32 s0, s1, s18 // (op 34844) 

s_sub_i32 s1, s0, s18 // (op 34844) 


s_mul_hi_i32 s0, s1, s12 // (op 34844) 
s_add_i32 s18, s0, s1 // (op 34844) 

s_ashr_i32 s0, s18, 31 // (op 34844) 
s_cmp_eq_i32 s12, 0 // (op 34844) 
s_mov_b32 s19, scc // (op 34844) 
s_cmp_eq_i32 s19, 0 // (op 34844) 
s_cselect_b32 s22, 0, -1 // (op 34844) 

s_and_b32 s19, s13, 31 // (op 34844) 
s_lshl_b32 s23, 1, s19 // (op 34844) 
s_add_i32 s24, s23, s22 // (op 34844) 


s_and_b32 s25, s0, s24 // (op 34844) 


s_add_i32 s0, s18, s25 // (op 34844) 


s_ashr_i32 s18, s0, s19 // (op 34844) 


s_lshl_b32 s0, s13, 24 // (op 34844) 
s_ashr_i32 s19, s0, 31 // (op 34844) 

s_xor_b32 s0, s18, s19 // (op 34844) 

s_sub_i32 s18, s0, s19 // (op 34844) 


s_mul_i32 s0, s1, s15 // (op 34844) 

s_sub_u32 s1, s16, s0 // (op 34844) 

s_mul_i32 s0, s18, s15 // (op 34844) 
s_add_u32 s0, s0, s1 // (op 34844) 


s_mul_hi_i32 s1, s16, s9 // (op 34844) 
s_add_i32 s18, s1, s16 // (op 34844) 

s_ashr_i32 s1, s18, 31 // (op 34844) 
s_cmp_eq_i32 s9, 0 // (op 34844) 
s_mov_b32 s19, scc // (op 34844) 
s_cmp_eq_i32 s19, 0 // (op 34844) 
s_cselect_b32 s22, 0, -1 // (op 34844) 

s_and_b32 s19, s46, 31 // (op 34844) 
s_lshl_b32 s23, 1, s19 // (op 34844) 
s_add_i32 s24, s23, s22 // (op 34844) 


s_and_b32 s25, s1, s24 // (op 34844) 


s_add_i32 s1, s18, s25 // (op 34844) 


s_ashr_i32 s18, s1, s19 // (op 34844) 


s_lshl_b32 s1, s46, 24 // (op 34844) 
s_ashr_i32 s19, s1, 31 // (op 34844) 

s_xor_b32 s1, s18, s19 // (op 34844) 

s_sub_i32 s18, s1, s19 // (op 34844) 


s_add_u32 s50, s6, 256 // (op 34844) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34844) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34844) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34844) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34844) 

s_mul_i32 s1, s15, s24 // (op 34844) 

s_mul_i32 s19, s8, s1 // (op 34844) 

s_mul_i32 s1, s18, s19 // (op 34844) 

s_sub_u32 s19, s16, s1 // (op 34844) 

s_mov_b32 s1, s19 // (op 34844) 
s_mul_hi_i32 s22, s1, s14 // (op 34844) 
s_add_i32 s23, s22, s1 // (op 34844) 


s_ashr_i32 s1, s23, 31 // (op 34844) 
s_cmp_eq_i32 s14, 0 // (op 34844) 
s_mov_b32 s22, scc // (op 34844) 
s_cmp_eq_i32 s22, 0 // (op 34844) 
s_cselect_b32 s24, 0, -1 // (op 34844) 

s_and_b32 s22, s48, 31 // (op 34844) 
s_lshl_b32 s25, 1, s22 // (op 34844) 
s_add_i32 s49, s25, s24 // (op 34844) 


s_and_b32 s24, s1, s49 // (op 34844) 


s_add_i32 s1, s23, s24 // (op 34844) 


s_ashr_i32 s23, s1, s22 // (op 34844) 


s_lshl_b32 s1, s48, 24 // (op 34844) 
s_ashr_i32 s22, s1, 31 // (op 34844) 

s_xor_b32 s1, s23, s22 // (op 34844) 

s_sub_i32 s23, s1, s22 // (op 34844) 


s_mul_i32 s1, s47, s15 // (op 34844) 
s_add_u32 s24, s20, 256 // (op 34844) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 34844) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 34844) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 34844) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 34844) 

s_sub_i32 s22, s24, s1 // (op 34844) 

s_mul_i32 s24, s23, s22 // (op 34844) 


s_sub_u32 s25, s19, s24 // (op 34844) 


s_add_u32 s19, s25, s1 // (op 34844) 


s_cmp_eq_i32 s18, 0 // (op 34844) 
s_mov_b32 s1, scc // (op 34844) 

s_cmp_eq_i32 s1, 0 // (op 34844) 
s_cselect_b32 s18, s19, s0 // (op 34844) 



v_lshl_add_u32 v17, s18, 8, v15 // (op 34844) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 34844) 
s_lshr_b32 s18, s0, 1 // (op 34844) 

v_lshl_add_u32 v15, s18, 3, v16 // (op 34844) 


v_mul_lo_u32 v16, v15, s36 // (op 34844) 

v_add_u32 v15, v17, v16 // (op 34844) 


s_mul_i32 s1, s36, s41 // (op 34846) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 34846) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 34846) least significant
s_mul_hi_u32 s19, s36, s40 // (op 34846) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34846) most significant: sum
s_add_u32 s1, s1, s19 // (op 34846) most significant: sum


s_mov_b32 s68, s38 // (op 34846) reg expression
s_mov_b32 s69, s39 // (op 34846) reg expression
s_mov_b32 s70, s0 // (op 34846) reg expression
s_mov_b32 s71, 131072 // (op 34846) call()

s_mul_hi_i32 s0, s16, s10 // (op 34908) 
s_add_i32 s1, s0, s16 // (op 34908) 

s_ashr_i32 s0, s1, 31 // (op 34908) 
s_cmp_eq_i32 s10, 0 // (op 34908) 
s_mov_b32 s18, scc // (op 34908) 
s_cmp_eq_i32 s18, 0 // (op 34908) 
s_cselect_b32 s19, 0, -1 // (op 34908) 

s_and_b32 s18, s11, 31 // (op 34908) 
s_lshl_b32 s22, 1, s18 // (op 34908) 
s_add_i32 s23, s22, s19 // (op 34908) 


s_and_b32 s19, s0, s23 // (op 34908) 


s_add_i32 s0, s1, s19 // (op 34908) 


s_ashr_i32 s1, s0, s18 // (op 34908) 


s_lshl_b32 s0, s11, 24 // (op 34908) 
s_ashr_i32 s18, s0, 31 // (op 34908) 

s_xor_b32 s0, s1, s18 // (op 34908) 

s_sub_i32 s1, s0, s18 // (op 34908) 


s_mul_hi_i32 s0, s1, s12 // (op 34908) 
s_add_i32 s18, s0, s1 // (op 34908) 

s_ashr_i32 s0, s18, 31 // (op 34908) 
s_cmp_eq_i32 s12, 0 // (op 34908) 
s_mov_b32 s19, scc // (op 34908) 
s_cmp_eq_i32 s19, 0 // (op 34908) 
s_cselect_b32 s22, 0, -1 // (op 34908) 

s_and_b32 s19, s13, 31 // (op 34908) 
s_lshl_b32 s23, 1, s19 // (op 34908) 
s_add_i32 s24, s23, s22 // (op 34908) 


s_and_b32 s25, s0, s24 // (op 34908) 


s_add_i32 s0, s18, s25 // (op 34908) 


s_ashr_i32 s18, s0, s19 // (op 34908) 


s_lshl_b32 s0, s13, 24 // (op 34908) 
s_ashr_i32 s19, s0, 31 // (op 34908) 

s_xor_b32 s0, s18, s19 // (op 34908) 

s_sub_i32 s18, s0, s19 // (op 34908) 


s_add_u32 s50, s6, 256 // (op 34908) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 34908) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 34908) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 34908) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 34908) 

s_mul_i32 s0, s18, s24 // (op 34908) 

s_sub_i32 s18, s1, s0 // (op 34908) 


s_mul_hi_i32 s19, s16, s9 // (op 34908) 
s_add_i32 s0, s19, s16 // (op 34908) 

s_ashr_i32 s1, s0, 31 // (op 34908) 
s_cmp_eq_i32 s9, 0 // (op 34908) 
s_mov_b32 s19, scc // (op 34908) 
s_cmp_eq_i32 s19, 0 // (op 34908) 
s_cselect_b32 s22, 0, -1 // (op 34908) 

s_and_b32 s19, s46, 31 // (op 34908) 
s_lshl_b32 s23, 1, s19 // (op 34908) 
s_add_i32 s49, s23, s22 // (op 34908) 


s_and_b32 s22, s1, s49 // (op 34908) 


s_add_i32 s1, s0, s22 // (op 34908) 


s_ashr_i32 s0, s1, s19 // (op 34908) 


s_lshl_b32 s1, s46, 24 // (op 34908) 
s_ashr_i32 s19, s1, 31 // (op 34908) 

s_xor_b32 s1, s0, s19 // (op 34908) 

s_sub_i32 s0, s1, s19 // (op 34908) 


s_mul_i32 s1, s15, s24 // (op 34908) 

s_mul_i32 s19, s8, s1 // (op 34908) 

s_mul_i32 s1, s0, s19 // (op 34908) 

s_sub_u32 s19, s16, s1 // (op 34908) 

s_mul_hi_i32 s1, s19, s14 // (op 34908) 
s_add_i32 s22, s1, s19 // (op 34908) 


s_ashr_i32 s1, s22, 31 // (op 34908) 
s_cmp_eq_i32 s14, 0 // (op 34908) 
s_mov_b32 s19, scc // (op 34908) 
s_cmp_eq_i32 s19, 0 // (op 34908) 
s_cselect_b32 s23, 0, -1 // (op 34908) 

s_and_b32 s19, s48, 31 // (op 34908) 
s_lshl_b32 s24, 1, s19 // (op 34908) 
s_add_i32 s25, s24, s23 // (op 34908) 


s_and_b32 s23, s1, s25 // (op 34908) 


s_add_i32 s1, s22, s23 // (op 34908) 


s_ashr_i32 s22, s1, s19 // (op 34908) 


s_lshl_b32 s1, s48, 24 // (op 34908) 
s_ashr_i32 s19, s1, 31 // (op 34908) 

s_xor_b32 s1, s22, s19 // (op 34908) 

s_sub_i32 s22, s1, s19 // (op 34908) 


s_cmp_eq_i32 s0, 0 // (op 34908) 
s_mov_b32 s1, scc // (op 34908) 

s_cmp_eq_i32 s1, 0 // (op 34908) 
s_cselect_b32 s0, s22, s18 // (op 34908) 



v_lshl_add_u32 v16, s0, 8, v1 // (op 34908) 

v_mul_lo_u32 v17, v16, s42 // (op 34908) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 34908) 
s_lshr_b32 s18, s0, 1 // (op 34908) 

v_lshl_add_u32 v16, s18, 3, v17 // (op 34908) 


s_mul_i32 s1, s42, s7 // (op 34910) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 34910) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 34910) least significant
s_mul_hi_u32 s19, s42, s6 // (op 34910) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 34910) most significant: sum
s_add_u32 s1, s1, s19 // (op 34910) most significant: sum


s_mov_b32 s72, s44 // (op 34910) reg expression
s_mov_b32 s73, s45 // (op 34910) reg expression
s_mov_b32 s74, s0 // (op 34910) reg expression
s_mov_b32 s75, 131072 // (op 34910) call()

v_and_b32 v17, -16, v1 // (op 35006) 
v_sub_u32 v18, v1, v17 // (op 35006) 

v_lshlrev_b32 v17, 4, v18 // (op 35006) 

v_lshrrev_b32 v18, 4, v1 // (op 35006) 
v_lshl_add_u32 v19, v18, 8, v17 // (op 35006) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 35006) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 35063) 
v_sub_u32 v18, v1, v17 // (op 35063) 

v_lshlrev_b32 v17, 4, v18 // (op 35063) 

v_lshrrev_b32 v18, 3, v1 // (op 35063) 
v_lshl_add_u32 v19, v18, 7, v17 // (op 35063) 


s_nop 0
v_readfirstlane_b32 s1, v19 // (op 35063) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -64, v1 // (op 35120) 
v_sub_u32 v18, v1, v17 // (op 35120) 

v_lshlrev_b32 v17, 2, v18 // (op 35120) 

v_lshrrev_b32 v18, 6, v1 // (op 35120) 
v_lshl_add_u32 v19, v18, 8, v17 // (op 35120) 


v_lshlrev_b32 v17, 2, v1 // (op 35178) 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 11196) 
s_and_b32 s22, s18, 4294967294 // (op 11196) 

s_ashr_i64 s[18:19], s[30:31], 7 // (op 11195) 
s_mov_b32 s23, s18 // (op 11195) convert

s_cmp_lt_u32 s22, s23 // (op 11195) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_11195 // (op 11195) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_11195 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_11195:
s_barrier  // (op 35679) 
s_add_u32 s22, s22, 1 // (op 11197) 

s_lshl_b64 s[18:19], s[26:27], 4 // (op 34602) 
v_mov_b64 v[24:25], s[18:19] // (op 34602) convert


v_mov_b64 v[26:27], v[20:21] // (op 34602) 
v_mov_b64 v[28:29], v[26:27] // (op 34602) reg expression
s_add_u32 s18, 32768, s0 // (op 34602) 
s_mov_b32 m0, s18 // (op 34602) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34602) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v24 // (op 34602) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 34602) most significant half

s_add_u32 m0, m0, 4096 // (op 34602) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34602) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v24 // (op 34602) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 34602) most significant half

s_add_u32 m0, m0, 4096 // (op 34602) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34602) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v24 // (op 34602) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 34602) most significant half

s_add_u32 m0, m0, 4096 // (op 34602) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34602) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v24 // (op 34602) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 34602) most significant half

s_add_u32 m0, m0, 4096 // (op 34602) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34602) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v24 // (op 34602) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 34602) most significant half

s_add_u32 m0, m0, 4096 // (op 34602) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34602) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v24 // (op 34602) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 34602) most significant half

s_add_u32 m0, m0, 4096 // (op 34602) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34602) Load value direct to lds
v_add_co_u32 v28, s[24:25], v28, v24 // (op 34602) least significant half
v_addc_co_u32 v29, s[24:25], 0, v29, s[24:25] // (op 34602) most significant half

s_add_u32 m0, m0, 4096 // (op 34602) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 34602) Load value direct to lds




s_lshl_b64 s[18:19], s[26:27], 7 // (op 34640) 
v_mov_b32 v14, s18 // (op 34640) 
v_mov_b32 v18, s19 // (op 34640) 
v_add_co_u32 v20, s[24:25], v20, v14 // (op 34640) least significant half
v_addc_co_u32 v21, s[24:25], v21, v18, s[24:25] // (op 34640) most significant half




s_barrier  // (op 35695) 

s_lshl_b64 s[18:19], s[32:33], 5 // (op 34608) 
v_mov_b64 v[24:25], s[18:19] // (op 34608) convert


v_mov_b64 v[26:27], v[22:23] // (op 34608) 
v_mov_b64 v[28:29], v[26:27] // (op 34608) 
s_mov_b32 m0, s1 // (op 34608) reg expression
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34608) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34608) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34608) most significant half

s_add_u32 m0, m0, 4096 // (op 34608) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34608) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34608) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34608) most significant half

s_add_u32 m0, m0, 4096 // (op 34608) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34608) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34608) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34608) most significant half

s_add_u32 m0, m0, 4096 // (op 34608) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34608) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34608) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34608) most significant half

s_add_u32 m0, m0, 4096 // (op 34608) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34608) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34608) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34608) most significant half

s_add_u32 m0, m0, 4096 // (op 34608) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34608) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34608) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34608) most significant half

s_add_u32 m0, m0, 4096 // (op 34608) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34608) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v24 // (op 34608) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 34608) most significant half

s_add_u32 m0, m0, 4096 // (op 34608) 
buffer_load_dwordx4 v28, s[64:67], 0 offen offset: 0 lds // (op 34608) Load value direct to lds



v_add_co_u32 v22, vcc, 128, v22 // (op 34772) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 34772) most significant half


v_mov_b32 v14, v15 // (op 191) 
v_mov_b32 v18, v14 // (op 191) reg expression

buffer_load_dword v24, v18, s[68:71], 0 offen // (op 191) Load value


s_lshl_b32 s18, s36, 2 // (op 34836) 
v_add_u32 v15, v15, s18 // (op 34836) 



s_barrier  // (op 35712) 
v_mov_b32 v14, v16 // (op 192) 




v_add_u32 v18, 132096, v19 // (op 188) 
buffer_load_dword v25, v14, s[72:75], 0 offen // (op 192) Load value

v_add_u32 v16, 4, v16 // (op 34900) 
s_waitcnt vmcnt(1)
ds_write_b32 v18, v24 // (op 188) Store local data 


s_barrier  // (op 35685) 



s_barrier  // (op 35729) 
ds_read_b64_tr_b8 v[26:27], v0 offset:32768 // (op 179) Transpose load local data 



ds_read_b64_tr_b8 v[28:29], v0 offset:34816 // (op 179) Transpose load local data 
v_add_u32 v14, 131072, v17 // (op 189) 
ds_read_b64_tr_b8 v[30:31], v0 offset:49152 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:51200 // (op 179) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v14, v25 // (op 189) Store local data 


s_barrier  // (op 35702) 



ds_read_b128 v[34:37], v4 // (op 181) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 181) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 35719) 


v_add_u32 v14, 132096, v3 // (op 183) 
ds_read_u8 v18, v14 // (op 183) Load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 35736) 



v_add_u32 v14, 131072, v5 // (op 185) 
ds_read_u8 v24, v14 // (op 185) Load local data 




ds_read_b128 v[42:45], v4 offset:4096 // (op 350) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v18, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 186) 
ds_read_b128 v[46:49], v4 offset:4160 // (op 350) Load local data 



v_add_u32 v14, 131200, v5 // (op 354) 
ds_read_u8 v25, v14 // (op 354) Load local data 




ds_read_b128 v[50:53], v4 offset:8192 // (op 526) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[42:49], a[4:7], v18, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 355) 
ds_read_b128 v[54:57], v4 offset:8256 // (op 526) Load local data 



v_add_u32 v14, 131328, v5 // (op 530) 
ds_read_u8 v58, v14 // (op 530) Load local data 




ds_read_b128 v[60:63], v4 offset:12288 // (op 702) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:33], v[50:57], a[8:11], v18, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 531) 
ds_read_b128 v[64:67], v4 offset:12352 // (op 702) Load local data 



v_add_u32 v14, 131456, v5 // (op 706) 
ds_read_u8 v59, v14 // (op 706) Load local data 




ds_read_b128 v[68:71], v4 offset:16384 // (op 878) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:33], v[60:67], a[12:15], v18, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 707) 
ds_read_b128 v[72:75], v4 offset:16448 // (op 878) Load local data 



v_add_u32 v14, 131584, v5 // (op 882) 
ds_read_u8 v76, v14 // (op 882) Load local data 




ds_read_b128 v[78:81], v4 offset:20480 // (op 1054) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:33], v[68:75], a[16:19], v18, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 883) 
ds_read_b128 v[82:85], v4 offset:20544 // (op 1054) Load local data 



v_add_u32 v14, 131712, v5 // (op 1058) 
ds_read_u8 v77, v14 // (op 1058) Load local data 




ds_read_b128 v[86:89], v4 offset:24576 // (op 1230) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:33], v[78:85], a[20:23], v18, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1059) 
ds_read_b128 v[90:93], v4 offset:24640 // (op 1230) Load local data 



v_add_u32 v14, 131840, v5 // (op 1234) 
ds_read_u8 v94, v14 // (op 1234) Load local data 




ds_read_b128 v[96:99], v4 offset:28672 // (op 1406) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:33], v[86:93], a[24:27], v18, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1235) 
ds_read_b128 v[100:103], v4 offset:28736 // (op 1406) Load local data 



v_add_u32 v14, 131968, v5 // (op 1410) 
ds_read_u8 v95, v14 // (op 1410) Load local data 




ds_read_b64_tr_b8 v[104:105], v0 offset:32800 // (op 1756) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:33], v[96:103], a[28:31], v18, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1411) 
ds_read_b64_tr_b8 v[106:107], v0 offset:34848 // (op 1756) Transpose load local data 


ds_read_b64_tr_b8 v[108:109], v0 offset:49184 // (op 1756) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v0 offset:51232 // (op 1756) Transpose load local data 


v_add_u32 v14, 132128, v3 // (op 1759) 
ds_read_u8 v18, v14 // (op 1759) Load local data 




ds_read_b64_tr_b8 v[26:27], v0 offset:32832 // (op 3294) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:34880 // (op 3294) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:111], v[34:41], a[32:35], v18, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1761) 
ds_read_b64_tr_b8 v[30:31], v0 offset:49216 // (op 3294) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:111], v[42:49], a[36:39], v18, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1792) 
ds_read_b64_tr_b8 v[32:33], v0 offset:51264 // (op 3294) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:111], v[50:57], a[40:43], v18, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1823) 
v_add_u32 v14, 132160, v3 // (op 3297) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:111], v[60:67], a[44:47], v18, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1854) 
ds_read_u8 v112, v14 // (op 3297) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:111], v[68:75], a[48:51], v18, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1885) 
ds_read_b64_tr_b8 v[114:115], v0 offset:32864 // (op 4832) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:111], v[78:85], a[52:55], v18, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1916) 
ds_read_b64_tr_b8 v[116:117], v0 offset:34912 // (op 4832) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:111], v[86:93], a[56:59], v18, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1947) 
ds_read_b64_tr_b8 v[118:119], v0 offset:49248 // (op 4832) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:111], v[96:103], a[60:63], v18, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1978) 
ds_read_b64_tr_b8 v[120:121], v0 offset:51296 // (op 4832) Transpose load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[26:33], v[34:41], a[64:67], v112, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3299) 
v_add_u32 v14, 132192, v3 // (op 4835) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:33], v[42:49], a[68:71], v112, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3330) 
ds_read_u8 v18, v14 // (op 4835) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:33], v[50:57], a[72:75], v112, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3361) 
ds_read_b64_tr_b8 v[104:105], v0 offset:32896 // (op 6370) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[26:33], v[60:67], a[76:79], v112, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3392) 
ds_read_b64_tr_b8 v[106:107], v0 offset:34944 // (op 6370) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[26:33], v[68:75], a[80:83], v112, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3423) 
ds_read_b64_tr_b8 v[108:109], v0 offset:49280 // (op 6370) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[26:33], v[78:85], a[84:87], v112, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3454) 
ds_read_b64_tr_b8 v[110:111], v0 offset:51328 // (op 6370) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[26:33], v[86:93], a[88:91], v112, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3485) 
v_add_u32 v14, 132224, v3 // (op 6373) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[26:33], v[96:103], a[92:95], v112, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 3516) 
ds_read_u8 v113, v14 // (op 6373) Load local data 






s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[114:121], v[34:41], a[96:99], v18, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 4837) 
ds_read_b64_tr_b8 v[26:27], v0 offset:32928 // (op 7908) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[114:121], v[42:49], a[100:103], v18, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 4868) 
ds_read_b64_tr_b8 v[28:29], v0 offset:34976 // (op 7908) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[114:121], v[50:57], a[104:107], v18, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 4899) 
ds_read_b64_tr_b8 v[30:31], v0 offset:49312 // (op 7908) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[114:121], v[60:67], a[108:111], v18, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 4930) 
ds_read_b64_tr_b8 v[32:33], v0 offset:51360 // (op 7908) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[114:121], v[68:75], a[112:115], v18, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 4961) 
v_add_u32 v14, 132256, v3 // (op 7911) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[114:121], v[78:85], a[116:119], v18, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 4992) 
ds_read_u8 v112, v14 // (op 7911) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[114:121], v[86:93], a[120:123], v18, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5023) 
ds_read_b64_tr_b8 v[122:123], v0 offset:32960 // (op 9446) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[114:121], v[96:103], a[124:127], v18, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 5054) 
ds_read_b64_tr_b8 v[124:125], v0 offset:35008 // (op 9446) Transpose load local data 


ds_read_b64_tr_b8 v[126:127], v0 offset:49344 // (op 9446) Transpose load local data 
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[104:111], v[34:41], a[128:131], v113, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6375) 
ds_read_b64_tr_b8 v[128:129], v0 offset:51392 // (op 9446) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[104:111], v[42:49], a[132:135], v113, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6406) 
v_add_u32 v14, 132288, v3 // (op 9449) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[104:111], v[50:57], a[136:139], v113, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6437) 
ds_read_u8 v18, v14 // (op 9449) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[104:111], v[60:67], a[140:143], v113, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6468) 
ds_read_b64_tr_b8 v[114:115], v0 offset:32992 // (op 10984) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[104:111], v[68:75], a[144:147], v113, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6499) 
ds_read_b64_tr_b8 v[116:117], v0 offset:35040 // (op 10984) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[104:111], v[78:85], a[148:151], v113, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6530) 
ds_read_b64_tr_b8 v[118:119], v0 offset:49376 // (op 10984) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[104:111], v[86:93], a[152:155], v113, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6561) 
ds_read_b64_tr_b8 v[120:121], v0 offset:51424 // (op 10984) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[104:111], v[96:103], a[156:159], v113, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 6592) 
v_add_u32 v14, 132320, v3 // (op 10987) 


ds_read_u8 v113, v14 // (op 10987) Load local data 

s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[26:33], v[34:41], a[160:163], v112, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 7913) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:33], v[42:49], a[164:167], v112, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 7944) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:33], v[50:57], a[168:171], v112, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 7975) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:33], v[60:67], a[172:175], v112, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8006) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[26:33], v[68:75], a[176:179], v112, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8037) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[26:33], v[78:85], a[180:183], v112, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8068) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[26:33], v[86:93], a[184:187], v112, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8099) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[26:33], v[96:103], a[188:191], v112, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 8130) 


s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[122:129], v[34:41], a[192:195], v18, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9451) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[122:129], v[42:49], a[196:199], v18, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9482) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[122:129], v[50:57], a[200:203], v18, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9513) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[122:129], v[60:67], a[204:207], v18, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9544) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[122:129], v[68:75], a[208:211], v18, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9575) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[122:129], v[78:85], a[212:215], v18, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9606) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[122:129], v[86:93], a[216:219], v18, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9637) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[122:129], v[96:103], a[220:223], v18, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 9668) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[114:121], v[34:41], a[224:227], v113, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 10989) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[114:121], v[42:49], a[228:231], v113, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11020) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:121], v[50:57], a[232:235], v113, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11051) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:121], v[60:67], a[236:239], v113, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11082) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[114:121], v[68:75], a[240:243], v113, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11113) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[114:121], v[78:85], a[244:247], v113, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11144) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[114:121], v[86:93], a[248:251], v113, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11175) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[114:121], v[96:103], a[252:255], v113, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 11206) 




s_ashr_i64 s[18:19], s[30:31], 7 // (op 11195) 
s_mov_b32 s23, s18 // (op 11195) convert

s_cmp_lt_u32 s22, s23 // (op 11195) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_11195 // (op 11195) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_11195 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_11195:























s_barrier  // (op 35849) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_133
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
s_branch RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_133
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_133:


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
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_133:








































































s_endpgm  // End of RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.LRR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__end:
.size RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_, .LRR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__end-RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 508
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 135168 // lds bytes
  .amdhsa_accum_offset 252
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
    DataTypeB: B8
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
  - .name: RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 135168
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 250
    .agpr_count: 256
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

