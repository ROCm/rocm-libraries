// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_,@function
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 2421) 
s_mul_hi_i32 s1, s16, s10 // (op 2676) 
v_and_b32 v0, 48, v1 // (op 2482) 
v_and_b32 v3, 63, v1 // (op 2482) 
v_sub_u32 v4, v3, v0 // (op 2482) 

v_lshrrev_b32 v3, 1, v4 // (op 2482) 
v_add_lshl_u32 v5, v0, v3, 6 // (op 2482) 


v_and_b32 v0, 1, v4 // (op 2482) 

v_lshlrev_b32 v3, 3, v0 // (op 2482) 

v_lshrrev_b32 v0, 7, v1 // (op 2482) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2482) 


v_add_u32 v0, v5, v4 // (op 2482) 


v_lshrrev_b32 v3, 6, v1 // (op 2539) 
v_and_b32 v4, 1, v3 // (op 2539) 

v_and_b32 v3, 15, v1 // (op 2539) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 2539) 


v_and_b32 v3, 48, v1 // (op 2539) 
v_lshl_add_u32 v4, v5, 7, v3 // (op 2539) 


v_lshrrev_b32 v3, 7, v1 // (op 2597) 
v_and_b32 v5, 15, v1 // (op 2597) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 2597) 


v_and_b32 v3, 63, v1 // (op 2597) 
v_lshrrev_b32 v5, 4, v3 // (op 2597) 

v_lshl_add_u32 v3, v5, 6, v6 // (op 2597) 


v_lshrrev_b32 v5, 6, v1 // (op 2655) 
v_and_b32 v6, 1, v5 // (op 2655) 

v_and_b32 v5, 15, v1 // (op 2655) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2655) 


v_and_b32 v5, 63, v1 // (op 2655) 
v_lshrrev_b32 v6, 4, v5 // (op 2655) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 2655) 


v_and_b32 v6, 48, v1 // (op 2722) 
v_and_b32 v7, 63, v1 // (op 2722) 
v_sub_u32 v8, v7, v6 // (op 2722) 

v_lshrrev_b32 v7, 1, v8 // (op 2722) 
v_add_lshl_u32 v9, v6, v7, 6 // (op 2722) 


v_and_b32 v6, 1, v8 // (op 2722) 

v_lshlrev_b32 v7, 3, v6 // (op 2722) 

v_lshrrev_b32 v6, 7, v1 // (op 2722) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 2722) 


v_add_u32 v6, v9, v8 // (op 2722) 


v_lshrrev_b32 v7, 6, v1 // (op 2766) 
v_and_b32 v8, 1, v7 // (op 2766) 

v_and_b32 v7, 15, v1 // (op 2766) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 2766) 


v_and_b32 v7, 48, v1 // (op 2766) 
v_lshl_add_u32 v8, v9, 7, v7 // (op 2766) 


v_lshrrev_b32 v7, 7, v1 // (op 2795) 
v_and_b32 v9, 15, v1 // (op 2795) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 2795) 


v_and_b32 v7, 63, v1 // (op 2795) 
v_lshrrev_b32 v9, 4, v7 // (op 2795) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 2795) 


v_lshrrev_b32 v9, 6, v1 // (op 2824) 
v_and_b32 v10, 1, v9 // (op 2824) 

v_and_b32 v9, 15, v1 // (op 2824) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 2824) 


v_and_b32 v9, 63, v1 // (op 2824) 
v_lshrrev_b32 v10, 4, v9 // (op 2824) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 2824) 


v_lshrrev_b32 v10, 7, v1 // (op 2421) 
s_add_i32 s18, s0, s16 // (op 2421) 

v_and_b32 v11, 63, v1 // (op 2421) 
s_ashr_i32 s0, s18, 31 // (op 2421) 
v_lshrrev_b32 v12, 4, v11 // (op 2421) 
s_cmp_eq_i32 s10, 0 // (op 2421) 
s_mov_b32 s19, scc // (op 2421) 
v_lshlrev_b32 v13, 2, v12 // (op 2421) 

s_cmp_eq_i32 s19, 0 // (op 2421) 
s_cselect_b32 s51, 0, -1 // (op 2421) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 2421) 


s_and_b32 s19, s11, 31 // (op 2421) 
v_lshrrev_b32 v10, 2, v11 // (op 2421) 

v_and_b32 v11, 3, v10 // (op 2421) 

v_lshrrev_b32 v10, 7, v1 // (op 2676) 
v_and_b32 v13, 63, v1 // (op 2676) 
s_lshl_b32 s52, 1, s19 // (op 2421) 
v_and_b32 v14, 3, v1 // (op 2421) 
s_add_i32 s53, s52, s51 // (op 2421) 


s_and_b32 s51, s0, s53 // (op 2421) 


s_add_i32 s0, s1, s16 // (op 2676) 

s_ashr_i32 s1, s0, 31 // (op 2676) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 2421) 


s_add_i32 s52, s18, s51 // (op 2421) 


v_lshrrev_b32 v11, 6, v1 // (op 2421) 
s_ashr_i32 s18, s52, s19 // (op 2421) 


v_and_b32 v14, 1, v11 // (op 2421) 

s_lshl_b32 s19, s11, 24 // (op 2421) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 2421) 


s_ashr_i32 s51, s19, 31 // (op 2421) 

s_xor_b32 s19, s18, s51 // (op 2421) 

s_sub_i32 s18, s19, s51 // (op 2421) 


s_mul_hi_i32 s19, s18, s12 // (op 2421) 
s_add_i32 s51, s19, s18 // (op 2421) 

s_ashr_i32 s19, s51, 31 // (op 2421) 
s_cmp_eq_i32 s12, 0 // (op 2421) 
s_mov_b32 s52, scc // (op 2421) 
s_cmp_eq_i32 s52, 0 // (op 2421) 
s_cselect_b32 s53, 0, -1 // (op 2421) 

s_and_b32 s52, s13, 31 // (op 2421) 
v_lshrrev_b32 v14, 4, v13 // (op 2676) 
s_cmp_eq_i32 s10, 0 // (op 2676) 
s_mov_b32 s54, scc // (op 2676) 
v_lshlrev_b32 v15, 2, v14 // (op 2676) 

s_cmp_eq_i32 s54, 0 // (op 2676) 
s_cselect_b32 s55, 0, -1 // (op 2676) 

s_lshl_b32 s54, 1, s52 // (op 2421) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 2676) 


v_lshrrev_b32 v10, 2, v13 // (op 2676) 

s_add_i32 s56, s54, s53 // (op 2421) 


s_and_b32 s53, s19, s56 // (op 2421) 


s_add_i32 s19, s51, s53 // (op 2421) 


s_ashr_i32 s51, s19, s52 // (op 2421) 


s_lshl_b32 s19, s13, 24 // (op 2421) 
s_ashr_i32 s52, s19, 31 // (op 2421) 

s_xor_b32 s19, s51, s52 // (op 2421) 

s_sub_i32 s51, s19, s52 // (op 2421) 


s_mul_i32 s19, s18, s15 // (op 2421) 
s_sub_u32 s52, s16, s19 // (op 2421) 

s_mul_i32 s19, s51, s15 // (op 2421) 
s_add_u32 s19, s19, s52 // (op 2421) 

s_mul_hi_i32 s52, s16, s9 // (op 2421) 
s_add_i32 s53, s52, s16 // (op 2421) 

s_ashr_i32 s52, s53, 31 // (op 2421) 
s_cmp_eq_i32 s9, 0 // (op 2421) 
s_mov_b32 s54, scc // (op 2421) 
s_cmp_eq_i32 s54, 0 // (op 2421) 
s_cselect_b32 s56, 0, -1 // (op 2421) 

s_and_b32 s54, s11, 31 // (op 2676) 
v_and_b32 v13, 3, v10 // (op 2676) 

s_lshl_b32 s57, 1, s54 // (op 2676) 
v_and_b32 v10, 3, v1 // (op 2676) 
s_add_i32 s58, s57, s55 // (op 2676) 


s_and_b32 s55, s1, s58 // (op 2676) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 2676) 


s_add_i32 s1, s0, s55 // (op 2676) 


v_lshrrev_b32 v10, 6, v1 // (op 2676) 
s_ashr_i32 s0, s1, s54 // (op 2676) 


v_and_b32 v13, 1, v10 // (op 2676) 

s_lshl_b32 s1, s11, 24 // (op 2676) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 2676) 


s_ashr_i32 s54, s1, 31 // (op 2676) 

s_xor_b32 s1, s0, s54 // (op 2676) 

s_sub_i32 s0, s1, s54 // (op 2676) 


s_mul_hi_i32 s1, s0, s12 // (op 2676) 
s_add_i32 s54, s1, s0 // (op 2676) 

s_ashr_i32 s1, s54, 31 // (op 2676) 
s_cmp_eq_i32 s12, 0 // (op 2676) 
s_mov_b32 s55, scc // (op 2676) 
s_cmp_eq_i32 s55, 0 // (op 2676) 
s_cselect_b32 s57, 0, -1 // (op 2676) 

s_and_b32 s55, s13, 31 // (op 2676) 
s_lshl_b32 s58, 1, s55 // (op 2676) 
s_add_i32 s59, s58, s57 // (op 2676) 


s_and_b32 s57, s1, s59 // (op 2676) 


s_add_i32 s1, s54, s57 // (op 2676) 


s_ashr_i32 s54, s1, s55 // (op 2676) 


s_lshl_b32 s1, s13, 24 // (op 2676) 
s_ashr_i32 s55, s1, 31 // (op 2676) 

s_xor_b32 s1, s54, s55 // (op 2676) 

s_sub_i32 s54, s1, s55 // (op 2676) 


s_mul_i32 s1, s0, s15 // (op 2676) 
s_sub_u32 s55, s16, s1 // (op 2676) 

s_mul_i32 s1, s54, s15 // (op 2676) 
s_add_u32 s1, s1, s55 // (op 2676) 

s_mul_hi_i32 s55, s16, s9 // (op 2676) 
s_add_i32 s57, s55, s16 // (op 2676) 

s_ashr_i32 s55, s57, 31 // (op 2676) 
s_cmp_eq_i32 s9, 0 // (op 2676) 
s_mov_b32 s58, scc // (op 2676) 
s_cmp_eq_i32 s58, 0 // (op 2676) 
s_cselect_b32 s59, 0, -1 // (op 2676) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 2421) 
s_lshl_b32 s60, 1, s58 // (op 2421) 
s_add_i32 s61, s60, s56 // (op 2421) 


s_and_b32 s56, s52, s61 // (op 2421) 


s_add_i32 s52, s53, s56 // (op 2421) 


s_ashr_i32 s53, s52, s58 // (op 2421) 


s_lshl_b32 s52, s46, 24 // (op 2421) 
s_ashr_i32 s56, s52, 31 // (op 2421) 

s_xor_b32 s52, s53, s56 // (op 2421) 

s_sub_i32 s53, s52, s56 // (op 2421) 


s_and_b32 s52, s46, 31 // (op 2676) 
s_lshl_b32 s56, 1, s52 // (op 2676) 
s_add_i32 s58, s56, s59 // (op 2676) 


s_and_b32 s56, s55, s58 // (op 2676) 


s_add_i32 s55, s57, s56 // (op 2676) 


s_ashr_i32 s56, s55, s52 // (op 2676) 


s_lshl_b32 s52, s46, 24 // (op 2676) 
s_ashr_i32 s55, s52, 31 // (op 2676) 

s_xor_b32 s52, s56, s55 // (op 2676) 

s_sub_i32 s56, s52, s55 // (op 2676) 


s_add_u32 s58, s6, 64 // (op 2421) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 2421) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2421) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2421) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2421) 

s_mul_i32 s52, s15, s58 // (op 2421) 
s_mul_i32 s55, s8, s52 // (op 2421) 

s_mul_i32 s52, s53, s55 // (op 2421) 

s_sub_u32 s55, s16, s52 // (op 2421) 

s_mov_b32 s52, s55 // (op 2421) 
s_mul_hi_i32 s57, s52, s14 // (op 2421) 
s_add_i32 s60, s57, s52 // (op 2421) 


s_ashr_i32 s52, s60, 31 // (op 2421) 
s_cmp_eq_i32 s14, 0 // (op 2421) 
s_mov_b32 s57, scc // (op 2421) 
s_cmp_eq_i32 s57, 0 // (op 2421) 
s_cselect_b32 s61, 0, -1 // (op 2421) 

s_and_b32 s57, s48, 31 // (op 2421) 
s_lshl_b32 s62, 1, s57 // (op 2421) 
s_add_i32 s63, s62, s61 // (op 2421) 


s_and_b32 s61, s52, s63 // (op 2421) 


s_add_i32 s52, s60, s61 // (op 2421) 


s_ashr_i32 s60, s52, s57 // (op 2421) 


s_lshl_b32 s52, s48, 24 // (op 2421) 
s_ashr_i32 s57, s52, 31 // (op 2421) 

s_xor_b32 s52, s60, s57 // (op 2421) 

s_sub_i32 s60, s52, s57 // (op 2421) 


s_mul_i32 s52, s47, s15 // (op 2421) 
s_add_u32 s62, s20, 64 // (op 2421) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 2421) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2421) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2421) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 6 // (op 2421) 

s_sub_i32 s57, s62, s52 // (op 2421) 

s_mul_i32 s61, s60, s57 // (op 2421) 

s_sub_u32 s57, s55, s61 // (op 2421) 


s_add_u32 s55, s57, s52 // (op 2421) 


s_cmp_eq_i32 s53, 0 // (op 2421) 
s_mov_b32 s52, scc // (op 2421) 

s_cmp_eq_i32 s52, 0 // (op 2421) 
s_cselect_b32 s53, s55, s19 // (op 2421) 


s_mul_i32 s19, s51, s58 // (op 2421) 


v_lshl_add_u32 v13, s53, 6, v12 // (op 2421) 


s_sub_i32 s51, s18, s19 // (op 2421) 


s_cmp_eq_i32 s52, 0 // (op 2421) 
s_cselect_b32 s53, s60, s51 // (op 2421) 



v_lshl_add_u32 v12, s53, 6, v11 // (op 2421) 


v_mul_lo_u32 v11, v12, s2 // (op 2421) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 2421) 


s_add_u32 s18, s6, 64 // (op 2676) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 2676) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 2676) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 2676) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 6 // (op 2676) 

s_mul_i32 s55, s15, s18 // (op 2676) 
s_mul_i32 s51, s8, s55 // (op 2676) 

s_mul_i32 s55, s56, s51 // (op 2676) 

s_sub_u32 s51, s16, s55 // (op 2676) 

s_mov_b32 s55, s51 // (op 2676) 
s_mul_hi_i32 s52, s55, s14 // (op 2676) 
s_add_i32 s53, s52, s55 // (op 2676) 


s_ashr_i32 s52, s53, 31 // (op 2676) 
s_cmp_eq_i32 s14, 0 // (op 2676) 
s_mov_b32 s55, scc // (op 2676) 
s_cmp_eq_i32 s55, 0 // (op 2676) 
s_cselect_b32 s57, 0, -1 // (op 2676) 

s_and_b32 s55, s48, 31 // (op 2676) 
s_lshl_b32 s58, 1, s55 // (op 2676) 
s_add_i32 s59, s58, s57 // (op 2676) 


s_and_b32 s57, s52, s59 // (op 2676) 


s_add_i32 s52, s53, s57 // (op 2676) 


s_ashr_i32 s53, s52, s55 // (op 2676) 


s_lshl_b32 s52, s48, 24 // (op 2676) 
s_ashr_i32 s55, s52, 31 // (op 2676) 

s_xor_b32 s52, s53, s55 // (op 2676) 

s_sub_i32 s53, s52, s55 // (op 2676) 


s_mul_i32 s52, s47, s15 // (op 2676) 
s_add_u32 s58, s20, 64 // (op 2676) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 2676) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 2676) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 2676) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 6 // (op 2676) 

s_sub_i32 s55, s58, s52 // (op 2676) 

s_mul_i32 s57, s53, s55 // (op 2676) 

s_sub_u32 s55, s51, s57 // (op 2676) 


s_add_u32 s51, s55, s52 // (op 2676) 


s_cmp_eq_i32 s56, 0 // (op 2676) 
s_mov_b32 s52, scc // (op 2676) 

s_cmp_eq_i32 s52, 0 // (op 2676) 
s_cselect_b32 s55, s51, s1 // (op 2676) 


s_mul_i32 s1, s54, s18 // (op 2676) 


v_lshl_add_u32 v11, s55, 6, v14 // (op 2676) 


s_sub_i32 s51, s0, s1 // (op 2676) 


s_cmp_eq_i32 s52, 0 // (op 2676) 
s_cselect_b32 s0, s53, s51 // (op 2676) 



v_lshl_add_u32 v13, s0, 6, v10 // (op 2676) 


v_mul_lo_u32 v10, v13, s4 // (op 2676) 

v_add_lshl_u32 v13, v11, v10, 1 // (op 2676) 


s_mul_i32 s1, s2, s7 // (op 2423) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 2423) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 2423) least significant
s_mul_hi_u32 s19, s2, s6 // (op 2423) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2423) most significant: sum
s_add_u32 s1, s1, s19 // (op 2423) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2423) 

s_mul_i32 s1, s4, s7 // (op 2678) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 2678) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 2678) least significant
s_mul_hi_u32 s52, s4, s6 // (op 2678) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 2678) most significant: sum
s_add_u32 s1, s1, s52 // (op 2678) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 2678) 

s_mov_b32 s56, s22 // (op 2423) reg expression
s_mov_b32 s57, s23 // (op 2423) reg expression
s_mov_b32 s58, s18 // (op 2423) reg expression
s_mov_b32 s59, 131072 // (op 2423) call()

s_mov_b32 s60, s24 // (op 2678) reg expression

s_mov_b32 s61, s25 // (op 2678) reg expression
s_mov_b32 s62, s52 // (op 2678) reg expression
s_mov_b32 s63, 131072 // (op 2678) call()


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
s_mul_hi_i32 s0, s16, s10 // (op 2183) 
v_and_b32 v14, -4, v1 // (op 2183) 
s_add_i32 s1, s0, s16 // (op 2183) 

v_sub_u32 v15, v1, v14 // (op 2183) 

s_ashr_i32 s0, s1, 31 // (op 2183) 
v_lshlrev_b32 v14, 4, v15 // (op 2183) 

s_cmp_eq_i32 s10, 0 // (op 2183) 
s_mov_b32 s18, scc // (op 2183) 
v_lshrrev_b32 v15, 2, v1 // (op 2183) 
s_cmp_eq_i32 s18, 0 // (op 2183) 
s_cselect_b32 s19, 0, -1 // (op 2183) 

s_and_b32 s18, s11, 31 // (op 2183) 
s_lshl_b32 s22, 1, s18 // (op 2183) 
s_add_i32 s23, s22, s19 // (op 2183) 


s_and_b32 s19, s0, s23 // (op 2183) 


s_add_i32 s0, s1, s19 // (op 2183) 


s_ashr_i32 s1, s0, s18 // (op 2183) 


s_lshl_b32 s0, s11, 24 // (op 2183) 
s_ashr_i32 s18, s0, 31 // (op 2183) 

s_xor_b32 s0, s1, s18 // (op 2183) 

s_sub_i32 s1, s0, s18 // (op 2183) 


s_mul_hi_i32 s0, s1, s12 // (op 2183) 
s_add_i32 s18, s0, s1 // (op 2183) 

s_ashr_i32 s0, s18, 31 // (op 2183) 
s_cmp_eq_i32 s12, 0 // (op 2183) 
s_mov_b32 s19, scc // (op 2183) 
s_cmp_eq_i32 s19, 0 // (op 2183) 
s_cselect_b32 s22, 0, -1 // (op 2183) 

s_and_b32 s19, s13, 31 // (op 2183) 
s_lshl_b32 s23, 1, s19 // (op 2183) 
s_add_i32 s24, s23, s22 // (op 2183) 


s_and_b32 s25, s0, s24 // (op 2183) 


s_add_i32 s0, s18, s25 // (op 2183) 


s_ashr_i32 s18, s0, s19 // (op 2183) 


s_lshl_b32 s0, s13, 24 // (op 2183) 
s_ashr_i32 s19, s0, 31 // (op 2183) 

s_xor_b32 s0, s18, s19 // (op 2183) 

s_sub_i32 s18, s0, s19 // (op 2183) 


s_mul_i32 s0, s1, s15 // (op 2183) 

s_sub_u32 s1, s16, s0 // (op 2183) 

s_mul_i32 s0, s18, s15 // (op 2183) 
s_add_u32 s0, s0, s1 // (op 2183) 


s_mul_hi_i32 s1, s16, s9 // (op 2183) 
s_add_i32 s18, s1, s16 // (op 2183) 

s_ashr_i32 s1, s18, 31 // (op 2183) 
s_cmp_eq_i32 s9, 0 // (op 2183) 
s_mov_b32 s19, scc // (op 2183) 
s_cmp_eq_i32 s19, 0 // (op 2183) 
s_cselect_b32 s22, 0, -1 // (op 2183) 

s_and_b32 s19, s46, 31 // (op 2183) 
s_lshl_b32 s23, 1, s19 // (op 2183) 
s_add_i32 s24, s23, s22 // (op 2183) 


s_and_b32 s25, s1, s24 // (op 2183) 


s_add_i32 s1, s18, s25 // (op 2183) 


s_ashr_i32 s18, s1, s19 // (op 2183) 


s_lshl_b32 s1, s46, 24 // (op 2183) 
s_ashr_i32 s19, s1, 31 // (op 2183) 

s_xor_b32 s1, s18, s19 // (op 2183) 

s_sub_i32 s18, s1, s19 // (op 2183) 


s_add_u32 s22, s6, 64 // (op 2183) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2183) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2183) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2183) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2183) 

s_mul_i32 s1, s15, s22 // (op 2183) 

s_mul_i32 s19, s8, s1 // (op 2183) 

s_mul_i32 s1, s18, s19 // (op 2183) 

s_sub_u32 s19, s16, s1 // (op 2183) 

s_mov_b32 s1, s19 // (op 2183) 
s_mul_hi_i32 s22, s1, s14 // (op 2183) 
s_add_i32 s23, s22, s1 // (op 2183) 


s_ashr_i32 s1, s23, 31 // (op 2183) 
s_cmp_eq_i32 s14, 0 // (op 2183) 
s_mov_b32 s22, scc // (op 2183) 
s_cmp_eq_i32 s22, 0 // (op 2183) 
s_cselect_b32 s24, 0, -1 // (op 2183) 

s_and_b32 s22, s48, 31 // (op 2183) 
s_lshl_b32 s25, 1, s22 // (op 2183) 
s_add_i32 s49, s25, s24 // (op 2183) 


s_and_b32 s24, s1, s49 // (op 2183) 


s_add_i32 s1, s23, s24 // (op 2183) 


s_ashr_i32 s23, s1, s22 // (op 2183) 


s_lshl_b32 s1, s48, 24 // (op 2183) 
s_ashr_i32 s22, s1, 31 // (op 2183) 

s_xor_b32 s1, s23, s22 // (op 2183) 

s_sub_i32 s23, s1, s22 // (op 2183) 


s_mul_i32 s1, s47, s15 // (op 2183) 
s_add_u32 s24, s20, 64 // (op 2183) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2183) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 2183) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 2183) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 6 // (op 2183) 

s_sub_i32 s22, s24, s1 // (op 2183) 

s_mul_i32 s24, s23, s22 // (op 2183) 


s_sub_u32 s25, s19, s24 // (op 2183) 


s_add_u32 s19, s25, s1 // (op 2183) 


s_cmp_eq_i32 s18, 0 // (op 2183) 
s_mov_b32 s1, scc // (op 2183) 

s_cmp_eq_i32 s1, 0 // (op 2183) 
s_cselect_b32 s18, s19, s0 // (op 2183) 



v_lshl_add_u32 v16, s18, 6, v14 // (op 2183) 


v_mov_b32 v19, 0 // (op 2183) convert
v_mov_b32 v18, v16 // (op 2183) convert

v_mul_lo_u32 v17, v15, s27 // (op 2183) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 2183) most significant: high of low * low
v_mul_lo_u32 v16, v15, s26 // (op 2183) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 2183) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 2183) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 2183) most significant half



s_mul_i32 s1, s26, s31 // (op 2185) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2185) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2185) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2185) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2185) most significant: sum
s_add_u32 s1, s1, s19 // (op 2185) most significant: sum


s_mov_b32 s52, s28 // (op 2185) reg expression
s_mov_b32 s53, s29 // (op 2185) reg expression
s_mov_b32 s54, s0 // (op 2185) reg expression
s_mov_b32 s55, 131072 // (op 2185) call()

s_mul_hi_i32 s0, s16, s10 // (op 2202) 
v_and_b32 v16, -4, v1 // (op 2202) 
s_add_i32 s1, s0, s16 // (op 2202) 

v_sub_u32 v17, v1, v16 // (op 2202) 

s_ashr_i32 s0, s1, 31 // (op 2202) 
v_lshlrev_b32 v16, 4, v17 // (op 2202) 

s_cmp_eq_i32 s10, 0 // (op 2202) 
s_mov_b32 s18, scc // (op 2202) 
v_lshrrev_b32 v17, 2, v1 // (op 2202) 
s_cmp_eq_i32 s18, 0 // (op 2202) 
s_cselect_b32 s19, 0, -1 // (op 2202) 

s_and_b32 s18, s11, 31 // (op 2202) 
s_lshl_b32 s22, 1, s18 // (op 2202) 
s_add_i32 s23, s22, s19 // (op 2202) 


s_and_b32 s19, s0, s23 // (op 2202) 


s_add_i32 s0, s1, s19 // (op 2202) 


s_ashr_i32 s1, s0, s18 // (op 2202) 


s_lshl_b32 s0, s11, 24 // (op 2202) 
s_ashr_i32 s18, s0, 31 // (op 2202) 

s_xor_b32 s0, s1, s18 // (op 2202) 

s_sub_i32 s1, s0, s18 // (op 2202) 


s_mul_hi_i32 s0, s1, s12 // (op 2202) 
s_add_i32 s18, s0, s1 // (op 2202) 

s_ashr_i32 s0, s18, 31 // (op 2202) 
s_cmp_eq_i32 s12, 0 // (op 2202) 
s_mov_b32 s19, scc // (op 2202) 
s_cmp_eq_i32 s19, 0 // (op 2202) 
s_cselect_b32 s22, 0, -1 // (op 2202) 

s_and_b32 s19, s13, 31 // (op 2202) 
s_lshl_b32 s23, 1, s19 // (op 2202) 
s_add_i32 s24, s23, s22 // (op 2202) 


s_and_b32 s25, s0, s24 // (op 2202) 


s_add_i32 s0, s18, s25 // (op 2202) 


s_ashr_i32 s18, s0, s19 // (op 2202) 


s_lshl_b32 s0, s13, 24 // (op 2202) 
s_ashr_i32 s19, s0, 31 // (op 2202) 

s_xor_b32 s0, s18, s19 // (op 2202) 

s_sub_i32 s18, s0, s19 // (op 2202) 


s_mul_i32 s0, s1, s15 // (op 2202) 

s_sub_u32 s1, s16, s0 // (op 2202) 

s_mul_i32 s0, s18, s15 // (op 2202) 
s_add_u32 s0, s0, s1 // (op 2202) 


s_mul_hi_i32 s1, s16, s9 // (op 2202) 
s_add_i32 s18, s1, s16 // (op 2202) 

s_ashr_i32 s1, s18, 31 // (op 2202) 
s_cmp_eq_i32 s9, 0 // (op 2202) 
s_mov_b32 s19, scc // (op 2202) 
s_cmp_eq_i32 s19, 0 // (op 2202) 
s_cselect_b32 s22, 0, -1 // (op 2202) 

s_and_b32 s19, s46, 31 // (op 2202) 
s_lshl_b32 s23, 1, s19 // (op 2202) 
s_add_i32 s24, s23, s22 // (op 2202) 


s_and_b32 s25, s1, s24 // (op 2202) 


s_add_i32 s1, s18, s25 // (op 2202) 


s_ashr_i32 s18, s1, s19 // (op 2202) 


s_lshl_b32 s1, s46, 24 // (op 2202) 
s_ashr_i32 s19, s1, 31 // (op 2202) 

s_xor_b32 s1, s18, s19 // (op 2202) 

s_sub_i32 s18, s1, s19 // (op 2202) 


s_add_u32 s50, s6, 64 // (op 2202) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2202) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2202) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2202) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2202) 

s_mul_i32 s1, s15, s24 // (op 2202) 

s_mul_i32 s19, s8, s1 // (op 2202) 

s_mul_i32 s1, s18, s19 // (op 2202) 

s_sub_u32 s19, s16, s1 // (op 2202) 

s_mov_b32 s1, s19 // (op 2202) 
s_mul_hi_i32 s22, s1, s14 // (op 2202) 
s_add_i32 s23, s22, s1 // (op 2202) 


s_ashr_i32 s1, s23, 31 // (op 2202) 
s_cmp_eq_i32 s14, 0 // (op 2202) 
s_mov_b32 s22, scc // (op 2202) 
s_cmp_eq_i32 s22, 0 // (op 2202) 
s_cselect_b32 s24, 0, -1 // (op 2202) 

s_and_b32 s22, s48, 31 // (op 2202) 
s_lshl_b32 s25, 1, s22 // (op 2202) 
s_add_i32 s49, s25, s24 // (op 2202) 


s_and_b32 s24, s1, s49 // (op 2202) 


s_add_i32 s1, s23, s24 // (op 2202) 


s_ashr_i32 s23, s1, s22 // (op 2202) 


s_lshl_b32 s1, s48, 24 // (op 2202) 
s_ashr_i32 s22, s1, 31 // (op 2202) 

s_xor_b32 s1, s23, s22 // (op 2202) 

s_sub_i32 s23, s1, s22 // (op 2202) 


s_mul_i32 s1, s47, s15 // (op 2202) 
s_add_u32 s24, s20, 64 // (op 2202) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2202) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2202) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2202) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2202) 

s_sub_i32 s22, s24, s1 // (op 2202) 

s_mul_i32 s24, s23, s22 // (op 2202) 


s_sub_u32 s25, s19, s24 // (op 2202) 


s_add_u32 s19, s25, s1 // (op 2202) 


s_cmp_eq_i32 s18, 0 // (op 2202) 
s_mov_b32 s1, scc // (op 2202) 

s_cmp_eq_i32 s1, 0 // (op 2202) 
s_cselect_b32 s18, s19, s0 // (op 2202) 



v_lshl_add_u32 v18, s18, 6, v16 // (op 2202) 


v_mov_b32 v21, 0 // (op 2202) convert
v_mov_b32 v20, v18 // (op 2202) convert

v_mul_lo_u32 v19, v17, s27 // (op 2202) most significant: low of low * high
v_mul_hi_u32 v16, v17, s26 // (op 2202) most significant: high of low * low
v_mul_lo_u32 v18, v17, s26 // (op 2202) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 2202) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v18 // (op 2202) least significant half
v_addc_co_u32 v17, s[0:1], v21, v19, s[0:1] // (op 2202) most significant half



s_mul_i32 s1, s26, s31 // (op 2204) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2204) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2204) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2204) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2204) most significant: sum
s_add_u32 s1, s1, s19 // (op 2204) most significant: sum


s_mov_b32 s52, s28 // (op 2204) reg expression
s_mov_b32 s53, s29 // (op 2204) reg expression
s_mov_b32 s54, s0 // (op 2204) reg expression
s_mov_b32 s55, 131072 // (op 2204) call()

s_mul_hi_i32 s0, s16, s10 // (op 2247) 
v_lshrrev_b32 v18, 3, v1 // (op 2247) 
s_add_i32 s1, s0, s16 // (op 2247) 

v_and_b32 v19, -8, v1 // (op 2247) 
s_ashr_i32 s0, s1, 31 // (op 2247) 
v_sub_u32 v20, v1, v19 // (op 2247) 

s_cmp_eq_i32 s10, 0 // (op 2247) 
s_mov_b32 s18, scc // (op 2247) 
v_lshlrev_b32 v19, 4, v20 // (op 2247) 

s_cmp_eq_i32 s18, 0 // (op 2247) 
s_cselect_b32 s19, 0, -1 // (op 2247) 

v_mov_b32 v21, 0 // (op 2247) convert
v_mov_b32 v20, v19 // (op 2247) convert

s_and_b32 s18, s11, 31 // (op 2247) 
s_lshl_b32 s22, 1, s18 // (op 2247) 
s_add_i32 s23, s22, s19 // (op 2247) 


s_and_b32 s19, s0, s23 // (op 2247) 


s_add_i32 s0, s1, s19 // (op 2247) 


s_ashr_i32 s1, s0, s18 // (op 2247) 


s_lshl_b32 s0, s11, 24 // (op 2247) 
s_ashr_i32 s18, s0, 31 // (op 2247) 

s_xor_b32 s0, s1, s18 // (op 2247) 

s_sub_i32 s1, s0, s18 // (op 2247) 


s_mul_hi_i32 s0, s1, s12 // (op 2247) 
s_add_i32 s18, s0, s1 // (op 2247) 

s_ashr_i32 s0, s18, 31 // (op 2247) 
s_cmp_eq_i32 s12, 0 // (op 2247) 
s_mov_b32 s19, scc // (op 2247) 
s_cmp_eq_i32 s19, 0 // (op 2247) 
s_cselect_b32 s22, 0, -1 // (op 2247) 

s_and_b32 s19, s13, 31 // (op 2247) 
s_lshl_b32 s23, 1, s19 // (op 2247) 
s_add_i32 s24, s23, s22 // (op 2247) 


s_and_b32 s25, s0, s24 // (op 2247) 


s_add_i32 s0, s18, s25 // (op 2247) 


s_ashr_i32 s18, s0, s19 // (op 2247) 


s_lshl_b32 s0, s13, 24 // (op 2247) 
s_ashr_i32 s19, s0, 31 // (op 2247) 

s_xor_b32 s0, s18, s19 // (op 2247) 

s_sub_i32 s18, s0, s19 // (op 2247) 


s_add_u32 s50, s6, 64 // (op 2247) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2247) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2247) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2247) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2247) 

s_mul_i32 s0, s18, s24 // (op 2247) 

s_sub_i32 s18, s1, s0 // (op 2247) 


s_mul_hi_i32 s19, s16, s9 // (op 2247) 
s_add_i32 s0, s19, s16 // (op 2247) 

s_ashr_i32 s1, s0, 31 // (op 2247) 
s_cmp_eq_i32 s9, 0 // (op 2247) 
s_mov_b32 s19, scc // (op 2247) 
s_cmp_eq_i32 s19, 0 // (op 2247) 
s_cselect_b32 s22, 0, -1 // (op 2247) 

s_and_b32 s19, s46, 31 // (op 2247) 
s_lshl_b32 s23, 1, s19 // (op 2247) 
s_add_i32 s49, s23, s22 // (op 2247) 


s_and_b32 s22, s1, s49 // (op 2247) 


s_add_i32 s1, s0, s22 // (op 2247) 


s_ashr_i32 s0, s1, s19 // (op 2247) 


s_lshl_b32 s1, s46, 24 // (op 2247) 
s_ashr_i32 s19, s1, 31 // (op 2247) 

s_xor_b32 s1, s0, s19 // (op 2247) 

s_sub_i32 s0, s1, s19 // (op 2247) 


s_mul_i32 s1, s15, s24 // (op 2247) 

s_mul_i32 s19, s8, s1 // (op 2247) 

s_mul_i32 s1, s0, s19 // (op 2247) 

s_sub_u32 s19, s16, s1 // (op 2247) 

s_mul_hi_i32 s1, s19, s14 // (op 2247) 
s_add_i32 s22, s1, s19 // (op 2247) 


s_ashr_i32 s1, s22, 31 // (op 2247) 
s_cmp_eq_i32 s14, 0 // (op 2247) 
s_mov_b32 s19, scc // (op 2247) 
s_cmp_eq_i32 s19, 0 // (op 2247) 
s_cselect_b32 s23, 0, -1 // (op 2247) 

s_and_b32 s19, s48, 31 // (op 2247) 
s_lshl_b32 s24, 1, s19 // (op 2247) 
s_add_i32 s25, s24, s23 // (op 2247) 


s_and_b32 s23, s1, s25 // (op 2247) 


s_add_i32 s1, s22, s23 // (op 2247) 


s_ashr_i32 s22, s1, s19 // (op 2247) 


s_lshl_b32 s1, s48, 24 // (op 2247) 
s_ashr_i32 s19, s1, 31 // (op 2247) 

s_xor_b32 s1, s22, s19 // (op 2247) 

s_sub_i32 s22, s1, s19 // (op 2247) 


s_cmp_eq_i32 s0, 0 // (op 2247) 
s_mov_b32 s1, scc // (op 2247) 

s_cmp_eq_i32 s1, 0 // (op 2247) 
s_cselect_b32 s0, s22, s18 // (op 2247) 



v_lshl_add_u32 v19, s0, 6, v18 // (op 2247) 


v_mul_lo_u32 v23, v19, s33 // (op 2247) most significant: low of low * high
v_mul_hi_u32 v18, v19, s32 // (op 2247) most significant: high of low * low
v_mul_lo_u32 v22, v19, s32 // (op 2247) least significant: low of low * low
v_add_u32 v23, v23, v18 // (op 2247) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v22 // (op 2247) least significant half
v_addc_co_u32 v19, s[0:1], v21, v23, s[0:1] // (op 2247) most significant half



s_mul_i32 s1, s32, s7 // (op 2249) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2249) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2249) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2249) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2249) most significant: sum
s_add_u32 s1, s1, s19 // (op 2249) most significant: sum


s_mov_b32 s64, s34 // (op 2249) reg expression
s_mov_b32 s65, s35 // (op 2249) reg expression
s_mov_b32 s66, s0 // (op 2249) reg expression
s_mov_b32 s67, 131072 // (op 2249) call()

s_mul_hi_i32 s0, s16, s10 // (op 2266) 
v_lshrrev_b32 v20, 3, v1 // (op 2266) 
s_add_i32 s1, s0, s16 // (op 2266) 

v_and_b32 v21, -8, v1 // (op 2266) 
s_ashr_i32 s0, s1, 31 // (op 2266) 
v_sub_u32 v22, v1, v21 // (op 2266) 

s_cmp_eq_i32 s10, 0 // (op 2266) 
s_mov_b32 s18, scc // (op 2266) 
v_lshlrev_b32 v21, 4, v22 // (op 2266) 

s_cmp_eq_i32 s18, 0 // (op 2266) 
s_cselect_b32 s19, 0, -1 // (op 2266) 

v_mov_b32 v23, 0 // (op 2266) convert
v_mov_b32 v22, v21 // (op 2266) convert

s_and_b32 s18, s11, 31 // (op 2266) 
s_lshl_b32 s22, 1, s18 // (op 2266) 
s_add_i32 s23, s22, s19 // (op 2266) 


s_and_b32 s19, s0, s23 // (op 2266) 


s_add_i32 s0, s1, s19 // (op 2266) 


s_ashr_i32 s1, s0, s18 // (op 2266) 


s_lshl_b32 s0, s11, 24 // (op 2266) 
s_ashr_i32 s18, s0, 31 // (op 2266) 

s_xor_b32 s0, s1, s18 // (op 2266) 

s_sub_i32 s1, s0, s18 // (op 2266) 


s_mul_hi_i32 s0, s1, s12 // (op 2266) 
s_add_i32 s18, s0, s1 // (op 2266) 

s_ashr_i32 s0, s18, 31 // (op 2266) 
s_cmp_eq_i32 s12, 0 // (op 2266) 
s_mov_b32 s19, scc // (op 2266) 
s_cmp_eq_i32 s19, 0 // (op 2266) 
s_cselect_b32 s22, 0, -1 // (op 2266) 

s_and_b32 s19, s13, 31 // (op 2266) 
s_lshl_b32 s23, 1, s19 // (op 2266) 
s_add_i32 s24, s23, s22 // (op 2266) 


s_and_b32 s25, s0, s24 // (op 2266) 


s_add_i32 s0, s18, s25 // (op 2266) 


s_ashr_i32 s18, s0, s19 // (op 2266) 


s_lshl_b32 s0, s13, 24 // (op 2266) 
s_ashr_i32 s19, s0, 31 // (op 2266) 

s_xor_b32 s0, s18, s19 // (op 2266) 

s_sub_i32 s18, s0, s19 // (op 2266) 


s_add_u32 s50, s6, 64 // (op 2266) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2266) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2266) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2266) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2266) 

s_mul_i32 s0, s18, s24 // (op 2266) 

s_sub_i32 s18, s1, s0 // (op 2266) 


s_mul_hi_i32 s19, s16, s9 // (op 2266) 
s_add_i32 s0, s19, s16 // (op 2266) 

s_ashr_i32 s1, s0, 31 // (op 2266) 
s_cmp_eq_i32 s9, 0 // (op 2266) 
s_mov_b32 s19, scc // (op 2266) 
s_cmp_eq_i32 s19, 0 // (op 2266) 
s_cselect_b32 s22, 0, -1 // (op 2266) 

s_and_b32 s19, s46, 31 // (op 2266) 
s_lshl_b32 s23, 1, s19 // (op 2266) 
s_add_i32 s49, s23, s22 // (op 2266) 


s_and_b32 s22, s1, s49 // (op 2266) 


s_add_i32 s1, s0, s22 // (op 2266) 


s_ashr_i32 s0, s1, s19 // (op 2266) 


s_lshl_b32 s1, s46, 24 // (op 2266) 
s_ashr_i32 s19, s1, 31 // (op 2266) 

s_xor_b32 s1, s0, s19 // (op 2266) 

s_sub_i32 s0, s1, s19 // (op 2266) 


s_mul_i32 s1, s15, s24 // (op 2266) 

s_mul_i32 s19, s8, s1 // (op 2266) 

s_mul_i32 s1, s0, s19 // (op 2266) 

s_sub_u32 s19, s16, s1 // (op 2266) 

s_mul_hi_i32 s1, s19, s14 // (op 2266) 
s_add_i32 s22, s1, s19 // (op 2266) 


s_ashr_i32 s1, s22, 31 // (op 2266) 
s_cmp_eq_i32 s14, 0 // (op 2266) 
s_mov_b32 s19, scc // (op 2266) 
s_cmp_eq_i32 s19, 0 // (op 2266) 
s_cselect_b32 s23, 0, -1 // (op 2266) 

s_and_b32 s19, s48, 31 // (op 2266) 
s_lshl_b32 s24, 1, s19 // (op 2266) 
s_add_i32 s25, s24, s23 // (op 2266) 


s_and_b32 s23, s1, s25 // (op 2266) 


s_add_i32 s1, s22, s23 // (op 2266) 


s_ashr_i32 s22, s1, s19 // (op 2266) 


s_lshl_b32 s1, s48, 24 // (op 2266) 
s_ashr_i32 s19, s1, 31 // (op 2266) 

s_xor_b32 s1, s22, s19 // (op 2266) 

s_sub_i32 s22, s1, s19 // (op 2266) 


s_cmp_eq_i32 s0, 0 // (op 2266) 
s_mov_b32 s1, scc // (op 2266) 

s_cmp_eq_i32 s1, 0 // (op 2266) 
s_cselect_b32 s0, s22, s18 // (op 2266) 



v_lshl_add_u32 v21, s0, 6, v20 // (op 2266) 


v_mul_lo_u32 v25, v21, s33 // (op 2266) most significant: low of low * high
v_mul_hi_u32 v20, v21, s32 // (op 2266) most significant: high of low * low
v_mul_lo_u32 v24, v21, s32 // (op 2266) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 2266) most significant: sum


v_add_co_u32 v20, s[0:1], v22, v24 // (op 2266) least significant half
v_addc_co_u32 v21, s[0:1], v23, v25, s[0:1] // (op 2266) most significant half



s_mul_i32 s1, s32, s7 // (op 2268) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2268) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2268) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2268) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2268) most significant: sum
s_add_u32 s1, s1, s19 // (op 2268) most significant: sum


s_mov_b32 s64, s34 // (op 2268) reg expression
s_mov_b32 s65, s35 // (op 2268) reg expression
s_mov_b32 s66, s0 // (op 2268) reg expression
s_mov_b32 s67, 131072 // (op 2268) call()

s_mul_hi_i32 s0, s16, s10 // (op 2311) 
v_and_b32 v22, -64, v1 // (op 2311) 
s_add_i32 s1, s0, s16 // (op 2311) 

v_sub_u32 v23, v1, v22 // (op 2311) 

s_ashr_i32 s0, s1, 31 // (op 2311) 
v_lshrrev_b32 v22, 6, v1 // (op 2311) 
s_cmp_eq_i32 s10, 0 // (op 2311) 
s_mov_b32 s18, scc // (op 2311) 
s_cmp_eq_i32 s18, 0 // (op 2311) 
s_cselect_b32 s19, 0, -1 // (op 2311) 

s_and_b32 s18, s11, 31 // (op 2311) 
s_lshl_b32 s22, 1, s18 // (op 2311) 
s_add_i32 s23, s22, s19 // (op 2311) 


s_and_b32 s19, s0, s23 // (op 2311) 


s_add_i32 s0, s1, s19 // (op 2311) 


s_ashr_i32 s1, s0, s18 // (op 2311) 


s_lshl_b32 s0, s11, 24 // (op 2311) 
s_ashr_i32 s18, s0, 31 // (op 2311) 

s_xor_b32 s0, s1, s18 // (op 2311) 

s_sub_i32 s1, s0, s18 // (op 2311) 


s_mul_hi_i32 s0, s1, s12 // (op 2311) 
s_add_i32 s18, s0, s1 // (op 2311) 

s_ashr_i32 s0, s18, 31 // (op 2311) 
s_cmp_eq_i32 s12, 0 // (op 2311) 
s_mov_b32 s19, scc // (op 2311) 
s_cmp_eq_i32 s19, 0 // (op 2311) 
s_cselect_b32 s22, 0, -1 // (op 2311) 

s_and_b32 s19, s13, 31 // (op 2311) 
s_lshl_b32 s23, 1, s19 // (op 2311) 
s_add_i32 s24, s23, s22 // (op 2311) 


s_and_b32 s25, s0, s24 // (op 2311) 


s_add_i32 s0, s18, s25 // (op 2311) 


s_ashr_i32 s18, s0, s19 // (op 2311) 


s_lshl_b32 s0, s13, 24 // (op 2311) 
s_ashr_i32 s19, s0, 31 // (op 2311) 

s_xor_b32 s0, s18, s19 // (op 2311) 

s_sub_i32 s18, s0, s19 // (op 2311) 


s_mul_i32 s0, s1, s15 // (op 2311) 

s_sub_u32 s1, s16, s0 // (op 2311) 

s_mul_i32 s0, s18, s15 // (op 2311) 
s_add_u32 s0, s0, s1 // (op 2311) 


s_mul_hi_i32 s1, s16, s9 // (op 2311) 
s_add_i32 s18, s1, s16 // (op 2311) 

s_ashr_i32 s1, s18, 31 // (op 2311) 
s_cmp_eq_i32 s9, 0 // (op 2311) 
s_mov_b32 s19, scc // (op 2311) 
s_cmp_eq_i32 s19, 0 // (op 2311) 
s_cselect_b32 s22, 0, -1 // (op 2311) 

s_and_b32 s19, s46, 31 // (op 2311) 
s_lshl_b32 s23, 1, s19 // (op 2311) 
s_add_i32 s24, s23, s22 // (op 2311) 


s_and_b32 s25, s1, s24 // (op 2311) 


s_add_i32 s1, s18, s25 // (op 2311) 


s_ashr_i32 s18, s1, s19 // (op 2311) 


s_lshl_b32 s1, s46, 24 // (op 2311) 
s_ashr_i32 s19, s1, 31 // (op 2311) 

s_xor_b32 s1, s18, s19 // (op 2311) 

s_sub_i32 s18, s1, s19 // (op 2311) 


s_add_u32 s50, s6, 64 // (op 2311) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2311) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2311) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2311) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2311) 

s_mul_i32 s1, s15, s24 // (op 2311) 

s_mul_i32 s19, s8, s1 // (op 2311) 

s_mul_i32 s1, s18, s19 // (op 2311) 

s_sub_u32 s19, s16, s1 // (op 2311) 

s_mov_b32 s1, s19 // (op 2311) 
s_mul_hi_i32 s22, s1, s14 // (op 2311) 
s_add_i32 s23, s22, s1 // (op 2311) 


s_ashr_i32 s1, s23, 31 // (op 2311) 
s_cmp_eq_i32 s14, 0 // (op 2311) 
s_mov_b32 s22, scc // (op 2311) 
s_cmp_eq_i32 s22, 0 // (op 2311) 
s_cselect_b32 s24, 0, -1 // (op 2311) 

s_and_b32 s22, s48, 31 // (op 2311) 
s_lshl_b32 s25, 1, s22 // (op 2311) 
s_add_i32 s49, s25, s24 // (op 2311) 


s_and_b32 s24, s1, s49 // (op 2311) 


s_add_i32 s1, s23, s24 // (op 2311) 


s_ashr_i32 s23, s1, s22 // (op 2311) 


s_lshl_b32 s1, s48, 24 // (op 2311) 
s_ashr_i32 s22, s1, 31 // (op 2311) 

s_xor_b32 s1, s23, s22 // (op 2311) 

s_sub_i32 s23, s1, s22 // (op 2311) 


s_mul_i32 s1, s47, s15 // (op 2311) 
s_add_u32 s24, s20, 64 // (op 2311) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2311) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2311) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2311) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2311) 

s_sub_i32 s22, s24, s1 // (op 2311) 

s_mul_i32 s24, s23, s22 // (op 2311) 


s_sub_u32 s25, s19, s24 // (op 2311) 


s_add_u32 s19, s25, s1 // (op 2311) 


s_cmp_eq_i32 s18, 0 // (op 2311) 
s_mov_b32 s1, scc // (op 2311) 

s_cmp_eq_i32 s1, 0 // (op 2311) 
s_cselect_b32 s18, s19, s0 // (op 2311) 



v_lshl_add_u32 v24, s18, 6, v23 // (op 2311) 


v_mul_lo_u32 v23, v22, s36 // (op 2311) 

v_add_u32 v22, v24, v23 // (op 2311) 


s_mul_i32 s1, s36, s41 // (op 2313) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2313) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2313) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2313) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2313) most significant: sum
s_add_u32 s1, s1, s19 // (op 2313) most significant: sum


s_mov_b32 s68, s38 // (op 2313) reg expression
s_mov_b32 s69, s39 // (op 2313) reg expression
s_mov_b32 s70, s0 // (op 2313) reg expression
s_mov_b32 s71, 131072 // (op 2313) call()

s_mul_hi_i32 s0, s16, s10 // (op 2330) 
v_and_b32 v23, -64, v1 // (op 2330) 
s_add_i32 s1, s0, s16 // (op 2330) 

v_sub_u32 v24, v1, v23 // (op 2330) 

s_ashr_i32 s0, s1, 31 // (op 2330) 
v_lshrrev_b32 v23, 6, v1 // (op 2330) 
s_cmp_eq_i32 s10, 0 // (op 2330) 
s_mov_b32 s18, scc // (op 2330) 
s_cmp_eq_i32 s18, 0 // (op 2330) 
s_cselect_b32 s19, 0, -1 // (op 2330) 

s_and_b32 s18, s11, 31 // (op 2330) 
s_lshl_b32 s22, 1, s18 // (op 2330) 
s_add_i32 s23, s22, s19 // (op 2330) 


s_and_b32 s19, s0, s23 // (op 2330) 


s_add_i32 s0, s1, s19 // (op 2330) 


s_ashr_i32 s1, s0, s18 // (op 2330) 


s_lshl_b32 s0, s11, 24 // (op 2330) 
s_ashr_i32 s18, s0, 31 // (op 2330) 

s_xor_b32 s0, s1, s18 // (op 2330) 

s_sub_i32 s1, s0, s18 // (op 2330) 


s_mul_hi_i32 s0, s1, s12 // (op 2330) 
s_add_i32 s18, s0, s1 // (op 2330) 

s_ashr_i32 s0, s18, 31 // (op 2330) 
s_cmp_eq_i32 s12, 0 // (op 2330) 
s_mov_b32 s19, scc // (op 2330) 
s_cmp_eq_i32 s19, 0 // (op 2330) 
s_cselect_b32 s22, 0, -1 // (op 2330) 

s_and_b32 s19, s13, 31 // (op 2330) 
s_lshl_b32 s23, 1, s19 // (op 2330) 
s_add_i32 s24, s23, s22 // (op 2330) 


s_and_b32 s25, s0, s24 // (op 2330) 


s_add_i32 s0, s18, s25 // (op 2330) 


s_ashr_i32 s18, s0, s19 // (op 2330) 


s_lshl_b32 s0, s13, 24 // (op 2330) 
s_ashr_i32 s19, s0, 31 // (op 2330) 

s_xor_b32 s0, s18, s19 // (op 2330) 

s_sub_i32 s18, s0, s19 // (op 2330) 


s_mul_i32 s0, s1, s15 // (op 2330) 

s_sub_u32 s1, s16, s0 // (op 2330) 

s_mul_i32 s0, s18, s15 // (op 2330) 
s_add_u32 s0, s0, s1 // (op 2330) 


s_mul_hi_i32 s1, s16, s9 // (op 2330) 
s_add_i32 s18, s1, s16 // (op 2330) 

s_ashr_i32 s1, s18, 31 // (op 2330) 
s_cmp_eq_i32 s9, 0 // (op 2330) 
s_mov_b32 s19, scc // (op 2330) 
s_cmp_eq_i32 s19, 0 // (op 2330) 
s_cselect_b32 s22, 0, -1 // (op 2330) 

s_and_b32 s19, s46, 31 // (op 2330) 
s_lshl_b32 s23, 1, s19 // (op 2330) 
s_add_i32 s24, s23, s22 // (op 2330) 


s_and_b32 s25, s1, s24 // (op 2330) 


s_add_i32 s1, s18, s25 // (op 2330) 


s_ashr_i32 s18, s1, s19 // (op 2330) 


s_lshl_b32 s1, s46, 24 // (op 2330) 
s_ashr_i32 s19, s1, 31 // (op 2330) 

s_xor_b32 s1, s18, s19 // (op 2330) 

s_sub_i32 s18, s1, s19 // (op 2330) 


s_add_u32 s50, s6, 64 // (op 2330) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2330) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2330) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2330) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2330) 

s_mul_i32 s1, s15, s24 // (op 2330) 

s_mul_i32 s19, s8, s1 // (op 2330) 

s_mul_i32 s1, s18, s19 // (op 2330) 

s_sub_u32 s19, s16, s1 // (op 2330) 

s_mov_b32 s1, s19 // (op 2330) 
s_mul_hi_i32 s22, s1, s14 // (op 2330) 
s_add_i32 s23, s22, s1 // (op 2330) 


s_ashr_i32 s1, s23, 31 // (op 2330) 
s_cmp_eq_i32 s14, 0 // (op 2330) 
s_mov_b32 s22, scc // (op 2330) 
s_cmp_eq_i32 s22, 0 // (op 2330) 
s_cselect_b32 s24, 0, -1 // (op 2330) 

s_and_b32 s22, s48, 31 // (op 2330) 
s_lshl_b32 s25, 1, s22 // (op 2330) 
s_add_i32 s49, s25, s24 // (op 2330) 


s_and_b32 s24, s1, s49 // (op 2330) 


s_add_i32 s1, s23, s24 // (op 2330) 


s_ashr_i32 s23, s1, s22 // (op 2330) 


s_lshl_b32 s1, s48, 24 // (op 2330) 
s_ashr_i32 s22, s1, 31 // (op 2330) 

s_xor_b32 s1, s23, s22 // (op 2330) 

s_sub_i32 s23, s1, s22 // (op 2330) 


s_mul_i32 s1, s47, s15 // (op 2330) 
s_add_u32 s24, s20, 64 // (op 2330) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2330) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2330) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2330) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2330) 

s_sub_i32 s22, s24, s1 // (op 2330) 

s_mul_i32 s24, s23, s22 // (op 2330) 


s_sub_u32 s25, s19, s24 // (op 2330) 


s_add_u32 s19, s25, s1 // (op 2330) 


s_cmp_eq_i32 s18, 0 // (op 2330) 
s_mov_b32 s1, scc // (op 2330) 

s_cmp_eq_i32 s1, 0 // (op 2330) 
s_cselect_b32 s18, s19, s0 // (op 2330) 



v_lshl_add_u32 v25, s18, 6, v24 // (op 2330) 


v_mul_lo_u32 v24, v23, s36 // (op 2330) 

v_add_u32 v23, v25, v24 // (op 2330) 


s_mul_i32 s1, s36, s41 // (op 2332) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2332) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2332) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2332) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2332) most significant: sum
s_add_u32 s1, s1, s19 // (op 2332) most significant: sum


s_mov_b32 s68, s38 // (op 2332) reg expression
s_mov_b32 s69, s39 // (op 2332) reg expression
s_mov_b32 s70, s0 // (op 2332) reg expression
s_mov_b32 s71, 131072 // (op 2332) call()

s_mul_hi_i32 s0, s16, s10 // (op 2375) 
v_lshrrev_b32 v24, 2, v1 // (op 2375) 
s_add_i32 s1, s0, s16 // (op 2375) 

v_and_b32 v25, -4, v1 // (op 2375) 
s_ashr_i32 s0, s1, 31 // (op 2375) 
v_sub_u32 v26, v1, v25 // (op 2375) 

s_cmp_eq_i32 s10, 0 // (op 2375) 
s_mov_b32 s18, scc // (op 2375) 
s_cmp_eq_i32 s18, 0 // (op 2375) 
s_cselect_b32 s19, 0, -1 // (op 2375) 

s_and_b32 s18, s11, 31 // (op 2375) 
s_lshl_b32 s22, 1, s18 // (op 2375) 
s_add_i32 s23, s22, s19 // (op 2375) 


s_and_b32 s19, s0, s23 // (op 2375) 


s_add_i32 s0, s1, s19 // (op 2375) 


s_ashr_i32 s1, s0, s18 // (op 2375) 


s_lshl_b32 s0, s11, 24 // (op 2375) 
s_ashr_i32 s18, s0, 31 // (op 2375) 

s_xor_b32 s0, s1, s18 // (op 2375) 

s_sub_i32 s1, s0, s18 // (op 2375) 


s_mul_hi_i32 s0, s1, s12 // (op 2375) 
s_add_i32 s18, s0, s1 // (op 2375) 

s_ashr_i32 s0, s18, 31 // (op 2375) 
s_cmp_eq_i32 s12, 0 // (op 2375) 
s_mov_b32 s19, scc // (op 2375) 
s_cmp_eq_i32 s19, 0 // (op 2375) 
s_cselect_b32 s22, 0, -1 // (op 2375) 

s_and_b32 s19, s13, 31 // (op 2375) 
s_lshl_b32 s23, 1, s19 // (op 2375) 
s_add_i32 s24, s23, s22 // (op 2375) 


s_and_b32 s25, s0, s24 // (op 2375) 


s_add_i32 s0, s18, s25 // (op 2375) 


s_ashr_i32 s18, s0, s19 // (op 2375) 


s_lshl_b32 s0, s13, 24 // (op 2375) 
s_ashr_i32 s19, s0, 31 // (op 2375) 

s_xor_b32 s0, s18, s19 // (op 2375) 

s_sub_i32 s18, s0, s19 // (op 2375) 


s_add_u32 s50, s6, 64 // (op 2375) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2375) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2375) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2375) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2375) 

s_mul_i32 s0, s18, s24 // (op 2375) 

s_sub_i32 s18, s1, s0 // (op 2375) 


s_mul_hi_i32 s19, s16, s9 // (op 2375) 
s_add_i32 s0, s19, s16 // (op 2375) 

s_ashr_i32 s1, s0, 31 // (op 2375) 
s_cmp_eq_i32 s9, 0 // (op 2375) 
s_mov_b32 s19, scc // (op 2375) 
s_cmp_eq_i32 s19, 0 // (op 2375) 
s_cselect_b32 s22, 0, -1 // (op 2375) 

s_and_b32 s19, s46, 31 // (op 2375) 
s_lshl_b32 s23, 1, s19 // (op 2375) 
s_add_i32 s49, s23, s22 // (op 2375) 


s_and_b32 s22, s1, s49 // (op 2375) 


s_add_i32 s1, s0, s22 // (op 2375) 


s_ashr_i32 s0, s1, s19 // (op 2375) 


s_lshl_b32 s1, s46, 24 // (op 2375) 
s_ashr_i32 s19, s1, 31 // (op 2375) 

s_xor_b32 s1, s0, s19 // (op 2375) 

s_sub_i32 s0, s1, s19 // (op 2375) 


s_mul_i32 s1, s15, s24 // (op 2375) 

s_mul_i32 s19, s8, s1 // (op 2375) 

s_mul_i32 s1, s0, s19 // (op 2375) 

s_sub_u32 s19, s16, s1 // (op 2375) 

s_mul_hi_i32 s1, s19, s14 // (op 2375) 
s_add_i32 s22, s1, s19 // (op 2375) 


s_ashr_i32 s1, s22, 31 // (op 2375) 
s_cmp_eq_i32 s14, 0 // (op 2375) 
s_mov_b32 s19, scc // (op 2375) 
s_cmp_eq_i32 s19, 0 // (op 2375) 
s_cselect_b32 s23, 0, -1 // (op 2375) 

s_and_b32 s19, s48, 31 // (op 2375) 
s_lshl_b32 s24, 1, s19 // (op 2375) 
s_add_i32 s25, s24, s23 // (op 2375) 


s_and_b32 s23, s1, s25 // (op 2375) 


s_add_i32 s1, s22, s23 // (op 2375) 


s_ashr_i32 s22, s1, s19 // (op 2375) 


s_lshl_b32 s1, s48, 24 // (op 2375) 
s_ashr_i32 s19, s1, 31 // (op 2375) 

s_xor_b32 s1, s22, s19 // (op 2375) 

s_sub_i32 s22, s1, s19 // (op 2375) 


s_cmp_eq_i32 s0, 0 // (op 2375) 
s_mov_b32 s1, scc // (op 2375) 

s_cmp_eq_i32 s1, 0 // (op 2375) 
s_cselect_b32 s0, s22, s18 // (op 2375) 



v_lshl_add_u32 v25, s0, 6, v24 // (op 2375) 


v_mul_lo_u32 v24, v25, s42 // (op 2375) 

v_add_u32 v25, v26, v24 // (op 2375) 


s_mul_i32 s1, s42, s7 // (op 2377) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2377) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2377) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2377) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2377) most significant: sum
s_add_u32 s1, s1, s19 // (op 2377) most significant: sum


s_mov_b32 s72, s44 // (op 2377) reg expression
s_mov_b32 s73, s45 // (op 2377) reg expression
s_mov_b32 s74, s0 // (op 2377) reg expression
s_mov_b32 s75, 131072 // (op 2377) call()

s_mul_hi_i32 s0, s16, s10 // (op 2394) 
v_lshrrev_b32 v24, 2, v1 // (op 2394) 
s_add_i32 s1, s0, s16 // (op 2394) 

v_and_b32 v26, -4, v1 // (op 2394) 
s_ashr_i32 s0, s1, 31 // (op 2394) 
v_sub_u32 v27, v1, v26 // (op 2394) 

s_cmp_eq_i32 s10, 0 // (op 2394) 
s_mov_b32 s18, scc // (op 2394) 
s_cmp_eq_i32 s18, 0 // (op 2394) 
s_cselect_b32 s19, 0, -1 // (op 2394) 

s_and_b32 s18, s11, 31 // (op 2394) 
s_lshl_b32 s22, 1, s18 // (op 2394) 
s_add_i32 s23, s22, s19 // (op 2394) 


s_and_b32 s19, s0, s23 // (op 2394) 


s_add_i32 s0, s1, s19 // (op 2394) 


s_ashr_i32 s1, s0, s18 // (op 2394) 


s_lshl_b32 s0, s11, 24 // (op 2394) 
s_ashr_i32 s18, s0, 31 // (op 2394) 

s_xor_b32 s0, s1, s18 // (op 2394) 

s_sub_i32 s1, s0, s18 // (op 2394) 


s_mul_hi_i32 s0, s1, s12 // (op 2394) 
s_add_i32 s18, s0, s1 // (op 2394) 

s_ashr_i32 s0, s18, 31 // (op 2394) 
s_cmp_eq_i32 s12, 0 // (op 2394) 
s_mov_b32 s19, scc // (op 2394) 
s_cmp_eq_i32 s19, 0 // (op 2394) 
s_cselect_b32 s22, 0, -1 // (op 2394) 

s_and_b32 s19, s13, 31 // (op 2394) 
s_lshl_b32 s23, 1, s19 // (op 2394) 
s_add_i32 s24, s23, s22 // (op 2394) 


s_and_b32 s25, s0, s24 // (op 2394) 


s_add_i32 s0, s18, s25 // (op 2394) 


s_ashr_i32 s18, s0, s19 // (op 2394) 


s_lshl_b32 s0, s13, 24 // (op 2394) 
s_ashr_i32 s19, s0, 31 // (op 2394) 

s_xor_b32 s0, s18, s19 // (op 2394) 

s_sub_i32 s18, s0, s19 // (op 2394) 


s_add_u32 s50, s6, 64 // (op 2394) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2394) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2394) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2394) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2394) 

s_mul_i32 s0, s18, s24 // (op 2394) 

s_sub_i32 s18, s1, s0 // (op 2394) 


s_mul_hi_i32 s19, s16, s9 // (op 2394) 
s_add_i32 s0, s19, s16 // (op 2394) 

s_ashr_i32 s1, s0, 31 // (op 2394) 
s_cmp_eq_i32 s9, 0 // (op 2394) 
s_mov_b32 s19, scc // (op 2394) 
s_cmp_eq_i32 s19, 0 // (op 2394) 
s_cselect_b32 s22, 0, -1 // (op 2394) 

s_and_b32 s19, s46, 31 // (op 2394) 
s_lshl_b32 s23, 1, s19 // (op 2394) 
s_add_i32 s49, s23, s22 // (op 2394) 


s_and_b32 s22, s1, s49 // (op 2394) 


s_add_i32 s1, s0, s22 // (op 2394) 


s_ashr_i32 s0, s1, s19 // (op 2394) 


s_lshl_b32 s1, s46, 24 // (op 2394) 
s_ashr_i32 s19, s1, 31 // (op 2394) 

s_xor_b32 s1, s0, s19 // (op 2394) 

s_sub_i32 s0, s1, s19 // (op 2394) 


s_mul_i32 s1, s15, s24 // (op 2394) 

s_mul_i32 s19, s8, s1 // (op 2394) 

s_mul_i32 s1, s0, s19 // (op 2394) 

s_sub_u32 s19, s16, s1 // (op 2394) 

s_mul_hi_i32 s1, s19, s14 // (op 2394) 
s_add_i32 s22, s1, s19 // (op 2394) 


s_ashr_i32 s1, s22, 31 // (op 2394) 
s_cmp_eq_i32 s14, 0 // (op 2394) 
s_mov_b32 s19, scc // (op 2394) 
s_cmp_eq_i32 s19, 0 // (op 2394) 
s_cselect_b32 s23, 0, -1 // (op 2394) 

s_and_b32 s19, s48, 31 // (op 2394) 
s_lshl_b32 s24, 1, s19 // (op 2394) 
s_add_i32 s25, s24, s23 // (op 2394) 


s_and_b32 s23, s1, s25 // (op 2394) 


s_add_i32 s1, s22, s23 // (op 2394) 


s_ashr_i32 s22, s1, s19 // (op 2394) 


s_lshl_b32 s1, s48, 24 // (op 2394) 
s_ashr_i32 s19, s1, 31 // (op 2394) 

s_xor_b32 s1, s22, s19 // (op 2394) 

s_sub_i32 s22, s1, s19 // (op 2394) 


s_cmp_eq_i32 s0, 0 // (op 2394) 
s_mov_b32 s1, scc // (op 2394) 

s_cmp_eq_i32 s1, 0 // (op 2394) 
s_cselect_b32 s0, s22, s18 // (op 2394) 



v_lshl_add_u32 v26, s0, 6, v24 // (op 2394) 


v_mul_lo_u32 v24, v26, s42 // (op 2394) 

v_add_u32 v26, v27, v24 // (op 2394) 


s_mul_i32 s1, s42, s7 // (op 2396) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2396) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2396) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2396) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2396) most significant: sum
s_add_u32 s1, s1, s19 // (op 2396) most significant: sum


s_mov_b32 s72, s44 // (op 2396) reg expression
s_mov_b32 s73, s45 // (op 2396) reg expression
s_mov_b32 s74, s0 // (op 2396) reg expression
s_mov_b32 s75, 131072 // (op 2396) call()

v_and_b32 v24, -8, v1 // (op 2522) 
v_sub_u32 v27, v1, v24 // (op 2522) 

v_lshlrev_b32 v24, 4, v27 // (op 2522) 

v_lshrrev_b32 v27, 3, v1 // (op 2522) 
v_lshl_add_u32 v28, v27, 7, v24 // (op 2522) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 2522) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[32:33], 5 // (op 2129) 
v_mov_b64 v[28:29], s[18:19] // (op 2129) convert


v_mov_b64 v[30:31], v[18:19] // (op 2129) 
s_mov_b32 m0, s0 // (op 2129) reg expression
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 2129) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 2129) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2129) most significant half

s_add_u32 m0, m0, 4096 // (op 2129) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 2129) Load value direct to lds






buffer_load_ubyte v24, v25, s[72:75], 0 offen offset: 0 // (op 1959) Load value
v_and_b32 v27, -4, v1 // (op 2465) 
v_sub_u32 v28, v1, v27 // (op 2465) 

v_lshlrev_b32 v27, 4, v28 // (op 2465) 

v_lshrrev_b32 v28, 2, v1 // (op 2465) 
v_lshl_add_u32 v29, v28, 6, v27 // (op 2465) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 2465) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 6 // (op 2134) 
v_mov_b64 v[28:29], s[18:19] // (op 2134) convert


v_mov_b64 v[30:31], v[14:15] // (op 2134) reg expression
s_add_u32 s1, 8192, s0 // (op 2134) 
s_mov_b32 m0, s1 // (op 2134) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 2134) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 2134) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2134) most significant half

s_add_u32 m0, m0, 4096 // (op 2134) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 2134) Load value direct to lds






v_mov_b32 v27, v22 // (op 1965) reg expression

buffer_load_ubyte v28, v27, s[68:71], 0 offen offset: 0 // (op 1965) Load value

v_and_b32 v27, -8, v1 // (op 2749) 
v_sub_u32 v29, v1, v27 // (op 2749) 

v_lshlrev_b32 v27, 4, v29 // (op 2749) 

v_lshrrev_b32 v29, 3, v1 // (op 2749) 
v_lshl_add_u32 v30, v29, 7, v27 // (op 2749) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 2749) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[32:33], 5 // (op 2139) 
v_mov_b64 v[30:31], s[18:19] // (op 2139) convert


v_mov_b64 v[32:33], v[18:19] // (op 2139) 
s_add_u32 s1, 16384, s0 // (op 2139) 
s_mov_b32 m0, s1 // (op 2139) reg expression
s_mov_b32 s18, 128 // (op 2139) call()
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 2139) Load value direct to lds
v_add_co_u32 v32, s[50:51], v32, v30 // (op 2139) least significant half
v_addc_co_u32 v33, s[50:51], 0, v33, s[50:51] // (op 2139) most significant half

s_add_u32 m0, m0, 4096 // (op 2139) 
buffer_load_dwordx4 v32, s[64:67], s18 offen offset: 0 lds // (op 2139) Load value direct to lds









buffer_load_ubyte v27, v25, s[72:75], 0 offen offset:4 // (op 1971) Load value

v_and_b32 v25, -4, v1 // (op 2705) 
v_sub_u32 v18, v1, v25 // (op 2705) 

v_lshlrev_b32 v19, 4, v18 // (op 2705) 

v_lshrrev_b32 v18, 2, v1 // (op 2705) 
v_lshl_add_u32 v25, v18, 6, v19 // (op 2705) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 2705) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[26:27], 6 // (op 2141) 
v_mov_b64 v[18:19], s[18:19] // (op 2141) convert


s_lshl_b64 s[18:19], s[26:27], 7 // (op 2141) 
v_mov_b32 v25, s18 // (op 2141) 
v_mov_b32 v29, s19 // (op 2141) 
v_add_co_u32 v30, s[50:51], v14, v25 // (op 2141) least significant half
v_addc_co_u32 v31, s[50:51], v15, v29, s[50:51] // (op 2141) most significant half




s_add_u32 s1, 24576, s0 // (op 2141) 
s_mov_b32 m0, s1 // (op 2141) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 2141) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 2141) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2141) most significant half

s_add_u32 m0, m0, 4096 // (op 2141) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 2141) Load value direct to lds







s_lshl_b32 s0, s36, 2 // (op 1977) 
v_add_u32 v25, v22, s0 // (op 1977) 


buffer_load_ubyte v14, v25, s[68:71], 0 offen offset: 0 // (op 1977) Load value


v_and_b32 v15, -4, v1 // (op 2638) 
v_sub_u32 v22, v1, v15 // (op 2638) 

v_and_b32 v15, 4294967292, v1 // (op 2638) 
v_add_u32 v25, v15, v22 // (op 2638) 





s_waitcnt vmcnt(9)
ds_write_b8 v25, v24 offset:32768 // (op 1980) Store local data 

v_and_b32 v15, -64, v1 // (op 2580) 
v_sub_u32 v22, v1, v15 // (op 2580) 

v_and_b32 v15, 4294967232, v1 // (op 2580) 
v_add_u32 v25, v15, v22 // (op 2580) 





s_waitcnt vmcnt(6)
ds_write_b8 v25, v28 offset:33024 // (op 1982) Store local data 

s_barrier  // (op 2905) 



ds_read_b128 v[30:33], v4 // (op 2007) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 2007) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2940) 



ds_read_u8 v15, v5 offset:32768 // (op 2027) Load local data 



ds_read_b128 v[38:41], v4 offset:4096 // (op 2012) Load local data 
ds_read_b128 v[42:45], v4 offset:4160 // (op 2012) Load local data 



ds_read_u8 v22, v5 offset:32896 // (op 2032) Load local data 
s_barrier  // (op 2889) 



ds_read_b64_tr_b8 v[46:47], v0 offset:8192 // (op 1997) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:8704 // (op 1997) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:12288 // (op 1997) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:12800 // (op 1997) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2922) 


ds_read_u8 v25, v3 offset:33024 // (op 2017) Load local data 



ds_read_b64_tr_b8 v[54:55], v0 offset:8224 // (op 2002) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:8736 // (op 2002) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:12320 // (op 2002) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:12832 // (op 2002) Transpose load local data 


ds_read_u8 v29, v3 offset:33056 // (op 2022) Load local data 
v_and_b32 v18, -4, v1 // (op 2437) 
v_sub_u32 v19, v1, v18 // (op 2437) 

v_lshlrev_b32 v18, 4, v19 // (op 2437) 

v_lshrrev_b32 v19, 2, v1 // (op 2437) 
v_lshl_add_u32 v62, v19, 6, v18 // (op 2437) 


s_nop 0
v_readfirstlane_b32 s0, v62 // (op 2437) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 2494) 
v_sub_u32 v19, v1, v18 // (op 2494) 

v_lshlrev_b32 v18, 4, v19 // (op 2494) 

v_lshrrev_b32 v19, 3, v1 // (op 2494) 
v_lshl_add_u32 v62, v19, 7, v18 // (op 2494) 


s_nop 0
v_readfirstlane_b32 s1, v62 // (op 2494) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -64, v1 // (op 2551) 
v_sub_u32 v19, v1, v18 // (op 2551) 

v_and_b32 v18, 4294967232, v1 // (op 2551) 
v_add_u32 v62, v18, v19 // (op 2551) 


v_and_b32 v18, -4, v1 // (op 2609) 
v_sub_u32 v19, v1, v18 // (op 2609) 

v_and_b32 v18, 4294967292, v1 // (op 2609) 
v_add_u32 v63, v18, v19 // (op 2609) 


v_and_b32 v18, -4, v1 // (op 2690) 
v_sub_u32 v19, v1, v18 // (op 2690) 

v_lshlrev_b32 v18, 4, v19 // (op 2690) 

v_lshrrev_b32 v19, 2, v1 // (op 2690) 
v_lshl_add_u32 v64, v19, 6, v18 // (op 2690) 


s_nop 0
v_readfirstlane_b32 s18, v64 // (op 2690) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 2734) 
v_sub_u32 v19, v1, v18 // (op 2734) 

v_lshlrev_b32 v18, 4, v19 // (op 2734) 

v_lshrrev_b32 v19, 3, v1 // (op 2734) 
v_lshl_add_u32 v64, v19, 7, v18 // (op 2734) 


s_nop 0
v_readfirstlane_b32 s19, v64 // (op 2734) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -64, v1 // (op 2778) 
v_sub_u32 v19, v1, v18 // (op 2778) 

v_and_b32 v18, 4294967232, v1 // (op 2778) 
v_add_u32 v64, v18, v19 // (op 2778) 


v_and_b32 v18, -4, v1 // (op 2807) 
v_sub_u32 v19, v1, v18 // (op 2807) 

v_and_b32 v18, 4294967292, v1 // (op 2807) 
v_add_u32 v65, v18, v19 // (op 2807) 


s_mov_b32 s22, 0 // (op 593) call()
s_ashr_i64 s[24:25], s[30:31], 7 // (op 592) 
s_mov_b32 s23, s24 // (op 592) convert

s_and_b32 s24, s23, 4294967294 // (op 592) 

s_cmp_lt_u32 s22, s24 // (op 592) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoop_592 // (op 592) Condition: Top (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoop_592 if false)
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoop_592:
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[30:37], a[0:3], v25, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 2908) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[38:45], a[4:7], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 380) 
s_add_u32 s22, s22, 2 // (op 594) 
s_lshl_b64 s[24:25], s[32:33], 5 // (op 2101) 
v_mov_b64 v[18:19], s[24:25] // (op 2101) convert


v_mov_b64 v[66:67], v[20:21] // (op 2101) 
v_mov_b64 v[68:69], v[66:67] // (op 2101) 
s_mov_b32 m0, s1 // (op 2101) reg expression
s_mov_b32 s23, 256 // (op 2101) call()
buffer_load_dwordx4 v68, s[64:67], s23 offen offset: 0 lds // (op 2101) Load value direct to lds
v_add_co_u32 v68, s[24:25], v68, v18 // (op 2101) least significant half
v_addc_co_u32 v69, s[24:25], 0, v69, s[24:25] // (op 2101) most significant half

s_add_u32 m0, m0, 4096 // (op 2101) 
buffer_load_dwordx4 v68, s[64:67], s23 offen offset: 0 lds // (op 2101) Load value direct to lds



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[30:37], a[8:11], v29, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 677) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:61], v[38:45], a[12:15], v29, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 631) 
v_mov_b32 v18, v26 // (op 25) 

buffer_load_ubyte v24, v18, s[72:75], 0 offen offset:8 // (op 25) Load value
s_barrier  // (op 2892) 

s_lshl_b64 s[24:25], s[26:27], 6 // (op 2096) 
v_mov_b64 v[68:69], s[24:25] // (op 2096) convert


v_mov_b64 v[70:71], v[16:17] // (op 2096) 
s_lshl_b64 s[24:25], s[26:27], 7 // (op 2096) 
s_lshl_b64 s[50:51], s[24:25], 1 // (op 2096) 

v_mov_b32 v19, s50 // (op 2096) 
v_mov_b32 v72, s51 // (op 2096) 
v_add_co_u32 v74, s[24:25], v70, v19 // (op 2096) least significant half
v_addc_co_u32 v75, s[24:25], v71, v72, s[24:25] // (op 2096) most significant half




s_add_u32 s23, 8192, s0 // (op 2096) 
s_mov_b32 m0, s23 // (op 2096) reg expression
buffer_load_dwordx4 v74, s[52:55], 0 offen offset: 0 lds // (op 2096) Load value direct to lds
v_add_co_u32 v74, s[24:25], v74, v68 // (op 2096) least significant half
v_addc_co_u32 v75, s[24:25], 0, v75, s[24:25] // (op 2096) most significant half

s_add_u32 m0, m0, 4096 // (op 2096) 
buffer_load_dwordx4 v74, s[52:55], 0 offen offset: 0 lds // (op 2096) Load value direct to lds





v_mov_b32 v19, v23 // (op 17) 
s_lshl_b32 s23, s36, 3 // (op 17) 
v_add_u32 v68, v19, s23 // (op 17) 


buffer_load_ubyte v28, v68, s[68:71], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 2980) 



s_waitcnt vmcnt(9)
ds_write_b8 v65, v27 offset:33280 // (op 259) Store local data 
s_barrier  // (op 2974) 



s_waitcnt vmcnt(6)
ds_write_b8 v64, v14 offset:33536 // (op 258) Store local data 
s_barrier  // (op 2963) 



ds_read_b128 v[72:75], v8 offset:16384 // (op 251) Load local data 
ds_read_b128 v[76:79], v8 offset:16448 // (op 251) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2977) 



ds_read_u8 v68, v9 offset:33280 // (op 255) Load local data 



ds_read_b128 v[80:83], v8 offset:20480 // (op 385) Load local data 
ds_read_b128 v[84:87], v8 offset:20544 // (op 385) Load local data 



ds_read_u8 v69, v9 offset:33408 // (op 389) Load local data 
s_barrier  // (op 2957) 
s_barrier  // (op 3025) 



ds_read_b64_tr_b8 v[46:47], v6 offset:24576 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v6 offset:25088 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v6 offset:28672 // (op 249) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v6 offset:29184 // (op 249) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2970) 


ds_read_u8 v25, v7 offset:33536 // (op 253) Load local data 



ds_read_b64_tr_b8 v[54:55], v6 offset:24608 // (op 634) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v6 offset:25120 // (op 634) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v6 offset:28704 // (op 634) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v6 offset:29216 // (op 634) Transpose load local data 


ds_read_u8 v29, v7 offset:33568 // (op 637) Load local data 
s_barrier  // (op 3047) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[72:79], a[0:3], v25, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 256) 
s_barrier  // (op 2967) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[80:87], a[4:7], v25, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 381) 
s_lshl_b64 s[24:25], s[32:33], 5 // (op 2124) 
v_mov_b64 v[88:89], s[24:25] // (op 2124) convert


v_mov_b64 v[90:91], v[66:67] // (op 2124) 
s_add_u32 s23, 16384, s19 // (op 2124) 
s_mov_b32 m0, s23 // (op 2124) reg expression
s_mov_b32 s24, 384 // (op 2124) call()
buffer_load_dwordx4 v90, s[64:67], s24 offen offset: 0 lds // (op 2124) Load value direct to lds
v_add_co_u32 v90, s[50:51], v90, v88 // (op 2124) least significant half
v_addc_co_u32 v91, s[50:51], 0, v91, s[50:51] // (op 2124) most significant half

s_add_u32 m0, m0, 4096 // (op 2124) 
buffer_load_dwordx4 v90, s[64:67], s24 offen offset: 0 lds // (op 2124) Load value direct to lds




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[72:79], a[8:11], v29, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 678) 






v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:61], v[80:87], a[12:15], v29, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 632) 
buffer_load_ubyte v27, v18, s[72:75], 0 offen offset:12 // (op 262) Load value



v_add_co_u32 v20, vcc, 256, v20 // (op 2258) least significant half
s_barrier  // (op 2960) 
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 2258) most significant half

v_add_u32 v26, 8, v26 // (op 2386) 
s_lshl_b64 s[24:25], s[26:27], 6 // (op 2119) 
v_mov_b64 v[66:67], s[24:25] // (op 2119) convert


s_lshl_b64 s[24:25], s[26:27], 7 // (op 2119) 
s_mul_i32 s51, 3, s25 // (op 2119) most significant: low of low * high
s_mul_i32 s50, 3, s24 // (op 2119) least significant
s_mul_hi_u32 s23, 3, s24 // (op 2119) most significant: high of low * low
s_add_u32 s51, s51, s23 // (op 2119) most significant: sum


v_mov_b32 v18, s50 // (op 2119) 
v_mov_b32 v68, s51 // (op 2119) 
v_add_co_u32 v72, s[24:25], v70, v18 // (op 2119) least significant half
v_addc_co_u32 v73, s[24:25], v71, v68, s[24:25] // (op 2119) most significant half




s_add_u32 s23, 24576, s18 // (op 2119) 
s_mov_b32 m0, s23 // (op 2119) reg expression
buffer_load_dwordx4 v72, s[52:55], 0 offen offset: 0 lds // (op 2119) Load value direct to lds
v_add_co_u32 v72, s[24:25], v72, v66 // (op 2119) least significant half
v_addc_co_u32 v73, s[24:25], 0, v73, s[24:25] // (op 2119) most significant half

s_add_u32 m0, m0, 4096 // (op 2119) 
buffer_load_dwordx4 v72, s[52:55], 0 offen offset: 0 lds // (op 2119) Load value direct to lds






s_lshl_b32 s23, s36, 2 // (op 261) 
s_mul_i32 s24, 3, s23 // (op 261) 

v_add_u32 v18, v19, s24 // (op 261) 


s_lshl_b64 s[24:25], s[26:27], 7 // (op 2194) 
s_lshl_b64 s[50:51], s[24:25], 1 // (op 2194) 

v_mov_b32 v66, s50 // (op 2194) 
v_mov_b32 v67, s51 // (op 2194) 
v_add_co_u32 v16, s[24:25], v16, v66 // (op 2194) least significant half
v_addc_co_u32 v17, s[24:25], v17, v67, s[24:25] // (op 2194) most significant half




buffer_load_ubyte v14, v18, s[68:71], 0 offen offset: 0 // (op 261) Load value


s_barrier  // (op 2943) 
s_lshl_b32 s23, s36, 3 // (op 2322) 



v_add_u32 v23, v23, s23 // (op 2322) 

s_waitcnt vmcnt(9)
ds_write_b8 v63, v24 offset:32768 // (op 30) Store local data 
s_barrier  // (op 2925) 



s_waitcnt vmcnt(6)
ds_write_b8 v62, v28 offset:33024 // (op 21) Store local data 
s_barrier  // (op 2914) 



ds_read_b128 v[30:33], v4 // (op 77) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2950) 



ds_read_u8 v15, v5 offset:32768 // (op 87) Load local data 



ds_read_b128 v[38:41], v4 offset:4096 // (op 375) Load local data 
ds_read_b128 v[42:45], v4 offset:4160 // (op 375) Load local data 



ds_read_u8 v22, v5 offset:32896 // (op 379) Load local data 
s_barrier  // (op 2898) 



ds_read_b64_tr_b8 v[46:47], v0 offset:8192 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:8704 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:12288 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:12800 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 2932) 


ds_read_u8 v25, v3 offset:33024 // (op 82) Load local data 



ds_read_b64_tr_b8 v[54:55], v0 offset:8224 // (op 626) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:8736 // (op 626) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:12320 // (op 626) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:12832 // (op 626) Transpose load local data 


ds_read_u8 v29, v3 offset:33056 // (op 629) Load local data 
s_ashr_i64 s[24:25], s[30:31], 7 // (op 592) 
s_mov_b32 s23, s24 // (op 592) convert

s_and_b32 s24, s23, 4294967294 // (op 592) 

s_cmp_lt_u32 s22, s24 // (op 592) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoop_592 // (op 592) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoop_592 if true)
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoop_592:















s_barrier  // (op 3031) 
s_mul_hi_i32 s0, s16, s10 // (op 2154) 
v_and_b32 v26, -4, v1 // (op 2154) 
s_add_i32 s1, s0, s16 // (op 2154) 

v_sub_u32 v28, v1, v26 // (op 2154) 

s_ashr_i32 s0, s1, 31 // (op 2154) 
v_lshlrev_b32 v26, 4, v28 // (op 2154) 

s_cmp_eq_i32 s10, 0 // (op 2154) 
s_mov_b32 s18, scc // (op 2154) 
v_lshrrev_b32 v28, 2, v1 // (op 2154) 
s_cmp_eq_i32 s18, 0 // (op 2154) 
s_cselect_b32 s19, 0, -1 // (op 2154) 

s_and_b32 s18, s11, 31 // (op 2154) 
s_lshl_b32 s22, 1, s18 // (op 2154) 
s_add_i32 s23, s22, s19 // (op 2154) 


s_and_b32 s19, s0, s23 // (op 2154) 


s_add_i32 s0, s1, s19 // (op 2154) 


s_ashr_i32 s1, s0, s18 // (op 2154) 


s_lshl_b32 s0, s11, 24 // (op 2154) 
s_ashr_i32 s18, s0, 31 // (op 2154) 

s_xor_b32 s0, s1, s18 // (op 2154) 

s_sub_i32 s1, s0, s18 // (op 2154) 


s_mul_hi_i32 s0, s1, s12 // (op 2154) 
s_add_i32 s18, s0, s1 // (op 2154) 

s_ashr_i32 s0, s18, 31 // (op 2154) 
s_cmp_eq_i32 s12, 0 // (op 2154) 
s_mov_b32 s19, scc // (op 2154) 
s_cmp_eq_i32 s19, 0 // (op 2154) 
s_cselect_b32 s22, 0, -1 // (op 2154) 

s_and_b32 s19, s13, 31 // (op 2154) 
s_lshl_b32 s23, 1, s19 // (op 2154) 
s_add_i32 s24, s23, s22 // (op 2154) 


s_and_b32 s25, s0, s24 // (op 2154) 


s_add_i32 s0, s18, s25 // (op 2154) 


s_ashr_i32 s18, s0, s19 // (op 2154) 


s_lshl_b32 s0, s13, 24 // (op 2154) 
s_ashr_i32 s19, s0, 31 // (op 2154) 

s_xor_b32 s0, s18, s19 // (op 2154) 

s_sub_i32 s18, s0, s19 // (op 2154) 


s_mul_i32 s0, s1, s15 // (op 2154) 

s_sub_u32 s1, s16, s0 // (op 2154) 

s_mul_i32 s0, s18, s15 // (op 2154) 
s_add_u32 s0, s0, s1 // (op 2154) 


s_mul_hi_i32 s1, s16, s9 // (op 2154) 
s_add_i32 s18, s1, s16 // (op 2154) 

s_ashr_i32 s1, s18, 31 // (op 2154) 
s_cmp_eq_i32 s9, 0 // (op 2154) 
s_mov_b32 s19, scc // (op 2154) 
s_cmp_eq_i32 s19, 0 // (op 2154) 
s_cselect_b32 s22, 0, -1 // (op 2154) 

s_and_b32 s19, s46, 31 // (op 2154) 
s_lshl_b32 s23, 1, s19 // (op 2154) 
s_add_i32 s24, s23, s22 // (op 2154) 


s_and_b32 s25, s1, s24 // (op 2154) 


s_add_i32 s1, s18, s25 // (op 2154) 


s_ashr_i32 s18, s1, s19 // (op 2154) 


s_lshl_b32 s1, s46, 24 // (op 2154) 
s_ashr_i32 s19, s1, 31 // (op 2154) 

s_xor_b32 s1, s18, s19 // (op 2154) 

s_sub_i32 s18, s1, s19 // (op 2154) 


s_add_u32 s50, s6, 64 // (op 2154) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2154) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2154) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2154) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2154) 

s_mul_i32 s1, s15, s24 // (op 2154) 

s_mul_i32 s19, s8, s1 // (op 2154) 

s_mul_i32 s1, s18, s19 // (op 2154) 

s_sub_u32 s19, s16, s1 // (op 2154) 

s_mov_b32 s1, s19 // (op 2154) 
s_mul_hi_i32 s22, s1, s14 // (op 2154) 
s_add_i32 s23, s22, s1 // (op 2154) 


s_ashr_i32 s1, s23, 31 // (op 2154) 
s_cmp_eq_i32 s14, 0 // (op 2154) 
s_mov_b32 s22, scc // (op 2154) 
s_cmp_eq_i32 s22, 0 // (op 2154) 
s_cselect_b32 s24, 0, -1 // (op 2154) 

s_and_b32 s22, s48, 31 // (op 2154) 
s_lshl_b32 s25, 1, s22 // (op 2154) 
s_add_i32 s49, s25, s24 // (op 2154) 


s_and_b32 s24, s1, s49 // (op 2154) 


s_add_i32 s1, s23, s24 // (op 2154) 


s_ashr_i32 s23, s1, s22 // (op 2154) 


s_lshl_b32 s1, s48, 24 // (op 2154) 
s_ashr_i32 s22, s1, 31 // (op 2154) 

s_xor_b32 s1, s23, s22 // (op 2154) 

s_sub_i32 s23, s1, s22 // (op 2154) 


s_mul_i32 s1, s47, s15 // (op 2154) 
s_add_u32 s24, s20, 64 // (op 2154) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2154) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2154) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2154) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2154) 

s_sub_i32 s22, s24, s1 // (op 2154) 

s_mul_i32 s24, s23, s22 // (op 2154) 


s_sub_u32 s25, s19, s24 // (op 2154) 


s_add_u32 s19, s25, s1 // (op 2154) 


s_cmp_eq_i32 s18, 0 // (op 2154) 
s_mov_b32 s1, scc // (op 2154) 

s_cmp_eq_i32 s1, 0 // (op 2154) 
s_cselect_b32 s18, s19, s0 // (op 2154) 



v_lshl_add_u32 v16, s18, 6, v26 // (op 2154) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2154) 
v_mov_b32 v21, 0 // (op 2154) convert
v_mov_b32 v20, v16 // (op 2154) convert

s_lshr_b32 s18, s0, 1 // (op 2154) 

v_lshl_add_u32 v26, s18, 8, v28 // (op 2154) 


v_mul_lo_u32 v17, v26, s27 // (op 2154) most significant: low of low * high
v_mul_hi_u32 v28, v26, s26 // (op 2154) most significant: high of low * low
v_mul_lo_u32 v16, v26, s26 // (op 2154) least significant: low of low * low
v_add_u32 v17, v17, v28 // (op 2154) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v16 // (op 2154) least significant half
v_addc_co_u32 v19, s[0:1], v21, v17, s[0:1] // (op 2154) most significant half



s_mul_i32 s1, s26, s31 // (op 2156) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2156) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2156) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2156) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2156) most significant: sum
s_add_u32 s1, s1, s19 // (op 2156) most significant: sum


s_mov_b32 s52, s28 // (op 2156) reg expression
s_mov_b32 s53, s29 // (op 2156) reg expression
s_mov_b32 s54, s0 // (op 2156) reg expression
s_mov_b32 s55, 131072 // (op 2156) call()

s_mul_hi_i32 s0, s16, s10 // (op 2224) 
v_lshrrev_b32 v26, 3, v1 // (op 2224) 
s_add_i32 s1, s0, s16 // (op 2224) 

v_and_b32 v28, -8, v1 // (op 2224) 
s_ashr_i32 s0, s1, 31 // (op 2224) 
v_sub_u32 v16, v1, v28 // (op 2224) 

s_cmp_eq_i32 s10, 0 // (op 2224) 
s_mov_b32 s18, scc // (op 2224) 
v_lshlrev_b32 v17, 4, v16 // (op 2224) 

s_cmp_eq_i32 s18, 0 // (op 2224) 
s_cselect_b32 s19, 0, -1 // (op 2224) 

s_and_b32 s18, s11, 31 // (op 2224) 
s_lshl_b32 s22, 1, s18 // (op 2224) 
s_add_i32 s23, s22, s19 // (op 2224) 


s_and_b32 s19, s0, s23 // (op 2224) 


s_add_i32 s0, s1, s19 // (op 2224) 


s_ashr_i32 s1, s0, s18 // (op 2224) 


s_lshl_b32 s0, s11, 24 // (op 2224) 
s_ashr_i32 s18, s0, 31 // (op 2224) 

s_xor_b32 s0, s1, s18 // (op 2224) 

s_sub_i32 s1, s0, s18 // (op 2224) 


s_mul_hi_i32 s0, s1, s12 // (op 2224) 
s_add_i32 s18, s0, s1 // (op 2224) 

s_ashr_i32 s0, s18, 31 // (op 2224) 
s_cmp_eq_i32 s12, 0 // (op 2224) 
s_mov_b32 s19, scc // (op 2224) 
s_cmp_eq_i32 s19, 0 // (op 2224) 
s_cselect_b32 s22, 0, -1 // (op 2224) 

s_and_b32 s19, s13, 31 // (op 2224) 
s_lshl_b32 s23, 1, s19 // (op 2224) 
s_add_i32 s24, s23, s22 // (op 2224) 


s_and_b32 s25, s0, s24 // (op 2224) 


s_add_i32 s0, s18, s25 // (op 2224) 


s_ashr_i32 s18, s0, s19 // (op 2224) 


s_lshl_b32 s0, s13, 24 // (op 2224) 
s_ashr_i32 s19, s0, 31 // (op 2224) 

s_xor_b32 s0, s18, s19 // (op 2224) 

s_sub_i32 s18, s0, s19 // (op 2224) 


s_add_u32 s50, s6, 64 // (op 2224) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2224) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2224) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2224) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2224) 

s_mul_i32 s0, s18, s24 // (op 2224) 

s_sub_i32 s18, s1, s0 // (op 2224) 


s_mul_hi_i32 s19, s16, s9 // (op 2224) 
s_add_i32 s0, s19, s16 // (op 2224) 

s_ashr_i32 s1, s0, 31 // (op 2224) 
s_cmp_eq_i32 s9, 0 // (op 2224) 
s_mov_b32 s19, scc // (op 2224) 
s_cmp_eq_i32 s19, 0 // (op 2224) 
s_cselect_b32 s22, 0, -1 // (op 2224) 

s_and_b32 s19, s46, 31 // (op 2224) 
s_lshl_b32 s23, 1, s19 // (op 2224) 
s_add_i32 s49, s23, s22 // (op 2224) 


s_and_b32 s22, s1, s49 // (op 2224) 


s_add_i32 s1, s0, s22 // (op 2224) 


s_ashr_i32 s0, s1, s19 // (op 2224) 


s_lshl_b32 s1, s46, 24 // (op 2224) 
s_ashr_i32 s19, s1, 31 // (op 2224) 

s_xor_b32 s1, s0, s19 // (op 2224) 

s_sub_i32 s0, s1, s19 // (op 2224) 


s_mul_i32 s1, s15, s24 // (op 2224) 

s_mul_i32 s19, s8, s1 // (op 2224) 

s_mul_i32 s1, s0, s19 // (op 2224) 

s_sub_u32 s19, s16, s1 // (op 2224) 

s_mul_hi_i32 s1, s19, s14 // (op 2224) 
s_add_i32 s22, s1, s19 // (op 2224) 


s_ashr_i32 s1, s22, 31 // (op 2224) 
s_cmp_eq_i32 s14, 0 // (op 2224) 
s_mov_b32 s19, scc // (op 2224) 
s_cmp_eq_i32 s19, 0 // (op 2224) 
s_cselect_b32 s23, 0, -1 // (op 2224) 

s_and_b32 s19, s48, 31 // (op 2224) 
s_lshl_b32 s24, 1, s19 // (op 2224) 
s_add_i32 s25, s24, s23 // (op 2224) 


s_and_b32 s23, s1, s25 // (op 2224) 


s_add_i32 s1, s22, s23 // (op 2224) 


s_ashr_i32 s22, s1, s19 // (op 2224) 


s_lshl_b32 s1, s48, 24 // (op 2224) 
s_ashr_i32 s19, s1, 31 // (op 2224) 

s_xor_b32 s1, s22, s19 // (op 2224) 

s_sub_i32 s22, s1, s19 // (op 2224) 


s_cmp_eq_i32 s0, 0 // (op 2224) 
s_mov_b32 s1, scc // (op 2224) 

s_cmp_eq_i32 s1, 0 // (op 2224) 
s_cselect_b32 s0, s22, s18 // (op 2224) 



v_lshl_add_u32 v16, s0, 6, v26 // (op 2224) 


v_mul_lo_u32 v21, v16, s33 // (op 2224) most significant: low of low * high
v_mul_hi_u32 v26, v16, s32 // (op 2224) most significant: high of low * low
v_mul_lo_u32 v20, v16, s32 // (op 2224) least significant: low of low * low
v_add_u32 v21, v21, v26 // (op 2224) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2224) 
s_lshr_b32 s18, s0, 1 // (op 2224) 

v_lshl_add_u32 v16, s18, 8, v17 // (op 2224) 


v_mov_b32 v63, 0 // (op 2224) convert
v_mov_b32 v62, v16 // (op 2224) convert

v_add_co_u32 v16, s[0:1], v62, v20 // (op 2224) least significant half
v_addc_co_u32 v17, s[0:1], v63, v21, s[0:1] // (op 2224) most significant half



s_mul_i32 s1, s32, s7 // (op 2226) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2226) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2226) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2226) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2226) most significant: sum
s_add_u32 s1, s1, s19 // (op 2226) most significant: sum


s_mov_b32 s64, s34 // (op 2226) reg expression
s_mov_b32 s65, s35 // (op 2226) reg expression
s_mov_b32 s66, s0 // (op 2226) reg expression
s_mov_b32 s67, 131072 // (op 2226) call()

s_mul_hi_i32 s0, s16, s10 // (op 2288) 
v_and_b32 v26, -64, v1 // (op 2288) 
s_add_i32 s1, s0, s16 // (op 2288) 

v_sub_u32 v28, v1, v26 // (op 2288) 

s_ashr_i32 s0, s1, 31 // (op 2288) 
v_lshrrev_b32 v26, 6, v1 // (op 2288) 
s_cmp_eq_i32 s10, 0 // (op 2288) 
s_mov_b32 s18, scc // (op 2288) 
s_cmp_eq_i32 s18, 0 // (op 2288) 
s_cselect_b32 s19, 0, -1 // (op 2288) 

s_and_b32 s18, s11, 31 // (op 2288) 
s_lshl_b32 s22, 1, s18 // (op 2288) 
s_add_i32 s23, s22, s19 // (op 2288) 


s_and_b32 s19, s0, s23 // (op 2288) 


s_add_i32 s0, s1, s19 // (op 2288) 


s_ashr_i32 s1, s0, s18 // (op 2288) 


s_lshl_b32 s0, s11, 24 // (op 2288) 
s_ashr_i32 s18, s0, 31 // (op 2288) 

s_xor_b32 s0, s1, s18 // (op 2288) 

s_sub_i32 s1, s0, s18 // (op 2288) 


s_mul_hi_i32 s0, s1, s12 // (op 2288) 
s_add_i32 s18, s0, s1 // (op 2288) 

s_ashr_i32 s0, s18, 31 // (op 2288) 
s_cmp_eq_i32 s12, 0 // (op 2288) 
s_mov_b32 s19, scc // (op 2288) 
s_cmp_eq_i32 s19, 0 // (op 2288) 
s_cselect_b32 s22, 0, -1 // (op 2288) 

s_and_b32 s19, s13, 31 // (op 2288) 
s_lshl_b32 s23, 1, s19 // (op 2288) 
s_add_i32 s24, s23, s22 // (op 2288) 


s_and_b32 s25, s0, s24 // (op 2288) 


s_add_i32 s0, s18, s25 // (op 2288) 


s_ashr_i32 s18, s0, s19 // (op 2288) 


s_lshl_b32 s0, s13, 24 // (op 2288) 
s_ashr_i32 s19, s0, 31 // (op 2288) 

s_xor_b32 s0, s18, s19 // (op 2288) 

s_sub_i32 s18, s0, s19 // (op 2288) 


s_mul_i32 s0, s1, s15 // (op 2288) 

s_sub_u32 s1, s16, s0 // (op 2288) 

s_mul_i32 s0, s18, s15 // (op 2288) 
s_add_u32 s0, s0, s1 // (op 2288) 


s_mul_hi_i32 s1, s16, s9 // (op 2288) 
s_add_i32 s18, s1, s16 // (op 2288) 

s_ashr_i32 s1, s18, 31 // (op 2288) 
s_cmp_eq_i32 s9, 0 // (op 2288) 
s_mov_b32 s19, scc // (op 2288) 
s_cmp_eq_i32 s19, 0 // (op 2288) 
s_cselect_b32 s22, 0, -1 // (op 2288) 

s_and_b32 s19, s46, 31 // (op 2288) 
s_lshl_b32 s23, 1, s19 // (op 2288) 
s_add_i32 s24, s23, s22 // (op 2288) 


s_and_b32 s25, s1, s24 // (op 2288) 


s_add_i32 s1, s18, s25 // (op 2288) 


s_ashr_i32 s18, s1, s19 // (op 2288) 


s_lshl_b32 s1, s46, 24 // (op 2288) 
s_ashr_i32 s19, s1, 31 // (op 2288) 

s_xor_b32 s1, s18, s19 // (op 2288) 

s_sub_i32 s18, s1, s19 // (op 2288) 


s_add_u32 s50, s6, 64 // (op 2288) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2288) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2288) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2288) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2288) 

s_mul_i32 s1, s15, s24 // (op 2288) 

s_mul_i32 s19, s8, s1 // (op 2288) 

s_mul_i32 s1, s18, s19 // (op 2288) 

s_sub_u32 s19, s16, s1 // (op 2288) 

s_mov_b32 s1, s19 // (op 2288) 
s_mul_hi_i32 s22, s1, s14 // (op 2288) 
s_add_i32 s23, s22, s1 // (op 2288) 


s_ashr_i32 s1, s23, 31 // (op 2288) 
s_cmp_eq_i32 s14, 0 // (op 2288) 
s_mov_b32 s22, scc // (op 2288) 
s_cmp_eq_i32 s22, 0 // (op 2288) 
s_cselect_b32 s24, 0, -1 // (op 2288) 

s_and_b32 s22, s48, 31 // (op 2288) 
s_lshl_b32 s25, 1, s22 // (op 2288) 
s_add_i32 s49, s25, s24 // (op 2288) 


s_and_b32 s24, s1, s49 // (op 2288) 


s_add_i32 s1, s23, s24 // (op 2288) 


s_ashr_i32 s23, s1, s22 // (op 2288) 


s_lshl_b32 s1, s48, 24 // (op 2288) 
s_ashr_i32 s22, s1, 31 // (op 2288) 

s_xor_b32 s1, s23, s22 // (op 2288) 

s_sub_i32 s23, s1, s22 // (op 2288) 


s_mul_i32 s1, s47, s15 // (op 2288) 
s_add_u32 s24, s20, 64 // (op 2288) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2288) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2288) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2288) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2288) 

s_sub_i32 s22, s24, s1 // (op 2288) 

s_mul_i32 s24, s23, s22 // (op 2288) 


s_sub_u32 s25, s19, s24 // (op 2288) 


s_add_u32 s19, s25, s1 // (op 2288) 


s_cmp_eq_i32 s18, 0 // (op 2288) 
s_mov_b32 s1, scc // (op 2288) 

s_cmp_eq_i32 s1, 0 // (op 2288) 
s_cselect_b32 s18, s19, s0 // (op 2288) 



v_lshl_add_u32 v20, s18, 6, v28 // (op 2288) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 2288) 
s_lshr_b32 s18, s0, 1 // (op 2288) 

v_lshl_add_u32 v21, s18, 3, v26 // (op 2288) 


v_mul_lo_u32 v26, v21, s36 // (op 2288) 

v_add_u32 v21, v20, v26 // (op 2288) 


s_mul_i32 s1, s36, s41 // (op 2290) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 2290) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 2290) least significant
s_mul_hi_u32 s19, s36, s40 // (op 2290) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2290) most significant: sum
s_add_u32 s1, s1, s19 // (op 2290) most significant: sum


s_mov_b32 s68, s38 // (op 2290) reg expression
s_mov_b32 s69, s39 // (op 2290) reg expression
s_mov_b32 s70, s0 // (op 2290) reg expression
s_mov_b32 s71, 131072 // (op 2290) call()

s_mul_hi_i32 s0, s16, s10 // (op 2352) 
v_lshrrev_b32 v20, 2, v1 // (op 2352) 
s_add_i32 s1, s0, s16 // (op 2352) 

v_and_b32 v26, -4, v1 // (op 2352) 
s_ashr_i32 s0, s1, 31 // (op 2352) 
v_sub_u32 v28, v1, v26 // (op 2352) 

s_cmp_eq_i32 s10, 0 // (op 2352) 
s_mov_b32 s18, scc // (op 2352) 
s_cmp_eq_i32 s18, 0 // (op 2352) 
s_cselect_b32 s19, 0, -1 // (op 2352) 

s_and_b32 s18, s11, 31 // (op 2352) 
s_lshl_b32 s22, 1, s18 // (op 2352) 
s_add_i32 s23, s22, s19 // (op 2352) 


s_and_b32 s19, s0, s23 // (op 2352) 


s_add_i32 s0, s1, s19 // (op 2352) 


s_ashr_i32 s1, s0, s18 // (op 2352) 


s_lshl_b32 s0, s11, 24 // (op 2352) 
s_ashr_i32 s18, s0, 31 // (op 2352) 

s_xor_b32 s0, s1, s18 // (op 2352) 

s_sub_i32 s1, s0, s18 // (op 2352) 


s_mul_hi_i32 s0, s1, s12 // (op 2352) 
s_add_i32 s18, s0, s1 // (op 2352) 

s_ashr_i32 s0, s18, 31 // (op 2352) 
s_cmp_eq_i32 s12, 0 // (op 2352) 
s_mov_b32 s19, scc // (op 2352) 
s_cmp_eq_i32 s19, 0 // (op 2352) 
s_cselect_b32 s22, 0, -1 // (op 2352) 

s_and_b32 s19, s13, 31 // (op 2352) 
s_lshl_b32 s23, 1, s19 // (op 2352) 
s_add_i32 s24, s23, s22 // (op 2352) 


s_and_b32 s25, s0, s24 // (op 2352) 


s_add_i32 s0, s18, s25 // (op 2352) 


s_ashr_i32 s18, s0, s19 // (op 2352) 


s_lshl_b32 s0, s13, 24 // (op 2352) 
s_ashr_i32 s19, s0, 31 // (op 2352) 

s_xor_b32 s0, s18, s19 // (op 2352) 

s_sub_i32 s18, s0, s19 // (op 2352) 


s_add_u32 s50, s6, 64 // (op 2352) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2352) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2352) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2352) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 6 // (op 2352) 

s_mul_i32 s0, s18, s24 // (op 2352) 

s_sub_i32 s18, s1, s0 // (op 2352) 


s_mul_hi_i32 s19, s16, s9 // (op 2352) 
s_add_i32 s0, s19, s16 // (op 2352) 

s_ashr_i32 s1, s0, 31 // (op 2352) 
s_cmp_eq_i32 s9, 0 // (op 2352) 
s_mov_b32 s19, scc // (op 2352) 
s_cmp_eq_i32 s19, 0 // (op 2352) 
s_cselect_b32 s22, 0, -1 // (op 2352) 

s_and_b32 s19, s46, 31 // (op 2352) 
s_lshl_b32 s23, 1, s19 // (op 2352) 
s_add_i32 s49, s23, s22 // (op 2352) 


s_and_b32 s22, s1, s49 // (op 2352) 


s_add_i32 s1, s0, s22 // (op 2352) 


s_ashr_i32 s0, s1, s19 // (op 2352) 


s_lshl_b32 s1, s46, 24 // (op 2352) 
s_ashr_i32 s19, s1, 31 // (op 2352) 

s_xor_b32 s1, s0, s19 // (op 2352) 

s_sub_i32 s0, s1, s19 // (op 2352) 


s_mul_i32 s1, s15, s24 // (op 2352) 

s_mul_i32 s19, s8, s1 // (op 2352) 

s_mul_i32 s1, s0, s19 // (op 2352) 

s_sub_u32 s19, s16, s1 // (op 2352) 

s_mul_hi_i32 s1, s19, s14 // (op 2352) 
s_add_i32 s22, s1, s19 // (op 2352) 


s_ashr_i32 s1, s22, 31 // (op 2352) 
s_cmp_eq_i32 s14, 0 // (op 2352) 
s_mov_b32 s19, scc // (op 2352) 
s_cmp_eq_i32 s19, 0 // (op 2352) 
s_cselect_b32 s23, 0, -1 // (op 2352) 

s_and_b32 s19, s48, 31 // (op 2352) 
s_lshl_b32 s24, 1, s19 // (op 2352) 
s_add_i32 s25, s24, s23 // (op 2352) 


s_and_b32 s23, s1, s25 // (op 2352) 


s_add_i32 s1, s22, s23 // (op 2352) 


s_ashr_i32 s22, s1, s19 // (op 2352) 


s_lshl_b32 s1, s48, 24 // (op 2352) 
s_ashr_i32 s19, s1, 31 // (op 2352) 

s_xor_b32 s1, s22, s19 // (op 2352) 

s_sub_i32 s22, s1, s19 // (op 2352) 


s_cmp_eq_i32 s0, 0 // (op 2352) 
s_mov_b32 s1, scc // (op 2352) 

s_cmp_eq_i32 s1, 0 // (op 2352) 
s_cselect_b32 s0, s22, s18 // (op 2352) 



v_lshl_add_u32 v26, s0, 6, v20 // (op 2352) 


v_mul_lo_u32 v20, v26, s42 // (op 2352) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 2352) 
s_lshr_b32 s18, s0, 1 // (op 2352) 

v_lshl_add_u32 v26, s18, 3, v28 // (op 2352) 


v_add_u32 v28, v26, v20 // (op 2352) 


s_mul_i32 s1, s42, s7 // (op 2354) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 2354) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 2354) least significant
s_mul_hi_u32 s19, s42, s6 // (op 2354) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2354) most significant: sum
s_add_u32 s1, s1, s19 // (op 2354) most significant: sum


s_mov_b32 s72, s44 // (op 2354) reg expression
s_mov_b32 s73, s45 // (op 2354) reg expression
s_mov_b32 s74, s0 // (op 2354) reg expression
s_mov_b32 s75, 131072 // (op 2354) call()

v_and_b32 v20, -4, v1 // (op 2450) 
v_sub_u32 v26, v1, v20 // (op 2450) 

v_lshlrev_b32 v20, 4, v26 // (op 2450) 

v_lshrrev_b32 v26, 2, v1 // (op 2450) 
v_lshl_add_u32 v23, v26, 6, v20 // (op 2450) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 2450) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 2507) 
v_sub_u32 v26, v1, v20 // (op 2507) 

v_lshlrev_b32 v20, 4, v26 // (op 2507) 

v_lshrrev_b32 v26, 3, v1 // (op 2507) 
v_lshl_add_u32 v23, v26, 7, v20 // (op 2507) 


s_nop 0
v_readfirstlane_b32 s1, v23 // (op 2507) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -64, v1 // (op 2564) 
v_sub_u32 v26, v1, v20 // (op 2564) 

v_and_b32 v20, 4294967232, v1 // (op 2564) 
v_add_u32 v23, v20, v26 // (op 2564) 


v_and_b32 v20, -4, v1 // (op 2622) 
v_sub_u32 v24, v1, v20 // (op 2622) 

v_and_b32 v20, 4294967292, v1 // (op 2622) 
v_add_u32 v26, v20, v24 // (op 2622) 


s_ashr_i64 s[18:19], s[30:31], 7 // (op 598) 
s_and_b32 s22, s18, 4294967294 // (op 598) 

s_ashr_i64 s[18:19], s[30:31], 7 // (op 597) 
s_mov_b32 s23, s18 // (op 597) convert

s_cmp_lt_u32 s22, s23 // (op 597) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_597 // (op 597) Condition: Top (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_597 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoopTail_597:
s_barrier  // (op 2895) 
s_add_u32 s22, s22, 1 // (op 599) 

s_lshl_b64 s[18:19], s[26:27], 6 // (op 2106) 
v_mov_b64 v[14:15], s[18:19] // (op 2106) convert


v_mov_b64 v[24:25], v[18:19] // (op 2106) 
v_mov_b64 v[30:31], v[24:25] // (op 2106) reg expression
s_add_u32 s18, 8192, s0 // (op 2106) 
s_mov_b32 m0, s18 // (op 2106) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 2106) Load value direct to lds
v_add_co_u32 v30, s[24:25], v30, v14 // (op 2106) least significant half
v_addc_co_u32 v31, s[24:25], 0, v31, s[24:25] // (op 2106) most significant half

s_add_u32 m0, m0, 4096 // (op 2106) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 2106) Load value direct to lds




s_lshl_b64 s[18:19], s[26:27], 7 // (op 2144) 
v_mov_b32 v20, s18 // (op 2144) 
v_mov_b32 v22, s19 // (op 2144) 
v_add_co_u32 v18, s[24:25], v18, v20 // (op 2144) least significant half
v_addc_co_u32 v19, s[24:25], v19, v22, s[24:25] // (op 2144) most significant half




s_barrier  // (op 2911) 

s_lshl_b64 s[18:19], s[32:33], 5 // (op 2112) 
v_mov_b64 v[14:15], s[18:19] // (op 2112) convert


v_mov_b64 v[24:25], v[16:17] // (op 2112) 
v_mov_b64 v[30:31], v[24:25] // (op 2112) 
s_mov_b32 m0, s1 // (op 2112) reg expression
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 2112) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v14 // (op 2112) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2112) most significant half

s_add_u32 m0, m0, 4096 // (op 2112) 
buffer_load_dwordx4 v30, s[64:67], 0 offen offset: 0 lds // (op 2112) Load value direct to lds



v_add_co_u32 v16, vcc, 128, v16 // (op 2216) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 2216) most significant half


v_mov_b32 v20, v21 // (op 195) 
v_mov_b32 v22, v20 // (op 195) reg expression

buffer_load_ubyte v27, v22, s[68:71], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s18, s36, 2 // (op 2280) 
v_add_u32 v21, v21, s18 // (op 2280) 



s_barrier  // (op 2928) 
v_mov_b32 v20, v28 // (op 196) 




buffer_load_ubyte v22, v20, s[72:75], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v28, 4, v28 // (op 2344) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v27 offset:33024 // (op 192) Store local data 

s_barrier  // (op 2901) 



s_barrier  // (op 2946) 
ds_read_b64_tr_b8 v[30:31], v0 offset:8192 // (op 183) Transpose load local data 



ds_read_b64_tr_b8 v[32:33], v0 offset:8704 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:12288 // (op 183) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:12800 // (op 183) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v26, v22 offset:32768 // (op 193) Store local data 

s_barrier  // (op 2918) 



ds_read_b128 v[38:41], v4 // (op 185) Load local data 
ds_read_b128 v[42:45], v4 offset:64 // (op 185) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 2936) 


ds_read_u8 v20, v3 offset:33024 // (op 187) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2953) 



ds_read_u8 v22, v5 offset:32768 // (op 189) Load local data 



ds_read_b128 v[46:49], v4 offset:4096 // (op 354) Load local data 
ds_read_b128 v[50:53], v4 offset:4160 // (op 354) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 190) 



ds_read_u8 v27, v5 offset:32896 // (op 358) Load local data 



ds_read_b64_tr_b8 v[54:55], v0 offset:8224 // (op 572) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:8736 // (op 572) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:37], v[46:53], a[4:7], v20, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 359) 
ds_read_b64_tr_b8 v[58:59], v0 offset:12320 // (op 572) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v0 offset:12832 // (op 572) Transpose load local data 


ds_read_u8 v20, v3 offset:33056 // (op 575) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[38:45], a[8:11], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 577) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:61], v[46:53], a[12:15], v20, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 608) 




s_ashr_i64 s[18:19], s[30:31], 7 // (op 597) 
s_mov_b32 s23, s18 // (op 597) convert

s_cmp_lt_u32 s22, s23 // (op 597) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoopTail_597 // (op 597) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoopTail_597 if true)
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_597:























s_barrier  // (op 3035) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 
v_cvt_f16_f32 v8, v16 // (op 56) 
v_cvt_f16_f32 v9, v17 // (op 56) 


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

v_cvt_f16_f32 v6, v14 // (op 1023) 
v_cvt_f16_f32 v7, v15 // (op 1023) 
v_cvt_f16_f32 v8, v16 // (op 1023) 
v_cvt_f16_f32 v9, v17 // (op 1023) 


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

v_cvt_f16_f32 v6, v14 // (op 1049) 
v_cvt_f16_f32 v7, v15 // (op 1049) 
v_cvt_f16_f32 v8, v16 // (op 1049) 
v_cvt_f16_f32 v9, v17 // (op 1049) 


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

v_cvt_f16_f32 v6, v14 // (op 1059) 
v_cvt_f16_f32 v7, v15 // (op 1059) 
v_cvt_f16_f32 v8, v16 // (op 1059) 
v_cvt_f16_f32 v9, v17 // (op 1059) 


v_pack_b32_f16 v4, v6, v7 // (op 1053) 
v_pack_b32_f16 v5, v8, v9 // (op 1053) 

s_lshl_b32 s0, s4, 6 // (op 1053) 
s_add_u32 s0, s0, 64 // (op 1053) 
v_add_u32 v0, v13, s0 // (op 1053) 


buffer_store_dwordx2 v[4:5], v0, s[60:63], 0 offen // (op 1053) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalBottom_0_beta_135
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalFalse_0_beta_135:


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
v_cvt_f32_f16 v6, v8 // (op 1142) 
v_lshrrev_b32 v7, 16, v8 // (op 1142) 
v_cvt_f32_f16 v7, v7 // (op 1142) 
v_mul_f32 v18, v10, v6 // (op 1142) 
v_mul_f32 v19, v10, v7 // (op 1142) 
v_cvt_f32_f16 v6, v9 // (op 1142) 
v_lshrrev_b32 v7, 16, v9 // (op 1142) 
v_cvt_f32_f16 v7, v7 // (op 1142) 
v_mul_f32 v20, v10, v6 // (op 1142) 
v_mul_f32 v21, v10, v7 // (op 1142) 


v_add_f32 v6, v18, v14 // (op 1135) 
v_add_f32 v7, v19, v15 // (op 1135) 
v_add_f32 v8, v20, v16 // (op 1135) 
v_add_f32 v9, v21, v17 // (op 1135) 


v_cvt_f16_f32 v14, v6 // (op 1136) 
v_cvt_f16_f32 v15, v7 // (op 1136) 
v_cvt_f16_f32 v16, v8 // (op 1136) 
v_cvt_f16_f32 v17, v9 // (op 1136) 


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
v_cvt_f32_f16 v6, v8 // (op 1145) 
v_lshrrev_b32 v7, 16, v8 // (op 1145) 
v_cvt_f32_f16 v7, v7 // (op 1145) 
v_mul_f32 v18, v10, v6 // (op 1145) 
v_mul_f32 v19, v10, v7 // (op 1145) 
v_cvt_f32_f16 v6, v9 // (op 1145) 
v_lshrrev_b32 v7, 16, v9 // (op 1145) 
v_cvt_f32_f16 v7, v7 // (op 1145) 
v_mul_f32 v20, v10, v6 // (op 1145) 
v_mul_f32 v21, v10, v7 // (op 1145) 


v_add_f32 v6, v18, v14 // (op 1146) 
v_add_f32 v7, v19, v15 // (op 1146) 
v_add_f32 v8, v20, v16 // (op 1146) 
v_add_f32 v9, v21, v17 // (op 1146) 


v_cvt_f16_f32 v14, v6 // (op 1147) 
v_cvt_f16_f32 v15, v7 // (op 1147) 
v_cvt_f16_f32 v16, v8 // (op 1147) 
v_cvt_f16_f32 v17, v9 // (op 1147) 


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
v_cvt_f32_f16 v6, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_cvt_f32_f16 v6, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_cvt_f16_f32 v14, v6 // (op 146) 
v_cvt_f16_f32 v15, v7 // (op 146) 
v_cvt_f16_f32 v16, v8 // (op 146) 
v_cvt_f16_f32 v17, v9 // (op 146) 


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
v_cvt_f32_f16 v6, v8 // (op 1104) 
v_lshrrev_b32 v7, 16, v8 // (op 1104) 
v_cvt_f32_f16 v7, v7 // (op 1104) 
v_mul_f32 v18, v10, v6 // (op 1104) 
v_mul_f32 v19, v10, v7 // (op 1104) 
v_cvt_f32_f16 v6, v9 // (op 1104) 
v_lshrrev_b32 v7, 16, v9 // (op 1104) 
v_cvt_f32_f16 v7, v7 // (op 1104) 
v_mul_f32 v20, v10, v6 // (op 1104) 
v_mul_f32 v21, v10, v7 // (op 1104) 


v_add_f32 v6, v18, v14 // (op 1100) 
v_add_f32 v7, v19, v15 // (op 1100) 
v_add_f32 v8, v20, v16 // (op 1100) 
v_add_f32 v9, v21, v17 // (op 1100) 


v_cvt_f16_f32 v14, v6 // (op 1101) 
v_cvt_f16_f32 v15, v7 // (op 1101) 
v_cvt_f16_f32 v16, v8 // (op 1101) 
v_cvt_f16_f32 v17, v9 // (op 1101) 


v_pack_b32_f16 v8, v14, v15 // (op 1102) 
v_pack_b32_f16 v9, v16, v17 // (op 1102) 

s_lshl_b32 s0, s4, 6 // (op 1102) 
v_add_u32 v0, v13, s0 // (op 1102) 


buffer_store_dwordx2 v[8:9], v0, s[60:63], 0 offen // (op 1102) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalBottom_0_beta_135:












s_endpgm  // End of RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
.LRR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__end:
.size RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_, .LRR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__end-RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 108
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 33792 // lds bytes
  .amdhsa_accum_offset 92
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
    DataTypeB: B8
    DestDataType: h
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
  - .name: RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 33792
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 92
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

