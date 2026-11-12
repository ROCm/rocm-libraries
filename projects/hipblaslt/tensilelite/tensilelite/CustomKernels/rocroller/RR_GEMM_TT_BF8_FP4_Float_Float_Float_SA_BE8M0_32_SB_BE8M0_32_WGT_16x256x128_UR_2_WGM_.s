// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_,@function
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 3660) 
s_mul_hi_i32 s1, s16, s11 // (op 3849) 
v_lshrrev_b32 v0, 7, v1 // (op 3717) 
v_and_b32 v3, 15, v1 // (op 3717) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3717) 


v_and_b32 v0, 48, v1 // (op 3717) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 3717) 


v_lshrrev_b32 v0, 7, v1 // (op 3875) 
v_and_b32 v4, 15, v1 // (op 3875) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3875) 


v_and_b32 v0, 48, v1 // (op 3875) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 3875) 


v_lshrrev_b32 v0, 7, v1 // (op 3660) 
s_add_i32 s18, s0, s16 // (op 3660) 

v_and_b32 v5, 63, v1 // (op 3660) 
s_ashr_i32 s0, s18, 31 // (op 3660) 
v_lshrrev_b32 v6, 4, v5 // (op 3660) 
s_cmp_eq_i32 s11, 0 // (op 3660) 
s_mov_b32 s19, scc // (op 3660) 
v_lshlrev_b32 v7, 2, v6 // (op 3660) 

s_cmp_eq_i32 s19, 0 // (op 3660) 
s_cselect_b32 s51, 0, -1 // (op 3660) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3660) 


s_and_b32 s19, s12, 31 // (op 3660) 
v_lshrrev_b32 v0, 2, v5 // (op 3660) 

v_and_b32 v5, 3, v0 // (op 3660) 

v_and_b32 v0, 63, v1 // (op 3775) 
v_and_b32 v7, 48, v1 // (op 3775) 
v_sub_u32 v8, v0, v7 // (op 3775) 

v_lshrrev_b32 v7, 4, v0 // (op 3775) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 3775) 


v_lshrrev_b32 v7, 6, v1 // (op 3775) 
v_and_b32 v8, 1, v7 // (op 3775) 

v_lshlrev_b32 v7, 4, v8 // (op 3775) 

v_lshl_add_u32 v8, v0, 8, v7 // (op 3775) 


v_lshlrev_b32 v0, 2, v8 // (op 3775) 

v_lshrrev_b32 v7, 3, v0 // (op 3775) 

v_lshrrev_b32 v0, 6, v1 // (op 3833) 
v_and_b32 v8, 1, v0 // (op 3833) 

v_and_b32 v0, 15, v1 // (op 3833) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 3833) 


v_and_b32 v0, 63, v1 // (op 3833) 
v_lshrrev_b32 v8, 4, v0 // (op 3833) 

v_lshl_add_u32 v0, v8, 8, v9 // (op 3833) 


v_lshrrev_b32 v8, 7, v1 // (op 3849) 
v_and_b32 v9, 63, v1 // (op 3849) 
v_and_b32 v10, 63, v1 // (op 3904) 
v_and_b32 v11, 48, v1 // (op 3904) 
v_sub_u32 v12, v10, v11 // (op 3904) 

v_lshrrev_b32 v11, 4, v10 // (op 3904) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 3904) 


v_lshrrev_b32 v11, 6, v1 // (op 3904) 
v_and_b32 v12, 1, v11 // (op 3904) 

v_lshlrev_b32 v11, 4, v12 // (op 3904) 

v_lshl_add_u32 v12, v10, 8, v11 // (op 3904) 


v_lshlrev_b32 v10, 2, v12 // (op 3904) 

v_lshrrev_b32 v11, 3, v10 // (op 3904) 

v_lshrrev_b32 v10, 6, v1 // (op 3933) 
v_and_b32 v12, 1, v10 // (op 3933) 

v_and_b32 v10, 15, v1 // (op 3933) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 3933) 


v_and_b32 v10, 63, v1 // (op 3933) 
v_lshrrev_b32 v12, 4, v10 // (op 3933) 

v_lshl_add_u32 v10, v12, 8, v13 // (op 3933) 


s_lshl_b32 s52, 1, s19 // (op 3660) 
v_and_b32 v12, 3, v1 // (op 3660) 
s_add_i32 s53, s52, s51 // (op 3660) 


s_and_b32 s51, s0, s53 // (op 3660) 


s_add_i32 s0, s1, s16 // (op 3849) 

s_ashr_i32 s1, s0, 31 // (op 3849) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 3660) 


s_add_i32 s52, s18, s51 // (op 3660) 


v_lshrrev_b32 v5, 6, v1 // (op 3660) 
s_ashr_i32 s18, s52, s19 // (op 3660) 


v_and_b32 v12, 1, v5 // (op 3660) 

s_lshl_b32 s19, s12, 24 // (op 3660) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 3660) 


s_ashr_i32 s51, s19, 31 // (op 3660) 

s_xor_b32 s19, s18, s51 // (op 3660) 

s_sub_i32 s18, s19, s51 // (op 3660) 


s_mul_hi_i32 s19, s18, s13 // (op 3660) 
s_add_i32 s51, s19, s18 // (op 3660) 

s_ashr_i32 s19, s51, 31 // (op 3660) 
s_cmp_eq_i32 s13, 0 // (op 3660) 
s_mov_b32 s52, scc // (op 3660) 
s_cmp_eq_i32 s52, 0 // (op 3660) 
s_cselect_b32 s53, 0, -1 // (op 3660) 

s_and_b32 s52, s14, 31 // (op 3660) 
v_lshrrev_b32 v12, 4, v9 // (op 3849) 
s_cmp_eq_i32 s11, 0 // (op 3849) 
s_mov_b32 s54, scc // (op 3849) 
v_lshlrev_b32 v13, 2, v12 // (op 3849) 

s_cmp_eq_i32 s54, 0 // (op 3849) 
s_cselect_b32 s55, 0, -1 // (op 3849) 

s_lshl_b32 s54, 1, s52 // (op 3660) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 3849) 


v_lshrrev_b32 v8, 2, v9 // (op 3849) 

s_add_i32 s56, s54, s53 // (op 3660) 


s_and_b32 s53, s19, s56 // (op 3660) 


s_add_i32 s19, s51, s53 // (op 3660) 


s_ashr_i32 s51, s19, s52 // (op 3660) 


s_lshl_b32 s19, s14, 24 // (op 3660) 
s_ashr_i32 s52, s19, 31 // (op 3660) 

s_xor_b32 s19, s51, s52 // (op 3660) 

s_sub_i32 s51, s19, s52 // (op 3660) 


s_and_b32 s19, s12, 31 // (op 3849) 
s_lshl_b32 s52, 1, s19 // (op 3849) 
v_and_b32 v9, 3, v8 // (op 3849) 

s_add_i32 s53, s52, s55 // (op 3849) 


v_and_b32 v8, 3, v1 // (op 3849) 
s_and_b32 s52, s1, s53 // (op 3849) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 3849) 


s_add_i32 s1, s0, s52 // (op 3849) 


v_lshrrev_b32 v8, 6, v1 // (op 3849) 
s_ashr_i32 s0, s1, s19 // (op 3849) 


v_and_b32 v9, 1, v8 // (op 3849) 

s_lshl_b32 s1, s12, 24 // (op 3849) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 3849) 


s_ashr_i32 s19, s1, 31 // (op 3849) 

s_xor_b32 s1, s0, s19 // (op 3849) 

s_sub_i32 s0, s1, s19 // (op 3849) 


s_mul_hi_i32 s1, s0, s13 // (op 3849) 
s_add_i32 s19, s1, s0 // (op 3849) 

s_ashr_i32 s1, s19, 31 // (op 3849) 
s_cmp_eq_i32 s13, 0 // (op 3849) 
s_mov_b32 s52, scc // (op 3849) 
s_cmp_eq_i32 s52, 0 // (op 3849) 
s_cselect_b32 s53, 0, -1 // (op 3849) 

s_and_b32 s52, s14, 31 // (op 3849) 
s_lshl_b32 s54, 1, s52 // (op 3849) 
s_add_i32 s55, s54, s53 // (op 3849) 


s_and_b32 s53, s1, s55 // (op 3849) 


s_add_i32 s1, s19, s53 // (op 3849) 


s_ashr_i32 s19, s1, s52 // (op 3849) 


s_lshl_b32 s1, s14, 24 // (op 3849) 
s_ashr_i32 s52, s1, 31 // (op 3849) 

s_xor_b32 s1, s19, s52 // (op 3849) 

s_sub_i32 s19, s1, s52 // (op 3849) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 3660) 
s_sub_u32 s52, s16, s1 // (op 3660) 

s_mul_i32 s1, s51, s44 // (op 3660) 
s_add_u32 s1, s1, s52 // (op 3660) 

s_mul_hi_i32 s52, s16, s10 // (op 3660) 
s_add_i32 s53, s52, s16 // (op 3660) 

s_ashr_i32 s52, s53, 31 // (op 3660) 
s_cmp_eq_i32 s10, 0 // (op 3660) 
s_mov_b32 s54, scc // (op 3660) 
s_cmp_eq_i32 s54, 0 // (op 3660) 
s_cselect_b32 s55, 0, -1 // (op 3660) 

s_and_b32 s54, s47, 31 // (op 3660) 
s_lshl_b32 s56, 1, s54 // (op 3660) 
s_add_i32 s57, s56, s55 // (op 3660) 


s_and_b32 s55, s52, s57 // (op 3660) 


s_add_i32 s52, s53, s55 // (op 3660) 


s_ashr_i32 s53, s52, s54 // (op 3660) 


s_lshl_b32 s52, s47, 24 // (op 3660) 
s_ashr_i32 s54, s52, 31 // (op 3660) 

s_xor_b32 s52, s53, s54 // (op 3660) 

s_sub_i32 s53, s52, s54 // (op 3660) 


s_mul_i32 s52, s0, s44 // (op 3849) 
s_sub_u32 s54, s16, s52 // (op 3849) 

s_mul_i32 s52, s19, s44 // (op 3849) 
s_add_u32 s52, s52, s54 // (op 3849) 

s_mul_hi_i32 s54, s16, s10 // (op 3849) 
s_add_i32 s55, s54, s16 // (op 3849) 

s_ashr_i32 s54, s55, 31 // (op 3849) 
s_cmp_eq_i32 s10, 0 // (op 3849) 
s_mov_b32 s56, scc // (op 3849) 
s_cmp_eq_i32 s56, 0 // (op 3849) 
s_cselect_b32 s57, 0, -1 // (op 3849) 

s_and_b32 s56, s47, 31 // (op 3849) 
s_lshl_b32 s58, 1, s56 // (op 3849) 
s_add_i32 s59, s58, s57 // (op 3849) 


s_and_b32 s57, s54, s59 // (op 3849) 


s_add_i32 s54, s55, s57 // (op 3849) 


s_ashr_i32 s55, s54, s56 // (op 3849) 


s_lshl_b32 s54, s47, 24 // (op 3849) 
s_ashr_i32 s56, s54, 31 // (op 3849) 

s_xor_b32 s54, s55, s56 // (op 3849) 

s_sub_i32 s55, s54, s56 // (op 3849) 


s_add_u32 s56, s6, 256 // (op 3660) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 3660) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3660) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3660) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 3660) 

s_mul_i32 s54, s44, s56 // (op 3660) 
s_mul_i32 s58, s48, s54 // (op 3660) 

s_mul_i32 s54, s53, s58 // (op 3660) 

s_sub_u32 s58, s16, s54 // (op 3660) 

s_mov_b32 s54, s58 // (op 3660) 
s_mul_hi_i32 s59, s54, s15 // (op 3660) 
s_add_i32 s60, s59, s54 // (op 3660) 


s_ashr_i32 s54, s60, 31 // (op 3660) 
s_cmp_eq_i32 s15, 0 // (op 3660) 
s_mov_b32 s59, scc // (op 3660) 
s_cmp_eq_i32 s59, 0 // (op 3660) 
s_cselect_b32 s61, 0, -1 // (op 3660) 

s_and_b32 s59, s46, 31 // (op 3660) 
s_lshl_b32 s62, 1, s59 // (op 3660) 
s_add_i32 s63, s62, s61 // (op 3660) 


s_and_b32 s61, s54, s63 // (op 3660) 


s_add_i32 s54, s60, s61 // (op 3660) 


s_ashr_i32 s60, s54, s59 // (op 3660) 


s_lshl_b32 s54, s46, 24 // (op 3660) 
s_ashr_i32 s59, s54, 31 // (op 3660) 

s_xor_b32 s54, s60, s59 // (op 3660) 

s_sub_i32 s60, s54, s59 // (op 3660) 


s_mul_i32 s54, s45, s44 // (op 3660) 
s_add_u32 s62, s8, 16 // (op 3660) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 3660) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 3660) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 3660) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 4 // (op 3660) 

s_sub_i32 s59, s62, s54 // (op 3660) 

s_mul_i32 s61, s60, s59 // (op 3660) 

s_sub_u32 s59, s58, s61 // (op 3660) 


s_add_u32 s58, s59, s54 // (op 3660) 


s_cmp_eq_i32 s53, 0 // (op 3660) 
s_mov_b32 s54, scc // (op 3660) 

s_cmp_eq_i32 s54, 0 // (op 3660) 
s_cselect_b32 s53, s58, s1 // (op 3660) 


s_mul_i32 s1, s51, s56 // (op 3660) 


v_lshl_add_u32 v9, s53, 4, v6 // (op 3660) 


s_sub_i32 s51, s18, s1 // (op 3660) 


s_cmp_eq_i32 s54, 0 // (op 3660) 
s_cselect_b32 s1, s60, s51 // (op 3660) 



v_lshl_add_u32 v6, s1, 8, v5 // (op 3660) 


v_mul_lo_u32 v5, v6, s2 // (op 3660) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 3660) 


s_add_u32 s56, s6, 256 // (op 3849) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 3849) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3849) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3849) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 3849) 

s_mul_i32 s1, s44, s56 // (op 3849) 
s_mul_i32 s18, s48, s1 // (op 3849) 

s_mul_i32 s1, s55, s18 // (op 3849) 

s_sub_u32 s18, s16, s1 // (op 3849) 

s_mov_b32 s1, s18 // (op 3849) 
s_mul_hi_i32 s51, s1, s15 // (op 3849) 
s_add_i32 s53, s51, s1 // (op 3849) 


s_ashr_i32 s1, s53, 31 // (op 3849) 
s_cmp_eq_i32 s15, 0 // (op 3849) 
s_mov_b32 s51, scc // (op 3849) 
s_cmp_eq_i32 s51, 0 // (op 3849) 
s_cselect_b32 s54, 0, -1 // (op 3849) 

s_and_b32 s51, s46, 31 // (op 3849) 
s_lshl_b32 s58, 1, s51 // (op 3849) 
s_add_i32 s59, s58, s54 // (op 3849) 


s_and_b32 s54, s1, s59 // (op 3849) 


s_add_i32 s1, s53, s54 // (op 3849) 


s_ashr_i32 s53, s1, s51 // (op 3849) 


s_lshl_b32 s1, s46, 24 // (op 3849) 
s_ashr_i32 s51, s1, 31 // (op 3849) 

s_xor_b32 s1, s53, s51 // (op 3849) 

s_sub_i32 s53, s1, s51 // (op 3849) 


s_mul_i32 s1, s45, s44 // (op 3849) 
s_add_u32 s58, s8, 16 // (op 3849) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 3849) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 3849) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 3849) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 4 // (op 3849) 

s_sub_i32 s51, s58, s1 // (op 3849) 

s_mul_i32 s54, s53, s51 // (op 3849) 

s_sub_u32 s51, s18, s54 // (op 3849) 


s_add_u32 s18, s51, s1 // (op 3849) 


s_cmp_eq_i32 s55, 0 // (op 3849) 
s_mov_b32 s1, scc // (op 3849) 

s_cmp_eq_i32 s1, 0 // (op 3849) 
s_cselect_b32 s51, s18, s52 // (op 3849) 


s_mul_i32 s18, s19, s56 // (op 3849) 


v_lshl_add_u32 v5, s51, 4, v12 // (op 3849) 


s_sub_i32 s19, s0, s18 // (op 3849) 


s_cmp_eq_i32 s1, 0 // (op 3849) 
s_cselect_b32 s0, s53, s19 // (op 3849) 



v_lshl_add_u32 v9, s0, 8, v8 // (op 3849) 


v_mul_lo_u32 v8, v9, s4 // (op 3849) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 3849) 


s_mul_i32 s1, s2, s7 // (op 3662) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 3662) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 3662) least significant
s_mul_hi_u32 s19, s2, s6 // (op 3662) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3662) most significant: sum
s_add_u32 s1, s1, s19 // (op 3662) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3662) 

s_mul_i32 s1, s4, s7 // (op 3851) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 3851) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 3851) least significant
s_mul_hi_u32 s52, s4, s6 // (op 3851) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 3851) most significant: sum
s_add_u32 s1, s1, s52 // (op 3851) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 3851) 

s_mov_b32 s56, s20 // (op 3662) reg expression
s_mov_b32 s57, s21 // (op 3662) reg expression
s_mov_b32 s58, s18 // (op 3662) reg expression
s_mov_b32 s59, 131072 // (op 3662) call()

s_mov_b32 s60, s22 // (op 3851) reg expression

s_mov_b32 s61, s23 // (op 3851) reg expression
s_mov_b32 s62, s52 // (op 3851) reg expression
s_mov_b32 s63, 131072 // (op 3851) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 306) call()
v_accvgpr_write a5, 0.00000 // (op 306) call()
v_accvgpr_write a6, 0.00000 // (op 306) call()
v_accvgpr_write a7, 0.00000 // (op 306) call()
v_accvgpr_write a8, 0.00000 // (op 462) call()
v_accvgpr_write a9, 0.00000 // (op 462) call()
v_accvgpr_write a10, 0.00000 // (op 462) call()
v_accvgpr_write a11, 0.00000 // (op 462) call()
v_accvgpr_write a12, 0.00000 // (op 618) call()
v_accvgpr_write a13, 0.00000 // (op 618) call()
v_accvgpr_write a14, 0.00000 // (op 618) call()
v_accvgpr_write a15, 0.00000 // (op 618) call()
v_accvgpr_write a16, 0.00000 // (op 774) call()
v_accvgpr_write a17, 0.00000 // (op 774) call()
v_accvgpr_write a18, 0.00000 // (op 774) call()
v_accvgpr_write a19, 0.00000 // (op 774) call()
v_accvgpr_write a20, 0.00000 // (op 930) call()
v_accvgpr_write a21, 0.00000 // (op 930) call()
v_accvgpr_write a22, 0.00000 // (op 930) call()
v_accvgpr_write a23, 0.00000 // (op 930) call()
v_accvgpr_write a24, 0.00000 // (op 1086) call()
v_accvgpr_write a25, 0.00000 // (op 1086) call()
v_accvgpr_write a26, 0.00000 // (op 1086) call()
v_accvgpr_write a27, 0.00000 // (op 1086) call()
v_accvgpr_write a28, 0.00000 // (op 1242) call()
v_accvgpr_write a29, 0.00000 // (op 1242) call()
v_accvgpr_write a30, 0.00000 // (op 1242) call()
v_accvgpr_write a31, 0.00000 // (op 1242) call()
s_mul_hi_i32 s0, s16, s11 // (op 3427) 
v_lshrrev_b32 v12, 3, v1 // (op 3427) 
s_add_i32 s1, s0, s16 // (op 3427) 

v_and_b32 v13, -8, v1 // (op 3427) 
s_ashr_i32 s0, s1, 31 // (op 3427) 
v_sub_u32 v14, v1, v13 // (op 3427) 

s_cmp_eq_i32 s11, 0 // (op 3427) 
s_mov_b32 s18, scc // (op 3427) 
v_lshlrev_b32 v13, 4, v14 // (op 3427) 

s_cmp_eq_i32 s18, 0 // (op 3427) 
s_cselect_b32 s19, 0, -1 // (op 3427) 

s_and_b32 s18, s12, 31 // (op 3427) 
s_lshl_b32 s20, 1, s18 // (op 3427) 
s_add_i32 s21, s20, s19 // (op 3427) 


s_and_b32 s19, s0, s21 // (op 3427) 


s_add_i32 s0, s1, s19 // (op 3427) 


s_ashr_i32 s1, s0, s18 // (op 3427) 


s_lshl_b32 s0, s12, 24 // (op 3427) 
s_ashr_i32 s18, s0, 31 // (op 3427) 

s_xor_b32 s0, s1, s18 // (op 3427) 

s_sub_i32 s1, s0, s18 // (op 3427) 


s_mul_hi_i32 s0, s1, s13 // (op 3427) 
s_add_i32 s18, s0, s1 // (op 3427) 

s_ashr_i32 s0, s18, 31 // (op 3427) 
s_cmp_eq_i32 s13, 0 // (op 3427) 
s_mov_b32 s19, scc // (op 3427) 
s_cmp_eq_i32 s19, 0 // (op 3427) 
s_cselect_b32 s20, 0, -1 // (op 3427) 

s_and_b32 s19, s14, 31 // (op 3427) 
s_lshl_b32 s21, 1, s19 // (op 3427) 
s_add_i32 s22, s21, s20 // (op 3427) 


s_and_b32 s23, s0, s22 // (op 3427) 


s_add_i32 s0, s18, s23 // (op 3427) 


s_ashr_i32 s18, s0, s19 // (op 3427) 


s_lshl_b32 s0, s14, 24 // (op 3427) 
s_ashr_i32 s19, s0, 31 // (op 3427) 

s_xor_b32 s0, s18, s19 // (op 3427) 

s_sub_i32 s18, s0, s19 // (op 3427) 


s_mul_i32 s0, s1, s44 // (op 3427) 

s_sub_u32 s1, s16, s0 // (op 3427) 

s_mul_i32 s0, s18, s44 // (op 3427) 
s_add_u32 s0, s0, s1 // (op 3427) 


s_mul_hi_i32 s1, s16, s10 // (op 3427) 
s_add_i32 s18, s1, s16 // (op 3427) 

s_ashr_i32 s1, s18, 31 // (op 3427) 
s_cmp_eq_i32 s10, 0 // (op 3427) 
s_mov_b32 s19, scc // (op 3427) 
s_cmp_eq_i32 s19, 0 // (op 3427) 
s_cselect_b32 s20, 0, -1 // (op 3427) 

s_and_b32 s19, s47, 31 // (op 3427) 
s_lshl_b32 s21, 1, s19 // (op 3427) 
s_add_i32 s22, s21, s20 // (op 3427) 


s_and_b32 s23, s1, s22 // (op 3427) 


s_add_i32 s1, s18, s23 // (op 3427) 


s_ashr_i32 s18, s1, s19 // (op 3427) 


s_lshl_b32 s1, s47, 24 // (op 3427) 
s_ashr_i32 s19, s1, 31 // (op 3427) 

s_xor_b32 s1, s18, s19 // (op 3427) 

s_sub_i32 s18, s1, s19 // (op 3427) 


s_add_u32 s22, s6, 256 // (op 3427) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 3427) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 3427) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 3427) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 8 // (op 3427) 

s_mul_i32 s1, s44, s22 // (op 3427) 

s_mul_i32 s19, s48, s1 // (op 3427) 

s_mul_i32 s1, s18, s19 // (op 3427) 

s_sub_u32 s19, s16, s1 // (op 3427) 

s_mov_b32 s1, s19 // (op 3427) 
s_mul_hi_i32 s20, s1, s15 // (op 3427) 
s_add_i32 s21, s20, s1 // (op 3427) 


s_ashr_i32 s1, s21, 31 // (op 3427) 
s_cmp_eq_i32 s15, 0 // (op 3427) 
s_mov_b32 s20, scc // (op 3427) 
s_cmp_eq_i32 s20, 0 // (op 3427) 
s_cselect_b32 s22, 0, -1 // (op 3427) 

s_and_b32 s20, s46, 31 // (op 3427) 
s_lshl_b32 s23, 1, s20 // (op 3427) 
s_add_i32 s49, s23, s22 // (op 3427) 


s_and_b32 s22, s1, s49 // (op 3427) 


s_add_i32 s1, s21, s22 // (op 3427) 


s_ashr_i32 s21, s1, s20 // (op 3427) 


s_lshl_b32 s1, s46, 24 // (op 3427) 
s_ashr_i32 s20, s1, 31 // (op 3427) 

s_xor_b32 s1, s21, s20 // (op 3427) 

s_sub_i32 s21, s1, s20 // (op 3427) 


s_mul_i32 s1, s45, s44 // (op 3427) 
s_add_u32 s22, s8, 16 // (op 3427) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 3427) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 3427) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 3427) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 4 // (op 3427) 

s_sub_i32 s20, s22, s1 // (op 3427) 

s_mul_i32 s22, s21, s20 // (op 3427) 


s_sub_u32 s23, s19, s22 // (op 3427) 


s_add_u32 s19, s23, s1 // (op 3427) 


s_cmp_eq_i32 s18, 0 // (op 3427) 
s_mov_b32 s1, scc // (op 3427) 

s_cmp_eq_i32 s1, 0 // (op 3427) 
s_cselect_b32 s18, s19, s0 // (op 3427) 



v_lshl_add_u32 v14, s18, 4, v12 // (op 3427) 


v_mul_lo_u32 v12, v14, s24 // (op 3427) 
v_add_u32 v12, v12, v13 // (op 3427) 


s_mul_i32 s1, s24, s9 // (op 3429) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 3429) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3429) least significant
s_mul_hi_u32 s19, s24, s8 // (op 3429) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3429) most significant: sum
s_add_u32 s1, s1, s19 // (op 3429) most significant: sum


s_mov_b32 s20, s26 // (op 3429) reg expression
s_mov_b32 s21, s27 // (op 3429) reg expression
s_mov_b32 s22, s0 // (op 3429) reg expression
s_mov_b32 s23, 131072 // (op 3429) call()

s_mul_hi_i32 s0, s16, s11 // (op 3446) 
v_lshrrev_b32 v13, 3, v1 // (op 3446) 
s_add_i32 s1, s0, s16 // (op 3446) 

v_and_b32 v14, -8, v1 // (op 3446) 
s_ashr_i32 s0, s1, 31 // (op 3446) 
v_sub_u32 v15, v1, v14 // (op 3446) 

s_cmp_eq_i32 s11, 0 // (op 3446) 
s_mov_b32 s18, scc // (op 3446) 
v_lshlrev_b32 v14, 4, v15 // (op 3446) 

s_cmp_eq_i32 s18, 0 // (op 3446) 
s_cselect_b32 s19, 0, -1 // (op 3446) 

s_and_b32 s18, s12, 31 // (op 3446) 
s_lshl_b32 s49, 1, s18 // (op 3446) 
s_add_i32 s50, s49, s19 // (op 3446) 


s_and_b32 s19, s0, s50 // (op 3446) 


s_add_i32 s0, s1, s19 // (op 3446) 


s_ashr_i32 s1, s0, s18 // (op 3446) 


s_lshl_b32 s0, s12, 24 // (op 3446) 
s_ashr_i32 s18, s0, 31 // (op 3446) 

s_xor_b32 s0, s1, s18 // (op 3446) 

s_sub_i32 s1, s0, s18 // (op 3446) 


s_mul_hi_i32 s0, s1, s13 // (op 3446) 
s_add_i32 s18, s0, s1 // (op 3446) 

s_ashr_i32 s0, s18, 31 // (op 3446) 
s_cmp_eq_i32 s13, 0 // (op 3446) 
s_mov_b32 s19, scc // (op 3446) 
s_cmp_eq_i32 s19, 0 // (op 3446) 
s_cselect_b32 s49, 0, -1 // (op 3446) 

s_and_b32 s19, s14, 31 // (op 3446) 
s_lshl_b32 s50, 1, s19 // (op 3446) 
s_add_i32 s51, s50, s49 // (op 3446) 


s_and_b32 s49, s0, s51 // (op 3446) 


s_add_i32 s0, s18, s49 // (op 3446) 


s_ashr_i32 s18, s0, s19 // (op 3446) 


s_lshl_b32 s0, s14, 24 // (op 3446) 
s_ashr_i32 s19, s0, 31 // (op 3446) 

s_xor_b32 s0, s18, s19 // (op 3446) 

s_sub_i32 s18, s0, s19 // (op 3446) 


s_mul_i32 s0, s1, s44 // (op 3446) 

s_sub_u32 s1, s16, s0 // (op 3446) 

s_mul_i32 s0, s18, s44 // (op 3446) 
s_add_u32 s0, s0, s1 // (op 3446) 


s_mul_hi_i32 s1, s16, s10 // (op 3446) 
s_add_i32 s18, s1, s16 // (op 3446) 

s_ashr_i32 s1, s18, 31 // (op 3446) 
s_cmp_eq_i32 s10, 0 // (op 3446) 
s_mov_b32 s19, scc // (op 3446) 
s_cmp_eq_i32 s19, 0 // (op 3446) 
s_cselect_b32 s49, 0, -1 // (op 3446) 

s_and_b32 s19, s47, 31 // (op 3446) 
s_lshl_b32 s50, 1, s19 // (op 3446) 
s_add_i32 s51, s50, s49 // (op 3446) 


s_and_b32 s49, s1, s51 // (op 3446) 


s_add_i32 s1, s18, s49 // (op 3446) 


s_ashr_i32 s18, s1, s19 // (op 3446) 


s_lshl_b32 s1, s47, 24 // (op 3446) 
s_ashr_i32 s19, s1, 31 // (op 3446) 

s_xor_b32 s1, s18, s19 // (op 3446) 

s_sub_i32 s18, s1, s19 // (op 3446) 


s_add_u32 s54, s6, 256 // (op 3446) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 3446) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3446) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3446) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 3446) 

s_mul_i32 s1, s44, s50 // (op 3446) 

s_mul_i32 s19, s48, s1 // (op 3446) 

s_mul_i32 s1, s18, s19 // (op 3446) 

s_sub_u32 s19, s16, s1 // (op 3446) 

s_mov_b32 s1, s19 // (op 3446) 
s_mul_hi_i32 s49, s1, s15 // (op 3446) 
s_add_i32 s50, s49, s1 // (op 3446) 


s_ashr_i32 s1, s50, 31 // (op 3446) 
s_cmp_eq_i32 s15, 0 // (op 3446) 
s_mov_b32 s49, scc // (op 3446) 
s_cmp_eq_i32 s49, 0 // (op 3446) 
s_cselect_b32 s51, 0, -1 // (op 3446) 

s_and_b32 s49, s46, 31 // (op 3446) 
s_lshl_b32 s52, 1, s49 // (op 3446) 
s_add_i32 s53, s52, s51 // (op 3446) 


s_and_b32 s51, s1, s53 // (op 3446) 


s_add_i32 s1, s50, s51 // (op 3446) 


s_ashr_i32 s50, s1, s49 // (op 3446) 


s_lshl_b32 s1, s46, 24 // (op 3446) 
s_ashr_i32 s49, s1, 31 // (op 3446) 

s_xor_b32 s1, s50, s49 // (op 3446) 

s_sub_i32 s50, s1, s49 // (op 3446) 


s_mul_i32 s1, s45, s44 // (op 3446) 
s_add_u32 s54, s8, 16 // (op 3446) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 3446) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3446) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3446) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 4 // (op 3446) 

s_sub_i32 s49, s54, s1 // (op 3446) 

s_mul_i32 s51, s50, s49 // (op 3446) 


s_sub_u32 s49, s19, s51 // (op 3446) 


s_add_u32 s19, s49, s1 // (op 3446) 


s_cmp_eq_i32 s18, 0 // (op 3446) 
s_mov_b32 s1, scc // (op 3446) 

s_cmp_eq_i32 s1, 0 // (op 3446) 
s_cselect_b32 s18, s19, s0 // (op 3446) 



v_lshl_add_u32 v15, s18, 4, v13 // (op 3446) 


v_mul_lo_u32 v13, v15, s24 // (op 3446) 
v_add_u32 v13, v13, v14 // (op 3446) 


s_mul_i32 s1, s24, s9 // (op 3448) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 3448) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3448) least significant
s_mul_hi_u32 s19, s24, s8 // (op 3448) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3448) most significant: sum
s_add_u32 s1, s1, s19 // (op 3448) most significant: sum


s_mov_b32 s20, s26 // (op 3448) reg expression
s_mov_b32 s21, s27 // (op 3448) reg expression
s_mov_b32 s22, s0 // (op 3448) reg expression
s_mov_b32 s23, 131072 // (op 3448) call()

s_mul_hi_i32 s0, s16, s11 // (op 3491) 
v_and_b32 v14, -8, v1 // (op 3491) 
s_add_i32 s1, s0, s16 // (op 3491) 

v_sub_u32 v15, v1, v14 // (op 3491) 

s_ashr_i32 s0, s1, 31 // (op 3491) 
v_lshlrev_b32 v14, 5, v15 // (op 3491) 

s_cmp_eq_i32 s11, 0 // (op 3491) 
s_mov_b32 s18, scc // (op 3491) 
v_and_b32 v15, 4294967288, v1 // (op 3491) 
s_cmp_eq_i32 s18, 0 // (op 3491) 
s_cselect_b32 s19, 0, -1 // (op 3491) 

s_and_b32 s18, s12, 31 // (op 3491) 
s_lshl_b32 s49, 1, s18 // (op 3491) 
s_add_i32 s50, s49, s19 // (op 3491) 


s_and_b32 s19, s0, s50 // (op 3491) 


s_add_i32 s0, s1, s19 // (op 3491) 


s_ashr_i32 s1, s0, s18 // (op 3491) 


s_lshl_b32 s0, s12, 24 // (op 3491) 
s_ashr_i32 s18, s0, 31 // (op 3491) 

s_xor_b32 s0, s1, s18 // (op 3491) 

s_sub_i32 s1, s0, s18 // (op 3491) 


s_mul_hi_i32 s0, s1, s13 // (op 3491) 
s_add_i32 s18, s0, s1 // (op 3491) 

s_ashr_i32 s0, s18, 31 // (op 3491) 
s_cmp_eq_i32 s13, 0 // (op 3491) 
s_mov_b32 s19, scc // (op 3491) 
s_cmp_eq_i32 s19, 0 // (op 3491) 
s_cselect_b32 s49, 0, -1 // (op 3491) 

s_and_b32 s19, s14, 31 // (op 3491) 
s_lshl_b32 s50, 1, s19 // (op 3491) 
s_add_i32 s51, s50, s49 // (op 3491) 


s_and_b32 s49, s0, s51 // (op 3491) 


s_add_i32 s0, s18, s49 // (op 3491) 


s_ashr_i32 s18, s0, s19 // (op 3491) 


s_lshl_b32 s0, s14, 24 // (op 3491) 
s_ashr_i32 s19, s0, 31 // (op 3491) 

s_xor_b32 s0, s18, s19 // (op 3491) 

s_sub_i32 s18, s0, s19 // (op 3491) 


s_add_u32 s54, s6, 256 // (op 3491) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 3491) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3491) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3491) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 3491) 

s_mul_i32 s0, s18, s50 // (op 3491) 

s_sub_i32 s49, s1, s0 // (op 3491) 


s_mul_hi_i32 s0, s16, s10 // (op 3491) 
s_add_i32 s1, s0, s16 // (op 3491) 

s_ashr_i32 s0, s1, 31 // (op 3491) 
s_cmp_eq_i32 s10, 0 // (op 3491) 
s_mov_b32 s18, scc // (op 3491) 
s_cmp_eq_i32 s18, 0 // (op 3491) 
s_cselect_b32 s19, 0, -1 // (op 3491) 

s_and_b32 s18, s47, 31 // (op 3491) 
s_lshl_b32 s52, 1, s18 // (op 3491) 
s_add_i32 s53, s52, s19 // (op 3491) 


s_and_b32 s19, s0, s53 // (op 3491) 


s_add_i32 s0, s1, s19 // (op 3491) 


s_ashr_i32 s1, s0, s18 // (op 3491) 


s_lshl_b32 s0, s47, 24 // (op 3491) 
s_ashr_i32 s18, s0, 31 // (op 3491) 

s_xor_b32 s0, s1, s18 // (op 3491) 

s_sub_i32 s1, s0, s18 // (op 3491) 


s_mul_i32 s0, s44, s50 // (op 3491) 

s_mul_i32 s18, s48, s0 // (op 3491) 

s_mul_i32 s0, s1, s18 // (op 3491) 

s_sub_u32 s18, s16, s0 // (op 3491) 

s_mul_hi_i32 s0, s18, s15 // (op 3491) 
s_add_i32 s19, s0, s18 // (op 3491) 


s_ashr_i32 s0, s19, 31 // (op 3491) 
s_cmp_eq_i32 s15, 0 // (op 3491) 
s_mov_b32 s18, scc // (op 3491) 
s_cmp_eq_i32 s18, 0 // (op 3491) 
s_cselect_b32 s50, 0, -1 // (op 3491) 

s_and_b32 s18, s46, 31 // (op 3491) 
s_lshl_b32 s51, 1, s18 // (op 3491) 
s_add_i32 s52, s51, s50 // (op 3491) 


s_and_b32 s50, s0, s52 // (op 3491) 


s_add_i32 s0, s19, s50 // (op 3491) 


s_ashr_i32 s19, s0, s18 // (op 3491) 


s_lshl_b32 s0, s46, 24 // (op 3491) 
s_ashr_i32 s18, s0, 31 // (op 3491) 

s_xor_b32 s0, s19, s18 // (op 3491) 

s_sub_i32 s19, s0, s18 // (op 3491) 


s_cmp_eq_i32 s1, 0 // (op 3491) 
s_mov_b32 s0, scc // (op 3491) 

s_cmp_eq_i32 s0, 0 // (op 3491) 
s_cselect_b32 s1, s19, s49 // (op 3491) 



v_lshl_add_u32 v16, s1, 8, v14 // (op 3491) 


v_mov_b32 v19, 0 // (op 3491) convert
v_mov_b32 v18, v16 // (op 3491) convert

v_mul_lo_u32 v17, v15, s29 // (op 3491) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 3491) most significant: high of low * low
v_mul_lo_u32 v16, v15, s28 // (op 3491) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 3491) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 3491) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 3491) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 3491) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 3491) 

v_mov_b32 v14, v16 // (op 3491) convert

s_mul_i32 s1, s28, s33 // (op 3493) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 3493) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3493) least significant
s_mul_hi_u32 s19, s28, s32 // (op 3493) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3493) most significant: sum
s_add_u32 s1, s1, s19 // (op 3493) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3493) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3493) 

s_mov_b32 s52, s30 // (op 3493) reg expression
s_mov_b32 s53, s31 // (op 3493) reg expression
s_mov_b32 s54, s0 // (op 3493) reg expression
s_mov_b32 s55, 131072 // (op 3493) call()

s_mul_hi_i32 s0, s16, s11 // (op 3510) 
v_and_b32 v15, -8, v1 // (op 3510) 
s_add_i32 s1, s0, s16 // (op 3510) 

v_sub_u32 v16, v1, v15 // (op 3510) 

s_ashr_i32 s0, s1, 31 // (op 3510) 
v_lshlrev_b32 v15, 5, v16 // (op 3510) 

s_cmp_eq_i32 s11, 0 // (op 3510) 
s_mov_b32 s18, scc // (op 3510) 
v_and_b32 v16, 4294967288, v1 // (op 3510) 
s_cmp_eq_i32 s18, 0 // (op 3510) 
s_cselect_b32 s19, 0, -1 // (op 3510) 

s_and_b32 s18, s12, 31 // (op 3510) 
s_lshl_b32 s49, 1, s18 // (op 3510) 
s_add_i32 s50, s49, s19 // (op 3510) 


s_and_b32 s19, s0, s50 // (op 3510) 


s_add_i32 s0, s1, s19 // (op 3510) 


s_ashr_i32 s1, s0, s18 // (op 3510) 


s_lshl_b32 s0, s12, 24 // (op 3510) 
s_ashr_i32 s18, s0, 31 // (op 3510) 

s_xor_b32 s0, s1, s18 // (op 3510) 

s_sub_i32 s1, s0, s18 // (op 3510) 


s_mul_hi_i32 s0, s1, s13 // (op 3510) 
s_add_i32 s18, s0, s1 // (op 3510) 

s_ashr_i32 s0, s18, 31 // (op 3510) 
s_cmp_eq_i32 s13, 0 // (op 3510) 
s_mov_b32 s19, scc // (op 3510) 
s_cmp_eq_i32 s19, 0 // (op 3510) 
s_cselect_b32 s49, 0, -1 // (op 3510) 

s_and_b32 s19, s14, 31 // (op 3510) 
s_lshl_b32 s50, 1, s19 // (op 3510) 
s_add_i32 s51, s50, s49 // (op 3510) 


s_and_b32 s49, s0, s51 // (op 3510) 


s_add_i32 s0, s18, s49 // (op 3510) 


s_ashr_i32 s18, s0, s19 // (op 3510) 


s_lshl_b32 s0, s14, 24 // (op 3510) 
s_ashr_i32 s19, s0, 31 // (op 3510) 

s_xor_b32 s0, s18, s19 // (op 3510) 

s_sub_i32 s18, s0, s19 // (op 3510) 


s_add_u32 s50, s6, 256 // (op 3510) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3510) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3510) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3510) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 3510) 

s_mul_i32 s0, s18, s50 // (op 3510) 

s_sub_i32 s49, s1, s0 // (op 3510) 


s_mul_hi_i32 s0, s16, s10 // (op 3510) 
s_add_i32 s1, s0, s16 // (op 3510) 

s_ashr_i32 s0, s1, 31 // (op 3510) 
s_cmp_eq_i32 s10, 0 // (op 3510) 
s_mov_b32 s18, scc // (op 3510) 
s_cmp_eq_i32 s18, 0 // (op 3510) 
s_cselect_b32 s19, 0, -1 // (op 3510) 

s_and_b32 s18, s47, 31 // (op 3510) 
s_lshl_b32 s64, 1, s18 // (op 3510) 
s_add_i32 s65, s64, s19 // (op 3510) 


s_and_b32 s19, s0, s65 // (op 3510) 


s_add_i32 s0, s1, s19 // (op 3510) 


s_ashr_i32 s1, s0, s18 // (op 3510) 


s_lshl_b32 s0, s47, 24 // (op 3510) 
s_ashr_i32 s18, s0, 31 // (op 3510) 

s_xor_b32 s0, s1, s18 // (op 3510) 

s_sub_i32 s1, s0, s18 // (op 3510) 


s_mul_i32 s0, s44, s50 // (op 3510) 

s_mul_i32 s18, s48, s0 // (op 3510) 

s_mul_i32 s0, s1, s18 // (op 3510) 

s_sub_u32 s18, s16, s0 // (op 3510) 

s_mul_hi_i32 s0, s18, s15 // (op 3510) 
s_add_i32 s19, s0, s18 // (op 3510) 


s_ashr_i32 s0, s19, 31 // (op 3510) 
s_cmp_eq_i32 s15, 0 // (op 3510) 
s_mov_b32 s18, scc // (op 3510) 
s_cmp_eq_i32 s18, 0 // (op 3510) 
s_cselect_b32 s50, 0, -1 // (op 3510) 

s_and_b32 s18, s46, 31 // (op 3510) 
s_lshl_b32 s51, 1, s18 // (op 3510) 
s_add_i32 s64, s51, s50 // (op 3510) 


s_and_b32 s50, s0, s64 // (op 3510) 


s_add_i32 s0, s19, s50 // (op 3510) 


s_ashr_i32 s19, s0, s18 // (op 3510) 


s_lshl_b32 s0, s46, 24 // (op 3510) 
s_ashr_i32 s18, s0, 31 // (op 3510) 

s_xor_b32 s0, s19, s18 // (op 3510) 

s_sub_i32 s19, s0, s18 // (op 3510) 


s_cmp_eq_i32 s1, 0 // (op 3510) 
s_mov_b32 s0, scc // (op 3510) 

s_cmp_eq_i32 s0, 0 // (op 3510) 
s_cselect_b32 s1, s19, s49 // (op 3510) 



v_lshl_add_u32 v17, s1, 8, v15 // (op 3510) 


v_mov_b32 v19, 0 // (op 3510) convert
v_mov_b32 v18, v17 // (op 3510) convert

v_mul_lo_u32 v21, v16, s29 // (op 3510) most significant: low of low * high
v_mul_hi_u32 v15, v16, s28 // (op 3510) most significant: high of low * low
v_mul_lo_u32 v20, v16, s28 // (op 3510) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 3510) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 3510) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 3510) most significant half



v_lshlrev_b64 v[18:19], 2, v[20:21] // (op 3510) 

v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 3510) 

v_mov_b32 v15, v16 // (op 3510) convert

s_mul_i32 s1, s28, s33 // (op 3512) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 3512) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3512) least significant
s_mul_hi_u32 s19, s28, s32 // (op 3512) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3512) most significant: sum
s_add_u32 s1, s1, s19 // (op 3512) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3512) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3512) 

s_mov_b32 s52, s30 // (op 3512) reg expression
s_mov_b32 s53, s31 // (op 3512) reg expression
s_mov_b32 s54, s0 // (op 3512) reg expression
s_mov_b32 s55, 131072 // (op 3512) call()

s_mul_hi_i32 s0, s16, s11 // (op 3555) 
v_lshrrev_b32 v16, 7, v1 // (op 3555) 
s_add_i32 s1, s0, s16 // (op 3555) 

v_and_b32 v17, 15, v1 // (op 3555) 
s_ashr_i32 s0, s1, 31 // (op 3555) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 3555) 


s_cmp_eq_i32 s11, 0 // (op 3555) 
s_mov_b32 s18, scc // (op 3555) 
v_and_b32 v16, 63, v1 // (op 3555) 
s_cmp_eq_i32 s18, 0 // (op 3555) 
s_cselect_b32 s19, 0, -1 // (op 3555) 

v_lshrrev_b32 v17, 4, v16 // (op 3555) 

s_and_b32 s18, s12, 31 // (op 3555) 
s_lshl_b32 s49, 1, s18 // (op 3555) 
s_add_i32 s50, s49, s19 // (op 3555) 


s_and_b32 s19, s0, s50 // (op 3555) 


s_add_i32 s0, s1, s19 // (op 3555) 


s_ashr_i32 s1, s0, s18 // (op 3555) 


s_lshl_b32 s0, s12, 24 // (op 3555) 
s_ashr_i32 s18, s0, 31 // (op 3555) 

s_xor_b32 s0, s1, s18 // (op 3555) 

s_sub_i32 s1, s0, s18 // (op 3555) 


s_mul_hi_i32 s0, s1, s13 // (op 3555) 
s_add_i32 s18, s0, s1 // (op 3555) 

s_ashr_i32 s0, s18, 31 // (op 3555) 
s_cmp_eq_i32 s13, 0 // (op 3555) 
s_mov_b32 s19, scc // (op 3555) 
s_cmp_eq_i32 s19, 0 // (op 3555) 
s_cselect_b32 s49, 0, -1 // (op 3555) 

s_and_b32 s19, s14, 31 // (op 3555) 
s_lshl_b32 s50, 1, s19 // (op 3555) 
s_add_i32 s51, s50, s49 // (op 3555) 


s_and_b32 s49, s0, s51 // (op 3555) 


s_add_i32 s0, s18, s49 // (op 3555) 


s_ashr_i32 s18, s0, s19 // (op 3555) 


s_lshl_b32 s0, s14, 24 // (op 3555) 
s_ashr_i32 s19, s0, 31 // (op 3555) 

s_xor_b32 s0, s18, s19 // (op 3555) 

s_sub_i32 s18, s0, s19 // (op 3555) 


s_mul_i32 s0, s1, s44 // (op 3555) 

s_sub_u32 s1, s16, s0 // (op 3555) 

s_mul_i32 s0, s18, s44 // (op 3555) 
s_add_u32 s0, s0, s1 // (op 3555) 


s_mul_hi_i32 s1, s16, s10 // (op 3555) 
s_add_i32 s18, s1, s16 // (op 3555) 

s_ashr_i32 s1, s18, 31 // (op 3555) 
s_cmp_eq_i32 s10, 0 // (op 3555) 
s_mov_b32 s19, scc // (op 3555) 
s_cmp_eq_i32 s19, 0 // (op 3555) 
s_cselect_b32 s49, 0, -1 // (op 3555) 

s_and_b32 s19, s47, 31 // (op 3555) 
s_lshl_b32 s50, 1, s19 // (op 3555) 
s_add_i32 s51, s50, s49 // (op 3555) 


s_and_b32 s49, s1, s51 // (op 3555) 


s_add_i32 s1, s18, s49 // (op 3555) 


s_ashr_i32 s18, s1, s19 // (op 3555) 


s_lshl_b32 s1, s47, 24 // (op 3555) 
s_ashr_i32 s19, s1, 31 // (op 3555) 

s_xor_b32 s1, s18, s19 // (op 3555) 

s_sub_i32 s18, s1, s19 // (op 3555) 


s_add_u32 s50, s6, 256 // (op 3555) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3555) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3555) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3555) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 3555) 

s_mul_i32 s1, s44, s50 // (op 3555) 

s_mul_i32 s19, s48, s1 // (op 3555) 

s_mul_i32 s1, s18, s19 // (op 3555) 

s_sub_u32 s19, s16, s1 // (op 3555) 

s_mov_b32 s1, s19 // (op 3555) 
s_mul_hi_i32 s49, s1, s15 // (op 3555) 
s_add_i32 s50, s49, s1 // (op 3555) 


s_ashr_i32 s1, s50, 31 // (op 3555) 
s_cmp_eq_i32 s15, 0 // (op 3555) 
s_mov_b32 s49, scc // (op 3555) 
s_cmp_eq_i32 s49, 0 // (op 3555) 
s_cselect_b32 s51, 0, -1 // (op 3555) 

s_and_b32 s49, s46, 31 // (op 3555) 
s_lshl_b32 s64, 1, s49 // (op 3555) 
s_add_i32 s65, s64, s51 // (op 3555) 


s_and_b32 s51, s1, s65 // (op 3555) 


s_add_i32 s1, s50, s51 // (op 3555) 


s_ashr_i32 s50, s1, s49 // (op 3555) 


s_lshl_b32 s1, s46, 24 // (op 3555) 
s_ashr_i32 s49, s1, 31 // (op 3555) 

s_xor_b32 s1, s50, s49 // (op 3555) 

s_sub_i32 s50, s1, s49 // (op 3555) 


s_mul_i32 s1, s45, s44 // (op 3555) 
s_add_u32 s64, s8, 16 // (op 3555) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 3555) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 3555) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 3555) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 3555) 

s_sub_i32 s49, s64, s1 // (op 3555) 

s_mul_i32 s51, s50, s49 // (op 3555) 


s_sub_u32 s49, s19, s51 // (op 3555) 


s_add_u32 s19, s49, s1 // (op 3555) 


s_cmp_eq_i32 s18, 0 // (op 3555) 
s_mov_b32 s1, scc // (op 3555) 

s_cmp_eq_i32 s1, 0 // (op 3555) 
s_cselect_b32 s18, s19, s0 // (op 3555) 



v_lshl_add_u32 v16, s18, 4, v18 // (op 3555) 


v_mul_lo_u32 v18, v16, s34 // (op 3555) 
v_add_u32 v18, v18, v17 // (op 3555) 


s_mul_i32 s1, s34, s9 // (op 3557) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 3557) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3557) least significant
s_mul_hi_u32 s19, s34, s8 // (op 3557) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3557) most significant: sum
s_add_u32 s1, s1, s19 // (op 3557) most significant: sum


s_mov_b32 s64, s36 // (op 3557) reg expression
s_mov_b32 s65, s37 // (op 3557) reg expression
s_mov_b32 s66, s0 // (op 3557) reg expression
s_mov_b32 s67, 131072 // (op 3557) call()

s_mul_hi_i32 s0, s16, s11 // (op 3574) 
v_lshrrev_b32 v16, 7, v1 // (op 3574) 
s_add_i32 s1, s0, s16 // (op 3574) 

v_and_b32 v17, 15, v1 // (op 3574) 
s_ashr_i32 s0, s1, 31 // (op 3574) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 3574) 


s_cmp_eq_i32 s11, 0 // (op 3574) 
s_mov_b32 s18, scc // (op 3574) 
v_and_b32 v16, 63, v1 // (op 3574) 
s_cmp_eq_i32 s18, 0 // (op 3574) 
s_cselect_b32 s19, 0, -1 // (op 3574) 

v_lshrrev_b32 v17, 4, v16 // (op 3574) 

s_and_b32 s18, s12, 31 // (op 3574) 
s_lshl_b32 s49, 1, s18 // (op 3574) 
s_add_i32 s50, s49, s19 // (op 3574) 


s_and_b32 s19, s0, s50 // (op 3574) 


s_add_i32 s0, s1, s19 // (op 3574) 


s_ashr_i32 s1, s0, s18 // (op 3574) 


s_lshl_b32 s0, s12, 24 // (op 3574) 
s_ashr_i32 s18, s0, 31 // (op 3574) 

s_xor_b32 s0, s1, s18 // (op 3574) 

s_sub_i32 s1, s0, s18 // (op 3574) 


s_mul_hi_i32 s0, s1, s13 // (op 3574) 
s_add_i32 s18, s0, s1 // (op 3574) 

s_ashr_i32 s0, s18, 31 // (op 3574) 
s_cmp_eq_i32 s13, 0 // (op 3574) 
s_mov_b32 s19, scc // (op 3574) 
s_cmp_eq_i32 s19, 0 // (op 3574) 
s_cselect_b32 s49, 0, -1 // (op 3574) 

s_and_b32 s19, s14, 31 // (op 3574) 
s_lshl_b32 s50, 1, s19 // (op 3574) 
s_add_i32 s51, s50, s49 // (op 3574) 


s_and_b32 s49, s0, s51 // (op 3574) 


s_add_i32 s0, s18, s49 // (op 3574) 


s_ashr_i32 s18, s0, s19 // (op 3574) 


s_lshl_b32 s0, s14, 24 // (op 3574) 
s_ashr_i32 s19, s0, 31 // (op 3574) 

s_xor_b32 s0, s18, s19 // (op 3574) 

s_sub_i32 s18, s0, s19 // (op 3574) 


s_mul_i32 s0, s1, s44 // (op 3574) 

s_sub_u32 s1, s16, s0 // (op 3574) 

s_mul_i32 s0, s18, s44 // (op 3574) 
s_add_u32 s0, s0, s1 // (op 3574) 


s_mul_hi_i32 s1, s16, s10 // (op 3574) 
s_add_i32 s18, s1, s16 // (op 3574) 

s_ashr_i32 s1, s18, 31 // (op 3574) 
s_cmp_eq_i32 s10, 0 // (op 3574) 
s_mov_b32 s19, scc // (op 3574) 
s_cmp_eq_i32 s19, 0 // (op 3574) 
s_cselect_b32 s49, 0, -1 // (op 3574) 

s_and_b32 s19, s47, 31 // (op 3574) 
s_lshl_b32 s50, 1, s19 // (op 3574) 
s_add_i32 s51, s50, s49 // (op 3574) 


s_and_b32 s49, s1, s51 // (op 3574) 


s_add_i32 s1, s18, s49 // (op 3574) 


s_ashr_i32 s18, s1, s19 // (op 3574) 


s_lshl_b32 s1, s47, 24 // (op 3574) 
s_ashr_i32 s19, s1, 31 // (op 3574) 

s_xor_b32 s1, s18, s19 // (op 3574) 

s_sub_i32 s18, s1, s19 // (op 3574) 


s_add_u32 s50, s6, 256 // (op 3574) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3574) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3574) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3574) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 3574) 

s_mul_i32 s1, s44, s50 // (op 3574) 

s_mul_i32 s19, s48, s1 // (op 3574) 

s_mul_i32 s1, s18, s19 // (op 3574) 

s_sub_u32 s19, s16, s1 // (op 3574) 

s_mov_b32 s1, s19 // (op 3574) 
s_mul_hi_i32 s49, s1, s15 // (op 3574) 
s_add_i32 s50, s49, s1 // (op 3574) 


s_ashr_i32 s1, s50, 31 // (op 3574) 
s_cmp_eq_i32 s15, 0 // (op 3574) 
s_mov_b32 s49, scc // (op 3574) 
s_cmp_eq_i32 s49, 0 // (op 3574) 
s_cselect_b32 s51, 0, -1 // (op 3574) 

s_and_b32 s49, s46, 31 // (op 3574) 
s_lshl_b32 s68, 1, s49 // (op 3574) 
s_add_i32 s69, s68, s51 // (op 3574) 


s_and_b32 s51, s1, s69 // (op 3574) 


s_add_i32 s1, s50, s51 // (op 3574) 


s_ashr_i32 s50, s1, s49 // (op 3574) 


s_lshl_b32 s1, s46, 24 // (op 3574) 
s_ashr_i32 s49, s1, 31 // (op 3574) 

s_xor_b32 s1, s50, s49 // (op 3574) 

s_sub_i32 s50, s1, s49 // (op 3574) 


s_mul_i32 s1, s45, s44 // (op 3574) 
s_add_u32 s68, s8, 16 // (op 3574) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 3574) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 3574) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 3574) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 3574) 

s_sub_i32 s49, s68, s1 // (op 3574) 

s_mul_i32 s51, s50, s49 // (op 3574) 


s_sub_u32 s49, s19, s51 // (op 3574) 


s_add_u32 s19, s49, s1 // (op 3574) 


s_cmp_eq_i32 s18, 0 // (op 3574) 
s_mov_b32 s1, scc // (op 3574) 

s_cmp_eq_i32 s1, 0 // (op 3574) 
s_cselect_b32 s18, s19, s0 // (op 3574) 



v_lshl_add_u32 v16, s18, 4, v19 // (op 3574) 


v_mul_lo_u32 v19, v16, s34 // (op 3574) 
v_add_u32 v19, v19, v17 // (op 3574) 


s_mul_i32 s1, s34, s9 // (op 3576) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 3576) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3576) least significant
s_mul_hi_u32 s19, s34, s8 // (op 3576) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3576) most significant: sum
s_add_u32 s1, s1, s19 // (op 3576) most significant: sum


s_mov_b32 s64, s36 // (op 3576) reg expression
s_mov_b32 s65, s37 // (op 3576) reg expression
s_mov_b32 s66, s0 // (op 3576) reg expression
s_mov_b32 s67, 131072 // (op 3576) call()

s_mul_hi_i32 s0, s16, s11 // (op 3619) 
v_and_b32 v16, -32, v1 // (op 3619) 
s_add_i32 s1, s0, s16 // (op 3619) 

v_sub_u32 v17, v1, v16 // (op 3619) 

s_ashr_i32 s0, s1, 31 // (op 3619) 
v_lshlrev_b32 v16, 3, v17 // (op 3619) 

s_cmp_eq_i32 s11, 0 // (op 3619) 
s_mov_b32 s18, scc // (op 3619) 
v_lshrrev_b32 v17, 5, v1 // (op 3619) 
s_cmp_eq_i32 s18, 0 // (op 3619) 
s_cselect_b32 s19, 0, -1 // (op 3619) 

s_and_b32 s18, s12, 31 // (op 3619) 
s_lshl_b32 s49, 1, s18 // (op 3619) 
s_add_i32 s50, s49, s19 // (op 3619) 


s_and_b32 s19, s0, s50 // (op 3619) 


s_add_i32 s0, s1, s19 // (op 3619) 


s_ashr_i32 s1, s0, s18 // (op 3619) 


s_lshl_b32 s0, s12, 24 // (op 3619) 
s_ashr_i32 s18, s0, 31 // (op 3619) 

s_xor_b32 s0, s1, s18 // (op 3619) 

s_sub_i32 s1, s0, s18 // (op 3619) 


s_mul_hi_i32 s0, s1, s13 // (op 3619) 
s_add_i32 s18, s0, s1 // (op 3619) 

s_ashr_i32 s0, s18, 31 // (op 3619) 
s_cmp_eq_i32 s13, 0 // (op 3619) 
s_mov_b32 s19, scc // (op 3619) 
s_cmp_eq_i32 s19, 0 // (op 3619) 
s_cselect_b32 s49, 0, -1 // (op 3619) 

s_and_b32 s19, s14, 31 // (op 3619) 
s_lshl_b32 s50, 1, s19 // (op 3619) 
s_add_i32 s51, s50, s49 // (op 3619) 


s_and_b32 s49, s0, s51 // (op 3619) 


s_add_i32 s0, s18, s49 // (op 3619) 


s_ashr_i32 s18, s0, s19 // (op 3619) 


s_lshl_b32 s0, s14, 24 // (op 3619) 
s_ashr_i32 s19, s0, 31 // (op 3619) 

s_xor_b32 s0, s18, s19 // (op 3619) 

s_sub_i32 s18, s0, s19 // (op 3619) 


s_add_u32 s50, s6, 256 // (op 3619) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3619) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3619) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3619) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 3619) 

s_mul_i32 s0, s18, s50 // (op 3619) 

s_sub_i32 s49, s1, s0 // (op 3619) 


s_mul_hi_i32 s0, s16, s10 // (op 3619) 
s_add_i32 s1, s0, s16 // (op 3619) 

s_ashr_i32 s0, s1, 31 // (op 3619) 
s_cmp_eq_i32 s10, 0 // (op 3619) 
s_mov_b32 s18, scc // (op 3619) 
s_cmp_eq_i32 s18, 0 // (op 3619) 
s_cselect_b32 s19, 0, -1 // (op 3619) 

s_and_b32 s18, s47, 31 // (op 3619) 
s_lshl_b32 s68, 1, s18 // (op 3619) 
s_add_i32 s69, s68, s19 // (op 3619) 


s_and_b32 s19, s0, s69 // (op 3619) 


s_add_i32 s0, s1, s19 // (op 3619) 


s_ashr_i32 s1, s0, s18 // (op 3619) 


s_lshl_b32 s0, s47, 24 // (op 3619) 
s_ashr_i32 s18, s0, 31 // (op 3619) 

s_xor_b32 s0, s1, s18 // (op 3619) 

s_sub_i32 s1, s0, s18 // (op 3619) 


s_mul_i32 s0, s44, s50 // (op 3619) 

s_mul_i32 s18, s48, s0 // (op 3619) 

s_mul_i32 s0, s1, s18 // (op 3619) 

s_sub_u32 s18, s16, s0 // (op 3619) 

s_mul_hi_i32 s0, s18, s15 // (op 3619) 
s_add_i32 s19, s0, s18 // (op 3619) 


s_ashr_i32 s0, s19, 31 // (op 3619) 
s_cmp_eq_i32 s15, 0 // (op 3619) 
s_mov_b32 s18, scc // (op 3619) 
s_cmp_eq_i32 s18, 0 // (op 3619) 
s_cselect_b32 s50, 0, -1 // (op 3619) 

s_and_b32 s18, s46, 31 // (op 3619) 
s_lshl_b32 s51, 1, s18 // (op 3619) 
s_add_i32 s68, s51, s50 // (op 3619) 


s_and_b32 s50, s0, s68 // (op 3619) 


s_add_i32 s0, s19, s50 // (op 3619) 


s_ashr_i32 s19, s0, s18 // (op 3619) 


s_lshl_b32 s0, s46, 24 // (op 3619) 
s_ashr_i32 s18, s0, 31 // (op 3619) 

s_xor_b32 s0, s19, s18 // (op 3619) 

s_sub_i32 s19, s0, s18 // (op 3619) 


s_cmp_eq_i32 s1, 0 // (op 3619) 
s_mov_b32 s0, scc // (op 3619) 

s_cmp_eq_i32 s0, 0 // (op 3619) 
s_cselect_b32 s1, s19, s49 // (op 3619) 



v_lshl_add_u32 v20, s1, 8, v16 // (op 3619) 


v_mul_lo_u32 v16, v17, s38 // (op 3619) 
v_add_u32 v16, v16, v20 // (op 3619) 


s_mul_i32 s1, s38, s43 // (op 3621) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3621) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3621) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3621) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3621) most significant: sum
s_add_u32 s1, s1, s19 // (op 3621) most significant: sum


s_mov_b32 s68, s40 // (op 3621) reg expression
s_mov_b32 s69, s41 // (op 3621) reg expression
s_mov_b32 s70, s0 // (op 3621) reg expression
s_mov_b32 s71, 131072 // (op 3621) call()

s_mul_hi_i32 s0, s16, s11 // (op 3638) 
v_and_b32 v17, -32, v1 // (op 3638) 
s_add_i32 s1, s0, s16 // (op 3638) 

v_sub_u32 v20, v1, v17 // (op 3638) 

s_ashr_i32 s0, s1, 31 // (op 3638) 
v_lshlrev_b32 v17, 3, v20 // (op 3638) 

s_cmp_eq_i32 s11, 0 // (op 3638) 
s_mov_b32 s18, scc // (op 3638) 
v_lshrrev_b32 v20, 5, v1 // (op 3638) 
s_cmp_eq_i32 s18, 0 // (op 3638) 
s_cselect_b32 s19, 0, -1 // (op 3638) 

s_and_b32 s18, s12, 31 // (op 3638) 
s_lshl_b32 s49, 1, s18 // (op 3638) 
s_add_i32 s50, s49, s19 // (op 3638) 


s_and_b32 s19, s0, s50 // (op 3638) 


s_add_i32 s0, s1, s19 // (op 3638) 


s_ashr_i32 s1, s0, s18 // (op 3638) 


s_lshl_b32 s0, s12, 24 // (op 3638) 
s_ashr_i32 s18, s0, 31 // (op 3638) 

s_xor_b32 s0, s1, s18 // (op 3638) 

s_sub_i32 s1, s0, s18 // (op 3638) 


s_mul_hi_i32 s0, s1, s13 // (op 3638) 
s_add_i32 s18, s0, s1 // (op 3638) 

s_ashr_i32 s0, s18, 31 // (op 3638) 
s_cmp_eq_i32 s13, 0 // (op 3638) 
s_mov_b32 s19, scc // (op 3638) 
s_cmp_eq_i32 s19, 0 // (op 3638) 
s_cselect_b32 s49, 0, -1 // (op 3638) 

s_and_b32 s19, s14, 31 // (op 3638) 
s_lshl_b32 s50, 1, s19 // (op 3638) 
s_add_i32 s51, s50, s49 // (op 3638) 


s_and_b32 s49, s0, s51 // (op 3638) 


s_add_i32 s0, s18, s49 // (op 3638) 


s_ashr_i32 s18, s0, s19 // (op 3638) 


s_lshl_b32 s0, s14, 24 // (op 3638) 
s_ashr_i32 s19, s0, 31 // (op 3638) 

s_xor_b32 s0, s18, s19 // (op 3638) 

s_sub_i32 s18, s0, s19 // (op 3638) 


s_add_u32 s50, s6, 256 // (op 3638) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3638) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3638) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3638) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3638) 

s_mul_i32 s0, s18, s50 // (op 3638) 

s_sub_i32 s49, s1, s0 // (op 3638) 


s_mul_hi_i32 s0, s16, s10 // (op 3638) 
s_add_i32 s1, s0, s16 // (op 3638) 

s_ashr_i32 s0, s1, 31 // (op 3638) 
s_cmp_eq_i32 s10, 0 // (op 3638) 
s_mov_b32 s18, scc // (op 3638) 
s_cmp_eq_i32 s18, 0 // (op 3638) 
s_cselect_b32 s19, 0, -1 // (op 3638) 

s_and_b32 s18, s47, 31 // (op 3638) 
s_lshl_b32 s72, 1, s18 // (op 3638) 
s_add_i32 s73, s72, s19 // (op 3638) 


s_and_b32 s19, s0, s73 // (op 3638) 


s_add_i32 s0, s1, s19 // (op 3638) 


s_ashr_i32 s1, s0, s18 // (op 3638) 


s_lshl_b32 s0, s47, 24 // (op 3638) 
s_ashr_i32 s18, s0, 31 // (op 3638) 

s_xor_b32 s0, s1, s18 // (op 3638) 

s_sub_i32 s1, s0, s18 // (op 3638) 


s_mul_i32 s0, s44, s50 // (op 3638) 

s_mul_i32 s18, s48, s0 // (op 3638) 

s_mul_i32 s0, s1, s18 // (op 3638) 

s_sub_u32 s18, s16, s0 // (op 3638) 

s_mul_hi_i32 s0, s18, s15 // (op 3638) 
s_add_i32 s19, s0, s18 // (op 3638) 


s_ashr_i32 s0, s19, 31 // (op 3638) 
s_cmp_eq_i32 s15, 0 // (op 3638) 
s_mov_b32 s18, scc // (op 3638) 
s_cmp_eq_i32 s18, 0 // (op 3638) 
s_cselect_b32 s50, 0, -1 // (op 3638) 

s_and_b32 s18, s46, 31 // (op 3638) 
s_lshl_b32 s51, 1, s18 // (op 3638) 
s_add_i32 s72, s51, s50 // (op 3638) 


s_and_b32 s50, s0, s72 // (op 3638) 


s_add_i32 s0, s19, s50 // (op 3638) 


s_ashr_i32 s19, s0, s18 // (op 3638) 


s_lshl_b32 s0, s46, 24 // (op 3638) 
s_ashr_i32 s18, s0, 31 // (op 3638) 

s_xor_b32 s0, s19, s18 // (op 3638) 

s_sub_i32 s19, s0, s18 // (op 3638) 


s_cmp_eq_i32 s1, 0 // (op 3638) 
s_mov_b32 s0, scc // (op 3638) 

s_cmp_eq_i32 s0, 0 // (op 3638) 
s_cselect_b32 s1, s19, s49 // (op 3638) 



v_lshl_add_u32 v21, s1, 8, v17 // (op 3638) 


v_mul_lo_u32 v17, v20, s38 // (op 3638) 
v_add_u32 v17, v17, v21 // (op 3638) 


s_mul_i32 s1, s38, s43 // (op 3640) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3640) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3640) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3640) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3640) most significant: sum
s_add_u32 s1, s1, s19 // (op 3640) most significant: sum


s_mov_b32 s68, s40 // (op 3640) reg expression
s_mov_b32 s69, s41 // (op 3640) reg expression
s_mov_b32 s70, s0 // (op 3640) reg expression
s_mov_b32 s71, 131072 // (op 3640) call()




buffer_load_ubyte v20, v18, s[64:67], 0 offen offset: 0 // (op 3108) Load value

s_lshl_b64 s[0:1], s[28:29], 2 // (op 3112) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 3112) 


v_mov_b32 v21, v14 // (op 3112) reg expression
buffer_load_dwordx4 v[24:27], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[28:31], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[32:35], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[36:39], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[40:43], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[44:47], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[48:51], v21, s[52:55], 0 offen // (op 3112) Load value
v_add_u32 v21, v21, v22 // (op 3112) 
buffer_load_dwordx4 v[52:55], v21, s[52:55], 0 offen // (op 3112) Load value




v_mov_b32 v21, v16 // (op 3115) reg expression

buffer_load_dwordx2 v[22:23], v21, s[68:71], 0 offen // (op 3115) Load value




buffer_load_dwordx4 v[56:59], v12, s[20:23], 0 offen // (op 3118) Load value



buffer_load_ubyte v21, v18, s[64:67], 0 offen offset:4 // (op 3122) Load value


s_lshl_b64 s[0:1], s[28:29], 2 // (op 3126) 
v_lshrrev_b64 v[60:61], 3, s[0:1] // (op 3126) 


s_lshl_b64 s[0:1], s[28:29], 7 // (op 3126) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 3126) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 3126) 

v_add_u32 v18, v14, s0 // (op 3126) 

buffer_load_dwordx4 v[62:65], v18, s[52:55], 0 offen // (op 3126) Load value
v_add_u32 v18, v18, v60 // (op 3126) 
buffer_load_dwordx4 v[66:69], v18, s[52:55], 0 offen // (op 3126) Load value
v_add_u32 v18, v18, v60 // (op 3126) 
buffer_load_dwordx4 v[70:73], v18, s[52:55], 0 offen // (op 3126) Load value
v_add_u32 v18, v18, v60 // (op 3126) 
buffer_load_dwordx4 v[74:77], v18, s[52:55], 0 offen // (op 3126) Load value
v_add_u32 v18, v18, v60 // (op 3126) 
buffer_load_dwordx4 v[78:81], v18, s[52:55], 0 offen // (op 3126) Load value
v_add_u32 v18, v18, v60 // (op 3126) 
buffer_load_dwordx4 v[82:85], v18, s[52:55], 0 offen // (op 3126) Load value
v_add_u32 v18, v18, v60 // (op 3126) 
buffer_load_dwordx4 v[86:89], v18, s[52:55], 0 offen // (op 3126) Load value
v_add_u32 v18, v18, v60 // (op 3126) 
buffer_load_dwordx4 v[90:93], v18, s[52:55], 0 offen // (op 3126) Load value





s_lshl_b32 s0, s38, 2 // (op 3129) 
v_add_u32 v14, v16, s0 // (op 3129) 


buffer_load_dwordx2 v[60:61], v14, s[68:71], 0 offen // (op 3129) Load value





buffer_load_dwordx4 v[94:97], v12, s[20:23], 0 offen offset:128 // (op 3132) Load value

v_and_b32 v12, -8, v1 // (op 3758) 
v_sub_u32 v14, v1, v12 // (op 3758) 

v_lshlrev_b32 v12, 5, v14 // (op 3758) 

v_lshrrev_b32 v14, 3, v1 // (op 3758) 
v_lshl_add_u32 v16, v14, 11, v12 // (op 3758) 


v_lshlrev_b32 v12, 2, v16 // (op 3758) 

v_lshrrev_b32 v14, 3, v12 // (op 3758) 




s_waitcnt vmcnt(20)
ds_write_b128 v14, v[24:27] // (op 3134) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v14, v[28:31] offset:128 // (op 3134) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v14, v[32:35] offset:256 // (op 3134) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v14, v[36:39] offset:384 // (op 3134) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v14, v[40:43] offset:512 // (op 3134) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v14, v[44:47] offset:640 // (op 3134) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v14, v[48:51] offset:768 // (op 3134) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v14, v[52:55] offset:896 // (op 3134) Store local data 

v_and_b32 v12, -32, v1 // (op 3816) 
v_sub_u32 v14, v1, v12 // (op 3816) 

v_lshlrev_b32 v12, 3, v14 // (op 3816) 

v_lshrrev_b32 v14, 5, v1 // (op 3816) 
v_lshl_add_u32 v16, v14, 8, v12 // (op 3816) 





s_waitcnt vmcnt(12)
ds_write_b64 v16, v[22:23] offset:16384 // (op 3135) Store local data 

v_and_b32 v12, -8, v1 // (op 3705) 
v_sub_u32 v14, v1, v12 // (op 3705) 

v_lshlrev_b32 v12, 4, v14 // (op 3705) 

v_lshrrev_b32 v14, 3, v1 // (op 3705) 
v_lshl_add_u32 v16, v14, 7, v12 // (op 3705) 





s_waitcnt vmcnt(11)
ds_write_b128 v16, v[56:59] offset:17408 // (op 3136) Store local data 

v_and_b32 v12, -8, v1 // (op 3676) 
v_sub_u32 v14, v1, v12 // (op 3676) 

v_lshlrev_b32 v12, 4, v14 // (op 3676) 

v_lshrrev_b32 v14, 3, v1 // (op 3676) 
v_lshl_add_u32 v16, v14, 7, v12 // (op 3676) 


v_and_b32 v12, -8, v1 // (op 3729) 
v_sub_u32 v14, v1, v12 // (op 3729) 

v_lshlrev_b32 v12, 5, v14 // (op 3729) 

v_lshrrev_b32 v14, 3, v1 // (op 3729) 
v_lshl_add_u32 v18, v14, 11, v12 // (op 3729) 


v_lshlrev_b32 v12, 2, v18 // (op 3729) 

v_lshrrev_b32 v14, 3, v12 // (op 3729) 

v_and_b32 v12, -32, v1 // (op 3787) 
v_sub_u32 v18, v1, v12 // (op 3787) 

v_lshlrev_b32 v12, 3, v18 // (op 3787) 

v_lshrrev_b32 v18, 5, v1 // (op 3787) 
v_lshl_add_u32 v98, v18, 8, v12 // (op 3787) 


v_and_b32 v12, -8, v1 // (op 3863) 
v_sub_u32 v18, v1, v12 // (op 3863) 

v_lshlrev_b32 v12, 4, v18 // (op 3863) 

v_lshrrev_b32 v18, 3, v1 // (op 3863) 
v_lshl_add_u32 v99, v18, 7, v12 // (op 3863) 


v_and_b32 v12, -8, v1 // (op 3887) 
v_sub_u32 v18, v1, v12 // (op 3887) 

v_lshlrev_b32 v12, 5, v18 // (op 3887) 

v_lshrrev_b32 v18, 3, v1 // (op 3887) 
v_lshl_add_u32 v100, v18, 11, v12 // (op 3887) 


v_lshlrev_b32 v12, 2, v100 // (op 3887) 

v_lshrrev_b32 v18, 3, v12 // (op 3887) 

v_and_b32 v12, -32, v1 // (op 3916) 
v_sub_u32 v100, v1, v12 // (op 3916) 

v_lshlrev_b32 v12, 3, v100 // (op 3916) 

v_lshrrev_b32 v100, 5, v1 // (op 3916) 
v_lshl_add_u32 v101, v100, 8, v12 // (op 3916) 


s_mov_b32 s0, 0 // (op 1244) call()
s_ashr_i64 s[18:19], s[32:33], 7 // (op 1243) 
s_mov_b32 s1, s18 // (op 1243) convert

s_and_b32 s18, s1, 4294967294 // (op 1243) 

s_cmp_lt_u32 s0, s18 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoop_1243 // (op 1243) Condition: Top (jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoop_1243 if false)
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoop_1243:
s_add_u32 s0, s0, 2 // (op 1245) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3992) 



ds_read_b128 v[102:105], v3 offset:17408 // (op 65) Load local data 
ds_read_b128 v[106:109], v3 offset:17472 // (op 65) Load local data 
s_barrier  // (op 4005) 



ds_read_b64_tr_b4 v[110:111], v7 // (op 70) Transpose load local data 
ds_read_b64_tr_b4 v[112:113], v7 offset:2048 // (op 70) Transpose load local data 
s_barrier  // (op 4019) 


ds_read_u8 v12, v0 offset:16384 // (op 79) Load local data 



ds_read_b64_tr_b4 v[114:115], v7 offset:16 // (op 339) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[102:109], v[110:113], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 82) 
ds_read_b64_tr_b4 v[116:117], v7 offset:2064 // (op 339) Transpose load local data 




ds_read_u8 v12, v0 offset:16416 // (op 342) Load local data 



ds_read_b64_tr_b4 v[110:111], v7 offset:32 // (op 495) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:109], v[114:117], a[4:7], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 343) 
ds_read_b64_tr_b4 v[112:113], v7 offset:2080 // (op 495) Transpose load local data 




ds_read_u8 v12, v0 offset:16448 // (op 498) Load local data 



ds_read_b64_tr_b4 v[114:115], v7 offset:48 // (op 651) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[102:109], v[110:113], a[8:11], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 499) 
ds_read_b64_tr_b4 v[116:117], v7 offset:2096 // (op 651) Transpose load local data 




ds_read_u8 v12, v0 offset:16480 // (op 654) Load local data 



ds_read_b64_tr_b4 v[110:111], v7 offset:64 // (op 807) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:109], v[114:117], a[12:15], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 655) 
ds_read_b64_tr_b4 v[112:113], v7 offset:2112 // (op 807) Transpose load local data 




ds_read_u8 v12, v0 offset:16512 // (op 810) Load local data 



ds_read_b64_tr_b4 v[114:115], v7 offset:80 // (op 963) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[102:109], v[110:113], a[16:19], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 811) 
ds_read_b64_tr_b4 v[116:117], v7 offset:2128 // (op 963) Transpose load local data 




ds_read_u8 v12, v0 offset:16544 // (op 966) Load local data 



ds_read_b64_tr_b4 v[110:111], v7 offset:96 // (op 1119) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:109], v[114:117], a[20:23], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 967) 
ds_read_b64_tr_b4 v[112:113], v7 offset:2144 // (op 1119) Transpose load local data 




ds_read_u8 v12, v0 offset:16576 // (op 1122) Load local data 



ds_read_b64_tr_b4 v[114:115], v7 offset:112 // (op 1275) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[102:109], v[110:113], a[24:27], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1123) 
ds_read_b64_tr_b4 v[116:117], v7 offset:2160 // (op 1275) Transpose load local data 




ds_read_u8 v12, v0 offset:16608 // (op 1278) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[102:109], v[114:117], a[28:31], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1279) 





v_mov_b32 v12, v19 // (op 75) 

buffer_load_ubyte v20, v12, s[64:67], 0 offen offset:8 // (op 75) Load value

s_lshl_b64 s[18:19], s[28:29], 2 // (op 10) 
v_lshrrev_b64 v[102:103], 3, s[18:19] // (op 10) 


v_mov_b32 v100, v15 // (op 10) 
s_lshl_b64 s[18:19], s[28:29], 7 // (op 10) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 10) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 10) 

s_lshl_b32 s1, s18, 1 // (op 10) 

v_add_u32 v104, v100, s1 // (op 10) 

buffer_load_dwordx4 v[24:27], v104, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v104, v104, v102 // (op 10) 
buffer_load_dwordx4 v[28:31], v104, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v104, v104, v102 // (op 10) 
buffer_load_dwordx4 v[32:35], v104, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v104, v104, v102 // (op 10) 
buffer_load_dwordx4 v[36:39], v104, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v104, v104, v102 // (op 10) 
buffer_load_dwordx4 v[40:43], v104, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v104, v104, v102 // (op 10) 
buffer_load_dwordx4 v[44:47], v104, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v104, v104, v102 // (op 10) 
buffer_load_dwordx4 v[48:51], v104, s[52:55], 0 offen // (op 10) Load value
v_add_u32 v104, v104, v102 // (op 10) 
buffer_load_dwordx4 v[52:55], v104, s[52:55], 0 offen // (op 10) Load value




v_mov_b32 v102, v17 // (op 21) 
s_lshl_b32 s1, s38, 3 // (op 21) 
v_add_u32 v103, v102, s1 // (op 21) 


buffer_load_dwordx2 v[22:23], v103, s[68:71], 0 offen // (op 21) Load value



v_mov_b32 v103, v13 // (op 4) 

buffer_load_dwordx4 v[56:59], v103, s[20:23], 0 offen offset:256 // (op 4) Load value
s_barrier  // (op 4042) 



s_waitcnt vmcnt(20)
ds_write_b128 v18, v[62:65] offset:19456 // (op 235) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v18, v[66:69] offset:19584 // (op 235) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v18, v[70:73] offset:19712 // (op 235) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v18, v[74:77] offset:19840 // (op 235) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v18, v[78:81] offset:19968 // (op 235) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v18, v[82:85] offset:20096 // (op 235) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v18, v[86:89] offset:20224 // (op 235) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v18, v[90:93] offset:20352 // (op 235) Store local data 
s_barrier  // (op 4048) 



s_waitcnt vmcnt(12)
ds_write_b64 v101, v[60:61] offset:35840 // (op 236) Store local data 
s_barrier  // (op 4036) 



s_waitcnt vmcnt(11)
ds_write_b128 v99, v[94:97] offset:36864 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4033) 



ds_read_b128 v[104:107], v4 offset:36864 // (op 227) Load local data 
ds_read_b128 v[108:111], v4 offset:36928 // (op 227) Load local data 
s_barrier  // (op 4039) 
s_barrier  // (op 4085) 



ds_read_b64_tr_b4 v[112:113], v11 offset:19456 // (op 229) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v11 offset:21504 // (op 229) Transpose load local data 
s_barrier  // (op 4045) 


ds_read_u8 v116, v10 offset:35840 // (op 233) Load local data 



ds_read_b64_tr_b4 v[118:119], v11 offset:19472 // (op 348) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[112:115], a[0:3], v21, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 234) 
ds_read_b64_tr_b4 v[120:121], v11 offset:21520 // (op 348) Transpose load local data 




ds_read_u8 v112, v10 offset:35872 // (op 351) Load local data 



ds_read_b64_tr_b4 v[114:115], v11 offset:19488 // (op 504) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[118:121], a[4:7], v21, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 344) 
ds_read_b64_tr_b4 v[116:117], v11 offset:21536 // (op 504) Transpose load local data 




ds_read_u8 v112, v10 offset:35904 // (op 507) Load local data 



ds_read_b64_tr_b4 v[118:119], v11 offset:19504 // (op 660) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:111], v[114:117], a[8:11], v21, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 500) 
ds_read_b64_tr_b4 v[120:121], v11 offset:21552 // (op 660) Transpose load local data 




ds_read_u8 v112, v10 offset:35936 // (op 663) Load local data 



ds_read_b64_tr_b4 v[114:115], v11 offset:19520 // (op 816) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:111], v[118:121], a[12:15], v21, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 656) 
ds_read_b64_tr_b4 v[116:117], v11 offset:21568 // (op 816) Transpose load local data 




ds_read_u8 v112, v10 offset:35968 // (op 819) Load local data 



ds_read_b64_tr_b4 v[118:119], v11 offset:19536 // (op 972) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:111], v[114:117], a[16:19], v21, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 812) 
ds_read_b64_tr_b4 v[120:121], v11 offset:21584 // (op 972) Transpose load local data 




ds_read_u8 v112, v10 offset:36000 // (op 975) Load local data 



ds_read_b64_tr_b4 v[114:115], v11 offset:19552 // (op 1128) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:111], v[118:121], a[20:23], v21, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 968) 
ds_read_b64_tr_b4 v[116:117], v11 offset:21600 // (op 1128) Transpose load local data 




ds_read_u8 v112, v10 offset:36032 // (op 1131) Load local data 



ds_read_b64_tr_b4 v[118:119], v11 offset:19568 // (op 1284) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[104:111], v[114:117], a[24:27], v21, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1124) 
ds_read_b64_tr_b4 v[120:121], v11 offset:21616 // (op 1284) Transpose load local data 




s_barrier  // (op 4155) 
ds_read_u8 v112, v10 offset:36064 // (op 1287) Load local data 
s_barrier  // (op 4159) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:111], v[118:121], a[28:31], v21, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1280) 






buffer_load_ubyte v21, v12, s[64:67], 0 offen offset:12 // (op 231) Load value


v_add_u32 v19, 8, v19 // (op 3566) 
s_lshl_b64 s[18:19], s[28:29], 2 // (op 237) 
v_lshrrev_b64 v[104:105], 3, s[18:19] // (op 237) 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 237) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 237) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 237) 

s_mul_i32 s1, 3, s18 // (op 237) 

v_add_u32 v12, v100, s1 // (op 237) 

buffer_load_dwordx4 v[62:65], v12, s[52:55], 0 offen // (op 237) Load value
v_add_u32 v12, v12, v104 // (op 237) 
buffer_load_dwordx4 v[66:69], v12, s[52:55], 0 offen // (op 237) Load value
v_add_u32 v12, v12, v104 // (op 237) 
buffer_load_dwordx4 v[70:73], v12, s[52:55], 0 offen // (op 237) Load value
v_add_u32 v12, v12, v104 // (op 237) 
buffer_load_dwordx4 v[74:77], v12, s[52:55], 0 offen // (op 237) Load value
v_add_u32 v12, v12, v104 // (op 237) 
buffer_load_dwordx4 v[78:81], v12, s[52:55], 0 offen // (op 237) Load value
v_add_u32 v12, v12, v104 // (op 237) 
buffer_load_dwordx4 v[82:85], v12, s[52:55], 0 offen // (op 237) Load value
v_add_u32 v12, v12, v104 // (op 237) 
buffer_load_dwordx4 v[86:89], v12, s[52:55], 0 offen // (op 237) Load value
v_add_u32 v12, v12, v104 // (op 237) 
buffer_load_dwordx4 v[90:93], v12, s[52:55], 0 offen // (op 237) Load value





s_lshl_b32 s1, s38, 2 // (op 238) 
s_mul_i32 s18, 3, s1 // (op 238) 

v_add_u32 v12, v102, s18 // (op 238) 


s_lshl_b64 s[18:19], s[28:29], 7 // (op 3502) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 3502) 

s_lshr_b64 s[18:19], s[50:51], 3 // (op 3502) 

s_lshl_b32 s1, s18, 1 // (op 3502) 

v_add_u32 v15, v15, s1 // (op 3502) 

buffer_load_dwordx2 v[60:61], v12, s[68:71], 0 offen // (op 238) Load value





s_lshl_b32 s1, s38, 3 // (op 3630) 
v_add_u32 v17, v17, s1 // (op 3630) 

buffer_load_dwordx4 v[94:97], v103, s[20:23], 0 offen offset:384 // (op 224) Load value

s_barrier  // (op 4008) 
v_add_u32 v13, 256, v13 // (op 3438) 



s_waitcnt vmcnt(20)
ds_write_b128 v14, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v14, v[28:31] offset:128 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v14, v[32:35] offset:256 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v14, v[36:39] offset:384 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v14, v[40:43] offset:512 // (op 13) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v14, v[44:47] offset:640 // (op 13) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v14, v[48:51] offset:768 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v14, v[52:55] offset:896 // (op 13) Store local data 
s_barrier  // (op 4022) 



s_waitcnt vmcnt(12)
ds_write_b64 v98, v[22:23] offset:16384 // (op 26) Store local data 
s_barrier  // (op 3995) 



s_waitcnt vmcnt(11)
ds_write_b128 v16, v[56:59] offset:17408 // (op 6) Store local data 
s_ashr_i64 s[18:19], s[32:33], 7 // (op 1243) 
s_mov_b32 s1, s18 // (op 1243) convert

s_and_b32 s18, s1, 4294967294 // (op 1243) 

s_cmp_lt_u32 s0, s18 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoop_1243 // (op 1243) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoop_1243 if true)
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoop_1243:















s_mul_hi_i32 s0, s16, s11 // (op 3394) 
v_lshrrev_b32 v12, 3, v1 // (op 3394) 
s_add_i32 s1, s0, s16 // (op 3394) 

v_and_b32 v13, -8, v1 // (op 3394) 
s_ashr_i32 s0, s1, 31 // (op 3394) 
v_sub_u32 v14, v1, v13 // (op 3394) 

s_cmp_eq_i32 s11, 0 // (op 3394) 
s_mov_b32 s18, scc // (op 3394) 
v_lshlrev_b32 v13, 4, v14 // (op 3394) 

s_cmp_eq_i32 s18, 0 // (op 3394) 
s_cselect_b32 s19, 0, -1 // (op 3394) 

s_and_b32 s18, s12, 31 // (op 3394) 
s_lshl_b32 s49, 1, s18 // (op 3394) 
s_add_i32 s50, s49, s19 // (op 3394) 


s_and_b32 s19, s0, s50 // (op 3394) 


s_add_i32 s0, s1, s19 // (op 3394) 


s_ashr_i32 s1, s0, s18 // (op 3394) 


s_lshl_b32 s0, s12, 24 // (op 3394) 
s_ashr_i32 s18, s0, 31 // (op 3394) 

s_xor_b32 s0, s1, s18 // (op 3394) 

s_sub_i32 s1, s0, s18 // (op 3394) 


s_mul_hi_i32 s0, s1, s13 // (op 3394) 
s_add_i32 s18, s0, s1 // (op 3394) 

s_ashr_i32 s0, s18, 31 // (op 3394) 
s_cmp_eq_i32 s13, 0 // (op 3394) 
s_mov_b32 s19, scc // (op 3394) 
s_cmp_eq_i32 s19, 0 // (op 3394) 
s_cselect_b32 s49, 0, -1 // (op 3394) 

s_and_b32 s19, s14, 31 // (op 3394) 
s_lshl_b32 s50, 1, s19 // (op 3394) 
s_add_i32 s51, s50, s49 // (op 3394) 


s_and_b32 s49, s0, s51 // (op 3394) 


s_add_i32 s0, s18, s49 // (op 3394) 


s_ashr_i32 s18, s0, s19 // (op 3394) 


s_lshl_b32 s0, s14, 24 // (op 3394) 
s_ashr_i32 s19, s0, 31 // (op 3394) 

s_xor_b32 s0, s18, s19 // (op 3394) 

s_sub_i32 s18, s0, s19 // (op 3394) 


s_mul_i32 s0, s1, s44 // (op 3394) 

s_sub_u32 s1, s16, s0 // (op 3394) 

s_mul_i32 s0, s18, s44 // (op 3394) 
s_add_u32 s0, s0, s1 // (op 3394) 


s_mul_hi_i32 s1, s16, s10 // (op 3394) 
s_add_i32 s18, s1, s16 // (op 3394) 

s_ashr_i32 s1, s18, 31 // (op 3394) 
s_cmp_eq_i32 s10, 0 // (op 3394) 
s_mov_b32 s19, scc // (op 3394) 
s_cmp_eq_i32 s19, 0 // (op 3394) 
s_cselect_b32 s49, 0, -1 // (op 3394) 

s_and_b32 s19, s47, 31 // (op 3394) 
s_lshl_b32 s50, 1, s19 // (op 3394) 
s_add_i32 s51, s50, s49 // (op 3394) 


s_and_b32 s49, s1, s51 // (op 3394) 


s_add_i32 s1, s18, s49 // (op 3394) 


s_ashr_i32 s18, s1, s19 // (op 3394) 


s_lshl_b32 s1, s47, 24 // (op 3394) 
s_ashr_i32 s19, s1, 31 // (op 3394) 

s_xor_b32 s1, s18, s19 // (op 3394) 

s_sub_i32 s18, s1, s19 // (op 3394) 


s_add_u32 s50, s6, 256 // (op 3394) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3394) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3394) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3394) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3394) 

s_mul_i32 s1, s44, s50 // (op 3394) 

s_mul_i32 s19, s48, s1 // (op 3394) 

s_mul_i32 s1, s18, s19 // (op 3394) 

s_sub_u32 s19, s16, s1 // (op 3394) 

s_mov_b32 s1, s19 // (op 3394) 
s_mul_hi_i32 s49, s1, s15 // (op 3394) 
s_add_i32 s50, s49, s1 // (op 3394) 


s_ashr_i32 s1, s50, 31 // (op 3394) 
s_cmp_eq_i32 s15, 0 // (op 3394) 
s_mov_b32 s49, scc // (op 3394) 
s_cmp_eq_i32 s49, 0 // (op 3394) 
s_cselect_b32 s51, 0, -1 // (op 3394) 

s_and_b32 s49, s46, 31 // (op 3394) 
s_lshl_b32 s72, 1, s49 // (op 3394) 
s_add_i32 s73, s72, s51 // (op 3394) 


s_and_b32 s51, s1, s73 // (op 3394) 


s_add_i32 s1, s50, s51 // (op 3394) 


s_ashr_i32 s50, s1, s49 // (op 3394) 


s_lshl_b32 s1, s46, 24 // (op 3394) 
s_ashr_i32 s49, s1, 31 // (op 3394) 

s_xor_b32 s1, s50, s49 // (op 3394) 

s_sub_i32 s50, s1, s49 // (op 3394) 


s_mul_i32 s1, s45, s44 // (op 3394) 
s_add_u32 s72, s8, 16 // (op 3394) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 3394) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3394) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3394) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 3394) 

s_sub_i32 s49, s72, s1 // (op 3394) 

s_mul_i32 s51, s50, s49 // (op 3394) 


s_sub_u32 s49, s19, s51 // (op 3394) 


s_add_u32 s19, s49, s1 // (op 3394) 


s_cmp_eq_i32 s18, 0 // (op 3394) 
s_mov_b32 s1, scc // (op 3394) 

s_cmp_eq_i32 s1, 0 // (op 3394) 
s_cselect_b32 s18, s19, s0 // (op 3394) 



v_lshl_add_u32 v14, s18, 4, v12 // (op 3394) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3394) 
s_lshr_b32 s18, s0, 1 // (op 3394) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 3394) 


v_mul_lo_u32 v13, v14, s24 // (op 3394) 
v_add_u32 v13, v13, v12 // (op 3394) 


s_mul_i32 s1, s24, s9 // (op 3396) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 3396) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3396) least significant
s_mul_hi_u32 s19, s24, s8 // (op 3396) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3396) most significant: sum
s_add_u32 s1, s1, s19 // (op 3396) most significant: sum


s_mov_b32 s20, s26 // (op 3396) reg expression
s_mov_b32 s21, s27 // (op 3396) reg expression
s_mov_b32 s22, s0 // (op 3396) reg expression
s_mov_b32 s23, 131072 // (op 3396) call()

s_mul_hi_i32 s0, s16, s11 // (op 3468) 
v_and_b32 v12, -8, v1 // (op 3468) 
s_add_i32 s1, s0, s16 // (op 3468) 

v_sub_u32 v14, v1, v12 // (op 3468) 

s_ashr_i32 s0, s1, 31 // (op 3468) 
v_lshlrev_b32 v12, 5, v14 // (op 3468) 

s_cmp_eq_i32 s11, 0 // (op 3468) 
s_mov_b32 s18, scc // (op 3468) 
v_and_b32 v14, 4294967288, v1 // (op 3468) 
s_cmp_eq_i32 s18, 0 // (op 3468) 
s_cselect_b32 s19, 0, -1 // (op 3468) 

s_and_b32 s18, s12, 31 // (op 3468) 
s_lshl_b32 s49, 1, s18 // (op 3468) 
s_add_i32 s50, s49, s19 // (op 3468) 


s_and_b32 s19, s0, s50 // (op 3468) 


s_add_i32 s0, s1, s19 // (op 3468) 


s_ashr_i32 s1, s0, s18 // (op 3468) 


s_lshl_b32 s0, s12, 24 // (op 3468) 
s_ashr_i32 s18, s0, 31 // (op 3468) 

s_xor_b32 s0, s1, s18 // (op 3468) 

s_sub_i32 s1, s0, s18 // (op 3468) 


s_mul_hi_i32 s0, s1, s13 // (op 3468) 
s_add_i32 s18, s0, s1 // (op 3468) 

s_ashr_i32 s0, s18, 31 // (op 3468) 
s_cmp_eq_i32 s13, 0 // (op 3468) 
s_mov_b32 s19, scc // (op 3468) 
s_cmp_eq_i32 s19, 0 // (op 3468) 
s_cselect_b32 s49, 0, -1 // (op 3468) 

s_and_b32 s19, s14, 31 // (op 3468) 
s_lshl_b32 s50, 1, s19 // (op 3468) 
s_add_i32 s51, s50, s49 // (op 3468) 


s_and_b32 s49, s0, s51 // (op 3468) 


s_add_i32 s0, s18, s49 // (op 3468) 


s_ashr_i32 s18, s0, s19 // (op 3468) 


s_lshl_b32 s0, s14, 24 // (op 3468) 
s_ashr_i32 s19, s0, 31 // (op 3468) 

s_xor_b32 s0, s18, s19 // (op 3468) 

s_sub_i32 s18, s0, s19 // (op 3468) 


s_add_u32 s50, s6, 256 // (op 3468) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3468) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3468) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3468) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3468) 

s_mul_i32 s0, s18, s50 // (op 3468) 

s_sub_i32 s49, s1, s0 // (op 3468) 


s_mul_hi_i32 s0, s16, s10 // (op 3468) 
s_add_i32 s1, s0, s16 // (op 3468) 

s_ashr_i32 s0, s1, 31 // (op 3468) 
s_cmp_eq_i32 s10, 0 // (op 3468) 
s_mov_b32 s18, scc // (op 3468) 
s_cmp_eq_i32 s18, 0 // (op 3468) 
s_cselect_b32 s19, 0, -1 // (op 3468) 

s_and_b32 s18, s47, 31 // (op 3468) 
s_lshl_b32 s72, 1, s18 // (op 3468) 
s_add_i32 s73, s72, s19 // (op 3468) 


s_and_b32 s19, s0, s73 // (op 3468) 


s_add_i32 s0, s1, s19 // (op 3468) 


s_ashr_i32 s1, s0, s18 // (op 3468) 


s_lshl_b32 s0, s47, 24 // (op 3468) 
s_ashr_i32 s18, s0, 31 // (op 3468) 

s_xor_b32 s0, s1, s18 // (op 3468) 

s_sub_i32 s1, s0, s18 // (op 3468) 


s_mul_i32 s0, s44, s50 // (op 3468) 

s_mul_i32 s18, s48, s0 // (op 3468) 

s_mul_i32 s0, s1, s18 // (op 3468) 

s_sub_u32 s18, s16, s0 // (op 3468) 

s_mul_hi_i32 s0, s18, s15 // (op 3468) 
s_add_i32 s19, s0, s18 // (op 3468) 


s_ashr_i32 s0, s19, 31 // (op 3468) 
s_cmp_eq_i32 s15, 0 // (op 3468) 
s_mov_b32 s18, scc // (op 3468) 
s_cmp_eq_i32 s18, 0 // (op 3468) 
s_cselect_b32 s50, 0, -1 // (op 3468) 

s_and_b32 s18, s46, 31 // (op 3468) 
s_lshl_b32 s51, 1, s18 // (op 3468) 
s_add_i32 s72, s51, s50 // (op 3468) 


s_and_b32 s50, s0, s72 // (op 3468) 


s_add_i32 s0, s19, s50 // (op 3468) 


s_ashr_i32 s19, s0, s18 // (op 3468) 


s_lshl_b32 s0, s46, 24 // (op 3468) 
s_ashr_i32 s18, s0, 31 // (op 3468) 

s_xor_b32 s0, s19, s18 // (op 3468) 

s_sub_i32 s19, s0, s18 // (op 3468) 


s_cmp_eq_i32 s1, 0 // (op 3468) 
s_mov_b32 s0, scc // (op 3468) 

s_cmp_eq_i32 s0, 0 // (op 3468) 
s_cselect_b32 s1, s19, s49 // (op 3468) 



v_lshl_add_u32 v15, s1, 8, v12 // (op 3468) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3468) 
v_mov_b32 v17, 0 // (op 3468) convert
v_mov_b32 v16, v15 // (op 3468) convert

s_lshr_b32 s18, s0, 1 // (op 3468) 

v_lshl_add_u32 v12, s18, 8, v14 // (op 3468) 


v_mul_lo_u32 v15, v12, s29 // (op 3468) most significant: low of low * high
v_mul_hi_u32 v18, v12, s28 // (op 3468) most significant: high of low * low
v_mul_lo_u32 v14, v12, s28 // (op 3468) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 3468) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v16 // (op 3468) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 3468) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 3468) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3468) 

v_mov_b32 v12, v14 // (op 3468) convert

s_mul_i32 s1, s28, s33 // (op 3470) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 3470) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3470) least significant
s_mul_hi_u32 s19, s28, s32 // (op 3470) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3470) most significant: sum
s_add_u32 s1, s1, s19 // (op 3470) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3470) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3470) 

s_mov_b32 s52, s30 // (op 3470) reg expression
s_mov_b32 s53, s31 // (op 3470) reg expression
s_mov_b32 s54, s0 // (op 3470) reg expression
s_mov_b32 s55, 131072 // (op 3470) call()

s_mul_hi_i32 s0, s16, s11 // (op 3532) 
v_lshrrev_b32 v14, 7, v1 // (op 3532) 
s_add_i32 s1, s0, s16 // (op 3532) 

v_and_b32 v15, 15, v1 // (op 3532) 
s_ashr_i32 s0, s1, 31 // (op 3532) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 3532) 


s_cmp_eq_i32 s11, 0 // (op 3532) 
s_mov_b32 s18, scc // (op 3532) 
v_and_b32 v14, 63, v1 // (op 3532) 
s_cmp_eq_i32 s18, 0 // (op 3532) 
s_cselect_b32 s19, 0, -1 // (op 3532) 

v_lshrrev_b32 v15, 4, v14 // (op 3532) 

s_and_b32 s18, s12, 31 // (op 3532) 
s_lshl_b32 s49, 1, s18 // (op 3532) 
s_add_i32 s50, s49, s19 // (op 3532) 


s_and_b32 s19, s0, s50 // (op 3532) 


s_add_i32 s0, s1, s19 // (op 3532) 


s_ashr_i32 s1, s0, s18 // (op 3532) 


s_lshl_b32 s0, s12, 24 // (op 3532) 
s_ashr_i32 s18, s0, 31 // (op 3532) 

s_xor_b32 s0, s1, s18 // (op 3532) 

s_sub_i32 s1, s0, s18 // (op 3532) 


s_mul_hi_i32 s0, s1, s13 // (op 3532) 
s_add_i32 s18, s0, s1 // (op 3532) 

s_ashr_i32 s0, s18, 31 // (op 3532) 
s_cmp_eq_i32 s13, 0 // (op 3532) 
s_mov_b32 s19, scc // (op 3532) 
s_cmp_eq_i32 s19, 0 // (op 3532) 
s_cselect_b32 s49, 0, -1 // (op 3532) 

s_and_b32 s19, s14, 31 // (op 3532) 
s_lshl_b32 s50, 1, s19 // (op 3532) 
s_add_i32 s51, s50, s49 // (op 3532) 


s_and_b32 s49, s0, s51 // (op 3532) 


s_add_i32 s0, s18, s49 // (op 3532) 


s_ashr_i32 s18, s0, s19 // (op 3532) 


s_lshl_b32 s0, s14, 24 // (op 3532) 
s_ashr_i32 s19, s0, 31 // (op 3532) 

s_xor_b32 s0, s18, s19 // (op 3532) 

s_sub_i32 s18, s0, s19 // (op 3532) 


s_mul_i32 s0, s1, s44 // (op 3532) 

s_sub_u32 s1, s16, s0 // (op 3532) 

s_mul_i32 s0, s18, s44 // (op 3532) 
s_add_u32 s0, s0, s1 // (op 3532) 


s_mul_hi_i32 s1, s16, s10 // (op 3532) 
s_add_i32 s18, s1, s16 // (op 3532) 

s_ashr_i32 s1, s18, 31 // (op 3532) 
s_cmp_eq_i32 s10, 0 // (op 3532) 
s_mov_b32 s19, scc // (op 3532) 
s_cmp_eq_i32 s19, 0 // (op 3532) 
s_cselect_b32 s49, 0, -1 // (op 3532) 

s_and_b32 s19, s47, 31 // (op 3532) 
s_lshl_b32 s50, 1, s19 // (op 3532) 
s_add_i32 s51, s50, s49 // (op 3532) 


s_and_b32 s49, s1, s51 // (op 3532) 


s_add_i32 s1, s18, s49 // (op 3532) 


s_ashr_i32 s18, s1, s19 // (op 3532) 


s_lshl_b32 s1, s47, 24 // (op 3532) 
s_ashr_i32 s19, s1, 31 // (op 3532) 

s_xor_b32 s1, s18, s19 // (op 3532) 

s_sub_i32 s18, s1, s19 // (op 3532) 


s_add_u32 s50, s6, 256 // (op 3532) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3532) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3532) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3532) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3532) 

s_mul_i32 s1, s44, s50 // (op 3532) 

s_mul_i32 s19, s48, s1 // (op 3532) 

s_mul_i32 s1, s18, s19 // (op 3532) 

s_sub_u32 s19, s16, s1 // (op 3532) 

s_mov_b32 s1, s19 // (op 3532) 
s_mul_hi_i32 s49, s1, s15 // (op 3532) 
s_add_i32 s50, s49, s1 // (op 3532) 


s_ashr_i32 s1, s50, 31 // (op 3532) 
s_cmp_eq_i32 s15, 0 // (op 3532) 
s_mov_b32 s49, scc // (op 3532) 
s_cmp_eq_i32 s49, 0 // (op 3532) 
s_cselect_b32 s51, 0, -1 // (op 3532) 

s_and_b32 s49, s46, 31 // (op 3532) 
s_lshl_b32 s72, 1, s49 // (op 3532) 
s_add_i32 s73, s72, s51 // (op 3532) 


s_and_b32 s51, s1, s73 // (op 3532) 


s_add_i32 s1, s50, s51 // (op 3532) 


s_ashr_i32 s50, s1, s49 // (op 3532) 


s_lshl_b32 s1, s46, 24 // (op 3532) 
s_ashr_i32 s49, s1, 31 // (op 3532) 

s_xor_b32 s1, s50, s49 // (op 3532) 

s_sub_i32 s50, s1, s49 // (op 3532) 


s_mul_i32 s1, s45, s44 // (op 3532) 
s_add_u32 s72, s8, 16 // (op 3532) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 3532) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3532) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3532) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 4 // (op 3532) 

s_sub_i32 s49, s72, s1 // (op 3532) 

s_mul_i32 s51, s50, s49 // (op 3532) 


s_sub_u32 s49, s19, s51 // (op 3532) 


s_add_u32 s19, s49, s1 // (op 3532) 


s_cmp_eq_i32 s18, 0 // (op 3532) 
s_mov_b32 s1, scc // (op 3532) 

s_cmp_eq_i32 s1, 0 // (op 3532) 
s_cselect_b32 s18, s19, s0 // (op 3532) 



v_lshl_add_u32 v14, s18, 4, v16 // (op 3532) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3532) 
s_lshr_b32 s18, s0, 1 // (op 3532) 

v_lshl_add_u32 v16, s18, 3, v15 // (op 3532) 


v_mul_lo_u32 v15, v14, s34 // (op 3532) 
v_add_u32 v15, v15, v16 // (op 3532) 


s_mul_i32 s1, s34, s9 // (op 3534) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 3534) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3534) least significant
s_mul_hi_u32 s19, s34, s8 // (op 3534) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3534) most significant: sum
s_add_u32 s1, s1, s19 // (op 3534) most significant: sum


s_mov_b32 s64, s36 // (op 3534) reg expression
s_mov_b32 s65, s37 // (op 3534) reg expression
s_mov_b32 s66, s0 // (op 3534) reg expression
s_mov_b32 s67, 131072 // (op 3534) call()

s_mul_hi_i32 s0, s16, s11 // (op 3596) 
v_and_b32 v14, -32, v1 // (op 3596) 
s_add_i32 s1, s0, s16 // (op 3596) 

v_sub_u32 v16, v1, v14 // (op 3596) 

s_ashr_i32 s0, s1, 31 // (op 3596) 
v_lshlrev_b32 v14, 3, v16 // (op 3596) 

s_cmp_eq_i32 s11, 0 // (op 3596) 
s_mov_b32 s18, scc // (op 3596) 
v_lshrrev_b32 v16, 5, v1 // (op 3596) 
s_cmp_eq_i32 s18, 0 // (op 3596) 
s_cselect_b32 s19, 0, -1 // (op 3596) 

s_and_b32 s18, s12, 31 // (op 3596) 
s_lshl_b32 s49, 1, s18 // (op 3596) 
s_add_i32 s50, s49, s19 // (op 3596) 


s_and_b32 s19, s0, s50 // (op 3596) 


s_add_i32 s0, s1, s19 // (op 3596) 


s_ashr_i32 s1, s0, s18 // (op 3596) 


s_lshl_b32 s0, s12, 24 // (op 3596) 
s_ashr_i32 s18, s0, 31 // (op 3596) 

s_xor_b32 s0, s1, s18 // (op 3596) 

s_sub_i32 s1, s0, s18 // (op 3596) 


s_mul_hi_i32 s0, s1, s13 // (op 3596) 
s_add_i32 s18, s0, s1 // (op 3596) 

s_ashr_i32 s0, s18, 31 // (op 3596) 
s_cmp_eq_i32 s13, 0 // (op 3596) 
s_mov_b32 s19, scc // (op 3596) 
s_cmp_eq_i32 s19, 0 // (op 3596) 
s_cselect_b32 s49, 0, -1 // (op 3596) 

s_and_b32 s19, s14, 31 // (op 3596) 
s_lshl_b32 s50, 1, s19 // (op 3596) 
s_add_i32 s51, s50, s49 // (op 3596) 


s_and_b32 s49, s0, s51 // (op 3596) 


s_add_i32 s0, s18, s49 // (op 3596) 


s_ashr_i32 s18, s0, s19 // (op 3596) 


s_lshl_b32 s0, s14, 24 // (op 3596) 
s_ashr_i32 s19, s0, 31 // (op 3596) 

s_xor_b32 s0, s18, s19 // (op 3596) 

s_sub_i32 s18, s0, s19 // (op 3596) 


s_add_u32 s50, s6, 256 // (op 3596) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3596) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3596) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3596) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 3596) 

s_mul_i32 s0, s18, s50 // (op 3596) 

s_sub_i32 s49, s1, s0 // (op 3596) 


s_mul_hi_i32 s0, s16, s10 // (op 3596) 
s_add_i32 s1, s0, s16 // (op 3596) 

s_ashr_i32 s0, s1, 31 // (op 3596) 
s_cmp_eq_i32 s10, 0 // (op 3596) 
s_mov_b32 s18, scc // (op 3596) 
s_cmp_eq_i32 s18, 0 // (op 3596) 
s_cselect_b32 s19, 0, -1 // (op 3596) 

s_and_b32 s18, s47, 31 // (op 3596) 
s_lshl_b32 s72, 1, s18 // (op 3596) 
s_add_i32 s73, s72, s19 // (op 3596) 


s_and_b32 s19, s0, s73 // (op 3596) 


s_add_i32 s0, s1, s19 // (op 3596) 


s_ashr_i32 s1, s0, s18 // (op 3596) 


s_lshl_b32 s0, s47, 24 // (op 3596) 
s_ashr_i32 s18, s0, 31 // (op 3596) 

s_xor_b32 s0, s1, s18 // (op 3596) 

s_sub_i32 s1, s0, s18 // (op 3596) 


s_mul_i32 s0, s44, s50 // (op 3596) 

s_mul_i32 s18, s48, s0 // (op 3596) 

s_mul_i32 s0, s1, s18 // (op 3596) 

s_sub_u32 s18, s16, s0 // (op 3596) 

s_mul_hi_i32 s0, s18, s15 // (op 3596) 
s_add_i32 s19, s0, s18 // (op 3596) 


s_ashr_i32 s0, s19, 31 // (op 3596) 
s_cmp_eq_i32 s15, 0 // (op 3596) 
s_mov_b32 s18, scc // (op 3596) 
s_cmp_eq_i32 s18, 0 // (op 3596) 
s_cselect_b32 s50, 0, -1 // (op 3596) 

s_and_b32 s18, s46, 31 // (op 3596) 
s_lshl_b32 s51, 1, s18 // (op 3596) 
s_add_i32 s72, s51, s50 // (op 3596) 


s_and_b32 s50, s0, s72 // (op 3596) 


s_add_i32 s0, s19, s50 // (op 3596) 


s_ashr_i32 s19, s0, s18 // (op 3596) 


s_lshl_b32 s0, s46, 24 // (op 3596) 
s_ashr_i32 s18, s0, 31 // (op 3596) 

s_xor_b32 s0, s19, s18 // (op 3596) 

s_sub_i32 s19, s0, s18 // (op 3596) 


s_cmp_eq_i32 s1, 0 // (op 3596) 
s_mov_b32 s0, scc // (op 3596) 

s_cmp_eq_i32 s0, 0 // (op 3596) 
s_cselect_b32 s1, s19, s49 // (op 3596) 



v_lshl_add_u32 v17, s1, 8, v14 // (op 3596) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3596) 
s_lshr_b32 s18, s0, 1 // (op 3596) 

v_lshl_add_u32 v14, s18, 3, v16 // (op 3596) 


v_mul_lo_u32 v16, v14, s38 // (op 3596) 
v_add_u32 v16, v16, v17 // (op 3596) 


s_mul_i32 s1, s38, s43 // (op 3598) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3598) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3598) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3598) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3598) most significant: sum
s_add_u32 s1, s1, s19 // (op 3598) most significant: sum


s_mov_b32 s68, s40 // (op 3598) reg expression
s_mov_b32 s69, s41 // (op 3598) reg expression
s_mov_b32 s70, s0 // (op 3598) reg expression
s_mov_b32 s71, 131072 // (op 3598) call()

v_and_b32 v14, -8, v1 // (op 3689) 
v_sub_u32 v17, v1, v14 // (op 3689) 

v_lshlrev_b32 v14, 4, v17 // (op 3689) 

v_lshrrev_b32 v17, 3, v1 // (op 3689) 
v_lshl_add_u32 v18, v17, 7, v14 // (op 3689) 


v_and_b32 v14, -8, v1 // (op 3742) 
v_sub_u32 v17, v1, v14 // (op 3742) 

v_lshlrev_b32 v14, 5, v17 // (op 3742) 

v_lshrrev_b32 v17, 3, v1 // (op 3742) 
v_lshl_add_u32 v19, v17, 11, v14 // (op 3742) 


v_lshlrev_b32 v14, 2, v19 // (op 3742) 

v_lshrrev_b32 v17, 3, v14 // (op 3742) 

v_and_b32 v14, -32, v1 // (op 3800) 
v_sub_u32 v19, v1, v14 // (op 3800) 

v_lshlrev_b32 v14, 3, v19 // (op 3800) 

v_lshrrev_b32 v19, 5, v1 // (op 3800) 
v_lshl_add_u32 v20, v19, 8, v14 // (op 3800) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1249) 
s_and_b32 s18, s0, 4294967294 // (op 1249) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 1248) 
s_mov_b32 s19, s0 // (op 1248) convert

s_cmp_lt_u32 s18, s19 // (op 1248) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1248 // (op 1248) Condition: Top (jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1248 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1248:


s_add_u32 s18, s18, 1 // (op 1250) 
v_mov_b32 v14, v13 // (op 166) 

buffer_load_dwordx4 v[22:25], v14, s[20:23], 0 offen // (op 166) Load value

v_add_u32 v13, 128, v13 // (op 3384) 

s_barrier  // (op 3998) 
s_lshl_b64 s[0:1], s[28:29], 2 // (op 179) 



v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 179) 


v_mov_b32 v14, v12 // (op 179) 
v_mov_b32 v19, v14 // (op 179) reg expression
buffer_load_dwordx4 v[28:31], v19, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v19, v19, v26 // (op 179) 
buffer_load_dwordx4 v[32:35], v19, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v19, v19, v26 // (op 179) 
s_waitcnt vmcnt(2)
ds_write_b128 v18, v[22:25] offset:17408 // (op 167) Store local data 
buffer_load_dwordx4 v[36:39], v19, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v19, v19, v26 // (op 179) 
buffer_load_dwordx4 v[40:43], v19, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v19, v19, v26 // (op 179) 
buffer_load_dwordx4 v[44:47], v19, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v19, v19, v26 // (op 179) 
buffer_load_dwordx4 v[48:51], v19, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v19, v19, v26 // (op 179) 
buffer_load_dwordx4 v[52:55], v19, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v19, v19, v26 // (op 179) 
buffer_load_dwordx4 v[56:59], v19, s[52:55], 0 offen // (op 179) Load value



s_lshl_b64 s[0:1], s[28:29], 7 // (op 3460) 
s_lshl_b64 s[50:51], s[0:1], 2 // (op 3460) 

s_lshr_b64 s[0:1], s[50:51], 3 // (op 3460) 

v_add_u32 v12, v12, s0 // (op 3460) 



s_barrier  // (op 4011) 
v_mov_b32 v14, v16 // (op 180) 



v_mov_b32 v19, v14 // (op 180) reg expression

s_waitcnt vmcnt(7)
ds_write_b128 v17, v[28:31] // (op 177) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v17, v[32:35] offset:128 // (op 177) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v17, v[36:39] offset:256 // (op 177) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v17, v[40:43] offset:384 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v17, v[44:47] offset:512 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[48:51] offset:640 // (op 177) Store local data 
buffer_load_dwordx2 v[26:27], v19, s[68:71], 0 offen // (op 180) Load value


s_lshl_b32 s19, s38, 2 // (op 3588) 
v_add_u32 v16, v16, s19 // (op 3588) 

s_waitcnt vmcnt(2)
ds_write_b128 v17, v[52:55] offset:768 // (op 177) Store local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 4001) 
s_waitcnt vmcnt(1)
ds_write_b128 v17, v[56:59] offset:896 // (op 177) Store local data 




ds_read_b128 v[28:31], v3 offset:17408 // (op 169) Load local data 
s_barrier  // (op 4025) 
ds_read_b128 v[32:35], v3 offset:17472 // (op 169) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 4015) 
s_waitcnt vmcnt(0)
ds_write_b64 v20, v[26:27] offset:16384 // (op 178) Store local data 




ds_read_b64_tr_b4 v[24:25], v7 // (op 171) Transpose load local data 
ds_read_b64_tr_b4 v[26:27], v7 offset:2048 // (op 171) Transpose load local data 


v_mov_b32 v14, v15 // (op 173) 

buffer_load_ubyte v19, v14, s[64:67], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v15, 4, v15 // (op 3524) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4029) 


ds_read_u8 v14, v0 offset:16384 // (op 175) Load local data 



ds_read_b64_tr_b4 v[36:37], v7 offset:16 // (op 320) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[24:27], a[0:3], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 176) 
ds_read_b64_tr_b4 v[38:39], v7 offset:2064 // (op 320) Transpose load local data 




ds_read_u8 v14, v0 offset:16416 // (op 323) Load local data 



ds_read_b64_tr_b4 v[24:25], v7 offset:32 // (op 476) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:35], v[36:39], a[4:7], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 324) 
ds_read_b64_tr_b4 v[26:27], v7 offset:2080 // (op 476) Transpose load local data 




ds_read_u8 v14, v0 offset:16448 // (op 479) Load local data 



ds_read_b64_tr_b4 v[36:37], v7 offset:48 // (op 632) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:35], v[24:27], a[8:11], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 480) 
ds_read_b64_tr_b4 v[38:39], v7 offset:2096 // (op 632) Transpose load local data 




ds_read_u8 v14, v0 offset:16480 // (op 635) Load local data 



ds_read_b64_tr_b4 v[24:25], v7 offset:64 // (op 788) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[28:35], v[36:39], a[12:15], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 636) 
ds_read_b64_tr_b4 v[26:27], v7 offset:2112 // (op 788) Transpose load local data 




ds_read_u8 v14, v0 offset:16512 // (op 791) Load local data 



ds_read_b64_tr_b4 v[36:37], v7 offset:80 // (op 944) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[28:35], v[24:27], a[16:19], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 792) 
ds_read_b64_tr_b4 v[38:39], v7 offset:2128 // (op 944) Transpose load local data 




ds_read_u8 v14, v0 offset:16544 // (op 947) Load local data 



ds_read_b64_tr_b4 v[24:25], v7 offset:96 // (op 1100) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[28:35], v[36:39], a[20:23], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 948) 
ds_read_b64_tr_b4 v[26:27], v7 offset:2144 // (op 1100) Transpose load local data 




ds_read_u8 v14, v0 offset:16576 // (op 1103) Load local data 



ds_read_b64_tr_b4 v[36:37], v7 offset:112 // (op 1256) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[28:35], v[24:27], a[24:27], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1104) 
ds_read_b64_tr_b4 v[38:39], v7 offset:2160 // (op 1256) Transpose load local data 




ds_read_u8 v14, v0 offset:16608 // (op 1259) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[28:35], v[36:39], a[28:31], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1260) 




s_ashr_i64 s[0:1], s[32:33], 7 // (op 1248) 
s_mov_b32 s19, s0 // (op 1248) convert

s_cmp_lt_u32 s18, s19 // (op 1248) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1248 // (op 1248) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1248 if true)
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1248:






















s_barrier  // (op 4143) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 



v_mov_b32 v0, v9 // (op 52) reg expression

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 52) Store value


v_accvgpr_read v10, a4 // (op 1572) 
v_accvgpr_read v11, a5 // (op 1572) 
v_accvgpr_read v12, a6 // (op 1572) 
v_accvgpr_read v13, a7 // (op 1572) 
v_mul_f32 v14, v8, v10 // (op 1572) 
v_mul_f32 v15, v8, v11 // (op 1572) 
v_mul_f32 v16, v8, v12 // (op 1572) 
v_mul_f32 v17, v8, v13 // (op 1572) 



s_lshl_b32 s0, s4, 7 // (op 1574) 
v_add_u32 v0, v9, s0 // (op 1574) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1574) Store value


v_accvgpr_read v10, a8 // (op 1584) 
v_accvgpr_read v11, a9 // (op 1584) 
v_accvgpr_read v12, a10 // (op 1584) 
v_accvgpr_read v13, a11 // (op 1584) 
v_mul_f32 v14, v8, v10 // (op 1584) 
v_mul_f32 v15, v8, v11 // (op 1584) 
v_mul_f32 v16, v8, v12 // (op 1584) 
v_mul_f32 v17, v8, v13 // (op 1584) 



s_lshl_b32 s0, s4, 8 // (op 1586) 
v_add_u32 v0, v9, s0 // (op 1586) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1586) Store value


v_accvgpr_read v10, a12 // (op 1596) 
v_accvgpr_read v11, a13 // (op 1596) 
v_accvgpr_read v12, a14 // (op 1596) 
v_accvgpr_read v13, a15 // (op 1596) 
v_mul_f32 v14, v8, v10 // (op 1596) 
v_mul_f32 v15, v8, v11 // (op 1596) 
v_mul_f32 v16, v8, v12 // (op 1596) 
v_mul_f32 v17, v8, v13 // (op 1596) 



s_lshl_b32 s0, s4, 7 // (op 1598) 
s_mul_i32 s1, 3, s0 // (op 1598) 

v_add_u32 v0, v9, s1 // (op 1598) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1598) Store value


v_accvgpr_read v10, a16 // (op 1608) 
v_accvgpr_read v11, a17 // (op 1608) 
v_accvgpr_read v12, a18 // (op 1608) 
v_accvgpr_read v13, a19 // (op 1608) 
v_mul_f32 v14, v8, v10 // (op 1608) 
v_mul_f32 v15, v8, v11 // (op 1608) 
v_mul_f32 v16, v8, v12 // (op 1608) 
v_mul_f32 v17, v8, v13 // (op 1608) 



s_lshl_b32 s0, s4, 9 // (op 1610) 
v_add_u32 v0, v9, s0 // (op 1610) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1610) Store value


v_accvgpr_read v10, a20 // (op 1620) 
v_accvgpr_read v11, a21 // (op 1620) 
v_accvgpr_read v12, a22 // (op 1620) 
v_accvgpr_read v13, a23 // (op 1620) 
v_mul_f32 v14, v8, v10 // (op 1620) 
v_mul_f32 v15, v8, v11 // (op 1620) 
v_mul_f32 v16, v8, v12 // (op 1620) 
v_mul_f32 v17, v8, v13 // (op 1620) 



s_lshl_b32 s0, s4, 7 // (op 1622) 
s_mul_i32 s1, 5, s0 // (op 1622) 

v_add_u32 v0, v9, s1 // (op 1622) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1622) Store value


v_accvgpr_read v10, a24 // (op 1632) 
v_accvgpr_read v11, a25 // (op 1632) 
v_accvgpr_read v12, a26 // (op 1632) 
v_accvgpr_read v13, a27 // (op 1632) 
v_mul_f32 v14, v8, v10 // (op 1632) 
v_mul_f32 v15, v8, v11 // (op 1632) 
v_mul_f32 v16, v8, v12 // (op 1632) 
v_mul_f32 v17, v8, v13 // (op 1632) 



s_lshl_b32 s0, s4, 7 // (op 1634) 
s_mul_i32 s1, 6, s0 // (op 1634) 

v_add_u32 v0, v9, s1 // (op 1634) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1634) Store value


v_accvgpr_read v10, a28 // (op 1644) 
v_accvgpr_read v11, a29 // (op 1644) 
v_accvgpr_read v12, a30 // (op 1644) 
v_accvgpr_read v13, a31 // (op 1644) 
v_mul_f32 v14, v8, v10 // (op 1644) 
v_mul_f32 v15, v8, v11 // (op 1644) 
v_mul_f32 v16, v8, v12 // (op 1644) 
v_mul_f32 v17, v8, v13 // (op 1644) 



s_lshl_b32 s0, s4, 7 // (op 1646) 
s_mul_i32 s1, 7, s0 // (op 1646) 

v_add_u32 v0, v9, s1 // (op 1646) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 1646) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalFalse_0_beta_123:


v_mov_b32 v0, v6 // (op 135) reg expression

buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 135) Load value

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




v_mov_b32 v0, v9 // (op 134) reg expression

buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 134) Store value




s_lshl_b32 s0, s2, 7 // (op 1698) 
v_add_u32 v0, v6, s0 // (op 1698) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1698) Load value

v_accvgpr_read v14, a4 // (op 1695) 
v_accvgpr_read v15, a5 // (op 1695) 
v_accvgpr_read v16, a6 // (op 1695) 
v_accvgpr_read v17, a7 // (op 1695) 
v_mul_f32 v18, v8, v14 // (op 1695) 
v_mul_f32 v19, v8, v15 // (op 1695) 
v_mul_f32 v20, v8, v16 // (op 1695) 
v_mul_f32 v21, v8, v17 // (op 1695) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1699) 
v_mul_f32 v15, v5, v11 // (op 1699) 
v_mul_f32 v16, v5, v12 // (op 1699) 
v_mul_f32 v17, v5, v13 // (op 1699) 

v_add_f32 v10, v14, v18 // (op 1696) 
v_add_f32 v11, v15, v19 // (op 1696) 
v_add_f32 v12, v16, v20 // (op 1696) 
v_add_f32 v13, v17, v21 // (op 1696) 




s_lshl_b32 s0, s4, 7 // (op 1697) 
v_add_u32 v0, v9, s0 // (op 1697) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1697) Store value




s_lshl_b32 s0, s2, 8 // (op 1710) 
v_add_u32 v0, v6, s0 // (op 1710) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1710) Load value

v_accvgpr_read v14, a8 // (op 1707) 
v_accvgpr_read v15, a9 // (op 1707) 
v_accvgpr_read v16, a10 // (op 1707) 
v_accvgpr_read v17, a11 // (op 1707) 
v_mul_f32 v18, v8, v14 // (op 1707) 
v_mul_f32 v19, v8, v15 // (op 1707) 
v_mul_f32 v20, v8, v16 // (op 1707) 
v_mul_f32 v21, v8, v17 // (op 1707) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1711) 
v_mul_f32 v15, v5, v11 // (op 1711) 
v_mul_f32 v16, v5, v12 // (op 1711) 
v_mul_f32 v17, v5, v13 // (op 1711) 

v_add_f32 v10, v14, v18 // (op 1708) 
v_add_f32 v11, v15, v19 // (op 1708) 
v_add_f32 v12, v16, v20 // (op 1708) 
v_add_f32 v13, v17, v21 // (op 1708) 




s_lshl_b32 s0, s4, 8 // (op 1709) 
v_add_u32 v0, v9, s0 // (op 1709) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1709) Store value




s_lshl_b32 s0, s2, 7 // (op 1722) 
s_mul_i32 s1, 3, s0 // (op 1722) 

v_add_u32 v0, v6, s1 // (op 1722) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1722) Load value

v_accvgpr_read v14, a12 // (op 1719) 
v_accvgpr_read v15, a13 // (op 1719) 
v_accvgpr_read v16, a14 // (op 1719) 
v_accvgpr_read v17, a15 // (op 1719) 
v_mul_f32 v18, v8, v14 // (op 1719) 
v_mul_f32 v19, v8, v15 // (op 1719) 
v_mul_f32 v20, v8, v16 // (op 1719) 
v_mul_f32 v21, v8, v17 // (op 1719) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1723) 
v_mul_f32 v15, v5, v11 // (op 1723) 
v_mul_f32 v16, v5, v12 // (op 1723) 
v_mul_f32 v17, v5, v13 // (op 1723) 

v_add_f32 v10, v14, v18 // (op 1720) 
v_add_f32 v11, v15, v19 // (op 1720) 
v_add_f32 v12, v16, v20 // (op 1720) 
v_add_f32 v13, v17, v21 // (op 1720) 




s_lshl_b32 s0, s4, 7 // (op 1721) 
s_mul_i32 s1, 3, s0 // (op 1721) 

v_add_u32 v0, v9, s1 // (op 1721) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1721) Store value




s_lshl_b32 s0, s2, 9 // (op 1734) 
v_add_u32 v0, v6, s0 // (op 1734) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1734) Load value

v_accvgpr_read v14, a16 // (op 1731) 
v_accvgpr_read v15, a17 // (op 1731) 
v_accvgpr_read v16, a18 // (op 1731) 
v_accvgpr_read v17, a19 // (op 1731) 
v_mul_f32 v18, v8, v14 // (op 1731) 
v_mul_f32 v19, v8, v15 // (op 1731) 
v_mul_f32 v20, v8, v16 // (op 1731) 
v_mul_f32 v21, v8, v17 // (op 1731) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1735) 
v_mul_f32 v15, v5, v11 // (op 1735) 
v_mul_f32 v16, v5, v12 // (op 1735) 
v_mul_f32 v17, v5, v13 // (op 1735) 

v_add_f32 v10, v14, v18 // (op 1732) 
v_add_f32 v11, v15, v19 // (op 1732) 
v_add_f32 v12, v16, v20 // (op 1732) 
v_add_f32 v13, v17, v21 // (op 1732) 




s_lshl_b32 s0, s4, 9 // (op 1733) 
v_add_u32 v0, v9, s0 // (op 1733) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1733) Store value




s_lshl_b32 s0, s2, 7 // (op 1746) 
s_mul_i32 s1, 5, s0 // (op 1746) 

v_add_u32 v0, v6, s1 // (op 1746) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1746) Load value

v_accvgpr_read v14, a20 // (op 1743) 
v_accvgpr_read v15, a21 // (op 1743) 
v_accvgpr_read v16, a22 // (op 1743) 
v_accvgpr_read v17, a23 // (op 1743) 
v_mul_f32 v18, v8, v14 // (op 1743) 
v_mul_f32 v19, v8, v15 // (op 1743) 
v_mul_f32 v20, v8, v16 // (op 1743) 
v_mul_f32 v21, v8, v17 // (op 1743) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1747) 
v_mul_f32 v15, v5, v11 // (op 1747) 
v_mul_f32 v16, v5, v12 // (op 1747) 
v_mul_f32 v17, v5, v13 // (op 1747) 

v_add_f32 v10, v14, v18 // (op 1744) 
v_add_f32 v11, v15, v19 // (op 1744) 
v_add_f32 v12, v16, v20 // (op 1744) 
v_add_f32 v13, v17, v21 // (op 1744) 




s_lshl_b32 s0, s4, 7 // (op 1745) 
s_mul_i32 s1, 5, s0 // (op 1745) 

v_add_u32 v0, v9, s1 // (op 1745) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1745) Store value




s_lshl_b32 s0, s2, 7 // (op 1758) 
s_mul_i32 s1, 6, s0 // (op 1758) 

v_add_u32 v0, v6, s1 // (op 1758) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1758) Load value

v_accvgpr_read v14, a24 // (op 1755) 
v_accvgpr_read v15, a25 // (op 1755) 
v_accvgpr_read v16, a26 // (op 1755) 
v_accvgpr_read v17, a27 // (op 1755) 
v_mul_f32 v18, v8, v14 // (op 1755) 
v_mul_f32 v19, v8, v15 // (op 1755) 
v_mul_f32 v20, v8, v16 // (op 1755) 
v_mul_f32 v21, v8, v17 // (op 1755) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1759) 
v_mul_f32 v15, v5, v11 // (op 1759) 
v_mul_f32 v16, v5, v12 // (op 1759) 
v_mul_f32 v17, v5, v13 // (op 1759) 

v_add_f32 v10, v14, v18 // (op 1756) 
v_add_f32 v11, v15, v19 // (op 1756) 
v_add_f32 v12, v16, v20 // (op 1756) 
v_add_f32 v13, v17, v21 // (op 1756) 




s_lshl_b32 s0, s4, 7 // (op 1757) 
s_mul_i32 s1, 6, s0 // (op 1757) 

v_add_u32 v0, v9, s1 // (op 1757) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1757) Store value




s_lshl_b32 s0, s2, 7 // (op 1770) 
s_mul_i32 s1, 7, s0 // (op 1770) 

v_add_u32 v0, v6, s1 // (op 1770) 


buffer_load_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 1770) Load value

v_accvgpr_read v14, a28 // (op 1767) 
v_accvgpr_read v15, a29 // (op 1767) 
v_accvgpr_read v16, a30 // (op 1767) 
v_accvgpr_read v17, a31 // (op 1767) 
v_mul_f32 v18, v8, v14 // (op 1767) 
v_mul_f32 v19, v8, v15 // (op 1767) 
v_mul_f32 v20, v8, v16 // (op 1767) 
v_mul_f32 v21, v8, v17 // (op 1767) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 1771) 
v_mul_f32 v15, v5, v11 // (op 1771) 
v_mul_f32 v16, v5, v12 // (op 1771) 
v_mul_f32 v17, v5, v13 // (op 1771) 

v_add_f32 v10, v14, v18 // (op 1768) 
v_add_f32 v11, v15, v19 // (op 1768) 
v_add_f32 v12, v16, v20 // (op 1768) 
v_add_f32 v13, v17, v21 // (op 1768) 




s_lshl_b32 s0, s4, 7 // (op 1769) 
s_mul_i32 s1, 7, s0 // (op 1769) 

v_add_u32 v0, v9, s1 // (op 1769) 


buffer_store_dwordx4 v[10:13], v0, s[60:63], 0 offen // (op 1769) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_
.LRR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__end:
.size RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_, .LRR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM__end-RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 156
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 38912 // lds bytes
  .amdhsa_accum_offset 124
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
    DataTypeB: F4
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
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
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: SizeFree0 },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [16, 256, 128]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 38912
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 122
    .agpr_count: 32
    .max_flat_workgroup_size: 128
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
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicMultiple_30
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
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_4
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: Subtract_24
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
      - .name: Subtract_21
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

