// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_,@function
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 3660) 
s_mul_hi_i32 s1, s16, s10 // (op 3849) 
v_and_b32 v0, 48, v1 // (op 3717) 
v_and_b32 v3, 63, v1 // (op 3717) 
v_sub_u32 v4, v3, v0 // (op 3717) 

v_lshrrev_b32 v3, 1, v4 // (op 3717) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 3717) 


v_and_b32 v0, 1, v4 // (op 3717) 

v_lshlrev_b32 v3, 3, v0 // (op 3717) 

v_lshrrev_b32 v0, 7, v1 // (op 3717) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3717) 


v_add_u32 v0, v5, v4 // (op 3717) 


v_and_b32 v3, 48, v1 // (op 3875) 
v_and_b32 v4, 63, v1 // (op 3875) 
v_sub_u32 v5, v4, v3 // (op 3875) 

v_lshrrev_b32 v4, 1, v5 // (op 3875) 
v_add_lshl_u32 v6, v3, v4, 5 // (op 3875) 


v_and_b32 v3, 1, v5 // (op 3875) 

v_lshlrev_b32 v4, 3, v3 // (op 3875) 

v_lshrrev_b32 v3, 7, v1 // (op 3875) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 3875) 


v_add_u32 v3, v6, v5 // (op 3875) 


v_lshrrev_b32 v4, 7, v1 // (op 3660) 
s_add_i32 s18, s0, s16 // (op 3660) 

v_and_b32 v5, 63, v1 // (op 3660) 
s_ashr_i32 s0, s18, 31 // (op 3660) 
v_lshrrev_b32 v6, 4, v5 // (op 3660) 
s_cmp_eq_i32 s10, 0 // (op 3660) 
s_mov_b32 s19, scc // (op 3660) 
v_lshlrev_b32 v7, 2, v6 // (op 3660) 

s_cmp_eq_i32 s19, 0 // (op 3660) 
s_cselect_b32 s51, 0, -1 // (op 3660) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 3660) 


s_and_b32 s19, s11, 31 // (op 3660) 
v_lshrrev_b32 v4, 2, v5 // (op 3660) 

v_and_b32 v5, 3, v4 // (op 3660) 

v_lshrrev_b32 v4, 6, v1 // (op 3775) 
v_and_b32 v7, 1, v4 // (op 3775) 

v_and_b32 v4, 15, v1 // (op 3775) 
v_lshl_add_u32 v8, v7, 4, v4 // (op 3775) 


v_and_b32 v4, 63, v1 // (op 3775) 
v_lshrrev_b32 v7, 4, v4 // (op 3775) 

v_lshlrev_b32 v4, 5, v7 // (op 3775) 

v_lshl_add_u32 v7, v8, 7, v4 // (op 3775) 


v_lshlrev_b32 v4, 2, v7 // (op 3775) 

v_lshrrev_b32 v7, 3, v4 // (op 3775) 

v_lshrrev_b32 v4, 6, v1 // (op 3833) 
v_and_b32 v8, 1, v4 // (op 3833) 

v_and_b32 v4, 15, v1 // (op 3833) 
v_lshl_add_u32 v9, v8, 4, v4 // (op 3833) 


v_and_b32 v4, 63, v1 // (op 3833) 
v_lshrrev_b32 v8, 4, v4 // (op 3833) 

v_lshl_add_u32 v4, v9, 2, v8 // (op 3833) 


v_lshrrev_b32 v8, 7, v1 // (op 3849) 
v_and_b32 v9, 63, v1 // (op 3849) 
v_lshrrev_b32 v10, 6, v1 // (op 3904) 
v_and_b32 v11, 1, v10 // (op 3904) 

v_and_b32 v10, 15, v1 // (op 3904) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 3904) 


v_and_b32 v10, 63, v1 // (op 3904) 
v_lshrrev_b32 v11, 4, v10 // (op 3904) 

v_lshlrev_b32 v10, 5, v11 // (op 3904) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 3904) 


v_lshlrev_b32 v10, 2, v11 // (op 3904) 

v_lshrrev_b32 v11, 3, v10 // (op 3904) 

v_lshrrev_b32 v10, 6, v1 // (op 3933) 
v_and_b32 v12, 1, v10 // (op 3933) 

v_and_b32 v10, 15, v1 // (op 3933) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 3933) 


v_and_b32 v10, 63, v1 // (op 3933) 
v_lshrrev_b32 v12, 4, v10 // (op 3933) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 3933) 


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

s_lshl_b32 s19, s11, 24 // (op 3660) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 3660) 


s_ashr_i32 s51, s19, 31 // (op 3660) 

s_xor_b32 s19, s18, s51 // (op 3660) 

s_sub_i32 s18, s19, s51 // (op 3660) 


s_mul_hi_i32 s19, s18, s12 // (op 3660) 
s_add_i32 s51, s19, s18 // (op 3660) 

s_ashr_i32 s19, s51, 31 // (op 3660) 
s_cmp_eq_i32 s12, 0 // (op 3660) 
s_mov_b32 s52, scc // (op 3660) 
s_cmp_eq_i32 s52, 0 // (op 3660) 
s_cselect_b32 s53, 0, -1 // (op 3660) 

s_and_b32 s52, s13, 31 // (op 3660) 
v_lshrrev_b32 v12, 4, v9 // (op 3849) 
s_cmp_eq_i32 s10, 0 // (op 3849) 
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


s_lshl_b32 s19, s13, 24 // (op 3660) 
s_ashr_i32 s52, s19, 31 // (op 3660) 

s_xor_b32 s19, s51, s52 // (op 3660) 

s_sub_i32 s51, s19, s52 // (op 3660) 


s_mul_i32 s19, s18, s15 // (op 3660) 
s_sub_u32 s52, s16, s19 // (op 3660) 

s_mul_i32 s19, s51, s15 // (op 3660) 
s_add_u32 s19, s19, s52 // (op 3660) 

s_mul_hi_i32 s52, s16, s9 // (op 3660) 
s_add_i32 s53, s52, s16 // (op 3660) 

s_ashr_i32 s52, s53, 31 // (op 3660) 
s_cmp_eq_i32 s9, 0 // (op 3660) 
s_mov_b32 s54, scc // (op 3660) 
s_cmp_eq_i32 s54, 0 // (op 3660) 
s_cselect_b32 s56, 0, -1 // (op 3660) 

s_and_b32 s54, s11, 31 // (op 3849) 
v_and_b32 v9, 3, v8 // (op 3849) 

s_lshl_b32 s57, 1, s54 // (op 3849) 
v_and_b32 v8, 3, v1 // (op 3849) 
s_add_i32 s58, s57, s55 // (op 3849) 


s_and_b32 s55, s1, s58 // (op 3849) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 3849) 


s_add_i32 s1, s0, s55 // (op 3849) 


v_lshrrev_b32 v8, 6, v1 // (op 3849) 
s_ashr_i32 s0, s1, s54 // (op 3849) 


v_and_b32 v9, 1, v8 // (op 3849) 

s_lshl_b32 s1, s11, 24 // (op 3849) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 3849) 


s_ashr_i32 s54, s1, 31 // (op 3849) 

s_xor_b32 s1, s0, s54 // (op 3849) 

s_sub_i32 s0, s1, s54 // (op 3849) 


s_mul_hi_i32 s1, s0, s12 // (op 3849) 
s_add_i32 s54, s1, s0 // (op 3849) 

s_ashr_i32 s1, s54, 31 // (op 3849) 
s_cmp_eq_i32 s12, 0 // (op 3849) 
s_mov_b32 s55, scc // (op 3849) 
s_cmp_eq_i32 s55, 0 // (op 3849) 
s_cselect_b32 s57, 0, -1 // (op 3849) 

s_and_b32 s55, s13, 31 // (op 3849) 
s_lshl_b32 s58, 1, s55 // (op 3849) 
s_add_i32 s59, s58, s57 // (op 3849) 


s_and_b32 s57, s1, s59 // (op 3849) 


s_add_i32 s1, s54, s57 // (op 3849) 


s_ashr_i32 s54, s1, s55 // (op 3849) 


s_lshl_b32 s1, s13, 24 // (op 3849) 
s_ashr_i32 s55, s1, 31 // (op 3849) 

s_xor_b32 s1, s54, s55 // (op 3849) 

s_sub_i32 s54, s1, s55 // (op 3849) 


s_mul_i32 s1, s0, s15 // (op 3849) 
s_sub_u32 s55, s16, s1 // (op 3849) 

s_mul_i32 s1, s54, s15 // (op 3849) 
s_add_u32 s1, s1, s55 // (op 3849) 

s_mul_hi_i32 s55, s16, s9 // (op 3849) 
s_add_i32 s57, s55, s16 // (op 3849) 

s_ashr_i32 s55, s57, 31 // (op 3849) 
s_cmp_eq_i32 s9, 0 // (op 3849) 
s_mov_b32 s58, scc // (op 3849) 
s_cmp_eq_i32 s58, 0 // (op 3849) 
s_cselect_b32 s59, 0, -1 // (op 3849) 


s_waitcnt lgkmcnt(0)
s_and_b32 s58, s46, 31 // (op 3660) 
s_lshl_b32 s60, 1, s58 // (op 3660) 
s_add_i32 s61, s60, s56 // (op 3660) 


s_and_b32 s56, s52, s61 // (op 3660) 


s_add_i32 s52, s53, s56 // (op 3660) 


s_ashr_i32 s53, s52, s58 // (op 3660) 


s_lshl_b32 s52, s46, 24 // (op 3660) 
s_ashr_i32 s56, s52, 31 // (op 3660) 

s_xor_b32 s52, s53, s56 // (op 3660) 

s_sub_i32 s53, s52, s56 // (op 3660) 


s_and_b32 s52, s46, 31 // (op 3849) 
s_lshl_b32 s56, 1, s52 // (op 3849) 
s_add_i32 s58, s56, s59 // (op 3849) 


s_and_b32 s56, s55, s58 // (op 3849) 


s_add_i32 s55, s57, s56 // (op 3849) 


s_ashr_i32 s56, s55, s52 // (op 3849) 


s_lshl_b32 s52, s46, 24 // (op 3849) 
s_ashr_i32 s55, s52, 31 // (op 3849) 

s_xor_b32 s52, s56, s55 // (op 3849) 

s_sub_i32 s56, s52, s55 // (op 3849) 


s_add_u32 s58, s6, 256 // (op 3660) Start of Int64 add, locking SCC
s_addc_u32 s59, s7, 0 // (op 3660) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 3660) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 3660) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 8 // (op 3660) 

s_mul_i32 s52, s15, s58 // (op 3660) 
s_mul_i32 s55, s8, s52 // (op 3660) 

s_mul_i32 s52, s53, s55 // (op 3660) 

s_sub_u32 s55, s16, s52 // (op 3660) 

s_mov_b32 s52, s55 // (op 3660) 
s_mul_hi_i32 s57, s52, s14 // (op 3660) 
s_add_i32 s60, s57, s52 // (op 3660) 


s_ashr_i32 s52, s60, 31 // (op 3660) 
s_cmp_eq_i32 s14, 0 // (op 3660) 
s_mov_b32 s57, scc // (op 3660) 
s_cmp_eq_i32 s57, 0 // (op 3660) 
s_cselect_b32 s61, 0, -1 // (op 3660) 

s_and_b32 s57, s48, 31 // (op 3660) 
s_lshl_b32 s62, 1, s57 // (op 3660) 
s_add_i32 s63, s62, s61 // (op 3660) 


s_and_b32 s61, s52, s63 // (op 3660) 


s_add_i32 s52, s60, s61 // (op 3660) 


s_ashr_i32 s60, s52, s57 // (op 3660) 


s_lshl_b32 s52, s48, 24 // (op 3660) 
s_ashr_i32 s57, s52, 31 // (op 3660) 

s_xor_b32 s52, s60, s57 // (op 3660) 

s_sub_i32 s60, s52, s57 // (op 3660) 


s_mul_i32 s52, s47, s15 // (op 3660) 
s_add_u32 s62, s20, 32 // (op 3660) Start of Int64 add, locking SCC
s_addc_u32 s63, s21, 0 // (op 3660) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 3660) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 3660) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 3660) 

s_sub_i32 s57, s62, s52 // (op 3660) 

s_mul_i32 s61, s60, s57 // (op 3660) 

s_sub_u32 s57, s55, s61 // (op 3660) 


s_add_u32 s55, s57, s52 // (op 3660) 


s_cmp_eq_i32 s53, 0 // (op 3660) 
s_mov_b32 s52, scc // (op 3660) 

s_cmp_eq_i32 s52, 0 // (op 3660) 
s_cselect_b32 s53, s55, s19 // (op 3660) 


s_mul_i32 s19, s51, s58 // (op 3660) 


v_lshl_add_u32 v9, s53, 5, v6 // (op 3660) 


s_sub_i32 s51, s18, s19 // (op 3660) 


s_cmp_eq_i32 s52, 0 // (op 3660) 
s_cselect_b32 s53, s60, s51 // (op 3660) 



v_lshl_add_u32 v6, s53, 8, v5 // (op 3660) 


v_mul_lo_u32 v5, v6, s2 // (op 3660) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 3660) 


s_add_u32 s18, s6, 256 // (op 3849) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 3849) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 3849) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 3849) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 8 // (op 3849) 

s_mul_i32 s55, s15, s18 // (op 3849) 
s_mul_i32 s51, s8, s55 // (op 3849) 

s_mul_i32 s55, s56, s51 // (op 3849) 

s_sub_u32 s51, s16, s55 // (op 3849) 

s_mov_b32 s55, s51 // (op 3849) 
s_mul_hi_i32 s52, s55, s14 // (op 3849) 
s_add_i32 s53, s52, s55 // (op 3849) 


s_ashr_i32 s52, s53, 31 // (op 3849) 
s_cmp_eq_i32 s14, 0 // (op 3849) 
s_mov_b32 s55, scc // (op 3849) 
s_cmp_eq_i32 s55, 0 // (op 3849) 
s_cselect_b32 s57, 0, -1 // (op 3849) 

s_and_b32 s55, s48, 31 // (op 3849) 
s_lshl_b32 s58, 1, s55 // (op 3849) 
s_add_i32 s59, s58, s57 // (op 3849) 


s_and_b32 s57, s52, s59 // (op 3849) 


s_add_i32 s52, s53, s57 // (op 3849) 


s_ashr_i32 s53, s52, s55 // (op 3849) 


s_lshl_b32 s52, s48, 24 // (op 3849) 
s_ashr_i32 s55, s52, 31 // (op 3849) 

s_xor_b32 s52, s53, s55 // (op 3849) 

s_sub_i32 s53, s52, s55 // (op 3849) 


s_mul_i32 s52, s47, s15 // (op 3849) 
s_add_u32 s58, s20, 32 // (op 3849) Start of Int64 add, locking SCC
s_addc_u32 s59, s21, 0 // (op 3849) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 3849) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 3849) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 5 // (op 3849) 

s_sub_i32 s55, s58, s52 // (op 3849) 

s_mul_i32 s57, s53, s55 // (op 3849) 

s_sub_u32 s55, s51, s57 // (op 3849) 


s_add_u32 s51, s55, s52 // (op 3849) 


s_cmp_eq_i32 s56, 0 // (op 3849) 
s_mov_b32 s52, scc // (op 3849) 

s_cmp_eq_i32 s52, 0 // (op 3849) 
s_cselect_b32 s55, s51, s1 // (op 3849) 


s_mul_i32 s1, s54, s18 // (op 3849) 


v_lshl_add_u32 v5, s55, 5, v12 // (op 3849) 


s_sub_i32 s51, s0, s1 // (op 3849) 


s_cmp_eq_i32 s52, 0 // (op 3849) 
s_cselect_b32 s0, s53, s51 // (op 3849) 



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

s_mov_b32 s56, s22 // (op 3662) reg expression
s_mov_b32 s57, s23 // (op 3662) reg expression
s_mov_b32 s58, s18 // (op 3662) reg expression
s_mov_b32 s59, 131072 // (op 3662) call()

s_mov_b32 s60, s24 // (op 3851) reg expression

s_mov_b32 s61, s25 // (op 3851) reg expression
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
s_mul_hi_i32 s0, s16, s10 // (op 3427) 
v_and_b32 v12, -2, v1 // (op 3427) 
s_add_i32 s1, s0, s16 // (op 3427) 

v_sub_u32 v13, v1, v12 // (op 3427) 

s_ashr_i32 s0, s1, 31 // (op 3427) 
v_lshlrev_b32 v12, 4, v13 // (op 3427) 

s_cmp_eq_i32 s10, 0 // (op 3427) 
s_mov_b32 s18, scc // (op 3427) 
v_lshrrev_b32 v13, 1, v1 // (op 3427) 
s_cmp_eq_i32 s18, 0 // (op 3427) 
s_cselect_b32 s19, 0, -1 // (op 3427) 

s_and_b32 s18, s11, 31 // (op 3427) 
s_lshl_b32 s22, 1, s18 // (op 3427) 
s_add_i32 s23, s22, s19 // (op 3427) 


s_and_b32 s19, s0, s23 // (op 3427) 


s_add_i32 s0, s1, s19 // (op 3427) 


s_ashr_i32 s1, s0, s18 // (op 3427) 


s_lshl_b32 s0, s11, 24 // (op 3427) 
s_ashr_i32 s18, s0, 31 // (op 3427) 

s_xor_b32 s0, s1, s18 // (op 3427) 

s_sub_i32 s1, s0, s18 // (op 3427) 


s_mul_hi_i32 s0, s1, s12 // (op 3427) 
s_add_i32 s18, s0, s1 // (op 3427) 

s_ashr_i32 s0, s18, 31 // (op 3427) 
s_cmp_eq_i32 s12, 0 // (op 3427) 
s_mov_b32 s19, scc // (op 3427) 
s_cmp_eq_i32 s19, 0 // (op 3427) 
s_cselect_b32 s22, 0, -1 // (op 3427) 

s_and_b32 s19, s13, 31 // (op 3427) 
s_lshl_b32 s23, 1, s19 // (op 3427) 
s_add_i32 s24, s23, s22 // (op 3427) 


s_and_b32 s25, s0, s24 // (op 3427) 


s_add_i32 s0, s18, s25 // (op 3427) 


s_ashr_i32 s18, s0, s19 // (op 3427) 


s_lshl_b32 s0, s13, 24 // (op 3427) 
s_ashr_i32 s19, s0, 31 // (op 3427) 

s_xor_b32 s0, s18, s19 // (op 3427) 

s_sub_i32 s18, s0, s19 // (op 3427) 


s_mul_i32 s0, s1, s15 // (op 3427) 

s_sub_u32 s1, s16, s0 // (op 3427) 

s_mul_i32 s0, s18, s15 // (op 3427) 
s_add_u32 s0, s0, s1 // (op 3427) 


s_mul_hi_i32 s1, s16, s9 // (op 3427) 
s_add_i32 s18, s1, s16 // (op 3427) 

s_ashr_i32 s1, s18, 31 // (op 3427) 
s_cmp_eq_i32 s9, 0 // (op 3427) 
s_mov_b32 s19, scc // (op 3427) 
s_cmp_eq_i32 s19, 0 // (op 3427) 
s_cselect_b32 s22, 0, -1 // (op 3427) 

s_and_b32 s19, s46, 31 // (op 3427) 
s_lshl_b32 s23, 1, s19 // (op 3427) 
s_add_i32 s24, s23, s22 // (op 3427) 


s_and_b32 s25, s1, s24 // (op 3427) 


s_add_i32 s1, s18, s25 // (op 3427) 


s_ashr_i32 s18, s1, s19 // (op 3427) 


s_lshl_b32 s1, s46, 24 // (op 3427) 
s_ashr_i32 s19, s1, 31 // (op 3427) 

s_xor_b32 s1, s18, s19 // (op 3427) 

s_sub_i32 s18, s1, s19 // (op 3427) 


s_add_u32 s22, s6, 256 // (op 3427) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 3427) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3427) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3427) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 3427) 

s_mul_i32 s1, s15, s22 // (op 3427) 

s_mul_i32 s19, s8, s1 // (op 3427) 

s_mul_i32 s1, s18, s19 // (op 3427) 

s_sub_u32 s19, s16, s1 // (op 3427) 

s_mov_b32 s1, s19 // (op 3427) 
s_mul_hi_i32 s22, s1, s14 // (op 3427) 
s_add_i32 s23, s22, s1 // (op 3427) 


s_ashr_i32 s1, s23, 31 // (op 3427) 
s_cmp_eq_i32 s14, 0 // (op 3427) 
s_mov_b32 s22, scc // (op 3427) 
s_cmp_eq_i32 s22, 0 // (op 3427) 
s_cselect_b32 s24, 0, -1 // (op 3427) 

s_and_b32 s22, s48, 31 // (op 3427) 
s_lshl_b32 s25, 1, s22 // (op 3427) 
s_add_i32 s49, s25, s24 // (op 3427) 


s_and_b32 s24, s1, s49 // (op 3427) 


s_add_i32 s1, s23, s24 // (op 3427) 


s_ashr_i32 s23, s1, s22 // (op 3427) 


s_lshl_b32 s1, s48, 24 // (op 3427) 
s_ashr_i32 s22, s1, 31 // (op 3427) 

s_xor_b32 s1, s23, s22 // (op 3427) 

s_sub_i32 s23, s1, s22 // (op 3427) 


s_mul_i32 s1, s47, s15 // (op 3427) 
s_add_u32 s24, s20, 32 // (op 3427) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3427) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 3427) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 3427) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 5 // (op 3427) 

s_sub_i32 s22, s24, s1 // (op 3427) 

s_mul_i32 s24, s23, s22 // (op 3427) 


s_sub_u32 s25, s19, s24 // (op 3427) 


s_add_u32 s19, s25, s1 // (op 3427) 


s_cmp_eq_i32 s18, 0 // (op 3427) 
s_mov_b32 s1, scc // (op 3427) 

s_cmp_eq_i32 s1, 0 // (op 3427) 
s_cselect_b32 s18, s19, s0 // (op 3427) 



v_lshl_add_u32 v14, s18, 5, v12 // (op 3427) 


v_mul_lo_u32 v12, v13, s26 // (op 3427) 

v_add_u32 v13, v14, v12 // (op 3427) 


s_mul_i32 s1, s26, s31 // (op 3429) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 3429) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 3429) least significant
s_mul_hi_u32 s19, s26, s30 // (op 3429) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3429) most significant: sum
s_add_u32 s1, s1, s19 // (op 3429) most significant: sum


s_mov_b32 s52, s28 // (op 3429) reg expression
s_mov_b32 s53, s29 // (op 3429) reg expression
s_mov_b32 s54, s0 // (op 3429) reg expression
s_mov_b32 s55, 131072 // (op 3429) call()

s_mul_hi_i32 s0, s16, s10 // (op 3446) 
v_and_b32 v12, -2, v1 // (op 3446) 
s_add_i32 s1, s0, s16 // (op 3446) 

v_sub_u32 v14, v1, v12 // (op 3446) 

s_ashr_i32 s0, s1, 31 // (op 3446) 
v_lshlrev_b32 v12, 4, v14 // (op 3446) 

s_cmp_eq_i32 s10, 0 // (op 3446) 
s_mov_b32 s18, scc // (op 3446) 
v_lshrrev_b32 v14, 1, v1 // (op 3446) 
s_cmp_eq_i32 s18, 0 // (op 3446) 
s_cselect_b32 s19, 0, -1 // (op 3446) 

s_and_b32 s18, s11, 31 // (op 3446) 
s_lshl_b32 s22, 1, s18 // (op 3446) 
s_add_i32 s23, s22, s19 // (op 3446) 


s_and_b32 s19, s0, s23 // (op 3446) 


s_add_i32 s0, s1, s19 // (op 3446) 


s_ashr_i32 s1, s0, s18 // (op 3446) 


s_lshl_b32 s0, s11, 24 // (op 3446) 
s_ashr_i32 s18, s0, 31 // (op 3446) 

s_xor_b32 s0, s1, s18 // (op 3446) 

s_sub_i32 s1, s0, s18 // (op 3446) 


s_mul_hi_i32 s0, s1, s12 // (op 3446) 
s_add_i32 s18, s0, s1 // (op 3446) 

s_ashr_i32 s0, s18, 31 // (op 3446) 
s_cmp_eq_i32 s12, 0 // (op 3446) 
s_mov_b32 s19, scc // (op 3446) 
s_cmp_eq_i32 s19, 0 // (op 3446) 
s_cselect_b32 s22, 0, -1 // (op 3446) 

s_and_b32 s19, s13, 31 // (op 3446) 
s_lshl_b32 s23, 1, s19 // (op 3446) 
s_add_i32 s24, s23, s22 // (op 3446) 


s_and_b32 s25, s0, s24 // (op 3446) 


s_add_i32 s0, s18, s25 // (op 3446) 


s_ashr_i32 s18, s0, s19 // (op 3446) 


s_lshl_b32 s0, s13, 24 // (op 3446) 
s_ashr_i32 s19, s0, 31 // (op 3446) 

s_xor_b32 s0, s18, s19 // (op 3446) 

s_sub_i32 s18, s0, s19 // (op 3446) 


s_mul_i32 s0, s1, s15 // (op 3446) 

s_sub_u32 s1, s16, s0 // (op 3446) 

s_mul_i32 s0, s18, s15 // (op 3446) 
s_add_u32 s0, s0, s1 // (op 3446) 


s_mul_hi_i32 s1, s16, s9 // (op 3446) 
s_add_i32 s18, s1, s16 // (op 3446) 

s_ashr_i32 s1, s18, 31 // (op 3446) 
s_cmp_eq_i32 s9, 0 // (op 3446) 
s_mov_b32 s19, scc // (op 3446) 
s_cmp_eq_i32 s19, 0 // (op 3446) 
s_cselect_b32 s22, 0, -1 // (op 3446) 

s_and_b32 s19, s46, 31 // (op 3446) 
s_lshl_b32 s23, 1, s19 // (op 3446) 
s_add_i32 s24, s23, s22 // (op 3446) 


s_and_b32 s25, s1, s24 // (op 3446) 


s_add_i32 s1, s18, s25 // (op 3446) 


s_ashr_i32 s18, s1, s19 // (op 3446) 


s_lshl_b32 s1, s46, 24 // (op 3446) 
s_ashr_i32 s19, s1, 31 // (op 3446) 

s_xor_b32 s1, s18, s19 // (op 3446) 

s_sub_i32 s18, s1, s19 // (op 3446) 


s_add_u32 s50, s6, 256 // (op 3446) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3446) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3446) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3446) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3446) 

s_mul_i32 s1, s15, s24 // (op 3446) 

s_mul_i32 s19, s8, s1 // (op 3446) 

s_mul_i32 s1, s18, s19 // (op 3446) 

s_sub_u32 s19, s16, s1 // (op 3446) 

s_mov_b32 s1, s19 // (op 3446) 
s_mul_hi_i32 s22, s1, s14 // (op 3446) 
s_add_i32 s23, s22, s1 // (op 3446) 


s_ashr_i32 s1, s23, 31 // (op 3446) 
s_cmp_eq_i32 s14, 0 // (op 3446) 
s_mov_b32 s22, scc // (op 3446) 
s_cmp_eq_i32 s22, 0 // (op 3446) 
s_cselect_b32 s24, 0, -1 // (op 3446) 

s_and_b32 s22, s48, 31 // (op 3446) 
s_lshl_b32 s25, 1, s22 // (op 3446) 
s_add_i32 s49, s25, s24 // (op 3446) 


s_and_b32 s24, s1, s49 // (op 3446) 


s_add_i32 s1, s23, s24 // (op 3446) 


s_ashr_i32 s23, s1, s22 // (op 3446) 


s_lshl_b32 s1, s48, 24 // (op 3446) 
s_ashr_i32 s22, s1, 31 // (op 3446) 

s_xor_b32 s1, s23, s22 // (op 3446) 

s_sub_i32 s23, s1, s22 // (op 3446) 


s_mul_i32 s1, s47, s15 // (op 3446) 
s_add_u32 s24, s20, 32 // (op 3446) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3446) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3446) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3446) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 3446) 

s_sub_i32 s22, s24, s1 // (op 3446) 

s_mul_i32 s24, s23, s22 // (op 3446) 


s_sub_u32 s25, s19, s24 // (op 3446) 


s_add_u32 s19, s25, s1 // (op 3446) 


s_cmp_eq_i32 s18, 0 // (op 3446) 
s_mov_b32 s1, scc // (op 3446) 

s_cmp_eq_i32 s1, 0 // (op 3446) 
s_cselect_b32 s18, s19, s0 // (op 3446) 



v_lshl_add_u32 v15, s18, 5, v12 // (op 3446) 


v_mul_lo_u32 v12, v14, s26 // (op 3446) 

v_add_u32 v14, v15, v12 // (op 3446) 


s_mul_i32 s1, s26, s31 // (op 3448) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 3448) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 3448) least significant
s_mul_hi_u32 s19, s26, s30 // (op 3448) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3448) most significant: sum
s_add_u32 s1, s1, s19 // (op 3448) most significant: sum


s_mov_b32 s52, s28 // (op 3448) reg expression
s_mov_b32 s53, s29 // (op 3448) reg expression
s_mov_b32 s54, s0 // (op 3448) reg expression
s_mov_b32 s55, 131072 // (op 3448) call()

s_mul_hi_i32 s0, s16, s10 // (op 3491) 
v_and_b32 v12, 4294967292, v1 // (op 3491) 
s_add_i32 s1, s0, s16 // (op 3491) 

v_and_b32 v15, -4, v1 // (op 3491) 
s_ashr_i32 s0, s1, 31 // (op 3491) 
v_sub_u32 v16, v1, v15 // (op 3491) 

s_cmp_eq_i32 s10, 0 // (op 3491) 
s_mov_b32 s18, scc // (op 3491) 
v_lshlrev_b32 v15, 5, v16 // (op 3491) 

s_cmp_eq_i32 s18, 0 // (op 3491) 
s_cselect_b32 s19, 0, -1 // (op 3491) 

v_mov_b32 v17, 0 // (op 3491) convert
v_mov_b32 v16, v15 // (op 3491) convert

s_and_b32 s18, s11, 31 // (op 3491) 
s_lshl_b32 s22, 1, s18 // (op 3491) 
s_add_i32 s23, s22, s19 // (op 3491) 


s_and_b32 s19, s0, s23 // (op 3491) 


s_add_i32 s0, s1, s19 // (op 3491) 


s_ashr_i32 s1, s0, s18 // (op 3491) 


s_lshl_b32 s0, s11, 24 // (op 3491) 
s_ashr_i32 s18, s0, 31 // (op 3491) 

s_xor_b32 s0, s1, s18 // (op 3491) 

s_sub_i32 s1, s0, s18 // (op 3491) 


s_mul_hi_i32 s0, s1, s12 // (op 3491) 
s_add_i32 s18, s0, s1 // (op 3491) 

s_ashr_i32 s0, s18, 31 // (op 3491) 
s_cmp_eq_i32 s12, 0 // (op 3491) 
s_mov_b32 s19, scc // (op 3491) 
s_cmp_eq_i32 s19, 0 // (op 3491) 
s_cselect_b32 s22, 0, -1 // (op 3491) 

s_and_b32 s19, s13, 31 // (op 3491) 
s_lshl_b32 s23, 1, s19 // (op 3491) 
s_add_i32 s24, s23, s22 // (op 3491) 


s_and_b32 s25, s0, s24 // (op 3491) 


s_add_i32 s0, s18, s25 // (op 3491) 


s_ashr_i32 s18, s0, s19 // (op 3491) 


s_lshl_b32 s0, s13, 24 // (op 3491) 
s_ashr_i32 s19, s0, 31 // (op 3491) 

s_xor_b32 s0, s18, s19 // (op 3491) 

s_sub_i32 s18, s0, s19 // (op 3491) 


s_add_u32 s50, s6, 256 // (op 3491) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3491) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3491) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3491) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3491) 

s_mul_i32 s0, s18, s24 // (op 3491) 

s_sub_i32 s18, s1, s0 // (op 3491) 


s_mul_hi_i32 s19, s16, s9 // (op 3491) 
s_add_i32 s0, s19, s16 // (op 3491) 

s_ashr_i32 s1, s0, 31 // (op 3491) 
s_cmp_eq_i32 s9, 0 // (op 3491) 
s_mov_b32 s19, scc // (op 3491) 
s_cmp_eq_i32 s19, 0 // (op 3491) 
s_cselect_b32 s22, 0, -1 // (op 3491) 

s_and_b32 s19, s46, 31 // (op 3491) 
s_lshl_b32 s23, 1, s19 // (op 3491) 
s_add_i32 s49, s23, s22 // (op 3491) 


s_and_b32 s22, s1, s49 // (op 3491) 


s_add_i32 s1, s0, s22 // (op 3491) 


s_ashr_i32 s0, s1, s19 // (op 3491) 


s_lshl_b32 s1, s46, 24 // (op 3491) 
s_ashr_i32 s19, s1, 31 // (op 3491) 

s_xor_b32 s1, s0, s19 // (op 3491) 

s_sub_i32 s0, s1, s19 // (op 3491) 


s_mul_i32 s1, s15, s24 // (op 3491) 

s_mul_i32 s19, s8, s1 // (op 3491) 

s_mul_i32 s1, s0, s19 // (op 3491) 

s_sub_u32 s19, s16, s1 // (op 3491) 

s_mul_hi_i32 s1, s19, s14 // (op 3491) 
s_add_i32 s22, s1, s19 // (op 3491) 


s_ashr_i32 s1, s22, 31 // (op 3491) 
s_cmp_eq_i32 s14, 0 // (op 3491) 
s_mov_b32 s19, scc // (op 3491) 
s_cmp_eq_i32 s19, 0 // (op 3491) 
s_cselect_b32 s23, 0, -1 // (op 3491) 

s_and_b32 s19, s48, 31 // (op 3491) 
s_lshl_b32 s24, 1, s19 // (op 3491) 
s_add_i32 s25, s24, s23 // (op 3491) 


s_and_b32 s23, s1, s25 // (op 3491) 


s_add_i32 s1, s22, s23 // (op 3491) 


s_ashr_i32 s22, s1, s19 // (op 3491) 


s_lshl_b32 s1, s48, 24 // (op 3491) 
s_ashr_i32 s19, s1, 31 // (op 3491) 

s_xor_b32 s1, s22, s19 // (op 3491) 

s_sub_i32 s22, s1, s19 // (op 3491) 


s_cmp_eq_i32 s0, 0 // (op 3491) 
s_mov_b32 s1, scc // (op 3491) 

s_cmp_eq_i32 s1, 0 // (op 3491) 
s_cselect_b32 s0, s22, s18 // (op 3491) 



v_lshl_add_u32 v15, s0, 8, v12 // (op 3491) 


v_mul_lo_u32 v19, v15, s33 // (op 3491) most significant: low of low * high
v_mul_hi_u32 v12, v15, s32 // (op 3491) most significant: high of low * low
v_mul_lo_u32 v18, v15, s32 // (op 3491) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3491) most significant: sum


v_add_co_u32 v20, s[0:1], v16, v18 // (op 3491) least significant half
v_addc_co_u32 v21, s[0:1], v17, v19, s[0:1] // (op 3491) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 3491) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 3491) 

v_mov_b32 v12, v18 // (op 3491) convert

s_mul_i32 s1, s32, s7 // (op 3493) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 3493) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 3493) least significant
s_mul_hi_u32 s19, s32, s6 // (op 3493) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3493) most significant: sum
s_add_u32 s1, s1, s19 // (op 3493) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3493) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3493) 

s_mov_b32 s64, s34 // (op 3493) reg expression
s_mov_b32 s65, s35 // (op 3493) reg expression
s_mov_b32 s66, s0 // (op 3493) reg expression
s_mov_b32 s67, 131072 // (op 3493) call()

s_mul_hi_i32 s0, s16, s10 // (op 3510) 
v_and_b32 v15, 4294967292, v1 // (op 3510) 
s_add_i32 s1, s0, s16 // (op 3510) 

v_and_b32 v16, -4, v1 // (op 3510) 
s_ashr_i32 s0, s1, 31 // (op 3510) 
v_sub_u32 v17, v1, v16 // (op 3510) 

s_cmp_eq_i32 s10, 0 // (op 3510) 
s_mov_b32 s18, scc // (op 3510) 
v_lshlrev_b32 v16, 5, v17 // (op 3510) 

s_cmp_eq_i32 s18, 0 // (op 3510) 
s_cselect_b32 s19, 0, -1 // (op 3510) 

v_mov_b32 v19, 0 // (op 3510) convert
v_mov_b32 v18, v16 // (op 3510) convert

s_and_b32 s18, s11, 31 // (op 3510) 
s_lshl_b32 s22, 1, s18 // (op 3510) 
s_add_i32 s23, s22, s19 // (op 3510) 


s_and_b32 s19, s0, s23 // (op 3510) 


s_add_i32 s0, s1, s19 // (op 3510) 


s_ashr_i32 s1, s0, s18 // (op 3510) 


s_lshl_b32 s0, s11, 24 // (op 3510) 
s_ashr_i32 s18, s0, 31 // (op 3510) 

s_xor_b32 s0, s1, s18 // (op 3510) 

s_sub_i32 s1, s0, s18 // (op 3510) 


s_mul_hi_i32 s0, s1, s12 // (op 3510) 
s_add_i32 s18, s0, s1 // (op 3510) 

s_ashr_i32 s0, s18, 31 // (op 3510) 
s_cmp_eq_i32 s12, 0 // (op 3510) 
s_mov_b32 s19, scc // (op 3510) 
s_cmp_eq_i32 s19, 0 // (op 3510) 
s_cselect_b32 s22, 0, -1 // (op 3510) 

s_and_b32 s19, s13, 31 // (op 3510) 
s_lshl_b32 s23, 1, s19 // (op 3510) 
s_add_i32 s24, s23, s22 // (op 3510) 


s_and_b32 s25, s0, s24 // (op 3510) 


s_add_i32 s0, s18, s25 // (op 3510) 


s_ashr_i32 s18, s0, s19 // (op 3510) 


s_lshl_b32 s0, s13, 24 // (op 3510) 
s_ashr_i32 s19, s0, 31 // (op 3510) 

s_xor_b32 s0, s18, s19 // (op 3510) 

s_sub_i32 s18, s0, s19 // (op 3510) 


s_add_u32 s50, s6, 256 // (op 3510) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3510) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3510) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3510) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3510) 

s_mul_i32 s0, s18, s24 // (op 3510) 

s_sub_i32 s18, s1, s0 // (op 3510) 


s_mul_hi_i32 s19, s16, s9 // (op 3510) 
s_add_i32 s0, s19, s16 // (op 3510) 

s_ashr_i32 s1, s0, 31 // (op 3510) 
s_cmp_eq_i32 s9, 0 // (op 3510) 
s_mov_b32 s19, scc // (op 3510) 
s_cmp_eq_i32 s19, 0 // (op 3510) 
s_cselect_b32 s22, 0, -1 // (op 3510) 

s_and_b32 s19, s46, 31 // (op 3510) 
s_lshl_b32 s23, 1, s19 // (op 3510) 
s_add_i32 s49, s23, s22 // (op 3510) 


s_and_b32 s22, s1, s49 // (op 3510) 


s_add_i32 s1, s0, s22 // (op 3510) 


s_ashr_i32 s0, s1, s19 // (op 3510) 


s_lshl_b32 s1, s46, 24 // (op 3510) 
s_ashr_i32 s19, s1, 31 // (op 3510) 

s_xor_b32 s1, s0, s19 // (op 3510) 

s_sub_i32 s0, s1, s19 // (op 3510) 


s_mul_i32 s1, s15, s24 // (op 3510) 

s_mul_i32 s19, s8, s1 // (op 3510) 

s_mul_i32 s1, s0, s19 // (op 3510) 

s_sub_u32 s19, s16, s1 // (op 3510) 

s_mul_hi_i32 s1, s19, s14 // (op 3510) 
s_add_i32 s22, s1, s19 // (op 3510) 


s_ashr_i32 s1, s22, 31 // (op 3510) 
s_cmp_eq_i32 s14, 0 // (op 3510) 
s_mov_b32 s19, scc // (op 3510) 
s_cmp_eq_i32 s19, 0 // (op 3510) 
s_cselect_b32 s23, 0, -1 // (op 3510) 

s_and_b32 s19, s48, 31 // (op 3510) 
s_lshl_b32 s24, 1, s19 // (op 3510) 
s_add_i32 s25, s24, s23 // (op 3510) 


s_and_b32 s23, s1, s25 // (op 3510) 


s_add_i32 s1, s22, s23 // (op 3510) 


s_ashr_i32 s22, s1, s19 // (op 3510) 


s_lshl_b32 s1, s48, 24 // (op 3510) 
s_ashr_i32 s19, s1, 31 // (op 3510) 

s_xor_b32 s1, s22, s19 // (op 3510) 

s_sub_i32 s22, s1, s19 // (op 3510) 


s_cmp_eq_i32 s0, 0 // (op 3510) 
s_mov_b32 s1, scc // (op 3510) 

s_cmp_eq_i32 s1, 0 // (op 3510) 
s_cselect_b32 s0, s22, s18 // (op 3510) 



v_lshl_add_u32 v16, s0, 8, v15 // (op 3510) 


v_mul_lo_u32 v21, v16, s33 // (op 3510) most significant: low of low * high
v_mul_hi_u32 v15, v16, s32 // (op 3510) most significant: high of low * low
v_mul_lo_u32 v20, v16, s32 // (op 3510) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 3510) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 3510) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 3510) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 3510) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 3510) 

v_mov_b32 v15, v18 // (op 3510) convert

s_mul_i32 s1, s32, s7 // (op 3512) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 3512) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 3512) least significant
s_mul_hi_u32 s19, s32, s6 // (op 3512) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3512) most significant: sum
s_add_u32 s1, s1, s19 // (op 3512) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3512) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3512) 

s_mov_b32 s64, s34 // (op 3512) reg expression
s_mov_b32 s65, s35 // (op 3512) reg expression
s_mov_b32 s66, s0 // (op 3512) reg expression
s_mov_b32 s67, 131072 // (op 3512) call()

s_mul_hi_i32 s0, s16, s10 // (op 3555) 
v_lshrrev_b32 v16, 7, v1 // (op 3555) 
s_add_i32 s1, s0, s16 // (op 3555) 

v_and_b32 v17, 15, v1 // (op 3555) 
s_ashr_i32 s0, s1, 31 // (op 3555) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 3555) 


s_cmp_eq_i32 s10, 0 // (op 3555) 
s_mov_b32 s18, scc // (op 3555) 
v_and_b32 v16, 63, v1 // (op 3555) 
s_cmp_eq_i32 s18, 0 // (op 3555) 
s_cselect_b32 s19, 0, -1 // (op 3555) 

v_lshrrev_b32 v17, 4, v16 // (op 3555) 

s_and_b32 s18, s11, 31 // (op 3555) 
s_lshl_b32 s22, 1, s18 // (op 3555) 
s_add_i32 s23, s22, s19 // (op 3555) 


s_and_b32 s19, s0, s23 // (op 3555) 


s_add_i32 s0, s1, s19 // (op 3555) 


s_ashr_i32 s1, s0, s18 // (op 3555) 


s_lshl_b32 s0, s11, 24 // (op 3555) 
s_ashr_i32 s18, s0, 31 // (op 3555) 

s_xor_b32 s0, s1, s18 // (op 3555) 

s_sub_i32 s1, s0, s18 // (op 3555) 


s_mul_hi_i32 s0, s1, s12 // (op 3555) 
s_add_i32 s18, s0, s1 // (op 3555) 

s_ashr_i32 s0, s18, 31 // (op 3555) 
s_cmp_eq_i32 s12, 0 // (op 3555) 
s_mov_b32 s19, scc // (op 3555) 
s_cmp_eq_i32 s19, 0 // (op 3555) 
s_cselect_b32 s22, 0, -1 // (op 3555) 

s_and_b32 s19, s13, 31 // (op 3555) 
s_lshl_b32 s23, 1, s19 // (op 3555) 
s_add_i32 s24, s23, s22 // (op 3555) 


s_and_b32 s25, s0, s24 // (op 3555) 


s_add_i32 s0, s18, s25 // (op 3555) 


s_ashr_i32 s18, s0, s19 // (op 3555) 


s_lshl_b32 s0, s13, 24 // (op 3555) 
s_ashr_i32 s19, s0, 31 // (op 3555) 

s_xor_b32 s0, s18, s19 // (op 3555) 

s_sub_i32 s18, s0, s19 // (op 3555) 


s_mul_i32 s0, s1, s15 // (op 3555) 

s_sub_u32 s1, s16, s0 // (op 3555) 

s_mul_i32 s0, s18, s15 // (op 3555) 
s_add_u32 s0, s0, s1 // (op 3555) 


s_mul_hi_i32 s1, s16, s9 // (op 3555) 
s_add_i32 s18, s1, s16 // (op 3555) 

s_ashr_i32 s1, s18, 31 // (op 3555) 
s_cmp_eq_i32 s9, 0 // (op 3555) 
s_mov_b32 s19, scc // (op 3555) 
s_cmp_eq_i32 s19, 0 // (op 3555) 
s_cselect_b32 s22, 0, -1 // (op 3555) 

s_and_b32 s19, s46, 31 // (op 3555) 
s_lshl_b32 s23, 1, s19 // (op 3555) 
s_add_i32 s24, s23, s22 // (op 3555) 


s_and_b32 s25, s1, s24 // (op 3555) 


s_add_i32 s1, s18, s25 // (op 3555) 


s_ashr_i32 s18, s1, s19 // (op 3555) 


s_lshl_b32 s1, s46, 24 // (op 3555) 
s_ashr_i32 s19, s1, 31 // (op 3555) 

s_xor_b32 s1, s18, s19 // (op 3555) 

s_sub_i32 s18, s1, s19 // (op 3555) 


s_add_u32 s50, s6, 256 // (op 3555) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3555) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3555) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3555) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3555) 

s_mul_i32 s1, s15, s24 // (op 3555) 

s_mul_i32 s19, s8, s1 // (op 3555) 

s_mul_i32 s1, s18, s19 // (op 3555) 

s_sub_u32 s19, s16, s1 // (op 3555) 

s_mov_b32 s1, s19 // (op 3555) 
s_mul_hi_i32 s22, s1, s14 // (op 3555) 
s_add_i32 s23, s22, s1 // (op 3555) 


s_ashr_i32 s1, s23, 31 // (op 3555) 
s_cmp_eq_i32 s14, 0 // (op 3555) 
s_mov_b32 s22, scc // (op 3555) 
s_cmp_eq_i32 s22, 0 // (op 3555) 
s_cselect_b32 s24, 0, -1 // (op 3555) 

s_and_b32 s22, s48, 31 // (op 3555) 
s_lshl_b32 s25, 1, s22 // (op 3555) 
s_add_i32 s49, s25, s24 // (op 3555) 


s_and_b32 s24, s1, s49 // (op 3555) 


s_add_i32 s1, s23, s24 // (op 3555) 


s_ashr_i32 s23, s1, s22 // (op 3555) 


s_lshl_b32 s1, s48, 24 // (op 3555) 
s_ashr_i32 s22, s1, 31 // (op 3555) 

s_xor_b32 s1, s23, s22 // (op 3555) 

s_sub_i32 s23, s1, s22 // (op 3555) 


s_mul_i32 s1, s47, s15 // (op 3555) 
s_add_u32 s24, s20, 32 // (op 3555) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3555) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3555) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3555) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 3555) 

s_sub_i32 s22, s24, s1 // (op 3555) 

s_mul_i32 s24, s23, s22 // (op 3555) 


s_sub_u32 s25, s19, s24 // (op 3555) 


s_add_u32 s19, s25, s1 // (op 3555) 


s_cmp_eq_i32 s18, 0 // (op 3555) 
s_mov_b32 s1, scc // (op 3555) 

s_cmp_eq_i32 s1, 0 // (op 3555) 
s_cselect_b32 s18, s19, s0 // (op 3555) 



v_lshl_add_u32 v16, s18, 5, v18 // (op 3555) 


v_mul_lo_u32 v18, v17, s36 // (op 3555) 

v_add_u32 v17, v16, v18 // (op 3555) 


s_mul_i32 s1, s36, s41 // (op 3557) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 3557) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 3557) least significant
s_mul_hi_u32 s19, s36, s40 // (op 3557) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3557) most significant: sum
s_add_u32 s1, s1, s19 // (op 3557) most significant: sum


s_mov_b32 s68, s38 // (op 3557) reg expression
s_mov_b32 s69, s39 // (op 3557) reg expression
s_mov_b32 s70, s0 // (op 3557) reg expression
s_mov_b32 s71, 131072 // (op 3557) call()

s_mul_hi_i32 s0, s16, s10 // (op 3574) 
v_lshrrev_b32 v16, 7, v1 // (op 3574) 
s_add_i32 s1, s0, s16 // (op 3574) 

v_and_b32 v18, 15, v1 // (op 3574) 
s_ashr_i32 s0, s1, 31 // (op 3574) 
v_lshl_add_u32 v19, v16, 4, v18 // (op 3574) 


s_cmp_eq_i32 s10, 0 // (op 3574) 
s_mov_b32 s18, scc // (op 3574) 
v_and_b32 v16, 63, v1 // (op 3574) 
s_cmp_eq_i32 s18, 0 // (op 3574) 
s_cselect_b32 s19, 0, -1 // (op 3574) 

v_lshrrev_b32 v18, 4, v16 // (op 3574) 

s_and_b32 s18, s11, 31 // (op 3574) 
s_lshl_b32 s22, 1, s18 // (op 3574) 
s_add_i32 s23, s22, s19 // (op 3574) 


s_and_b32 s19, s0, s23 // (op 3574) 


s_add_i32 s0, s1, s19 // (op 3574) 


s_ashr_i32 s1, s0, s18 // (op 3574) 


s_lshl_b32 s0, s11, 24 // (op 3574) 
s_ashr_i32 s18, s0, 31 // (op 3574) 

s_xor_b32 s0, s1, s18 // (op 3574) 

s_sub_i32 s1, s0, s18 // (op 3574) 


s_mul_hi_i32 s0, s1, s12 // (op 3574) 
s_add_i32 s18, s0, s1 // (op 3574) 

s_ashr_i32 s0, s18, 31 // (op 3574) 
s_cmp_eq_i32 s12, 0 // (op 3574) 
s_mov_b32 s19, scc // (op 3574) 
s_cmp_eq_i32 s19, 0 // (op 3574) 
s_cselect_b32 s22, 0, -1 // (op 3574) 

s_and_b32 s19, s13, 31 // (op 3574) 
s_lshl_b32 s23, 1, s19 // (op 3574) 
s_add_i32 s24, s23, s22 // (op 3574) 


s_and_b32 s25, s0, s24 // (op 3574) 


s_add_i32 s0, s18, s25 // (op 3574) 


s_ashr_i32 s18, s0, s19 // (op 3574) 


s_lshl_b32 s0, s13, 24 // (op 3574) 
s_ashr_i32 s19, s0, 31 // (op 3574) 

s_xor_b32 s0, s18, s19 // (op 3574) 

s_sub_i32 s18, s0, s19 // (op 3574) 


s_mul_i32 s0, s1, s15 // (op 3574) 

s_sub_u32 s1, s16, s0 // (op 3574) 

s_mul_i32 s0, s18, s15 // (op 3574) 
s_add_u32 s0, s0, s1 // (op 3574) 


s_mul_hi_i32 s1, s16, s9 // (op 3574) 
s_add_i32 s18, s1, s16 // (op 3574) 

s_ashr_i32 s1, s18, 31 // (op 3574) 
s_cmp_eq_i32 s9, 0 // (op 3574) 
s_mov_b32 s19, scc // (op 3574) 
s_cmp_eq_i32 s19, 0 // (op 3574) 
s_cselect_b32 s22, 0, -1 // (op 3574) 

s_and_b32 s19, s46, 31 // (op 3574) 
s_lshl_b32 s23, 1, s19 // (op 3574) 
s_add_i32 s24, s23, s22 // (op 3574) 


s_and_b32 s25, s1, s24 // (op 3574) 


s_add_i32 s1, s18, s25 // (op 3574) 


s_ashr_i32 s18, s1, s19 // (op 3574) 


s_lshl_b32 s1, s46, 24 // (op 3574) 
s_ashr_i32 s19, s1, 31 // (op 3574) 

s_xor_b32 s1, s18, s19 // (op 3574) 

s_sub_i32 s18, s1, s19 // (op 3574) 


s_add_u32 s50, s6, 256 // (op 3574) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3574) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3574) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3574) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3574) 

s_mul_i32 s1, s15, s24 // (op 3574) 

s_mul_i32 s19, s8, s1 // (op 3574) 

s_mul_i32 s1, s18, s19 // (op 3574) 

s_sub_u32 s19, s16, s1 // (op 3574) 

s_mov_b32 s1, s19 // (op 3574) 
s_mul_hi_i32 s22, s1, s14 // (op 3574) 
s_add_i32 s23, s22, s1 // (op 3574) 


s_ashr_i32 s1, s23, 31 // (op 3574) 
s_cmp_eq_i32 s14, 0 // (op 3574) 
s_mov_b32 s22, scc // (op 3574) 
s_cmp_eq_i32 s22, 0 // (op 3574) 
s_cselect_b32 s24, 0, -1 // (op 3574) 

s_and_b32 s22, s48, 31 // (op 3574) 
s_lshl_b32 s25, 1, s22 // (op 3574) 
s_add_i32 s49, s25, s24 // (op 3574) 


s_and_b32 s24, s1, s49 // (op 3574) 


s_add_i32 s1, s23, s24 // (op 3574) 


s_ashr_i32 s23, s1, s22 // (op 3574) 


s_lshl_b32 s1, s48, 24 // (op 3574) 
s_ashr_i32 s22, s1, 31 // (op 3574) 

s_xor_b32 s1, s23, s22 // (op 3574) 

s_sub_i32 s23, s1, s22 // (op 3574) 


s_mul_i32 s1, s47, s15 // (op 3574) 
s_add_u32 s24, s20, 32 // (op 3574) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3574) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3574) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3574) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 3574) 

s_sub_i32 s22, s24, s1 // (op 3574) 

s_mul_i32 s24, s23, s22 // (op 3574) 


s_sub_u32 s25, s19, s24 // (op 3574) 


s_add_u32 s19, s25, s1 // (op 3574) 


s_cmp_eq_i32 s18, 0 // (op 3574) 
s_mov_b32 s1, scc // (op 3574) 

s_cmp_eq_i32 s1, 0 // (op 3574) 
s_cselect_b32 s18, s19, s0 // (op 3574) 



v_lshl_add_u32 v16, s18, 5, v19 // (op 3574) 


v_mul_lo_u32 v19, v18, s36 // (op 3574) 

v_add_u32 v18, v16, v19 // (op 3574) 


s_mul_i32 s1, s36, s41 // (op 3576) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 3576) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 3576) least significant
s_mul_hi_u32 s19, s36, s40 // (op 3576) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3576) most significant: sum
s_add_u32 s1, s1, s19 // (op 3576) most significant: sum


s_mov_b32 s68, s38 // (op 3576) reg expression
s_mov_b32 s69, s39 // (op 3576) reg expression
s_mov_b32 s70, s0 // (op 3576) reg expression
s_mov_b32 s71, 131072 // (op 3576) call()

s_mul_hi_i32 s0, s16, s10 // (op 3619) 
s_add_i32 s1, s0, s16 // (op 3619) 

s_ashr_i32 s0, s1, 31 // (op 3619) 
s_cmp_eq_i32 s10, 0 // (op 3619) 
s_mov_b32 s18, scc // (op 3619) 
s_cmp_eq_i32 s18, 0 // (op 3619) 
s_cselect_b32 s19, 0, -1 // (op 3619) 

s_and_b32 s18, s11, 31 // (op 3619) 
s_lshl_b32 s22, 1, s18 // (op 3619) 
s_add_i32 s23, s22, s19 // (op 3619) 


s_and_b32 s19, s0, s23 // (op 3619) 


s_add_i32 s0, s1, s19 // (op 3619) 


s_ashr_i32 s1, s0, s18 // (op 3619) 


s_lshl_b32 s0, s11, 24 // (op 3619) 
s_ashr_i32 s18, s0, 31 // (op 3619) 

s_xor_b32 s0, s1, s18 // (op 3619) 

s_sub_i32 s1, s0, s18 // (op 3619) 


s_mul_hi_i32 s0, s1, s12 // (op 3619) 
s_add_i32 s18, s0, s1 // (op 3619) 

s_ashr_i32 s0, s18, 31 // (op 3619) 
s_cmp_eq_i32 s12, 0 // (op 3619) 
s_mov_b32 s19, scc // (op 3619) 
s_cmp_eq_i32 s19, 0 // (op 3619) 
s_cselect_b32 s22, 0, -1 // (op 3619) 

s_and_b32 s19, s13, 31 // (op 3619) 
s_lshl_b32 s23, 1, s19 // (op 3619) 
s_add_i32 s24, s23, s22 // (op 3619) 


s_and_b32 s25, s0, s24 // (op 3619) 


s_add_i32 s0, s18, s25 // (op 3619) 


s_ashr_i32 s18, s0, s19 // (op 3619) 


s_lshl_b32 s0, s13, 24 // (op 3619) 
s_ashr_i32 s19, s0, 31 // (op 3619) 

s_xor_b32 s0, s18, s19 // (op 3619) 

s_sub_i32 s18, s0, s19 // (op 3619) 


s_add_u32 s50, s6, 256 // (op 3619) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3619) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3619) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3619) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3619) 

s_mul_i32 s0, s18, s24 // (op 3619) 

s_sub_i32 s18, s1, s0 // (op 3619) 


s_mul_hi_i32 s19, s16, s9 // (op 3619) 
s_add_i32 s0, s19, s16 // (op 3619) 

s_ashr_i32 s1, s0, 31 // (op 3619) 
s_cmp_eq_i32 s9, 0 // (op 3619) 
s_mov_b32 s19, scc // (op 3619) 
s_cmp_eq_i32 s19, 0 // (op 3619) 
s_cselect_b32 s22, 0, -1 // (op 3619) 

s_and_b32 s19, s46, 31 // (op 3619) 
s_lshl_b32 s23, 1, s19 // (op 3619) 
s_add_i32 s49, s23, s22 // (op 3619) 


s_and_b32 s22, s1, s49 // (op 3619) 


s_add_i32 s1, s0, s22 // (op 3619) 


s_ashr_i32 s0, s1, s19 // (op 3619) 


s_lshl_b32 s1, s46, 24 // (op 3619) 
s_ashr_i32 s19, s1, 31 // (op 3619) 

s_xor_b32 s1, s0, s19 // (op 3619) 

s_sub_i32 s0, s1, s19 // (op 3619) 


s_mul_i32 s1, s15, s24 // (op 3619) 

s_mul_i32 s19, s8, s1 // (op 3619) 

s_mul_i32 s1, s0, s19 // (op 3619) 

s_sub_u32 s19, s16, s1 // (op 3619) 

s_mul_hi_i32 s1, s19, s14 // (op 3619) 
s_add_i32 s22, s1, s19 // (op 3619) 


s_ashr_i32 s1, s22, 31 // (op 3619) 
s_cmp_eq_i32 s14, 0 // (op 3619) 
s_mov_b32 s19, scc // (op 3619) 
s_cmp_eq_i32 s19, 0 // (op 3619) 
s_cselect_b32 s23, 0, -1 // (op 3619) 

s_and_b32 s19, s48, 31 // (op 3619) 
s_lshl_b32 s24, 1, s19 // (op 3619) 
s_add_i32 s25, s24, s23 // (op 3619) 


s_and_b32 s23, s1, s25 // (op 3619) 


s_add_i32 s1, s22, s23 // (op 3619) 


s_ashr_i32 s22, s1, s19 // (op 3619) 


s_lshl_b32 s1, s48, 24 // (op 3619) 
s_ashr_i32 s19, s1, 31 // (op 3619) 

s_xor_b32 s1, s22, s19 // (op 3619) 

s_sub_i32 s22, s1, s19 // (op 3619) 


s_cmp_eq_i32 s0, 0 // (op 3619) 
s_mov_b32 s1, scc // (op 3619) 

s_cmp_eq_i32 s1, 0 // (op 3619) 
s_cselect_b32 s0, s22, s18 // (op 3619) 



v_lshl_add_u32 v16, s0, 8, v1 // (op 3619) 

v_mul_lo_u32 v19, v16, s42 // (op 3619) 

s_mul_i32 s1, s42, s7 // (op 3621) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 3621) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 3621) least significant
s_mul_hi_u32 s19, s42, s6 // (op 3621) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3621) most significant: sum
s_add_u32 s1, s1, s19 // (op 3621) most significant: sum


s_mov_b32 s72, s44 // (op 3621) reg expression
s_mov_b32 s73, s45 // (op 3621) reg expression
s_mov_b32 s74, s0 // (op 3621) reg expression
s_mov_b32 s75, 131072 // (op 3621) call()

s_mul_hi_i32 s0, s16, s10 // (op 3638) 
s_add_i32 s1, s0, s16 // (op 3638) 

s_ashr_i32 s0, s1, 31 // (op 3638) 
s_cmp_eq_i32 s10, 0 // (op 3638) 
s_mov_b32 s18, scc // (op 3638) 
s_cmp_eq_i32 s18, 0 // (op 3638) 
s_cselect_b32 s19, 0, -1 // (op 3638) 

s_and_b32 s18, s11, 31 // (op 3638) 
s_lshl_b32 s22, 1, s18 // (op 3638) 
s_add_i32 s23, s22, s19 // (op 3638) 


s_and_b32 s19, s0, s23 // (op 3638) 


s_add_i32 s0, s1, s19 // (op 3638) 


s_ashr_i32 s1, s0, s18 // (op 3638) 


s_lshl_b32 s0, s11, 24 // (op 3638) 
s_ashr_i32 s18, s0, 31 // (op 3638) 

s_xor_b32 s0, s1, s18 // (op 3638) 

s_sub_i32 s1, s0, s18 // (op 3638) 


s_mul_hi_i32 s0, s1, s12 // (op 3638) 
s_add_i32 s18, s0, s1 // (op 3638) 

s_ashr_i32 s0, s18, 31 // (op 3638) 
s_cmp_eq_i32 s12, 0 // (op 3638) 
s_mov_b32 s19, scc // (op 3638) 
s_cmp_eq_i32 s19, 0 // (op 3638) 
s_cselect_b32 s22, 0, -1 // (op 3638) 

s_and_b32 s19, s13, 31 // (op 3638) 
s_lshl_b32 s23, 1, s19 // (op 3638) 
s_add_i32 s24, s23, s22 // (op 3638) 


s_and_b32 s25, s0, s24 // (op 3638) 


s_add_i32 s0, s18, s25 // (op 3638) 


s_ashr_i32 s18, s0, s19 // (op 3638) 


s_lshl_b32 s0, s13, 24 // (op 3638) 
s_ashr_i32 s19, s0, 31 // (op 3638) 

s_xor_b32 s0, s18, s19 // (op 3638) 

s_sub_i32 s18, s0, s19 // (op 3638) 


s_add_u32 s50, s6, 256 // (op 3638) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3638) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3638) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3638) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3638) 

s_mul_i32 s0, s18, s24 // (op 3638) 

s_sub_i32 s18, s1, s0 // (op 3638) 


s_mul_hi_i32 s19, s16, s9 // (op 3638) 
s_add_i32 s0, s19, s16 // (op 3638) 

s_ashr_i32 s1, s0, 31 // (op 3638) 
s_cmp_eq_i32 s9, 0 // (op 3638) 
s_mov_b32 s19, scc // (op 3638) 
s_cmp_eq_i32 s19, 0 // (op 3638) 
s_cselect_b32 s22, 0, -1 // (op 3638) 

s_and_b32 s19, s46, 31 // (op 3638) 
s_lshl_b32 s23, 1, s19 // (op 3638) 
s_add_i32 s49, s23, s22 // (op 3638) 


s_and_b32 s22, s1, s49 // (op 3638) 


s_add_i32 s1, s0, s22 // (op 3638) 


s_ashr_i32 s0, s1, s19 // (op 3638) 


s_lshl_b32 s1, s46, 24 // (op 3638) 
s_ashr_i32 s19, s1, 31 // (op 3638) 

s_xor_b32 s1, s0, s19 // (op 3638) 

s_sub_i32 s0, s1, s19 // (op 3638) 


s_mul_i32 s1, s15, s24 // (op 3638) 

s_mul_i32 s19, s8, s1 // (op 3638) 

s_mul_i32 s1, s0, s19 // (op 3638) 

s_sub_u32 s19, s16, s1 // (op 3638) 

s_mul_hi_i32 s1, s19, s14 // (op 3638) 
s_add_i32 s22, s1, s19 // (op 3638) 


s_ashr_i32 s1, s22, 31 // (op 3638) 
s_cmp_eq_i32 s14, 0 // (op 3638) 
s_mov_b32 s19, scc // (op 3638) 
s_cmp_eq_i32 s19, 0 // (op 3638) 
s_cselect_b32 s23, 0, -1 // (op 3638) 

s_and_b32 s19, s48, 31 // (op 3638) 
s_lshl_b32 s24, 1, s19 // (op 3638) 
s_add_i32 s25, s24, s23 // (op 3638) 


s_and_b32 s23, s1, s25 // (op 3638) 


s_add_i32 s1, s22, s23 // (op 3638) 


s_ashr_i32 s22, s1, s19 // (op 3638) 


s_lshl_b32 s1, s48, 24 // (op 3638) 
s_ashr_i32 s19, s1, 31 // (op 3638) 

s_xor_b32 s1, s22, s19 // (op 3638) 

s_sub_i32 s22, s1, s19 // (op 3638) 


s_cmp_eq_i32 s0, 0 // (op 3638) 
s_mov_b32 s1, scc // (op 3638) 

s_cmp_eq_i32 s1, 0 // (op 3638) 
s_cselect_b32 s0, s22, s18 // (op 3638) 



v_lshl_add_u32 v16, s0, 8, v1 // (op 3638) 

v_mul_lo_u32 v20, v16, s42 // (op 3638) 

s_mul_i32 s1, s42, s7 // (op 3640) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 3640) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 3640) least significant
s_mul_hi_u32 s19, s42, s6 // (op 3640) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3640) most significant: sum
s_add_u32 s1, s1, s19 // (op 3640) most significant: sum


s_mov_b32 s72, s44 // (op 3640) reg expression
s_mov_b32 s73, s45 // (op 3640) reg expression
s_mov_b32 s74, s0 // (op 3640) reg expression
s_mov_b32 s75, 131072 // (op 3640) call()


v_mov_b64 v[22:23], s[36:37] // (op 3108) convert

v_mov_b32 v16, v17 // (op 3108) reg expression
v_mov_b32 v21, v16 // (op 3108) 
buffer_load_ubyte v24, v21, s[68:71], 0 offen offset: 0 // (op 3108) Load value




s_lshl_b64 s[0:1], s[32:33], 2 // (op 3112) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 3112) 


v_mov_b32 v16, v12 // (op 3112) 
buffer_load_dwordx4 v[26:29], v16, s[64:67], 0 offen // (op 3112) Load value
v_add_u32 v16, v16, v22 // (op 3112) 
buffer_load_dwordx4 v[30:33], v16, s[64:67], 0 offen // (op 3112) Load value
v_add_u32 v16, v16, v22 // (op 3112) 
buffer_load_dwordx4 v[34:37], v16, s[64:67], 0 offen // (op 3112) Load value
v_add_u32 v16, v16, v22 // (op 3112) 
buffer_load_dwordx4 v[38:41], v16, s[64:67], 0 offen // (op 3112) Load value





buffer_load_dword v16, v19, s[72:75], 0 offen // (op 3115) Load value


v_mov_b32 v25, v13 // (op 3118) reg expression

buffer_load_dwordx4 v[42:45], v25, s[52:55], 0 offen // (op 3118) Load value


v_mov_b64 v[22:23], s[36:37] // (op 3122) convert

s_lshl_b32 s0, s36, 2 // (op 3122) 
v_add_u32 v21, v17, s0 // (op 3122) 

v_mov_b32 v25, v21 // (op 3122) 
buffer_load_ubyte v46, v25, s[68:71], 0 offen offset: 0 // (op 3122) Load value





s_lshl_b64 s[0:1], s[32:33], 2 // (op 3126) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 3126) 


v_mov_b32 v17, v12 // (op 3126) 
buffer_load_dwordx4 v[48:51], v17, s[64:67], 0 offen offset:64 // (op 3126) Load value
v_add_u32 v17, v17, v22 // (op 3126) 
buffer_load_dwordx4 v[52:55], v17, s[64:67], 0 offen offset:64 // (op 3126) Load value
v_add_u32 v17, v17, v22 // (op 3126) 
buffer_load_dwordx4 v[56:59], v17, s[64:67], 0 offen offset:64 // (op 3126) Load value
v_add_u32 v17, v17, v22 // (op 3126) 
buffer_load_dwordx4 v[60:63], v17, s[64:67], 0 offen offset:64 // (op 3126) Load value






buffer_load_dword v12, v19, s[72:75], 0 offen offset:4 // (op 3129) Load value



s_lshl_b32 s0, s26, 7 // (op 3132) 
v_add_u32 v17, v13, s0 // (op 3132) 


buffer_load_dwordx4 v[64:67], v17, s[52:55], 0 offen // (op 3132) Load value


v_and_b32 v13, -4, v1 // (op 3758) 
v_sub_u32 v17, v1, v13 // (op 3758) 

v_lshlrev_b32 v13, 5, v17 // (op 3758) 

v_lshrrev_b32 v17, 2, v1 // (op 3758) 
v_lshl_add_u32 v19, v17, 9, v13 // (op 3758) 


v_lshlrev_b32 v13, 2, v19 // (op 3758) 

v_lshrrev_b32 v17, 3, v13 // (op 3758) 




s_waitcnt vmcnt(12)
ds_write_b128 v17, v[26:29] // (op 3134) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v17, v[30:33] offset:64 // (op 3134) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v17, v[34:37] offset:128 // (op 3134) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v17, v[38:41] offset:192 // (op 3134) Store local data 

v_lshlrev_b32 v13, 2, v1 // (op 3816) 



s_waitcnt vmcnt(8)
ds_write_b32 v13, v16 offset:16384 // (op 3135) Store local data 

v_and_b32 v13, -2, v1 // (op 3705) 
v_sub_u32 v17, v1, v13 // (op 3705) 

v_lshlrev_b32 v13, 4, v17 // (op 3705) 

v_lshrrev_b32 v17, 1, v1 // (op 3705) 
v_lshl_add_u32 v19, v17, 5, v13 // (op 3705) 





s_waitcnt vmcnt(7)
ds_write_b128 v19, v[42:45] offset:17408 // (op 3136) Store local data 

v_and_b32 v13, -2, v1 // (op 3676) 
v_sub_u32 v17, v1, v13 // (op 3676) 

v_lshlrev_b32 v13, 4, v17 // (op 3676) 

v_lshrrev_b32 v17, 1, v1 // (op 3676) 
v_lshl_add_u32 v19, v17, 5, v13 // (op 3676) 


v_and_b32 v13, -4, v1 // (op 3729) 
v_sub_u32 v17, v1, v13 // (op 3729) 

v_lshlrev_b32 v13, 5, v17 // (op 3729) 

v_lshrrev_b32 v17, 2, v1 // (op 3729) 
v_lshl_add_u32 v25, v17, 9, v13 // (op 3729) 


v_lshlrev_b32 v13, 2, v25 // (op 3729) 

v_lshrrev_b32 v17, 3, v13 // (op 3729) 

v_lshlrev_b32 v13, 2, v1 // (op 3787) 
v_and_b32 v25, -2, v1 // (op 3863) 
v_sub_u32 v47, v1, v25 // (op 3863) 

v_lshlrev_b32 v25, 4, v47 // (op 3863) 

v_lshrrev_b32 v47, 1, v1 // (op 3863) 
v_lshl_add_u32 v21, v47, 5, v25 // (op 3863) 


v_and_b32 v25, -4, v1 // (op 3887) 
v_sub_u32 v47, v1, v25 // (op 3887) 

v_lshlrev_b32 v25, 5, v47 // (op 3887) 

v_lshrrev_b32 v47, 2, v1 // (op 3887) 
v_lshl_add_u32 v22, v47, 9, v25 // (op 3887) 


v_lshlrev_b32 v23, 2, v22 // (op 3887) 

v_lshrrev_b32 v22, 3, v23 // (op 3887) 

v_lshlrev_b32 v23, 2, v1 // (op 3916) 
s_mov_b32 s0, 0 // (op 1244) call()
s_ashr_i64 s[18:19], s[30:31], 7 // (op 1243) 
s_mov_b32 s1, s18 // (op 1243) convert

s_and_b32 s18, s1, 4294967294 // (op 1243) 

s_cmp_lt_u32 s0, s18 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoop_1243 // (op 1243) Condition: Top (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoop_1243 if false)
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoop_1243:
s_add_u32 s0, s0, 2 // (op 1245) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3992) 



ds_read_b64_tr_b8 v[68:69], v0 offset:17408 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:17664 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:19456 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:19712 // (op 65) Transpose load local data 
s_barrier  // (op 4005) 



ds_read_b128 v[76:79], v7 // (op 70) Load local data 
s_barrier  // (op 4019) 



ds_read_u8 v25, v4 offset:16384 // (op 79) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[76:79], a[0:3], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 82) 
ds_read_b128 v[80:83], v7 offset:2048 // (op 339) Load local data 





ds_read_u8 v25, v4 offset:16512 // (op 342) Load local data 



ds_read_b128 v[76:79], v7 offset:4096 // (op 495) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[80:83], a[4:7], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 343) 





ds_read_u8 v25, v4 offset:16640 // (op 498) Load local data 



ds_read_b128 v[80:83], v7 offset:6144 // (op 651) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:75], v[76:79], a[8:11], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 499) 





ds_read_u8 v25, v4 offset:16768 // (op 654) Load local data 



ds_read_b128 v[76:79], v7 offset:8192 // (op 807) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:75], v[80:83], a[12:15], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 655) 





ds_read_u8 v25, v4 offset:16896 // (op 810) Load local data 



ds_read_b128 v[80:83], v7 offset:10240 // (op 963) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:75], v[76:79], a[16:19], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 811) 





ds_read_u8 v25, v4 offset:17024 // (op 966) Load local data 



ds_read_b128 v[76:79], v7 offset:12288 // (op 1119) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:75], v[80:83], a[20:23], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 967) 





ds_read_u8 v25, v4 offset:17152 // (op 1122) Load local data 



ds_read_b128 v[80:83], v7 offset:14336 // (op 1275) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:75], v[76:79], a[24:27], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1123) 





ds_read_u8 v25, v4 offset:17280 // (op 1278) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[68:75], v[80:83], a[28:31], v24, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1279) 




v_mov_b64 v[68:69], s[36:37] // (op 75) convert

v_mov_b32 v25, v18 // (op 75) 
s_lshl_b32 s1, s36, 3 // (op 75) 
v_add_u32 v47, v25, s1 // (op 75) 

v_mov_b32 v70, v47 // (op 75) 
buffer_load_ubyte v24, v70, s[68:71], 0 offen offset: 0 // (op 75) Load value




s_lshl_b64 s[18:19], s[32:33], 2 // (op 10) 
v_lshrrev_b64 v[68:69], 3, s[18:19] // (op 10) 


v_mov_b32 v47, v15 // (op 10) 
v_mov_b32 v70, v47 // (op 10) 
buffer_load_dwordx4 v[26:29], v70, s[64:67], 0 offen offset:128 // (op 10) Load value
v_add_u32 v70, v70, v68 // (op 10) 
buffer_load_dwordx4 v[30:33], v70, s[64:67], 0 offen offset:128 // (op 10) Load value
v_add_u32 v70, v70, v68 // (op 10) 
buffer_load_dwordx4 v[34:37], v70, s[64:67], 0 offen offset:128 // (op 10) Load value
v_add_u32 v70, v70, v68 // (op 10) 
buffer_load_dwordx4 v[38:41], v70, s[64:67], 0 offen offset:128 // (op 10) Load value




v_mov_b32 v68, v20 // (op 21) 

buffer_load_dword v16, v68, s[72:75], 0 offen offset:8 // (op 21) Load value


v_mov_b32 v69, v14 // (op 4) 
s_lshl_b32 s1, s26, 8 // (op 4) 
v_add_u32 v70, v69, s1 // (op 4) 


buffer_load_dwordx4 v[42:45], v70, s[52:55], 0 offen // (op 4) Load value

s_barrier  // (op 4042) 



s_waitcnt vmcnt(12)
ds_write_b128 v22, v[48:51] offset:21504 // (op 235) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v22, v[52:55] offset:21568 // (op 235) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v22, v[56:59] offset:21632 // (op 235) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v22, v[60:63] offset:21696 // (op 235) Store local data 
s_barrier  // (op 4048) 



s_waitcnt vmcnt(8)
ds_write_b32 v23, v12 offset:37888 // (op 236) Store local data 
s_barrier  // (op 4036) 



s_waitcnt vmcnt(7)
ds_write_b128 v21, v[64:67] offset:38912 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4033) 



ds_read_b64_tr_b8 v[70:71], v3 offset:38912 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v3 offset:39168 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v3 offset:40960 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v3 offset:41216 // (op 227) Transpose load local data 
s_barrier  // (op 4039) 
s_barrier  // (op 4085) 



ds_read_b128 v[78:81], v11 offset:21504 // (op 229) Load local data 
s_barrier  // (op 4045) 



ds_read_u8 v82, v10 offset:37888 // (op 233) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[78:81], a[0:3], v46, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 234) 
ds_read_b128 v[84:87], v11 offset:23552 // (op 348) Load local data 





ds_read_u8 v78, v10 offset:38016 // (op 351) Load local data 



ds_read_b128 v[80:83], v11 offset:25600 // (op 504) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:77], v[84:87], a[4:7], v46, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 344) 





ds_read_u8 v78, v10 offset:38144 // (op 507) Load local data 



ds_read_b128 v[84:87], v11 offset:27648 // (op 660) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[70:77], v[80:83], a[8:11], v46, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 500) 





ds_read_u8 v78, v10 offset:38272 // (op 663) Load local data 



ds_read_b128 v[80:83], v11 offset:29696 // (op 816) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:77], v[84:87], a[12:15], v46, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 656) 





ds_read_u8 v78, v10 offset:38400 // (op 819) Load local data 



ds_read_b128 v[84:87], v11 offset:31744 // (op 972) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:77], v[80:83], a[16:19], v46, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 812) 





ds_read_u8 v78, v10 offset:38528 // (op 975) Load local data 



ds_read_b128 v[80:83], v11 offset:33792 // (op 1128) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:77], v[84:87], a[20:23], v46, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 968) 





ds_read_u8 v78, v10 offset:38656 // (op 1131) Load local data 



ds_read_b128 v[84:87], v11 offset:35840 // (op 1284) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:77], v[80:83], a[24:27], v46, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1124) 





s_barrier  // (op 4155) 
ds_read_u8 v78, v10 offset:38784 // (op 1287) Load local data 
s_barrier  // (op 4159) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:77], v[84:87], a[28:31], v46, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1280) 




v_mov_b64 v[70:71], s[36:37] // (op 231) convert

s_lshl_b32 s1, s36, 2 // (op 231) 
s_mul_i32 s18, 3, s1 // (op 231) 

v_add_u32 v72, v25, s18 // (op 231) 

v_mov_b32 v73, v72 // (op 231) 
buffer_load_ubyte v46, v73, s[68:71], 0 offen offset: 0 // (op 231) Load value





s_lshl_b32 s1, s36, 3 // (op 3566) 
v_add_u32 v18, v18, s1 // (op 3566) 

s_lshl_b64 s[18:19], s[32:33], 2 // (op 237) 
v_lshrrev_b64 v[70:71], 3, s[18:19] // (op 237) 


v_mov_b32 v25, v47 // (op 237) 
buffer_load_dwordx4 v[48:51], v25, s[64:67], 0 offen offset:192 // (op 237) Load value
v_add_u32 v25, v25, v70 // (op 237) 
buffer_load_dwordx4 v[52:55], v25, s[64:67], 0 offen offset:192 // (op 237) Load value
v_add_u32 v25, v25, v70 // (op 237) 
buffer_load_dwordx4 v[56:59], v25, s[64:67], 0 offen offset:192 // (op 237) Load value
v_add_u32 v25, v25, v70 // (op 237) 
buffer_load_dwordx4 v[60:63], v25, s[64:67], 0 offen offset:192 // (op 237) Load value






v_add_u32 v15, 128, v15 // (op 3502) 
buffer_load_dword v12, v68, s[72:75], 0 offen offset:12 // (op 238) Load value



v_add_u32 v20, 8, v20 // (op 3630) 
s_lshl_b32 s1, s26, 7 // (op 224) 
s_mul_i32 s18, 3, s1 // (op 224) 

v_add_u32 v25, v69, s18 // (op 224) 


buffer_load_dwordx4 v[64:67], v25, s[52:55], 0 offen // (op 224) Load value


s_barrier  // (op 4008) 
s_lshl_b32 s1, s26, 8 // (op 3438) 



v_add_u32 v14, v14, s1 // (op 3438) 

s_waitcnt vmcnt(12)
ds_write_b128 v17, v[26:29] // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v17, v[30:33] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v17, v[34:37] offset:128 // (op 13) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v17, v[38:41] offset:192 // (op 13) Store local data 
s_barrier  // (op 4022) 



s_waitcnt vmcnt(8)
ds_write_b32 v13, v16 offset:16384 // (op 26) Store local data 
s_barrier  // (op 3995) 



s_waitcnt vmcnt(7)
ds_write_b128 v19, v[42:45] offset:17408 // (op 6) Store local data 
s_ashr_i64 s[18:19], s[30:31], 7 // (op 1243) 
s_mov_b32 s1, s18 // (op 1243) convert

s_and_b32 s18, s1, 4294967294 // (op 1243) 

s_cmp_lt_u32 s0, s18 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoop_1243 // (op 1243) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoop_1243 if true)
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoop_1243:















s_mul_hi_i32 s0, s16, s10 // (op 3394) 
v_and_b32 v47, -2, v1 // (op 3394) 
s_add_i32 s1, s0, s16 // (op 3394) 

v_sub_u32 v13, v1, v47 // (op 3394) 

s_ashr_i32 s0, s1, 31 // (op 3394) 
v_lshlrev_b32 v47, 4, v13 // (op 3394) 

s_cmp_eq_i32 s10, 0 // (op 3394) 
s_mov_b32 s18, scc // (op 3394) 
v_lshrrev_b32 v13, 1, v1 // (op 3394) 
s_cmp_eq_i32 s18, 0 // (op 3394) 
s_cselect_b32 s19, 0, -1 // (op 3394) 

s_and_b32 s18, s11, 31 // (op 3394) 
s_lshl_b32 s22, 1, s18 // (op 3394) 
s_add_i32 s23, s22, s19 // (op 3394) 


s_and_b32 s19, s0, s23 // (op 3394) 


s_add_i32 s0, s1, s19 // (op 3394) 


s_ashr_i32 s1, s0, s18 // (op 3394) 


s_lshl_b32 s0, s11, 24 // (op 3394) 
s_ashr_i32 s18, s0, 31 // (op 3394) 

s_xor_b32 s0, s1, s18 // (op 3394) 

s_sub_i32 s1, s0, s18 // (op 3394) 


s_mul_hi_i32 s0, s1, s12 // (op 3394) 
s_add_i32 s18, s0, s1 // (op 3394) 

s_ashr_i32 s0, s18, 31 // (op 3394) 
s_cmp_eq_i32 s12, 0 // (op 3394) 
s_mov_b32 s19, scc // (op 3394) 
s_cmp_eq_i32 s19, 0 // (op 3394) 
s_cselect_b32 s22, 0, -1 // (op 3394) 

s_and_b32 s19, s13, 31 // (op 3394) 
s_lshl_b32 s23, 1, s19 // (op 3394) 
s_add_i32 s24, s23, s22 // (op 3394) 


s_and_b32 s25, s0, s24 // (op 3394) 


s_add_i32 s0, s18, s25 // (op 3394) 


s_ashr_i32 s18, s0, s19 // (op 3394) 


s_lshl_b32 s0, s13, 24 // (op 3394) 
s_ashr_i32 s19, s0, 31 // (op 3394) 

s_xor_b32 s0, s18, s19 // (op 3394) 

s_sub_i32 s18, s0, s19 // (op 3394) 


s_mul_i32 s0, s1, s15 // (op 3394) 

s_sub_u32 s1, s16, s0 // (op 3394) 

s_mul_i32 s0, s18, s15 // (op 3394) 
s_add_u32 s0, s0, s1 // (op 3394) 


s_mul_hi_i32 s1, s16, s9 // (op 3394) 
s_add_i32 s18, s1, s16 // (op 3394) 

s_ashr_i32 s1, s18, 31 // (op 3394) 
s_cmp_eq_i32 s9, 0 // (op 3394) 
s_mov_b32 s19, scc // (op 3394) 
s_cmp_eq_i32 s19, 0 // (op 3394) 
s_cselect_b32 s22, 0, -1 // (op 3394) 

s_and_b32 s19, s46, 31 // (op 3394) 
s_lshl_b32 s23, 1, s19 // (op 3394) 
s_add_i32 s24, s23, s22 // (op 3394) 


s_and_b32 s25, s1, s24 // (op 3394) 


s_add_i32 s1, s18, s25 // (op 3394) 


s_ashr_i32 s18, s1, s19 // (op 3394) 


s_lshl_b32 s1, s46, 24 // (op 3394) 
s_ashr_i32 s19, s1, 31 // (op 3394) 

s_xor_b32 s1, s18, s19 // (op 3394) 

s_sub_i32 s18, s1, s19 // (op 3394) 


s_add_u32 s50, s6, 256 // (op 3394) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3394) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3394) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3394) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3394) 

s_mul_i32 s1, s15, s24 // (op 3394) 

s_mul_i32 s19, s8, s1 // (op 3394) 

s_mul_i32 s1, s18, s19 // (op 3394) 

s_sub_u32 s19, s16, s1 // (op 3394) 

s_mov_b32 s1, s19 // (op 3394) 
s_mul_hi_i32 s22, s1, s14 // (op 3394) 
s_add_i32 s23, s22, s1 // (op 3394) 


s_ashr_i32 s1, s23, 31 // (op 3394) 
s_cmp_eq_i32 s14, 0 // (op 3394) 
s_mov_b32 s22, scc // (op 3394) 
s_cmp_eq_i32 s22, 0 // (op 3394) 
s_cselect_b32 s24, 0, -1 // (op 3394) 

s_and_b32 s22, s48, 31 // (op 3394) 
s_lshl_b32 s25, 1, s22 // (op 3394) 
s_add_i32 s49, s25, s24 // (op 3394) 


s_and_b32 s24, s1, s49 // (op 3394) 


s_add_i32 s1, s23, s24 // (op 3394) 


s_ashr_i32 s23, s1, s22 // (op 3394) 


s_lshl_b32 s1, s48, 24 // (op 3394) 
s_ashr_i32 s22, s1, 31 // (op 3394) 

s_xor_b32 s1, s23, s22 // (op 3394) 

s_sub_i32 s23, s1, s22 // (op 3394) 


s_mul_i32 s1, s47, s15 // (op 3394) 
s_add_u32 s24, s20, 32 // (op 3394) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3394) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3394) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3394) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 3394) 

s_sub_i32 s22, s24, s1 // (op 3394) 

s_mul_i32 s24, s23, s22 // (op 3394) 


s_sub_u32 s25, s19, s24 // (op 3394) 


s_add_u32 s19, s25, s1 // (op 3394) 


s_cmp_eq_i32 s18, 0 // (op 3394) 
s_mov_b32 s1, scc // (op 3394) 

s_cmp_eq_i32 s1, 0 // (op 3394) 
s_cselect_b32 s18, s19, s0 // (op 3394) 



v_lshl_add_u32 v14, s18, 5, v47 // (op 3394) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 3394) 
s_lshr_b32 s18, s0, 1 // (op 3394) 

v_lshl_add_u32 v47, s18, 8, v13 // (op 3394) 


v_mul_lo_u32 v13, v47, s26 // (op 3394) 

v_add_u32 v47, v14, v13 // (op 3394) 


s_mul_i32 s1, s26, s31 // (op 3396) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 3396) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 3396) least significant
s_mul_hi_u32 s19, s26, s30 // (op 3396) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3396) most significant: sum
s_add_u32 s1, s1, s19 // (op 3396) most significant: sum


s_mov_b32 s52, s28 // (op 3396) reg expression
s_mov_b32 s53, s29 // (op 3396) reg expression
s_mov_b32 s54, s0 // (op 3396) reg expression
s_mov_b32 s55, 131072 // (op 3396) call()

s_mul_hi_i32 s0, s16, s10 // (op 3468) 
v_and_b32 v13, 4294967292, v1 // (op 3468) 
s_add_i32 s1, s0, s16 // (op 3468) 

v_and_b32 v14, -4, v1 // (op 3468) 
s_ashr_i32 s0, s1, 31 // (op 3468) 
v_sub_u32 v15, v1, v14 // (op 3468) 

s_cmp_eq_i32 s10, 0 // (op 3468) 
s_mov_b32 s18, scc // (op 3468) 
v_lshlrev_b32 v14, 5, v15 // (op 3468) 

s_cmp_eq_i32 s18, 0 // (op 3468) 
s_cselect_b32 s19, 0, -1 // (op 3468) 

s_and_b32 s18, s11, 31 // (op 3468) 
s_lshl_b32 s22, 1, s18 // (op 3468) 
s_add_i32 s23, s22, s19 // (op 3468) 


s_and_b32 s19, s0, s23 // (op 3468) 


s_add_i32 s0, s1, s19 // (op 3468) 


s_ashr_i32 s1, s0, s18 // (op 3468) 


s_lshl_b32 s0, s11, 24 // (op 3468) 
s_ashr_i32 s18, s0, 31 // (op 3468) 

s_xor_b32 s0, s1, s18 // (op 3468) 

s_sub_i32 s1, s0, s18 // (op 3468) 


s_mul_hi_i32 s0, s1, s12 // (op 3468) 
s_add_i32 s18, s0, s1 // (op 3468) 

s_ashr_i32 s0, s18, 31 // (op 3468) 
s_cmp_eq_i32 s12, 0 // (op 3468) 
s_mov_b32 s19, scc // (op 3468) 
s_cmp_eq_i32 s19, 0 // (op 3468) 
s_cselect_b32 s22, 0, -1 // (op 3468) 

s_and_b32 s19, s13, 31 // (op 3468) 
s_lshl_b32 s23, 1, s19 // (op 3468) 
s_add_i32 s24, s23, s22 // (op 3468) 


s_and_b32 s25, s0, s24 // (op 3468) 


s_add_i32 s0, s18, s25 // (op 3468) 


s_ashr_i32 s18, s0, s19 // (op 3468) 


s_lshl_b32 s0, s13, 24 // (op 3468) 
s_ashr_i32 s19, s0, 31 // (op 3468) 

s_xor_b32 s0, s18, s19 // (op 3468) 

s_sub_i32 s18, s0, s19 // (op 3468) 


s_add_u32 s50, s6, 256 // (op 3468) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3468) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3468) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3468) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3468) 

s_mul_i32 s0, s18, s24 // (op 3468) 

s_sub_i32 s18, s1, s0 // (op 3468) 


s_mul_hi_i32 s19, s16, s9 // (op 3468) 
s_add_i32 s0, s19, s16 // (op 3468) 

s_ashr_i32 s1, s0, 31 // (op 3468) 
s_cmp_eq_i32 s9, 0 // (op 3468) 
s_mov_b32 s19, scc // (op 3468) 
s_cmp_eq_i32 s19, 0 // (op 3468) 
s_cselect_b32 s22, 0, -1 // (op 3468) 

s_and_b32 s19, s46, 31 // (op 3468) 
s_lshl_b32 s23, 1, s19 // (op 3468) 
s_add_i32 s49, s23, s22 // (op 3468) 


s_and_b32 s22, s1, s49 // (op 3468) 


s_add_i32 s1, s0, s22 // (op 3468) 


s_ashr_i32 s0, s1, s19 // (op 3468) 


s_lshl_b32 s1, s46, 24 // (op 3468) 
s_ashr_i32 s19, s1, 31 // (op 3468) 

s_xor_b32 s1, s0, s19 // (op 3468) 

s_sub_i32 s0, s1, s19 // (op 3468) 


s_mul_i32 s1, s15, s24 // (op 3468) 

s_mul_i32 s19, s8, s1 // (op 3468) 

s_mul_i32 s1, s0, s19 // (op 3468) 

s_sub_u32 s19, s16, s1 // (op 3468) 

s_mul_hi_i32 s1, s19, s14 // (op 3468) 
s_add_i32 s22, s1, s19 // (op 3468) 


s_ashr_i32 s1, s22, 31 // (op 3468) 
s_cmp_eq_i32 s14, 0 // (op 3468) 
s_mov_b32 s19, scc // (op 3468) 
s_cmp_eq_i32 s19, 0 // (op 3468) 
s_cselect_b32 s23, 0, -1 // (op 3468) 

s_and_b32 s19, s48, 31 // (op 3468) 
s_lshl_b32 s24, 1, s19 // (op 3468) 
s_add_i32 s25, s24, s23 // (op 3468) 


s_and_b32 s23, s1, s25 // (op 3468) 


s_add_i32 s1, s22, s23 // (op 3468) 


s_ashr_i32 s22, s1, s19 // (op 3468) 


s_lshl_b32 s1, s48, 24 // (op 3468) 
s_ashr_i32 s19, s1, 31 // (op 3468) 

s_xor_b32 s1, s22, s19 // (op 3468) 

s_sub_i32 s22, s1, s19 // (op 3468) 


s_cmp_eq_i32 s0, 0 // (op 3468) 
s_mov_b32 s1, scc // (op 3468) 

s_cmp_eq_i32 s1, 0 // (op 3468) 
s_cselect_b32 s0, s22, s18 // (op 3468) 



v_lshl_add_u32 v15, s0, 8, v13 // (op 3468) 


v_mul_lo_u32 v17, v15, s33 // (op 3468) most significant: low of low * high
v_mul_hi_u32 v13, v15, s32 // (op 3468) most significant: high of low * low
v_mul_lo_u32 v16, v15, s32 // (op 3468) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3468) most significant: sum


s_ashr_i64 s[0:1], s[30:31], 7 // (op 3468) 
s_lshr_b32 s18, s0, 1 // (op 3468) 

v_lshl_add_u32 v13, s18, 8, v14 // (op 3468) 


v_mov_b32 v15, 0 // (op 3468) convert
v_mov_b32 v14, v13 // (op 3468) convert

v_add_co_u32 v18, s[0:1], v14, v16 // (op 3468) least significant half
v_addc_co_u32 v19, s[0:1], v15, v17, s[0:1] // (op 3468) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 3468) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 3468) 

v_mov_b32 v13, v16 // (op 3468) convert

s_mul_i32 s1, s32, s7 // (op 3470) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 3470) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 3470) least significant
s_mul_hi_u32 s19, s32, s6 // (op 3470) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3470) most significant: sum
s_add_u32 s1, s1, s19 // (op 3470) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3470) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 3470) 

s_mov_b32 s64, s34 // (op 3470) reg expression
s_mov_b32 s65, s35 // (op 3470) reg expression
s_mov_b32 s66, s0 // (op 3470) reg expression
s_mov_b32 s67, 131072 // (op 3470) call()

s_mul_hi_i32 s0, s16, s10 // (op 3532) 
v_lshrrev_b32 v14, 7, v1 // (op 3532) 
s_add_i32 s1, s0, s16 // (op 3532) 

v_and_b32 v15, 15, v1 // (op 3532) 
s_ashr_i32 s0, s1, 31 // (op 3532) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 3532) 


s_cmp_eq_i32 s10, 0 // (op 3532) 
s_mov_b32 s18, scc // (op 3532) 
v_and_b32 v14, 63, v1 // (op 3532) 
s_cmp_eq_i32 s18, 0 // (op 3532) 
s_cselect_b32 s19, 0, -1 // (op 3532) 

v_lshrrev_b32 v15, 4, v14 // (op 3532) 

s_and_b32 s18, s11, 31 // (op 3532) 
s_lshl_b32 s22, 1, s18 // (op 3532) 
s_add_i32 s23, s22, s19 // (op 3532) 


s_and_b32 s19, s0, s23 // (op 3532) 


s_add_i32 s0, s1, s19 // (op 3532) 


s_ashr_i32 s1, s0, s18 // (op 3532) 


s_lshl_b32 s0, s11, 24 // (op 3532) 
s_ashr_i32 s18, s0, 31 // (op 3532) 

s_xor_b32 s0, s1, s18 // (op 3532) 

s_sub_i32 s1, s0, s18 // (op 3532) 


s_mul_hi_i32 s0, s1, s12 // (op 3532) 
s_add_i32 s18, s0, s1 // (op 3532) 

s_ashr_i32 s0, s18, 31 // (op 3532) 
s_cmp_eq_i32 s12, 0 // (op 3532) 
s_mov_b32 s19, scc // (op 3532) 
s_cmp_eq_i32 s19, 0 // (op 3532) 
s_cselect_b32 s22, 0, -1 // (op 3532) 

s_and_b32 s19, s13, 31 // (op 3532) 
s_lshl_b32 s23, 1, s19 // (op 3532) 
s_add_i32 s24, s23, s22 // (op 3532) 


s_and_b32 s25, s0, s24 // (op 3532) 


s_add_i32 s0, s18, s25 // (op 3532) 


s_ashr_i32 s18, s0, s19 // (op 3532) 


s_lshl_b32 s0, s13, 24 // (op 3532) 
s_ashr_i32 s19, s0, 31 // (op 3532) 

s_xor_b32 s0, s18, s19 // (op 3532) 

s_sub_i32 s18, s0, s19 // (op 3532) 


s_mul_i32 s0, s1, s15 // (op 3532) 

s_sub_u32 s1, s16, s0 // (op 3532) 

s_mul_i32 s0, s18, s15 // (op 3532) 
s_add_u32 s0, s0, s1 // (op 3532) 


s_mul_hi_i32 s1, s16, s9 // (op 3532) 
s_add_i32 s18, s1, s16 // (op 3532) 

s_ashr_i32 s1, s18, 31 // (op 3532) 
s_cmp_eq_i32 s9, 0 // (op 3532) 
s_mov_b32 s19, scc // (op 3532) 
s_cmp_eq_i32 s19, 0 // (op 3532) 
s_cselect_b32 s22, 0, -1 // (op 3532) 

s_and_b32 s19, s46, 31 // (op 3532) 
s_lshl_b32 s23, 1, s19 // (op 3532) 
s_add_i32 s24, s23, s22 // (op 3532) 


s_and_b32 s25, s1, s24 // (op 3532) 


s_add_i32 s1, s18, s25 // (op 3532) 


s_ashr_i32 s18, s1, s19 // (op 3532) 


s_lshl_b32 s1, s46, 24 // (op 3532) 
s_ashr_i32 s19, s1, 31 // (op 3532) 

s_xor_b32 s1, s18, s19 // (op 3532) 

s_sub_i32 s18, s1, s19 // (op 3532) 


s_add_u32 s50, s6, 256 // (op 3532) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3532) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3532) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3532) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3532) 

s_mul_i32 s1, s15, s24 // (op 3532) 

s_mul_i32 s19, s8, s1 // (op 3532) 

s_mul_i32 s1, s18, s19 // (op 3532) 

s_sub_u32 s19, s16, s1 // (op 3532) 

s_mov_b32 s1, s19 // (op 3532) 
s_mul_hi_i32 s22, s1, s14 // (op 3532) 
s_add_i32 s23, s22, s1 // (op 3532) 


s_ashr_i32 s1, s23, 31 // (op 3532) 
s_cmp_eq_i32 s14, 0 // (op 3532) 
s_mov_b32 s22, scc // (op 3532) 
s_cmp_eq_i32 s22, 0 // (op 3532) 
s_cselect_b32 s24, 0, -1 // (op 3532) 

s_and_b32 s22, s48, 31 // (op 3532) 
s_lshl_b32 s25, 1, s22 // (op 3532) 
s_add_i32 s49, s25, s24 // (op 3532) 


s_and_b32 s24, s1, s49 // (op 3532) 


s_add_i32 s1, s23, s24 // (op 3532) 


s_ashr_i32 s23, s1, s22 // (op 3532) 


s_lshl_b32 s1, s48, 24 // (op 3532) 
s_ashr_i32 s22, s1, 31 // (op 3532) 

s_xor_b32 s1, s23, s22 // (op 3532) 

s_sub_i32 s23, s1, s22 // (op 3532) 


s_mul_i32 s1, s47, s15 // (op 3532) 
s_add_u32 s24, s20, 32 // (op 3532) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 3532) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3532) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3532) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 3532) 

s_sub_i32 s22, s24, s1 // (op 3532) 

s_mul_i32 s24, s23, s22 // (op 3532) 


s_sub_u32 s25, s19, s24 // (op 3532) 


s_add_u32 s19, s25, s1 // (op 3532) 


s_cmp_eq_i32 s18, 0 // (op 3532) 
s_mov_b32 s1, scc // (op 3532) 

s_cmp_eq_i32 s1, 0 // (op 3532) 
s_cselect_b32 s18, s19, s0 // (op 3532) 



v_lshl_add_u32 v14, s18, 5, v16 // (op 3532) 


s_ashr_i64 s[0:1], s[30:31], 7 // (op 3532) 
s_lshr_b32 s18, s0, 1 // (op 3532) 

v_lshl_add_u32 v16, s18, 3, v15 // (op 3532) 


v_mul_lo_u32 v15, v16, s36 // (op 3532) 

v_add_u32 v16, v14, v15 // (op 3532) 


s_mul_i32 s1, s36, s41 // (op 3534) most significant: low of low * high
s_mul_i32 s18, s37, s40 // (op 3534) most significant: low of high * low
s_mul_i32 s0, s36, s40 // (op 3534) least significant
s_mul_hi_u32 s19, s36, s40 // (op 3534) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3534) most significant: sum
s_add_u32 s1, s1, s19 // (op 3534) most significant: sum


s_mov_b32 s68, s38 // (op 3534) reg expression
s_mov_b32 s69, s39 // (op 3534) reg expression
s_mov_b32 s70, s0 // (op 3534) reg expression
s_mov_b32 s71, 131072 // (op 3534) call()

s_mul_hi_i32 s0, s16, s10 // (op 3596) 
s_add_i32 s1, s0, s16 // (op 3596) 

s_ashr_i32 s0, s1, 31 // (op 3596) 
s_cmp_eq_i32 s10, 0 // (op 3596) 
s_mov_b32 s18, scc // (op 3596) 
s_cmp_eq_i32 s18, 0 // (op 3596) 
s_cselect_b32 s19, 0, -1 // (op 3596) 

s_and_b32 s18, s11, 31 // (op 3596) 
s_lshl_b32 s22, 1, s18 // (op 3596) 
s_add_i32 s23, s22, s19 // (op 3596) 


s_and_b32 s19, s0, s23 // (op 3596) 


s_add_i32 s0, s1, s19 // (op 3596) 


s_ashr_i32 s1, s0, s18 // (op 3596) 


s_lshl_b32 s0, s11, 24 // (op 3596) 
s_ashr_i32 s18, s0, 31 // (op 3596) 

s_xor_b32 s0, s1, s18 // (op 3596) 

s_sub_i32 s1, s0, s18 // (op 3596) 


s_mul_hi_i32 s0, s1, s12 // (op 3596) 
s_add_i32 s18, s0, s1 // (op 3596) 

s_ashr_i32 s0, s18, 31 // (op 3596) 
s_cmp_eq_i32 s12, 0 // (op 3596) 
s_mov_b32 s19, scc // (op 3596) 
s_cmp_eq_i32 s19, 0 // (op 3596) 
s_cselect_b32 s22, 0, -1 // (op 3596) 

s_and_b32 s19, s13, 31 // (op 3596) 
s_lshl_b32 s23, 1, s19 // (op 3596) 
s_add_i32 s24, s23, s22 // (op 3596) 


s_and_b32 s25, s0, s24 // (op 3596) 


s_add_i32 s0, s18, s25 // (op 3596) 


s_ashr_i32 s18, s0, s19 // (op 3596) 


s_lshl_b32 s0, s13, 24 // (op 3596) 
s_ashr_i32 s19, s0, 31 // (op 3596) 

s_xor_b32 s0, s18, s19 // (op 3596) 

s_sub_i32 s18, s0, s19 // (op 3596) 


s_add_u32 s50, s6, 256 // (op 3596) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 3596) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 3596) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 3596) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 8 // (op 3596) 

s_mul_i32 s0, s18, s24 // (op 3596) 

s_sub_i32 s18, s1, s0 // (op 3596) 


s_mul_hi_i32 s19, s16, s9 // (op 3596) 
s_add_i32 s0, s19, s16 // (op 3596) 

s_ashr_i32 s1, s0, 31 // (op 3596) 
s_cmp_eq_i32 s9, 0 // (op 3596) 
s_mov_b32 s19, scc // (op 3596) 
s_cmp_eq_i32 s19, 0 // (op 3596) 
s_cselect_b32 s22, 0, -1 // (op 3596) 

s_and_b32 s19, s46, 31 // (op 3596) 
s_lshl_b32 s23, 1, s19 // (op 3596) 
s_add_i32 s49, s23, s22 // (op 3596) 


s_and_b32 s22, s1, s49 // (op 3596) 


s_add_i32 s1, s0, s22 // (op 3596) 


s_ashr_i32 s0, s1, s19 // (op 3596) 


s_lshl_b32 s1, s46, 24 // (op 3596) 
s_ashr_i32 s19, s1, 31 // (op 3596) 

s_xor_b32 s1, s0, s19 // (op 3596) 

s_sub_i32 s0, s1, s19 // (op 3596) 


s_mul_i32 s1, s15, s24 // (op 3596) 

s_mul_i32 s19, s8, s1 // (op 3596) 

s_mul_i32 s1, s0, s19 // (op 3596) 

s_sub_u32 s19, s16, s1 // (op 3596) 

s_mul_hi_i32 s1, s19, s14 // (op 3596) 
s_add_i32 s22, s1, s19 // (op 3596) 


s_ashr_i32 s1, s22, 31 // (op 3596) 
s_cmp_eq_i32 s14, 0 // (op 3596) 
s_mov_b32 s19, scc // (op 3596) 
s_cmp_eq_i32 s19, 0 // (op 3596) 
s_cselect_b32 s23, 0, -1 // (op 3596) 

s_and_b32 s19, s48, 31 // (op 3596) 
s_lshl_b32 s24, 1, s19 // (op 3596) 
s_add_i32 s25, s24, s23 // (op 3596) 


s_and_b32 s23, s1, s25 // (op 3596) 


s_add_i32 s1, s22, s23 // (op 3596) 


s_ashr_i32 s22, s1, s19 // (op 3596) 


s_lshl_b32 s1, s48, 24 // (op 3596) 
s_ashr_i32 s19, s1, 31 // (op 3596) 

s_xor_b32 s1, s22, s19 // (op 3596) 

s_sub_i32 s22, s1, s19 // (op 3596) 


s_cmp_eq_i32 s0, 0 // (op 3596) 
s_mov_b32 s1, scc // (op 3596) 

s_cmp_eq_i32 s1, 0 // (op 3596) 
s_cselect_b32 s0, s22, s18 // (op 3596) 



v_lshl_add_u32 v14, s0, 8, v1 // (op 3596) 

v_mul_lo_u32 v15, v14, s42 // (op 3596) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 3596) 
s_lshr_b32 s18, s0, 1 // (op 3596) 

v_lshl_add_u32 v14, s18, 3, v15 // (op 3596) 


s_mul_i32 s1, s42, s7 // (op 3598) most significant: low of low * high
s_mul_i32 s18, s43, s6 // (op 3598) most significant: low of high * low
s_mul_i32 s0, s42, s6 // (op 3598) least significant
s_mul_hi_u32 s19, s42, s6 // (op 3598) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3598) most significant: sum
s_add_u32 s1, s1, s19 // (op 3598) most significant: sum


s_mov_b32 s72, s44 // (op 3598) reg expression
s_mov_b32 s73, s45 // (op 3598) reg expression
s_mov_b32 s74, s0 // (op 3598) reg expression
s_mov_b32 s75, 131072 // (op 3598) call()

v_and_b32 v15, -2, v1 // (op 3689) 
v_sub_u32 v17, v1, v15 // (op 3689) 

v_lshlrev_b32 v15, 4, v17 // (op 3689) 

v_lshrrev_b32 v17, 1, v1 // (op 3689) 
v_lshl_add_u32 v18, v17, 5, v15 // (op 3689) 


v_and_b32 v15, -4, v1 // (op 3742) 
v_sub_u32 v17, v1, v15 // (op 3742) 

v_lshlrev_b32 v15, 5, v17 // (op 3742) 

v_lshrrev_b32 v17, 2, v1 // (op 3742) 
v_lshl_add_u32 v19, v17, 9, v15 // (op 3742) 


v_lshlrev_b32 v15, 2, v19 // (op 3742) 

v_lshrrev_b32 v17, 3, v15 // (op 3742) 

v_lshlrev_b32 v15, 2, v1 // (op 3800) 
s_ashr_i64 s[0:1], s[30:31], 7 // (op 1249) 
s_and_b32 s18, s0, 4294967294 // (op 1249) 

s_ashr_i64 s[0:1], s[30:31], 7 // (op 1248) 
s_mov_b32 s19, s0 // (op 1248) convert

s_cmp_lt_u32 s18, s19 // (op 1248) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1248 // (op 1248) Condition: Top (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1248 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1248:


s_add_u32 s18, s18, 1 // (op 1250) 
v_mov_b32 v12, v47 // (op 166) 
v_mov_b32 v19, v12 // (op 166) reg expression

buffer_load_dwordx4 v[20:23], v19, s[52:55], 0 offen // (op 166) Load value


s_lshl_b32 s19, s26, 7 // (op 3384) 
v_add_u32 v47, v47, s19 // (op 3384) 


s_barrier  // (op 3998) 
s_lshl_b64 s[0:1], s[32:33], 2 // (op 179) 



v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 179) 


v_mov_b32 v12, v13 // (op 179) 
v_mov_b32 v19, v12 // (op 179) 
buffer_load_dwordx4 v[26:29], v19, s[64:67], 0 offen // (op 179) Load value
v_add_u32 v19, v19, v24 // (op 179) 
buffer_load_dwordx4 v[30:33], v19, s[64:67], 0 offen // (op 179) Load value
v_add_u32 v19, v19, v24 // (op 179) 
s_waitcnt vmcnt(2)
ds_write_b128 v18, v[20:23] offset:17408 // (op 167) Store local data 
buffer_load_dwordx4 v[34:37], v19, s[64:67], 0 offen // (op 179) Load value

v_add_u32 v19, v19, v24 // (op 179) 
buffer_load_dwordx4 v[38:41], v19, s[64:67], 0 offen // (op 179) Load value



v_add_u32 v13, 64, v13 // (op 3460) 


s_barrier  // (op 4011) 
v_mov_b32 v12, v14 // (op 180) 




buffer_load_dword v19, v12, s[72:75], 0 offen // (op 180) Load value

v_add_u32 v14, 4, v14 // (op 3588) 
s_waitcnt vmcnt(4)
ds_write_b128 v17, v[26:29] // (op 177) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4001) 
s_waitcnt vmcnt(3)
ds_write_b128 v17, v[30:33] offset:64 // (op 177) Store local data 



ds_read_b64_tr_b8 v[48:49], v0 offset:17408 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:17664 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:19456 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:19712 // (op 169) Transpose load local data 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[34:37] offset:128 // (op 177) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v17, v[38:41] offset:192 // (op 177) Store local data 

s_barrier  // (op 4025) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4015) 






ds_read_b128 v[20:23], v7 // (op 171) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v15, v19 offset:16384 // (op 178) Store local data 


v_mov_b64 v[24:25], s[36:37] // (op 173) convert

v_mov_b32 v12, v16 // (op 173) 
v_mov_b32 v19, v12 // (op 173) reg expression
v_mov_b32 v26, v19 // (op 173) 
buffer_load_ubyte v27, v26, s[68:71], 0 offen offset: 0 // (op 173) Load value




s_lshl_b32 s19, s36, 2 // (op 3524) 
v_add_u32 v16, v16, s19 // (op 3524) 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 4029) 



ds_read_u8 v12, v4 offset:16384 // (op 175) Load local data 



ds_read_b128 v[28:31], v7 offset:2048 // (op 320) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[20:23], a[0:3], v27, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 176) 





ds_read_u8 v12, v4 offset:16512 // (op 323) Load local data 



ds_read_b128 v[32:35], v7 offset:4096 // (op 476) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[28:31], a[4:7], v27, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 324) 





ds_read_u8 v12, v4 offset:16640 // (op 479) Load local data 



ds_read_b128 v[28:31], v7 offset:6144 // (op 632) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:55], v[32:35], a[8:11], v27, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 480) 





ds_read_u8 v12, v4 offset:16768 // (op 635) Load local data 



ds_read_b128 v[32:35], v7 offset:8192 // (op 788) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:55], v[28:31], a[12:15], v27, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 636) 





ds_read_u8 v12, v4 offset:16896 // (op 791) Load local data 



ds_read_b128 v[28:31], v7 offset:10240 // (op 944) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:55], v[32:35], a[16:19], v27, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 792) 





ds_read_u8 v12, v4 offset:17024 // (op 947) Load local data 



ds_read_b128 v[32:35], v7 offset:12288 // (op 1100) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:55], v[28:31], a[20:23], v27, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 948) 





ds_read_u8 v12, v4 offset:17152 // (op 1103) Load local data 



ds_read_b128 v[28:31], v7 offset:14336 // (op 1256) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:55], v[32:35], a[24:27], v27, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1104) 





ds_read_u8 v12, v4 offset:17280 // (op 1259) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:55], v[28:31], a[28:31], v27, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1260) 




s_ashr_i64 s[0:1], s[30:31], 7 // (op 1248) 
s_mov_b32 s19, s0 // (op 1248) convert

s_cmp_lt_u32 s18, s19 // (op 1248) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1248 // (op 1248) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1248 if true)
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1248:






















s_barrier  // (op 4143) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalFalse_0_beta_123:


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
RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_
.LRR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__end:
.size RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_, .LRR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM__end-RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 120
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 43008 // lds bytes
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
    DataType: B8
    DataTypeA: B8
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
    macrotile: [32, 256, 128]
    threads: [256, 1, 1]
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
  - .name: RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 43008
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 88
    .agpr_count: 32
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

