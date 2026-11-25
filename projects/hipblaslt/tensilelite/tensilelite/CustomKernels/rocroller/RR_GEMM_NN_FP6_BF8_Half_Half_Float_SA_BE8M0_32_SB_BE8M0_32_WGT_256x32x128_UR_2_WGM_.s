// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_,@function
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s8 // (op 3856) 
s_mul_hi_i32 s1, s16, s8 // (op 4044) 
v_lshrrev_b32 v0, 6, v1 // (op 3970) 
v_and_b32 v3, 1, v0 // (op 3970) 

v_and_b32 v0, 15, v1 // (op 3970) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3970) 


v_and_b32 v0, 48, v1 // (op 3970) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 3970) 


v_lshrrev_b32 v0, 6, v1 // (op 4114) 
v_and_b32 v4, 1, v0 // (op 4114) 

v_and_b32 v0, 15, v1 // (op 4114) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4114) 


v_and_b32 v0, 48, v1 // (op 4114) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4114) 


v_lshrrev_b32 v0, 7, v1 // (op 3856) 
s_add_i32 s18, s0, s16 // (op 3856) 

v_and_b32 v5, 63, v1 // (op 3856) 
s_ashr_i32 s0, s18, 31 // (op 3856) 
v_lshrrev_b32 v6, 4, v5 // (op 3856) 
s_cmp_eq_i32 s8, 0 // (op 3856) 
s_mov_b32 s19, scc // (op 3856) 
v_lshlrev_b32 v7, 2, v6 // (op 3856) 

s_cmp_eq_i32 s19, 0 // (op 3856) 
s_cselect_b32 s51, 0, -1 // (op 3856) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3856) 


s_and_b32 s19, s9, 31 // (op 3856) 
v_lshrrev_b32 v0, 2, v5 // (op 3856) 

v_and_b32 v5, 3, v0 // (op 3856) 

v_and_b32 v0, 63, v1 // (op 3918) 
v_and_b32 v7, 48, v1 // (op 3918) 
v_sub_u32 v8, v0, v7 // (op 3918) 

v_lshrrev_b32 v7, 4, v0 // (op 3918) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 3918) 


v_lshrrev_b32 v7, 7, v1 // (op 3918) 
v_lshlrev_b32 v8, 4, v7 // (op 3918) 

v_lshl_add_u32 v7, v0, 8, v8 // (op 3918) 


v_mul_lo_u32 v0, v7, 6 // (op 3918) 
v_lshrrev_b32 v8, 3, v0 // (op 3918) 

v_lshrrev_b32 v0, 4, v7 // (op 3918) 

v_lshlrev_b32 v7, 2, v0 // (op 3918) 

v_add_u32 v0, v8, v7 // (op 3918) 


v_lshrrev_b32 v7, 7, v1 // (op 4028) 
v_and_b32 v8, 15, v1 // (op 4028) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 4028) 


v_and_b32 v7, 63, v1 // (op 4028) 
v_lshrrev_b32 v8, 4, v7 // (op 4028) 

v_lshl_add_u32 v7, v8, 8, v9 // (op 4028) 


v_lshrrev_b32 v8, 7, v1 // (op 4044) 
v_and_b32 v9, 63, v1 // (op 4044) 
v_and_b32 v10, 63, v1 // (op 4075) 
v_and_b32 v11, 48, v1 // (op 4075) 
v_sub_u32 v12, v10, v11 // (op 4075) 

v_lshrrev_b32 v11, 4, v10 // (op 4075) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 4075) 


v_lshrrev_b32 v11, 7, v1 // (op 4075) 
v_lshlrev_b32 v12, 4, v11 // (op 4075) 

v_lshl_add_u32 v11, v10, 8, v12 // (op 4075) 


v_mul_lo_u32 v10, v11, 6 // (op 4075) 
v_lshrrev_b32 v12, 3, v10 // (op 4075) 

v_lshrrev_b32 v10, 4, v11 // (op 4075) 

v_lshlrev_b32 v11, 2, v10 // (op 4075) 

v_add_u32 v10, v12, v11 // (op 4075) 


v_lshrrev_b32 v11, 7, v1 // (op 4143) 
v_and_b32 v12, 15, v1 // (op 4143) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 4143) 


v_and_b32 v11, 63, v1 // (op 4143) 
v_lshrrev_b32 v12, 4, v11 // (op 4143) 

v_lshl_add_u32 v11, v12, 8, v13 // (op 4143) 


s_lshl_b32 s52, 1, s19 // (op 3856) 
v_and_b32 v12, 3, v1 // (op 3856) 
s_add_i32 s53, s52, s51 // (op 3856) 


s_and_b32 s51, s0, s53 // (op 3856) 


s_add_i32 s0, s1, s16 // (op 4044) 

s_ashr_i32 s1, s0, 31 // (op 4044) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 3856) 


s_add_i32 s52, s18, s51 // (op 3856) 


v_lshrrev_b32 v5, 6, v1 // (op 3856) 
s_ashr_i32 s18, s52, s19 // (op 3856) 


v_and_b32 v12, 1, v5 // (op 3856) 

s_lshl_b32 s19, s9, 24 // (op 3856) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 3856) 


s_ashr_i32 s51, s19, 31 // (op 3856) 

s_xor_b32 s19, s18, s51 // (op 3856) 

s_sub_i32 s18, s19, s51 // (op 3856) 


s_mul_hi_i32 s19, s18, s10 // (op 3856) 
s_add_i32 s51, s19, s18 // (op 3856) 

s_ashr_i32 s19, s51, 31 // (op 3856) 
s_cmp_eq_i32 s10, 0 // (op 3856) 
s_mov_b32 s52, scc // (op 3856) 
s_cmp_eq_i32 s52, 0 // (op 3856) 
s_cselect_b32 s53, 0, -1 // (op 3856) 

s_and_b32 s52, s11, 31 // (op 3856) 
v_lshrrev_b32 v12, 4, v9 // (op 4044) 
s_cmp_eq_i32 s8, 0 // (op 4044) 
s_mov_b32 s54, scc // (op 4044) 
v_lshlrev_b32 v13, 2, v12 // (op 4044) 

s_cmp_eq_i32 s54, 0 // (op 4044) 
s_cselect_b32 s55, 0, -1 // (op 4044) 

s_lshl_b32 s54, 1, s52 // (op 3856) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 4044) 


v_lshrrev_b32 v8, 2, v9 // (op 4044) 

s_add_i32 s56, s54, s53 // (op 3856) 


s_and_b32 s53, s19, s56 // (op 3856) 


s_add_i32 s19, s51, s53 // (op 3856) 


s_ashr_i32 s51, s19, s52 // (op 3856) 


s_lshl_b32 s19, s11, 24 // (op 3856) 
s_ashr_i32 s52, s19, 31 // (op 3856) 

s_xor_b32 s19, s51, s52 // (op 3856) 

s_sub_i32 s51, s19, s52 // (op 3856) 


s_mul_i32 s19, s18, s13 // (op 3856) 
s_sub_u32 s52, s16, s19 // (op 3856) 

s_mul_i32 s19, s51, s13 // (op 3856) 
s_add_u32 s19, s19, s52 // (op 3856) 

s_mul_hi_i32 s52, s16, s7 // (op 3856) 
s_add_i32 s53, s52, s16 // (op 3856) 

s_ashr_i32 s52, s53, 31 // (op 3856) 
s_cmp_eq_i32 s7, 0 // (op 3856) 
s_mov_b32 s54, scc // (op 3856) 
s_cmp_eq_i32 s54, 0 // (op 3856) 
s_cselect_b32 s56, 0, -1 // (op 3856) 

s_and_b32 s54, s14, 31 // (op 3856) 
s_lshl_b32 s57, 1, s54 // (op 3856) 
s_add_i32 s58, s57, s56 // (op 3856) 


s_and_b32 s56, s52, s58 // (op 3856) 


s_add_i32 s52, s53, s56 // (op 3856) 


s_ashr_i32 s53, s52, s54 // (op 3856) 


s_lshl_b32 s52, s14, 24 // (op 3856) 
s_ashr_i32 s54, s52, 31 // (op 3856) 

s_xor_b32 s52, s53, s54 // (op 3856) 

s_sub_i32 s53, s52, s54 // (op 3856) 


s_and_b32 s52, s9, 31 // (op 4044) 
s_lshl_b32 s54, 1, s52 // (op 4044) 
v_and_b32 v9, 3, v8 // (op 4044) 

s_add_i32 s56, s54, s55 // (op 4044) 


v_and_b32 v8, 3, v1 // (op 4044) 
s_and_b32 s54, s1, s56 // (op 4044) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 4044) 


s_add_i32 s1, s0, s54 // (op 4044) 


v_lshrrev_b32 v8, 6, v1 // (op 4044) 
s_ashr_i32 s0, s1, s52 // (op 4044) 


v_and_b32 v9, 1, v8 // (op 4044) 

s_lshl_b32 s1, s9, 24 // (op 4044) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 4044) 


s_ashr_i32 s52, s1, 31 // (op 4044) 

s_xor_b32 s1, s0, s52 // (op 4044) 

s_sub_i32 s0, s1, s52 // (op 4044) 


s_mul_hi_i32 s1, s0, s10 // (op 4044) 
s_add_i32 s52, s1, s0 // (op 4044) 

s_ashr_i32 s1, s52, 31 // (op 4044) 
s_cmp_eq_i32 s10, 0 // (op 4044) 
s_mov_b32 s54, scc // (op 4044) 
s_cmp_eq_i32 s54, 0 // (op 4044) 
s_cselect_b32 s55, 0, -1 // (op 4044) 

s_and_b32 s54, s11, 31 // (op 4044) 
s_lshl_b32 s56, 1, s54 // (op 4044) 
s_add_i32 s57, s56, s55 // (op 4044) 


s_and_b32 s55, s1, s57 // (op 4044) 


s_add_i32 s1, s52, s55 // (op 4044) 


s_ashr_i32 s52, s1, s54 // (op 4044) 


s_lshl_b32 s1, s11, 24 // (op 4044) 
s_ashr_i32 s54, s1, 31 // (op 4044) 

s_xor_b32 s1, s52, s54 // (op 4044) 

s_sub_i32 s52, s1, s54 // (op 4044) 


s_mul_i32 s1, s0, s13 // (op 4044) 
s_sub_u32 s54, s16, s1 // (op 4044) 

s_mul_i32 s1, s52, s13 // (op 4044) 
s_add_u32 s1, s1, s54 // (op 4044) 

s_mul_hi_i32 s54, s16, s7 // (op 4044) 
s_add_i32 s55, s54, s16 // (op 4044) 

s_ashr_i32 s54, s55, 31 // (op 4044) 
s_cmp_eq_i32 s7, 0 // (op 4044) 
s_mov_b32 s56, scc // (op 4044) 
s_cmp_eq_i32 s56, 0 // (op 4044) 
s_cselect_b32 s57, 0, -1 // (op 4044) 

s_and_b32 s56, s14, 31 // (op 4044) 
s_lshl_b32 s58, 1, s56 // (op 4044) 
s_add_i32 s59, s58, s57 // (op 4044) 


s_and_b32 s57, s54, s59 // (op 4044) 


s_add_i32 s54, s55, s57 // (op 4044) 


s_ashr_i32 s55, s54, s56 // (op 4044) 


s_lshl_b32 s54, s14, 24 // (op 4044) 
s_ashr_i32 s56, s54, 31 // (op 4044) 

s_xor_b32 s54, s55, s56 // (op 4044) 

s_sub_i32 s55, s54, s56 // (op 4044) 


s_add_u32 s56, s4, 32 // (op 3856) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 3856) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3856) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3856) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 3856) 

s_mul_i32 s54, s13, s56 // (op 3856) 
s_mul_i32 s58, s6, s54 // (op 3856) 

s_mul_i32 s54, s53, s58 // (op 3856) 

s_sub_u32 s58, s16, s54 // (op 3856) 

s_mov_b32 s54, s58 // (op 3856) 
s_mul_hi_i32 s59, s54, s12 // (op 3856) 
s_add_i32 s60, s59, s54 // (op 3856) 


s_ashr_i32 s54, s60, 31 // (op 3856) 
s_cmp_eq_i32 s12, 0 // (op 3856) 
s_mov_b32 s59, scc // (op 3856) 
s_cmp_eq_i32 s59, 0 // (op 3856) 
s_cselect_b32 s61, 0, -1 // (op 3856) 

s_add_u32 s62, s4, 32 // (op 4044) Start of Int64 add, locking SCC
s_addc_u32 s63, s5, 0 // (op 4044) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 4044) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 4044) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 4044) 

s_mul_i32 s59, s13, s62 // (op 4044) 
s_mul_i32 s64, s6, s59 // (op 4044) 

s_mul_i32 s59, s55, s64 // (op 4044) 

s_sub_u32 s64, s16, s59 // (op 4044) 

s_mov_b32 s59, s64 // (op 4044) 
s_mul_hi_i32 s65, s59, s12 // (op 4044) 
s_add_i32 s66, s65, s59 // (op 4044) 


s_ashr_i32 s59, s66, 31 // (op 4044) 
s_cmp_eq_i32 s12, 0 // (op 4044) 
s_mov_b32 s65, scc // (op 4044) 
s_cmp_eq_i32 s65, 0 // (op 4044) 
s_cselect_b32 s67, 0, -1 // (op 4044) 


s_waitcnt lgkmcnt(0)
s_and_b32 s65, s48, 31 // (op 3856) 
s_lshl_b32 s68, 1, s65 // (op 3856) 
s_add_i32 s69, s68, s61 // (op 3856) 


s_and_b32 s61, s54, s69 // (op 3856) 


s_add_i32 s54, s60, s61 // (op 3856) 


s_ashr_i32 s60, s54, s65 // (op 3856) 


s_lshl_b32 s54, s48, 24 // (op 3856) 
s_ashr_i32 s61, s54, 31 // (op 3856) 

s_xor_b32 s54, s60, s61 // (op 3856) 

s_sub_i32 s60, s54, s61 // (op 3856) 


s_mul_i32 s54, s15, s13 // (op 3856) 
s_and_b32 s61, s48, 31 // (op 4044) 
s_lshl_b32 s65, 1, s61 // (op 4044) 
s_add_i32 s68, s65, s67 // (op 4044) 


s_and_b32 s65, s59, s68 // (op 4044) 


s_add_i32 s59, s66, s65 // (op 4044) 


s_ashr_i32 s65, s59, s61 // (op 4044) 


s_lshl_b32 s59, s48, 24 // (op 4044) 
s_ashr_i32 s61, s59, 31 // (op 4044) 

s_xor_b32 s59, s65, s61 // (op 4044) 

s_sub_i32 s65, s59, s61 // (op 4044) 


s_mul_i32 s59, s15, s13 // (op 4044) 
s_add_u32 s66, s20, 256 // (op 3856) Start of Int64 add, locking SCC
s_addc_u32 s67, s21, 0 // (op 3856) End of Int64 add, unlocking SCC
s_sub_u32 s68, s66, 1 // (op 3856) least significant half; sets scc
s_subb_u32 s69, s67, 0 // (op 3856) most significant half; uses scc

s_ashr_i64 s[66:67], s[68:69], 8 // (op 3856) 

s_sub_i32 s61, s66, s54 // (op 3856) 

s_mul_i32 s66, s60, s61 // (op 3856) 

s_sub_u32 s61, s58, s66 // (op 3856) 


s_add_u32 s58, s61, s54 // (op 3856) 


s_cmp_eq_i32 s53, 0 // (op 3856) 
s_mov_b32 s54, scc // (op 3856) 

s_cmp_eq_i32 s54, 0 // (op 3856) 
s_cselect_b32 s53, s58, s19 // (op 3856) 


s_mul_i32 s19, s51, s56 // (op 3856) 


v_lshl_add_u32 v9, s53, 8, v6 // (op 3856) 


s_sub_i32 s51, s18, s19 // (op 3856) 


s_cmp_eq_i32 s54, 0 // (op 3856) 
s_cselect_b32 s53, s60, s51 // (op 3856) 



v_lshl_add_u32 v6, s53, 5, v5 // (op 3856) 


v_mul_lo_u32 v5, v6, s2 // (op 3856) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 3856) 


s_add_u32 s18, s20, 256 // (op 4044) Start of Int64 add, locking SCC
s_addc_u32 s19, s21, 0 // (op 4044) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 4044) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 4044) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 8 // (op 4044) 

s_sub_i32 s51, s18, s59 // (op 4044) 

s_mul_i32 s18, s65, s51 // (op 4044) 

s_sub_u32 s19, s64, s18 // (op 4044) 


s_add_u32 s18, s19, s59 // (op 4044) 


s_cmp_eq_i32 s55, 0 // (op 4044) 
s_mov_b32 s19, scc // (op 4044) 

s_cmp_eq_i32 s19, 0 // (op 4044) 
s_cselect_b32 s51, s18, s1 // (op 4044) 


s_mul_i32 s1, s52, s62 // (op 4044) 


v_lshl_add_u32 v5, s51, 8, v12 // (op 4044) 


s_sub_i32 s18, s0, s1 // (op 4044) 


s_cmp_eq_i32 s19, 0 // (op 4044) 
s_cselect_b32 s0, s65, s18 // (op 4044) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 4044) 


v_mul_lo_u32 v8, v9, s22 // (op 4044) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 4044) 


s_mul_i32 s1, s2, s5 // (op 3858) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 3858) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 3858) least significant
s_mul_hi_u32 s19, s2, s4 // (op 3858) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3858) most significant: sum
s_add_u32 s1, s1, s19 // (op 3858) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3858) 

s_mul_i32 s1, s22, s5 // (op 4046) most significant: low of low * high
s_mul_i32 s51, s23, s4 // (op 4046) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 4046) least significant
s_mul_hi_u32 s52, s22, s4 // (op 4046) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 4046) most significant: sum
s_add_u32 s1, s1, s52 // (op 4046) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 4046) 

s_mov_b32 s56, s24 // (op 3858) reg expression
s_mov_b32 s57, s25 // (op 3858) reg expression
s_mov_b32 s58, s18 // (op 3858) reg expression
s_mov_b32 s59, 131072 // (op 3858) call()

s_mov_b32 s60, s26 // (op 4046) reg expression

s_mov_b32 s61, s27 // (op 4046) reg expression
s_mov_b32 s62, s52 // (op 4046) reg expression
s_mov_b32 s63, 131072 // (op 4046) call()


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
v_accvgpr_write a8, 0.00000 // (op 475) call()
v_accvgpr_write a9, 0.00000 // (op 475) call()
v_accvgpr_write a10, 0.00000 // (op 475) call()
v_accvgpr_write a11, 0.00000 // (op 475) call()
v_accvgpr_write a12, 0.00000 // (op 637) call()
v_accvgpr_write a13, 0.00000 // (op 637) call()
v_accvgpr_write a14, 0.00000 // (op 637) call()
v_accvgpr_write a15, 0.00000 // (op 637) call()
v_accvgpr_write a16, 0.00000 // (op 799) call()
v_accvgpr_write a17, 0.00000 // (op 799) call()
v_accvgpr_write a18, 0.00000 // (op 799) call()
v_accvgpr_write a19, 0.00000 // (op 799) call()
v_accvgpr_write a20, 0.00000 // (op 961) call()
v_accvgpr_write a21, 0.00000 // (op 961) call()
v_accvgpr_write a22, 0.00000 // (op 961) call()
v_accvgpr_write a23, 0.00000 // (op 961) call()
v_accvgpr_write a24, 0.00000 // (op 1123) call()
v_accvgpr_write a25, 0.00000 // (op 1123) call()
v_accvgpr_write a26, 0.00000 // (op 1123) call()
v_accvgpr_write a27, 0.00000 // (op 1123) call()
v_accvgpr_write a28, 0.00000 // (op 1285) call()
v_accvgpr_write a29, 0.00000 // (op 1285) call()
v_accvgpr_write a30, 0.00000 // (op 1285) call()
v_accvgpr_write a31, 0.00000 // (op 1285) call()
s_mul_hi_i32 s0, s16, s8 // (op 3623) 
v_and_b32 v12, -4, v1 // (op 3623) 
s_add_i32 s1, s0, s16 // (op 3623) 

v_sub_u32 v13, v1, v12 // (op 3623) 

s_ashr_i32 s0, s1, 31 // (op 3623) 
v_lshlrev_b32 v12, 6, v13 // (op 3623) 

s_cmp_eq_i32 s8, 0 // (op 3623) 
s_mov_b32 s18, scc // (op 3623) 
v_lshrrev_b32 v13, 2, v1 // (op 3623) 
s_cmp_eq_i32 s18, 0 // (op 3623) 
s_cselect_b32 s19, 0, -1 // (op 3623) 

v_lshlrev_b32 v14, 1, v13 // (op 3623) 

s_and_b32 s18, s9, 31 // (op 3623) 
s_lshl_b32 s24, 1, s18 // (op 3623) 
s_add_i32 s25, s24, s19 // (op 3623) 


s_and_b32 s19, s0, s25 // (op 3623) 


s_add_i32 s0, s1, s19 // (op 3623) 


s_ashr_i32 s1, s0, s18 // (op 3623) 


s_lshl_b32 s0, s9, 24 // (op 3623) 
s_ashr_i32 s18, s0, 31 // (op 3623) 

s_xor_b32 s0, s1, s18 // (op 3623) 

s_sub_i32 s1, s0, s18 // (op 3623) 


s_mul_hi_i32 s0, s1, s10 // (op 3623) 
s_add_i32 s18, s0, s1 // (op 3623) 

s_ashr_i32 s0, s18, 31 // (op 3623) 
s_cmp_eq_i32 s10, 0 // (op 3623) 
s_mov_b32 s19, scc // (op 3623) 
s_cmp_eq_i32 s19, 0 // (op 3623) 
s_cselect_b32 s24, 0, -1 // (op 3623) 

s_and_b32 s19, s11, 31 // (op 3623) 
s_lshl_b32 s25, 1, s19 // (op 3623) 
s_add_i32 s26, s25, s24 // (op 3623) 


s_and_b32 s27, s0, s26 // (op 3623) 


s_add_i32 s0, s18, s27 // (op 3623) 


s_ashr_i32 s18, s0, s19 // (op 3623) 


s_lshl_b32 s0, s11, 24 // (op 3623) 
s_ashr_i32 s19, s0, 31 // (op 3623) 

s_xor_b32 s0, s18, s19 // (op 3623) 

s_sub_i32 s18, s0, s19 // (op 3623) 


s_mul_i32 s0, s1, s13 // (op 3623) 

s_sub_u32 s1, s16, s0 // (op 3623) 

s_mul_i32 s0, s18, s13 // (op 3623) 
s_add_u32 s0, s0, s1 // (op 3623) 


s_mul_hi_i32 s1, s16, s7 // (op 3623) 
s_add_i32 s18, s1, s16 // (op 3623) 

s_ashr_i32 s1, s18, 31 // (op 3623) 
s_cmp_eq_i32 s7, 0 // (op 3623) 
s_mov_b32 s19, scc // (op 3623) 
s_cmp_eq_i32 s19, 0 // (op 3623) 
s_cselect_b32 s24, 0, -1 // (op 3623) 

s_and_b32 s19, s14, 31 // (op 3623) 
s_lshl_b32 s25, 1, s19 // (op 3623) 
s_add_i32 s26, s25, s24 // (op 3623) 


s_and_b32 s27, s1, s26 // (op 3623) 


s_add_i32 s1, s18, s27 // (op 3623) 


s_ashr_i32 s18, s1, s19 // (op 3623) 


s_lshl_b32 s1, s14, 24 // (op 3623) 
s_ashr_i32 s19, s1, 31 // (op 3623) 

s_xor_b32 s1, s18, s19 // (op 3623) 

s_sub_i32 s18, s1, s19 // (op 3623) 


s_add_u32 s24, s4, 32 // (op 3623) Start of Int64 add, locking SCC
s_addc_u32 s25, s5, 0 // (op 3623) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 3623) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 3623) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 3623) 

s_mul_i32 s1, s13, s24 // (op 3623) 

s_mul_i32 s19, s6, s1 // (op 3623) 

s_mul_i32 s1, s18, s19 // (op 3623) 

s_sub_u32 s19, s16, s1 // (op 3623) 

s_mov_b32 s1, s19 // (op 3623) 
s_mul_hi_i32 s24, s1, s12 // (op 3623) 
s_add_i32 s25, s24, s1 // (op 3623) 


s_ashr_i32 s1, s25, 31 // (op 3623) 
s_cmp_eq_i32 s12, 0 // (op 3623) 
s_mov_b32 s24, scc // (op 3623) 
s_cmp_eq_i32 s24, 0 // (op 3623) 
s_cselect_b32 s26, 0, -1 // (op 3623) 

s_and_b32 s24, s48, 31 // (op 3623) 
s_lshl_b32 s27, 1, s24 // (op 3623) 
s_add_i32 s49, s27, s26 // (op 3623) 


s_and_b32 s26, s1, s49 // (op 3623) 


s_add_i32 s1, s25, s26 // (op 3623) 


s_ashr_i32 s25, s1, s24 // (op 3623) 


s_lshl_b32 s1, s48, 24 // (op 3623) 
s_ashr_i32 s24, s1, 31 // (op 3623) 

s_xor_b32 s1, s25, s24 // (op 3623) 

s_sub_i32 s25, s1, s24 // (op 3623) 


s_mul_i32 s1, s15, s13 // (op 3623) 
s_add_u32 s26, s20, 256 // (op 3623) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 3623) End of Int64 add, unlocking SCC
s_sub_u32 s54, s26, 1 // (op 3623) least significant half; sets scc
s_subb_u32 s55, s27, 0 // (op 3623) most significant half; uses scc

s_ashr_i64 s[26:27], s[54:55], 8 // (op 3623) 

s_sub_i32 s24, s26, s1 // (op 3623) 

s_mul_i32 s26, s25, s24 // (op 3623) 


s_sub_u32 s27, s19, s26 // (op 3623) 


s_add_u32 s19, s27, s1 // (op 3623) 


s_cmp_eq_i32 s18, 0 // (op 3623) 
s_mov_b32 s1, scc // (op 3623) 

s_cmp_eq_i32 s1, 0 // (op 3623) 
s_cselect_b32 s18, s19, s0 // (op 3623) 



v_lshl_add_u32 v13, s18, 8, v12 // (op 3623) 


v_mov_b32 v17, 0 // (op 3623) convert
v_mov_b32 v16, v13 // (op 3623) convert

v_mul_lo_u32 v13, v14, s29 // (op 3623) most significant: low of low * high
v_mul_hi_u32 v15, v14, s28 // (op 3623) most significant: high of low * low
v_mul_lo_u32 v12, v14, s28 // (op 3623) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 3623) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v12 // (op 3623) least significant half
v_addc_co_u32 v15, s[0:1], v17, v13, s[0:1] // (op 3623) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 3623) most significant: low of high * low
v_mul_hi_u32 v16, v14, 6 // (op 3623) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 3623) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 3623) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3623) 

v_mov_b32 v12, v14 // (op 3623) convert

s_mul_i32 s1, s33, 6 // (op 3625) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 3625) least significant
s_mul_hi_u32 s18, s32, 6 // (op 3625) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3625) most significant: sum

s_mul_i32 s19, s28, s1 // (op 3625) most significant: low of low * high
s_mul_i32 s24, s29, s0 // (op 3625) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 3625) least significant
s_mul_hi_u32 s25, s28, s0 // (op 3625) most significant: high of low * low
s_add_u32 s19, s19, s24 // (op 3625) most significant: sum
s_add_u32 s19, s19, s25 // (op 3625) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 3625) 

s_mov_b32 s24, s30 // (op 3625) reg expression
s_mov_b32 s25, s31 // (op 3625) reg expression
s_mov_b32 s26, s0 // (op 3625) reg expression
s_mov_b32 s27, 131072 // (op 3625) call()

s_mul_hi_i32 s0, s16, s8 // (op 3642) 
v_and_b32 v13, -4, v1 // (op 3642) 
s_add_i32 s1, s0, s16 // (op 3642) 

v_sub_u32 v14, v1, v13 // (op 3642) 

s_ashr_i32 s0, s1, 31 // (op 3642) 
v_lshlrev_b32 v13, 6, v14 // (op 3642) 

s_cmp_eq_i32 s8, 0 // (op 3642) 
s_mov_b32 s18, scc // (op 3642) 
v_lshrrev_b32 v14, 2, v1 // (op 3642) 
s_cmp_eq_i32 s18, 0 // (op 3642) 
s_cselect_b32 s19, 0, -1 // (op 3642) 

v_lshlrev_b32 v15, 1, v14 // (op 3642) 

s_and_b32 s18, s9, 31 // (op 3642) 
s_lshl_b32 s49, 1, s18 // (op 3642) 
s_add_i32 s50, s49, s19 // (op 3642) 


s_and_b32 s19, s0, s50 // (op 3642) 


s_add_i32 s0, s1, s19 // (op 3642) 


s_ashr_i32 s1, s0, s18 // (op 3642) 


s_lshl_b32 s0, s9, 24 // (op 3642) 
s_ashr_i32 s18, s0, 31 // (op 3642) 

s_xor_b32 s0, s1, s18 // (op 3642) 

s_sub_i32 s1, s0, s18 // (op 3642) 


s_mul_hi_i32 s0, s1, s10 // (op 3642) 
s_add_i32 s18, s0, s1 // (op 3642) 

s_ashr_i32 s0, s18, 31 // (op 3642) 
s_cmp_eq_i32 s10, 0 // (op 3642) 
s_mov_b32 s19, scc // (op 3642) 
s_cmp_eq_i32 s19, 0 // (op 3642) 
s_cselect_b32 s49, 0, -1 // (op 3642) 

s_and_b32 s19, s11, 31 // (op 3642) 
s_lshl_b32 s50, 1, s19 // (op 3642) 
s_add_i32 s51, s50, s49 // (op 3642) 


s_and_b32 s49, s0, s51 // (op 3642) 


s_add_i32 s0, s18, s49 // (op 3642) 


s_ashr_i32 s18, s0, s19 // (op 3642) 


s_lshl_b32 s0, s11, 24 // (op 3642) 
s_ashr_i32 s19, s0, 31 // (op 3642) 

s_xor_b32 s0, s18, s19 // (op 3642) 

s_sub_i32 s18, s0, s19 // (op 3642) 


s_mul_i32 s0, s1, s13 // (op 3642) 

s_sub_u32 s1, s16, s0 // (op 3642) 

s_mul_i32 s0, s18, s13 // (op 3642) 
s_add_u32 s0, s0, s1 // (op 3642) 


s_mul_hi_i32 s1, s16, s7 // (op 3642) 
s_add_i32 s18, s1, s16 // (op 3642) 

s_ashr_i32 s1, s18, 31 // (op 3642) 
s_cmp_eq_i32 s7, 0 // (op 3642) 
s_mov_b32 s19, scc // (op 3642) 
s_cmp_eq_i32 s19, 0 // (op 3642) 
s_cselect_b32 s49, 0, -1 // (op 3642) 

s_and_b32 s19, s14, 31 // (op 3642) 
s_lshl_b32 s50, 1, s19 // (op 3642) 
s_add_i32 s51, s50, s49 // (op 3642) 


s_and_b32 s49, s1, s51 // (op 3642) 


s_add_i32 s1, s18, s49 // (op 3642) 


s_ashr_i32 s18, s1, s19 // (op 3642) 


s_lshl_b32 s1, s14, 24 // (op 3642) 
s_ashr_i32 s19, s1, 31 // (op 3642) 

s_xor_b32 s1, s18, s19 // (op 3642) 

s_sub_i32 s18, s1, s19 // (op 3642) 


s_add_u32 s54, s4, 32 // (op 3642) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 3642) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3642) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3642) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 3642) 

s_mul_i32 s1, s13, s50 // (op 3642) 

s_mul_i32 s19, s6, s1 // (op 3642) 

s_mul_i32 s1, s18, s19 // (op 3642) 

s_sub_u32 s19, s16, s1 // (op 3642) 

s_mov_b32 s1, s19 // (op 3642) 
s_mul_hi_i32 s49, s1, s12 // (op 3642) 
s_add_i32 s50, s49, s1 // (op 3642) 


s_ashr_i32 s1, s50, 31 // (op 3642) 
s_cmp_eq_i32 s12, 0 // (op 3642) 
s_mov_b32 s49, scc // (op 3642) 
s_cmp_eq_i32 s49, 0 // (op 3642) 
s_cselect_b32 s51, 0, -1 // (op 3642) 

s_and_b32 s49, s48, 31 // (op 3642) 
s_lshl_b32 s52, 1, s49 // (op 3642) 
s_add_i32 s53, s52, s51 // (op 3642) 


s_and_b32 s51, s1, s53 // (op 3642) 


s_add_i32 s1, s50, s51 // (op 3642) 


s_ashr_i32 s50, s1, s49 // (op 3642) 


s_lshl_b32 s1, s48, 24 // (op 3642) 
s_ashr_i32 s49, s1, 31 // (op 3642) 

s_xor_b32 s1, s50, s49 // (op 3642) 

s_sub_i32 s50, s1, s49 // (op 3642) 


s_mul_i32 s1, s15, s13 // (op 3642) 
s_add_u32 s54, s20, 256 // (op 3642) Start of Int64 add, locking SCC
s_addc_u32 s55, s21, 0 // (op 3642) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3642) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3642) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 8 // (op 3642) 

s_sub_i32 s49, s54, s1 // (op 3642) 

s_mul_i32 s51, s50, s49 // (op 3642) 


s_sub_u32 s49, s19, s51 // (op 3642) 


s_add_u32 s19, s49, s1 // (op 3642) 


s_cmp_eq_i32 s18, 0 // (op 3642) 
s_mov_b32 s1, scc // (op 3642) 

s_cmp_eq_i32 s1, 0 // (op 3642) 
s_cselect_b32 s18, s19, s0 // (op 3642) 



v_lshl_add_u32 v14, s18, 8, v13 // (op 3642) 


v_mov_b32 v17, 0 // (op 3642) convert
v_mov_b32 v16, v14 // (op 3642) convert

v_mul_lo_u32 v19, v15, s29 // (op 3642) most significant: low of low * high
v_mul_hi_u32 v13, v15, s28 // (op 3642) most significant: high of low * low
v_mul_lo_u32 v18, v15, s28 // (op 3642) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3642) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3642) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3642) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 3642) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 3642) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 3642) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3642) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3642) 

v_mov_b32 v13, v14 // (op 3642) convert

s_mul_i32 s1, s33, 6 // (op 3644) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 3644) least significant
s_mul_hi_u32 s18, s32, 6 // (op 3644) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3644) most significant: sum

s_mul_i32 s19, s28, s1 // (op 3644) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 3644) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 3644) least significant
s_mul_hi_u32 s50, s28, s0 // (op 3644) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 3644) most significant: sum
s_add_u32 s19, s19, s50 // (op 3644) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 3644) 

s_mov_b32 s24, s30 // (op 3644) reg expression
s_mov_b32 s25, s31 // (op 3644) reg expression
s_mov_b32 s26, s0 // (op 3644) reg expression
s_mov_b32 s27, 131072 // (op 3644) call()

s_mul_hi_i32 s0, s16, s8 // (op 3687) 
v_lshrrev_b32 v14, 3, v1 // (op 3687) 
s_add_i32 s1, s0, s16 // (op 3687) 

v_and_b32 v15, -8, v1 // (op 3687) 
s_ashr_i32 s0, s1, 31 // (op 3687) 
v_sub_u32 v16, v1, v15 // (op 3687) 

s_cmp_eq_i32 s8, 0 // (op 3687) 
s_mov_b32 s18, scc // (op 3687) 
v_lshlrev_b32 v15, 4, v16 // (op 3687) 

s_cmp_eq_i32 s18, 0 // (op 3687) 
s_cselect_b32 s19, 0, -1 // (op 3687) 

v_mov_b32 v17, 0 // (op 3687) convert
v_mov_b32 v16, v15 // (op 3687) convert

s_and_b32 s18, s9, 31 // (op 3687) 
s_lshl_b32 s49, 1, s18 // (op 3687) 
s_add_i32 s50, s49, s19 // (op 3687) 


s_and_b32 s19, s0, s50 // (op 3687) 


s_add_i32 s0, s1, s19 // (op 3687) 


s_ashr_i32 s1, s0, s18 // (op 3687) 


s_lshl_b32 s0, s9, 24 // (op 3687) 
s_ashr_i32 s18, s0, 31 // (op 3687) 

s_xor_b32 s0, s1, s18 // (op 3687) 

s_sub_i32 s1, s0, s18 // (op 3687) 


s_mul_hi_i32 s0, s1, s10 // (op 3687) 
s_add_i32 s18, s0, s1 // (op 3687) 

s_ashr_i32 s0, s18, 31 // (op 3687) 
s_cmp_eq_i32 s10, 0 // (op 3687) 
s_mov_b32 s19, scc // (op 3687) 
s_cmp_eq_i32 s19, 0 // (op 3687) 
s_cselect_b32 s49, 0, -1 // (op 3687) 

s_and_b32 s19, s11, 31 // (op 3687) 
s_lshl_b32 s50, 1, s19 // (op 3687) 
s_add_i32 s51, s50, s49 // (op 3687) 


s_and_b32 s49, s0, s51 // (op 3687) 


s_add_i32 s0, s18, s49 // (op 3687) 


s_ashr_i32 s18, s0, s19 // (op 3687) 


s_lshl_b32 s0, s11, 24 // (op 3687) 
s_ashr_i32 s19, s0, 31 // (op 3687) 

s_xor_b32 s0, s18, s19 // (op 3687) 

s_sub_i32 s18, s0, s19 // (op 3687) 


s_add_u32 s54, s4, 32 // (op 3687) Start of Int64 add, locking SCC
s_addc_u32 s55, s5, 0 // (op 3687) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 3687) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 3687) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 3687) 

s_mul_i32 s0, s18, s50 // (op 3687) 

s_sub_i32 s49, s1, s0 // (op 3687) 


s_mul_hi_i32 s0, s16, s7 // (op 3687) 
s_add_i32 s1, s0, s16 // (op 3687) 

s_ashr_i32 s0, s1, 31 // (op 3687) 
s_cmp_eq_i32 s7, 0 // (op 3687) 
s_mov_b32 s18, scc // (op 3687) 
s_cmp_eq_i32 s18, 0 // (op 3687) 
s_cselect_b32 s19, 0, -1 // (op 3687) 

s_and_b32 s18, s14, 31 // (op 3687) 
s_lshl_b32 s52, 1, s18 // (op 3687) 
s_add_i32 s53, s52, s19 // (op 3687) 


s_and_b32 s19, s0, s53 // (op 3687) 


s_add_i32 s0, s1, s19 // (op 3687) 


s_ashr_i32 s1, s0, s18 // (op 3687) 


s_lshl_b32 s0, s14, 24 // (op 3687) 
s_ashr_i32 s18, s0, 31 // (op 3687) 

s_xor_b32 s0, s1, s18 // (op 3687) 

s_sub_i32 s1, s0, s18 // (op 3687) 


s_mul_i32 s0, s13, s50 // (op 3687) 

s_mul_i32 s18, s6, s0 // (op 3687) 

s_mul_i32 s0, s1, s18 // (op 3687) 

s_sub_u32 s18, s16, s0 // (op 3687) 

s_mul_hi_i32 s0, s18, s12 // (op 3687) 
s_add_i32 s19, s0, s18 // (op 3687) 


s_ashr_i32 s0, s19, 31 // (op 3687) 
s_cmp_eq_i32 s12, 0 // (op 3687) 
s_mov_b32 s18, scc // (op 3687) 
s_cmp_eq_i32 s18, 0 // (op 3687) 
s_cselect_b32 s50, 0, -1 // (op 3687) 

s_and_b32 s18, s48, 31 // (op 3687) 
s_lshl_b32 s51, 1, s18 // (op 3687) 
s_add_i32 s52, s51, s50 // (op 3687) 


s_and_b32 s50, s0, s52 // (op 3687) 


s_add_i32 s0, s19, s50 // (op 3687) 


s_ashr_i32 s19, s0, s18 // (op 3687) 


s_lshl_b32 s0, s48, 24 // (op 3687) 
s_ashr_i32 s18, s0, 31 // (op 3687) 

s_xor_b32 s0, s19, s18 // (op 3687) 

s_sub_i32 s19, s0, s18 // (op 3687) 


s_cmp_eq_i32 s1, 0 // (op 3687) 
s_mov_b32 s0, scc // (op 3687) 

s_cmp_eq_i32 s0, 0 // (op 3687) 
s_cselect_b32 s1, s19, s49 // (op 3687) 



v_lshl_add_u32 v15, s1, 5, v14 // (op 3687) 


v_mul_lo_u32 v19, v15, s35 // (op 3687) most significant: low of low * high
v_mul_hi_u32 v14, v15, s34 // (op 3687) most significant: high of low * low
v_mul_lo_u32 v18, v15, s34 // (op 3687) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 3687) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3687) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3687) most significant half



s_mul_i32 s1, s34, s5 // (op 3689) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 3689) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 3689) least significant
s_mul_hi_u32 s19, s34, s4 // (op 3689) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3689) most significant: sum
s_add_u32 s1, s1, s19 // (op 3689) most significant: sum


s_mov_b32 s52, s36 // (op 3689) reg expression
s_mov_b32 s53, s37 // (op 3689) reg expression
s_mov_b32 s54, s0 // (op 3689) reg expression
s_mov_b32 s55, 131072 // (op 3689) call()

s_mul_hi_i32 s0, s16, s8 // (op 3706) 
v_lshrrev_b32 v16, 3, v1 // (op 3706) 
s_add_i32 s1, s0, s16 // (op 3706) 

v_and_b32 v17, -8, v1 // (op 3706) 
s_ashr_i32 s0, s1, 31 // (op 3706) 
v_sub_u32 v18, v1, v17 // (op 3706) 

s_cmp_eq_i32 s8, 0 // (op 3706) 
s_mov_b32 s18, scc // (op 3706) 
v_lshlrev_b32 v17, 4, v18 // (op 3706) 

s_cmp_eq_i32 s18, 0 // (op 3706) 
s_cselect_b32 s19, 0, -1 // (op 3706) 

v_mov_b32 v19, 0 // (op 3706) convert
v_mov_b32 v18, v17 // (op 3706) convert

s_and_b32 s18, s9, 31 // (op 3706) 
s_lshl_b32 s49, 1, s18 // (op 3706) 
s_add_i32 s50, s49, s19 // (op 3706) 


s_and_b32 s19, s0, s50 // (op 3706) 


s_add_i32 s0, s1, s19 // (op 3706) 


s_ashr_i32 s1, s0, s18 // (op 3706) 


s_lshl_b32 s0, s9, 24 // (op 3706) 
s_ashr_i32 s18, s0, 31 // (op 3706) 

s_xor_b32 s0, s1, s18 // (op 3706) 

s_sub_i32 s1, s0, s18 // (op 3706) 


s_mul_hi_i32 s0, s1, s10 // (op 3706) 
s_add_i32 s18, s0, s1 // (op 3706) 

s_ashr_i32 s0, s18, 31 // (op 3706) 
s_cmp_eq_i32 s10, 0 // (op 3706) 
s_mov_b32 s19, scc // (op 3706) 
s_cmp_eq_i32 s19, 0 // (op 3706) 
s_cselect_b32 s49, 0, -1 // (op 3706) 

s_and_b32 s19, s11, 31 // (op 3706) 
s_lshl_b32 s50, 1, s19 // (op 3706) 
s_add_i32 s51, s50, s49 // (op 3706) 


s_and_b32 s49, s0, s51 // (op 3706) 


s_add_i32 s0, s18, s49 // (op 3706) 


s_ashr_i32 s18, s0, s19 // (op 3706) 


s_lshl_b32 s0, s11, 24 // (op 3706) 
s_ashr_i32 s19, s0, 31 // (op 3706) 

s_xor_b32 s0, s18, s19 // (op 3706) 

s_sub_i32 s18, s0, s19 // (op 3706) 


s_add_u32 s50, s4, 32 // (op 3706) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3706) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3706) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3706) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 3706) 

s_mul_i32 s0, s18, s50 // (op 3706) 

s_sub_i32 s49, s1, s0 // (op 3706) 


s_mul_hi_i32 s0, s16, s7 // (op 3706) 
s_add_i32 s1, s0, s16 // (op 3706) 

s_ashr_i32 s0, s1, 31 // (op 3706) 
s_cmp_eq_i32 s7, 0 // (op 3706) 
s_mov_b32 s18, scc // (op 3706) 
s_cmp_eq_i32 s18, 0 // (op 3706) 
s_cselect_b32 s19, 0, -1 // (op 3706) 

s_and_b32 s18, s14, 31 // (op 3706) 
s_lshl_b32 s64, 1, s18 // (op 3706) 
s_add_i32 s65, s64, s19 // (op 3706) 


s_and_b32 s19, s0, s65 // (op 3706) 


s_add_i32 s0, s1, s19 // (op 3706) 


s_ashr_i32 s1, s0, s18 // (op 3706) 


s_lshl_b32 s0, s14, 24 // (op 3706) 
s_ashr_i32 s18, s0, 31 // (op 3706) 

s_xor_b32 s0, s1, s18 // (op 3706) 

s_sub_i32 s1, s0, s18 // (op 3706) 


s_mul_i32 s0, s13, s50 // (op 3706) 

s_mul_i32 s18, s6, s0 // (op 3706) 

s_mul_i32 s0, s1, s18 // (op 3706) 

s_sub_u32 s18, s16, s0 // (op 3706) 

s_mul_hi_i32 s0, s18, s12 // (op 3706) 
s_add_i32 s19, s0, s18 // (op 3706) 


s_ashr_i32 s0, s19, 31 // (op 3706) 
s_cmp_eq_i32 s12, 0 // (op 3706) 
s_mov_b32 s18, scc // (op 3706) 
s_cmp_eq_i32 s18, 0 // (op 3706) 
s_cselect_b32 s50, 0, -1 // (op 3706) 

s_and_b32 s18, s48, 31 // (op 3706) 
s_lshl_b32 s51, 1, s18 // (op 3706) 
s_add_i32 s64, s51, s50 // (op 3706) 


s_and_b32 s50, s0, s64 // (op 3706) 


s_add_i32 s0, s19, s50 // (op 3706) 


s_ashr_i32 s19, s0, s18 // (op 3706) 


s_lshl_b32 s0, s48, 24 // (op 3706) 
s_ashr_i32 s18, s0, 31 // (op 3706) 

s_xor_b32 s0, s19, s18 // (op 3706) 

s_sub_i32 s19, s0, s18 // (op 3706) 


s_cmp_eq_i32 s1, 0 // (op 3706) 
s_mov_b32 s0, scc // (op 3706) 

s_cmp_eq_i32 s0, 0 // (op 3706) 
s_cselect_b32 s1, s19, s49 // (op 3706) 



v_lshl_add_u32 v17, s1, 5, v16 // (op 3706) 


v_mul_lo_u32 v21, v17, s35 // (op 3706) most significant: low of low * high
v_mul_hi_u32 v16, v17, s34 // (op 3706) most significant: high of low * low
v_mul_lo_u32 v20, v17, s34 // (op 3706) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 3706) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 3706) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 3706) most significant half



s_mul_i32 s1, s34, s5 // (op 3708) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 3708) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 3708) least significant
s_mul_hi_u32 s19, s34, s4 // (op 3708) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3708) most significant: sum
s_add_u32 s1, s1, s19 // (op 3708) most significant: sum


s_mov_b32 s52, s36 // (op 3708) reg expression
s_mov_b32 s53, s37 // (op 3708) reg expression
s_mov_b32 s54, s0 // (op 3708) reg expression
s_mov_b32 s55, 131072 // (op 3708) call()

s_mul_hi_i32 s0, s16, s8 // (op 3751) 
v_and_b32 v18, -64, v1 // (op 3751) 
s_add_i32 s1, s0, s16 // (op 3751) 

v_sub_u32 v19, v1, v18 // (op 3751) 

s_ashr_i32 s0, s1, 31 // (op 3751) 
v_lshlrev_b32 v18, 2, v19 // (op 3751) 

s_cmp_eq_i32 s8, 0 // (op 3751) 
s_mov_b32 s18, scc // (op 3751) 
v_lshrrev_b32 v19, 6, v1 // (op 3751) 
s_cmp_eq_i32 s18, 0 // (op 3751) 
s_cselect_b32 s19, 0, -1 // (op 3751) 

s_and_b32 s18, s9, 31 // (op 3751) 
s_lshl_b32 s49, 1, s18 // (op 3751) 
s_add_i32 s50, s49, s19 // (op 3751) 


s_and_b32 s19, s0, s50 // (op 3751) 


s_add_i32 s0, s1, s19 // (op 3751) 


s_ashr_i32 s1, s0, s18 // (op 3751) 


s_lshl_b32 s0, s9, 24 // (op 3751) 
s_ashr_i32 s18, s0, 31 // (op 3751) 

s_xor_b32 s0, s1, s18 // (op 3751) 

s_sub_i32 s1, s0, s18 // (op 3751) 


s_mul_hi_i32 s0, s1, s10 // (op 3751) 
s_add_i32 s18, s0, s1 // (op 3751) 

s_ashr_i32 s0, s18, 31 // (op 3751) 
s_cmp_eq_i32 s10, 0 // (op 3751) 
s_mov_b32 s19, scc // (op 3751) 
s_cmp_eq_i32 s19, 0 // (op 3751) 
s_cselect_b32 s49, 0, -1 // (op 3751) 

s_and_b32 s19, s11, 31 // (op 3751) 
s_lshl_b32 s50, 1, s19 // (op 3751) 
s_add_i32 s51, s50, s49 // (op 3751) 


s_and_b32 s49, s0, s51 // (op 3751) 


s_add_i32 s0, s18, s49 // (op 3751) 


s_ashr_i32 s18, s0, s19 // (op 3751) 


s_lshl_b32 s0, s11, 24 // (op 3751) 
s_ashr_i32 s19, s0, 31 // (op 3751) 

s_xor_b32 s0, s18, s19 // (op 3751) 

s_sub_i32 s18, s0, s19 // (op 3751) 


s_mul_i32 s0, s1, s13 // (op 3751) 

s_sub_u32 s1, s16, s0 // (op 3751) 

s_mul_i32 s0, s18, s13 // (op 3751) 
s_add_u32 s0, s0, s1 // (op 3751) 


s_mul_hi_i32 s1, s16, s7 // (op 3751) 
s_add_i32 s18, s1, s16 // (op 3751) 

s_ashr_i32 s1, s18, 31 // (op 3751) 
s_cmp_eq_i32 s7, 0 // (op 3751) 
s_mov_b32 s19, scc // (op 3751) 
s_cmp_eq_i32 s19, 0 // (op 3751) 
s_cselect_b32 s49, 0, -1 // (op 3751) 

s_and_b32 s19, s14, 31 // (op 3751) 
s_lshl_b32 s50, 1, s19 // (op 3751) 
s_add_i32 s51, s50, s49 // (op 3751) 


s_and_b32 s49, s1, s51 // (op 3751) 


s_add_i32 s1, s18, s49 // (op 3751) 


s_ashr_i32 s18, s1, s19 // (op 3751) 


s_lshl_b32 s1, s14, 24 // (op 3751) 
s_ashr_i32 s19, s1, 31 // (op 3751) 

s_xor_b32 s1, s18, s19 // (op 3751) 

s_sub_i32 s18, s1, s19 // (op 3751) 


s_add_u32 s50, s4, 32 // (op 3751) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3751) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 3751) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 3751) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 3751) 

s_mul_i32 s1, s13, s50 // (op 3751) 

s_mul_i32 s19, s6, s1 // (op 3751) 

s_mul_i32 s1, s18, s19 // (op 3751) 

s_sub_u32 s19, s16, s1 // (op 3751) 

s_mov_b32 s1, s19 // (op 3751) 
s_mul_hi_i32 s49, s1, s12 // (op 3751) 
s_add_i32 s50, s49, s1 // (op 3751) 


s_ashr_i32 s1, s50, 31 // (op 3751) 
s_cmp_eq_i32 s12, 0 // (op 3751) 
s_mov_b32 s49, scc // (op 3751) 
s_cmp_eq_i32 s49, 0 // (op 3751) 
s_cselect_b32 s51, 0, -1 // (op 3751) 

s_and_b32 s49, s48, 31 // (op 3751) 
s_lshl_b32 s64, 1, s49 // (op 3751) 
s_add_i32 s65, s64, s51 // (op 3751) 


s_and_b32 s51, s1, s65 // (op 3751) 


s_add_i32 s1, s50, s51 // (op 3751) 


s_ashr_i32 s50, s1, s49 // (op 3751) 


s_lshl_b32 s1, s48, 24 // (op 3751) 
s_ashr_i32 s49, s1, 31 // (op 3751) 

s_xor_b32 s1, s50, s49 // (op 3751) 

s_sub_i32 s50, s1, s49 // (op 3751) 


s_mul_i32 s1, s15, s13 // (op 3751) 
s_add_u32 s64, s20, 256 // (op 3751) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 3751) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 3751) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 3751) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 8 // (op 3751) 

s_sub_i32 s49, s64, s1 // (op 3751) 

s_mul_i32 s51, s50, s49 // (op 3751) 


s_sub_u32 s49, s19, s51 // (op 3751) 


s_add_u32 s19, s49, s1 // (op 3751) 


s_cmp_eq_i32 s18, 0 // (op 3751) 
s_mov_b32 s1, scc // (op 3751) 

s_cmp_eq_i32 s1, 0 // (op 3751) 
s_cselect_b32 s18, s19, s0 // (op 3751) 



v_lshl_add_u32 v20, s18, 8, v18 // (op 3751) 


v_mul_lo_u32 v18, v19, s38 // (op 3751) 

v_add_u32 v19, v20, v18 // (op 3751) 


s_mul_i32 s1, s38, s43 // (op 3753) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3753) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3753) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3753) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3753) most significant: sum
s_add_u32 s1, s1, s19 // (op 3753) most significant: sum


s_mov_b32 s64, s40 // (op 3753) reg expression
s_mov_b32 s65, s41 // (op 3753) reg expression
s_mov_b32 s66, s0 // (op 3753) reg expression
s_mov_b32 s67, 131072 // (op 3753) call()

s_mul_hi_i32 s0, s16, s8 // (op 3770) 
v_and_b32 v18, -64, v1 // (op 3770) 
s_add_i32 s1, s0, s16 // (op 3770) 

v_sub_u32 v20, v1, v18 // (op 3770) 

s_ashr_i32 s0, s1, 31 // (op 3770) 
v_lshlrev_b32 v18, 2, v20 // (op 3770) 

s_cmp_eq_i32 s8, 0 // (op 3770) 
s_mov_b32 s18, scc // (op 3770) 
v_lshrrev_b32 v20, 6, v1 // (op 3770) 
s_cmp_eq_i32 s18, 0 // (op 3770) 
s_cselect_b32 s19, 0, -1 // (op 3770) 

s_and_b32 s18, s9, 31 // (op 3770) 
s_lshl_b32 s49, 1, s18 // (op 3770) 
s_add_i32 s50, s49, s19 // (op 3770) 


s_and_b32 s19, s0, s50 // (op 3770) 


s_add_i32 s0, s1, s19 // (op 3770) 


s_ashr_i32 s1, s0, s18 // (op 3770) 


s_lshl_b32 s0, s9, 24 // (op 3770) 
s_ashr_i32 s18, s0, 31 // (op 3770) 

s_xor_b32 s0, s1, s18 // (op 3770) 

s_sub_i32 s1, s0, s18 // (op 3770) 


s_mul_hi_i32 s0, s1, s10 // (op 3770) 
s_add_i32 s18, s0, s1 // (op 3770) 

s_ashr_i32 s0, s18, 31 // (op 3770) 
s_cmp_eq_i32 s10, 0 // (op 3770) 
s_mov_b32 s19, scc // (op 3770) 
s_cmp_eq_i32 s19, 0 // (op 3770) 
s_cselect_b32 s49, 0, -1 // (op 3770) 

s_and_b32 s19, s11, 31 // (op 3770) 
s_lshl_b32 s50, 1, s19 // (op 3770) 
s_add_i32 s51, s50, s49 // (op 3770) 


s_and_b32 s49, s0, s51 // (op 3770) 


s_add_i32 s0, s18, s49 // (op 3770) 


s_ashr_i32 s18, s0, s19 // (op 3770) 


s_lshl_b32 s0, s11, 24 // (op 3770) 
s_ashr_i32 s19, s0, 31 // (op 3770) 

s_xor_b32 s0, s18, s19 // (op 3770) 

s_sub_i32 s18, s0, s19 // (op 3770) 


s_mul_i32 s0, s1, s13 // (op 3770) 

s_sub_u32 s1, s16, s0 // (op 3770) 

s_mul_i32 s0, s18, s13 // (op 3770) 
s_add_u32 s0, s0, s1 // (op 3770) 


s_mul_hi_i32 s1, s16, s7 // (op 3770) 
s_add_i32 s18, s1, s16 // (op 3770) 

s_ashr_i32 s1, s18, 31 // (op 3770) 
s_cmp_eq_i32 s7, 0 // (op 3770) 
s_mov_b32 s19, scc // (op 3770) 
s_cmp_eq_i32 s19, 0 // (op 3770) 
s_cselect_b32 s49, 0, -1 // (op 3770) 

s_and_b32 s19, s14, 31 // (op 3770) 
s_lshl_b32 s50, 1, s19 // (op 3770) 
s_add_i32 s51, s50, s49 // (op 3770) 


s_and_b32 s49, s1, s51 // (op 3770) 


s_add_i32 s1, s18, s49 // (op 3770) 


s_ashr_i32 s18, s1, s19 // (op 3770) 


s_lshl_b32 s1, s14, 24 // (op 3770) 
s_ashr_i32 s19, s1, 31 // (op 3770) 

s_xor_b32 s1, s18, s19 // (op 3770) 

s_sub_i32 s18, s1, s19 // (op 3770) 


s_add_u32 s50, s4, 32 // (op 3770) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3770) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3770) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3770) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 3770) 

s_mul_i32 s1, s13, s50 // (op 3770) 

s_mul_i32 s19, s6, s1 // (op 3770) 

s_mul_i32 s1, s18, s19 // (op 3770) 

s_sub_u32 s19, s16, s1 // (op 3770) 

s_mov_b32 s1, s19 // (op 3770) 
s_mul_hi_i32 s49, s1, s12 // (op 3770) 
s_add_i32 s50, s49, s1 // (op 3770) 


s_ashr_i32 s1, s50, 31 // (op 3770) 
s_cmp_eq_i32 s12, 0 // (op 3770) 
s_mov_b32 s49, scc // (op 3770) 
s_cmp_eq_i32 s49, 0 // (op 3770) 
s_cselect_b32 s51, 0, -1 // (op 3770) 

s_and_b32 s49, s48, 31 // (op 3770) 
s_lshl_b32 s68, 1, s49 // (op 3770) 
s_add_i32 s69, s68, s51 // (op 3770) 


s_and_b32 s51, s1, s69 // (op 3770) 


s_add_i32 s1, s50, s51 // (op 3770) 


s_ashr_i32 s50, s1, s49 // (op 3770) 


s_lshl_b32 s1, s48, 24 // (op 3770) 
s_ashr_i32 s49, s1, 31 // (op 3770) 

s_xor_b32 s1, s50, s49 // (op 3770) 

s_sub_i32 s50, s1, s49 // (op 3770) 


s_mul_i32 s1, s15, s13 // (op 3770) 
s_add_u32 s68, s20, 256 // (op 3770) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 3770) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 3770) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 3770) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 8 // (op 3770) 

s_sub_i32 s49, s68, s1 // (op 3770) 

s_mul_i32 s51, s50, s49 // (op 3770) 


s_sub_u32 s49, s19, s51 // (op 3770) 


s_add_u32 s19, s49, s1 // (op 3770) 


s_cmp_eq_i32 s18, 0 // (op 3770) 
s_mov_b32 s1, scc // (op 3770) 

s_cmp_eq_i32 s1, 0 // (op 3770) 
s_cselect_b32 s18, s19, s0 // (op 3770) 



v_lshl_add_u32 v21, s18, 8, v18 // (op 3770) 


v_mul_lo_u32 v18, v20, s38 // (op 3770) 

v_add_u32 v20, v21, v18 // (op 3770) 


s_mul_i32 s1, s38, s43 // (op 3772) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3772) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3772) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3772) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3772) most significant: sum
s_add_u32 s1, s1, s19 // (op 3772) most significant: sum


s_mov_b32 s64, s40 // (op 3772) reg expression
s_mov_b32 s65, s41 // (op 3772) reg expression
s_mov_b32 s66, s0 // (op 3772) reg expression
s_mov_b32 s67, 131072 // (op 3772) call()

s_mul_hi_i32 s0, s16, s8 // (op 3815) 
v_lshrrev_b32 v18, 6, v1 // (op 3815) 
s_add_i32 s1, s0, s16 // (op 3815) 

v_and_b32 v21, 1, v18 // (op 3815) 

s_ashr_i32 s0, s1, 31 // (op 3815) 
v_and_b32 v18, 15, v1 // (op 3815) 
s_cmp_eq_i32 s8, 0 // (op 3815) 
s_mov_b32 s18, scc // (op 3815) 
v_lshl_add_u32 v22, v21, 4, v18 // (op 3815) 


s_cmp_eq_i32 s18, 0 // (op 3815) 
s_cselect_b32 s19, 0, -1 // (op 3815) 

v_and_b32 v18, 63, v1 // (op 3815) 
s_and_b32 s18, s9, 31 // (op 3815) 
v_lshrrev_b32 v21, 4, v18 // (op 3815) 

s_lshl_b32 s49, 1, s18 // (op 3815) 
s_add_i32 s50, s49, s19 // (op 3815) 


s_and_b32 s19, s0, s50 // (op 3815) 


s_add_i32 s0, s1, s19 // (op 3815) 


s_ashr_i32 s1, s0, s18 // (op 3815) 


s_lshl_b32 s0, s9, 24 // (op 3815) 
s_ashr_i32 s18, s0, 31 // (op 3815) 

s_xor_b32 s0, s1, s18 // (op 3815) 

s_sub_i32 s1, s0, s18 // (op 3815) 


s_mul_hi_i32 s0, s1, s10 // (op 3815) 
s_add_i32 s18, s0, s1 // (op 3815) 

s_ashr_i32 s0, s18, 31 // (op 3815) 
s_cmp_eq_i32 s10, 0 // (op 3815) 
s_mov_b32 s19, scc // (op 3815) 
s_cmp_eq_i32 s19, 0 // (op 3815) 
s_cselect_b32 s49, 0, -1 // (op 3815) 

s_and_b32 s19, s11, 31 // (op 3815) 
s_lshl_b32 s50, 1, s19 // (op 3815) 
s_add_i32 s51, s50, s49 // (op 3815) 


s_and_b32 s49, s0, s51 // (op 3815) 


s_add_i32 s0, s18, s49 // (op 3815) 


s_ashr_i32 s18, s0, s19 // (op 3815) 


s_lshl_b32 s0, s11, 24 // (op 3815) 
s_ashr_i32 s19, s0, 31 // (op 3815) 

s_xor_b32 s0, s18, s19 // (op 3815) 

s_sub_i32 s18, s0, s19 // (op 3815) 


s_add_u32 s50, s4, 32 // (op 3815) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3815) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 3815) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 3815) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 3815) 

s_mul_i32 s0, s18, s50 // (op 3815) 

s_sub_i32 s49, s1, s0 // (op 3815) 


s_mul_hi_i32 s0, s16, s7 // (op 3815) 
s_add_i32 s1, s0, s16 // (op 3815) 

s_ashr_i32 s0, s1, 31 // (op 3815) 
s_cmp_eq_i32 s7, 0 // (op 3815) 
s_mov_b32 s18, scc // (op 3815) 
s_cmp_eq_i32 s18, 0 // (op 3815) 
s_cselect_b32 s19, 0, -1 // (op 3815) 

s_and_b32 s18, s14, 31 // (op 3815) 
s_lshl_b32 s68, 1, s18 // (op 3815) 
s_add_i32 s69, s68, s19 // (op 3815) 


s_and_b32 s19, s0, s69 // (op 3815) 


s_add_i32 s0, s1, s19 // (op 3815) 


s_ashr_i32 s1, s0, s18 // (op 3815) 


s_lshl_b32 s0, s14, 24 // (op 3815) 
s_ashr_i32 s18, s0, 31 // (op 3815) 

s_xor_b32 s0, s1, s18 // (op 3815) 

s_sub_i32 s1, s0, s18 // (op 3815) 


s_mul_i32 s0, s13, s50 // (op 3815) 

s_mul_i32 s18, s6, s0 // (op 3815) 

s_mul_i32 s0, s1, s18 // (op 3815) 

s_sub_u32 s18, s16, s0 // (op 3815) 

s_mul_hi_i32 s0, s18, s12 // (op 3815) 
s_add_i32 s19, s0, s18 // (op 3815) 


s_ashr_i32 s0, s19, 31 // (op 3815) 
s_cmp_eq_i32 s12, 0 // (op 3815) 
s_mov_b32 s18, scc // (op 3815) 
s_cmp_eq_i32 s18, 0 // (op 3815) 
s_cselect_b32 s50, 0, -1 // (op 3815) 

s_and_b32 s18, s48, 31 // (op 3815) 
s_lshl_b32 s51, 1, s18 // (op 3815) 
s_add_i32 s68, s51, s50 // (op 3815) 


s_and_b32 s50, s0, s68 // (op 3815) 


s_add_i32 s0, s19, s50 // (op 3815) 


s_ashr_i32 s19, s0, s18 // (op 3815) 


s_lshl_b32 s0, s48, 24 // (op 3815) 
s_ashr_i32 s18, s0, 31 // (op 3815) 

s_xor_b32 s0, s19, s18 // (op 3815) 

s_sub_i32 s19, s0, s18 // (op 3815) 


s_cmp_eq_i32 s1, 0 // (op 3815) 
s_mov_b32 s0, scc // (op 3815) 

s_cmp_eq_i32 s0, 0 // (op 3815) 
s_cselect_b32 s1, s19, s49 // (op 3815) 



v_lshl_add_u32 v18, s1, 5, v22 // (op 3815) 


v_mul_lo_u32 v22, v18, s44 // (op 3815) 

v_add_u32 v18, v21, v22 // (op 3815) 


s_mul_i32 s1, s44, s5 // (op 3817) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 3817) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 3817) least significant
s_mul_hi_u32 s19, s44, s4 // (op 3817) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3817) most significant: sum
s_add_u32 s1, s1, s19 // (op 3817) most significant: sum


s_mov_b32 s68, s46 // (op 3817) reg expression
s_mov_b32 s69, s47 // (op 3817) reg expression
s_mov_b32 s70, s0 // (op 3817) reg expression
s_mov_b32 s71, 131072 // (op 3817) call()

s_mul_hi_i32 s0, s16, s8 // (op 3834) 
v_lshrrev_b32 v21, 6, v1 // (op 3834) 
s_add_i32 s1, s0, s16 // (op 3834) 

v_and_b32 v22, 1, v21 // (op 3834) 

s_ashr_i32 s0, s1, 31 // (op 3834) 
v_and_b32 v21, 15, v1 // (op 3834) 
s_cmp_eq_i32 s8, 0 // (op 3834) 
s_mov_b32 s18, scc // (op 3834) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 3834) 


s_cmp_eq_i32 s18, 0 // (op 3834) 
s_cselect_b32 s19, 0, -1 // (op 3834) 

v_and_b32 v21, 63, v1 // (op 3834) 
s_and_b32 s18, s9, 31 // (op 3834) 
v_lshrrev_b32 v22, 4, v21 // (op 3834) 

s_lshl_b32 s49, 1, s18 // (op 3834) 
s_add_i32 s50, s49, s19 // (op 3834) 


s_and_b32 s19, s0, s50 // (op 3834) 


s_add_i32 s0, s1, s19 // (op 3834) 


s_ashr_i32 s1, s0, s18 // (op 3834) 


s_lshl_b32 s0, s9, 24 // (op 3834) 
s_ashr_i32 s18, s0, 31 // (op 3834) 

s_xor_b32 s0, s1, s18 // (op 3834) 

s_sub_i32 s1, s0, s18 // (op 3834) 


s_mul_hi_i32 s0, s1, s10 // (op 3834) 
s_add_i32 s18, s0, s1 // (op 3834) 

s_ashr_i32 s0, s18, 31 // (op 3834) 
s_cmp_eq_i32 s10, 0 // (op 3834) 
s_mov_b32 s19, scc // (op 3834) 
s_cmp_eq_i32 s19, 0 // (op 3834) 
s_cselect_b32 s49, 0, -1 // (op 3834) 

s_and_b32 s19, s11, 31 // (op 3834) 
s_lshl_b32 s50, 1, s19 // (op 3834) 
s_add_i32 s51, s50, s49 // (op 3834) 


s_and_b32 s49, s0, s51 // (op 3834) 


s_add_i32 s0, s18, s49 // (op 3834) 


s_ashr_i32 s18, s0, s19 // (op 3834) 


s_lshl_b32 s0, s11, 24 // (op 3834) 
s_ashr_i32 s19, s0, 31 // (op 3834) 

s_xor_b32 s0, s18, s19 // (op 3834) 

s_sub_i32 s18, s0, s19 // (op 3834) 


s_add_u32 s50, s4, 32 // (op 3834) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3834) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3834) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3834) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 3834) 

s_mul_i32 s0, s18, s50 // (op 3834) 

s_sub_i32 s49, s1, s0 // (op 3834) 


s_mul_hi_i32 s0, s16, s7 // (op 3834) 
s_add_i32 s1, s0, s16 // (op 3834) 

s_ashr_i32 s0, s1, 31 // (op 3834) 
s_cmp_eq_i32 s7, 0 // (op 3834) 
s_mov_b32 s18, scc // (op 3834) 
s_cmp_eq_i32 s18, 0 // (op 3834) 
s_cselect_b32 s19, 0, -1 // (op 3834) 

s_and_b32 s18, s14, 31 // (op 3834) 
s_lshl_b32 s72, 1, s18 // (op 3834) 
s_add_i32 s73, s72, s19 // (op 3834) 


s_and_b32 s19, s0, s73 // (op 3834) 


s_add_i32 s0, s1, s19 // (op 3834) 


s_ashr_i32 s1, s0, s18 // (op 3834) 


s_lshl_b32 s0, s14, 24 // (op 3834) 
s_ashr_i32 s18, s0, 31 // (op 3834) 

s_xor_b32 s0, s1, s18 // (op 3834) 

s_sub_i32 s1, s0, s18 // (op 3834) 


s_mul_i32 s0, s13, s50 // (op 3834) 

s_mul_i32 s18, s6, s0 // (op 3834) 

s_mul_i32 s0, s1, s18 // (op 3834) 

s_sub_u32 s18, s16, s0 // (op 3834) 

s_mul_hi_i32 s0, s18, s12 // (op 3834) 
s_add_i32 s19, s0, s18 // (op 3834) 


s_ashr_i32 s0, s19, 31 // (op 3834) 
s_cmp_eq_i32 s12, 0 // (op 3834) 
s_mov_b32 s18, scc // (op 3834) 
s_cmp_eq_i32 s18, 0 // (op 3834) 
s_cselect_b32 s50, 0, -1 // (op 3834) 

s_and_b32 s18, s48, 31 // (op 3834) 
s_lshl_b32 s51, 1, s18 // (op 3834) 
s_add_i32 s72, s51, s50 // (op 3834) 


s_and_b32 s50, s0, s72 // (op 3834) 


s_add_i32 s0, s19, s50 // (op 3834) 


s_ashr_i32 s19, s0, s18 // (op 3834) 


s_lshl_b32 s0, s48, 24 // (op 3834) 
s_ashr_i32 s18, s0, 31 // (op 3834) 

s_xor_b32 s0, s19, s18 // (op 3834) 

s_sub_i32 s19, s0, s18 // (op 3834) 


s_cmp_eq_i32 s1, 0 // (op 3834) 
s_mov_b32 s0, scc // (op 3834) 

s_cmp_eq_i32 s0, 0 // (op 3834) 
s_cselect_b32 s1, s19, s49 // (op 3834) 



v_lshl_add_u32 v21, s1, 5, v23 // (op 3834) 


v_mul_lo_u32 v23, v21, s44 // (op 3834) 

v_add_u32 v21, v22, v23 // (op 3834) 


s_mul_i32 s1, s44, s5 // (op 3836) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 3836) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 3836) least significant
s_mul_hi_u32 s19, s44, s4 // (op 3836) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3836) most significant: sum
s_add_u32 s1, s1, s19 // (op 3836) most significant: sum


s_mov_b32 s68, s46 // (op 3836) reg expression
s_mov_b32 s69, s47 // (op 3836) reg expression
s_mov_b32 s70, s0 // (op 3836) reg expression
s_mov_b32 s71, 131072 // (op 3836) call()




buffer_load_ubyte v22, v18, s[68:71], 0 offen offset: 0 // (op 3284) Load value
v_and_b32 v23, -8, v1 // (op 3958) 
v_sub_u32 v24, v1, v23 // (op 3958) 

v_lshlrev_b32 v23, 4, v24 // (op 3958) 

v_lshrrev_b32 v24, 3, v1 // (op 3958) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 3958) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 3958) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 3572) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 3572) Load value direct to lds


s_mul_i32 s1, s29, 6 // (op 3291) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 3291) least significant
s_mul_hi_u32 s18, s28, 6 // (op 3291) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3291) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3291) 


v_mov_b32 v23, v12 // (op 3291) reg expression
buffer_load_dwordx3 v[26:28], v23, s[24:27], 0 offen // (op 3291) Load value
buffer_load_dwordx3 v[30:32], v23, s[24:27], 0 offen offset:12 // (op 3291) Load value
buffer_load_dwordx3 v[34:36], v23, s[24:27], 0 offen offset:24 // (op 3291) Load value
buffer_load_dwordx3 v[38:40], v23, s[24:27], 0 offen offset:36 // (op 3291) Load value
v_add_u32 v23, v23, v24 // (op 3291) 
buffer_load_dwordx3 v[42:44], v23, s[24:27], 0 offen // (op 3291) Load value
buffer_load_dwordx3 v[46:48], v23, s[24:27], 0 offen offset:12 // (op 3291) Load value
buffer_load_dwordx3 v[50:52], v23, s[24:27], 0 offen offset:24 // (op 3291) Load value
buffer_load_dwordx3 v[54:56], v23, s[24:27], 0 offen offset:36 // (op 3291) Load value




v_mov_b32 v29, v19 // (op 3294) reg expression

buffer_load_dword v33, v29, s[64:67], 0 offen // (op 3294) Load value




buffer_load_ubyte v29, v18, s[68:71], 0 offen offset:4 // (op 3298) Load value

v_and_b32 v18, -8, v1 // (op 4102) 
v_sub_u32 v37, v1, v18 // (op 4102) 

v_lshlrev_b32 v18, 4, v37 // (op 4102) 

v_lshrrev_b32 v37, 3, v1 // (op 4102) 
v_lshl_add_u32 v41, v37, 7, v18 // (op 4102) 


s_nop 0
v_readfirstlane_b32 s0, v41 // (op 4102) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 3577) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3577) reg expression
s_mov_b32 s18, 128 // (op 3577) call()
buffer_load_dwordx4 v14, s[52:55], s18 offen offset: 0 lds // (op 3577) Load value direct to lds





s_mul_i32 s1, s29, 6 // (op 3305) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 3305) least significant
s_mul_hi_u32 s18, s28, 6 // (op 3305) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3305) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[0:1] // (op 3305) 


s_lshl_b64 s[0:1], s[28:29], 7 // (op 3305) 
s_mul_i32 s19, s1, 6 // (op 3305) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 3305) least significant
s_mul_hi_u32 s49, s0, 6 // (op 3305) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 3305) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 3305) 

v_add_u32 v18, v12, s0 // (op 3305) 

buffer_load_dwordx3 v[58:60], v18, s[24:27], 0 offen // (op 3305) Load value
buffer_load_dwordx3 v[62:64], v18, s[24:27], 0 offen offset:12 // (op 3305) Load value
buffer_load_dwordx3 v[66:68], v18, s[24:27], 0 offen offset:24 // (op 3305) Load value
buffer_load_dwordx3 v[70:72], v18, s[24:27], 0 offen offset:36 // (op 3305) Load value
v_add_u32 v18, v18, v14 // (op 3305) 
buffer_load_dwordx3 v[74:76], v18, s[24:27], 0 offen // (op 3305) Load value
buffer_load_dwordx3 v[78:80], v18, s[24:27], 0 offen offset:12 // (op 3305) Load value
buffer_load_dwordx3 v[82:84], v18, s[24:27], 0 offen offset:24 // (op 3305) Load value
buffer_load_dwordx3 v[86:88], v18, s[24:27], 0 offen offset:36 // (op 3305) Load value





s_lshl_b32 s0, s38, 2 // (op 3308) 
v_add_u32 v12, v19, s0 // (op 3308) 


buffer_load_dword v18, v12, s[64:67], 0 offen // (op 3308) Load value


v_and_b32 v12, -4, v1 // (op 3901) 
v_sub_u32 v19, v1, v12 // (op 3901) 

v_lshlrev_b32 v12, 6, v19 // (op 3901) 

v_lshrrev_b32 v19, 2, v1 // (op 3901) 
v_lshl_add_u32 v37, v19, 9, v12 // (op 3901) 


v_mul_lo_u32 v12, v37, 6 // (op 3901) 
v_lshrrev_b32 v19, 3, v12 // (op 3901) 

v_lshrrev_b32 v12, 4, v37 // (op 3901) 

v_lshlrev_b32 v37, 2, v12 // (op 3901) 

v_add_u32 v12, v19, v37 // (op 3901) 





s_waitcnt vmcnt(19)
ds_write_b96 v12, v[26:28] offset:8192 // (op 3311) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v12, v[30:32] offset:8208 // (op 3311) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v12, v[34:36] offset:8224 // (op 3311) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v12, v[38:40] offset:8240 // (op 3311) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v12, v[42:44] offset:8448 // (op 3311) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v12, v[46:48] offset:8464 // (op 3311) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v12, v[50:52] offset:8480 // (op 3311) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v12, v[54:56] offset:8496 // (op 3311) Store local data 

v_and_b32 v12, -64, v1 // (op 4011) 
v_sub_u32 v19, v1, v12 // (op 4011) 

v_lshlrev_b32 v12, 2, v19 // (op 4011) 

v_lshrrev_b32 v19, 6, v1 // (op 4011) 
v_lshl_add_u32 v37, v19, 8, v12 // (op 4011) 





s_waitcnt vmcnt(11)
ds_write_b32 v37, v33 offset:40960 // (op 3312) Store local data 

v_and_b32 v12, -4, v1 // (op 3872) 
v_sub_u32 v19, v1, v12 // (op 3872) 

v_lshlrev_b32 v12, 6, v19 // (op 3872) 

v_lshrrev_b32 v19, 2, v1 // (op 3872) 
v_lshl_add_u32 v37, v19, 9, v12 // (op 3872) 


v_mul_lo_u32 v12, v37, 6 // (op 3872) 
v_lshrrev_b32 v19, 3, v12 // (op 3872) 

v_lshrrev_b32 v12, 4, v37 // (op 3872) 

v_lshlrev_b32 v37, 2, v12 // (op 3872) 

v_add_u32 v12, v19, v37 // (op 3872) 


v_and_b32 v19, -8, v1 // (op 3930) 
v_sub_u32 v37, v1, v19 // (op 3930) 

v_lshlrev_b32 v19, 4, v37 // (op 3930) 

v_lshrrev_b32 v37, 3, v1 // (op 3930) 
v_lshl_add_u32 v41, v37, 7, v19 // (op 3930) 


s_nop 0
v_readfirstlane_b32 s0, v41 // (op 3930) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -64, v1 // (op 3982) 
v_sub_u32 v37, v1, v19 // (op 3982) 

v_lshlrev_b32 v19, 2, v37 // (op 3982) 

v_lshrrev_b32 v37, 6, v1 // (op 3982) 
v_lshl_add_u32 v41, v37, 8, v19 // (op 3982) 


v_and_b32 v19, -4, v1 // (op 4058) 
v_sub_u32 v37, v1, v19 // (op 4058) 

v_lshlrev_b32 v19, 6, v37 // (op 4058) 

v_lshrrev_b32 v37, 2, v1 // (op 4058) 
v_lshl_add_u32 v45, v37, 9, v19 // (op 4058) 


v_mul_lo_u32 v19, v45, 6 // (op 4058) 
v_lshrrev_b32 v37, 3, v19 // (op 4058) 

v_lshrrev_b32 v19, 4, v45 // (op 4058) 

v_lshlrev_b32 v45, 2, v19 // (op 4058) 

v_add_u32 v19, v37, v45 // (op 4058) 


v_and_b32 v37, -8, v1 // (op 4087) 
v_sub_u32 v45, v1, v37 // (op 4087) 

v_lshlrev_b32 v37, 4, v45 // (op 4087) 

v_lshrrev_b32 v45, 3, v1 // (op 4087) 
v_lshl_add_u32 v49, v45, 7, v37 // (op 4087) 


s_nop 0
v_readfirstlane_b32 s1, v49 // (op 4087) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v37, -64, v1 // (op 4126) 
v_sub_u32 v45, v1, v37 // (op 4126) 

v_lshlrev_b32 v37, 2, v45 // (op 4126) 

v_lshrrev_b32 v45, 6, v1 // (op 4126) 
v_lshl_add_u32 v49, v45, 8, v37 // (op 4126) 


s_mov_b32 s18, 0 // (op 1287) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 1286) 
s_mov_b32 s19, s50 // (op 1286) convert

s_and_b32 s49, s19, 4294967294 // (op 1286) 

s_cmp_lt_u32 s18, s49 // (op 1286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoop_1286 // (op 1286) Condition: Top (jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoop_1286 if false)
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoop_1286:
s_add_u32 s18, s18, 2 // (op 1288) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4202) 



ds_read_b96_tr_b6 v[90:92], v0 offset:8192 // (op 67) Transpose load local data 
ds_read_b96_tr_b6 v[93:95], v0 offset:12288 // (op 67) Transpose load local data 
s_barrier  // (op 4215) 



ds_read_b128 v[96:99], v3 // (op 72) Load local data 
ds_read_b128 v[100:103], v3 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4228) 


ds_read_u8 v37, v7 offset:40960 // (op 77) Load local data 



ds_read_b96_tr_b6 v[104:106], v0 offset:8224 // (op 344) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:95], v[96:103], a[0:3], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 84) 
ds_read_b96_tr_b6 v[107:109], v0 offset:12320 // (op 344) Transpose load local data 




ds_read_u8 v37, v7 offset:40992 // (op 348) Load local data 



ds_read_b96_tr_b6 v[90:92], v0 offset:8256 // (op 506) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:109], v[96:103], a[4:7], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 350) 
ds_read_b96_tr_b6 v[93:95], v0 offset:12352 // (op 506) Transpose load local data 




ds_read_u8 v37, v7 offset:41024 // (op 510) Load local data 



ds_read_b96_tr_b6 v[104:106], v0 offset:8288 // (op 668) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:95], v[96:103], a[8:11], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 512) 
ds_read_b96_tr_b6 v[107:109], v0 offset:12384 // (op 668) Transpose load local data 




ds_read_u8 v37, v7 offset:41056 // (op 672) Load local data 



ds_read_b96_tr_b6 v[90:92], v0 offset:8320 // (op 830) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:109], v[96:103], a[12:15], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 674) 
ds_read_b96_tr_b6 v[93:95], v0 offset:12416 // (op 830) Transpose load local data 




ds_read_u8 v37, v7 offset:41088 // (op 834) Load local data 



ds_read_b96_tr_b6 v[104:106], v0 offset:8352 // (op 992) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:95], v[96:103], a[16:19], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 836) 
ds_read_b96_tr_b6 v[107:109], v0 offset:12448 // (op 992) Transpose load local data 




ds_read_u8 v37, v7 offset:41120 // (op 996) Load local data 



ds_read_b96_tr_b6 v[90:92], v0 offset:8384 // (op 1154) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:109], v[96:103], a[20:23], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 998) 
ds_read_b96_tr_b6 v[93:95], v0 offset:12480 // (op 1154) Transpose load local data 




ds_read_u8 v37, v7 offset:41152 // (op 1158) Load local data 



ds_read_b96_tr_b6 v[104:106], v0 offset:8416 // (op 1316) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[96:103], a[24:27], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 1160) 
ds_read_b96_tr_b6 v[107:109], v0 offset:12512 // (op 1316) Transpose load local data 




ds_read_u8 v37, v7 offset:41184 // (op 1320) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:109], v[96:103], a[28:31], v37, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 1322) 





v_mov_b32 v37, v21 // (op 82) 

buffer_load_ubyte v22, v37, s[68:71], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 4218) 


v_mov_b64 v[14:15], v[16:17] // (op 3555) 

s_mov_b32 m0, s0 // (op 3555) reg expression
s_mov_b32 s19, 256 // (op 3555) call()
buffer_load_dwordx4 v14, s[52:55], s19 offen offset: 0 lds // (op 3555) Load value direct to lds


s_mul_i32 s51, s29, 6 // (op 4) most significant: low of high * low
s_mul_i32 s50, s28, 6 // (op 4) least significant
s_mul_hi_u32 s19, s28, 6 // (op 4) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 4) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[50:51] // (op 4) 


v_mov_b32 v23, v13 // (op 4) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 4) 
s_mul_i32 s73, s51, 6 // (op 4) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 4) least significant
s_mul_hi_u32 s19, s50, 6 // (op 4) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 4) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 4) 

s_lshl_b32 s19, s50, 1 // (op 4) 

v_add_u32 v45, v23, s19 // (op 4) 

buffer_load_dwordx3 v[26:28], v45, s[24:27], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[30:32], v45, s[24:27], 0 offen offset:12 // (op 4) Load value
buffer_load_dwordx3 v[34:36], v45, s[24:27], 0 offen offset:24 // (op 4) Load value
buffer_load_dwordx3 v[38:40], v45, s[24:27], 0 offen offset:36 // (op 4) Load value
v_add_u32 v45, v45, v24 // (op 4) 
buffer_load_dwordx3 v[42:44], v45, s[24:27], 0 offen // (op 4) Load value
buffer_load_dwordx3 v[46:48], v45, s[24:27], 0 offen offset:12 // (op 4) Load value
buffer_load_dwordx3 v[50:52], v45, s[24:27], 0 offen offset:24 // (op 4) Load value
buffer_load_dwordx3 v[54:56], v45, s[24:27], 0 offen offset:36 // (op 4) Load value




v_mov_b32 v45, v20 // (op 17) 
s_lshl_b32 s19, s38, 3 // (op 17) 
v_add_u32 v53, v45, s19 // (op 17) 


buffer_load_dword v33, v53, s[64:67], 0 offen // (op 17) Load value

s_barrier  // (op 4245) 



s_waitcnt vmcnt(19)
ds_write_b96 v19, v[58:60] offset:41984 // (op 229) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v19, v[62:64] offset:42000 // (op 229) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v19, v[66:68] offset:42016 // (op 229) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v19, v[70:72] offset:42032 // (op 229) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v19, v[74:76] offset:42240 // (op 229) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v19, v[78:80] offset:42256 // (op 229) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v19, v[82:84] offset:42272 // (op 229) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v19, v[86:88] offset:42288 // (op 229) Store local data 
s_barrier  // (op 4257) 



v_add_u32 v53, 74752, v49 // (op 240) 
s_waitcnt vmcnt(11)
ds_write_b32 v53, v18 // (op 240) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 4242) 



ds_read_b96_tr_b6 v[90:92], v10 offset:41984 // (op 231) Transpose load local data 
ds_read_b96_tr_b6 v[93:95], v10 offset:46080 // (op 231) Transpose load local data 
s_barrier  // (op 4248) 



ds_read_b128 v[96:99], v4 offset:4096 // (op 233) Load local data 
ds_read_b128 v[100:103], v4 offset:4160 // (op 233) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4254) 


v_add_u32 v53, 74752, v11 // (op 235) 
ds_read_u8 v57, v53 // (op 235) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42016 // (op 357) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:95], v[96:103], a[0:3], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 238) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46112 // (op 357) Transpose load local data 




v_add_u32 v53, 74784, v11 // (op 361) 
ds_read_u8 v57, v53 // (op 361) Load local data 




ds_read_b96_tr_b6 v[90:92], v10 offset:42048 // (op 519) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:109], v[96:103], a[4:7], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 351) 
ds_read_b96_tr_b6 v[93:95], v10 offset:46144 // (op 519) Transpose load local data 




v_add_u32 v53, 74816, v11 // (op 523) 
ds_read_u8 v57, v53 // (op 523) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42080 // (op 681) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:95], v[96:103], a[8:11], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 513) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46176 // (op 681) Transpose load local data 




v_add_u32 v53, 74848, v11 // (op 685) 
ds_read_u8 v57, v53 // (op 685) Load local data 




ds_read_b96_tr_b6 v[90:92], v10 offset:42112 // (op 843) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:109], v[96:103], a[12:15], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 675) 
ds_read_b96_tr_b6 v[93:95], v10 offset:46208 // (op 843) Transpose load local data 




v_add_u32 v53, 74880, v11 // (op 847) 
ds_read_u8 v57, v53 // (op 847) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42144 // (op 1005) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:95], v[96:103], a[16:19], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 837) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46240 // (op 1005) Transpose load local data 




v_add_u32 v53, 74912, v11 // (op 1009) 
ds_read_u8 v57, v53 // (op 1009) Load local data 




ds_read_b96_tr_b6 v[90:92], v10 offset:42176 // (op 1167) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:109], v[96:103], a[20:23], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 999) 
ds_read_b96_tr_b6 v[93:95], v10 offset:46272 // (op 1167) Transpose load local data 




v_add_u32 v53, 74944, v11 // (op 1171) 
ds_read_u8 v57, v53 // (op 1171) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:42208 // (op 1329) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[96:103], a[24:27], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 1161) 
ds_read_b96_tr_b6 v[107:109], v10 offset:46304 // (op 1329) Transpose load local data 




s_barrier  // (op 4358) 
v_add_u32 v53, 74976, v11 // (op 1333) 
ds_read_u8 v57, v53 // (op 1333) Load local data 

s_barrier  // (op 4362) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:109], v[96:103], a[28:31], v57, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 1323) 






buffer_load_ubyte v29, v37, s[68:71], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 4251) 
v_add_u32 v21, 8, v21 // (op 3826) 


s_add_u32 s19, 4096, s1 // (op 3567) 

s_mov_b32 m0, s19 // (op 3567) reg expression
s_mov_b32 s49, 384 // (op 3567) call()
buffer_load_dwordx4 v14, s[52:55], s49 offen offset: 0 lds // (op 3567) Load value direct to lds




v_add_co_u32 v16, vcc, 256, v16 // (op 3698) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 3698) most significant half
s_mul_i32 s51, s29, 6 // (op 228) most significant: low of high * low
s_mul_i32 s50, s28, 6 // (op 228) least significant
s_mul_hi_u32 s19, s28, 6 // (op 228) most significant: high of low * low
s_add_u32 s51, s51, s19 // (op 228) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[50:51] // (op 228) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 228) 
s_mul_i32 s73, s51, 6 // (op 228) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 228) least significant
s_mul_hi_u32 s19, s50, 6 // (op 228) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 228) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 228) 

s_mul_i32 s19, 3, s50 // (op 228) 

v_add_u32 v37, v23, s19 // (op 228) 

buffer_load_dwordx3 v[58:60], v37, s[24:27], 0 offen // (op 228) Load value
buffer_load_dwordx3 v[62:64], v37, s[24:27], 0 offen offset:12 // (op 228) Load value
buffer_load_dwordx3 v[66:68], v37, s[24:27], 0 offen offset:24 // (op 228) Load value
buffer_load_dwordx3 v[70:72], v37, s[24:27], 0 offen offset:36 // (op 228) Load value
v_add_u32 v37, v37, v14 // (op 228) 
buffer_load_dwordx3 v[74:76], v37, s[24:27], 0 offen // (op 228) Load value
buffer_load_dwordx3 v[78:80], v37, s[24:27], 0 offen offset:12 // (op 228) Load value
buffer_load_dwordx3 v[82:84], v37, s[24:27], 0 offen offset:24 // (op 228) Load value
buffer_load_dwordx3 v[86:88], v37, s[24:27], 0 offen offset:36 // (op 228) Load value





s_lshl_b32 s19, s38, 2 // (op 242) 
s_mul_i32 s49, 3, s19 // (op 242) 

v_add_u32 v37, v45, s49 // (op 242) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 3634) 
s_mul_i32 s73, s51, 6 // (op 3634) most significant: low of high * low
s_mul_i32 s72, s50, 6 // (op 3634) least significant
s_mul_hi_u32 s19, s50, 6 // (op 3634) most significant: high of low * low
s_add_u32 s73, s73, s19 // (op 3634) most significant: sum


s_lshr_b64 s[50:51], s[72:73], 3 // (op 3634) 

s_lshl_b32 s19, s50, 1 // (op 3634) 

v_add_u32 v13, v13, s19 // (op 3634) 

buffer_load_dword v18, v37, s[64:67], 0 offen // (op 242) Load value


s_barrier  // (op 4205) 
s_lshl_b32 s19, s38, 3 // (op 3762) 



v_add_u32 v20, v20, s19 // (op 3762) 

s_waitcnt vmcnt(19)
ds_write_b96 v12, v[26:28] offset:8192 // (op 6) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v12, v[30:32] offset:8208 // (op 6) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v12, v[34:36] offset:8224 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v12, v[38:40] offset:8240 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v12, v[42:44] offset:8448 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v12, v[46:48] offset:8464 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v12, v[50:52] offset:8480 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v12, v[54:56] offset:8496 // (op 6) Store local data 
s_barrier  // (op 4231) 



s_waitcnt vmcnt(11)
ds_write_b32 v41, v33 offset:40960 // (op 21) Store local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 1286) 
s_mov_b32 s19, s50 // (op 1286) convert

s_and_b32 s49, s19, 4294967294 // (op 1286) 

s_cmp_lt_u32 s18, s49 // (op 1286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoop_1286 // (op 1286) Condition: Bottom (jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoop_1286 if true)
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoop_1286:














s_barrier  // (op 4342) 
s_mul_hi_i32 s0, s16, s8 // (op 3590) 
v_and_b32 v61, -4, v1 // (op 3590) 
s_add_i32 s1, s0, s16 // (op 3590) 

v_sub_u32 v65, v1, v61 // (op 3590) 

s_ashr_i32 s0, s1, 31 // (op 3590) 
v_lshlrev_b32 v61, 6, v65 // (op 3590) 

s_cmp_eq_i32 s8, 0 // (op 3590) 
s_mov_b32 s18, scc // (op 3590) 
v_lshrrev_b32 v65, 2, v1 // (op 3590) 
s_cmp_eq_i32 s18, 0 // (op 3590) 
s_cselect_b32 s19, 0, -1 // (op 3590) 

v_lshlrev_b32 v69, 1, v65 // (op 3590) 

s_and_b32 s18, s9, 31 // (op 3590) 
s_lshl_b32 s49, 1, s18 // (op 3590) 
s_add_i32 s50, s49, s19 // (op 3590) 


s_and_b32 s19, s0, s50 // (op 3590) 


s_add_i32 s0, s1, s19 // (op 3590) 


s_ashr_i32 s1, s0, s18 // (op 3590) 


s_lshl_b32 s0, s9, 24 // (op 3590) 
s_ashr_i32 s18, s0, 31 // (op 3590) 

s_xor_b32 s0, s1, s18 // (op 3590) 

s_sub_i32 s1, s0, s18 // (op 3590) 


s_mul_hi_i32 s0, s1, s10 // (op 3590) 
s_add_i32 s18, s0, s1 // (op 3590) 

s_ashr_i32 s0, s18, 31 // (op 3590) 
s_cmp_eq_i32 s10, 0 // (op 3590) 
s_mov_b32 s19, scc // (op 3590) 
s_cmp_eq_i32 s19, 0 // (op 3590) 
s_cselect_b32 s49, 0, -1 // (op 3590) 

s_and_b32 s19, s11, 31 // (op 3590) 
s_lshl_b32 s50, 1, s19 // (op 3590) 
s_add_i32 s51, s50, s49 // (op 3590) 


s_and_b32 s49, s0, s51 // (op 3590) 


s_add_i32 s0, s18, s49 // (op 3590) 


s_ashr_i32 s18, s0, s19 // (op 3590) 


s_lshl_b32 s0, s11, 24 // (op 3590) 
s_ashr_i32 s19, s0, 31 // (op 3590) 

s_xor_b32 s0, s18, s19 // (op 3590) 

s_sub_i32 s18, s0, s19 // (op 3590) 


s_mul_i32 s0, s1, s13 // (op 3590) 

s_sub_u32 s1, s16, s0 // (op 3590) 

s_mul_i32 s0, s18, s13 // (op 3590) 
s_add_u32 s0, s0, s1 // (op 3590) 


s_mul_hi_i32 s1, s16, s7 // (op 3590) 
s_add_i32 s18, s1, s16 // (op 3590) 

s_ashr_i32 s1, s18, 31 // (op 3590) 
s_cmp_eq_i32 s7, 0 // (op 3590) 
s_mov_b32 s19, scc // (op 3590) 
s_cmp_eq_i32 s19, 0 // (op 3590) 
s_cselect_b32 s49, 0, -1 // (op 3590) 

s_and_b32 s19, s14, 31 // (op 3590) 
s_lshl_b32 s50, 1, s19 // (op 3590) 
s_add_i32 s51, s50, s49 // (op 3590) 


s_and_b32 s49, s1, s51 // (op 3590) 


s_add_i32 s1, s18, s49 // (op 3590) 


s_ashr_i32 s18, s1, s19 // (op 3590) 


s_lshl_b32 s1, s14, 24 // (op 3590) 
s_ashr_i32 s19, s1, 31 // (op 3590) 

s_xor_b32 s1, s18, s19 // (op 3590) 

s_sub_i32 s18, s1, s19 // (op 3590) 


s_add_u32 s50, s4, 32 // (op 3590) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3590) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3590) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3590) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 3590) 

s_mul_i32 s1, s13, s50 // (op 3590) 

s_mul_i32 s19, s6, s1 // (op 3590) 

s_mul_i32 s1, s18, s19 // (op 3590) 

s_sub_u32 s19, s16, s1 // (op 3590) 

s_mov_b32 s1, s19 // (op 3590) 
s_mul_hi_i32 s49, s1, s12 // (op 3590) 
s_add_i32 s50, s49, s1 // (op 3590) 


s_ashr_i32 s1, s50, 31 // (op 3590) 
s_cmp_eq_i32 s12, 0 // (op 3590) 
s_mov_b32 s49, scc // (op 3590) 
s_cmp_eq_i32 s49, 0 // (op 3590) 
s_cselect_b32 s51, 0, -1 // (op 3590) 

s_and_b32 s49, s48, 31 // (op 3590) 
s_lshl_b32 s72, 1, s49 // (op 3590) 
s_add_i32 s73, s72, s51 // (op 3590) 


s_and_b32 s51, s1, s73 // (op 3590) 


s_add_i32 s1, s50, s51 // (op 3590) 


s_ashr_i32 s50, s1, s49 // (op 3590) 


s_lshl_b32 s1, s48, 24 // (op 3590) 
s_ashr_i32 s49, s1, 31 // (op 3590) 

s_xor_b32 s1, s50, s49 // (op 3590) 

s_sub_i32 s50, s1, s49 // (op 3590) 


s_mul_i32 s1, s15, s13 // (op 3590) 
s_add_u32 s72, s20, 256 // (op 3590) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 3590) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3590) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3590) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 8 // (op 3590) 

s_sub_i32 s49, s72, s1 // (op 3590) 

s_mul_i32 s51, s50, s49 // (op 3590) 


s_sub_u32 s49, s19, s51 // (op 3590) 


s_add_u32 s19, s49, s1 // (op 3590) 


s_cmp_eq_i32 s18, 0 // (op 3590) 
s_mov_b32 s1, scc // (op 3590) 

s_cmp_eq_i32 s1, 0 // (op 3590) 
s_cselect_b32 s18, s19, s0 // (op 3590) 



v_lshl_add_u32 v65, s18, 8, v61 // (op 3590) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3590) 
v_mov_b32 v13, 0 // (op 3590) convert
v_mov_b32 v12, v65 // (op 3590) convert

s_lshr_b32 s18, s0, 1 // (op 3590) 

v_lshl_add_u32 v61, s18, 8, v69 // (op 3590) 


v_mul_lo_u32 v15, v61, s29 // (op 3590) most significant: low of low * high
v_mul_hi_u32 v65, v61, s28 // (op 3590) most significant: high of low * low
v_mul_lo_u32 v14, v61, s28 // (op 3590) least significant: low of low * low
v_add_u32 v15, v15, v65 // (op 3590) most significant: sum


v_add_co_u32 v16, s[0:1], v12, v14 // (op 3590) least significant half
v_addc_co_u32 v17, s[0:1], v13, v15, s[0:1] // (op 3590) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 3590) most significant: low of high * low
v_mul_hi_u32 v61, v16, 6 // (op 3590) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 3590) least significant: low of low * low
v_add_u32 v13, v13, v61 // (op 3590) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3590) 

v_mov_b32 v61, v14 // (op 3590) convert

s_mul_i32 s1, s33, 6 // (op 3592) most significant: low of high * low
s_mul_i32 s0, s32, 6 // (op 3592) least significant
s_mul_hi_u32 s18, s32, 6 // (op 3592) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3592) most significant: sum

s_mul_i32 s19, s28, s1 // (op 3592) most significant: low of low * high
s_mul_i32 s49, s29, s0 // (op 3592) most significant: low of high * low
s_mul_i32 s18, s28, s0 // (op 3592) least significant
s_mul_hi_u32 s50, s28, s0 // (op 3592) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 3592) most significant: sum
s_add_u32 s19, s19, s50 // (op 3592) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 3592) 

s_mov_b32 s24, s30 // (op 3592) reg expression
s_mov_b32 s25, s31 // (op 3592) reg expression
s_mov_b32 s26, s0 // (op 3592) reg expression
s_mov_b32 s27, 131072 // (op 3592) call()

s_mul_hi_i32 s0, s16, s8 // (op 3664) 
v_lshrrev_b32 v65, 3, v1 // (op 3664) 
s_add_i32 s1, s0, s16 // (op 3664) 

v_and_b32 v69, -8, v1 // (op 3664) 
s_ashr_i32 s0, s1, 31 // (op 3664) 
v_sub_u32 v73, v1, v69 // (op 3664) 

s_cmp_eq_i32 s8, 0 // (op 3664) 
s_mov_b32 s18, scc // (op 3664) 
v_lshlrev_b32 v69, 4, v73 // (op 3664) 

s_cmp_eq_i32 s18, 0 // (op 3664) 
s_cselect_b32 s19, 0, -1 // (op 3664) 

s_and_b32 s18, s9, 31 // (op 3664) 
s_lshl_b32 s49, 1, s18 // (op 3664) 
s_add_i32 s50, s49, s19 // (op 3664) 


s_and_b32 s19, s0, s50 // (op 3664) 


s_add_i32 s0, s1, s19 // (op 3664) 


s_ashr_i32 s1, s0, s18 // (op 3664) 


s_lshl_b32 s0, s9, 24 // (op 3664) 
s_ashr_i32 s18, s0, 31 // (op 3664) 

s_xor_b32 s0, s1, s18 // (op 3664) 

s_sub_i32 s1, s0, s18 // (op 3664) 


s_mul_hi_i32 s0, s1, s10 // (op 3664) 
s_add_i32 s18, s0, s1 // (op 3664) 

s_ashr_i32 s0, s18, 31 // (op 3664) 
s_cmp_eq_i32 s10, 0 // (op 3664) 
s_mov_b32 s19, scc // (op 3664) 
s_cmp_eq_i32 s19, 0 // (op 3664) 
s_cselect_b32 s49, 0, -1 // (op 3664) 

s_and_b32 s19, s11, 31 // (op 3664) 
s_lshl_b32 s50, 1, s19 // (op 3664) 
s_add_i32 s51, s50, s49 // (op 3664) 


s_and_b32 s49, s0, s51 // (op 3664) 


s_add_i32 s0, s18, s49 // (op 3664) 


s_ashr_i32 s18, s0, s19 // (op 3664) 


s_lshl_b32 s0, s11, 24 // (op 3664) 
s_ashr_i32 s19, s0, 31 // (op 3664) 

s_xor_b32 s0, s18, s19 // (op 3664) 

s_sub_i32 s18, s0, s19 // (op 3664) 


s_add_u32 s50, s4, 32 // (op 3664) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3664) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3664) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3664) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 3664) 

s_mul_i32 s0, s18, s50 // (op 3664) 

s_sub_i32 s49, s1, s0 // (op 3664) 


s_mul_hi_i32 s0, s16, s7 // (op 3664) 
s_add_i32 s1, s0, s16 // (op 3664) 

s_ashr_i32 s0, s1, 31 // (op 3664) 
s_cmp_eq_i32 s7, 0 // (op 3664) 
s_mov_b32 s18, scc // (op 3664) 
s_cmp_eq_i32 s18, 0 // (op 3664) 
s_cselect_b32 s19, 0, -1 // (op 3664) 

s_and_b32 s18, s14, 31 // (op 3664) 
s_lshl_b32 s72, 1, s18 // (op 3664) 
s_add_i32 s73, s72, s19 // (op 3664) 


s_and_b32 s19, s0, s73 // (op 3664) 


s_add_i32 s0, s1, s19 // (op 3664) 


s_ashr_i32 s1, s0, s18 // (op 3664) 


s_lshl_b32 s0, s14, 24 // (op 3664) 
s_ashr_i32 s18, s0, 31 // (op 3664) 

s_xor_b32 s0, s1, s18 // (op 3664) 

s_sub_i32 s1, s0, s18 // (op 3664) 


s_mul_i32 s0, s13, s50 // (op 3664) 

s_mul_i32 s18, s6, s0 // (op 3664) 

s_mul_i32 s0, s1, s18 // (op 3664) 

s_sub_u32 s18, s16, s0 // (op 3664) 

s_mul_hi_i32 s0, s18, s12 // (op 3664) 
s_add_i32 s19, s0, s18 // (op 3664) 


s_ashr_i32 s0, s19, 31 // (op 3664) 
s_cmp_eq_i32 s12, 0 // (op 3664) 
s_mov_b32 s18, scc // (op 3664) 
s_cmp_eq_i32 s18, 0 // (op 3664) 
s_cselect_b32 s50, 0, -1 // (op 3664) 

s_and_b32 s18, s48, 31 // (op 3664) 
s_lshl_b32 s51, 1, s18 // (op 3664) 
s_add_i32 s72, s51, s50 // (op 3664) 


s_and_b32 s50, s0, s72 // (op 3664) 


s_add_i32 s0, s19, s50 // (op 3664) 


s_ashr_i32 s19, s0, s18 // (op 3664) 


s_lshl_b32 s0, s48, 24 // (op 3664) 
s_ashr_i32 s18, s0, 31 // (op 3664) 

s_xor_b32 s0, s19, s18 // (op 3664) 

s_sub_i32 s19, s0, s18 // (op 3664) 


s_cmp_eq_i32 s1, 0 // (op 3664) 
s_mov_b32 s0, scc // (op 3664) 

s_cmp_eq_i32 s0, 0 // (op 3664) 
s_cselect_b32 s1, s19, s49 // (op 3664) 



v_lshl_add_u32 v73, s1, 5, v65 // (op 3664) 


v_mul_lo_u32 v13, v73, s35 // (op 3664) most significant: low of low * high
v_mul_hi_u32 v65, v73, s34 // (op 3664) most significant: high of low * low
v_mul_lo_u32 v12, v73, s34 // (op 3664) least significant: low of low * low
v_add_u32 v13, v13, v65 // (op 3664) most significant: sum


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3664) 
s_lshr_b32 s18, s0, 1 // (op 3664) 

v_lshl_add_u32 v65, s18, 8, v69 // (op 3664) 


v_mov_b32 v15, 0 // (op 3664) convert
v_mov_b32 v14, v65 // (op 3664) convert

v_add_co_u32 v16, s[0:1], v14, v12 // (op 3664) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 3664) most significant half



s_mul_i32 s1, s34, s5 // (op 3666) most significant: low of low * high
s_mul_i32 s18, s35, s4 // (op 3666) most significant: low of high * low
s_mul_i32 s0, s34, s4 // (op 3666) least significant
s_mul_hi_u32 s19, s34, s4 // (op 3666) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3666) most significant: sum
s_add_u32 s1, s1, s19 // (op 3666) most significant: sum


s_mov_b32 s52, s36 // (op 3666) reg expression
s_mov_b32 s53, s37 // (op 3666) reg expression
s_mov_b32 s54, s0 // (op 3666) reg expression
s_mov_b32 s55, 131072 // (op 3666) call()

s_mul_hi_i32 s0, s16, s8 // (op 3728) 
v_and_b32 v65, -64, v1 // (op 3728) 
s_add_i32 s1, s0, s16 // (op 3728) 

v_sub_u32 v69, v1, v65 // (op 3728) 

s_ashr_i32 s0, s1, 31 // (op 3728) 
v_lshlrev_b32 v65, 2, v69 // (op 3728) 

s_cmp_eq_i32 s8, 0 // (op 3728) 
s_mov_b32 s18, scc // (op 3728) 
v_lshrrev_b32 v69, 6, v1 // (op 3728) 
s_cmp_eq_i32 s18, 0 // (op 3728) 
s_cselect_b32 s19, 0, -1 // (op 3728) 

s_and_b32 s18, s9, 31 // (op 3728) 
s_lshl_b32 s49, 1, s18 // (op 3728) 
s_add_i32 s50, s49, s19 // (op 3728) 


s_and_b32 s19, s0, s50 // (op 3728) 


s_add_i32 s0, s1, s19 // (op 3728) 


s_ashr_i32 s1, s0, s18 // (op 3728) 


s_lshl_b32 s0, s9, 24 // (op 3728) 
s_ashr_i32 s18, s0, 31 // (op 3728) 

s_xor_b32 s0, s1, s18 // (op 3728) 

s_sub_i32 s1, s0, s18 // (op 3728) 


s_mul_hi_i32 s0, s1, s10 // (op 3728) 
s_add_i32 s18, s0, s1 // (op 3728) 

s_ashr_i32 s0, s18, 31 // (op 3728) 
s_cmp_eq_i32 s10, 0 // (op 3728) 
s_mov_b32 s19, scc // (op 3728) 
s_cmp_eq_i32 s19, 0 // (op 3728) 
s_cselect_b32 s49, 0, -1 // (op 3728) 

s_and_b32 s19, s11, 31 // (op 3728) 
s_lshl_b32 s50, 1, s19 // (op 3728) 
s_add_i32 s51, s50, s49 // (op 3728) 


s_and_b32 s49, s0, s51 // (op 3728) 


s_add_i32 s0, s18, s49 // (op 3728) 


s_ashr_i32 s18, s0, s19 // (op 3728) 


s_lshl_b32 s0, s11, 24 // (op 3728) 
s_ashr_i32 s19, s0, 31 // (op 3728) 

s_xor_b32 s0, s18, s19 // (op 3728) 

s_sub_i32 s18, s0, s19 // (op 3728) 


s_mul_i32 s0, s1, s13 // (op 3728) 

s_sub_u32 s1, s16, s0 // (op 3728) 

s_mul_i32 s0, s18, s13 // (op 3728) 
s_add_u32 s0, s0, s1 // (op 3728) 


s_mul_hi_i32 s1, s16, s7 // (op 3728) 
s_add_i32 s18, s1, s16 // (op 3728) 

s_ashr_i32 s1, s18, 31 // (op 3728) 
s_cmp_eq_i32 s7, 0 // (op 3728) 
s_mov_b32 s19, scc // (op 3728) 
s_cmp_eq_i32 s19, 0 // (op 3728) 
s_cselect_b32 s49, 0, -1 // (op 3728) 

s_and_b32 s19, s14, 31 // (op 3728) 
s_lshl_b32 s50, 1, s19 // (op 3728) 
s_add_i32 s51, s50, s49 // (op 3728) 


s_and_b32 s49, s1, s51 // (op 3728) 


s_add_i32 s1, s18, s49 // (op 3728) 


s_ashr_i32 s18, s1, s19 // (op 3728) 


s_lshl_b32 s1, s14, 24 // (op 3728) 
s_ashr_i32 s19, s1, 31 // (op 3728) 

s_xor_b32 s1, s18, s19 // (op 3728) 

s_sub_i32 s18, s1, s19 // (op 3728) 


s_add_u32 s50, s4, 32 // (op 3728) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3728) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3728) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3728) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 3728) 

s_mul_i32 s1, s13, s50 // (op 3728) 

s_mul_i32 s19, s6, s1 // (op 3728) 

s_mul_i32 s1, s18, s19 // (op 3728) 

s_sub_u32 s19, s16, s1 // (op 3728) 

s_mov_b32 s1, s19 // (op 3728) 
s_mul_hi_i32 s49, s1, s12 // (op 3728) 
s_add_i32 s50, s49, s1 // (op 3728) 


s_ashr_i32 s1, s50, 31 // (op 3728) 
s_cmp_eq_i32 s12, 0 // (op 3728) 
s_mov_b32 s49, scc // (op 3728) 
s_cmp_eq_i32 s49, 0 // (op 3728) 
s_cselect_b32 s51, 0, -1 // (op 3728) 

s_and_b32 s49, s48, 31 // (op 3728) 
s_lshl_b32 s72, 1, s49 // (op 3728) 
s_add_i32 s73, s72, s51 // (op 3728) 


s_and_b32 s51, s1, s73 // (op 3728) 


s_add_i32 s1, s50, s51 // (op 3728) 


s_ashr_i32 s50, s1, s49 // (op 3728) 


s_lshl_b32 s1, s48, 24 // (op 3728) 
s_ashr_i32 s49, s1, 31 // (op 3728) 

s_xor_b32 s1, s50, s49 // (op 3728) 

s_sub_i32 s50, s1, s49 // (op 3728) 


s_mul_i32 s1, s15, s13 // (op 3728) 
s_add_u32 s72, s20, 256 // (op 3728) Start of Int64 add, locking SCC
s_addc_u32 s73, s21, 0 // (op 3728) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 3728) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 3728) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 8 // (op 3728) 

s_sub_i32 s49, s72, s1 // (op 3728) 

s_mul_i32 s51, s50, s49 // (op 3728) 


s_sub_u32 s49, s19, s51 // (op 3728) 


s_add_u32 s19, s49, s1 // (op 3728) 


s_cmp_eq_i32 s18, 0 // (op 3728) 
s_mov_b32 s1, scc // (op 3728) 

s_cmp_eq_i32 s1, 0 // (op 3728) 
s_cselect_b32 s18, s19, s0 // (op 3728) 



v_lshl_add_u32 v73, s18, 8, v65 // (op 3728) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 3728) 
s_lshr_b32 s18, s0, 1 // (op 3728) 

v_lshl_add_u32 v65, s18, 3, v69 // (op 3728) 


v_mul_lo_u32 v69, v65, s38 // (op 3728) 

v_add_u32 v65, v73, v69 // (op 3728) 


s_mul_i32 s1, s38, s43 // (op 3730) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 3730) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 3730) least significant
s_mul_hi_u32 s19, s38, s42 // (op 3730) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3730) most significant: sum
s_add_u32 s1, s1, s19 // (op 3730) most significant: sum


s_mov_b32 s64, s40 // (op 3730) reg expression
s_mov_b32 s65, s41 // (op 3730) reg expression
s_mov_b32 s66, s0 // (op 3730) reg expression
s_mov_b32 s67, 131072 // (op 3730) call()

s_mul_hi_i32 s0, s16, s8 // (op 3792) 
v_lshrrev_b32 v69, 6, v1 // (op 3792) 
s_add_i32 s1, s0, s16 // (op 3792) 

v_and_b32 v73, 1, v69 // (op 3792) 

s_ashr_i32 s0, s1, 31 // (op 3792) 
v_and_b32 v69, 15, v1 // (op 3792) 
s_cmp_eq_i32 s8, 0 // (op 3792) 
s_mov_b32 s18, scc // (op 3792) 
v_lshl_add_u32 v77, v73, 4, v69 // (op 3792) 


s_cmp_eq_i32 s18, 0 // (op 3792) 
s_cselect_b32 s19, 0, -1 // (op 3792) 

v_and_b32 v69, 63, v1 // (op 3792) 
s_and_b32 s18, s9, 31 // (op 3792) 
v_lshrrev_b32 v73, 4, v69 // (op 3792) 

s_lshl_b32 s49, 1, s18 // (op 3792) 
s_add_i32 s50, s49, s19 // (op 3792) 


s_and_b32 s19, s0, s50 // (op 3792) 


s_add_i32 s0, s1, s19 // (op 3792) 


s_ashr_i32 s1, s0, s18 // (op 3792) 


s_lshl_b32 s0, s9, 24 // (op 3792) 
s_ashr_i32 s18, s0, 31 // (op 3792) 

s_xor_b32 s0, s1, s18 // (op 3792) 

s_sub_i32 s1, s0, s18 // (op 3792) 


s_mul_hi_i32 s0, s1, s10 // (op 3792) 
s_add_i32 s18, s0, s1 // (op 3792) 

s_ashr_i32 s0, s18, 31 // (op 3792) 
s_cmp_eq_i32 s10, 0 // (op 3792) 
s_mov_b32 s19, scc // (op 3792) 
s_cmp_eq_i32 s19, 0 // (op 3792) 
s_cselect_b32 s49, 0, -1 // (op 3792) 

s_and_b32 s19, s11, 31 // (op 3792) 
s_lshl_b32 s50, 1, s19 // (op 3792) 
s_add_i32 s51, s50, s49 // (op 3792) 


s_and_b32 s49, s0, s51 // (op 3792) 


s_add_i32 s0, s18, s49 // (op 3792) 


s_ashr_i32 s18, s0, s19 // (op 3792) 


s_lshl_b32 s0, s11, 24 // (op 3792) 
s_ashr_i32 s19, s0, 31 // (op 3792) 

s_xor_b32 s0, s18, s19 // (op 3792) 

s_sub_i32 s18, s0, s19 // (op 3792) 


s_add_u32 s50, s4, 32 // (op 3792) Start of Int64 add, locking SCC
s_addc_u32 s51, s5, 0 // (op 3792) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 3792) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 3792) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 5 // (op 3792) 

s_mul_i32 s0, s18, s50 // (op 3792) 

s_sub_i32 s49, s1, s0 // (op 3792) 


s_mul_hi_i32 s0, s16, s7 // (op 3792) 
s_add_i32 s1, s0, s16 // (op 3792) 

s_ashr_i32 s0, s1, 31 // (op 3792) 
s_cmp_eq_i32 s7, 0 // (op 3792) 
s_mov_b32 s18, scc // (op 3792) 
s_cmp_eq_i32 s18, 0 // (op 3792) 
s_cselect_b32 s19, 0, -1 // (op 3792) 

s_and_b32 s18, s14, 31 // (op 3792) 
s_lshl_b32 s72, 1, s18 // (op 3792) 
s_add_i32 s73, s72, s19 // (op 3792) 


s_and_b32 s19, s0, s73 // (op 3792) 


s_add_i32 s0, s1, s19 // (op 3792) 


s_ashr_i32 s1, s0, s18 // (op 3792) 


s_lshl_b32 s0, s14, 24 // (op 3792) 
s_ashr_i32 s18, s0, 31 // (op 3792) 

s_xor_b32 s0, s1, s18 // (op 3792) 

s_sub_i32 s1, s0, s18 // (op 3792) 


s_mul_i32 s0, s13, s50 // (op 3792) 

s_mul_i32 s18, s6, s0 // (op 3792) 

s_mul_i32 s0, s1, s18 // (op 3792) 

s_sub_u32 s18, s16, s0 // (op 3792) 

s_mul_hi_i32 s0, s18, s12 // (op 3792) 
s_add_i32 s19, s0, s18 // (op 3792) 


s_ashr_i32 s0, s19, 31 // (op 3792) 
s_cmp_eq_i32 s12, 0 // (op 3792) 
s_mov_b32 s18, scc // (op 3792) 
s_cmp_eq_i32 s18, 0 // (op 3792) 
s_cselect_b32 s50, 0, -1 // (op 3792) 

s_and_b32 s18, s48, 31 // (op 3792) 
s_lshl_b32 s51, 1, s18 // (op 3792) 
s_add_i32 s72, s51, s50 // (op 3792) 


s_and_b32 s50, s0, s72 // (op 3792) 


s_add_i32 s0, s19, s50 // (op 3792) 


s_ashr_i32 s19, s0, s18 // (op 3792) 


s_lshl_b32 s0, s48, 24 // (op 3792) 
s_ashr_i32 s18, s0, 31 // (op 3792) 

s_xor_b32 s0, s19, s18 // (op 3792) 

s_sub_i32 s19, s0, s18 // (op 3792) 


s_cmp_eq_i32 s1, 0 // (op 3792) 
s_mov_b32 s0, scc // (op 3792) 

s_cmp_eq_i32 s0, 0 // (op 3792) 
s_cselect_b32 s1, s19, s49 // (op 3792) 



v_lshl_add_u32 v69, s1, 5, v77 // (op 3792) 


v_mul_lo_u32 v77, v69, s44 // (op 3792) 

s_ashr_i64 s[0:1], s[32:33], 7 // (op 3792) 
s_lshr_b32 s18, s0, 1 // (op 3792) 

v_lshl_add_u32 v69, s18, 3, v73 // (op 3792) 


v_add_u32 v73, v69, v77 // (op 3792) 


s_mul_i32 s1, s44, s5 // (op 3794) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 3794) most significant: low of high * low
s_mul_i32 s0, s44, s4 // (op 3794) least significant
s_mul_hi_u32 s19, s44, s4 // (op 3794) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3794) most significant: sum
s_add_u32 s1, s1, s19 // (op 3794) most significant: sum


s_mov_b32 s68, s46 // (op 3794) reg expression
s_mov_b32 s69, s47 // (op 3794) reg expression
s_mov_b32 s70, s0 // (op 3794) reg expression
s_mov_b32 s71, 131072 // (op 3794) call()

v_and_b32 v69, -4, v1 // (op 3885) 
v_sub_u32 v77, v1, v69 // (op 3885) 

v_lshlrev_b32 v69, 6, v77 // (op 3885) 

v_lshrrev_b32 v77, 2, v1 // (op 3885) 
v_lshl_add_u32 v81, v77, 9, v69 // (op 3885) 


v_mul_lo_u32 v69, v81, 6 // (op 3885) 
v_lshrrev_b32 v77, 3, v69 // (op 3885) 

v_lshrrev_b32 v69, 4, v81 // (op 3885) 

v_lshlrev_b32 v81, 2, v69 // (op 3885) 

v_add_u32 v69, v77, v81 // (op 3885) 


v_and_b32 v77, -8, v1 // (op 3943) 
v_sub_u32 v81, v1, v77 // (op 3943) 

v_lshlrev_b32 v77, 4, v81 // (op 3943) 

v_lshrrev_b32 v81, 3, v1 // (op 3943) 
v_lshl_add_u32 v85, v81, 7, v77 // (op 3943) 


s_nop 0
v_readfirstlane_b32 s0, v85 // (op 3943) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v77, -64, v1 // (op 3995) 
v_sub_u32 v81, v1, v77 // (op 3995) 

v_lshlrev_b32 v77, 2, v81 // (op 3995) 

v_lshrrev_b32 v81, 6, v1 // (op 3995) 
v_lshl_add_u32 v85, v81, 8, v77 // (op 3995) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 1292) 
s_and_b32 s1, s18, 4294967294 // (op 1292) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 1291) 
s_mov_b32 s49, s18 // (op 1291) convert

s_cmp_lt_u32 s1, s49 // (op 1291) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_1291 // (op 1291) Condition: Top (jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_1291 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoopTail_1291:

s_add_u32 s1, s1, 1 // (op 1293) 
s_mul_i32 s19, s29, 6 // (op 170) most significant: low of high * low
s_mul_i32 s18, s28, 6 // (op 170) least significant
s_mul_hi_u32 s49, s28, 6 // (op 170) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 170) most significant: sum

v_lshrrev_b64 v[12:13], 3, s[18:19] // (op 170) 


v_mov_b32 v14, v61 // (op 170) 
v_mov_b32 v15, v14 // (op 170) reg expression
buffer_load_dwordx3 v[18:20], v15, s[24:27], 0 offen // (op 170) Load value
buffer_load_dwordx3 v[22:24], v15, s[24:27], 0 offen offset:12 // (op 170) Load value
buffer_load_dwordx3 v[26:28], v15, s[24:27], 0 offen offset:24 // (op 170) Load value
buffer_load_dwordx3 v[30:32], v15, s[24:27], 0 offen offset:36 // (op 170) Load value
v_add_u32 v15, v15, v12 // (op 170) 
buffer_load_dwordx3 v[34:36], v15, s[24:27], 0 offen // (op 170) Load value
buffer_load_dwordx3 v[38:40], v15, s[24:27], 0 offen offset:12 // (op 170) Load value
buffer_load_dwordx3 v[42:44], v15, s[24:27], 0 offen offset:24 // (op 170) Load value
buffer_load_dwordx3 v[46:48], v15, s[24:27], 0 offen offset:36 // (op 170) Load value



s_lshl_b64 s[18:19], s[28:29], 7 // (op 3580) 
s_mul_i32 s51, s19, 6 // (op 3580) most significant: low of high * low
s_mul_i32 s50, s18, 6 // (op 3580) least significant
s_mul_hi_u32 s49, s18, 6 // (op 3580) most significant: high of low * low
s_add_u32 s51, s51, s49 // (op 3580) most significant: sum


s_lshr_b64 s[18:19], s[50:51], 3 // (op 3580) 

v_add_u32 v61, v61, s18 // (op 3580) 

s_barrier  // (op 4208) 
s_barrier  // (op 4221) 





v_mov_b64 v[12:13], v[16:17] // (op 3560) 

s_mov_b32 m0, s0 // (op 3560) reg expression
buffer_load_dwordx4 v12, s[52:55], 0 offen offset: 0 lds // (op 3560) Load value direct to lds

v_add_co_u32 v16, vcc, 128, v16 // (op 3656) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 3656) most significant half
s_waitcnt vmcnt(8)
ds_write_b96 v69, v[18:20] offset:8192 // (op 171) Store local data 


v_mov_b32 v21, v65 // (op 184) 
v_mov_b32 v25, v21 // (op 184) reg expression

s_waitcnt vmcnt(7)
ds_write_b96 v69, v[22:24] offset:8208 // (op 171) Store local data 
s_waitcnt vmcnt(6)
ds_write_b96 v69, v[26:28] offset:8224 // (op 171) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v69, v[30:32] offset:8240 // (op 171) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v69, v[34:36] offset:8448 // (op 171) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v69, v[38:40] offset:8464 // (op 171) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v69, v[42:44] offset:8480 // (op 171) Store local data 
buffer_load_dword v29, v25, s[64:67], 0 offen // (op 184) Load value


s_lshl_b32 s18, s38, 2 // (op 3720) 
v_add_u32 v65, v65, s18 // (op 3720) 

s_waitcnt vmcnt(2)
ds_write_b96 v69, v[46:48] offset:8496 // (op 171) Store local data 

s_barrier  // (op 4234) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4211) 






ds_read_b96_tr_b6 v[18:20], v0 offset:8192 // (op 173) Transpose load local data 
ds_read_b96_tr_b6 v[21:23], v0 offset:12288 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v85, v29 offset:40960 // (op 182) Store local data 

s_barrier  // (op 4224) 



ds_read_b128 v[24:27], v3 // (op 175) Load local data 
ds_read_b128 v[28:31], v3 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4238) 


ds_read_u8 v12, v7 offset:40960 // (op 177) Load local data 


v_mov_b32 v13, v73 // (op 179) 

buffer_load_ubyte v14, v13, s[68:71], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v73, 4, v73 // (op 3784) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:31], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 180) 
ds_read_b96_tr_b6 v[32:34], v0 offset:8224 // (op 325) Transpose load local data 


ds_read_b96_tr_b6 v[35:37], v0 offset:12320 // (op 325) Transpose load local data 


ds_read_u8 v15, v7 offset:40992 // (op 329) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:37], v[24:31], a[4:7], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 331) 
ds_read_b96_tr_b6 v[18:20], v0 offset:8256 // (op 487) Transpose load local data 


ds_read_b96_tr_b6 v[21:23], v0 offset:12352 // (op 487) Transpose load local data 


ds_read_u8 v15, v7 offset:41024 // (op 491) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[24:31], a[8:11], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 493) 
ds_read_b96_tr_b6 v[32:34], v0 offset:8288 // (op 649) Transpose load local data 


ds_read_b96_tr_b6 v[35:37], v0 offset:12384 // (op 649) Transpose load local data 


ds_read_u8 v15, v7 offset:41056 // (op 653) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:37], v[24:31], a[12:15], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 655) 
ds_read_b96_tr_b6 v[18:20], v0 offset:8320 // (op 811) Transpose load local data 


ds_read_b96_tr_b6 v[21:23], v0 offset:12416 // (op 811) Transpose load local data 


ds_read_u8 v15, v7 offset:41088 // (op 815) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[18:23], v[24:31], a[16:19], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 817) 
ds_read_b96_tr_b6 v[32:34], v0 offset:8352 // (op 973) Transpose load local data 


ds_read_b96_tr_b6 v[35:37], v0 offset:12448 // (op 973) Transpose load local data 


ds_read_u8 v15, v7 offset:41120 // (op 977) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[32:37], v[24:31], a[20:23], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 979) 
ds_read_b96_tr_b6 v[18:20], v0 offset:8384 // (op 1135) Transpose load local data 


ds_read_b96_tr_b6 v[21:23], v0 offset:12480 // (op 1135) Transpose load local data 


ds_read_u8 v15, v7 offset:41152 // (op 1139) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[18:23], v[24:31], a[24:27], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 1141) 
ds_read_b96_tr_b6 v[32:34], v0 offset:8416 // (op 1297) Transpose load local data 


ds_read_b96_tr_b6 v[35:37], v0 offset:12512 // (op 1297) Transpose load local data 


ds_read_u8 v15, v7 offset:41184 // (op 1301) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[32:37], v[24:31], a[28:31], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b001 // (op 1303) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 1291) 
s_mov_b32 s49, s18 // (op 1291) convert

s_cmp_lt_u32 s1, s49 // (op 1291) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoopTail_1291 // (op 1291) Condition: Bottom (jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopTop_KLoopTail_1291 if true)
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_1291:






















s_barrier  // (op 4346) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 

v_cvt_f16_f32 v10, v14 // (op 52) 
v_cvt_f16_f32 v11, v15 // (op 52) 
v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 


v_pack_b32_f16 v14, v10, v11 // (op 54) 
v_pack_b32_f16 v15, v12, v13 // (op 54) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen // (op 54) Store value


v_accvgpr_read v10, a4 // (op 1621) 
v_accvgpr_read v11, a5 // (op 1621) 
v_accvgpr_read v12, a6 // (op 1621) 
v_accvgpr_read v13, a7 // (op 1621) 
v_mul_f32 v14, v8, v10 // (op 1621) 
v_mul_f32 v15, v8, v11 // (op 1621) 
v_mul_f32 v16, v8, v12 // (op 1621) 
v_mul_f32 v17, v8, v13 // (op 1621) 

v_cvt_f16_f32 v10, v14 // (op 1623) 
v_cvt_f16_f32 v11, v15 // (op 1623) 
v_cvt_f16_f32 v12, v16 // (op 1623) 
v_cvt_f16_f32 v13, v17 // (op 1623) 


v_pack_b32_f16 v14, v10, v11 // (op 1624) 
v_pack_b32_f16 v15, v12, v13 // (op 1624) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:64 // (op 1624) Store value


v_accvgpr_read v10, a8 // (op 1641) 
v_accvgpr_read v11, a9 // (op 1641) 
v_accvgpr_read v12, a10 // (op 1641) 
v_accvgpr_read v13, a11 // (op 1641) 
v_mul_f32 v14, v8, v10 // (op 1641) 
v_mul_f32 v15, v8, v11 // (op 1641) 
v_mul_f32 v16, v8, v12 // (op 1641) 
v_mul_f32 v17, v8, v13 // (op 1641) 

v_cvt_f16_f32 v10, v14 // (op 1643) 
v_cvt_f16_f32 v11, v15 // (op 1643) 
v_cvt_f16_f32 v12, v16 // (op 1643) 
v_cvt_f16_f32 v13, v17 // (op 1643) 


v_pack_b32_f16 v14, v10, v11 // (op 1644) 
v_pack_b32_f16 v15, v12, v13 // (op 1644) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:128 // (op 1644) Store value


v_accvgpr_read v10, a12 // (op 1661) 
v_accvgpr_read v11, a13 // (op 1661) 
v_accvgpr_read v12, a14 // (op 1661) 
v_accvgpr_read v13, a15 // (op 1661) 
v_mul_f32 v14, v8, v10 // (op 1661) 
v_mul_f32 v15, v8, v11 // (op 1661) 
v_mul_f32 v16, v8, v12 // (op 1661) 
v_mul_f32 v17, v8, v13 // (op 1661) 

v_cvt_f16_f32 v10, v14 // (op 1663) 
v_cvt_f16_f32 v11, v15 // (op 1663) 
v_cvt_f16_f32 v12, v16 // (op 1663) 
v_cvt_f16_f32 v13, v17 // (op 1663) 


v_pack_b32_f16 v14, v10, v11 // (op 1664) 
v_pack_b32_f16 v15, v12, v13 // (op 1664) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:192 // (op 1664) Store value


v_accvgpr_read v10, a16 // (op 1681) 
v_accvgpr_read v11, a17 // (op 1681) 
v_accvgpr_read v12, a18 // (op 1681) 
v_accvgpr_read v13, a19 // (op 1681) 
v_mul_f32 v14, v8, v10 // (op 1681) 
v_mul_f32 v15, v8, v11 // (op 1681) 
v_mul_f32 v16, v8, v12 // (op 1681) 
v_mul_f32 v17, v8, v13 // (op 1681) 

v_cvt_f16_f32 v10, v14 // (op 1683) 
v_cvt_f16_f32 v11, v15 // (op 1683) 
v_cvt_f16_f32 v12, v16 // (op 1683) 
v_cvt_f16_f32 v13, v17 // (op 1683) 


v_pack_b32_f16 v14, v10, v11 // (op 1684) 
v_pack_b32_f16 v15, v12, v13 // (op 1684) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:256 // (op 1684) Store value


v_accvgpr_read v10, a20 // (op 1701) 
v_accvgpr_read v11, a21 // (op 1701) 
v_accvgpr_read v12, a22 // (op 1701) 
v_accvgpr_read v13, a23 // (op 1701) 
v_mul_f32 v14, v8, v10 // (op 1701) 
v_mul_f32 v15, v8, v11 // (op 1701) 
v_mul_f32 v16, v8, v12 // (op 1701) 
v_mul_f32 v17, v8, v13 // (op 1701) 

v_cvt_f16_f32 v10, v14 // (op 1703) 
v_cvt_f16_f32 v11, v15 // (op 1703) 
v_cvt_f16_f32 v12, v16 // (op 1703) 
v_cvt_f16_f32 v13, v17 // (op 1703) 


v_pack_b32_f16 v14, v10, v11 // (op 1704) 
v_pack_b32_f16 v15, v12, v13 // (op 1704) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:320 // (op 1704) Store value


v_accvgpr_read v10, a24 // (op 1721) 
v_accvgpr_read v11, a25 // (op 1721) 
v_accvgpr_read v12, a26 // (op 1721) 
v_accvgpr_read v13, a27 // (op 1721) 
v_mul_f32 v14, v8, v10 // (op 1721) 
v_mul_f32 v15, v8, v11 // (op 1721) 
v_mul_f32 v16, v8, v12 // (op 1721) 
v_mul_f32 v17, v8, v13 // (op 1721) 

v_cvt_f16_f32 v10, v14 // (op 1723) 
v_cvt_f16_f32 v11, v15 // (op 1723) 
v_cvt_f16_f32 v12, v16 // (op 1723) 
v_cvt_f16_f32 v13, v17 // (op 1723) 


v_pack_b32_f16 v14, v10, v11 // (op 1724) 
v_pack_b32_f16 v15, v12, v13 // (op 1724) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:384 // (op 1724) Store value


v_accvgpr_read v10, a28 // (op 1741) 
v_accvgpr_read v11, a29 // (op 1741) 
v_accvgpr_read v12, a30 // (op 1741) 
v_accvgpr_read v13, a31 // (op 1741) 
v_mul_f32 v14, v8, v10 // (op 1741) 
v_mul_f32 v15, v8, v11 // (op 1741) 
v_mul_f32 v16, v8, v12 // (op 1741) 
v_mul_f32 v17, v8, v13 // (op 1741) 

v_cvt_f16_f32 v10, v14 // (op 1743) 
v_cvt_f16_f32 v11, v15 // (op 1743) 
v_cvt_f16_f32 v12, v16 // (op 1743) 
v_cvt_f16_f32 v13, v17 // (op 1743) 


v_pack_b32_f16 v14, v10, v11 // (op 1744) 
v_pack_b32_f16 v15, v12, v13 // (op 1744) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:448 // (op 1744) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:448 // (op 1924) Load value
v_accvgpr_read v12, a28 // (op 1920) 
v_accvgpr_read v13, a29 // (op 1920) 
v_accvgpr_read v14, a30 // (op 1920) 
v_accvgpr_read v15, a31 // (op 1920) 
v_mul_f32 v16, v8, v12 // (op 1920) 
v_mul_f32 v17, v8, v13 // (op 1920) 
v_mul_f32 v18, v8, v14 // (op 1920) 
v_mul_f32 v19, v8, v15 // (op 1920) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1925) 
v_lshrrev_b32 v13, 16, v10 // (op 1925) 
v_cvt_f32_f16 v13, v13 // (op 1925) 
v_mul_f32 v20, v5, v12 // (op 1925) 
v_mul_f32 v21, v5, v13 // (op 1925) 
v_cvt_f32_f16 v12, v11 // (op 1925) 
v_lshrrev_b32 v13, 16, v11 // (op 1925) 
v_cvt_f32_f16 v13, v13 // (op 1925) 
v_mul_f32 v22, v5, v12 // (op 1925) 
v_mul_f32 v23, v5, v13 // (op 1925) 


v_add_f32 v10, v20, v16 // (op 1921) 
v_add_f32 v11, v21, v17 // (op 1921) 
v_add_f32 v12, v22, v18 // (op 1921) 
v_add_f32 v13, v23, v19 // (op 1921) 


v_cvt_f16_f32 v14, v10 // (op 1922) 
v_cvt_f16_f32 v15, v11 // (op 1922) 
v_cvt_f16_f32 v16, v12 // (op 1922) 
v_cvt_f16_f32 v17, v13 // (op 1922) 


v_pack_b32_f16 v10, v14, v15 // (op 1923) 
v_pack_b32_f16 v11, v16, v17 // (op 1923) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:448 // (op 1923) Store value





buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:384 // (op 1904) Load value
v_accvgpr_read v12, a24 // (op 1900) 
v_accvgpr_read v13, a25 // (op 1900) 
v_accvgpr_read v14, a26 // (op 1900) 
v_accvgpr_read v15, a27 // (op 1900) 
v_mul_f32 v16, v8, v12 // (op 1900) 
v_mul_f32 v17, v8, v13 // (op 1900) 
v_mul_f32 v18, v8, v14 // (op 1900) 
v_mul_f32 v19, v8, v15 // (op 1900) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1905) 
v_lshrrev_b32 v13, 16, v10 // (op 1905) 
v_cvt_f32_f16 v13, v13 // (op 1905) 
v_mul_f32 v20, v5, v12 // (op 1905) 
v_mul_f32 v21, v5, v13 // (op 1905) 
v_cvt_f32_f16 v12, v11 // (op 1905) 
v_lshrrev_b32 v13, 16, v11 // (op 1905) 
v_cvt_f32_f16 v13, v13 // (op 1905) 
v_mul_f32 v22, v5, v12 // (op 1905) 
v_mul_f32 v23, v5, v13 // (op 1905) 


v_add_f32 v10, v20, v16 // (op 1901) 
v_add_f32 v11, v21, v17 // (op 1901) 
v_add_f32 v12, v22, v18 // (op 1901) 
v_add_f32 v13, v23, v19 // (op 1901) 


v_cvt_f16_f32 v14, v10 // (op 1902) 
v_cvt_f16_f32 v15, v11 // (op 1902) 
v_cvt_f16_f32 v16, v12 // (op 1902) 
v_cvt_f16_f32 v17, v13 // (op 1902) 


v_pack_b32_f16 v10, v14, v15 // (op 1903) 
v_pack_b32_f16 v11, v16, v17 // (op 1903) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:384 // (op 1903) Store value





buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:320 // (op 1884) Load value
v_accvgpr_read v12, a20 // (op 1880) 
v_accvgpr_read v13, a21 // (op 1880) 
v_accvgpr_read v14, a22 // (op 1880) 
v_accvgpr_read v15, a23 // (op 1880) 
v_mul_f32 v16, v8, v12 // (op 1880) 
v_mul_f32 v17, v8, v13 // (op 1880) 
v_mul_f32 v18, v8, v14 // (op 1880) 
v_mul_f32 v19, v8, v15 // (op 1880) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1885) 
v_lshrrev_b32 v13, 16, v10 // (op 1885) 
v_cvt_f32_f16 v13, v13 // (op 1885) 
v_mul_f32 v20, v5, v12 // (op 1885) 
v_mul_f32 v21, v5, v13 // (op 1885) 
v_cvt_f32_f16 v12, v11 // (op 1885) 
v_lshrrev_b32 v13, 16, v11 // (op 1885) 
v_cvt_f32_f16 v13, v13 // (op 1885) 
v_mul_f32 v22, v5, v12 // (op 1885) 
v_mul_f32 v23, v5, v13 // (op 1885) 


v_add_f32 v10, v20, v16 // (op 1881) 
v_add_f32 v11, v21, v17 // (op 1881) 
v_add_f32 v12, v22, v18 // (op 1881) 
v_add_f32 v13, v23, v19 // (op 1881) 


v_cvt_f16_f32 v14, v10 // (op 1882) 
v_cvt_f16_f32 v15, v11 // (op 1882) 
v_cvt_f16_f32 v16, v12 // (op 1882) 
v_cvt_f16_f32 v17, v13 // (op 1882) 


v_pack_b32_f16 v10, v14, v15 // (op 1883) 
v_pack_b32_f16 v11, v16, v17 // (op 1883) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:320 // (op 1883) Store value





buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:256 // (op 1864) Load value
v_accvgpr_read v12, a16 // (op 1860) 
v_accvgpr_read v13, a17 // (op 1860) 
v_accvgpr_read v14, a18 // (op 1860) 
v_accvgpr_read v15, a19 // (op 1860) 
v_mul_f32 v16, v8, v12 // (op 1860) 
v_mul_f32 v17, v8, v13 // (op 1860) 
v_mul_f32 v18, v8, v14 // (op 1860) 
v_mul_f32 v19, v8, v15 // (op 1860) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1865) 
v_lshrrev_b32 v13, 16, v10 // (op 1865) 
v_cvt_f32_f16 v13, v13 // (op 1865) 
v_mul_f32 v20, v5, v12 // (op 1865) 
v_mul_f32 v21, v5, v13 // (op 1865) 
v_cvt_f32_f16 v12, v11 // (op 1865) 
v_lshrrev_b32 v13, 16, v11 // (op 1865) 
v_cvt_f32_f16 v13, v13 // (op 1865) 
v_mul_f32 v22, v5, v12 // (op 1865) 
v_mul_f32 v23, v5, v13 // (op 1865) 


v_add_f32 v10, v20, v16 // (op 1861) 
v_add_f32 v11, v21, v17 // (op 1861) 
v_add_f32 v12, v22, v18 // (op 1861) 
v_add_f32 v13, v23, v19 // (op 1861) 


v_cvt_f16_f32 v14, v10 // (op 1862) 
v_cvt_f16_f32 v15, v11 // (op 1862) 
v_cvt_f16_f32 v16, v12 // (op 1862) 
v_cvt_f16_f32 v17, v13 // (op 1862) 


v_pack_b32_f16 v10, v14, v15 // (op 1863) 
v_pack_b32_f16 v11, v16, v17 // (op 1863) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:256 // (op 1863) Store value





buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:192 // (op 1844) Load value
v_accvgpr_read v12, a12 // (op 1840) 
v_accvgpr_read v13, a13 // (op 1840) 
v_accvgpr_read v14, a14 // (op 1840) 
v_accvgpr_read v15, a15 // (op 1840) 
v_mul_f32 v16, v8, v12 // (op 1840) 
v_mul_f32 v17, v8, v13 // (op 1840) 
v_mul_f32 v18, v8, v14 // (op 1840) 
v_mul_f32 v19, v8, v15 // (op 1840) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1845) 
v_lshrrev_b32 v13, 16, v10 // (op 1845) 
v_cvt_f32_f16 v13, v13 // (op 1845) 
v_mul_f32 v20, v5, v12 // (op 1845) 
v_mul_f32 v21, v5, v13 // (op 1845) 
v_cvt_f32_f16 v12, v11 // (op 1845) 
v_lshrrev_b32 v13, 16, v11 // (op 1845) 
v_cvt_f32_f16 v13, v13 // (op 1845) 
v_mul_f32 v22, v5, v12 // (op 1845) 
v_mul_f32 v23, v5, v13 // (op 1845) 


v_add_f32 v10, v20, v16 // (op 1841) 
v_add_f32 v11, v21, v17 // (op 1841) 
v_add_f32 v12, v22, v18 // (op 1841) 
v_add_f32 v13, v23, v19 // (op 1841) 


v_cvt_f16_f32 v14, v10 // (op 1842) 
v_cvt_f16_f32 v15, v11 // (op 1842) 
v_cvt_f16_f32 v16, v12 // (op 1842) 
v_cvt_f16_f32 v17, v13 // (op 1842) 


v_pack_b32_f16 v10, v14, v15 // (op 1843) 
v_pack_b32_f16 v11, v16, v17 // (op 1843) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:192 // (op 1843) Store value





buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:128 // (op 1824) Load value
v_accvgpr_read v12, a8 // (op 1820) 
v_accvgpr_read v13, a9 // (op 1820) 
v_accvgpr_read v14, a10 // (op 1820) 
v_accvgpr_read v15, a11 // (op 1820) 
v_mul_f32 v16, v8, v12 // (op 1820) 
v_mul_f32 v17, v8, v13 // (op 1820) 
v_mul_f32 v18, v8, v14 // (op 1820) 
v_mul_f32 v19, v8, v15 // (op 1820) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1825) 
v_lshrrev_b32 v13, 16, v10 // (op 1825) 
v_cvt_f32_f16 v13, v13 // (op 1825) 
v_mul_f32 v20, v5, v12 // (op 1825) 
v_mul_f32 v21, v5, v13 // (op 1825) 
v_cvt_f32_f16 v12, v11 // (op 1825) 
v_lshrrev_b32 v13, 16, v11 // (op 1825) 
v_cvt_f32_f16 v13, v13 // (op 1825) 
v_mul_f32 v22, v5, v12 // (op 1825) 
v_mul_f32 v23, v5, v13 // (op 1825) 


v_add_f32 v10, v20, v16 // (op 1821) 
v_add_f32 v11, v21, v17 // (op 1821) 
v_add_f32 v12, v22, v18 // (op 1821) 
v_add_f32 v13, v23, v19 // (op 1821) 


v_cvt_f16_f32 v14, v10 // (op 1822) 
v_cvt_f16_f32 v15, v11 // (op 1822) 
v_cvt_f16_f32 v16, v12 // (op 1822) 
v_cvt_f16_f32 v17, v13 // (op 1822) 


v_pack_b32_f16 v10, v14, v15 // (op 1823) 
v_pack_b32_f16 v11, v16, v17 // (op 1823) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:128 // (op 1823) Store value





buffer_load_dwordx2 v[10:11], v6, s[56:59], 0 offen offset:64 // (op 1804) Load value
v_accvgpr_read v12, a4 // (op 1800) 
v_accvgpr_read v13, a5 // (op 1800) 
v_accvgpr_read v14, a6 // (op 1800) 
v_accvgpr_read v15, a7 // (op 1800) 
v_mul_f32 v16, v8, v12 // (op 1800) 
v_mul_f32 v17, v8, v13 // (op 1800) 
v_mul_f32 v18, v8, v14 // (op 1800) 
v_mul_f32 v19, v8, v15 // (op 1800) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1805) 
v_lshrrev_b32 v13, 16, v10 // (op 1805) 
v_cvt_f32_f16 v13, v13 // (op 1805) 
v_mul_f32 v20, v5, v12 // (op 1805) 
v_mul_f32 v21, v5, v13 // (op 1805) 
v_cvt_f32_f16 v12, v11 // (op 1805) 
v_lshrrev_b32 v13, 16, v11 // (op 1805) 
v_cvt_f32_f16 v13, v13 // (op 1805) 
v_mul_f32 v22, v5, v12 // (op 1805) 
v_mul_f32 v23, v5, v13 // (op 1805) 


v_add_f32 v10, v20, v16 // (op 1801) 
v_add_f32 v11, v21, v17 // (op 1801) 
v_add_f32 v12, v22, v18 // (op 1801) 
v_add_f32 v13, v23, v19 // (op 1801) 


v_cvt_f16_f32 v14, v10 // (op 1802) 
v_cvt_f16_f32 v15, v11 // (op 1802) 
v_cvt_f16_f32 v16, v12 // (op 1802) 
v_cvt_f16_f32 v17, v13 // (op 1802) 


v_pack_b32_f16 v10, v14, v15 // (op 1803) 
v_pack_b32_f16 v11, v16, v17 // (op 1803) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:64 // (op 1803) Store value





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
v_cvt_f32_f16 v12, v10 // (op 139) 
v_lshrrev_b32 v13, 16, v10 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v5, v12 // (op 139) 
v_mul_f32 v21, v5, v13 // (op 139) 
v_cvt_f32_f16 v12, v11 // (op 139) 
v_lshrrev_b32 v13, 16, v11 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v22, v5, v12 // (op 139) 
v_mul_f32 v23, v5, v13 // (op 139) 


v_add_f32 v10, v20, v16 // (op 135) 
v_add_f32 v11, v21, v17 // (op 135) 
v_add_f32 v12, v22, v18 // (op 135) 
v_add_f32 v13, v23, v19 // (op 135) 


v_cvt_f16_f32 v14, v10 // (op 136) 
v_cvt_f16_f32 v15, v11 // (op 136) 
v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 


v_pack_b32_f16 v10, v14, v15 // (op 137) 
v_pack_b32_f16 v11, v16, v17 // (op 137) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
.LRR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__end:
.size RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_, .LRR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM__end-RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 144
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 75776 // lds bytes
  .amdhsa_accum_offset 112
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
    DataType: F6
    DataTypeA: F6
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
    macrotile: [256, 32, 128]
    threads: [256, 1, 1]
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
  - .name: RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NN_FP6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 75776
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 110
    .agpr_count: 32
    .max_flat_workgroup_size: 256
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

