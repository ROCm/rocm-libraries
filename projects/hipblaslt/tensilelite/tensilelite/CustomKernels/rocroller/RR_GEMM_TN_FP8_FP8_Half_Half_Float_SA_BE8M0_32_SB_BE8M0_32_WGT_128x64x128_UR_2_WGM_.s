// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__exec_begin:


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
s_load_dword s48, s[0:1], 168 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


s_mul_hi_i32 s0, s16, s10 // (op 4573) 
s_mul_hi_i32 s1, s16, s10 // (op 4828) 
v_lshrrev_b32 v0, 7, v1 // (op 4634) 
v_and_b32 v3, 15, v1 // (op 4634) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4634) 


v_and_b32 v0, 48, v1 // (op 4634) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 4634) 


v_lshrrev_b32 v0, 6, v1 // (op 4691) 
v_and_b32 v4, 1, v0 // (op 4691) 

v_and_b32 v0, 15, v1 // (op 4691) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4691) 


v_and_b32 v0, 48, v1 // (op 4691) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4691) 


v_lshrrev_b32 v0, 7, v1 // (op 4749) 
v_and_b32 v5, 15, v1 // (op 4749) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 4749) 


v_and_b32 v0, 63, v1 // (op 4749) 
v_lshrrev_b32 v5, 4, v0 // (op 4749) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 4749) 


v_lshrrev_b32 v5, 6, v1 // (op 4807) 
v_and_b32 v6, 1, v5 // (op 4807) 

v_and_b32 v5, 15, v1 // (op 4807) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4807) 


v_and_b32 v5, 63, v1 // (op 4807) 
v_lshrrev_b32 v6, 4, v5 // (op 4807) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 4807) 


v_lshrrev_b32 v6, 7, v1 // (op 4874) 
v_and_b32 v7, 15, v1 // (op 4874) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 4874) 


v_and_b32 v6, 48, v1 // (op 4874) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 4874) 


v_lshrrev_b32 v6, 6, v1 // (op 4918) 
v_and_b32 v8, 1, v6 // (op 4918) 

v_and_b32 v6, 15, v1 // (op 4918) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 4918) 


v_and_b32 v6, 48, v1 // (op 4918) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 4918) 


v_lshrrev_b32 v6, 7, v1 // (op 4947) 
v_and_b32 v9, 15, v1 // (op 4947) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 4947) 


v_and_b32 v6, 63, v1 // (op 4947) 
v_lshrrev_b32 v9, 4, v6 // (op 4947) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 4947) 


v_lshrrev_b32 v9, 6, v1 // (op 4976) 
v_and_b32 v10, 1, v9 // (op 4976) 

v_and_b32 v9, 15, v1 // (op 4976) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4976) 


v_and_b32 v9, 63, v1 // (op 4976) 
v_lshrrev_b32 v10, 4, v9 // (op 4976) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 4976) 


v_lshrrev_b32 v10, 7, v1 // (op 4573) 
s_add_i32 s18, s0, s16 // (op 4573) 

v_and_b32 v11, 63, v1 // (op 4573) 
s_ashr_i32 s0, s18, 31 // (op 4573) 
v_lshrrev_b32 v12, 4, v11 // (op 4573) 
s_cmp_eq_i32 s10, 0 // (op 4573) 
s_mov_b32 s19, scc // (op 4573) 
v_lshlrev_b32 v13, 2, v12 // (op 4573) 

s_cmp_eq_i32 s19, 0 // (op 4573) 
s_cselect_b32 s49, 0, -1 // (op 4573) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4573) 


s_and_b32 s19, s13, 31 // (op 4573) 
v_lshrrev_b32 v10, 2, v11 // (op 4573) 

v_and_b32 v11, 3, v10 // (op 4573) 

v_lshrrev_b32 v10, 7, v1 // (op 4828) 
v_and_b32 v13, 63, v1 // (op 4828) 
s_lshl_b32 s50, 1, s19 // (op 4573) 
v_and_b32 v14, 3, v1 // (op 4573) 
s_add_i32 s51, s50, s49 // (op 4573) 


s_and_b32 s49, s0, s51 // (op 4573) 


s_add_i32 s0, s1, s16 // (op 4828) 

s_ashr_i32 s1, s0, 31 // (op 4828) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 4573) 


s_add_i32 s50, s18, s49 // (op 4573) 


v_lshrrev_b32 v11, 6, v1 // (op 4573) 
s_ashr_i32 s18, s50, s19 // (op 4573) 


v_and_b32 v14, 1, v11 // (op 4573) 

s_lshl_b32 s19, s13, 24 // (op 4573) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 4573) 


s_ashr_i32 s49, s19, 31 // (op 4573) 

s_xor_b32 s19, s18, s49 // (op 4573) 

s_sub_i32 s18, s19, s49 // (op 4573) 


s_mul_hi_i32 s19, s18, s14 // (op 4573) 
s_add_i32 s49, s19, s18 // (op 4573) 

s_ashr_i32 s19, s49, 31 // (op 4573) 
s_cmp_eq_i32 s14, 0 // (op 4573) 
s_mov_b32 s50, scc // (op 4573) 
s_cmp_eq_i32 s50, 0 // (op 4573) 
s_cselect_b32 s51, 0, -1 // (op 4573) 

s_and_b32 s50, s15, 31 // (op 4573) 
v_lshrrev_b32 v14, 4, v13 // (op 4828) 
s_cmp_eq_i32 s10, 0 // (op 4828) 
s_mov_b32 s52, scc // (op 4828) 
v_lshlrev_b32 v15, 2, v14 // (op 4828) 

s_cmp_eq_i32 s52, 0 // (op 4828) 
s_cselect_b32 s53, 0, -1 // (op 4828) 

s_lshl_b32 s52, 1, s50 // (op 4573) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 4828) 


v_lshrrev_b32 v10, 2, v13 // (op 4828) 

s_add_i32 s54, s52, s51 // (op 4573) 


s_and_b32 s51, s19, s54 // (op 4573) 


s_add_i32 s19, s49, s51 // (op 4573) 


s_ashr_i32 s49, s19, s50 // (op 4573) 


s_lshl_b32 s19, s15, 24 // (op 4573) 
s_ashr_i32 s50, s19, 31 // (op 4573) 

s_xor_b32 s19, s49, s50 // (op 4573) 

s_sub_i32 s49, s19, s50 // (op 4573) 


s_mul_i32 s19, s18, s11 // (op 4573) 
s_sub_u32 s50, s16, s19 // (op 4573) 

s_mul_i32 s19, s49, s11 // (op 4573) 
s_add_u32 s19, s19, s50 // (op 4573) 

s_mul_hi_i32 s50, s16, s9 // (op 4573) 
s_add_i32 s51, s50, s16 // (op 4573) 

s_ashr_i32 s50, s51, 31 // (op 4573) 
s_cmp_eq_i32 s9, 0 // (op 4573) 
s_mov_b32 s52, scc // (op 4573) 
s_cmp_eq_i32 s52, 0 // (op 4573) 
s_cselect_b32 s54, 0, -1 // (op 4573) 

s_and_b32 s52, s13, 31 // (op 4828) 
v_and_b32 v13, 3, v10 // (op 4828) 

s_lshl_b32 s55, 1, s52 // (op 4828) 
v_and_b32 v10, 3, v1 // (op 4828) 
s_add_i32 s56, s55, s53 // (op 4828) 


s_and_b32 s53, s1, s56 // (op 4828) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 4828) 


s_add_i32 s1, s0, s53 // (op 4828) 


v_lshrrev_b32 v10, 6, v1 // (op 4828) 
s_ashr_i32 s0, s1, s52 // (op 4828) 


v_and_b32 v13, 1, v10 // (op 4828) 

s_lshl_b32 s1, s13, 24 // (op 4828) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 4828) 


s_ashr_i32 s52, s1, 31 // (op 4828) 

s_xor_b32 s1, s0, s52 // (op 4828) 

s_sub_i32 s0, s1, s52 // (op 4828) 


s_mul_hi_i32 s1, s0, s14 // (op 4828) 
s_add_i32 s52, s1, s0 // (op 4828) 

s_ashr_i32 s1, s52, 31 // (op 4828) 
s_cmp_eq_i32 s14, 0 // (op 4828) 
s_mov_b32 s53, scc // (op 4828) 
s_cmp_eq_i32 s53, 0 // (op 4828) 
s_cselect_b32 s55, 0, -1 // (op 4828) 

s_and_b32 s53, s15, 31 // (op 4828) 
s_lshl_b32 s56, 1, s53 // (op 4828) 
s_add_i32 s57, s56, s55 // (op 4828) 


s_and_b32 s55, s1, s57 // (op 4828) 


s_add_i32 s1, s52, s55 // (op 4828) 


s_ashr_i32 s52, s1, s53 // (op 4828) 


s_lshl_b32 s1, s15, 24 // (op 4828) 
s_ashr_i32 s53, s1, 31 // (op 4828) 

s_xor_b32 s1, s52, s53 // (op 4828) 

s_sub_i32 s52, s1, s53 // (op 4828) 


s_mul_i32 s1, s0, s11 // (op 4828) 
s_sub_u32 s53, s16, s1 // (op 4828) 

s_mul_i32 s1, s52, s11 // (op 4828) 
s_add_u32 s1, s1, s53 // (op 4828) 

s_mul_hi_i32 s53, s16, s9 // (op 4828) 
s_add_i32 s55, s53, s16 // (op 4828) 

s_ashr_i32 s53, s55, 31 // (op 4828) 
s_cmp_eq_i32 s9, 0 // (op 4828) 
s_mov_b32 s56, scc // (op 4828) 
s_cmp_eq_i32 s56, 0 // (op 4828) 
s_cselect_b32 s57, 0, -1 // (op 4828) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 4573) 
s_lshl_b32 s58, 1, s56 // (op 4573) 
s_add_i32 s59, s58, s54 // (op 4573) 


s_and_b32 s54, s50, s59 // (op 4573) 


s_add_i32 s50, s51, s54 // (op 4573) 


s_ashr_i32 s51, s50, s56 // (op 4573) 


s_lshl_b32 s50, s45, 24 // (op 4573) 
s_ashr_i32 s54, s50, 31 // (op 4573) 

s_xor_b32 s50, s51, s54 // (op 4573) 

s_sub_i32 s51, s50, s54 // (op 4573) 


s_and_b32 s50, s45, 31 // (op 4828) 
s_lshl_b32 s54, 1, s50 // (op 4828) 
s_add_i32 s56, s54, s57 // (op 4828) 


s_and_b32 s54, s53, s56 // (op 4828) 


s_add_i32 s53, s55, s54 // (op 4828) 


s_ashr_i32 s54, s53, s50 // (op 4828) 


s_lshl_b32 s50, s45, 24 // (op 4828) 
s_ashr_i32 s53, s50, 31 // (op 4828) 

s_xor_b32 s50, s54, s53 // (op 4828) 

s_sub_i32 s54, s50, s53 // (op 4828) 


s_add_u32 s56, s6, 64 // (op 4573) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 4573) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4573) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4573) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 4573) 

s_mul_i32 s50, s11, s56 // (op 4573) 
s_mul_i32 s53, s8, s50 // (op 4573) 

s_mul_i32 s50, s51, s53 // (op 4573) 

s_sub_u32 s53, s16, s50 // (op 4573) 

s_mov_b32 s50, s53 // (op 4573) 
s_mul_hi_i32 s55, s50, s12 // (op 4573) 
s_add_i32 s58, s55, s50 // (op 4573) 


s_ashr_i32 s50, s58, 31 // (op 4573) 
s_cmp_eq_i32 s12, 0 // (op 4573) 
s_mov_b32 s55, scc // (op 4573) 
s_cmp_eq_i32 s55, 0 // (op 4573) 
s_cselect_b32 s59, 0, -1 // (op 4573) 

s_and_b32 s55, s44, 31 // (op 4573) 
s_lshl_b32 s60, 1, s55 // (op 4573) 
s_add_i32 s61, s60, s59 // (op 4573) 


s_and_b32 s59, s50, s61 // (op 4573) 


s_add_i32 s50, s58, s59 // (op 4573) 


s_ashr_i32 s58, s50, s55 // (op 4573) 


s_lshl_b32 s50, s44, 24 // (op 4573) 
s_ashr_i32 s55, s50, 31 // (op 4573) 

s_xor_b32 s50, s58, s55 // (op 4573) 

s_sub_i32 s58, s50, s55 // (op 4573) 


s_mul_i32 s50, s46, s11 // (op 4573) 
s_add_u32 s60, s20, 128 // (op 4573) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 4573) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 4573) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 4573) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 4573) 

s_sub_i32 s55, s60, s50 // (op 4573) 

s_mul_i32 s59, s58, s55 // (op 4573) 

s_sub_u32 s55, s53, s59 // (op 4573) 


s_add_u32 s53, s55, s50 // (op 4573) 


s_cmp_eq_i32 s51, 0 // (op 4573) 
s_mov_b32 s50, scc // (op 4573) 

s_cmp_eq_i32 s50, 0 // (op 4573) 
s_cselect_b32 s51, s53, s19 // (op 4573) 


s_mul_i32 s19, s49, s56 // (op 4573) 


v_lshl_add_u32 v13, s51, 7, v12 // (op 4573) 


s_sub_i32 s49, s18, s19 // (op 4573) 


s_cmp_eq_i32 s50, 0 // (op 4573) 
s_cselect_b32 s51, s58, s49 // (op 4573) 



v_lshl_add_u32 v12, s51, 6, v11 // (op 4573) 


v_mul_lo_u32 v11, v12, s2 // (op 4573) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 4573) 


s_add_u32 s18, s6, 64 // (op 4828) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 4828) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 4828) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 4828) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 6 // (op 4828) 

s_mul_i32 s53, s11, s18 // (op 4828) 
s_mul_i32 s49, s8, s53 // (op 4828) 

s_mul_i32 s53, s54, s49 // (op 4828) 

s_sub_u32 s49, s16, s53 // (op 4828) 

s_mov_b32 s53, s49 // (op 4828) 
s_mul_hi_i32 s50, s53, s12 // (op 4828) 
s_add_i32 s51, s50, s53 // (op 4828) 


s_ashr_i32 s50, s51, 31 // (op 4828) 
s_cmp_eq_i32 s12, 0 // (op 4828) 
s_mov_b32 s53, scc // (op 4828) 
s_cmp_eq_i32 s53, 0 // (op 4828) 
s_cselect_b32 s55, 0, -1 // (op 4828) 

s_and_b32 s53, s44, 31 // (op 4828) 
s_lshl_b32 s56, 1, s53 // (op 4828) 
s_add_i32 s57, s56, s55 // (op 4828) 


s_and_b32 s55, s50, s57 // (op 4828) 


s_add_i32 s50, s51, s55 // (op 4828) 


s_ashr_i32 s51, s50, s53 // (op 4828) 


s_lshl_b32 s50, s44, 24 // (op 4828) 
s_ashr_i32 s53, s50, 31 // (op 4828) 

s_xor_b32 s50, s51, s53 // (op 4828) 

s_sub_i32 s51, s50, s53 // (op 4828) 


s_mul_i32 s50, s46, s11 // (op 4828) 
s_add_u32 s56, s20, 128 // (op 4828) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 4828) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4828) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4828) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 4828) 

s_sub_i32 s53, s56, s50 // (op 4828) 

s_mul_i32 s55, s51, s53 // (op 4828) 

s_sub_u32 s53, s49, s55 // (op 4828) 


s_add_u32 s49, s53, s50 // (op 4828) 


s_cmp_eq_i32 s54, 0 // (op 4828) 
s_mov_b32 s50, scc // (op 4828) 

s_cmp_eq_i32 s50, 0 // (op 4828) 
s_cselect_b32 s53, s49, s1 // (op 4828) 


s_mul_i32 s1, s52, s18 // (op 4828) 


v_lshl_add_u32 v11, s53, 7, v14 // (op 4828) 


s_sub_i32 s49, s0, s1 // (op 4828) 


s_cmp_eq_i32 s50, 0 // (op 4828) 
s_cselect_b32 s0, s51, s49 // (op 4828) 



v_lshl_add_u32 v13, s0, 6, v10 // (op 4828) 


v_mul_lo_u32 v10, v13, s4 // (op 4828) 

v_add_lshl_u32 v13, v11, v10, 1 // (op 4828) 


s_mul_i32 s1, s2, s7 // (op 4575) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 4575) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 4575) least significant
s_mul_hi_u32 s19, s2, s6 // (op 4575) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4575) most significant: sum
s_add_u32 s1, s1, s19 // (op 4575) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 4575) 

s_mul_i32 s1, s4, s7 // (op 4830) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 4830) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 4830) least significant
s_mul_hi_u32 s50, s4, s6 // (op 4830) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 4830) most significant: sum
s_add_u32 s1, s1, s50 // (op 4830) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 4830) 

s_mov_b32 s52, s22 // (op 4575) reg expression
s_mov_b32 s53, s23 // (op 4575) reg expression
s_mov_b32 s54, s18 // (op 4575) reg expression
s_mov_b32 s55, 131072 // (op 4575) call()

s_mov_b32 s56, s24 // (op 4830) reg expression

s_mov_b32 s57, s25 // (op 4830) reg expression
s_mov_b32 s58, s50 // (op 4830) reg expression
s_mov_b32 s59, 131072 // (op 4830) call()


v_mov_b32 v10, s47 // (op 2) Move value
v_mov_b32 v11, s48 // (op 43) Move value


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
v_accvgpr_write a16, 0.00000 // (op 946) call()
v_accvgpr_write a17, 0.00000 // (op 946) call()
v_accvgpr_write a18, 0.00000 // (op 946) call()
v_accvgpr_write a19, 0.00000 // (op 946) call()
v_accvgpr_write a20, 0.00000 // (op 1101) call()
v_accvgpr_write a21, 0.00000 // (op 1101) call()
v_accvgpr_write a22, 0.00000 // (op 1101) call()
v_accvgpr_write a23, 0.00000 // (op 1101) call()
v_accvgpr_write a24, 0.00000 // (op 1332) call()
v_accvgpr_write a25, 0.00000 // (op 1332) call()
v_accvgpr_write a26, 0.00000 // (op 1332) call()
v_accvgpr_write a27, 0.00000 // (op 1332) call()
v_accvgpr_write a28, 0.00000 // (op 1487) call()
v_accvgpr_write a29, 0.00000 // (op 1487) call()
v_accvgpr_write a30, 0.00000 // (op 1487) call()
v_accvgpr_write a31, 0.00000 // (op 1487) call()
s_mul_hi_i32 s0, s16, s10 // (op 4335) 
v_lshrrev_b32 v14, 3, v1 // (op 4335) 
s_add_i32 s1, s0, s16 // (op 4335) 

v_and_b32 v15, -8, v1 // (op 4335) 
s_ashr_i32 s0, s1, 31 // (op 4335) 
v_sub_u32 v16, v1, v15 // (op 4335) 

s_cmp_eq_i32 s10, 0 // (op 4335) 
s_mov_b32 s18, scc // (op 4335) 
v_lshlrev_b32 v15, 4, v16 // (op 4335) 

s_cmp_eq_i32 s18, 0 // (op 4335) 
s_cselect_b32 s19, 0, -1 // (op 4335) 

v_mov_b32 v17, 0 // (op 4335) convert
v_mov_b32 v16, v15 // (op 4335) convert

s_and_b32 s18, s13, 31 // (op 4335) 
s_lshl_b32 s22, 1, s18 // (op 4335) 
s_add_i32 s23, s22, s19 // (op 4335) 


s_and_b32 s19, s0, s23 // (op 4335) 


s_add_i32 s0, s1, s19 // (op 4335) 


s_ashr_i32 s1, s0, s18 // (op 4335) 


s_lshl_b32 s0, s13, 24 // (op 4335) 
s_ashr_i32 s18, s0, 31 // (op 4335) 

s_xor_b32 s0, s1, s18 // (op 4335) 

s_sub_i32 s1, s0, s18 // (op 4335) 


s_mul_hi_i32 s0, s1, s14 // (op 4335) 
s_add_i32 s18, s0, s1 // (op 4335) 

s_ashr_i32 s0, s18, 31 // (op 4335) 
s_cmp_eq_i32 s14, 0 // (op 4335) 
s_mov_b32 s19, scc // (op 4335) 
s_cmp_eq_i32 s19, 0 // (op 4335) 
s_cselect_b32 s22, 0, -1 // (op 4335) 

s_and_b32 s19, s15, 31 // (op 4335) 
s_lshl_b32 s23, 1, s19 // (op 4335) 
s_add_i32 s24, s23, s22 // (op 4335) 


s_and_b32 s25, s0, s24 // (op 4335) 


s_add_i32 s0, s18, s25 // (op 4335) 


s_ashr_i32 s18, s0, s19 // (op 4335) 


s_lshl_b32 s0, s15, 24 // (op 4335) 
s_ashr_i32 s19, s0, 31 // (op 4335) 

s_xor_b32 s0, s18, s19 // (op 4335) 

s_sub_i32 s18, s0, s19 // (op 4335) 


s_mul_i32 s0, s1, s11 // (op 4335) 

s_sub_u32 s1, s16, s0 // (op 4335) 

s_mul_i32 s0, s18, s11 // (op 4335) 
s_add_u32 s0, s0, s1 // (op 4335) 


s_mul_hi_i32 s1, s16, s9 // (op 4335) 
s_add_i32 s18, s1, s16 // (op 4335) 

s_ashr_i32 s1, s18, 31 // (op 4335) 
s_cmp_eq_i32 s9, 0 // (op 4335) 
s_mov_b32 s19, scc // (op 4335) 
s_cmp_eq_i32 s19, 0 // (op 4335) 
s_cselect_b32 s22, 0, -1 // (op 4335) 

s_and_b32 s19, s45, 31 // (op 4335) 
s_lshl_b32 s23, 1, s19 // (op 4335) 
s_add_i32 s24, s23, s22 // (op 4335) 


s_and_b32 s25, s1, s24 // (op 4335) 


s_add_i32 s1, s18, s25 // (op 4335) 


s_ashr_i32 s18, s1, s19 // (op 4335) 


s_lshl_b32 s1, s45, 24 // (op 4335) 
s_ashr_i32 s19, s1, 31 // (op 4335) 

s_xor_b32 s1, s18, s19 // (op 4335) 

s_sub_i32 s18, s1, s19 // (op 4335) 


s_add_u32 s22, s6, 64 // (op 4335) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4335) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4335) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4335) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4335) 

s_mul_i32 s1, s11, s22 // (op 4335) 

s_mul_i32 s19, s8, s1 // (op 4335) 

s_mul_i32 s1, s18, s19 // (op 4335) 

s_sub_u32 s19, s16, s1 // (op 4335) 

s_mov_b32 s1, s19 // (op 4335) 
s_mul_hi_i32 s22, s1, s12 // (op 4335) 
s_add_i32 s23, s22, s1 // (op 4335) 


s_ashr_i32 s1, s23, 31 // (op 4335) 
s_cmp_eq_i32 s12, 0 // (op 4335) 
s_mov_b32 s22, scc // (op 4335) 
s_cmp_eq_i32 s22, 0 // (op 4335) 
s_cselect_b32 s24, 0, -1 // (op 4335) 

s_and_b32 s22, s44, 31 // (op 4335) 
s_lshl_b32 s25, 1, s22 // (op 4335) 
s_add_i32 s47, s25, s24 // (op 4335) 


s_and_b32 s24, s1, s47 // (op 4335) 


s_add_i32 s1, s23, s24 // (op 4335) 


s_ashr_i32 s23, s1, s22 // (op 4335) 


s_lshl_b32 s1, s44, 24 // (op 4335) 
s_ashr_i32 s22, s1, 31 // (op 4335) 

s_xor_b32 s1, s23, s22 // (op 4335) 

s_sub_i32 s23, s1, s22 // (op 4335) 


s_mul_i32 s1, s46, s11 // (op 4335) 
s_add_u32 s24, s20, 128 // (op 4335) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4335) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 4335) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 4335) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 4335) 

s_sub_i32 s22, s24, s1 // (op 4335) 

s_mul_i32 s24, s23, s22 // (op 4335) 


s_sub_u32 s25, s19, s24 // (op 4335) 


s_add_u32 s19, s25, s1 // (op 4335) 


s_cmp_eq_i32 s18, 0 // (op 4335) 
s_mov_b32 s1, scc // (op 4335) 

s_cmp_eq_i32 s1, 0 // (op 4335) 
s_cselect_b32 s18, s19, s0 // (op 4335) 



v_lshl_add_u32 v15, s18, 7, v14 // (op 4335) 


v_mul_lo_u32 v19, v15, s27 // (op 4335) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 4335) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 4335) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 4335) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 4335) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 4335) most significant half



s_mul_i32 s1, s26, s21 // (op 4337) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 4337) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 4337) least significant
s_mul_hi_u32 s19, s26, s20 // (op 4337) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4337) most significant: sum
s_add_u32 s1, s1, s19 // (op 4337) most significant: sum


s_mov_b32 s48, s28 // (op 4337) reg expression
s_mov_b32 s49, s29 // (op 4337) reg expression
s_mov_b32 s50, s0 // (op 4337) reg expression
s_mov_b32 s51, 131072 // (op 4337) call()

s_mul_hi_i32 s47, s16, s10 // (op 4354) 
v_lshrrev_b32 v14, 3, v1 // (op 4354) 
s_add_i32 s0, s47, s16 // (op 4354) 

v_and_b32 v15, -8, v1 // (op 4354) 
s_ashr_i32 s1, s0, 31 // (op 4354) 
v_sub_u32 v16, v1, v15 // (op 4354) 

s_cmp_eq_i32 s10, 0 // (op 4354) 
s_mov_b32 s47, scc // (op 4354) 
v_lshlrev_b32 v15, 4, v16 // (op 4354) 

s_cmp_eq_i32 s47, 0 // (op 4354) 
s_cselect_b32 s18, 0, -1 // (op 4354) 

v_mov_b32 v17, 0 // (op 4354) convert
v_mov_b32 v16, v15 // (op 4354) convert

s_and_b32 s19, s13, 31 // (op 4354) 
s_lshl_b32 s47, 1, s19 // (op 4354) 
s_add_i32 s22, s47, s18 // (op 4354) 


s_and_b32 s18, s1, s22 // (op 4354) 


s_add_i32 s1, s0, s18 // (op 4354) 


s_ashr_i32 s0, s1, s19 // (op 4354) 


s_lshl_b32 s1, s13, 24 // (op 4354) 
s_ashr_i32 s47, s1, 31 // (op 4354) 

s_xor_b32 s1, s0, s47 // (op 4354) 

s_sub_i32 s0, s1, s47 // (op 4354) 


s_mul_hi_i32 s1, s0, s14 // (op 4354) 
s_add_i32 s47, s1, s0 // (op 4354) 

s_ashr_i32 s1, s47, 31 // (op 4354) 
s_cmp_eq_i32 s14, 0 // (op 4354) 
s_mov_b32 s18, scc // (op 4354) 
s_cmp_eq_i32 s18, 0 // (op 4354) 
s_cselect_b32 s19, 0, -1 // (op 4354) 

s_and_b32 s18, s15, 31 // (op 4354) 
s_lshl_b32 s22, 1, s18 // (op 4354) 
s_add_i32 s23, s22, s19 // (op 4354) 


s_and_b32 s19, s1, s23 // (op 4354) 


s_add_i32 s1, s47, s19 // (op 4354) 


s_ashr_i32 s19, s1, s18 // (op 4354) 


s_lshl_b32 s1, s15, 24 // (op 4354) 
s_ashr_i32 s18, s1, 31 // (op 4354) 

s_xor_b32 s1, s19, s18 // (op 4354) 

s_sub_i32 s19, s1, s18 // (op 4354) 


s_mul_i32 s1, s0, s11 // (op 4354) 

s_sub_u32 s0, s16, s1 // (op 4354) 

s_mul_i32 s1, s19, s11 // (op 4354) 
s_add_u32 s1, s1, s0 // (op 4354) 


s_mul_hi_i32 s0, s16, s9 // (op 4354) 
s_add_i32 s47, s0, s16 // (op 4354) 

s_ashr_i32 s0, s47, 31 // (op 4354) 
s_cmp_eq_i32 s9, 0 // (op 4354) 
s_mov_b32 s18, scc // (op 4354) 
s_cmp_eq_i32 s18, 0 // (op 4354) 
s_cselect_b32 s19, 0, -1 // (op 4354) 

s_and_b32 s18, s45, 31 // (op 4354) 
s_lshl_b32 s22, 1, s18 // (op 4354) 
s_add_i32 s23, s22, s19 // (op 4354) 


s_and_b32 s19, s0, s23 // (op 4354) 


s_add_i32 s0, s47, s19 // (op 4354) 


s_ashr_i32 s19, s0, s18 // (op 4354) 


s_lshl_b32 s0, s45, 24 // (op 4354) 
s_ashr_i32 s18, s0, 31 // (op 4354) 

s_xor_b32 s0, s19, s18 // (op 4354) 

s_sub_i32 s19, s0, s18 // (op 4354) 


s_add_u32 s22, s6, 64 // (op 4354) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4354) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4354) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4354) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4354) 

s_mul_i32 s0, s11, s22 // (op 4354) 

s_mul_i32 s18, s8, s0 // (op 4354) 

s_mul_i32 s0, s19, s18 // (op 4354) 

s_sub_u32 s18, s16, s0 // (op 4354) 

s_mov_b32 s0, s18 // (op 4354) 
s_mul_hi_i32 s47, s0, s12 // (op 4354) 
s_add_i32 s22, s47, s0 // (op 4354) 


s_ashr_i32 s0, s22, 31 // (op 4354) 
s_cmp_eq_i32 s12, 0 // (op 4354) 
s_mov_b32 s47, scc // (op 4354) 
s_cmp_eq_i32 s47, 0 // (op 4354) 
s_cselect_b32 s23, 0, -1 // (op 4354) 

s_and_b32 s47, s44, 31 // (op 4354) 
s_lshl_b32 s24, 1, s47 // (op 4354) 
s_add_i32 s25, s24, s23 // (op 4354) 


s_and_b32 s23, s0, s25 // (op 4354) 


s_add_i32 s0, s22, s23 // (op 4354) 


s_ashr_i32 s22, s0, s47 // (op 4354) 


s_lshl_b32 s0, s44, 24 // (op 4354) 
s_ashr_i32 s47, s0, 31 // (op 4354) 

s_xor_b32 s0, s22, s47 // (op 4354) 

s_sub_i32 s22, s0, s47 // (op 4354) 


s_mul_i32 s0, s46, s11 // (op 4354) 
s_add_u32 s24, s20, 128 // (op 4354) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4354) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 4354) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 4354) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 4354) 

s_sub_i32 s23, s24, s0 // (op 4354) 

s_mul_i32 s47, s22, s23 // (op 4354) 


s_sub_u32 s22, s18, s47 // (op 4354) 


s_add_u32 s18, s22, s0 // (op 4354) 


s_cmp_eq_i32 s19, 0 // (op 4354) 
s_mov_b32 s0, scc // (op 4354) 

s_cmp_eq_i32 s0, 0 // (op 4354) 
s_cselect_b32 s19, s18, s1 // (op 4354) 



v_lshl_add_u32 v15, s19, 7, v14 // (op 4354) 


v_mul_lo_u32 v21, v15, s27 // (op 4354) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 4354) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 4354) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 4354) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 4354) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 4354) most significant half



s_mul_i32 s1, s26, s21 // (op 4356) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 4356) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 4356) least significant
s_mul_hi_u32 s18, s26, s20 // (op 4356) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4356) most significant: sum
s_add_u32 s1, s1, s18 // (op 4356) most significant: sum


s_mov_b32 s48, s28 // (op 4356) reg expression
s_mov_b32 s49, s29 // (op 4356) reg expression
s_mov_b32 s50, s0 // (op 4356) reg expression
s_mov_b32 s51, 131072 // (op 4356) call()

s_mul_hi_i32 s47, s16, s10 // (op 4399) 
v_lshrrev_b32 v14, 3, v1 // (op 4399) 
s_add_i32 s0, s47, s16 // (op 4399) 

v_and_b32 v15, -8, v1 // (op 4399) 
s_ashr_i32 s1, s0, 31 // (op 4399) 
v_sub_u32 v16, v1, v15 // (op 4399) 

s_cmp_eq_i32 s10, 0 // (op 4399) 
s_mov_b32 s47, scc // (op 4399) 
v_lshlrev_b32 v15, 4, v16 // (op 4399) 

s_cmp_eq_i32 s47, 0 // (op 4399) 
s_cselect_b32 s18, 0, -1 // (op 4399) 

v_mov_b32 v17, 0 // (op 4399) convert
v_mov_b32 v16, v15 // (op 4399) convert

s_and_b32 s19, s13, 31 // (op 4399) 
s_lshl_b32 s47, 1, s19 // (op 4399) 
s_add_i32 s22, s47, s18 // (op 4399) 


s_and_b32 s18, s1, s22 // (op 4399) 


s_add_i32 s1, s0, s18 // (op 4399) 


s_ashr_i32 s0, s1, s19 // (op 4399) 


s_lshl_b32 s1, s13, 24 // (op 4399) 
s_ashr_i32 s47, s1, 31 // (op 4399) 

s_xor_b32 s1, s0, s47 // (op 4399) 

s_sub_i32 s0, s1, s47 // (op 4399) 


s_mul_hi_i32 s1, s0, s14 // (op 4399) 
s_add_i32 s47, s1, s0 // (op 4399) 

s_ashr_i32 s1, s47, 31 // (op 4399) 
s_cmp_eq_i32 s14, 0 // (op 4399) 
s_mov_b32 s18, scc // (op 4399) 
s_cmp_eq_i32 s18, 0 // (op 4399) 
s_cselect_b32 s19, 0, -1 // (op 4399) 

s_and_b32 s18, s15, 31 // (op 4399) 
s_lshl_b32 s22, 1, s18 // (op 4399) 
s_add_i32 s23, s22, s19 // (op 4399) 


s_and_b32 s19, s1, s23 // (op 4399) 


s_add_i32 s1, s47, s19 // (op 4399) 


s_ashr_i32 s19, s1, s18 // (op 4399) 


s_lshl_b32 s1, s15, 24 // (op 4399) 
s_ashr_i32 s18, s1, 31 // (op 4399) 

s_xor_b32 s1, s19, s18 // (op 4399) 

s_sub_i32 s19, s1, s18 // (op 4399) 


s_add_u32 s22, s6, 64 // (op 4399) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4399) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4399) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4399) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4399) 

s_mul_i32 s1, s19, s22 // (op 4399) 

s_sub_i32 s47, s0, s1 // (op 4399) 


s_mul_hi_i32 s0, s16, s9 // (op 4399) 
s_add_i32 s1, s0, s16 // (op 4399) 

s_ashr_i32 s0, s1, 31 // (op 4399) 
s_cmp_eq_i32 s9, 0 // (op 4399) 
s_mov_b32 s18, scc // (op 4399) 
s_cmp_eq_i32 s18, 0 // (op 4399) 
s_cselect_b32 s19, 0, -1 // (op 4399) 

s_and_b32 s18, s45, 31 // (op 4399) 
s_lshl_b32 s24, 1, s18 // (op 4399) 
s_add_i32 s25, s24, s19 // (op 4399) 


s_and_b32 s19, s0, s25 // (op 4399) 


s_add_i32 s0, s1, s19 // (op 4399) 


s_ashr_i32 s1, s0, s18 // (op 4399) 


s_lshl_b32 s0, s45, 24 // (op 4399) 
s_ashr_i32 s18, s0, 31 // (op 4399) 

s_xor_b32 s0, s1, s18 // (op 4399) 

s_sub_i32 s1, s0, s18 // (op 4399) 


s_mul_i32 s0, s11, s22 // (op 4399) 

s_mul_i32 s18, s8, s0 // (op 4399) 

s_mul_i32 s0, s1, s18 // (op 4399) 

s_sub_u32 s18, s16, s0 // (op 4399) 

s_mul_hi_i32 s0, s18, s12 // (op 4399) 
s_add_i32 s19, s0, s18 // (op 4399) 


s_ashr_i32 s0, s19, 31 // (op 4399) 
s_cmp_eq_i32 s12, 0 // (op 4399) 
s_mov_b32 s18, scc // (op 4399) 
s_cmp_eq_i32 s18, 0 // (op 4399) 
s_cselect_b32 s22, 0, -1 // (op 4399) 

s_and_b32 s18, s44, 31 // (op 4399) 
s_lshl_b32 s23, 1, s18 // (op 4399) 
s_add_i32 s24, s23, s22 // (op 4399) 


s_and_b32 s25, s0, s24 // (op 4399) 


s_add_i32 s0, s19, s25 // (op 4399) 


s_ashr_i32 s19, s0, s18 // (op 4399) 


s_lshl_b32 s0, s44, 24 // (op 4399) 
s_ashr_i32 s18, s0, 31 // (op 4399) 

s_xor_b32 s0, s19, s18 // (op 4399) 

s_sub_i32 s19, s0, s18 // (op 4399) 


s_cmp_eq_i32 s1, 0 // (op 4399) 
s_mov_b32 s0, scc // (op 4399) 

s_cmp_eq_i32 s0, 0 // (op 4399) 
s_cselect_b32 s1, s19, s47 // (op 4399) 



v_lshl_add_u32 v15, s1, 6, v14 // (op 4399) 


v_mul_lo_u32 v23, v15, s31 // (op 4399) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 4399) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 4399) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 4399) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 4399) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 4399) most significant half



s_mul_i32 s1, s30, s7 // (op 4401) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 4401) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 4401) least significant
s_mul_hi_u32 s18, s30, s6 // (op 4401) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4401) most significant: sum
s_add_u32 s1, s1, s18 // (op 4401) most significant: sum


s_mov_b32 s60, s32 // (op 4401) reg expression
s_mov_b32 s61, s33 // (op 4401) reg expression
s_mov_b32 s62, s0 // (op 4401) reg expression
s_mov_b32 s63, 131072 // (op 4401) call()

s_mul_hi_i32 s47, s16, s10 // (op 4418) 
v_lshrrev_b32 v16, 3, v1 // (op 4418) 
s_add_i32 s0, s47, s16 // (op 4418) 

v_and_b32 v17, -8, v1 // (op 4418) 
s_ashr_i32 s1, s0, 31 // (op 4418) 
v_sub_u32 v22, v1, v17 // (op 4418) 

s_cmp_eq_i32 s10, 0 // (op 4418) 
s_mov_b32 s47, scc // (op 4418) 
v_lshlrev_b32 v17, 4, v22 // (op 4418) 

s_cmp_eq_i32 s47, 0 // (op 4418) 
s_cselect_b32 s18, 0, -1 // (op 4418) 

v_mov_b32 v23, 0 // (op 4418) convert
v_mov_b32 v22, v17 // (op 4418) convert

s_and_b32 s19, s13, 31 // (op 4418) 
s_lshl_b32 s47, 1, s19 // (op 4418) 
s_add_i32 s22, s47, s18 // (op 4418) 


s_and_b32 s18, s1, s22 // (op 4418) 


s_add_i32 s1, s0, s18 // (op 4418) 


s_ashr_i32 s0, s1, s19 // (op 4418) 


s_lshl_b32 s1, s13, 24 // (op 4418) 
s_ashr_i32 s47, s1, 31 // (op 4418) 

s_xor_b32 s1, s0, s47 // (op 4418) 

s_sub_i32 s0, s1, s47 // (op 4418) 


s_mul_hi_i32 s1, s0, s14 // (op 4418) 
s_add_i32 s47, s1, s0 // (op 4418) 

s_ashr_i32 s1, s47, 31 // (op 4418) 
s_cmp_eq_i32 s14, 0 // (op 4418) 
s_mov_b32 s18, scc // (op 4418) 
s_cmp_eq_i32 s18, 0 // (op 4418) 
s_cselect_b32 s19, 0, -1 // (op 4418) 

s_and_b32 s18, s15, 31 // (op 4418) 
s_lshl_b32 s22, 1, s18 // (op 4418) 
s_add_i32 s23, s22, s19 // (op 4418) 


s_and_b32 s19, s1, s23 // (op 4418) 


s_add_i32 s1, s47, s19 // (op 4418) 


s_ashr_i32 s19, s1, s18 // (op 4418) 


s_lshl_b32 s1, s15, 24 // (op 4418) 
s_ashr_i32 s18, s1, 31 // (op 4418) 

s_xor_b32 s1, s19, s18 // (op 4418) 

s_sub_i32 s19, s1, s18 // (op 4418) 


s_add_u32 s22, s6, 64 // (op 4418) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4418) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4418) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4418) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4418) 

s_mul_i32 s1, s19, s22 // (op 4418) 

s_sub_i32 s47, s0, s1 // (op 4418) 


s_mul_hi_i32 s0, s16, s9 // (op 4418) 
s_add_i32 s1, s0, s16 // (op 4418) 

s_ashr_i32 s0, s1, 31 // (op 4418) 
s_cmp_eq_i32 s9, 0 // (op 4418) 
s_mov_b32 s18, scc // (op 4418) 
s_cmp_eq_i32 s18, 0 // (op 4418) 
s_cselect_b32 s19, 0, -1 // (op 4418) 

s_and_b32 s18, s45, 31 // (op 4418) 
s_lshl_b32 s24, 1, s18 // (op 4418) 
s_add_i32 s25, s24, s19 // (op 4418) 


s_and_b32 s19, s0, s25 // (op 4418) 


s_add_i32 s0, s1, s19 // (op 4418) 


s_ashr_i32 s1, s0, s18 // (op 4418) 


s_lshl_b32 s0, s45, 24 // (op 4418) 
s_ashr_i32 s18, s0, 31 // (op 4418) 

s_xor_b32 s0, s1, s18 // (op 4418) 

s_sub_i32 s1, s0, s18 // (op 4418) 


s_mul_i32 s0, s11, s22 // (op 4418) 

s_mul_i32 s18, s8, s0 // (op 4418) 

s_mul_i32 s0, s1, s18 // (op 4418) 

s_sub_u32 s18, s16, s0 // (op 4418) 

s_mul_hi_i32 s0, s18, s12 // (op 4418) 
s_add_i32 s19, s0, s18 // (op 4418) 


s_ashr_i32 s0, s19, 31 // (op 4418) 
s_cmp_eq_i32 s12, 0 // (op 4418) 
s_mov_b32 s18, scc // (op 4418) 
s_cmp_eq_i32 s18, 0 // (op 4418) 
s_cselect_b32 s22, 0, -1 // (op 4418) 

s_and_b32 s18, s44, 31 // (op 4418) 
s_lshl_b32 s23, 1, s18 // (op 4418) 
s_add_i32 s24, s23, s22 // (op 4418) 


s_and_b32 s25, s0, s24 // (op 4418) 


s_add_i32 s0, s19, s25 // (op 4418) 


s_ashr_i32 s19, s0, s18 // (op 4418) 


s_lshl_b32 s0, s44, 24 // (op 4418) 
s_ashr_i32 s18, s0, 31 // (op 4418) 

s_xor_b32 s0, s19, s18 // (op 4418) 

s_sub_i32 s19, s0, s18 // (op 4418) 


s_cmp_eq_i32 s1, 0 // (op 4418) 
s_mov_b32 s0, scc // (op 4418) 

s_cmp_eq_i32 s0, 0 // (op 4418) 
s_cselect_b32 s1, s19, s47 // (op 4418) 



v_lshl_add_u32 v17, s1, 6, v16 // (op 4418) 


v_mul_lo_u32 v25, v17, s31 // (op 4418) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 4418) most significant: high of low * low
v_mul_lo_u32 v24, v17, s30 // (op 4418) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 4418) most significant: sum


v_add_co_u32 v16, s[0:1], v22, v24 // (op 4418) least significant half
v_addc_co_u32 v17, s[0:1], v23, v25, s[0:1] // (op 4418) most significant half



s_mul_i32 s1, s30, s7 // (op 4420) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 4420) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 4420) least significant
s_mul_hi_u32 s18, s30, s6 // (op 4420) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4420) most significant: sum
s_add_u32 s1, s1, s18 // (op 4420) most significant: sum


s_mov_b32 s60, s32 // (op 4420) reg expression
s_mov_b32 s61, s33 // (op 4420) reg expression
s_mov_b32 s62, s0 // (op 4420) reg expression
s_mov_b32 s63, 131072 // (op 4420) call()

s_mul_hi_i32 s47, s16, s10 // (op 4463) 
v_lshrrev_b32 v22, 2, v1 // (op 4463) 
s_add_i32 s0, s47, s16 // (op 4463) 

v_lshlrev_b32 v23, 1, v22 // (op 4463) 

s_ashr_i32 s1, s0, 31 // (op 4463) 
v_and_b32 v22, -4, v1 // (op 4463) 
s_cmp_eq_i32 s10, 0 // (op 4463) 
s_mov_b32 s47, scc // (op 4463) 
v_sub_u32 v24, v1, v22 // (op 4463) 

s_cmp_eq_i32 s47, 0 // (op 4463) 
s_cselect_b32 s18, 0, -1 // (op 4463) 

s_and_b32 s19, s13, 31 // (op 4463) 
s_lshl_b32 s47, 1, s19 // (op 4463) 
s_add_i32 s22, s47, s18 // (op 4463) 


s_and_b32 s18, s1, s22 // (op 4463) 


s_add_i32 s1, s0, s18 // (op 4463) 


s_ashr_i32 s0, s1, s19 // (op 4463) 


s_lshl_b32 s1, s13, 24 // (op 4463) 
s_ashr_i32 s47, s1, 31 // (op 4463) 

s_xor_b32 s1, s0, s47 // (op 4463) 

s_sub_i32 s0, s1, s47 // (op 4463) 


s_mul_hi_i32 s1, s0, s14 // (op 4463) 
s_add_i32 s47, s1, s0 // (op 4463) 

s_ashr_i32 s1, s47, 31 // (op 4463) 
s_cmp_eq_i32 s14, 0 // (op 4463) 
s_mov_b32 s18, scc // (op 4463) 
s_cmp_eq_i32 s18, 0 // (op 4463) 
s_cselect_b32 s19, 0, -1 // (op 4463) 

s_and_b32 s18, s15, 31 // (op 4463) 
s_lshl_b32 s22, 1, s18 // (op 4463) 
s_add_i32 s23, s22, s19 // (op 4463) 


s_and_b32 s19, s1, s23 // (op 4463) 


s_add_i32 s1, s47, s19 // (op 4463) 


s_ashr_i32 s19, s1, s18 // (op 4463) 


s_lshl_b32 s1, s15, 24 // (op 4463) 
s_ashr_i32 s18, s1, 31 // (op 4463) 

s_xor_b32 s1, s19, s18 // (op 4463) 

s_sub_i32 s19, s1, s18 // (op 4463) 


s_mul_i32 s1, s0, s11 // (op 4463) 

s_sub_u32 s0, s16, s1 // (op 4463) 

s_mul_i32 s1, s19, s11 // (op 4463) 
s_add_u32 s1, s1, s0 // (op 4463) 


s_mul_hi_i32 s0, s16, s9 // (op 4463) 
s_add_i32 s47, s0, s16 // (op 4463) 

s_ashr_i32 s0, s47, 31 // (op 4463) 
s_cmp_eq_i32 s9, 0 // (op 4463) 
s_mov_b32 s18, scc // (op 4463) 
s_cmp_eq_i32 s18, 0 // (op 4463) 
s_cselect_b32 s19, 0, -1 // (op 4463) 

s_and_b32 s18, s45, 31 // (op 4463) 
s_lshl_b32 s22, 1, s18 // (op 4463) 
s_add_i32 s23, s22, s19 // (op 4463) 


s_and_b32 s19, s0, s23 // (op 4463) 


s_add_i32 s0, s47, s19 // (op 4463) 


s_ashr_i32 s19, s0, s18 // (op 4463) 


s_lshl_b32 s0, s45, 24 // (op 4463) 
s_ashr_i32 s18, s0, 31 // (op 4463) 

s_xor_b32 s0, s19, s18 // (op 4463) 

s_sub_i32 s19, s0, s18 // (op 4463) 


s_add_u32 s22, s6, 64 // (op 4463) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4463) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4463) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4463) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4463) 

s_mul_i32 s0, s11, s22 // (op 4463) 

s_mul_i32 s18, s8, s0 // (op 4463) 

s_mul_i32 s0, s19, s18 // (op 4463) 

s_sub_u32 s18, s16, s0 // (op 4463) 

s_mov_b32 s0, s18 // (op 4463) 
s_mul_hi_i32 s47, s0, s12 // (op 4463) 
s_add_i32 s22, s47, s0 // (op 4463) 


s_ashr_i32 s0, s22, 31 // (op 4463) 
s_cmp_eq_i32 s12, 0 // (op 4463) 
s_mov_b32 s47, scc // (op 4463) 
s_cmp_eq_i32 s47, 0 // (op 4463) 
s_cselect_b32 s23, 0, -1 // (op 4463) 

s_and_b32 s47, s44, 31 // (op 4463) 
s_lshl_b32 s24, 1, s47 // (op 4463) 
s_add_i32 s25, s24, s23 // (op 4463) 


s_and_b32 s23, s0, s25 // (op 4463) 


s_add_i32 s0, s22, s23 // (op 4463) 


s_ashr_i32 s22, s0, s47 // (op 4463) 


s_lshl_b32 s0, s44, 24 // (op 4463) 
s_ashr_i32 s47, s0, 31 // (op 4463) 

s_xor_b32 s0, s22, s47 // (op 4463) 

s_sub_i32 s22, s0, s47 // (op 4463) 


s_mul_i32 s0, s46, s11 // (op 4463) 
s_add_u32 s24, s20, 128 // (op 4463) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4463) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 4463) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 4463) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 4463) 

s_sub_i32 s23, s24, s0 // (op 4463) 

s_mul_i32 s47, s22, s23 // (op 4463) 


s_sub_u32 s22, s18, s47 // (op 4463) 


s_add_u32 s18, s22, s0 // (op 4463) 


s_cmp_eq_i32 s19, 0 // (op 4463) 
s_mov_b32 s0, scc // (op 4463) 

s_cmp_eq_i32 s0, 0 // (op 4463) 
s_cselect_b32 s19, s18, s1 // (op 4463) 



v_lshl_add_u32 v22, s19, 7, v23 // (op 4463) 


v_mul_lo_u32 v23, v22, s34 // (op 4463) 
v_add_u32 v23, v23, v24 // (op 4463) 


s_mul_i32 s1, s34, s21 // (op 4465) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 4465) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 4465) least significant
s_mul_hi_u32 s18, s34, s20 // (op 4465) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4465) most significant: sum
s_add_u32 s1, s1, s18 // (op 4465) most significant: sum


s_mov_b32 s64, s36 // (op 4465) reg expression
s_mov_b32 s65, s37 // (op 4465) reg expression
s_mov_b32 s66, s0 // (op 4465) reg expression
s_mov_b32 s67, 131072 // (op 4465) call()

s_mul_hi_i32 s47, s16, s10 // (op 4482) 
v_lshrrev_b32 v22, 2, v1 // (op 4482) 
s_add_i32 s0, s47, s16 // (op 4482) 

v_lshlrev_b32 v24, 1, v22 // (op 4482) 

s_ashr_i32 s1, s0, 31 // (op 4482) 
v_and_b32 v22, -4, v1 // (op 4482) 
s_cmp_eq_i32 s10, 0 // (op 4482) 
s_mov_b32 s47, scc // (op 4482) 
v_sub_u32 v25, v1, v22 // (op 4482) 

s_cmp_eq_i32 s47, 0 // (op 4482) 
s_cselect_b32 s18, 0, -1 // (op 4482) 

s_and_b32 s19, s13, 31 // (op 4482) 
s_lshl_b32 s47, 1, s19 // (op 4482) 
s_add_i32 s22, s47, s18 // (op 4482) 


s_and_b32 s18, s1, s22 // (op 4482) 


s_add_i32 s1, s0, s18 // (op 4482) 


s_ashr_i32 s0, s1, s19 // (op 4482) 


s_lshl_b32 s1, s13, 24 // (op 4482) 
s_ashr_i32 s47, s1, 31 // (op 4482) 

s_xor_b32 s1, s0, s47 // (op 4482) 

s_sub_i32 s0, s1, s47 // (op 4482) 


s_mul_hi_i32 s1, s0, s14 // (op 4482) 
s_add_i32 s47, s1, s0 // (op 4482) 

s_ashr_i32 s1, s47, 31 // (op 4482) 
s_cmp_eq_i32 s14, 0 // (op 4482) 
s_mov_b32 s18, scc // (op 4482) 
s_cmp_eq_i32 s18, 0 // (op 4482) 
s_cselect_b32 s19, 0, -1 // (op 4482) 

s_and_b32 s18, s15, 31 // (op 4482) 
s_lshl_b32 s22, 1, s18 // (op 4482) 
s_add_i32 s23, s22, s19 // (op 4482) 


s_and_b32 s19, s1, s23 // (op 4482) 


s_add_i32 s1, s47, s19 // (op 4482) 


s_ashr_i32 s19, s1, s18 // (op 4482) 


s_lshl_b32 s1, s15, 24 // (op 4482) 
s_ashr_i32 s18, s1, 31 // (op 4482) 

s_xor_b32 s1, s19, s18 // (op 4482) 

s_sub_i32 s19, s1, s18 // (op 4482) 


s_mul_i32 s1, s0, s11 // (op 4482) 

s_sub_u32 s0, s16, s1 // (op 4482) 

s_mul_i32 s1, s19, s11 // (op 4482) 
s_add_u32 s1, s1, s0 // (op 4482) 


s_mul_hi_i32 s0, s16, s9 // (op 4482) 
s_add_i32 s47, s0, s16 // (op 4482) 

s_ashr_i32 s0, s47, 31 // (op 4482) 
s_cmp_eq_i32 s9, 0 // (op 4482) 
s_mov_b32 s18, scc // (op 4482) 
s_cmp_eq_i32 s18, 0 // (op 4482) 
s_cselect_b32 s19, 0, -1 // (op 4482) 

s_and_b32 s18, s45, 31 // (op 4482) 
s_lshl_b32 s22, 1, s18 // (op 4482) 
s_add_i32 s23, s22, s19 // (op 4482) 


s_and_b32 s19, s0, s23 // (op 4482) 


s_add_i32 s0, s47, s19 // (op 4482) 


s_ashr_i32 s19, s0, s18 // (op 4482) 


s_lshl_b32 s0, s45, 24 // (op 4482) 
s_ashr_i32 s18, s0, 31 // (op 4482) 

s_xor_b32 s0, s19, s18 // (op 4482) 

s_sub_i32 s19, s0, s18 // (op 4482) 


s_add_u32 s22, s6, 64 // (op 4482) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4482) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4482) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4482) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4482) 

s_mul_i32 s0, s11, s22 // (op 4482) 

s_mul_i32 s18, s8, s0 // (op 4482) 

s_mul_i32 s0, s19, s18 // (op 4482) 

s_sub_u32 s18, s16, s0 // (op 4482) 

s_mov_b32 s0, s18 // (op 4482) 
s_mul_hi_i32 s47, s0, s12 // (op 4482) 
s_add_i32 s22, s47, s0 // (op 4482) 


s_ashr_i32 s0, s22, 31 // (op 4482) 
s_cmp_eq_i32 s12, 0 // (op 4482) 
s_mov_b32 s47, scc // (op 4482) 
s_cmp_eq_i32 s47, 0 // (op 4482) 
s_cselect_b32 s23, 0, -1 // (op 4482) 

s_and_b32 s47, s44, 31 // (op 4482) 
s_lshl_b32 s24, 1, s47 // (op 4482) 
s_add_i32 s25, s24, s23 // (op 4482) 


s_and_b32 s23, s0, s25 // (op 4482) 


s_add_i32 s0, s22, s23 // (op 4482) 


s_ashr_i32 s22, s0, s47 // (op 4482) 


s_lshl_b32 s0, s44, 24 // (op 4482) 
s_ashr_i32 s47, s0, 31 // (op 4482) 

s_xor_b32 s0, s22, s47 // (op 4482) 

s_sub_i32 s22, s0, s47 // (op 4482) 


s_mul_i32 s0, s46, s11 // (op 4482) 
s_add_u32 s24, s20, 128 // (op 4482) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4482) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 4482) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 4482) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 4482) 

s_sub_i32 s23, s24, s0 // (op 4482) 

s_mul_i32 s47, s22, s23 // (op 4482) 


s_sub_u32 s22, s18, s47 // (op 4482) 


s_add_u32 s18, s22, s0 // (op 4482) 


s_cmp_eq_i32 s19, 0 // (op 4482) 
s_mov_b32 s0, scc // (op 4482) 

s_cmp_eq_i32 s0, 0 // (op 4482) 
s_cselect_b32 s19, s18, s1 // (op 4482) 



v_lshl_add_u32 v22, s19, 7, v24 // (op 4482) 


v_mul_lo_u32 v24, v22, s34 // (op 4482) 
v_add_u32 v24, v24, v25 // (op 4482) 


s_mul_i32 s1, s34, s21 // (op 4484) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 4484) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 4484) least significant
s_mul_hi_u32 s18, s34, s20 // (op 4484) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4484) most significant: sum
s_add_u32 s1, s1, s18 // (op 4484) most significant: sum


s_mov_b32 s64, s36 // (op 4484) reg expression
s_mov_b32 s65, s37 // (op 4484) reg expression
s_mov_b32 s66, s0 // (op 4484) reg expression
s_mov_b32 s67, 131072 // (op 4484) call()

s_mul_hi_i32 s47, s16, s10 // (op 4527) 
v_lshrrev_b32 v22, 2, v1 // (op 4527) 
s_add_i32 s0, s47, s16 // (op 4527) 

v_and_b32 v25, -4, v1 // (op 4527) 
s_ashr_i32 s1, s0, 31 // (op 4527) 
v_sub_u32 v26, v1, v25 // (op 4527) 

s_cmp_eq_i32 s10, 0 // (op 4527) 
s_mov_b32 s47, scc // (op 4527) 
s_cmp_eq_i32 s47, 0 // (op 4527) 
s_cselect_b32 s18, 0, -1 // (op 4527) 

s_and_b32 s19, s13, 31 // (op 4527) 
s_lshl_b32 s47, 1, s19 // (op 4527) 
s_add_i32 s22, s47, s18 // (op 4527) 


s_and_b32 s18, s1, s22 // (op 4527) 


s_add_i32 s1, s0, s18 // (op 4527) 


s_ashr_i32 s0, s1, s19 // (op 4527) 


s_lshl_b32 s1, s13, 24 // (op 4527) 
s_ashr_i32 s47, s1, 31 // (op 4527) 

s_xor_b32 s1, s0, s47 // (op 4527) 

s_sub_i32 s0, s1, s47 // (op 4527) 


s_mul_hi_i32 s1, s0, s14 // (op 4527) 
s_add_i32 s47, s1, s0 // (op 4527) 

s_ashr_i32 s1, s47, 31 // (op 4527) 
s_cmp_eq_i32 s14, 0 // (op 4527) 
s_mov_b32 s18, scc // (op 4527) 
s_cmp_eq_i32 s18, 0 // (op 4527) 
s_cselect_b32 s19, 0, -1 // (op 4527) 

s_and_b32 s18, s15, 31 // (op 4527) 
s_lshl_b32 s22, 1, s18 // (op 4527) 
s_add_i32 s23, s22, s19 // (op 4527) 


s_and_b32 s19, s1, s23 // (op 4527) 


s_add_i32 s1, s47, s19 // (op 4527) 


s_ashr_i32 s19, s1, s18 // (op 4527) 


s_lshl_b32 s1, s15, 24 // (op 4527) 
s_ashr_i32 s18, s1, 31 // (op 4527) 

s_xor_b32 s1, s19, s18 // (op 4527) 

s_sub_i32 s19, s1, s18 // (op 4527) 


s_add_u32 s22, s6, 64 // (op 4527) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4527) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4527) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4527) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4527) 

s_mul_i32 s1, s19, s22 // (op 4527) 

s_sub_i32 s47, s0, s1 // (op 4527) 


s_mul_hi_i32 s0, s16, s9 // (op 4527) 
s_add_i32 s1, s0, s16 // (op 4527) 

s_ashr_i32 s0, s1, 31 // (op 4527) 
s_cmp_eq_i32 s9, 0 // (op 4527) 
s_mov_b32 s18, scc // (op 4527) 
s_cmp_eq_i32 s18, 0 // (op 4527) 
s_cselect_b32 s19, 0, -1 // (op 4527) 

s_and_b32 s18, s45, 31 // (op 4527) 
s_lshl_b32 s24, 1, s18 // (op 4527) 
s_add_i32 s25, s24, s19 // (op 4527) 


s_and_b32 s19, s0, s25 // (op 4527) 


s_add_i32 s0, s1, s19 // (op 4527) 


s_ashr_i32 s1, s0, s18 // (op 4527) 


s_lshl_b32 s0, s45, 24 // (op 4527) 
s_ashr_i32 s18, s0, 31 // (op 4527) 

s_xor_b32 s0, s1, s18 // (op 4527) 

s_sub_i32 s1, s0, s18 // (op 4527) 


s_mul_i32 s0, s11, s22 // (op 4527) 

s_mul_i32 s18, s8, s0 // (op 4527) 

s_mul_i32 s0, s1, s18 // (op 4527) 

s_sub_u32 s18, s16, s0 // (op 4527) 

s_mul_hi_i32 s0, s18, s12 // (op 4527) 
s_add_i32 s19, s0, s18 // (op 4527) 


s_ashr_i32 s0, s19, 31 // (op 4527) 
s_cmp_eq_i32 s12, 0 // (op 4527) 
s_mov_b32 s18, scc // (op 4527) 
s_cmp_eq_i32 s18, 0 // (op 4527) 
s_cselect_b32 s22, 0, -1 // (op 4527) 

s_and_b32 s18, s44, 31 // (op 4527) 
s_lshl_b32 s23, 1, s18 // (op 4527) 
s_add_i32 s24, s23, s22 // (op 4527) 


s_and_b32 s25, s0, s24 // (op 4527) 


s_add_i32 s0, s19, s25 // (op 4527) 


s_ashr_i32 s19, s0, s18 // (op 4527) 


s_lshl_b32 s0, s44, 24 // (op 4527) 
s_ashr_i32 s18, s0, 31 // (op 4527) 

s_xor_b32 s0, s19, s18 // (op 4527) 

s_sub_i32 s19, s0, s18 // (op 4527) 


s_cmp_eq_i32 s1, 0 // (op 4527) 
s_mov_b32 s0, scc // (op 4527) 

s_cmp_eq_i32 s0, 0 // (op 4527) 
s_cselect_b32 s1, s19, s47 // (op 4527) 



v_lshl_add_u32 v25, s1, 6, v22 // (op 4527) 


v_mul_lo_u32 v22, v25, s38 // (op 4527) 

v_add_u32 v25, v26, v22 // (op 4527) 


s_mul_i32 s1, s38, s7 // (op 4529) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 4529) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 4529) least significant
s_mul_hi_u32 s18, s38, s6 // (op 4529) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4529) most significant: sum
s_add_u32 s1, s1, s18 // (op 4529) most significant: sum


s_mov_b32 s68, s40 // (op 4529) reg expression
s_mov_b32 s69, s41 // (op 4529) reg expression
s_mov_b32 s70, s0 // (op 4529) reg expression
s_mov_b32 s71, 131072 // (op 4529) call()

s_mul_hi_i32 s47, s16, s10 // (op 4546) 
v_lshrrev_b32 v22, 2, v1 // (op 4546) 
s_add_i32 s0, s47, s16 // (op 4546) 

v_and_b32 v26, -4, v1 // (op 4546) 
s_ashr_i32 s1, s0, 31 // (op 4546) 
v_sub_u32 v27, v1, v26 // (op 4546) 

s_cmp_eq_i32 s10, 0 // (op 4546) 
s_mov_b32 s47, scc // (op 4546) 
s_cmp_eq_i32 s47, 0 // (op 4546) 
s_cselect_b32 s18, 0, -1 // (op 4546) 

s_and_b32 s19, s13, 31 // (op 4546) 
s_lshl_b32 s47, 1, s19 // (op 4546) 
s_add_i32 s22, s47, s18 // (op 4546) 


s_and_b32 s18, s1, s22 // (op 4546) 


s_add_i32 s1, s0, s18 // (op 4546) 


s_ashr_i32 s0, s1, s19 // (op 4546) 


s_lshl_b32 s1, s13, 24 // (op 4546) 
s_ashr_i32 s47, s1, 31 // (op 4546) 

s_xor_b32 s1, s0, s47 // (op 4546) 

s_sub_i32 s0, s1, s47 // (op 4546) 


s_mul_hi_i32 s1, s0, s14 // (op 4546) 
s_add_i32 s47, s1, s0 // (op 4546) 

s_ashr_i32 s1, s47, 31 // (op 4546) 
s_cmp_eq_i32 s14, 0 // (op 4546) 
s_mov_b32 s18, scc // (op 4546) 
s_cmp_eq_i32 s18, 0 // (op 4546) 
s_cselect_b32 s19, 0, -1 // (op 4546) 

s_and_b32 s18, s15, 31 // (op 4546) 
s_lshl_b32 s22, 1, s18 // (op 4546) 
s_add_i32 s23, s22, s19 // (op 4546) 


s_and_b32 s19, s1, s23 // (op 4546) 


s_add_i32 s1, s47, s19 // (op 4546) 


s_ashr_i32 s19, s1, s18 // (op 4546) 


s_lshl_b32 s1, s15, 24 // (op 4546) 
s_ashr_i32 s18, s1, 31 // (op 4546) 

s_xor_b32 s1, s19, s18 // (op 4546) 

s_sub_i32 s19, s1, s18 // (op 4546) 


s_add_u32 s22, s6, 64 // (op 4546) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4546) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4546) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4546) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4546) 

s_mul_i32 s1, s19, s22 // (op 4546) 

s_sub_i32 s47, s0, s1 // (op 4546) 


s_mul_hi_i32 s0, s16, s9 // (op 4546) 
s_add_i32 s1, s0, s16 // (op 4546) 

s_ashr_i32 s0, s1, 31 // (op 4546) 
s_cmp_eq_i32 s9, 0 // (op 4546) 
s_mov_b32 s18, scc // (op 4546) 
s_cmp_eq_i32 s18, 0 // (op 4546) 
s_cselect_b32 s19, 0, -1 // (op 4546) 

s_and_b32 s18, s45, 31 // (op 4546) 
s_lshl_b32 s24, 1, s18 // (op 4546) 
s_add_i32 s25, s24, s19 // (op 4546) 


s_and_b32 s19, s0, s25 // (op 4546) 


s_add_i32 s0, s1, s19 // (op 4546) 


s_ashr_i32 s1, s0, s18 // (op 4546) 


s_lshl_b32 s0, s45, 24 // (op 4546) 
s_ashr_i32 s18, s0, 31 // (op 4546) 

s_xor_b32 s0, s1, s18 // (op 4546) 

s_sub_i32 s1, s0, s18 // (op 4546) 


s_mul_i32 s0, s11, s22 // (op 4546) 

s_mul_i32 s18, s8, s0 // (op 4546) 

s_mul_i32 s0, s1, s18 // (op 4546) 

s_sub_u32 s18, s16, s0 // (op 4546) 

s_mul_hi_i32 s0, s18, s12 // (op 4546) 
s_add_i32 s19, s0, s18 // (op 4546) 


s_ashr_i32 s0, s19, 31 // (op 4546) 
s_cmp_eq_i32 s12, 0 // (op 4546) 
s_mov_b32 s18, scc // (op 4546) 
s_cmp_eq_i32 s18, 0 // (op 4546) 
s_cselect_b32 s22, 0, -1 // (op 4546) 

s_and_b32 s18, s44, 31 // (op 4546) 
s_lshl_b32 s23, 1, s18 // (op 4546) 
s_add_i32 s24, s23, s22 // (op 4546) 


s_and_b32 s25, s0, s24 // (op 4546) 


s_add_i32 s0, s19, s25 // (op 4546) 


s_ashr_i32 s19, s0, s18 // (op 4546) 


s_lshl_b32 s0, s44, 24 // (op 4546) 
s_ashr_i32 s18, s0, 31 // (op 4546) 

s_xor_b32 s0, s19, s18 // (op 4546) 

s_sub_i32 s19, s0, s18 // (op 4546) 


s_cmp_eq_i32 s1, 0 // (op 4546) 
s_mov_b32 s0, scc // (op 4546) 

s_cmp_eq_i32 s0, 0 // (op 4546) 
s_cselect_b32 s1, s19, s47 // (op 4546) 



v_lshl_add_u32 v26, s1, 6, v22 // (op 4546) 


v_mul_lo_u32 v22, v26, s38 // (op 4546) 

v_add_u32 v26, v27, v22 // (op 4546) 


s_mul_i32 s1, s38, s7 // (op 4548) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 4548) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 4548) least significant
s_mul_hi_u32 s18, s38, s6 // (op 4548) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4548) most significant: sum
s_add_u32 s1, s1, s18 // (op 4548) most significant: sum


s_mov_b32 s68, s40 // (op 4548) reg expression
s_mov_b32 s69, s41 // (op 4548) reg expression
s_mov_b32 s70, s0 // (op 4548) reg expression
s_mov_b32 s71, 131072 // (op 4548) call()

v_and_b32 v22, -8, v1 // (op 4674) 
v_sub_u32 v27, v1, v22 // (op 4674) 

v_lshlrev_b32 v22, 4, v27 // (op 4674) 

v_lshrrev_b32 v27, 3, v1 // (op 4674) 
v_lshl_add_u32 v28, v27, 7, v22 // (op 4674) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 4674) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 4277) 
v_mov_b64 v[28:29], s[0:1] // (op 4277) convert


v_mov_b64 v[30:31], v[14:15] // (op 4277) 
s_mov_b32 m0, s47 // (op 4277) reg expression
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 4277) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 4277) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 4277) most significant half

s_add_u32 m0, m0, 4096 // (op 4277) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 4277) Load value direct to lds






buffer_load_ubyte v22, v25, s[68:71], 0 offen offset: 0 // (op 4059) Load value
v_and_b32 v27, -8, v1 // (op 4617) 
v_sub_u32 v28, v1, v27 // (op 4617) 

v_lshlrev_b32 v27, 4, v28 // (op 4617) 

v_lshrrev_b32 v28, 3, v1 // (op 4617) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 4617) 


s_nop 0
v_readfirstlane_b32 s47, v29 // (op 4617) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 4282) 
v_mov_b64 v[28:29], s[0:1] // (op 4282) convert


v_mov_b64 v[30:31], v[18:19] // (op 4282) 
s_add_u32 s0, 8192, s47 // (op 4282) 
s_mov_b32 m0, s0 // (op 4282) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4282) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 4282) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 4282) most significant half

s_add_u32 m0, m0, 4096 // (op 4282) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4282) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 4282) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 4282) most significant half

s_add_u32 m0, m0, 4096 // (op 4282) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4282) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 4282) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 4282) most significant half

s_add_u32 m0, m0, 4096 // (op 4282) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4282) Load value direct to lds





v_mov_b64 v[28:29], s[34:35] // (op 4065) convert

v_mov_b32 v27, v23 // (op 4065) 
buffer_load_ubyte v30, v27, s[64:67], 0 offen offset: 0 // (op 4065) Load value
v_add_u32 v27, v27, v28 // (op 4065) 
buffer_load_ubyte v31, v27, s[64:67], 0 offen offset: 0 // (op 4065) Load value


v_and_b32 v27, -8, v1 // (op 4901) 
v_sub_u32 v28, v1, v27 // (op 4901) 

v_lshlrev_b32 v27, 4, v28 // (op 4901) 

v_lshrrev_b32 v28, 3, v1 // (op 4901) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 4901) 


s_nop 0
v_readfirstlane_b32 s47, v29 // (op 4901) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 4287) 
v_mov_b64 v[28:29], s[0:1] // (op 4287) convert


v_mov_b64 v[32:33], v[14:15] // (op 4287) 
s_add_u32 s0, 24576, s47 // (op 4287) 
s_mov_b32 m0, s0 // (op 4287) reg expression
s_mov_b32 s1, 128 // (op 4287) call()
buffer_load_dwordx4 v32, s[60:63], s1 offen offset: 0 lds // (op 4287) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 4287) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 4287) most significant half

s_add_u32 m0, m0, 4096 // (op 4287) 
buffer_load_dwordx4 v32, s[60:63], s1 offen offset: 0 lds // (op 4287) Load value direct to lds









buffer_load_ubyte v14, v25, s[68:71], 0 offen offset:4 // (op 4071) Load value

v_and_b32 v15, -8, v1 // (op 4857) 
v_sub_u32 v25, v1, v15 // (op 4857) 

v_lshlrev_b32 v15, 4, v25 // (op 4857) 

v_lshrrev_b32 v25, 3, v1 // (op 4857) 
v_lshl_add_u32 v27, v25, 7, v15 // (op 4857) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 4857) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 4289) 
v_mov_b64 v[28:29], s[0:1] // (op 4289) convert


v_mov_b64 v[32:33], v[18:19] // (op 4289) 
s_add_u32 s0, 32768, s47 // (op 4289) 
s_mov_b32 m0, s0 // (op 4289) reg expression
s_mov_b32 s1, 128 // (op 4289) call()
buffer_load_dwordx4 v32, s[48:51], s1 offen offset: 0 lds // (op 4289) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 4289) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 4289) most significant half

s_add_u32 m0, m0, 4096 // (op 4289) 
buffer_load_dwordx4 v32, s[48:51], s1 offen offset: 0 lds // (op 4289) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 4289) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 4289) most significant half

s_add_u32 m0, m0, 4096 // (op 4289) 
buffer_load_dwordx4 v32, s[48:51], s1 offen offset: 0 lds // (op 4289) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 4289) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 4289) most significant half

s_add_u32 m0, m0, 4096 // (op 4289) 
buffer_load_dwordx4 v32, s[48:51], s1 offen offset: 0 lds // (op 4289) Load value direct to lds







v_mov_b64 v[18:19], s[34:35] // (op 4077) convert

v_mov_b32 v15, v23 // (op 4077) 
buffer_load_ubyte v28, v15, s[64:67], 0 offen offset:4 // (op 4077) Load value
v_add_u32 v15, v15, v18 // (op 4077) 
buffer_load_ubyte v29, v15, s[64:67], 0 offen offset:4 // (op 4077) Load value



v_and_b32 v15, -4, v1 // (op 4790) 
v_sub_u32 v23, v1, v15 // (op 4790) 

v_and_b32 v15, 4294967292, v1 // (op 4790) 
v_add_u32 v25, v15, v23 // (op 4790) 





s_waitcnt vmcnt(15)
ds_write_b8 v25, v22 offset:49152 // (op 4080) Store local data 

v_and_b32 v15, -4, v1 // (op 4732) 
v_sub_u32 v23, v1, v15 // (op 4732) 

v_lshrrev_b32 v15, 2, v1 // (op 4732) 
v_lshl_add_u32 v25, v15, 3, v23 // (op 4732) 





s_waitcnt vmcnt(10)
ds_write_b8 v25, v30 offset:49408 // (op 4082) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v25, v31 offset:49412 // (op 4082) Store local data 

s_barrier  // (op 5077) 



ds_read_b128 v[32:35], v4 // (op 4117) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 4117) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5112) 



ds_read_u8 v15, v5 offset:49152 // (op 4147) Load local data 



ds_read_b128 v[40:43], v4 offset:4096 // (op 4122) Load local data 
ds_read_b128 v[44:47], v4 offset:4160 // (op 4122) Load local data 



ds_read_u8 v23, v5 offset:49280 // (op 4152) Load local data 
s_barrier  // (op 5061) 



ds_read_b128 v[48:51], v3 offset:8192 // (op 4097) Load local data 
ds_read_b128 v[52:55], v3 offset:8256 // (op 4097) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 5094) 



ds_read_u8 v25, v0 offset:49408 // (op 4127) Load local data 



ds_read_b128 v[56:59], v3 offset:12288 // (op 4102) Load local data 
ds_read_b128 v[60:63], v3 offset:12352 // (op 4102) Load local data 



ds_read_u8 v27, v0 offset:49536 // (op 4132) Load local data 



ds_read_b128 v[64:67], v3 offset:16384 // (op 4107) Load local data 
ds_read_b128 v[68:71], v3 offset:16448 // (op 4107) Load local data 



ds_read_u8 v18, v0 offset:49664 // (op 4137) Load local data 



ds_read_b128 v[72:75], v3 offset:20480 // (op 4112) Load local data 
ds_read_b128 v[76:79], v3 offset:20544 // (op 4112) Load local data 



ds_read_u8 v19, v0 offset:49792 // (op 4142) Load local data 
v_and_b32 v80, -8, v1 // (op 4589) 
v_sub_u32 v81, v1, v80 // (op 4589) 

v_lshlrev_b32 v80, 4, v81 // (op 4589) 

v_lshrrev_b32 v81, 3, v1 // (op 4589) 
v_lshl_add_u32 v82, v81, 7, v80 // (op 4589) 


s_nop 0
v_readfirstlane_b32 s47, v82 // (op 4589) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -8, v1 // (op 4646) 
v_sub_u32 v81, v1, v80 // (op 4646) 

v_lshlrev_b32 v80, 4, v81 // (op 4646) 

v_lshrrev_b32 v81, 3, v1 // (op 4646) 
v_lshl_add_u32 v82, v81, 7, v80 // (op 4646) 


s_nop 0
v_readfirstlane_b32 s0, v82 // (op 4646) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -4, v1 // (op 4703) 
v_sub_u32 v81, v1, v80 // (op 4703) 

v_lshrrev_b32 v80, 2, v1 // (op 4703) 
v_lshl_add_u32 v82, v80, 3, v81 // (op 4703) 


v_and_b32 v80, -4, v1 // (op 4761) 
v_sub_u32 v81, v1, v80 // (op 4761) 

v_and_b32 v80, 4294967292, v1 // (op 4761) 
v_add_u32 v83, v80, v81 // (op 4761) 


v_and_b32 v80, -8, v1 // (op 4842) 
v_sub_u32 v81, v1, v80 // (op 4842) 

v_lshlrev_b32 v80, 4, v81 // (op 4842) 

v_lshrrev_b32 v81, 3, v1 // (op 4842) 
v_lshl_add_u32 v84, v81, 7, v80 // (op 4842) 


s_nop 0
v_readfirstlane_b32 s1, v84 // (op 4842) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -8, v1 // (op 4886) 
v_sub_u32 v81, v1, v80 // (op 4886) 

v_lshlrev_b32 v80, 4, v81 // (op 4886) 

v_lshrrev_b32 v81, 3, v1 // (op 4886) 
v_lshl_add_u32 v84, v81, 7, v80 // (op 4886) 


s_nop 0
v_readfirstlane_b32 s18, v84 // (op 4886) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -4, v1 // (op 4930) 
v_sub_u32 v81, v1, v80 // (op 4930) 

v_lshrrev_b32 v80, 2, v1 // (op 4930) 
v_lshl_add_u32 v84, v80, 3, v81 // (op 4930) 


v_and_b32 v80, -4, v1 // (op 4959) 
v_sub_u32 v81, v1, v80 // (op 4959) 

v_and_b32 v80, 4294967292, v1 // (op 4959) 
v_add_u32 v85, v80, v81 // (op 4959) 


s_mov_b32 s19, 0 // (op 1365) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 1364) 
s_mov_b32 s24, s22 // (op 1364) convert

s_and_b32 s25, s24, 4294967294 // (op 1364) 

s_cmp_lt_u32 s19, s25 // (op 1364) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopBottom_KLoop_1364 // (op 1364) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopBottom_KLoop_1364 if false)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopTop_KLoop_1364:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v25, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 90) 
s_barrier  // (op 5080) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[40:47], a[4:7], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 380) 
s_add_u32 s19, s19, 2 // (op 1366) 
s_lshl_b64 s[22:23], s[30:31], 5 // (op 4249) 
v_mov_b64 v[80:81], s[22:23] // (op 4249) convert


v_mov_b64 v[86:87], v[16:17] // (op 4249) 
v_mov_b64 v[88:89], v[86:87] // (op 4249) 
s_mov_b32 m0, s0 // (op 4249) reg expression
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[32:39], a[8:11], v27, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 677) 
s_mov_b32 s22, 256 // (op 4249) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[40:47], a[12:15], v27, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 631) 
buffer_load_dwordx4 v88, s[60:63], s22 offen offset: 0 lds // (op 4249) Load value direct to lds
v_add_co_u32 v88, s[24:25], v88, v80 // (op 4249) least significant half
v_addc_co_u32 v89, s[24:25], 0, v89, s[24:25] // (op 4249) most significant half

s_add_u32 m0, m0, 4096 // (op 4249) 
buffer_load_dwordx4 v88, s[60:63], s22 offen offset: 0 lds // (op 4249) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:71], v[32:39], a[16:19], v18, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1063) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:71], v[40:47], a[20:23], v18, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1017) 
v_mov_b32 v80, v26 // (op 25) 

buffer_load_ubyte v22, v80, s[68:71], 0 offen offset:8 // (op 25) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[32:39], a[24:27], v19, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1449) 
s_barrier  // (op 5064) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[40:47], a[28:31], v19, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1403) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 4244) 
v_mov_b64 v[88:89], s[22:23] // (op 4244) convert


v_mov_b64 v[90:91], v[20:21] // (op 4244) 
v_mov_b64 v[92:93], v[90:91] // (op 4244) 
s_add_u32 s22, 8192, s47 // (op 4244) 
s_mov_b32 m0, s22 // (op 4244) reg expression
s_mov_b32 s23, 256 // (op 4244) call()
buffer_load_dwordx4 v92, s[48:51], s23 offen offset: 0 lds // (op 4244) Load value direct to lds
v_add_co_u32 v92, s[24:25], v92, v88 // (op 4244) least significant half
v_addc_co_u32 v93, s[24:25], 0, v93, s[24:25] // (op 4244) most significant half

s_add_u32 m0, m0, 4096 // (op 4244) 
buffer_load_dwordx4 v92, s[48:51], s23 offen offset: 0 lds // (op 4244) Load value direct to lds
v_add_co_u32 v92, s[24:25], v92, v88 // (op 4244) least significant half
v_addc_co_u32 v93, s[24:25], 0, v93, s[24:25] // (op 4244) most significant half

s_add_u32 m0, m0, 4096 // (op 4244) 
buffer_load_dwordx4 v92, s[48:51], s23 offen offset: 0 lds // (op 4244) Load value direct to lds
v_add_co_u32 v92, s[24:25], v92, v88 // (op 4244) least significant half
v_addc_co_u32 v93, s[24:25], 0, v93, s[24:25] // (op 4244) most significant half

s_add_u32 m0, m0, 4096 // (op 4244) 
buffer_load_dwordx4 v92, s[48:51], s23 offen offset: 0 lds // (op 4244) Load value direct to lds





v_mov_b64 v[88:89], s[34:35] // (op 17) convert

v_mov_b32 v81, v24 // (op 17) 
v_mov_b32 v92, v81 // (op 17) 
buffer_load_ubyte v30, v92, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v92, v92, v88 // (op 17) 
buffer_load_ubyte v31, v92, s[64:67], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 5152) 



s_waitcnt vmcnt(15)
ds_write_b8 v85, v14 offset:49920 // (op 259) Store local data 
s_barrier  // (op 5146) 



s_waitcnt vmcnt(10)
ds_write_b8 v84, v28 offset:50176 // (op 258) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v84, v29 offset:50180 // (op 258) Store local data 
s_barrier  // (op 5135) 



ds_read_b128 v[92:95], v8 offset:24576 // (op 251) Load local data 
ds_read_b128 v[96:99], v8 offset:24640 // (op 251) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5149) 



ds_read_u8 v88, v9 offset:49920 // (op 255) Load local data 



ds_read_b128 v[100:103], v8 offset:28672 // (op 385) Load local data 
ds_read_b128 v[104:107], v8 offset:28736 // (op 385) Load local data 



ds_read_u8 v89, v9 offset:50048 // (op 389) Load local data 
s_barrier  // (op 5129) 
s_barrier  // (op 5197) 



ds_read_b128 v[48:51], v7 offset:32768 // (op 249) Load local data 
ds_read_b128 v[52:55], v7 offset:32832 // (op 249) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 5142) 



ds_read_u8 v25, v6 offset:50176 // (op 253) Load local data 



ds_read_b128 v[56:59], v7 offset:36864 // (op 634) Load local data 
ds_read_b128 v[60:63], v7 offset:36928 // (op 634) Load local data 



ds_read_u8 v27, v6 offset:50304 // (op 637) Load local data 



ds_read_b128 v[64:67], v7 offset:40960 // (op 1020) Load local data 
ds_read_b128 v[68:71], v7 offset:41024 // (op 1020) Load local data 



ds_read_u8 v18, v6 offset:50432 // (op 1023) Load local data 



ds_read_b128 v[72:75], v7 offset:45056 // (op 1406) Load local data 
ds_read_b128 v[76:79], v7 offset:45120 // (op 1406) Load local data 



ds_read_u8 v19, v6 offset:50560 // (op 1409) Load local data 
s_barrier  // (op 5227) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[92:99], a[0:3], v25, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 256) 
s_barrier  // (op 5139) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[100:107], a[4:7], v25, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 381) 
s_lshl_b64 s[22:23], s[30:31], 5 // (op 4272) 
v_mov_b64 v[108:109], s[22:23] // (op 4272) convert


v_mov_b64 v[110:111], v[86:87] // (op 4272) 
s_add_u32 s22, 24576, s18 // (op 4272) 
s_mov_b32 m0, s22 // (op 4272) reg expression
s_mov_b32 s23, 384 // (op 4272) call()
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:63], v[92:99], a[8:11], v27, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 678) 
buffer_load_dwordx4 v110, s[60:63], s23 offen offset: 0 lds // (op 4272) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[100:107], a[12:15], v27, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 632) 
v_add_co_u32 v110, s[24:25], v110, v108 // (op 4272) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 4272) most significant half

s_add_u32 m0, m0, 4096 // (op 4272) 
buffer_load_dwordx4 v110, s[60:63], s23 offen offset: 0 lds // (op 4272) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:71], v[92:99], a[16:19], v18, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1064) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:71], v[100:107], a[20:23], v18, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1018) 
buffer_load_ubyte v14, v80, s[68:71], 0 offen offset:12 // (op 262) Load value

v_add_co_u32 v16, vcc, 256, v16 // (op 4410) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 4410) most significant half
v_add_u32 v26, 8, v26 // (op 4538) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[92:99], a[24:27], v19, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1450) 
s_barrier  // (op 5132) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[100:107], a[28:31], v19, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1404) 
s_lshl_b64 s[22:23], s[26:27], 5 // (op 4267) 


v_mov_b64 v[86:87], s[22:23] // (op 4267) convert


v_mov_b64 v[88:89], v[90:91] // (op 4267) 
s_add_u32 s22, 32768, s1 // (op 4267) 
s_mov_b32 m0, s22 // (op 4267) reg expression
s_mov_b32 s23, 384 // (op 4267) call()
buffer_load_dwordx4 v88, s[48:51], s23 offen offset: 0 lds // (op 4267) Load value direct to lds
v_add_co_u32 v88, s[24:25], v88, v86 // (op 4267) least significant half
v_addc_co_u32 v89, s[24:25], 0, v89, s[24:25] // (op 4267) most significant half

s_add_u32 m0, m0, 4096 // (op 4267) 
buffer_load_dwordx4 v88, s[48:51], s23 offen offset: 0 lds // (op 4267) Load value direct to lds
v_add_co_u32 v88, s[24:25], v88, v86 // (op 4267) least significant half
v_addc_co_u32 v89, s[24:25], 0, v89, s[24:25] // (op 4267) most significant half

s_add_u32 m0, m0, 4096 // (op 4267) 
buffer_load_dwordx4 v88, s[48:51], s23 offen offset: 0 lds // (op 4267) Load value direct to lds
v_add_co_u32 v88, s[24:25], v88, v86 // (op 4267) least significant half
v_addc_co_u32 v89, s[24:25], 0, v89, s[24:25] // (op 4267) most significant half

s_add_u32 m0, m0, 4096 // (op 4267) 
buffer_load_dwordx4 v88, s[48:51], s23 offen offset: 0 lds // (op 4267) Load value direct to lds






v_add_co_u32 v20, vcc, 256, v20 // (op 4346) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 4346) most significant half
v_mov_b64 v[86:87], s[34:35] // (op 261) convert

v_mov_b32 v80, v81 // (op 261) 
buffer_load_ubyte v28, v80, s[64:67], 0 offen offset:12 // (op 261) Load value
v_add_u32 v80, v80, v86 // (op 261) 
buffer_load_ubyte v29, v80, s[64:67], 0 offen offset:12 // (op 261) Load value



s_barrier  // (op 5115) 
v_add_u32 v24, 8, v24 // (op 4474) 



s_waitcnt vmcnt(15)
ds_write_b8 v83, v22 offset:49152 // (op 30) Store local data 
s_barrier  // (op 5097) 



s_waitcnt vmcnt(10)
ds_write_b8 v82, v30 offset:49408 // (op 21) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v82, v31 offset:49412 // (op 21) Store local data 
s_barrier  // (op 5086) 



ds_read_b128 v[32:35], v4 // (op 77) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5122) 



ds_read_u8 v15, v5 offset:49152 // (op 87) Load local data 



ds_read_b128 v[40:43], v4 offset:4096 // (op 375) Load local data 
ds_read_b128 v[44:47], v4 offset:4160 // (op 375) Load local data 



ds_read_u8 v23, v5 offset:49280 // (op 379) Load local data 
s_barrier  // (op 5070) 



ds_read_b128 v[48:51], v3 offset:8192 // (op 72) Load local data 
ds_read_b128 v[52:55], v3 offset:8256 // (op 72) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 5104) 



ds_read_u8 v25, v0 offset:49408 // (op 82) Load local data 



ds_read_b128 v[56:59], v3 offset:12288 // (op 626) Load local data 
ds_read_b128 v[60:63], v3 offset:12352 // (op 626) Load local data 



ds_read_u8 v27, v0 offset:49536 // (op 629) Load local data 



ds_read_b128 v[64:67], v3 offset:16384 // (op 1012) Load local data 
ds_read_b128 v[68:71], v3 offset:16448 // (op 1012) Load local data 



ds_read_u8 v18, v0 offset:49664 // (op 1015) Load local data 



ds_read_b128 v[72:75], v3 offset:20480 // (op 1398) Load local data 
ds_read_b128 v[76:79], v3 offset:20544 // (op 1398) Load local data 



ds_read_u8 v19, v0 offset:49792 // (op 1401) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 1364) 
s_mov_b32 s24, s22 // (op 1364) convert

s_and_b32 s25, s24, 4294967294 // (op 1364) 

s_cmp_lt_u32 s19, s25 // (op 1364) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopTop_KLoop_1364 // (op 1364) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopTop_KLoop_1364 if true)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopBottom_KLoop_1364:

















s_barrier  // (op 5211) 
s_mul_hi_i32 s47, s16, s10 // (op 4302) 
v_lshrrev_b32 v24, 3, v1 // (op 4302) 
s_add_i32 s0, s47, s16 // (op 4302) 

v_and_b32 v26, -8, v1 // (op 4302) 
s_ashr_i32 s1, s0, 31 // (op 4302) 
v_sub_u32 v15, v1, v26 // (op 4302) 

s_cmp_eq_i32 s10, 0 // (op 4302) 
s_mov_b32 s47, scc // (op 4302) 
v_lshlrev_b32 v26, 4, v15 // (op 4302) 

s_cmp_eq_i32 s47, 0 // (op 4302) 
s_cselect_b32 s18, 0, -1 // (op 4302) 

s_and_b32 s19, s13, 31 // (op 4302) 
s_lshl_b32 s47, 1, s19 // (op 4302) 
s_add_i32 s22, s47, s18 // (op 4302) 


s_and_b32 s18, s1, s22 // (op 4302) 


s_add_i32 s1, s0, s18 // (op 4302) 


s_ashr_i32 s0, s1, s19 // (op 4302) 


s_lshl_b32 s1, s13, 24 // (op 4302) 
s_ashr_i32 s47, s1, 31 // (op 4302) 

s_xor_b32 s1, s0, s47 // (op 4302) 

s_sub_i32 s0, s1, s47 // (op 4302) 


s_mul_hi_i32 s1, s0, s14 // (op 4302) 
s_add_i32 s47, s1, s0 // (op 4302) 

s_ashr_i32 s1, s47, 31 // (op 4302) 
s_cmp_eq_i32 s14, 0 // (op 4302) 
s_mov_b32 s18, scc // (op 4302) 
s_cmp_eq_i32 s18, 0 // (op 4302) 
s_cselect_b32 s19, 0, -1 // (op 4302) 

s_and_b32 s18, s15, 31 // (op 4302) 
s_lshl_b32 s22, 1, s18 // (op 4302) 
s_add_i32 s23, s22, s19 // (op 4302) 


s_and_b32 s19, s1, s23 // (op 4302) 


s_add_i32 s1, s47, s19 // (op 4302) 


s_ashr_i32 s19, s1, s18 // (op 4302) 


s_lshl_b32 s1, s15, 24 // (op 4302) 
s_ashr_i32 s18, s1, 31 // (op 4302) 

s_xor_b32 s1, s19, s18 // (op 4302) 

s_sub_i32 s19, s1, s18 // (op 4302) 


s_mul_i32 s1, s0, s11 // (op 4302) 

s_sub_u32 s0, s16, s1 // (op 4302) 

s_mul_i32 s1, s19, s11 // (op 4302) 
s_add_u32 s1, s1, s0 // (op 4302) 


s_mul_hi_i32 s0, s16, s9 // (op 4302) 
s_add_i32 s47, s0, s16 // (op 4302) 

s_ashr_i32 s0, s47, 31 // (op 4302) 
s_cmp_eq_i32 s9, 0 // (op 4302) 
s_mov_b32 s18, scc // (op 4302) 
s_cmp_eq_i32 s18, 0 // (op 4302) 
s_cselect_b32 s19, 0, -1 // (op 4302) 

s_and_b32 s18, s45, 31 // (op 4302) 
s_lshl_b32 s22, 1, s18 // (op 4302) 
s_add_i32 s23, s22, s19 // (op 4302) 


s_and_b32 s19, s0, s23 // (op 4302) 


s_add_i32 s0, s47, s19 // (op 4302) 


s_ashr_i32 s19, s0, s18 // (op 4302) 


s_lshl_b32 s0, s45, 24 // (op 4302) 
s_ashr_i32 s18, s0, 31 // (op 4302) 

s_xor_b32 s0, s19, s18 // (op 4302) 

s_sub_i32 s19, s0, s18 // (op 4302) 


s_add_u32 s22, s6, 64 // (op 4302) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4302) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4302) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4302) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4302) 

s_mul_i32 s0, s11, s22 // (op 4302) 

s_mul_i32 s18, s8, s0 // (op 4302) 

s_mul_i32 s0, s19, s18 // (op 4302) 

s_sub_u32 s18, s16, s0 // (op 4302) 

s_mov_b32 s0, s18 // (op 4302) 
s_mul_hi_i32 s47, s0, s12 // (op 4302) 
s_add_i32 s22, s47, s0 // (op 4302) 


s_ashr_i32 s0, s22, 31 // (op 4302) 
s_cmp_eq_i32 s12, 0 // (op 4302) 
s_mov_b32 s47, scc // (op 4302) 
s_cmp_eq_i32 s47, 0 // (op 4302) 
s_cselect_b32 s23, 0, -1 // (op 4302) 

s_and_b32 s47, s44, 31 // (op 4302) 
s_lshl_b32 s24, 1, s47 // (op 4302) 
s_add_i32 s25, s24, s23 // (op 4302) 


s_and_b32 s23, s0, s25 // (op 4302) 


s_add_i32 s0, s22, s23 // (op 4302) 


s_ashr_i32 s22, s0, s47 // (op 4302) 


s_lshl_b32 s0, s44, 24 // (op 4302) 
s_ashr_i32 s47, s0, 31 // (op 4302) 

s_xor_b32 s0, s22, s47 // (op 4302) 

s_sub_i32 s22, s0, s47 // (op 4302) 


s_mul_i32 s0, s46, s11 // (op 4302) 
s_add_u32 s24, s20, 128 // (op 4302) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4302) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 4302) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 4302) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 7 // (op 4302) 

s_sub_i32 s23, s24, s0 // (op 4302) 

s_mul_i32 s47, s22, s23 // (op 4302) 


s_sub_u32 s22, s18, s47 // (op 4302) 


s_add_u32 s18, s22, s0 // (op 4302) 


s_cmp_eq_i32 s19, 0 // (op 4302) 
s_mov_b32 s0, scc // (op 4302) 

s_cmp_eq_i32 s0, 0 // (op 4302) 
s_cselect_b32 s19, s18, s1 // (op 4302) 



v_lshl_add_u32 v15, s19, 7, v24 // (op 4302) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 4302) 
s_lshr_b32 s47, s0, 1 // (op 4302) 

v_lshl_add_u32 v24, s47, 8, v26 // (op 4302) 


v_mov_b32 v17, 0 // (op 4302) convert
v_mov_b32 v16, v24 // (op 4302) convert

v_mul_lo_u32 v21, v15, s27 // (op 4302) most significant: low of low * high
v_mul_hi_u32 v22, v15, s26 // (op 4302) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 4302) least significant: low of low * low
v_add_u32 v21, v21, v22 // (op 4302) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 4302) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 4302) most significant half



s_mul_i32 s1, s26, s21 // (op 4304) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 4304) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 4304) least significant
s_mul_hi_u32 s18, s26, s20 // (op 4304) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4304) most significant: sum
s_add_u32 s1, s1, s18 // (op 4304) most significant: sum


s_mov_b32 s48, s28 // (op 4304) reg expression
s_mov_b32 s49, s29 // (op 4304) reg expression
s_mov_b32 s50, s0 // (op 4304) reg expression
s_mov_b32 s51, 131072 // (op 4304) call()

s_mul_hi_i32 s47, s16, s10 // (op 4376) 
v_lshrrev_b32 v22, 3, v1 // (op 4376) 
s_add_i32 s0, s47, s16 // (op 4376) 

v_and_b32 v24, -8, v1 // (op 4376) 
s_ashr_i32 s1, s0, 31 // (op 4376) 
v_sub_u32 v26, v1, v24 // (op 4376) 

s_cmp_eq_i32 s10, 0 // (op 4376) 
s_mov_b32 s47, scc // (op 4376) 
v_lshlrev_b32 v24, 4, v26 // (op 4376) 

s_cmp_eq_i32 s47, 0 // (op 4376) 
s_cselect_b32 s18, 0, -1 // (op 4376) 

s_and_b32 s19, s13, 31 // (op 4376) 
s_lshl_b32 s47, 1, s19 // (op 4376) 
s_add_i32 s22, s47, s18 // (op 4376) 


s_and_b32 s18, s1, s22 // (op 4376) 


s_add_i32 s1, s0, s18 // (op 4376) 


s_ashr_i32 s0, s1, s19 // (op 4376) 


s_lshl_b32 s1, s13, 24 // (op 4376) 
s_ashr_i32 s47, s1, 31 // (op 4376) 

s_xor_b32 s1, s0, s47 // (op 4376) 

s_sub_i32 s0, s1, s47 // (op 4376) 


s_mul_hi_i32 s1, s0, s14 // (op 4376) 
s_add_i32 s47, s1, s0 // (op 4376) 

s_ashr_i32 s1, s47, 31 // (op 4376) 
s_cmp_eq_i32 s14, 0 // (op 4376) 
s_mov_b32 s18, scc // (op 4376) 
s_cmp_eq_i32 s18, 0 // (op 4376) 
s_cselect_b32 s19, 0, -1 // (op 4376) 

s_and_b32 s18, s15, 31 // (op 4376) 
s_lshl_b32 s22, 1, s18 // (op 4376) 
s_add_i32 s23, s22, s19 // (op 4376) 


s_and_b32 s19, s1, s23 // (op 4376) 


s_add_i32 s1, s47, s19 // (op 4376) 


s_ashr_i32 s19, s1, s18 // (op 4376) 


s_lshl_b32 s1, s15, 24 // (op 4376) 
s_ashr_i32 s18, s1, 31 // (op 4376) 

s_xor_b32 s1, s19, s18 // (op 4376) 

s_sub_i32 s19, s1, s18 // (op 4376) 


s_add_u32 s22, s6, 64 // (op 4376) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4376) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4376) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4376) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4376) 

s_mul_i32 s1, s19, s22 // (op 4376) 

s_sub_i32 s47, s0, s1 // (op 4376) 


s_mul_hi_i32 s0, s16, s9 // (op 4376) 
s_add_i32 s1, s0, s16 // (op 4376) 

s_ashr_i32 s0, s1, 31 // (op 4376) 
s_cmp_eq_i32 s9, 0 // (op 4376) 
s_mov_b32 s18, scc // (op 4376) 
s_cmp_eq_i32 s18, 0 // (op 4376) 
s_cselect_b32 s19, 0, -1 // (op 4376) 

s_and_b32 s18, s45, 31 // (op 4376) 
s_lshl_b32 s24, 1, s18 // (op 4376) 
s_add_i32 s25, s24, s19 // (op 4376) 


s_and_b32 s19, s0, s25 // (op 4376) 


s_add_i32 s0, s1, s19 // (op 4376) 


s_ashr_i32 s1, s0, s18 // (op 4376) 


s_lshl_b32 s0, s45, 24 // (op 4376) 
s_ashr_i32 s18, s0, 31 // (op 4376) 

s_xor_b32 s0, s1, s18 // (op 4376) 

s_sub_i32 s1, s0, s18 // (op 4376) 


s_mul_i32 s0, s11, s22 // (op 4376) 

s_mul_i32 s18, s8, s0 // (op 4376) 

s_mul_i32 s0, s1, s18 // (op 4376) 

s_sub_u32 s18, s16, s0 // (op 4376) 

s_mul_hi_i32 s0, s18, s12 // (op 4376) 
s_add_i32 s19, s0, s18 // (op 4376) 


s_ashr_i32 s0, s19, 31 // (op 4376) 
s_cmp_eq_i32 s12, 0 // (op 4376) 
s_mov_b32 s18, scc // (op 4376) 
s_cmp_eq_i32 s18, 0 // (op 4376) 
s_cselect_b32 s22, 0, -1 // (op 4376) 

s_and_b32 s18, s44, 31 // (op 4376) 
s_lshl_b32 s23, 1, s18 // (op 4376) 
s_add_i32 s24, s23, s22 // (op 4376) 


s_and_b32 s25, s0, s24 // (op 4376) 


s_add_i32 s0, s19, s25 // (op 4376) 


s_ashr_i32 s19, s0, s18 // (op 4376) 


s_lshl_b32 s0, s44, 24 // (op 4376) 
s_ashr_i32 s18, s0, 31 // (op 4376) 

s_xor_b32 s0, s19, s18 // (op 4376) 

s_sub_i32 s19, s0, s18 // (op 4376) 


s_cmp_eq_i32 s1, 0 // (op 4376) 
s_mov_b32 s0, scc // (op 4376) 

s_cmp_eq_i32 s0, 0 // (op 4376) 
s_cselect_b32 s1, s19, s47 // (op 4376) 



v_lshl_add_u32 v26, s1, 6, v22 // (op 4376) 


v_mul_lo_u32 v17, v26, s31 // (op 4376) most significant: low of low * high
v_mul_hi_u32 v15, v26, s30 // (op 4376) most significant: high of low * low
v_mul_lo_u32 v16, v26, s30 // (op 4376) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 4376) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 4376) 
s_lshr_b32 s47, s0, 1 // (op 4376) 

v_lshl_add_u32 v15, s47, 8, v24 // (op 4376) 


v_mov_b32 v31, 0 // (op 4376) convert
v_mov_b32 v30, v15 // (op 4376) convert

v_add_co_u32 v32, s[0:1], v30, v16 // (op 4376) least significant half
v_addc_co_u32 v33, s[0:1], v31, v17, s[0:1] // (op 4376) most significant half



s_mul_i32 s1, s30, s7 // (op 4378) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 4378) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 4378) least significant
s_mul_hi_u32 s18, s30, s6 // (op 4378) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4378) most significant: sum
s_add_u32 s1, s1, s18 // (op 4378) most significant: sum


s_mov_b32 s60, s32 // (op 4378) reg expression
s_mov_b32 s61, s33 // (op 4378) reg expression
s_mov_b32 s62, s0 // (op 4378) reg expression
s_mov_b32 s63, 131072 // (op 4378) call()

s_mul_hi_i32 s47, s16, s10 // (op 4440) 
v_lshrrev_b32 v22, 2, v1 // (op 4440) 
s_add_i32 s0, s47, s16 // (op 4440) 

v_lshlrev_b32 v24, 1, v22 // (op 4440) 

s_ashr_i32 s1, s0, 31 // (op 4440) 
v_and_b32 v22, -4, v1 // (op 4440) 
s_cmp_eq_i32 s10, 0 // (op 4440) 
s_mov_b32 s47, scc // (op 4440) 
v_sub_u32 v26, v1, v22 // (op 4440) 

s_cmp_eq_i32 s47, 0 // (op 4440) 
s_cselect_b32 s18, 0, -1 // (op 4440) 

s_and_b32 s19, s13, 31 // (op 4440) 
s_lshl_b32 s47, 1, s19 // (op 4440) 
s_add_i32 s22, s47, s18 // (op 4440) 


s_and_b32 s18, s1, s22 // (op 4440) 


s_add_i32 s1, s0, s18 // (op 4440) 


s_ashr_i32 s0, s1, s19 // (op 4440) 


s_lshl_b32 s1, s13, 24 // (op 4440) 
s_ashr_i32 s47, s1, 31 // (op 4440) 

s_xor_b32 s1, s0, s47 // (op 4440) 

s_sub_i32 s0, s1, s47 // (op 4440) 


s_mul_hi_i32 s1, s0, s14 // (op 4440) 
s_add_i32 s47, s1, s0 // (op 4440) 

s_ashr_i32 s1, s47, 31 // (op 4440) 
s_cmp_eq_i32 s14, 0 // (op 4440) 
s_mov_b32 s18, scc // (op 4440) 
s_cmp_eq_i32 s18, 0 // (op 4440) 
s_cselect_b32 s19, 0, -1 // (op 4440) 

s_and_b32 s18, s15, 31 // (op 4440) 
s_lshl_b32 s22, 1, s18 // (op 4440) 
s_add_i32 s23, s22, s19 // (op 4440) 


s_and_b32 s19, s1, s23 // (op 4440) 


s_add_i32 s1, s47, s19 // (op 4440) 


s_ashr_i32 s19, s1, s18 // (op 4440) 


s_lshl_b32 s1, s15, 24 // (op 4440) 
s_ashr_i32 s18, s1, 31 // (op 4440) 

s_xor_b32 s1, s19, s18 // (op 4440) 

s_sub_i32 s19, s1, s18 // (op 4440) 


s_mul_i32 s1, s0, s11 // (op 4440) 

s_sub_u32 s0, s16, s1 // (op 4440) 

s_mul_i32 s1, s19, s11 // (op 4440) 
s_add_u32 s1, s1, s0 // (op 4440) 


s_mul_hi_i32 s0, s16, s9 // (op 4440) 
s_add_i32 s47, s0, s16 // (op 4440) 

s_ashr_i32 s0, s47, 31 // (op 4440) 
s_cmp_eq_i32 s9, 0 // (op 4440) 
s_mov_b32 s18, scc // (op 4440) 
s_cmp_eq_i32 s18, 0 // (op 4440) 
s_cselect_b32 s19, 0, -1 // (op 4440) 

s_and_b32 s18, s45, 31 // (op 4440) 
s_lshl_b32 s22, 1, s18 // (op 4440) 
s_add_i32 s23, s22, s19 // (op 4440) 


s_and_b32 s19, s0, s23 // (op 4440) 


s_add_i32 s0, s47, s19 // (op 4440) 


s_ashr_i32 s19, s0, s18 // (op 4440) 


s_lshl_b32 s0, s45, 24 // (op 4440) 
s_ashr_i32 s18, s0, 31 // (op 4440) 

s_xor_b32 s0, s19, s18 // (op 4440) 

s_sub_i32 s19, s0, s18 // (op 4440) 


s_add_u32 s22, s6, 64 // (op 4440) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4440) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4440) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4440) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4440) 

s_mul_i32 s0, s11, s22 // (op 4440) 

s_mul_i32 s18, s8, s0 // (op 4440) 

s_mul_i32 s0, s19, s18 // (op 4440) 

s_sub_u32 s18, s16, s0 // (op 4440) 

s_mov_b32 s0, s18 // (op 4440) 
s_mul_hi_i32 s47, s0, s12 // (op 4440) 
s_add_i32 s22, s47, s0 // (op 4440) 


s_ashr_i32 s0, s22, 31 // (op 4440) 
s_cmp_eq_i32 s12, 0 // (op 4440) 
s_mov_b32 s47, scc // (op 4440) 
s_cmp_eq_i32 s47, 0 // (op 4440) 
s_cselect_b32 s23, 0, -1 // (op 4440) 

s_and_b32 s47, s44, 31 // (op 4440) 
s_lshl_b32 s24, 1, s47 // (op 4440) 
s_add_i32 s25, s24, s23 // (op 4440) 


s_and_b32 s23, s0, s25 // (op 4440) 


s_add_i32 s0, s22, s23 // (op 4440) 


s_ashr_i32 s22, s0, s47 // (op 4440) 


s_lshl_b32 s0, s44, 24 // (op 4440) 
s_ashr_i32 s47, s0, 31 // (op 4440) 

s_xor_b32 s0, s22, s47 // (op 4440) 

s_sub_i32 s22, s0, s47 // (op 4440) 


s_mul_i32 s0, s46, s11 // (op 4440) 
s_add_u32 s24, s20, 128 // (op 4440) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4440) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 4440) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 4440) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 7 // (op 4440) 

s_sub_i32 s23, s24, s0 // (op 4440) 

s_mul_i32 s47, s22, s23 // (op 4440) 


s_sub_u32 s22, s18, s47 // (op 4440) 


s_add_u32 s18, s22, s0 // (op 4440) 


s_cmp_eq_i32 s19, 0 // (op 4440) 
s_mov_b32 s0, scc // (op 4440) 

s_cmp_eq_i32 s0, 0 // (op 4440) 
s_cselect_b32 s19, s18, s1 // (op 4440) 



v_lshl_add_u32 v22, s19, 7, v24 // (op 4440) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 4440) 
s_lshr_b32 s47, s0, 1 // (op 4440) 

v_lshl_add_u32 v24, s47, 3, v26 // (op 4440) 


v_mul_lo_u32 v26, v22, s34 // (op 4440) 
v_add_u32 v26, v26, v24 // (op 4440) 


s_mul_i32 s1, s34, s21 // (op 4442) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 4442) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 4442) least significant
s_mul_hi_u32 s18, s34, s20 // (op 4442) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4442) most significant: sum
s_add_u32 s1, s1, s18 // (op 4442) most significant: sum


s_mov_b32 s64, s36 // (op 4442) reg expression
s_mov_b32 s65, s37 // (op 4442) reg expression
s_mov_b32 s66, s0 // (op 4442) reg expression
s_mov_b32 s67, 131072 // (op 4442) call()

s_mul_hi_i32 s47, s16, s10 // (op 4504) 
v_lshrrev_b32 v22, 2, v1 // (op 4504) 
s_add_i32 s0, s47, s16 // (op 4504) 

v_and_b32 v24, -4, v1 // (op 4504) 
s_ashr_i32 s1, s0, 31 // (op 4504) 
v_sub_u32 v15, v1, v24 // (op 4504) 

s_cmp_eq_i32 s10, 0 // (op 4504) 
s_mov_b32 s47, scc // (op 4504) 
s_cmp_eq_i32 s47, 0 // (op 4504) 
s_cselect_b32 s18, 0, -1 // (op 4504) 

s_and_b32 s19, s13, 31 // (op 4504) 
s_lshl_b32 s47, 1, s19 // (op 4504) 
s_add_i32 s22, s47, s18 // (op 4504) 


s_and_b32 s18, s1, s22 // (op 4504) 


s_add_i32 s1, s0, s18 // (op 4504) 


s_ashr_i32 s0, s1, s19 // (op 4504) 


s_lshl_b32 s1, s13, 24 // (op 4504) 
s_ashr_i32 s47, s1, 31 // (op 4504) 

s_xor_b32 s1, s0, s47 // (op 4504) 

s_sub_i32 s0, s1, s47 // (op 4504) 


s_mul_hi_i32 s1, s0, s14 // (op 4504) 
s_add_i32 s47, s1, s0 // (op 4504) 

s_ashr_i32 s1, s47, 31 // (op 4504) 
s_cmp_eq_i32 s14, 0 // (op 4504) 
s_mov_b32 s18, scc // (op 4504) 
s_cmp_eq_i32 s18, 0 // (op 4504) 
s_cselect_b32 s19, 0, -1 // (op 4504) 

s_and_b32 s18, s15, 31 // (op 4504) 
s_lshl_b32 s22, 1, s18 // (op 4504) 
s_add_i32 s23, s22, s19 // (op 4504) 


s_and_b32 s19, s1, s23 // (op 4504) 


s_add_i32 s1, s47, s19 // (op 4504) 


s_ashr_i32 s19, s1, s18 // (op 4504) 


s_lshl_b32 s1, s15, 24 // (op 4504) 
s_ashr_i32 s18, s1, 31 // (op 4504) 

s_xor_b32 s1, s19, s18 // (op 4504) 

s_sub_i32 s19, s1, s18 // (op 4504) 


s_add_u32 s22, s6, 64 // (op 4504) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4504) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4504) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4504) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 4504) 

s_mul_i32 s1, s19, s22 // (op 4504) 

s_sub_i32 s47, s0, s1 // (op 4504) 


s_mul_hi_i32 s0, s16, s9 // (op 4504) 
s_add_i32 s1, s0, s16 // (op 4504) 

s_ashr_i32 s0, s1, 31 // (op 4504) 
s_cmp_eq_i32 s9, 0 // (op 4504) 
s_mov_b32 s18, scc // (op 4504) 
s_cmp_eq_i32 s18, 0 // (op 4504) 
s_cselect_b32 s19, 0, -1 // (op 4504) 

s_and_b32 s18, s45, 31 // (op 4504) 
s_lshl_b32 s24, 1, s18 // (op 4504) 
s_add_i32 s25, s24, s19 // (op 4504) 


s_and_b32 s19, s0, s25 // (op 4504) 


s_add_i32 s0, s1, s19 // (op 4504) 


s_ashr_i32 s1, s0, s18 // (op 4504) 


s_lshl_b32 s0, s45, 24 // (op 4504) 
s_ashr_i32 s18, s0, 31 // (op 4504) 

s_xor_b32 s0, s1, s18 // (op 4504) 

s_sub_i32 s1, s0, s18 // (op 4504) 


s_mul_i32 s0, s11, s22 // (op 4504) 

s_mul_i32 s18, s8, s0 // (op 4504) 

s_mul_i32 s0, s1, s18 // (op 4504) 

s_sub_u32 s18, s16, s0 // (op 4504) 

s_mul_hi_i32 s0, s18, s12 // (op 4504) 
s_add_i32 s19, s0, s18 // (op 4504) 


s_ashr_i32 s0, s19, 31 // (op 4504) 
s_cmp_eq_i32 s12, 0 // (op 4504) 
s_mov_b32 s18, scc // (op 4504) 
s_cmp_eq_i32 s18, 0 // (op 4504) 
s_cselect_b32 s22, 0, -1 // (op 4504) 

s_and_b32 s18, s44, 31 // (op 4504) 
s_lshl_b32 s23, 1, s18 // (op 4504) 
s_add_i32 s24, s23, s22 // (op 4504) 


s_and_b32 s25, s0, s24 // (op 4504) 


s_add_i32 s0, s19, s25 // (op 4504) 


s_ashr_i32 s19, s0, s18 // (op 4504) 


s_lshl_b32 s0, s44, 24 // (op 4504) 
s_ashr_i32 s18, s0, 31 // (op 4504) 

s_xor_b32 s0, s19, s18 // (op 4504) 

s_sub_i32 s19, s0, s18 // (op 4504) 


s_cmp_eq_i32 s1, 0 // (op 4504) 
s_mov_b32 s0, scc // (op 4504) 

s_cmp_eq_i32 s0, 0 // (op 4504) 
s_cselect_b32 s1, s19, s47 // (op 4504) 



v_lshl_add_u32 v24, s1, 6, v22 // (op 4504) 


v_mul_lo_u32 v22, v24, s38 // (op 4504) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 4504) 
s_lshr_b32 s47, s0, 1 // (op 4504) 

v_lshl_add_u32 v24, s47, 3, v15 // (op 4504) 


v_add_u32 v15, v24, v22 // (op 4504) 


s_mul_i32 s1, s38, s7 // (op 4506) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 4506) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 4506) least significant
s_mul_hi_u32 s18, s38, s6 // (op 4506) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4506) most significant: sum
s_add_u32 s1, s1, s18 // (op 4506) most significant: sum


s_mov_b32 s68, s40 // (op 4506) reg expression
s_mov_b32 s69, s41 // (op 4506) reg expression
s_mov_b32 s70, s0 // (op 4506) reg expression
s_mov_b32 s71, 131072 // (op 4506) call()

v_and_b32 v22, -8, v1 // (op 4602) 
v_sub_u32 v24, v1, v22 // (op 4602) 

v_lshlrev_b32 v22, 4, v24 // (op 4602) 

v_lshrrev_b32 v24, 3, v1 // (op 4602) 
v_lshl_add_u32 v16, v24, 7, v22 // (op 4602) 


s_nop 0
v_readfirstlane_b32 s47, v16 // (op 4602) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 4659) 
v_sub_u32 v24, v1, v22 // (op 4659) 

v_lshlrev_b32 v22, 4, v24 // (op 4659) 

v_lshrrev_b32 v24, 3, v1 // (op 4659) 
v_lshl_add_u32 v16, v24, 7, v22 // (op 4659) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 4659) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 4716) 
v_sub_u32 v24, v1, v22 // (op 4716) 

v_lshrrev_b32 v22, 2, v1 // (op 4716) 
v_lshl_add_u32 v16, v22, 3, v24 // (op 4716) 


v_and_b32 v17, -4, v1 // (op 4774) 
v_sub_u32 v22, v1, v17 // (op 4774) 

v_and_b32 v17, 4294967292, v1 // (op 4774) 
v_add_u32 v24, v17, v22 // (op 4774) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 1370) 
s_and_b32 s1, s18, 4294967294 // (op 1370) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 1369) 
s_mov_b32 s22, s18 // (op 1369) convert

s_cmp_lt_u32 s1, s22 // (op 1369) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_1369 // (op 1369) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_1369 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopTop_KLoopTail_1369:
s_barrier  // (op 5067) 
s_add_u32 s1, s1, 1 // (op 1371) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 4254) 
v_mov_b64 v[18:19], s[18:19] // (op 4254) convert


v_mov_b64 v[22:23], v[20:21] // (op 4254) 
v_mov_b64 v[30:31], v[22:23] // (op 4254) 
s_add_u32 s18, 8192, s47 // (op 4254) 
s_mov_b32 m0, s18 // (op 4254) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4254) Load value direct to lds
v_add_co_u32 v30, s[22:23], v30, v18 // (op 4254) least significant half
v_addc_co_u32 v31, s[22:23], 0, v31, s[22:23] // (op 4254) most significant half

s_add_u32 m0, m0, 4096 // (op 4254) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4254) Load value direct to lds
v_add_co_u32 v30, s[22:23], v30, v18 // (op 4254) least significant half
v_addc_co_u32 v31, s[22:23], 0, v31, s[22:23] // (op 4254) most significant half

s_add_u32 m0, m0, 4096 // (op 4254) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4254) Load value direct to lds
v_add_co_u32 v30, s[22:23], v30, v18 // (op 4254) least significant half
v_addc_co_u32 v31, s[22:23], 0, v31, s[22:23] // (op 4254) most significant half

s_add_u32 m0, m0, 4096 // (op 4254) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 4254) Load value direct to lds




v_add_co_u32 v20, vcc, 128, v20 // (op 4292) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 4292) most significant half
s_barrier  // (op 5083) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4260) 
v_mov_b64 v[18:19], s[18:19] // (op 4260) convert


v_mov_b64 v[22:23], v[32:33] // (op 4260) 
v_mov_b64 v[30:31], v[22:23] // (op 4260) 
s_mov_b32 m0, s0 // (op 4260) reg expression
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 4260) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v18 // (op 4260) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 4260) most significant half

s_add_u32 m0, m0, 4096 // (op 4260) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 4260) Load value direct to lds



v_add_co_u32 v32, vcc, 128, v32 // (op 4368) least significant half
v_addc_co_u32 v33, vcc, 0, v33, vcc // (op 4368) most significant half

v_mov_b64 v[18:19], s[34:35] // (op 195) convert

v_mov_b32 v14, v26 // (op 195) 
v_mov_b32 v17, v14 // (op 195) 
buffer_load_ubyte v22, v17, s[64:67], 0 offen offset: 0 // (op 195) Load value
v_add_u32 v17, v17, v18 // (op 195) 
buffer_load_ubyte v23, v17, s[64:67], 0 offen offset: 0 // (op 195) Load value



v_add_u32 v26, 4, v26 // (op 4432) 


s_barrier  // (op 5100) 
v_mov_b32 v14, v15 // (op 196) 




buffer_load_ubyte v25, v14, s[68:71], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v15, 4, v15 // (op 4496) 
s_waitcnt vmcnt(2)
ds_write_b8 v16, v22 offset:49408 // (op 192) Store local data 
s_barrier  // (op 5073) 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v23 offset:49412 // (op 192) Store local data 




ds_read_b128 v[34:37], v3 offset:8192 // (op 183) Load local data 
s_barrier  // (op 5118) 
ds_read_b128 v[38:41], v3 offset:8256 // (op 183) Load local data 



s_barrier  // (op 5090) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v25 offset:49152 // (op 193) Store local data 




ds_read_b128 v[42:45], v4 // (op 185) Load local data 
ds_read_b128 v[46:49], v4 offset:64 // (op 185) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 5108) 



ds_read_u8 v14, v0 offset:49408 // (op 187) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5125) 



ds_read_u8 v25, v5 offset:49152 // (op 189) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:49], a[0:3], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 190) 
ds_read_b128 v[50:53], v4 offset:4096 // (op 354) Load local data 
ds_read_b128 v[54:57], v4 offset:4160 // (op 354) Load local data 



ds_read_u8 v17, v5 offset:49280 // (op 358) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:41], v[50:57], a[4:7], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 359) 
ds_read_b128 v[58:61], v3 offset:12288 // (op 572) Load local data 


ds_read_b128 v[62:65], v3 offset:12352 // (op 572) Load local data 



ds_read_u8 v14, v0 offset:49536 // (op 575) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[42:49], a[8:11], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 577) 
ds_read_b128 v[34:37], v3 offset:16384 // (op 958) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[50:57], a[12:15], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 608) 
ds_read_b128 v[38:41], v3 offset:16448 // (op 958) Load local data 





ds_read_u8 v14, v0 offset:49664 // (op 961) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[34:41], v[42:49], a[16:19], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 963) 
ds_read_b128 v[58:61], v3 offset:20480 // (op 1344) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:41], v[50:57], a[20:23], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 994) 
ds_read_b128 v[62:65], v3 offset:20544 // (op 1344) Load local data 





ds_read_u8 v14, v0 offset:49792 // (op 1347) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:65], v[42:49], a[24:27], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1349) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[50:57], a[28:31], v14, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1380) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 1369) 
s_mov_b32 s22, s18 // (op 1369) convert

s_cmp_lt_u32 s1, s22 // (op 1369) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopTop_KLoopTail_1369 // (op 1369) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopTop_KLoopTail_1369 if true)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_1369:






















s_barrier  // (op 5215) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ConditionalFalse_0_beta_135
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

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 1873) 
v_accvgpr_read v7, a5 // (op 1873) 
v_accvgpr_read v8, a6 // (op 1873) 
v_accvgpr_read v9, a7 // (op 1873) 
v_mul_f32 v14, v11, v6 // (op 1873) 
v_mul_f32 v15, v11, v7 // (op 1873) 
v_mul_f32 v16, v11, v8 // (op 1873) 
v_mul_f32 v17, v11, v9 // (op 1873) 

v_cvt_f16_f32 v6, v14 // (op 1875) 
v_cvt_f16_f32 v7, v15 // (op 1875) 
v_cvt_f16_f32 v8, v16 // (op 1875) 
v_cvt_f16_f32 v9, v17 // (op 1875) 


v_pack_b32_f16 v4, v6, v7 // (op 1876) 
v_pack_b32_f16 v5, v8, v9 // (op 1876) 

s_lshl_b32 s0, s4, 6 // (op 1876) 
v_add_u32 v0, v13, s0 // (op 1876) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1876) Store value



v_accvgpr_read v6, a8 // (op 1899) 
v_accvgpr_read v7, a9 // (op 1899) 
v_accvgpr_read v8, a10 // (op 1899) 
v_accvgpr_read v9, a11 // (op 1899) 
v_mul_f32 v14, v11, v6 // (op 1899) 
v_mul_f32 v15, v11, v7 // (op 1899) 
v_mul_f32 v16, v11, v8 // (op 1899) 
v_mul_f32 v17, v11, v9 // (op 1899) 

v_cvt_f16_f32 v6, v14 // (op 1901) 
v_cvt_f16_f32 v7, v15 // (op 1901) 
v_cvt_f16_f32 v8, v16 // (op 1901) 
v_cvt_f16_f32 v9, v17 // (op 1901) 


v_pack_b32_f16 v4, v6, v7 // (op 1903) 
v_pack_b32_f16 v5, v8, v9 // (op 1903) 

v_add_u32 v0, 64, v13 // (op 1903) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1903) Store value



v_accvgpr_read v6, a12 // (op 1909) 
v_accvgpr_read v7, a13 // (op 1909) 
v_accvgpr_read v8, a14 // (op 1909) 
v_accvgpr_read v9, a15 // (op 1909) 
v_mul_f32 v14, v11, v6 // (op 1909) 
v_mul_f32 v15, v11, v7 // (op 1909) 
v_mul_f32 v16, v11, v8 // (op 1909) 
v_mul_f32 v17, v11, v9 // (op 1909) 

v_cvt_f16_f32 v6, v14 // (op 1911) 
v_cvt_f16_f32 v7, v15 // (op 1911) 
v_cvt_f16_f32 v8, v16 // (op 1911) 
v_cvt_f16_f32 v9, v17 // (op 1911) 


v_pack_b32_f16 v4, v6, v7 // (op 1905) 
v_pack_b32_f16 v5, v8, v9 // (op 1905) 

s_lshl_b32 s0, s4, 6 // (op 1905) 
s_add_u32 s0, s0, 64 // (op 1905) 
v_add_u32 v0, v13, s0 // (op 1905) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1905) Store value



v_accvgpr_read v6, a16 // (op 1939) 
v_accvgpr_read v7, a17 // (op 1939) 
v_accvgpr_read v8, a18 // (op 1939) 
v_accvgpr_read v9, a19 // (op 1939) 
v_mul_f32 v14, v11, v6 // (op 1939) 
v_mul_f32 v15, v11, v7 // (op 1939) 
v_mul_f32 v16, v11, v8 // (op 1939) 
v_mul_f32 v17, v11, v9 // (op 1939) 

v_cvt_f16_f32 v6, v14 // (op 1941) 
v_cvt_f16_f32 v7, v15 // (op 1941) 
v_cvt_f16_f32 v8, v16 // (op 1941) 
v_cvt_f16_f32 v9, v17 // (op 1941) 


v_pack_b32_f16 v4, v6, v7 // (op 1943) 
v_pack_b32_f16 v5, v8, v9 // (op 1943) 

v_add_u32 v0, 128, v13 // (op 1943) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1943) Store value



v_accvgpr_read v6, a20 // (op 1949) 
v_accvgpr_read v7, a21 // (op 1949) 
v_accvgpr_read v8, a22 // (op 1949) 
v_accvgpr_read v9, a23 // (op 1949) 
v_mul_f32 v14, v11, v6 // (op 1949) 
v_mul_f32 v15, v11, v7 // (op 1949) 
v_mul_f32 v16, v11, v8 // (op 1949) 
v_mul_f32 v17, v11, v9 // (op 1949) 

v_cvt_f16_f32 v6, v14 // (op 1951) 
v_cvt_f16_f32 v7, v15 // (op 1951) 
v_cvt_f16_f32 v8, v16 // (op 1951) 
v_cvt_f16_f32 v9, v17 // (op 1951) 


v_pack_b32_f16 v4, v6, v7 // (op 1945) 
v_pack_b32_f16 v5, v8, v9 // (op 1945) 

s_lshl_b32 s0, s4, 6 // (op 1945) 
s_add_u32 s0, s0, 128 // (op 1945) 
v_add_u32 v0, v13, s0 // (op 1945) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1945) Store value



v_accvgpr_read v6, a24 // (op 1979) 
v_accvgpr_read v7, a25 // (op 1979) 
v_accvgpr_read v8, a26 // (op 1979) 
v_accvgpr_read v9, a27 // (op 1979) 
v_mul_f32 v14, v11, v6 // (op 1979) 
v_mul_f32 v15, v11, v7 // (op 1979) 
v_mul_f32 v16, v11, v8 // (op 1979) 
v_mul_f32 v17, v11, v9 // (op 1979) 

v_cvt_f16_f32 v6, v14 // (op 1981) 
v_cvt_f16_f32 v7, v15 // (op 1981) 
v_cvt_f16_f32 v8, v16 // (op 1981) 
v_cvt_f16_f32 v9, v17 // (op 1981) 


v_pack_b32_f16 v4, v6, v7 // (op 1983) 
v_pack_b32_f16 v5, v8, v9 // (op 1983) 

v_add_u32 v0, 192, v13 // (op 1983) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1983) Store value



v_accvgpr_read v6, a28 // (op 1989) 
v_accvgpr_read v7, a29 // (op 1989) 
v_accvgpr_read v8, a30 // (op 1989) 
v_accvgpr_read v9, a31 // (op 1989) 
v_mul_f32 v14, v11, v6 // (op 1989) 
v_mul_f32 v15, v11, v7 // (op 1989) 
v_mul_f32 v16, v11, v8 // (op 1989) 
v_mul_f32 v17, v11, v9 // (op 1989) 

v_cvt_f16_f32 v6, v14 // (op 1991) 
v_cvt_f16_f32 v7, v15 // (op 1991) 
v_cvt_f16_f32 v8, v16 // (op 1991) 
v_cvt_f16_f32 v9, v17 // (op 1991) 


v_pack_b32_f16 v4, v6, v7 // (op 1985) 
v_pack_b32_f16 v5, v8, v9 // (op 1985) 

s_lshl_b32 s0, s4, 6 // (op 1985) 
s_add_u32 s0, s0, 192 // (op 1985) 
v_add_u32 v0, v13, s0 // (op 1985) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 1985) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ConditionalBottom_0_beta_135
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ConditionalFalse_0_beta_135:


v_add_u32 v0, 128, v12 // (op 2136) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2136) Load value

v_accvgpr_read v4, a16 // (op 2127) 
v_accvgpr_read v5, a17 // (op 2127) 
v_accvgpr_read v6, a18 // (op 2127) 
v_accvgpr_read v7, a19 // (op 2127) 
v_mul_f32 v14, v11, v4 // (op 2127) 
v_mul_f32 v15, v11, v5 // (op 2127) 
v_mul_f32 v16, v11, v6 // (op 2127) 
v_mul_f32 v17, v11, v7 // (op 2127) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2135) 
v_lshrrev_b32 v7, 16, v8 // (op 2135) 
v_cvt_f32_f16 v7, v7 // (op 2135) 
v_mul_f32 v18, v10, v6 // (op 2135) 
v_mul_f32 v19, v10, v7 // (op 2135) 
v_cvt_f32_f16 v6, v9 // (op 2135) 
v_lshrrev_b32 v7, 16, v9 // (op 2135) 
v_cvt_f32_f16 v7, v7 // (op 2135) 
v_mul_f32 v20, v10, v6 // (op 2135) 
v_mul_f32 v21, v10, v7 // (op 2135) 


v_add_f32 v6, v18, v14 // (op 2128) 
v_add_f32 v7, v19, v15 // (op 2128) 
v_add_f32 v8, v20, v16 // (op 2128) 
v_add_f32 v9, v21, v17 // (op 2128) 


v_cvt_f16_f32 v14, v6 // (op 2129) 
v_cvt_f16_f32 v15, v7 // (op 2129) 
v_cvt_f16_f32 v16, v8 // (op 2129) 
v_cvt_f16_f32 v17, v9 // (op 2129) 


v_pack_b32_f16 v8, v14, v15 // (op 2131) 
v_pack_b32_f16 v9, v16, v17 // (op 2131) 

v_add_u32 v0, 128, v13 // (op 2131) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2131) Store value





s_lshl_b32 s0, s2, 6 // (op 2141) 
s_add_u32 s0, s0, 128 // (op 2141) 
v_add_u32 v0, v12, s0 // (op 2141) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2141) Load value

v_accvgpr_read v4, a20 // (op 2143) 
v_accvgpr_read v5, a21 // (op 2143) 
v_accvgpr_read v6, a22 // (op 2143) 
v_accvgpr_read v7, a23 // (op 2143) 
v_mul_f32 v14, v11, v4 // (op 2143) 
v_mul_f32 v15, v11, v5 // (op 2143) 
v_mul_f32 v16, v11, v6 // (op 2143) 
v_mul_f32 v17, v11, v7 // (op 2143) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2138) 
v_lshrrev_b32 v7, 16, v8 // (op 2138) 
v_cvt_f32_f16 v7, v7 // (op 2138) 
v_mul_f32 v18, v10, v6 // (op 2138) 
v_mul_f32 v19, v10, v7 // (op 2138) 
v_cvt_f32_f16 v6, v9 // (op 2138) 
v_lshrrev_b32 v7, 16, v9 // (op 2138) 
v_cvt_f32_f16 v7, v7 // (op 2138) 
v_mul_f32 v20, v10, v6 // (op 2138) 
v_mul_f32 v21, v10, v7 // (op 2138) 


v_add_f32 v6, v18, v14 // (op 2139) 
v_add_f32 v7, v19, v15 // (op 2139) 
v_add_f32 v8, v20, v16 // (op 2139) 
v_add_f32 v9, v21, v17 // (op 2139) 


v_cvt_f16_f32 v14, v6 // (op 2140) 
v_cvt_f16_f32 v15, v7 // (op 2140) 
v_cvt_f16_f32 v16, v8 // (op 2140) 
v_cvt_f16_f32 v17, v9 // (op 2140) 


v_pack_b32_f16 v8, v14, v15 // (op 2133) 
v_pack_b32_f16 v9, v16, v17 // (op 2133) 

s_lshl_b32 s0, s4, 6 // (op 2133) 
s_add_u32 s0, s0, 128 // (op 2133) 
v_add_u32 v0, v13, s0 // (op 2133) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2133) Store value





v_add_u32 v0, 64, v12 // (op 2091) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2091) Load value

v_accvgpr_read v4, a8 // (op 2082) 
v_accvgpr_read v5, a9 // (op 2082) 
v_accvgpr_read v6, a10 // (op 2082) 
v_accvgpr_read v7, a11 // (op 2082) 
v_mul_f32 v14, v11, v4 // (op 2082) 
v_mul_f32 v15, v11, v5 // (op 2082) 
v_mul_f32 v16, v11, v6 // (op 2082) 
v_mul_f32 v17, v11, v7 // (op 2082) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2090) 
v_lshrrev_b32 v7, 16, v8 // (op 2090) 
v_cvt_f32_f16 v7, v7 // (op 2090) 
v_mul_f32 v18, v10, v6 // (op 2090) 
v_mul_f32 v19, v10, v7 // (op 2090) 
v_cvt_f32_f16 v6, v9 // (op 2090) 
v_lshrrev_b32 v7, 16, v9 // (op 2090) 
v_cvt_f32_f16 v7, v7 // (op 2090) 
v_mul_f32 v20, v10, v6 // (op 2090) 
v_mul_f32 v21, v10, v7 // (op 2090) 


v_add_f32 v6, v18, v14 // (op 2083) 
v_add_f32 v7, v19, v15 // (op 2083) 
v_add_f32 v8, v20, v16 // (op 2083) 
v_add_f32 v9, v21, v17 // (op 2083) 


v_cvt_f16_f32 v14, v6 // (op 2084) 
v_cvt_f16_f32 v15, v7 // (op 2084) 
v_cvt_f16_f32 v16, v8 // (op 2084) 
v_cvt_f16_f32 v17, v9 // (op 2084) 


v_pack_b32_f16 v8, v14, v15 // (op 2086) 
v_pack_b32_f16 v9, v16, v17 // (op 2086) 

v_add_u32 v0, 64, v13 // (op 2086) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2086) Store value





s_lshl_b32 s0, s2, 6 // (op 2096) 
s_add_u32 s0, s0, 64 // (op 2096) 
v_add_u32 v0, v12, s0 // (op 2096) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2096) Load value

v_accvgpr_read v4, a12 // (op 2098) 
v_accvgpr_read v5, a13 // (op 2098) 
v_accvgpr_read v6, a14 // (op 2098) 
v_accvgpr_read v7, a15 // (op 2098) 
v_mul_f32 v14, v11, v4 // (op 2098) 
v_mul_f32 v15, v11, v5 // (op 2098) 
v_mul_f32 v16, v11, v6 // (op 2098) 
v_mul_f32 v17, v11, v7 // (op 2098) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2093) 
v_lshrrev_b32 v7, 16, v8 // (op 2093) 
v_cvt_f32_f16 v7, v7 // (op 2093) 
v_mul_f32 v18, v10, v6 // (op 2093) 
v_mul_f32 v19, v10, v7 // (op 2093) 
v_cvt_f32_f16 v6, v9 // (op 2093) 
v_lshrrev_b32 v7, 16, v9 // (op 2093) 
v_cvt_f32_f16 v7, v7 // (op 2093) 
v_mul_f32 v20, v10, v6 // (op 2093) 
v_mul_f32 v21, v10, v7 // (op 2093) 


v_add_f32 v6, v18, v14 // (op 2094) 
v_add_f32 v7, v19, v15 // (op 2094) 
v_add_f32 v8, v20, v16 // (op 2094) 
v_add_f32 v9, v21, v17 // (op 2094) 


v_cvt_f16_f32 v14, v6 // (op 2095) 
v_cvt_f16_f32 v15, v7 // (op 2095) 
v_cvt_f16_f32 v16, v8 // (op 2095) 
v_cvt_f16_f32 v17, v9 // (op 2095) 


v_pack_b32_f16 v8, v14, v15 // (op 2088) 
v_pack_b32_f16 v9, v16, v17 // (op 2088) 

s_lshl_b32 s0, s4, 6 // (op 2088) 
s_add_u32 s0, s0, 64 // (op 2088) 
v_add_u32 v0, v13, s0 // (op 2088) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2088) Store value





v_add_u32 v0, 192, v12 // (op 2181) 

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2181) Load value

v_accvgpr_read v4, a24 // (op 2172) 
v_accvgpr_read v5, a25 // (op 2172) 
v_accvgpr_read v6, a26 // (op 2172) 
v_accvgpr_read v7, a27 // (op 2172) 
v_mul_f32 v14, v11, v4 // (op 2172) 
v_mul_f32 v15, v11, v5 // (op 2172) 
v_mul_f32 v16, v11, v6 // (op 2172) 
v_mul_f32 v17, v11, v7 // (op 2172) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2180) 
v_lshrrev_b32 v7, 16, v8 // (op 2180) 
v_cvt_f32_f16 v7, v7 // (op 2180) 
v_mul_f32 v18, v10, v6 // (op 2180) 
v_mul_f32 v19, v10, v7 // (op 2180) 
v_cvt_f32_f16 v6, v9 // (op 2180) 
v_lshrrev_b32 v7, 16, v9 // (op 2180) 
v_cvt_f32_f16 v7, v7 // (op 2180) 
v_mul_f32 v20, v10, v6 // (op 2180) 
v_mul_f32 v21, v10, v7 // (op 2180) 


v_add_f32 v6, v18, v14 // (op 2173) 
v_add_f32 v7, v19, v15 // (op 2173) 
v_add_f32 v8, v20, v16 // (op 2173) 
v_add_f32 v9, v21, v17 // (op 2173) 


v_cvt_f16_f32 v14, v6 // (op 2174) 
v_cvt_f16_f32 v15, v7 // (op 2174) 
v_cvt_f16_f32 v16, v8 // (op 2174) 
v_cvt_f16_f32 v17, v9 // (op 2174) 


v_pack_b32_f16 v8, v14, v15 // (op 2176) 
v_pack_b32_f16 v9, v16, v17 // (op 2176) 

v_add_u32 v0, 192, v13 // (op 2176) 

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2176) Store value





s_lshl_b32 s0, s2, 6 // (op 2186) 
s_add_u32 s0, s0, 192 // (op 2186) 
v_add_u32 v0, v12, s0 // (op 2186) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2186) Load value

v_accvgpr_read v4, a28 // (op 2188) 
v_accvgpr_read v5, a29 // (op 2188) 
v_accvgpr_read v6, a30 // (op 2188) 
v_accvgpr_read v7, a31 // (op 2188) 
v_mul_f32 v14, v11, v4 // (op 2188) 
v_mul_f32 v15, v11, v5 // (op 2188) 
v_mul_f32 v16, v11, v6 // (op 2188) 
v_mul_f32 v17, v11, v7 // (op 2188) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2183) 
v_lshrrev_b32 v7, 16, v8 // (op 2183) 
v_cvt_f32_f16 v7, v7 // (op 2183) 
v_mul_f32 v18, v10, v6 // (op 2183) 
v_mul_f32 v19, v10, v7 // (op 2183) 
v_cvt_f32_f16 v6, v9 // (op 2183) 
v_lshrrev_b32 v7, 16, v9 // (op 2183) 
v_cvt_f32_f16 v7, v7 // (op 2183) 
v_mul_f32 v20, v10, v6 // (op 2183) 
v_mul_f32 v21, v10, v7 // (op 2183) 


v_add_f32 v6, v18, v14 // (op 2184) 
v_add_f32 v7, v19, v15 // (op 2184) 
v_add_f32 v8, v20, v16 // (op 2184) 
v_add_f32 v9, v21, v17 // (op 2184) 


v_cvt_f16_f32 v14, v6 // (op 2185) 
v_cvt_f16_f32 v15, v7 // (op 2185) 
v_cvt_f16_f32 v16, v8 // (op 2185) 
v_cvt_f16_f32 v17, v9 // (op 2185) 


v_pack_b32_f16 v8, v14, v15 // (op 2178) 
v_pack_b32_f16 v9, v16, v17 // (op 2178) 

s_lshl_b32 s0, s4, 6 // (op 2178) 
s_add_u32 s0, s0, 192 // (op 2178) 
v_add_u32 v0, v13, s0 // (op 2178) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2178) Store value





v_mov_b32 v0, v12 // (op 148) reg expression

buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 148) Load value

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

buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 2051) 
v_add_u32 v0, v12, s0 // (op 2051) 


buffer_load_dwordx2 v[8:9], v0, s[52:55], 0 offen // (op 2051) Load value

v_accvgpr_read v4, a4 // (op 2047) 
v_accvgpr_read v5, a5 // (op 2047) 
v_accvgpr_read v6, a6 // (op 2047) 
v_accvgpr_read v7, a7 // (op 2047) 
v_mul_f32 v14, v11, v4 // (op 2047) 
v_mul_f32 v15, v11, v5 // (op 2047) 
v_mul_f32 v16, v11, v6 // (op 2047) 
v_mul_f32 v17, v11, v7 // (op 2047) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2052) 
v_lshrrev_b32 v7, 16, v8 // (op 2052) 
v_cvt_f32_f16 v7, v7 // (op 2052) 
v_mul_f32 v18, v10, v6 // (op 2052) 
v_mul_f32 v19, v10, v7 // (op 2052) 
v_cvt_f32_f16 v6, v9 // (op 2052) 
v_lshrrev_b32 v7, 16, v9 // (op 2052) 
v_cvt_f32_f16 v7, v7 // (op 2052) 
v_mul_f32 v20, v10, v6 // (op 2052) 
v_mul_f32 v21, v10, v7 // (op 2052) 


v_add_f32 v6, v18, v14 // (op 2048) 
v_add_f32 v7, v19, v15 // (op 2048) 
v_add_f32 v8, v20, v16 // (op 2048) 
v_add_f32 v9, v21, v17 // (op 2048) 


v_cvt_f16_f32 v14, v6 // (op 2049) 
v_cvt_f16_f32 v15, v7 // (op 2049) 
v_cvt_f16_f32 v16, v8 // (op 2049) 
v_cvt_f16_f32 v17, v9 // (op 2049) 


v_pack_b32_f16 v8, v14, v15 // (op 2050) 
v_pack_b32_f16 v9, v16, v17 // (op 2050) 

s_lshl_b32 s0, s4, 6 // (op 2050) 
v_add_u32 v0, v13, s0 // (op 2050) 


buffer_store_dwordx2 v[8:9], v0, s[56:59], 0 offen // (op 2050) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 144
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 50688 // lds bytes
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
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
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
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int64, semantic: SizeFree0 },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [128, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 50688
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 112
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 24
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 32
        .expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 36
        .expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 44
        .expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 48
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 128
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 144
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 152
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 164
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 168
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

