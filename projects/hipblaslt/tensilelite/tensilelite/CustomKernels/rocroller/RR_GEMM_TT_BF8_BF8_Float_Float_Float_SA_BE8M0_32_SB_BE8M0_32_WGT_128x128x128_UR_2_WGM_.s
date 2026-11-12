// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_,@function
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s11 // (op 8857) 
s_mul_hi_i32 s1, s16, s11 // (op 9112) 
v_lshrrev_b32 v0, 7, v1 // (op 8918) 
v_and_b32 v3, 15, v1 // (op 8918) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 8918) 


v_and_b32 v0, 48, v1 // (op 8918) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 8918) 


v_and_b32 v0, 48, v1 // (op 8975) 
v_and_b32 v4, 63, v1 // (op 8975) 
v_sub_u32 v5, v4, v0 // (op 8975) 

v_lshrrev_b32 v4, 1, v5 // (op 8975) 
v_add_lshl_u32 v6, v0, v4, 7 // (op 8975) 


v_and_b32 v0, 1, v5 // (op 8975) 

v_lshlrev_b32 v4, 3, v0 // (op 8975) 

v_lshrrev_b32 v0, 6, v1 // (op 8975) 
v_and_b32 v5, 1, v0 // (op 8975) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 8975) 


v_add_u32 v4, v6, v0 // (op 8975) 


v_lshrrev_b32 v0, 7, v1 // (op 9033) 
v_and_b32 v5, 15, v1 // (op 9033) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9033) 


v_and_b32 v0, 63, v1 // (op 9033) 
v_lshrrev_b32 v5, 4, v0 // (op 9033) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 9033) 


v_lshrrev_b32 v5, 6, v1 // (op 9091) 
v_and_b32 v6, 1, v5 // (op 9091) 

v_and_b32 v5, 15, v1 // (op 9091) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9091) 


v_and_b32 v5, 63, v1 // (op 9091) 
v_lshrrev_b32 v6, 4, v5 // (op 9091) 

v_lshl_add_u32 v5, v6, 7, v7 // (op 9091) 


v_lshrrev_b32 v6, 7, v1 // (op 9158) 
v_and_b32 v7, 15, v1 // (op 9158) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9158) 


v_and_b32 v6, 48, v1 // (op 9158) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9158) 


v_and_b32 v6, 48, v1 // (op 9202) 
v_and_b32 v8, 63, v1 // (op 9202) 
v_sub_u32 v9, v8, v6 // (op 9202) 

v_lshrrev_b32 v8, 1, v9 // (op 9202) 
v_add_lshl_u32 v10, v6, v8, 7 // (op 9202) 


v_and_b32 v6, 1, v9 // (op 9202) 

v_lshlrev_b32 v8, 3, v6 // (op 9202) 

v_lshrrev_b32 v6, 6, v1 // (op 9202) 
v_and_b32 v9, 1, v6 // (op 9202) 

v_lshl_add_u32 v6, v9, 4, v8 // (op 9202) 


v_add_u32 v8, v10, v6 // (op 9202) 


v_lshrrev_b32 v6, 7, v1 // (op 9231) 
v_and_b32 v9, 15, v1 // (op 9231) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9231) 


v_and_b32 v6, 63, v1 // (op 9231) 
v_lshrrev_b32 v9, 4, v6 // (op 9231) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 9231) 


v_lshrrev_b32 v9, 6, v1 // (op 9260) 
v_and_b32 v10, 1, v9 // (op 9260) 

v_and_b32 v9, 15, v1 // (op 9260) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9260) 


v_and_b32 v9, 63, v1 // (op 9260) 
v_lshrrev_b32 v10, 4, v9 // (op 9260) 

v_lshl_add_u32 v9, v10, 7, v11 // (op 9260) 


v_lshrrev_b32 v10, 7, v1 // (op 8857) 
s_add_i32 s18, s0, s16 // (op 8857) 

v_and_b32 v11, 63, v1 // (op 8857) 
s_ashr_i32 s0, s18, 31 // (op 8857) 
v_lshrrev_b32 v12, 4, v11 // (op 8857) 
s_cmp_eq_i32 s11, 0 // (op 8857) 
s_mov_b32 s19, scc // (op 8857) 
v_lshlrev_b32 v13, 2, v12 // (op 8857) 

s_cmp_eq_i32 s19, 0 // (op 8857) 
s_cselect_b32 s51, 0, -1 // (op 8857) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8857) 


s_and_b32 s19, s12, 31 // (op 8857) 
v_lshrrev_b32 v10, 2, v11 // (op 8857) 

v_and_b32 v11, 3, v10 // (op 8857) 

v_lshrrev_b32 v10, 7, v1 // (op 9112) 
v_and_b32 v13, 63, v1 // (op 9112) 
s_lshl_b32 s52, 1, s19 // (op 8857) 
v_and_b32 v14, 3, v1 // (op 8857) 
s_add_i32 s53, s52, s51 // (op 8857) 


s_and_b32 s51, s0, s53 // (op 8857) 


s_add_i32 s0, s1, s16 // (op 9112) 

s_ashr_i32 s1, s0, 31 // (op 9112) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 8857) 


s_add_i32 s52, s18, s51 // (op 8857) 


v_lshrrev_b32 v11, 6, v1 // (op 8857) 
s_ashr_i32 s18, s52, s19 // (op 8857) 


v_and_b32 v14, 1, v11 // (op 8857) 

s_lshl_b32 s19, s12, 24 // (op 8857) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 8857) 


s_ashr_i32 s51, s19, 31 // (op 8857) 

s_xor_b32 s19, s18, s51 // (op 8857) 

s_sub_i32 s18, s19, s51 // (op 8857) 


s_mul_hi_i32 s19, s18, s13 // (op 8857) 
s_add_i32 s51, s19, s18 // (op 8857) 

s_ashr_i32 s19, s51, 31 // (op 8857) 
s_cmp_eq_i32 s13, 0 // (op 8857) 
s_mov_b32 s52, scc // (op 8857) 
s_cmp_eq_i32 s52, 0 // (op 8857) 
s_cselect_b32 s53, 0, -1 // (op 8857) 

s_and_b32 s52, s14, 31 // (op 8857) 
v_lshrrev_b32 v14, 4, v13 // (op 9112) 
s_cmp_eq_i32 s11, 0 // (op 9112) 
s_mov_b32 s54, scc // (op 9112) 
v_lshlrev_b32 v15, 2, v14 // (op 9112) 

s_cmp_eq_i32 s54, 0 // (op 9112) 
s_cselect_b32 s55, 0, -1 // (op 9112) 

s_lshl_b32 s54, 1, s52 // (op 8857) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 9112) 


v_lshrrev_b32 v10, 2, v13 // (op 9112) 

s_add_i32 s56, s54, s53 // (op 8857) 


s_and_b32 s53, s19, s56 // (op 8857) 


s_add_i32 s19, s51, s53 // (op 8857) 


s_ashr_i32 s51, s19, s52 // (op 8857) 


s_lshl_b32 s19, s14, 24 // (op 8857) 
s_ashr_i32 s52, s19, 31 // (op 8857) 

s_xor_b32 s19, s51, s52 // (op 8857) 

s_sub_i32 s51, s19, s52 // (op 8857) 


s_and_b32 s19, s12, 31 // (op 9112) 
s_lshl_b32 s52, 1, s19 // (op 9112) 
v_and_b32 v13, 3, v10 // (op 9112) 

s_add_i32 s53, s52, s55 // (op 9112) 


v_and_b32 v10, 3, v1 // (op 9112) 
s_and_b32 s52, s1, s53 // (op 9112) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 9112) 


s_add_i32 s1, s0, s52 // (op 9112) 


v_lshrrev_b32 v10, 6, v1 // (op 9112) 
s_ashr_i32 s0, s1, s19 // (op 9112) 


v_and_b32 v13, 1, v10 // (op 9112) 

s_lshl_b32 s1, s12, 24 // (op 9112) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 9112) 


s_ashr_i32 s19, s1, 31 // (op 9112) 

s_xor_b32 s1, s0, s19 // (op 9112) 

s_sub_i32 s0, s1, s19 // (op 9112) 


s_mul_hi_i32 s1, s0, s13 // (op 9112) 
s_add_i32 s19, s1, s0 // (op 9112) 

s_ashr_i32 s1, s19, 31 // (op 9112) 
s_cmp_eq_i32 s13, 0 // (op 9112) 
s_mov_b32 s52, scc // (op 9112) 
s_cmp_eq_i32 s52, 0 // (op 9112) 
s_cselect_b32 s53, 0, -1 // (op 9112) 

s_and_b32 s52, s14, 31 // (op 9112) 
s_lshl_b32 s54, 1, s52 // (op 9112) 
s_add_i32 s55, s54, s53 // (op 9112) 


s_and_b32 s53, s1, s55 // (op 9112) 


s_add_i32 s1, s19, s53 // (op 9112) 


s_ashr_i32 s19, s1, s52 // (op 9112) 


s_lshl_b32 s1, s14, 24 // (op 9112) 
s_ashr_i32 s52, s1, 31 // (op 9112) 

s_xor_b32 s1, s19, s52 // (op 9112) 

s_sub_i32 s19, s1, s52 // (op 9112) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 8857) 
s_sub_u32 s52, s16, s1 // (op 8857) 

s_mul_i32 s1, s51, s44 // (op 8857) 
s_add_u32 s1, s1, s52 // (op 8857) 

s_mul_hi_i32 s52, s16, s10 // (op 8857) 
s_add_i32 s53, s52, s16 // (op 8857) 

s_ashr_i32 s52, s53, 31 // (op 8857) 
s_cmp_eq_i32 s10, 0 // (op 8857) 
s_mov_b32 s54, scc // (op 8857) 
s_cmp_eq_i32 s54, 0 // (op 8857) 
s_cselect_b32 s55, 0, -1 // (op 8857) 

s_and_b32 s54, s47, 31 // (op 8857) 
s_lshl_b32 s56, 1, s54 // (op 8857) 
s_add_i32 s57, s56, s55 // (op 8857) 


s_and_b32 s55, s52, s57 // (op 8857) 


s_add_i32 s52, s53, s55 // (op 8857) 


s_ashr_i32 s53, s52, s54 // (op 8857) 


s_lshl_b32 s52, s47, 24 // (op 8857) 
s_ashr_i32 s54, s52, 31 // (op 8857) 

s_xor_b32 s52, s53, s54 // (op 8857) 

s_sub_i32 s53, s52, s54 // (op 8857) 


s_mul_i32 s52, s0, s44 // (op 9112) 
s_sub_u32 s54, s16, s52 // (op 9112) 

s_mul_i32 s52, s19, s44 // (op 9112) 
s_add_u32 s52, s52, s54 // (op 9112) 

s_mul_hi_i32 s54, s16, s10 // (op 9112) 
s_add_i32 s55, s54, s16 // (op 9112) 

s_ashr_i32 s54, s55, 31 // (op 9112) 
s_cmp_eq_i32 s10, 0 // (op 9112) 
s_mov_b32 s56, scc // (op 9112) 
s_cmp_eq_i32 s56, 0 // (op 9112) 
s_cselect_b32 s57, 0, -1 // (op 9112) 

s_and_b32 s56, s47, 31 // (op 9112) 
s_lshl_b32 s58, 1, s56 // (op 9112) 
s_add_i32 s59, s58, s57 // (op 9112) 


s_and_b32 s57, s54, s59 // (op 9112) 


s_add_i32 s54, s55, s57 // (op 9112) 


s_ashr_i32 s55, s54, s56 // (op 9112) 


s_lshl_b32 s54, s47, 24 // (op 9112) 
s_ashr_i32 s56, s54, 31 // (op 9112) 

s_xor_b32 s54, s55, s56 // (op 9112) 

s_sub_i32 s55, s54, s56 // (op 9112) 


s_add_u32 s56, s6, 128 // (op 8857) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 8857) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 8857) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 8857) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 8857) 

s_mul_i32 s54, s44, s56 // (op 8857) 
s_mul_i32 s58, s48, s54 // (op 8857) 

s_mul_i32 s54, s53, s58 // (op 8857) 

s_sub_u32 s58, s16, s54 // (op 8857) 

s_mov_b32 s54, s58 // (op 8857) 
s_mul_hi_i32 s59, s54, s15 // (op 8857) 
s_add_i32 s60, s59, s54 // (op 8857) 


s_ashr_i32 s54, s60, 31 // (op 8857) 
s_cmp_eq_i32 s15, 0 // (op 8857) 
s_mov_b32 s59, scc // (op 8857) 
s_cmp_eq_i32 s59, 0 // (op 8857) 
s_cselect_b32 s61, 0, -1 // (op 8857) 

s_and_b32 s59, s46, 31 // (op 8857) 
s_lshl_b32 s62, 1, s59 // (op 8857) 
s_add_i32 s63, s62, s61 // (op 8857) 


s_and_b32 s61, s54, s63 // (op 8857) 


s_add_i32 s54, s60, s61 // (op 8857) 


s_ashr_i32 s60, s54, s59 // (op 8857) 


s_lshl_b32 s54, s46, 24 // (op 8857) 
s_ashr_i32 s59, s54, 31 // (op 8857) 

s_xor_b32 s54, s60, s59 // (op 8857) 

s_sub_i32 s60, s54, s59 // (op 8857) 


s_mul_i32 s54, s45, s44 // (op 8857) 
s_add_u32 s62, s8, 128 // (op 8857) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 8857) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 8857) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 8857) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 7 // (op 8857) 

s_sub_i32 s59, s62, s54 // (op 8857) 

s_mul_i32 s61, s60, s59 // (op 8857) 

s_sub_u32 s59, s58, s61 // (op 8857) 


s_add_u32 s58, s59, s54 // (op 8857) 


s_cmp_eq_i32 s53, 0 // (op 8857) 
s_mov_b32 s54, scc // (op 8857) 

s_cmp_eq_i32 s54, 0 // (op 8857) 
s_cselect_b32 s53, s58, s1 // (op 8857) 


s_mul_i32 s1, s51, s56 // (op 8857) 


v_lshl_add_u32 v13, s53, 7, v12 // (op 8857) 


s_sub_i32 s51, s18, s1 // (op 8857) 


s_cmp_eq_i32 s54, 0 // (op 8857) 
s_cselect_b32 s1, s60, s51 // (op 8857) 



v_lshl_add_u32 v12, s1, 7, v11 // (op 8857) 


v_mul_lo_u32 v11, v12, s2 // (op 8857) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 8857) 


s_add_u32 s56, s6, 128 // (op 9112) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 9112) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 9112) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 9112) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 9112) 

s_mul_i32 s1, s44, s56 // (op 9112) 
s_mul_i32 s18, s48, s1 // (op 9112) 

s_mul_i32 s1, s55, s18 // (op 9112) 

s_sub_u32 s18, s16, s1 // (op 9112) 

s_mov_b32 s1, s18 // (op 9112) 
s_mul_hi_i32 s51, s1, s15 // (op 9112) 
s_add_i32 s53, s51, s1 // (op 9112) 


s_ashr_i32 s1, s53, 31 // (op 9112) 
s_cmp_eq_i32 s15, 0 // (op 9112) 
s_mov_b32 s51, scc // (op 9112) 
s_cmp_eq_i32 s51, 0 // (op 9112) 
s_cselect_b32 s54, 0, -1 // (op 9112) 

s_and_b32 s51, s46, 31 // (op 9112) 
s_lshl_b32 s58, 1, s51 // (op 9112) 
s_add_i32 s59, s58, s54 // (op 9112) 


s_and_b32 s54, s1, s59 // (op 9112) 


s_add_i32 s1, s53, s54 // (op 9112) 


s_ashr_i32 s53, s1, s51 // (op 9112) 


s_lshl_b32 s1, s46, 24 // (op 9112) 
s_ashr_i32 s51, s1, 31 // (op 9112) 

s_xor_b32 s1, s53, s51 // (op 9112) 

s_sub_i32 s53, s1, s51 // (op 9112) 


s_mul_i32 s1, s45, s44 // (op 9112) 
s_add_u32 s58, s8, 128 // (op 9112) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 9112) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 9112) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 9112) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 7 // (op 9112) 

s_sub_i32 s51, s58, s1 // (op 9112) 

s_mul_i32 s54, s53, s51 // (op 9112) 

s_sub_u32 s51, s18, s54 // (op 9112) 


s_add_u32 s18, s51, s1 // (op 9112) 


s_cmp_eq_i32 s55, 0 // (op 9112) 
s_mov_b32 s1, scc // (op 9112) 

s_cmp_eq_i32 s1, 0 // (op 9112) 
s_cselect_b32 s51, s18, s52 // (op 9112) 


s_mul_i32 s18, s19, s56 // (op 9112) 


v_lshl_add_u32 v11, s51, 7, v14 // (op 9112) 


s_sub_i32 s19, s0, s18 // (op 9112) 


s_cmp_eq_i32 s1, 0 // (op 9112) 
s_cselect_b32 s0, s53, s19 // (op 9112) 



v_lshl_add_u32 v13, s0, 7, v10 // (op 9112) 


v_mul_lo_u32 v10, v13, s4 // (op 9112) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 9112) 


s_mul_i32 s1, s2, s7 // (op 8859) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 8859) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 8859) least significant
s_mul_hi_u32 s19, s2, s6 // (op 8859) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8859) most significant: sum
s_add_u32 s1, s1, s19 // (op 8859) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8859) 

s_mul_i32 s1, s4, s7 // (op 9114) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 9114) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 9114) least significant
s_mul_hi_u32 s52, s4, s6 // (op 9114) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 9114) most significant: sum
s_add_u32 s1, s1, s52 // (op 9114) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 9114) 

s_mov_b32 s56, s20 // (op 8859) reg expression
s_mov_b32 s57, s21 // (op 8859) reg expression
s_mov_b32 s58, s18 // (op 8859) reg expression
s_mov_b32 s59, 131072 // (op 8859) call()

s_mov_b32 s60, s22 // (op 9114) reg expression

s_mov_b32 s61, s23 // (op 9114) reg expression
s_mov_b32 s62, s52 // (op 9114) reg expression
s_mov_b32 s63, 131072 // (op 9114) call()


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
v_accvgpr_write a16, 0.00000 // (op 952) call()
v_accvgpr_write a17, 0.00000 // (op 952) call()
v_accvgpr_write a18, 0.00000 // (op 952) call()
v_accvgpr_write a19, 0.00000 // (op 952) call()
v_accvgpr_write a20, 0.00000 // (op 1261) call()
v_accvgpr_write a21, 0.00000 // (op 1261) call()
v_accvgpr_write a22, 0.00000 // (op 1261) call()
v_accvgpr_write a23, 0.00000 // (op 1261) call()
v_accvgpr_write a24, 0.00000 // (op 1262) call()
v_accvgpr_write a25, 0.00000 // (op 1262) call()
v_accvgpr_write a26, 0.00000 // (op 1262) call()
v_accvgpr_write a27, 0.00000 // (op 1262) call()
v_accvgpr_write a28, 0.00000 // (op 1263) call()
v_accvgpr_write a29, 0.00000 // (op 1263) call()
v_accvgpr_write a30, 0.00000 // (op 1263) call()
v_accvgpr_write a31, 0.00000 // (op 1263) call()
v_accvgpr_write a32, 0.00000 // (op 1722) call()
v_accvgpr_write a33, 0.00000 // (op 1722) call()
v_accvgpr_write a34, 0.00000 // (op 1722) call()
v_accvgpr_write a35, 0.00000 // (op 1722) call()
v_accvgpr_write a36, 0.00000 // (op 2031) call()
v_accvgpr_write a37, 0.00000 // (op 2031) call()
v_accvgpr_write a38, 0.00000 // (op 2031) call()
v_accvgpr_write a39, 0.00000 // (op 2031) call()
v_accvgpr_write a40, 0.00000 // (op 2032) call()
v_accvgpr_write a41, 0.00000 // (op 2032) call()
v_accvgpr_write a42, 0.00000 // (op 2032) call()
v_accvgpr_write a43, 0.00000 // (op 2032) call()
v_accvgpr_write a44, 0.00000 // (op 2033) call()
v_accvgpr_write a45, 0.00000 // (op 2033) call()
v_accvgpr_write a46, 0.00000 // (op 2033) call()
v_accvgpr_write a47, 0.00000 // (op 2033) call()
v_accvgpr_write a48, 0.00000 // (op 2492) call()
v_accvgpr_write a49, 0.00000 // (op 2492) call()
v_accvgpr_write a50, 0.00000 // (op 2492) call()
v_accvgpr_write a51, 0.00000 // (op 2492) call()
v_accvgpr_write a52, 0.00000 // (op 2801) call()
v_accvgpr_write a53, 0.00000 // (op 2801) call()
v_accvgpr_write a54, 0.00000 // (op 2801) call()
v_accvgpr_write a55, 0.00000 // (op 2801) call()
v_accvgpr_write a56, 0.00000 // (op 2802) call()
v_accvgpr_write a57, 0.00000 // (op 2802) call()
v_accvgpr_write a58, 0.00000 // (op 2802) call()
v_accvgpr_write a59, 0.00000 // (op 2802) call()
v_accvgpr_write a60, 0.00000 // (op 2803) call()
v_accvgpr_write a61, 0.00000 // (op 2803) call()
v_accvgpr_write a62, 0.00000 // (op 2803) call()
v_accvgpr_write a63, 0.00000 // (op 2803) call()
s_mul_hi_i32 s0, s16, s11 // (op 8619) 
v_lshrrev_b32 v14, 3, v1 // (op 8619) 
s_add_i32 s1, s0, s16 // (op 8619) 

v_and_b32 v15, -8, v1 // (op 8619) 
s_ashr_i32 s0, s1, 31 // (op 8619) 
v_sub_u32 v16, v1, v15 // (op 8619) 

s_cmp_eq_i32 s11, 0 // (op 8619) 
s_mov_b32 s18, scc // (op 8619) 
v_lshlrev_b32 v15, 4, v16 // (op 8619) 

s_cmp_eq_i32 s18, 0 // (op 8619) 
s_cselect_b32 s19, 0, -1 // (op 8619) 

v_mov_b32 v17, 0 // (op 8619) convert
v_mov_b32 v16, v15 // (op 8619) convert

s_and_b32 s18, s12, 31 // (op 8619) 
s_lshl_b32 s20, 1, s18 // (op 8619) 
s_add_i32 s21, s20, s19 // (op 8619) 


s_and_b32 s19, s0, s21 // (op 8619) 


s_add_i32 s0, s1, s19 // (op 8619) 


s_ashr_i32 s1, s0, s18 // (op 8619) 


s_lshl_b32 s0, s12, 24 // (op 8619) 
s_ashr_i32 s18, s0, 31 // (op 8619) 

s_xor_b32 s0, s1, s18 // (op 8619) 

s_sub_i32 s1, s0, s18 // (op 8619) 


s_mul_hi_i32 s0, s1, s13 // (op 8619) 
s_add_i32 s18, s0, s1 // (op 8619) 

s_ashr_i32 s0, s18, 31 // (op 8619) 
s_cmp_eq_i32 s13, 0 // (op 8619) 
s_mov_b32 s19, scc // (op 8619) 
s_cmp_eq_i32 s19, 0 // (op 8619) 
s_cselect_b32 s20, 0, -1 // (op 8619) 

s_and_b32 s19, s14, 31 // (op 8619) 
s_lshl_b32 s21, 1, s19 // (op 8619) 
s_add_i32 s22, s21, s20 // (op 8619) 


s_and_b32 s23, s0, s22 // (op 8619) 


s_add_i32 s0, s18, s23 // (op 8619) 


s_ashr_i32 s18, s0, s19 // (op 8619) 


s_lshl_b32 s0, s14, 24 // (op 8619) 
s_ashr_i32 s19, s0, 31 // (op 8619) 

s_xor_b32 s0, s18, s19 // (op 8619) 

s_sub_i32 s18, s0, s19 // (op 8619) 


s_mul_i32 s0, s1, s44 // (op 8619) 

s_sub_u32 s1, s16, s0 // (op 8619) 

s_mul_i32 s0, s18, s44 // (op 8619) 
s_add_u32 s0, s0, s1 // (op 8619) 


s_mul_hi_i32 s1, s16, s10 // (op 8619) 
s_add_i32 s18, s1, s16 // (op 8619) 

s_ashr_i32 s1, s18, 31 // (op 8619) 
s_cmp_eq_i32 s10, 0 // (op 8619) 
s_mov_b32 s19, scc // (op 8619) 
s_cmp_eq_i32 s19, 0 // (op 8619) 
s_cselect_b32 s20, 0, -1 // (op 8619) 

s_and_b32 s19, s47, 31 // (op 8619) 
s_lshl_b32 s21, 1, s19 // (op 8619) 
s_add_i32 s22, s21, s20 // (op 8619) 


s_and_b32 s23, s1, s22 // (op 8619) 


s_add_i32 s1, s18, s23 // (op 8619) 


s_ashr_i32 s18, s1, s19 // (op 8619) 


s_lshl_b32 s1, s47, 24 // (op 8619) 
s_ashr_i32 s19, s1, 31 // (op 8619) 

s_xor_b32 s1, s18, s19 // (op 8619) 

s_sub_i32 s18, s1, s19 // (op 8619) 


s_add_u32 s22, s6, 128 // (op 8619) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8619) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 8619) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 8619) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 7 // (op 8619) 

s_mul_i32 s1, s44, s22 // (op 8619) 

s_mul_i32 s19, s48, s1 // (op 8619) 

s_mul_i32 s1, s18, s19 // (op 8619) 

s_sub_u32 s19, s16, s1 // (op 8619) 

s_mov_b32 s1, s19 // (op 8619) 
s_mul_hi_i32 s20, s1, s15 // (op 8619) 
s_add_i32 s21, s20, s1 // (op 8619) 


s_ashr_i32 s1, s21, 31 // (op 8619) 
s_cmp_eq_i32 s15, 0 // (op 8619) 
s_mov_b32 s20, scc // (op 8619) 
s_cmp_eq_i32 s20, 0 // (op 8619) 
s_cselect_b32 s22, 0, -1 // (op 8619) 

s_and_b32 s20, s46, 31 // (op 8619) 
s_lshl_b32 s23, 1, s20 // (op 8619) 
s_add_i32 s49, s23, s22 // (op 8619) 


s_and_b32 s22, s1, s49 // (op 8619) 


s_add_i32 s1, s21, s22 // (op 8619) 


s_ashr_i32 s21, s1, s20 // (op 8619) 


s_lshl_b32 s1, s46, 24 // (op 8619) 
s_ashr_i32 s20, s1, 31 // (op 8619) 

s_xor_b32 s1, s21, s20 // (op 8619) 

s_sub_i32 s21, s1, s20 // (op 8619) 


s_mul_i32 s1, s45, s44 // (op 8619) 
s_add_u32 s22, s8, 128 // (op 8619) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 8619) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 8619) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 8619) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 7 // (op 8619) 

s_sub_i32 s20, s22, s1 // (op 8619) 

s_mul_i32 s22, s21, s20 // (op 8619) 


s_sub_u32 s23, s19, s22 // (op 8619) 


s_add_u32 s19, s23, s1 // (op 8619) 


s_cmp_eq_i32 s18, 0 // (op 8619) 
s_mov_b32 s1, scc // (op 8619) 

s_cmp_eq_i32 s1, 0 // (op 8619) 
s_cselect_b32 s18, s19, s0 // (op 8619) 



v_lshl_add_u32 v15, s18, 7, v14 // (op 8619) 


v_mul_lo_u32 v19, v15, s25 // (op 8619) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 8619) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 8619) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 8619) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 8619) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 8619) most significant half



s_mul_i32 s1, s24, s9 // (op 8621) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 8621) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8621) least significant
s_mul_hi_u32 s19, s24, s8 // (op 8621) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8621) most significant: sum
s_add_u32 s1, s1, s19 // (op 8621) most significant: sum


s_mov_b32 s20, s26 // (op 8621) reg expression
s_mov_b32 s21, s27 // (op 8621) reg expression
s_mov_b32 s22, s0 // (op 8621) reg expression
s_mov_b32 s23, 131072 // (op 8621) call()

s_mul_hi_i32 s0, s16, s11 // (op 8638) 
v_lshrrev_b32 v14, 3, v1 // (op 8638) 
s_add_i32 s1, s0, s16 // (op 8638) 

v_and_b32 v15, -8, v1 // (op 8638) 
s_ashr_i32 s0, s1, 31 // (op 8638) 
v_sub_u32 v16, v1, v15 // (op 8638) 

s_cmp_eq_i32 s11, 0 // (op 8638) 
s_mov_b32 s18, scc // (op 8638) 
v_lshlrev_b32 v15, 4, v16 // (op 8638) 

s_cmp_eq_i32 s18, 0 // (op 8638) 
s_cselect_b32 s19, 0, -1 // (op 8638) 

v_mov_b32 v17, 0 // (op 8638) convert
v_mov_b32 v16, v15 // (op 8638) convert

s_and_b32 s18, s12, 31 // (op 8638) 
s_lshl_b32 s49, 1, s18 // (op 8638) 
s_add_i32 s50, s49, s19 // (op 8638) 


s_and_b32 s19, s0, s50 // (op 8638) 


s_add_i32 s0, s1, s19 // (op 8638) 


s_ashr_i32 s1, s0, s18 // (op 8638) 


s_lshl_b32 s0, s12, 24 // (op 8638) 
s_ashr_i32 s18, s0, 31 // (op 8638) 

s_xor_b32 s0, s1, s18 // (op 8638) 

s_sub_i32 s1, s0, s18 // (op 8638) 


s_mul_hi_i32 s0, s1, s13 // (op 8638) 
s_add_i32 s18, s0, s1 // (op 8638) 

s_ashr_i32 s0, s18, 31 // (op 8638) 
s_cmp_eq_i32 s13, 0 // (op 8638) 
s_mov_b32 s19, scc // (op 8638) 
s_cmp_eq_i32 s19, 0 // (op 8638) 
s_cselect_b32 s49, 0, -1 // (op 8638) 

s_and_b32 s19, s14, 31 // (op 8638) 
s_lshl_b32 s50, 1, s19 // (op 8638) 
s_add_i32 s51, s50, s49 // (op 8638) 


s_and_b32 s49, s0, s51 // (op 8638) 


s_add_i32 s0, s18, s49 // (op 8638) 


s_ashr_i32 s18, s0, s19 // (op 8638) 


s_lshl_b32 s0, s14, 24 // (op 8638) 
s_ashr_i32 s19, s0, 31 // (op 8638) 

s_xor_b32 s0, s18, s19 // (op 8638) 

s_sub_i32 s18, s0, s19 // (op 8638) 


s_mul_i32 s0, s1, s44 // (op 8638) 

s_sub_u32 s1, s16, s0 // (op 8638) 

s_mul_i32 s0, s18, s44 // (op 8638) 
s_add_u32 s0, s0, s1 // (op 8638) 


s_mul_hi_i32 s1, s16, s10 // (op 8638) 
s_add_i32 s18, s1, s16 // (op 8638) 

s_ashr_i32 s1, s18, 31 // (op 8638) 
s_cmp_eq_i32 s10, 0 // (op 8638) 
s_mov_b32 s19, scc // (op 8638) 
s_cmp_eq_i32 s19, 0 // (op 8638) 
s_cselect_b32 s49, 0, -1 // (op 8638) 

s_and_b32 s19, s47, 31 // (op 8638) 
s_lshl_b32 s50, 1, s19 // (op 8638) 
s_add_i32 s51, s50, s49 // (op 8638) 


s_and_b32 s49, s1, s51 // (op 8638) 


s_add_i32 s1, s18, s49 // (op 8638) 


s_ashr_i32 s18, s1, s19 // (op 8638) 


s_lshl_b32 s1, s47, 24 // (op 8638) 
s_ashr_i32 s19, s1, 31 // (op 8638) 

s_xor_b32 s1, s18, s19 // (op 8638) 

s_sub_i32 s18, s1, s19 // (op 8638) 


s_add_u32 s54, s6, 128 // (op 8638) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 8638) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 8638) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 8638) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 8638) 

s_mul_i32 s1, s44, s50 // (op 8638) 

s_mul_i32 s19, s48, s1 // (op 8638) 

s_mul_i32 s1, s18, s19 // (op 8638) 

s_sub_u32 s19, s16, s1 // (op 8638) 

s_mov_b32 s1, s19 // (op 8638) 
s_mul_hi_i32 s49, s1, s15 // (op 8638) 
s_add_i32 s50, s49, s1 // (op 8638) 


s_ashr_i32 s1, s50, 31 // (op 8638) 
s_cmp_eq_i32 s15, 0 // (op 8638) 
s_mov_b32 s49, scc // (op 8638) 
s_cmp_eq_i32 s49, 0 // (op 8638) 
s_cselect_b32 s51, 0, -1 // (op 8638) 

s_and_b32 s49, s46, 31 // (op 8638) 
s_lshl_b32 s52, 1, s49 // (op 8638) 
s_add_i32 s53, s52, s51 // (op 8638) 


s_and_b32 s51, s1, s53 // (op 8638) 


s_add_i32 s1, s50, s51 // (op 8638) 


s_ashr_i32 s50, s1, s49 // (op 8638) 


s_lshl_b32 s1, s46, 24 // (op 8638) 
s_ashr_i32 s49, s1, 31 // (op 8638) 

s_xor_b32 s1, s50, s49 // (op 8638) 

s_sub_i32 s50, s1, s49 // (op 8638) 


s_mul_i32 s1, s45, s44 // (op 8638) 
s_add_u32 s54, s8, 128 // (op 8638) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 8638) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 8638) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 8638) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 7 // (op 8638) 

s_sub_i32 s49, s54, s1 // (op 8638) 

s_mul_i32 s51, s50, s49 // (op 8638) 


s_sub_u32 s49, s19, s51 // (op 8638) 


s_add_u32 s19, s49, s1 // (op 8638) 


s_cmp_eq_i32 s18, 0 // (op 8638) 
s_mov_b32 s1, scc // (op 8638) 

s_cmp_eq_i32 s1, 0 // (op 8638) 
s_cselect_b32 s18, s19, s0 // (op 8638) 



v_lshl_add_u32 v15, s18, 7, v14 // (op 8638) 


v_mul_lo_u32 v21, v15, s25 // (op 8638) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 8638) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 8638) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8638) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 8638) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 8638) most significant half



s_mul_i32 s1, s24, s9 // (op 8640) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 8640) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8640) least significant
s_mul_hi_u32 s19, s24, s8 // (op 8640) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8640) most significant: sum
s_add_u32 s1, s1, s19 // (op 8640) most significant: sum


s_mov_b32 s20, s26 // (op 8640) reg expression
s_mov_b32 s21, s27 // (op 8640) reg expression
s_mov_b32 s22, s0 // (op 8640) reg expression
s_mov_b32 s23, 131072 // (op 8640) call()

s_mul_hi_i32 s0, s16, s11 // (op 8683) 
v_and_b32 v14, -8, v1 // (op 8683) 
s_add_i32 s1, s0, s16 // (op 8683) 

v_sub_u32 v15, v1, v14 // (op 8683) 

s_ashr_i32 s0, s1, 31 // (op 8683) 
v_lshlrev_b32 v14, 4, v15 // (op 8683) 

s_cmp_eq_i32 s11, 0 // (op 8683) 
s_mov_b32 s18, scc // (op 8683) 
v_lshrrev_b32 v15, 3, v1 // (op 8683) 
s_cmp_eq_i32 s18, 0 // (op 8683) 
s_cselect_b32 s19, 0, -1 // (op 8683) 

s_and_b32 s18, s12, 31 // (op 8683) 
s_lshl_b32 s49, 1, s18 // (op 8683) 
s_add_i32 s50, s49, s19 // (op 8683) 


s_and_b32 s19, s0, s50 // (op 8683) 


s_add_i32 s0, s1, s19 // (op 8683) 


s_ashr_i32 s1, s0, s18 // (op 8683) 


s_lshl_b32 s0, s12, 24 // (op 8683) 
s_ashr_i32 s18, s0, 31 // (op 8683) 

s_xor_b32 s0, s1, s18 // (op 8683) 

s_sub_i32 s1, s0, s18 // (op 8683) 


s_mul_hi_i32 s0, s1, s13 // (op 8683) 
s_add_i32 s18, s0, s1 // (op 8683) 

s_ashr_i32 s0, s18, 31 // (op 8683) 
s_cmp_eq_i32 s13, 0 // (op 8683) 
s_mov_b32 s19, scc // (op 8683) 
s_cmp_eq_i32 s19, 0 // (op 8683) 
s_cselect_b32 s49, 0, -1 // (op 8683) 

s_and_b32 s19, s14, 31 // (op 8683) 
s_lshl_b32 s50, 1, s19 // (op 8683) 
s_add_i32 s51, s50, s49 // (op 8683) 


s_and_b32 s49, s0, s51 // (op 8683) 


s_add_i32 s0, s18, s49 // (op 8683) 


s_ashr_i32 s18, s0, s19 // (op 8683) 


s_lshl_b32 s0, s14, 24 // (op 8683) 
s_ashr_i32 s19, s0, 31 // (op 8683) 

s_xor_b32 s0, s18, s19 // (op 8683) 

s_sub_i32 s18, s0, s19 // (op 8683) 


s_add_u32 s54, s6, 128 // (op 8683) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 8683) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 8683) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 8683) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 7 // (op 8683) 

s_mul_i32 s0, s18, s50 // (op 8683) 

s_sub_i32 s49, s1, s0 // (op 8683) 


s_mul_hi_i32 s0, s16, s10 // (op 8683) 
s_add_i32 s1, s0, s16 // (op 8683) 

s_ashr_i32 s0, s1, 31 // (op 8683) 
s_cmp_eq_i32 s10, 0 // (op 8683) 
s_mov_b32 s18, scc // (op 8683) 
s_cmp_eq_i32 s18, 0 // (op 8683) 
s_cselect_b32 s19, 0, -1 // (op 8683) 

s_and_b32 s18, s47, 31 // (op 8683) 
s_lshl_b32 s52, 1, s18 // (op 8683) 
s_add_i32 s53, s52, s19 // (op 8683) 


s_and_b32 s19, s0, s53 // (op 8683) 


s_add_i32 s0, s1, s19 // (op 8683) 


s_ashr_i32 s1, s0, s18 // (op 8683) 


s_lshl_b32 s0, s47, 24 // (op 8683) 
s_ashr_i32 s18, s0, 31 // (op 8683) 

s_xor_b32 s0, s1, s18 // (op 8683) 

s_sub_i32 s1, s0, s18 // (op 8683) 


s_mul_i32 s0, s44, s50 // (op 8683) 

s_mul_i32 s18, s48, s0 // (op 8683) 

s_mul_i32 s0, s1, s18 // (op 8683) 

s_sub_u32 s18, s16, s0 // (op 8683) 

s_mul_hi_i32 s0, s18, s15 // (op 8683) 
s_add_i32 s19, s0, s18 // (op 8683) 


s_ashr_i32 s0, s19, 31 // (op 8683) 
s_cmp_eq_i32 s15, 0 // (op 8683) 
s_mov_b32 s18, scc // (op 8683) 
s_cmp_eq_i32 s18, 0 // (op 8683) 
s_cselect_b32 s50, 0, -1 // (op 8683) 

s_and_b32 s18, s46, 31 // (op 8683) 
s_lshl_b32 s51, 1, s18 // (op 8683) 
s_add_i32 s52, s51, s50 // (op 8683) 


s_and_b32 s50, s0, s52 // (op 8683) 


s_add_i32 s0, s19, s50 // (op 8683) 


s_ashr_i32 s19, s0, s18 // (op 8683) 


s_lshl_b32 s0, s46, 24 // (op 8683) 
s_ashr_i32 s18, s0, 31 // (op 8683) 

s_xor_b32 s0, s19, s18 // (op 8683) 

s_sub_i32 s19, s0, s18 // (op 8683) 


s_cmp_eq_i32 s1, 0 // (op 8683) 
s_mov_b32 s0, scc // (op 8683) 

s_cmp_eq_i32 s0, 0 // (op 8683) 
s_cselect_b32 s1, s19, s49 // (op 8683) 



v_lshl_add_u32 v16, s1, 7, v14 // (op 8683) 


v_mov_b32 v23, 0 // (op 8683) convert
v_mov_b32 v22, v16 // (op 8683) convert

v_mul_lo_u32 v17, v15, s29 // (op 8683) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 8683) most significant: high of low * low
v_mul_lo_u32 v16, v15, s28 // (op 8683) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 8683) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v22 // (op 8683) least significant half
v_addc_co_u32 v17, s[0:1], v17, v23, s[0:1] // (op 8683) most significant half



s_mul_i32 s1, s28, s33 // (op 8685) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 8685) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8685) least significant
s_mul_hi_u32 s19, s28, s32 // (op 8685) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8685) most significant: sum
s_add_u32 s1, s1, s19 // (op 8685) most significant: sum


s_mov_b32 s52, s30 // (op 8685) reg expression
s_mov_b32 s53, s31 // (op 8685) reg expression
s_mov_b32 s54, s0 // (op 8685) reg expression
s_mov_b32 s55, 131072 // (op 8685) call()

s_mul_hi_i32 s0, s16, s11 // (op 8702) 
v_and_b32 v14, -8, v1 // (op 8702) 
s_add_i32 s1, s0, s16 // (op 8702) 

v_sub_u32 v15, v1, v14 // (op 8702) 

s_ashr_i32 s0, s1, 31 // (op 8702) 
v_lshlrev_b32 v14, 4, v15 // (op 8702) 

s_cmp_eq_i32 s11, 0 // (op 8702) 
s_mov_b32 s18, scc // (op 8702) 
v_lshrrev_b32 v15, 3, v1 // (op 8702) 
s_cmp_eq_i32 s18, 0 // (op 8702) 
s_cselect_b32 s19, 0, -1 // (op 8702) 

s_and_b32 s18, s12, 31 // (op 8702) 
s_lshl_b32 s49, 1, s18 // (op 8702) 
s_add_i32 s50, s49, s19 // (op 8702) 


s_and_b32 s19, s0, s50 // (op 8702) 


s_add_i32 s0, s1, s19 // (op 8702) 


s_ashr_i32 s1, s0, s18 // (op 8702) 


s_lshl_b32 s0, s12, 24 // (op 8702) 
s_ashr_i32 s18, s0, 31 // (op 8702) 

s_xor_b32 s0, s1, s18 // (op 8702) 

s_sub_i32 s1, s0, s18 // (op 8702) 


s_mul_hi_i32 s0, s1, s13 // (op 8702) 
s_add_i32 s18, s0, s1 // (op 8702) 

s_ashr_i32 s0, s18, 31 // (op 8702) 
s_cmp_eq_i32 s13, 0 // (op 8702) 
s_mov_b32 s19, scc // (op 8702) 
s_cmp_eq_i32 s19, 0 // (op 8702) 
s_cselect_b32 s49, 0, -1 // (op 8702) 

s_and_b32 s19, s14, 31 // (op 8702) 
s_lshl_b32 s50, 1, s19 // (op 8702) 
s_add_i32 s51, s50, s49 // (op 8702) 


s_and_b32 s49, s0, s51 // (op 8702) 


s_add_i32 s0, s18, s49 // (op 8702) 


s_ashr_i32 s18, s0, s19 // (op 8702) 


s_lshl_b32 s0, s14, 24 // (op 8702) 
s_ashr_i32 s19, s0, 31 // (op 8702) 

s_xor_b32 s0, s18, s19 // (op 8702) 

s_sub_i32 s18, s0, s19 // (op 8702) 


s_add_u32 s50, s6, 128 // (op 8702) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8702) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 8702) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 8702) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 8702) 

s_mul_i32 s0, s18, s50 // (op 8702) 

s_sub_i32 s49, s1, s0 // (op 8702) 


s_mul_hi_i32 s0, s16, s10 // (op 8702) 
s_add_i32 s1, s0, s16 // (op 8702) 

s_ashr_i32 s0, s1, 31 // (op 8702) 
s_cmp_eq_i32 s10, 0 // (op 8702) 
s_mov_b32 s18, scc // (op 8702) 
s_cmp_eq_i32 s18, 0 // (op 8702) 
s_cselect_b32 s19, 0, -1 // (op 8702) 

s_and_b32 s18, s47, 31 // (op 8702) 
s_lshl_b32 s64, 1, s18 // (op 8702) 
s_add_i32 s65, s64, s19 // (op 8702) 


s_and_b32 s19, s0, s65 // (op 8702) 


s_add_i32 s0, s1, s19 // (op 8702) 


s_ashr_i32 s1, s0, s18 // (op 8702) 


s_lshl_b32 s0, s47, 24 // (op 8702) 
s_ashr_i32 s18, s0, 31 // (op 8702) 

s_xor_b32 s0, s1, s18 // (op 8702) 

s_sub_i32 s1, s0, s18 // (op 8702) 


s_mul_i32 s0, s44, s50 // (op 8702) 

s_mul_i32 s18, s48, s0 // (op 8702) 

s_mul_i32 s0, s1, s18 // (op 8702) 

s_sub_u32 s18, s16, s0 // (op 8702) 

s_mul_hi_i32 s0, s18, s15 // (op 8702) 
s_add_i32 s19, s0, s18 // (op 8702) 


s_ashr_i32 s0, s19, 31 // (op 8702) 
s_cmp_eq_i32 s15, 0 // (op 8702) 
s_mov_b32 s18, scc // (op 8702) 
s_cmp_eq_i32 s18, 0 // (op 8702) 
s_cselect_b32 s50, 0, -1 // (op 8702) 

s_and_b32 s18, s46, 31 // (op 8702) 
s_lshl_b32 s51, 1, s18 // (op 8702) 
s_add_i32 s64, s51, s50 // (op 8702) 


s_and_b32 s50, s0, s64 // (op 8702) 


s_add_i32 s0, s19, s50 // (op 8702) 


s_ashr_i32 s19, s0, s18 // (op 8702) 


s_lshl_b32 s0, s46, 24 // (op 8702) 
s_ashr_i32 s18, s0, 31 // (op 8702) 

s_xor_b32 s0, s19, s18 // (op 8702) 

s_sub_i32 s19, s0, s18 // (op 8702) 


s_cmp_eq_i32 s1, 0 // (op 8702) 
s_mov_b32 s0, scc // (op 8702) 

s_cmp_eq_i32 s0, 0 // (op 8702) 
s_cselect_b32 s1, s19, s49 // (op 8702) 



v_lshl_add_u32 v22, s1, 7, v14 // (op 8702) 


v_mov_b32 v25, 0 // (op 8702) convert
v_mov_b32 v24, v22 // (op 8702) convert

v_mul_lo_u32 v23, v15, s29 // (op 8702) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 8702) most significant: high of low * low
v_mul_lo_u32 v22, v15, s28 // (op 8702) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 8702) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v24 // (op 8702) least significant half
v_addc_co_u32 v23, s[0:1], v23, v25, s[0:1] // (op 8702) most significant half



s_mul_i32 s1, s28, s33 // (op 8704) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 8704) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8704) least significant
s_mul_hi_u32 s19, s28, s32 // (op 8704) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8704) most significant: sum
s_add_u32 s1, s1, s19 // (op 8704) most significant: sum


s_mov_b32 s52, s30 // (op 8704) reg expression
s_mov_b32 s53, s31 // (op 8704) reg expression
s_mov_b32 s54, s0 // (op 8704) reg expression
s_mov_b32 s55, 131072 // (op 8704) call()

s_mul_hi_i32 s0, s16, s11 // (op 8747) 
v_lshrrev_b32 v14, 2, v1 // (op 8747) 
s_add_i32 s1, s0, s16 // (op 8747) 

v_lshlrev_b32 v15, 1, v14 // (op 8747) 

s_ashr_i32 s0, s1, 31 // (op 8747) 
v_and_b32 v14, -4, v1 // (op 8747) 
s_cmp_eq_i32 s11, 0 // (op 8747) 
s_mov_b32 s18, scc // (op 8747) 
v_sub_u32 v24, v1, v14 // (op 8747) 

s_cmp_eq_i32 s18, 0 // (op 8747) 
s_cselect_b32 s19, 0, -1 // (op 8747) 

s_and_b32 s18, s12, 31 // (op 8747) 
s_lshl_b32 s49, 1, s18 // (op 8747) 
s_add_i32 s50, s49, s19 // (op 8747) 


s_and_b32 s19, s0, s50 // (op 8747) 


s_add_i32 s0, s1, s19 // (op 8747) 


s_ashr_i32 s1, s0, s18 // (op 8747) 


s_lshl_b32 s0, s12, 24 // (op 8747) 
s_ashr_i32 s18, s0, 31 // (op 8747) 

s_xor_b32 s0, s1, s18 // (op 8747) 

s_sub_i32 s1, s0, s18 // (op 8747) 


s_mul_hi_i32 s0, s1, s13 // (op 8747) 
s_add_i32 s18, s0, s1 // (op 8747) 

s_ashr_i32 s0, s18, 31 // (op 8747) 
s_cmp_eq_i32 s13, 0 // (op 8747) 
s_mov_b32 s19, scc // (op 8747) 
s_cmp_eq_i32 s19, 0 // (op 8747) 
s_cselect_b32 s49, 0, -1 // (op 8747) 

s_and_b32 s19, s14, 31 // (op 8747) 
s_lshl_b32 s50, 1, s19 // (op 8747) 
s_add_i32 s51, s50, s49 // (op 8747) 


s_and_b32 s49, s0, s51 // (op 8747) 


s_add_i32 s0, s18, s49 // (op 8747) 


s_ashr_i32 s18, s0, s19 // (op 8747) 


s_lshl_b32 s0, s14, 24 // (op 8747) 
s_ashr_i32 s19, s0, 31 // (op 8747) 

s_xor_b32 s0, s18, s19 // (op 8747) 

s_sub_i32 s18, s0, s19 // (op 8747) 


s_mul_i32 s0, s1, s44 // (op 8747) 

s_sub_u32 s1, s16, s0 // (op 8747) 

s_mul_i32 s0, s18, s44 // (op 8747) 
s_add_u32 s0, s0, s1 // (op 8747) 


s_mul_hi_i32 s1, s16, s10 // (op 8747) 
s_add_i32 s18, s1, s16 // (op 8747) 

s_ashr_i32 s1, s18, 31 // (op 8747) 
s_cmp_eq_i32 s10, 0 // (op 8747) 
s_mov_b32 s19, scc // (op 8747) 
s_cmp_eq_i32 s19, 0 // (op 8747) 
s_cselect_b32 s49, 0, -1 // (op 8747) 

s_and_b32 s19, s47, 31 // (op 8747) 
s_lshl_b32 s50, 1, s19 // (op 8747) 
s_add_i32 s51, s50, s49 // (op 8747) 


s_and_b32 s49, s1, s51 // (op 8747) 


s_add_i32 s1, s18, s49 // (op 8747) 


s_ashr_i32 s18, s1, s19 // (op 8747) 


s_lshl_b32 s1, s47, 24 // (op 8747) 
s_ashr_i32 s19, s1, 31 // (op 8747) 

s_xor_b32 s1, s18, s19 // (op 8747) 

s_sub_i32 s18, s1, s19 // (op 8747) 


s_add_u32 s50, s6, 128 // (op 8747) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8747) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 8747) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 8747) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 7 // (op 8747) 

s_mul_i32 s1, s44, s50 // (op 8747) 

s_mul_i32 s19, s48, s1 // (op 8747) 

s_mul_i32 s1, s18, s19 // (op 8747) 

s_sub_u32 s19, s16, s1 // (op 8747) 

s_mov_b32 s1, s19 // (op 8747) 
s_mul_hi_i32 s49, s1, s15 // (op 8747) 
s_add_i32 s50, s49, s1 // (op 8747) 


s_ashr_i32 s1, s50, 31 // (op 8747) 
s_cmp_eq_i32 s15, 0 // (op 8747) 
s_mov_b32 s49, scc // (op 8747) 
s_cmp_eq_i32 s49, 0 // (op 8747) 
s_cselect_b32 s51, 0, -1 // (op 8747) 

s_and_b32 s49, s46, 31 // (op 8747) 
s_lshl_b32 s64, 1, s49 // (op 8747) 
s_add_i32 s65, s64, s51 // (op 8747) 


s_and_b32 s51, s1, s65 // (op 8747) 


s_add_i32 s1, s50, s51 // (op 8747) 


s_ashr_i32 s50, s1, s49 // (op 8747) 


s_lshl_b32 s1, s46, 24 // (op 8747) 
s_ashr_i32 s49, s1, 31 // (op 8747) 

s_xor_b32 s1, s50, s49 // (op 8747) 

s_sub_i32 s50, s1, s49 // (op 8747) 


s_mul_i32 s1, s45, s44 // (op 8747) 
s_add_u32 s64, s8, 128 // (op 8747) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 8747) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 8747) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 8747) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 8747) 

s_sub_i32 s49, s64, s1 // (op 8747) 

s_mul_i32 s51, s50, s49 // (op 8747) 


s_sub_u32 s49, s19, s51 // (op 8747) 


s_add_u32 s19, s49, s1 // (op 8747) 


s_cmp_eq_i32 s18, 0 // (op 8747) 
s_mov_b32 s1, scc // (op 8747) 

s_cmp_eq_i32 s1, 0 // (op 8747) 
s_cselect_b32 s18, s19, s0 // (op 8747) 



v_lshl_add_u32 v14, s18, 7, v15 // (op 8747) 


v_mul_lo_u32 v15, v14, s34 // (op 8747) 
v_add_u32 v15, v15, v24 // (op 8747) 


s_mul_i32 s1, s34, s9 // (op 8749) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 8749) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8749) least significant
s_mul_hi_u32 s19, s34, s8 // (op 8749) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8749) most significant: sum
s_add_u32 s1, s1, s19 // (op 8749) most significant: sum


s_mov_b32 s64, s36 // (op 8749) reg expression
s_mov_b32 s65, s37 // (op 8749) reg expression
s_mov_b32 s66, s0 // (op 8749) reg expression
s_mov_b32 s67, 131072 // (op 8749) call()

s_mul_hi_i32 s0, s16, s11 // (op 8766) 
v_lshrrev_b32 v14, 2, v1 // (op 8766) 
s_add_i32 s1, s0, s16 // (op 8766) 

v_lshlrev_b32 v24, 1, v14 // (op 8766) 

s_ashr_i32 s0, s1, 31 // (op 8766) 
v_and_b32 v14, -4, v1 // (op 8766) 
s_cmp_eq_i32 s11, 0 // (op 8766) 
s_mov_b32 s18, scc // (op 8766) 
v_sub_u32 v25, v1, v14 // (op 8766) 

s_cmp_eq_i32 s18, 0 // (op 8766) 
s_cselect_b32 s19, 0, -1 // (op 8766) 

s_and_b32 s18, s12, 31 // (op 8766) 
s_lshl_b32 s49, 1, s18 // (op 8766) 
s_add_i32 s50, s49, s19 // (op 8766) 


s_and_b32 s19, s0, s50 // (op 8766) 


s_add_i32 s0, s1, s19 // (op 8766) 


s_ashr_i32 s1, s0, s18 // (op 8766) 


s_lshl_b32 s0, s12, 24 // (op 8766) 
s_ashr_i32 s18, s0, 31 // (op 8766) 

s_xor_b32 s0, s1, s18 // (op 8766) 

s_sub_i32 s1, s0, s18 // (op 8766) 


s_mul_hi_i32 s0, s1, s13 // (op 8766) 
s_add_i32 s18, s0, s1 // (op 8766) 

s_ashr_i32 s0, s18, 31 // (op 8766) 
s_cmp_eq_i32 s13, 0 // (op 8766) 
s_mov_b32 s19, scc // (op 8766) 
s_cmp_eq_i32 s19, 0 // (op 8766) 
s_cselect_b32 s49, 0, -1 // (op 8766) 

s_and_b32 s19, s14, 31 // (op 8766) 
s_lshl_b32 s50, 1, s19 // (op 8766) 
s_add_i32 s51, s50, s49 // (op 8766) 


s_and_b32 s49, s0, s51 // (op 8766) 


s_add_i32 s0, s18, s49 // (op 8766) 


s_ashr_i32 s18, s0, s19 // (op 8766) 


s_lshl_b32 s0, s14, 24 // (op 8766) 
s_ashr_i32 s19, s0, 31 // (op 8766) 

s_xor_b32 s0, s18, s19 // (op 8766) 

s_sub_i32 s18, s0, s19 // (op 8766) 


s_mul_i32 s0, s1, s44 // (op 8766) 

s_sub_u32 s1, s16, s0 // (op 8766) 

s_mul_i32 s0, s18, s44 // (op 8766) 
s_add_u32 s0, s0, s1 // (op 8766) 


s_mul_hi_i32 s1, s16, s10 // (op 8766) 
s_add_i32 s18, s1, s16 // (op 8766) 

s_ashr_i32 s1, s18, 31 // (op 8766) 
s_cmp_eq_i32 s10, 0 // (op 8766) 
s_mov_b32 s19, scc // (op 8766) 
s_cmp_eq_i32 s19, 0 // (op 8766) 
s_cselect_b32 s49, 0, -1 // (op 8766) 

s_and_b32 s19, s47, 31 // (op 8766) 
s_lshl_b32 s50, 1, s19 // (op 8766) 
s_add_i32 s51, s50, s49 // (op 8766) 


s_and_b32 s49, s1, s51 // (op 8766) 


s_add_i32 s1, s18, s49 // (op 8766) 


s_ashr_i32 s18, s1, s19 // (op 8766) 


s_lshl_b32 s1, s47, 24 // (op 8766) 
s_ashr_i32 s19, s1, 31 // (op 8766) 

s_xor_b32 s1, s18, s19 // (op 8766) 

s_sub_i32 s18, s1, s19 // (op 8766) 


s_add_u32 s50, s6, 128 // (op 8766) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8766) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 8766) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 8766) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 8766) 

s_mul_i32 s1, s44, s50 // (op 8766) 

s_mul_i32 s19, s48, s1 // (op 8766) 

s_mul_i32 s1, s18, s19 // (op 8766) 

s_sub_u32 s19, s16, s1 // (op 8766) 

s_mov_b32 s1, s19 // (op 8766) 
s_mul_hi_i32 s49, s1, s15 // (op 8766) 
s_add_i32 s50, s49, s1 // (op 8766) 


s_ashr_i32 s1, s50, 31 // (op 8766) 
s_cmp_eq_i32 s15, 0 // (op 8766) 
s_mov_b32 s49, scc // (op 8766) 
s_cmp_eq_i32 s49, 0 // (op 8766) 
s_cselect_b32 s51, 0, -1 // (op 8766) 

s_and_b32 s49, s46, 31 // (op 8766) 
s_lshl_b32 s68, 1, s49 // (op 8766) 
s_add_i32 s69, s68, s51 // (op 8766) 


s_and_b32 s51, s1, s69 // (op 8766) 


s_add_i32 s1, s50, s51 // (op 8766) 


s_ashr_i32 s50, s1, s49 // (op 8766) 


s_lshl_b32 s1, s46, 24 // (op 8766) 
s_ashr_i32 s49, s1, 31 // (op 8766) 

s_xor_b32 s1, s50, s49 // (op 8766) 

s_sub_i32 s50, s1, s49 // (op 8766) 


s_mul_i32 s1, s45, s44 // (op 8766) 
s_add_u32 s68, s8, 128 // (op 8766) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 8766) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 8766) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 8766) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 7 // (op 8766) 

s_sub_i32 s49, s68, s1 // (op 8766) 

s_mul_i32 s51, s50, s49 // (op 8766) 


s_sub_u32 s49, s19, s51 // (op 8766) 


s_add_u32 s19, s49, s1 // (op 8766) 


s_cmp_eq_i32 s18, 0 // (op 8766) 
s_mov_b32 s1, scc // (op 8766) 

s_cmp_eq_i32 s1, 0 // (op 8766) 
s_cselect_b32 s18, s19, s0 // (op 8766) 



v_lshl_add_u32 v14, s18, 7, v24 // (op 8766) 


v_mul_lo_u32 v24, v14, s34 // (op 8766) 
v_add_u32 v24, v24, v25 // (op 8766) 


s_mul_i32 s1, s34, s9 // (op 8768) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 8768) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8768) least significant
s_mul_hi_u32 s19, s34, s8 // (op 8768) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8768) most significant: sum
s_add_u32 s1, s1, s19 // (op 8768) most significant: sum


s_mov_b32 s64, s36 // (op 8768) reg expression
s_mov_b32 s65, s37 // (op 8768) reg expression
s_mov_b32 s66, s0 // (op 8768) reg expression
s_mov_b32 s67, 131072 // (op 8768) call()

s_mul_hi_i32 s0, s16, s11 // (op 8811) 
v_and_b32 v14, -128, v1 // (op 8811) 
s_add_i32 s1, s0, s16 // (op 8811) 

v_sub_u32 v25, v1, v14 // (op 8811) 

s_ashr_i32 s0, s1, 31 // (op 8811) 
v_lshrrev_b32 v14, 7, v1 // (op 8811) 
s_cmp_eq_i32 s11, 0 // (op 8811) 
s_mov_b32 s18, scc // (op 8811) 
v_lshlrev_b32 v26, 1, v14 // (op 8811) 

s_cmp_eq_i32 s18, 0 // (op 8811) 
s_cselect_b32 s19, 0, -1 // (op 8811) 

s_and_b32 s18, s12, 31 // (op 8811) 
s_lshl_b32 s49, 1, s18 // (op 8811) 
s_add_i32 s50, s49, s19 // (op 8811) 


s_and_b32 s19, s0, s50 // (op 8811) 


s_add_i32 s0, s1, s19 // (op 8811) 


s_ashr_i32 s1, s0, s18 // (op 8811) 


s_lshl_b32 s0, s12, 24 // (op 8811) 
s_ashr_i32 s18, s0, 31 // (op 8811) 

s_xor_b32 s0, s1, s18 // (op 8811) 

s_sub_i32 s1, s0, s18 // (op 8811) 


s_mul_hi_i32 s0, s1, s13 // (op 8811) 
s_add_i32 s18, s0, s1 // (op 8811) 

s_ashr_i32 s0, s18, 31 // (op 8811) 
s_cmp_eq_i32 s13, 0 // (op 8811) 
s_mov_b32 s19, scc // (op 8811) 
s_cmp_eq_i32 s19, 0 // (op 8811) 
s_cselect_b32 s49, 0, -1 // (op 8811) 

s_and_b32 s19, s14, 31 // (op 8811) 
s_lshl_b32 s50, 1, s19 // (op 8811) 
s_add_i32 s51, s50, s49 // (op 8811) 


s_and_b32 s49, s0, s51 // (op 8811) 


s_add_i32 s0, s18, s49 // (op 8811) 


s_ashr_i32 s18, s0, s19 // (op 8811) 


s_lshl_b32 s0, s14, 24 // (op 8811) 
s_ashr_i32 s19, s0, 31 // (op 8811) 

s_xor_b32 s0, s18, s19 // (op 8811) 

s_sub_i32 s18, s0, s19 // (op 8811) 


s_add_u32 s50, s6, 128 // (op 8811) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8811) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 8811) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 8811) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 7 // (op 8811) 

s_mul_i32 s0, s18, s50 // (op 8811) 

s_sub_i32 s49, s1, s0 // (op 8811) 


s_mul_hi_i32 s0, s16, s10 // (op 8811) 
s_add_i32 s1, s0, s16 // (op 8811) 

s_ashr_i32 s0, s1, 31 // (op 8811) 
s_cmp_eq_i32 s10, 0 // (op 8811) 
s_mov_b32 s18, scc // (op 8811) 
s_cmp_eq_i32 s18, 0 // (op 8811) 
s_cselect_b32 s19, 0, -1 // (op 8811) 

s_and_b32 s18, s47, 31 // (op 8811) 
s_lshl_b32 s68, 1, s18 // (op 8811) 
s_add_i32 s69, s68, s19 // (op 8811) 


s_and_b32 s19, s0, s69 // (op 8811) 


s_add_i32 s0, s1, s19 // (op 8811) 


s_ashr_i32 s1, s0, s18 // (op 8811) 


s_lshl_b32 s0, s47, 24 // (op 8811) 
s_ashr_i32 s18, s0, 31 // (op 8811) 

s_xor_b32 s0, s1, s18 // (op 8811) 

s_sub_i32 s1, s0, s18 // (op 8811) 


s_mul_i32 s0, s44, s50 // (op 8811) 

s_mul_i32 s18, s48, s0 // (op 8811) 

s_mul_i32 s0, s1, s18 // (op 8811) 

s_sub_u32 s18, s16, s0 // (op 8811) 

s_mul_hi_i32 s0, s18, s15 // (op 8811) 
s_add_i32 s19, s0, s18 // (op 8811) 


s_ashr_i32 s0, s19, 31 // (op 8811) 
s_cmp_eq_i32 s15, 0 // (op 8811) 
s_mov_b32 s18, scc // (op 8811) 
s_cmp_eq_i32 s18, 0 // (op 8811) 
s_cselect_b32 s50, 0, -1 // (op 8811) 

s_and_b32 s18, s46, 31 // (op 8811) 
s_lshl_b32 s51, 1, s18 // (op 8811) 
s_add_i32 s68, s51, s50 // (op 8811) 


s_and_b32 s50, s0, s68 // (op 8811) 


s_add_i32 s0, s19, s50 // (op 8811) 


s_ashr_i32 s19, s0, s18 // (op 8811) 


s_lshl_b32 s0, s46, 24 // (op 8811) 
s_ashr_i32 s18, s0, 31 // (op 8811) 

s_xor_b32 s0, s19, s18 // (op 8811) 

s_sub_i32 s19, s0, s18 // (op 8811) 


s_cmp_eq_i32 s1, 0 // (op 8811) 
s_mov_b32 s0, scc // (op 8811) 

s_cmp_eq_i32 s0, 0 // (op 8811) 
s_cselect_b32 s1, s19, s49 // (op 8811) 



v_lshl_add_u32 v14, s1, 7, v25 // (op 8811) 


v_mul_lo_u32 v25, v26, s38 // (op 8811) 
v_add_u32 v25, v25, v14 // (op 8811) 


s_mul_i32 s1, s38, s43 // (op 8813) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 8813) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 8813) least significant
s_mul_hi_u32 s19, s38, s42 // (op 8813) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8813) most significant: sum
s_add_u32 s1, s1, s19 // (op 8813) most significant: sum


s_mov_b32 s68, s40 // (op 8813) reg expression
s_mov_b32 s69, s41 // (op 8813) reg expression
s_mov_b32 s70, s0 // (op 8813) reg expression
s_mov_b32 s71, 131072 // (op 8813) call()

s_mul_hi_i32 s0, s16, s11 // (op 8830) 
v_and_b32 v14, -128, v1 // (op 8830) 
s_add_i32 s1, s0, s16 // (op 8830) 

v_sub_u32 v26, v1, v14 // (op 8830) 

s_ashr_i32 s0, s1, 31 // (op 8830) 
v_lshrrev_b32 v14, 7, v1 // (op 8830) 
s_cmp_eq_i32 s11, 0 // (op 8830) 
s_mov_b32 s18, scc // (op 8830) 
v_lshlrev_b32 v27, 1, v14 // (op 8830) 

s_cmp_eq_i32 s18, 0 // (op 8830) 
s_cselect_b32 s19, 0, -1 // (op 8830) 

s_and_b32 s18, s12, 31 // (op 8830) 
s_lshl_b32 s49, 1, s18 // (op 8830) 
s_add_i32 s50, s49, s19 // (op 8830) 


s_and_b32 s19, s0, s50 // (op 8830) 


s_add_i32 s0, s1, s19 // (op 8830) 


s_ashr_i32 s1, s0, s18 // (op 8830) 


s_lshl_b32 s0, s12, 24 // (op 8830) 
s_ashr_i32 s18, s0, 31 // (op 8830) 

s_xor_b32 s0, s1, s18 // (op 8830) 

s_sub_i32 s1, s0, s18 // (op 8830) 


s_mul_hi_i32 s0, s1, s13 // (op 8830) 
s_add_i32 s18, s0, s1 // (op 8830) 

s_ashr_i32 s0, s18, 31 // (op 8830) 
s_cmp_eq_i32 s13, 0 // (op 8830) 
s_mov_b32 s19, scc // (op 8830) 
s_cmp_eq_i32 s19, 0 // (op 8830) 
s_cselect_b32 s49, 0, -1 // (op 8830) 

s_and_b32 s19, s14, 31 // (op 8830) 
s_lshl_b32 s50, 1, s19 // (op 8830) 
s_add_i32 s51, s50, s49 // (op 8830) 


s_and_b32 s49, s0, s51 // (op 8830) 


s_add_i32 s0, s18, s49 // (op 8830) 


s_ashr_i32 s18, s0, s19 // (op 8830) 


s_lshl_b32 s0, s14, 24 // (op 8830) 
s_ashr_i32 s19, s0, 31 // (op 8830) 

s_xor_b32 s0, s18, s19 // (op 8830) 

s_sub_i32 s18, s0, s19 // (op 8830) 


s_add_u32 s50, s6, 128 // (op 8830) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8830) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 8830) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 8830) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 8830) 

s_mul_i32 s0, s18, s50 // (op 8830) 

s_sub_i32 s49, s1, s0 // (op 8830) 


s_mul_hi_i32 s0, s16, s10 // (op 8830) 
s_add_i32 s1, s0, s16 // (op 8830) 

s_ashr_i32 s0, s1, 31 // (op 8830) 
s_cmp_eq_i32 s10, 0 // (op 8830) 
s_mov_b32 s18, scc // (op 8830) 
s_cmp_eq_i32 s18, 0 // (op 8830) 
s_cselect_b32 s19, 0, -1 // (op 8830) 

s_and_b32 s18, s47, 31 // (op 8830) 
s_lshl_b32 s72, 1, s18 // (op 8830) 
s_add_i32 s73, s72, s19 // (op 8830) 


s_and_b32 s19, s0, s73 // (op 8830) 


s_add_i32 s0, s1, s19 // (op 8830) 


s_ashr_i32 s1, s0, s18 // (op 8830) 


s_lshl_b32 s0, s47, 24 // (op 8830) 
s_ashr_i32 s18, s0, 31 // (op 8830) 

s_xor_b32 s0, s1, s18 // (op 8830) 

s_sub_i32 s1, s0, s18 // (op 8830) 


s_mul_i32 s0, s44, s50 // (op 8830) 

s_mul_i32 s18, s48, s0 // (op 8830) 

s_mul_i32 s0, s1, s18 // (op 8830) 

s_sub_u32 s18, s16, s0 // (op 8830) 

s_mul_hi_i32 s0, s18, s15 // (op 8830) 
s_add_i32 s19, s0, s18 // (op 8830) 


s_ashr_i32 s0, s19, 31 // (op 8830) 
s_cmp_eq_i32 s15, 0 // (op 8830) 
s_mov_b32 s18, scc // (op 8830) 
s_cmp_eq_i32 s18, 0 // (op 8830) 
s_cselect_b32 s50, 0, -1 // (op 8830) 

s_and_b32 s18, s46, 31 // (op 8830) 
s_lshl_b32 s51, 1, s18 // (op 8830) 
s_add_i32 s72, s51, s50 // (op 8830) 


s_and_b32 s50, s0, s72 // (op 8830) 


s_add_i32 s0, s19, s50 // (op 8830) 


s_ashr_i32 s19, s0, s18 // (op 8830) 


s_lshl_b32 s0, s46, 24 // (op 8830) 
s_ashr_i32 s18, s0, 31 // (op 8830) 

s_xor_b32 s0, s19, s18 // (op 8830) 

s_sub_i32 s19, s0, s18 // (op 8830) 


s_cmp_eq_i32 s1, 0 // (op 8830) 
s_mov_b32 s0, scc // (op 8830) 

s_cmp_eq_i32 s0, 0 // (op 8830) 
s_cselect_b32 s1, s19, s49 // (op 8830) 



v_lshl_add_u32 v14, s1, 7, v26 // (op 8830) 


v_mul_lo_u32 v26, v27, s38 // (op 8830) 
v_add_u32 v26, v26, v14 // (op 8830) 


s_mul_i32 s1, s38, s43 // (op 8832) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 8832) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 8832) least significant
s_mul_hi_u32 s19, s38, s42 // (op 8832) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8832) most significant: sum
s_add_u32 s1, s1, s19 // (op 8832) most significant: sum


s_mov_b32 s68, s40 // (op 8832) reg expression
s_mov_b32 s69, s41 // (op 8832) reg expression
s_mov_b32 s70, s0 // (op 8832) reg expression
s_mov_b32 s71, 131072 // (op 8832) call()

v_and_b32 v14, -8, v1 // (op 8958) 
v_sub_u32 v27, v1, v14 // (op 8958) 

v_lshlrev_b32 v14, 4, v27 // (op 8958) 

v_lshrrev_b32 v27, 3, v1 // (op 8958) 
v_lshl_add_u32 v28, v27, 7, v14 // (op 8958) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 8958) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 5 // (op 8553) 
v_mov_b64 v[28:29], s[18:19] // (op 8553) convert


v_mov_b64 v[30:31], v[16:17] // (op 8553) reg expression
s_mov_b32 m0, s0 // (op 8553) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8553) Load value direct to lds




v_mov_b64 v[28:29], s[38:39] // (op 8271) convert

v_mov_b32 v14, v25 // (op 8271) reg expression
buffer_load_ubyte v30, v14, s[68:71], 0 offen offset: 0 // (op 8271) Load value
v_add_u32 v14, v14, v28 // (op 8271) 
buffer_load_ubyte v31, v14, s[68:71], 0 offen offset: 0 // (op 8271) Load value


v_and_b32 v14, -8, v1 // (op 8901) 
v_sub_u32 v27, v1, v14 // (op 8901) 

v_lshlrev_b32 v14, 4, v27 // (op 8901) 

v_lshrrev_b32 v27, 3, v1 // (op 8901) 
v_lshl_add_u32 v28, v27, 7, v14 // (op 8901) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 8901) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 5 // (op 8558) 
v_mov_b64 v[28:29], s[18:19] // (op 8558) convert


v_mov_b64 v[32:33], v[18:19] // (op 8558) 
s_add_u32 s1, 16384, s0 // (op 8558) 
s_mov_b32 m0, s1 // (op 8558) reg expression
buffer_load_dwordx4 v32, s[20:23], 0 offen offset: 0 lds // (op 8558) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 8558) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8558) most significant half

s_add_u32 m0, m0, 4096 // (op 8558) 
buffer_load_dwordx4 v32, s[20:23], 0 offen offset: 0 lds // (op 8558) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 8558) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8558) most significant half

s_add_u32 m0, m0, 4096 // (op 8558) 
buffer_load_dwordx4 v32, s[20:23], 0 offen offset: 0 lds // (op 8558) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 8558) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 8558) most significant half

s_add_u32 m0, m0, 4096 // (op 8558) 
buffer_load_dwordx4 v32, s[20:23], 0 offen offset: 0 lds // (op 8558) Load value direct to lds





v_mov_b64 v[28:29], s[34:35] // (op 8277) convert

v_mov_b32 v14, v15 // (op 8277) 
buffer_load_ubyte v32, v14, s[64:67], 0 offen offset: 0 // (op 8277) Load value
v_add_u32 v14, v14, v28 // (op 8277) 
buffer_load_ubyte v33, v14, s[64:67], 0 offen offset: 0 // (op 8277) Load value


v_and_b32 v14, -8, v1 // (op 9185) 
v_sub_u32 v27, v1, v14 // (op 9185) 

v_lshlrev_b32 v14, 4, v27 // (op 9185) 

v_lshrrev_b32 v27, 3, v1 // (op 9185) 
v_lshl_add_u32 v28, v27, 7, v14 // (op 9185) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 9185) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 5 // (op 8563) 
v_mov_b64 v[28:29], s[18:19] // (op 8563) convert


s_lshl_b64 s[18:19], s[28:29], 7 // (op 8563) 
v_mov_b32 v14, s18 // (op 8563) 
v_mov_b32 v27, s19 // (op 8563) 
v_add_co_u32 v34, s[50:51], v16, v14 // (op 8563) least significant half
v_addc_co_u32 v35, s[50:51], v17, v27, s[50:51] // (op 8563) most significant half




s_add_u32 s1, 32768, s0 // (op 8563) 
s_mov_b32 m0, s1 // (op 8563) reg expression
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 8563) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v28 // (op 8563) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8563) most significant half

s_add_u32 m0, m0, 4096 // (op 8563) 
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 8563) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v28 // (op 8563) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8563) most significant half

s_add_u32 m0, m0, 4096 // (op 8563) 
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 8563) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v28 // (op 8563) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 8563) most significant half

s_add_u32 m0, m0, 4096 // (op 8563) 
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 8563) Load value direct to lds






v_mov_b64 v[16:17], s[38:39] // (op 8283) convert

s_lshl_b32 s0, s38, 2 // (op 8283) 
v_add_u32 v14, v25, s0 // (op 8283) 

buffer_load_ubyte v28, v14, s[68:71], 0 offen offset: 0 // (op 8283) Load value
v_add_u32 v14, v14, v16 // (op 8283) 
buffer_load_ubyte v29, v14, s[68:71], 0 offen offset: 0 // (op 8283) Load value



v_and_b32 v14, -8, v1 // (op 9141) 
v_sub_u32 v25, v1, v14 // (op 9141) 

v_lshlrev_b32 v14, 4, v25 // (op 9141) 

v_lshrrev_b32 v25, 3, v1 // (op 9141) 
v_lshl_add_u32 v27, v25, 7, v14 // (op 9141) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 9141) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 5 // (op 8565) 
v_mov_b64 v[16:17], s[18:19] // (op 8565) convert


v_mov_b64 v[34:35], v[18:19] // (op 8565) 
s_add_u32 s1, 49152, s0 // (op 8565) 
s_mov_b32 m0, s1 // (op 8565) reg expression
s_mov_b32 s18, 128 // (op 8565) call()
buffer_load_dwordx4 v34, s[20:23], s18 offen offset: 0 lds // (op 8565) Load value direct to lds
v_add_co_u32 v34, s[50:51], v34, v16 // (op 8565) least significant half
v_addc_co_u32 v35, s[50:51], 0, v35, s[50:51] // (op 8565) most significant half

s_add_u32 m0, m0, 4096 // (op 8565) 
buffer_load_dwordx4 v34, s[20:23], s18 offen offset: 0 lds // (op 8565) Load value direct to lds
v_add_co_u32 v34, s[50:51], v34, v16 // (op 8565) least significant half
v_addc_co_u32 v35, s[50:51], 0, v35, s[50:51] // (op 8565) most significant half

s_add_u32 m0, m0, 4096 // (op 8565) 
buffer_load_dwordx4 v34, s[20:23], s18 offen offset: 0 lds // (op 8565) Load value direct to lds
v_add_co_u32 v34, s[50:51], v34, v16 // (op 8565) least significant half
v_addc_co_u32 v35, s[50:51], 0, v35, s[50:51] // (op 8565) most significant half

s_add_u32 m0, m0, 4096 // (op 8565) 
buffer_load_dwordx4 v34, s[20:23], s18 offen offset: 0 lds // (op 8565) Load value direct to lds







v_mov_b64 v[16:17], s[34:35] // (op 8289) convert

v_mov_b32 v14, v15 // (op 8289) 
buffer_load_ubyte v18, v14, s[64:67], 0 offen offset:4 // (op 8289) Load value
v_add_u32 v14, v14, v16 // (op 8289) 
buffer_load_ubyte v19, v14, s[64:67], 0 offen offset:4 // (op 8289) Load value



v_and_b32 v25, -128, v1 // (op 9074) 
v_sub_u32 v27, v1, v25 // (op 9074) 

v_lshrrev_b32 v25, 7, v1 // (op 9074) 
v_lshl_add_u32 v14, v25, 8, v27 // (op 9074) 





v_add_u32 v25, 65536, v14 // (op 8292) 
s_waitcnt vmcnt(19)
ds_write_b8 v25, v30 // (op 8292) Store local data 

v_add_u32 v25, 65664, v14 // (op 8292) 
s_waitcnt vmcnt(18)
ds_write_b8 v25, v31 // (op 8292) Store local data 


v_and_b32 v25, -4, v1 // (op 9016) 
v_sub_u32 v27, v1, v25 // (op 9016) 

v_lshrrev_b32 v25, 2, v1 // (op 9016) 
v_lshl_add_u32 v14, v25, 3, v27 // (op 9016) 





v_add_u32 v25, 66048, v14 // (op 8294) 
s_waitcnt vmcnt(13)
ds_write_b8 v25, v32 // (op 8294) Store local data 

v_add_u32 v25, 66052, v14 // (op 8294) 
s_waitcnt vmcnt(12)
ds_write_b8 v25, v33 // (op 8294) Store local data 


s_barrier  // (op 9393) 



ds_read_b64_tr_b8 v[34:35], v4 // (op 8329) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:1024 // (op 8329) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:8192 // (op 8329) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:9216 // (op 8329) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 9428) 


v_add_u32 v25, 65536, v5 // (op 8369) 
ds_read_u8 v27, v25 // (op 8369) Load local data 




ds_read_b64_tr_b8 v[42:43], v4 offset:32 // (op 8334) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:1056 // (op 8334) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:8224 // (op 8334) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v4 offset:9248 // (op 8334) Transpose load local data 


v_add_u32 v25, 65568, v5 // (op 8374) 
ds_read_u8 v14, v25 // (op 8374) Load local data 




ds_read_b64_tr_b8 v[50:51], v4 offset:64 // (op 8339) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:1088 // (op 8339) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:8256 // (op 8339) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:9280 // (op 8339) Transpose load local data 


v_add_u32 v25, 65600, v5 // (op 8379) 
ds_read_u8 v15, v25 // (op 8379) Load local data 




ds_read_b64_tr_b8 v[58:59], v4 offset:96 // (op 8344) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v4 offset:1120 // (op 8344) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v4 offset:8288 // (op 8344) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v4 offset:9312 // (op 8344) Transpose load local data 


v_add_u32 v25, 65632, v5 // (op 8384) 
ds_read_u8 v16, v25 // (op 8384) Load local data 

s_barrier  // (op 9377) 



ds_read_b128 v[66:69], v3 offset:16384 // (op 8309) Load local data 
ds_read_b128 v[70:73], v3 offset:16448 // (op 8309) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 9410) 



v_add_u32 v17, 66048, v0 // (op 8349) 
ds_read_u8 v25, v17 // (op 8349) Load local data 




ds_read_b128 v[74:77], v3 offset:20480 // (op 8314) Load local data 
ds_read_b128 v[78:81], v3 offset:20544 // (op 8314) Load local data 



v_add_u32 v17, 66176, v0 // (op 8354) 
ds_read_u8 v82, v17 // (op 8354) Load local data 




ds_read_b128 v[84:87], v3 offset:24576 // (op 8319) Load local data 
ds_read_b128 v[88:91], v3 offset:24640 // (op 8319) Load local data 



v_add_u32 v17, 66304, v0 // (op 8359) 
ds_read_u8 v83, v17 // (op 8359) Load local data 




ds_read_b128 v[92:95], v3 offset:28672 // (op 8324) Load local data 
ds_read_b128 v[96:99], v3 offset:28736 // (op 8324) Load local data 



v_add_u32 v17, 66432, v0 // (op 8364) 
ds_read_u8 v100, v17 // (op 8364) Load local data 

v_and_b32 v17, -8, v1 // (op 8873) 
v_sub_u32 v101, v1, v17 // (op 8873) 

v_lshlrev_b32 v17, 4, v101 // (op 8873) 

v_lshrrev_b32 v101, 3, v1 // (op 8873) 
v_lshl_add_u32 v102, v101, 7, v17 // (op 8873) 


s_nop 0
v_readfirstlane_b32 s0, v102 // (op 8873) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 8930) 
v_sub_u32 v101, v1, v17 // (op 8930) 

v_lshlrev_b32 v17, 4, v101 // (op 8930) 

v_lshrrev_b32 v101, 3, v1 // (op 8930) 
v_lshl_add_u32 v102, v101, 7, v17 // (op 8930) 


s_nop 0
v_readfirstlane_b32 s1, v102 // (op 8930) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -4, v1 // (op 8987) 
v_sub_u32 v101, v1, v17 // (op 8987) 

v_lshrrev_b32 v17, 2, v1 // (op 8987) 
v_lshl_add_u32 v102, v17, 3, v101 // (op 8987) 


v_and_b32 v17, -128, v1 // (op 9045) 
v_sub_u32 v101, v1, v17 // (op 9045) 

v_lshrrev_b32 v17, 7, v1 // (op 9045) 
v_lshl_add_u32 v103, v17, 8, v101 // (op 9045) 


v_and_b32 v17, -8, v1 // (op 9126) 
v_sub_u32 v101, v1, v17 // (op 9126) 

v_lshlrev_b32 v17, 4, v101 // (op 9126) 

v_lshrrev_b32 v101, 3, v1 // (op 9126) 
v_lshl_add_u32 v104, v101, 7, v17 // (op 9126) 


s_nop 0
v_readfirstlane_b32 s18, v104 // (op 9126) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 9170) 
v_sub_u32 v101, v1, v17 // (op 9170) 

v_lshlrev_b32 v17, 4, v101 // (op 9170) 

v_lshrrev_b32 v101, 3, v1 // (op 9170) 
v_lshl_add_u32 v104, v101, 7, v17 // (op 9170) 


s_nop 0
v_readfirstlane_b32 s19, v104 // (op 9170) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -4, v1 // (op 9214) 
v_sub_u32 v101, v1, v17 // (op 9214) 

v_lshrrev_b32 v17, 2, v1 // (op 9214) 
v_lshl_add_u32 v104, v17, 3, v101 // (op 9214) 


v_and_b32 v17, -128, v1 // (op 9243) 
v_sub_u32 v101, v1, v17 // (op 9243) 

v_lshrrev_b32 v17, 7, v1 // (op 9243) 
v_lshl_add_u32 v105, v17, 8, v101 // (op 9243) 


s_mov_b32 s49, 0 // (op 2587) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 2586) 
s_mov_b32 s72, s50 // (op 2586) convert

s_and_b32 s50, s72, 4294967294 // (op 2586) 

s_cmp_lt_u32 s49, s50 // (op 2586) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoop_2586 // (op 2586) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoop_2586 if false)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoop_2586:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[34:41], a[0:3], v25, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 88) 
s_barrier  // (op 9396) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[42:49], a[4:7], v25, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 376) 
s_add_u32 s49, s49, 2 // (op 2588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[50:57], a[8:11], v25, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 552) 
s_lshl_b64 s[50:51], s[28:29], 5 // (op 8525) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[58:65], a[12:15], v25, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 728) 
v_mov_b64 v[106:107], s[50:51] // (op 8525) convert


v_mov_b64 v[108:109], v[22:23] // (op 8525) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 8525) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 8525) 

v_mov_b32 v17, s72 // (op 8525) 
v_mov_b32 v101, s73 // (op 8525) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:81], v[34:41], a[16:19], v82, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1223) 
v_add_co_u32 v110, s[50:51], v108, v17 // (op 8525) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:81], v[42:49], a[20:23], v82, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1177) 
v_addc_co_u32 v111, s[50:51], v109, v101, s[50:51] // (op 8525) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:81], v[50:57], a[24:27], v82, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1131) 
s_mov_b32 m0, s1 // (op 8525) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:81], v[58:65], a[28:31], v82, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1085) 
buffer_load_dwordx4 v110, s[52:55], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v110, s[50:51], v110, v106 // (op 8525) least significant half
v_addc_co_u32 v111, s[50:51], 0, v111, s[50:51] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
buffer_load_dwordx4 v110, s[52:55], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v110, s[50:51], v110, v106 // (op 8525) least significant half
v_addc_co_u32 v111, s[50:51], 0, v111, s[50:51] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
buffer_load_dwordx4 v110, s[52:55], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v110, s[50:51], v110, v106 // (op 8525) least significant half
v_addc_co_u32 v111, s[50:51], 0, v111, s[50:51] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:91], v[34:41], a[32:35], v83, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1993) 
buffer_load_dwordx4 v110, s[52:55], 0 offen offset: 0 lds // (op 8525) Load value direct to lds



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:91], v[42:49], a[36:39], v83, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1947) 
v_mov_b64 v[106:107], s[38:39] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:91], v[50:57], a[40:43], v83, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1901) 
v_mov_b32 v17, v26 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[84:91], v[58:65], a[44:47], v83, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1855) 
s_lshl_b32 s50, s38, 3 // (op 25) 
v_add_u32 v101, v17, s50 // (op 25) 

buffer_load_ubyte v30, v101, s[68:71], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v101, v101, v106 // (op 25) 
buffer_load_ubyte v31, v101, s[68:71], 0 offen offset: 0 // (op 25) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:99], v[34:41], a[48:51], v100, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2763) 
s_barrier  // (op 9380) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:99], v[42:49], a[52:55], v100, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2717) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:99], v[50:57], a[56:59], v100, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2671) 
s_lshl_b64 s[50:51], s[24:25], 5 // (op 8520) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:99], v[58:65], a[60:63], v100, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2625) 
v_mov_b64 v[106:107], s[50:51] // (op 8520) convert


v_mov_b64 v[110:111], v[20:21] // (op 8520) 
v_mov_b64 v[112:113], v[110:111] // (op 8520) 
s_add_u32 s50, 16384, s0 // (op 8520) 
s_mov_b32 m0, s50 // (op 8520) reg expression
s_mov_b32 s51, 256 // (op 8520) call()
buffer_load_dwordx4 v112, s[20:23], s51 offen offset: 0 lds // (op 8520) Load value direct to lds
v_add_co_u32 v112, s[72:73], v112, v106 // (op 8520) least significant half
v_addc_co_u32 v113, s[72:73], 0, v113, s[72:73] // (op 8520) most significant half

s_add_u32 m0, m0, 4096 // (op 8520) 
buffer_load_dwordx4 v112, s[20:23], s51 offen offset: 0 lds // (op 8520) Load value direct to lds
v_add_co_u32 v112, s[72:73], v112, v106 // (op 8520) least significant half
v_addc_co_u32 v113, s[72:73], 0, v113, s[72:73] // (op 8520) most significant half

s_add_u32 m0, m0, 4096 // (op 8520) 
buffer_load_dwordx4 v112, s[20:23], s51 offen offset: 0 lds // (op 8520) Load value direct to lds
v_add_co_u32 v112, s[72:73], v112, v106 // (op 8520) least significant half
v_addc_co_u32 v113, s[72:73], 0, v113, s[72:73] // (op 8520) most significant half

s_add_u32 m0, m0, 4096 // (op 8520) 
buffer_load_dwordx4 v112, s[20:23], s51 offen offset: 0 lds // (op 8520) Load value direct to lds





v_mov_b64 v[106:107], s[34:35] // (op 17) convert

v_mov_b32 v101, v24 // (op 17) 
v_mov_b32 v112, v101 // (op 17) 
buffer_load_ubyte v32, v112, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v112, v112, v106 // (op 17) 
buffer_load_ubyte v33, v112, s[64:67], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 9468) 



v_add_u32 v106, 66560, v105 // (op 255) 
s_waitcnt vmcnt(19)
ds_write_b8 v106, v28 // (op 255) Store local data 

v_add_u32 v106, 66688, v105 // (op 255) 
s_waitcnt vmcnt(18)
ds_write_b8 v106, v29 // (op 255) Store local data 

s_barrier  // (op 9462) 



v_add_u32 v106, 67072, v104 // (op 254) 
s_waitcnt vmcnt(13)
ds_write_b8 v106, v18 // (op 254) Store local data 

v_add_u32 v106, 67076, v104 // (op 254) 
s_waitcnt vmcnt(12)
ds_write_b8 v106, v19 // (op 254) Store local data 

s_barrier  // (op 9451) 



ds_read_b64_tr_b8 v[112:113], v8 offset:32768 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v8 offset:33792 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[116:117], v8 offset:40960 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v8 offset:41984 // (op 247) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 9465) 


v_add_u32 v106, 66560, v9 // (op 251) 
ds_read_u8 v107, v106 // (op 251) Load local data 




ds_read_b64_tr_b8 v[120:121], v8 offset:32800 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v8 offset:33824 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[124:125], v8 offset:40992 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[126:127], v8 offset:42016 // (op 381) Transpose load local data 


v_add_u32 v106, 66592, v9 // (op 385) 
ds_read_u8 v82, v106 // (op 385) Load local data 




ds_read_b64_tr_b8 v[128:129], v8 offset:32832 // (op 557) Transpose load local data 
ds_read_b64_tr_b8 v[130:131], v8 offset:33856 // (op 557) Transpose load local data 
ds_read_b64_tr_b8 v[132:133], v8 offset:41024 // (op 557) Transpose load local data 
ds_read_b64_tr_b8 v[134:135], v8 offset:42048 // (op 557) Transpose load local data 


v_add_u32 v106, 66624, v9 // (op 561) 
ds_read_u8 v83, v106 // (op 561) Load local data 




ds_read_b64_tr_b8 v[136:137], v8 offset:32864 // (op 733) Transpose load local data 
ds_read_b64_tr_b8 v[138:139], v8 offset:33888 // (op 733) Transpose load local data 
ds_read_b64_tr_b8 v[140:141], v8 offset:41056 // (op 733) Transpose load local data 
ds_read_b64_tr_b8 v[142:143], v8 offset:42080 // (op 733) Transpose load local data 


v_add_u32 v106, 66656, v9 // (op 737) 
ds_read_u8 v144, v106 // (op 737) Load local data 

s_barrier  // (op 9445) 
s_barrier  // (op 9509) 



ds_read_b128 v[66:69], v7 offset:49152 // (op 245) Load local data 
ds_read_b128 v[70:73], v7 offset:49216 // (op 245) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 9458) 



v_add_u32 v106, 67072, v6 // (op 249) 
ds_read_u8 v25, v106 // (op 249) Load local data 




ds_read_b128 v[74:77], v7 offset:53248 // (op 1088) Load local data 
ds_read_b128 v[78:81], v7 offset:53312 // (op 1088) Load local data 



v_add_u32 v106, 67200, v6 // (op 1091) 
ds_read_u8 v14, v106 // (op 1091) Load local data 




ds_read_b128 v[84:87], v7 offset:57344 // (op 1858) Load local data 
ds_read_b128 v[88:91], v7 offset:57408 // (op 1858) Load local data 



v_add_u32 v106, 67328, v6 // (op 1861) 
ds_read_u8 v15, v106 // (op 1861) Load local data 




ds_read_b128 v[92:95], v7 offset:61440 // (op 2628) Load local data 
ds_read_b128 v[96:99], v7 offset:61504 // (op 2628) Load local data 



v_add_u32 v106, 67456, v6 // (op 2631) 
ds_read_u8 v100, v106 // (op 2631) Load local data 

s_barrier  // (op 9543) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[112:119], a[0:3], v25, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 252) 
s_barrier  // (op 9455) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[120:127], a[4:7], v25, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 377) 
s_lshl_b64 s[50:51], s[28:29], 5 // (op 8548) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[128:135], a[8:11], v25, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 553) 
v_mov_b64 v[146:147], s[50:51] // (op 8548) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[136:143], a[12:15], v25, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 729) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 8548) 
s_mul_i32 s73, 3, s51 // (op 8548) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 8548) least significant
s_mul_hi_u32 s74, 3, s50 // (op 8548) most significant: high of low * low
s_add_u32 s73, s73, s74 // (op 8548) most significant: sum


v_mov_b32 v106, s72 // (op 8548) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:81], v[112:119], a[16:19], v14, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1224) 
v_mov_b32 v145, s73 // (op 8548) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:81], v[120:127], a[20:23], v14, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1178) 
v_add_co_u32 v148, s[50:51], v108, v106 // (op 8548) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:81], v[128:135], a[24:27], v14, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1132) 
v_addc_co_u32 v149, s[50:51], v109, v145, s[50:51] // (op 8548) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:81], v[136:143], a[28:31], v14, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1086) 
s_add_u32 s50, 32768, s19 // (op 8548) 
s_mov_b32 m0, s50 // (op 8548) reg expression
buffer_load_dwordx4 v148, s[52:55], 0 offen offset: 0 lds // (op 8548) Load value direct to lds
v_add_co_u32 v148, s[72:73], v148, v146 // (op 8548) least significant half
v_addc_co_u32 v149, s[72:73], 0, v149, s[72:73] // (op 8548) most significant half

s_add_u32 m0, m0, 4096 // (op 8548) 
buffer_load_dwordx4 v148, s[52:55], 0 offen offset: 0 lds // (op 8548) Load value direct to lds
v_add_co_u32 v148, s[72:73], v148, v146 // (op 8548) least significant half
v_addc_co_u32 v149, s[72:73], 0, v149, s[72:73] // (op 8548) most significant half

s_add_u32 m0, m0, 4096 // (op 8548) 
buffer_load_dwordx4 v148, s[52:55], 0 offen offset: 0 lds // (op 8548) Load value direct to lds
v_add_co_u32 v148, s[72:73], v148, v146 // (op 8548) least significant half
v_addc_co_u32 v149, s[72:73], 0, v149, s[72:73] // (op 8548) most significant half

s_add_u32 m0, m0, 4096 // (op 8548) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:91], v[112:119], a[32:35], v15, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1994) 
buffer_load_dwordx4 v148, s[52:55], 0 offen offset: 0 lds // (op 8548) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:91], v[120:127], a[36:39], v15, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1948) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:91], v[128:135], a[40:43], v15, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1902) 
v_mov_b64 v[108:109], s[38:39] // (op 258) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[84:91], v[136:143], a[44:47], v15, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1856) 
s_lshl_b32 s50, s38, 2 // (op 258) 
s_mul_i32 s51, 3, s50 // (op 258) 

v_add_u32 v106, v17, s51 // (op 258) 

buffer_load_ubyte v28, v106, s[68:71], 0 offen offset: 0 // (op 258) Load value
v_add_u32 v106, v106, v108 // (op 258) 
buffer_load_ubyte v29, v106, s[68:71], 0 offen offset: 0 // (op 258) Load value


s_lshl_b64 s[50:51], s[28:29], 7 // (op 8694) 

s_lshl_b64 s[72:73], s[50:51], 1 // (op 8694) 

s_lshl_b32 s50, s38, 3 // (op 8822) 
v_add_u32 v26, v26, s50 // (op 8822) 

v_mov_b32 v17, s72 // (op 8694) 
v_mov_b32 v106, s73 // (op 8694) 
v_add_co_u32 v22, s[50:51], v22, v17 // (op 8694) least significant half
v_addc_co_u32 v23, s[50:51], v23, v106, s[50:51] // (op 8694) most significant half




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:99], v[112:119], a[48:51], v100, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2764) 
s_barrier  // (op 9448) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:99], v[120:127], a[52:55], v100, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2718) 
s_lshl_b64 s[50:51], s[24:25], 5 // (op 8543) 

v_mov_b64 v[106:107], s[50:51] // (op 8543) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:99], v[128:135], a[56:59], v100, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2672) 
v_mov_b64 v[108:109], v[110:111] // (op 8543) 

s_add_u32 s50, 49152, s18 // (op 8543) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:99], v[136:143], a[60:63], v100, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2626) 
s_mov_b32 m0, s50 // (op 8543) reg expression


s_mov_b32 s51, 384 // (op 8543) call()
buffer_load_dwordx4 v108, s[20:23], s51 offen offset: 0 lds // (op 8543) Load value direct to lds
v_add_co_u32 v108, s[72:73], v108, v106 // (op 8543) least significant half
v_addc_co_u32 v109, s[72:73], 0, v109, s[72:73] // (op 8543) most significant half

s_add_u32 m0, m0, 4096 // (op 8543) 
buffer_load_dwordx4 v108, s[20:23], s51 offen offset: 0 lds // (op 8543) Load value direct to lds
v_add_co_u32 v108, s[72:73], v108, v106 // (op 8543) least significant half
v_addc_co_u32 v109, s[72:73], 0, v109, s[72:73] // (op 8543) most significant half

s_add_u32 m0, m0, 4096 // (op 8543) 
buffer_load_dwordx4 v108, s[20:23], s51 offen offset: 0 lds // (op 8543) Load value direct to lds
v_add_co_u32 v108, s[72:73], v108, v106 // (op 8543) least significant half
v_addc_co_u32 v109, s[72:73], 0, v109, s[72:73] // (op 8543) most significant half

s_add_u32 m0, m0, 4096 // (op 8543) 
buffer_load_dwordx4 v108, s[20:23], s51 offen offset: 0 lds // (op 8543) Load value direct to lds






v_add_co_u32 v20, vcc, 256, v20 // (op 8630) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 8630) most significant half
v_mov_b64 v[106:107], s[34:35] // (op 257) convert

v_mov_b32 v17, v101 // (op 257) 
buffer_load_ubyte v18, v17, s[64:67], 0 offen offset:12 // (op 257) Load value
v_add_u32 v17, v17, v106 // (op 257) 
buffer_load_ubyte v19, v17, s[64:67], 0 offen offset:12 // (op 257) Load value



s_barrier  // (op 9431) 
v_add_u32 v24, 8, v24 // (op 8758) 



v_add_u32 v17, 65536, v103 // (op 30) 
s_waitcnt vmcnt(19)
ds_write_b8 v17, v30 // (op 30) Store local data 

v_add_u32 v17, 65664, v103 // (op 30) 
s_waitcnt vmcnt(18)
ds_write_b8 v17, v31 // (op 30) Store local data 

s_barrier  // (op 9413) 



v_add_u32 v17, 66048, v102 // (op 21) 
s_waitcnt vmcnt(13)
ds_write_b8 v17, v32 // (op 21) Store local data 

v_add_u32 v17, 66052, v102 // (op 21) 
s_waitcnt vmcnt(12)
ds_write_b8 v17, v33 // (op 21) Store local data 

s_barrier  // (op 9402) 



ds_read_b64_tr_b8 v[34:35], v4 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:1024 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:8192 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:9216 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 9438) 


v_add_u32 v17, 65536, v5 // (op 85) 
ds_read_u8 v27, v17 // (op 85) Load local data 




ds_read_b64_tr_b8 v[42:43], v4 offset:32 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:1056 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:8224 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v4 offset:9248 // (op 371) Transpose load local data 


v_add_u32 v17, 65568, v5 // (op 375) 
ds_read_u8 v14, v17 // (op 375) Load local data 




ds_read_b64_tr_b8 v[50:51], v4 offset:64 // (op 547) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:1088 // (op 547) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:8256 // (op 547) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:9280 // (op 547) Transpose load local data 


v_add_u32 v17, 65600, v5 // (op 551) 
ds_read_u8 v15, v17 // (op 551) Load local data 




ds_read_b64_tr_b8 v[58:59], v4 offset:96 // (op 723) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v4 offset:1120 // (op 723) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v4 offset:8288 // (op 723) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v4 offset:9312 // (op 723) Transpose load local data 


v_add_u32 v17, 65632, v5 // (op 727) 
ds_read_u8 v16, v17 // (op 727) Load local data 

s_barrier  // (op 9386) 



ds_read_b128 v[66:69], v3 offset:16384 // (op 70) Load local data 
ds_read_b128 v[70:73], v3 offset:16448 // (op 70) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 9420) 



v_add_u32 v17, 66048, v0 // (op 80) 
ds_read_u8 v25, v17 // (op 80) Load local data 




ds_read_b128 v[74:77], v3 offset:20480 // (op 1080) Load local data 
ds_read_b128 v[78:81], v3 offset:20544 // (op 1080) Load local data 



v_add_u32 v17, 66176, v0 // (op 1083) 
ds_read_u8 v82, v17 // (op 1083) Load local data 




ds_read_b128 v[84:87], v3 offset:24576 // (op 1850) Load local data 
ds_read_b128 v[88:91], v3 offset:24640 // (op 1850) Load local data 



v_add_u32 v17, 66304, v0 // (op 1853) 
ds_read_u8 v83, v17 // (op 1853) Load local data 




ds_read_b128 v[92:95], v3 offset:28672 // (op 2620) Load local data 
ds_read_b128 v[96:99], v3 offset:28736 // (op 2620) Load local data 



v_add_u32 v17, 66432, v0 // (op 2623) 
ds_read_u8 v100, v17 // (op 2623) Load local data 

s_ashr_i64 s[50:51], s[32:33], 7 // (op 2586) 
s_mov_b32 s72, s50 // (op 2586) convert

s_and_b32 s50, s72, 4294967294 // (op 2586) 

s_cmp_lt_u32 s49, s50 // (op 2586) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoop_2586 // (op 2586) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoop_2586 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoop_2586:





















s_barrier  // (op 9527) 
s_mul_hi_i32 s0, s16, s11 // (op 8578) 
v_lshrrev_b32 v17, 3, v1 // (op 8578) 
s_add_i32 s1, s0, s16 // (op 8578) 

v_and_b32 v14, -8, v1 // (op 8578) 
s_ashr_i32 s0, s1, 31 // (op 8578) 
v_sub_u32 v15, v1, v14 // (op 8578) 

s_cmp_eq_i32 s11, 0 // (op 8578) 
s_mov_b32 s18, scc // (op 8578) 
v_lshlrev_b32 v14, 4, v15 // (op 8578) 

s_cmp_eq_i32 s18, 0 // (op 8578) 
s_cselect_b32 s19, 0, -1 // (op 8578) 

s_and_b32 s18, s12, 31 // (op 8578) 
s_lshl_b32 s49, 1, s18 // (op 8578) 
s_add_i32 s50, s49, s19 // (op 8578) 


s_and_b32 s19, s0, s50 // (op 8578) 


s_add_i32 s0, s1, s19 // (op 8578) 


s_ashr_i32 s1, s0, s18 // (op 8578) 


s_lshl_b32 s0, s12, 24 // (op 8578) 
s_ashr_i32 s18, s0, 31 // (op 8578) 

s_xor_b32 s0, s1, s18 // (op 8578) 

s_sub_i32 s1, s0, s18 // (op 8578) 


s_mul_hi_i32 s0, s1, s13 // (op 8578) 
s_add_i32 s18, s0, s1 // (op 8578) 

s_ashr_i32 s0, s18, 31 // (op 8578) 
s_cmp_eq_i32 s13, 0 // (op 8578) 
s_mov_b32 s19, scc // (op 8578) 
s_cmp_eq_i32 s19, 0 // (op 8578) 
s_cselect_b32 s49, 0, -1 // (op 8578) 

s_and_b32 s19, s14, 31 // (op 8578) 
s_lshl_b32 s50, 1, s19 // (op 8578) 
s_add_i32 s51, s50, s49 // (op 8578) 


s_and_b32 s49, s0, s51 // (op 8578) 


s_add_i32 s0, s18, s49 // (op 8578) 


s_ashr_i32 s18, s0, s19 // (op 8578) 


s_lshl_b32 s0, s14, 24 // (op 8578) 
s_ashr_i32 s19, s0, 31 // (op 8578) 

s_xor_b32 s0, s18, s19 // (op 8578) 

s_sub_i32 s18, s0, s19 // (op 8578) 


s_mul_i32 s0, s1, s44 // (op 8578) 

s_sub_u32 s1, s16, s0 // (op 8578) 

s_mul_i32 s0, s18, s44 // (op 8578) 
s_add_u32 s0, s0, s1 // (op 8578) 


s_mul_hi_i32 s1, s16, s10 // (op 8578) 
s_add_i32 s18, s1, s16 // (op 8578) 

s_ashr_i32 s1, s18, 31 // (op 8578) 
s_cmp_eq_i32 s10, 0 // (op 8578) 
s_mov_b32 s19, scc // (op 8578) 
s_cmp_eq_i32 s19, 0 // (op 8578) 
s_cselect_b32 s49, 0, -1 // (op 8578) 

s_and_b32 s19, s47, 31 // (op 8578) 
s_lshl_b32 s50, 1, s19 // (op 8578) 
s_add_i32 s51, s50, s49 // (op 8578) 


s_and_b32 s49, s1, s51 // (op 8578) 


s_add_i32 s1, s18, s49 // (op 8578) 


s_ashr_i32 s18, s1, s19 // (op 8578) 


s_lshl_b32 s1, s47, 24 // (op 8578) 
s_ashr_i32 s19, s1, 31 // (op 8578) 

s_xor_b32 s1, s18, s19 // (op 8578) 

s_sub_i32 s18, s1, s19 // (op 8578) 


s_add_u32 s50, s6, 128 // (op 8578) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8578) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 8578) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 8578) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 8578) 

s_mul_i32 s1, s44, s50 // (op 8578) 

s_mul_i32 s19, s48, s1 // (op 8578) 

s_mul_i32 s1, s18, s19 // (op 8578) 

s_sub_u32 s19, s16, s1 // (op 8578) 

s_mov_b32 s1, s19 // (op 8578) 
s_mul_hi_i32 s49, s1, s15 // (op 8578) 
s_add_i32 s50, s49, s1 // (op 8578) 


s_ashr_i32 s1, s50, 31 // (op 8578) 
s_cmp_eq_i32 s15, 0 // (op 8578) 
s_mov_b32 s49, scc // (op 8578) 
s_cmp_eq_i32 s49, 0 // (op 8578) 
s_cselect_b32 s51, 0, -1 // (op 8578) 

s_and_b32 s49, s46, 31 // (op 8578) 
s_lshl_b32 s72, 1, s49 // (op 8578) 
s_add_i32 s73, s72, s51 // (op 8578) 


s_and_b32 s51, s1, s73 // (op 8578) 


s_add_i32 s1, s50, s51 // (op 8578) 


s_ashr_i32 s50, s1, s49 // (op 8578) 


s_lshl_b32 s1, s46, 24 // (op 8578) 
s_ashr_i32 s49, s1, 31 // (op 8578) 

s_xor_b32 s1, s50, s49 // (op 8578) 

s_sub_i32 s50, s1, s49 // (op 8578) 


s_mul_i32 s1, s45, s44 // (op 8578) 
s_add_u32 s72, s8, 128 // (op 8578) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 8578) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 8578) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 8578) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 8578) 

s_sub_i32 s49, s72, s1 // (op 8578) 

s_mul_i32 s51, s50, s49 // (op 8578) 


s_sub_u32 s49, s19, s51 // (op 8578) 


s_add_u32 s19, s49, s1 // (op 8578) 


s_cmp_eq_i32 s18, 0 // (op 8578) 
s_mov_b32 s1, scc // (op 8578) 

s_cmp_eq_i32 s1, 0 // (op 8578) 
s_cselect_b32 s18, s19, s0 // (op 8578) 



v_lshl_add_u32 v15, s18, 7, v17 // (op 8578) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 8578) 
s_lshr_b32 s18, s0, 1 // (op 8578) 

v_lshl_add_u32 v17, s18, 8, v14 // (op 8578) 


v_mov_b32 v27, 0 // (op 8578) convert
v_mov_b32 v26, v17 // (op 8578) convert

v_mul_lo_u32 v21, v15, s25 // (op 8578) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 8578) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 8578) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8578) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v26 // (op 8578) least significant half
v_addc_co_u32 v21, s[0:1], v21, v27, s[0:1] // (op 8578) most significant half



s_mul_i32 s1, s24, s9 // (op 8580) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 8580) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8580) least significant
s_mul_hi_u32 s19, s24, s8 // (op 8580) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8580) most significant: sum
s_add_u32 s1, s1, s19 // (op 8580) most significant: sum


s_mov_b32 s20, s26 // (op 8580) reg expression
s_mov_b32 s21, s27 // (op 8580) reg expression
s_mov_b32 s22, s0 // (op 8580) reg expression
s_mov_b32 s23, 131072 // (op 8580) call()

s_mul_hi_i32 s0, s16, s11 // (op 8660) 
v_and_b32 v17, -8, v1 // (op 8660) 
s_add_i32 s1, s0, s16 // (op 8660) 

v_sub_u32 v14, v1, v17 // (op 8660) 

s_ashr_i32 s0, s1, 31 // (op 8660) 
v_lshlrev_b32 v15, 4, v14 // (op 8660) 

s_cmp_eq_i32 s11, 0 // (op 8660) 
s_mov_b32 s18, scc // (op 8660) 
v_lshrrev_b32 v14, 3, v1 // (op 8660) 
s_cmp_eq_i32 s18, 0 // (op 8660) 
s_cselect_b32 s19, 0, -1 // (op 8660) 

s_and_b32 s18, s12, 31 // (op 8660) 
s_lshl_b32 s49, 1, s18 // (op 8660) 
s_add_i32 s50, s49, s19 // (op 8660) 


s_and_b32 s19, s0, s50 // (op 8660) 


s_add_i32 s0, s1, s19 // (op 8660) 


s_ashr_i32 s1, s0, s18 // (op 8660) 


s_lshl_b32 s0, s12, 24 // (op 8660) 
s_ashr_i32 s18, s0, 31 // (op 8660) 

s_xor_b32 s0, s1, s18 // (op 8660) 

s_sub_i32 s1, s0, s18 // (op 8660) 


s_mul_hi_i32 s0, s1, s13 // (op 8660) 
s_add_i32 s18, s0, s1 // (op 8660) 

s_ashr_i32 s0, s18, 31 // (op 8660) 
s_cmp_eq_i32 s13, 0 // (op 8660) 
s_mov_b32 s19, scc // (op 8660) 
s_cmp_eq_i32 s19, 0 // (op 8660) 
s_cselect_b32 s49, 0, -1 // (op 8660) 

s_and_b32 s19, s14, 31 // (op 8660) 
s_lshl_b32 s50, 1, s19 // (op 8660) 
s_add_i32 s51, s50, s49 // (op 8660) 


s_and_b32 s49, s0, s51 // (op 8660) 


s_add_i32 s0, s18, s49 // (op 8660) 


s_ashr_i32 s18, s0, s19 // (op 8660) 


s_lshl_b32 s0, s14, 24 // (op 8660) 
s_ashr_i32 s19, s0, 31 // (op 8660) 

s_xor_b32 s0, s18, s19 // (op 8660) 

s_sub_i32 s18, s0, s19 // (op 8660) 


s_add_u32 s50, s6, 128 // (op 8660) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8660) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 8660) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 8660) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 8660) 

s_mul_i32 s0, s18, s50 // (op 8660) 

s_sub_i32 s49, s1, s0 // (op 8660) 


s_mul_hi_i32 s0, s16, s10 // (op 8660) 
s_add_i32 s1, s0, s16 // (op 8660) 

s_ashr_i32 s0, s1, 31 // (op 8660) 
s_cmp_eq_i32 s10, 0 // (op 8660) 
s_mov_b32 s18, scc // (op 8660) 
s_cmp_eq_i32 s18, 0 // (op 8660) 
s_cselect_b32 s19, 0, -1 // (op 8660) 

s_and_b32 s18, s47, 31 // (op 8660) 
s_lshl_b32 s72, 1, s18 // (op 8660) 
s_add_i32 s73, s72, s19 // (op 8660) 


s_and_b32 s19, s0, s73 // (op 8660) 


s_add_i32 s0, s1, s19 // (op 8660) 


s_ashr_i32 s1, s0, s18 // (op 8660) 


s_lshl_b32 s0, s47, 24 // (op 8660) 
s_ashr_i32 s18, s0, 31 // (op 8660) 

s_xor_b32 s0, s1, s18 // (op 8660) 

s_sub_i32 s1, s0, s18 // (op 8660) 


s_mul_i32 s0, s44, s50 // (op 8660) 

s_mul_i32 s18, s48, s0 // (op 8660) 

s_mul_i32 s0, s1, s18 // (op 8660) 

s_sub_u32 s18, s16, s0 // (op 8660) 

s_mul_hi_i32 s0, s18, s15 // (op 8660) 
s_add_i32 s19, s0, s18 // (op 8660) 


s_ashr_i32 s0, s19, 31 // (op 8660) 
s_cmp_eq_i32 s15, 0 // (op 8660) 
s_mov_b32 s18, scc // (op 8660) 
s_cmp_eq_i32 s18, 0 // (op 8660) 
s_cselect_b32 s50, 0, -1 // (op 8660) 

s_and_b32 s18, s46, 31 // (op 8660) 
s_lshl_b32 s51, 1, s18 // (op 8660) 
s_add_i32 s72, s51, s50 // (op 8660) 


s_and_b32 s50, s0, s72 // (op 8660) 


s_add_i32 s0, s19, s50 // (op 8660) 


s_ashr_i32 s19, s0, s18 // (op 8660) 


s_lshl_b32 s0, s46, 24 // (op 8660) 
s_ashr_i32 s18, s0, 31 // (op 8660) 

s_xor_b32 s0, s19, s18 // (op 8660) 

s_sub_i32 s19, s0, s18 // (op 8660) 


s_cmp_eq_i32 s1, 0 // (op 8660) 
s_mov_b32 s0, scc // (op 8660) 

s_cmp_eq_i32 s0, 0 // (op 8660) 
s_cselect_b32 s1, s19, s49 // (op 8660) 



v_lshl_add_u32 v17, s1, 7, v15 // (op 8660) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 8660) 
v_mov_b32 v27, 0 // (op 8660) convert
v_mov_b32 v26, v17 // (op 8660) convert

s_lshr_b32 s18, s0, 1 // (op 8660) 

v_lshl_add_u32 v15, s18, 8, v14 // (op 8660) 


v_mul_lo_u32 v23, v15, s29 // (op 8660) most significant: low of low * high
v_mul_hi_u32 v14, v15, s28 // (op 8660) most significant: high of low * low
v_mul_lo_u32 v22, v15, s28 // (op 8660) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 8660) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v26 // (op 8660) least significant half
v_addc_co_u32 v23, s[0:1], v23, v27, s[0:1] // (op 8660) most significant half



s_mul_i32 s1, s28, s33 // (op 8662) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 8662) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8662) least significant
s_mul_hi_u32 s19, s28, s32 // (op 8662) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8662) most significant: sum
s_add_u32 s1, s1, s19 // (op 8662) most significant: sum


s_mov_b32 s52, s30 // (op 8662) reg expression
s_mov_b32 s53, s31 // (op 8662) reg expression
s_mov_b32 s54, s0 // (op 8662) reg expression
s_mov_b32 s55, 131072 // (op 8662) call()

s_mul_hi_i32 s0, s16, s11 // (op 8724) 
v_lshrrev_b32 v17, 2, v1 // (op 8724) 
s_add_i32 s1, s0, s16 // (op 8724) 

v_lshlrev_b32 v24, 1, v17 // (op 8724) 

s_ashr_i32 s0, s1, 31 // (op 8724) 
v_and_b32 v17, -4, v1 // (op 8724) 
s_cmp_eq_i32 s11, 0 // (op 8724) 
s_mov_b32 s18, scc // (op 8724) 
v_sub_u32 v14, v1, v17 // (op 8724) 

s_cmp_eq_i32 s18, 0 // (op 8724) 
s_cselect_b32 s19, 0, -1 // (op 8724) 

s_and_b32 s18, s12, 31 // (op 8724) 
s_lshl_b32 s49, 1, s18 // (op 8724) 
s_add_i32 s50, s49, s19 // (op 8724) 


s_and_b32 s19, s0, s50 // (op 8724) 


s_add_i32 s0, s1, s19 // (op 8724) 


s_ashr_i32 s1, s0, s18 // (op 8724) 


s_lshl_b32 s0, s12, 24 // (op 8724) 
s_ashr_i32 s18, s0, 31 // (op 8724) 

s_xor_b32 s0, s1, s18 // (op 8724) 

s_sub_i32 s1, s0, s18 // (op 8724) 


s_mul_hi_i32 s0, s1, s13 // (op 8724) 
s_add_i32 s18, s0, s1 // (op 8724) 

s_ashr_i32 s0, s18, 31 // (op 8724) 
s_cmp_eq_i32 s13, 0 // (op 8724) 
s_mov_b32 s19, scc // (op 8724) 
s_cmp_eq_i32 s19, 0 // (op 8724) 
s_cselect_b32 s49, 0, -1 // (op 8724) 

s_and_b32 s19, s14, 31 // (op 8724) 
s_lshl_b32 s50, 1, s19 // (op 8724) 
s_add_i32 s51, s50, s49 // (op 8724) 


s_and_b32 s49, s0, s51 // (op 8724) 


s_add_i32 s0, s18, s49 // (op 8724) 


s_ashr_i32 s18, s0, s19 // (op 8724) 


s_lshl_b32 s0, s14, 24 // (op 8724) 
s_ashr_i32 s19, s0, 31 // (op 8724) 

s_xor_b32 s0, s18, s19 // (op 8724) 

s_sub_i32 s18, s0, s19 // (op 8724) 


s_mul_i32 s0, s1, s44 // (op 8724) 

s_sub_u32 s1, s16, s0 // (op 8724) 

s_mul_i32 s0, s18, s44 // (op 8724) 
s_add_u32 s0, s0, s1 // (op 8724) 


s_mul_hi_i32 s1, s16, s10 // (op 8724) 
s_add_i32 s18, s1, s16 // (op 8724) 

s_ashr_i32 s1, s18, 31 // (op 8724) 
s_cmp_eq_i32 s10, 0 // (op 8724) 
s_mov_b32 s19, scc // (op 8724) 
s_cmp_eq_i32 s19, 0 // (op 8724) 
s_cselect_b32 s49, 0, -1 // (op 8724) 

s_and_b32 s19, s47, 31 // (op 8724) 
s_lshl_b32 s50, 1, s19 // (op 8724) 
s_add_i32 s51, s50, s49 // (op 8724) 


s_and_b32 s49, s1, s51 // (op 8724) 


s_add_i32 s1, s18, s49 // (op 8724) 


s_ashr_i32 s18, s1, s19 // (op 8724) 


s_lshl_b32 s1, s47, 24 // (op 8724) 
s_ashr_i32 s19, s1, 31 // (op 8724) 

s_xor_b32 s1, s18, s19 // (op 8724) 

s_sub_i32 s18, s1, s19 // (op 8724) 


s_add_u32 s50, s6, 128 // (op 8724) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8724) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 8724) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 8724) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 8724) 

s_mul_i32 s1, s44, s50 // (op 8724) 

s_mul_i32 s19, s48, s1 // (op 8724) 

s_mul_i32 s1, s18, s19 // (op 8724) 

s_sub_u32 s19, s16, s1 // (op 8724) 

s_mov_b32 s1, s19 // (op 8724) 
s_mul_hi_i32 s49, s1, s15 // (op 8724) 
s_add_i32 s50, s49, s1 // (op 8724) 


s_ashr_i32 s1, s50, 31 // (op 8724) 
s_cmp_eq_i32 s15, 0 // (op 8724) 
s_mov_b32 s49, scc // (op 8724) 
s_cmp_eq_i32 s49, 0 // (op 8724) 
s_cselect_b32 s51, 0, -1 // (op 8724) 

s_and_b32 s49, s46, 31 // (op 8724) 
s_lshl_b32 s72, 1, s49 // (op 8724) 
s_add_i32 s73, s72, s51 // (op 8724) 


s_and_b32 s51, s1, s73 // (op 8724) 


s_add_i32 s1, s50, s51 // (op 8724) 


s_ashr_i32 s50, s1, s49 // (op 8724) 


s_lshl_b32 s1, s46, 24 // (op 8724) 
s_ashr_i32 s49, s1, 31 // (op 8724) 

s_xor_b32 s1, s50, s49 // (op 8724) 

s_sub_i32 s50, s1, s49 // (op 8724) 


s_mul_i32 s1, s45, s44 // (op 8724) 
s_add_u32 s72, s8, 128 // (op 8724) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 8724) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 8724) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 8724) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 7 // (op 8724) 

s_sub_i32 s49, s72, s1 // (op 8724) 

s_mul_i32 s51, s50, s49 // (op 8724) 


s_sub_u32 s49, s19, s51 // (op 8724) 


s_add_u32 s19, s49, s1 // (op 8724) 


s_cmp_eq_i32 s18, 0 // (op 8724) 
s_mov_b32 s1, scc // (op 8724) 

s_cmp_eq_i32 s1, 0 // (op 8724) 
s_cselect_b32 s18, s19, s0 // (op 8724) 



v_lshl_add_u32 v15, s18, 7, v24 // (op 8724) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 8724) 
s_lshr_b32 s18, s0, 1 // (op 8724) 

v_lshl_add_u32 v17, s18, 3, v14 // (op 8724) 


v_mul_lo_u32 v14, v15, s34 // (op 8724) 
v_add_u32 v14, v14, v17 // (op 8724) 


s_mul_i32 s1, s34, s9 // (op 8726) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 8726) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8726) least significant
s_mul_hi_u32 s19, s34, s8 // (op 8726) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8726) most significant: sum
s_add_u32 s1, s1, s19 // (op 8726) most significant: sum


s_mov_b32 s64, s36 // (op 8726) reg expression
s_mov_b32 s65, s37 // (op 8726) reg expression
s_mov_b32 s66, s0 // (op 8726) reg expression
s_mov_b32 s67, 131072 // (op 8726) call()

s_mul_hi_i32 s0, s16, s11 // (op 8788) 
v_and_b32 v15, -128, v1 // (op 8788) 
s_add_i32 s1, s0, s16 // (op 8788) 

v_sub_u32 v17, v1, v15 // (op 8788) 

s_ashr_i32 s0, s1, 31 // (op 8788) 
v_lshrrev_b32 v15, 7, v1 // (op 8788) 
s_cmp_eq_i32 s11, 0 // (op 8788) 
s_mov_b32 s18, scc // (op 8788) 
v_lshlrev_b32 v24, 1, v15 // (op 8788) 

s_cmp_eq_i32 s18, 0 // (op 8788) 
s_cselect_b32 s19, 0, -1 // (op 8788) 

s_and_b32 s18, s12, 31 // (op 8788) 
s_lshl_b32 s49, 1, s18 // (op 8788) 
s_add_i32 s50, s49, s19 // (op 8788) 


s_and_b32 s19, s0, s50 // (op 8788) 


s_add_i32 s0, s1, s19 // (op 8788) 


s_ashr_i32 s1, s0, s18 // (op 8788) 


s_lshl_b32 s0, s12, 24 // (op 8788) 
s_ashr_i32 s18, s0, 31 // (op 8788) 

s_xor_b32 s0, s1, s18 // (op 8788) 

s_sub_i32 s1, s0, s18 // (op 8788) 


s_mul_hi_i32 s0, s1, s13 // (op 8788) 
s_add_i32 s18, s0, s1 // (op 8788) 

s_ashr_i32 s0, s18, 31 // (op 8788) 
s_cmp_eq_i32 s13, 0 // (op 8788) 
s_mov_b32 s19, scc // (op 8788) 
s_cmp_eq_i32 s19, 0 // (op 8788) 
s_cselect_b32 s49, 0, -1 // (op 8788) 

s_and_b32 s19, s14, 31 // (op 8788) 
s_lshl_b32 s50, 1, s19 // (op 8788) 
s_add_i32 s51, s50, s49 // (op 8788) 


s_and_b32 s49, s0, s51 // (op 8788) 


s_add_i32 s0, s18, s49 // (op 8788) 


s_ashr_i32 s18, s0, s19 // (op 8788) 


s_lshl_b32 s0, s14, 24 // (op 8788) 
s_ashr_i32 s19, s0, 31 // (op 8788) 

s_xor_b32 s0, s18, s19 // (op 8788) 

s_sub_i32 s18, s0, s19 // (op 8788) 


s_add_u32 s50, s6, 128 // (op 8788) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 8788) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 8788) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 8788) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 7 // (op 8788) 

s_mul_i32 s0, s18, s50 // (op 8788) 

s_sub_i32 s49, s1, s0 // (op 8788) 


s_mul_hi_i32 s0, s16, s10 // (op 8788) 
s_add_i32 s1, s0, s16 // (op 8788) 

s_ashr_i32 s0, s1, 31 // (op 8788) 
s_cmp_eq_i32 s10, 0 // (op 8788) 
s_mov_b32 s18, scc // (op 8788) 
s_cmp_eq_i32 s18, 0 // (op 8788) 
s_cselect_b32 s19, 0, -1 // (op 8788) 

s_and_b32 s18, s47, 31 // (op 8788) 
s_lshl_b32 s72, 1, s18 // (op 8788) 
s_add_i32 s73, s72, s19 // (op 8788) 


s_and_b32 s19, s0, s73 // (op 8788) 


s_add_i32 s0, s1, s19 // (op 8788) 


s_ashr_i32 s1, s0, s18 // (op 8788) 


s_lshl_b32 s0, s47, 24 // (op 8788) 
s_ashr_i32 s18, s0, 31 // (op 8788) 

s_xor_b32 s0, s1, s18 // (op 8788) 

s_sub_i32 s1, s0, s18 // (op 8788) 


s_mul_i32 s0, s44, s50 // (op 8788) 

s_mul_i32 s18, s48, s0 // (op 8788) 

s_mul_i32 s0, s1, s18 // (op 8788) 

s_sub_u32 s18, s16, s0 // (op 8788) 

s_mul_hi_i32 s0, s18, s15 // (op 8788) 
s_add_i32 s19, s0, s18 // (op 8788) 


s_ashr_i32 s0, s19, 31 // (op 8788) 
s_cmp_eq_i32 s15, 0 // (op 8788) 
s_mov_b32 s18, scc // (op 8788) 
s_cmp_eq_i32 s18, 0 // (op 8788) 
s_cselect_b32 s50, 0, -1 // (op 8788) 

s_and_b32 s18, s46, 31 // (op 8788) 
s_lshl_b32 s51, 1, s18 // (op 8788) 
s_add_i32 s72, s51, s50 // (op 8788) 


s_and_b32 s50, s0, s72 // (op 8788) 


s_add_i32 s0, s19, s50 // (op 8788) 


s_ashr_i32 s19, s0, s18 // (op 8788) 


s_lshl_b32 s0, s46, 24 // (op 8788) 
s_ashr_i32 s18, s0, 31 // (op 8788) 

s_xor_b32 s0, s19, s18 // (op 8788) 

s_sub_i32 s19, s0, s18 // (op 8788) 


s_cmp_eq_i32 s1, 0 // (op 8788) 
s_mov_b32 s0, scc // (op 8788) 

s_cmp_eq_i32 s0, 0 // (op 8788) 
s_cselect_b32 s1, s19, s49 // (op 8788) 



v_lshl_add_u32 v15, s1, 7, v17 // (op 8788) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 8788) 
s_lshr_b32 s18, s0, 1 // (op 8788) 

v_lshl_add_u32 v17, s18, 3, v24 // (op 8788) 


v_mul_lo_u32 v24, v17, s38 // (op 8788) 
v_add_u32 v24, v24, v15 // (op 8788) 


s_mul_i32 s1, s38, s43 // (op 8790) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 8790) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 8790) least significant
s_mul_hi_u32 s19, s38, s42 // (op 8790) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8790) most significant: sum
s_add_u32 s1, s1, s19 // (op 8790) most significant: sum


s_mov_b32 s68, s40 // (op 8790) reg expression
s_mov_b32 s69, s41 // (op 8790) reg expression
s_mov_b32 s70, s0 // (op 8790) reg expression
s_mov_b32 s71, 131072 // (op 8790) call()

v_and_b32 v15, -8, v1 // (op 8886) 
v_sub_u32 v17, v1, v15 // (op 8886) 

v_lshlrev_b32 v15, 4, v17 // (op 8886) 

v_lshrrev_b32 v17, 3, v1 // (op 8886) 
v_lshl_add_u32 v26, v17, 7, v15 // (op 8886) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 8886) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -8, v1 // (op 8943) 
v_sub_u32 v17, v1, v15 // (op 8943) 

v_lshlrev_b32 v15, 4, v17 // (op 8943) 

v_lshrrev_b32 v17, 3, v1 // (op 8943) 
v_lshl_add_u32 v26, v17, 7, v15 // (op 8943) 


s_nop 0
v_readfirstlane_b32 s1, v26 // (op 8943) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -4, v1 // (op 9000) 
v_sub_u32 v17, v1, v15 // (op 9000) 

v_lshrrev_b32 v15, 2, v1 // (op 9000) 
v_lshl_add_u32 v26, v15, 3, v17 // (op 9000) 


v_and_b32 v15, -128, v1 // (op 9058) 
v_sub_u32 v17, v1, v15 // (op 9058) 

v_lshrrev_b32 v15, 7, v1 // (op 9058) 
v_lshl_add_u32 v27, v15, 8, v17 // (op 9058) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 2592) 
s_and_b32 s49, s18, 4294967294 // (op 2592) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 2591) 
s_mov_b32 s50, s18 // (op 2591) convert

s_cmp_lt_u32 s49, s50 // (op 2591) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_2591 // (op 2591) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_2591 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoopTail_2591:
s_barrier  // (op 9383) 
s_add_u32 s49, s49, 1 // (op 2593) 

s_lshl_b64 s[18:19], s[24:25], 5 // (op 8530) 
v_mov_b64 v[18:19], s[18:19] // (op 8530) convert


v_mov_b64 v[16:17], v[20:21] // (op 8530) 
v_mov_b64 v[28:29], v[16:17] // (op 8530) 
s_add_u32 s18, 16384, s0 // (op 8530) 
s_mov_b32 m0, s18 // (op 8530) reg expression
buffer_load_dwordx4 v28, s[20:23], 0 offen offset: 0 lds // (op 8530) Load value direct to lds
v_add_co_u32 v28, s[50:51], v28, v18 // (op 8530) least significant half
v_addc_co_u32 v29, s[50:51], 0, v29, s[50:51] // (op 8530) most significant half

s_add_u32 m0, m0, 4096 // (op 8530) 
buffer_load_dwordx4 v28, s[20:23], 0 offen offset: 0 lds // (op 8530) Load value direct to lds
v_add_co_u32 v28, s[50:51], v28, v18 // (op 8530) least significant half
v_addc_co_u32 v29, s[50:51], 0, v29, s[50:51] // (op 8530) most significant half

s_add_u32 m0, m0, 4096 // (op 8530) 
buffer_load_dwordx4 v28, s[20:23], 0 offen offset: 0 lds // (op 8530) Load value direct to lds
v_add_co_u32 v28, s[50:51], v28, v18 // (op 8530) least significant half
v_addc_co_u32 v29, s[50:51], 0, v29, s[50:51] // (op 8530) most significant half

s_add_u32 m0, m0, 4096 // (op 8530) 
buffer_load_dwordx4 v28, s[20:23], 0 offen offset: 0 lds // (op 8530) Load value direct to lds




v_add_co_u32 v20, vcc, 128, v20 // (op 8568) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 8568) most significant half
s_barrier  // (op 9399) 

s_lshl_b64 s[18:19], s[28:29], 5 // (op 8536) 
v_mov_b64 v[18:19], s[18:19] // (op 8536) convert


v_mov_b64 v[16:17], v[22:23] // (op 8536) 
v_mov_b64 v[28:29], v[16:17] // (op 8536) reg expression
s_mov_b32 m0, s1 // (op 8536) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8536) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v18 // (op 8536) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 8536) most significant half

s_add_u32 m0, m0, 4096 // (op 8536) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8536) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v18 // (op 8536) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 8536) most significant half

s_add_u32 m0, m0, 4096 // (op 8536) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8536) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v18 // (op 8536) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 8536) most significant half

s_add_u32 m0, m0, 4096 // (op 8536) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8536) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 7 // (op 8652) 
v_mov_b32 v25, s18 // (op 8652) 
v_mov_b32 v15, s19 // (op 8652) 
v_add_co_u32 v22, s[50:51], v22, v25 // (op 8652) least significant half
v_addc_co_u32 v23, s[50:51], v23, v15, s[50:51] // (op 8652) most significant half





v_mov_b64 v[18:19], s[34:35] // (op 191) convert

v_mov_b32 v25, v14 // (op 191) 
v_mov_b32 v15, v25 // (op 191) 
buffer_load_ubyte v16, v15, s[64:67], 0 offen offset: 0 // (op 191) Load value
v_add_u32 v15, v15, v18 // (op 191) 
buffer_load_ubyte v17, v15, s[64:67], 0 offen offset: 0 // (op 191) Load value



v_add_u32 v14, 4, v14 // (op 8716) 

s_barrier  // (op 9416) 
v_mov_b64 v[18:19], s[38:39] // (op 192) convert




v_mov_b32 v15, v24 // (op 192) 
v_mov_b32 v25, v15 // (op 192) reg expression
v_add_u32 v28, 66048, v26 // (op 188) 
buffer_load_ubyte v30, v25, s[68:71], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v25, v25, v18 // (op 192) 
buffer_load_ubyte v31, v25, s[68:71], 0 offen offset: 0 // (op 192) Load value



s_lshl_b32 s18, s38, 2 // (op 8780) 
v_add_u32 v24, v24, s18 // (op 8780) 

s_waitcnt vmcnt(3)
ds_write_b8 v28, v16 // (op 188) Store local data 

s_barrier  // (op 9389) 
v_add_u32 v15, 66052, v26 // (op 188) 



ds_read_b128 v[32:35], v3 offset:16384 // (op 179) Load local data 
ds_read_b128 v[36:39], v3 offset:16448 // (op 179) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v15, v17 // (op 188) Store local data 


s_barrier  // (op 9406) 



s_barrier  // (op 9434) 
ds_read_b64_tr_b8 v[40:41], v4 // (op 181) Transpose load local data 



ds_read_b64_tr_b8 v[42:43], v4 offset:1024 // (op 181) Transpose load local data 
v_add_u32 v25, 65536, v27 // (op 189) 
ds_read_b64_tr_b8 v[44:45], v4 offset:8192 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:9216 // (op 181) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v25, v30 // (op 189) Store local data 

v_add_u32 v25, 65664, v27 // (op 189) 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 9424) 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v31 // (op 189) Store local data 





v_add_u32 v25, 66048, v0 // (op 183) 
ds_read_u8 v15, v25 // (op 183) Load local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 9441) 


v_add_u32 v25, 65536, v5 // (op 185) 
ds_read_u8 v16, v25 // (op 185) Load local data 




ds_read_b64_tr_b8 v[48:49], v4 offset:32 // (op 350) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v4 offset:1056 // (op 350) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v15, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 186) 
ds_read_b64_tr_b8 v[52:53], v4 offset:8224 // (op 350) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:9248 // (op 350) Transpose load local data 


v_add_u32 v25, 65568, v5 // (op 354) 
ds_read_u8 v17, v25 // (op 354) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:39], v[48:55], a[4:7], v15, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 355) 
ds_read_b64_tr_b8 v[56:57], v4 offset:64 // (op 526) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v4 offset:1088 // (op 526) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v4 offset:8256 // (op 526) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v4 offset:9280 // (op 526) Transpose load local data 


v_add_u32 v25, 65600, v5 // (op 530) 
ds_read_u8 v18, v25 // (op 530) Load local data 




ds_read_b64_tr_b8 v[64:65], v4 offset:96 // (op 702) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:39], v[56:63], a[8:11], v15, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 531) 
ds_read_b64_tr_b8 v[66:67], v4 offset:1120 // (op 702) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v4 offset:8288 // (op 702) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v4 offset:9312 // (op 702) Transpose load local data 


v_add_u32 v19, 65632, v5 // (op 706) 
ds_read_u8 v25, v19 // (op 706) Load local data 




ds_read_b128 v[72:75], v3 offset:20480 // (op 964) Load local data 
ds_read_b128 v[76:79], v3 offset:20544 // (op 964) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:39], v[64:71], a[12:15], v15, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 707) 





v_add_u32 v15, 66176, v0 // (op 967) 
ds_read_u8 v19, v15 // (op 967) Load local data 




ds_read_b128 v[28:31], v3 offset:24576 // (op 1734) Load local data 
ds_read_b128 v[32:35], v3 offset:24640 // (op 1734) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:79], v[40:47], a[16:19], v19, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 969) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:79], v[48:55], a[20:23], v19, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1000) 
v_add_u32 v15, 66304, v0 // (op 1737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[56:63], a[24:27], v19, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1031) 
ds_read_u8 v36, v15 // (op 1737) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[64:71], a[28:31], v19, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1062) 
ds_read_b128 v[80:83], v3 offset:28672 // (op 2504) Load local data 


ds_read_b128 v[84:87], v3 offset:28736 // (op 2504) Load local data 



v_add_u32 v15, 66432, v0 // (op 2507) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[28:35], v[40:47], a[32:35], v36, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1739) 
ds_read_u8 v19, v15 // (op 2507) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[28:35], v[48:55], a[36:39], v36, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[28:35], v[56:63], a[40:43], v36, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1801) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[28:35], v[64:71], a[44:47], v36, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1832) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:87], v[40:47], a[48:51], v19, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2509) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:87], v[48:55], a[52:55], v19, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2540) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:87], v[56:63], a[56:59], v19, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2571) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:87], v[64:71], a[60:63], v19, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2602) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 2591) 
s_mov_b32 s50, s18 // (op 2591) convert

s_cmp_lt_u32 s49, s50 // (op 2591) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoopTail_2591 // (op 2591) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoopTail_2591 if true)
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_2591:























s_barrier  // (op 9531) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalFalse_0_beta_133
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


v_accvgpr_read v6, a4 // (op 3561) 
v_accvgpr_read v7, a5 // (op 3561) 
v_accvgpr_read v8, a6 // (op 3561) 
v_accvgpr_read v9, a7 // (op 3561) 
v_mul_f32 v14, v11, v6 // (op 3561) 
v_mul_f32 v15, v11, v7 // (op 3561) 
v_mul_f32 v16, v11, v8 // (op 3561) 
v_mul_f32 v17, v11, v9 // (op 3561) 



s_lshl_b32 s0, s4, 7 // (op 3563) 
v_add_u32 v0, v13, s0 // (op 3563) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3563) Store value


v_accvgpr_read v6, a8 // (op 3573) 
v_accvgpr_read v7, a9 // (op 3573) 
v_accvgpr_read v8, a10 // (op 3573) 
v_accvgpr_read v9, a11 // (op 3573) 
v_mul_f32 v14, v11, v6 // (op 3573) 
v_mul_f32 v15, v11, v7 // (op 3573) 
v_mul_f32 v16, v11, v8 // (op 3573) 
v_mul_f32 v17, v11, v9 // (op 3573) 



s_lshl_b32 s0, s4, 8 // (op 3575) 
v_add_u32 v0, v13, s0 // (op 3575) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3575) Store value


v_accvgpr_read v6, a12 // (op 3585) 
v_accvgpr_read v7, a13 // (op 3585) 
v_accvgpr_read v8, a14 // (op 3585) 
v_accvgpr_read v9, a15 // (op 3585) 
v_mul_f32 v14, v11, v6 // (op 3585) 
v_mul_f32 v15, v11, v7 // (op 3585) 
v_mul_f32 v16, v11, v8 // (op 3585) 
v_mul_f32 v17, v11, v9 // (op 3585) 



s_lshl_b32 s0, s4, 7 // (op 3587) 
s_mul_i32 s1, 3, s0 // (op 3587) 

v_add_u32 v0, v13, s1 // (op 3587) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3587) Store value


v_accvgpr_read v6, a16 // (op 3619) 
v_accvgpr_read v7, a17 // (op 3619) 
v_accvgpr_read v8, a18 // (op 3619) 
v_accvgpr_read v9, a19 // (op 3619) 
v_mul_f32 v14, v11, v6 // (op 3619) 
v_mul_f32 v15, v11, v7 // (op 3619) 
v_mul_f32 v16, v11, v8 // (op 3619) 
v_mul_f32 v17, v11, v9 // (op 3619) 



v_add_u32 v0, 128, v13 // (op 3622) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3622) Store value


v_accvgpr_read v6, a20 // (op 3632) 
v_accvgpr_read v7, a21 // (op 3632) 
v_accvgpr_read v8, a22 // (op 3632) 
v_accvgpr_read v9, a23 // (op 3632) 
v_mul_f32 v14, v11, v6 // (op 3632) 
v_mul_f32 v15, v11, v7 // (op 3632) 
v_mul_f32 v16, v11, v8 // (op 3632) 
v_mul_f32 v17, v11, v9 // (op 3632) 



s_lshl_b32 s0, s4, 7 // (op 3624) 
s_add_u32 s0, s0, 128 // (op 3624) 
v_add_u32 v0, v13, s0 // (op 3624) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3624) Store value


v_accvgpr_read v6, a24 // (op 3637) 
v_accvgpr_read v7, a25 // (op 3637) 
v_accvgpr_read v8, a26 // (op 3637) 
v_accvgpr_read v9, a27 // (op 3637) 
v_mul_f32 v14, v11, v6 // (op 3637) 
v_mul_f32 v15, v11, v7 // (op 3637) 
v_mul_f32 v16, v11, v8 // (op 3637) 
v_mul_f32 v17, v11, v9 // (op 3637) 



s_lshl_b32 s0, s4, 8 // (op 3626) 
s_add_u32 s0, s0, 128 // (op 3626) 
v_add_u32 v0, v13, s0 // (op 3626) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3626) Store value


v_accvgpr_read v6, a28 // (op 3642) 
v_accvgpr_read v7, a29 // (op 3642) 
v_accvgpr_read v8, a30 // (op 3642) 
v_accvgpr_read v9, a31 // (op 3642) 
v_mul_f32 v14, v11, v6 // (op 3642) 
v_mul_f32 v15, v11, v7 // (op 3642) 
v_mul_f32 v16, v11, v8 // (op 3642) 
v_mul_f32 v17, v11, v9 // (op 3642) 



s_lshl_b32 s0, s4, 7 // (op 3628) 
s_mul_i32 s1, 3, s0 // (op 3628) 
s_add_u32 s1, s1, 128 // (op 3628) 

v_add_u32 v0, v13, s1 // (op 3628) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3628) Store value


v_accvgpr_read v6, a32 // (op 3687) 
v_accvgpr_read v7, a33 // (op 3687) 
v_accvgpr_read v8, a34 // (op 3687) 
v_accvgpr_read v9, a35 // (op 3687) 
v_mul_f32 v14, v11, v6 // (op 3687) 
v_mul_f32 v15, v11, v7 // (op 3687) 
v_mul_f32 v16, v11, v8 // (op 3687) 
v_mul_f32 v17, v11, v9 // (op 3687) 



v_add_u32 v0, 256, v13 // (op 3690) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3690) Store value


v_accvgpr_read v6, a36 // (op 3700) 
v_accvgpr_read v7, a37 // (op 3700) 
v_accvgpr_read v8, a38 // (op 3700) 
v_accvgpr_read v9, a39 // (op 3700) 
v_mul_f32 v14, v11, v6 // (op 3700) 
v_mul_f32 v15, v11, v7 // (op 3700) 
v_mul_f32 v16, v11, v8 // (op 3700) 
v_mul_f32 v17, v11, v9 // (op 3700) 



s_lshl_b32 s0, s4, 7 // (op 3692) 
s_add_u32 s0, s0, 256 // (op 3692) 
v_add_u32 v0, v13, s0 // (op 3692) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3692) Store value


v_accvgpr_read v6, a40 // (op 3705) 
v_accvgpr_read v7, a41 // (op 3705) 
v_accvgpr_read v8, a42 // (op 3705) 
v_accvgpr_read v9, a43 // (op 3705) 
v_mul_f32 v14, v11, v6 // (op 3705) 
v_mul_f32 v15, v11, v7 // (op 3705) 
v_mul_f32 v16, v11, v8 // (op 3705) 
v_mul_f32 v17, v11, v9 // (op 3705) 



s_lshl_b32 s0, s4, 8 // (op 3694) 
s_add_u32 s0, s0, 256 // (op 3694) 
v_add_u32 v0, v13, s0 // (op 3694) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3694) Store value


v_accvgpr_read v6, a44 // (op 3710) 
v_accvgpr_read v7, a45 // (op 3710) 
v_accvgpr_read v8, a46 // (op 3710) 
v_accvgpr_read v9, a47 // (op 3710) 
v_mul_f32 v14, v11, v6 // (op 3710) 
v_mul_f32 v15, v11, v7 // (op 3710) 
v_mul_f32 v16, v11, v8 // (op 3710) 
v_mul_f32 v17, v11, v9 // (op 3710) 



s_lshl_b32 s0, s4, 7 // (op 3696) 
s_mul_i32 s1, 3, s0 // (op 3696) 
s_add_u32 s1, s1, 256 // (op 3696) 

v_add_u32 v0, v13, s1 // (op 3696) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3696) Store value


v_accvgpr_read v6, a48 // (op 3755) 
v_accvgpr_read v7, a49 // (op 3755) 
v_accvgpr_read v8, a50 // (op 3755) 
v_accvgpr_read v9, a51 // (op 3755) 
v_mul_f32 v14, v11, v6 // (op 3755) 
v_mul_f32 v15, v11, v7 // (op 3755) 
v_mul_f32 v16, v11, v8 // (op 3755) 
v_mul_f32 v17, v11, v9 // (op 3755) 



v_add_u32 v0, 384, v13 // (op 3758) 

buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3758) Store value


v_accvgpr_read v6, a52 // (op 3768) 
v_accvgpr_read v7, a53 // (op 3768) 
v_accvgpr_read v8, a54 // (op 3768) 
v_accvgpr_read v9, a55 // (op 3768) 
v_mul_f32 v14, v11, v6 // (op 3768) 
v_mul_f32 v15, v11, v7 // (op 3768) 
v_mul_f32 v16, v11, v8 // (op 3768) 
v_mul_f32 v17, v11, v9 // (op 3768) 



s_lshl_b32 s0, s4, 7 // (op 3760) 
s_add_u32 s0, s0, 384 // (op 3760) 
v_add_u32 v0, v13, s0 // (op 3760) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3760) Store value


v_accvgpr_read v6, a56 // (op 3773) 
v_accvgpr_read v7, a57 // (op 3773) 
v_accvgpr_read v8, a58 // (op 3773) 
v_accvgpr_read v9, a59 // (op 3773) 
v_mul_f32 v14, v11, v6 // (op 3773) 
v_mul_f32 v15, v11, v7 // (op 3773) 
v_mul_f32 v16, v11, v8 // (op 3773) 
v_mul_f32 v17, v11, v9 // (op 3773) 



s_lshl_b32 s0, s4, 8 // (op 3762) 
s_add_u32 s0, s0, 384 // (op 3762) 
v_add_u32 v0, v13, s0 // (op 3762) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3762) Store value


v_accvgpr_read v6, a60 // (op 3778) 
v_accvgpr_read v7, a61 // (op 3778) 
v_accvgpr_read v8, a62 // (op 3778) 
v_accvgpr_read v9, a63 // (op 3778) 
v_mul_f32 v14, v11, v6 // (op 3778) 
v_mul_f32 v15, v11, v7 // (op 3778) 
v_mul_f32 v16, v11, v8 // (op 3778) 
v_mul_f32 v17, v11, v9 // (op 3778) 



s_lshl_b32 s0, s4, 7 // (op 3764) 
s_mul_i32 s1, 3, s0 // (op 3764) 
s_add_u32 s1, s1, 384 // (op 3764) 

v_add_u32 v0, v13, s1 // (op 3764) 


buffer_store_dwordx4 v[14:17], v0, s[60:63], 0 offen // (op 3764) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalBottom_0_beta_133
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalFalse_0_beta_133:


v_add_u32 v0, 384, v12 // (op 4122) 

buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4122) Load value

v_accvgpr_read v14, a48 // (op 4110) 
v_accvgpr_read v15, a49 // (op 4110) 
v_accvgpr_read v16, a50 // (op 4110) 
v_accvgpr_read v17, a51 // (op 4110) 
v_mul_f32 v18, v11, v14 // (op 4110) 
v_mul_f32 v19, v11, v15 // (op 4110) 
v_mul_f32 v20, v11, v16 // (op 4110) 
v_mul_f32 v21, v11, v17 // (op 4110) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4121) 
v_mul_f32 v15, v10, v7 // (op 4121) 
v_mul_f32 v16, v10, v8 // (op 4121) 
v_mul_f32 v17, v10, v9 // (op 4121) 

v_add_f32 v6, v14, v18 // (op 4111) 
v_add_f32 v7, v15, v19 // (op 4111) 
v_add_f32 v8, v16, v20 // (op 4111) 
v_add_f32 v9, v17, v21 // (op 4111) 




v_add_u32 v0, 384, v13 // (op 4113) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4113) Store value




s_lshl_b32 s0, s2, 7 // (op 4126) 
s_add_u32 s0, s0, 384 // (op 4126) 
v_add_u32 v0, v12, s0 // (op 4126) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4126) Load value

v_accvgpr_read v14, a52 // (op 4136) 
v_accvgpr_read v15, a53 // (op 4136) 
v_accvgpr_read v16, a54 // (op 4136) 
v_accvgpr_read v17, a55 // (op 4136) 
v_mul_f32 v18, v11, v14 // (op 4136) 
v_mul_f32 v19, v11, v15 // (op 4136) 
v_mul_f32 v20, v11, v16 // (op 4136) 
v_mul_f32 v21, v11, v17 // (op 4136) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4124) 
v_mul_f32 v15, v10, v7 // (op 4124) 
v_mul_f32 v16, v10, v8 // (op 4124) 
v_mul_f32 v17, v10, v9 // (op 4124) 

v_add_f32 v6, v14, v18 // (op 4125) 
v_add_f32 v7, v15, v19 // (op 4125) 
v_add_f32 v8, v16, v20 // (op 4125) 
v_add_f32 v9, v17, v21 // (op 4125) 




s_lshl_b32 s0, s4, 7 // (op 4115) 
s_add_u32 s0, s0, 384 // (op 4115) 
v_add_u32 v0, v13, s0 // (op 4115) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4115) Store value




s_lshl_b32 s0, s2, 8 // (op 4130) 
s_add_u32 s0, s0, 384 // (op 4130) 
v_add_u32 v0, v12, s0 // (op 4130) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4130) Load value

v_accvgpr_read v14, a56 // (op 4138) 
v_accvgpr_read v15, a57 // (op 4138) 
v_accvgpr_read v16, a58 // (op 4138) 
v_accvgpr_read v17, a59 // (op 4138) 
v_mul_f32 v18, v11, v14 // (op 4138) 
v_mul_f32 v19, v11, v15 // (op 4138) 
v_mul_f32 v20, v11, v16 // (op 4138) 
v_mul_f32 v21, v11, v17 // (op 4138) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4128) 
v_mul_f32 v15, v10, v7 // (op 4128) 
v_mul_f32 v16, v10, v8 // (op 4128) 
v_mul_f32 v17, v10, v9 // (op 4128) 

v_add_f32 v6, v14, v18 // (op 4129) 
v_add_f32 v7, v15, v19 // (op 4129) 
v_add_f32 v8, v16, v20 // (op 4129) 
v_add_f32 v9, v17, v21 // (op 4129) 




s_lshl_b32 s0, s4, 8 // (op 4117) 
s_add_u32 s0, s0, 384 // (op 4117) 
v_add_u32 v0, v13, s0 // (op 4117) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4117) Store value




s_lshl_b32 s0, s2, 7 // (op 4134) 
s_mul_i32 s1, 3, s0 // (op 4134) 
s_add_u32 s1, s1, 384 // (op 4134) 

v_add_u32 v0, v12, s1 // (op 4134) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4134) Load value

v_accvgpr_read v14, a60 // (op 4140) 
v_accvgpr_read v15, a61 // (op 4140) 
v_accvgpr_read v16, a62 // (op 4140) 
v_accvgpr_read v17, a63 // (op 4140) 
v_mul_f32 v18, v11, v14 // (op 4140) 
v_mul_f32 v19, v11, v15 // (op 4140) 
v_mul_f32 v20, v11, v16 // (op 4140) 
v_mul_f32 v21, v11, v17 // (op 4140) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4132) 
v_mul_f32 v15, v10, v7 // (op 4132) 
v_mul_f32 v16, v10, v8 // (op 4132) 
v_mul_f32 v17, v10, v9 // (op 4132) 

v_add_f32 v6, v14, v18 // (op 4133) 
v_add_f32 v7, v15, v19 // (op 4133) 
v_add_f32 v8, v16, v20 // (op 4133) 
v_add_f32 v9, v17, v21 // (op 4133) 




s_lshl_b32 s0, s4, 7 // (op 4119) 
s_mul_i32 s1, 3, s0 // (op 4119) 
s_add_u32 s1, s1, 384 // (op 4119) 

v_add_u32 v0, v13, s1 // (op 4119) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4119) Store value




v_add_u32 v0, 256, v12 // (op 4043) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4043) Load value

v_accvgpr_read v14, a32 // (op 4031) 
v_accvgpr_read v15, a33 // (op 4031) 
v_accvgpr_read v16, a34 // (op 4031) 
v_accvgpr_read v17, a35 // (op 4031) 
v_mul_f32 v18, v11, v14 // (op 4031) 
v_mul_f32 v19, v11, v15 // (op 4031) 
v_mul_f32 v20, v11, v16 // (op 4031) 
v_mul_f32 v21, v11, v17 // (op 4031) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4042) 
v_mul_f32 v15, v10, v7 // (op 4042) 
v_mul_f32 v16, v10, v8 // (op 4042) 
v_mul_f32 v17, v10, v9 // (op 4042) 

v_add_f32 v6, v14, v18 // (op 4032) 
v_add_f32 v7, v15, v19 // (op 4032) 
v_add_f32 v8, v16, v20 // (op 4032) 
v_add_f32 v9, v17, v21 // (op 4032) 




v_add_u32 v0, 256, v13 // (op 4034) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4034) Store value




s_lshl_b32 s0, s2, 7 // (op 4047) 
s_add_u32 s0, s0, 256 // (op 4047) 
v_add_u32 v0, v12, s0 // (op 4047) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4047) Load value

v_accvgpr_read v14, a36 // (op 4057) 
v_accvgpr_read v15, a37 // (op 4057) 
v_accvgpr_read v16, a38 // (op 4057) 
v_accvgpr_read v17, a39 // (op 4057) 
v_mul_f32 v18, v11, v14 // (op 4057) 
v_mul_f32 v19, v11, v15 // (op 4057) 
v_mul_f32 v20, v11, v16 // (op 4057) 
v_mul_f32 v21, v11, v17 // (op 4057) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4045) 
v_mul_f32 v15, v10, v7 // (op 4045) 
v_mul_f32 v16, v10, v8 // (op 4045) 
v_mul_f32 v17, v10, v9 // (op 4045) 

v_add_f32 v6, v14, v18 // (op 4046) 
v_add_f32 v7, v15, v19 // (op 4046) 
v_add_f32 v8, v16, v20 // (op 4046) 
v_add_f32 v9, v17, v21 // (op 4046) 




s_lshl_b32 s0, s4, 7 // (op 4036) 
s_add_u32 s0, s0, 256 // (op 4036) 
v_add_u32 v0, v13, s0 // (op 4036) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4036) Store value




s_lshl_b32 s0, s2, 8 // (op 4051) 
s_add_u32 s0, s0, 256 // (op 4051) 
v_add_u32 v0, v12, s0 // (op 4051) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4051) Load value

v_accvgpr_read v14, a40 // (op 4059) 
v_accvgpr_read v15, a41 // (op 4059) 
v_accvgpr_read v16, a42 // (op 4059) 
v_accvgpr_read v17, a43 // (op 4059) 
v_mul_f32 v18, v11, v14 // (op 4059) 
v_mul_f32 v19, v11, v15 // (op 4059) 
v_mul_f32 v20, v11, v16 // (op 4059) 
v_mul_f32 v21, v11, v17 // (op 4059) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4049) 
v_mul_f32 v15, v10, v7 // (op 4049) 
v_mul_f32 v16, v10, v8 // (op 4049) 
v_mul_f32 v17, v10, v9 // (op 4049) 

v_add_f32 v6, v14, v18 // (op 4050) 
v_add_f32 v7, v15, v19 // (op 4050) 
v_add_f32 v8, v16, v20 // (op 4050) 
v_add_f32 v9, v17, v21 // (op 4050) 




s_lshl_b32 s0, s4, 8 // (op 4038) 
s_add_u32 s0, s0, 256 // (op 4038) 
v_add_u32 v0, v13, s0 // (op 4038) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4038) Store value




s_lshl_b32 s0, s2, 7 // (op 4055) 
s_mul_i32 s1, 3, s0 // (op 4055) 
s_add_u32 s1, s1, 256 // (op 4055) 

v_add_u32 v0, v12, s1 // (op 4055) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4055) Load value

v_accvgpr_read v14, a44 // (op 4061) 
v_accvgpr_read v15, a45 // (op 4061) 
v_accvgpr_read v16, a46 // (op 4061) 
v_accvgpr_read v17, a47 // (op 4061) 
v_mul_f32 v18, v11, v14 // (op 4061) 
v_mul_f32 v19, v11, v15 // (op 4061) 
v_mul_f32 v20, v11, v16 // (op 4061) 
v_mul_f32 v21, v11, v17 // (op 4061) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4053) 
v_mul_f32 v15, v10, v7 // (op 4053) 
v_mul_f32 v16, v10, v8 // (op 4053) 
v_mul_f32 v17, v10, v9 // (op 4053) 

v_add_f32 v6, v14, v18 // (op 4054) 
v_add_f32 v7, v15, v19 // (op 4054) 
v_add_f32 v8, v16, v20 // (op 4054) 
v_add_f32 v9, v17, v21 // (op 4054) 




s_lshl_b32 s0, s4, 7 // (op 4040) 
s_mul_i32 s1, 3, s0 // (op 4040) 
s_add_u32 s1, s1, 256 // (op 4040) 

v_add_u32 v0, v13, s1 // (op 4040) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 4040) Store value




v_add_u32 v0, 128, v12 // (op 3964) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3964) Load value

v_accvgpr_read v14, a16 // (op 3952) 
v_accvgpr_read v15, a17 // (op 3952) 
v_accvgpr_read v16, a18 // (op 3952) 
v_accvgpr_read v17, a19 // (op 3952) 
v_mul_f32 v18, v11, v14 // (op 3952) 
v_mul_f32 v19, v11, v15 // (op 3952) 
v_mul_f32 v20, v11, v16 // (op 3952) 
v_mul_f32 v21, v11, v17 // (op 3952) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3963) 
v_mul_f32 v15, v10, v7 // (op 3963) 
v_mul_f32 v16, v10, v8 // (op 3963) 
v_mul_f32 v17, v10, v9 // (op 3963) 

v_add_f32 v6, v14, v18 // (op 3953) 
v_add_f32 v7, v15, v19 // (op 3953) 
v_add_f32 v8, v16, v20 // (op 3953) 
v_add_f32 v9, v17, v21 // (op 3953) 




v_add_u32 v0, 128, v13 // (op 3955) 

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3955) Store value




s_lshl_b32 s0, s2, 7 // (op 3968) 
s_add_u32 s0, s0, 128 // (op 3968) 
v_add_u32 v0, v12, s0 // (op 3968) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3968) Load value

v_accvgpr_read v14, a20 // (op 3978) 
v_accvgpr_read v15, a21 // (op 3978) 
v_accvgpr_read v16, a22 // (op 3978) 
v_accvgpr_read v17, a23 // (op 3978) 
v_mul_f32 v18, v11, v14 // (op 3978) 
v_mul_f32 v19, v11, v15 // (op 3978) 
v_mul_f32 v20, v11, v16 // (op 3978) 
v_mul_f32 v21, v11, v17 // (op 3978) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3966) 
v_mul_f32 v15, v10, v7 // (op 3966) 
v_mul_f32 v16, v10, v8 // (op 3966) 
v_mul_f32 v17, v10, v9 // (op 3966) 

v_add_f32 v6, v14, v18 // (op 3967) 
v_add_f32 v7, v15, v19 // (op 3967) 
v_add_f32 v8, v16, v20 // (op 3967) 
v_add_f32 v9, v17, v21 // (op 3967) 




s_lshl_b32 s0, s4, 7 // (op 3957) 
s_add_u32 s0, s0, 128 // (op 3957) 
v_add_u32 v0, v13, s0 // (op 3957) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3957) Store value




s_lshl_b32 s0, s2, 8 // (op 3972) 
s_add_u32 s0, s0, 128 // (op 3972) 
v_add_u32 v0, v12, s0 // (op 3972) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3972) Load value

v_accvgpr_read v14, a24 // (op 3980) 
v_accvgpr_read v15, a25 // (op 3980) 
v_accvgpr_read v16, a26 // (op 3980) 
v_accvgpr_read v17, a27 // (op 3980) 
v_mul_f32 v18, v11, v14 // (op 3980) 
v_mul_f32 v19, v11, v15 // (op 3980) 
v_mul_f32 v20, v11, v16 // (op 3980) 
v_mul_f32 v21, v11, v17 // (op 3980) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3970) 
v_mul_f32 v15, v10, v7 // (op 3970) 
v_mul_f32 v16, v10, v8 // (op 3970) 
v_mul_f32 v17, v10, v9 // (op 3970) 

v_add_f32 v6, v14, v18 // (op 3971) 
v_add_f32 v7, v15, v19 // (op 3971) 
v_add_f32 v8, v16, v20 // (op 3971) 
v_add_f32 v9, v17, v21 // (op 3971) 




s_lshl_b32 s0, s4, 8 // (op 3959) 
s_add_u32 s0, s0, 128 // (op 3959) 
v_add_u32 v0, v13, s0 // (op 3959) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3959) Store value




s_lshl_b32 s0, s2, 7 // (op 3976) 
s_mul_i32 s1, 3, s0 // (op 3976) 
s_add_u32 s1, s1, 128 // (op 3976) 

v_add_u32 v0, v12, s1 // (op 3976) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3976) Load value

v_accvgpr_read v14, a28 // (op 3982) 
v_accvgpr_read v15, a29 // (op 3982) 
v_accvgpr_read v16, a30 // (op 3982) 
v_accvgpr_read v17, a31 // (op 3982) 
v_mul_f32 v18, v11, v14 // (op 3982) 
v_mul_f32 v19, v11, v15 // (op 3982) 
v_mul_f32 v20, v11, v16 // (op 3982) 
v_mul_f32 v21, v11, v17 // (op 3982) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3974) 
v_mul_f32 v15, v10, v7 // (op 3974) 
v_mul_f32 v16, v10, v8 // (op 3974) 
v_mul_f32 v17, v10, v9 // (op 3974) 

v_add_f32 v6, v14, v18 // (op 3975) 
v_add_f32 v7, v15, v19 // (op 3975) 
v_add_f32 v8, v16, v20 // (op 3975) 
v_add_f32 v9, v17, v21 // (op 3975) 




s_lshl_b32 s0, s4, 7 // (op 3961) 
s_mul_i32 s1, 3, s0 // (op 3961) 
s_add_u32 s1, s1, 128 // (op 3961) 

v_add_u32 v0, v13, s1 // (op 3961) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3961) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 145) Load value

v_accvgpr_read v14, a0 // (op 142) 
v_accvgpr_read v15, a1 // (op 142) 
v_accvgpr_read v16, a2 // (op 142) 
v_accvgpr_read v17, a3 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 
v_mul_f32 v20, v11, v16 // (op 142) 
v_mul_f32 v21, v11, v17 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 146) 
v_mul_f32 v15, v10, v7 // (op 146) 
v_mul_f32 v16, v10, v8 // (op 146) 
v_mul_f32 v17, v10, v9 // (op 146) 

v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 
v_add_f32 v8, v16, v20 // (op 143) 
v_add_f32 v9, v17, v21 // (op 143) 




v_mov_b32 v0, v13 // (op 144) reg expression

buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 3878) 
v_add_u32 v0, v12, s0 // (op 3878) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3878) Load value

v_accvgpr_read v14, a4 // (op 3875) 
v_accvgpr_read v15, a5 // (op 3875) 
v_accvgpr_read v16, a6 // (op 3875) 
v_accvgpr_read v17, a7 // (op 3875) 
v_mul_f32 v18, v11, v14 // (op 3875) 
v_mul_f32 v19, v11, v15 // (op 3875) 
v_mul_f32 v20, v11, v16 // (op 3875) 
v_mul_f32 v21, v11, v17 // (op 3875) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3879) 
v_mul_f32 v15, v10, v7 // (op 3879) 
v_mul_f32 v16, v10, v8 // (op 3879) 
v_mul_f32 v17, v10, v9 // (op 3879) 

v_add_f32 v6, v14, v18 // (op 3876) 
v_add_f32 v7, v15, v19 // (op 3876) 
v_add_f32 v8, v16, v20 // (op 3876) 
v_add_f32 v9, v17, v21 // (op 3876) 




s_lshl_b32 s0, s4, 7 // (op 3877) 
v_add_u32 v0, v13, s0 // (op 3877) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3877) Store value




s_lshl_b32 s0, s2, 8 // (op 3890) 
v_add_u32 v0, v12, s0 // (op 3890) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3890) Load value

v_accvgpr_read v14, a8 // (op 3887) 
v_accvgpr_read v15, a9 // (op 3887) 
v_accvgpr_read v16, a10 // (op 3887) 
v_accvgpr_read v17, a11 // (op 3887) 
v_mul_f32 v18, v11, v14 // (op 3887) 
v_mul_f32 v19, v11, v15 // (op 3887) 
v_mul_f32 v20, v11, v16 // (op 3887) 
v_mul_f32 v21, v11, v17 // (op 3887) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3891) 
v_mul_f32 v15, v10, v7 // (op 3891) 
v_mul_f32 v16, v10, v8 // (op 3891) 
v_mul_f32 v17, v10, v9 // (op 3891) 

v_add_f32 v6, v14, v18 // (op 3888) 
v_add_f32 v7, v15, v19 // (op 3888) 
v_add_f32 v8, v16, v20 // (op 3888) 
v_add_f32 v9, v17, v21 // (op 3888) 




s_lshl_b32 s0, s4, 8 // (op 3889) 
v_add_u32 v0, v13, s0 // (op 3889) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3889) Store value




s_lshl_b32 s0, s2, 7 // (op 3902) 
s_mul_i32 s1, 3, s0 // (op 3902) 

v_add_u32 v0, v12, s1 // (op 3902) 


buffer_load_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3902) Load value

v_accvgpr_read v14, a12 // (op 3899) 
v_accvgpr_read v15, a13 // (op 3899) 
v_accvgpr_read v16, a14 // (op 3899) 
v_accvgpr_read v17, a15 // (op 3899) 
v_mul_f32 v18, v11, v14 // (op 3899) 
v_mul_f32 v19, v11, v15 // (op 3899) 
v_mul_f32 v20, v11, v16 // (op 3899) 
v_mul_f32 v21, v11, v17 // (op 3899) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3903) 
v_mul_f32 v15, v10, v7 // (op 3903) 
v_mul_f32 v16, v10, v8 // (op 3903) 
v_mul_f32 v17, v10, v9 // (op 3903) 

v_add_f32 v6, v14, v18 // (op 3900) 
v_add_f32 v7, v15, v19 // (op 3900) 
v_add_f32 v8, v16, v20 // (op 3900) 
v_add_f32 v9, v17, v21 // (op 3900) 




s_lshl_b32 s0, s4, 7 // (op 3901) 
s_mul_i32 s1, 3, s0 // (op 3901) 

v_add_u32 v0, v13, s1 // (op 3901) 


buffer_store_dwordx4 v[6:9], v0, s[60:63], 0 offen // (op 3901) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalBottom_0_beta_133:
























s_endpgm  // End of RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
.LRR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__end:
.size RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_, .LRR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__end-RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 216
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 67584 // lds bytes
  .amdhsa_accum_offset 152
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
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 67584
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 150
    .agpr_count: 64
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

