

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_,@function
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 2648) 
s_mul_hi_i32 s1, s16, s10 // (op 2890) 
v_and_b32 v0, 63, v1 // (op 2705) 
v_lshrrev_b32 v3, 4, v0 // (op 2705) 

v_and_b32 v0, 1, v3 // (op 2705) 

v_and_b32 v3, 15, v1 // (op 2705) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2705) 


v_lshrrev_b32 v0, 7, v1 // (op 2705) 
v_lshl_add_u32 v3, v0, 5, v4 // (op 2705) 


v_and_b32 v0, 32, v1 // (op 2705) 
v_lshl_add_u32 v4, v3, 7, v0 // (op 2705) 


v_mul_lo_u32 v0, v4, 6 // (op 2705) 

v_lshrrev_b32 v3, 3, v0 // (op 2705) 

v_lshrrev_b32 v0, 7, v1 // (op 2816) 
v_and_b32 v4, 31, v1 // (op 2816) 
v_lshl_add_u32 v5, v0, 5, v4 // (op 2816) 


v_and_b32 v0, 63, v1 // (op 2816) 
v_lshrrev_b32 v4, 5, v0 // (op 2816) 

v_lshl_add_u32 v0, v5, 2, v4 // (op 2816) 


v_and_b32 v4, 63, v1 // (op 2916) 
v_lshrrev_b32 v5, 4, v4 // (op 2916) 

v_and_b32 v4, 1, v5 // (op 2916) 

v_and_b32 v5, 15, v1 // (op 2916) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 2916) 


v_lshrrev_b32 v4, 7, v1 // (op 2916) 
v_lshl_add_u32 v5, v4, 5, v6 // (op 2916) 


v_and_b32 v4, 32, v1 // (op 2916) 
v_lshl_add_u32 v6, v5, 7, v4 // (op 2916) 


v_mul_lo_u32 v4, v6, 6 // (op 2916) 

v_lshrrev_b32 v5, 3, v4 // (op 2916) 

v_lshrrev_b32 v4, 7, v1 // (op 2969) 
v_and_b32 v6, 31, v1 // (op 2969) 
v_lshl_add_u32 v7, v4, 5, v6 // (op 2969) 


v_and_b32 v4, 63, v1 // (op 2969) 
v_lshrrev_b32 v6, 5, v4 // (op 2969) 

v_lshl_add_u32 v4, v7, 2, v6 // (op 2969) 


v_lshrrev_b32 v6, 7, v1 // (op 2648) 
s_add_i32 s18, s0, s16 // (op 2648) 

v_and_b32 v7, 63, v1 // (op 2648) 
s_ashr_i32 s0, s18, 31 // (op 2648) 
v_lshrrev_b32 v8, 5, v7 // (op 2648) 
s_cmp_eq_i32 s10, 0 // (op 2648) 
s_mov_b32 s19, scc // (op 2648) 
v_lshlrev_b32 v9, 2, v8 // (op 2648) 

s_cmp_eq_i32 s19, 0 // (op 2648) 
s_cselect_b32 s49, 0, -1 // (op 2648) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2648) 


s_and_b32 s19, s13, 31 // (op 2648) 
v_lshrrev_b32 v6, 2, v7 // (op 2648) 

v_and_b32 v7, 7, v6 // (op 2648) 

v_and_b32 v6, 63, v1 // (op 2763) 
v_lshrrev_b32 v9, 4, v6 // (op 2763) 

v_and_b32 v6, 1, v9 // (op 2763) 

v_and_b32 v9, 15, v1 // (op 2763) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 2763) 


v_lshrrev_b32 v6, 6, v1 // (op 2763) 
v_and_b32 v9, 1, v6 // (op 2763) 

v_lshl_add_u32 v6, v9, 5, v10 // (op 2763) 


v_and_b32 v9, 32, v1 // (op 2763) 
v_lshl_add_u32 v10, v6, 7, v9 // (op 2763) 


v_mul_lo_u32 v6, v10, 6 // (op 2763) 

v_lshrrev_b32 v9, 3, v6 // (op 2763) 

v_lshrrev_b32 v6, 6, v1 // (op 2874) 
v_and_b32 v10, 1, v6 // (op 2874) 

v_and_b32 v6, 31, v1 // (op 2874) 
v_lshl_add_u32 v11, v10, 5, v6 // (op 2874) 


v_and_b32 v6, 63, v1 // (op 2874) 
v_lshrrev_b32 v10, 5, v6 // (op 2874) 

v_lshl_add_u32 v6, v11, 2, v10 // (op 2874) 


v_lshrrev_b32 v10, 7, v1 // (op 2890) 
v_and_b32 v11, 63, v1 // (op 2890) 
v_and_b32 v12, 63, v1 // (op 2945) 
v_lshrrev_b32 v13, 4, v12 // (op 2945) 

v_and_b32 v12, 1, v13 // (op 2945) 

v_and_b32 v13, 15, v1 // (op 2945) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2945) 


v_lshrrev_b32 v12, 6, v1 // (op 2945) 
v_and_b32 v13, 1, v12 // (op 2945) 

v_lshl_add_u32 v12, v13, 5, v14 // (op 2945) 


v_and_b32 v13, 32, v1 // (op 2945) 
v_lshl_add_u32 v14, v12, 7, v13 // (op 2945) 


v_mul_lo_u32 v12, v14, 6 // (op 2945) 

v_lshrrev_b32 v13, 3, v12 // (op 2945) 

v_lshrrev_b32 v12, 6, v1 // (op 2998) 
v_and_b32 v14, 1, v12 // (op 2998) 

v_and_b32 v12, 31, v1 // (op 2998) 
v_lshl_add_u32 v15, v14, 5, v12 // (op 2998) 


v_and_b32 v12, 63, v1 // (op 2998) 
v_lshrrev_b32 v14, 5, v12 // (op 2998) 

v_lshl_add_u32 v12, v15, 2, v14 // (op 2998) 


s_lshl_b32 s50, 1, s19 // (op 2648) 
v_and_b32 v14, 3, v1 // (op 2648) 
s_add_i32 s51, s50, s49 // (op 2648) 


s_and_b32 s49, s0, s51 // (op 2648) 


s_add_i32 s0, s1, s16 // (op 2890) 

s_ashr_i32 s1, s0, 31 // (op 2890) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 2648) 


s_add_i32 s50, s18, s49 // (op 2648) 


v_lshrrev_b32 v7, 6, v1 // (op 2648) 
s_ashr_i32 s18, s50, s19 // (op 2648) 


v_and_b32 v14, 1, v7 // (op 2648) 

s_lshl_b32 s19, s13, 24 // (op 2648) 
v_lshl_add_u32 v7, v14, 5, v15 // (op 2648) 


s_ashr_i32 s49, s19, 31 // (op 2648) 

s_xor_b32 s19, s18, s49 // (op 2648) 

s_sub_i32 s18, s19, s49 // (op 2648) 


s_mul_hi_i32 s19, s18, s14 // (op 2648) 
s_add_i32 s49, s19, s18 // (op 2648) 

s_ashr_i32 s19, s49, 31 // (op 2648) 
s_cmp_eq_i32 s14, 0 // (op 2648) 
s_mov_b32 s50, scc // (op 2648) 
s_cmp_eq_i32 s50, 0 // (op 2648) 
s_cselect_b32 s51, 0, -1 // (op 2648) 

s_and_b32 s50, s15, 31 // (op 2648) 
v_lshrrev_b32 v14, 5, v11 // (op 2890) 
s_cmp_eq_i32 s10, 0 // (op 2890) 
s_mov_b32 s52, scc // (op 2890) 
v_lshlrev_b32 v15, 2, v14 // (op 2890) 

s_cmp_eq_i32 s52, 0 // (op 2890) 
s_cselect_b32 s53, 0, -1 // (op 2890) 

s_lshl_b32 s52, 1, s50 // (op 2648) 
v_lshl_add_u32 v14, v10, 5, v15 // (op 2890) 


v_lshrrev_b32 v10, 2, v11 // (op 2890) 

s_add_i32 s54, s52, s51 // (op 2648) 


s_and_b32 s51, s19, s54 // (op 2648) 


s_add_i32 s19, s49, s51 // (op 2648) 


s_ashr_i32 s49, s19, s50 // (op 2648) 


s_lshl_b32 s19, s15, 24 // (op 2648) 
s_ashr_i32 s50, s19, 31 // (op 2648) 

s_xor_b32 s19, s49, s50 // (op 2648) 

s_sub_i32 s49, s19, s50 // (op 2648) 


s_mul_i32 s19, s18, s11 // (op 2648) 
s_sub_u32 s50, s16, s19 // (op 2648) 

s_mul_i32 s19, s49, s11 // (op 2648) 
s_add_u32 s19, s19, s50 // (op 2648) 

s_mul_hi_i32 s50, s16, s9 // (op 2648) 
s_add_i32 s51, s50, s16 // (op 2648) 

s_ashr_i32 s50, s51, 31 // (op 2648) 
s_cmp_eq_i32 s9, 0 // (op 2648) 
s_mov_b32 s52, scc // (op 2648) 
s_cmp_eq_i32 s52, 0 // (op 2648) 
s_cselect_b32 s54, 0, -1 // (op 2648) 

s_and_b32 s52, s13, 31 // (op 2890) 
v_and_b32 v11, 7, v10 // (op 2890) 

s_lshl_b32 s55, 1, s52 // (op 2890) 
v_and_b32 v10, 3, v1 // (op 2890) 
s_add_i32 s56, s55, s53 // (op 2890) 


s_and_b32 s53, s1, s56 // (op 2890) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 2890) 


s_add_i32 s1, s0, s53 // (op 2890) 


v_lshrrev_b32 v10, 6, v1 // (op 2890) 
s_ashr_i32 s0, s1, s52 // (op 2890) 


v_and_b32 v11, 1, v10 // (op 2890) 

s_lshl_b32 s1, s13, 24 // (op 2890) 
v_lshl_add_u32 v10, v11, 5, v15 // (op 2890) 


s_ashr_i32 s52, s1, 31 // (op 2890) 

s_xor_b32 s1, s0, s52 // (op 2890) 

s_sub_i32 s0, s1, s52 // (op 2890) 


s_mul_hi_i32 s1, s0, s14 // (op 2890) 
s_add_i32 s52, s1, s0 // (op 2890) 

s_ashr_i32 s1, s52, 31 // (op 2890) 
s_cmp_eq_i32 s14, 0 // (op 2890) 
s_mov_b32 s53, scc // (op 2890) 
s_cmp_eq_i32 s53, 0 // (op 2890) 
s_cselect_b32 s55, 0, -1 // (op 2890) 

s_and_b32 s53, s15, 31 // (op 2890) 
s_lshl_b32 s56, 1, s53 // (op 2890) 
s_add_i32 s57, s56, s55 // (op 2890) 


s_and_b32 s55, s1, s57 // (op 2890) 


s_add_i32 s1, s52, s55 // (op 2890) 


s_ashr_i32 s52, s1, s53 // (op 2890) 


s_lshl_b32 s1, s15, 24 // (op 2890) 
s_ashr_i32 s53, s1, 31 // (op 2890) 

s_xor_b32 s1, s52, s53 // (op 2890) 

s_sub_i32 s52, s1, s53 // (op 2890) 


s_mul_i32 s1, s0, s11 // (op 2890) 
s_sub_u32 s53, s16, s1 // (op 2890) 

s_mul_i32 s1, s52, s11 // (op 2890) 
s_add_u32 s1, s1, s53 // (op 2890) 

s_mul_hi_i32 s53, s16, s9 // (op 2890) 
s_add_i32 s55, s53, s16 // (op 2890) 

s_ashr_i32 s53, s55, 31 // (op 2890) 
s_cmp_eq_i32 s9, 0 // (op 2890) 
s_mov_b32 s56, scc // (op 2890) 
s_cmp_eq_i32 s56, 0 // (op 2890) 
s_cselect_b32 s57, 0, -1 // (op 2890) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 2648) 
s_lshl_b32 s58, 1, s56 // (op 2648) 
s_add_i32 s59, s58, s54 // (op 2648) 


s_and_b32 s54, s50, s59 // (op 2648) 


s_add_i32 s50, s51, s54 // (op 2648) 


s_ashr_i32 s51, s50, s56 // (op 2648) 


s_lshl_b32 s50, s45, 24 // (op 2648) 
s_ashr_i32 s54, s50, 31 // (op 2648) 

s_xor_b32 s50, s51, s54 // (op 2648) 

s_sub_i32 s51, s50, s54 // (op 2648) 


s_and_b32 s50, s45, 31 // (op 2890) 
s_lshl_b32 s54, 1, s50 // (op 2890) 
s_add_i32 s56, s54, s57 // (op 2890) 


s_and_b32 s54, s53, s56 // (op 2890) 


s_add_i32 s53, s55, s54 // (op 2890) 


s_ashr_i32 s54, s53, s50 // (op 2890) 


s_lshl_b32 s50, s45, 24 // (op 2890) 
s_ashr_i32 s53, s50, 31 // (op 2890) 

s_xor_b32 s50, s54, s53 // (op 2890) 

s_sub_i32 s54, s50, s53 // (op 2890) 


s_add_u32 s56, s6, 256 // (op 2648) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 2648) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2648) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2648) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 2648) 

s_mul_i32 s50, s11, s56 // (op 2648) 
s_mul_i32 s53, s8, s50 // (op 2648) 

s_mul_i32 s50, s51, s53 // (op 2648) 

s_sub_u32 s53, s16, s50 // (op 2648) 

s_mov_b32 s50, s53 // (op 2648) 
s_mul_hi_i32 s55, s50, s12 // (op 2648) 
s_add_i32 s58, s55, s50 // (op 2648) 


s_ashr_i32 s50, s58, 31 // (op 2648) 
s_cmp_eq_i32 s12, 0 // (op 2648) 
s_mov_b32 s55, scc // (op 2648) 
s_cmp_eq_i32 s55, 0 // (op 2648) 
s_cselect_b32 s59, 0, -1 // (op 2648) 

s_and_b32 s55, s44, 31 // (op 2648) 
s_lshl_b32 s60, 1, s55 // (op 2648) 
s_add_i32 s61, s60, s59 // (op 2648) 


s_and_b32 s59, s50, s61 // (op 2648) 


s_add_i32 s50, s58, s59 // (op 2648) 


s_ashr_i32 s58, s50, s55 // (op 2648) 


s_lshl_b32 s50, s44, 24 // (op 2648) 
s_ashr_i32 s55, s50, 31 // (op 2648) 

s_xor_b32 s50, s58, s55 // (op 2648) 

s_sub_i32 s58, s50, s55 // (op 2648) 


s_mul_i32 s50, s46, s11 // (op 2648) 
s_add_u32 s60, s20, 64 // (op 2648) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 2648) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 2648) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 2648) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 6 // (op 2648) 

s_sub_i32 s55, s60, s50 // (op 2648) 

s_mul_i32 s59, s58, s55 // (op 2648) 

s_sub_u32 s55, s53, s59 // (op 2648) 


s_add_u32 s53, s55, s50 // (op 2648) 


s_cmp_eq_i32 s51, 0 // (op 2648) 
s_mov_b32 s50, scc // (op 2648) 

s_cmp_eq_i32 s50, 0 // (op 2648) 
s_cselect_b32 s51, s53, s19 // (op 2648) 


s_mul_i32 s19, s49, s56 // (op 2648) 


v_lshl_add_u32 v11, s51, 6, v8 // (op 2648) 


s_sub_i32 s49, s18, s19 // (op 2648) 


s_cmp_eq_i32 s50, 0 // (op 2648) 
s_cselect_b32 s51, s58, s49 // (op 2648) 



v_lshl_add_u32 v8, s51, 8, v7 // (op 2648) 


v_mul_lo_u32 v7, v8, s2 // (op 2648) 

v_add_lshl_u32 v8, v11, v7, 2 // (op 2648) 


s_add_u32 s18, s6, 256 // (op 2890) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 2890) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 2890) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 2890) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 2890) 

s_mul_i32 s53, s11, s18 // (op 2890) 
s_mul_i32 s49, s8, s53 // (op 2890) 

s_mul_i32 s53, s54, s49 // (op 2890) 

s_sub_u32 s49, s16, s53 // (op 2890) 

s_mov_b32 s53, s49 // (op 2890) 
s_mul_hi_i32 s50, s53, s12 // (op 2890) 
s_add_i32 s51, s50, s53 // (op 2890) 


s_ashr_i32 s50, s51, 31 // (op 2890) 
s_cmp_eq_i32 s12, 0 // (op 2890) 
s_mov_b32 s53, scc // (op 2890) 
s_cmp_eq_i32 s53, 0 // (op 2890) 
s_cselect_b32 s55, 0, -1 // (op 2890) 

s_and_b32 s53, s44, 31 // (op 2890) 
s_lshl_b32 s56, 1, s53 // (op 2890) 
s_add_i32 s57, s56, s55 // (op 2890) 


s_and_b32 s55, s50, s57 // (op 2890) 


s_add_i32 s50, s51, s55 // (op 2890) 


s_ashr_i32 s51, s50, s53 // (op 2890) 


s_lshl_b32 s50, s44, 24 // (op 2890) 
s_ashr_i32 s53, s50, 31 // (op 2890) 

s_xor_b32 s50, s51, s53 // (op 2890) 

s_sub_i32 s51, s50, s53 // (op 2890) 


s_mul_i32 s50, s46, s11 // (op 2890) 
s_add_u32 s56, s20, 64 // (op 2890) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 2890) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2890) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2890) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 2890) 

s_sub_i32 s53, s56, s50 // (op 2890) 

s_mul_i32 s55, s51, s53 // (op 2890) 

s_sub_u32 s53, s49, s55 // (op 2890) 


s_add_u32 s49, s53, s50 // (op 2890) 


s_cmp_eq_i32 s54, 0 // (op 2890) 
s_mov_b32 s50, scc // (op 2890) 

s_cmp_eq_i32 s50, 0 // (op 2890) 
s_cselect_b32 s53, s49, s1 // (op 2890) 


s_mul_i32 s1, s52, s18 // (op 2890) 


v_lshl_add_u32 v7, s53, 6, v14 // (op 2890) 


s_sub_i32 s49, s0, s1 // (op 2890) 


s_cmp_eq_i32 s50, 0 // (op 2890) 
s_cselect_b32 s0, s51, s49 // (op 2890) 



v_lshl_add_u32 v11, s0, 8, v10 // (op 2890) 


v_mul_lo_u32 v10, v11, s4 // (op 2890) 

v_add_lshl_u32 v11, v7, v10, 2 // (op 2890) 


s_mul_i32 s1, s2, s7 // (op 2650) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 2650) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 2650) least significant
s_mul_hi_u32 s19, s2, s6 // (op 2650) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2650) most significant: sum
s_add_u32 s1, s1, s19 // (op 2650) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2650) 

s_mul_i32 s1, s4, s7 // (op 2892) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 2892) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 2892) least significant
s_mul_hi_u32 s50, s4, s6 // (op 2892) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 2892) most significant: sum
s_add_u32 s1, s1, s50 // (op 2892) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 2892) 

s_mov_b32 s52, s22 // (op 2650) reg expression
s_mov_b32 s53, s23 // (op 2650) reg expression
s_mov_b32 s54, s18 // (op 2650) reg expression
s_mov_b32 s55, 131072 // (op 2650) call()

s_mov_b32 s56, s24 // (op 2892) reg expression

s_mov_b32 s57, s25 // (op 2892) reg expression
s_mov_b32 s58, s50 // (op 2892) reg expression
s_mov_b32 s59, 131072 // (op 2892) call()


v_mov_b32 v7, s47 // (op 2) Move value
v_mov_b32 v10, s48 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 63) call()
v_accvgpr_write a5, 0.00000 // (op 63) call()
v_accvgpr_write a6, 0.00000 // (op 63) call()
v_accvgpr_write a7, 0.00000 // (op 63) call()
v_accvgpr_write a8, 0.00000 // (op 63) call()
v_accvgpr_write a9, 0.00000 // (op 63) call()
v_accvgpr_write a10, 0.00000 // (op 63) call()
v_accvgpr_write a11, 0.00000 // (op 63) call()
v_accvgpr_write a12, 0.00000 // (op 63) call()
v_accvgpr_write a13, 0.00000 // (op 63) call()
v_accvgpr_write a14, 0.00000 // (op 63) call()
v_accvgpr_write a15, 0.00000 // (op 63) call()
v_accvgpr_write a16, 0.00000 // (op 398) call()
v_accvgpr_write a17, 0.00000 // (op 398) call()
v_accvgpr_write a18, 0.00000 // (op 398) call()
v_accvgpr_write a19, 0.00000 // (op 398) call()
v_accvgpr_write a20, 0.00000 // (op 398) call()
v_accvgpr_write a21, 0.00000 // (op 398) call()
v_accvgpr_write a22, 0.00000 // (op 398) call()
v_accvgpr_write a23, 0.00000 // (op 398) call()
v_accvgpr_write a24, 0.00000 // (op 398) call()
v_accvgpr_write a25, 0.00000 // (op 398) call()
v_accvgpr_write a26, 0.00000 // (op 398) call()
v_accvgpr_write a27, 0.00000 // (op 398) call()
v_accvgpr_write a28, 0.00000 // (op 398) call()
v_accvgpr_write a29, 0.00000 // (op 398) call()
v_accvgpr_write a30, 0.00000 // (op 398) call()
v_accvgpr_write a31, 0.00000 // (op 398) call()
v_accvgpr_write a32, 0.00000 // (op 632) call()
v_accvgpr_write a33, 0.00000 // (op 632) call()
v_accvgpr_write a34, 0.00000 // (op 632) call()
v_accvgpr_write a35, 0.00000 // (op 632) call()
v_accvgpr_write a36, 0.00000 // (op 632) call()
v_accvgpr_write a37, 0.00000 // (op 632) call()
v_accvgpr_write a38, 0.00000 // (op 632) call()
v_accvgpr_write a39, 0.00000 // (op 632) call()
v_accvgpr_write a40, 0.00000 // (op 632) call()
v_accvgpr_write a41, 0.00000 // (op 632) call()
v_accvgpr_write a42, 0.00000 // (op 632) call()
v_accvgpr_write a43, 0.00000 // (op 632) call()
v_accvgpr_write a44, 0.00000 // (op 632) call()
v_accvgpr_write a45, 0.00000 // (op 632) call()
v_accvgpr_write a46, 0.00000 // (op 632) call()
v_accvgpr_write a47, 0.00000 // (op 632) call()
v_accvgpr_write a48, 0.00000 // (op 866) call()
v_accvgpr_write a49, 0.00000 // (op 866) call()
v_accvgpr_write a50, 0.00000 // (op 866) call()
v_accvgpr_write a51, 0.00000 // (op 866) call()
v_accvgpr_write a52, 0.00000 // (op 866) call()
v_accvgpr_write a53, 0.00000 // (op 866) call()
v_accvgpr_write a54, 0.00000 // (op 866) call()
v_accvgpr_write a55, 0.00000 // (op 866) call()
v_accvgpr_write a56, 0.00000 // (op 866) call()
v_accvgpr_write a57, 0.00000 // (op 866) call()
v_accvgpr_write a58, 0.00000 // (op 866) call()
v_accvgpr_write a59, 0.00000 // (op 866) call()
v_accvgpr_write a60, 0.00000 // (op 866) call()
v_accvgpr_write a61, 0.00000 // (op 866) call()
v_accvgpr_write a62, 0.00000 // (op 866) call()
v_accvgpr_write a63, 0.00000 // (op 866) call()
s_mul_hi_i32 s0, s16, s10 // (op 2415) 
v_lshrrev_b32 v14, 2, v1 // (op 2415) 
s_add_i32 s1, s0, s16 // (op 2415) 

v_and_b32 v15, -4, v1 // (op 2415) 
s_ashr_i32 s0, s1, 31 // (op 2415) 
v_sub_u32 v16, v1, v15 // (op 2415) 

s_cmp_eq_i32 s10, 0 // (op 2415) 
s_mov_b32 s18, scc // (op 2415) 
v_lshlrev_b32 v15, 5, v16 // (op 2415) 

s_cmp_eq_i32 s18, 0 // (op 2415) 
s_cselect_b32 s19, 0, -1 // (op 2415) 

v_mov_b32 v17, 0 // (op 2415) convert
v_mov_b32 v16, v15 // (op 2415) convert

s_and_b32 s18, s13, 31 // (op 2415) 
s_lshl_b32 s22, 1, s18 // (op 2415) 
s_add_i32 s23, s22, s19 // (op 2415) 


s_and_b32 s19, s0, s23 // (op 2415) 


s_add_i32 s0, s1, s19 // (op 2415) 


s_ashr_i32 s1, s0, s18 // (op 2415) 


s_lshl_b32 s0, s13, 24 // (op 2415) 
s_ashr_i32 s18, s0, 31 // (op 2415) 

s_xor_b32 s0, s1, s18 // (op 2415) 

s_sub_i32 s1, s0, s18 // (op 2415) 


s_mul_hi_i32 s0, s1, s14 // (op 2415) 
s_add_i32 s18, s0, s1 // (op 2415) 

s_ashr_i32 s0, s18, 31 // (op 2415) 
s_cmp_eq_i32 s14, 0 // (op 2415) 
s_mov_b32 s19, scc // (op 2415) 
s_cmp_eq_i32 s19, 0 // (op 2415) 
s_cselect_b32 s22, 0, -1 // (op 2415) 

s_and_b32 s19, s15, 31 // (op 2415) 
s_lshl_b32 s23, 1, s19 // (op 2415) 
s_add_i32 s24, s23, s22 // (op 2415) 


s_and_b32 s25, s0, s24 // (op 2415) 


s_add_i32 s0, s18, s25 // (op 2415) 


s_ashr_i32 s18, s0, s19 // (op 2415) 


s_lshl_b32 s0, s15, 24 // (op 2415) 
s_ashr_i32 s19, s0, 31 // (op 2415) 

s_xor_b32 s0, s18, s19 // (op 2415) 

s_sub_i32 s18, s0, s19 // (op 2415) 


s_mul_i32 s0, s1, s11 // (op 2415) 

s_sub_u32 s1, s16, s0 // (op 2415) 

s_mul_i32 s0, s18, s11 // (op 2415) 
s_add_u32 s0, s0, s1 // (op 2415) 


s_mul_hi_i32 s1, s16, s9 // (op 2415) 
s_add_i32 s18, s1, s16 // (op 2415) 

s_ashr_i32 s1, s18, 31 // (op 2415) 
s_cmp_eq_i32 s9, 0 // (op 2415) 
s_mov_b32 s19, scc // (op 2415) 
s_cmp_eq_i32 s19, 0 // (op 2415) 
s_cselect_b32 s22, 0, -1 // (op 2415) 

s_and_b32 s19, s45, 31 // (op 2415) 
s_lshl_b32 s23, 1, s19 // (op 2415) 
s_add_i32 s24, s23, s22 // (op 2415) 


s_and_b32 s25, s1, s24 // (op 2415) 


s_add_i32 s1, s18, s25 // (op 2415) 


s_ashr_i32 s18, s1, s19 // (op 2415) 


s_lshl_b32 s1, s45, 24 // (op 2415) 
s_ashr_i32 s19, s1, 31 // (op 2415) 

s_xor_b32 s1, s18, s19 // (op 2415) 

s_sub_i32 s18, s1, s19 // (op 2415) 


s_add_u32 s22, s6, 256 // (op 2415) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2415) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2415) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2415) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2415) 

s_mul_i32 s1, s11, s22 // (op 2415) 

s_mul_i32 s19, s8, s1 // (op 2415) 

s_mul_i32 s1, s18, s19 // (op 2415) 

s_sub_u32 s19, s16, s1 // (op 2415) 

s_mov_b32 s1, s19 // (op 2415) 
s_mul_hi_i32 s22, s1, s12 // (op 2415) 
s_add_i32 s23, s22, s1 // (op 2415) 


s_ashr_i32 s1, s23, 31 // (op 2415) 
s_cmp_eq_i32 s12, 0 // (op 2415) 
s_mov_b32 s22, scc // (op 2415) 
s_cmp_eq_i32 s22, 0 // (op 2415) 
s_cselect_b32 s24, 0, -1 // (op 2415) 

s_and_b32 s22, s44, 31 // (op 2415) 
s_lshl_b32 s25, 1, s22 // (op 2415) 
s_add_i32 s47, s25, s24 // (op 2415) 


s_and_b32 s24, s1, s47 // (op 2415) 


s_add_i32 s1, s23, s24 // (op 2415) 


s_ashr_i32 s23, s1, s22 // (op 2415) 


s_lshl_b32 s1, s44, 24 // (op 2415) 
s_ashr_i32 s22, s1, 31 // (op 2415) 

s_xor_b32 s1, s23, s22 // (op 2415) 

s_sub_i32 s23, s1, s22 // (op 2415) 


s_mul_i32 s1, s46, s11 // (op 2415) 
s_add_u32 s24, s20, 64 // (op 2415) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2415) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2415) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2415) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2415) 

s_sub_i32 s22, s24, s1 // (op 2415) 

s_mul_i32 s24, s23, s22 // (op 2415) 


s_sub_u32 s25, s19, s24 // (op 2415) 


s_add_u32 s19, s25, s1 // (op 2415) 


s_cmp_eq_i32 s18, 0 // (op 2415) 
s_mov_b32 s1, scc // (op 2415) 

s_cmp_eq_i32 s1, 0 // (op 2415) 
s_cselect_b32 s18, s19, s0 // (op 2415) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 2415) 


v_mul_lo_u32 v19, v15, s27 // (op 2415) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 2415) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 2415) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 2415) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 2415) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 2415) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 2415) most significant: low of high * low
v_mul_hi_u32 v16, v18, 6 // (op 2415) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 2415) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 2415) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 2415) 

v_mov_b32 v14, v16 // (op 2415) convert

s_mul_i32 s1, s21, 6 // (op 2417) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 2417) least significant
s_mul_hi_u32 s18, s20, 6 // (op 2417) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2417) most significant: sum

s_mul_i32 s19, s26, s1 // (op 2417) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 2417) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 2417) least significant
s_mul_hi_u32 s23, s26, s0 // (op 2417) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 2417) most significant: sum
s_add_u32 s19, s19, s23 // (op 2417) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2417) 

s_mov_b32 s48, s28 // (op 2417) reg expression
s_mov_b32 s49, s29 // (op 2417) reg expression
s_mov_b32 s50, s0 // (op 2417) reg expression
s_mov_b32 s51, 131072 // (op 2417) call()

s_mul_hi_i32 s47, s16, s10 // (op 2434) 
v_lshrrev_b32 v15, 2, v1 // (op 2434) 
s_add_i32 s0, s47, s16 // (op 2434) 

v_and_b32 v16, -4, v1 // (op 2434) 
s_ashr_i32 s1, s0, 31 // (op 2434) 
v_sub_u32 v17, v1, v16 // (op 2434) 

s_cmp_eq_i32 s10, 0 // (op 2434) 
s_mov_b32 s47, scc // (op 2434) 
v_lshlrev_b32 v16, 5, v17 // (op 2434) 

s_cmp_eq_i32 s47, 0 // (op 2434) 
s_cselect_b32 s18, 0, -1 // (op 2434) 

v_mov_b32 v19, 0 // (op 2434) convert
v_mov_b32 v18, v16 // (op 2434) convert

s_and_b32 s19, s13, 31 // (op 2434) 
s_lshl_b32 s47, 1, s19 // (op 2434) 
s_add_i32 s22, s47, s18 // (op 2434) 


s_and_b32 s18, s1, s22 // (op 2434) 


s_add_i32 s1, s0, s18 // (op 2434) 


s_ashr_i32 s0, s1, s19 // (op 2434) 


s_lshl_b32 s1, s13, 24 // (op 2434) 
s_ashr_i32 s47, s1, 31 // (op 2434) 

s_xor_b32 s1, s0, s47 // (op 2434) 

s_sub_i32 s0, s1, s47 // (op 2434) 


s_mul_hi_i32 s1, s0, s14 // (op 2434) 
s_add_i32 s47, s1, s0 // (op 2434) 

s_ashr_i32 s1, s47, 31 // (op 2434) 
s_cmp_eq_i32 s14, 0 // (op 2434) 
s_mov_b32 s18, scc // (op 2434) 
s_cmp_eq_i32 s18, 0 // (op 2434) 
s_cselect_b32 s19, 0, -1 // (op 2434) 

s_and_b32 s18, s15, 31 // (op 2434) 
s_lshl_b32 s22, 1, s18 // (op 2434) 
s_add_i32 s23, s22, s19 // (op 2434) 


s_and_b32 s19, s1, s23 // (op 2434) 


s_add_i32 s1, s47, s19 // (op 2434) 


s_ashr_i32 s19, s1, s18 // (op 2434) 


s_lshl_b32 s1, s15, 24 // (op 2434) 
s_ashr_i32 s18, s1, 31 // (op 2434) 

s_xor_b32 s1, s19, s18 // (op 2434) 

s_sub_i32 s19, s1, s18 // (op 2434) 


s_mul_i32 s1, s0, s11 // (op 2434) 

s_sub_u32 s0, s16, s1 // (op 2434) 

s_mul_i32 s1, s19, s11 // (op 2434) 
s_add_u32 s1, s1, s0 // (op 2434) 


s_mul_hi_i32 s0, s16, s9 // (op 2434) 
s_add_i32 s47, s0, s16 // (op 2434) 

s_ashr_i32 s0, s47, 31 // (op 2434) 
s_cmp_eq_i32 s9, 0 // (op 2434) 
s_mov_b32 s18, scc // (op 2434) 
s_cmp_eq_i32 s18, 0 // (op 2434) 
s_cselect_b32 s19, 0, -1 // (op 2434) 

s_and_b32 s18, s45, 31 // (op 2434) 
s_lshl_b32 s22, 1, s18 // (op 2434) 
s_add_i32 s23, s22, s19 // (op 2434) 


s_and_b32 s19, s0, s23 // (op 2434) 


s_add_i32 s0, s47, s19 // (op 2434) 


s_ashr_i32 s19, s0, s18 // (op 2434) 


s_lshl_b32 s0, s45, 24 // (op 2434) 
s_ashr_i32 s18, s0, 31 // (op 2434) 

s_xor_b32 s0, s19, s18 // (op 2434) 

s_sub_i32 s19, s0, s18 // (op 2434) 


s_add_u32 s22, s6, 256 // (op 2434) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2434) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2434) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2434) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2434) 

s_mul_i32 s0, s11, s22 // (op 2434) 

s_mul_i32 s18, s8, s0 // (op 2434) 

s_mul_i32 s0, s19, s18 // (op 2434) 

s_sub_u32 s18, s16, s0 // (op 2434) 

s_mov_b32 s0, s18 // (op 2434) 
s_mul_hi_i32 s47, s0, s12 // (op 2434) 
s_add_i32 s22, s47, s0 // (op 2434) 


s_ashr_i32 s0, s22, 31 // (op 2434) 
s_cmp_eq_i32 s12, 0 // (op 2434) 
s_mov_b32 s47, scc // (op 2434) 
s_cmp_eq_i32 s47, 0 // (op 2434) 
s_cselect_b32 s23, 0, -1 // (op 2434) 

s_and_b32 s47, s44, 31 // (op 2434) 
s_lshl_b32 s24, 1, s47 // (op 2434) 
s_add_i32 s25, s24, s23 // (op 2434) 


s_and_b32 s23, s0, s25 // (op 2434) 


s_add_i32 s0, s22, s23 // (op 2434) 


s_ashr_i32 s22, s0, s47 // (op 2434) 


s_lshl_b32 s0, s44, 24 // (op 2434) 
s_ashr_i32 s47, s0, 31 // (op 2434) 

s_xor_b32 s0, s22, s47 // (op 2434) 

s_sub_i32 s22, s0, s47 // (op 2434) 


s_mul_i32 s0, s46, s11 // (op 2434) 
s_add_u32 s24, s20, 64 // (op 2434) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2434) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 2434) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 2434) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 6 // (op 2434) 

s_sub_i32 s23, s24, s0 // (op 2434) 

s_mul_i32 s47, s22, s23 // (op 2434) 


s_sub_u32 s22, s18, s47 // (op 2434) 


s_add_u32 s18, s22, s0 // (op 2434) 


s_cmp_eq_i32 s19, 0 // (op 2434) 
s_mov_b32 s0, scc // (op 2434) 

s_cmp_eq_i32 s0, 0 // (op 2434) 
s_cselect_b32 s19, s18, s1 // (op 2434) 



v_lshl_add_u32 v16, s19, 6, v15 // (op 2434) 


v_mul_lo_u32 v21, v16, s27 // (op 2434) most significant: low of low * high
v_mul_hi_u32 v15, v16, s26 // (op 2434) most significant: high of low * low
v_mul_lo_u32 v20, v16, s26 // (op 2434) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 2434) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 2434) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 2434) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 2434) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 2434) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 2434) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 2434) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 2434) 

v_mov_b32 v15, v16 // (op 2434) convert

s_mul_i32 s1, s21, 6 // (op 2436) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 2436) least significant
s_mul_hi_u32 s47, s20, 6 // (op 2436) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2436) most significant: sum

s_mul_i32 s19, s26, s1 // (op 2436) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 2436) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 2436) least significant
s_mul_hi_u32 s22, s26, s0 // (op 2436) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 2436) most significant: sum
s_add_u32 s19, s19, s22 // (op 2436) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2436) 

s_mov_b32 s48, s28 // (op 2436) reg expression
s_mov_b32 s49, s29 // (op 2436) reg expression
s_mov_b32 s50, s0 // (op 2436) reg expression
s_mov_b32 s51, 131072 // (op 2436) call()

s_mul_hi_i32 s47, s16, s10 // (op 2479) 
v_and_b32 v16, 4294967294, v1 // (op 2479) 
s_add_i32 s0, s47, s16 // (op 2479) 

v_and_b32 v17, -2, v1 // (op 2479) 
s_ashr_i32 s1, s0, 31 // (op 2479) 
v_sub_u32 v18, v1, v17 // (op 2479) 

s_cmp_eq_i32 s10, 0 // (op 2479) 
s_mov_b32 s47, scc // (op 2479) 
v_lshlrev_b32 v17, 6, v18 // (op 2479) 

s_cmp_eq_i32 s47, 0 // (op 2479) 
s_cselect_b32 s18, 0, -1 // (op 2479) 

v_mov_b32 v19, 0 // (op 2479) convert
v_mov_b32 v18, v17 // (op 2479) convert

s_and_b32 s19, s13, 31 // (op 2479) 
s_lshl_b32 s47, 1, s19 // (op 2479) 
s_add_i32 s22, s47, s18 // (op 2479) 


s_and_b32 s18, s1, s22 // (op 2479) 


s_add_i32 s1, s0, s18 // (op 2479) 


s_ashr_i32 s0, s1, s19 // (op 2479) 


s_lshl_b32 s1, s13, 24 // (op 2479) 
s_ashr_i32 s47, s1, 31 // (op 2479) 

s_xor_b32 s1, s0, s47 // (op 2479) 

s_sub_i32 s0, s1, s47 // (op 2479) 


s_mul_hi_i32 s1, s0, s14 // (op 2479) 
s_add_i32 s47, s1, s0 // (op 2479) 

s_ashr_i32 s1, s47, 31 // (op 2479) 
s_cmp_eq_i32 s14, 0 // (op 2479) 
s_mov_b32 s18, scc // (op 2479) 
s_cmp_eq_i32 s18, 0 // (op 2479) 
s_cselect_b32 s19, 0, -1 // (op 2479) 

s_and_b32 s18, s15, 31 // (op 2479) 
s_lshl_b32 s22, 1, s18 // (op 2479) 
s_add_i32 s23, s22, s19 // (op 2479) 


s_and_b32 s19, s1, s23 // (op 2479) 


s_add_i32 s1, s47, s19 // (op 2479) 


s_ashr_i32 s19, s1, s18 // (op 2479) 


s_lshl_b32 s1, s15, 24 // (op 2479) 
s_ashr_i32 s18, s1, 31 // (op 2479) 

s_xor_b32 s1, s19, s18 // (op 2479) 

s_sub_i32 s19, s1, s18 // (op 2479) 


s_add_u32 s22, s6, 256 // (op 2479) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2479) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2479) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2479) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2479) 

s_mul_i32 s1, s19, s22 // (op 2479) 

s_sub_i32 s47, s0, s1 // (op 2479) 


s_mul_hi_i32 s0, s16, s9 // (op 2479) 
s_add_i32 s1, s0, s16 // (op 2479) 

s_ashr_i32 s0, s1, 31 // (op 2479) 
s_cmp_eq_i32 s9, 0 // (op 2479) 
s_mov_b32 s18, scc // (op 2479) 
s_cmp_eq_i32 s18, 0 // (op 2479) 
s_cselect_b32 s19, 0, -1 // (op 2479) 

s_and_b32 s18, s45, 31 // (op 2479) 
s_lshl_b32 s24, 1, s18 // (op 2479) 
s_add_i32 s25, s24, s19 // (op 2479) 


s_and_b32 s19, s0, s25 // (op 2479) 


s_add_i32 s0, s1, s19 // (op 2479) 


s_ashr_i32 s1, s0, s18 // (op 2479) 


s_lshl_b32 s0, s45, 24 // (op 2479) 
s_ashr_i32 s18, s0, 31 // (op 2479) 

s_xor_b32 s0, s1, s18 // (op 2479) 

s_sub_i32 s1, s0, s18 // (op 2479) 


s_mul_i32 s0, s11, s22 // (op 2479) 

s_mul_i32 s18, s8, s0 // (op 2479) 

s_mul_i32 s0, s1, s18 // (op 2479) 

s_sub_u32 s18, s16, s0 // (op 2479) 

s_mul_hi_i32 s0, s18, s12 // (op 2479) 
s_add_i32 s19, s0, s18 // (op 2479) 


s_ashr_i32 s0, s19, 31 // (op 2479) 
s_cmp_eq_i32 s12, 0 // (op 2479) 
s_mov_b32 s18, scc // (op 2479) 
s_cmp_eq_i32 s18, 0 // (op 2479) 
s_cselect_b32 s22, 0, -1 // (op 2479) 

s_and_b32 s18, s44, 31 // (op 2479) 
s_lshl_b32 s23, 1, s18 // (op 2479) 
s_add_i32 s24, s23, s22 // (op 2479) 


s_and_b32 s25, s0, s24 // (op 2479) 


s_add_i32 s0, s19, s25 // (op 2479) 


s_ashr_i32 s19, s0, s18 // (op 2479) 


s_lshl_b32 s0, s44, 24 // (op 2479) 
s_ashr_i32 s18, s0, 31 // (op 2479) 

s_xor_b32 s0, s19, s18 // (op 2479) 

s_sub_i32 s19, s0, s18 // (op 2479) 


s_cmp_eq_i32 s1, 0 // (op 2479) 
s_mov_b32 s0, scc // (op 2479) 

s_cmp_eq_i32 s0, 0 // (op 2479) 
s_cselect_b32 s1, s19, s47 // (op 2479) 



v_lshl_add_u32 v17, s1, 8, v16 // (op 2479) 


v_mul_lo_u32 v21, v17, s31 // (op 2479) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 2479) most significant: high of low * low
v_mul_lo_u32 v20, v17, s30 // (op 2479) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2479) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 2479) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 2479) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 2479) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 2479) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 2479) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 2479) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 2479) 

v_mov_b32 v18, v16 // (op 2479) convert

s_mul_i32 s1, s7, 6 // (op 2481) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 2481) least significant
s_mul_hi_u32 s47, s6, 6 // (op 2481) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2481) most significant: sum

s_mul_i32 s19, s30, s1 // (op 2481) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 2481) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 2481) least significant
s_mul_hi_u32 s22, s30, s0 // (op 2481) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 2481) most significant: sum
s_add_u32 s19, s19, s22 // (op 2481) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2481) 

s_mov_b32 s60, s32 // (op 2481) reg expression
s_mov_b32 s61, s33 // (op 2481) reg expression
s_mov_b32 s62, s0 // (op 2481) reg expression
s_mov_b32 s63, 131072 // (op 2481) call()

s_mul_hi_i32 s47, s16, s10 // (op 2498) 
v_and_b32 v16, 4294967294, v1 // (op 2498) 
s_add_i32 s0, s47, s16 // (op 2498) 

v_and_b32 v17, -2, v1 // (op 2498) 
s_ashr_i32 s1, s0, 31 // (op 2498) 
v_sub_u32 v19, v1, v17 // (op 2498) 

s_cmp_eq_i32 s10, 0 // (op 2498) 
s_mov_b32 s47, scc // (op 2498) 
v_lshlrev_b32 v17, 6, v19 // (op 2498) 

s_cmp_eq_i32 s47, 0 // (op 2498) 
s_cselect_b32 s18, 0, -1 // (op 2498) 

v_mov_b32 v21, 0 // (op 2498) convert
v_mov_b32 v20, v17 // (op 2498) convert

s_and_b32 s19, s13, 31 // (op 2498) 
s_lshl_b32 s47, 1, s19 // (op 2498) 
s_add_i32 s22, s47, s18 // (op 2498) 


s_and_b32 s18, s1, s22 // (op 2498) 


s_add_i32 s1, s0, s18 // (op 2498) 


s_ashr_i32 s0, s1, s19 // (op 2498) 


s_lshl_b32 s1, s13, 24 // (op 2498) 
s_ashr_i32 s47, s1, 31 // (op 2498) 

s_xor_b32 s1, s0, s47 // (op 2498) 

s_sub_i32 s0, s1, s47 // (op 2498) 


s_mul_hi_i32 s1, s0, s14 // (op 2498) 
s_add_i32 s47, s1, s0 // (op 2498) 

s_ashr_i32 s1, s47, 31 // (op 2498) 
s_cmp_eq_i32 s14, 0 // (op 2498) 
s_mov_b32 s18, scc // (op 2498) 
s_cmp_eq_i32 s18, 0 // (op 2498) 
s_cselect_b32 s19, 0, -1 // (op 2498) 

s_and_b32 s18, s15, 31 // (op 2498) 
s_lshl_b32 s22, 1, s18 // (op 2498) 
s_add_i32 s23, s22, s19 // (op 2498) 


s_and_b32 s19, s1, s23 // (op 2498) 


s_add_i32 s1, s47, s19 // (op 2498) 


s_ashr_i32 s19, s1, s18 // (op 2498) 


s_lshl_b32 s1, s15, 24 // (op 2498) 
s_ashr_i32 s18, s1, 31 // (op 2498) 

s_xor_b32 s1, s19, s18 // (op 2498) 

s_sub_i32 s19, s1, s18 // (op 2498) 


s_add_u32 s22, s6, 256 // (op 2498) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2498) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2498) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2498) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2498) 

s_mul_i32 s1, s19, s22 // (op 2498) 

s_sub_i32 s47, s0, s1 // (op 2498) 


s_mul_hi_i32 s0, s16, s9 // (op 2498) 
s_add_i32 s1, s0, s16 // (op 2498) 

s_ashr_i32 s0, s1, 31 // (op 2498) 
s_cmp_eq_i32 s9, 0 // (op 2498) 
s_mov_b32 s18, scc // (op 2498) 
s_cmp_eq_i32 s18, 0 // (op 2498) 
s_cselect_b32 s19, 0, -1 // (op 2498) 

s_and_b32 s18, s45, 31 // (op 2498) 
s_lshl_b32 s24, 1, s18 // (op 2498) 
s_add_i32 s25, s24, s19 // (op 2498) 


s_and_b32 s19, s0, s25 // (op 2498) 


s_add_i32 s0, s1, s19 // (op 2498) 


s_ashr_i32 s1, s0, s18 // (op 2498) 


s_lshl_b32 s0, s45, 24 // (op 2498) 
s_ashr_i32 s18, s0, 31 // (op 2498) 

s_xor_b32 s0, s1, s18 // (op 2498) 

s_sub_i32 s1, s0, s18 // (op 2498) 


s_mul_i32 s0, s11, s22 // (op 2498) 

s_mul_i32 s18, s8, s0 // (op 2498) 

s_mul_i32 s0, s1, s18 // (op 2498) 

s_sub_u32 s18, s16, s0 // (op 2498) 

s_mul_hi_i32 s0, s18, s12 // (op 2498) 
s_add_i32 s19, s0, s18 // (op 2498) 


s_ashr_i32 s0, s19, 31 // (op 2498) 
s_cmp_eq_i32 s12, 0 // (op 2498) 
s_mov_b32 s18, scc // (op 2498) 
s_cmp_eq_i32 s18, 0 // (op 2498) 
s_cselect_b32 s22, 0, -1 // (op 2498) 

s_and_b32 s18, s44, 31 // (op 2498) 
s_lshl_b32 s23, 1, s18 // (op 2498) 
s_add_i32 s24, s23, s22 // (op 2498) 


s_and_b32 s25, s0, s24 // (op 2498) 


s_add_i32 s0, s19, s25 // (op 2498) 


s_ashr_i32 s19, s0, s18 // (op 2498) 


s_lshl_b32 s0, s44, 24 // (op 2498) 
s_ashr_i32 s18, s0, 31 // (op 2498) 

s_xor_b32 s0, s19, s18 // (op 2498) 

s_sub_i32 s19, s0, s18 // (op 2498) 


s_cmp_eq_i32 s1, 0 // (op 2498) 
s_mov_b32 s0, scc // (op 2498) 

s_cmp_eq_i32 s0, 0 // (op 2498) 
s_cselect_b32 s1, s19, s47 // (op 2498) 



v_lshl_add_u32 v17, s1, 8, v16 // (op 2498) 


v_mul_lo_u32 v23, v17, s31 // (op 2498) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 2498) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 2498) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 2498) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 2498) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 2498) most significant half



v_mul_lo_u32 v21, v17, 6 // (op 2498) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 2498) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 2498) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 2498) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 2498) 

v_mov_b32 v19, v16 // (op 2498) convert

s_mul_i32 s1, s7, 6 // (op 2500) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 2500) least significant
s_mul_hi_u32 s47, s6, 6 // (op 2500) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2500) most significant: sum

s_mul_i32 s19, s30, s1 // (op 2500) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 2500) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 2500) least significant
s_mul_hi_u32 s22, s30, s0 // (op 2500) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 2500) most significant: sum
s_add_u32 s19, s19, s22 // (op 2500) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2500) 

s_mov_b32 s60, s32 // (op 2500) reg expression
s_mov_b32 s61, s33 // (op 2500) reg expression
s_mov_b32 s62, s0 // (op 2500) reg expression
s_mov_b32 s63, 131072 // (op 2500) call()

s_mul_hi_i32 s47, s16, s10 // (op 2543) 
v_lshrrev_b32 v16, 2, v1 // (op 2543) 
s_add_i32 s0, s47, s16 // (op 2543) 

v_and_b32 v17, -4, v1 // (op 2543) 
s_ashr_i32 s1, s0, 31 // (op 2543) 
v_sub_u32 v20, v1, v17 // (op 2543) 

s_cmp_eq_i32 s10, 0 // (op 2543) 
s_mov_b32 s47, scc // (op 2543) 
s_cmp_eq_i32 s47, 0 // (op 2543) 
s_cselect_b32 s18, 0, -1 // (op 2543) 

s_and_b32 s19, s13, 31 // (op 2543) 
s_lshl_b32 s47, 1, s19 // (op 2543) 
s_add_i32 s22, s47, s18 // (op 2543) 


s_and_b32 s18, s1, s22 // (op 2543) 


s_add_i32 s1, s0, s18 // (op 2543) 


s_ashr_i32 s0, s1, s19 // (op 2543) 


s_lshl_b32 s1, s13, 24 // (op 2543) 
s_ashr_i32 s47, s1, 31 // (op 2543) 

s_xor_b32 s1, s0, s47 // (op 2543) 

s_sub_i32 s0, s1, s47 // (op 2543) 


s_mul_hi_i32 s1, s0, s14 // (op 2543) 
s_add_i32 s47, s1, s0 // (op 2543) 

s_ashr_i32 s1, s47, 31 // (op 2543) 
s_cmp_eq_i32 s14, 0 // (op 2543) 
s_mov_b32 s18, scc // (op 2543) 
s_cmp_eq_i32 s18, 0 // (op 2543) 
s_cselect_b32 s19, 0, -1 // (op 2543) 

s_and_b32 s18, s15, 31 // (op 2543) 
s_lshl_b32 s22, 1, s18 // (op 2543) 
s_add_i32 s23, s22, s19 // (op 2543) 


s_and_b32 s19, s1, s23 // (op 2543) 


s_add_i32 s1, s47, s19 // (op 2543) 


s_ashr_i32 s19, s1, s18 // (op 2543) 


s_lshl_b32 s1, s15, 24 // (op 2543) 
s_ashr_i32 s18, s1, 31 // (op 2543) 

s_xor_b32 s1, s19, s18 // (op 2543) 

s_sub_i32 s19, s1, s18 // (op 2543) 


s_mul_i32 s1, s0, s11 // (op 2543) 

s_sub_u32 s0, s16, s1 // (op 2543) 

s_mul_i32 s1, s19, s11 // (op 2543) 
s_add_u32 s1, s1, s0 // (op 2543) 


s_mul_hi_i32 s0, s16, s9 // (op 2543) 
s_add_i32 s47, s0, s16 // (op 2543) 

s_ashr_i32 s0, s47, 31 // (op 2543) 
s_cmp_eq_i32 s9, 0 // (op 2543) 
s_mov_b32 s18, scc // (op 2543) 
s_cmp_eq_i32 s18, 0 // (op 2543) 
s_cselect_b32 s19, 0, -1 // (op 2543) 

s_and_b32 s18, s45, 31 // (op 2543) 
s_lshl_b32 s22, 1, s18 // (op 2543) 
s_add_i32 s23, s22, s19 // (op 2543) 


s_and_b32 s19, s0, s23 // (op 2543) 


s_add_i32 s0, s47, s19 // (op 2543) 


s_ashr_i32 s19, s0, s18 // (op 2543) 


s_lshl_b32 s0, s45, 24 // (op 2543) 
s_ashr_i32 s18, s0, 31 // (op 2543) 

s_xor_b32 s0, s19, s18 // (op 2543) 

s_sub_i32 s19, s0, s18 // (op 2543) 


s_add_u32 s22, s6, 256 // (op 2543) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2543) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2543) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2543) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2543) 

s_mul_i32 s0, s11, s22 // (op 2543) 

s_mul_i32 s18, s8, s0 // (op 2543) 

s_mul_i32 s0, s19, s18 // (op 2543) 

s_sub_u32 s18, s16, s0 // (op 2543) 

s_mov_b32 s0, s18 // (op 2543) 
s_mul_hi_i32 s47, s0, s12 // (op 2543) 
s_add_i32 s22, s47, s0 // (op 2543) 


s_ashr_i32 s0, s22, 31 // (op 2543) 
s_cmp_eq_i32 s12, 0 // (op 2543) 
s_mov_b32 s47, scc // (op 2543) 
s_cmp_eq_i32 s47, 0 // (op 2543) 
s_cselect_b32 s23, 0, -1 // (op 2543) 

s_and_b32 s47, s44, 31 // (op 2543) 
s_lshl_b32 s24, 1, s47 // (op 2543) 
s_add_i32 s25, s24, s23 // (op 2543) 


s_and_b32 s23, s0, s25 // (op 2543) 


s_add_i32 s0, s22, s23 // (op 2543) 


s_ashr_i32 s22, s0, s47 // (op 2543) 


s_lshl_b32 s0, s44, 24 // (op 2543) 
s_ashr_i32 s47, s0, 31 // (op 2543) 

s_xor_b32 s0, s22, s47 // (op 2543) 

s_sub_i32 s22, s0, s47 // (op 2543) 


s_mul_i32 s0, s46, s11 // (op 2543) 
s_add_u32 s24, s20, 64 // (op 2543) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2543) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 2543) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 2543) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 6 // (op 2543) 

s_sub_i32 s23, s24, s0 // (op 2543) 

s_mul_i32 s47, s22, s23 // (op 2543) 


s_sub_u32 s22, s18, s47 // (op 2543) 


s_add_u32 s18, s22, s0 // (op 2543) 


s_cmp_eq_i32 s19, 0 // (op 2543) 
s_mov_b32 s0, scc // (op 2543) 

s_cmp_eq_i32 s0, 0 // (op 2543) 
s_cselect_b32 s19, s18, s1 // (op 2543) 



v_lshl_add_u32 v17, s19, 6, v16 // (op 2543) 


v_mul_lo_u32 v16, v17, s34 // (op 2543) 
v_add_u32 v16, v16, v20 // (op 2543) 


s_mul_i32 s1, s34, s21 // (op 2545) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 2545) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 2545) least significant
s_mul_hi_u32 s18, s34, s20 // (op 2545) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2545) most significant: sum
s_add_u32 s1, s1, s18 // (op 2545) most significant: sum


s_mov_b32 s64, s36 // (op 2545) reg expression
s_mov_b32 s65, s37 // (op 2545) reg expression
s_mov_b32 s66, s0 // (op 2545) reg expression
s_mov_b32 s67, 131072 // (op 2545) call()

s_mul_hi_i32 s47, s16, s10 // (op 2562) 
v_lshrrev_b32 v17, 2, v1 // (op 2562) 
s_add_i32 s0, s47, s16 // (op 2562) 

v_and_b32 v20, -4, v1 // (op 2562) 
s_ashr_i32 s1, s0, 31 // (op 2562) 
v_sub_u32 v21, v1, v20 // (op 2562) 

s_cmp_eq_i32 s10, 0 // (op 2562) 
s_mov_b32 s47, scc // (op 2562) 
s_cmp_eq_i32 s47, 0 // (op 2562) 
s_cselect_b32 s18, 0, -1 // (op 2562) 

s_and_b32 s19, s13, 31 // (op 2562) 
s_lshl_b32 s47, 1, s19 // (op 2562) 
s_add_i32 s22, s47, s18 // (op 2562) 


s_and_b32 s18, s1, s22 // (op 2562) 


s_add_i32 s1, s0, s18 // (op 2562) 


s_ashr_i32 s0, s1, s19 // (op 2562) 


s_lshl_b32 s1, s13, 24 // (op 2562) 
s_ashr_i32 s47, s1, 31 // (op 2562) 

s_xor_b32 s1, s0, s47 // (op 2562) 

s_sub_i32 s0, s1, s47 // (op 2562) 


s_mul_hi_i32 s1, s0, s14 // (op 2562) 
s_add_i32 s47, s1, s0 // (op 2562) 

s_ashr_i32 s1, s47, 31 // (op 2562) 
s_cmp_eq_i32 s14, 0 // (op 2562) 
s_mov_b32 s18, scc // (op 2562) 
s_cmp_eq_i32 s18, 0 // (op 2562) 
s_cselect_b32 s19, 0, -1 // (op 2562) 

s_and_b32 s18, s15, 31 // (op 2562) 
s_lshl_b32 s22, 1, s18 // (op 2562) 
s_add_i32 s23, s22, s19 // (op 2562) 


s_and_b32 s19, s1, s23 // (op 2562) 


s_add_i32 s1, s47, s19 // (op 2562) 


s_ashr_i32 s19, s1, s18 // (op 2562) 


s_lshl_b32 s1, s15, 24 // (op 2562) 
s_ashr_i32 s18, s1, 31 // (op 2562) 

s_xor_b32 s1, s19, s18 // (op 2562) 

s_sub_i32 s19, s1, s18 // (op 2562) 


s_mul_i32 s1, s0, s11 // (op 2562) 

s_sub_u32 s0, s16, s1 // (op 2562) 

s_mul_i32 s1, s19, s11 // (op 2562) 
s_add_u32 s1, s1, s0 // (op 2562) 


s_mul_hi_i32 s0, s16, s9 // (op 2562) 
s_add_i32 s47, s0, s16 // (op 2562) 

s_ashr_i32 s0, s47, 31 // (op 2562) 
s_cmp_eq_i32 s9, 0 // (op 2562) 
s_mov_b32 s18, scc // (op 2562) 
s_cmp_eq_i32 s18, 0 // (op 2562) 
s_cselect_b32 s19, 0, -1 // (op 2562) 

s_and_b32 s18, s45, 31 // (op 2562) 
s_lshl_b32 s22, 1, s18 // (op 2562) 
s_add_i32 s23, s22, s19 // (op 2562) 


s_and_b32 s19, s0, s23 // (op 2562) 


s_add_i32 s0, s47, s19 // (op 2562) 


s_ashr_i32 s19, s0, s18 // (op 2562) 


s_lshl_b32 s0, s45, 24 // (op 2562) 
s_ashr_i32 s18, s0, 31 // (op 2562) 

s_xor_b32 s0, s19, s18 // (op 2562) 

s_sub_i32 s19, s0, s18 // (op 2562) 


s_add_u32 s22, s6, 256 // (op 2562) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2562) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2562) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2562) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2562) 

s_mul_i32 s0, s11, s22 // (op 2562) 

s_mul_i32 s18, s8, s0 // (op 2562) 

s_mul_i32 s0, s19, s18 // (op 2562) 

s_sub_u32 s18, s16, s0 // (op 2562) 

s_mov_b32 s0, s18 // (op 2562) 
s_mul_hi_i32 s47, s0, s12 // (op 2562) 
s_add_i32 s22, s47, s0 // (op 2562) 


s_ashr_i32 s0, s22, 31 // (op 2562) 
s_cmp_eq_i32 s12, 0 // (op 2562) 
s_mov_b32 s47, scc // (op 2562) 
s_cmp_eq_i32 s47, 0 // (op 2562) 
s_cselect_b32 s23, 0, -1 // (op 2562) 

s_and_b32 s47, s44, 31 // (op 2562) 
s_lshl_b32 s24, 1, s47 // (op 2562) 
s_add_i32 s25, s24, s23 // (op 2562) 


s_and_b32 s23, s0, s25 // (op 2562) 


s_add_i32 s0, s22, s23 // (op 2562) 


s_ashr_i32 s22, s0, s47 // (op 2562) 


s_lshl_b32 s0, s44, 24 // (op 2562) 
s_ashr_i32 s47, s0, 31 // (op 2562) 

s_xor_b32 s0, s22, s47 // (op 2562) 

s_sub_i32 s22, s0, s47 // (op 2562) 


s_mul_i32 s0, s46, s11 // (op 2562) 
s_add_u32 s24, s20, 64 // (op 2562) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2562) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 2562) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 2562) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 6 // (op 2562) 

s_sub_i32 s23, s24, s0 // (op 2562) 

s_mul_i32 s47, s22, s23 // (op 2562) 


s_sub_u32 s22, s18, s47 // (op 2562) 


s_add_u32 s18, s22, s0 // (op 2562) 


s_cmp_eq_i32 s19, 0 // (op 2562) 
s_mov_b32 s0, scc // (op 2562) 

s_cmp_eq_i32 s0, 0 // (op 2562) 
s_cselect_b32 s19, s18, s1 // (op 2562) 



v_lshl_add_u32 v20, s19, 6, v17 // (op 2562) 


v_mul_lo_u32 v17, v20, s34 // (op 2562) 
v_add_u32 v17, v17, v21 // (op 2562) 


s_mul_i32 s1, s34, s21 // (op 2564) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 2564) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 2564) least significant
s_mul_hi_u32 s18, s34, s20 // (op 2564) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2564) most significant: sum
s_add_u32 s1, s1, s18 // (op 2564) most significant: sum


s_mov_b32 s64, s36 // (op 2564) reg expression
s_mov_b32 s65, s37 // (op 2564) reg expression
s_mov_b32 s66, s0 // (op 2564) reg expression
s_mov_b32 s67, 131072 // (op 2564) call()

s_mul_hi_i32 s47, s16, s10 // (op 2607) 
s_add_i32 s0, s47, s16 // (op 2607) 

s_ashr_i32 s1, s0, 31 // (op 2607) 
s_cmp_eq_i32 s10, 0 // (op 2607) 
s_mov_b32 s47, scc // (op 2607) 
s_cmp_eq_i32 s47, 0 // (op 2607) 
s_cselect_b32 s18, 0, -1 // (op 2607) 

s_and_b32 s19, s13, 31 // (op 2607) 
s_lshl_b32 s47, 1, s19 // (op 2607) 
s_add_i32 s22, s47, s18 // (op 2607) 


s_and_b32 s18, s1, s22 // (op 2607) 


s_add_i32 s1, s0, s18 // (op 2607) 


s_ashr_i32 s0, s1, s19 // (op 2607) 


s_lshl_b32 s1, s13, 24 // (op 2607) 
s_ashr_i32 s47, s1, 31 // (op 2607) 

s_xor_b32 s1, s0, s47 // (op 2607) 

s_sub_i32 s0, s1, s47 // (op 2607) 


s_mul_hi_i32 s1, s0, s14 // (op 2607) 
s_add_i32 s47, s1, s0 // (op 2607) 

s_ashr_i32 s1, s47, 31 // (op 2607) 
s_cmp_eq_i32 s14, 0 // (op 2607) 
s_mov_b32 s18, scc // (op 2607) 
s_cmp_eq_i32 s18, 0 // (op 2607) 
s_cselect_b32 s19, 0, -1 // (op 2607) 

s_and_b32 s18, s15, 31 // (op 2607) 
s_lshl_b32 s22, 1, s18 // (op 2607) 
s_add_i32 s23, s22, s19 // (op 2607) 


s_and_b32 s19, s1, s23 // (op 2607) 


s_add_i32 s1, s47, s19 // (op 2607) 


s_ashr_i32 s19, s1, s18 // (op 2607) 


s_lshl_b32 s1, s15, 24 // (op 2607) 
s_ashr_i32 s18, s1, 31 // (op 2607) 

s_xor_b32 s1, s19, s18 // (op 2607) 

s_sub_i32 s19, s1, s18 // (op 2607) 


s_add_u32 s22, s6, 256 // (op 2607) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2607) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2607) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2607) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2607) 

s_mul_i32 s1, s19, s22 // (op 2607) 

s_sub_i32 s47, s0, s1 // (op 2607) 


s_mul_hi_i32 s0, s16, s9 // (op 2607) 
s_add_i32 s1, s0, s16 // (op 2607) 

s_ashr_i32 s0, s1, 31 // (op 2607) 
s_cmp_eq_i32 s9, 0 // (op 2607) 
s_mov_b32 s18, scc // (op 2607) 
s_cmp_eq_i32 s18, 0 // (op 2607) 
s_cselect_b32 s19, 0, -1 // (op 2607) 

s_and_b32 s18, s45, 31 // (op 2607) 
s_lshl_b32 s24, 1, s18 // (op 2607) 
s_add_i32 s25, s24, s19 // (op 2607) 


s_and_b32 s19, s0, s25 // (op 2607) 


s_add_i32 s0, s1, s19 // (op 2607) 


s_ashr_i32 s1, s0, s18 // (op 2607) 


s_lshl_b32 s0, s45, 24 // (op 2607) 
s_ashr_i32 s18, s0, 31 // (op 2607) 

s_xor_b32 s0, s1, s18 // (op 2607) 

s_sub_i32 s1, s0, s18 // (op 2607) 


s_mul_i32 s0, s11, s22 // (op 2607) 

s_mul_i32 s18, s8, s0 // (op 2607) 

s_mul_i32 s0, s1, s18 // (op 2607) 

s_sub_u32 s18, s16, s0 // (op 2607) 

s_mul_hi_i32 s0, s18, s12 // (op 2607) 
s_add_i32 s19, s0, s18 // (op 2607) 


s_ashr_i32 s0, s19, 31 // (op 2607) 
s_cmp_eq_i32 s12, 0 // (op 2607) 
s_mov_b32 s18, scc // (op 2607) 
s_cmp_eq_i32 s18, 0 // (op 2607) 
s_cselect_b32 s22, 0, -1 // (op 2607) 

s_and_b32 s18, s44, 31 // (op 2607) 
s_lshl_b32 s23, 1, s18 // (op 2607) 
s_add_i32 s24, s23, s22 // (op 2607) 


s_and_b32 s25, s0, s24 // (op 2607) 


s_add_i32 s0, s19, s25 // (op 2607) 


s_ashr_i32 s19, s0, s18 // (op 2607) 


s_lshl_b32 s0, s44, 24 // (op 2607) 
s_ashr_i32 s18, s0, 31 // (op 2607) 

s_xor_b32 s0, s19, s18 // (op 2607) 

s_sub_i32 s19, s0, s18 // (op 2607) 


s_cmp_eq_i32 s1, 0 // (op 2607) 
s_mov_b32 s0, scc // (op 2607) 

s_cmp_eq_i32 s0, 0 // (op 2607) 
s_cselect_b32 s1, s19, s47 // (op 2607) 



v_lshl_add_u32 v20, s1, 8, v1 // (op 2607) 

v_mul_lo_u32 v21, v20, s38 // (op 2607) 

s_mul_i32 s1, s38, s7 // (op 2609) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 2609) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 2609) least significant
s_mul_hi_u32 s18, s38, s6 // (op 2609) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2609) most significant: sum
s_add_u32 s1, s1, s18 // (op 2609) most significant: sum


s_mov_b32 s68, s40 // (op 2609) reg expression
s_mov_b32 s69, s41 // (op 2609) reg expression
s_mov_b32 s70, s0 // (op 2609) reg expression
s_mov_b32 s71, 131072 // (op 2609) call()

s_mul_hi_i32 s47, s16, s10 // (op 2626) 
s_add_i32 s0, s47, s16 // (op 2626) 

s_ashr_i32 s1, s0, 31 // (op 2626) 
s_cmp_eq_i32 s10, 0 // (op 2626) 
s_mov_b32 s47, scc // (op 2626) 
s_cmp_eq_i32 s47, 0 // (op 2626) 
s_cselect_b32 s18, 0, -1 // (op 2626) 

s_and_b32 s19, s13, 31 // (op 2626) 
s_lshl_b32 s47, 1, s19 // (op 2626) 
s_add_i32 s22, s47, s18 // (op 2626) 


s_and_b32 s18, s1, s22 // (op 2626) 


s_add_i32 s1, s0, s18 // (op 2626) 


s_ashr_i32 s0, s1, s19 // (op 2626) 


s_lshl_b32 s1, s13, 24 // (op 2626) 
s_ashr_i32 s47, s1, 31 // (op 2626) 

s_xor_b32 s1, s0, s47 // (op 2626) 

s_sub_i32 s0, s1, s47 // (op 2626) 


s_mul_hi_i32 s1, s0, s14 // (op 2626) 
s_add_i32 s47, s1, s0 // (op 2626) 

s_ashr_i32 s1, s47, 31 // (op 2626) 
s_cmp_eq_i32 s14, 0 // (op 2626) 
s_mov_b32 s18, scc // (op 2626) 
s_cmp_eq_i32 s18, 0 // (op 2626) 
s_cselect_b32 s19, 0, -1 // (op 2626) 

s_and_b32 s18, s15, 31 // (op 2626) 
s_lshl_b32 s22, 1, s18 // (op 2626) 
s_add_i32 s23, s22, s19 // (op 2626) 


s_and_b32 s19, s1, s23 // (op 2626) 


s_add_i32 s1, s47, s19 // (op 2626) 


s_ashr_i32 s19, s1, s18 // (op 2626) 


s_lshl_b32 s1, s15, 24 // (op 2626) 
s_ashr_i32 s18, s1, 31 // (op 2626) 

s_xor_b32 s1, s19, s18 // (op 2626) 

s_sub_i32 s19, s1, s18 // (op 2626) 


s_add_u32 s22, s6, 256 // (op 2626) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2626) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2626) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2626) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2626) 

s_mul_i32 s1, s19, s22 // (op 2626) 

s_sub_i32 s47, s0, s1 // (op 2626) 


s_mul_hi_i32 s0, s16, s9 // (op 2626) 
s_add_i32 s1, s0, s16 // (op 2626) 

s_ashr_i32 s0, s1, 31 // (op 2626) 
s_cmp_eq_i32 s9, 0 // (op 2626) 
s_mov_b32 s18, scc // (op 2626) 
s_cmp_eq_i32 s18, 0 // (op 2626) 
s_cselect_b32 s19, 0, -1 // (op 2626) 

s_and_b32 s18, s45, 31 // (op 2626) 
s_lshl_b32 s24, 1, s18 // (op 2626) 
s_add_i32 s25, s24, s19 // (op 2626) 


s_and_b32 s19, s0, s25 // (op 2626) 


s_add_i32 s0, s1, s19 // (op 2626) 


s_ashr_i32 s1, s0, s18 // (op 2626) 


s_lshl_b32 s0, s45, 24 // (op 2626) 
s_ashr_i32 s18, s0, 31 // (op 2626) 

s_xor_b32 s0, s1, s18 // (op 2626) 

s_sub_i32 s1, s0, s18 // (op 2626) 


s_mul_i32 s0, s11, s22 // (op 2626) 

s_mul_i32 s18, s8, s0 // (op 2626) 

s_mul_i32 s0, s1, s18 // (op 2626) 

s_sub_u32 s18, s16, s0 // (op 2626) 

s_mul_hi_i32 s0, s18, s12 // (op 2626) 
s_add_i32 s19, s0, s18 // (op 2626) 


s_ashr_i32 s0, s19, 31 // (op 2626) 
s_cmp_eq_i32 s12, 0 // (op 2626) 
s_mov_b32 s18, scc // (op 2626) 
s_cmp_eq_i32 s18, 0 // (op 2626) 
s_cselect_b32 s22, 0, -1 // (op 2626) 

s_and_b32 s18, s44, 31 // (op 2626) 
s_lshl_b32 s23, 1, s18 // (op 2626) 
s_add_i32 s24, s23, s22 // (op 2626) 


s_and_b32 s25, s0, s24 // (op 2626) 


s_add_i32 s0, s19, s25 // (op 2626) 


s_ashr_i32 s19, s0, s18 // (op 2626) 


s_lshl_b32 s0, s44, 24 // (op 2626) 
s_ashr_i32 s18, s0, 31 // (op 2626) 

s_xor_b32 s0, s19, s18 // (op 2626) 

s_sub_i32 s19, s0, s18 // (op 2626) 


s_cmp_eq_i32 s1, 0 // (op 2626) 
s_mov_b32 s0, scc // (op 2626) 

s_cmp_eq_i32 s0, 0 // (op 2626) 
s_cselect_b32 s1, s19, s47 // (op 2626) 



v_lshl_add_u32 v20, s1, 8, v1 // (op 2626) 

v_mul_lo_u32 v22, v20, s38 // (op 2626) 

s_mul_i32 s1, s38, s7 // (op 2628) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 2628) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 2628) least significant
s_mul_hi_u32 s18, s38, s6 // (op 2628) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2628) most significant: sum
s_add_u32 s1, s1, s18 // (op 2628) most significant: sum


s_mov_b32 s68, s40 // (op 2628) reg expression
s_mov_b32 s69, s41 // (op 2628) reg expression
s_mov_b32 s70, s0 // (op 2628) reg expression
s_mov_b32 s71, 131072 // (op 2628) call()


s_mul_i32 s1, s31, 6 // (op 2153) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 2153) least significant
s_mul_hi_u32 s47, s30, 6 // (op 2153) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2153) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2153) 


v_mov_b32 v20, v18 // (op 2153) 
buffer_load_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 2153) Load value
buffer_load_dwordx2 v[30:31], v20, s[60:63], 0 offen offset:16 // (op 2153) Load value
buffer_load_dwordx4 v[32:35], v20, s[60:63], 0 offen offset:24 // (op 2153) Load value
buffer_load_dwordx2 v[36:37], v20, s[60:63], 0 offen offset:40 // (op 2153) Load value
v_add_u32 v20, v20, v24 // (op 2153) 
buffer_load_dwordx4 v[38:41], v20, s[60:63], 0 offen // (op 2153) Load value
buffer_load_dwordx2 v[42:43], v20, s[60:63], 0 offen offset:16 // (op 2153) Load value
buffer_load_dwordx4 v[44:47], v20, s[60:63], 0 offen offset:24 // (op 2153) Load value
buffer_load_dwordx2 v[48:49], v20, s[60:63], 0 offen offset:40 // (op 2153) Load value





buffer_load_dword v20, v21, s[68:71], 0 offen // (op 2156) Load value



buffer_load_dwordx4 v[50:53], v14, s[48:51], 0 offen // (op 2159) Load value
buffer_load_dwordx2 v[54:55], v14, s[48:51], 0 offen offset:16 // (op 2159) Load value



buffer_load_ubyte v23, v16, s[64:67], 0 offen offset: 0 // (op 2162) Load value

s_mul_i32 s1, s31, 6 // (op 2165) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 2165) least significant
s_mul_hi_u32 s47, s30, 6 // (op 2165) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2165) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2165) 


v_mov_b32 v56, v18 // (op 2165) 
buffer_load_dwordx4 v[58:61], v56, s[60:63], 0 offen offset:96 // (op 2165) Load value
buffer_load_dwordx2 v[62:63], v56, s[60:63], 0 offen offset:112 // (op 2165) Load value
buffer_load_dwordx4 v[64:67], v56, s[60:63], 0 offen offset:120 // (op 2165) Load value
buffer_load_dwordx2 v[68:69], v56, s[60:63], 0 offen offset:136 // (op 2165) Load value
v_add_u32 v56, v56, v24 // (op 2165) 
buffer_load_dwordx4 v[70:73], v56, s[60:63], 0 offen offset:96 // (op 2165) Load value
buffer_load_dwordx2 v[74:75], v56, s[60:63], 0 offen offset:112 // (op 2165) Load value
buffer_load_dwordx4 v[76:79], v56, s[60:63], 0 offen offset:120 // (op 2165) Load value
buffer_load_dwordx2 v[80:81], v56, s[60:63], 0 offen offset:136 // (op 2165) Load value






buffer_load_dword v18, v21, s[68:71], 0 offen offset:4 // (op 2168) Load value




buffer_load_dwordx4 v[82:85], v14, s[48:51], 0 offen offset:96 // (op 2171) Load value
buffer_load_dwordx2 v[86:87], v14, s[48:51], 0 offen offset:112 // (op 2171) Load value




buffer_load_ubyte v14, v16, s[64:67], 0 offen offset:4 // (op 2174) Load value

v_and_b32 v16, -2, v1 // (op 2746) 
v_sub_u32 v21, v1, v16 // (op 2746) 

v_lshlrev_b32 v16, 6, v21 // (op 2746) 

v_lshrrev_b32 v21, 1, v1 // (op 2746) 
v_lshl_add_u32 v24, v21, 8, v16 // (op 2746) 


v_mul_lo_u32 v16, v24, 6 // (op 2746) 

v_lshrrev_b32 v21, 3, v16 // (op 2746) 




s_waitcnt vmcnt(23)
ds_write_b128 v21, v[26:29] // (op 2176) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v21, v[30:31] offset:16 // (op 2176) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v21, v[32:35] offset:24 // (op 2176) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v21, v[36:37] offset:40 // (op 2176) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v21, v[38:41] offset:96 // (op 2176) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v21, v[42:43] offset:112 // (op 2176) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v21, v[44:47] offset:120 // (op 2176) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v21, v[48:49] offset:136 // (op 2176) Store local data 

v_lshlrev_b32 v16, 2, v1 // (op 2857) 



s_waitcnt vmcnt(15)
ds_write_b32 v16, v20 offset:24576 // (op 2177) Store local data 

v_and_b32 v16, -4, v1 // (op 2693) 
v_sub_u32 v21, v1, v16 // (op 2693) 

v_lshlrev_b32 v16, 5, v21 // (op 2693) 

v_lshrrev_b32 v21, 2, v1 // (op 2693) 
v_lshl_add_u32 v24, v21, 7, v16 // (op 2693) 


v_mul_lo_u32 v16, v24, 6 // (op 2693) 

v_lshrrev_b32 v21, 3, v16 // (op 2693) 




s_waitcnt vmcnt(14)
ds_write_b128 v21, v[50:53] offset:25600 // (op 2178) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v21, v[54:55] offset:25616 // (op 2178) Store local data 

v_and_b32 v16, -4, v1 // (op 2804) 
v_sub_u32 v21, v1, v16 // (op 2804) 

v_and_b32 v16, 4294967292, v1 // (op 2804) 
v_add_u32 v24, v16, v21 // (op 2804) 





s_waitcnt vmcnt(12)
ds_write_b8 v24, v23 offset:31744 // (op 2179) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 3080) 



ds_read_b128 v[88:91], v9 // (op 2197) Load local data 
ds_read_b64 v[92:93], v9 offset:16 // (op 2197) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 3121) 



ds_read_u8 v16, v6 offset:24576 // (op 2220) Load local data 



ds_read_b128 v[94:97], v9 offset:6144 // (op 2202) Load local data 
ds_read_b64 v[98:99], v9 offset:6160 // (op 2202) Load local data 



ds_read_u8 v21, v6 offset:24832 // (op 2225) Load local data 



ds_read_b128 v[100:103], v9 offset:12288 // (op 2207) Load local data 
ds_read_b64 v[104:105], v9 offset:12304 // (op 2207) Load local data 



ds_read_u8 v24, v6 offset:25088 // (op 2230) Load local data 



ds_read_b128 v[106:109], v9 offset:18432 // (op 2212) Load local data 
ds_read_b64 v[110:111], v9 offset:18448 // (op 2212) Load local data 



ds_read_u8 v25, v6 offset:25344 // (op 2235) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 3061) 



ds_read_b128 v[112:115], v3 offset:25600 // (op 2193) Load local data 
ds_read_b64 v[116:117], v3 offset:25616 // (op 2193) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3101) 



ds_read_u8 v56, v0 offset:31744 // (op 2216) Load local data 
v_and_b32 v57, -4, v1 // (op 2664) 
v_sub_u32 v118, v1, v57 // (op 2664) 

v_lshlrev_b32 v57, 5, v118 // (op 2664) 

v_lshrrev_b32 v118, 2, v1 // (op 2664) 
v_lshl_add_u32 v119, v118, 7, v57 // (op 2664) 


v_mul_lo_u32 v57, v119, 6 // (op 2664) 

v_lshrrev_b32 v118, 3, v57 // (op 2664) 

v_and_b32 v57, -2, v1 // (op 2717) 
v_sub_u32 v119, v1, v57 // (op 2717) 

v_lshlrev_b32 v57, 6, v119 // (op 2717) 

v_lshrrev_b32 v119, 1, v1 // (op 2717) 
v_lshl_add_u32 v120, v119, 8, v57 // (op 2717) 


v_mul_lo_u32 v57, v120, 6 // (op 2717) 

v_lshrrev_b32 v119, 3, v57 // (op 2717) 

v_and_b32 v57, -4, v1 // (op 2775) 
v_sub_u32 v120, v1, v57 // (op 2775) 

v_and_b32 v57, 4294967292, v1 // (op 2775) 
v_add_u32 v121, v57, v120 // (op 2775) 


v_lshlrev_b32 v57, 2, v1 // (op 2828) 
v_and_b32 v120, -4, v1 // (op 2904) 
v_sub_u32 v122, v1, v120 // (op 2904) 

v_lshlrev_b32 v120, 5, v122 // (op 2904) 

v_lshrrev_b32 v122, 2, v1 // (op 2904) 
v_lshl_add_u32 v123, v122, 7, v120 // (op 2904) 


v_mul_lo_u32 v120, v123, 6 // (op 2904) 

v_lshrrev_b32 v122, 3, v120 // (op 2904) 

v_and_b32 v120, -2, v1 // (op 2928) 
v_sub_u32 v123, v1, v120 // (op 2928) 

v_lshlrev_b32 v120, 6, v123 // (op 2928) 

v_lshrrev_b32 v123, 1, v1 // (op 2928) 
v_lshl_add_u32 v124, v123, 8, v120 // (op 2928) 


v_mul_lo_u32 v120, v124, 6 // (op 2928) 

v_lshrrev_b32 v123, 3, v120 // (op 2928) 

v_and_b32 v120, -4, v1 // (op 2957) 
v_sub_u32 v124, v1, v120 // (op 2957) 

v_and_b32 v120, 4294967292, v1 // (op 2957) 
v_add_u32 v125, v120, v124 // (op 2957) 


v_lshlrev_b32 v120, 2, v1 // (op 2981) 
s_mov_b32 s47, 0 // (op 868) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 867) 
s_mov_b32 s18, s0 // (op 867) convert

s_and_b32 s19, s18, 4294967294 // (op 867) 

s_cmp_lt_u32 s47, s19 // (op 867) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_867 // (op 867) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_867 if false)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_867:

s_add_u32 s47, s47, 2 // (op 869) 
s_mul_i32 s1, s31, 6 // (op 10) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 10) least significant
s_mul_hi_u32 s18, s30, 6 // (op 10) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10) most significant: sum

v_lshrrev_b64 v[126:127], 3, s[0:1] // (op 10) 


v_mov_b32 v124, v19 // (op 10) 
v_mov_b32 v128, v124 // (op 10) 
buffer_load_dwordx4 v[26:29], v128, s[60:63], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[30:31], v128, s[60:63], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[32:35], v128, s[60:63], 0 offen offset:216 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[112:117], v[88:93], a[0:15], v56, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 88) 
buffer_load_dwordx2 v[36:37], v128, s[60:63], 0 offen offset:232 // (op 10) Load value
s_barrier  // (op 3070) 
v_add_u32 v128, v128, v126 // (op 10) 



buffer_load_dwordx4 v[38:41], v128, s[60:63], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[42:43], v128, s[60:63], 0 offen offset:208 // (op 10) Load value
ds_read_b128 v[130:133], v3 offset:25648 // (op 91) Load local data 
ds_read_b64 v[134:135], v3 offset:25664 // (op 91) Load local data 
buffer_load_dwordx4 v[44:47], v128, s[60:63], 0 offen offset:216 // (op 10) Load value
s_barrier  // (op 3090) 
buffer_load_dwordx2 v[48:49], v128, s[60:63], 0 offen offset:232 // (op 10) Load value







ds_read_b128 v[88:91], v9 offset:48 // (op 95) Load local data 
ds_read_b64 v[92:93], v9 offset:64 // (op 95) Load local data 
v_mov_b32 v126, v22 // (op 25) 

s_barrier  // (op 3111) 
buffer_load_dword v20, v126, s[68:71], 0 offen offset:8 // (op 25) Load value





ds_read_u8 v127, v0 offset:31746 // (op 99) Load local data 
v_mov_b32 v128, v15 // (op 4) 

s_barrier  // (op 3131) 
buffer_load_dwordx4 v[50:53], v128, s[48:51], 0 offen offset:192 // (op 4) Load value



ds_read_u8 v16, v6 offset:24578 // (op 103) Load local data 
buffer_load_dwordx2 v[54:55], v128, s[48:51], 0 offen offset:208 // (op 4) Load value


v_mov_b32 v129, v17 // (op 17) 

buffer_load_ubyte v23, v129, s[64:67], 0 offen offset:8 // (op 17) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[130:135], v[88:93], a[0:15], v127, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 105) 
s_barrier  // (op 3151) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[112:117], v[94:99], a[16:31], v56, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 447) 
s_waitcnt vmcnt(23)
ds_write_b128 v123, v[58:61] offset:32000 // (op 302) Store local data 



ds_read_b128 v[88:91], v9 offset:6192 // (op 451) Load local data 
ds_read_b64 v[92:93], v9 offset:6208 // (op 451) Load local data 
s_waitcnt vmcnt(22)
ds_write_b64 v123, v[62:63] offset:32016 // (op 302) Store local data 



ds_read_u8 v16, v6 offset:24834 // (op 455) Load local data 
s_waitcnt vmcnt(21)
ds_write_b128 v123, v[64:67] offset:32024 // (op 302) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[130:135], v[88:93], a[16:31], v127, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 456) 
s_waitcnt vmcnt(20)
ds_write_b64 v123, v[68:69] offset:32040 // (op 302) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v123, v[70:73] offset:32096 // (op 302) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[112:117], v[100:105], a[32:47], v56, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 681) 
s_waitcnt vmcnt(18)
ds_write_b64 v123, v[74:75] offset:32112 // (op 302) Store local data 



ds_read_b128 v[88:91], v9 offset:12336 // (op 685) Load local data 
ds_read_b64 v[92:93], v9 offset:12352 // (op 685) Load local data 
s_waitcnt vmcnt(17)
ds_write_b128 v123, v[76:79] offset:32120 // (op 302) Store local data 



ds_read_u8 v16, v6 offset:25090 // (op 689) Load local data 
s_waitcnt vmcnt(16)
ds_write_b64 v123, v[80:81] offset:32136 // (op 302) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[130:135], v[88:93], a[32:47], v127, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 690) 
s_barrier  // (op 3163) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[112:117], v[106:111], a[48:63], v56, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 915) 
s_waitcnt vmcnt(15)
ds_write_b32 v120, v18 offset:56576 // (op 304) Store local data 



ds_read_b128 v[106:109], v9 offset:18480 // (op 919) Load local data 
s_barrier  // (op 3144) 
ds_read_b64 v[110:111], v9 offset:18496 // (op 919) Load local data 






ds_read_u8 v25, v6 offset:25346 // (op 923) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v122, v[82:85] offset:57600 // (op 283) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[130:135], v[106:111], a[48:63], v127, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 924) 
s_waitcnt vmcnt(13)
ds_write_b64 v122, v[86:87] offset:57616 // (op 283) Store local data 


s_barrier  // (op 3157) 



s_waitcnt vmcnt(12)
ds_write_b8 v125, v14 offset:63744 // (op 303) Store local data 
s_barrier  // (op 3147) 



ds_read_b128 v[88:91], v13 offset:32000 // (op 287) Load local data 
ds_read_b64 v[92:93], v13 offset:32016 // (op 287) Load local data 
s_barrier  // (op 3160) 



ds_read_u8 v16, v12 offset:56576 // (op 291) Load local data 



ds_read_b128 v[94:97], v13 offset:38144 // (op 470) Load local data 
ds_read_b64 v[98:99], v13 offset:38160 // (op 470) Load local data 



ds_read_u8 v21, v12 offset:56832 // (op 474) Load local data 



ds_read_b128 v[100:103], v13 offset:44288 // (op 704) Load local data 
ds_read_b64 v[104:105], v13 offset:44304 // (op 704) Load local data 



ds_read_u8 v24, v12 offset:57088 // (op 708) Load local data 



ds_read_b128 v[130:133], v13 offset:50432 // (op 938) Load local data 
ds_read_b64 v[134:135], v13 offset:50448 // (op 938) Load local data 



ds_read_u8 v127, v12 offset:57344 // (op 942) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 3141) 



ds_read_b128 v[112:115], v5 offset:57600 // (op 285) Load local data 
ds_read_b64 v[116:117], v5 offset:57616 // (op 285) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3154) 



ds_read_u8 v56, v4 offset:63744 // (op 289) Load local data 

s_mul_i32 s1, s31, 6 // (op 305) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 305) least significant
s_mul_hi_u32 s18, s30, 6 // (op 305) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 305) most significant: sum

v_lshrrev_b64 v[136:137], 3, s[0:1] // (op 305) 


v_mov_b32 v138, v124 // (op 305) 
buffer_load_dwordx4 v[58:61], v138, s[60:63], 0 offen offset:288 // (op 305) Load value
buffer_load_dwordx2 v[62:63], v138, s[60:63], 0 offen offset:304 // (op 305) Load value
buffer_load_dwordx4 v[64:67], v138, s[60:63], 0 offen offset:312 // (op 305) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[112:117], v[88:93], a[0:15], v56, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 292) 
buffer_load_dwordx2 v[68:69], v138, s[60:63], 0 offen offset:328 // (op 305) Load value
v_add_u32 v138, v138, v136 // (op 305) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[112:117], v[94:99], a[16:31], v56, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 457) 
buffer_load_dwordx4 v[70:73], v138, s[60:63], 0 offen offset:288 // (op 305) Load value
buffer_load_dwordx2 v[74:75], v138, s[60:63], 0 offen offset:304 // (op 305) Load value
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[112:117], v[100:105], a[32:47], v56, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 691) 
ds_read_b128 v[140:143], v5 offset:57648 // (op 294) Load local data 
ds_read_b64 v[144:145], v5 offset:57664 // (op 294) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[112:117], v[130:135], a[48:63], v56, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 925) 





s_barrier  // (op 3208) 
ds_read_b128 v[130:133], v13 offset:32048 // (op 296) Load local data 
ds_read_b64 v[134:135], v13 offset:32064 // (op 296) Load local data 
buffer_load_dwordx4 v[76:79], v138, s[60:63], 0 offen offset:312 // (op 305) Load value



ds_read_u8 v127, v4 offset:63746 // (op 298) Load local data 
buffer_load_dwordx2 v[80:81], v138, s[60:63], 0 offen offset:328 // (op 305) Load value









s_barrier  // (op 3212) 
buffer_load_dword v18, v126, s[68:71], 0 offen offset:12 // (op 307) Load value
v_add_u32 v19, 192, v19 // (op 2490) 

ds_read_u8 v124, v12 offset:56578 // (op 300) Load local data 






v_add_u32 v22, 8, v22 // (op 2618) 
ds_read_b128 v[146:149], v13 offset:38192 // (op 461) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[140:145], v[130:135], a[0:15], v127, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 301) 
ds_read_b64 v[150:151], v13 offset:38208 // (op 461) Load local data 





ds_read_u8 v124, v12 offset:56834 // (op 465) Load local data 
buffer_load_dwordx4 v[82:85], v128, s[48:51], 0 offen offset:288 // (op 282) Load value



ds_read_b128 v[130:133], v13 offset:44336 // (op 695) Load local data 
ds_read_b64 v[134:135], v13 offset:44352 // (op 695) Load local data 
buffer_load_dwordx2 v[86:87], v128, s[48:51], 0 offen offset:304 // (op 282) Load value




ds_read_u8 v126, v12 offset:57090 // (op 699) Load local data 






buffer_load_ubyte v14, v129, s[64:67], 0 offen offset:12 // (op 306) Load value
v_add_u32 v15, 192, v15 // (op 2426) 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[140:145], v[146:151], a[16:31], v127, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 466) 


s_barrier  // (op 3083) 



v_add_u32 v17, 8, v17 // (op 2554) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[140:145], v[130:135], a[32:47], v127, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 700) 
ds_read_b128 v[146:149], v13 offset:50480 // (op 929) Load local data 


ds_read_b64 v[150:151], v13 offset:50496 // (op 929) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v119, v[26:29] // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v119, v[30:31] offset:16 // (op 13) Store local data 



s_barrier  // (op 3268) 
ds_read_u8 v124, v12 offset:57346 // (op 933) Load local data 
s_waitcnt vmcnt(21)
ds_write_b128 v119, v[32:35] offset:24 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v119, v[36:37] offset:40 // (op 13) Store local data 
s_barrier  // (op 3272) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[140:145], v[146:151], a[48:63], v127, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 934) 
s_waitcnt vmcnt(19)
ds_write_b128 v119, v[38:41] offset:96 // (op 13) Store local data 




s_waitcnt vmcnt(18)
ds_write_b64 v119, v[42:43] offset:112 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v119, v[44:47] offset:120 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v119, v[48:49] offset:136 // (op 13) Store local data 
s_barrier  // (op 3124) 



s_waitcnt vmcnt(15)
ds_write_b32 v57, v20 offset:24576 // (op 30) Store local data 
s_barrier  // (op 3064) 



s_waitcnt vmcnt(14)
ds_write_b128 v118, v[50:53] offset:25600 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v118, v[54:55] offset:25616 // (op 6) Store local data 
s_barrier  // (op 3104) 



s_waitcnt vmcnt(12)
ds_write_b8 v121, v23 offset:31744 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3093) 



ds_read_b128 v[88:91], v9 // (op 75) Load local data 
ds_read_b64 v[92:93], v9 offset:16 // (op 75) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 3134) 



ds_read_u8 v16, v6 offset:24576 // (op 85) Load local data 



ds_read_b128 v[94:97], v9 offset:6144 // (op 442) Load local data 
ds_read_b64 v[98:99], v9 offset:6160 // (op 442) Load local data 



ds_read_u8 v21, v6 offset:24832 // (op 446) Load local data 



ds_read_b128 v[100:103], v9 offset:12288 // (op 676) Load local data 
ds_read_b64 v[104:105], v9 offset:12304 // (op 676) Load local data 



ds_read_u8 v24, v6 offset:25088 // (op 680) Load local data 



ds_read_b128 v[106:109], v9 offset:18432 // (op 910) Load local data 
ds_read_b64 v[110:111], v9 offset:18448 // (op 910) Load local data 



ds_read_u8 v25, v6 offset:25344 // (op 914) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 3073) 



ds_read_b128 v[112:115], v3 offset:25600 // (op 70) Load local data 
ds_read_b64 v[116:117], v3 offset:25616 // (op 70) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3114) 



ds_read_u8 v56, v0 offset:31744 // (op 80) Load local data 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 867) 
s_mov_b32 s18, s0 // (op 867) convert

s_and_b32 s19, s18, 4294967294 // (op 867) 

s_cmp_lt_u32 s47, s19 // (op 867) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_867 // (op 867) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_867 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_867:

















s_mul_hi_i32 s47, s16, s10 // (op 2386) 
v_lshrrev_b32 v15, 2, v1 // (op 2386) 
s_add_i32 s0, s47, s16 // (op 2386) 

v_and_b32 v17, -4, v1 // (op 2386) 
s_ashr_i32 s1, s0, 31 // (op 2386) 
v_sub_u32 v57, v1, v17 // (op 2386) 

s_cmp_eq_i32 s10, 0 // (op 2386) 
s_mov_b32 s47, scc // (op 2386) 
v_lshlrev_b32 v17, 5, v57 // (op 2386) 

s_cmp_eq_i32 s47, 0 // (op 2386) 
s_cselect_b32 s18, 0, -1 // (op 2386) 

s_and_b32 s19, s13, 31 // (op 2386) 
s_lshl_b32 s47, 1, s19 // (op 2386) 
s_add_i32 s22, s47, s18 // (op 2386) 


s_and_b32 s18, s1, s22 // (op 2386) 


s_add_i32 s1, s0, s18 // (op 2386) 


s_ashr_i32 s0, s1, s19 // (op 2386) 


s_lshl_b32 s1, s13, 24 // (op 2386) 
s_ashr_i32 s47, s1, 31 // (op 2386) 

s_xor_b32 s1, s0, s47 // (op 2386) 

s_sub_i32 s0, s1, s47 // (op 2386) 


s_mul_hi_i32 s1, s0, s14 // (op 2386) 
s_add_i32 s47, s1, s0 // (op 2386) 

s_ashr_i32 s1, s47, 31 // (op 2386) 
s_cmp_eq_i32 s14, 0 // (op 2386) 
s_mov_b32 s18, scc // (op 2386) 
s_cmp_eq_i32 s18, 0 // (op 2386) 
s_cselect_b32 s19, 0, -1 // (op 2386) 

s_and_b32 s18, s15, 31 // (op 2386) 
s_lshl_b32 s22, 1, s18 // (op 2386) 
s_add_i32 s23, s22, s19 // (op 2386) 


s_and_b32 s19, s1, s23 // (op 2386) 


s_add_i32 s1, s47, s19 // (op 2386) 


s_ashr_i32 s19, s1, s18 // (op 2386) 


s_lshl_b32 s1, s15, 24 // (op 2386) 
s_ashr_i32 s18, s1, 31 // (op 2386) 

s_xor_b32 s1, s19, s18 // (op 2386) 

s_sub_i32 s19, s1, s18 // (op 2386) 


s_mul_i32 s1, s0, s11 // (op 2386) 

s_sub_u32 s0, s16, s1 // (op 2386) 

s_mul_i32 s1, s19, s11 // (op 2386) 
s_add_u32 s1, s1, s0 // (op 2386) 


s_mul_hi_i32 s0, s16, s9 // (op 2386) 
s_add_i32 s47, s0, s16 // (op 2386) 

s_ashr_i32 s0, s47, 31 // (op 2386) 
s_cmp_eq_i32 s9, 0 // (op 2386) 
s_mov_b32 s18, scc // (op 2386) 
s_cmp_eq_i32 s18, 0 // (op 2386) 
s_cselect_b32 s19, 0, -1 // (op 2386) 

s_and_b32 s18, s45, 31 // (op 2386) 
s_lshl_b32 s22, 1, s18 // (op 2386) 
s_add_i32 s23, s22, s19 // (op 2386) 


s_and_b32 s19, s0, s23 // (op 2386) 


s_add_i32 s0, s47, s19 // (op 2386) 


s_ashr_i32 s19, s0, s18 // (op 2386) 


s_lshl_b32 s0, s45, 24 // (op 2386) 
s_ashr_i32 s18, s0, 31 // (op 2386) 

s_xor_b32 s0, s19, s18 // (op 2386) 

s_sub_i32 s19, s0, s18 // (op 2386) 


s_add_u32 s22, s6, 256 // (op 2386) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2386) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2386) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2386) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2386) 

s_mul_i32 s0, s11, s22 // (op 2386) 

s_mul_i32 s18, s8, s0 // (op 2386) 

s_mul_i32 s0, s19, s18 // (op 2386) 

s_sub_u32 s18, s16, s0 // (op 2386) 

s_mov_b32 s0, s18 // (op 2386) 
s_mul_hi_i32 s47, s0, s12 // (op 2386) 
s_add_i32 s22, s47, s0 // (op 2386) 


s_ashr_i32 s0, s22, 31 // (op 2386) 
s_cmp_eq_i32 s12, 0 // (op 2386) 
s_mov_b32 s47, scc // (op 2386) 
s_cmp_eq_i32 s47, 0 // (op 2386) 
s_cselect_b32 s23, 0, -1 // (op 2386) 

s_and_b32 s47, s44, 31 // (op 2386) 
s_lshl_b32 s24, 1, s47 // (op 2386) 
s_add_i32 s25, s24, s23 // (op 2386) 


s_and_b32 s23, s0, s25 // (op 2386) 


s_add_i32 s0, s22, s23 // (op 2386) 


s_ashr_i32 s22, s0, s47 // (op 2386) 


s_lshl_b32 s0, s44, 24 // (op 2386) 
s_ashr_i32 s47, s0, 31 // (op 2386) 

s_xor_b32 s0, s22, s47 // (op 2386) 

s_sub_i32 s22, s0, s47 // (op 2386) 


s_mul_i32 s0, s46, s11 // (op 2386) 
s_add_u32 s24, s20, 64 // (op 2386) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2386) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 2386) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 2386) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 2386) 

s_sub_i32 s23, s24, s0 // (op 2386) 

s_mul_i32 s47, s22, s23 // (op 2386) 


s_sub_u32 s22, s18, s47 // (op 2386) 


s_add_u32 s18, s22, s0 // (op 2386) 


s_cmp_eq_i32 s19, 0 // (op 2386) 
s_mov_b32 s0, scc // (op 2386) 

s_cmp_eq_i32 s0, 0 // (op 2386) 
s_cselect_b32 s19, s18, s1 // (op 2386) 



v_lshl_add_u32 v57, s19, 6, v15 // (op 2386) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2386) 
s_lshr_b32 s47, s0, 1 // (op 2386) 

v_lshl_add_u32 v15, s47, 8, v17 // (op 2386) 


v_mov_b32 v23, 0 // (op 2386) convert
v_mov_b32 v22, v15 // (op 2386) convert

v_mul_lo_u32 v27, v57, s27 // (op 2386) most significant: low of low * high
v_mul_hi_u32 v15, v57, s26 // (op 2386) most significant: high of low * low
v_mul_lo_u32 v26, v57, s26 // (op 2386) least significant: low of low * low
v_add_u32 v27, v27, v15 // (op 2386) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v22 // (op 2386) least significant half
v_addc_co_u32 v27, s[0:1], v27, v23, s[0:1] // (op 2386) most significant half



v_mul_lo_u32 v23, v27, 6 // (op 2386) most significant: low of high * low
v_mul_hi_u32 v15, v26, 6 // (op 2386) most significant: high of low * low
v_mul_lo_u32 v22, v26, 6 // (op 2386) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 2386) most significant: sum


v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 2386) 

v_mov_b32 v15, v26 // (op 2386) convert

s_mul_i32 s1, s21, 6 // (op 2388) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 2388) least significant
s_mul_hi_u32 s47, s20, 6 // (op 2388) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2388) most significant: sum

s_mul_i32 s19, s26, s1 // (op 2388) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 2388) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 2388) least significant
s_mul_hi_u32 s22, s26, s0 // (op 2388) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 2388) most significant: sum
s_add_u32 s19, s19, s22 // (op 2388) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2388) 

s_mov_b32 s48, s28 // (op 2388) reg expression
s_mov_b32 s49, s29 // (op 2388) reg expression
s_mov_b32 s50, s0 // (op 2388) reg expression
s_mov_b32 s51, 131072 // (op 2388) call()

s_mul_hi_i32 s47, s16, s10 // (op 2456) 
v_and_b32 v17, 4294967294, v1 // (op 2456) 
s_add_i32 s0, s47, s16 // (op 2456) 

v_and_b32 v57, -2, v1 // (op 2456) 
s_ashr_i32 s1, s0, 31 // (op 2456) 
v_sub_u32 v19, v1, v57 // (op 2456) 

s_cmp_eq_i32 s10, 0 // (op 2456) 
s_mov_b32 s47, scc // (op 2456) 
v_lshlrev_b32 v20, 6, v19 // (op 2456) 

s_cmp_eq_i32 s47, 0 // (op 2456) 
s_cselect_b32 s18, 0, -1 // (op 2456) 

s_and_b32 s19, s13, 31 // (op 2456) 
s_lshl_b32 s47, 1, s19 // (op 2456) 
s_add_i32 s22, s47, s18 // (op 2456) 


s_and_b32 s18, s1, s22 // (op 2456) 


s_add_i32 s1, s0, s18 // (op 2456) 


s_ashr_i32 s0, s1, s19 // (op 2456) 


s_lshl_b32 s1, s13, 24 // (op 2456) 
s_ashr_i32 s47, s1, 31 // (op 2456) 

s_xor_b32 s1, s0, s47 // (op 2456) 

s_sub_i32 s0, s1, s47 // (op 2456) 


s_mul_hi_i32 s1, s0, s14 // (op 2456) 
s_add_i32 s47, s1, s0 // (op 2456) 

s_ashr_i32 s1, s47, 31 // (op 2456) 
s_cmp_eq_i32 s14, 0 // (op 2456) 
s_mov_b32 s18, scc // (op 2456) 
s_cmp_eq_i32 s18, 0 // (op 2456) 
s_cselect_b32 s19, 0, -1 // (op 2456) 

s_and_b32 s18, s15, 31 // (op 2456) 
s_lshl_b32 s22, 1, s18 // (op 2456) 
s_add_i32 s23, s22, s19 // (op 2456) 


s_and_b32 s19, s1, s23 // (op 2456) 


s_add_i32 s1, s47, s19 // (op 2456) 


s_ashr_i32 s19, s1, s18 // (op 2456) 


s_lshl_b32 s1, s15, 24 // (op 2456) 
s_ashr_i32 s18, s1, 31 // (op 2456) 

s_xor_b32 s1, s19, s18 // (op 2456) 

s_sub_i32 s19, s1, s18 // (op 2456) 


s_add_u32 s22, s6, 256 // (op 2456) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2456) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2456) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2456) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2456) 

s_mul_i32 s1, s19, s22 // (op 2456) 

s_sub_i32 s47, s0, s1 // (op 2456) 


s_mul_hi_i32 s0, s16, s9 // (op 2456) 
s_add_i32 s1, s0, s16 // (op 2456) 

s_ashr_i32 s0, s1, 31 // (op 2456) 
s_cmp_eq_i32 s9, 0 // (op 2456) 
s_mov_b32 s18, scc // (op 2456) 
s_cmp_eq_i32 s18, 0 // (op 2456) 
s_cselect_b32 s19, 0, -1 // (op 2456) 

s_and_b32 s18, s45, 31 // (op 2456) 
s_lshl_b32 s24, 1, s18 // (op 2456) 
s_add_i32 s25, s24, s19 // (op 2456) 


s_and_b32 s19, s0, s25 // (op 2456) 


s_add_i32 s0, s1, s19 // (op 2456) 


s_ashr_i32 s1, s0, s18 // (op 2456) 


s_lshl_b32 s0, s45, 24 // (op 2456) 
s_ashr_i32 s18, s0, 31 // (op 2456) 

s_xor_b32 s0, s1, s18 // (op 2456) 

s_sub_i32 s1, s0, s18 // (op 2456) 


s_mul_i32 s0, s11, s22 // (op 2456) 

s_mul_i32 s18, s8, s0 // (op 2456) 

s_mul_i32 s0, s1, s18 // (op 2456) 

s_sub_u32 s18, s16, s0 // (op 2456) 

s_mul_hi_i32 s0, s18, s12 // (op 2456) 
s_add_i32 s19, s0, s18 // (op 2456) 


s_ashr_i32 s0, s19, 31 // (op 2456) 
s_cmp_eq_i32 s12, 0 // (op 2456) 
s_mov_b32 s18, scc // (op 2456) 
s_cmp_eq_i32 s18, 0 // (op 2456) 
s_cselect_b32 s22, 0, -1 // (op 2456) 

s_and_b32 s18, s44, 31 // (op 2456) 
s_lshl_b32 s23, 1, s18 // (op 2456) 
s_add_i32 s24, s23, s22 // (op 2456) 


s_and_b32 s25, s0, s24 // (op 2456) 


s_add_i32 s0, s19, s25 // (op 2456) 


s_ashr_i32 s19, s0, s18 // (op 2456) 


s_lshl_b32 s0, s44, 24 // (op 2456) 
s_ashr_i32 s18, s0, 31 // (op 2456) 

s_xor_b32 s0, s19, s18 // (op 2456) 

s_sub_i32 s19, s0, s18 // (op 2456) 


s_cmp_eq_i32 s1, 0 // (op 2456) 
s_mov_b32 s0, scc // (op 2456) 

s_cmp_eq_i32 s0, 0 // (op 2456) 
s_cselect_b32 s1, s19, s47 // (op 2456) 



v_lshl_add_u32 v19, s1, 8, v17 // (op 2456) 


v_mul_lo_u32 v23, v19, s31 // (op 2456) most significant: low of low * high
v_mul_hi_u32 v17, v19, s30 // (op 2456) most significant: high of low * low
v_mul_lo_u32 v22, v19, s30 // (op 2456) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 2456) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2456) 
s_lshr_b32 s47, s0, 1 // (op 2456) 

v_lshl_add_u32 v17, s47, 8, v20 // (op 2456) 


v_mov_b32 v27, 0 // (op 2456) convert
v_mov_b32 v26, v17 // (op 2456) convert

v_add_co_u32 v28, s[0:1], v26, v22 // (op 2456) least significant half
v_addc_co_u32 v29, s[0:1], v27, v23, s[0:1] // (op 2456) most significant half



v_mul_lo_u32 v23, v29, 6 // (op 2456) most significant: low of high * low
v_mul_hi_u32 v17, v28, 6 // (op 2456) most significant: high of low * low
v_mul_lo_u32 v22, v28, 6 // (op 2456) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 2456) most significant: sum


v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 2456) 

v_mov_b32 v17, v26 // (op 2456) convert

s_mul_i32 s1, s7, 6 // (op 2458) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 2458) least significant
s_mul_hi_u32 s47, s6, 6 // (op 2458) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2458) most significant: sum

s_mul_i32 s19, s30, s1 // (op 2458) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 2458) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 2458) least significant
s_mul_hi_u32 s22, s30, s0 // (op 2458) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 2458) most significant: sum
s_add_u32 s19, s19, s22 // (op 2458) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2458) 

s_mov_b32 s60, s32 // (op 2458) reg expression
s_mov_b32 s61, s33 // (op 2458) reg expression
s_mov_b32 s62, s0 // (op 2458) reg expression
s_mov_b32 s63, 131072 // (op 2458) call()

s_mul_hi_i32 s47, s16, s10 // (op 2520) 
v_lshrrev_b32 v57, 2, v1 // (op 2520) 
s_add_i32 s0, s47, s16 // (op 2520) 

v_and_b32 v19, -4, v1 // (op 2520) 
s_ashr_i32 s1, s0, 31 // (op 2520) 
v_sub_u32 v20, v1, v19 // (op 2520) 

s_cmp_eq_i32 s10, 0 // (op 2520) 
s_mov_b32 s47, scc // (op 2520) 
s_cmp_eq_i32 s47, 0 // (op 2520) 
s_cselect_b32 s18, 0, -1 // (op 2520) 

s_and_b32 s19, s13, 31 // (op 2520) 
s_lshl_b32 s47, 1, s19 // (op 2520) 
s_add_i32 s22, s47, s18 // (op 2520) 


s_and_b32 s18, s1, s22 // (op 2520) 


s_add_i32 s1, s0, s18 // (op 2520) 


s_ashr_i32 s0, s1, s19 // (op 2520) 


s_lshl_b32 s1, s13, 24 // (op 2520) 
s_ashr_i32 s47, s1, 31 // (op 2520) 

s_xor_b32 s1, s0, s47 // (op 2520) 

s_sub_i32 s0, s1, s47 // (op 2520) 


s_mul_hi_i32 s1, s0, s14 // (op 2520) 
s_add_i32 s47, s1, s0 // (op 2520) 

s_ashr_i32 s1, s47, 31 // (op 2520) 
s_cmp_eq_i32 s14, 0 // (op 2520) 
s_mov_b32 s18, scc // (op 2520) 
s_cmp_eq_i32 s18, 0 // (op 2520) 
s_cselect_b32 s19, 0, -1 // (op 2520) 

s_and_b32 s18, s15, 31 // (op 2520) 
s_lshl_b32 s22, 1, s18 // (op 2520) 
s_add_i32 s23, s22, s19 // (op 2520) 


s_and_b32 s19, s1, s23 // (op 2520) 


s_add_i32 s1, s47, s19 // (op 2520) 


s_ashr_i32 s19, s1, s18 // (op 2520) 


s_lshl_b32 s1, s15, 24 // (op 2520) 
s_ashr_i32 s18, s1, 31 // (op 2520) 

s_xor_b32 s1, s19, s18 // (op 2520) 

s_sub_i32 s19, s1, s18 // (op 2520) 


s_mul_i32 s1, s0, s11 // (op 2520) 

s_sub_u32 s0, s16, s1 // (op 2520) 

s_mul_i32 s1, s19, s11 // (op 2520) 
s_add_u32 s1, s1, s0 // (op 2520) 


s_mul_hi_i32 s0, s16, s9 // (op 2520) 
s_add_i32 s47, s0, s16 // (op 2520) 

s_ashr_i32 s0, s47, 31 // (op 2520) 
s_cmp_eq_i32 s9, 0 // (op 2520) 
s_mov_b32 s18, scc // (op 2520) 
s_cmp_eq_i32 s18, 0 // (op 2520) 
s_cselect_b32 s19, 0, -1 // (op 2520) 

s_and_b32 s18, s45, 31 // (op 2520) 
s_lshl_b32 s22, 1, s18 // (op 2520) 
s_add_i32 s23, s22, s19 // (op 2520) 


s_and_b32 s19, s0, s23 // (op 2520) 


s_add_i32 s0, s47, s19 // (op 2520) 


s_ashr_i32 s19, s0, s18 // (op 2520) 


s_lshl_b32 s0, s45, 24 // (op 2520) 
s_ashr_i32 s18, s0, 31 // (op 2520) 

s_xor_b32 s0, s19, s18 // (op 2520) 

s_sub_i32 s19, s0, s18 // (op 2520) 


s_add_u32 s22, s6, 256 // (op 2520) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2520) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2520) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2520) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2520) 

s_mul_i32 s0, s11, s22 // (op 2520) 

s_mul_i32 s18, s8, s0 // (op 2520) 

s_mul_i32 s0, s19, s18 // (op 2520) 

s_sub_u32 s18, s16, s0 // (op 2520) 

s_mov_b32 s0, s18 // (op 2520) 
s_mul_hi_i32 s47, s0, s12 // (op 2520) 
s_add_i32 s22, s47, s0 // (op 2520) 


s_ashr_i32 s0, s22, 31 // (op 2520) 
s_cmp_eq_i32 s12, 0 // (op 2520) 
s_mov_b32 s47, scc // (op 2520) 
s_cmp_eq_i32 s47, 0 // (op 2520) 
s_cselect_b32 s23, 0, -1 // (op 2520) 

s_and_b32 s47, s44, 31 // (op 2520) 
s_lshl_b32 s24, 1, s47 // (op 2520) 
s_add_i32 s25, s24, s23 // (op 2520) 


s_and_b32 s23, s0, s25 // (op 2520) 


s_add_i32 s0, s22, s23 // (op 2520) 


s_ashr_i32 s22, s0, s47 // (op 2520) 


s_lshl_b32 s0, s44, 24 // (op 2520) 
s_ashr_i32 s47, s0, 31 // (op 2520) 

s_xor_b32 s0, s22, s47 // (op 2520) 

s_sub_i32 s22, s0, s47 // (op 2520) 


s_mul_i32 s0, s46, s11 // (op 2520) 
s_add_u32 s24, s20, 64 // (op 2520) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2520) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 2520) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 2520) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 2520) 

s_sub_i32 s23, s24, s0 // (op 2520) 

s_mul_i32 s47, s22, s23 // (op 2520) 


s_sub_u32 s22, s18, s47 // (op 2520) 


s_add_u32 s18, s22, s0 // (op 2520) 


s_cmp_eq_i32 s19, 0 // (op 2520) 
s_mov_b32 s0, scc // (op 2520) 

s_cmp_eq_i32 s0, 0 // (op 2520) 
s_cselect_b32 s19, s18, s1 // (op 2520) 



v_lshl_add_u32 v19, s19, 6, v57 // (op 2520) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2520) 
s_lshr_b32 s47, s0, 1 // (op 2520) 

v_lshl_add_u32 v57, s47, 3, v20 // (op 2520) 


v_mul_lo_u32 v20, v19, s34 // (op 2520) 
v_add_u32 v20, v20, v57 // (op 2520) 


s_mul_i32 s1, s34, s21 // (op 2522) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 2522) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 2522) least significant
s_mul_hi_u32 s18, s34, s20 // (op 2522) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2522) most significant: sum
s_add_u32 s1, s1, s18 // (op 2522) most significant: sum


s_mov_b32 s64, s36 // (op 2522) reg expression
s_mov_b32 s65, s37 // (op 2522) reg expression
s_mov_b32 s66, s0 // (op 2522) reg expression
s_mov_b32 s67, 131072 // (op 2522) call()

s_mul_hi_i32 s47, s16, s10 // (op 2584) 
s_add_i32 s0, s47, s16 // (op 2584) 

s_ashr_i32 s1, s0, 31 // (op 2584) 
s_cmp_eq_i32 s10, 0 // (op 2584) 
s_mov_b32 s47, scc // (op 2584) 
s_cmp_eq_i32 s47, 0 // (op 2584) 
s_cselect_b32 s18, 0, -1 // (op 2584) 

s_and_b32 s19, s13, 31 // (op 2584) 
s_lshl_b32 s47, 1, s19 // (op 2584) 
s_add_i32 s22, s47, s18 // (op 2584) 


s_and_b32 s18, s1, s22 // (op 2584) 


s_add_i32 s1, s0, s18 // (op 2584) 


s_ashr_i32 s0, s1, s19 // (op 2584) 


s_lshl_b32 s1, s13, 24 // (op 2584) 
s_ashr_i32 s47, s1, 31 // (op 2584) 

s_xor_b32 s1, s0, s47 // (op 2584) 

s_sub_i32 s0, s1, s47 // (op 2584) 


s_mul_hi_i32 s1, s0, s14 // (op 2584) 
s_add_i32 s47, s1, s0 // (op 2584) 

s_ashr_i32 s1, s47, 31 // (op 2584) 
s_cmp_eq_i32 s14, 0 // (op 2584) 
s_mov_b32 s18, scc // (op 2584) 
s_cmp_eq_i32 s18, 0 // (op 2584) 
s_cselect_b32 s19, 0, -1 // (op 2584) 

s_and_b32 s18, s15, 31 // (op 2584) 
s_lshl_b32 s22, 1, s18 // (op 2584) 
s_add_i32 s23, s22, s19 // (op 2584) 


s_and_b32 s19, s1, s23 // (op 2584) 


s_add_i32 s1, s47, s19 // (op 2584) 


s_ashr_i32 s19, s1, s18 // (op 2584) 


s_lshl_b32 s1, s15, 24 // (op 2584) 
s_ashr_i32 s18, s1, 31 // (op 2584) 

s_xor_b32 s1, s19, s18 // (op 2584) 

s_sub_i32 s19, s1, s18 // (op 2584) 


s_add_u32 s22, s6, 256 // (op 2584) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2584) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2584) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2584) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2584) 

s_mul_i32 s1, s19, s22 // (op 2584) 

s_sub_i32 s47, s0, s1 // (op 2584) 


s_mul_hi_i32 s0, s16, s9 // (op 2584) 
s_add_i32 s1, s0, s16 // (op 2584) 

s_ashr_i32 s0, s1, 31 // (op 2584) 
s_cmp_eq_i32 s9, 0 // (op 2584) 
s_mov_b32 s18, scc // (op 2584) 
s_cmp_eq_i32 s18, 0 // (op 2584) 
s_cselect_b32 s19, 0, -1 // (op 2584) 

s_and_b32 s18, s45, 31 // (op 2584) 
s_lshl_b32 s24, 1, s18 // (op 2584) 
s_add_i32 s25, s24, s19 // (op 2584) 


s_and_b32 s19, s0, s25 // (op 2584) 


s_add_i32 s0, s1, s19 // (op 2584) 


s_ashr_i32 s1, s0, s18 // (op 2584) 


s_lshl_b32 s0, s45, 24 // (op 2584) 
s_ashr_i32 s18, s0, 31 // (op 2584) 

s_xor_b32 s0, s1, s18 // (op 2584) 

s_sub_i32 s1, s0, s18 // (op 2584) 


s_mul_i32 s0, s11, s22 // (op 2584) 

s_mul_i32 s18, s8, s0 // (op 2584) 

s_mul_i32 s0, s1, s18 // (op 2584) 

s_sub_u32 s18, s16, s0 // (op 2584) 

s_mul_hi_i32 s0, s18, s12 // (op 2584) 
s_add_i32 s19, s0, s18 // (op 2584) 


s_ashr_i32 s0, s19, 31 // (op 2584) 
s_cmp_eq_i32 s12, 0 // (op 2584) 
s_mov_b32 s18, scc // (op 2584) 
s_cmp_eq_i32 s18, 0 // (op 2584) 
s_cselect_b32 s22, 0, -1 // (op 2584) 

s_and_b32 s18, s44, 31 // (op 2584) 
s_lshl_b32 s23, 1, s18 // (op 2584) 
s_add_i32 s24, s23, s22 // (op 2584) 


s_and_b32 s25, s0, s24 // (op 2584) 


s_add_i32 s0, s19, s25 // (op 2584) 


s_ashr_i32 s19, s0, s18 // (op 2584) 


s_lshl_b32 s0, s44, 24 // (op 2584) 
s_ashr_i32 s18, s0, 31 // (op 2584) 

s_xor_b32 s0, s19, s18 // (op 2584) 

s_sub_i32 s19, s0, s18 // (op 2584) 


s_cmp_eq_i32 s1, 0 // (op 2584) 
s_mov_b32 s0, scc // (op 2584) 

s_cmp_eq_i32 s0, 0 // (op 2584) 
s_cselect_b32 s1, s19, s47 // (op 2584) 



v_lshl_add_u32 v19, s1, 8, v1 // (op 2584) 

v_mul_lo_u32 v57, v19, s38 // (op 2584) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 2584) 
s_lshr_b32 s47, s0, 1 // (op 2584) 

v_lshl_add_u32 v19, s47, 3, v57 // (op 2584) 


s_mul_i32 s1, s38, s7 // (op 2586) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 2586) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 2586) least significant
s_mul_hi_u32 s18, s38, s6 // (op 2586) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2586) most significant: sum
s_add_u32 s1, s1, s18 // (op 2586) most significant: sum


s_mov_b32 s68, s40 // (op 2586) reg expression
s_mov_b32 s69, s41 // (op 2586) reg expression
s_mov_b32 s70, s0 // (op 2586) reg expression
s_mov_b32 s71, 131072 // (op 2586) call()

v_and_b32 v57, -4, v1 // (op 2677) 
v_sub_u32 v22, v1, v57 // (op 2677) 

v_lshlrev_b32 v23, 5, v22 // (op 2677) 

v_lshrrev_b32 v22, 2, v1 // (op 2677) 
v_lshl_add_u32 v57, v22, 7, v23 // (op 2677) 


v_mul_lo_u32 v22, v57, 6 // (op 2677) 

v_lshrrev_b32 v23, 3, v22 // (op 2677) 

v_and_b32 v22, -2, v1 // (op 2730) 
v_sub_u32 v57, v1, v22 // (op 2730) 

v_lshlrev_b32 v22, 6, v57 // (op 2730) 

v_lshrrev_b32 v57, 1, v1 // (op 2730) 
v_lshl_add_u32 v26, v57, 8, v22 // (op 2730) 


v_mul_lo_u32 v22, v26, 6 // (op 2730) 

v_lshrrev_b32 v57, 3, v22 // (op 2730) 

v_and_b32 v22, -4, v1 // (op 2788) 
v_sub_u32 v26, v1, v22 // (op 2788) 

v_and_b32 v22, 4294967292, v1 // (op 2788) 
v_add_u32 v27, v22, v26 // (op 2788) 


v_lshlrev_b32 v22, 2, v1 // (op 2841) 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 873) 
s_and_b32 s47, s0, 4294967294 // (op 873) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 872) 
s_mov_b32 s18, s0 // (op 872) convert

s_cmp_lt_u32 s47, s18 // (op 872) 















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_872 // (op 872) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_872 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_872:


s_add_u32 s47, s47, 1 // (op 874) 
v_mov_b32 v14, v15 // (op 198) 

buffer_load_dwordx4 v[28:31], v14, s[48:51], 0 offen // (op 198) Load value
buffer_load_dwordx2 v[32:33], v14, s[48:51], 0 offen offset:16 // (op 198) Load value

v_add_u32 v15, 96, v15 // (op 2376) 

s_barrier  // (op 3067) 
s_mul_i32 s1, s31, 6 // (op 221) most significant: low of high * low



s_mul_i32 s0, s30, 6 // (op 221) least significant
s_mul_hi_u32 s18, s30, 6 // (op 221) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 221) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 221) 


v_mov_b32 v14, v17 // (op 221) 
v_mov_b32 v16, v14 // (op 221) 
buffer_load_dwordx4 v[58:61], v16, s[60:63], 0 offen // (op 221) Load value
s_waitcnt vmcnt(2)
ds_write_b128 v23, v[28:31] offset:25600 // (op 199) Store local data 
buffer_load_dwordx2 v[62:63], v16, s[60:63], 0 offen offset:16 // (op 221) Load value
buffer_load_dwordx4 v[64:67], v16, s[60:63], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx2 v[68:69], v16, s[60:63], 0 offen offset:40 // (op 221) Load value
v_add_u32 v16, v16, v24 // (op 221) 
s_waitcnt vmcnt(4)
ds_write_b64 v23, v[32:33] offset:25616 // (op 199) Store local data 
buffer_load_dwordx4 v[70:73], v16, s[60:63], 0 offen // (op 221) Load value

buffer_load_dwordx2 v[74:75], v16, s[60:63], 0 offen offset:16 // (op 221) Load value
buffer_load_dwordx4 v[76:79], v16, s[60:63], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx2 v[80:81], v16, s[60:63], 0 offen offset:40 // (op 221) Load value



v_add_u32 v17, 96, v17 // (op 2448) 


s_barrier  // (op 3086) 
v_mov_b32 v14, v20 // (op 222) 




buffer_load_ubyte v16, v14, s[64:67], 0 offen offset: 0 // (op 222) Load value

v_add_u32 v20, 4, v20 // (op 2512) 
s_waitcnt vmcnt(8)
ds_write_b128 v57, v[58:61] // (op 218) Store local data 


v_mov_b32 v14, v19 // (op 223) 

buffer_load_dword v18, v14, s[68:71], 0 offen // (op 223) Load value

v_add_u32 v19, 4, v19 // (op 2576) 
s_waitcnt vmcnt(8)
ds_write_b64 v57, v[62:63] offset:16 // (op 218) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3076) 
s_waitcnt vmcnt(7)
ds_write_b128 v57, v[64:67] offset:24 // (op 218) Store local data 



ds_read_b128 v[28:31], v3 offset:25600 // (op 201) Load local data 
ds_read_b64 v[32:33], v3 offset:25616 // (op 201) Load local data 
s_waitcnt vmcnt(6)
ds_write_b64 v57, v[68:69] offset:40 // (op 218) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v57, v[70:73] offset:96 // (op 218) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v57, v[74:75] offset:112 // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v57, v[76:79] offset:120 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v57, v[80:81] offset:136 // (op 218) Store local data 

s_barrier  // (op 3107) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3097) 






ds_read_b128 v[34:37], v9 // (op 203) Load local data 
ds_read_b64 v[38:39], v9 offset:16 // (op 203) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v27, v16 offset:31744 // (op 219) Store local data 

s_barrier  // (op 3127) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3117) 






ds_read_u8 v14, v0 offset:31744 // (op 205) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v22, v18 offset:24576 // (op 220) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3137) 



ds_read_u8 v16, v6 offset:24576 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[28:33], v[34:39], a[0:15], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 208) 





ds_read_b128 v[34:37], v3 offset:25648 // (op 210) Load local data 
ds_read_b64 v[38:39], v3 offset:25664 // (op 210) Load local data 



ds_read_b128 v[40:43], v9 offset:48 // (op 212) Load local data 
ds_read_b64 v[44:45], v9 offset:64 // (op 212) Load local data 



ds_read_u8 v16, v0 offset:31746 // (op 214) Load local data 



ds_read_u8 v18, v6 offset:24578 // (op 216) Load local data 



ds_read_b128 v[46:49], v9 offset:6144 // (op 412) Load local data 
ds_read_b64 v[50:51], v9 offset:6160 // (op 412) Load local data 



ds_read_u8 v21, v6 offset:24832 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[28:33], v[46:51], a[16:31], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 417) 





ds_read_b128 v[46:49], v9 offset:6192 // (op 421) Load local data 
ds_read_b64 v[50:51], v9 offset:6208 // (op 421) Load local data 



ds_read_u8 v21, v6 offset:24834 // (op 425) Load local data 



ds_read_b128 v[58:61], v9 offset:12288 // (op 646) Load local data 
ds_read_b64 v[62:63], v9 offset:12304 // (op 646) Load local data 



ds_read_u8 v24, v6 offset:25088 // (op 650) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[28:33], v[58:63], a[32:47], v14, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 651) 





ds_read_b128 v[58:61], v9 offset:12336 // (op 655) Load local data 
ds_read_b64 v[62:63], v9 offset:12352 // (op 655) Load local data 



ds_read_u8 v24, v6 offset:25090 // (op 659) Load local data 



ds_read_b128 v[64:67], v9 offset:18432 // (op 880) Load local data 
ds_read_b64 v[68:69], v9 offset:18448 // (op 880) Load local data 



ds_read_u8 v25, v6 offset:25344 // (op 884) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[28:33], v[64:69], a[48:63], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 885) 







v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[34:39], v[40:45], a[0:15], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 217) 
ds_read_b128 v[28:31], v9 offset:18480 // (op 889) Load local data 


ds_read_b64 v[32:33], v9 offset:18496 // (op 889) Load local data 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[34:39], v[46:51], a[16:31], v16, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 426) 
ds_read_u8 v14, v6 offset:25346 // (op 893) Load local data 


v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[34:39], v[58:63], a[32:47], v16, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 660) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[34:39], v[28:33], a[48:63], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 894) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 872) 
s_mov_b32 s18, s0 // (op 872) convert

s_cmp_lt_u32 s47, s18 // (op 872) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_872 // (op 872) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_872 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_872:






















s_barrier  // (op 3254) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_155
v_accvgpr_read v12, a0 // (op 50) 
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_accvgpr_read v16, a4 // (op 50) 
v_accvgpr_read v17, a5 // (op 50) 
v_accvgpr_read v18, a6 // (op 50) 
v_accvgpr_read v19, a7 // (op 50) 
v_accvgpr_read v20, a8 // (op 50) 
v_accvgpr_read v21, a9 // (op 50) 
v_accvgpr_read v22, a10 // (op 50) 
v_accvgpr_read v23, a11 // (op 50) 
v_accvgpr_read v24, a12 // (op 50) 
v_accvgpr_read v25, a13 // (op 50) 
v_accvgpr_read v26, a14 // (op 50) 
v_accvgpr_read v27, a15 // (op 50) 
v_mul_f32 v28, v10, v12 // (op 50) 
v_mul_f32 v29, v10, v13 // (op 50) 
v_mul_f32 v30, v10, v14 // (op 50) 
v_mul_f32 v31, v10, v15 // (op 50) 
v_mul_f32 v32, v10, v16 // (op 50) 
v_mul_f32 v33, v10, v17 // (op 50) 
v_mul_f32 v34, v10, v18 // (op 50) 
v_mul_f32 v35, v10, v19 // (op 50) 
v_mul_f32 v36, v10, v20 // (op 50) 
v_mul_f32 v37, v10, v21 // (op 50) 
v_mul_f32 v38, v10, v22 // (op 50) 
v_mul_f32 v39, v10, v23 // (op 50) 
v_mul_f32 v40, v10, v24 // (op 50) 
v_mul_f32 v41, v10, v25 // (op 50) 
v_mul_f32 v42, v10, v26 // (op 50) 
v_mul_f32 v43, v10, v27 // (op 50) 



v_mov_b32 v0, v11 // (op 56) reg expression

buffer_store_dwordx4 v[28:31], v0, s[56:59], 0 offen // (op 56) Store value
buffer_store_dwordx4 v[32:35], v0, s[56:59], 0 offen offset:32 // (op 56) Store value
buffer_store_dwordx4 v[36:39], v0, s[56:59], 0 offen offset:64 // (op 56) Store value
buffer_store_dwordx4 v[40:43], v0, s[56:59], 0 offen offset:96 // (op 56) Store value


v_accvgpr_read v12, a16 // (op 1258) 
v_accvgpr_read v13, a17 // (op 1258) 
v_accvgpr_read v14, a18 // (op 1258) 
v_accvgpr_read v15, a19 // (op 1258) 
v_accvgpr_read v16, a20 // (op 1258) 
v_accvgpr_read v17, a21 // (op 1258) 
v_accvgpr_read v18, a22 // (op 1258) 
v_accvgpr_read v19, a23 // (op 1258) 
v_accvgpr_read v20, a24 // (op 1258) 
v_accvgpr_read v21, a25 // (op 1258) 
v_accvgpr_read v22, a26 // (op 1258) 
v_accvgpr_read v23, a27 // (op 1258) 
v_accvgpr_read v24, a28 // (op 1258) 
v_accvgpr_read v25, a29 // (op 1258) 
v_accvgpr_read v26, a30 // (op 1258) 
v_accvgpr_read v27, a31 // (op 1258) 
v_mul_f32 v28, v10, v12 // (op 1258) 
v_mul_f32 v29, v10, v13 // (op 1258) 
v_mul_f32 v30, v10, v14 // (op 1258) 
v_mul_f32 v31, v10, v15 // (op 1258) 
v_mul_f32 v32, v10, v16 // (op 1258) 
v_mul_f32 v33, v10, v17 // (op 1258) 
v_mul_f32 v34, v10, v18 // (op 1258) 
v_mul_f32 v35, v10, v19 // (op 1258) 
v_mul_f32 v36, v10, v20 // (op 1258) 
v_mul_f32 v37, v10, v21 // (op 1258) 
v_mul_f32 v38, v10, v22 // (op 1258) 
v_mul_f32 v39, v10, v23 // (op 1258) 
v_mul_f32 v40, v10, v24 // (op 1258) 
v_mul_f32 v41, v10, v25 // (op 1258) 
v_mul_f32 v42, v10, v26 // (op 1258) 
v_mul_f32 v43, v10, v27 // (op 1258) 



s_lshl_b32 s0, s4, 8 // (op 1260) 
v_add_u32 v0, v11, s0 // (op 1260) 


buffer_store_dwordx4 v[28:31], v0, s[56:59], 0 offen // (op 1260) Store value
buffer_store_dwordx4 v[32:35], v0, s[56:59], 0 offen offset:32 // (op 1260) Store value
buffer_store_dwordx4 v[36:39], v0, s[56:59], 0 offen offset:64 // (op 1260) Store value
buffer_store_dwordx4 v[40:43], v0, s[56:59], 0 offen offset:96 // (op 1260) Store value


v_accvgpr_read v12, a32 // (op 1270) 
v_accvgpr_read v13, a33 // (op 1270) 
v_accvgpr_read v14, a34 // (op 1270) 
v_accvgpr_read v15, a35 // (op 1270) 
v_accvgpr_read v16, a36 // (op 1270) 
v_accvgpr_read v17, a37 // (op 1270) 
v_accvgpr_read v18, a38 // (op 1270) 
v_accvgpr_read v19, a39 // (op 1270) 
v_accvgpr_read v20, a40 // (op 1270) 
v_accvgpr_read v21, a41 // (op 1270) 
v_accvgpr_read v22, a42 // (op 1270) 
v_accvgpr_read v23, a43 // (op 1270) 
v_accvgpr_read v24, a44 // (op 1270) 
v_accvgpr_read v25, a45 // (op 1270) 
v_accvgpr_read v26, a46 // (op 1270) 
v_accvgpr_read v27, a47 // (op 1270) 
v_mul_f32 v28, v10, v12 // (op 1270) 
v_mul_f32 v29, v10, v13 // (op 1270) 
v_mul_f32 v30, v10, v14 // (op 1270) 
v_mul_f32 v31, v10, v15 // (op 1270) 
v_mul_f32 v32, v10, v16 // (op 1270) 
v_mul_f32 v33, v10, v17 // (op 1270) 
v_mul_f32 v34, v10, v18 // (op 1270) 
v_mul_f32 v35, v10, v19 // (op 1270) 
v_mul_f32 v36, v10, v20 // (op 1270) 
v_mul_f32 v37, v10, v21 // (op 1270) 
v_mul_f32 v38, v10, v22 // (op 1270) 
v_mul_f32 v39, v10, v23 // (op 1270) 
v_mul_f32 v40, v10, v24 // (op 1270) 
v_mul_f32 v41, v10, v25 // (op 1270) 
v_mul_f32 v42, v10, v26 // (op 1270) 
v_mul_f32 v43, v10, v27 // (op 1270) 



s_lshl_b32 s0, s4, 9 // (op 1272) 
v_add_u32 v0, v11, s0 // (op 1272) 


buffer_store_dwordx4 v[28:31], v0, s[56:59], 0 offen // (op 1272) Store value
buffer_store_dwordx4 v[32:35], v0, s[56:59], 0 offen offset:32 // (op 1272) Store value
buffer_store_dwordx4 v[36:39], v0, s[56:59], 0 offen offset:64 // (op 1272) Store value
buffer_store_dwordx4 v[40:43], v0, s[56:59], 0 offen offset:96 // (op 1272) Store value


v_accvgpr_read v12, a48 // (op 1282) 
v_accvgpr_read v13, a49 // (op 1282) 
v_accvgpr_read v14, a50 // (op 1282) 
v_accvgpr_read v15, a51 // (op 1282) 
v_accvgpr_read v16, a52 // (op 1282) 
v_accvgpr_read v17, a53 // (op 1282) 
v_accvgpr_read v18, a54 // (op 1282) 
v_accvgpr_read v19, a55 // (op 1282) 
v_accvgpr_read v20, a56 // (op 1282) 
v_accvgpr_read v21, a57 // (op 1282) 
v_accvgpr_read v22, a58 // (op 1282) 
v_accvgpr_read v23, a59 // (op 1282) 
v_accvgpr_read v24, a60 // (op 1282) 
v_accvgpr_read v25, a61 // (op 1282) 
v_accvgpr_read v26, a62 // (op 1282) 
v_accvgpr_read v27, a63 // (op 1282) 
v_mul_f32 v28, v10, v12 // (op 1282) 
v_mul_f32 v29, v10, v13 // (op 1282) 
v_mul_f32 v30, v10, v14 // (op 1282) 
v_mul_f32 v31, v10, v15 // (op 1282) 
v_mul_f32 v32, v10, v16 // (op 1282) 
v_mul_f32 v33, v10, v17 // (op 1282) 
v_mul_f32 v34, v10, v18 // (op 1282) 
v_mul_f32 v35, v10, v19 // (op 1282) 
v_mul_f32 v36, v10, v20 // (op 1282) 
v_mul_f32 v37, v10, v21 // (op 1282) 
v_mul_f32 v38, v10, v22 // (op 1282) 
v_mul_f32 v39, v10, v23 // (op 1282) 
v_mul_f32 v40, v10, v24 // (op 1282) 
v_mul_f32 v41, v10, v25 // (op 1282) 
v_mul_f32 v42, v10, v26 // (op 1282) 
v_mul_f32 v43, v10, v27 // (op 1282) 



s_lshl_b32 s0, s4, 8 // (op 1284) 
s_mul_i32 s1, 3, s0 // (op 1284) 

v_add_u32 v0, v11, s1 // (op 1284) 


buffer_store_dwordx4 v[28:31], v0, s[56:59], 0 offen // (op 1284) Store value
buffer_store_dwordx4 v[32:35], v0, s[56:59], 0 offen offset:32 // (op 1284) Store value
buffer_store_dwordx4 v[36:39], v0, s[56:59], 0 offen offset:64 // (op 1284) Store value
buffer_store_dwordx4 v[40:43], v0, s[56:59], 0 offen offset:96 // (op 1284) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_155
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_155:


v_mov_b32 v0, v8 // (op 167) reg expression

buffer_load_dwordx4 v[12:15], v0, s[52:55], 0 offen // (op 167) Load value
buffer_load_dwordx4 v[16:19], v0, s[52:55], 0 offen offset:32 // (op 167) Load value
buffer_load_dwordx4 v[20:23], v0, s[52:55], 0 offen offset:64 // (op 167) Load value
v_accvgpr_read v28, a0 // (op 164) 
v_accvgpr_read v29, a1 // (op 164) 
v_accvgpr_read v30, a2 // (op 164) 
v_accvgpr_read v31, a3 // (op 164) 
v_accvgpr_read v32, a4 // (op 164) 
v_accvgpr_read v33, a5 // (op 164) 
v_accvgpr_read v34, a6 // (op 164) 
v_accvgpr_read v35, a7 // (op 164) 
v_accvgpr_read v36, a8 // (op 164) 
v_accvgpr_read v37, a9 // (op 164) 
v_accvgpr_read v38, a10 // (op 164) 
v_accvgpr_read v39, a11 // (op 164) 
v_accvgpr_read v40, a12 // (op 164) 
v_accvgpr_read v41, a13 // (op 164) 
v_accvgpr_read v42, a14 // (op 164) 
v_accvgpr_read v43, a15 // (op 164) 
v_mul_f32 v44, v10, v28 // (op 164) 
v_mul_f32 v45, v10, v29 // (op 164) 
v_mul_f32 v46, v10, v30 // (op 164) 
v_mul_f32 v47, v10, v31 // (op 164) 
v_mul_f32 v48, v10, v32 // (op 164) 
v_mul_f32 v49, v10, v33 // (op 164) 
v_mul_f32 v50, v10, v34 // (op 164) 
v_mul_f32 v51, v10, v35 // (op 164) 
v_mul_f32 v52, v10, v36 // (op 164) 
v_mul_f32 v53, v10, v37 // (op 164) 
v_mul_f32 v54, v10, v38 // (op 164) 
v_mul_f32 v55, v10, v39 // (op 164) 
v_mul_f32 v56, v10, v40 // (op 164) 
v_mul_f32 v57, v10, v41 // (op 164) 
v_mul_f32 v58, v10, v42 // (op 164) 
v_mul_f32 v59, v10, v43 // (op 164) 

buffer_load_dwordx4 v[24:27], v0, s[52:55], 0 offen offset:96 // (op 167) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v28, v7, v12 // (op 168) 
v_mul_f32 v29, v7, v13 // (op 168) 
v_mul_f32 v30, v7, v14 // (op 168) 
v_mul_f32 v31, v7, v15 // (op 168) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v7, v16 // (op 168) 
v_mul_f32 v33, v7, v17 // (op 168) 
v_mul_f32 v34, v7, v18 // (op 168) 
v_mul_f32 v35, v7, v19 // (op 168) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v7, v20 // (op 168) 
v_mul_f32 v37, v7, v21 // (op 168) 
v_mul_f32 v38, v7, v22 // (op 168) 
v_mul_f32 v39, v7, v23 // (op 168) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v7, v24 // (op 168) 
v_mul_f32 v41, v7, v25 // (op 168) 
v_mul_f32 v42, v7, v26 // (op 168) 
v_mul_f32 v43, v7, v27 // (op 168) 

v_add_f32 v12, v28, v44 // (op 165) 
v_add_f32 v13, v29, v45 // (op 165) 
v_add_f32 v14, v30, v46 // (op 165) 
v_add_f32 v15, v31, v47 // (op 165) 
v_add_f32 v16, v32, v48 // (op 165) 
v_add_f32 v17, v33, v49 // (op 165) 
v_add_f32 v18, v34, v50 // (op 165) 
v_add_f32 v19, v35, v51 // (op 165) 
v_add_f32 v20, v36, v52 // (op 165) 
v_add_f32 v21, v37, v53 // (op 165) 
v_add_f32 v22, v38, v54 // (op 165) 
v_add_f32 v23, v39, v55 // (op 165) 
v_add_f32 v24, v40, v56 // (op 165) 
v_add_f32 v25, v41, v57 // (op 165) 
v_add_f32 v26, v42, v58 // (op 165) 
v_add_f32 v27, v43, v59 // (op 165) 




v_mov_b32 v0, v11 // (op 166) reg expression

buffer_store_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 166) Store value
buffer_store_dwordx4 v[16:19], v0, s[56:59], 0 offen offset:32 // (op 166) Store value
buffer_store_dwordx4 v[20:23], v0, s[56:59], 0 offen offset:64 // (op 166) Store value
buffer_store_dwordx4 v[24:27], v0, s[56:59], 0 offen offset:96 // (op 166) Store value




s_lshl_b32 s0, s2, 8 // (op 1316) 
v_add_u32 v0, v8, s0 // (op 1316) 


buffer_load_dwordx4 v[12:15], v0, s[52:55], 0 offen // (op 1316) Load value
buffer_load_dwordx4 v[16:19], v0, s[52:55], 0 offen offset:32 // (op 1316) Load value
v_accvgpr_read v28, a16 // (op 1313) 
v_accvgpr_read v29, a17 // (op 1313) 
v_accvgpr_read v30, a18 // (op 1313) 
v_accvgpr_read v31, a19 // (op 1313) 
v_accvgpr_read v32, a20 // (op 1313) 
v_accvgpr_read v33, a21 // (op 1313) 
v_accvgpr_read v34, a22 // (op 1313) 
v_accvgpr_read v35, a23 // (op 1313) 
v_accvgpr_read v36, a24 // (op 1313) 
v_accvgpr_read v37, a25 // (op 1313) 
v_accvgpr_read v38, a26 // (op 1313) 
v_accvgpr_read v39, a27 // (op 1313) 
v_accvgpr_read v40, a28 // (op 1313) 
v_accvgpr_read v41, a29 // (op 1313) 
v_accvgpr_read v42, a30 // (op 1313) 
v_accvgpr_read v43, a31 // (op 1313) 
v_mul_f32 v44, v10, v28 // (op 1313) 
v_mul_f32 v45, v10, v29 // (op 1313) 
v_mul_f32 v46, v10, v30 // (op 1313) 
v_mul_f32 v47, v10, v31 // (op 1313) 
v_mul_f32 v48, v10, v32 // (op 1313) 
v_mul_f32 v49, v10, v33 // (op 1313) 
v_mul_f32 v50, v10, v34 // (op 1313) 
v_mul_f32 v51, v10, v35 // (op 1313) 
v_mul_f32 v52, v10, v36 // (op 1313) 
v_mul_f32 v53, v10, v37 // (op 1313) 
v_mul_f32 v54, v10, v38 // (op 1313) 
v_mul_f32 v55, v10, v39 // (op 1313) 
v_mul_f32 v56, v10, v40 // (op 1313) 
v_mul_f32 v57, v10, v41 // (op 1313) 
v_mul_f32 v58, v10, v42 // (op 1313) 
v_mul_f32 v59, v10, v43 // (op 1313) 

buffer_load_dwordx4 v[20:23], v0, s[52:55], 0 offen offset:64 // (op 1316) Load value
buffer_load_dwordx4 v[24:27], v0, s[52:55], 0 offen offset:96 // (op 1316) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v28, v7, v12 // (op 1317) 
v_mul_f32 v29, v7, v13 // (op 1317) 
v_mul_f32 v30, v7, v14 // (op 1317) 
v_mul_f32 v31, v7, v15 // (op 1317) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v7, v16 // (op 1317) 
v_mul_f32 v33, v7, v17 // (op 1317) 
v_mul_f32 v34, v7, v18 // (op 1317) 
v_mul_f32 v35, v7, v19 // (op 1317) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v7, v20 // (op 1317) 
v_mul_f32 v37, v7, v21 // (op 1317) 
v_mul_f32 v38, v7, v22 // (op 1317) 
v_mul_f32 v39, v7, v23 // (op 1317) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v7, v24 // (op 1317) 
v_mul_f32 v41, v7, v25 // (op 1317) 
v_mul_f32 v42, v7, v26 // (op 1317) 
v_mul_f32 v43, v7, v27 // (op 1317) 

v_add_f32 v12, v28, v44 // (op 1314) 
v_add_f32 v13, v29, v45 // (op 1314) 
v_add_f32 v14, v30, v46 // (op 1314) 
v_add_f32 v15, v31, v47 // (op 1314) 
v_add_f32 v16, v32, v48 // (op 1314) 
v_add_f32 v17, v33, v49 // (op 1314) 
v_add_f32 v18, v34, v50 // (op 1314) 
v_add_f32 v19, v35, v51 // (op 1314) 
v_add_f32 v20, v36, v52 // (op 1314) 
v_add_f32 v21, v37, v53 // (op 1314) 
v_add_f32 v22, v38, v54 // (op 1314) 
v_add_f32 v23, v39, v55 // (op 1314) 
v_add_f32 v24, v40, v56 // (op 1314) 
v_add_f32 v25, v41, v57 // (op 1314) 
v_add_f32 v26, v42, v58 // (op 1314) 
v_add_f32 v27, v43, v59 // (op 1314) 




s_lshl_b32 s0, s4, 8 // (op 1315) 
v_add_u32 v0, v11, s0 // (op 1315) 


buffer_store_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 1315) Store value
buffer_store_dwordx4 v[16:19], v0, s[56:59], 0 offen offset:32 // (op 1315) Store value
buffer_store_dwordx4 v[20:23], v0, s[56:59], 0 offen offset:64 // (op 1315) Store value
buffer_store_dwordx4 v[24:27], v0, s[56:59], 0 offen offset:96 // (op 1315) Store value




s_lshl_b32 s0, s2, 9 // (op 1328) 
v_add_u32 v0, v8, s0 // (op 1328) 


buffer_load_dwordx4 v[12:15], v0, s[52:55], 0 offen // (op 1328) Load value
buffer_load_dwordx4 v[16:19], v0, s[52:55], 0 offen offset:32 // (op 1328) Load value
v_accvgpr_read v28, a32 // (op 1325) 
v_accvgpr_read v29, a33 // (op 1325) 
v_accvgpr_read v30, a34 // (op 1325) 
v_accvgpr_read v31, a35 // (op 1325) 
v_accvgpr_read v32, a36 // (op 1325) 
v_accvgpr_read v33, a37 // (op 1325) 
v_accvgpr_read v34, a38 // (op 1325) 
v_accvgpr_read v35, a39 // (op 1325) 
v_accvgpr_read v36, a40 // (op 1325) 
v_accvgpr_read v37, a41 // (op 1325) 
v_accvgpr_read v38, a42 // (op 1325) 
v_accvgpr_read v39, a43 // (op 1325) 
v_accvgpr_read v40, a44 // (op 1325) 
v_accvgpr_read v41, a45 // (op 1325) 
v_accvgpr_read v42, a46 // (op 1325) 
v_accvgpr_read v43, a47 // (op 1325) 
v_mul_f32 v44, v10, v28 // (op 1325) 
v_mul_f32 v45, v10, v29 // (op 1325) 
v_mul_f32 v46, v10, v30 // (op 1325) 
v_mul_f32 v47, v10, v31 // (op 1325) 
v_mul_f32 v48, v10, v32 // (op 1325) 
v_mul_f32 v49, v10, v33 // (op 1325) 
v_mul_f32 v50, v10, v34 // (op 1325) 
v_mul_f32 v51, v10, v35 // (op 1325) 
v_mul_f32 v52, v10, v36 // (op 1325) 
v_mul_f32 v53, v10, v37 // (op 1325) 
v_mul_f32 v54, v10, v38 // (op 1325) 
v_mul_f32 v55, v10, v39 // (op 1325) 
v_mul_f32 v56, v10, v40 // (op 1325) 
v_mul_f32 v57, v10, v41 // (op 1325) 
v_mul_f32 v58, v10, v42 // (op 1325) 
v_mul_f32 v59, v10, v43 // (op 1325) 

buffer_load_dwordx4 v[20:23], v0, s[52:55], 0 offen offset:64 // (op 1328) Load value
buffer_load_dwordx4 v[24:27], v0, s[52:55], 0 offen offset:96 // (op 1328) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v28, v7, v12 // (op 1329) 
v_mul_f32 v29, v7, v13 // (op 1329) 
v_mul_f32 v30, v7, v14 // (op 1329) 
v_mul_f32 v31, v7, v15 // (op 1329) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v7, v16 // (op 1329) 
v_mul_f32 v33, v7, v17 // (op 1329) 
v_mul_f32 v34, v7, v18 // (op 1329) 
v_mul_f32 v35, v7, v19 // (op 1329) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v7, v20 // (op 1329) 
v_mul_f32 v37, v7, v21 // (op 1329) 
v_mul_f32 v38, v7, v22 // (op 1329) 
v_mul_f32 v39, v7, v23 // (op 1329) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v7, v24 // (op 1329) 
v_mul_f32 v41, v7, v25 // (op 1329) 
v_mul_f32 v42, v7, v26 // (op 1329) 
v_mul_f32 v43, v7, v27 // (op 1329) 

v_add_f32 v12, v28, v44 // (op 1326) 
v_add_f32 v13, v29, v45 // (op 1326) 
v_add_f32 v14, v30, v46 // (op 1326) 
v_add_f32 v15, v31, v47 // (op 1326) 
v_add_f32 v16, v32, v48 // (op 1326) 
v_add_f32 v17, v33, v49 // (op 1326) 
v_add_f32 v18, v34, v50 // (op 1326) 
v_add_f32 v19, v35, v51 // (op 1326) 
v_add_f32 v20, v36, v52 // (op 1326) 
v_add_f32 v21, v37, v53 // (op 1326) 
v_add_f32 v22, v38, v54 // (op 1326) 
v_add_f32 v23, v39, v55 // (op 1326) 
v_add_f32 v24, v40, v56 // (op 1326) 
v_add_f32 v25, v41, v57 // (op 1326) 
v_add_f32 v26, v42, v58 // (op 1326) 
v_add_f32 v27, v43, v59 // (op 1326) 




s_lshl_b32 s0, s4, 9 // (op 1327) 
v_add_u32 v0, v11, s0 // (op 1327) 


buffer_store_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 1327) Store value
buffer_store_dwordx4 v[16:19], v0, s[56:59], 0 offen offset:32 // (op 1327) Store value
buffer_store_dwordx4 v[20:23], v0, s[56:59], 0 offen offset:64 // (op 1327) Store value
buffer_store_dwordx4 v[24:27], v0, s[56:59], 0 offen offset:96 // (op 1327) Store value




s_lshl_b32 s0, s2, 8 // (op 1340) 
s_mul_i32 s1, 3, s0 // (op 1340) 

v_add_u32 v0, v8, s1 // (op 1340) 


buffer_load_dwordx4 v[12:15], v0, s[52:55], 0 offen // (op 1340) Load value
buffer_load_dwordx4 v[16:19], v0, s[52:55], 0 offen offset:32 // (op 1340) Load value
v_accvgpr_read v28, a48 // (op 1337) 
v_accvgpr_read v29, a49 // (op 1337) 
v_accvgpr_read v30, a50 // (op 1337) 
v_accvgpr_read v31, a51 // (op 1337) 
v_accvgpr_read v32, a52 // (op 1337) 
v_accvgpr_read v33, a53 // (op 1337) 
v_accvgpr_read v34, a54 // (op 1337) 
v_accvgpr_read v35, a55 // (op 1337) 
v_accvgpr_read v36, a56 // (op 1337) 
v_accvgpr_read v37, a57 // (op 1337) 
v_accvgpr_read v38, a58 // (op 1337) 
v_accvgpr_read v39, a59 // (op 1337) 
v_accvgpr_read v40, a60 // (op 1337) 
v_accvgpr_read v41, a61 // (op 1337) 
v_accvgpr_read v42, a62 // (op 1337) 
v_accvgpr_read v43, a63 // (op 1337) 
v_mul_f32 v44, v10, v28 // (op 1337) 
v_mul_f32 v45, v10, v29 // (op 1337) 
v_mul_f32 v46, v10, v30 // (op 1337) 
v_mul_f32 v47, v10, v31 // (op 1337) 
v_mul_f32 v48, v10, v32 // (op 1337) 
v_mul_f32 v49, v10, v33 // (op 1337) 
v_mul_f32 v50, v10, v34 // (op 1337) 
v_mul_f32 v51, v10, v35 // (op 1337) 
v_mul_f32 v52, v10, v36 // (op 1337) 
v_mul_f32 v53, v10, v37 // (op 1337) 
v_mul_f32 v54, v10, v38 // (op 1337) 
v_mul_f32 v55, v10, v39 // (op 1337) 
v_mul_f32 v56, v10, v40 // (op 1337) 
v_mul_f32 v57, v10, v41 // (op 1337) 
v_mul_f32 v58, v10, v42 // (op 1337) 
v_mul_f32 v59, v10, v43 // (op 1337) 

buffer_load_dwordx4 v[20:23], v0, s[52:55], 0 offen offset:64 // (op 1340) Load value
buffer_load_dwordx4 v[24:27], v0, s[52:55], 0 offen offset:96 // (op 1340) Load value

s_waitcnt vmcnt(3)
v_mul_f32 v28, v7, v12 // (op 1341) 
v_mul_f32 v29, v7, v13 // (op 1341) 
v_mul_f32 v30, v7, v14 // (op 1341) 
v_mul_f32 v31, v7, v15 // (op 1341) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v7, v16 // (op 1341) 
v_mul_f32 v33, v7, v17 // (op 1341) 
v_mul_f32 v34, v7, v18 // (op 1341) 
v_mul_f32 v35, v7, v19 // (op 1341) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v7, v20 // (op 1341) 
v_mul_f32 v37, v7, v21 // (op 1341) 
v_mul_f32 v38, v7, v22 // (op 1341) 
v_mul_f32 v39, v7, v23 // (op 1341) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v7, v24 // (op 1341) 
v_mul_f32 v41, v7, v25 // (op 1341) 
v_mul_f32 v42, v7, v26 // (op 1341) 
v_mul_f32 v43, v7, v27 // (op 1341) 

v_add_f32 v12, v28, v44 // (op 1338) 
v_add_f32 v13, v29, v45 // (op 1338) 
v_add_f32 v14, v30, v46 // (op 1338) 
v_add_f32 v15, v31, v47 // (op 1338) 
v_add_f32 v16, v32, v48 // (op 1338) 
v_add_f32 v17, v33, v49 // (op 1338) 
v_add_f32 v18, v34, v50 // (op 1338) 
v_add_f32 v19, v35, v51 // (op 1338) 
v_add_f32 v20, v36, v52 // (op 1338) 
v_add_f32 v21, v37, v53 // (op 1338) 
v_add_f32 v22, v38, v54 // (op 1338) 
v_add_f32 v23, v39, v55 // (op 1338) 
v_add_f32 v24, v40, v56 // (op 1338) 
v_add_f32 v25, v41, v57 // (op 1338) 
v_add_f32 v26, v42, v58 // (op 1338) 
v_add_f32 v27, v43, v59 // (op 1338) 




s_lshl_b32 s0, s4, 8 // (op 1339) 
s_mul_i32 s1, 3, s0 // (op 1339) 

v_add_u32 v0, v11, s1 // (op 1339) 


buffer_store_dwordx4 v[12:15], v0, s[56:59], 0 offen // (op 1339) Store value
buffer_store_dwordx4 v[16:19], v0, s[56:59], 0 offen offset:32 // (op 1339) Store value
buffer_store_dwordx4 v[20:23], v0, s[56:59], 0 offen offset:64 // (op 1339) Store value
buffer_store_dwordx4 v[24:27], v0, s[56:59], 0 offen offset:96 // (op 1339) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_155:












s_endpgm  // End of RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_, .LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end-RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 216
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 64000 // lds bytes
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
  ProblemType:
    OperationType: GEMM
    DataType: B6
    DataTypeA: B6
    DataTypeB: B6
    DestDataType: s
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
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 64000
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 152
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
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

