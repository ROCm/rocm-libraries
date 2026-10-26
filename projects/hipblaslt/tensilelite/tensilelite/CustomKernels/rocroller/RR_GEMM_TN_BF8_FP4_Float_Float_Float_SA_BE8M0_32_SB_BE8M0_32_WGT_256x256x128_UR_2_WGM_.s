

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_,@function
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 26856) 
s_mul_hi_i32 s1, s16, s10 // (op 26995) 
v_and_b32 v0, 4294967168, v1 // (op 26917) 
v_and_b32 v3, 15, v1 // (op 26917) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 26917) 


v_and_b32 v0, 48, v1 // (op 26917) 
v_add_u32 v3, v4, v0 // (op 26917) 


v_lshrrev_b32 v0, 6, v1 // (op 26974) 
v_and_b32 v4, 1, v0 // (op 26974) 

v_and_b32 v0, 15, v1 // (op 26974) 
v_lshl_add_u32 v5, v4, 7, v0 // (op 26974) 


v_and_b32 v0, 63, v1 // (op 26974) 
v_lshrrev_b32 v4, 4, v0 // (op 26974) 

v_lshlrev_b32 v0, 5, v4 // (op 26974) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 26974) 


v_lshlrev_b32 v0, 2, v4 // (op 26974) 

v_lshrrev_b32 v4, 3, v0 // (op 26974) 

v_and_b32 v0, 4294967168, v1 // (op 27041) 
v_and_b32 v5, 15, v1 // (op 27041) 
v_add_lshl_u32 v6, v0, v5, 7 // (op 27041) 


v_and_b32 v0, 48, v1 // (op 27041) 
v_add_u32 v5, v6, v0 // (op 27041) 


v_lshrrev_b32 v0, 6, v1 // (op 27085) 
v_and_b32 v6, 1, v0 // (op 27085) 

v_and_b32 v0, 15, v1 // (op 27085) 
v_lshl_add_u32 v7, v6, 7, v0 // (op 27085) 


v_and_b32 v0, 63, v1 // (op 27085) 
v_lshrrev_b32 v6, 4, v0 // (op 27085) 

v_lshlrev_b32 v0, 5, v6 // (op 27085) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 27085) 


v_lshlrev_b32 v0, 2, v6 // (op 27085) 

v_lshrrev_b32 v6, 3, v0 // (op 27085) 

v_and_b32 v0, 4294967168, v1 // (op 26856) 
s_add_i32 s18, s0, s16 // (op 26856) 

v_and_b32 v7, 63, v1 // (op 26856) 
s_ashr_i32 s0, s18, 31 // (op 26856) 
v_lshrrev_b32 v8, 4, v7 // (op 26856) 
s_cmp_eq_i32 s10, 0 // (op 26856) 
s_mov_b32 s19, scc // (op 26856) 
v_lshlrev_b32 v9, 2, v8 // (op 26856) 

s_cmp_eq_i32 s19, 0 // (op 26856) 
s_cselect_b32 s49, 0, -1 // (op 26856) 

v_add_u32 v8, v0, v9 // (op 26856) 


s_and_b32 s19, s13, 31 // (op 26856) 
v_lshrrev_b32 v0, 2, v7 // (op 26856) 

v_and_b32 v7, 3, v0 // (op 26856) 

v_and_b32 v0, 4294967168, v1 // (op 26995) 
v_and_b32 v9, 63, v1 // (op 26995) 
s_lshl_b32 s50, 1, s19 // (op 26856) 
v_and_b32 v10, 3, v1 // (op 26856) 
s_add_i32 s51, s50, s49 // (op 26856) 


s_and_b32 s49, s0, s51 // (op 26856) 


s_add_i32 s0, s1, s16 // (op 26995) 

s_ashr_i32 s1, s0, 31 // (op 26995) 
v_lshl_add_u32 v11, v7, 2, v10 // (op 26856) 


s_add_i32 s50, s18, s49 // (op 26856) 


v_lshrrev_b32 v7, 6, v1 // (op 26856) 
s_ashr_i32 s18, s50, s19 // (op 26856) 


v_and_b32 v10, 1, v7 // (op 26856) 

s_lshl_b32 s19, s13, 24 // (op 26856) 
v_lshl_add_u32 v7, v10, 7, v11 // (op 26856) 


s_ashr_i32 s49, s19, 31 // (op 26856) 

s_xor_b32 s19, s18, s49 // (op 26856) 

s_sub_i32 s18, s19, s49 // (op 26856) 


s_mul_hi_i32 s19, s18, s14 // (op 26856) 
s_add_i32 s49, s19, s18 // (op 26856) 

s_ashr_i32 s19, s49, 31 // (op 26856) 
s_cmp_eq_i32 s14, 0 // (op 26856) 
s_mov_b32 s50, scc // (op 26856) 
s_cmp_eq_i32 s50, 0 // (op 26856) 
s_cselect_b32 s51, 0, -1 // (op 26856) 

s_and_b32 s50, s15, 31 // (op 26856) 
v_lshrrev_b32 v10, 4, v9 // (op 26995) 
s_cmp_eq_i32 s10, 0 // (op 26995) 
s_mov_b32 s52, scc // (op 26995) 
v_lshlrev_b32 v11, 2, v10 // (op 26995) 

s_cmp_eq_i32 s52, 0 // (op 26995) 
s_cselect_b32 s53, 0, -1 // (op 26995) 

s_lshl_b32 s52, 1, s50 // (op 26856) 
v_add_u32 v10, v0, v11 // (op 26995) 


v_lshrrev_b32 v0, 2, v9 // (op 26995) 

s_add_i32 s54, s52, s51 // (op 26856) 


s_and_b32 s51, s19, s54 // (op 26856) 


s_add_i32 s19, s49, s51 // (op 26856) 


s_ashr_i32 s49, s19, s50 // (op 26856) 


s_lshl_b32 s19, s15, 24 // (op 26856) 
s_ashr_i32 s50, s19, 31 // (op 26856) 

s_xor_b32 s19, s49, s50 // (op 26856) 

s_sub_i32 s49, s19, s50 // (op 26856) 


s_mul_i32 s19, s18, s11 // (op 26856) 
s_sub_u32 s50, s16, s19 // (op 26856) 

s_mul_i32 s19, s49, s11 // (op 26856) 
s_add_u32 s19, s19, s50 // (op 26856) 

s_mul_hi_i32 s50, s16, s9 // (op 26856) 
s_add_i32 s51, s50, s16 // (op 26856) 

s_ashr_i32 s50, s51, 31 // (op 26856) 
s_cmp_eq_i32 s9, 0 // (op 26856) 
s_mov_b32 s52, scc // (op 26856) 
s_cmp_eq_i32 s52, 0 // (op 26856) 
s_cselect_b32 s54, 0, -1 // (op 26856) 

s_and_b32 s52, s13, 31 // (op 26995) 
v_and_b32 v9, 3, v0 // (op 26995) 

s_lshl_b32 s55, 1, s52 // (op 26995) 
v_and_b32 v0, 3, v1 // (op 26995) 
s_add_i32 s56, s55, s53 // (op 26995) 


s_and_b32 s53, s1, s56 // (op 26995) 


v_lshl_add_u32 v11, v9, 2, v0 // (op 26995) 


s_add_i32 s1, s0, s53 // (op 26995) 


v_lshrrev_b32 v0, 6, v1 // (op 26995) 
s_ashr_i32 s0, s1, s52 // (op 26995) 


v_and_b32 v9, 1, v0 // (op 26995) 

s_lshl_b32 s1, s13, 24 // (op 26995) 
v_lshl_add_u32 v0, v9, 7, v11 // (op 26995) 


s_ashr_i32 s52, s1, 31 // (op 26995) 

s_xor_b32 s1, s0, s52 // (op 26995) 

s_sub_i32 s0, s1, s52 // (op 26995) 


s_mul_hi_i32 s1, s0, s14 // (op 26995) 
s_add_i32 s52, s1, s0 // (op 26995) 

s_ashr_i32 s1, s52, 31 // (op 26995) 
s_cmp_eq_i32 s14, 0 // (op 26995) 
s_mov_b32 s53, scc // (op 26995) 
s_cmp_eq_i32 s53, 0 // (op 26995) 
s_cselect_b32 s55, 0, -1 // (op 26995) 

s_and_b32 s53, s15, 31 // (op 26995) 
s_lshl_b32 s56, 1, s53 // (op 26995) 
s_add_i32 s57, s56, s55 // (op 26995) 


s_and_b32 s55, s1, s57 // (op 26995) 


s_add_i32 s1, s52, s55 // (op 26995) 


s_ashr_i32 s52, s1, s53 // (op 26995) 


s_lshl_b32 s1, s15, 24 // (op 26995) 
s_ashr_i32 s53, s1, 31 // (op 26995) 

s_xor_b32 s1, s52, s53 // (op 26995) 

s_sub_i32 s52, s1, s53 // (op 26995) 


s_mul_i32 s1, s0, s11 // (op 26995) 
s_sub_u32 s53, s16, s1 // (op 26995) 

s_mul_i32 s1, s52, s11 // (op 26995) 
s_add_u32 s1, s1, s53 // (op 26995) 

s_mul_hi_i32 s53, s16, s9 // (op 26995) 
s_add_i32 s55, s53, s16 // (op 26995) 

s_ashr_i32 s53, s55, 31 // (op 26995) 
s_cmp_eq_i32 s9, 0 // (op 26995) 
s_mov_b32 s56, scc // (op 26995) 
s_cmp_eq_i32 s56, 0 // (op 26995) 
s_cselect_b32 s57, 0, -1 // (op 26995) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 26856) 
s_lshl_b32 s58, 1, s56 // (op 26856) 
s_add_i32 s59, s58, s54 // (op 26856) 


s_and_b32 s54, s50, s59 // (op 26856) 


s_add_i32 s50, s51, s54 // (op 26856) 


s_ashr_i32 s51, s50, s56 // (op 26856) 


s_lshl_b32 s50, s45, 24 // (op 26856) 
s_ashr_i32 s54, s50, 31 // (op 26856) 

s_xor_b32 s50, s51, s54 // (op 26856) 

s_sub_i32 s51, s50, s54 // (op 26856) 


s_and_b32 s50, s45, 31 // (op 26995) 
s_lshl_b32 s54, 1, s50 // (op 26995) 
s_add_i32 s56, s54, s57 // (op 26995) 


s_and_b32 s54, s53, s56 // (op 26995) 


s_add_i32 s53, s55, s54 // (op 26995) 


s_ashr_i32 s54, s53, s50 // (op 26995) 


s_lshl_b32 s50, s45, 24 // (op 26995) 
s_ashr_i32 s53, s50, 31 // (op 26995) 

s_xor_b32 s50, s54, s53 // (op 26995) 

s_sub_i32 s54, s50, s53 // (op 26995) 


s_add_u32 s56, s6, 256 // (op 26856) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 26856) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 26856) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 26856) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 26856) 

s_mul_i32 s50, s11, s56 // (op 26856) 
s_mul_i32 s53, s8, s50 // (op 26856) 

s_mul_i32 s50, s51, s53 // (op 26856) 

s_sub_u32 s53, s16, s50 // (op 26856) 

s_mov_b32 s50, s53 // (op 26856) 
s_mul_hi_i32 s55, s50, s12 // (op 26856) 
s_add_i32 s58, s55, s50 // (op 26856) 


s_ashr_i32 s50, s58, 31 // (op 26856) 
s_cmp_eq_i32 s12, 0 // (op 26856) 
s_mov_b32 s55, scc // (op 26856) 
s_cmp_eq_i32 s55, 0 // (op 26856) 
s_cselect_b32 s59, 0, -1 // (op 26856) 

s_and_b32 s55, s44, 31 // (op 26856) 
s_lshl_b32 s60, 1, s55 // (op 26856) 
s_add_i32 s61, s60, s59 // (op 26856) 


s_and_b32 s59, s50, s61 // (op 26856) 


s_add_i32 s50, s58, s59 // (op 26856) 


s_ashr_i32 s58, s50, s55 // (op 26856) 


s_lshl_b32 s50, s44, 24 // (op 26856) 
s_ashr_i32 s55, s50, 31 // (op 26856) 

s_xor_b32 s50, s58, s55 // (op 26856) 

s_sub_i32 s58, s50, s55 // (op 26856) 


s_mul_i32 s50, s46, s11 // (op 26856) 
s_add_u32 s60, s20, 256 // (op 26856) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 26856) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 26856) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 26856) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 26856) 

s_sub_i32 s55, s60, s50 // (op 26856) 

s_mul_i32 s59, s58, s55 // (op 26856) 

s_sub_u32 s55, s53, s59 // (op 26856) 


s_add_u32 s53, s55, s50 // (op 26856) 


s_cmp_eq_i32 s51, 0 // (op 26856) 
s_mov_b32 s50, scc // (op 26856) 

s_cmp_eq_i32 s50, 0 // (op 26856) 
s_cselect_b32 s51, s53, s19 // (op 26856) 


s_mul_i32 s19, s49, s56 // (op 26856) 


v_lshl_add_u32 v9, s51, 8, v8 // (op 26856) 


s_sub_i32 s49, s18, s19 // (op 26856) 


s_cmp_eq_i32 s50, 0 // (op 26856) 
s_cselect_b32 s51, s58, s49 // (op 26856) 



v_lshl_add_u32 v8, s51, 8, v7 // (op 26856) 


v_mul_lo_u32 v7, v8, s2 // (op 26856) 

v_add_lshl_u32 v8, v9, v7, 2 // (op 26856) 


s_add_u32 s18, s6, 256 // (op 26995) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 26995) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 26995) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 26995) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 26995) 

s_mul_i32 s53, s11, s18 // (op 26995) 
s_mul_i32 s49, s8, s53 // (op 26995) 

s_mul_i32 s53, s54, s49 // (op 26995) 

s_sub_u32 s49, s16, s53 // (op 26995) 

s_mov_b32 s53, s49 // (op 26995) 
s_mul_hi_i32 s50, s53, s12 // (op 26995) 
s_add_i32 s51, s50, s53 // (op 26995) 


s_ashr_i32 s50, s51, 31 // (op 26995) 
s_cmp_eq_i32 s12, 0 // (op 26995) 
s_mov_b32 s53, scc // (op 26995) 
s_cmp_eq_i32 s53, 0 // (op 26995) 
s_cselect_b32 s55, 0, -1 // (op 26995) 

s_and_b32 s53, s44, 31 // (op 26995) 
s_lshl_b32 s56, 1, s53 // (op 26995) 
s_add_i32 s57, s56, s55 // (op 26995) 


s_and_b32 s55, s50, s57 // (op 26995) 


s_add_i32 s50, s51, s55 // (op 26995) 


s_ashr_i32 s51, s50, s53 // (op 26995) 


s_lshl_b32 s50, s44, 24 // (op 26995) 
s_ashr_i32 s53, s50, 31 // (op 26995) 

s_xor_b32 s50, s51, s53 // (op 26995) 

s_sub_i32 s51, s50, s53 // (op 26995) 


s_mul_i32 s50, s46, s11 // (op 26995) 
s_add_u32 s56, s20, 256 // (op 26995) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 26995) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 26995) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 26995) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 26995) 

s_sub_i32 s53, s56, s50 // (op 26995) 

s_mul_i32 s55, s51, s53 // (op 26995) 

s_sub_u32 s53, s49, s55 // (op 26995) 


s_add_u32 s49, s53, s50 // (op 26995) 


s_cmp_eq_i32 s54, 0 // (op 26995) 
s_mov_b32 s50, scc // (op 26995) 

s_cmp_eq_i32 s50, 0 // (op 26995) 
s_cselect_b32 s53, s49, s1 // (op 26995) 


s_mul_i32 s1, s52, s18 // (op 26995) 


v_lshl_add_u32 v7, s53, 8, v10 // (op 26995) 


s_sub_i32 s49, s0, s1 // (op 26995) 


s_cmp_eq_i32 s50, 0 // (op 26995) 
s_cselect_b32 s0, s51, s49 // (op 26995) 



v_lshl_add_u32 v9, s0, 8, v0 // (op 26995) 


v_mul_lo_u32 v0, v9, s4 // (op 26995) 

v_add_lshl_u32 v9, v7, v0, 2 // (op 26995) 


s_mul_i32 s1, s2, s7 // (op 26858) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 26858) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 26858) least significant
s_mul_hi_u32 s19, s2, s6 // (op 26858) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 26858) most significant: sum
s_add_u32 s1, s1, s19 // (op 26858) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 26858) 

s_mul_i32 s1, s4, s7 // (op 26997) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 26997) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 26997) least significant
s_mul_hi_u32 s50, s4, s6 // (op 26997) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 26997) most significant: sum
s_add_u32 s1, s1, s50 // (op 26997) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 26997) 

s_mov_b32 s52, s22 // (op 26858) reg expression
s_mov_b32 s53, s23 // (op 26858) reg expression
s_mov_b32 s54, s18 // (op 26858) reg expression
s_mov_b32 s55, 131072 // (op 26858) call()

s_mov_b32 s56, s24 // (op 26997) reg expression

s_mov_b32 s57, s25 // (op 26997) reg expression
s_mov_b32 s58, s50 // (op 26997) reg expression
s_mov_b32 s59, 131072 // (op 26997) call()


v_mov_b32 v0, s47 // (op 2) Move value
v_mov_b32 v7, s48 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_accvgpr_write a4, 0.00000 // (op 277) call()
v_accvgpr_write a5, 0.00000 // (op 277) call()
v_accvgpr_write a6, 0.00000 // (op 277) call()
v_accvgpr_write a7, 0.00000 // (op 277) call()
v_accvgpr_write a8, 0.00000 // (op 413) call()
v_accvgpr_write a9, 0.00000 // (op 413) call()
v_accvgpr_write a10, 0.00000 // (op 413) call()
v_accvgpr_write a11, 0.00000 // (op 413) call()
v_accvgpr_write a12, 0.00000 // (op 549) call()
v_accvgpr_write a13, 0.00000 // (op 549) call()
v_accvgpr_write a14, 0.00000 // (op 549) call()
v_accvgpr_write a15, 0.00000 // (op 549) call()
v_accvgpr_write a16, 0.00000 // (op 685) call()
v_accvgpr_write a17, 0.00000 // (op 685) call()
v_accvgpr_write a18, 0.00000 // (op 685) call()
v_accvgpr_write a19, 0.00000 // (op 685) call()
v_accvgpr_write a20, 0.00000 // (op 821) call()
v_accvgpr_write a21, 0.00000 // (op 821) call()
v_accvgpr_write a22, 0.00000 // (op 821) call()
v_accvgpr_write a23, 0.00000 // (op 821) call()
v_accvgpr_write a24, 0.00000 // (op 957) call()
v_accvgpr_write a25, 0.00000 // (op 957) call()
v_accvgpr_write a26, 0.00000 // (op 957) call()
v_accvgpr_write a27, 0.00000 // (op 957) call()
v_accvgpr_write a28, 0.00000 // (op 1093) call()
v_accvgpr_write a29, 0.00000 // (op 1093) call()
v_accvgpr_write a30, 0.00000 // (op 1093) call()
v_accvgpr_write a31, 0.00000 // (op 1093) call()
v_accvgpr_write a32, 0.00000 // (op 1405) call()
v_accvgpr_write a33, 0.00000 // (op 1405) call()
v_accvgpr_write a34, 0.00000 // (op 1405) call()
v_accvgpr_write a35, 0.00000 // (op 1405) call()
v_accvgpr_write a36, 0.00000 // (op 1926) call()
v_accvgpr_write a37, 0.00000 // (op 1926) call()
v_accvgpr_write a38, 0.00000 // (op 1926) call()
v_accvgpr_write a39, 0.00000 // (op 1926) call()
v_accvgpr_write a40, 0.00000 // (op 1927) call()
v_accvgpr_write a41, 0.00000 // (op 1927) call()
v_accvgpr_write a42, 0.00000 // (op 1927) call()
v_accvgpr_write a43, 0.00000 // (op 1927) call()
v_accvgpr_write a44, 0.00000 // (op 1928) call()
v_accvgpr_write a45, 0.00000 // (op 1928) call()
v_accvgpr_write a46, 0.00000 // (op 1928) call()
v_accvgpr_write a47, 0.00000 // (op 1928) call()
v_accvgpr_write a48, 0.00000 // (op 1929) call()
v_accvgpr_write a49, 0.00000 // (op 1929) call()
v_accvgpr_write a50, 0.00000 // (op 1929) call()
v_accvgpr_write a51, 0.00000 // (op 1929) call()
v_accvgpr_write a52, 0.00000 // (op 1930) call()
v_accvgpr_write a53, 0.00000 // (op 1930) call()
v_accvgpr_write a54, 0.00000 // (op 1930) call()
v_accvgpr_write a55, 0.00000 // (op 1930) call()
v_accvgpr_write a56, 0.00000 // (op 1931) call()
v_accvgpr_write a57, 0.00000 // (op 1931) call()
v_accvgpr_write a58, 0.00000 // (op 1931) call()
v_accvgpr_write a59, 0.00000 // (op 1931) call()
v_accvgpr_write a60, 0.00000 // (op 1932) call()
v_accvgpr_write a61, 0.00000 // (op 1932) call()
v_accvgpr_write a62, 0.00000 // (op 1932) call()
v_accvgpr_write a63, 0.00000 // (op 1932) call()
v_accvgpr_write a64, 0.00000 // (op 2623) call()
v_accvgpr_write a65, 0.00000 // (op 2623) call()
v_accvgpr_write a66, 0.00000 // (op 2623) call()
v_accvgpr_write a67, 0.00000 // (op 2623) call()
v_accvgpr_write a68, 0.00000 // (op 3144) call()
v_accvgpr_write a69, 0.00000 // (op 3144) call()
v_accvgpr_write a70, 0.00000 // (op 3144) call()
v_accvgpr_write a71, 0.00000 // (op 3144) call()
v_accvgpr_write a72, 0.00000 // (op 3145) call()
v_accvgpr_write a73, 0.00000 // (op 3145) call()
v_accvgpr_write a74, 0.00000 // (op 3145) call()
v_accvgpr_write a75, 0.00000 // (op 3145) call()
v_accvgpr_write a76, 0.00000 // (op 3146) call()
v_accvgpr_write a77, 0.00000 // (op 3146) call()
v_accvgpr_write a78, 0.00000 // (op 3146) call()
v_accvgpr_write a79, 0.00000 // (op 3146) call()
v_accvgpr_write a80, 0.00000 // (op 3147) call()
v_accvgpr_write a81, 0.00000 // (op 3147) call()
v_accvgpr_write a82, 0.00000 // (op 3147) call()
v_accvgpr_write a83, 0.00000 // (op 3147) call()
v_accvgpr_write a84, 0.00000 // (op 3148) call()
v_accvgpr_write a85, 0.00000 // (op 3148) call()
v_accvgpr_write a86, 0.00000 // (op 3148) call()
v_accvgpr_write a87, 0.00000 // (op 3148) call()
v_accvgpr_write a88, 0.00000 // (op 3149) call()
v_accvgpr_write a89, 0.00000 // (op 3149) call()
v_accvgpr_write a90, 0.00000 // (op 3149) call()
v_accvgpr_write a91, 0.00000 // (op 3149) call()
v_accvgpr_write a92, 0.00000 // (op 3150) call()
v_accvgpr_write a93, 0.00000 // (op 3150) call()
v_accvgpr_write a94, 0.00000 // (op 3150) call()
v_accvgpr_write a95, 0.00000 // (op 3150) call()
v_accvgpr_write a96, 0.00000 // (op 3841) call()
v_accvgpr_write a97, 0.00000 // (op 3841) call()
v_accvgpr_write a98, 0.00000 // (op 3841) call()
v_accvgpr_write a99, 0.00000 // (op 3841) call()
v_accvgpr_write a100, 0.00000 // (op 4362) call()
v_accvgpr_write a101, 0.00000 // (op 4362) call()
v_accvgpr_write a102, 0.00000 // (op 4362) call()
v_accvgpr_write a103, 0.00000 // (op 4362) call()
v_accvgpr_write a104, 0.00000 // (op 4363) call()
v_accvgpr_write a105, 0.00000 // (op 4363) call()
v_accvgpr_write a106, 0.00000 // (op 4363) call()
v_accvgpr_write a107, 0.00000 // (op 4363) call()
v_accvgpr_write a108, 0.00000 // (op 4364) call()
v_accvgpr_write a109, 0.00000 // (op 4364) call()
v_accvgpr_write a110, 0.00000 // (op 4364) call()
v_accvgpr_write a111, 0.00000 // (op 4364) call()
v_accvgpr_write a112, 0.00000 // (op 4365) call()
v_accvgpr_write a113, 0.00000 // (op 4365) call()
v_accvgpr_write a114, 0.00000 // (op 4365) call()
v_accvgpr_write a115, 0.00000 // (op 4365) call()
v_accvgpr_write a116, 0.00000 // (op 4366) call()
v_accvgpr_write a117, 0.00000 // (op 4366) call()
v_accvgpr_write a118, 0.00000 // (op 4366) call()
v_accvgpr_write a119, 0.00000 // (op 4366) call()
v_accvgpr_write a120, 0.00000 // (op 4367) call()
v_accvgpr_write a121, 0.00000 // (op 4367) call()
v_accvgpr_write a122, 0.00000 // (op 4367) call()
v_accvgpr_write a123, 0.00000 // (op 4367) call()
v_accvgpr_write a124, 0.00000 // (op 4368) call()
v_accvgpr_write a125, 0.00000 // (op 4368) call()
v_accvgpr_write a126, 0.00000 // (op 4368) call()
v_accvgpr_write a127, 0.00000 // (op 4368) call()
v_accvgpr_write a128, 0.00000 // (op 5059) call()
v_accvgpr_write a129, 0.00000 // (op 5059) call()
v_accvgpr_write a130, 0.00000 // (op 5059) call()
v_accvgpr_write a131, 0.00000 // (op 5059) call()
v_accvgpr_write a132, 0.00000 // (op 5580) call()
v_accvgpr_write a133, 0.00000 // (op 5580) call()
v_accvgpr_write a134, 0.00000 // (op 5580) call()
v_accvgpr_write a135, 0.00000 // (op 5580) call()
v_accvgpr_write a136, 0.00000 // (op 5581) call()
v_accvgpr_write a137, 0.00000 // (op 5581) call()
v_accvgpr_write a138, 0.00000 // (op 5581) call()
v_accvgpr_write a139, 0.00000 // (op 5581) call()
v_accvgpr_write a140, 0.00000 // (op 5582) call()
v_accvgpr_write a141, 0.00000 // (op 5582) call()
v_accvgpr_write a142, 0.00000 // (op 5582) call()
v_accvgpr_write a143, 0.00000 // (op 5582) call()
v_accvgpr_write a144, 0.00000 // (op 5583) call()
v_accvgpr_write a145, 0.00000 // (op 5583) call()
v_accvgpr_write a146, 0.00000 // (op 5583) call()
v_accvgpr_write a147, 0.00000 // (op 5583) call()
v_accvgpr_write a148, 0.00000 // (op 5584) call()
v_accvgpr_write a149, 0.00000 // (op 5584) call()
v_accvgpr_write a150, 0.00000 // (op 5584) call()
v_accvgpr_write a151, 0.00000 // (op 5584) call()
v_accvgpr_write a152, 0.00000 // (op 5585) call()
v_accvgpr_write a153, 0.00000 // (op 5585) call()
v_accvgpr_write a154, 0.00000 // (op 5585) call()
v_accvgpr_write a155, 0.00000 // (op 5585) call()
v_accvgpr_write a156, 0.00000 // (op 5586) call()
v_accvgpr_write a157, 0.00000 // (op 5586) call()
v_accvgpr_write a158, 0.00000 // (op 5586) call()
v_accvgpr_write a159, 0.00000 // (op 5586) call()
v_accvgpr_write a160, 0.00000 // (op 6277) call()
v_accvgpr_write a161, 0.00000 // (op 6277) call()
v_accvgpr_write a162, 0.00000 // (op 6277) call()
v_accvgpr_write a163, 0.00000 // (op 6277) call()
v_accvgpr_write a164, 0.00000 // (op 6798) call()
v_accvgpr_write a165, 0.00000 // (op 6798) call()
v_accvgpr_write a166, 0.00000 // (op 6798) call()
v_accvgpr_write a167, 0.00000 // (op 6798) call()
v_accvgpr_write a168, 0.00000 // (op 6799) call()
v_accvgpr_write a169, 0.00000 // (op 6799) call()
v_accvgpr_write a170, 0.00000 // (op 6799) call()
v_accvgpr_write a171, 0.00000 // (op 6799) call()
v_accvgpr_write a172, 0.00000 // (op 6800) call()
v_accvgpr_write a173, 0.00000 // (op 6800) call()
v_accvgpr_write a174, 0.00000 // (op 6800) call()
v_accvgpr_write a175, 0.00000 // (op 6800) call()
v_accvgpr_write a176, 0.00000 // (op 6801) call()
v_accvgpr_write a177, 0.00000 // (op 6801) call()
v_accvgpr_write a178, 0.00000 // (op 6801) call()
v_accvgpr_write a179, 0.00000 // (op 6801) call()
v_accvgpr_write a180, 0.00000 // (op 6802) call()
v_accvgpr_write a181, 0.00000 // (op 6802) call()
v_accvgpr_write a182, 0.00000 // (op 6802) call()
v_accvgpr_write a183, 0.00000 // (op 6802) call()
v_accvgpr_write a184, 0.00000 // (op 6803) call()
v_accvgpr_write a185, 0.00000 // (op 6803) call()
v_accvgpr_write a186, 0.00000 // (op 6803) call()
v_accvgpr_write a187, 0.00000 // (op 6803) call()
v_accvgpr_write a188, 0.00000 // (op 6804) call()
v_accvgpr_write a189, 0.00000 // (op 6804) call()
v_accvgpr_write a190, 0.00000 // (op 6804) call()
v_accvgpr_write a191, 0.00000 // (op 6804) call()
v_accvgpr_write a192, 0.00000 // (op 7495) call()
v_accvgpr_write a193, 0.00000 // (op 7495) call()
v_accvgpr_write a194, 0.00000 // (op 7495) call()
v_accvgpr_write a195, 0.00000 // (op 7495) call()
v_accvgpr_write a196, 0.00000 // (op 8016) call()
v_accvgpr_write a197, 0.00000 // (op 8016) call()
v_accvgpr_write a198, 0.00000 // (op 8016) call()
v_accvgpr_write a199, 0.00000 // (op 8016) call()
v_accvgpr_write a200, 0.00000 // (op 8017) call()
v_accvgpr_write a201, 0.00000 // (op 8017) call()
v_accvgpr_write a202, 0.00000 // (op 8017) call()
v_accvgpr_write a203, 0.00000 // (op 8017) call()
v_accvgpr_write a204, 0.00000 // (op 8018) call()
v_accvgpr_write a205, 0.00000 // (op 8018) call()
v_accvgpr_write a206, 0.00000 // (op 8018) call()
v_accvgpr_write a207, 0.00000 // (op 8018) call()
v_accvgpr_write a208, 0.00000 // (op 8019) call()
v_accvgpr_write a209, 0.00000 // (op 8019) call()
v_accvgpr_write a210, 0.00000 // (op 8019) call()
v_accvgpr_write a211, 0.00000 // (op 8019) call()
v_accvgpr_write a212, 0.00000 // (op 8020) call()
v_accvgpr_write a213, 0.00000 // (op 8020) call()
v_accvgpr_write a214, 0.00000 // (op 8020) call()
v_accvgpr_write a215, 0.00000 // (op 8020) call()
v_accvgpr_write a216, 0.00000 // (op 8021) call()
v_accvgpr_write a217, 0.00000 // (op 8021) call()
v_accvgpr_write a218, 0.00000 // (op 8021) call()
v_accvgpr_write a219, 0.00000 // (op 8021) call()
v_accvgpr_write a220, 0.00000 // (op 8022) call()
v_accvgpr_write a221, 0.00000 // (op 8022) call()
v_accvgpr_write a222, 0.00000 // (op 8022) call()
v_accvgpr_write a223, 0.00000 // (op 8022) call()
v_accvgpr_write a224, 0.00000 // (op 8713) call()
v_accvgpr_write a225, 0.00000 // (op 8713) call()
v_accvgpr_write a226, 0.00000 // (op 8713) call()
v_accvgpr_write a227, 0.00000 // (op 8713) call()
v_accvgpr_write a228, 0.00000 // (op 9234) call()
v_accvgpr_write a229, 0.00000 // (op 9234) call()
v_accvgpr_write a230, 0.00000 // (op 9234) call()
v_accvgpr_write a231, 0.00000 // (op 9234) call()
v_accvgpr_write a232, 0.00000 // (op 9235) call()
v_accvgpr_write a233, 0.00000 // (op 9235) call()
v_accvgpr_write a234, 0.00000 // (op 9235) call()
v_accvgpr_write a235, 0.00000 // (op 9235) call()
v_accvgpr_write a236, 0.00000 // (op 9236) call()
v_accvgpr_write a237, 0.00000 // (op 9236) call()
v_accvgpr_write a238, 0.00000 // (op 9236) call()
v_accvgpr_write a239, 0.00000 // (op 9236) call()
v_accvgpr_write a240, 0.00000 // (op 9237) call()
v_accvgpr_write a241, 0.00000 // (op 9237) call()
v_accvgpr_write a242, 0.00000 // (op 9237) call()
v_accvgpr_write a243, 0.00000 // (op 9237) call()
v_accvgpr_write a244, 0.00000 // (op 9238) call()
v_accvgpr_write a245, 0.00000 // (op 9238) call()
v_accvgpr_write a246, 0.00000 // (op 9238) call()
v_accvgpr_write a247, 0.00000 // (op 9238) call()
v_accvgpr_write a248, 0.00000 // (op 9239) call()
v_accvgpr_write a249, 0.00000 // (op 9239) call()
v_accvgpr_write a250, 0.00000 // (op 9239) call()
v_accvgpr_write a251, 0.00000 // (op 9239) call()
v_accvgpr_write a252, 0.00000 // (op 9240) call()
v_accvgpr_write a253, 0.00000 // (op 9240) call()
v_accvgpr_write a254, 0.00000 // (op 9240) call()
v_accvgpr_write a255, 0.00000 // (op 9240) call()
s_mul_hi_i32 s0, s16, s10 // (op 26746) 
v_lshrrev_b32 v10, 3, v1 // (op 26746) 
s_add_i32 s1, s0, s16 // (op 26746) 

v_and_b32 v11, -8, v1 // (op 26746) 
s_ashr_i32 s0, s1, 31 // (op 26746) 
v_sub_u32 v12, v1, v11 // (op 26746) 

s_cmp_eq_i32 s10, 0 // (op 26746) 
s_mov_b32 s18, scc // (op 26746) 
v_lshlrev_b32 v11, 4, v12 // (op 26746) 

s_cmp_eq_i32 s18, 0 // (op 26746) 
s_cselect_b32 s19, 0, -1 // (op 26746) 

v_mov_b32 v13, 0 // (op 26746) convert
v_mov_b32 v12, v11 // (op 26746) convert

s_and_b32 s18, s13, 31 // (op 26746) 
s_lshl_b32 s22, 1, s18 // (op 26746) 
s_add_i32 s23, s22, s19 // (op 26746) 


s_and_b32 s19, s0, s23 // (op 26746) 


s_add_i32 s0, s1, s19 // (op 26746) 


s_ashr_i32 s1, s0, s18 // (op 26746) 


s_lshl_b32 s0, s13, 24 // (op 26746) 
s_ashr_i32 s18, s0, 31 // (op 26746) 

s_xor_b32 s0, s1, s18 // (op 26746) 

s_sub_i32 s1, s0, s18 // (op 26746) 


s_mul_hi_i32 s0, s1, s14 // (op 26746) 
s_add_i32 s18, s0, s1 // (op 26746) 

s_ashr_i32 s0, s18, 31 // (op 26746) 
s_cmp_eq_i32 s14, 0 // (op 26746) 
s_mov_b32 s19, scc // (op 26746) 
s_cmp_eq_i32 s19, 0 // (op 26746) 
s_cselect_b32 s22, 0, -1 // (op 26746) 

s_and_b32 s19, s15, 31 // (op 26746) 
s_lshl_b32 s23, 1, s19 // (op 26746) 
s_add_i32 s24, s23, s22 // (op 26746) 


s_and_b32 s25, s0, s24 // (op 26746) 


s_add_i32 s0, s18, s25 // (op 26746) 


s_ashr_i32 s18, s0, s19 // (op 26746) 


s_lshl_b32 s0, s15, 24 // (op 26746) 
s_ashr_i32 s19, s0, 31 // (op 26746) 

s_xor_b32 s0, s18, s19 // (op 26746) 

s_sub_i32 s18, s0, s19 // (op 26746) 


s_mul_i32 s0, s1, s11 // (op 26746) 

s_sub_u32 s1, s16, s0 // (op 26746) 

s_mul_i32 s0, s18, s11 // (op 26746) 
s_add_u32 s0, s0, s1 // (op 26746) 


s_mul_hi_i32 s1, s16, s9 // (op 26746) 
s_add_i32 s18, s1, s16 // (op 26746) 

s_ashr_i32 s1, s18, 31 // (op 26746) 
s_cmp_eq_i32 s9, 0 // (op 26746) 
s_mov_b32 s19, scc // (op 26746) 
s_cmp_eq_i32 s19, 0 // (op 26746) 
s_cselect_b32 s22, 0, -1 // (op 26746) 

s_and_b32 s19, s45, 31 // (op 26746) 
s_lshl_b32 s23, 1, s19 // (op 26746) 
s_add_i32 s24, s23, s22 // (op 26746) 


s_and_b32 s25, s1, s24 // (op 26746) 


s_add_i32 s1, s18, s25 // (op 26746) 


s_ashr_i32 s18, s1, s19 // (op 26746) 


s_lshl_b32 s1, s45, 24 // (op 26746) 
s_ashr_i32 s19, s1, 31 // (op 26746) 

s_xor_b32 s1, s18, s19 // (op 26746) 

s_sub_i32 s18, s1, s19 // (op 26746) 


s_add_u32 s22, s6, 256 // (op 26746) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 26746) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 26746) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 26746) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 26746) 

s_mul_i32 s1, s11, s22 // (op 26746) 

s_mul_i32 s19, s8, s1 // (op 26746) 

s_mul_i32 s1, s18, s19 // (op 26746) 

s_sub_u32 s19, s16, s1 // (op 26746) 

s_mov_b32 s1, s19 // (op 26746) 
s_mul_hi_i32 s22, s1, s12 // (op 26746) 
s_add_i32 s23, s22, s1 // (op 26746) 


s_ashr_i32 s1, s23, 31 // (op 26746) 
s_cmp_eq_i32 s12, 0 // (op 26746) 
s_mov_b32 s22, scc // (op 26746) 
s_cmp_eq_i32 s22, 0 // (op 26746) 
s_cselect_b32 s24, 0, -1 // (op 26746) 

s_and_b32 s22, s44, 31 // (op 26746) 
s_lshl_b32 s25, 1, s22 // (op 26746) 
s_add_i32 s47, s25, s24 // (op 26746) 


s_and_b32 s24, s1, s47 // (op 26746) 


s_add_i32 s1, s23, s24 // (op 26746) 


s_ashr_i32 s23, s1, s22 // (op 26746) 


s_lshl_b32 s1, s44, 24 // (op 26746) 
s_ashr_i32 s22, s1, 31 // (op 26746) 

s_xor_b32 s1, s23, s22 // (op 26746) 

s_sub_i32 s23, s1, s22 // (op 26746) 


s_mul_i32 s1, s46, s11 // (op 26746) 
s_add_u32 s24, s20, 256 // (op 26746) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 26746) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 26746) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 26746) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 26746) 

s_sub_i32 s22, s24, s1 // (op 26746) 

s_mul_i32 s24, s23, s22 // (op 26746) 


s_sub_u32 s25, s19, s24 // (op 26746) 


s_add_u32 s19, s25, s1 // (op 26746) 


s_cmp_eq_i32 s18, 0 // (op 26746) 
s_mov_b32 s1, scc // (op 26746) 

s_cmp_eq_i32 s1, 0 // (op 26746) 
s_cselect_b32 s18, s19, s0 // (op 26746) 



v_lshl_add_u32 v11, s18, 8, v10 // (op 26746) 


v_mul_lo_u32 v15, v11, s27 // (op 26746) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 26746) most significant: high of low * low
v_mul_lo_u32 v14, v11, s26 // (op 26746) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 26746) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 26746) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 26746) most significant half



s_mul_i32 s1, s26, s21 // (op 26748) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 26748) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 26748) least significant
s_mul_hi_u32 s19, s26, s20 // (op 26748) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 26748) most significant: sum
s_add_u32 s1, s1, s19 // (op 26748) most significant: sum


s_mov_b32 s48, s28 // (op 26748) reg expression
s_mov_b32 s49, s29 // (op 26748) reg expression
s_mov_b32 s50, s0 // (op 26748) reg expression
s_mov_b32 s51, 131072 // (op 26748) call()

s_mul_hi_i32 s47, s16, s10 // (op 26765) 
v_lshrrev_b32 v10, 3, v1 // (op 26765) 
s_add_i32 s0, s47, s16 // (op 26765) 

v_and_b32 v11, -8, v1 // (op 26765) 
s_ashr_i32 s1, s0, 31 // (op 26765) 
v_sub_u32 v12, v1, v11 // (op 26765) 

s_cmp_eq_i32 s10, 0 // (op 26765) 
s_mov_b32 s47, scc // (op 26765) 
v_lshlrev_b32 v11, 4, v12 // (op 26765) 

s_cmp_eq_i32 s47, 0 // (op 26765) 
s_cselect_b32 s18, 0, -1 // (op 26765) 

v_mov_b32 v13, 0 // (op 26765) convert
v_mov_b32 v12, v11 // (op 26765) convert

s_and_b32 s19, s13, 31 // (op 26765) 
s_lshl_b32 s47, 1, s19 // (op 26765) 
s_add_i32 s22, s47, s18 // (op 26765) 


s_and_b32 s18, s1, s22 // (op 26765) 


s_add_i32 s1, s0, s18 // (op 26765) 


s_ashr_i32 s0, s1, s19 // (op 26765) 


s_lshl_b32 s1, s13, 24 // (op 26765) 
s_ashr_i32 s47, s1, 31 // (op 26765) 

s_xor_b32 s1, s0, s47 // (op 26765) 

s_sub_i32 s0, s1, s47 // (op 26765) 


s_mul_hi_i32 s1, s0, s14 // (op 26765) 
s_add_i32 s47, s1, s0 // (op 26765) 

s_ashr_i32 s1, s47, 31 // (op 26765) 
s_cmp_eq_i32 s14, 0 // (op 26765) 
s_mov_b32 s18, scc // (op 26765) 
s_cmp_eq_i32 s18, 0 // (op 26765) 
s_cselect_b32 s19, 0, -1 // (op 26765) 

s_and_b32 s18, s15, 31 // (op 26765) 
s_lshl_b32 s22, 1, s18 // (op 26765) 
s_add_i32 s23, s22, s19 // (op 26765) 


s_and_b32 s19, s1, s23 // (op 26765) 


s_add_i32 s1, s47, s19 // (op 26765) 


s_ashr_i32 s19, s1, s18 // (op 26765) 


s_lshl_b32 s1, s15, 24 // (op 26765) 
s_ashr_i32 s18, s1, 31 // (op 26765) 

s_xor_b32 s1, s19, s18 // (op 26765) 

s_sub_i32 s19, s1, s18 // (op 26765) 


s_mul_i32 s1, s0, s11 // (op 26765) 

s_sub_u32 s0, s16, s1 // (op 26765) 

s_mul_i32 s1, s19, s11 // (op 26765) 
s_add_u32 s1, s1, s0 // (op 26765) 


s_mul_hi_i32 s0, s16, s9 // (op 26765) 
s_add_i32 s47, s0, s16 // (op 26765) 

s_ashr_i32 s0, s47, 31 // (op 26765) 
s_cmp_eq_i32 s9, 0 // (op 26765) 
s_mov_b32 s18, scc // (op 26765) 
s_cmp_eq_i32 s18, 0 // (op 26765) 
s_cselect_b32 s19, 0, -1 // (op 26765) 

s_and_b32 s18, s45, 31 // (op 26765) 
s_lshl_b32 s22, 1, s18 // (op 26765) 
s_add_i32 s23, s22, s19 // (op 26765) 


s_and_b32 s19, s0, s23 // (op 26765) 


s_add_i32 s0, s47, s19 // (op 26765) 


s_ashr_i32 s19, s0, s18 // (op 26765) 


s_lshl_b32 s0, s45, 24 // (op 26765) 
s_ashr_i32 s18, s0, 31 // (op 26765) 

s_xor_b32 s0, s19, s18 // (op 26765) 

s_sub_i32 s19, s0, s18 // (op 26765) 


s_add_u32 s22, s6, 256 // (op 26765) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 26765) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 26765) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 26765) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 26765) 

s_mul_i32 s0, s11, s22 // (op 26765) 

s_mul_i32 s18, s8, s0 // (op 26765) 

s_mul_i32 s0, s19, s18 // (op 26765) 

s_sub_u32 s18, s16, s0 // (op 26765) 

s_mov_b32 s0, s18 // (op 26765) 
s_mul_hi_i32 s47, s0, s12 // (op 26765) 
s_add_i32 s22, s47, s0 // (op 26765) 


s_ashr_i32 s0, s22, 31 // (op 26765) 
s_cmp_eq_i32 s12, 0 // (op 26765) 
s_mov_b32 s47, scc // (op 26765) 
s_cmp_eq_i32 s47, 0 // (op 26765) 
s_cselect_b32 s23, 0, -1 // (op 26765) 

s_and_b32 s47, s44, 31 // (op 26765) 
s_lshl_b32 s24, 1, s47 // (op 26765) 
s_add_i32 s25, s24, s23 // (op 26765) 


s_and_b32 s23, s0, s25 // (op 26765) 


s_add_i32 s0, s22, s23 // (op 26765) 


s_ashr_i32 s22, s0, s47 // (op 26765) 


s_lshl_b32 s0, s44, 24 // (op 26765) 
s_ashr_i32 s47, s0, 31 // (op 26765) 

s_xor_b32 s0, s22, s47 // (op 26765) 

s_sub_i32 s22, s0, s47 // (op 26765) 


s_mul_i32 s0, s46, s11 // (op 26765) 
s_add_u32 s24, s20, 256 // (op 26765) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 26765) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 26765) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 26765) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 26765) 

s_sub_i32 s23, s24, s0 // (op 26765) 

s_mul_i32 s47, s22, s23 // (op 26765) 


s_sub_u32 s22, s18, s47 // (op 26765) 


s_add_u32 s18, s22, s0 // (op 26765) 


s_cmp_eq_i32 s19, 0 // (op 26765) 
s_mov_b32 s0, scc // (op 26765) 

s_cmp_eq_i32 s0, 0 // (op 26765) 
s_cselect_b32 s19, s18, s1 // (op 26765) 



v_lshl_add_u32 v11, s19, 8, v10 // (op 26765) 


v_mul_lo_u32 v17, v11, s27 // (op 26765) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 26765) most significant: high of low * low
v_mul_lo_u32 v16, v11, s26 // (op 26765) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 26765) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 26765) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 26765) most significant half



s_mul_i32 s1, s26, s21 // (op 26767) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 26767) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 26767) least significant
s_mul_hi_u32 s18, s26, s20 // (op 26767) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 26767) most significant: sum
s_add_u32 s1, s1, s18 // (op 26767) most significant: sum


s_mov_b32 s48, s28 // (op 26767) reg expression
s_mov_b32 s49, s29 // (op 26767) reg expression
s_mov_b32 s50, s0 // (op 26767) reg expression
s_mov_b32 s51, 131072 // (op 26767) call()

s_mul_hi_i32 s47, s16, s10 // (op 26810) 
v_lshrrev_b32 v10, 2, v1 // (op 26810) 
s_add_i32 s0, s47, s16 // (op 26810) 

v_and_b32 v11, -4, v1 // (op 26810) 
s_ashr_i32 s1, s0, 31 // (op 26810) 
v_sub_u32 v12, v1, v11 // (op 26810) 

s_cmp_eq_i32 s10, 0 // (op 26810) 
s_mov_b32 s47, scc // (op 26810) 
v_lshlrev_b32 v11, 5, v12 // (op 26810) 

s_cmp_eq_i32 s47, 0 // (op 26810) 
s_cselect_b32 s18, 0, -1 // (op 26810) 

v_mov_b32 v13, 0 // (op 26810) convert
v_mov_b32 v12, v11 // (op 26810) convert

s_and_b32 s19, s13, 31 // (op 26810) 
s_lshl_b32 s47, 1, s19 // (op 26810) 
s_add_i32 s22, s47, s18 // (op 26810) 


s_and_b32 s18, s1, s22 // (op 26810) 


s_add_i32 s1, s0, s18 // (op 26810) 


s_ashr_i32 s0, s1, s19 // (op 26810) 


s_lshl_b32 s1, s13, 24 // (op 26810) 
s_ashr_i32 s47, s1, 31 // (op 26810) 

s_xor_b32 s1, s0, s47 // (op 26810) 

s_sub_i32 s0, s1, s47 // (op 26810) 


s_mul_hi_i32 s1, s0, s14 // (op 26810) 
s_add_i32 s47, s1, s0 // (op 26810) 

s_ashr_i32 s1, s47, 31 // (op 26810) 
s_cmp_eq_i32 s14, 0 // (op 26810) 
s_mov_b32 s18, scc // (op 26810) 
s_cmp_eq_i32 s18, 0 // (op 26810) 
s_cselect_b32 s19, 0, -1 // (op 26810) 

s_and_b32 s18, s15, 31 // (op 26810) 
s_lshl_b32 s22, 1, s18 // (op 26810) 
s_add_i32 s23, s22, s19 // (op 26810) 


s_and_b32 s19, s1, s23 // (op 26810) 


s_add_i32 s1, s47, s19 // (op 26810) 


s_ashr_i32 s19, s1, s18 // (op 26810) 


s_lshl_b32 s1, s15, 24 // (op 26810) 
s_ashr_i32 s18, s1, 31 // (op 26810) 

s_xor_b32 s1, s19, s18 // (op 26810) 

s_sub_i32 s19, s1, s18 // (op 26810) 


s_add_u32 s22, s6, 256 // (op 26810) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 26810) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 26810) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 26810) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 26810) 

s_mul_i32 s1, s19, s22 // (op 26810) 

s_sub_i32 s47, s0, s1 // (op 26810) 


s_mul_hi_i32 s0, s16, s9 // (op 26810) 
s_add_i32 s1, s0, s16 // (op 26810) 

s_ashr_i32 s0, s1, 31 // (op 26810) 
s_cmp_eq_i32 s9, 0 // (op 26810) 
s_mov_b32 s18, scc // (op 26810) 
s_cmp_eq_i32 s18, 0 // (op 26810) 
s_cselect_b32 s19, 0, -1 // (op 26810) 

s_and_b32 s18, s45, 31 // (op 26810) 
s_lshl_b32 s24, 1, s18 // (op 26810) 
s_add_i32 s25, s24, s19 // (op 26810) 


s_and_b32 s19, s0, s25 // (op 26810) 


s_add_i32 s0, s1, s19 // (op 26810) 


s_ashr_i32 s1, s0, s18 // (op 26810) 


s_lshl_b32 s0, s45, 24 // (op 26810) 
s_ashr_i32 s18, s0, 31 // (op 26810) 

s_xor_b32 s0, s1, s18 // (op 26810) 

s_sub_i32 s1, s0, s18 // (op 26810) 


s_mul_i32 s0, s11, s22 // (op 26810) 

s_mul_i32 s18, s8, s0 // (op 26810) 

s_mul_i32 s0, s1, s18 // (op 26810) 

s_sub_u32 s18, s16, s0 // (op 26810) 

s_mul_hi_i32 s0, s18, s12 // (op 26810) 
s_add_i32 s19, s0, s18 // (op 26810) 


s_ashr_i32 s0, s19, 31 // (op 26810) 
s_cmp_eq_i32 s12, 0 // (op 26810) 
s_mov_b32 s18, scc // (op 26810) 
s_cmp_eq_i32 s18, 0 // (op 26810) 
s_cselect_b32 s22, 0, -1 // (op 26810) 

s_and_b32 s18, s44, 31 // (op 26810) 
s_lshl_b32 s23, 1, s18 // (op 26810) 
s_add_i32 s24, s23, s22 // (op 26810) 


s_and_b32 s25, s0, s24 // (op 26810) 


s_add_i32 s0, s19, s25 // (op 26810) 


s_ashr_i32 s19, s0, s18 // (op 26810) 


s_lshl_b32 s0, s44, 24 // (op 26810) 
s_ashr_i32 s18, s0, 31 // (op 26810) 

s_xor_b32 s0, s19, s18 // (op 26810) 

s_sub_i32 s19, s0, s18 // (op 26810) 


s_cmp_eq_i32 s1, 0 // (op 26810) 
s_mov_b32 s0, scc // (op 26810) 

s_cmp_eq_i32 s0, 0 // (op 26810) 
s_cselect_b32 s1, s19, s47 // (op 26810) 



v_lshl_add_u32 v11, s1, 8, v10 // (op 26810) 


v_mul_lo_u32 v19, v11, s31 // (op 26810) most significant: low of low * high
v_mul_hi_u32 v10, v11, s30 // (op 26810) most significant: high of low * low
v_mul_lo_u32 v18, v11, s30 // (op 26810) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 26810) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 26810) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 26810) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 26810) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 26810) 

s_mul_i32 s1, s30, s7 // (op 26812) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 26812) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 26812) least significant
s_mul_hi_u32 s18, s30, s6 // (op 26812) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 26812) most significant: sum
s_add_u32 s1, s1, s18 // (op 26812) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 26812) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 26812) 

s_mov_b32 s60, s32 // (op 26812) reg expression
s_mov_b32 s61, s33 // (op 26812) reg expression
s_mov_b32 s62, s0 // (op 26812) reg expression
s_mov_b32 s63, 131072 // (op 26812) call()

s_mul_hi_i32 s47, s16, s10 // (op 26829) 
v_lshrrev_b32 v10, 2, v1 // (op 26829) 
s_add_i32 s0, s47, s16 // (op 26829) 

v_and_b32 v11, -4, v1 // (op 26829) 
s_ashr_i32 s1, s0, 31 // (op 26829) 
v_sub_u32 v18, v1, v11 // (op 26829) 

s_cmp_eq_i32 s10, 0 // (op 26829) 
s_mov_b32 s47, scc // (op 26829) 
v_lshlrev_b32 v11, 5, v18 // (op 26829) 

s_cmp_eq_i32 s47, 0 // (op 26829) 
s_cselect_b32 s18, 0, -1 // (op 26829) 

v_mov_b32 v19, 0 // (op 26829) convert
v_mov_b32 v18, v11 // (op 26829) convert

s_and_b32 s19, s13, 31 // (op 26829) 
s_lshl_b32 s47, 1, s19 // (op 26829) 
s_add_i32 s22, s47, s18 // (op 26829) 


s_and_b32 s18, s1, s22 // (op 26829) 


s_add_i32 s1, s0, s18 // (op 26829) 


s_ashr_i32 s0, s1, s19 // (op 26829) 


s_lshl_b32 s1, s13, 24 // (op 26829) 
s_ashr_i32 s47, s1, 31 // (op 26829) 

s_xor_b32 s1, s0, s47 // (op 26829) 

s_sub_i32 s0, s1, s47 // (op 26829) 


s_mul_hi_i32 s1, s0, s14 // (op 26829) 
s_add_i32 s47, s1, s0 // (op 26829) 

s_ashr_i32 s1, s47, 31 // (op 26829) 
s_cmp_eq_i32 s14, 0 // (op 26829) 
s_mov_b32 s18, scc // (op 26829) 
s_cmp_eq_i32 s18, 0 // (op 26829) 
s_cselect_b32 s19, 0, -1 // (op 26829) 

s_and_b32 s18, s15, 31 // (op 26829) 
s_lshl_b32 s22, 1, s18 // (op 26829) 
s_add_i32 s23, s22, s19 // (op 26829) 


s_and_b32 s19, s1, s23 // (op 26829) 


s_add_i32 s1, s47, s19 // (op 26829) 


s_ashr_i32 s19, s1, s18 // (op 26829) 


s_lshl_b32 s1, s15, 24 // (op 26829) 
s_ashr_i32 s18, s1, 31 // (op 26829) 

s_xor_b32 s1, s19, s18 // (op 26829) 

s_sub_i32 s19, s1, s18 // (op 26829) 


s_add_u32 s22, s6, 256 // (op 26829) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 26829) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 26829) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 26829) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 26829) 

s_mul_i32 s1, s19, s22 // (op 26829) 

s_sub_i32 s47, s0, s1 // (op 26829) 


s_mul_hi_i32 s0, s16, s9 // (op 26829) 
s_add_i32 s1, s0, s16 // (op 26829) 

s_ashr_i32 s0, s1, 31 // (op 26829) 
s_cmp_eq_i32 s9, 0 // (op 26829) 
s_mov_b32 s18, scc // (op 26829) 
s_cmp_eq_i32 s18, 0 // (op 26829) 
s_cselect_b32 s19, 0, -1 // (op 26829) 

s_and_b32 s18, s45, 31 // (op 26829) 
s_lshl_b32 s24, 1, s18 // (op 26829) 
s_add_i32 s25, s24, s19 // (op 26829) 


s_and_b32 s19, s0, s25 // (op 26829) 


s_add_i32 s0, s1, s19 // (op 26829) 


s_ashr_i32 s1, s0, s18 // (op 26829) 


s_lshl_b32 s0, s45, 24 // (op 26829) 
s_ashr_i32 s18, s0, 31 // (op 26829) 

s_xor_b32 s0, s1, s18 // (op 26829) 

s_sub_i32 s1, s0, s18 // (op 26829) 


s_mul_i32 s0, s11, s22 // (op 26829) 

s_mul_i32 s18, s8, s0 // (op 26829) 

s_mul_i32 s0, s1, s18 // (op 26829) 

s_sub_u32 s18, s16, s0 // (op 26829) 

s_mul_hi_i32 s0, s18, s12 // (op 26829) 
s_add_i32 s19, s0, s18 // (op 26829) 


s_ashr_i32 s0, s19, 31 // (op 26829) 
s_cmp_eq_i32 s12, 0 // (op 26829) 
s_mov_b32 s18, scc // (op 26829) 
s_cmp_eq_i32 s18, 0 // (op 26829) 
s_cselect_b32 s22, 0, -1 // (op 26829) 

s_and_b32 s18, s44, 31 // (op 26829) 
s_lshl_b32 s23, 1, s18 // (op 26829) 
s_add_i32 s24, s23, s22 // (op 26829) 


s_and_b32 s25, s0, s24 // (op 26829) 


s_add_i32 s0, s19, s25 // (op 26829) 


s_ashr_i32 s19, s0, s18 // (op 26829) 


s_lshl_b32 s0, s44, 24 // (op 26829) 
s_ashr_i32 s18, s0, 31 // (op 26829) 

s_xor_b32 s0, s19, s18 // (op 26829) 

s_sub_i32 s19, s0, s18 // (op 26829) 


s_cmp_eq_i32 s1, 0 // (op 26829) 
s_mov_b32 s0, scc // (op 26829) 

s_cmp_eq_i32 s0, 0 // (op 26829) 
s_cselect_b32 s1, s19, s47 // (op 26829) 



v_lshl_add_u32 v11, s1, 8, v10 // (op 26829) 


v_mul_lo_u32 v21, v11, s31 // (op 26829) most significant: low of low * high
v_mul_hi_u32 v10, v11, s30 // (op 26829) most significant: high of low * low
v_mul_lo_u32 v20, v11, s30 // (op 26829) least significant: low of low * low
v_add_u32 v21, v21, v10 // (op 26829) most significant: sum


v_add_co_u32 v10, s[0:1], v18, v20 // (op 26829) least significant half
v_addc_co_u32 v11, s[0:1], v19, v21, s[0:1] // (op 26829) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 26829) 


v_lshrrev_b64 v[18:19], 3, v[10:11] // (op 26829) 

s_mul_i32 s1, s30, s7 // (op 26831) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 26831) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 26831) least significant
s_mul_hi_u32 s18, s30, s6 // (op 26831) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 26831) most significant: sum
s_add_u32 s1, s1, s18 // (op 26831) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 26831) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 26831) 

s_mov_b32 s60, s32 // (op 26831) reg expression
s_mov_b32 s61, s33 // (op 26831) reg expression
s_mov_b32 s62, s0 // (op 26831) reg expression
s_mov_b32 s63, 131072 // (op 26831) call()

s_mul_hi_i32 s47, s16, s10 // (op 27107) 
v_and_b32 v10, 63, v1 // (op 27107) 
s_add_i32 s0, s47, s16 // (op 27107) 

v_lshrrev_b32 v11, 4, v10 // (op 27107) 
s_ashr_i32 s1, s0, 31 // (op 27107) 
v_and_b32 v20, 3, v11 // (op 27107) 

s_cmp_eq_i32 s10, 0 // (op 27107) 
s_mov_b32 s47, scc // (op 27107) 
v_and_b32 v11, 15, v1 // (op 27107) 
s_cmp_eq_i32 s47, 0 // (op 27107) 
s_cselect_b32 s18, 0, -1 // (op 27107) 

v_lshl_add_u32 v21, v20, 4, v11 // (op 27107) 


s_and_b32 s19, s13, 31 // (op 27107) 
v_and_b32 v11, 4294967168, v1 // (op 27107) 
s_lshl_b32 s47, 1, s19 // (op 27107) 
v_add_u32 v20, v11, v21 // (op 27107) 


s_add_i32 s22, s47, s18 // (op 27107) 


v_lshrrev_b32 v11, 6, v10 // (op 27107) 

s_and_b32 s18, s1, s22 // (op 27107) 


v_lshlrev_b32 v10, 2, v11 // (op 27107) 

s_add_i32 s1, s0, s18 // (op 27107) 


s_ashr_i32 s0, s1, s19 // (op 27107) 


s_lshl_b32 s1, s13, 24 // (op 27107) 
s_ashr_i32 s47, s1, 31 // (op 27107) 

s_xor_b32 s1, s0, s47 // (op 27107) 

s_sub_i32 s0, s1, s47 // (op 27107) 


s_mul_hi_i32 s1, s0, s14 // (op 27107) 
s_add_i32 s47, s1, s0 // (op 27107) 

s_ashr_i32 s1, s47, 31 // (op 27107) 
s_cmp_eq_i32 s14, 0 // (op 27107) 
s_mov_b32 s18, scc // (op 27107) 
s_cmp_eq_i32 s18, 0 // (op 27107) 
s_cselect_b32 s19, 0, -1 // (op 27107) 

s_and_b32 s18, s15, 31 // (op 27107) 
s_lshl_b32 s22, 1, s18 // (op 27107) 
s_add_i32 s23, s22, s19 // (op 27107) 


s_and_b32 s19, s1, s23 // (op 27107) 


s_add_i32 s1, s47, s19 // (op 27107) 


s_ashr_i32 s19, s1, s18 // (op 27107) 


s_lshl_b32 s1, s15, 24 // (op 27107) 
s_ashr_i32 s18, s1, 31 // (op 27107) 

s_xor_b32 s1, s19, s18 // (op 27107) 

s_sub_i32 s19, s1, s18 // (op 27107) 


s_mul_i32 s1, s0, s11 // (op 27107) 

s_sub_u32 s0, s16, s1 // (op 27107) 

s_mul_i32 s1, s19, s11 // (op 27107) 
s_add_u32 s1, s1, s0 // (op 27107) 


s_mul_hi_i32 s0, s16, s9 // (op 27107) 
s_add_i32 s47, s0, s16 // (op 27107) 

s_ashr_i32 s0, s47, 31 // (op 27107) 
s_cmp_eq_i32 s9, 0 // (op 27107) 
s_mov_b32 s18, scc // (op 27107) 
s_cmp_eq_i32 s18, 0 // (op 27107) 
s_cselect_b32 s19, 0, -1 // (op 27107) 

s_and_b32 s18, s45, 31 // (op 27107) 
s_lshl_b32 s22, 1, s18 // (op 27107) 
s_add_i32 s23, s22, s19 // (op 27107) 


s_and_b32 s19, s0, s23 // (op 27107) 


s_add_i32 s0, s47, s19 // (op 27107) 


s_ashr_i32 s19, s0, s18 // (op 27107) 


s_lshl_b32 s0, s45, 24 // (op 27107) 
s_ashr_i32 s18, s0, 31 // (op 27107) 

s_xor_b32 s0, s19, s18 // (op 27107) 

s_sub_i32 s19, s0, s18 // (op 27107) 


s_add_u32 s22, s6, 256 // (op 27107) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27107) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27107) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27107) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27107) 

s_mul_i32 s0, s11, s22 // (op 27107) 

s_mul_i32 s18, s8, s0 // (op 27107) 

s_mul_i32 s0, s19, s18 // (op 27107) 

s_sub_u32 s18, s16, s0 // (op 27107) 

s_mov_b32 s0, s18 // (op 27107) 
s_mul_hi_i32 s47, s0, s12 // (op 27107) 
s_add_i32 s22, s47, s0 // (op 27107) 


s_ashr_i32 s0, s22, 31 // (op 27107) 
s_cmp_eq_i32 s12, 0 // (op 27107) 
s_mov_b32 s47, scc // (op 27107) 
s_cmp_eq_i32 s47, 0 // (op 27107) 
s_cselect_b32 s23, 0, -1 // (op 27107) 

s_and_b32 s47, s44, 31 // (op 27107) 
s_lshl_b32 s24, 1, s47 // (op 27107) 
s_add_i32 s25, s24, s23 // (op 27107) 


s_and_b32 s23, s0, s25 // (op 27107) 


s_add_i32 s0, s22, s23 // (op 27107) 


s_ashr_i32 s22, s0, s47 // (op 27107) 


s_lshl_b32 s0, s44, 24 // (op 27107) 
s_ashr_i32 s47, s0, 31 // (op 27107) 

s_xor_b32 s0, s22, s47 // (op 27107) 

s_sub_i32 s22, s0, s47 // (op 27107) 


s_mul_i32 s0, s46, s11 // (op 27107) 
s_add_u32 s24, s20, 256 // (op 27107) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 27107) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 27107) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 27107) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 27107) 

s_sub_i32 s23, s24, s0 // (op 27107) 

s_mul_i32 s47, s22, s23 // (op 27107) 


s_sub_u32 s22, s18, s47 // (op 27107) 


s_add_u32 s18, s22, s0 // (op 27107) 


s_cmp_eq_i32 s19, 0 // (op 27107) 
s_mov_b32 s0, scc // (op 27107) 

s_cmp_eq_i32 s0, 0 // (op 27107) 
s_cselect_b32 s19, s18, s1 // (op 27107) 



v_lshl_add_u32 v11, s19, 8, v20 // (op 27107) 


v_mul_lo_u32 v20, v11, s34 // (op 27107) 
v_add_u32 v20, v20, v10 // (op 27107) 


s_mul_i32 s1, s34, s21 // (op 27109) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 27109) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 27109) least significant
s_mul_hi_u32 s18, s34, s20 // (op 27109) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27109) most significant: sum
s_add_u32 s1, s1, s18 // (op 27109) most significant: sum


s_mov_b32 s64, s36 // (op 27109) reg expression
s_mov_b32 s65, s37 // (op 27109) reg expression
s_mov_b32 s66, s0 // (op 27109) reg expression
s_mov_b32 s67, 131072 // (op 27109) call()

s_mul_hi_i32 s47, s16, s10 // (op 27128) 
v_and_b32 v10, 63, v1 // (op 27128) 
s_add_i32 s0, s47, s16 // (op 27128) 

v_lshrrev_b32 v11, 4, v10 // (op 27128) 
s_ashr_i32 s1, s0, 31 // (op 27128) 
v_and_b32 v21, 3, v11 // (op 27128) 

s_cmp_eq_i32 s10, 0 // (op 27128) 
s_mov_b32 s47, scc // (op 27128) 
v_and_b32 v11, 15, v1 // (op 27128) 
s_cmp_eq_i32 s47, 0 // (op 27128) 
s_cselect_b32 s18, 0, -1 // (op 27128) 

v_lshl_add_u32 v22, v21, 4, v11 // (op 27128) 


s_and_b32 s19, s13, 31 // (op 27128) 
v_and_b32 v11, 4294967168, v1 // (op 27128) 
s_lshl_b32 s47, 1, s19 // (op 27128) 
v_add_u32 v21, v11, v22 // (op 27128) 


s_add_i32 s22, s47, s18 // (op 27128) 


v_lshrrev_b32 v11, 6, v10 // (op 27128) 

s_and_b32 s18, s1, s22 // (op 27128) 


v_lshlrev_b32 v10, 2, v11 // (op 27128) 

s_add_i32 s1, s0, s18 // (op 27128) 


s_ashr_i32 s0, s1, s19 // (op 27128) 


s_lshl_b32 s1, s13, 24 // (op 27128) 
s_ashr_i32 s47, s1, 31 // (op 27128) 

s_xor_b32 s1, s0, s47 // (op 27128) 

s_sub_i32 s0, s1, s47 // (op 27128) 


s_mul_hi_i32 s1, s0, s14 // (op 27128) 
s_add_i32 s47, s1, s0 // (op 27128) 

s_ashr_i32 s1, s47, 31 // (op 27128) 
s_cmp_eq_i32 s14, 0 // (op 27128) 
s_mov_b32 s18, scc // (op 27128) 
s_cmp_eq_i32 s18, 0 // (op 27128) 
s_cselect_b32 s19, 0, -1 // (op 27128) 

s_and_b32 s18, s15, 31 // (op 27128) 
s_lshl_b32 s22, 1, s18 // (op 27128) 
s_add_i32 s23, s22, s19 // (op 27128) 


s_and_b32 s19, s1, s23 // (op 27128) 


s_add_i32 s1, s47, s19 // (op 27128) 


s_ashr_i32 s19, s1, s18 // (op 27128) 


s_lshl_b32 s1, s15, 24 // (op 27128) 
s_ashr_i32 s18, s1, 31 // (op 27128) 

s_xor_b32 s1, s19, s18 // (op 27128) 

s_sub_i32 s19, s1, s18 // (op 27128) 


s_mul_i32 s1, s0, s11 // (op 27128) 

s_sub_u32 s0, s16, s1 // (op 27128) 

s_mul_i32 s1, s19, s11 // (op 27128) 
s_add_u32 s1, s1, s0 // (op 27128) 


s_mul_hi_i32 s0, s16, s9 // (op 27128) 
s_add_i32 s47, s0, s16 // (op 27128) 

s_ashr_i32 s0, s47, 31 // (op 27128) 
s_cmp_eq_i32 s9, 0 // (op 27128) 
s_mov_b32 s18, scc // (op 27128) 
s_cmp_eq_i32 s18, 0 // (op 27128) 
s_cselect_b32 s19, 0, -1 // (op 27128) 

s_and_b32 s18, s45, 31 // (op 27128) 
s_lshl_b32 s22, 1, s18 // (op 27128) 
s_add_i32 s23, s22, s19 // (op 27128) 


s_and_b32 s19, s0, s23 // (op 27128) 


s_add_i32 s0, s47, s19 // (op 27128) 


s_ashr_i32 s19, s0, s18 // (op 27128) 


s_lshl_b32 s0, s45, 24 // (op 27128) 
s_ashr_i32 s18, s0, 31 // (op 27128) 

s_xor_b32 s0, s19, s18 // (op 27128) 

s_sub_i32 s19, s0, s18 // (op 27128) 


s_add_u32 s22, s6, 256 // (op 27128) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27128) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27128) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27128) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27128) 

s_mul_i32 s0, s11, s22 // (op 27128) 

s_mul_i32 s18, s8, s0 // (op 27128) 

s_mul_i32 s0, s19, s18 // (op 27128) 

s_sub_u32 s18, s16, s0 // (op 27128) 

s_mov_b32 s0, s18 // (op 27128) 
s_mul_hi_i32 s47, s0, s12 // (op 27128) 
s_add_i32 s22, s47, s0 // (op 27128) 


s_ashr_i32 s0, s22, 31 // (op 27128) 
s_cmp_eq_i32 s12, 0 // (op 27128) 
s_mov_b32 s47, scc // (op 27128) 
s_cmp_eq_i32 s47, 0 // (op 27128) 
s_cselect_b32 s23, 0, -1 // (op 27128) 

s_and_b32 s47, s44, 31 // (op 27128) 
s_lshl_b32 s24, 1, s47 // (op 27128) 
s_add_i32 s25, s24, s23 // (op 27128) 


s_and_b32 s23, s0, s25 // (op 27128) 


s_add_i32 s0, s22, s23 // (op 27128) 


s_ashr_i32 s22, s0, s47 // (op 27128) 


s_lshl_b32 s0, s44, 24 // (op 27128) 
s_ashr_i32 s47, s0, 31 // (op 27128) 

s_xor_b32 s0, s22, s47 // (op 27128) 

s_sub_i32 s22, s0, s47 // (op 27128) 


s_mul_i32 s0, s46, s11 // (op 27128) 
s_add_u32 s24, s20, 256 // (op 27128) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 27128) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 27128) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 27128) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 27128) 

s_sub_i32 s23, s24, s0 // (op 27128) 

s_mul_i32 s47, s22, s23 // (op 27128) 


s_sub_u32 s22, s18, s47 // (op 27128) 


s_add_u32 s18, s22, s0 // (op 27128) 


s_cmp_eq_i32 s19, 0 // (op 27128) 
s_mov_b32 s0, scc // (op 27128) 

s_cmp_eq_i32 s0, 0 // (op 27128) 
s_cselect_b32 s19, s18, s1 // (op 27128) 



v_lshl_add_u32 v11, s19, 8, v21 // (op 27128) 


v_mul_lo_u32 v21, v11, s34 // (op 27128) 
v_add_u32 v21, v21, v10 // (op 27128) 


s_mul_i32 s1, s34, s21 // (op 27130) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 27130) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 27130) least significant
s_mul_hi_u32 s18, s34, s20 // (op 27130) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27130) most significant: sum
s_add_u32 s1, s1, s18 // (op 27130) most significant: sum


s_mov_b32 s64, s36 // (op 27130) reg expression
s_mov_b32 s65, s37 // (op 27130) reg expression
s_mov_b32 s66, s0 // (op 27130) reg expression
s_mov_b32 s67, 131072 // (op 27130) call()

s_mul_hi_i32 s47, s16, s10 // (op 27159) 
v_and_b32 v10, 63, v1 // (op 27159) 
s_add_i32 s0, s47, s16 // (op 27159) 

v_lshrrev_b32 v11, 4, v10 // (op 27159) 
s_ashr_i32 s1, s0, 31 // (op 27159) 
v_and_b32 v22, 3, v11 // (op 27159) 

s_cmp_eq_i32 s10, 0 // (op 27159) 
s_mov_b32 s47, scc // (op 27159) 
v_and_b32 v11, 15, v1 // (op 27159) 
s_cmp_eq_i32 s47, 0 // (op 27159) 
s_cselect_b32 s18, 0, -1 // (op 27159) 

v_lshl_add_u32 v23, v22, 4, v11 // (op 27159) 


s_and_b32 s19, s13, 31 // (op 27159) 
v_lshrrev_b32 v11, 6, v1 // (op 27159) 
s_lshl_b32 s47, 1, s19 // (op 27159) 
v_and_b32 v22, 1, v11 // (op 27159) 

s_add_i32 s22, s47, s18 // (op 27159) 


v_lshl_add_u32 v11, v22, 7, v23 // (op 27159) 


s_and_b32 s18, s1, s22 // (op 27159) 


v_lshrrev_b32 v22, 6, v10 // (op 27159) 

s_add_i32 s1, s0, s18 // (op 27159) 


s_ashr_i32 s0, s1, s19 // (op 27159) 


s_lshl_b32 s1, s13, 24 // (op 27159) 
s_ashr_i32 s47, s1, 31 // (op 27159) 

s_xor_b32 s1, s0, s47 // (op 27159) 

s_sub_i32 s0, s1, s47 // (op 27159) 


s_mul_hi_i32 s1, s0, s14 // (op 27159) 
s_add_i32 s47, s1, s0 // (op 27159) 

s_ashr_i32 s1, s47, 31 // (op 27159) 
s_cmp_eq_i32 s14, 0 // (op 27159) 
s_mov_b32 s18, scc // (op 27159) 
s_cmp_eq_i32 s18, 0 // (op 27159) 
s_cselect_b32 s19, 0, -1 // (op 27159) 

s_and_b32 s18, s15, 31 // (op 27159) 
s_lshl_b32 s22, 1, s18 // (op 27159) 
s_add_i32 s23, s22, s19 // (op 27159) 


s_and_b32 s19, s1, s23 // (op 27159) 


s_add_i32 s1, s47, s19 // (op 27159) 


s_ashr_i32 s19, s1, s18 // (op 27159) 


s_lshl_b32 s1, s15, 24 // (op 27159) 
s_ashr_i32 s18, s1, 31 // (op 27159) 

s_xor_b32 s1, s19, s18 // (op 27159) 

s_sub_i32 s19, s1, s18 // (op 27159) 


s_add_u32 s22, s6, 256 // (op 27159) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27159) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27159) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27159) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27159) 

s_mul_i32 s1, s19, s22 // (op 27159) 

s_sub_i32 s47, s0, s1 // (op 27159) 


s_mul_hi_i32 s0, s16, s9 // (op 27159) 
s_add_i32 s1, s0, s16 // (op 27159) 

s_ashr_i32 s0, s1, 31 // (op 27159) 
s_cmp_eq_i32 s9, 0 // (op 27159) 
s_mov_b32 s18, scc // (op 27159) 
s_cmp_eq_i32 s18, 0 // (op 27159) 
s_cselect_b32 s19, 0, -1 // (op 27159) 

s_and_b32 s18, s45, 31 // (op 27159) 
s_lshl_b32 s24, 1, s18 // (op 27159) 
s_add_i32 s25, s24, s19 // (op 27159) 


s_and_b32 s19, s0, s25 // (op 27159) 


s_add_i32 s0, s1, s19 // (op 27159) 


s_ashr_i32 s1, s0, s18 // (op 27159) 


s_lshl_b32 s0, s45, 24 // (op 27159) 
s_ashr_i32 s18, s0, 31 // (op 27159) 

s_xor_b32 s0, s1, s18 // (op 27159) 

s_sub_i32 s1, s0, s18 // (op 27159) 


s_mul_i32 s0, s11, s22 // (op 27159) 

s_mul_i32 s18, s8, s0 // (op 27159) 

s_mul_i32 s0, s1, s18 // (op 27159) 

s_sub_u32 s18, s16, s0 // (op 27159) 

s_mul_hi_i32 s0, s18, s12 // (op 27159) 
s_add_i32 s19, s0, s18 // (op 27159) 


s_ashr_i32 s0, s19, 31 // (op 27159) 
s_cmp_eq_i32 s12, 0 // (op 27159) 
s_mov_b32 s18, scc // (op 27159) 
s_cmp_eq_i32 s18, 0 // (op 27159) 
s_cselect_b32 s22, 0, -1 // (op 27159) 

s_and_b32 s18, s44, 31 // (op 27159) 
s_lshl_b32 s23, 1, s18 // (op 27159) 
s_add_i32 s24, s23, s22 // (op 27159) 


s_and_b32 s25, s0, s24 // (op 27159) 


s_add_i32 s0, s19, s25 // (op 27159) 


s_ashr_i32 s19, s0, s18 // (op 27159) 


s_lshl_b32 s0, s44, 24 // (op 27159) 
s_ashr_i32 s18, s0, 31 // (op 27159) 

s_xor_b32 s0, s19, s18 // (op 27159) 

s_sub_i32 s19, s0, s18 // (op 27159) 


s_cmp_eq_i32 s1, 0 // (op 27159) 
s_mov_b32 s0, scc // (op 27159) 

s_cmp_eq_i32 s0, 0 // (op 27159) 
s_cselect_b32 s1, s19, s47 // (op 27159) 



v_lshl_add_u32 v10, s1, 8, v11 // (op 27159) 


v_mul_lo_u32 v11, v10, s38 // (op 27159) 

v_lshl_add_u32 v10, v22, 2, v11 // (op 27159) 


s_mul_i32 s1, s38, s7 // (op 27161) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 27161) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 27161) least significant
s_mul_hi_u32 s18, s38, s6 // (op 27161) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27161) most significant: sum
s_add_u32 s1, s1, s18 // (op 27161) most significant: sum


s_mov_b32 s68, s40 // (op 27161) reg expression
s_mov_b32 s69, s41 // (op 27161) reg expression
s_mov_b32 s70, s0 // (op 27161) reg expression
s_mov_b32 s71, 131072 // (op 27161) call()

s_mul_hi_i32 s47, s16, s10 // (op 27180) 
v_and_b32 v11, 63, v1 // (op 27180) 
s_add_i32 s0, s47, s16 // (op 27180) 

v_lshrrev_b32 v22, 4, v11 // (op 27180) 
s_ashr_i32 s1, s0, 31 // (op 27180) 
v_and_b32 v23, 3, v22 // (op 27180) 

s_cmp_eq_i32 s10, 0 // (op 27180) 
s_mov_b32 s47, scc // (op 27180) 
v_and_b32 v22, 15, v1 // (op 27180) 
s_cmp_eq_i32 s47, 0 // (op 27180) 
s_cselect_b32 s18, 0, -1 // (op 27180) 

v_lshl_add_u32 v24, v23, 4, v22 // (op 27180) 


s_and_b32 s19, s13, 31 // (op 27180) 
v_lshrrev_b32 v22, 6, v1 // (op 27180) 
s_lshl_b32 s47, 1, s19 // (op 27180) 
v_and_b32 v23, 1, v22 // (op 27180) 

s_add_i32 s22, s47, s18 // (op 27180) 


v_lshl_add_u32 v22, v23, 7, v24 // (op 27180) 


s_and_b32 s18, s1, s22 // (op 27180) 


v_lshrrev_b32 v23, 6, v11 // (op 27180) 

s_add_i32 s1, s0, s18 // (op 27180) 


s_ashr_i32 s0, s1, s19 // (op 27180) 


s_lshl_b32 s1, s13, 24 // (op 27180) 
s_ashr_i32 s47, s1, 31 // (op 27180) 

s_xor_b32 s1, s0, s47 // (op 27180) 

s_sub_i32 s0, s1, s47 // (op 27180) 


s_mul_hi_i32 s1, s0, s14 // (op 27180) 
s_add_i32 s47, s1, s0 // (op 27180) 

s_ashr_i32 s1, s47, 31 // (op 27180) 
s_cmp_eq_i32 s14, 0 // (op 27180) 
s_mov_b32 s18, scc // (op 27180) 
s_cmp_eq_i32 s18, 0 // (op 27180) 
s_cselect_b32 s19, 0, -1 // (op 27180) 

s_and_b32 s18, s15, 31 // (op 27180) 
s_lshl_b32 s22, 1, s18 // (op 27180) 
s_add_i32 s23, s22, s19 // (op 27180) 


s_and_b32 s19, s1, s23 // (op 27180) 


s_add_i32 s1, s47, s19 // (op 27180) 


s_ashr_i32 s19, s1, s18 // (op 27180) 


s_lshl_b32 s1, s15, 24 // (op 27180) 
s_ashr_i32 s18, s1, 31 // (op 27180) 

s_xor_b32 s1, s19, s18 // (op 27180) 

s_sub_i32 s19, s1, s18 // (op 27180) 


s_add_u32 s22, s6, 256 // (op 27180) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27180) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27180) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27180) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27180) 

s_mul_i32 s1, s19, s22 // (op 27180) 

s_sub_i32 s47, s0, s1 // (op 27180) 


s_mul_hi_i32 s0, s16, s9 // (op 27180) 
s_add_i32 s1, s0, s16 // (op 27180) 

s_ashr_i32 s0, s1, 31 // (op 27180) 
s_cmp_eq_i32 s9, 0 // (op 27180) 
s_mov_b32 s18, scc // (op 27180) 
s_cmp_eq_i32 s18, 0 // (op 27180) 
s_cselect_b32 s19, 0, -1 // (op 27180) 

s_and_b32 s18, s45, 31 // (op 27180) 
s_lshl_b32 s24, 1, s18 // (op 27180) 
s_add_i32 s25, s24, s19 // (op 27180) 


s_and_b32 s19, s0, s25 // (op 27180) 


s_add_i32 s0, s1, s19 // (op 27180) 


s_ashr_i32 s1, s0, s18 // (op 27180) 


s_lshl_b32 s0, s45, 24 // (op 27180) 
s_ashr_i32 s18, s0, 31 // (op 27180) 

s_xor_b32 s0, s1, s18 // (op 27180) 

s_sub_i32 s1, s0, s18 // (op 27180) 


s_mul_i32 s0, s11, s22 // (op 27180) 

s_mul_i32 s18, s8, s0 // (op 27180) 

s_mul_i32 s0, s1, s18 // (op 27180) 

s_sub_u32 s18, s16, s0 // (op 27180) 

s_mul_hi_i32 s0, s18, s12 // (op 27180) 
s_add_i32 s19, s0, s18 // (op 27180) 


s_ashr_i32 s0, s19, 31 // (op 27180) 
s_cmp_eq_i32 s12, 0 // (op 27180) 
s_mov_b32 s18, scc // (op 27180) 
s_cmp_eq_i32 s18, 0 // (op 27180) 
s_cselect_b32 s22, 0, -1 // (op 27180) 

s_and_b32 s18, s44, 31 // (op 27180) 
s_lshl_b32 s23, 1, s18 // (op 27180) 
s_add_i32 s24, s23, s22 // (op 27180) 


s_and_b32 s25, s0, s24 // (op 27180) 


s_add_i32 s0, s19, s25 // (op 27180) 


s_ashr_i32 s19, s0, s18 // (op 27180) 


s_lshl_b32 s0, s44, 24 // (op 27180) 
s_ashr_i32 s18, s0, 31 // (op 27180) 

s_xor_b32 s0, s19, s18 // (op 27180) 

s_sub_i32 s19, s0, s18 // (op 27180) 


s_cmp_eq_i32 s1, 0 // (op 27180) 
s_mov_b32 s0, scc // (op 27180) 

s_cmp_eq_i32 s0, 0 // (op 27180) 
s_cselect_b32 s1, s19, s47 // (op 27180) 



v_lshl_add_u32 v11, s1, 8, v22 // (op 27180) 


v_mul_lo_u32 v22, v11, s38 // (op 27180) 

v_lshl_add_u32 v11, v23, 2, v22 // (op 27180) 


s_mul_i32 s1, s38, s7 // (op 27182) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 27182) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 27182) least significant
s_mul_hi_u32 s18, s38, s6 // (op 27182) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 27182) most significant: sum
s_add_u32 s1, s1, s18 // (op 27182) most significant: sum


s_mov_b32 s68, s40 // (op 27182) reg expression
s_mov_b32 s69, s41 // (op 27182) reg expression
s_mov_b32 s70, s0 // (op 27182) reg expression
s_mov_b32 s71, 131072 // (op 27182) call()

v_and_b32 v22, -4, v1 // (op 26957) 
v_sub_u32 v23, v1, v22 // (op 26957) 

v_lshlrev_b32 v22, 5, v23 // (op 26957) 

v_lshrrev_b32 v23, 2, v1 // (op 26957) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 26957) 


v_lshlrev_b32 v22, 2, v24 // (op 26957) 

v_lshrrev_b32 v23, 3, v22 // (op 26957) 

s_nop 0
v_readfirstlane_b32 s47, v23 // (op 26957) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 26632) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 26632) 

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 26632) 


v_mov_b64 v[24:25], v[12:13] // (op 26632) 
s_mov_b32 m0, s47 // (op 26632) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 26632) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 26632) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 26632) most significant half

s_add_u32 m0, m0, 4096 // (op 26632) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 26632) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 26632) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 26632) most significant half

s_add_u32 m0, m0, 4096 // (op 26632) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 26632) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 26632) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 26632) most significant half

s_add_u32 m0, m0, 4096 // (op 26632) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 26632) Load value direct to lds



v_and_b32 v22, -8, v1 // (op 26900) 
v_sub_u32 v23, v1, v22 // (op 26900) 

v_lshlrev_b32 v22, 4, v23 // (op 26900) 

v_lshrrev_b32 v23, 3, v1 // (op 26900) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 26900) 


s_nop 0
v_readfirstlane_b32 s47, v24 // (op 26900) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 26637) 
v_mov_b64 v[22:23], s[0:1] // (op 26637) convert


v_mov_b64 v[24:25], v[14:15] // (op 26637) 
s_add_u32 s0, 16384, s47 // (op 26637) 
s_mov_b32 m0, s0 // (op 26637) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26637) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26637) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26637) most significant half

s_add_u32 m0, m0, 4096 // (op 26637) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26637) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26637) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26637) most significant half

s_add_u32 m0, m0, 4096 // (op 26637) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26637) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26637) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26637) most significant half

s_add_u32 m0, m0, 4096 // (op 26637) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26637) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26637) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26637) most significant half

s_add_u32 m0, m0, 4096 // (op 26637) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26637) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26637) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26637) most significant half

s_add_u32 m0, m0, 4096 // (op 26637) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26637) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26637) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26637) most significant half

s_add_u32 m0, m0, 4096 // (op 26637) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26637) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 26637) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 26637) most significant half

s_add_u32 m0, m0, 4096 // (op 26637) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26637) Load value direct to lds






v_mov_b32 v22, v10 // (op 74) reg expression

buffer_load_dwordx2 v[24:25], v22, s[68:71], 0 offen // (op 74) Load value



s_lshl_b32 s47, s38, 6 // (op 722) 
v_add_u32 v22, v10, s47 // (op 722) 


buffer_load_dwordx2 v[26:27], v22, s[68:71], 0 offen // (op 722) Load value




s_lshl_b32 s47, s34, 6 // (op 5307) 
v_add_u32 v10, v20, s47 // (op 5307) 


buffer_load_dwordx2 v[22:23], v10, s[64:67], 0 offen // (op 5307) Load value



v_mov_b32 v10, v20 // (op 70) reg expression

buffer_load_dwordx2 v[28:29], v10, s[64:67], 0 offen // (op 70) Load value


v_and_b32 v10, -4, v1 // (op 27068) 
v_sub_u32 v20, v1, v10 // (op 27068) 

v_lshlrev_b32 v10, 5, v20 // (op 27068) 

v_lshrrev_b32 v20, 2, v1 // (op 27068) 
v_lshl_add_u32 v30, v20, 7, v10 // (op 27068) 


v_lshlrev_b32 v10, 2, v30 // (op 27068) 

v_lshrrev_b32 v20, 3, v10 // (op 27068) 

s_nop 0
v_readfirstlane_b32 s47, v20 // (op 27068) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 26642) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 26642) 

v_lshrrev_b64 v[30:31], 3, s[18:19] // (op 26642) 


v_mov_b64 v[32:33], v[12:13] // (op 26642) 
s_add_u32 s0, 49152, s47 // (op 26642) 
s_mov_b32 m0, s0 // (op 26642) reg expression
buffer_load_dwordx4 v32, s[60:63], 64 offen offset: 0 lds // (op 26642) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 26642) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 26642) most significant half

s_add_u32 m0, m0, 4096 // (op 26642) 
buffer_load_dwordx4 v32, s[60:63], 64 offen offset: 0 lds // (op 26642) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 26642) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 26642) most significant half

s_add_u32 m0, m0, 4096 // (op 26642) 
buffer_load_dwordx4 v32, s[60:63], 64 offen offset: 0 lds // (op 26642) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 26642) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 26642) most significant half

s_add_u32 m0, m0, 4096 // (op 26642) 
buffer_load_dwordx4 v32, s[60:63], 64 offen offset: 0 lds // (op 26642) Load value direct to lds





v_and_b32 v10, -8, v1 // (op 27024) 
v_sub_u32 v20, v1, v10 // (op 27024) 

v_lshlrev_b32 v10, 4, v20 // (op 27024) 

v_lshrrev_b32 v20, 3, v1 // (op 27024) 
v_lshl_add_u32 v12, v20, 7, v10 // (op 27024) 


s_nop 0
v_readfirstlane_b32 s47, v12 // (op 27024) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 26644) 
v_mov_b64 v[12:13], s[0:1] // (op 26644) convert


v_mov_b64 v[30:31], v[14:15] // (op 26644) 
s_add_u32 s0, 65536, s47 // (op 26644) 
s_mov_b32 m0, s0 // (op 26644) reg expression
s_mov_b32 s1, 128 // (op 26644) call()
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26644) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26644) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26644) most significant half

s_add_u32 m0, m0, 4096 // (op 26644) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26644) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26644) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26644) most significant half

s_add_u32 m0, m0, 4096 // (op 26644) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26644) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26644) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26644) most significant half

s_add_u32 m0, m0, 4096 // (op 26644) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26644) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26644) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26644) most significant half

s_add_u32 m0, m0, 4096 // (op 26644) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26644) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26644) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26644) most significant half

s_add_u32 m0, m0, 4096 // (op 26644) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26644) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26644) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26644) most significant half

s_add_u32 m0, m0, 4096 // (op 26644) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26644) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v12 // (op 26644) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 26644) most significant half

s_add_u32 m0, m0, 4096 // (op 26644) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 26644) Load value direct to lds






s_waitcnt vmcnt(24)
s_barrier  // (op 27597) 



ds_read_b128 v[12:15], v4 // (op 25983) Load local data 



ds_read_b128 v[30:33], v4 offset:1024 // (op 25988) Load local data 



ds_read_b128 v[34:37], v4 offset:2048 // (op 25993) Load local data 



ds_read_b128 v[38:41], v4 offset:3072 // (op 25998) Load local data 



ds_read_b128 v[42:45], v4 offset:4096 // (op 26003) Load local data 



ds_read_b128 v[46:49], v4 offset:5120 // (op 26008) Load local data 



ds_read_b128 v[50:53], v4 offset:6144 // (op 26013) Load local data 



ds_read_b128 v[54:57], v4 offset:7168 // (op 26018) Load local data 
s_waitcnt vmcnt(16)
s_barrier  // (op 27581) 



ds_read_b128 v[58:61], v3 offset:16384 // (op 25943) Load local data 
ds_read_b128 v[62:65], v3 offset:16448 // (op 25943) Load local data 



ds_read_b128 v[66:69], v3 offset:18432 // (op 25948) Load local data 
ds_read_b128 v[70:73], v3 offset:18496 // (op 25948) Load local data 



ds_read_b128 v[74:77], v3 offset:20480 // (op 25953) Load local data 
ds_read_b128 v[78:81], v3 offset:20544 // (op 25953) Load local data 



ds_read_b128 v[82:85], v3 offset:22528 // (op 25958) Load local data 
ds_read_b128 v[86:89], v3 offset:22592 // (op 25958) Load local data 



ds_read_b128 v[90:93], v3 offset:24576 // (op 25963) Load local data 
ds_read_b128 v[94:97], v3 offset:24640 // (op 25963) Load local data 



ds_read_b128 v[98:101], v3 offset:26624 // (op 25968) Load local data 
ds_read_b128 v[102:105], v3 offset:26688 // (op 25968) Load local data 



ds_read_b128 v[106:109], v3 offset:28672 // (op 25973) Load local data 
ds_read_b128 v[110:113], v3 offset:28736 // (op 25973) Load local data 



ds_read_b128 v[114:117], v3 offset:30720 // (op 25978) Load local data 
ds_read_b128 v[118:121], v3 offset:30784 // (op 25978) Load local data 
v_and_b32 v10, -8, v1 // (op 26872) 
v_sub_u32 v20, v1, v10 // (op 26872) 

v_lshlrev_b32 v10, 4, v20 // (op 26872) 

v_lshrrev_b32 v20, 3, v1 // (op 26872) 
v_lshl_add_u32 v122, v20, 7, v10 // (op 26872) 


s_nop 0
v_readfirstlane_b32 s47, v122 // (op 26872) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 26929) 
v_sub_u32 v20, v1, v10 // (op 26929) 

v_lshlrev_b32 v10, 5, v20 // (op 26929) 

v_lshrrev_b32 v20, 2, v1 // (op 26929) 
v_lshl_add_u32 v122, v20, 7, v10 // (op 26929) 


v_lshlrev_b32 v10, 2, v122 // (op 26929) 

v_lshrrev_b32 v20, 3, v10 // (op 26929) 

s_nop 0
v_readfirstlane_b32 s0, v20 // (op 26929) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 27009) 
v_sub_u32 v20, v1, v10 // (op 27009) 

v_lshlrev_b32 v10, 4, v20 // (op 27009) 

v_lshrrev_b32 v20, 3, v1 // (op 27009) 
v_lshl_add_u32 v122, v20, 7, v10 // (op 27009) 


s_nop 0
v_readfirstlane_b32 s1, v122 // (op 27009) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 27053) 
v_sub_u32 v20, v1, v10 // (op 27053) 

v_lshlrev_b32 v10, 5, v20 // (op 27053) 

v_lshrrev_b32 v20, 2, v1 // (op 27053) 
v_lshl_add_u32 v122, v20, 7, v10 // (op 27053) 


v_lshlrev_b32 v10, 2, v122 // (op 27053) 

v_lshrrev_b32 v20, 3, v10 // (op 27053) 

s_nop 0
v_readfirstlane_b32 s18, v20 // (op 27053) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s19, 0 // (op 1095) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 1094) 
s_mov_b32 s24, s22 // (op 1094) convert

s_and_b32 s25, s24, 4294967294 // (op 1094) 

s_cmp_lt_u32 s19, s25 // (op 1094) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_1094 // (op 1094) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_1094 if false)
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_1094:
s_add_u32 s19, s19, 2 // (op 1096) 
s_waitcnt vmcnt(15)
v_mov_b32 v122, v24 // (op 26518) reg expression
v_mov_b32 v123, v25 // (op 26518) reg expression
s_barrier  // (op 27600) 
s_waitcnt vmcnt(14)
v_mov_b32 v124, v26 // (op 26531) reg expression

v_mov_b32 v125, v27 // (op 26531) reg expression
s_lshl_b64 s[22:23], s[30:31], 6 // (op 26605) 
s_lshl_b64 s[24:25], s[22:23], 2 // (op 26605) 

v_lshrrev_b64 v[126:127], 3, s[24:25] // (op 26605) 


v_mov_b64 v[128:129], v[18:19] // (op 26605) 
v_mov_b64 v[130:131], v[128:129] // (op 26605) 
s_mov_b32 m0, s0 // (op 26605) reg expression
s_mov_b32 s22, 128 // (op 26605) call()
buffer_load_dwordx4 v130, s[60:63], s22 offen offset: 0 lds // (op 26605) Load value direct to lds
v_add_co_u32 v130, s[24:25], v130, v126 // (op 26605) least significant half
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 26605) most significant half

s_add_u32 m0, m0, 4096 // (op 26605) 
buffer_load_dwordx4 v130, s[60:63], s22 offen offset: 0 lds // (op 26605) Load value direct to lds
v_add_co_u32 v130, s[24:25], v130, v126 // (op 26605) least significant half
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 26605) most significant half

s_add_u32 m0, m0, 4096 // (op 26605) 
buffer_load_dwordx4 v130, s[60:63], s22 offen offset: 0 lds // (op 26605) Load value direct to lds
v_add_co_u32 v130, s[24:25], v130, v126 // (op 26605) least significant half
v_addc_co_u32 v131, s[24:25], 0, v131, s[24:25] // (op 26605) most significant half

s_add_u32 m0, m0, 4096 // (op 26605) 
s_waitcnt vmcnt(16)
v_mov_b32 v132, v22 // (op 26544) reg expression
v_mov_b32 v133, v23 // (op 26544) reg expression
s_waitcnt vmcnt(15)
v_mov_b32 v134, v28 // (op 26557) reg expression
v_mov_b32 v135, v29 // (op 26557) reg expression
buffer_load_dwordx4 v130, s[60:63], s22 offen offset: 0 lds // (op 26605) Load value direct to lds



v_bfe_u32 v136, v134, 0, 8 // (op 24596) 
v_bfe_u32 v137, v134, 8, 8 // (op 24596) 
v_bfe_u32 v138, v134, 16, 8 // (op 24596) 
v_bfe_u32 v139, v134, 24, 8 // (op 24596) 
v_permlane16_swap_b32 v136, v137 // (op 24596) 
s_nop 0
v_permlane16_swap_b32 v138, v139 // (op 24596) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 24596) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v137, v139 // (op 24596) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 27584) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 26600) 
v_mov_b64 v[126:127], s[22:23] // (op 26600) convert


v_mov_b64 v[130:131], v[16:17] // (op 26600) 
v_mov_b64 v[140:141], v[130:131] // (op 26600) 
s_add_u32 s22, 16384, s47 // (op 26600) 
s_mov_b32 m0, s22 // (op 26600) reg expression
s_mov_b32 s23, 256 // (op 26600) call()
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26600) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26600) least significant half
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26600) most significant half

s_add_u32 m0, m0, 4096 // (op 26600) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26600) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26600) least significant half
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26600) most significant half

s_add_u32 m0, m0, 4096 // (op 26600) 
v_bfe_u32 v142, v122, 0, 8 // (op 24787) 
v_bfe_u32 v143, v122, 8, 8 // (op 24787) 
v_bfe_u32 v144, v122, 16, 8 // (op 24787) 
v_bfe_u32 v145, v122, 24, 8 // (op 24787) 
v_permlane16_swap_b32 v142, v143 // (op 24787) 
s_nop 0
v_permlane16_swap_b32 v144, v145 // (op 24787) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v142, v144 // (op 24787) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v143, v145 // (op 24787) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26600) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26600) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[12:15], a[0:3], v136, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 76) 
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26600) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[30:33], a[4:7], v136, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 311) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[34:37], a[8:11], v136, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 447) 
s_add_u32 m0, m0, 4096 // (op 26600) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[38:41], a[12:15], v136, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 583) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26600) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26600) least significant half
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26600) most significant half

s_add_u32 m0, m0, 4096 // (op 26600) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26600) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26600) least significant half
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26600) most significant half

s_add_u32 m0, m0, 4096 // (op 26600) 
v_bfe_u32 v146, v124, 0, 8 // (op 24819) 
v_bfe_u32 v147, v124, 8, 8 // (op 24819) 
v_bfe_u32 v148, v124, 16, 8 // (op 24819) 
v_bfe_u32 v149, v124, 24, 8 // (op 24819) 
v_permlane16_swap_b32 v146, v147 // (op 24819) 
s_nop 0
v_permlane16_swap_b32 v148, v149 // (op 24819) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v146, v148 // (op 24819) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v147, v149 // (op 24819) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26600) Load value direct to lds
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26600) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:65], v[42:45], a[16:19], v136, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 719) 
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26600) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[46:49], a[20:23], v136, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 855) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:65], v[50:53], a[24:27], v136, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 991) 
s_add_u32 m0, m0, 4096 // (op 26600) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[54:57], a[28:31], v136, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1127) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26600) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:73], v[12:15], a[32:35], v137, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1895) 
v_add_co_u32 v140, s[24:25], v140, v126 // (op 26600) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:73], v[30:33], a[36:39], v137, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1857) 
v_addc_co_u32 v141, s[24:25], 0, v141, s[24:25] // (op 26600) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[66:73], v[34:37], a[40:43], v137, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1819) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[66:73], v[38:41], a[44:47], v137, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1781) 
s_add_u32 m0, m0, 4096 // (op 26600) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:73], v[42:45], a[48:51], v137, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1743) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:73], v[46:49], a[52:55], v137, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1705) 
buffer_load_dwordx4 v140, s[48:51], s23 offen offset: 0 lds // (op 26600) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[66:73], v[50:53], a[56:59], v137, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1667) 
v_mov_b32 v10, v11 // (op 26526) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:73], v[54:57], a[60:63], v137, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1629) 
v_add_u32 v20, 8, v10 // (op 26526) 

v_bfe_u32 v150, v132, 0, 8 // (op 24627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:81], v[12:15], a[64:67], v138, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3113) 
v_bfe_u32 v151, v132, 8, 8 // (op 24627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:81], v[30:33], a[68:71], v138, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3075) 
v_bfe_u32 v152, v132, 16, 8 // (op 24627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:81], v[34:37], a[72:75], v138, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3037) 
v_bfe_u32 v153, v132, 24, 8 // (op 24627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[74:81], v[38:41], a[76:79], v138, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2999) 
v_permlane16_swap_b32 v150, v151 // (op 24627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:81], v[42:45], a[80:83], v138, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2961) 
v_permlane16_swap_b32 v152, v153 // (op 24627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:81], v[46:49], a[84:87], v138, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2923) 
s_nop 0
v_permlane32_swap_b32 v150, v152 // (op 24627) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:81], v[50:53], a[88:91], v138, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2885) 
v_permlane32_swap_b32 v151, v153 // (op 24627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:81], v[54:57], a[92:95], v138, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2847) 
s_waitcnt vmcnt(20)
s_barrier  // (op 27619) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:89], v[12:15], a[96:99], v139, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4331) 
ds_read_b128 v[154:157], v6 offset:49152 // (op 212) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:89], v[30:33], a[100:103], v139, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4293) 
ds_read_b128 v[158:161], v6 offset:50176 // (op 320) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:89], v[34:37], a[104:107], v139, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4255) 
ds_read_b128 v[162:165], v6 offset:51200 // (op 456) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:89], v[38:41], a[108:111], v139, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4217) 
buffer_load_dwordx2 v[24:25], v20, s[68:71], 0 offen // (op 26526) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:89], v[42:45], a[112:115], v139, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4179) 
s_lshl_b32 s22, s38, 6 // (op 26539) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:89], v[46:49], a[116:119], v139, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4141) 
s_add_u32 s22, s22, 8 // (op 26539) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:89], v[50:53], a[120:123], v139, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4103) 
v_add_u32 v20, v10, s22 // (op 26539) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:89], v[54:57], a[124:127], v139, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4065) 
buffer_load_dwordx2 v[26:27], v20, s[68:71], 0 offen // (op 26539) Load value





v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[90:97], v[12:15], a[128:131], v150, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5549) 
v_add_u32 v11, 8, v11 // (op 27170) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[90:97], v[30:33], a[132:135], v150, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5511) 
v_mov_b32 v10, v21 // (op 26552) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:97], v[34:37], a[136:139], v150, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5473) 
s_lshl_b32 s22, s34, 6 // (op 26552) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[90:97], v[38:41], a[140:143], v150, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5435) 
s_add_u32 s23, 8, s22 // (op 26552) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[90:97], v[42:45], a[144:147], v150, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5397) 
v_add_u32 v20, v10, s23 // (op 26552) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[90:97], v[46:49], a[148:151], v150, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5359) 
buffer_load_dwordx2 v[22:23], v20, s[64:67], 0 offen // (op 26552) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[90:97], v[50:53], a[152:155], v150, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5321) 
v_add_u32 v20, 8, v10 // (op 26565) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:97], v[54:57], a[156:159], v150, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5283) 
buffer_load_dwordx2 v[28:29], v20, s[64:67], 0 offen // (op 26565) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[98:105], v[12:15], a[160:163], v151, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6767) 
v_add_u32 v21, 8, v21 // (op 27118) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[98:105], v[30:33], a[164:167], v151, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6729) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:105], v[34:37], a[168:171], v151, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6691) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[98:105], v[38:41], a[172:175], v151, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6653) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[98:105], v[42:45], a[176:179], v151, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6615) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[98:105], v[46:49], a[180:183], v151, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6577) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[98:105], v[50:53], a[184:187], v151, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6539) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[98:105], v[54:57], a[188:191], v151, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6501) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[106:113], v[12:15], a[192:195], v152, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7985) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[106:113], v[30:33], a[196:199], v152, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7947) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[106:113], v[34:37], a[200:203], v152, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7909) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[106:113], v[38:41], a[204:207], v152, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7871) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[106:113], v[42:45], a[208:211], v152, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[106:113], v[46:49], a[212:215], v152, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7795) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[106:113], v[50:53], a[216:219], v152, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7757) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[106:113], v[54:57], a[220:223], v152, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7719) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[114:121], v[12:15], a[224:227], v153, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9203) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[114:121], v[30:33], a[228:231], v153, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9165) 
ds_read_b128 v[136:139], v6 offset:52224 // (op 592) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:121], v[34:37], a[232:235], v153, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9127) 
ds_read_b128 v[166:169], v6 offset:53248 // (op 728) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:121], v[38:41], a[236:239], v153, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9089) 
ds_read_b128 v[170:173], v6 offset:54272 // (op 864) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[114:121], v[42:45], a[240:243], v153, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9051) 
ds_read_b128 v[140:143], v6 offset:55296 // (op 1000) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[114:121], v[46:49], a[244:247], v153, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9013) 
ds_read_b128 v[174:177], v6 offset:56320 // (op 1136) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[114:121], v[50:53], a[248:251], v153, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8975) 
s_waitcnt vmcnt(16)
s_barrier  // (op 27613) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[114:121], v[54:57], a[252:255], v153, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8937) 
v_add_u32 v10, 65536, v5 // (op 210) 


ds_read_b128 v[58:61], v10 // (op 210) Load local data 

v_add_u32 v10, 65600, v5 // (op 210) 
ds_read_b128 v[62:65], v10 // (op 210) Load local data 




v_add_u32 v10, 67584, v5 // (op 1637) 
ds_read_b128 v[66:69], v10 // (op 1637) Load local data 

v_add_u32 v10, 67648, v5 // (op 1637) 
ds_read_b128 v[70:73], v10 // (op 1637) Load local data 




v_add_u32 v10, 69632, v5 // (op 2855) 
ds_read_b128 v[74:77], v10 // (op 2855) Load local data 

v_add_u32 v10, 69696, v5 // (op 2855) 
ds_read_b128 v[78:81], v10 // (op 2855) Load local data 




v_add_u32 v10, 71680, v5 // (op 4073) 
ds_read_b128 v[82:85], v10 // (op 4073) Load local data 

v_add_u32 v10, 71744, v5 // (op 4073) 
ds_read_b128 v[86:89], v10 // (op 4073) Load local data 




v_add_u32 v10, 73728, v5 // (op 5291) 
ds_read_b128 v[90:93], v10 // (op 5291) Load local data 

v_add_u32 v10, 73792, v5 // (op 5291) 
ds_read_b128 v[94:97], v10 // (op 5291) Load local data 




v_add_u32 v10, 75776, v5 // (op 6509) 
ds_read_b128 v[98:101], v10 // (op 6509) Load local data 

v_add_u32 v10, 75840, v5 // (op 6509) 
ds_read_b128 v[102:105], v10 // (op 6509) Load local data 




v_add_u32 v10, 77824, v5 // (op 7727) 
ds_read_b128 v[106:109], v10 // (op 7727) Load local data 

v_add_u32 v10, 77888, v5 // (op 7727) 
ds_read_b128 v[110:113], v10 // (op 7727) Load local data 




v_add_u32 v10, 79872, v5 // (op 8945) 
ds_read_b128 v[144:147], v10 // (op 8945) Load local data 

v_add_u32 v10, 79936, v5 // (op 8945) 
ds_read_b128 v[148:151], v10 // (op 8945) Load local data 

v_bfe_u32 v178, v135, 0, 8 // (op 24611) 
v_bfe_u32 v179, v135, 8, 8 // (op 24611) 
v_bfe_u32 v180, v135, 16, 8 // (op 24611) 
v_bfe_u32 v181, v135, 24, 8 // (op 24611) 
v_permlane16_swap_b32 v178, v179 // (op 24611) 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 27623) 
v_permlane16_swap_b32 v180, v181 // (op 24611) 

s_lshl_b64 s[22:23], s[30:31], 6 // (op 26627) 
s_lshl_b64 s[24:25], s[22:23], 2 // (op 26627) 

v_permlane32_swap_b32 v178, v180 // (op 24611) 
v_permlane32_swap_b32 v179, v181 // (op 24611) 
v_lshrrev_b64 v[126:127], 3, s[24:25] // (op 26627) 



v_mov_b64 v[134:135], v[128:129] // (op 26627) 
s_add_u32 s22, 49152, s18 // (op 26627) 
s_mov_b32 m0, s22 // (op 26627) reg expression
s_mov_b32 s23, 192 // (op 26627) call()
buffer_load_dwordx4 v134, s[60:63], s23 offen offset: 0 lds // (op 26627) Load value direct to lds
v_add_co_u32 v134, s[24:25], v134, v126 // (op 26627) least significant half
v_addc_co_u32 v135, s[24:25], 0, v135, s[24:25] // (op 26627) most significant half

s_add_u32 m0, m0, 4096 // (op 26627) 
buffer_load_dwordx4 v134, s[60:63], s23 offen offset: 0 lds // (op 26627) Load value direct to lds
v_add_co_u32 v134, s[24:25], v134, v126 // (op 26627) least significant half
v_addc_co_u32 v135, s[24:25], 0, v135, s[24:25] // (op 26627) most significant half

s_add_u32 m0, m0, 4096 // (op 26627) 
buffer_load_dwordx4 v134, s[60:63], s23 offen offset: 0 lds // (op 26627) Load value direct to lds
v_add_co_u32 v134, s[24:25], v134, v126 // (op 26627) least significant half
v_addc_co_u32 v135, s[24:25], 0, v135, s[24:25] // (op 26627) most significant half

s_add_u32 m0, m0, 4096 // (op 26627) 
v_bfe_u32 v182, v123, 0, 8 // (op 24802) 
v_bfe_u32 v183, v123, 8, 8 // (op 24802) 
v_bfe_u32 v184, v123, 16, 8 // (op 24802) 
v_bfe_u32 v185, v123, 24, 8 // (op 24802) 
v_permlane16_swap_b32 v182, v183 // (op 24802) 
s_nop 0
v_permlane16_swap_b32 v184, v185 // (op 24802) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v182, v184 // (op 24802) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v183, v185 // (op 24802) 
buffer_load_dwordx4 v134, s[60:63], s23 offen offset: 0 lds // (op 26627) Load value direct to lds






s_waitcnt lgkmcnt(14)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[154:157], a[0:3], v178, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 217) 
v_add_co_u32 v18, vcc, 128, v18 // (op 26821) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[158:161], a[4:7], v178, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 312) 
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 26821) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[162:165], a[8:11], v178, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 448) 
v_bfe_u32 v126, v125, 0, 8 // (op 24834) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[136:139], a[12:15], v178, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 584) 
v_bfe_u32 v127, v125, 8, 8 // (op 24834) 
v_bfe_u32 v128, v125, 16, 8 // (op 24834) 
v_bfe_u32 v129, v125, 24, 8 // (op 24834) 
v_permlane16_swap_b32 v126, v127 // (op 24834) 
s_nop 0
v_permlane16_swap_b32 v128, v129 // (op 24834) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 24834) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 24834) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 27616) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:65], v[166:169], a[16:19], v178, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 720) 
s_lshl_b64 s[22:23], s[26:27], 5 // (op 26622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[170:173], a[20:23], v178, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 856) 
v_mov_b64 v[134:135], s[22:23] // (op 26622) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:65], v[140:143], a[24:27], v178, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 992) 
v_mov_b64 v[152:153], v[130:131] // (op 26622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[174:177], a[28:31], v178, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1128) 
s_add_u32 s22, 65536, s1 // (op 26622) 
s_mov_b32 m0, s22 // (op 26622) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:73], v[154:157], a[32:35], v179, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1896) 
s_mov_b32 s23, 384 // (op 26622) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:73], v[158:161], a[36:39], v179, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1858) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26622) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[66:73], v[162:165], a[40:43], v179, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1820) 
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26622) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[66:73], v[136:139], a[44:47], v179, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1782) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26622) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:73], v[166:169], a[48:51], v179, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1744) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:73], v[170:173], a[52:55], v179, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1706) 
s_add_u32 m0, m0, 4096 // (op 26622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[66:73], v[140:143], a[56:59], v179, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1668) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:73], v[174:177], a[60:63], v179, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1630) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26622) Load value direct to lds
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26622) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:81], v[154:157], a[64:67], v180, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3114) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26622) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:81], v[158:161], a[68:71], v180, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3076) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:81], v[162:165], a[72:75], v180, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3038) 
s_add_u32 m0, m0, 4096 // (op 26622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[74:81], v[136:139], a[76:79], v180, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3000) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:81], v[166:169], a[80:83], v180, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2962) 
v_bfe_u32 v122, v133, 0, 8 // (op 24642) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:81], v[170:173], a[84:87], v180, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2924) 
v_bfe_u32 v123, v133, 8, 8 // (op 24642) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:81], v[140:143], a[88:91], v180, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2886) 
v_bfe_u32 v124, v133, 16, 8 // (op 24642) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:81], v[174:177], a[92:95], v180, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2848) 
v_bfe_u32 v125, v133, 24, 8 // (op 24642) 
v_permlane16_swap_b32 v122, v123 // (op 24642) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:89], v[154:157], a[96:99], v181, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4332) 
v_permlane16_swap_b32 v124, v125 // (op 24642) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:89], v[158:161], a[100:103], v181, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4294) 
s_nop 0
v_permlane32_swap_b32 v122, v124 // (op 24642) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:89], v[162:165], a[104:107], v181, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4256) 
v_permlane32_swap_b32 v123, v125 // (op 24642) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:89], v[136:139], a[108:111], v181, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4218) 
s_waitcnt vmcnt(18)
s_barrier  // (op 27606) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:89], v[166:169], a[112:115], v181, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4180) 
ds_read_b128 v[12:15], v4 // (op 65) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:89], v[170:173], a[116:119], v181, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4142) 
ds_read_b128 v[30:33], v4 offset:1024 // (op 308) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:89], v[140:143], a[120:123], v181, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4104) 
ds_read_b128 v[34:37], v4 offset:2048 // (op 444) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:89], v[174:177], a[124:127], v181, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4066) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26622) Load value direct to lds

v_add_co_u32 v152, s[24:25], v152, v134 // (op 26622) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[90:97], v[154:157], a[128:131], v122, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5550) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26622) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[90:97], v[158:161], a[132:135], v122, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5512) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:97], v[162:165], a[136:139], v122, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5474) 
s_add_u32 m0, m0, 4096 // (op 26622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[90:97], v[136:139], a[140:143], v122, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5436) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[90:97], v[166:169], a[144:147], v122, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5398) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26622) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[90:97], v[170:173], a[148:151], v122, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5360) 
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26622) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[90:97], v[140:143], a[152:155], v122, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5322) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26622) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:97], v[174:177], a[156:159], v122, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5284) 
s_add_u32 m0, m0, 4096 // (op 26622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[98:105], v[154:157], a[160:163], v123, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6768) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[98:105], v[158:161], a[164:167], v123, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6730) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26622) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:105], v[162:165], a[168:171], v123, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6692) 
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26622) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[98:105], v[136:139], a[172:175], v123, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6654) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26622) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[98:105], v[166:169], a[176:179], v123, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[98:105], v[170:173], a[180:183], v123, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6578) 
s_add_u32 m0, m0, 4096 // (op 26622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[98:105], v[140:143], a[184:187], v123, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6540) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[98:105], v[174:177], a[188:191], v123, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6502) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26622) Load value direct to lds
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26622) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[106:113], v[154:157], a[192:195], v124, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7986) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26622) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[106:113], v[158:161], a[196:199], v124, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7948) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[106:113], v[162:165], a[200:203], v124, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7910) 
s_add_u32 m0, m0, 4096 // (op 26622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[106:113], v[136:139], a[204:207], v124, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7872) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[106:113], v[166:169], a[208:211], v124, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7834) 
buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26622) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[106:113], v[170:173], a[212:215], v124, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7796) 
v_add_co_u32 v152, s[24:25], v152, v134 // (op 26622) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[106:113], v[140:143], a[216:219], v124, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7758) 
v_addc_co_u32 v153, s[24:25], 0, v153, s[24:25] // (op 26622) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[106:113], v[174:177], a[220:223], v124, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7720) 
s_add_u32 m0, m0, 4096 // (op 26622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[144:151], v[154:157], a[224:227], v125, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9204) 

buffer_load_dwordx4 v152, s[48:51], s23 offen offset: 0 lds // (op 26622) Load value direct to lds








v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[144:151], v[158:161], a[228:231], v125, v183 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9166) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[144:151], v[162:165], a[232:235], v125, v184 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9128) 
ds_read_b128 v[38:41], v4 offset:3072 // (op 580) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[144:151], v[136:139], a[236:239], v125, v185 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9090) 
ds_read_b128 v[42:45], v4 offset:4096 // (op 716) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[144:151], v[166:169], a[240:243], v125, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9052) 
ds_read_b128 v[46:49], v4 offset:5120 // (op 852) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[144:151], v[170:173], a[244:247], v125, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 9014) 
v_add_co_u32 v16, vcc, 256, v16 // (op 26757) least significant half

v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 26757) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[144:151], v[140:143], a[248:251], v125, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8976) 
ds_read_b128 v[50:53], v4 offset:6144 // (op 988) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[144:151], v[174:177], a[252:255], v125, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8938) 
ds_read_b128 v[54:57], v4 offset:7168 // (op 1124) Load local data 




s_waitcnt vmcnt(16)
s_barrier  // (op 27590) 



ds_read_b128 v[58:61], v3 offset:16384 // (op 60) Load local data 
ds_read_b128 v[62:65], v3 offset:16448 // (op 60) Load local data 



ds_read_b128 v[66:69], v3 offset:18432 // (op 1625) Load local data 
ds_read_b128 v[70:73], v3 offset:18496 // (op 1625) Load local data 



ds_read_b128 v[74:77], v3 offset:20480 // (op 2843) Load local data 
ds_read_b128 v[78:81], v3 offset:20544 // (op 2843) Load local data 



ds_read_b128 v[82:85], v3 offset:22528 // (op 4061) Load local data 
ds_read_b128 v[86:89], v3 offset:22592 // (op 4061) Load local data 



ds_read_b128 v[90:93], v3 offset:24576 // (op 5279) Load local data 
ds_read_b128 v[94:97], v3 offset:24640 // (op 5279) Load local data 



ds_read_b128 v[98:101], v3 offset:26624 // (op 6497) Load local data 
ds_read_b128 v[102:105], v3 offset:26688 // (op 6497) Load local data 



ds_read_b128 v[106:109], v3 offset:28672 // (op 7715) Load local data 
ds_read_b128 v[110:113], v3 offset:28736 // (op 7715) Load local data 



ds_read_b128 v[114:117], v3 offset:30720 // (op 8933) Load local data 
ds_read_b128 v[118:121], v3 offset:30784 // (op 8933) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 1094) 
s_mov_b32 s24, s22 // (op 1094) convert

s_and_b32 s25, s24, 4294967294 // (op 1094) 

s_cmp_lt_u32 s19, s25 // (op 1094) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_1094 // (op 1094) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoop_1094 if true)
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoop_1094:



















s_barrier  // (op 27645) 
s_mul_hi_i32 s47, s16, s10 // (op 26657) 
v_lshrrev_b32 v10, 3, v1 // (op 26657) 
s_add_i32 s0, s47, s16 // (op 26657) 

v_and_b32 v11, -8, v1 // (op 26657) 
s_ashr_i32 s1, s0, 31 // (op 26657) 
v_sub_u32 v12, v1, v11 // (op 26657) 

s_cmp_eq_i32 s10, 0 // (op 26657) 
s_mov_b32 s47, scc // (op 26657) 
v_lshlrev_b32 v11, 4, v12 // (op 26657) 

s_cmp_eq_i32 s47, 0 // (op 26657) 
s_cselect_b32 s18, 0, -1 // (op 26657) 

s_and_b32 s19, s13, 31 // (op 26657) 
s_lshl_b32 s47, 1, s19 // (op 26657) 
s_add_i32 s22, s47, s18 // (op 26657) 


s_and_b32 s18, s1, s22 // (op 26657) 


s_add_i32 s1, s0, s18 // (op 26657) 


s_ashr_i32 s0, s1, s19 // (op 26657) 


s_lshl_b32 s1, s13, 24 // (op 26657) 
s_ashr_i32 s47, s1, 31 // (op 26657) 

s_xor_b32 s1, s0, s47 // (op 26657) 

s_sub_i32 s0, s1, s47 // (op 26657) 


s_mul_hi_i32 s1, s0, s14 // (op 26657) 
s_add_i32 s47, s1, s0 // (op 26657) 

s_ashr_i32 s1, s47, 31 // (op 26657) 
s_cmp_eq_i32 s14, 0 // (op 26657) 
s_mov_b32 s18, scc // (op 26657) 
s_cmp_eq_i32 s18, 0 // (op 26657) 
s_cselect_b32 s19, 0, -1 // (op 26657) 

s_and_b32 s18, s15, 31 // (op 26657) 
s_lshl_b32 s22, 1, s18 // (op 26657) 
s_add_i32 s23, s22, s19 // (op 26657) 


s_and_b32 s19, s1, s23 // (op 26657) 


s_add_i32 s1, s47, s19 // (op 26657) 


s_ashr_i32 s19, s1, s18 // (op 26657) 


s_lshl_b32 s1, s15, 24 // (op 26657) 
s_ashr_i32 s18, s1, 31 // (op 26657) 

s_xor_b32 s1, s19, s18 // (op 26657) 

s_sub_i32 s19, s1, s18 // (op 26657) 


s_mul_i32 s1, s0, s11 // (op 26657) 

s_sub_u32 s0, s16, s1 // (op 26657) 

s_mul_i32 s1, s19, s11 // (op 26657) 
s_add_u32 s1, s1, s0 // (op 26657) 


s_mul_hi_i32 s0, s16, s9 // (op 26657) 
s_add_i32 s47, s0, s16 // (op 26657) 

s_ashr_i32 s0, s47, 31 // (op 26657) 
s_cmp_eq_i32 s9, 0 // (op 26657) 
s_mov_b32 s18, scc // (op 26657) 
s_cmp_eq_i32 s18, 0 // (op 26657) 
s_cselect_b32 s19, 0, -1 // (op 26657) 

s_and_b32 s18, s45, 31 // (op 26657) 
s_lshl_b32 s22, 1, s18 // (op 26657) 
s_add_i32 s23, s22, s19 // (op 26657) 


s_and_b32 s19, s0, s23 // (op 26657) 


s_add_i32 s0, s47, s19 // (op 26657) 


s_ashr_i32 s19, s0, s18 // (op 26657) 


s_lshl_b32 s0, s45, 24 // (op 26657) 
s_ashr_i32 s18, s0, 31 // (op 26657) 

s_xor_b32 s0, s19, s18 // (op 26657) 

s_sub_i32 s19, s0, s18 // (op 26657) 


s_add_u32 s22, s6, 256 // (op 26657) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 26657) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 26657) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 26657) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 26657) 

s_mul_i32 s0, s11, s22 // (op 26657) 

s_mul_i32 s18, s8, s0 // (op 26657) 

s_mul_i32 s0, s19, s18 // (op 26657) 

s_sub_u32 s18, s16, s0 // (op 26657) 

s_mov_b32 s0, s18 // (op 26657) 
s_mul_hi_i32 s47, s0, s12 // (op 26657) 
s_add_i32 s22, s47, s0 // (op 26657) 


s_ashr_i32 s0, s22, 31 // (op 26657) 
s_cmp_eq_i32 s12, 0 // (op 26657) 
s_mov_b32 s47, scc // (op 26657) 
s_cmp_eq_i32 s47, 0 // (op 26657) 
s_cselect_b32 s23, 0, -1 // (op 26657) 

s_and_b32 s47, s44, 31 // (op 26657) 
s_lshl_b32 s24, 1, s47 // (op 26657) 
s_add_i32 s25, s24, s23 // (op 26657) 


s_and_b32 s23, s0, s25 // (op 26657) 


s_add_i32 s0, s22, s23 // (op 26657) 


s_ashr_i32 s22, s0, s47 // (op 26657) 


s_lshl_b32 s0, s44, 24 // (op 26657) 
s_ashr_i32 s47, s0, 31 // (op 26657) 

s_xor_b32 s0, s22, s47 // (op 26657) 

s_sub_i32 s22, s0, s47 // (op 26657) 


s_mul_i32 s0, s46, s11 // (op 26657) 
s_add_u32 s24, s20, 256 // (op 26657) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 26657) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 26657) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 26657) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 26657) 

s_sub_i32 s23, s24, s0 // (op 26657) 

s_mul_i32 s47, s22, s23 // (op 26657) 


s_sub_u32 s22, s18, s47 // (op 26657) 


s_add_u32 s18, s22, s0 // (op 26657) 


s_cmp_eq_i32 s19, 0 // (op 26657) 
s_mov_b32 s0, scc // (op 26657) 

s_cmp_eq_i32 s0, 0 // (op 26657) 
s_cselect_b32 s19, s18, s1 // (op 26657) 



v_lshl_add_u32 v12, s19, 8, v10 // (op 26657) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 26657) 
s_lshr_b32 s47, s0, 1 // (op 26657) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 26657) 


v_mov_b32 v21, 0 // (op 26657) convert
v_mov_b32 v20, v10 // (op 26657) convert

v_mul_lo_u32 v11, v12, s27 // (op 26657) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 26657) most significant: high of low * low
v_mul_lo_u32 v10, v12, s26 // (op 26657) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 26657) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v20 // (op 26657) least significant half
v_addc_co_u32 v11, s[0:1], v11, v21, s[0:1] // (op 26657) most significant half



s_mul_i32 s1, s26, s21 // (op 26659) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 26659) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 26659) least significant
s_mul_hi_u32 s18, s26, s20 // (op 26659) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 26659) most significant: sum
s_add_u32 s1, s1, s18 // (op 26659) most significant: sum


s_mov_b32 s48, s28 // (op 26659) reg expression
s_mov_b32 s49, s29 // (op 26659) reg expression
s_mov_b32 s50, s0 // (op 26659) reg expression
s_mov_b32 s51, 131072 // (op 26659) call()

s_mul_hi_i32 s47, s16, s10 // (op 26787) 
v_lshrrev_b32 v12, 2, v1 // (op 26787) 
s_add_i32 s0, s47, s16 // (op 26787) 

v_and_b32 v13, -4, v1 // (op 26787) 
s_ashr_i32 s1, s0, 31 // (op 26787) 
v_sub_u32 v14, v1, v13 // (op 26787) 

s_cmp_eq_i32 s10, 0 // (op 26787) 
s_mov_b32 s47, scc // (op 26787) 
v_lshlrev_b32 v13, 5, v14 // (op 26787) 

s_cmp_eq_i32 s47, 0 // (op 26787) 
s_cselect_b32 s18, 0, -1 // (op 26787) 

s_and_b32 s19, s13, 31 // (op 26787) 
s_lshl_b32 s47, 1, s19 // (op 26787) 
s_add_i32 s22, s47, s18 // (op 26787) 


s_and_b32 s18, s1, s22 // (op 26787) 


s_add_i32 s1, s0, s18 // (op 26787) 


s_ashr_i32 s0, s1, s19 // (op 26787) 


s_lshl_b32 s1, s13, 24 // (op 26787) 
s_ashr_i32 s47, s1, 31 // (op 26787) 

s_xor_b32 s1, s0, s47 // (op 26787) 

s_sub_i32 s0, s1, s47 // (op 26787) 


s_mul_hi_i32 s1, s0, s14 // (op 26787) 
s_add_i32 s47, s1, s0 // (op 26787) 

s_ashr_i32 s1, s47, 31 // (op 26787) 
s_cmp_eq_i32 s14, 0 // (op 26787) 
s_mov_b32 s18, scc // (op 26787) 
s_cmp_eq_i32 s18, 0 // (op 26787) 
s_cselect_b32 s19, 0, -1 // (op 26787) 

s_and_b32 s18, s15, 31 // (op 26787) 
s_lshl_b32 s22, 1, s18 // (op 26787) 
s_add_i32 s23, s22, s19 // (op 26787) 


s_and_b32 s19, s1, s23 // (op 26787) 


s_add_i32 s1, s47, s19 // (op 26787) 


s_ashr_i32 s19, s1, s18 // (op 26787) 


s_lshl_b32 s1, s15, 24 // (op 26787) 
s_ashr_i32 s18, s1, 31 // (op 26787) 

s_xor_b32 s1, s19, s18 // (op 26787) 

s_sub_i32 s19, s1, s18 // (op 26787) 


s_add_u32 s22, s6, 256 // (op 26787) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 26787) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 26787) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 26787) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 26787) 

s_mul_i32 s1, s19, s22 // (op 26787) 

s_sub_i32 s47, s0, s1 // (op 26787) 


s_mul_hi_i32 s0, s16, s9 // (op 26787) 
s_add_i32 s1, s0, s16 // (op 26787) 

s_ashr_i32 s0, s1, 31 // (op 26787) 
s_cmp_eq_i32 s9, 0 // (op 26787) 
s_mov_b32 s18, scc // (op 26787) 
s_cmp_eq_i32 s18, 0 // (op 26787) 
s_cselect_b32 s19, 0, -1 // (op 26787) 

s_and_b32 s18, s45, 31 // (op 26787) 
s_lshl_b32 s24, 1, s18 // (op 26787) 
s_add_i32 s25, s24, s19 // (op 26787) 


s_and_b32 s19, s0, s25 // (op 26787) 


s_add_i32 s0, s1, s19 // (op 26787) 


s_ashr_i32 s1, s0, s18 // (op 26787) 


s_lshl_b32 s0, s45, 24 // (op 26787) 
s_ashr_i32 s18, s0, 31 // (op 26787) 

s_xor_b32 s0, s1, s18 // (op 26787) 

s_sub_i32 s1, s0, s18 // (op 26787) 


s_mul_i32 s0, s11, s22 // (op 26787) 

s_mul_i32 s18, s8, s0 // (op 26787) 

s_mul_i32 s0, s1, s18 // (op 26787) 

s_sub_u32 s18, s16, s0 // (op 26787) 

s_mul_hi_i32 s0, s18, s12 // (op 26787) 
s_add_i32 s19, s0, s18 // (op 26787) 


s_ashr_i32 s0, s19, 31 // (op 26787) 
s_cmp_eq_i32 s12, 0 // (op 26787) 
s_mov_b32 s18, scc // (op 26787) 
s_cmp_eq_i32 s18, 0 // (op 26787) 
s_cselect_b32 s22, 0, -1 // (op 26787) 

s_and_b32 s18, s44, 31 // (op 26787) 
s_lshl_b32 s23, 1, s18 // (op 26787) 
s_add_i32 s24, s23, s22 // (op 26787) 


s_and_b32 s25, s0, s24 // (op 26787) 


s_add_i32 s0, s19, s25 // (op 26787) 


s_ashr_i32 s19, s0, s18 // (op 26787) 


s_lshl_b32 s0, s44, 24 // (op 26787) 
s_ashr_i32 s18, s0, 31 // (op 26787) 

s_xor_b32 s0, s19, s18 // (op 26787) 

s_sub_i32 s19, s0, s18 // (op 26787) 


s_cmp_eq_i32 s1, 0 // (op 26787) 
s_mov_b32 s0, scc // (op 26787) 

s_cmp_eq_i32 s0, 0 // (op 26787) 
s_cselect_b32 s1, s19, s47 // (op 26787) 



v_lshl_add_u32 v14, s1, 8, v12 // (op 26787) 


v_mul_lo_u32 v21, v14, s31 // (op 26787) most significant: low of low * high
v_mul_hi_u32 v12, v14, s30 // (op 26787) most significant: high of low * low
v_mul_lo_u32 v20, v14, s30 // (op 26787) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 26787) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 26787) 
s_lshr_b32 s47, s0, 1 // (op 26787) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 26787) 


v_mov_b32 v19, 0 // (op 26787) convert
v_mov_b32 v18, v12 // (op 26787) convert

v_add_co_u32 v12, s[0:1], v18, v20 // (op 26787) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 26787) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 26787) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 26787) 

s_mul_i32 s1, s30, s7 // (op 26789) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 26789) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 26789) least significant
s_mul_hi_u32 s18, s30, s6 // (op 26789) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 26789) most significant: sum
s_add_u32 s1, s1, s18 // (op 26789) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 26789) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 26789) 

s_mov_b32 s60, s32 // (op 26789) reg expression
s_mov_b32 s61, s33 // (op 26789) reg expression
s_mov_b32 s62, s0 // (op 26789) reg expression
s_mov_b32 s63, 131072 // (op 26789) call()

v_and_b32 v12, -8, v1 // (op 26885) 
v_sub_u32 v13, v1, v12 // (op 26885) 

v_lshlrev_b32 v12, 4, v13 // (op 26885) 

v_lshrrev_b32 v13, 3, v1 // (op 26885) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 26885) 


s_nop 0
v_readfirstlane_b32 s47, v16 // (op 26885) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 26942) 
v_sub_u32 v13, v1, v12 // (op 26942) 

v_lshlrev_b32 v12, 5, v13 // (op 26942) 

v_lshrrev_b32 v13, 2, v1 // (op 26942) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 26942) 


v_lshlrev_b32 v12, 2, v16 // (op 26942) 

v_lshrrev_b32 v13, 3, v12 // (op 26942) 

s_nop 0
v_readfirstlane_b32 s0, v13 // (op 26942) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s1, s16, s10 // (op 27207) 
v_and_b32 v12, 63, v1 // (op 27207) 
s_add_i32 s18, s1, s16 // (op 27207) 

v_lshrrev_b32 v13, 4, v12 // (op 27207) 
s_ashr_i32 s1, s18, 31 // (op 27207) 
v_and_b32 v16, 3, v13 // (op 27207) 

s_cmp_eq_i32 s10, 0 // (op 27207) 
s_mov_b32 s19, scc // (op 27207) 
v_and_b32 v13, 15, v1 // (op 27207) 
s_cmp_eq_i32 s19, 0 // (op 27207) 
s_cselect_b32 s22, 0, -1 // (op 27207) 

v_lshl_add_u32 v17, v16, 4, v13 // (op 27207) 


s_and_b32 s19, s13, 31 // (op 27207) 
v_and_b32 v13, 4294967168, v1 // (op 27207) 
s_lshl_b32 s23, 1, s19 // (op 27207) 
v_add_u32 v16, v13, v17 // (op 27207) 


s_add_i32 s24, s23, s22 // (op 27207) 


v_lshrrev_b32 v13, 6, v12 // (op 27207) 

s_and_b32 s25, s1, s24 // (op 27207) 


v_lshlrev_b32 v12, 2, v13 // (op 27207) 

s_add_i32 s1, s18, s25 // (op 27207) 


s_ashr_i32 s18, s1, s19 // (op 27207) 


s_lshl_b32 s1, s13, 24 // (op 27207) 
s_ashr_i32 s19, s1, 31 // (op 27207) 

s_xor_b32 s1, s18, s19 // (op 27207) 

s_sub_i32 s18, s1, s19 // (op 27207) 


s_mul_hi_i32 s1, s18, s14 // (op 27207) 
s_add_i32 s19, s1, s18 // (op 27207) 

s_ashr_i32 s1, s19, 31 // (op 27207) 
s_cmp_eq_i32 s14, 0 // (op 27207) 
s_mov_b32 s22, scc // (op 27207) 
s_cmp_eq_i32 s22, 0 // (op 27207) 
s_cselect_b32 s23, 0, -1 // (op 27207) 

s_and_b32 s22, s15, 31 // (op 27207) 
s_lshl_b32 s24, 1, s22 // (op 27207) 
s_add_i32 s25, s24, s23 // (op 27207) 


s_and_b32 s23, s1, s25 // (op 27207) 


s_add_i32 s1, s19, s23 // (op 27207) 


s_ashr_i32 s19, s1, s22 // (op 27207) 


s_lshl_b32 s1, s15, 24 // (op 27207) 
s_ashr_i32 s22, s1, 31 // (op 27207) 

s_xor_b32 s1, s19, s22 // (op 27207) 

s_sub_i32 s19, s1, s22 // (op 27207) 


s_mul_i32 s1, s18, s11 // (op 27207) 

s_sub_u32 s18, s16, s1 // (op 27207) 

s_mul_i32 s1, s19, s11 // (op 27207) 
s_add_u32 s1, s1, s18 // (op 27207) 


s_mul_hi_i32 s18, s16, s9 // (op 27207) 
s_add_i32 s19, s18, s16 // (op 27207) 

s_ashr_i32 s18, s19, 31 // (op 27207) 
s_cmp_eq_i32 s9, 0 // (op 27207) 
s_mov_b32 s22, scc // (op 27207) 
s_cmp_eq_i32 s22, 0 // (op 27207) 
s_cselect_b32 s23, 0, -1 // (op 27207) 

s_and_b32 s22, s45, 31 // (op 27207) 
s_lshl_b32 s24, 1, s22 // (op 27207) 
s_add_i32 s25, s24, s23 // (op 27207) 


s_and_b32 s23, s18, s25 // (op 27207) 


s_add_i32 s18, s19, s23 // (op 27207) 


s_ashr_i32 s19, s18, s22 // (op 27207) 


s_lshl_b32 s18, s45, 24 // (op 27207) 
s_ashr_i32 s22, s18, 31 // (op 27207) 

s_xor_b32 s18, s19, s22 // (op 27207) 

s_sub_i32 s19, s18, s22 // (op 27207) 


s_add_u32 s22, s6, 256 // (op 27207) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27207) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27207) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27207) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27207) 

s_mul_i32 s18, s11, s22 // (op 27207) 

s_mul_i32 s22, s8, s18 // (op 27207) 

s_mul_i32 s18, s19, s22 // (op 27207) 

s_sub_u32 s22, s16, s18 // (op 27207) 

s_mov_b32 s18, s22 // (op 27207) 
s_mul_hi_i32 s23, s18, s12 // (op 27207) 
s_add_i32 s24, s23, s18 // (op 27207) 


s_ashr_i32 s18, s24, 31 // (op 27207) 
s_cmp_eq_i32 s12, 0 // (op 27207) 
s_mov_b32 s23, scc // (op 27207) 
s_cmp_eq_i32 s23, 0 // (op 27207) 
s_cselect_b32 s25, 0, -1 // (op 27207) 

s_and_b32 s23, s44, 31 // (op 27207) 
s_lshl_b32 s64, 1, s23 // (op 27207) 
s_add_i32 s65, s64, s25 // (op 27207) 


s_and_b32 s25, s18, s65 // (op 27207) 


s_add_i32 s18, s24, s25 // (op 27207) 


s_ashr_i32 s24, s18, s23 // (op 27207) 


s_lshl_b32 s18, s44, 24 // (op 27207) 
s_ashr_i32 s23, s18, 31 // (op 27207) 

s_xor_b32 s18, s24, s23 // (op 27207) 

s_sub_i32 s24, s18, s23 // (op 27207) 


s_mul_i32 s18, s46, s11 // (op 27207) 
s_add_u32 s64, s20, 256 // (op 27207) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 27207) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 27207) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 27207) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 8 // (op 27207) 

s_sub_i32 s23, s64, s18 // (op 27207) 

s_mul_i32 s25, s24, s23 // (op 27207) 


s_sub_u32 s23, s22, s25 // (op 27207) 


s_add_u32 s22, s23, s18 // (op 27207) 


s_cmp_eq_i32 s19, 0 // (op 27207) 
s_mov_b32 s18, scc // (op 27207) 

s_cmp_eq_i32 s18, 0 // (op 27207) 
s_cselect_b32 s19, s22, s1 // (op 27207) 



v_lshl_add_u32 v13, s19, 8, v16 // (op 27207) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 27207) 
s_lshr_b32 s1, s18, 1 // (op 27207) 

v_lshl_add_u32 v16, s1, 3, v12 // (op 27207) 


v_mul_lo_u32 v12, v13, s34 // (op 27207) 
v_add_u32 v12, v12, v16 // (op 27207) 


s_mul_i32 s19, s34, s21 // (op 27209) most significant: low of low * high
s_mul_i32 s1, s35, s20 // (op 27209) most significant: low of high * low
s_mul_i32 s18, s34, s20 // (op 27209) least significant
s_mul_hi_u32 s22, s34, s20 // (op 27209) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 27209) most significant: sum
s_add_u32 s19, s19, s22 // (op 27209) most significant: sum


s_mov_b32 s64, s36 // (op 27209) reg expression
s_mov_b32 s65, s37 // (op 27209) reg expression
s_mov_b32 s66, s18 // (op 27209) reg expression
s_mov_b32 s67, 131072 // (op 27209) call()

s_mul_hi_i32 s1, s16, s10 // (op 27234) 
v_and_b32 v13, 63, v1 // (op 27234) 
s_add_i32 s18, s1, s16 // (op 27234) 

v_lshrrev_b32 v16, 4, v13 // (op 27234) 
s_ashr_i32 s1, s18, 31 // (op 27234) 
v_and_b32 v17, 3, v16 // (op 27234) 

s_cmp_eq_i32 s10, 0 // (op 27234) 
s_mov_b32 s19, scc // (op 27234) 
v_and_b32 v16, 15, v1 // (op 27234) 
s_cmp_eq_i32 s19, 0 // (op 27234) 
s_cselect_b32 s22, 0, -1 // (op 27234) 

v_lshl_add_u32 v18, v17, 4, v16 // (op 27234) 


s_and_b32 s19, s13, 31 // (op 27234) 
v_lshrrev_b32 v16, 6, v1 // (op 27234) 
s_lshl_b32 s23, 1, s19 // (op 27234) 
v_and_b32 v17, 1, v16 // (op 27234) 

s_add_i32 s24, s23, s22 // (op 27234) 


v_lshl_add_u32 v16, v17, 7, v18 // (op 27234) 


s_and_b32 s25, s1, s24 // (op 27234) 


v_lshrrev_b32 v17, 6, v13 // (op 27234) 

s_add_i32 s1, s18, s25 // (op 27234) 


v_lshlrev_b32 v13, 2, v17 // (op 27234) 

s_ashr_i32 s18, s1, s19 // (op 27234) 


s_lshl_b32 s1, s13, 24 // (op 27234) 
s_ashr_i32 s19, s1, 31 // (op 27234) 

s_xor_b32 s1, s18, s19 // (op 27234) 

s_sub_i32 s18, s1, s19 // (op 27234) 


s_mul_hi_i32 s1, s18, s14 // (op 27234) 
s_add_i32 s19, s1, s18 // (op 27234) 

s_ashr_i32 s1, s19, 31 // (op 27234) 
s_cmp_eq_i32 s14, 0 // (op 27234) 
s_mov_b32 s22, scc // (op 27234) 
s_cmp_eq_i32 s22, 0 // (op 27234) 
s_cselect_b32 s23, 0, -1 // (op 27234) 

s_and_b32 s22, s15, 31 // (op 27234) 
s_lshl_b32 s24, 1, s22 // (op 27234) 
s_add_i32 s25, s24, s23 // (op 27234) 


s_and_b32 s23, s1, s25 // (op 27234) 


s_add_i32 s1, s19, s23 // (op 27234) 


s_ashr_i32 s19, s1, s22 // (op 27234) 


s_lshl_b32 s1, s15, 24 // (op 27234) 
s_ashr_i32 s22, s1, 31 // (op 27234) 

s_xor_b32 s1, s19, s22 // (op 27234) 

s_sub_i32 s19, s1, s22 // (op 27234) 


s_add_u32 s22, s6, 256 // (op 27234) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 27234) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 27234) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 27234) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 27234) 

s_mul_i32 s1, s19, s22 // (op 27234) 

s_sub_i32 s19, s18, s1 // (op 27234) 


s_mul_hi_i32 s1, s16, s9 // (op 27234) 
s_add_i32 s18, s1, s16 // (op 27234) 

s_ashr_i32 s1, s18, 31 // (op 27234) 
s_cmp_eq_i32 s9, 0 // (op 27234) 
s_mov_b32 s24, scc // (op 27234) 
s_cmp_eq_i32 s24, 0 // (op 27234) 
s_cselect_b32 s25, 0, -1 // (op 27234) 

s_and_b32 s24, s45, 31 // (op 27234) 
s_lshl_b32 s68, 1, s24 // (op 27234) 
s_add_i32 s69, s68, s25 // (op 27234) 


s_and_b32 s25, s1, s69 // (op 27234) 


s_add_i32 s1, s18, s25 // (op 27234) 


s_ashr_i32 s18, s1, s24 // (op 27234) 


s_lshl_b32 s1, s45, 24 // (op 27234) 
s_ashr_i32 s24, s1, 31 // (op 27234) 

s_xor_b32 s1, s18, s24 // (op 27234) 

s_sub_i32 s18, s1, s24 // (op 27234) 


s_mul_i32 s1, s11, s22 // (op 27234) 

s_mul_i32 s22, s8, s1 // (op 27234) 

s_mul_i32 s1, s18, s22 // (op 27234) 

s_sub_u32 s22, s16, s1 // (op 27234) 

s_mul_hi_i32 s1, s22, s12 // (op 27234) 
s_add_i32 s23, s1, s22 // (op 27234) 


s_ashr_i32 s1, s23, 31 // (op 27234) 
s_cmp_eq_i32 s12, 0 // (op 27234) 
s_mov_b32 s22, scc // (op 27234) 
s_cmp_eq_i32 s22, 0 // (op 27234) 
s_cselect_b32 s24, 0, -1 // (op 27234) 

s_and_b32 s22, s44, 31 // (op 27234) 
s_lshl_b32 s25, 1, s22 // (op 27234) 
s_add_i32 s68, s25, s24 // (op 27234) 


s_and_b32 s24, s1, s68 // (op 27234) 


s_add_i32 s1, s23, s24 // (op 27234) 


s_ashr_i32 s23, s1, s22 // (op 27234) 


s_lshl_b32 s1, s44, 24 // (op 27234) 
s_ashr_i32 s22, s1, 31 // (op 27234) 

s_xor_b32 s1, s23, s22 // (op 27234) 

s_sub_i32 s23, s1, s22 // (op 27234) 


s_cmp_eq_i32 s18, 0 // (op 27234) 
s_mov_b32 s1, scc // (op 27234) 

s_cmp_eq_i32 s1, 0 // (op 27234) 
s_cselect_b32 s18, s23, s19 // (op 27234) 



v_lshl_add_u32 v17, s18, 8, v16 // (op 27234) 


v_mul_lo_u32 v16, v17, s38 // (op 27234) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 27234) 
s_lshr_b32 s1, s18, 1 // (op 27234) 

v_lshl_add_u32 v17, s1, 3, v13 // (op 27234) 


v_add_u32 v13, v17, v16 // (op 27234) 


s_mul_i32 s19, s38, s7 // (op 27236) most significant: low of low * high
s_mul_i32 s1, s39, s6 // (op 27236) most significant: low of high * low
s_mul_i32 s18, s38, s6 // (op 27236) least significant
s_mul_hi_u32 s22, s38, s6 // (op 27236) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 27236) most significant: sum
s_add_u32 s19, s19, s22 // (op 27236) most significant: sum


s_mov_b32 s68, s40 // (op 27236) reg expression
s_mov_b32 s69, s41 // (op 27236) reg expression
s_mov_b32 s70, s18 // (op 27236) reg expression
s_mov_b32 s71, 131072 // (op 27236) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 2819) 
s_and_b32 s1, s18, 4294967294 // (op 2819) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 2818) 
s_mov_b32 s22, s18 // (op 2818) convert

s_cmp_lt_u32 s1, s22 // (op 2818) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_2818 // (op 2818) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_2818 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_2818:
s_barrier  // (op 27587) 
s_add_u32 s1, s1, 1 // (op 2820) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 26610) 
v_mov_b64 v[16:17], s[18:19] // (op 26610) convert


v_mov_b64 v[18:19], v[10:11] // (op 26610) 
v_mov_b64 v[20:21], v[18:19] // (op 26610) 
s_add_u32 s18, 16384, s47 // (op 26610) 
s_mov_b32 m0, s18 // (op 26610) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26610) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26610) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26610) most significant half

s_add_u32 m0, m0, 4096 // (op 26610) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26610) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26610) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26610) most significant half

s_add_u32 m0, m0, 4096 // (op 26610) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26610) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26610) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26610) most significant half

s_add_u32 m0, m0, 4096 // (op 26610) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26610) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26610) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26610) most significant half

s_add_u32 m0, m0, 4096 // (op 26610) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26610) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26610) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26610) most significant half

s_add_u32 m0, m0, 4096 // (op 26610) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26610) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26610) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26610) most significant half

s_add_u32 m0, m0, 4096 // (op 26610) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26610) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v16 // (op 26610) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 26610) most significant half

s_add_u32 m0, m0, 4096 // (op 26610) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 26610) Load value direct to lds




v_add_co_u32 v10, vcc, 128, v10 // (op 26647) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 26647) most significant half
s_barrier  // (op 27603) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 26616) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 26616) 

v_lshrrev_b64 v[16:17], 3, s[22:23] // (op 26616) 


v_mov_b64 v[18:19], v[14:15] // (op 26616) 
v_mov_b64 v[20:21], v[18:19] // (op 26616) 
s_mov_b32 m0, s0 // (op 26616) reg expression
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 26616) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 26616) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 26616) most significant half

s_add_u32 m0, m0, 4096 // (op 26616) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 26616) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 26616) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 26616) most significant half

s_add_u32 m0, m0, 4096 // (op 26616) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 26616) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 26616) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 26616) most significant half

s_add_u32 m0, m0, 4096 // (op 26616) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 26616) Load value direct to lds



v_add_co_u32 v14, s[18:19], 64, v14 // (op 26779) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 26779) most significant half

s_waitcnt vmcnt(4)
s_barrier  // (op 27593) 



ds_read_b128 v[16:19], v3 offset:16384 // (op 160) Load local data 
ds_read_b128 v[20:23], v3 offset:16448 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 27609) 



ds_read_b128 v[24:27], v4 // (op 162) Load local data 


v_mov_b32 v28, v12 // (op 164) 
v_mov_b32 v29, v28 // (op 164) reg expression

buffer_load_dwordx2 v[30:31], v29, s[64:67], 0 offen // (op 164) Load value



v_mov_b32 v29, v13 // (op 166) 
v_mov_b32 v32, v29 // (op 166) reg expression

buffer_load_dwordx2 v[34:35], v32, s[68:71], 0 offen // (op 166) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v36, v30, 0, 8 // (op 24979) 



v_bfe_u32 v37, v30, 8, 8 // (op 24979) 
v_bfe_u32 v38, v30, 16, 8 // (op 24979) 
v_bfe_u32 v39, v30, 24, 8 // (op 24979) 
v_permlane16_swap_b32 v36, v37 // (op 24979) 
s_nop 0
v_permlane16_swap_b32 v38, v39 // (op 24979) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v36, v38 // (op 24979) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v37, v39 // (op 24979) 
ds_read_b128 v[40:43], v4 offset:1024 // (op 291) Load local data 




ds_read_b128 v[30:33], v4 offset:2048 // (op 427) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v44, v34, 0, 8 // (op 25071) 



v_bfe_u32 v45, v34, 8, 8 // (op 25071) 
v_bfe_u32 v46, v34, 16, 8 // (op 25071) 
v_bfe_u32 v47, v34, 24, 8 // (op 25071) 
v_permlane16_swap_b32 v44, v45 // (op 25071) 
ds_read_b128 v[48:51], v4 offset:3072 // (op 563) Load local data 
v_permlane16_swap_b32 v46, v47 // (op 25071) 



s_nop 1
v_permlane32_swap_b32 v44, v46 // (op 25071) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v45, v47 // (op 25071) 
ds_read_b128 v[52:55], v4 offset:4096 // (op 699) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:23], v[24:27], a[0:3], v36, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 167) 
s_lshl_b32 s18, s38, 6 // (op 704) 
v_add_u32 v34, v29, s18 // (op 704) 


buffer_load_dwordx2 v[56:57], v34, s[68:71], 0 offen // (op 704) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:23], v[40:43], a[4:7], v36, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 294) 




v_add_u32 v13, 4, v13 // (op 27224) 
ds_read_b128 v[58:61], v4 offset:5120 // (op 835) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:23], v[30:33], a[8:11], v36, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 430) 



ds_read_b128 v[62:65], v4 offset:6144 // (op 971) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:23], v[48:51], a[12:15], v36, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 566) 



s_waitcnt vmcnt(0)
v_bfe_u32 v66, v56, 0, 8 // (op 25085) 
v_bfe_u32 v67, v56, 8, 8 // (op 25085) 
v_bfe_u32 v68, v56, 16, 8 // (op 25085) 
v_bfe_u32 v69, v56, 24, 8 // (op 25085) 
v_permlane16_swap_b32 v66, v67 // (op 25085) 
s_nop 0
v_permlane16_swap_b32 v68, v69 // (op 25085) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v66, v68 // (op 25085) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 25085) 
ds_read_b128 v[70:73], v4 offset:7168 // (op 1107) Load local data 




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[16:23], v[52:55], a[16:19], v36, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 702) 
ds_read_b128 v[74:77], v3 offset:18432 // (op 1417) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[16:23], v[58:61], a[20:23], v36, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 838) 
ds_read_b128 v[78:81], v3 offset:18496 // (op 1417) Load local data 



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:23], v[62:65], a[24:27], v36, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 974) 
ds_read_b128 v[82:85], v3 offset:20480 // (op 2635) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[16:23], v[70:73], a[28:31], v36, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1110) 
ds_read_b128 v[86:89], v3 offset:20544 // (op 2635) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:81], v[24:27], a[32:35], v37, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1421) 
ds_read_b128 v[16:19], v3 offset:22528 // (op 3853) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:81], v[30:33], a[40:43], v37, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1475) 
ds_read_b128 v[20:23], v3 offset:22592 // (op 3853) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:81], v[48:51], a[44:47], v37, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1502) 
ds_read_b128 v[90:93], v3 offset:24576 // (op 5071) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:81], v[40:43], a[36:39], v37, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1448) 
ds_read_b128 v[94:97], v3 offset:24640 // (op 5071) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:81], v[52:55], a[48:51], v37, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1529) 
s_lshl_b32 s18, s34, 6 // (op 5080) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:81], v[62:65], a[56:59], v37, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1583) 
v_add_u32 v29, v28, s18 // (op 5080) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:81], v[70:73], a[60:63], v37, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1610) 
buffer_load_dwordx2 v[34:35], v29, s[64:67], 0 offen // (op 5080) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:81], v[58:61], a[52:55], v37, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1556) 




v_add_u32 v12, 4, v12 // (op 27197) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[82:89], v[24:27], a[64:67], v38, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2639) 
ds_read_b128 v[74:77], v3 offset:26624 // (op 6289) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:89], v[30:33], a[72:75], v38, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2693) 
ds_read_b128 v[78:81], v3 offset:26688 // (op 6289) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[82:89], v[48:51], a[76:79], v38, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2720) 
ds_read_b128 v[98:101], v3 offset:28672 // (op 7507) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:89], v[40:43], a[68:71], v38, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2666) 
s_waitcnt vmcnt(0)
v_bfe_u32 v106, v34, 0, 8 // (op 24993) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[82:89], v[52:55], a[80:83], v38, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2747) 
v_bfe_u32 v107, v34, 8, 8 // (op 24993) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[82:89], v[62:65], a[88:91], v38, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2801) 
v_bfe_u32 v108, v34, 16, 8 // (op 24993) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[82:89], v[70:73], a[92:95], v38, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2828) 
v_bfe_u32 v109, v34, 24, 8 // (op 24993) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[82:89], v[58:61], a[84:87], v38, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 2774) 
v_permlane16_swap_b32 v106, v107 // (op 24993) 

v_permlane16_swap_b32 v108, v109 // (op 24993) 
s_nop 1
v_permlane32_swap_b32 v106, v108 // (op 24993) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 24993) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[16:23], v[24:27], a[96:99], v39, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3857) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[16:23], v[30:33], a[104:107], v39, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3911) 
ds_read_b128 v[102:105], v3 offset:28736 // (op 7507) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[16:23], v[48:51], a[108:111], v39, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3938) 
ds_read_b128 v[82:85], v3 offset:30720 // (op 8725) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[16:23], v[40:43], a[100:103], v39, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3884) 
ds_read_b128 v[86:89], v3 offset:30784 // (op 8725) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[16:23], v[52:55], a[112:115], v39, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3965) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[16:23], v[62:65], a[120:123], v39, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4019) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[16:23], v[70:73], a[124:127], v39, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 4046) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[16:23], v[58:61], a[116:119], v39, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 3992) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[90:97], v[40:43], a[132:135], v106, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5102) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[90:97], v[58:61], a[148:151], v106, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5210) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[90:97], v[24:27], a[128:131], v106, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5075) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:97], v[30:33], a[136:139], v106, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5129) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[90:97], v[48:51], a[140:143], v106, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5156) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[90:97], v[52:55], a[144:147], v106, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5183) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[90:97], v[62:65], a[152:155], v106, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5237) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:97], v[70:73], a[156:159], v106, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 5264) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[74:81], v[40:43], a[164:167], v107, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6320) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[74:81], v[58:61], a[180:183], v107, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6428) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[74:81], v[24:27], a[160:163], v107, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6293) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:81], v[30:33], a[168:171], v107, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6347) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:81], v[48:51], a[172:175], v107, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6374) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[74:81], v[52:55], a[176:179], v107, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6401) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[74:81], v[62:65], a[184:187], v107, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6455) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[74:81], v[70:73], a[188:191], v107, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 6482) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[98:105], v[40:43], a[196:199], v108, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7538) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[98:105], v[58:61], a[212:215], v108, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7646) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[98:105], v[24:27], a[192:195], v108, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7511) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[98:105], v[30:33], a[200:203], v108, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7565) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[98:105], v[48:51], a[204:207], v108, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7592) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[98:105], v[52:55], a[208:211], v108, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7619) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[98:105], v[62:65], a[216:219], v108, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7673) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[98:105], v[70:73], a[220:223], v108, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 7700) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[82:89], v[40:43], a[228:231], v109, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8756) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[82:89], v[58:61], a[244:247], v109, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8864) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[82:89], v[24:27], a[224:227], v109, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8729) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[82:89], v[30:33], a[232:235], v109, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8783) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[82:89], v[48:51], a[236:239], v109, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8810) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[82:89], v[52:55], a[240:243], v109, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8837) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[82:89], v[62:65], a[248:251], v109, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8891) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[82:89], v[70:73], a[252:255], v109, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 8918) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 2818) 
s_mov_b32 s22, s18 // (op 2818) convert

s_cmp_lt_u32 s1, s22 // (op 2818) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_2818 // (op 2818) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopTop_KLoopTail_2818 if true)
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_2818:




















s_barrier  // (op 27665) 

















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_114
v_accvgpr_read v10, a0 // (op 42) 
v_accvgpr_read v11, a1 // (op 42) 
v_accvgpr_read v12, a2 // (op 42) 
v_accvgpr_read v13, a3 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 
v_mul_f32 v16, v7, v12 // (op 42) 
v_mul_f32 v17, v7, v13 // (op 42) 



v_mov_b32 v3, v9 // (op 48) reg expression

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 48) Store value


v_accvgpr_read v10, a4 // (op 11110) 
v_accvgpr_read v11, a5 // (op 11110) 
v_accvgpr_read v12, a6 // (op 11110) 
v_accvgpr_read v13, a7 // (op 11110) 
v_mul_f32 v14, v7, v10 // (op 11110) 
v_mul_f32 v15, v7, v11 // (op 11110) 
v_mul_f32 v16, v7, v12 // (op 11110) 
v_mul_f32 v17, v7, v13 // (op 11110) 



s_lshl_b32 s0, s4, 6 // (op 11112) 
v_add_u32 v3, v9, s0 // (op 11112) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11112) Store value


v_accvgpr_read v10, a8 // (op 11122) 
v_accvgpr_read v11, a9 // (op 11122) 
v_accvgpr_read v12, a10 // (op 11122) 
v_accvgpr_read v13, a11 // (op 11122) 
v_mul_f32 v14, v7, v10 // (op 11122) 
v_mul_f32 v15, v7, v11 // (op 11122) 
v_mul_f32 v16, v7, v12 // (op 11122) 
v_mul_f32 v17, v7, v13 // (op 11122) 



s_lshl_b32 s0, s4, 7 // (op 11124) 
v_add_u32 v3, v9, s0 // (op 11124) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11124) Store value


v_accvgpr_read v10, a12 // (op 11134) 
v_accvgpr_read v11, a13 // (op 11134) 
v_accvgpr_read v12, a14 // (op 11134) 
v_accvgpr_read v13, a15 // (op 11134) 
v_mul_f32 v14, v7, v10 // (op 11134) 
v_mul_f32 v15, v7, v11 // (op 11134) 
v_mul_f32 v16, v7, v12 // (op 11134) 
v_mul_f32 v17, v7, v13 // (op 11134) 



s_lshl_b32 s0, s4, 6 // (op 11136) 
s_mul_i32 s1, 3, s0 // (op 11136) 

v_add_u32 v3, v9, s1 // (op 11136) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11136) Store value


v_accvgpr_read v10, a16 // (op 11146) 
v_accvgpr_read v11, a17 // (op 11146) 
v_accvgpr_read v12, a18 // (op 11146) 
v_accvgpr_read v13, a19 // (op 11146) 
v_mul_f32 v14, v7, v10 // (op 11146) 
v_mul_f32 v15, v7, v11 // (op 11146) 
v_mul_f32 v16, v7, v12 // (op 11146) 
v_mul_f32 v17, v7, v13 // (op 11146) 



s_lshl_b32 s0, s4, 8 // (op 11148) 
v_add_u32 v3, v9, s0 // (op 11148) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11148) Store value


v_accvgpr_read v10, a20 // (op 11158) 
v_accvgpr_read v11, a21 // (op 11158) 
v_accvgpr_read v12, a22 // (op 11158) 
v_accvgpr_read v13, a23 // (op 11158) 
v_mul_f32 v14, v7, v10 // (op 11158) 
v_mul_f32 v15, v7, v11 // (op 11158) 
v_mul_f32 v16, v7, v12 // (op 11158) 
v_mul_f32 v17, v7, v13 // (op 11158) 



s_lshl_b32 s0, s4, 6 // (op 11160) 
s_mul_i32 s1, 5, s0 // (op 11160) 

v_add_u32 v3, v9, s1 // (op 11160) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11160) Store value


v_accvgpr_read v10, a24 // (op 11170) 
v_accvgpr_read v11, a25 // (op 11170) 
v_accvgpr_read v12, a26 // (op 11170) 
v_accvgpr_read v13, a27 // (op 11170) 
v_mul_f32 v14, v7, v10 // (op 11170) 
v_mul_f32 v15, v7, v11 // (op 11170) 
v_mul_f32 v16, v7, v12 // (op 11170) 
v_mul_f32 v17, v7, v13 // (op 11170) 



s_lshl_b32 s0, s4, 6 // (op 11172) 
s_mul_i32 s1, 6, s0 // (op 11172) 

v_add_u32 v3, v9, s1 // (op 11172) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11172) Store value


v_accvgpr_read v10, a28 // (op 11182) 
v_accvgpr_read v11, a29 // (op 11182) 
v_accvgpr_read v12, a30 // (op 11182) 
v_accvgpr_read v13, a31 // (op 11182) 
v_mul_f32 v14, v7, v10 // (op 11182) 
v_mul_f32 v15, v7, v11 // (op 11182) 
v_mul_f32 v16, v7, v12 // (op 11182) 
v_mul_f32 v17, v7, v13 // (op 11182) 



s_lshl_b32 s0, s4, 6 // (op 11184) 
s_mul_i32 s1, 7, s0 // (op 11184) 

v_add_u32 v3, v9, s1 // (op 11184) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11184) Store value


v_accvgpr_read v10, a32 // (op 11236) 
v_accvgpr_read v11, a33 // (op 11236) 
v_accvgpr_read v12, a34 // (op 11236) 
v_accvgpr_read v13, a35 // (op 11236) 
v_mul_f32 v14, v7, v10 // (op 11236) 
v_mul_f32 v15, v7, v11 // (op 11236) 
v_mul_f32 v16, v7, v12 // (op 11236) 
v_mul_f32 v17, v7, v13 // (op 11236) 



v_add_u32 v3, 64, v9 // (op 11239) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11239) Store value


v_accvgpr_read v10, a36 // (op 11257) 
v_accvgpr_read v11, a37 // (op 11257) 
v_accvgpr_read v12, a38 // (op 11257) 
v_accvgpr_read v13, a39 // (op 11257) 
v_mul_f32 v14, v7, v10 // (op 11257) 
v_mul_f32 v15, v7, v11 // (op 11257) 
v_mul_f32 v16, v7, v12 // (op 11257) 
v_mul_f32 v17, v7, v13 // (op 11257) 



s_lshl_b32 s0, s4, 6 // (op 11241) 
s_add_u32 s0, s0, 64 // (op 11241) 
v_add_u32 v3, v9, s0 // (op 11241) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11241) Store value


v_accvgpr_read v10, a40 // (op 11262) 
v_accvgpr_read v11, a41 // (op 11262) 
v_accvgpr_read v12, a42 // (op 11262) 
v_accvgpr_read v13, a43 // (op 11262) 
v_mul_f32 v14, v7, v10 // (op 11262) 
v_mul_f32 v15, v7, v11 // (op 11262) 
v_mul_f32 v16, v7, v12 // (op 11262) 
v_mul_f32 v17, v7, v13 // (op 11262) 



s_lshl_b32 s0, s4, 7 // (op 11243) 
s_add_u32 s0, s0, 64 // (op 11243) 
v_add_u32 v3, v9, s0 // (op 11243) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11243) Store value


v_accvgpr_read v10, a44 // (op 11267) 
v_accvgpr_read v11, a45 // (op 11267) 
v_accvgpr_read v12, a46 // (op 11267) 
v_accvgpr_read v13, a47 // (op 11267) 
v_mul_f32 v14, v7, v10 // (op 11267) 
v_mul_f32 v15, v7, v11 // (op 11267) 
v_mul_f32 v16, v7, v12 // (op 11267) 
v_mul_f32 v17, v7, v13 // (op 11267) 



s_lshl_b32 s0, s4, 6 // (op 11245) 
s_mul_i32 s1, 3, s0 // (op 11245) 
s_add_u32 s1, s1, 64 // (op 11245) 

v_add_u32 v3, v9, s1 // (op 11245) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11245) Store value


v_accvgpr_read v10, a48 // (op 11272) 
v_accvgpr_read v11, a49 // (op 11272) 
v_accvgpr_read v12, a50 // (op 11272) 
v_accvgpr_read v13, a51 // (op 11272) 
v_mul_f32 v14, v7, v10 // (op 11272) 
v_mul_f32 v15, v7, v11 // (op 11272) 
v_mul_f32 v16, v7, v12 // (op 11272) 
v_mul_f32 v17, v7, v13 // (op 11272) 



s_lshl_b32 s0, s4, 8 // (op 11247) 
s_add_u32 s0, s0, 64 // (op 11247) 
v_add_u32 v3, v9, s0 // (op 11247) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11247) Store value


v_accvgpr_read v10, a52 // (op 11277) 
v_accvgpr_read v11, a53 // (op 11277) 
v_accvgpr_read v12, a54 // (op 11277) 
v_accvgpr_read v13, a55 // (op 11277) 
v_mul_f32 v14, v7, v10 // (op 11277) 
v_mul_f32 v15, v7, v11 // (op 11277) 
v_mul_f32 v16, v7, v12 // (op 11277) 
v_mul_f32 v17, v7, v13 // (op 11277) 



s_lshl_b32 s0, s4, 6 // (op 11249) 
s_mul_i32 s1, 5, s0 // (op 11249) 
s_add_u32 s1, s1, 64 // (op 11249) 

v_add_u32 v3, v9, s1 // (op 11249) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11249) Store value


v_accvgpr_read v10, a56 // (op 11282) 
v_accvgpr_read v11, a57 // (op 11282) 
v_accvgpr_read v12, a58 // (op 11282) 
v_accvgpr_read v13, a59 // (op 11282) 
v_mul_f32 v14, v7, v10 // (op 11282) 
v_mul_f32 v15, v7, v11 // (op 11282) 
v_mul_f32 v16, v7, v12 // (op 11282) 
v_mul_f32 v17, v7, v13 // (op 11282) 



s_lshl_b32 s0, s4, 6 // (op 11251) 
s_mul_i32 s1, 6, s0 // (op 11251) 
s_add_u32 s1, s1, 64 // (op 11251) 

v_add_u32 v3, v9, s1 // (op 11251) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11251) Store value


v_accvgpr_read v10, a60 // (op 11287) 
v_accvgpr_read v11, a61 // (op 11287) 
v_accvgpr_read v12, a62 // (op 11287) 
v_accvgpr_read v13, a63 // (op 11287) 
v_mul_f32 v14, v7, v10 // (op 11287) 
v_mul_f32 v15, v7, v11 // (op 11287) 
v_mul_f32 v16, v7, v12 // (op 11287) 
v_mul_f32 v17, v7, v13 // (op 11287) 



s_lshl_b32 s0, s4, 6 // (op 11253) 
s_mul_i32 s1, 7, s0 // (op 11253) 
s_add_u32 s1, s1, 64 // (op 11253) 

v_add_u32 v3, v9, s1 // (op 11253) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11253) Store value


v_accvgpr_read v10, a64 // (op 11368) 
v_accvgpr_read v11, a65 // (op 11368) 
v_accvgpr_read v12, a66 // (op 11368) 
v_accvgpr_read v13, a67 // (op 11368) 
v_mul_f32 v14, v7, v10 // (op 11368) 
v_mul_f32 v15, v7, v11 // (op 11368) 
v_mul_f32 v16, v7, v12 // (op 11368) 
v_mul_f32 v17, v7, v13 // (op 11368) 



v_add_u32 v3, 128, v9 // (op 11371) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11371) Store value


v_accvgpr_read v10, a68 // (op 11389) 
v_accvgpr_read v11, a69 // (op 11389) 
v_accvgpr_read v12, a70 // (op 11389) 
v_accvgpr_read v13, a71 // (op 11389) 
v_mul_f32 v14, v7, v10 // (op 11389) 
v_mul_f32 v15, v7, v11 // (op 11389) 
v_mul_f32 v16, v7, v12 // (op 11389) 
v_mul_f32 v17, v7, v13 // (op 11389) 



s_lshl_b32 s0, s4, 6 // (op 11373) 
s_add_u32 s0, s0, 128 // (op 11373) 
v_add_u32 v3, v9, s0 // (op 11373) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11373) Store value


v_accvgpr_read v10, a72 // (op 11394) 
v_accvgpr_read v11, a73 // (op 11394) 
v_accvgpr_read v12, a74 // (op 11394) 
v_accvgpr_read v13, a75 // (op 11394) 
v_mul_f32 v14, v7, v10 // (op 11394) 
v_mul_f32 v15, v7, v11 // (op 11394) 
v_mul_f32 v16, v7, v12 // (op 11394) 
v_mul_f32 v17, v7, v13 // (op 11394) 



s_lshl_b32 s0, s4, 7 // (op 11375) 
s_add_u32 s0, s0, 128 // (op 11375) 
v_add_u32 v3, v9, s0 // (op 11375) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11375) Store value


v_accvgpr_read v10, a76 // (op 11399) 
v_accvgpr_read v11, a77 // (op 11399) 
v_accvgpr_read v12, a78 // (op 11399) 
v_accvgpr_read v13, a79 // (op 11399) 
v_mul_f32 v14, v7, v10 // (op 11399) 
v_mul_f32 v15, v7, v11 // (op 11399) 
v_mul_f32 v16, v7, v12 // (op 11399) 
v_mul_f32 v17, v7, v13 // (op 11399) 



s_lshl_b32 s0, s4, 6 // (op 11377) 
s_mul_i32 s1, 3, s0 // (op 11377) 
s_add_u32 s1, s1, 128 // (op 11377) 

v_add_u32 v3, v9, s1 // (op 11377) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11377) Store value


v_accvgpr_read v10, a80 // (op 11404) 
v_accvgpr_read v11, a81 // (op 11404) 
v_accvgpr_read v12, a82 // (op 11404) 
v_accvgpr_read v13, a83 // (op 11404) 
v_mul_f32 v14, v7, v10 // (op 11404) 
v_mul_f32 v15, v7, v11 // (op 11404) 
v_mul_f32 v16, v7, v12 // (op 11404) 
v_mul_f32 v17, v7, v13 // (op 11404) 



s_lshl_b32 s0, s4, 8 // (op 11379) 
s_add_u32 s0, s0, 128 // (op 11379) 
v_add_u32 v3, v9, s0 // (op 11379) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11379) Store value


v_accvgpr_read v10, a84 // (op 11409) 
v_accvgpr_read v11, a85 // (op 11409) 
v_accvgpr_read v12, a86 // (op 11409) 
v_accvgpr_read v13, a87 // (op 11409) 
v_mul_f32 v14, v7, v10 // (op 11409) 
v_mul_f32 v15, v7, v11 // (op 11409) 
v_mul_f32 v16, v7, v12 // (op 11409) 
v_mul_f32 v17, v7, v13 // (op 11409) 



s_lshl_b32 s0, s4, 6 // (op 11381) 
s_mul_i32 s1, 5, s0 // (op 11381) 
s_add_u32 s1, s1, 128 // (op 11381) 

v_add_u32 v3, v9, s1 // (op 11381) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11381) Store value


v_accvgpr_read v10, a88 // (op 11414) 
v_accvgpr_read v11, a89 // (op 11414) 
v_accvgpr_read v12, a90 // (op 11414) 
v_accvgpr_read v13, a91 // (op 11414) 
v_mul_f32 v14, v7, v10 // (op 11414) 
v_mul_f32 v15, v7, v11 // (op 11414) 
v_mul_f32 v16, v7, v12 // (op 11414) 
v_mul_f32 v17, v7, v13 // (op 11414) 



s_lshl_b32 s0, s4, 6 // (op 11383) 
s_mul_i32 s1, 6, s0 // (op 11383) 
s_add_u32 s1, s1, 128 // (op 11383) 

v_add_u32 v3, v9, s1 // (op 11383) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11383) Store value


v_accvgpr_read v10, a92 // (op 11419) 
v_accvgpr_read v11, a93 // (op 11419) 
v_accvgpr_read v12, a94 // (op 11419) 
v_accvgpr_read v13, a95 // (op 11419) 
v_mul_f32 v14, v7, v10 // (op 11419) 
v_mul_f32 v15, v7, v11 // (op 11419) 
v_mul_f32 v16, v7, v12 // (op 11419) 
v_mul_f32 v17, v7, v13 // (op 11419) 



s_lshl_b32 s0, s4, 6 // (op 11385) 
s_mul_i32 s1, 7, s0 // (op 11385) 
s_add_u32 s1, s1, 128 // (op 11385) 

v_add_u32 v3, v9, s1 // (op 11385) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11385) Store value


v_accvgpr_read v10, a96 // (op 11500) 
v_accvgpr_read v11, a97 // (op 11500) 
v_accvgpr_read v12, a98 // (op 11500) 
v_accvgpr_read v13, a99 // (op 11500) 
v_mul_f32 v14, v7, v10 // (op 11500) 
v_mul_f32 v15, v7, v11 // (op 11500) 
v_mul_f32 v16, v7, v12 // (op 11500) 
v_mul_f32 v17, v7, v13 // (op 11500) 



v_add_u32 v3, 192, v9 // (op 11503) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11503) Store value


v_accvgpr_read v10, a100 // (op 11521) 
v_accvgpr_read v11, a101 // (op 11521) 
v_accvgpr_read v12, a102 // (op 11521) 
v_accvgpr_read v13, a103 // (op 11521) 
v_mul_f32 v14, v7, v10 // (op 11521) 
v_mul_f32 v15, v7, v11 // (op 11521) 
v_mul_f32 v16, v7, v12 // (op 11521) 
v_mul_f32 v17, v7, v13 // (op 11521) 



s_lshl_b32 s0, s4, 6 // (op 11505) 
s_add_u32 s0, s0, 192 // (op 11505) 
v_add_u32 v3, v9, s0 // (op 11505) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11505) Store value


v_accvgpr_read v10, a104 // (op 11526) 
v_accvgpr_read v11, a105 // (op 11526) 
v_accvgpr_read v12, a106 // (op 11526) 
v_accvgpr_read v13, a107 // (op 11526) 
v_mul_f32 v14, v7, v10 // (op 11526) 
v_mul_f32 v15, v7, v11 // (op 11526) 
v_mul_f32 v16, v7, v12 // (op 11526) 
v_mul_f32 v17, v7, v13 // (op 11526) 



s_lshl_b32 s0, s4, 7 // (op 11507) 
s_add_u32 s0, s0, 192 // (op 11507) 
v_add_u32 v3, v9, s0 // (op 11507) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11507) Store value


v_accvgpr_read v10, a108 // (op 11531) 
v_accvgpr_read v11, a109 // (op 11531) 
v_accvgpr_read v12, a110 // (op 11531) 
v_accvgpr_read v13, a111 // (op 11531) 
v_mul_f32 v14, v7, v10 // (op 11531) 
v_mul_f32 v15, v7, v11 // (op 11531) 
v_mul_f32 v16, v7, v12 // (op 11531) 
v_mul_f32 v17, v7, v13 // (op 11531) 



s_lshl_b32 s0, s4, 6 // (op 11509) 
s_mul_i32 s1, 3, s0 // (op 11509) 
s_add_u32 s1, s1, 192 // (op 11509) 

v_add_u32 v3, v9, s1 // (op 11509) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11509) Store value


v_accvgpr_read v10, a112 // (op 11536) 
v_accvgpr_read v11, a113 // (op 11536) 
v_accvgpr_read v12, a114 // (op 11536) 
v_accvgpr_read v13, a115 // (op 11536) 
v_mul_f32 v14, v7, v10 // (op 11536) 
v_mul_f32 v15, v7, v11 // (op 11536) 
v_mul_f32 v16, v7, v12 // (op 11536) 
v_mul_f32 v17, v7, v13 // (op 11536) 



s_lshl_b32 s0, s4, 8 // (op 11511) 
s_add_u32 s0, s0, 192 // (op 11511) 
v_add_u32 v3, v9, s0 // (op 11511) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11511) Store value


v_accvgpr_read v10, a116 // (op 11541) 
v_accvgpr_read v11, a117 // (op 11541) 
v_accvgpr_read v12, a118 // (op 11541) 
v_accvgpr_read v13, a119 // (op 11541) 
v_mul_f32 v14, v7, v10 // (op 11541) 
v_mul_f32 v15, v7, v11 // (op 11541) 
v_mul_f32 v16, v7, v12 // (op 11541) 
v_mul_f32 v17, v7, v13 // (op 11541) 



s_lshl_b32 s0, s4, 6 // (op 11513) 
s_mul_i32 s1, 5, s0 // (op 11513) 
s_add_u32 s1, s1, 192 // (op 11513) 

v_add_u32 v3, v9, s1 // (op 11513) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11513) Store value


v_accvgpr_read v10, a120 // (op 11546) 
v_accvgpr_read v11, a121 // (op 11546) 
v_accvgpr_read v12, a122 // (op 11546) 
v_accvgpr_read v13, a123 // (op 11546) 
v_mul_f32 v14, v7, v10 // (op 11546) 
v_mul_f32 v15, v7, v11 // (op 11546) 
v_mul_f32 v16, v7, v12 // (op 11546) 
v_mul_f32 v17, v7, v13 // (op 11546) 



s_lshl_b32 s0, s4, 6 // (op 11515) 
s_mul_i32 s1, 6, s0 // (op 11515) 
s_add_u32 s1, s1, 192 // (op 11515) 

v_add_u32 v3, v9, s1 // (op 11515) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11515) Store value


v_accvgpr_read v10, a124 // (op 11551) 
v_accvgpr_read v11, a125 // (op 11551) 
v_accvgpr_read v12, a126 // (op 11551) 
v_accvgpr_read v13, a127 // (op 11551) 
v_mul_f32 v14, v7, v10 // (op 11551) 
v_mul_f32 v15, v7, v11 // (op 11551) 
v_mul_f32 v16, v7, v12 // (op 11551) 
v_mul_f32 v17, v7, v13 // (op 11551) 



s_lshl_b32 s0, s4, 6 // (op 11517) 
s_mul_i32 s1, 7, s0 // (op 11517) 
s_add_u32 s1, s1, 192 // (op 11517) 

v_add_u32 v3, v9, s1 // (op 11517) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11517) Store value


v_accvgpr_read v10, a128 // (op 11632) 
v_accvgpr_read v11, a129 // (op 11632) 
v_accvgpr_read v12, a130 // (op 11632) 
v_accvgpr_read v13, a131 // (op 11632) 
v_mul_f32 v14, v7, v10 // (op 11632) 
v_mul_f32 v15, v7, v11 // (op 11632) 
v_mul_f32 v16, v7, v12 // (op 11632) 
v_mul_f32 v17, v7, v13 // (op 11632) 



v_add_u32 v3, 256, v9 // (op 11635) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11635) Store value


v_accvgpr_read v10, a132 // (op 11653) 
v_accvgpr_read v11, a133 // (op 11653) 
v_accvgpr_read v12, a134 // (op 11653) 
v_accvgpr_read v13, a135 // (op 11653) 
v_mul_f32 v14, v7, v10 // (op 11653) 
v_mul_f32 v15, v7, v11 // (op 11653) 
v_mul_f32 v16, v7, v12 // (op 11653) 
v_mul_f32 v17, v7, v13 // (op 11653) 



s_lshl_b32 s0, s4, 6 // (op 11637) 
s_add_u32 s0, s0, 256 // (op 11637) 
v_add_u32 v3, v9, s0 // (op 11637) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11637) Store value


v_accvgpr_read v10, a136 // (op 11658) 
v_accvgpr_read v11, a137 // (op 11658) 
v_accvgpr_read v12, a138 // (op 11658) 
v_accvgpr_read v13, a139 // (op 11658) 
v_mul_f32 v14, v7, v10 // (op 11658) 
v_mul_f32 v15, v7, v11 // (op 11658) 
v_mul_f32 v16, v7, v12 // (op 11658) 
v_mul_f32 v17, v7, v13 // (op 11658) 



s_lshl_b32 s0, s4, 7 // (op 11639) 
s_add_u32 s0, s0, 256 // (op 11639) 
v_add_u32 v3, v9, s0 // (op 11639) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11639) Store value


v_accvgpr_read v10, a140 // (op 11663) 
v_accvgpr_read v11, a141 // (op 11663) 
v_accvgpr_read v12, a142 // (op 11663) 
v_accvgpr_read v13, a143 // (op 11663) 
v_mul_f32 v14, v7, v10 // (op 11663) 
v_mul_f32 v15, v7, v11 // (op 11663) 
v_mul_f32 v16, v7, v12 // (op 11663) 
v_mul_f32 v17, v7, v13 // (op 11663) 



s_lshl_b32 s0, s4, 6 // (op 11641) 
s_mul_i32 s1, 3, s0 // (op 11641) 
s_add_u32 s1, s1, 256 // (op 11641) 

v_add_u32 v3, v9, s1 // (op 11641) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11641) Store value


v_accvgpr_read v10, a144 // (op 11668) 
v_accvgpr_read v11, a145 // (op 11668) 
v_accvgpr_read v12, a146 // (op 11668) 
v_accvgpr_read v13, a147 // (op 11668) 
v_mul_f32 v14, v7, v10 // (op 11668) 
v_mul_f32 v15, v7, v11 // (op 11668) 
v_mul_f32 v16, v7, v12 // (op 11668) 
v_mul_f32 v17, v7, v13 // (op 11668) 



s_lshl_b32 s0, s4, 8 // (op 11643) 
s_add_u32 s0, s0, 256 // (op 11643) 
v_add_u32 v3, v9, s0 // (op 11643) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11643) Store value


v_accvgpr_read v10, a148 // (op 11673) 
v_accvgpr_read v11, a149 // (op 11673) 
v_accvgpr_read v12, a150 // (op 11673) 
v_accvgpr_read v13, a151 // (op 11673) 
v_mul_f32 v14, v7, v10 // (op 11673) 
v_mul_f32 v15, v7, v11 // (op 11673) 
v_mul_f32 v16, v7, v12 // (op 11673) 
v_mul_f32 v17, v7, v13 // (op 11673) 



s_lshl_b32 s0, s4, 6 // (op 11645) 
s_mul_i32 s1, 5, s0 // (op 11645) 
s_add_u32 s1, s1, 256 // (op 11645) 

v_add_u32 v3, v9, s1 // (op 11645) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11645) Store value


v_accvgpr_read v10, a152 // (op 11678) 
v_accvgpr_read v11, a153 // (op 11678) 
v_accvgpr_read v12, a154 // (op 11678) 
v_accvgpr_read v13, a155 // (op 11678) 
v_mul_f32 v14, v7, v10 // (op 11678) 
v_mul_f32 v15, v7, v11 // (op 11678) 
v_mul_f32 v16, v7, v12 // (op 11678) 
v_mul_f32 v17, v7, v13 // (op 11678) 



s_lshl_b32 s0, s4, 6 // (op 11647) 
s_mul_i32 s1, 6, s0 // (op 11647) 
s_add_u32 s1, s1, 256 // (op 11647) 

v_add_u32 v3, v9, s1 // (op 11647) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11647) Store value


v_accvgpr_read v10, a156 // (op 11683) 
v_accvgpr_read v11, a157 // (op 11683) 
v_accvgpr_read v12, a158 // (op 11683) 
v_accvgpr_read v13, a159 // (op 11683) 
v_mul_f32 v14, v7, v10 // (op 11683) 
v_mul_f32 v15, v7, v11 // (op 11683) 
v_mul_f32 v16, v7, v12 // (op 11683) 
v_mul_f32 v17, v7, v13 // (op 11683) 



s_lshl_b32 s0, s4, 6 // (op 11649) 
s_mul_i32 s1, 7, s0 // (op 11649) 
s_add_u32 s1, s1, 256 // (op 11649) 

v_add_u32 v3, v9, s1 // (op 11649) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11649) Store value


v_accvgpr_read v10, a160 // (op 11764) 
v_accvgpr_read v11, a161 // (op 11764) 
v_accvgpr_read v12, a162 // (op 11764) 
v_accvgpr_read v13, a163 // (op 11764) 
v_mul_f32 v14, v7, v10 // (op 11764) 
v_mul_f32 v15, v7, v11 // (op 11764) 
v_mul_f32 v16, v7, v12 // (op 11764) 
v_mul_f32 v17, v7, v13 // (op 11764) 



v_add_u32 v3, 320, v9 // (op 11767) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11767) Store value


v_accvgpr_read v10, a164 // (op 11785) 
v_accvgpr_read v11, a165 // (op 11785) 
v_accvgpr_read v12, a166 // (op 11785) 
v_accvgpr_read v13, a167 // (op 11785) 
v_mul_f32 v14, v7, v10 // (op 11785) 
v_mul_f32 v15, v7, v11 // (op 11785) 
v_mul_f32 v16, v7, v12 // (op 11785) 
v_mul_f32 v17, v7, v13 // (op 11785) 



s_lshl_b32 s0, s4, 6 // (op 11769) 
s_add_u32 s0, s0, 320 // (op 11769) 
v_add_u32 v3, v9, s0 // (op 11769) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11769) Store value


v_accvgpr_read v10, a168 // (op 11790) 
v_accvgpr_read v11, a169 // (op 11790) 
v_accvgpr_read v12, a170 // (op 11790) 
v_accvgpr_read v13, a171 // (op 11790) 
v_mul_f32 v14, v7, v10 // (op 11790) 
v_mul_f32 v15, v7, v11 // (op 11790) 
v_mul_f32 v16, v7, v12 // (op 11790) 
v_mul_f32 v17, v7, v13 // (op 11790) 



s_lshl_b32 s0, s4, 7 // (op 11771) 
s_add_u32 s0, s0, 320 // (op 11771) 
v_add_u32 v3, v9, s0 // (op 11771) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11771) Store value


v_accvgpr_read v10, a172 // (op 11795) 
v_accvgpr_read v11, a173 // (op 11795) 
v_accvgpr_read v12, a174 // (op 11795) 
v_accvgpr_read v13, a175 // (op 11795) 
v_mul_f32 v14, v7, v10 // (op 11795) 
v_mul_f32 v15, v7, v11 // (op 11795) 
v_mul_f32 v16, v7, v12 // (op 11795) 
v_mul_f32 v17, v7, v13 // (op 11795) 



s_lshl_b32 s0, s4, 6 // (op 11773) 
s_mul_i32 s1, 3, s0 // (op 11773) 
s_add_u32 s1, s1, 320 // (op 11773) 

v_add_u32 v3, v9, s1 // (op 11773) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11773) Store value


v_accvgpr_read v10, a176 // (op 11800) 
v_accvgpr_read v11, a177 // (op 11800) 
v_accvgpr_read v12, a178 // (op 11800) 
v_accvgpr_read v13, a179 // (op 11800) 
v_mul_f32 v14, v7, v10 // (op 11800) 
v_mul_f32 v15, v7, v11 // (op 11800) 
v_mul_f32 v16, v7, v12 // (op 11800) 
v_mul_f32 v17, v7, v13 // (op 11800) 



s_lshl_b32 s0, s4, 8 // (op 11775) 
s_add_u32 s0, s0, 320 // (op 11775) 
v_add_u32 v3, v9, s0 // (op 11775) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11775) Store value


v_accvgpr_read v10, a180 // (op 11805) 
v_accvgpr_read v11, a181 // (op 11805) 
v_accvgpr_read v12, a182 // (op 11805) 
v_accvgpr_read v13, a183 // (op 11805) 
v_mul_f32 v14, v7, v10 // (op 11805) 
v_mul_f32 v15, v7, v11 // (op 11805) 
v_mul_f32 v16, v7, v12 // (op 11805) 
v_mul_f32 v17, v7, v13 // (op 11805) 



s_lshl_b32 s0, s4, 6 // (op 11777) 
s_mul_i32 s1, 5, s0 // (op 11777) 
s_add_u32 s1, s1, 320 // (op 11777) 

v_add_u32 v3, v9, s1 // (op 11777) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11777) Store value


v_accvgpr_read v10, a184 // (op 11810) 
v_accvgpr_read v11, a185 // (op 11810) 
v_accvgpr_read v12, a186 // (op 11810) 
v_accvgpr_read v13, a187 // (op 11810) 
v_mul_f32 v14, v7, v10 // (op 11810) 
v_mul_f32 v15, v7, v11 // (op 11810) 
v_mul_f32 v16, v7, v12 // (op 11810) 
v_mul_f32 v17, v7, v13 // (op 11810) 



s_lshl_b32 s0, s4, 6 // (op 11779) 
s_mul_i32 s1, 6, s0 // (op 11779) 
s_add_u32 s1, s1, 320 // (op 11779) 

v_add_u32 v3, v9, s1 // (op 11779) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11779) Store value


v_accvgpr_read v10, a188 // (op 11815) 
v_accvgpr_read v11, a189 // (op 11815) 
v_accvgpr_read v12, a190 // (op 11815) 
v_accvgpr_read v13, a191 // (op 11815) 
v_mul_f32 v14, v7, v10 // (op 11815) 
v_mul_f32 v15, v7, v11 // (op 11815) 
v_mul_f32 v16, v7, v12 // (op 11815) 
v_mul_f32 v17, v7, v13 // (op 11815) 



s_lshl_b32 s0, s4, 6 // (op 11781) 
s_mul_i32 s1, 7, s0 // (op 11781) 
s_add_u32 s1, s1, 320 // (op 11781) 

v_add_u32 v3, v9, s1 // (op 11781) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11781) Store value


v_accvgpr_read v10, a192 // (op 11896) 
v_accvgpr_read v11, a193 // (op 11896) 
v_accvgpr_read v12, a194 // (op 11896) 
v_accvgpr_read v13, a195 // (op 11896) 
v_mul_f32 v14, v7, v10 // (op 11896) 
v_mul_f32 v15, v7, v11 // (op 11896) 
v_mul_f32 v16, v7, v12 // (op 11896) 
v_mul_f32 v17, v7, v13 // (op 11896) 



v_add_u32 v3, 384, v9 // (op 11899) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11899) Store value


v_accvgpr_read v10, a196 // (op 11917) 
v_accvgpr_read v11, a197 // (op 11917) 
v_accvgpr_read v12, a198 // (op 11917) 
v_accvgpr_read v13, a199 // (op 11917) 
v_mul_f32 v14, v7, v10 // (op 11917) 
v_mul_f32 v15, v7, v11 // (op 11917) 
v_mul_f32 v16, v7, v12 // (op 11917) 
v_mul_f32 v17, v7, v13 // (op 11917) 



s_lshl_b32 s0, s4, 6 // (op 11901) 
s_add_u32 s0, s0, 384 // (op 11901) 
v_add_u32 v3, v9, s0 // (op 11901) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11901) Store value


v_accvgpr_read v10, a200 // (op 11922) 
v_accvgpr_read v11, a201 // (op 11922) 
v_accvgpr_read v12, a202 // (op 11922) 
v_accvgpr_read v13, a203 // (op 11922) 
v_mul_f32 v14, v7, v10 // (op 11922) 
v_mul_f32 v15, v7, v11 // (op 11922) 
v_mul_f32 v16, v7, v12 // (op 11922) 
v_mul_f32 v17, v7, v13 // (op 11922) 



s_lshl_b32 s0, s4, 7 // (op 11903) 
s_add_u32 s0, s0, 384 // (op 11903) 
v_add_u32 v3, v9, s0 // (op 11903) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11903) Store value


v_accvgpr_read v10, a204 // (op 11927) 
v_accvgpr_read v11, a205 // (op 11927) 
v_accvgpr_read v12, a206 // (op 11927) 
v_accvgpr_read v13, a207 // (op 11927) 
v_mul_f32 v14, v7, v10 // (op 11927) 
v_mul_f32 v15, v7, v11 // (op 11927) 
v_mul_f32 v16, v7, v12 // (op 11927) 
v_mul_f32 v17, v7, v13 // (op 11927) 



s_lshl_b32 s0, s4, 6 // (op 11905) 
s_mul_i32 s1, 3, s0 // (op 11905) 
s_add_u32 s1, s1, 384 // (op 11905) 

v_add_u32 v3, v9, s1 // (op 11905) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11905) Store value


v_accvgpr_read v10, a208 // (op 11932) 
v_accvgpr_read v11, a209 // (op 11932) 
v_accvgpr_read v12, a210 // (op 11932) 
v_accvgpr_read v13, a211 // (op 11932) 
v_mul_f32 v14, v7, v10 // (op 11932) 
v_mul_f32 v15, v7, v11 // (op 11932) 
v_mul_f32 v16, v7, v12 // (op 11932) 
v_mul_f32 v17, v7, v13 // (op 11932) 



s_lshl_b32 s0, s4, 8 // (op 11907) 
s_add_u32 s0, s0, 384 // (op 11907) 
v_add_u32 v3, v9, s0 // (op 11907) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11907) Store value


v_accvgpr_read v10, a212 // (op 11937) 
v_accvgpr_read v11, a213 // (op 11937) 
v_accvgpr_read v12, a214 // (op 11937) 
v_accvgpr_read v13, a215 // (op 11937) 
v_mul_f32 v14, v7, v10 // (op 11937) 
v_mul_f32 v15, v7, v11 // (op 11937) 
v_mul_f32 v16, v7, v12 // (op 11937) 
v_mul_f32 v17, v7, v13 // (op 11937) 



s_lshl_b32 s0, s4, 6 // (op 11909) 
s_mul_i32 s1, 5, s0 // (op 11909) 
s_add_u32 s1, s1, 384 // (op 11909) 

v_add_u32 v3, v9, s1 // (op 11909) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11909) Store value


v_accvgpr_read v10, a216 // (op 11942) 
v_accvgpr_read v11, a217 // (op 11942) 
v_accvgpr_read v12, a218 // (op 11942) 
v_accvgpr_read v13, a219 // (op 11942) 
v_mul_f32 v14, v7, v10 // (op 11942) 
v_mul_f32 v15, v7, v11 // (op 11942) 
v_mul_f32 v16, v7, v12 // (op 11942) 
v_mul_f32 v17, v7, v13 // (op 11942) 



s_lshl_b32 s0, s4, 6 // (op 11911) 
s_mul_i32 s1, 6, s0 // (op 11911) 
s_add_u32 s1, s1, 384 // (op 11911) 

v_add_u32 v3, v9, s1 // (op 11911) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11911) Store value


v_accvgpr_read v10, a220 // (op 11947) 
v_accvgpr_read v11, a221 // (op 11947) 
v_accvgpr_read v12, a222 // (op 11947) 
v_accvgpr_read v13, a223 // (op 11947) 
v_mul_f32 v14, v7, v10 // (op 11947) 
v_mul_f32 v15, v7, v11 // (op 11947) 
v_mul_f32 v16, v7, v12 // (op 11947) 
v_mul_f32 v17, v7, v13 // (op 11947) 



s_lshl_b32 s0, s4, 6 // (op 11913) 
s_mul_i32 s1, 7, s0 // (op 11913) 
s_add_u32 s1, s1, 384 // (op 11913) 

v_add_u32 v3, v9, s1 // (op 11913) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 11913) Store value


v_accvgpr_read v10, a224 // (op 12028) 
v_accvgpr_read v11, a225 // (op 12028) 
v_accvgpr_read v12, a226 // (op 12028) 
v_accvgpr_read v13, a227 // (op 12028) 
v_mul_f32 v14, v7, v10 // (op 12028) 
v_mul_f32 v15, v7, v11 // (op 12028) 
v_mul_f32 v16, v7, v12 // (op 12028) 
v_mul_f32 v17, v7, v13 // (op 12028) 



v_add_u32 v3, 448, v9 // (op 12031) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 12031) Store value


v_accvgpr_read v10, a228 // (op 12049) 
v_accvgpr_read v11, a229 // (op 12049) 
v_accvgpr_read v12, a230 // (op 12049) 
v_accvgpr_read v13, a231 // (op 12049) 
v_mul_f32 v14, v7, v10 // (op 12049) 
v_mul_f32 v15, v7, v11 // (op 12049) 
v_mul_f32 v16, v7, v12 // (op 12049) 
v_mul_f32 v17, v7, v13 // (op 12049) 



s_lshl_b32 s0, s4, 6 // (op 12033) 
s_add_u32 s0, s0, 448 // (op 12033) 
v_add_u32 v3, v9, s0 // (op 12033) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 12033) Store value


v_accvgpr_read v10, a232 // (op 12054) 
v_accvgpr_read v11, a233 // (op 12054) 
v_accvgpr_read v12, a234 // (op 12054) 
v_accvgpr_read v13, a235 // (op 12054) 
v_mul_f32 v14, v7, v10 // (op 12054) 
v_mul_f32 v15, v7, v11 // (op 12054) 
v_mul_f32 v16, v7, v12 // (op 12054) 
v_mul_f32 v17, v7, v13 // (op 12054) 



s_lshl_b32 s0, s4, 7 // (op 12035) 
s_add_u32 s0, s0, 448 // (op 12035) 
v_add_u32 v3, v9, s0 // (op 12035) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 12035) Store value


v_accvgpr_read v10, a236 // (op 12059) 
v_accvgpr_read v11, a237 // (op 12059) 
v_accvgpr_read v12, a238 // (op 12059) 
v_accvgpr_read v13, a239 // (op 12059) 
v_mul_f32 v14, v7, v10 // (op 12059) 
v_mul_f32 v15, v7, v11 // (op 12059) 
v_mul_f32 v16, v7, v12 // (op 12059) 
v_mul_f32 v17, v7, v13 // (op 12059) 



s_lshl_b32 s0, s4, 6 // (op 12037) 
s_mul_i32 s1, 3, s0 // (op 12037) 
s_add_u32 s1, s1, 448 // (op 12037) 

v_add_u32 v3, v9, s1 // (op 12037) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 12037) Store value


v_accvgpr_read v10, a240 // (op 12064) 
v_accvgpr_read v11, a241 // (op 12064) 
v_accvgpr_read v12, a242 // (op 12064) 
v_accvgpr_read v13, a243 // (op 12064) 
v_mul_f32 v14, v7, v10 // (op 12064) 
v_mul_f32 v15, v7, v11 // (op 12064) 
v_mul_f32 v16, v7, v12 // (op 12064) 
v_mul_f32 v17, v7, v13 // (op 12064) 



s_lshl_b32 s0, s4, 8 // (op 12039) 
s_add_u32 s0, s0, 448 // (op 12039) 
v_add_u32 v3, v9, s0 // (op 12039) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 12039) Store value


v_accvgpr_read v10, a244 // (op 12069) 
v_accvgpr_read v11, a245 // (op 12069) 
v_accvgpr_read v12, a246 // (op 12069) 
v_accvgpr_read v13, a247 // (op 12069) 
v_mul_f32 v14, v7, v10 // (op 12069) 
v_mul_f32 v15, v7, v11 // (op 12069) 
v_mul_f32 v16, v7, v12 // (op 12069) 
v_mul_f32 v17, v7, v13 // (op 12069) 



s_lshl_b32 s0, s4, 6 // (op 12041) 
s_mul_i32 s1, 5, s0 // (op 12041) 
s_add_u32 s1, s1, 448 // (op 12041) 

v_add_u32 v3, v9, s1 // (op 12041) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 12041) Store value


v_accvgpr_read v10, a248 // (op 12074) 
v_accvgpr_read v11, a249 // (op 12074) 
v_accvgpr_read v12, a250 // (op 12074) 
v_accvgpr_read v13, a251 // (op 12074) 
v_mul_f32 v14, v7, v10 // (op 12074) 
v_mul_f32 v15, v7, v11 // (op 12074) 
v_mul_f32 v16, v7, v12 // (op 12074) 
v_mul_f32 v17, v7, v13 // (op 12074) 



s_lshl_b32 s0, s4, 6 // (op 12043) 
s_mul_i32 s1, 6, s0 // (op 12043) 
s_add_u32 s1, s1, 448 // (op 12043) 

v_add_u32 v3, v9, s1 // (op 12043) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 12043) Store value


v_accvgpr_read v10, a252 // (op 12079) 
v_accvgpr_read v11, a253 // (op 12079) 
v_accvgpr_read v12, a254 // (op 12079) 
v_accvgpr_read v13, a255 // (op 12079) 
v_mul_f32 v14, v7, v10 // (op 12079) 
v_mul_f32 v15, v7, v11 // (op 12079) 
v_mul_f32 v16, v7, v12 // (op 12079) 
v_mul_f32 v17, v7, v13 // (op 12079) 



s_lshl_b32 s0, s4, 6 // (op 12045) 
s_mul_i32 s1, 7, s0 // (op 12045) 
s_add_u32 s1, s1, 448 // (op 12045) 

v_add_u32 v3, v9, s1 // (op 12045) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 12045) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_114
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalFalse_0_beta_114:


v_add_u32 v3, 448, v8 // (op 13479) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13479) Load value

v_accvgpr_read v14, a224 // (op 13459) 
v_accvgpr_read v15, a225 // (op 13459) 
v_accvgpr_read v16, a226 // (op 13459) 
v_accvgpr_read v17, a227 // (op 13459) 
v_mul_f32 v18, v7, v14 // (op 13459) 
v_mul_f32 v19, v7, v15 // (op 13459) 
v_mul_f32 v20, v7, v16 // (op 13459) 
v_mul_f32 v21, v7, v17 // (op 13459) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13478) 
v_mul_f32 v15, v0, v11 // (op 13478) 
v_mul_f32 v16, v0, v12 // (op 13478) 
v_mul_f32 v17, v0, v13 // (op 13478) 

v_add_f32 v10, v14, v18 // (op 13460) 
v_add_f32 v11, v15, v19 // (op 13460) 
v_add_f32 v12, v16, v20 // (op 13460) 
v_add_f32 v13, v17, v21 // (op 13460) 




v_add_u32 v3, 448, v9 // (op 13462) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13462) Store value




s_lshl_b32 s0, s2, 6 // (op 13483) 
s_add_u32 s0, s0, 448 // (op 13483) 
v_add_u32 v3, v8, s0 // (op 13483) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13483) Load value

v_accvgpr_read v14, a228 // (op 13509) 
v_accvgpr_read v15, a229 // (op 13509) 
v_accvgpr_read v16, a230 // (op 13509) 
v_accvgpr_read v17, a231 // (op 13509) 
v_mul_f32 v18, v7, v14 // (op 13509) 
v_mul_f32 v19, v7, v15 // (op 13509) 
v_mul_f32 v20, v7, v16 // (op 13509) 
v_mul_f32 v21, v7, v17 // (op 13509) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13481) 
v_mul_f32 v15, v0, v11 // (op 13481) 
v_mul_f32 v16, v0, v12 // (op 13481) 
v_mul_f32 v17, v0, v13 // (op 13481) 

v_add_f32 v10, v14, v18 // (op 13482) 
v_add_f32 v11, v15, v19 // (op 13482) 
v_add_f32 v12, v16, v20 // (op 13482) 
v_add_f32 v13, v17, v21 // (op 13482) 




s_lshl_b32 s0, s4, 6 // (op 13464) 
s_add_u32 s0, s0, 448 // (op 13464) 
v_add_u32 v3, v9, s0 // (op 13464) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13464) Store value




s_lshl_b32 s0, s2, 7 // (op 13487) 
s_add_u32 s0, s0, 448 // (op 13487) 
v_add_u32 v3, v8, s0 // (op 13487) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13487) Load value

v_accvgpr_read v14, a232 // (op 13511) 
v_accvgpr_read v15, a233 // (op 13511) 
v_accvgpr_read v16, a234 // (op 13511) 
v_accvgpr_read v17, a235 // (op 13511) 
v_mul_f32 v18, v7, v14 // (op 13511) 
v_mul_f32 v19, v7, v15 // (op 13511) 
v_mul_f32 v20, v7, v16 // (op 13511) 
v_mul_f32 v21, v7, v17 // (op 13511) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13485) 
v_mul_f32 v15, v0, v11 // (op 13485) 
v_mul_f32 v16, v0, v12 // (op 13485) 
v_mul_f32 v17, v0, v13 // (op 13485) 

v_add_f32 v10, v14, v18 // (op 13486) 
v_add_f32 v11, v15, v19 // (op 13486) 
v_add_f32 v12, v16, v20 // (op 13486) 
v_add_f32 v13, v17, v21 // (op 13486) 




s_lshl_b32 s0, s4, 7 // (op 13466) 
s_add_u32 s0, s0, 448 // (op 13466) 
v_add_u32 v3, v9, s0 // (op 13466) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13466) Store value




s_lshl_b32 s0, s2, 6 // (op 13491) 
s_mul_i32 s1, 3, s0 // (op 13491) 
s_add_u32 s1, s1, 448 // (op 13491) 

v_add_u32 v3, v8, s1 // (op 13491) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13491) Load value

v_accvgpr_read v14, a236 // (op 13513) 
v_accvgpr_read v15, a237 // (op 13513) 
v_accvgpr_read v16, a238 // (op 13513) 
v_accvgpr_read v17, a239 // (op 13513) 
v_mul_f32 v18, v7, v14 // (op 13513) 
v_mul_f32 v19, v7, v15 // (op 13513) 
v_mul_f32 v20, v7, v16 // (op 13513) 
v_mul_f32 v21, v7, v17 // (op 13513) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13489) 
v_mul_f32 v15, v0, v11 // (op 13489) 
v_mul_f32 v16, v0, v12 // (op 13489) 
v_mul_f32 v17, v0, v13 // (op 13489) 

v_add_f32 v10, v14, v18 // (op 13490) 
v_add_f32 v11, v15, v19 // (op 13490) 
v_add_f32 v12, v16, v20 // (op 13490) 
v_add_f32 v13, v17, v21 // (op 13490) 




s_lshl_b32 s0, s4, 6 // (op 13468) 
s_mul_i32 s1, 3, s0 // (op 13468) 
s_add_u32 s1, s1, 448 // (op 13468) 

v_add_u32 v3, v9, s1 // (op 13468) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13468) Store value




s_lshl_b32 s0, s2, 8 // (op 13495) 
s_add_u32 s0, s0, 448 // (op 13495) 
v_add_u32 v3, v8, s0 // (op 13495) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13495) Load value

v_accvgpr_read v14, a240 // (op 13515) 
v_accvgpr_read v15, a241 // (op 13515) 
v_accvgpr_read v16, a242 // (op 13515) 
v_accvgpr_read v17, a243 // (op 13515) 
v_mul_f32 v18, v7, v14 // (op 13515) 
v_mul_f32 v19, v7, v15 // (op 13515) 
v_mul_f32 v20, v7, v16 // (op 13515) 
v_mul_f32 v21, v7, v17 // (op 13515) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13493) 
v_mul_f32 v15, v0, v11 // (op 13493) 
v_mul_f32 v16, v0, v12 // (op 13493) 
v_mul_f32 v17, v0, v13 // (op 13493) 

v_add_f32 v10, v14, v18 // (op 13494) 
v_add_f32 v11, v15, v19 // (op 13494) 
v_add_f32 v12, v16, v20 // (op 13494) 
v_add_f32 v13, v17, v21 // (op 13494) 




s_lshl_b32 s0, s4, 8 // (op 13470) 
s_add_u32 s0, s0, 448 // (op 13470) 
v_add_u32 v3, v9, s0 // (op 13470) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13470) Store value




s_lshl_b32 s0, s2, 6 // (op 13499) 
s_mul_i32 s1, 5, s0 // (op 13499) 
s_add_u32 s1, s1, 448 // (op 13499) 

v_add_u32 v3, v8, s1 // (op 13499) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13499) Load value

v_accvgpr_read v14, a244 // (op 13517) 
v_accvgpr_read v15, a245 // (op 13517) 
v_accvgpr_read v16, a246 // (op 13517) 
v_accvgpr_read v17, a247 // (op 13517) 
v_mul_f32 v18, v7, v14 // (op 13517) 
v_mul_f32 v19, v7, v15 // (op 13517) 
v_mul_f32 v20, v7, v16 // (op 13517) 
v_mul_f32 v21, v7, v17 // (op 13517) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13497) 
v_mul_f32 v15, v0, v11 // (op 13497) 
v_mul_f32 v16, v0, v12 // (op 13497) 
v_mul_f32 v17, v0, v13 // (op 13497) 

v_add_f32 v10, v14, v18 // (op 13498) 
v_add_f32 v11, v15, v19 // (op 13498) 
v_add_f32 v12, v16, v20 // (op 13498) 
v_add_f32 v13, v17, v21 // (op 13498) 




s_lshl_b32 s0, s4, 6 // (op 13472) 
s_mul_i32 s1, 5, s0 // (op 13472) 
s_add_u32 s1, s1, 448 // (op 13472) 

v_add_u32 v3, v9, s1 // (op 13472) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13472) Store value




s_lshl_b32 s0, s2, 6 // (op 13503) 
s_mul_i32 s1, 6, s0 // (op 13503) 
s_add_u32 s1, s1, 448 // (op 13503) 

v_add_u32 v3, v8, s1 // (op 13503) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13503) Load value

v_accvgpr_read v14, a248 // (op 13519) 
v_accvgpr_read v15, a249 // (op 13519) 
v_accvgpr_read v16, a250 // (op 13519) 
v_accvgpr_read v17, a251 // (op 13519) 
v_mul_f32 v18, v7, v14 // (op 13519) 
v_mul_f32 v19, v7, v15 // (op 13519) 
v_mul_f32 v20, v7, v16 // (op 13519) 
v_mul_f32 v21, v7, v17 // (op 13519) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13501) 
v_mul_f32 v15, v0, v11 // (op 13501) 
v_mul_f32 v16, v0, v12 // (op 13501) 
v_mul_f32 v17, v0, v13 // (op 13501) 

v_add_f32 v10, v14, v18 // (op 13502) 
v_add_f32 v11, v15, v19 // (op 13502) 
v_add_f32 v12, v16, v20 // (op 13502) 
v_add_f32 v13, v17, v21 // (op 13502) 




s_lshl_b32 s0, s4, 6 // (op 13474) 
s_mul_i32 s1, 6, s0 // (op 13474) 
s_add_u32 s1, s1, 448 // (op 13474) 

v_add_u32 v3, v9, s1 // (op 13474) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13474) Store value




s_lshl_b32 s0, s2, 6 // (op 13507) 
s_mul_i32 s1, 7, s0 // (op 13507) 
s_add_u32 s1, s1, 448 // (op 13507) 

v_add_u32 v3, v8, s1 // (op 13507) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13507) Load value

v_accvgpr_read v14, a252 // (op 13521) 
v_accvgpr_read v15, a253 // (op 13521) 
v_accvgpr_read v16, a254 // (op 13521) 
v_accvgpr_read v17, a255 // (op 13521) 
v_mul_f32 v18, v7, v14 // (op 13521) 
v_mul_f32 v19, v7, v15 // (op 13521) 
v_mul_f32 v20, v7, v16 // (op 13521) 
v_mul_f32 v21, v7, v17 // (op 13521) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13505) 
v_mul_f32 v15, v0, v11 // (op 13505) 
v_mul_f32 v16, v0, v12 // (op 13505) 
v_mul_f32 v17, v0, v13 // (op 13505) 

v_add_f32 v10, v14, v18 // (op 13506) 
v_add_f32 v11, v15, v19 // (op 13506) 
v_add_f32 v12, v16, v20 // (op 13506) 
v_add_f32 v13, v17, v21 // (op 13506) 




s_lshl_b32 s0, s4, 6 // (op 13476) 
s_mul_i32 s1, 7, s0 // (op 13476) 
s_add_u32 s1, s1, 448 // (op 13476) 

v_add_u32 v3, v9, s1 // (op 13476) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13476) Store value




v_add_u32 v3, 384, v8 // (op 13324) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13324) Load value

v_accvgpr_read v14, a192 // (op 13304) 
v_accvgpr_read v15, a193 // (op 13304) 
v_accvgpr_read v16, a194 // (op 13304) 
v_accvgpr_read v17, a195 // (op 13304) 
v_mul_f32 v18, v7, v14 // (op 13304) 
v_mul_f32 v19, v7, v15 // (op 13304) 
v_mul_f32 v20, v7, v16 // (op 13304) 
v_mul_f32 v21, v7, v17 // (op 13304) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13323) 
v_mul_f32 v15, v0, v11 // (op 13323) 
v_mul_f32 v16, v0, v12 // (op 13323) 
v_mul_f32 v17, v0, v13 // (op 13323) 

v_add_f32 v10, v14, v18 // (op 13305) 
v_add_f32 v11, v15, v19 // (op 13305) 
v_add_f32 v12, v16, v20 // (op 13305) 
v_add_f32 v13, v17, v21 // (op 13305) 




v_add_u32 v3, 384, v9 // (op 13307) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13307) Store value




s_lshl_b32 s0, s2, 6 // (op 13328) 
s_add_u32 s0, s0, 384 // (op 13328) 
v_add_u32 v3, v8, s0 // (op 13328) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13328) Load value

v_accvgpr_read v14, a196 // (op 13354) 
v_accvgpr_read v15, a197 // (op 13354) 
v_accvgpr_read v16, a198 // (op 13354) 
v_accvgpr_read v17, a199 // (op 13354) 
v_mul_f32 v18, v7, v14 // (op 13354) 
v_mul_f32 v19, v7, v15 // (op 13354) 
v_mul_f32 v20, v7, v16 // (op 13354) 
v_mul_f32 v21, v7, v17 // (op 13354) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13326) 
v_mul_f32 v15, v0, v11 // (op 13326) 
v_mul_f32 v16, v0, v12 // (op 13326) 
v_mul_f32 v17, v0, v13 // (op 13326) 

v_add_f32 v10, v14, v18 // (op 13327) 
v_add_f32 v11, v15, v19 // (op 13327) 
v_add_f32 v12, v16, v20 // (op 13327) 
v_add_f32 v13, v17, v21 // (op 13327) 




s_lshl_b32 s0, s4, 6 // (op 13309) 
s_add_u32 s0, s0, 384 // (op 13309) 
v_add_u32 v3, v9, s0 // (op 13309) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13309) Store value




s_lshl_b32 s0, s2, 7 // (op 13332) 
s_add_u32 s0, s0, 384 // (op 13332) 
v_add_u32 v3, v8, s0 // (op 13332) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13332) Load value

v_accvgpr_read v14, a200 // (op 13356) 
v_accvgpr_read v15, a201 // (op 13356) 
v_accvgpr_read v16, a202 // (op 13356) 
v_accvgpr_read v17, a203 // (op 13356) 
v_mul_f32 v18, v7, v14 // (op 13356) 
v_mul_f32 v19, v7, v15 // (op 13356) 
v_mul_f32 v20, v7, v16 // (op 13356) 
v_mul_f32 v21, v7, v17 // (op 13356) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13330) 
v_mul_f32 v15, v0, v11 // (op 13330) 
v_mul_f32 v16, v0, v12 // (op 13330) 
v_mul_f32 v17, v0, v13 // (op 13330) 

v_add_f32 v10, v14, v18 // (op 13331) 
v_add_f32 v11, v15, v19 // (op 13331) 
v_add_f32 v12, v16, v20 // (op 13331) 
v_add_f32 v13, v17, v21 // (op 13331) 




s_lshl_b32 s0, s4, 7 // (op 13311) 
s_add_u32 s0, s0, 384 // (op 13311) 
v_add_u32 v3, v9, s0 // (op 13311) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13311) Store value




s_lshl_b32 s0, s2, 6 // (op 13336) 
s_mul_i32 s1, 3, s0 // (op 13336) 
s_add_u32 s1, s1, 384 // (op 13336) 

v_add_u32 v3, v8, s1 // (op 13336) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13336) Load value

v_accvgpr_read v14, a204 // (op 13358) 
v_accvgpr_read v15, a205 // (op 13358) 
v_accvgpr_read v16, a206 // (op 13358) 
v_accvgpr_read v17, a207 // (op 13358) 
v_mul_f32 v18, v7, v14 // (op 13358) 
v_mul_f32 v19, v7, v15 // (op 13358) 
v_mul_f32 v20, v7, v16 // (op 13358) 
v_mul_f32 v21, v7, v17 // (op 13358) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13334) 
v_mul_f32 v15, v0, v11 // (op 13334) 
v_mul_f32 v16, v0, v12 // (op 13334) 
v_mul_f32 v17, v0, v13 // (op 13334) 

v_add_f32 v10, v14, v18 // (op 13335) 
v_add_f32 v11, v15, v19 // (op 13335) 
v_add_f32 v12, v16, v20 // (op 13335) 
v_add_f32 v13, v17, v21 // (op 13335) 




s_lshl_b32 s0, s4, 6 // (op 13313) 
s_mul_i32 s1, 3, s0 // (op 13313) 
s_add_u32 s1, s1, 384 // (op 13313) 

v_add_u32 v3, v9, s1 // (op 13313) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13313) Store value




s_lshl_b32 s0, s2, 8 // (op 13340) 
s_add_u32 s0, s0, 384 // (op 13340) 
v_add_u32 v3, v8, s0 // (op 13340) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13340) Load value

v_accvgpr_read v14, a208 // (op 13360) 
v_accvgpr_read v15, a209 // (op 13360) 
v_accvgpr_read v16, a210 // (op 13360) 
v_accvgpr_read v17, a211 // (op 13360) 
v_mul_f32 v18, v7, v14 // (op 13360) 
v_mul_f32 v19, v7, v15 // (op 13360) 
v_mul_f32 v20, v7, v16 // (op 13360) 
v_mul_f32 v21, v7, v17 // (op 13360) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13338) 
v_mul_f32 v15, v0, v11 // (op 13338) 
v_mul_f32 v16, v0, v12 // (op 13338) 
v_mul_f32 v17, v0, v13 // (op 13338) 

v_add_f32 v10, v14, v18 // (op 13339) 
v_add_f32 v11, v15, v19 // (op 13339) 
v_add_f32 v12, v16, v20 // (op 13339) 
v_add_f32 v13, v17, v21 // (op 13339) 




s_lshl_b32 s0, s4, 8 // (op 13315) 
s_add_u32 s0, s0, 384 // (op 13315) 
v_add_u32 v3, v9, s0 // (op 13315) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13315) Store value




s_lshl_b32 s0, s2, 6 // (op 13344) 
s_mul_i32 s1, 5, s0 // (op 13344) 
s_add_u32 s1, s1, 384 // (op 13344) 

v_add_u32 v3, v8, s1 // (op 13344) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13344) Load value

v_accvgpr_read v14, a212 // (op 13362) 
v_accvgpr_read v15, a213 // (op 13362) 
v_accvgpr_read v16, a214 // (op 13362) 
v_accvgpr_read v17, a215 // (op 13362) 
v_mul_f32 v18, v7, v14 // (op 13362) 
v_mul_f32 v19, v7, v15 // (op 13362) 
v_mul_f32 v20, v7, v16 // (op 13362) 
v_mul_f32 v21, v7, v17 // (op 13362) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13342) 
v_mul_f32 v15, v0, v11 // (op 13342) 
v_mul_f32 v16, v0, v12 // (op 13342) 
v_mul_f32 v17, v0, v13 // (op 13342) 

v_add_f32 v10, v14, v18 // (op 13343) 
v_add_f32 v11, v15, v19 // (op 13343) 
v_add_f32 v12, v16, v20 // (op 13343) 
v_add_f32 v13, v17, v21 // (op 13343) 




s_lshl_b32 s0, s4, 6 // (op 13317) 
s_mul_i32 s1, 5, s0 // (op 13317) 
s_add_u32 s1, s1, 384 // (op 13317) 

v_add_u32 v3, v9, s1 // (op 13317) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13317) Store value




s_lshl_b32 s0, s2, 6 // (op 13348) 
s_mul_i32 s1, 6, s0 // (op 13348) 
s_add_u32 s1, s1, 384 // (op 13348) 

v_add_u32 v3, v8, s1 // (op 13348) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13348) Load value

v_accvgpr_read v14, a216 // (op 13364) 
v_accvgpr_read v15, a217 // (op 13364) 
v_accvgpr_read v16, a218 // (op 13364) 
v_accvgpr_read v17, a219 // (op 13364) 
v_mul_f32 v18, v7, v14 // (op 13364) 
v_mul_f32 v19, v7, v15 // (op 13364) 
v_mul_f32 v20, v7, v16 // (op 13364) 
v_mul_f32 v21, v7, v17 // (op 13364) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13346) 
v_mul_f32 v15, v0, v11 // (op 13346) 
v_mul_f32 v16, v0, v12 // (op 13346) 
v_mul_f32 v17, v0, v13 // (op 13346) 

v_add_f32 v10, v14, v18 // (op 13347) 
v_add_f32 v11, v15, v19 // (op 13347) 
v_add_f32 v12, v16, v20 // (op 13347) 
v_add_f32 v13, v17, v21 // (op 13347) 




s_lshl_b32 s0, s4, 6 // (op 13319) 
s_mul_i32 s1, 6, s0 // (op 13319) 
s_add_u32 s1, s1, 384 // (op 13319) 

v_add_u32 v3, v9, s1 // (op 13319) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13319) Store value




s_lshl_b32 s0, s2, 6 // (op 13352) 
s_mul_i32 s1, 7, s0 // (op 13352) 
s_add_u32 s1, s1, 384 // (op 13352) 

v_add_u32 v3, v8, s1 // (op 13352) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13352) Load value

v_accvgpr_read v14, a220 // (op 13366) 
v_accvgpr_read v15, a221 // (op 13366) 
v_accvgpr_read v16, a222 // (op 13366) 
v_accvgpr_read v17, a223 // (op 13366) 
v_mul_f32 v18, v7, v14 // (op 13366) 
v_mul_f32 v19, v7, v15 // (op 13366) 
v_mul_f32 v20, v7, v16 // (op 13366) 
v_mul_f32 v21, v7, v17 // (op 13366) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13350) 
v_mul_f32 v15, v0, v11 // (op 13350) 
v_mul_f32 v16, v0, v12 // (op 13350) 
v_mul_f32 v17, v0, v13 // (op 13350) 

v_add_f32 v10, v14, v18 // (op 13351) 
v_add_f32 v11, v15, v19 // (op 13351) 
v_add_f32 v12, v16, v20 // (op 13351) 
v_add_f32 v13, v17, v21 // (op 13351) 




s_lshl_b32 s0, s4, 6 // (op 13321) 
s_mul_i32 s1, 7, s0 // (op 13321) 
s_add_u32 s1, s1, 384 // (op 13321) 

v_add_u32 v3, v9, s1 // (op 13321) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13321) Store value




v_add_u32 v3, 256, v8 // (op 13014) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13014) Load value

v_accvgpr_read v14, a128 // (op 12994) 
v_accvgpr_read v15, a129 // (op 12994) 
v_accvgpr_read v16, a130 // (op 12994) 
v_accvgpr_read v17, a131 // (op 12994) 
v_mul_f32 v18, v7, v14 // (op 12994) 
v_mul_f32 v19, v7, v15 // (op 12994) 
v_mul_f32 v20, v7, v16 // (op 12994) 
v_mul_f32 v21, v7, v17 // (op 12994) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13013) 
v_mul_f32 v15, v0, v11 // (op 13013) 
v_mul_f32 v16, v0, v12 // (op 13013) 
v_mul_f32 v17, v0, v13 // (op 13013) 

v_add_f32 v10, v14, v18 // (op 12995) 
v_add_f32 v11, v15, v19 // (op 12995) 
v_add_f32 v12, v16, v20 // (op 12995) 
v_add_f32 v13, v17, v21 // (op 12995) 




v_add_u32 v3, 256, v9 // (op 12997) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12997) Store value




s_lshl_b32 s0, s2, 6 // (op 13018) 
s_add_u32 s0, s0, 256 // (op 13018) 
v_add_u32 v3, v8, s0 // (op 13018) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13018) Load value

v_accvgpr_read v14, a132 // (op 13044) 
v_accvgpr_read v15, a133 // (op 13044) 
v_accvgpr_read v16, a134 // (op 13044) 
v_accvgpr_read v17, a135 // (op 13044) 
v_mul_f32 v18, v7, v14 // (op 13044) 
v_mul_f32 v19, v7, v15 // (op 13044) 
v_mul_f32 v20, v7, v16 // (op 13044) 
v_mul_f32 v21, v7, v17 // (op 13044) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13016) 
v_mul_f32 v15, v0, v11 // (op 13016) 
v_mul_f32 v16, v0, v12 // (op 13016) 
v_mul_f32 v17, v0, v13 // (op 13016) 

v_add_f32 v10, v14, v18 // (op 13017) 
v_add_f32 v11, v15, v19 // (op 13017) 
v_add_f32 v12, v16, v20 // (op 13017) 
v_add_f32 v13, v17, v21 // (op 13017) 




s_lshl_b32 s0, s4, 6 // (op 12999) 
s_add_u32 s0, s0, 256 // (op 12999) 
v_add_u32 v3, v9, s0 // (op 12999) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12999) Store value




s_lshl_b32 s0, s2, 7 // (op 13022) 
s_add_u32 s0, s0, 256 // (op 13022) 
v_add_u32 v3, v8, s0 // (op 13022) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13022) Load value

v_accvgpr_read v14, a136 // (op 13046) 
v_accvgpr_read v15, a137 // (op 13046) 
v_accvgpr_read v16, a138 // (op 13046) 
v_accvgpr_read v17, a139 // (op 13046) 
v_mul_f32 v18, v7, v14 // (op 13046) 
v_mul_f32 v19, v7, v15 // (op 13046) 
v_mul_f32 v20, v7, v16 // (op 13046) 
v_mul_f32 v21, v7, v17 // (op 13046) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13020) 
v_mul_f32 v15, v0, v11 // (op 13020) 
v_mul_f32 v16, v0, v12 // (op 13020) 
v_mul_f32 v17, v0, v13 // (op 13020) 

v_add_f32 v10, v14, v18 // (op 13021) 
v_add_f32 v11, v15, v19 // (op 13021) 
v_add_f32 v12, v16, v20 // (op 13021) 
v_add_f32 v13, v17, v21 // (op 13021) 




s_lshl_b32 s0, s4, 7 // (op 13001) 
s_add_u32 s0, s0, 256 // (op 13001) 
v_add_u32 v3, v9, s0 // (op 13001) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13001) Store value




s_lshl_b32 s0, s2, 6 // (op 13026) 
s_mul_i32 s1, 3, s0 // (op 13026) 
s_add_u32 s1, s1, 256 // (op 13026) 

v_add_u32 v3, v8, s1 // (op 13026) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13026) Load value

v_accvgpr_read v14, a140 // (op 13048) 
v_accvgpr_read v15, a141 // (op 13048) 
v_accvgpr_read v16, a142 // (op 13048) 
v_accvgpr_read v17, a143 // (op 13048) 
v_mul_f32 v18, v7, v14 // (op 13048) 
v_mul_f32 v19, v7, v15 // (op 13048) 
v_mul_f32 v20, v7, v16 // (op 13048) 
v_mul_f32 v21, v7, v17 // (op 13048) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13024) 
v_mul_f32 v15, v0, v11 // (op 13024) 
v_mul_f32 v16, v0, v12 // (op 13024) 
v_mul_f32 v17, v0, v13 // (op 13024) 

v_add_f32 v10, v14, v18 // (op 13025) 
v_add_f32 v11, v15, v19 // (op 13025) 
v_add_f32 v12, v16, v20 // (op 13025) 
v_add_f32 v13, v17, v21 // (op 13025) 




s_lshl_b32 s0, s4, 6 // (op 13003) 
s_mul_i32 s1, 3, s0 // (op 13003) 
s_add_u32 s1, s1, 256 // (op 13003) 

v_add_u32 v3, v9, s1 // (op 13003) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13003) Store value




s_lshl_b32 s0, s2, 8 // (op 13030) 
s_add_u32 s0, s0, 256 // (op 13030) 
v_add_u32 v3, v8, s0 // (op 13030) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13030) Load value

v_accvgpr_read v14, a144 // (op 13050) 
v_accvgpr_read v15, a145 // (op 13050) 
v_accvgpr_read v16, a146 // (op 13050) 
v_accvgpr_read v17, a147 // (op 13050) 
v_mul_f32 v18, v7, v14 // (op 13050) 
v_mul_f32 v19, v7, v15 // (op 13050) 
v_mul_f32 v20, v7, v16 // (op 13050) 
v_mul_f32 v21, v7, v17 // (op 13050) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13028) 
v_mul_f32 v15, v0, v11 // (op 13028) 
v_mul_f32 v16, v0, v12 // (op 13028) 
v_mul_f32 v17, v0, v13 // (op 13028) 

v_add_f32 v10, v14, v18 // (op 13029) 
v_add_f32 v11, v15, v19 // (op 13029) 
v_add_f32 v12, v16, v20 // (op 13029) 
v_add_f32 v13, v17, v21 // (op 13029) 




s_lshl_b32 s0, s4, 8 // (op 13005) 
s_add_u32 s0, s0, 256 // (op 13005) 
v_add_u32 v3, v9, s0 // (op 13005) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13005) Store value




s_lshl_b32 s0, s2, 6 // (op 13034) 
s_mul_i32 s1, 5, s0 // (op 13034) 
s_add_u32 s1, s1, 256 // (op 13034) 

v_add_u32 v3, v8, s1 // (op 13034) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13034) Load value

v_accvgpr_read v14, a148 // (op 13052) 
v_accvgpr_read v15, a149 // (op 13052) 
v_accvgpr_read v16, a150 // (op 13052) 
v_accvgpr_read v17, a151 // (op 13052) 
v_mul_f32 v18, v7, v14 // (op 13052) 
v_mul_f32 v19, v7, v15 // (op 13052) 
v_mul_f32 v20, v7, v16 // (op 13052) 
v_mul_f32 v21, v7, v17 // (op 13052) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13032) 
v_mul_f32 v15, v0, v11 // (op 13032) 
v_mul_f32 v16, v0, v12 // (op 13032) 
v_mul_f32 v17, v0, v13 // (op 13032) 

v_add_f32 v10, v14, v18 // (op 13033) 
v_add_f32 v11, v15, v19 // (op 13033) 
v_add_f32 v12, v16, v20 // (op 13033) 
v_add_f32 v13, v17, v21 // (op 13033) 




s_lshl_b32 s0, s4, 6 // (op 13007) 
s_mul_i32 s1, 5, s0 // (op 13007) 
s_add_u32 s1, s1, 256 // (op 13007) 

v_add_u32 v3, v9, s1 // (op 13007) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13007) Store value




s_lshl_b32 s0, s2, 6 // (op 13038) 
s_mul_i32 s1, 6, s0 // (op 13038) 
s_add_u32 s1, s1, 256 // (op 13038) 

v_add_u32 v3, v8, s1 // (op 13038) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13038) Load value

v_accvgpr_read v14, a152 // (op 13054) 
v_accvgpr_read v15, a153 // (op 13054) 
v_accvgpr_read v16, a154 // (op 13054) 
v_accvgpr_read v17, a155 // (op 13054) 
v_mul_f32 v18, v7, v14 // (op 13054) 
v_mul_f32 v19, v7, v15 // (op 13054) 
v_mul_f32 v20, v7, v16 // (op 13054) 
v_mul_f32 v21, v7, v17 // (op 13054) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13036) 
v_mul_f32 v15, v0, v11 // (op 13036) 
v_mul_f32 v16, v0, v12 // (op 13036) 
v_mul_f32 v17, v0, v13 // (op 13036) 

v_add_f32 v10, v14, v18 // (op 13037) 
v_add_f32 v11, v15, v19 // (op 13037) 
v_add_f32 v12, v16, v20 // (op 13037) 
v_add_f32 v13, v17, v21 // (op 13037) 




s_lshl_b32 s0, s4, 6 // (op 13009) 
s_mul_i32 s1, 6, s0 // (op 13009) 
s_add_u32 s1, s1, 256 // (op 13009) 

v_add_u32 v3, v9, s1 // (op 13009) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13009) Store value




s_lshl_b32 s0, s2, 6 // (op 13042) 
s_mul_i32 s1, 7, s0 // (op 13042) 
s_add_u32 s1, s1, 256 // (op 13042) 

v_add_u32 v3, v8, s1 // (op 13042) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13042) Load value

v_accvgpr_read v14, a156 // (op 13056) 
v_accvgpr_read v15, a157 // (op 13056) 
v_accvgpr_read v16, a158 // (op 13056) 
v_accvgpr_read v17, a159 // (op 13056) 
v_mul_f32 v18, v7, v14 // (op 13056) 
v_mul_f32 v19, v7, v15 // (op 13056) 
v_mul_f32 v20, v7, v16 // (op 13056) 
v_mul_f32 v21, v7, v17 // (op 13056) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 13040) 
v_mul_f32 v15, v0, v11 // (op 13040) 
v_mul_f32 v16, v0, v12 // (op 13040) 
v_mul_f32 v17, v0, v13 // (op 13040) 

v_add_f32 v10, v14, v18 // (op 13041) 
v_add_f32 v11, v15, v19 // (op 13041) 
v_add_f32 v12, v16, v20 // (op 13041) 
v_add_f32 v13, v17, v21 // (op 13041) 




s_lshl_b32 s0, s4, 6 // (op 13011) 
s_mul_i32 s1, 7, s0 // (op 13011) 
s_add_u32 s1, s1, 256 // (op 13011) 

v_add_u32 v3, v9, s1 // (op 13011) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13011) Store value




v_add_u32 v3, 192, v8 // (op 12859) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12859) Load value

v_accvgpr_read v14, a96 // (op 12839) 
v_accvgpr_read v15, a97 // (op 12839) 
v_accvgpr_read v16, a98 // (op 12839) 
v_accvgpr_read v17, a99 // (op 12839) 
v_mul_f32 v18, v7, v14 // (op 12839) 
v_mul_f32 v19, v7, v15 // (op 12839) 
v_mul_f32 v20, v7, v16 // (op 12839) 
v_mul_f32 v21, v7, v17 // (op 12839) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12858) 
v_mul_f32 v15, v0, v11 // (op 12858) 
v_mul_f32 v16, v0, v12 // (op 12858) 
v_mul_f32 v17, v0, v13 // (op 12858) 

v_add_f32 v10, v14, v18 // (op 12840) 
v_add_f32 v11, v15, v19 // (op 12840) 
v_add_f32 v12, v16, v20 // (op 12840) 
v_add_f32 v13, v17, v21 // (op 12840) 




v_add_u32 v3, 192, v9 // (op 12842) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12842) Store value




s_lshl_b32 s0, s2, 6 // (op 12863) 
s_add_u32 s0, s0, 192 // (op 12863) 
v_add_u32 v3, v8, s0 // (op 12863) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12863) Load value

v_accvgpr_read v14, a100 // (op 12889) 
v_accvgpr_read v15, a101 // (op 12889) 
v_accvgpr_read v16, a102 // (op 12889) 
v_accvgpr_read v17, a103 // (op 12889) 
v_mul_f32 v18, v7, v14 // (op 12889) 
v_mul_f32 v19, v7, v15 // (op 12889) 
v_mul_f32 v20, v7, v16 // (op 12889) 
v_mul_f32 v21, v7, v17 // (op 12889) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12861) 
v_mul_f32 v15, v0, v11 // (op 12861) 
v_mul_f32 v16, v0, v12 // (op 12861) 
v_mul_f32 v17, v0, v13 // (op 12861) 

v_add_f32 v10, v14, v18 // (op 12862) 
v_add_f32 v11, v15, v19 // (op 12862) 
v_add_f32 v12, v16, v20 // (op 12862) 
v_add_f32 v13, v17, v21 // (op 12862) 




s_lshl_b32 s0, s4, 6 // (op 12844) 
s_add_u32 s0, s0, 192 // (op 12844) 
v_add_u32 v3, v9, s0 // (op 12844) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12844) Store value




s_lshl_b32 s0, s2, 7 // (op 12867) 
s_add_u32 s0, s0, 192 // (op 12867) 
v_add_u32 v3, v8, s0 // (op 12867) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12867) Load value

v_accvgpr_read v14, a104 // (op 12891) 
v_accvgpr_read v15, a105 // (op 12891) 
v_accvgpr_read v16, a106 // (op 12891) 
v_accvgpr_read v17, a107 // (op 12891) 
v_mul_f32 v18, v7, v14 // (op 12891) 
v_mul_f32 v19, v7, v15 // (op 12891) 
v_mul_f32 v20, v7, v16 // (op 12891) 
v_mul_f32 v21, v7, v17 // (op 12891) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12865) 
v_mul_f32 v15, v0, v11 // (op 12865) 
v_mul_f32 v16, v0, v12 // (op 12865) 
v_mul_f32 v17, v0, v13 // (op 12865) 

v_add_f32 v10, v14, v18 // (op 12866) 
v_add_f32 v11, v15, v19 // (op 12866) 
v_add_f32 v12, v16, v20 // (op 12866) 
v_add_f32 v13, v17, v21 // (op 12866) 




s_lshl_b32 s0, s4, 7 // (op 12846) 
s_add_u32 s0, s0, 192 // (op 12846) 
v_add_u32 v3, v9, s0 // (op 12846) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12846) Store value




s_lshl_b32 s0, s2, 6 // (op 12871) 
s_mul_i32 s1, 3, s0 // (op 12871) 
s_add_u32 s1, s1, 192 // (op 12871) 

v_add_u32 v3, v8, s1 // (op 12871) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12871) Load value

v_accvgpr_read v14, a108 // (op 12893) 
v_accvgpr_read v15, a109 // (op 12893) 
v_accvgpr_read v16, a110 // (op 12893) 
v_accvgpr_read v17, a111 // (op 12893) 
v_mul_f32 v18, v7, v14 // (op 12893) 
v_mul_f32 v19, v7, v15 // (op 12893) 
v_mul_f32 v20, v7, v16 // (op 12893) 
v_mul_f32 v21, v7, v17 // (op 12893) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12869) 
v_mul_f32 v15, v0, v11 // (op 12869) 
v_mul_f32 v16, v0, v12 // (op 12869) 
v_mul_f32 v17, v0, v13 // (op 12869) 

v_add_f32 v10, v14, v18 // (op 12870) 
v_add_f32 v11, v15, v19 // (op 12870) 
v_add_f32 v12, v16, v20 // (op 12870) 
v_add_f32 v13, v17, v21 // (op 12870) 




s_lshl_b32 s0, s4, 6 // (op 12848) 
s_mul_i32 s1, 3, s0 // (op 12848) 
s_add_u32 s1, s1, 192 // (op 12848) 

v_add_u32 v3, v9, s1 // (op 12848) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12848) Store value




s_lshl_b32 s0, s2, 8 // (op 12875) 
s_add_u32 s0, s0, 192 // (op 12875) 
v_add_u32 v3, v8, s0 // (op 12875) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12875) Load value

v_accvgpr_read v14, a112 // (op 12895) 
v_accvgpr_read v15, a113 // (op 12895) 
v_accvgpr_read v16, a114 // (op 12895) 
v_accvgpr_read v17, a115 // (op 12895) 
v_mul_f32 v18, v7, v14 // (op 12895) 
v_mul_f32 v19, v7, v15 // (op 12895) 
v_mul_f32 v20, v7, v16 // (op 12895) 
v_mul_f32 v21, v7, v17 // (op 12895) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12873) 
v_mul_f32 v15, v0, v11 // (op 12873) 
v_mul_f32 v16, v0, v12 // (op 12873) 
v_mul_f32 v17, v0, v13 // (op 12873) 

v_add_f32 v10, v14, v18 // (op 12874) 
v_add_f32 v11, v15, v19 // (op 12874) 
v_add_f32 v12, v16, v20 // (op 12874) 
v_add_f32 v13, v17, v21 // (op 12874) 




s_lshl_b32 s0, s4, 8 // (op 12850) 
s_add_u32 s0, s0, 192 // (op 12850) 
v_add_u32 v3, v9, s0 // (op 12850) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12850) Store value




s_lshl_b32 s0, s2, 6 // (op 12879) 
s_mul_i32 s1, 5, s0 // (op 12879) 
s_add_u32 s1, s1, 192 // (op 12879) 

v_add_u32 v3, v8, s1 // (op 12879) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12879) Load value

v_accvgpr_read v14, a116 // (op 12897) 
v_accvgpr_read v15, a117 // (op 12897) 
v_accvgpr_read v16, a118 // (op 12897) 
v_accvgpr_read v17, a119 // (op 12897) 
v_mul_f32 v18, v7, v14 // (op 12897) 
v_mul_f32 v19, v7, v15 // (op 12897) 
v_mul_f32 v20, v7, v16 // (op 12897) 
v_mul_f32 v21, v7, v17 // (op 12897) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12877) 
v_mul_f32 v15, v0, v11 // (op 12877) 
v_mul_f32 v16, v0, v12 // (op 12877) 
v_mul_f32 v17, v0, v13 // (op 12877) 

v_add_f32 v10, v14, v18 // (op 12878) 
v_add_f32 v11, v15, v19 // (op 12878) 
v_add_f32 v12, v16, v20 // (op 12878) 
v_add_f32 v13, v17, v21 // (op 12878) 




s_lshl_b32 s0, s4, 6 // (op 12852) 
s_mul_i32 s1, 5, s0 // (op 12852) 
s_add_u32 s1, s1, 192 // (op 12852) 

v_add_u32 v3, v9, s1 // (op 12852) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12852) Store value




s_lshl_b32 s0, s2, 6 // (op 12883) 
s_mul_i32 s1, 6, s0 // (op 12883) 
s_add_u32 s1, s1, 192 // (op 12883) 

v_add_u32 v3, v8, s1 // (op 12883) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12883) Load value

v_accvgpr_read v14, a120 // (op 12899) 
v_accvgpr_read v15, a121 // (op 12899) 
v_accvgpr_read v16, a122 // (op 12899) 
v_accvgpr_read v17, a123 // (op 12899) 
v_mul_f32 v18, v7, v14 // (op 12899) 
v_mul_f32 v19, v7, v15 // (op 12899) 
v_mul_f32 v20, v7, v16 // (op 12899) 
v_mul_f32 v21, v7, v17 // (op 12899) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12881) 
v_mul_f32 v15, v0, v11 // (op 12881) 
v_mul_f32 v16, v0, v12 // (op 12881) 
v_mul_f32 v17, v0, v13 // (op 12881) 

v_add_f32 v10, v14, v18 // (op 12882) 
v_add_f32 v11, v15, v19 // (op 12882) 
v_add_f32 v12, v16, v20 // (op 12882) 
v_add_f32 v13, v17, v21 // (op 12882) 




s_lshl_b32 s0, s4, 6 // (op 12854) 
s_mul_i32 s1, 6, s0 // (op 12854) 
s_add_u32 s1, s1, 192 // (op 12854) 

v_add_u32 v3, v9, s1 // (op 12854) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12854) Store value




s_lshl_b32 s0, s2, 6 // (op 12887) 
s_mul_i32 s1, 7, s0 // (op 12887) 
s_add_u32 s1, s1, 192 // (op 12887) 

v_add_u32 v3, v8, s1 // (op 12887) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12887) Load value

v_accvgpr_read v14, a124 // (op 12901) 
v_accvgpr_read v15, a125 // (op 12901) 
v_accvgpr_read v16, a126 // (op 12901) 
v_accvgpr_read v17, a127 // (op 12901) 
v_mul_f32 v18, v7, v14 // (op 12901) 
v_mul_f32 v19, v7, v15 // (op 12901) 
v_mul_f32 v20, v7, v16 // (op 12901) 
v_mul_f32 v21, v7, v17 // (op 12901) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12885) 
v_mul_f32 v15, v0, v11 // (op 12885) 
v_mul_f32 v16, v0, v12 // (op 12885) 
v_mul_f32 v17, v0, v13 // (op 12885) 

v_add_f32 v10, v14, v18 // (op 12886) 
v_add_f32 v11, v15, v19 // (op 12886) 
v_add_f32 v12, v16, v20 // (op 12886) 
v_add_f32 v13, v17, v21 // (op 12886) 




s_lshl_b32 s0, s4, 6 // (op 12856) 
s_mul_i32 s1, 7, s0 // (op 12856) 
s_add_u32 s1, s1, 192 // (op 12856) 

v_add_u32 v3, v9, s1 // (op 12856) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12856) Store value




v_add_u32 v3, 128, v8 // (op 12704) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12704) Load value

v_accvgpr_read v14, a64 // (op 12684) 
v_accvgpr_read v15, a65 // (op 12684) 
v_accvgpr_read v16, a66 // (op 12684) 
v_accvgpr_read v17, a67 // (op 12684) 
v_mul_f32 v18, v7, v14 // (op 12684) 
v_mul_f32 v19, v7, v15 // (op 12684) 
v_mul_f32 v20, v7, v16 // (op 12684) 
v_mul_f32 v21, v7, v17 // (op 12684) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12703) 
v_mul_f32 v15, v0, v11 // (op 12703) 
v_mul_f32 v16, v0, v12 // (op 12703) 
v_mul_f32 v17, v0, v13 // (op 12703) 

v_add_f32 v10, v14, v18 // (op 12685) 
v_add_f32 v11, v15, v19 // (op 12685) 
v_add_f32 v12, v16, v20 // (op 12685) 
v_add_f32 v13, v17, v21 // (op 12685) 




v_add_u32 v3, 128, v9 // (op 12687) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12687) Store value




s_lshl_b32 s0, s2, 6 // (op 12708) 
s_add_u32 s0, s0, 128 // (op 12708) 
v_add_u32 v3, v8, s0 // (op 12708) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12708) Load value

v_accvgpr_read v14, a68 // (op 12734) 
v_accvgpr_read v15, a69 // (op 12734) 
v_accvgpr_read v16, a70 // (op 12734) 
v_accvgpr_read v17, a71 // (op 12734) 
v_mul_f32 v18, v7, v14 // (op 12734) 
v_mul_f32 v19, v7, v15 // (op 12734) 
v_mul_f32 v20, v7, v16 // (op 12734) 
v_mul_f32 v21, v7, v17 // (op 12734) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12706) 
v_mul_f32 v15, v0, v11 // (op 12706) 
v_mul_f32 v16, v0, v12 // (op 12706) 
v_mul_f32 v17, v0, v13 // (op 12706) 

v_add_f32 v10, v14, v18 // (op 12707) 
v_add_f32 v11, v15, v19 // (op 12707) 
v_add_f32 v12, v16, v20 // (op 12707) 
v_add_f32 v13, v17, v21 // (op 12707) 




s_lshl_b32 s0, s4, 6 // (op 12689) 
s_add_u32 s0, s0, 128 // (op 12689) 
v_add_u32 v3, v9, s0 // (op 12689) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12689) Store value




s_lshl_b32 s0, s2, 7 // (op 12712) 
s_add_u32 s0, s0, 128 // (op 12712) 
v_add_u32 v3, v8, s0 // (op 12712) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12712) Load value

v_accvgpr_read v14, a72 // (op 12736) 
v_accvgpr_read v15, a73 // (op 12736) 
v_accvgpr_read v16, a74 // (op 12736) 
v_accvgpr_read v17, a75 // (op 12736) 
v_mul_f32 v18, v7, v14 // (op 12736) 
v_mul_f32 v19, v7, v15 // (op 12736) 
v_mul_f32 v20, v7, v16 // (op 12736) 
v_mul_f32 v21, v7, v17 // (op 12736) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 12710) 
v_mul_f32 v15, v0, v11 // (op 12710) 
v_mul_f32 v16, v0, v12 // (op 12710) 
v_mul_f32 v17, v0, v13 // (op 12710) 

v_add_f32 v10, v14, v18 // (op 12711) 
v_add_f32 v11, v15, v19 // (op 12711) 
v_add_f32 v12, v16, v20 // (op 12711) 
v_add_f32 v13, v17, v21 // (op 12711) 




s_lshl_b32 s0, s4, 7 // (op 12691) 
s_add_u32 s0, s0, 128 // (op 12691) 
v_add_u32 v3, v9, s0 // (op 12691) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12691) Store value




s_lshl_b32 s0, s2, 6 // (op 12716) 
s_mul_i32 s1, 3, s0 // (op 12716) 
s_add_u32 s1, s1, 128 // (op 12716) 

v_add_u32 v3, v8, s1 // (op 12716) 


v_accvgpr_read v10, a76 // (op 12738) 
v_accvgpr_read v11, a77 // (op 12738) 
v_accvgpr_read v12, a78 // (op 12738) 
v_accvgpr_read v13, a79 // (op 12738) 
v_mul_f32 v14, v7, v10 // (op 12738) 
v_mul_f32 v15, v7, v11 // (op 12738) 
v_mul_f32 v16, v7, v12 // (op 12738) 
v_mul_f32 v17, v7, v13 // (op 12738) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12716) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12714) 
v_mul_f32 v19, v0, v11 // (op 12714) 
v_mul_f32 v20, v0, v12 // (op 12714) 
v_mul_f32 v21, v0, v13 // (op 12714) 

v_add_f32 v10, v18, v14 // (op 12715) 
v_add_f32 v11, v19, v15 // (op 12715) 
v_add_f32 v12, v20, v16 // (op 12715) 
v_add_f32 v13, v21, v17 // (op 12715) 




s_lshl_b32 s0, s4, 6 // (op 12693) 
s_mul_i32 s1, 3, s0 // (op 12693) 
s_add_u32 s1, s1, 128 // (op 12693) 

v_add_u32 v3, v9, s1 // (op 12693) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12693) Store value




s_lshl_b32 s0, s2, 8 // (op 12720) 
s_add_u32 s0, s0, 128 // (op 12720) 
v_add_u32 v3, v8, s0 // (op 12720) 


v_accvgpr_read v10, a80 // (op 12740) 
v_accvgpr_read v11, a81 // (op 12740) 
v_accvgpr_read v12, a82 // (op 12740) 
v_accvgpr_read v13, a83 // (op 12740) 
v_mul_f32 v14, v7, v10 // (op 12740) 
v_mul_f32 v15, v7, v11 // (op 12740) 
v_mul_f32 v16, v7, v12 // (op 12740) 
v_mul_f32 v17, v7, v13 // (op 12740) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12720) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12718) 
v_mul_f32 v19, v0, v11 // (op 12718) 
v_mul_f32 v20, v0, v12 // (op 12718) 
v_mul_f32 v21, v0, v13 // (op 12718) 

v_add_f32 v10, v18, v14 // (op 12719) 
v_add_f32 v11, v19, v15 // (op 12719) 
v_add_f32 v12, v20, v16 // (op 12719) 
v_add_f32 v13, v21, v17 // (op 12719) 




s_lshl_b32 s0, s4, 8 // (op 12695) 
s_add_u32 s0, s0, 128 // (op 12695) 
v_add_u32 v3, v9, s0 // (op 12695) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12695) Store value




s_lshl_b32 s0, s2, 6 // (op 12724) 
s_mul_i32 s1, 5, s0 // (op 12724) 
s_add_u32 s1, s1, 128 // (op 12724) 

v_add_u32 v3, v8, s1 // (op 12724) 


v_accvgpr_read v10, a84 // (op 12742) 
v_accvgpr_read v11, a85 // (op 12742) 
v_accvgpr_read v12, a86 // (op 12742) 
v_accvgpr_read v13, a87 // (op 12742) 
v_mul_f32 v14, v7, v10 // (op 12742) 
v_mul_f32 v15, v7, v11 // (op 12742) 
v_mul_f32 v16, v7, v12 // (op 12742) 
v_mul_f32 v17, v7, v13 // (op 12742) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12724) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12722) 
v_mul_f32 v19, v0, v11 // (op 12722) 
v_mul_f32 v20, v0, v12 // (op 12722) 
v_mul_f32 v21, v0, v13 // (op 12722) 

v_add_f32 v10, v18, v14 // (op 12723) 
v_add_f32 v11, v19, v15 // (op 12723) 
v_add_f32 v12, v20, v16 // (op 12723) 
v_add_f32 v13, v21, v17 // (op 12723) 




s_lshl_b32 s0, s4, 6 // (op 12697) 
s_mul_i32 s1, 5, s0 // (op 12697) 
s_add_u32 s1, s1, 128 // (op 12697) 

v_add_u32 v3, v9, s1 // (op 12697) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12697) Store value




s_lshl_b32 s0, s2, 6 // (op 12728) 
s_mul_i32 s1, 6, s0 // (op 12728) 
s_add_u32 s1, s1, 128 // (op 12728) 

v_add_u32 v3, v8, s1 // (op 12728) 


v_accvgpr_read v10, a88 // (op 12744) 
v_accvgpr_read v11, a89 // (op 12744) 
v_accvgpr_read v12, a90 // (op 12744) 
v_accvgpr_read v13, a91 // (op 12744) 
v_mul_f32 v14, v7, v10 // (op 12744) 
v_mul_f32 v15, v7, v11 // (op 12744) 
v_mul_f32 v16, v7, v12 // (op 12744) 
v_mul_f32 v17, v7, v13 // (op 12744) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12728) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12726) 
v_mul_f32 v19, v0, v11 // (op 12726) 
v_mul_f32 v20, v0, v12 // (op 12726) 
v_mul_f32 v21, v0, v13 // (op 12726) 

v_add_f32 v10, v18, v14 // (op 12727) 
v_add_f32 v11, v19, v15 // (op 12727) 
v_add_f32 v12, v20, v16 // (op 12727) 
v_add_f32 v13, v21, v17 // (op 12727) 




s_lshl_b32 s0, s4, 6 // (op 12699) 
s_mul_i32 s1, 6, s0 // (op 12699) 
s_add_u32 s1, s1, 128 // (op 12699) 

v_add_u32 v3, v9, s1 // (op 12699) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12699) Store value




s_lshl_b32 s0, s2, 6 // (op 12732) 
s_mul_i32 s1, 7, s0 // (op 12732) 
s_add_u32 s1, s1, 128 // (op 12732) 

v_add_u32 v3, v8, s1 // (op 12732) 


v_accvgpr_read v10, a92 // (op 12746) 
v_accvgpr_read v11, a93 // (op 12746) 
v_accvgpr_read v12, a94 // (op 12746) 
v_accvgpr_read v13, a95 // (op 12746) 
v_mul_f32 v14, v7, v10 // (op 12746) 
v_mul_f32 v15, v7, v11 // (op 12746) 
v_mul_f32 v16, v7, v12 // (op 12746) 
v_mul_f32 v17, v7, v13 // (op 12746) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12732) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12730) 
v_mul_f32 v19, v0, v11 // (op 12730) 
v_mul_f32 v20, v0, v12 // (op 12730) 
v_mul_f32 v21, v0, v13 // (op 12730) 

v_add_f32 v10, v18, v14 // (op 12731) 
v_add_f32 v11, v19, v15 // (op 12731) 
v_add_f32 v12, v20, v16 // (op 12731) 
v_add_f32 v13, v21, v17 // (op 12731) 




s_lshl_b32 s0, s4, 6 // (op 12701) 
s_mul_i32 s1, 7, s0 // (op 12701) 
s_add_u32 s1, s1, 128 // (op 12701) 

v_add_u32 v3, v9, s1 // (op 12701) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12701) Store value




v_add_u32 v3, 64, v8 // (op 12549) 

s_nop 0
v_accvgpr_read v10, a32 // (op 12529) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v11, a33 // (op 12529) 
v_accvgpr_read v12, a34 // (op 12529) 
v_accvgpr_read v13, a35 // (op 12529) 
v_mul_f32 v14, v7, v10 // (op 12529) 
v_mul_f32 v15, v7, v11 // (op 12529) 
v_mul_f32 v16, v7, v12 // (op 12529) 
v_mul_f32 v17, v7, v13 // (op 12529) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12549) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12548) 
v_mul_f32 v19, v0, v11 // (op 12548) 
v_mul_f32 v20, v0, v12 // (op 12548) 
v_mul_f32 v21, v0, v13 // (op 12548) 

v_add_f32 v10, v18, v14 // (op 12530) 
v_add_f32 v11, v19, v15 // (op 12530) 
v_add_f32 v12, v20, v16 // (op 12530) 
v_add_f32 v13, v21, v17 // (op 12530) 




v_add_u32 v3, 64, v9 // (op 12532) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12532) Store value




s_lshl_b32 s0, s2, 6 // (op 12553) 
s_add_u32 s0, s0, 64 // (op 12553) 
v_add_u32 v3, v8, s0 // (op 12553) 


v_accvgpr_read v10, a36 // (op 12579) 
v_accvgpr_read v11, a37 // (op 12579) 
v_accvgpr_read v12, a38 // (op 12579) 
v_accvgpr_read v13, a39 // (op 12579) 
v_mul_f32 v14, v7, v10 // (op 12579) 
v_mul_f32 v15, v7, v11 // (op 12579) 
v_mul_f32 v16, v7, v12 // (op 12579) 
v_mul_f32 v17, v7, v13 // (op 12579) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12553) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12551) 
v_mul_f32 v19, v0, v11 // (op 12551) 
v_mul_f32 v20, v0, v12 // (op 12551) 
v_mul_f32 v21, v0, v13 // (op 12551) 

v_add_f32 v10, v18, v14 // (op 12552) 
v_add_f32 v11, v19, v15 // (op 12552) 
v_add_f32 v12, v20, v16 // (op 12552) 
v_add_f32 v13, v21, v17 // (op 12552) 




s_lshl_b32 s0, s4, 6 // (op 12534) 
s_add_u32 s0, s0, 64 // (op 12534) 
v_add_u32 v3, v9, s0 // (op 12534) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12534) Store value




s_lshl_b32 s0, s2, 7 // (op 12557) 
s_add_u32 s0, s0, 64 // (op 12557) 
v_add_u32 v3, v8, s0 // (op 12557) 


v_accvgpr_read v10, a40 // (op 12581) 
v_accvgpr_read v11, a41 // (op 12581) 
v_accvgpr_read v12, a42 // (op 12581) 
v_accvgpr_read v13, a43 // (op 12581) 
v_mul_f32 v14, v7, v10 // (op 12581) 
v_mul_f32 v15, v7, v11 // (op 12581) 
v_mul_f32 v16, v7, v12 // (op 12581) 
v_mul_f32 v17, v7, v13 // (op 12581) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12557) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12555) 
v_mul_f32 v19, v0, v11 // (op 12555) 
v_mul_f32 v20, v0, v12 // (op 12555) 
v_mul_f32 v21, v0, v13 // (op 12555) 

v_add_f32 v10, v18, v14 // (op 12556) 
v_add_f32 v11, v19, v15 // (op 12556) 
v_add_f32 v12, v20, v16 // (op 12556) 
v_add_f32 v13, v21, v17 // (op 12556) 




s_lshl_b32 s0, s4, 7 // (op 12536) 
s_add_u32 s0, s0, 64 // (op 12536) 
v_add_u32 v3, v9, s0 // (op 12536) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12536) Store value




s_lshl_b32 s0, s2, 6 // (op 12561) 
s_mul_i32 s1, 3, s0 // (op 12561) 
s_add_u32 s1, s1, 64 // (op 12561) 

v_add_u32 v3, v8, s1 // (op 12561) 


v_accvgpr_read v10, a44 // (op 12583) 
v_accvgpr_read v11, a45 // (op 12583) 
v_accvgpr_read v12, a46 // (op 12583) 
v_accvgpr_read v13, a47 // (op 12583) 
v_mul_f32 v14, v7, v10 // (op 12583) 
v_mul_f32 v15, v7, v11 // (op 12583) 
v_mul_f32 v16, v7, v12 // (op 12583) 
v_mul_f32 v17, v7, v13 // (op 12583) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12561) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12559) 
v_mul_f32 v19, v0, v11 // (op 12559) 
v_mul_f32 v20, v0, v12 // (op 12559) 
v_mul_f32 v21, v0, v13 // (op 12559) 

v_add_f32 v10, v18, v14 // (op 12560) 
v_add_f32 v11, v19, v15 // (op 12560) 
v_add_f32 v12, v20, v16 // (op 12560) 
v_add_f32 v13, v21, v17 // (op 12560) 




s_lshl_b32 s0, s4, 6 // (op 12538) 
s_mul_i32 s1, 3, s0 // (op 12538) 
s_add_u32 s1, s1, 64 // (op 12538) 

v_add_u32 v3, v9, s1 // (op 12538) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12538) Store value




s_lshl_b32 s0, s2, 8 // (op 12565) 
s_add_u32 s0, s0, 64 // (op 12565) 
v_add_u32 v3, v8, s0 // (op 12565) 


v_accvgpr_read v10, a48 // (op 12585) 
v_accvgpr_read v11, a49 // (op 12585) 
v_accvgpr_read v12, a50 // (op 12585) 
v_accvgpr_read v13, a51 // (op 12585) 
v_mul_f32 v14, v7, v10 // (op 12585) 
v_mul_f32 v15, v7, v11 // (op 12585) 
v_mul_f32 v16, v7, v12 // (op 12585) 
v_mul_f32 v17, v7, v13 // (op 12585) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12565) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12563) 
v_mul_f32 v19, v0, v11 // (op 12563) 
v_mul_f32 v20, v0, v12 // (op 12563) 
v_mul_f32 v21, v0, v13 // (op 12563) 

v_add_f32 v10, v18, v14 // (op 12564) 
v_add_f32 v11, v19, v15 // (op 12564) 
v_add_f32 v12, v20, v16 // (op 12564) 
v_add_f32 v13, v21, v17 // (op 12564) 




s_lshl_b32 s0, s4, 8 // (op 12540) 
s_add_u32 s0, s0, 64 // (op 12540) 
v_add_u32 v3, v9, s0 // (op 12540) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12540) Store value




s_lshl_b32 s0, s2, 6 // (op 12569) 
s_mul_i32 s1, 5, s0 // (op 12569) 
s_add_u32 s1, s1, 64 // (op 12569) 

v_add_u32 v3, v8, s1 // (op 12569) 


v_accvgpr_read v10, a52 // (op 12587) 
v_accvgpr_read v11, a53 // (op 12587) 
v_accvgpr_read v12, a54 // (op 12587) 
v_accvgpr_read v13, a55 // (op 12587) 
v_mul_f32 v14, v7, v10 // (op 12587) 
v_mul_f32 v15, v7, v11 // (op 12587) 
v_mul_f32 v16, v7, v12 // (op 12587) 
v_mul_f32 v17, v7, v13 // (op 12587) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12569) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12567) 
v_mul_f32 v19, v0, v11 // (op 12567) 
v_mul_f32 v20, v0, v12 // (op 12567) 
v_mul_f32 v21, v0, v13 // (op 12567) 

v_add_f32 v10, v18, v14 // (op 12568) 
v_add_f32 v11, v19, v15 // (op 12568) 
v_add_f32 v12, v20, v16 // (op 12568) 
v_add_f32 v13, v21, v17 // (op 12568) 




s_lshl_b32 s0, s4, 6 // (op 12542) 
s_mul_i32 s1, 5, s0 // (op 12542) 
s_add_u32 s1, s1, 64 // (op 12542) 

v_add_u32 v3, v9, s1 // (op 12542) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12542) Store value




s_lshl_b32 s0, s2, 6 // (op 12573) 
s_mul_i32 s1, 6, s0 // (op 12573) 
s_add_u32 s1, s1, 64 // (op 12573) 

v_add_u32 v3, v8, s1 // (op 12573) 


v_accvgpr_read v10, a56 // (op 12589) 
v_accvgpr_read v11, a57 // (op 12589) 
v_accvgpr_read v12, a58 // (op 12589) 
v_accvgpr_read v13, a59 // (op 12589) 
v_mul_f32 v14, v7, v10 // (op 12589) 
v_mul_f32 v15, v7, v11 // (op 12589) 
v_mul_f32 v16, v7, v12 // (op 12589) 
v_mul_f32 v17, v7, v13 // (op 12589) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12573) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12571) 
v_mul_f32 v19, v0, v11 // (op 12571) 
v_mul_f32 v20, v0, v12 // (op 12571) 
v_mul_f32 v21, v0, v13 // (op 12571) 

v_add_f32 v10, v18, v14 // (op 12572) 
v_add_f32 v11, v19, v15 // (op 12572) 
v_add_f32 v12, v20, v16 // (op 12572) 
v_add_f32 v13, v21, v17 // (op 12572) 




s_lshl_b32 s0, s4, 6 // (op 12544) 
s_mul_i32 s1, 6, s0 // (op 12544) 
s_add_u32 s1, s1, 64 // (op 12544) 

v_add_u32 v3, v9, s1 // (op 12544) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12544) Store value




s_lshl_b32 s0, s2, 6 // (op 12577) 
s_mul_i32 s1, 7, s0 // (op 12577) 
s_add_u32 s1, s1, 64 // (op 12577) 

v_add_u32 v3, v8, s1 // (op 12577) 


v_accvgpr_read v10, a60 // (op 12591) 
v_accvgpr_read v11, a61 // (op 12591) 
v_accvgpr_read v12, a62 // (op 12591) 
v_accvgpr_read v13, a63 // (op 12591) 
v_mul_f32 v14, v7, v10 // (op 12591) 
v_mul_f32 v15, v7, v11 // (op 12591) 
v_mul_f32 v16, v7, v12 // (op 12591) 
v_mul_f32 v17, v7, v13 // (op 12591) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12577) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12575) 
v_mul_f32 v19, v0, v11 // (op 12575) 
v_mul_f32 v20, v0, v12 // (op 12575) 
v_mul_f32 v21, v0, v13 // (op 12575) 

v_add_f32 v10, v18, v14 // (op 12576) 
v_add_f32 v11, v19, v15 // (op 12576) 
v_add_f32 v12, v20, v16 // (op 12576) 
v_add_f32 v13, v21, v17 // (op 12576) 




s_lshl_b32 s0, s4, 6 // (op 12546) 
s_mul_i32 s1, 7, s0 // (op 12546) 
s_add_u32 s1, s1, 64 // (op 12546) 

v_add_u32 v3, v9, s1 // (op 12546) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12546) Store value




v_add_u32 v3, 320, v8 // (op 13169) 

s_nop 0
v_accvgpr_read v10, a160 // (op 13149) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v11, a161 // (op 13149) 
v_accvgpr_read v12, a162 // (op 13149) 
v_accvgpr_read v13, a163 // (op 13149) 
v_mul_f32 v14, v7, v10 // (op 13149) 
v_mul_f32 v15, v7, v11 // (op 13149) 
v_mul_f32 v16, v7, v12 // (op 13149) 
v_mul_f32 v17, v7, v13 // (op 13149) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13169) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 13168) 
v_mul_f32 v19, v0, v11 // (op 13168) 
v_mul_f32 v20, v0, v12 // (op 13168) 
v_mul_f32 v21, v0, v13 // (op 13168) 

v_add_f32 v10, v18, v14 // (op 13150) 
v_add_f32 v11, v19, v15 // (op 13150) 
v_add_f32 v12, v20, v16 // (op 13150) 
v_add_f32 v13, v21, v17 // (op 13150) 




v_add_u32 v3, 320, v9 // (op 13152) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13152) Store value




s_lshl_b32 s0, s2, 6 // (op 13173) 
s_add_u32 s0, s0, 320 // (op 13173) 
v_add_u32 v3, v8, s0 // (op 13173) 


v_accvgpr_read v10, a164 // (op 13199) 
v_accvgpr_read v11, a165 // (op 13199) 
v_accvgpr_read v12, a166 // (op 13199) 
v_accvgpr_read v13, a167 // (op 13199) 
v_mul_f32 v14, v7, v10 // (op 13199) 
v_mul_f32 v15, v7, v11 // (op 13199) 
v_mul_f32 v16, v7, v12 // (op 13199) 
v_mul_f32 v17, v7, v13 // (op 13199) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13173) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 13171) 
v_mul_f32 v19, v0, v11 // (op 13171) 
v_mul_f32 v20, v0, v12 // (op 13171) 
v_mul_f32 v21, v0, v13 // (op 13171) 

v_add_f32 v10, v18, v14 // (op 13172) 
v_add_f32 v11, v19, v15 // (op 13172) 
v_add_f32 v12, v20, v16 // (op 13172) 
v_add_f32 v13, v21, v17 // (op 13172) 




s_lshl_b32 s0, s4, 6 // (op 13154) 
s_add_u32 s0, s0, 320 // (op 13154) 
v_add_u32 v3, v9, s0 // (op 13154) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13154) Store value




s_lshl_b32 s0, s2, 7 // (op 13177) 
s_add_u32 s0, s0, 320 // (op 13177) 
v_add_u32 v3, v8, s0 // (op 13177) 


v_accvgpr_read v10, a168 // (op 13201) 
v_accvgpr_read v11, a169 // (op 13201) 
v_accvgpr_read v12, a170 // (op 13201) 
v_accvgpr_read v13, a171 // (op 13201) 
v_mul_f32 v14, v7, v10 // (op 13201) 
v_mul_f32 v15, v7, v11 // (op 13201) 
v_mul_f32 v16, v7, v12 // (op 13201) 
v_mul_f32 v17, v7, v13 // (op 13201) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13177) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 13175) 
v_mul_f32 v19, v0, v11 // (op 13175) 
v_mul_f32 v20, v0, v12 // (op 13175) 
v_mul_f32 v21, v0, v13 // (op 13175) 

v_add_f32 v10, v18, v14 // (op 13176) 
v_add_f32 v11, v19, v15 // (op 13176) 
v_add_f32 v12, v20, v16 // (op 13176) 
v_add_f32 v13, v21, v17 // (op 13176) 




s_lshl_b32 s0, s4, 7 // (op 13156) 
s_add_u32 s0, s0, 320 // (op 13156) 
v_add_u32 v3, v9, s0 // (op 13156) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13156) Store value




s_lshl_b32 s0, s2, 6 // (op 13181) 
s_mul_i32 s1, 3, s0 // (op 13181) 
s_add_u32 s1, s1, 320 // (op 13181) 

v_add_u32 v3, v8, s1 // (op 13181) 


v_accvgpr_read v10, a172 // (op 13203) 
v_accvgpr_read v11, a173 // (op 13203) 
v_accvgpr_read v12, a174 // (op 13203) 
v_accvgpr_read v13, a175 // (op 13203) 
v_mul_f32 v14, v7, v10 // (op 13203) 
v_mul_f32 v15, v7, v11 // (op 13203) 
v_mul_f32 v16, v7, v12 // (op 13203) 
v_mul_f32 v17, v7, v13 // (op 13203) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13181) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 13179) 
v_mul_f32 v19, v0, v11 // (op 13179) 
v_mul_f32 v20, v0, v12 // (op 13179) 
v_mul_f32 v21, v0, v13 // (op 13179) 

v_add_f32 v10, v18, v14 // (op 13180) 
v_add_f32 v11, v19, v15 // (op 13180) 
v_add_f32 v12, v20, v16 // (op 13180) 
v_add_f32 v13, v21, v17 // (op 13180) 




s_lshl_b32 s0, s4, 6 // (op 13158) 
s_mul_i32 s1, 3, s0 // (op 13158) 
s_add_u32 s1, s1, 320 // (op 13158) 

v_add_u32 v3, v9, s1 // (op 13158) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13158) Store value




s_lshl_b32 s0, s2, 8 // (op 13185) 
s_add_u32 s0, s0, 320 // (op 13185) 
v_add_u32 v3, v8, s0 // (op 13185) 


v_accvgpr_read v10, a176 // (op 13205) 
v_accvgpr_read v11, a177 // (op 13205) 
v_accvgpr_read v12, a178 // (op 13205) 
v_accvgpr_read v13, a179 // (op 13205) 
v_mul_f32 v14, v7, v10 // (op 13205) 
v_mul_f32 v15, v7, v11 // (op 13205) 
v_mul_f32 v16, v7, v12 // (op 13205) 
v_mul_f32 v17, v7, v13 // (op 13205) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13185) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 13183) 
v_mul_f32 v19, v0, v11 // (op 13183) 
v_mul_f32 v20, v0, v12 // (op 13183) 
v_mul_f32 v21, v0, v13 // (op 13183) 

v_add_f32 v10, v18, v14 // (op 13184) 
v_add_f32 v11, v19, v15 // (op 13184) 
v_add_f32 v12, v20, v16 // (op 13184) 
v_add_f32 v13, v21, v17 // (op 13184) 




s_lshl_b32 s0, s4, 8 // (op 13160) 
s_add_u32 s0, s0, 320 // (op 13160) 
v_add_u32 v3, v9, s0 // (op 13160) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13160) Store value




s_lshl_b32 s0, s2, 6 // (op 13189) 
s_mul_i32 s1, 5, s0 // (op 13189) 
s_add_u32 s1, s1, 320 // (op 13189) 

v_add_u32 v3, v8, s1 // (op 13189) 


v_accvgpr_read v10, a180 // (op 13207) 
v_accvgpr_read v11, a181 // (op 13207) 
v_accvgpr_read v12, a182 // (op 13207) 
v_accvgpr_read v13, a183 // (op 13207) 
v_mul_f32 v14, v7, v10 // (op 13207) 
v_mul_f32 v15, v7, v11 // (op 13207) 
v_mul_f32 v16, v7, v12 // (op 13207) 
v_mul_f32 v17, v7, v13 // (op 13207) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13189) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 13187) 
v_mul_f32 v19, v0, v11 // (op 13187) 
v_mul_f32 v20, v0, v12 // (op 13187) 
v_mul_f32 v21, v0, v13 // (op 13187) 

v_add_f32 v10, v18, v14 // (op 13188) 
v_add_f32 v11, v19, v15 // (op 13188) 
v_add_f32 v12, v20, v16 // (op 13188) 
v_add_f32 v13, v21, v17 // (op 13188) 




s_lshl_b32 s0, s4, 6 // (op 13162) 
s_mul_i32 s1, 5, s0 // (op 13162) 
s_add_u32 s1, s1, 320 // (op 13162) 

v_add_u32 v3, v9, s1 // (op 13162) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13162) Store value




s_lshl_b32 s0, s2, 6 // (op 13193) 
s_mul_i32 s1, 6, s0 // (op 13193) 
s_add_u32 s1, s1, 320 // (op 13193) 

v_add_u32 v3, v8, s1 // (op 13193) 


v_accvgpr_read v10, a184 // (op 13209) 
v_accvgpr_read v11, a185 // (op 13209) 
v_accvgpr_read v12, a186 // (op 13209) 
v_accvgpr_read v13, a187 // (op 13209) 
v_mul_f32 v14, v7, v10 // (op 13209) 
v_mul_f32 v15, v7, v11 // (op 13209) 
v_mul_f32 v16, v7, v12 // (op 13209) 
v_mul_f32 v17, v7, v13 // (op 13209) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13193) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 13191) 
v_mul_f32 v19, v0, v11 // (op 13191) 
v_mul_f32 v20, v0, v12 // (op 13191) 
v_mul_f32 v21, v0, v13 // (op 13191) 

v_add_f32 v10, v18, v14 // (op 13192) 
v_add_f32 v11, v19, v15 // (op 13192) 
v_add_f32 v12, v20, v16 // (op 13192) 
v_add_f32 v13, v21, v17 // (op 13192) 




s_lshl_b32 s0, s4, 6 // (op 13164) 
s_mul_i32 s1, 6, s0 // (op 13164) 
s_add_u32 s1, s1, 320 // (op 13164) 

v_add_u32 v3, v9, s1 // (op 13164) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13164) Store value




s_lshl_b32 s0, s2, 6 // (op 13197) 
s_mul_i32 s1, 7, s0 // (op 13197) 
s_add_u32 s1, s1, 320 // (op 13197) 

v_add_u32 v3, v8, s1 // (op 13197) 


v_accvgpr_read v10, a188 // (op 13211) 
v_accvgpr_read v11, a189 // (op 13211) 
v_accvgpr_read v12, a190 // (op 13211) 
v_accvgpr_read v13, a191 // (op 13211) 
v_mul_f32 v14, v7, v10 // (op 13211) 
v_mul_f32 v15, v7, v11 // (op 13211) 
v_mul_f32 v16, v7, v12 // (op 13211) 
v_mul_f32 v17, v7, v13 // (op 13211) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 13197) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 13195) 
v_mul_f32 v19, v0, v11 // (op 13195) 
v_mul_f32 v20, v0, v12 // (op 13195) 
v_mul_f32 v21, v0, v13 // (op 13195) 

v_add_f32 v10, v18, v14 // (op 13196) 
v_add_f32 v11, v19, v15 // (op 13196) 
v_add_f32 v12, v20, v16 // (op 13196) 
v_add_f32 v13, v21, v17 // (op 13196) 




s_lshl_b32 s0, s4, 6 // (op 13166) 
s_mul_i32 s1, 7, s0 // (op 13166) 
s_add_u32 s1, s1, 320 // (op 13166) 

v_add_u32 v3, v9, s1 // (op 13166) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 13166) Store value




v_mov_b32 v3, v8 // (op 126) reg expression

s_nop 0
v_accvgpr_read v10, a0 // (op 123) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v11, a1 // (op 123) 
v_accvgpr_read v12, a2 // (op 123) 
v_accvgpr_read v13, a3 // (op 123) 
v_mul_f32 v14, v7, v10 // (op 123) 
v_mul_f32 v15, v7, v11 // (op 123) 
v_mul_f32 v16, v7, v12 // (op 123) 
v_mul_f32 v17, v7, v13 // (op 123) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 126) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 127) 
v_mul_f32 v19, v0, v11 // (op 127) 
v_mul_f32 v20, v0, v12 // (op 127) 
v_mul_f32 v21, v0, v13 // (op 127) 

v_add_f32 v10, v18, v14 // (op 124) 
v_add_f32 v11, v19, v15 // (op 124) 
v_add_f32 v12, v20, v16 // (op 124) 
v_add_f32 v13, v21, v17 // (op 124) 




v_mov_b32 v3, v9 // (op 125) reg expression

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 125) Store value




s_lshl_b32 s0, s2, 6 // (op 12367) 
v_add_u32 v3, v8, s0 // (op 12367) 


v_accvgpr_read v10, a4 // (op 12364) 
v_accvgpr_read v11, a5 // (op 12364) 
v_accvgpr_read v12, a6 // (op 12364) 
v_accvgpr_read v13, a7 // (op 12364) 
v_mul_f32 v14, v7, v10 // (op 12364) 
v_mul_f32 v15, v7, v11 // (op 12364) 
v_mul_f32 v16, v7, v12 // (op 12364) 
v_mul_f32 v17, v7, v13 // (op 12364) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12367) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12368) 
v_mul_f32 v19, v0, v11 // (op 12368) 
v_mul_f32 v20, v0, v12 // (op 12368) 
v_mul_f32 v21, v0, v13 // (op 12368) 

v_add_f32 v10, v18, v14 // (op 12365) 
v_add_f32 v11, v19, v15 // (op 12365) 
v_add_f32 v12, v20, v16 // (op 12365) 
v_add_f32 v13, v21, v17 // (op 12365) 




s_lshl_b32 s0, s4, 6 // (op 12366) 
v_add_u32 v3, v9, s0 // (op 12366) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12366) Store value




s_lshl_b32 s0, s2, 7 // (op 12379) 
v_add_u32 v3, v8, s0 // (op 12379) 


v_accvgpr_read v10, a8 // (op 12376) 
v_accvgpr_read v11, a9 // (op 12376) 
v_accvgpr_read v12, a10 // (op 12376) 
v_accvgpr_read v13, a11 // (op 12376) 
v_mul_f32 v14, v7, v10 // (op 12376) 
v_mul_f32 v15, v7, v11 // (op 12376) 
v_mul_f32 v16, v7, v12 // (op 12376) 
v_mul_f32 v17, v7, v13 // (op 12376) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12379) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12380) 
v_mul_f32 v19, v0, v11 // (op 12380) 
v_mul_f32 v20, v0, v12 // (op 12380) 
v_mul_f32 v21, v0, v13 // (op 12380) 

v_add_f32 v10, v18, v14 // (op 12377) 
v_add_f32 v11, v19, v15 // (op 12377) 
v_add_f32 v12, v20, v16 // (op 12377) 
v_add_f32 v13, v21, v17 // (op 12377) 




s_lshl_b32 s0, s4, 7 // (op 12378) 
v_add_u32 v3, v9, s0 // (op 12378) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12378) Store value




s_lshl_b32 s0, s2, 6 // (op 12391) 
s_mul_i32 s1, 3, s0 // (op 12391) 

v_add_u32 v3, v8, s1 // (op 12391) 


v_accvgpr_read v10, a12 // (op 12388) 
v_accvgpr_read v11, a13 // (op 12388) 
v_accvgpr_read v12, a14 // (op 12388) 
v_accvgpr_read v13, a15 // (op 12388) 
v_mul_f32 v14, v7, v10 // (op 12388) 
v_mul_f32 v15, v7, v11 // (op 12388) 
v_mul_f32 v16, v7, v12 // (op 12388) 
v_mul_f32 v17, v7, v13 // (op 12388) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12391) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12392) 
v_mul_f32 v19, v0, v11 // (op 12392) 
v_mul_f32 v20, v0, v12 // (op 12392) 
v_mul_f32 v21, v0, v13 // (op 12392) 

v_add_f32 v10, v18, v14 // (op 12389) 
v_add_f32 v11, v19, v15 // (op 12389) 
v_add_f32 v12, v20, v16 // (op 12389) 
v_add_f32 v13, v21, v17 // (op 12389) 




s_lshl_b32 s0, s4, 6 // (op 12390) 
s_mul_i32 s1, 3, s0 // (op 12390) 

v_add_u32 v3, v9, s1 // (op 12390) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12390) Store value




s_lshl_b32 s0, s2, 8 // (op 12403) 
v_add_u32 v3, v8, s0 // (op 12403) 


v_accvgpr_read v10, a16 // (op 12400) 
v_accvgpr_read v11, a17 // (op 12400) 
v_accvgpr_read v12, a18 // (op 12400) 
v_accvgpr_read v13, a19 // (op 12400) 
v_mul_f32 v14, v7, v10 // (op 12400) 
v_mul_f32 v15, v7, v11 // (op 12400) 
v_mul_f32 v16, v7, v12 // (op 12400) 
v_mul_f32 v17, v7, v13 // (op 12400) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12403) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12404) 
v_mul_f32 v19, v0, v11 // (op 12404) 
v_mul_f32 v20, v0, v12 // (op 12404) 
v_mul_f32 v21, v0, v13 // (op 12404) 

v_add_f32 v10, v18, v14 // (op 12401) 
v_add_f32 v11, v19, v15 // (op 12401) 
v_add_f32 v12, v20, v16 // (op 12401) 
v_add_f32 v13, v21, v17 // (op 12401) 




s_lshl_b32 s0, s4, 8 // (op 12402) 
v_add_u32 v3, v9, s0 // (op 12402) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12402) Store value




s_lshl_b32 s0, s2, 6 // (op 12415) 
s_mul_i32 s1, 5, s0 // (op 12415) 

v_add_u32 v3, v8, s1 // (op 12415) 


v_accvgpr_read v10, a20 // (op 12412) 
v_accvgpr_read v11, a21 // (op 12412) 
v_accvgpr_read v12, a22 // (op 12412) 
v_accvgpr_read v13, a23 // (op 12412) 
v_mul_f32 v14, v7, v10 // (op 12412) 
v_mul_f32 v15, v7, v11 // (op 12412) 
v_mul_f32 v16, v7, v12 // (op 12412) 
v_mul_f32 v17, v7, v13 // (op 12412) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12415) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12416) 
v_mul_f32 v19, v0, v11 // (op 12416) 
v_mul_f32 v20, v0, v12 // (op 12416) 
v_mul_f32 v21, v0, v13 // (op 12416) 

v_add_f32 v10, v18, v14 // (op 12413) 
v_add_f32 v11, v19, v15 // (op 12413) 
v_add_f32 v12, v20, v16 // (op 12413) 
v_add_f32 v13, v21, v17 // (op 12413) 




s_lshl_b32 s0, s4, 6 // (op 12414) 
s_mul_i32 s1, 5, s0 // (op 12414) 

v_add_u32 v3, v9, s1 // (op 12414) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12414) Store value




s_lshl_b32 s0, s2, 6 // (op 12427) 
s_mul_i32 s1, 6, s0 // (op 12427) 

v_add_u32 v3, v8, s1 // (op 12427) 


v_accvgpr_read v10, a24 // (op 12424) 
v_accvgpr_read v11, a25 // (op 12424) 
v_accvgpr_read v12, a26 // (op 12424) 
v_accvgpr_read v13, a27 // (op 12424) 
v_mul_f32 v14, v7, v10 // (op 12424) 
v_mul_f32 v15, v7, v11 // (op 12424) 
v_mul_f32 v16, v7, v12 // (op 12424) 
v_mul_f32 v17, v7, v13 // (op 12424) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12427) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12428) 
v_mul_f32 v19, v0, v11 // (op 12428) 
v_mul_f32 v20, v0, v12 // (op 12428) 
v_mul_f32 v21, v0, v13 // (op 12428) 

v_add_f32 v10, v18, v14 // (op 12425) 
v_add_f32 v11, v19, v15 // (op 12425) 
v_add_f32 v12, v20, v16 // (op 12425) 
v_add_f32 v13, v21, v17 // (op 12425) 




s_lshl_b32 s0, s4, 6 // (op 12426) 
s_mul_i32 s1, 6, s0 // (op 12426) 

v_add_u32 v3, v9, s1 // (op 12426) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12426) Store value




s_lshl_b32 s0, s2, 6 // (op 12439) 
s_mul_i32 s1, 7, s0 // (op 12439) 

v_add_u32 v3, v8, s1 // (op 12439) 


v_accvgpr_read v10, a28 // (op 12436) 
v_accvgpr_read v11, a29 // (op 12436) 
v_accvgpr_read v12, a30 // (op 12436) 
v_accvgpr_read v13, a31 // (op 12436) 
v_mul_f32 v14, v7, v10 // (op 12436) 
v_mul_f32 v15, v7, v11 // (op 12436) 
v_mul_f32 v16, v7, v12 // (op 12436) 
v_mul_f32 v17, v7, v13 // (op 12436) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 12439) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v10 // (op 12440) 
v_mul_f32 v19, v0, v11 // (op 12440) 
v_mul_f32 v20, v0, v12 // (op 12440) 
v_mul_f32 v21, v0, v13 // (op 12440) 

v_add_f32 v10, v18, v14 // (op 12437) 
v_add_f32 v11, v19, v15 // (op 12437) 
v_add_f32 v12, v20, v16 // (op 12437) 
v_add_f32 v13, v21, v17 // (op 12437) 




s_lshl_b32 s0, s4, 6 // (op 12438) 
s_mul_i32 s1, 7, s0 // (op 12438) 

v_add_u32 v3, v9, s1 // (op 12438) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 12438) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__ConditionalBottom_0_beta_114:








































































s_endpgm  // End of RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.LRR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_, .LRR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM__end-RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 444
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 98304 // lds bytes
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
  ProblemType:
    OperationType: GEMM
    DataType: B8
    DataTypeA: B8
    DataTypeB: F4
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
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 98304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 186
    .agpr_count: 256
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
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

