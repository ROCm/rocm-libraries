

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_,@function
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1201) 
s_mul_hi_i32 s1, s16, s10 // (op 1388) 
v_and_b32 v0, 15, v1 // (op 1314) 
v_and_b32 v3, 48, v1 // (op 1314) 
v_lshl_add_u32 v4, v0, 7, v3 // (op 1314) 


v_and_b32 v0, 15, v1 // (op 1473) 
v_and_b32 v3, 48, v1 // (op 1473) 
v_lshl_add_u32 v5, v0, 7, v3 // (op 1473) 


v_lshrrev_b32 v0, 6, v1 // (op 1201) 
s_add_i32 s18, s0, s16 // (op 1201) 

v_and_b32 v3, 63, v1 // (op 1201) 
s_ashr_i32 s0, s18, 31 // (op 1201) 
v_lshrrev_b32 v6, 4, v3 // (op 1201) 
s_cmp_eq_i32 s10, 0 // (op 1201) 
s_mov_b32 s19, scc // (op 1201) 
v_lshlrev_b32 v7, 2, v6 // (op 1201) 

s_cmp_eq_i32 s19, 0 // (op 1201) 
s_cselect_b32 s49, 0, -1 // (op 1201) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1201) 


s_and_b32 s19, s13, 31 // (op 1201) 
v_lshrrev_b32 v0, 2, v3 // (op 1201) 

v_and_b32 v3, 3, v0 // (op 1201) 

v_lshrrev_b32 v0, 6, v1 // (op 1262) 
v_and_b32 v7, 15, v1 // (op 1262) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 1262) 


v_and_b32 v0, 48, v1 // (op 1262) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 1262) 


v_lshrrev_b32 v0, 6, v1 // (op 1372) 
v_and_b32 v8, 15, v1 // (op 1372) 
v_lshl_add_u32 v9, v0, 4, v8 // (op 1372) 


v_and_b32 v0, 63, v1 // (op 1372) 
v_lshrrev_b32 v8, 4, v0 // (op 1372) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 1372) 


v_lshrrev_b32 v8, 6, v1 // (op 1388) 
v_and_b32 v9, 63, v1 // (op 1388) 
v_lshrrev_b32 v10, 6, v1 // (op 1434) 
v_and_b32 v11, 15, v1 // (op 1434) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 1434) 


v_and_b32 v10, 48, v1 // (op 1434) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 1434) 


v_lshrrev_b32 v10, 6, v1 // (op 1502) 
v_and_b32 v12, 15, v1 // (op 1502) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 1502) 


v_and_b32 v10, 63, v1 // (op 1502) 
v_lshrrev_b32 v12, 4, v10 // (op 1502) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 1502) 


s_lshl_b32 s50, 1, s19 // (op 1201) 
v_and_b32 v12, 3, v1 // (op 1201) 
s_add_i32 s51, s50, s49 // (op 1201) 


s_and_b32 s49, s0, s51 // (op 1201) 


s_add_i32 s0, s1, s16 // (op 1388) 

s_ashr_i32 s1, s0, 31 // (op 1388) 
v_lshl_add_u32 v13, v3, 2, v12 // (op 1201) 


s_add_i32 s50, s18, s49 // (op 1201) 


s_ashr_i32 s18, s50, s19 // (op 1201) 


s_lshl_b32 s19, s13, 24 // (op 1201) 
s_ashr_i32 s49, s19, 31 // (op 1201) 

s_xor_b32 s19, s18, s49 // (op 1201) 

s_sub_i32 s18, s19, s49 // (op 1201) 


s_mul_hi_i32 s19, s18, s14 // (op 1201) 
s_add_i32 s49, s19, s18 // (op 1201) 

s_ashr_i32 s19, s49, 31 // (op 1201) 
s_cmp_eq_i32 s14, 0 // (op 1201) 
s_mov_b32 s50, scc // (op 1201) 
s_cmp_eq_i32 s50, 0 // (op 1201) 
s_cselect_b32 s51, 0, -1 // (op 1201) 

s_and_b32 s50, s15, 31 // (op 1201) 
v_lshrrev_b32 v3, 4, v9 // (op 1388) 
s_cmp_eq_i32 s10, 0 // (op 1388) 
s_mov_b32 s52, scc // (op 1388) 
v_lshlrev_b32 v12, 2, v3 // (op 1388) 

s_cmp_eq_i32 s52, 0 // (op 1388) 
s_cselect_b32 s53, 0, -1 // (op 1388) 

s_lshl_b32 s52, 1, s50 // (op 1201) 
v_lshl_add_u32 v3, v8, 4, v12 // (op 1388) 


v_lshrrev_b32 v8, 2, v9 // (op 1388) 

s_add_i32 s54, s52, s51 // (op 1201) 


s_and_b32 s51, s19, s54 // (op 1201) 


s_add_i32 s19, s49, s51 // (op 1201) 


s_ashr_i32 s49, s19, s50 // (op 1201) 


s_lshl_b32 s19, s15, 24 // (op 1201) 
s_ashr_i32 s50, s19, 31 // (op 1201) 

s_xor_b32 s19, s49, s50 // (op 1201) 

s_sub_i32 s49, s19, s50 // (op 1201) 


s_mul_i32 s19, s18, s11 // (op 1201) 
s_sub_u32 s50, s16, s19 // (op 1201) 

s_mul_i32 s19, s49, s11 // (op 1201) 
s_add_u32 s19, s19, s50 // (op 1201) 

s_mul_hi_i32 s50, s16, s9 // (op 1201) 
s_add_i32 s51, s50, s16 // (op 1201) 

s_ashr_i32 s50, s51, 31 // (op 1201) 
s_cmp_eq_i32 s9, 0 // (op 1201) 
s_mov_b32 s52, scc // (op 1201) 
s_cmp_eq_i32 s52, 0 // (op 1201) 
s_cselect_b32 s54, 0, -1 // (op 1201) 

s_and_b32 s52, s13, 31 // (op 1388) 
v_and_b32 v9, 3, v8 // (op 1388) 

s_lshl_b32 s55, 1, s52 // (op 1388) 
v_and_b32 v8, 3, v1 // (op 1388) 
s_add_i32 s56, s55, s53 // (op 1388) 


s_and_b32 s53, s1, s56 // (op 1388) 


v_lshl_add_u32 v12, v9, 2, v8 // (op 1388) 


s_add_i32 s1, s0, s53 // (op 1388) 


s_ashr_i32 s0, s1, s52 // (op 1388) 


s_lshl_b32 s1, s13, 24 // (op 1388) 
s_ashr_i32 s52, s1, 31 // (op 1388) 

s_xor_b32 s1, s0, s52 // (op 1388) 

s_sub_i32 s0, s1, s52 // (op 1388) 


s_mul_hi_i32 s1, s0, s14 // (op 1388) 
s_add_i32 s52, s1, s0 // (op 1388) 

s_ashr_i32 s1, s52, 31 // (op 1388) 
s_cmp_eq_i32 s14, 0 // (op 1388) 
s_mov_b32 s53, scc // (op 1388) 
s_cmp_eq_i32 s53, 0 // (op 1388) 
s_cselect_b32 s55, 0, -1 // (op 1388) 

s_and_b32 s53, s15, 31 // (op 1388) 
s_lshl_b32 s56, 1, s53 // (op 1388) 
s_add_i32 s57, s56, s55 // (op 1388) 


s_and_b32 s55, s1, s57 // (op 1388) 


s_add_i32 s1, s52, s55 // (op 1388) 


s_ashr_i32 s52, s1, s53 // (op 1388) 


s_lshl_b32 s1, s15, 24 // (op 1388) 
s_ashr_i32 s53, s1, 31 // (op 1388) 

s_xor_b32 s1, s52, s53 // (op 1388) 

s_sub_i32 s52, s1, s53 // (op 1388) 


s_mul_i32 s1, s0, s11 // (op 1388) 
s_sub_u32 s53, s16, s1 // (op 1388) 

s_mul_i32 s1, s52, s11 // (op 1388) 
s_add_u32 s1, s1, s53 // (op 1388) 

s_mul_hi_i32 s53, s16, s9 // (op 1388) 
s_add_i32 s55, s53, s16 // (op 1388) 

s_ashr_i32 s53, s55, 31 // (op 1388) 
s_cmp_eq_i32 s9, 0 // (op 1388) 
s_mov_b32 s56, scc // (op 1388) 
s_cmp_eq_i32 s56, 0 // (op 1388) 
s_cselect_b32 s57, 0, -1 // (op 1388) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 1201) 
s_lshl_b32 s58, 1, s56 // (op 1201) 
s_add_i32 s59, s58, s54 // (op 1201) 


s_and_b32 s54, s50, s59 // (op 1201) 


s_add_i32 s50, s51, s54 // (op 1201) 


s_ashr_i32 s51, s50, s56 // (op 1201) 


s_lshl_b32 s50, s45, 24 // (op 1201) 
s_ashr_i32 s54, s50, 31 // (op 1201) 

s_xor_b32 s50, s51, s54 // (op 1201) 

s_sub_i32 s51, s50, s54 // (op 1201) 


s_and_b32 s50, s45, 31 // (op 1388) 
s_lshl_b32 s54, 1, s50 // (op 1388) 
s_add_i32 s56, s54, s57 // (op 1388) 


s_and_b32 s54, s53, s56 // (op 1388) 


s_add_i32 s53, s55, s54 // (op 1388) 


s_ashr_i32 s54, s53, s50 // (op 1388) 


s_lshl_b32 s50, s45, 24 // (op 1388) 
s_ashr_i32 s53, s50, 31 // (op 1388) 

s_xor_b32 s50, s54, s53 // (op 1388) 

s_sub_i32 s54, s50, s53 // (op 1388) 


s_add_u32 s56, s4, 16 // (op 1201) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 1201) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1201) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1201) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 1201) 

s_mul_i32 s50, s11, s56 // (op 1201) 
s_mul_i32 s53, s8, s50 // (op 1201) 

s_mul_i32 s50, s51, s53 // (op 1201) 

s_sub_u32 s53, s16, s50 // (op 1201) 

s_mov_b32 s50, s53 // (op 1201) 
s_mul_hi_i32 s55, s50, s12 // (op 1201) 
s_add_i32 s58, s55, s50 // (op 1201) 


s_ashr_i32 s50, s58, 31 // (op 1201) 
s_cmp_eq_i32 s12, 0 // (op 1201) 
s_mov_b32 s55, scc // (op 1201) 
s_cmp_eq_i32 s55, 0 // (op 1201) 
s_cselect_b32 s59, 0, -1 // (op 1201) 

s_and_b32 s55, s44, 31 // (op 1201) 
s_lshl_b32 s60, 1, s55 // (op 1201) 
s_add_i32 s61, s60, s59 // (op 1201) 


s_and_b32 s59, s50, s61 // (op 1201) 


s_add_i32 s50, s58, s59 // (op 1201) 


s_ashr_i32 s58, s50, s55 // (op 1201) 


s_lshl_b32 s50, s44, 24 // (op 1201) 
s_ashr_i32 s55, s50, 31 // (op 1201) 

s_xor_b32 s50, s58, s55 // (op 1201) 

s_sub_i32 s58, s50, s55 // (op 1201) 


s_mul_i32 s50, s46, s11 // (op 1201) 
s_add_u32 s60, s6, 64 // (op 1201) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 1201) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 1201) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 1201) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 6 // (op 1201) 

s_sub_i32 s55, s60, s50 // (op 1201) 

s_mul_i32 s59, s58, s55 // (op 1201) 

s_sub_u32 s55, s53, s59 // (op 1201) 


s_add_u32 s53, s55, s50 // (op 1201) 


s_cmp_eq_i32 s51, 0 // (op 1201) 
s_mov_b32 s50, scc // (op 1201) 

s_cmp_eq_i32 s50, 0 // (op 1201) 
s_cselect_b32 s51, s53, s19 // (op 1201) 


s_mul_i32 s19, s49, s56 // (op 1201) 


v_lshl_add_u32 v8, s51, 6, v6 // (op 1201) 


s_sub_i32 s49, s18, s19 // (op 1201) 


s_cmp_eq_i32 s50, 0 // (op 1201) 
s_cselect_b32 s51, s58, s49 // (op 1201) 



v_lshl_add_u32 v6, s51, 4, v13 // (op 1201) 


v_mul_lo_u32 v9, v6, s2 // (op 1201) 

v_add_lshl_u32 v6, v8, v9, 2 // (op 1201) 


s_add_u32 s18, s4, 16 // (op 1388) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1388) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 1388) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 1388) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 1388) 

s_mul_i32 s53, s11, s18 // (op 1388) 
s_mul_i32 s49, s8, s53 // (op 1388) 

s_mul_i32 s53, s54, s49 // (op 1388) 

s_sub_u32 s49, s16, s53 // (op 1388) 

s_mov_b32 s53, s49 // (op 1388) 
s_mul_hi_i32 s50, s53, s12 // (op 1388) 
s_add_i32 s51, s50, s53 // (op 1388) 


s_ashr_i32 s50, s51, 31 // (op 1388) 
s_cmp_eq_i32 s12, 0 // (op 1388) 
s_mov_b32 s53, scc // (op 1388) 
s_cmp_eq_i32 s53, 0 // (op 1388) 
s_cselect_b32 s55, 0, -1 // (op 1388) 

s_and_b32 s53, s44, 31 // (op 1388) 
s_lshl_b32 s56, 1, s53 // (op 1388) 
s_add_i32 s57, s56, s55 // (op 1388) 


s_and_b32 s55, s50, s57 // (op 1388) 


s_add_i32 s50, s51, s55 // (op 1388) 


s_ashr_i32 s51, s50, s53 // (op 1388) 


s_lshl_b32 s50, s44, 24 // (op 1388) 
s_ashr_i32 s53, s50, 31 // (op 1388) 

s_xor_b32 s50, s51, s53 // (op 1388) 

s_sub_i32 s51, s50, s53 // (op 1388) 


s_mul_i32 s50, s46, s11 // (op 1388) 
s_add_u32 s56, s6, 64 // (op 1388) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1388) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1388) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1388) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1388) 

s_sub_i32 s53, s56, s50 // (op 1388) 

s_mul_i32 s55, s51, s53 // (op 1388) 

s_sub_u32 s53, s49, s55 // (op 1388) 


s_add_u32 s49, s53, s50 // (op 1388) 


s_cmp_eq_i32 s54, 0 // (op 1388) 
s_mov_b32 s50, scc // (op 1388) 

s_cmp_eq_i32 s50, 0 // (op 1388) 
s_cselect_b32 s53, s49, s1 // (op 1388) 


s_mul_i32 s1, s52, s18 // (op 1388) 


v_lshl_add_u32 v8, s53, 6, v3 // (op 1388) 


s_sub_i32 s49, s0, s1 // (op 1388) 


s_cmp_eq_i32 s50, 0 // (op 1388) 
s_cselect_b32 s0, s51, s49 // (op 1388) 



v_lshl_add_u32 v3, s0, 4, v12 // (op 1388) 


v_mul_lo_u32 v9, v3, s20 // (op 1388) 

v_add_lshl_u32 v3, v8, v9, 2 // (op 1388) 


s_mul_i32 s1, s2, s5 // (op 1203) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 1203) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 1203) least significant
s_mul_hi_u32 s19, s2, s4 // (op 1203) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1203) most significant: sum
s_add_u32 s1, s1, s19 // (op 1203) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1203) 

s_mul_i32 s1, s20, s5 // (op 1390) most significant: low of low * high
s_mul_i32 s49, s21, s4 // (op 1390) most significant: low of high * low
s_mul_i32 s0, s20, s4 // (op 1390) least significant
s_mul_hi_u32 s50, s20, s4 // (op 1390) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 1390) most significant: sum
s_add_u32 s1, s1, s50 // (op 1390) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 1390) 

s_mov_b32 s52, s22 // (op 1203) reg expression
s_mov_b32 s53, s23 // (op 1203) reg expression
s_mov_b32 s54, s18 // (op 1203) reg expression
s_mov_b32 s55, 131072 // (op 1203) call()

s_mov_b32 s56, s24 // (op 1390) reg expression

s_mov_b32 s57, s25 // (op 1390) reg expression
s_mov_b32 s58, s50 // (op 1390) reg expression
s_mov_b32 s59, 131072 // (op 1390) call()


v_mov_b32 v8, s47 // (op 2) Move value
v_mov_b32 v9, s48 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 309) call()
v_accvgpr_write a5, 0.00000 // (op 309) call()
v_accvgpr_write a6, 0.00000 // (op 309) call()
v_accvgpr_write a7, 0.00000 // (op 309) call()
s_mul_hi_i32 s0, s16, s10 // (op 968) 
v_lshrrev_b32 v12, 3, v1 // (op 968) 
s_add_i32 s1, s0, s16 // (op 968) 

v_and_b32 v13, -8, v1 // (op 968) 
s_ashr_i32 s0, s1, 31 // (op 968) 
v_sub_u32 v14, v1, v13 // (op 968) 

s_cmp_eq_i32 s10, 0 // (op 968) 
s_mov_b32 s18, scc // (op 968) 
v_lshlrev_b32 v13, 4, v14 // (op 968) 

s_cmp_eq_i32 s18, 0 // (op 968) 
s_cselect_b32 s19, 0, -1 // (op 968) 

v_mov_b32 v15, 0 // (op 968) convert
v_mov_b32 v14, v13 // (op 968) convert

s_and_b32 s18, s13, 31 // (op 968) 
s_lshl_b32 s22, 1, s18 // (op 968) 
s_add_i32 s23, s22, s19 // (op 968) 


s_and_b32 s19, s0, s23 // (op 968) 


s_add_i32 s0, s1, s19 // (op 968) 


s_ashr_i32 s1, s0, s18 // (op 968) 


s_lshl_b32 s0, s13, 24 // (op 968) 
s_ashr_i32 s18, s0, 31 // (op 968) 

s_xor_b32 s0, s1, s18 // (op 968) 

s_sub_i32 s1, s0, s18 // (op 968) 


s_mul_hi_i32 s0, s1, s14 // (op 968) 
s_add_i32 s18, s0, s1 // (op 968) 

s_ashr_i32 s0, s18, 31 // (op 968) 
s_cmp_eq_i32 s14, 0 // (op 968) 
s_mov_b32 s19, scc // (op 968) 
s_cmp_eq_i32 s19, 0 // (op 968) 
s_cselect_b32 s22, 0, -1 // (op 968) 

s_and_b32 s19, s15, 31 // (op 968) 
s_lshl_b32 s23, 1, s19 // (op 968) 
s_add_i32 s24, s23, s22 // (op 968) 


s_and_b32 s25, s0, s24 // (op 968) 


s_add_i32 s0, s18, s25 // (op 968) 


s_ashr_i32 s18, s0, s19 // (op 968) 


s_lshl_b32 s0, s15, 24 // (op 968) 
s_ashr_i32 s19, s0, 31 // (op 968) 

s_xor_b32 s0, s18, s19 // (op 968) 

s_sub_i32 s18, s0, s19 // (op 968) 


s_mul_i32 s0, s1, s11 // (op 968) 

s_sub_u32 s1, s16, s0 // (op 968) 

s_mul_i32 s0, s18, s11 // (op 968) 
s_add_u32 s0, s0, s1 // (op 968) 


s_mul_hi_i32 s1, s16, s9 // (op 968) 
s_add_i32 s18, s1, s16 // (op 968) 

s_ashr_i32 s1, s18, 31 // (op 968) 
s_cmp_eq_i32 s9, 0 // (op 968) 
s_mov_b32 s19, scc // (op 968) 
s_cmp_eq_i32 s19, 0 // (op 968) 
s_cselect_b32 s22, 0, -1 // (op 968) 

s_and_b32 s19, s45, 31 // (op 968) 
s_lshl_b32 s23, 1, s19 // (op 968) 
s_add_i32 s24, s23, s22 // (op 968) 


s_and_b32 s25, s1, s24 // (op 968) 


s_add_i32 s1, s18, s25 // (op 968) 


s_ashr_i32 s18, s1, s19 // (op 968) 


s_lshl_b32 s1, s45, 24 // (op 968) 
s_ashr_i32 s19, s1, 31 // (op 968) 

s_xor_b32 s1, s18, s19 // (op 968) 

s_sub_i32 s18, s1, s19 // (op 968) 


s_add_u32 s22, s4, 16 // (op 968) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 968) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 968) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 968) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 968) 

s_mul_i32 s1, s11, s22 // (op 968) 

s_mul_i32 s19, s8, s1 // (op 968) 

s_mul_i32 s1, s18, s19 // (op 968) 

s_sub_u32 s19, s16, s1 // (op 968) 

s_mov_b32 s1, s19 // (op 968) 
s_mul_hi_i32 s22, s1, s12 // (op 968) 
s_add_i32 s23, s22, s1 // (op 968) 


s_ashr_i32 s1, s23, 31 // (op 968) 
s_cmp_eq_i32 s12, 0 // (op 968) 
s_mov_b32 s22, scc // (op 968) 
s_cmp_eq_i32 s22, 0 // (op 968) 
s_cselect_b32 s24, 0, -1 // (op 968) 

s_and_b32 s22, s44, 31 // (op 968) 
s_lshl_b32 s25, 1, s22 // (op 968) 
s_add_i32 s47, s25, s24 // (op 968) 


s_and_b32 s24, s1, s47 // (op 968) 


s_add_i32 s1, s23, s24 // (op 968) 


s_ashr_i32 s23, s1, s22 // (op 968) 


s_lshl_b32 s1, s44, 24 // (op 968) 
s_ashr_i32 s22, s1, 31 // (op 968) 

s_xor_b32 s1, s23, s22 // (op 968) 

s_sub_i32 s23, s1, s22 // (op 968) 


s_mul_i32 s1, s46, s11 // (op 968) 
s_add_u32 s24, s6, 64 // (op 968) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 968) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 968) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 968) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 968) 

s_sub_i32 s22, s24, s1 // (op 968) 

s_mul_i32 s24, s23, s22 // (op 968) 


s_sub_u32 s25, s19, s24 // (op 968) 


s_add_u32 s19, s25, s1 // (op 968) 


s_cmp_eq_i32 s18, 0 // (op 968) 
s_mov_b32 s1, scc // (op 968) 

s_cmp_eq_i32 s1, 0 // (op 968) 
s_cselect_b32 s18, s19, s0 // (op 968) 



v_lshl_add_u32 v13, s18, 6, v12 // (op 968) 


v_mul_lo_u32 v17, v13, s27 // (op 968) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 968) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 968) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 968) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 968) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 968) most significant half



s_mul_i32 s1, s26, s7 // (op 970) most significant: low of low * high
s_mul_i32 s18, s27, s6 // (op 970) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 970) least significant
s_mul_hi_u32 s19, s26, s6 // (op 970) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 970) most significant: sum
s_add_u32 s1, s1, s19 // (op 970) most significant: sum


s_mov_b32 s48, s28 // (op 970) reg expression
s_mov_b32 s49, s29 // (op 970) reg expression
s_mov_b32 s50, s0 // (op 970) reg expression
s_mov_b32 s51, 131072 // (op 970) call()

s_mul_hi_i32 s47, s16, s10 // (op 987) 
v_lshrrev_b32 v12, 3, v1 // (op 987) 
s_add_i32 s0, s47, s16 // (op 987) 

v_and_b32 v13, -8, v1 // (op 987) 
s_ashr_i32 s1, s0, 31 // (op 987) 
v_sub_u32 v14, v1, v13 // (op 987) 

s_cmp_eq_i32 s10, 0 // (op 987) 
s_mov_b32 s47, scc // (op 987) 
v_lshlrev_b32 v13, 4, v14 // (op 987) 

s_cmp_eq_i32 s47, 0 // (op 987) 
s_cselect_b32 s18, 0, -1 // (op 987) 

v_mov_b32 v15, 0 // (op 987) convert
v_mov_b32 v14, v13 // (op 987) convert

s_and_b32 s19, s13, 31 // (op 987) 
s_lshl_b32 s47, 1, s19 // (op 987) 
s_add_i32 s22, s47, s18 // (op 987) 


s_and_b32 s18, s1, s22 // (op 987) 


s_add_i32 s1, s0, s18 // (op 987) 


s_ashr_i32 s0, s1, s19 // (op 987) 


s_lshl_b32 s1, s13, 24 // (op 987) 
s_ashr_i32 s47, s1, 31 // (op 987) 

s_xor_b32 s1, s0, s47 // (op 987) 

s_sub_i32 s0, s1, s47 // (op 987) 


s_mul_hi_i32 s1, s0, s14 // (op 987) 
s_add_i32 s47, s1, s0 // (op 987) 

s_ashr_i32 s1, s47, 31 // (op 987) 
s_cmp_eq_i32 s14, 0 // (op 987) 
s_mov_b32 s18, scc // (op 987) 
s_cmp_eq_i32 s18, 0 // (op 987) 
s_cselect_b32 s19, 0, -1 // (op 987) 

s_and_b32 s18, s15, 31 // (op 987) 
s_lshl_b32 s22, 1, s18 // (op 987) 
s_add_i32 s23, s22, s19 // (op 987) 


s_and_b32 s19, s1, s23 // (op 987) 


s_add_i32 s1, s47, s19 // (op 987) 


s_ashr_i32 s19, s1, s18 // (op 987) 


s_lshl_b32 s1, s15, 24 // (op 987) 
s_ashr_i32 s18, s1, 31 // (op 987) 

s_xor_b32 s1, s19, s18 // (op 987) 

s_sub_i32 s19, s1, s18 // (op 987) 


s_mul_i32 s1, s0, s11 // (op 987) 

s_sub_u32 s0, s16, s1 // (op 987) 

s_mul_i32 s1, s19, s11 // (op 987) 
s_add_u32 s1, s1, s0 // (op 987) 


s_mul_hi_i32 s0, s16, s9 // (op 987) 
s_add_i32 s47, s0, s16 // (op 987) 

s_ashr_i32 s0, s47, 31 // (op 987) 
s_cmp_eq_i32 s9, 0 // (op 987) 
s_mov_b32 s18, scc // (op 987) 
s_cmp_eq_i32 s18, 0 // (op 987) 
s_cselect_b32 s19, 0, -1 // (op 987) 

s_and_b32 s18, s45, 31 // (op 987) 
s_lshl_b32 s22, 1, s18 // (op 987) 
s_add_i32 s23, s22, s19 // (op 987) 


s_and_b32 s19, s0, s23 // (op 987) 


s_add_i32 s0, s47, s19 // (op 987) 


s_ashr_i32 s19, s0, s18 // (op 987) 


s_lshl_b32 s0, s45, 24 // (op 987) 
s_ashr_i32 s18, s0, 31 // (op 987) 

s_xor_b32 s0, s19, s18 // (op 987) 

s_sub_i32 s19, s0, s18 // (op 987) 


s_add_u32 s22, s4, 16 // (op 987) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 987) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 987) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 987) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 987) 

s_mul_i32 s0, s11, s22 // (op 987) 

s_mul_i32 s18, s8, s0 // (op 987) 

s_mul_i32 s0, s19, s18 // (op 987) 

s_sub_u32 s18, s16, s0 // (op 987) 

s_mov_b32 s0, s18 // (op 987) 
s_mul_hi_i32 s47, s0, s12 // (op 987) 
s_add_i32 s22, s47, s0 // (op 987) 


s_ashr_i32 s0, s22, 31 // (op 987) 
s_cmp_eq_i32 s12, 0 // (op 987) 
s_mov_b32 s47, scc // (op 987) 
s_cmp_eq_i32 s47, 0 // (op 987) 
s_cselect_b32 s23, 0, -1 // (op 987) 

s_and_b32 s47, s44, 31 // (op 987) 
s_lshl_b32 s24, 1, s47 // (op 987) 
s_add_i32 s25, s24, s23 // (op 987) 


s_and_b32 s23, s0, s25 // (op 987) 


s_add_i32 s0, s22, s23 // (op 987) 


s_ashr_i32 s22, s0, s47 // (op 987) 


s_lshl_b32 s0, s44, 24 // (op 987) 
s_ashr_i32 s47, s0, 31 // (op 987) 

s_xor_b32 s0, s22, s47 // (op 987) 

s_sub_i32 s22, s0, s47 // (op 987) 


s_mul_i32 s0, s46, s11 // (op 987) 
s_add_u32 s24, s6, 64 // (op 987) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 987) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 987) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 987) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 6 // (op 987) 

s_sub_i32 s23, s24, s0 // (op 987) 

s_mul_i32 s47, s22, s23 // (op 987) 


s_sub_u32 s22, s18, s47 // (op 987) 


s_add_u32 s18, s22, s0 // (op 987) 


s_cmp_eq_i32 s19, 0 // (op 987) 
s_mov_b32 s0, scc // (op 987) 

s_cmp_eq_i32 s0, 0 // (op 987) 
s_cselect_b32 s19, s18, s1 // (op 987) 



v_lshl_add_u32 v13, s19, 6, v12 // (op 987) 


v_mul_lo_u32 v19, v13, s27 // (op 987) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 987) most significant: high of low * low
v_mul_lo_u32 v18, v13, s26 // (op 987) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 987) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 987) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 987) most significant half



s_mul_i32 s1, s26, s7 // (op 989) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 989) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 989) least significant
s_mul_hi_u32 s18, s26, s6 // (op 989) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 989) most significant: sum
s_add_u32 s1, s1, s18 // (op 989) most significant: sum


s_mov_b32 s48, s28 // (op 989) reg expression
s_mov_b32 s49, s29 // (op 989) reg expression
s_mov_b32 s50, s0 // (op 989) reg expression
s_mov_b32 s51, 131072 // (op 989) call()

s_mul_hi_i32 s47, s16, s10 // (op 1032) 
v_lshrrev_b32 v12, 3, v1 // (op 1032) 
s_add_i32 s0, s47, s16 // (op 1032) 

v_and_b32 v13, -8, v1 // (op 1032) 
s_ashr_i32 s1, s0, 31 // (op 1032) 
v_sub_u32 v14, v1, v13 // (op 1032) 

s_cmp_eq_i32 s10, 0 // (op 1032) 
s_mov_b32 s47, scc // (op 1032) 
v_lshlrev_b32 v13, 4, v14 // (op 1032) 

s_cmp_eq_i32 s47, 0 // (op 1032) 
s_cselect_b32 s18, 0, -1 // (op 1032) 

v_mov_b32 v15, 0 // (op 1032) convert
v_mov_b32 v14, v13 // (op 1032) convert

s_and_b32 s19, s13, 31 // (op 1032) 
s_lshl_b32 s47, 1, s19 // (op 1032) 
s_add_i32 s22, s47, s18 // (op 1032) 


s_and_b32 s18, s1, s22 // (op 1032) 


s_add_i32 s1, s0, s18 // (op 1032) 


s_ashr_i32 s0, s1, s19 // (op 1032) 


s_lshl_b32 s1, s13, 24 // (op 1032) 
s_ashr_i32 s47, s1, 31 // (op 1032) 

s_xor_b32 s1, s0, s47 // (op 1032) 

s_sub_i32 s0, s1, s47 // (op 1032) 


s_mul_hi_i32 s1, s0, s14 // (op 1032) 
s_add_i32 s47, s1, s0 // (op 1032) 

s_ashr_i32 s1, s47, 31 // (op 1032) 
s_cmp_eq_i32 s14, 0 // (op 1032) 
s_mov_b32 s18, scc // (op 1032) 
s_cmp_eq_i32 s18, 0 // (op 1032) 
s_cselect_b32 s19, 0, -1 // (op 1032) 

s_and_b32 s18, s15, 31 // (op 1032) 
s_lshl_b32 s22, 1, s18 // (op 1032) 
s_add_i32 s23, s22, s19 // (op 1032) 


s_and_b32 s19, s1, s23 // (op 1032) 


s_add_i32 s1, s47, s19 // (op 1032) 


s_ashr_i32 s19, s1, s18 // (op 1032) 


s_lshl_b32 s1, s15, 24 // (op 1032) 
s_ashr_i32 s18, s1, 31 // (op 1032) 

s_xor_b32 s1, s19, s18 // (op 1032) 

s_sub_i32 s19, s1, s18 // (op 1032) 


s_add_u32 s22, s4, 16 // (op 1032) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1032) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1032) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1032) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1032) 

s_mul_i32 s1, s19, s22 // (op 1032) 

s_sub_i32 s47, s0, s1 // (op 1032) 


s_mul_hi_i32 s0, s16, s9 // (op 1032) 
s_add_i32 s1, s0, s16 // (op 1032) 

s_ashr_i32 s0, s1, 31 // (op 1032) 
s_cmp_eq_i32 s9, 0 // (op 1032) 
s_mov_b32 s18, scc // (op 1032) 
s_cmp_eq_i32 s18, 0 // (op 1032) 
s_cselect_b32 s19, 0, -1 // (op 1032) 

s_and_b32 s18, s45, 31 // (op 1032) 
s_lshl_b32 s24, 1, s18 // (op 1032) 
s_add_i32 s25, s24, s19 // (op 1032) 


s_and_b32 s19, s0, s25 // (op 1032) 


s_add_i32 s0, s1, s19 // (op 1032) 


s_ashr_i32 s1, s0, s18 // (op 1032) 


s_lshl_b32 s0, s45, 24 // (op 1032) 
s_ashr_i32 s18, s0, 31 // (op 1032) 

s_xor_b32 s0, s1, s18 // (op 1032) 

s_sub_i32 s1, s0, s18 // (op 1032) 


s_mul_i32 s0, s11, s22 // (op 1032) 

s_mul_i32 s18, s8, s0 // (op 1032) 

s_mul_i32 s0, s1, s18 // (op 1032) 

s_sub_u32 s18, s16, s0 // (op 1032) 

s_mul_hi_i32 s0, s18, s12 // (op 1032) 
s_add_i32 s19, s0, s18 // (op 1032) 


s_ashr_i32 s0, s19, 31 // (op 1032) 
s_cmp_eq_i32 s12, 0 // (op 1032) 
s_mov_b32 s18, scc // (op 1032) 
s_cmp_eq_i32 s18, 0 // (op 1032) 
s_cselect_b32 s22, 0, -1 // (op 1032) 

s_and_b32 s18, s44, 31 // (op 1032) 
s_lshl_b32 s23, 1, s18 // (op 1032) 
s_add_i32 s24, s23, s22 // (op 1032) 


s_and_b32 s25, s0, s24 // (op 1032) 


s_add_i32 s0, s19, s25 // (op 1032) 


s_ashr_i32 s19, s0, s18 // (op 1032) 


s_lshl_b32 s0, s44, 24 // (op 1032) 
s_ashr_i32 s18, s0, 31 // (op 1032) 

s_xor_b32 s0, s19, s18 // (op 1032) 

s_sub_i32 s19, s0, s18 // (op 1032) 


s_cmp_eq_i32 s1, 0 // (op 1032) 
s_mov_b32 s0, scc // (op 1032) 

s_cmp_eq_i32 s0, 0 // (op 1032) 
s_cselect_b32 s1, s19, s47 // (op 1032) 



v_lshl_add_u32 v13, s1, 4, v12 // (op 1032) 


v_mul_lo_u32 v21, v13, s31 // (op 1032) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 1032) most significant: high of low * low
v_mul_lo_u32 v20, v13, s30 // (op 1032) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1032) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 1032) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 1032) most significant half



s_mul_i32 s1, s30, s5 // (op 1034) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 1034) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 1034) least significant
s_mul_hi_u32 s18, s30, s4 // (op 1034) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1034) most significant: sum
s_add_u32 s1, s1, s18 // (op 1034) most significant: sum


s_mov_b32 s60, s32 // (op 1034) reg expression
s_mov_b32 s61, s33 // (op 1034) reg expression
s_mov_b32 s62, s0 // (op 1034) reg expression
s_mov_b32 s63, 131072 // (op 1034) call()

s_mul_hi_i32 s47, s16, s10 // (op 1051) 
v_lshrrev_b32 v14, 3, v1 // (op 1051) 
s_add_i32 s0, s47, s16 // (op 1051) 

v_and_b32 v15, -8, v1 // (op 1051) 
s_ashr_i32 s1, s0, 31 // (op 1051) 
v_sub_u32 v20, v1, v15 // (op 1051) 

s_cmp_eq_i32 s10, 0 // (op 1051) 
s_mov_b32 s47, scc // (op 1051) 
v_lshlrev_b32 v15, 4, v20 // (op 1051) 

s_cmp_eq_i32 s47, 0 // (op 1051) 
s_cselect_b32 s18, 0, -1 // (op 1051) 

v_mov_b32 v21, 0 // (op 1051) convert
v_mov_b32 v20, v15 // (op 1051) convert

s_and_b32 s19, s13, 31 // (op 1051) 
s_lshl_b32 s47, 1, s19 // (op 1051) 
s_add_i32 s22, s47, s18 // (op 1051) 


s_and_b32 s18, s1, s22 // (op 1051) 


s_add_i32 s1, s0, s18 // (op 1051) 


s_ashr_i32 s0, s1, s19 // (op 1051) 


s_lshl_b32 s1, s13, 24 // (op 1051) 
s_ashr_i32 s47, s1, 31 // (op 1051) 

s_xor_b32 s1, s0, s47 // (op 1051) 

s_sub_i32 s0, s1, s47 // (op 1051) 


s_mul_hi_i32 s1, s0, s14 // (op 1051) 
s_add_i32 s47, s1, s0 // (op 1051) 

s_ashr_i32 s1, s47, 31 // (op 1051) 
s_cmp_eq_i32 s14, 0 // (op 1051) 
s_mov_b32 s18, scc // (op 1051) 
s_cmp_eq_i32 s18, 0 // (op 1051) 
s_cselect_b32 s19, 0, -1 // (op 1051) 

s_and_b32 s18, s15, 31 // (op 1051) 
s_lshl_b32 s22, 1, s18 // (op 1051) 
s_add_i32 s23, s22, s19 // (op 1051) 


s_and_b32 s19, s1, s23 // (op 1051) 


s_add_i32 s1, s47, s19 // (op 1051) 


s_ashr_i32 s19, s1, s18 // (op 1051) 


s_lshl_b32 s1, s15, 24 // (op 1051) 
s_ashr_i32 s18, s1, 31 // (op 1051) 

s_xor_b32 s1, s19, s18 // (op 1051) 

s_sub_i32 s19, s1, s18 // (op 1051) 


s_add_u32 s22, s4, 16 // (op 1051) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1051) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1051) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1051) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1051) 

s_mul_i32 s1, s19, s22 // (op 1051) 

s_sub_i32 s47, s0, s1 // (op 1051) 


s_mul_hi_i32 s0, s16, s9 // (op 1051) 
s_add_i32 s1, s0, s16 // (op 1051) 

s_ashr_i32 s0, s1, 31 // (op 1051) 
s_cmp_eq_i32 s9, 0 // (op 1051) 
s_mov_b32 s18, scc // (op 1051) 
s_cmp_eq_i32 s18, 0 // (op 1051) 
s_cselect_b32 s19, 0, -1 // (op 1051) 

s_and_b32 s18, s45, 31 // (op 1051) 
s_lshl_b32 s24, 1, s18 // (op 1051) 
s_add_i32 s25, s24, s19 // (op 1051) 


s_and_b32 s19, s0, s25 // (op 1051) 


s_add_i32 s0, s1, s19 // (op 1051) 


s_ashr_i32 s1, s0, s18 // (op 1051) 


s_lshl_b32 s0, s45, 24 // (op 1051) 
s_ashr_i32 s18, s0, 31 // (op 1051) 

s_xor_b32 s0, s1, s18 // (op 1051) 

s_sub_i32 s1, s0, s18 // (op 1051) 


s_mul_i32 s0, s11, s22 // (op 1051) 

s_mul_i32 s18, s8, s0 // (op 1051) 

s_mul_i32 s0, s1, s18 // (op 1051) 

s_sub_u32 s18, s16, s0 // (op 1051) 

s_mul_hi_i32 s0, s18, s12 // (op 1051) 
s_add_i32 s19, s0, s18 // (op 1051) 


s_ashr_i32 s0, s19, 31 // (op 1051) 
s_cmp_eq_i32 s12, 0 // (op 1051) 
s_mov_b32 s18, scc // (op 1051) 
s_cmp_eq_i32 s18, 0 // (op 1051) 
s_cselect_b32 s22, 0, -1 // (op 1051) 

s_and_b32 s18, s44, 31 // (op 1051) 
s_lshl_b32 s23, 1, s18 // (op 1051) 
s_add_i32 s24, s23, s22 // (op 1051) 


s_and_b32 s25, s0, s24 // (op 1051) 


s_add_i32 s0, s19, s25 // (op 1051) 


s_ashr_i32 s19, s0, s18 // (op 1051) 


s_lshl_b32 s0, s44, 24 // (op 1051) 
s_ashr_i32 s18, s0, 31 // (op 1051) 

s_xor_b32 s0, s19, s18 // (op 1051) 

s_sub_i32 s19, s0, s18 // (op 1051) 


s_cmp_eq_i32 s1, 0 // (op 1051) 
s_mov_b32 s0, scc // (op 1051) 

s_cmp_eq_i32 s0, 0 // (op 1051) 
s_cselect_b32 s1, s19, s47 // (op 1051) 



v_lshl_add_u32 v15, s1, 4, v14 // (op 1051) 


v_mul_lo_u32 v23, v15, s31 // (op 1051) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 1051) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 1051) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 1051) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 1051) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 1051) most significant half



s_mul_i32 s1, s30, s5 // (op 1053) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 1053) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 1053) least significant
s_mul_hi_u32 s18, s30, s4 // (op 1053) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1053) most significant: sum
s_add_u32 s1, s1, s18 // (op 1053) most significant: sum


s_mov_b32 s60, s32 // (op 1053) reg expression
s_mov_b32 s61, s33 // (op 1053) reg expression
s_mov_b32 s62, s0 // (op 1053) reg expression
s_mov_b32 s63, 131072 // (op 1053) call()

s_mul_hi_i32 s47, s16, s10 // (op 1096) 
v_lshrrev_b32 v20, 2, v1 // (op 1096) 
s_add_i32 s0, s47, s16 // (op 1096) 

v_lshlrev_b32 v21, 1, v20 // (op 1096) 

s_ashr_i32 s1, s0, 31 // (op 1096) 
v_and_b32 v20, -4, v1 // (op 1096) 
s_cmp_eq_i32 s10, 0 // (op 1096) 
s_mov_b32 s47, scc // (op 1096) 
v_sub_u32 v22, v1, v20 // (op 1096) 

s_cmp_eq_i32 s47, 0 // (op 1096) 
s_cselect_b32 s18, 0, -1 // (op 1096) 

s_and_b32 s19, s13, 31 // (op 1096) 
s_lshl_b32 s47, 1, s19 // (op 1096) 
s_add_i32 s22, s47, s18 // (op 1096) 


s_and_b32 s18, s1, s22 // (op 1096) 


s_add_i32 s1, s0, s18 // (op 1096) 


s_ashr_i32 s0, s1, s19 // (op 1096) 


s_lshl_b32 s1, s13, 24 // (op 1096) 
s_ashr_i32 s47, s1, 31 // (op 1096) 

s_xor_b32 s1, s0, s47 // (op 1096) 

s_sub_i32 s0, s1, s47 // (op 1096) 


s_mul_hi_i32 s1, s0, s14 // (op 1096) 
s_add_i32 s47, s1, s0 // (op 1096) 

s_ashr_i32 s1, s47, 31 // (op 1096) 
s_cmp_eq_i32 s14, 0 // (op 1096) 
s_mov_b32 s18, scc // (op 1096) 
s_cmp_eq_i32 s18, 0 // (op 1096) 
s_cselect_b32 s19, 0, -1 // (op 1096) 

s_and_b32 s18, s15, 31 // (op 1096) 
s_lshl_b32 s22, 1, s18 // (op 1096) 
s_add_i32 s23, s22, s19 // (op 1096) 


s_and_b32 s19, s1, s23 // (op 1096) 


s_add_i32 s1, s47, s19 // (op 1096) 


s_ashr_i32 s19, s1, s18 // (op 1096) 


s_lshl_b32 s1, s15, 24 // (op 1096) 
s_ashr_i32 s18, s1, 31 // (op 1096) 

s_xor_b32 s1, s19, s18 // (op 1096) 

s_sub_i32 s19, s1, s18 // (op 1096) 


s_mul_i32 s1, s0, s11 // (op 1096) 

s_sub_u32 s0, s16, s1 // (op 1096) 

s_mul_i32 s1, s19, s11 // (op 1096) 
s_add_u32 s1, s1, s0 // (op 1096) 


s_mul_hi_i32 s0, s16, s9 // (op 1096) 
s_add_i32 s47, s0, s16 // (op 1096) 

s_ashr_i32 s0, s47, 31 // (op 1096) 
s_cmp_eq_i32 s9, 0 // (op 1096) 
s_mov_b32 s18, scc // (op 1096) 
s_cmp_eq_i32 s18, 0 // (op 1096) 
s_cselect_b32 s19, 0, -1 // (op 1096) 

s_and_b32 s18, s45, 31 // (op 1096) 
s_lshl_b32 s22, 1, s18 // (op 1096) 
s_add_i32 s23, s22, s19 // (op 1096) 


s_and_b32 s19, s0, s23 // (op 1096) 


s_add_i32 s0, s47, s19 // (op 1096) 


s_ashr_i32 s19, s0, s18 // (op 1096) 


s_lshl_b32 s0, s45, 24 // (op 1096) 
s_ashr_i32 s18, s0, 31 // (op 1096) 

s_xor_b32 s0, s19, s18 // (op 1096) 

s_sub_i32 s19, s0, s18 // (op 1096) 


s_add_u32 s22, s4, 16 // (op 1096) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1096) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1096) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1096) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1096) 

s_mul_i32 s0, s11, s22 // (op 1096) 

s_mul_i32 s18, s8, s0 // (op 1096) 

s_mul_i32 s0, s19, s18 // (op 1096) 

s_sub_u32 s18, s16, s0 // (op 1096) 

s_mov_b32 s0, s18 // (op 1096) 
s_mul_hi_i32 s47, s0, s12 // (op 1096) 
s_add_i32 s22, s47, s0 // (op 1096) 


s_ashr_i32 s0, s22, 31 // (op 1096) 
s_cmp_eq_i32 s12, 0 // (op 1096) 
s_mov_b32 s47, scc // (op 1096) 
s_cmp_eq_i32 s47, 0 // (op 1096) 
s_cselect_b32 s23, 0, -1 // (op 1096) 

s_and_b32 s47, s44, 31 // (op 1096) 
s_lshl_b32 s24, 1, s47 // (op 1096) 
s_add_i32 s25, s24, s23 // (op 1096) 


s_and_b32 s23, s0, s25 // (op 1096) 


s_add_i32 s0, s22, s23 // (op 1096) 


s_ashr_i32 s22, s0, s47 // (op 1096) 


s_lshl_b32 s0, s44, 24 // (op 1096) 
s_ashr_i32 s47, s0, 31 // (op 1096) 

s_xor_b32 s0, s22, s47 // (op 1096) 

s_sub_i32 s22, s0, s47 // (op 1096) 


s_mul_i32 s0, s46, s11 // (op 1096) 
s_add_u32 s24, s6, 64 // (op 1096) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1096) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 1096) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 1096) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 6 // (op 1096) 

s_sub_i32 s23, s24, s0 // (op 1096) 

s_mul_i32 s47, s22, s23 // (op 1096) 


s_sub_u32 s22, s18, s47 // (op 1096) 


s_add_u32 s18, s22, s0 // (op 1096) 


s_cmp_eq_i32 s19, 0 // (op 1096) 
s_mov_b32 s0, scc // (op 1096) 

s_cmp_eq_i32 s0, 0 // (op 1096) 
s_cselect_b32 s19, s18, s1 // (op 1096) 



v_lshl_add_u32 v20, s19, 6, v21 // (op 1096) 


v_mul_lo_u32 v21, v20, s34 // (op 1096) 
v_add_u32 v21, v21, v22 // (op 1096) 


s_mul_i32 s1, s34, s7 // (op 1098) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1098) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1098) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1098) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1098) most significant: sum
s_add_u32 s1, s1, s18 // (op 1098) most significant: sum


s_mov_b32 s64, s36 // (op 1098) reg expression
s_mov_b32 s65, s37 // (op 1098) reg expression
s_mov_b32 s66, s0 // (op 1098) reg expression
s_mov_b32 s67, 131072 // (op 1098) call()

s_mul_hi_i32 s47, s16, s10 // (op 1115) 
v_lshrrev_b32 v20, 2, v1 // (op 1115) 
s_add_i32 s0, s47, s16 // (op 1115) 

v_lshlrev_b32 v22, 1, v20 // (op 1115) 

s_ashr_i32 s1, s0, 31 // (op 1115) 
v_and_b32 v20, -4, v1 // (op 1115) 
s_cmp_eq_i32 s10, 0 // (op 1115) 
s_mov_b32 s47, scc // (op 1115) 
v_sub_u32 v23, v1, v20 // (op 1115) 

s_cmp_eq_i32 s47, 0 // (op 1115) 
s_cselect_b32 s18, 0, -1 // (op 1115) 

s_and_b32 s19, s13, 31 // (op 1115) 
s_lshl_b32 s47, 1, s19 // (op 1115) 
s_add_i32 s22, s47, s18 // (op 1115) 


s_and_b32 s18, s1, s22 // (op 1115) 


s_add_i32 s1, s0, s18 // (op 1115) 


s_ashr_i32 s0, s1, s19 // (op 1115) 


s_lshl_b32 s1, s13, 24 // (op 1115) 
s_ashr_i32 s47, s1, 31 // (op 1115) 

s_xor_b32 s1, s0, s47 // (op 1115) 

s_sub_i32 s0, s1, s47 // (op 1115) 


s_mul_hi_i32 s1, s0, s14 // (op 1115) 
s_add_i32 s47, s1, s0 // (op 1115) 

s_ashr_i32 s1, s47, 31 // (op 1115) 
s_cmp_eq_i32 s14, 0 // (op 1115) 
s_mov_b32 s18, scc // (op 1115) 
s_cmp_eq_i32 s18, 0 // (op 1115) 
s_cselect_b32 s19, 0, -1 // (op 1115) 

s_and_b32 s18, s15, 31 // (op 1115) 
s_lshl_b32 s22, 1, s18 // (op 1115) 
s_add_i32 s23, s22, s19 // (op 1115) 


s_and_b32 s19, s1, s23 // (op 1115) 


s_add_i32 s1, s47, s19 // (op 1115) 


s_ashr_i32 s19, s1, s18 // (op 1115) 


s_lshl_b32 s1, s15, 24 // (op 1115) 
s_ashr_i32 s18, s1, 31 // (op 1115) 

s_xor_b32 s1, s19, s18 // (op 1115) 

s_sub_i32 s19, s1, s18 // (op 1115) 


s_mul_i32 s1, s0, s11 // (op 1115) 

s_sub_u32 s0, s16, s1 // (op 1115) 

s_mul_i32 s1, s19, s11 // (op 1115) 
s_add_u32 s1, s1, s0 // (op 1115) 


s_mul_hi_i32 s0, s16, s9 // (op 1115) 
s_add_i32 s47, s0, s16 // (op 1115) 

s_ashr_i32 s0, s47, 31 // (op 1115) 
s_cmp_eq_i32 s9, 0 // (op 1115) 
s_mov_b32 s18, scc // (op 1115) 
s_cmp_eq_i32 s18, 0 // (op 1115) 
s_cselect_b32 s19, 0, -1 // (op 1115) 

s_and_b32 s18, s45, 31 // (op 1115) 
s_lshl_b32 s22, 1, s18 // (op 1115) 
s_add_i32 s23, s22, s19 // (op 1115) 


s_and_b32 s19, s0, s23 // (op 1115) 


s_add_i32 s0, s47, s19 // (op 1115) 


s_ashr_i32 s19, s0, s18 // (op 1115) 


s_lshl_b32 s0, s45, 24 // (op 1115) 
s_ashr_i32 s18, s0, 31 // (op 1115) 

s_xor_b32 s0, s19, s18 // (op 1115) 

s_sub_i32 s19, s0, s18 // (op 1115) 


s_add_u32 s22, s4, 16 // (op 1115) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1115) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1115) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1115) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1115) 

s_mul_i32 s0, s11, s22 // (op 1115) 

s_mul_i32 s18, s8, s0 // (op 1115) 

s_mul_i32 s0, s19, s18 // (op 1115) 

s_sub_u32 s18, s16, s0 // (op 1115) 

s_mov_b32 s0, s18 // (op 1115) 
s_mul_hi_i32 s47, s0, s12 // (op 1115) 
s_add_i32 s22, s47, s0 // (op 1115) 


s_ashr_i32 s0, s22, 31 // (op 1115) 
s_cmp_eq_i32 s12, 0 // (op 1115) 
s_mov_b32 s47, scc // (op 1115) 
s_cmp_eq_i32 s47, 0 // (op 1115) 
s_cselect_b32 s23, 0, -1 // (op 1115) 

s_and_b32 s47, s44, 31 // (op 1115) 
s_lshl_b32 s24, 1, s47 // (op 1115) 
s_add_i32 s25, s24, s23 // (op 1115) 


s_and_b32 s23, s0, s25 // (op 1115) 


s_add_i32 s0, s22, s23 // (op 1115) 


s_ashr_i32 s22, s0, s47 // (op 1115) 


s_lshl_b32 s0, s44, 24 // (op 1115) 
s_ashr_i32 s47, s0, 31 // (op 1115) 

s_xor_b32 s0, s22, s47 // (op 1115) 

s_sub_i32 s22, s0, s47 // (op 1115) 


s_mul_i32 s0, s46, s11 // (op 1115) 
s_add_u32 s24, s6, 64 // (op 1115) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1115) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 1115) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 1115) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 6 // (op 1115) 

s_sub_i32 s23, s24, s0 // (op 1115) 

s_mul_i32 s47, s22, s23 // (op 1115) 


s_sub_u32 s22, s18, s47 // (op 1115) 


s_add_u32 s18, s22, s0 // (op 1115) 


s_cmp_eq_i32 s19, 0 // (op 1115) 
s_mov_b32 s0, scc // (op 1115) 

s_cmp_eq_i32 s0, 0 // (op 1115) 
s_cselect_b32 s19, s18, s1 // (op 1115) 



v_lshl_add_u32 v20, s19, 6, v22 // (op 1115) 


v_mul_lo_u32 v22, v20, s34 // (op 1115) 
v_add_u32 v22, v22, v23 // (op 1115) 


s_mul_i32 s1, s34, s7 // (op 1117) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1117) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1117) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1117) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1117) most significant: sum
s_add_u32 s1, s1, s18 // (op 1117) most significant: sum


s_mov_b32 s64, s36 // (op 1117) reg expression
s_mov_b32 s65, s37 // (op 1117) reg expression
s_mov_b32 s66, s0 // (op 1117) reg expression
s_mov_b32 s67, 131072 // (op 1117) call()

s_mul_hi_i32 s47, s16, s10 // (op 1160) 
v_and_b32 v20, 15, v1 // (op 1160) 
s_add_i32 s0, s47, s16 // (op 1160) 

v_and_b32 v23, 63, v1 // (op 1160) 
s_ashr_i32 s1, s0, 31 // (op 1160) 
v_lshrrev_b32 v24, 4, v23 // (op 1160) 

s_cmp_eq_i32 s10, 0 // (op 1160) 
s_mov_b32 s47, scc // (op 1160) 
s_cmp_eq_i32 s47, 0 // (op 1160) 
s_cselect_b32 s18, 0, -1 // (op 1160) 

s_and_b32 s19, s13, 31 // (op 1160) 
s_lshl_b32 s47, 1, s19 // (op 1160) 
s_add_i32 s22, s47, s18 // (op 1160) 


s_and_b32 s18, s1, s22 // (op 1160) 


s_add_i32 s1, s0, s18 // (op 1160) 


s_ashr_i32 s0, s1, s19 // (op 1160) 


s_lshl_b32 s1, s13, 24 // (op 1160) 
s_ashr_i32 s47, s1, 31 // (op 1160) 

s_xor_b32 s1, s0, s47 // (op 1160) 

s_sub_i32 s0, s1, s47 // (op 1160) 


s_mul_hi_i32 s1, s0, s14 // (op 1160) 
s_add_i32 s47, s1, s0 // (op 1160) 

s_ashr_i32 s1, s47, 31 // (op 1160) 
s_cmp_eq_i32 s14, 0 // (op 1160) 
s_mov_b32 s18, scc // (op 1160) 
s_cmp_eq_i32 s18, 0 // (op 1160) 
s_cselect_b32 s19, 0, -1 // (op 1160) 

s_and_b32 s18, s15, 31 // (op 1160) 
s_lshl_b32 s22, 1, s18 // (op 1160) 
s_add_i32 s23, s22, s19 // (op 1160) 


s_and_b32 s19, s1, s23 // (op 1160) 


s_add_i32 s1, s47, s19 // (op 1160) 


s_ashr_i32 s19, s1, s18 // (op 1160) 


s_lshl_b32 s1, s15, 24 // (op 1160) 
s_ashr_i32 s18, s1, 31 // (op 1160) 

s_xor_b32 s1, s19, s18 // (op 1160) 

s_sub_i32 s19, s1, s18 // (op 1160) 


s_add_u32 s22, s4, 16 // (op 1160) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1160) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1160) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1160) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1160) 

s_mul_i32 s1, s19, s22 // (op 1160) 

s_sub_i32 s47, s0, s1 // (op 1160) 


s_mul_hi_i32 s0, s16, s9 // (op 1160) 
s_add_i32 s1, s0, s16 // (op 1160) 

s_ashr_i32 s0, s1, 31 // (op 1160) 
s_cmp_eq_i32 s9, 0 // (op 1160) 
s_mov_b32 s18, scc // (op 1160) 
s_cmp_eq_i32 s18, 0 // (op 1160) 
s_cselect_b32 s19, 0, -1 // (op 1160) 

s_and_b32 s18, s45, 31 // (op 1160) 
s_lshl_b32 s24, 1, s18 // (op 1160) 
s_add_i32 s25, s24, s19 // (op 1160) 


s_and_b32 s19, s0, s25 // (op 1160) 


s_add_i32 s0, s1, s19 // (op 1160) 


s_ashr_i32 s1, s0, s18 // (op 1160) 


s_lshl_b32 s0, s45, 24 // (op 1160) 
s_ashr_i32 s18, s0, 31 // (op 1160) 

s_xor_b32 s0, s1, s18 // (op 1160) 

s_sub_i32 s1, s0, s18 // (op 1160) 


s_mul_i32 s0, s11, s22 // (op 1160) 

s_mul_i32 s18, s8, s0 // (op 1160) 

s_mul_i32 s0, s1, s18 // (op 1160) 

s_sub_u32 s18, s16, s0 // (op 1160) 

s_mul_hi_i32 s0, s18, s12 // (op 1160) 
s_add_i32 s19, s0, s18 // (op 1160) 


s_ashr_i32 s0, s19, 31 // (op 1160) 
s_cmp_eq_i32 s12, 0 // (op 1160) 
s_mov_b32 s18, scc // (op 1160) 
s_cmp_eq_i32 s18, 0 // (op 1160) 
s_cselect_b32 s22, 0, -1 // (op 1160) 

s_and_b32 s18, s44, 31 // (op 1160) 
s_lshl_b32 s23, 1, s18 // (op 1160) 
s_add_i32 s24, s23, s22 // (op 1160) 


s_and_b32 s25, s0, s24 // (op 1160) 


s_add_i32 s0, s19, s25 // (op 1160) 


s_ashr_i32 s19, s0, s18 // (op 1160) 


s_lshl_b32 s0, s44, 24 // (op 1160) 
s_ashr_i32 s18, s0, 31 // (op 1160) 

s_xor_b32 s0, s19, s18 // (op 1160) 

s_sub_i32 s19, s0, s18 // (op 1160) 


s_cmp_eq_i32 s1, 0 // (op 1160) 
s_mov_b32 s0, scc // (op 1160) 

s_cmp_eq_i32 s0, 0 // (op 1160) 
s_cselect_b32 s1, s19, s47 // (op 1160) 



v_lshl_add_u32 v23, s1, 4, v20 // (op 1160) 


v_mul_lo_u32 v20, v23, s38 // (op 1160) 

v_add_u32 v23, v24, v20 // (op 1160) 


s_mul_i32 s1, s38, s5 // (op 1162) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1162) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1162) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1162) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1162) most significant: sum
s_add_u32 s1, s1, s18 // (op 1162) most significant: sum


s_mov_b32 s68, s40 // (op 1162) reg expression
s_mov_b32 s69, s41 // (op 1162) reg expression
s_mov_b32 s70, s0 // (op 1162) reg expression
s_mov_b32 s71, 131072 // (op 1162) call()

s_mul_hi_i32 s47, s16, s10 // (op 1179) 
v_and_b32 v20, 15, v1 // (op 1179) 
s_add_i32 s0, s47, s16 // (op 1179) 

v_and_b32 v24, 63, v1 // (op 1179) 
s_ashr_i32 s1, s0, 31 // (op 1179) 
v_lshrrev_b32 v25, 4, v24 // (op 1179) 

s_cmp_eq_i32 s10, 0 // (op 1179) 
s_mov_b32 s47, scc // (op 1179) 
s_cmp_eq_i32 s47, 0 // (op 1179) 
s_cselect_b32 s18, 0, -1 // (op 1179) 

s_and_b32 s19, s13, 31 // (op 1179) 
s_lshl_b32 s47, 1, s19 // (op 1179) 
s_add_i32 s22, s47, s18 // (op 1179) 


s_and_b32 s18, s1, s22 // (op 1179) 


s_add_i32 s1, s0, s18 // (op 1179) 


s_ashr_i32 s0, s1, s19 // (op 1179) 


s_lshl_b32 s1, s13, 24 // (op 1179) 
s_ashr_i32 s47, s1, 31 // (op 1179) 

s_xor_b32 s1, s0, s47 // (op 1179) 

s_sub_i32 s0, s1, s47 // (op 1179) 


s_mul_hi_i32 s1, s0, s14 // (op 1179) 
s_add_i32 s47, s1, s0 // (op 1179) 

s_ashr_i32 s1, s47, 31 // (op 1179) 
s_cmp_eq_i32 s14, 0 // (op 1179) 
s_mov_b32 s18, scc // (op 1179) 
s_cmp_eq_i32 s18, 0 // (op 1179) 
s_cselect_b32 s19, 0, -1 // (op 1179) 

s_and_b32 s18, s15, 31 // (op 1179) 
s_lshl_b32 s22, 1, s18 // (op 1179) 
s_add_i32 s23, s22, s19 // (op 1179) 


s_and_b32 s19, s1, s23 // (op 1179) 


s_add_i32 s1, s47, s19 // (op 1179) 


s_ashr_i32 s19, s1, s18 // (op 1179) 


s_lshl_b32 s1, s15, 24 // (op 1179) 
s_ashr_i32 s18, s1, 31 // (op 1179) 

s_xor_b32 s1, s19, s18 // (op 1179) 

s_sub_i32 s19, s1, s18 // (op 1179) 


s_add_u32 s22, s4, 16 // (op 1179) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1179) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1179) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1179) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1179) 

s_mul_i32 s1, s19, s22 // (op 1179) 

s_sub_i32 s47, s0, s1 // (op 1179) 


s_mul_hi_i32 s0, s16, s9 // (op 1179) 
s_add_i32 s1, s0, s16 // (op 1179) 

s_ashr_i32 s0, s1, 31 // (op 1179) 
s_cmp_eq_i32 s9, 0 // (op 1179) 
s_mov_b32 s18, scc // (op 1179) 
s_cmp_eq_i32 s18, 0 // (op 1179) 
s_cselect_b32 s19, 0, -1 // (op 1179) 

s_and_b32 s18, s45, 31 // (op 1179) 
s_lshl_b32 s24, 1, s18 // (op 1179) 
s_add_i32 s25, s24, s19 // (op 1179) 


s_and_b32 s19, s0, s25 // (op 1179) 


s_add_i32 s0, s1, s19 // (op 1179) 


s_ashr_i32 s1, s0, s18 // (op 1179) 


s_lshl_b32 s0, s45, 24 // (op 1179) 
s_ashr_i32 s18, s0, 31 // (op 1179) 

s_xor_b32 s0, s1, s18 // (op 1179) 

s_sub_i32 s1, s0, s18 // (op 1179) 


s_mul_i32 s0, s11, s22 // (op 1179) 

s_mul_i32 s18, s8, s0 // (op 1179) 

s_mul_i32 s0, s1, s18 // (op 1179) 

s_sub_u32 s18, s16, s0 // (op 1179) 

s_mul_hi_i32 s0, s18, s12 // (op 1179) 
s_add_i32 s19, s0, s18 // (op 1179) 


s_ashr_i32 s0, s19, 31 // (op 1179) 
s_cmp_eq_i32 s12, 0 // (op 1179) 
s_mov_b32 s18, scc // (op 1179) 
s_cmp_eq_i32 s18, 0 // (op 1179) 
s_cselect_b32 s22, 0, -1 // (op 1179) 

s_and_b32 s18, s44, 31 // (op 1179) 
s_lshl_b32 s23, 1, s18 // (op 1179) 
s_add_i32 s24, s23, s22 // (op 1179) 


s_and_b32 s25, s0, s24 // (op 1179) 


s_add_i32 s0, s19, s25 // (op 1179) 


s_ashr_i32 s19, s0, s18 // (op 1179) 


s_lshl_b32 s0, s44, 24 // (op 1179) 
s_ashr_i32 s18, s0, 31 // (op 1179) 

s_xor_b32 s0, s19, s18 // (op 1179) 

s_sub_i32 s19, s0, s18 // (op 1179) 


s_cmp_eq_i32 s1, 0 // (op 1179) 
s_mov_b32 s0, scc // (op 1179) 

s_cmp_eq_i32 s0, 0 // (op 1179) 
s_cselect_b32 s1, s19, s47 // (op 1179) 



v_lshl_add_u32 v24, s1, 4, v20 // (op 1179) 


v_mul_lo_u32 v20, v24, s38 // (op 1179) 

v_add_u32 v24, v25, v20 // (op 1179) 


s_mul_i32 s1, s38, s5 // (op 1181) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1181) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1181) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1181) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1181) most significant: sum
s_add_u32 s1, s1, s18 // (op 1181) most significant: sum


s_mov_b32 s68, s40 // (op 1181) reg expression
s_mov_b32 s69, s41 // (op 1181) reg expression
s_mov_b32 s70, s0 // (op 1181) reg expression
s_mov_b32 s71, 131072 // (op 1181) call()




buffer_load_ubyte v20, v23, s[68:71], 0 offen offset: 0 // (op 774) Load value
v_and_b32 v25, -8, v1 // (op 1302) 
v_sub_u32 v26, v1, v25 // (op 1302) 

v_lshlrev_b32 v25, 4, v26 // (op 1302) 

v_lshrrev_b32 v26, 3, v1 // (op 1302) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1302) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 1302) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s47 // (op 916) reg expression
buffer_load_dwordx4 v12, s[60:63], 0 offen offset: 0 lds // (op 916) Load value direct to lds

v_and_b32 v25, -8, v1 // (op 1245) 
v_sub_u32 v26, v1, v25 // (op 1245) 

v_lshlrev_b32 v25, 4, v26 // (op 1245) 

v_lshrrev_b32 v26, 3, v1 // (op 1245) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1245) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 1245) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 4 // (op 921) 
v_mov_b64 v[26:27], s[0:1] // (op 921) convert


v_mov_b64 v[28:29], v[16:17] // (op 921) 
s_add_u32 s0, 2048, s47 // (op 921) 
s_mov_b32 m0, s0 // (op 921) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 921) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 921) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 921) most significant half

s_add_u32 m0, m0, 2048 // (op 921) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 921) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 921) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 921) most significant half

s_add_u32 m0, m0, 2048 // (op 921) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 921) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 921) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 921) most significant half

s_add_u32 m0, m0, 2048 // (op 921) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 921) Load value direct to lds





v_mov_b64 v[26:27], s[34:35] // (op 784) convert

v_mov_b32 v25, v21 // (op 784) 
buffer_load_ubyte v28, v25, s[64:67], 0 offen offset: 0 // (op 784) Load value
v_add_u32 v25, v25, v26 // (op 784) 
buffer_load_ubyte v29, v25, s[64:67], 0 offen offset: 0 // (op 784) Load value





buffer_load_ubyte v25, v23, s[68:71], 0 offen offset:4 // (op 788) Load value

v_and_b32 v23, -8, v1 // (op 1461) 
v_sub_u32 v26, v1, v23 // (op 1461) 

v_lshlrev_b32 v23, 4, v26 // (op 1461) 

v_lshrrev_b32 v26, 3, v1 // (op 1461) 
v_lshl_add_u32 v27, v26, 7, v23 // (op 1461) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 1461) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s0, 10240, s47 // (op 926) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s0 // (op 926) reg expression
s_mov_b32 s1, 128 // (op 926) call()
buffer_load_dwordx4 v12, s[60:63], s1 offen offset: 0 lds // (op 926) Load value direct to lds




v_and_b32 v23, -8, v1 // (op 1417) 
v_sub_u32 v12, v1, v23 // (op 1417) 

v_lshlrev_b32 v13, 4, v12 // (op 1417) 

v_lshrrev_b32 v12, 3, v1 // (op 1417) 
v_lshl_add_u32 v23, v12, 7, v13 // (op 1417) 


s_nop 0
v_readfirstlane_b32 s47, v23 // (op 1417) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 4 // (op 928) 
v_mov_b64 v[12:13], s[0:1] // (op 928) convert


v_mov_b64 v[26:27], v[16:17] // (op 928) 
s_add_u32 s0, 12288, s47 // (op 928) 
s_mov_b32 m0, s0 // (op 928) reg expression
s_mov_b32 s1, 128 // (op 928) call()
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 928) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 928) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 928) most significant half

s_add_u32 m0, m0, 2048 // (op 928) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 928) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 928) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 928) most significant half

s_add_u32 m0, m0, 2048 // (op 928) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 928) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 928) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 928) most significant half

s_add_u32 m0, m0, 2048 // (op 928) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 928) Load value direct to lds







v_mov_b64 v[12:13], s[34:35] // (op 798) convert

v_mov_b32 v23, v21 // (op 798) 
buffer_load_ubyte v16, v23, s[64:67], 0 offen offset:4 // (op 798) Load value
v_add_u32 v23, v23, v12 // (op 798) 
buffer_load_ubyte v17, v23, s[64:67], 0 offen offset:4 // (op 798) Load value



v_and_b32 v21, -4, v1 // (op 1355) 
v_sub_u32 v23, v1, v21 // (op 1355) 

v_lshrrev_b32 v21, 2, v1 // (op 1355) 
v_lshl_add_u32 v12, v21, 3, v23 // (op 1355) 





s_waitcnt vmcnt(9)
ds_write_b8 v12, v28 offset:20480 // (op 802) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v12, v29 offset:20484 // (op 802) Store local data 

s_barrier  // (op 1559) 



ds_read_b128 v[30:33], v4 // (op 825) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 825) Load local data 
s_barrier  // (op 1543) 



ds_read_b128 v[38:41], v7 offset:2048 // (op 816) Load local data 
ds_read_b128 v[42:45], v7 offset:2112 // (op 816) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1575) 



ds_read_u8 v21, v0 offset:20480 // (op 829) Load local data 



ds_read_b128 v[46:49], v7 offset:6144 // (op 821) Load local data 
ds_read_b128 v[50:53], v7 offset:6208 // (op 821) Load local data 



ds_read_u8 v23, v0 offset:20608 // (op 834) Load local data 
v_and_b32 v12, -8, v1 // (op 1217) 
v_sub_u32 v13, v1, v12 // (op 1217) 

v_lshlrev_b32 v12, 4, v13 // (op 1217) 

v_lshrrev_b32 v13, 3, v1 // (op 1217) 
v_lshl_add_u32 v26, v13, 7, v12 // (op 1217) 


s_nop 0
v_readfirstlane_b32 s47, v26 // (op 1217) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 1274) 
v_sub_u32 v13, v1, v12 // (op 1274) 

v_lshlrev_b32 v12, 4, v13 // (op 1274) 

v_lshrrev_b32 v13, 3, v1 // (op 1274) 
v_lshl_add_u32 v26, v13, 7, v12 // (op 1274) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1274) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 1326) 
v_sub_u32 v13, v1, v12 // (op 1326) 

v_lshrrev_b32 v12, 2, v1 // (op 1326) 
v_lshl_add_u32 v26, v12, 3, v13 // (op 1326) 


v_and_b32 v27, -8, v1 // (op 1402) 
v_sub_u32 v12, v1, v27 // (op 1402) 

v_lshlrev_b32 v13, 4, v12 // (op 1402) 

v_lshrrev_b32 v12, 3, v1 // (op 1402) 
v_lshl_add_u32 v27, v12, 7, v13 // (op 1402) 


s_nop 0
v_readfirstlane_b32 s1, v27 // (op 1402) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -8, v1 // (op 1446) 
v_sub_u32 v12, v1, v27 // (op 1446) 

v_lshlrev_b32 v13, 4, v12 // (op 1446) 

v_lshrrev_b32 v12, 3, v1 // (op 1446) 
v_lshl_add_u32 v27, v12, 7, v13 // (op 1446) 


s_nop 0
v_readfirstlane_b32 s18, v27 // (op 1446) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -4, v1 // (op 1485) 
v_sub_u32 v12, v1, v27 // (op 1485) 

v_lshrrev_b32 v13, 2, v1 // (op 1485) 
v_lshl_add_u32 v27, v13, 3, v12 // (op 1485) 


s_mov_b32 s19, 0 // (op 311) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 310) 
s_mov_b32 s24, s22 // (op 310) convert

s_and_b32 s25, s24, 4294967294 // (op 310) 

s_cmp_lt_u32 s19, s25 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_310 // (op 310) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_310 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_310:
s_add_u32 s19, s19, 2 // (op 312) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v21, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 82) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[30:37], a[4:7], v23, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 346) 


v_mov_b32 v12, v24 // (op 80) 

buffer_load_ubyte v20, v12, s[68:71], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 1562) 


v_mov_b64 v[54:55], v[14:15] // (op 888) 

s_mov_b32 m0, s0 // (op 888) reg expression
s_mov_b32 s22, 256 // (op 888) call()
buffer_load_dwordx4 v54, s[60:63], s22 offen offset: 0 lds // (op 888) Load value direct to lds

s_barrier  // (op 1546) 

s_lshl_b64 s[22:23], s[26:27], 4 // (op 883) 
v_mov_b64 v[56:57], s[22:23] // (op 883) convert


v_mov_b64 v[58:59], v[18:19] // (op 883) 
v_mov_b64 v[60:61], v[58:59] // (op 883) 
s_add_u32 s22, 2048, s47 // (op 883) 
s_mov_b32 m0, s22 // (op 883) reg expression
s_mov_b32 s23, 256 // (op 883) call()
buffer_load_dwordx4 v60, s[48:51], s23 offen offset: 0 lds // (op 883) Load value direct to lds
v_add_co_u32 v60, s[24:25], v60, v56 // (op 883) least significant half
v_addc_co_u32 v61, s[24:25], 0, v61, s[24:25] // (op 883) most significant half

s_add_u32 m0, m0, 2048 // (op 883) 
buffer_load_dwordx4 v60, s[48:51], s23 offen offset: 0 lds // (op 883) Load value direct to lds
v_add_co_u32 v60, s[24:25], v60, v56 // (op 883) least significant half
v_addc_co_u32 v61, s[24:25], 0, v61, s[24:25] // (op 883) most significant half

s_add_u32 m0, m0, 2048 // (op 883) 
buffer_load_dwordx4 v60, s[48:51], s23 offen offset: 0 lds // (op 883) Load value direct to lds
v_add_co_u32 v60, s[24:25], v60, v56 // (op 883) least significant half
v_addc_co_u32 v61, s[24:25], 0, v61, s[24:25] // (op 883) most significant half

s_add_u32 m0, m0, 2048 // (op 883) 
buffer_load_dwordx4 v60, s[48:51], s23 offen offset: 0 lds // (op 883) Load value direct to lds





v_mov_b64 v[56:57], s[34:35] // (op 17) convert

v_mov_b32 v13, v22 // (op 17) 
v_mov_b32 v60, v13 // (op 17) 
buffer_load_ubyte v28, v60, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v60, v60, v56 // (op 17) 
buffer_load_ubyte v29, v60, s[64:67], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 1607) 



s_waitcnt vmcnt(9)
ds_write_b8 v27, v16 offset:20736 // (op 236) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v27, v17 offset:20740 // (op 236) Store local data 
s_barrier  // (op 1598) 



ds_read_b128 v[30:33], v5 offset:10240 // (op 229) Load local data 
ds_read_b128 v[34:37], v5 offset:10304 // (op 229) Load local data 
s_barrier  // (op 1592) 



ds_read_b128 v[38:41], v11 offset:12288 // (op 227) Load local data 
ds_read_b128 v[42:45], v11 offset:12352 // (op 227) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1604) 



ds_read_u8 v21, v10 offset:20736 // (op 231) Load local data 



ds_read_b128 v[46:49], v11 offset:16384 // (op 353) Load local data 
ds_read_b128 v[50:53], v11 offset:16448 // (op 353) Load local data 



ds_read_u8 v23, v10 offset:20864 // (op 357) Load local data 
s_barrier  // (op 1658) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v21, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 234) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[30:37], a[4:7], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 347) 



buffer_load_ubyte v25, v12, s[68:71], 0 offen offset:12 // (op 233) Load value

s_barrier  // (op 1601) 
v_add_u32 v24, 8, v24 // (op 1171) 


s_add_u32 s22, 10240, s18 // (op 911) 

s_mov_b32 m0, s22 // (op 911) reg expression
s_mov_b32 s23, 384 // (op 911) call()
buffer_load_dwordx4 v54, s[60:63], s23 offen offset: 0 lds // (op 911) Load value direct to lds



s_barrier  // (op 1595) 
v_add_co_u32 v14, vcc, 256, v14 // (op 1043) least significant half

v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1043) most significant half
s_lshl_b64 s[22:23], s[26:27], 4 // (op 906) 
v_mov_b64 v[54:55], s[22:23] // (op 906) convert


v_mov_b64 v[56:57], v[58:59] // (op 906) 
s_add_u32 s22, 12288, s1 // (op 906) 
s_mov_b32 m0, s22 // (op 906) reg expression
s_mov_b32 s23, 384 // (op 906) call()
buffer_load_dwordx4 v56, s[48:51], s23 offen offset: 0 lds // (op 906) Load value direct to lds
v_add_co_u32 v56, s[24:25], v56, v54 // (op 906) least significant half
v_addc_co_u32 v57, s[24:25], 0, v57, s[24:25] // (op 906) most significant half

s_add_u32 m0, m0, 2048 // (op 906) 
buffer_load_dwordx4 v56, s[48:51], s23 offen offset: 0 lds // (op 906) Load value direct to lds
v_add_co_u32 v56, s[24:25], v56, v54 // (op 906) least significant half
v_addc_co_u32 v57, s[24:25], 0, v57, s[24:25] // (op 906) most significant half

s_add_u32 m0, m0, 2048 // (op 906) 
buffer_load_dwordx4 v56, s[48:51], s23 offen offset: 0 lds // (op 906) Load value direct to lds
v_add_co_u32 v56, s[24:25], v56, v54 // (op 906) least significant half
v_addc_co_u32 v57, s[24:25], 0, v57, s[24:25] // (op 906) most significant half

s_add_u32 m0, m0, 2048 // (op 906) 
buffer_load_dwordx4 v56, s[48:51], s23 offen offset: 0 lds // (op 906) Load value direct to lds






v_add_co_u32 v18, vcc, 256, v18 // (op 979) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 979) most significant half
v_mov_b64 v[54:55], s[34:35] // (op 238) convert

v_mov_b32 v12, v13 // (op 238) 
buffer_load_ubyte v16, v12, s[64:67], 0 offen offset:12 // (op 238) Load value
v_add_u32 v12, v12, v54 // (op 238) 
buffer_load_ubyte v17, v12, s[64:67], 0 offen offset:12 // (op 238) Load value



v_add_u32 v22, 8, v22 // (op 1107) 
s_barrier  // (op 1578) 



s_waitcnt vmcnt(9)
ds_write_b8 v26, v28 offset:20480 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v26, v29 offset:20484 // (op 21) Store local data 
s_barrier  // (op 1568) 



ds_read_b128 v[30:33], v4 // (op 70) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 70) Load local data 
s_barrier  // (op 1552) 



ds_read_b128 v[38:41], v7 offset:2048 // (op 65) Load local data 
ds_read_b128 v[42:45], v7 offset:2112 // (op 65) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1585) 



ds_read_u8 v21, v0 offset:20480 // (op 75) Load local data 



ds_read_b128 v[46:49], v7 offset:6144 // (op 340) Load local data 
ds_read_b128 v[50:53], v7 offset:6208 // (op 340) Load local data 



ds_read_u8 v23, v0 offset:20608 // (op 344) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 310) 
s_mov_b32 s24, s22 // (op 310) convert

s_and_b32 s25, s24, 4294967294 // (op 310) 

s_cmp_lt_u32 s19, s25 // (op 310) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_310 // (op 310) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_310 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_310:













s_barrier  // (op 1642) 
s_mul_hi_i32 s47, s16, s10 // (op 941) 
v_lshrrev_b32 v22, 3, v1 // (op 941) 
s_add_i32 s0, s47, s16 // (op 941) 

v_and_b32 v24, -8, v1 // (op 941) 
s_ashr_i32 s1, s0, 31 // (op 941) 
v_sub_u32 v12, v1, v24 // (op 941) 

s_cmp_eq_i32 s10, 0 // (op 941) 
s_mov_b32 s47, scc // (op 941) 
v_lshlrev_b32 v24, 4, v12 // (op 941) 

s_cmp_eq_i32 s47, 0 // (op 941) 
s_cselect_b32 s18, 0, -1 // (op 941) 

s_and_b32 s19, s13, 31 // (op 941) 
s_lshl_b32 s47, 1, s19 // (op 941) 
s_add_i32 s22, s47, s18 // (op 941) 


s_and_b32 s18, s1, s22 // (op 941) 


s_add_i32 s1, s0, s18 // (op 941) 


s_ashr_i32 s0, s1, s19 // (op 941) 


s_lshl_b32 s1, s13, 24 // (op 941) 
s_ashr_i32 s47, s1, 31 // (op 941) 

s_xor_b32 s1, s0, s47 // (op 941) 

s_sub_i32 s0, s1, s47 // (op 941) 


s_mul_hi_i32 s1, s0, s14 // (op 941) 
s_add_i32 s47, s1, s0 // (op 941) 

s_ashr_i32 s1, s47, 31 // (op 941) 
s_cmp_eq_i32 s14, 0 // (op 941) 
s_mov_b32 s18, scc // (op 941) 
s_cmp_eq_i32 s18, 0 // (op 941) 
s_cselect_b32 s19, 0, -1 // (op 941) 

s_and_b32 s18, s15, 31 // (op 941) 
s_lshl_b32 s22, 1, s18 // (op 941) 
s_add_i32 s23, s22, s19 // (op 941) 


s_and_b32 s19, s1, s23 // (op 941) 


s_add_i32 s1, s47, s19 // (op 941) 


s_ashr_i32 s19, s1, s18 // (op 941) 


s_lshl_b32 s1, s15, 24 // (op 941) 
s_ashr_i32 s18, s1, 31 // (op 941) 

s_xor_b32 s1, s19, s18 // (op 941) 

s_sub_i32 s19, s1, s18 // (op 941) 


s_mul_i32 s1, s0, s11 // (op 941) 

s_sub_u32 s0, s16, s1 // (op 941) 

s_mul_i32 s1, s19, s11 // (op 941) 
s_add_u32 s1, s1, s0 // (op 941) 


s_mul_hi_i32 s0, s16, s9 // (op 941) 
s_add_i32 s47, s0, s16 // (op 941) 

s_ashr_i32 s0, s47, 31 // (op 941) 
s_cmp_eq_i32 s9, 0 // (op 941) 
s_mov_b32 s18, scc // (op 941) 
s_cmp_eq_i32 s18, 0 // (op 941) 
s_cselect_b32 s19, 0, -1 // (op 941) 

s_and_b32 s18, s45, 31 // (op 941) 
s_lshl_b32 s22, 1, s18 // (op 941) 
s_add_i32 s23, s22, s19 // (op 941) 


s_and_b32 s19, s0, s23 // (op 941) 


s_add_i32 s0, s47, s19 // (op 941) 


s_ashr_i32 s19, s0, s18 // (op 941) 


s_lshl_b32 s0, s45, 24 // (op 941) 
s_ashr_i32 s18, s0, 31 // (op 941) 

s_xor_b32 s0, s19, s18 // (op 941) 

s_sub_i32 s19, s0, s18 // (op 941) 


s_add_u32 s22, s4, 16 // (op 941) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 941) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 941) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 941) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 941) 

s_mul_i32 s0, s11, s22 // (op 941) 

s_mul_i32 s18, s8, s0 // (op 941) 

s_mul_i32 s0, s19, s18 // (op 941) 

s_sub_u32 s18, s16, s0 // (op 941) 

s_mov_b32 s0, s18 // (op 941) 
s_mul_hi_i32 s47, s0, s12 // (op 941) 
s_add_i32 s22, s47, s0 // (op 941) 


s_ashr_i32 s0, s22, 31 // (op 941) 
s_cmp_eq_i32 s12, 0 // (op 941) 
s_mov_b32 s47, scc // (op 941) 
s_cmp_eq_i32 s47, 0 // (op 941) 
s_cselect_b32 s23, 0, -1 // (op 941) 

s_and_b32 s47, s44, 31 // (op 941) 
s_lshl_b32 s24, 1, s47 // (op 941) 
s_add_i32 s25, s24, s23 // (op 941) 


s_and_b32 s23, s0, s25 // (op 941) 


s_add_i32 s0, s22, s23 // (op 941) 


s_ashr_i32 s22, s0, s47 // (op 941) 


s_lshl_b32 s0, s44, 24 // (op 941) 
s_ashr_i32 s47, s0, 31 // (op 941) 

s_xor_b32 s0, s22, s47 // (op 941) 

s_sub_i32 s22, s0, s47 // (op 941) 


s_mul_i32 s0, s46, s11 // (op 941) 
s_add_u32 s24, s6, 64 // (op 941) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 941) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 941) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 941) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 941) 

s_sub_i32 s23, s24, s0 // (op 941) 

s_mul_i32 s47, s22, s23 // (op 941) 


s_sub_u32 s22, s18, s47 // (op 941) 


s_add_u32 s18, s22, s0 // (op 941) 


s_cmp_eq_i32 s19, 0 // (op 941) 
s_mov_b32 s0, scc // (op 941) 

s_cmp_eq_i32 s0, 0 // (op 941) 
s_cselect_b32 s19, s18, s1 // (op 941) 



v_lshl_add_u32 v12, s19, 6, v22 // (op 941) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 941) 
s_lshr_b32 s47, s0, 1 // (op 941) 

v_lshl_add_u32 v22, s47, 8, v24 // (op 941) 


v_mov_b32 v15, 0 // (op 941) convert
v_mov_b32 v14, v22 // (op 941) convert

v_mul_lo_u32 v19, v12, s27 // (op 941) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 941) most significant: high of low * low
v_mul_lo_u32 v18, v12, s26 // (op 941) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 941) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 941) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 941) most significant half



s_mul_i32 s1, s26, s7 // (op 943) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 943) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 943) least significant
s_mul_hi_u32 s18, s26, s6 // (op 943) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 943) most significant: sum
s_add_u32 s1, s1, s18 // (op 943) most significant: sum


s_mov_b32 s48, s28 // (op 943) reg expression
s_mov_b32 s49, s29 // (op 943) reg expression
s_mov_b32 s50, s0 // (op 943) reg expression
s_mov_b32 s51, 131072 // (op 943) call()

s_mul_hi_i32 s47, s16, s10 // (op 1009) 
v_lshrrev_b32 v20, 3, v1 // (op 1009) 
s_add_i32 s0, s47, s16 // (op 1009) 

v_and_b32 v22, -8, v1 // (op 1009) 
s_ashr_i32 s1, s0, 31 // (op 1009) 
v_sub_u32 v24, v1, v22 // (op 1009) 

s_cmp_eq_i32 s10, 0 // (op 1009) 
s_mov_b32 s47, scc // (op 1009) 
v_lshlrev_b32 v22, 4, v24 // (op 1009) 

s_cmp_eq_i32 s47, 0 // (op 1009) 
s_cselect_b32 s18, 0, -1 // (op 1009) 

s_and_b32 s19, s13, 31 // (op 1009) 
s_lshl_b32 s47, 1, s19 // (op 1009) 
s_add_i32 s22, s47, s18 // (op 1009) 


s_and_b32 s18, s1, s22 // (op 1009) 


s_add_i32 s1, s0, s18 // (op 1009) 


s_ashr_i32 s0, s1, s19 // (op 1009) 


s_lshl_b32 s1, s13, 24 // (op 1009) 
s_ashr_i32 s47, s1, 31 // (op 1009) 

s_xor_b32 s1, s0, s47 // (op 1009) 

s_sub_i32 s0, s1, s47 // (op 1009) 


s_mul_hi_i32 s1, s0, s14 // (op 1009) 
s_add_i32 s47, s1, s0 // (op 1009) 

s_ashr_i32 s1, s47, 31 // (op 1009) 
s_cmp_eq_i32 s14, 0 // (op 1009) 
s_mov_b32 s18, scc // (op 1009) 
s_cmp_eq_i32 s18, 0 // (op 1009) 
s_cselect_b32 s19, 0, -1 // (op 1009) 

s_and_b32 s18, s15, 31 // (op 1009) 
s_lshl_b32 s22, 1, s18 // (op 1009) 
s_add_i32 s23, s22, s19 // (op 1009) 


s_and_b32 s19, s1, s23 // (op 1009) 


s_add_i32 s1, s47, s19 // (op 1009) 


s_ashr_i32 s19, s1, s18 // (op 1009) 


s_lshl_b32 s1, s15, 24 // (op 1009) 
s_ashr_i32 s18, s1, 31 // (op 1009) 

s_xor_b32 s1, s19, s18 // (op 1009) 

s_sub_i32 s19, s1, s18 // (op 1009) 


s_add_u32 s22, s4, 16 // (op 1009) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1009) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1009) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1009) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1009) 

s_mul_i32 s1, s19, s22 // (op 1009) 

s_sub_i32 s47, s0, s1 // (op 1009) 


s_mul_hi_i32 s0, s16, s9 // (op 1009) 
s_add_i32 s1, s0, s16 // (op 1009) 

s_ashr_i32 s0, s1, 31 // (op 1009) 
s_cmp_eq_i32 s9, 0 // (op 1009) 
s_mov_b32 s18, scc // (op 1009) 
s_cmp_eq_i32 s18, 0 // (op 1009) 
s_cselect_b32 s19, 0, -1 // (op 1009) 

s_and_b32 s18, s45, 31 // (op 1009) 
s_lshl_b32 s24, 1, s18 // (op 1009) 
s_add_i32 s25, s24, s19 // (op 1009) 


s_and_b32 s19, s0, s25 // (op 1009) 


s_add_i32 s0, s1, s19 // (op 1009) 


s_ashr_i32 s1, s0, s18 // (op 1009) 


s_lshl_b32 s0, s45, 24 // (op 1009) 
s_ashr_i32 s18, s0, 31 // (op 1009) 

s_xor_b32 s0, s1, s18 // (op 1009) 

s_sub_i32 s1, s0, s18 // (op 1009) 


s_mul_i32 s0, s11, s22 // (op 1009) 

s_mul_i32 s18, s8, s0 // (op 1009) 

s_mul_i32 s0, s1, s18 // (op 1009) 

s_sub_u32 s18, s16, s0 // (op 1009) 

s_mul_hi_i32 s0, s18, s12 // (op 1009) 
s_add_i32 s19, s0, s18 // (op 1009) 


s_ashr_i32 s0, s19, 31 // (op 1009) 
s_cmp_eq_i32 s12, 0 // (op 1009) 
s_mov_b32 s18, scc // (op 1009) 
s_cmp_eq_i32 s18, 0 // (op 1009) 
s_cselect_b32 s22, 0, -1 // (op 1009) 

s_and_b32 s18, s44, 31 // (op 1009) 
s_lshl_b32 s23, 1, s18 // (op 1009) 
s_add_i32 s24, s23, s22 // (op 1009) 


s_and_b32 s25, s0, s24 // (op 1009) 


s_add_i32 s0, s19, s25 // (op 1009) 


s_ashr_i32 s19, s0, s18 // (op 1009) 


s_lshl_b32 s0, s44, 24 // (op 1009) 
s_ashr_i32 s18, s0, 31 // (op 1009) 

s_xor_b32 s0, s19, s18 // (op 1009) 

s_sub_i32 s19, s0, s18 // (op 1009) 


s_cmp_eq_i32 s1, 0 // (op 1009) 
s_mov_b32 s0, scc // (op 1009) 

s_cmp_eq_i32 s0, 0 // (op 1009) 
s_cselect_b32 s1, s19, s47 // (op 1009) 



v_lshl_add_u32 v24, s1, 4, v20 // (op 1009) 


v_mul_lo_u32 v13, v24, s31 // (op 1009) most significant: low of low * high
v_mul_hi_u32 v20, v24, s30 // (op 1009) most significant: high of low * low
v_mul_lo_u32 v12, v24, s30 // (op 1009) least significant: low of low * low
v_add_u32 v13, v13, v20 // (op 1009) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1009) 
s_lshr_b32 s47, s0, 1 // (op 1009) 

v_lshl_add_u32 v20, s47, 8, v22 // (op 1009) 


v_mov_b32 v15, 0 // (op 1009) convert
v_mov_b32 v14, v20 // (op 1009) convert

v_add_co_u32 v26, s[0:1], v14, v12 // (op 1009) least significant half
v_addc_co_u32 v27, s[0:1], v15, v13, s[0:1] // (op 1009) most significant half



s_mul_i32 s1, s30, s5 // (op 1011) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 1011) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 1011) least significant
s_mul_hi_u32 s18, s30, s4 // (op 1011) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1011) most significant: sum
s_add_u32 s1, s1, s18 // (op 1011) most significant: sum


s_mov_b32 s60, s32 // (op 1011) reg expression
s_mov_b32 s61, s33 // (op 1011) reg expression
s_mov_b32 s62, s0 // (op 1011) reg expression
s_mov_b32 s63, 131072 // (op 1011) call()

s_mul_hi_i32 s47, s16, s10 // (op 1073) 
v_lshrrev_b32 v20, 2, v1 // (op 1073) 
s_add_i32 s0, s47, s16 // (op 1073) 

v_lshlrev_b32 v22, 1, v20 // (op 1073) 

s_ashr_i32 s1, s0, 31 // (op 1073) 
v_and_b32 v20, -4, v1 // (op 1073) 
s_cmp_eq_i32 s10, 0 // (op 1073) 
s_mov_b32 s47, scc // (op 1073) 
v_sub_u32 v24, v1, v20 // (op 1073) 

s_cmp_eq_i32 s47, 0 // (op 1073) 
s_cselect_b32 s18, 0, -1 // (op 1073) 

s_and_b32 s19, s13, 31 // (op 1073) 
s_lshl_b32 s47, 1, s19 // (op 1073) 
s_add_i32 s22, s47, s18 // (op 1073) 


s_and_b32 s18, s1, s22 // (op 1073) 


s_add_i32 s1, s0, s18 // (op 1073) 


s_ashr_i32 s0, s1, s19 // (op 1073) 


s_lshl_b32 s1, s13, 24 // (op 1073) 
s_ashr_i32 s47, s1, 31 // (op 1073) 

s_xor_b32 s1, s0, s47 // (op 1073) 

s_sub_i32 s0, s1, s47 // (op 1073) 


s_mul_hi_i32 s1, s0, s14 // (op 1073) 
s_add_i32 s47, s1, s0 // (op 1073) 

s_ashr_i32 s1, s47, 31 // (op 1073) 
s_cmp_eq_i32 s14, 0 // (op 1073) 
s_mov_b32 s18, scc // (op 1073) 
s_cmp_eq_i32 s18, 0 // (op 1073) 
s_cselect_b32 s19, 0, -1 // (op 1073) 

s_and_b32 s18, s15, 31 // (op 1073) 
s_lshl_b32 s22, 1, s18 // (op 1073) 
s_add_i32 s23, s22, s19 // (op 1073) 


s_and_b32 s19, s1, s23 // (op 1073) 


s_add_i32 s1, s47, s19 // (op 1073) 


s_ashr_i32 s19, s1, s18 // (op 1073) 


s_lshl_b32 s1, s15, 24 // (op 1073) 
s_ashr_i32 s18, s1, 31 // (op 1073) 

s_xor_b32 s1, s19, s18 // (op 1073) 

s_sub_i32 s19, s1, s18 // (op 1073) 


s_mul_i32 s1, s0, s11 // (op 1073) 

s_sub_u32 s0, s16, s1 // (op 1073) 

s_mul_i32 s1, s19, s11 // (op 1073) 
s_add_u32 s1, s1, s0 // (op 1073) 


s_mul_hi_i32 s0, s16, s9 // (op 1073) 
s_add_i32 s47, s0, s16 // (op 1073) 

s_ashr_i32 s0, s47, 31 // (op 1073) 
s_cmp_eq_i32 s9, 0 // (op 1073) 
s_mov_b32 s18, scc // (op 1073) 
s_cmp_eq_i32 s18, 0 // (op 1073) 
s_cselect_b32 s19, 0, -1 // (op 1073) 

s_and_b32 s18, s45, 31 // (op 1073) 
s_lshl_b32 s22, 1, s18 // (op 1073) 
s_add_i32 s23, s22, s19 // (op 1073) 


s_and_b32 s19, s0, s23 // (op 1073) 


s_add_i32 s0, s47, s19 // (op 1073) 


s_ashr_i32 s19, s0, s18 // (op 1073) 


s_lshl_b32 s0, s45, 24 // (op 1073) 
s_ashr_i32 s18, s0, 31 // (op 1073) 

s_xor_b32 s0, s19, s18 // (op 1073) 

s_sub_i32 s19, s0, s18 // (op 1073) 


s_add_u32 s22, s4, 16 // (op 1073) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1073) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1073) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1073) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1073) 

s_mul_i32 s0, s11, s22 // (op 1073) 

s_mul_i32 s18, s8, s0 // (op 1073) 

s_mul_i32 s0, s19, s18 // (op 1073) 

s_sub_u32 s18, s16, s0 // (op 1073) 

s_mov_b32 s0, s18 // (op 1073) 
s_mul_hi_i32 s47, s0, s12 // (op 1073) 
s_add_i32 s22, s47, s0 // (op 1073) 


s_ashr_i32 s0, s22, 31 // (op 1073) 
s_cmp_eq_i32 s12, 0 // (op 1073) 
s_mov_b32 s47, scc // (op 1073) 
s_cmp_eq_i32 s47, 0 // (op 1073) 
s_cselect_b32 s23, 0, -1 // (op 1073) 

s_and_b32 s47, s44, 31 // (op 1073) 
s_lshl_b32 s24, 1, s47 // (op 1073) 
s_add_i32 s25, s24, s23 // (op 1073) 


s_and_b32 s23, s0, s25 // (op 1073) 


s_add_i32 s0, s22, s23 // (op 1073) 


s_ashr_i32 s22, s0, s47 // (op 1073) 


s_lshl_b32 s0, s44, 24 // (op 1073) 
s_ashr_i32 s47, s0, 31 // (op 1073) 

s_xor_b32 s0, s22, s47 // (op 1073) 

s_sub_i32 s22, s0, s47 // (op 1073) 


s_mul_i32 s0, s46, s11 // (op 1073) 
s_add_u32 s24, s6, 64 // (op 1073) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1073) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1073) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1073) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 1073) 

s_sub_i32 s23, s24, s0 // (op 1073) 

s_mul_i32 s47, s22, s23 // (op 1073) 


s_sub_u32 s22, s18, s47 // (op 1073) 


s_add_u32 s18, s22, s0 // (op 1073) 


s_cmp_eq_i32 s19, 0 // (op 1073) 
s_mov_b32 s0, scc // (op 1073) 

s_cmp_eq_i32 s0, 0 // (op 1073) 
s_cselect_b32 s19, s18, s1 // (op 1073) 



v_lshl_add_u32 v20, s19, 6, v22 // (op 1073) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1073) 
s_lshr_b32 s47, s0, 1 // (op 1073) 

v_lshl_add_u32 v22, s47, 3, v24 // (op 1073) 


v_mul_lo_u32 v24, v20, s34 // (op 1073) 
v_add_u32 v24, v24, v22 // (op 1073) 


s_mul_i32 s1, s34, s7 // (op 1075) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1075) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1075) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1075) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1075) most significant: sum
s_add_u32 s1, s1, s18 // (op 1075) most significant: sum


s_mov_b32 s64, s36 // (op 1075) reg expression
s_mov_b32 s65, s37 // (op 1075) reg expression
s_mov_b32 s66, s0 // (op 1075) reg expression
s_mov_b32 s67, 131072 // (op 1075) call()

s_mul_hi_i32 s47, s16, s10 // (op 1137) 
v_and_b32 v20, 15, v1 // (op 1137) 
s_add_i32 s0, s47, s16 // (op 1137) 

v_and_b32 v22, 63, v1 // (op 1137) 
s_ashr_i32 s1, s0, 31 // (op 1137) 
v_lshrrev_b32 v12, 4, v22 // (op 1137) 

s_cmp_eq_i32 s10, 0 // (op 1137) 
s_mov_b32 s47, scc // (op 1137) 
s_cmp_eq_i32 s47, 0 // (op 1137) 
s_cselect_b32 s18, 0, -1 // (op 1137) 

s_and_b32 s19, s13, 31 // (op 1137) 
s_lshl_b32 s47, 1, s19 // (op 1137) 
s_add_i32 s22, s47, s18 // (op 1137) 


s_and_b32 s18, s1, s22 // (op 1137) 


s_add_i32 s1, s0, s18 // (op 1137) 


s_ashr_i32 s0, s1, s19 // (op 1137) 


s_lshl_b32 s1, s13, 24 // (op 1137) 
s_ashr_i32 s47, s1, 31 // (op 1137) 

s_xor_b32 s1, s0, s47 // (op 1137) 

s_sub_i32 s0, s1, s47 // (op 1137) 


s_mul_hi_i32 s1, s0, s14 // (op 1137) 
s_add_i32 s47, s1, s0 // (op 1137) 

s_ashr_i32 s1, s47, 31 // (op 1137) 
s_cmp_eq_i32 s14, 0 // (op 1137) 
s_mov_b32 s18, scc // (op 1137) 
s_cmp_eq_i32 s18, 0 // (op 1137) 
s_cselect_b32 s19, 0, -1 // (op 1137) 

s_and_b32 s18, s15, 31 // (op 1137) 
s_lshl_b32 s22, 1, s18 // (op 1137) 
s_add_i32 s23, s22, s19 // (op 1137) 


s_and_b32 s19, s1, s23 // (op 1137) 


s_add_i32 s1, s47, s19 // (op 1137) 


s_ashr_i32 s19, s1, s18 // (op 1137) 


s_lshl_b32 s1, s15, 24 // (op 1137) 
s_ashr_i32 s18, s1, 31 // (op 1137) 

s_xor_b32 s1, s19, s18 // (op 1137) 

s_sub_i32 s19, s1, s18 // (op 1137) 


s_add_u32 s22, s4, 16 // (op 1137) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1137) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1137) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1137) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1137) 

s_mul_i32 s1, s19, s22 // (op 1137) 

s_sub_i32 s47, s0, s1 // (op 1137) 


s_mul_hi_i32 s0, s16, s9 // (op 1137) 
s_add_i32 s1, s0, s16 // (op 1137) 

s_ashr_i32 s0, s1, 31 // (op 1137) 
s_cmp_eq_i32 s9, 0 // (op 1137) 
s_mov_b32 s18, scc // (op 1137) 
s_cmp_eq_i32 s18, 0 // (op 1137) 
s_cselect_b32 s19, 0, -1 // (op 1137) 

s_and_b32 s18, s45, 31 // (op 1137) 
s_lshl_b32 s24, 1, s18 // (op 1137) 
s_add_i32 s25, s24, s19 // (op 1137) 


s_and_b32 s19, s0, s25 // (op 1137) 


s_add_i32 s0, s1, s19 // (op 1137) 


s_ashr_i32 s1, s0, s18 // (op 1137) 


s_lshl_b32 s0, s45, 24 // (op 1137) 
s_ashr_i32 s18, s0, 31 // (op 1137) 

s_xor_b32 s0, s1, s18 // (op 1137) 

s_sub_i32 s1, s0, s18 // (op 1137) 


s_mul_i32 s0, s11, s22 // (op 1137) 

s_mul_i32 s18, s8, s0 // (op 1137) 

s_mul_i32 s0, s1, s18 // (op 1137) 

s_sub_u32 s18, s16, s0 // (op 1137) 

s_mul_hi_i32 s0, s18, s12 // (op 1137) 
s_add_i32 s19, s0, s18 // (op 1137) 


s_ashr_i32 s0, s19, 31 // (op 1137) 
s_cmp_eq_i32 s12, 0 // (op 1137) 
s_mov_b32 s18, scc // (op 1137) 
s_cmp_eq_i32 s18, 0 // (op 1137) 
s_cselect_b32 s22, 0, -1 // (op 1137) 

s_and_b32 s18, s44, 31 // (op 1137) 
s_lshl_b32 s23, 1, s18 // (op 1137) 
s_add_i32 s24, s23, s22 // (op 1137) 


s_and_b32 s25, s0, s24 // (op 1137) 


s_add_i32 s0, s19, s25 // (op 1137) 


s_ashr_i32 s19, s0, s18 // (op 1137) 


s_lshl_b32 s0, s44, 24 // (op 1137) 
s_ashr_i32 s18, s0, 31 // (op 1137) 

s_xor_b32 s0, s19, s18 // (op 1137) 

s_sub_i32 s19, s0, s18 // (op 1137) 


s_cmp_eq_i32 s1, 0 // (op 1137) 
s_mov_b32 s0, scc // (op 1137) 

s_cmp_eq_i32 s0, 0 // (op 1137) 
s_cselect_b32 s1, s19, s47 // (op 1137) 



v_lshl_add_u32 v22, s1, 4, v20 // (op 1137) 


v_mul_lo_u32 v20, v22, s38 // (op 1137) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1137) 
s_lshr_b32 s47, s0, 1 // (op 1137) 

v_lshl_add_u32 v22, s47, 3, v12 // (op 1137) 


v_add_u32 v12, v22, v20 // (op 1137) 


s_mul_i32 s1, s38, s5 // (op 1139) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1139) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1139) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1139) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1139) most significant: sum
s_add_u32 s1, s1, s18 // (op 1139) most significant: sum


s_mov_b32 s68, s40 // (op 1139) reg expression
s_mov_b32 s69, s41 // (op 1139) reg expression
s_mov_b32 s70, s0 // (op 1139) reg expression
s_mov_b32 s71, 131072 // (op 1139) call()

v_and_b32 v20, -8, v1 // (op 1230) 
v_sub_u32 v22, v1, v20 // (op 1230) 

v_lshlrev_b32 v20, 4, v22 // (op 1230) 

v_lshrrev_b32 v22, 3, v1 // (op 1230) 
v_lshl_add_u32 v13, v22, 7, v20 // (op 1230) 


s_nop 0
v_readfirstlane_b32 s47, v13 // (op 1230) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 1287) 
v_sub_u32 v22, v1, v20 // (op 1287) 

v_lshlrev_b32 v20, 4, v22 // (op 1287) 

v_lshrrev_b32 v22, 3, v1 // (op 1287) 
v_lshl_add_u32 v13, v22, 7, v20 // (op 1287) 


s_nop 0
v_readfirstlane_b32 s0, v13 // (op 1287) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -4, v1 // (op 1339) 
v_sub_u32 v22, v1, v20 // (op 1339) 

v_lshrrev_b32 v20, 2, v1 // (op 1339) 
v_lshl_add_u32 v13, v20, 3, v22 // (op 1339) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 316) 
s_and_b32 s1, s18, 4294967294 // (op 316) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 315) 
s_mov_b32 s22, s18 // (op 315) convert

s_cmp_lt_u32 s1, s22 // (op 315) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 // (op 315) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_315 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_315:
s_barrier  // (op 1549) 
s_add_u32 s1, s1, 1 // (op 317) 

s_lshl_b64 s[18:19], s[26:27], 4 // (op 893) 
v_mov_b64 v[14:15], s[18:19] // (op 893) convert


v_mov_b64 v[16:17], v[18:19] // (op 893) 
v_mov_b64 v[20:21], v[16:17] // (op 893) 
s_add_u32 s18, 2048, s47 // (op 893) 
s_mov_b32 m0, s18 // (op 893) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 893) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v14 // (op 893) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 893) most significant half

s_add_u32 m0, m0, 2048 // (op 893) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 893) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v14 // (op 893) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 893) most significant half

s_add_u32 m0, m0, 2048 // (op 893) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 893) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v14 // (op 893) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 893) most significant half

s_add_u32 m0, m0, 2048 // (op 893) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 893) Load value direct to lds




v_add_co_u32 v18, vcc, 128, v18 // (op 931) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 931) most significant half
s_barrier  // (op 1565) 


v_mov_b64 v[14:15], v[26:27] // (op 899) 

s_mov_b32 m0, s0 // (op 899) reg expression
buffer_load_dwordx4 v14, s[60:63], 0 offen offset: 0 lds // (op 899) Load value direct to lds

v_add_co_u32 v26, vcc, 128, v26 // (op 1001) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 1001) most significant half

v_mov_b64 v[14:15], s[34:35] // (op 180) convert

v_mov_b32 v25, v24 // (op 180) 
v_mov_b32 v16, v25 // (op 180) 
buffer_load_ubyte v20, v16, s[64:67], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v16, v16, v14 // (op 180) 
buffer_load_ubyte v21, v16, s[64:67], 0 offen offset: 0 // (op 180) Load value



v_add_u32 v24, 4, v24 // (op 1065) 
s_barrier  // (op 1581) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1555) 






ds_read_b128 v[28:31], v7 offset:2048 // (op 169) Load local data 
ds_read_b128 v[32:35], v7 offset:2112 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v13, v20 offset:20480 // (op 178) Store local data 
s_barrier  // (op 1571) 
s_waitcnt vmcnt(0)
ds_write_b8 v13, v21 offset:20484 // (op 178) Store local data 




ds_read_b128 v[36:39], v4 // (op 171) Load local data 
ds_read_b128 v[40:43], v4 offset:64 // (op 171) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1588) 



ds_read_u8 v25, v0 offset:20480 // (op 173) Load local data 


v_mov_b32 v14, v12 // (op 175) 

buffer_load_ubyte v15, v14, s[68:71], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v12, 4, v12 // (op 1129) 



ds_read_b128 v[44:47], v7 offset:6144 // (op 321) Load local data 
ds_read_b128 v[48:51], v7 offset:6208 // (op 321) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:43], a[0:3], v25, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 176) 





ds_read_u8 v14, v0 offset:20608 // (op 325) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v14, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 327) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 315) 
s_mov_b32 s22, s18 // (op 315) convert

s_cmp_lt_u32 s1, s22 // (op 315) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_315 // (op 315) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_315 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_315:





















s_barrier  // (op 1646) 



















v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 




buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 52) Store value

v_accvgpr_read v10, a4 // (op 513) 
v_accvgpr_read v11, a5 // (op 513) 
v_accvgpr_read v12, a6 // (op 513) 
v_accvgpr_read v13, a7 // (op 513) 
v_mul_f32 v14, v9, v10 // (op 513) 
v_mul_f32 v15, v9, v11 // (op 513) 
v_mul_f32 v16, v9, v12 // (op 513) 
v_mul_f32 v17, v9, v13 // (op 513) 




buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen offset:128 // (op 515) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen offset:128 // (op 543) Load value
v_accvgpr_read v14, a4 // (op 540) 
v_accvgpr_read v15, a5 // (op 540) 
v_accvgpr_read v16, a6 // (op 540) 
v_accvgpr_read v17, a7 // (op 540) 
v_mul_f32 v18, v9, v14 // (op 540) 
v_mul_f32 v19, v9, v15 // (op 540) 
v_mul_f32 v20, v9, v16 // (op 540) 
v_mul_f32 v21, v9, v17 // (op 540) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 544) 
v_mul_f32 v15, v8, v11 // (op 544) 
v_mul_f32 v16, v8, v12 // (op 544) 
v_mul_f32 v17, v8, v13 // (op 544) 

v_add_f32 v10, v14, v18 // (op 541) 
v_add_f32 v11, v15, v19 // (op 541) 
v_add_f32 v12, v16, v20 // (op 541) 
v_add_f32 v13, v17, v21 // (op 541) 





buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen offset:128 // (op 542) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen // (op 135) Load value
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





buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_, .LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__end-RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 64
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
    DataTypeB: B8
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
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: SizeFree0 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int64, semantic: StrideD0 },
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
    macrotile: [64, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 62
    .agpr_count: 8
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 24
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
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

