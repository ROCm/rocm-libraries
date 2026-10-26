

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1188) 
s_mul_hi_i32 s1, s16, s10 // (op 1375) 
v_lshrrev_b32 v0, 7, v1 // (op 1244) 
v_and_b32 v3, 15, v1 // (op 1244) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1244) 


v_and_b32 v0, 48, v1 // (op 1244) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1244) 


v_lshrrev_b32 v0, 7, v1 // (op 1416) 
v_and_b32 v4, 15, v1 // (op 1416) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1416) 


v_and_b32 v0, 48, v1 // (op 1416) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1416) 


v_lshrrev_b32 v0, 7, v1 // (op 1188) 
s_add_i32 s18, s0, s16 // (op 1188) 

v_and_b32 v5, 63, v1 // (op 1188) 
s_ashr_i32 s0, s18, 31 // (op 1188) 
v_lshrrev_b32 v6, 4, v5 // (op 1188) 
s_cmp_eq_i32 s10, 0 // (op 1188) 
s_mov_b32 s19, scc // (op 1188) 
v_lshlrev_b32 v7, 2, v6 // (op 1188) 

s_cmp_eq_i32 s19, 0 // (op 1188) 
s_cselect_b32 s49, 0, -1 // (op 1188) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1188) 


s_and_b32 s19, s13, 31 // (op 1188) 
v_lshrrev_b32 v0, 2, v5 // (op 1188) 

v_and_b32 v5, 3, v0 // (op 1188) 

v_lshrrev_b32 v0, 6, v1 // (op 1301) 
v_and_b32 v7, 1, v0 // (op 1301) 

v_and_b32 v0, 15, v1 // (op 1301) 
v_lshl_add_u32 v8, v7, 4, v0 // (op 1301) 


v_and_b32 v0, 48, v1 // (op 1301) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 1301) 


v_lshrrev_b32 v0, 6, v1 // (op 1359) 
v_and_b32 v8, 1, v0 // (op 1359) 

v_and_b32 v0, 15, v1 // (op 1359) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 1359) 


v_and_b32 v0, 63, v1 // (op 1359) 
v_lshrrev_b32 v8, 4, v0 // (op 1359) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 1359) 


v_lshrrev_b32 v8, 7, v1 // (op 1375) 
v_and_b32 v9, 63, v1 // (op 1375) 
v_lshrrev_b32 v10, 6, v1 // (op 1460) 
v_and_b32 v11, 1, v10 // (op 1460) 

v_and_b32 v10, 15, v1 // (op 1460) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 1460) 


v_and_b32 v10, 48, v1 // (op 1460) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 1460) 


v_lshrrev_b32 v10, 6, v1 // (op 1489) 
v_and_b32 v12, 1, v10 // (op 1489) 

v_and_b32 v10, 15, v1 // (op 1489) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 1489) 


v_and_b32 v10, 63, v1 // (op 1489) 
v_lshrrev_b32 v12, 4, v10 // (op 1489) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 1489) 


s_lshl_b32 s50, 1, s19 // (op 1188) 
v_and_b32 v12, 3, v1 // (op 1188) 
s_add_i32 s51, s50, s49 // (op 1188) 


s_and_b32 s49, s0, s51 // (op 1188) 


s_add_i32 s0, s1, s16 // (op 1375) 

s_ashr_i32 s1, s0, 31 // (op 1375) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1188) 


s_add_i32 s50, s18, s49 // (op 1188) 


v_lshrrev_b32 v5, 6, v1 // (op 1188) 
s_ashr_i32 s18, s50, s19 // (op 1188) 


v_and_b32 v12, 1, v5 // (op 1188) 

s_lshl_b32 s19, s13, 24 // (op 1188) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1188) 


s_ashr_i32 s49, s19, 31 // (op 1188) 

s_xor_b32 s19, s18, s49 // (op 1188) 

s_sub_i32 s18, s19, s49 // (op 1188) 


s_mul_hi_i32 s19, s18, s14 // (op 1188) 
s_add_i32 s49, s19, s18 // (op 1188) 

s_ashr_i32 s19, s49, 31 // (op 1188) 
s_cmp_eq_i32 s14, 0 // (op 1188) 
s_mov_b32 s50, scc // (op 1188) 
s_cmp_eq_i32 s50, 0 // (op 1188) 
s_cselect_b32 s51, 0, -1 // (op 1188) 

s_and_b32 s50, s15, 31 // (op 1188) 
v_lshrrev_b32 v12, 4, v9 // (op 1375) 
s_cmp_eq_i32 s10, 0 // (op 1375) 
s_mov_b32 s52, scc // (op 1375) 
v_lshlrev_b32 v13, 2, v12 // (op 1375) 

s_cmp_eq_i32 s52, 0 // (op 1375) 
s_cselect_b32 s53, 0, -1 // (op 1375) 

s_lshl_b32 s52, 1, s50 // (op 1188) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1375) 


v_lshrrev_b32 v8, 2, v9 // (op 1375) 

s_add_i32 s54, s52, s51 // (op 1188) 


s_and_b32 s51, s19, s54 // (op 1188) 


s_add_i32 s19, s49, s51 // (op 1188) 


s_ashr_i32 s49, s19, s50 // (op 1188) 


s_lshl_b32 s19, s15, 24 // (op 1188) 
s_ashr_i32 s50, s19, 31 // (op 1188) 

s_xor_b32 s19, s49, s50 // (op 1188) 

s_sub_i32 s49, s19, s50 // (op 1188) 


s_mul_i32 s19, s18, s11 // (op 1188) 
s_sub_u32 s50, s16, s19 // (op 1188) 

s_mul_i32 s19, s49, s11 // (op 1188) 
s_add_u32 s19, s19, s50 // (op 1188) 

s_mul_hi_i32 s50, s16, s9 // (op 1188) 
s_add_i32 s51, s50, s16 // (op 1188) 

s_ashr_i32 s50, s51, 31 // (op 1188) 
s_cmp_eq_i32 s9, 0 // (op 1188) 
s_mov_b32 s52, scc // (op 1188) 
s_cmp_eq_i32 s52, 0 // (op 1188) 
s_cselect_b32 s54, 0, -1 // (op 1188) 

s_and_b32 s52, s13, 31 // (op 1375) 
v_and_b32 v9, 3, v8 // (op 1375) 

s_lshl_b32 s55, 1, s52 // (op 1375) 
v_and_b32 v8, 3, v1 // (op 1375) 
s_add_i32 s56, s55, s53 // (op 1375) 


s_and_b32 s53, s1, s56 // (op 1375) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1375) 


s_add_i32 s1, s0, s53 // (op 1375) 


v_lshrrev_b32 v8, 6, v1 // (op 1375) 
s_ashr_i32 s0, s1, s52 // (op 1375) 


v_and_b32 v9, 1, v8 // (op 1375) 

s_lshl_b32 s1, s13, 24 // (op 1375) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1375) 


s_ashr_i32 s52, s1, 31 // (op 1375) 

s_xor_b32 s1, s0, s52 // (op 1375) 

s_sub_i32 s0, s1, s52 // (op 1375) 


s_mul_hi_i32 s1, s0, s14 // (op 1375) 
s_add_i32 s52, s1, s0 // (op 1375) 

s_ashr_i32 s1, s52, 31 // (op 1375) 
s_cmp_eq_i32 s14, 0 // (op 1375) 
s_mov_b32 s53, scc // (op 1375) 
s_cmp_eq_i32 s53, 0 // (op 1375) 
s_cselect_b32 s55, 0, -1 // (op 1375) 

s_and_b32 s53, s15, 31 // (op 1375) 
s_lshl_b32 s56, 1, s53 // (op 1375) 
s_add_i32 s57, s56, s55 // (op 1375) 


s_and_b32 s55, s1, s57 // (op 1375) 


s_add_i32 s1, s52, s55 // (op 1375) 


s_ashr_i32 s52, s1, s53 // (op 1375) 


s_lshl_b32 s1, s15, 24 // (op 1375) 
s_ashr_i32 s53, s1, 31 // (op 1375) 

s_xor_b32 s1, s52, s53 // (op 1375) 

s_sub_i32 s52, s1, s53 // (op 1375) 


s_mul_i32 s1, s0, s11 // (op 1375) 
s_sub_u32 s53, s16, s1 // (op 1375) 

s_mul_i32 s1, s52, s11 // (op 1375) 
s_add_u32 s1, s1, s53 // (op 1375) 

s_mul_hi_i32 s53, s16, s9 // (op 1375) 
s_add_i32 s55, s53, s16 // (op 1375) 

s_ashr_i32 s53, s55, 31 // (op 1375) 
s_cmp_eq_i32 s9, 0 // (op 1375) 
s_mov_b32 s56, scc // (op 1375) 
s_cmp_eq_i32 s56, 0 // (op 1375) 
s_cselect_b32 s57, 0, -1 // (op 1375) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 1188) 
s_lshl_b32 s58, 1, s56 // (op 1188) 
s_add_i32 s59, s58, s54 // (op 1188) 


s_and_b32 s54, s50, s59 // (op 1188) 


s_add_i32 s50, s51, s54 // (op 1188) 


s_ashr_i32 s51, s50, s56 // (op 1188) 


s_lshl_b32 s50, s45, 24 // (op 1188) 
s_ashr_i32 s54, s50, 31 // (op 1188) 

s_xor_b32 s50, s51, s54 // (op 1188) 

s_sub_i32 s51, s50, s54 // (op 1188) 


s_and_b32 s50, s45, 31 // (op 1375) 
s_lshl_b32 s54, 1, s50 // (op 1375) 
s_add_i32 s56, s54, s57 // (op 1375) 


s_and_b32 s54, s53, s56 // (op 1375) 


s_add_i32 s53, s55, s54 // (op 1375) 


s_ashr_i32 s54, s53, s50 // (op 1375) 


s_lshl_b32 s50, s45, 24 // (op 1375) 
s_ashr_i32 s53, s50, 31 // (op 1375) 

s_xor_b32 s50, s54, s53 // (op 1375) 

s_sub_i32 s54, s50, s53 // (op 1375) 


s_add_u32 s56, s6, 64 // (op 1188) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1188) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1188) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1188) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1188) 

s_mul_i32 s50, s11, s56 // (op 1188) 
s_mul_i32 s53, s8, s50 // (op 1188) 

s_mul_i32 s50, s51, s53 // (op 1188) 

s_sub_u32 s53, s16, s50 // (op 1188) 

s_mov_b32 s50, s53 // (op 1188) 
s_mul_hi_i32 s55, s50, s12 // (op 1188) 
s_add_i32 s58, s55, s50 // (op 1188) 


s_ashr_i32 s50, s58, 31 // (op 1188) 
s_cmp_eq_i32 s12, 0 // (op 1188) 
s_mov_b32 s55, scc // (op 1188) 
s_cmp_eq_i32 s55, 0 // (op 1188) 
s_cselect_b32 s59, 0, -1 // (op 1188) 

s_and_b32 s55, s44, 31 // (op 1188) 
s_lshl_b32 s60, 1, s55 // (op 1188) 
s_add_i32 s61, s60, s59 // (op 1188) 


s_and_b32 s59, s50, s61 // (op 1188) 


s_add_i32 s50, s58, s59 // (op 1188) 


s_ashr_i32 s58, s50, s55 // (op 1188) 


s_lshl_b32 s50, s44, 24 // (op 1188) 
s_ashr_i32 s55, s50, 31 // (op 1188) 

s_xor_b32 s50, s58, s55 // (op 1188) 

s_sub_i32 s58, s50, s55 // (op 1188) 


s_mul_i32 s50, s46, s11 // (op 1188) 
s_add_u32 s60, s20, 32 // (op 1188) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 1188) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 1188) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 1188) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 5 // (op 1188) 

s_sub_i32 s55, s60, s50 // (op 1188) 

s_mul_i32 s59, s58, s55 // (op 1188) 

s_sub_u32 s55, s53, s59 // (op 1188) 


s_add_u32 s53, s55, s50 // (op 1188) 


s_cmp_eq_i32 s51, 0 // (op 1188) 
s_mov_b32 s50, scc // (op 1188) 

s_cmp_eq_i32 s50, 0 // (op 1188) 
s_cselect_b32 s51, s53, s19 // (op 1188) 


s_mul_i32 s19, s49, s56 // (op 1188) 


v_lshl_add_u32 v9, s51, 5, v6 // (op 1188) 


s_sub_i32 s49, s18, s19 // (op 1188) 


s_cmp_eq_i32 s50, 0 // (op 1188) 
s_cselect_b32 s51, s58, s49 // (op 1188) 



v_lshl_add_u32 v6, s51, 6, v5 // (op 1188) 


v_mul_lo_u32 v5, v6, s2 // (op 1188) 

v_add_lshl_u32 v6, v9, v5, 2 // (op 1188) 


s_add_u32 s18, s6, 64 // (op 1375) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1375) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 1375) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 1375) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 6 // (op 1375) 

s_mul_i32 s53, s11, s18 // (op 1375) 
s_mul_i32 s49, s8, s53 // (op 1375) 

s_mul_i32 s53, s54, s49 // (op 1375) 

s_sub_u32 s49, s16, s53 // (op 1375) 

s_mov_b32 s53, s49 // (op 1375) 
s_mul_hi_i32 s50, s53, s12 // (op 1375) 
s_add_i32 s51, s50, s53 // (op 1375) 


s_ashr_i32 s50, s51, 31 // (op 1375) 
s_cmp_eq_i32 s12, 0 // (op 1375) 
s_mov_b32 s53, scc // (op 1375) 
s_cmp_eq_i32 s53, 0 // (op 1375) 
s_cselect_b32 s55, 0, -1 // (op 1375) 

s_and_b32 s53, s44, 31 // (op 1375) 
s_lshl_b32 s56, 1, s53 // (op 1375) 
s_add_i32 s57, s56, s55 // (op 1375) 


s_and_b32 s55, s50, s57 // (op 1375) 


s_add_i32 s50, s51, s55 // (op 1375) 


s_ashr_i32 s51, s50, s53 // (op 1375) 


s_lshl_b32 s50, s44, 24 // (op 1375) 
s_ashr_i32 s53, s50, 31 // (op 1375) 

s_xor_b32 s50, s51, s53 // (op 1375) 

s_sub_i32 s51, s50, s53 // (op 1375) 


s_mul_i32 s50, s46, s11 // (op 1375) 
s_add_u32 s56, s20, 32 // (op 1375) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 1375) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1375) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1375) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 1375) 

s_sub_i32 s53, s56, s50 // (op 1375) 

s_mul_i32 s55, s51, s53 // (op 1375) 

s_sub_u32 s53, s49, s55 // (op 1375) 


s_add_u32 s49, s53, s50 // (op 1375) 


s_cmp_eq_i32 s54, 0 // (op 1375) 
s_mov_b32 s50, scc // (op 1375) 

s_cmp_eq_i32 s50, 0 // (op 1375) 
s_cselect_b32 s53, s49, s1 // (op 1375) 


s_mul_i32 s1, s52, s18 // (op 1375) 


v_lshl_add_u32 v5, s53, 5, v12 // (op 1375) 


s_sub_i32 s49, s0, s1 // (op 1375) 


s_cmp_eq_i32 s50, 0 // (op 1375) 
s_cselect_b32 s0, s51, s49 // (op 1375) 



v_lshl_add_u32 v9, s0, 6, v8 // (op 1375) 


v_mul_lo_u32 v8, v9, s4 // (op 1375) 

v_add_lshl_u32 v9, v5, v8, 2 // (op 1375) 


s_mul_i32 s1, s2, s7 // (op 1190) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1190) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1190) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1190) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1190) most significant: sum
s_add_u32 s1, s1, s19 // (op 1190) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1190) 

s_mul_i32 s1, s4, s7 // (op 1377) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 1377) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1377) least significant
s_mul_hi_u32 s50, s4, s6 // (op 1377) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 1377) most significant: sum
s_add_u32 s1, s1, s50 // (op 1377) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 1377) 

s_mov_b32 s52, s22 // (op 1190) reg expression
s_mov_b32 s53, s23 // (op 1190) reg expression
s_mov_b32 s54, s18 // (op 1190) reg expression
s_mov_b32 s55, 131072 // (op 1190) call()

s_mov_b32 s56, s24 // (op 1377) reg expression

s_mov_b32 s57, s25 // (op 1377) reg expression
s_mov_b32 s58, s50 // (op 1377) reg expression
s_mov_b32 s59, 131072 // (op 1377) call()


v_mov_b32 v5, s47 // (op 2) Move value
v_mov_b32 v8, s48 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 306) call()
v_accvgpr_write a5, 0.00000 // (op 306) call()
v_accvgpr_write a6, 0.00000 // (op 306) call()
v_accvgpr_write a7, 0.00000 // (op 306) call()
s_mul_hi_i32 s0, s16, s10 // (op 955) 
v_lshrrev_b32 v12, 3, v1 // (op 955) 
s_add_i32 s1, s0, s16 // (op 955) 

v_and_b32 v13, -8, v1 // (op 955) 
s_ashr_i32 s0, s1, 31 // (op 955) 
v_sub_u32 v14, v1, v13 // (op 955) 

s_cmp_eq_i32 s10, 0 // (op 955) 
s_mov_b32 s18, scc // (op 955) 
v_lshlrev_b32 v13, 4, v14 // (op 955) 

s_cmp_eq_i32 s18, 0 // (op 955) 
s_cselect_b32 s19, 0, -1 // (op 955) 

v_mov_b32 v15, 0 // (op 955) convert
v_mov_b32 v14, v13 // (op 955) convert

s_and_b32 s18, s13, 31 // (op 955) 
s_lshl_b32 s22, 1, s18 // (op 955) 
s_add_i32 s23, s22, s19 // (op 955) 


s_and_b32 s19, s0, s23 // (op 955) 


s_add_i32 s0, s1, s19 // (op 955) 


s_ashr_i32 s1, s0, s18 // (op 955) 


s_lshl_b32 s0, s13, 24 // (op 955) 
s_ashr_i32 s18, s0, 31 // (op 955) 

s_xor_b32 s0, s1, s18 // (op 955) 

s_sub_i32 s1, s0, s18 // (op 955) 


s_mul_hi_i32 s0, s1, s14 // (op 955) 
s_add_i32 s18, s0, s1 // (op 955) 

s_ashr_i32 s0, s18, 31 // (op 955) 
s_cmp_eq_i32 s14, 0 // (op 955) 
s_mov_b32 s19, scc // (op 955) 
s_cmp_eq_i32 s19, 0 // (op 955) 
s_cselect_b32 s22, 0, -1 // (op 955) 

s_and_b32 s19, s15, 31 // (op 955) 
s_lshl_b32 s23, 1, s19 // (op 955) 
s_add_i32 s24, s23, s22 // (op 955) 


s_and_b32 s25, s0, s24 // (op 955) 


s_add_i32 s0, s18, s25 // (op 955) 


s_ashr_i32 s18, s0, s19 // (op 955) 


s_lshl_b32 s0, s15, 24 // (op 955) 
s_ashr_i32 s19, s0, 31 // (op 955) 

s_xor_b32 s0, s18, s19 // (op 955) 

s_sub_i32 s18, s0, s19 // (op 955) 


s_mul_i32 s0, s1, s11 // (op 955) 

s_sub_u32 s1, s16, s0 // (op 955) 

s_mul_i32 s0, s18, s11 // (op 955) 
s_add_u32 s0, s0, s1 // (op 955) 


s_mul_hi_i32 s1, s16, s9 // (op 955) 
s_add_i32 s18, s1, s16 // (op 955) 

s_ashr_i32 s1, s18, 31 // (op 955) 
s_cmp_eq_i32 s9, 0 // (op 955) 
s_mov_b32 s19, scc // (op 955) 
s_cmp_eq_i32 s19, 0 // (op 955) 
s_cselect_b32 s22, 0, -1 // (op 955) 

s_and_b32 s19, s45, 31 // (op 955) 
s_lshl_b32 s23, 1, s19 // (op 955) 
s_add_i32 s24, s23, s22 // (op 955) 


s_and_b32 s25, s1, s24 // (op 955) 


s_add_i32 s1, s18, s25 // (op 955) 


s_ashr_i32 s18, s1, s19 // (op 955) 


s_lshl_b32 s1, s45, 24 // (op 955) 
s_ashr_i32 s19, s1, 31 // (op 955) 

s_xor_b32 s1, s18, s19 // (op 955) 

s_sub_i32 s18, s1, s19 // (op 955) 


s_add_u32 s22, s6, 64 // (op 955) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 955) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 955) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 955) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 955) 

s_mul_i32 s1, s11, s22 // (op 955) 

s_mul_i32 s19, s8, s1 // (op 955) 

s_mul_i32 s1, s18, s19 // (op 955) 

s_sub_u32 s19, s16, s1 // (op 955) 

s_mov_b32 s1, s19 // (op 955) 
s_mul_hi_i32 s22, s1, s12 // (op 955) 
s_add_i32 s23, s22, s1 // (op 955) 


s_ashr_i32 s1, s23, 31 // (op 955) 
s_cmp_eq_i32 s12, 0 // (op 955) 
s_mov_b32 s22, scc // (op 955) 
s_cmp_eq_i32 s22, 0 // (op 955) 
s_cselect_b32 s24, 0, -1 // (op 955) 

s_and_b32 s22, s44, 31 // (op 955) 
s_lshl_b32 s25, 1, s22 // (op 955) 
s_add_i32 s47, s25, s24 // (op 955) 


s_and_b32 s24, s1, s47 // (op 955) 


s_add_i32 s1, s23, s24 // (op 955) 


s_ashr_i32 s23, s1, s22 // (op 955) 


s_lshl_b32 s1, s44, 24 // (op 955) 
s_ashr_i32 s22, s1, 31 // (op 955) 

s_xor_b32 s1, s23, s22 // (op 955) 

s_sub_i32 s23, s1, s22 // (op 955) 


s_mul_i32 s1, s46, s11 // (op 955) 
s_add_u32 s24, s20, 32 // (op 955) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 955) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 955) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 955) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 955) 

s_sub_i32 s22, s24, s1 // (op 955) 

s_mul_i32 s24, s23, s22 // (op 955) 


s_sub_u32 s25, s19, s24 // (op 955) 


s_add_u32 s19, s25, s1 // (op 955) 


s_cmp_eq_i32 s18, 0 // (op 955) 
s_mov_b32 s1, scc // (op 955) 

s_cmp_eq_i32 s1, 0 // (op 955) 
s_cselect_b32 s18, s19, s0 // (op 955) 



v_lshl_add_u32 v13, s18, 5, v12 // (op 955) 


v_mul_lo_u32 v17, v13, s27 // (op 955) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 955) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 955) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 955) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 955) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 955) most significant half



s_mul_i32 s1, s26, s21 // (op 957) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 957) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 957) least significant
s_mul_hi_u32 s19, s26, s20 // (op 957) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 957) most significant: sum
s_add_u32 s1, s1, s19 // (op 957) most significant: sum


s_mov_b32 s48, s28 // (op 957) reg expression
s_mov_b32 s49, s29 // (op 957) reg expression
s_mov_b32 s50, s0 // (op 957) reg expression
s_mov_b32 s51, 131072 // (op 957) call()

s_mul_hi_i32 s47, s16, s10 // (op 974) 
v_lshrrev_b32 v12, 3, v1 // (op 974) 
s_add_i32 s0, s47, s16 // (op 974) 

v_and_b32 v13, -8, v1 // (op 974) 
s_ashr_i32 s1, s0, 31 // (op 974) 
v_sub_u32 v14, v1, v13 // (op 974) 

s_cmp_eq_i32 s10, 0 // (op 974) 
s_mov_b32 s47, scc // (op 974) 
v_lshlrev_b32 v13, 4, v14 // (op 974) 

s_cmp_eq_i32 s47, 0 // (op 974) 
s_cselect_b32 s18, 0, -1 // (op 974) 

v_mov_b32 v15, 0 // (op 974) convert
v_mov_b32 v14, v13 // (op 974) convert

s_and_b32 s19, s13, 31 // (op 974) 
s_lshl_b32 s47, 1, s19 // (op 974) 
s_add_i32 s22, s47, s18 // (op 974) 


s_and_b32 s18, s1, s22 // (op 974) 


s_add_i32 s1, s0, s18 // (op 974) 


s_ashr_i32 s0, s1, s19 // (op 974) 


s_lshl_b32 s1, s13, 24 // (op 974) 
s_ashr_i32 s47, s1, 31 // (op 974) 

s_xor_b32 s1, s0, s47 // (op 974) 

s_sub_i32 s0, s1, s47 // (op 974) 


s_mul_hi_i32 s1, s0, s14 // (op 974) 
s_add_i32 s47, s1, s0 // (op 974) 

s_ashr_i32 s1, s47, 31 // (op 974) 
s_cmp_eq_i32 s14, 0 // (op 974) 
s_mov_b32 s18, scc // (op 974) 
s_cmp_eq_i32 s18, 0 // (op 974) 
s_cselect_b32 s19, 0, -1 // (op 974) 

s_and_b32 s18, s15, 31 // (op 974) 
s_lshl_b32 s22, 1, s18 // (op 974) 
s_add_i32 s23, s22, s19 // (op 974) 


s_and_b32 s19, s1, s23 // (op 974) 


s_add_i32 s1, s47, s19 // (op 974) 


s_ashr_i32 s19, s1, s18 // (op 974) 


s_lshl_b32 s1, s15, 24 // (op 974) 
s_ashr_i32 s18, s1, 31 // (op 974) 

s_xor_b32 s1, s19, s18 // (op 974) 

s_sub_i32 s19, s1, s18 // (op 974) 


s_mul_i32 s1, s0, s11 // (op 974) 

s_sub_u32 s0, s16, s1 // (op 974) 

s_mul_i32 s1, s19, s11 // (op 974) 
s_add_u32 s1, s1, s0 // (op 974) 


s_mul_hi_i32 s0, s16, s9 // (op 974) 
s_add_i32 s47, s0, s16 // (op 974) 

s_ashr_i32 s0, s47, 31 // (op 974) 
s_cmp_eq_i32 s9, 0 // (op 974) 
s_mov_b32 s18, scc // (op 974) 
s_cmp_eq_i32 s18, 0 // (op 974) 
s_cselect_b32 s19, 0, -1 // (op 974) 

s_and_b32 s18, s45, 31 // (op 974) 
s_lshl_b32 s22, 1, s18 // (op 974) 
s_add_i32 s23, s22, s19 // (op 974) 


s_and_b32 s19, s0, s23 // (op 974) 


s_add_i32 s0, s47, s19 // (op 974) 


s_ashr_i32 s19, s0, s18 // (op 974) 


s_lshl_b32 s0, s45, 24 // (op 974) 
s_ashr_i32 s18, s0, 31 // (op 974) 

s_xor_b32 s0, s19, s18 // (op 974) 

s_sub_i32 s19, s0, s18 // (op 974) 


s_add_u32 s22, s6, 64 // (op 974) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 974) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 974) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 974) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 974) 

s_mul_i32 s0, s11, s22 // (op 974) 

s_mul_i32 s18, s8, s0 // (op 974) 

s_mul_i32 s0, s19, s18 // (op 974) 

s_sub_u32 s18, s16, s0 // (op 974) 

s_mov_b32 s0, s18 // (op 974) 
s_mul_hi_i32 s47, s0, s12 // (op 974) 
s_add_i32 s22, s47, s0 // (op 974) 


s_ashr_i32 s0, s22, 31 // (op 974) 
s_cmp_eq_i32 s12, 0 // (op 974) 
s_mov_b32 s47, scc // (op 974) 
s_cmp_eq_i32 s47, 0 // (op 974) 
s_cselect_b32 s23, 0, -1 // (op 974) 

s_and_b32 s47, s44, 31 // (op 974) 
s_lshl_b32 s24, 1, s47 // (op 974) 
s_add_i32 s25, s24, s23 // (op 974) 


s_and_b32 s23, s0, s25 // (op 974) 


s_add_i32 s0, s22, s23 // (op 974) 


s_ashr_i32 s22, s0, s47 // (op 974) 


s_lshl_b32 s0, s44, 24 // (op 974) 
s_ashr_i32 s47, s0, 31 // (op 974) 

s_xor_b32 s0, s22, s47 // (op 974) 

s_sub_i32 s22, s0, s47 // (op 974) 


s_mul_i32 s0, s46, s11 // (op 974) 
s_add_u32 s24, s20, 32 // (op 974) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 974) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 974) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 974) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 5 // (op 974) 

s_sub_i32 s23, s24, s0 // (op 974) 

s_mul_i32 s47, s22, s23 // (op 974) 


s_sub_u32 s22, s18, s47 // (op 974) 


s_add_u32 s18, s22, s0 // (op 974) 


s_cmp_eq_i32 s19, 0 // (op 974) 
s_mov_b32 s0, scc // (op 974) 

s_cmp_eq_i32 s0, 0 // (op 974) 
s_cselect_b32 s19, s18, s1 // (op 974) 



v_lshl_add_u32 v13, s19, 5, v12 // (op 974) 


v_mul_lo_u32 v19, v13, s27 // (op 974) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 974) most significant: high of low * low
v_mul_lo_u32 v18, v13, s26 // (op 974) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 974) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 974) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 974) most significant half



s_mul_i32 s1, s26, s21 // (op 976) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 976) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 976) least significant
s_mul_hi_u32 s18, s26, s20 // (op 976) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 976) most significant: sum
s_add_u32 s1, s1, s18 // (op 976) most significant: sum


s_mov_b32 s48, s28 // (op 976) reg expression
s_mov_b32 s49, s29 // (op 976) reg expression
s_mov_b32 s50, s0 // (op 976) reg expression
s_mov_b32 s51, 131072 // (op 976) call()

s_mul_hi_i32 s47, s16, s10 // (op 1019) 
v_lshrrev_b32 v12, 3, v1 // (op 1019) 
s_add_i32 s0, s47, s16 // (op 1019) 

v_and_b32 v13, -8, v1 // (op 1019) 
s_ashr_i32 s1, s0, 31 // (op 1019) 
v_sub_u32 v14, v1, v13 // (op 1019) 

s_cmp_eq_i32 s10, 0 // (op 1019) 
s_mov_b32 s47, scc // (op 1019) 
v_lshlrev_b32 v13, 4, v14 // (op 1019) 

s_cmp_eq_i32 s47, 0 // (op 1019) 
s_cselect_b32 s18, 0, -1 // (op 1019) 

v_mov_b32 v15, 0 // (op 1019) convert
v_mov_b32 v14, v13 // (op 1019) convert

s_and_b32 s19, s13, 31 // (op 1019) 
s_lshl_b32 s47, 1, s19 // (op 1019) 
s_add_i32 s22, s47, s18 // (op 1019) 


s_and_b32 s18, s1, s22 // (op 1019) 


s_add_i32 s1, s0, s18 // (op 1019) 


s_ashr_i32 s0, s1, s19 // (op 1019) 


s_lshl_b32 s1, s13, 24 // (op 1019) 
s_ashr_i32 s47, s1, 31 // (op 1019) 

s_xor_b32 s1, s0, s47 // (op 1019) 

s_sub_i32 s0, s1, s47 // (op 1019) 


s_mul_hi_i32 s1, s0, s14 // (op 1019) 
s_add_i32 s47, s1, s0 // (op 1019) 

s_ashr_i32 s1, s47, 31 // (op 1019) 
s_cmp_eq_i32 s14, 0 // (op 1019) 
s_mov_b32 s18, scc // (op 1019) 
s_cmp_eq_i32 s18, 0 // (op 1019) 
s_cselect_b32 s19, 0, -1 // (op 1019) 

s_and_b32 s18, s15, 31 // (op 1019) 
s_lshl_b32 s22, 1, s18 // (op 1019) 
s_add_i32 s23, s22, s19 // (op 1019) 


s_and_b32 s19, s1, s23 // (op 1019) 


s_add_i32 s1, s47, s19 // (op 1019) 


s_ashr_i32 s19, s1, s18 // (op 1019) 


s_lshl_b32 s1, s15, 24 // (op 1019) 
s_ashr_i32 s18, s1, 31 // (op 1019) 

s_xor_b32 s1, s19, s18 // (op 1019) 

s_sub_i32 s19, s1, s18 // (op 1019) 


s_add_u32 s22, s6, 64 // (op 1019) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1019) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1019) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1019) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1019) 

s_mul_i32 s1, s19, s22 // (op 1019) 

s_sub_i32 s47, s0, s1 // (op 1019) 


s_mul_hi_i32 s0, s16, s9 // (op 1019) 
s_add_i32 s1, s0, s16 // (op 1019) 

s_ashr_i32 s0, s1, 31 // (op 1019) 
s_cmp_eq_i32 s9, 0 // (op 1019) 
s_mov_b32 s18, scc // (op 1019) 
s_cmp_eq_i32 s18, 0 // (op 1019) 
s_cselect_b32 s19, 0, -1 // (op 1019) 

s_and_b32 s18, s45, 31 // (op 1019) 
s_lshl_b32 s24, 1, s18 // (op 1019) 
s_add_i32 s25, s24, s19 // (op 1019) 


s_and_b32 s19, s0, s25 // (op 1019) 


s_add_i32 s0, s1, s19 // (op 1019) 


s_ashr_i32 s1, s0, s18 // (op 1019) 


s_lshl_b32 s0, s45, 24 // (op 1019) 
s_ashr_i32 s18, s0, 31 // (op 1019) 

s_xor_b32 s0, s1, s18 // (op 1019) 

s_sub_i32 s1, s0, s18 // (op 1019) 


s_mul_i32 s0, s11, s22 // (op 1019) 

s_mul_i32 s18, s8, s0 // (op 1019) 

s_mul_i32 s0, s1, s18 // (op 1019) 

s_sub_u32 s18, s16, s0 // (op 1019) 

s_mul_hi_i32 s0, s18, s12 // (op 1019) 
s_add_i32 s19, s0, s18 // (op 1019) 


s_ashr_i32 s0, s19, 31 // (op 1019) 
s_cmp_eq_i32 s12, 0 // (op 1019) 
s_mov_b32 s18, scc // (op 1019) 
s_cmp_eq_i32 s18, 0 // (op 1019) 
s_cselect_b32 s22, 0, -1 // (op 1019) 

s_and_b32 s18, s44, 31 // (op 1019) 
s_lshl_b32 s23, 1, s18 // (op 1019) 
s_add_i32 s24, s23, s22 // (op 1019) 


s_and_b32 s25, s0, s24 // (op 1019) 


s_add_i32 s0, s19, s25 // (op 1019) 


s_ashr_i32 s19, s0, s18 // (op 1019) 


s_lshl_b32 s0, s44, 24 // (op 1019) 
s_ashr_i32 s18, s0, 31 // (op 1019) 

s_xor_b32 s0, s19, s18 // (op 1019) 

s_sub_i32 s19, s0, s18 // (op 1019) 


s_cmp_eq_i32 s1, 0 // (op 1019) 
s_mov_b32 s0, scc // (op 1019) 

s_cmp_eq_i32 s0, 0 // (op 1019) 
s_cselect_b32 s1, s19, s47 // (op 1019) 



v_lshl_add_u32 v13, s1, 6, v12 // (op 1019) 


v_mul_lo_u32 v21, v13, s31 // (op 1019) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 1019) most significant: high of low * low
v_mul_lo_u32 v20, v13, s30 // (op 1019) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1019) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 1019) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 1019) most significant half



s_mul_i32 s1, s30, s7 // (op 1021) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 1021) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 1021) least significant
s_mul_hi_u32 s18, s30, s6 // (op 1021) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1021) most significant: sum
s_add_u32 s1, s1, s18 // (op 1021) most significant: sum


s_mov_b32 s60, s32 // (op 1021) reg expression
s_mov_b32 s61, s33 // (op 1021) reg expression
s_mov_b32 s62, s0 // (op 1021) reg expression
s_mov_b32 s63, 131072 // (op 1021) call()

s_mul_hi_i32 s47, s16, s10 // (op 1038) 
v_lshrrev_b32 v14, 3, v1 // (op 1038) 
s_add_i32 s0, s47, s16 // (op 1038) 

v_and_b32 v15, -8, v1 // (op 1038) 
s_ashr_i32 s1, s0, 31 // (op 1038) 
v_sub_u32 v20, v1, v15 // (op 1038) 

s_cmp_eq_i32 s10, 0 // (op 1038) 
s_mov_b32 s47, scc // (op 1038) 
v_lshlrev_b32 v15, 4, v20 // (op 1038) 

s_cmp_eq_i32 s47, 0 // (op 1038) 
s_cselect_b32 s18, 0, -1 // (op 1038) 

v_mov_b32 v21, 0 // (op 1038) convert
v_mov_b32 v20, v15 // (op 1038) convert

s_and_b32 s19, s13, 31 // (op 1038) 
s_lshl_b32 s47, 1, s19 // (op 1038) 
s_add_i32 s22, s47, s18 // (op 1038) 


s_and_b32 s18, s1, s22 // (op 1038) 


s_add_i32 s1, s0, s18 // (op 1038) 


s_ashr_i32 s0, s1, s19 // (op 1038) 


s_lshl_b32 s1, s13, 24 // (op 1038) 
s_ashr_i32 s47, s1, 31 // (op 1038) 

s_xor_b32 s1, s0, s47 // (op 1038) 

s_sub_i32 s0, s1, s47 // (op 1038) 


s_mul_hi_i32 s1, s0, s14 // (op 1038) 
s_add_i32 s47, s1, s0 // (op 1038) 

s_ashr_i32 s1, s47, 31 // (op 1038) 
s_cmp_eq_i32 s14, 0 // (op 1038) 
s_mov_b32 s18, scc // (op 1038) 
s_cmp_eq_i32 s18, 0 // (op 1038) 
s_cselect_b32 s19, 0, -1 // (op 1038) 

s_and_b32 s18, s15, 31 // (op 1038) 
s_lshl_b32 s22, 1, s18 // (op 1038) 
s_add_i32 s23, s22, s19 // (op 1038) 


s_and_b32 s19, s1, s23 // (op 1038) 


s_add_i32 s1, s47, s19 // (op 1038) 


s_ashr_i32 s19, s1, s18 // (op 1038) 


s_lshl_b32 s1, s15, 24 // (op 1038) 
s_ashr_i32 s18, s1, 31 // (op 1038) 

s_xor_b32 s1, s19, s18 // (op 1038) 

s_sub_i32 s19, s1, s18 // (op 1038) 


s_add_u32 s22, s6, 64 // (op 1038) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1038) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1038) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1038) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1038) 

s_mul_i32 s1, s19, s22 // (op 1038) 

s_sub_i32 s47, s0, s1 // (op 1038) 


s_mul_hi_i32 s0, s16, s9 // (op 1038) 
s_add_i32 s1, s0, s16 // (op 1038) 

s_ashr_i32 s0, s1, 31 // (op 1038) 
s_cmp_eq_i32 s9, 0 // (op 1038) 
s_mov_b32 s18, scc // (op 1038) 
s_cmp_eq_i32 s18, 0 // (op 1038) 
s_cselect_b32 s19, 0, -1 // (op 1038) 

s_and_b32 s18, s45, 31 // (op 1038) 
s_lshl_b32 s24, 1, s18 // (op 1038) 
s_add_i32 s25, s24, s19 // (op 1038) 


s_and_b32 s19, s0, s25 // (op 1038) 


s_add_i32 s0, s1, s19 // (op 1038) 


s_ashr_i32 s1, s0, s18 // (op 1038) 


s_lshl_b32 s0, s45, 24 // (op 1038) 
s_ashr_i32 s18, s0, 31 // (op 1038) 

s_xor_b32 s0, s1, s18 // (op 1038) 

s_sub_i32 s1, s0, s18 // (op 1038) 


s_mul_i32 s0, s11, s22 // (op 1038) 

s_mul_i32 s18, s8, s0 // (op 1038) 

s_mul_i32 s0, s1, s18 // (op 1038) 

s_sub_u32 s18, s16, s0 // (op 1038) 

s_mul_hi_i32 s0, s18, s12 // (op 1038) 
s_add_i32 s19, s0, s18 // (op 1038) 


s_ashr_i32 s0, s19, 31 // (op 1038) 
s_cmp_eq_i32 s12, 0 // (op 1038) 
s_mov_b32 s18, scc // (op 1038) 
s_cmp_eq_i32 s18, 0 // (op 1038) 
s_cselect_b32 s22, 0, -1 // (op 1038) 

s_and_b32 s18, s44, 31 // (op 1038) 
s_lshl_b32 s23, 1, s18 // (op 1038) 
s_add_i32 s24, s23, s22 // (op 1038) 


s_and_b32 s25, s0, s24 // (op 1038) 


s_add_i32 s0, s19, s25 // (op 1038) 


s_ashr_i32 s19, s0, s18 // (op 1038) 


s_lshl_b32 s0, s44, 24 // (op 1038) 
s_ashr_i32 s18, s0, 31 // (op 1038) 

s_xor_b32 s0, s19, s18 // (op 1038) 

s_sub_i32 s19, s0, s18 // (op 1038) 


s_cmp_eq_i32 s1, 0 // (op 1038) 
s_mov_b32 s0, scc // (op 1038) 

s_cmp_eq_i32 s0, 0 // (op 1038) 
s_cselect_b32 s1, s19, s47 // (op 1038) 



v_lshl_add_u32 v15, s1, 6, v14 // (op 1038) 


v_mul_lo_u32 v23, v15, s31 // (op 1038) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 1038) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 1038) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 1038) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 1038) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 1038) most significant half



s_mul_i32 s1, s30, s7 // (op 1040) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 1040) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 1040) least significant
s_mul_hi_u32 s18, s30, s6 // (op 1040) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1040) most significant: sum
s_add_u32 s1, s1, s18 // (op 1040) most significant: sum


s_mov_b32 s60, s32 // (op 1040) reg expression
s_mov_b32 s61, s33 // (op 1040) reg expression
s_mov_b32 s62, s0 // (op 1040) reg expression
s_mov_b32 s63, 131072 // (op 1040) call()

s_mul_hi_i32 s47, s16, s10 // (op 1083) 
v_lshrrev_b32 v20, 7, v1 // (op 1083) 
s_add_i32 s0, s47, s16 // (op 1083) 

v_and_b32 v21, 15, v1 // (op 1083) 
s_ashr_i32 s1, s0, 31 // (op 1083) 
v_lshl_add_u32 v22, v20, 4, v21 // (op 1083) 


s_cmp_eq_i32 s10, 0 // (op 1083) 
s_mov_b32 s47, scc // (op 1083) 
v_and_b32 v20, 63, v1 // (op 1083) 
s_cmp_eq_i32 s47, 0 // (op 1083) 
s_cselect_b32 s18, 0, -1 // (op 1083) 

v_lshrrev_b32 v21, 4, v20 // (op 1083) 

s_and_b32 s19, s13, 31 // (op 1083) 
s_lshl_b32 s47, 1, s19 // (op 1083) 
s_add_i32 s22, s47, s18 // (op 1083) 


s_and_b32 s18, s1, s22 // (op 1083) 


s_add_i32 s1, s0, s18 // (op 1083) 


s_ashr_i32 s0, s1, s19 // (op 1083) 


s_lshl_b32 s1, s13, 24 // (op 1083) 
s_ashr_i32 s47, s1, 31 // (op 1083) 

s_xor_b32 s1, s0, s47 // (op 1083) 

s_sub_i32 s0, s1, s47 // (op 1083) 


s_mul_hi_i32 s1, s0, s14 // (op 1083) 
s_add_i32 s47, s1, s0 // (op 1083) 

s_ashr_i32 s1, s47, 31 // (op 1083) 
s_cmp_eq_i32 s14, 0 // (op 1083) 
s_mov_b32 s18, scc // (op 1083) 
s_cmp_eq_i32 s18, 0 // (op 1083) 
s_cselect_b32 s19, 0, -1 // (op 1083) 

s_and_b32 s18, s15, 31 // (op 1083) 
s_lshl_b32 s22, 1, s18 // (op 1083) 
s_add_i32 s23, s22, s19 // (op 1083) 


s_and_b32 s19, s1, s23 // (op 1083) 


s_add_i32 s1, s47, s19 // (op 1083) 


s_ashr_i32 s19, s1, s18 // (op 1083) 


s_lshl_b32 s1, s15, 24 // (op 1083) 
s_ashr_i32 s18, s1, 31 // (op 1083) 

s_xor_b32 s1, s19, s18 // (op 1083) 

s_sub_i32 s19, s1, s18 // (op 1083) 


s_mul_i32 s1, s0, s11 // (op 1083) 

s_sub_u32 s0, s16, s1 // (op 1083) 

s_mul_i32 s1, s19, s11 // (op 1083) 
s_add_u32 s1, s1, s0 // (op 1083) 


s_mul_hi_i32 s0, s16, s9 // (op 1083) 
s_add_i32 s47, s0, s16 // (op 1083) 

s_ashr_i32 s0, s47, 31 // (op 1083) 
s_cmp_eq_i32 s9, 0 // (op 1083) 
s_mov_b32 s18, scc // (op 1083) 
s_cmp_eq_i32 s18, 0 // (op 1083) 
s_cselect_b32 s19, 0, -1 // (op 1083) 

s_and_b32 s18, s45, 31 // (op 1083) 
s_lshl_b32 s22, 1, s18 // (op 1083) 
s_add_i32 s23, s22, s19 // (op 1083) 


s_and_b32 s19, s0, s23 // (op 1083) 


s_add_i32 s0, s47, s19 // (op 1083) 


s_ashr_i32 s19, s0, s18 // (op 1083) 


s_lshl_b32 s0, s45, 24 // (op 1083) 
s_ashr_i32 s18, s0, 31 // (op 1083) 

s_xor_b32 s0, s19, s18 // (op 1083) 

s_sub_i32 s19, s0, s18 // (op 1083) 


s_add_u32 s22, s6, 64 // (op 1083) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1083) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1083) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1083) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1083) 

s_mul_i32 s0, s11, s22 // (op 1083) 

s_mul_i32 s18, s8, s0 // (op 1083) 

s_mul_i32 s0, s19, s18 // (op 1083) 

s_sub_u32 s18, s16, s0 // (op 1083) 

s_mov_b32 s0, s18 // (op 1083) 
s_mul_hi_i32 s47, s0, s12 // (op 1083) 
s_add_i32 s22, s47, s0 // (op 1083) 


s_ashr_i32 s0, s22, 31 // (op 1083) 
s_cmp_eq_i32 s12, 0 // (op 1083) 
s_mov_b32 s47, scc // (op 1083) 
s_cmp_eq_i32 s47, 0 // (op 1083) 
s_cselect_b32 s23, 0, -1 // (op 1083) 

s_and_b32 s47, s44, 31 // (op 1083) 
s_lshl_b32 s24, 1, s47 // (op 1083) 
s_add_i32 s25, s24, s23 // (op 1083) 


s_and_b32 s23, s0, s25 // (op 1083) 


s_add_i32 s0, s22, s23 // (op 1083) 


s_ashr_i32 s22, s0, s47 // (op 1083) 


s_lshl_b32 s0, s44, 24 // (op 1083) 
s_ashr_i32 s47, s0, 31 // (op 1083) 

s_xor_b32 s0, s22, s47 // (op 1083) 

s_sub_i32 s22, s0, s47 // (op 1083) 


s_mul_i32 s0, s46, s11 // (op 1083) 
s_add_u32 s24, s20, 32 // (op 1083) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1083) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 1083) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 1083) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 5 // (op 1083) 

s_sub_i32 s23, s24, s0 // (op 1083) 

s_mul_i32 s47, s22, s23 // (op 1083) 


s_sub_u32 s22, s18, s47 // (op 1083) 


s_add_u32 s18, s22, s0 // (op 1083) 


s_cmp_eq_i32 s19, 0 // (op 1083) 
s_mov_b32 s0, scc // (op 1083) 

s_cmp_eq_i32 s0, 0 // (op 1083) 
s_cselect_b32 s19, s18, s1 // (op 1083) 



v_lshl_add_u32 v20, s19, 5, v22 // (op 1083) 


v_mul_lo_u32 v22, v20, s34 // (op 1083) 
v_add_u32 v22, v22, v21 // (op 1083) 


s_mul_i32 s1, s34, s21 // (op 1085) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 1085) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 1085) least significant
s_mul_hi_u32 s18, s34, s20 // (op 1085) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1085) most significant: sum
s_add_u32 s1, s1, s18 // (op 1085) most significant: sum


s_mov_b32 s64, s36 // (op 1085) reg expression
s_mov_b32 s65, s37 // (op 1085) reg expression
s_mov_b32 s66, s0 // (op 1085) reg expression
s_mov_b32 s67, 131072 // (op 1085) call()

s_mul_hi_i32 s47, s16, s10 // (op 1102) 
v_lshrrev_b32 v20, 7, v1 // (op 1102) 
s_add_i32 s0, s47, s16 // (op 1102) 

v_and_b32 v21, 15, v1 // (op 1102) 
s_ashr_i32 s1, s0, 31 // (op 1102) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 1102) 


s_cmp_eq_i32 s10, 0 // (op 1102) 
s_mov_b32 s47, scc // (op 1102) 
v_and_b32 v20, 63, v1 // (op 1102) 
s_cmp_eq_i32 s47, 0 // (op 1102) 
s_cselect_b32 s18, 0, -1 // (op 1102) 

v_lshrrev_b32 v21, 4, v20 // (op 1102) 

s_and_b32 s19, s13, 31 // (op 1102) 
s_lshl_b32 s47, 1, s19 // (op 1102) 
s_add_i32 s22, s47, s18 // (op 1102) 


s_and_b32 s18, s1, s22 // (op 1102) 


s_add_i32 s1, s0, s18 // (op 1102) 


s_ashr_i32 s0, s1, s19 // (op 1102) 


s_lshl_b32 s1, s13, 24 // (op 1102) 
s_ashr_i32 s47, s1, 31 // (op 1102) 

s_xor_b32 s1, s0, s47 // (op 1102) 

s_sub_i32 s0, s1, s47 // (op 1102) 


s_mul_hi_i32 s1, s0, s14 // (op 1102) 
s_add_i32 s47, s1, s0 // (op 1102) 

s_ashr_i32 s1, s47, 31 // (op 1102) 
s_cmp_eq_i32 s14, 0 // (op 1102) 
s_mov_b32 s18, scc // (op 1102) 
s_cmp_eq_i32 s18, 0 // (op 1102) 
s_cselect_b32 s19, 0, -1 // (op 1102) 

s_and_b32 s18, s15, 31 // (op 1102) 
s_lshl_b32 s22, 1, s18 // (op 1102) 
s_add_i32 s23, s22, s19 // (op 1102) 


s_and_b32 s19, s1, s23 // (op 1102) 


s_add_i32 s1, s47, s19 // (op 1102) 


s_ashr_i32 s19, s1, s18 // (op 1102) 


s_lshl_b32 s1, s15, 24 // (op 1102) 
s_ashr_i32 s18, s1, 31 // (op 1102) 

s_xor_b32 s1, s19, s18 // (op 1102) 

s_sub_i32 s19, s1, s18 // (op 1102) 


s_mul_i32 s1, s0, s11 // (op 1102) 

s_sub_u32 s0, s16, s1 // (op 1102) 

s_mul_i32 s1, s19, s11 // (op 1102) 
s_add_u32 s1, s1, s0 // (op 1102) 


s_mul_hi_i32 s0, s16, s9 // (op 1102) 
s_add_i32 s47, s0, s16 // (op 1102) 

s_ashr_i32 s0, s47, 31 // (op 1102) 
s_cmp_eq_i32 s9, 0 // (op 1102) 
s_mov_b32 s18, scc // (op 1102) 
s_cmp_eq_i32 s18, 0 // (op 1102) 
s_cselect_b32 s19, 0, -1 // (op 1102) 

s_and_b32 s18, s45, 31 // (op 1102) 
s_lshl_b32 s22, 1, s18 // (op 1102) 
s_add_i32 s23, s22, s19 // (op 1102) 


s_and_b32 s19, s0, s23 // (op 1102) 


s_add_i32 s0, s47, s19 // (op 1102) 


s_ashr_i32 s19, s0, s18 // (op 1102) 


s_lshl_b32 s0, s45, 24 // (op 1102) 
s_ashr_i32 s18, s0, 31 // (op 1102) 

s_xor_b32 s0, s19, s18 // (op 1102) 

s_sub_i32 s19, s0, s18 // (op 1102) 


s_add_u32 s22, s6, 64 // (op 1102) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1102) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1102) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1102) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1102) 

s_mul_i32 s0, s11, s22 // (op 1102) 

s_mul_i32 s18, s8, s0 // (op 1102) 

s_mul_i32 s0, s19, s18 // (op 1102) 

s_sub_u32 s18, s16, s0 // (op 1102) 

s_mov_b32 s0, s18 // (op 1102) 
s_mul_hi_i32 s47, s0, s12 // (op 1102) 
s_add_i32 s22, s47, s0 // (op 1102) 


s_ashr_i32 s0, s22, 31 // (op 1102) 
s_cmp_eq_i32 s12, 0 // (op 1102) 
s_mov_b32 s47, scc // (op 1102) 
s_cmp_eq_i32 s47, 0 // (op 1102) 
s_cselect_b32 s23, 0, -1 // (op 1102) 

s_and_b32 s47, s44, 31 // (op 1102) 
s_lshl_b32 s24, 1, s47 // (op 1102) 
s_add_i32 s25, s24, s23 // (op 1102) 


s_and_b32 s23, s0, s25 // (op 1102) 


s_add_i32 s0, s22, s23 // (op 1102) 


s_ashr_i32 s22, s0, s47 // (op 1102) 


s_lshl_b32 s0, s44, 24 // (op 1102) 
s_ashr_i32 s47, s0, 31 // (op 1102) 

s_xor_b32 s0, s22, s47 // (op 1102) 

s_sub_i32 s22, s0, s47 // (op 1102) 


s_mul_i32 s0, s46, s11 // (op 1102) 
s_add_u32 s24, s20, 32 // (op 1102) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1102) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 1102) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 1102) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 5 // (op 1102) 

s_sub_i32 s23, s24, s0 // (op 1102) 

s_mul_i32 s47, s22, s23 // (op 1102) 


s_sub_u32 s22, s18, s47 // (op 1102) 


s_add_u32 s18, s22, s0 // (op 1102) 


s_cmp_eq_i32 s19, 0 // (op 1102) 
s_mov_b32 s0, scc // (op 1102) 

s_cmp_eq_i32 s0, 0 // (op 1102) 
s_cselect_b32 s19, s18, s1 // (op 1102) 



v_lshl_add_u32 v20, s19, 5, v23 // (op 1102) 


v_mul_lo_u32 v23, v20, s34 // (op 1102) 
v_add_u32 v23, v23, v21 // (op 1102) 


s_mul_i32 s1, s34, s21 // (op 1104) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 1104) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 1104) least significant
s_mul_hi_u32 s18, s34, s20 // (op 1104) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1104) most significant: sum
s_add_u32 s1, s1, s18 // (op 1104) most significant: sum


s_mov_b32 s64, s36 // (op 1104) reg expression
s_mov_b32 s65, s37 // (op 1104) reg expression
s_mov_b32 s66, s0 // (op 1104) reg expression
s_mov_b32 s67, 131072 // (op 1104) call()

s_mul_hi_i32 s47, s16, s10 // (op 1147) 
v_lshrrev_b32 v20, 2, v1 // (op 1147) 
s_add_i32 s0, s47, s16 // (op 1147) 

v_and_b32 v21, -4, v1 // (op 1147) 
s_ashr_i32 s1, s0, 31 // (op 1147) 
v_sub_u32 v24, v1, v21 // (op 1147) 

s_cmp_eq_i32 s10, 0 // (op 1147) 
s_mov_b32 s47, scc // (op 1147) 
s_cmp_eq_i32 s47, 0 // (op 1147) 
s_cselect_b32 s18, 0, -1 // (op 1147) 

s_and_b32 s19, s13, 31 // (op 1147) 
s_lshl_b32 s47, 1, s19 // (op 1147) 
s_add_i32 s22, s47, s18 // (op 1147) 


s_and_b32 s18, s1, s22 // (op 1147) 


s_add_i32 s1, s0, s18 // (op 1147) 


s_ashr_i32 s0, s1, s19 // (op 1147) 


s_lshl_b32 s1, s13, 24 // (op 1147) 
s_ashr_i32 s47, s1, 31 // (op 1147) 

s_xor_b32 s1, s0, s47 // (op 1147) 

s_sub_i32 s0, s1, s47 // (op 1147) 


s_mul_hi_i32 s1, s0, s14 // (op 1147) 
s_add_i32 s47, s1, s0 // (op 1147) 

s_ashr_i32 s1, s47, 31 // (op 1147) 
s_cmp_eq_i32 s14, 0 // (op 1147) 
s_mov_b32 s18, scc // (op 1147) 
s_cmp_eq_i32 s18, 0 // (op 1147) 
s_cselect_b32 s19, 0, -1 // (op 1147) 

s_and_b32 s18, s15, 31 // (op 1147) 
s_lshl_b32 s22, 1, s18 // (op 1147) 
s_add_i32 s23, s22, s19 // (op 1147) 


s_and_b32 s19, s1, s23 // (op 1147) 


s_add_i32 s1, s47, s19 // (op 1147) 


s_ashr_i32 s19, s1, s18 // (op 1147) 


s_lshl_b32 s1, s15, 24 // (op 1147) 
s_ashr_i32 s18, s1, 31 // (op 1147) 

s_xor_b32 s1, s19, s18 // (op 1147) 

s_sub_i32 s19, s1, s18 // (op 1147) 


s_add_u32 s22, s6, 64 // (op 1147) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1147) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1147) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1147) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1147) 

s_mul_i32 s1, s19, s22 // (op 1147) 

s_sub_i32 s47, s0, s1 // (op 1147) 


s_mul_hi_i32 s0, s16, s9 // (op 1147) 
s_add_i32 s1, s0, s16 // (op 1147) 

s_ashr_i32 s0, s1, 31 // (op 1147) 
s_cmp_eq_i32 s9, 0 // (op 1147) 
s_mov_b32 s18, scc // (op 1147) 
s_cmp_eq_i32 s18, 0 // (op 1147) 
s_cselect_b32 s19, 0, -1 // (op 1147) 

s_and_b32 s18, s45, 31 // (op 1147) 
s_lshl_b32 s24, 1, s18 // (op 1147) 
s_add_i32 s25, s24, s19 // (op 1147) 


s_and_b32 s19, s0, s25 // (op 1147) 


s_add_i32 s0, s1, s19 // (op 1147) 


s_ashr_i32 s1, s0, s18 // (op 1147) 


s_lshl_b32 s0, s45, 24 // (op 1147) 
s_ashr_i32 s18, s0, 31 // (op 1147) 

s_xor_b32 s0, s1, s18 // (op 1147) 

s_sub_i32 s1, s0, s18 // (op 1147) 


s_mul_i32 s0, s11, s22 // (op 1147) 

s_mul_i32 s18, s8, s0 // (op 1147) 

s_mul_i32 s0, s1, s18 // (op 1147) 

s_sub_u32 s18, s16, s0 // (op 1147) 

s_mul_hi_i32 s0, s18, s12 // (op 1147) 
s_add_i32 s19, s0, s18 // (op 1147) 


s_ashr_i32 s0, s19, 31 // (op 1147) 
s_cmp_eq_i32 s12, 0 // (op 1147) 
s_mov_b32 s18, scc // (op 1147) 
s_cmp_eq_i32 s18, 0 // (op 1147) 
s_cselect_b32 s22, 0, -1 // (op 1147) 

s_and_b32 s18, s44, 31 // (op 1147) 
s_lshl_b32 s23, 1, s18 // (op 1147) 
s_add_i32 s24, s23, s22 // (op 1147) 


s_and_b32 s25, s0, s24 // (op 1147) 


s_add_i32 s0, s19, s25 // (op 1147) 


s_ashr_i32 s19, s0, s18 // (op 1147) 


s_lshl_b32 s0, s44, 24 // (op 1147) 
s_ashr_i32 s18, s0, 31 // (op 1147) 

s_xor_b32 s0, s19, s18 // (op 1147) 

s_sub_i32 s19, s0, s18 // (op 1147) 


s_cmp_eq_i32 s1, 0 // (op 1147) 
s_mov_b32 s0, scc // (op 1147) 

s_cmp_eq_i32 s0, 0 // (op 1147) 
s_cselect_b32 s1, s19, s47 // (op 1147) 



v_lshl_add_u32 v21, s1, 6, v20 // (op 1147) 


v_mul_lo_u32 v20, v21, s38 // (op 1147) 

v_add_u32 v21, v24, v20 // (op 1147) 


s_mul_i32 s1, s38, s7 // (op 1149) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 1149) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 1149) least significant
s_mul_hi_u32 s18, s38, s6 // (op 1149) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1149) most significant: sum
s_add_u32 s1, s1, s18 // (op 1149) most significant: sum


s_mov_b32 s68, s40 // (op 1149) reg expression
s_mov_b32 s69, s41 // (op 1149) reg expression
s_mov_b32 s70, s0 // (op 1149) reg expression
s_mov_b32 s71, 131072 // (op 1149) call()

s_mul_hi_i32 s47, s16, s10 // (op 1166) 
v_lshrrev_b32 v20, 2, v1 // (op 1166) 
s_add_i32 s0, s47, s16 // (op 1166) 

v_and_b32 v24, -4, v1 // (op 1166) 
s_ashr_i32 s1, s0, 31 // (op 1166) 
v_sub_u32 v25, v1, v24 // (op 1166) 

s_cmp_eq_i32 s10, 0 // (op 1166) 
s_mov_b32 s47, scc // (op 1166) 
s_cmp_eq_i32 s47, 0 // (op 1166) 
s_cselect_b32 s18, 0, -1 // (op 1166) 

s_and_b32 s19, s13, 31 // (op 1166) 
s_lshl_b32 s47, 1, s19 // (op 1166) 
s_add_i32 s22, s47, s18 // (op 1166) 


s_and_b32 s18, s1, s22 // (op 1166) 


s_add_i32 s1, s0, s18 // (op 1166) 


s_ashr_i32 s0, s1, s19 // (op 1166) 


s_lshl_b32 s1, s13, 24 // (op 1166) 
s_ashr_i32 s47, s1, 31 // (op 1166) 

s_xor_b32 s1, s0, s47 // (op 1166) 

s_sub_i32 s0, s1, s47 // (op 1166) 


s_mul_hi_i32 s1, s0, s14 // (op 1166) 
s_add_i32 s47, s1, s0 // (op 1166) 

s_ashr_i32 s1, s47, 31 // (op 1166) 
s_cmp_eq_i32 s14, 0 // (op 1166) 
s_mov_b32 s18, scc // (op 1166) 
s_cmp_eq_i32 s18, 0 // (op 1166) 
s_cselect_b32 s19, 0, -1 // (op 1166) 

s_and_b32 s18, s15, 31 // (op 1166) 
s_lshl_b32 s22, 1, s18 // (op 1166) 
s_add_i32 s23, s22, s19 // (op 1166) 


s_and_b32 s19, s1, s23 // (op 1166) 


s_add_i32 s1, s47, s19 // (op 1166) 


s_ashr_i32 s19, s1, s18 // (op 1166) 


s_lshl_b32 s1, s15, 24 // (op 1166) 
s_ashr_i32 s18, s1, 31 // (op 1166) 

s_xor_b32 s1, s19, s18 // (op 1166) 

s_sub_i32 s19, s1, s18 // (op 1166) 


s_add_u32 s22, s6, 64 // (op 1166) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1166) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1166) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1166) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1166) 

s_mul_i32 s1, s19, s22 // (op 1166) 

s_sub_i32 s47, s0, s1 // (op 1166) 


s_mul_hi_i32 s0, s16, s9 // (op 1166) 
s_add_i32 s1, s0, s16 // (op 1166) 

s_ashr_i32 s0, s1, 31 // (op 1166) 
s_cmp_eq_i32 s9, 0 // (op 1166) 
s_mov_b32 s18, scc // (op 1166) 
s_cmp_eq_i32 s18, 0 // (op 1166) 
s_cselect_b32 s19, 0, -1 // (op 1166) 

s_and_b32 s18, s45, 31 // (op 1166) 
s_lshl_b32 s24, 1, s18 // (op 1166) 
s_add_i32 s25, s24, s19 // (op 1166) 


s_and_b32 s19, s0, s25 // (op 1166) 


s_add_i32 s0, s1, s19 // (op 1166) 


s_ashr_i32 s1, s0, s18 // (op 1166) 


s_lshl_b32 s0, s45, 24 // (op 1166) 
s_ashr_i32 s18, s0, 31 // (op 1166) 

s_xor_b32 s0, s1, s18 // (op 1166) 

s_sub_i32 s1, s0, s18 // (op 1166) 


s_mul_i32 s0, s11, s22 // (op 1166) 

s_mul_i32 s18, s8, s0 // (op 1166) 

s_mul_i32 s0, s1, s18 // (op 1166) 

s_sub_u32 s18, s16, s0 // (op 1166) 

s_mul_hi_i32 s0, s18, s12 // (op 1166) 
s_add_i32 s19, s0, s18 // (op 1166) 


s_ashr_i32 s0, s19, 31 // (op 1166) 
s_cmp_eq_i32 s12, 0 // (op 1166) 
s_mov_b32 s18, scc // (op 1166) 
s_cmp_eq_i32 s18, 0 // (op 1166) 
s_cselect_b32 s22, 0, -1 // (op 1166) 

s_and_b32 s18, s44, 31 // (op 1166) 
s_lshl_b32 s23, 1, s18 // (op 1166) 
s_add_i32 s24, s23, s22 // (op 1166) 


s_and_b32 s25, s0, s24 // (op 1166) 


s_add_i32 s0, s19, s25 // (op 1166) 


s_ashr_i32 s19, s0, s18 // (op 1166) 


s_lshl_b32 s0, s44, 24 // (op 1166) 
s_ashr_i32 s18, s0, 31 // (op 1166) 

s_xor_b32 s0, s19, s18 // (op 1166) 

s_sub_i32 s19, s0, s18 // (op 1166) 


s_cmp_eq_i32 s1, 0 // (op 1166) 
s_mov_b32 s0, scc // (op 1166) 

s_cmp_eq_i32 s0, 0 // (op 1166) 
s_cselect_b32 s1, s19, s47 // (op 1166) 



v_lshl_add_u32 v24, s1, 6, v20 // (op 1166) 


v_mul_lo_u32 v20, v24, s38 // (op 1166) 

v_add_u32 v24, v25, v20 // (op 1166) 


s_mul_i32 s1, s38, s7 // (op 1168) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 1168) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 1168) least significant
s_mul_hi_u32 s18, s38, s6 // (op 1168) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1168) most significant: sum
s_add_u32 s1, s1, s18 // (op 1168) most significant: sum


s_mov_b32 s68, s40 // (op 1168) reg expression
s_mov_b32 s69, s41 // (op 1168) reg expression
s_mov_b32 s70, s0 // (op 1168) reg expression
s_mov_b32 s71, 131072 // (op 1168) call()




buffer_load_ubyte v20, v22, s[64:67], 0 offen offset: 0 // (op 761) Load value
v_and_b32 v25, -8, v1 // (op 1284) 
v_sub_u32 v26, v1, v25 // (op 1284) 

v_lshlrev_b32 v25, 4, v26 // (op 1284) 

v_lshrrev_b32 v26, 3, v1 // (op 1284) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1284) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 1284) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 903) 
v_mov_b64 v[26:27], s[0:1] // (op 903) convert


v_mov_b64 v[28:29], v[12:13] // (op 903) 
s_mov_b32 m0, s47 // (op 903) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 903) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 903) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 903) most significant half

s_add_u32 m0, m0, 4096 // (op 903) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 903) Load value direct to lds






buffer_load_ubyte v25, v21, s[68:71], 0 offen offset: 0 // (op 768) Load value
v_and_b32 v26, -8, v1 // (op 1232) 
v_sub_u32 v27, v1, v26 // (op 1232) 

v_lshlrev_b32 v26, 4, v27 // (op 1232) 

v_lshrrev_b32 v27, 3, v1 // (op 1232) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 1232) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 1232) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s0, 8192, s47 // (op 908) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s0 // (op 908) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 908) Load value direct to lds





buffer_load_ubyte v26, v22, s[64:67], 0 offen offset:4 // (op 775) Load value

v_and_b32 v22, -8, v1 // (op 1443) 
v_sub_u32 v27, v1, v22 // (op 1443) 

v_lshlrev_b32 v22, 4, v27 // (op 1443) 

v_lshrrev_b32 v27, 3, v1 // (op 1443) 
v_lshl_add_u32 v28, v27, 7, v22 // (op 1443) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 1443) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 913) 
v_mov_b64 v[28:29], s[0:1] // (op 913) convert


v_mov_b64 v[30:31], v[12:13] // (op 913) 
s_add_u32 s0, 12288, s47 // (op 913) 
s_mov_b32 m0, s0 // (op 913) reg expression
s_mov_b32 s1, 128 // (op 913) call()
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 913) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 913) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 913) most significant half

s_add_u32 m0, m0, 4096 // (op 913) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 913) Load value direct to lds









buffer_load_ubyte v22, v21, s[68:71], 0 offen offset:4 // (op 782) Load value

v_and_b32 v21, -8, v1 // (op 1404) 
v_sub_u32 v12, v1, v21 // (op 1404) 

v_lshlrev_b32 v13, 4, v12 // (op 1404) 

v_lshrrev_b32 v12, 3, v1 // (op 1404) 
v_lshl_add_u32 v21, v12, 7, v13 // (op 1404) 


s_nop 0
v_readfirstlane_b32 s47, v21 // (op 1404) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s0, 20480, s47 // (op 915) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s0 // (op 915) reg expression
s_mov_b32 s1, 128 // (op 915) call()
buffer_load_dwordx4 v16, s[48:51], s1 offen offset: 0 lds // (op 915) Load value direct to lds




v_and_b32 v21, -4, v1 // (op 1342) 
v_sub_u32 v12, v1, v21 // (op 1342) 

v_and_b32 v13, 4294967292, v1 // (op 1342) 
v_add_u32 v21, v13, v12 // (op 1342) 





s_waitcnt vmcnt(6)
ds_write_b8 v21, v25 offset:24576 // (op 788) Store local data 

s_barrier  // (op 1546) 



ds_read_b128 v[28:31], v7 // (op 806) Load local data 
ds_read_b128 v[32:35], v7 offset:64 // (op 806) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1562) 



ds_read_u8 v21, v0 offset:24576 // (op 816) Load local data 



ds_read_b128 v[36:39], v7 offset:4096 // (op 811) Load local data 
ds_read_b128 v[40:43], v7 offset:4160 // (op 811) Load local data 



ds_read_u8 v27, v0 offset:24704 // (op 821) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1530) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 802) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 802) Load local data 
v_and_b32 v12, -8, v1 // (op 1204) 
v_sub_u32 v13, v1, v12 // (op 1204) 

v_lshlrev_b32 v12, 4, v13 // (op 1204) 

v_lshrrev_b32 v13, 3, v1 // (op 1204) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 1204) 


s_nop 0
v_readfirstlane_b32 s47, v16 // (op 1204) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 1256) 
v_sub_u32 v13, v1, v12 // (op 1256) 

v_lshlrev_b32 v12, 4, v13 // (op 1256) 

v_lshrrev_b32 v13, 3, v1 // (op 1256) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 1256) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 1256) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 1313) 
v_sub_u32 v13, v1, v12 // (op 1313) 

v_and_b32 v12, 4294967292, v1 // (op 1313) 
v_add_u32 v16, v12, v13 // (op 1313) 


v_and_b32 v17, -8, v1 // (op 1389) 
v_sub_u32 v12, v1, v17 // (op 1389) 

v_lshlrev_b32 v13, 4, v12 // (op 1389) 

v_lshrrev_b32 v12, 3, v1 // (op 1389) 
v_lshl_add_u32 v17, v12, 7, v13 // (op 1389) 


s_nop 0
v_readfirstlane_b32 s1, v17 // (op 1389) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 1428) 
v_sub_u32 v12, v1, v17 // (op 1428) 

v_lshlrev_b32 v13, 4, v12 // (op 1428) 

v_lshrrev_b32 v12, 3, v1 // (op 1428) 
v_lshl_add_u32 v17, v12, 7, v13 // (op 1428) 


s_nop 0
v_readfirstlane_b32 s18, v17 // (op 1428) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -4, v1 // (op 1472) 
v_sub_u32 v12, v1, v17 // (op 1472) 

v_and_b32 v13, 4294967292, v1 // (op 1472) 
v_add_u32 v17, v13, v12 // (op 1472) 


s_mov_b32 s19, 0 // (op 308) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 307) 
s_mov_b32 s24, s22 // (op 307) convert

s_and_b32 s25, s24, 4294967294 // (op 307) 

s_cmp_lt_u32 s19, s25 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_307 // (op 307) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_307 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_307:
s_add_u32 s19, s19, 2 // (op 309) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v20, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 82) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v20, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 343) 


v_mov_b32 v12, v23 // (op 75) 

buffer_load_ubyte v20, v12, s[64:67], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 1549) 

s_lshl_b64 s[22:23], s[30:31], 5 // (op 875) 
v_mov_b64 v[52:53], s[22:23] // (op 875) convert


v_mov_b64 v[54:55], v[14:15] // (op 875) 
v_mov_b64 v[56:57], v[54:55] // (op 875) 
s_mov_b32 m0, s0 // (op 875) reg expression
s_mov_b32 s22, 256 // (op 875) call()
buffer_load_dwordx4 v56, s[60:63], s22 offen offset: 0 lds // (op 875) Load value direct to lds
v_add_co_u32 v56, s[24:25], v56, v52 // (op 875) least significant half
v_addc_co_u32 v57, s[24:25], 0, v57, s[24:25] // (op 875) most significant half

s_add_u32 m0, m0, 4096 // (op 875) 
buffer_load_dwordx4 v56, s[60:63], s22 offen offset: 0 lds // (op 875) Load value direct to lds





v_mov_b32 v13, v24 // (op 21) 

buffer_load_ubyte v25, v13, s[68:71], 0 offen offset:8 // (op 21) Load value
s_barrier  // (op 1533) 


v_mov_b64 v[52:53], v[18:19] // (op 870) 
s_add_u32 s22, 8192, s47 // (op 870) 

s_mov_b32 m0, s22 // (op 870) reg expression
s_mov_b32 s23, 256 // (op 870) call()
buffer_load_dwordx4 v52, s[48:51], s23 offen offset: 0 lds // (op 870) Load value direct to lds


s_barrier  // (op 1594) 



s_waitcnt vmcnt(6)
ds_write_b8 v17, v22 offset:24832 // (op 236) Store local data 
s_barrier  // (op 1585) 



ds_read_b128 v[28:31], v11 offset:12288 // (op 229) Load local data 
ds_read_b128 v[32:35], v11 offset:12352 // (op 229) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1591) 



ds_read_u8 v21, v10 offset:24832 // (op 233) Load local data 



ds_read_b128 v[36:39], v11 offset:16384 // (op 348) Load local data 
ds_read_b128 v[40:43], v11 offset:16448 // (op 348) Load local data 



ds_read_u8 v27, v10 offset:24960 // (op 351) Load local data 
s_barrier  // (op 1645) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1579) 



ds_read_b128 v[44:47], v4 offset:20480 // (op 227) Load local data 
ds_read_b128 v[48:51], v4 offset:20544 // (op 227) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v26, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 234) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v26, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 344) 



buffer_load_ubyte v26, v12, s[64:67], 0 offen offset:12 // (op 231) Load value

s_barrier  // (op 1588) 
v_add_u32 v23, 8, v23 // (op 1094) 

s_lshl_b64 s[22:23], s[30:31], 5 // (op 898) 
v_mov_b64 v[56:57], s[22:23] // (op 898) convert


v_mov_b64 v[58:59], v[54:55] // (op 898) 
s_add_u32 s22, 12288, s18 // (op 898) 
s_mov_b32 m0, s22 // (op 898) reg expression
s_mov_b32 s23, 384 // (op 898) call()
buffer_load_dwordx4 v58, s[60:63], s23 offen offset: 0 lds // (op 898) Load value direct to lds
v_add_co_u32 v58, s[24:25], v58, v56 // (op 898) least significant half
v_addc_co_u32 v59, s[24:25], 0, v59, s[24:25] // (op 898) most significant half

s_add_u32 m0, m0, 4096 // (op 898) 
buffer_load_dwordx4 v58, s[60:63], s23 offen offset: 0 lds // (op 898) Load value direct to lds








v_add_co_u32 v14, vcc, 256, v14 // (op 1030) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1030) most significant half
buffer_load_ubyte v22, v13, s[68:71], 0 offen offset:12 // (op 238) Load value

s_barrier  // (op 1582) 
v_add_u32 v24, 8, v24 // (op 1158) 


s_add_u32 s22, 20480, s1 // (op 893) 

s_mov_b32 m0, s22 // (op 893) reg expression
s_mov_b32 s23, 384 // (op 893) call()
buffer_load_dwordx4 v52, s[48:51], s23 offen offset: 0 lds // (op 893) Load value direct to lds



s_barrier  // (op 1565) 
v_add_co_u32 v18, vcc, 256, v18 // (op 966) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 966) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v16, v25 offset:24576 // (op 26) Store local data 
s_barrier  // (op 1555) 



ds_read_b128 v[28:31], v7 // (op 70) Load local data 
ds_read_b128 v[32:35], v7 offset:64 // (op 70) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1572) 



ds_read_u8 v21, v0 offset:24576 // (op 79) Load local data 



ds_read_b128 v[36:39], v7 offset:4096 // (op 339) Load local data 
ds_read_b128 v[40:43], v7 offset:4160 // (op 339) Load local data 



ds_read_u8 v27, v0 offset:24704 // (op 342) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1539) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 65) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 65) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 307) 
s_mov_b32 s24, s22 // (op 307) convert

s_and_b32 s25, s24, 4294967294 // (op 307) 

s_cmp_lt_u32 s19, s25 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_307 // (op 307) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_307 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_307:













s_barrier  // (op 1629) 
s_mul_hi_i32 s47, s16, s10 // (op 928) 
v_lshrrev_b32 v12, 3, v1 // (op 928) 
s_add_i32 s0, s47, s16 // (op 928) 

v_and_b32 v13, -8, v1 // (op 928) 
s_ashr_i32 s1, s0, 31 // (op 928) 
v_sub_u32 v14, v1, v13 // (op 928) 

s_cmp_eq_i32 s10, 0 // (op 928) 
s_mov_b32 s47, scc // (op 928) 
v_lshlrev_b32 v13, 4, v14 // (op 928) 

s_cmp_eq_i32 s47, 0 // (op 928) 
s_cselect_b32 s18, 0, -1 // (op 928) 

s_and_b32 s19, s13, 31 // (op 928) 
s_lshl_b32 s47, 1, s19 // (op 928) 
s_add_i32 s22, s47, s18 // (op 928) 


s_and_b32 s18, s1, s22 // (op 928) 


s_add_i32 s1, s0, s18 // (op 928) 


s_ashr_i32 s0, s1, s19 // (op 928) 


s_lshl_b32 s1, s13, 24 // (op 928) 
s_ashr_i32 s47, s1, 31 // (op 928) 

s_xor_b32 s1, s0, s47 // (op 928) 

s_sub_i32 s0, s1, s47 // (op 928) 


s_mul_hi_i32 s1, s0, s14 // (op 928) 
s_add_i32 s47, s1, s0 // (op 928) 

s_ashr_i32 s1, s47, 31 // (op 928) 
s_cmp_eq_i32 s14, 0 // (op 928) 
s_mov_b32 s18, scc // (op 928) 
s_cmp_eq_i32 s18, 0 // (op 928) 
s_cselect_b32 s19, 0, -1 // (op 928) 

s_and_b32 s18, s15, 31 // (op 928) 
s_lshl_b32 s22, 1, s18 // (op 928) 
s_add_i32 s23, s22, s19 // (op 928) 


s_and_b32 s19, s1, s23 // (op 928) 


s_add_i32 s1, s47, s19 // (op 928) 


s_ashr_i32 s19, s1, s18 // (op 928) 


s_lshl_b32 s1, s15, 24 // (op 928) 
s_ashr_i32 s18, s1, 31 // (op 928) 

s_xor_b32 s1, s19, s18 // (op 928) 

s_sub_i32 s19, s1, s18 // (op 928) 


s_mul_i32 s1, s0, s11 // (op 928) 

s_sub_u32 s0, s16, s1 // (op 928) 

s_mul_i32 s1, s19, s11 // (op 928) 
s_add_u32 s1, s1, s0 // (op 928) 


s_mul_hi_i32 s0, s16, s9 // (op 928) 
s_add_i32 s47, s0, s16 // (op 928) 

s_ashr_i32 s0, s47, 31 // (op 928) 
s_cmp_eq_i32 s9, 0 // (op 928) 
s_mov_b32 s18, scc // (op 928) 
s_cmp_eq_i32 s18, 0 // (op 928) 
s_cselect_b32 s19, 0, -1 // (op 928) 

s_and_b32 s18, s45, 31 // (op 928) 
s_lshl_b32 s22, 1, s18 // (op 928) 
s_add_i32 s23, s22, s19 // (op 928) 


s_and_b32 s19, s0, s23 // (op 928) 


s_add_i32 s0, s47, s19 // (op 928) 


s_ashr_i32 s19, s0, s18 // (op 928) 


s_lshl_b32 s0, s45, 24 // (op 928) 
s_ashr_i32 s18, s0, 31 // (op 928) 

s_xor_b32 s0, s19, s18 // (op 928) 

s_sub_i32 s19, s0, s18 // (op 928) 


s_add_u32 s22, s6, 64 // (op 928) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 928) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 928) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 928) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 928) 

s_mul_i32 s0, s11, s22 // (op 928) 

s_mul_i32 s18, s8, s0 // (op 928) 

s_mul_i32 s0, s19, s18 // (op 928) 

s_sub_u32 s18, s16, s0 // (op 928) 

s_mov_b32 s0, s18 // (op 928) 
s_mul_hi_i32 s47, s0, s12 // (op 928) 
s_add_i32 s22, s47, s0 // (op 928) 


s_ashr_i32 s0, s22, 31 // (op 928) 
s_cmp_eq_i32 s12, 0 // (op 928) 
s_mov_b32 s47, scc // (op 928) 
s_cmp_eq_i32 s47, 0 // (op 928) 
s_cselect_b32 s23, 0, -1 // (op 928) 

s_and_b32 s47, s44, 31 // (op 928) 
s_lshl_b32 s24, 1, s47 // (op 928) 
s_add_i32 s25, s24, s23 // (op 928) 


s_and_b32 s23, s0, s25 // (op 928) 


s_add_i32 s0, s22, s23 // (op 928) 


s_ashr_i32 s22, s0, s47 // (op 928) 


s_lshl_b32 s0, s44, 24 // (op 928) 
s_ashr_i32 s47, s0, 31 // (op 928) 

s_xor_b32 s0, s22, s47 // (op 928) 

s_sub_i32 s22, s0, s47 // (op 928) 


s_mul_i32 s0, s46, s11 // (op 928) 
s_add_u32 s24, s20, 32 // (op 928) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 928) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 928) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 928) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 5 // (op 928) 

s_sub_i32 s23, s24, s0 // (op 928) 

s_mul_i32 s47, s22, s23 // (op 928) 


s_sub_u32 s22, s18, s47 // (op 928) 


s_add_u32 s18, s22, s0 // (op 928) 


s_cmp_eq_i32 s19, 0 // (op 928) 
s_mov_b32 s0, scc // (op 928) 

s_cmp_eq_i32 s0, 0 // (op 928) 
s_cselect_b32 s19, s18, s1 // (op 928) 



v_lshl_add_u32 v14, s19, 5, v12 // (op 928) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 928) 
s_lshr_b32 s47, s0, 1 // (op 928) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 928) 


v_mov_b32 v25, 0 // (op 928) convert
v_mov_b32 v24, v12 // (op 928) convert

v_mul_lo_u32 v13, v14, s27 // (op 928) most significant: low of low * high
v_mul_hi_u32 v23, v14, s26 // (op 928) most significant: high of low * low
v_mul_lo_u32 v12, v14, s26 // (op 928) least significant: low of low * low
v_add_u32 v13, v13, v23 // (op 928) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v24 // (op 928) least significant half
v_addc_co_u32 v13, s[0:1], v13, v25, s[0:1] // (op 928) most significant half



s_mul_i32 s1, s26, s21 // (op 930) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 930) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 930) least significant
s_mul_hi_u32 s18, s26, s20 // (op 930) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 930) most significant: sum
s_add_u32 s1, s1, s18 // (op 930) most significant: sum


s_mov_b32 s48, s28 // (op 930) reg expression
s_mov_b32 s49, s29 // (op 930) reg expression
s_mov_b32 s50, s0 // (op 930) reg expression
s_mov_b32 s51, 131072 // (op 930) call()

s_mul_hi_i32 s47, s16, s10 // (op 996) 
v_lshrrev_b32 v14, 3, v1 // (op 996) 
s_add_i32 s0, s47, s16 // (op 996) 

v_and_b32 v15, -8, v1 // (op 996) 
s_ashr_i32 s1, s0, 31 // (op 996) 
v_sub_u32 v16, v1, v15 // (op 996) 

s_cmp_eq_i32 s10, 0 // (op 996) 
s_mov_b32 s47, scc // (op 996) 
v_lshlrev_b32 v15, 4, v16 // (op 996) 

s_cmp_eq_i32 s47, 0 // (op 996) 
s_cselect_b32 s18, 0, -1 // (op 996) 

s_and_b32 s19, s13, 31 // (op 996) 
s_lshl_b32 s47, 1, s19 // (op 996) 
s_add_i32 s22, s47, s18 // (op 996) 


s_and_b32 s18, s1, s22 // (op 996) 


s_add_i32 s1, s0, s18 // (op 996) 


s_ashr_i32 s0, s1, s19 // (op 996) 


s_lshl_b32 s1, s13, 24 // (op 996) 
s_ashr_i32 s47, s1, 31 // (op 996) 

s_xor_b32 s1, s0, s47 // (op 996) 

s_sub_i32 s0, s1, s47 // (op 996) 


s_mul_hi_i32 s1, s0, s14 // (op 996) 
s_add_i32 s47, s1, s0 // (op 996) 

s_ashr_i32 s1, s47, 31 // (op 996) 
s_cmp_eq_i32 s14, 0 // (op 996) 
s_mov_b32 s18, scc // (op 996) 
s_cmp_eq_i32 s18, 0 // (op 996) 
s_cselect_b32 s19, 0, -1 // (op 996) 

s_and_b32 s18, s15, 31 // (op 996) 
s_lshl_b32 s22, 1, s18 // (op 996) 
s_add_i32 s23, s22, s19 // (op 996) 


s_and_b32 s19, s1, s23 // (op 996) 


s_add_i32 s1, s47, s19 // (op 996) 


s_ashr_i32 s19, s1, s18 // (op 996) 


s_lshl_b32 s1, s15, 24 // (op 996) 
s_ashr_i32 s18, s1, 31 // (op 996) 

s_xor_b32 s1, s19, s18 // (op 996) 

s_sub_i32 s19, s1, s18 // (op 996) 


s_add_u32 s22, s6, 64 // (op 996) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 996) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 996) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 996) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 996) 

s_mul_i32 s1, s19, s22 // (op 996) 

s_sub_i32 s47, s0, s1 // (op 996) 


s_mul_hi_i32 s0, s16, s9 // (op 996) 
s_add_i32 s1, s0, s16 // (op 996) 

s_ashr_i32 s0, s1, 31 // (op 996) 
s_cmp_eq_i32 s9, 0 // (op 996) 
s_mov_b32 s18, scc // (op 996) 
s_cmp_eq_i32 s18, 0 // (op 996) 
s_cselect_b32 s19, 0, -1 // (op 996) 

s_and_b32 s18, s45, 31 // (op 996) 
s_lshl_b32 s24, 1, s18 // (op 996) 
s_add_i32 s25, s24, s19 // (op 996) 


s_and_b32 s19, s0, s25 // (op 996) 


s_add_i32 s0, s1, s19 // (op 996) 


s_ashr_i32 s1, s0, s18 // (op 996) 


s_lshl_b32 s0, s45, 24 // (op 996) 
s_ashr_i32 s18, s0, 31 // (op 996) 

s_xor_b32 s0, s1, s18 // (op 996) 

s_sub_i32 s1, s0, s18 // (op 996) 


s_mul_i32 s0, s11, s22 // (op 996) 

s_mul_i32 s18, s8, s0 // (op 996) 

s_mul_i32 s0, s1, s18 // (op 996) 

s_sub_u32 s18, s16, s0 // (op 996) 

s_mul_hi_i32 s0, s18, s12 // (op 996) 
s_add_i32 s19, s0, s18 // (op 996) 


s_ashr_i32 s0, s19, 31 // (op 996) 
s_cmp_eq_i32 s12, 0 // (op 996) 
s_mov_b32 s18, scc // (op 996) 
s_cmp_eq_i32 s18, 0 // (op 996) 
s_cselect_b32 s22, 0, -1 // (op 996) 

s_and_b32 s18, s44, 31 // (op 996) 
s_lshl_b32 s23, 1, s18 // (op 996) 
s_add_i32 s24, s23, s22 // (op 996) 


s_and_b32 s25, s0, s24 // (op 996) 


s_add_i32 s0, s19, s25 // (op 996) 


s_ashr_i32 s19, s0, s18 // (op 996) 


s_lshl_b32 s0, s44, 24 // (op 996) 
s_ashr_i32 s18, s0, 31 // (op 996) 

s_xor_b32 s0, s19, s18 // (op 996) 

s_sub_i32 s19, s0, s18 // (op 996) 


s_cmp_eq_i32 s1, 0 // (op 996) 
s_mov_b32 s0, scc // (op 996) 

s_cmp_eq_i32 s0, 0 // (op 996) 
s_cselect_b32 s1, s19, s47 // (op 996) 



v_lshl_add_u32 v16, s1, 6, v14 // (op 996) 


v_mul_lo_u32 v25, v16, s31 // (op 996) most significant: low of low * high
v_mul_hi_u32 v14, v16, s30 // (op 996) most significant: high of low * low
v_mul_lo_u32 v24, v16, s30 // (op 996) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 996) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 996) 
s_lshr_b32 s47, s0, 1 // (op 996) 

v_lshl_add_u32 v14, s47, 8, v15 // (op 996) 


v_mov_b32 v17, 0 // (op 996) convert
v_mov_b32 v16, v14 // (op 996) convert

v_add_co_u32 v14, s[0:1], v16, v24 // (op 996) least significant half
v_addc_co_u32 v15, s[0:1], v17, v25, s[0:1] // (op 996) most significant half



s_mul_i32 s1, s30, s7 // (op 998) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 998) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 998) least significant
s_mul_hi_u32 s18, s30, s6 // (op 998) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 998) most significant: sum
s_add_u32 s1, s1, s18 // (op 998) most significant: sum


s_mov_b32 s60, s32 // (op 998) reg expression
s_mov_b32 s61, s33 // (op 998) reg expression
s_mov_b32 s62, s0 // (op 998) reg expression
s_mov_b32 s63, 131072 // (op 998) call()

s_mul_hi_i32 s47, s16, s10 // (op 1060) 
v_lshrrev_b32 v16, 7, v1 // (op 1060) 
s_add_i32 s0, s47, s16 // (op 1060) 

v_and_b32 v17, 15, v1 // (op 1060) 
s_ashr_i32 s1, s0, 31 // (op 1060) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 1060) 


s_cmp_eq_i32 s10, 0 // (op 1060) 
s_mov_b32 s47, scc // (op 1060) 
v_and_b32 v16, 63, v1 // (op 1060) 
s_cmp_eq_i32 s47, 0 // (op 1060) 
s_cselect_b32 s18, 0, -1 // (op 1060) 

v_lshrrev_b32 v17, 4, v16 // (op 1060) 

s_and_b32 s19, s13, 31 // (op 1060) 
s_lshl_b32 s47, 1, s19 // (op 1060) 
s_add_i32 s22, s47, s18 // (op 1060) 


s_and_b32 s18, s1, s22 // (op 1060) 


s_add_i32 s1, s0, s18 // (op 1060) 


s_ashr_i32 s0, s1, s19 // (op 1060) 


s_lshl_b32 s1, s13, 24 // (op 1060) 
s_ashr_i32 s47, s1, 31 // (op 1060) 

s_xor_b32 s1, s0, s47 // (op 1060) 

s_sub_i32 s0, s1, s47 // (op 1060) 


s_mul_hi_i32 s1, s0, s14 // (op 1060) 
s_add_i32 s47, s1, s0 // (op 1060) 

s_ashr_i32 s1, s47, 31 // (op 1060) 
s_cmp_eq_i32 s14, 0 // (op 1060) 
s_mov_b32 s18, scc // (op 1060) 
s_cmp_eq_i32 s18, 0 // (op 1060) 
s_cselect_b32 s19, 0, -1 // (op 1060) 

s_and_b32 s18, s15, 31 // (op 1060) 
s_lshl_b32 s22, 1, s18 // (op 1060) 
s_add_i32 s23, s22, s19 // (op 1060) 


s_and_b32 s19, s1, s23 // (op 1060) 


s_add_i32 s1, s47, s19 // (op 1060) 


s_ashr_i32 s19, s1, s18 // (op 1060) 


s_lshl_b32 s1, s15, 24 // (op 1060) 
s_ashr_i32 s18, s1, 31 // (op 1060) 

s_xor_b32 s1, s19, s18 // (op 1060) 

s_sub_i32 s19, s1, s18 // (op 1060) 


s_mul_i32 s1, s0, s11 // (op 1060) 

s_sub_u32 s0, s16, s1 // (op 1060) 

s_mul_i32 s1, s19, s11 // (op 1060) 
s_add_u32 s1, s1, s0 // (op 1060) 


s_mul_hi_i32 s0, s16, s9 // (op 1060) 
s_add_i32 s47, s0, s16 // (op 1060) 

s_ashr_i32 s0, s47, 31 // (op 1060) 
s_cmp_eq_i32 s9, 0 // (op 1060) 
s_mov_b32 s18, scc // (op 1060) 
s_cmp_eq_i32 s18, 0 // (op 1060) 
s_cselect_b32 s19, 0, -1 // (op 1060) 

s_and_b32 s18, s45, 31 // (op 1060) 
s_lshl_b32 s22, 1, s18 // (op 1060) 
s_add_i32 s23, s22, s19 // (op 1060) 


s_and_b32 s19, s0, s23 // (op 1060) 


s_add_i32 s0, s47, s19 // (op 1060) 


s_ashr_i32 s19, s0, s18 // (op 1060) 


s_lshl_b32 s0, s45, 24 // (op 1060) 
s_ashr_i32 s18, s0, 31 // (op 1060) 

s_xor_b32 s0, s19, s18 // (op 1060) 

s_sub_i32 s19, s0, s18 // (op 1060) 


s_add_u32 s22, s6, 64 // (op 1060) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1060) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1060) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1060) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1060) 

s_mul_i32 s0, s11, s22 // (op 1060) 

s_mul_i32 s18, s8, s0 // (op 1060) 

s_mul_i32 s0, s19, s18 // (op 1060) 

s_sub_u32 s18, s16, s0 // (op 1060) 

s_mov_b32 s0, s18 // (op 1060) 
s_mul_hi_i32 s47, s0, s12 // (op 1060) 
s_add_i32 s22, s47, s0 // (op 1060) 


s_ashr_i32 s0, s22, 31 // (op 1060) 
s_cmp_eq_i32 s12, 0 // (op 1060) 
s_mov_b32 s47, scc // (op 1060) 
s_cmp_eq_i32 s47, 0 // (op 1060) 
s_cselect_b32 s23, 0, -1 // (op 1060) 

s_and_b32 s47, s44, 31 // (op 1060) 
s_lshl_b32 s24, 1, s47 // (op 1060) 
s_add_i32 s25, s24, s23 // (op 1060) 


s_and_b32 s23, s0, s25 // (op 1060) 


s_add_i32 s0, s22, s23 // (op 1060) 


s_ashr_i32 s22, s0, s47 // (op 1060) 


s_lshl_b32 s0, s44, 24 // (op 1060) 
s_ashr_i32 s47, s0, 31 // (op 1060) 

s_xor_b32 s0, s22, s47 // (op 1060) 

s_sub_i32 s22, s0, s47 // (op 1060) 


s_mul_i32 s0, s46, s11 // (op 1060) 
s_add_u32 s24, s20, 32 // (op 1060) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1060) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1060) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1060) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 5 // (op 1060) 

s_sub_i32 s23, s24, s0 // (op 1060) 

s_mul_i32 s47, s22, s23 // (op 1060) 


s_sub_u32 s22, s18, s47 // (op 1060) 


s_add_u32 s18, s22, s0 // (op 1060) 


s_cmp_eq_i32 s19, 0 // (op 1060) 
s_mov_b32 s0, scc // (op 1060) 

s_cmp_eq_i32 s0, 0 // (op 1060) 
s_cselect_b32 s19, s18, s1 // (op 1060) 



v_lshl_add_u32 v16, s19, 5, v18 // (op 1060) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1060) 
s_lshr_b32 s47, s0, 1 // (op 1060) 

v_lshl_add_u32 v18, s47, 3, v17 // (op 1060) 


v_mul_lo_u32 v17, v16, s34 // (op 1060) 
v_add_u32 v17, v17, v18 // (op 1060) 


s_mul_i32 s1, s34, s21 // (op 1062) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 1062) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 1062) least significant
s_mul_hi_u32 s18, s34, s20 // (op 1062) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1062) most significant: sum
s_add_u32 s1, s1, s18 // (op 1062) most significant: sum


s_mov_b32 s64, s36 // (op 1062) reg expression
s_mov_b32 s65, s37 // (op 1062) reg expression
s_mov_b32 s66, s0 // (op 1062) reg expression
s_mov_b32 s67, 131072 // (op 1062) call()

s_mul_hi_i32 s47, s16, s10 // (op 1124) 
v_lshrrev_b32 v16, 2, v1 // (op 1124) 
s_add_i32 s0, s47, s16 // (op 1124) 

v_and_b32 v18, -4, v1 // (op 1124) 
s_ashr_i32 s1, s0, 31 // (op 1124) 
v_sub_u32 v19, v1, v18 // (op 1124) 

s_cmp_eq_i32 s10, 0 // (op 1124) 
s_mov_b32 s47, scc // (op 1124) 
s_cmp_eq_i32 s47, 0 // (op 1124) 
s_cselect_b32 s18, 0, -1 // (op 1124) 

s_and_b32 s19, s13, 31 // (op 1124) 
s_lshl_b32 s47, 1, s19 // (op 1124) 
s_add_i32 s22, s47, s18 // (op 1124) 


s_and_b32 s18, s1, s22 // (op 1124) 


s_add_i32 s1, s0, s18 // (op 1124) 


s_ashr_i32 s0, s1, s19 // (op 1124) 


s_lshl_b32 s1, s13, 24 // (op 1124) 
s_ashr_i32 s47, s1, 31 // (op 1124) 

s_xor_b32 s1, s0, s47 // (op 1124) 

s_sub_i32 s0, s1, s47 // (op 1124) 


s_mul_hi_i32 s1, s0, s14 // (op 1124) 
s_add_i32 s47, s1, s0 // (op 1124) 

s_ashr_i32 s1, s47, 31 // (op 1124) 
s_cmp_eq_i32 s14, 0 // (op 1124) 
s_mov_b32 s18, scc // (op 1124) 
s_cmp_eq_i32 s18, 0 // (op 1124) 
s_cselect_b32 s19, 0, -1 // (op 1124) 

s_and_b32 s18, s15, 31 // (op 1124) 
s_lshl_b32 s22, 1, s18 // (op 1124) 
s_add_i32 s23, s22, s19 // (op 1124) 


s_and_b32 s19, s1, s23 // (op 1124) 


s_add_i32 s1, s47, s19 // (op 1124) 


s_ashr_i32 s19, s1, s18 // (op 1124) 


s_lshl_b32 s1, s15, 24 // (op 1124) 
s_ashr_i32 s18, s1, 31 // (op 1124) 

s_xor_b32 s1, s19, s18 // (op 1124) 

s_sub_i32 s19, s1, s18 // (op 1124) 


s_add_u32 s22, s6, 64 // (op 1124) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1124) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1124) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1124) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1124) 

s_mul_i32 s1, s19, s22 // (op 1124) 

s_sub_i32 s47, s0, s1 // (op 1124) 


s_mul_hi_i32 s0, s16, s9 // (op 1124) 
s_add_i32 s1, s0, s16 // (op 1124) 

s_ashr_i32 s0, s1, 31 // (op 1124) 
s_cmp_eq_i32 s9, 0 // (op 1124) 
s_mov_b32 s18, scc // (op 1124) 
s_cmp_eq_i32 s18, 0 // (op 1124) 
s_cselect_b32 s19, 0, -1 // (op 1124) 

s_and_b32 s18, s45, 31 // (op 1124) 
s_lshl_b32 s24, 1, s18 // (op 1124) 
s_add_i32 s25, s24, s19 // (op 1124) 


s_and_b32 s19, s0, s25 // (op 1124) 


s_add_i32 s0, s1, s19 // (op 1124) 


s_ashr_i32 s1, s0, s18 // (op 1124) 


s_lshl_b32 s0, s45, 24 // (op 1124) 
s_ashr_i32 s18, s0, 31 // (op 1124) 

s_xor_b32 s0, s1, s18 // (op 1124) 

s_sub_i32 s1, s0, s18 // (op 1124) 


s_mul_i32 s0, s11, s22 // (op 1124) 

s_mul_i32 s18, s8, s0 // (op 1124) 

s_mul_i32 s0, s1, s18 // (op 1124) 

s_sub_u32 s18, s16, s0 // (op 1124) 

s_mul_hi_i32 s0, s18, s12 // (op 1124) 
s_add_i32 s19, s0, s18 // (op 1124) 


s_ashr_i32 s0, s19, 31 // (op 1124) 
s_cmp_eq_i32 s12, 0 // (op 1124) 
s_mov_b32 s18, scc // (op 1124) 
s_cmp_eq_i32 s18, 0 // (op 1124) 
s_cselect_b32 s22, 0, -1 // (op 1124) 

s_and_b32 s18, s44, 31 // (op 1124) 
s_lshl_b32 s23, 1, s18 // (op 1124) 
s_add_i32 s24, s23, s22 // (op 1124) 


s_and_b32 s25, s0, s24 // (op 1124) 


s_add_i32 s0, s19, s25 // (op 1124) 


s_ashr_i32 s19, s0, s18 // (op 1124) 


s_lshl_b32 s0, s44, 24 // (op 1124) 
s_ashr_i32 s18, s0, 31 // (op 1124) 

s_xor_b32 s0, s19, s18 // (op 1124) 

s_sub_i32 s19, s0, s18 // (op 1124) 


s_cmp_eq_i32 s1, 0 // (op 1124) 
s_mov_b32 s0, scc // (op 1124) 

s_cmp_eq_i32 s0, 0 // (op 1124) 
s_cselect_b32 s1, s19, s47 // (op 1124) 



v_lshl_add_u32 v18, s1, 6, v16 // (op 1124) 


v_mul_lo_u32 v16, v18, s38 // (op 1124) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1124) 
s_lshr_b32 s47, s0, 1 // (op 1124) 

v_lshl_add_u32 v18, s47, 3, v19 // (op 1124) 


v_add_u32 v19, v18, v16 // (op 1124) 


s_mul_i32 s1, s38, s7 // (op 1126) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 1126) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 1126) least significant
s_mul_hi_u32 s18, s38, s6 // (op 1126) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1126) most significant: sum
s_add_u32 s1, s1, s18 // (op 1126) most significant: sum


s_mov_b32 s68, s40 // (op 1126) reg expression
s_mov_b32 s69, s41 // (op 1126) reg expression
s_mov_b32 s70, s0 // (op 1126) reg expression
s_mov_b32 s71, 131072 // (op 1126) call()

v_and_b32 v16, -8, v1 // (op 1217) 
v_sub_u32 v18, v1, v16 // (op 1217) 

v_lshlrev_b32 v16, 4, v18 // (op 1217) 

v_lshrrev_b32 v18, 3, v1 // (op 1217) 
v_lshl_add_u32 v20, v18, 7, v16 // (op 1217) 


s_nop 0
v_readfirstlane_b32 s47, v20 // (op 1217) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 1269) 
v_sub_u32 v18, v1, v16 // (op 1269) 

v_lshlrev_b32 v16, 4, v18 // (op 1269) 

v_lshrrev_b32 v18, 3, v1 // (op 1269) 
v_lshl_add_u32 v20, v18, 7, v16 // (op 1269) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1269) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 1326) 
v_sub_u32 v18, v1, v16 // (op 1326) 

v_and_b32 v16, 4294967292, v1 // (op 1326) 
v_add_u32 v20, v16, v18 // (op 1326) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 313) 
s_and_b32 s1, s18, 4294967294 // (op 313) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 312) 
s_mov_b32 s22, s18 // (op 312) convert

s_cmp_lt_u32 s1, s22 // (op 312) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312 // (op 312) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312:
s_barrier  // (op 1536) 
s_add_u32 s1, s1, 1 // (op 314) 


v_mov_b64 v[22:23], v[12:13] // (op 880) 
s_add_u32 s18, 8192, s47 // (op 880) 

s_mov_b32 m0, s18 // (op 880) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 880) Load value direct to lds


v_add_co_u32 v12, vcc, 128, v12 // (op 918) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 918) most significant half
s_barrier  // (op 1552) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 886) 
v_mov_b64 v[22:23], s[18:19] // (op 886) convert


v_mov_b64 v[24:25], v[14:15] // (op 886) 
v_mov_b64 v[26:27], v[24:25] // (op 886) 
s_mov_b32 m0, s0 // (op 886) reg expression
buffer_load_dwordx4 v26, s[60:63], 0 offen offset: 0 lds // (op 886) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v22 // (op 886) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 886) most significant half

s_add_u32 m0, m0, 4096 // (op 886) 
buffer_load_dwordx4 v26, s[60:63], 0 offen offset: 0 lds // (op 886) Load value direct to lds



v_add_co_u32 v14, vcc, 128, v14 // (op 988) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 988) most significant half


v_mov_b32 v16, v19 // (op 180) 

buffer_load_ubyte v18, v16, s[68:71], 0 offen offset: 0 // (op 180) Load value

v_add_u32 v19, 4, v19 // (op 1116) 
s_barrier  // (op 1568) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1542) 






ds_read_b128 v[22:25], v3 offset:8192 // (op 169) Load local data 
ds_read_b128 v[26:29], v3 offset:8256 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v18 offset:24576 // (op 178) Store local data 

s_barrier  // (op 1558) 



ds_read_b128 v[30:33], v7 // (op 171) Load local data 
ds_read_b128 v[34:37], v7 offset:64 // (op 171) Load local data 


v_mov_b32 v16, v17 // (op 173) 

buffer_load_ubyte v18, v16, s[64:67], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v17, 4, v17 // (op 1052) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1575) 



ds_read_u8 v16, v0 offset:24576 // (op 175) Load local data 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:37], a[0:3], v18, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 176) 
ds_read_b128 v[38:41], v7 offset:4096 // (op 320) Load local data 


ds_read_b128 v[42:45], v7 offset:4160 // (op 320) Load local data 



ds_read_u8 v16, v0 offset:24704 // (op 323) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:29], v[38:45], a[4:7], v18, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 324) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 312) 
s_mov_b32 s22, s18 // (op 312) convert

s_cmp_lt_u32 s1, s22 // (op 312) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312 // (op 312) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_312 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_312:





















s_barrier  // (op 1633) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 



v_mov_b32 v0, v9 // (op 52) reg expression

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 52) Store value


v_accvgpr_read v10, a4 // (op 504) 
v_accvgpr_read v11, a5 // (op 504) 
v_accvgpr_read v12, a6 // (op 504) 
v_accvgpr_read v13, a7 // (op 504) 
v_mul_f32 v14, v8, v10 // (op 504) 
v_mul_f32 v15, v8, v11 // (op 504) 
v_mul_f32 v16, v8, v12 // (op 504) 
v_mul_f32 v17, v8, v13 // (op 504) 



s_lshl_b32 s0, s4, 7 // (op 506) 
v_add_u32 v0, v9, s0 // (op 506) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 506) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_123:


v_mov_b32 v0, v6 // (op 135) reg expression

buffer_load_dwordx4 v[10:13], v0, s[52:55], 0 offen // (op 135) Load value

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

buffer_store_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 134) Store value




s_lshl_b32 s0, s2, 7 // (op 528) 
v_add_u32 v0, v6, s0 // (op 528) 


buffer_load_dwordx4 v[10:13], v0, s[52:55], 0 offen // (op 528) Load value

v_accvgpr_read v14, a4 // (op 525) 
v_accvgpr_read v15, a5 // (op 525) 
v_accvgpr_read v16, a6 // (op 525) 
v_accvgpr_read v17, a7 // (op 525) 
v_mul_f32 v18, v8, v14 // (op 525) 
v_mul_f32 v19, v8, v15 // (op 525) 
v_mul_f32 v20, v8, v16 // (op 525) 
v_mul_f32 v21, v8, v17 // (op 525) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v5, v10 // (op 529) 
v_mul_f32 v15, v5, v11 // (op 529) 
v_mul_f32 v16, v5, v12 // (op 529) 
v_mul_f32 v17, v5, v13 // (op 529) 

v_add_f32 v10, v14, v18 // (op 526) 
v_add_f32 v11, v15, v19 // (op 526) 
v_add_f32 v12, v16, v20 // (op 526) 
v_add_f32 v13, v17, v21 // (op 526) 




s_lshl_b32 s0, s4, 7 // (op 527) 
v_add_u32 v0, v9, s0 // (op 527) 


buffer_store_dwordx4 v[10:13], v0, s[56:59], 0 offen // (op 527) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 68
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
  .amdhsa_accum_offset 60
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
    macrotile: [32, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 60
    .agpr_count: 8
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
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

