

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_,@function
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1144) 
s_mul_hi_i32 s1, s16, s10 // (op 1333) 
v_and_b32 v0, 15, v1 // (op 1259) 
v_and_b32 v3, 63, v1 // (op 1259) 
v_lshrrev_b32 v4, 4, v3 // (op 1259) 

v_lshlrev_b32 v3, 5, v4 // (op 1259) 

v_lshl_add_u32 v4, v0, 7, v3 // (op 1259) 


v_mul_lo_u32 v0, v4, 6 // (op 1259) 

v_lshrrev_b32 v3, 3, v0 // (op 1259) 

v_and_b32 v0, 15, v1 // (op 1388) 
v_and_b32 v4, 63, v1 // (op 1388) 
v_lshrrev_b32 v5, 4, v4 // (op 1388) 

v_lshlrev_b32 v4, 5, v5 // (op 1388) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 1388) 


v_mul_lo_u32 v0, v5, 6 // (op 1388) 

v_lshrrev_b32 v4, 3, v0 // (op 1388) 

v_lshrrev_b32 v0, 6, v1 // (op 1144) 
s_add_i32 s18, s0, s16 // (op 1144) 

v_and_b32 v5, 63, v1 // (op 1144) 
s_ashr_i32 s0, s18, 31 // (op 1144) 
v_lshrrev_b32 v6, 4, v5 // (op 1144) 
s_cmp_eq_i32 s10, 0 // (op 1144) 
s_mov_b32 s19, scc // (op 1144) 
v_lshlrev_b32 v7, 2, v6 // (op 1144) 

s_cmp_eq_i32 s19, 0 // (op 1144) 
s_cselect_b32 s49, 0, -1 // (op 1144) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1144) 


s_and_b32 s19, s13, 31 // (op 1144) 
v_lshrrev_b32 v0, 2, v5 // (op 1144) 

v_and_b32 v5, 3, v0 // (op 1144) 

v_lshrrev_b32 v0, 6, v1 // (op 1206) 
v_and_b32 v7, 15, v1 // (op 1206) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 1206) 


v_and_b32 v0, 63, v1 // (op 1206) 
v_lshrrev_b32 v7, 4, v0 // (op 1206) 

v_lshlrev_b32 v0, 5, v7 // (op 1206) 

v_lshl_add_u32 v7, v8, 7, v0 // (op 1206) 


v_mul_lo_u32 v0, v7, 6 // (op 1206) 

v_lshrrev_b32 v7, 3, v0 // (op 1206) 

v_lshrrev_b32 v0, 6, v1 // (op 1317) 
v_and_b32 v8, 15, v1 // (op 1317) 
v_lshl_add_u32 v9, v0, 4, v8 // (op 1317) 


v_and_b32 v0, 63, v1 // (op 1317) 
v_lshrrev_b32 v8, 4, v0 // (op 1317) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 1317) 


v_lshrrev_b32 v8, 6, v1 // (op 1333) 
v_and_b32 v9, 63, v1 // (op 1333) 
v_lshrrev_b32 v10, 6, v1 // (op 1364) 
v_and_b32 v11, 15, v1 // (op 1364) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 1364) 


v_and_b32 v10, 63, v1 // (op 1364) 
v_lshrrev_b32 v11, 4, v10 // (op 1364) 

v_lshlrev_b32 v10, 5, v11 // (op 1364) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 1364) 


v_mul_lo_u32 v10, v11, 6 // (op 1364) 

v_lshrrev_b32 v11, 3, v10 // (op 1364) 

v_lshrrev_b32 v10, 6, v1 // (op 1417) 
v_and_b32 v12, 15, v1 // (op 1417) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 1417) 


v_and_b32 v10, 63, v1 // (op 1417) 
v_lshrrev_b32 v12, 4, v10 // (op 1417) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 1417) 


s_lshl_b32 s50, 1, s19 // (op 1144) 
v_and_b32 v12, 3, v1 // (op 1144) 
s_add_i32 s51, s50, s49 // (op 1144) 


s_and_b32 s49, s0, s51 // (op 1144) 


s_add_i32 s0, s1, s16 // (op 1333) 

s_ashr_i32 s1, s0, 31 // (op 1333) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1144) 


s_add_i32 s50, s18, s49 // (op 1144) 


s_ashr_i32 s18, s50, s19 // (op 1144) 


s_lshl_b32 s19, s13, 24 // (op 1144) 
s_ashr_i32 s49, s19, 31 // (op 1144) 

s_xor_b32 s19, s18, s49 // (op 1144) 

s_sub_i32 s18, s19, s49 // (op 1144) 


s_mul_hi_i32 s19, s18, s14 // (op 1144) 
s_add_i32 s49, s19, s18 // (op 1144) 

s_ashr_i32 s19, s49, 31 // (op 1144) 
s_cmp_eq_i32 s14, 0 // (op 1144) 
s_mov_b32 s50, scc // (op 1144) 
s_cmp_eq_i32 s50, 0 // (op 1144) 
s_cselect_b32 s51, 0, -1 // (op 1144) 

s_and_b32 s50, s15, 31 // (op 1144) 
v_lshrrev_b32 v5, 4, v9 // (op 1333) 
s_cmp_eq_i32 s10, 0 // (op 1333) 
s_mov_b32 s52, scc // (op 1333) 
v_lshlrev_b32 v12, 2, v5 // (op 1333) 

s_cmp_eq_i32 s52, 0 // (op 1333) 
s_cselect_b32 s53, 0, -1 // (op 1333) 

s_lshl_b32 s52, 1, s50 // (op 1144) 
v_lshl_add_u32 v5, v8, 4, v12 // (op 1333) 


v_lshrrev_b32 v8, 2, v9 // (op 1333) 

s_add_i32 s54, s52, s51 // (op 1144) 


s_and_b32 s51, s19, s54 // (op 1144) 


s_add_i32 s19, s49, s51 // (op 1144) 


s_ashr_i32 s49, s19, s50 // (op 1144) 


s_lshl_b32 s19, s15, 24 // (op 1144) 
s_ashr_i32 s50, s19, 31 // (op 1144) 

s_xor_b32 s19, s49, s50 // (op 1144) 

s_sub_i32 s49, s19, s50 // (op 1144) 


s_mul_i32 s19, s18, s11 // (op 1144) 
s_sub_u32 s50, s16, s19 // (op 1144) 

s_mul_i32 s19, s49, s11 // (op 1144) 
s_add_u32 s19, s19, s50 // (op 1144) 

s_mul_hi_i32 s50, s16, s9 // (op 1144) 
s_add_i32 s51, s50, s16 // (op 1144) 

s_ashr_i32 s50, s51, 31 // (op 1144) 
s_cmp_eq_i32 s9, 0 // (op 1144) 
s_mov_b32 s52, scc // (op 1144) 
s_cmp_eq_i32 s52, 0 // (op 1144) 
s_cselect_b32 s54, 0, -1 // (op 1144) 

s_and_b32 s52, s13, 31 // (op 1333) 
v_and_b32 v9, 3, v8 // (op 1333) 

s_lshl_b32 s55, 1, s52 // (op 1333) 
v_and_b32 v8, 3, v1 // (op 1333) 
s_add_i32 s56, s55, s53 // (op 1333) 


s_and_b32 s53, s1, s56 // (op 1333) 


v_lshl_add_u32 v12, v9, 2, v8 // (op 1333) 


s_add_i32 s1, s0, s53 // (op 1333) 


s_ashr_i32 s0, s1, s52 // (op 1333) 


s_lshl_b32 s1, s13, 24 // (op 1333) 
s_ashr_i32 s52, s1, 31 // (op 1333) 

s_xor_b32 s1, s0, s52 // (op 1333) 

s_sub_i32 s0, s1, s52 // (op 1333) 


s_mul_hi_i32 s1, s0, s14 // (op 1333) 
s_add_i32 s52, s1, s0 // (op 1333) 

s_ashr_i32 s1, s52, 31 // (op 1333) 
s_cmp_eq_i32 s14, 0 // (op 1333) 
s_mov_b32 s53, scc // (op 1333) 
s_cmp_eq_i32 s53, 0 // (op 1333) 
s_cselect_b32 s55, 0, -1 // (op 1333) 

s_and_b32 s53, s15, 31 // (op 1333) 
s_lshl_b32 s56, 1, s53 // (op 1333) 
s_add_i32 s57, s56, s55 // (op 1333) 


s_and_b32 s55, s1, s57 // (op 1333) 


s_add_i32 s1, s52, s55 // (op 1333) 


s_ashr_i32 s52, s1, s53 // (op 1333) 


s_lshl_b32 s1, s15, 24 // (op 1333) 
s_ashr_i32 s53, s1, 31 // (op 1333) 

s_xor_b32 s1, s52, s53 // (op 1333) 

s_sub_i32 s52, s1, s53 // (op 1333) 


s_mul_i32 s1, s0, s11 // (op 1333) 
s_sub_u32 s53, s16, s1 // (op 1333) 

s_mul_i32 s1, s52, s11 // (op 1333) 
s_add_u32 s1, s1, s53 // (op 1333) 

s_mul_hi_i32 s53, s16, s9 // (op 1333) 
s_add_i32 s55, s53, s16 // (op 1333) 

s_ashr_i32 s53, s55, 31 // (op 1333) 
s_cmp_eq_i32 s9, 0 // (op 1333) 
s_mov_b32 s56, scc // (op 1333) 
s_cmp_eq_i32 s56, 0 // (op 1333) 
s_cselect_b32 s57, 0, -1 // (op 1333) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 1144) 
s_lshl_b32 s58, 1, s56 // (op 1144) 
s_add_i32 s59, s58, s54 // (op 1144) 


s_and_b32 s54, s50, s59 // (op 1144) 


s_add_i32 s50, s51, s54 // (op 1144) 


s_ashr_i32 s51, s50, s56 // (op 1144) 


s_lshl_b32 s50, s45, 24 // (op 1144) 
s_ashr_i32 s54, s50, 31 // (op 1144) 

s_xor_b32 s50, s51, s54 // (op 1144) 

s_sub_i32 s51, s50, s54 // (op 1144) 


s_and_b32 s50, s45, 31 // (op 1333) 
s_lshl_b32 s54, 1, s50 // (op 1333) 
s_add_i32 s56, s54, s57 // (op 1333) 


s_and_b32 s54, s53, s56 // (op 1333) 


s_add_i32 s53, s55, s54 // (op 1333) 


s_ashr_i32 s54, s53, s50 // (op 1333) 


s_lshl_b32 s50, s45, 24 // (op 1333) 
s_ashr_i32 s53, s50, 31 // (op 1333) 

s_xor_b32 s50, s54, s53 // (op 1333) 

s_sub_i32 s54, s50, s53 // (op 1333) 


s_add_u32 s56, s4, 16 // (op 1144) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 1144) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1144) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1144) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 1144) 

s_mul_i32 s50, s11, s56 // (op 1144) 
s_mul_i32 s53, s8, s50 // (op 1144) 

s_mul_i32 s50, s51, s53 // (op 1144) 

s_sub_u32 s53, s16, s50 // (op 1144) 

s_mov_b32 s50, s53 // (op 1144) 
s_mul_hi_i32 s55, s50, s12 // (op 1144) 
s_add_i32 s58, s55, s50 // (op 1144) 


s_ashr_i32 s50, s58, 31 // (op 1144) 
s_cmp_eq_i32 s12, 0 // (op 1144) 
s_mov_b32 s55, scc // (op 1144) 
s_cmp_eq_i32 s55, 0 // (op 1144) 
s_cselect_b32 s59, 0, -1 // (op 1144) 

s_and_b32 s55, s44, 31 // (op 1144) 
s_lshl_b32 s60, 1, s55 // (op 1144) 
s_add_i32 s61, s60, s59 // (op 1144) 


s_and_b32 s59, s50, s61 // (op 1144) 


s_add_i32 s50, s58, s59 // (op 1144) 


s_ashr_i32 s58, s50, s55 // (op 1144) 


s_lshl_b32 s50, s44, 24 // (op 1144) 
s_ashr_i32 s55, s50, 31 // (op 1144) 

s_xor_b32 s50, s58, s55 // (op 1144) 

s_sub_i32 s58, s50, s55 // (op 1144) 


s_mul_i32 s50, s46, s11 // (op 1144) 
s_add_u32 s60, s6, 64 // (op 1144) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 1144) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 1144) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 1144) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 6 // (op 1144) 

s_sub_i32 s55, s60, s50 // (op 1144) 

s_mul_i32 s59, s58, s55 // (op 1144) 

s_sub_u32 s55, s53, s59 // (op 1144) 


s_add_u32 s53, s55, s50 // (op 1144) 


s_cmp_eq_i32 s51, 0 // (op 1144) 
s_mov_b32 s50, scc // (op 1144) 

s_cmp_eq_i32 s50, 0 // (op 1144) 
s_cselect_b32 s51, s53, s19 // (op 1144) 


s_mul_i32 s19, s49, s56 // (op 1144) 


v_lshl_add_u32 v8, s51, 6, v6 // (op 1144) 


s_sub_i32 s49, s18, s19 // (op 1144) 


s_cmp_eq_i32 s50, 0 // (op 1144) 
s_cselect_b32 s51, s58, s49 // (op 1144) 



v_lshl_add_u32 v6, s51, 4, v13 // (op 1144) 


v_mul_lo_u32 v9, v6, s2 // (op 1144) 

v_add_lshl_u32 v6, v8, v9, 1 // (op 1144) 


s_add_u32 s18, s4, 16 // (op 1333) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1333) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 1333) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 1333) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 1333) 

s_mul_i32 s53, s11, s18 // (op 1333) 
s_mul_i32 s49, s8, s53 // (op 1333) 

s_mul_i32 s53, s54, s49 // (op 1333) 

s_sub_u32 s49, s16, s53 // (op 1333) 

s_mov_b32 s53, s49 // (op 1333) 
s_mul_hi_i32 s50, s53, s12 // (op 1333) 
s_add_i32 s51, s50, s53 // (op 1333) 


s_ashr_i32 s50, s51, 31 // (op 1333) 
s_cmp_eq_i32 s12, 0 // (op 1333) 
s_mov_b32 s53, scc // (op 1333) 
s_cmp_eq_i32 s53, 0 // (op 1333) 
s_cselect_b32 s55, 0, -1 // (op 1333) 

s_and_b32 s53, s44, 31 // (op 1333) 
s_lshl_b32 s56, 1, s53 // (op 1333) 
s_add_i32 s57, s56, s55 // (op 1333) 


s_and_b32 s55, s50, s57 // (op 1333) 


s_add_i32 s50, s51, s55 // (op 1333) 


s_ashr_i32 s51, s50, s53 // (op 1333) 


s_lshl_b32 s50, s44, 24 // (op 1333) 
s_ashr_i32 s53, s50, 31 // (op 1333) 

s_xor_b32 s50, s51, s53 // (op 1333) 

s_sub_i32 s51, s50, s53 // (op 1333) 


s_mul_i32 s50, s46, s11 // (op 1333) 
s_add_u32 s56, s6, 64 // (op 1333) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1333) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1333) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1333) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1333) 

s_sub_i32 s53, s56, s50 // (op 1333) 

s_mul_i32 s55, s51, s53 // (op 1333) 

s_sub_u32 s53, s49, s55 // (op 1333) 


s_add_u32 s49, s53, s50 // (op 1333) 


s_cmp_eq_i32 s54, 0 // (op 1333) 
s_mov_b32 s50, scc // (op 1333) 

s_cmp_eq_i32 s50, 0 // (op 1333) 
s_cselect_b32 s53, s49, s1 // (op 1333) 


s_mul_i32 s1, s52, s18 // (op 1333) 


v_lshl_add_u32 v8, s53, 6, v5 // (op 1333) 


s_sub_i32 s49, s0, s1 // (op 1333) 


s_cmp_eq_i32 s50, 0 // (op 1333) 
s_cselect_b32 s0, s51, s49 // (op 1333) 



v_lshl_add_u32 v5, s0, 4, v12 // (op 1333) 


v_mul_lo_u32 v9, v5, s20 // (op 1333) 

v_add_lshl_u32 v5, v8, v9, 1 // (op 1333) 


s_mul_i32 s1, s2, s5 // (op 1146) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 1146) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 1146) least significant
s_mul_hi_u32 s19, s2, s4 // (op 1146) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1146) most significant: sum
s_add_u32 s1, s1, s19 // (op 1146) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1146) 

s_mul_i32 s1, s20, s5 // (op 1335) most significant: low of low * high
s_mul_i32 s49, s21, s4 // (op 1335) most significant: low of high * low
s_mul_i32 s0, s20, s4 // (op 1335) least significant
s_mul_hi_u32 s50, s20, s4 // (op 1335) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 1335) most significant: sum
s_add_u32 s1, s1, s50 // (op 1335) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 1335) 

s_mov_b32 s52, s22 // (op 1146) reg expression
s_mov_b32 s53, s23 // (op 1146) reg expression
s_mov_b32 s54, s18 // (op 1146) reg expression
s_mov_b32 s55, 131072 // (op 1146) call()

s_mov_b32 s56, s24 // (op 1335) reg expression

s_mov_b32 s57, s25 // (op 1335) reg expression
s_mov_b32 s58, s50 // (op 1335) reg expression
s_mov_b32 s59, 131072 // (op 1335) call()


v_mov_b32 v8, s47 // (op 2) Move value
v_mov_b32 v9, s48 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
s_mul_hi_i32 s0, s16, s10 // (op 911) 
v_lshrrev_b32 v12, 1, v1 // (op 911) 
s_add_i32 s1, s0, s16 // (op 911) 

v_and_b32 v13, -2, v1 // (op 911) 
s_ashr_i32 s0, s1, 31 // (op 911) 
v_sub_u32 v14, v1, v13 // (op 911) 

s_cmp_eq_i32 s10, 0 // (op 911) 
s_mov_b32 s18, scc // (op 911) 
v_lshlrev_b32 v13, 6, v14 // (op 911) 

s_cmp_eq_i32 s18, 0 // (op 911) 
s_cselect_b32 s19, 0, -1 // (op 911) 

v_mov_b32 v15, 0 // (op 911) convert
v_mov_b32 v14, v13 // (op 911) convert

s_and_b32 s18, s13, 31 // (op 911) 
s_lshl_b32 s22, 1, s18 // (op 911) 
s_add_i32 s23, s22, s19 // (op 911) 


s_and_b32 s19, s0, s23 // (op 911) 


s_add_i32 s0, s1, s19 // (op 911) 


s_ashr_i32 s1, s0, s18 // (op 911) 


s_lshl_b32 s0, s13, 24 // (op 911) 
s_ashr_i32 s18, s0, 31 // (op 911) 

s_xor_b32 s0, s1, s18 // (op 911) 

s_sub_i32 s1, s0, s18 // (op 911) 


s_mul_hi_i32 s0, s1, s14 // (op 911) 
s_add_i32 s18, s0, s1 // (op 911) 

s_ashr_i32 s0, s18, 31 // (op 911) 
s_cmp_eq_i32 s14, 0 // (op 911) 
s_mov_b32 s19, scc // (op 911) 
s_cmp_eq_i32 s19, 0 // (op 911) 
s_cselect_b32 s22, 0, -1 // (op 911) 

s_and_b32 s19, s15, 31 // (op 911) 
s_lshl_b32 s23, 1, s19 // (op 911) 
s_add_i32 s24, s23, s22 // (op 911) 


s_and_b32 s25, s0, s24 // (op 911) 


s_add_i32 s0, s18, s25 // (op 911) 


s_ashr_i32 s18, s0, s19 // (op 911) 


s_lshl_b32 s0, s15, 24 // (op 911) 
s_ashr_i32 s19, s0, 31 // (op 911) 

s_xor_b32 s0, s18, s19 // (op 911) 

s_sub_i32 s18, s0, s19 // (op 911) 


s_mul_i32 s0, s1, s11 // (op 911) 

s_sub_u32 s1, s16, s0 // (op 911) 

s_mul_i32 s0, s18, s11 // (op 911) 
s_add_u32 s0, s0, s1 // (op 911) 


s_mul_hi_i32 s1, s16, s9 // (op 911) 
s_add_i32 s18, s1, s16 // (op 911) 

s_ashr_i32 s1, s18, 31 // (op 911) 
s_cmp_eq_i32 s9, 0 // (op 911) 
s_mov_b32 s19, scc // (op 911) 
s_cmp_eq_i32 s19, 0 // (op 911) 
s_cselect_b32 s22, 0, -1 // (op 911) 

s_and_b32 s19, s45, 31 // (op 911) 
s_lshl_b32 s23, 1, s19 // (op 911) 
s_add_i32 s24, s23, s22 // (op 911) 


s_and_b32 s25, s1, s24 // (op 911) 


s_add_i32 s1, s18, s25 // (op 911) 


s_ashr_i32 s18, s1, s19 // (op 911) 


s_lshl_b32 s1, s45, 24 // (op 911) 
s_ashr_i32 s19, s1, 31 // (op 911) 

s_xor_b32 s1, s18, s19 // (op 911) 

s_sub_i32 s18, s1, s19 // (op 911) 


s_add_u32 s22, s4, 16 // (op 911) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 911) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 911) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 911) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 911) 

s_mul_i32 s1, s11, s22 // (op 911) 

s_mul_i32 s19, s8, s1 // (op 911) 

s_mul_i32 s1, s18, s19 // (op 911) 

s_sub_u32 s19, s16, s1 // (op 911) 

s_mov_b32 s1, s19 // (op 911) 
s_mul_hi_i32 s22, s1, s12 // (op 911) 
s_add_i32 s23, s22, s1 // (op 911) 


s_ashr_i32 s1, s23, 31 // (op 911) 
s_cmp_eq_i32 s12, 0 // (op 911) 
s_mov_b32 s22, scc // (op 911) 
s_cmp_eq_i32 s22, 0 // (op 911) 
s_cselect_b32 s24, 0, -1 // (op 911) 

s_and_b32 s22, s44, 31 // (op 911) 
s_lshl_b32 s25, 1, s22 // (op 911) 
s_add_i32 s47, s25, s24 // (op 911) 


s_and_b32 s24, s1, s47 // (op 911) 


s_add_i32 s1, s23, s24 // (op 911) 


s_ashr_i32 s23, s1, s22 // (op 911) 


s_lshl_b32 s1, s44, 24 // (op 911) 
s_ashr_i32 s22, s1, 31 // (op 911) 

s_xor_b32 s1, s23, s22 // (op 911) 

s_sub_i32 s23, s1, s22 // (op 911) 


s_mul_i32 s1, s46, s11 // (op 911) 
s_add_u32 s24, s6, 64 // (op 911) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 911) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 911) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 911) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 911) 

s_sub_i32 s22, s24, s1 // (op 911) 

s_mul_i32 s24, s23, s22 // (op 911) 


s_sub_u32 s25, s19, s24 // (op 911) 


s_add_u32 s19, s25, s1 // (op 911) 


s_cmp_eq_i32 s18, 0 // (op 911) 
s_mov_b32 s1, scc // (op 911) 

s_cmp_eq_i32 s1, 0 // (op 911) 
s_cselect_b32 s18, s19, s0 // (op 911) 



v_lshl_add_u32 v13, s18, 6, v12 // (op 911) 


v_mul_lo_u32 v17, v13, s27 // (op 911) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 911) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 911) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 911) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 911) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 911) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 911) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 911) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 911) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 911) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 911) 

v_mov_b32 v12, v14 // (op 911) convert

s_mul_i32 s1, s7, 6 // (op 913) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 913) least significant
s_mul_hi_u32 s18, s6, 6 // (op 913) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 913) most significant: sum

s_mul_i32 s19, s26, s1 // (op 913) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 913) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 913) least significant
s_mul_hi_u32 s23, s26, s0 // (op 913) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 913) most significant: sum
s_add_u32 s19, s19, s23 // (op 913) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 913) 

s_mov_b32 s48, s28 // (op 913) reg expression
s_mov_b32 s49, s29 // (op 913) reg expression
s_mov_b32 s50, s0 // (op 913) reg expression
s_mov_b32 s51, 131072 // (op 913) call()

s_mul_hi_i32 s47, s16, s10 // (op 930) 
v_lshrrev_b32 v13, 1, v1 // (op 930) 
s_add_i32 s0, s47, s16 // (op 930) 

v_and_b32 v14, -2, v1 // (op 930) 
s_ashr_i32 s1, s0, 31 // (op 930) 
v_sub_u32 v15, v1, v14 // (op 930) 

s_cmp_eq_i32 s10, 0 // (op 930) 
s_mov_b32 s47, scc // (op 930) 
v_lshlrev_b32 v14, 6, v15 // (op 930) 

s_cmp_eq_i32 s47, 0 // (op 930) 
s_cselect_b32 s18, 0, -1 // (op 930) 

v_mov_b32 v17, 0 // (op 930) convert
v_mov_b32 v16, v14 // (op 930) convert

s_and_b32 s19, s13, 31 // (op 930) 
s_lshl_b32 s47, 1, s19 // (op 930) 
s_add_i32 s22, s47, s18 // (op 930) 


s_and_b32 s18, s1, s22 // (op 930) 


s_add_i32 s1, s0, s18 // (op 930) 


s_ashr_i32 s0, s1, s19 // (op 930) 


s_lshl_b32 s1, s13, 24 // (op 930) 
s_ashr_i32 s47, s1, 31 // (op 930) 

s_xor_b32 s1, s0, s47 // (op 930) 

s_sub_i32 s0, s1, s47 // (op 930) 


s_mul_hi_i32 s1, s0, s14 // (op 930) 
s_add_i32 s47, s1, s0 // (op 930) 

s_ashr_i32 s1, s47, 31 // (op 930) 
s_cmp_eq_i32 s14, 0 // (op 930) 
s_mov_b32 s18, scc // (op 930) 
s_cmp_eq_i32 s18, 0 // (op 930) 
s_cselect_b32 s19, 0, -1 // (op 930) 

s_and_b32 s18, s15, 31 // (op 930) 
s_lshl_b32 s22, 1, s18 // (op 930) 
s_add_i32 s23, s22, s19 // (op 930) 


s_and_b32 s19, s1, s23 // (op 930) 


s_add_i32 s1, s47, s19 // (op 930) 


s_ashr_i32 s19, s1, s18 // (op 930) 


s_lshl_b32 s1, s15, 24 // (op 930) 
s_ashr_i32 s18, s1, 31 // (op 930) 

s_xor_b32 s1, s19, s18 // (op 930) 

s_sub_i32 s19, s1, s18 // (op 930) 


s_mul_i32 s1, s0, s11 // (op 930) 

s_sub_u32 s0, s16, s1 // (op 930) 

s_mul_i32 s1, s19, s11 // (op 930) 
s_add_u32 s1, s1, s0 // (op 930) 


s_mul_hi_i32 s0, s16, s9 // (op 930) 
s_add_i32 s47, s0, s16 // (op 930) 

s_ashr_i32 s0, s47, 31 // (op 930) 
s_cmp_eq_i32 s9, 0 // (op 930) 
s_mov_b32 s18, scc // (op 930) 
s_cmp_eq_i32 s18, 0 // (op 930) 
s_cselect_b32 s19, 0, -1 // (op 930) 

s_and_b32 s18, s45, 31 // (op 930) 
s_lshl_b32 s22, 1, s18 // (op 930) 
s_add_i32 s23, s22, s19 // (op 930) 


s_and_b32 s19, s0, s23 // (op 930) 


s_add_i32 s0, s47, s19 // (op 930) 


s_ashr_i32 s19, s0, s18 // (op 930) 


s_lshl_b32 s0, s45, 24 // (op 930) 
s_ashr_i32 s18, s0, 31 // (op 930) 

s_xor_b32 s0, s19, s18 // (op 930) 

s_sub_i32 s19, s0, s18 // (op 930) 


s_add_u32 s22, s4, 16 // (op 930) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 930) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 930) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 930) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 930) 

s_mul_i32 s0, s11, s22 // (op 930) 

s_mul_i32 s18, s8, s0 // (op 930) 

s_mul_i32 s0, s19, s18 // (op 930) 

s_sub_u32 s18, s16, s0 // (op 930) 

s_mov_b32 s0, s18 // (op 930) 
s_mul_hi_i32 s47, s0, s12 // (op 930) 
s_add_i32 s22, s47, s0 // (op 930) 


s_ashr_i32 s0, s22, 31 // (op 930) 
s_cmp_eq_i32 s12, 0 // (op 930) 
s_mov_b32 s47, scc // (op 930) 
s_cmp_eq_i32 s47, 0 // (op 930) 
s_cselect_b32 s23, 0, -1 // (op 930) 

s_and_b32 s47, s44, 31 // (op 930) 
s_lshl_b32 s24, 1, s47 // (op 930) 
s_add_i32 s25, s24, s23 // (op 930) 


s_and_b32 s23, s0, s25 // (op 930) 


s_add_i32 s0, s22, s23 // (op 930) 


s_ashr_i32 s22, s0, s47 // (op 930) 


s_lshl_b32 s0, s44, 24 // (op 930) 
s_ashr_i32 s47, s0, 31 // (op 930) 

s_xor_b32 s0, s22, s47 // (op 930) 

s_sub_i32 s22, s0, s47 // (op 930) 


s_mul_i32 s0, s46, s11 // (op 930) 
s_add_u32 s24, s6, 64 // (op 930) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 930) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 930) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 930) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 6 // (op 930) 

s_sub_i32 s23, s24, s0 // (op 930) 

s_mul_i32 s47, s22, s23 // (op 930) 


s_sub_u32 s22, s18, s47 // (op 930) 


s_add_u32 s18, s22, s0 // (op 930) 


s_cmp_eq_i32 s19, 0 // (op 930) 
s_mov_b32 s0, scc // (op 930) 

s_cmp_eq_i32 s0, 0 // (op 930) 
s_cselect_b32 s19, s18, s1 // (op 930) 



v_lshl_add_u32 v14, s19, 6, v13 // (op 930) 


v_mul_lo_u32 v19, v14, s27 // (op 930) most significant: low of low * high
v_mul_hi_u32 v13, v14, s26 // (op 930) most significant: high of low * low
v_mul_lo_u32 v18, v14, s26 // (op 930) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 930) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 930) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 930) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 930) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 930) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 930) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 930) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 930) 

v_mov_b32 v13, v14 // (op 930) convert

s_mul_i32 s1, s7, 6 // (op 932) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 932) least significant
s_mul_hi_u32 s47, s6, 6 // (op 932) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 932) most significant: sum

s_mul_i32 s19, s26, s1 // (op 932) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 932) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 932) least significant
s_mul_hi_u32 s22, s26, s0 // (op 932) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 932) most significant: sum
s_add_u32 s19, s19, s22 // (op 932) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 932) 

s_mov_b32 s48, s28 // (op 932) reg expression
s_mov_b32 s49, s29 // (op 932) reg expression
s_mov_b32 s50, s0 // (op 932) reg expression
s_mov_b32 s51, 131072 // (op 932) call()

s_mul_hi_i32 s47, s16, s10 // (op 975) 
v_lshrrev_b32 v14, 3, v1 // (op 975) 
s_add_i32 s0, s47, s16 // (op 975) 

v_and_b32 v15, -8, v1 // (op 975) 
s_ashr_i32 s1, s0, 31 // (op 975) 
v_sub_u32 v16, v1, v15 // (op 975) 

s_cmp_eq_i32 s10, 0 // (op 975) 
s_mov_b32 s47, scc // (op 975) 
v_lshlrev_b32 v15, 4, v16 // (op 975) 

s_cmp_eq_i32 s47, 0 // (op 975) 
s_cselect_b32 s18, 0, -1 // (op 975) 

v_mov_b32 v17, 0 // (op 975) convert
v_mov_b32 v16, v15 // (op 975) convert

s_and_b32 s19, s13, 31 // (op 975) 
s_lshl_b32 s47, 1, s19 // (op 975) 
s_add_i32 s22, s47, s18 // (op 975) 


s_and_b32 s18, s1, s22 // (op 975) 


s_add_i32 s1, s0, s18 // (op 975) 


s_ashr_i32 s0, s1, s19 // (op 975) 


s_lshl_b32 s1, s13, 24 // (op 975) 
s_ashr_i32 s47, s1, 31 // (op 975) 

s_xor_b32 s1, s0, s47 // (op 975) 

s_sub_i32 s0, s1, s47 // (op 975) 


s_mul_hi_i32 s1, s0, s14 // (op 975) 
s_add_i32 s47, s1, s0 // (op 975) 

s_ashr_i32 s1, s47, 31 // (op 975) 
s_cmp_eq_i32 s14, 0 // (op 975) 
s_mov_b32 s18, scc // (op 975) 
s_cmp_eq_i32 s18, 0 // (op 975) 
s_cselect_b32 s19, 0, -1 // (op 975) 

s_and_b32 s18, s15, 31 // (op 975) 
s_lshl_b32 s22, 1, s18 // (op 975) 
s_add_i32 s23, s22, s19 // (op 975) 


s_and_b32 s19, s1, s23 // (op 975) 


s_add_i32 s1, s47, s19 // (op 975) 


s_ashr_i32 s19, s1, s18 // (op 975) 


s_lshl_b32 s1, s15, 24 // (op 975) 
s_ashr_i32 s18, s1, 31 // (op 975) 

s_xor_b32 s1, s19, s18 // (op 975) 

s_sub_i32 s19, s1, s18 // (op 975) 


s_add_u32 s22, s4, 16 // (op 975) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 975) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 975) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 975) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 975) 

s_mul_i32 s1, s19, s22 // (op 975) 

s_sub_i32 s47, s0, s1 // (op 975) 


s_mul_hi_i32 s0, s16, s9 // (op 975) 
s_add_i32 s1, s0, s16 // (op 975) 

s_ashr_i32 s0, s1, 31 // (op 975) 
s_cmp_eq_i32 s9, 0 // (op 975) 
s_mov_b32 s18, scc // (op 975) 
s_cmp_eq_i32 s18, 0 // (op 975) 
s_cselect_b32 s19, 0, -1 // (op 975) 

s_and_b32 s18, s45, 31 // (op 975) 
s_lshl_b32 s24, 1, s18 // (op 975) 
s_add_i32 s25, s24, s19 // (op 975) 


s_and_b32 s19, s0, s25 // (op 975) 


s_add_i32 s0, s1, s19 // (op 975) 


s_ashr_i32 s1, s0, s18 // (op 975) 


s_lshl_b32 s0, s45, 24 // (op 975) 
s_ashr_i32 s18, s0, 31 // (op 975) 

s_xor_b32 s0, s1, s18 // (op 975) 

s_sub_i32 s1, s0, s18 // (op 975) 


s_mul_i32 s0, s11, s22 // (op 975) 

s_mul_i32 s18, s8, s0 // (op 975) 

s_mul_i32 s0, s1, s18 // (op 975) 

s_sub_u32 s18, s16, s0 // (op 975) 

s_mul_hi_i32 s0, s18, s12 // (op 975) 
s_add_i32 s19, s0, s18 // (op 975) 


s_ashr_i32 s0, s19, 31 // (op 975) 
s_cmp_eq_i32 s12, 0 // (op 975) 
s_mov_b32 s18, scc // (op 975) 
s_cmp_eq_i32 s18, 0 // (op 975) 
s_cselect_b32 s22, 0, -1 // (op 975) 

s_and_b32 s18, s44, 31 // (op 975) 
s_lshl_b32 s23, 1, s18 // (op 975) 
s_add_i32 s24, s23, s22 // (op 975) 


s_and_b32 s25, s0, s24 // (op 975) 


s_add_i32 s0, s19, s25 // (op 975) 


s_ashr_i32 s19, s0, s18 // (op 975) 


s_lshl_b32 s0, s44, 24 // (op 975) 
s_ashr_i32 s18, s0, 31 // (op 975) 

s_xor_b32 s0, s19, s18 // (op 975) 

s_sub_i32 s19, s0, s18 // (op 975) 


s_cmp_eq_i32 s1, 0 // (op 975) 
s_mov_b32 s0, scc // (op 975) 

s_cmp_eq_i32 s0, 0 // (op 975) 
s_cselect_b32 s1, s19, s47 // (op 975) 



v_lshl_add_u32 v15, s1, 4, v14 // (op 975) 


v_mul_lo_u32 v19, v15, s31 // (op 975) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 975) most significant: high of low * low
v_mul_lo_u32 v18, v15, s30 // (op 975) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 975) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 975) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 975) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 975) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 975) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 975) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 975) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 975) 

v_mov_b32 v16, v14 // (op 975) convert

s_mul_i32 s1, s5, 6 // (op 977) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 977) least significant
s_mul_hi_u32 s47, s4, 6 // (op 977) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 977) most significant: sum

s_mul_i32 s19, s30, s1 // (op 977) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 977) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 977) least significant
s_mul_hi_u32 s22, s30, s0 // (op 977) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 977) most significant: sum
s_add_u32 s19, s19, s22 // (op 977) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 977) 

s_mov_b32 s60, s32 // (op 977) reg expression
s_mov_b32 s61, s33 // (op 977) reg expression
s_mov_b32 s62, s0 // (op 977) reg expression
s_mov_b32 s63, 131072 // (op 977) call()

s_mul_hi_i32 s47, s16, s10 // (op 994) 
v_lshrrev_b32 v14, 3, v1 // (op 994) 
s_add_i32 s0, s47, s16 // (op 994) 

v_and_b32 v15, -8, v1 // (op 994) 
s_ashr_i32 s1, s0, 31 // (op 994) 
v_sub_u32 v17, v1, v15 // (op 994) 

s_cmp_eq_i32 s10, 0 // (op 994) 
s_mov_b32 s47, scc // (op 994) 
v_lshlrev_b32 v15, 4, v17 // (op 994) 

s_cmp_eq_i32 s47, 0 // (op 994) 
s_cselect_b32 s18, 0, -1 // (op 994) 

v_mov_b32 v19, 0 // (op 994) convert
v_mov_b32 v18, v15 // (op 994) convert

s_and_b32 s19, s13, 31 // (op 994) 
s_lshl_b32 s47, 1, s19 // (op 994) 
s_add_i32 s22, s47, s18 // (op 994) 


s_and_b32 s18, s1, s22 // (op 994) 


s_add_i32 s1, s0, s18 // (op 994) 


s_ashr_i32 s0, s1, s19 // (op 994) 


s_lshl_b32 s1, s13, 24 // (op 994) 
s_ashr_i32 s47, s1, 31 // (op 994) 

s_xor_b32 s1, s0, s47 // (op 994) 

s_sub_i32 s0, s1, s47 // (op 994) 


s_mul_hi_i32 s1, s0, s14 // (op 994) 
s_add_i32 s47, s1, s0 // (op 994) 

s_ashr_i32 s1, s47, 31 // (op 994) 
s_cmp_eq_i32 s14, 0 // (op 994) 
s_mov_b32 s18, scc // (op 994) 
s_cmp_eq_i32 s18, 0 // (op 994) 
s_cselect_b32 s19, 0, -1 // (op 994) 

s_and_b32 s18, s15, 31 // (op 994) 
s_lshl_b32 s22, 1, s18 // (op 994) 
s_add_i32 s23, s22, s19 // (op 994) 


s_and_b32 s19, s1, s23 // (op 994) 


s_add_i32 s1, s47, s19 // (op 994) 


s_ashr_i32 s19, s1, s18 // (op 994) 


s_lshl_b32 s1, s15, 24 // (op 994) 
s_ashr_i32 s18, s1, 31 // (op 994) 

s_xor_b32 s1, s19, s18 // (op 994) 

s_sub_i32 s19, s1, s18 // (op 994) 


s_add_u32 s22, s4, 16 // (op 994) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 994) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 994) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 994) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 994) 

s_mul_i32 s1, s19, s22 // (op 994) 

s_sub_i32 s47, s0, s1 // (op 994) 


s_mul_hi_i32 s0, s16, s9 // (op 994) 
s_add_i32 s1, s0, s16 // (op 994) 

s_ashr_i32 s0, s1, 31 // (op 994) 
s_cmp_eq_i32 s9, 0 // (op 994) 
s_mov_b32 s18, scc // (op 994) 
s_cmp_eq_i32 s18, 0 // (op 994) 
s_cselect_b32 s19, 0, -1 // (op 994) 

s_and_b32 s18, s45, 31 // (op 994) 
s_lshl_b32 s24, 1, s18 // (op 994) 
s_add_i32 s25, s24, s19 // (op 994) 


s_and_b32 s19, s0, s25 // (op 994) 


s_add_i32 s0, s1, s19 // (op 994) 


s_ashr_i32 s1, s0, s18 // (op 994) 


s_lshl_b32 s0, s45, 24 // (op 994) 
s_ashr_i32 s18, s0, 31 // (op 994) 

s_xor_b32 s0, s1, s18 // (op 994) 

s_sub_i32 s1, s0, s18 // (op 994) 


s_mul_i32 s0, s11, s22 // (op 994) 

s_mul_i32 s18, s8, s0 // (op 994) 

s_mul_i32 s0, s1, s18 // (op 994) 

s_sub_u32 s18, s16, s0 // (op 994) 

s_mul_hi_i32 s0, s18, s12 // (op 994) 
s_add_i32 s19, s0, s18 // (op 994) 


s_ashr_i32 s0, s19, 31 // (op 994) 
s_cmp_eq_i32 s12, 0 // (op 994) 
s_mov_b32 s18, scc // (op 994) 
s_cmp_eq_i32 s18, 0 // (op 994) 
s_cselect_b32 s22, 0, -1 // (op 994) 

s_and_b32 s18, s44, 31 // (op 994) 
s_lshl_b32 s23, 1, s18 // (op 994) 
s_add_i32 s24, s23, s22 // (op 994) 


s_and_b32 s25, s0, s24 // (op 994) 


s_add_i32 s0, s19, s25 // (op 994) 


s_ashr_i32 s19, s0, s18 // (op 994) 


s_lshl_b32 s0, s44, 24 // (op 994) 
s_ashr_i32 s18, s0, 31 // (op 994) 

s_xor_b32 s0, s19, s18 // (op 994) 

s_sub_i32 s19, s0, s18 // (op 994) 


s_cmp_eq_i32 s1, 0 // (op 994) 
s_mov_b32 s0, scc // (op 994) 

s_cmp_eq_i32 s0, 0 // (op 994) 
s_cselect_b32 s1, s19, s47 // (op 994) 



v_lshl_add_u32 v15, s1, 4, v14 // (op 994) 


v_mul_lo_u32 v21, v15, s31 // (op 994) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 994) most significant: high of low * low
v_mul_lo_u32 v20, v15, s30 // (op 994) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 994) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 994) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 994) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 994) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 994) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 994) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 994) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 994) 

v_mov_b32 v17, v14 // (op 994) convert

s_mul_i32 s1, s5, 6 // (op 996) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 996) least significant
s_mul_hi_u32 s47, s4, 6 // (op 996) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 996) most significant: sum

s_mul_i32 s19, s30, s1 // (op 996) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 996) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 996) least significant
s_mul_hi_u32 s22, s30, s0 // (op 996) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 996) most significant: sum
s_add_u32 s19, s19, s22 // (op 996) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 996) 

s_mov_b32 s60, s32 // (op 996) reg expression
s_mov_b32 s61, s33 // (op 996) reg expression
s_mov_b32 s62, s0 // (op 996) reg expression
s_mov_b32 s63, 131072 // (op 996) call()

s_mul_hi_i32 s47, s16, s10 // (op 1039) 
v_lshrrev_b32 v14, 2, v1 // (op 1039) 
s_add_i32 s0, s47, s16 // (op 1039) 

v_lshlrev_b32 v15, 1, v14 // (op 1039) 

s_ashr_i32 s1, s0, 31 // (op 1039) 
v_and_b32 v14, -4, v1 // (op 1039) 
s_cmp_eq_i32 s10, 0 // (op 1039) 
s_mov_b32 s47, scc // (op 1039) 
v_sub_u32 v18, v1, v14 // (op 1039) 

s_cmp_eq_i32 s47, 0 // (op 1039) 
s_cselect_b32 s18, 0, -1 // (op 1039) 

s_and_b32 s19, s13, 31 // (op 1039) 
s_lshl_b32 s47, 1, s19 // (op 1039) 
s_add_i32 s22, s47, s18 // (op 1039) 


s_and_b32 s18, s1, s22 // (op 1039) 


s_add_i32 s1, s0, s18 // (op 1039) 


s_ashr_i32 s0, s1, s19 // (op 1039) 


s_lshl_b32 s1, s13, 24 // (op 1039) 
s_ashr_i32 s47, s1, 31 // (op 1039) 

s_xor_b32 s1, s0, s47 // (op 1039) 

s_sub_i32 s0, s1, s47 // (op 1039) 


s_mul_hi_i32 s1, s0, s14 // (op 1039) 
s_add_i32 s47, s1, s0 // (op 1039) 

s_ashr_i32 s1, s47, 31 // (op 1039) 
s_cmp_eq_i32 s14, 0 // (op 1039) 
s_mov_b32 s18, scc // (op 1039) 
s_cmp_eq_i32 s18, 0 // (op 1039) 
s_cselect_b32 s19, 0, -1 // (op 1039) 

s_and_b32 s18, s15, 31 // (op 1039) 
s_lshl_b32 s22, 1, s18 // (op 1039) 
s_add_i32 s23, s22, s19 // (op 1039) 


s_and_b32 s19, s1, s23 // (op 1039) 


s_add_i32 s1, s47, s19 // (op 1039) 


s_ashr_i32 s19, s1, s18 // (op 1039) 


s_lshl_b32 s1, s15, 24 // (op 1039) 
s_ashr_i32 s18, s1, 31 // (op 1039) 

s_xor_b32 s1, s19, s18 // (op 1039) 

s_sub_i32 s19, s1, s18 // (op 1039) 


s_mul_i32 s1, s0, s11 // (op 1039) 

s_sub_u32 s0, s16, s1 // (op 1039) 

s_mul_i32 s1, s19, s11 // (op 1039) 
s_add_u32 s1, s1, s0 // (op 1039) 


s_mul_hi_i32 s0, s16, s9 // (op 1039) 
s_add_i32 s47, s0, s16 // (op 1039) 

s_ashr_i32 s0, s47, 31 // (op 1039) 
s_cmp_eq_i32 s9, 0 // (op 1039) 
s_mov_b32 s18, scc // (op 1039) 
s_cmp_eq_i32 s18, 0 // (op 1039) 
s_cselect_b32 s19, 0, -1 // (op 1039) 

s_and_b32 s18, s45, 31 // (op 1039) 
s_lshl_b32 s22, 1, s18 // (op 1039) 
s_add_i32 s23, s22, s19 // (op 1039) 


s_and_b32 s19, s0, s23 // (op 1039) 


s_add_i32 s0, s47, s19 // (op 1039) 


s_ashr_i32 s19, s0, s18 // (op 1039) 


s_lshl_b32 s0, s45, 24 // (op 1039) 
s_ashr_i32 s18, s0, 31 // (op 1039) 

s_xor_b32 s0, s19, s18 // (op 1039) 

s_sub_i32 s19, s0, s18 // (op 1039) 


s_add_u32 s22, s4, 16 // (op 1039) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1039) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1039) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1039) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1039) 

s_mul_i32 s0, s11, s22 // (op 1039) 

s_mul_i32 s18, s8, s0 // (op 1039) 

s_mul_i32 s0, s19, s18 // (op 1039) 

s_sub_u32 s18, s16, s0 // (op 1039) 

s_mov_b32 s0, s18 // (op 1039) 
s_mul_hi_i32 s47, s0, s12 // (op 1039) 
s_add_i32 s22, s47, s0 // (op 1039) 


s_ashr_i32 s0, s22, 31 // (op 1039) 
s_cmp_eq_i32 s12, 0 // (op 1039) 
s_mov_b32 s47, scc // (op 1039) 
s_cmp_eq_i32 s47, 0 // (op 1039) 
s_cselect_b32 s23, 0, -1 // (op 1039) 

s_and_b32 s47, s44, 31 // (op 1039) 
s_lshl_b32 s24, 1, s47 // (op 1039) 
s_add_i32 s25, s24, s23 // (op 1039) 


s_and_b32 s23, s0, s25 // (op 1039) 


s_add_i32 s0, s22, s23 // (op 1039) 


s_ashr_i32 s22, s0, s47 // (op 1039) 


s_lshl_b32 s0, s44, 24 // (op 1039) 
s_ashr_i32 s47, s0, 31 // (op 1039) 

s_xor_b32 s0, s22, s47 // (op 1039) 

s_sub_i32 s22, s0, s47 // (op 1039) 


s_mul_i32 s0, s46, s11 // (op 1039) 
s_add_u32 s24, s6, 64 // (op 1039) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1039) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 1039) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 1039) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 6 // (op 1039) 

s_sub_i32 s23, s24, s0 // (op 1039) 

s_mul_i32 s47, s22, s23 // (op 1039) 


s_sub_u32 s22, s18, s47 // (op 1039) 


s_add_u32 s18, s22, s0 // (op 1039) 


s_cmp_eq_i32 s19, 0 // (op 1039) 
s_mov_b32 s0, scc // (op 1039) 

s_cmp_eq_i32 s0, 0 // (op 1039) 
s_cselect_b32 s19, s18, s1 // (op 1039) 



v_lshl_add_u32 v14, s19, 6, v15 // (op 1039) 


v_mul_lo_u32 v15, v14, s34 // (op 1039) 
v_add_u32 v15, v15, v18 // (op 1039) 


s_mul_i32 s1, s34, s7 // (op 1041) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1041) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1041) most significant: sum
s_add_u32 s1, s1, s18 // (op 1041) most significant: sum


s_mov_b32 s64, s36 // (op 1041) reg expression
s_mov_b32 s65, s37 // (op 1041) reg expression
s_mov_b32 s66, s0 // (op 1041) reg expression
s_mov_b32 s67, 131072 // (op 1041) call()

s_mul_hi_i32 s47, s16, s10 // (op 1058) 
v_lshrrev_b32 v14, 2, v1 // (op 1058) 
s_add_i32 s0, s47, s16 // (op 1058) 

v_lshlrev_b32 v18, 1, v14 // (op 1058) 

s_ashr_i32 s1, s0, 31 // (op 1058) 
v_and_b32 v14, -4, v1 // (op 1058) 
s_cmp_eq_i32 s10, 0 // (op 1058) 
s_mov_b32 s47, scc // (op 1058) 
v_sub_u32 v19, v1, v14 // (op 1058) 

s_cmp_eq_i32 s47, 0 // (op 1058) 
s_cselect_b32 s18, 0, -1 // (op 1058) 

s_and_b32 s19, s13, 31 // (op 1058) 
s_lshl_b32 s47, 1, s19 // (op 1058) 
s_add_i32 s22, s47, s18 // (op 1058) 


s_and_b32 s18, s1, s22 // (op 1058) 


s_add_i32 s1, s0, s18 // (op 1058) 


s_ashr_i32 s0, s1, s19 // (op 1058) 


s_lshl_b32 s1, s13, 24 // (op 1058) 
s_ashr_i32 s47, s1, 31 // (op 1058) 

s_xor_b32 s1, s0, s47 // (op 1058) 

s_sub_i32 s0, s1, s47 // (op 1058) 


s_mul_hi_i32 s1, s0, s14 // (op 1058) 
s_add_i32 s47, s1, s0 // (op 1058) 

s_ashr_i32 s1, s47, 31 // (op 1058) 
s_cmp_eq_i32 s14, 0 // (op 1058) 
s_mov_b32 s18, scc // (op 1058) 
s_cmp_eq_i32 s18, 0 // (op 1058) 
s_cselect_b32 s19, 0, -1 // (op 1058) 

s_and_b32 s18, s15, 31 // (op 1058) 
s_lshl_b32 s22, 1, s18 // (op 1058) 
s_add_i32 s23, s22, s19 // (op 1058) 


s_and_b32 s19, s1, s23 // (op 1058) 


s_add_i32 s1, s47, s19 // (op 1058) 


s_ashr_i32 s19, s1, s18 // (op 1058) 


s_lshl_b32 s1, s15, 24 // (op 1058) 
s_ashr_i32 s18, s1, 31 // (op 1058) 

s_xor_b32 s1, s19, s18 // (op 1058) 

s_sub_i32 s19, s1, s18 // (op 1058) 


s_mul_i32 s1, s0, s11 // (op 1058) 

s_sub_u32 s0, s16, s1 // (op 1058) 

s_mul_i32 s1, s19, s11 // (op 1058) 
s_add_u32 s1, s1, s0 // (op 1058) 


s_mul_hi_i32 s0, s16, s9 // (op 1058) 
s_add_i32 s47, s0, s16 // (op 1058) 

s_ashr_i32 s0, s47, 31 // (op 1058) 
s_cmp_eq_i32 s9, 0 // (op 1058) 
s_mov_b32 s18, scc // (op 1058) 
s_cmp_eq_i32 s18, 0 // (op 1058) 
s_cselect_b32 s19, 0, -1 // (op 1058) 

s_and_b32 s18, s45, 31 // (op 1058) 
s_lshl_b32 s22, 1, s18 // (op 1058) 
s_add_i32 s23, s22, s19 // (op 1058) 


s_and_b32 s19, s0, s23 // (op 1058) 


s_add_i32 s0, s47, s19 // (op 1058) 


s_ashr_i32 s19, s0, s18 // (op 1058) 


s_lshl_b32 s0, s45, 24 // (op 1058) 
s_ashr_i32 s18, s0, 31 // (op 1058) 

s_xor_b32 s0, s19, s18 // (op 1058) 

s_sub_i32 s19, s0, s18 // (op 1058) 


s_add_u32 s22, s4, 16 // (op 1058) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1058) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1058) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1058) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1058) 

s_mul_i32 s0, s11, s22 // (op 1058) 

s_mul_i32 s18, s8, s0 // (op 1058) 

s_mul_i32 s0, s19, s18 // (op 1058) 

s_sub_u32 s18, s16, s0 // (op 1058) 

s_mov_b32 s0, s18 // (op 1058) 
s_mul_hi_i32 s47, s0, s12 // (op 1058) 
s_add_i32 s22, s47, s0 // (op 1058) 


s_ashr_i32 s0, s22, 31 // (op 1058) 
s_cmp_eq_i32 s12, 0 // (op 1058) 
s_mov_b32 s47, scc // (op 1058) 
s_cmp_eq_i32 s47, 0 // (op 1058) 
s_cselect_b32 s23, 0, -1 // (op 1058) 

s_and_b32 s47, s44, 31 // (op 1058) 
s_lshl_b32 s24, 1, s47 // (op 1058) 
s_add_i32 s25, s24, s23 // (op 1058) 


s_and_b32 s23, s0, s25 // (op 1058) 


s_add_i32 s0, s22, s23 // (op 1058) 


s_ashr_i32 s22, s0, s47 // (op 1058) 


s_lshl_b32 s0, s44, 24 // (op 1058) 
s_ashr_i32 s47, s0, 31 // (op 1058) 

s_xor_b32 s0, s22, s47 // (op 1058) 

s_sub_i32 s22, s0, s47 // (op 1058) 


s_mul_i32 s0, s46, s11 // (op 1058) 
s_add_u32 s24, s6, 64 // (op 1058) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1058) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 1058) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 1058) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 6 // (op 1058) 

s_sub_i32 s23, s24, s0 // (op 1058) 

s_mul_i32 s47, s22, s23 // (op 1058) 


s_sub_u32 s22, s18, s47 // (op 1058) 


s_add_u32 s18, s22, s0 // (op 1058) 


s_cmp_eq_i32 s19, 0 // (op 1058) 
s_mov_b32 s0, scc // (op 1058) 

s_cmp_eq_i32 s0, 0 // (op 1058) 
s_cselect_b32 s19, s18, s1 // (op 1058) 



v_lshl_add_u32 v14, s19, 6, v18 // (op 1058) 


v_mul_lo_u32 v18, v14, s34 // (op 1058) 
v_add_u32 v18, v18, v19 // (op 1058) 


s_mul_i32 s1, s34, s7 // (op 1060) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1060) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1060) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1060) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1060) most significant: sum
s_add_u32 s1, s1, s18 // (op 1060) most significant: sum


s_mov_b32 s64, s36 // (op 1060) reg expression
s_mov_b32 s65, s37 // (op 1060) reg expression
s_mov_b32 s66, s0 // (op 1060) reg expression
s_mov_b32 s67, 131072 // (op 1060) call()

s_mul_hi_i32 s47, s16, s10 // (op 1103) 
v_and_b32 v14, 15, v1 // (op 1103) 
s_add_i32 s0, s47, s16 // (op 1103) 

v_and_b32 v19, 63, v1 // (op 1103) 
s_ashr_i32 s1, s0, 31 // (op 1103) 
v_lshrrev_b32 v20, 4, v19 // (op 1103) 

s_cmp_eq_i32 s10, 0 // (op 1103) 
s_mov_b32 s47, scc // (op 1103) 
s_cmp_eq_i32 s47, 0 // (op 1103) 
s_cselect_b32 s18, 0, -1 // (op 1103) 

s_and_b32 s19, s13, 31 // (op 1103) 
s_lshl_b32 s47, 1, s19 // (op 1103) 
s_add_i32 s22, s47, s18 // (op 1103) 


s_and_b32 s18, s1, s22 // (op 1103) 


s_add_i32 s1, s0, s18 // (op 1103) 


s_ashr_i32 s0, s1, s19 // (op 1103) 


s_lshl_b32 s1, s13, 24 // (op 1103) 
s_ashr_i32 s47, s1, 31 // (op 1103) 

s_xor_b32 s1, s0, s47 // (op 1103) 

s_sub_i32 s0, s1, s47 // (op 1103) 


s_mul_hi_i32 s1, s0, s14 // (op 1103) 
s_add_i32 s47, s1, s0 // (op 1103) 

s_ashr_i32 s1, s47, 31 // (op 1103) 
s_cmp_eq_i32 s14, 0 // (op 1103) 
s_mov_b32 s18, scc // (op 1103) 
s_cmp_eq_i32 s18, 0 // (op 1103) 
s_cselect_b32 s19, 0, -1 // (op 1103) 

s_and_b32 s18, s15, 31 // (op 1103) 
s_lshl_b32 s22, 1, s18 // (op 1103) 
s_add_i32 s23, s22, s19 // (op 1103) 


s_and_b32 s19, s1, s23 // (op 1103) 


s_add_i32 s1, s47, s19 // (op 1103) 


s_ashr_i32 s19, s1, s18 // (op 1103) 


s_lshl_b32 s1, s15, 24 // (op 1103) 
s_ashr_i32 s18, s1, 31 // (op 1103) 

s_xor_b32 s1, s19, s18 // (op 1103) 

s_sub_i32 s19, s1, s18 // (op 1103) 


s_add_u32 s22, s4, 16 // (op 1103) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1103) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1103) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1103) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1103) 

s_mul_i32 s1, s19, s22 // (op 1103) 

s_sub_i32 s47, s0, s1 // (op 1103) 


s_mul_hi_i32 s0, s16, s9 // (op 1103) 
s_add_i32 s1, s0, s16 // (op 1103) 

s_ashr_i32 s0, s1, 31 // (op 1103) 
s_cmp_eq_i32 s9, 0 // (op 1103) 
s_mov_b32 s18, scc // (op 1103) 
s_cmp_eq_i32 s18, 0 // (op 1103) 
s_cselect_b32 s19, 0, -1 // (op 1103) 

s_and_b32 s18, s45, 31 // (op 1103) 
s_lshl_b32 s24, 1, s18 // (op 1103) 
s_add_i32 s25, s24, s19 // (op 1103) 


s_and_b32 s19, s0, s25 // (op 1103) 


s_add_i32 s0, s1, s19 // (op 1103) 


s_ashr_i32 s1, s0, s18 // (op 1103) 


s_lshl_b32 s0, s45, 24 // (op 1103) 
s_ashr_i32 s18, s0, 31 // (op 1103) 

s_xor_b32 s0, s1, s18 // (op 1103) 

s_sub_i32 s1, s0, s18 // (op 1103) 


s_mul_i32 s0, s11, s22 // (op 1103) 

s_mul_i32 s18, s8, s0 // (op 1103) 

s_mul_i32 s0, s1, s18 // (op 1103) 

s_sub_u32 s18, s16, s0 // (op 1103) 

s_mul_hi_i32 s0, s18, s12 // (op 1103) 
s_add_i32 s19, s0, s18 // (op 1103) 


s_ashr_i32 s0, s19, 31 // (op 1103) 
s_cmp_eq_i32 s12, 0 // (op 1103) 
s_mov_b32 s18, scc // (op 1103) 
s_cmp_eq_i32 s18, 0 // (op 1103) 
s_cselect_b32 s22, 0, -1 // (op 1103) 

s_and_b32 s18, s44, 31 // (op 1103) 
s_lshl_b32 s23, 1, s18 // (op 1103) 
s_add_i32 s24, s23, s22 // (op 1103) 


s_and_b32 s25, s0, s24 // (op 1103) 


s_add_i32 s0, s19, s25 // (op 1103) 


s_ashr_i32 s19, s0, s18 // (op 1103) 


s_lshl_b32 s0, s44, 24 // (op 1103) 
s_ashr_i32 s18, s0, 31 // (op 1103) 

s_xor_b32 s0, s19, s18 // (op 1103) 

s_sub_i32 s19, s0, s18 // (op 1103) 


s_cmp_eq_i32 s1, 0 // (op 1103) 
s_mov_b32 s0, scc // (op 1103) 

s_cmp_eq_i32 s0, 0 // (op 1103) 
s_cselect_b32 s1, s19, s47 // (op 1103) 



v_lshl_add_u32 v19, s1, 4, v14 // (op 1103) 


v_mul_lo_u32 v14, v19, s38 // (op 1103) 

v_add_u32 v19, v20, v14 // (op 1103) 


s_mul_i32 s1, s38, s5 // (op 1105) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1105) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1105) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1105) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1105) most significant: sum
s_add_u32 s1, s1, s18 // (op 1105) most significant: sum


s_mov_b32 s68, s40 // (op 1105) reg expression
s_mov_b32 s69, s41 // (op 1105) reg expression
s_mov_b32 s70, s0 // (op 1105) reg expression
s_mov_b32 s71, 131072 // (op 1105) call()

s_mul_hi_i32 s47, s16, s10 // (op 1122) 
v_and_b32 v14, 15, v1 // (op 1122) 
s_add_i32 s0, s47, s16 // (op 1122) 

v_and_b32 v20, 63, v1 // (op 1122) 
s_ashr_i32 s1, s0, 31 // (op 1122) 
v_lshrrev_b32 v21, 4, v20 // (op 1122) 

s_cmp_eq_i32 s10, 0 // (op 1122) 
s_mov_b32 s47, scc // (op 1122) 
s_cmp_eq_i32 s47, 0 // (op 1122) 
s_cselect_b32 s18, 0, -1 // (op 1122) 

s_and_b32 s19, s13, 31 // (op 1122) 
s_lshl_b32 s47, 1, s19 // (op 1122) 
s_add_i32 s22, s47, s18 // (op 1122) 


s_and_b32 s18, s1, s22 // (op 1122) 


s_add_i32 s1, s0, s18 // (op 1122) 


s_ashr_i32 s0, s1, s19 // (op 1122) 


s_lshl_b32 s1, s13, 24 // (op 1122) 
s_ashr_i32 s47, s1, 31 // (op 1122) 

s_xor_b32 s1, s0, s47 // (op 1122) 

s_sub_i32 s0, s1, s47 // (op 1122) 


s_mul_hi_i32 s1, s0, s14 // (op 1122) 
s_add_i32 s47, s1, s0 // (op 1122) 

s_ashr_i32 s1, s47, 31 // (op 1122) 
s_cmp_eq_i32 s14, 0 // (op 1122) 
s_mov_b32 s18, scc // (op 1122) 
s_cmp_eq_i32 s18, 0 // (op 1122) 
s_cselect_b32 s19, 0, -1 // (op 1122) 

s_and_b32 s18, s15, 31 // (op 1122) 
s_lshl_b32 s22, 1, s18 // (op 1122) 
s_add_i32 s23, s22, s19 // (op 1122) 


s_and_b32 s19, s1, s23 // (op 1122) 


s_add_i32 s1, s47, s19 // (op 1122) 


s_ashr_i32 s19, s1, s18 // (op 1122) 


s_lshl_b32 s1, s15, 24 // (op 1122) 
s_ashr_i32 s18, s1, 31 // (op 1122) 

s_xor_b32 s1, s19, s18 // (op 1122) 

s_sub_i32 s19, s1, s18 // (op 1122) 


s_add_u32 s22, s4, 16 // (op 1122) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1122) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1122) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1122) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1122) 

s_mul_i32 s1, s19, s22 // (op 1122) 

s_sub_i32 s47, s0, s1 // (op 1122) 


s_mul_hi_i32 s0, s16, s9 // (op 1122) 
s_add_i32 s1, s0, s16 // (op 1122) 

s_ashr_i32 s0, s1, 31 // (op 1122) 
s_cmp_eq_i32 s9, 0 // (op 1122) 
s_mov_b32 s18, scc // (op 1122) 
s_cmp_eq_i32 s18, 0 // (op 1122) 
s_cselect_b32 s19, 0, -1 // (op 1122) 

s_and_b32 s18, s45, 31 // (op 1122) 
s_lshl_b32 s24, 1, s18 // (op 1122) 
s_add_i32 s25, s24, s19 // (op 1122) 


s_and_b32 s19, s0, s25 // (op 1122) 


s_add_i32 s0, s1, s19 // (op 1122) 


s_ashr_i32 s1, s0, s18 // (op 1122) 


s_lshl_b32 s0, s45, 24 // (op 1122) 
s_ashr_i32 s18, s0, 31 // (op 1122) 

s_xor_b32 s0, s1, s18 // (op 1122) 

s_sub_i32 s1, s0, s18 // (op 1122) 


s_mul_i32 s0, s11, s22 // (op 1122) 

s_mul_i32 s18, s8, s0 // (op 1122) 

s_mul_i32 s0, s1, s18 // (op 1122) 

s_sub_u32 s18, s16, s0 // (op 1122) 

s_mul_hi_i32 s0, s18, s12 // (op 1122) 
s_add_i32 s19, s0, s18 // (op 1122) 


s_ashr_i32 s0, s19, 31 // (op 1122) 
s_cmp_eq_i32 s12, 0 // (op 1122) 
s_mov_b32 s18, scc // (op 1122) 
s_cmp_eq_i32 s18, 0 // (op 1122) 
s_cselect_b32 s22, 0, -1 // (op 1122) 

s_and_b32 s18, s44, 31 // (op 1122) 
s_lshl_b32 s23, 1, s18 // (op 1122) 
s_add_i32 s24, s23, s22 // (op 1122) 


s_and_b32 s25, s0, s24 // (op 1122) 


s_add_i32 s0, s19, s25 // (op 1122) 


s_ashr_i32 s19, s0, s18 // (op 1122) 


s_lshl_b32 s0, s44, 24 // (op 1122) 
s_ashr_i32 s18, s0, 31 // (op 1122) 

s_xor_b32 s0, s19, s18 // (op 1122) 

s_sub_i32 s19, s0, s18 // (op 1122) 


s_cmp_eq_i32 s1, 0 // (op 1122) 
s_mov_b32 s0, scc // (op 1122) 

s_cmp_eq_i32 s0, 0 // (op 1122) 
s_cselect_b32 s1, s19, s47 // (op 1122) 



v_lshl_add_u32 v20, s1, 4, v14 // (op 1122) 


v_mul_lo_u32 v14, v20, s38 // (op 1122) 

v_add_u32 v20, v21, v14 // (op 1122) 


s_mul_i32 s1, s38, s5 // (op 1124) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1124) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1124) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1124) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1124) most significant: sum
s_add_u32 s1, s1, s18 // (op 1124) most significant: sum


s_mov_b32 s68, s40 // (op 1124) reg expression
s_mov_b32 s69, s41 // (op 1124) reg expression
s_mov_b32 s70, s0 // (op 1124) reg expression
s_mov_b32 s71, 131072 // (op 1124) call()




buffer_load_ubyte v14, v19, s[68:71], 0 offen offset: 0 // (op 782) Load value



buffer_load_dwordx3 v[22:24], v16, s[60:63], 0 offen // (op 786) Load value



buffer_load_dwordx4 v[26:29], v12, s[48:51], 0 offen // (op 789) Load value
buffer_load_dwordx2 v[30:31], v12, s[48:51], 0 offen offset:16 // (op 789) Load value
buffer_load_dwordx4 v[32:35], v12, s[48:51], 0 offen offset:24 // (op 789) Load value
buffer_load_dwordx2 v[36:37], v12, s[48:51], 0 offen offset:40 // (op 789) Load value

v_mov_b64 v[38:39], s[34:35] // (op 792) convert

v_mov_b32 v21, v15 // (op 792) 
buffer_load_ubyte v40, v21, s[64:67], 0 offen offset: 0 // (op 792) Load value
v_add_u32 v21, v21, v38 // (op 792) 
buffer_load_ubyte v41, v21, s[64:67], 0 offen offset: 0 // (op 792) Load value





buffer_load_ubyte v21, v19, s[68:71], 0 offen offset:4 // (op 796) Load value




buffer_load_dwordx3 v[42:44], v16, s[60:63], 0 offen offset:96 // (op 800) Load value




buffer_load_dwordx4 v[46:49], v12, s[48:51], 0 offen offset:96 // (op 803) Load value
buffer_load_dwordx2 v[50:51], v12, s[48:51], 0 offen offset:112 // (op 803) Load value
buffer_load_dwordx4 v[52:55], v12, s[48:51], 0 offen offset:120 // (op 803) Load value
buffer_load_dwordx2 v[56:57], v12, s[48:51], 0 offen offset:136 // (op 803) Load value


v_mov_b64 v[38:39], s[34:35] // (op 806) convert

v_mov_b32 v12, v15 // (op 806) 
buffer_load_ubyte v58, v12, s[64:67], 0 offen offset:4 // (op 806) Load value
v_add_u32 v12, v12, v38 // (op 806) 
buffer_load_ubyte v59, v12, s[64:67], 0 offen offset:4 // (op 806) Load value



v_and_b32 v12, -8, v1 // (op 1247) 
v_sub_u32 v19, v1, v12 // (op 1247) 

v_lshlrev_b32 v12, 4, v19 // (op 1247) 

v_lshrrev_b32 v19, 3, v1 // (op 1247) 
v_lshl_add_u32 v25, v19, 7, v12 // (op 1247) 


v_mul_lo_u32 v12, v25, 6 // (op 1247) 

v_lshrrev_b32 v19, 3, v12 // (op 1247) 




s_waitcnt vmcnt(14)
ds_write_b96 v19, v[22:24] // (op 808) Store local data 

v_and_b32 v12, -2, v1 // (op 1189) 
v_sub_u32 v19, v1, v12 // (op 1189) 

v_lshlrev_b32 v12, 6, v19 // (op 1189) 

v_lshrrev_b32 v19, 1, v1 // (op 1189) 
v_lshl_add_u32 v25, v19, 7, v12 // (op 1189) 


v_mul_lo_u32 v12, v25, 6 // (op 1189) 

v_lshrrev_b32 v19, 3, v12 // (op 1189) 




s_waitcnt vmcnt(13)
ds_write_b128 v19, v[26:29] offset:1536 // (op 809) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v19, v[30:31] offset:1552 // (op 809) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v19, v[32:35] offset:1560 // (op 809) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v19, v[36:37] offset:1576 // (op 809) Store local data 

v_and_b32 v12, -4, v1 // (op 1300) 
v_sub_u32 v19, v1, v12 // (op 1300) 

v_lshrrev_b32 v12, 2, v1 // (op 1300) 
v_lshl_add_u32 v25, v12, 3, v19 // (op 1300) 





s_waitcnt vmcnt(9)
ds_write_b8 v25, v40 offset:7680 // (op 810) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v25, v41 offset:7684 // (op 810) Store local data 

v_and_b32 v12, -2, v1 // (op 1160) 
v_sub_u32 v19, v1, v12 // (op 1160) 

v_lshlrev_b32 v12, 6, v19 // (op 1160) 

v_lshrrev_b32 v19, 1, v1 // (op 1160) 
v_lshl_add_u32 v25, v19, 7, v12 // (op 1160) 


v_mul_lo_u32 v12, v25, 6 // (op 1160) 

v_lshrrev_b32 v19, 3, v12 // (op 1160) 

v_and_b32 v12, -8, v1 // (op 1218) 
v_sub_u32 v25, v1, v12 // (op 1218) 

v_lshlrev_b32 v12, 4, v25 // (op 1218) 

v_lshrrev_b32 v25, 3, v1 // (op 1218) 
v_lshl_add_u32 v45, v25, 7, v12 // (op 1218) 


v_mul_lo_u32 v12, v45, 6 // (op 1218) 

v_lshrrev_b32 v25, 3, v12 // (op 1218) 

v_and_b32 v12, -4, v1 // (op 1271) 
v_sub_u32 v45, v1, v12 // (op 1271) 

v_lshrrev_b32 v12, 2, v1 // (op 1271) 
v_lshl_add_u32 v15, v12, 3, v45 // (op 1271) 


v_and_b32 v12, -2, v1 // (op 1347) 
v_sub_u32 v16, v1, v12 // (op 1347) 

v_lshlrev_b32 v12, 6, v16 // (op 1347) 

v_lshrrev_b32 v16, 1, v1 // (op 1347) 
v_lshl_add_u32 v45, v16, 7, v12 // (op 1347) 


v_mul_lo_u32 v12, v45, 6 // (op 1347) 

v_lshrrev_b32 v16, 3, v12 // (op 1347) 

v_and_b32 v12, -8, v1 // (op 1376) 
v_sub_u32 v45, v1, v12 // (op 1376) 

v_lshlrev_b32 v12, 4, v45 // (op 1376) 

v_lshrrev_b32 v45, 3, v1 // (op 1376) 
v_lshl_add_u32 v38, v45, 7, v12 // (op 1376) 


v_mul_lo_u32 v12, v38, 6 // (op 1376) 

v_lshrrev_b32 v45, 3, v12 // (op 1376) 

v_and_b32 v12, -4, v1 // (op 1400) 
v_sub_u32 v38, v1, v12 // (op 1400) 

v_lshrrev_b32 v12, 2, v1 // (op 1400) 
v_lshl_add_u32 v39, v12, 3, v38 // (op 1400) 


s_mov_b32 s47, 0 // (op 315) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 314) 
s_mov_b32 s18, s0 // (op 314) convert

s_and_b32 s19, s18, 4294967294 // (op 314) 

s_cmp_lt_u32 s47, s19 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_314 if false)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_314:
s_add_u32 s47, s47, 2 // (op 316) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1458) 



ds_read_b128 v[60:63], v7 offset:1536 // (op 67) Load local data 
ds_read_b64 v[64:65], v7 offset:1552 // (op 67) Load local data 
s_barrier  // (op 1471) 



ds_read_b128 v[66:69], v3 // (op 72) Load local data 
ds_read_b64 v[70:71], v3 offset:16 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1485) 



ds_read_u8 v12, v0 offset:7680 // (op 77) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:71], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 84) 
ds_read_b128 v[72:75], v7 offset:4608 // (op 344) Load local data 


ds_read_b64 v[76:77], v7 offset:4624 // (op 344) Load local data 



ds_read_u8 v12, v0 offset:7808 // (op 348) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[66:71], a[4:7], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 350) 





v_mov_b32 v12, v20 // (op 82) 

buffer_load_ubyte v14, v12, s[68:71], 0 offen offset:8 // (op 82) Load value


v_mov_b32 v38, v17 // (op 10) 

buffer_load_dwordx3 v[22:24], v38, s[60:63], 0 offen offset:192 // (op 10) Load value


v_mov_b32 v60, v13 // (op 4) 

buffer_load_dwordx4 v[26:29], v60, s[48:51], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[30:31], v60, s[48:51], 0 offen offset:208 // (op 4) Load value
buffer_load_dwordx4 v[32:35], v60, s[48:51], 0 offen offset:216 // (op 4) Load value
buffer_load_dwordx2 v[36:37], v60, s[48:51], 0 offen offset:232 // (op 4) Load value

v_mov_b64 v[62:63], s[34:35] // (op 17) convert

v_mov_b32 v61, v18 // (op 17) 
v_mov_b32 v64, v61 // (op 17) 
buffer_load_ubyte v40, v64, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v64, v64, v62 // (op 17) 
buffer_load_ubyte v41, v64, s[64:67], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 1508) 



s_waitcnt vmcnt(14)
ds_write_b96 v45, v[42:44] offset:7936 // (op 239) Store local data 
s_barrier  // (op 1502) 



s_waitcnt vmcnt(13)
ds_write_b128 v16, v[46:49] offset:9472 // (op 229) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v16, v[50:51] offset:9488 // (op 229) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v16, v[52:55] offset:9496 // (op 229) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[56:57] offset:9512 // (op 229) Store local data 
s_barrier  // (op 1514) 



s_waitcnt vmcnt(9)
ds_write_b8 v39, v58 offset:15616 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v39, v59 offset:15620 // (op 240) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1499) 



ds_read_b128 v[62:65], v11 offset:9472 // (op 231) Load local data 
ds_read_b64 v[66:67], v11 offset:9488 // (op 231) Load local data 
s_barrier  // (op 1505) 



ds_read_b128 v[68:71], v4 offset:7936 // (op 233) Load local data 
ds_read_b64 v[72:73], v4 offset:7952 // (op 233) Load local data 
s_barrier  // (op 1555) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1511) 



ds_read_u8 v74, v10 offset:15616 // (op 235) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:67], v[68:73], a[0:3], v74, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 238) 
ds_read_b128 v[76:79], v11 offset:12544 // (op 357) Load local data 


ds_read_b64 v[80:81], v11 offset:12560 // (op 357) Load local data 



s_barrier  // (op 1589) 
ds_read_u8 v62, v10 offset:15744 // (op 361) Load local data 
s_barrier  // (op 1593) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:81], v[68:73], a[4:7], v62, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 351) 






buffer_load_ubyte v21, v12, s[68:71], 0 offen offset:12 // (op 237) Load value




v_add_u32 v20, 8, v20 // (op 1114) 
buffer_load_dwordx3 v[42:44], v38, s[60:63], 0 offen offset:288 // (op 241) Load value




v_add_u32 v17, 192, v17 // (op 986) 
buffer_load_dwordx4 v[46:49], v60, s[48:51], 0 offen offset:288 // (op 228) Load value
buffer_load_dwordx2 v[50:51], v60, s[48:51], 0 offen offset:304 // (op 228) Load value
buffer_load_dwordx4 v[52:55], v60, s[48:51], 0 offen offset:312 // (op 228) Load value
buffer_load_dwordx2 v[56:57], v60, s[48:51], 0 offen offset:328 // (op 228) Load value


v_add_u32 v13, 192, v13 // (op 922) 
v_mov_b64 v[62:63], s[34:35] // (op 242) convert

v_mov_b32 v12, v61 // (op 242) 
buffer_load_ubyte v58, v12, s[64:67], 0 offen offset:12 // (op 242) Load value
v_add_u32 v12, v12, v62 // (op 242) 
buffer_load_ubyte v59, v12, s[64:67], 0 offen offset:12 // (op 242) Load value



s_barrier  // (op 1474) 
v_add_u32 v18, 8, v18 // (op 1050) 



s_waitcnt vmcnt(14)
ds_write_b96 v25, v[22:24] // (op 13) Store local data 
s_barrier  // (op 1461) 



s_waitcnt vmcnt(13)
ds_write_b128 v19, v[26:29] offset:1536 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v19, v[30:31] offset:1552 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v19, v[32:35] offset:1560 // (op 6) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v19, v[36:37] offset:1576 // (op 6) Store local data 
s_barrier  // (op 1488) 



s_waitcnt vmcnt(9)
ds_write_b8 v15, v40 offset:7680 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v15, v41 offset:7684 // (op 21) Store local data 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 314) 
s_mov_b32 s18, s0 // (op 314) convert

s_and_b32 s19, s18, 4294967294 // (op 314) 

s_cmp_lt_u32 s47, s19 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoop_314 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoop_314:















s_mul_hi_i32 s47, s16, s10 // (op 884) 
v_lshrrev_b32 v45, 1, v1 // (op 884) 
s_add_i32 s0, s47, s16 // (op 884) 

v_and_b32 v12, -2, v1 // (op 884) 
s_ashr_i32 s1, s0, 31 // (op 884) 
v_sub_u32 v13, v1, v12 // (op 884) 

s_cmp_eq_i32 s10, 0 // (op 884) 
s_mov_b32 s47, scc // (op 884) 
v_lshlrev_b32 v12, 6, v13 // (op 884) 

s_cmp_eq_i32 s47, 0 // (op 884) 
s_cselect_b32 s18, 0, -1 // (op 884) 

s_and_b32 s19, s13, 31 // (op 884) 
s_lshl_b32 s47, 1, s19 // (op 884) 
s_add_i32 s22, s47, s18 // (op 884) 


s_and_b32 s18, s1, s22 // (op 884) 


s_add_i32 s1, s0, s18 // (op 884) 


s_ashr_i32 s0, s1, s19 // (op 884) 


s_lshl_b32 s1, s13, 24 // (op 884) 
s_ashr_i32 s47, s1, 31 // (op 884) 

s_xor_b32 s1, s0, s47 // (op 884) 

s_sub_i32 s0, s1, s47 // (op 884) 


s_mul_hi_i32 s1, s0, s14 // (op 884) 
s_add_i32 s47, s1, s0 // (op 884) 

s_ashr_i32 s1, s47, 31 // (op 884) 
s_cmp_eq_i32 s14, 0 // (op 884) 
s_mov_b32 s18, scc // (op 884) 
s_cmp_eq_i32 s18, 0 // (op 884) 
s_cselect_b32 s19, 0, -1 // (op 884) 

s_and_b32 s18, s15, 31 // (op 884) 
s_lshl_b32 s22, 1, s18 // (op 884) 
s_add_i32 s23, s22, s19 // (op 884) 


s_and_b32 s19, s1, s23 // (op 884) 


s_add_i32 s1, s47, s19 // (op 884) 


s_ashr_i32 s19, s1, s18 // (op 884) 


s_lshl_b32 s1, s15, 24 // (op 884) 
s_ashr_i32 s18, s1, 31 // (op 884) 

s_xor_b32 s1, s19, s18 // (op 884) 

s_sub_i32 s19, s1, s18 // (op 884) 


s_mul_i32 s1, s0, s11 // (op 884) 

s_sub_u32 s0, s16, s1 // (op 884) 

s_mul_i32 s1, s19, s11 // (op 884) 
s_add_u32 s1, s1, s0 // (op 884) 


s_mul_hi_i32 s0, s16, s9 // (op 884) 
s_add_i32 s47, s0, s16 // (op 884) 

s_ashr_i32 s0, s47, 31 // (op 884) 
s_cmp_eq_i32 s9, 0 // (op 884) 
s_mov_b32 s18, scc // (op 884) 
s_cmp_eq_i32 s18, 0 // (op 884) 
s_cselect_b32 s19, 0, -1 // (op 884) 

s_and_b32 s18, s45, 31 // (op 884) 
s_lshl_b32 s22, 1, s18 // (op 884) 
s_add_i32 s23, s22, s19 // (op 884) 


s_and_b32 s19, s0, s23 // (op 884) 


s_add_i32 s0, s47, s19 // (op 884) 


s_ashr_i32 s19, s0, s18 // (op 884) 


s_lshl_b32 s0, s45, 24 // (op 884) 
s_ashr_i32 s18, s0, 31 // (op 884) 

s_xor_b32 s0, s19, s18 // (op 884) 

s_sub_i32 s19, s0, s18 // (op 884) 


s_add_u32 s22, s4, 16 // (op 884) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 884) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 884) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 884) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 884) 

s_mul_i32 s0, s11, s22 // (op 884) 

s_mul_i32 s18, s8, s0 // (op 884) 

s_mul_i32 s0, s19, s18 // (op 884) 

s_sub_u32 s18, s16, s0 // (op 884) 

s_mov_b32 s0, s18 // (op 884) 
s_mul_hi_i32 s47, s0, s12 // (op 884) 
s_add_i32 s22, s47, s0 // (op 884) 


s_ashr_i32 s0, s22, 31 // (op 884) 
s_cmp_eq_i32 s12, 0 // (op 884) 
s_mov_b32 s47, scc // (op 884) 
s_cmp_eq_i32 s47, 0 // (op 884) 
s_cselect_b32 s23, 0, -1 // (op 884) 

s_and_b32 s47, s44, 31 // (op 884) 
s_lshl_b32 s24, 1, s47 // (op 884) 
s_add_i32 s25, s24, s23 // (op 884) 


s_and_b32 s23, s0, s25 // (op 884) 


s_add_i32 s0, s22, s23 // (op 884) 


s_ashr_i32 s22, s0, s47 // (op 884) 


s_lshl_b32 s0, s44, 24 // (op 884) 
s_ashr_i32 s47, s0, 31 // (op 884) 

s_xor_b32 s0, s22, s47 // (op 884) 

s_sub_i32 s22, s0, s47 // (op 884) 


s_mul_i32 s0, s46, s11 // (op 884) 
s_add_u32 s24, s6, 64 // (op 884) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 884) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 884) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 884) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 884) 

s_sub_i32 s23, s24, s0 // (op 884) 

s_mul_i32 s47, s22, s23 // (op 884) 


s_sub_u32 s22, s18, s47 // (op 884) 


s_add_u32 s18, s22, s0 // (op 884) 


s_cmp_eq_i32 s19, 0 // (op 884) 
s_mov_b32 s0, scc // (op 884) 

s_cmp_eq_i32 s0, 0 // (op 884) 
s_cselect_b32 s19, s18, s1 // (op 884) 



v_lshl_add_u32 v13, s19, 6, v45 // (op 884) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 884) 
s_lshr_b32 s47, s0, 1 // (op 884) 

v_lshl_add_u32 v45, s47, 8, v12 // (op 884) 


v_mov_b32 v15, 0 // (op 884) convert
v_mov_b32 v14, v45 // (op 884) convert

v_mul_lo_u32 v17, v13, s27 // (op 884) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 884) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 884) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 884) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 884) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 884) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 884) most significant: low of high * low
v_mul_hi_u32 v45, v16, 6 // (op 884) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 884) least significant: low of low * low
v_add_u32 v13, v13, v45 // (op 884) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 884) 

v_mov_b32 v45, v14 // (op 884) convert

s_mul_i32 s1, s7, 6 // (op 886) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 886) least significant
s_mul_hi_u32 s47, s6, 6 // (op 886) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 886) most significant: sum

s_mul_i32 s19, s26, s1 // (op 886) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 886) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 886) least significant
s_mul_hi_u32 s22, s26, s0 // (op 886) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 886) most significant: sum
s_add_u32 s19, s19, s22 // (op 886) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 886) 

s_mov_b32 s48, s28 // (op 886) reg expression
s_mov_b32 s49, s29 // (op 886) reg expression
s_mov_b32 s50, s0 // (op 886) reg expression
s_mov_b32 s51, 131072 // (op 886) call()

s_mul_hi_i32 s47, s16, s10 // (op 952) 
v_lshrrev_b32 v12, 3, v1 // (op 952) 
s_add_i32 s0, s47, s16 // (op 952) 

v_and_b32 v13, -8, v1 // (op 952) 
s_ashr_i32 s1, s0, 31 // (op 952) 
v_sub_u32 v14, v1, v13 // (op 952) 

s_cmp_eq_i32 s10, 0 // (op 952) 
s_mov_b32 s47, scc // (op 952) 
v_lshlrev_b32 v13, 4, v14 // (op 952) 

s_cmp_eq_i32 s47, 0 // (op 952) 
s_cselect_b32 s18, 0, -1 // (op 952) 

s_and_b32 s19, s13, 31 // (op 952) 
s_lshl_b32 s47, 1, s19 // (op 952) 
s_add_i32 s22, s47, s18 // (op 952) 


s_and_b32 s18, s1, s22 // (op 952) 


s_add_i32 s1, s0, s18 // (op 952) 


s_ashr_i32 s0, s1, s19 // (op 952) 


s_lshl_b32 s1, s13, 24 // (op 952) 
s_ashr_i32 s47, s1, 31 // (op 952) 

s_xor_b32 s1, s0, s47 // (op 952) 

s_sub_i32 s0, s1, s47 // (op 952) 


s_mul_hi_i32 s1, s0, s14 // (op 952) 
s_add_i32 s47, s1, s0 // (op 952) 

s_ashr_i32 s1, s47, 31 // (op 952) 
s_cmp_eq_i32 s14, 0 // (op 952) 
s_mov_b32 s18, scc // (op 952) 
s_cmp_eq_i32 s18, 0 // (op 952) 
s_cselect_b32 s19, 0, -1 // (op 952) 

s_and_b32 s18, s15, 31 // (op 952) 
s_lshl_b32 s22, 1, s18 // (op 952) 
s_add_i32 s23, s22, s19 // (op 952) 


s_and_b32 s19, s1, s23 // (op 952) 


s_add_i32 s1, s47, s19 // (op 952) 


s_ashr_i32 s19, s1, s18 // (op 952) 


s_lshl_b32 s1, s15, 24 // (op 952) 
s_ashr_i32 s18, s1, 31 // (op 952) 

s_xor_b32 s1, s19, s18 // (op 952) 

s_sub_i32 s19, s1, s18 // (op 952) 


s_add_u32 s22, s4, 16 // (op 952) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 952) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 952) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 952) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 952) 

s_mul_i32 s1, s19, s22 // (op 952) 

s_sub_i32 s47, s0, s1 // (op 952) 


s_mul_hi_i32 s0, s16, s9 // (op 952) 
s_add_i32 s1, s0, s16 // (op 952) 

s_ashr_i32 s0, s1, 31 // (op 952) 
s_cmp_eq_i32 s9, 0 // (op 952) 
s_mov_b32 s18, scc // (op 952) 
s_cmp_eq_i32 s18, 0 // (op 952) 
s_cselect_b32 s19, 0, -1 // (op 952) 

s_and_b32 s18, s45, 31 // (op 952) 
s_lshl_b32 s24, 1, s18 // (op 952) 
s_add_i32 s25, s24, s19 // (op 952) 


s_and_b32 s19, s0, s25 // (op 952) 


s_add_i32 s0, s1, s19 // (op 952) 


s_ashr_i32 s1, s0, s18 // (op 952) 


s_lshl_b32 s0, s45, 24 // (op 952) 
s_ashr_i32 s18, s0, 31 // (op 952) 

s_xor_b32 s0, s1, s18 // (op 952) 

s_sub_i32 s1, s0, s18 // (op 952) 


s_mul_i32 s0, s11, s22 // (op 952) 

s_mul_i32 s18, s8, s0 // (op 952) 

s_mul_i32 s0, s1, s18 // (op 952) 

s_sub_u32 s18, s16, s0 // (op 952) 

s_mul_hi_i32 s0, s18, s12 // (op 952) 
s_add_i32 s19, s0, s18 // (op 952) 


s_ashr_i32 s0, s19, 31 // (op 952) 
s_cmp_eq_i32 s12, 0 // (op 952) 
s_mov_b32 s18, scc // (op 952) 
s_cmp_eq_i32 s18, 0 // (op 952) 
s_cselect_b32 s22, 0, -1 // (op 952) 

s_and_b32 s18, s44, 31 // (op 952) 
s_lshl_b32 s23, 1, s18 // (op 952) 
s_add_i32 s24, s23, s22 // (op 952) 


s_and_b32 s25, s0, s24 // (op 952) 


s_add_i32 s0, s19, s25 // (op 952) 


s_ashr_i32 s19, s0, s18 // (op 952) 


s_lshl_b32 s0, s44, 24 // (op 952) 
s_ashr_i32 s18, s0, 31 // (op 952) 

s_xor_b32 s0, s19, s18 // (op 952) 

s_sub_i32 s19, s0, s18 // (op 952) 


s_cmp_eq_i32 s1, 0 // (op 952) 
s_mov_b32 s0, scc // (op 952) 

s_cmp_eq_i32 s0, 0 // (op 952) 
s_cselect_b32 s1, s19, s47 // (op 952) 



v_lshl_add_u32 v14, s1, 4, v12 // (op 952) 


v_mul_lo_u32 v23, v14, s31 // (op 952) most significant: low of low * high
v_mul_hi_u32 v12, v14, s30 // (op 952) most significant: high of low * low
v_mul_lo_u32 v22, v14, s30 // (op 952) least significant: low of low * low
v_add_u32 v23, v23, v12 // (op 952) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 952) 
s_lshr_b32 s47, s0, 1 // (op 952) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 952) 


v_mov_b32 v25, 0 // (op 952) convert
v_mov_b32 v24, v12 // (op 952) convert

v_add_co_u32 v12, s[0:1], v24, v22 // (op 952) least significant half
v_addc_co_u32 v13, s[0:1], v25, v23, s[0:1] // (op 952) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 952) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 952) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 952) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 952) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 952) 

v_mov_b32 v14, v12 // (op 952) convert

s_mul_i32 s1, s5, 6 // (op 954) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 954) least significant
s_mul_hi_u32 s47, s4, 6 // (op 954) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 954) most significant: sum

s_mul_i32 s19, s30, s1 // (op 954) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 954) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 954) least significant
s_mul_hi_u32 s22, s30, s0 // (op 954) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 954) most significant: sum
s_add_u32 s19, s19, s22 // (op 954) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 954) 

s_mov_b32 s60, s32 // (op 954) reg expression
s_mov_b32 s61, s33 // (op 954) reg expression
s_mov_b32 s62, s0 // (op 954) reg expression
s_mov_b32 s63, 131072 // (op 954) call()

s_mul_hi_i32 s47, s16, s10 // (op 1016) 
v_lshrrev_b32 v12, 2, v1 // (op 1016) 
s_add_i32 s0, s47, s16 // (op 1016) 

v_lshlrev_b32 v13, 1, v12 // (op 1016) 

s_ashr_i32 s1, s0, 31 // (op 1016) 
v_and_b32 v12, -4, v1 // (op 1016) 
s_cmp_eq_i32 s10, 0 // (op 1016) 
s_mov_b32 s47, scc // (op 1016) 
v_sub_u32 v15, v1, v12 // (op 1016) 

s_cmp_eq_i32 s47, 0 // (op 1016) 
s_cselect_b32 s18, 0, -1 // (op 1016) 

s_and_b32 s19, s13, 31 // (op 1016) 
s_lshl_b32 s47, 1, s19 // (op 1016) 
s_add_i32 s22, s47, s18 // (op 1016) 


s_and_b32 s18, s1, s22 // (op 1016) 


s_add_i32 s1, s0, s18 // (op 1016) 


s_ashr_i32 s0, s1, s19 // (op 1016) 


s_lshl_b32 s1, s13, 24 // (op 1016) 
s_ashr_i32 s47, s1, 31 // (op 1016) 

s_xor_b32 s1, s0, s47 // (op 1016) 

s_sub_i32 s0, s1, s47 // (op 1016) 


s_mul_hi_i32 s1, s0, s14 // (op 1016) 
s_add_i32 s47, s1, s0 // (op 1016) 

s_ashr_i32 s1, s47, 31 // (op 1016) 
s_cmp_eq_i32 s14, 0 // (op 1016) 
s_mov_b32 s18, scc // (op 1016) 
s_cmp_eq_i32 s18, 0 // (op 1016) 
s_cselect_b32 s19, 0, -1 // (op 1016) 

s_and_b32 s18, s15, 31 // (op 1016) 
s_lshl_b32 s22, 1, s18 // (op 1016) 
s_add_i32 s23, s22, s19 // (op 1016) 


s_and_b32 s19, s1, s23 // (op 1016) 


s_add_i32 s1, s47, s19 // (op 1016) 


s_ashr_i32 s19, s1, s18 // (op 1016) 


s_lshl_b32 s1, s15, 24 // (op 1016) 
s_ashr_i32 s18, s1, 31 // (op 1016) 

s_xor_b32 s1, s19, s18 // (op 1016) 

s_sub_i32 s19, s1, s18 // (op 1016) 


s_mul_i32 s1, s0, s11 // (op 1016) 

s_sub_u32 s0, s16, s1 // (op 1016) 

s_mul_i32 s1, s19, s11 // (op 1016) 
s_add_u32 s1, s1, s0 // (op 1016) 


s_mul_hi_i32 s0, s16, s9 // (op 1016) 
s_add_i32 s47, s0, s16 // (op 1016) 

s_ashr_i32 s0, s47, 31 // (op 1016) 
s_cmp_eq_i32 s9, 0 // (op 1016) 
s_mov_b32 s18, scc // (op 1016) 
s_cmp_eq_i32 s18, 0 // (op 1016) 
s_cselect_b32 s19, 0, -1 // (op 1016) 

s_and_b32 s18, s45, 31 // (op 1016) 
s_lshl_b32 s22, 1, s18 // (op 1016) 
s_add_i32 s23, s22, s19 // (op 1016) 


s_and_b32 s19, s0, s23 // (op 1016) 


s_add_i32 s0, s47, s19 // (op 1016) 


s_ashr_i32 s19, s0, s18 // (op 1016) 


s_lshl_b32 s0, s45, 24 // (op 1016) 
s_ashr_i32 s18, s0, 31 // (op 1016) 

s_xor_b32 s0, s19, s18 // (op 1016) 

s_sub_i32 s19, s0, s18 // (op 1016) 


s_add_u32 s22, s4, 16 // (op 1016) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1016) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1016) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1016) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1016) 

s_mul_i32 s0, s11, s22 // (op 1016) 

s_mul_i32 s18, s8, s0 // (op 1016) 

s_mul_i32 s0, s19, s18 // (op 1016) 

s_sub_u32 s18, s16, s0 // (op 1016) 

s_mov_b32 s0, s18 // (op 1016) 
s_mul_hi_i32 s47, s0, s12 // (op 1016) 
s_add_i32 s22, s47, s0 // (op 1016) 


s_ashr_i32 s0, s22, 31 // (op 1016) 
s_cmp_eq_i32 s12, 0 // (op 1016) 
s_mov_b32 s47, scc // (op 1016) 
s_cmp_eq_i32 s47, 0 // (op 1016) 
s_cselect_b32 s23, 0, -1 // (op 1016) 

s_and_b32 s47, s44, 31 // (op 1016) 
s_lshl_b32 s24, 1, s47 // (op 1016) 
s_add_i32 s25, s24, s23 // (op 1016) 


s_and_b32 s23, s0, s25 // (op 1016) 


s_add_i32 s0, s22, s23 // (op 1016) 


s_ashr_i32 s22, s0, s47 // (op 1016) 


s_lshl_b32 s0, s44, 24 // (op 1016) 
s_ashr_i32 s47, s0, 31 // (op 1016) 

s_xor_b32 s0, s22, s47 // (op 1016) 

s_sub_i32 s22, s0, s47 // (op 1016) 


s_mul_i32 s0, s46, s11 // (op 1016) 
s_add_u32 s24, s6, 64 // (op 1016) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1016) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1016) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1016) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 1016) 

s_sub_i32 s23, s24, s0 // (op 1016) 

s_mul_i32 s47, s22, s23 // (op 1016) 


s_sub_u32 s22, s18, s47 // (op 1016) 


s_add_u32 s18, s22, s0 // (op 1016) 


s_cmp_eq_i32 s19, 0 // (op 1016) 
s_mov_b32 s0, scc // (op 1016) 

s_cmp_eq_i32 s0, 0 // (op 1016) 
s_cselect_b32 s19, s18, s1 // (op 1016) 



v_lshl_add_u32 v12, s19, 6, v13 // (op 1016) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1016) 
s_lshr_b32 s47, s0, 1 // (op 1016) 

v_lshl_add_u32 v13, s47, 3, v15 // (op 1016) 


v_mul_lo_u32 v15, v12, s34 // (op 1016) 
v_add_u32 v15, v15, v13 // (op 1016) 


s_mul_i32 s1, s34, s7 // (op 1018) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1018) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1018) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1018) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1018) most significant: sum
s_add_u32 s1, s1, s18 // (op 1018) most significant: sum


s_mov_b32 s64, s36 // (op 1018) reg expression
s_mov_b32 s65, s37 // (op 1018) reg expression
s_mov_b32 s66, s0 // (op 1018) reg expression
s_mov_b32 s67, 131072 // (op 1018) call()

s_mul_hi_i32 s47, s16, s10 // (op 1080) 
v_and_b32 v12, 15, v1 // (op 1080) 
s_add_i32 s0, s47, s16 // (op 1080) 

v_and_b32 v13, 63, v1 // (op 1080) 
s_ashr_i32 s1, s0, 31 // (op 1080) 
v_lshrrev_b32 v16, 4, v13 // (op 1080) 

s_cmp_eq_i32 s10, 0 // (op 1080) 
s_mov_b32 s47, scc // (op 1080) 
s_cmp_eq_i32 s47, 0 // (op 1080) 
s_cselect_b32 s18, 0, -1 // (op 1080) 

s_and_b32 s19, s13, 31 // (op 1080) 
s_lshl_b32 s47, 1, s19 // (op 1080) 
s_add_i32 s22, s47, s18 // (op 1080) 


s_and_b32 s18, s1, s22 // (op 1080) 


s_add_i32 s1, s0, s18 // (op 1080) 


s_ashr_i32 s0, s1, s19 // (op 1080) 


s_lshl_b32 s1, s13, 24 // (op 1080) 
s_ashr_i32 s47, s1, 31 // (op 1080) 

s_xor_b32 s1, s0, s47 // (op 1080) 

s_sub_i32 s0, s1, s47 // (op 1080) 


s_mul_hi_i32 s1, s0, s14 // (op 1080) 
s_add_i32 s47, s1, s0 // (op 1080) 

s_ashr_i32 s1, s47, 31 // (op 1080) 
s_cmp_eq_i32 s14, 0 // (op 1080) 
s_mov_b32 s18, scc // (op 1080) 
s_cmp_eq_i32 s18, 0 // (op 1080) 
s_cselect_b32 s19, 0, -1 // (op 1080) 

s_and_b32 s18, s15, 31 // (op 1080) 
s_lshl_b32 s22, 1, s18 // (op 1080) 
s_add_i32 s23, s22, s19 // (op 1080) 


s_and_b32 s19, s1, s23 // (op 1080) 


s_add_i32 s1, s47, s19 // (op 1080) 


s_ashr_i32 s19, s1, s18 // (op 1080) 


s_lshl_b32 s1, s15, 24 // (op 1080) 
s_ashr_i32 s18, s1, 31 // (op 1080) 

s_xor_b32 s1, s19, s18 // (op 1080) 

s_sub_i32 s19, s1, s18 // (op 1080) 


s_add_u32 s22, s4, 16 // (op 1080) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1080) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1080) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1080) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 1080) 

s_mul_i32 s1, s19, s22 // (op 1080) 

s_sub_i32 s47, s0, s1 // (op 1080) 


s_mul_hi_i32 s0, s16, s9 // (op 1080) 
s_add_i32 s1, s0, s16 // (op 1080) 

s_ashr_i32 s0, s1, 31 // (op 1080) 
s_cmp_eq_i32 s9, 0 // (op 1080) 
s_mov_b32 s18, scc // (op 1080) 
s_cmp_eq_i32 s18, 0 // (op 1080) 
s_cselect_b32 s19, 0, -1 // (op 1080) 

s_and_b32 s18, s45, 31 // (op 1080) 
s_lshl_b32 s24, 1, s18 // (op 1080) 
s_add_i32 s25, s24, s19 // (op 1080) 


s_and_b32 s19, s0, s25 // (op 1080) 


s_add_i32 s0, s1, s19 // (op 1080) 


s_ashr_i32 s1, s0, s18 // (op 1080) 


s_lshl_b32 s0, s45, 24 // (op 1080) 
s_ashr_i32 s18, s0, 31 // (op 1080) 

s_xor_b32 s0, s1, s18 // (op 1080) 

s_sub_i32 s1, s0, s18 // (op 1080) 


s_mul_i32 s0, s11, s22 // (op 1080) 

s_mul_i32 s18, s8, s0 // (op 1080) 

s_mul_i32 s0, s1, s18 // (op 1080) 

s_sub_u32 s18, s16, s0 // (op 1080) 

s_mul_hi_i32 s0, s18, s12 // (op 1080) 
s_add_i32 s19, s0, s18 // (op 1080) 


s_ashr_i32 s0, s19, 31 // (op 1080) 
s_cmp_eq_i32 s12, 0 // (op 1080) 
s_mov_b32 s18, scc // (op 1080) 
s_cmp_eq_i32 s18, 0 // (op 1080) 
s_cselect_b32 s22, 0, -1 // (op 1080) 

s_and_b32 s18, s44, 31 // (op 1080) 
s_lshl_b32 s23, 1, s18 // (op 1080) 
s_add_i32 s24, s23, s22 // (op 1080) 


s_and_b32 s25, s0, s24 // (op 1080) 


s_add_i32 s0, s19, s25 // (op 1080) 


s_ashr_i32 s19, s0, s18 // (op 1080) 


s_lshl_b32 s0, s44, 24 // (op 1080) 
s_ashr_i32 s18, s0, 31 // (op 1080) 

s_xor_b32 s0, s19, s18 // (op 1080) 

s_sub_i32 s19, s0, s18 // (op 1080) 


s_cmp_eq_i32 s1, 0 // (op 1080) 
s_mov_b32 s0, scc // (op 1080) 

s_cmp_eq_i32 s0, 0 // (op 1080) 
s_cselect_b32 s1, s19, s47 // (op 1080) 



v_lshl_add_u32 v13, s1, 4, v12 // (op 1080) 


v_mul_lo_u32 v12, v13, s38 // (op 1080) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1080) 
s_lshr_b32 s47, s0, 1 // (op 1080) 

v_lshl_add_u32 v13, s47, 3, v16 // (op 1080) 


v_add_u32 v16, v13, v12 // (op 1080) 


s_mul_i32 s1, s38, s5 // (op 1082) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1082) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1082) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1082) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1082) most significant: sum
s_add_u32 s1, s1, s18 // (op 1082) most significant: sum


s_mov_b32 s68, s40 // (op 1082) reg expression
s_mov_b32 s69, s41 // (op 1082) reg expression
s_mov_b32 s70, s0 // (op 1082) reg expression
s_mov_b32 s71, 131072 // (op 1082) call()

v_and_b32 v12, -2, v1 // (op 1173) 
v_sub_u32 v13, v1, v12 // (op 1173) 

v_lshlrev_b32 v12, 6, v13 // (op 1173) 

v_lshrrev_b32 v13, 1, v1 // (op 1173) 
v_lshl_add_u32 v17, v13, 7, v12 // (op 1173) 


v_mul_lo_u32 v12, v17, 6 // (op 1173) 

v_lshrrev_b32 v13, 3, v12 // (op 1173) 

v_and_b32 v12, -8, v1 // (op 1231) 
v_sub_u32 v17, v1, v12 // (op 1231) 

v_lshlrev_b32 v12, 4, v17 // (op 1231) 

v_lshrrev_b32 v17, 3, v1 // (op 1231) 
v_lshl_add_u32 v18, v17, 7, v12 // (op 1231) 


v_mul_lo_u32 v12, v18, 6 // (op 1231) 

v_lshrrev_b32 v17, 3, v12 // (op 1231) 

v_and_b32 v12, -4, v1 // (op 1284) 
v_sub_u32 v18, v1, v12 // (op 1284) 

v_lshrrev_b32 v12, 2, v1 // (op 1284) 
v_lshl_add_u32 v19, v12, 3, v18 // (op 1284) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 320) 
s_and_b32 s47, s0, 4294967294 // (op 320) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 319) 
s_mov_b32 s18, s0 // (op 319) convert

s_cmp_lt_u32 s47, s18 // (op 319) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_319:


s_add_u32 s47, s47, 1 // (op 321) 
v_mov_b32 v12, v45 // (op 170) 

buffer_load_dwordx4 v[20:23], v12, s[48:51], 0 offen // (op 170) Load value
buffer_load_dwordx2 v[24:25], v12, s[48:51], 0 offen offset:16 // (op 170) Load value
buffer_load_dwordx4 v[26:29], v12, s[48:51], 0 offen offset:24 // (op 170) Load value
buffer_load_dwordx2 v[30:31], v12, s[48:51], 0 offen offset:40 // (op 170) Load value

v_add_u32 v45, 96, v45 // (op 874) 


s_barrier  // (op 1464) 
v_mov_b32 v12, v14 // (op 183) 




buffer_load_dwordx3 v[32:34], v12, s[60:63], 0 offen // (op 183) Load value

v_add_u32 v14, 96, v14 // (op 944) 
s_waitcnt vmcnt(4)
ds_write_b128 v13, v[20:23] offset:1536 // (op 171) Store local data 

v_mov_b64 v[36:37], s[34:35] // (op 184) convert

v_mov_b32 v12, v15 // (op 184) 
v_mov_b32 v18, v12 // (op 184) 
buffer_load_ubyte v38, v18, s[64:67], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v18, v18, v36 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b64 v13, v[24:25] offset:1552 // (op 171) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v13, v[26:29] offset:1560 // (op 171) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v13, v[30:31] offset:1576 // (op 171) Store local data 
buffer_load_ubyte v39, v18, s[64:67], 0 offen offset: 0 // (op 184) Load value




v_add_u32 v15, 4, v15 // (op 1008) 
s_barrier  // (op 1477) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1467) 






ds_read_b128 v[20:23], v7 offset:1536 // (op 173) Load local data 
ds_read_b64 v[24:25], v7 offset:1552 // (op 173) Load local data 
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[32:34] // (op 181) Store local data 

s_barrier  // (op 1491) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1481) 






ds_read_b128 v[26:29], v3 // (op 175) Load local data 
ds_read_b64 v[30:31], v3 offset:16 // (op 175) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v38 offset:7680 // (op 182) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v39 offset:7684 // (op 182) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1495) 



ds_read_u8 v12, v0 offset:7680 // (op 177) Load local data 


v_mov_b32 v18, v16 // (op 179) 

buffer_load_ubyte v32, v18, s[68:71], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v16, 4, v16 // (op 1072) 



ds_read_b128 v[34:37], v7 offset:4608 // (op 325) Load local data 
ds_read_b64 v[38:39], v7 offset:4624 // (op 325) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 180) 





ds_read_u8 v12, v0 offset:7808 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:39], v[26:31], a[4:7], v12, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 331) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 319) 
s_mov_b32 s18, s0 // (op 319) convert

s_cmp_lt_u32 s47, s18 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopTop_KLoopTail_319 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_319:





















s_barrier  // (op 1577) 



















v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 

v_cvt_f16_f32 v10, v14 // (op 52) 
v_cvt_f16_f32 v11, v15 // (op 52) 
v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 


v_pack_b32_f16 v14, v10, v11 // (op 54) 
v_pack_b32_f16 v15, v12, v13 // (op 54) 


buffer_store_dwordx2 v[14:15], v5, s[56:59], 0 offen // (op 54) Store value


v_accvgpr_read v10, a4 // (op 517) 
v_accvgpr_read v11, a5 // (op 517) 
v_accvgpr_read v12, a6 // (op 517) 
v_accvgpr_read v13, a7 // (op 517) 
v_mul_f32 v14, v9, v10 // (op 517) 
v_mul_f32 v15, v9, v11 // (op 517) 
v_mul_f32 v16, v9, v12 // (op 517) 
v_mul_f32 v17, v9, v13 // (op 517) 

v_cvt_f16_f32 v10, v14 // (op 519) 
v_cvt_f16_f32 v11, v15 // (op 519) 
v_cvt_f16_f32 v12, v16 // (op 519) 
v_cvt_f16_f32 v13, v17 // (op 519) 


v_pack_b32_f16 v14, v10, v11 // (op 520) 
v_pack_b32_f16 v15, v12, v13 // (op 520) 


buffer_store_dwordx2 v[14:15], v5, s[56:59], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:64 // (op 550) Load value
v_accvgpr_read v12, a4 // (op 546) 
v_accvgpr_read v13, a5 // (op 546) 
v_accvgpr_read v14, a6 // (op 546) 
v_accvgpr_read v15, a7 // (op 546) 
v_mul_f32 v16, v9, v12 // (op 546) 
v_mul_f32 v17, v9, v13 // (op 546) 
v_mul_f32 v18, v9, v14 // (op 546) 
v_mul_f32 v19, v9, v15 // (op 546) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 551) 
v_lshrrev_b32 v13, 16, v10 // (op 551) 
v_cvt_f32_f16 v13, v13 // (op 551) 
v_mul_f32 v20, v8, v12 // (op 551) 
v_mul_f32 v21, v8, v13 // (op 551) 
v_cvt_f32_f16 v12, v11 // (op 551) 
v_lshrrev_b32 v13, 16, v11 // (op 551) 
v_cvt_f32_f16 v13, v13 // (op 551) 
v_mul_f32 v22, v8, v12 // (op 551) 
v_mul_f32 v23, v8, v13 // (op 551) 


v_add_f32 v10, v20, v16 // (op 547) 
v_add_f32 v11, v21, v17 // (op 547) 
v_add_f32 v12, v22, v18 // (op 547) 
v_add_f32 v13, v23, v19 // (op 547) 


v_cvt_f16_f32 v14, v10 // (op 548) 
v_cvt_f16_f32 v15, v11 // (op 548) 
v_cvt_f16_f32 v16, v12 // (op 548) 
v_cvt_f16_f32 v17, v13 // (op 548) 


v_pack_b32_f16 v10, v14, v15 // (op 549) 
v_pack_b32_f16 v11, v16, v17 // (op 549) 


buffer_store_dwordx2 v[10:11], v5, s[56:59], 0 offen offset:64 // (op 549) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen // (op 138) Load value
v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 
v_mul_f32 v18, v9, v14 // (op 134) 
v_mul_f32 v19, v9, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 139) 
v_lshrrev_b32 v13, 16, v10 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v8, v12 // (op 139) 
v_mul_f32 v21, v8, v13 // (op 139) 
v_cvt_f32_f16 v12, v11 // (op 139) 
v_lshrrev_b32 v13, 16, v11 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v22, v8, v12 // (op 139) 
v_mul_f32 v23, v8, v13 // (op 139) 


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


buffer_store_dwordx2 v[10:11], v5, s[56:59], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_, .LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM__end-RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 92
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 15872 // lds bytes
  .amdhsa_accum_offset 84
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F6
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
  - .name: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 15872
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 82
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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

