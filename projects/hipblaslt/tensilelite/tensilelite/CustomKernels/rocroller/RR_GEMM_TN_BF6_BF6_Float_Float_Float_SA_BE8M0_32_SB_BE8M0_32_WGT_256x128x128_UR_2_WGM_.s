

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_,@function
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 13656) 
s_mul_hi_i32 s1, s16, s10 // (op 13797) 
v_and_b32 v0, 4294967168, v1 // (op 13718) 
v_and_b32 v3, 15, v1 // (op 13718) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 13718) 


v_and_b32 v0, 63, v1 // (op 13718) 
v_lshrrev_b32 v3, 4, v0 // (op 13718) 

v_lshlrev_b32 v0, 5, v3 // (op 13718) 

v_add_u32 v3, v4, v0 // (op 13718) 


v_mul_lo_u32 v0, v3, 6 // (op 13718) 

v_lshrrev_b32 v3, 3, v0 // (op 13718) 

v_lshrrev_b32 v0, 6, v1 // (op 13776) 
v_and_b32 v4, 1, v0 // (op 13776) 

v_and_b32 v0, 15, v1 // (op 13776) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 13776) 


v_and_b32 v0, 63, v1 // (op 13776) 
v_lshrrev_b32 v4, 4, v0 // (op 13776) 

v_lshlrev_b32 v0, 5, v4 // (op 13776) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 13776) 


v_mul_lo_u32 v0, v4, 6 // (op 13776) 

v_lshrrev_b32 v4, 3, v0 // (op 13776) 

v_and_b32 v0, 4294967168, v1 // (op 13828) 
v_and_b32 v5, 15, v1 // (op 13828) 
v_add_lshl_u32 v6, v0, v5, 7 // (op 13828) 


v_and_b32 v0, 63, v1 // (op 13828) 
v_lshrrev_b32 v5, 4, v0 // (op 13828) 

v_lshlrev_b32 v0, 5, v5 // (op 13828) 

v_add_u32 v5, v6, v0 // (op 13828) 


v_mul_lo_u32 v0, v5, 6 // (op 13828) 

v_lshrrev_b32 v5, 3, v0 // (op 13828) 

v_lshrrev_b32 v0, 6, v1 // (op 13857) 
v_and_b32 v6, 1, v0 // (op 13857) 

v_and_b32 v0, 15, v1 // (op 13857) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 13857) 


v_and_b32 v0, 63, v1 // (op 13857) 
v_lshrrev_b32 v6, 4, v0 // (op 13857) 

v_lshlrev_b32 v0, 5, v6 // (op 13857) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 13857) 


v_mul_lo_u32 v0, v6, 6 // (op 13857) 

v_lshrrev_b32 v6, 3, v0 // (op 13857) 

v_and_b32 v0, 4294967168, v1 // (op 13656) 
s_add_i32 s18, s0, s16 // (op 13656) 

v_and_b32 v7, 63, v1 // (op 13656) 
s_ashr_i32 s0, s18, 31 // (op 13656) 
v_lshrrev_b32 v8, 4, v7 // (op 13656) 
s_cmp_eq_i32 s10, 0 // (op 13656) 
s_mov_b32 s19, scc // (op 13656) 
v_lshlrev_b32 v9, 2, v8 // (op 13656) 

s_cmp_eq_i32 s19, 0 // (op 13656) 
s_cselect_b32 s49, 0, -1 // (op 13656) 

v_add_u32 v8, v0, v9 // (op 13656) 


s_and_b32 s19, s13, 31 // (op 13656) 
v_lshrrev_b32 v0, 2, v7 // (op 13656) 

v_and_b32 v7, 3, v0 // (op 13656) 

v_and_b32 v0, 4294967168, v1 // (op 13797) 
v_and_b32 v9, 63, v1 // (op 13797) 
s_lshl_b32 s50, 1, s19 // (op 13656) 
v_and_b32 v10, 3, v1 // (op 13656) 
s_add_i32 s51, s50, s49 // (op 13656) 


s_and_b32 s49, s0, s51 // (op 13656) 


s_add_i32 s0, s1, s16 // (op 13797) 

s_ashr_i32 s1, s0, 31 // (op 13797) 
v_lshl_add_u32 v11, v7, 2, v10 // (op 13656) 


s_add_i32 s50, s18, s49 // (op 13656) 


v_lshrrev_b32 v7, 6, v1 // (op 13656) 
s_ashr_i32 s18, s50, s19 // (op 13656) 


v_and_b32 v10, 1, v7 // (op 13656) 

s_lshl_b32 s19, s13, 24 // (op 13656) 
v_lshl_add_u32 v7, v10, 6, v11 // (op 13656) 


s_ashr_i32 s49, s19, 31 // (op 13656) 

s_xor_b32 s19, s18, s49 // (op 13656) 

s_sub_i32 s18, s19, s49 // (op 13656) 


s_mul_hi_i32 s19, s18, s14 // (op 13656) 
s_add_i32 s49, s19, s18 // (op 13656) 

s_ashr_i32 s19, s49, 31 // (op 13656) 
s_cmp_eq_i32 s14, 0 // (op 13656) 
s_mov_b32 s50, scc // (op 13656) 
s_cmp_eq_i32 s50, 0 // (op 13656) 
s_cselect_b32 s51, 0, -1 // (op 13656) 

s_and_b32 s50, s15, 31 // (op 13656) 
v_lshrrev_b32 v10, 4, v9 // (op 13797) 
s_cmp_eq_i32 s10, 0 // (op 13797) 
s_mov_b32 s52, scc // (op 13797) 
v_lshlrev_b32 v11, 2, v10 // (op 13797) 

s_cmp_eq_i32 s52, 0 // (op 13797) 
s_cselect_b32 s53, 0, -1 // (op 13797) 

s_lshl_b32 s52, 1, s50 // (op 13656) 
v_add_u32 v10, v0, v11 // (op 13797) 


v_lshrrev_b32 v0, 2, v9 // (op 13797) 

s_add_i32 s54, s52, s51 // (op 13656) 


s_and_b32 s51, s19, s54 // (op 13656) 


s_add_i32 s19, s49, s51 // (op 13656) 


s_ashr_i32 s49, s19, s50 // (op 13656) 


s_lshl_b32 s19, s15, 24 // (op 13656) 
s_ashr_i32 s50, s19, 31 // (op 13656) 

s_xor_b32 s19, s49, s50 // (op 13656) 

s_sub_i32 s49, s19, s50 // (op 13656) 


s_mul_i32 s19, s18, s11 // (op 13656) 
s_sub_u32 s50, s16, s19 // (op 13656) 

s_mul_i32 s19, s49, s11 // (op 13656) 
s_add_u32 s19, s19, s50 // (op 13656) 

s_mul_hi_i32 s50, s16, s9 // (op 13656) 
s_add_i32 s51, s50, s16 // (op 13656) 

s_ashr_i32 s50, s51, 31 // (op 13656) 
s_cmp_eq_i32 s9, 0 // (op 13656) 
s_mov_b32 s52, scc // (op 13656) 
s_cmp_eq_i32 s52, 0 // (op 13656) 
s_cselect_b32 s54, 0, -1 // (op 13656) 

s_and_b32 s52, s13, 31 // (op 13797) 
v_and_b32 v9, 3, v0 // (op 13797) 

s_lshl_b32 s55, 1, s52 // (op 13797) 
v_and_b32 v0, 3, v1 // (op 13797) 
s_add_i32 s56, s55, s53 // (op 13797) 


s_and_b32 s53, s1, s56 // (op 13797) 


v_lshl_add_u32 v11, v9, 2, v0 // (op 13797) 


s_add_i32 s1, s0, s53 // (op 13797) 


v_lshrrev_b32 v0, 6, v1 // (op 13797) 
s_ashr_i32 s0, s1, s52 // (op 13797) 


v_and_b32 v9, 1, v0 // (op 13797) 

s_lshl_b32 s1, s13, 24 // (op 13797) 
v_lshl_add_u32 v0, v9, 6, v11 // (op 13797) 


s_ashr_i32 s52, s1, 31 // (op 13797) 

s_xor_b32 s1, s0, s52 // (op 13797) 

s_sub_i32 s0, s1, s52 // (op 13797) 


s_mul_hi_i32 s1, s0, s14 // (op 13797) 
s_add_i32 s52, s1, s0 // (op 13797) 

s_ashr_i32 s1, s52, 31 // (op 13797) 
s_cmp_eq_i32 s14, 0 // (op 13797) 
s_mov_b32 s53, scc // (op 13797) 
s_cmp_eq_i32 s53, 0 // (op 13797) 
s_cselect_b32 s55, 0, -1 // (op 13797) 

s_and_b32 s53, s15, 31 // (op 13797) 
s_lshl_b32 s56, 1, s53 // (op 13797) 
s_add_i32 s57, s56, s55 // (op 13797) 


s_and_b32 s55, s1, s57 // (op 13797) 


s_add_i32 s1, s52, s55 // (op 13797) 


s_ashr_i32 s52, s1, s53 // (op 13797) 


s_lshl_b32 s1, s15, 24 // (op 13797) 
s_ashr_i32 s53, s1, 31 // (op 13797) 

s_xor_b32 s1, s52, s53 // (op 13797) 

s_sub_i32 s52, s1, s53 // (op 13797) 


s_mul_i32 s1, s0, s11 // (op 13797) 
s_sub_u32 s53, s16, s1 // (op 13797) 

s_mul_i32 s1, s52, s11 // (op 13797) 
s_add_u32 s1, s1, s53 // (op 13797) 

s_mul_hi_i32 s53, s16, s9 // (op 13797) 
s_add_i32 s55, s53, s16 // (op 13797) 

s_ashr_i32 s53, s55, 31 // (op 13797) 
s_cmp_eq_i32 s9, 0 // (op 13797) 
s_mov_b32 s56, scc // (op 13797) 
s_cmp_eq_i32 s56, 0 // (op 13797) 
s_cselect_b32 s57, 0, -1 // (op 13797) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 13656) 
s_lshl_b32 s58, 1, s56 // (op 13656) 
s_add_i32 s59, s58, s54 // (op 13656) 


s_and_b32 s54, s50, s59 // (op 13656) 


s_add_i32 s50, s51, s54 // (op 13656) 


s_ashr_i32 s51, s50, s56 // (op 13656) 


s_lshl_b32 s50, s45, 24 // (op 13656) 
s_ashr_i32 s54, s50, 31 // (op 13656) 

s_xor_b32 s50, s51, s54 // (op 13656) 

s_sub_i32 s51, s50, s54 // (op 13656) 


s_and_b32 s50, s45, 31 // (op 13797) 
s_lshl_b32 s54, 1, s50 // (op 13797) 
s_add_i32 s56, s54, s57 // (op 13797) 


s_and_b32 s54, s53, s56 // (op 13797) 


s_add_i32 s53, s55, s54 // (op 13797) 


s_ashr_i32 s54, s53, s50 // (op 13797) 


s_lshl_b32 s50, s45, 24 // (op 13797) 
s_ashr_i32 s53, s50, 31 // (op 13797) 

s_xor_b32 s50, s54, s53 // (op 13797) 

s_sub_i32 s54, s50, s53 // (op 13797) 


s_add_u32 s56, s6, 128 // (op 13656) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 13656) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13656) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13656) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 13656) 

s_mul_i32 s50, s11, s56 // (op 13656) 
s_mul_i32 s53, s8, s50 // (op 13656) 

s_mul_i32 s50, s51, s53 // (op 13656) 

s_sub_u32 s53, s16, s50 // (op 13656) 

s_mov_b32 s50, s53 // (op 13656) 
s_mul_hi_i32 s55, s50, s12 // (op 13656) 
s_add_i32 s58, s55, s50 // (op 13656) 


s_ashr_i32 s50, s58, 31 // (op 13656) 
s_cmp_eq_i32 s12, 0 // (op 13656) 
s_mov_b32 s55, scc // (op 13656) 
s_cmp_eq_i32 s55, 0 // (op 13656) 
s_cselect_b32 s59, 0, -1 // (op 13656) 

s_and_b32 s55, s44, 31 // (op 13656) 
s_lshl_b32 s60, 1, s55 // (op 13656) 
s_add_i32 s61, s60, s59 // (op 13656) 


s_and_b32 s59, s50, s61 // (op 13656) 


s_add_i32 s50, s58, s59 // (op 13656) 


s_ashr_i32 s58, s50, s55 // (op 13656) 


s_lshl_b32 s50, s44, 24 // (op 13656) 
s_ashr_i32 s55, s50, 31 // (op 13656) 

s_xor_b32 s50, s58, s55 // (op 13656) 

s_sub_i32 s58, s50, s55 // (op 13656) 


s_mul_i32 s50, s46, s11 // (op 13656) 
s_add_u32 s60, s20, 256 // (op 13656) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 13656) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 13656) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 13656) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 13656) 

s_sub_i32 s55, s60, s50 // (op 13656) 

s_mul_i32 s59, s58, s55 // (op 13656) 

s_sub_u32 s55, s53, s59 // (op 13656) 


s_add_u32 s53, s55, s50 // (op 13656) 


s_cmp_eq_i32 s51, 0 // (op 13656) 
s_mov_b32 s50, scc // (op 13656) 

s_cmp_eq_i32 s50, 0 // (op 13656) 
s_cselect_b32 s51, s53, s19 // (op 13656) 


s_mul_i32 s19, s49, s56 // (op 13656) 


v_lshl_add_u32 v9, s51, 8, v8 // (op 13656) 


s_sub_i32 s49, s18, s19 // (op 13656) 


s_cmp_eq_i32 s50, 0 // (op 13656) 
s_cselect_b32 s51, s58, s49 // (op 13656) 



v_lshl_add_u32 v8, s51, 7, v7 // (op 13656) 


v_mul_lo_u32 v7, v8, s2 // (op 13656) 

v_add_lshl_u32 v8, v9, v7, 2 // (op 13656) 


s_add_u32 s18, s6, 128 // (op 13797) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 13797) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 13797) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 13797) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 7 // (op 13797) 

s_mul_i32 s53, s11, s18 // (op 13797) 
s_mul_i32 s49, s8, s53 // (op 13797) 

s_mul_i32 s53, s54, s49 // (op 13797) 

s_sub_u32 s49, s16, s53 // (op 13797) 

s_mov_b32 s53, s49 // (op 13797) 
s_mul_hi_i32 s50, s53, s12 // (op 13797) 
s_add_i32 s51, s50, s53 // (op 13797) 


s_ashr_i32 s50, s51, 31 // (op 13797) 
s_cmp_eq_i32 s12, 0 // (op 13797) 
s_mov_b32 s53, scc // (op 13797) 
s_cmp_eq_i32 s53, 0 // (op 13797) 
s_cselect_b32 s55, 0, -1 // (op 13797) 

s_and_b32 s53, s44, 31 // (op 13797) 
s_lshl_b32 s56, 1, s53 // (op 13797) 
s_add_i32 s57, s56, s55 // (op 13797) 


s_and_b32 s55, s50, s57 // (op 13797) 


s_add_i32 s50, s51, s55 // (op 13797) 


s_ashr_i32 s51, s50, s53 // (op 13797) 


s_lshl_b32 s50, s44, 24 // (op 13797) 
s_ashr_i32 s53, s50, 31 // (op 13797) 

s_xor_b32 s50, s51, s53 // (op 13797) 

s_sub_i32 s51, s50, s53 // (op 13797) 


s_mul_i32 s50, s46, s11 // (op 13797) 
s_add_u32 s56, s20, 256 // (op 13797) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 13797) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13797) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13797) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 13797) 

s_sub_i32 s53, s56, s50 // (op 13797) 

s_mul_i32 s55, s51, s53 // (op 13797) 

s_sub_u32 s53, s49, s55 // (op 13797) 


s_add_u32 s49, s53, s50 // (op 13797) 


s_cmp_eq_i32 s54, 0 // (op 13797) 
s_mov_b32 s50, scc // (op 13797) 

s_cmp_eq_i32 s50, 0 // (op 13797) 
s_cselect_b32 s53, s49, s1 // (op 13797) 


s_mul_i32 s1, s52, s18 // (op 13797) 


v_lshl_add_u32 v7, s53, 8, v10 // (op 13797) 


s_sub_i32 s49, s0, s1 // (op 13797) 


s_cmp_eq_i32 s50, 0 // (op 13797) 
s_cselect_b32 s0, s51, s49 // (op 13797) 



v_lshl_add_u32 v9, s0, 7, v0 // (op 13797) 


v_mul_lo_u32 v0, v9, s4 // (op 13797) 

v_add_lshl_u32 v9, v7, v0, 2 // (op 13797) 


s_mul_i32 s1, s2, s7 // (op 13658) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 13658) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 13658) least significant
s_mul_hi_u32 s19, s2, s6 // (op 13658) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13658) most significant: sum
s_add_u32 s1, s1, s19 // (op 13658) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13658) 

s_mul_i32 s1, s4, s7 // (op 13799) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 13799) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 13799) least significant
s_mul_hi_u32 s50, s4, s6 // (op 13799) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 13799) most significant: sum
s_add_u32 s1, s1, s50 // (op 13799) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 13799) 

s_mov_b32 s52, s22 // (op 13658) reg expression
s_mov_b32 s53, s23 // (op 13658) reg expression
s_mov_b32 s54, s18 // (op 13658) reg expression
s_mov_b32 s55, 131072 // (op 13658) call()

s_mov_b32 s56, s24 // (op 13799) reg expression

s_mov_b32 s57, s25 // (op 13799) reg expression
s_mov_b32 s58, s50 // (op 13799) reg expression
s_mov_b32 s59, 131072 // (op 13799) call()


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
v_accvgpr_write a16, 0.00000 // (op 773) call()
v_accvgpr_write a17, 0.00000 // (op 773) call()
v_accvgpr_write a18, 0.00000 // (op 773) call()
v_accvgpr_write a19, 0.00000 // (op 773) call()
v_accvgpr_write a20, 0.00000 // (op 1034) call()
v_accvgpr_write a21, 0.00000 // (op 1034) call()
v_accvgpr_write a22, 0.00000 // (op 1034) call()
v_accvgpr_write a23, 0.00000 // (op 1034) call()
v_accvgpr_write a24, 0.00000 // (op 1035) call()
v_accvgpr_write a25, 0.00000 // (op 1035) call()
v_accvgpr_write a26, 0.00000 // (op 1035) call()
v_accvgpr_write a27, 0.00000 // (op 1035) call()
v_accvgpr_write a28, 0.00000 // (op 1036) call()
v_accvgpr_write a29, 0.00000 // (op 1036) call()
v_accvgpr_write a30, 0.00000 // (op 1036) call()
v_accvgpr_write a31, 0.00000 // (op 1036) call()
v_accvgpr_write a32, 0.00000 // (op 1383) call()
v_accvgpr_write a33, 0.00000 // (op 1383) call()
v_accvgpr_write a34, 0.00000 // (op 1383) call()
v_accvgpr_write a35, 0.00000 // (op 1383) call()
v_accvgpr_write a36, 0.00000 // (op 1644) call()
v_accvgpr_write a37, 0.00000 // (op 1644) call()
v_accvgpr_write a38, 0.00000 // (op 1644) call()
v_accvgpr_write a39, 0.00000 // (op 1644) call()
v_accvgpr_write a40, 0.00000 // (op 1645) call()
v_accvgpr_write a41, 0.00000 // (op 1645) call()
v_accvgpr_write a42, 0.00000 // (op 1645) call()
v_accvgpr_write a43, 0.00000 // (op 1645) call()
v_accvgpr_write a44, 0.00000 // (op 1646) call()
v_accvgpr_write a45, 0.00000 // (op 1646) call()
v_accvgpr_write a46, 0.00000 // (op 1646) call()
v_accvgpr_write a47, 0.00000 // (op 1646) call()
v_accvgpr_write a48, 0.00000 // (op 1993) call()
v_accvgpr_write a49, 0.00000 // (op 1993) call()
v_accvgpr_write a50, 0.00000 // (op 1993) call()
v_accvgpr_write a51, 0.00000 // (op 1993) call()
v_accvgpr_write a52, 0.00000 // (op 2254) call()
v_accvgpr_write a53, 0.00000 // (op 2254) call()
v_accvgpr_write a54, 0.00000 // (op 2254) call()
v_accvgpr_write a55, 0.00000 // (op 2254) call()
v_accvgpr_write a56, 0.00000 // (op 2255) call()
v_accvgpr_write a57, 0.00000 // (op 2255) call()
v_accvgpr_write a58, 0.00000 // (op 2255) call()
v_accvgpr_write a59, 0.00000 // (op 2255) call()
v_accvgpr_write a60, 0.00000 // (op 2256) call()
v_accvgpr_write a61, 0.00000 // (op 2256) call()
v_accvgpr_write a62, 0.00000 // (op 2256) call()
v_accvgpr_write a63, 0.00000 // (op 2256) call()
v_accvgpr_write a64, 0.00000 // (op 2603) call()
v_accvgpr_write a65, 0.00000 // (op 2603) call()
v_accvgpr_write a66, 0.00000 // (op 2603) call()
v_accvgpr_write a67, 0.00000 // (op 2603) call()
v_accvgpr_write a68, 0.00000 // (op 2864) call()
v_accvgpr_write a69, 0.00000 // (op 2864) call()
v_accvgpr_write a70, 0.00000 // (op 2864) call()
v_accvgpr_write a71, 0.00000 // (op 2864) call()
v_accvgpr_write a72, 0.00000 // (op 2865) call()
v_accvgpr_write a73, 0.00000 // (op 2865) call()
v_accvgpr_write a74, 0.00000 // (op 2865) call()
v_accvgpr_write a75, 0.00000 // (op 2865) call()
v_accvgpr_write a76, 0.00000 // (op 2866) call()
v_accvgpr_write a77, 0.00000 // (op 2866) call()
v_accvgpr_write a78, 0.00000 // (op 2866) call()
v_accvgpr_write a79, 0.00000 // (op 2866) call()
v_accvgpr_write a80, 0.00000 // (op 3213) call()
v_accvgpr_write a81, 0.00000 // (op 3213) call()
v_accvgpr_write a82, 0.00000 // (op 3213) call()
v_accvgpr_write a83, 0.00000 // (op 3213) call()
v_accvgpr_write a84, 0.00000 // (op 3474) call()
v_accvgpr_write a85, 0.00000 // (op 3474) call()
v_accvgpr_write a86, 0.00000 // (op 3474) call()
v_accvgpr_write a87, 0.00000 // (op 3474) call()
v_accvgpr_write a88, 0.00000 // (op 3475) call()
v_accvgpr_write a89, 0.00000 // (op 3475) call()
v_accvgpr_write a90, 0.00000 // (op 3475) call()
v_accvgpr_write a91, 0.00000 // (op 3475) call()
v_accvgpr_write a92, 0.00000 // (op 3476) call()
v_accvgpr_write a93, 0.00000 // (op 3476) call()
v_accvgpr_write a94, 0.00000 // (op 3476) call()
v_accvgpr_write a95, 0.00000 // (op 3476) call()
v_accvgpr_write a96, 0.00000 // (op 3823) call()
v_accvgpr_write a97, 0.00000 // (op 3823) call()
v_accvgpr_write a98, 0.00000 // (op 3823) call()
v_accvgpr_write a99, 0.00000 // (op 3823) call()
v_accvgpr_write a100, 0.00000 // (op 4084) call()
v_accvgpr_write a101, 0.00000 // (op 4084) call()
v_accvgpr_write a102, 0.00000 // (op 4084) call()
v_accvgpr_write a103, 0.00000 // (op 4084) call()
v_accvgpr_write a104, 0.00000 // (op 4085) call()
v_accvgpr_write a105, 0.00000 // (op 4085) call()
v_accvgpr_write a106, 0.00000 // (op 4085) call()
v_accvgpr_write a107, 0.00000 // (op 4085) call()
v_accvgpr_write a108, 0.00000 // (op 4086) call()
v_accvgpr_write a109, 0.00000 // (op 4086) call()
v_accvgpr_write a110, 0.00000 // (op 4086) call()
v_accvgpr_write a111, 0.00000 // (op 4086) call()
v_accvgpr_write a112, 0.00000 // (op 4433) call()
v_accvgpr_write a113, 0.00000 // (op 4433) call()
v_accvgpr_write a114, 0.00000 // (op 4433) call()
v_accvgpr_write a115, 0.00000 // (op 4433) call()
v_accvgpr_write a116, 0.00000 // (op 4694) call()
v_accvgpr_write a117, 0.00000 // (op 4694) call()
v_accvgpr_write a118, 0.00000 // (op 4694) call()
v_accvgpr_write a119, 0.00000 // (op 4694) call()
v_accvgpr_write a120, 0.00000 // (op 4695) call()
v_accvgpr_write a121, 0.00000 // (op 4695) call()
v_accvgpr_write a122, 0.00000 // (op 4695) call()
v_accvgpr_write a123, 0.00000 // (op 4695) call()
v_accvgpr_write a124, 0.00000 // (op 4696) call()
v_accvgpr_write a125, 0.00000 // (op 4696) call()
v_accvgpr_write a126, 0.00000 // (op 4696) call()
v_accvgpr_write a127, 0.00000 // (op 4696) call()
s_mul_hi_i32 s0, s16, s10 // (op 13546) 
v_and_b32 v10, 4294967294, v1 // (op 13546) 
s_add_i32 s1, s0, s16 // (op 13546) 

v_and_b32 v11, -2, v1 // (op 13546) 
s_ashr_i32 s0, s1, 31 // (op 13546) 
v_sub_u32 v12, v1, v11 // (op 13546) 

s_cmp_eq_i32 s10, 0 // (op 13546) 
s_mov_b32 s18, scc // (op 13546) 
v_lshlrev_b32 v11, 6, v12 // (op 13546) 

s_cmp_eq_i32 s18, 0 // (op 13546) 
s_cselect_b32 s19, 0, -1 // (op 13546) 

v_mov_b32 v13, 0 // (op 13546) convert
v_mov_b32 v12, v11 // (op 13546) convert

s_and_b32 s18, s13, 31 // (op 13546) 
s_lshl_b32 s22, 1, s18 // (op 13546) 
s_add_i32 s23, s22, s19 // (op 13546) 


s_and_b32 s19, s0, s23 // (op 13546) 


s_add_i32 s0, s1, s19 // (op 13546) 


s_ashr_i32 s1, s0, s18 // (op 13546) 


s_lshl_b32 s0, s13, 24 // (op 13546) 
s_ashr_i32 s18, s0, 31 // (op 13546) 

s_xor_b32 s0, s1, s18 // (op 13546) 

s_sub_i32 s1, s0, s18 // (op 13546) 


s_mul_hi_i32 s0, s1, s14 // (op 13546) 
s_add_i32 s18, s0, s1 // (op 13546) 

s_ashr_i32 s0, s18, 31 // (op 13546) 
s_cmp_eq_i32 s14, 0 // (op 13546) 
s_mov_b32 s19, scc // (op 13546) 
s_cmp_eq_i32 s19, 0 // (op 13546) 
s_cselect_b32 s22, 0, -1 // (op 13546) 

s_and_b32 s19, s15, 31 // (op 13546) 
s_lshl_b32 s23, 1, s19 // (op 13546) 
s_add_i32 s24, s23, s22 // (op 13546) 


s_and_b32 s25, s0, s24 // (op 13546) 


s_add_i32 s0, s18, s25 // (op 13546) 


s_ashr_i32 s18, s0, s19 // (op 13546) 


s_lshl_b32 s0, s15, 24 // (op 13546) 
s_ashr_i32 s19, s0, 31 // (op 13546) 

s_xor_b32 s0, s18, s19 // (op 13546) 

s_sub_i32 s18, s0, s19 // (op 13546) 


s_mul_i32 s0, s1, s11 // (op 13546) 

s_sub_u32 s1, s16, s0 // (op 13546) 

s_mul_i32 s0, s18, s11 // (op 13546) 
s_add_u32 s0, s0, s1 // (op 13546) 


s_mul_hi_i32 s1, s16, s9 // (op 13546) 
s_add_i32 s18, s1, s16 // (op 13546) 

s_ashr_i32 s1, s18, 31 // (op 13546) 
s_cmp_eq_i32 s9, 0 // (op 13546) 
s_mov_b32 s19, scc // (op 13546) 
s_cmp_eq_i32 s19, 0 // (op 13546) 
s_cselect_b32 s22, 0, -1 // (op 13546) 

s_and_b32 s19, s45, 31 // (op 13546) 
s_lshl_b32 s23, 1, s19 // (op 13546) 
s_add_i32 s24, s23, s22 // (op 13546) 


s_and_b32 s25, s1, s24 // (op 13546) 


s_add_i32 s1, s18, s25 // (op 13546) 


s_ashr_i32 s18, s1, s19 // (op 13546) 


s_lshl_b32 s1, s45, 24 // (op 13546) 
s_ashr_i32 s19, s1, 31 // (op 13546) 

s_xor_b32 s1, s18, s19 // (op 13546) 

s_sub_i32 s18, s1, s19 // (op 13546) 


s_add_u32 s22, s6, 128 // (op 13546) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13546) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13546) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13546) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13546) 

s_mul_i32 s1, s11, s22 // (op 13546) 

s_mul_i32 s19, s8, s1 // (op 13546) 

s_mul_i32 s1, s18, s19 // (op 13546) 

s_sub_u32 s19, s16, s1 // (op 13546) 

s_mov_b32 s1, s19 // (op 13546) 
s_mul_hi_i32 s22, s1, s12 // (op 13546) 
s_add_i32 s23, s22, s1 // (op 13546) 


s_ashr_i32 s1, s23, 31 // (op 13546) 
s_cmp_eq_i32 s12, 0 // (op 13546) 
s_mov_b32 s22, scc // (op 13546) 
s_cmp_eq_i32 s22, 0 // (op 13546) 
s_cselect_b32 s24, 0, -1 // (op 13546) 

s_and_b32 s22, s44, 31 // (op 13546) 
s_lshl_b32 s25, 1, s22 // (op 13546) 
s_add_i32 s47, s25, s24 // (op 13546) 


s_and_b32 s24, s1, s47 // (op 13546) 


s_add_i32 s1, s23, s24 // (op 13546) 


s_ashr_i32 s23, s1, s22 // (op 13546) 


s_lshl_b32 s1, s44, 24 // (op 13546) 
s_ashr_i32 s22, s1, 31 // (op 13546) 

s_xor_b32 s1, s23, s22 // (op 13546) 

s_sub_i32 s23, s1, s22 // (op 13546) 


s_mul_i32 s1, s46, s11 // (op 13546) 
s_add_u32 s24, s20, 256 // (op 13546) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13546) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 13546) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 13546) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 13546) 

s_sub_i32 s22, s24, s1 // (op 13546) 

s_mul_i32 s24, s23, s22 // (op 13546) 


s_sub_u32 s25, s19, s24 // (op 13546) 


s_add_u32 s19, s25, s1 // (op 13546) 


s_cmp_eq_i32 s18, 0 // (op 13546) 
s_mov_b32 s1, scc // (op 13546) 

s_cmp_eq_i32 s1, 0 // (op 13546) 
s_cselect_b32 s18, s19, s0 // (op 13546) 



v_lshl_add_u32 v11, s18, 8, v10 // (op 13546) 


v_mul_lo_u32 v15, v11, s27 // (op 13546) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 13546) most significant: high of low * low
v_mul_lo_u32 v14, v11, s26 // (op 13546) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 13546) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 13546) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 13546) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 13546) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 13546) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 13546) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 13546) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 13546) 

v_mov_b32 v10, v12 // (op 13546) convert

s_mul_i32 s1, s21, 6 // (op 13548) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 13548) least significant
s_mul_hi_u32 s18, s20, 6 // (op 13548) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13548) most significant: sum

s_mul_i32 s19, s26, s1 // (op 13548) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 13548) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 13548) least significant
s_mul_hi_u32 s23, s26, s0 // (op 13548) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 13548) most significant: sum
s_add_u32 s19, s19, s23 // (op 13548) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13548) 

s_mov_b32 s48, s28 // (op 13548) reg expression
s_mov_b32 s49, s29 // (op 13548) reg expression
s_mov_b32 s50, s0 // (op 13548) reg expression
s_mov_b32 s51, 131072 // (op 13548) call()

s_mul_hi_i32 s47, s16, s10 // (op 13565) 
v_and_b32 v11, 4294967294, v1 // (op 13565) 
s_add_i32 s0, s47, s16 // (op 13565) 

v_and_b32 v12, -2, v1 // (op 13565) 
s_ashr_i32 s1, s0, 31 // (op 13565) 
v_sub_u32 v13, v1, v12 // (op 13565) 

s_cmp_eq_i32 s10, 0 // (op 13565) 
s_mov_b32 s47, scc // (op 13565) 
v_lshlrev_b32 v12, 6, v13 // (op 13565) 

s_cmp_eq_i32 s47, 0 // (op 13565) 
s_cselect_b32 s18, 0, -1 // (op 13565) 

v_mov_b32 v15, 0 // (op 13565) convert
v_mov_b32 v14, v12 // (op 13565) convert

s_and_b32 s19, s13, 31 // (op 13565) 
s_lshl_b32 s47, 1, s19 // (op 13565) 
s_add_i32 s22, s47, s18 // (op 13565) 


s_and_b32 s18, s1, s22 // (op 13565) 


s_add_i32 s1, s0, s18 // (op 13565) 


s_ashr_i32 s0, s1, s19 // (op 13565) 


s_lshl_b32 s1, s13, 24 // (op 13565) 
s_ashr_i32 s47, s1, 31 // (op 13565) 

s_xor_b32 s1, s0, s47 // (op 13565) 

s_sub_i32 s0, s1, s47 // (op 13565) 


s_mul_hi_i32 s1, s0, s14 // (op 13565) 
s_add_i32 s47, s1, s0 // (op 13565) 

s_ashr_i32 s1, s47, 31 // (op 13565) 
s_cmp_eq_i32 s14, 0 // (op 13565) 
s_mov_b32 s18, scc // (op 13565) 
s_cmp_eq_i32 s18, 0 // (op 13565) 
s_cselect_b32 s19, 0, -1 // (op 13565) 

s_and_b32 s18, s15, 31 // (op 13565) 
s_lshl_b32 s22, 1, s18 // (op 13565) 
s_add_i32 s23, s22, s19 // (op 13565) 


s_and_b32 s19, s1, s23 // (op 13565) 


s_add_i32 s1, s47, s19 // (op 13565) 


s_ashr_i32 s19, s1, s18 // (op 13565) 


s_lshl_b32 s1, s15, 24 // (op 13565) 
s_ashr_i32 s18, s1, 31 // (op 13565) 

s_xor_b32 s1, s19, s18 // (op 13565) 

s_sub_i32 s19, s1, s18 // (op 13565) 


s_mul_i32 s1, s0, s11 // (op 13565) 

s_sub_u32 s0, s16, s1 // (op 13565) 

s_mul_i32 s1, s19, s11 // (op 13565) 
s_add_u32 s1, s1, s0 // (op 13565) 


s_mul_hi_i32 s0, s16, s9 // (op 13565) 
s_add_i32 s47, s0, s16 // (op 13565) 

s_ashr_i32 s0, s47, 31 // (op 13565) 
s_cmp_eq_i32 s9, 0 // (op 13565) 
s_mov_b32 s18, scc // (op 13565) 
s_cmp_eq_i32 s18, 0 // (op 13565) 
s_cselect_b32 s19, 0, -1 // (op 13565) 

s_and_b32 s18, s45, 31 // (op 13565) 
s_lshl_b32 s22, 1, s18 // (op 13565) 
s_add_i32 s23, s22, s19 // (op 13565) 


s_and_b32 s19, s0, s23 // (op 13565) 


s_add_i32 s0, s47, s19 // (op 13565) 


s_ashr_i32 s19, s0, s18 // (op 13565) 


s_lshl_b32 s0, s45, 24 // (op 13565) 
s_ashr_i32 s18, s0, 31 // (op 13565) 

s_xor_b32 s0, s19, s18 // (op 13565) 

s_sub_i32 s19, s0, s18 // (op 13565) 


s_add_u32 s22, s6, 128 // (op 13565) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13565) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13565) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13565) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13565) 

s_mul_i32 s0, s11, s22 // (op 13565) 

s_mul_i32 s18, s8, s0 // (op 13565) 

s_mul_i32 s0, s19, s18 // (op 13565) 

s_sub_u32 s18, s16, s0 // (op 13565) 

s_mov_b32 s0, s18 // (op 13565) 
s_mul_hi_i32 s47, s0, s12 // (op 13565) 
s_add_i32 s22, s47, s0 // (op 13565) 


s_ashr_i32 s0, s22, 31 // (op 13565) 
s_cmp_eq_i32 s12, 0 // (op 13565) 
s_mov_b32 s47, scc // (op 13565) 
s_cmp_eq_i32 s47, 0 // (op 13565) 
s_cselect_b32 s23, 0, -1 // (op 13565) 

s_and_b32 s47, s44, 31 // (op 13565) 
s_lshl_b32 s24, 1, s47 // (op 13565) 
s_add_i32 s25, s24, s23 // (op 13565) 


s_and_b32 s23, s0, s25 // (op 13565) 


s_add_i32 s0, s22, s23 // (op 13565) 


s_ashr_i32 s22, s0, s47 // (op 13565) 


s_lshl_b32 s0, s44, 24 // (op 13565) 
s_ashr_i32 s47, s0, 31 // (op 13565) 

s_xor_b32 s0, s22, s47 // (op 13565) 

s_sub_i32 s22, s0, s47 // (op 13565) 


s_mul_i32 s0, s46, s11 // (op 13565) 
s_add_u32 s24, s20, 256 // (op 13565) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13565) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 13565) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 13565) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 13565) 

s_sub_i32 s23, s24, s0 // (op 13565) 

s_mul_i32 s47, s22, s23 // (op 13565) 


s_sub_u32 s22, s18, s47 // (op 13565) 


s_add_u32 s18, s22, s0 // (op 13565) 


s_cmp_eq_i32 s19, 0 // (op 13565) 
s_mov_b32 s0, scc // (op 13565) 

s_cmp_eq_i32 s0, 0 // (op 13565) 
s_cselect_b32 s19, s18, s1 // (op 13565) 



v_lshl_add_u32 v12, s19, 8, v11 // (op 13565) 


v_mul_lo_u32 v17, v12, s27 // (op 13565) most significant: low of low * high
v_mul_hi_u32 v11, v12, s26 // (op 13565) most significant: high of low * low
v_mul_lo_u32 v16, v12, s26 // (op 13565) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 13565) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 13565) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 13565) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 13565) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 13565) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 13565) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 13565) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 13565) 

v_mov_b32 v11, v12 // (op 13565) convert

s_mul_i32 s1, s21, 6 // (op 13567) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 13567) least significant
s_mul_hi_u32 s47, s20, 6 // (op 13567) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13567) most significant: sum

s_mul_i32 s19, s26, s1 // (op 13567) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 13567) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 13567) least significant
s_mul_hi_u32 s22, s26, s0 // (op 13567) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13567) most significant: sum
s_add_u32 s19, s19, s22 // (op 13567) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13567) 

s_mov_b32 s48, s28 // (op 13567) reg expression
s_mov_b32 s49, s29 // (op 13567) reg expression
s_mov_b32 s50, s0 // (op 13567) reg expression
s_mov_b32 s51, 131072 // (op 13567) call()

s_mul_hi_i32 s47, s16, s10 // (op 13610) 
v_lshrrev_b32 v12, 1, v1 // (op 13610) 
s_add_i32 s0, s47, s16 // (op 13610) 

v_and_b32 v13, -2, v1 // (op 13610) 
s_ashr_i32 s1, s0, 31 // (op 13610) 
v_sub_u32 v14, v1, v13 // (op 13610) 

s_cmp_eq_i32 s10, 0 // (op 13610) 
s_mov_b32 s47, scc // (op 13610) 
v_lshlrev_b32 v13, 6, v14 // (op 13610) 

s_cmp_eq_i32 s47, 0 // (op 13610) 
s_cselect_b32 s18, 0, -1 // (op 13610) 

v_mov_b32 v15, 0 // (op 13610) convert
v_mov_b32 v14, v13 // (op 13610) convert

s_and_b32 s19, s13, 31 // (op 13610) 
s_lshl_b32 s47, 1, s19 // (op 13610) 
s_add_i32 s22, s47, s18 // (op 13610) 


s_and_b32 s18, s1, s22 // (op 13610) 


s_add_i32 s1, s0, s18 // (op 13610) 


s_ashr_i32 s0, s1, s19 // (op 13610) 


s_lshl_b32 s1, s13, 24 // (op 13610) 
s_ashr_i32 s47, s1, 31 // (op 13610) 

s_xor_b32 s1, s0, s47 // (op 13610) 

s_sub_i32 s0, s1, s47 // (op 13610) 


s_mul_hi_i32 s1, s0, s14 // (op 13610) 
s_add_i32 s47, s1, s0 // (op 13610) 

s_ashr_i32 s1, s47, 31 // (op 13610) 
s_cmp_eq_i32 s14, 0 // (op 13610) 
s_mov_b32 s18, scc // (op 13610) 
s_cmp_eq_i32 s18, 0 // (op 13610) 
s_cselect_b32 s19, 0, -1 // (op 13610) 

s_and_b32 s18, s15, 31 // (op 13610) 
s_lshl_b32 s22, 1, s18 // (op 13610) 
s_add_i32 s23, s22, s19 // (op 13610) 


s_and_b32 s19, s1, s23 // (op 13610) 


s_add_i32 s1, s47, s19 // (op 13610) 


s_ashr_i32 s19, s1, s18 // (op 13610) 


s_lshl_b32 s1, s15, 24 // (op 13610) 
s_ashr_i32 s18, s1, 31 // (op 13610) 

s_xor_b32 s1, s19, s18 // (op 13610) 

s_sub_i32 s19, s1, s18 // (op 13610) 


s_add_u32 s22, s6, 128 // (op 13610) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13610) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13610) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13610) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13610) 

s_mul_i32 s1, s19, s22 // (op 13610) 

s_sub_i32 s47, s0, s1 // (op 13610) 


s_mul_hi_i32 s0, s16, s9 // (op 13610) 
s_add_i32 s1, s0, s16 // (op 13610) 

s_ashr_i32 s0, s1, 31 // (op 13610) 
s_cmp_eq_i32 s9, 0 // (op 13610) 
s_mov_b32 s18, scc // (op 13610) 
s_cmp_eq_i32 s18, 0 // (op 13610) 
s_cselect_b32 s19, 0, -1 // (op 13610) 

s_and_b32 s18, s45, 31 // (op 13610) 
s_lshl_b32 s24, 1, s18 // (op 13610) 
s_add_i32 s25, s24, s19 // (op 13610) 


s_and_b32 s19, s0, s25 // (op 13610) 


s_add_i32 s0, s1, s19 // (op 13610) 


s_ashr_i32 s1, s0, s18 // (op 13610) 


s_lshl_b32 s0, s45, 24 // (op 13610) 
s_ashr_i32 s18, s0, 31 // (op 13610) 

s_xor_b32 s0, s1, s18 // (op 13610) 

s_sub_i32 s1, s0, s18 // (op 13610) 


s_mul_i32 s0, s11, s22 // (op 13610) 

s_mul_i32 s18, s8, s0 // (op 13610) 

s_mul_i32 s0, s1, s18 // (op 13610) 

s_sub_u32 s18, s16, s0 // (op 13610) 

s_mul_hi_i32 s0, s18, s12 // (op 13610) 
s_add_i32 s19, s0, s18 // (op 13610) 


s_ashr_i32 s0, s19, 31 // (op 13610) 
s_cmp_eq_i32 s12, 0 // (op 13610) 
s_mov_b32 s18, scc // (op 13610) 
s_cmp_eq_i32 s18, 0 // (op 13610) 
s_cselect_b32 s22, 0, -1 // (op 13610) 

s_and_b32 s18, s44, 31 // (op 13610) 
s_lshl_b32 s23, 1, s18 // (op 13610) 
s_add_i32 s24, s23, s22 // (op 13610) 


s_and_b32 s25, s0, s24 // (op 13610) 


s_add_i32 s0, s19, s25 // (op 13610) 


s_ashr_i32 s19, s0, s18 // (op 13610) 


s_lshl_b32 s0, s44, 24 // (op 13610) 
s_ashr_i32 s18, s0, 31 // (op 13610) 

s_xor_b32 s0, s19, s18 // (op 13610) 

s_sub_i32 s19, s0, s18 // (op 13610) 


s_cmp_eq_i32 s1, 0 // (op 13610) 
s_mov_b32 s0, scc // (op 13610) 

s_cmp_eq_i32 s0, 0 // (op 13610) 
s_cselect_b32 s1, s19, s47 // (op 13610) 



v_lshl_add_u32 v13, s1, 7, v12 // (op 13610) 


v_mul_lo_u32 v17, v13, s31 // (op 13610) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 13610) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 13610) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 13610) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 13610) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 13610) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 13610) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 13610) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 13610) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 13610) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 13610) 

v_mov_b32 v14, v12 // (op 13610) convert

s_mul_i32 s1, s7, 6 // (op 13612) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 13612) least significant
s_mul_hi_u32 s47, s6, 6 // (op 13612) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13612) most significant: sum

s_mul_i32 s19, s30, s1 // (op 13612) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 13612) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 13612) least significant
s_mul_hi_u32 s22, s30, s0 // (op 13612) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13612) most significant: sum
s_add_u32 s19, s19, s22 // (op 13612) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13612) 

s_mov_b32 s60, s32 // (op 13612) reg expression
s_mov_b32 s61, s33 // (op 13612) reg expression
s_mov_b32 s62, s0 // (op 13612) reg expression
s_mov_b32 s63, 131072 // (op 13612) call()

s_mul_hi_i32 s47, s16, s10 // (op 13629) 
v_lshrrev_b32 v12, 1, v1 // (op 13629) 
s_add_i32 s0, s47, s16 // (op 13629) 

v_and_b32 v13, -2, v1 // (op 13629) 
s_ashr_i32 s1, s0, 31 // (op 13629) 
v_sub_u32 v15, v1, v13 // (op 13629) 

s_cmp_eq_i32 s10, 0 // (op 13629) 
s_mov_b32 s47, scc // (op 13629) 
v_lshlrev_b32 v13, 6, v15 // (op 13629) 

s_cmp_eq_i32 s47, 0 // (op 13629) 
s_cselect_b32 s18, 0, -1 // (op 13629) 

v_mov_b32 v17, 0 // (op 13629) convert
v_mov_b32 v16, v13 // (op 13629) convert

s_and_b32 s19, s13, 31 // (op 13629) 
s_lshl_b32 s47, 1, s19 // (op 13629) 
s_add_i32 s22, s47, s18 // (op 13629) 


s_and_b32 s18, s1, s22 // (op 13629) 


s_add_i32 s1, s0, s18 // (op 13629) 


s_ashr_i32 s0, s1, s19 // (op 13629) 


s_lshl_b32 s1, s13, 24 // (op 13629) 
s_ashr_i32 s47, s1, 31 // (op 13629) 

s_xor_b32 s1, s0, s47 // (op 13629) 

s_sub_i32 s0, s1, s47 // (op 13629) 


s_mul_hi_i32 s1, s0, s14 // (op 13629) 
s_add_i32 s47, s1, s0 // (op 13629) 

s_ashr_i32 s1, s47, 31 // (op 13629) 
s_cmp_eq_i32 s14, 0 // (op 13629) 
s_mov_b32 s18, scc // (op 13629) 
s_cmp_eq_i32 s18, 0 // (op 13629) 
s_cselect_b32 s19, 0, -1 // (op 13629) 

s_and_b32 s18, s15, 31 // (op 13629) 
s_lshl_b32 s22, 1, s18 // (op 13629) 
s_add_i32 s23, s22, s19 // (op 13629) 


s_and_b32 s19, s1, s23 // (op 13629) 


s_add_i32 s1, s47, s19 // (op 13629) 


s_ashr_i32 s19, s1, s18 // (op 13629) 


s_lshl_b32 s1, s15, 24 // (op 13629) 
s_ashr_i32 s18, s1, 31 // (op 13629) 

s_xor_b32 s1, s19, s18 // (op 13629) 

s_sub_i32 s19, s1, s18 // (op 13629) 


s_add_u32 s22, s6, 128 // (op 13629) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13629) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13629) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13629) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13629) 

s_mul_i32 s1, s19, s22 // (op 13629) 

s_sub_i32 s47, s0, s1 // (op 13629) 


s_mul_hi_i32 s0, s16, s9 // (op 13629) 
s_add_i32 s1, s0, s16 // (op 13629) 

s_ashr_i32 s0, s1, 31 // (op 13629) 
s_cmp_eq_i32 s9, 0 // (op 13629) 
s_mov_b32 s18, scc // (op 13629) 
s_cmp_eq_i32 s18, 0 // (op 13629) 
s_cselect_b32 s19, 0, -1 // (op 13629) 

s_and_b32 s18, s45, 31 // (op 13629) 
s_lshl_b32 s24, 1, s18 // (op 13629) 
s_add_i32 s25, s24, s19 // (op 13629) 


s_and_b32 s19, s0, s25 // (op 13629) 


s_add_i32 s0, s1, s19 // (op 13629) 


s_ashr_i32 s1, s0, s18 // (op 13629) 


s_lshl_b32 s0, s45, 24 // (op 13629) 
s_ashr_i32 s18, s0, 31 // (op 13629) 

s_xor_b32 s0, s1, s18 // (op 13629) 

s_sub_i32 s1, s0, s18 // (op 13629) 


s_mul_i32 s0, s11, s22 // (op 13629) 

s_mul_i32 s18, s8, s0 // (op 13629) 

s_mul_i32 s0, s1, s18 // (op 13629) 

s_sub_u32 s18, s16, s0 // (op 13629) 

s_mul_hi_i32 s0, s18, s12 // (op 13629) 
s_add_i32 s19, s0, s18 // (op 13629) 


s_ashr_i32 s0, s19, 31 // (op 13629) 
s_cmp_eq_i32 s12, 0 // (op 13629) 
s_mov_b32 s18, scc // (op 13629) 
s_cmp_eq_i32 s18, 0 // (op 13629) 
s_cselect_b32 s22, 0, -1 // (op 13629) 

s_and_b32 s18, s44, 31 // (op 13629) 
s_lshl_b32 s23, 1, s18 // (op 13629) 
s_add_i32 s24, s23, s22 // (op 13629) 


s_and_b32 s25, s0, s24 // (op 13629) 


s_add_i32 s0, s19, s25 // (op 13629) 


s_ashr_i32 s19, s0, s18 // (op 13629) 


s_lshl_b32 s0, s44, 24 // (op 13629) 
s_ashr_i32 s18, s0, 31 // (op 13629) 

s_xor_b32 s0, s19, s18 // (op 13629) 

s_sub_i32 s19, s0, s18 // (op 13629) 


s_cmp_eq_i32 s1, 0 // (op 13629) 
s_mov_b32 s0, scc // (op 13629) 

s_cmp_eq_i32 s0, 0 // (op 13629) 
s_cselect_b32 s1, s19, s47 // (op 13629) 



v_lshl_add_u32 v13, s1, 7, v12 // (op 13629) 


v_mul_lo_u32 v19, v13, s31 // (op 13629) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 13629) most significant: high of low * low
v_mul_lo_u32 v18, v13, s30 // (op 13629) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 13629) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 13629) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 13629) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 13629) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 13629) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 13629) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 13629) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 13629) 

v_mov_b32 v15, v12 // (op 13629) convert

s_mul_i32 s1, s7, 6 // (op 13631) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 13631) least significant
s_mul_hi_u32 s47, s6, 6 // (op 13631) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13631) most significant: sum

s_mul_i32 s19, s30, s1 // (op 13631) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 13631) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 13631) least significant
s_mul_hi_u32 s22, s30, s0 // (op 13631) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13631) most significant: sum
s_add_u32 s19, s19, s22 // (op 13631) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13631) 

s_mov_b32 s60, s32 // (op 13631) reg expression
s_mov_b32 s61, s33 // (op 13631) reg expression
s_mov_b32 s62, s0 // (op 13631) reg expression
s_mov_b32 s63, 131072 // (op 13631) call()

s_mul_hi_i32 s47, s16, s10 // (op 13879) 
v_and_b32 v12, 63, v1 // (op 13879) 
s_add_i32 s0, s47, s16 // (op 13879) 

v_lshrrev_b32 v13, 4, v12 // (op 13879) 
s_ashr_i32 s1, s0, 31 // (op 13879) 
v_and_b32 v16, 3, v13 // (op 13879) 

s_cmp_eq_i32 s10, 0 // (op 13879) 
s_mov_b32 s47, scc // (op 13879) 
v_and_b32 v13, 15, v1 // (op 13879) 
s_cmp_eq_i32 s47, 0 // (op 13879) 
s_cselect_b32 s18, 0, -1 // (op 13879) 

v_lshl_add_u32 v17, v16, 4, v13 // (op 13879) 


s_and_b32 s19, s13, 31 // (op 13879) 
v_and_b32 v13, 4294967168, v1 // (op 13879) 
s_lshl_b32 s47, 1, s19 // (op 13879) 
v_add_u32 v16, v13, v17 // (op 13879) 


s_add_i32 s22, s47, s18 // (op 13879) 


v_lshrrev_b32 v13, 6, v12 // (op 13879) 

s_and_b32 s18, s1, s22 // (op 13879) 


v_lshlrev_b32 v12, 2, v13 // (op 13879) 

s_add_i32 s1, s0, s18 // (op 13879) 


s_ashr_i32 s0, s1, s19 // (op 13879) 


s_lshl_b32 s1, s13, 24 // (op 13879) 
s_ashr_i32 s47, s1, 31 // (op 13879) 

s_xor_b32 s1, s0, s47 // (op 13879) 

s_sub_i32 s0, s1, s47 // (op 13879) 


s_mul_hi_i32 s1, s0, s14 // (op 13879) 
s_add_i32 s47, s1, s0 // (op 13879) 

s_ashr_i32 s1, s47, 31 // (op 13879) 
s_cmp_eq_i32 s14, 0 // (op 13879) 
s_mov_b32 s18, scc // (op 13879) 
s_cmp_eq_i32 s18, 0 // (op 13879) 
s_cselect_b32 s19, 0, -1 // (op 13879) 

s_and_b32 s18, s15, 31 // (op 13879) 
s_lshl_b32 s22, 1, s18 // (op 13879) 
s_add_i32 s23, s22, s19 // (op 13879) 


s_and_b32 s19, s1, s23 // (op 13879) 


s_add_i32 s1, s47, s19 // (op 13879) 


s_ashr_i32 s19, s1, s18 // (op 13879) 


s_lshl_b32 s1, s15, 24 // (op 13879) 
s_ashr_i32 s18, s1, 31 // (op 13879) 

s_xor_b32 s1, s19, s18 // (op 13879) 

s_sub_i32 s19, s1, s18 // (op 13879) 


s_mul_i32 s1, s0, s11 // (op 13879) 

s_sub_u32 s0, s16, s1 // (op 13879) 

s_mul_i32 s1, s19, s11 // (op 13879) 
s_add_u32 s1, s1, s0 // (op 13879) 


s_mul_hi_i32 s0, s16, s9 // (op 13879) 
s_add_i32 s47, s0, s16 // (op 13879) 

s_ashr_i32 s0, s47, 31 // (op 13879) 
s_cmp_eq_i32 s9, 0 // (op 13879) 
s_mov_b32 s18, scc // (op 13879) 
s_cmp_eq_i32 s18, 0 // (op 13879) 
s_cselect_b32 s19, 0, -1 // (op 13879) 

s_and_b32 s18, s45, 31 // (op 13879) 
s_lshl_b32 s22, 1, s18 // (op 13879) 
s_add_i32 s23, s22, s19 // (op 13879) 


s_and_b32 s19, s0, s23 // (op 13879) 


s_add_i32 s0, s47, s19 // (op 13879) 


s_ashr_i32 s19, s0, s18 // (op 13879) 


s_lshl_b32 s0, s45, 24 // (op 13879) 
s_ashr_i32 s18, s0, 31 // (op 13879) 

s_xor_b32 s0, s19, s18 // (op 13879) 

s_sub_i32 s19, s0, s18 // (op 13879) 


s_add_u32 s22, s6, 128 // (op 13879) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13879) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13879) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13879) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13879) 

s_mul_i32 s0, s11, s22 // (op 13879) 

s_mul_i32 s18, s8, s0 // (op 13879) 

s_mul_i32 s0, s19, s18 // (op 13879) 

s_sub_u32 s18, s16, s0 // (op 13879) 

s_mov_b32 s0, s18 // (op 13879) 
s_mul_hi_i32 s47, s0, s12 // (op 13879) 
s_add_i32 s22, s47, s0 // (op 13879) 


s_ashr_i32 s0, s22, 31 // (op 13879) 
s_cmp_eq_i32 s12, 0 // (op 13879) 
s_mov_b32 s47, scc // (op 13879) 
s_cmp_eq_i32 s47, 0 // (op 13879) 
s_cselect_b32 s23, 0, -1 // (op 13879) 

s_and_b32 s47, s44, 31 // (op 13879) 
s_lshl_b32 s24, 1, s47 // (op 13879) 
s_add_i32 s25, s24, s23 // (op 13879) 


s_and_b32 s23, s0, s25 // (op 13879) 


s_add_i32 s0, s22, s23 // (op 13879) 


s_ashr_i32 s22, s0, s47 // (op 13879) 


s_lshl_b32 s0, s44, 24 // (op 13879) 
s_ashr_i32 s47, s0, 31 // (op 13879) 

s_xor_b32 s0, s22, s47 // (op 13879) 

s_sub_i32 s22, s0, s47 // (op 13879) 


s_mul_i32 s0, s46, s11 // (op 13879) 
s_add_u32 s24, s20, 256 // (op 13879) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13879) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13879) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13879) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 13879) 

s_sub_i32 s23, s24, s0 // (op 13879) 

s_mul_i32 s47, s22, s23 // (op 13879) 


s_sub_u32 s22, s18, s47 // (op 13879) 


s_add_u32 s18, s22, s0 // (op 13879) 


s_cmp_eq_i32 s19, 0 // (op 13879) 
s_mov_b32 s0, scc // (op 13879) 

s_cmp_eq_i32 s0, 0 // (op 13879) 
s_cselect_b32 s19, s18, s1 // (op 13879) 



v_lshl_add_u32 v13, s19, 8, v16 // (op 13879) 


v_mul_lo_u32 v16, v13, s34 // (op 13879) 
v_add_u32 v16, v16, v12 // (op 13879) 


s_mul_i32 s1, s34, s21 // (op 13881) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 13881) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 13881) least significant
s_mul_hi_u32 s18, s34, s20 // (op 13881) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13881) most significant: sum
s_add_u32 s1, s1, s18 // (op 13881) most significant: sum


s_mov_b32 s64, s36 // (op 13881) reg expression
s_mov_b32 s65, s37 // (op 13881) reg expression
s_mov_b32 s66, s0 // (op 13881) reg expression
s_mov_b32 s67, 131072 // (op 13881) call()

s_mul_hi_i32 s47, s16, s10 // (op 13900) 
v_and_b32 v12, 63, v1 // (op 13900) 
s_add_i32 s0, s47, s16 // (op 13900) 

v_lshrrev_b32 v13, 4, v12 // (op 13900) 
s_ashr_i32 s1, s0, 31 // (op 13900) 
v_and_b32 v17, 3, v13 // (op 13900) 

s_cmp_eq_i32 s10, 0 // (op 13900) 
s_mov_b32 s47, scc // (op 13900) 
v_and_b32 v13, 15, v1 // (op 13900) 
s_cmp_eq_i32 s47, 0 // (op 13900) 
s_cselect_b32 s18, 0, -1 // (op 13900) 

v_lshl_add_u32 v18, v17, 4, v13 // (op 13900) 


s_and_b32 s19, s13, 31 // (op 13900) 
v_and_b32 v13, 4294967168, v1 // (op 13900) 
s_lshl_b32 s47, 1, s19 // (op 13900) 
v_add_u32 v17, v13, v18 // (op 13900) 


s_add_i32 s22, s47, s18 // (op 13900) 


v_lshrrev_b32 v13, 6, v12 // (op 13900) 

s_and_b32 s18, s1, s22 // (op 13900) 


v_lshlrev_b32 v12, 2, v13 // (op 13900) 

s_add_i32 s1, s0, s18 // (op 13900) 


s_ashr_i32 s0, s1, s19 // (op 13900) 


s_lshl_b32 s1, s13, 24 // (op 13900) 
s_ashr_i32 s47, s1, 31 // (op 13900) 

s_xor_b32 s1, s0, s47 // (op 13900) 

s_sub_i32 s0, s1, s47 // (op 13900) 


s_mul_hi_i32 s1, s0, s14 // (op 13900) 
s_add_i32 s47, s1, s0 // (op 13900) 

s_ashr_i32 s1, s47, 31 // (op 13900) 
s_cmp_eq_i32 s14, 0 // (op 13900) 
s_mov_b32 s18, scc // (op 13900) 
s_cmp_eq_i32 s18, 0 // (op 13900) 
s_cselect_b32 s19, 0, -1 // (op 13900) 

s_and_b32 s18, s15, 31 // (op 13900) 
s_lshl_b32 s22, 1, s18 // (op 13900) 
s_add_i32 s23, s22, s19 // (op 13900) 


s_and_b32 s19, s1, s23 // (op 13900) 


s_add_i32 s1, s47, s19 // (op 13900) 


s_ashr_i32 s19, s1, s18 // (op 13900) 


s_lshl_b32 s1, s15, 24 // (op 13900) 
s_ashr_i32 s18, s1, 31 // (op 13900) 

s_xor_b32 s1, s19, s18 // (op 13900) 

s_sub_i32 s19, s1, s18 // (op 13900) 


s_mul_i32 s1, s0, s11 // (op 13900) 

s_sub_u32 s0, s16, s1 // (op 13900) 

s_mul_i32 s1, s19, s11 // (op 13900) 
s_add_u32 s1, s1, s0 // (op 13900) 


s_mul_hi_i32 s0, s16, s9 // (op 13900) 
s_add_i32 s47, s0, s16 // (op 13900) 

s_ashr_i32 s0, s47, 31 // (op 13900) 
s_cmp_eq_i32 s9, 0 // (op 13900) 
s_mov_b32 s18, scc // (op 13900) 
s_cmp_eq_i32 s18, 0 // (op 13900) 
s_cselect_b32 s19, 0, -1 // (op 13900) 

s_and_b32 s18, s45, 31 // (op 13900) 
s_lshl_b32 s22, 1, s18 // (op 13900) 
s_add_i32 s23, s22, s19 // (op 13900) 


s_and_b32 s19, s0, s23 // (op 13900) 


s_add_i32 s0, s47, s19 // (op 13900) 


s_ashr_i32 s19, s0, s18 // (op 13900) 


s_lshl_b32 s0, s45, 24 // (op 13900) 
s_ashr_i32 s18, s0, 31 // (op 13900) 

s_xor_b32 s0, s19, s18 // (op 13900) 

s_sub_i32 s19, s0, s18 // (op 13900) 


s_add_u32 s22, s6, 128 // (op 13900) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13900) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13900) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13900) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13900) 

s_mul_i32 s0, s11, s22 // (op 13900) 

s_mul_i32 s18, s8, s0 // (op 13900) 

s_mul_i32 s0, s19, s18 // (op 13900) 

s_sub_u32 s18, s16, s0 // (op 13900) 

s_mov_b32 s0, s18 // (op 13900) 
s_mul_hi_i32 s47, s0, s12 // (op 13900) 
s_add_i32 s22, s47, s0 // (op 13900) 


s_ashr_i32 s0, s22, 31 // (op 13900) 
s_cmp_eq_i32 s12, 0 // (op 13900) 
s_mov_b32 s47, scc // (op 13900) 
s_cmp_eq_i32 s47, 0 // (op 13900) 
s_cselect_b32 s23, 0, -1 // (op 13900) 

s_and_b32 s47, s44, 31 // (op 13900) 
s_lshl_b32 s24, 1, s47 // (op 13900) 
s_add_i32 s25, s24, s23 // (op 13900) 


s_and_b32 s23, s0, s25 // (op 13900) 


s_add_i32 s0, s22, s23 // (op 13900) 


s_ashr_i32 s22, s0, s47 // (op 13900) 


s_lshl_b32 s0, s44, 24 // (op 13900) 
s_ashr_i32 s47, s0, 31 // (op 13900) 

s_xor_b32 s0, s22, s47 // (op 13900) 

s_sub_i32 s22, s0, s47 // (op 13900) 


s_mul_i32 s0, s46, s11 // (op 13900) 
s_add_u32 s24, s20, 256 // (op 13900) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13900) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 13900) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 13900) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 13900) 

s_sub_i32 s23, s24, s0 // (op 13900) 

s_mul_i32 s47, s22, s23 // (op 13900) 


s_sub_u32 s22, s18, s47 // (op 13900) 


s_add_u32 s18, s22, s0 // (op 13900) 


s_cmp_eq_i32 s19, 0 // (op 13900) 
s_mov_b32 s0, scc // (op 13900) 

s_cmp_eq_i32 s0, 0 // (op 13900) 
s_cselect_b32 s19, s18, s1 // (op 13900) 



v_lshl_add_u32 v13, s19, 8, v17 // (op 13900) 


v_mul_lo_u32 v17, v13, s34 // (op 13900) 
v_add_u32 v17, v17, v12 // (op 13900) 


s_mul_i32 s1, s34, s21 // (op 13902) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 13902) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 13902) least significant
s_mul_hi_u32 s18, s34, s20 // (op 13902) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13902) most significant: sum
s_add_u32 s1, s1, s18 // (op 13902) most significant: sum


s_mov_b32 s64, s36 // (op 13902) reg expression
s_mov_b32 s65, s37 // (op 13902) reg expression
s_mov_b32 s66, s0 // (op 13902) reg expression
s_mov_b32 s67, 131072 // (op 13902) call()

s_mul_hi_i32 s47, s16, s10 // (op 13931) 
v_and_b32 v12, 63, v1 // (op 13931) 
s_add_i32 s0, s47, s16 // (op 13931) 

v_lshrrev_b32 v13, 4, v12 // (op 13931) 
s_ashr_i32 s1, s0, 31 // (op 13931) 
v_and_b32 v18, 3, v13 // (op 13931) 

s_cmp_eq_i32 s10, 0 // (op 13931) 
s_mov_b32 s47, scc // (op 13931) 
v_and_b32 v13, 15, v1 // (op 13931) 
s_cmp_eq_i32 s47, 0 // (op 13931) 
s_cselect_b32 s18, 0, -1 // (op 13931) 

v_lshl_add_u32 v19, v18, 4, v13 // (op 13931) 


s_and_b32 s19, s13, 31 // (op 13931) 
v_lshrrev_b32 v13, 6, v1 // (op 13931) 
s_lshl_b32 s47, 1, s19 // (op 13931) 
v_and_b32 v18, 1, v13 // (op 13931) 

s_add_i32 s22, s47, s18 // (op 13931) 


v_lshl_add_u32 v13, v18, 6, v19 // (op 13931) 


s_and_b32 s18, s1, s22 // (op 13931) 


v_lshrrev_b32 v18, 6, v12 // (op 13931) 

s_add_i32 s1, s0, s18 // (op 13931) 


s_ashr_i32 s0, s1, s19 // (op 13931) 


s_lshl_b32 s1, s13, 24 // (op 13931) 
s_ashr_i32 s47, s1, 31 // (op 13931) 

s_xor_b32 s1, s0, s47 // (op 13931) 

s_sub_i32 s0, s1, s47 // (op 13931) 


s_mul_hi_i32 s1, s0, s14 // (op 13931) 
s_add_i32 s47, s1, s0 // (op 13931) 

s_ashr_i32 s1, s47, 31 // (op 13931) 
s_cmp_eq_i32 s14, 0 // (op 13931) 
s_mov_b32 s18, scc // (op 13931) 
s_cmp_eq_i32 s18, 0 // (op 13931) 
s_cselect_b32 s19, 0, -1 // (op 13931) 

s_and_b32 s18, s15, 31 // (op 13931) 
s_lshl_b32 s22, 1, s18 // (op 13931) 
s_add_i32 s23, s22, s19 // (op 13931) 


s_and_b32 s19, s1, s23 // (op 13931) 


s_add_i32 s1, s47, s19 // (op 13931) 


s_ashr_i32 s19, s1, s18 // (op 13931) 


s_lshl_b32 s1, s15, 24 // (op 13931) 
s_ashr_i32 s18, s1, 31 // (op 13931) 

s_xor_b32 s1, s19, s18 // (op 13931) 

s_sub_i32 s19, s1, s18 // (op 13931) 


s_add_u32 s22, s6, 128 // (op 13931) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13931) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13931) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13931) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13931) 

s_mul_i32 s1, s19, s22 // (op 13931) 

s_sub_i32 s47, s0, s1 // (op 13931) 


s_mul_hi_i32 s0, s16, s9 // (op 13931) 
s_add_i32 s1, s0, s16 // (op 13931) 

s_ashr_i32 s0, s1, 31 // (op 13931) 
s_cmp_eq_i32 s9, 0 // (op 13931) 
s_mov_b32 s18, scc // (op 13931) 
s_cmp_eq_i32 s18, 0 // (op 13931) 
s_cselect_b32 s19, 0, -1 // (op 13931) 

s_and_b32 s18, s45, 31 // (op 13931) 
s_lshl_b32 s24, 1, s18 // (op 13931) 
s_add_i32 s25, s24, s19 // (op 13931) 


s_and_b32 s19, s0, s25 // (op 13931) 


s_add_i32 s0, s1, s19 // (op 13931) 


s_ashr_i32 s1, s0, s18 // (op 13931) 


s_lshl_b32 s0, s45, 24 // (op 13931) 
s_ashr_i32 s18, s0, 31 // (op 13931) 

s_xor_b32 s0, s1, s18 // (op 13931) 

s_sub_i32 s1, s0, s18 // (op 13931) 


s_mul_i32 s0, s11, s22 // (op 13931) 

s_mul_i32 s18, s8, s0 // (op 13931) 

s_mul_i32 s0, s1, s18 // (op 13931) 

s_sub_u32 s18, s16, s0 // (op 13931) 

s_mul_hi_i32 s0, s18, s12 // (op 13931) 
s_add_i32 s19, s0, s18 // (op 13931) 


s_ashr_i32 s0, s19, 31 // (op 13931) 
s_cmp_eq_i32 s12, 0 // (op 13931) 
s_mov_b32 s18, scc // (op 13931) 
s_cmp_eq_i32 s18, 0 // (op 13931) 
s_cselect_b32 s22, 0, -1 // (op 13931) 

s_and_b32 s18, s44, 31 // (op 13931) 
s_lshl_b32 s23, 1, s18 // (op 13931) 
s_add_i32 s24, s23, s22 // (op 13931) 


s_and_b32 s25, s0, s24 // (op 13931) 


s_add_i32 s0, s19, s25 // (op 13931) 


s_ashr_i32 s19, s0, s18 // (op 13931) 


s_lshl_b32 s0, s44, 24 // (op 13931) 
s_ashr_i32 s18, s0, 31 // (op 13931) 

s_xor_b32 s0, s19, s18 // (op 13931) 

s_sub_i32 s19, s0, s18 // (op 13931) 


s_cmp_eq_i32 s1, 0 // (op 13931) 
s_mov_b32 s0, scc // (op 13931) 

s_cmp_eq_i32 s0, 0 // (op 13931) 
s_cselect_b32 s1, s19, s47 // (op 13931) 



v_lshl_add_u32 v12, s1, 7, v13 // (op 13931) 


v_mul_lo_u32 v13, v12, s38 // (op 13931) 

v_lshl_add_u32 v12, v18, 2, v13 // (op 13931) 


s_mul_i32 s1, s38, s7 // (op 13933) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 13933) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 13933) least significant
s_mul_hi_u32 s18, s38, s6 // (op 13933) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13933) most significant: sum
s_add_u32 s1, s1, s18 // (op 13933) most significant: sum


s_mov_b32 s68, s40 // (op 13933) reg expression
s_mov_b32 s69, s41 // (op 13933) reg expression
s_mov_b32 s70, s0 // (op 13933) reg expression
s_mov_b32 s71, 131072 // (op 13933) call()

s_mul_hi_i32 s47, s16, s10 // (op 13952) 
v_and_b32 v13, 63, v1 // (op 13952) 
s_add_i32 s0, s47, s16 // (op 13952) 

v_lshrrev_b32 v18, 4, v13 // (op 13952) 
s_ashr_i32 s1, s0, 31 // (op 13952) 
v_and_b32 v19, 3, v18 // (op 13952) 

s_cmp_eq_i32 s10, 0 // (op 13952) 
s_mov_b32 s47, scc // (op 13952) 
v_and_b32 v18, 15, v1 // (op 13952) 
s_cmp_eq_i32 s47, 0 // (op 13952) 
s_cselect_b32 s18, 0, -1 // (op 13952) 

v_lshl_add_u32 v20, v19, 4, v18 // (op 13952) 


s_and_b32 s19, s13, 31 // (op 13952) 
v_lshrrev_b32 v18, 6, v1 // (op 13952) 
s_lshl_b32 s47, 1, s19 // (op 13952) 
v_and_b32 v19, 1, v18 // (op 13952) 

s_add_i32 s22, s47, s18 // (op 13952) 


v_lshl_add_u32 v18, v19, 6, v20 // (op 13952) 


s_and_b32 s18, s1, s22 // (op 13952) 


v_lshrrev_b32 v19, 6, v13 // (op 13952) 

s_add_i32 s1, s0, s18 // (op 13952) 


s_ashr_i32 s0, s1, s19 // (op 13952) 


s_lshl_b32 s1, s13, 24 // (op 13952) 
s_ashr_i32 s47, s1, 31 // (op 13952) 

s_xor_b32 s1, s0, s47 // (op 13952) 

s_sub_i32 s0, s1, s47 // (op 13952) 


s_mul_hi_i32 s1, s0, s14 // (op 13952) 
s_add_i32 s47, s1, s0 // (op 13952) 

s_ashr_i32 s1, s47, 31 // (op 13952) 
s_cmp_eq_i32 s14, 0 // (op 13952) 
s_mov_b32 s18, scc // (op 13952) 
s_cmp_eq_i32 s18, 0 // (op 13952) 
s_cselect_b32 s19, 0, -1 // (op 13952) 

s_and_b32 s18, s15, 31 // (op 13952) 
s_lshl_b32 s22, 1, s18 // (op 13952) 
s_add_i32 s23, s22, s19 // (op 13952) 


s_and_b32 s19, s1, s23 // (op 13952) 


s_add_i32 s1, s47, s19 // (op 13952) 


s_ashr_i32 s19, s1, s18 // (op 13952) 


s_lshl_b32 s1, s15, 24 // (op 13952) 
s_ashr_i32 s18, s1, 31 // (op 13952) 

s_xor_b32 s1, s19, s18 // (op 13952) 

s_sub_i32 s19, s1, s18 // (op 13952) 


s_add_u32 s22, s6, 128 // (op 13952) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13952) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13952) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13952) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13952) 

s_mul_i32 s1, s19, s22 // (op 13952) 

s_sub_i32 s47, s0, s1 // (op 13952) 


s_mul_hi_i32 s0, s16, s9 // (op 13952) 
s_add_i32 s1, s0, s16 // (op 13952) 

s_ashr_i32 s0, s1, 31 // (op 13952) 
s_cmp_eq_i32 s9, 0 // (op 13952) 
s_mov_b32 s18, scc // (op 13952) 
s_cmp_eq_i32 s18, 0 // (op 13952) 
s_cselect_b32 s19, 0, -1 // (op 13952) 

s_and_b32 s18, s45, 31 // (op 13952) 
s_lshl_b32 s24, 1, s18 // (op 13952) 
s_add_i32 s25, s24, s19 // (op 13952) 


s_and_b32 s19, s0, s25 // (op 13952) 


s_add_i32 s0, s1, s19 // (op 13952) 


s_ashr_i32 s1, s0, s18 // (op 13952) 


s_lshl_b32 s0, s45, 24 // (op 13952) 
s_ashr_i32 s18, s0, 31 // (op 13952) 

s_xor_b32 s0, s1, s18 // (op 13952) 

s_sub_i32 s1, s0, s18 // (op 13952) 


s_mul_i32 s0, s11, s22 // (op 13952) 

s_mul_i32 s18, s8, s0 // (op 13952) 

s_mul_i32 s0, s1, s18 // (op 13952) 

s_sub_u32 s18, s16, s0 // (op 13952) 

s_mul_hi_i32 s0, s18, s12 // (op 13952) 
s_add_i32 s19, s0, s18 // (op 13952) 


s_ashr_i32 s0, s19, 31 // (op 13952) 
s_cmp_eq_i32 s12, 0 // (op 13952) 
s_mov_b32 s18, scc // (op 13952) 
s_cmp_eq_i32 s18, 0 // (op 13952) 
s_cselect_b32 s22, 0, -1 // (op 13952) 

s_and_b32 s18, s44, 31 // (op 13952) 
s_lshl_b32 s23, 1, s18 // (op 13952) 
s_add_i32 s24, s23, s22 // (op 13952) 


s_and_b32 s25, s0, s24 // (op 13952) 


s_add_i32 s0, s19, s25 // (op 13952) 


s_ashr_i32 s19, s0, s18 // (op 13952) 


s_lshl_b32 s0, s44, 24 // (op 13952) 
s_ashr_i32 s18, s0, 31 // (op 13952) 

s_xor_b32 s0, s19, s18 // (op 13952) 

s_sub_i32 s19, s0, s18 // (op 13952) 


s_cmp_eq_i32 s1, 0 // (op 13952) 
s_mov_b32 s0, scc // (op 13952) 

s_cmp_eq_i32 s0, 0 // (op 13952) 
s_cselect_b32 s1, s19, s47 // (op 13952) 



v_lshl_add_u32 v13, s1, 7, v18 // (op 13952) 


v_mul_lo_u32 v18, v13, s38 // (op 13952) 

v_lshl_add_u32 v13, v19, 2, v18 // (op 13952) 


s_mul_i32 s1, s38, s7 // (op 13954) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 13954) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 13954) least significant
s_mul_hi_u32 s18, s38, s6 // (op 13954) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13954) most significant: sum
s_add_u32 s1, s1, s18 // (op 13954) most significant: sum


s_mov_b32 s68, s40 // (op 13954) reg expression
s_mov_b32 s69, s41 // (op 13954) reg expression
s_mov_b32 s70, s0 // (op 13954) reg expression
s_mov_b32 s71, 131072 // (op 13954) call()




buffer_load_dwordx4 v[18:21], v14, s[60:63], 0 offen // (op 12900) Load value
buffer_load_dwordx2 v[22:23], v14, s[60:63], 0 offen offset:16 // (op 12900) Load value
buffer_load_dwordx4 v[24:27], v14, s[60:63], 0 offen offset:24 // (op 12900) Load value
buffer_load_dwordx2 v[28:29], v14, s[60:63], 0 offen offset:40 // (op 12900) Load value

s_mul_i32 s1, s27, 6 // (op 12903) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 12903) least significant
s_mul_hi_u32 s47, s26, 6 // (op 12903) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 12903) most significant: sum

v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 12903) 


v_mov_b32 v32, v10 // (op 12903) 
buffer_load_dwordx4 v[34:37], v32, s[48:51], 0 offen // (op 12903) Load value
buffer_load_dwordx2 v[38:39], v32, s[48:51], 0 offen offset:16 // (op 12903) Load value
buffer_load_dwordx4 v[40:43], v32, s[48:51], 0 offen offset:24 // (op 12903) Load value
buffer_load_dwordx2 v[44:45], v32, s[48:51], 0 offen offset:40 // (op 12903) Load value
v_add_u32 v32, v32, v30 // (op 12903) 
buffer_load_dwordx4 v[46:49], v32, s[48:51], 0 offen // (op 12903) Load value
buffer_load_dwordx2 v[50:51], v32, s[48:51], 0 offen offset:16 // (op 12903) Load value
buffer_load_dwordx4 v[52:55], v32, s[48:51], 0 offen offset:24 // (op 12903) Load value
buffer_load_dwordx2 v[56:57], v32, s[48:51], 0 offen offset:40 // (op 12903) Load value




v_mov_b32 v30, v12 // (op 74) reg expression

buffer_load_dwordx2 v[32:33], v30, s[68:71], 0 offen // (op 74) Load value




s_lshl_b32 s47, s34, 6 // (op 2743) 
v_add_u32 v12, v16, s47 // (op 2743) 


buffer_load_dwordx2 v[30:31], v12, s[64:67], 0 offen // (op 2743) Load value



v_mov_b32 v12, v16 // (op 70) reg expression

buffer_load_dwordx2 v[58:59], v12, s[64:67], 0 offen // (op 70) Load value





buffer_load_dwordx4 v[60:63], v14, s[60:63], 0 offen offset:96 // (op 12906) Load value
buffer_load_dwordx2 v[64:65], v14, s[60:63], 0 offen offset:112 // (op 12906) Load value
buffer_load_dwordx4 v[66:69], v14, s[60:63], 0 offen offset:120 // (op 12906) Load value
buffer_load_dwordx2 v[70:71], v14, s[60:63], 0 offen offset:136 // (op 12906) Load value


s_mul_i32 s1, s27, 6 // (op 12909) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 12909) least significant
s_mul_hi_u32 s47, s26, 6 // (op 12909) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 12909) most significant: sum

v_lshrrev_b64 v[72:73], 3, s[0:1] // (op 12909) 


v_mov_b32 v12, v10 // (op 12909) 
buffer_load_dwordx4 v[74:77], v12, s[48:51], 0 offen offset:96 // (op 12909) Load value
buffer_load_dwordx2 v[78:79], v12, s[48:51], 0 offen offset:112 // (op 12909) Load value
buffer_load_dwordx4 v[80:83], v12, s[48:51], 0 offen offset:120 // (op 12909) Load value
buffer_load_dwordx2 v[84:85], v12, s[48:51], 0 offen offset:136 // (op 12909) Load value
v_add_u32 v12, v12, v72 // (op 12909) 
buffer_load_dwordx4 v[86:89], v12, s[48:51], 0 offen offset:96 // (op 12909) Load value
buffer_load_dwordx2 v[90:91], v12, s[48:51], 0 offen offset:112 // (op 12909) Load value
buffer_load_dwordx4 v[92:95], v12, s[48:51], 0 offen offset:120 // (op 12909) Load value
buffer_load_dwordx2 v[96:97], v12, s[48:51], 0 offen offset:136 // (op 12909) Load value



v_and_b32 v10, -2, v1 // (op 13759) 
v_sub_u32 v12, v1, v10 // (op 13759) 

v_lshlrev_b32 v10, 6, v12 // (op 13759) 

v_lshrrev_b32 v12, 1, v1 // (op 13759) 
v_lshl_add_u32 v14, v12, 7, v10 // (op 13759) 


v_mul_lo_u32 v10, v14, 6 // (op 13759) 

v_lshrrev_b32 v12, 3, v10 // (op 13759) 




s_waitcnt vmcnt(26)
ds_write_b128 v12, v[18:21] // (op 12911) Store local data 
s_waitcnt vmcnt(25)
ds_write_b64 v12, v[22:23] offset:16 // (op 12911) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v12, v[24:27] offset:24 // (op 12911) Store local data 
s_waitcnt vmcnt(23)
ds_write_b64 v12, v[28:29] offset:40 // (op 12911) Store local data 

v_and_b32 v10, -2, v1 // (op 13701) 
v_sub_u32 v12, v1, v10 // (op 13701) 

v_lshlrev_b32 v10, 6, v12 // (op 13701) 

v_lshrrev_b32 v12, 1, v1 // (op 13701) 
v_lshl_add_u32 v14, v12, 8, v10 // (op 13701) 


v_mul_lo_u32 v10, v14, 6 // (op 13701) 

v_lshrrev_b32 v12, 3, v10 // (op 13701) 




s_waitcnt vmcnt(22)
ds_write_b128 v12, v[34:37] offset:12288 // (op 12912) Store local data 
s_waitcnt vmcnt(21)
ds_write_b64 v12, v[38:39] offset:12304 // (op 12912) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v12, v[40:43] offset:12312 // (op 12912) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v12, v[44:45] offset:12328 // (op 12912) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v12, v[46:49] offset:12384 // (op 12912) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v12, v[50:51] offset:12400 // (op 12912) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v12, v[52:55] offset:12408 // (op 12912) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v12, v[56:57] offset:12424 // (op 12912) Store local data 

v_and_b32 v10, -2, v1 // (op 13672) 
v_sub_u32 v12, v1, v10 // (op 13672) 

v_lshlrev_b32 v10, 6, v12 // (op 13672) 

v_lshrrev_b32 v12, 1, v1 // (op 13672) 
v_lshl_add_u32 v14, v12, 8, v10 // (op 13672) 


v_mul_lo_u32 v10, v14, 6 // (op 13672) 

v_lshrrev_b32 v12, 3, v10 // (op 13672) 

v_and_b32 v10, -2, v1 // (op 13730) 
v_sub_u32 v14, v1, v10 // (op 13730) 

v_lshlrev_b32 v10, 6, v14 // (op 13730) 

v_lshrrev_b32 v14, 1, v1 // (op 13730) 
v_lshl_add_u32 v16, v14, 7, v10 // (op 13730) 


v_mul_lo_u32 v10, v16, 6 // (op 13730) 

v_lshrrev_b32 v14, 3, v10 // (op 13730) 

v_and_b32 v10, -2, v1 // (op 13811) 
v_sub_u32 v16, v1, v10 // (op 13811) 

v_lshlrev_b32 v10, 6, v16 // (op 13811) 

v_lshrrev_b32 v16, 1, v1 // (op 13811) 
v_lshl_add_u32 v72, v16, 8, v10 // (op 13811) 


v_mul_lo_u32 v10, v72, 6 // (op 13811) 

v_lshrrev_b32 v16, 3, v10 // (op 13811) 

v_and_b32 v10, -2, v1 // (op 13840) 
v_sub_u32 v72, v1, v10 // (op 13840) 

v_lshlrev_b32 v10, 6, v72 // (op 13840) 

v_lshrrev_b32 v72, 1, v1 // (op 13840) 
v_lshl_add_u32 v73, v72, 7, v10 // (op 13840) 


v_mul_lo_u32 v10, v73, 6 // (op 13840) 

v_lshrrev_b32 v72, 3, v10 // (op 13840) 

s_mov_b32 s47, 0 // (op 4435) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 4434) 
s_mov_b32 s18, s0 // (op 4434) convert

s_and_b32 s19, s18, 4294967294 // (op 4434) 

s_cmp_lt_u32 s47, s19 // (op 4434) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoop_4434 // (op 4434) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoop_4434 if false)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoop_4434:
s_add_u32 s47, s47, 2 // (op 4436) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14152) 



ds_read_b128 v[98:101], v3 offset:12288 // (op 60) Load local data 
ds_read_b64 v[102:103], v3 offset:12304 // (op 60) Load local data 
s_barrier  // (op 14165) 



ds_read_b128 v[104:107], v4 // (op 65) Load local data 
ds_read_b64 v[108:109], v4 offset:16 // (op 65) Load local data 



ds_read_b128 v[110:113], v4 offset:1536 // (op 308) Load local data 
ds_read_b64 v[114:115], v4 offset:1552 // (op 308) Load local data 
s_waitcnt vmcnt(14)
v_mov_b32 v116, v32 // (op 13415) reg expression



v_mov_b32 v117, v33 // (op 13415) reg expression
s_waitcnt vmcnt(12)
v_mov_b32 v118, v58 // (op 13442) reg expression
v_mov_b32 v119, v59 // (op 13442) reg expression
ds_read_b128 v[120:123], v4 offset:3072 // (op 444) Load local data 
ds_read_b64 v[124:125], v4 offset:3088 // (op 444) Load local data 
v_bfe_u32 v126, v118, 0, 8 // (op 12224) 



v_bfe_u32 v127, v118, 8, 8 // (op 12224) 
v_bfe_u32 v128, v118, 16, 8 // (op 12224) 
v_bfe_u32 v129, v118, 24, 8 // (op 12224) 
v_permlane16_swap_b32 v126, v127 // (op 12224) 
ds_read_b128 v[130:133], v4 offset:4608 // (op 580) Load local data 
v_permlane16_swap_b32 v128, v129 // (op 12224) 
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 12224) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 12224) 
ds_read_b64 v[134:135], v4 offset:4624 // (op 580) Load local data 



v_bfe_u32 v136, v116, 0, 8 // (op 12351) 
v_bfe_u32 v137, v116, 8, 8 // (op 12351) 
v_bfe_u32 v138, v116, 16, 8 // (op 12351) 
v_bfe_u32 v139, v116, 24, 8 // (op 12351) 
v_permlane16_swap_b32 v136, v137 // (op 12351) 
ds_read_b128 v[140:143], v3 offset:13824 // (op 885) Load local data 
ds_read_b64 v[144:145], v3 offset:13840 // (op 885) Load local data 
v_permlane16_swap_b32 v138, v139 // (op 12351) 



s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 12351) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v137, v139 // (op 12351) 
ds_read_b128 v[146:149], v3 offset:15360 // (op 1495) Load local data 
ds_read_b64 v[150:151], v3 offset:15376 // (op 1495) Load local data 



s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:103], v[104:109], a[0:3], v126, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 76) 
ds_read_b128 v[152:155], v3 offset:16896 // (op 2105) Load local data 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:103], v[110:115], a[4:7], v126, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 311) 
ds_read_b64 v[156:157], v3 offset:16912 // (op 2105) Load local data 



s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:103], v[120:125], a[8:11], v126, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 447) 
ds_read_b128 v[158:161], v3 offset:18432 // (op 2715) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:103], v[130:135], a[12:15], v126, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 583) 
ds_read_b64 v[162:163], v3 offset:18448 // (op 2715) Load local data 






v_mov_b32 v10, v15 // (op 10) 

s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:145], v[110:115], a[20:23], v127, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 965) 
buffer_load_dwordx4 v[18:21], v10, s[60:63], 0 offen offset:192 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:145], v[120:125], a[24:27], v127, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 927) 
buffer_load_dwordx2 v[22:23], v10, s[60:63], 0 offen offset:208 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:145], v[130:135], a[28:31], v127, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 889) 
buffer_load_dwordx4 v[24:27], v10, s[60:63], 0 offen offset:216 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:145], v[104:109], a[16:19], v127, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1003) 
v_mov_b32 v98, v30 // (op 13428) reg expression

v_mov_b32 v99, v31 // (op 13428) reg expression
v_bfe_u32 v100, v98, 0, 8 // (op 12255) 
v_bfe_u32 v101, v98, 8, 8 // (op 12255) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[146:151], v[104:109], a[32:35], v128, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1613) 
v_bfe_u32 v102, v98, 16, 8 // (op 12255) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[146:151], v[110:115], a[36:39], v128, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1575) 
v_bfe_u32 v103, v98, 24, 8 // (op 12255) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[146:151], v[120:125], a[40:43], v128, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1537) 
v_permlane16_swap_b32 v100, v101 // (op 12255) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[146:151], v[130:135], a[44:47], v128, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1499) 
v_permlane16_swap_b32 v102, v103 // (op 12255) 

s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 12255) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 12255) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:157], v[104:109], a[48:51], v129, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2223) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:157], v[110:115], a[52:55], v129, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2185) 
ds_read_b128 v[140:143], v3 offset:19968 // (op 3325) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[152:157], v[120:125], a[56:59], v129, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2147) 
ds_read_b64 v[144:145], v3 offset:19984 // (op 3325) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[152:157], v[130:135], a[60:63], v129, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2109) 
ds_read_b128 v[146:149], v3 offset:21504 // (op 3935) Load local data 


ds_read_b64 v[150:151], v3 offset:21520 // (op 3935) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[158:163], v[110:115], a[68:71], v100, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2795) 
ds_read_b128 v[152:155], v3 offset:23040 // (op 4545) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[158:163], v[120:125], a[72:75], v100, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2757) 
ds_read_b64 v[156:157], v3 offset:23056 // (op 4545) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[158:163], v[130:135], a[76:79], v100, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2719) 
buffer_load_dwordx2 v[28:29], v10, s[60:63], 0 offen offset:232 // (op 10) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[158:163], v[104:109], a[64:67], v100, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2833) 
s_mul_i32 s1, s27, 6 // (op 4) most significant: low of high * low

s_mul_i32 s0, s26, 6 // (op 4) least significant
s_mul_hi_u32 s18, s26, 6 // (op 4) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4) most significant: sum

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:145], v[110:115], a[84:87], v101, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3405) 
v_lshrrev_b64 v[126:127], 3, s[0:1] // (op 4) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[120:125], a[88:91], v101, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3367) 
v_mov_b32 v73, v11 // (op 4) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:145], v[130:135], a[92:95], v101, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3329) 
v_mov_b32 v128, v73 // (op 4) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:145], v[104:109], a[80:83], v101, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3443) 
buffer_load_dwordx4 v[34:37], v128, s[48:51], 0 offen offset:192 // (op 4) Load value

buffer_load_dwordx2 v[38:39], v128, s[48:51], 0 offen offset:208 // (op 4) Load value
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[110:115], a[100:103], v102, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4015) 
buffer_load_dwordx4 v[40:43], v128, s[48:51], 0 offen offset:216 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[120:125], a[104:107], v102, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3977) 
buffer_load_dwordx2 v[44:45], v128, s[48:51], 0 offen offset:232 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[130:135], a[108:111], v102, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3939) 
v_add_u32 v128, v128, v126 // (op 4) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[104:109], a[96:99], v102, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4053) 
buffer_load_dwordx4 v[46:49], v128, s[48:51], 0 offen offset:192 // (op 4) Load value

buffer_load_dwordx2 v[50:51], v128, s[48:51], 0 offen offset:208 // (op 4) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[152:157], v[110:115], a[116:119], v103, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4625) 
buffer_load_dwordx4 v[52:55], v128, s[48:51], 0 offen offset:216 // (op 4) Load value

buffer_load_dwordx2 v[56:57], v128, s[48:51], 0 offen offset:232 // (op 4) Load value




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[152:157], v[120:125], a[120:123], v103, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4587) 
s_barrier  // (op 14188) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[152:157], v[130:135], a[124:127], v103, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4549) 
v_mov_b32 v110, v13 // (op 13423) 

v_add_u32 v111, 8, v110 // (op 13423) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[152:157], v[104:109], a[112:115], v103, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4663) 
s_waitcnt vmcnt(23)
ds_write_b128 v72, v[60:63] offset:36864 // (op 218) Store local data 




s_waitcnt vmcnt(22)
ds_write_b64 v72, v[64:65] offset:36880 // (op 218) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v72, v[66:69] offset:36888 // (op 218) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v72, v[70:71] offset:36904 // (op 218) Store local data 
buffer_load_dwordx2 v[32:33], v111, s[68:71], 0 offen // (op 13423) Load value


v_add_u32 v13, 8, v13 // (op 13942) 


s_barrier  // (op 14182) 
v_mov_b32 v100, v17 // (op 13437) 



s_lshl_b32 s0, s34, 6 // (op 13437) 
s_add_u32 s1, 8, s0 // (op 13437) 

v_add_u32 v101, v100, s1 // (op 13437) 


buffer_load_dwordx2 v[30:31], v101, s[64:67], 0 offen // (op 13437) Load value

s_waitcnt vmcnt(21)
ds_write_b128 v16, v[74:77] offset:49152 // (op 208) Store local data 


v_add_u32 v101, 8, v100 // (op 13450) 

buffer_load_dwordx2 v[58:59], v101, s[64:67], 0 offen // (op 13450) Load value

s_waitcnt vmcnt(21)
ds_write_b64 v16, v[78:79] offset:49168 // (op 208) Store local data 

s_waitcnt vmcnt(20)
ds_write_b128 v16, v[80:83] offset:49176 // (op 208) Store local data 
v_add_u32 v17, 8, v17 // (op 13890) 
s_waitcnt vmcnt(19)
ds_write_b64 v16, v[84:85] offset:49192 // (op 208) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v16, v[86:89] offset:49248 // (op 208) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v16, v[90:91] offset:49264 // (op 208) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v16, v[92:95] offset:49272 // (op 208) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v16, v[96:97] offset:49288 // (op 208) Store local data 



s_waitcnt lgkmcnt(0)
s_barrier  // (op 14179) 
buffer_load_dwordx4 v[60:63], v10, s[60:63], 0 offen offset:288 // (op 219) Load value



ds_read_b128 v[100:103], v5 offset:49152 // (op 210) Load local data 
ds_read_b64 v[104:105], v5 offset:49168 // (op 210) Load local data 
buffer_load_dwordx2 v[64:65], v10, s[60:63], 0 offen offset:304 // (op 219) Load value
s_barrier  // (op 14185) 
buffer_load_dwordx4 v[66:69], v10, s[60:63], 0 offen offset:312 // (op 219) Load value



ds_read_b128 v[106:109], v6 offset:36864 // (op 212) Load local data 
ds_read_b64 v[110:111], v6 offset:36880 // (op 212) Load local data 
buffer_load_dwordx2 v[70:71], v10, s[60:63], 0 offen offset:328 // (op 219) Load value




v_bfe_u32 v112, v119, 0, 8 // (op 12239) 

v_bfe_u32 v113, v119, 8, 8 // (op 12239) 
v_bfe_u32 v114, v119, 16, 8 // (op 12239) 
v_bfe_u32 v115, v119, 24, 8 // (op 12239) 
v_permlane16_swap_b32 v112, v113 // (op 12239) 
v_add_u32 v15, 192, v15 // (op 13621) 
v_permlane16_swap_b32 v114, v115 // (op 12239) 
s_mul_i32 s1, s27, 6 // (op 207) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 207) least significant
v_permlane32_swap_b32 v112, v114 // (op 12239) 
v_permlane32_swap_b32 v113, v115 // (op 12239) 
s_mul_hi_u32 s18, s26, 6 // (op 207) most significant: high of low * low

s_add_u32 s1, s1, s18 // (op 207) most significant: sum

v_lshrrev_b64 v[118:119], 3, s[0:1] // (op 207) 


v_mov_b32 v10, v73 // (op 207) 
v_bfe_u32 v120, v117, 0, 8 // (op 12366) 
v_bfe_u32 v121, v117, 8, 8 // (op 12366) 
v_bfe_u32 v122, v117, 16, 8 // (op 12366) 
v_bfe_u32 v123, v117, 24, 8 // (op 12366) 
v_permlane16_swap_b32 v120, v121 // (op 12366) 
ds_read_b128 v[124:127], v6 offset:38400 // (op 320) Load local data 
ds_read_b64 v[128:129], v6 offset:38416 // (op 320) Load local data 
buffer_load_dwordx4 v[74:77], v10, s[48:51], 0 offen offset:288 // (op 207) Load value



v_permlane16_swap_b32 v122, v123 // (op 12366) 
ds_read_b128 v[130:133], v6 offset:39936 // (op 456) Load local data 
s_nop 0
v_permlane32_swap_b32 v120, v122 // (op 12366) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v121, v123 // (op 12366) 
buffer_load_dwordx2 v[78:79], v10, s[48:51], 0 offen offset:304 // (op 207) Load value

ds_read_b64 v[134:135], v6 offset:39952 // (op 456) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[100:105], v[106:111], a[0:3], v112, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 217) 
ds_read_b128 v[136:139], v6 offset:41472 // (op 592) Load local data 
ds_read_b64 v[140:141], v6 offset:41488 // (op 592) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[100:105], v[124:129], a[4:7], v112, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 312) 



s_barrier  // (op 14233) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:105], v[130:135], a[8:11], v112, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 448) 
ds_read_b128 v[142:145], v5 offset:50688 // (op 897) Load local data 
ds_read_b64 v[146:147], v5 offset:50704 // (op 897) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:105], v[136:141], a[12:15], v112, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 584) 




ds_read_b128 v[100:103], v5 offset:52224 // (op 1507) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[142:147], v[106:111], a[16:19], v113, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1004) 
ds_read_b64 v[104:105], v5 offset:52240 // (op 1507) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[142:147], v[130:135], a[24:27], v113, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 928) 
ds_read_b128 v[148:151], v5 offset:53760 // (op 2117) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[142:147], v[136:141], a[28:31], v113, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 890) 
ds_read_b64 v[152:153], v5 offset:53776 // (op 2117) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[142:147], v[124:129], a[20:23], v113, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 966) 
ds_read_b128 v[154:157], v5 offset:55296 // (op 2727) Load local data 

ds_read_b64 v[158:159], v5 offset:55312 // (op 2727) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[100:105], v[106:111], a[32:35], v114, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1614) 
v_bfe_u32 v142, v99, 0, 8 // (op 12270) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:105], v[130:135], a[40:43], v114, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1538) 
v_bfe_u32 v143, v99, 8, 8 // (op 12270) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:105], v[136:141], a[44:47], v114, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1500) 
v_bfe_u32 v144, v99, 16, 8 // (op 12270) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[100:105], v[124:129], a[36:39], v114, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1576) 
v_bfe_u32 v145, v99, 24, 8 // (op 12270) 

v_permlane16_swap_b32 v142, v143 // (op 12270) 
ds_read_b128 v[100:103], v5 offset:56832 // (op 3337) Load local data 
v_permlane16_swap_b32 v144, v145 // (op 12270) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:153], v[106:111], a[48:51], v115, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2224) 
s_nop 0
v_permlane32_swap_b32 v142, v144 // (op 12270) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:153], v[130:135], a[56:59], v115, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2148) 
v_permlane32_swap_b32 v143, v145 // (op 12270) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:153], v[136:141], a[60:63], v115, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2110) 
ds_read_b64 v[104:105], v5 offset:56848 // (op 3337) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[148:153], v[124:129], a[52:55], v115, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2186) 
ds_read_b128 v[160:163], v5 offset:58368 // (op 3947) Load local data 


ds_read_b64 v[164:165], v5 offset:58384 // (op 3947) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[154:159], v[124:129], a[68:71], v142, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2796) 
ds_read_b128 v[112:115], v5 offset:59904 // (op 4557) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[154:159], v[106:111], a[64:67], v142, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2834) 
ds_read_b64 v[116:117], v5 offset:59920 // (op 4557) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[154:159], v[130:135], a[72:75], v142, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2758) 
s_barrier  // (op 14295) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[136:141], a[76:79], v142, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2720) 
buffer_load_dwordx4 v[80:83], v10, s[48:51], 0 offen offset:312 // (op 207) Load value

buffer_load_dwordx2 v[84:85], v10, s[48:51], 0 offen offset:328 // (op 207) Load value
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[100:105], v[124:129], a[84:87], v143, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3406) 
v_add_u32 v10, v10, v118 // (op 207) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[100:105], v[106:111], a[80:83], v143, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3444) 
buffer_load_dwordx4 v[86:89], v10, s[48:51], 0 offen offset:288 // (op 207) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[100:105], v[130:135], a[88:91], v143, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3368) 
buffer_load_dwordx2 v[90:91], v10, s[48:51], 0 offen offset:304 // (op 207) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[100:105], v[136:141], a[92:95], v143, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3330) 
buffer_load_dwordx4 v[92:95], v10, s[48:51], 0 offen offset:312 // (op 207) Load value

buffer_load_dwordx2 v[96:97], v10, s[48:51], 0 offen offset:328 // (op 207) Load value



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[160:165], v[124:129], a[100:103], v144, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4016) 
s_barrier  // (op 14168) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[160:165], v[106:111], a[96:99], v144, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4054) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[160:165], v[130:135], a[104:107], v144, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3978) 
v_add_u32 v11, 192, v11 // (op 13557) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[160:165], v[136:141], a[108:111], v144, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3940) 
s_waitcnt vmcnt(26)
ds_write_b128 v14, v[18:21] // (op 13) Store local data 

s_waitcnt vmcnt(25)
ds_write_b64 v14, v[22:23] offset:16 // (op 13) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:117], v[124:129], a[116:119], v145, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4626) 
s_waitcnt vmcnt(24)
ds_write_b128 v14, v[24:27] offset:24 // (op 13) Store local data 

s_waitcnt vmcnt(23)
ds_write_b64 v14, v[28:29] offset:40 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:117], v[106:111], a[112:115], v145, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4664) 

s_barrier  // (op 14155) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[112:117], v[130:135], a[120:123], v145, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4588) 
s_waitcnt vmcnt(22)
ds_write_b128 v12, v[34:37] offset:12288 // (op 6) Store local data 

s_waitcnt vmcnt(21)
ds_write_b64 v12, v[38:39] offset:12304 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[112:117], v[136:141], a[124:127], v145, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4550) 
s_waitcnt vmcnt(20)
ds_write_b128 v12, v[40:43] offset:12312 // (op 6) Store local data 




s_waitcnt vmcnt(19)
ds_write_b64 v12, v[44:45] offset:12328 // (op 6) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v12, v[46:49] offset:12384 // (op 6) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v12, v[50:51] offset:12400 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v12, v[52:55] offset:12408 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v12, v[56:57] offset:12424 // (op 6) Store local data 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 4434) 
s_mov_b32 s18, s0 // (op 4434) convert

s_and_b32 s19, s18, 4294967294 // (op 4434) 

s_cmp_lt_u32 s47, s19 // (op 4434) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoop_4434 // (op 4434) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoop_4434 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoop_4434:













s_mul_hi_i32 s47, s16, s10 // (op 13489) 
v_and_b32 v10, 4294967294, v1 // (op 13489) 
s_add_i32 s0, s47, s16 // (op 13489) 

v_and_b32 v11, -2, v1 // (op 13489) 
s_ashr_i32 s1, s0, 31 // (op 13489) 
v_sub_u32 v12, v1, v11 // (op 13489) 

s_cmp_eq_i32 s10, 0 // (op 13489) 
s_mov_b32 s47, scc // (op 13489) 
v_lshlrev_b32 v11, 6, v12 // (op 13489) 

s_cmp_eq_i32 s47, 0 // (op 13489) 
s_cselect_b32 s18, 0, -1 // (op 13489) 

s_and_b32 s19, s13, 31 // (op 13489) 
s_lshl_b32 s47, 1, s19 // (op 13489) 
s_add_i32 s22, s47, s18 // (op 13489) 


s_and_b32 s18, s1, s22 // (op 13489) 


s_add_i32 s1, s0, s18 // (op 13489) 


s_ashr_i32 s0, s1, s19 // (op 13489) 


s_lshl_b32 s1, s13, 24 // (op 13489) 
s_ashr_i32 s47, s1, 31 // (op 13489) 

s_xor_b32 s1, s0, s47 // (op 13489) 

s_sub_i32 s0, s1, s47 // (op 13489) 


s_mul_hi_i32 s1, s0, s14 // (op 13489) 
s_add_i32 s47, s1, s0 // (op 13489) 

s_ashr_i32 s1, s47, 31 // (op 13489) 
s_cmp_eq_i32 s14, 0 // (op 13489) 
s_mov_b32 s18, scc // (op 13489) 
s_cmp_eq_i32 s18, 0 // (op 13489) 
s_cselect_b32 s19, 0, -1 // (op 13489) 

s_and_b32 s18, s15, 31 // (op 13489) 
s_lshl_b32 s22, 1, s18 // (op 13489) 
s_add_i32 s23, s22, s19 // (op 13489) 


s_and_b32 s19, s1, s23 // (op 13489) 


s_add_i32 s1, s47, s19 // (op 13489) 


s_ashr_i32 s19, s1, s18 // (op 13489) 


s_lshl_b32 s1, s15, 24 // (op 13489) 
s_ashr_i32 s18, s1, 31 // (op 13489) 

s_xor_b32 s1, s19, s18 // (op 13489) 

s_sub_i32 s19, s1, s18 // (op 13489) 


s_mul_i32 s1, s0, s11 // (op 13489) 

s_sub_u32 s0, s16, s1 // (op 13489) 

s_mul_i32 s1, s19, s11 // (op 13489) 
s_add_u32 s1, s1, s0 // (op 13489) 


s_mul_hi_i32 s0, s16, s9 // (op 13489) 
s_add_i32 s47, s0, s16 // (op 13489) 

s_ashr_i32 s0, s47, 31 // (op 13489) 
s_cmp_eq_i32 s9, 0 // (op 13489) 
s_mov_b32 s18, scc // (op 13489) 
s_cmp_eq_i32 s18, 0 // (op 13489) 
s_cselect_b32 s19, 0, -1 // (op 13489) 

s_and_b32 s18, s45, 31 // (op 13489) 
s_lshl_b32 s22, 1, s18 // (op 13489) 
s_add_i32 s23, s22, s19 // (op 13489) 


s_and_b32 s19, s0, s23 // (op 13489) 


s_add_i32 s0, s47, s19 // (op 13489) 


s_ashr_i32 s19, s0, s18 // (op 13489) 


s_lshl_b32 s0, s45, 24 // (op 13489) 
s_ashr_i32 s18, s0, 31 // (op 13489) 

s_xor_b32 s0, s19, s18 // (op 13489) 

s_sub_i32 s19, s0, s18 // (op 13489) 


s_add_u32 s22, s6, 128 // (op 13489) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13489) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13489) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13489) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13489) 

s_mul_i32 s0, s11, s22 // (op 13489) 

s_mul_i32 s18, s8, s0 // (op 13489) 

s_mul_i32 s0, s19, s18 // (op 13489) 

s_sub_u32 s18, s16, s0 // (op 13489) 

s_mov_b32 s0, s18 // (op 13489) 
s_mul_hi_i32 s47, s0, s12 // (op 13489) 
s_add_i32 s22, s47, s0 // (op 13489) 


s_ashr_i32 s0, s22, 31 // (op 13489) 
s_cmp_eq_i32 s12, 0 // (op 13489) 
s_mov_b32 s47, scc // (op 13489) 
s_cmp_eq_i32 s47, 0 // (op 13489) 
s_cselect_b32 s23, 0, -1 // (op 13489) 

s_and_b32 s47, s44, 31 // (op 13489) 
s_lshl_b32 s24, 1, s47 // (op 13489) 
s_add_i32 s25, s24, s23 // (op 13489) 


s_and_b32 s23, s0, s25 // (op 13489) 


s_add_i32 s0, s22, s23 // (op 13489) 


s_ashr_i32 s22, s0, s47 // (op 13489) 


s_lshl_b32 s0, s44, 24 // (op 13489) 
s_ashr_i32 s47, s0, 31 // (op 13489) 

s_xor_b32 s0, s22, s47 // (op 13489) 

s_sub_i32 s22, s0, s47 // (op 13489) 


s_mul_i32 s0, s46, s11 // (op 13489) 
s_add_u32 s24, s20, 256 // (op 13489) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13489) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13489) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13489) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 13489) 

s_sub_i32 s23, s24, s0 // (op 13489) 

s_mul_i32 s47, s22, s23 // (op 13489) 


s_sub_u32 s22, s18, s47 // (op 13489) 


s_add_u32 s18, s22, s0 // (op 13489) 


s_cmp_eq_i32 s19, 0 // (op 13489) 
s_mov_b32 s0, scc // (op 13489) 

s_cmp_eq_i32 s0, 0 // (op 13489) 
s_cselect_b32 s19, s18, s1 // (op 13489) 



v_lshl_add_u32 v12, s19, 8, v10 // (op 13489) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13489) 
s_lshr_b32 s47, s0, 1 // (op 13489) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 13489) 


v_mov_b32 v73, 0 // (op 13489) convert
v_mov_b32 v72, v10 // (op 13489) convert

v_mul_lo_u32 v11, v12, s27 // (op 13489) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 13489) most significant: high of low * low
v_mul_lo_u32 v10, v12, s26 // (op 13489) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 13489) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v72 // (op 13489) least significant half
v_addc_co_u32 v11, s[0:1], v11, v73, s[0:1] // (op 13489) most significant half



v_mul_lo_u32 v73, v11, 6 // (op 13489) most significant: low of high * low
v_mul_hi_u32 v12, v10, 6 // (op 13489) most significant: high of low * low
v_mul_lo_u32 v72, v10, 6 // (op 13489) least significant: low of low * low
v_add_u32 v73, v73, v12 // (op 13489) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[72:73] // (op 13489) 

v_mov_b32 v12, v10 // (op 13489) convert

s_mul_i32 s1, s21, 6 // (op 13491) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 13491) least significant
s_mul_hi_u32 s47, s20, 6 // (op 13491) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13491) most significant: sum

s_mul_i32 s19, s26, s1 // (op 13491) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 13491) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 13491) least significant
s_mul_hi_u32 s22, s26, s0 // (op 13491) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13491) most significant: sum
s_add_u32 s19, s19, s22 // (op 13491) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13491) 

s_mov_b32 s48, s28 // (op 13491) reg expression
s_mov_b32 s49, s29 // (op 13491) reg expression
s_mov_b32 s50, s0 // (op 13491) reg expression
s_mov_b32 s51, 131072 // (op 13491) call()

s_mul_hi_i32 s47, s16, s10 // (op 13587) 
v_lshrrev_b32 v10, 1, v1 // (op 13587) 
s_add_i32 s0, s47, s16 // (op 13587) 

v_and_b32 v11, -2, v1 // (op 13587) 
s_ashr_i32 s1, s0, 31 // (op 13587) 
v_sub_u32 v13, v1, v11 // (op 13587) 

s_cmp_eq_i32 s10, 0 // (op 13587) 
s_mov_b32 s47, scc // (op 13587) 
v_lshlrev_b32 v11, 6, v13 // (op 13587) 

s_cmp_eq_i32 s47, 0 // (op 13587) 
s_cselect_b32 s18, 0, -1 // (op 13587) 

s_and_b32 s19, s13, 31 // (op 13587) 
s_lshl_b32 s47, 1, s19 // (op 13587) 
s_add_i32 s22, s47, s18 // (op 13587) 


s_and_b32 s18, s1, s22 // (op 13587) 


s_add_i32 s1, s0, s18 // (op 13587) 


s_ashr_i32 s0, s1, s19 // (op 13587) 


s_lshl_b32 s1, s13, 24 // (op 13587) 
s_ashr_i32 s47, s1, 31 // (op 13587) 

s_xor_b32 s1, s0, s47 // (op 13587) 

s_sub_i32 s0, s1, s47 // (op 13587) 


s_mul_hi_i32 s1, s0, s14 // (op 13587) 
s_add_i32 s47, s1, s0 // (op 13587) 

s_ashr_i32 s1, s47, 31 // (op 13587) 
s_cmp_eq_i32 s14, 0 // (op 13587) 
s_mov_b32 s18, scc // (op 13587) 
s_cmp_eq_i32 s18, 0 // (op 13587) 
s_cselect_b32 s19, 0, -1 // (op 13587) 

s_and_b32 s18, s15, 31 // (op 13587) 
s_lshl_b32 s22, 1, s18 // (op 13587) 
s_add_i32 s23, s22, s19 // (op 13587) 


s_and_b32 s19, s1, s23 // (op 13587) 


s_add_i32 s1, s47, s19 // (op 13587) 


s_ashr_i32 s19, s1, s18 // (op 13587) 


s_lshl_b32 s1, s15, 24 // (op 13587) 
s_ashr_i32 s18, s1, 31 // (op 13587) 

s_xor_b32 s1, s19, s18 // (op 13587) 

s_sub_i32 s19, s1, s18 // (op 13587) 


s_add_u32 s22, s6, 128 // (op 13587) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13587) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13587) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13587) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13587) 

s_mul_i32 s1, s19, s22 // (op 13587) 

s_sub_i32 s47, s0, s1 // (op 13587) 


s_mul_hi_i32 s0, s16, s9 // (op 13587) 
s_add_i32 s1, s0, s16 // (op 13587) 

s_ashr_i32 s0, s1, 31 // (op 13587) 
s_cmp_eq_i32 s9, 0 // (op 13587) 
s_mov_b32 s18, scc // (op 13587) 
s_cmp_eq_i32 s18, 0 // (op 13587) 
s_cselect_b32 s19, 0, -1 // (op 13587) 

s_and_b32 s18, s45, 31 // (op 13587) 
s_lshl_b32 s24, 1, s18 // (op 13587) 
s_add_i32 s25, s24, s19 // (op 13587) 


s_and_b32 s19, s0, s25 // (op 13587) 


s_add_i32 s0, s1, s19 // (op 13587) 


s_ashr_i32 s1, s0, s18 // (op 13587) 


s_lshl_b32 s0, s45, 24 // (op 13587) 
s_ashr_i32 s18, s0, 31 // (op 13587) 

s_xor_b32 s0, s1, s18 // (op 13587) 

s_sub_i32 s1, s0, s18 // (op 13587) 


s_mul_i32 s0, s11, s22 // (op 13587) 

s_mul_i32 s18, s8, s0 // (op 13587) 

s_mul_i32 s0, s1, s18 // (op 13587) 

s_sub_u32 s18, s16, s0 // (op 13587) 

s_mul_hi_i32 s0, s18, s12 // (op 13587) 
s_add_i32 s19, s0, s18 // (op 13587) 


s_ashr_i32 s0, s19, 31 // (op 13587) 
s_cmp_eq_i32 s12, 0 // (op 13587) 
s_mov_b32 s18, scc // (op 13587) 
s_cmp_eq_i32 s18, 0 // (op 13587) 
s_cselect_b32 s22, 0, -1 // (op 13587) 

s_and_b32 s18, s44, 31 // (op 13587) 
s_lshl_b32 s23, 1, s18 // (op 13587) 
s_add_i32 s24, s23, s22 // (op 13587) 


s_and_b32 s25, s0, s24 // (op 13587) 


s_add_i32 s0, s19, s25 // (op 13587) 


s_ashr_i32 s19, s0, s18 // (op 13587) 


s_lshl_b32 s0, s44, 24 // (op 13587) 
s_ashr_i32 s18, s0, 31 // (op 13587) 

s_xor_b32 s0, s19, s18 // (op 13587) 

s_sub_i32 s19, s0, s18 // (op 13587) 


s_cmp_eq_i32 s1, 0 // (op 13587) 
s_mov_b32 s0, scc // (op 13587) 

s_cmp_eq_i32 s0, 0 // (op 13587) 
s_cselect_b32 s1, s19, s47 // (op 13587) 



v_lshl_add_u32 v13, s1, 7, v10 // (op 13587) 


v_mul_lo_u32 v73, v13, s31 // (op 13587) most significant: low of low * high
v_mul_hi_u32 v10, v13, s30 // (op 13587) most significant: high of low * low
v_mul_lo_u32 v72, v13, s30 // (op 13587) least significant: low of low * low
v_add_u32 v73, v73, v10 // (op 13587) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13587) 
s_lshr_b32 s47, s0, 1 // (op 13587) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 13587) 


v_mov_b32 v29, 0 // (op 13587) convert
v_mov_b32 v28, v10 // (op 13587) convert

v_add_co_u32 v10, s[0:1], v28, v72 // (op 13587) least significant half
v_addc_co_u32 v11, s[0:1], v29, v73, s[0:1] // (op 13587) most significant half



v_mul_lo_u32 v73, v11, 6 // (op 13587) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 13587) most significant: high of low * low
v_mul_lo_u32 v72, v10, 6 // (op 13587) least significant: low of low * low
v_add_u32 v73, v73, v13 // (op 13587) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[72:73] // (op 13587) 

v_mov_b32 v13, v10 // (op 13587) convert

s_mul_i32 s1, s7, 6 // (op 13589) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 13589) least significant
s_mul_hi_u32 s47, s6, 6 // (op 13589) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13589) most significant: sum

s_mul_i32 s19, s30, s1 // (op 13589) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 13589) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 13589) least significant
s_mul_hi_u32 s22, s30, s0 // (op 13589) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13589) most significant: sum
s_add_u32 s19, s19, s22 // (op 13589) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13589) 

s_mov_b32 s60, s32 // (op 13589) reg expression
s_mov_b32 s61, s33 // (op 13589) reg expression
s_mov_b32 s62, s0 // (op 13589) reg expression
s_mov_b32 s63, 131072 // (op 13589) call()

v_and_b32 v10, -2, v1 // (op 13685) 
v_sub_u32 v11, v1, v10 // (op 13685) 

v_lshlrev_b32 v10, 6, v11 // (op 13685) 

v_lshrrev_b32 v11, 1, v1 // (op 13685) 
v_lshl_add_u32 v14, v11, 8, v10 // (op 13685) 


v_mul_lo_u32 v10, v14, 6 // (op 13685) 

v_lshrrev_b32 v11, 3, v10 // (op 13685) 

v_and_b32 v10, -2, v1 // (op 13743) 
v_sub_u32 v14, v1, v10 // (op 13743) 

v_lshlrev_b32 v10, 6, v14 // (op 13743) 

v_lshrrev_b32 v14, 1, v1 // (op 13743) 
v_lshl_add_u32 v15, v14, 7, v10 // (op 13743) 


v_mul_lo_u32 v10, v15, 6 // (op 13743) 

v_lshrrev_b32 v14, 3, v10 // (op 13743) 

s_mul_hi_i32 s47, s16, s10 // (op 13979) 
v_and_b32 v10, 63, v1 // (op 13979) 
s_add_i32 s0, s47, s16 // (op 13979) 

v_lshrrev_b32 v15, 4, v10 // (op 13979) 
s_ashr_i32 s1, s0, 31 // (op 13979) 
v_and_b32 v16, 3, v15 // (op 13979) 

s_cmp_eq_i32 s10, 0 // (op 13979) 
s_mov_b32 s47, scc // (op 13979) 
v_and_b32 v15, 15, v1 // (op 13979) 
s_cmp_eq_i32 s47, 0 // (op 13979) 
s_cselect_b32 s18, 0, -1 // (op 13979) 

v_lshl_add_u32 v17, v16, 4, v15 // (op 13979) 


s_and_b32 s19, s13, 31 // (op 13979) 
v_and_b32 v15, 4294967168, v1 // (op 13979) 
s_lshl_b32 s47, 1, s19 // (op 13979) 
v_add_u32 v16, v15, v17 // (op 13979) 


s_add_i32 s22, s47, s18 // (op 13979) 


v_lshrrev_b32 v15, 6, v10 // (op 13979) 

s_and_b32 s18, s1, s22 // (op 13979) 


v_lshlrev_b32 v10, 2, v15 // (op 13979) 

s_add_i32 s1, s0, s18 // (op 13979) 


s_ashr_i32 s0, s1, s19 // (op 13979) 


s_lshl_b32 s1, s13, 24 // (op 13979) 
s_ashr_i32 s47, s1, 31 // (op 13979) 

s_xor_b32 s1, s0, s47 // (op 13979) 

s_sub_i32 s0, s1, s47 // (op 13979) 


s_mul_hi_i32 s1, s0, s14 // (op 13979) 
s_add_i32 s47, s1, s0 // (op 13979) 

s_ashr_i32 s1, s47, 31 // (op 13979) 
s_cmp_eq_i32 s14, 0 // (op 13979) 
s_mov_b32 s18, scc // (op 13979) 
s_cmp_eq_i32 s18, 0 // (op 13979) 
s_cselect_b32 s19, 0, -1 // (op 13979) 

s_and_b32 s18, s15, 31 // (op 13979) 
s_lshl_b32 s22, 1, s18 // (op 13979) 
s_add_i32 s23, s22, s19 // (op 13979) 


s_and_b32 s19, s1, s23 // (op 13979) 


s_add_i32 s1, s47, s19 // (op 13979) 


s_ashr_i32 s19, s1, s18 // (op 13979) 


s_lshl_b32 s1, s15, 24 // (op 13979) 
s_ashr_i32 s18, s1, 31 // (op 13979) 

s_xor_b32 s1, s19, s18 // (op 13979) 

s_sub_i32 s19, s1, s18 // (op 13979) 


s_mul_i32 s1, s0, s11 // (op 13979) 

s_sub_u32 s0, s16, s1 // (op 13979) 

s_mul_i32 s1, s19, s11 // (op 13979) 
s_add_u32 s1, s1, s0 // (op 13979) 


s_mul_hi_i32 s0, s16, s9 // (op 13979) 
s_add_i32 s47, s0, s16 // (op 13979) 

s_ashr_i32 s0, s47, 31 // (op 13979) 
s_cmp_eq_i32 s9, 0 // (op 13979) 
s_mov_b32 s18, scc // (op 13979) 
s_cmp_eq_i32 s18, 0 // (op 13979) 
s_cselect_b32 s19, 0, -1 // (op 13979) 

s_and_b32 s18, s45, 31 // (op 13979) 
s_lshl_b32 s22, 1, s18 // (op 13979) 
s_add_i32 s23, s22, s19 // (op 13979) 


s_and_b32 s19, s0, s23 // (op 13979) 


s_add_i32 s0, s47, s19 // (op 13979) 


s_ashr_i32 s19, s0, s18 // (op 13979) 


s_lshl_b32 s0, s45, 24 // (op 13979) 
s_ashr_i32 s18, s0, 31 // (op 13979) 

s_xor_b32 s0, s19, s18 // (op 13979) 

s_sub_i32 s19, s0, s18 // (op 13979) 


s_add_u32 s22, s6, 128 // (op 13979) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13979) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13979) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13979) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13979) 

s_mul_i32 s0, s11, s22 // (op 13979) 

s_mul_i32 s18, s8, s0 // (op 13979) 

s_mul_i32 s0, s19, s18 // (op 13979) 

s_sub_u32 s18, s16, s0 // (op 13979) 

s_mov_b32 s0, s18 // (op 13979) 
s_mul_hi_i32 s47, s0, s12 // (op 13979) 
s_add_i32 s22, s47, s0 // (op 13979) 


s_ashr_i32 s0, s22, 31 // (op 13979) 
s_cmp_eq_i32 s12, 0 // (op 13979) 
s_mov_b32 s47, scc // (op 13979) 
s_cmp_eq_i32 s47, 0 // (op 13979) 
s_cselect_b32 s23, 0, -1 // (op 13979) 

s_and_b32 s47, s44, 31 // (op 13979) 
s_lshl_b32 s24, 1, s47 // (op 13979) 
s_add_i32 s25, s24, s23 // (op 13979) 


s_and_b32 s23, s0, s25 // (op 13979) 


s_add_i32 s0, s22, s23 // (op 13979) 


s_ashr_i32 s22, s0, s47 // (op 13979) 


s_lshl_b32 s0, s44, 24 // (op 13979) 
s_ashr_i32 s47, s0, 31 // (op 13979) 

s_xor_b32 s0, s22, s47 // (op 13979) 

s_sub_i32 s22, s0, s47 // (op 13979) 


s_mul_i32 s0, s46, s11 // (op 13979) 
s_add_u32 s24, s20, 256 // (op 13979) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13979) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13979) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13979) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 13979) 

s_sub_i32 s23, s24, s0 // (op 13979) 

s_mul_i32 s47, s22, s23 // (op 13979) 


s_sub_u32 s22, s18, s47 // (op 13979) 


s_add_u32 s18, s22, s0 // (op 13979) 


s_cmp_eq_i32 s19, 0 // (op 13979) 
s_mov_b32 s0, scc // (op 13979) 

s_cmp_eq_i32 s0, 0 // (op 13979) 
s_cselect_b32 s19, s18, s1 // (op 13979) 



v_lshl_add_u32 v15, s19, 8, v16 // (op 13979) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13979) 
s_lshr_b32 s47, s0, 1 // (op 13979) 

v_lshl_add_u32 v16, s47, 3, v10 // (op 13979) 


v_mul_lo_u32 v10, v15, s34 // (op 13979) 
v_add_u32 v10, v10, v16 // (op 13979) 


s_mul_i32 s1, s34, s21 // (op 13981) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 13981) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 13981) least significant
s_mul_hi_u32 s18, s34, s20 // (op 13981) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13981) most significant: sum
s_add_u32 s1, s1, s18 // (op 13981) most significant: sum


s_mov_b32 s64, s36 // (op 13981) reg expression
s_mov_b32 s65, s37 // (op 13981) reg expression
s_mov_b32 s66, s0 // (op 13981) reg expression
s_mov_b32 s67, 131072 // (op 13981) call()

s_mul_hi_i32 s47, s16, s10 // (op 14006) 
v_and_b32 v15, 63, v1 // (op 14006) 
s_add_i32 s0, s47, s16 // (op 14006) 

v_lshrrev_b32 v16, 4, v15 // (op 14006) 
s_ashr_i32 s1, s0, 31 // (op 14006) 
v_and_b32 v17, 3, v16 // (op 14006) 

s_cmp_eq_i32 s10, 0 // (op 14006) 
s_mov_b32 s47, scc // (op 14006) 
v_and_b32 v16, 15, v1 // (op 14006) 
s_cmp_eq_i32 s47, 0 // (op 14006) 
s_cselect_b32 s18, 0, -1 // (op 14006) 

v_lshl_add_u32 v18, v17, 4, v16 // (op 14006) 


s_and_b32 s19, s13, 31 // (op 14006) 
v_lshrrev_b32 v16, 6, v1 // (op 14006) 
s_lshl_b32 s47, 1, s19 // (op 14006) 
v_and_b32 v17, 1, v16 // (op 14006) 

s_add_i32 s22, s47, s18 // (op 14006) 


v_lshl_add_u32 v16, v17, 6, v18 // (op 14006) 


s_and_b32 s18, s1, s22 // (op 14006) 


v_lshrrev_b32 v17, 6, v15 // (op 14006) 

s_add_i32 s1, s0, s18 // (op 14006) 


v_lshlrev_b32 v15, 2, v17 // (op 14006) 

s_ashr_i32 s0, s1, s19 // (op 14006) 


s_lshl_b32 s1, s13, 24 // (op 14006) 
s_ashr_i32 s47, s1, 31 // (op 14006) 

s_xor_b32 s1, s0, s47 // (op 14006) 

s_sub_i32 s0, s1, s47 // (op 14006) 


s_mul_hi_i32 s1, s0, s14 // (op 14006) 
s_add_i32 s47, s1, s0 // (op 14006) 

s_ashr_i32 s1, s47, 31 // (op 14006) 
s_cmp_eq_i32 s14, 0 // (op 14006) 
s_mov_b32 s18, scc // (op 14006) 
s_cmp_eq_i32 s18, 0 // (op 14006) 
s_cselect_b32 s19, 0, -1 // (op 14006) 

s_and_b32 s18, s15, 31 // (op 14006) 
s_lshl_b32 s22, 1, s18 // (op 14006) 
s_add_i32 s23, s22, s19 // (op 14006) 


s_and_b32 s19, s1, s23 // (op 14006) 


s_add_i32 s1, s47, s19 // (op 14006) 


s_ashr_i32 s19, s1, s18 // (op 14006) 


s_lshl_b32 s1, s15, 24 // (op 14006) 
s_ashr_i32 s18, s1, 31 // (op 14006) 

s_xor_b32 s1, s19, s18 // (op 14006) 

s_sub_i32 s19, s1, s18 // (op 14006) 


s_add_u32 s22, s6, 128 // (op 14006) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14006) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14006) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14006) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14006) 

s_mul_i32 s1, s19, s22 // (op 14006) 

s_sub_i32 s47, s0, s1 // (op 14006) 


s_mul_hi_i32 s0, s16, s9 // (op 14006) 
s_add_i32 s1, s0, s16 // (op 14006) 

s_ashr_i32 s0, s1, 31 // (op 14006) 
s_cmp_eq_i32 s9, 0 // (op 14006) 
s_mov_b32 s18, scc // (op 14006) 
s_cmp_eq_i32 s18, 0 // (op 14006) 
s_cselect_b32 s19, 0, -1 // (op 14006) 

s_and_b32 s18, s45, 31 // (op 14006) 
s_lshl_b32 s24, 1, s18 // (op 14006) 
s_add_i32 s25, s24, s19 // (op 14006) 


s_and_b32 s19, s0, s25 // (op 14006) 


s_add_i32 s0, s1, s19 // (op 14006) 


s_ashr_i32 s1, s0, s18 // (op 14006) 


s_lshl_b32 s0, s45, 24 // (op 14006) 
s_ashr_i32 s18, s0, 31 // (op 14006) 

s_xor_b32 s0, s1, s18 // (op 14006) 

s_sub_i32 s1, s0, s18 // (op 14006) 


s_mul_i32 s0, s11, s22 // (op 14006) 

s_mul_i32 s18, s8, s0 // (op 14006) 

s_mul_i32 s0, s1, s18 // (op 14006) 

s_sub_u32 s18, s16, s0 // (op 14006) 

s_mul_hi_i32 s0, s18, s12 // (op 14006) 
s_add_i32 s19, s0, s18 // (op 14006) 


s_ashr_i32 s0, s19, 31 // (op 14006) 
s_cmp_eq_i32 s12, 0 // (op 14006) 
s_mov_b32 s18, scc // (op 14006) 
s_cmp_eq_i32 s18, 0 // (op 14006) 
s_cselect_b32 s22, 0, -1 // (op 14006) 

s_and_b32 s18, s44, 31 // (op 14006) 
s_lshl_b32 s23, 1, s18 // (op 14006) 
s_add_i32 s24, s23, s22 // (op 14006) 


s_and_b32 s25, s0, s24 // (op 14006) 


s_add_i32 s0, s19, s25 // (op 14006) 


s_ashr_i32 s19, s0, s18 // (op 14006) 


s_lshl_b32 s0, s44, 24 // (op 14006) 
s_ashr_i32 s18, s0, 31 // (op 14006) 

s_xor_b32 s0, s19, s18 // (op 14006) 

s_sub_i32 s19, s0, s18 // (op 14006) 


s_cmp_eq_i32 s1, 0 // (op 14006) 
s_mov_b32 s0, scc // (op 14006) 

s_cmp_eq_i32 s0, 0 // (op 14006) 
s_cselect_b32 s1, s19, s47 // (op 14006) 



v_lshl_add_u32 v17, s1, 7, v16 // (op 14006) 


v_mul_lo_u32 v16, v17, s38 // (op 14006) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 14006) 
s_lshr_b32 s47, s0, 1 // (op 14006) 

v_lshl_add_u32 v17, s47, 3, v15 // (op 14006) 


v_add_u32 v15, v17, v16 // (op 14006) 


s_mul_i32 s1, s38, s7 // (op 14008) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 14008) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 14008) least significant
s_mul_hi_u32 s18, s38, s6 // (op 14008) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14008) most significant: sum
s_add_u32 s1, s1, s18 // (op 14008) most significant: sum


s_mov_b32 s68, s40 // (op 14008) reg expression
s_mov_b32 s69, s41 // (op 14008) reg expression
s_mov_b32 s70, s0 // (op 14008) reg expression
s_mov_b32 s71, 131072 // (op 14008) call()

s_ashr_i64 s[0:1], s[42:43], 7 // (op 4521) 
s_and_b32 s47, s0, 4294967294 // (op 4521) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 4520) 
s_mov_b32 s18, s0 // (op 4520) convert

s_cmp_lt_u32 s47, s18 // (op 4520) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_4520 // (op 4520) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_4520 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoopTail_4520:

s_add_u32 s47, s47, 1 // (op 4522) 
s_mul_i32 s1, s27, 6 // (op 157) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 157) least significant
s_mul_hi_u32 s18, s26, 6 // (op 157) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 157) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 157) 


v_mov_b32 v18, v12 // (op 157) 
v_mov_b32 v19, v18 // (op 157) 
buffer_load_dwordx4 v[20:23], v19, s[48:51], 0 offen // (op 157) Load value
buffer_load_dwordx2 v[24:25], v19, s[48:51], 0 offen offset:16 // (op 157) Load value
buffer_load_dwordx4 v[26:29], v19, s[48:51], 0 offen offset:24 // (op 157) Load value
buffer_load_dwordx2 v[30:31], v19, s[48:51], 0 offen offset:40 // (op 157) Load value
v_add_u32 v19, v19, v16 // (op 157) 
buffer_load_dwordx4 v[32:35], v19, s[48:51], 0 offen // (op 157) Load value
buffer_load_dwordx2 v[36:37], v19, s[48:51], 0 offen offset:16 // (op 157) Load value
buffer_load_dwordx4 v[38:41], v19, s[48:51], 0 offen offset:24 // (op 157) Load value
buffer_load_dwordx2 v[42:43], v19, s[48:51], 0 offen offset:40 // (op 157) Load value



v_add_u32 v12, 96, v12 // (op 13479) 


s_barrier  // (op 14158) 
v_mov_b32 v16, v13 // (op 169) 




buffer_load_dwordx4 v[44:47], v16, s[60:63], 0 offen // (op 169) Load value
s_waitcnt vmcnt(8)
ds_write_b128 v11, v[20:23] offset:12288 // (op 158) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v11, v[24:25] offset:12304 // (op 158) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v11, v[26:29] offset:12312 // (op 158) Store local data 
s_waitcnt vmcnt(5)
ds_write_b64 v11, v[30:31] offset:12328 // (op 158) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v11, v[32:35] offset:12384 // (op 158) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v11, v[36:37] offset:12400 // (op 158) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v11, v[38:41] offset:12408 // (op 158) Store local data 
buffer_load_dwordx2 v[48:49], v16, s[60:63], 0 offen offset:16 // (op 169) Load value
buffer_load_dwordx4 v[50:53], v16, s[60:63], 0 offen offset:24 // (op 169) Load value
buffer_load_dwordx2 v[54:55], v16, s[60:63], 0 offen offset:40 // (op 169) Load value

v_add_u32 v13, 96, v13 // (op 13579) 
s_waitcnt vmcnt(4)
ds_write_b64 v11, v[42:43] offset:12424 // (op 158) Store local data 

s_barrier  // (op 14171) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14161) 






ds_read_b128 v[16:19], v3 offset:12288 // (op 160) Load local data 
ds_read_b64 v[20:21], v3 offset:12304 // (op 160) Load local data 
s_waitcnt vmcnt(3)
ds_write_b128 v14, v[44:47] // (op 168) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v14, v[48:49] offset:16 // (op 168) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v14, v[50:53] offset:24 // (op 168) Store local data 
s_waitcnt vmcnt(0)
ds_write_b64 v14, v[54:55] offset:40 // (op 168) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 14175) 



ds_read_b128 v[22:25], v4 // (op 162) Load local data 
ds_read_b64 v[26:27], v4 offset:16 // (op 162) Load local data 


v_mov_b32 v28, v10 // (op 164) 
v_mov_b32 v29, v28 // (op 164) reg expression

buffer_load_dwordx2 v[30:31], v29, s[64:67], 0 offen // (op 164) Load value



v_mov_b32 v29, v15 // (op 166) 
v_mov_b32 v32, v29 // (op 166) reg expression

buffer_load_dwordx2 v[34:35], v32, s[68:71], 0 offen // (op 166) Load value


v_add_u32 v15, 4, v15 // (op 13996) 
s_waitcnt vmcnt(1)
v_bfe_u32 v36, v30, 0, 8 // (op 12447) 



v_bfe_u32 v37, v30, 8, 8 // (op 12447) 
v_bfe_u32 v38, v30, 16, 8 // (op 12447) 
v_bfe_u32 v39, v30, 24, 8 // (op 12447) 
v_permlane16_swap_b32 v36, v37 // (op 12447) 
ds_read_b128 v[40:43], v4 offset:1536 // (op 291) Load local data 
v_permlane16_swap_b32 v38, v39 // (op 12447) 
s_nop 1
v_permlane32_swap_b32 v36, v38 // (op 12447) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v37, v39 // (op 12447) 
ds_read_b64 v[44:45], v4 offset:1552 // (op 291) Load local data 




ds_read_b128 v[46:49], v4 offset:3072 // (op 427) Load local data 
ds_read_b64 v[50:51], v4 offset:3088 // (op 427) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v30, v34, 0, 8 // (op 12507) 



v_bfe_u32 v31, v34, 8, 8 // (op 12507) 
v_bfe_u32 v32, v34, 16, 8 // (op 12507) 
v_bfe_u32 v33, v34, 24, 8 // (op 12507) 
v_permlane16_swap_b32 v30, v31 // (op 12507) 
s_nop 0
v_permlane16_swap_b32 v32, v33 // (op 12507) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v30, v32 // (op 12507) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v31, v33 // (op 12507) 
ds_read_b128 v[52:55], v4 offset:4608 // (op 563) Load local data 

ds_read_b64 v[56:57], v4 offset:4624 // (op 563) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[22:27], a[0:3], v36, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 167) 
ds_read_b128 v[58:61], v3 offset:13824 // (op 785) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:21], v[40:45], a[4:7], v36, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 294) 
ds_read_b64 v[62:63], v3 offset:13840 // (op 785) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:21], v[46:51], a[8:11], v36, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 430) 
ds_read_b128 v[64:67], v3 offset:15360 // (op 1395) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:21], v[52:57], a[12:15], v36, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 566) 
ds_read_b64 v[68:69], v3 offset:15376 // (op 1395) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:63], v[22:27], a[16:19], v37, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 789) 
ds_read_b128 v[16:19], v3 offset:16896 // (op 2005) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:63], v[46:51], a[24:27], v37, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 843) 
ds_read_b64 v[20:21], v3 offset:16912 // (op 2005) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:63], v[52:57], a[28:31], v37, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 870) 
ds_read_b128 v[70:73], v3 offset:18432 // (op 2615) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:63], v[40:45], a[20:23], v37, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 816) 
ds_read_b64 v[74:75], v3 offset:18448 // (op 2615) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:69], v[22:27], a[32:35], v38, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1399) 
s_lshl_b32 s0, s34, 6 // (op 2624) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:69], v[46:51], a[40:43], v38, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1453) 
v_add_u32 v29, v28, s0 // (op 2624) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:69], v[52:57], a[44:47], v38, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1480) 
buffer_load_dwordx2 v[34:35], v29, s[64:67], 0 offen // (op 2624) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:69], v[40:45], a[36:39], v38, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1426) 




v_add_u32 v10, 4, v10 // (op 13969) 
ds_read_b128 v[58:61], v3 offset:19968 // (op 3225) Load local data 
ds_read_b64 v[62:63], v3 offset:19984 // (op 3225) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[16:21], v[22:27], a[48:51], v39, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2009) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[16:21], v[46:51], a[56:59], v39, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2063) 
s_waitcnt vmcnt(0)
v_bfe_u32 v64, v34, 0, 8 // (op 12461) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[16:21], v[52:57], a[60:63], v39, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2090) 
v_bfe_u32 v65, v34, 8, 8 // (op 12461) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[16:21], v[40:45], a[52:55], v39, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2036) 
v_bfe_u32 v66, v34, 16, 8 // (op 12461) 


v_bfe_u32 v67, v34, 24, 8 // (op 12461) 
v_permlane16_swap_b32 v64, v65 // (op 12461) 
s_nop 0
v_permlane16_swap_b32 v66, v67 // (op 12461) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v64, v66 // (op 12461) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v65, v67 // (op 12461) 
ds_read_b128 v[16:19], v3 offset:21504 // (op 3835) Load local data 

ds_read_b64 v[20:21], v3 offset:21520 // (op 3835) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:75], v[40:45], a[68:71], v64, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2646) 
ds_read_b128 v[34:37], v3 offset:23040 // (op 4445) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[70:75], v[22:27], a[64:67], v64, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2619) 
ds_read_b64 v[38:39], v3 offset:23056 // (op 4445) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[70:75], v[46:51], a[72:75], v64, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2673) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[70:75], v[52:57], a[76:79], v64, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2700) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[58:63], v[40:45], a[84:87], v65, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3256) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[58:63], v[22:27], a[80:83], v65, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3229) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[58:63], v[46:51], a[88:91], v65, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3283) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[58:63], v[52:57], a[92:95], v65, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3310) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[16:21], v[40:45], a[100:103], v66, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3866) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[16:21], v[22:27], a[96:99], v66, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3839) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[16:21], v[46:51], a[104:107], v66, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3893) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[16:21], v[52:57], a[108:111], v66, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3920) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[34:39], v[40:45], a[116:119], v67, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4476) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[34:39], v[22:27], a[112:115], v67, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4449) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[34:39], v[46:51], a[120:123], v67, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4503) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:39], v[52:57], a[124:127], v67, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4530) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 4520) 
s_mov_b32 s18, s0 // (op 4520) convert

s_cmp_lt_u32 s47, s18 // (op 4520) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoopTail_4520 // (op 4520) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoopTail_4520 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_4520:




















s_barrier  // (op 14283) 

















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalFalse_0_beta_114
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


v_accvgpr_read v10, a4 // (op 5646) 
v_accvgpr_read v11, a5 // (op 5646) 
v_accvgpr_read v12, a6 // (op 5646) 
v_accvgpr_read v13, a7 // (op 5646) 
v_mul_f32 v14, v7, v10 // (op 5646) 
v_mul_f32 v15, v7, v11 // (op 5646) 
v_mul_f32 v16, v7, v12 // (op 5646) 
v_mul_f32 v17, v7, v13 // (op 5646) 



s_lshl_b32 s0, s4, 6 // (op 5648) 
v_add_u32 v3, v9, s0 // (op 5648) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5648) Store value


v_accvgpr_read v10, a8 // (op 5658) 
v_accvgpr_read v11, a9 // (op 5658) 
v_accvgpr_read v12, a10 // (op 5658) 
v_accvgpr_read v13, a11 // (op 5658) 
v_mul_f32 v14, v7, v10 // (op 5658) 
v_mul_f32 v15, v7, v11 // (op 5658) 
v_mul_f32 v16, v7, v12 // (op 5658) 
v_mul_f32 v17, v7, v13 // (op 5658) 



s_lshl_b32 s0, s4, 7 // (op 5660) 
v_add_u32 v3, v9, s0 // (op 5660) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5660) Store value


v_accvgpr_read v10, a12 // (op 5670) 
v_accvgpr_read v11, a13 // (op 5670) 
v_accvgpr_read v12, a14 // (op 5670) 
v_accvgpr_read v13, a15 // (op 5670) 
v_mul_f32 v14, v7, v10 // (op 5670) 
v_mul_f32 v15, v7, v11 // (op 5670) 
v_mul_f32 v16, v7, v12 // (op 5670) 
v_mul_f32 v17, v7, v13 // (op 5670) 



s_lshl_b32 s0, s4, 6 // (op 5672) 
s_mul_i32 s1, 3, s0 // (op 5672) 

v_add_u32 v3, v9, s1 // (op 5672) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5672) Store value


v_accvgpr_read v10, a16 // (op 5704) 
v_accvgpr_read v11, a17 // (op 5704) 
v_accvgpr_read v12, a18 // (op 5704) 
v_accvgpr_read v13, a19 // (op 5704) 
v_mul_f32 v14, v7, v10 // (op 5704) 
v_mul_f32 v15, v7, v11 // (op 5704) 
v_mul_f32 v16, v7, v12 // (op 5704) 
v_mul_f32 v17, v7, v13 // (op 5704) 



v_add_u32 v3, 64, v9 // (op 5707) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5707) Store value


v_accvgpr_read v10, a20 // (op 5717) 
v_accvgpr_read v11, a21 // (op 5717) 
v_accvgpr_read v12, a22 // (op 5717) 
v_accvgpr_read v13, a23 // (op 5717) 
v_mul_f32 v14, v7, v10 // (op 5717) 
v_mul_f32 v15, v7, v11 // (op 5717) 
v_mul_f32 v16, v7, v12 // (op 5717) 
v_mul_f32 v17, v7, v13 // (op 5717) 



s_lshl_b32 s0, s4, 6 // (op 5709) 
s_add_u32 s0, s0, 64 // (op 5709) 
v_add_u32 v3, v9, s0 // (op 5709) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5709) Store value


v_accvgpr_read v10, a24 // (op 5722) 
v_accvgpr_read v11, a25 // (op 5722) 
v_accvgpr_read v12, a26 // (op 5722) 
v_accvgpr_read v13, a27 // (op 5722) 
v_mul_f32 v14, v7, v10 // (op 5722) 
v_mul_f32 v15, v7, v11 // (op 5722) 
v_mul_f32 v16, v7, v12 // (op 5722) 
v_mul_f32 v17, v7, v13 // (op 5722) 



s_lshl_b32 s0, s4, 7 // (op 5711) 
s_add_u32 s0, s0, 64 // (op 5711) 
v_add_u32 v3, v9, s0 // (op 5711) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5711) Store value


v_accvgpr_read v10, a28 // (op 5727) 
v_accvgpr_read v11, a29 // (op 5727) 
v_accvgpr_read v12, a30 // (op 5727) 
v_accvgpr_read v13, a31 // (op 5727) 
v_mul_f32 v14, v7, v10 // (op 5727) 
v_mul_f32 v15, v7, v11 // (op 5727) 
v_mul_f32 v16, v7, v12 // (op 5727) 
v_mul_f32 v17, v7, v13 // (op 5727) 



s_lshl_b32 s0, s4, 6 // (op 5713) 
s_mul_i32 s1, 3, s0 // (op 5713) 
s_add_u32 s1, s1, 64 // (op 5713) 

v_add_u32 v3, v9, s1 // (op 5713) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5713) Store value


v_accvgpr_read v10, a32 // (op 5772) 
v_accvgpr_read v11, a33 // (op 5772) 
v_accvgpr_read v12, a34 // (op 5772) 
v_accvgpr_read v13, a35 // (op 5772) 
v_mul_f32 v14, v7, v10 // (op 5772) 
v_mul_f32 v15, v7, v11 // (op 5772) 
v_mul_f32 v16, v7, v12 // (op 5772) 
v_mul_f32 v17, v7, v13 // (op 5772) 



v_add_u32 v3, 128, v9 // (op 5775) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5775) Store value


v_accvgpr_read v10, a36 // (op 5785) 
v_accvgpr_read v11, a37 // (op 5785) 
v_accvgpr_read v12, a38 // (op 5785) 
v_accvgpr_read v13, a39 // (op 5785) 
v_mul_f32 v14, v7, v10 // (op 5785) 
v_mul_f32 v15, v7, v11 // (op 5785) 
v_mul_f32 v16, v7, v12 // (op 5785) 
v_mul_f32 v17, v7, v13 // (op 5785) 



s_lshl_b32 s0, s4, 6 // (op 5777) 
s_add_u32 s0, s0, 128 // (op 5777) 
v_add_u32 v3, v9, s0 // (op 5777) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5777) Store value


v_accvgpr_read v10, a40 // (op 5790) 
v_accvgpr_read v11, a41 // (op 5790) 
v_accvgpr_read v12, a42 // (op 5790) 
v_accvgpr_read v13, a43 // (op 5790) 
v_mul_f32 v14, v7, v10 // (op 5790) 
v_mul_f32 v15, v7, v11 // (op 5790) 
v_mul_f32 v16, v7, v12 // (op 5790) 
v_mul_f32 v17, v7, v13 // (op 5790) 



s_lshl_b32 s0, s4, 7 // (op 5779) 
s_add_u32 s0, s0, 128 // (op 5779) 
v_add_u32 v3, v9, s0 // (op 5779) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5779) Store value


v_accvgpr_read v10, a44 // (op 5795) 
v_accvgpr_read v11, a45 // (op 5795) 
v_accvgpr_read v12, a46 // (op 5795) 
v_accvgpr_read v13, a47 // (op 5795) 
v_mul_f32 v14, v7, v10 // (op 5795) 
v_mul_f32 v15, v7, v11 // (op 5795) 
v_mul_f32 v16, v7, v12 // (op 5795) 
v_mul_f32 v17, v7, v13 // (op 5795) 



s_lshl_b32 s0, s4, 6 // (op 5781) 
s_mul_i32 s1, 3, s0 // (op 5781) 
s_add_u32 s1, s1, 128 // (op 5781) 

v_add_u32 v3, v9, s1 // (op 5781) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5781) Store value


v_accvgpr_read v10, a48 // (op 5840) 
v_accvgpr_read v11, a49 // (op 5840) 
v_accvgpr_read v12, a50 // (op 5840) 
v_accvgpr_read v13, a51 // (op 5840) 
v_mul_f32 v14, v7, v10 // (op 5840) 
v_mul_f32 v15, v7, v11 // (op 5840) 
v_mul_f32 v16, v7, v12 // (op 5840) 
v_mul_f32 v17, v7, v13 // (op 5840) 



v_add_u32 v3, 192, v9 // (op 5843) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5843) Store value


v_accvgpr_read v10, a52 // (op 5853) 
v_accvgpr_read v11, a53 // (op 5853) 
v_accvgpr_read v12, a54 // (op 5853) 
v_accvgpr_read v13, a55 // (op 5853) 
v_mul_f32 v14, v7, v10 // (op 5853) 
v_mul_f32 v15, v7, v11 // (op 5853) 
v_mul_f32 v16, v7, v12 // (op 5853) 
v_mul_f32 v17, v7, v13 // (op 5853) 



s_lshl_b32 s0, s4, 6 // (op 5845) 
s_add_u32 s0, s0, 192 // (op 5845) 
v_add_u32 v3, v9, s0 // (op 5845) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5845) Store value


v_accvgpr_read v10, a56 // (op 5858) 
v_accvgpr_read v11, a57 // (op 5858) 
v_accvgpr_read v12, a58 // (op 5858) 
v_accvgpr_read v13, a59 // (op 5858) 
v_mul_f32 v14, v7, v10 // (op 5858) 
v_mul_f32 v15, v7, v11 // (op 5858) 
v_mul_f32 v16, v7, v12 // (op 5858) 
v_mul_f32 v17, v7, v13 // (op 5858) 



s_lshl_b32 s0, s4, 7 // (op 5847) 
s_add_u32 s0, s0, 192 // (op 5847) 
v_add_u32 v3, v9, s0 // (op 5847) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5847) Store value


v_accvgpr_read v10, a60 // (op 5863) 
v_accvgpr_read v11, a61 // (op 5863) 
v_accvgpr_read v12, a62 // (op 5863) 
v_accvgpr_read v13, a63 // (op 5863) 
v_mul_f32 v14, v7, v10 // (op 5863) 
v_mul_f32 v15, v7, v11 // (op 5863) 
v_mul_f32 v16, v7, v12 // (op 5863) 
v_mul_f32 v17, v7, v13 // (op 5863) 



s_lshl_b32 s0, s4, 6 // (op 5849) 
s_mul_i32 s1, 3, s0 // (op 5849) 
s_add_u32 s1, s1, 192 // (op 5849) 

v_add_u32 v3, v9, s1 // (op 5849) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5849) Store value


v_accvgpr_read v10, a64 // (op 5908) 
v_accvgpr_read v11, a65 // (op 5908) 
v_accvgpr_read v12, a66 // (op 5908) 
v_accvgpr_read v13, a67 // (op 5908) 
v_mul_f32 v14, v7, v10 // (op 5908) 
v_mul_f32 v15, v7, v11 // (op 5908) 
v_mul_f32 v16, v7, v12 // (op 5908) 
v_mul_f32 v17, v7, v13 // (op 5908) 



v_add_u32 v3, 256, v9 // (op 5911) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5911) Store value


v_accvgpr_read v10, a68 // (op 5921) 
v_accvgpr_read v11, a69 // (op 5921) 
v_accvgpr_read v12, a70 // (op 5921) 
v_accvgpr_read v13, a71 // (op 5921) 
v_mul_f32 v14, v7, v10 // (op 5921) 
v_mul_f32 v15, v7, v11 // (op 5921) 
v_mul_f32 v16, v7, v12 // (op 5921) 
v_mul_f32 v17, v7, v13 // (op 5921) 



s_lshl_b32 s0, s4, 6 // (op 5913) 
s_add_u32 s0, s0, 256 // (op 5913) 
v_add_u32 v3, v9, s0 // (op 5913) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5913) Store value


v_accvgpr_read v10, a72 // (op 5926) 
v_accvgpr_read v11, a73 // (op 5926) 
v_accvgpr_read v12, a74 // (op 5926) 
v_accvgpr_read v13, a75 // (op 5926) 
v_mul_f32 v14, v7, v10 // (op 5926) 
v_mul_f32 v15, v7, v11 // (op 5926) 
v_mul_f32 v16, v7, v12 // (op 5926) 
v_mul_f32 v17, v7, v13 // (op 5926) 



s_lshl_b32 s0, s4, 7 // (op 5915) 
s_add_u32 s0, s0, 256 // (op 5915) 
v_add_u32 v3, v9, s0 // (op 5915) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5915) Store value


v_accvgpr_read v10, a76 // (op 5931) 
v_accvgpr_read v11, a77 // (op 5931) 
v_accvgpr_read v12, a78 // (op 5931) 
v_accvgpr_read v13, a79 // (op 5931) 
v_mul_f32 v14, v7, v10 // (op 5931) 
v_mul_f32 v15, v7, v11 // (op 5931) 
v_mul_f32 v16, v7, v12 // (op 5931) 
v_mul_f32 v17, v7, v13 // (op 5931) 



s_lshl_b32 s0, s4, 6 // (op 5917) 
s_mul_i32 s1, 3, s0 // (op 5917) 
s_add_u32 s1, s1, 256 // (op 5917) 

v_add_u32 v3, v9, s1 // (op 5917) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5917) Store value


v_accvgpr_read v10, a80 // (op 5976) 
v_accvgpr_read v11, a81 // (op 5976) 
v_accvgpr_read v12, a82 // (op 5976) 
v_accvgpr_read v13, a83 // (op 5976) 
v_mul_f32 v14, v7, v10 // (op 5976) 
v_mul_f32 v15, v7, v11 // (op 5976) 
v_mul_f32 v16, v7, v12 // (op 5976) 
v_mul_f32 v17, v7, v13 // (op 5976) 



v_add_u32 v3, 320, v9 // (op 5979) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5979) Store value


v_accvgpr_read v10, a84 // (op 5989) 
v_accvgpr_read v11, a85 // (op 5989) 
v_accvgpr_read v12, a86 // (op 5989) 
v_accvgpr_read v13, a87 // (op 5989) 
v_mul_f32 v14, v7, v10 // (op 5989) 
v_mul_f32 v15, v7, v11 // (op 5989) 
v_mul_f32 v16, v7, v12 // (op 5989) 
v_mul_f32 v17, v7, v13 // (op 5989) 



s_lshl_b32 s0, s4, 6 // (op 5981) 
s_add_u32 s0, s0, 320 // (op 5981) 
v_add_u32 v3, v9, s0 // (op 5981) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5981) Store value


v_accvgpr_read v10, a88 // (op 5994) 
v_accvgpr_read v11, a89 // (op 5994) 
v_accvgpr_read v12, a90 // (op 5994) 
v_accvgpr_read v13, a91 // (op 5994) 
v_mul_f32 v14, v7, v10 // (op 5994) 
v_mul_f32 v15, v7, v11 // (op 5994) 
v_mul_f32 v16, v7, v12 // (op 5994) 
v_mul_f32 v17, v7, v13 // (op 5994) 



s_lshl_b32 s0, s4, 7 // (op 5983) 
s_add_u32 s0, s0, 320 // (op 5983) 
v_add_u32 v3, v9, s0 // (op 5983) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5983) Store value


v_accvgpr_read v10, a92 // (op 5999) 
v_accvgpr_read v11, a93 // (op 5999) 
v_accvgpr_read v12, a94 // (op 5999) 
v_accvgpr_read v13, a95 // (op 5999) 
v_mul_f32 v14, v7, v10 // (op 5999) 
v_mul_f32 v15, v7, v11 // (op 5999) 
v_mul_f32 v16, v7, v12 // (op 5999) 
v_mul_f32 v17, v7, v13 // (op 5999) 



s_lshl_b32 s0, s4, 6 // (op 5985) 
s_mul_i32 s1, 3, s0 // (op 5985) 
s_add_u32 s1, s1, 320 // (op 5985) 

v_add_u32 v3, v9, s1 // (op 5985) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 5985) Store value


v_accvgpr_read v10, a96 // (op 6044) 
v_accvgpr_read v11, a97 // (op 6044) 
v_accvgpr_read v12, a98 // (op 6044) 
v_accvgpr_read v13, a99 // (op 6044) 
v_mul_f32 v14, v7, v10 // (op 6044) 
v_mul_f32 v15, v7, v11 // (op 6044) 
v_mul_f32 v16, v7, v12 // (op 6044) 
v_mul_f32 v17, v7, v13 // (op 6044) 



v_add_u32 v3, 384, v9 // (op 6047) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6047) Store value


v_accvgpr_read v10, a100 // (op 6057) 
v_accvgpr_read v11, a101 // (op 6057) 
v_accvgpr_read v12, a102 // (op 6057) 
v_accvgpr_read v13, a103 // (op 6057) 
v_mul_f32 v14, v7, v10 // (op 6057) 
v_mul_f32 v15, v7, v11 // (op 6057) 
v_mul_f32 v16, v7, v12 // (op 6057) 
v_mul_f32 v17, v7, v13 // (op 6057) 



s_lshl_b32 s0, s4, 6 // (op 6049) 
s_add_u32 s0, s0, 384 // (op 6049) 
v_add_u32 v3, v9, s0 // (op 6049) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6049) Store value


v_accvgpr_read v10, a104 // (op 6062) 
v_accvgpr_read v11, a105 // (op 6062) 
v_accvgpr_read v12, a106 // (op 6062) 
v_accvgpr_read v13, a107 // (op 6062) 
v_mul_f32 v14, v7, v10 // (op 6062) 
v_mul_f32 v15, v7, v11 // (op 6062) 
v_mul_f32 v16, v7, v12 // (op 6062) 
v_mul_f32 v17, v7, v13 // (op 6062) 



s_lshl_b32 s0, s4, 7 // (op 6051) 
s_add_u32 s0, s0, 384 // (op 6051) 
v_add_u32 v3, v9, s0 // (op 6051) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6051) Store value


v_accvgpr_read v10, a108 // (op 6067) 
v_accvgpr_read v11, a109 // (op 6067) 
v_accvgpr_read v12, a110 // (op 6067) 
v_accvgpr_read v13, a111 // (op 6067) 
v_mul_f32 v14, v7, v10 // (op 6067) 
v_mul_f32 v15, v7, v11 // (op 6067) 
v_mul_f32 v16, v7, v12 // (op 6067) 
v_mul_f32 v17, v7, v13 // (op 6067) 



s_lshl_b32 s0, s4, 6 // (op 6053) 
s_mul_i32 s1, 3, s0 // (op 6053) 
s_add_u32 s1, s1, 384 // (op 6053) 

v_add_u32 v3, v9, s1 // (op 6053) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6053) Store value


v_accvgpr_read v10, a112 // (op 6112) 
v_accvgpr_read v11, a113 // (op 6112) 
v_accvgpr_read v12, a114 // (op 6112) 
v_accvgpr_read v13, a115 // (op 6112) 
v_mul_f32 v14, v7, v10 // (op 6112) 
v_mul_f32 v15, v7, v11 // (op 6112) 
v_mul_f32 v16, v7, v12 // (op 6112) 
v_mul_f32 v17, v7, v13 // (op 6112) 



v_add_u32 v3, 448, v9 // (op 6115) 

buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6115) Store value


v_accvgpr_read v10, a116 // (op 6125) 
v_accvgpr_read v11, a117 // (op 6125) 
v_accvgpr_read v12, a118 // (op 6125) 
v_accvgpr_read v13, a119 // (op 6125) 
v_mul_f32 v14, v7, v10 // (op 6125) 
v_mul_f32 v15, v7, v11 // (op 6125) 
v_mul_f32 v16, v7, v12 // (op 6125) 
v_mul_f32 v17, v7, v13 // (op 6125) 



s_lshl_b32 s0, s4, 6 // (op 6117) 
s_add_u32 s0, s0, 448 // (op 6117) 
v_add_u32 v3, v9, s0 // (op 6117) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6117) Store value


v_accvgpr_read v10, a120 // (op 6130) 
v_accvgpr_read v11, a121 // (op 6130) 
v_accvgpr_read v12, a122 // (op 6130) 
v_accvgpr_read v13, a123 // (op 6130) 
v_mul_f32 v14, v7, v10 // (op 6130) 
v_mul_f32 v15, v7, v11 // (op 6130) 
v_mul_f32 v16, v7, v12 // (op 6130) 
v_mul_f32 v17, v7, v13 // (op 6130) 



s_lshl_b32 s0, s4, 7 // (op 6119) 
s_add_u32 s0, s0, 448 // (op 6119) 
v_add_u32 v3, v9, s0 // (op 6119) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6119) Store value


v_accvgpr_read v10, a124 // (op 6135) 
v_accvgpr_read v11, a125 // (op 6135) 
v_accvgpr_read v12, a126 // (op 6135) 
v_accvgpr_read v13, a127 // (op 6135) 
v_mul_f32 v14, v7, v10 // (op 6135) 
v_mul_f32 v15, v7, v11 // (op 6135) 
v_mul_f32 v16, v7, v12 // (op 6135) 
v_mul_f32 v17, v7, v13 // (op 6135) 



s_lshl_b32 s0, s4, 6 // (op 6121) 
s_mul_i32 s1, 3, s0 // (op 6121) 
s_add_u32 s1, s1, 448 // (op 6121) 

v_add_u32 v3, v9, s1 // (op 6121) 


buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 6121) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalBottom_0_beta_114
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalFalse_0_beta_114:


v_add_u32 v3, 448, v8 // (op 6851) 

buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6851) Load value

v_accvgpr_read v14, a112 // (op 6839) 
v_accvgpr_read v15, a113 // (op 6839) 
v_accvgpr_read v16, a114 // (op 6839) 
v_accvgpr_read v17, a115 // (op 6839) 
v_mul_f32 v18, v7, v14 // (op 6839) 
v_mul_f32 v19, v7, v15 // (op 6839) 
v_mul_f32 v20, v7, v16 // (op 6839) 
v_mul_f32 v21, v7, v17 // (op 6839) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6850) 
v_mul_f32 v15, v0, v11 // (op 6850) 
v_mul_f32 v16, v0, v12 // (op 6850) 
v_mul_f32 v17, v0, v13 // (op 6850) 

v_add_f32 v10, v14, v18 // (op 6840) 
v_add_f32 v11, v15, v19 // (op 6840) 
v_add_f32 v12, v16, v20 // (op 6840) 
v_add_f32 v13, v17, v21 // (op 6840) 




v_add_u32 v3, 448, v9 // (op 6842) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6842) Store value




s_lshl_b32 s0, s2, 6 // (op 6855) 
s_add_u32 s0, s0, 448 // (op 6855) 
v_add_u32 v3, v8, s0 // (op 6855) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6855) Load value

v_accvgpr_read v14, a116 // (op 6865) 
v_accvgpr_read v15, a117 // (op 6865) 
v_accvgpr_read v16, a118 // (op 6865) 
v_accvgpr_read v17, a119 // (op 6865) 
v_mul_f32 v18, v7, v14 // (op 6865) 
v_mul_f32 v19, v7, v15 // (op 6865) 
v_mul_f32 v20, v7, v16 // (op 6865) 
v_mul_f32 v21, v7, v17 // (op 6865) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6853) 
v_mul_f32 v15, v0, v11 // (op 6853) 
v_mul_f32 v16, v0, v12 // (op 6853) 
v_mul_f32 v17, v0, v13 // (op 6853) 

v_add_f32 v10, v14, v18 // (op 6854) 
v_add_f32 v11, v15, v19 // (op 6854) 
v_add_f32 v12, v16, v20 // (op 6854) 
v_add_f32 v13, v17, v21 // (op 6854) 




s_lshl_b32 s0, s4, 6 // (op 6844) 
s_add_u32 s0, s0, 448 // (op 6844) 
v_add_u32 v3, v9, s0 // (op 6844) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6844) Store value




s_lshl_b32 s0, s2, 7 // (op 6859) 
s_add_u32 s0, s0, 448 // (op 6859) 
v_add_u32 v3, v8, s0 // (op 6859) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6859) Load value

v_accvgpr_read v14, a120 // (op 6867) 
v_accvgpr_read v15, a121 // (op 6867) 
v_accvgpr_read v16, a122 // (op 6867) 
v_accvgpr_read v17, a123 // (op 6867) 
v_mul_f32 v18, v7, v14 // (op 6867) 
v_mul_f32 v19, v7, v15 // (op 6867) 
v_mul_f32 v20, v7, v16 // (op 6867) 
v_mul_f32 v21, v7, v17 // (op 6867) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6857) 
v_mul_f32 v15, v0, v11 // (op 6857) 
v_mul_f32 v16, v0, v12 // (op 6857) 
v_mul_f32 v17, v0, v13 // (op 6857) 

v_add_f32 v10, v14, v18 // (op 6858) 
v_add_f32 v11, v15, v19 // (op 6858) 
v_add_f32 v12, v16, v20 // (op 6858) 
v_add_f32 v13, v17, v21 // (op 6858) 




s_lshl_b32 s0, s4, 7 // (op 6846) 
s_add_u32 s0, s0, 448 // (op 6846) 
v_add_u32 v3, v9, s0 // (op 6846) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6846) Store value




s_lshl_b32 s0, s2, 6 // (op 6863) 
s_mul_i32 s1, 3, s0 // (op 6863) 
s_add_u32 s1, s1, 448 // (op 6863) 

v_add_u32 v3, v8, s1 // (op 6863) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6863) Load value

v_accvgpr_read v14, a124 // (op 6869) 
v_accvgpr_read v15, a125 // (op 6869) 
v_accvgpr_read v16, a126 // (op 6869) 
v_accvgpr_read v17, a127 // (op 6869) 
v_mul_f32 v18, v7, v14 // (op 6869) 
v_mul_f32 v19, v7, v15 // (op 6869) 
v_mul_f32 v20, v7, v16 // (op 6869) 
v_mul_f32 v21, v7, v17 // (op 6869) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6861) 
v_mul_f32 v15, v0, v11 // (op 6861) 
v_mul_f32 v16, v0, v12 // (op 6861) 
v_mul_f32 v17, v0, v13 // (op 6861) 

v_add_f32 v10, v14, v18 // (op 6862) 
v_add_f32 v11, v15, v19 // (op 6862) 
v_add_f32 v12, v16, v20 // (op 6862) 
v_add_f32 v13, v17, v21 // (op 6862) 




s_lshl_b32 s0, s4, 6 // (op 6848) 
s_mul_i32 s1, 3, s0 // (op 6848) 
s_add_u32 s1, s1, 448 // (op 6848) 

v_add_u32 v3, v9, s1 // (op 6848) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6848) Store value




v_add_u32 v3, 384, v8 // (op 6772) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6772) Load value

v_accvgpr_read v14, a96 // (op 6760) 
v_accvgpr_read v15, a97 // (op 6760) 
v_accvgpr_read v16, a98 // (op 6760) 
v_accvgpr_read v17, a99 // (op 6760) 
v_mul_f32 v18, v7, v14 // (op 6760) 
v_mul_f32 v19, v7, v15 // (op 6760) 
v_mul_f32 v20, v7, v16 // (op 6760) 
v_mul_f32 v21, v7, v17 // (op 6760) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6771) 
v_mul_f32 v15, v0, v11 // (op 6771) 
v_mul_f32 v16, v0, v12 // (op 6771) 
v_mul_f32 v17, v0, v13 // (op 6771) 

v_add_f32 v10, v14, v18 // (op 6761) 
v_add_f32 v11, v15, v19 // (op 6761) 
v_add_f32 v12, v16, v20 // (op 6761) 
v_add_f32 v13, v17, v21 // (op 6761) 




v_add_u32 v3, 384, v9 // (op 6763) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6763) Store value




s_lshl_b32 s0, s2, 6 // (op 6776) 
s_add_u32 s0, s0, 384 // (op 6776) 
v_add_u32 v3, v8, s0 // (op 6776) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6776) Load value

v_accvgpr_read v14, a100 // (op 6786) 
v_accvgpr_read v15, a101 // (op 6786) 
v_accvgpr_read v16, a102 // (op 6786) 
v_accvgpr_read v17, a103 // (op 6786) 
v_mul_f32 v18, v7, v14 // (op 6786) 
v_mul_f32 v19, v7, v15 // (op 6786) 
v_mul_f32 v20, v7, v16 // (op 6786) 
v_mul_f32 v21, v7, v17 // (op 6786) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6774) 
v_mul_f32 v15, v0, v11 // (op 6774) 
v_mul_f32 v16, v0, v12 // (op 6774) 
v_mul_f32 v17, v0, v13 // (op 6774) 

v_add_f32 v10, v14, v18 // (op 6775) 
v_add_f32 v11, v15, v19 // (op 6775) 
v_add_f32 v12, v16, v20 // (op 6775) 
v_add_f32 v13, v17, v21 // (op 6775) 




s_lshl_b32 s0, s4, 6 // (op 6765) 
s_add_u32 s0, s0, 384 // (op 6765) 
v_add_u32 v3, v9, s0 // (op 6765) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6765) Store value




s_lshl_b32 s0, s2, 7 // (op 6780) 
s_add_u32 s0, s0, 384 // (op 6780) 
v_add_u32 v3, v8, s0 // (op 6780) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6780) Load value

v_accvgpr_read v14, a104 // (op 6788) 
v_accvgpr_read v15, a105 // (op 6788) 
v_accvgpr_read v16, a106 // (op 6788) 
v_accvgpr_read v17, a107 // (op 6788) 
v_mul_f32 v18, v7, v14 // (op 6788) 
v_mul_f32 v19, v7, v15 // (op 6788) 
v_mul_f32 v20, v7, v16 // (op 6788) 
v_mul_f32 v21, v7, v17 // (op 6788) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6778) 
v_mul_f32 v15, v0, v11 // (op 6778) 
v_mul_f32 v16, v0, v12 // (op 6778) 
v_mul_f32 v17, v0, v13 // (op 6778) 

v_add_f32 v10, v14, v18 // (op 6779) 
v_add_f32 v11, v15, v19 // (op 6779) 
v_add_f32 v12, v16, v20 // (op 6779) 
v_add_f32 v13, v17, v21 // (op 6779) 




s_lshl_b32 s0, s4, 7 // (op 6767) 
s_add_u32 s0, s0, 384 // (op 6767) 
v_add_u32 v3, v9, s0 // (op 6767) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6767) Store value




s_lshl_b32 s0, s2, 6 // (op 6784) 
s_mul_i32 s1, 3, s0 // (op 6784) 
s_add_u32 s1, s1, 384 // (op 6784) 

v_add_u32 v3, v8, s1 // (op 6784) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6784) Load value

v_accvgpr_read v14, a108 // (op 6790) 
v_accvgpr_read v15, a109 // (op 6790) 
v_accvgpr_read v16, a110 // (op 6790) 
v_accvgpr_read v17, a111 // (op 6790) 
v_mul_f32 v18, v7, v14 // (op 6790) 
v_mul_f32 v19, v7, v15 // (op 6790) 
v_mul_f32 v20, v7, v16 // (op 6790) 
v_mul_f32 v21, v7, v17 // (op 6790) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6782) 
v_mul_f32 v15, v0, v11 // (op 6782) 
v_mul_f32 v16, v0, v12 // (op 6782) 
v_mul_f32 v17, v0, v13 // (op 6782) 

v_add_f32 v10, v14, v18 // (op 6783) 
v_add_f32 v11, v15, v19 // (op 6783) 
v_add_f32 v12, v16, v20 // (op 6783) 
v_add_f32 v13, v17, v21 // (op 6783) 




s_lshl_b32 s0, s4, 6 // (op 6769) 
s_mul_i32 s1, 3, s0 // (op 6769) 
s_add_u32 s1, s1, 384 // (op 6769) 

v_add_u32 v3, v9, s1 // (op 6769) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6769) Store value




v_add_u32 v3, 320, v8 // (op 6693) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6693) Load value

v_accvgpr_read v14, a80 // (op 6681) 
v_accvgpr_read v15, a81 // (op 6681) 
v_accvgpr_read v16, a82 // (op 6681) 
v_accvgpr_read v17, a83 // (op 6681) 
v_mul_f32 v18, v7, v14 // (op 6681) 
v_mul_f32 v19, v7, v15 // (op 6681) 
v_mul_f32 v20, v7, v16 // (op 6681) 
v_mul_f32 v21, v7, v17 // (op 6681) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6692) 
v_mul_f32 v15, v0, v11 // (op 6692) 
v_mul_f32 v16, v0, v12 // (op 6692) 
v_mul_f32 v17, v0, v13 // (op 6692) 

v_add_f32 v10, v14, v18 // (op 6682) 
v_add_f32 v11, v15, v19 // (op 6682) 
v_add_f32 v12, v16, v20 // (op 6682) 
v_add_f32 v13, v17, v21 // (op 6682) 




v_add_u32 v3, 320, v9 // (op 6684) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6684) Store value




s_lshl_b32 s0, s2, 6 // (op 6697) 
s_add_u32 s0, s0, 320 // (op 6697) 
v_add_u32 v3, v8, s0 // (op 6697) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6697) Load value

v_accvgpr_read v14, a84 // (op 6707) 
v_accvgpr_read v15, a85 // (op 6707) 
v_accvgpr_read v16, a86 // (op 6707) 
v_accvgpr_read v17, a87 // (op 6707) 
v_mul_f32 v18, v7, v14 // (op 6707) 
v_mul_f32 v19, v7, v15 // (op 6707) 
v_mul_f32 v20, v7, v16 // (op 6707) 
v_mul_f32 v21, v7, v17 // (op 6707) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6695) 
v_mul_f32 v15, v0, v11 // (op 6695) 
v_mul_f32 v16, v0, v12 // (op 6695) 
v_mul_f32 v17, v0, v13 // (op 6695) 

v_add_f32 v10, v14, v18 // (op 6696) 
v_add_f32 v11, v15, v19 // (op 6696) 
v_add_f32 v12, v16, v20 // (op 6696) 
v_add_f32 v13, v17, v21 // (op 6696) 




s_lshl_b32 s0, s4, 6 // (op 6686) 
s_add_u32 s0, s0, 320 // (op 6686) 
v_add_u32 v3, v9, s0 // (op 6686) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6686) Store value




s_lshl_b32 s0, s2, 7 // (op 6701) 
s_add_u32 s0, s0, 320 // (op 6701) 
v_add_u32 v3, v8, s0 // (op 6701) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6701) Load value

v_accvgpr_read v14, a88 // (op 6709) 
v_accvgpr_read v15, a89 // (op 6709) 
v_accvgpr_read v16, a90 // (op 6709) 
v_accvgpr_read v17, a91 // (op 6709) 
v_mul_f32 v18, v7, v14 // (op 6709) 
v_mul_f32 v19, v7, v15 // (op 6709) 
v_mul_f32 v20, v7, v16 // (op 6709) 
v_mul_f32 v21, v7, v17 // (op 6709) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6699) 
v_mul_f32 v15, v0, v11 // (op 6699) 
v_mul_f32 v16, v0, v12 // (op 6699) 
v_mul_f32 v17, v0, v13 // (op 6699) 

v_add_f32 v10, v14, v18 // (op 6700) 
v_add_f32 v11, v15, v19 // (op 6700) 
v_add_f32 v12, v16, v20 // (op 6700) 
v_add_f32 v13, v17, v21 // (op 6700) 




s_lshl_b32 s0, s4, 7 // (op 6688) 
s_add_u32 s0, s0, 320 // (op 6688) 
v_add_u32 v3, v9, s0 // (op 6688) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6688) Store value




s_lshl_b32 s0, s2, 6 // (op 6705) 
s_mul_i32 s1, 3, s0 // (op 6705) 
s_add_u32 s1, s1, 320 // (op 6705) 

v_add_u32 v3, v8, s1 // (op 6705) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6705) Load value

v_accvgpr_read v14, a92 // (op 6711) 
v_accvgpr_read v15, a93 // (op 6711) 
v_accvgpr_read v16, a94 // (op 6711) 
v_accvgpr_read v17, a95 // (op 6711) 
v_mul_f32 v18, v7, v14 // (op 6711) 
v_mul_f32 v19, v7, v15 // (op 6711) 
v_mul_f32 v20, v7, v16 // (op 6711) 
v_mul_f32 v21, v7, v17 // (op 6711) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6703) 
v_mul_f32 v15, v0, v11 // (op 6703) 
v_mul_f32 v16, v0, v12 // (op 6703) 
v_mul_f32 v17, v0, v13 // (op 6703) 

v_add_f32 v10, v14, v18 // (op 6704) 
v_add_f32 v11, v15, v19 // (op 6704) 
v_add_f32 v12, v16, v20 // (op 6704) 
v_add_f32 v13, v17, v21 // (op 6704) 




s_lshl_b32 s0, s4, 6 // (op 6690) 
s_mul_i32 s1, 3, s0 // (op 6690) 
s_add_u32 s1, s1, 320 // (op 6690) 

v_add_u32 v3, v9, s1 // (op 6690) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6690) Store value




v_add_u32 v3, 256, v8 // (op 6614) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6614) Load value

v_accvgpr_read v14, a64 // (op 6602) 
v_accvgpr_read v15, a65 // (op 6602) 
v_accvgpr_read v16, a66 // (op 6602) 
v_accvgpr_read v17, a67 // (op 6602) 
v_mul_f32 v18, v7, v14 // (op 6602) 
v_mul_f32 v19, v7, v15 // (op 6602) 
v_mul_f32 v20, v7, v16 // (op 6602) 
v_mul_f32 v21, v7, v17 // (op 6602) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6613) 
v_mul_f32 v15, v0, v11 // (op 6613) 
v_mul_f32 v16, v0, v12 // (op 6613) 
v_mul_f32 v17, v0, v13 // (op 6613) 

v_add_f32 v10, v14, v18 // (op 6603) 
v_add_f32 v11, v15, v19 // (op 6603) 
v_add_f32 v12, v16, v20 // (op 6603) 
v_add_f32 v13, v17, v21 // (op 6603) 




v_add_u32 v3, 256, v9 // (op 6605) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6605) Store value




s_lshl_b32 s0, s2, 6 // (op 6618) 
s_add_u32 s0, s0, 256 // (op 6618) 
v_add_u32 v3, v8, s0 // (op 6618) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6618) Load value

v_accvgpr_read v14, a68 // (op 6628) 
v_accvgpr_read v15, a69 // (op 6628) 
v_accvgpr_read v16, a70 // (op 6628) 
v_accvgpr_read v17, a71 // (op 6628) 
v_mul_f32 v18, v7, v14 // (op 6628) 
v_mul_f32 v19, v7, v15 // (op 6628) 
v_mul_f32 v20, v7, v16 // (op 6628) 
v_mul_f32 v21, v7, v17 // (op 6628) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6616) 
v_mul_f32 v15, v0, v11 // (op 6616) 
v_mul_f32 v16, v0, v12 // (op 6616) 
v_mul_f32 v17, v0, v13 // (op 6616) 

v_add_f32 v10, v14, v18 // (op 6617) 
v_add_f32 v11, v15, v19 // (op 6617) 
v_add_f32 v12, v16, v20 // (op 6617) 
v_add_f32 v13, v17, v21 // (op 6617) 




s_lshl_b32 s0, s4, 6 // (op 6607) 
s_add_u32 s0, s0, 256 // (op 6607) 
v_add_u32 v3, v9, s0 // (op 6607) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6607) Store value




s_lshl_b32 s0, s2, 7 // (op 6622) 
s_add_u32 s0, s0, 256 // (op 6622) 
v_add_u32 v3, v8, s0 // (op 6622) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6622) Load value

v_accvgpr_read v14, a72 // (op 6630) 
v_accvgpr_read v15, a73 // (op 6630) 
v_accvgpr_read v16, a74 // (op 6630) 
v_accvgpr_read v17, a75 // (op 6630) 
v_mul_f32 v18, v7, v14 // (op 6630) 
v_mul_f32 v19, v7, v15 // (op 6630) 
v_mul_f32 v20, v7, v16 // (op 6630) 
v_mul_f32 v21, v7, v17 // (op 6630) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6620) 
v_mul_f32 v15, v0, v11 // (op 6620) 
v_mul_f32 v16, v0, v12 // (op 6620) 
v_mul_f32 v17, v0, v13 // (op 6620) 

v_add_f32 v10, v14, v18 // (op 6621) 
v_add_f32 v11, v15, v19 // (op 6621) 
v_add_f32 v12, v16, v20 // (op 6621) 
v_add_f32 v13, v17, v21 // (op 6621) 




s_lshl_b32 s0, s4, 7 // (op 6609) 
s_add_u32 s0, s0, 256 // (op 6609) 
v_add_u32 v3, v9, s0 // (op 6609) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6609) Store value




s_lshl_b32 s0, s2, 6 // (op 6626) 
s_mul_i32 s1, 3, s0 // (op 6626) 
s_add_u32 s1, s1, 256 // (op 6626) 

v_add_u32 v3, v8, s1 // (op 6626) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6626) Load value

v_accvgpr_read v14, a76 // (op 6632) 
v_accvgpr_read v15, a77 // (op 6632) 
v_accvgpr_read v16, a78 // (op 6632) 
v_accvgpr_read v17, a79 // (op 6632) 
v_mul_f32 v18, v7, v14 // (op 6632) 
v_mul_f32 v19, v7, v15 // (op 6632) 
v_mul_f32 v20, v7, v16 // (op 6632) 
v_mul_f32 v21, v7, v17 // (op 6632) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6624) 
v_mul_f32 v15, v0, v11 // (op 6624) 
v_mul_f32 v16, v0, v12 // (op 6624) 
v_mul_f32 v17, v0, v13 // (op 6624) 

v_add_f32 v10, v14, v18 // (op 6625) 
v_add_f32 v11, v15, v19 // (op 6625) 
v_add_f32 v12, v16, v20 // (op 6625) 
v_add_f32 v13, v17, v21 // (op 6625) 




s_lshl_b32 s0, s4, 6 // (op 6611) 
s_mul_i32 s1, 3, s0 // (op 6611) 
s_add_u32 s1, s1, 256 // (op 6611) 

v_add_u32 v3, v9, s1 // (op 6611) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6611) Store value




v_add_u32 v3, 192, v8 // (op 6535) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6535) Load value

v_accvgpr_read v14, a48 // (op 6523) 
v_accvgpr_read v15, a49 // (op 6523) 
v_accvgpr_read v16, a50 // (op 6523) 
v_accvgpr_read v17, a51 // (op 6523) 
v_mul_f32 v18, v7, v14 // (op 6523) 
v_mul_f32 v19, v7, v15 // (op 6523) 
v_mul_f32 v20, v7, v16 // (op 6523) 
v_mul_f32 v21, v7, v17 // (op 6523) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6534) 
v_mul_f32 v15, v0, v11 // (op 6534) 
v_mul_f32 v16, v0, v12 // (op 6534) 
v_mul_f32 v17, v0, v13 // (op 6534) 

v_add_f32 v10, v14, v18 // (op 6524) 
v_add_f32 v11, v15, v19 // (op 6524) 
v_add_f32 v12, v16, v20 // (op 6524) 
v_add_f32 v13, v17, v21 // (op 6524) 




v_add_u32 v3, 192, v9 // (op 6526) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6526) Store value




s_lshl_b32 s0, s2, 6 // (op 6539) 
s_add_u32 s0, s0, 192 // (op 6539) 
v_add_u32 v3, v8, s0 // (op 6539) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6539) Load value

v_accvgpr_read v14, a52 // (op 6549) 
v_accvgpr_read v15, a53 // (op 6549) 
v_accvgpr_read v16, a54 // (op 6549) 
v_accvgpr_read v17, a55 // (op 6549) 
v_mul_f32 v18, v7, v14 // (op 6549) 
v_mul_f32 v19, v7, v15 // (op 6549) 
v_mul_f32 v20, v7, v16 // (op 6549) 
v_mul_f32 v21, v7, v17 // (op 6549) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6537) 
v_mul_f32 v15, v0, v11 // (op 6537) 
v_mul_f32 v16, v0, v12 // (op 6537) 
v_mul_f32 v17, v0, v13 // (op 6537) 

v_add_f32 v10, v14, v18 // (op 6538) 
v_add_f32 v11, v15, v19 // (op 6538) 
v_add_f32 v12, v16, v20 // (op 6538) 
v_add_f32 v13, v17, v21 // (op 6538) 




s_lshl_b32 s0, s4, 6 // (op 6528) 
s_add_u32 s0, s0, 192 // (op 6528) 
v_add_u32 v3, v9, s0 // (op 6528) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6528) Store value




s_lshl_b32 s0, s2, 7 // (op 6543) 
s_add_u32 s0, s0, 192 // (op 6543) 
v_add_u32 v3, v8, s0 // (op 6543) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6543) Load value

v_accvgpr_read v14, a56 // (op 6551) 
v_accvgpr_read v15, a57 // (op 6551) 
v_accvgpr_read v16, a58 // (op 6551) 
v_accvgpr_read v17, a59 // (op 6551) 
v_mul_f32 v18, v7, v14 // (op 6551) 
v_mul_f32 v19, v7, v15 // (op 6551) 
v_mul_f32 v20, v7, v16 // (op 6551) 
v_mul_f32 v21, v7, v17 // (op 6551) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6541) 
v_mul_f32 v15, v0, v11 // (op 6541) 
v_mul_f32 v16, v0, v12 // (op 6541) 
v_mul_f32 v17, v0, v13 // (op 6541) 

v_add_f32 v10, v14, v18 // (op 6542) 
v_add_f32 v11, v15, v19 // (op 6542) 
v_add_f32 v12, v16, v20 // (op 6542) 
v_add_f32 v13, v17, v21 // (op 6542) 




s_lshl_b32 s0, s4, 7 // (op 6530) 
s_add_u32 s0, s0, 192 // (op 6530) 
v_add_u32 v3, v9, s0 // (op 6530) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6530) Store value




s_lshl_b32 s0, s2, 6 // (op 6547) 
s_mul_i32 s1, 3, s0 // (op 6547) 
s_add_u32 s1, s1, 192 // (op 6547) 

v_add_u32 v3, v8, s1 // (op 6547) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6547) Load value

v_accvgpr_read v14, a60 // (op 6553) 
v_accvgpr_read v15, a61 // (op 6553) 
v_accvgpr_read v16, a62 // (op 6553) 
v_accvgpr_read v17, a63 // (op 6553) 
v_mul_f32 v18, v7, v14 // (op 6553) 
v_mul_f32 v19, v7, v15 // (op 6553) 
v_mul_f32 v20, v7, v16 // (op 6553) 
v_mul_f32 v21, v7, v17 // (op 6553) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6545) 
v_mul_f32 v15, v0, v11 // (op 6545) 
v_mul_f32 v16, v0, v12 // (op 6545) 
v_mul_f32 v17, v0, v13 // (op 6545) 

v_add_f32 v10, v14, v18 // (op 6546) 
v_add_f32 v11, v15, v19 // (op 6546) 
v_add_f32 v12, v16, v20 // (op 6546) 
v_add_f32 v13, v17, v21 // (op 6546) 




s_lshl_b32 s0, s4, 6 // (op 6532) 
s_mul_i32 s1, 3, s0 // (op 6532) 
s_add_u32 s1, s1, 192 // (op 6532) 

v_add_u32 v3, v9, s1 // (op 6532) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6532) Store value




v_add_u32 v3, 128, v8 // (op 6456) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6456) Load value

v_accvgpr_read v14, a32 // (op 6444) 
v_accvgpr_read v15, a33 // (op 6444) 
v_accvgpr_read v16, a34 // (op 6444) 
v_accvgpr_read v17, a35 // (op 6444) 
v_mul_f32 v18, v7, v14 // (op 6444) 
v_mul_f32 v19, v7, v15 // (op 6444) 
v_mul_f32 v20, v7, v16 // (op 6444) 
v_mul_f32 v21, v7, v17 // (op 6444) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6455) 
v_mul_f32 v15, v0, v11 // (op 6455) 
v_mul_f32 v16, v0, v12 // (op 6455) 
v_mul_f32 v17, v0, v13 // (op 6455) 

v_add_f32 v10, v14, v18 // (op 6445) 
v_add_f32 v11, v15, v19 // (op 6445) 
v_add_f32 v12, v16, v20 // (op 6445) 
v_add_f32 v13, v17, v21 // (op 6445) 




v_add_u32 v3, 128, v9 // (op 6447) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6447) Store value




s_lshl_b32 s0, s2, 6 // (op 6460) 
s_add_u32 s0, s0, 128 // (op 6460) 
v_add_u32 v3, v8, s0 // (op 6460) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6460) Load value

v_accvgpr_read v14, a36 // (op 6470) 
v_accvgpr_read v15, a37 // (op 6470) 
v_accvgpr_read v16, a38 // (op 6470) 
v_accvgpr_read v17, a39 // (op 6470) 
v_mul_f32 v18, v7, v14 // (op 6470) 
v_mul_f32 v19, v7, v15 // (op 6470) 
v_mul_f32 v20, v7, v16 // (op 6470) 
v_mul_f32 v21, v7, v17 // (op 6470) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6458) 
v_mul_f32 v15, v0, v11 // (op 6458) 
v_mul_f32 v16, v0, v12 // (op 6458) 
v_mul_f32 v17, v0, v13 // (op 6458) 

v_add_f32 v10, v14, v18 // (op 6459) 
v_add_f32 v11, v15, v19 // (op 6459) 
v_add_f32 v12, v16, v20 // (op 6459) 
v_add_f32 v13, v17, v21 // (op 6459) 




s_lshl_b32 s0, s4, 6 // (op 6449) 
s_add_u32 s0, s0, 128 // (op 6449) 
v_add_u32 v3, v9, s0 // (op 6449) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6449) Store value




s_lshl_b32 s0, s2, 7 // (op 6464) 
s_add_u32 s0, s0, 128 // (op 6464) 
v_add_u32 v3, v8, s0 // (op 6464) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6464) Load value

v_accvgpr_read v14, a40 // (op 6472) 
v_accvgpr_read v15, a41 // (op 6472) 
v_accvgpr_read v16, a42 // (op 6472) 
v_accvgpr_read v17, a43 // (op 6472) 
v_mul_f32 v18, v7, v14 // (op 6472) 
v_mul_f32 v19, v7, v15 // (op 6472) 
v_mul_f32 v20, v7, v16 // (op 6472) 
v_mul_f32 v21, v7, v17 // (op 6472) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6462) 
v_mul_f32 v15, v0, v11 // (op 6462) 
v_mul_f32 v16, v0, v12 // (op 6462) 
v_mul_f32 v17, v0, v13 // (op 6462) 

v_add_f32 v10, v14, v18 // (op 6463) 
v_add_f32 v11, v15, v19 // (op 6463) 
v_add_f32 v12, v16, v20 // (op 6463) 
v_add_f32 v13, v17, v21 // (op 6463) 




s_lshl_b32 s0, s4, 7 // (op 6451) 
s_add_u32 s0, s0, 128 // (op 6451) 
v_add_u32 v3, v9, s0 // (op 6451) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6451) Store value




s_lshl_b32 s0, s2, 6 // (op 6468) 
s_mul_i32 s1, 3, s0 // (op 6468) 
s_add_u32 s1, s1, 128 // (op 6468) 

v_add_u32 v3, v8, s1 // (op 6468) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6468) Load value

v_accvgpr_read v14, a44 // (op 6474) 
v_accvgpr_read v15, a45 // (op 6474) 
v_accvgpr_read v16, a46 // (op 6474) 
v_accvgpr_read v17, a47 // (op 6474) 
v_mul_f32 v18, v7, v14 // (op 6474) 
v_mul_f32 v19, v7, v15 // (op 6474) 
v_mul_f32 v20, v7, v16 // (op 6474) 
v_mul_f32 v21, v7, v17 // (op 6474) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6466) 
v_mul_f32 v15, v0, v11 // (op 6466) 
v_mul_f32 v16, v0, v12 // (op 6466) 
v_mul_f32 v17, v0, v13 // (op 6466) 

v_add_f32 v10, v14, v18 // (op 6467) 
v_add_f32 v11, v15, v19 // (op 6467) 
v_add_f32 v12, v16, v20 // (op 6467) 
v_add_f32 v13, v17, v21 // (op 6467) 




s_lshl_b32 s0, s4, 6 // (op 6453) 
s_mul_i32 s1, 3, s0 // (op 6453) 
s_add_u32 s1, s1, 128 // (op 6453) 

v_add_u32 v3, v9, s1 // (op 6453) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6453) Store value




v_add_u32 v3, 64, v8 // (op 6377) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6377) Load value

v_accvgpr_read v14, a16 // (op 6365) 
v_accvgpr_read v15, a17 // (op 6365) 
v_accvgpr_read v16, a18 // (op 6365) 
v_accvgpr_read v17, a19 // (op 6365) 
v_mul_f32 v18, v7, v14 // (op 6365) 
v_mul_f32 v19, v7, v15 // (op 6365) 
v_mul_f32 v20, v7, v16 // (op 6365) 
v_mul_f32 v21, v7, v17 // (op 6365) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6376) 
v_mul_f32 v15, v0, v11 // (op 6376) 
v_mul_f32 v16, v0, v12 // (op 6376) 
v_mul_f32 v17, v0, v13 // (op 6376) 

v_add_f32 v10, v14, v18 // (op 6366) 
v_add_f32 v11, v15, v19 // (op 6366) 
v_add_f32 v12, v16, v20 // (op 6366) 
v_add_f32 v13, v17, v21 // (op 6366) 




v_add_u32 v3, 64, v9 // (op 6368) 

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6368) Store value




s_lshl_b32 s0, s2, 6 // (op 6381) 
s_add_u32 s0, s0, 64 // (op 6381) 
v_add_u32 v3, v8, s0 // (op 6381) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6381) Load value

v_accvgpr_read v14, a20 // (op 6391) 
v_accvgpr_read v15, a21 // (op 6391) 
v_accvgpr_read v16, a22 // (op 6391) 
v_accvgpr_read v17, a23 // (op 6391) 
v_mul_f32 v18, v7, v14 // (op 6391) 
v_mul_f32 v19, v7, v15 // (op 6391) 
v_mul_f32 v20, v7, v16 // (op 6391) 
v_mul_f32 v21, v7, v17 // (op 6391) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6379) 
v_mul_f32 v15, v0, v11 // (op 6379) 
v_mul_f32 v16, v0, v12 // (op 6379) 
v_mul_f32 v17, v0, v13 // (op 6379) 

v_add_f32 v10, v14, v18 // (op 6380) 
v_add_f32 v11, v15, v19 // (op 6380) 
v_add_f32 v12, v16, v20 // (op 6380) 
v_add_f32 v13, v17, v21 // (op 6380) 




s_lshl_b32 s0, s4, 6 // (op 6370) 
s_add_u32 s0, s0, 64 // (op 6370) 
v_add_u32 v3, v9, s0 // (op 6370) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6370) Store value




s_lshl_b32 s0, s2, 7 // (op 6385) 
s_add_u32 s0, s0, 64 // (op 6385) 
v_add_u32 v3, v8, s0 // (op 6385) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6385) Load value

v_accvgpr_read v14, a24 // (op 6393) 
v_accvgpr_read v15, a25 // (op 6393) 
v_accvgpr_read v16, a26 // (op 6393) 
v_accvgpr_read v17, a27 // (op 6393) 
v_mul_f32 v18, v7, v14 // (op 6393) 
v_mul_f32 v19, v7, v15 // (op 6393) 
v_mul_f32 v20, v7, v16 // (op 6393) 
v_mul_f32 v21, v7, v17 // (op 6393) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6383) 
v_mul_f32 v15, v0, v11 // (op 6383) 
v_mul_f32 v16, v0, v12 // (op 6383) 
v_mul_f32 v17, v0, v13 // (op 6383) 

v_add_f32 v10, v14, v18 // (op 6384) 
v_add_f32 v11, v15, v19 // (op 6384) 
v_add_f32 v12, v16, v20 // (op 6384) 
v_add_f32 v13, v17, v21 // (op 6384) 




s_lshl_b32 s0, s4, 7 // (op 6372) 
s_add_u32 s0, s0, 64 // (op 6372) 
v_add_u32 v3, v9, s0 // (op 6372) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6372) Store value




s_lshl_b32 s0, s2, 6 // (op 6389) 
s_mul_i32 s1, 3, s0 // (op 6389) 
s_add_u32 s1, s1, 64 // (op 6389) 

v_add_u32 v3, v8, s1 // (op 6389) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6389) Load value

v_accvgpr_read v14, a28 // (op 6395) 
v_accvgpr_read v15, a29 // (op 6395) 
v_accvgpr_read v16, a30 // (op 6395) 
v_accvgpr_read v17, a31 // (op 6395) 
v_mul_f32 v18, v7, v14 // (op 6395) 
v_mul_f32 v19, v7, v15 // (op 6395) 
v_mul_f32 v20, v7, v16 // (op 6395) 
v_mul_f32 v21, v7, v17 // (op 6395) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6387) 
v_mul_f32 v15, v0, v11 // (op 6387) 
v_mul_f32 v16, v0, v12 // (op 6387) 
v_mul_f32 v17, v0, v13 // (op 6387) 

v_add_f32 v10, v14, v18 // (op 6388) 
v_add_f32 v11, v15, v19 // (op 6388) 
v_add_f32 v12, v16, v20 // (op 6388) 
v_add_f32 v13, v17, v21 // (op 6388) 




s_lshl_b32 s0, s4, 6 // (op 6374) 
s_mul_i32 s1, 3, s0 // (op 6374) 
s_add_u32 s1, s1, 64 // (op 6374) 

v_add_u32 v3, v9, s1 // (op 6374) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6374) Store value




v_mov_b32 v3, v8 // (op 126) reg expression

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 126) Load value

v_accvgpr_read v14, a0 // (op 123) 
v_accvgpr_read v15, a1 // (op 123) 
v_accvgpr_read v16, a2 // (op 123) 
v_accvgpr_read v17, a3 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 
v_mul_f32 v20, v7, v16 // (op 123) 
v_mul_f32 v21, v7, v17 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 127) 
v_mul_f32 v15, v0, v11 // (op 127) 
v_mul_f32 v16, v0, v12 // (op 127) 
v_mul_f32 v17, v0, v13 // (op 127) 

v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 
v_add_f32 v12, v16, v20 // (op 124) 
v_add_f32 v13, v17, v21 // (op 124) 




v_mov_b32 v3, v9 // (op 125) reg expression

buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 125) Store value




s_lshl_b32 s0, s2, 6 // (op 6291) 
v_add_u32 v3, v8, s0 // (op 6291) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6291) Load value

v_accvgpr_read v14, a4 // (op 6288) 
v_accvgpr_read v15, a5 // (op 6288) 
v_accvgpr_read v16, a6 // (op 6288) 
v_accvgpr_read v17, a7 // (op 6288) 
v_mul_f32 v18, v7, v14 // (op 6288) 
v_mul_f32 v19, v7, v15 // (op 6288) 
v_mul_f32 v20, v7, v16 // (op 6288) 
v_mul_f32 v21, v7, v17 // (op 6288) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6292) 
v_mul_f32 v15, v0, v11 // (op 6292) 
v_mul_f32 v16, v0, v12 // (op 6292) 
v_mul_f32 v17, v0, v13 // (op 6292) 

v_add_f32 v10, v14, v18 // (op 6289) 
v_add_f32 v11, v15, v19 // (op 6289) 
v_add_f32 v12, v16, v20 // (op 6289) 
v_add_f32 v13, v17, v21 // (op 6289) 




s_lshl_b32 s0, s4, 6 // (op 6290) 
v_add_u32 v3, v9, s0 // (op 6290) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6290) Store value




s_lshl_b32 s0, s2, 7 // (op 6303) 
v_add_u32 v3, v8, s0 // (op 6303) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6303) Load value

v_accvgpr_read v14, a8 // (op 6300) 
v_accvgpr_read v15, a9 // (op 6300) 
v_accvgpr_read v16, a10 // (op 6300) 
v_accvgpr_read v17, a11 // (op 6300) 
v_mul_f32 v18, v7, v14 // (op 6300) 
v_mul_f32 v19, v7, v15 // (op 6300) 
v_mul_f32 v20, v7, v16 // (op 6300) 
v_mul_f32 v21, v7, v17 // (op 6300) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6304) 
v_mul_f32 v15, v0, v11 // (op 6304) 
v_mul_f32 v16, v0, v12 // (op 6304) 
v_mul_f32 v17, v0, v13 // (op 6304) 

v_add_f32 v10, v14, v18 // (op 6301) 
v_add_f32 v11, v15, v19 // (op 6301) 
v_add_f32 v12, v16, v20 // (op 6301) 
v_add_f32 v13, v17, v21 // (op 6301) 




s_lshl_b32 s0, s4, 7 // (op 6302) 
v_add_u32 v3, v9, s0 // (op 6302) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6302) Store value




s_lshl_b32 s0, s2, 6 // (op 6315) 
s_mul_i32 s1, 3, s0 // (op 6315) 

v_add_u32 v3, v8, s1 // (op 6315) 


buffer_load_dwordx4 v[10:13], v3, s[52:55], 0 offen // (op 6315) Load value

v_accvgpr_read v14, a12 // (op 6312) 
v_accvgpr_read v15, a13 // (op 6312) 
v_accvgpr_read v16, a14 // (op 6312) 
v_accvgpr_read v17, a15 // (op 6312) 
v_mul_f32 v18, v7, v14 // (op 6312) 
v_mul_f32 v19, v7, v15 // (op 6312) 
v_mul_f32 v20, v7, v16 // (op 6312) 
v_mul_f32 v21, v7, v17 // (op 6312) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6316) 
v_mul_f32 v15, v0, v11 // (op 6316) 
v_mul_f32 v16, v0, v12 // (op 6316) 
v_mul_f32 v17, v0, v13 // (op 6316) 

v_add_f32 v10, v14, v18 // (op 6313) 
v_add_f32 v11, v15, v19 // (op 6313) 
v_add_f32 v12, v16, v20 // (op 6313) 
v_add_f32 v13, v17, v21 // (op 6313) 




s_lshl_b32 s0, s4, 6 // (op 6314) 
s_mul_i32 s1, 3, s0 // (op 6314) 

v_add_u32 v3, v9, s1 // (op 6314) 


buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 6314) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalBottom_0_beta_114:








































s_endpgm  // End of RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
.LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_, .LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__end-RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 296
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 73728 // lds bytes
  .amdhsa_accum_offset 168
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
    macrotile: [256, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 73728
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 166
    .agpr_count: 128
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}}
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

