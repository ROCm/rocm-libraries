

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 3837) 
s_mul_hi_i32 s1, s16, s10 // (op 4024) 
v_and_b32 v0, 15, v1 // (op 3950) 
v_and_b32 v3, 48, v1 // (op 3950) 
v_lshl_add_u32 v4, v0, 7, v3 // (op 3950) 


v_and_b32 v0, 15, v1 // (op 4109) 
v_and_b32 v3, 48, v1 // (op 4109) 
v_lshl_add_u32 v5, v0, 7, v3 // (op 4109) 


v_lshrrev_b32 v0, 6, v1 // (op 3837) 
s_add_i32 s18, s0, s16 // (op 3837) 

v_and_b32 v3, 63, v1 // (op 3837) 
s_ashr_i32 s0, s18, 31 // (op 3837) 
v_lshrrev_b32 v6, 4, v3 // (op 3837) 
s_cmp_eq_i32 s10, 0 // (op 3837) 
s_mov_b32 s19, scc // (op 3837) 
v_lshlrev_b32 v7, 2, v6 // (op 3837) 

s_cmp_eq_i32 s19, 0 // (op 3837) 
s_cselect_b32 s49, 0, -1 // (op 3837) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3837) 


s_and_b32 s19, s13, 31 // (op 3837) 
v_lshrrev_b32 v0, 2, v3 // (op 3837) 

v_and_b32 v3, 3, v0 // (op 3837) 

v_lshrrev_b32 v0, 6, v1 // (op 3898) 
v_and_b32 v7, 15, v1 // (op 3898) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 3898) 


v_and_b32 v0, 48, v1 // (op 3898) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 3898) 


v_lshrrev_b32 v0, 6, v1 // (op 4008) 
v_and_b32 v8, 15, v1 // (op 4008) 
v_lshl_add_u32 v9, v0, 4, v8 // (op 4008) 


v_and_b32 v0, 63, v1 // (op 4008) 
v_lshrrev_b32 v8, 4, v0 // (op 4008) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 4008) 


v_lshrrev_b32 v8, 6, v1 // (op 4024) 
v_and_b32 v9, 63, v1 // (op 4024) 
v_lshrrev_b32 v10, 6, v1 // (op 4070) 
v_and_b32 v11, 15, v1 // (op 4070) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 4070) 


v_and_b32 v10, 48, v1 // (op 4070) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 4070) 


v_lshrrev_b32 v10, 6, v1 // (op 4138) 
v_and_b32 v12, 15, v1 // (op 4138) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 4138) 


v_and_b32 v10, 63, v1 // (op 4138) 
v_lshrrev_b32 v12, 4, v10 // (op 4138) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 4138) 


s_lshl_b32 s50, 1, s19 // (op 3837) 
v_and_b32 v12, 3, v1 // (op 3837) 
s_add_i32 s51, s50, s49 // (op 3837) 


s_and_b32 s49, s0, s51 // (op 3837) 


s_add_i32 s0, s1, s16 // (op 4024) 

s_ashr_i32 s1, s0, 31 // (op 4024) 
v_lshl_add_u32 v13, v3, 2, v12 // (op 3837) 


s_add_i32 s50, s18, s49 // (op 3837) 


s_ashr_i32 s18, s50, s19 // (op 3837) 


s_lshl_b32 s19, s13, 24 // (op 3837) 
s_ashr_i32 s49, s19, 31 // (op 3837) 

s_xor_b32 s19, s18, s49 // (op 3837) 

s_sub_i32 s18, s19, s49 // (op 3837) 


s_mul_hi_i32 s19, s18, s14 // (op 3837) 
s_add_i32 s49, s19, s18 // (op 3837) 

s_ashr_i32 s19, s49, 31 // (op 3837) 
s_cmp_eq_i32 s14, 0 // (op 3837) 
s_mov_b32 s50, scc // (op 3837) 
s_cmp_eq_i32 s50, 0 // (op 3837) 
s_cselect_b32 s51, 0, -1 // (op 3837) 

s_and_b32 s50, s15, 31 // (op 3837) 
v_lshrrev_b32 v3, 4, v9 // (op 4024) 
s_cmp_eq_i32 s10, 0 // (op 4024) 
s_mov_b32 s52, scc // (op 4024) 
v_lshlrev_b32 v12, 2, v3 // (op 4024) 

s_cmp_eq_i32 s52, 0 // (op 4024) 
s_cselect_b32 s53, 0, -1 // (op 4024) 

s_lshl_b32 s52, 1, s50 // (op 3837) 
v_lshl_add_u32 v3, v8, 4, v12 // (op 4024) 


v_lshrrev_b32 v8, 2, v9 // (op 4024) 

s_add_i32 s54, s52, s51 // (op 3837) 


s_and_b32 s51, s19, s54 // (op 3837) 


s_add_i32 s19, s49, s51 // (op 3837) 


s_ashr_i32 s49, s19, s50 // (op 3837) 


s_lshl_b32 s19, s15, 24 // (op 3837) 
s_ashr_i32 s50, s19, 31 // (op 3837) 

s_xor_b32 s19, s49, s50 // (op 3837) 

s_sub_i32 s49, s19, s50 // (op 3837) 


s_mul_i32 s19, s18, s11 // (op 3837) 
s_sub_u32 s50, s16, s19 // (op 3837) 

s_mul_i32 s19, s49, s11 // (op 3837) 
s_add_u32 s19, s19, s50 // (op 3837) 

s_mul_hi_i32 s50, s16, s9 // (op 3837) 
s_add_i32 s51, s50, s16 // (op 3837) 

s_ashr_i32 s50, s51, 31 // (op 3837) 
s_cmp_eq_i32 s9, 0 // (op 3837) 
s_mov_b32 s52, scc // (op 3837) 
s_cmp_eq_i32 s52, 0 // (op 3837) 
s_cselect_b32 s54, 0, -1 // (op 3837) 

s_and_b32 s52, s13, 31 // (op 4024) 
v_and_b32 v9, 3, v8 // (op 4024) 

s_lshl_b32 s55, 1, s52 // (op 4024) 
v_and_b32 v8, 3, v1 // (op 4024) 
s_add_i32 s56, s55, s53 // (op 4024) 


s_and_b32 s53, s1, s56 // (op 4024) 


v_lshl_add_u32 v12, v9, 2, v8 // (op 4024) 


s_add_i32 s1, s0, s53 // (op 4024) 


s_ashr_i32 s0, s1, s52 // (op 4024) 


s_lshl_b32 s1, s13, 24 // (op 4024) 
s_ashr_i32 s52, s1, 31 // (op 4024) 

s_xor_b32 s1, s0, s52 // (op 4024) 

s_sub_i32 s0, s1, s52 // (op 4024) 


s_mul_hi_i32 s1, s0, s14 // (op 4024) 
s_add_i32 s52, s1, s0 // (op 4024) 

s_ashr_i32 s1, s52, 31 // (op 4024) 
s_cmp_eq_i32 s14, 0 // (op 4024) 
s_mov_b32 s53, scc // (op 4024) 
s_cmp_eq_i32 s53, 0 // (op 4024) 
s_cselect_b32 s55, 0, -1 // (op 4024) 

s_and_b32 s53, s15, 31 // (op 4024) 
s_lshl_b32 s56, 1, s53 // (op 4024) 
s_add_i32 s57, s56, s55 // (op 4024) 


s_and_b32 s55, s1, s57 // (op 4024) 


s_add_i32 s1, s52, s55 // (op 4024) 


s_ashr_i32 s52, s1, s53 // (op 4024) 


s_lshl_b32 s1, s15, 24 // (op 4024) 
s_ashr_i32 s53, s1, 31 // (op 4024) 

s_xor_b32 s1, s52, s53 // (op 4024) 

s_sub_i32 s52, s1, s53 // (op 4024) 


s_mul_i32 s1, s0, s11 // (op 4024) 
s_sub_u32 s53, s16, s1 // (op 4024) 

s_mul_i32 s1, s52, s11 // (op 4024) 
s_add_u32 s1, s1, s53 // (op 4024) 

s_mul_hi_i32 s53, s16, s9 // (op 4024) 
s_add_i32 s55, s53, s16 // (op 4024) 

s_ashr_i32 s53, s55, 31 // (op 4024) 
s_cmp_eq_i32 s9, 0 // (op 4024) 
s_mov_b32 s56, scc // (op 4024) 
s_cmp_eq_i32 s56, 0 // (op 4024) 
s_cselect_b32 s57, 0, -1 // (op 4024) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 3837) 
s_lshl_b32 s58, 1, s56 // (op 3837) 
s_add_i32 s59, s58, s54 // (op 3837) 


s_and_b32 s54, s50, s59 // (op 3837) 


s_add_i32 s50, s51, s54 // (op 3837) 


s_ashr_i32 s51, s50, s56 // (op 3837) 


s_lshl_b32 s50, s45, 24 // (op 3837) 
s_ashr_i32 s54, s50, 31 // (op 3837) 

s_xor_b32 s50, s51, s54 // (op 3837) 

s_sub_i32 s51, s50, s54 // (op 3837) 


s_and_b32 s50, s45, 31 // (op 4024) 
s_lshl_b32 s54, 1, s50 // (op 4024) 
s_add_i32 s56, s54, s57 // (op 4024) 


s_and_b32 s54, s53, s56 // (op 4024) 


s_add_i32 s53, s55, s54 // (op 4024) 


s_ashr_i32 s54, s53, s50 // (op 4024) 


s_lshl_b32 s50, s45, 24 // (op 4024) 
s_ashr_i32 s53, s50, 31 // (op 4024) 

s_xor_b32 s50, s54, s53 // (op 4024) 

s_sub_i32 s54, s50, s53 // (op 4024) 


s_add_u32 s56, s4, 16 // (op 3837) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 3837) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3837) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3837) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 3837) 

s_mul_i32 s50, s11, s56 // (op 3837) 
s_mul_i32 s53, s8, s50 // (op 3837) 

s_mul_i32 s50, s51, s53 // (op 3837) 

s_sub_u32 s53, s16, s50 // (op 3837) 

s_mov_b32 s50, s53 // (op 3837) 
s_mul_hi_i32 s55, s50, s12 // (op 3837) 
s_add_i32 s58, s55, s50 // (op 3837) 


s_ashr_i32 s50, s58, 31 // (op 3837) 
s_cmp_eq_i32 s12, 0 // (op 3837) 
s_mov_b32 s55, scc // (op 3837) 
s_cmp_eq_i32 s55, 0 // (op 3837) 
s_cselect_b32 s59, 0, -1 // (op 3837) 

s_and_b32 s55, s44, 31 // (op 3837) 
s_lshl_b32 s60, 1, s55 // (op 3837) 
s_add_i32 s61, s60, s59 // (op 3837) 


s_and_b32 s59, s50, s61 // (op 3837) 


s_add_i32 s50, s58, s59 // (op 3837) 


s_ashr_i32 s58, s50, s55 // (op 3837) 


s_lshl_b32 s50, s44, 24 // (op 3837) 
s_ashr_i32 s55, s50, 31 // (op 3837) 

s_xor_b32 s50, s58, s55 // (op 3837) 

s_sub_i32 s58, s50, s55 // (op 3837) 


s_mul_i32 s50, s46, s11 // (op 3837) 
s_add_u32 s60, s6, 256 // (op 3837) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 3837) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 3837) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 3837) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 3837) 

s_sub_i32 s55, s60, s50 // (op 3837) 

s_mul_i32 s59, s58, s55 // (op 3837) 

s_sub_u32 s55, s53, s59 // (op 3837) 


s_add_u32 s53, s55, s50 // (op 3837) 


s_cmp_eq_i32 s51, 0 // (op 3837) 
s_mov_b32 s50, scc // (op 3837) 

s_cmp_eq_i32 s50, 0 // (op 3837) 
s_cselect_b32 s51, s53, s19 // (op 3837) 


s_mul_i32 s19, s49, s56 // (op 3837) 


v_lshl_add_u32 v8, s51, 8, v6 // (op 3837) 


s_sub_i32 s49, s18, s19 // (op 3837) 


s_cmp_eq_i32 s50, 0 // (op 3837) 
s_cselect_b32 s51, s58, s49 // (op 3837) 



v_lshl_add_u32 v6, s51, 4, v13 // (op 3837) 


v_mul_lo_u32 v9, v6, s2 // (op 3837) 

v_add_lshl_u32 v6, v8, v9, 1 // (op 3837) 


s_add_u32 s18, s4, 16 // (op 4024) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 4024) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 4024) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 4024) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 4024) 

s_mul_i32 s53, s11, s18 // (op 4024) 
s_mul_i32 s49, s8, s53 // (op 4024) 

s_mul_i32 s53, s54, s49 // (op 4024) 

s_sub_u32 s49, s16, s53 // (op 4024) 

s_mov_b32 s53, s49 // (op 4024) 
s_mul_hi_i32 s50, s53, s12 // (op 4024) 
s_add_i32 s51, s50, s53 // (op 4024) 


s_ashr_i32 s50, s51, 31 // (op 4024) 
s_cmp_eq_i32 s12, 0 // (op 4024) 
s_mov_b32 s53, scc // (op 4024) 
s_cmp_eq_i32 s53, 0 // (op 4024) 
s_cselect_b32 s55, 0, -1 // (op 4024) 

s_and_b32 s53, s44, 31 // (op 4024) 
s_lshl_b32 s56, 1, s53 // (op 4024) 
s_add_i32 s57, s56, s55 // (op 4024) 


s_and_b32 s55, s50, s57 // (op 4024) 


s_add_i32 s50, s51, s55 // (op 4024) 


s_ashr_i32 s51, s50, s53 // (op 4024) 


s_lshl_b32 s50, s44, 24 // (op 4024) 
s_ashr_i32 s53, s50, 31 // (op 4024) 

s_xor_b32 s50, s51, s53 // (op 4024) 

s_sub_i32 s51, s50, s53 // (op 4024) 


s_mul_i32 s50, s46, s11 // (op 4024) 
s_add_u32 s56, s6, 256 // (op 4024) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 4024) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4024) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4024) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 4024) 

s_sub_i32 s53, s56, s50 // (op 4024) 

s_mul_i32 s55, s51, s53 // (op 4024) 

s_sub_u32 s53, s49, s55 // (op 4024) 


s_add_u32 s49, s53, s50 // (op 4024) 


s_cmp_eq_i32 s54, 0 // (op 4024) 
s_mov_b32 s50, scc // (op 4024) 

s_cmp_eq_i32 s50, 0 // (op 4024) 
s_cselect_b32 s53, s49, s1 // (op 4024) 


s_mul_i32 s1, s52, s18 // (op 4024) 


v_lshl_add_u32 v8, s53, 8, v3 // (op 4024) 


s_sub_i32 s49, s0, s1 // (op 4024) 


s_cmp_eq_i32 s50, 0 // (op 4024) 
s_cselect_b32 s0, s51, s49 // (op 4024) 



v_lshl_add_u32 v3, s0, 4, v12 // (op 4024) 


v_mul_lo_u32 v9, v3, s20 // (op 4024) 

v_add_lshl_u32 v3, v8, v9, 1 // (op 4024) 


s_mul_i32 s1, s2, s5 // (op 3839) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 3839) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 3839) least significant
s_mul_hi_u32 s19, s2, s4 // (op 3839) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3839) most significant: sum
s_add_u32 s1, s1, s19 // (op 3839) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3839) 

s_mul_i32 s1, s20, s5 // (op 4026) most significant: low of low * high
s_mul_i32 s49, s21, s4 // (op 4026) most significant: low of high * low
s_mul_i32 s0, s20, s4 // (op 4026) least significant
s_mul_hi_u32 s50, s20, s4 // (op 4026) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 4026) most significant: sum
s_add_u32 s1, s1, s50 // (op 4026) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 4026) 

s_mov_b32 s52, s22 // (op 3839) reg expression
s_mov_b32 s53, s23 // (op 3839) reg expression
s_mov_b32 s54, s18 // (op 3839) reg expression
s_mov_b32 s55, 131072 // (op 3839) call()

s_mov_b32 s56, s24 // (op 4026) reg expression

s_mov_b32 s57, s25 // (op 4026) reg expression
s_mov_b32 s58, s50 // (op 4026) reg expression
s_mov_b32 s59, 131072 // (op 4026) call()


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
s_mul_hi_i32 s0, s16, s10 // (op 3604) 
v_lshrrev_b32 v12, 3, v1 // (op 3604) 
s_add_i32 s1, s0, s16 // (op 3604) 

v_and_b32 v13, -8, v1 // (op 3604) 
s_ashr_i32 s0, s1, 31 // (op 3604) 
v_sub_u32 v14, v1, v13 // (op 3604) 

s_cmp_eq_i32 s10, 0 // (op 3604) 
s_mov_b32 s18, scc // (op 3604) 
v_lshlrev_b32 v13, 4, v14 // (op 3604) 

s_cmp_eq_i32 s18, 0 // (op 3604) 
s_cselect_b32 s19, 0, -1 // (op 3604) 

v_mov_b32 v15, 0 // (op 3604) convert
v_mov_b32 v14, v13 // (op 3604) convert

s_and_b32 s18, s13, 31 // (op 3604) 
s_lshl_b32 s22, 1, s18 // (op 3604) 
s_add_i32 s23, s22, s19 // (op 3604) 


s_and_b32 s19, s0, s23 // (op 3604) 


s_add_i32 s0, s1, s19 // (op 3604) 


s_ashr_i32 s1, s0, s18 // (op 3604) 


s_lshl_b32 s0, s13, 24 // (op 3604) 
s_ashr_i32 s18, s0, 31 // (op 3604) 

s_xor_b32 s0, s1, s18 // (op 3604) 

s_sub_i32 s1, s0, s18 // (op 3604) 


s_mul_hi_i32 s0, s1, s14 // (op 3604) 
s_add_i32 s18, s0, s1 // (op 3604) 

s_ashr_i32 s0, s18, 31 // (op 3604) 
s_cmp_eq_i32 s14, 0 // (op 3604) 
s_mov_b32 s19, scc // (op 3604) 
s_cmp_eq_i32 s19, 0 // (op 3604) 
s_cselect_b32 s22, 0, -1 // (op 3604) 

s_and_b32 s19, s15, 31 // (op 3604) 
s_lshl_b32 s23, 1, s19 // (op 3604) 
s_add_i32 s24, s23, s22 // (op 3604) 


s_and_b32 s25, s0, s24 // (op 3604) 


s_add_i32 s0, s18, s25 // (op 3604) 


s_ashr_i32 s18, s0, s19 // (op 3604) 


s_lshl_b32 s0, s15, 24 // (op 3604) 
s_ashr_i32 s19, s0, 31 // (op 3604) 

s_xor_b32 s0, s18, s19 // (op 3604) 

s_sub_i32 s18, s0, s19 // (op 3604) 


s_mul_i32 s0, s1, s11 // (op 3604) 

s_sub_u32 s1, s16, s0 // (op 3604) 

s_mul_i32 s0, s18, s11 // (op 3604) 
s_add_u32 s0, s0, s1 // (op 3604) 


s_mul_hi_i32 s1, s16, s9 // (op 3604) 
s_add_i32 s18, s1, s16 // (op 3604) 

s_ashr_i32 s1, s18, 31 // (op 3604) 
s_cmp_eq_i32 s9, 0 // (op 3604) 
s_mov_b32 s19, scc // (op 3604) 
s_cmp_eq_i32 s19, 0 // (op 3604) 
s_cselect_b32 s22, 0, -1 // (op 3604) 

s_and_b32 s19, s45, 31 // (op 3604) 
s_lshl_b32 s23, 1, s19 // (op 3604) 
s_add_i32 s24, s23, s22 // (op 3604) 


s_and_b32 s25, s1, s24 // (op 3604) 


s_add_i32 s1, s18, s25 // (op 3604) 


s_ashr_i32 s18, s1, s19 // (op 3604) 


s_lshl_b32 s1, s45, 24 // (op 3604) 
s_ashr_i32 s19, s1, 31 // (op 3604) 

s_xor_b32 s1, s18, s19 // (op 3604) 

s_sub_i32 s18, s1, s19 // (op 3604) 


s_add_u32 s22, s4, 16 // (op 3604) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3604) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3604) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3604) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3604) 

s_mul_i32 s1, s11, s22 // (op 3604) 

s_mul_i32 s19, s8, s1 // (op 3604) 

s_mul_i32 s1, s18, s19 // (op 3604) 

s_sub_u32 s19, s16, s1 // (op 3604) 

s_mov_b32 s1, s19 // (op 3604) 
s_mul_hi_i32 s22, s1, s12 // (op 3604) 
s_add_i32 s23, s22, s1 // (op 3604) 


s_ashr_i32 s1, s23, 31 // (op 3604) 
s_cmp_eq_i32 s12, 0 // (op 3604) 
s_mov_b32 s22, scc // (op 3604) 
s_cmp_eq_i32 s22, 0 // (op 3604) 
s_cselect_b32 s24, 0, -1 // (op 3604) 

s_and_b32 s22, s44, 31 // (op 3604) 
s_lshl_b32 s25, 1, s22 // (op 3604) 
s_add_i32 s47, s25, s24 // (op 3604) 


s_and_b32 s24, s1, s47 // (op 3604) 


s_add_i32 s1, s23, s24 // (op 3604) 


s_ashr_i32 s23, s1, s22 // (op 3604) 


s_lshl_b32 s1, s44, 24 // (op 3604) 
s_ashr_i32 s22, s1, 31 // (op 3604) 

s_xor_b32 s1, s23, s22 // (op 3604) 

s_sub_i32 s23, s1, s22 // (op 3604) 


s_mul_i32 s1, s46, s11 // (op 3604) 
s_add_u32 s24, s6, 256 // (op 3604) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3604) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3604) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3604) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 3604) 

s_sub_i32 s22, s24, s1 // (op 3604) 

s_mul_i32 s24, s23, s22 // (op 3604) 


s_sub_u32 s25, s19, s24 // (op 3604) 


s_add_u32 s19, s25, s1 // (op 3604) 


s_cmp_eq_i32 s18, 0 // (op 3604) 
s_mov_b32 s1, scc // (op 3604) 

s_cmp_eq_i32 s1, 0 // (op 3604) 
s_cselect_b32 s18, s19, s0 // (op 3604) 



v_lshl_add_u32 v13, s18, 8, v12 // (op 3604) 


v_mul_lo_u32 v17, v13, s27 // (op 3604) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 3604) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 3604) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3604) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3604) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3604) most significant half



s_mul_i32 s1, s26, s7 // (op 3606) most significant: low of low * high
s_mul_i32 s18, s27, s6 // (op 3606) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 3606) least significant
s_mul_hi_u32 s19, s26, s6 // (op 3606) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3606) most significant: sum
s_add_u32 s1, s1, s19 // (op 3606) most significant: sum


s_mov_b32 s48, s28 // (op 3606) reg expression
s_mov_b32 s49, s29 // (op 3606) reg expression
s_mov_b32 s50, s0 // (op 3606) reg expression
s_mov_b32 s51, 131072 // (op 3606) call()

s_mul_hi_i32 s47, s16, s10 // (op 3623) 
v_lshrrev_b32 v12, 3, v1 // (op 3623) 
s_add_i32 s0, s47, s16 // (op 3623) 

v_and_b32 v13, -8, v1 // (op 3623) 
s_ashr_i32 s1, s0, 31 // (op 3623) 
v_sub_u32 v14, v1, v13 // (op 3623) 

s_cmp_eq_i32 s10, 0 // (op 3623) 
s_mov_b32 s47, scc // (op 3623) 
v_lshlrev_b32 v13, 4, v14 // (op 3623) 

s_cmp_eq_i32 s47, 0 // (op 3623) 
s_cselect_b32 s18, 0, -1 // (op 3623) 

v_mov_b32 v15, 0 // (op 3623) convert
v_mov_b32 v14, v13 // (op 3623) convert

s_and_b32 s19, s13, 31 // (op 3623) 
s_lshl_b32 s47, 1, s19 // (op 3623) 
s_add_i32 s22, s47, s18 // (op 3623) 


s_and_b32 s18, s1, s22 // (op 3623) 


s_add_i32 s1, s0, s18 // (op 3623) 


s_ashr_i32 s0, s1, s19 // (op 3623) 


s_lshl_b32 s1, s13, 24 // (op 3623) 
s_ashr_i32 s47, s1, 31 // (op 3623) 

s_xor_b32 s1, s0, s47 // (op 3623) 

s_sub_i32 s0, s1, s47 // (op 3623) 


s_mul_hi_i32 s1, s0, s14 // (op 3623) 
s_add_i32 s47, s1, s0 // (op 3623) 

s_ashr_i32 s1, s47, 31 // (op 3623) 
s_cmp_eq_i32 s14, 0 // (op 3623) 
s_mov_b32 s18, scc // (op 3623) 
s_cmp_eq_i32 s18, 0 // (op 3623) 
s_cselect_b32 s19, 0, -1 // (op 3623) 

s_and_b32 s18, s15, 31 // (op 3623) 
s_lshl_b32 s22, 1, s18 // (op 3623) 
s_add_i32 s23, s22, s19 // (op 3623) 


s_and_b32 s19, s1, s23 // (op 3623) 


s_add_i32 s1, s47, s19 // (op 3623) 


s_ashr_i32 s19, s1, s18 // (op 3623) 


s_lshl_b32 s1, s15, 24 // (op 3623) 
s_ashr_i32 s18, s1, 31 // (op 3623) 

s_xor_b32 s1, s19, s18 // (op 3623) 

s_sub_i32 s19, s1, s18 // (op 3623) 


s_mul_i32 s1, s0, s11 // (op 3623) 

s_sub_u32 s0, s16, s1 // (op 3623) 

s_mul_i32 s1, s19, s11 // (op 3623) 
s_add_u32 s1, s1, s0 // (op 3623) 


s_mul_hi_i32 s0, s16, s9 // (op 3623) 
s_add_i32 s47, s0, s16 // (op 3623) 

s_ashr_i32 s0, s47, 31 // (op 3623) 
s_cmp_eq_i32 s9, 0 // (op 3623) 
s_mov_b32 s18, scc // (op 3623) 
s_cmp_eq_i32 s18, 0 // (op 3623) 
s_cselect_b32 s19, 0, -1 // (op 3623) 

s_and_b32 s18, s45, 31 // (op 3623) 
s_lshl_b32 s22, 1, s18 // (op 3623) 
s_add_i32 s23, s22, s19 // (op 3623) 


s_and_b32 s19, s0, s23 // (op 3623) 


s_add_i32 s0, s47, s19 // (op 3623) 


s_ashr_i32 s19, s0, s18 // (op 3623) 


s_lshl_b32 s0, s45, 24 // (op 3623) 
s_ashr_i32 s18, s0, 31 // (op 3623) 

s_xor_b32 s0, s19, s18 // (op 3623) 

s_sub_i32 s19, s0, s18 // (op 3623) 


s_add_u32 s22, s4, 16 // (op 3623) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3623) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3623) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3623) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3623) 

s_mul_i32 s0, s11, s22 // (op 3623) 

s_mul_i32 s18, s8, s0 // (op 3623) 

s_mul_i32 s0, s19, s18 // (op 3623) 

s_sub_u32 s18, s16, s0 // (op 3623) 

s_mov_b32 s0, s18 // (op 3623) 
s_mul_hi_i32 s47, s0, s12 // (op 3623) 
s_add_i32 s22, s47, s0 // (op 3623) 


s_ashr_i32 s0, s22, 31 // (op 3623) 
s_cmp_eq_i32 s12, 0 // (op 3623) 
s_mov_b32 s47, scc // (op 3623) 
s_cmp_eq_i32 s47, 0 // (op 3623) 
s_cselect_b32 s23, 0, -1 // (op 3623) 

s_and_b32 s47, s44, 31 // (op 3623) 
s_lshl_b32 s24, 1, s47 // (op 3623) 
s_add_i32 s25, s24, s23 // (op 3623) 


s_and_b32 s23, s0, s25 // (op 3623) 


s_add_i32 s0, s22, s23 // (op 3623) 


s_ashr_i32 s22, s0, s47 // (op 3623) 


s_lshl_b32 s0, s44, 24 // (op 3623) 
s_ashr_i32 s47, s0, 31 // (op 3623) 

s_xor_b32 s0, s22, s47 // (op 3623) 

s_sub_i32 s22, s0, s47 // (op 3623) 


s_mul_i32 s0, s46, s11 // (op 3623) 
s_add_u32 s24, s6, 256 // (op 3623) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3623) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 3623) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 3623) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 3623) 

s_sub_i32 s23, s24, s0 // (op 3623) 

s_mul_i32 s47, s22, s23 // (op 3623) 


s_sub_u32 s22, s18, s47 // (op 3623) 


s_add_u32 s18, s22, s0 // (op 3623) 


s_cmp_eq_i32 s19, 0 // (op 3623) 
s_mov_b32 s0, scc // (op 3623) 

s_cmp_eq_i32 s0, 0 // (op 3623) 
s_cselect_b32 s19, s18, s1 // (op 3623) 



v_lshl_add_u32 v13, s19, 8, v12 // (op 3623) 


v_mul_lo_u32 v19, v13, s27 // (op 3623) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 3623) most significant: high of low * low
v_mul_lo_u32 v18, v13, s26 // (op 3623) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3623) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 3623) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 3623) most significant half



s_mul_i32 s1, s26, s7 // (op 3625) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 3625) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 3625) least significant
s_mul_hi_u32 s18, s26, s6 // (op 3625) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3625) most significant: sum
s_add_u32 s1, s1, s18 // (op 3625) most significant: sum


s_mov_b32 s48, s28 // (op 3625) reg expression
s_mov_b32 s49, s29 // (op 3625) reg expression
s_mov_b32 s50, s0 // (op 3625) reg expression
s_mov_b32 s51, 131072 // (op 3625) call()

s_mul_hi_i32 s47, s16, s10 // (op 3668) 
v_lshrrev_b32 v12, 3, v1 // (op 3668) 
s_add_i32 s0, s47, s16 // (op 3668) 

v_and_b32 v13, -8, v1 // (op 3668) 
s_ashr_i32 s1, s0, 31 // (op 3668) 
v_sub_u32 v14, v1, v13 // (op 3668) 

s_cmp_eq_i32 s10, 0 // (op 3668) 
s_mov_b32 s47, scc // (op 3668) 
v_lshlrev_b32 v13, 4, v14 // (op 3668) 

s_cmp_eq_i32 s47, 0 // (op 3668) 
s_cselect_b32 s18, 0, -1 // (op 3668) 

v_mov_b32 v15, 0 // (op 3668) convert
v_mov_b32 v14, v13 // (op 3668) convert

s_and_b32 s19, s13, 31 // (op 3668) 
s_lshl_b32 s47, 1, s19 // (op 3668) 
s_add_i32 s22, s47, s18 // (op 3668) 


s_and_b32 s18, s1, s22 // (op 3668) 


s_add_i32 s1, s0, s18 // (op 3668) 


s_ashr_i32 s0, s1, s19 // (op 3668) 


s_lshl_b32 s1, s13, 24 // (op 3668) 
s_ashr_i32 s47, s1, 31 // (op 3668) 

s_xor_b32 s1, s0, s47 // (op 3668) 

s_sub_i32 s0, s1, s47 // (op 3668) 


s_mul_hi_i32 s1, s0, s14 // (op 3668) 
s_add_i32 s47, s1, s0 // (op 3668) 

s_ashr_i32 s1, s47, 31 // (op 3668) 
s_cmp_eq_i32 s14, 0 // (op 3668) 
s_mov_b32 s18, scc // (op 3668) 
s_cmp_eq_i32 s18, 0 // (op 3668) 
s_cselect_b32 s19, 0, -1 // (op 3668) 

s_and_b32 s18, s15, 31 // (op 3668) 
s_lshl_b32 s22, 1, s18 // (op 3668) 
s_add_i32 s23, s22, s19 // (op 3668) 


s_and_b32 s19, s1, s23 // (op 3668) 


s_add_i32 s1, s47, s19 // (op 3668) 


s_ashr_i32 s19, s1, s18 // (op 3668) 


s_lshl_b32 s1, s15, 24 // (op 3668) 
s_ashr_i32 s18, s1, 31 // (op 3668) 

s_xor_b32 s1, s19, s18 // (op 3668) 

s_sub_i32 s19, s1, s18 // (op 3668) 


s_add_u32 s22, s4, 16 // (op 3668) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3668) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3668) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3668) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3668) 

s_mul_i32 s1, s19, s22 // (op 3668) 

s_sub_i32 s47, s0, s1 // (op 3668) 


s_mul_hi_i32 s0, s16, s9 // (op 3668) 
s_add_i32 s1, s0, s16 // (op 3668) 

s_ashr_i32 s0, s1, 31 // (op 3668) 
s_cmp_eq_i32 s9, 0 // (op 3668) 
s_mov_b32 s18, scc // (op 3668) 
s_cmp_eq_i32 s18, 0 // (op 3668) 
s_cselect_b32 s19, 0, -1 // (op 3668) 

s_and_b32 s18, s45, 31 // (op 3668) 
s_lshl_b32 s24, 1, s18 // (op 3668) 
s_add_i32 s25, s24, s19 // (op 3668) 


s_and_b32 s19, s0, s25 // (op 3668) 


s_add_i32 s0, s1, s19 // (op 3668) 


s_ashr_i32 s1, s0, s18 // (op 3668) 


s_lshl_b32 s0, s45, 24 // (op 3668) 
s_ashr_i32 s18, s0, 31 // (op 3668) 

s_xor_b32 s0, s1, s18 // (op 3668) 

s_sub_i32 s1, s0, s18 // (op 3668) 


s_mul_i32 s0, s11, s22 // (op 3668) 

s_mul_i32 s18, s8, s0 // (op 3668) 

s_mul_i32 s0, s1, s18 // (op 3668) 

s_sub_u32 s18, s16, s0 // (op 3668) 

s_mul_hi_i32 s0, s18, s12 // (op 3668) 
s_add_i32 s19, s0, s18 // (op 3668) 


s_ashr_i32 s0, s19, 31 // (op 3668) 
s_cmp_eq_i32 s12, 0 // (op 3668) 
s_mov_b32 s18, scc // (op 3668) 
s_cmp_eq_i32 s18, 0 // (op 3668) 
s_cselect_b32 s22, 0, -1 // (op 3668) 

s_and_b32 s18, s44, 31 // (op 3668) 
s_lshl_b32 s23, 1, s18 // (op 3668) 
s_add_i32 s24, s23, s22 // (op 3668) 


s_and_b32 s25, s0, s24 // (op 3668) 


s_add_i32 s0, s19, s25 // (op 3668) 


s_ashr_i32 s19, s0, s18 // (op 3668) 


s_lshl_b32 s0, s44, 24 // (op 3668) 
s_ashr_i32 s18, s0, 31 // (op 3668) 

s_xor_b32 s0, s19, s18 // (op 3668) 

s_sub_i32 s19, s0, s18 // (op 3668) 


s_cmp_eq_i32 s1, 0 // (op 3668) 
s_mov_b32 s0, scc // (op 3668) 

s_cmp_eq_i32 s0, 0 // (op 3668) 
s_cselect_b32 s1, s19, s47 // (op 3668) 



v_lshl_add_u32 v13, s1, 4, v12 // (op 3668) 


v_mul_lo_u32 v21, v13, s31 // (op 3668) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 3668) most significant: high of low * low
v_mul_lo_u32 v20, v13, s30 // (op 3668) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 3668) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 3668) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 3668) most significant half



s_mul_i32 s1, s30, s5 // (op 3670) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 3670) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 3670) least significant
s_mul_hi_u32 s18, s30, s4 // (op 3670) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3670) most significant: sum
s_add_u32 s1, s1, s18 // (op 3670) most significant: sum


s_mov_b32 s60, s32 // (op 3670) reg expression
s_mov_b32 s61, s33 // (op 3670) reg expression
s_mov_b32 s62, s0 // (op 3670) reg expression
s_mov_b32 s63, 131072 // (op 3670) call()

s_mul_hi_i32 s47, s16, s10 // (op 3687) 
v_lshrrev_b32 v14, 3, v1 // (op 3687) 
s_add_i32 s0, s47, s16 // (op 3687) 

v_and_b32 v15, -8, v1 // (op 3687) 
s_ashr_i32 s1, s0, 31 // (op 3687) 
v_sub_u32 v20, v1, v15 // (op 3687) 

s_cmp_eq_i32 s10, 0 // (op 3687) 
s_mov_b32 s47, scc // (op 3687) 
v_lshlrev_b32 v15, 4, v20 // (op 3687) 

s_cmp_eq_i32 s47, 0 // (op 3687) 
s_cselect_b32 s18, 0, -1 // (op 3687) 

v_mov_b32 v21, 0 // (op 3687) convert
v_mov_b32 v20, v15 // (op 3687) convert

s_and_b32 s19, s13, 31 // (op 3687) 
s_lshl_b32 s47, 1, s19 // (op 3687) 
s_add_i32 s22, s47, s18 // (op 3687) 


s_and_b32 s18, s1, s22 // (op 3687) 


s_add_i32 s1, s0, s18 // (op 3687) 


s_ashr_i32 s0, s1, s19 // (op 3687) 


s_lshl_b32 s1, s13, 24 // (op 3687) 
s_ashr_i32 s47, s1, 31 // (op 3687) 

s_xor_b32 s1, s0, s47 // (op 3687) 

s_sub_i32 s0, s1, s47 // (op 3687) 


s_mul_hi_i32 s1, s0, s14 // (op 3687) 
s_add_i32 s47, s1, s0 // (op 3687) 

s_ashr_i32 s1, s47, 31 // (op 3687) 
s_cmp_eq_i32 s14, 0 // (op 3687) 
s_mov_b32 s18, scc // (op 3687) 
s_cmp_eq_i32 s18, 0 // (op 3687) 
s_cselect_b32 s19, 0, -1 // (op 3687) 

s_and_b32 s18, s15, 31 // (op 3687) 
s_lshl_b32 s22, 1, s18 // (op 3687) 
s_add_i32 s23, s22, s19 // (op 3687) 


s_and_b32 s19, s1, s23 // (op 3687) 


s_add_i32 s1, s47, s19 // (op 3687) 


s_ashr_i32 s19, s1, s18 // (op 3687) 


s_lshl_b32 s1, s15, 24 // (op 3687) 
s_ashr_i32 s18, s1, 31 // (op 3687) 

s_xor_b32 s1, s19, s18 // (op 3687) 

s_sub_i32 s19, s1, s18 // (op 3687) 


s_add_u32 s22, s4, 16 // (op 3687) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3687) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3687) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3687) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3687) 

s_mul_i32 s1, s19, s22 // (op 3687) 

s_sub_i32 s47, s0, s1 // (op 3687) 


s_mul_hi_i32 s0, s16, s9 // (op 3687) 
s_add_i32 s1, s0, s16 // (op 3687) 

s_ashr_i32 s0, s1, 31 // (op 3687) 
s_cmp_eq_i32 s9, 0 // (op 3687) 
s_mov_b32 s18, scc // (op 3687) 
s_cmp_eq_i32 s18, 0 // (op 3687) 
s_cselect_b32 s19, 0, -1 // (op 3687) 

s_and_b32 s18, s45, 31 // (op 3687) 
s_lshl_b32 s24, 1, s18 // (op 3687) 
s_add_i32 s25, s24, s19 // (op 3687) 


s_and_b32 s19, s0, s25 // (op 3687) 


s_add_i32 s0, s1, s19 // (op 3687) 


s_ashr_i32 s1, s0, s18 // (op 3687) 


s_lshl_b32 s0, s45, 24 // (op 3687) 
s_ashr_i32 s18, s0, 31 // (op 3687) 

s_xor_b32 s0, s1, s18 // (op 3687) 

s_sub_i32 s1, s0, s18 // (op 3687) 


s_mul_i32 s0, s11, s22 // (op 3687) 

s_mul_i32 s18, s8, s0 // (op 3687) 

s_mul_i32 s0, s1, s18 // (op 3687) 

s_sub_u32 s18, s16, s0 // (op 3687) 

s_mul_hi_i32 s0, s18, s12 // (op 3687) 
s_add_i32 s19, s0, s18 // (op 3687) 


s_ashr_i32 s0, s19, 31 // (op 3687) 
s_cmp_eq_i32 s12, 0 // (op 3687) 
s_mov_b32 s18, scc // (op 3687) 
s_cmp_eq_i32 s18, 0 // (op 3687) 
s_cselect_b32 s22, 0, -1 // (op 3687) 

s_and_b32 s18, s44, 31 // (op 3687) 
s_lshl_b32 s23, 1, s18 // (op 3687) 
s_add_i32 s24, s23, s22 // (op 3687) 


s_and_b32 s25, s0, s24 // (op 3687) 


s_add_i32 s0, s19, s25 // (op 3687) 


s_ashr_i32 s19, s0, s18 // (op 3687) 


s_lshl_b32 s0, s44, 24 // (op 3687) 
s_ashr_i32 s18, s0, 31 // (op 3687) 

s_xor_b32 s0, s19, s18 // (op 3687) 

s_sub_i32 s19, s0, s18 // (op 3687) 


s_cmp_eq_i32 s1, 0 // (op 3687) 
s_mov_b32 s0, scc // (op 3687) 

s_cmp_eq_i32 s0, 0 // (op 3687) 
s_cselect_b32 s1, s19, s47 // (op 3687) 



v_lshl_add_u32 v15, s1, 4, v14 // (op 3687) 


v_mul_lo_u32 v23, v15, s31 // (op 3687) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 3687) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 3687) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 3687) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 3687) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 3687) most significant half



s_mul_i32 s1, s30, s5 // (op 3689) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 3689) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 3689) least significant
s_mul_hi_u32 s18, s30, s4 // (op 3689) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3689) most significant: sum
s_add_u32 s1, s1, s18 // (op 3689) most significant: sum


s_mov_b32 s60, s32 // (op 3689) reg expression
s_mov_b32 s61, s33 // (op 3689) reg expression
s_mov_b32 s62, s0 // (op 3689) reg expression
s_mov_b32 s63, 131072 // (op 3689) call()

s_mul_hi_i32 s47, s16, s10 // (op 3732) 
v_lshlrev_b32 v20, 1, v1 // (op 3732) 
s_add_i32 s0, s47, s16 // (op 3732) 

s_ashr_i32 s1, s0, 31 // (op 3732) 
s_cmp_eq_i32 s10, 0 // (op 3732) 
s_mov_b32 s47, scc // (op 3732) 
s_cmp_eq_i32 s47, 0 // (op 3732) 
s_cselect_b32 s18, 0, -1 // (op 3732) 

s_and_b32 s19, s13, 31 // (op 3732) 
s_lshl_b32 s47, 1, s19 // (op 3732) 
s_add_i32 s22, s47, s18 // (op 3732) 


s_and_b32 s18, s1, s22 // (op 3732) 


s_add_i32 s1, s0, s18 // (op 3732) 


s_ashr_i32 s0, s1, s19 // (op 3732) 


s_lshl_b32 s1, s13, 24 // (op 3732) 
s_ashr_i32 s47, s1, 31 // (op 3732) 

s_xor_b32 s1, s0, s47 // (op 3732) 

s_sub_i32 s0, s1, s47 // (op 3732) 


s_mul_hi_i32 s1, s0, s14 // (op 3732) 
s_add_i32 s47, s1, s0 // (op 3732) 

s_ashr_i32 s1, s47, 31 // (op 3732) 
s_cmp_eq_i32 s14, 0 // (op 3732) 
s_mov_b32 s18, scc // (op 3732) 
s_cmp_eq_i32 s18, 0 // (op 3732) 
s_cselect_b32 s19, 0, -1 // (op 3732) 

s_and_b32 s18, s15, 31 // (op 3732) 
s_lshl_b32 s22, 1, s18 // (op 3732) 
s_add_i32 s23, s22, s19 // (op 3732) 


s_and_b32 s19, s1, s23 // (op 3732) 


s_add_i32 s1, s47, s19 // (op 3732) 


s_ashr_i32 s19, s1, s18 // (op 3732) 


s_lshl_b32 s1, s15, 24 // (op 3732) 
s_ashr_i32 s18, s1, 31 // (op 3732) 

s_xor_b32 s1, s19, s18 // (op 3732) 

s_sub_i32 s19, s1, s18 // (op 3732) 


s_mul_i32 s1, s0, s11 // (op 3732) 

s_sub_u32 s0, s16, s1 // (op 3732) 

s_mul_i32 s1, s19, s11 // (op 3732) 
s_add_u32 s1, s1, s0 // (op 3732) 


s_mul_hi_i32 s0, s16, s9 // (op 3732) 
s_add_i32 s47, s0, s16 // (op 3732) 

s_ashr_i32 s0, s47, 31 // (op 3732) 
s_cmp_eq_i32 s9, 0 // (op 3732) 
s_mov_b32 s18, scc // (op 3732) 
s_cmp_eq_i32 s18, 0 // (op 3732) 
s_cselect_b32 s19, 0, -1 // (op 3732) 

s_and_b32 s18, s45, 31 // (op 3732) 
s_lshl_b32 s22, 1, s18 // (op 3732) 
s_add_i32 s23, s22, s19 // (op 3732) 


s_and_b32 s19, s0, s23 // (op 3732) 


s_add_i32 s0, s47, s19 // (op 3732) 


s_ashr_i32 s19, s0, s18 // (op 3732) 


s_lshl_b32 s0, s45, 24 // (op 3732) 
s_ashr_i32 s18, s0, 31 // (op 3732) 

s_xor_b32 s0, s19, s18 // (op 3732) 

s_sub_i32 s19, s0, s18 // (op 3732) 


s_add_u32 s22, s4, 16 // (op 3732) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3732) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3732) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3732) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3732) 

s_mul_i32 s0, s11, s22 // (op 3732) 

s_mul_i32 s18, s8, s0 // (op 3732) 

s_mul_i32 s0, s19, s18 // (op 3732) 

s_sub_u32 s18, s16, s0 // (op 3732) 

s_mov_b32 s0, s18 // (op 3732) 
s_mul_hi_i32 s47, s0, s12 // (op 3732) 
s_add_i32 s22, s47, s0 // (op 3732) 


s_ashr_i32 s0, s22, 31 // (op 3732) 
s_cmp_eq_i32 s12, 0 // (op 3732) 
s_mov_b32 s47, scc // (op 3732) 
s_cmp_eq_i32 s47, 0 // (op 3732) 
s_cselect_b32 s23, 0, -1 // (op 3732) 

s_and_b32 s47, s44, 31 // (op 3732) 
s_lshl_b32 s24, 1, s47 // (op 3732) 
s_add_i32 s25, s24, s23 // (op 3732) 


s_and_b32 s23, s0, s25 // (op 3732) 


s_add_i32 s0, s22, s23 // (op 3732) 


s_ashr_i32 s22, s0, s47 // (op 3732) 


s_lshl_b32 s0, s44, 24 // (op 3732) 
s_ashr_i32 s47, s0, 31 // (op 3732) 

s_xor_b32 s0, s22, s47 // (op 3732) 

s_sub_i32 s22, s0, s47 // (op 3732) 


s_mul_i32 s0, s46, s11 // (op 3732) 
s_add_u32 s24, s6, 256 // (op 3732) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3732) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 3732) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 3732) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 3732) 

s_sub_i32 s23, s24, s0 // (op 3732) 

s_mul_i32 s47, s22, s23 // (op 3732) 


s_sub_u32 s22, s18, s47 // (op 3732) 


s_add_u32 s18, s22, s0 // (op 3732) 


s_cmp_eq_i32 s19, 0 // (op 3732) 
s_mov_b32 s0, scc // (op 3732) 

s_cmp_eq_i32 s0, 0 // (op 3732) 
s_cselect_b32 s19, s18, s1 // (op 3732) 



v_lshl_add_u32 v21, s19, 8, v20 // (op 3732) 


v_mul_lo_u32 v20, v21, s34 // (op 3732) 

s_mul_i32 s1, s34, s7 // (op 3734) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 3734) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 3734) least significant
s_mul_hi_u32 s18, s34, s6 // (op 3734) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3734) most significant: sum
s_add_u32 s1, s1, s18 // (op 3734) most significant: sum


s_mov_b32 s64, s36 // (op 3734) reg expression
s_mov_b32 s65, s37 // (op 3734) reg expression
s_mov_b32 s66, s0 // (op 3734) reg expression
s_mov_b32 s67, 131072 // (op 3734) call()

s_mul_hi_i32 s47, s16, s10 // (op 3751) 
v_lshlrev_b32 v21, 1, v1 // (op 3751) 
s_add_i32 s0, s47, s16 // (op 3751) 

s_ashr_i32 s1, s0, 31 // (op 3751) 
s_cmp_eq_i32 s10, 0 // (op 3751) 
s_mov_b32 s47, scc // (op 3751) 
s_cmp_eq_i32 s47, 0 // (op 3751) 
s_cselect_b32 s18, 0, -1 // (op 3751) 

s_and_b32 s19, s13, 31 // (op 3751) 
s_lshl_b32 s47, 1, s19 // (op 3751) 
s_add_i32 s22, s47, s18 // (op 3751) 


s_and_b32 s18, s1, s22 // (op 3751) 


s_add_i32 s1, s0, s18 // (op 3751) 


s_ashr_i32 s0, s1, s19 // (op 3751) 


s_lshl_b32 s1, s13, 24 // (op 3751) 
s_ashr_i32 s47, s1, 31 // (op 3751) 

s_xor_b32 s1, s0, s47 // (op 3751) 

s_sub_i32 s0, s1, s47 // (op 3751) 


s_mul_hi_i32 s1, s0, s14 // (op 3751) 
s_add_i32 s47, s1, s0 // (op 3751) 

s_ashr_i32 s1, s47, 31 // (op 3751) 
s_cmp_eq_i32 s14, 0 // (op 3751) 
s_mov_b32 s18, scc // (op 3751) 
s_cmp_eq_i32 s18, 0 // (op 3751) 
s_cselect_b32 s19, 0, -1 // (op 3751) 

s_and_b32 s18, s15, 31 // (op 3751) 
s_lshl_b32 s22, 1, s18 // (op 3751) 
s_add_i32 s23, s22, s19 // (op 3751) 


s_and_b32 s19, s1, s23 // (op 3751) 


s_add_i32 s1, s47, s19 // (op 3751) 


s_ashr_i32 s19, s1, s18 // (op 3751) 


s_lshl_b32 s1, s15, 24 // (op 3751) 
s_ashr_i32 s18, s1, 31 // (op 3751) 

s_xor_b32 s1, s19, s18 // (op 3751) 

s_sub_i32 s19, s1, s18 // (op 3751) 


s_mul_i32 s1, s0, s11 // (op 3751) 

s_sub_u32 s0, s16, s1 // (op 3751) 

s_mul_i32 s1, s19, s11 // (op 3751) 
s_add_u32 s1, s1, s0 // (op 3751) 


s_mul_hi_i32 s0, s16, s9 // (op 3751) 
s_add_i32 s47, s0, s16 // (op 3751) 

s_ashr_i32 s0, s47, 31 // (op 3751) 
s_cmp_eq_i32 s9, 0 // (op 3751) 
s_mov_b32 s18, scc // (op 3751) 
s_cmp_eq_i32 s18, 0 // (op 3751) 
s_cselect_b32 s19, 0, -1 // (op 3751) 

s_and_b32 s18, s45, 31 // (op 3751) 
s_lshl_b32 s22, 1, s18 // (op 3751) 
s_add_i32 s23, s22, s19 // (op 3751) 


s_and_b32 s19, s0, s23 // (op 3751) 


s_add_i32 s0, s47, s19 // (op 3751) 


s_ashr_i32 s19, s0, s18 // (op 3751) 


s_lshl_b32 s0, s45, 24 // (op 3751) 
s_ashr_i32 s18, s0, 31 // (op 3751) 

s_xor_b32 s0, s19, s18 // (op 3751) 

s_sub_i32 s19, s0, s18 // (op 3751) 


s_add_u32 s22, s4, 16 // (op 3751) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3751) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3751) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3751) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3751) 

s_mul_i32 s0, s11, s22 // (op 3751) 

s_mul_i32 s18, s8, s0 // (op 3751) 

s_mul_i32 s0, s19, s18 // (op 3751) 

s_sub_u32 s18, s16, s0 // (op 3751) 

s_mov_b32 s0, s18 // (op 3751) 
s_mul_hi_i32 s47, s0, s12 // (op 3751) 
s_add_i32 s22, s47, s0 // (op 3751) 


s_ashr_i32 s0, s22, 31 // (op 3751) 
s_cmp_eq_i32 s12, 0 // (op 3751) 
s_mov_b32 s47, scc // (op 3751) 
s_cmp_eq_i32 s47, 0 // (op 3751) 
s_cselect_b32 s23, 0, -1 // (op 3751) 

s_and_b32 s47, s44, 31 // (op 3751) 
s_lshl_b32 s24, 1, s47 // (op 3751) 
s_add_i32 s25, s24, s23 // (op 3751) 


s_and_b32 s23, s0, s25 // (op 3751) 


s_add_i32 s0, s22, s23 // (op 3751) 


s_ashr_i32 s22, s0, s47 // (op 3751) 


s_lshl_b32 s0, s44, 24 // (op 3751) 
s_ashr_i32 s47, s0, 31 // (op 3751) 

s_xor_b32 s0, s22, s47 // (op 3751) 

s_sub_i32 s22, s0, s47 // (op 3751) 


s_mul_i32 s0, s46, s11 // (op 3751) 
s_add_u32 s24, s6, 256 // (op 3751) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3751) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 3751) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 3751) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 3751) 

s_sub_i32 s23, s24, s0 // (op 3751) 

s_mul_i32 s47, s22, s23 // (op 3751) 


s_sub_u32 s22, s18, s47 // (op 3751) 


s_add_u32 s18, s22, s0 // (op 3751) 


s_cmp_eq_i32 s19, 0 // (op 3751) 
s_mov_b32 s0, scc // (op 3751) 

s_cmp_eq_i32 s0, 0 // (op 3751) 
s_cselect_b32 s19, s18, s1 // (op 3751) 



v_lshl_add_u32 v22, s19, 8, v21 // (op 3751) 


v_mul_lo_u32 v21, v22, s34 // (op 3751) 

s_mul_i32 s1, s34, s7 // (op 3753) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 3753) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 3753) least significant
s_mul_hi_u32 s18, s34, s6 // (op 3753) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3753) most significant: sum
s_add_u32 s1, s1, s18 // (op 3753) most significant: sum


s_mov_b32 s64, s36 // (op 3753) reg expression
s_mov_b32 s65, s37 // (op 3753) reg expression
s_mov_b32 s66, s0 // (op 3753) reg expression
s_mov_b32 s67, 131072 // (op 3753) call()

s_mul_hi_i32 s47, s16, s10 // (op 3796) 
v_and_b32 v22, 15, v1 // (op 3796) 
s_add_i32 s0, s47, s16 // (op 3796) 

v_and_b32 v23, 63, v1 // (op 3796) 
s_ashr_i32 s1, s0, 31 // (op 3796) 
v_lshrrev_b32 v24, 4, v23 // (op 3796) 

s_cmp_eq_i32 s10, 0 // (op 3796) 
s_mov_b32 s47, scc // (op 3796) 
s_cmp_eq_i32 s47, 0 // (op 3796) 
s_cselect_b32 s18, 0, -1 // (op 3796) 

s_and_b32 s19, s13, 31 // (op 3796) 
s_lshl_b32 s47, 1, s19 // (op 3796) 
s_add_i32 s22, s47, s18 // (op 3796) 


s_and_b32 s18, s1, s22 // (op 3796) 


s_add_i32 s1, s0, s18 // (op 3796) 


s_ashr_i32 s0, s1, s19 // (op 3796) 


s_lshl_b32 s1, s13, 24 // (op 3796) 
s_ashr_i32 s47, s1, 31 // (op 3796) 

s_xor_b32 s1, s0, s47 // (op 3796) 

s_sub_i32 s0, s1, s47 // (op 3796) 


s_mul_hi_i32 s1, s0, s14 // (op 3796) 
s_add_i32 s47, s1, s0 // (op 3796) 

s_ashr_i32 s1, s47, 31 // (op 3796) 
s_cmp_eq_i32 s14, 0 // (op 3796) 
s_mov_b32 s18, scc // (op 3796) 
s_cmp_eq_i32 s18, 0 // (op 3796) 
s_cselect_b32 s19, 0, -1 // (op 3796) 

s_and_b32 s18, s15, 31 // (op 3796) 
s_lshl_b32 s22, 1, s18 // (op 3796) 
s_add_i32 s23, s22, s19 // (op 3796) 


s_and_b32 s19, s1, s23 // (op 3796) 


s_add_i32 s1, s47, s19 // (op 3796) 


s_ashr_i32 s19, s1, s18 // (op 3796) 


s_lshl_b32 s1, s15, 24 // (op 3796) 
s_ashr_i32 s18, s1, 31 // (op 3796) 

s_xor_b32 s1, s19, s18 // (op 3796) 

s_sub_i32 s19, s1, s18 // (op 3796) 


s_add_u32 s22, s4, 16 // (op 3796) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3796) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3796) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3796) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3796) 

s_mul_i32 s1, s19, s22 // (op 3796) 

s_sub_i32 s47, s0, s1 // (op 3796) 


s_mul_hi_i32 s0, s16, s9 // (op 3796) 
s_add_i32 s1, s0, s16 // (op 3796) 

s_ashr_i32 s0, s1, 31 // (op 3796) 
s_cmp_eq_i32 s9, 0 // (op 3796) 
s_mov_b32 s18, scc // (op 3796) 
s_cmp_eq_i32 s18, 0 // (op 3796) 
s_cselect_b32 s19, 0, -1 // (op 3796) 

s_and_b32 s18, s45, 31 // (op 3796) 
s_lshl_b32 s24, 1, s18 // (op 3796) 
s_add_i32 s25, s24, s19 // (op 3796) 


s_and_b32 s19, s0, s25 // (op 3796) 


s_add_i32 s0, s1, s19 // (op 3796) 


s_ashr_i32 s1, s0, s18 // (op 3796) 


s_lshl_b32 s0, s45, 24 // (op 3796) 
s_ashr_i32 s18, s0, 31 // (op 3796) 

s_xor_b32 s0, s1, s18 // (op 3796) 

s_sub_i32 s1, s0, s18 // (op 3796) 


s_mul_i32 s0, s11, s22 // (op 3796) 

s_mul_i32 s18, s8, s0 // (op 3796) 

s_mul_i32 s0, s1, s18 // (op 3796) 

s_sub_u32 s18, s16, s0 // (op 3796) 

s_mul_hi_i32 s0, s18, s12 // (op 3796) 
s_add_i32 s19, s0, s18 // (op 3796) 


s_ashr_i32 s0, s19, 31 // (op 3796) 
s_cmp_eq_i32 s12, 0 // (op 3796) 
s_mov_b32 s18, scc // (op 3796) 
s_cmp_eq_i32 s18, 0 // (op 3796) 
s_cselect_b32 s22, 0, -1 // (op 3796) 

s_and_b32 s18, s44, 31 // (op 3796) 
s_lshl_b32 s23, 1, s18 // (op 3796) 
s_add_i32 s24, s23, s22 // (op 3796) 


s_and_b32 s25, s0, s24 // (op 3796) 


s_add_i32 s0, s19, s25 // (op 3796) 


s_ashr_i32 s19, s0, s18 // (op 3796) 


s_lshl_b32 s0, s44, 24 // (op 3796) 
s_ashr_i32 s18, s0, 31 // (op 3796) 

s_xor_b32 s0, s19, s18 // (op 3796) 

s_sub_i32 s19, s0, s18 // (op 3796) 


s_cmp_eq_i32 s1, 0 // (op 3796) 
s_mov_b32 s0, scc // (op 3796) 

s_cmp_eq_i32 s0, 0 // (op 3796) 
s_cselect_b32 s1, s19, s47 // (op 3796) 



v_lshl_add_u32 v23, s1, 4, v22 // (op 3796) 


v_mul_lo_u32 v22, v23, s38 // (op 3796) 

v_add_u32 v23, v24, v22 // (op 3796) 


s_mul_i32 s1, s38, s5 // (op 3798) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 3798) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 3798) least significant
s_mul_hi_u32 s18, s38, s4 // (op 3798) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3798) most significant: sum
s_add_u32 s1, s1, s18 // (op 3798) most significant: sum


s_mov_b32 s68, s40 // (op 3798) reg expression
s_mov_b32 s69, s41 // (op 3798) reg expression
s_mov_b32 s70, s0 // (op 3798) reg expression
s_mov_b32 s71, 131072 // (op 3798) call()

s_mul_hi_i32 s47, s16, s10 // (op 3815) 
v_and_b32 v22, 15, v1 // (op 3815) 
s_add_i32 s0, s47, s16 // (op 3815) 

v_and_b32 v24, 63, v1 // (op 3815) 
s_ashr_i32 s1, s0, 31 // (op 3815) 
v_lshrrev_b32 v25, 4, v24 // (op 3815) 

s_cmp_eq_i32 s10, 0 // (op 3815) 
s_mov_b32 s47, scc // (op 3815) 
s_cmp_eq_i32 s47, 0 // (op 3815) 
s_cselect_b32 s18, 0, -1 // (op 3815) 

s_and_b32 s19, s13, 31 // (op 3815) 
s_lshl_b32 s47, 1, s19 // (op 3815) 
s_add_i32 s22, s47, s18 // (op 3815) 


s_and_b32 s18, s1, s22 // (op 3815) 


s_add_i32 s1, s0, s18 // (op 3815) 


s_ashr_i32 s0, s1, s19 // (op 3815) 


s_lshl_b32 s1, s13, 24 // (op 3815) 
s_ashr_i32 s47, s1, 31 // (op 3815) 

s_xor_b32 s1, s0, s47 // (op 3815) 

s_sub_i32 s0, s1, s47 // (op 3815) 


s_mul_hi_i32 s1, s0, s14 // (op 3815) 
s_add_i32 s47, s1, s0 // (op 3815) 

s_ashr_i32 s1, s47, 31 // (op 3815) 
s_cmp_eq_i32 s14, 0 // (op 3815) 
s_mov_b32 s18, scc // (op 3815) 
s_cmp_eq_i32 s18, 0 // (op 3815) 
s_cselect_b32 s19, 0, -1 // (op 3815) 

s_and_b32 s18, s15, 31 // (op 3815) 
s_lshl_b32 s22, 1, s18 // (op 3815) 
s_add_i32 s23, s22, s19 // (op 3815) 


s_and_b32 s19, s1, s23 // (op 3815) 


s_add_i32 s1, s47, s19 // (op 3815) 


s_ashr_i32 s19, s1, s18 // (op 3815) 


s_lshl_b32 s1, s15, 24 // (op 3815) 
s_ashr_i32 s18, s1, 31 // (op 3815) 

s_xor_b32 s1, s19, s18 // (op 3815) 

s_sub_i32 s19, s1, s18 // (op 3815) 


s_add_u32 s22, s4, 16 // (op 3815) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3815) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3815) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3815) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3815) 

s_mul_i32 s1, s19, s22 // (op 3815) 

s_sub_i32 s47, s0, s1 // (op 3815) 


s_mul_hi_i32 s0, s16, s9 // (op 3815) 
s_add_i32 s1, s0, s16 // (op 3815) 

s_ashr_i32 s0, s1, 31 // (op 3815) 
s_cmp_eq_i32 s9, 0 // (op 3815) 
s_mov_b32 s18, scc // (op 3815) 
s_cmp_eq_i32 s18, 0 // (op 3815) 
s_cselect_b32 s19, 0, -1 // (op 3815) 

s_and_b32 s18, s45, 31 // (op 3815) 
s_lshl_b32 s24, 1, s18 // (op 3815) 
s_add_i32 s25, s24, s19 // (op 3815) 


s_and_b32 s19, s0, s25 // (op 3815) 


s_add_i32 s0, s1, s19 // (op 3815) 


s_ashr_i32 s1, s0, s18 // (op 3815) 


s_lshl_b32 s0, s45, 24 // (op 3815) 
s_ashr_i32 s18, s0, 31 // (op 3815) 

s_xor_b32 s0, s1, s18 // (op 3815) 

s_sub_i32 s1, s0, s18 // (op 3815) 


s_mul_i32 s0, s11, s22 // (op 3815) 

s_mul_i32 s18, s8, s0 // (op 3815) 

s_mul_i32 s0, s1, s18 // (op 3815) 

s_sub_u32 s18, s16, s0 // (op 3815) 

s_mul_hi_i32 s0, s18, s12 // (op 3815) 
s_add_i32 s19, s0, s18 // (op 3815) 


s_ashr_i32 s0, s19, 31 // (op 3815) 
s_cmp_eq_i32 s12, 0 // (op 3815) 
s_mov_b32 s18, scc // (op 3815) 
s_cmp_eq_i32 s18, 0 // (op 3815) 
s_cselect_b32 s22, 0, -1 // (op 3815) 

s_and_b32 s18, s44, 31 // (op 3815) 
s_lshl_b32 s23, 1, s18 // (op 3815) 
s_add_i32 s24, s23, s22 // (op 3815) 


s_and_b32 s25, s0, s24 // (op 3815) 


s_add_i32 s0, s19, s25 // (op 3815) 


s_ashr_i32 s19, s0, s18 // (op 3815) 


s_lshl_b32 s0, s44, 24 // (op 3815) 
s_ashr_i32 s18, s0, 31 // (op 3815) 

s_xor_b32 s0, s19, s18 // (op 3815) 

s_sub_i32 s19, s0, s18 // (op 3815) 


s_cmp_eq_i32 s1, 0 // (op 3815) 
s_mov_b32 s0, scc // (op 3815) 

s_cmp_eq_i32 s0, 0 // (op 3815) 
s_cselect_b32 s1, s19, s47 // (op 3815) 



v_lshl_add_u32 v24, s1, 4, v22 // (op 3815) 


v_mul_lo_u32 v22, v24, s38 // (op 3815) 

v_add_u32 v24, v25, v22 // (op 3815) 


s_mul_i32 s1, s38, s5 // (op 3817) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 3817) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 3817) least significant
s_mul_hi_u32 s18, s38, s4 // (op 3817) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3817) most significant: sum
s_add_u32 s1, s1, s18 // (op 3817) most significant: sum


s_mov_b32 s68, s40 // (op 3817) reg expression
s_mov_b32 s69, s41 // (op 3817) reg expression
s_mov_b32 s70, s0 // (op 3817) reg expression
s_mov_b32 s71, 131072 // (op 3817) call()




buffer_load_ubyte v22, v23, s[68:71], 0 offen offset: 0 // (op 3284) Load value
v_and_b32 v25, -8, v1 // (op 3938) 
v_sub_u32 v26, v1, v25 // (op 3938) 

v_lshlrev_b32 v25, 4, v26 // (op 3938) 

v_lshrrev_b32 v26, 3, v1 // (op 3938) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3938) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 3938) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s47 // (op 3546) reg expression
buffer_load_dwordx4 v12, s[60:63], 0 offen offset: 0 lds // (op 3546) Load value direct to lds

v_and_b32 v25, -8, v1 // (op 3881) 
v_sub_u32 v26, v1, v25 // (op 3881) 

v_lshlrev_b32 v25, 4, v26 // (op 3881) 

v_lshrrev_b32 v26, 3, v1 // (op 3881) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3881) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 3881) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 4 // (op 3551) 
v_mov_b64 v[26:27], s[0:1] // (op 3551) convert


v_mov_b64 v[28:29], v[16:17] // (op 3551) 
s_add_u32 s0, 2048, s47 // (op 3551) 
s_mov_b32 m0, s0 // (op 3551) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3551) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3551) most significant half

s_add_u32 m0, m0, 2048 // (op 3551) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3551) Load value direct to lds





v_mov_b64 v[26:27], s[34:35] // (op 3294) convert

v_mov_b32 v25, v20 // (op 3294) 
buffer_load_dword v28, v25, s[64:67], 0 offen // (op 3294) Load value
v_add_u32 v25, v25, v26 // (op 3294) 
buffer_load_dword v29, v25, s[64:67], 0 offen // (op 3294) Load value





buffer_load_ubyte v25, v23, s[68:71], 0 offen offset:4 // (op 3298) Load value

v_and_b32 v23, -8, v1 // (op 4097) 
v_sub_u32 v26, v1, v23 // (op 4097) 

v_lshlrev_b32 v23, 4, v26 // (op 4097) 

v_lshrrev_b32 v26, 3, v1 // (op 4097) 
v_lshl_add_u32 v27, v26, 7, v23 // (op 4097) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 4097) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s0, 34816, s47 // (op 3556) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s0 // (op 3556) reg expression
s_mov_b32 s1, 128 // (op 3556) call()
buffer_load_dwordx4 v12, s[60:63], s1 offen offset: 0 lds // (op 3556) Load value direct to lds




v_and_b32 v23, -8, v1 // (op 4053) 
v_sub_u32 v12, v1, v23 // (op 4053) 

v_lshlrev_b32 v13, 4, v12 // (op 4053) 

v_lshrrev_b32 v12, 3, v1 // (op 4053) 
v_lshl_add_u32 v23, v12, 7, v13 // (op 4053) 


s_nop 0
v_readfirstlane_b32 s47, v23 // (op 4053) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 4 // (op 3558) 
v_mov_b64 v[12:13], s[0:1] // (op 3558) convert


v_mov_b64 v[26:27], v[16:17] // (op 3558) 
s_add_u32 s0, 36864, s47 // (op 3558) 
s_mov_b32 m0, s0 // (op 3558) reg expression
s_mov_b32 s1, 128 // (op 3558) call()
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3558) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3558) most significant half

s_add_u32 m0, m0, 2048 // (op 3558) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3558) Load value direct to lds







v_mov_b64 v[12:13], s[34:35] // (op 3308) convert

v_mov_b32 v23, v20 // (op 3308) 
buffer_load_dword v16, v23, s[64:67], 0 offen offset:4 // (op 3308) Load value
v_add_u32 v23, v23, v12 // (op 3308) 
buffer_load_dword v17, v23, s[64:67], 0 offen offset:4 // (op 3308) Load value



v_lshlrev_b32 v20, 3, v1 // (op 3991) 



v_add_u32 v23, 69632, v20 // (op 3312) 
s_waitcnt vmcnt(21)
ds_write_b32 v23, v28 // (op 3312) Store local data 

v_add_u32 v23, 69636, v20 // (op 3312) 
s_waitcnt vmcnt(20)
ds_write_b32 v23, v29 // (op 3312) Store local data 


s_barrier  // (op 4213) 



ds_read_b128 v[30:33], v4 // (op 3365) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 3365) Load local data 
s_barrier  // (op 4197) 



ds_read_b128 v[38:41], v7 offset:2048 // (op 3326) Load local data 
ds_read_b128 v[42:45], v7 offset:2112 // (op 3326) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4229) 



v_add_u32 v20, 69632, v0 // (op 3369) 
ds_read_u8 v23, v20 // (op 3369) Load local data 




ds_read_b128 v[46:49], v7 offset:6144 // (op 3331) Load local data 
ds_read_b128 v[50:53], v7 offset:6208 // (op 3331) Load local data 



v_add_u32 v20, 69760, v0 // (op 3374) 
ds_read_u8 v12, v20 // (op 3374) Load local data 




ds_read_b128 v[54:57], v7 offset:10240 // (op 3336) Load local data 
ds_read_b128 v[58:61], v7 offset:10304 // (op 3336) Load local data 



v_add_u32 v13, 69888, v0 // (op 3379) 
ds_read_u8 v20, v13 // (op 3379) Load local data 




ds_read_b128 v[62:65], v7 offset:14336 // (op 3341) Load local data 
ds_read_b128 v[66:69], v7 offset:14400 // (op 3341) Load local data 



v_add_u32 v13, 70016, v0 // (op 3384) 
ds_read_u8 v26, v13 // (op 3384) Load local data 




ds_read_b128 v[70:73], v7 offset:18432 // (op 3346) Load local data 
ds_read_b128 v[74:77], v7 offset:18496 // (op 3346) Load local data 



v_add_u32 v13, 70144, v0 // (op 3389) 
ds_read_u8 v27, v13 // (op 3389) Load local data 




ds_read_b128 v[78:81], v7 offset:22528 // (op 3351) Load local data 
ds_read_b128 v[82:85], v7 offset:22592 // (op 3351) Load local data 



v_add_u32 v13, 70272, v0 // (op 3394) 
ds_read_u8 v86, v13 // (op 3394) Load local data 




ds_read_b128 v[88:91], v7 offset:26624 // (op 3356) Load local data 
ds_read_b128 v[92:95], v7 offset:26688 // (op 3356) Load local data 



v_add_u32 v13, 70400, v0 // (op 3399) 
ds_read_u8 v87, v13 // (op 3399) Load local data 




ds_read_b128 v[96:99], v7 offset:30720 // (op 3361) Load local data 
ds_read_b128 v[100:103], v7 offset:30784 // (op 3361) Load local data 



v_add_u32 v13, 70528, v0 // (op 3404) 
ds_read_u8 v104, v13 // (op 3404) Load local data 

v_and_b32 v13, -8, v1 // (op 3853) 
v_sub_u32 v105, v1, v13 // (op 3853) 

v_lshlrev_b32 v13, 4, v105 // (op 3853) 

v_lshrrev_b32 v105, 3, v1 // (op 3853) 
v_lshl_add_u32 v106, v105, 7, v13 // (op 3853) 


s_nop 0
v_readfirstlane_b32 s47, v106 // (op 3853) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 3910) 
v_sub_u32 v105, v1, v13 // (op 3910) 

v_lshlrev_b32 v13, 4, v105 // (op 3910) 

v_lshrrev_b32 v105, 3, v1 // (op 3910) 
v_lshl_add_u32 v106, v105, 7, v13 // (op 3910) 


s_nop 0
v_readfirstlane_b32 s0, v106 // (op 3910) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v13, 3, v1 // (op 3962) 
v_and_b32 v105, -8, v1 // (op 4038) 
v_sub_u32 v106, v1, v105 // (op 4038) 

v_lshlrev_b32 v105, 4, v106 // (op 4038) 

v_lshrrev_b32 v106, 3, v1 // (op 4038) 
v_lshl_add_u32 v107, v106, 7, v105 // (op 4038) 


s_nop 0
v_readfirstlane_b32 s1, v107 // (op 4038) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v105, -8, v1 // (op 4082) 
v_sub_u32 v106, v1, v105 // (op 4082) 

v_lshlrev_b32 v105, 4, v106 // (op 4082) 

v_lshrrev_b32 v106, 3, v1 // (op 4082) 
v_lshl_add_u32 v107, v106, 7, v105 // (op 4082) 


s_nop 0
v_readfirstlane_b32 s18, v107 // (op 4082) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v105, 3, v1 // (op 4121) 
s_mov_b32 s19, 0 // (op 1287) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 1286) 
s_mov_b32 s24, s22 // (op 1286) convert

s_and_b32 s25, s24, 4294967294 // (op 1286) 

s_cmp_lt_u32 s19, s25 // (op 1286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoop_1286 // (op 1286) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoop_1286 if false)
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoop_1286:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v23, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 84) 
s_add_u32 s19, s19, 2 // (op 1288) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[30:37], a[4:7], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 350) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[30:37], a[8:11], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 512) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[30:37], a[12:15], v26, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 674) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:77], v[30:37], a[16:19], v27, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 836) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[30:37], a[20:23], v86, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 998) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:95], v[30:37], a[24:27], v87, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1160) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:103], v[30:37], a[28:31], v104, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1322) 


v_mov_b32 v106, v24 // (op 82) 

buffer_load_ubyte v22, v106, s[68:71], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 4216) 


v_mov_b64 v[108:109], v[14:15] // (op 3518) 

s_mov_b32 m0, s0 // (op 3518) reg expression
s_mov_b32 s22, 256 // (op 3518) call()
buffer_load_dwordx4 v108, s[60:63], s22 offen offset: 0 lds // (op 3518) Load value direct to lds

s_barrier  // (op 4200) 

s_lshl_b64 s[22:23], s[26:27], 4 // (op 3513) 
v_mov_b64 v[110:111], s[22:23] // (op 3513) convert


v_mov_b64 v[112:113], v[18:19] // (op 3513) 
v_mov_b64 v[114:115], v[112:113] // (op 3513) 
s_add_u32 s22, 2048, s47 // (op 3513) 
s_mov_b32 m0, s22 // (op 3513) reg expression
s_mov_b32 s23, 256 // (op 3513) call()
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3513) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3513) most significant half

s_add_u32 m0, m0, 2048 // (op 3513) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3513) Load value direct to lds





v_mov_b64 v[110:111], s[34:35] // (op 17) convert

v_mov_b32 v107, v21 // (op 17) 
v_mov_b32 v114, v107 // (op 17) 
buffer_load_dword v28, v114, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v114, v114, v110 // (op 17) 
buffer_load_dword v29, v114, s[64:67], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 4261) 



v_add_u32 v110, 70656, v105 // (op 240) 
s_waitcnt vmcnt(21)
ds_write_b32 v110, v16 // (op 240) Store local data 

v_add_u32 v110, 70660, v105 // (op 240) 
s_waitcnt vmcnt(20)
ds_write_b32 v110, v17 // (op 240) Store local data 

s_barrier  // (op 4252) 



ds_read_b128 v[30:33], v5 offset:34816 // (op 233) Load local data 
ds_read_b128 v[34:37], v5 offset:34880 // (op 233) Load local data 
s_barrier  // (op 4246) 



ds_read_b128 v[38:41], v11 offset:36864 // (op 231) Load local data 
ds_read_b128 v[42:45], v11 offset:36928 // (op 231) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4258) 



v_add_u32 v110, 70656, v10 // (op 235) 
ds_read_u8 v23, v110 // (op 235) Load local data 




ds_read_b128 v[46:49], v11 offset:40960 // (op 357) Load local data 
ds_read_b128 v[50:53], v11 offset:41024 // (op 357) Load local data 



v_add_u32 v110, 70784, v10 // (op 361) 
ds_read_u8 v12, v110 // (op 361) Load local data 




ds_read_b128 v[54:57], v11 offset:45056 // (op 519) Load local data 
ds_read_b128 v[58:61], v11 offset:45120 // (op 519) Load local data 



v_add_u32 v110, 70912, v10 // (op 523) 
ds_read_u8 v20, v110 // (op 523) Load local data 




ds_read_b128 v[62:65], v11 offset:49152 // (op 681) Load local data 
ds_read_b128 v[66:69], v11 offset:49216 // (op 681) Load local data 



v_add_u32 v110, 71040, v10 // (op 685) 
ds_read_u8 v26, v110 // (op 685) Load local data 




ds_read_b128 v[70:73], v11 offset:53248 // (op 843) Load local data 
ds_read_b128 v[74:77], v11 offset:53312 // (op 843) Load local data 



v_add_u32 v110, 71168, v10 // (op 847) 
ds_read_u8 v27, v110 // (op 847) Load local data 




ds_read_b128 v[78:81], v11 offset:57344 // (op 1005) Load local data 
ds_read_b128 v[82:85], v11 offset:57408 // (op 1005) Load local data 



v_add_u32 v110, 71296, v10 // (op 1009) 
ds_read_u8 v86, v110 // (op 1009) Load local data 




ds_read_b128 v[88:91], v11 offset:61440 // (op 1167) Load local data 
ds_read_b128 v[92:95], v11 offset:61504 // (op 1167) Load local data 



v_add_u32 v110, 71424, v10 // (op 1171) 
ds_read_u8 v87, v110 // (op 1171) Load local data 




v_add_u32 v110, 65536, v11 // (op 1329) 
ds_read_b128 v[96:99], v110 // (op 1329) Load local data 

v_add_u32 v110, 65600, v11 // (op 1329) 
ds_read_b128 v[100:103], v110 // (op 1329) Load local data 




v_add_u32 v110, 71552, v10 // (op 1333) 
ds_read_u8 v104, v110 // (op 1333) Load local data 

s_barrier  // (op 4340) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 238) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[30:37], a[4:7], v12, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 351) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[30:37], a[8:11], v20, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 513) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[30:37], a[12:15], v26, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 675) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:77], v[30:37], a[16:19], v27, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 837) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[30:37], a[20:23], v86, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 999) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:95], v[30:37], a[24:27], v87, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1161) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:103], v[30:37], a[28:31], v104, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1323) 



buffer_load_ubyte v25, v106, s[68:71], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 4255) 
v_add_u32 v24, 8, v24 // (op 3807) 


s_add_u32 s22, 34816, s18 // (op 3541) 

s_mov_b32 m0, s22 // (op 3541) reg expression
s_mov_b32 s23, 384 // (op 3541) call()
buffer_load_dwordx4 v108, s[60:63], s23 offen offset: 0 lds // (op 3541) Load value direct to lds



s_barrier  // (op 4249) 
v_add_co_u32 v14, vcc, 256, v14 // (op 3679) least significant half

v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3679) most significant half
s_lshl_b64 s[22:23], s[26:27], 4 // (op 3536) 
v_mov_b64 v[108:109], s[22:23] // (op 3536) convert


v_mov_b64 v[110:111], v[112:113] // (op 3536) 
s_add_u32 s22, 36864, s1 // (op 3536) 
s_mov_b32 m0, s22 // (op 3536) reg expression
s_mov_b32 s23, 384 // (op 3536) call()
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3536) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3536) most significant half

s_add_u32 m0, m0, 2048 // (op 3536) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3536) Load value direct to lds






v_add_co_u32 v18, vcc, 256, v18 // (op 3615) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 3615) most significant half
v_mov_b64 v[108:109], s[34:35] // (op 242) convert

v_mov_b32 v106, v107 // (op 242) 
buffer_load_dword v16, v106, s[64:67], 0 offen offset:12 // (op 242) Load value
v_add_u32 v106, v106, v108 // (op 242) 
buffer_load_dword v17, v106, s[64:67], 0 offen offset:12 // (op 242) Load value



v_add_u32 v21, 8, v21 // (op 3743) 
s_barrier  // (op 4232) 



v_add_u32 v106, 69632, v13 // (op 21) 
s_waitcnt vmcnt(21)
ds_write_b32 v106, v28 // (op 21) Store local data 

v_add_u32 v106, 69636, v13 // (op 21) 
s_waitcnt vmcnt(20)
ds_write_b32 v106, v29 // (op 21) Store local data 

s_barrier  // (op 4222) 



ds_read_b128 v[30:33], v4 // (op 72) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 72) Load local data 
s_barrier  // (op 4206) 



ds_read_b128 v[38:41], v7 offset:2048 // (op 67) Load local data 
ds_read_b128 v[42:45], v7 offset:2112 // (op 67) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4239) 



v_add_u32 v106, 69632, v0 // (op 77) 
ds_read_u8 v23, v106 // (op 77) Load local data 




ds_read_b128 v[46:49], v7 offset:6144 // (op 344) Load local data 
ds_read_b128 v[50:53], v7 offset:6208 // (op 344) Load local data 



v_add_u32 v106, 69760, v0 // (op 348) 
ds_read_u8 v12, v106 // (op 348) Load local data 




ds_read_b128 v[54:57], v7 offset:10240 // (op 506) Load local data 
ds_read_b128 v[58:61], v7 offset:10304 // (op 506) Load local data 



v_add_u32 v106, 69888, v0 // (op 510) 
ds_read_u8 v20, v106 // (op 510) Load local data 




ds_read_b128 v[62:65], v7 offset:14336 // (op 668) Load local data 
ds_read_b128 v[66:69], v7 offset:14400 // (op 668) Load local data 



v_add_u32 v106, 70016, v0 // (op 672) 
ds_read_u8 v26, v106 // (op 672) Load local data 




ds_read_b128 v[70:73], v7 offset:18432 // (op 830) Load local data 
ds_read_b128 v[74:77], v7 offset:18496 // (op 830) Load local data 



v_add_u32 v106, 70144, v0 // (op 834) 
ds_read_u8 v27, v106 // (op 834) Load local data 




ds_read_b128 v[78:81], v7 offset:22528 // (op 992) Load local data 
ds_read_b128 v[82:85], v7 offset:22592 // (op 992) Load local data 



v_add_u32 v106, 70272, v0 // (op 996) 
ds_read_u8 v86, v106 // (op 996) Load local data 




ds_read_b128 v[88:91], v7 offset:26624 // (op 1154) Load local data 
ds_read_b128 v[92:95], v7 offset:26688 // (op 1154) Load local data 



v_add_u32 v106, 70400, v0 // (op 1158) 
ds_read_u8 v87, v106 // (op 1158) Load local data 




ds_read_b128 v[96:99], v7 offset:30720 // (op 1316) Load local data 
ds_read_b128 v[100:103], v7 offset:30784 // (op 1316) Load local data 



v_add_u32 v106, 70528, v0 // (op 1320) 
ds_read_u8 v104, v106 // (op 1320) Load local data 

s_ashr_i64 s[22:23], s[42:43], 7 // (op 1286) 
s_mov_b32 s24, s22 // (op 1286) convert

s_and_b32 s25, s24, 4294967294 // (op 1286) 

s_cmp_lt_u32 s19, s25 // (op 1286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoop_1286 // (op 1286) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoop_1286 if true)
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoop_1286:




















s_barrier  // (op 4324) 
s_mul_hi_i32 s47, s16, s10 // (op 3571) 
v_lshrrev_b32 v12, 3, v1 // (op 3571) 
s_add_i32 s0, s47, s16 // (op 3571) 

v_and_b32 v13, -8, v1 // (op 3571) 
s_ashr_i32 s1, s0, 31 // (op 3571) 
v_sub_u32 v14, v1, v13 // (op 3571) 

s_cmp_eq_i32 s10, 0 // (op 3571) 
s_mov_b32 s47, scc // (op 3571) 
v_lshlrev_b32 v13, 4, v14 // (op 3571) 

s_cmp_eq_i32 s47, 0 // (op 3571) 
s_cselect_b32 s18, 0, -1 // (op 3571) 

s_and_b32 s19, s13, 31 // (op 3571) 
s_lshl_b32 s47, 1, s19 // (op 3571) 
s_add_i32 s22, s47, s18 // (op 3571) 


s_and_b32 s18, s1, s22 // (op 3571) 


s_add_i32 s1, s0, s18 // (op 3571) 


s_ashr_i32 s0, s1, s19 // (op 3571) 


s_lshl_b32 s1, s13, 24 // (op 3571) 
s_ashr_i32 s47, s1, 31 // (op 3571) 

s_xor_b32 s1, s0, s47 // (op 3571) 

s_sub_i32 s0, s1, s47 // (op 3571) 


s_mul_hi_i32 s1, s0, s14 // (op 3571) 
s_add_i32 s47, s1, s0 // (op 3571) 

s_ashr_i32 s1, s47, 31 // (op 3571) 
s_cmp_eq_i32 s14, 0 // (op 3571) 
s_mov_b32 s18, scc // (op 3571) 
s_cmp_eq_i32 s18, 0 // (op 3571) 
s_cselect_b32 s19, 0, -1 // (op 3571) 

s_and_b32 s18, s15, 31 // (op 3571) 
s_lshl_b32 s22, 1, s18 // (op 3571) 
s_add_i32 s23, s22, s19 // (op 3571) 


s_and_b32 s19, s1, s23 // (op 3571) 


s_add_i32 s1, s47, s19 // (op 3571) 


s_ashr_i32 s19, s1, s18 // (op 3571) 


s_lshl_b32 s1, s15, 24 // (op 3571) 
s_ashr_i32 s18, s1, 31 // (op 3571) 

s_xor_b32 s1, s19, s18 // (op 3571) 

s_sub_i32 s19, s1, s18 // (op 3571) 


s_mul_i32 s1, s0, s11 // (op 3571) 

s_sub_u32 s0, s16, s1 // (op 3571) 

s_mul_i32 s1, s19, s11 // (op 3571) 
s_add_u32 s1, s1, s0 // (op 3571) 


s_mul_hi_i32 s0, s16, s9 // (op 3571) 
s_add_i32 s47, s0, s16 // (op 3571) 

s_ashr_i32 s0, s47, 31 // (op 3571) 
s_cmp_eq_i32 s9, 0 // (op 3571) 
s_mov_b32 s18, scc // (op 3571) 
s_cmp_eq_i32 s18, 0 // (op 3571) 
s_cselect_b32 s19, 0, -1 // (op 3571) 

s_and_b32 s18, s45, 31 // (op 3571) 
s_lshl_b32 s22, 1, s18 // (op 3571) 
s_add_i32 s23, s22, s19 // (op 3571) 


s_and_b32 s19, s0, s23 // (op 3571) 


s_add_i32 s0, s47, s19 // (op 3571) 


s_ashr_i32 s19, s0, s18 // (op 3571) 


s_lshl_b32 s0, s45, 24 // (op 3571) 
s_ashr_i32 s18, s0, 31 // (op 3571) 

s_xor_b32 s0, s19, s18 // (op 3571) 

s_sub_i32 s19, s0, s18 // (op 3571) 


s_add_u32 s22, s4, 16 // (op 3571) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3571) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3571) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3571) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3571) 

s_mul_i32 s0, s11, s22 // (op 3571) 

s_mul_i32 s18, s8, s0 // (op 3571) 

s_mul_i32 s0, s19, s18 // (op 3571) 

s_sub_u32 s18, s16, s0 // (op 3571) 

s_mov_b32 s0, s18 // (op 3571) 
s_mul_hi_i32 s47, s0, s12 // (op 3571) 
s_add_i32 s22, s47, s0 // (op 3571) 


s_ashr_i32 s0, s22, 31 // (op 3571) 
s_cmp_eq_i32 s12, 0 // (op 3571) 
s_mov_b32 s47, scc // (op 3571) 
s_cmp_eq_i32 s47, 0 // (op 3571) 
s_cselect_b32 s23, 0, -1 // (op 3571) 

s_and_b32 s47, s44, 31 // (op 3571) 
s_lshl_b32 s24, 1, s47 // (op 3571) 
s_add_i32 s25, s24, s23 // (op 3571) 


s_and_b32 s23, s0, s25 // (op 3571) 


s_add_i32 s0, s22, s23 // (op 3571) 


s_ashr_i32 s22, s0, s47 // (op 3571) 


s_lshl_b32 s0, s44, 24 // (op 3571) 
s_ashr_i32 s47, s0, 31 // (op 3571) 

s_xor_b32 s0, s22, s47 // (op 3571) 

s_sub_i32 s22, s0, s47 // (op 3571) 


s_mul_i32 s0, s46, s11 // (op 3571) 
s_add_u32 s24, s6, 256 // (op 3571) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3571) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 3571) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 3571) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 8 // (op 3571) 

s_sub_i32 s23, s24, s0 // (op 3571) 

s_mul_i32 s47, s22, s23 // (op 3571) 


s_sub_u32 s22, s18, s47 // (op 3571) 


s_add_u32 s18, s22, s0 // (op 3571) 


s_cmp_eq_i32 s19, 0 // (op 3571) 
s_mov_b32 s0, scc // (op 3571) 

s_cmp_eq_i32 s0, 0 // (op 3571) 
s_cselect_b32 s19, s18, s1 // (op 3571) 



v_lshl_add_u32 v14, s19, 8, v12 // (op 3571) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 3571) 
s_lshr_b32 s47, s0, 1 // (op 3571) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 3571) 


v_mov_b32 v19, 0 // (op 3571) convert
v_mov_b32 v18, v12 // (op 3571) convert

v_mul_lo_u32 v13, v14, s27 // (op 3571) most significant: low of low * high
v_mul_hi_u32 v15, v14, s26 // (op 3571) most significant: high of low * low
v_mul_lo_u32 v12, v14, s26 // (op 3571) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 3571) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v18 // (op 3571) least significant half
v_addc_co_u32 v13, s[0:1], v13, v19, s[0:1] // (op 3571) most significant half



s_mul_i32 s1, s26, s7 // (op 3573) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 3573) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 3573) least significant
s_mul_hi_u32 s18, s26, s6 // (op 3573) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3573) most significant: sum
s_add_u32 s1, s1, s18 // (op 3573) most significant: sum


s_mov_b32 s48, s28 // (op 3573) reg expression
s_mov_b32 s49, s29 // (op 3573) reg expression
s_mov_b32 s50, s0 // (op 3573) reg expression
s_mov_b32 s51, 131072 // (op 3573) call()

s_mul_hi_i32 s47, s16, s10 // (op 3645) 
v_lshrrev_b32 v14, 3, v1 // (op 3645) 
s_add_i32 s0, s47, s16 // (op 3645) 

v_and_b32 v15, -8, v1 // (op 3645) 
s_ashr_i32 s1, s0, 31 // (op 3645) 
v_sub_u32 v18, v1, v15 // (op 3645) 

s_cmp_eq_i32 s10, 0 // (op 3645) 
s_mov_b32 s47, scc // (op 3645) 
v_lshlrev_b32 v15, 4, v18 // (op 3645) 

s_cmp_eq_i32 s47, 0 // (op 3645) 
s_cselect_b32 s18, 0, -1 // (op 3645) 

s_and_b32 s19, s13, 31 // (op 3645) 
s_lshl_b32 s47, 1, s19 // (op 3645) 
s_add_i32 s22, s47, s18 // (op 3645) 


s_and_b32 s18, s1, s22 // (op 3645) 


s_add_i32 s1, s0, s18 // (op 3645) 


s_ashr_i32 s0, s1, s19 // (op 3645) 


s_lshl_b32 s1, s13, 24 // (op 3645) 
s_ashr_i32 s47, s1, 31 // (op 3645) 

s_xor_b32 s1, s0, s47 // (op 3645) 

s_sub_i32 s0, s1, s47 // (op 3645) 


s_mul_hi_i32 s1, s0, s14 // (op 3645) 
s_add_i32 s47, s1, s0 // (op 3645) 

s_ashr_i32 s1, s47, 31 // (op 3645) 
s_cmp_eq_i32 s14, 0 // (op 3645) 
s_mov_b32 s18, scc // (op 3645) 
s_cmp_eq_i32 s18, 0 // (op 3645) 
s_cselect_b32 s19, 0, -1 // (op 3645) 

s_and_b32 s18, s15, 31 // (op 3645) 
s_lshl_b32 s22, 1, s18 // (op 3645) 
s_add_i32 s23, s22, s19 // (op 3645) 


s_and_b32 s19, s1, s23 // (op 3645) 


s_add_i32 s1, s47, s19 // (op 3645) 


s_ashr_i32 s19, s1, s18 // (op 3645) 


s_lshl_b32 s1, s15, 24 // (op 3645) 
s_ashr_i32 s18, s1, 31 // (op 3645) 

s_xor_b32 s1, s19, s18 // (op 3645) 

s_sub_i32 s19, s1, s18 // (op 3645) 


s_add_u32 s22, s4, 16 // (op 3645) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3645) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3645) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3645) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3645) 

s_mul_i32 s1, s19, s22 // (op 3645) 

s_sub_i32 s47, s0, s1 // (op 3645) 


s_mul_hi_i32 s0, s16, s9 // (op 3645) 
s_add_i32 s1, s0, s16 // (op 3645) 

s_ashr_i32 s0, s1, 31 // (op 3645) 
s_cmp_eq_i32 s9, 0 // (op 3645) 
s_mov_b32 s18, scc // (op 3645) 
s_cmp_eq_i32 s18, 0 // (op 3645) 
s_cselect_b32 s19, 0, -1 // (op 3645) 

s_and_b32 s18, s45, 31 // (op 3645) 
s_lshl_b32 s24, 1, s18 // (op 3645) 
s_add_i32 s25, s24, s19 // (op 3645) 


s_and_b32 s19, s0, s25 // (op 3645) 


s_add_i32 s0, s1, s19 // (op 3645) 


s_ashr_i32 s1, s0, s18 // (op 3645) 


s_lshl_b32 s0, s45, 24 // (op 3645) 
s_ashr_i32 s18, s0, 31 // (op 3645) 

s_xor_b32 s0, s1, s18 // (op 3645) 

s_sub_i32 s1, s0, s18 // (op 3645) 


s_mul_i32 s0, s11, s22 // (op 3645) 

s_mul_i32 s18, s8, s0 // (op 3645) 

s_mul_i32 s0, s1, s18 // (op 3645) 

s_sub_u32 s18, s16, s0 // (op 3645) 

s_mul_hi_i32 s0, s18, s12 // (op 3645) 
s_add_i32 s19, s0, s18 // (op 3645) 


s_ashr_i32 s0, s19, 31 // (op 3645) 
s_cmp_eq_i32 s12, 0 // (op 3645) 
s_mov_b32 s18, scc // (op 3645) 
s_cmp_eq_i32 s18, 0 // (op 3645) 
s_cselect_b32 s22, 0, -1 // (op 3645) 

s_and_b32 s18, s44, 31 // (op 3645) 
s_lshl_b32 s23, 1, s18 // (op 3645) 
s_add_i32 s24, s23, s22 // (op 3645) 


s_and_b32 s25, s0, s24 // (op 3645) 


s_add_i32 s0, s19, s25 // (op 3645) 


s_ashr_i32 s19, s0, s18 // (op 3645) 


s_lshl_b32 s0, s44, 24 // (op 3645) 
s_ashr_i32 s18, s0, 31 // (op 3645) 

s_xor_b32 s0, s19, s18 // (op 3645) 

s_sub_i32 s19, s0, s18 // (op 3645) 


s_cmp_eq_i32 s1, 0 // (op 3645) 
s_mov_b32 s0, scc // (op 3645) 

s_cmp_eq_i32 s0, 0 // (op 3645) 
s_cselect_b32 s1, s19, s47 // (op 3645) 



v_lshl_add_u32 v18, s1, 4, v14 // (op 3645) 


v_mul_lo_u32 v29, v18, s31 // (op 3645) most significant: low of low * high
v_mul_hi_u32 v14, v18, s30 // (op 3645) most significant: high of low * low
v_mul_lo_u32 v28, v18, s30 // (op 3645) least significant: low of low * low
v_add_u32 v29, v29, v14 // (op 3645) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 3645) 
s_lshr_b32 s47, s0, 1 // (op 3645) 

v_lshl_add_u32 v14, s47, 8, v15 // (op 3645) 


v_mov_b32 v19, 0 // (op 3645) convert
v_mov_b32 v18, v14 // (op 3645) convert

v_add_co_u32 v14, s[0:1], v18, v28 // (op 3645) least significant half
v_addc_co_u32 v15, s[0:1], v19, v29, s[0:1] // (op 3645) most significant half



s_mul_i32 s1, s30, s5 // (op 3647) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 3647) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 3647) least significant
s_mul_hi_u32 s18, s30, s4 // (op 3647) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3647) most significant: sum
s_add_u32 s1, s1, s18 // (op 3647) most significant: sum


s_mov_b32 s60, s32 // (op 3647) reg expression
s_mov_b32 s61, s33 // (op 3647) reg expression
s_mov_b32 s62, s0 // (op 3647) reg expression
s_mov_b32 s63, 131072 // (op 3647) call()

s_mul_hi_i32 s47, s16, s10 // (op 3709) 
v_lshlrev_b32 v18, 1, v1 // (op 3709) 
s_add_i32 s0, s47, s16 // (op 3709) 

s_ashr_i32 s1, s0, 31 // (op 3709) 
s_cmp_eq_i32 s10, 0 // (op 3709) 
s_mov_b32 s47, scc // (op 3709) 
s_cmp_eq_i32 s47, 0 // (op 3709) 
s_cselect_b32 s18, 0, -1 // (op 3709) 

s_and_b32 s19, s13, 31 // (op 3709) 
s_lshl_b32 s47, 1, s19 // (op 3709) 
s_add_i32 s22, s47, s18 // (op 3709) 


s_and_b32 s18, s1, s22 // (op 3709) 


s_add_i32 s1, s0, s18 // (op 3709) 


s_ashr_i32 s0, s1, s19 // (op 3709) 


s_lshl_b32 s1, s13, 24 // (op 3709) 
s_ashr_i32 s47, s1, 31 // (op 3709) 

s_xor_b32 s1, s0, s47 // (op 3709) 

s_sub_i32 s0, s1, s47 // (op 3709) 


s_mul_hi_i32 s1, s0, s14 // (op 3709) 
s_add_i32 s47, s1, s0 // (op 3709) 

s_ashr_i32 s1, s47, 31 // (op 3709) 
s_cmp_eq_i32 s14, 0 // (op 3709) 
s_mov_b32 s18, scc // (op 3709) 
s_cmp_eq_i32 s18, 0 // (op 3709) 
s_cselect_b32 s19, 0, -1 // (op 3709) 

s_and_b32 s18, s15, 31 // (op 3709) 
s_lshl_b32 s22, 1, s18 // (op 3709) 
s_add_i32 s23, s22, s19 // (op 3709) 


s_and_b32 s19, s1, s23 // (op 3709) 


s_add_i32 s1, s47, s19 // (op 3709) 


s_ashr_i32 s19, s1, s18 // (op 3709) 


s_lshl_b32 s1, s15, 24 // (op 3709) 
s_ashr_i32 s18, s1, 31 // (op 3709) 

s_xor_b32 s1, s19, s18 // (op 3709) 

s_sub_i32 s19, s1, s18 // (op 3709) 


s_mul_i32 s1, s0, s11 // (op 3709) 

s_sub_u32 s0, s16, s1 // (op 3709) 

s_mul_i32 s1, s19, s11 // (op 3709) 
s_add_u32 s1, s1, s0 // (op 3709) 


s_mul_hi_i32 s0, s16, s9 // (op 3709) 
s_add_i32 s47, s0, s16 // (op 3709) 

s_ashr_i32 s0, s47, 31 // (op 3709) 
s_cmp_eq_i32 s9, 0 // (op 3709) 
s_mov_b32 s18, scc // (op 3709) 
s_cmp_eq_i32 s18, 0 // (op 3709) 
s_cselect_b32 s19, 0, -1 // (op 3709) 

s_and_b32 s18, s45, 31 // (op 3709) 
s_lshl_b32 s22, 1, s18 // (op 3709) 
s_add_i32 s23, s22, s19 // (op 3709) 


s_and_b32 s19, s0, s23 // (op 3709) 


s_add_i32 s0, s47, s19 // (op 3709) 


s_ashr_i32 s19, s0, s18 // (op 3709) 


s_lshl_b32 s0, s45, 24 // (op 3709) 
s_ashr_i32 s18, s0, 31 // (op 3709) 

s_xor_b32 s0, s19, s18 // (op 3709) 

s_sub_i32 s19, s0, s18 // (op 3709) 


s_add_u32 s22, s4, 16 // (op 3709) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3709) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3709) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3709) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3709) 

s_mul_i32 s0, s11, s22 // (op 3709) 

s_mul_i32 s18, s8, s0 // (op 3709) 

s_mul_i32 s0, s19, s18 // (op 3709) 

s_sub_u32 s18, s16, s0 // (op 3709) 

s_mov_b32 s0, s18 // (op 3709) 
s_mul_hi_i32 s47, s0, s12 // (op 3709) 
s_add_i32 s22, s47, s0 // (op 3709) 


s_ashr_i32 s0, s22, 31 // (op 3709) 
s_cmp_eq_i32 s12, 0 // (op 3709) 
s_mov_b32 s47, scc // (op 3709) 
s_cmp_eq_i32 s47, 0 // (op 3709) 
s_cselect_b32 s23, 0, -1 // (op 3709) 

s_and_b32 s47, s44, 31 // (op 3709) 
s_lshl_b32 s24, 1, s47 // (op 3709) 
s_add_i32 s25, s24, s23 // (op 3709) 


s_and_b32 s23, s0, s25 // (op 3709) 


s_add_i32 s0, s22, s23 // (op 3709) 


s_ashr_i32 s22, s0, s47 // (op 3709) 


s_lshl_b32 s0, s44, 24 // (op 3709) 
s_ashr_i32 s47, s0, 31 // (op 3709) 

s_xor_b32 s0, s22, s47 // (op 3709) 

s_sub_i32 s22, s0, s47 // (op 3709) 


s_mul_i32 s0, s46, s11 // (op 3709) 
s_add_u32 s24, s6, 256 // (op 3709) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3709) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 3709) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 3709) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 8 // (op 3709) 

s_sub_i32 s23, s24, s0 // (op 3709) 

s_mul_i32 s47, s22, s23 // (op 3709) 


s_sub_u32 s22, s18, s47 // (op 3709) 


s_add_u32 s18, s22, s0 // (op 3709) 


s_cmp_eq_i32 s19, 0 // (op 3709) 
s_mov_b32 s0, scc // (op 3709) 

s_cmp_eq_i32 s0, 0 // (op 3709) 
s_cselect_b32 s19, s18, s1 // (op 3709) 



v_lshl_add_u32 v19, s19, 8, v18 // (op 3709) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 3709) 
s_lshr_b32 s47, s0, 1 // (op 3709) 

s_lshl_b32 s0, s47, 3 // (op 3709) 

v_mul_lo_u32 v18, v19, s34 // (op 3709) 
v_add_u32 v18, v18, s0 // (op 3709) 


s_mul_i32 s1, s34, s7 // (op 3711) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 3711) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 3711) least significant
s_mul_hi_u32 s18, s34, s6 // (op 3711) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3711) most significant: sum
s_add_u32 s1, s1, s18 // (op 3711) most significant: sum


s_mov_b32 s64, s36 // (op 3711) reg expression
s_mov_b32 s65, s37 // (op 3711) reg expression
s_mov_b32 s66, s0 // (op 3711) reg expression
s_mov_b32 s67, 131072 // (op 3711) call()

s_mul_hi_i32 s47, s16, s10 // (op 3773) 
v_and_b32 v19, 15, v1 // (op 3773) 
s_add_i32 s0, s47, s16 // (op 3773) 

v_and_b32 v20, 63, v1 // (op 3773) 
s_ashr_i32 s1, s0, 31 // (op 3773) 
v_lshrrev_b32 v21, 4, v20 // (op 3773) 

s_cmp_eq_i32 s10, 0 // (op 3773) 
s_mov_b32 s47, scc // (op 3773) 
s_cmp_eq_i32 s47, 0 // (op 3773) 
s_cselect_b32 s18, 0, -1 // (op 3773) 

s_and_b32 s19, s13, 31 // (op 3773) 
s_lshl_b32 s47, 1, s19 // (op 3773) 
s_add_i32 s22, s47, s18 // (op 3773) 


s_and_b32 s18, s1, s22 // (op 3773) 


s_add_i32 s1, s0, s18 // (op 3773) 


s_ashr_i32 s0, s1, s19 // (op 3773) 


s_lshl_b32 s1, s13, 24 // (op 3773) 
s_ashr_i32 s47, s1, 31 // (op 3773) 

s_xor_b32 s1, s0, s47 // (op 3773) 

s_sub_i32 s0, s1, s47 // (op 3773) 


s_mul_hi_i32 s1, s0, s14 // (op 3773) 
s_add_i32 s47, s1, s0 // (op 3773) 

s_ashr_i32 s1, s47, 31 // (op 3773) 
s_cmp_eq_i32 s14, 0 // (op 3773) 
s_mov_b32 s18, scc // (op 3773) 
s_cmp_eq_i32 s18, 0 // (op 3773) 
s_cselect_b32 s19, 0, -1 // (op 3773) 

s_and_b32 s18, s15, 31 // (op 3773) 
s_lshl_b32 s22, 1, s18 // (op 3773) 
s_add_i32 s23, s22, s19 // (op 3773) 


s_and_b32 s19, s1, s23 // (op 3773) 


s_add_i32 s1, s47, s19 // (op 3773) 


s_ashr_i32 s19, s1, s18 // (op 3773) 


s_lshl_b32 s1, s15, 24 // (op 3773) 
s_ashr_i32 s18, s1, 31 // (op 3773) 

s_xor_b32 s1, s19, s18 // (op 3773) 

s_sub_i32 s19, s1, s18 // (op 3773) 


s_add_u32 s22, s4, 16 // (op 3773) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3773) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3773) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3773) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3773) 

s_mul_i32 s1, s19, s22 // (op 3773) 

s_sub_i32 s47, s0, s1 // (op 3773) 


s_mul_hi_i32 s0, s16, s9 // (op 3773) 
s_add_i32 s1, s0, s16 // (op 3773) 

s_ashr_i32 s0, s1, 31 // (op 3773) 
s_cmp_eq_i32 s9, 0 // (op 3773) 
s_mov_b32 s18, scc // (op 3773) 
s_cmp_eq_i32 s18, 0 // (op 3773) 
s_cselect_b32 s19, 0, -1 // (op 3773) 

s_and_b32 s18, s45, 31 // (op 3773) 
s_lshl_b32 s24, 1, s18 // (op 3773) 
s_add_i32 s25, s24, s19 // (op 3773) 


s_and_b32 s19, s0, s25 // (op 3773) 


s_add_i32 s0, s1, s19 // (op 3773) 


s_ashr_i32 s1, s0, s18 // (op 3773) 


s_lshl_b32 s0, s45, 24 // (op 3773) 
s_ashr_i32 s18, s0, 31 // (op 3773) 

s_xor_b32 s0, s1, s18 // (op 3773) 

s_sub_i32 s1, s0, s18 // (op 3773) 


s_mul_i32 s0, s11, s22 // (op 3773) 

s_mul_i32 s18, s8, s0 // (op 3773) 

s_mul_i32 s0, s1, s18 // (op 3773) 

s_sub_u32 s18, s16, s0 // (op 3773) 

s_mul_hi_i32 s0, s18, s12 // (op 3773) 
s_add_i32 s19, s0, s18 // (op 3773) 


s_ashr_i32 s0, s19, 31 // (op 3773) 
s_cmp_eq_i32 s12, 0 // (op 3773) 
s_mov_b32 s18, scc // (op 3773) 
s_cmp_eq_i32 s18, 0 // (op 3773) 
s_cselect_b32 s22, 0, -1 // (op 3773) 

s_and_b32 s18, s44, 31 // (op 3773) 
s_lshl_b32 s23, 1, s18 // (op 3773) 
s_add_i32 s24, s23, s22 // (op 3773) 


s_and_b32 s25, s0, s24 // (op 3773) 


s_add_i32 s0, s19, s25 // (op 3773) 


s_ashr_i32 s19, s0, s18 // (op 3773) 


s_lshl_b32 s0, s44, 24 // (op 3773) 
s_ashr_i32 s18, s0, 31 // (op 3773) 

s_xor_b32 s0, s19, s18 // (op 3773) 

s_sub_i32 s19, s0, s18 // (op 3773) 


s_cmp_eq_i32 s1, 0 // (op 3773) 
s_mov_b32 s0, scc // (op 3773) 

s_cmp_eq_i32 s0, 0 // (op 3773) 
s_cselect_b32 s1, s19, s47 // (op 3773) 



v_lshl_add_u32 v20, s1, 4, v19 // (op 3773) 


v_mul_lo_u32 v19, v20, s38 // (op 3773) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 3773) 
s_lshr_b32 s47, s0, 1 // (op 3773) 

v_lshl_add_u32 v20, s47, 3, v21 // (op 3773) 


v_add_u32 v21, v20, v19 // (op 3773) 


s_mul_i32 s1, s38, s5 // (op 3775) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 3775) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 3775) least significant
s_mul_hi_u32 s18, s38, s4 // (op 3775) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3775) most significant: sum
s_add_u32 s1, s1, s18 // (op 3775) most significant: sum


s_mov_b32 s68, s40 // (op 3775) reg expression
s_mov_b32 s69, s41 // (op 3775) reg expression
s_mov_b32 s70, s0 // (op 3775) reg expression
s_mov_b32 s71, 131072 // (op 3775) call()

v_and_b32 v19, -8, v1 // (op 3866) 
v_sub_u32 v20, v1, v19 // (op 3866) 

v_lshlrev_b32 v19, 4, v20 // (op 3866) 

v_lshrrev_b32 v20, 3, v1 // (op 3866) 
v_lshl_add_u32 v22, v20, 7, v19 // (op 3866) 


s_nop 0
v_readfirstlane_b32 s47, v22 // (op 3866) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 3923) 
v_sub_u32 v20, v1, v19 // (op 3923) 

v_lshlrev_b32 v19, 4, v20 // (op 3923) 

v_lshrrev_b32 v20, 3, v1 // (op 3923) 
v_lshl_add_u32 v22, v20, 7, v19 // (op 3923) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 3923) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 3, v1 // (op 3975) 
s_ashr_i64 s[18:19], s[42:43], 7 // (op 1292) 
s_and_b32 s1, s18, 4294967294 // (op 1292) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 1291) 
s_mov_b32 s22, s18 // (op 1291) convert

s_cmp_lt_u32 s1, s22 // (op 1291) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_1291 // (op 1291) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_1291 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoopTail_1291:
s_barrier  // (op 4203) 
s_add_u32 s1, s1, 1 // (op 1293) 

s_lshl_b64 s[18:19], s[26:27], 4 // (op 3523) 
v_mov_b64 v[16:17], s[18:19] // (op 3523) convert


v_mov_b64 v[22:23], v[12:13] // (op 3523) 
v_mov_b64 v[24:25], v[22:23] // (op 3523) 
s_add_u32 s18, 2048, s47 // (op 3523) 
s_mov_b32 m0, s18 // (op 3523) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3523) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3523) most significant half

s_add_u32 m0, m0, 2048 // (op 3523) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3523) Load value direct to lds




v_add_co_u32 v12, vcc, 128, v12 // (op 3561) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 3561) most significant half
s_barrier  // (op 4219) 


v_mov_b64 v[16:17], v[14:15] // (op 3529) 

s_mov_b32 m0, s0 // (op 3529) reg expression
buffer_load_dwordx4 v16, s[60:63], 0 offen offset: 0 lds // (op 3529) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 3637) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3637) most significant half

v_mov_b64 v[16:17], s[34:35] // (op 184) convert

v_mov_b32 v20, v18 // (op 184) 
v_mov_b32 v22, v20 // (op 184) 
buffer_load_dword v24, v22, s[64:67], 0 offen // (op 184) Load value
v_add_u32 v22, v22, v16 // (op 184) 
buffer_load_dword v25, v22, s[64:67], 0 offen // (op 184) Load value



v_add_u32 v18, 4, v18 // (op 3701) 
s_barrier  // (op 4235) 
s_waitcnt vmcnt(3)
s_barrier  // (op 4209) 






v_add_u32 v20, 69632, v19 // (op 182) 
ds_read_b128 v[26:29], v7 offset:2048 // (op 173) Load local data 
ds_read_b128 v[30:33], v7 offset:2112 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b32 v20, v24 // (op 182) Store local data 

s_barrier  // (op 4225) 
v_add_u32 v20, 69636, v19 // (op 182) 



ds_read_b128 v[34:37], v4 // (op 175) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 175) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v20, v25 // (op 182) Store local data 


s_waitcnt lgkmcnt(0)
s_barrier  // (op 4242) 



v_add_u32 v20, 69632, v0 // (op 177) 
ds_read_u8 v16, v20 // (op 177) Load local data 



v_mov_b32 v17, v21 // (op 179) 

buffer_load_ubyte v20, v17, s[68:71], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v21, 4, v21 // (op 3765) 



ds_read_b128 v[42:45], v7 offset:6144 // (op 325) Load local data 
ds_read_b128 v[46:49], v7 offset:6208 // (op 325) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v16, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 180) 





v_add_u32 v16, 69760, v0 // (op 329) 
ds_read_u8 v17, v16 // (op 329) Load local data 




ds_read_b128 v[22:25], v7 offset:10240 // (op 487) Load local data 
ds_read_b128 v[26:29], v7 offset:10304 // (op 487) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[34:41], a[4:7], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 331) 





v_add_u32 v16, 69888, v0 // (op 491) 
ds_read_u8 v17, v16 // (op 491) Load local data 




ds_read_b128 v[42:45], v7 offset:14336 // (op 649) Load local data 
ds_read_b128 v[46:49], v7 offset:14400 // (op 649) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:29], v[34:41], a[8:11], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 493) 





v_add_u32 v16, 70016, v0 // (op 653) 
ds_read_u8 v17, v16 // (op 653) Load local data 




ds_read_b128 v[22:25], v7 offset:18432 // (op 811) Load local data 
ds_read_b128 v[26:29], v7 offset:18496 // (op 811) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[34:41], a[12:15], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 655) 





v_add_u32 v16, 70144, v0 // (op 815) 
ds_read_u8 v17, v16 // (op 815) Load local data 




ds_read_b128 v[42:45], v7 offset:22528 // (op 973) Load local data 
ds_read_b128 v[46:49], v7 offset:22592 // (op 973) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:29], v[34:41], a[16:19], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 817) 





v_add_u32 v16, 70272, v0 // (op 977) 
ds_read_u8 v17, v16 // (op 977) Load local data 




ds_read_b128 v[22:25], v7 offset:26624 // (op 1135) Load local data 
ds_read_b128 v[26:29], v7 offset:26688 // (op 1135) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[42:49], v[34:41], a[20:23], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 979) 





v_add_u32 v16, 70400, v0 // (op 1139) 
ds_read_u8 v17, v16 // (op 1139) Load local data 




ds_read_b128 v[42:45], v7 offset:30720 // (op 1297) Load local data 
ds_read_b128 v[46:49], v7 offset:30784 // (op 1297) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:29], v[34:41], a[24:27], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1141) 





v_add_u32 v16, 70528, v0 // (op 1301) 
ds_read_u8 v17, v16 // (op 1301) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:49], v[34:41], a[28:31], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1303) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 1291) 
s_mov_b32 s22, s18 // (op 1291) convert

s_cmp_lt_u32 s1, s22 // (op 1291) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoopTail_1291 // (op 1291) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoopTail_1291 if true)
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_1291:





















s_barrier  // (op 4328) 



















v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 

v_lshrrev_b32 v10, 16, v14 // (op 52) 
v_lshrrev_b32 v11, 16, v15 // (op 52) 
v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 


v_pack_b32_f16 v4, v10, v11 // (op 54) 
v_pack_b32_f16 v5, v12, v13 // (op 54) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 54) Store value


v_accvgpr_read v10, a4 // (op 1621) 
v_accvgpr_read v11, a5 // (op 1621) 
v_accvgpr_read v12, a6 // (op 1621) 
v_accvgpr_read v13, a7 // (op 1621) 
v_mul_f32 v14, v9, v10 // (op 1621) 
v_mul_f32 v15, v9, v11 // (op 1621) 
v_mul_f32 v16, v9, v12 // (op 1621) 
v_mul_f32 v17, v9, v13 // (op 1621) 

v_lshrrev_b32 v10, 16, v14 // (op 1623) 
v_lshrrev_b32 v11, 16, v15 // (op 1623) 
v_lshrrev_b32 v12, 16, v16 // (op 1623) 
v_lshrrev_b32 v13, 16, v17 // (op 1623) 


v_pack_b32_f16 v4, v10, v11 // (op 1624) 
v_pack_b32_f16 v5, v12, v13 // (op 1624) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:64 // (op 1624) Store value


v_accvgpr_read v10, a8 // (op 1641) 
v_accvgpr_read v11, a9 // (op 1641) 
v_accvgpr_read v12, a10 // (op 1641) 
v_accvgpr_read v13, a11 // (op 1641) 
v_mul_f32 v14, v9, v10 // (op 1641) 
v_mul_f32 v15, v9, v11 // (op 1641) 
v_mul_f32 v16, v9, v12 // (op 1641) 
v_mul_f32 v17, v9, v13 // (op 1641) 

v_lshrrev_b32 v10, 16, v14 // (op 1643) 
v_lshrrev_b32 v11, 16, v15 // (op 1643) 
v_lshrrev_b32 v12, 16, v16 // (op 1643) 
v_lshrrev_b32 v13, 16, v17 // (op 1643) 


v_pack_b32_f16 v4, v10, v11 // (op 1644) 
v_pack_b32_f16 v5, v12, v13 // (op 1644) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:128 // (op 1644) Store value


v_accvgpr_read v10, a12 // (op 1661) 
v_accvgpr_read v11, a13 // (op 1661) 
v_accvgpr_read v12, a14 // (op 1661) 
v_accvgpr_read v13, a15 // (op 1661) 
v_mul_f32 v14, v9, v10 // (op 1661) 
v_mul_f32 v15, v9, v11 // (op 1661) 
v_mul_f32 v16, v9, v12 // (op 1661) 
v_mul_f32 v17, v9, v13 // (op 1661) 

v_lshrrev_b32 v10, 16, v14 // (op 1663) 
v_lshrrev_b32 v11, 16, v15 // (op 1663) 
v_lshrrev_b32 v12, 16, v16 // (op 1663) 
v_lshrrev_b32 v13, 16, v17 // (op 1663) 


v_pack_b32_f16 v4, v10, v11 // (op 1664) 
v_pack_b32_f16 v5, v12, v13 // (op 1664) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:192 // (op 1664) Store value


v_accvgpr_read v10, a16 // (op 1681) 
v_accvgpr_read v11, a17 // (op 1681) 
v_accvgpr_read v12, a18 // (op 1681) 
v_accvgpr_read v13, a19 // (op 1681) 
v_mul_f32 v14, v9, v10 // (op 1681) 
v_mul_f32 v15, v9, v11 // (op 1681) 
v_mul_f32 v16, v9, v12 // (op 1681) 
v_mul_f32 v17, v9, v13 // (op 1681) 

v_lshrrev_b32 v10, 16, v14 // (op 1683) 
v_lshrrev_b32 v11, 16, v15 // (op 1683) 
v_lshrrev_b32 v12, 16, v16 // (op 1683) 
v_lshrrev_b32 v13, 16, v17 // (op 1683) 


v_pack_b32_f16 v4, v10, v11 // (op 1684) 
v_pack_b32_f16 v5, v12, v13 // (op 1684) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:256 // (op 1684) Store value


v_accvgpr_read v10, a20 // (op 1701) 
v_accvgpr_read v11, a21 // (op 1701) 
v_accvgpr_read v12, a22 // (op 1701) 
v_accvgpr_read v13, a23 // (op 1701) 
v_mul_f32 v14, v9, v10 // (op 1701) 
v_mul_f32 v15, v9, v11 // (op 1701) 
v_mul_f32 v16, v9, v12 // (op 1701) 
v_mul_f32 v17, v9, v13 // (op 1701) 

v_lshrrev_b32 v10, 16, v14 // (op 1703) 
v_lshrrev_b32 v11, 16, v15 // (op 1703) 
v_lshrrev_b32 v12, 16, v16 // (op 1703) 
v_lshrrev_b32 v13, 16, v17 // (op 1703) 


v_pack_b32_f16 v4, v10, v11 // (op 1704) 
v_pack_b32_f16 v5, v12, v13 // (op 1704) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:320 // (op 1704) Store value


v_accvgpr_read v10, a24 // (op 1721) 
v_accvgpr_read v11, a25 // (op 1721) 
v_accvgpr_read v12, a26 // (op 1721) 
v_accvgpr_read v13, a27 // (op 1721) 
v_mul_f32 v14, v9, v10 // (op 1721) 
v_mul_f32 v15, v9, v11 // (op 1721) 
v_mul_f32 v16, v9, v12 // (op 1721) 
v_mul_f32 v17, v9, v13 // (op 1721) 

v_lshrrev_b32 v10, 16, v14 // (op 1723) 
v_lshrrev_b32 v11, 16, v15 // (op 1723) 
v_lshrrev_b32 v12, 16, v16 // (op 1723) 
v_lshrrev_b32 v13, 16, v17 // (op 1723) 


v_pack_b32_f16 v4, v10, v11 // (op 1724) 
v_pack_b32_f16 v5, v12, v13 // (op 1724) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:384 // (op 1724) Store value


v_accvgpr_read v10, a28 // (op 1741) 
v_accvgpr_read v11, a29 // (op 1741) 
v_accvgpr_read v12, a30 // (op 1741) 
v_accvgpr_read v13, a31 // (op 1741) 
v_mul_f32 v14, v9, v10 // (op 1741) 
v_mul_f32 v15, v9, v11 // (op 1741) 
v_mul_f32 v16, v9, v12 // (op 1741) 
v_mul_f32 v17, v9, v13 // (op 1741) 

v_lshrrev_b32 v10, 16, v14 // (op 1743) 
v_lshrrev_b32 v11, 16, v15 // (op 1743) 
v_lshrrev_b32 v12, 16, v16 // (op 1743) 
v_lshrrev_b32 v13, 16, v17 // (op 1743) 


v_pack_b32_f16 v4, v10, v11 // (op 1744) 
v_pack_b32_f16 v5, v12, v13 // (op 1744) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:448 // (op 1744) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[4:5], v6, s[52:55], 0 offen offset:448 // (op 1924) Load value
v_accvgpr_read v10, a28 // (op 1920) 
v_accvgpr_read v11, a29 // (op 1920) 
v_accvgpr_read v12, a30 // (op 1920) 
v_accvgpr_read v13, a31 // (op 1920) 
v_mul_f32 v14, v9, v10 // (op 1920) 
v_mul_f32 v15, v9, v11 // (op 1920) 
v_mul_f32 v16, v9, v12 // (op 1920) 
v_mul_f32 v17, v9, v13 // (op 1920) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 1925) 
v_lshrrev_b32 v11, 16, v4 // (op 1925) 
v_lshlrev_b32 v10, 16, v10 // (op 1925) 
v_lshlrev_b32 v11, 16, v11 // (op 1925) 
v_mul_f32 v18, v8, v10 // (op 1925) 
v_mul_f32 v19, v8, v11 // (op 1925) 
v_and_b32 v10, 65535, v5 // (op 1925) 
v_lshrrev_b32 v11, 16, v5 // (op 1925) 
v_lshlrev_b32 v10, 16, v10 // (op 1925) 
v_lshlrev_b32 v11, 16, v11 // (op 1925) 
v_mul_f32 v20, v8, v10 // (op 1925) 
v_mul_f32 v21, v8, v11 // (op 1925) 


v_add_f32 v10, v18, v14 // (op 1921) 
v_add_f32 v11, v19, v15 // (op 1921) 
v_add_f32 v12, v20, v16 // (op 1921) 
v_add_f32 v13, v21, v17 // (op 1921) 


v_lshrrev_b32 v14, 16, v10 // (op 1922) 
v_lshrrev_b32 v15, 16, v11 // (op 1922) 
v_lshrrev_b32 v16, 16, v12 // (op 1922) 
v_lshrrev_b32 v17, 16, v13 // (op 1922) 


v_pack_b32_f16 v4, v14, v15 // (op 1923) 
v_pack_b32_f16 v5, v16, v17 // (op 1923) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:448 // (op 1923) Store value





buffer_load_dwordx2 v[4:5], v6, s[52:55], 0 offen offset:384 // (op 1904) Load value
v_accvgpr_read v10, a24 // (op 1900) 
v_accvgpr_read v11, a25 // (op 1900) 
v_accvgpr_read v12, a26 // (op 1900) 
v_accvgpr_read v13, a27 // (op 1900) 
v_mul_f32 v14, v9, v10 // (op 1900) 
v_mul_f32 v15, v9, v11 // (op 1900) 
v_mul_f32 v16, v9, v12 // (op 1900) 
v_mul_f32 v17, v9, v13 // (op 1900) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 1905) 
v_lshrrev_b32 v11, 16, v4 // (op 1905) 
v_lshlrev_b32 v10, 16, v10 // (op 1905) 
v_lshlrev_b32 v11, 16, v11 // (op 1905) 
v_mul_f32 v18, v8, v10 // (op 1905) 
v_mul_f32 v19, v8, v11 // (op 1905) 
v_and_b32 v10, 65535, v5 // (op 1905) 
v_lshrrev_b32 v11, 16, v5 // (op 1905) 
v_lshlrev_b32 v10, 16, v10 // (op 1905) 
v_lshlrev_b32 v11, 16, v11 // (op 1905) 
v_mul_f32 v20, v8, v10 // (op 1905) 
v_mul_f32 v21, v8, v11 // (op 1905) 


v_add_f32 v10, v18, v14 // (op 1901) 
v_add_f32 v11, v19, v15 // (op 1901) 
v_add_f32 v12, v20, v16 // (op 1901) 
v_add_f32 v13, v21, v17 // (op 1901) 


v_lshrrev_b32 v14, 16, v10 // (op 1902) 
v_lshrrev_b32 v15, 16, v11 // (op 1902) 
v_lshrrev_b32 v16, 16, v12 // (op 1902) 
v_lshrrev_b32 v17, 16, v13 // (op 1902) 


v_pack_b32_f16 v4, v14, v15 // (op 1903) 
v_pack_b32_f16 v5, v16, v17 // (op 1903) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:384 // (op 1903) Store value





buffer_load_dwordx2 v[4:5], v6, s[52:55], 0 offen offset:320 // (op 1884) Load value
v_accvgpr_read v10, a20 // (op 1880) 
v_accvgpr_read v11, a21 // (op 1880) 
v_accvgpr_read v12, a22 // (op 1880) 
v_accvgpr_read v13, a23 // (op 1880) 
v_mul_f32 v14, v9, v10 // (op 1880) 
v_mul_f32 v15, v9, v11 // (op 1880) 
v_mul_f32 v16, v9, v12 // (op 1880) 
v_mul_f32 v17, v9, v13 // (op 1880) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 1885) 
v_lshrrev_b32 v11, 16, v4 // (op 1885) 
v_lshlrev_b32 v10, 16, v10 // (op 1885) 
v_lshlrev_b32 v11, 16, v11 // (op 1885) 
v_mul_f32 v18, v8, v10 // (op 1885) 
v_mul_f32 v19, v8, v11 // (op 1885) 
v_and_b32 v10, 65535, v5 // (op 1885) 
v_lshrrev_b32 v11, 16, v5 // (op 1885) 
v_lshlrev_b32 v10, 16, v10 // (op 1885) 
v_lshlrev_b32 v11, 16, v11 // (op 1885) 
v_mul_f32 v20, v8, v10 // (op 1885) 
v_mul_f32 v21, v8, v11 // (op 1885) 


v_add_f32 v10, v18, v14 // (op 1881) 
v_add_f32 v11, v19, v15 // (op 1881) 
v_add_f32 v12, v20, v16 // (op 1881) 
v_add_f32 v13, v21, v17 // (op 1881) 


v_lshrrev_b32 v14, 16, v10 // (op 1882) 
v_lshrrev_b32 v15, 16, v11 // (op 1882) 
v_lshrrev_b32 v16, 16, v12 // (op 1882) 
v_lshrrev_b32 v17, 16, v13 // (op 1882) 


v_pack_b32_f16 v4, v14, v15 // (op 1883) 
v_pack_b32_f16 v5, v16, v17 // (op 1883) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:320 // (op 1883) Store value





buffer_load_dwordx2 v[4:5], v6, s[52:55], 0 offen offset:256 // (op 1864) Load value
v_accvgpr_read v10, a16 // (op 1860) 
v_accvgpr_read v11, a17 // (op 1860) 
v_accvgpr_read v12, a18 // (op 1860) 
v_accvgpr_read v13, a19 // (op 1860) 
v_mul_f32 v14, v9, v10 // (op 1860) 
v_mul_f32 v15, v9, v11 // (op 1860) 
v_mul_f32 v16, v9, v12 // (op 1860) 
v_mul_f32 v17, v9, v13 // (op 1860) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 1865) 
v_lshrrev_b32 v11, 16, v4 // (op 1865) 
v_lshlrev_b32 v10, 16, v10 // (op 1865) 
v_lshlrev_b32 v11, 16, v11 // (op 1865) 
v_mul_f32 v18, v8, v10 // (op 1865) 
v_mul_f32 v19, v8, v11 // (op 1865) 
v_and_b32 v10, 65535, v5 // (op 1865) 
v_lshrrev_b32 v11, 16, v5 // (op 1865) 
v_lshlrev_b32 v10, 16, v10 // (op 1865) 
v_lshlrev_b32 v11, 16, v11 // (op 1865) 
v_mul_f32 v20, v8, v10 // (op 1865) 
v_mul_f32 v21, v8, v11 // (op 1865) 


v_add_f32 v10, v18, v14 // (op 1861) 
v_add_f32 v11, v19, v15 // (op 1861) 
v_add_f32 v12, v20, v16 // (op 1861) 
v_add_f32 v13, v21, v17 // (op 1861) 


v_lshrrev_b32 v14, 16, v10 // (op 1862) 
v_lshrrev_b32 v15, 16, v11 // (op 1862) 
v_lshrrev_b32 v16, 16, v12 // (op 1862) 
v_lshrrev_b32 v17, 16, v13 // (op 1862) 


v_pack_b32_f16 v4, v14, v15 // (op 1863) 
v_pack_b32_f16 v5, v16, v17 // (op 1863) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:256 // (op 1863) Store value





buffer_load_dwordx2 v[4:5], v6, s[52:55], 0 offen offset:192 // (op 1844) Load value
v_accvgpr_read v10, a12 // (op 1840) 
v_accvgpr_read v11, a13 // (op 1840) 
v_accvgpr_read v12, a14 // (op 1840) 
v_accvgpr_read v13, a15 // (op 1840) 
v_mul_f32 v14, v9, v10 // (op 1840) 
v_mul_f32 v15, v9, v11 // (op 1840) 
v_mul_f32 v16, v9, v12 // (op 1840) 
v_mul_f32 v17, v9, v13 // (op 1840) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 1845) 
v_lshrrev_b32 v11, 16, v4 // (op 1845) 
v_lshlrev_b32 v10, 16, v10 // (op 1845) 
v_lshlrev_b32 v11, 16, v11 // (op 1845) 
v_mul_f32 v18, v8, v10 // (op 1845) 
v_mul_f32 v19, v8, v11 // (op 1845) 
v_and_b32 v10, 65535, v5 // (op 1845) 
v_lshrrev_b32 v11, 16, v5 // (op 1845) 
v_lshlrev_b32 v10, 16, v10 // (op 1845) 
v_lshlrev_b32 v11, 16, v11 // (op 1845) 
v_mul_f32 v20, v8, v10 // (op 1845) 
v_mul_f32 v21, v8, v11 // (op 1845) 


v_add_f32 v10, v18, v14 // (op 1841) 
v_add_f32 v11, v19, v15 // (op 1841) 
v_add_f32 v12, v20, v16 // (op 1841) 
v_add_f32 v13, v21, v17 // (op 1841) 


v_lshrrev_b32 v14, 16, v10 // (op 1842) 
v_lshrrev_b32 v15, 16, v11 // (op 1842) 
v_lshrrev_b32 v16, 16, v12 // (op 1842) 
v_lshrrev_b32 v17, 16, v13 // (op 1842) 


v_pack_b32_f16 v4, v14, v15 // (op 1843) 
v_pack_b32_f16 v5, v16, v17 // (op 1843) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:192 // (op 1843) Store value





buffer_load_dwordx2 v[4:5], v6, s[52:55], 0 offen offset:128 // (op 1824) Load value
v_accvgpr_read v10, a8 // (op 1820) 
v_accvgpr_read v11, a9 // (op 1820) 
v_accvgpr_read v12, a10 // (op 1820) 
v_accvgpr_read v13, a11 // (op 1820) 
v_mul_f32 v14, v9, v10 // (op 1820) 
v_mul_f32 v15, v9, v11 // (op 1820) 
v_mul_f32 v16, v9, v12 // (op 1820) 
v_mul_f32 v17, v9, v13 // (op 1820) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 1825) 
v_lshrrev_b32 v11, 16, v4 // (op 1825) 
v_lshlrev_b32 v10, 16, v10 // (op 1825) 
v_lshlrev_b32 v11, 16, v11 // (op 1825) 
v_mul_f32 v18, v8, v10 // (op 1825) 
v_mul_f32 v19, v8, v11 // (op 1825) 
v_and_b32 v10, 65535, v5 // (op 1825) 
v_lshrrev_b32 v11, 16, v5 // (op 1825) 
v_lshlrev_b32 v10, 16, v10 // (op 1825) 
v_lshlrev_b32 v11, 16, v11 // (op 1825) 
v_mul_f32 v20, v8, v10 // (op 1825) 
v_mul_f32 v21, v8, v11 // (op 1825) 


v_add_f32 v10, v18, v14 // (op 1821) 
v_add_f32 v11, v19, v15 // (op 1821) 
v_add_f32 v12, v20, v16 // (op 1821) 
v_add_f32 v13, v21, v17 // (op 1821) 


v_lshrrev_b32 v14, 16, v10 // (op 1822) 
v_lshrrev_b32 v15, 16, v11 // (op 1822) 
v_lshrrev_b32 v16, 16, v12 // (op 1822) 
v_lshrrev_b32 v17, 16, v13 // (op 1822) 


v_pack_b32_f16 v4, v14, v15 // (op 1823) 
v_pack_b32_f16 v5, v16, v17 // (op 1823) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:128 // (op 1823) Store value





buffer_load_dwordx2 v[4:5], v6, s[52:55], 0 offen offset:64 // (op 1804) Load value
v_accvgpr_read v10, a4 // (op 1800) 
v_accvgpr_read v11, a5 // (op 1800) 
v_accvgpr_read v12, a6 // (op 1800) 
v_accvgpr_read v13, a7 // (op 1800) 
v_mul_f32 v14, v9, v10 // (op 1800) 
v_mul_f32 v15, v9, v11 // (op 1800) 
v_mul_f32 v16, v9, v12 // (op 1800) 
v_mul_f32 v17, v9, v13 // (op 1800) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 1805) 
v_lshrrev_b32 v11, 16, v4 // (op 1805) 
v_lshlrev_b32 v10, 16, v10 // (op 1805) 
v_lshlrev_b32 v11, 16, v11 // (op 1805) 
v_mul_f32 v18, v8, v10 // (op 1805) 
v_mul_f32 v19, v8, v11 // (op 1805) 
v_and_b32 v10, 65535, v5 // (op 1805) 
v_lshrrev_b32 v11, 16, v5 // (op 1805) 
v_lshlrev_b32 v10, 16, v10 // (op 1805) 
v_lshlrev_b32 v11, 16, v11 // (op 1805) 
v_mul_f32 v20, v8, v10 // (op 1805) 
v_mul_f32 v21, v8, v11 // (op 1805) 


v_add_f32 v10, v18, v14 // (op 1801) 
v_add_f32 v11, v19, v15 // (op 1801) 
v_add_f32 v12, v20, v16 // (op 1801) 
v_add_f32 v13, v21, v17 // (op 1801) 


v_lshrrev_b32 v14, 16, v10 // (op 1802) 
v_lshrrev_b32 v15, 16, v11 // (op 1802) 
v_lshrrev_b32 v16, 16, v12 // (op 1802) 
v_lshrrev_b32 v17, 16, v13 // (op 1802) 


v_pack_b32_f16 v4, v14, v15 // (op 1803) 
v_pack_b32_f16 v5, v16, v17 // (op 1803) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen offset:64 // (op 1803) Store value





buffer_load_dwordx2 v[4:5], v6, s[52:55], 0 offen // (op 138) Load value
v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 139) 
v_lshrrev_b32 v11, 16, v4 // (op 139) 
v_lshlrev_b32 v10, 16, v10 // (op 139) 
v_lshlrev_b32 v11, 16, v11 // (op 139) 
v_mul_f32 v18, v8, v10 // (op 139) 
v_mul_f32 v19, v8, v11 // (op 139) 
v_and_b32 v10, 65535, v5 // (op 139) 
v_lshrrev_b32 v11, 16, v5 // (op 139) 
v_lshlrev_b32 v10, 16, v10 // (op 139) 
v_lshlrev_b32 v11, 16, v11 // (op 139) 
v_mul_f32 v20, v8, v10 // (op 139) 
v_mul_f32 v21, v8, v11 // (op 139) 


v_add_f32 v10, v18, v14 // (op 135) 
v_add_f32 v11, v19, v15 // (op 135) 
v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 


v_lshrrev_b32 v14, 16, v10 // (op 136) 
v_lshrrev_b32 v15, 16, v11 // (op 136) 
v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 


v_pack_b32_f16 v4, v14, v15 // (op 137) 
v_pack_b32_f16 v5, v16, v17 // (op 137) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 148
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 71680 // lds bytes
  .amdhsa_accum_offset 116
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
    DestDataType: b
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
    macrotile: [256, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 71680
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 116
    .agpr_count: 32
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
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

