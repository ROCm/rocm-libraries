

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM_,@function
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 3034) 
v_and_b32 v3, 1, v0 // (op 3034) 

v_and_b32 v0, 15, v1 // (op 3034) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3034) 


v_and_b32 v0, 63, v1 // (op 3034) 
v_lshrrev_b32 v3, 4, v0 // (op 3034) 

v_lshlrev_b32 v0, 5, v3 // (op 3034) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 3034) 


v_lshlrev_b32 v0, 2, v3 // (op 3034) 

v_lshrrev_b32 v3, 3, v0 // (op 3034) 

v_lshrrev_b32 v0, 6, v1 // (op 3163) 
v_and_b32 v4, 1, v0 // (op 3163) 

v_and_b32 v0, 15, v1 // (op 3163) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 3163) 


v_and_b32 v0, 63, v1 // (op 3163) 
v_lshrrev_b32 v4, 4, v0 // (op 3163) 

v_lshlrev_b32 v0, 5, v4 // (op 3163) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 3163) 


v_lshlrev_b32 v0, 2, v4 // (op 3163) 

v_lshrrev_b32 v4, 3, v0 // (op 3163) 

v_lshrrev_b32 v0, 7, v1 // (op 2919) 
v_lshrrev_b32 v5, 7, v1 // (op 2981) 
v_and_b32 v6, 15, v1 // (op 2981) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 2981) 


v_and_b32 v5, 63, v1 // (op 2981) 
v_lshrrev_b32 v6, 4, v5 // (op 2981) 

v_lshlrev_b32 v5, 5, v6 // (op 2981) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 2981) 


v_lshlrev_b32 v5, 2, v6 // (op 2981) 

v_lshrrev_b32 v6, 3, v5 // (op 2981) 

v_lshrrev_b32 v5, 7, v1 // (op 3092) 
v_and_b32 v7, 15, v1 // (op 3092) 
v_lshl_add_u32 v8, v5, 4, v7 // (op 3092) 


v_and_b32 v5, 63, v1 // (op 3092) 
v_lshrrev_b32 v7, 4, v5 // (op 3092) 

v_lshl_add_u32 v5, v8, 2, v7 // (op 3092) 


v_lshrrev_b32 v7, 7, v1 // (op 3108) 
v_lshrrev_b32 v8, 7, v1 // (op 3139) 
v_and_b32 v9, 15, v1 // (op 3139) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 3139) 


v_and_b32 v8, 63, v1 // (op 3139) 
v_lshrrev_b32 v9, 4, v8 // (op 3139) 

v_lshlrev_b32 v8, 5, v9 // (op 3139) 

v_lshl_add_u32 v9, v10, 7, v8 // (op 3139) 


v_lshlrev_b32 v8, 2, v9 // (op 3139) 

v_lshrrev_b32 v9, 3, v8 // (op 3139) 

v_lshrrev_b32 v8, 7, v1 // (op 3192) 
v_and_b32 v10, 15, v1 // (op 3192) 
v_lshl_add_u32 v11, v8, 4, v10 // (op 3192) 


v_and_b32 v8, 63, v1 // (op 3192) 
v_lshrrev_b32 v10, 4, v8 // (op 3192) 

v_lshl_add_u32 v8, v11, 2, v10 // (op 3192) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s0, s16, s47 // (op 2919) 
s_add_i32 s1, s0, s16 // (op 2919) 

s_mul_hi_i32 s0, s16, s47 // (op 3108) 
s_add_i32 s18, s0, s16 // (op 3108) 

v_and_b32 v10, 63, v1 // (op 2919) 
s_ashr_i32 s0, s1, 31 // (op 2919) 
v_lshrrev_b32 v11, 4, v10 // (op 2919) 
s_cmp_eq_i32 s47, 0 // (op 2919) 
s_mov_b32 s19, scc // (op 2919) 
v_lshlrev_b32 v12, 2, v11 // (op 2919) 

s_cmp_eq_i32 s19, 0 // (op 2919) 
s_cselect_b32 s51, 0, -1 // (op 2919) 

v_lshl_add_u32 v11, v0, 4, v12 // (op 2919) 


s_and_b32 s19, s11, 31 // (op 2919) 
v_lshrrev_b32 v0, 2, v10 // (op 2919) 

v_and_b32 v10, 3, v0 // (op 2919) 

v_and_b32 v0, 63, v1 // (op 3108) 
v_lshrrev_b32 v12, 4, v0 // (op 3108) 
s_lshl_b32 s52, 1, s19 // (op 2919) 
v_and_b32 v13, 3, v1 // (op 2919) 
s_add_i32 s53, s52, s51 // (op 2919) 


s_and_b32 s51, s0, s53 // (op 2919) 


s_ashr_i32 s0, s18, 31 // (op 3108) 
s_cmp_eq_i32 s47, 0 // (op 3108) 
s_mov_b32 s52, scc // (op 3108) 
v_lshl_add_u32 v14, v10, 2, v13 // (op 2919) 


s_add_i32 s53, s1, s51 // (op 2919) 


v_lshrrev_b32 v10, 6, v1 // (op 2919) 
s_ashr_i32 s1, s53, s19 // (op 2919) 


v_and_b32 v13, 1, v10 // (op 2919) 

s_lshl_b32 s19, s11, 24 // (op 2919) 
v_lshl_add_u32 v10, v13, 4, v14 // (op 2919) 


s_ashr_i32 s51, s19, 31 // (op 2919) 

s_xor_b32 s19, s1, s51 // (op 2919) 

s_sub_i32 s1, s19, s51 // (op 2919) 


s_mul_hi_i32 s19, s1, s12 // (op 2919) 
s_add_i32 s51, s19, s1 // (op 2919) 

s_ashr_i32 s19, s51, 31 // (op 2919) 
s_cmp_eq_i32 s12, 0 // (op 2919) 
s_mov_b32 s53, scc // (op 2919) 
s_cmp_eq_i32 s53, 0 // (op 2919) 
s_cselect_b32 s54, 0, -1 // (op 2919) 

s_and_b32 s53, s13, 31 // (op 2919) 
v_lshlrev_b32 v13, 2, v12 // (op 3108) 

s_cmp_eq_i32 s52, 0 // (op 3108) 
s_cselect_b32 s55, 0, -1 // (op 3108) 

s_lshl_b32 s52, 1, s53 // (op 2919) 
v_lshl_add_u32 v12, v7, 4, v13 // (op 3108) 


v_lshrrev_b32 v7, 2, v0 // (op 3108) 

s_add_i32 s56, s52, s54 // (op 2919) 


s_and_b32 s52, s19, s56 // (op 2919) 


s_add_i32 s19, s51, s52 // (op 2919) 


s_ashr_i32 s54, s19, s53 // (op 2919) 


s_lshl_b32 s19, s13, 24 // (op 2919) 
s_ashr_i32 s51, s19, 31 // (op 2919) 

s_xor_b32 s19, s54, s51 // (op 2919) 

s_sub_i32 s52, s19, s51 // (op 2919) 


s_mul_i32 s19, s1, s48 // (op 2919) 
s_sub_u32 s51, s16, s19 // (op 2919) 

s_mul_i32 s19, s52, s48 // (op 2919) 
s_add_u32 s19, s19, s51 // (op 2919) 

s_mul_hi_i32 s51, s16, s8 // (op 2919) 
s_add_i32 s53, s51, s16 // (op 2919) 

s_ashr_i32 s51, s53, 31 // (op 2919) 
s_cmp_eq_i32 s8, 0 // (op 2919) 
s_mov_b32 s54, scc // (op 2919) 
s_cmp_eq_i32 s54, 0 // (op 2919) 
s_cselect_b32 s56, 0, -1 // (op 2919) 

s_and_b32 s54, s10, 31 // (op 2919) 
s_lshl_b32 s57, 1, s54 // (op 2919) 
s_add_i32 s58, s57, s56 // (op 2919) 


s_and_b32 s56, s51, s58 // (op 2919) 


s_add_i32 s51, s53, s56 // (op 2919) 


s_ashr_i32 s53, s51, s54 // (op 2919) 


s_lshl_b32 s51, s10, 24 // (op 2919) 
s_ashr_i32 s54, s51, 31 // (op 2919) 

s_xor_b32 s51, s53, s54 // (op 2919) 

s_sub_i32 s53, s51, s54 // (op 2919) 


s_and_b32 s51, s11, 31 // (op 3108) 
s_lshl_b32 s54, 1, s51 // (op 3108) 
v_and_b32 v0, 3, v7 // (op 3108) 

s_add_i32 s56, s54, s55 // (op 3108) 


v_and_b32 v7, 3, v1 // (op 3108) 
s_and_b32 s54, s0, s56 // (op 3108) 


v_lshl_add_u32 v13, v0, 2, v7 // (op 3108) 


s_add_i32 s0, s18, s54 // (op 3108) 


v_lshrrev_b32 v0, 6, v1 // (op 3108) 
s_ashr_i32 s18, s0, s51 // (op 3108) 


v_and_b32 v7, 1, v0 // (op 3108) 

s_lshl_b32 s0, s11, 24 // (op 3108) 
v_lshl_add_u32 v0, v7, 4, v13 // (op 3108) 


s_ashr_i32 s51, s0, 31 // (op 3108) 

s_xor_b32 s0, s18, s51 // (op 3108) 

s_sub_i32 s18, s0, s51 // (op 3108) 


s_mul_hi_i32 s0, s18, s12 // (op 3108) 
s_add_i32 s51, s0, s18 // (op 3108) 

s_ashr_i32 s0, s51, 31 // (op 3108) 
s_cmp_eq_i32 s12, 0 // (op 3108) 
s_mov_b32 s54, scc // (op 3108) 
s_cmp_eq_i32 s54, 0 // (op 3108) 
s_cselect_b32 s55, 0, -1 // (op 3108) 

s_and_b32 s54, s13, 31 // (op 3108) 
s_lshl_b32 s56, 1, s54 // (op 3108) 
s_add_i32 s57, s56, s55 // (op 3108) 


s_and_b32 s55, s0, s57 // (op 3108) 


s_add_i32 s0, s51, s55 // (op 3108) 


s_ashr_i32 s51, s0, s54 // (op 3108) 


s_lshl_b32 s0, s13, 24 // (op 3108) 
s_ashr_i32 s54, s0, 31 // (op 3108) 

s_xor_b32 s0, s51, s54 // (op 3108) 

s_sub_i32 s51, s0, s54 // (op 3108) 


s_mul_i32 s0, s18, s48 // (op 3108) 
s_sub_u32 s54, s16, s0 // (op 3108) 

s_mul_i32 s0, s51, s48 // (op 3108) 
s_add_u32 s0, s0, s54 // (op 3108) 

s_mul_hi_i32 s54, s16, s8 // (op 3108) 
s_add_i32 s55, s54, s16 // (op 3108) 

s_ashr_i32 s54, s55, 31 // (op 3108) 
s_cmp_eq_i32 s8, 0 // (op 3108) 
s_mov_b32 s56, scc // (op 3108) 
s_cmp_eq_i32 s56, 0 // (op 3108) 
s_cselect_b32 s57, 0, -1 // (op 3108) 

s_and_b32 s56, s10, 31 // (op 3108) 
s_lshl_b32 s58, 1, s56 // (op 3108) 
s_add_i32 s59, s58, s57 // (op 3108) 


s_and_b32 s57, s54, s59 // (op 3108) 


s_add_i32 s54, s55, s57 // (op 3108) 


s_ashr_i32 s55, s54, s56 // (op 3108) 


s_lshl_b32 s54, s10, 24 // (op 3108) 
s_ashr_i32 s56, s54, 31 // (op 3108) 

s_xor_b32 s54, s55, s56 // (op 3108) 

s_sub_i32 s55, s54, s56 // (op 3108) 


s_add_u32 s56, s6, 32 // (op 2919) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 2919) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2919) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2919) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 2919) 

s_mul_i32 s54, s48, s56 // (op 2919) 
s_mul_i32 s58, s14, s54 // (op 2919) 

s_mul_i32 s54, s53, s58 // (op 2919) 

s_sub_u32 s58, s16, s54 // (op 2919) 

s_mov_b32 s54, s58 // (op 2919) 
s_mul_hi_i32 s59, s54, s15 // (op 2919) 
s_add_i32 s60, s59, s54 // (op 2919) 


s_ashr_i32 s54, s60, 31 // (op 2919) 
s_cmp_eq_i32 s15, 0 // (op 2919) 
s_mov_b32 s59, scc // (op 2919) 
s_cmp_eq_i32 s59, 0 // (op 2919) 
s_cselect_b32 s61, 0, -1 // (op 2919) 

s_and_b32 s59, s9, 31 // (op 2919) 
s_lshl_b32 s62, 1, s59 // (op 2919) 
s_add_i32 s63, s62, s61 // (op 2919) 


s_and_b32 s61, s54, s63 // (op 2919) 


s_add_i32 s54, s60, s61 // (op 2919) 


s_ashr_i32 s60, s54, s59 // (op 2919) 


s_lshl_b32 s54, s9, 24 // (op 2919) 
s_ashr_i32 s59, s54, 31 // (op 2919) 

s_xor_b32 s54, s60, s59 // (op 2919) 

s_sub_i32 s60, s54, s59 // (op 2919) 


s_mul_i32 s54, s46, s48 // (op 2919) 
s_sub_i32 s59, s4, s54 // (op 2919) 
s_mul_i32 s61, s60, s59 // (op 2919) 

s_sub_u32 s59, s58, s61 // (op 2919) 


s_add_u32 s58, s59, s54 // (op 2919) 


s_cmp_eq_i32 s53, 0 // (op 2919) 
s_mov_b32 s54, scc // (op 2919) 

s_cmp_eq_i32 s54, 0 // (op 2919) 
s_cselect_b32 s53, s58, s19 // (op 2919) 


s_mul_i32 s19, s52, s56 // (op 2919) 


v_mov_b32 v7, 192 // (op 2919) 
s_sub_i32 s52, s1, s19 // (op 2919) 


v_mul_lo_u32 v13, s53, v7 // (op 2919) 

s_cmp_eq_i32 s54, 0 // (op 2919) 
s_cselect_b32 s1, s60, s52 // (op 2919) 



v_add_u32 v13, v13, v11 // (op 2919) 


v_lshl_add_u32 v7, s1, 5, v10 // (op 2919) 


v_mul_lo_u32 v10, v7, s2 // (op 2919) 

v_add_lshl_u32 v7, v13, v10, 1 // (op 2919) 


s_add_u32 s52, s6, 32 // (op 3108) Start of Int64 add, locking SCC
s_addc_u32 s53, s7, 0 // (op 3108) End of Int64 add, unlocking SCC
s_sub_u32 s56, s52, 1 // (op 3108) least significant half; sets scc
s_subb_u32 s57, s53, 0 // (op 3108) most significant half; uses scc

s_ashr_i64 s[52:53], s[56:57], 5 // (op 3108) 

s_mul_i32 s1, s48, s52 // (op 3108) 
s_mul_i32 s19, s14, s1 // (op 3108) 

s_mul_i32 s1, s55, s19 // (op 3108) 

s_sub_u32 s19, s16, s1 // (op 3108) 

s_mov_b32 s1, s19 // (op 3108) 
s_mul_hi_i32 s54, s1, s15 // (op 3108) 
s_add_i32 s56, s54, s1 // (op 3108) 


s_ashr_i32 s1, s56, 31 // (op 3108) 
s_cmp_eq_i32 s15, 0 // (op 3108) 
s_mov_b32 s54, scc // (op 3108) 
s_cmp_eq_i32 s54, 0 // (op 3108) 
s_cselect_b32 s57, 0, -1 // (op 3108) 

s_and_b32 s54, s9, 31 // (op 3108) 
s_lshl_b32 s58, 1, s54 // (op 3108) 
s_add_i32 s59, s58, s57 // (op 3108) 


s_and_b32 s57, s1, s59 // (op 3108) 


s_add_i32 s1, s56, s57 // (op 3108) 


s_ashr_i32 s56, s1, s54 // (op 3108) 


s_lshl_b32 s1, s9, 24 // (op 3108) 
s_ashr_i32 s54, s1, 31 // (op 3108) 

s_xor_b32 s1, s56, s54 // (op 3108) 

s_sub_i32 s56, s1, s54 // (op 3108) 


s_mul_i32 s1, s46, s48 // (op 3108) 
s_sub_i32 s54, s4, s1 // (op 3108) 
s_mul_i32 s57, s56, s54 // (op 3108) 

s_sub_u32 s54, s19, s57 // (op 3108) 


s_add_u32 s19, s54, s1 // (op 3108) 


s_cmp_eq_i32 s55, 0 // (op 3108) 
s_mov_b32 s1, scc // (op 3108) 

s_cmp_eq_i32 s1, 0 // (op 3108) 
s_cselect_b32 s54, s19, s0 // (op 3108) 


s_mul_i32 s0, s51, s52 // (op 3108) 


v_mov_b32 v10, 192 // (op 3108) 
s_sub_i32 s19, s18, s0 // (op 3108) 


v_mul_lo_u32 v11, s54, v10 // (op 3108) 

s_cmp_eq_i32 s1, 0 // (op 3108) 
s_cselect_b32 s0, s56, s19 // (op 3108) 



v_add_u32 v11, v11, v12 // (op 3108) 


v_lshl_add_u32 v10, s0, 5, v0 // (op 3108) 


v_mul_lo_u32 v0, v10, s20 // (op 3108) 

v_add_lshl_u32 v10, v11, v0, 1 // (op 3108) 


s_mul_i32 s1, s2, s7 // (op 2921) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 2921) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 2921) least significant
s_mul_hi_u32 s19, s2, s6 // (op 2921) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2921) most significant: sum
s_add_u32 s1, s1, s19 // (op 2921) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2921) 

s_mul_i32 s1, s20, s7 // (op 3110) most significant: low of low * high
s_mul_i32 s51, s21, s6 // (op 3110) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 3110) least significant
s_mul_hi_u32 s52, s20, s6 // (op 3110) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 3110) most significant: sum
s_add_u32 s1, s1, s52 // (op 3110) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 3110) 

s_mov_b32 s56, s22 // (op 2921) reg expression
s_mov_b32 s57, s23 // (op 2921) reg expression
s_mov_b32 s58, s18 // (op 2921) reg expression
s_mov_b32 s59, 131072 // (op 2921) call()

s_mov_b32 s60, s24 // (op 3110) reg expression

s_mov_b32 s61, s25 // (op 3110) reg expression
s_mov_b32 s62, s52 // (op 3110) reg expression
s_mov_b32 s63, 131072 // (op 3110) call()


v_mov_b32 v0, s49 // (op 2) Move value
v_mov_b32 v11, s50 // (op 39) Move value


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
s_mul_hi_i32 s0, s16, s47 // (op 2686) 
v_lshrrev_b32 v12, 2, v1 // (op 2686) 
s_add_i32 s1, s0, s16 // (op 2686) 

v_mul_lo_u32 v13, v12, 3 // (op 2686) 

s_ashr_i32 s0, s1, 31 // (op 2686) 
v_and_b32 v12, -4, v1 // (op 2686) 
s_cmp_eq_i32 s47, 0 // (op 2686) 
s_mov_b32 s18, scc // (op 2686) 
v_sub_u32 v14, v1, v12 // (op 2686) 

s_cmp_eq_i32 s18, 0 // (op 2686) 
s_cselect_b32 s19, 0, -1 // (op 2686) 

v_lshlrev_b32 v12, 5, v14 // (op 2686) 

v_mov_b32 v15, 0 // (op 2686) convert
v_mov_b32 v14, v12 // (op 2686) convert

s_and_b32 s18, s11, 31 // (op 2686) 
s_lshl_b32 s22, 1, s18 // (op 2686) 
s_add_i32 s23, s22, s19 // (op 2686) 


s_and_b32 s19, s0, s23 // (op 2686) 


s_add_i32 s0, s1, s19 // (op 2686) 


s_ashr_i32 s1, s0, s18 // (op 2686) 


s_lshl_b32 s0, s11, 24 // (op 2686) 
s_ashr_i32 s18, s0, 31 // (op 2686) 

s_xor_b32 s0, s1, s18 // (op 2686) 

s_sub_i32 s1, s0, s18 // (op 2686) 


s_mul_hi_i32 s0, s1, s12 // (op 2686) 
s_add_i32 s18, s0, s1 // (op 2686) 

s_ashr_i32 s0, s18, 31 // (op 2686) 
s_cmp_eq_i32 s12, 0 // (op 2686) 
s_mov_b32 s19, scc // (op 2686) 
s_cmp_eq_i32 s19, 0 // (op 2686) 
s_cselect_b32 s22, 0, -1 // (op 2686) 

s_and_b32 s19, s13, 31 // (op 2686) 
s_lshl_b32 s23, 1, s19 // (op 2686) 
s_add_i32 s24, s23, s22 // (op 2686) 


s_and_b32 s25, s0, s24 // (op 2686) 


s_add_i32 s0, s18, s25 // (op 2686) 


s_ashr_i32 s18, s0, s19 // (op 2686) 


s_lshl_b32 s0, s13, 24 // (op 2686) 
s_ashr_i32 s19, s0, 31 // (op 2686) 

s_xor_b32 s0, s18, s19 // (op 2686) 

s_sub_i32 s18, s0, s19 // (op 2686) 


s_mul_i32 s0, s1, s48 // (op 2686) 

s_sub_u32 s1, s16, s0 // (op 2686) 

s_mul_i32 s0, s18, s48 // (op 2686) 
s_add_u32 s0, s0, s1 // (op 2686) 


s_mul_hi_i32 s1, s16, s8 // (op 2686) 
s_add_i32 s18, s1, s16 // (op 2686) 

s_ashr_i32 s1, s18, 31 // (op 2686) 
s_cmp_eq_i32 s8, 0 // (op 2686) 
s_mov_b32 s19, scc // (op 2686) 
s_cmp_eq_i32 s19, 0 // (op 2686) 
s_cselect_b32 s22, 0, -1 // (op 2686) 

s_and_b32 s19, s10, 31 // (op 2686) 
s_lshl_b32 s23, 1, s19 // (op 2686) 
s_add_i32 s24, s23, s22 // (op 2686) 


s_and_b32 s25, s1, s24 // (op 2686) 


s_add_i32 s1, s18, s25 // (op 2686) 


s_ashr_i32 s18, s1, s19 // (op 2686) 


s_lshl_b32 s1, s10, 24 // (op 2686) 
s_ashr_i32 s19, s1, 31 // (op 2686) 

s_xor_b32 s1, s18, s19 // (op 2686) 

s_sub_i32 s18, s1, s19 // (op 2686) 


s_add_u32 s22, s6, 32 // (op 2686) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2686) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2686) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2686) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 2686) 

s_mul_i32 s1, s48, s22 // (op 2686) 

s_mul_i32 s19, s14, s1 // (op 2686) 

s_mul_i32 s1, s18, s19 // (op 2686) 

s_sub_u32 s19, s16, s1 // (op 2686) 

s_mov_b32 s1, s19 // (op 2686) 
s_mul_hi_i32 s22, s1, s15 // (op 2686) 
s_add_i32 s23, s22, s1 // (op 2686) 


s_ashr_i32 s1, s23, 31 // (op 2686) 
s_cmp_eq_i32 s15, 0 // (op 2686) 
s_mov_b32 s22, scc // (op 2686) 
s_cmp_eq_i32 s22, 0 // (op 2686) 
s_cselect_b32 s24, 0, -1 // (op 2686) 

s_and_b32 s22, s9, 31 // (op 2686) 
s_lshl_b32 s25, 1, s22 // (op 2686) 
s_add_i32 s49, s25, s24 // (op 2686) 


s_and_b32 s24, s1, s49 // (op 2686) 


s_add_i32 s1, s23, s24 // (op 2686) 


s_ashr_i32 s23, s1, s22 // (op 2686) 


s_lshl_b32 s1, s9, 24 // (op 2686) 
s_ashr_i32 s22, s1, 31 // (op 2686) 

s_xor_b32 s1, s23, s22 // (op 2686) 

s_sub_i32 s23, s1, s22 // (op 2686) 


s_mul_i32 s1, s46, s48 // (op 2686) 
s_sub_i32 s22, s4, s1 // (op 2686) 
s_mul_i32 s24, s23, s22 // (op 2686) 


s_sub_u32 s25, s19, s24 // (op 2686) 


s_add_u32 s19, s25, s1 // (op 2686) 


s_cmp_eq_i32 s18, 0 // (op 2686) 
s_mov_b32 s1, scc // (op 2686) 

s_cmp_eq_i32 s1, 0 // (op 2686) 
s_cselect_b32 s18, s19, s0 // (op 2686) 



v_mov_b32 v12, 192 // (op 2686) 
v_mul_lo_u32 v16, s18, v12 // (op 2686) 

v_add_u32 v16, v16, v13 // (op 2686) 


v_mul_lo_u32 v13, v16, s27 // (op 2686) most significant: low of low * high
v_mul_hi_u32 v17, v16, s26 // (op 2686) most significant: high of low * low
v_mul_lo_u32 v12, v16, s26 // (op 2686) least significant: low of low * low
v_add_u32 v13, v13, v17 // (op 2686) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v14 // (op 2686) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 2686) most significant half



v_lshlrev_b64 v[14:15], 2, v[12:13] // (op 2686) 

v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 2686) 

v_mov_b32 v14, v12 // (op 2686) convert

s_mul_i32 s1, s26, s31 // (op 2688) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2688) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2688) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2688) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2688) most significant: sum
s_add_u32 s1, s1, s19 // (op 2688) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2688) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2688) 

s_mov_b32 s52, s28 // (op 2688) reg expression
s_mov_b32 s53, s29 // (op 2688) reg expression
s_mov_b32 s54, s0 // (op 2688) reg expression
s_mov_b32 s55, 131072 // (op 2688) call()

s_mul_hi_i32 s0, s16, s47 // (op 2705) 
v_lshrrev_b32 v12, 2, v1 // (op 2705) 
s_add_i32 s1, s0, s16 // (op 2705) 

v_mul_lo_u32 v13, v12, 3 // (op 2705) 

s_ashr_i32 s0, s1, 31 // (op 2705) 
v_and_b32 v12, -4, v1 // (op 2705) 
s_cmp_eq_i32 s47, 0 // (op 2705) 
s_mov_b32 s18, scc // (op 2705) 
v_sub_u32 v15, v1, v12 // (op 2705) 

s_cmp_eq_i32 s18, 0 // (op 2705) 
s_cselect_b32 s19, 0, -1 // (op 2705) 

v_lshlrev_b32 v12, 5, v15 // (op 2705) 

v_mov_b32 v17, 0 // (op 2705) convert
v_mov_b32 v16, v12 // (op 2705) convert

s_and_b32 s18, s11, 31 // (op 2705) 
s_lshl_b32 s22, 1, s18 // (op 2705) 
s_add_i32 s23, s22, s19 // (op 2705) 


s_and_b32 s19, s0, s23 // (op 2705) 


s_add_i32 s0, s1, s19 // (op 2705) 


s_ashr_i32 s1, s0, s18 // (op 2705) 


s_lshl_b32 s0, s11, 24 // (op 2705) 
s_ashr_i32 s18, s0, 31 // (op 2705) 

s_xor_b32 s0, s1, s18 // (op 2705) 

s_sub_i32 s1, s0, s18 // (op 2705) 


s_mul_hi_i32 s0, s1, s12 // (op 2705) 
s_add_i32 s18, s0, s1 // (op 2705) 

s_ashr_i32 s0, s18, 31 // (op 2705) 
s_cmp_eq_i32 s12, 0 // (op 2705) 
s_mov_b32 s19, scc // (op 2705) 
s_cmp_eq_i32 s19, 0 // (op 2705) 
s_cselect_b32 s22, 0, -1 // (op 2705) 

s_and_b32 s19, s13, 31 // (op 2705) 
s_lshl_b32 s23, 1, s19 // (op 2705) 
s_add_i32 s24, s23, s22 // (op 2705) 


s_and_b32 s25, s0, s24 // (op 2705) 


s_add_i32 s0, s18, s25 // (op 2705) 


s_ashr_i32 s18, s0, s19 // (op 2705) 


s_lshl_b32 s0, s13, 24 // (op 2705) 
s_ashr_i32 s19, s0, 31 // (op 2705) 

s_xor_b32 s0, s18, s19 // (op 2705) 

s_sub_i32 s18, s0, s19 // (op 2705) 


s_mul_i32 s0, s1, s48 // (op 2705) 

s_sub_u32 s1, s16, s0 // (op 2705) 

s_mul_i32 s0, s18, s48 // (op 2705) 
s_add_u32 s0, s0, s1 // (op 2705) 


s_mul_hi_i32 s1, s16, s8 // (op 2705) 
s_add_i32 s18, s1, s16 // (op 2705) 

s_ashr_i32 s1, s18, 31 // (op 2705) 
s_cmp_eq_i32 s8, 0 // (op 2705) 
s_mov_b32 s19, scc // (op 2705) 
s_cmp_eq_i32 s19, 0 // (op 2705) 
s_cselect_b32 s22, 0, -1 // (op 2705) 

s_and_b32 s19, s10, 31 // (op 2705) 
s_lshl_b32 s23, 1, s19 // (op 2705) 
s_add_i32 s24, s23, s22 // (op 2705) 


s_and_b32 s25, s1, s24 // (op 2705) 


s_add_i32 s1, s18, s25 // (op 2705) 


s_ashr_i32 s18, s1, s19 // (op 2705) 


s_lshl_b32 s1, s10, 24 // (op 2705) 
s_ashr_i32 s19, s1, 31 // (op 2705) 

s_xor_b32 s1, s18, s19 // (op 2705) 

s_sub_i32 s18, s1, s19 // (op 2705) 


s_add_u32 s50, s6, 32 // (op 2705) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2705) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2705) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2705) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2705) 

s_mul_i32 s1, s48, s24 // (op 2705) 

s_mul_i32 s19, s14, s1 // (op 2705) 

s_mul_i32 s1, s18, s19 // (op 2705) 

s_sub_u32 s19, s16, s1 // (op 2705) 

s_mov_b32 s1, s19 // (op 2705) 
s_mul_hi_i32 s22, s1, s15 // (op 2705) 
s_add_i32 s23, s22, s1 // (op 2705) 


s_ashr_i32 s1, s23, 31 // (op 2705) 
s_cmp_eq_i32 s15, 0 // (op 2705) 
s_mov_b32 s22, scc // (op 2705) 
s_cmp_eq_i32 s22, 0 // (op 2705) 
s_cselect_b32 s24, 0, -1 // (op 2705) 

s_and_b32 s22, s9, 31 // (op 2705) 
s_lshl_b32 s25, 1, s22 // (op 2705) 
s_add_i32 s49, s25, s24 // (op 2705) 


s_and_b32 s24, s1, s49 // (op 2705) 


s_add_i32 s1, s23, s24 // (op 2705) 


s_ashr_i32 s23, s1, s22 // (op 2705) 


s_lshl_b32 s1, s9, 24 // (op 2705) 
s_ashr_i32 s22, s1, 31 // (op 2705) 

s_xor_b32 s1, s23, s22 // (op 2705) 

s_sub_i32 s23, s1, s22 // (op 2705) 


s_mul_i32 s1, s46, s48 // (op 2705) 
s_sub_i32 s22, s4, s1 // (op 2705) 
s_mul_i32 s24, s23, s22 // (op 2705) 


s_sub_u32 s25, s19, s24 // (op 2705) 


s_add_u32 s19, s25, s1 // (op 2705) 


s_cmp_eq_i32 s18, 0 // (op 2705) 
s_mov_b32 s1, scc // (op 2705) 

s_cmp_eq_i32 s1, 0 // (op 2705) 
s_cselect_b32 s18, s19, s0 // (op 2705) 



v_mov_b32 v12, 192 // (op 2705) 
v_mul_lo_u32 v15, s18, v12 // (op 2705) 

v_add_u32 v15, v15, v13 // (op 2705) 


v_mul_lo_u32 v13, v15, s27 // (op 2705) most significant: low of low * high
v_mul_hi_u32 v18, v15, s26 // (op 2705) most significant: high of low * low
v_mul_lo_u32 v12, v15, s26 // (op 2705) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 2705) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v16 // (op 2705) least significant half
v_addc_co_u32 v13, s[0:1], v13, v17, s[0:1] // (op 2705) most significant half



v_lshlrev_b64 v[16:17], 2, v[12:13] // (op 2705) 

v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 2705) 

v_mov_b32 v15, v12 // (op 2705) convert

s_mul_i32 s1, s26, s31 // (op 2707) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2707) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2707) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2707) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2707) most significant: sum
s_add_u32 s1, s1, s19 // (op 2707) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2707) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2707) 

s_mov_b32 s52, s28 // (op 2707) reg expression
s_mov_b32 s53, s29 // (op 2707) reg expression
s_mov_b32 s54, s0 // (op 2707) reg expression
s_mov_b32 s55, 131072 // (op 2707) call()

s_mul_hi_i32 s0, s16, s47 // (op 2750) 
v_lshrrev_b32 v12, 3, v1 // (op 2750) 
s_add_i32 s1, s0, s16 // (op 2750) 

v_and_b32 v13, -8, v1 // (op 2750) 
s_ashr_i32 s0, s1, 31 // (op 2750) 
v_sub_u32 v16, v1, v13 // (op 2750) 

s_cmp_eq_i32 s47, 0 // (op 2750) 
s_mov_b32 s18, scc // (op 2750) 
v_lshlrev_b32 v13, 4, v16 // (op 2750) 

s_cmp_eq_i32 s18, 0 // (op 2750) 
s_cselect_b32 s19, 0, -1 // (op 2750) 

v_mov_b32 v17, 0 // (op 2750) convert
v_mov_b32 v16, v13 // (op 2750) convert

s_and_b32 s18, s11, 31 // (op 2750) 
s_lshl_b32 s22, 1, s18 // (op 2750) 
s_add_i32 s23, s22, s19 // (op 2750) 


s_and_b32 s19, s0, s23 // (op 2750) 


s_add_i32 s0, s1, s19 // (op 2750) 


s_ashr_i32 s1, s0, s18 // (op 2750) 


s_lshl_b32 s0, s11, 24 // (op 2750) 
s_ashr_i32 s18, s0, 31 // (op 2750) 

s_xor_b32 s0, s1, s18 // (op 2750) 

s_sub_i32 s1, s0, s18 // (op 2750) 


s_mul_hi_i32 s0, s1, s12 // (op 2750) 
s_add_i32 s18, s0, s1 // (op 2750) 

s_ashr_i32 s0, s18, 31 // (op 2750) 
s_cmp_eq_i32 s12, 0 // (op 2750) 
s_mov_b32 s19, scc // (op 2750) 
s_cmp_eq_i32 s19, 0 // (op 2750) 
s_cselect_b32 s22, 0, -1 // (op 2750) 

s_and_b32 s19, s13, 31 // (op 2750) 
s_lshl_b32 s23, 1, s19 // (op 2750) 
s_add_i32 s24, s23, s22 // (op 2750) 


s_and_b32 s25, s0, s24 // (op 2750) 


s_add_i32 s0, s18, s25 // (op 2750) 


s_ashr_i32 s18, s0, s19 // (op 2750) 


s_lshl_b32 s0, s13, 24 // (op 2750) 
s_ashr_i32 s19, s0, 31 // (op 2750) 

s_xor_b32 s0, s18, s19 // (op 2750) 

s_sub_i32 s18, s0, s19 // (op 2750) 


s_add_u32 s50, s6, 32 // (op 2750) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2750) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2750) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2750) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2750) 

s_mul_i32 s0, s18, s24 // (op 2750) 

s_sub_i32 s18, s1, s0 // (op 2750) 


s_mul_hi_i32 s19, s16, s8 // (op 2750) 
s_add_i32 s0, s19, s16 // (op 2750) 

s_ashr_i32 s1, s0, 31 // (op 2750) 
s_cmp_eq_i32 s8, 0 // (op 2750) 
s_mov_b32 s19, scc // (op 2750) 
s_cmp_eq_i32 s19, 0 // (op 2750) 
s_cselect_b32 s22, 0, -1 // (op 2750) 

s_and_b32 s19, s10, 31 // (op 2750) 
s_lshl_b32 s23, 1, s19 // (op 2750) 
s_add_i32 s49, s23, s22 // (op 2750) 


s_and_b32 s22, s1, s49 // (op 2750) 


s_add_i32 s1, s0, s22 // (op 2750) 


s_ashr_i32 s0, s1, s19 // (op 2750) 


s_lshl_b32 s1, s10, 24 // (op 2750) 
s_ashr_i32 s19, s1, 31 // (op 2750) 

s_xor_b32 s1, s0, s19 // (op 2750) 

s_sub_i32 s0, s1, s19 // (op 2750) 


s_mul_i32 s1, s48, s24 // (op 2750) 

s_mul_i32 s19, s14, s1 // (op 2750) 

s_mul_i32 s1, s0, s19 // (op 2750) 

s_sub_u32 s19, s16, s1 // (op 2750) 

s_mul_hi_i32 s1, s19, s15 // (op 2750) 
s_add_i32 s22, s1, s19 // (op 2750) 


s_ashr_i32 s1, s22, 31 // (op 2750) 
s_cmp_eq_i32 s15, 0 // (op 2750) 
s_mov_b32 s19, scc // (op 2750) 
s_cmp_eq_i32 s19, 0 // (op 2750) 
s_cselect_b32 s23, 0, -1 // (op 2750) 

s_and_b32 s19, s9, 31 // (op 2750) 
s_lshl_b32 s24, 1, s19 // (op 2750) 
s_add_i32 s25, s24, s23 // (op 2750) 


s_and_b32 s23, s1, s25 // (op 2750) 


s_add_i32 s1, s22, s23 // (op 2750) 


s_ashr_i32 s22, s1, s19 // (op 2750) 


s_lshl_b32 s1, s9, 24 // (op 2750) 
s_ashr_i32 s19, s1, 31 // (op 2750) 

s_xor_b32 s1, s22, s19 // (op 2750) 

s_sub_i32 s22, s1, s19 // (op 2750) 


s_cmp_eq_i32 s0, 0 // (op 2750) 
s_mov_b32 s1, scc // (op 2750) 

s_cmp_eq_i32 s1, 0 // (op 2750) 
s_cselect_b32 s0, s22, s18 // (op 2750) 



v_lshl_add_u32 v13, s0, 5, v12 // (op 2750) 


v_mul_lo_u32 v19, v13, s33 // (op 2750) most significant: low of low * high
v_mul_hi_u32 v12, v13, s32 // (op 2750) most significant: high of low * low
v_mul_lo_u32 v18, v13, s32 // (op 2750) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 2750) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 2750) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 2750) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 2750) 


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 2750) 

v_mov_b32 v12, v16 // (op 2750) convert

s_mul_i32 s1, s32, s7 // (op 2752) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2752) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2752) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2752) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2752) most significant: sum
s_add_u32 s1, s1, s19 // (op 2752) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2752) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2752) 

s_mov_b32 s64, s34 // (op 2752) reg expression
s_mov_b32 s65, s35 // (op 2752) reg expression
s_mov_b32 s66, s0 // (op 2752) reg expression
s_mov_b32 s67, 131072 // (op 2752) call()

s_mul_hi_i32 s0, s16, s47 // (op 2769) 
v_lshrrev_b32 v13, 3, v1 // (op 2769) 
s_add_i32 s1, s0, s16 // (op 2769) 

v_and_b32 v16, -8, v1 // (op 2769) 
s_ashr_i32 s0, s1, 31 // (op 2769) 
v_sub_u32 v17, v1, v16 // (op 2769) 

s_cmp_eq_i32 s47, 0 // (op 2769) 
s_mov_b32 s18, scc // (op 2769) 
v_lshlrev_b32 v16, 4, v17 // (op 2769) 

s_cmp_eq_i32 s18, 0 // (op 2769) 
s_cselect_b32 s19, 0, -1 // (op 2769) 

v_mov_b32 v19, 0 // (op 2769) convert
v_mov_b32 v18, v16 // (op 2769) convert

s_and_b32 s18, s11, 31 // (op 2769) 
s_lshl_b32 s22, 1, s18 // (op 2769) 
s_add_i32 s23, s22, s19 // (op 2769) 


s_and_b32 s19, s0, s23 // (op 2769) 


s_add_i32 s0, s1, s19 // (op 2769) 


s_ashr_i32 s1, s0, s18 // (op 2769) 


s_lshl_b32 s0, s11, 24 // (op 2769) 
s_ashr_i32 s18, s0, 31 // (op 2769) 

s_xor_b32 s0, s1, s18 // (op 2769) 

s_sub_i32 s1, s0, s18 // (op 2769) 


s_mul_hi_i32 s0, s1, s12 // (op 2769) 
s_add_i32 s18, s0, s1 // (op 2769) 

s_ashr_i32 s0, s18, 31 // (op 2769) 
s_cmp_eq_i32 s12, 0 // (op 2769) 
s_mov_b32 s19, scc // (op 2769) 
s_cmp_eq_i32 s19, 0 // (op 2769) 
s_cselect_b32 s22, 0, -1 // (op 2769) 

s_and_b32 s19, s13, 31 // (op 2769) 
s_lshl_b32 s23, 1, s19 // (op 2769) 
s_add_i32 s24, s23, s22 // (op 2769) 


s_and_b32 s25, s0, s24 // (op 2769) 


s_add_i32 s0, s18, s25 // (op 2769) 


s_ashr_i32 s18, s0, s19 // (op 2769) 


s_lshl_b32 s0, s13, 24 // (op 2769) 
s_ashr_i32 s19, s0, 31 // (op 2769) 

s_xor_b32 s0, s18, s19 // (op 2769) 

s_sub_i32 s18, s0, s19 // (op 2769) 


s_add_u32 s50, s6, 32 // (op 2769) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2769) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2769) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2769) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2769) 

s_mul_i32 s0, s18, s24 // (op 2769) 

s_sub_i32 s18, s1, s0 // (op 2769) 


s_mul_hi_i32 s19, s16, s8 // (op 2769) 
s_add_i32 s0, s19, s16 // (op 2769) 

s_ashr_i32 s1, s0, 31 // (op 2769) 
s_cmp_eq_i32 s8, 0 // (op 2769) 
s_mov_b32 s19, scc // (op 2769) 
s_cmp_eq_i32 s19, 0 // (op 2769) 
s_cselect_b32 s22, 0, -1 // (op 2769) 

s_and_b32 s19, s10, 31 // (op 2769) 
s_lshl_b32 s23, 1, s19 // (op 2769) 
s_add_i32 s49, s23, s22 // (op 2769) 


s_and_b32 s22, s1, s49 // (op 2769) 


s_add_i32 s1, s0, s22 // (op 2769) 


s_ashr_i32 s0, s1, s19 // (op 2769) 


s_lshl_b32 s1, s10, 24 // (op 2769) 
s_ashr_i32 s19, s1, 31 // (op 2769) 

s_xor_b32 s1, s0, s19 // (op 2769) 

s_sub_i32 s0, s1, s19 // (op 2769) 


s_mul_i32 s1, s48, s24 // (op 2769) 

s_mul_i32 s19, s14, s1 // (op 2769) 

s_mul_i32 s1, s0, s19 // (op 2769) 

s_sub_u32 s19, s16, s1 // (op 2769) 

s_mul_hi_i32 s1, s19, s15 // (op 2769) 
s_add_i32 s22, s1, s19 // (op 2769) 


s_ashr_i32 s1, s22, 31 // (op 2769) 
s_cmp_eq_i32 s15, 0 // (op 2769) 
s_mov_b32 s19, scc // (op 2769) 
s_cmp_eq_i32 s19, 0 // (op 2769) 
s_cselect_b32 s23, 0, -1 // (op 2769) 

s_and_b32 s19, s9, 31 // (op 2769) 
s_lshl_b32 s24, 1, s19 // (op 2769) 
s_add_i32 s25, s24, s23 // (op 2769) 


s_and_b32 s23, s1, s25 // (op 2769) 


s_add_i32 s1, s22, s23 // (op 2769) 


s_ashr_i32 s22, s1, s19 // (op 2769) 


s_lshl_b32 s1, s9, 24 // (op 2769) 
s_ashr_i32 s19, s1, 31 // (op 2769) 

s_xor_b32 s1, s22, s19 // (op 2769) 

s_sub_i32 s22, s1, s19 // (op 2769) 


s_cmp_eq_i32 s0, 0 // (op 2769) 
s_mov_b32 s1, scc // (op 2769) 

s_cmp_eq_i32 s1, 0 // (op 2769) 
s_cselect_b32 s0, s22, s18 // (op 2769) 



v_lshl_add_u32 v16, s0, 5, v13 // (op 2769) 


v_mul_lo_u32 v21, v16, s33 // (op 2769) most significant: low of low * high
v_mul_hi_u32 v13, v16, s32 // (op 2769) most significant: high of low * low
v_mul_lo_u32 v20, v16, s32 // (op 2769) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 2769) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 2769) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 2769) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 2769) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 2769) 

v_mov_b32 v13, v18 // (op 2769) convert

s_mul_i32 s1, s32, s7 // (op 2771) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2771) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2771) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2771) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2771) most significant: sum
s_add_u32 s1, s1, s19 // (op 2771) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2771) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2771) 

s_mov_b32 s64, s34 // (op 2771) reg expression
s_mov_b32 s65, s35 // (op 2771) reg expression
s_mov_b32 s66, s0 // (op 2771) reg expression
s_mov_b32 s67, 131072 // (op 2771) call()

s_mul_hi_i32 s0, s16, s47 // (op 2814) 
v_lshrrev_b32 v16, 2, v1 // (op 2814) 
s_add_i32 s1, s0, s16 // (op 2814) 

v_mul_lo_u32 v17, v16, 3 // (op 2814) 

s_ashr_i32 s0, s1, 31 // (op 2814) 
v_and_b32 v16, -4, v1 // (op 2814) 
s_cmp_eq_i32 s47, 0 // (op 2814) 
s_mov_b32 s18, scc // (op 2814) 
v_sub_u32 v18, v1, v16 // (op 2814) 

s_cmp_eq_i32 s18, 0 // (op 2814) 
s_cselect_b32 s19, 0, -1 // (op 2814) 

s_and_b32 s18, s11, 31 // (op 2814) 
s_lshl_b32 s22, 1, s18 // (op 2814) 
s_add_i32 s23, s22, s19 // (op 2814) 


s_and_b32 s19, s0, s23 // (op 2814) 


s_add_i32 s0, s1, s19 // (op 2814) 


s_ashr_i32 s1, s0, s18 // (op 2814) 


s_lshl_b32 s0, s11, 24 // (op 2814) 
s_ashr_i32 s18, s0, 31 // (op 2814) 

s_xor_b32 s0, s1, s18 // (op 2814) 

s_sub_i32 s1, s0, s18 // (op 2814) 


s_mul_hi_i32 s0, s1, s12 // (op 2814) 
s_add_i32 s18, s0, s1 // (op 2814) 

s_ashr_i32 s0, s18, 31 // (op 2814) 
s_cmp_eq_i32 s12, 0 // (op 2814) 
s_mov_b32 s19, scc // (op 2814) 
s_cmp_eq_i32 s19, 0 // (op 2814) 
s_cselect_b32 s22, 0, -1 // (op 2814) 

s_and_b32 s19, s13, 31 // (op 2814) 
s_lshl_b32 s23, 1, s19 // (op 2814) 
s_add_i32 s24, s23, s22 // (op 2814) 


s_and_b32 s25, s0, s24 // (op 2814) 


s_add_i32 s0, s18, s25 // (op 2814) 


s_ashr_i32 s18, s0, s19 // (op 2814) 


s_lshl_b32 s0, s13, 24 // (op 2814) 
s_ashr_i32 s19, s0, 31 // (op 2814) 

s_xor_b32 s0, s18, s19 // (op 2814) 

s_sub_i32 s18, s0, s19 // (op 2814) 


s_mul_i32 s0, s1, s48 // (op 2814) 

s_sub_u32 s1, s16, s0 // (op 2814) 

s_mul_i32 s0, s18, s48 // (op 2814) 
s_add_u32 s0, s0, s1 // (op 2814) 


s_mul_hi_i32 s1, s16, s8 // (op 2814) 
s_add_i32 s18, s1, s16 // (op 2814) 

s_ashr_i32 s1, s18, 31 // (op 2814) 
s_cmp_eq_i32 s8, 0 // (op 2814) 
s_mov_b32 s19, scc // (op 2814) 
s_cmp_eq_i32 s19, 0 // (op 2814) 
s_cselect_b32 s22, 0, -1 // (op 2814) 

s_and_b32 s19, s10, 31 // (op 2814) 
s_lshl_b32 s23, 1, s19 // (op 2814) 
s_add_i32 s24, s23, s22 // (op 2814) 


s_and_b32 s25, s1, s24 // (op 2814) 


s_add_i32 s1, s18, s25 // (op 2814) 


s_ashr_i32 s18, s1, s19 // (op 2814) 


s_lshl_b32 s1, s10, 24 // (op 2814) 
s_ashr_i32 s19, s1, 31 // (op 2814) 

s_xor_b32 s1, s18, s19 // (op 2814) 

s_sub_i32 s18, s1, s19 // (op 2814) 


s_add_u32 s50, s6, 32 // (op 2814) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2814) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2814) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2814) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2814) 

s_mul_i32 s1, s48, s24 // (op 2814) 

s_mul_i32 s19, s14, s1 // (op 2814) 

s_mul_i32 s1, s18, s19 // (op 2814) 

s_sub_u32 s19, s16, s1 // (op 2814) 

s_mov_b32 s1, s19 // (op 2814) 
s_mul_hi_i32 s22, s1, s15 // (op 2814) 
s_add_i32 s23, s22, s1 // (op 2814) 


s_ashr_i32 s1, s23, 31 // (op 2814) 
s_cmp_eq_i32 s15, 0 // (op 2814) 
s_mov_b32 s22, scc // (op 2814) 
s_cmp_eq_i32 s22, 0 // (op 2814) 
s_cselect_b32 s24, 0, -1 // (op 2814) 

s_and_b32 s22, s9, 31 // (op 2814) 
s_lshl_b32 s25, 1, s22 // (op 2814) 
s_add_i32 s49, s25, s24 // (op 2814) 


s_and_b32 s24, s1, s49 // (op 2814) 


s_add_i32 s1, s23, s24 // (op 2814) 


s_ashr_i32 s23, s1, s22 // (op 2814) 


s_lshl_b32 s1, s9, 24 // (op 2814) 
s_ashr_i32 s22, s1, 31 // (op 2814) 

s_xor_b32 s1, s23, s22 // (op 2814) 

s_sub_i32 s23, s1, s22 // (op 2814) 


s_mul_i32 s1, s46, s48 // (op 2814) 
s_sub_i32 s22, s4, s1 // (op 2814) 
s_mul_i32 s24, s23, s22 // (op 2814) 


s_sub_u32 s25, s19, s24 // (op 2814) 


s_add_u32 s19, s25, s1 // (op 2814) 


s_cmp_eq_i32 s18, 0 // (op 2814) 
s_mov_b32 s1, scc // (op 2814) 

s_cmp_eq_i32 s1, 0 // (op 2814) 
s_cselect_b32 s18, s19, s0 // (op 2814) 



v_mov_b32 v16, 192 // (op 2814) 
v_mul_lo_u32 v19, s18, v16 // (op 2814) 

v_add_u32 v19, v19, v17 // (op 2814) 


v_mul_lo_u32 v16, v19, s36 // (op 2814) 
v_add_u32 v16, v16, v18 // (op 2814) 


s_mul_i32 s1, s36, s31 // (op 2816) most significant: low of low * high
s_mul_i32 s18, s37, s30 // (op 2816) most significant: low of high * low
s_mul_i32 s0, s36, s30 // (op 2816) least significant
s_mul_hi_u32 s19, s36, s30 // (op 2816) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2816) most significant: sum
s_add_u32 s1, s1, s19 // (op 2816) most significant: sum


s_mov_b32 s68, s38 // (op 2816) reg expression
s_mov_b32 s69, s39 // (op 2816) reg expression
s_mov_b32 s70, s0 // (op 2816) reg expression
s_mov_b32 s71, 131072 // (op 2816) call()

s_mul_hi_i32 s0, s16, s47 // (op 2833) 
v_lshrrev_b32 v17, 2, v1 // (op 2833) 
s_add_i32 s1, s0, s16 // (op 2833) 

v_mul_lo_u32 v18, v17, 3 // (op 2833) 

s_ashr_i32 s0, s1, 31 // (op 2833) 
v_and_b32 v17, -4, v1 // (op 2833) 
s_cmp_eq_i32 s47, 0 // (op 2833) 
s_mov_b32 s18, scc // (op 2833) 
v_sub_u32 v19, v1, v17 // (op 2833) 

s_cmp_eq_i32 s18, 0 // (op 2833) 
s_cselect_b32 s19, 0, -1 // (op 2833) 

s_and_b32 s18, s11, 31 // (op 2833) 
s_lshl_b32 s22, 1, s18 // (op 2833) 
s_add_i32 s23, s22, s19 // (op 2833) 


s_and_b32 s19, s0, s23 // (op 2833) 


s_add_i32 s0, s1, s19 // (op 2833) 


s_ashr_i32 s1, s0, s18 // (op 2833) 


s_lshl_b32 s0, s11, 24 // (op 2833) 
s_ashr_i32 s18, s0, 31 // (op 2833) 

s_xor_b32 s0, s1, s18 // (op 2833) 

s_sub_i32 s1, s0, s18 // (op 2833) 


s_mul_hi_i32 s0, s1, s12 // (op 2833) 
s_add_i32 s18, s0, s1 // (op 2833) 

s_ashr_i32 s0, s18, 31 // (op 2833) 
s_cmp_eq_i32 s12, 0 // (op 2833) 
s_mov_b32 s19, scc // (op 2833) 
s_cmp_eq_i32 s19, 0 // (op 2833) 
s_cselect_b32 s22, 0, -1 // (op 2833) 

s_and_b32 s19, s13, 31 // (op 2833) 
s_lshl_b32 s23, 1, s19 // (op 2833) 
s_add_i32 s24, s23, s22 // (op 2833) 


s_and_b32 s25, s0, s24 // (op 2833) 


s_add_i32 s0, s18, s25 // (op 2833) 


s_ashr_i32 s18, s0, s19 // (op 2833) 


s_lshl_b32 s0, s13, 24 // (op 2833) 
s_ashr_i32 s19, s0, 31 // (op 2833) 

s_xor_b32 s0, s18, s19 // (op 2833) 

s_sub_i32 s18, s0, s19 // (op 2833) 


s_mul_i32 s0, s1, s48 // (op 2833) 

s_sub_u32 s1, s16, s0 // (op 2833) 

s_mul_i32 s0, s18, s48 // (op 2833) 
s_add_u32 s0, s0, s1 // (op 2833) 


s_mul_hi_i32 s1, s16, s8 // (op 2833) 
s_add_i32 s18, s1, s16 // (op 2833) 

s_ashr_i32 s1, s18, 31 // (op 2833) 
s_cmp_eq_i32 s8, 0 // (op 2833) 
s_mov_b32 s19, scc // (op 2833) 
s_cmp_eq_i32 s19, 0 // (op 2833) 
s_cselect_b32 s22, 0, -1 // (op 2833) 

s_and_b32 s19, s10, 31 // (op 2833) 
s_lshl_b32 s23, 1, s19 // (op 2833) 
s_add_i32 s24, s23, s22 // (op 2833) 


s_and_b32 s25, s1, s24 // (op 2833) 


s_add_i32 s1, s18, s25 // (op 2833) 


s_ashr_i32 s18, s1, s19 // (op 2833) 


s_lshl_b32 s1, s10, 24 // (op 2833) 
s_ashr_i32 s19, s1, 31 // (op 2833) 

s_xor_b32 s1, s18, s19 // (op 2833) 

s_sub_i32 s18, s1, s19 // (op 2833) 


s_add_u32 s50, s6, 32 // (op 2833) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2833) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2833) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2833) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2833) 

s_mul_i32 s1, s48, s24 // (op 2833) 

s_mul_i32 s19, s14, s1 // (op 2833) 

s_mul_i32 s1, s18, s19 // (op 2833) 

s_sub_u32 s19, s16, s1 // (op 2833) 

s_mov_b32 s1, s19 // (op 2833) 
s_mul_hi_i32 s22, s1, s15 // (op 2833) 
s_add_i32 s23, s22, s1 // (op 2833) 


s_ashr_i32 s1, s23, 31 // (op 2833) 
s_cmp_eq_i32 s15, 0 // (op 2833) 
s_mov_b32 s22, scc // (op 2833) 
s_cmp_eq_i32 s22, 0 // (op 2833) 
s_cselect_b32 s24, 0, -1 // (op 2833) 

s_and_b32 s22, s9, 31 // (op 2833) 
s_lshl_b32 s25, 1, s22 // (op 2833) 
s_add_i32 s49, s25, s24 // (op 2833) 


s_and_b32 s24, s1, s49 // (op 2833) 


s_add_i32 s1, s23, s24 // (op 2833) 


s_ashr_i32 s23, s1, s22 // (op 2833) 


s_lshl_b32 s1, s9, 24 // (op 2833) 
s_ashr_i32 s22, s1, 31 // (op 2833) 

s_xor_b32 s1, s23, s22 // (op 2833) 

s_sub_i32 s23, s1, s22 // (op 2833) 


s_mul_i32 s1, s46, s48 // (op 2833) 
s_sub_i32 s22, s4, s1 // (op 2833) 
s_mul_i32 s24, s23, s22 // (op 2833) 


s_sub_u32 s25, s19, s24 // (op 2833) 


s_add_u32 s19, s25, s1 // (op 2833) 


s_cmp_eq_i32 s18, 0 // (op 2833) 
s_mov_b32 s1, scc // (op 2833) 

s_cmp_eq_i32 s1, 0 // (op 2833) 
s_cselect_b32 s18, s19, s0 // (op 2833) 



v_mov_b32 v17, 192 // (op 2833) 
v_mul_lo_u32 v20, s18, v17 // (op 2833) 

v_add_u32 v20, v20, v18 // (op 2833) 


v_mul_lo_u32 v17, v20, s36 // (op 2833) 
v_add_u32 v17, v17, v19 // (op 2833) 


s_mul_i32 s1, s36, s31 // (op 2835) most significant: low of low * high
s_mul_i32 s18, s37, s30 // (op 2835) most significant: low of high * low
s_mul_i32 s0, s36, s30 // (op 2835) least significant
s_mul_hi_u32 s19, s36, s30 // (op 2835) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2835) most significant: sum
s_add_u32 s1, s1, s19 // (op 2835) most significant: sum


s_mov_b32 s68, s38 // (op 2835) reg expression
s_mov_b32 s69, s39 // (op 2835) reg expression
s_mov_b32 s70, s0 // (op 2835) reg expression
s_mov_b32 s71, 131072 // (op 2835) call()

s_mul_hi_i32 s0, s16, s47 // (op 2878) 
v_lshrrev_b32 v18, 6, v1 // (op 2878) 
s_add_i32 s1, s0, s16 // (op 2878) 

v_and_b32 v19, 1, v18 // (op 2878) 

s_ashr_i32 s0, s1, 31 // (op 2878) 
v_and_b32 v18, 15, v1 // (op 2878) 
s_cmp_eq_i32 s47, 0 // (op 2878) 
s_mov_b32 s18, scc // (op 2878) 
v_lshl_add_u32 v20, v19, 4, v18 // (op 2878) 


s_cmp_eq_i32 s18, 0 // (op 2878) 
s_cselect_b32 s19, 0, -1 // (op 2878) 

v_and_b32 v18, 63, v1 // (op 2878) 
s_and_b32 s18, s11, 31 // (op 2878) 
v_lshrrev_b32 v19, 4, v18 // (op 2878) 

s_lshl_b32 s22, 1, s18 // (op 2878) 
s_add_i32 s23, s22, s19 // (op 2878) 


s_and_b32 s19, s0, s23 // (op 2878) 


s_add_i32 s0, s1, s19 // (op 2878) 


s_ashr_i32 s1, s0, s18 // (op 2878) 


s_lshl_b32 s0, s11, 24 // (op 2878) 
s_ashr_i32 s18, s0, 31 // (op 2878) 

s_xor_b32 s0, s1, s18 // (op 2878) 

s_sub_i32 s1, s0, s18 // (op 2878) 


s_mul_hi_i32 s0, s1, s12 // (op 2878) 
s_add_i32 s18, s0, s1 // (op 2878) 

s_ashr_i32 s0, s18, 31 // (op 2878) 
s_cmp_eq_i32 s12, 0 // (op 2878) 
s_mov_b32 s19, scc // (op 2878) 
s_cmp_eq_i32 s19, 0 // (op 2878) 
s_cselect_b32 s22, 0, -1 // (op 2878) 

s_and_b32 s19, s13, 31 // (op 2878) 
s_lshl_b32 s23, 1, s19 // (op 2878) 
s_add_i32 s24, s23, s22 // (op 2878) 


s_and_b32 s25, s0, s24 // (op 2878) 


s_add_i32 s0, s18, s25 // (op 2878) 


s_ashr_i32 s18, s0, s19 // (op 2878) 


s_lshl_b32 s0, s13, 24 // (op 2878) 
s_ashr_i32 s19, s0, 31 // (op 2878) 

s_xor_b32 s0, s18, s19 // (op 2878) 

s_sub_i32 s18, s0, s19 // (op 2878) 


s_add_u32 s50, s6, 32 // (op 2878) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2878) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2878) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2878) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2878) 

s_mul_i32 s0, s18, s24 // (op 2878) 

s_sub_i32 s18, s1, s0 // (op 2878) 


s_mul_hi_i32 s19, s16, s8 // (op 2878) 
s_add_i32 s0, s19, s16 // (op 2878) 

s_ashr_i32 s1, s0, 31 // (op 2878) 
s_cmp_eq_i32 s8, 0 // (op 2878) 
s_mov_b32 s19, scc // (op 2878) 
s_cmp_eq_i32 s19, 0 // (op 2878) 
s_cselect_b32 s22, 0, -1 // (op 2878) 

s_and_b32 s19, s10, 31 // (op 2878) 
s_lshl_b32 s23, 1, s19 // (op 2878) 
s_add_i32 s49, s23, s22 // (op 2878) 


s_and_b32 s22, s1, s49 // (op 2878) 


s_add_i32 s1, s0, s22 // (op 2878) 


s_ashr_i32 s0, s1, s19 // (op 2878) 


s_lshl_b32 s1, s10, 24 // (op 2878) 
s_ashr_i32 s19, s1, 31 // (op 2878) 

s_xor_b32 s1, s0, s19 // (op 2878) 

s_sub_i32 s0, s1, s19 // (op 2878) 


s_mul_i32 s1, s48, s24 // (op 2878) 

s_mul_i32 s19, s14, s1 // (op 2878) 

s_mul_i32 s1, s0, s19 // (op 2878) 

s_sub_u32 s19, s16, s1 // (op 2878) 

s_mul_hi_i32 s1, s19, s15 // (op 2878) 
s_add_i32 s22, s1, s19 // (op 2878) 


s_ashr_i32 s1, s22, 31 // (op 2878) 
s_cmp_eq_i32 s15, 0 // (op 2878) 
s_mov_b32 s19, scc // (op 2878) 
s_cmp_eq_i32 s19, 0 // (op 2878) 
s_cselect_b32 s23, 0, -1 // (op 2878) 

s_and_b32 s19, s9, 31 // (op 2878) 
s_lshl_b32 s24, 1, s19 // (op 2878) 
s_add_i32 s25, s24, s23 // (op 2878) 


s_and_b32 s23, s1, s25 // (op 2878) 


s_add_i32 s1, s22, s23 // (op 2878) 


s_ashr_i32 s22, s1, s19 // (op 2878) 


s_lshl_b32 s1, s9, 24 // (op 2878) 
s_ashr_i32 s19, s1, 31 // (op 2878) 

s_xor_b32 s1, s22, s19 // (op 2878) 

s_sub_i32 s22, s1, s19 // (op 2878) 


s_cmp_eq_i32 s0, 0 // (op 2878) 
s_mov_b32 s1, scc // (op 2878) 

s_cmp_eq_i32 s1, 0 // (op 2878) 
s_cselect_b32 s0, s22, s18 // (op 2878) 



v_lshl_add_u32 v18, s0, 5, v20 // (op 2878) 


v_mul_lo_u32 v20, v18, s40 // (op 2878) 

v_add_u32 v18, v19, v20 // (op 2878) 


s_mul_i32 s1, s40, s7 // (op 2880) most significant: low of low * high
s_mul_i32 s18, s41, s6 // (op 2880) most significant: low of high * low
s_mul_i32 s0, s40, s6 // (op 2880) least significant
s_mul_hi_u32 s19, s40, s6 // (op 2880) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2880) most significant: sum
s_add_u32 s1, s1, s19 // (op 2880) most significant: sum


s_mov_b32 s72, s42 // (op 2880) reg expression
s_mov_b32 s73, s43 // (op 2880) reg expression
s_mov_b32 s74, s0 // (op 2880) reg expression
s_mov_b32 s75, 131072 // (op 2880) call()

s_mul_hi_i32 s0, s16, s47 // (op 2897) 
v_lshrrev_b32 v19, 6, v1 // (op 2897) 
s_add_i32 s1, s0, s16 // (op 2897) 

v_and_b32 v20, 1, v19 // (op 2897) 

s_ashr_i32 s0, s1, 31 // (op 2897) 
v_and_b32 v19, 15, v1 // (op 2897) 
s_cmp_eq_i32 s47, 0 // (op 2897) 
s_mov_b32 s18, scc // (op 2897) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 2897) 


s_cmp_eq_i32 s18, 0 // (op 2897) 
s_cselect_b32 s19, 0, -1 // (op 2897) 

v_and_b32 v19, 63, v1 // (op 2897) 
s_and_b32 s18, s11, 31 // (op 2897) 
v_lshrrev_b32 v20, 4, v19 // (op 2897) 

s_lshl_b32 s22, 1, s18 // (op 2897) 
s_add_i32 s23, s22, s19 // (op 2897) 


s_and_b32 s19, s0, s23 // (op 2897) 


s_add_i32 s0, s1, s19 // (op 2897) 


s_ashr_i32 s1, s0, s18 // (op 2897) 


s_lshl_b32 s0, s11, 24 // (op 2897) 
s_ashr_i32 s18, s0, 31 // (op 2897) 

s_xor_b32 s0, s1, s18 // (op 2897) 

s_sub_i32 s1, s0, s18 // (op 2897) 


s_mul_hi_i32 s0, s1, s12 // (op 2897) 
s_add_i32 s18, s0, s1 // (op 2897) 

s_ashr_i32 s0, s18, 31 // (op 2897) 
s_cmp_eq_i32 s12, 0 // (op 2897) 
s_mov_b32 s19, scc // (op 2897) 
s_cmp_eq_i32 s19, 0 // (op 2897) 
s_cselect_b32 s22, 0, -1 // (op 2897) 

s_and_b32 s19, s13, 31 // (op 2897) 
s_lshl_b32 s23, 1, s19 // (op 2897) 
s_add_i32 s24, s23, s22 // (op 2897) 


s_and_b32 s25, s0, s24 // (op 2897) 


s_add_i32 s0, s18, s25 // (op 2897) 


s_ashr_i32 s18, s0, s19 // (op 2897) 


s_lshl_b32 s0, s13, 24 // (op 2897) 
s_ashr_i32 s19, s0, 31 // (op 2897) 

s_xor_b32 s0, s18, s19 // (op 2897) 

s_sub_i32 s18, s0, s19 // (op 2897) 


s_add_u32 s50, s6, 32 // (op 2897) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2897) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2897) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2897) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2897) 

s_mul_i32 s0, s18, s24 // (op 2897) 

s_sub_i32 s18, s1, s0 // (op 2897) 


s_mul_hi_i32 s19, s16, s8 // (op 2897) 
s_add_i32 s0, s19, s16 // (op 2897) 

s_ashr_i32 s1, s0, 31 // (op 2897) 
s_cmp_eq_i32 s8, 0 // (op 2897) 
s_mov_b32 s19, scc // (op 2897) 
s_cmp_eq_i32 s19, 0 // (op 2897) 
s_cselect_b32 s22, 0, -1 // (op 2897) 

s_and_b32 s19, s10, 31 // (op 2897) 
s_lshl_b32 s23, 1, s19 // (op 2897) 
s_add_i32 s49, s23, s22 // (op 2897) 


s_and_b32 s22, s1, s49 // (op 2897) 


s_add_i32 s1, s0, s22 // (op 2897) 


s_ashr_i32 s0, s1, s19 // (op 2897) 


s_lshl_b32 s1, s10, 24 // (op 2897) 
s_ashr_i32 s19, s1, 31 // (op 2897) 

s_xor_b32 s1, s0, s19 // (op 2897) 

s_sub_i32 s0, s1, s19 // (op 2897) 


s_mul_i32 s1, s48, s24 // (op 2897) 

s_mul_i32 s19, s14, s1 // (op 2897) 

s_mul_i32 s1, s0, s19 // (op 2897) 

s_sub_u32 s19, s16, s1 // (op 2897) 

s_mul_hi_i32 s1, s19, s15 // (op 2897) 
s_add_i32 s22, s1, s19 // (op 2897) 


s_ashr_i32 s1, s22, 31 // (op 2897) 
s_cmp_eq_i32 s15, 0 // (op 2897) 
s_mov_b32 s19, scc // (op 2897) 
s_cmp_eq_i32 s19, 0 // (op 2897) 
s_cselect_b32 s23, 0, -1 // (op 2897) 

s_and_b32 s19, s9, 31 // (op 2897) 
s_lshl_b32 s24, 1, s19 // (op 2897) 
s_add_i32 s25, s24, s23 // (op 2897) 


s_and_b32 s23, s1, s25 // (op 2897) 


s_add_i32 s1, s22, s23 // (op 2897) 


s_ashr_i32 s22, s1, s19 // (op 2897) 


s_lshl_b32 s1, s9, 24 // (op 2897) 
s_ashr_i32 s19, s1, 31 // (op 2897) 

s_xor_b32 s1, s22, s19 // (op 2897) 

s_sub_i32 s22, s1, s19 // (op 2897) 


s_cmp_eq_i32 s0, 0 // (op 2897) 
s_mov_b32 s1, scc // (op 2897) 

s_cmp_eq_i32 s1, 0 // (op 2897) 
s_cselect_b32 s0, s22, s18 // (op 2897) 



v_lshl_add_u32 v19, s0, 5, v21 // (op 2897) 


v_mul_lo_u32 v21, v19, s40 // (op 2897) 

v_add_u32 v19, v20, v21 // (op 2897) 


s_mul_i32 s1, s40, s7 // (op 2899) most significant: low of low * high
s_mul_i32 s18, s41, s6 // (op 2899) most significant: low of high * low
s_mul_i32 s0, s40, s6 // (op 2899) least significant
s_mul_hi_u32 s19, s40, s6 // (op 2899) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2899) most significant: sum
s_add_u32 s1, s1, s19 // (op 2899) most significant: sum


s_mov_b32 s72, s42 // (op 2899) reg expression
s_mov_b32 s73, s43 // (op 2899) reg expression
s_mov_b32 s74, s0 // (op 2899) reg expression
s_mov_b32 s75, 131072 // (op 2899) call()




buffer_load_ubyte v20, v18, s[72:75], 0 offen offset: 0 // (op 2449) Load value



buffer_load_dwordx2 v[22:23], v12, s[64:67], 0 offen // (op 2453) Load value

s_lshl_b64 s[0:1], s[26:27], 2 // (op 2456) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2456) 


v_mov_b32 v21, v14 // (op 2456) 
buffer_load_dwordx4 v[26:29], v21, s[52:55], 0 offen // (op 2456) Load value
v_add_u32 v21, v21, v24 // (op 2456) 
buffer_load_dwordx4 v[30:33], v21, s[52:55], 0 offen // (op 2456) Load value
v_add_u32 v21, v21, v24 // (op 2456) 
buffer_load_dwordx4 v[34:37], v21, s[52:55], 0 offen // (op 2456) Load value



v_mov_b64 v[24:25], s[36:37] // (op 2459) convert

v_mov_b32 v21, v16 // (op 2459) 
buffer_load_ubyte v38, v21, s[68:71], 0 offen offset: 0 // (op 2459) Load value
v_add_u32 v21, v21, v24 // (op 2459) 
buffer_load_ubyte v39, v21, s[68:71], 0 offen offset: 0 // (op 2459) Load value
v_add_u32 v21, v21, v24 // (op 2459) 
buffer_load_ubyte v40, v21, s[68:71], 0 offen offset: 0 // (op 2459) Load value





buffer_load_ubyte v21, v18, s[72:75], 0 offen offset:4 // (op 2463) Load value




buffer_load_dwordx2 v[24:25], v12, s[64:67], 0 offen offset:64 // (op 2467) Load value


s_lshl_b64 s[0:1], s[26:27], 2 // (op 2470) 
v_lshrrev_b64 v[42:43], 3, s[0:1] // (op 2470) 


v_mov_b32 v12, v14 // (op 2470) 
buffer_load_dwordx4 v[44:47], v12, s[52:55], 0 offen offset:64 // (op 2470) Load value
v_add_u32 v12, v12, v42 // (op 2470) 
buffer_load_dwordx4 v[48:51], v12, s[52:55], 0 offen offset:64 // (op 2470) Load value
v_add_u32 v12, v12, v42 // (op 2470) 
buffer_load_dwordx4 v[52:55], v12, s[52:55], 0 offen offset:64 // (op 2470) Load value




v_mov_b64 v[42:43], s[36:37] // (op 2473) convert

v_mov_b32 v12, v16 // (op 2473) 
buffer_load_ubyte v56, v12, s[68:71], 0 offen offset:4 // (op 2473) Load value
v_add_u32 v12, v12, v42 // (op 2473) 
buffer_load_ubyte v57, v12, s[68:71], 0 offen offset:4 // (op 2473) Load value
v_add_u32 v12, v12, v42 // (op 2473) 
buffer_load_ubyte v58, v12, s[68:71], 0 offen offset:4 // (op 2473) Load value



v_and_b32 v12, -8, v1 // (op 3022) 
v_sub_u32 v14, v1, v12 // (op 3022) 

v_lshlrev_b32 v12, 4, v14 // (op 3022) 

v_lshrrev_b32 v14, 3, v1 // (op 3022) 
v_lshl_add_u32 v16, v14, 7, v12 // (op 3022) 


v_lshlrev_b32 v12, 2, v16 // (op 3022) 

v_lshrrev_b32 v14, 3, v12 // (op 3022) 




s_waitcnt vmcnt(14)
ds_write_b64 v14, v[22:23] // (op 2475) Store local data 

v_and_b32 v12, -4, v1 // (op 2964) 
v_sub_u32 v14, v1, v12 // (op 2964) 

v_lshlrev_b32 v12, 5, v14 // (op 2964) 

v_lshrrev_b32 v14, 2, v1 // (op 2964) 
v_mul_lo_u32 v16, v14, 3 // (op 2964) 

v_lshl_add_u32 v14, v16, 7, v12 // (op 2964) 


v_lshlrev_b32 v12, 2, v14 // (op 2964) 

v_lshrrev_b32 v14, 3, v12 // (op 2964) 




s_waitcnt vmcnt(13)
ds_write_b128 v14, v[26:29] offset:2048 // (op 2476) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v14, v[30:33] offset:2112 // (op 2476) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v14, v[34:37] offset:2176 // (op 2476) Store local data 

v_and_b32 v12, -4, v1 // (op 3075) 
v_sub_u32 v14, v1, v12 // (op 3075) 

v_lshrrev_b32 v12, 2, v1 // (op 3075) 
v_mul_lo_u32 v16, v12, 3 // (op 3075) 

v_lshl_add_u32 v12, v16, 2, v14 // (op 3075) 





s_waitcnt vmcnt(10)
ds_write_b8 v12, v38 offset:14336 // (op 2477) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v12, v39 offset:14340 // (op 2477) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v12, v40 offset:14344 // (op 2477) Store local data 

v_and_b32 v12, -4, v1 // (op 2935) 
v_sub_u32 v14, v1, v12 // (op 2935) 

v_lshlrev_b32 v12, 5, v14 // (op 2935) 

v_lshrrev_b32 v14, 2, v1 // (op 2935) 
v_mul_lo_u32 v16, v14, 3 // (op 2935) 

v_lshl_add_u32 v14, v16, 7, v12 // (op 2935) 


v_lshlrev_b32 v12, 2, v14 // (op 2935) 

v_lshrrev_b32 v14, 3, v12 // (op 2935) 

v_and_b32 v12, -8, v1 // (op 2993) 
v_sub_u32 v16, v1, v12 // (op 2993) 

v_lshlrev_b32 v12, 4, v16 // (op 2993) 

v_lshrrev_b32 v16, 3, v1 // (op 2993) 
v_lshl_add_u32 v18, v16, 7, v12 // (op 2993) 


v_lshlrev_b32 v12, 2, v18 // (op 2993) 

v_lshrrev_b32 v16, 3, v12 // (op 2993) 

v_and_b32 v12, -4, v1 // (op 3046) 
v_sub_u32 v18, v1, v12 // (op 3046) 

v_lshrrev_b32 v12, 2, v1 // (op 3046) 
v_mul_lo_u32 v41, v12, 3 // (op 3046) 

v_lshl_add_u32 v12, v41, 2, v18 // (op 3046) 


v_and_b32 v18, -4, v1 // (op 3122) 
v_sub_u32 v41, v1, v18 // (op 3122) 

v_lshlrev_b32 v18, 5, v41 // (op 3122) 

v_lshrrev_b32 v41, 2, v1 // (op 3122) 
v_mul_lo_u32 v42, v41, 3 // (op 3122) 

v_lshl_add_u32 v41, v42, 7, v18 // (op 3122) 


v_lshlrev_b32 v18, 2, v41 // (op 3122) 

v_lshrrev_b32 v41, 3, v18 // (op 3122) 

v_and_b32 v18, -8, v1 // (op 3151) 
v_sub_u32 v42, v1, v18 // (op 3151) 

v_lshlrev_b32 v18, 4, v42 // (op 3151) 

v_lshrrev_b32 v42, 3, v1 // (op 3151) 
v_lshl_add_u32 v43, v42, 7, v18 // (op 3151) 


v_lshlrev_b32 v18, 2, v43 // (op 3151) 

v_lshrrev_b32 v42, 3, v18 // (op 3151) 

v_and_b32 v18, -4, v1 // (op 3175) 
v_sub_u32 v43, v1, v18 // (op 3175) 

v_lshrrev_b32 v18, 2, v1 // (op 3175) 
v_mul_lo_u32 v59, v18, 3 // (op 3175) 

v_lshl_add_u32 v18, v59, 2, v43 // (op 3175) 


s_mov_b32 s0, 0 // (op 963) call()
s_ashr_i64 s[18:19], s[44:45], 7 // (op 962) 
s_mov_b32 s1, s18 // (op 962) convert

s_and_b32 s18, s1, 4294967294 // (op 962) 

s_cmp_lt_u32 s0, s18 // (op 962) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopBottom_KLoop_962 // (op 962) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopBottom_KLoop_962 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopTop_KLoop_962:
s_add_u32 s0, s0, 2 // (op 964) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3245) 



ds_read_b128 v[60:63], v6 offset:2048 // (op 67) Load local data 
s_barrier  // (op 3258) 



ds_read_b128 v[64:67], v3 // (op 72) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3272) 



ds_read_u8 v43, v5 offset:14336 // (op 77) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:63], v[64:67], a[0:3], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 84) 
ds_read_b128 v[68:71], v6 offset:4096 // (op 344) Load local data 





ds_read_u8 v43, v5 offset:14464 // (op 348) Load local data 



ds_read_b128 v[60:63], v6 offset:6144 // (op 506) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:71], v[64:67], a[4:7], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 350) 





ds_read_u8 v43, v5 offset:14592 // (op 510) Load local data 



ds_read_b128 v[68:71], v6 offset:8192 // (op 668) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:63], v[64:67], a[8:11], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 512) 





ds_read_u8 v43, v5 offset:14720 // (op 672) Load local data 



ds_read_b128 v[60:63], v6 offset:10240 // (op 830) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:71], v[64:67], a[12:15], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 674) 





ds_read_u8 v43, v5 offset:14848 // (op 834) Load local data 



ds_read_b128 v[68:71], v6 offset:12288 // (op 992) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[64:67], a[16:19], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 836) 





ds_read_u8 v43, v5 offset:14976 // (op 996) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:71], v[64:67], a[20:23], v43, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 998) 





v_mov_b32 v43, v19 // (op 82) 

buffer_load_ubyte v20, v43, s[72:75], 0 offen offset:8 // (op 82) Load value


v_mov_b32 v59, v13 // (op 10) 

buffer_load_dwordx2 v[22:23], v59, s[64:67], 0 offen offset:128 // (op 10) Load value

s_lshl_b64 s[18:19], s[26:27], 2 // (op 4) 
v_lshrrev_b64 v[60:61], 3, s[18:19] // (op 4) 


v_mov_b32 v62, v15 // (op 4) 
v_mov_b32 v63, v62 // (op 4) 
buffer_load_dwordx4 v[26:29], v63, s[52:55], 0 offen offset:128 // (op 4) Load value
v_add_u32 v63, v63, v60 // (op 4) 
buffer_load_dwordx4 v[30:33], v63, s[52:55], 0 offen offset:128 // (op 4) Load value
v_add_u32 v63, v63, v60 // (op 4) 
buffer_load_dwordx4 v[34:37], v63, s[52:55], 0 offen offset:128 // (op 4) Load value



v_mov_b64 v[60:61], s[36:37] // (op 17) convert

v_mov_b32 v63, v17 // (op 17) 
v_mov_b32 v64, v63 // (op 17) 
buffer_load_ubyte v38, v64, s[68:71], 0 offen offset:8 // (op 17) Load value
v_add_u32 v64, v64, v60 // (op 17) 
buffer_load_ubyte v39, v64, s[68:71], 0 offen offset:8 // (op 17) Load value
v_add_u32 v64, v64, v60 // (op 17) 
buffer_load_ubyte v40, v64, s[68:71], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 3295) 



s_waitcnt vmcnt(14)
ds_write_b64 v42, v[24:25] offset:15104 // (op 239) Store local data 
s_barrier  // (op 3289) 



s_waitcnt vmcnt(13)
ds_write_b128 v41, v[44:47] offset:17152 // (op 229) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v41, v[48:51] offset:17216 // (op 229) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v41, v[52:55] offset:17280 // (op 229) Store local data 
s_barrier  // (op 3301) 



s_waitcnt vmcnt(10)
ds_write_b8 v18, v56 offset:29440 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v18, v57 offset:29444 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v18, v58 offset:29448 // (op 240) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3286) 



ds_read_b128 v[64:67], v9 offset:17152 // (op 231) Load local data 
s_barrier  // (op 3292) 



ds_read_b128 v[68:71], v4 offset:15104 // (op 233) Load local data 
s_barrier  // (op 3342) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3298) 



ds_read_u8 v60, v8 offset:29440 // (op 235) Load local data 



ds_read_b128 v[72:75], v9 offset:19200 // (op 357) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:67], v[68:71], a[0:3], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 238) 





ds_read_u8 v60, v8 offset:29568 // (op 361) Load local data 



ds_read_b128 v[64:67], v9 offset:21248 // (op 519) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[68:71], a[4:7], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 351) 





ds_read_u8 v60, v8 offset:29696 // (op 523) Load local data 



ds_read_b128 v[72:75], v9 offset:23296 // (op 681) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:67], v[68:71], a[8:11], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 513) 





ds_read_u8 v60, v8 offset:29824 // (op 685) Load local data 



ds_read_b128 v[64:67], v9 offset:25344 // (op 843) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[68:71], a[12:15], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 675) 





ds_read_u8 v60, v8 offset:29952 // (op 847) Load local data 



ds_read_b128 v[72:75], v9 offset:27392 // (op 1005) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:67], v[68:71], a[16:19], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 837) 





s_barrier  // (op 3400) 
ds_read_u8 v60, v8 offset:30080 // (op 1009) Load local data 
s_barrier  // (op 3404) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[68:71], a[20:23], v60, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 999) 






buffer_load_ubyte v21, v43, s[72:75], 0 offen offset:12 // (op 237) Load value




v_add_u32 v19, 8, v19 // (op 2889) 
buffer_load_dwordx2 v[24:25], v59, s[64:67], 0 offen offset:192 // (op 241) Load value


v_add_u32 v13, 128, v13 // (op 2761) 
s_lshl_b64 s[18:19], s[26:27], 2 // (op 228) 
v_lshrrev_b64 v[60:61], 3, s[18:19] // (op 228) 


v_mov_b32 v43, v62 // (op 228) 
buffer_load_dwordx4 v[44:47], v43, s[52:55], 0 offen offset:192 // (op 228) Load value
v_add_u32 v43, v43, v60 // (op 228) 
buffer_load_dwordx4 v[48:51], v43, s[52:55], 0 offen offset:192 // (op 228) Load value
v_add_u32 v43, v43, v60 // (op 228) 
buffer_load_dwordx4 v[52:55], v43, s[52:55], 0 offen offset:192 // (op 228) Load value




v_add_u32 v15, 128, v15 // (op 2697) 
v_mov_b64 v[60:61], s[36:37] // (op 242) convert

v_mov_b32 v43, v63 // (op 242) 
buffer_load_ubyte v56, v43, s[68:71], 0 offen offset:12 // (op 242) Load value
v_add_u32 v43, v43, v60 // (op 242) 
buffer_load_ubyte v57, v43, s[68:71], 0 offen offset:12 // (op 242) Load value
v_add_u32 v43, v43, v60 // (op 242) 
buffer_load_ubyte v58, v43, s[68:71], 0 offen offset:12 // (op 242) Load value



s_barrier  // (op 3261) 
v_add_u32 v17, 8, v17 // (op 2825) 



s_waitcnt vmcnt(14)
ds_write_b64 v16, v[22:23] // (op 13) Store local data 
s_barrier  // (op 3248) 



s_waitcnt vmcnt(13)
ds_write_b128 v14, v[26:29] offset:2048 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v14, v[30:33] offset:2112 // (op 6) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v14, v[34:37] offset:2176 // (op 6) Store local data 
s_barrier  // (op 3275) 



s_waitcnt vmcnt(10)
ds_write_b8 v12, v38 offset:14336 // (op 21) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v12, v39 offset:14340 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v12, v40 offset:14344 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[44:45], 7 // (op 962) 
s_mov_b32 s1, s18 // (op 962) convert

s_and_b32 s18, s1, 4294967294 // (op 962) 

s_cmp_lt_u32 s0, s18 // (op 962) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopTop_KLoop_962 // (op 962) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopTop_KLoop_962 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopBottom_KLoop_962:















s_mul_hi_i32 s0, s16, s47 // (op 2655) 
v_lshrrev_b32 v12, 2, v1 // (op 2655) 
s_add_i32 s1, s0, s16 // (op 2655) 

v_mul_lo_u32 v13, v12, 3 // (op 2655) 

s_ashr_i32 s0, s1, 31 // (op 2655) 
v_and_b32 v12, -4, v1 // (op 2655) 
s_cmp_eq_i32 s47, 0 // (op 2655) 
s_mov_b32 s18, scc // (op 2655) 
v_sub_u32 v14, v1, v12 // (op 2655) 

s_cmp_eq_i32 s18, 0 // (op 2655) 
s_cselect_b32 s19, 0, -1 // (op 2655) 

v_lshlrev_b32 v12, 5, v14 // (op 2655) 

s_and_b32 s18, s11, 31 // (op 2655) 
s_lshl_b32 s22, 1, s18 // (op 2655) 
s_add_i32 s23, s22, s19 // (op 2655) 


s_and_b32 s19, s0, s23 // (op 2655) 


s_add_i32 s0, s1, s19 // (op 2655) 


s_ashr_i32 s1, s0, s18 // (op 2655) 


s_lshl_b32 s0, s11, 24 // (op 2655) 
s_ashr_i32 s18, s0, 31 // (op 2655) 

s_xor_b32 s0, s1, s18 // (op 2655) 

s_sub_i32 s1, s0, s18 // (op 2655) 


s_mul_hi_i32 s0, s1, s12 // (op 2655) 
s_add_i32 s18, s0, s1 // (op 2655) 

s_ashr_i32 s0, s18, 31 // (op 2655) 
s_cmp_eq_i32 s12, 0 // (op 2655) 
s_mov_b32 s19, scc // (op 2655) 
s_cmp_eq_i32 s19, 0 // (op 2655) 
s_cselect_b32 s22, 0, -1 // (op 2655) 

s_and_b32 s19, s13, 31 // (op 2655) 
s_lshl_b32 s23, 1, s19 // (op 2655) 
s_add_i32 s24, s23, s22 // (op 2655) 


s_and_b32 s25, s0, s24 // (op 2655) 


s_add_i32 s0, s18, s25 // (op 2655) 


s_ashr_i32 s18, s0, s19 // (op 2655) 


s_lshl_b32 s0, s13, 24 // (op 2655) 
s_ashr_i32 s19, s0, 31 // (op 2655) 

s_xor_b32 s0, s18, s19 // (op 2655) 

s_sub_i32 s18, s0, s19 // (op 2655) 


s_mul_i32 s0, s1, s48 // (op 2655) 

s_sub_u32 s1, s16, s0 // (op 2655) 

s_mul_i32 s0, s18, s48 // (op 2655) 
s_add_u32 s0, s0, s1 // (op 2655) 


s_mul_hi_i32 s1, s16, s8 // (op 2655) 
s_add_i32 s18, s1, s16 // (op 2655) 

s_ashr_i32 s1, s18, 31 // (op 2655) 
s_cmp_eq_i32 s8, 0 // (op 2655) 
s_mov_b32 s19, scc // (op 2655) 
s_cmp_eq_i32 s19, 0 // (op 2655) 
s_cselect_b32 s22, 0, -1 // (op 2655) 

s_and_b32 s19, s10, 31 // (op 2655) 
s_lshl_b32 s23, 1, s19 // (op 2655) 
s_add_i32 s24, s23, s22 // (op 2655) 


s_and_b32 s25, s1, s24 // (op 2655) 


s_add_i32 s1, s18, s25 // (op 2655) 


s_ashr_i32 s18, s1, s19 // (op 2655) 


s_lshl_b32 s1, s10, 24 // (op 2655) 
s_ashr_i32 s19, s1, 31 // (op 2655) 

s_xor_b32 s1, s18, s19 // (op 2655) 

s_sub_i32 s18, s1, s19 // (op 2655) 


s_add_u32 s50, s6, 32 // (op 2655) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2655) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2655) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2655) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2655) 

s_mul_i32 s1, s48, s24 // (op 2655) 

s_mul_i32 s19, s14, s1 // (op 2655) 

s_mul_i32 s1, s18, s19 // (op 2655) 

s_sub_u32 s19, s16, s1 // (op 2655) 

s_mov_b32 s1, s19 // (op 2655) 
s_mul_hi_i32 s22, s1, s15 // (op 2655) 
s_add_i32 s23, s22, s1 // (op 2655) 


s_ashr_i32 s1, s23, 31 // (op 2655) 
s_cmp_eq_i32 s15, 0 // (op 2655) 
s_mov_b32 s22, scc // (op 2655) 
s_cmp_eq_i32 s22, 0 // (op 2655) 
s_cselect_b32 s24, 0, -1 // (op 2655) 

s_and_b32 s22, s9, 31 // (op 2655) 
s_lshl_b32 s25, 1, s22 // (op 2655) 
s_add_i32 s49, s25, s24 // (op 2655) 


s_and_b32 s24, s1, s49 // (op 2655) 


s_add_i32 s1, s23, s24 // (op 2655) 


s_ashr_i32 s23, s1, s22 // (op 2655) 


s_lshl_b32 s1, s9, 24 // (op 2655) 
s_ashr_i32 s22, s1, 31 // (op 2655) 

s_xor_b32 s1, s23, s22 // (op 2655) 

s_sub_i32 s23, s1, s22 // (op 2655) 


s_mul_i32 s1, s46, s48 // (op 2655) 
s_sub_i32 s22, s4, s1 // (op 2655) 
s_mul_i32 s24, s23, s22 // (op 2655) 


s_sub_u32 s25, s19, s24 // (op 2655) 


s_add_u32 s19, s25, s1 // (op 2655) 


s_cmp_eq_i32 s18, 0 // (op 2655) 
s_mov_b32 s1, scc // (op 2655) 

s_cmp_eq_i32 s1, 0 // (op 2655) 
s_cselect_b32 s18, s19, s0 // (op 2655) 



v_mov_b32 v14, 192 // (op 2655) 
v_mul_lo_u32 v15, s18, v14 // (op 2655) 

v_add_u32 v15, v15, v13 // (op 2655) 


s_ashr_i64 s[0:1], s[44:45], 7 // (op 2655) 
s_lshr_b32 s18, s0, 1 // (op 2655) 

v_lshl_add_u32 v13, s18, 8, v12 // (op 2655) 


v_mov_b32 v23, 0 // (op 2655) convert
v_mov_b32 v22, v13 // (op 2655) convert

v_mul_lo_u32 v13, v15, s27 // (op 2655) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 2655) most significant: high of low * low
v_mul_lo_u32 v12, v15, s26 // (op 2655) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 2655) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v22 // (op 2655) least significant half
v_addc_co_u32 v13, s[0:1], v13, v23, s[0:1] // (op 2655) most significant half



v_lshlrev_b64 v[22:23], 2, v[12:13] // (op 2655) 

v_lshrrev_b64 v[12:13], 3, v[22:23] // (op 2655) 

v_mov_b32 v14, v12 // (op 2655) convert

s_mul_i32 s1, s26, s31 // (op 2657) most significant: low of low * high
s_mul_i32 s18, s27, s30 // (op 2657) most significant: low of high * low
s_mul_i32 s0, s26, s30 // (op 2657) least significant
s_mul_hi_u32 s19, s26, s30 // (op 2657) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2657) most significant: sum
s_add_u32 s1, s1, s19 // (op 2657) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2657) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2657) 

s_mov_b32 s52, s28 // (op 2657) reg expression
s_mov_b32 s53, s29 // (op 2657) reg expression
s_mov_b32 s54, s0 // (op 2657) reg expression
s_mov_b32 s55, 131072 // (op 2657) call()

s_mul_hi_i32 s0, s16, s47 // (op 2727) 
v_lshrrev_b32 v12, 3, v1 // (op 2727) 
s_add_i32 s1, s0, s16 // (op 2727) 

v_and_b32 v13, -8, v1 // (op 2727) 
s_ashr_i32 s0, s1, 31 // (op 2727) 
v_sub_u32 v15, v1, v13 // (op 2727) 

s_cmp_eq_i32 s47, 0 // (op 2727) 
s_mov_b32 s18, scc // (op 2727) 
v_lshlrev_b32 v13, 4, v15 // (op 2727) 

s_cmp_eq_i32 s18, 0 // (op 2727) 
s_cselect_b32 s19, 0, -1 // (op 2727) 

s_and_b32 s18, s11, 31 // (op 2727) 
s_lshl_b32 s22, 1, s18 // (op 2727) 
s_add_i32 s23, s22, s19 // (op 2727) 


s_and_b32 s19, s0, s23 // (op 2727) 


s_add_i32 s0, s1, s19 // (op 2727) 


s_ashr_i32 s1, s0, s18 // (op 2727) 


s_lshl_b32 s0, s11, 24 // (op 2727) 
s_ashr_i32 s18, s0, 31 // (op 2727) 

s_xor_b32 s0, s1, s18 // (op 2727) 

s_sub_i32 s1, s0, s18 // (op 2727) 


s_mul_hi_i32 s0, s1, s12 // (op 2727) 
s_add_i32 s18, s0, s1 // (op 2727) 

s_ashr_i32 s0, s18, 31 // (op 2727) 
s_cmp_eq_i32 s12, 0 // (op 2727) 
s_mov_b32 s19, scc // (op 2727) 
s_cmp_eq_i32 s19, 0 // (op 2727) 
s_cselect_b32 s22, 0, -1 // (op 2727) 

s_and_b32 s19, s13, 31 // (op 2727) 
s_lshl_b32 s23, 1, s19 // (op 2727) 
s_add_i32 s24, s23, s22 // (op 2727) 


s_and_b32 s25, s0, s24 // (op 2727) 


s_add_i32 s0, s18, s25 // (op 2727) 


s_ashr_i32 s18, s0, s19 // (op 2727) 


s_lshl_b32 s0, s13, 24 // (op 2727) 
s_ashr_i32 s19, s0, 31 // (op 2727) 

s_xor_b32 s0, s18, s19 // (op 2727) 

s_sub_i32 s18, s0, s19 // (op 2727) 


s_add_u32 s50, s6, 32 // (op 2727) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2727) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2727) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2727) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2727) 

s_mul_i32 s0, s18, s24 // (op 2727) 

s_sub_i32 s18, s1, s0 // (op 2727) 


s_mul_hi_i32 s19, s16, s8 // (op 2727) 
s_add_i32 s0, s19, s16 // (op 2727) 

s_ashr_i32 s1, s0, 31 // (op 2727) 
s_cmp_eq_i32 s8, 0 // (op 2727) 
s_mov_b32 s19, scc // (op 2727) 
s_cmp_eq_i32 s19, 0 // (op 2727) 
s_cselect_b32 s22, 0, -1 // (op 2727) 

s_and_b32 s19, s10, 31 // (op 2727) 
s_lshl_b32 s23, 1, s19 // (op 2727) 
s_add_i32 s49, s23, s22 // (op 2727) 


s_and_b32 s22, s1, s49 // (op 2727) 


s_add_i32 s1, s0, s22 // (op 2727) 


s_ashr_i32 s0, s1, s19 // (op 2727) 


s_lshl_b32 s1, s10, 24 // (op 2727) 
s_ashr_i32 s19, s1, 31 // (op 2727) 

s_xor_b32 s1, s0, s19 // (op 2727) 

s_sub_i32 s0, s1, s19 // (op 2727) 


s_mul_i32 s1, s48, s24 // (op 2727) 

s_mul_i32 s19, s14, s1 // (op 2727) 

s_mul_i32 s1, s0, s19 // (op 2727) 

s_sub_u32 s19, s16, s1 // (op 2727) 

s_mul_hi_i32 s1, s19, s15 // (op 2727) 
s_add_i32 s22, s1, s19 // (op 2727) 


s_ashr_i32 s1, s22, 31 // (op 2727) 
s_cmp_eq_i32 s15, 0 // (op 2727) 
s_mov_b32 s19, scc // (op 2727) 
s_cmp_eq_i32 s19, 0 // (op 2727) 
s_cselect_b32 s23, 0, -1 // (op 2727) 

s_and_b32 s19, s9, 31 // (op 2727) 
s_lshl_b32 s24, 1, s19 // (op 2727) 
s_add_i32 s25, s24, s23 // (op 2727) 


s_and_b32 s23, s1, s25 // (op 2727) 


s_add_i32 s1, s22, s23 // (op 2727) 


s_ashr_i32 s22, s1, s19 // (op 2727) 


s_lshl_b32 s1, s9, 24 // (op 2727) 
s_ashr_i32 s19, s1, 31 // (op 2727) 

s_xor_b32 s1, s22, s19 // (op 2727) 

s_sub_i32 s22, s1, s19 // (op 2727) 


s_cmp_eq_i32 s0, 0 // (op 2727) 
s_mov_b32 s1, scc // (op 2727) 

s_cmp_eq_i32 s1, 0 // (op 2727) 
s_cselect_b32 s0, s22, s18 // (op 2727) 



v_lshl_add_u32 v15, s0, 5, v12 // (op 2727) 


v_mul_lo_u32 v23, v15, s33 // (op 2727) most significant: low of low * high
v_mul_hi_u32 v12, v15, s32 // (op 2727) most significant: high of low * low
v_mul_lo_u32 v22, v15, s32 // (op 2727) least significant: low of low * low
v_add_u32 v23, v23, v12 // (op 2727) most significant: sum


s_ashr_i64 s[0:1], s[44:45], 7 // (op 2727) 
s_lshr_b32 s18, s0, 1 // (op 2727) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 2727) 


v_mov_b32 v27, 0 // (op 2727) convert
v_mov_b32 v26, v12 // (op 2727) convert

v_add_co_u32 v12, s[0:1], v26, v22 // (op 2727) least significant half
v_addc_co_u32 v13, s[0:1], v27, v23, s[0:1] // (op 2727) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 2727) 


v_lshrrev_b64 v[22:23], 3, v[12:13] // (op 2727) 

v_mov_b32 v12, v22 // (op 2727) convert

s_mul_i32 s1, s32, s7 // (op 2729) most significant: low of low * high
s_mul_i32 s18, s33, s6 // (op 2729) most significant: low of high * low
s_mul_i32 s0, s32, s6 // (op 2729) least significant
s_mul_hi_u32 s19, s32, s6 // (op 2729) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2729) most significant: sum
s_add_u32 s1, s1, s19 // (op 2729) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2729) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 2729) 

s_mov_b32 s64, s34 // (op 2729) reg expression
s_mov_b32 s65, s35 // (op 2729) reg expression
s_mov_b32 s66, s0 // (op 2729) reg expression
s_mov_b32 s67, 131072 // (op 2729) call()

s_mul_hi_i32 s0, s16, s47 // (op 2791) 
v_lshrrev_b32 v13, 2, v1 // (op 2791) 
s_add_i32 s1, s0, s16 // (op 2791) 

v_mul_lo_u32 v15, v13, 3 // (op 2791) 

s_ashr_i32 s0, s1, 31 // (op 2791) 
v_and_b32 v13, -4, v1 // (op 2791) 
s_cmp_eq_i32 s47, 0 // (op 2791) 
s_mov_b32 s18, scc // (op 2791) 
v_sub_u32 v16, v1, v13 // (op 2791) 

s_cmp_eq_i32 s18, 0 // (op 2791) 
s_cselect_b32 s19, 0, -1 // (op 2791) 

s_and_b32 s18, s11, 31 // (op 2791) 
s_lshl_b32 s22, 1, s18 // (op 2791) 
s_add_i32 s23, s22, s19 // (op 2791) 


s_and_b32 s19, s0, s23 // (op 2791) 


s_add_i32 s0, s1, s19 // (op 2791) 


s_ashr_i32 s1, s0, s18 // (op 2791) 


s_lshl_b32 s0, s11, 24 // (op 2791) 
s_ashr_i32 s18, s0, 31 // (op 2791) 

s_xor_b32 s0, s1, s18 // (op 2791) 

s_sub_i32 s1, s0, s18 // (op 2791) 


s_mul_hi_i32 s0, s1, s12 // (op 2791) 
s_add_i32 s18, s0, s1 // (op 2791) 

s_ashr_i32 s0, s18, 31 // (op 2791) 
s_cmp_eq_i32 s12, 0 // (op 2791) 
s_mov_b32 s19, scc // (op 2791) 
s_cmp_eq_i32 s19, 0 // (op 2791) 
s_cselect_b32 s22, 0, -1 // (op 2791) 

s_and_b32 s19, s13, 31 // (op 2791) 
s_lshl_b32 s23, 1, s19 // (op 2791) 
s_add_i32 s24, s23, s22 // (op 2791) 


s_and_b32 s25, s0, s24 // (op 2791) 


s_add_i32 s0, s18, s25 // (op 2791) 


s_ashr_i32 s18, s0, s19 // (op 2791) 


s_lshl_b32 s0, s13, 24 // (op 2791) 
s_ashr_i32 s19, s0, 31 // (op 2791) 

s_xor_b32 s0, s18, s19 // (op 2791) 

s_sub_i32 s18, s0, s19 // (op 2791) 


s_mul_i32 s0, s1, s48 // (op 2791) 

s_sub_u32 s1, s16, s0 // (op 2791) 

s_mul_i32 s0, s18, s48 // (op 2791) 
s_add_u32 s0, s0, s1 // (op 2791) 


s_mul_hi_i32 s1, s16, s8 // (op 2791) 
s_add_i32 s18, s1, s16 // (op 2791) 

s_ashr_i32 s1, s18, 31 // (op 2791) 
s_cmp_eq_i32 s8, 0 // (op 2791) 
s_mov_b32 s19, scc // (op 2791) 
s_cmp_eq_i32 s19, 0 // (op 2791) 
s_cselect_b32 s22, 0, -1 // (op 2791) 

s_and_b32 s19, s10, 31 // (op 2791) 
s_lshl_b32 s23, 1, s19 // (op 2791) 
s_add_i32 s24, s23, s22 // (op 2791) 


s_and_b32 s25, s1, s24 // (op 2791) 


s_add_i32 s1, s18, s25 // (op 2791) 


s_ashr_i32 s18, s1, s19 // (op 2791) 


s_lshl_b32 s1, s10, 24 // (op 2791) 
s_ashr_i32 s19, s1, 31 // (op 2791) 

s_xor_b32 s1, s18, s19 // (op 2791) 

s_sub_i32 s18, s1, s19 // (op 2791) 


s_add_u32 s50, s6, 32 // (op 2791) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2791) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2791) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2791) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2791) 

s_mul_i32 s1, s48, s24 // (op 2791) 

s_mul_i32 s19, s14, s1 // (op 2791) 

s_mul_i32 s1, s18, s19 // (op 2791) 

s_sub_u32 s19, s16, s1 // (op 2791) 

s_mov_b32 s1, s19 // (op 2791) 
s_mul_hi_i32 s22, s1, s15 // (op 2791) 
s_add_i32 s23, s22, s1 // (op 2791) 


s_ashr_i32 s1, s23, 31 // (op 2791) 
s_cmp_eq_i32 s15, 0 // (op 2791) 
s_mov_b32 s22, scc // (op 2791) 
s_cmp_eq_i32 s22, 0 // (op 2791) 
s_cselect_b32 s24, 0, -1 // (op 2791) 

s_and_b32 s22, s9, 31 // (op 2791) 
s_lshl_b32 s25, 1, s22 // (op 2791) 
s_add_i32 s49, s25, s24 // (op 2791) 


s_and_b32 s24, s1, s49 // (op 2791) 


s_add_i32 s1, s23, s24 // (op 2791) 


s_ashr_i32 s23, s1, s22 // (op 2791) 


s_lshl_b32 s1, s9, 24 // (op 2791) 
s_ashr_i32 s22, s1, 31 // (op 2791) 

s_xor_b32 s1, s23, s22 // (op 2791) 

s_sub_i32 s23, s1, s22 // (op 2791) 


s_mul_i32 s1, s46, s48 // (op 2791) 
s_sub_i32 s22, s4, s1 // (op 2791) 
s_mul_i32 s24, s23, s22 // (op 2791) 


s_sub_u32 s25, s19, s24 // (op 2791) 


s_add_u32 s19, s25, s1 // (op 2791) 


s_cmp_eq_i32 s18, 0 // (op 2791) 
s_mov_b32 s1, scc // (op 2791) 

s_cmp_eq_i32 s1, 0 // (op 2791) 
s_cselect_b32 s18, s19, s0 // (op 2791) 



v_mov_b32 v13, 192 // (op 2791) 
v_mul_lo_u32 v17, s18, v13 // (op 2791) 

v_add_u32 v17, v17, v15 // (op 2791) 


s_ashr_i64 s[0:1], s[44:45], 7 // (op 2791) 
s_lshr_b32 s18, s0, 1 // (op 2791) 

v_lshl_add_u32 v13, s18, 3, v16 // (op 2791) 


v_mul_lo_u32 v15, v17, s36 // (op 2791) 
v_add_u32 v15, v15, v13 // (op 2791) 


s_mul_i32 s1, s36, s31 // (op 2793) most significant: low of low * high
s_mul_i32 s18, s37, s30 // (op 2793) most significant: low of high * low
s_mul_i32 s0, s36, s30 // (op 2793) least significant
s_mul_hi_u32 s19, s36, s30 // (op 2793) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2793) most significant: sum
s_add_u32 s1, s1, s19 // (op 2793) most significant: sum


s_mov_b32 s68, s38 // (op 2793) reg expression
s_mov_b32 s69, s39 // (op 2793) reg expression
s_mov_b32 s70, s0 // (op 2793) reg expression
s_mov_b32 s71, 131072 // (op 2793) call()

s_mul_hi_i32 s0, s16, s47 // (op 2855) 
v_lshrrev_b32 v13, 6, v1 // (op 2855) 
s_add_i32 s1, s0, s16 // (op 2855) 

v_and_b32 v16, 1, v13 // (op 2855) 

s_ashr_i32 s0, s1, 31 // (op 2855) 
v_and_b32 v13, 15, v1 // (op 2855) 
s_cmp_eq_i32 s47, 0 // (op 2855) 
s_mov_b32 s18, scc // (op 2855) 
v_lshl_add_u32 v17, v16, 4, v13 // (op 2855) 


s_cmp_eq_i32 s18, 0 // (op 2855) 
s_cselect_b32 s19, 0, -1 // (op 2855) 

v_and_b32 v13, 63, v1 // (op 2855) 
s_and_b32 s18, s11, 31 // (op 2855) 
v_lshrrev_b32 v16, 4, v13 // (op 2855) 

s_lshl_b32 s22, 1, s18 // (op 2855) 
s_add_i32 s23, s22, s19 // (op 2855) 


s_and_b32 s19, s0, s23 // (op 2855) 


s_add_i32 s0, s1, s19 // (op 2855) 


s_ashr_i32 s1, s0, s18 // (op 2855) 


s_lshl_b32 s0, s11, 24 // (op 2855) 
s_ashr_i32 s18, s0, 31 // (op 2855) 

s_xor_b32 s0, s1, s18 // (op 2855) 

s_sub_i32 s1, s0, s18 // (op 2855) 


s_mul_hi_i32 s0, s1, s12 // (op 2855) 
s_add_i32 s18, s0, s1 // (op 2855) 

s_ashr_i32 s0, s18, 31 // (op 2855) 
s_cmp_eq_i32 s12, 0 // (op 2855) 
s_mov_b32 s19, scc // (op 2855) 
s_cmp_eq_i32 s19, 0 // (op 2855) 
s_cselect_b32 s22, 0, -1 // (op 2855) 

s_and_b32 s19, s13, 31 // (op 2855) 
s_lshl_b32 s23, 1, s19 // (op 2855) 
s_add_i32 s24, s23, s22 // (op 2855) 


s_and_b32 s25, s0, s24 // (op 2855) 


s_add_i32 s0, s18, s25 // (op 2855) 


s_ashr_i32 s18, s0, s19 // (op 2855) 


s_lshl_b32 s0, s13, 24 // (op 2855) 
s_ashr_i32 s19, s0, 31 // (op 2855) 

s_xor_b32 s0, s18, s19 // (op 2855) 

s_sub_i32 s18, s0, s19 // (op 2855) 


s_add_u32 s50, s6, 32 // (op 2855) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 2855) End of Int64 add, unlocking SCC
s_sub_u32 s22, s50, 1 // (op 2855) least significant half; sets scc
s_subb_u32 s23, s51, 0 // (op 2855) most significant half; uses scc

s_ashr_i64 s[24:25], s[22:23], 5 // (op 2855) 

s_mul_i32 s0, s18, s24 // (op 2855) 

s_sub_i32 s18, s1, s0 // (op 2855) 


s_mul_hi_i32 s19, s16, s8 // (op 2855) 
s_add_i32 s0, s19, s16 // (op 2855) 

s_ashr_i32 s1, s0, 31 // (op 2855) 
s_cmp_eq_i32 s8, 0 // (op 2855) 
s_mov_b32 s19, scc // (op 2855) 
s_cmp_eq_i32 s19, 0 // (op 2855) 
s_cselect_b32 s22, 0, -1 // (op 2855) 

s_and_b32 s19, s10, 31 // (op 2855) 
s_lshl_b32 s23, 1, s19 // (op 2855) 
s_add_i32 s49, s23, s22 // (op 2855) 


s_and_b32 s22, s1, s49 // (op 2855) 


s_add_i32 s1, s0, s22 // (op 2855) 


s_ashr_i32 s0, s1, s19 // (op 2855) 


s_lshl_b32 s1, s10, 24 // (op 2855) 
s_ashr_i32 s19, s1, 31 // (op 2855) 

s_xor_b32 s1, s0, s19 // (op 2855) 

s_sub_i32 s0, s1, s19 // (op 2855) 


s_mul_i32 s1, s48, s24 // (op 2855) 

s_mul_i32 s19, s14, s1 // (op 2855) 

s_mul_i32 s1, s0, s19 // (op 2855) 

s_sub_u32 s19, s16, s1 // (op 2855) 

s_mul_hi_i32 s1, s19, s15 // (op 2855) 
s_add_i32 s22, s1, s19 // (op 2855) 


s_ashr_i32 s1, s22, 31 // (op 2855) 
s_cmp_eq_i32 s15, 0 // (op 2855) 
s_mov_b32 s19, scc // (op 2855) 
s_cmp_eq_i32 s19, 0 // (op 2855) 
s_cselect_b32 s23, 0, -1 // (op 2855) 

s_and_b32 s19, s9, 31 // (op 2855) 
s_lshl_b32 s24, 1, s19 // (op 2855) 
s_add_i32 s25, s24, s23 // (op 2855) 


s_and_b32 s23, s1, s25 // (op 2855) 


s_add_i32 s1, s22, s23 // (op 2855) 


s_ashr_i32 s22, s1, s19 // (op 2855) 


s_lshl_b32 s1, s9, 24 // (op 2855) 
s_ashr_i32 s19, s1, 31 // (op 2855) 

s_xor_b32 s1, s22, s19 // (op 2855) 

s_sub_i32 s22, s1, s19 // (op 2855) 


s_cmp_eq_i32 s0, 0 // (op 2855) 
s_mov_b32 s1, scc // (op 2855) 

s_cmp_eq_i32 s1, 0 // (op 2855) 
s_cselect_b32 s0, s22, s18 // (op 2855) 



v_lshl_add_u32 v13, s0, 5, v17 // (op 2855) 


v_mul_lo_u32 v17, v13, s40 // (op 2855) 

s_ashr_i64 s[0:1], s[44:45], 7 // (op 2855) 
s_lshr_b32 s18, s0, 1 // (op 2855) 

v_lshl_add_u32 v13, s18, 3, v16 // (op 2855) 


v_add_u32 v16, v13, v17 // (op 2855) 


s_mul_i32 s1, s40, s7 // (op 2857) most significant: low of low * high
s_mul_i32 s18, s41, s6 // (op 2857) most significant: low of high * low
s_mul_i32 s0, s40, s6 // (op 2857) least significant
s_mul_hi_u32 s19, s40, s6 // (op 2857) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2857) most significant: sum
s_add_u32 s1, s1, s19 // (op 2857) most significant: sum


s_mov_b32 s72, s42 // (op 2857) reg expression
s_mov_b32 s73, s43 // (op 2857) reg expression
s_mov_b32 s74, s0 // (op 2857) reg expression
s_mov_b32 s75, 131072 // (op 2857) call()

v_and_b32 v13, -4, v1 // (op 2948) 
v_sub_u32 v17, v1, v13 // (op 2948) 

v_lshlrev_b32 v13, 5, v17 // (op 2948) 

v_lshrrev_b32 v17, 2, v1 // (op 2948) 
v_mul_lo_u32 v18, v17, 3 // (op 2948) 

v_lshl_add_u32 v17, v18, 7, v13 // (op 2948) 


v_lshlrev_b32 v13, 2, v17 // (op 2948) 

v_lshrrev_b32 v17, 3, v13 // (op 2948) 

v_and_b32 v13, -8, v1 // (op 3006) 
v_sub_u32 v18, v1, v13 // (op 3006) 

v_lshlrev_b32 v13, 4, v18 // (op 3006) 

v_lshrrev_b32 v18, 3, v1 // (op 3006) 
v_lshl_add_u32 v19, v18, 7, v13 // (op 3006) 


v_lshlrev_b32 v13, 2, v19 // (op 3006) 

v_lshrrev_b32 v18, 3, v13 // (op 3006) 

v_and_b32 v13, -4, v1 // (op 3059) 
v_sub_u32 v19, v1, v13 // (op 3059) 

v_lshrrev_b32 v13, 2, v1 // (op 3059) 
v_mul_lo_u32 v20, v13, 3 // (op 3059) 

v_lshl_add_u32 v13, v20, 2, v19 // (op 3059) 


s_ashr_i64 s[0:1], s[44:45], 7 // (op 968) 
s_and_b32 s18, s0, 4294967294 // (op 968) 

s_ashr_i64 s[0:1], s[44:45], 7 // (op 967) 
s_mov_b32 s19, s0 // (op 967) convert

s_cmp_lt_u32 s18, s19 // (op 967) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_967 // (op 967) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_967 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopTop_KLoopTail_967:

s_add_u32 s18, s18, 1 // (op 969) 
s_lshl_b64 s[0:1], s[26:27], 2 // (op 170) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 170) 


v_mov_b32 v19, v14 // (op 170) 
v_mov_b32 v22, v19 // (op 170) 
buffer_load_dwordx4 v[24:27], v22, s[52:55], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[28:31], v22, s[52:55], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[32:35], v22, s[52:55], 0 offen // (op 170) Load value



v_add_u32 v14, 64, v14 // (op 2645) 


s_barrier  // (op 3251) 
v_mov_b32 v19, v12 // (op 183) 




s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] offset:2048 // (op 171) Store local data 
buffer_load_dwordx2 v[20:21], v19, s[64:67], 0 offen // (op 183) Load value

v_add_u32 v12, 64, v12 // (op 2719) 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[28:31] offset:2112 // (op 171) Store local data 

v_mov_b64 v[22:23], s[36:37] // (op 184) convert

v_mov_b32 v19, v15 // (op 184) 
v_mov_b32 v36, v19 // (op 184) 
buffer_load_ubyte v38, v36, s[68:71], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v36, v36, v22 // (op 184) 
buffer_load_ubyte v39, v36, s[68:71], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v36, v36, v22 // (op 184) 
s_waitcnt vmcnt(3)
ds_write_b128 v17, v[32:35] offset:2176 // (op 171) Store local data 
buffer_load_ubyte v40, v36, s[68:71], 0 offen offset: 0 // (op 184) Load value




v_add_u32 v15, 4, v15 // (op 2783) 
s_barrier  // (op 3264) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3254) 






ds_read_b128 v[22:25], v6 offset:2048 // (op 173) Load local data 
s_waitcnt vmcnt(3)
ds_write_b64 v18, v[20:21] // (op 181) Store local data 

s_barrier  // (op 3278) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3268) 






ds_read_b128 v[26:29], v3 // (op 175) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v13, v38 offset:14336 // (op 182) Store local data 
s_waitcnt vmcnt(1)
ds_write_b8 v13, v39 offset:14340 // (op 182) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v13, v40 offset:14344 // (op 182) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3282) 



ds_read_u8 v19, v5 offset:14336 // (op 177) Load local data 


v_mov_b32 v20, v16 // (op 179) 

buffer_load_ubyte v21, v20, s[72:75], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v16, 4, v16 // (op 2847) 



ds_read_b128 v[30:33], v6 offset:4096 // (op 325) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 180) 





ds_read_u8 v19, v5 offset:14464 // (op 329) Load local data 



ds_read_b128 v[22:25], v6 offset:6144 // (op 487) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[26:29], a[4:7], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 331) 





ds_read_u8 v19, v5 offset:14592 // (op 491) Load local data 



ds_read_b128 v[30:33], v6 offset:8192 // (op 649) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[26:29], a[8:11], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 493) 





ds_read_u8 v19, v5 offset:14720 // (op 653) Load local data 



ds_read_b128 v[22:25], v6 offset:10240 // (op 811) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:33], v[26:29], a[12:15], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 655) 





ds_read_u8 v19, v5 offset:14848 // (op 815) Load local data 



ds_read_b128 v[30:33], v6 offset:12288 // (op 973) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[26:29], a[16:19], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 817) 





ds_read_u8 v19, v5 offset:14976 // (op 977) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:33], v[26:29], a[20:23], v19, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 979) 




s_ashr_i64 s[0:1], s[44:45], 7 // (op 967) 
s_mov_b32 s19, s0 // (op 967) convert

s_cmp_lt_u32 s18, s19 // (op 967) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopTop_KLoopTail_967 // (op 967) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopTop_KLoopTail_967 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_967:






















s_barrier  // (op 3388) 



















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 

v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 
v_lshrrev_b32 v14, 16, v18 // (op 52) 
v_lshrrev_b32 v15, 16, v19 // (op 52) 


v_pack_b32_f16 v8, v12, v13 // (op 54) 
v_pack_b32_f16 v9, v14, v15 // (op 54) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen // (op 54) Store value


v_accvgpr_read v12, a4 // (op 1253) 
v_accvgpr_read v13, a5 // (op 1253) 
v_accvgpr_read v14, a6 // (op 1253) 
v_accvgpr_read v15, a7 // (op 1253) 
v_mul_f32 v16, v11, v12 // (op 1253) 
v_mul_f32 v17, v11, v13 // (op 1253) 
v_mul_f32 v18, v11, v14 // (op 1253) 
v_mul_f32 v19, v11, v15 // (op 1253) 

v_lshrrev_b32 v12, 16, v16 // (op 1255) 
v_lshrrev_b32 v13, 16, v17 // (op 1255) 
v_lshrrev_b32 v14, 16, v18 // (op 1255) 
v_lshrrev_b32 v15, 16, v19 // (op 1255) 


v_pack_b32_f16 v8, v12, v13 // (op 1256) 
v_pack_b32_f16 v9, v14, v15 // (op 1256) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen offset:64 // (op 1256) Store value


v_accvgpr_read v12, a8 // (op 1273) 
v_accvgpr_read v13, a9 // (op 1273) 
v_accvgpr_read v14, a10 // (op 1273) 
v_accvgpr_read v15, a11 // (op 1273) 
v_mul_f32 v16, v11, v12 // (op 1273) 
v_mul_f32 v17, v11, v13 // (op 1273) 
v_mul_f32 v18, v11, v14 // (op 1273) 
v_mul_f32 v19, v11, v15 // (op 1273) 

v_lshrrev_b32 v12, 16, v16 // (op 1275) 
v_lshrrev_b32 v13, 16, v17 // (op 1275) 
v_lshrrev_b32 v14, 16, v18 // (op 1275) 
v_lshrrev_b32 v15, 16, v19 // (op 1275) 


v_pack_b32_f16 v8, v12, v13 // (op 1276) 
v_pack_b32_f16 v9, v14, v15 // (op 1276) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen offset:128 // (op 1276) Store value


v_accvgpr_read v12, a12 // (op 1293) 
v_accvgpr_read v13, a13 // (op 1293) 
v_accvgpr_read v14, a14 // (op 1293) 
v_accvgpr_read v15, a15 // (op 1293) 
v_mul_f32 v16, v11, v12 // (op 1293) 
v_mul_f32 v17, v11, v13 // (op 1293) 
v_mul_f32 v18, v11, v14 // (op 1293) 
v_mul_f32 v19, v11, v15 // (op 1293) 

v_lshrrev_b32 v12, 16, v16 // (op 1295) 
v_lshrrev_b32 v13, 16, v17 // (op 1295) 
v_lshrrev_b32 v14, 16, v18 // (op 1295) 
v_lshrrev_b32 v15, 16, v19 // (op 1295) 


v_pack_b32_f16 v8, v12, v13 // (op 1296) 
v_pack_b32_f16 v9, v14, v15 // (op 1296) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen offset:192 // (op 1296) Store value


v_accvgpr_read v12, a16 // (op 1313) 
v_accvgpr_read v13, a17 // (op 1313) 
v_accvgpr_read v14, a18 // (op 1313) 
v_accvgpr_read v15, a19 // (op 1313) 
v_mul_f32 v16, v11, v12 // (op 1313) 
v_mul_f32 v17, v11, v13 // (op 1313) 
v_mul_f32 v18, v11, v14 // (op 1313) 
v_mul_f32 v19, v11, v15 // (op 1313) 

v_lshrrev_b32 v12, 16, v16 // (op 1315) 
v_lshrrev_b32 v13, 16, v17 // (op 1315) 
v_lshrrev_b32 v14, 16, v18 // (op 1315) 
v_lshrrev_b32 v15, 16, v19 // (op 1315) 


v_pack_b32_f16 v8, v12, v13 // (op 1316) 
v_pack_b32_f16 v9, v14, v15 // (op 1316) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen offset:256 // (op 1316) Store value


v_accvgpr_read v12, a20 // (op 1333) 
v_accvgpr_read v13, a21 // (op 1333) 
v_accvgpr_read v14, a22 // (op 1333) 
v_accvgpr_read v15, a23 // (op 1333) 
v_mul_f32 v16, v11, v12 // (op 1333) 
v_mul_f32 v17, v11, v13 // (op 1333) 
v_mul_f32 v18, v11, v14 // (op 1333) 
v_mul_f32 v19, v11, v15 // (op 1333) 

v_lshrrev_b32 v12, 16, v16 // (op 1335) 
v_lshrrev_b32 v13, 16, v17 // (op 1335) 
v_lshrrev_b32 v14, 16, v18 // (op 1335) 
v_lshrrev_b32 v15, 16, v19 // (op 1335) 


v_pack_b32_f16 v8, v12, v13 // (op 1336) 
v_pack_b32_f16 v9, v14, v15 // (op 1336) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen offset:320 // (op 1336) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[8:9], v7, s[56:59], 0 offen offset:320 // (op 1466) Load value
v_accvgpr_read v12, a20 // (op 1462) 
v_accvgpr_read v13, a21 // (op 1462) 
v_accvgpr_read v14, a22 // (op 1462) 
v_accvgpr_read v15, a23 // (op 1462) 
v_mul_f32 v16, v11, v12 // (op 1462) 
v_mul_f32 v17, v11, v13 // (op 1462) 
v_mul_f32 v18, v11, v14 // (op 1462) 
v_mul_f32 v19, v11, v15 // (op 1462) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 1467) 
v_lshrrev_b32 v13, 16, v8 // (op 1467) 
v_lshlrev_b32 v12, 16, v12 // (op 1467) 
v_lshlrev_b32 v13, 16, v13 // (op 1467) 
v_mul_f32 v20, v0, v12 // (op 1467) 
v_mul_f32 v21, v0, v13 // (op 1467) 
v_and_b32 v12, 65535, v9 // (op 1467) 
v_lshrrev_b32 v13, 16, v9 // (op 1467) 
v_lshlrev_b32 v12, 16, v12 // (op 1467) 
v_lshlrev_b32 v13, 16, v13 // (op 1467) 
v_mul_f32 v22, v0, v12 // (op 1467) 
v_mul_f32 v23, v0, v13 // (op 1467) 


v_add_f32 v12, v20, v16 // (op 1463) 
v_add_f32 v13, v21, v17 // (op 1463) 
v_add_f32 v14, v22, v18 // (op 1463) 
v_add_f32 v15, v23, v19 // (op 1463) 


v_lshrrev_b32 v16, 16, v12 // (op 1464) 
v_lshrrev_b32 v17, 16, v13 // (op 1464) 
v_lshrrev_b32 v18, 16, v14 // (op 1464) 
v_lshrrev_b32 v19, 16, v15 // (op 1464) 


v_pack_b32_f16 v8, v16, v17 // (op 1465) 
v_pack_b32_f16 v9, v18, v19 // (op 1465) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen offset:320 // (op 1465) Store value





buffer_load_dwordx2 v[8:9], v7, s[56:59], 0 offen offset:256 // (op 1446) Load value
v_accvgpr_read v12, a16 // (op 1442) 
v_accvgpr_read v13, a17 // (op 1442) 
v_accvgpr_read v14, a18 // (op 1442) 
v_accvgpr_read v15, a19 // (op 1442) 
v_mul_f32 v16, v11, v12 // (op 1442) 
v_mul_f32 v17, v11, v13 // (op 1442) 
v_mul_f32 v18, v11, v14 // (op 1442) 
v_mul_f32 v19, v11, v15 // (op 1442) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 1447) 
v_lshrrev_b32 v13, 16, v8 // (op 1447) 
v_lshlrev_b32 v12, 16, v12 // (op 1447) 
v_lshlrev_b32 v13, 16, v13 // (op 1447) 
v_mul_f32 v20, v0, v12 // (op 1447) 
v_mul_f32 v21, v0, v13 // (op 1447) 
v_and_b32 v12, 65535, v9 // (op 1447) 
v_lshrrev_b32 v13, 16, v9 // (op 1447) 
v_lshlrev_b32 v12, 16, v12 // (op 1447) 
v_lshlrev_b32 v13, 16, v13 // (op 1447) 
v_mul_f32 v22, v0, v12 // (op 1447) 
v_mul_f32 v23, v0, v13 // (op 1447) 


v_add_f32 v12, v20, v16 // (op 1443) 
v_add_f32 v13, v21, v17 // (op 1443) 
v_add_f32 v14, v22, v18 // (op 1443) 
v_add_f32 v15, v23, v19 // (op 1443) 


v_lshrrev_b32 v16, 16, v12 // (op 1444) 
v_lshrrev_b32 v17, 16, v13 // (op 1444) 
v_lshrrev_b32 v18, 16, v14 // (op 1444) 
v_lshrrev_b32 v19, 16, v15 // (op 1444) 


v_pack_b32_f16 v8, v16, v17 // (op 1445) 
v_pack_b32_f16 v9, v18, v19 // (op 1445) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen offset:256 // (op 1445) Store value





buffer_load_dwordx2 v[8:9], v7, s[56:59], 0 offen offset:192 // (op 1426) Load value
v_accvgpr_read v12, a12 // (op 1422) 
v_accvgpr_read v13, a13 // (op 1422) 
v_accvgpr_read v14, a14 // (op 1422) 
v_accvgpr_read v15, a15 // (op 1422) 
v_mul_f32 v16, v11, v12 // (op 1422) 
v_mul_f32 v17, v11, v13 // (op 1422) 
v_mul_f32 v18, v11, v14 // (op 1422) 
v_mul_f32 v19, v11, v15 // (op 1422) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 1427) 
v_lshrrev_b32 v13, 16, v8 // (op 1427) 
v_lshlrev_b32 v12, 16, v12 // (op 1427) 
v_lshlrev_b32 v13, 16, v13 // (op 1427) 
v_mul_f32 v20, v0, v12 // (op 1427) 
v_mul_f32 v21, v0, v13 // (op 1427) 
v_and_b32 v12, 65535, v9 // (op 1427) 
v_lshrrev_b32 v13, 16, v9 // (op 1427) 
v_lshlrev_b32 v12, 16, v12 // (op 1427) 
v_lshlrev_b32 v13, 16, v13 // (op 1427) 
v_mul_f32 v22, v0, v12 // (op 1427) 
v_mul_f32 v23, v0, v13 // (op 1427) 


v_add_f32 v12, v20, v16 // (op 1423) 
v_add_f32 v13, v21, v17 // (op 1423) 
v_add_f32 v14, v22, v18 // (op 1423) 
v_add_f32 v15, v23, v19 // (op 1423) 


v_lshrrev_b32 v16, 16, v12 // (op 1424) 
v_lshrrev_b32 v17, 16, v13 // (op 1424) 
v_lshrrev_b32 v18, 16, v14 // (op 1424) 
v_lshrrev_b32 v19, 16, v15 // (op 1424) 


v_pack_b32_f16 v8, v16, v17 // (op 1425) 
v_pack_b32_f16 v9, v18, v19 // (op 1425) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen offset:192 // (op 1425) Store value





buffer_load_dwordx2 v[8:9], v7, s[56:59], 0 offen offset:128 // (op 1406) Load value
v_accvgpr_read v12, a8 // (op 1402) 
v_accvgpr_read v13, a9 // (op 1402) 
v_accvgpr_read v14, a10 // (op 1402) 
v_accvgpr_read v15, a11 // (op 1402) 
v_mul_f32 v16, v11, v12 // (op 1402) 
v_mul_f32 v17, v11, v13 // (op 1402) 
v_mul_f32 v18, v11, v14 // (op 1402) 
v_mul_f32 v19, v11, v15 // (op 1402) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 1407) 
v_lshrrev_b32 v13, 16, v8 // (op 1407) 
v_lshlrev_b32 v12, 16, v12 // (op 1407) 
v_lshlrev_b32 v13, 16, v13 // (op 1407) 
v_mul_f32 v20, v0, v12 // (op 1407) 
v_mul_f32 v21, v0, v13 // (op 1407) 
v_and_b32 v12, 65535, v9 // (op 1407) 
v_lshrrev_b32 v13, 16, v9 // (op 1407) 
v_lshlrev_b32 v12, 16, v12 // (op 1407) 
v_lshlrev_b32 v13, 16, v13 // (op 1407) 
v_mul_f32 v22, v0, v12 // (op 1407) 
v_mul_f32 v23, v0, v13 // (op 1407) 


v_add_f32 v12, v20, v16 // (op 1403) 
v_add_f32 v13, v21, v17 // (op 1403) 
v_add_f32 v14, v22, v18 // (op 1403) 
v_add_f32 v15, v23, v19 // (op 1403) 


v_lshrrev_b32 v16, 16, v12 // (op 1404) 
v_lshrrev_b32 v17, 16, v13 // (op 1404) 
v_lshrrev_b32 v18, 16, v14 // (op 1404) 
v_lshrrev_b32 v19, 16, v15 // (op 1404) 


v_pack_b32_f16 v8, v16, v17 // (op 1405) 
v_pack_b32_f16 v9, v18, v19 // (op 1405) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen offset:128 // (op 1405) Store value





buffer_load_dwordx2 v[8:9], v7, s[56:59], 0 offen offset:64 // (op 1386) Load value
v_accvgpr_read v12, a4 // (op 1382) 
v_accvgpr_read v13, a5 // (op 1382) 
v_accvgpr_read v14, a6 // (op 1382) 
v_accvgpr_read v15, a7 // (op 1382) 
v_mul_f32 v16, v11, v12 // (op 1382) 
v_mul_f32 v17, v11, v13 // (op 1382) 
v_mul_f32 v18, v11, v14 // (op 1382) 
v_mul_f32 v19, v11, v15 // (op 1382) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 1387) 
v_lshrrev_b32 v13, 16, v8 // (op 1387) 
v_lshlrev_b32 v12, 16, v12 // (op 1387) 
v_lshlrev_b32 v13, 16, v13 // (op 1387) 
v_mul_f32 v20, v0, v12 // (op 1387) 
v_mul_f32 v21, v0, v13 // (op 1387) 
v_and_b32 v12, 65535, v9 // (op 1387) 
v_lshrrev_b32 v13, 16, v9 // (op 1387) 
v_lshlrev_b32 v12, 16, v12 // (op 1387) 
v_lshlrev_b32 v13, 16, v13 // (op 1387) 
v_mul_f32 v22, v0, v12 // (op 1387) 
v_mul_f32 v23, v0, v13 // (op 1387) 


v_add_f32 v12, v20, v16 // (op 1383) 
v_add_f32 v13, v21, v17 // (op 1383) 
v_add_f32 v14, v22, v18 // (op 1383) 
v_add_f32 v15, v23, v19 // (op 1383) 


v_lshrrev_b32 v16, 16, v12 // (op 1384) 
v_lshrrev_b32 v17, 16, v13 // (op 1384) 
v_lshrrev_b32 v18, 16, v14 // (op 1384) 
v_lshrrev_b32 v19, 16, v15 // (op 1384) 


v_pack_b32_f16 v8, v16, v17 // (op 1385) 
v_pack_b32_f16 v9, v18, v19 // (op 1385) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen offset:64 // (op 1385) Store value





buffer_load_dwordx2 v[8:9], v7, s[56:59], 0 offen // (op 138) Load value
v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v11, v12 // (op 134) 
v_mul_f32 v17, v11, v13 // (op 134) 
v_mul_f32 v18, v11, v14 // (op 134) 
v_mul_f32 v19, v11, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 139) 
v_lshrrev_b32 v13, 16, v8 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v20, v0, v12 // (op 139) 
v_mul_f32 v21, v0, v13 // (op 139) 
v_and_b32 v12, 65535, v9 // (op 139) 
v_lshrrev_b32 v13, 16, v9 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v22, v0, v12 // (op 139) 
v_mul_f32 v23, v0, v13 // (op 139) 


v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 
v_add_f32 v14, v22, v18 // (op 135) 
v_add_f32 v15, v23, v19 // (op 135) 


v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 
v_lshrrev_b32 v18, 16, v14 // (op 136) 
v_lshrrev_b32 v19, 16, v15 // (op 136) 


v_pack_b32_f16 v8, v16, v17 // (op 137) 
v_pack_b32_f16 v9, v18, v19 // (op 137) 


buffer_store_dwordx2 v[8:9], v10, s[60:63], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:














s_endpgm  // End of RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM_
.LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM_, .LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM__end-RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 30208 // lds bytes
  .amdhsa_accum_offset 76
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: F4
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
            { type: int64, semantic: RR_NumTilesM },
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [192, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [6, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 30208
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 76
    .agpr_count: 24
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: BitwiseXor_18
        .size: 8
        .offset: 8
        .expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 24
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 156, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 28
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_25, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 168, size: 4}, rhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 32
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 156, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_24
        .size: 4
        .offset: 36
        .expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 40
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 44
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_22
        .size: 4
        .offset: 48
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 52
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_25, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 168, size: 4}, rhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}}}
        .variableType: {dataType: Int32, pointerType: Value}
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 128
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 136
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 144
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 152
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_25
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_23
        .size: 4
        .offset: 164
        .expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: WGM_19
        .size: 4
        .offset: 168
        .expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 172
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 176
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

