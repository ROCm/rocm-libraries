

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM_,@function
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 10483) 
v_and_b32 v3, 15, v1 // (op 10483) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 10483) 


v_and_b32 v0, 63, v1 // (op 10483) 
v_lshrrev_b32 v3, 4, v0 // (op 10483) 

v_lshlrev_b32 v0, 5, v3 // (op 10483) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 10483) 


v_lshlrev_b32 v0, 2, v3 // (op 10483) 

v_lshrrev_b32 v3, 3, v0 // (op 10483) 

v_lshrrev_b32 v0, 6, v1 // (op 10541) 
v_and_b32 v4, 1, v0 // (op 10541) 

v_and_b32 v0, 15, v1 // (op 10541) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 10541) 


v_and_b32 v0, 63, v1 // (op 10541) 
v_lshrrev_b32 v4, 4, v0 // (op 10541) 

v_lshlrev_b32 v0, 5, v4 // (op 10541) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 10541) 


v_lshlrev_b32 v0, 2, v4 // (op 10541) 

v_lshrrev_b32 v4, 3, v0 // (op 10541) 

v_lshrrev_b32 v0, 7, v1 // (op 10599) 
v_and_b32 v5, 15, v1 // (op 10599) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 10599) 


v_and_b32 v0, 63, v1 // (op 10599) 
v_lshrrev_b32 v5, 4, v0 // (op 10599) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 10599) 


v_lshrrev_b32 v5, 6, v1 // (op 10657) 
v_and_b32 v6, 1, v5 // (op 10657) 

v_and_b32 v5, 15, v1 // (op 10657) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 10657) 


v_and_b32 v5, 63, v1 // (op 10657) 
v_lshrrev_b32 v6, 4, v5 // (op 10657) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 10657) 


v_lshrrev_b32 v6, 7, v1 // (op 10709) 
v_and_b32 v7, 15, v1 // (op 10709) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 10709) 


v_and_b32 v6, 63, v1 // (op 10709) 
v_lshrrev_b32 v7, 4, v6 // (op 10709) 

v_lshlrev_b32 v6, 5, v7 // (op 10709) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 10709) 


v_lshlrev_b32 v6, 2, v7 // (op 10709) 

v_lshrrev_b32 v7, 3, v6 // (op 10709) 

v_lshrrev_b32 v6, 6, v1 // (op 10738) 
v_and_b32 v8, 1, v6 // (op 10738) 

v_and_b32 v6, 15, v1 // (op 10738) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 10738) 


v_and_b32 v6, 63, v1 // (op 10738) 
v_lshrrev_b32 v8, 4, v6 // (op 10738) 

v_lshlrev_b32 v6, 5, v8 // (op 10738) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 10738) 


v_lshlrev_b32 v6, 2, v8 // (op 10738) 

v_lshrrev_b32 v8, 3, v6 // (op 10738) 

v_lshrrev_b32 v6, 7, v1 // (op 10767) 
v_and_b32 v9, 15, v1 // (op 10767) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 10767) 


v_and_b32 v6, 63, v1 // (op 10767) 
v_lshrrev_b32 v9, 4, v6 // (op 10767) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 10767) 


v_lshrrev_b32 v9, 6, v1 // (op 10796) 
v_and_b32 v10, 1, v9 // (op 10796) 

v_and_b32 v9, 15, v1 // (op 10796) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 10796) 


v_and_b32 v9, 63, v1 // (op 10796) 
v_lshrrev_b32 v10, 4, v9 // (op 10796) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 10796) 


v_lshrrev_b32 v10, 7, v1 // (op 10825) 
v_and_b32 v11, 15, v1 // (op 10825) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 10825) 


v_and_b32 v10, 63, v1 // (op 10825) 
v_lshrrev_b32 v11, 4, v10 // (op 10825) 

v_lshlrev_b32 v10, 5, v11 // (op 10825) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 10825) 


v_lshlrev_b32 v10, 2, v11 // (op 10825) 

v_lshrrev_b32 v11, 3, v10 // (op 10825) 

v_lshrrev_b32 v10, 6, v1 // (op 10854) 
v_and_b32 v12, 1, v10 // (op 10854) 

v_and_b32 v10, 15, v1 // (op 10854) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 10854) 


v_and_b32 v10, 63, v1 // (op 10854) 
v_lshrrev_b32 v12, 4, v10 // (op 10854) 

v_lshlrev_b32 v10, 5, v12 // (op 10854) 

v_lshl_add_u32 v12, v13, 7, v10 // (op 10854) 


v_lshlrev_b32 v10, 2, v12 // (op 10854) 

v_lshrrev_b32 v12, 3, v10 // (op 10854) 

v_lshrrev_b32 v10, 7, v1 // (op 10883) 
v_and_b32 v13, 15, v1 // (op 10883) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 10883) 


v_and_b32 v10, 63, v1 // (op 10883) 
v_lshrrev_b32 v13, 4, v10 // (op 10883) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 10883) 


v_lshrrev_b32 v13, 6, v1 // (op 10912) 
v_and_b32 v14, 1, v13 // (op 10912) 

v_and_b32 v13, 15, v1 // (op 10912) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 10912) 


v_and_b32 v13, 63, v1 // (op 10912) 
v_lshrrev_b32 v14, 4, v13 // (op 10912) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 10912) 


v_lshrrev_b32 v14, 7, v1 // (op 10941) 
v_and_b32 v15, 15, v1 // (op 10941) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 10941) 


v_and_b32 v14, 63, v1 // (op 10941) 
v_lshrrev_b32 v15, 4, v14 // (op 10941) 

v_lshlrev_b32 v14, 5, v15 // (op 10941) 

v_lshl_add_u32 v15, v16, 7, v14 // (op 10941) 


v_lshlrev_b32 v14, 2, v15 // (op 10941) 

v_lshrrev_b32 v15, 3, v14 // (op 10941) 

v_lshrrev_b32 v14, 6, v1 // (op 10970) 
v_and_b32 v16, 1, v14 // (op 10970) 

v_and_b32 v14, 15, v1 // (op 10970) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 10970) 


v_and_b32 v14, 63, v1 // (op 10970) 
v_lshrrev_b32 v16, 4, v14 // (op 10970) 

v_lshlrev_b32 v14, 5, v16 // (op 10970) 

v_lshl_add_u32 v16, v17, 7, v14 // (op 10970) 


v_lshlrev_b32 v14, 2, v16 // (op 10970) 

v_lshrrev_b32 v16, 3, v14 // (op 10970) 

v_lshrrev_b32 v14, 7, v1 // (op 10999) 
v_and_b32 v17, 15, v1 // (op 10999) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 10999) 


v_and_b32 v14, 63, v1 // (op 10999) 
v_lshrrev_b32 v17, 4, v14 // (op 10999) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 10999) 


v_lshrrev_b32 v17, 6, v1 // (op 11028) 
v_and_b32 v18, 1, v17 // (op 11028) 

v_and_b32 v17, 15, v1 // (op 11028) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 11028) 


v_and_b32 v17, 63, v1 // (op 11028) 
v_lshrrev_b32 v18, 4, v17 // (op 11028) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 11028) 


v_lshrrev_b32 v18, 7, v1 // (op 10421) 
v_lshrrev_b32 v19, 7, v1 // (op 10678) 

s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s0, s16, s47 // (op 10421) 
s_add_i32 s1, s0, s16 // (op 10421) 

s_mul_hi_i32 s0, s16, s47 // (op 10678) 
s_add_i32 s18, s0, s16 // (op 10678) 

v_and_b32 v20, 63, v1 // (op 10421) 
s_ashr_i32 s0, s1, 31 // (op 10421) 
v_lshrrev_b32 v21, 4, v20 // (op 10421) 
s_cmp_eq_i32 s47, 0 // (op 10421) 
s_mov_b32 s19, scc // (op 10421) 
v_lshlrev_b32 v22, 2, v21 // (op 10421) 

s_cmp_eq_i32 s19, 0 // (op 10421) 
s_cselect_b32 s51, 0, -1 // (op 10421) 

v_lshl_add_u32 v21, v18, 4, v22 // (op 10421) 


s_and_b32 s19, s13, 31 // (op 10421) 
v_lshrrev_b32 v18, 2, v20 // (op 10421) 

v_and_b32 v20, 3, v18 // (op 10421) 

v_and_b32 v18, 63, v1 // (op 10678) 
v_lshrrev_b32 v22, 4, v18 // (op 10678) 
s_lshl_b32 s52, 1, s19 // (op 10421) 
v_and_b32 v23, 3, v1 // (op 10421) 
s_add_i32 s53, s52, s51 // (op 10421) 


s_and_b32 s51, s0, s53 // (op 10421) 


s_ashr_i32 s0, s18, 31 // (op 10678) 
s_cmp_eq_i32 s47, 0 // (op 10678) 
s_mov_b32 s52, scc // (op 10678) 
v_lshl_add_u32 v24, v20, 2, v23 // (op 10421) 


s_add_i32 s53, s1, s51 // (op 10421) 


v_lshrrev_b32 v20, 6, v1 // (op 10421) 
s_ashr_i32 s1, s53, s19 // (op 10421) 


v_and_b32 v23, 1, v20 // (op 10421) 

s_lshl_b32 s19, s13, 24 // (op 10421) 
v_lshl_add_u32 v20, v23, 4, v24 // (op 10421) 


s_ashr_i32 s51, s19, 31 // (op 10421) 

s_xor_b32 s19, s1, s51 // (op 10421) 

s_sub_i32 s1, s19, s51 // (op 10421) 


s_mul_hi_i32 s19, s1, s14 // (op 10421) 
s_add_i32 s51, s19, s1 // (op 10421) 

s_ashr_i32 s19, s51, 31 // (op 10421) 
s_cmp_eq_i32 s14, 0 // (op 10421) 
s_mov_b32 s53, scc // (op 10421) 
s_cmp_eq_i32 s53, 0 // (op 10421) 
s_cselect_b32 s54, 0, -1 // (op 10421) 

s_and_b32 s53, s15, 31 // (op 10421) 
v_lshlrev_b32 v23, 2, v22 // (op 10678) 

s_cmp_eq_i32 s52, 0 // (op 10678) 
s_cselect_b32 s55, 0, -1 // (op 10678) 

s_lshl_b32 s52, 1, s53 // (op 10421) 
v_lshl_add_u32 v22, v19, 4, v23 // (op 10678) 


v_lshrrev_b32 v19, 2, v18 // (op 10678) 

s_add_i32 s56, s52, s54 // (op 10421) 


s_and_b32 s52, s19, s56 // (op 10421) 


s_add_i32 s19, s51, s52 // (op 10421) 


s_ashr_i32 s54, s19, s53 // (op 10421) 


s_lshl_b32 s19, s15, 24 // (op 10421) 
s_ashr_i32 s51, s19, 31 // (op 10421) 

s_xor_b32 s19, s54, s51 // (op 10421) 

s_sub_i32 s52, s19, s51 // (op 10421) 


s_mul_i32 s19, s1, s48 // (op 10421) 
s_sub_u32 s51, s16, s19 // (op 10421) 

s_mul_i32 s19, s52, s48 // (op 10421) 
s_add_u32 s19, s19, s51 // (op 10421) 

s_mul_hi_i32 s51, s16, s10 // (op 10421) 
s_add_i32 s53, s51, s16 // (op 10421) 

s_ashr_i32 s51, s53, 31 // (op 10421) 
s_cmp_eq_i32 s10, 0 // (op 10421) 
s_mov_b32 s54, scc // (op 10421) 
s_cmp_eq_i32 s54, 0 // (op 10421) 
s_cselect_b32 s56, 0, -1 // (op 10421) 

s_and_b32 s54, s12, 31 // (op 10421) 
s_lshl_b32 s57, 1, s54 // (op 10421) 
s_add_i32 s58, s57, s56 // (op 10421) 


s_and_b32 s56, s51, s58 // (op 10421) 


s_add_i32 s51, s53, s56 // (op 10421) 


s_ashr_i32 s53, s51, s54 // (op 10421) 


s_lshl_b32 s51, s12, 24 // (op 10421) 
s_ashr_i32 s54, s51, 31 // (op 10421) 

s_xor_b32 s51, s53, s54 // (op 10421) 

s_sub_i32 s53, s51, s54 // (op 10421) 


s_and_b32 s51, s13, 31 // (op 10678) 
s_lshl_b32 s54, 1, s51 // (op 10678) 
v_and_b32 v18, 3, v19 // (op 10678) 

s_add_i32 s56, s54, s55 // (op 10678) 


v_and_b32 v19, 3, v1 // (op 10678) 
s_and_b32 s54, s0, s56 // (op 10678) 


v_lshl_add_u32 v23, v18, 2, v19 // (op 10678) 


s_add_i32 s0, s18, s54 // (op 10678) 


v_lshrrev_b32 v18, 6, v1 // (op 10678) 
s_ashr_i32 s18, s0, s51 // (op 10678) 


v_and_b32 v19, 1, v18 // (op 10678) 

s_lshl_b32 s0, s13, 24 // (op 10678) 
v_lshl_add_u32 v18, v19, 4, v23 // (op 10678) 


s_ashr_i32 s51, s0, 31 // (op 10678) 

s_xor_b32 s0, s18, s51 // (op 10678) 

s_sub_i32 s18, s0, s51 // (op 10678) 


s_mul_hi_i32 s0, s18, s14 // (op 10678) 
s_add_i32 s51, s0, s18 // (op 10678) 

s_ashr_i32 s0, s51, 31 // (op 10678) 
s_cmp_eq_i32 s14, 0 // (op 10678) 
s_mov_b32 s54, scc // (op 10678) 
s_cmp_eq_i32 s54, 0 // (op 10678) 
s_cselect_b32 s55, 0, -1 // (op 10678) 

s_and_b32 s54, s15, 31 // (op 10678) 
s_lshl_b32 s56, 1, s54 // (op 10678) 
s_add_i32 s57, s56, s55 // (op 10678) 


s_and_b32 s55, s0, s57 // (op 10678) 


s_add_i32 s0, s51, s55 // (op 10678) 


s_ashr_i32 s51, s0, s54 // (op 10678) 


s_lshl_b32 s0, s15, 24 // (op 10678) 
s_ashr_i32 s54, s0, 31 // (op 10678) 

s_xor_b32 s0, s51, s54 // (op 10678) 

s_sub_i32 s51, s0, s54 // (op 10678) 


s_mul_i32 s0, s18, s48 // (op 10678) 
s_sub_u32 s54, s16, s0 // (op 10678) 

s_mul_i32 s0, s51, s48 // (op 10678) 
s_add_u32 s0, s0, s54 // (op 10678) 

s_mul_hi_i32 s54, s16, s10 // (op 10678) 
s_add_i32 s55, s54, s16 // (op 10678) 

s_ashr_i32 s54, s55, 31 // (op 10678) 
s_cmp_eq_i32 s10, 0 // (op 10678) 
s_mov_b32 s56, scc // (op 10678) 
s_cmp_eq_i32 s56, 0 // (op 10678) 
s_cselect_b32 s57, 0, -1 // (op 10678) 

s_and_b32 s56, s12, 31 // (op 10678) 
s_lshl_b32 s58, 1, s56 // (op 10678) 
s_add_i32 s59, s58, s57 // (op 10678) 


s_and_b32 s57, s54, s59 // (op 10678) 


s_add_i32 s54, s55, s57 // (op 10678) 


s_ashr_i32 s55, s54, s56 // (op 10678) 


s_lshl_b32 s54, s12, 24 // (op 10678) 
s_ashr_i32 s56, s54, 31 // (op 10678) 

s_xor_b32 s54, s55, s56 // (op 10678) 

s_sub_i32 s55, s54, s56 // (op 10678) 


s_add_u32 s56, s8, 64 // (op 10421) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 10421) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 10421) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 10421) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 10421) 

s_mul_i32 s54, s48, s56 // (op 10421) 
s_mul_i32 s58, s44, s54 // (op 10421) 

s_mul_i32 s54, s53, s58 // (op 10421) 

s_sub_u32 s58, s16, s54 // (op 10421) 

s_mov_b32 s54, s58 // (op 10421) 
s_mul_hi_i32 s59, s54, s45 // (op 10421) 
s_add_i32 s60, s59, s54 // (op 10421) 


s_ashr_i32 s54, s60, 31 // (op 10421) 
s_cmp_eq_i32 s45, 0 // (op 10421) 
s_mov_b32 s59, scc // (op 10421) 
s_cmp_eq_i32 s59, 0 // (op 10421) 
s_cselect_b32 s61, 0, -1 // (op 10421) 

s_and_b32 s59, s11, 31 // (op 10421) 
s_lshl_b32 s62, 1, s59 // (op 10421) 
s_add_i32 s63, s62, s61 // (op 10421) 


s_and_b32 s61, s54, s63 // (op 10421) 


s_add_i32 s54, s60, s61 // (op 10421) 


s_ashr_i32 s60, s54, s59 // (op 10421) 


s_lshl_b32 s54, s11, 24 // (op 10421) 
s_ashr_i32 s59, s54, 31 // (op 10421) 

s_xor_b32 s54, s60, s59 // (op 10421) 

s_sub_i32 s60, s54, s59 // (op 10421) 


s_mul_i32 s54, s46, s48 // (op 10421) 
s_sub_i32 s59, s6, s54 // (op 10421) 
s_mul_i32 s61, s60, s59 // (op 10421) 

s_sub_u32 s59, s58, s61 // (op 10421) 


s_add_u32 s58, s59, s54 // (op 10421) 


s_cmp_eq_i32 s53, 0 // (op 10421) 
s_mov_b32 s54, scc // (op 10421) 

s_cmp_eq_i32 s54, 0 // (op 10421) 
s_cselect_b32 s53, s58, s19 // (op 10421) 


s_mul_i32 s19, s52, s56 // (op 10421) 


v_mov_b32 v19, 192 // (op 10421) 
s_sub_i32 s52, s1, s19 // (op 10421) 


v_mul_lo_u32 v23, s53, v19 // (op 10421) 

s_cmp_eq_i32 s54, 0 // (op 10421) 
s_cselect_b32 s1, s60, s52 // (op 10421) 



v_add_u32 v23, v23, v21 // (op 10421) 


v_lshl_add_u32 v19, s1, 6, v20 // (op 10421) 


v_mul_lo_u32 v20, v19, s2 // (op 10421) 

v_add_lshl_u32 v19, v23, v20, 1 // (op 10421) 


s_add_u32 s52, s8, 64 // (op 10678) Start of Int64 add, locking SCC
s_addc_u32 s53, s9, 0 // (op 10678) End of Int64 add, unlocking SCC
s_sub_u32 s56, s52, 1 // (op 10678) least significant half; sets scc
s_subb_u32 s57, s53, 0 // (op 10678) most significant half; uses scc

s_ashr_i64 s[52:53], s[56:57], 6 // (op 10678) 

s_mul_i32 s1, s48, s52 // (op 10678) 
s_mul_i32 s19, s44, s1 // (op 10678) 

s_mul_i32 s1, s55, s19 // (op 10678) 

s_sub_u32 s19, s16, s1 // (op 10678) 

s_mov_b32 s1, s19 // (op 10678) 
s_mul_hi_i32 s54, s1, s45 // (op 10678) 
s_add_i32 s56, s54, s1 // (op 10678) 


s_ashr_i32 s1, s56, 31 // (op 10678) 
s_cmp_eq_i32 s45, 0 // (op 10678) 
s_mov_b32 s54, scc // (op 10678) 
s_cmp_eq_i32 s54, 0 // (op 10678) 
s_cselect_b32 s57, 0, -1 // (op 10678) 

s_and_b32 s54, s11, 31 // (op 10678) 
s_lshl_b32 s58, 1, s54 // (op 10678) 
s_add_i32 s59, s58, s57 // (op 10678) 


s_and_b32 s57, s1, s59 // (op 10678) 


s_add_i32 s1, s56, s57 // (op 10678) 


s_ashr_i32 s56, s1, s54 // (op 10678) 


s_lshl_b32 s1, s11, 24 // (op 10678) 
s_ashr_i32 s54, s1, 31 // (op 10678) 

s_xor_b32 s1, s56, s54 // (op 10678) 

s_sub_i32 s56, s1, s54 // (op 10678) 


s_mul_i32 s1, s46, s48 // (op 10678) 
s_sub_i32 s54, s6, s1 // (op 10678) 
s_mul_i32 s57, s56, s54 // (op 10678) 

s_sub_u32 s54, s19, s57 // (op 10678) 


s_add_u32 s19, s54, s1 // (op 10678) 


s_cmp_eq_i32 s55, 0 // (op 10678) 
s_mov_b32 s1, scc // (op 10678) 

s_cmp_eq_i32 s1, 0 // (op 10678) 
s_cselect_b32 s54, s19, s0 // (op 10678) 


s_mul_i32 s0, s51, s52 // (op 10678) 


v_mov_b32 v20, 192 // (op 10678) 
s_sub_i32 s19, s18, s0 // (op 10678) 


v_mul_lo_u32 v21, s54, v20 // (op 10678) 

s_cmp_eq_i32 s1, 0 // (op 10678) 
s_cselect_b32 s0, s56, s19 // (op 10678) 



v_add_u32 v21, v21, v22 // (op 10678) 


v_lshl_add_u32 v20, s0, 6, v18 // (op 10678) 


v_mul_lo_u32 v18, v20, s4 // (op 10678) 

v_add_lshl_u32 v20, v21, v18, 1 // (op 10678) 


s_mul_i32 s1, s2, s9 // (op 10423) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 10423) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 10423) least significant
s_mul_hi_u32 s19, s2, s8 // (op 10423) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10423) most significant: sum
s_add_u32 s1, s1, s19 // (op 10423) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 10423) 

s_mul_i32 s1, s4, s9 // (op 10680) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 10680) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 10680) least significant
s_mul_hi_u32 s52, s4, s8 // (op 10680) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 10680) most significant: sum
s_add_u32 s1, s1, s52 // (op 10680) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 10680) 

s_mov_b32 s56, s20 // (op 10423) reg expression
s_mov_b32 s57, s21 // (op 10423) reg expression
s_mov_b32 s58, s18 // (op 10423) reg expression
s_mov_b32 s59, 131072 // (op 10423) call()

s_mov_b32 s60, s22 // (op 10680) reg expression

s_mov_b32 s61, s23 // (op 10680) reg expression
s_mov_b32 s62, s52 // (op 10680) reg expression
s_mov_b32 s63, 131072 // (op 10680) call()


v_mov_b32 v18, s49 // (op 2) Move value
v_mov_b32 v21, s50 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 486) call()
v_accvgpr_write a5, 0.00000 // (op 486) call()
v_accvgpr_write a6, 0.00000 // (op 486) call()
v_accvgpr_write a7, 0.00000 // (op 486) call()
v_accvgpr_write a8, 0.00000 // (op 852) call()
v_accvgpr_write a9, 0.00000 // (op 852) call()
v_accvgpr_write a10, 0.00000 // (op 852) call()
v_accvgpr_write a11, 0.00000 // (op 852) call()
v_accvgpr_write a12, 0.00000 // (op 1099) call()
v_accvgpr_write a13, 0.00000 // (op 1099) call()
v_accvgpr_write a14, 0.00000 // (op 1099) call()
v_accvgpr_write a15, 0.00000 // (op 1099) call()
v_accvgpr_write a16, 0.00000 // (op 1498) call()
v_accvgpr_write a17, 0.00000 // (op 1498) call()
v_accvgpr_write a18, 0.00000 // (op 1498) call()
v_accvgpr_write a19, 0.00000 // (op 1498) call()
v_accvgpr_write a20, 0.00000 // (op 1745) call()
v_accvgpr_write a21, 0.00000 // (op 1745) call()
v_accvgpr_write a22, 0.00000 // (op 1745) call()
v_accvgpr_write a23, 0.00000 // (op 1745) call()
v_accvgpr_write a24, 0.00000 // (op 2144) call()
v_accvgpr_write a25, 0.00000 // (op 2144) call()
v_accvgpr_write a26, 0.00000 // (op 2144) call()
v_accvgpr_write a27, 0.00000 // (op 2144) call()
v_accvgpr_write a28, 0.00000 // (op 2391) call()
v_accvgpr_write a29, 0.00000 // (op 2391) call()
v_accvgpr_write a30, 0.00000 // (op 2391) call()
v_accvgpr_write a31, 0.00000 // (op 2391) call()
v_accvgpr_write a32, 0.00000 // (op 2790) call()
v_accvgpr_write a33, 0.00000 // (op 2790) call()
v_accvgpr_write a34, 0.00000 // (op 2790) call()
v_accvgpr_write a35, 0.00000 // (op 2790) call()
v_accvgpr_write a36, 0.00000 // (op 3037) call()
v_accvgpr_write a37, 0.00000 // (op 3037) call()
v_accvgpr_write a38, 0.00000 // (op 3037) call()
v_accvgpr_write a39, 0.00000 // (op 3037) call()
v_accvgpr_write a40, 0.00000 // (op 3436) call()
v_accvgpr_write a41, 0.00000 // (op 3436) call()
v_accvgpr_write a42, 0.00000 // (op 3436) call()
v_accvgpr_write a43, 0.00000 // (op 3436) call()
v_accvgpr_write a44, 0.00000 // (op 3683) call()
v_accvgpr_write a45, 0.00000 // (op 3683) call()
v_accvgpr_write a46, 0.00000 // (op 3683) call()
v_accvgpr_write a47, 0.00000 // (op 3683) call()
s_mul_hi_i32 s0, s16, s47 // (op 10183) 
v_lshrrev_b32 v22, 2, v1 // (op 10183) 
s_add_i32 s1, s0, s16 // (op 10183) 

v_mul_lo_u32 v23, v22, 3 // (op 10183) 

s_ashr_i32 s0, s1, 31 // (op 10183) 
v_and_b32 v22, -4, v1 // (op 10183) 
s_cmp_eq_i32 s47, 0 // (op 10183) 
s_mov_b32 s18, scc // (op 10183) 
v_sub_u32 v24, v1, v22 // (op 10183) 

s_cmp_eq_i32 s18, 0 // (op 10183) 
s_cselect_b32 s19, 0, -1 // (op 10183) 

v_lshlrev_b32 v22, 5, v24 // (op 10183) 

v_mov_b32 v25, 0 // (op 10183) convert
v_mov_b32 v24, v22 // (op 10183) convert

s_and_b32 s18, s13, 31 // (op 10183) 
s_lshl_b32 s20, 1, s18 // (op 10183) 
s_add_i32 s21, s20, s19 // (op 10183) 


s_and_b32 s19, s0, s21 // (op 10183) 


s_add_i32 s0, s1, s19 // (op 10183) 


s_ashr_i32 s1, s0, s18 // (op 10183) 


s_lshl_b32 s0, s13, 24 // (op 10183) 
s_ashr_i32 s18, s0, 31 // (op 10183) 

s_xor_b32 s0, s1, s18 // (op 10183) 

s_sub_i32 s1, s0, s18 // (op 10183) 


s_mul_hi_i32 s0, s1, s14 // (op 10183) 
s_add_i32 s18, s0, s1 // (op 10183) 

s_ashr_i32 s0, s18, 31 // (op 10183) 
s_cmp_eq_i32 s14, 0 // (op 10183) 
s_mov_b32 s19, scc // (op 10183) 
s_cmp_eq_i32 s19, 0 // (op 10183) 
s_cselect_b32 s20, 0, -1 // (op 10183) 

s_and_b32 s19, s15, 31 // (op 10183) 
s_lshl_b32 s21, 1, s19 // (op 10183) 
s_add_i32 s22, s21, s20 // (op 10183) 


s_and_b32 s23, s0, s22 // (op 10183) 


s_add_i32 s0, s18, s23 // (op 10183) 


s_ashr_i32 s18, s0, s19 // (op 10183) 


s_lshl_b32 s0, s15, 24 // (op 10183) 
s_ashr_i32 s19, s0, 31 // (op 10183) 

s_xor_b32 s0, s18, s19 // (op 10183) 

s_sub_i32 s18, s0, s19 // (op 10183) 


s_mul_i32 s0, s1, s48 // (op 10183) 

s_sub_u32 s1, s16, s0 // (op 10183) 

s_mul_i32 s0, s18, s48 // (op 10183) 
s_add_u32 s0, s0, s1 // (op 10183) 


s_mul_hi_i32 s1, s16, s10 // (op 10183) 
s_add_i32 s18, s1, s16 // (op 10183) 

s_ashr_i32 s1, s18, 31 // (op 10183) 
s_cmp_eq_i32 s10, 0 // (op 10183) 
s_mov_b32 s19, scc // (op 10183) 
s_cmp_eq_i32 s19, 0 // (op 10183) 
s_cselect_b32 s20, 0, -1 // (op 10183) 

s_and_b32 s19, s12, 31 // (op 10183) 
s_lshl_b32 s21, 1, s19 // (op 10183) 
s_add_i32 s22, s21, s20 // (op 10183) 


s_and_b32 s23, s1, s22 // (op 10183) 


s_add_i32 s1, s18, s23 // (op 10183) 


s_ashr_i32 s18, s1, s19 // (op 10183) 


s_lshl_b32 s1, s12, 24 // (op 10183) 
s_ashr_i32 s19, s1, 31 // (op 10183) 

s_xor_b32 s1, s18, s19 // (op 10183) 

s_sub_i32 s18, s1, s19 // (op 10183) 


s_add_u32 s22, s8, 64 // (op 10183) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 10183) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 10183) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 10183) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 10183) 

s_mul_i32 s1, s48, s22 // (op 10183) 

s_mul_i32 s19, s44, s1 // (op 10183) 

s_mul_i32 s1, s18, s19 // (op 10183) 

s_sub_u32 s19, s16, s1 // (op 10183) 

s_mov_b32 s1, s19 // (op 10183) 
s_mul_hi_i32 s20, s1, s45 // (op 10183) 
s_add_i32 s21, s20, s1 // (op 10183) 


s_ashr_i32 s1, s21, 31 // (op 10183) 
s_cmp_eq_i32 s45, 0 // (op 10183) 
s_mov_b32 s20, scc // (op 10183) 
s_cmp_eq_i32 s20, 0 // (op 10183) 
s_cselect_b32 s22, 0, -1 // (op 10183) 

s_and_b32 s20, s11, 31 // (op 10183) 
s_lshl_b32 s23, 1, s20 // (op 10183) 
s_add_i32 s49, s23, s22 // (op 10183) 


s_and_b32 s22, s1, s49 // (op 10183) 


s_add_i32 s1, s21, s22 // (op 10183) 


s_ashr_i32 s21, s1, s20 // (op 10183) 


s_lshl_b32 s1, s11, 24 // (op 10183) 
s_ashr_i32 s20, s1, 31 // (op 10183) 

s_xor_b32 s1, s21, s20 // (op 10183) 

s_sub_i32 s21, s1, s20 // (op 10183) 


s_mul_i32 s1, s46, s48 // (op 10183) 
s_sub_i32 s20, s6, s1 // (op 10183) 
s_mul_i32 s22, s21, s20 // (op 10183) 


s_sub_u32 s23, s19, s22 // (op 10183) 


s_add_u32 s19, s23, s1 // (op 10183) 


s_cmp_eq_i32 s18, 0 // (op 10183) 
s_mov_b32 s1, scc // (op 10183) 

s_cmp_eq_i32 s1, 0 // (op 10183) 
s_cselect_b32 s18, s19, s0 // (op 10183) 



v_mov_b32 v22, 192 // (op 10183) 
v_mul_lo_u32 v26, s18, v22 // (op 10183) 

v_add_u32 v26, v26, v23 // (op 10183) 


v_mul_lo_u32 v23, v26, s25 // (op 10183) most significant: low of low * high
v_mul_hi_u32 v27, v26, s24 // (op 10183) most significant: high of low * low
v_mul_lo_u32 v22, v26, s24 // (op 10183) least significant: low of low * low
v_add_u32 v23, v23, v27 // (op 10183) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v24 // (op 10183) least significant half
v_addc_co_u32 v23, s[0:1], v23, v25, s[0:1] // (op 10183) most significant half



v_lshlrev_b64 v[24:25], 2, v[22:23] // (op 10183) 

v_lshrrev_b64 v[22:23], 3, v[24:25] // (op 10183) 

v_mov_b32 v24, v22 // (op 10183) convert

s_mul_i32 s1, s24, s29 // (op 10185) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 10185) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 10185) least significant
s_mul_hi_u32 s19, s24, s28 // (op 10185) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10185) most significant: sum
s_add_u32 s1, s1, s19 // (op 10185) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10185) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10185) 

s_mov_b32 s20, s26 // (op 10185) reg expression
s_mov_b32 s21, s27 // (op 10185) reg expression
s_mov_b32 s22, s0 // (op 10185) reg expression
s_mov_b32 s23, 131072 // (op 10185) call()

s_mul_hi_i32 s0, s16, s47 // (op 10202) 
v_lshrrev_b32 v22, 2, v1 // (op 10202) 
s_add_i32 s1, s0, s16 // (op 10202) 

v_mul_lo_u32 v23, v22, 3 // (op 10202) 

s_ashr_i32 s0, s1, 31 // (op 10202) 
v_and_b32 v22, -4, v1 // (op 10202) 
s_cmp_eq_i32 s47, 0 // (op 10202) 
s_mov_b32 s18, scc // (op 10202) 
v_sub_u32 v25, v1, v22 // (op 10202) 

s_cmp_eq_i32 s18, 0 // (op 10202) 
s_cselect_b32 s19, 0, -1 // (op 10202) 

v_lshlrev_b32 v22, 5, v25 // (op 10202) 

v_mov_b32 v27, 0 // (op 10202) convert
v_mov_b32 v26, v22 // (op 10202) convert

s_and_b32 s18, s13, 31 // (op 10202) 
s_lshl_b32 s49, 1, s18 // (op 10202) 
s_add_i32 s50, s49, s19 // (op 10202) 


s_and_b32 s19, s0, s50 // (op 10202) 


s_add_i32 s0, s1, s19 // (op 10202) 


s_ashr_i32 s1, s0, s18 // (op 10202) 


s_lshl_b32 s0, s13, 24 // (op 10202) 
s_ashr_i32 s18, s0, 31 // (op 10202) 

s_xor_b32 s0, s1, s18 // (op 10202) 

s_sub_i32 s1, s0, s18 // (op 10202) 


s_mul_hi_i32 s0, s1, s14 // (op 10202) 
s_add_i32 s18, s0, s1 // (op 10202) 

s_ashr_i32 s0, s18, 31 // (op 10202) 
s_cmp_eq_i32 s14, 0 // (op 10202) 
s_mov_b32 s19, scc // (op 10202) 
s_cmp_eq_i32 s19, 0 // (op 10202) 
s_cselect_b32 s49, 0, -1 // (op 10202) 

s_and_b32 s19, s15, 31 // (op 10202) 
s_lshl_b32 s50, 1, s19 // (op 10202) 
s_add_i32 s51, s50, s49 // (op 10202) 


s_and_b32 s49, s0, s51 // (op 10202) 


s_add_i32 s0, s18, s49 // (op 10202) 


s_ashr_i32 s18, s0, s19 // (op 10202) 


s_lshl_b32 s0, s15, 24 // (op 10202) 
s_ashr_i32 s19, s0, 31 // (op 10202) 

s_xor_b32 s0, s18, s19 // (op 10202) 

s_sub_i32 s18, s0, s19 // (op 10202) 


s_mul_i32 s0, s1, s48 // (op 10202) 

s_sub_u32 s1, s16, s0 // (op 10202) 

s_mul_i32 s0, s18, s48 // (op 10202) 
s_add_u32 s0, s0, s1 // (op 10202) 


s_mul_hi_i32 s1, s16, s10 // (op 10202) 
s_add_i32 s18, s1, s16 // (op 10202) 

s_ashr_i32 s1, s18, 31 // (op 10202) 
s_cmp_eq_i32 s10, 0 // (op 10202) 
s_mov_b32 s19, scc // (op 10202) 
s_cmp_eq_i32 s19, 0 // (op 10202) 
s_cselect_b32 s49, 0, -1 // (op 10202) 

s_and_b32 s19, s12, 31 // (op 10202) 
s_lshl_b32 s50, 1, s19 // (op 10202) 
s_add_i32 s51, s50, s49 // (op 10202) 


s_and_b32 s49, s1, s51 // (op 10202) 


s_add_i32 s1, s18, s49 // (op 10202) 


s_ashr_i32 s18, s1, s19 // (op 10202) 


s_lshl_b32 s1, s12, 24 // (op 10202) 
s_ashr_i32 s19, s1, 31 // (op 10202) 

s_xor_b32 s1, s18, s19 // (op 10202) 

s_sub_i32 s18, s1, s19 // (op 10202) 


s_add_u32 s54, s8, 64 // (op 10202) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 10202) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 10202) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 10202) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 10202) 

s_mul_i32 s1, s48, s50 // (op 10202) 

s_mul_i32 s19, s44, s1 // (op 10202) 

s_mul_i32 s1, s18, s19 // (op 10202) 

s_sub_u32 s19, s16, s1 // (op 10202) 

s_mov_b32 s1, s19 // (op 10202) 
s_mul_hi_i32 s49, s1, s45 // (op 10202) 
s_add_i32 s50, s49, s1 // (op 10202) 


s_ashr_i32 s1, s50, 31 // (op 10202) 
s_cmp_eq_i32 s45, 0 // (op 10202) 
s_mov_b32 s49, scc // (op 10202) 
s_cmp_eq_i32 s49, 0 // (op 10202) 
s_cselect_b32 s51, 0, -1 // (op 10202) 

s_and_b32 s49, s11, 31 // (op 10202) 
s_lshl_b32 s52, 1, s49 // (op 10202) 
s_add_i32 s53, s52, s51 // (op 10202) 


s_and_b32 s51, s1, s53 // (op 10202) 


s_add_i32 s1, s50, s51 // (op 10202) 


s_ashr_i32 s50, s1, s49 // (op 10202) 


s_lshl_b32 s1, s11, 24 // (op 10202) 
s_ashr_i32 s49, s1, 31 // (op 10202) 

s_xor_b32 s1, s50, s49 // (op 10202) 

s_sub_i32 s50, s1, s49 // (op 10202) 


s_mul_i32 s1, s46, s48 // (op 10202) 
s_sub_i32 s49, s6, s1 // (op 10202) 
s_mul_i32 s51, s50, s49 // (op 10202) 


s_sub_u32 s49, s19, s51 // (op 10202) 


s_add_u32 s19, s49, s1 // (op 10202) 


s_cmp_eq_i32 s18, 0 // (op 10202) 
s_mov_b32 s1, scc // (op 10202) 

s_cmp_eq_i32 s1, 0 // (op 10202) 
s_cselect_b32 s18, s19, s0 // (op 10202) 



v_mov_b32 v22, 192 // (op 10202) 
v_mul_lo_u32 v25, s18, v22 // (op 10202) 

v_add_u32 v25, v25, v23 // (op 10202) 


v_mul_lo_u32 v23, v25, s25 // (op 10202) most significant: low of low * high
v_mul_hi_u32 v28, v25, s24 // (op 10202) most significant: high of low * low
v_mul_lo_u32 v22, v25, s24 // (op 10202) least significant: low of low * low
v_add_u32 v23, v23, v28 // (op 10202) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v26 // (op 10202) least significant half
v_addc_co_u32 v23, s[0:1], v23, v27, s[0:1] // (op 10202) most significant half



v_lshlrev_b64 v[26:27], 2, v[22:23] // (op 10202) 

v_lshrrev_b64 v[22:23], 3, v[26:27] // (op 10202) 

v_mov_b32 v25, v22 // (op 10202) convert

s_mul_i32 s1, s24, s29 // (op 10204) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 10204) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 10204) least significant
s_mul_hi_u32 s19, s24, s28 // (op 10204) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10204) most significant: sum
s_add_u32 s1, s1, s19 // (op 10204) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10204) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10204) 

s_mov_b32 s20, s26 // (op 10204) reg expression
s_mov_b32 s21, s27 // (op 10204) reg expression
s_mov_b32 s22, s0 // (op 10204) reg expression
s_mov_b32 s23, 131072 // (op 10204) call()

s_mul_hi_i32 s0, s16, s47 // (op 10247) 
v_lshrrev_b32 v22, 2, v1 // (op 10247) 
s_add_i32 s1, s0, s16 // (op 10247) 

v_and_b32 v23, -4, v1 // (op 10247) 
s_ashr_i32 s0, s1, 31 // (op 10247) 
v_sub_u32 v26, v1, v23 // (op 10247) 

s_cmp_eq_i32 s47, 0 // (op 10247) 
s_mov_b32 s18, scc // (op 10247) 
v_lshlrev_b32 v23, 5, v26 // (op 10247) 

s_cmp_eq_i32 s18, 0 // (op 10247) 
s_cselect_b32 s19, 0, -1 // (op 10247) 

v_mov_b32 v27, 0 // (op 10247) convert
v_mov_b32 v26, v23 // (op 10247) convert

s_and_b32 s18, s13, 31 // (op 10247) 
s_lshl_b32 s49, 1, s18 // (op 10247) 
s_add_i32 s50, s49, s19 // (op 10247) 


s_and_b32 s19, s0, s50 // (op 10247) 


s_add_i32 s0, s1, s19 // (op 10247) 


s_ashr_i32 s1, s0, s18 // (op 10247) 


s_lshl_b32 s0, s13, 24 // (op 10247) 
s_ashr_i32 s18, s0, 31 // (op 10247) 

s_xor_b32 s0, s1, s18 // (op 10247) 

s_sub_i32 s1, s0, s18 // (op 10247) 


s_mul_hi_i32 s0, s1, s14 // (op 10247) 
s_add_i32 s18, s0, s1 // (op 10247) 

s_ashr_i32 s0, s18, 31 // (op 10247) 
s_cmp_eq_i32 s14, 0 // (op 10247) 
s_mov_b32 s19, scc // (op 10247) 
s_cmp_eq_i32 s19, 0 // (op 10247) 
s_cselect_b32 s49, 0, -1 // (op 10247) 

s_and_b32 s19, s15, 31 // (op 10247) 
s_lshl_b32 s50, 1, s19 // (op 10247) 
s_add_i32 s51, s50, s49 // (op 10247) 


s_and_b32 s49, s0, s51 // (op 10247) 


s_add_i32 s0, s18, s49 // (op 10247) 


s_ashr_i32 s18, s0, s19 // (op 10247) 


s_lshl_b32 s0, s15, 24 // (op 10247) 
s_ashr_i32 s19, s0, 31 // (op 10247) 

s_xor_b32 s0, s18, s19 // (op 10247) 

s_sub_i32 s18, s0, s19 // (op 10247) 


s_add_u32 s54, s8, 64 // (op 10247) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 10247) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 10247) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 10247) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 10247) 

s_mul_i32 s0, s18, s50 // (op 10247) 

s_sub_i32 s49, s1, s0 // (op 10247) 


s_mul_hi_i32 s0, s16, s10 // (op 10247) 
s_add_i32 s1, s0, s16 // (op 10247) 

s_ashr_i32 s0, s1, 31 // (op 10247) 
s_cmp_eq_i32 s10, 0 // (op 10247) 
s_mov_b32 s18, scc // (op 10247) 
s_cmp_eq_i32 s18, 0 // (op 10247) 
s_cselect_b32 s19, 0, -1 // (op 10247) 

s_and_b32 s18, s12, 31 // (op 10247) 
s_lshl_b32 s52, 1, s18 // (op 10247) 
s_add_i32 s53, s52, s19 // (op 10247) 


s_and_b32 s19, s0, s53 // (op 10247) 


s_add_i32 s0, s1, s19 // (op 10247) 


s_ashr_i32 s1, s0, s18 // (op 10247) 


s_lshl_b32 s0, s12, 24 // (op 10247) 
s_ashr_i32 s18, s0, 31 // (op 10247) 

s_xor_b32 s0, s1, s18 // (op 10247) 

s_sub_i32 s1, s0, s18 // (op 10247) 


s_mul_i32 s0, s48, s50 // (op 10247) 

s_mul_i32 s18, s44, s0 // (op 10247) 

s_mul_i32 s0, s1, s18 // (op 10247) 

s_sub_u32 s18, s16, s0 // (op 10247) 

s_mul_hi_i32 s0, s18, s45 // (op 10247) 
s_add_i32 s19, s0, s18 // (op 10247) 


s_ashr_i32 s0, s19, 31 // (op 10247) 
s_cmp_eq_i32 s45, 0 // (op 10247) 
s_mov_b32 s18, scc // (op 10247) 
s_cmp_eq_i32 s18, 0 // (op 10247) 
s_cselect_b32 s50, 0, -1 // (op 10247) 

s_and_b32 s18, s11, 31 // (op 10247) 
s_lshl_b32 s51, 1, s18 // (op 10247) 
s_add_i32 s52, s51, s50 // (op 10247) 


s_and_b32 s50, s0, s52 // (op 10247) 


s_add_i32 s0, s19, s50 // (op 10247) 


s_ashr_i32 s19, s0, s18 // (op 10247) 


s_lshl_b32 s0, s11, 24 // (op 10247) 
s_ashr_i32 s18, s0, 31 // (op 10247) 

s_xor_b32 s0, s19, s18 // (op 10247) 

s_sub_i32 s19, s0, s18 // (op 10247) 


s_cmp_eq_i32 s1, 0 // (op 10247) 
s_mov_b32 s0, scc // (op 10247) 

s_cmp_eq_i32 s0, 0 // (op 10247) 
s_cselect_b32 s1, s19, s49 // (op 10247) 



v_lshl_add_u32 v23, s1, 6, v22 // (op 10247) 


v_mul_lo_u32 v29, v23, s31 // (op 10247) most significant: low of low * high
v_mul_hi_u32 v22, v23, s30 // (op 10247) most significant: high of low * low
v_mul_lo_u32 v28, v23, s30 // (op 10247) least significant: low of low * low
v_add_u32 v29, v29, v22 // (op 10247) most significant: sum


v_add_co_u32 v22, s[0:1], v26, v28 // (op 10247) least significant half
v_addc_co_u32 v23, s[0:1], v27, v29, s[0:1] // (op 10247) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 10247) 


v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 10247) 

v_mov_b32 v22, v26 // (op 10247) convert

s_mul_i32 s1, s30, s9 // (op 10249) most significant: low of low * high
s_mul_i32 s18, s31, s8 // (op 10249) most significant: low of high * low
s_mul_i32 s0, s30, s8 // (op 10249) least significant
s_mul_hi_u32 s19, s30, s8 // (op 10249) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10249) most significant: sum
s_add_u32 s1, s1, s19 // (op 10249) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10249) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10249) 

s_mov_b32 s52, s32 // (op 10249) reg expression
s_mov_b32 s53, s33 // (op 10249) reg expression
s_mov_b32 s54, s0 // (op 10249) reg expression
s_mov_b32 s55, 131072 // (op 10249) call()

s_mul_hi_i32 s0, s16, s47 // (op 10266) 
v_lshrrev_b32 v23, 2, v1 // (op 10266) 
s_add_i32 s1, s0, s16 // (op 10266) 

v_and_b32 v26, -4, v1 // (op 10266) 
s_ashr_i32 s0, s1, 31 // (op 10266) 
v_sub_u32 v27, v1, v26 // (op 10266) 

s_cmp_eq_i32 s47, 0 // (op 10266) 
s_mov_b32 s18, scc // (op 10266) 
v_lshlrev_b32 v26, 5, v27 // (op 10266) 

s_cmp_eq_i32 s18, 0 // (op 10266) 
s_cselect_b32 s19, 0, -1 // (op 10266) 

v_mov_b32 v29, 0 // (op 10266) convert
v_mov_b32 v28, v26 // (op 10266) convert

s_and_b32 s18, s13, 31 // (op 10266) 
s_lshl_b32 s49, 1, s18 // (op 10266) 
s_add_i32 s50, s49, s19 // (op 10266) 


s_and_b32 s19, s0, s50 // (op 10266) 


s_add_i32 s0, s1, s19 // (op 10266) 


s_ashr_i32 s1, s0, s18 // (op 10266) 


s_lshl_b32 s0, s13, 24 // (op 10266) 
s_ashr_i32 s18, s0, 31 // (op 10266) 

s_xor_b32 s0, s1, s18 // (op 10266) 

s_sub_i32 s1, s0, s18 // (op 10266) 


s_mul_hi_i32 s0, s1, s14 // (op 10266) 
s_add_i32 s18, s0, s1 // (op 10266) 

s_ashr_i32 s0, s18, 31 // (op 10266) 
s_cmp_eq_i32 s14, 0 // (op 10266) 
s_mov_b32 s19, scc // (op 10266) 
s_cmp_eq_i32 s19, 0 // (op 10266) 
s_cselect_b32 s49, 0, -1 // (op 10266) 

s_and_b32 s19, s15, 31 // (op 10266) 
s_lshl_b32 s50, 1, s19 // (op 10266) 
s_add_i32 s51, s50, s49 // (op 10266) 


s_and_b32 s49, s0, s51 // (op 10266) 


s_add_i32 s0, s18, s49 // (op 10266) 


s_ashr_i32 s18, s0, s19 // (op 10266) 


s_lshl_b32 s0, s15, 24 // (op 10266) 
s_ashr_i32 s19, s0, 31 // (op 10266) 

s_xor_b32 s0, s18, s19 // (op 10266) 

s_sub_i32 s18, s0, s19 // (op 10266) 


s_add_u32 s50, s8, 64 // (op 10266) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 10266) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 10266) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 10266) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 10266) 

s_mul_i32 s0, s18, s50 // (op 10266) 

s_sub_i32 s49, s1, s0 // (op 10266) 


s_mul_hi_i32 s0, s16, s10 // (op 10266) 
s_add_i32 s1, s0, s16 // (op 10266) 

s_ashr_i32 s0, s1, 31 // (op 10266) 
s_cmp_eq_i32 s10, 0 // (op 10266) 
s_mov_b32 s18, scc // (op 10266) 
s_cmp_eq_i32 s18, 0 // (op 10266) 
s_cselect_b32 s19, 0, -1 // (op 10266) 

s_and_b32 s18, s12, 31 // (op 10266) 
s_lshl_b32 s64, 1, s18 // (op 10266) 
s_add_i32 s65, s64, s19 // (op 10266) 


s_and_b32 s19, s0, s65 // (op 10266) 


s_add_i32 s0, s1, s19 // (op 10266) 


s_ashr_i32 s1, s0, s18 // (op 10266) 


s_lshl_b32 s0, s12, 24 // (op 10266) 
s_ashr_i32 s18, s0, 31 // (op 10266) 

s_xor_b32 s0, s1, s18 // (op 10266) 

s_sub_i32 s1, s0, s18 // (op 10266) 


s_mul_i32 s0, s48, s50 // (op 10266) 

s_mul_i32 s18, s44, s0 // (op 10266) 

s_mul_i32 s0, s1, s18 // (op 10266) 

s_sub_u32 s18, s16, s0 // (op 10266) 

s_mul_hi_i32 s0, s18, s45 // (op 10266) 
s_add_i32 s19, s0, s18 // (op 10266) 


s_ashr_i32 s0, s19, 31 // (op 10266) 
s_cmp_eq_i32 s45, 0 // (op 10266) 
s_mov_b32 s18, scc // (op 10266) 
s_cmp_eq_i32 s18, 0 // (op 10266) 
s_cselect_b32 s50, 0, -1 // (op 10266) 

s_and_b32 s18, s11, 31 // (op 10266) 
s_lshl_b32 s51, 1, s18 // (op 10266) 
s_add_i32 s64, s51, s50 // (op 10266) 


s_and_b32 s50, s0, s64 // (op 10266) 


s_add_i32 s0, s19, s50 // (op 10266) 


s_ashr_i32 s19, s0, s18 // (op 10266) 


s_lshl_b32 s0, s11, 24 // (op 10266) 
s_ashr_i32 s18, s0, 31 // (op 10266) 

s_xor_b32 s0, s19, s18 // (op 10266) 

s_sub_i32 s19, s0, s18 // (op 10266) 


s_cmp_eq_i32 s1, 0 // (op 10266) 
s_mov_b32 s0, scc // (op 10266) 

s_cmp_eq_i32 s0, 0 // (op 10266) 
s_cselect_b32 s1, s19, s49 // (op 10266) 



v_lshl_add_u32 v26, s1, 6, v23 // (op 10266) 


v_mul_lo_u32 v31, v26, s31 // (op 10266) most significant: low of low * high
v_mul_hi_u32 v23, v26, s30 // (op 10266) most significant: high of low * low
v_mul_lo_u32 v30, v26, s30 // (op 10266) least significant: low of low * low
v_add_u32 v31, v31, v23 // (op 10266) most significant: sum


v_add_co_u32 v26, s[0:1], v28, v30 // (op 10266) least significant half
v_addc_co_u32 v27, s[0:1], v29, v31, s[0:1] // (op 10266) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 10266) 


v_lshrrev_b64 v[28:29], 3, v[26:27] // (op 10266) 

v_mov_b32 v23, v28 // (op 10266) convert

s_mul_i32 s1, s30, s9 // (op 10268) most significant: low of low * high
s_mul_i32 s18, s31, s8 // (op 10268) most significant: low of high * low
s_mul_i32 s0, s30, s8 // (op 10268) least significant
s_mul_hi_u32 s19, s30, s8 // (op 10268) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10268) most significant: sum
s_add_u32 s1, s1, s19 // (op 10268) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10268) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10268) 

s_mov_b32 s52, s32 // (op 10268) reg expression
s_mov_b32 s53, s33 // (op 10268) reg expression
s_mov_b32 s54, s0 // (op 10268) reg expression
s_mov_b32 s55, 131072 // (op 10268) call()

s_mul_hi_i32 s0, s16, s47 // (op 10311) 
v_lshrrev_b32 v26, 2, v1 // (op 10311) 
s_add_i32 s1, s0, s16 // (op 10311) 

v_mul_lo_u32 v27, v26, 3 // (op 10311) 

s_ashr_i32 s0, s1, 31 // (op 10311) 
v_and_b32 v26, -4, v1 // (op 10311) 
s_cmp_eq_i32 s47, 0 // (op 10311) 
s_mov_b32 s18, scc // (op 10311) 
v_sub_u32 v28, v1, v26 // (op 10311) 

s_cmp_eq_i32 s18, 0 // (op 10311) 
s_cselect_b32 s19, 0, -1 // (op 10311) 

s_and_b32 s18, s13, 31 // (op 10311) 
s_lshl_b32 s49, 1, s18 // (op 10311) 
s_add_i32 s50, s49, s19 // (op 10311) 


s_and_b32 s19, s0, s50 // (op 10311) 


s_add_i32 s0, s1, s19 // (op 10311) 


s_ashr_i32 s1, s0, s18 // (op 10311) 


s_lshl_b32 s0, s13, 24 // (op 10311) 
s_ashr_i32 s18, s0, 31 // (op 10311) 

s_xor_b32 s0, s1, s18 // (op 10311) 

s_sub_i32 s1, s0, s18 // (op 10311) 


s_mul_hi_i32 s0, s1, s14 // (op 10311) 
s_add_i32 s18, s0, s1 // (op 10311) 

s_ashr_i32 s0, s18, 31 // (op 10311) 
s_cmp_eq_i32 s14, 0 // (op 10311) 
s_mov_b32 s19, scc // (op 10311) 
s_cmp_eq_i32 s19, 0 // (op 10311) 
s_cselect_b32 s49, 0, -1 // (op 10311) 

s_and_b32 s19, s15, 31 // (op 10311) 
s_lshl_b32 s50, 1, s19 // (op 10311) 
s_add_i32 s51, s50, s49 // (op 10311) 


s_and_b32 s49, s0, s51 // (op 10311) 


s_add_i32 s0, s18, s49 // (op 10311) 


s_ashr_i32 s18, s0, s19 // (op 10311) 


s_lshl_b32 s0, s15, 24 // (op 10311) 
s_ashr_i32 s19, s0, 31 // (op 10311) 

s_xor_b32 s0, s18, s19 // (op 10311) 

s_sub_i32 s18, s0, s19 // (op 10311) 


s_mul_i32 s0, s1, s48 // (op 10311) 

s_sub_u32 s1, s16, s0 // (op 10311) 

s_mul_i32 s0, s18, s48 // (op 10311) 
s_add_u32 s0, s0, s1 // (op 10311) 


s_mul_hi_i32 s1, s16, s10 // (op 10311) 
s_add_i32 s18, s1, s16 // (op 10311) 

s_ashr_i32 s1, s18, 31 // (op 10311) 
s_cmp_eq_i32 s10, 0 // (op 10311) 
s_mov_b32 s19, scc // (op 10311) 
s_cmp_eq_i32 s19, 0 // (op 10311) 
s_cselect_b32 s49, 0, -1 // (op 10311) 

s_and_b32 s19, s12, 31 // (op 10311) 
s_lshl_b32 s50, 1, s19 // (op 10311) 
s_add_i32 s51, s50, s49 // (op 10311) 


s_and_b32 s49, s1, s51 // (op 10311) 


s_add_i32 s1, s18, s49 // (op 10311) 


s_ashr_i32 s18, s1, s19 // (op 10311) 


s_lshl_b32 s1, s12, 24 // (op 10311) 
s_ashr_i32 s19, s1, 31 // (op 10311) 

s_xor_b32 s1, s18, s19 // (op 10311) 

s_sub_i32 s18, s1, s19 // (op 10311) 


s_add_u32 s50, s8, 64 // (op 10311) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 10311) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 10311) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 10311) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 10311) 

s_mul_i32 s1, s48, s50 // (op 10311) 

s_mul_i32 s19, s44, s1 // (op 10311) 

s_mul_i32 s1, s18, s19 // (op 10311) 

s_sub_u32 s19, s16, s1 // (op 10311) 

s_mov_b32 s1, s19 // (op 10311) 
s_mul_hi_i32 s49, s1, s45 // (op 10311) 
s_add_i32 s50, s49, s1 // (op 10311) 


s_ashr_i32 s1, s50, 31 // (op 10311) 
s_cmp_eq_i32 s45, 0 // (op 10311) 
s_mov_b32 s49, scc // (op 10311) 
s_cmp_eq_i32 s49, 0 // (op 10311) 
s_cselect_b32 s51, 0, -1 // (op 10311) 

s_and_b32 s49, s11, 31 // (op 10311) 
s_lshl_b32 s64, 1, s49 // (op 10311) 
s_add_i32 s65, s64, s51 // (op 10311) 


s_and_b32 s51, s1, s65 // (op 10311) 


s_add_i32 s1, s50, s51 // (op 10311) 


s_ashr_i32 s50, s1, s49 // (op 10311) 


s_lshl_b32 s1, s11, 24 // (op 10311) 
s_ashr_i32 s49, s1, 31 // (op 10311) 

s_xor_b32 s1, s50, s49 // (op 10311) 

s_sub_i32 s50, s1, s49 // (op 10311) 


s_mul_i32 s1, s46, s48 // (op 10311) 
s_sub_i32 s49, s6, s1 // (op 10311) 
s_mul_i32 s51, s50, s49 // (op 10311) 


s_sub_u32 s49, s19, s51 // (op 10311) 


s_add_u32 s19, s49, s1 // (op 10311) 


s_cmp_eq_i32 s18, 0 // (op 10311) 
s_mov_b32 s1, scc // (op 10311) 

s_cmp_eq_i32 s1, 0 // (op 10311) 
s_cselect_b32 s18, s19, s0 // (op 10311) 



v_mov_b32 v26, 192 // (op 10311) 
v_mul_lo_u32 v29, s18, v26 // (op 10311) 

v_add_u32 v29, v29, v27 // (op 10311) 


v_mul_lo_u32 v26, v29, s34 // (op 10311) 
v_add_u32 v26, v26, v28 // (op 10311) 


s_mul_i32 s1, s34, s29 // (op 10313) most significant: low of low * high
s_mul_i32 s18, s35, s28 // (op 10313) most significant: low of high * low
s_mul_i32 s0, s34, s28 // (op 10313) least significant
s_mul_hi_u32 s19, s34, s28 // (op 10313) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10313) most significant: sum
s_add_u32 s1, s1, s19 // (op 10313) most significant: sum


s_mov_b32 s64, s36 // (op 10313) reg expression
s_mov_b32 s65, s37 // (op 10313) reg expression
s_mov_b32 s66, s0 // (op 10313) reg expression
s_mov_b32 s67, 131072 // (op 10313) call()

s_mul_hi_i32 s0, s16, s47 // (op 10330) 
v_lshrrev_b32 v27, 2, v1 // (op 10330) 
s_add_i32 s1, s0, s16 // (op 10330) 

v_mul_lo_u32 v28, v27, 3 // (op 10330) 

s_ashr_i32 s0, s1, 31 // (op 10330) 
v_and_b32 v27, -4, v1 // (op 10330) 
s_cmp_eq_i32 s47, 0 // (op 10330) 
s_mov_b32 s18, scc // (op 10330) 
v_sub_u32 v29, v1, v27 // (op 10330) 

s_cmp_eq_i32 s18, 0 // (op 10330) 
s_cselect_b32 s19, 0, -1 // (op 10330) 

s_and_b32 s18, s13, 31 // (op 10330) 
s_lshl_b32 s49, 1, s18 // (op 10330) 
s_add_i32 s50, s49, s19 // (op 10330) 


s_and_b32 s19, s0, s50 // (op 10330) 


s_add_i32 s0, s1, s19 // (op 10330) 


s_ashr_i32 s1, s0, s18 // (op 10330) 


s_lshl_b32 s0, s13, 24 // (op 10330) 
s_ashr_i32 s18, s0, 31 // (op 10330) 

s_xor_b32 s0, s1, s18 // (op 10330) 

s_sub_i32 s1, s0, s18 // (op 10330) 


s_mul_hi_i32 s0, s1, s14 // (op 10330) 
s_add_i32 s18, s0, s1 // (op 10330) 

s_ashr_i32 s0, s18, 31 // (op 10330) 
s_cmp_eq_i32 s14, 0 // (op 10330) 
s_mov_b32 s19, scc // (op 10330) 
s_cmp_eq_i32 s19, 0 // (op 10330) 
s_cselect_b32 s49, 0, -1 // (op 10330) 

s_and_b32 s19, s15, 31 // (op 10330) 
s_lshl_b32 s50, 1, s19 // (op 10330) 
s_add_i32 s51, s50, s49 // (op 10330) 


s_and_b32 s49, s0, s51 // (op 10330) 


s_add_i32 s0, s18, s49 // (op 10330) 


s_ashr_i32 s18, s0, s19 // (op 10330) 


s_lshl_b32 s0, s15, 24 // (op 10330) 
s_ashr_i32 s19, s0, 31 // (op 10330) 

s_xor_b32 s0, s18, s19 // (op 10330) 

s_sub_i32 s18, s0, s19 // (op 10330) 


s_mul_i32 s0, s1, s48 // (op 10330) 

s_sub_u32 s1, s16, s0 // (op 10330) 

s_mul_i32 s0, s18, s48 // (op 10330) 
s_add_u32 s0, s0, s1 // (op 10330) 


s_mul_hi_i32 s1, s16, s10 // (op 10330) 
s_add_i32 s18, s1, s16 // (op 10330) 

s_ashr_i32 s1, s18, 31 // (op 10330) 
s_cmp_eq_i32 s10, 0 // (op 10330) 
s_mov_b32 s19, scc // (op 10330) 
s_cmp_eq_i32 s19, 0 // (op 10330) 
s_cselect_b32 s49, 0, -1 // (op 10330) 

s_and_b32 s19, s12, 31 // (op 10330) 
s_lshl_b32 s50, 1, s19 // (op 10330) 
s_add_i32 s51, s50, s49 // (op 10330) 


s_and_b32 s49, s1, s51 // (op 10330) 


s_add_i32 s1, s18, s49 // (op 10330) 


s_ashr_i32 s18, s1, s19 // (op 10330) 


s_lshl_b32 s1, s12, 24 // (op 10330) 
s_ashr_i32 s19, s1, 31 // (op 10330) 

s_xor_b32 s1, s18, s19 // (op 10330) 

s_sub_i32 s18, s1, s19 // (op 10330) 


s_add_u32 s50, s8, 64 // (op 10330) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 10330) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 10330) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 10330) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 10330) 

s_mul_i32 s1, s48, s50 // (op 10330) 

s_mul_i32 s19, s44, s1 // (op 10330) 

s_mul_i32 s1, s18, s19 // (op 10330) 

s_sub_u32 s19, s16, s1 // (op 10330) 

s_mov_b32 s1, s19 // (op 10330) 
s_mul_hi_i32 s49, s1, s45 // (op 10330) 
s_add_i32 s50, s49, s1 // (op 10330) 


s_ashr_i32 s1, s50, 31 // (op 10330) 
s_cmp_eq_i32 s45, 0 // (op 10330) 
s_mov_b32 s49, scc // (op 10330) 
s_cmp_eq_i32 s49, 0 // (op 10330) 
s_cselect_b32 s51, 0, -1 // (op 10330) 

s_and_b32 s49, s11, 31 // (op 10330) 
s_lshl_b32 s68, 1, s49 // (op 10330) 
s_add_i32 s69, s68, s51 // (op 10330) 


s_and_b32 s51, s1, s69 // (op 10330) 


s_add_i32 s1, s50, s51 // (op 10330) 


s_ashr_i32 s50, s1, s49 // (op 10330) 


s_lshl_b32 s1, s11, 24 // (op 10330) 
s_ashr_i32 s49, s1, 31 // (op 10330) 

s_xor_b32 s1, s50, s49 // (op 10330) 

s_sub_i32 s50, s1, s49 // (op 10330) 


s_mul_i32 s1, s46, s48 // (op 10330) 
s_sub_i32 s49, s6, s1 // (op 10330) 
s_mul_i32 s51, s50, s49 // (op 10330) 


s_sub_u32 s49, s19, s51 // (op 10330) 


s_add_u32 s19, s49, s1 // (op 10330) 


s_cmp_eq_i32 s18, 0 // (op 10330) 
s_mov_b32 s1, scc // (op 10330) 

s_cmp_eq_i32 s1, 0 // (op 10330) 
s_cselect_b32 s18, s19, s0 // (op 10330) 



v_mov_b32 v27, 192 // (op 10330) 
v_mul_lo_u32 v30, s18, v27 // (op 10330) 

v_add_u32 v30, v30, v28 // (op 10330) 


v_mul_lo_u32 v27, v30, s34 // (op 10330) 
v_add_u32 v27, v27, v29 // (op 10330) 


s_mul_i32 s1, s34, s29 // (op 10332) most significant: low of low * high
s_mul_i32 s18, s35, s28 // (op 10332) most significant: low of high * low
s_mul_i32 s0, s34, s28 // (op 10332) least significant
s_mul_hi_u32 s19, s34, s28 // (op 10332) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10332) most significant: sum
s_add_u32 s1, s1, s19 // (op 10332) most significant: sum


s_mov_b32 s64, s36 // (op 10332) reg expression
s_mov_b32 s65, s37 // (op 10332) reg expression
s_mov_b32 s66, s0 // (op 10332) reg expression
s_mov_b32 s67, 131072 // (op 10332) call()

s_mul_hi_i32 s0, s16, s47 // (op 10375) 
v_lshrrev_b32 v28, 2, v1 // (op 10375) 
s_add_i32 s1, s0, s16 // (op 10375) 

v_and_b32 v29, -4, v1 // (op 10375) 
s_ashr_i32 s0, s1, 31 // (op 10375) 
v_sub_u32 v30, v1, v29 // (op 10375) 

s_cmp_eq_i32 s47, 0 // (op 10375) 
s_mov_b32 s18, scc // (op 10375) 
s_cmp_eq_i32 s18, 0 // (op 10375) 
s_cselect_b32 s19, 0, -1 // (op 10375) 

s_and_b32 s18, s13, 31 // (op 10375) 
s_lshl_b32 s49, 1, s18 // (op 10375) 
s_add_i32 s50, s49, s19 // (op 10375) 


s_and_b32 s19, s0, s50 // (op 10375) 


s_add_i32 s0, s1, s19 // (op 10375) 


s_ashr_i32 s1, s0, s18 // (op 10375) 


s_lshl_b32 s0, s13, 24 // (op 10375) 
s_ashr_i32 s18, s0, 31 // (op 10375) 

s_xor_b32 s0, s1, s18 // (op 10375) 

s_sub_i32 s1, s0, s18 // (op 10375) 


s_mul_hi_i32 s0, s1, s14 // (op 10375) 
s_add_i32 s18, s0, s1 // (op 10375) 

s_ashr_i32 s0, s18, 31 // (op 10375) 
s_cmp_eq_i32 s14, 0 // (op 10375) 
s_mov_b32 s19, scc // (op 10375) 
s_cmp_eq_i32 s19, 0 // (op 10375) 
s_cselect_b32 s49, 0, -1 // (op 10375) 

s_and_b32 s19, s15, 31 // (op 10375) 
s_lshl_b32 s50, 1, s19 // (op 10375) 
s_add_i32 s51, s50, s49 // (op 10375) 


s_and_b32 s49, s0, s51 // (op 10375) 


s_add_i32 s0, s18, s49 // (op 10375) 


s_ashr_i32 s18, s0, s19 // (op 10375) 


s_lshl_b32 s0, s15, 24 // (op 10375) 
s_ashr_i32 s19, s0, 31 // (op 10375) 

s_xor_b32 s0, s18, s19 // (op 10375) 

s_sub_i32 s18, s0, s19 // (op 10375) 


s_add_u32 s50, s8, 64 // (op 10375) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 10375) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 10375) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 10375) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 10375) 

s_mul_i32 s0, s18, s50 // (op 10375) 

s_sub_i32 s49, s1, s0 // (op 10375) 


s_mul_hi_i32 s0, s16, s10 // (op 10375) 
s_add_i32 s1, s0, s16 // (op 10375) 

s_ashr_i32 s0, s1, 31 // (op 10375) 
s_cmp_eq_i32 s10, 0 // (op 10375) 
s_mov_b32 s18, scc // (op 10375) 
s_cmp_eq_i32 s18, 0 // (op 10375) 
s_cselect_b32 s19, 0, -1 // (op 10375) 

s_and_b32 s18, s12, 31 // (op 10375) 
s_lshl_b32 s68, 1, s18 // (op 10375) 
s_add_i32 s69, s68, s19 // (op 10375) 


s_and_b32 s19, s0, s69 // (op 10375) 


s_add_i32 s0, s1, s19 // (op 10375) 


s_ashr_i32 s1, s0, s18 // (op 10375) 


s_lshl_b32 s0, s12, 24 // (op 10375) 
s_ashr_i32 s18, s0, 31 // (op 10375) 

s_xor_b32 s0, s1, s18 // (op 10375) 

s_sub_i32 s1, s0, s18 // (op 10375) 


s_mul_i32 s0, s48, s50 // (op 10375) 

s_mul_i32 s18, s44, s0 // (op 10375) 

s_mul_i32 s0, s1, s18 // (op 10375) 

s_sub_u32 s18, s16, s0 // (op 10375) 

s_mul_hi_i32 s0, s18, s45 // (op 10375) 
s_add_i32 s19, s0, s18 // (op 10375) 


s_ashr_i32 s0, s19, 31 // (op 10375) 
s_cmp_eq_i32 s45, 0 // (op 10375) 
s_mov_b32 s18, scc // (op 10375) 
s_cmp_eq_i32 s18, 0 // (op 10375) 
s_cselect_b32 s50, 0, -1 // (op 10375) 

s_and_b32 s18, s11, 31 // (op 10375) 
s_lshl_b32 s51, 1, s18 // (op 10375) 
s_add_i32 s68, s51, s50 // (op 10375) 


s_and_b32 s50, s0, s68 // (op 10375) 


s_add_i32 s0, s19, s50 // (op 10375) 


s_ashr_i32 s19, s0, s18 // (op 10375) 


s_lshl_b32 s0, s11, 24 // (op 10375) 
s_ashr_i32 s18, s0, 31 // (op 10375) 

s_xor_b32 s0, s19, s18 // (op 10375) 

s_sub_i32 s19, s0, s18 // (op 10375) 


s_cmp_eq_i32 s1, 0 // (op 10375) 
s_mov_b32 s0, scc // (op 10375) 

s_cmp_eq_i32 s0, 0 // (op 10375) 
s_cselect_b32 s1, s19, s49 // (op 10375) 



v_lshl_add_u32 v29, s1, 6, v28 // (op 10375) 


v_mul_lo_u32 v28, v29, s38 // (op 10375) 

v_add_u32 v29, v30, v28 // (op 10375) 


s_mul_i32 s1, s38, s9 // (op 10377) most significant: low of low * high
s_mul_i32 s18, s39, s8 // (op 10377) most significant: low of high * low
s_mul_i32 s0, s38, s8 // (op 10377) least significant
s_mul_hi_u32 s19, s38, s8 // (op 10377) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10377) most significant: sum
s_add_u32 s1, s1, s19 // (op 10377) most significant: sum


s_mov_b32 s68, s40 // (op 10377) reg expression
s_mov_b32 s69, s41 // (op 10377) reg expression
s_mov_b32 s70, s0 // (op 10377) reg expression
s_mov_b32 s71, 131072 // (op 10377) call()

s_mul_hi_i32 s0, s16, s47 // (op 10394) 
v_lshrrev_b32 v28, 2, v1 // (op 10394) 
s_add_i32 s1, s0, s16 // (op 10394) 

v_and_b32 v30, -4, v1 // (op 10394) 
s_ashr_i32 s0, s1, 31 // (op 10394) 
v_sub_u32 v31, v1, v30 // (op 10394) 

s_cmp_eq_i32 s47, 0 // (op 10394) 
s_mov_b32 s18, scc // (op 10394) 
s_cmp_eq_i32 s18, 0 // (op 10394) 
s_cselect_b32 s19, 0, -1 // (op 10394) 

s_and_b32 s18, s13, 31 // (op 10394) 
s_lshl_b32 s49, 1, s18 // (op 10394) 
s_add_i32 s50, s49, s19 // (op 10394) 


s_and_b32 s19, s0, s50 // (op 10394) 


s_add_i32 s0, s1, s19 // (op 10394) 


s_ashr_i32 s1, s0, s18 // (op 10394) 


s_lshl_b32 s0, s13, 24 // (op 10394) 
s_ashr_i32 s18, s0, 31 // (op 10394) 

s_xor_b32 s0, s1, s18 // (op 10394) 

s_sub_i32 s1, s0, s18 // (op 10394) 


s_mul_hi_i32 s0, s1, s14 // (op 10394) 
s_add_i32 s18, s0, s1 // (op 10394) 

s_ashr_i32 s0, s18, 31 // (op 10394) 
s_cmp_eq_i32 s14, 0 // (op 10394) 
s_mov_b32 s19, scc // (op 10394) 
s_cmp_eq_i32 s19, 0 // (op 10394) 
s_cselect_b32 s49, 0, -1 // (op 10394) 

s_and_b32 s19, s15, 31 // (op 10394) 
s_lshl_b32 s50, 1, s19 // (op 10394) 
s_add_i32 s51, s50, s49 // (op 10394) 


s_and_b32 s49, s0, s51 // (op 10394) 


s_add_i32 s0, s18, s49 // (op 10394) 


s_ashr_i32 s18, s0, s19 // (op 10394) 


s_lshl_b32 s0, s15, 24 // (op 10394) 
s_ashr_i32 s19, s0, 31 // (op 10394) 

s_xor_b32 s0, s18, s19 // (op 10394) 

s_sub_i32 s18, s0, s19 // (op 10394) 


s_add_u32 s50, s8, 64 // (op 10394) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 10394) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 10394) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 10394) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 10394) 

s_mul_i32 s0, s18, s50 // (op 10394) 

s_sub_i32 s49, s1, s0 // (op 10394) 


s_mul_hi_i32 s0, s16, s10 // (op 10394) 
s_add_i32 s1, s0, s16 // (op 10394) 

s_ashr_i32 s0, s1, 31 // (op 10394) 
s_cmp_eq_i32 s10, 0 // (op 10394) 
s_mov_b32 s18, scc // (op 10394) 
s_cmp_eq_i32 s18, 0 // (op 10394) 
s_cselect_b32 s19, 0, -1 // (op 10394) 

s_and_b32 s18, s12, 31 // (op 10394) 
s_lshl_b32 s72, 1, s18 // (op 10394) 
s_add_i32 s73, s72, s19 // (op 10394) 


s_and_b32 s19, s0, s73 // (op 10394) 


s_add_i32 s0, s1, s19 // (op 10394) 


s_ashr_i32 s1, s0, s18 // (op 10394) 


s_lshl_b32 s0, s12, 24 // (op 10394) 
s_ashr_i32 s18, s0, 31 // (op 10394) 

s_xor_b32 s0, s1, s18 // (op 10394) 

s_sub_i32 s1, s0, s18 // (op 10394) 


s_mul_i32 s0, s48, s50 // (op 10394) 

s_mul_i32 s18, s44, s0 // (op 10394) 

s_mul_i32 s0, s1, s18 // (op 10394) 

s_sub_u32 s18, s16, s0 // (op 10394) 

s_mul_hi_i32 s0, s18, s45 // (op 10394) 
s_add_i32 s19, s0, s18 // (op 10394) 


s_ashr_i32 s0, s19, 31 // (op 10394) 
s_cmp_eq_i32 s45, 0 // (op 10394) 
s_mov_b32 s18, scc // (op 10394) 
s_cmp_eq_i32 s18, 0 // (op 10394) 
s_cselect_b32 s50, 0, -1 // (op 10394) 

s_and_b32 s18, s11, 31 // (op 10394) 
s_lshl_b32 s51, 1, s18 // (op 10394) 
s_add_i32 s72, s51, s50 // (op 10394) 


s_and_b32 s50, s0, s72 // (op 10394) 


s_add_i32 s0, s19, s50 // (op 10394) 


s_ashr_i32 s19, s0, s18 // (op 10394) 


s_lshl_b32 s0, s11, 24 // (op 10394) 
s_ashr_i32 s18, s0, 31 // (op 10394) 

s_xor_b32 s0, s19, s18 // (op 10394) 

s_sub_i32 s19, s0, s18 // (op 10394) 


s_cmp_eq_i32 s1, 0 // (op 10394) 
s_mov_b32 s0, scc // (op 10394) 

s_cmp_eq_i32 s0, 0 // (op 10394) 
s_cselect_b32 s1, s19, s49 // (op 10394) 



v_lshl_add_u32 v30, s1, 6, v28 // (op 10394) 


v_mul_lo_u32 v28, v30, s38 // (op 10394) 

v_add_u32 v30, v31, v28 // (op 10394) 


s_mul_i32 s1, s38, s9 // (op 10396) most significant: low of low * high
s_mul_i32 s18, s39, s8 // (op 10396) most significant: low of high * low
s_mul_i32 s0, s38, s8 // (op 10396) least significant
s_mul_hi_u32 s19, s38, s8 // (op 10396) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10396) most significant: sum
s_add_u32 s1, s1, s19 // (op 10396) most significant: sum


s_mov_b32 s68, s40 // (op 10396) reg expression
s_mov_b32 s69, s41 // (op 10396) reg expression
s_mov_b32 s70, s0 // (op 10396) reg expression
s_mov_b32 s71, 131072 // (op 10396) call()




buffer_load_dwordx4 v[32:35], v22, s[52:55], 0 offen // (op 9519) Load value



buffer_load_ubyte v28, v29, s[68:71], 0 offen offset: 0 // (op 9522) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 9525) 
v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 9525) 


v_mov_b32 v31, v24 // (op 9525) 
buffer_load_dwordx4 v[38:41], v31, s[20:23], 0 offen // (op 9525) Load value
v_add_u32 v31, v31, v36 // (op 9525) 
buffer_load_dwordx4 v[42:45], v31, s[20:23], 0 offen // (op 9525) Load value
v_add_u32 v31, v31, v36 // (op 9525) 
buffer_load_dwordx4 v[46:49], v31, s[20:23], 0 offen // (op 9525) Load value



v_mov_b64 v[36:37], s[34:35] // (op 9528) convert

v_mov_b32 v31, v26 // (op 9528) 
buffer_load_ubyte v50, v31, s[64:67], 0 offen offset: 0 // (op 9528) Load value
v_add_u32 v31, v31, v36 // (op 9528) 
buffer_load_ubyte v51, v31, s[64:67], 0 offen offset: 0 // (op 9528) Load value
v_add_u32 v31, v31, v36 // (op 9528) 
buffer_load_ubyte v52, v31, s[64:67], 0 offen offset: 0 // (op 9528) Load value





buffer_load_dwordx4 v[54:57], v22, s[52:55], 0 offen offset:64 // (op 9531) Load value



buffer_load_ubyte v31, v29, s[68:71], 0 offen offset:4 // (op 9534) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 9537) 
v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 9537) 


v_mov_b32 v53, v24 // (op 9537) 
buffer_load_dwordx4 v[58:61], v53, s[20:23], 0 offen offset:64 // (op 9537) Load value
v_add_u32 v53, v53, v36 // (op 9537) 
buffer_load_dwordx4 v[62:65], v53, s[20:23], 0 offen offset:64 // (op 9537) Load value
v_add_u32 v53, v53, v36 // (op 9537) 
buffer_load_dwordx4 v[66:69], v53, s[20:23], 0 offen offset:64 // (op 9537) Load value



v_mov_b64 v[36:37], s[34:35] // (op 9540) convert

v_mov_b32 v53, v26 // (op 9540) 
buffer_load_ubyte v70, v53, s[64:67], 0 offen offset:4 // (op 9540) Load value
v_add_u32 v53, v53, v36 // (op 9540) 
buffer_load_ubyte v71, v53, s[64:67], 0 offen offset:4 // (op 9540) Load value
v_add_u32 v53, v53, v36 // (op 9540) 
buffer_load_ubyte v72, v53, s[64:67], 0 offen offset:4 // (op 9540) Load value





buffer_load_dwordx4 v[74:77], v22, s[52:55], 0 offen offset:128 // (op 9543) Load value



buffer_load_ubyte v53, v29, s[68:71], 0 offen offset:8 // (op 9546) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 9549) 
v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 9549) 


v_mov_b32 v73, v24 // (op 9549) 
buffer_load_dwordx4 v[78:81], v73, s[20:23], 0 offen offset:128 // (op 9549) Load value
v_add_u32 v73, v73, v36 // (op 9549) 
buffer_load_dwordx4 v[82:85], v73, s[20:23], 0 offen offset:128 // (op 9549) Load value
v_add_u32 v73, v73, v36 // (op 9549) 
buffer_load_dwordx4 v[86:89], v73, s[20:23], 0 offen offset:128 // (op 9549) Load value



v_mov_b64 v[36:37], s[34:35] // (op 9552) convert

v_mov_b32 v73, v26 // (op 9552) 
buffer_load_ubyte v90, v73, s[64:67], 0 offen offset:8 // (op 9552) Load value
v_add_u32 v73, v73, v36 // (op 9552) 
buffer_load_ubyte v91, v73, s[64:67], 0 offen offset:8 // (op 9552) Load value
v_add_u32 v73, v73, v36 // (op 9552) 
buffer_load_ubyte v92, v73, s[64:67], 0 offen offset:8 // (op 9552) Load value





buffer_load_dwordx4 v[94:97], v22, s[52:55], 0 offen offset:192 // (op 9555) Load value




buffer_load_ubyte v22, v29, s[68:71], 0 offen offset:12 // (op 9558) Load value


s_lshl_b64 s[0:1], s[24:25], 2 // (op 9561) 
v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 9561) 


v_mov_b32 v29, v24 // (op 9561) 
buffer_load_dwordx4 v[98:101], v29, s[20:23], 0 offen offset:192 // (op 9561) Load value
v_add_u32 v29, v29, v36 // (op 9561) 
buffer_load_dwordx4 v[102:105], v29, s[20:23], 0 offen offset:192 // (op 9561) Load value
v_add_u32 v29, v29, v36 // (op 9561) 
buffer_load_dwordx4 v[106:109], v29, s[20:23], 0 offen offset:192 // (op 9561) Load value




v_mov_b64 v[36:37], s[34:35] // (op 9564) convert

v_mov_b32 v24, v26 // (op 9564) 
buffer_load_ubyte v110, v24, s[64:67], 0 offen offset:12 // (op 9564) Load value
v_add_u32 v24, v24, v36 // (op 9564) 
buffer_load_ubyte v111, v24, s[64:67], 0 offen offset:12 // (op 9564) Load value
v_add_u32 v24, v24, v36 // (op 9564) 
buffer_load_ubyte v112, v24, s[64:67], 0 offen offset:12 // (op 9564) Load value



v_and_b32 v24, -4, v1 // (op 10524) 
v_sub_u32 v26, v1, v24 // (op 10524) 

v_lshlrev_b32 v24, 5, v26 // (op 10524) 

v_lshrrev_b32 v26, 2, v1 // (op 10524) 
v_lshl_add_u32 v29, v26, 7, v24 // (op 10524) 


v_lshlrev_b32 v24, 2, v29 // (op 10524) 

v_lshrrev_b32 v26, 3, v24 // (op 10524) 




s_waitcnt vmcnt(31)
ds_write_b128 v26, v[32:35] // (op 9566) Store local data 

v_and_b32 v24, -4, v1 // (op 10640) 
v_sub_u32 v26, v1, v24 // (op 10640) 

v_and_b32 v24, 4294967292, v1 // (op 10640) 
v_add_u32 v29, v24, v26 // (op 10640) 





s_waitcnt vmcnt(30)
ds_write_b8 v29, v28 offset:4096 // (op 9567) Store local data 

v_and_b32 v24, -4, v1 // (op 10466) 
v_sub_u32 v26, v1, v24 // (op 10466) 

v_lshlrev_b32 v24, 5, v26 // (op 10466) 

v_lshrrev_b32 v26, 2, v1 // (op 10466) 
v_mul_lo_u32 v29, v26, 3 // (op 10466) 

v_lshl_add_u32 v26, v29, 7, v24 // (op 10466) 


v_lshlrev_b32 v24, 2, v26 // (op 10466) 

v_lshrrev_b32 v26, 3, v24 // (op 10466) 




s_waitcnt vmcnt(29)
ds_write_b128 v26, v[38:41] offset:4352 // (op 9568) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v26, v[42:45] offset:4416 // (op 9568) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v26, v[46:49] offset:4480 // (op 9568) Store local data 

v_and_b32 v24, -4, v1 // (op 10582) 
v_sub_u32 v26, v1, v24 // (op 10582) 

v_lshrrev_b32 v24, 2, v1 // (op 10582) 
v_mul_lo_u32 v29, v24, 3 // (op 10582) 

v_lshl_add_u32 v24, v29, 2, v26 // (op 10582) 





s_waitcnt vmcnt(26)
ds_write_b8 v24, v50 offset:16640 // (op 9569) Store local data 
s_waitcnt vmcnt(25)
ds_write_b8 v24, v51 offset:16644 // (op 9569) Store local data 
s_waitcnt vmcnt(24)
ds_write_b8 v24, v52 offset:16648 // (op 9569) Store local data 

v_and_b32 v24, -4, v1 // (op 10437) 
v_sub_u32 v26, v1, v24 // (op 10437) 

v_lshlrev_b32 v24, 5, v26 // (op 10437) 

v_lshrrev_b32 v26, 2, v1 // (op 10437) 
v_mul_lo_u32 v29, v26, 3 // (op 10437) 

v_lshl_add_u32 v26, v29, 7, v24 // (op 10437) 


v_lshlrev_b32 v24, 2, v26 // (op 10437) 

v_lshrrev_b32 v26, 3, v24 // (op 10437) 

v_and_b32 v24, -4, v1 // (op 10495) 
v_sub_u32 v29, v1, v24 // (op 10495) 

v_lshlrev_b32 v24, 5, v29 // (op 10495) 

v_lshrrev_b32 v29, 2, v1 // (op 10495) 
v_lshl_add_u32 v73, v29, 7, v24 // (op 10495) 


v_lshlrev_b32 v24, 2, v73 // (op 10495) 

v_lshrrev_b32 v29, 3, v24 // (op 10495) 

v_and_b32 v24, -4, v1 // (op 10553) 
v_sub_u32 v73, v1, v24 // (op 10553) 

v_lshrrev_b32 v24, 2, v1 // (op 10553) 
v_mul_lo_u32 v93, v24, 3 // (op 10553) 

v_lshl_add_u32 v24, v93, 2, v73 // (op 10553) 


v_and_b32 v73, -4, v1 // (op 10611) 
v_sub_u32 v93, v1, v73 // (op 10611) 

v_and_b32 v73, 4294967292, v1 // (op 10611) 
v_add_u32 v36, v73, v93 // (op 10611) 


v_and_b32 v37, -4, v1 // (op 10692) 
v_sub_u32 v73, v1, v37 // (op 10692) 

v_lshlrev_b32 v37, 5, v73 // (op 10692) 

v_lshrrev_b32 v73, 2, v1 // (op 10692) 
v_mul_lo_u32 v93, v73, 3 // (op 10692) 

v_lshl_add_u32 v73, v93, 7, v37 // (op 10692) 


v_lshlrev_b32 v37, 2, v73 // (op 10692) 

v_lshrrev_b32 v73, 3, v37 // (op 10692) 

v_and_b32 v37, -4, v1 // (op 10721) 
v_sub_u32 v93, v1, v37 // (op 10721) 

v_lshlrev_b32 v37, 5, v93 // (op 10721) 

v_lshrrev_b32 v93, 2, v1 // (op 10721) 
v_lshl_add_u32 v113, v93, 7, v37 // (op 10721) 


v_lshlrev_b32 v37, 2, v113 // (op 10721) 

v_lshrrev_b32 v93, 3, v37 // (op 10721) 

v_and_b32 v37, -4, v1 // (op 10750) 
v_sub_u32 v113, v1, v37 // (op 10750) 

v_lshrrev_b32 v37, 2, v1 // (op 10750) 
v_mul_lo_u32 v114, v37, 3 // (op 10750) 

v_lshl_add_u32 v37, v114, 2, v113 // (op 10750) 


v_and_b32 v113, -4, v1 // (op 10779) 
v_sub_u32 v114, v1, v113 // (op 10779) 

v_and_b32 v113, 4294967292, v1 // (op 10779) 
v_add_u32 v115, v113, v114 // (op 10779) 


v_and_b32 v113, -4, v1 // (op 10808) 
v_sub_u32 v114, v1, v113 // (op 10808) 

v_lshlrev_b32 v113, 5, v114 // (op 10808) 

v_lshrrev_b32 v114, 2, v1 // (op 10808) 
v_mul_lo_u32 v116, v114, 3 // (op 10808) 

v_lshl_add_u32 v114, v116, 7, v113 // (op 10808) 


v_lshlrev_b32 v113, 2, v114 // (op 10808) 

v_lshrrev_b32 v114, 3, v113 // (op 10808) 

v_and_b32 v113, -4, v1 // (op 10837) 
v_sub_u32 v116, v1, v113 // (op 10837) 

v_lshlrev_b32 v113, 5, v116 // (op 10837) 

v_lshrrev_b32 v116, 2, v1 // (op 10837) 
v_lshl_add_u32 v117, v116, 7, v113 // (op 10837) 


v_lshlrev_b32 v113, 2, v117 // (op 10837) 

v_lshrrev_b32 v116, 3, v113 // (op 10837) 

v_and_b32 v113, -4, v1 // (op 10866) 
v_sub_u32 v117, v1, v113 // (op 10866) 

v_lshrrev_b32 v113, 2, v1 // (op 10866) 
v_mul_lo_u32 v118, v113, 3 // (op 10866) 

v_lshl_add_u32 v113, v118, 2, v117 // (op 10866) 


v_and_b32 v117, -4, v1 // (op 10895) 
v_sub_u32 v118, v1, v117 // (op 10895) 

v_and_b32 v117, 4294967292, v1 // (op 10895) 
v_add_u32 v119, v117, v118 // (op 10895) 


v_and_b32 v117, -4, v1 // (op 10924) 
v_sub_u32 v118, v1, v117 // (op 10924) 

v_lshlrev_b32 v117, 5, v118 // (op 10924) 

v_lshrrev_b32 v118, 2, v1 // (op 10924) 
v_mul_lo_u32 v120, v118, 3 // (op 10924) 

v_lshl_add_u32 v118, v120, 7, v117 // (op 10924) 


v_lshlrev_b32 v117, 2, v118 // (op 10924) 

v_lshrrev_b32 v118, 3, v117 // (op 10924) 

v_and_b32 v117, -4, v1 // (op 10953) 
v_sub_u32 v120, v1, v117 // (op 10953) 

v_lshlrev_b32 v117, 5, v120 // (op 10953) 

v_lshrrev_b32 v120, 2, v1 // (op 10953) 
v_lshl_add_u32 v121, v120, 7, v117 // (op 10953) 


v_lshlrev_b32 v117, 2, v121 // (op 10953) 

v_lshrrev_b32 v120, 3, v117 // (op 10953) 

v_and_b32 v117, -4, v1 // (op 10982) 
v_sub_u32 v121, v1, v117 // (op 10982) 

v_lshrrev_b32 v117, 2, v1 // (op 10982) 
v_mul_lo_u32 v122, v117, 3 // (op 10982) 

v_lshl_add_u32 v117, v122, 2, v121 // (op 10982) 


v_and_b32 v121, -4, v1 // (op 11011) 
v_sub_u32 v122, v1, v121 // (op 11011) 

v_and_b32 v121, 4294967292, v1 // (op 11011) 
v_add_u32 v123, v121, v122 // (op 11011) 


s_mov_b32 s0, 0 // (op 3469) call()
s_ashr_i64 s[18:19], s[42:43], 7 // (op 3468) 
s_mov_b32 s1, s18 // (op 3468) convert

s_and_b32 s18, s1, 4294967292 // (op 3468) 

s_cmp_lt_u32 s0, s18 // (op 3468) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopBottom_KLoop_3468 // (op 3468) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopBottom_KLoop_3468 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopTop_KLoop_3468:


s_add_u32 s0, s0, 4 // (op 3470) 
v_mov_b32 v121, v23 // (op 10) 

buffer_load_dwordx4 v[32:35], v121, s[52:55], 0 offen offset:256 // (op 10) Load value
s_waitcnt lgkmcnt(3)
s_barrier  // (op 11151) 





v_mov_b32 v122, v30 // (op 25) 

ds_read_b128 v[124:127], v3 offset:4352 // (op 72) Load local data 
buffer_load_ubyte v28, v122, s[68:71], 0 offen offset:16 // (op 25) Load value

s_barrier  // (op 11164) 
s_lshl_b64 s[18:19], s[24:25], 2 // (op 4) 



v_lshrrev_b64 v[128:129], 3, s[18:19] // (op 4) 


v_mov_b32 v130, v25 // (op 4) 
v_mov_b32 v131, v130 // (op 4) 
buffer_load_dwordx4 v[38:41], v131, s[20:23], 0 offen offset:256 // (op 4) Load value
v_add_u32 v131, v131, v128 // (op 4) 
ds_read_b128 v[132:135], v4 // (op 77) Load local data 
buffer_load_dwordx4 v[42:45], v131, s[20:23], 0 offen offset:256 // (op 4) Load value
v_add_u32 v131, v131, v128 // (op 4) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 11178) 
buffer_load_dwordx4 v[46:49], v131, s[20:23], 0 offen offset:256 // (op 4) Load value






ds_read_u8 v131, v0 offset:16640 // (op 82) Load local data 
v_mov_b64 v[128:129], s[34:35] // (op 17) convert

s_barrier  // (op 11192) 
v_mov_b32 v136, v27 // (op 17) 



ds_read_u8 v137, v5 offset:4096 // (op 87) Load local data 
v_mov_b32 v138, v136 // (op 17) 



buffer_load_ubyte v50, v138, s[64:67], 0 offen offset:16 // (op 17) Load value
v_add_u32 v138, v138, v128 // (op 17) 
ds_read_b128 v[140:143], v4 offset:2048 // (op 521) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:127], v[132:135], a[0:3], v131, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 90) 



ds_read_u8 v139, v5 offset:4224 // (op 525) Load local data 
buffer_load_ubyte v51, v138, s[64:67], 0 offen offset:16 // (op 17) Load value



v_add_u32 v138, v138, v128 // (op 17) 
ds_read_b128 v[144:147], v3 offset:6400 // (op 918) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:127], v[140:143], a[4:7], v131, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 526) 





ds_read_u8 v131, v0 offset:16768 // (op 921) Load local data 
buffer_load_ubyte v52, v138, s[64:67], 0 offen offset:16 // (op 17) Load value





s_barrier  // (op 11215) 
ds_read_b128 v[124:127], v3 offset:8448 // (op 1564) Load local data 






ds_read_u8 v138, v0 offset:16896 // (op 1567) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[144:147], v[132:135], a[8:11], v131, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1015) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:147], v[140:143], a[12:15], v131, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 923) 
ds_read_b128 v[148:151], v3 offset:10496 // (op 2210) Load local data 





s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:127], v[132:135], a[16:19], v138, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1661) 
ds_read_u8 v131, v0 offset:17024 // (op 2213) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[140:143], a[20:23], v138, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1569) 
ds_read_b128 v[144:147], v3 offset:12544 // (op 2856) Load local data 





ds_read_u8 v138, v0 offset:17152 // (op 2859) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[132:135], a[24:27], v131, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2307) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[140:143], a[28:31], v131, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2215) 
ds_read_b128 v[124:127], v3 offset:14592 // (op 3502) Load local data 





ds_read_u8 v131, v0 offset:17280 // (op 3505) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[132:135], a[32:35], v138, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2953) 
s_waitcnt vmcnt(31)
ds_write_b128 v93, v[54:57] offset:17408 // (op 257) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[140:143], a[36:39], v138, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2861) 


s_barrier  // (op 11227) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[124:127], v[132:135], a[40:43], v131, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3599) 
s_waitcnt vmcnt(30)
ds_write_b8 v115, v31 offset:21504 // (op 259) Store local data 


s_barrier  // (op 11209) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:127], v[140:143], a[44:47], v131, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3507) 







s_waitcnt vmcnt(29)
ds_write_b128 v73, v[58:61] offset:21760 // (op 247) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v73, v[62:65] offset:21824 // (op 247) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v73, v[66:69] offset:21888 // (op 247) Store local data 
s_barrier  // (op 11221) 



s_waitcnt vmcnt(26)
ds_write_b8 v37, v70 offset:34048 // (op 258) Store local data 
s_waitcnt vmcnt(25)
ds_write_b8 v37, v71 offset:34052 // (op 258) Store local data 
s_waitcnt vmcnt(24)
ds_write_b8 v37, v72 offset:34056 // (op 258) Store local data 



buffer_load_dwordx4 v[54:57], v121, s[52:55], 0 offen offset:320 // (op 260) Load value
s_waitcnt lgkmcnt(3)
s_barrier  // (op 11206) 






ds_read_b128 v[132:135], v7 offset:21760 // (op 249) Load local data 
buffer_load_ubyte v31, v122, s[68:71], 0 offen offset:20 // (op 262) Load value

s_barrier  // (op 11212) 
s_lshl_b64 s[18:19], s[24:25], 2 // (op 246) 



v_lshrrev_b64 v[124:125], 3, s[18:19] // (op 246) 


v_mov_b32 v131, v130 // (op 246) 
ds_read_b128 v[126:129], v8 offset:17408 // (op 251) Load local data 
buffer_load_dwordx4 v[58:61], v131, s[20:23], 0 offen offset:320 // (op 246) Load value
v_add_u32 v131, v131, v124 // (op 246) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 11218) 
buffer_load_dwordx4 v[62:65], v131, s[20:23], 0 offen offset:320 // (op 246) Load value



v_add_u32 v131, v131, v124 // (op 246) 
ds_read_u8 v137, v6 offset:34048 // (op 253) Load local data 
buffer_load_dwordx4 v[66:69], v131, s[20:23], 0 offen offset:320 // (op 246) Load value



s_barrier  // (op 11224) 
v_mov_b64 v[124:125], s[34:35] // (op 261) convert




ds_read_u8 v131, v9 offset:21504 // (op 255) Load local data 
v_mov_b32 v138, v136 // (op 261) 



ds_read_b128 v[140:143], v8 offset:19456 // (op 533) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[132:135], v[126:129], a[0:3], v137, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 256) 
buffer_load_ubyte v70, v138, s[64:67], 0 offen offset:20 // (op 261) Load value



s_barrier  // (op 11356) 
v_add_u32 v138, v138, v124 // (op 261) 
ds_read_u8 v139, v9 offset:21632 // (op 537) Load local data 
buffer_load_ubyte v71, v138, s[64:67], 0 offen offset:20 // (op 261) Load value
v_add_u32 v138, v138, v124 // (op 261) 



s_barrier  // (op 11360) 
ds_read_b128 v[144:147], v7 offset:23808 // (op 928) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[132:135], v[140:143], a[4:7], v137, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 527) 





ds_read_u8 v137, v6 offset:34176 // (op 931) Load local data 
buffer_load_ubyte v72, v138, s[64:67], 0 offen offset:20 // (op 261) Load value





s_barrier  // (op 11239) 
ds_read_b128 v[132:135], v7 offset:25856 // (op 1574) Load local data 






s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[144:147], v[126:129], a[8:11], v137, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1016) 
ds_read_u8 v138, v6 offset:34304 // (op 1577) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:147], v[140:143], a[12:15], v137, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 924) 
ds_read_b128 v[148:151], v7 offset:27904 // (op 2220) Load local data 





s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[126:129], a[16:19], v138, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1662) 
ds_read_u8 v137, v6 offset:34432 // (op 2223) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[140:143], a[20:23], v138, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1570) 
ds_read_b128 v[144:147], v7 offset:29952 // (op 2866) Load local data 





ds_read_u8 v138, v6 offset:34560 // (op 2869) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[126:129], a[24:27], v137, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2308) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[140:143], a[28:31], v137, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2216) 
ds_read_b128 v[132:135], v7 offset:32000 // (op 3512) Load local data 





s_barrier  // (op 11440) 
ds_read_u8 v137, v6 offset:34688 // (op 3515) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[126:129], a[32:35], v138, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2954) 
s_barrier  // (op 11444) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[140:143], a[36:39], v138, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2862) 
s_waitcnt vmcnt(31)
ds_write_b128 v116, v[74:77] offset:17408 // (op 297) Store local data 


s_barrier  // (op 11251) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:135], v[126:129], a[40:43], v137, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3600) 





s_waitcnt vmcnt(30)
ds_write_b8 v119, v53 offset:21504 // (op 299) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:135], v[140:143], a[44:47], v137, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3508) 




s_barrier  // (op 11233) 



s_waitcnt vmcnt(29)
ds_write_b128 v114, v[78:81] offset:21760 // (op 287) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v114, v[82:85] offset:21824 // (op 287) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v114, v[86:89] offset:21888 // (op 287) Store local data 
s_barrier  // (op 11245) 



s_waitcnt vmcnt(26)
ds_write_b8 v113, v90 offset:34048 // (op 298) Store local data 
s_waitcnt vmcnt(25)
ds_write_b8 v113, v91 offset:34052 // (op 298) Store local data 
s_waitcnt vmcnt(24)
ds_write_b8 v113, v92 offset:34056 // (op 298) Store local data 



buffer_load_dwordx4 v[74:77], v121, s[52:55], 0 offen offset:384 // (op 300) Load value
s_waitcnt lgkmcnt(3)
s_barrier  // (op 11230) 






ds_read_b128 v[132:135], v11 offset:21760 // (op 289) Load local data 
buffer_load_ubyte v53, v122, s[68:71], 0 offen offset:24 // (op 302) Load value

s_barrier  // (op 11236) 
s_lshl_b64 s[18:19], s[24:25], 2 // (op 286) 



v_lshrrev_b64 v[124:125], 3, s[18:19] // (op 286) 


v_mov_b32 v131, v130 // (op 286) 
ds_read_b128 v[126:129], v12 offset:17408 // (op 291) Load local data 
buffer_load_dwordx4 v[78:81], v131, s[20:23], 0 offen offset:384 // (op 286) Load value
v_add_u32 v131, v131, v124 // (op 286) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 11242) 
buffer_load_dwordx4 v[82:85], v131, s[20:23], 0 offen offset:384 // (op 286) Load value



v_add_u32 v131, v131, v124 // (op 286) 
ds_read_u8 v137, v10 offset:34048 // (op 293) Load local data 
buffer_load_dwordx4 v[86:89], v131, s[20:23], 0 offen offset:384 // (op 286) Load value



s_barrier  // (op 11248) 
v_mov_b64 v[124:125], s[34:35] // (op 301) convert




ds_read_u8 v131, v13 offset:21504 // (op 295) Load local data 
v_mov_b32 v138, v136 // (op 301) 



ds_read_b128 v[140:143], v12 offset:19456 // (op 541) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[132:135], v[126:129], a[0:3], v137, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 296) 
buffer_load_ubyte v90, v138, s[64:67], 0 offen offset:24 // (op 301) Load value



s_barrier  // (op 11364) 
v_add_u32 v138, v138, v124 // (op 301) 
ds_read_u8 v139, v13 offset:21632 // (op 545) Load local data 
buffer_load_ubyte v91, v138, s[64:67], 0 offen offset:24 // (op 301) Load value
v_add_u32 v138, v138, v124 // (op 301) 



s_barrier  // (op 11368) 
ds_read_b128 v[144:147], v11 offset:23808 // (op 934) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[132:135], v[140:143], a[4:7], v137, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 528) 





ds_read_u8 v137, v10 offset:34176 // (op 937) Load local data 
buffer_load_ubyte v92, v138, s[64:67], 0 offen offset:24 // (op 301) Load value





s_barrier  // (op 11263) 
ds_read_b128 v[132:135], v11 offset:25856 // (op 1580) Load local data 






ds_read_u8 v138, v10 offset:34304 // (op 1583) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[144:147], v[126:129], a[8:11], v137, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1017) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:147], v[140:143], a[12:15], v137, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 925) 
ds_read_b128 v[148:151], v11 offset:27904 // (op 2226) Load local data 





s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[126:129], a[16:19], v138, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1663) 
ds_read_u8 v137, v10 offset:34432 // (op 2229) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[140:143], a[20:23], v138, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1571) 
ds_read_b128 v[144:147], v11 offset:29952 // (op 2872) Load local data 





ds_read_u8 v138, v10 offset:34560 // (op 2875) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[126:129], a[24:27], v137, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2309) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[140:143], a[28:31], v137, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2217) 
ds_read_b128 v[132:135], v11 offset:32000 // (op 3518) Load local data 





s_barrier  // (op 11448) 
ds_read_u8 v137, v10 offset:34688 // (op 3521) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[126:129], a[32:35], v138, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2955) 
s_barrier  // (op 11452) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[140:143], a[36:39], v138, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2863) 
s_waitcnt vmcnt(31)
ds_write_b128 v120, v[94:97] offset:17408 // (op 337) Store local data 


s_barrier  // (op 11275) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:135], v[126:129], a[40:43], v137, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3601) 





s_waitcnt vmcnt(30)
ds_write_b8 v123, v22 offset:21504 // (op 339) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:135], v[140:143], a[44:47], v137, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3509) 




s_barrier  // (op 11257) 



s_waitcnt vmcnt(29)
ds_write_b128 v118, v[98:101] offset:21760 // (op 327) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v118, v[102:105] offset:21824 // (op 327) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v118, v[106:109] offset:21888 // (op 327) Store local data 
s_barrier  // (op 11269) 



s_waitcnt vmcnt(26)
ds_write_b8 v117, v110 offset:34048 // (op 338) Store local data 
s_waitcnt vmcnt(25)
ds_write_b8 v117, v111 offset:34052 // (op 338) Store local data 
s_waitcnt vmcnt(24)
ds_write_b8 v117, v112 offset:34056 // (op 338) Store local data 



buffer_load_dwordx4 v[94:97], v121, s[52:55], 0 offen offset:448 // (op 340) Load value
s_waitcnt lgkmcnt(3)
s_barrier  // (op 11254) 







v_add_u32 v23, 256, v23 // (op 10258) 
ds_read_b128 v[132:135], v15 offset:21760 // (op 329) Load local data 
buffer_load_ubyte v22, v122, s[68:71], 0 offen offset:28 // (op 342) Load value

s_barrier  // (op 11260) 




v_add_u32 v30, 16, v30 // (op 10386) 
s_lshl_b64 s[18:19], s[24:25], 2 // (op 326) 
v_lshrrev_b64 v[124:125], 3, s[18:19] // (op 326) 


v_mov_b32 v131, v130 // (op 326) 
ds_read_b128 v[126:129], v16 offset:17408 // (op 331) Load local data 
buffer_load_dwordx4 v[98:101], v131, s[20:23], 0 offen offset:448 // (op 326) Load value
v_add_u32 v131, v131, v124 // (op 326) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 11266) 
buffer_load_dwordx4 v[102:105], v131, s[20:23], 0 offen offset:448 // (op 326) Load value



v_add_u32 v131, v131, v124 // (op 326) 
ds_read_u8 v121, v14 offset:34048 // (op 333) Load local data 
buffer_load_dwordx4 v[106:109], v131, s[20:23], 0 offen offset:448 // (op 326) Load value



s_barrier  // (op 11272) 




v_add_u32 v25, 256, v25 // (op 10194) 
ds_read_u8 v122, v17 offset:21504 // (op 335) Load local data 
v_mov_b64 v[124:125], s[34:35] // (op 341) convert




v_mov_b32 v130, v136 // (op 341) 
ds_read_b128 v[138:141], v16 offset:19456 // (op 549) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[132:135], v[126:129], a[0:3], v121, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 336) 
buffer_load_ubyte v110, v130, s[64:67], 0 offen offset:28 // (op 341) Load value



s_barrier  // (op 11372) 
v_add_u32 v130, v130, v124 // (op 341) 
ds_read_u8 v131, v17 offset:21632 // (op 553) Load local data 
buffer_load_ubyte v111, v130, s[64:67], 0 offen offset:28 // (op 341) Load value
v_add_u32 v130, v130, v124 // (op 341) 



s_barrier  // (op 11376) 
ds_read_b128 v[142:145], v15 offset:23808 // (op 940) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[132:135], v[138:141], a[4:7], v121, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 529) 





ds_read_u8 v121, v14 offset:34176 // (op 943) Load local data 
buffer_load_ubyte v112, v130, s[64:67], 0 offen offset:28 // (op 341) Load value






ds_read_b128 v[132:135], v15 offset:25856 // (op 1586) Load local data 



s_barrier  // (op 11167) 
v_add_u32 v27, 16, v27 // (op 10322) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[142:145], v[126:129], a[8:11], v121, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1018) 
ds_read_u8 v130, v14 offset:34304 // (op 1589) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[142:145], v[138:141], a[12:15], v121, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 926) 
ds_read_b128 v[146:149], v15 offset:27904 // (op 2232) Load local data 





s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[126:129], a[16:19], v130, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1664) 
ds_read_u8 v121, v14 offset:34432 // (op 2235) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[138:141], a[20:23], v130, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1572) 
ds_read_b128 v[142:145], v15 offset:29952 // (op 2878) Load local data 





ds_read_u8 v130, v14 offset:34560 // (op 2881) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[146:149], v[126:129], a[24:27], v121, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2310) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[146:149], v[138:141], a[28:31], v121, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2218) 
ds_read_b128 v[132:135], v15 offset:32000 // (op 3524) Load local data 





s_barrier  // (op 11456) 
ds_read_u8 v121, v14 offset:34688 // (op 3527) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[142:145], v[126:129], a[32:35], v130, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2956) 
s_barrier  // (op 11460) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[142:145], v[138:141], a[36:39], v130, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2864) 
s_waitcnt vmcnt(31)
ds_write_b128 v29, v[32:35] // (op 13) Store local data 


s_barrier  // (op 11195) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:135], v[126:129], a[40:43], v121, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3602) 





s_waitcnt vmcnt(30)
ds_write_b8 v36, v28 offset:4096 // (op 30) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:135], v[138:141], a[44:47], v121, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3510) 




s_barrier  // (op 11154) 



s_waitcnt vmcnt(29)
ds_write_b128 v26, v[38:41] offset:4352 // (op 6) Store local data 
s_waitcnt vmcnt(28)
ds_write_b128 v26, v[42:45] offset:4416 // (op 6) Store local data 
s_waitcnt vmcnt(27)
ds_write_b128 v26, v[46:49] offset:4480 // (op 6) Store local data 
s_barrier  // (op 11181) 



s_waitcnt vmcnt(26)
ds_write_b8 v24, v50 offset:16640 // (op 21) Store local data 
s_waitcnt vmcnt(25)
ds_write_b8 v24, v51 offset:16644 // (op 21) Store local data 
s_waitcnt vmcnt(24)
ds_write_b8 v24, v52 offset:16648 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[42:43], 7 // (op 3468) 
s_mov_b32 s1, s18 // (op 3468) convert

s_and_b32 s18, s1, 4294967292 // (op 3468) 

s_cmp_lt_u32 s0, s18 // (op 3468) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopTop_KLoop_3468 // (op 3468) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopTop_KLoop_3468 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopBottom_KLoop_3468:

























s_mul_hi_i32 s0, s16, s47 // (op 10146) 
v_lshrrev_b32 v73, 2, v1 // (op 10146) 
s_add_i32 s1, s0, s16 // (op 10146) 

v_mul_lo_u32 v93, v73, 3 // (op 10146) 

s_ashr_i32 s0, s1, 31 // (op 10146) 
v_and_b32 v73, -4, v1 // (op 10146) 
s_cmp_eq_i32 s47, 0 // (op 10146) 
s_mov_b32 s18, scc // (op 10146) 
v_sub_u32 v23, v1, v73 // (op 10146) 

s_cmp_eq_i32 s18, 0 // (op 10146) 
s_cselect_b32 s19, 0, -1 // (op 10146) 

v_lshlrev_b32 v73, 5, v23 // (op 10146) 

s_and_b32 s18, s13, 31 // (op 10146) 
s_lshl_b32 s49, 1, s18 // (op 10146) 
s_add_i32 s50, s49, s19 // (op 10146) 


s_and_b32 s19, s0, s50 // (op 10146) 


s_add_i32 s0, s1, s19 // (op 10146) 


s_ashr_i32 s1, s0, s18 // (op 10146) 


s_lshl_b32 s0, s13, 24 // (op 10146) 
s_ashr_i32 s18, s0, 31 // (op 10146) 

s_xor_b32 s0, s1, s18 // (op 10146) 

s_sub_i32 s1, s0, s18 // (op 10146) 


s_mul_hi_i32 s0, s1, s14 // (op 10146) 
s_add_i32 s18, s0, s1 // (op 10146) 

s_ashr_i32 s0, s18, 31 // (op 10146) 
s_cmp_eq_i32 s14, 0 // (op 10146) 
s_mov_b32 s19, scc // (op 10146) 
s_cmp_eq_i32 s19, 0 // (op 10146) 
s_cselect_b32 s49, 0, -1 // (op 10146) 

s_and_b32 s19, s15, 31 // (op 10146) 
s_lshl_b32 s50, 1, s19 // (op 10146) 
s_add_i32 s51, s50, s49 // (op 10146) 


s_and_b32 s49, s0, s51 // (op 10146) 


s_add_i32 s0, s18, s49 // (op 10146) 


s_ashr_i32 s18, s0, s19 // (op 10146) 


s_lshl_b32 s0, s15, 24 // (op 10146) 
s_ashr_i32 s19, s0, 31 // (op 10146) 

s_xor_b32 s0, s18, s19 // (op 10146) 

s_sub_i32 s18, s0, s19 // (op 10146) 


s_mul_i32 s0, s1, s48 // (op 10146) 

s_sub_u32 s1, s16, s0 // (op 10146) 

s_mul_i32 s0, s18, s48 // (op 10146) 
s_add_u32 s0, s0, s1 // (op 10146) 


s_mul_hi_i32 s1, s16, s10 // (op 10146) 
s_add_i32 s18, s1, s16 // (op 10146) 

s_ashr_i32 s1, s18, 31 // (op 10146) 
s_cmp_eq_i32 s10, 0 // (op 10146) 
s_mov_b32 s19, scc // (op 10146) 
s_cmp_eq_i32 s19, 0 // (op 10146) 
s_cselect_b32 s49, 0, -1 // (op 10146) 

s_and_b32 s19, s12, 31 // (op 10146) 
s_lshl_b32 s50, 1, s19 // (op 10146) 
s_add_i32 s51, s50, s49 // (op 10146) 


s_and_b32 s49, s1, s51 // (op 10146) 


s_add_i32 s1, s18, s49 // (op 10146) 


s_ashr_i32 s18, s1, s19 // (op 10146) 


s_lshl_b32 s1, s12, 24 // (op 10146) 
s_ashr_i32 s19, s1, 31 // (op 10146) 

s_xor_b32 s1, s18, s19 // (op 10146) 

s_sub_i32 s18, s1, s19 // (op 10146) 


s_add_u32 s50, s8, 64 // (op 10146) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 10146) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 10146) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 10146) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 10146) 

s_mul_i32 s1, s48, s50 // (op 10146) 

s_mul_i32 s19, s44, s1 // (op 10146) 

s_mul_i32 s1, s18, s19 // (op 10146) 

s_sub_u32 s19, s16, s1 // (op 10146) 

s_mov_b32 s1, s19 // (op 10146) 
s_mul_hi_i32 s49, s1, s45 // (op 10146) 
s_add_i32 s50, s49, s1 // (op 10146) 


s_ashr_i32 s1, s50, 31 // (op 10146) 
s_cmp_eq_i32 s45, 0 // (op 10146) 
s_mov_b32 s49, scc // (op 10146) 
s_cmp_eq_i32 s49, 0 // (op 10146) 
s_cselect_b32 s51, 0, -1 // (op 10146) 

s_and_b32 s49, s11, 31 // (op 10146) 
s_lshl_b32 s72, 1, s49 // (op 10146) 
s_add_i32 s73, s72, s51 // (op 10146) 


s_and_b32 s51, s1, s73 // (op 10146) 


s_add_i32 s1, s50, s51 // (op 10146) 


s_ashr_i32 s50, s1, s49 // (op 10146) 


s_lshl_b32 s1, s11, 24 // (op 10146) 
s_ashr_i32 s49, s1, 31 // (op 10146) 

s_xor_b32 s1, s50, s49 // (op 10146) 

s_sub_i32 s50, s1, s49 // (op 10146) 


s_mul_i32 s1, s46, s48 // (op 10146) 
s_sub_i32 s49, s6, s1 // (op 10146) 
s_mul_i32 s51, s50, s49 // (op 10146) 


s_sub_u32 s49, s19, s51 // (op 10146) 


s_add_u32 s19, s49, s1 // (op 10146) 


s_cmp_eq_i32 s18, 0 // (op 10146) 
s_mov_b32 s1, scc // (op 10146) 

s_cmp_eq_i32 s1, 0 // (op 10146) 
s_cselect_b32 s18, s19, s0 // (op 10146) 



v_mov_b32 v23, 192 // (op 10146) 
v_mul_lo_u32 v24, s18, v23 // (op 10146) 

v_add_u32 v24, v24, v93 // (op 10146) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 10146) 
s_lshr_b32 s18, s0, 2 // (op 10146) 

v_lshl_add_u32 v23, s18, 9, v73 // (op 10146) 


v_mov_b32 v33, 0 // (op 10146) convert
v_mov_b32 v32, v23 // (op 10146) convert

v_mul_lo_u32 v35, v24, s25 // (op 10146) most significant: low of low * high
v_mul_hi_u32 v23, v24, s24 // (op 10146) most significant: high of low * low
v_mul_lo_u32 v34, v24, s24 // (op 10146) least significant: low of low * low
v_add_u32 v35, v35, v23 // (op 10146) most significant: sum

v_add_co_u32 v34, s[0:1], v34, v32 // (op 10146) least significant half
v_addc_co_u32 v35, s[0:1], v35, v33, s[0:1] // (op 10146) most significant half



v_lshlrev_b64 v[32:33], 2, v[34:35] // (op 10146) 

v_lshrrev_b64 v[34:35], 3, v[32:33] // (op 10146) 

v_mov_b32 v73, v34 // (op 10146) convert

s_mul_i32 s1, s24, s29 // (op 10148) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 10148) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 10148) least significant
s_mul_hi_u32 s19, s24, s28 // (op 10148) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10148) most significant: sum
s_add_u32 s1, s1, s19 // (op 10148) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10148) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10148) 

s_mov_b32 s20, s26 // (op 10148) reg expression
s_mov_b32 s21, s27 // (op 10148) reg expression
s_mov_b32 s22, s0 // (op 10148) reg expression
s_mov_b32 s23, 131072 // (op 10148) call()

s_mul_hi_i32 s0, s16, s47 // (op 10224) 
v_lshrrev_b32 v93, 2, v1 // (op 10224) 
s_add_i32 s1, s0, s16 // (op 10224) 

v_and_b32 v23, -4, v1 // (op 10224) 
s_ashr_i32 s0, s1, 31 // (op 10224) 
v_sub_u32 v24, v1, v23 // (op 10224) 

s_cmp_eq_i32 s47, 0 // (op 10224) 
s_mov_b32 s18, scc // (op 10224) 
v_lshlrev_b32 v23, 5, v24 // (op 10224) 

s_cmp_eq_i32 s18, 0 // (op 10224) 
s_cselect_b32 s19, 0, -1 // (op 10224) 

s_and_b32 s18, s13, 31 // (op 10224) 
s_lshl_b32 s49, 1, s18 // (op 10224) 
s_add_i32 s50, s49, s19 // (op 10224) 


s_and_b32 s19, s0, s50 // (op 10224) 


s_add_i32 s0, s1, s19 // (op 10224) 


s_ashr_i32 s1, s0, s18 // (op 10224) 


s_lshl_b32 s0, s13, 24 // (op 10224) 
s_ashr_i32 s18, s0, 31 // (op 10224) 

s_xor_b32 s0, s1, s18 // (op 10224) 

s_sub_i32 s1, s0, s18 // (op 10224) 


s_mul_hi_i32 s0, s1, s14 // (op 10224) 
s_add_i32 s18, s0, s1 // (op 10224) 

s_ashr_i32 s0, s18, 31 // (op 10224) 
s_cmp_eq_i32 s14, 0 // (op 10224) 
s_mov_b32 s19, scc // (op 10224) 
s_cmp_eq_i32 s19, 0 // (op 10224) 
s_cselect_b32 s49, 0, -1 // (op 10224) 

s_and_b32 s19, s15, 31 // (op 10224) 
s_lshl_b32 s50, 1, s19 // (op 10224) 
s_add_i32 s51, s50, s49 // (op 10224) 


s_and_b32 s49, s0, s51 // (op 10224) 


s_add_i32 s0, s18, s49 // (op 10224) 


s_ashr_i32 s18, s0, s19 // (op 10224) 


s_lshl_b32 s0, s15, 24 // (op 10224) 
s_ashr_i32 s19, s0, 31 // (op 10224) 

s_xor_b32 s0, s18, s19 // (op 10224) 

s_sub_i32 s18, s0, s19 // (op 10224) 


s_add_u32 s50, s8, 64 // (op 10224) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 10224) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 10224) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 10224) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 10224) 

s_mul_i32 s0, s18, s50 // (op 10224) 

s_sub_i32 s49, s1, s0 // (op 10224) 


s_mul_hi_i32 s0, s16, s10 // (op 10224) 
s_add_i32 s1, s0, s16 // (op 10224) 

s_ashr_i32 s0, s1, 31 // (op 10224) 
s_cmp_eq_i32 s10, 0 // (op 10224) 
s_mov_b32 s18, scc // (op 10224) 
s_cmp_eq_i32 s18, 0 // (op 10224) 
s_cselect_b32 s19, 0, -1 // (op 10224) 

s_and_b32 s18, s12, 31 // (op 10224) 
s_lshl_b32 s72, 1, s18 // (op 10224) 
s_add_i32 s73, s72, s19 // (op 10224) 


s_and_b32 s19, s0, s73 // (op 10224) 


s_add_i32 s0, s1, s19 // (op 10224) 


s_ashr_i32 s1, s0, s18 // (op 10224) 


s_lshl_b32 s0, s12, 24 // (op 10224) 
s_ashr_i32 s18, s0, 31 // (op 10224) 

s_xor_b32 s0, s1, s18 // (op 10224) 

s_sub_i32 s1, s0, s18 // (op 10224) 


s_mul_i32 s0, s48, s50 // (op 10224) 

s_mul_i32 s18, s44, s0 // (op 10224) 

s_mul_i32 s0, s1, s18 // (op 10224) 

s_sub_u32 s18, s16, s0 // (op 10224) 

s_mul_hi_i32 s0, s18, s45 // (op 10224) 
s_add_i32 s19, s0, s18 // (op 10224) 


s_ashr_i32 s0, s19, 31 // (op 10224) 
s_cmp_eq_i32 s45, 0 // (op 10224) 
s_mov_b32 s18, scc // (op 10224) 
s_cmp_eq_i32 s18, 0 // (op 10224) 
s_cselect_b32 s50, 0, -1 // (op 10224) 

s_and_b32 s18, s11, 31 // (op 10224) 
s_lshl_b32 s51, 1, s18 // (op 10224) 
s_add_i32 s72, s51, s50 // (op 10224) 


s_and_b32 s50, s0, s72 // (op 10224) 


s_add_i32 s0, s19, s50 // (op 10224) 


s_ashr_i32 s19, s0, s18 // (op 10224) 


s_lshl_b32 s0, s11, 24 // (op 10224) 
s_ashr_i32 s18, s0, 31 // (op 10224) 

s_xor_b32 s0, s19, s18 // (op 10224) 

s_sub_i32 s19, s0, s18 // (op 10224) 


s_cmp_eq_i32 s1, 0 // (op 10224) 
s_mov_b32 s0, scc // (op 10224) 

s_cmp_eq_i32 s0, 0 // (op 10224) 
s_cselect_b32 s1, s19, s49 // (op 10224) 



v_lshl_add_u32 v24, s1, 6, v93 // (op 10224) 


v_mul_lo_u32 v33, v24, s31 // (op 10224) most significant: low of low * high
v_mul_hi_u32 v93, v24, s30 // (op 10224) most significant: high of low * low
v_mul_lo_u32 v32, v24, s30 // (op 10224) least significant: low of low * low
v_add_u32 v33, v33, v93 // (op 10224) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 10224) 
s_lshr_b32 s18, s0, 2 // (op 10224) 

v_lshl_add_u32 v93, s18, 9, v23 // (op 10224) 


v_mov_b32 v35, 0 // (op 10224) convert
v_mov_b32 v34, v93 // (op 10224) convert

v_add_co_u32 v36, s[0:1], v34, v32 // (op 10224) least significant half
v_addc_co_u32 v37, s[0:1], v35, v33, s[0:1] // (op 10224) most significant half

v_lshlrev_b64 v[36:37], 2, v[36:37] // (op 10224) 


v_lshrrev_b64 v[32:33], 3, v[36:37] // (op 10224) 

v_mov_b32 v93, v32 // (op 10224) convert

s_mul_i32 s1, s30, s9 // (op 10226) most significant: low of low * high
s_mul_i32 s18, s31, s8 // (op 10226) most significant: low of high * low
s_mul_i32 s0, s30, s8 // (op 10226) least significant
s_mul_hi_u32 s19, s30, s8 // (op 10226) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10226) most significant: sum
s_add_u32 s1, s1, s19 // (op 10226) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10226) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10226) 

s_mov_b32 s52, s32 // (op 10226) reg expression
s_mov_b32 s53, s33 // (op 10226) reg expression
s_mov_b32 s54, s0 // (op 10226) reg expression
s_mov_b32 s55, 131072 // (op 10226) call()

s_mul_hi_i32 s0, s16, s47 // (op 10288) 
v_lshrrev_b32 v23, 2, v1 // (op 10288) 
s_add_i32 s1, s0, s16 // (op 10288) 

v_mul_lo_u32 v24, v23, 3 // (op 10288) 

s_ashr_i32 s0, s1, 31 // (op 10288) 
v_and_b32 v23, -4, v1 // (op 10288) 
s_cmp_eq_i32 s47, 0 // (op 10288) 
s_mov_b32 s18, scc // (op 10288) 
v_sub_u32 v25, v1, v23 // (op 10288) 

s_cmp_eq_i32 s18, 0 // (op 10288) 
s_cselect_b32 s19, 0, -1 // (op 10288) 

s_and_b32 s18, s13, 31 // (op 10288) 
s_lshl_b32 s49, 1, s18 // (op 10288) 
s_add_i32 s50, s49, s19 // (op 10288) 


s_and_b32 s19, s0, s50 // (op 10288) 


s_add_i32 s0, s1, s19 // (op 10288) 


s_ashr_i32 s1, s0, s18 // (op 10288) 


s_lshl_b32 s0, s13, 24 // (op 10288) 
s_ashr_i32 s18, s0, 31 // (op 10288) 

s_xor_b32 s0, s1, s18 // (op 10288) 

s_sub_i32 s1, s0, s18 // (op 10288) 


s_mul_hi_i32 s0, s1, s14 // (op 10288) 
s_add_i32 s18, s0, s1 // (op 10288) 

s_ashr_i32 s0, s18, 31 // (op 10288) 
s_cmp_eq_i32 s14, 0 // (op 10288) 
s_mov_b32 s19, scc // (op 10288) 
s_cmp_eq_i32 s19, 0 // (op 10288) 
s_cselect_b32 s49, 0, -1 // (op 10288) 

s_and_b32 s19, s15, 31 // (op 10288) 
s_lshl_b32 s50, 1, s19 // (op 10288) 
s_add_i32 s51, s50, s49 // (op 10288) 


s_and_b32 s49, s0, s51 // (op 10288) 


s_add_i32 s0, s18, s49 // (op 10288) 


s_ashr_i32 s18, s0, s19 // (op 10288) 


s_lshl_b32 s0, s15, 24 // (op 10288) 
s_ashr_i32 s19, s0, 31 // (op 10288) 

s_xor_b32 s0, s18, s19 // (op 10288) 

s_sub_i32 s18, s0, s19 // (op 10288) 


s_mul_i32 s0, s1, s48 // (op 10288) 

s_sub_u32 s1, s16, s0 // (op 10288) 

s_mul_i32 s0, s18, s48 // (op 10288) 
s_add_u32 s0, s0, s1 // (op 10288) 


s_mul_hi_i32 s1, s16, s10 // (op 10288) 
s_add_i32 s18, s1, s16 // (op 10288) 

s_ashr_i32 s1, s18, 31 // (op 10288) 
s_cmp_eq_i32 s10, 0 // (op 10288) 
s_mov_b32 s19, scc // (op 10288) 
s_cmp_eq_i32 s19, 0 // (op 10288) 
s_cselect_b32 s49, 0, -1 // (op 10288) 

s_and_b32 s19, s12, 31 // (op 10288) 
s_lshl_b32 s50, 1, s19 // (op 10288) 
s_add_i32 s51, s50, s49 // (op 10288) 


s_and_b32 s49, s1, s51 // (op 10288) 


s_add_i32 s1, s18, s49 // (op 10288) 


s_ashr_i32 s18, s1, s19 // (op 10288) 


s_lshl_b32 s1, s12, 24 // (op 10288) 
s_ashr_i32 s19, s1, 31 // (op 10288) 

s_xor_b32 s1, s18, s19 // (op 10288) 

s_sub_i32 s18, s1, s19 // (op 10288) 


s_add_u32 s50, s8, 64 // (op 10288) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 10288) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 10288) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 10288) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 10288) 

s_mul_i32 s1, s48, s50 // (op 10288) 

s_mul_i32 s19, s44, s1 // (op 10288) 

s_mul_i32 s1, s18, s19 // (op 10288) 

s_sub_u32 s19, s16, s1 // (op 10288) 

s_mov_b32 s1, s19 // (op 10288) 
s_mul_hi_i32 s49, s1, s45 // (op 10288) 
s_add_i32 s50, s49, s1 // (op 10288) 


s_ashr_i32 s1, s50, 31 // (op 10288) 
s_cmp_eq_i32 s45, 0 // (op 10288) 
s_mov_b32 s49, scc // (op 10288) 
s_cmp_eq_i32 s49, 0 // (op 10288) 
s_cselect_b32 s51, 0, -1 // (op 10288) 

s_and_b32 s49, s11, 31 // (op 10288) 
s_lshl_b32 s72, 1, s49 // (op 10288) 
s_add_i32 s73, s72, s51 // (op 10288) 


s_and_b32 s51, s1, s73 // (op 10288) 


s_add_i32 s1, s50, s51 // (op 10288) 


s_ashr_i32 s50, s1, s49 // (op 10288) 


s_lshl_b32 s1, s11, 24 // (op 10288) 
s_ashr_i32 s49, s1, 31 // (op 10288) 

s_xor_b32 s1, s50, s49 // (op 10288) 

s_sub_i32 s50, s1, s49 // (op 10288) 


s_mul_i32 s1, s46, s48 // (op 10288) 
s_sub_i32 s49, s6, s1 // (op 10288) 
s_mul_i32 s51, s50, s49 // (op 10288) 


s_sub_u32 s49, s19, s51 // (op 10288) 


s_add_u32 s19, s49, s1 // (op 10288) 


s_cmp_eq_i32 s18, 0 // (op 10288) 
s_mov_b32 s1, scc // (op 10288) 

s_cmp_eq_i32 s1, 0 // (op 10288) 
s_cselect_b32 s18, s19, s0 // (op 10288) 



v_mov_b32 v23, 192 // (op 10288) 
v_mul_lo_u32 v26, s18, v23 // (op 10288) 

v_add_u32 v26, v26, v24 // (op 10288) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 10288) 
s_lshr_b32 s18, s0, 2 // (op 10288) 

v_lshl_add_u32 v23, s18, 4, v25 // (op 10288) 


v_mul_lo_u32 v24, v26, s34 // (op 10288) 
v_add_u32 v24, v24, v23 // (op 10288) 


s_mul_i32 s1, s34, s29 // (op 10290) most significant: low of low * high
s_mul_i32 s18, s35, s28 // (op 10290) most significant: low of high * low
s_mul_i32 s0, s34, s28 // (op 10290) least significant
s_mul_hi_u32 s19, s34, s28 // (op 10290) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10290) most significant: sum
s_add_u32 s1, s1, s19 // (op 10290) most significant: sum


s_mov_b32 s64, s36 // (op 10290) reg expression
s_mov_b32 s65, s37 // (op 10290) reg expression
s_mov_b32 s66, s0 // (op 10290) reg expression
s_mov_b32 s67, 131072 // (op 10290) call()

s_mul_hi_i32 s0, s16, s47 // (op 10352) 
v_lshrrev_b32 v23, 2, v1 // (op 10352) 
s_add_i32 s1, s0, s16 // (op 10352) 

v_and_b32 v25, -4, v1 // (op 10352) 
s_ashr_i32 s0, s1, 31 // (op 10352) 
v_sub_u32 v26, v1, v25 // (op 10352) 

s_cmp_eq_i32 s47, 0 // (op 10352) 
s_mov_b32 s18, scc // (op 10352) 
s_cmp_eq_i32 s18, 0 // (op 10352) 
s_cselect_b32 s19, 0, -1 // (op 10352) 

s_and_b32 s18, s13, 31 // (op 10352) 
s_lshl_b32 s49, 1, s18 // (op 10352) 
s_add_i32 s50, s49, s19 // (op 10352) 


s_and_b32 s19, s0, s50 // (op 10352) 


s_add_i32 s0, s1, s19 // (op 10352) 


s_ashr_i32 s1, s0, s18 // (op 10352) 


s_lshl_b32 s0, s13, 24 // (op 10352) 
s_ashr_i32 s18, s0, 31 // (op 10352) 

s_xor_b32 s0, s1, s18 // (op 10352) 

s_sub_i32 s1, s0, s18 // (op 10352) 


s_mul_hi_i32 s0, s1, s14 // (op 10352) 
s_add_i32 s18, s0, s1 // (op 10352) 

s_ashr_i32 s0, s18, 31 // (op 10352) 
s_cmp_eq_i32 s14, 0 // (op 10352) 
s_mov_b32 s19, scc // (op 10352) 
s_cmp_eq_i32 s19, 0 // (op 10352) 
s_cselect_b32 s49, 0, -1 // (op 10352) 

s_and_b32 s19, s15, 31 // (op 10352) 
s_lshl_b32 s50, 1, s19 // (op 10352) 
s_add_i32 s51, s50, s49 // (op 10352) 


s_and_b32 s49, s0, s51 // (op 10352) 


s_add_i32 s0, s18, s49 // (op 10352) 


s_ashr_i32 s18, s0, s19 // (op 10352) 


s_lshl_b32 s0, s15, 24 // (op 10352) 
s_ashr_i32 s19, s0, 31 // (op 10352) 

s_xor_b32 s0, s18, s19 // (op 10352) 

s_sub_i32 s18, s0, s19 // (op 10352) 


s_add_u32 s50, s8, 64 // (op 10352) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 10352) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 10352) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 10352) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 10352) 

s_mul_i32 s0, s18, s50 // (op 10352) 

s_sub_i32 s49, s1, s0 // (op 10352) 


s_mul_hi_i32 s0, s16, s10 // (op 10352) 
s_add_i32 s1, s0, s16 // (op 10352) 

s_ashr_i32 s0, s1, 31 // (op 10352) 
s_cmp_eq_i32 s10, 0 // (op 10352) 
s_mov_b32 s18, scc // (op 10352) 
s_cmp_eq_i32 s18, 0 // (op 10352) 
s_cselect_b32 s19, 0, -1 // (op 10352) 

s_and_b32 s18, s12, 31 // (op 10352) 
s_lshl_b32 s72, 1, s18 // (op 10352) 
s_add_i32 s73, s72, s19 // (op 10352) 


s_and_b32 s19, s0, s73 // (op 10352) 


s_add_i32 s0, s1, s19 // (op 10352) 


s_ashr_i32 s1, s0, s18 // (op 10352) 


s_lshl_b32 s0, s12, 24 // (op 10352) 
s_ashr_i32 s18, s0, 31 // (op 10352) 

s_xor_b32 s0, s1, s18 // (op 10352) 

s_sub_i32 s1, s0, s18 // (op 10352) 


s_mul_i32 s0, s48, s50 // (op 10352) 

s_mul_i32 s18, s44, s0 // (op 10352) 

s_mul_i32 s0, s1, s18 // (op 10352) 

s_sub_u32 s18, s16, s0 // (op 10352) 

s_mul_hi_i32 s0, s18, s45 // (op 10352) 
s_add_i32 s19, s0, s18 // (op 10352) 


s_ashr_i32 s0, s19, 31 // (op 10352) 
s_cmp_eq_i32 s45, 0 // (op 10352) 
s_mov_b32 s18, scc // (op 10352) 
s_cmp_eq_i32 s18, 0 // (op 10352) 
s_cselect_b32 s50, 0, -1 // (op 10352) 

s_and_b32 s18, s11, 31 // (op 10352) 
s_lshl_b32 s51, 1, s18 // (op 10352) 
s_add_i32 s72, s51, s50 // (op 10352) 


s_and_b32 s50, s0, s72 // (op 10352) 


s_add_i32 s0, s19, s50 // (op 10352) 


s_ashr_i32 s19, s0, s18 // (op 10352) 


s_lshl_b32 s0, s11, 24 // (op 10352) 
s_ashr_i32 s18, s0, 31 // (op 10352) 

s_xor_b32 s0, s19, s18 // (op 10352) 

s_sub_i32 s19, s0, s18 // (op 10352) 


s_cmp_eq_i32 s1, 0 // (op 10352) 
s_mov_b32 s0, scc // (op 10352) 

s_cmp_eq_i32 s0, 0 // (op 10352) 
s_cselect_b32 s1, s19, s49 // (op 10352) 



v_lshl_add_u32 v25, s1, 6, v23 // (op 10352) 


v_mul_lo_u32 v23, v25, s38 // (op 10352) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 10352) 
s_lshr_b32 s18, s0, 2 // (op 10352) 

v_lshl_add_u32 v25, s18, 4, v26 // (op 10352) 


v_add_u32 v26, v25, v23 // (op 10352) 


s_mul_i32 s1, s38, s9 // (op 10354) most significant: low of low * high
s_mul_i32 s18, s39, s8 // (op 10354) most significant: low of high * low
s_mul_i32 s0, s38, s8 // (op 10354) least significant
s_mul_hi_u32 s19, s38, s8 // (op 10354) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10354) most significant: sum
s_add_u32 s1, s1, s19 // (op 10354) most significant: sum


s_mov_b32 s68, s40 // (op 10354) reg expression
s_mov_b32 s69, s41 // (op 10354) reg expression
s_mov_b32 s70, s0 // (op 10354) reg expression
s_mov_b32 s71, 131072 // (op 10354) call()

v_and_b32 v23, -4, v1 // (op 10450) 
v_sub_u32 v25, v1, v23 // (op 10450) 

v_lshlrev_b32 v23, 5, v25 // (op 10450) 

v_lshrrev_b32 v25, 2, v1 // (op 10450) 
v_mul_lo_u32 v27, v25, 3 // (op 10450) 

v_lshl_add_u32 v25, v27, 7, v23 // (op 10450) 


v_lshlrev_b32 v23, 2, v25 // (op 10450) 

v_lshrrev_b32 v25, 3, v23 // (op 10450) 

v_and_b32 v23, -4, v1 // (op 10508) 
v_sub_u32 v27, v1, v23 // (op 10508) 

v_lshlrev_b32 v23, 5, v27 // (op 10508) 

v_lshrrev_b32 v27, 2, v1 // (op 10508) 
v_lshl_add_u32 v28, v27, 7, v23 // (op 10508) 


v_lshlrev_b32 v23, 2, v28 // (op 10508) 

v_lshrrev_b32 v27, 3, v23 // (op 10508) 

v_and_b32 v23, -4, v1 // (op 10566) 
v_sub_u32 v28, v1, v23 // (op 10566) 

v_lshrrev_b32 v23, 2, v1 // (op 10566) 
v_mul_lo_u32 v29, v23, 3 // (op 10566) 

v_lshl_add_u32 v23, v29, 2, v28 // (op 10566) 


v_and_b32 v28, -4, v1 // (op 10624) 
v_sub_u32 v29, v1, v28 // (op 10624) 

v_and_b32 v28, 4294967292, v1 // (op 10624) 
v_add_u32 v30, v28, v29 // (op 10624) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 3474) 
s_and_b32 s18, s0, 4294967292 // (op 3474) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 3473) 
s_mov_b32 s19, s0 // (op 3473) convert

s_cmp_lt_u32 s18, s19 // (op 3473) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopBottom_KLoopTail_3473 // (op 3473) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopBottom_KLoopTail_3473 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopTop_KLoopTail_3473:

s_add_u32 s18, s18, 1 // (op 3475) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 180) 
v_lshrrev_b64 v[28:29], 3, s[0:1] // (op 180) 


v_mov_b32 v22, v73 // (op 180) 
v_mov_b32 v31, v22 // (op 180) 
buffer_load_dwordx4 v[32:35], v31, s[20:23], 0 offen // (op 180) Load value
v_add_u32 v31, v31, v28 // (op 180) 
buffer_load_dwordx4 v[36:39], v31, s[20:23], 0 offen // (op 180) Load value
v_add_u32 v31, v31, v28 // (op 180) 
buffer_load_dwordx4 v[40:43], v31, s[20:23], 0 offen // (op 180) Load value



v_add_u32 v73, 64, v73 // (op 10136) 


s_barrier  // (op 11157) 
v_mov_b32 v22, v93 // (op 194) 




s_waitcnt vmcnt(2)
ds_write_b128 v25, v[32:35] offset:4352 // (op 181) Store local data 
buffer_load_dwordx4 v[44:47], v22, s[52:55], 0 offen // (op 194) Load value

v_add_u32 v93, 64, v93 // (op 10216) 
s_waitcnt vmcnt(2)
ds_write_b128 v25, v[36:39] offset:4416 // (op 181) Store local data 

v_mov_b64 v[28:29], s[34:35] // (op 195) convert

v_mov_b32 v22, v24 // (op 195) 
v_mov_b32 v31, v22 // (op 195) 
buffer_load_ubyte v48, v31, s[64:67], 0 offen offset: 0 // (op 195) Load value
v_add_u32 v31, v31, v28 // (op 195) 
buffer_load_ubyte v49, v31, s[64:67], 0 offen offset: 0 // (op 195) Load value
v_add_u32 v31, v31, v28 // (op 195) 
s_waitcnt vmcnt(3)
ds_write_b128 v25, v[40:43] offset:4480 // (op 181) Store local data 
buffer_load_ubyte v50, v31, s[64:67], 0 offen offset: 0 // (op 195) Load value




v_add_u32 v24, 4, v24 // (op 10280) 


s_barrier  // (op 11170) 
v_mov_b32 v22, v26 // (op 196) 




buffer_load_ubyte v28, v22, s[68:71], 0 offen offset: 0 // (op 196) Load value

v_add_u32 v26, 4, v26 // (op 10344) 
s_waitcnt vmcnt(4)
ds_write_b128 v27, v[44:47] // (op 191) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 11160) 



s_barrier  // (op 11184) 
ds_read_b128 v[44:47], v3 offset:4352 // (op 183) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 11174) 
s_waitcnt vmcnt(3)
ds_write_b8 v23, v48 offset:16640 // (op 192) Store local data 



ds_read_b128 v[40:43], v4 // (op 185) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v23, v49 offset:16644 // (op 192) Store local data 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v50 offset:16648 // (op 192) Store local data 

s_barrier  // (op 11198) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 11188) 






ds_read_u8 v22, v0 offset:16640 // (op 187) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v30, v28 offset:4096 // (op 193) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 11202) 



ds_read_u8 v28, v5 offset:4096 // (op 189) Load local data 



ds_read_b128 v[36:39], v4 offset:2048 // (op 500) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:47], v[40:43], a[0:3], v22, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 190) 



ds_read_u8 v29, v5 offset:4224 // (op 504) Load local data 



ds_read_b128 v[32:35], v3 offset:6400 // (op 864) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:47], v[36:39], a[4:7], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 505) 





ds_read_u8 v22, v0 offset:16768 // (op 867) Load local data 



ds_read_b128 v[44:47], v3 offset:8448 // (op 1510) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:35], v[40:43], a[8:11], v22, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 869) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:35], v[36:39], a[12:15], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 900) 
ds_read_u8 v31, v0 offset:16896 // (op 1513) Load local data 





ds_read_b128 v[32:35], v3 offset:10496 // (op 2156) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[44:47], v[40:43], a[16:19], v31, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1515) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[44:47], v[36:39], a[20:23], v31, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1546) 
ds_read_u8 v22, v0 offset:17024 // (op 2159) Load local data 





ds_read_b128 v[44:47], v3 offset:12544 // (op 2802) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[32:35], v[40:43], a[24:27], v22, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2161) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[32:35], v[36:39], a[28:31], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2192) 
ds_read_u8 v31, v0 offset:17152 // (op 2805) Load local data 





ds_read_b128 v[32:35], v3 offset:14592 // (op 3448) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[44:47], v[40:43], a[32:35], v31, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2807) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[44:47], v[36:39], a[36:39], v31, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2838) 
ds_read_u8 v22, v0 offset:17280 // (op 3451) Load local data 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[32:35], v[40:43], a[40:43], v22, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3453) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[32:35], v[36:39], a[44:47], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3484) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 3473) 
s_mov_b32 s19, s0 // (op 3473) convert

s_cmp_lt_u32 s18, s19 // (op 3473) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopTop_KLoopTail_3473 // (op 3473) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopTop_KLoopTail_3473 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ForLoopBottom_KLoopTail_3473:























s_barrier  // (op 11424) 





























v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ConditionalFalse_0_beta_135
v_accvgpr_read v14, a0 // (op 50) 
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_mul_f32 v10, v21, v14 // (op 50) 
v_mul_f32 v11, v21, v15 // (op 50) 
v_mul_f32 v12, v21, v16 // (op 50) 
v_mul_f32 v13, v21, v17 // (op 50) 

v_lshrrev_b32 v14, 16, v10 // (op 56) 
v_lshrrev_b32 v15, 16, v11 // (op 56) 
v_lshrrev_b32 v16, 16, v12 // (op 56) 
v_lshrrev_b32 v17, 16, v13 // (op 56) 


v_pack_b32_f16 v12, v14, v15 // (op 58) 
v_pack_b32_f16 v13, v16, v17 // (op 58) 

v_mov_b32 v0, v20 // (op 58) reg expression

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v14, a4 // (op 4461) 
v_accvgpr_read v15, a5 // (op 4461) 
v_accvgpr_read v16, a6 // (op 4461) 
v_accvgpr_read v17, a7 // (op 4461) 
v_mul_f32 v10, v21, v14 // (op 4461) 
v_mul_f32 v11, v21, v15 // (op 4461) 
v_mul_f32 v12, v21, v16 // (op 4461) 
v_mul_f32 v13, v21, v17 // (op 4461) 

v_lshrrev_b32 v14, 16, v10 // (op 4463) 
v_lshrrev_b32 v15, 16, v11 // (op 4463) 
v_lshrrev_b32 v16, 16, v12 // (op 4463) 
v_lshrrev_b32 v17, 16, v13 // (op 4463) 


v_pack_b32_f16 v12, v14, v15 // (op 4464) 
v_pack_b32_f16 v13, v16, v17 // (op 4464) 

s_lshl_b32 s0, s4, 6 // (op 4464) 
v_add_u32 v0, v20, s0 // (op 4464) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4464) Store value



v_accvgpr_read v14, a8 // (op 4487) 
v_accvgpr_read v15, a9 // (op 4487) 
v_accvgpr_read v16, a10 // (op 4487) 
v_accvgpr_read v17, a11 // (op 4487) 
v_mul_f32 v10, v21, v14 // (op 4487) 
v_mul_f32 v11, v21, v15 // (op 4487) 
v_mul_f32 v12, v21, v16 // (op 4487) 
v_mul_f32 v13, v21, v17 // (op 4487) 

v_lshrrev_b32 v14, 16, v10 // (op 4489) 
v_lshrrev_b32 v15, 16, v11 // (op 4489) 
v_lshrrev_b32 v16, 16, v12 // (op 4489) 
v_lshrrev_b32 v17, 16, v13 // (op 4489) 


v_pack_b32_f16 v12, v14, v15 // (op 4491) 
v_pack_b32_f16 v13, v16, v17 // (op 4491) 

v_add_u32 v0, 64, v20 // (op 4491) 

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4491) Store value



v_accvgpr_read v14, a12 // (op 4497) 
v_accvgpr_read v15, a13 // (op 4497) 
v_accvgpr_read v16, a14 // (op 4497) 
v_accvgpr_read v17, a15 // (op 4497) 
v_mul_f32 v10, v21, v14 // (op 4497) 
v_mul_f32 v11, v21, v15 // (op 4497) 
v_mul_f32 v12, v21, v16 // (op 4497) 
v_mul_f32 v13, v21, v17 // (op 4497) 

v_lshrrev_b32 v14, 16, v10 // (op 4499) 
v_lshrrev_b32 v15, 16, v11 // (op 4499) 
v_lshrrev_b32 v16, 16, v12 // (op 4499) 
v_lshrrev_b32 v17, 16, v13 // (op 4499) 


v_pack_b32_f16 v12, v14, v15 // (op 4493) 
v_pack_b32_f16 v13, v16, v17 // (op 4493) 

s_lshl_b32 s0, s4, 6 // (op 4493) 
s_add_u32 s0, s0, 64 // (op 4493) 
v_add_u32 v0, v20, s0 // (op 4493) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4493) Store value



v_accvgpr_read v14, a16 // (op 4527) 
v_accvgpr_read v15, a17 // (op 4527) 
v_accvgpr_read v16, a18 // (op 4527) 
v_accvgpr_read v17, a19 // (op 4527) 
v_mul_f32 v10, v21, v14 // (op 4527) 
v_mul_f32 v11, v21, v15 // (op 4527) 
v_mul_f32 v12, v21, v16 // (op 4527) 
v_mul_f32 v13, v21, v17 // (op 4527) 

v_lshrrev_b32 v14, 16, v10 // (op 4529) 
v_lshrrev_b32 v15, 16, v11 // (op 4529) 
v_lshrrev_b32 v16, 16, v12 // (op 4529) 
v_lshrrev_b32 v17, 16, v13 // (op 4529) 


v_pack_b32_f16 v12, v14, v15 // (op 4531) 
v_pack_b32_f16 v13, v16, v17 // (op 4531) 

v_add_u32 v0, 128, v20 // (op 4531) 

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4531) Store value



v_accvgpr_read v14, a20 // (op 4537) 
v_accvgpr_read v15, a21 // (op 4537) 
v_accvgpr_read v16, a22 // (op 4537) 
v_accvgpr_read v17, a23 // (op 4537) 
v_mul_f32 v10, v21, v14 // (op 4537) 
v_mul_f32 v11, v21, v15 // (op 4537) 
v_mul_f32 v12, v21, v16 // (op 4537) 
v_mul_f32 v13, v21, v17 // (op 4537) 

v_lshrrev_b32 v14, 16, v10 // (op 4539) 
v_lshrrev_b32 v15, 16, v11 // (op 4539) 
v_lshrrev_b32 v16, 16, v12 // (op 4539) 
v_lshrrev_b32 v17, 16, v13 // (op 4539) 


v_pack_b32_f16 v12, v14, v15 // (op 4533) 
v_pack_b32_f16 v13, v16, v17 // (op 4533) 

s_lshl_b32 s0, s4, 6 // (op 4533) 
s_add_u32 s0, s0, 128 // (op 4533) 
v_add_u32 v0, v20, s0 // (op 4533) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4533) Store value



v_accvgpr_read v14, a24 // (op 4567) 
v_accvgpr_read v15, a25 // (op 4567) 
v_accvgpr_read v16, a26 // (op 4567) 
v_accvgpr_read v17, a27 // (op 4567) 
v_mul_f32 v10, v21, v14 // (op 4567) 
v_mul_f32 v11, v21, v15 // (op 4567) 
v_mul_f32 v12, v21, v16 // (op 4567) 
v_mul_f32 v13, v21, v17 // (op 4567) 

v_lshrrev_b32 v14, 16, v10 // (op 4569) 
v_lshrrev_b32 v15, 16, v11 // (op 4569) 
v_lshrrev_b32 v16, 16, v12 // (op 4569) 
v_lshrrev_b32 v17, 16, v13 // (op 4569) 


v_pack_b32_f16 v12, v14, v15 // (op 4571) 
v_pack_b32_f16 v13, v16, v17 // (op 4571) 

v_add_u32 v0, 192, v20 // (op 4571) 

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4571) Store value



v_accvgpr_read v14, a28 // (op 4577) 
v_accvgpr_read v15, a29 // (op 4577) 
v_accvgpr_read v16, a30 // (op 4577) 
v_accvgpr_read v17, a31 // (op 4577) 
v_mul_f32 v10, v21, v14 // (op 4577) 
v_mul_f32 v11, v21, v15 // (op 4577) 
v_mul_f32 v12, v21, v16 // (op 4577) 
v_mul_f32 v13, v21, v17 // (op 4577) 

v_lshrrev_b32 v14, 16, v10 // (op 4579) 
v_lshrrev_b32 v15, 16, v11 // (op 4579) 
v_lshrrev_b32 v16, 16, v12 // (op 4579) 
v_lshrrev_b32 v17, 16, v13 // (op 4579) 


v_pack_b32_f16 v12, v14, v15 // (op 4573) 
v_pack_b32_f16 v13, v16, v17 // (op 4573) 

s_lshl_b32 s0, s4, 6 // (op 4573) 
s_add_u32 s0, s0, 192 // (op 4573) 
v_add_u32 v0, v20, s0 // (op 4573) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4573) Store value



v_accvgpr_read v14, a32 // (op 4607) 
v_accvgpr_read v15, a33 // (op 4607) 
v_accvgpr_read v16, a34 // (op 4607) 
v_accvgpr_read v17, a35 // (op 4607) 
v_mul_f32 v10, v21, v14 // (op 4607) 
v_mul_f32 v11, v21, v15 // (op 4607) 
v_mul_f32 v12, v21, v16 // (op 4607) 
v_mul_f32 v13, v21, v17 // (op 4607) 

v_lshrrev_b32 v14, 16, v10 // (op 4609) 
v_lshrrev_b32 v15, 16, v11 // (op 4609) 
v_lshrrev_b32 v16, 16, v12 // (op 4609) 
v_lshrrev_b32 v17, 16, v13 // (op 4609) 


v_pack_b32_f16 v12, v14, v15 // (op 4611) 
v_pack_b32_f16 v13, v16, v17 // (op 4611) 

v_add_u32 v0, 256, v20 // (op 4611) 

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4611) Store value



v_accvgpr_read v14, a36 // (op 4617) 
v_accvgpr_read v15, a37 // (op 4617) 
v_accvgpr_read v16, a38 // (op 4617) 
v_accvgpr_read v17, a39 // (op 4617) 
v_mul_f32 v10, v21, v14 // (op 4617) 
v_mul_f32 v11, v21, v15 // (op 4617) 
v_mul_f32 v12, v21, v16 // (op 4617) 
v_mul_f32 v13, v21, v17 // (op 4617) 

v_lshrrev_b32 v14, 16, v10 // (op 4619) 
v_lshrrev_b32 v15, 16, v11 // (op 4619) 
v_lshrrev_b32 v16, 16, v12 // (op 4619) 
v_lshrrev_b32 v17, 16, v13 // (op 4619) 


v_pack_b32_f16 v12, v14, v15 // (op 4613) 
v_pack_b32_f16 v13, v16, v17 // (op 4613) 

s_lshl_b32 s0, s4, 6 // (op 4613) 
s_add_u32 s0, s0, 256 // (op 4613) 
v_add_u32 v0, v20, s0 // (op 4613) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4613) Store value



v_accvgpr_read v14, a40 // (op 4647) 
v_accvgpr_read v15, a41 // (op 4647) 
v_accvgpr_read v16, a42 // (op 4647) 
v_accvgpr_read v17, a43 // (op 4647) 
v_mul_f32 v10, v21, v14 // (op 4647) 
v_mul_f32 v11, v21, v15 // (op 4647) 
v_mul_f32 v12, v21, v16 // (op 4647) 
v_mul_f32 v13, v21, v17 // (op 4647) 

v_lshrrev_b32 v14, 16, v10 // (op 4649) 
v_lshrrev_b32 v15, 16, v11 // (op 4649) 
v_lshrrev_b32 v16, 16, v12 // (op 4649) 
v_lshrrev_b32 v17, 16, v13 // (op 4649) 


v_pack_b32_f16 v12, v14, v15 // (op 4651) 
v_pack_b32_f16 v13, v16, v17 // (op 4651) 

v_add_u32 v0, 320, v20 // (op 4651) 

buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4651) Store value



v_accvgpr_read v14, a44 // (op 4657) 
v_accvgpr_read v15, a45 // (op 4657) 
v_accvgpr_read v16, a46 // (op 4657) 
v_accvgpr_read v17, a47 // (op 4657) 
v_mul_f32 v10, v21, v14 // (op 4657) 
v_mul_f32 v11, v21, v15 // (op 4657) 
v_mul_f32 v12, v21, v16 // (op 4657) 
v_mul_f32 v13, v21, v17 // (op 4657) 

v_lshrrev_b32 v14, 16, v10 // (op 4659) 
v_lshrrev_b32 v15, 16, v11 // (op 4659) 
v_lshrrev_b32 v16, 16, v12 // (op 4659) 
v_lshrrev_b32 v17, 16, v13 // (op 4659) 


v_pack_b32_f16 v12, v14, v15 // (op 4653) 
v_pack_b32_f16 v13, v16, v17 // (op 4653) 

s_lshl_b32 s0, s4, 6 // (op 4653) 
s_add_u32 s0, s0, 320 // (op 4653) 
v_add_u32 v0, v20, s0 // (op 4653) 


buffer_store_dwordx2 v[12:13], v0, s[60:63], 0 offen // (op 4653) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ConditionalBottom_0_beta_135
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ConditionalFalse_0_beta_135:


v_add_u32 v0, 320, v19 // (op 4955) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4955) Load value

v_accvgpr_read v12, a40 // (op 4946) 
v_accvgpr_read v13, a41 // (op 4946) 
v_accvgpr_read v14, a42 // (op 4946) 
v_accvgpr_read v15, a43 // (op 4946) 
v_mul_f32 v8, v21, v12 // (op 4946) 
v_mul_f32 v9, v21, v13 // (op 4946) 
v_mul_f32 v10, v21, v14 // (op 4946) 
v_mul_f32 v11, v21, v15 // (op 4946) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4954) 
v_lshrrev_b32 v7, 16, v16 // (op 4954) 
v_lshlrev_b32 v6, 16, v6 // (op 4954) 
v_lshlrev_b32 v7, 16, v7 // (op 4954) 
v_mul_f32 v12, v18, v6 // (op 4954) 
v_mul_f32 v13, v18, v7 // (op 4954) 
v_and_b32 v6, 65535, v17 // (op 4954) 
v_lshrrev_b32 v7, 16, v17 // (op 4954) 
v_lshlrev_b32 v6, 16, v6 // (op 4954) 
v_lshlrev_b32 v7, 16, v7 // (op 4954) 
v_mul_f32 v14, v18, v6 // (op 4954) 
v_mul_f32 v15, v18, v7 // (op 4954) 


v_add_f32 v4, v12, v8 // (op 4947) 
v_add_f32 v5, v13, v9 // (op 4947) 
v_add_f32 v6, v14, v10 // (op 4947) 
v_add_f32 v7, v15, v11 // (op 4947) 


v_lshrrev_b32 v8, 16, v4 // (op 4948) 
v_lshrrev_b32 v9, 16, v5 // (op 4948) 
v_lshrrev_b32 v10, 16, v6 // (op 4948) 
v_lshrrev_b32 v11, 16, v7 // (op 4948) 


v_pack_b32_f16 v6, v8, v9 // (op 4950) 
v_pack_b32_f16 v7, v10, v11 // (op 4950) 

v_add_u32 v0, 320, v20 // (op 4950) 

buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4950) Store value





s_lshl_b32 s0, s2, 6 // (op 4960) 
s_add_u32 s0, s0, 320 // (op 4960) 
v_add_u32 v0, v19, s0 // (op 4960) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4960) Load value

v_accvgpr_read v12, a44 // (op 4962) 
v_accvgpr_read v13, a45 // (op 4962) 
v_accvgpr_read v14, a46 // (op 4962) 
v_accvgpr_read v15, a47 // (op 4962) 
v_mul_f32 v8, v21, v12 // (op 4962) 
v_mul_f32 v9, v21, v13 // (op 4962) 
v_mul_f32 v10, v21, v14 // (op 4962) 
v_mul_f32 v11, v21, v15 // (op 4962) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4957) 
v_lshrrev_b32 v7, 16, v16 // (op 4957) 
v_lshlrev_b32 v6, 16, v6 // (op 4957) 
v_lshlrev_b32 v7, 16, v7 // (op 4957) 
v_mul_f32 v12, v18, v6 // (op 4957) 
v_mul_f32 v13, v18, v7 // (op 4957) 
v_and_b32 v6, 65535, v17 // (op 4957) 
v_lshrrev_b32 v7, 16, v17 // (op 4957) 
v_lshlrev_b32 v6, 16, v6 // (op 4957) 
v_lshlrev_b32 v7, 16, v7 // (op 4957) 
v_mul_f32 v14, v18, v6 // (op 4957) 
v_mul_f32 v15, v18, v7 // (op 4957) 


v_add_f32 v4, v12, v8 // (op 4958) 
v_add_f32 v5, v13, v9 // (op 4958) 
v_add_f32 v6, v14, v10 // (op 4958) 
v_add_f32 v7, v15, v11 // (op 4958) 


v_lshrrev_b32 v8, 16, v4 // (op 4959) 
v_lshrrev_b32 v9, 16, v5 // (op 4959) 
v_lshrrev_b32 v10, 16, v6 // (op 4959) 
v_lshrrev_b32 v11, 16, v7 // (op 4959) 


v_pack_b32_f16 v6, v8, v9 // (op 4952) 
v_pack_b32_f16 v7, v10, v11 // (op 4952) 

s_lshl_b32 s0, s4, 6 // (op 4952) 
s_add_u32 s0, s0, 320 // (op 4952) 
v_add_u32 v0, v20, s0 // (op 4952) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4952) Store value





v_add_u32 v0, 256, v19 // (op 4910) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4910) Load value

v_accvgpr_read v12, a32 // (op 4901) 
v_accvgpr_read v13, a33 // (op 4901) 
v_accvgpr_read v14, a34 // (op 4901) 
v_accvgpr_read v15, a35 // (op 4901) 
v_mul_f32 v8, v21, v12 // (op 4901) 
v_mul_f32 v9, v21, v13 // (op 4901) 
v_mul_f32 v10, v21, v14 // (op 4901) 
v_mul_f32 v11, v21, v15 // (op 4901) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4909) 
v_lshrrev_b32 v7, 16, v16 // (op 4909) 
v_lshlrev_b32 v6, 16, v6 // (op 4909) 
v_lshlrev_b32 v7, 16, v7 // (op 4909) 
v_mul_f32 v12, v18, v6 // (op 4909) 
v_mul_f32 v13, v18, v7 // (op 4909) 
v_and_b32 v6, 65535, v17 // (op 4909) 
v_lshrrev_b32 v7, 16, v17 // (op 4909) 
v_lshlrev_b32 v6, 16, v6 // (op 4909) 
v_lshlrev_b32 v7, 16, v7 // (op 4909) 
v_mul_f32 v14, v18, v6 // (op 4909) 
v_mul_f32 v15, v18, v7 // (op 4909) 


v_add_f32 v4, v12, v8 // (op 4902) 
v_add_f32 v5, v13, v9 // (op 4902) 
v_add_f32 v6, v14, v10 // (op 4902) 
v_add_f32 v7, v15, v11 // (op 4902) 


v_lshrrev_b32 v8, 16, v4 // (op 4903) 
v_lshrrev_b32 v9, 16, v5 // (op 4903) 
v_lshrrev_b32 v10, 16, v6 // (op 4903) 
v_lshrrev_b32 v11, 16, v7 // (op 4903) 


v_pack_b32_f16 v6, v8, v9 // (op 4905) 
v_pack_b32_f16 v7, v10, v11 // (op 4905) 

v_add_u32 v0, 256, v20 // (op 4905) 

buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4905) Store value





s_lshl_b32 s0, s2, 6 // (op 4915) 
s_add_u32 s0, s0, 256 // (op 4915) 
v_add_u32 v0, v19, s0 // (op 4915) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4915) Load value

v_accvgpr_read v12, a36 // (op 4917) 
v_accvgpr_read v13, a37 // (op 4917) 
v_accvgpr_read v14, a38 // (op 4917) 
v_accvgpr_read v15, a39 // (op 4917) 
v_mul_f32 v8, v21, v12 // (op 4917) 
v_mul_f32 v9, v21, v13 // (op 4917) 
v_mul_f32 v10, v21, v14 // (op 4917) 
v_mul_f32 v11, v21, v15 // (op 4917) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4912) 
v_lshrrev_b32 v7, 16, v16 // (op 4912) 
v_lshlrev_b32 v6, 16, v6 // (op 4912) 
v_lshlrev_b32 v7, 16, v7 // (op 4912) 
v_mul_f32 v12, v18, v6 // (op 4912) 
v_mul_f32 v13, v18, v7 // (op 4912) 
v_and_b32 v6, 65535, v17 // (op 4912) 
v_lshrrev_b32 v7, 16, v17 // (op 4912) 
v_lshlrev_b32 v6, 16, v6 // (op 4912) 
v_lshlrev_b32 v7, 16, v7 // (op 4912) 
v_mul_f32 v14, v18, v6 // (op 4912) 
v_mul_f32 v15, v18, v7 // (op 4912) 


v_add_f32 v4, v12, v8 // (op 4913) 
v_add_f32 v5, v13, v9 // (op 4913) 
v_add_f32 v6, v14, v10 // (op 4913) 
v_add_f32 v7, v15, v11 // (op 4913) 


v_lshrrev_b32 v8, 16, v4 // (op 4914) 
v_lshrrev_b32 v9, 16, v5 // (op 4914) 
v_lshrrev_b32 v10, 16, v6 // (op 4914) 
v_lshrrev_b32 v11, 16, v7 // (op 4914) 


v_pack_b32_f16 v6, v8, v9 // (op 4907) 
v_pack_b32_f16 v7, v10, v11 // (op 4907) 

s_lshl_b32 s0, s4, 6 // (op 4907) 
s_add_u32 s0, s0, 256 // (op 4907) 
v_add_u32 v0, v20, s0 // (op 4907) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4907) Store value





v_add_u32 v0, 192, v19 // (op 4865) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4865) Load value

v_accvgpr_read v12, a24 // (op 4856) 
v_accvgpr_read v13, a25 // (op 4856) 
v_accvgpr_read v14, a26 // (op 4856) 
v_accvgpr_read v15, a27 // (op 4856) 
v_mul_f32 v8, v21, v12 // (op 4856) 
v_mul_f32 v9, v21, v13 // (op 4856) 
v_mul_f32 v10, v21, v14 // (op 4856) 
v_mul_f32 v11, v21, v15 // (op 4856) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4864) 
v_lshrrev_b32 v7, 16, v16 // (op 4864) 
v_lshlrev_b32 v6, 16, v6 // (op 4864) 
v_lshlrev_b32 v7, 16, v7 // (op 4864) 
v_mul_f32 v12, v18, v6 // (op 4864) 
v_mul_f32 v13, v18, v7 // (op 4864) 
v_and_b32 v6, 65535, v17 // (op 4864) 
v_lshrrev_b32 v7, 16, v17 // (op 4864) 
v_lshlrev_b32 v6, 16, v6 // (op 4864) 
v_lshlrev_b32 v7, 16, v7 // (op 4864) 
v_mul_f32 v14, v18, v6 // (op 4864) 
v_mul_f32 v15, v18, v7 // (op 4864) 


v_add_f32 v4, v12, v8 // (op 4857) 
v_add_f32 v5, v13, v9 // (op 4857) 
v_add_f32 v6, v14, v10 // (op 4857) 
v_add_f32 v7, v15, v11 // (op 4857) 


v_lshrrev_b32 v8, 16, v4 // (op 4858) 
v_lshrrev_b32 v9, 16, v5 // (op 4858) 
v_lshrrev_b32 v10, 16, v6 // (op 4858) 
v_lshrrev_b32 v11, 16, v7 // (op 4858) 


v_pack_b32_f16 v6, v8, v9 // (op 4860) 
v_pack_b32_f16 v7, v10, v11 // (op 4860) 

v_add_u32 v0, 192, v20 // (op 4860) 

buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4860) Store value





s_lshl_b32 s0, s2, 6 // (op 4870) 
s_add_u32 s0, s0, 192 // (op 4870) 
v_add_u32 v0, v19, s0 // (op 4870) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4870) Load value

v_accvgpr_read v12, a28 // (op 4872) 
v_accvgpr_read v13, a29 // (op 4872) 
v_accvgpr_read v14, a30 // (op 4872) 
v_accvgpr_read v15, a31 // (op 4872) 
v_mul_f32 v8, v21, v12 // (op 4872) 
v_mul_f32 v9, v21, v13 // (op 4872) 
v_mul_f32 v10, v21, v14 // (op 4872) 
v_mul_f32 v11, v21, v15 // (op 4872) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4867) 
v_lshrrev_b32 v7, 16, v16 // (op 4867) 
v_lshlrev_b32 v6, 16, v6 // (op 4867) 
v_lshlrev_b32 v7, 16, v7 // (op 4867) 
v_mul_f32 v12, v18, v6 // (op 4867) 
v_mul_f32 v13, v18, v7 // (op 4867) 
v_and_b32 v6, 65535, v17 // (op 4867) 
v_lshrrev_b32 v7, 16, v17 // (op 4867) 
v_lshlrev_b32 v6, 16, v6 // (op 4867) 
v_lshlrev_b32 v7, 16, v7 // (op 4867) 
v_mul_f32 v14, v18, v6 // (op 4867) 
v_mul_f32 v15, v18, v7 // (op 4867) 


v_add_f32 v4, v12, v8 // (op 4868) 
v_add_f32 v5, v13, v9 // (op 4868) 
v_add_f32 v6, v14, v10 // (op 4868) 
v_add_f32 v7, v15, v11 // (op 4868) 


v_lshrrev_b32 v8, 16, v4 // (op 4869) 
v_lshrrev_b32 v9, 16, v5 // (op 4869) 
v_lshrrev_b32 v10, 16, v6 // (op 4869) 
v_lshrrev_b32 v11, 16, v7 // (op 4869) 


v_pack_b32_f16 v6, v8, v9 // (op 4862) 
v_pack_b32_f16 v7, v10, v11 // (op 4862) 

s_lshl_b32 s0, s4, 6 // (op 4862) 
s_add_u32 s0, s0, 192 // (op 4862) 
v_add_u32 v0, v20, s0 // (op 4862) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4862) Store value





v_add_u32 v0, 64, v19 // (op 4775) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4775) Load value

v_accvgpr_read v12, a8 // (op 4766) 
v_accvgpr_read v13, a9 // (op 4766) 
v_accvgpr_read v14, a10 // (op 4766) 
v_accvgpr_read v15, a11 // (op 4766) 
v_mul_f32 v8, v21, v12 // (op 4766) 
v_mul_f32 v9, v21, v13 // (op 4766) 
v_mul_f32 v10, v21, v14 // (op 4766) 
v_mul_f32 v11, v21, v15 // (op 4766) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4774) 
v_lshrrev_b32 v7, 16, v16 // (op 4774) 
v_lshlrev_b32 v6, 16, v6 // (op 4774) 
v_lshlrev_b32 v7, 16, v7 // (op 4774) 
v_mul_f32 v12, v18, v6 // (op 4774) 
v_mul_f32 v13, v18, v7 // (op 4774) 
v_and_b32 v6, 65535, v17 // (op 4774) 
v_lshrrev_b32 v7, 16, v17 // (op 4774) 
v_lshlrev_b32 v6, 16, v6 // (op 4774) 
v_lshlrev_b32 v7, 16, v7 // (op 4774) 
v_mul_f32 v14, v18, v6 // (op 4774) 
v_mul_f32 v15, v18, v7 // (op 4774) 


v_add_f32 v4, v12, v8 // (op 4767) 
v_add_f32 v5, v13, v9 // (op 4767) 
v_add_f32 v6, v14, v10 // (op 4767) 
v_add_f32 v7, v15, v11 // (op 4767) 


v_lshrrev_b32 v8, 16, v4 // (op 4768) 
v_lshrrev_b32 v9, 16, v5 // (op 4768) 
v_lshrrev_b32 v10, 16, v6 // (op 4768) 
v_lshrrev_b32 v11, 16, v7 // (op 4768) 


v_pack_b32_f16 v6, v8, v9 // (op 4770) 
v_pack_b32_f16 v7, v10, v11 // (op 4770) 

v_add_u32 v0, 64, v20 // (op 4770) 

buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4770) Store value





s_lshl_b32 s0, s2, 6 // (op 4780) 
s_add_u32 s0, s0, 64 // (op 4780) 
v_add_u32 v0, v19, s0 // (op 4780) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4780) Load value

v_accvgpr_read v12, a12 // (op 4782) 
v_accvgpr_read v13, a13 // (op 4782) 
v_accvgpr_read v14, a14 // (op 4782) 
v_accvgpr_read v15, a15 // (op 4782) 
v_mul_f32 v8, v21, v12 // (op 4782) 
v_mul_f32 v9, v21, v13 // (op 4782) 
v_mul_f32 v10, v21, v14 // (op 4782) 
v_mul_f32 v11, v21, v15 // (op 4782) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4777) 
v_lshrrev_b32 v7, 16, v16 // (op 4777) 
v_lshlrev_b32 v6, 16, v6 // (op 4777) 
v_lshlrev_b32 v7, 16, v7 // (op 4777) 
v_mul_f32 v12, v18, v6 // (op 4777) 
v_mul_f32 v13, v18, v7 // (op 4777) 
v_and_b32 v6, 65535, v17 // (op 4777) 
v_lshrrev_b32 v7, 16, v17 // (op 4777) 
v_lshlrev_b32 v6, 16, v6 // (op 4777) 
v_lshlrev_b32 v7, 16, v7 // (op 4777) 
v_mul_f32 v14, v18, v6 // (op 4777) 
v_mul_f32 v15, v18, v7 // (op 4777) 


v_add_f32 v4, v12, v8 // (op 4778) 
v_add_f32 v5, v13, v9 // (op 4778) 
v_add_f32 v6, v14, v10 // (op 4778) 
v_add_f32 v7, v15, v11 // (op 4778) 


v_lshrrev_b32 v8, 16, v4 // (op 4779) 
v_lshrrev_b32 v9, 16, v5 // (op 4779) 
v_lshrrev_b32 v10, 16, v6 // (op 4779) 
v_lshrrev_b32 v11, 16, v7 // (op 4779) 


v_pack_b32_f16 v6, v8, v9 // (op 4772) 
v_pack_b32_f16 v7, v10, v11 // (op 4772) 

s_lshl_b32 s0, s4, 6 // (op 4772) 
s_add_u32 s0, s0, 64 // (op 4772) 
v_add_u32 v0, v20, s0 // (op 4772) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4772) Store value





v_add_u32 v0, 128, v19 // (op 4820) 

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4820) Load value

v_accvgpr_read v12, a16 // (op 4811) 
v_accvgpr_read v13, a17 // (op 4811) 
v_accvgpr_read v14, a18 // (op 4811) 
v_accvgpr_read v15, a19 // (op 4811) 
v_mul_f32 v8, v21, v12 // (op 4811) 
v_mul_f32 v9, v21, v13 // (op 4811) 
v_mul_f32 v10, v21, v14 // (op 4811) 
v_mul_f32 v11, v21, v15 // (op 4811) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4819) 
v_lshrrev_b32 v7, 16, v16 // (op 4819) 
v_lshlrev_b32 v6, 16, v6 // (op 4819) 
v_lshlrev_b32 v7, 16, v7 // (op 4819) 
v_mul_f32 v12, v18, v6 // (op 4819) 
v_mul_f32 v13, v18, v7 // (op 4819) 
v_and_b32 v6, 65535, v17 // (op 4819) 
v_lshrrev_b32 v7, 16, v17 // (op 4819) 
v_lshlrev_b32 v6, 16, v6 // (op 4819) 
v_lshlrev_b32 v7, 16, v7 // (op 4819) 
v_mul_f32 v14, v18, v6 // (op 4819) 
v_mul_f32 v15, v18, v7 // (op 4819) 


v_add_f32 v4, v12, v8 // (op 4812) 
v_add_f32 v5, v13, v9 // (op 4812) 
v_add_f32 v6, v14, v10 // (op 4812) 
v_add_f32 v7, v15, v11 // (op 4812) 


v_lshrrev_b32 v8, 16, v4 // (op 4813) 
v_lshrrev_b32 v9, 16, v5 // (op 4813) 
v_lshrrev_b32 v10, 16, v6 // (op 4813) 
v_lshrrev_b32 v11, 16, v7 // (op 4813) 


v_pack_b32_f16 v6, v8, v9 // (op 4815) 
v_pack_b32_f16 v7, v10, v11 // (op 4815) 

v_add_u32 v0, 128, v20 // (op 4815) 

buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4815) Store value





s_lshl_b32 s0, s2, 6 // (op 4825) 
s_add_u32 s0, s0, 128 // (op 4825) 
v_add_u32 v0, v19, s0 // (op 4825) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4825) Load value

v_accvgpr_read v12, a20 // (op 4827) 
v_accvgpr_read v13, a21 // (op 4827) 
v_accvgpr_read v14, a22 // (op 4827) 
v_accvgpr_read v15, a23 // (op 4827) 
v_mul_f32 v8, v21, v12 // (op 4827) 
v_mul_f32 v9, v21, v13 // (op 4827) 
v_mul_f32 v10, v21, v14 // (op 4827) 
v_mul_f32 v11, v21, v15 // (op 4827) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4822) 
v_lshrrev_b32 v7, 16, v16 // (op 4822) 
v_lshlrev_b32 v6, 16, v6 // (op 4822) 
v_lshlrev_b32 v7, 16, v7 // (op 4822) 
v_mul_f32 v12, v18, v6 // (op 4822) 
v_mul_f32 v13, v18, v7 // (op 4822) 
v_and_b32 v6, 65535, v17 // (op 4822) 
v_lshrrev_b32 v7, 16, v17 // (op 4822) 
v_lshlrev_b32 v6, 16, v6 // (op 4822) 
v_lshlrev_b32 v7, 16, v7 // (op 4822) 
v_mul_f32 v14, v18, v6 // (op 4822) 
v_mul_f32 v15, v18, v7 // (op 4822) 


v_add_f32 v4, v12, v8 // (op 4823) 
v_add_f32 v5, v13, v9 // (op 4823) 
v_add_f32 v6, v14, v10 // (op 4823) 
v_add_f32 v7, v15, v11 // (op 4823) 


v_lshrrev_b32 v8, 16, v4 // (op 4824) 
v_lshrrev_b32 v9, 16, v5 // (op 4824) 
v_lshrrev_b32 v10, 16, v6 // (op 4824) 
v_lshrrev_b32 v11, 16, v7 // (op 4824) 


v_pack_b32_f16 v6, v8, v9 // (op 4817) 
v_pack_b32_f16 v7, v10, v11 // (op 4817) 

s_lshl_b32 s0, s4, 6 // (op 4817) 
s_add_u32 s0, s0, 128 // (op 4817) 
v_add_u32 v0, v20, s0 // (op 4817) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4817) Store value





v_mov_b32 v0, v19 // (op 148) reg expression

buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 148) Load value

v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v8, v21, v12 // (op 144) 
v_mul_f32 v9, v21, v13 // (op 144) 
v_mul_f32 v10, v21, v14 // (op 144) 
v_mul_f32 v11, v21, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 149) 
v_lshrrev_b32 v7, 16, v16 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v12, v18, v6 // (op 149) 
v_mul_f32 v13, v18, v7 // (op 149) 
v_and_b32 v6, 65535, v17 // (op 149) 
v_lshrrev_b32 v7, 16, v17 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v14, v18, v6 // (op 149) 
v_mul_f32 v15, v18, v7 // (op 149) 


v_add_f32 v4, v12, v8 // (op 145) 
v_add_f32 v5, v13, v9 // (op 145) 
v_add_f32 v6, v14, v10 // (op 145) 
v_add_f32 v7, v15, v11 // (op 145) 


v_lshrrev_b32 v8, 16, v4 // (op 146) 
v_lshrrev_b32 v9, 16, v5 // (op 146) 
v_lshrrev_b32 v10, 16, v6 // (op 146) 
v_lshrrev_b32 v11, 16, v7 // (op 146) 


v_pack_b32_f16 v6, v8, v9 // (op 147) 
v_pack_b32_f16 v7, v10, v11 // (op 147) 

v_mov_b32 v0, v20 // (op 147) reg expression

buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 4735) 
v_add_u32 v0, v19, s0 // (op 4735) 


buffer_load_dwordx2 v[16:17], v0, s[56:59], 0 offen // (op 4735) Load value

v_accvgpr_read v12, a4 // (op 4731) 
v_accvgpr_read v13, a5 // (op 4731) 
v_accvgpr_read v14, a6 // (op 4731) 
v_accvgpr_read v15, a7 // (op 4731) 
v_mul_f32 v8, v21, v12 // (op 4731) 
v_mul_f32 v9, v21, v13 // (op 4731) 
v_mul_f32 v10, v21, v14 // (op 4731) 
v_mul_f32 v11, v21, v15 // (op 4731) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 4736) 
v_lshrrev_b32 v7, 16, v16 // (op 4736) 
v_lshlrev_b32 v6, 16, v6 // (op 4736) 
v_lshlrev_b32 v7, 16, v7 // (op 4736) 
v_mul_f32 v12, v18, v6 // (op 4736) 
v_mul_f32 v13, v18, v7 // (op 4736) 
v_and_b32 v6, 65535, v17 // (op 4736) 
v_lshrrev_b32 v7, 16, v17 // (op 4736) 
v_lshlrev_b32 v6, 16, v6 // (op 4736) 
v_lshlrev_b32 v7, 16, v7 // (op 4736) 
v_mul_f32 v14, v18, v6 // (op 4736) 
v_mul_f32 v15, v18, v7 // (op 4736) 


v_add_f32 v4, v12, v8 // (op 4732) 
v_add_f32 v5, v13, v9 // (op 4732) 
v_add_f32 v6, v14, v10 // (op 4732) 
v_add_f32 v7, v15, v11 // (op 4732) 


v_lshrrev_b32 v8, 16, v4 // (op 4733) 
v_lshrrev_b32 v9, 16, v5 // (op 4733) 
v_lshrrev_b32 v10, 16, v6 // (op 4733) 
v_lshrrev_b32 v11, 16, v7 // (op 4733) 


v_pack_b32_f16 v6, v8, v9 // (op 4734) 
v_pack_b32_f16 v7, v10, v11 // (op 4734) 

s_lshl_b32 s0, s4, 6 // (op 4734) 
v_add_u32 v0, v20, s0 // (op 4734) 


buffer_store_dwordx2 v[6:7], v0, s[60:63], 0 offen // (op 4734) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__ConditionalBottom_0_beta_135:




















s_endpgm  // End of RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM_
.LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__end:
.size RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM_, .LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM__end-RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM_
  .amdhsa_next_free_vgpr 200
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 34816 // lds bytes
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
            { type: int64, semantic: StrideD0 },
            { type: int64, semantic: RR_NumTilesM },
            { type: int64, semantic: SizeFree1 },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
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
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [192, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [6, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x64x128_UR_4_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 34816
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 152
    .agpr_count: 48
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
      - .name: BitwiseXor_18
        .size: 8
        .offset: 16
        .expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 32
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 156, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 36
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_25, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 168, size: 4}, rhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 40
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 156, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_24
        .size: 4
        .offset: 44
        .expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}
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
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
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
      - .name: Subtract_22
        .size: 4
        .offset: 152
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 156
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_25, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 168, size: 4}, rhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}}}
        .variableType: {dataType: Int32, pointerType: Value}
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

