

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM_,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__exec_begin:


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


v_lshrrev_b32 v0, 8, v1 // (op 30941) 
v_mul_lo_u32 v3, v0, 12 // (op 30941) 

v_and_b32 v0, 15, v1 // (op 30941) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 30941) 


v_and_b32 v0, 63, v1 // (op 30941) 
v_lshrrev_b32 v3, 4, v0 // (op 30941) 

v_lshlrev_b32 v0, 5, v3 // (op 30941) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 30941) 


v_lshlrev_b32 v0, 2, v3 // (op 30941) 

v_lshrrev_b32 v3, 3, v0 // (op 30941) 

v_lshrrev_b32 v0, 6, v1 // (op 30998) 
v_and_b32 v4, 3, v0 // (op 30998) 

v_and_b32 v0, 15, v1 // (op 30998) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 30998) 


v_and_b32 v0, 63, v1 // (op 30998) 
v_lshrrev_b32 v4, 4, v0 // (op 30998) 

v_lshlrev_b32 v0, 5, v4 // (op 30998) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 30998) 


v_lshlrev_b32 v0, 2, v4 // (op 30998) 

v_lshrrev_b32 v4, 3, v0 // (op 30998) 

v_lshrrev_b32 v0, 8, v1 // (op 31065) 
v_mul_lo_u32 v5, v0, 12 // (op 31065) 

v_and_b32 v0, 15, v1 // (op 31065) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 31065) 


v_and_b32 v0, 63, v1 // (op 31065) 
v_lshrrev_b32 v5, 4, v0 // (op 31065) 

v_lshlrev_b32 v0, 5, v5 // (op 31065) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 31065) 


v_lshlrev_b32 v0, 2, v5 // (op 31065) 

v_lshrrev_b32 v5, 3, v0 // (op 31065) 

v_lshrrev_b32 v0, 6, v1 // (op 31109) 
v_and_b32 v6, 3, v0 // (op 31109) 

v_and_b32 v0, 15, v1 // (op 31109) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 31109) 


v_and_b32 v0, 63, v1 // (op 31109) 
v_lshrrev_b32 v6, 4, v0 // (op 31109) 

v_lshlrev_b32 v0, 5, v6 // (op 31109) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 31109) 


v_lshlrev_b32 v0, 2, v6 // (op 31109) 

v_lshrrev_b32 v6, 3, v0 // (op 31109) 

v_lshrrev_b32 v0, 8, v1 // (op 31153) 
v_mul_lo_u32 v7, v0, 12 // (op 31153) 

v_and_b32 v0, 15, v1 // (op 31153) 
v_lshl_add_u32 v8, v7, 4, v0 // (op 31153) 


v_and_b32 v0, 63, v1 // (op 31153) 
v_lshrrev_b32 v7, 4, v0 // (op 31153) 

v_lshlrev_b32 v0, 5, v7 // (op 31153) 

v_lshl_add_u32 v7, v8, 7, v0 // (op 31153) 


v_lshlrev_b32 v0, 2, v7 // (op 31153) 

v_lshrrev_b32 v7, 3, v0 // (op 31153) 

v_lshrrev_b32 v0, 6, v1 // (op 31197) 
v_and_b32 v8, 3, v0 // (op 31197) 

v_and_b32 v0, 15, v1 // (op 31197) 
v_lshl_add_u32 v9, v8, 6, v0 // (op 31197) 


v_and_b32 v0, 63, v1 // (op 31197) 
v_lshrrev_b32 v8, 4, v0 // (op 31197) 

v_lshlrev_b32 v0, 5, v8 // (op 31197) 

v_lshl_add_u32 v8, v9, 7, v0 // (op 31197) 


v_lshlrev_b32 v0, 2, v8 // (op 31197) 

v_lshrrev_b32 v8, 3, v0 // (op 31197) 

v_lshrrev_b32 v0, 8, v1 // (op 31241) 
v_mul_lo_u32 v9, v0, 12 // (op 31241) 

v_and_b32 v0, 15, v1 // (op 31241) 
v_lshl_add_u32 v10, v9, 4, v0 // (op 31241) 


v_and_b32 v0, 63, v1 // (op 31241) 
v_lshrrev_b32 v9, 4, v0 // (op 31241) 

v_lshlrev_b32 v0, 5, v9 // (op 31241) 

v_lshl_add_u32 v9, v10, 7, v0 // (op 31241) 


v_lshlrev_b32 v0, 2, v9 // (op 31241) 

v_lshrrev_b32 v9, 3, v0 // (op 31241) 

v_lshrrev_b32 v0, 6, v1 // (op 31285) 
v_and_b32 v10, 3, v0 // (op 31285) 

v_and_b32 v0, 15, v1 // (op 31285) 
v_lshl_add_u32 v11, v10, 6, v0 // (op 31285) 


v_and_b32 v0, 63, v1 // (op 31285) 
v_lshrrev_b32 v10, 4, v0 // (op 31285) 

v_lshlrev_b32 v0, 5, v10 // (op 31285) 

v_lshl_add_u32 v10, v11, 7, v0 // (op 31285) 


v_lshlrev_b32 v0, 2, v10 // (op 31285) 

v_lshrrev_b32 v10, 3, v0 // (op 31285) 

v_lshrrev_b32 v0, 8, v1 // (op 30880) 
v_lshrrev_b32 v11, 8, v1 // (op 31019) 

s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s0, s16, s47 // (op 30880) 
s_add_i32 s1, s0, s16 // (op 30880) 

s_mul_hi_i32 s0, s16, s47 // (op 31019) 
s_add_i32 s18, s0, s16 // (op 31019) 

v_mul_lo_u32 v12, v0, 12 // (op 30880) 

s_ashr_i32 s0, s1, 31 // (op 30880) 
v_and_b32 v0, 63, v1 // (op 30880) 
s_cmp_eq_i32 s47, 0 // (op 30880) 
s_mov_b32 s19, scc // (op 30880) 
v_lshrrev_b32 v13, 4, v0 // (op 30880) 
s_cmp_eq_i32 s19, 0 // (op 30880) 
s_cselect_b32 s51, 0, -1 // (op 30880) 

v_lshlrev_b32 v14, 2, v13 // (op 30880) 

s_and_b32 s19, s13, 31 // (op 30880) 
v_lshl_add_u32 v13, v12, 4, v14 // (op 30880) 


v_lshrrev_b32 v12, 2, v0 // (op 30880) 

v_mul_lo_u32 v0, v11, 12 // (op 31019) 

v_and_b32 v11, 63, v1 // (op 31019) 
s_lshl_b32 s52, 1, s19 // (op 30880) 
v_and_b32 v14, 3, v12 // (op 30880) 

s_add_i32 s53, s52, s51 // (op 30880) 


s_and_b32 s51, s0, s53 // (op 30880) 


s_ashr_i32 s0, s18, 31 // (op 31019) 
s_cmp_eq_i32 s47, 0 // (op 31019) 
s_mov_b32 s52, scc // (op 31019) 
v_and_b32 v12, 3, v1 // (op 30880) 
s_add_i32 s53, s1, s51 // (op 30880) 


v_lshl_add_u32 v15, v14, 2, v12 // (op 30880) 


s_ashr_i32 s1, s53, s19 // (op 30880) 


v_lshrrev_b32 v12, 6, v1 // (op 30880) 
s_lshl_b32 s19, s13, 24 // (op 30880) 
v_and_b32 v14, 3, v12 // (op 30880) 

s_ashr_i32 s51, s19, 31 // (op 30880) 

v_lshl_add_u32 v12, v14, 6, v15 // (op 30880) 


s_xor_b32 s19, s1, s51 // (op 30880) 

s_sub_i32 s1, s19, s51 // (op 30880) 


s_mul_hi_i32 s19, s1, s14 // (op 30880) 
s_add_i32 s51, s19, s1 // (op 30880) 

s_ashr_i32 s19, s51, 31 // (op 30880) 
s_cmp_eq_i32 s14, 0 // (op 30880) 
s_mov_b32 s53, scc // (op 30880) 
s_cmp_eq_i32 s53, 0 // (op 30880) 
s_cselect_b32 s54, 0, -1 // (op 30880) 

s_and_b32 s53, s15, 31 // (op 30880) 
v_lshrrev_b32 v14, 4, v11 // (op 31019) 
s_cmp_eq_i32 s52, 0 // (op 31019) 
s_cselect_b32 s55, 0, -1 // (op 31019) 

s_lshl_b32 s52, 1, s53 // (op 30880) 
v_lshlrev_b32 v15, 2, v14 // (op 31019) 

v_lshl_add_u32 v14, v0, 4, v15 // (op 31019) 


s_add_i32 s56, s52, s54 // (op 30880) 


s_and_b32 s52, s19, s56 // (op 30880) 


s_add_i32 s19, s51, s52 // (op 30880) 


s_ashr_i32 s54, s19, s53 // (op 30880) 


s_lshl_b32 s19, s15, 24 // (op 30880) 
s_ashr_i32 s51, s19, 31 // (op 30880) 

s_xor_b32 s19, s54, s51 // (op 30880) 

s_sub_i32 s52, s19, s51 // (op 30880) 


s_mul_i32 s19, s1, s48 // (op 30880) 
s_sub_u32 s51, s16, s19 // (op 30880) 

s_mul_i32 s19, s52, s48 // (op 30880) 
s_add_u32 s19, s19, s51 // (op 30880) 

s_mul_hi_i32 s51, s16, s10 // (op 30880) 
s_add_i32 s53, s51, s16 // (op 30880) 

s_ashr_i32 s51, s53, 31 // (op 30880) 
s_cmp_eq_i32 s10, 0 // (op 30880) 
s_mov_b32 s54, scc // (op 30880) 
s_cmp_eq_i32 s54, 0 // (op 30880) 
s_cselect_b32 s56, 0, -1 // (op 30880) 

s_and_b32 s54, s12, 31 // (op 30880) 
s_lshl_b32 s57, 1, s54 // (op 30880) 
s_add_i32 s58, s57, s56 // (op 30880) 


s_and_b32 s56, s51, s58 // (op 30880) 


s_add_i32 s51, s53, s56 // (op 30880) 


s_ashr_i32 s53, s51, s54 // (op 30880) 


s_lshl_b32 s51, s12, 24 // (op 30880) 
s_ashr_i32 s54, s51, 31 // (op 30880) 

s_xor_b32 s51, s53, s54 // (op 30880) 

s_sub_i32 s53, s51, s54 // (op 30880) 


s_and_b32 s51, s13, 31 // (op 31019) 
s_lshl_b32 s54, 1, s51 // (op 31019) 
v_lshrrev_b32 v0, 2, v11 // (op 31019) 

s_add_i32 s56, s54, s55 // (op 31019) 


v_and_b32 v11, 3, v0 // (op 31019) 

s_and_b32 s54, s0, s56 // (op 31019) 


v_and_b32 v0, 3, v1 // (op 31019) 
s_add_i32 s0, s18, s54 // (op 31019) 


v_lshl_add_u32 v15, v11, 2, v0 // (op 31019) 


s_ashr_i32 s18, s0, s51 // (op 31019) 


v_lshrrev_b32 v0, 6, v1 // (op 31019) 
s_lshl_b32 s0, s13, 24 // (op 31019) 
v_and_b32 v11, 3, v0 // (op 31019) 

s_ashr_i32 s51, s0, 31 // (op 31019) 

v_lshl_add_u32 v0, v11, 6, v15 // (op 31019) 


s_xor_b32 s0, s18, s51 // (op 31019) 

s_sub_i32 s18, s0, s51 // (op 31019) 


s_mul_hi_i32 s0, s18, s14 // (op 31019) 
s_add_i32 s51, s0, s18 // (op 31019) 

s_ashr_i32 s0, s51, 31 // (op 31019) 
s_cmp_eq_i32 s14, 0 // (op 31019) 
s_mov_b32 s54, scc // (op 31019) 
s_cmp_eq_i32 s54, 0 // (op 31019) 
s_cselect_b32 s55, 0, -1 // (op 31019) 

s_and_b32 s54, s15, 31 // (op 31019) 
s_lshl_b32 s56, 1, s54 // (op 31019) 
s_add_i32 s57, s56, s55 // (op 31019) 


s_and_b32 s55, s0, s57 // (op 31019) 


s_add_i32 s0, s51, s55 // (op 31019) 


s_ashr_i32 s51, s0, s54 // (op 31019) 


s_lshl_b32 s0, s15, 24 // (op 31019) 
s_ashr_i32 s54, s0, 31 // (op 31019) 

s_xor_b32 s0, s51, s54 // (op 31019) 

s_sub_i32 s51, s0, s54 // (op 31019) 


s_mul_i32 s0, s18, s48 // (op 31019) 
s_sub_u32 s54, s16, s0 // (op 31019) 

s_mul_i32 s0, s51, s48 // (op 31019) 
s_add_u32 s0, s0, s54 // (op 31019) 

s_mul_hi_i32 s54, s16, s10 // (op 31019) 
s_add_i32 s55, s54, s16 // (op 31019) 

s_ashr_i32 s54, s55, 31 // (op 31019) 
s_cmp_eq_i32 s10, 0 // (op 31019) 
s_mov_b32 s56, scc // (op 31019) 
s_cmp_eq_i32 s56, 0 // (op 31019) 
s_cselect_b32 s57, 0, -1 // (op 31019) 

s_and_b32 s56, s12, 31 // (op 31019) 
s_lshl_b32 s58, 1, s56 // (op 31019) 
s_add_i32 s59, s58, s57 // (op 31019) 


s_and_b32 s57, s54, s59 // (op 31019) 


s_add_i32 s54, s55, s57 // (op 31019) 


s_ashr_i32 s55, s54, s56 // (op 31019) 


s_lshl_b32 s54, s12, 24 // (op 31019) 
s_ashr_i32 s56, s54, 31 // (op 31019) 

s_xor_b32 s54, s55, s56 // (op 31019) 

s_sub_i32 s55, s54, s56 // (op 31019) 


s_add_u32 s56, s8, 256 // (op 30880) Start of Int64 add, locking SCC
s_addc_u32 s57, s9, 0 // (op 30880) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 30880) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 30880) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 30880) 

s_mul_i32 s54, s48, s56 // (op 30880) 
s_mul_i32 s58, s44, s54 // (op 30880) 

s_mul_i32 s54, s53, s58 // (op 30880) 

s_sub_u32 s58, s16, s54 // (op 30880) 

s_mov_b32 s54, s58 // (op 30880) 
s_mul_hi_i32 s59, s54, s45 // (op 30880) 
s_add_i32 s60, s59, s54 // (op 30880) 


s_ashr_i32 s54, s60, 31 // (op 30880) 
s_cmp_eq_i32 s45, 0 // (op 30880) 
s_mov_b32 s59, scc // (op 30880) 
s_cmp_eq_i32 s59, 0 // (op 30880) 
s_cselect_b32 s61, 0, -1 // (op 30880) 

s_and_b32 s59, s11, 31 // (op 30880) 
s_lshl_b32 s62, 1, s59 // (op 30880) 
s_add_i32 s63, s62, s61 // (op 30880) 


s_and_b32 s61, s54, s63 // (op 30880) 


s_add_i32 s54, s60, s61 // (op 30880) 


s_ashr_i32 s60, s54, s59 // (op 30880) 


s_lshl_b32 s54, s11, 24 // (op 30880) 
s_ashr_i32 s59, s54, 31 // (op 30880) 

s_xor_b32 s54, s60, s59 // (op 30880) 

s_sub_i32 s60, s54, s59 // (op 30880) 


s_mul_i32 s54, s46, s48 // (op 30880) 
s_sub_i32 s59, s6, s54 // (op 30880) 
s_mul_i32 s61, s60, s59 // (op 30880) 

s_sub_u32 s59, s58, s61 // (op 30880) 


s_add_u32 s58, s59, s54 // (op 30880) 


s_cmp_eq_i32 s53, 0 // (op 30880) 
s_mov_b32 s54, scc // (op 30880) 

s_cmp_eq_i32 s54, 0 // (op 30880) 
s_cselect_b32 s53, s58, s19 // (op 30880) 


s_mul_i32 s19, s52, s56 // (op 30880) 


v_mov_b32 v11, 192 // (op 30880) 
s_sub_i32 s52, s1, s19 // (op 30880) 


v_mul_lo_u32 v15, s53, v11 // (op 30880) 

s_cmp_eq_i32 s54, 0 // (op 30880) 
s_cselect_b32 s1, s60, s52 // (op 30880) 



v_add_u32 v15, v15, v13 // (op 30880) 


v_lshl_add_u32 v11, s1, 8, v12 // (op 30880) 


v_mul_lo_u32 v12, v11, s2 // (op 30880) 

v_add_lshl_u32 v11, v15, v12, 2 // (op 30880) 


s_add_u32 s52, s8, 256 // (op 31019) Start of Int64 add, locking SCC
s_addc_u32 s53, s9, 0 // (op 31019) End of Int64 add, unlocking SCC
s_sub_u32 s56, s52, 1 // (op 31019) least significant half; sets scc
s_subb_u32 s57, s53, 0 // (op 31019) most significant half; uses scc

s_ashr_i64 s[52:53], s[56:57], 8 // (op 31019) 

s_mul_i32 s1, s48, s52 // (op 31019) 
s_mul_i32 s19, s44, s1 // (op 31019) 

s_mul_i32 s1, s55, s19 // (op 31019) 

s_sub_u32 s19, s16, s1 // (op 31019) 

s_mov_b32 s1, s19 // (op 31019) 
s_mul_hi_i32 s54, s1, s45 // (op 31019) 
s_add_i32 s56, s54, s1 // (op 31019) 


s_ashr_i32 s1, s56, 31 // (op 31019) 
s_cmp_eq_i32 s45, 0 // (op 31019) 
s_mov_b32 s54, scc // (op 31019) 
s_cmp_eq_i32 s54, 0 // (op 31019) 
s_cselect_b32 s57, 0, -1 // (op 31019) 

s_and_b32 s54, s11, 31 // (op 31019) 
s_lshl_b32 s58, 1, s54 // (op 31019) 
s_add_i32 s59, s58, s57 // (op 31019) 


s_and_b32 s57, s1, s59 // (op 31019) 


s_add_i32 s1, s56, s57 // (op 31019) 


s_ashr_i32 s56, s1, s54 // (op 31019) 


s_lshl_b32 s1, s11, 24 // (op 31019) 
s_ashr_i32 s54, s1, 31 // (op 31019) 

s_xor_b32 s1, s56, s54 // (op 31019) 

s_sub_i32 s56, s1, s54 // (op 31019) 


s_mul_i32 s1, s46, s48 // (op 31019) 
s_sub_i32 s54, s6, s1 // (op 31019) 
s_mul_i32 s57, s56, s54 // (op 31019) 

s_sub_u32 s54, s19, s57 // (op 31019) 


s_add_u32 s19, s54, s1 // (op 31019) 


s_cmp_eq_i32 s55, 0 // (op 31019) 
s_mov_b32 s1, scc // (op 31019) 

s_cmp_eq_i32 s1, 0 // (op 31019) 
s_cselect_b32 s54, s19, s0 // (op 31019) 


s_mul_i32 s0, s51, s52 // (op 31019) 


v_mov_b32 v12, 192 // (op 31019) 
s_sub_i32 s19, s18, s0 // (op 31019) 


v_mul_lo_u32 v13, s54, v12 // (op 31019) 

s_cmp_eq_i32 s1, 0 // (op 31019) 
s_cselect_b32 s0, s56, s19 // (op 31019) 



v_add_u32 v13, v13, v14 // (op 31019) 


v_lshl_add_u32 v12, s0, 8, v0 // (op 31019) 


v_mul_lo_u32 v0, v12, s4 // (op 31019) 

v_add_lshl_u32 v12, v13, v0, 2 // (op 31019) 


s_mul_i32 s1, s2, s9 // (op 30882) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 30882) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 30882) least significant
s_mul_hi_u32 s19, s2, s8 // (op 30882) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30882) most significant: sum
s_add_u32 s1, s1, s19 // (op 30882) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 30882) 

s_mul_i32 s1, s4, s9 // (op 31021) most significant: low of low * high
s_mul_i32 s51, s5, s8 // (op 31021) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 31021) least significant
s_mul_hi_u32 s52, s4, s8 // (op 31021) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 31021) most significant: sum
s_add_u32 s1, s1, s52 // (op 31021) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 2 // (op 31021) 

s_mov_b32 s56, s20 // (op 30882) reg expression
s_mov_b32 s57, s21 // (op 30882) reg expression
s_mov_b32 s58, s18 // (op 30882) reg expression
s_mov_b32 s59, 131072 // (op 30882) call()

s_mov_b32 s60, s22 // (op 31021) reg expression

s_mov_b32 s61, s23 // (op 31021) reg expression
s_mov_b32 s62, s52 // (op 31021) reg expression
s_mov_b32 s63, 131072 // (op 31021) call()


v_mov_b32 v0, s49 // (op 2) Move value
v_mov_b32 v13, s50 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_accvgpr_write a4, 0.00000 // (op 383) call()
v_accvgpr_write a5, 0.00000 // (op 383) call()
v_accvgpr_write a6, 0.00000 // (op 383) call()
v_accvgpr_write a7, 0.00000 // (op 383) call()
v_accvgpr_write a8, 0.00000 // (op 609) call()
v_accvgpr_write a9, 0.00000 // (op 609) call()
v_accvgpr_write a10, 0.00000 // (op 609) call()
v_accvgpr_write a11, 0.00000 // (op 609) call()
v_accvgpr_write a12, 0.00000 // (op 835) call()
v_accvgpr_write a13, 0.00000 // (op 835) call()
v_accvgpr_write a14, 0.00000 // (op 835) call()
v_accvgpr_write a15, 0.00000 // (op 835) call()
v_accvgpr_write a16, 0.00000 // (op 1197) call()
v_accvgpr_write a17, 0.00000 // (op 1197) call()
v_accvgpr_write a18, 0.00000 // (op 1197) call()
v_accvgpr_write a19, 0.00000 // (op 1197) call()
v_accvgpr_write a20, 0.00000 // (op 1610) call()
v_accvgpr_write a21, 0.00000 // (op 1610) call()
v_accvgpr_write a22, 0.00000 // (op 1610) call()
v_accvgpr_write a23, 0.00000 // (op 1610) call()
v_accvgpr_write a24, 0.00000 // (op 1611) call()
v_accvgpr_write a25, 0.00000 // (op 1611) call()
v_accvgpr_write a26, 0.00000 // (op 1611) call()
v_accvgpr_write a27, 0.00000 // (op 1611) call()
v_accvgpr_write a28, 0.00000 // (op 1612) call()
v_accvgpr_write a29, 0.00000 // (op 1612) call()
v_accvgpr_write a30, 0.00000 // (op 1612) call()
v_accvgpr_write a31, 0.00000 // (op 1612) call()
v_accvgpr_write a32, 0.00000 // (op 2199) call()
v_accvgpr_write a33, 0.00000 // (op 2199) call()
v_accvgpr_write a34, 0.00000 // (op 2199) call()
v_accvgpr_write a35, 0.00000 // (op 2199) call()
v_accvgpr_write a36, 0.00000 // (op 2612) call()
v_accvgpr_write a37, 0.00000 // (op 2612) call()
v_accvgpr_write a38, 0.00000 // (op 2612) call()
v_accvgpr_write a39, 0.00000 // (op 2612) call()
v_accvgpr_write a40, 0.00000 // (op 2613) call()
v_accvgpr_write a41, 0.00000 // (op 2613) call()
v_accvgpr_write a42, 0.00000 // (op 2613) call()
v_accvgpr_write a43, 0.00000 // (op 2613) call()
v_accvgpr_write a44, 0.00000 // (op 2614) call()
v_accvgpr_write a45, 0.00000 // (op 2614) call()
v_accvgpr_write a46, 0.00000 // (op 2614) call()
v_accvgpr_write a47, 0.00000 // (op 2614) call()
v_accvgpr_write a48, 0.00000 // (op 3201) call()
v_accvgpr_write a49, 0.00000 // (op 3201) call()
v_accvgpr_write a50, 0.00000 // (op 3201) call()
v_accvgpr_write a51, 0.00000 // (op 3201) call()
v_accvgpr_write a52, 0.00000 // (op 3614) call()
v_accvgpr_write a53, 0.00000 // (op 3614) call()
v_accvgpr_write a54, 0.00000 // (op 3614) call()
v_accvgpr_write a55, 0.00000 // (op 3614) call()
v_accvgpr_write a56, 0.00000 // (op 3615) call()
v_accvgpr_write a57, 0.00000 // (op 3615) call()
v_accvgpr_write a58, 0.00000 // (op 3615) call()
v_accvgpr_write a59, 0.00000 // (op 3615) call()
v_accvgpr_write a60, 0.00000 // (op 3616) call()
v_accvgpr_write a61, 0.00000 // (op 3616) call()
v_accvgpr_write a62, 0.00000 // (op 3616) call()
v_accvgpr_write a63, 0.00000 // (op 3616) call()
v_accvgpr_write a64, 0.00000 // (op 4203) call()
v_accvgpr_write a65, 0.00000 // (op 4203) call()
v_accvgpr_write a66, 0.00000 // (op 4203) call()
v_accvgpr_write a67, 0.00000 // (op 4203) call()
v_accvgpr_write a68, 0.00000 // (op 4616) call()
v_accvgpr_write a69, 0.00000 // (op 4616) call()
v_accvgpr_write a70, 0.00000 // (op 4616) call()
v_accvgpr_write a71, 0.00000 // (op 4616) call()
v_accvgpr_write a72, 0.00000 // (op 4617) call()
v_accvgpr_write a73, 0.00000 // (op 4617) call()
v_accvgpr_write a74, 0.00000 // (op 4617) call()
v_accvgpr_write a75, 0.00000 // (op 4617) call()
v_accvgpr_write a76, 0.00000 // (op 4618) call()
v_accvgpr_write a77, 0.00000 // (op 4618) call()
v_accvgpr_write a78, 0.00000 // (op 4618) call()
v_accvgpr_write a79, 0.00000 // (op 4618) call()
v_accvgpr_write a80, 0.00000 // (op 5205) call()
v_accvgpr_write a81, 0.00000 // (op 5205) call()
v_accvgpr_write a82, 0.00000 // (op 5205) call()
v_accvgpr_write a83, 0.00000 // (op 5205) call()
v_accvgpr_write a84, 0.00000 // (op 5618) call()
v_accvgpr_write a85, 0.00000 // (op 5618) call()
v_accvgpr_write a86, 0.00000 // (op 5618) call()
v_accvgpr_write a87, 0.00000 // (op 5618) call()
v_accvgpr_write a88, 0.00000 // (op 5619) call()
v_accvgpr_write a89, 0.00000 // (op 5619) call()
v_accvgpr_write a90, 0.00000 // (op 5619) call()
v_accvgpr_write a91, 0.00000 // (op 5619) call()
v_accvgpr_write a92, 0.00000 // (op 5620) call()
v_accvgpr_write a93, 0.00000 // (op 5620) call()
v_accvgpr_write a94, 0.00000 // (op 5620) call()
v_accvgpr_write a95, 0.00000 // (op 5620) call()
v_accvgpr_write a96, 0.00000 // (op 6207) call()
v_accvgpr_write a97, 0.00000 // (op 6207) call()
v_accvgpr_write a98, 0.00000 // (op 6207) call()
v_accvgpr_write a99, 0.00000 // (op 6207) call()
v_accvgpr_write a100, 0.00000 // (op 6620) call()
v_accvgpr_write a101, 0.00000 // (op 6620) call()
v_accvgpr_write a102, 0.00000 // (op 6620) call()
v_accvgpr_write a103, 0.00000 // (op 6620) call()
v_accvgpr_write a104, 0.00000 // (op 6621) call()
v_accvgpr_write a105, 0.00000 // (op 6621) call()
v_accvgpr_write a106, 0.00000 // (op 6621) call()
v_accvgpr_write a107, 0.00000 // (op 6621) call()
v_accvgpr_write a108, 0.00000 // (op 6622) call()
v_accvgpr_write a109, 0.00000 // (op 6622) call()
v_accvgpr_write a110, 0.00000 // (op 6622) call()
v_accvgpr_write a111, 0.00000 // (op 6622) call()
v_accvgpr_write a112, 0.00000 // (op 7209) call()
v_accvgpr_write a113, 0.00000 // (op 7209) call()
v_accvgpr_write a114, 0.00000 // (op 7209) call()
v_accvgpr_write a115, 0.00000 // (op 7209) call()
v_accvgpr_write a116, 0.00000 // (op 7622) call()
v_accvgpr_write a117, 0.00000 // (op 7622) call()
v_accvgpr_write a118, 0.00000 // (op 7622) call()
v_accvgpr_write a119, 0.00000 // (op 7622) call()
v_accvgpr_write a120, 0.00000 // (op 7623) call()
v_accvgpr_write a121, 0.00000 // (op 7623) call()
v_accvgpr_write a122, 0.00000 // (op 7623) call()
v_accvgpr_write a123, 0.00000 // (op 7623) call()
v_accvgpr_write a124, 0.00000 // (op 7624) call()
v_accvgpr_write a125, 0.00000 // (op 7624) call()
v_accvgpr_write a126, 0.00000 // (op 7624) call()
v_accvgpr_write a127, 0.00000 // (op 7624) call()
v_accvgpr_write a128, 0.00000 // (op 8211) call()
v_accvgpr_write a129, 0.00000 // (op 8211) call()
v_accvgpr_write a130, 0.00000 // (op 8211) call()
v_accvgpr_write a131, 0.00000 // (op 8211) call()
v_accvgpr_write a132, 0.00000 // (op 8624) call()
v_accvgpr_write a133, 0.00000 // (op 8624) call()
v_accvgpr_write a134, 0.00000 // (op 8624) call()
v_accvgpr_write a135, 0.00000 // (op 8624) call()
v_accvgpr_write a136, 0.00000 // (op 8625) call()
v_accvgpr_write a137, 0.00000 // (op 8625) call()
v_accvgpr_write a138, 0.00000 // (op 8625) call()
v_accvgpr_write a139, 0.00000 // (op 8625) call()
v_accvgpr_write a140, 0.00000 // (op 8626) call()
v_accvgpr_write a141, 0.00000 // (op 8626) call()
v_accvgpr_write a142, 0.00000 // (op 8626) call()
v_accvgpr_write a143, 0.00000 // (op 8626) call()
v_accvgpr_write a144, 0.00000 // (op 9213) call()
v_accvgpr_write a145, 0.00000 // (op 9213) call()
v_accvgpr_write a146, 0.00000 // (op 9213) call()
v_accvgpr_write a147, 0.00000 // (op 9213) call()
v_accvgpr_write a148, 0.00000 // (op 9626) call()
v_accvgpr_write a149, 0.00000 // (op 9626) call()
v_accvgpr_write a150, 0.00000 // (op 9626) call()
v_accvgpr_write a151, 0.00000 // (op 9626) call()
v_accvgpr_write a152, 0.00000 // (op 9627) call()
v_accvgpr_write a153, 0.00000 // (op 9627) call()
v_accvgpr_write a154, 0.00000 // (op 9627) call()
v_accvgpr_write a155, 0.00000 // (op 9627) call()
v_accvgpr_write a156, 0.00000 // (op 9628) call()
v_accvgpr_write a157, 0.00000 // (op 9628) call()
v_accvgpr_write a158, 0.00000 // (op 9628) call()
v_accvgpr_write a159, 0.00000 // (op 9628) call()
v_accvgpr_write a160, 0.00000 // (op 10215) call()
v_accvgpr_write a161, 0.00000 // (op 10215) call()
v_accvgpr_write a162, 0.00000 // (op 10215) call()
v_accvgpr_write a163, 0.00000 // (op 10215) call()
v_accvgpr_write a164, 0.00000 // (op 10628) call()
v_accvgpr_write a165, 0.00000 // (op 10628) call()
v_accvgpr_write a166, 0.00000 // (op 10628) call()
v_accvgpr_write a167, 0.00000 // (op 10628) call()
v_accvgpr_write a168, 0.00000 // (op 10629) call()
v_accvgpr_write a169, 0.00000 // (op 10629) call()
v_accvgpr_write a170, 0.00000 // (op 10629) call()
v_accvgpr_write a171, 0.00000 // (op 10629) call()
v_accvgpr_write a172, 0.00000 // (op 10630) call()
v_accvgpr_write a173, 0.00000 // (op 10630) call()
v_accvgpr_write a174, 0.00000 // (op 10630) call()
v_accvgpr_write a175, 0.00000 // (op 10630) call()
v_accvgpr_write a176, 0.00000 // (op 11217) call()
v_accvgpr_write a177, 0.00000 // (op 11217) call()
v_accvgpr_write a178, 0.00000 // (op 11217) call()
v_accvgpr_write a179, 0.00000 // (op 11217) call()
v_accvgpr_write a180, 0.00000 // (op 11630) call()
v_accvgpr_write a181, 0.00000 // (op 11630) call()
v_accvgpr_write a182, 0.00000 // (op 11630) call()
v_accvgpr_write a183, 0.00000 // (op 11630) call()
v_accvgpr_write a184, 0.00000 // (op 11631) call()
v_accvgpr_write a185, 0.00000 // (op 11631) call()
v_accvgpr_write a186, 0.00000 // (op 11631) call()
v_accvgpr_write a187, 0.00000 // (op 11631) call()
v_accvgpr_write a188, 0.00000 // (op 11632) call()
v_accvgpr_write a189, 0.00000 // (op 11632) call()
v_accvgpr_write a190, 0.00000 // (op 11632) call()
v_accvgpr_write a191, 0.00000 // (op 11632) call()
s_mul_hi_i32 s0, s16, s47 // (op 30770) 
v_lshrrev_b32 v14, 2, v1 // (op 30770) 
s_add_i32 s1, s0, s16 // (op 30770) 

v_and_b32 v15, -4, v1 // (op 30770) 
s_ashr_i32 s0, s1, 31 // (op 30770) 
v_sub_u32 v16, v1, v15 // (op 30770) 

s_cmp_eq_i32 s47, 0 // (op 30770) 
s_mov_b32 s18, scc // (op 30770) 
v_lshlrev_b32 v15, 5, v16 // (op 30770) 

s_cmp_eq_i32 s18, 0 // (op 30770) 
s_cselect_b32 s19, 0, -1 // (op 30770) 

v_mov_b32 v17, 0 // (op 30770) convert
v_mov_b32 v16, v15 // (op 30770) convert

s_and_b32 s18, s13, 31 // (op 30770) 
s_lshl_b32 s20, 1, s18 // (op 30770) 
s_add_i32 s21, s20, s19 // (op 30770) 


s_and_b32 s19, s0, s21 // (op 30770) 


s_add_i32 s0, s1, s19 // (op 30770) 


s_ashr_i32 s1, s0, s18 // (op 30770) 


s_lshl_b32 s0, s13, 24 // (op 30770) 
s_ashr_i32 s18, s0, 31 // (op 30770) 

s_xor_b32 s0, s1, s18 // (op 30770) 

s_sub_i32 s1, s0, s18 // (op 30770) 


s_mul_hi_i32 s0, s1, s14 // (op 30770) 
s_add_i32 s18, s0, s1 // (op 30770) 

s_ashr_i32 s0, s18, 31 // (op 30770) 
s_cmp_eq_i32 s14, 0 // (op 30770) 
s_mov_b32 s19, scc // (op 30770) 
s_cmp_eq_i32 s19, 0 // (op 30770) 
s_cselect_b32 s20, 0, -1 // (op 30770) 

s_and_b32 s19, s15, 31 // (op 30770) 
s_lshl_b32 s21, 1, s19 // (op 30770) 
s_add_i32 s22, s21, s20 // (op 30770) 


s_and_b32 s23, s0, s22 // (op 30770) 


s_add_i32 s0, s18, s23 // (op 30770) 


s_ashr_i32 s18, s0, s19 // (op 30770) 


s_lshl_b32 s0, s15, 24 // (op 30770) 
s_ashr_i32 s19, s0, 31 // (op 30770) 

s_xor_b32 s0, s18, s19 // (op 30770) 

s_sub_i32 s18, s0, s19 // (op 30770) 


s_mul_i32 s0, s1, s48 // (op 30770) 

s_sub_u32 s1, s16, s0 // (op 30770) 

s_mul_i32 s0, s18, s48 // (op 30770) 
s_add_u32 s0, s0, s1 // (op 30770) 


s_mul_hi_i32 s1, s16, s10 // (op 30770) 
s_add_i32 s18, s1, s16 // (op 30770) 

s_ashr_i32 s1, s18, 31 // (op 30770) 
s_cmp_eq_i32 s10, 0 // (op 30770) 
s_mov_b32 s19, scc // (op 30770) 
s_cmp_eq_i32 s19, 0 // (op 30770) 
s_cselect_b32 s20, 0, -1 // (op 30770) 

s_and_b32 s19, s12, 31 // (op 30770) 
s_lshl_b32 s21, 1, s19 // (op 30770) 
s_add_i32 s22, s21, s20 // (op 30770) 


s_and_b32 s23, s1, s22 // (op 30770) 


s_add_i32 s1, s18, s23 // (op 30770) 


s_ashr_i32 s18, s1, s19 // (op 30770) 


s_lshl_b32 s1, s12, 24 // (op 30770) 
s_ashr_i32 s19, s1, 31 // (op 30770) 

s_xor_b32 s1, s18, s19 // (op 30770) 

s_sub_i32 s18, s1, s19 // (op 30770) 


s_add_u32 s22, s8, 256 // (op 30770) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 30770) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 30770) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 30770) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 8 // (op 30770) 

s_mul_i32 s1, s48, s22 // (op 30770) 

s_mul_i32 s19, s44, s1 // (op 30770) 

s_mul_i32 s1, s18, s19 // (op 30770) 

s_sub_u32 s19, s16, s1 // (op 30770) 

s_mov_b32 s1, s19 // (op 30770) 
s_mul_hi_i32 s20, s1, s45 // (op 30770) 
s_add_i32 s21, s20, s1 // (op 30770) 


s_ashr_i32 s1, s21, 31 // (op 30770) 
s_cmp_eq_i32 s45, 0 // (op 30770) 
s_mov_b32 s20, scc // (op 30770) 
s_cmp_eq_i32 s20, 0 // (op 30770) 
s_cselect_b32 s22, 0, -1 // (op 30770) 

s_and_b32 s20, s11, 31 // (op 30770) 
s_lshl_b32 s23, 1, s20 // (op 30770) 
s_add_i32 s49, s23, s22 // (op 30770) 


s_and_b32 s22, s1, s49 // (op 30770) 


s_add_i32 s1, s21, s22 // (op 30770) 


s_ashr_i32 s21, s1, s20 // (op 30770) 


s_lshl_b32 s1, s11, 24 // (op 30770) 
s_ashr_i32 s20, s1, 31 // (op 30770) 

s_xor_b32 s1, s21, s20 // (op 30770) 

s_sub_i32 s21, s1, s20 // (op 30770) 


s_mul_i32 s1, s46, s48 // (op 30770) 
s_sub_i32 s20, s6, s1 // (op 30770) 
s_mul_i32 s22, s21, s20 // (op 30770) 


s_sub_u32 s23, s19, s22 // (op 30770) 


s_add_u32 s19, s23, s1 // (op 30770) 


s_cmp_eq_i32 s18, 0 // (op 30770) 
s_mov_b32 s1, scc // (op 30770) 

s_cmp_eq_i32 s1, 0 // (op 30770) 
s_cselect_b32 s18, s19, s0 // (op 30770) 



v_mov_b32 v15, 192 // (op 30770) 
v_mul_lo_u32 v18, s18, v15 // (op 30770) 

v_add_u32 v18, v18, v14 // (op 30770) 


v_mul_lo_u32 v15, v18, s25 // (op 30770) most significant: low of low * high
v_mul_hi_u32 v19, v18, s24 // (op 30770) most significant: high of low * low
v_mul_lo_u32 v14, v18, s24 // (op 30770) least significant: low of low * low
v_add_u32 v15, v15, v19 // (op 30770) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v16 // (op 30770) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 30770) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 30770) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 30770) 

s_mul_i32 s1, s24, s29 // (op 30772) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 30772) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 30772) least significant
s_mul_hi_u32 s19, s24, s28 // (op 30772) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30772) most significant: sum
s_add_u32 s1, s1, s19 // (op 30772) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 30772) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 30772) 

s_mov_b32 s20, s26 // (op 30772) reg expression
s_mov_b32 s21, s27 // (op 30772) reg expression
s_mov_b32 s22, s0 // (op 30772) reg expression
s_mov_b32 s23, 131072 // (op 30772) call()

s_mul_hi_i32 s0, s16, s47 // (op 30789) 
v_lshrrev_b32 v16, 2, v1 // (op 30789) 
s_add_i32 s1, s0, s16 // (op 30789) 

v_and_b32 v17, -4, v1 // (op 30789) 
s_ashr_i32 s0, s1, 31 // (op 30789) 
v_sub_u32 v18, v1, v17 // (op 30789) 

s_cmp_eq_i32 s47, 0 // (op 30789) 
s_mov_b32 s18, scc // (op 30789) 
v_lshlrev_b32 v17, 5, v18 // (op 30789) 

s_cmp_eq_i32 s18, 0 // (op 30789) 
s_cselect_b32 s19, 0, -1 // (op 30789) 

v_mov_b32 v19, 0 // (op 30789) convert
v_mov_b32 v18, v17 // (op 30789) convert

s_and_b32 s18, s13, 31 // (op 30789) 
s_lshl_b32 s49, 1, s18 // (op 30789) 
s_add_i32 s50, s49, s19 // (op 30789) 


s_and_b32 s19, s0, s50 // (op 30789) 


s_add_i32 s0, s1, s19 // (op 30789) 


s_ashr_i32 s1, s0, s18 // (op 30789) 


s_lshl_b32 s0, s13, 24 // (op 30789) 
s_ashr_i32 s18, s0, 31 // (op 30789) 

s_xor_b32 s0, s1, s18 // (op 30789) 

s_sub_i32 s1, s0, s18 // (op 30789) 


s_mul_hi_i32 s0, s1, s14 // (op 30789) 
s_add_i32 s18, s0, s1 // (op 30789) 

s_ashr_i32 s0, s18, 31 // (op 30789) 
s_cmp_eq_i32 s14, 0 // (op 30789) 
s_mov_b32 s19, scc // (op 30789) 
s_cmp_eq_i32 s19, 0 // (op 30789) 
s_cselect_b32 s49, 0, -1 // (op 30789) 

s_and_b32 s19, s15, 31 // (op 30789) 
s_lshl_b32 s50, 1, s19 // (op 30789) 
s_add_i32 s51, s50, s49 // (op 30789) 


s_and_b32 s49, s0, s51 // (op 30789) 


s_add_i32 s0, s18, s49 // (op 30789) 


s_ashr_i32 s18, s0, s19 // (op 30789) 


s_lshl_b32 s0, s15, 24 // (op 30789) 
s_ashr_i32 s19, s0, 31 // (op 30789) 

s_xor_b32 s0, s18, s19 // (op 30789) 

s_sub_i32 s18, s0, s19 // (op 30789) 


s_mul_i32 s0, s1, s48 // (op 30789) 

s_sub_u32 s1, s16, s0 // (op 30789) 

s_mul_i32 s0, s18, s48 // (op 30789) 
s_add_u32 s0, s0, s1 // (op 30789) 


s_mul_hi_i32 s1, s16, s10 // (op 30789) 
s_add_i32 s18, s1, s16 // (op 30789) 

s_ashr_i32 s1, s18, 31 // (op 30789) 
s_cmp_eq_i32 s10, 0 // (op 30789) 
s_mov_b32 s19, scc // (op 30789) 
s_cmp_eq_i32 s19, 0 // (op 30789) 
s_cselect_b32 s49, 0, -1 // (op 30789) 

s_and_b32 s19, s12, 31 // (op 30789) 
s_lshl_b32 s50, 1, s19 // (op 30789) 
s_add_i32 s51, s50, s49 // (op 30789) 


s_and_b32 s49, s1, s51 // (op 30789) 


s_add_i32 s1, s18, s49 // (op 30789) 


s_ashr_i32 s18, s1, s19 // (op 30789) 


s_lshl_b32 s1, s12, 24 // (op 30789) 
s_ashr_i32 s19, s1, 31 // (op 30789) 

s_xor_b32 s1, s18, s19 // (op 30789) 

s_sub_i32 s18, s1, s19 // (op 30789) 


s_add_u32 s54, s8, 256 // (op 30789) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 30789) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 30789) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 30789) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 30789) 

s_mul_i32 s1, s48, s50 // (op 30789) 

s_mul_i32 s19, s44, s1 // (op 30789) 

s_mul_i32 s1, s18, s19 // (op 30789) 

s_sub_u32 s19, s16, s1 // (op 30789) 

s_mov_b32 s1, s19 // (op 30789) 
s_mul_hi_i32 s49, s1, s45 // (op 30789) 
s_add_i32 s50, s49, s1 // (op 30789) 


s_ashr_i32 s1, s50, 31 // (op 30789) 
s_cmp_eq_i32 s45, 0 // (op 30789) 
s_mov_b32 s49, scc // (op 30789) 
s_cmp_eq_i32 s49, 0 // (op 30789) 
s_cselect_b32 s51, 0, -1 // (op 30789) 

s_and_b32 s49, s11, 31 // (op 30789) 
s_lshl_b32 s52, 1, s49 // (op 30789) 
s_add_i32 s53, s52, s51 // (op 30789) 


s_and_b32 s51, s1, s53 // (op 30789) 


s_add_i32 s1, s50, s51 // (op 30789) 


s_ashr_i32 s50, s1, s49 // (op 30789) 


s_lshl_b32 s1, s11, 24 // (op 30789) 
s_ashr_i32 s49, s1, 31 // (op 30789) 

s_xor_b32 s1, s50, s49 // (op 30789) 

s_sub_i32 s50, s1, s49 // (op 30789) 


s_mul_i32 s1, s46, s48 // (op 30789) 
s_sub_i32 s49, s6, s1 // (op 30789) 
s_mul_i32 s51, s50, s49 // (op 30789) 


s_sub_u32 s49, s19, s51 // (op 30789) 


s_add_u32 s19, s49, s1 // (op 30789) 


s_cmp_eq_i32 s18, 0 // (op 30789) 
s_mov_b32 s1, scc // (op 30789) 

s_cmp_eq_i32 s1, 0 // (op 30789) 
s_cselect_b32 s18, s19, s0 // (op 30789) 



v_mov_b32 v17, 192 // (op 30789) 
v_mul_lo_u32 v20, s18, v17 // (op 30789) 

v_add_u32 v20, v20, v16 // (op 30789) 


v_mul_lo_u32 v17, v20, s25 // (op 30789) most significant: low of low * high
v_mul_hi_u32 v21, v20, s24 // (op 30789) most significant: high of low * low
v_mul_lo_u32 v16, v20, s24 // (op 30789) least significant: low of low * low
v_add_u32 v17, v17, v21 // (op 30789) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 30789) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 30789) most significant half



v_lshlrev_b64 v[18:19], 2, v[16:17] // (op 30789) 

v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 30789) 

s_mul_i32 s1, s24, s29 // (op 30791) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 30791) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 30791) least significant
s_mul_hi_u32 s19, s24, s28 // (op 30791) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30791) most significant: sum
s_add_u32 s1, s1, s19 // (op 30791) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 30791) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 30791) 

s_mov_b32 s20, s26 // (op 30791) reg expression
s_mov_b32 s21, s27 // (op 30791) reg expression
s_mov_b32 s22, s0 // (op 30791) reg expression
s_mov_b32 s23, 131072 // (op 30791) call()

s_mul_hi_i32 s0, s16, s47 // (op 30834) 
v_lshrrev_b32 v18, 2, v1 // (op 30834) 
s_add_i32 s1, s0, s16 // (op 30834) 

v_and_b32 v19, -4, v1 // (op 30834) 
s_ashr_i32 s0, s1, 31 // (op 30834) 
v_sub_u32 v20, v1, v19 // (op 30834) 

s_cmp_eq_i32 s47, 0 // (op 30834) 
s_mov_b32 s18, scc // (op 30834) 
v_lshlrev_b32 v19, 5, v20 // (op 30834) 

s_cmp_eq_i32 s18, 0 // (op 30834) 
s_cselect_b32 s19, 0, -1 // (op 30834) 

v_mov_b32 v21, 0 // (op 30834) convert
v_mov_b32 v20, v19 // (op 30834) convert

s_and_b32 s18, s13, 31 // (op 30834) 
s_lshl_b32 s49, 1, s18 // (op 30834) 
s_add_i32 s50, s49, s19 // (op 30834) 


s_and_b32 s19, s0, s50 // (op 30834) 


s_add_i32 s0, s1, s19 // (op 30834) 


s_ashr_i32 s1, s0, s18 // (op 30834) 


s_lshl_b32 s0, s13, 24 // (op 30834) 
s_ashr_i32 s18, s0, 31 // (op 30834) 

s_xor_b32 s0, s1, s18 // (op 30834) 

s_sub_i32 s1, s0, s18 // (op 30834) 


s_mul_hi_i32 s0, s1, s14 // (op 30834) 
s_add_i32 s18, s0, s1 // (op 30834) 

s_ashr_i32 s0, s18, 31 // (op 30834) 
s_cmp_eq_i32 s14, 0 // (op 30834) 
s_mov_b32 s19, scc // (op 30834) 
s_cmp_eq_i32 s19, 0 // (op 30834) 
s_cselect_b32 s49, 0, -1 // (op 30834) 

s_and_b32 s19, s15, 31 // (op 30834) 
s_lshl_b32 s50, 1, s19 // (op 30834) 
s_add_i32 s51, s50, s49 // (op 30834) 


s_and_b32 s49, s0, s51 // (op 30834) 


s_add_i32 s0, s18, s49 // (op 30834) 


s_ashr_i32 s18, s0, s19 // (op 30834) 


s_lshl_b32 s0, s15, 24 // (op 30834) 
s_ashr_i32 s19, s0, 31 // (op 30834) 

s_xor_b32 s0, s18, s19 // (op 30834) 

s_sub_i32 s18, s0, s19 // (op 30834) 


s_add_u32 s54, s8, 256 // (op 30834) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 30834) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 30834) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 30834) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 8 // (op 30834) 

s_mul_i32 s0, s18, s50 // (op 30834) 

s_sub_i32 s49, s1, s0 // (op 30834) 


s_mul_hi_i32 s0, s16, s10 // (op 30834) 
s_add_i32 s1, s0, s16 // (op 30834) 

s_ashr_i32 s0, s1, 31 // (op 30834) 
s_cmp_eq_i32 s10, 0 // (op 30834) 
s_mov_b32 s18, scc // (op 30834) 
s_cmp_eq_i32 s18, 0 // (op 30834) 
s_cselect_b32 s19, 0, -1 // (op 30834) 

s_and_b32 s18, s12, 31 // (op 30834) 
s_lshl_b32 s52, 1, s18 // (op 30834) 
s_add_i32 s53, s52, s19 // (op 30834) 


s_and_b32 s19, s0, s53 // (op 30834) 


s_add_i32 s0, s1, s19 // (op 30834) 


s_ashr_i32 s1, s0, s18 // (op 30834) 


s_lshl_b32 s0, s12, 24 // (op 30834) 
s_ashr_i32 s18, s0, 31 // (op 30834) 

s_xor_b32 s0, s1, s18 // (op 30834) 

s_sub_i32 s1, s0, s18 // (op 30834) 


s_mul_i32 s0, s48, s50 // (op 30834) 

s_mul_i32 s18, s44, s0 // (op 30834) 

s_mul_i32 s0, s1, s18 // (op 30834) 

s_sub_u32 s18, s16, s0 // (op 30834) 

s_mul_hi_i32 s0, s18, s45 // (op 30834) 
s_add_i32 s19, s0, s18 // (op 30834) 


s_ashr_i32 s0, s19, 31 // (op 30834) 
s_cmp_eq_i32 s45, 0 // (op 30834) 
s_mov_b32 s18, scc // (op 30834) 
s_cmp_eq_i32 s18, 0 // (op 30834) 
s_cselect_b32 s50, 0, -1 // (op 30834) 

s_and_b32 s18, s11, 31 // (op 30834) 
s_lshl_b32 s51, 1, s18 // (op 30834) 
s_add_i32 s52, s51, s50 // (op 30834) 


s_and_b32 s50, s0, s52 // (op 30834) 


s_add_i32 s0, s19, s50 // (op 30834) 


s_ashr_i32 s19, s0, s18 // (op 30834) 


s_lshl_b32 s0, s11, 24 // (op 30834) 
s_ashr_i32 s18, s0, 31 // (op 30834) 

s_xor_b32 s0, s19, s18 // (op 30834) 

s_sub_i32 s19, s0, s18 // (op 30834) 


s_cmp_eq_i32 s1, 0 // (op 30834) 
s_mov_b32 s0, scc // (op 30834) 

s_cmp_eq_i32 s0, 0 // (op 30834) 
s_cselect_b32 s1, s19, s49 // (op 30834) 



v_lshl_add_u32 v19, s1, 8, v18 // (op 30834) 


v_mul_lo_u32 v23, v19, s31 // (op 30834) most significant: low of low * high
v_mul_hi_u32 v18, v19, s30 // (op 30834) most significant: high of low * low
v_mul_lo_u32 v22, v19, s30 // (op 30834) least significant: low of low * low
v_add_u32 v23, v23, v18 // (op 30834) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v22 // (op 30834) least significant half
v_addc_co_u32 v19, s[0:1], v21, v23, s[0:1] // (op 30834) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 30834) 


v_lshrrev_b64 v[20:21], 3, v[18:19] // (op 30834) 

s_mul_i32 s1, s30, s9 // (op 30836) most significant: low of low * high
s_mul_i32 s18, s31, s8 // (op 30836) most significant: low of high * low
s_mul_i32 s0, s30, s8 // (op 30836) least significant
s_mul_hi_u32 s19, s30, s8 // (op 30836) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30836) most significant: sum
s_add_u32 s1, s1, s19 // (op 30836) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 30836) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 30836) 

s_mov_b32 s52, s32 // (op 30836) reg expression
s_mov_b32 s53, s33 // (op 30836) reg expression
s_mov_b32 s54, s0 // (op 30836) reg expression
s_mov_b32 s55, 131072 // (op 30836) call()

s_mul_hi_i32 s0, s16, s47 // (op 30853) 
v_lshrrev_b32 v18, 2, v1 // (op 30853) 
s_add_i32 s1, s0, s16 // (op 30853) 

v_and_b32 v19, -4, v1 // (op 30853) 
s_ashr_i32 s0, s1, 31 // (op 30853) 
v_sub_u32 v22, v1, v19 // (op 30853) 

s_cmp_eq_i32 s47, 0 // (op 30853) 
s_mov_b32 s18, scc // (op 30853) 
v_lshlrev_b32 v19, 5, v22 // (op 30853) 

s_cmp_eq_i32 s18, 0 // (op 30853) 
s_cselect_b32 s19, 0, -1 // (op 30853) 

v_mov_b32 v23, 0 // (op 30853) convert
v_mov_b32 v22, v19 // (op 30853) convert

s_and_b32 s18, s13, 31 // (op 30853) 
s_lshl_b32 s49, 1, s18 // (op 30853) 
s_add_i32 s50, s49, s19 // (op 30853) 


s_and_b32 s19, s0, s50 // (op 30853) 


s_add_i32 s0, s1, s19 // (op 30853) 


s_ashr_i32 s1, s0, s18 // (op 30853) 


s_lshl_b32 s0, s13, 24 // (op 30853) 
s_ashr_i32 s18, s0, 31 // (op 30853) 

s_xor_b32 s0, s1, s18 // (op 30853) 

s_sub_i32 s1, s0, s18 // (op 30853) 


s_mul_hi_i32 s0, s1, s14 // (op 30853) 
s_add_i32 s18, s0, s1 // (op 30853) 

s_ashr_i32 s0, s18, 31 // (op 30853) 
s_cmp_eq_i32 s14, 0 // (op 30853) 
s_mov_b32 s19, scc // (op 30853) 
s_cmp_eq_i32 s19, 0 // (op 30853) 
s_cselect_b32 s49, 0, -1 // (op 30853) 

s_and_b32 s19, s15, 31 // (op 30853) 
s_lshl_b32 s50, 1, s19 // (op 30853) 
s_add_i32 s51, s50, s49 // (op 30853) 


s_and_b32 s49, s0, s51 // (op 30853) 


s_add_i32 s0, s18, s49 // (op 30853) 


s_ashr_i32 s18, s0, s19 // (op 30853) 


s_lshl_b32 s0, s15, 24 // (op 30853) 
s_ashr_i32 s19, s0, 31 // (op 30853) 

s_xor_b32 s0, s18, s19 // (op 30853) 

s_sub_i32 s18, s0, s19 // (op 30853) 


s_add_u32 s50, s8, 256 // (op 30853) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 30853) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 30853) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 30853) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 30853) 

s_mul_i32 s0, s18, s50 // (op 30853) 

s_sub_i32 s49, s1, s0 // (op 30853) 


s_mul_hi_i32 s0, s16, s10 // (op 30853) 
s_add_i32 s1, s0, s16 // (op 30853) 

s_ashr_i32 s0, s1, 31 // (op 30853) 
s_cmp_eq_i32 s10, 0 // (op 30853) 
s_mov_b32 s18, scc // (op 30853) 
s_cmp_eq_i32 s18, 0 // (op 30853) 
s_cselect_b32 s19, 0, -1 // (op 30853) 

s_and_b32 s18, s12, 31 // (op 30853) 
s_lshl_b32 s64, 1, s18 // (op 30853) 
s_add_i32 s65, s64, s19 // (op 30853) 


s_and_b32 s19, s0, s65 // (op 30853) 


s_add_i32 s0, s1, s19 // (op 30853) 


s_ashr_i32 s1, s0, s18 // (op 30853) 


s_lshl_b32 s0, s12, 24 // (op 30853) 
s_ashr_i32 s18, s0, 31 // (op 30853) 

s_xor_b32 s0, s1, s18 // (op 30853) 

s_sub_i32 s1, s0, s18 // (op 30853) 


s_mul_i32 s0, s48, s50 // (op 30853) 

s_mul_i32 s18, s44, s0 // (op 30853) 

s_mul_i32 s0, s1, s18 // (op 30853) 

s_sub_u32 s18, s16, s0 // (op 30853) 

s_mul_hi_i32 s0, s18, s45 // (op 30853) 
s_add_i32 s19, s0, s18 // (op 30853) 


s_ashr_i32 s0, s19, 31 // (op 30853) 
s_cmp_eq_i32 s45, 0 // (op 30853) 
s_mov_b32 s18, scc // (op 30853) 
s_cmp_eq_i32 s18, 0 // (op 30853) 
s_cselect_b32 s50, 0, -1 // (op 30853) 

s_and_b32 s18, s11, 31 // (op 30853) 
s_lshl_b32 s51, 1, s18 // (op 30853) 
s_add_i32 s64, s51, s50 // (op 30853) 


s_and_b32 s50, s0, s64 // (op 30853) 


s_add_i32 s0, s19, s50 // (op 30853) 


s_ashr_i32 s19, s0, s18 // (op 30853) 


s_lshl_b32 s0, s11, 24 // (op 30853) 
s_ashr_i32 s18, s0, 31 // (op 30853) 

s_xor_b32 s0, s19, s18 // (op 30853) 

s_sub_i32 s19, s0, s18 // (op 30853) 


s_cmp_eq_i32 s1, 0 // (op 30853) 
s_mov_b32 s0, scc // (op 30853) 

s_cmp_eq_i32 s0, 0 // (op 30853) 
s_cselect_b32 s1, s19, s49 // (op 30853) 



v_lshl_add_u32 v19, s1, 8, v18 // (op 30853) 


v_mul_lo_u32 v25, v19, s31 // (op 30853) most significant: low of low * high
v_mul_hi_u32 v18, v19, s30 // (op 30853) most significant: high of low * low
v_mul_lo_u32 v24, v19, s30 // (op 30853) least significant: low of low * low
v_add_u32 v25, v25, v18 // (op 30853) most significant: sum


v_add_co_u32 v18, s[0:1], v22, v24 // (op 30853) least significant half
v_addc_co_u32 v19, s[0:1], v23, v25, s[0:1] // (op 30853) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 30853) 


v_lshrrev_b64 v[22:23], 3, v[18:19] // (op 30853) 

s_mul_i32 s1, s30, s9 // (op 30855) most significant: low of low * high
s_mul_i32 s18, s31, s8 // (op 30855) most significant: low of high * low
s_mul_i32 s0, s30, s8 // (op 30855) least significant
s_mul_hi_u32 s19, s30, s8 // (op 30855) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30855) most significant: sum
s_add_u32 s1, s1, s19 // (op 30855) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 30855) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 30855) 

s_mov_b32 s52, s32 // (op 30855) reg expression
s_mov_b32 s53, s33 // (op 30855) reg expression
s_mov_b32 s54, s0 // (op 30855) reg expression
s_mov_b32 s55, 131072 // (op 30855) call()

s_mul_hi_i32 s0, s16, s47 // (op 31307) 
v_and_b32 v18, 63, v1 // (op 31307) 
s_add_i32 s1, s0, s16 // (op 31307) 

v_lshrrev_b32 v19, 4, v18 // (op 31307) 
s_ashr_i32 s0, s1, 31 // (op 31307) 
v_and_b32 v24, 3, v19 // (op 31307) 

s_cmp_eq_i32 s47, 0 // (op 31307) 
s_mov_b32 s18, scc // (op 31307) 
v_and_b32 v19, 15, v1 // (op 31307) 
s_cmp_eq_i32 s18, 0 // (op 31307) 
s_cselect_b32 s19, 0, -1 // (op 31307) 

v_lshl_add_u32 v25, v24, 4, v19 // (op 31307) 


s_and_b32 s18, s13, 31 // (op 31307) 
v_lshrrev_b32 v19, 8, v1 // (op 31307) 
s_lshl_b32 s49, 1, s18 // (op 31307) 
v_mul_lo_u32 v24, v19, 3 // (op 31307) 

s_add_i32 s50, s49, s19 // (op 31307) 


v_lshl_add_u32 v19, v24, 6, v25 // (op 31307) 


s_and_b32 s19, s0, s50 // (op 31307) 


v_lshrrev_b32 v24, 6, v18 // (op 31307) 

s_add_i32 s0, s1, s19 // (op 31307) 


v_lshlrev_b32 v18, 2, v24 // (op 31307) 

s_ashr_i32 s1, s0, s18 // (op 31307) 


s_lshl_b32 s0, s13, 24 // (op 31307) 
s_ashr_i32 s18, s0, 31 // (op 31307) 

s_xor_b32 s0, s1, s18 // (op 31307) 

s_sub_i32 s1, s0, s18 // (op 31307) 


s_mul_hi_i32 s0, s1, s14 // (op 31307) 
s_add_i32 s18, s0, s1 // (op 31307) 

s_ashr_i32 s0, s18, 31 // (op 31307) 
s_cmp_eq_i32 s14, 0 // (op 31307) 
s_mov_b32 s19, scc // (op 31307) 
s_cmp_eq_i32 s19, 0 // (op 31307) 
s_cselect_b32 s49, 0, -1 // (op 31307) 

s_and_b32 s19, s15, 31 // (op 31307) 
s_lshl_b32 s50, 1, s19 // (op 31307) 
s_add_i32 s51, s50, s49 // (op 31307) 


s_and_b32 s49, s0, s51 // (op 31307) 


s_add_i32 s0, s18, s49 // (op 31307) 


s_ashr_i32 s18, s0, s19 // (op 31307) 


s_lshl_b32 s0, s15, 24 // (op 31307) 
s_ashr_i32 s19, s0, 31 // (op 31307) 

s_xor_b32 s0, s18, s19 // (op 31307) 

s_sub_i32 s18, s0, s19 // (op 31307) 


s_mul_i32 s0, s1, s48 // (op 31307) 

s_sub_u32 s1, s16, s0 // (op 31307) 

s_mul_i32 s0, s18, s48 // (op 31307) 
s_add_u32 s0, s0, s1 // (op 31307) 


s_mul_hi_i32 s1, s16, s10 // (op 31307) 
s_add_i32 s18, s1, s16 // (op 31307) 

s_ashr_i32 s1, s18, 31 // (op 31307) 
s_cmp_eq_i32 s10, 0 // (op 31307) 
s_mov_b32 s19, scc // (op 31307) 
s_cmp_eq_i32 s19, 0 // (op 31307) 
s_cselect_b32 s49, 0, -1 // (op 31307) 

s_and_b32 s19, s12, 31 // (op 31307) 
s_lshl_b32 s50, 1, s19 // (op 31307) 
s_add_i32 s51, s50, s49 // (op 31307) 


s_and_b32 s49, s1, s51 // (op 31307) 


s_add_i32 s1, s18, s49 // (op 31307) 


s_ashr_i32 s18, s1, s19 // (op 31307) 


s_lshl_b32 s1, s12, 24 // (op 31307) 
s_ashr_i32 s19, s1, 31 // (op 31307) 

s_xor_b32 s1, s18, s19 // (op 31307) 

s_sub_i32 s18, s1, s19 // (op 31307) 


s_add_u32 s50, s8, 256 // (op 31307) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 31307) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 31307) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 31307) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 31307) 

s_mul_i32 s1, s48, s50 // (op 31307) 

s_mul_i32 s19, s44, s1 // (op 31307) 

s_mul_i32 s1, s18, s19 // (op 31307) 

s_sub_u32 s19, s16, s1 // (op 31307) 

s_mov_b32 s1, s19 // (op 31307) 
s_mul_hi_i32 s49, s1, s45 // (op 31307) 
s_add_i32 s50, s49, s1 // (op 31307) 


s_ashr_i32 s1, s50, 31 // (op 31307) 
s_cmp_eq_i32 s45, 0 // (op 31307) 
s_mov_b32 s49, scc // (op 31307) 
s_cmp_eq_i32 s49, 0 // (op 31307) 
s_cselect_b32 s51, 0, -1 // (op 31307) 

s_and_b32 s49, s11, 31 // (op 31307) 
s_lshl_b32 s64, 1, s49 // (op 31307) 
s_add_i32 s65, s64, s51 // (op 31307) 


s_and_b32 s51, s1, s65 // (op 31307) 


s_add_i32 s1, s50, s51 // (op 31307) 


s_ashr_i32 s50, s1, s49 // (op 31307) 


s_lshl_b32 s1, s11, 24 // (op 31307) 
s_ashr_i32 s49, s1, 31 // (op 31307) 

s_xor_b32 s1, s50, s49 // (op 31307) 

s_sub_i32 s50, s1, s49 // (op 31307) 


s_mul_i32 s1, s46, s48 // (op 31307) 
s_sub_i32 s49, s6, s1 // (op 31307) 
s_mul_i32 s51, s50, s49 // (op 31307) 


s_sub_u32 s49, s19, s51 // (op 31307) 


s_add_u32 s19, s49, s1 // (op 31307) 


s_cmp_eq_i32 s18, 0 // (op 31307) 
s_mov_b32 s1, scc // (op 31307) 

s_cmp_eq_i32 s1, 0 // (op 31307) 
s_cselect_b32 s18, s19, s0 // (op 31307) 



v_mov_b32 v24, 192 // (op 31307) 
v_mul_lo_u32 v25, s18, v24 // (op 31307) 

v_add_u32 v25, v25, v19 // (op 31307) 


v_mul_lo_u32 v19, v25, s34 // (op 31307) 
v_add_u32 v19, v19, v18 // (op 31307) 


s_mul_i32 s1, s34, s29 // (op 31309) most significant: low of low * high
s_mul_i32 s18, s35, s28 // (op 31309) most significant: low of high * low
s_mul_i32 s0, s34, s28 // (op 31309) least significant
s_mul_hi_u32 s19, s34, s28 // (op 31309) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 31309) most significant: sum
s_add_u32 s1, s1, s19 // (op 31309) most significant: sum


s_mov_b32 s64, s36 // (op 31309) reg expression
s_mov_b32 s65, s37 // (op 31309) reg expression
s_mov_b32 s66, s0 // (op 31309) reg expression
s_mov_b32 s67, 131072 // (op 31309) call()

s_mul_hi_i32 s0, s16, s47 // (op 31328) 
v_and_b32 v18, 63, v1 // (op 31328) 
s_add_i32 s1, s0, s16 // (op 31328) 

v_lshrrev_b32 v24, 4, v18 // (op 31328) 
s_ashr_i32 s0, s1, 31 // (op 31328) 
v_and_b32 v25, 3, v24 // (op 31328) 

s_cmp_eq_i32 s47, 0 // (op 31328) 
s_mov_b32 s18, scc // (op 31328) 
v_and_b32 v24, 15, v1 // (op 31328) 
s_cmp_eq_i32 s18, 0 // (op 31328) 
s_cselect_b32 s19, 0, -1 // (op 31328) 

v_lshl_add_u32 v26, v25, 4, v24 // (op 31328) 


s_and_b32 s18, s13, 31 // (op 31328) 
v_lshrrev_b32 v24, 8, v1 // (op 31328) 
s_lshl_b32 s49, 1, s18 // (op 31328) 
v_mul_lo_u32 v25, v24, 3 // (op 31328) 

s_add_i32 s50, s49, s19 // (op 31328) 


v_lshl_add_u32 v24, v25, 6, v26 // (op 31328) 


s_and_b32 s19, s0, s50 // (op 31328) 


v_lshrrev_b32 v25, 6, v18 // (op 31328) 

s_add_i32 s0, s1, s19 // (op 31328) 


v_lshlrev_b32 v18, 2, v25 // (op 31328) 

s_ashr_i32 s1, s0, s18 // (op 31328) 


s_lshl_b32 s0, s13, 24 // (op 31328) 
s_ashr_i32 s18, s0, 31 // (op 31328) 

s_xor_b32 s0, s1, s18 // (op 31328) 

s_sub_i32 s1, s0, s18 // (op 31328) 


s_mul_hi_i32 s0, s1, s14 // (op 31328) 
s_add_i32 s18, s0, s1 // (op 31328) 

s_ashr_i32 s0, s18, 31 // (op 31328) 
s_cmp_eq_i32 s14, 0 // (op 31328) 
s_mov_b32 s19, scc // (op 31328) 
s_cmp_eq_i32 s19, 0 // (op 31328) 
s_cselect_b32 s49, 0, -1 // (op 31328) 

s_and_b32 s19, s15, 31 // (op 31328) 
s_lshl_b32 s50, 1, s19 // (op 31328) 
s_add_i32 s51, s50, s49 // (op 31328) 


s_and_b32 s49, s0, s51 // (op 31328) 


s_add_i32 s0, s18, s49 // (op 31328) 


s_ashr_i32 s18, s0, s19 // (op 31328) 


s_lshl_b32 s0, s15, 24 // (op 31328) 
s_ashr_i32 s19, s0, 31 // (op 31328) 

s_xor_b32 s0, s18, s19 // (op 31328) 

s_sub_i32 s18, s0, s19 // (op 31328) 


s_mul_i32 s0, s1, s48 // (op 31328) 

s_sub_u32 s1, s16, s0 // (op 31328) 

s_mul_i32 s0, s18, s48 // (op 31328) 
s_add_u32 s0, s0, s1 // (op 31328) 


s_mul_hi_i32 s1, s16, s10 // (op 31328) 
s_add_i32 s18, s1, s16 // (op 31328) 

s_ashr_i32 s1, s18, 31 // (op 31328) 
s_cmp_eq_i32 s10, 0 // (op 31328) 
s_mov_b32 s19, scc // (op 31328) 
s_cmp_eq_i32 s19, 0 // (op 31328) 
s_cselect_b32 s49, 0, -1 // (op 31328) 

s_and_b32 s19, s12, 31 // (op 31328) 
s_lshl_b32 s50, 1, s19 // (op 31328) 
s_add_i32 s51, s50, s49 // (op 31328) 


s_and_b32 s49, s1, s51 // (op 31328) 


s_add_i32 s1, s18, s49 // (op 31328) 


s_ashr_i32 s18, s1, s19 // (op 31328) 


s_lshl_b32 s1, s12, 24 // (op 31328) 
s_ashr_i32 s19, s1, 31 // (op 31328) 

s_xor_b32 s1, s18, s19 // (op 31328) 

s_sub_i32 s18, s1, s19 // (op 31328) 


s_add_u32 s50, s8, 256 // (op 31328) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 31328) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 31328) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 31328) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 31328) 

s_mul_i32 s1, s48, s50 // (op 31328) 

s_mul_i32 s19, s44, s1 // (op 31328) 

s_mul_i32 s1, s18, s19 // (op 31328) 

s_sub_u32 s19, s16, s1 // (op 31328) 

s_mov_b32 s1, s19 // (op 31328) 
s_mul_hi_i32 s49, s1, s45 // (op 31328) 
s_add_i32 s50, s49, s1 // (op 31328) 


s_ashr_i32 s1, s50, 31 // (op 31328) 
s_cmp_eq_i32 s45, 0 // (op 31328) 
s_mov_b32 s49, scc // (op 31328) 
s_cmp_eq_i32 s49, 0 // (op 31328) 
s_cselect_b32 s51, 0, -1 // (op 31328) 

s_and_b32 s49, s11, 31 // (op 31328) 
s_lshl_b32 s68, 1, s49 // (op 31328) 
s_add_i32 s69, s68, s51 // (op 31328) 


s_and_b32 s51, s1, s69 // (op 31328) 


s_add_i32 s1, s50, s51 // (op 31328) 


s_ashr_i32 s50, s1, s49 // (op 31328) 


s_lshl_b32 s1, s11, 24 // (op 31328) 
s_ashr_i32 s49, s1, 31 // (op 31328) 

s_xor_b32 s1, s50, s49 // (op 31328) 

s_sub_i32 s50, s1, s49 // (op 31328) 


s_mul_i32 s1, s46, s48 // (op 31328) 
s_sub_i32 s49, s6, s1 // (op 31328) 
s_mul_i32 s51, s50, s49 // (op 31328) 


s_sub_u32 s49, s19, s51 // (op 31328) 


s_add_u32 s19, s49, s1 // (op 31328) 


s_cmp_eq_i32 s18, 0 // (op 31328) 
s_mov_b32 s1, scc // (op 31328) 

s_cmp_eq_i32 s1, 0 // (op 31328) 
s_cselect_b32 s18, s19, s0 // (op 31328) 



v_mov_b32 v25, 192 // (op 31328) 
v_mul_lo_u32 v26, s18, v25 // (op 31328) 

v_add_u32 v26, v26, v24 // (op 31328) 


v_mul_lo_u32 v24, v26, s34 // (op 31328) 
v_add_u32 v24, v24, v18 // (op 31328) 


s_mul_i32 s1, s34, s29 // (op 31330) most significant: low of low * high
s_mul_i32 s18, s35, s28 // (op 31330) most significant: low of high * low
s_mul_i32 s0, s34, s28 // (op 31330) least significant
s_mul_hi_u32 s19, s34, s28 // (op 31330) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 31330) most significant: sum
s_add_u32 s1, s1, s19 // (op 31330) most significant: sum


s_mov_b32 s64, s36 // (op 31330) reg expression
s_mov_b32 s65, s37 // (op 31330) reg expression
s_mov_b32 s66, s0 // (op 31330) reg expression
s_mov_b32 s67, 131072 // (op 31330) call()

s_mul_hi_i32 s0, s16, s47 // (op 31359) 
v_and_b32 v18, 63, v1 // (op 31359) 
s_add_i32 s1, s0, s16 // (op 31359) 

v_lshrrev_b32 v25, 4, v18 // (op 31359) 
s_ashr_i32 s0, s1, 31 // (op 31359) 
v_and_b32 v26, 3, v25 // (op 31359) 

s_cmp_eq_i32 s47, 0 // (op 31359) 
s_mov_b32 s18, scc // (op 31359) 
v_and_b32 v25, 15, v1 // (op 31359) 
s_cmp_eq_i32 s18, 0 // (op 31359) 
s_cselect_b32 s19, 0, -1 // (op 31359) 

v_lshl_add_u32 v27, v26, 4, v25 // (op 31359) 


s_and_b32 s18, s13, 31 // (op 31359) 
v_lshrrev_b32 v25, 6, v1 // (op 31359) 
s_lshl_b32 s49, 1, s18 // (op 31359) 
v_and_b32 v26, 3, v25 // (op 31359) 

s_add_i32 s50, s49, s19 // (op 31359) 


v_lshl_add_u32 v25, v26, 6, v27 // (op 31359) 


s_and_b32 s19, s0, s50 // (op 31359) 


v_lshrrev_b32 v26, 6, v18 // (op 31359) 

s_add_i32 s0, s1, s19 // (op 31359) 


s_ashr_i32 s1, s0, s18 // (op 31359) 


s_lshl_b32 s0, s13, 24 // (op 31359) 
s_ashr_i32 s18, s0, 31 // (op 31359) 

s_xor_b32 s0, s1, s18 // (op 31359) 

s_sub_i32 s1, s0, s18 // (op 31359) 


s_mul_hi_i32 s0, s1, s14 // (op 31359) 
s_add_i32 s18, s0, s1 // (op 31359) 

s_ashr_i32 s0, s18, 31 // (op 31359) 
s_cmp_eq_i32 s14, 0 // (op 31359) 
s_mov_b32 s19, scc // (op 31359) 
s_cmp_eq_i32 s19, 0 // (op 31359) 
s_cselect_b32 s49, 0, -1 // (op 31359) 

s_and_b32 s19, s15, 31 // (op 31359) 
s_lshl_b32 s50, 1, s19 // (op 31359) 
s_add_i32 s51, s50, s49 // (op 31359) 


s_and_b32 s49, s0, s51 // (op 31359) 


s_add_i32 s0, s18, s49 // (op 31359) 


s_ashr_i32 s18, s0, s19 // (op 31359) 


s_lshl_b32 s0, s15, 24 // (op 31359) 
s_ashr_i32 s19, s0, 31 // (op 31359) 

s_xor_b32 s0, s18, s19 // (op 31359) 

s_sub_i32 s18, s0, s19 // (op 31359) 


s_add_u32 s50, s8, 256 // (op 31359) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 31359) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 31359) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 31359) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 31359) 

s_mul_i32 s0, s18, s50 // (op 31359) 

s_sub_i32 s49, s1, s0 // (op 31359) 


s_mul_hi_i32 s0, s16, s10 // (op 31359) 
s_add_i32 s1, s0, s16 // (op 31359) 

s_ashr_i32 s0, s1, 31 // (op 31359) 
s_cmp_eq_i32 s10, 0 // (op 31359) 
s_mov_b32 s18, scc // (op 31359) 
s_cmp_eq_i32 s18, 0 // (op 31359) 
s_cselect_b32 s19, 0, -1 // (op 31359) 

s_and_b32 s18, s12, 31 // (op 31359) 
s_lshl_b32 s68, 1, s18 // (op 31359) 
s_add_i32 s69, s68, s19 // (op 31359) 


s_and_b32 s19, s0, s69 // (op 31359) 


s_add_i32 s0, s1, s19 // (op 31359) 


s_ashr_i32 s1, s0, s18 // (op 31359) 


s_lshl_b32 s0, s12, 24 // (op 31359) 
s_ashr_i32 s18, s0, 31 // (op 31359) 

s_xor_b32 s0, s1, s18 // (op 31359) 

s_sub_i32 s1, s0, s18 // (op 31359) 


s_mul_i32 s0, s48, s50 // (op 31359) 

s_mul_i32 s18, s44, s0 // (op 31359) 

s_mul_i32 s0, s1, s18 // (op 31359) 

s_sub_u32 s18, s16, s0 // (op 31359) 

s_mul_hi_i32 s0, s18, s45 // (op 31359) 
s_add_i32 s19, s0, s18 // (op 31359) 


s_ashr_i32 s0, s19, 31 // (op 31359) 
s_cmp_eq_i32 s45, 0 // (op 31359) 
s_mov_b32 s18, scc // (op 31359) 
s_cmp_eq_i32 s18, 0 // (op 31359) 
s_cselect_b32 s50, 0, -1 // (op 31359) 

s_and_b32 s18, s11, 31 // (op 31359) 
s_lshl_b32 s51, 1, s18 // (op 31359) 
s_add_i32 s68, s51, s50 // (op 31359) 


s_and_b32 s50, s0, s68 // (op 31359) 


s_add_i32 s0, s19, s50 // (op 31359) 


s_ashr_i32 s19, s0, s18 // (op 31359) 


s_lshl_b32 s0, s11, 24 // (op 31359) 
s_ashr_i32 s18, s0, 31 // (op 31359) 

s_xor_b32 s0, s19, s18 // (op 31359) 

s_sub_i32 s19, s0, s18 // (op 31359) 


s_cmp_eq_i32 s1, 0 // (op 31359) 
s_mov_b32 s0, scc // (op 31359) 

s_cmp_eq_i32 s0, 0 // (op 31359) 
s_cselect_b32 s1, s19, s49 // (op 31359) 



v_lshl_add_u32 v18, s1, 8, v25 // (op 31359) 


v_mul_lo_u32 v25, v18, s38 // (op 31359) 

v_lshl_add_u32 v18, v26, 2, v25 // (op 31359) 


s_mul_i32 s1, s38, s9 // (op 31361) most significant: low of low * high
s_mul_i32 s18, s39, s8 // (op 31361) most significant: low of high * low
s_mul_i32 s0, s38, s8 // (op 31361) least significant
s_mul_hi_u32 s19, s38, s8 // (op 31361) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 31361) most significant: sum
s_add_u32 s1, s1, s19 // (op 31361) most significant: sum


s_mov_b32 s68, s40 // (op 31361) reg expression
s_mov_b32 s69, s41 // (op 31361) reg expression
s_mov_b32 s70, s0 // (op 31361) reg expression
s_mov_b32 s71, 131072 // (op 31361) call()

s_mul_hi_i32 s0, s16, s47 // (op 31380) 
v_and_b32 v25, 63, v1 // (op 31380) 
s_add_i32 s1, s0, s16 // (op 31380) 

v_lshrrev_b32 v26, 4, v25 // (op 31380) 
s_ashr_i32 s0, s1, 31 // (op 31380) 
v_and_b32 v27, 3, v26 // (op 31380) 

s_cmp_eq_i32 s47, 0 // (op 31380) 
s_mov_b32 s18, scc // (op 31380) 
v_and_b32 v26, 15, v1 // (op 31380) 
s_cmp_eq_i32 s18, 0 // (op 31380) 
s_cselect_b32 s19, 0, -1 // (op 31380) 

v_lshl_add_u32 v28, v27, 4, v26 // (op 31380) 


s_and_b32 s18, s13, 31 // (op 31380) 
v_lshrrev_b32 v26, 6, v1 // (op 31380) 
s_lshl_b32 s49, 1, s18 // (op 31380) 
v_and_b32 v27, 3, v26 // (op 31380) 

s_add_i32 s50, s49, s19 // (op 31380) 


v_lshl_add_u32 v26, v27, 6, v28 // (op 31380) 


s_and_b32 s19, s0, s50 // (op 31380) 


v_lshrrev_b32 v27, 6, v25 // (op 31380) 

s_add_i32 s0, s1, s19 // (op 31380) 


s_ashr_i32 s1, s0, s18 // (op 31380) 


s_lshl_b32 s0, s13, 24 // (op 31380) 
s_ashr_i32 s18, s0, 31 // (op 31380) 

s_xor_b32 s0, s1, s18 // (op 31380) 

s_sub_i32 s1, s0, s18 // (op 31380) 


s_mul_hi_i32 s0, s1, s14 // (op 31380) 
s_add_i32 s18, s0, s1 // (op 31380) 

s_ashr_i32 s0, s18, 31 // (op 31380) 
s_cmp_eq_i32 s14, 0 // (op 31380) 
s_mov_b32 s19, scc // (op 31380) 
s_cmp_eq_i32 s19, 0 // (op 31380) 
s_cselect_b32 s49, 0, -1 // (op 31380) 

s_and_b32 s19, s15, 31 // (op 31380) 
s_lshl_b32 s50, 1, s19 // (op 31380) 
s_add_i32 s51, s50, s49 // (op 31380) 


s_and_b32 s49, s0, s51 // (op 31380) 


s_add_i32 s0, s18, s49 // (op 31380) 


s_ashr_i32 s18, s0, s19 // (op 31380) 


s_lshl_b32 s0, s15, 24 // (op 31380) 
s_ashr_i32 s19, s0, 31 // (op 31380) 

s_xor_b32 s0, s18, s19 // (op 31380) 

s_sub_i32 s18, s0, s19 // (op 31380) 


s_add_u32 s50, s8, 256 // (op 31380) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 31380) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 31380) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 31380) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 8 // (op 31380) 

s_mul_i32 s0, s18, s50 // (op 31380) 

s_sub_i32 s49, s1, s0 // (op 31380) 


s_mul_hi_i32 s0, s16, s10 // (op 31380) 
s_add_i32 s1, s0, s16 // (op 31380) 

s_ashr_i32 s0, s1, 31 // (op 31380) 
s_cmp_eq_i32 s10, 0 // (op 31380) 
s_mov_b32 s18, scc // (op 31380) 
s_cmp_eq_i32 s18, 0 // (op 31380) 
s_cselect_b32 s19, 0, -1 // (op 31380) 

s_and_b32 s18, s12, 31 // (op 31380) 
s_lshl_b32 s72, 1, s18 // (op 31380) 
s_add_i32 s73, s72, s19 // (op 31380) 


s_and_b32 s19, s0, s73 // (op 31380) 


s_add_i32 s0, s1, s19 // (op 31380) 


s_ashr_i32 s1, s0, s18 // (op 31380) 


s_lshl_b32 s0, s12, 24 // (op 31380) 
s_ashr_i32 s18, s0, 31 // (op 31380) 

s_xor_b32 s0, s1, s18 // (op 31380) 

s_sub_i32 s1, s0, s18 // (op 31380) 


s_mul_i32 s0, s48, s50 // (op 31380) 

s_mul_i32 s18, s44, s0 // (op 31380) 

s_mul_i32 s0, s1, s18 // (op 31380) 

s_sub_u32 s18, s16, s0 // (op 31380) 

s_mul_hi_i32 s0, s18, s45 // (op 31380) 
s_add_i32 s19, s0, s18 // (op 31380) 


s_ashr_i32 s0, s19, 31 // (op 31380) 
s_cmp_eq_i32 s45, 0 // (op 31380) 
s_mov_b32 s18, scc // (op 31380) 
s_cmp_eq_i32 s18, 0 // (op 31380) 
s_cselect_b32 s50, 0, -1 // (op 31380) 

s_and_b32 s18, s11, 31 // (op 31380) 
s_lshl_b32 s51, 1, s18 // (op 31380) 
s_add_i32 s72, s51, s50 // (op 31380) 


s_and_b32 s50, s0, s72 // (op 31380) 


s_add_i32 s0, s19, s50 // (op 31380) 


s_ashr_i32 s19, s0, s18 // (op 31380) 


s_lshl_b32 s0, s11, 24 // (op 31380) 
s_ashr_i32 s18, s0, 31 // (op 31380) 

s_xor_b32 s0, s19, s18 // (op 31380) 

s_sub_i32 s19, s0, s18 // (op 31380) 


s_cmp_eq_i32 s1, 0 // (op 31380) 
s_mov_b32 s0, scc // (op 31380) 

s_cmp_eq_i32 s0, 0 // (op 31380) 
s_cselect_b32 s1, s19, s49 // (op 31380) 



v_lshl_add_u32 v25, s1, 8, v26 // (op 31380) 


v_mul_lo_u32 v26, v25, s38 // (op 31380) 

v_lshl_add_u32 v25, v27, 2, v26 // (op 31380) 


s_mul_i32 s1, s38, s9 // (op 31382) most significant: low of low * high
s_mul_i32 s18, s39, s8 // (op 31382) most significant: low of high * low
s_mul_i32 s0, s38, s8 // (op 31382) least significant
s_mul_hi_u32 s19, s38, s8 // (op 31382) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 31382) most significant: sum
s_add_u32 s1, s1, s19 // (op 31382) most significant: sum


s_mov_b32 s68, s40 // (op 31382) reg expression
s_mov_b32 s69, s41 // (op 31382) reg expression
s_mov_b32 s70, s0 // (op 31382) reg expression
s_mov_b32 s71, 131072 // (op 31382) call()

v_and_b32 v26, -4, v1 // (op 30981) 
v_sub_u32 v27, v1, v26 // (op 30981) 

v_lshlrev_b32 v26, 5, v27 // (op 30981) 

v_lshrrev_b32 v27, 2, v1 // (op 30981) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 30981) 


v_lshlrev_b32 v26, 2, v28 // (op 30981) 

v_lshrrev_b32 v27, 3, v26 // (op 30981) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 30981) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[30:31], 6 // (op 30664) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 30664) 

v_lshrrev_b64 v[26:27], 3, s[50:51] // (op 30664) 


v_mov_b64 v[28:29], v[20:21] // (op 30664) 
s_mov_b32 m0, s0 // (op 30664) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30664) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 30664) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 30664) most significant half

s_add_u32 m0, m0, 4096 // (op 30664) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30664) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 30664) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 30664) most significant half

s_add_u32 m0, m0, 4096 // (op 30664) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30664) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 30664) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 30664) most significant half

s_add_u32 m0, m0, 4096 // (op 30664) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30664) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 30924) 
v_sub_u32 v27, v1, v26 // (op 30924) 

v_lshlrev_b32 v26, 5, v27 // (op 30924) 

v_lshrrev_b32 v27, 2, v1 // (op 30924) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 30924) 


v_lshlrev_b32 v26, 2, v28 // (op 30924) 

v_lshrrev_b32 v27, 3, v26 // (op 30924) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 30924) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 30669) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 30669) 

v_lshrrev_b64 v[26:27], 3, s[50:51] // (op 30669) 


v_mov_b64 v[28:29], v[14:15] // (op 30669) 
s_add_u32 s1, 16384, s0 // (op 30669) 
s_mov_b32 m0, s1 // (op 30669) reg expression
buffer_load_dwordx4 v28, s[20:23], 0 offen offset: 0 lds // (op 30669) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 30669) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 30669) most significant half

s_add_u32 m0, m0, 4096 // (op 30669) 
buffer_load_dwordx4 v28, s[20:23], 0 offen offset: 0 lds // (op 30669) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 30669) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 30669) most significant half

s_add_u32 m0, m0, 4096 // (op 30669) 
buffer_load_dwordx4 v28, s[20:23], 0 offen offset: 0 lds // (op 30669) Load value direct to lds






v_mov_b32 v26, v18 // (op 74) reg expression

buffer_load_dwordx4 v[28:31], v26, s[68:71], 0 offen // (op 74) Load value




s_lshl_b32 s0, s34, 6 // (op 4381) 
v_add_u32 v18, v19, s0 // (op 4381) 


buffer_load_dwordx4 v[32:35], v18, s[64:67], 0 offen // (op 4381) Load value



s_lshl_b32 s0, s34, 7 // (op 8389) 
v_add_u32 v18, v19, s0 // (op 8389) 


buffer_load_dwordx4 v[36:39], v18, s[64:67], 0 offen // (op 8389) Load value



v_mov_b32 v18, v19 // (op 70) reg expression

buffer_load_dwordx4 v[40:43], v18, s[64:67], 0 offen // (op 70) Load value


v_and_b32 v18, -4, v1 // (op 31092) 
v_sub_u32 v19, v1, v18 // (op 31092) 

v_lshlrev_b32 v18, 5, v19 // (op 31092) 

v_lshrrev_b32 v19, 2, v1 // (op 31092) 
v_lshl_add_u32 v26, v19, 7, v18 // (op 31092) 


v_lshlrev_b32 v27, 2, v26 // (op 31092) 

v_lshrrev_b32 v26, 3, v27 // (op 31092) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31092) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[30:31], 6 // (op 30674) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 30674) 

v_lshrrev_b64 v[18:19], 3, s[50:51] // (op 30674) 


v_mov_b64 v[26:27], v[20:21] // (op 30674) 
s_add_u32 s1, 28672, s0 // (op 30674) 
s_mov_b32 m0, s1 // (op 30674) reg expression
buffer_load_dwordx4 v26, s[52:55], 64 offen offset: 0 lds // (op 30674) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v18 // (op 30674) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 30674) most significant half

s_add_u32 m0, m0, 4096 // (op 30674) 
buffer_load_dwordx4 v26, s[52:55], 64 offen offset: 0 lds // (op 30674) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v18 // (op 30674) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 30674) most significant half

s_add_u32 m0, m0, 4096 // (op 30674) 
buffer_load_dwordx4 v26, s[52:55], 64 offen offset: 0 lds // (op 30674) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v18 // (op 30674) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 30674) most significant half

s_add_u32 m0, m0, 4096 // (op 30674) 
buffer_load_dwordx4 v26, s[52:55], 64 offen offset: 0 lds // (op 30674) Load value direct to lds




v_and_b32 v18, -4, v1 // (op 31048) 
v_sub_u32 v19, v1, v18 // (op 31048) 

v_lshlrev_b32 v18, 5, v19 // (op 31048) 

v_lshrrev_b32 v19, 2, v1 // (op 31048) 
v_lshl_add_u32 v26, v19, 7, v18 // (op 31048) 


v_lshlrev_b32 v27, 2, v26 // (op 31048) 

v_lshrrev_b32 v26, 3, v27 // (op 31048) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31048) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 30676) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 30676) 

v_lshrrev_b64 v[18:19], 3, s[50:51] // (op 30676) 


v_mov_b64 v[26:27], v[14:15] // (op 30676) 
s_add_u32 s1, 45056, s0 // (op 30676) 
s_mov_b32 m0, s1 // (op 30676) reg expression
buffer_load_dwordx4 v26, s[20:23], 64 offen offset: 0 lds // (op 30676) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v18 // (op 30676) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 30676) most significant half

s_add_u32 m0, m0, 4096 // (op 30676) 
buffer_load_dwordx4 v26, s[20:23], 64 offen offset: 0 lds // (op 30676) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v18 // (op 30676) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 30676) most significant half

s_add_u32 m0, m0, 4096 // (op 30676) 
buffer_load_dwordx4 v26, s[20:23], 64 offen offset: 0 lds // (op 30676) Load value direct to lds




v_and_b32 v18, -4, v1 // (op 31180) 
v_sub_u32 v19, v1, v18 // (op 31180) 

v_lshlrev_b32 v18, 5, v19 // (op 31180) 

v_lshrrev_b32 v19, 2, v1 // (op 31180) 
v_lshl_add_u32 v26, v19, 7, v18 // (op 31180) 


v_lshlrev_b32 v27, 2, v26 // (op 31180) 

v_lshrrev_b32 v26, 3, v27 // (op 31180) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31180) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[30:31], 6 // (op 30678) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 30678) 

v_lshrrev_b64 v[18:19], 3, s[50:51] // (op 30678) 


v_mov_b64 v[26:27], v[20:21] // (op 30678) 
s_add_u32 s1, 57344, s0 // (op 30678) 
s_mov_b32 m0, s1 // (op 30678) reg expression
s_mov_b32 s18, 128 // (op 30678) call()
buffer_load_dwordx4 v26, s[52:55], s18 offen offset: 0 lds // (op 30678) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 30678) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 30678) most significant half

s_add_u32 m0, m0, 4096 // (op 30678) 
buffer_load_dwordx4 v26, s[52:55], s18 offen offset: 0 lds // (op 30678) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 30678) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 30678) most significant half

s_add_u32 m0, m0, 4096 // (op 30678) 
buffer_load_dwordx4 v26, s[52:55], s18 offen offset: 0 lds // (op 30678) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 30678) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 30678) most significant half

s_add_u32 m0, m0, 4096 // (op 30678) 
buffer_load_dwordx4 v26, s[52:55], s18 offen offset: 0 lds // (op 30678) Load value direct to lds





v_and_b32 v18, -4, v1 // (op 31136) 
v_sub_u32 v19, v1, v18 // (op 31136) 

v_lshlrev_b32 v18, 5, v19 // (op 31136) 

v_lshrrev_b32 v19, 2, v1 // (op 31136) 
v_lshl_add_u32 v26, v19, 7, v18 // (op 31136) 


v_lshlrev_b32 v27, 2, v26 // (op 31136) 

v_lshrrev_b32 v26, 3, v27 // (op 31136) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31136) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 30680) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 30680) 

v_lshrrev_b64 v[18:19], 3, s[50:51] // (op 30680) 


v_mov_b64 v[26:27], v[14:15] // (op 30680) 
s_add_u32 s1, 73728, s0 // (op 30680) 
s_mov_b32 m0, s1 // (op 30680) reg expression
s_mov_b32 s18, 128 // (op 30680) call()
buffer_load_dwordx4 v26, s[20:23], s18 offen offset: 0 lds // (op 30680) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 30680) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 30680) most significant half

s_add_u32 m0, m0, 4096 // (op 30680) 
buffer_load_dwordx4 v26, s[20:23], s18 offen offset: 0 lds // (op 30680) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 30680) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 30680) most significant half

s_add_u32 m0, m0, 4096 // (op 30680) 
buffer_load_dwordx4 v26, s[20:23], s18 offen offset: 0 lds // (op 30680) Load value direct to lds





v_and_b32 v18, -4, v1 // (op 31268) 
v_sub_u32 v19, v1, v18 // (op 31268) 

v_lshlrev_b32 v18, 5, v19 // (op 31268) 

v_lshrrev_b32 v19, 2, v1 // (op 31268) 
v_lshl_add_u32 v26, v19, 7, v18 // (op 31268) 


v_lshlrev_b32 v27, 2, v26 // (op 31268) 

v_lshrrev_b32 v26, 3, v27 // (op 31268) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31268) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[30:31], 6 // (op 30682) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 30682) 

v_lshrrev_b64 v[18:19], 3, s[50:51] // (op 30682) 


v_mov_b64 v[26:27], v[20:21] // (op 30682) 
s_add_u32 s1, 86016, s0 // (op 30682) 
s_mov_b32 m0, s1 // (op 30682) reg expression
s_mov_b32 s18, 192 // (op 30682) call()
buffer_load_dwordx4 v26, s[52:55], s18 offen offset: 0 lds // (op 30682) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 30682) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 30682) most significant half

s_add_u32 m0, m0, 4096 // (op 30682) 
buffer_load_dwordx4 v26, s[52:55], s18 offen offset: 0 lds // (op 30682) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 30682) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 30682) most significant half

s_add_u32 m0, m0, 4096 // (op 30682) 
buffer_load_dwordx4 v26, s[52:55], s18 offen offset: 0 lds // (op 30682) Load value direct to lds
v_add_co_u32 v26, s[50:51], v26, v18 // (op 30682) least significant half
v_addc_co_u32 v27, s[50:51], 0, v27, s[50:51] // (op 30682) most significant half

s_add_u32 m0, m0, 4096 // (op 30682) 
buffer_load_dwordx4 v26, s[52:55], s18 offen offset: 0 lds // (op 30682) Load value direct to lds






v_and_b32 v18, -4, v1 // (op 31224) 
v_sub_u32 v19, v1, v18 // (op 31224) 

v_lshlrev_b32 v18, 5, v19 // (op 31224) 

v_lshrrev_b32 v19, 2, v1 // (op 31224) 
v_lshl_add_u32 v20, v19, 7, v18 // (op 31224) 


v_lshlrev_b32 v21, 2, v20 // (op 31224) 

v_lshrrev_b32 v18, 3, v21 // (op 31224) 

s_nop 0
v_readfirstlane_b32 s0, v18 // (op 31224) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[24:25], 6 // (op 30684) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 30684) 

v_lshrrev_b64 v[26:27], 3, s[50:51] // (op 30684) 


v_mov_b64 v[18:19], v[14:15] // (op 30684) 
s_add_u32 s1, 102400, s0 // (op 30684) 
s_mov_b32 m0, s1 // (op 30684) reg expression
s_mov_b32 s18, 192 // (op 30684) call()
buffer_load_dwordx4 v18, s[20:23], s18 offen offset: 0 lds // (op 30684) Load value direct to lds
v_add_co_u32 v18, s[50:51], v18, v26 // (op 30684) least significant half
v_addc_co_u32 v19, s[50:51], 0, v19, s[50:51] // (op 30684) most significant half

s_add_u32 m0, m0, 4096 // (op 30684) 
buffer_load_dwordx4 v18, s[20:23], s18 offen offset: 0 lds // (op 30684) Load value direct to lds
v_add_co_u32 v18, s[50:51], v18, v26 // (op 30684) least significant half
v_addc_co_u32 v19, s[50:51], 0, v19, s[50:51] // (op 30684) most significant half

s_add_u32 m0, m0, 4096 // (op 30684) 
buffer_load_dwordx4 v18, s[20:23], s18 offen offset: 0 lds // (op 30684) Load value direct to lds






s_waitcnt vmcnt(28)
s_barrier  // (op 31981) 



ds_read_b128 v[18:21], v4 // (op 29737) Load local data 



ds_read_b128 v[44:47], v4 offset:1024 // (op 29742) Load local data 



ds_read_b128 v[48:51], v4 offset:2048 // (op 29747) Load local data 



ds_read_b128 v[52:55], v4 offset:3072 // (op 29752) Load local data 
s_waitcnt vmcnt(25)
s_barrier  // (op 31965) 



ds_read_b128 v[56:59], v3 offset:16384 // (op 29677) Load local data 



ds_read_b128 v[60:63], v3 offset:17408 // (op 29682) Load local data 



ds_read_b128 v[64:67], v3 offset:18432 // (op 29687) Load local data 



ds_read_b128 v[68:71], v3 offset:19456 // (op 29692) Load local data 



ds_read_b128 v[72:75], v3 offset:20480 // (op 29697) Load local data 



ds_read_b128 v[76:79], v3 offset:21504 // (op 29702) Load local data 



ds_read_b128 v[80:83], v3 offset:22528 // (op 29707) Load local data 



ds_read_b128 v[84:87], v3 offset:23552 // (op 29712) Load local data 



ds_read_b128 v[88:91], v3 offset:24576 // (op 29717) Load local data 



ds_read_b128 v[92:95], v3 offset:25600 // (op 29722) Load local data 



ds_read_b128 v[96:99], v3 offset:26624 // (op 29727) Load local data 



ds_read_b128 v[100:103], v3 offset:27648 // (op 29732) Load local data 
v_and_b32 v14, -4, v1 // (op 30909) 
v_sub_u32 v15, v1, v14 // (op 30909) 

v_lshlrev_b32 v14, 5, v15 // (op 30909) 

v_lshrrev_b32 v15, 2, v1 // (op 30909) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 30909) 


v_lshlrev_b32 v27, 2, v26 // (op 30909) 

v_lshrrev_b32 v26, 3, v27 // (op 30909) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 30909) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 30966) 
v_sub_u32 v15, v1, v14 // (op 30966) 

v_lshlrev_b32 v14, 5, v15 // (op 30966) 

v_lshrrev_b32 v15, 2, v1 // (op 30966) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 30966) 


v_lshlrev_b32 v27, 2, v26 // (op 30966) 

v_lshrrev_b32 v26, 3, v27 // (op 30966) 

s_nop 0
v_readfirstlane_b32 s1, v26 // (op 30966) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31033) 
v_sub_u32 v15, v1, v14 // (op 31033) 

v_lshlrev_b32 v14, 5, v15 // (op 31033) 

v_lshrrev_b32 v15, 2, v1 // (op 31033) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31033) 


v_lshlrev_b32 v27, 2, v26 // (op 31033) 

v_lshrrev_b32 v26, 3, v27 // (op 31033) 

s_nop 0
v_readfirstlane_b32 s18, v26 // (op 31033) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31077) 
v_sub_u32 v15, v1, v14 // (op 31077) 

v_lshlrev_b32 v14, 5, v15 // (op 31077) 

v_lshrrev_b32 v15, 2, v1 // (op 31077) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31077) 


v_lshlrev_b32 v27, 2, v26 // (op 31077) 

v_lshrrev_b32 v26, 3, v27 // (op 31077) 

s_nop 0
v_readfirstlane_b32 s19, v26 // (op 31077) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31121) 
v_sub_u32 v15, v1, v14 // (op 31121) 

v_lshlrev_b32 v14, 5, v15 // (op 31121) 

v_lshrrev_b32 v15, 2, v1 // (op 31121) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31121) 


v_lshlrev_b32 v27, 2, v26 // (op 31121) 

v_lshrrev_b32 v26, 3, v27 // (op 31121) 

s_nop 0
v_readfirstlane_b32 s49, v26 // (op 31121) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31165) 
v_sub_u32 v15, v1, v14 // (op 31165) 

v_lshlrev_b32 v14, 5, v15 // (op 31165) 

v_lshrrev_b32 v15, 2, v1 // (op 31165) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31165) 


v_lshlrev_b32 v27, 2, v26 // (op 31165) 

v_lshrrev_b32 v26, 3, v27 // (op 31165) 

s_nop 0
v_readfirstlane_b32 s50, v26 // (op 31165) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31209) 
v_sub_u32 v15, v1, v14 // (op 31209) 

v_lshlrev_b32 v14, 5, v15 // (op 31209) 

v_lshrrev_b32 v15, 2, v1 // (op 31209) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31209) 


v_lshlrev_b32 v27, 2, v26 // (op 31209) 

v_lshrrev_b32 v26, 3, v27 // (op 31209) 

s_nop 0
v_readfirstlane_b32 s51, v26 // (op 31209) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31253) 
v_sub_u32 v15, v1, v14 // (op 31253) 

v_lshlrev_b32 v14, 5, v15 // (op 31253) 

v_lshrrev_b32 v15, 2, v1 // (op 31253) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31253) 


v_lshlrev_b32 v27, 2, v26 // (op 31253) 

v_lshrrev_b32 v26, 3, v27 // (op 31253) 

s_nop 0
v_readfirstlane_b32 s72, v26 // (op 31253) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s73, 0 // (op 11300) call()
s_ashr_i64 s[74:75], s[42:43], 7 // (op 11299) 
s_mov_b32 s76, s74 // (op 11299) convert

s_and_b32 s74, s76, 4294967292 // (op 11299) 

s_cmp_lt_u32 s73, s74 // (op 11299) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopBottom_KLoop_11299 // (op 11299) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopBottom_KLoop_11299 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopTop_KLoop_11299:
s_add_u32 s73, s73, 4 // (op 11301) 
s_waitcnt vmcnt(24)
v_mov_b32 v104, v28 // (op 30522) reg expression
v_mov_b32 v105, v29 // (op 30522) reg expression
v_mov_b32 v106, v30 // (op 30522) reg expression
v_mov_b32 v107, v31 // (op 30522) reg expression
s_waitcnt lgkmcnt(12)
s_barrier  // (op 31984) 
s_waitcnt vmcnt(23)
v_mov_b32 v108, v32 // (op 30535) reg expression

v_mov_b32 v109, v33 // (op 30535) reg expression
v_mov_b32 v110, v34 // (op 30535) reg expression
v_mov_b32 v111, v35 // (op 30535) reg expression
s_lshl_b64 s[74:75], s[30:31], 6 // (op 30617) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 30617) 

v_lshrrev_b64 v[14:15], 3, s[76:77] // (op 30617) 


v_mov_b64 v[26:27], v[22:23] // (op 30617) 
v_mov_b64 v[112:113], v[26:27] // (op 30617) 
s_mov_b32 m0, s1 // (op 30617) reg expression
s_mov_b32 s74, 256 // (op 30617) call()
buffer_load_dwordx4 v112, s[52:55], s74 offen offset: 0 lds // (op 30617) Load value direct to lds
v_add_co_u32 v112, s[76:77], v112, v14 // (op 30617) least significant half
v_addc_co_u32 v113, s[76:77], 0, v113, s[76:77] // (op 30617) most significant half

s_add_u32 m0, m0, 4096 // (op 30617) 
buffer_load_dwordx4 v112, s[52:55], s74 offen offset: 0 lds // (op 30617) Load value direct to lds
v_add_co_u32 v112, s[76:77], v112, v14 // (op 30617) least significant half
v_addc_co_u32 v113, s[76:77], 0, v113, s[76:77] // (op 30617) most significant half

s_add_u32 m0, m0, 4096 // (op 30617) 
buffer_load_dwordx4 v112, s[52:55], s74 offen offset: 0 lds // (op 30617) Load value direct to lds
v_add_co_u32 v112, s[76:77], v112, v14 // (op 30617) least significant half
v_addc_co_u32 v113, s[76:77], 0, v113, s[76:77] // (op 30617) most significant half

s_add_u32 m0, m0, 4096 // (op 30617) 
s_waitcnt vmcnt(25)
v_mov_b32 v114, v36 // (op 30548) reg expression
v_mov_b32 v115, v37 // (op 30548) reg expression
v_mov_b32 v116, v38 // (op 30548) reg expression
v_mov_b32 v117, v39 // (op 30548) reg expression
s_waitcnt vmcnt(24)
v_mov_b32 v118, v40 // (op 30561) reg expression
v_mov_b32 v119, v41 // (op 30561) reg expression
v_mov_b32 v120, v42 // (op 30561) reg expression
v_mov_b32 v121, v43 // (op 30561) reg expression
buffer_load_dwordx4 v112, s[52:55], s74 offen offset: 0 lds // (op 30617) Load value direct to lds



v_bfe_u32 v122, v118, 0, 8 // (op 27960) 
v_bfe_u32 v123, v118, 8, 8 // (op 27960) 
v_bfe_u32 v124, v118, 16, 8 // (op 27960) 
v_bfe_u32 v125, v118, 24, 8 // (op 27960) 
v_permlane16_swap_b32 v122, v123 // (op 27960) 
s_nop 0
v_permlane16_swap_b32 v124, v125 // (op 27960) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v122, v124 // (op 27960) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v123, v125 // (op 27960) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 31968) 

s_lshl_b64 s[74:75], s[24:25], 6 // (op 30612) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 30612) 

v_lshrrev_b64 v[14:15], 3, s[76:77] // (op 30612) 


v_mov_b64 v[112:113], v[16:17] // (op 30612) 
v_mov_b64 v[126:127], v[112:113] // (op 30612) 
s_add_u32 s74, 16384, s0 // (op 30612) 
s_mov_b32 m0, s74 // (op 30612) reg expression
s_mov_b32 s75, 256 // (op 30612) call()
buffer_load_dwordx4 v126, s[20:23], s75 offen offset: 0 lds // (op 30612) Load value direct to lds
v_add_co_u32 v126, s[76:77], v126, v14 // (op 30612) least significant half
v_addc_co_u32 v127, s[76:77], 0, v127, s[76:77] // (op 30612) most significant half

s_add_u32 m0, m0, 4096 // (op 30612) 
buffer_load_dwordx4 v126, s[20:23], s75 offen offset: 0 lds // (op 30612) Load value direct to lds
v_add_co_u32 v126, s[76:77], v126, v14 // (op 30612) least significant half
v_addc_co_u32 v127, s[76:77], 0, v127, s[76:77] // (op 30612) most significant half

s_add_u32 m0, m0, 4096 // (op 30612) 
v_bfe_u32 v128, v104, 0, 8 // (op 28353) 
v_bfe_u32 v129, v104, 8, 8 // (op 28353) 
v_bfe_u32 v130, v104, 16, 8 // (op 28353) 
v_bfe_u32 v131, v104, 24, 8 // (op 28353) 
v_permlane16_swap_b32 v128, v129 // (op 28353) 
s_nop 0
v_permlane16_swap_b32 v130, v131 // (op 28353) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v128, v130 // (op 28353) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v129, v131 // (op 28353) 
buffer_load_dwordx4 v126, s[20:23], s75 offen offset: 0 lds // (op 30612) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[18:21], a[0:3], v122, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 76) 
v_mov_b32 v14, v25 // (op 30530) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[44:47], a[4:7], v122, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 417) 
v_add_u32 v15, 16, v14 // (op 30530) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[48:51], a[8:11], v122, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 643) 
v_bfe_u32 v132, v108, 0, 8 // (op 28025) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[52:55], a[12:15], v122, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 869) 
v_bfe_u32 v133, v108, 8, 8 // (op 28025) 
v_bfe_u32 v134, v108, 16, 8 // (op 28025) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[18:21], a[16:19], v123, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1541) 
buffer_load_dwordx4 v[28:31], v15, s[68:71], 0 offen // (op 30530) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[44:47], a[20:23], v123, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1465) 
v_add_u32 v25, 16, v25 // (op 31370) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[48:51], a[24:27], v123, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1389) 
v_bfe_u32 v135, v108, 24, 8 // (op 28025) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[52:55], a[28:31], v123, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1313) 
v_permlane16_swap_b32 v132, v133 // (op 28025) 
v_permlane16_swap_b32 v134, v135 // (op 28025) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[18:21], a[32:35], v124, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2543) 
v_mov_b32 v14, v24 // (op 30543) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[44:47], a[36:39], v124, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2467) 
v_permlane32_swap_b32 v132, v134 // (op 28025) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[48:51], a[40:43], v124, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2391) 
v_permlane32_swap_b32 v133, v135 // (op 28025) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[52:55], a[44:47], v124, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2315) 
s_lshl_b32 s74, s34, 6 // (op 30543) 
s_add_u32 s75, 16, s74 // (op 30543) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[18:21], a[48:51], v125, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3545) 
v_add_u32 v15, v14, s75 // (op 30543) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[44:47], a[52:55], v125, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3469) 
buffer_load_dwordx4 v[32:35], v15, s[64:67], 0 offen // (op 30543) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[48:51], a[56:59], v125, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3393) 
s_lshl_b32 s74, s34, 7 // (op 30556) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[52:55], a[60:63], v125, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3317) 
s_add_u32 s75, 16, s74 // (op 30556) 


v_add_u32 v15, v14, s75 // (op 30556) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:75], v[18:21], a[64:67], v132, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4547) 
s_waitcnt vmcnt(26)
s_barrier  // (op 32003) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:75], v[44:47], a[68:71], v132, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4471) 
ds_read_b128 v[122:125], v6 offset:28672 // (op 212) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[48:51], a[72:75], v132, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4395) 
ds_read_b128 v[136:139], v6 offset:29696 // (op 428) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[52:55], a[76:79], v132, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4319) 
ds_read_b128 v[140:143], v6 offset:30720 // (op 654) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[18:21], a[80:83], v133, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5549) 
v_bfe_u32 v144, v114, 0, 8 // (op 28093) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:79], v[44:47], a[84:87], v133, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5473) 
v_bfe_u32 v145, v114, 8, 8 // (op 28093) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:79], v[48:51], a[88:91], v133, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5397) 
v_bfe_u32 v146, v114, 16, 8 // (op 28093) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:79], v[52:55], a[92:95], v133, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5321) 
v_bfe_u32 v147, v114, 24, 8 // (op 28093) 
v_permlane16_swap_b32 v144, v145 // (op 28093) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[18:21], a[96:99], v134, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6551) 
v_permlane16_swap_b32 v146, v147 // (op 28093) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[44:47], a[100:103], v134, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6475) 
s_nop 0
v_permlane32_swap_b32 v144, v146 // (op 28093) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[48:51], a[104:107], v134, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6399) 
v_permlane32_swap_b32 v145, v147 // (op 28093) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[52:55], a[108:111], v134, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6323) 
ds_read_b128 v[148:151], v6 offset:31744 // (op 880) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:87], v[18:21], a[112:115], v135, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7553) 
s_waitcnt vmcnt(23)
s_barrier  // (op 31997) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:87], v[44:47], a[116:119], v135, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7477) 
ds_read_b128 v[56:59], v5 offset:45056 // (op 210) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:87], v[48:51], a[120:123], v135, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7401) 
ds_read_b128 v[152:155], v5 offset:46080 // (op 1323) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:87], v[52:55], a[124:127], v135, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7325) 
ds_read_b128 v[156:159], v5 offset:47104 // (op 2325) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[88:91], v[18:21], a[128:131], v144, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8555) 
ds_read_b128 v[132:135], v5 offset:48128 // (op 3327) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[88:91], v[44:47], a[132:135], v144, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8479) 
ds_read_b128 v[160:163], v5 offset:49152 // (op 4329) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:91], v[48:51], a[136:139], v144, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8403) 
ds_read_b128 v[68:71], v5 offset:50176 // (op 5331) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[88:91], v[52:55], a[140:143], v144, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8327) 
ds_read_b128 v[72:75], v5 offset:51200 // (op 6333) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:95], v[18:21], a[144:147], v145, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9557) 
ds_read_b128 v[164:167], v5 offset:52224 // (op 7335) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[92:95], v[44:47], a[148:151], v145, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9481) 
ds_read_b128 v[80:83], v5 offset:53248 // (op 8337) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[48:51], a[152:155], v145, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9405) 
ds_read_b128 v[168:171], v5 offset:54272 // (op 9339) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[52:55], a[156:159], v145, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9329) 
buffer_load_dwordx4 v[36:39], v15, s[64:67], 0 offen // (op 30556) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:99], v[18:21], a[160:163], v146, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10559) 
ds_read_b128 v[88:91], v5 offset:55296 // (op 10341) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[96:99], v[44:47], a[164:167], v146, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10483) 
v_add_u32 v15, 16, v14 // (op 30569) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[96:99], v[48:51], a[168:171], v146, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10407) 
buffer_load_dwordx4 v[40:43], v15, s[64:67], 0 offen // (op 30569) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:99], v[52:55], a[172:175], v146, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10331) 
v_add_u32 v24, 16, v24 // (op 31318) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[18:21], a[176:179], v147, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11561) 
ds_read_b128 v[172:175], v5 offset:56320 // (op 11343) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[44:47], a[180:183], v147, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11485) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[48:51], a[184:187], v147, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11409) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[52:55], a[188:191], v147, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11333) 


v_bfe_u32 v144, v119, 0, 8 // (op 27975) 
v_bfe_u32 v145, v119, 8, 8 // (op 27975) 
v_bfe_u32 v146, v119, 16, 8 // (op 27975) 
v_bfe_u32 v147, v119, 24, 8 // (op 27975) 
v_permlane16_swap_b32 v144, v145 // (op 27975) 
s_nop 0
v_permlane16_swap_b32 v146, v147 // (op 27975) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v144, v146 // (op 27975) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v145, v147 // (op 27975) 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 32007) 

s_lshl_b64 s[74:75], s[30:31], 6 // (op 30639) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 30639) 

v_lshrrev_b64 v[14:15], 3, s[76:77] // (op 30639) 


v_mov_b64 v[126:127], v[26:27] // (op 30639) 
s_add_u32 s74, 28672, s19 // (op 30639) 
s_mov_b32 m0, s74 // (op 30639) reg expression
s_mov_b32 s75, 320 // (op 30639) call()
buffer_load_dwordx4 v126, s[52:55], s75 offen offset: 0 lds // (op 30639) Load value direct to lds
v_add_co_u32 v126, s[76:77], v126, v14 // (op 30639) least significant half
v_addc_co_u32 v127, s[76:77], 0, v127, s[76:77] // (op 30639) most significant half

s_add_u32 m0, m0, 4096 // (op 30639) 
v_bfe_u32 v128, v105, 0, 8 // (op 28368) 
v_bfe_u32 v129, v105, 8, 8 // (op 28368) 
v_bfe_u32 v130, v105, 16, 8 // (op 28368) 
v_bfe_u32 v131, v105, 24, 8 // (op 28368) 
v_permlane16_swap_b32 v128, v129 // (op 28368) 
s_nop 0
v_permlane16_swap_b32 v130, v131 // (op 28368) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v128, v130 // (op 28368) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v129, v131 // (op 28368) 
buffer_load_dwordx4 v126, s[52:55], s75 offen offset: 0 lds // (op 30639) Load value direct to lds
v_add_co_u32 v126, s[76:77], v126, v14 // (op 30639) least significant half
s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[122:125], a[0:3], v144, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 217) 
v_addc_co_u32 v127, s[76:77], 0, v127, s[76:77] // (op 30639) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[136:139], a[4:7], v144, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 418) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[140:143], a[8:11], v144, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 644) 
s_add_u32 m0, m0, 4096 // (op 30639) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[148:151], a[12:15], v144, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 870) 
v_bfe_u32 v176, v109, 0, 8 // (op 28040) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[122:125], a[16:19], v145, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1542) 
buffer_load_dwordx4 v126, s[52:55], s75 offen offset: 0 lds // (op 30639) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[152:155], v[136:139], a[20:23], v145, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1466) 
v_bfe_u32 v177, v109, 8, 8 // (op 28040) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[152:155], v[140:143], a[24:27], v145, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1390) 
v_bfe_u32 v178, v109, 16, 8 // (op 28040) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[152:155], v[148:151], a[28:31], v145, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1314) 
v_bfe_u32 v179, v109, 24, 8 // (op 28040) 

v_permlane16_swap_b32 v176, v177 // (op 28040) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[156:159], v[122:125], a[32:35], v146, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2544) 
v_permlane16_swap_b32 v178, v179 // (op 28040) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[156:159], v[136:139], a[36:39], v146, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2468) 
v_add_co_u32 v126, s[76:77], v126, v14 // (op 30639) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[140:143], a[40:43], v146, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2392) 
v_addc_co_u32 v127, s[76:77], 0, v127, s[76:77] // (op 30639) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:159], v[148:151], a[44:47], v146, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2316) 

s_add_u32 m0, m0, 4096 // (op 30639) 
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[132:135], v[122:125], a[48:51], v147, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3546) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:135], v[136:139], a[52:55], v147, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3470) 
v_permlane32_swap_b32 v176, v178 // (op 28040) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[140:143], a[56:59], v147, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3394) 
v_permlane32_swap_b32 v177, v179 // (op 28040) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[148:151], a[60:63], v147, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3318) 
s_waitcnt vmcnt(24)
s_barrier  // (op 32017) 





s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[122:125], a[64:67], v176, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4548) 
ds_read_b128 v[18:21], v8 offset:57344 // (op 240) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[160:163], v[136:139], a[68:71], v176, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4472) 
ds_read_b128 v[44:47], v8 offset:58368 // (op 438) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[160:163], v[140:143], a[72:75], v176, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4396) 
ds_read_b128 v[48:51], v8 offset:59392 // (op 664) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[148:151], a[76:79], v176, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4320) 
v_bfe_u32 v132, v115, 0, 8 // (op 28108) 

v_bfe_u32 v133, v115, 8, 8 // (op 28108) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[122:125], a[80:83], v177, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5550) 
v_bfe_u32 v134, v115, 16, 8 // (op 28108) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[68:71], v[136:139], a[84:87], v177, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5474) 
v_bfe_u32 v135, v115, 24, 8 // (op 28108) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[68:71], v[140:143], a[88:91], v177, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5398) 
buffer_load_dwordx4 v126, s[52:55], s75 offen offset: 0 lds // (op 30639) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[68:71], v[148:151], a[92:95], v177, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5322) 
v_permlane16_swap_b32 v132, v133 // (op 28108) 
v_permlane16_swap_b32 v134, v135 // (op 28108) 
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[72:75], v[122:125], a[96:99], v178, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6552) 
s_nop 0
v_permlane32_swap_b32 v132, v134 // (op 28108) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[72:75], v[136:139], a[100:103], v178, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6476) 
v_permlane32_swap_b32 v133, v135 // (op 28108) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[72:75], v[140:143], a[104:107], v178, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6400) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 32000) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:75], v[148:151], a[108:111], v178, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6324) 
s_lshl_b64 s[74:75], s[24:25], 6 // (op 30634) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 30634) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[164:167], v[122:125], a[112:115], v179, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7554) 
v_lshrrev_b64 v[14:15], 3, s[76:77] // (op 30634) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[164:167], v[136:139], a[116:119], v179, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7478) 
v_mov_b64 v[126:127], v[112:113] // (op 30634) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[140:143], a[120:123], v179, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7402) 
s_add_u32 s74, 45056, s18 // (op 30634) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[148:151], a[124:127], v179, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7326) 
s_mov_b32 m0, s74 // (op 30634) reg expression


s_mov_b32 s75, 320 // (op 30634) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[80:83], v[122:125], a[128:131], v132, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8556) 
ds_read_b128 v[52:55], v8 offset:60416 // (op 890) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[80:83], v[136:139], a[132:135], v132, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8480) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[80:83], v[140:143], a[136:139], v132, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8404) 
s_waitcnt vmcnt(22)
s_barrier  // (op 32011) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[80:83], v[148:151], a[140:143], v132, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8328) 
v_add_u32 v144, 73728, v7 // (op 238) 
ds_read_b128 v[56:59], v144 // (op 238) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[168:171], v[122:125], a[144:147], v133, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9558) 
v_add_u32 v144, 74752, v7 // (op 1333) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[168:171], v[136:139], a[148:151], v133, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9482) 
ds_read_b128 v[60:63], v144 // (op 1333) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[168:171], v[140:143], a[152:155], v133, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9406) 
v_add_u32 v144, 75776, v7 // (op 2335) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[168:171], v[148:151], a[156:159], v133, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9330) 
ds_read_b128 v[64:67], v144 // (op 2335) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[88:91], v[122:125], a[160:163], v134, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10560) 
v_add_u32 v144, 76800, v7 // (op 3337) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[88:91], v[136:139], a[164:167], v134, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10484) 
ds_read_b128 v[76:79], v144 // (op 3337) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:91], v[140:143], a[168:171], v134, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10408) 
v_add_u32 v144, 77824, v7 // (op 4339) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[88:91], v[148:151], a[172:175], v134, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10332) 
ds_read_b128 v[84:87], v144 // (op 4339) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[172:175], v[122:125], a[176:179], v135, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11562) 
v_add_u32 v144, 78848, v7 // (op 5341) 

ds_read_b128 v[92:95], v144 // (op 5341) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[172:175], v[136:139], a[180:183], v135, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11486) 
v_add_u32 v122, 79872, v7 // (op 6343) 

ds_read_b128 v[100:103], v122 // (op 6343) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[172:175], v[140:143], a[184:187], v135, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11410) 
v_add_u32 v122, 80896, v7 // (op 7345) 

ds_read_b128 v[136:139], v122 // (op 7345) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[172:175], v[148:151], a[188:191], v135, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11334) 
v_add_u32 v122, 81920, v7 // (op 8347) 




ds_read_b128 v[128:131], v122 // (op 8347) Load local data 

buffer_load_dwordx4 v126, s[20:23], s75 offen offset: 0 lds // (op 30634) Load value direct to lds



v_add_u32 v122, 82944, v7 // (op 9349) 
v_add_co_u32 v126, s[76:77], v126, v14 // (op 30634) least significant half
v_addc_co_u32 v127, s[76:77], 0, v127, s[76:77] // (op 30634) most significant half

s_add_u32 m0, m0, 4096 // (op 30634) 
ds_read_b128 v[132:135], v122 // (op 9349) Load local data 

buffer_load_dwordx4 v126, s[20:23], s75 offen offset: 0 lds // (op 30634) Load value direct to lds



v_add_u32 v122, 83968, v7 // (op 10351) 
ds_read_b128 v[140:143], v122 // (op 10351) Load local data 

v_add_co_u32 v126, s[76:77], v126, v14 // (op 30634) least significant half



v_addc_co_u32 v127, s[76:77], 0, v127, s[76:77] // (op 30634) most significant half

s_add_u32 m0, m0, 4096 // (op 30634) 
v_add_u32 v122, 84992, v7 // (op 11353) 
ds_read_b128 v[144:147], v122 // (op 11353) Load local data 

buffer_load_dwordx4 v126, s[20:23], s75 offen offset: 0 lds // (op 30634) Load value direct to lds




v_bfe_u32 v122, v120, 0, 8 // (op 27992) 
v_bfe_u32 v123, v120, 8, 8 // (op 27992) 
v_bfe_u32 v124, v120, 16, 8 // (op 27992) 
v_bfe_u32 v125, v120, 24, 8 // (op 27992) 
v_permlane16_swap_b32 v122, v123 // (op 27992) 
s_nop 0
v_permlane16_swap_b32 v124, v125 // (op 27992) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v122, v124 // (op 27992) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v123, v125 // (op 27992) 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 32020) 

s_lshl_b64 s[74:75], s[30:31], 6 // (op 30649) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 30649) 

v_lshrrev_b64 v[14:15], 3, s[76:77] // (op 30649) 


v_mov_b64 v[126:127], v[26:27] // (op 30649) 
s_add_u32 s74, 57344, s50 // (op 30649) 
s_mov_b32 m0, s74 // (op 30649) reg expression
s_mov_b32 s75, 384 // (op 30649) call()
v_bfe_u32 v148, v106, 0, 8 // (op 28386) 
v_bfe_u32 v149, v106, 8, 8 // (op 28386) 
v_bfe_u32 v150, v106, 16, 8 // (op 28386) 
v_bfe_u32 v151, v106, 24, 8 // (op 28386) 
v_permlane16_swap_b32 v148, v149 // (op 28386) 
s_nop 0
v_permlane16_swap_b32 v150, v151 // (op 28386) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v148, v150 // (op 28386) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v149, v151 // (op 28386) 
buffer_load_dwordx4 v126, s[52:55], s75 offen offset: 0 lds // (op 30649) Load value direct to lds
v_add_co_u32 v126, s[76:77], v126, v14 // (op 30649) least significant half
s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[18:21], a[0:3], v122, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 245) 
v_addc_co_u32 v127, s[76:77], 0, v127, s[76:77] // (op 30649) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[44:47], a[4:7], v122, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 419) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[48:51], a[8:11], v122, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 645) 
s_add_u32 m0, m0, 4096 // (op 30649) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[52:55], a[12:15], v122, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 871) 
v_bfe_u32 v152, v110, 0, 8 // (op 28058) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[18:21], a[16:19], v123, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1543) 
buffer_load_dwordx4 v126, s[52:55], s75 offen offset: 0 lds // (op 30649) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[44:47], a[20:23], v123, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1467) 
v_bfe_u32 v153, v110, 8, 8 // (op 28058) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[48:51], a[24:27], v123, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1391) 
v_bfe_u32 v154, v110, 16, 8 // (op 28058) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[52:55], a[28:31], v123, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1315) 
v_bfe_u32 v155, v110, 24, 8 // (op 28058) 
v_permlane16_swap_b32 v152, v153 // (op 28058) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[18:21], a[32:35], v124, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2545) 
v_permlane16_swap_b32 v154, v155 // (op 28058) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[44:47], a[36:39], v124, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2469) 
v_add_co_u32 v126, s[76:77], v126, v14 // (op 30649) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[48:51], a[40:43], v124, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2393) 
v_addc_co_u32 v127, s[76:77], 0, v127, s[76:77] // (op 30649) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[52:55], a[44:47], v124, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2317) 
s_add_u32 m0, m0, 4096 // (op 30649) 
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[18:21], a[48:51], v125, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3547) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[44:47], a[52:55], v125, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3471) 
v_permlane32_swap_b32 v152, v154 // (op 28058) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[48:51], a[56:59], v125, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3395) 
v_permlane32_swap_b32 v153, v155 // (op 28058) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[52:55], a[60:63], v125, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3319) 
s_waitcnt vmcnt(23)
s_barrier  // (op 32029) 




s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:87], v[18:21], a[64:67], v152, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4549) 
v_add_u32 v122, 86016, v10 // (op 268) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:87], v[44:47], a[68:71], v152, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4473) 
ds_read_b128 v[156:159], v122 // (op 268) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:87], v[48:51], a[72:75], v152, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4397) 
v_add_u32 v122, 87040, v10 // (op 448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:87], v[52:55], a[76:79], v152, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4321) 
ds_read_b128 v[160:163], v122 // (op 448) Load local data 




s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:95], v[18:21], a[80:83], v153, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5551) 
v_add_u32 v122, 88064, v10 // (op 674) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:95], v[44:47], a[84:87], v153, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5475) 
ds_read_b128 v[164:167], v122 // (op 674) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:95], v[48:51], a[88:91], v153, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5399) 
buffer_load_dwordx4 v126, s[52:55], s75 offen offset: 0 lds // (op 30649) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:95], v[52:55], a[92:95], v153, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5323) 
v_add_u32 v122, 89088, v10 // (op 900) 
v_add_co_u32 v126, s[76:77], v126, v14 // (op 30649) least significant half
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[100:103], v[18:21], a[96:99], v154, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6553) 
v_addc_co_u32 v127, s[76:77], 0, v127, s[76:77] // (op 30649) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[100:103], v[44:47], a[100:103], v154, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6477) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[100:103], v[48:51], a[104:107], v154, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6401) 
s_add_u32 m0, m0, 4096 // (op 30649) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[100:103], v[52:55], a[108:111], v154, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6325) 
v_bfe_u32 v168, v116, 0, 8 // (op 28126) 
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[18:21], a[112:115], v155, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7555) 
v_bfe_u32 v169, v116, 8, 8 // (op 28126) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[44:47], a[116:119], v155, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7479) 
v_bfe_u32 v170, v116, 16, 8 // (op 28126) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[48:51], a[120:123], v155, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7403) 
v_bfe_u32 v171, v116, 24, 8 // (op 28126) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[52:55], a[124:127], v155, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7327) 
v_permlane16_swap_b32 v168, v169 // (op 28126) 


v_permlane16_swap_b32 v170, v171 // (op 28126) 
s_nop 1
v_permlane32_swap_b32 v168, v170 // (op 28126) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v169, v171 // (op 28126) 
ds_read_b128 v[136:139], v122 // (op 900) Load local data 

s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[128:131], v[18:21], a[128:131], v168, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8557) 
s_waitcnt vmcnt(21)
s_barrier  // (op 32023) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[128:131], v[44:47], a[132:135], v168, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8481) 
v_add_u32 v122, 102400, v9 // (op 266) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[128:131], v[48:51], a[136:139], v168, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8405) 
ds_read_b128 v[56:59], v122 // (op 266) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[128:131], v[52:55], a[140:143], v168, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8329) 
v_add_u32 v122, 103424, v9 // (op 1343) 

ds_read_b128 v[68:71], v122 // (op 1343) Load local data 




s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[132:135], v[18:21], a[144:147], v169, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9559) 
v_add_u32 v122, 104448, v9 // (op 2345) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[132:135], v[44:47], a[148:151], v169, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9483) 
ds_read_b128 v[72:75], v122 // (op 2345) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[132:135], v[48:51], a[152:155], v169, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9407) 
v_add_u32 v122, 105472, v9 // (op 3347) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[132:135], v[52:55], a[156:159], v169, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9331) 
ds_read_b128 v[80:83], v122 // (op 3347) Load local data 





s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[140:143], v[18:21], a[160:163], v170, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10561) 
v_add_u32 v122, 106496, v9 // (op 4349) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[140:143], v[44:47], a[164:167], v170, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10485) 
ds_read_b128 v[88:91], v122 // (op 4349) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[48:51], a[168:171], v170, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10409) 
v_add_u32 v122, 107520, v9 // (op 5351) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[52:55], a[172:175], v170, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10333) 
ds_read_b128 v[96:99], v122 // (op 5351) Load local data 





s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[144:147], v[18:21], a[176:179], v171, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11563) 
v_add_u32 v122, 108544, v9 // (op 6353) 
ds_read_b128 v[140:143], v122 // (op 6353) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[144:147], v[44:47], a[180:183], v171, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11487) 
v_add_u32 v122, 109568, v9 // (op 7355) 
ds_read_b128 v[92:95], v122 // (op 7355) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[144:147], v[48:51], a[184:187], v171, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11411) 
v_add_u32 v122, 110592, v9 // (op 8357) 
ds_read_b128 v[100:103], v122 // (op 8357) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[144:147], v[52:55], a[188:191], v171, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11335) 
v_add_u32 v122, 111616, v9 // (op 9359) 



buffer_load_dwordx4 v126, s[52:55], s75 offen offset: 0 lds // (op 30649) Load value direct to lds




ds_read_b128 v[132:135], v122 // (op 9359) Load local data 




s_barrier  // (op 32014) 
v_add_u32 v14, 112640, v9 // (op 10361) 

s_lshl_b64 s[74:75], s[24:25], 6 // (op 30644) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 30644) 

v_lshrrev_b64 v[122:123], 3, s[76:77] // (op 30644) 


v_mov_b64 v[124:125], v[112:113] // (op 30644) 
s_add_u32 s74, 73728, s49 // (op 30644) 
s_mov_b32 m0, s74 // (op 30644) reg expression
s_mov_b32 s75, 384 // (op 30644) call()
ds_read_b128 v[126:129], v14 // (op 10361) Load local data 

buffer_load_dwordx4 v124, s[20:23], s75 offen offset: 0 lds // (op 30644) Load value direct to lds



v_add_u32 v14, 113664, v9 // (op 11363) 
v_add_co_u32 v124, s[76:77], v124, v122 // (op 30644) least significant half
v_addc_co_u32 v125, s[76:77], 0, v125, s[76:77] // (op 30644) most significant half

s_add_u32 m0, m0, 4096 // (op 30644) 
ds_read_b128 v[144:147], v14 // (op 11363) Load local data 

buffer_load_dwordx4 v124, s[20:23], s75 offen offset: 0 lds // (op 30644) Load value direct to lds
v_add_co_u32 v124, s[76:77], v124, v122 // (op 30644) least significant half
v_addc_co_u32 v125, s[76:77], 0, v125, s[76:77] // (op 30644) most significant half

s_add_u32 m0, m0, 4096 // (op 30644) 
buffer_load_dwordx4 v124, s[20:23], s75 offen offset: 0 lds // (op 30644) Load value direct to lds




v_bfe_u32 v122, v121, 0, 8 // (op 28009) 
v_bfe_u32 v123, v121, 8, 8 // (op 28009) 
v_bfe_u32 v124, v121, 16, 8 // (op 28009) 
v_bfe_u32 v125, v121, 24, 8 // (op 28009) 
v_permlane16_swap_b32 v122, v123 // (op 28009) 
s_nop 0
v_permlane16_swap_b32 v124, v125 // (op 28009) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v122, v124 // (op 28009) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v123, v125 // (op 28009) 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 32032) 


s_lshl_b64 s[74:75], s[30:31], 6 // (op 30659) 
s_lshl_b64 s[76:77], s[74:75], 2 // (op 30659) 

v_lshrrev_b64 v[14:15], 3, s[76:77] // (op 30659) 


v_mov_b64 v[130:131], v[26:27] // (op 30659) 
s_add_u32 s74, 86016, s72 // (op 30659) 
s_mov_b32 m0, s74 // (op 30659) reg expression
s_mov_b32 s75, 448 // (op 30659) call()
v_bfe_u32 v118, v107, 0, 8 // (op 28404) 
v_bfe_u32 v119, v107, 8, 8 // (op 28404) 
v_bfe_u32 v120, v107, 16, 8 // (op 28404) 
v_bfe_u32 v121, v107, 24, 8 // (op 28404) 
buffer_load_dwordx4 v130, s[52:55], s75 offen offset: 0 lds // (op 30659) Load value direct to lds
v_permlane16_swap_b32 v118, v119 // (op 28404) 
v_permlane16_swap_b32 v120, v121 // (op 28404) 
v_add_co_u32 v130, s[76:77], v130, v14 // (op 30659) least significant half
v_addc_co_u32 v131, s[76:77], 0, v131, s[76:77] // (op 30659) most significant half

s_add_u32 m0, m0, 4096 // (op 30659) 
v_permlane32_swap_b32 v118, v120 // (op 28404) 
v_permlane32_swap_b32 v119, v121 // (op 28404) 
buffer_load_dwordx4 v130, s[52:55], s75 offen offset: 0 lds // (op 30659) Load value direct to lds

v_add_co_u32 v130, s[76:77], v130, v14 // (op 30659) least significant half
s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[156:159], a[0:3], v122, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 273) 
v_addc_co_u32 v131, s[76:77], 0, v131, s[76:77] // (op 30659) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[160:163], a[4:7], v122, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 420) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[164:167], a[8:11], v122, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 646) 
s_add_u32 m0, m0, 4096 // (op 30659) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[136:139], a[12:15], v122, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 872) 
v_bfe_u32 v104, v111, 0, 8 // (op 28076) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:71], v[156:159], a[16:19], v123, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1544) 
v_bfe_u32 v105, v111, 8, 8 // (op 28076) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:71], v[160:163], a[20:23], v123, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1468) 
v_bfe_u32 v106, v111, 16, 8 // (op 28076) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:71], v[164:167], a[24:27], v123, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1392) 
v_bfe_u32 v107, v111, 24, 8 // (op 28076) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[68:71], v[136:139], a[28:31], v123, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1316) 
v_permlane16_swap_b32 v104, v105 // (op 28076) 
v_permlane16_swap_b32 v106, v107 // (op 28076) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:75], v[156:159], a[32:35], v124, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2546) 
s_nop 0
v_permlane32_swap_b32 v104, v106 // (op 28076) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[72:75], v[160:163], a[36:39], v124, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2470) 
v_permlane32_swap_b32 v105, v107 // (op 28076) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[72:75], v[164:167], a[40:43], v124, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2394) 
s_waitcnt vmcnt(23)
s_barrier  // (op 31990) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:75], v[136:139], a[44:47], v124, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2318) 
ds_read_b128 v[18:21], v4 // (op 65) Load local data 



s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[156:159], a[48:51], v125, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3548) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 414) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[160:163], a[52:55], v125, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3472) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 640) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[164:167], a[56:59], v125, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3396) 
ds_read_b128 v[52:55], v4 offset:3072 // (op 866) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[136:139], a[60:63], v125, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3320) 
s_waitcnt vmcnt(20)
s_barrier  // (op 31974) 




s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[156:159], a[64:67], v104, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4550) 
ds_read_b128 v[56:59], v3 offset:16384 // (op 60) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[160:163], a[68:71], v104, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4474) 
ds_read_b128 v[60:63], v3 offset:17408 // (op 1309) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:91], v[164:167], a[72:75], v104, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4398) 
v_bfe_u32 v108, v117, 0, 8 // (op 28144) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[88:91], v[136:139], a[76:79], v104, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4322) 
v_bfe_u32 v109, v117, 8, 8 // (op 28144) 
v_bfe_u32 v110, v117, 16, 8 // (op 28144) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[96:99], v[156:159], a[80:83], v105, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5552) 
v_bfe_u32 v111, v117, 24, 8 // (op 28144) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[96:99], v[160:163], a[84:87], v105, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5476) 
v_permlane16_swap_b32 v108, v109 // (op 28144) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[96:99], v[164:167], a[88:91], v105, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5400) 
v_permlane16_swap_b32 v110, v111 // (op 28144) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[96:99], v[136:139], a[92:95], v105, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5324) 
s_nop 0
v_permlane32_swap_b32 v108, v110 // (op 28144) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 28144) 

s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[140:143], v[156:159], a[96:99], v106, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6554) 
ds_read_b128 v[64:67], v3 offset:18432 // (op 2311) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[140:143], v[160:163], a[100:103], v106, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6478) 
ds_read_b128 v[68:71], v3 offset:19456 // (op 3313) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[164:167], a[104:107], v106, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6402) 
ds_read_b128 v[72:75], v3 offset:20480 // (op 4315) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[140:143], v[136:139], a[108:111], v106, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6326) 
ds_read_b128 v[76:79], v3 offset:21504 // (op 5317) Load local data 




s_waitcnt lgkmcnt(14)
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:95], v[156:159], a[112:115], v107, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7556) 
ds_read_b128 v[80:83], v3 offset:22528 // (op 6319) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:95], v[160:163], a[116:119], v107, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7480) 
ds_read_b128 v[84:87], v3 offset:23552 // (op 7321) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[92:95], v[164:167], a[120:123], v107, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7404) 
buffer_load_dwordx4 v130, s[52:55], s75 offen offset: 0 lds // (op 30659) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[92:95], v[136:139], a[124:127], v107, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7328) 
v_add_co_u32 v130, s[76:77], v130, v14 // (op 30659) least significant half

v_addc_co_u32 v131, s[76:77], 0, v131, s[76:77] // (op 30659) most significant half

s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[100:103], v[156:159], a[128:131], v108, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8558) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[100:103], v[160:163], a[132:135], v108, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8482) 
s_add_u32 m0, m0, 4096 // (op 30659) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[100:103], v[164:167], a[136:139], v108, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8406) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[100:103], v[136:139], a[140:143], v108, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8330) 
buffer_load_dwordx4 v130, s[52:55], s75 offen offset: 0 lds // (op 30659) Load value direct to lds





s_waitcnt lgkmcnt(14)
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[132:135], v[156:159], a[144:147], v109, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9560) 
v_add_co_u32 v22, vcc, 256, v22 // (op 30845) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[132:135], v[160:163], a[148:151], v109, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9484) 
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 30845) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[132:135], v[164:167], a[152:155], v109, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9408) 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 32026) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[132:135], v[136:139], a[156:159], v109, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9332) 
s_lshl_b64 s[74:75], s[24:25], 6 // (op 30654) 

s_lshl_b64 s[76:77], s[74:75], 2 // (op 30654) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[126:129], v[156:159], a[160:163], v110, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10562) 
v_lshrrev_b64 v[14:15], 3, s[76:77] // (op 30654) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[126:129], v[160:163], a[164:167], v110, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10486) 
v_mov_b64 v[26:27], v[112:113] // (op 30654) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[126:129], v[164:167], a[168:171], v110, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10410) 
s_add_u32 s74, 102400, s51 // (op 30654) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[126:129], v[136:139], a[172:175], v110, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10334) 
s_mov_b32 m0, s74 // (op 30654) reg expression

s_mov_b32 s75, 448 // (op 30654) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[144:147], v[156:159], a[176:179], v111, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11564) 
ds_read_b128 v[88:91], v3 offset:24576 // (op 8323) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[144:147], v[160:163], a[180:183], v111, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11488) 
ds_read_b128 v[92:95], v3 offset:25600 // (op 9325) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[144:147], v[164:167], a[184:187], v111, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11412) 
ds_read_b128 v[96:99], v3 offset:26624 // (op 10327) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[144:147], v[136:139], a[188:191], v111, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11336) 
buffer_load_dwordx4 v26, s[20:23], s75 offen offset: 0 lds // (op 30654) Load value direct to lds




v_add_co_u32 v26, s[76:77], v26, v14 // (op 30654) least significant half
v_addc_co_u32 v27, s[76:77], 0, v27, s[76:77] // (op 30654) most significant half

s_add_u32 m0, m0, 4096 // (op 30654) 
ds_read_b128 v[100:103], v3 offset:27648 // (op 11329) Load local data 
buffer_load_dwordx4 v26, s[20:23], s75 offen offset: 0 lds // (op 30654) Load value direct to lds
v_add_co_u32 v26, s[76:77], v26, v14 // (op 30654) least significant half
v_addc_co_u32 v27, s[76:77], 0, v27, s[76:77] // (op 30654) most significant half

s_add_u32 m0, m0, 4096 // (op 30654) 
buffer_load_dwordx4 v26, s[20:23], s75 offen offset: 0 lds // (op 30654) Load value direct to lds





v_add_co_u32 v16, vcc, 256, v16 // (op 30781) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 30781) most significant half
s_ashr_i64 s[74:75], s[42:43], 7 // (op 11299) 
s_mov_b32 s76, s74 // (op 11299) convert

s_and_b32 s74, s76, 4294967292 // (op 11299) 

s_cmp_lt_u32 s73, s74 // (op 11299) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopTop_KLoop_11299 // (op 11299) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopTop_KLoop_11299 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopBottom_KLoop_11299:
















s_barrier  // (op 32185) 
s_mul_hi_i32 s0, s16, s47 // (op 30697) 
v_lshrrev_b32 v14, 2, v1 // (op 30697) 
s_add_i32 s1, s0, s16 // (op 30697) 

v_and_b32 v15, -4, v1 // (op 30697) 
s_ashr_i32 s0, s1, 31 // (op 30697) 
v_sub_u32 v16, v1, v15 // (op 30697) 

s_cmp_eq_i32 s47, 0 // (op 30697) 
s_mov_b32 s18, scc // (op 30697) 
v_lshlrev_b32 v15, 5, v16 // (op 30697) 

s_cmp_eq_i32 s18, 0 // (op 30697) 
s_cselect_b32 s19, 0, -1 // (op 30697) 

s_and_b32 s18, s13, 31 // (op 30697) 
s_lshl_b32 s49, 1, s18 // (op 30697) 
s_add_i32 s50, s49, s19 // (op 30697) 


s_and_b32 s19, s0, s50 // (op 30697) 


s_add_i32 s0, s1, s19 // (op 30697) 


s_ashr_i32 s1, s0, s18 // (op 30697) 


s_lshl_b32 s0, s13, 24 // (op 30697) 
s_ashr_i32 s18, s0, 31 // (op 30697) 

s_xor_b32 s0, s1, s18 // (op 30697) 

s_sub_i32 s1, s0, s18 // (op 30697) 


s_mul_hi_i32 s0, s1, s14 // (op 30697) 
s_add_i32 s18, s0, s1 // (op 30697) 

s_ashr_i32 s0, s18, 31 // (op 30697) 
s_cmp_eq_i32 s14, 0 // (op 30697) 
s_mov_b32 s19, scc // (op 30697) 
s_cmp_eq_i32 s19, 0 // (op 30697) 
s_cselect_b32 s49, 0, -1 // (op 30697) 

s_and_b32 s19, s15, 31 // (op 30697) 
s_lshl_b32 s50, 1, s19 // (op 30697) 
s_add_i32 s51, s50, s49 // (op 30697) 


s_and_b32 s49, s0, s51 // (op 30697) 


s_add_i32 s0, s18, s49 // (op 30697) 


s_ashr_i32 s18, s0, s19 // (op 30697) 


s_lshl_b32 s0, s15, 24 // (op 30697) 
s_ashr_i32 s19, s0, 31 // (op 30697) 

s_xor_b32 s0, s18, s19 // (op 30697) 

s_sub_i32 s18, s0, s19 // (op 30697) 


s_mul_i32 s0, s1, s48 // (op 30697) 

s_sub_u32 s1, s16, s0 // (op 30697) 

s_mul_i32 s0, s18, s48 // (op 30697) 
s_add_u32 s0, s0, s1 // (op 30697) 


s_mul_hi_i32 s1, s16, s10 // (op 30697) 
s_add_i32 s18, s1, s16 // (op 30697) 

s_ashr_i32 s1, s18, 31 // (op 30697) 
s_cmp_eq_i32 s10, 0 // (op 30697) 
s_mov_b32 s19, scc // (op 30697) 
s_cmp_eq_i32 s19, 0 // (op 30697) 
s_cselect_b32 s49, 0, -1 // (op 30697) 

s_and_b32 s19, s12, 31 // (op 30697) 
s_lshl_b32 s50, 1, s19 // (op 30697) 
s_add_i32 s51, s50, s49 // (op 30697) 


s_and_b32 s49, s1, s51 // (op 30697) 


s_add_i32 s1, s18, s49 // (op 30697) 


s_ashr_i32 s18, s1, s19 // (op 30697) 


s_lshl_b32 s1, s12, 24 // (op 30697) 
s_ashr_i32 s19, s1, 31 // (op 30697) 

s_xor_b32 s1, s18, s19 // (op 30697) 

s_sub_i32 s18, s1, s19 // (op 30697) 


s_add_u32 s50, s8, 256 // (op 30697) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 30697) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 30697) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 30697) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 30697) 

s_mul_i32 s1, s48, s50 // (op 30697) 

s_mul_i32 s19, s44, s1 // (op 30697) 

s_mul_i32 s1, s18, s19 // (op 30697) 

s_sub_u32 s19, s16, s1 // (op 30697) 

s_mov_b32 s1, s19 // (op 30697) 
s_mul_hi_i32 s49, s1, s45 // (op 30697) 
s_add_i32 s50, s49, s1 // (op 30697) 


s_ashr_i32 s1, s50, 31 // (op 30697) 
s_cmp_eq_i32 s45, 0 // (op 30697) 
s_mov_b32 s49, scc // (op 30697) 
s_cmp_eq_i32 s49, 0 // (op 30697) 
s_cselect_b32 s51, 0, -1 // (op 30697) 

s_and_b32 s49, s11, 31 // (op 30697) 
s_lshl_b32 s64, 1, s49 // (op 30697) 
s_add_i32 s65, s64, s51 // (op 30697) 


s_and_b32 s51, s1, s65 // (op 30697) 


s_add_i32 s1, s50, s51 // (op 30697) 


s_ashr_i32 s50, s1, s49 // (op 30697) 


s_lshl_b32 s1, s11, 24 // (op 30697) 
s_ashr_i32 s49, s1, 31 // (op 30697) 

s_xor_b32 s1, s50, s49 // (op 30697) 

s_sub_i32 s50, s1, s49 // (op 30697) 


s_mul_i32 s1, s46, s48 // (op 30697) 
s_sub_i32 s49, s6, s1 // (op 30697) 
s_mul_i32 s51, s50, s49 // (op 30697) 


s_sub_u32 s49, s19, s51 // (op 30697) 


s_add_u32 s19, s49, s1 // (op 30697) 


s_cmp_eq_i32 s18, 0 // (op 30697) 
s_mov_b32 s1, scc // (op 30697) 

s_cmp_eq_i32 s1, 0 // (op 30697) 
s_cselect_b32 s18, s19, s0 // (op 30697) 



v_mov_b32 v16, 192 // (op 30697) 
v_mul_lo_u32 v17, s18, v16 // (op 30697) 

v_add_u32 v17, v17, v14 // (op 30697) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 30697) 
s_lshr_b32 s18, s0, 2 // (op 30697) 

v_lshl_add_u32 v14, s18, 9, v15 // (op 30697) 


v_mov_b32 v19, 0 // (op 30697) convert
v_mov_b32 v18, v14 // (op 30697) convert

v_mul_lo_u32 v15, v17, s25 // (op 30697) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 30697) most significant: high of low * low
v_mul_lo_u32 v14, v17, s24 // (op 30697) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 30697) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v18 // (op 30697) least significant half
v_addc_co_u32 v15, s[0:1], v15, v19, s[0:1] // (op 30697) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 30697) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 30697) 

s_mul_i32 s1, s24, s29 // (op 30699) most significant: low of low * high
s_mul_i32 s18, s25, s28 // (op 30699) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 30699) least significant
s_mul_hi_u32 s19, s24, s28 // (op 30699) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30699) most significant: sum
s_add_u32 s1, s1, s19 // (op 30699) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 30699) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 30699) 

s_mov_b32 s20, s26 // (op 30699) reg expression
s_mov_b32 s21, s27 // (op 30699) reg expression
s_mov_b32 s22, s0 // (op 30699) reg expression
s_mov_b32 s23, 131072 // (op 30699) call()

s_mul_hi_i32 s0, s16, s47 // (op 30811) 
v_lshrrev_b32 v16, 2, v1 // (op 30811) 
s_add_i32 s1, s0, s16 // (op 30811) 

v_and_b32 v17, -4, v1 // (op 30811) 
s_ashr_i32 s0, s1, 31 // (op 30811) 
v_sub_u32 v18, v1, v17 // (op 30811) 

s_cmp_eq_i32 s47, 0 // (op 30811) 
s_mov_b32 s18, scc // (op 30811) 
v_lshlrev_b32 v17, 5, v18 // (op 30811) 

s_cmp_eq_i32 s18, 0 // (op 30811) 
s_cselect_b32 s19, 0, -1 // (op 30811) 

s_and_b32 s18, s13, 31 // (op 30811) 
s_lshl_b32 s49, 1, s18 // (op 30811) 
s_add_i32 s50, s49, s19 // (op 30811) 


s_and_b32 s19, s0, s50 // (op 30811) 


s_add_i32 s0, s1, s19 // (op 30811) 


s_ashr_i32 s1, s0, s18 // (op 30811) 


s_lshl_b32 s0, s13, 24 // (op 30811) 
s_ashr_i32 s18, s0, 31 // (op 30811) 

s_xor_b32 s0, s1, s18 // (op 30811) 

s_sub_i32 s1, s0, s18 // (op 30811) 


s_mul_hi_i32 s0, s1, s14 // (op 30811) 
s_add_i32 s18, s0, s1 // (op 30811) 

s_ashr_i32 s0, s18, 31 // (op 30811) 
s_cmp_eq_i32 s14, 0 // (op 30811) 
s_mov_b32 s19, scc // (op 30811) 
s_cmp_eq_i32 s19, 0 // (op 30811) 
s_cselect_b32 s49, 0, -1 // (op 30811) 

s_and_b32 s19, s15, 31 // (op 30811) 
s_lshl_b32 s50, 1, s19 // (op 30811) 
s_add_i32 s51, s50, s49 // (op 30811) 


s_and_b32 s49, s0, s51 // (op 30811) 


s_add_i32 s0, s18, s49 // (op 30811) 


s_ashr_i32 s18, s0, s19 // (op 30811) 


s_lshl_b32 s0, s15, 24 // (op 30811) 
s_ashr_i32 s19, s0, 31 // (op 30811) 

s_xor_b32 s0, s18, s19 // (op 30811) 

s_sub_i32 s18, s0, s19 // (op 30811) 


s_add_u32 s50, s8, 256 // (op 30811) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 30811) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 30811) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 30811) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 30811) 

s_mul_i32 s0, s18, s50 // (op 30811) 

s_sub_i32 s49, s1, s0 // (op 30811) 


s_mul_hi_i32 s0, s16, s10 // (op 30811) 
s_add_i32 s1, s0, s16 // (op 30811) 

s_ashr_i32 s0, s1, 31 // (op 30811) 
s_cmp_eq_i32 s10, 0 // (op 30811) 
s_mov_b32 s18, scc // (op 30811) 
s_cmp_eq_i32 s18, 0 // (op 30811) 
s_cselect_b32 s19, 0, -1 // (op 30811) 

s_and_b32 s18, s12, 31 // (op 30811) 
s_lshl_b32 s64, 1, s18 // (op 30811) 
s_add_i32 s65, s64, s19 // (op 30811) 


s_and_b32 s19, s0, s65 // (op 30811) 


s_add_i32 s0, s1, s19 // (op 30811) 


s_ashr_i32 s1, s0, s18 // (op 30811) 


s_lshl_b32 s0, s12, 24 // (op 30811) 
s_ashr_i32 s18, s0, 31 // (op 30811) 

s_xor_b32 s0, s1, s18 // (op 30811) 

s_sub_i32 s1, s0, s18 // (op 30811) 


s_mul_i32 s0, s48, s50 // (op 30811) 

s_mul_i32 s18, s44, s0 // (op 30811) 

s_mul_i32 s0, s1, s18 // (op 30811) 

s_sub_u32 s18, s16, s0 // (op 30811) 

s_mul_hi_i32 s0, s18, s45 // (op 30811) 
s_add_i32 s19, s0, s18 // (op 30811) 


s_ashr_i32 s0, s19, 31 // (op 30811) 
s_cmp_eq_i32 s45, 0 // (op 30811) 
s_mov_b32 s18, scc // (op 30811) 
s_cmp_eq_i32 s18, 0 // (op 30811) 
s_cselect_b32 s50, 0, -1 // (op 30811) 

s_and_b32 s18, s11, 31 // (op 30811) 
s_lshl_b32 s51, 1, s18 // (op 30811) 
s_add_i32 s64, s51, s50 // (op 30811) 


s_and_b32 s50, s0, s64 // (op 30811) 


s_add_i32 s0, s19, s50 // (op 30811) 


s_ashr_i32 s19, s0, s18 // (op 30811) 


s_lshl_b32 s0, s11, 24 // (op 30811) 
s_ashr_i32 s18, s0, 31 // (op 30811) 

s_xor_b32 s0, s19, s18 // (op 30811) 

s_sub_i32 s19, s0, s18 // (op 30811) 


s_cmp_eq_i32 s1, 0 // (op 30811) 
s_mov_b32 s0, scc // (op 30811) 

s_cmp_eq_i32 s0, 0 // (op 30811) 
s_cselect_b32 s1, s19, s49 // (op 30811) 



v_lshl_add_u32 v18, s1, 8, v16 // (op 30811) 


v_mul_lo_u32 v27, v18, s31 // (op 30811) most significant: low of low * high
v_mul_hi_u32 v16, v18, s30 // (op 30811) most significant: high of low * low
v_mul_lo_u32 v26, v18, s30 // (op 30811) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 30811) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 30811) 
s_lshr_b32 s18, s0, 2 // (op 30811) 

v_lshl_add_u32 v16, s18, 9, v17 // (op 30811) 


v_mov_b32 v25, 0 // (op 30811) convert
v_mov_b32 v24, v16 // (op 30811) convert

v_add_co_u32 v16, s[0:1], v24, v26 // (op 30811) least significant half
v_addc_co_u32 v17, s[0:1], v25, v27, s[0:1] // (op 30811) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 30811) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 30811) 

s_mul_i32 s1, s30, s9 // (op 30813) most significant: low of low * high
s_mul_i32 s18, s31, s8 // (op 30813) most significant: low of high * low
s_mul_i32 s0, s30, s8 // (op 30813) least significant
s_mul_hi_u32 s19, s30, s8 // (op 30813) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30813) most significant: sum
s_add_u32 s1, s1, s19 // (op 30813) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 30813) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 30813) 

s_mov_b32 s52, s32 // (op 30813) reg expression
s_mov_b32 s53, s33 // (op 30813) reg expression
s_mov_b32 s54, s0 // (op 30813) reg expression
s_mov_b32 s55, 131072 // (op 30813) call()

v_and_b32 v16, -4, v1 // (op 30894) 
v_sub_u32 v17, v1, v16 // (op 30894) 

v_lshlrev_b32 v16, 5, v17 // (op 30894) 

v_lshrrev_b32 v17, 2, v1 // (op 30894) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 30894) 


v_lshlrev_b32 v16, 2, v20 // (op 30894) 

v_lshrrev_b32 v17, 3, v16 // (op 30894) 

s_nop 0
v_readfirstlane_b32 s0, v17 // (op 30894) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 30953) 
v_sub_u32 v17, v1, v16 // (op 30953) 

v_lshlrev_b32 v16, 5, v17 // (op 30953) 

v_lshrrev_b32 v17, 2, v1 // (op 30953) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 30953) 


v_lshlrev_b32 v16, 2, v20 // (op 30953) 

v_lshrrev_b32 v17, 3, v16 // (op 30953) 

s_nop 0
v_readfirstlane_b32 s1, v17 // (op 30953) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s18, s16, s47 // (op 31407) 
v_and_b32 v16, 63, v1 // (op 31407) 
s_add_i32 s19, s18, s16 // (op 31407) 

v_lshrrev_b32 v17, 4, v16 // (op 31407) 
s_ashr_i32 s18, s19, 31 // (op 31407) 
v_and_b32 v20, 3, v17 // (op 31407) 

s_cmp_eq_i32 s47, 0 // (op 31407) 
s_mov_b32 s49, scc // (op 31407) 
v_and_b32 v17, 15, v1 // (op 31407) 
s_cmp_eq_i32 s49, 0 // (op 31407) 
s_cselect_b32 s50, 0, -1 // (op 31407) 

v_lshl_add_u32 v21, v20, 4, v17 // (op 31407) 


s_and_b32 s49, s13, 31 // (op 31407) 
v_lshrrev_b32 v17, 8, v1 // (op 31407) 
s_lshl_b32 s51, 1, s49 // (op 31407) 
v_mul_lo_u32 v20, v17, 3 // (op 31407) 

s_add_i32 s64, s51, s50 // (op 31407) 


v_lshl_add_u32 v17, v20, 6, v21 // (op 31407) 


s_and_b32 s50, s18, s64 // (op 31407) 


v_lshrrev_b32 v20, 6, v16 // (op 31407) 

s_add_i32 s18, s19, s50 // (op 31407) 


v_lshlrev_b32 v16, 2, v20 // (op 31407) 

s_ashr_i32 s19, s18, s49 // (op 31407) 


s_lshl_b32 s18, s13, 24 // (op 31407) 
s_ashr_i32 s49, s18, 31 // (op 31407) 

s_xor_b32 s18, s19, s49 // (op 31407) 

s_sub_i32 s19, s18, s49 // (op 31407) 


s_mul_hi_i32 s18, s19, s14 // (op 31407) 
s_add_i32 s49, s18, s19 // (op 31407) 

s_ashr_i32 s18, s49, 31 // (op 31407) 
s_cmp_eq_i32 s14, 0 // (op 31407) 
s_mov_b32 s50, scc // (op 31407) 
s_cmp_eq_i32 s50, 0 // (op 31407) 
s_cselect_b32 s51, 0, -1 // (op 31407) 

s_and_b32 s50, s15, 31 // (op 31407) 
s_lshl_b32 s64, 1, s50 // (op 31407) 
s_add_i32 s65, s64, s51 // (op 31407) 


s_and_b32 s51, s18, s65 // (op 31407) 


s_add_i32 s18, s49, s51 // (op 31407) 


s_ashr_i32 s49, s18, s50 // (op 31407) 


s_lshl_b32 s18, s15, 24 // (op 31407) 
s_ashr_i32 s50, s18, 31 // (op 31407) 

s_xor_b32 s18, s49, s50 // (op 31407) 

s_sub_i32 s49, s18, s50 // (op 31407) 


s_mul_i32 s18, s19, s48 // (op 31407) 

s_sub_u32 s19, s16, s18 // (op 31407) 

s_mul_i32 s18, s49, s48 // (op 31407) 
s_add_u32 s18, s18, s19 // (op 31407) 


s_mul_hi_i32 s19, s16, s10 // (op 31407) 
s_add_i32 s49, s19, s16 // (op 31407) 

s_ashr_i32 s19, s49, 31 // (op 31407) 
s_cmp_eq_i32 s10, 0 // (op 31407) 
s_mov_b32 s50, scc // (op 31407) 
s_cmp_eq_i32 s50, 0 // (op 31407) 
s_cselect_b32 s51, 0, -1 // (op 31407) 

s_and_b32 s50, s12, 31 // (op 31407) 
s_lshl_b32 s64, 1, s50 // (op 31407) 
s_add_i32 s65, s64, s51 // (op 31407) 


s_and_b32 s51, s19, s65 // (op 31407) 


s_add_i32 s19, s49, s51 // (op 31407) 


s_ashr_i32 s49, s19, s50 // (op 31407) 


s_lshl_b32 s19, s12, 24 // (op 31407) 
s_ashr_i32 s50, s19, 31 // (op 31407) 

s_xor_b32 s19, s49, s50 // (op 31407) 

s_sub_i32 s49, s19, s50 // (op 31407) 


s_add_u32 s50, s8, 256 // (op 31407) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 31407) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 31407) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 31407) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 8 // (op 31407) 

s_mul_i32 s19, s48, s50 // (op 31407) 

s_mul_i32 s50, s44, s19 // (op 31407) 

s_mul_i32 s19, s49, s50 // (op 31407) 

s_sub_u32 s50, s16, s19 // (op 31407) 

s_mov_b32 s19, s50 // (op 31407) 
s_mul_hi_i32 s51, s19, s45 // (op 31407) 
s_add_i32 s64, s51, s19 // (op 31407) 


s_ashr_i32 s19, s64, 31 // (op 31407) 
s_cmp_eq_i32 s45, 0 // (op 31407) 
s_mov_b32 s51, scc // (op 31407) 
s_cmp_eq_i32 s51, 0 // (op 31407) 
s_cselect_b32 s65, 0, -1 // (op 31407) 

s_and_b32 s51, s11, 31 // (op 31407) 
s_lshl_b32 s66, 1, s51 // (op 31407) 
s_add_i32 s67, s66, s65 // (op 31407) 


s_and_b32 s65, s19, s67 // (op 31407) 


s_add_i32 s19, s64, s65 // (op 31407) 


s_ashr_i32 s64, s19, s51 // (op 31407) 


s_lshl_b32 s19, s11, 24 // (op 31407) 
s_ashr_i32 s51, s19, 31 // (op 31407) 

s_xor_b32 s19, s64, s51 // (op 31407) 

s_sub_i32 s64, s19, s51 // (op 31407) 


s_mul_i32 s19, s46, s48 // (op 31407) 
s_sub_i32 s51, s6, s19 // (op 31407) 
s_mul_i32 s65, s64, s51 // (op 31407) 


s_sub_u32 s51, s50, s65 // (op 31407) 


s_add_u32 s50, s51, s19 // (op 31407) 


s_cmp_eq_i32 s49, 0 // (op 31407) 
s_mov_b32 s19, scc // (op 31407) 

s_cmp_eq_i32 s19, 0 // (op 31407) 
s_cselect_b32 s49, s50, s18 // (op 31407) 



v_mov_b32 v20, 192 // (op 31407) 
v_mul_lo_u32 v21, s49, v20 // (op 31407) 

v_add_u32 v21, v21, v17 // (op 31407) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 31407) 
s_lshr_b32 s49, s18, 2 // (op 31407) 

v_lshl_add_u32 v17, s49, 4, v16 // (op 31407) 


v_mul_lo_u32 v16, v21, s34 // (op 31407) 
v_add_u32 v16, v16, v17 // (op 31407) 


s_mul_i32 s19, s34, s29 // (op 31409) most significant: low of low * high
s_mul_i32 s49, s35, s28 // (op 31409) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 31409) least significant
s_mul_hi_u32 s50, s34, s28 // (op 31409) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 31409) most significant: sum
s_add_u32 s19, s19, s50 // (op 31409) most significant: sum


s_mov_b32 s64, s36 // (op 31409) reg expression
s_mov_b32 s65, s37 // (op 31409) reg expression
s_mov_b32 s66, s18 // (op 31409) reg expression
s_mov_b32 s67, 131072 // (op 31409) call()

s_mul_hi_i32 s18, s16, s47 // (op 31434) 
v_and_b32 v17, 63, v1 // (op 31434) 
s_add_i32 s19, s18, s16 // (op 31434) 

v_lshrrev_b32 v20, 4, v17 // (op 31434) 
s_ashr_i32 s18, s19, 31 // (op 31434) 
v_and_b32 v21, 3, v20 // (op 31434) 

s_cmp_eq_i32 s47, 0 // (op 31434) 
s_mov_b32 s49, scc // (op 31434) 
v_and_b32 v20, 15, v1 // (op 31434) 
s_cmp_eq_i32 s49, 0 // (op 31434) 
s_cselect_b32 s50, 0, -1 // (op 31434) 

v_lshl_add_u32 v22, v21, 4, v20 // (op 31434) 


s_and_b32 s49, s13, 31 // (op 31434) 
v_lshrrev_b32 v20, 6, v1 // (op 31434) 
s_lshl_b32 s51, 1, s49 // (op 31434) 
v_and_b32 v21, 3, v20 // (op 31434) 

s_add_i32 s68, s51, s50 // (op 31434) 


v_lshl_add_u32 v20, v21, 6, v22 // (op 31434) 


s_and_b32 s50, s18, s68 // (op 31434) 


v_lshrrev_b32 v21, 6, v17 // (op 31434) 

s_add_i32 s18, s19, s50 // (op 31434) 


v_lshlrev_b32 v17, 2, v21 // (op 31434) 

s_ashr_i32 s19, s18, s49 // (op 31434) 


s_lshl_b32 s18, s13, 24 // (op 31434) 
s_ashr_i32 s49, s18, 31 // (op 31434) 

s_xor_b32 s18, s19, s49 // (op 31434) 

s_sub_i32 s19, s18, s49 // (op 31434) 


s_mul_hi_i32 s18, s19, s14 // (op 31434) 
s_add_i32 s49, s18, s19 // (op 31434) 

s_ashr_i32 s18, s49, 31 // (op 31434) 
s_cmp_eq_i32 s14, 0 // (op 31434) 
s_mov_b32 s50, scc // (op 31434) 
s_cmp_eq_i32 s50, 0 // (op 31434) 
s_cselect_b32 s51, 0, -1 // (op 31434) 

s_and_b32 s50, s15, 31 // (op 31434) 
s_lshl_b32 s68, 1, s50 // (op 31434) 
s_add_i32 s69, s68, s51 // (op 31434) 


s_and_b32 s51, s18, s69 // (op 31434) 


s_add_i32 s18, s49, s51 // (op 31434) 


s_ashr_i32 s49, s18, s50 // (op 31434) 


s_lshl_b32 s18, s15, 24 // (op 31434) 
s_ashr_i32 s50, s18, 31 // (op 31434) 

s_xor_b32 s18, s49, s50 // (op 31434) 

s_sub_i32 s49, s18, s50 // (op 31434) 


s_add_u32 s50, s8, 256 // (op 31434) Start of Int64 add, locking SCC
s_addc_u32 s51, s9, 0 // (op 31434) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 31434) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 31434) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 8 // (op 31434) 

s_mul_i32 s18, s49, s50 // (op 31434) 

s_sub_i32 s49, s19, s18 // (op 31434) 


s_mul_hi_i32 s18, s16, s10 // (op 31434) 
s_add_i32 s19, s18, s16 // (op 31434) 

s_ashr_i32 s18, s19, 31 // (op 31434) 
s_cmp_eq_i32 s10, 0 // (op 31434) 
s_mov_b32 s68, scc // (op 31434) 
s_cmp_eq_i32 s68, 0 // (op 31434) 
s_cselect_b32 s69, 0, -1 // (op 31434) 

s_and_b32 s68, s12, 31 // (op 31434) 
s_lshl_b32 s70, 1, s68 // (op 31434) 
s_add_i32 s71, s70, s69 // (op 31434) 


s_and_b32 s69, s18, s71 // (op 31434) 


s_add_i32 s18, s19, s69 // (op 31434) 


s_ashr_i32 s19, s18, s68 // (op 31434) 


s_lshl_b32 s18, s12, 24 // (op 31434) 
s_ashr_i32 s68, s18, 31 // (op 31434) 

s_xor_b32 s18, s19, s68 // (op 31434) 

s_sub_i32 s19, s18, s68 // (op 31434) 


s_mul_i32 s18, s48, s50 // (op 31434) 

s_mul_i32 s50, s44, s18 // (op 31434) 

s_mul_i32 s18, s19, s50 // (op 31434) 

s_sub_u32 s50, s16, s18 // (op 31434) 

s_mul_hi_i32 s18, s50, s45 // (op 31434) 
s_add_i32 s51, s18, s50 // (op 31434) 


s_ashr_i32 s18, s51, 31 // (op 31434) 
s_cmp_eq_i32 s45, 0 // (op 31434) 
s_mov_b32 s50, scc // (op 31434) 
s_cmp_eq_i32 s50, 0 // (op 31434) 
s_cselect_b32 s68, 0, -1 // (op 31434) 

s_and_b32 s50, s11, 31 // (op 31434) 
s_lshl_b32 s69, 1, s50 // (op 31434) 
s_add_i32 s70, s69, s68 // (op 31434) 


s_and_b32 s68, s18, s70 // (op 31434) 


s_add_i32 s18, s51, s68 // (op 31434) 


s_ashr_i32 s51, s18, s50 // (op 31434) 


s_lshl_b32 s18, s11, 24 // (op 31434) 
s_ashr_i32 s50, s18, 31 // (op 31434) 

s_xor_b32 s18, s51, s50 // (op 31434) 

s_sub_i32 s51, s18, s50 // (op 31434) 


s_cmp_eq_i32 s19, 0 // (op 31434) 
s_mov_b32 s18, scc // (op 31434) 

s_cmp_eq_i32 s18, 0 // (op 31434) 
s_cselect_b32 s19, s51, s49 // (op 31434) 



v_lshl_add_u32 v21, s19, 8, v20 // (op 31434) 


v_mul_lo_u32 v20, v21, s38 // (op 31434) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 31434) 
s_lshr_b32 s49, s18, 2 // (op 31434) 

v_lshl_add_u32 v21, s49, 4, v17 // (op 31434) 


v_add_u32 v17, v21, v20 // (op 31434) 


s_mul_i32 s19, s38, s9 // (op 31436) most significant: low of low * high
s_mul_i32 s49, s39, s8 // (op 31436) most significant: low of high * low
s_mul_i32 s18, s38, s8 // (op 31436) least significant
s_mul_hi_u32 s50, s38, s8 // (op 31436) most significant: high of low * low
s_add_u32 s19, s19, s49 // (op 31436) most significant: sum
s_add_u32 s19, s19, s50 // (op 31436) most significant: sum


s_mov_b32 s68, s40 // (op 31436) reg expression
s_mov_b32 s69, s41 // (op 31436) reg expression
s_mov_b32 s70, s18 // (op 31436) reg expression
s_mov_b32 s71, 131072 // (op 31436) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 11278) 
s_and_b32 s49, s18, 4294967292 // (op 11278) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 11277) 
s_mov_b32 s50, s18 // (op 11277) convert

s_cmp_lt_u32 s49, s50 // (op 11277) 




















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopBottom_KLoopTail_11277 // (op 11277) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopBottom_KLoopTail_11277 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopTop_KLoopTail_11277:
s_barrier  // (op 31971) 
s_add_u32 s49, s49, 1 // (op 11279) 

s_lshl_b64 s[18:19], s[24:25], 6 // (op 30622) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 30622) 

v_lshrrev_b64 v[20:21], 3, s[50:51] // (op 30622) 


v_mov_b64 v[22:23], v[14:15] // (op 30622) 
v_mov_b64 v[24:25], v[22:23] // (op 30622) 
s_add_u32 s18, 16384, s0 // (op 30622) 
s_mov_b32 m0, s18 // (op 30622) reg expression
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 30622) Load value direct to lds
v_add_co_u32 v24, s[50:51], v24, v20 // (op 30622) least significant half
v_addc_co_u32 v25, s[50:51], 0, v25, s[50:51] // (op 30622) most significant half

s_add_u32 m0, m0, 4096 // (op 30622) 
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 30622) Load value direct to lds
v_add_co_u32 v24, s[50:51], v24, v20 // (op 30622) least significant half
v_addc_co_u32 v25, s[50:51], 0, v25, s[50:51] // (op 30622) most significant half

s_add_u32 m0, m0, 4096 // (op 30622) 
buffer_load_dwordx4 v24, s[20:23], 0 offen offset: 0 lds // (op 30622) Load value direct to lds




v_add_co_u32 v14, s[18:19], 64, v14 // (op 30687) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 30687) most significant half

s_barrier  // (op 31987) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 30628) 
s_lshl_b64 s[50:51], s[18:19], 2 // (op 30628) 

v_lshrrev_b64 v[20:21], 3, s[50:51] // (op 30628) 


v_mov_b64 v[22:23], v[18:19] // (op 30628) 
v_mov_b64 v[24:25], v[22:23] // (op 30628) 
s_mov_b32 m0, s1 // (op 30628) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 30628) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 30628) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 30628) most significant half

s_add_u32 m0, m0, 4096 // (op 30628) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 30628) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 30628) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 30628) most significant half

s_add_u32 m0, m0, 4096 // (op 30628) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 30628) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 30628) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 30628) most significant half

s_add_u32 m0, m0, 4096 // (op 30628) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 30628) Load value direct to lds



v_add_co_u32 v18, s[18:19], 64, v18 // (op 30803) least significant half
v_addc_co_u32 v19, s[18:19], 0, v19, s[18:19] // (op 30803) most significant half

s_waitcnt vmcnt(4)
s_barrier  // (op 31977) 



ds_read_b128 v[20:23], v3 offset:16384 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 31993) 



ds_read_b128 v[24:27], v4 // (op 162) Load local data 


v_mov_b32 v28, v16 // (op 164) 
v_mov_b32 v29, v28 // (op 164) reg expression

buffer_load_dwordx4 v[30:33], v29, s[64:67], 0 offen // (op 164) Load value



v_mov_b32 v29, v17 // (op 166) 
v_mov_b32 v34, v29 // (op 166) reg expression

buffer_load_dwordx4 v[36:39], v34, s[68:71], 0 offen // (op 166) Load value


v_add_u32 v17, 4, v17 // (op 31424) 
s_waitcnt vmcnt(1)
v_bfe_u32 v40, v30, 0, 8 // (op 28613) 



v_bfe_u32 v41, v30, 8, 8 // (op 28613) 
v_bfe_u32 v42, v30, 16, 8 // (op 28613) 
v_bfe_u32 v43, v30, 24, 8 // (op 28613) 
v_permlane16_swap_b32 v40, v41 // (op 28613) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 397) Load local data 
v_permlane16_swap_b32 v42, v43 // (op 28613) 



s_nop 1
v_permlane32_swap_b32 v40, v42 // (op 28613) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v41, v43 // (op 28613) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 623) Load local data 




ds_read_b128 v[32:35], v4 offset:3072 // (op 849) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v52, v36, 0, 8 // (op 28703) 



v_bfe_u32 v53, v36, 8, 8 // (op 28703) 
v_bfe_u32 v54, v36, 16, 8 // (op 28703) 
v_bfe_u32 v55, v36, 24, 8 // (op 28703) 
v_permlane16_swap_b32 v52, v53 // (op 28703) 
s_nop 0
v_permlane16_swap_b32 v54, v55 // (op 28703) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v52, v54 // (op 28703) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v53, v55 // (op 28703) 
ds_read_b128 v[56:59], v3 offset:17408 // (op 1209) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 167) 
ds_read_b128 v[36:39], v3 offset:18432 // (op 2211) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:23], v[44:47], a[4:7], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 400) 
ds_read_b128 v[60:63], v3 offset:19456 // (op 3213) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[48:51], a[8:11], v40, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 626) 
ds_read_b128 v[64:67], v3 offset:20480 // (op 4215) Load local data 


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:23], v[32:35], a[12:15], v40, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 852) 
s_lshl_b32 s18, s34, 6 // (op 4224) 

v_add_u32 v20, v28, s18 // (op 4224) 


buffer_load_dwordx4 v[68:71], v20, s[64:67], 0 offen // (op 4224) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[24:27], a[16:19], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1213) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v41, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1267) 
s_waitcnt vmcnt(0)
v_bfe_u32 v20, v68, 0, 8 // (op 28627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[32:35], a[28:31], v41, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1294) 
v_bfe_u32 v21, v68, 8, 8 // (op 28627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[44:47], a[20:23], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1240) 
v_bfe_u32 v22, v68, 16, 8 // (op 28627) 

v_bfe_u32 v23, v68, 24, 8 // (op 28627) 
v_permlane16_swap_b32 v20, v21 // (op 28627) 
s_nop 0
v_permlane16_swap_b32 v22, v23 // (op 28627) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[36:39], v[24:27], a[32:35], v42, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2215) 
ds_read_b128 v[56:59], v3 offset:21504 // (op 5217) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:39], v[48:51], a[40:43], v42, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2269) 
v_permlane32_swap_b32 v20, v22 // (op 28627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[36:39], v[32:35], a[44:47], v42, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2296) 
v_permlane32_swap_b32 v21, v23 // (op 28627) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[36:39], v[44:47], a[36:39], v42, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2242) 
ds_read_b128 v[68:71], v3 offset:22528 // (op 6219) Load local data 




ds_read_b128 v[36:39], v3 offset:23552 // (op 7221) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:63], v[24:27], a[48:51], v43, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3217) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:63], v[48:51], a[56:59], v43, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3271) 
ds_read_b128 v[72:75], v3 offset:24576 // (op 8223) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[60:63], v[32:35], a[60:63], v43, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3298) 
s_lshl_b32 s18, s34, 7 // (op 8232) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:63], v[44:47], a[52:55], v43, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3244) 
v_add_u32 v29, v28, s18 // (op 8232) 




buffer_load_dwordx4 v[40:43], v29, s[64:67], 0 offen // (op 8232) Load value


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[64:67], v[44:47], a[68:71], v20, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4246) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[64:67], v[24:27], a[64:67], v20, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4219) 
v_add_u32 v16, 4, v16 // (op 31397) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[64:67], v[48:51], a[72:75], v20, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4273) 
ds_read_b128 v[28:31], v3 offset:25600 // (op 9225) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[64:67], v[32:35], a[76:79], v20, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4300) 
ds_read_b128 v[60:63], v3 offset:26624 // (op 10227) Load local data 




s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[56:59], v[44:47], a[84:87], v21, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5248) 
s_waitcnt vmcnt(0)
v_bfe_u32 v64, v40, 0, 8 // (op 28641) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[56:59], v[24:27], a[80:83], v21, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5221) 
v_bfe_u32 v65, v40, 8, 8 // (op 28641) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[56:59], v[48:51], a[88:91], v21, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5275) 
v_bfe_u32 v66, v40, 16, 8 // (op 28641) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[56:59], v[32:35], a[92:95], v21, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5302) 
v_bfe_u32 v67, v40, 24, 8 // (op 28641) 

v_permlane16_swap_b32 v64, v65 // (op 28641) 
s_nop 0
v_permlane16_swap_b32 v66, v67 // (op 28641) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[68:71], v[44:47], a[100:103], v22, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6250) 
s_nop 0
v_permlane32_swap_b32 v64, v66 // (op 28641) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[68:71], v[24:27], a[96:99], v22, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6223) 
v_permlane32_swap_b32 v65, v67 // (op 28641) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[68:71], v[48:51], a[104:107], v22, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6277) 
ds_read_b128 v[40:43], v3 offset:27648 // (op 11229) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[32:35], a[108:111], v22, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6304) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[36:39], v[44:47], a[116:119], v23, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7252) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[36:39], v[24:27], a[112:115], v23, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7225) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[36:39], v[48:51], a[120:123], v23, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7279) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[36:39], v[32:35], a[124:127], v23, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7306) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:75], v[44:47], a[132:135], v64, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8254) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:75], v[24:27], a[128:131], v64, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8227) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[72:75], v[48:51], a[136:139], v64, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8281) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:75], v[32:35], a[140:143], v64, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8308) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[28:31], v[44:47], a[148:151], v65, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9256) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[28:31], v[24:27], a[144:147], v65, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9229) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[28:31], v[48:51], a[152:155], v65, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9283) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[28:31], v[32:35], a[156:159], v65, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9310) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[60:63], v[44:47], a[164:167], v66, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10258) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[60:63], v[24:27], a[160:163], v66, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10231) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[60:63], v[48:51], a[168:171], v66, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10285) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[60:63], v[32:35], a[172:175], v66, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10312) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[40:43], v[44:47], a[180:183], v67, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11260) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[40:43], v[24:27], a[176:179], v67, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11233) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[40:43], v[48:51], a[184:187], v67, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11287) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[40:43], v[32:35], a[188:191], v67, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11314) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 11277) 
s_mov_b32 s50, s18 // (op 11277) convert

s_cmp_lt_u32 s49, s50 // (op 11277) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopTop_KLoopTail_11277 // (op 11277) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopTop_KLoopTail_11277 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ForLoopBottom_KLoopTail_11277:





















s_barrier  // (op 32177) 





















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ConditionalFalse_0_beta_114
v_accvgpr_read v4, a0 // (op 42) 
v_accvgpr_read v5, a1 // (op 42) 
v_accvgpr_read v6, a2 // (op 42) 
v_accvgpr_read v7, a3 // (op 42) 
v_mul_f32 v14, v13, v4 // (op 42) 
v_mul_f32 v15, v13, v5 // (op 42) 
v_mul_f32 v16, v13, v6 // (op 42) 
v_mul_f32 v17, v13, v7 // (op 42) 



v_mov_b32 v3, v12 // (op 48) reg expression

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 48) Store value


v_accvgpr_read v4, a4 // (op 13702) 
v_accvgpr_read v5, a5 // (op 13702) 
v_accvgpr_read v6, a6 // (op 13702) 
v_accvgpr_read v7, a7 // (op 13702) 
v_mul_f32 v14, v13, v4 // (op 13702) 
v_mul_f32 v15, v13, v5 // (op 13702) 
v_mul_f32 v16, v13, v6 // (op 13702) 
v_mul_f32 v17, v13, v7 // (op 13702) 



s_lshl_b32 s0, s4, 6 // (op 13704) 
v_add_u32 v3, v12, s0 // (op 13704) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13704) Store value


v_accvgpr_read v4, a8 // (op 13714) 
v_accvgpr_read v5, a9 // (op 13714) 
v_accvgpr_read v6, a10 // (op 13714) 
v_accvgpr_read v7, a11 // (op 13714) 
v_mul_f32 v14, v13, v4 // (op 13714) 
v_mul_f32 v15, v13, v5 // (op 13714) 
v_mul_f32 v16, v13, v6 // (op 13714) 
v_mul_f32 v17, v13, v7 // (op 13714) 



s_lshl_b32 s0, s4, 7 // (op 13716) 
v_add_u32 v3, v12, s0 // (op 13716) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13716) Store value


v_accvgpr_read v4, a12 // (op 13726) 
v_accvgpr_read v5, a13 // (op 13726) 
v_accvgpr_read v6, a14 // (op 13726) 
v_accvgpr_read v7, a15 // (op 13726) 
v_mul_f32 v14, v13, v4 // (op 13726) 
v_mul_f32 v15, v13, v5 // (op 13726) 
v_mul_f32 v16, v13, v6 // (op 13726) 
v_mul_f32 v17, v13, v7 // (op 13726) 



s_lshl_b32 s0, s4, 6 // (op 13728) 
s_mul_i32 s1, 3, s0 // (op 13728) 

v_add_u32 v3, v12, s1 // (op 13728) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13728) Store value


v_accvgpr_read v4, a16 // (op 13760) 
v_accvgpr_read v5, a17 // (op 13760) 
v_accvgpr_read v6, a18 // (op 13760) 
v_accvgpr_read v7, a19 // (op 13760) 
v_mul_f32 v14, v13, v4 // (op 13760) 
v_mul_f32 v15, v13, v5 // (op 13760) 
v_mul_f32 v16, v13, v6 // (op 13760) 
v_mul_f32 v17, v13, v7 // (op 13760) 



v_add_u32 v3, 64, v12 // (op 13763) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13763) Store value


v_accvgpr_read v4, a20 // (op 13773) 
v_accvgpr_read v5, a21 // (op 13773) 
v_accvgpr_read v6, a22 // (op 13773) 
v_accvgpr_read v7, a23 // (op 13773) 
v_mul_f32 v14, v13, v4 // (op 13773) 
v_mul_f32 v15, v13, v5 // (op 13773) 
v_mul_f32 v16, v13, v6 // (op 13773) 
v_mul_f32 v17, v13, v7 // (op 13773) 



s_lshl_b32 s0, s4, 6 // (op 13765) 
s_add_u32 s0, s0, 64 // (op 13765) 
v_add_u32 v3, v12, s0 // (op 13765) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13765) Store value


v_accvgpr_read v4, a24 // (op 13778) 
v_accvgpr_read v5, a25 // (op 13778) 
v_accvgpr_read v6, a26 // (op 13778) 
v_accvgpr_read v7, a27 // (op 13778) 
v_mul_f32 v14, v13, v4 // (op 13778) 
v_mul_f32 v15, v13, v5 // (op 13778) 
v_mul_f32 v16, v13, v6 // (op 13778) 
v_mul_f32 v17, v13, v7 // (op 13778) 



s_lshl_b32 s0, s4, 7 // (op 13767) 
s_add_u32 s0, s0, 64 // (op 13767) 
v_add_u32 v3, v12, s0 // (op 13767) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13767) Store value


v_accvgpr_read v4, a28 // (op 13783) 
v_accvgpr_read v5, a29 // (op 13783) 
v_accvgpr_read v6, a30 // (op 13783) 
v_accvgpr_read v7, a31 // (op 13783) 
v_mul_f32 v14, v13, v4 // (op 13783) 
v_mul_f32 v15, v13, v5 // (op 13783) 
v_mul_f32 v16, v13, v6 // (op 13783) 
v_mul_f32 v17, v13, v7 // (op 13783) 



s_lshl_b32 s0, s4, 6 // (op 13769) 
s_mul_i32 s1, 3, s0 // (op 13769) 
s_add_u32 s1, s1, 64 // (op 13769) 

v_add_u32 v3, v12, s1 // (op 13769) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13769) Store value


v_accvgpr_read v4, a32 // (op 13828) 
v_accvgpr_read v5, a33 // (op 13828) 
v_accvgpr_read v6, a34 // (op 13828) 
v_accvgpr_read v7, a35 // (op 13828) 
v_mul_f32 v14, v13, v4 // (op 13828) 
v_mul_f32 v15, v13, v5 // (op 13828) 
v_mul_f32 v16, v13, v6 // (op 13828) 
v_mul_f32 v17, v13, v7 // (op 13828) 



v_add_u32 v3, 128, v12 // (op 13831) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13831) Store value


v_accvgpr_read v4, a36 // (op 13841) 
v_accvgpr_read v5, a37 // (op 13841) 
v_accvgpr_read v6, a38 // (op 13841) 
v_accvgpr_read v7, a39 // (op 13841) 
v_mul_f32 v14, v13, v4 // (op 13841) 
v_mul_f32 v15, v13, v5 // (op 13841) 
v_mul_f32 v16, v13, v6 // (op 13841) 
v_mul_f32 v17, v13, v7 // (op 13841) 



s_lshl_b32 s0, s4, 6 // (op 13833) 
s_add_u32 s0, s0, 128 // (op 13833) 
v_add_u32 v3, v12, s0 // (op 13833) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13833) Store value


v_accvgpr_read v4, a40 // (op 13846) 
v_accvgpr_read v5, a41 // (op 13846) 
v_accvgpr_read v6, a42 // (op 13846) 
v_accvgpr_read v7, a43 // (op 13846) 
v_mul_f32 v14, v13, v4 // (op 13846) 
v_mul_f32 v15, v13, v5 // (op 13846) 
v_mul_f32 v16, v13, v6 // (op 13846) 
v_mul_f32 v17, v13, v7 // (op 13846) 



s_lshl_b32 s0, s4, 7 // (op 13835) 
s_add_u32 s0, s0, 128 // (op 13835) 
v_add_u32 v3, v12, s0 // (op 13835) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13835) Store value


v_accvgpr_read v4, a44 // (op 13851) 
v_accvgpr_read v5, a45 // (op 13851) 
v_accvgpr_read v6, a46 // (op 13851) 
v_accvgpr_read v7, a47 // (op 13851) 
v_mul_f32 v14, v13, v4 // (op 13851) 
v_mul_f32 v15, v13, v5 // (op 13851) 
v_mul_f32 v16, v13, v6 // (op 13851) 
v_mul_f32 v17, v13, v7 // (op 13851) 



s_lshl_b32 s0, s4, 6 // (op 13837) 
s_mul_i32 s1, 3, s0 // (op 13837) 
s_add_u32 s1, s1, 128 // (op 13837) 

v_add_u32 v3, v12, s1 // (op 13837) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13837) Store value


v_accvgpr_read v4, a48 // (op 13896) 
v_accvgpr_read v5, a49 // (op 13896) 
v_accvgpr_read v6, a50 // (op 13896) 
v_accvgpr_read v7, a51 // (op 13896) 
v_mul_f32 v14, v13, v4 // (op 13896) 
v_mul_f32 v15, v13, v5 // (op 13896) 
v_mul_f32 v16, v13, v6 // (op 13896) 
v_mul_f32 v17, v13, v7 // (op 13896) 



v_add_u32 v3, 192, v12 // (op 13899) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13899) Store value


v_accvgpr_read v4, a52 // (op 13909) 
v_accvgpr_read v5, a53 // (op 13909) 
v_accvgpr_read v6, a54 // (op 13909) 
v_accvgpr_read v7, a55 // (op 13909) 
v_mul_f32 v14, v13, v4 // (op 13909) 
v_mul_f32 v15, v13, v5 // (op 13909) 
v_mul_f32 v16, v13, v6 // (op 13909) 
v_mul_f32 v17, v13, v7 // (op 13909) 



s_lshl_b32 s0, s4, 6 // (op 13901) 
s_add_u32 s0, s0, 192 // (op 13901) 
v_add_u32 v3, v12, s0 // (op 13901) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13901) Store value


v_accvgpr_read v4, a56 // (op 13914) 
v_accvgpr_read v5, a57 // (op 13914) 
v_accvgpr_read v6, a58 // (op 13914) 
v_accvgpr_read v7, a59 // (op 13914) 
v_mul_f32 v14, v13, v4 // (op 13914) 
v_mul_f32 v15, v13, v5 // (op 13914) 
v_mul_f32 v16, v13, v6 // (op 13914) 
v_mul_f32 v17, v13, v7 // (op 13914) 



s_lshl_b32 s0, s4, 7 // (op 13903) 
s_add_u32 s0, s0, 192 // (op 13903) 
v_add_u32 v3, v12, s0 // (op 13903) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13903) Store value


v_accvgpr_read v4, a60 // (op 13919) 
v_accvgpr_read v5, a61 // (op 13919) 
v_accvgpr_read v6, a62 // (op 13919) 
v_accvgpr_read v7, a63 // (op 13919) 
v_mul_f32 v14, v13, v4 // (op 13919) 
v_mul_f32 v15, v13, v5 // (op 13919) 
v_mul_f32 v16, v13, v6 // (op 13919) 
v_mul_f32 v17, v13, v7 // (op 13919) 



s_lshl_b32 s0, s4, 6 // (op 13905) 
s_mul_i32 s1, 3, s0 // (op 13905) 
s_add_u32 s1, s1, 192 // (op 13905) 

v_add_u32 v3, v12, s1 // (op 13905) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13905) Store value


v_accvgpr_read v4, a64 // (op 13964) 
v_accvgpr_read v5, a65 // (op 13964) 
v_accvgpr_read v6, a66 // (op 13964) 
v_accvgpr_read v7, a67 // (op 13964) 
v_mul_f32 v14, v13, v4 // (op 13964) 
v_mul_f32 v15, v13, v5 // (op 13964) 
v_mul_f32 v16, v13, v6 // (op 13964) 
v_mul_f32 v17, v13, v7 // (op 13964) 



v_add_u32 v3, 256, v12 // (op 13967) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13967) Store value


v_accvgpr_read v4, a68 // (op 13977) 
v_accvgpr_read v5, a69 // (op 13977) 
v_accvgpr_read v6, a70 // (op 13977) 
v_accvgpr_read v7, a71 // (op 13977) 
v_mul_f32 v14, v13, v4 // (op 13977) 
v_mul_f32 v15, v13, v5 // (op 13977) 
v_mul_f32 v16, v13, v6 // (op 13977) 
v_mul_f32 v17, v13, v7 // (op 13977) 



s_lshl_b32 s0, s4, 6 // (op 13969) 
s_add_u32 s0, s0, 256 // (op 13969) 
v_add_u32 v3, v12, s0 // (op 13969) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13969) Store value


v_accvgpr_read v4, a72 // (op 13982) 
v_accvgpr_read v5, a73 // (op 13982) 
v_accvgpr_read v6, a74 // (op 13982) 
v_accvgpr_read v7, a75 // (op 13982) 
v_mul_f32 v14, v13, v4 // (op 13982) 
v_mul_f32 v15, v13, v5 // (op 13982) 
v_mul_f32 v16, v13, v6 // (op 13982) 
v_mul_f32 v17, v13, v7 // (op 13982) 



s_lshl_b32 s0, s4, 7 // (op 13971) 
s_add_u32 s0, s0, 256 // (op 13971) 
v_add_u32 v3, v12, s0 // (op 13971) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13971) Store value


v_accvgpr_read v4, a76 // (op 13987) 
v_accvgpr_read v5, a77 // (op 13987) 
v_accvgpr_read v6, a78 // (op 13987) 
v_accvgpr_read v7, a79 // (op 13987) 
v_mul_f32 v14, v13, v4 // (op 13987) 
v_mul_f32 v15, v13, v5 // (op 13987) 
v_mul_f32 v16, v13, v6 // (op 13987) 
v_mul_f32 v17, v13, v7 // (op 13987) 



s_lshl_b32 s0, s4, 6 // (op 13973) 
s_mul_i32 s1, 3, s0 // (op 13973) 
s_add_u32 s1, s1, 256 // (op 13973) 

v_add_u32 v3, v12, s1 // (op 13973) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 13973) Store value


v_accvgpr_read v4, a80 // (op 14032) 
v_accvgpr_read v5, a81 // (op 14032) 
v_accvgpr_read v6, a82 // (op 14032) 
v_accvgpr_read v7, a83 // (op 14032) 
v_mul_f32 v14, v13, v4 // (op 14032) 
v_mul_f32 v15, v13, v5 // (op 14032) 
v_mul_f32 v16, v13, v6 // (op 14032) 
v_mul_f32 v17, v13, v7 // (op 14032) 



v_add_u32 v3, 320, v12 // (op 14035) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14035) Store value


v_accvgpr_read v4, a84 // (op 14045) 
v_accvgpr_read v5, a85 // (op 14045) 
v_accvgpr_read v6, a86 // (op 14045) 
v_accvgpr_read v7, a87 // (op 14045) 
v_mul_f32 v14, v13, v4 // (op 14045) 
v_mul_f32 v15, v13, v5 // (op 14045) 
v_mul_f32 v16, v13, v6 // (op 14045) 
v_mul_f32 v17, v13, v7 // (op 14045) 



s_lshl_b32 s0, s4, 6 // (op 14037) 
s_add_u32 s0, s0, 320 // (op 14037) 
v_add_u32 v3, v12, s0 // (op 14037) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14037) Store value


v_accvgpr_read v4, a88 // (op 14050) 
v_accvgpr_read v5, a89 // (op 14050) 
v_accvgpr_read v6, a90 // (op 14050) 
v_accvgpr_read v7, a91 // (op 14050) 
v_mul_f32 v14, v13, v4 // (op 14050) 
v_mul_f32 v15, v13, v5 // (op 14050) 
v_mul_f32 v16, v13, v6 // (op 14050) 
v_mul_f32 v17, v13, v7 // (op 14050) 



s_lshl_b32 s0, s4, 7 // (op 14039) 
s_add_u32 s0, s0, 320 // (op 14039) 
v_add_u32 v3, v12, s0 // (op 14039) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14039) Store value


v_accvgpr_read v4, a92 // (op 14055) 
v_accvgpr_read v5, a93 // (op 14055) 
v_accvgpr_read v6, a94 // (op 14055) 
v_accvgpr_read v7, a95 // (op 14055) 
v_mul_f32 v14, v13, v4 // (op 14055) 
v_mul_f32 v15, v13, v5 // (op 14055) 
v_mul_f32 v16, v13, v6 // (op 14055) 
v_mul_f32 v17, v13, v7 // (op 14055) 



s_lshl_b32 s0, s4, 6 // (op 14041) 
s_mul_i32 s1, 3, s0 // (op 14041) 
s_add_u32 s1, s1, 320 // (op 14041) 

v_add_u32 v3, v12, s1 // (op 14041) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14041) Store value


v_accvgpr_read v4, a96 // (op 14100) 
v_accvgpr_read v5, a97 // (op 14100) 
v_accvgpr_read v6, a98 // (op 14100) 
v_accvgpr_read v7, a99 // (op 14100) 
v_mul_f32 v14, v13, v4 // (op 14100) 
v_mul_f32 v15, v13, v5 // (op 14100) 
v_mul_f32 v16, v13, v6 // (op 14100) 
v_mul_f32 v17, v13, v7 // (op 14100) 



v_add_u32 v3, 384, v12 // (op 14103) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14103) Store value


v_accvgpr_read v4, a100 // (op 14113) 
v_accvgpr_read v5, a101 // (op 14113) 
v_accvgpr_read v6, a102 // (op 14113) 
v_accvgpr_read v7, a103 // (op 14113) 
v_mul_f32 v14, v13, v4 // (op 14113) 
v_mul_f32 v15, v13, v5 // (op 14113) 
v_mul_f32 v16, v13, v6 // (op 14113) 
v_mul_f32 v17, v13, v7 // (op 14113) 



s_lshl_b32 s0, s4, 6 // (op 14105) 
s_add_u32 s0, s0, 384 // (op 14105) 
v_add_u32 v3, v12, s0 // (op 14105) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14105) Store value


v_accvgpr_read v4, a104 // (op 14118) 
v_accvgpr_read v5, a105 // (op 14118) 
v_accvgpr_read v6, a106 // (op 14118) 
v_accvgpr_read v7, a107 // (op 14118) 
v_mul_f32 v14, v13, v4 // (op 14118) 
v_mul_f32 v15, v13, v5 // (op 14118) 
v_mul_f32 v16, v13, v6 // (op 14118) 
v_mul_f32 v17, v13, v7 // (op 14118) 



s_lshl_b32 s0, s4, 7 // (op 14107) 
s_add_u32 s0, s0, 384 // (op 14107) 
v_add_u32 v3, v12, s0 // (op 14107) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14107) Store value


v_accvgpr_read v4, a108 // (op 14123) 
v_accvgpr_read v5, a109 // (op 14123) 
v_accvgpr_read v6, a110 // (op 14123) 
v_accvgpr_read v7, a111 // (op 14123) 
v_mul_f32 v14, v13, v4 // (op 14123) 
v_mul_f32 v15, v13, v5 // (op 14123) 
v_mul_f32 v16, v13, v6 // (op 14123) 
v_mul_f32 v17, v13, v7 // (op 14123) 



s_lshl_b32 s0, s4, 6 // (op 14109) 
s_mul_i32 s1, 3, s0 // (op 14109) 
s_add_u32 s1, s1, 384 // (op 14109) 

v_add_u32 v3, v12, s1 // (op 14109) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14109) Store value


v_accvgpr_read v4, a112 // (op 14168) 
v_accvgpr_read v5, a113 // (op 14168) 
v_accvgpr_read v6, a114 // (op 14168) 
v_accvgpr_read v7, a115 // (op 14168) 
v_mul_f32 v14, v13, v4 // (op 14168) 
v_mul_f32 v15, v13, v5 // (op 14168) 
v_mul_f32 v16, v13, v6 // (op 14168) 
v_mul_f32 v17, v13, v7 // (op 14168) 



v_add_u32 v3, 448, v12 // (op 14171) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14171) Store value


v_accvgpr_read v4, a116 // (op 14181) 
v_accvgpr_read v5, a117 // (op 14181) 
v_accvgpr_read v6, a118 // (op 14181) 
v_accvgpr_read v7, a119 // (op 14181) 
v_mul_f32 v14, v13, v4 // (op 14181) 
v_mul_f32 v15, v13, v5 // (op 14181) 
v_mul_f32 v16, v13, v6 // (op 14181) 
v_mul_f32 v17, v13, v7 // (op 14181) 



s_lshl_b32 s0, s4, 6 // (op 14173) 
s_add_u32 s0, s0, 448 // (op 14173) 
v_add_u32 v3, v12, s0 // (op 14173) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14173) Store value


v_accvgpr_read v4, a120 // (op 14186) 
v_accvgpr_read v5, a121 // (op 14186) 
v_accvgpr_read v6, a122 // (op 14186) 
v_accvgpr_read v7, a123 // (op 14186) 
v_mul_f32 v14, v13, v4 // (op 14186) 
v_mul_f32 v15, v13, v5 // (op 14186) 
v_mul_f32 v16, v13, v6 // (op 14186) 
v_mul_f32 v17, v13, v7 // (op 14186) 



s_lshl_b32 s0, s4, 7 // (op 14175) 
s_add_u32 s0, s0, 448 // (op 14175) 
v_add_u32 v3, v12, s0 // (op 14175) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14175) Store value


v_accvgpr_read v4, a124 // (op 14191) 
v_accvgpr_read v5, a125 // (op 14191) 
v_accvgpr_read v6, a126 // (op 14191) 
v_accvgpr_read v7, a127 // (op 14191) 
v_mul_f32 v14, v13, v4 // (op 14191) 
v_mul_f32 v15, v13, v5 // (op 14191) 
v_mul_f32 v16, v13, v6 // (op 14191) 
v_mul_f32 v17, v13, v7 // (op 14191) 



s_lshl_b32 s0, s4, 6 // (op 14177) 
s_mul_i32 s1, 3, s0 // (op 14177) 
s_add_u32 s1, s1, 448 // (op 14177) 

v_add_u32 v3, v12, s1 // (op 14177) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14177) Store value


v_accvgpr_read v4, a128 // (op 14236) 
v_accvgpr_read v5, a129 // (op 14236) 
v_accvgpr_read v6, a130 // (op 14236) 
v_accvgpr_read v7, a131 // (op 14236) 
v_mul_f32 v14, v13, v4 // (op 14236) 
v_mul_f32 v15, v13, v5 // (op 14236) 
v_mul_f32 v16, v13, v6 // (op 14236) 
v_mul_f32 v17, v13, v7 // (op 14236) 



v_add_u32 v3, 512, v12 // (op 14239) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14239) Store value


v_accvgpr_read v4, a132 // (op 14249) 
v_accvgpr_read v5, a133 // (op 14249) 
v_accvgpr_read v6, a134 // (op 14249) 
v_accvgpr_read v7, a135 // (op 14249) 
v_mul_f32 v14, v13, v4 // (op 14249) 
v_mul_f32 v15, v13, v5 // (op 14249) 
v_mul_f32 v16, v13, v6 // (op 14249) 
v_mul_f32 v17, v13, v7 // (op 14249) 



s_lshl_b32 s0, s4, 6 // (op 14241) 
s_add_u32 s0, s0, 512 // (op 14241) 
v_add_u32 v3, v12, s0 // (op 14241) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14241) Store value


v_accvgpr_read v4, a136 // (op 14254) 
v_accvgpr_read v5, a137 // (op 14254) 
v_accvgpr_read v6, a138 // (op 14254) 
v_accvgpr_read v7, a139 // (op 14254) 
v_mul_f32 v14, v13, v4 // (op 14254) 
v_mul_f32 v15, v13, v5 // (op 14254) 
v_mul_f32 v16, v13, v6 // (op 14254) 
v_mul_f32 v17, v13, v7 // (op 14254) 



s_lshl_b32 s0, s4, 7 // (op 14243) 
s_add_u32 s0, s0, 512 // (op 14243) 
v_add_u32 v3, v12, s0 // (op 14243) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14243) Store value


v_accvgpr_read v4, a140 // (op 14259) 
v_accvgpr_read v5, a141 // (op 14259) 
v_accvgpr_read v6, a142 // (op 14259) 
v_accvgpr_read v7, a143 // (op 14259) 
v_mul_f32 v14, v13, v4 // (op 14259) 
v_mul_f32 v15, v13, v5 // (op 14259) 
v_mul_f32 v16, v13, v6 // (op 14259) 
v_mul_f32 v17, v13, v7 // (op 14259) 



s_lshl_b32 s0, s4, 6 // (op 14245) 
s_mul_i32 s1, 3, s0 // (op 14245) 
s_add_u32 s1, s1, 512 // (op 14245) 

v_add_u32 v3, v12, s1 // (op 14245) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14245) Store value


v_accvgpr_read v4, a144 // (op 14304) 
v_accvgpr_read v5, a145 // (op 14304) 
v_accvgpr_read v6, a146 // (op 14304) 
v_accvgpr_read v7, a147 // (op 14304) 
v_mul_f32 v14, v13, v4 // (op 14304) 
v_mul_f32 v15, v13, v5 // (op 14304) 
v_mul_f32 v16, v13, v6 // (op 14304) 
v_mul_f32 v17, v13, v7 // (op 14304) 



v_add_u32 v3, 576, v12 // (op 14307) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14307) Store value


v_accvgpr_read v4, a148 // (op 14317) 
v_accvgpr_read v5, a149 // (op 14317) 
v_accvgpr_read v6, a150 // (op 14317) 
v_accvgpr_read v7, a151 // (op 14317) 
v_mul_f32 v14, v13, v4 // (op 14317) 
v_mul_f32 v15, v13, v5 // (op 14317) 
v_mul_f32 v16, v13, v6 // (op 14317) 
v_mul_f32 v17, v13, v7 // (op 14317) 



s_lshl_b32 s0, s4, 6 // (op 14309) 
s_add_u32 s0, s0, 576 // (op 14309) 
v_add_u32 v3, v12, s0 // (op 14309) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14309) Store value


v_accvgpr_read v4, a152 // (op 14322) 
v_accvgpr_read v5, a153 // (op 14322) 
v_accvgpr_read v6, a154 // (op 14322) 
v_accvgpr_read v7, a155 // (op 14322) 
v_mul_f32 v14, v13, v4 // (op 14322) 
v_mul_f32 v15, v13, v5 // (op 14322) 
v_mul_f32 v16, v13, v6 // (op 14322) 
v_mul_f32 v17, v13, v7 // (op 14322) 



s_lshl_b32 s0, s4, 7 // (op 14311) 
s_add_u32 s0, s0, 576 // (op 14311) 
v_add_u32 v3, v12, s0 // (op 14311) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14311) Store value


v_accvgpr_read v4, a156 // (op 14327) 
v_accvgpr_read v5, a157 // (op 14327) 
v_accvgpr_read v6, a158 // (op 14327) 
v_accvgpr_read v7, a159 // (op 14327) 
v_mul_f32 v14, v13, v4 // (op 14327) 
v_mul_f32 v15, v13, v5 // (op 14327) 
v_mul_f32 v16, v13, v6 // (op 14327) 
v_mul_f32 v17, v13, v7 // (op 14327) 



s_lshl_b32 s0, s4, 6 // (op 14313) 
s_mul_i32 s1, 3, s0 // (op 14313) 
s_add_u32 s1, s1, 576 // (op 14313) 

v_add_u32 v3, v12, s1 // (op 14313) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14313) Store value


v_accvgpr_read v4, a160 // (op 14372) 
v_accvgpr_read v5, a161 // (op 14372) 
v_accvgpr_read v6, a162 // (op 14372) 
v_accvgpr_read v7, a163 // (op 14372) 
v_mul_f32 v14, v13, v4 // (op 14372) 
v_mul_f32 v15, v13, v5 // (op 14372) 
v_mul_f32 v16, v13, v6 // (op 14372) 
v_mul_f32 v17, v13, v7 // (op 14372) 



v_add_u32 v3, 640, v12 // (op 14375) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14375) Store value


v_accvgpr_read v4, a164 // (op 14385) 
v_accvgpr_read v5, a165 // (op 14385) 
v_accvgpr_read v6, a166 // (op 14385) 
v_accvgpr_read v7, a167 // (op 14385) 
v_mul_f32 v14, v13, v4 // (op 14385) 
v_mul_f32 v15, v13, v5 // (op 14385) 
v_mul_f32 v16, v13, v6 // (op 14385) 
v_mul_f32 v17, v13, v7 // (op 14385) 



s_lshl_b32 s0, s4, 6 // (op 14377) 
s_add_u32 s0, s0, 640 // (op 14377) 
v_add_u32 v3, v12, s0 // (op 14377) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14377) Store value


v_accvgpr_read v4, a168 // (op 14390) 
v_accvgpr_read v5, a169 // (op 14390) 
v_accvgpr_read v6, a170 // (op 14390) 
v_accvgpr_read v7, a171 // (op 14390) 
v_mul_f32 v14, v13, v4 // (op 14390) 
v_mul_f32 v15, v13, v5 // (op 14390) 
v_mul_f32 v16, v13, v6 // (op 14390) 
v_mul_f32 v17, v13, v7 // (op 14390) 



s_lshl_b32 s0, s4, 7 // (op 14379) 
s_add_u32 s0, s0, 640 // (op 14379) 
v_add_u32 v3, v12, s0 // (op 14379) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14379) Store value


v_accvgpr_read v4, a172 // (op 14395) 
v_accvgpr_read v5, a173 // (op 14395) 
v_accvgpr_read v6, a174 // (op 14395) 
v_accvgpr_read v7, a175 // (op 14395) 
v_mul_f32 v14, v13, v4 // (op 14395) 
v_mul_f32 v15, v13, v5 // (op 14395) 
v_mul_f32 v16, v13, v6 // (op 14395) 
v_mul_f32 v17, v13, v7 // (op 14395) 



s_lshl_b32 s0, s4, 6 // (op 14381) 
s_mul_i32 s1, 3, s0 // (op 14381) 
s_add_u32 s1, s1, 640 // (op 14381) 

v_add_u32 v3, v12, s1 // (op 14381) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14381) Store value


v_accvgpr_read v4, a176 // (op 14440) 
v_accvgpr_read v5, a177 // (op 14440) 
v_accvgpr_read v6, a178 // (op 14440) 
v_accvgpr_read v7, a179 // (op 14440) 
v_mul_f32 v14, v13, v4 // (op 14440) 
v_mul_f32 v15, v13, v5 // (op 14440) 
v_mul_f32 v16, v13, v6 // (op 14440) 
v_mul_f32 v17, v13, v7 // (op 14440) 



v_add_u32 v3, 704, v12 // (op 14443) 

buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14443) Store value


v_accvgpr_read v4, a180 // (op 14453) 
v_accvgpr_read v5, a181 // (op 14453) 
v_accvgpr_read v6, a182 // (op 14453) 
v_accvgpr_read v7, a183 // (op 14453) 
v_mul_f32 v14, v13, v4 // (op 14453) 
v_mul_f32 v15, v13, v5 // (op 14453) 
v_mul_f32 v16, v13, v6 // (op 14453) 
v_mul_f32 v17, v13, v7 // (op 14453) 



s_lshl_b32 s0, s4, 6 // (op 14445) 
s_add_u32 s0, s0, 704 // (op 14445) 
v_add_u32 v3, v12, s0 // (op 14445) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14445) Store value


v_accvgpr_read v4, a184 // (op 14458) 
v_accvgpr_read v5, a185 // (op 14458) 
v_accvgpr_read v6, a186 // (op 14458) 
v_accvgpr_read v7, a187 // (op 14458) 
v_mul_f32 v14, v13, v4 // (op 14458) 
v_mul_f32 v15, v13, v5 // (op 14458) 
v_mul_f32 v16, v13, v6 // (op 14458) 
v_mul_f32 v17, v13, v7 // (op 14458) 



s_lshl_b32 s0, s4, 7 // (op 14447) 
s_add_u32 s0, s0, 704 // (op 14447) 
v_add_u32 v3, v12, s0 // (op 14447) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14447) Store value


v_accvgpr_read v4, a188 // (op 14463) 
v_accvgpr_read v5, a189 // (op 14463) 
v_accvgpr_read v6, a190 // (op 14463) 
v_accvgpr_read v7, a191 // (op 14463) 
v_mul_f32 v14, v13, v4 // (op 14463) 
v_mul_f32 v15, v13, v5 // (op 14463) 
v_mul_f32 v16, v13, v6 // (op 14463) 
v_mul_f32 v17, v13, v7 // (op 14463) 



s_lshl_b32 s0, s4, 6 // (op 14449) 
s_mul_i32 s1, 3, s0 // (op 14449) 
s_add_u32 s1, s1, 704 // (op 14449) 

v_add_u32 v3, v12, s1 // (op 14449) 


buffer_store_dwordx4 v[14:17], v3, s[60:63], 0 offen // (op 14449) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ConditionalBottom_0_beta_114
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ConditionalFalse_0_beta_114:


v_add_u32 v3, 704, v11 // (op 15551) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15551) Load value

v_accvgpr_read v14, a176 // (op 15539) 
v_accvgpr_read v15, a177 // (op 15539) 
v_accvgpr_read v16, a178 // (op 15539) 
v_accvgpr_read v17, a179 // (op 15539) 
v_mul_f32 v18, v13, v14 // (op 15539) 
v_mul_f32 v19, v13, v15 // (op 15539) 
v_mul_f32 v20, v13, v16 // (op 15539) 
v_mul_f32 v21, v13, v17 // (op 15539) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15550) 
v_mul_f32 v15, v0, v5 // (op 15550) 
v_mul_f32 v16, v0, v6 // (op 15550) 
v_mul_f32 v17, v0, v7 // (op 15550) 

v_add_f32 v4, v14, v18 // (op 15540) 
v_add_f32 v5, v15, v19 // (op 15540) 
v_add_f32 v6, v16, v20 // (op 15540) 
v_add_f32 v7, v17, v21 // (op 15540) 




v_add_u32 v3, 704, v12 // (op 15542) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15542) Store value




s_lshl_b32 s0, s2, 6 // (op 15555) 
s_add_u32 s0, s0, 704 // (op 15555) 
v_add_u32 v3, v11, s0 // (op 15555) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15555) Load value

v_accvgpr_read v14, a180 // (op 15565) 
v_accvgpr_read v15, a181 // (op 15565) 
v_accvgpr_read v16, a182 // (op 15565) 
v_accvgpr_read v17, a183 // (op 15565) 
v_mul_f32 v18, v13, v14 // (op 15565) 
v_mul_f32 v19, v13, v15 // (op 15565) 
v_mul_f32 v20, v13, v16 // (op 15565) 
v_mul_f32 v21, v13, v17 // (op 15565) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15553) 
v_mul_f32 v15, v0, v5 // (op 15553) 
v_mul_f32 v16, v0, v6 // (op 15553) 
v_mul_f32 v17, v0, v7 // (op 15553) 

v_add_f32 v4, v14, v18 // (op 15554) 
v_add_f32 v5, v15, v19 // (op 15554) 
v_add_f32 v6, v16, v20 // (op 15554) 
v_add_f32 v7, v17, v21 // (op 15554) 




s_lshl_b32 s0, s4, 6 // (op 15544) 
s_add_u32 s0, s0, 704 // (op 15544) 
v_add_u32 v3, v12, s0 // (op 15544) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15544) Store value




s_lshl_b32 s0, s2, 7 // (op 15559) 
s_add_u32 s0, s0, 704 // (op 15559) 
v_add_u32 v3, v11, s0 // (op 15559) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15559) Load value

v_accvgpr_read v14, a184 // (op 15567) 
v_accvgpr_read v15, a185 // (op 15567) 
v_accvgpr_read v16, a186 // (op 15567) 
v_accvgpr_read v17, a187 // (op 15567) 
v_mul_f32 v18, v13, v14 // (op 15567) 
v_mul_f32 v19, v13, v15 // (op 15567) 
v_mul_f32 v20, v13, v16 // (op 15567) 
v_mul_f32 v21, v13, v17 // (op 15567) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15557) 
v_mul_f32 v15, v0, v5 // (op 15557) 
v_mul_f32 v16, v0, v6 // (op 15557) 
v_mul_f32 v17, v0, v7 // (op 15557) 

v_add_f32 v4, v14, v18 // (op 15558) 
v_add_f32 v5, v15, v19 // (op 15558) 
v_add_f32 v6, v16, v20 // (op 15558) 
v_add_f32 v7, v17, v21 // (op 15558) 




s_lshl_b32 s0, s4, 7 // (op 15546) 
s_add_u32 s0, s0, 704 // (op 15546) 
v_add_u32 v3, v12, s0 // (op 15546) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15546) Store value




s_lshl_b32 s0, s2, 6 // (op 15563) 
s_mul_i32 s1, 3, s0 // (op 15563) 
s_add_u32 s1, s1, 704 // (op 15563) 

v_add_u32 v3, v11, s1 // (op 15563) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15563) Load value

v_accvgpr_read v14, a188 // (op 15569) 
v_accvgpr_read v15, a189 // (op 15569) 
v_accvgpr_read v16, a190 // (op 15569) 
v_accvgpr_read v17, a191 // (op 15569) 
v_mul_f32 v18, v13, v14 // (op 15569) 
v_mul_f32 v19, v13, v15 // (op 15569) 
v_mul_f32 v20, v13, v16 // (op 15569) 
v_mul_f32 v21, v13, v17 // (op 15569) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15561) 
v_mul_f32 v15, v0, v5 // (op 15561) 
v_mul_f32 v16, v0, v6 // (op 15561) 
v_mul_f32 v17, v0, v7 // (op 15561) 

v_add_f32 v4, v14, v18 // (op 15562) 
v_add_f32 v5, v15, v19 // (op 15562) 
v_add_f32 v6, v16, v20 // (op 15562) 
v_add_f32 v7, v17, v21 // (op 15562) 




s_lshl_b32 s0, s4, 6 // (op 15548) 
s_mul_i32 s1, 3, s0 // (op 15548) 
s_add_u32 s1, s1, 704 // (op 15548) 

v_add_u32 v3, v12, s1 // (op 15548) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15548) Store value




v_add_u32 v3, 640, v11 // (op 15472) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15472) Load value

v_accvgpr_read v14, a160 // (op 15460) 
v_accvgpr_read v15, a161 // (op 15460) 
v_accvgpr_read v16, a162 // (op 15460) 
v_accvgpr_read v17, a163 // (op 15460) 
v_mul_f32 v18, v13, v14 // (op 15460) 
v_mul_f32 v19, v13, v15 // (op 15460) 
v_mul_f32 v20, v13, v16 // (op 15460) 
v_mul_f32 v21, v13, v17 // (op 15460) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15471) 
v_mul_f32 v15, v0, v5 // (op 15471) 
v_mul_f32 v16, v0, v6 // (op 15471) 
v_mul_f32 v17, v0, v7 // (op 15471) 

v_add_f32 v4, v14, v18 // (op 15461) 
v_add_f32 v5, v15, v19 // (op 15461) 
v_add_f32 v6, v16, v20 // (op 15461) 
v_add_f32 v7, v17, v21 // (op 15461) 




v_add_u32 v3, 640, v12 // (op 15463) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15463) Store value




s_lshl_b32 s0, s2, 6 // (op 15476) 
s_add_u32 s0, s0, 640 // (op 15476) 
v_add_u32 v3, v11, s0 // (op 15476) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15476) Load value

v_accvgpr_read v14, a164 // (op 15486) 
v_accvgpr_read v15, a165 // (op 15486) 
v_accvgpr_read v16, a166 // (op 15486) 
v_accvgpr_read v17, a167 // (op 15486) 
v_mul_f32 v18, v13, v14 // (op 15486) 
v_mul_f32 v19, v13, v15 // (op 15486) 
v_mul_f32 v20, v13, v16 // (op 15486) 
v_mul_f32 v21, v13, v17 // (op 15486) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15474) 
v_mul_f32 v15, v0, v5 // (op 15474) 
v_mul_f32 v16, v0, v6 // (op 15474) 
v_mul_f32 v17, v0, v7 // (op 15474) 

v_add_f32 v4, v14, v18 // (op 15475) 
v_add_f32 v5, v15, v19 // (op 15475) 
v_add_f32 v6, v16, v20 // (op 15475) 
v_add_f32 v7, v17, v21 // (op 15475) 




s_lshl_b32 s0, s4, 6 // (op 15465) 
s_add_u32 s0, s0, 640 // (op 15465) 
v_add_u32 v3, v12, s0 // (op 15465) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15465) Store value




s_lshl_b32 s0, s2, 7 // (op 15480) 
s_add_u32 s0, s0, 640 // (op 15480) 
v_add_u32 v3, v11, s0 // (op 15480) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15480) Load value

v_accvgpr_read v14, a168 // (op 15488) 
v_accvgpr_read v15, a169 // (op 15488) 
v_accvgpr_read v16, a170 // (op 15488) 
v_accvgpr_read v17, a171 // (op 15488) 
v_mul_f32 v18, v13, v14 // (op 15488) 
v_mul_f32 v19, v13, v15 // (op 15488) 
v_mul_f32 v20, v13, v16 // (op 15488) 
v_mul_f32 v21, v13, v17 // (op 15488) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15478) 
v_mul_f32 v15, v0, v5 // (op 15478) 
v_mul_f32 v16, v0, v6 // (op 15478) 
v_mul_f32 v17, v0, v7 // (op 15478) 

v_add_f32 v4, v14, v18 // (op 15479) 
v_add_f32 v5, v15, v19 // (op 15479) 
v_add_f32 v6, v16, v20 // (op 15479) 
v_add_f32 v7, v17, v21 // (op 15479) 




s_lshl_b32 s0, s4, 7 // (op 15467) 
s_add_u32 s0, s0, 640 // (op 15467) 
v_add_u32 v3, v12, s0 // (op 15467) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15467) Store value




s_lshl_b32 s0, s2, 6 // (op 15484) 
s_mul_i32 s1, 3, s0 // (op 15484) 
s_add_u32 s1, s1, 640 // (op 15484) 

v_add_u32 v3, v11, s1 // (op 15484) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15484) Load value

v_accvgpr_read v14, a172 // (op 15490) 
v_accvgpr_read v15, a173 // (op 15490) 
v_accvgpr_read v16, a174 // (op 15490) 
v_accvgpr_read v17, a175 // (op 15490) 
v_mul_f32 v18, v13, v14 // (op 15490) 
v_mul_f32 v19, v13, v15 // (op 15490) 
v_mul_f32 v20, v13, v16 // (op 15490) 
v_mul_f32 v21, v13, v17 // (op 15490) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15482) 
v_mul_f32 v15, v0, v5 // (op 15482) 
v_mul_f32 v16, v0, v6 // (op 15482) 
v_mul_f32 v17, v0, v7 // (op 15482) 

v_add_f32 v4, v14, v18 // (op 15483) 
v_add_f32 v5, v15, v19 // (op 15483) 
v_add_f32 v6, v16, v20 // (op 15483) 
v_add_f32 v7, v17, v21 // (op 15483) 




s_lshl_b32 s0, s4, 6 // (op 15469) 
s_mul_i32 s1, 3, s0 // (op 15469) 
s_add_u32 s1, s1, 640 // (op 15469) 

v_add_u32 v3, v12, s1 // (op 15469) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15469) Store value




v_add_u32 v3, 576, v11 // (op 15393) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15393) Load value

v_accvgpr_read v14, a144 // (op 15381) 
v_accvgpr_read v15, a145 // (op 15381) 
v_accvgpr_read v16, a146 // (op 15381) 
v_accvgpr_read v17, a147 // (op 15381) 
v_mul_f32 v18, v13, v14 // (op 15381) 
v_mul_f32 v19, v13, v15 // (op 15381) 
v_mul_f32 v20, v13, v16 // (op 15381) 
v_mul_f32 v21, v13, v17 // (op 15381) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15392) 
v_mul_f32 v15, v0, v5 // (op 15392) 
v_mul_f32 v16, v0, v6 // (op 15392) 
v_mul_f32 v17, v0, v7 // (op 15392) 

v_add_f32 v4, v14, v18 // (op 15382) 
v_add_f32 v5, v15, v19 // (op 15382) 
v_add_f32 v6, v16, v20 // (op 15382) 
v_add_f32 v7, v17, v21 // (op 15382) 




v_add_u32 v3, 576, v12 // (op 15384) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15384) Store value




s_lshl_b32 s0, s2, 6 // (op 15397) 
s_add_u32 s0, s0, 576 // (op 15397) 
v_add_u32 v3, v11, s0 // (op 15397) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15397) Load value

v_accvgpr_read v14, a148 // (op 15407) 
v_accvgpr_read v15, a149 // (op 15407) 
v_accvgpr_read v16, a150 // (op 15407) 
v_accvgpr_read v17, a151 // (op 15407) 
v_mul_f32 v18, v13, v14 // (op 15407) 
v_mul_f32 v19, v13, v15 // (op 15407) 
v_mul_f32 v20, v13, v16 // (op 15407) 
v_mul_f32 v21, v13, v17 // (op 15407) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15395) 
v_mul_f32 v15, v0, v5 // (op 15395) 
v_mul_f32 v16, v0, v6 // (op 15395) 
v_mul_f32 v17, v0, v7 // (op 15395) 

v_add_f32 v4, v14, v18 // (op 15396) 
v_add_f32 v5, v15, v19 // (op 15396) 
v_add_f32 v6, v16, v20 // (op 15396) 
v_add_f32 v7, v17, v21 // (op 15396) 




s_lshl_b32 s0, s4, 6 // (op 15386) 
s_add_u32 s0, s0, 576 // (op 15386) 
v_add_u32 v3, v12, s0 // (op 15386) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15386) Store value




s_lshl_b32 s0, s2, 7 // (op 15401) 
s_add_u32 s0, s0, 576 // (op 15401) 
v_add_u32 v3, v11, s0 // (op 15401) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15401) Load value

v_accvgpr_read v14, a152 // (op 15409) 
v_accvgpr_read v15, a153 // (op 15409) 
v_accvgpr_read v16, a154 // (op 15409) 
v_accvgpr_read v17, a155 // (op 15409) 
v_mul_f32 v18, v13, v14 // (op 15409) 
v_mul_f32 v19, v13, v15 // (op 15409) 
v_mul_f32 v20, v13, v16 // (op 15409) 
v_mul_f32 v21, v13, v17 // (op 15409) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15399) 
v_mul_f32 v15, v0, v5 // (op 15399) 
v_mul_f32 v16, v0, v6 // (op 15399) 
v_mul_f32 v17, v0, v7 // (op 15399) 

v_add_f32 v4, v14, v18 // (op 15400) 
v_add_f32 v5, v15, v19 // (op 15400) 
v_add_f32 v6, v16, v20 // (op 15400) 
v_add_f32 v7, v17, v21 // (op 15400) 




s_lshl_b32 s0, s4, 7 // (op 15388) 
s_add_u32 s0, s0, 576 // (op 15388) 
v_add_u32 v3, v12, s0 // (op 15388) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15388) Store value




s_lshl_b32 s0, s2, 6 // (op 15405) 
s_mul_i32 s1, 3, s0 // (op 15405) 
s_add_u32 s1, s1, 576 // (op 15405) 

v_add_u32 v3, v11, s1 // (op 15405) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15405) Load value

v_accvgpr_read v14, a156 // (op 15411) 
v_accvgpr_read v15, a157 // (op 15411) 
v_accvgpr_read v16, a158 // (op 15411) 
v_accvgpr_read v17, a159 // (op 15411) 
v_mul_f32 v18, v13, v14 // (op 15411) 
v_mul_f32 v19, v13, v15 // (op 15411) 
v_mul_f32 v20, v13, v16 // (op 15411) 
v_mul_f32 v21, v13, v17 // (op 15411) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15403) 
v_mul_f32 v15, v0, v5 // (op 15403) 
v_mul_f32 v16, v0, v6 // (op 15403) 
v_mul_f32 v17, v0, v7 // (op 15403) 

v_add_f32 v4, v14, v18 // (op 15404) 
v_add_f32 v5, v15, v19 // (op 15404) 
v_add_f32 v6, v16, v20 // (op 15404) 
v_add_f32 v7, v17, v21 // (op 15404) 




s_lshl_b32 s0, s4, 6 // (op 15390) 
s_mul_i32 s1, 3, s0 // (op 15390) 
s_add_u32 s1, s1, 576 // (op 15390) 

v_add_u32 v3, v12, s1 // (op 15390) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15390) Store value




v_add_u32 v3, 512, v11 // (op 15314) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15314) Load value

v_accvgpr_read v14, a128 // (op 15302) 
v_accvgpr_read v15, a129 // (op 15302) 
v_accvgpr_read v16, a130 // (op 15302) 
v_accvgpr_read v17, a131 // (op 15302) 
v_mul_f32 v18, v13, v14 // (op 15302) 
v_mul_f32 v19, v13, v15 // (op 15302) 
v_mul_f32 v20, v13, v16 // (op 15302) 
v_mul_f32 v21, v13, v17 // (op 15302) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15313) 
v_mul_f32 v15, v0, v5 // (op 15313) 
v_mul_f32 v16, v0, v6 // (op 15313) 
v_mul_f32 v17, v0, v7 // (op 15313) 

v_add_f32 v4, v14, v18 // (op 15303) 
v_add_f32 v5, v15, v19 // (op 15303) 
v_add_f32 v6, v16, v20 // (op 15303) 
v_add_f32 v7, v17, v21 // (op 15303) 




v_add_u32 v3, 512, v12 // (op 15305) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15305) Store value




s_lshl_b32 s0, s2, 6 // (op 15318) 
s_add_u32 s0, s0, 512 // (op 15318) 
v_add_u32 v3, v11, s0 // (op 15318) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15318) Load value

v_accvgpr_read v14, a132 // (op 15328) 
v_accvgpr_read v15, a133 // (op 15328) 
v_accvgpr_read v16, a134 // (op 15328) 
v_accvgpr_read v17, a135 // (op 15328) 
v_mul_f32 v18, v13, v14 // (op 15328) 
v_mul_f32 v19, v13, v15 // (op 15328) 
v_mul_f32 v20, v13, v16 // (op 15328) 
v_mul_f32 v21, v13, v17 // (op 15328) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15316) 
v_mul_f32 v15, v0, v5 // (op 15316) 
v_mul_f32 v16, v0, v6 // (op 15316) 
v_mul_f32 v17, v0, v7 // (op 15316) 

v_add_f32 v4, v14, v18 // (op 15317) 
v_add_f32 v5, v15, v19 // (op 15317) 
v_add_f32 v6, v16, v20 // (op 15317) 
v_add_f32 v7, v17, v21 // (op 15317) 




s_lshl_b32 s0, s4, 6 // (op 15307) 
s_add_u32 s0, s0, 512 // (op 15307) 
v_add_u32 v3, v12, s0 // (op 15307) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15307) Store value




s_lshl_b32 s0, s2, 7 // (op 15322) 
s_add_u32 s0, s0, 512 // (op 15322) 
v_add_u32 v3, v11, s0 // (op 15322) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15322) Load value

v_accvgpr_read v14, a136 // (op 15330) 
v_accvgpr_read v15, a137 // (op 15330) 
v_accvgpr_read v16, a138 // (op 15330) 
v_accvgpr_read v17, a139 // (op 15330) 
v_mul_f32 v18, v13, v14 // (op 15330) 
v_mul_f32 v19, v13, v15 // (op 15330) 
v_mul_f32 v20, v13, v16 // (op 15330) 
v_mul_f32 v21, v13, v17 // (op 15330) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15320) 
v_mul_f32 v15, v0, v5 // (op 15320) 
v_mul_f32 v16, v0, v6 // (op 15320) 
v_mul_f32 v17, v0, v7 // (op 15320) 

v_add_f32 v4, v14, v18 // (op 15321) 
v_add_f32 v5, v15, v19 // (op 15321) 
v_add_f32 v6, v16, v20 // (op 15321) 
v_add_f32 v7, v17, v21 // (op 15321) 




s_lshl_b32 s0, s4, 7 // (op 15309) 
s_add_u32 s0, s0, 512 // (op 15309) 
v_add_u32 v3, v12, s0 // (op 15309) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15309) Store value




s_lshl_b32 s0, s2, 6 // (op 15326) 
s_mul_i32 s1, 3, s0 // (op 15326) 
s_add_u32 s1, s1, 512 // (op 15326) 

v_add_u32 v3, v11, s1 // (op 15326) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15326) Load value

v_accvgpr_read v14, a140 // (op 15332) 
v_accvgpr_read v15, a141 // (op 15332) 
v_accvgpr_read v16, a142 // (op 15332) 
v_accvgpr_read v17, a143 // (op 15332) 
v_mul_f32 v18, v13, v14 // (op 15332) 
v_mul_f32 v19, v13, v15 // (op 15332) 
v_mul_f32 v20, v13, v16 // (op 15332) 
v_mul_f32 v21, v13, v17 // (op 15332) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15324) 
v_mul_f32 v15, v0, v5 // (op 15324) 
v_mul_f32 v16, v0, v6 // (op 15324) 
v_mul_f32 v17, v0, v7 // (op 15324) 

v_add_f32 v4, v14, v18 // (op 15325) 
v_add_f32 v5, v15, v19 // (op 15325) 
v_add_f32 v6, v16, v20 // (op 15325) 
v_add_f32 v7, v17, v21 // (op 15325) 




s_lshl_b32 s0, s4, 6 // (op 15311) 
s_mul_i32 s1, 3, s0 // (op 15311) 
s_add_u32 s1, s1, 512 // (op 15311) 

v_add_u32 v3, v12, s1 // (op 15311) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15311) Store value




v_add_u32 v3, 448, v11 // (op 15235) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15235) Load value

v_accvgpr_read v14, a112 // (op 15223) 
v_accvgpr_read v15, a113 // (op 15223) 
v_accvgpr_read v16, a114 // (op 15223) 
v_accvgpr_read v17, a115 // (op 15223) 
v_mul_f32 v18, v13, v14 // (op 15223) 
v_mul_f32 v19, v13, v15 // (op 15223) 
v_mul_f32 v20, v13, v16 // (op 15223) 
v_mul_f32 v21, v13, v17 // (op 15223) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15234) 
v_mul_f32 v15, v0, v5 // (op 15234) 
v_mul_f32 v16, v0, v6 // (op 15234) 
v_mul_f32 v17, v0, v7 // (op 15234) 

v_add_f32 v4, v14, v18 // (op 15224) 
v_add_f32 v5, v15, v19 // (op 15224) 
v_add_f32 v6, v16, v20 // (op 15224) 
v_add_f32 v7, v17, v21 // (op 15224) 




v_add_u32 v3, 448, v12 // (op 15226) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15226) Store value




s_lshl_b32 s0, s2, 6 // (op 15239) 
s_add_u32 s0, s0, 448 // (op 15239) 
v_add_u32 v3, v11, s0 // (op 15239) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15239) Load value

v_accvgpr_read v14, a116 // (op 15249) 
v_accvgpr_read v15, a117 // (op 15249) 
v_accvgpr_read v16, a118 // (op 15249) 
v_accvgpr_read v17, a119 // (op 15249) 
v_mul_f32 v18, v13, v14 // (op 15249) 
v_mul_f32 v19, v13, v15 // (op 15249) 
v_mul_f32 v20, v13, v16 // (op 15249) 
v_mul_f32 v21, v13, v17 // (op 15249) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15237) 
v_mul_f32 v15, v0, v5 // (op 15237) 
v_mul_f32 v16, v0, v6 // (op 15237) 
v_mul_f32 v17, v0, v7 // (op 15237) 

v_add_f32 v4, v14, v18 // (op 15238) 
v_add_f32 v5, v15, v19 // (op 15238) 
v_add_f32 v6, v16, v20 // (op 15238) 
v_add_f32 v7, v17, v21 // (op 15238) 




s_lshl_b32 s0, s4, 6 // (op 15228) 
s_add_u32 s0, s0, 448 // (op 15228) 
v_add_u32 v3, v12, s0 // (op 15228) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15228) Store value




s_lshl_b32 s0, s2, 7 // (op 15243) 
s_add_u32 s0, s0, 448 // (op 15243) 
v_add_u32 v3, v11, s0 // (op 15243) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15243) Load value

v_accvgpr_read v14, a120 // (op 15251) 
v_accvgpr_read v15, a121 // (op 15251) 
v_accvgpr_read v16, a122 // (op 15251) 
v_accvgpr_read v17, a123 // (op 15251) 
v_mul_f32 v18, v13, v14 // (op 15251) 
v_mul_f32 v19, v13, v15 // (op 15251) 
v_mul_f32 v20, v13, v16 // (op 15251) 
v_mul_f32 v21, v13, v17 // (op 15251) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15241) 
v_mul_f32 v15, v0, v5 // (op 15241) 
v_mul_f32 v16, v0, v6 // (op 15241) 
v_mul_f32 v17, v0, v7 // (op 15241) 

v_add_f32 v4, v14, v18 // (op 15242) 
v_add_f32 v5, v15, v19 // (op 15242) 
v_add_f32 v6, v16, v20 // (op 15242) 
v_add_f32 v7, v17, v21 // (op 15242) 




s_lshl_b32 s0, s4, 7 // (op 15230) 
s_add_u32 s0, s0, 448 // (op 15230) 
v_add_u32 v3, v12, s0 // (op 15230) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15230) Store value




s_lshl_b32 s0, s2, 6 // (op 15247) 
s_mul_i32 s1, 3, s0 // (op 15247) 
s_add_u32 s1, s1, 448 // (op 15247) 

v_add_u32 v3, v11, s1 // (op 15247) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15247) Load value

v_accvgpr_read v14, a124 // (op 15253) 
v_accvgpr_read v15, a125 // (op 15253) 
v_accvgpr_read v16, a126 // (op 15253) 
v_accvgpr_read v17, a127 // (op 15253) 
v_mul_f32 v18, v13, v14 // (op 15253) 
v_mul_f32 v19, v13, v15 // (op 15253) 
v_mul_f32 v20, v13, v16 // (op 15253) 
v_mul_f32 v21, v13, v17 // (op 15253) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15245) 
v_mul_f32 v15, v0, v5 // (op 15245) 
v_mul_f32 v16, v0, v6 // (op 15245) 
v_mul_f32 v17, v0, v7 // (op 15245) 

v_add_f32 v4, v14, v18 // (op 15246) 
v_add_f32 v5, v15, v19 // (op 15246) 
v_add_f32 v6, v16, v20 // (op 15246) 
v_add_f32 v7, v17, v21 // (op 15246) 




s_lshl_b32 s0, s4, 6 // (op 15232) 
s_mul_i32 s1, 3, s0 // (op 15232) 
s_add_u32 s1, s1, 448 // (op 15232) 

v_add_u32 v3, v12, s1 // (op 15232) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15232) Store value




v_add_u32 v3, 384, v11 // (op 15156) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15156) Load value

v_accvgpr_read v14, a96 // (op 15144) 
v_accvgpr_read v15, a97 // (op 15144) 
v_accvgpr_read v16, a98 // (op 15144) 
v_accvgpr_read v17, a99 // (op 15144) 
v_mul_f32 v18, v13, v14 // (op 15144) 
v_mul_f32 v19, v13, v15 // (op 15144) 
v_mul_f32 v20, v13, v16 // (op 15144) 
v_mul_f32 v21, v13, v17 // (op 15144) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15155) 
v_mul_f32 v15, v0, v5 // (op 15155) 
v_mul_f32 v16, v0, v6 // (op 15155) 
v_mul_f32 v17, v0, v7 // (op 15155) 

v_add_f32 v4, v14, v18 // (op 15145) 
v_add_f32 v5, v15, v19 // (op 15145) 
v_add_f32 v6, v16, v20 // (op 15145) 
v_add_f32 v7, v17, v21 // (op 15145) 




v_add_u32 v3, 384, v12 // (op 15147) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15147) Store value




s_lshl_b32 s0, s2, 6 // (op 15160) 
s_add_u32 s0, s0, 384 // (op 15160) 
v_add_u32 v3, v11, s0 // (op 15160) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15160) Load value

v_accvgpr_read v14, a100 // (op 15170) 
v_accvgpr_read v15, a101 // (op 15170) 
v_accvgpr_read v16, a102 // (op 15170) 
v_accvgpr_read v17, a103 // (op 15170) 
v_mul_f32 v18, v13, v14 // (op 15170) 
v_mul_f32 v19, v13, v15 // (op 15170) 
v_mul_f32 v20, v13, v16 // (op 15170) 
v_mul_f32 v21, v13, v17 // (op 15170) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15158) 
v_mul_f32 v15, v0, v5 // (op 15158) 
v_mul_f32 v16, v0, v6 // (op 15158) 
v_mul_f32 v17, v0, v7 // (op 15158) 

v_add_f32 v4, v14, v18 // (op 15159) 
v_add_f32 v5, v15, v19 // (op 15159) 
v_add_f32 v6, v16, v20 // (op 15159) 
v_add_f32 v7, v17, v21 // (op 15159) 




s_lshl_b32 s0, s4, 6 // (op 15149) 
s_add_u32 s0, s0, 384 // (op 15149) 
v_add_u32 v3, v12, s0 // (op 15149) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15149) Store value




s_lshl_b32 s0, s2, 7 // (op 15164) 
s_add_u32 s0, s0, 384 // (op 15164) 
v_add_u32 v3, v11, s0 // (op 15164) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15164) Load value

v_accvgpr_read v14, a104 // (op 15172) 
v_accvgpr_read v15, a105 // (op 15172) 
v_accvgpr_read v16, a106 // (op 15172) 
v_accvgpr_read v17, a107 // (op 15172) 
v_mul_f32 v18, v13, v14 // (op 15172) 
v_mul_f32 v19, v13, v15 // (op 15172) 
v_mul_f32 v20, v13, v16 // (op 15172) 
v_mul_f32 v21, v13, v17 // (op 15172) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15162) 
v_mul_f32 v15, v0, v5 // (op 15162) 
v_mul_f32 v16, v0, v6 // (op 15162) 
v_mul_f32 v17, v0, v7 // (op 15162) 

v_add_f32 v4, v14, v18 // (op 15163) 
v_add_f32 v5, v15, v19 // (op 15163) 
v_add_f32 v6, v16, v20 // (op 15163) 
v_add_f32 v7, v17, v21 // (op 15163) 




s_lshl_b32 s0, s4, 7 // (op 15151) 
s_add_u32 s0, s0, 384 // (op 15151) 
v_add_u32 v3, v12, s0 // (op 15151) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15151) Store value




s_lshl_b32 s0, s2, 6 // (op 15168) 
s_mul_i32 s1, 3, s0 // (op 15168) 
s_add_u32 s1, s1, 384 // (op 15168) 

v_add_u32 v3, v11, s1 // (op 15168) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15168) Load value

v_accvgpr_read v14, a108 // (op 15174) 
v_accvgpr_read v15, a109 // (op 15174) 
v_accvgpr_read v16, a110 // (op 15174) 
v_accvgpr_read v17, a111 // (op 15174) 
v_mul_f32 v18, v13, v14 // (op 15174) 
v_mul_f32 v19, v13, v15 // (op 15174) 
v_mul_f32 v20, v13, v16 // (op 15174) 
v_mul_f32 v21, v13, v17 // (op 15174) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15166) 
v_mul_f32 v15, v0, v5 // (op 15166) 
v_mul_f32 v16, v0, v6 // (op 15166) 
v_mul_f32 v17, v0, v7 // (op 15166) 

v_add_f32 v4, v14, v18 // (op 15167) 
v_add_f32 v5, v15, v19 // (op 15167) 
v_add_f32 v6, v16, v20 // (op 15167) 
v_add_f32 v7, v17, v21 // (op 15167) 




s_lshl_b32 s0, s4, 6 // (op 15153) 
s_mul_i32 s1, 3, s0 // (op 15153) 
s_add_u32 s1, s1, 384 // (op 15153) 

v_add_u32 v3, v12, s1 // (op 15153) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15153) Store value




v_add_u32 v3, 320, v11 // (op 15077) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15077) Load value

v_accvgpr_read v14, a80 // (op 15065) 
v_accvgpr_read v15, a81 // (op 15065) 
v_accvgpr_read v16, a82 // (op 15065) 
v_accvgpr_read v17, a83 // (op 15065) 
v_mul_f32 v18, v13, v14 // (op 15065) 
v_mul_f32 v19, v13, v15 // (op 15065) 
v_mul_f32 v20, v13, v16 // (op 15065) 
v_mul_f32 v21, v13, v17 // (op 15065) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15076) 
v_mul_f32 v15, v0, v5 // (op 15076) 
v_mul_f32 v16, v0, v6 // (op 15076) 
v_mul_f32 v17, v0, v7 // (op 15076) 

v_add_f32 v4, v14, v18 // (op 15066) 
v_add_f32 v5, v15, v19 // (op 15066) 
v_add_f32 v6, v16, v20 // (op 15066) 
v_add_f32 v7, v17, v21 // (op 15066) 




v_add_u32 v3, 320, v12 // (op 15068) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15068) Store value




s_lshl_b32 s0, s2, 6 // (op 15081) 
s_add_u32 s0, s0, 320 // (op 15081) 
v_add_u32 v3, v11, s0 // (op 15081) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15081) Load value

v_accvgpr_read v14, a84 // (op 15091) 
v_accvgpr_read v15, a85 // (op 15091) 
v_accvgpr_read v16, a86 // (op 15091) 
v_accvgpr_read v17, a87 // (op 15091) 
v_mul_f32 v18, v13, v14 // (op 15091) 
v_mul_f32 v19, v13, v15 // (op 15091) 
v_mul_f32 v20, v13, v16 // (op 15091) 
v_mul_f32 v21, v13, v17 // (op 15091) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15079) 
v_mul_f32 v15, v0, v5 // (op 15079) 
v_mul_f32 v16, v0, v6 // (op 15079) 
v_mul_f32 v17, v0, v7 // (op 15079) 

v_add_f32 v4, v14, v18 // (op 15080) 
v_add_f32 v5, v15, v19 // (op 15080) 
v_add_f32 v6, v16, v20 // (op 15080) 
v_add_f32 v7, v17, v21 // (op 15080) 




s_lshl_b32 s0, s4, 6 // (op 15070) 
s_add_u32 s0, s0, 320 // (op 15070) 
v_add_u32 v3, v12, s0 // (op 15070) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15070) Store value




s_lshl_b32 s0, s2, 7 // (op 15085) 
s_add_u32 s0, s0, 320 // (op 15085) 
v_add_u32 v3, v11, s0 // (op 15085) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15085) Load value

v_accvgpr_read v14, a88 // (op 15093) 
v_accvgpr_read v15, a89 // (op 15093) 
v_accvgpr_read v16, a90 // (op 15093) 
v_accvgpr_read v17, a91 // (op 15093) 
v_mul_f32 v18, v13, v14 // (op 15093) 
v_mul_f32 v19, v13, v15 // (op 15093) 
v_mul_f32 v20, v13, v16 // (op 15093) 
v_mul_f32 v21, v13, v17 // (op 15093) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15083) 
v_mul_f32 v15, v0, v5 // (op 15083) 
v_mul_f32 v16, v0, v6 // (op 15083) 
v_mul_f32 v17, v0, v7 // (op 15083) 

v_add_f32 v4, v14, v18 // (op 15084) 
v_add_f32 v5, v15, v19 // (op 15084) 
v_add_f32 v6, v16, v20 // (op 15084) 
v_add_f32 v7, v17, v21 // (op 15084) 




s_lshl_b32 s0, s4, 7 // (op 15072) 
s_add_u32 s0, s0, 320 // (op 15072) 
v_add_u32 v3, v12, s0 // (op 15072) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15072) Store value




s_lshl_b32 s0, s2, 6 // (op 15089) 
s_mul_i32 s1, 3, s0 // (op 15089) 
s_add_u32 s1, s1, 320 // (op 15089) 

v_add_u32 v3, v11, s1 // (op 15089) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15089) Load value

v_accvgpr_read v14, a92 // (op 15095) 
v_accvgpr_read v15, a93 // (op 15095) 
v_accvgpr_read v16, a94 // (op 15095) 
v_accvgpr_read v17, a95 // (op 15095) 
v_mul_f32 v18, v13, v14 // (op 15095) 
v_mul_f32 v19, v13, v15 // (op 15095) 
v_mul_f32 v20, v13, v16 // (op 15095) 
v_mul_f32 v21, v13, v17 // (op 15095) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15087) 
v_mul_f32 v15, v0, v5 // (op 15087) 
v_mul_f32 v16, v0, v6 // (op 15087) 
v_mul_f32 v17, v0, v7 // (op 15087) 

v_add_f32 v4, v14, v18 // (op 15088) 
v_add_f32 v5, v15, v19 // (op 15088) 
v_add_f32 v6, v16, v20 // (op 15088) 
v_add_f32 v7, v17, v21 // (op 15088) 




s_lshl_b32 s0, s4, 6 // (op 15074) 
s_mul_i32 s1, 3, s0 // (op 15074) 
s_add_u32 s1, s1, 320 // (op 15074) 

v_add_u32 v3, v12, s1 // (op 15074) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 15074) Store value




v_add_u32 v3, 256, v11 // (op 14998) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14998) Load value

v_accvgpr_read v14, a64 // (op 14986) 
v_accvgpr_read v15, a65 // (op 14986) 
v_accvgpr_read v16, a66 // (op 14986) 
v_accvgpr_read v17, a67 // (op 14986) 
v_mul_f32 v18, v13, v14 // (op 14986) 
v_mul_f32 v19, v13, v15 // (op 14986) 
v_mul_f32 v20, v13, v16 // (op 14986) 
v_mul_f32 v21, v13, v17 // (op 14986) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 14997) 
v_mul_f32 v15, v0, v5 // (op 14997) 
v_mul_f32 v16, v0, v6 // (op 14997) 
v_mul_f32 v17, v0, v7 // (op 14997) 

v_add_f32 v4, v14, v18 // (op 14987) 
v_add_f32 v5, v15, v19 // (op 14987) 
v_add_f32 v6, v16, v20 // (op 14987) 
v_add_f32 v7, v17, v21 // (op 14987) 




v_add_u32 v3, 256, v12 // (op 14989) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14989) Store value




s_lshl_b32 s0, s2, 6 // (op 15002) 
s_add_u32 s0, s0, 256 // (op 15002) 
v_add_u32 v3, v11, s0 // (op 15002) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15002) Load value

v_accvgpr_read v14, a68 // (op 15012) 
v_accvgpr_read v15, a69 // (op 15012) 
v_accvgpr_read v16, a70 // (op 15012) 
v_accvgpr_read v17, a71 // (op 15012) 
v_mul_f32 v18, v13, v14 // (op 15012) 
v_mul_f32 v19, v13, v15 // (op 15012) 
v_mul_f32 v20, v13, v16 // (op 15012) 
v_mul_f32 v21, v13, v17 // (op 15012) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15000) 
v_mul_f32 v15, v0, v5 // (op 15000) 
v_mul_f32 v16, v0, v6 // (op 15000) 
v_mul_f32 v17, v0, v7 // (op 15000) 

v_add_f32 v4, v14, v18 // (op 15001) 
v_add_f32 v5, v15, v19 // (op 15001) 
v_add_f32 v6, v16, v20 // (op 15001) 
v_add_f32 v7, v17, v21 // (op 15001) 




s_lshl_b32 s0, s4, 6 // (op 14991) 
s_add_u32 s0, s0, 256 // (op 14991) 
v_add_u32 v3, v12, s0 // (op 14991) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14991) Store value




s_lshl_b32 s0, s2, 7 // (op 15006) 
s_add_u32 s0, s0, 256 // (op 15006) 
v_add_u32 v3, v11, s0 // (op 15006) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15006) Load value

v_accvgpr_read v14, a72 // (op 15014) 
v_accvgpr_read v15, a73 // (op 15014) 
v_accvgpr_read v16, a74 // (op 15014) 
v_accvgpr_read v17, a75 // (op 15014) 
v_mul_f32 v18, v13, v14 // (op 15014) 
v_mul_f32 v19, v13, v15 // (op 15014) 
v_mul_f32 v20, v13, v16 // (op 15014) 
v_mul_f32 v21, v13, v17 // (op 15014) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15004) 
v_mul_f32 v15, v0, v5 // (op 15004) 
v_mul_f32 v16, v0, v6 // (op 15004) 
v_mul_f32 v17, v0, v7 // (op 15004) 

v_add_f32 v4, v14, v18 // (op 15005) 
v_add_f32 v5, v15, v19 // (op 15005) 
v_add_f32 v6, v16, v20 // (op 15005) 
v_add_f32 v7, v17, v21 // (op 15005) 




s_lshl_b32 s0, s4, 7 // (op 14993) 
s_add_u32 s0, s0, 256 // (op 14993) 
v_add_u32 v3, v12, s0 // (op 14993) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14993) Store value




s_lshl_b32 s0, s2, 6 // (op 15010) 
s_mul_i32 s1, 3, s0 // (op 15010) 
s_add_u32 s1, s1, 256 // (op 15010) 

v_add_u32 v3, v11, s1 // (op 15010) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 15010) Load value

v_accvgpr_read v14, a76 // (op 15016) 
v_accvgpr_read v15, a77 // (op 15016) 
v_accvgpr_read v16, a78 // (op 15016) 
v_accvgpr_read v17, a79 // (op 15016) 
v_mul_f32 v18, v13, v14 // (op 15016) 
v_mul_f32 v19, v13, v15 // (op 15016) 
v_mul_f32 v20, v13, v16 // (op 15016) 
v_mul_f32 v21, v13, v17 // (op 15016) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 15008) 
v_mul_f32 v15, v0, v5 // (op 15008) 
v_mul_f32 v16, v0, v6 // (op 15008) 
v_mul_f32 v17, v0, v7 // (op 15008) 

v_add_f32 v4, v14, v18 // (op 15009) 
v_add_f32 v5, v15, v19 // (op 15009) 
v_add_f32 v6, v16, v20 // (op 15009) 
v_add_f32 v7, v17, v21 // (op 15009) 




s_lshl_b32 s0, s4, 6 // (op 14995) 
s_mul_i32 s1, 3, s0 // (op 14995) 
s_add_u32 s1, s1, 256 // (op 14995) 

v_add_u32 v3, v12, s1 // (op 14995) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14995) Store value




v_add_u32 v3, 192, v11 // (op 14919) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14919) Load value

v_accvgpr_read v14, a48 // (op 14907) 
v_accvgpr_read v15, a49 // (op 14907) 
v_accvgpr_read v16, a50 // (op 14907) 
v_accvgpr_read v17, a51 // (op 14907) 
v_mul_f32 v18, v13, v14 // (op 14907) 
v_mul_f32 v19, v13, v15 // (op 14907) 
v_mul_f32 v20, v13, v16 // (op 14907) 
v_mul_f32 v21, v13, v17 // (op 14907) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 14918) 
v_mul_f32 v15, v0, v5 // (op 14918) 
v_mul_f32 v16, v0, v6 // (op 14918) 
v_mul_f32 v17, v0, v7 // (op 14918) 

v_add_f32 v4, v14, v18 // (op 14908) 
v_add_f32 v5, v15, v19 // (op 14908) 
v_add_f32 v6, v16, v20 // (op 14908) 
v_add_f32 v7, v17, v21 // (op 14908) 




v_add_u32 v3, 192, v12 // (op 14910) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14910) Store value




s_lshl_b32 s0, s2, 6 // (op 14923) 
s_add_u32 s0, s0, 192 // (op 14923) 
v_add_u32 v3, v11, s0 // (op 14923) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14923) Load value

v_accvgpr_read v14, a52 // (op 14933) 
v_accvgpr_read v15, a53 // (op 14933) 
v_accvgpr_read v16, a54 // (op 14933) 
v_accvgpr_read v17, a55 // (op 14933) 
v_mul_f32 v18, v13, v14 // (op 14933) 
v_mul_f32 v19, v13, v15 // (op 14933) 
v_mul_f32 v20, v13, v16 // (op 14933) 
v_mul_f32 v21, v13, v17 // (op 14933) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 14921) 
v_mul_f32 v15, v0, v5 // (op 14921) 
v_mul_f32 v16, v0, v6 // (op 14921) 
v_mul_f32 v17, v0, v7 // (op 14921) 

v_add_f32 v4, v14, v18 // (op 14922) 
v_add_f32 v5, v15, v19 // (op 14922) 
v_add_f32 v6, v16, v20 // (op 14922) 
v_add_f32 v7, v17, v21 // (op 14922) 




s_lshl_b32 s0, s4, 6 // (op 14912) 
s_add_u32 s0, s0, 192 // (op 14912) 
v_add_u32 v3, v12, s0 // (op 14912) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14912) Store value




s_lshl_b32 s0, s2, 7 // (op 14927) 
s_add_u32 s0, s0, 192 // (op 14927) 
v_add_u32 v3, v11, s0 // (op 14927) 


buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14927) Load value

v_accvgpr_read v14, a56 // (op 14935) 
v_accvgpr_read v15, a57 // (op 14935) 
v_accvgpr_read v16, a58 // (op 14935) 
v_accvgpr_read v17, a59 // (op 14935) 
v_mul_f32 v18, v13, v14 // (op 14935) 
v_mul_f32 v19, v13, v15 // (op 14935) 
v_mul_f32 v20, v13, v16 // (op 14935) 
v_mul_f32 v21, v13, v17 // (op 14935) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 14925) 
v_mul_f32 v15, v0, v5 // (op 14925) 
v_mul_f32 v16, v0, v6 // (op 14925) 
v_mul_f32 v17, v0, v7 // (op 14925) 

v_add_f32 v4, v14, v18 // (op 14926) 
v_add_f32 v5, v15, v19 // (op 14926) 
v_add_f32 v6, v16, v20 // (op 14926) 
v_add_f32 v7, v17, v21 // (op 14926) 




s_lshl_b32 s0, s4, 7 // (op 14914) 
s_add_u32 s0, s0, 192 // (op 14914) 
v_add_u32 v3, v12, s0 // (op 14914) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14914) Store value




s_lshl_b32 s0, s2, 6 // (op 14931) 
s_mul_i32 s1, 3, s0 // (op 14931) 
s_add_u32 s1, s1, 192 // (op 14931) 

v_add_u32 v3, v11, s1 // (op 14931) 


v_accvgpr_read v4, a60 // (op 14937) 
v_accvgpr_read v5, a61 // (op 14937) 
v_accvgpr_read v6, a62 // (op 14937) 
v_accvgpr_read v7, a63 // (op 14937) 
v_mul_f32 v14, v13, v4 // (op 14937) 
v_mul_f32 v15, v13, v5 // (op 14937) 
v_mul_f32 v16, v13, v6 // (op 14937) 
v_mul_f32 v17, v13, v7 // (op 14937) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14931) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14929) 
v_mul_f32 v19, v0, v5 // (op 14929) 
v_mul_f32 v20, v0, v6 // (op 14929) 
v_mul_f32 v21, v0, v7 // (op 14929) 

v_add_f32 v4, v18, v14 // (op 14930) 
v_add_f32 v5, v19, v15 // (op 14930) 
v_add_f32 v6, v20, v16 // (op 14930) 
v_add_f32 v7, v21, v17 // (op 14930) 




s_lshl_b32 s0, s4, 6 // (op 14916) 
s_mul_i32 s1, 3, s0 // (op 14916) 
s_add_u32 s1, s1, 192 // (op 14916) 

v_add_u32 v3, v12, s1 // (op 14916) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14916) Store value




v_add_u32 v3, 128, v11 // (op 14840) 

s_nop 0
v_accvgpr_read v4, a32 // (op 14828) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v5, a33 // (op 14828) 
v_accvgpr_read v6, a34 // (op 14828) 
v_accvgpr_read v7, a35 // (op 14828) 
v_mul_f32 v14, v13, v4 // (op 14828) 
v_mul_f32 v15, v13, v5 // (op 14828) 
v_mul_f32 v16, v13, v6 // (op 14828) 
v_mul_f32 v17, v13, v7 // (op 14828) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14840) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14839) 
v_mul_f32 v19, v0, v5 // (op 14839) 
v_mul_f32 v20, v0, v6 // (op 14839) 
v_mul_f32 v21, v0, v7 // (op 14839) 

v_add_f32 v4, v18, v14 // (op 14829) 
v_add_f32 v5, v19, v15 // (op 14829) 
v_add_f32 v6, v20, v16 // (op 14829) 
v_add_f32 v7, v21, v17 // (op 14829) 




v_add_u32 v3, 128, v12 // (op 14831) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14831) Store value




s_lshl_b32 s0, s2, 6 // (op 14844) 
s_add_u32 s0, s0, 128 // (op 14844) 
v_add_u32 v3, v11, s0 // (op 14844) 


v_accvgpr_read v4, a36 // (op 14854) 
v_accvgpr_read v5, a37 // (op 14854) 
v_accvgpr_read v6, a38 // (op 14854) 
v_accvgpr_read v7, a39 // (op 14854) 
v_mul_f32 v14, v13, v4 // (op 14854) 
v_mul_f32 v15, v13, v5 // (op 14854) 
v_mul_f32 v16, v13, v6 // (op 14854) 
v_mul_f32 v17, v13, v7 // (op 14854) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14844) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14842) 
v_mul_f32 v19, v0, v5 // (op 14842) 
v_mul_f32 v20, v0, v6 // (op 14842) 
v_mul_f32 v21, v0, v7 // (op 14842) 

v_add_f32 v4, v18, v14 // (op 14843) 
v_add_f32 v5, v19, v15 // (op 14843) 
v_add_f32 v6, v20, v16 // (op 14843) 
v_add_f32 v7, v21, v17 // (op 14843) 




s_lshl_b32 s0, s4, 6 // (op 14833) 
s_add_u32 s0, s0, 128 // (op 14833) 
v_add_u32 v3, v12, s0 // (op 14833) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14833) Store value




s_lshl_b32 s0, s2, 7 // (op 14848) 
s_add_u32 s0, s0, 128 // (op 14848) 
v_add_u32 v3, v11, s0 // (op 14848) 


v_accvgpr_read v4, a40 // (op 14856) 
v_accvgpr_read v5, a41 // (op 14856) 
v_accvgpr_read v6, a42 // (op 14856) 
v_accvgpr_read v7, a43 // (op 14856) 
v_mul_f32 v14, v13, v4 // (op 14856) 
v_mul_f32 v15, v13, v5 // (op 14856) 
v_mul_f32 v16, v13, v6 // (op 14856) 
v_mul_f32 v17, v13, v7 // (op 14856) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14848) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14846) 
v_mul_f32 v19, v0, v5 // (op 14846) 
v_mul_f32 v20, v0, v6 // (op 14846) 
v_mul_f32 v21, v0, v7 // (op 14846) 

v_add_f32 v4, v18, v14 // (op 14847) 
v_add_f32 v5, v19, v15 // (op 14847) 
v_add_f32 v6, v20, v16 // (op 14847) 
v_add_f32 v7, v21, v17 // (op 14847) 




s_lshl_b32 s0, s4, 7 // (op 14835) 
s_add_u32 s0, s0, 128 // (op 14835) 
v_add_u32 v3, v12, s0 // (op 14835) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14835) Store value




s_lshl_b32 s0, s2, 6 // (op 14852) 
s_mul_i32 s1, 3, s0 // (op 14852) 
s_add_u32 s1, s1, 128 // (op 14852) 

v_add_u32 v3, v11, s1 // (op 14852) 


v_accvgpr_read v4, a44 // (op 14858) 
v_accvgpr_read v5, a45 // (op 14858) 
v_accvgpr_read v6, a46 // (op 14858) 
v_accvgpr_read v7, a47 // (op 14858) 
v_mul_f32 v14, v13, v4 // (op 14858) 
v_mul_f32 v15, v13, v5 // (op 14858) 
v_mul_f32 v16, v13, v6 // (op 14858) 
v_mul_f32 v17, v13, v7 // (op 14858) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14852) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14850) 
v_mul_f32 v19, v0, v5 // (op 14850) 
v_mul_f32 v20, v0, v6 // (op 14850) 
v_mul_f32 v21, v0, v7 // (op 14850) 

v_add_f32 v4, v18, v14 // (op 14851) 
v_add_f32 v5, v19, v15 // (op 14851) 
v_add_f32 v6, v20, v16 // (op 14851) 
v_add_f32 v7, v21, v17 // (op 14851) 




s_lshl_b32 s0, s4, 6 // (op 14837) 
s_mul_i32 s1, 3, s0 // (op 14837) 
s_add_u32 s1, s1, 128 // (op 14837) 

v_add_u32 v3, v12, s1 // (op 14837) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14837) Store value




v_add_u32 v3, 64, v11 // (op 14761) 

s_nop 0
v_accvgpr_read v4, a16 // (op 14749) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v5, a17 // (op 14749) 
v_accvgpr_read v6, a18 // (op 14749) 
v_accvgpr_read v7, a19 // (op 14749) 
v_mul_f32 v14, v13, v4 // (op 14749) 
v_mul_f32 v15, v13, v5 // (op 14749) 
v_mul_f32 v16, v13, v6 // (op 14749) 
v_mul_f32 v17, v13, v7 // (op 14749) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14761) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14760) 
v_mul_f32 v19, v0, v5 // (op 14760) 
v_mul_f32 v20, v0, v6 // (op 14760) 
v_mul_f32 v21, v0, v7 // (op 14760) 

v_add_f32 v4, v18, v14 // (op 14750) 
v_add_f32 v5, v19, v15 // (op 14750) 
v_add_f32 v6, v20, v16 // (op 14750) 
v_add_f32 v7, v21, v17 // (op 14750) 




v_add_u32 v3, 64, v12 // (op 14752) 

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14752) Store value




s_lshl_b32 s0, s2, 6 // (op 14765) 
s_add_u32 s0, s0, 64 // (op 14765) 
v_add_u32 v3, v11, s0 // (op 14765) 


v_accvgpr_read v4, a20 // (op 14775) 
v_accvgpr_read v5, a21 // (op 14775) 
v_accvgpr_read v6, a22 // (op 14775) 
v_accvgpr_read v7, a23 // (op 14775) 
v_mul_f32 v14, v13, v4 // (op 14775) 
v_mul_f32 v15, v13, v5 // (op 14775) 
v_mul_f32 v16, v13, v6 // (op 14775) 
v_mul_f32 v17, v13, v7 // (op 14775) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14765) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14763) 
v_mul_f32 v19, v0, v5 // (op 14763) 
v_mul_f32 v20, v0, v6 // (op 14763) 
v_mul_f32 v21, v0, v7 // (op 14763) 

v_add_f32 v4, v18, v14 // (op 14764) 
v_add_f32 v5, v19, v15 // (op 14764) 
v_add_f32 v6, v20, v16 // (op 14764) 
v_add_f32 v7, v21, v17 // (op 14764) 




s_lshl_b32 s0, s4, 6 // (op 14754) 
s_add_u32 s0, s0, 64 // (op 14754) 
v_add_u32 v3, v12, s0 // (op 14754) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14754) Store value




s_lshl_b32 s0, s2, 7 // (op 14769) 
s_add_u32 s0, s0, 64 // (op 14769) 
v_add_u32 v3, v11, s0 // (op 14769) 


v_accvgpr_read v4, a24 // (op 14777) 
v_accvgpr_read v5, a25 // (op 14777) 
v_accvgpr_read v6, a26 // (op 14777) 
v_accvgpr_read v7, a27 // (op 14777) 
v_mul_f32 v14, v13, v4 // (op 14777) 
v_mul_f32 v15, v13, v5 // (op 14777) 
v_mul_f32 v16, v13, v6 // (op 14777) 
v_mul_f32 v17, v13, v7 // (op 14777) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14769) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14767) 
v_mul_f32 v19, v0, v5 // (op 14767) 
v_mul_f32 v20, v0, v6 // (op 14767) 
v_mul_f32 v21, v0, v7 // (op 14767) 

v_add_f32 v4, v18, v14 // (op 14768) 
v_add_f32 v5, v19, v15 // (op 14768) 
v_add_f32 v6, v20, v16 // (op 14768) 
v_add_f32 v7, v21, v17 // (op 14768) 




s_lshl_b32 s0, s4, 7 // (op 14756) 
s_add_u32 s0, s0, 64 // (op 14756) 
v_add_u32 v3, v12, s0 // (op 14756) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14756) Store value




s_lshl_b32 s0, s2, 6 // (op 14773) 
s_mul_i32 s1, 3, s0 // (op 14773) 
s_add_u32 s1, s1, 64 // (op 14773) 

v_add_u32 v3, v11, s1 // (op 14773) 


v_accvgpr_read v4, a28 // (op 14779) 
v_accvgpr_read v5, a29 // (op 14779) 
v_accvgpr_read v6, a30 // (op 14779) 
v_accvgpr_read v7, a31 // (op 14779) 
v_mul_f32 v14, v13, v4 // (op 14779) 
v_mul_f32 v15, v13, v5 // (op 14779) 
v_mul_f32 v16, v13, v6 // (op 14779) 
v_mul_f32 v17, v13, v7 // (op 14779) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14773) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14771) 
v_mul_f32 v19, v0, v5 // (op 14771) 
v_mul_f32 v20, v0, v6 // (op 14771) 
v_mul_f32 v21, v0, v7 // (op 14771) 

v_add_f32 v4, v18, v14 // (op 14772) 
v_add_f32 v5, v19, v15 // (op 14772) 
v_add_f32 v6, v20, v16 // (op 14772) 
v_add_f32 v7, v21, v17 // (op 14772) 




s_lshl_b32 s0, s4, 6 // (op 14758) 
s_mul_i32 s1, 3, s0 // (op 14758) 
s_add_u32 s1, s1, 64 // (op 14758) 

v_add_u32 v3, v12, s1 // (op 14758) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14758) Store value




v_mov_b32 v3, v11 // (op 126) reg expression

s_nop 0
v_accvgpr_read v4, a0 // (op 123) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v5, a1 // (op 123) 
v_accvgpr_read v6, a2 // (op 123) 
v_accvgpr_read v7, a3 // (op 123) 
v_mul_f32 v14, v13, v4 // (op 123) 
v_mul_f32 v15, v13, v5 // (op 123) 
v_mul_f32 v16, v13, v6 // (op 123) 
v_mul_f32 v17, v13, v7 // (op 123) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 126) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 127) 
v_mul_f32 v19, v0, v5 // (op 127) 
v_mul_f32 v20, v0, v6 // (op 127) 
v_mul_f32 v21, v0, v7 // (op 127) 

v_add_f32 v4, v18, v14 // (op 124) 
v_add_f32 v5, v19, v15 // (op 124) 
v_add_f32 v6, v20, v16 // (op 124) 
v_add_f32 v7, v21, v17 // (op 124) 




v_mov_b32 v3, v12 // (op 125) reg expression

buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 125) Store value




s_lshl_b32 s0, s2, 6 // (op 14675) 
v_add_u32 v3, v11, s0 // (op 14675) 


v_accvgpr_read v4, a4 // (op 14672) 
v_accvgpr_read v5, a5 // (op 14672) 
v_accvgpr_read v6, a6 // (op 14672) 
v_accvgpr_read v7, a7 // (op 14672) 
v_mul_f32 v14, v13, v4 // (op 14672) 
v_mul_f32 v15, v13, v5 // (op 14672) 
v_mul_f32 v16, v13, v6 // (op 14672) 
v_mul_f32 v17, v13, v7 // (op 14672) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14675) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14676) 
v_mul_f32 v19, v0, v5 // (op 14676) 
v_mul_f32 v20, v0, v6 // (op 14676) 
v_mul_f32 v21, v0, v7 // (op 14676) 

v_add_f32 v4, v18, v14 // (op 14673) 
v_add_f32 v5, v19, v15 // (op 14673) 
v_add_f32 v6, v20, v16 // (op 14673) 
v_add_f32 v7, v21, v17 // (op 14673) 




s_lshl_b32 s0, s4, 6 // (op 14674) 
v_add_u32 v3, v12, s0 // (op 14674) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14674) Store value




s_lshl_b32 s0, s2, 7 // (op 14687) 
v_add_u32 v3, v11, s0 // (op 14687) 


v_accvgpr_read v4, a8 // (op 14684) 
v_accvgpr_read v5, a9 // (op 14684) 
v_accvgpr_read v6, a10 // (op 14684) 
v_accvgpr_read v7, a11 // (op 14684) 
v_mul_f32 v14, v13, v4 // (op 14684) 
v_mul_f32 v15, v13, v5 // (op 14684) 
v_mul_f32 v16, v13, v6 // (op 14684) 
v_mul_f32 v17, v13, v7 // (op 14684) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14687) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14688) 
v_mul_f32 v19, v0, v5 // (op 14688) 
v_mul_f32 v20, v0, v6 // (op 14688) 
v_mul_f32 v21, v0, v7 // (op 14688) 

v_add_f32 v4, v18, v14 // (op 14685) 
v_add_f32 v5, v19, v15 // (op 14685) 
v_add_f32 v6, v20, v16 // (op 14685) 
v_add_f32 v7, v21, v17 // (op 14685) 




s_lshl_b32 s0, s4, 7 // (op 14686) 
v_add_u32 v3, v12, s0 // (op 14686) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14686) Store value




s_lshl_b32 s0, s2, 6 // (op 14699) 
s_mul_i32 s1, 3, s0 // (op 14699) 

v_add_u32 v3, v11, s1 // (op 14699) 


v_accvgpr_read v4, a12 // (op 14696) 
v_accvgpr_read v5, a13 // (op 14696) 
v_accvgpr_read v6, a14 // (op 14696) 
v_accvgpr_read v7, a15 // (op 14696) 
v_mul_f32 v14, v13, v4 // (op 14696) 
v_mul_f32 v15, v13, v5 // (op 14696) 
v_mul_f32 v16, v13, v6 // (op 14696) 
v_mul_f32 v17, v13, v7 // (op 14696) 

buffer_load_dwordx4 v[4:7], v3, s[56:59], 0 offen // (op 14699) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v0, v4 // (op 14700) 
v_mul_f32 v19, v0, v5 // (op 14700) 
v_mul_f32 v20, v0, v6 // (op 14700) 
v_mul_f32 v21, v0, v7 // (op 14700) 

v_add_f32 v4, v18, v14 // (op 14697) 
v_add_f32 v5, v19, v15 // (op 14697) 
v_add_f32 v6, v20, v16 // (op 14697) 
v_add_f32 v7, v21, v17 // (op 14697) 




s_lshl_b32 s0, s4, 6 // (op 14698) 
s_mul_i32 s1, 3, s0 // (op 14698) 

v_add_u32 v3, v12, s1 // (op 14698) 


buffer_store_dwordx4 v[4:7], v3, s[60:63], 0 offen // (op 14698) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__ConditionalBottom_0_beta_114:
























































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM_
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM_, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM__end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM_
  .amdhsa_next_free_vgpr 372
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 114688 // lds bytes
  .amdhsa_accum_offset 180
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
    macrotile: [192, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [12, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x256x128_UR_4_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 114688
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 78
    .vgpr_count: 180
    .agpr_count: 192
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 156, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 156, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
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

