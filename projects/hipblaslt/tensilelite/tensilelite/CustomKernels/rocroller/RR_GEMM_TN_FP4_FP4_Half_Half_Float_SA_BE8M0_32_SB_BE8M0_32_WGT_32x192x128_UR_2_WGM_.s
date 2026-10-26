

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM_,@function
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2881) 
v_and_b32 v3, 15, v1 // (op 2881) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2881) 


v_and_b32 v0, 63, v1 // (op 2881) 
v_lshrrev_b32 v3, 4, v0 // (op 2881) 

v_lshlrev_b32 v0, 5, v3 // (op 2881) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2881) 


v_lshlrev_b32 v0, 2, v3 // (op 2881) 

v_lshrrev_b32 v3, 3, v0 // (op 2881) 

v_lshrrev_b32 v0, 7, v1 // (op 3039) 
v_and_b32 v4, 15, v1 // (op 3039) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3039) 


v_and_b32 v0, 63, v1 // (op 3039) 
v_lshrrev_b32 v4, 4, v0 // (op 3039) 

v_lshlrev_b32 v0, 5, v4 // (op 3039) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 3039) 


v_lshlrev_b32 v0, 2, v4 // (op 3039) 

v_lshrrev_b32 v4, 3, v0 // (op 3039) 

v_lshrrev_b32 v0, 7, v1 // (op 2824) 
v_lshrrev_b32 v5, 6, v1 // (op 2939) 
v_and_b32 v6, 1, v5 // (op 2939) 

v_and_b32 v5, 15, v1 // (op 2939) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2939) 


v_and_b32 v5, 63, v1 // (op 2939) 
v_lshrrev_b32 v6, 4, v5 // (op 2939) 

v_lshlrev_b32 v5, 5, v6 // (op 2939) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 2939) 


v_lshlrev_b32 v5, 2, v6 // (op 2939) 

v_lshrrev_b32 v6, 3, v5 // (op 2939) 

v_lshrrev_b32 v5, 6, v1 // (op 2997) 
v_and_b32 v7, 1, v5 // (op 2997) 

v_and_b32 v5, 15, v1 // (op 2997) 
v_lshl_add_u32 v8, v7, 4, v5 // (op 2997) 


v_and_b32 v5, 63, v1 // (op 2997) 
v_lshrrev_b32 v7, 4, v5 // (op 2997) 

v_lshl_add_u32 v5, v8, 2, v7 // (op 2997) 


v_lshrrev_b32 v7, 7, v1 // (op 3013) 
v_lshrrev_b32 v8, 6, v1 // (op 3068) 
v_and_b32 v9, 1, v8 // (op 3068) 

v_and_b32 v8, 15, v1 // (op 3068) 
v_lshl_add_u32 v10, v9, 4, v8 // (op 3068) 


v_and_b32 v8, 63, v1 // (op 3068) 
v_lshrrev_b32 v9, 4, v8 // (op 3068) 

v_lshlrev_b32 v8, 5, v9 // (op 3068) 

v_lshl_add_u32 v9, v10, 7, v8 // (op 3068) 


v_lshlrev_b32 v8, 2, v9 // (op 3068) 

v_lshrrev_b32 v9, 3, v8 // (op 3068) 

v_lshrrev_b32 v8, 6, v1 // (op 3097) 
v_and_b32 v10, 1, v8 // (op 3097) 

v_and_b32 v8, 15, v1 // (op 3097) 
v_lshl_add_u32 v11, v10, 4, v8 // (op 3097) 


v_and_b32 v8, 63, v1 // (op 3097) 
v_lshrrev_b32 v10, 4, v8 // (op 3097) 

v_lshl_add_u32 v8, v11, 2, v10 // (op 3097) 



s_waitcnt lgkmcnt(0)
s_mul_hi_i32 s0, s16, s47 // (op 2824) 
s_add_i32 s1, s0, s16 // (op 2824) 

s_mul_hi_i32 s0, s16, s47 // (op 3013) 
s_add_i32 s18, s0, s16 // (op 3013) 

v_and_b32 v10, 63, v1 // (op 2824) 
s_ashr_i32 s0, s1, 31 // (op 2824) 
v_lshrrev_b32 v11, 4, v10 // (op 2824) 
s_cmp_eq_i32 s47, 0 // (op 2824) 
s_mov_b32 s19, scc // (op 2824) 
v_lshlrev_b32 v12, 2, v11 // (op 2824) 

s_cmp_eq_i32 s19, 0 // (op 2824) 
s_cselect_b32 s51, 0, -1 // (op 2824) 

v_lshl_add_u32 v11, v0, 4, v12 // (op 2824) 


s_and_b32 s19, s11, 31 // (op 2824) 
v_lshrrev_b32 v0, 2, v10 // (op 2824) 

v_and_b32 v10, 3, v0 // (op 2824) 

v_and_b32 v0, 63, v1 // (op 3013) 
v_lshrrev_b32 v12, 4, v0 // (op 3013) 
s_lshl_b32 s52, 1, s19 // (op 2824) 
v_and_b32 v13, 3, v1 // (op 2824) 
s_add_i32 s53, s52, s51 // (op 2824) 


s_and_b32 s51, s0, s53 // (op 2824) 


s_ashr_i32 s0, s18, 31 // (op 3013) 
s_cmp_eq_i32 s47, 0 // (op 3013) 
s_mov_b32 s52, scc // (op 3013) 
v_lshl_add_u32 v14, v10, 2, v13 // (op 2824) 


s_add_i32 s53, s1, s51 // (op 2824) 


v_lshrrev_b32 v10, 6, v1 // (op 2824) 
s_ashr_i32 s1, s53, s19 // (op 2824) 


v_and_b32 v13, 1, v10 // (op 2824) 

s_lshl_b32 s19, s11, 24 // (op 2824) 
v_lshl_add_u32 v10, v13, 4, v14 // (op 2824) 


s_ashr_i32 s51, s19, 31 // (op 2824) 

s_xor_b32 s19, s1, s51 // (op 2824) 

s_sub_i32 s1, s19, s51 // (op 2824) 


s_mul_hi_i32 s19, s1, s12 // (op 2824) 
s_add_i32 s51, s19, s1 // (op 2824) 

s_ashr_i32 s19, s51, 31 // (op 2824) 
s_cmp_eq_i32 s12, 0 // (op 2824) 
s_mov_b32 s53, scc // (op 2824) 
s_cmp_eq_i32 s53, 0 // (op 2824) 
s_cselect_b32 s54, 0, -1 // (op 2824) 

s_and_b32 s53, s13, 31 // (op 2824) 
v_lshlrev_b32 v13, 2, v12 // (op 3013) 

s_cmp_eq_i32 s52, 0 // (op 3013) 
s_cselect_b32 s55, 0, -1 // (op 3013) 

s_lshl_b32 s52, 1, s53 // (op 2824) 
v_lshl_add_u32 v12, v7, 4, v13 // (op 3013) 


v_lshrrev_b32 v7, 2, v0 // (op 3013) 

s_add_i32 s56, s52, s54 // (op 2824) 


s_and_b32 s52, s19, s56 // (op 2824) 


s_add_i32 s19, s51, s52 // (op 2824) 


s_ashr_i32 s54, s19, s53 // (op 2824) 


s_lshl_b32 s19, s13, 24 // (op 2824) 
s_ashr_i32 s51, s19, 31 // (op 2824) 

s_xor_b32 s19, s54, s51 // (op 2824) 

s_sub_i32 s52, s19, s51 // (op 2824) 


s_mul_i32 s19, s1, s48 // (op 2824) 
s_sub_u32 s51, s16, s19 // (op 2824) 

s_mul_i32 s19, s52, s48 // (op 2824) 
s_add_u32 s19, s19, s51 // (op 2824) 

s_mul_hi_i32 s51, s16, s10 // (op 2824) 
s_add_i32 s53, s51, s16 // (op 2824) 

s_ashr_i32 s51, s53, 31 // (op 2824) 
s_cmp_eq_i32 s10, 0 // (op 2824) 
s_mov_b32 s54, scc // (op 2824) 
s_cmp_eq_i32 s54, 0 // (op 2824) 
s_cselect_b32 s56, 0, -1 // (op 2824) 

s_and_b32 s54, s44, 31 // (op 2824) 
s_lshl_b32 s57, 1, s54 // (op 2824) 
s_add_i32 s58, s57, s56 // (op 2824) 


s_and_b32 s56, s51, s58 // (op 2824) 


s_add_i32 s51, s53, s56 // (op 2824) 


s_ashr_i32 s53, s51, s54 // (op 2824) 


s_lshl_b32 s51, s44, 24 // (op 2824) 
s_ashr_i32 s54, s51, 31 // (op 2824) 

s_xor_b32 s51, s53, s54 // (op 2824) 

s_sub_i32 s53, s51, s54 // (op 2824) 


s_mul_i32 s51, s48, s6 // (op 2824) 
s_mul_i32 s54, s14, s51 // (op 2824) 

s_mul_i32 s51, s53, s54 // (op 2824) 

s_sub_u32 s54, s16, s51 // (op 2824) 

s_mov_b32 s51, s54 // (op 2824) 
s_mul_hi_i32 s56, s51, s45 // (op 2824) 
s_add_i32 s57, s56, s51 // (op 2824) 


s_ashr_i32 s51, s57, 31 // (op 2824) 
s_cmp_eq_i32 s45, 0 // (op 2824) 
s_mov_b32 s56, scc // (op 2824) 
s_cmp_eq_i32 s56, 0 // (op 2824) 
s_cselect_b32 s58, 0, -1 // (op 2824) 

s_and_b32 s56, s15, 31 // (op 2824) 
s_lshl_b32 s59, 1, s56 // (op 2824) 
s_add_i32 s60, s59, s58 // (op 2824) 


s_and_b32 s58, s51, s60 // (op 2824) 


s_add_i32 s51, s57, s58 // (op 2824) 


s_ashr_i32 s57, s51, s56 // (op 2824) 


s_lshl_b32 s51, s15, 24 // (op 2824) 
s_ashr_i32 s56, s51, 31 // (op 2824) 

s_xor_b32 s51, s57, s56 // (op 2824) 

s_sub_i32 s57, s51, s56 // (op 2824) 


s_mul_i32 s51, s46, s48 // (op 2824) 
s_and_b32 s56, s11, 31 // (op 3013) 
s_lshl_b32 s58, 1, s56 // (op 3013) 
v_and_b32 v0, 3, v7 // (op 3013) 

s_add_i32 s59, s58, s55 // (op 3013) 


v_and_b32 v7, 3, v1 // (op 3013) 
s_and_b32 s55, s0, s59 // (op 3013) 


v_lshl_add_u32 v13, v0, 2, v7 // (op 3013) 


s_add_i32 s0, s18, s55 // (op 3013) 


v_lshrrev_b32 v0, 6, v1 // (op 3013) 
s_ashr_i32 s18, s0, s56 // (op 3013) 


v_and_b32 v7, 1, v0 // (op 3013) 

s_lshl_b32 s0, s11, 24 // (op 3013) 
v_lshl_add_u32 v0, v7, 4, v13 // (op 3013) 


s_ashr_i32 s55, s0, 31 // (op 3013) 

s_xor_b32 s0, s18, s55 // (op 3013) 

s_sub_i32 s18, s0, s55 // (op 3013) 


s_mul_hi_i32 s0, s18, s12 // (op 3013) 
s_add_i32 s55, s0, s18 // (op 3013) 

s_ashr_i32 s0, s55, 31 // (op 3013) 
s_cmp_eq_i32 s12, 0 // (op 3013) 
s_mov_b32 s56, scc // (op 3013) 
s_cmp_eq_i32 s56, 0 // (op 3013) 
s_cselect_b32 s58, 0, -1 // (op 3013) 

s_and_b32 s56, s13, 31 // (op 3013) 
s_lshl_b32 s59, 1, s56 // (op 3013) 
s_add_i32 s60, s59, s58 // (op 3013) 


s_and_b32 s58, s0, s60 // (op 3013) 


s_add_i32 s0, s55, s58 // (op 3013) 


s_ashr_i32 s55, s0, s56 // (op 3013) 


s_lshl_b32 s0, s13, 24 // (op 3013) 
s_ashr_i32 s56, s0, 31 // (op 3013) 

s_xor_b32 s0, s55, s56 // (op 3013) 

s_sub_i32 s55, s0, s56 // (op 3013) 


s_mul_i32 s0, s18, s48 // (op 3013) 
s_sub_u32 s56, s16, s0 // (op 3013) 

s_mul_i32 s0, s55, s48 // (op 3013) 
s_add_u32 s0, s0, s56 // (op 3013) 

s_mul_hi_i32 s56, s16, s10 // (op 3013) 
s_add_i32 s58, s56, s16 // (op 3013) 

s_ashr_i32 s56, s58, 31 // (op 3013) 
s_cmp_eq_i32 s10, 0 // (op 3013) 
s_mov_b32 s59, scc // (op 3013) 
s_cmp_eq_i32 s59, 0 // (op 3013) 
s_cselect_b32 s60, 0, -1 // (op 3013) 

s_and_b32 s59, s44, 31 // (op 3013) 
s_lshl_b32 s61, 1, s59 // (op 3013) 
s_add_i32 s62, s61, s60 // (op 3013) 


s_and_b32 s60, s56, s62 // (op 3013) 


s_add_i32 s56, s58, s60 // (op 3013) 


s_ashr_i32 s58, s56, s59 // (op 3013) 


s_lshl_b32 s56, s44, 24 // (op 3013) 
s_ashr_i32 s59, s56, 31 // (op 3013) 

s_xor_b32 s56, s58, s59 // (op 3013) 

s_sub_i32 s58, s56, s59 // (op 3013) 


s_mul_i32 s56, s48, s6 // (op 3013) 
s_mul_i32 s59, s14, s56 // (op 3013) 

s_mul_i32 s56, s58, s59 // (op 3013) 

s_sub_u32 s59, s16, s56 // (op 3013) 

s_mov_b32 s56, s59 // (op 3013) 
s_mul_hi_i32 s60, s56, s45 // (op 3013) 
s_add_i32 s61, s60, s56 // (op 3013) 


s_ashr_i32 s56, s61, 31 // (op 3013) 
s_cmp_eq_i32 s45, 0 // (op 3013) 
s_mov_b32 s60, scc // (op 3013) 
s_cmp_eq_i32 s60, 0 // (op 3013) 
s_cselect_b32 s62, 0, -1 // (op 3013) 

s_and_b32 s60, s15, 31 // (op 3013) 
s_lshl_b32 s63, 1, s60 // (op 3013) 
s_add_i32 s64, s63, s62 // (op 3013) 


s_and_b32 s62, s56, s64 // (op 3013) 


s_add_i32 s56, s61, s62 // (op 3013) 


s_ashr_i32 s61, s56, s60 // (op 3013) 


s_lshl_b32 s56, s15, 24 // (op 3013) 
s_ashr_i32 s60, s56, 31 // (op 3013) 

s_xor_b32 s56, s61, s60 // (op 3013) 

s_sub_i32 s61, s56, s60 // (op 3013) 


s_mul_i32 s56, s46, s48 // (op 3013) 
s_add_u32 s62, s8, 32 // (op 2824) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 2824) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 2824) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 2824) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 2824) 

s_sub_i32 s60, s62, s51 // (op 2824) 

s_mul_i32 s62, s57, s60 // (op 2824) 

s_sub_u32 s60, s54, s62 // (op 2824) 


s_add_u32 s54, s60, s51 // (op 2824) 


s_cmp_eq_i32 s53, 0 // (op 2824) 
s_mov_b32 s51, scc // (op 2824) 

s_cmp_eq_i32 s51, 0 // (op 2824) 
s_cselect_b32 s53, s54, s19 // (op 2824) 


s_mul_i32 s19, s52, s6 // (op 2824) 

v_lshl_add_u32 v7, s53, 5, v11 // (op 2824) 


s_sub_i32 s60, s1, s19 // (op 2824) 


s_cmp_eq_i32 s51, 0 // (op 2824) 
s_cselect_b32 s1, s57, s60 // (op 2824) 



v_mov_b32 v11, 192 // (op 2824) 
v_mul_lo_u32 v13, s1, v11 // (op 2824) 

v_add_u32 v13, v13, v10 // (op 2824) 


v_mul_lo_u32 v10, v13, s2 // (op 2824) 

v_add_lshl_u32 v11, v7, v10, 1 // (op 2824) 


s_add_u32 s52, s8, 32 // (op 3013) Start of Int64 add, locking SCC
s_addc_u32 s53, s9, 0 // (op 3013) End of Int64 add, unlocking SCC
s_sub_u32 s62, s52, 1 // (op 3013) least significant half; sets scc
s_subb_u32 s63, s53, 0 // (op 3013) most significant half; uses scc

s_ashr_i64 s[52:53], s[62:63], 5 // (op 3013) 

s_sub_i32 s1, s52, s56 // (op 3013) 

s_mul_i32 s19, s61, s1 // (op 3013) 

s_sub_u32 s1, s59, s19 // (op 3013) 


s_add_u32 s19, s1, s56 // (op 3013) 


s_cmp_eq_i32 s58, 0 // (op 3013) 
s_mov_b32 s1, scc // (op 3013) 

s_cmp_eq_i32 s1, 0 // (op 3013) 
s_cselect_b32 s51, s19, s0 // (op 3013) 


s_mul_i32 s0, s55, s6 // (op 3013) 

v_lshl_add_u32 v7, s51, 5, v12 // (op 3013) 


s_sub_i32 s19, s18, s0 // (op 3013) 


s_cmp_eq_i32 s1, 0 // (op 3013) 
s_cselect_b32 s0, s61, s19 // (op 3013) 



v_mov_b32 v10, 192 // (op 3013) 
v_mul_lo_u32 v12, s0, v10 // (op 3013) 

v_add_u32 v12, v12, v0 // (op 3013) 


v_mul_lo_u32 v0, v12, s4 // (op 3013) 

v_add_lshl_u32 v10, v7, v0, 1 // (op 3013) 


s_mul_i32 s1, s2, s23 // (op 2826) most significant: low of low * high
s_mul_i32 s18, s3, s22 // (op 2826) most significant: low of high * low
s_mul_i32 s0, s2, s22 // (op 2826) least significant
s_mul_hi_u32 s19, s2, s22 // (op 2826) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2826) most significant: sum
s_add_u32 s1, s1, s19 // (op 2826) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2826) 

s_mul_i32 s1, s4, s23 // (op 3015) most significant: low of low * high
s_mul_i32 s51, s5, s22 // (op 3015) most significant: low of high * low
s_mul_i32 s0, s4, s22 // (op 3015) least significant
s_mul_hi_u32 s52, s4, s22 // (op 3015) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 3015) most significant: sum
s_add_u32 s1, s1, s52 // (op 3015) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 3015) 

s_mov_b32 s56, s20 // (op 2826) reg expression
s_mov_b32 s57, s21 // (op 2826) reg expression
s_mov_b32 s58, s18 // (op 2826) reg expression
s_mov_b32 s59, 131072 // (op 2826) call()

s_mov_b32 s60, s24 // (op 3015) reg expression

s_mov_b32 s61, s25 // (op 3015) reg expression
s_mov_b32 s62, s52 // (op 3015) reg expression
s_mov_b32 s63, 131072 // (op 3015) call()


v_mov_b32 v0, s49 // (op 2) Move value
v_mov_b32 v7, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 310) call()
v_accvgpr_write a5, 0.00000 // (op 310) call()
v_accvgpr_write a6, 0.00000 // (op 310) call()
v_accvgpr_write a7, 0.00000 // (op 310) call()
v_accvgpr_write a8, 0.00000 // (op 466) call()
v_accvgpr_write a9, 0.00000 // (op 466) call()
v_accvgpr_write a10, 0.00000 // (op 466) call()
v_accvgpr_write a11, 0.00000 // (op 466) call()
v_accvgpr_write a12, 0.00000 // (op 622) call()
v_accvgpr_write a13, 0.00000 // (op 622) call()
v_accvgpr_write a14, 0.00000 // (op 622) call()
v_accvgpr_write a15, 0.00000 // (op 622) call()
v_accvgpr_write a16, 0.00000 // (op 778) call()
v_accvgpr_write a17, 0.00000 // (op 778) call()
v_accvgpr_write a18, 0.00000 // (op 778) call()
v_accvgpr_write a19, 0.00000 // (op 778) call()
v_accvgpr_write a20, 0.00000 // (op 934) call()
v_accvgpr_write a21, 0.00000 // (op 934) call()
v_accvgpr_write a22, 0.00000 // (op 934) call()
v_accvgpr_write a23, 0.00000 // (op 934) call()
s_mul_hi_i32 s0, s16, s47 // (op 2591) 
v_lshrrev_b32 v12, 3, v1 // (op 2591) 
s_add_i32 s1, s0, s16 // (op 2591) 

v_and_b32 v13, -8, v1 // (op 2591) 
s_ashr_i32 s0, s1, 31 // (op 2591) 
v_sub_u32 v14, v1, v13 // (op 2591) 

s_cmp_eq_i32 s47, 0 // (op 2591) 
s_mov_b32 s18, scc // (op 2591) 
v_lshlrev_b32 v13, 4, v14 // (op 2591) 

s_cmp_eq_i32 s18, 0 // (op 2591) 
s_cselect_b32 s19, 0, -1 // (op 2591) 

v_mov_b32 v15, 0 // (op 2591) convert
v_mov_b32 v14, v13 // (op 2591) convert

s_and_b32 s18, s11, 31 // (op 2591) 
s_lshl_b32 s20, 1, s18 // (op 2591) 
s_add_i32 s21, s20, s19 // (op 2591) 


s_and_b32 s19, s0, s21 // (op 2591) 


s_add_i32 s0, s1, s19 // (op 2591) 


s_ashr_i32 s1, s0, s18 // (op 2591) 


s_lshl_b32 s0, s11, 24 // (op 2591) 
s_ashr_i32 s18, s0, 31 // (op 2591) 

s_xor_b32 s0, s1, s18 // (op 2591) 

s_sub_i32 s1, s0, s18 // (op 2591) 


s_mul_hi_i32 s0, s1, s12 // (op 2591) 
s_add_i32 s18, s0, s1 // (op 2591) 

s_ashr_i32 s0, s18, 31 // (op 2591) 
s_cmp_eq_i32 s12, 0 // (op 2591) 
s_mov_b32 s19, scc // (op 2591) 
s_cmp_eq_i32 s19, 0 // (op 2591) 
s_cselect_b32 s20, 0, -1 // (op 2591) 

s_and_b32 s19, s13, 31 // (op 2591) 
s_lshl_b32 s21, 1, s19 // (op 2591) 
s_add_i32 s24, s21, s20 // (op 2591) 


s_and_b32 s25, s0, s24 // (op 2591) 


s_add_i32 s0, s18, s25 // (op 2591) 


s_ashr_i32 s18, s0, s19 // (op 2591) 


s_lshl_b32 s0, s13, 24 // (op 2591) 
s_ashr_i32 s19, s0, 31 // (op 2591) 

s_xor_b32 s0, s18, s19 // (op 2591) 

s_sub_i32 s18, s0, s19 // (op 2591) 


s_mul_i32 s0, s1, s48 // (op 2591) 

s_sub_u32 s1, s16, s0 // (op 2591) 

s_mul_i32 s0, s18, s48 // (op 2591) 
s_add_u32 s0, s0, s1 // (op 2591) 


s_mul_hi_i32 s1, s16, s10 // (op 2591) 
s_add_i32 s18, s1, s16 // (op 2591) 

s_ashr_i32 s1, s18, 31 // (op 2591) 
s_cmp_eq_i32 s10, 0 // (op 2591) 
s_mov_b32 s19, scc // (op 2591) 
s_cmp_eq_i32 s19, 0 // (op 2591) 
s_cselect_b32 s20, 0, -1 // (op 2591) 

s_and_b32 s19, s44, 31 // (op 2591) 
s_lshl_b32 s21, 1, s19 // (op 2591) 
s_add_i32 s24, s21, s20 // (op 2591) 


s_and_b32 s25, s1, s24 // (op 2591) 


s_add_i32 s1, s18, s25 // (op 2591) 


s_ashr_i32 s18, s1, s19 // (op 2591) 


s_lshl_b32 s1, s44, 24 // (op 2591) 
s_ashr_i32 s19, s1, 31 // (op 2591) 

s_xor_b32 s1, s18, s19 // (op 2591) 

s_sub_i32 s18, s1, s19 // (op 2591) 


s_mul_i32 s1, s48, s6 // (op 2591) 
s_mul_i32 s19, s14, s1 // (op 2591) 

s_mul_i32 s1, s18, s19 // (op 2591) 

s_sub_u32 s19, s16, s1 // (op 2591) 

s_mov_b32 s1, s19 // (op 2591) 
s_mul_hi_i32 s20, s1, s45 // (op 2591) 
s_add_i32 s21, s20, s1 // (op 2591) 


s_ashr_i32 s1, s21, 31 // (op 2591) 
s_cmp_eq_i32 s45, 0 // (op 2591) 
s_mov_b32 s20, scc // (op 2591) 
s_cmp_eq_i32 s20, 0 // (op 2591) 
s_cselect_b32 s24, 0, -1 // (op 2591) 

s_and_b32 s20, s15, 31 // (op 2591) 
s_lshl_b32 s25, 1, s20 // (op 2591) 
s_add_i32 s49, s25, s24 // (op 2591) 


s_and_b32 s24, s1, s49 // (op 2591) 


s_add_i32 s1, s21, s24 // (op 2591) 


s_ashr_i32 s21, s1, s20 // (op 2591) 


s_lshl_b32 s1, s15, 24 // (op 2591) 
s_ashr_i32 s20, s1, 31 // (op 2591) 

s_xor_b32 s1, s21, s20 // (op 2591) 

s_sub_i32 s21, s1, s20 // (op 2591) 


s_mul_i32 s1, s46, s48 // (op 2591) 
s_add_u32 s24, s8, 32 // (op 2591) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 2591) End of Int64 add, unlocking SCC
s_sub_u32 s54, s24, 1 // (op 2591) least significant half; sets scc
s_subb_u32 s55, s25, 0 // (op 2591) most significant half; uses scc

s_ashr_i64 s[24:25], s[54:55], 5 // (op 2591) 

s_sub_i32 s20, s24, s1 // (op 2591) 

s_mul_i32 s24, s21, s20 // (op 2591) 


s_sub_u32 s25, s19, s24 // (op 2591) 


s_add_u32 s24, s25, s1 // (op 2591) 


s_cmp_eq_i32 s18, 0 // (op 2591) 
s_mov_b32 s1, scc // (op 2591) 

s_cmp_eq_i32 s1, 0 // (op 2591) 
s_cselect_b32 s25, s24, s0 // (op 2591) 



v_lshl_add_u32 v13, s25, 5, v12 // (op 2591) 


v_mul_lo_u32 v17, v13, s27 // (op 2591) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 2591) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 2591) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 2591) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 2591) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 2591) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 2591) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 2591) 

v_mov_b32 v12, v14 // (op 2591) convert

s_mul_i32 s1, s26, s9 // (op 2593) most significant: low of low * high
s_mul_i32 s18, s27, s8 // (op 2593) most significant: low of high * low
s_mul_i32 s0, s26, s8 // (op 2593) least significant
s_mul_hi_u32 s19, s26, s8 // (op 2593) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2593) most significant: sum
s_add_u32 s1, s1, s19 // (op 2593) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 2593) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 2593) 

s_mov_b32 s52, s28 // (op 2593) reg expression
s_mov_b32 s53, s29 // (op 2593) reg expression
s_mov_b32 s54, s0 // (op 2593) reg expression
s_mov_b32 s55, 131072 // (op 2593) call()

s_mul_hi_i32 s0, s16, s47 // (op 2610) 
v_lshrrev_b32 v13, 3, v1 // (op 2610) 
s_add_i32 s1, s0, s16 // (op 2610) 

v_and_b32 v14, -8, v1 // (op 2610) 
s_ashr_i32 s0, s1, 31 // (op 2610) 
v_sub_u32 v15, v1, v14 // (op 2610) 

s_cmp_eq_i32 s47, 0 // (op 2610) 
s_mov_b32 s18, scc // (op 2610) 
v_lshlrev_b32 v14, 4, v15 // (op 2610) 

s_cmp_eq_i32 s18, 0 // (op 2610) 
s_cselect_b32 s19, 0, -1 // (op 2610) 

v_mov_b32 v17, 0 // (op 2610) convert
v_mov_b32 v16, v14 // (op 2610) convert

s_and_b32 s18, s11, 31 // (op 2610) 
s_lshl_b32 s20, 1, s18 // (op 2610) 
s_add_i32 s21, s20, s19 // (op 2610) 


s_and_b32 s19, s0, s21 // (op 2610) 


s_add_i32 s0, s1, s19 // (op 2610) 


s_ashr_i32 s1, s0, s18 // (op 2610) 


s_lshl_b32 s0, s11, 24 // (op 2610) 
s_ashr_i32 s18, s0, 31 // (op 2610) 

s_xor_b32 s0, s1, s18 // (op 2610) 

s_sub_i32 s1, s0, s18 // (op 2610) 


s_mul_hi_i32 s0, s1, s12 // (op 2610) 
s_add_i32 s18, s0, s1 // (op 2610) 

s_ashr_i32 s0, s18, 31 // (op 2610) 
s_cmp_eq_i32 s12, 0 // (op 2610) 
s_mov_b32 s19, scc // (op 2610) 
s_cmp_eq_i32 s19, 0 // (op 2610) 
s_cselect_b32 s20, 0, -1 // (op 2610) 

s_and_b32 s19, s13, 31 // (op 2610) 
s_lshl_b32 s21, 1, s19 // (op 2610) 
s_add_i32 s24, s21, s20 // (op 2610) 


s_and_b32 s25, s0, s24 // (op 2610) 


s_add_i32 s0, s18, s25 // (op 2610) 


s_ashr_i32 s18, s0, s19 // (op 2610) 


s_lshl_b32 s0, s13, 24 // (op 2610) 
s_ashr_i32 s19, s0, 31 // (op 2610) 

s_xor_b32 s0, s18, s19 // (op 2610) 

s_sub_i32 s18, s0, s19 // (op 2610) 


s_mul_i32 s0, s1, s48 // (op 2610) 

s_sub_u32 s1, s16, s0 // (op 2610) 

s_mul_i32 s0, s18, s48 // (op 2610) 
s_add_u32 s0, s0, s1 // (op 2610) 


s_mul_hi_i32 s1, s16, s10 // (op 2610) 
s_add_i32 s18, s1, s16 // (op 2610) 

s_ashr_i32 s1, s18, 31 // (op 2610) 
s_cmp_eq_i32 s10, 0 // (op 2610) 
s_mov_b32 s19, scc // (op 2610) 
s_cmp_eq_i32 s19, 0 // (op 2610) 
s_cselect_b32 s20, 0, -1 // (op 2610) 

s_and_b32 s19, s44, 31 // (op 2610) 
s_lshl_b32 s21, 1, s19 // (op 2610) 
s_add_i32 s24, s21, s20 // (op 2610) 


s_and_b32 s25, s1, s24 // (op 2610) 


s_add_i32 s1, s18, s25 // (op 2610) 


s_ashr_i32 s18, s1, s19 // (op 2610) 


s_lshl_b32 s1, s44, 24 // (op 2610) 
s_ashr_i32 s19, s1, 31 // (op 2610) 

s_xor_b32 s1, s18, s19 // (op 2610) 

s_sub_i32 s18, s1, s19 // (op 2610) 


s_mul_i32 s1, s48, s6 // (op 2610) 
s_mul_i32 s19, s14, s1 // (op 2610) 

s_mul_i32 s1, s18, s19 // (op 2610) 

s_sub_u32 s19, s16, s1 // (op 2610) 

s_mov_b32 s1, s19 // (op 2610) 
s_mul_hi_i32 s20, s1, s45 // (op 2610) 
s_add_i32 s21, s20, s1 // (op 2610) 


s_ashr_i32 s1, s21, 31 // (op 2610) 
s_cmp_eq_i32 s45, 0 // (op 2610) 
s_mov_b32 s20, scc // (op 2610) 
s_cmp_eq_i32 s20, 0 // (op 2610) 
s_cselect_b32 s24, 0, -1 // (op 2610) 

s_and_b32 s20, s15, 31 // (op 2610) 
s_lshl_b32 s25, 1, s20 // (op 2610) 
s_add_i32 s49, s25, s24 // (op 2610) 


s_and_b32 s24, s1, s49 // (op 2610) 


s_add_i32 s1, s21, s24 // (op 2610) 


s_ashr_i32 s21, s1, s20 // (op 2610) 


s_lshl_b32 s1, s15, 24 // (op 2610) 
s_ashr_i32 s20, s1, 31 // (op 2610) 

s_xor_b32 s1, s21, s20 // (op 2610) 

s_sub_i32 s21, s1, s20 // (op 2610) 


s_mul_i32 s1, s46, s48 // (op 2610) 
s_add_u32 s24, s8, 32 // (op 2610) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 2610) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2610) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2610) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 2610) 

s_sub_i32 s20, s24, s1 // (op 2610) 

s_mul_i32 s24, s21, s20 // (op 2610) 


s_sub_u32 s25, s19, s24 // (op 2610) 


s_add_u32 s24, s25, s1 // (op 2610) 


s_cmp_eq_i32 s18, 0 // (op 2610) 
s_mov_b32 s1, scc // (op 2610) 

s_cmp_eq_i32 s1, 0 // (op 2610) 
s_cselect_b32 s25, s24, s0 // (op 2610) 



v_lshl_add_u32 v14, s25, 5, v13 // (op 2610) 


v_mul_lo_u32 v19, v14, s27 // (op 2610) most significant: low of low * high
v_mul_hi_u32 v13, v14, s26 // (op 2610) most significant: high of low * low
v_mul_lo_u32 v18, v14, s26 // (op 2610) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 2610) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 2610) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 2610) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 2610) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 2610) 

v_mov_b32 v13, v14 // (op 2610) convert

s_mul_i32 s1, s26, s9 // (op 2612) most significant: low of low * high
s_mul_i32 s18, s27, s8 // (op 2612) most significant: low of high * low
s_mul_i32 s0, s26, s8 // (op 2612) least significant
s_mul_hi_u32 s19, s26, s8 // (op 2612) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2612) most significant: sum
s_add_u32 s1, s1, s19 // (op 2612) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 2612) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 2612) 

s_mov_b32 s52, s28 // (op 2612) reg expression
s_mov_b32 s53, s29 // (op 2612) reg expression
s_mov_b32 s54, s0 // (op 2612) reg expression
s_mov_b32 s55, 131072 // (op 2612) call()

s_mul_hi_i32 s0, s16, s47 // (op 2655) 
v_lshrrev_b32 v14, 2, v1 // (op 2655) 
s_add_i32 s1, s0, s16 // (op 2655) 

v_mul_lo_u32 v15, v14, 3 // (op 2655) 

s_ashr_i32 s0, s1, 31 // (op 2655) 
v_and_b32 v14, -4, v1 // (op 2655) 
s_cmp_eq_i32 s47, 0 // (op 2655) 
s_mov_b32 s18, scc // (op 2655) 
v_sub_u32 v16, v1, v14 // (op 2655) 

s_cmp_eq_i32 s18, 0 // (op 2655) 
s_cselect_b32 s19, 0, -1 // (op 2655) 

v_lshlrev_b32 v14, 5, v16 // (op 2655) 

v_mov_b32 v17, 0 // (op 2655) convert
v_mov_b32 v16, v14 // (op 2655) convert

s_and_b32 s18, s11, 31 // (op 2655) 
s_lshl_b32 s20, 1, s18 // (op 2655) 
s_add_i32 s21, s20, s19 // (op 2655) 


s_and_b32 s19, s0, s21 // (op 2655) 


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
s_cselect_b32 s20, 0, -1 // (op 2655) 

s_and_b32 s19, s13, 31 // (op 2655) 
s_lshl_b32 s21, 1, s19 // (op 2655) 
s_add_i32 s24, s21, s20 // (op 2655) 


s_and_b32 s25, s0, s24 // (op 2655) 


s_add_i32 s0, s18, s25 // (op 2655) 


s_ashr_i32 s18, s0, s19 // (op 2655) 


s_lshl_b32 s0, s13, 24 // (op 2655) 
s_ashr_i32 s19, s0, 31 // (op 2655) 

s_xor_b32 s0, s18, s19 // (op 2655) 

s_sub_i32 s18, s0, s19 // (op 2655) 


s_mul_i32 s0, s18, s6 // (op 2655) 

s_sub_i32 s18, s1, s0 // (op 2655) 


s_mul_hi_i32 s0, s16, s10 // (op 2655) 
s_add_i32 s1, s0, s16 // (op 2655) 

s_ashr_i32 s0, s1, 31 // (op 2655) 
s_cmp_eq_i32 s10, 0 // (op 2655) 
s_mov_b32 s19, scc // (op 2655) 
s_cmp_eq_i32 s19, 0 // (op 2655) 
s_cselect_b32 s20, 0, -1 // (op 2655) 

s_and_b32 s19, s44, 31 // (op 2655) 
s_lshl_b32 s21, 1, s19 // (op 2655) 
s_add_i32 s24, s21, s20 // (op 2655) 


s_and_b32 s25, s0, s24 // (op 2655) 


s_add_i32 s0, s1, s25 // (op 2655) 


s_ashr_i32 s1, s0, s19 // (op 2655) 


s_lshl_b32 s0, s44, 24 // (op 2655) 
s_ashr_i32 s19, s0, 31 // (op 2655) 

s_xor_b32 s0, s1, s19 // (op 2655) 

s_sub_i32 s1, s0, s19 // (op 2655) 


s_mul_i32 s0, s48, s6 // (op 2655) 
s_mul_i32 s19, s14, s0 // (op 2655) 

s_mul_i32 s0, s1, s19 // (op 2655) 

s_sub_u32 s19, s16, s0 // (op 2655) 

s_mul_hi_i32 s0, s19, s45 // (op 2655) 
s_add_i32 s20, s0, s19 // (op 2655) 


s_ashr_i32 s0, s20, 31 // (op 2655) 
s_cmp_eq_i32 s45, 0 // (op 2655) 
s_mov_b32 s19, scc // (op 2655) 
s_cmp_eq_i32 s19, 0 // (op 2655) 
s_cselect_b32 s21, 0, -1 // (op 2655) 

s_and_b32 s19, s15, 31 // (op 2655) 
s_lshl_b32 s24, 1, s19 // (op 2655) 
s_add_i32 s25, s24, s21 // (op 2655) 


s_and_b32 s21, s0, s25 // (op 2655) 


s_add_i32 s0, s20, s21 // (op 2655) 


s_ashr_i32 s20, s0, s19 // (op 2655) 


s_lshl_b32 s0, s15, 24 // (op 2655) 
s_ashr_i32 s19, s0, 31 // (op 2655) 

s_xor_b32 s0, s20, s19 // (op 2655) 

s_sub_i32 s20, s0, s19 // (op 2655) 


s_cmp_eq_i32 s1, 0 // (op 2655) 
s_mov_b32 s0, scc // (op 2655) 

s_cmp_eq_i32 s0, 0 // (op 2655) 
s_cselect_b32 s1, s20, s18 // (op 2655) 



v_mov_b32 v14, 192 // (op 2655) 
v_mul_lo_u32 v18, s1, v14 // (op 2655) 

v_add_u32 v18, v18, v15 // (op 2655) 


v_mul_lo_u32 v15, v18, s31 // (op 2655) most significant: low of low * high
v_mul_hi_u32 v19, v18, s30 // (op 2655) most significant: high of low * low
v_mul_lo_u32 v14, v18, s30 // (op 2655) least significant: low of low * low
v_add_u32 v15, v15, v19 // (op 2655) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v14 // (op 2655) least significant half
v_addc_co_u32 v19, s[0:1], v17, v15, s[0:1] // (op 2655) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 2655) 


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 2655) 

v_mov_b32 v16, v14 // (op 2655) convert

s_mul_i32 s1, s30, s23 // (op 2657) most significant: low of low * high
s_mul_i32 s18, s31, s22 // (op 2657) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 2657) least significant
s_mul_hi_u32 s19, s30, s22 // (op 2657) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2657) most significant: sum
s_add_u32 s1, s1, s19 // (op 2657) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 2657) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 2657) 

s_mov_b32 s64, s32 // (op 2657) reg expression
s_mov_b32 s65, s33 // (op 2657) reg expression
s_mov_b32 s66, s0 // (op 2657) reg expression
s_mov_b32 s67, 131072 // (op 2657) call()

s_mul_hi_i32 s0, s16, s47 // (op 2674) 
v_lshrrev_b32 v14, 2, v1 // (op 2674) 
s_add_i32 s1, s0, s16 // (op 2674) 

v_mul_lo_u32 v15, v14, 3 // (op 2674) 

s_ashr_i32 s0, s1, 31 // (op 2674) 
v_and_b32 v14, -4, v1 // (op 2674) 
s_cmp_eq_i32 s47, 0 // (op 2674) 
s_mov_b32 s18, scc // (op 2674) 
v_sub_u32 v17, v1, v14 // (op 2674) 

s_cmp_eq_i32 s18, 0 // (op 2674) 
s_cselect_b32 s19, 0, -1 // (op 2674) 

v_lshlrev_b32 v14, 5, v17 // (op 2674) 

v_mov_b32 v19, 0 // (op 2674) convert
v_mov_b32 v18, v14 // (op 2674) convert

s_and_b32 s18, s11, 31 // (op 2674) 
s_lshl_b32 s20, 1, s18 // (op 2674) 
s_add_i32 s21, s20, s19 // (op 2674) 


s_and_b32 s19, s0, s21 // (op 2674) 


s_add_i32 s0, s1, s19 // (op 2674) 


s_ashr_i32 s1, s0, s18 // (op 2674) 


s_lshl_b32 s0, s11, 24 // (op 2674) 
s_ashr_i32 s18, s0, 31 // (op 2674) 

s_xor_b32 s0, s1, s18 // (op 2674) 

s_sub_i32 s1, s0, s18 // (op 2674) 


s_mul_hi_i32 s0, s1, s12 // (op 2674) 
s_add_i32 s18, s0, s1 // (op 2674) 

s_ashr_i32 s0, s18, 31 // (op 2674) 
s_cmp_eq_i32 s12, 0 // (op 2674) 
s_mov_b32 s19, scc // (op 2674) 
s_cmp_eq_i32 s19, 0 // (op 2674) 
s_cselect_b32 s20, 0, -1 // (op 2674) 

s_and_b32 s19, s13, 31 // (op 2674) 
s_lshl_b32 s21, 1, s19 // (op 2674) 
s_add_i32 s24, s21, s20 // (op 2674) 


s_and_b32 s25, s0, s24 // (op 2674) 


s_add_i32 s0, s18, s25 // (op 2674) 


s_ashr_i32 s18, s0, s19 // (op 2674) 


s_lshl_b32 s0, s13, 24 // (op 2674) 
s_ashr_i32 s19, s0, 31 // (op 2674) 

s_xor_b32 s0, s18, s19 // (op 2674) 

s_sub_i32 s18, s0, s19 // (op 2674) 


s_mul_i32 s0, s18, s6 // (op 2674) 

s_sub_i32 s18, s1, s0 // (op 2674) 


s_mul_hi_i32 s0, s16, s10 // (op 2674) 
s_add_i32 s1, s0, s16 // (op 2674) 

s_ashr_i32 s0, s1, 31 // (op 2674) 
s_cmp_eq_i32 s10, 0 // (op 2674) 
s_mov_b32 s19, scc // (op 2674) 
s_cmp_eq_i32 s19, 0 // (op 2674) 
s_cselect_b32 s20, 0, -1 // (op 2674) 

s_and_b32 s19, s44, 31 // (op 2674) 
s_lshl_b32 s21, 1, s19 // (op 2674) 
s_add_i32 s24, s21, s20 // (op 2674) 


s_and_b32 s25, s0, s24 // (op 2674) 


s_add_i32 s0, s1, s25 // (op 2674) 


s_ashr_i32 s1, s0, s19 // (op 2674) 


s_lshl_b32 s0, s44, 24 // (op 2674) 
s_ashr_i32 s19, s0, 31 // (op 2674) 

s_xor_b32 s0, s1, s19 // (op 2674) 

s_sub_i32 s1, s0, s19 // (op 2674) 


s_mul_i32 s0, s48, s6 // (op 2674) 
s_mul_i32 s19, s14, s0 // (op 2674) 

s_mul_i32 s0, s1, s19 // (op 2674) 

s_sub_u32 s19, s16, s0 // (op 2674) 

s_mul_hi_i32 s0, s19, s45 // (op 2674) 
s_add_i32 s20, s0, s19 // (op 2674) 


s_ashr_i32 s0, s20, 31 // (op 2674) 
s_cmp_eq_i32 s45, 0 // (op 2674) 
s_mov_b32 s19, scc // (op 2674) 
s_cmp_eq_i32 s19, 0 // (op 2674) 
s_cselect_b32 s21, 0, -1 // (op 2674) 

s_and_b32 s19, s15, 31 // (op 2674) 
s_lshl_b32 s24, 1, s19 // (op 2674) 
s_add_i32 s25, s24, s21 // (op 2674) 


s_and_b32 s21, s0, s25 // (op 2674) 


s_add_i32 s0, s20, s21 // (op 2674) 


s_ashr_i32 s20, s0, s19 // (op 2674) 


s_lshl_b32 s0, s15, 24 // (op 2674) 
s_ashr_i32 s19, s0, 31 // (op 2674) 

s_xor_b32 s0, s20, s19 // (op 2674) 

s_sub_i32 s20, s0, s19 // (op 2674) 


s_cmp_eq_i32 s1, 0 // (op 2674) 
s_mov_b32 s0, scc // (op 2674) 

s_cmp_eq_i32 s0, 0 // (op 2674) 
s_cselect_b32 s1, s20, s18 // (op 2674) 



v_mov_b32 v14, 192 // (op 2674) 
v_mul_lo_u32 v17, s1, v14 // (op 2674) 

v_add_u32 v17, v17, v15 // (op 2674) 


v_mul_lo_u32 v15, v17, s31 // (op 2674) most significant: low of low * high
v_mul_hi_u32 v20, v17, s30 // (op 2674) most significant: high of low * low
v_mul_lo_u32 v14, v17, s30 // (op 2674) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 2674) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v14 // (op 2674) least significant half
v_addc_co_u32 v21, s[0:1], v19, v15, s[0:1] // (op 2674) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 2674) 


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 2674) 

v_mov_b32 v17, v14 // (op 2674) convert

s_mul_i32 s1, s30, s23 // (op 2676) most significant: low of low * high
s_mul_i32 s18, s31, s22 // (op 2676) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 2676) least significant
s_mul_hi_u32 s19, s30, s22 // (op 2676) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2676) most significant: sum
s_add_u32 s1, s1, s19 // (op 2676) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 2676) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 2676) 

s_mov_b32 s64, s32 // (op 2676) reg expression
s_mov_b32 s65, s33 // (op 2676) reg expression
s_mov_b32 s66, s0 // (op 2676) reg expression
s_mov_b32 s67, 131072 // (op 2676) call()

s_mul_hi_i32 s0, s16, s47 // (op 2719) 
v_lshrrev_b32 v14, 7, v1 // (op 2719) 
s_add_i32 s1, s0, s16 // (op 2719) 

v_and_b32 v15, 15, v1 // (op 2719) 
s_ashr_i32 s0, s1, 31 // (op 2719) 
v_lshl_add_u32 v18, v14, 4, v15 // (op 2719) 


s_cmp_eq_i32 s47, 0 // (op 2719) 
s_mov_b32 s18, scc // (op 2719) 
v_and_b32 v14, 63, v1 // (op 2719) 
s_cmp_eq_i32 s18, 0 // (op 2719) 
s_cselect_b32 s19, 0, -1 // (op 2719) 

v_lshrrev_b32 v15, 4, v14 // (op 2719) 

s_and_b32 s18, s11, 31 // (op 2719) 
s_lshl_b32 s20, 1, s18 // (op 2719) 
s_add_i32 s21, s20, s19 // (op 2719) 


s_and_b32 s19, s0, s21 // (op 2719) 


s_add_i32 s0, s1, s19 // (op 2719) 


s_ashr_i32 s1, s0, s18 // (op 2719) 


s_lshl_b32 s0, s11, 24 // (op 2719) 
s_ashr_i32 s18, s0, 31 // (op 2719) 

s_xor_b32 s0, s1, s18 // (op 2719) 

s_sub_i32 s1, s0, s18 // (op 2719) 


s_mul_hi_i32 s0, s1, s12 // (op 2719) 
s_add_i32 s18, s0, s1 // (op 2719) 

s_ashr_i32 s0, s18, 31 // (op 2719) 
s_cmp_eq_i32 s12, 0 // (op 2719) 
s_mov_b32 s19, scc // (op 2719) 
s_cmp_eq_i32 s19, 0 // (op 2719) 
s_cselect_b32 s20, 0, -1 // (op 2719) 

s_and_b32 s19, s13, 31 // (op 2719) 
s_lshl_b32 s21, 1, s19 // (op 2719) 
s_add_i32 s24, s21, s20 // (op 2719) 


s_and_b32 s25, s0, s24 // (op 2719) 


s_add_i32 s0, s18, s25 // (op 2719) 


s_ashr_i32 s18, s0, s19 // (op 2719) 


s_lshl_b32 s0, s13, 24 // (op 2719) 
s_ashr_i32 s19, s0, 31 // (op 2719) 

s_xor_b32 s0, s18, s19 // (op 2719) 

s_sub_i32 s18, s0, s19 // (op 2719) 


s_mul_i32 s0, s1, s48 // (op 2719) 

s_sub_u32 s1, s16, s0 // (op 2719) 

s_mul_i32 s0, s18, s48 // (op 2719) 
s_add_u32 s0, s0, s1 // (op 2719) 


s_mul_hi_i32 s1, s16, s10 // (op 2719) 
s_add_i32 s18, s1, s16 // (op 2719) 

s_ashr_i32 s1, s18, 31 // (op 2719) 
s_cmp_eq_i32 s10, 0 // (op 2719) 
s_mov_b32 s19, scc // (op 2719) 
s_cmp_eq_i32 s19, 0 // (op 2719) 
s_cselect_b32 s20, 0, -1 // (op 2719) 

s_and_b32 s19, s44, 31 // (op 2719) 
s_lshl_b32 s21, 1, s19 // (op 2719) 
s_add_i32 s24, s21, s20 // (op 2719) 


s_and_b32 s25, s1, s24 // (op 2719) 


s_add_i32 s1, s18, s25 // (op 2719) 


s_ashr_i32 s18, s1, s19 // (op 2719) 


s_lshl_b32 s1, s44, 24 // (op 2719) 
s_ashr_i32 s19, s1, 31 // (op 2719) 

s_xor_b32 s1, s18, s19 // (op 2719) 

s_sub_i32 s18, s1, s19 // (op 2719) 


s_mul_i32 s1, s48, s6 // (op 2719) 
s_mul_i32 s19, s14, s1 // (op 2719) 

s_mul_i32 s1, s18, s19 // (op 2719) 

s_sub_u32 s19, s16, s1 // (op 2719) 

s_mov_b32 s1, s19 // (op 2719) 
s_mul_hi_i32 s20, s1, s45 // (op 2719) 
s_add_i32 s21, s20, s1 // (op 2719) 


s_ashr_i32 s1, s21, 31 // (op 2719) 
s_cmp_eq_i32 s45, 0 // (op 2719) 
s_mov_b32 s20, scc // (op 2719) 
s_cmp_eq_i32 s20, 0 // (op 2719) 
s_cselect_b32 s24, 0, -1 // (op 2719) 

s_and_b32 s20, s15, 31 // (op 2719) 
s_lshl_b32 s25, 1, s20 // (op 2719) 
s_add_i32 s49, s25, s24 // (op 2719) 


s_and_b32 s24, s1, s49 // (op 2719) 


s_add_i32 s1, s21, s24 // (op 2719) 


s_ashr_i32 s21, s1, s20 // (op 2719) 


s_lshl_b32 s1, s15, 24 // (op 2719) 
s_ashr_i32 s20, s1, 31 // (op 2719) 

s_xor_b32 s1, s21, s20 // (op 2719) 

s_sub_i32 s21, s1, s20 // (op 2719) 


s_mul_i32 s1, s46, s48 // (op 2719) 
s_add_u32 s24, s8, 32 // (op 2719) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 2719) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2719) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2719) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 2719) 

s_sub_i32 s20, s24, s1 // (op 2719) 

s_mul_i32 s24, s21, s20 // (op 2719) 


s_sub_u32 s25, s19, s24 // (op 2719) 


s_add_u32 s24, s25, s1 // (op 2719) 


s_cmp_eq_i32 s18, 0 // (op 2719) 
s_mov_b32 s1, scc // (op 2719) 

s_cmp_eq_i32 s1, 0 // (op 2719) 
s_cselect_b32 s25, s24, s0 // (op 2719) 



v_lshl_add_u32 v14, s25, 5, v18 // (op 2719) 


v_mul_lo_u32 v18, v14, s34 // (op 2719) 
v_add_u32 v18, v18, v15 // (op 2719) 


s_mul_i32 s1, s34, s9 // (op 2721) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 2721) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2721) least significant
s_mul_hi_u32 s19, s34, s8 // (op 2721) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2721) most significant: sum
s_add_u32 s1, s1, s19 // (op 2721) most significant: sum


s_mov_b32 s68, s36 // (op 2721) reg expression
s_mov_b32 s69, s37 // (op 2721) reg expression
s_mov_b32 s70, s0 // (op 2721) reg expression
s_mov_b32 s71, 131072 // (op 2721) call()

s_mul_hi_i32 s0, s16, s47 // (op 2738) 
v_lshrrev_b32 v14, 7, v1 // (op 2738) 
s_add_i32 s1, s0, s16 // (op 2738) 

v_and_b32 v15, 15, v1 // (op 2738) 
s_ashr_i32 s0, s1, 31 // (op 2738) 
v_lshl_add_u32 v19, v14, 4, v15 // (op 2738) 


s_cmp_eq_i32 s47, 0 // (op 2738) 
s_mov_b32 s18, scc // (op 2738) 
v_and_b32 v14, 63, v1 // (op 2738) 
s_cmp_eq_i32 s18, 0 // (op 2738) 
s_cselect_b32 s19, 0, -1 // (op 2738) 

v_lshrrev_b32 v15, 4, v14 // (op 2738) 

s_and_b32 s18, s11, 31 // (op 2738) 
s_lshl_b32 s20, 1, s18 // (op 2738) 
s_add_i32 s21, s20, s19 // (op 2738) 


s_and_b32 s19, s0, s21 // (op 2738) 


s_add_i32 s0, s1, s19 // (op 2738) 


s_ashr_i32 s1, s0, s18 // (op 2738) 


s_lshl_b32 s0, s11, 24 // (op 2738) 
s_ashr_i32 s18, s0, 31 // (op 2738) 

s_xor_b32 s0, s1, s18 // (op 2738) 

s_sub_i32 s1, s0, s18 // (op 2738) 


s_mul_hi_i32 s0, s1, s12 // (op 2738) 
s_add_i32 s18, s0, s1 // (op 2738) 

s_ashr_i32 s0, s18, 31 // (op 2738) 
s_cmp_eq_i32 s12, 0 // (op 2738) 
s_mov_b32 s19, scc // (op 2738) 
s_cmp_eq_i32 s19, 0 // (op 2738) 
s_cselect_b32 s20, 0, -1 // (op 2738) 

s_and_b32 s19, s13, 31 // (op 2738) 
s_lshl_b32 s21, 1, s19 // (op 2738) 
s_add_i32 s24, s21, s20 // (op 2738) 


s_and_b32 s25, s0, s24 // (op 2738) 


s_add_i32 s0, s18, s25 // (op 2738) 


s_ashr_i32 s18, s0, s19 // (op 2738) 


s_lshl_b32 s0, s13, 24 // (op 2738) 
s_ashr_i32 s19, s0, 31 // (op 2738) 

s_xor_b32 s0, s18, s19 // (op 2738) 

s_sub_i32 s18, s0, s19 // (op 2738) 


s_mul_i32 s0, s1, s48 // (op 2738) 

s_sub_u32 s1, s16, s0 // (op 2738) 

s_mul_i32 s0, s18, s48 // (op 2738) 
s_add_u32 s0, s0, s1 // (op 2738) 


s_mul_hi_i32 s1, s16, s10 // (op 2738) 
s_add_i32 s18, s1, s16 // (op 2738) 

s_ashr_i32 s1, s18, 31 // (op 2738) 
s_cmp_eq_i32 s10, 0 // (op 2738) 
s_mov_b32 s19, scc // (op 2738) 
s_cmp_eq_i32 s19, 0 // (op 2738) 
s_cselect_b32 s20, 0, -1 // (op 2738) 

s_and_b32 s19, s44, 31 // (op 2738) 
s_lshl_b32 s21, 1, s19 // (op 2738) 
s_add_i32 s24, s21, s20 // (op 2738) 


s_and_b32 s25, s1, s24 // (op 2738) 


s_add_i32 s1, s18, s25 // (op 2738) 


s_ashr_i32 s18, s1, s19 // (op 2738) 


s_lshl_b32 s1, s44, 24 // (op 2738) 
s_ashr_i32 s19, s1, 31 // (op 2738) 

s_xor_b32 s1, s18, s19 // (op 2738) 

s_sub_i32 s18, s1, s19 // (op 2738) 


s_mul_i32 s1, s48, s6 // (op 2738) 
s_mul_i32 s19, s14, s1 // (op 2738) 

s_mul_i32 s1, s18, s19 // (op 2738) 

s_sub_u32 s19, s16, s1 // (op 2738) 

s_mov_b32 s1, s19 // (op 2738) 
s_mul_hi_i32 s20, s1, s45 // (op 2738) 
s_add_i32 s21, s20, s1 // (op 2738) 


s_ashr_i32 s1, s21, 31 // (op 2738) 
s_cmp_eq_i32 s45, 0 // (op 2738) 
s_mov_b32 s20, scc // (op 2738) 
s_cmp_eq_i32 s20, 0 // (op 2738) 
s_cselect_b32 s24, 0, -1 // (op 2738) 

s_and_b32 s20, s15, 31 // (op 2738) 
s_lshl_b32 s25, 1, s20 // (op 2738) 
s_add_i32 s49, s25, s24 // (op 2738) 


s_and_b32 s24, s1, s49 // (op 2738) 


s_add_i32 s1, s21, s24 // (op 2738) 


s_ashr_i32 s21, s1, s20 // (op 2738) 


s_lshl_b32 s1, s15, 24 // (op 2738) 
s_ashr_i32 s20, s1, 31 // (op 2738) 

s_xor_b32 s1, s21, s20 // (op 2738) 

s_sub_i32 s21, s1, s20 // (op 2738) 


s_mul_i32 s1, s46, s48 // (op 2738) 
s_add_u32 s24, s8, 32 // (op 2738) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 2738) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2738) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2738) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 2738) 

s_sub_i32 s20, s24, s1 // (op 2738) 

s_mul_i32 s24, s21, s20 // (op 2738) 


s_sub_u32 s25, s19, s24 // (op 2738) 


s_add_u32 s24, s25, s1 // (op 2738) 


s_cmp_eq_i32 s18, 0 // (op 2738) 
s_mov_b32 s1, scc // (op 2738) 

s_cmp_eq_i32 s1, 0 // (op 2738) 
s_cselect_b32 s25, s24, s0 // (op 2738) 



v_lshl_add_u32 v14, s25, 5, v19 // (op 2738) 


v_mul_lo_u32 v19, v14, s34 // (op 2738) 
v_add_u32 v19, v19, v15 // (op 2738) 


s_mul_i32 s1, s34, s9 // (op 2740) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 2740) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2740) least significant
s_mul_hi_u32 s19, s34, s8 // (op 2740) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2740) most significant: sum
s_add_u32 s1, s1, s19 // (op 2740) most significant: sum


s_mov_b32 s68, s36 // (op 2740) reg expression
s_mov_b32 s69, s37 // (op 2740) reg expression
s_mov_b32 s70, s0 // (op 2740) reg expression
s_mov_b32 s71, 131072 // (op 2740) call()

s_mul_hi_i32 s0, s16, s47 // (op 2783) 
v_lshrrev_b32 v14, 2, v1 // (op 2783) 
s_add_i32 s1, s0, s16 // (op 2783) 

v_mul_lo_u32 v15, v14, 3 // (op 2783) 

s_ashr_i32 s0, s1, 31 // (op 2783) 
v_and_b32 v14, -4, v1 // (op 2783) 
s_cmp_eq_i32 s47, 0 // (op 2783) 
s_mov_b32 s18, scc // (op 2783) 
v_sub_u32 v20, v1, v14 // (op 2783) 

s_cmp_eq_i32 s18, 0 // (op 2783) 
s_cselect_b32 s19, 0, -1 // (op 2783) 

s_and_b32 s18, s11, 31 // (op 2783) 
s_lshl_b32 s20, 1, s18 // (op 2783) 
s_add_i32 s21, s20, s19 // (op 2783) 


s_and_b32 s19, s0, s21 // (op 2783) 


s_add_i32 s0, s1, s19 // (op 2783) 


s_ashr_i32 s1, s0, s18 // (op 2783) 


s_lshl_b32 s0, s11, 24 // (op 2783) 
s_ashr_i32 s18, s0, 31 // (op 2783) 

s_xor_b32 s0, s1, s18 // (op 2783) 

s_sub_i32 s1, s0, s18 // (op 2783) 


s_mul_hi_i32 s0, s1, s12 // (op 2783) 
s_add_i32 s18, s0, s1 // (op 2783) 

s_ashr_i32 s0, s18, 31 // (op 2783) 
s_cmp_eq_i32 s12, 0 // (op 2783) 
s_mov_b32 s19, scc // (op 2783) 
s_cmp_eq_i32 s19, 0 // (op 2783) 
s_cselect_b32 s20, 0, -1 // (op 2783) 

s_and_b32 s19, s13, 31 // (op 2783) 
s_lshl_b32 s21, 1, s19 // (op 2783) 
s_add_i32 s24, s21, s20 // (op 2783) 


s_and_b32 s25, s0, s24 // (op 2783) 


s_add_i32 s0, s18, s25 // (op 2783) 


s_ashr_i32 s18, s0, s19 // (op 2783) 


s_lshl_b32 s0, s13, 24 // (op 2783) 
s_ashr_i32 s19, s0, 31 // (op 2783) 

s_xor_b32 s0, s18, s19 // (op 2783) 

s_sub_i32 s18, s0, s19 // (op 2783) 


s_mul_i32 s0, s18, s6 // (op 2783) 

s_sub_i32 s18, s1, s0 // (op 2783) 


s_mul_hi_i32 s0, s16, s10 // (op 2783) 
s_add_i32 s1, s0, s16 // (op 2783) 

s_ashr_i32 s0, s1, 31 // (op 2783) 
s_cmp_eq_i32 s10, 0 // (op 2783) 
s_mov_b32 s19, scc // (op 2783) 
s_cmp_eq_i32 s19, 0 // (op 2783) 
s_cselect_b32 s20, 0, -1 // (op 2783) 

s_and_b32 s19, s44, 31 // (op 2783) 
s_lshl_b32 s21, 1, s19 // (op 2783) 
s_add_i32 s24, s21, s20 // (op 2783) 


s_and_b32 s25, s0, s24 // (op 2783) 


s_add_i32 s0, s1, s25 // (op 2783) 


s_ashr_i32 s1, s0, s19 // (op 2783) 


s_lshl_b32 s0, s44, 24 // (op 2783) 
s_ashr_i32 s19, s0, 31 // (op 2783) 

s_xor_b32 s0, s1, s19 // (op 2783) 

s_sub_i32 s1, s0, s19 // (op 2783) 


s_mul_i32 s0, s48, s6 // (op 2783) 
s_mul_i32 s19, s14, s0 // (op 2783) 

s_mul_i32 s0, s1, s19 // (op 2783) 

s_sub_u32 s19, s16, s0 // (op 2783) 

s_mul_hi_i32 s0, s19, s45 // (op 2783) 
s_add_i32 s20, s0, s19 // (op 2783) 


s_ashr_i32 s0, s20, 31 // (op 2783) 
s_cmp_eq_i32 s45, 0 // (op 2783) 
s_mov_b32 s19, scc // (op 2783) 
s_cmp_eq_i32 s19, 0 // (op 2783) 
s_cselect_b32 s21, 0, -1 // (op 2783) 

s_and_b32 s19, s15, 31 // (op 2783) 
s_lshl_b32 s24, 1, s19 // (op 2783) 
s_add_i32 s25, s24, s21 // (op 2783) 


s_and_b32 s21, s0, s25 // (op 2783) 


s_add_i32 s0, s20, s21 // (op 2783) 


s_ashr_i32 s20, s0, s19 // (op 2783) 


s_lshl_b32 s0, s15, 24 // (op 2783) 
s_ashr_i32 s19, s0, 31 // (op 2783) 

s_xor_b32 s0, s20, s19 // (op 2783) 

s_sub_i32 s20, s0, s19 // (op 2783) 


s_cmp_eq_i32 s1, 0 // (op 2783) 
s_mov_b32 s0, scc // (op 2783) 

s_cmp_eq_i32 s0, 0 // (op 2783) 
s_cselect_b32 s1, s20, s18 // (op 2783) 



v_mov_b32 v14, 192 // (op 2783) 
v_mul_lo_u32 v21, s1, v14 // (op 2783) 

v_add_u32 v21, v21, v15 // (op 2783) 


v_mul_lo_u32 v14, v21, s38 // (op 2783) 

v_add_u32 v15, v20, v14 // (op 2783) 


s_mul_i32 s1, s38, s23 // (op 2785) most significant: low of low * high
s_mul_i32 s18, s39, s22 // (op 2785) most significant: low of high * low
s_mul_i32 s0, s38, s22 // (op 2785) least significant
s_mul_hi_u32 s19, s38, s22 // (op 2785) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2785) most significant: sum
s_add_u32 s1, s1, s19 // (op 2785) most significant: sum


s_mov_b32 s72, s40 // (op 2785) reg expression
s_mov_b32 s73, s41 // (op 2785) reg expression
s_mov_b32 s74, s0 // (op 2785) reg expression
s_mov_b32 s75, 131072 // (op 2785) call()

s_mul_hi_i32 s0, s16, s47 // (op 2802) 
v_lshrrev_b32 v14, 2, v1 // (op 2802) 
s_add_i32 s1, s0, s16 // (op 2802) 

v_mul_lo_u32 v20, v14, 3 // (op 2802) 

s_ashr_i32 s0, s1, 31 // (op 2802) 
v_and_b32 v14, -4, v1 // (op 2802) 
s_cmp_eq_i32 s47, 0 // (op 2802) 
s_mov_b32 s18, scc // (op 2802) 
v_sub_u32 v21, v1, v14 // (op 2802) 

s_cmp_eq_i32 s18, 0 // (op 2802) 
s_cselect_b32 s19, 0, -1 // (op 2802) 

s_and_b32 s18, s11, 31 // (op 2802) 
s_lshl_b32 s20, 1, s18 // (op 2802) 
s_add_i32 s21, s20, s19 // (op 2802) 


s_and_b32 s19, s0, s21 // (op 2802) 


s_add_i32 s0, s1, s19 // (op 2802) 


s_ashr_i32 s1, s0, s18 // (op 2802) 


s_lshl_b32 s0, s11, 24 // (op 2802) 
s_ashr_i32 s18, s0, 31 // (op 2802) 

s_xor_b32 s0, s1, s18 // (op 2802) 

s_sub_i32 s1, s0, s18 // (op 2802) 


s_mul_hi_i32 s0, s1, s12 // (op 2802) 
s_add_i32 s18, s0, s1 // (op 2802) 

s_ashr_i32 s0, s18, 31 // (op 2802) 
s_cmp_eq_i32 s12, 0 // (op 2802) 
s_mov_b32 s19, scc // (op 2802) 
s_cmp_eq_i32 s19, 0 // (op 2802) 
s_cselect_b32 s20, 0, -1 // (op 2802) 

s_and_b32 s19, s13, 31 // (op 2802) 
s_lshl_b32 s21, 1, s19 // (op 2802) 
s_add_i32 s24, s21, s20 // (op 2802) 


s_and_b32 s25, s0, s24 // (op 2802) 


s_add_i32 s0, s18, s25 // (op 2802) 


s_ashr_i32 s18, s0, s19 // (op 2802) 


s_lshl_b32 s0, s13, 24 // (op 2802) 
s_ashr_i32 s19, s0, 31 // (op 2802) 

s_xor_b32 s0, s18, s19 // (op 2802) 

s_sub_i32 s18, s0, s19 // (op 2802) 


s_mul_i32 s0, s18, s6 // (op 2802) 

s_sub_i32 s18, s1, s0 // (op 2802) 


s_mul_hi_i32 s0, s16, s10 // (op 2802) 
s_add_i32 s1, s0, s16 // (op 2802) 

s_ashr_i32 s0, s1, 31 // (op 2802) 
s_cmp_eq_i32 s10, 0 // (op 2802) 
s_mov_b32 s19, scc // (op 2802) 
s_cmp_eq_i32 s19, 0 // (op 2802) 
s_cselect_b32 s20, 0, -1 // (op 2802) 

s_and_b32 s19, s44, 31 // (op 2802) 
s_lshl_b32 s21, 1, s19 // (op 2802) 
s_add_i32 s24, s21, s20 // (op 2802) 


s_and_b32 s25, s0, s24 // (op 2802) 


s_add_i32 s0, s1, s25 // (op 2802) 


s_ashr_i32 s1, s0, s19 // (op 2802) 


s_lshl_b32 s0, s44, 24 // (op 2802) 
s_ashr_i32 s19, s0, 31 // (op 2802) 

s_xor_b32 s0, s1, s19 // (op 2802) 

s_sub_i32 s1, s0, s19 // (op 2802) 


s_mul_i32 s0, s48, s6 // (op 2802) 
s_mul_i32 s19, s14, s0 // (op 2802) 

s_mul_i32 s0, s1, s19 // (op 2802) 

s_sub_u32 s19, s16, s0 // (op 2802) 

s_mul_hi_i32 s0, s19, s45 // (op 2802) 
s_add_i32 s20, s0, s19 // (op 2802) 


s_ashr_i32 s0, s20, 31 // (op 2802) 
s_cmp_eq_i32 s45, 0 // (op 2802) 
s_mov_b32 s19, scc // (op 2802) 
s_cmp_eq_i32 s19, 0 // (op 2802) 
s_cselect_b32 s21, 0, -1 // (op 2802) 

s_and_b32 s19, s15, 31 // (op 2802) 
s_lshl_b32 s24, 1, s19 // (op 2802) 
s_add_i32 s25, s24, s21 // (op 2802) 


s_and_b32 s21, s0, s25 // (op 2802) 


s_add_i32 s0, s20, s21 // (op 2802) 


s_ashr_i32 s20, s0, s19 // (op 2802) 


s_lshl_b32 s0, s15, 24 // (op 2802) 
s_ashr_i32 s19, s0, 31 // (op 2802) 

s_xor_b32 s0, s20, s19 // (op 2802) 

s_sub_i32 s20, s0, s19 // (op 2802) 


s_cmp_eq_i32 s1, 0 // (op 2802) 
s_mov_b32 s0, scc // (op 2802) 

s_cmp_eq_i32 s0, 0 // (op 2802) 
s_cselect_b32 s1, s20, s18 // (op 2802) 



v_mov_b32 v14, 192 // (op 2802) 
v_mul_lo_u32 v22, s1, v14 // (op 2802) 

v_add_u32 v22, v22, v20 // (op 2802) 


v_mul_lo_u32 v14, v22, s38 // (op 2802) 

v_add_u32 v20, v21, v14 // (op 2802) 


s_mul_i32 s1, s38, s23 // (op 2804) most significant: low of low * high
s_mul_i32 s18, s39, s22 // (op 2804) most significant: low of high * low
s_mul_i32 s0, s38, s22 // (op 2804) least significant
s_mul_hi_u32 s19, s38, s22 // (op 2804) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2804) most significant: sum
s_add_u32 s1, s1, s19 // (op 2804) most significant: sum


s_mov_b32 s72, s40 // (op 2804) reg expression
s_mov_b32 s73, s41 // (op 2804) reg expression
s_mov_b32 s74, s0 // (op 2804) reg expression
s_mov_b32 s75, 131072 // (op 2804) call()




buffer_load_ubyte v14, v18, s[68:71], 0 offen offset: 0 // (op 2350) Load value

s_lshl_b64 s[0:1], s[30:31], 2 // (op 2354) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 2354) 


v_mov_b32 v21, v16 // (op 2354) 
buffer_load_dwordx4 v[24:27], v21, s[64:67], 0 offen // (op 2354) Load value
v_add_u32 v21, v21, v22 // (op 2354) 
buffer_load_dwordx4 v[28:31], v21, s[64:67], 0 offen // (op 2354) Load value
v_add_u32 v21, v21, v22 // (op 2354) 
buffer_load_dwordx4 v[32:35], v21, s[64:67], 0 offen // (op 2354) Load value



v_mov_b64 v[22:23], s[38:39] // (op 2357) convert

v_mov_b32 v21, v15 // (op 2357) 
buffer_load_ubyte v36, v21, s[72:75], 0 offen offset: 0 // (op 2357) Load value
v_add_u32 v21, v21, v22 // (op 2357) 
buffer_load_ubyte v37, v21, s[72:75], 0 offen offset: 0 // (op 2357) Load value
v_add_u32 v21, v21, v22 // (op 2357) 
buffer_load_ubyte v38, v21, s[72:75], 0 offen offset: 0 // (op 2357) Load value





buffer_load_dwordx2 v[22:23], v12, s[52:55], 0 offen // (op 2360) Load value



buffer_load_ubyte v21, v18, s[68:71], 0 offen offset:4 // (op 2364) Load value


s_lshl_b64 s[0:1], s[30:31], 2 // (op 2368) 
v_lshrrev_b64 v[40:41], 3, s[0:1] // (op 2368) 


v_mov_b32 v18, v16 // (op 2368) 
buffer_load_dwordx4 v[42:45], v18, s[64:67], 0 offen offset:64 // (op 2368) Load value
v_add_u32 v18, v18, v40 // (op 2368) 
buffer_load_dwordx4 v[46:49], v18, s[64:67], 0 offen offset:64 // (op 2368) Load value
v_add_u32 v18, v18, v40 // (op 2368) 
buffer_load_dwordx4 v[50:53], v18, s[64:67], 0 offen offset:64 // (op 2368) Load value




v_mov_b64 v[40:41], s[38:39] // (op 2371) convert

v_mov_b32 v16, v15 // (op 2371) 
buffer_load_ubyte v54, v16, s[72:75], 0 offen offset:4 // (op 2371) Load value
v_add_u32 v16, v16, v40 // (op 2371) 
buffer_load_ubyte v55, v16, s[72:75], 0 offen offset:4 // (op 2371) Load value
v_add_u32 v16, v16, v40 // (op 2371) 
buffer_load_ubyte v56, v16, s[72:75], 0 offen offset:4 // (op 2371) Load value






buffer_load_dwordx2 v[40:41], v12, s[52:55], 0 offen offset:64 // (op 2374) Load value

v_and_b32 v12, -4, v1 // (op 2922) 
v_sub_u32 v18, v1, v12 // (op 2922) 

v_lshlrev_b32 v12, 5, v18 // (op 2922) 

v_lshrrev_b32 v18, 2, v1 // (op 2922) 
v_mul_lo_u32 v39, v18, 3 // (op 2922) 

v_lshl_add_u32 v18, v39, 7, v12 // (op 2922) 


v_lshlrev_b32 v12, 2, v18 // (op 2922) 

v_lshrrev_b32 v18, 3, v12 // (op 2922) 




s_waitcnt vmcnt(14)
ds_write_b128 v18, v[24:27] // (op 2376) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v18, v[28:31] offset:64 // (op 2376) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[32:35] offset:128 // (op 2376) Store local data 

v_and_b32 v12, -4, v1 // (op 2980) 
v_sub_u32 v18, v1, v12 // (op 2980) 

v_lshrrev_b32 v12, 2, v1 // (op 2980) 
v_mul_lo_u32 v39, v12, 3 // (op 2980) 

v_lshl_add_u32 v12, v39, 2, v18 // (op 2980) 





s_waitcnt vmcnt(11)
ds_write_b8 v12, v36 offset:12288 // (op 2377) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v12, v37 offset:12292 // (op 2377) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v12, v38 offset:12296 // (op 2377) Store local data 

v_and_b32 v12, -8, v1 // (op 2869) 
v_sub_u32 v18, v1, v12 // (op 2869) 

v_lshlrev_b32 v12, 4, v18 // (op 2869) 

v_lshrrev_b32 v18, 3, v1 // (op 2869) 
v_lshl_add_u32 v39, v18, 7, v12 // (op 2869) 


v_lshlrev_b32 v12, 2, v39 // (op 2869) 

v_lshrrev_b32 v18, 3, v12 // (op 2869) 




s_waitcnt vmcnt(8)
ds_write_b64 v18, v[22:23] offset:13056 // (op 2378) Store local data 

v_and_b32 v12, -8, v1 // (op 2840) 
v_sub_u32 v18, v1, v12 // (op 2840) 

v_lshlrev_b32 v12, 4, v18 // (op 2840) 

v_lshrrev_b32 v18, 3, v1 // (op 2840) 
v_lshl_add_u32 v39, v18, 7, v12 // (op 2840) 


v_lshlrev_b32 v12, 2, v39 // (op 2840) 

v_lshrrev_b32 v18, 3, v12 // (op 2840) 

v_and_b32 v12, -4, v1 // (op 2893) 
v_sub_u32 v39, v1, v12 // (op 2893) 

v_lshlrev_b32 v12, 5, v39 // (op 2893) 

v_lshrrev_b32 v39, 2, v1 // (op 2893) 
v_mul_lo_u32 v15, v39, 3 // (op 2893) 

v_lshl_add_u32 v16, v15, 7, v12 // (op 2893) 


v_lshlrev_b32 v12, 2, v16 // (op 2893) 

v_lshrrev_b32 v39, 3, v12 // (op 2893) 

v_and_b32 v12, -4, v1 // (op 2951) 
v_sub_u32 v15, v1, v12 // (op 2951) 

v_lshrrev_b32 v12, 2, v1 // (op 2951) 
v_mul_lo_u32 v16, v12, 3 // (op 2951) 

v_lshl_add_u32 v12, v16, 2, v15 // (op 2951) 


v_and_b32 v15, -8, v1 // (op 3027) 
v_sub_u32 v16, v1, v15 // (op 3027) 

v_lshlrev_b32 v15, 4, v16 // (op 3027) 

v_lshrrev_b32 v16, 3, v1 // (op 3027) 
v_lshl_add_u32 v57, v16, 7, v15 // (op 3027) 


v_lshlrev_b32 v15, 2, v57 // (op 3027) 

v_lshrrev_b32 v16, 3, v15 // (op 3027) 

v_and_b32 v15, -4, v1 // (op 3051) 
v_sub_u32 v57, v1, v15 // (op 3051) 

v_lshlrev_b32 v15, 5, v57 // (op 3051) 

v_lshrrev_b32 v57, 2, v1 // (op 3051) 
v_mul_lo_u32 v58, v57, 3 // (op 3051) 

v_lshl_add_u32 v57, v58, 7, v15 // (op 3051) 


v_lshlrev_b32 v15, 2, v57 // (op 3051) 

v_lshrrev_b32 v57, 3, v15 // (op 3051) 

v_and_b32 v15, -4, v1 // (op 3080) 
v_sub_u32 v58, v1, v15 // (op 3080) 

v_lshrrev_b32 v15, 2, v1 // (op 3080) 
v_mul_lo_u32 v59, v15, 3 // (op 3080) 

v_lshl_add_u32 v15, v59, 2, v58 // (op 3080) 


s_mov_b32 s0, 0 // (op 936) call()
s_ashr_i64 s[24:25], s[42:43], 7 // (op 935) 
s_mov_b32 s1, s24 // (op 935) convert

s_and_b32 s18, s1, 4294967294 // (op 935) 

s_cmp_lt_u32 s0, s18 // (op 935) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopBottom_KLoop_935 // (op 935) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopBottom_KLoop_935 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopTop_KLoop_935:
s_add_u32 s0, s0, 2 // (op 937) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3150) 



ds_read_b128 v[58:61], v3 offset:13056 // (op 67) Load local data 
s_barrier  // (op 3163) 



ds_read_b128 v[62:65], v6 // (op 72) Load local data 
s_barrier  // (op 3177) 



ds_read_u8 v66, v5 offset:12288 // (op 81) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:61], v[62:65], a[0:3], v14, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 84) 
ds_read_b128 v[68:71], v6 offset:2048 // (op 343) Load local data 





ds_read_u8 v62, v5 offset:12416 // (op 346) Load local data 



ds_read_b128 v[64:67], v6 offset:4096 // (op 499) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:61], v[68:71], a[4:7], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 347) 





ds_read_u8 v62, v5 offset:12544 // (op 502) Load local data 



ds_read_b128 v[68:71], v6 offset:6144 // (op 655) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:61], v[64:67], a[8:11], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 503) 





ds_read_u8 v62, v5 offset:12672 // (op 658) Load local data 



ds_read_b128 v[64:67], v6 offset:8192 // (op 811) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:61], v[68:71], a[12:15], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 659) 





ds_read_u8 v62, v5 offset:12800 // (op 814) Load local data 



ds_read_b128 v[68:71], v6 offset:10240 // (op 967) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:61], v[64:67], a[16:19], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 815) 





ds_read_u8 v62, v5 offset:12928 // (op 970) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:61], v[68:71], a[20:23], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 971) 





v_mov_b32 v58, v19 // (op 77) 

buffer_load_ubyte v14, v58, s[68:71], 0 offen offset:8 // (op 77) Load value

s_lshl_b64 s[24:25], s[30:31], 2 // (op 10) 
v_lshrrev_b64 v[60:61], 3, s[24:25] // (op 10) 


v_mov_b32 v59, v17 // (op 10) 
v_mov_b32 v62, v59 // (op 10) 
buffer_load_dwordx4 v[24:27], v62, s[64:67], 0 offen offset:128 // (op 10) Load value
v_add_u32 v62, v62, v60 // (op 10) 
buffer_load_dwordx4 v[28:31], v62, s[64:67], 0 offen offset:128 // (op 10) Load value
v_add_u32 v62, v62, v60 // (op 10) 
buffer_load_dwordx4 v[32:35], v62, s[64:67], 0 offen offset:128 // (op 10) Load value



v_mov_b64 v[60:61], s[38:39] // (op 21) convert

v_mov_b32 v62, v20 // (op 21) 
v_mov_b32 v63, v62 // (op 21) 
buffer_load_ubyte v36, v63, s[72:75], 0 offen offset:8 // (op 21) Load value
v_add_u32 v63, v63, v60 // (op 21) 
buffer_load_ubyte v37, v63, s[72:75], 0 offen offset:8 // (op 21) Load value
v_add_u32 v63, v63, v60 // (op 21) 
buffer_load_ubyte v38, v63, s[72:75], 0 offen offset:8 // (op 21) Load value




v_mov_b32 v60, v13 // (op 4) 

buffer_load_dwordx2 v[22:23], v60, s[52:55], 0 offen offset:128 // (op 4) Load value
s_barrier  // (op 3200) 



s_waitcnt vmcnt(14)
ds_write_b128 v57, v[42:45] offset:15104 // (op 239) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v57, v[46:49] offset:15168 // (op 239) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v57, v[50:53] offset:15232 // (op 239) Store local data 
s_barrier  // (op 3206) 



s_waitcnt vmcnt(11)
ds_write_b8 v15, v54 offset:27392 // (op 240) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v15, v55 offset:27396 // (op 240) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v15, v56 offset:27400 // (op 240) Store local data 
s_barrier  // (op 3194) 



s_waitcnt vmcnt(8)
ds_write_b64 v16, v[40:41] offset:28160 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3191) 



ds_read_b128 v[64:67], v4 offset:28160 // (op 231) Load local data 
s_barrier  // (op 3197) 
s_barrier  // (op 3247) 



ds_read_b128 v[68:71], v9 offset:15104 // (op 233) Load local data 
s_barrier  // (op 3203) 



ds_read_u8 v61, v8 offset:27392 // (op 237) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:67], v[68:71], a[0:3], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 238) 
ds_read_b128 v[72:75], v9 offset:17152 // (op 352) Load local data 





ds_read_u8 v61, v8 offset:27520 // (op 355) Load local data 



ds_read_b128 v[68:71], v9 offset:19200 // (op 508) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:67], v[72:75], a[4:7], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 348) 





ds_read_u8 v61, v8 offset:27648 // (op 511) Load local data 



ds_read_b128 v[72:75], v9 offset:21248 // (op 664) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:67], v[68:71], a[8:11], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 504) 





ds_read_u8 v61, v8 offset:27776 // (op 667) Load local data 



ds_read_b128 v[68:71], v9 offset:23296 // (op 820) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:67], v[72:75], a[12:15], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 660) 





ds_read_u8 v61, v8 offset:27904 // (op 823) Load local data 



ds_read_b128 v[72:75], v9 offset:25344 // (op 976) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:67], v[68:71], a[16:19], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 816) 





s_barrier  // (op 3305) 
ds_read_u8 v61, v8 offset:28032 // (op 979) Load local data 
s_barrier  // (op 3309) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:67], v[72:75], a[20:23], v21, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 972) 






buffer_load_ubyte v21, v58, s[68:71], 0 offen offset:12 // (op 235) Load value


v_add_u32 v19, 8, v19 // (op 2730) 
s_lshl_b64 s[24:25], s[30:31], 2 // (op 241) 
v_lshrrev_b64 v[64:65], 3, s[24:25] // (op 241) 


v_mov_b32 v58, v59 // (op 241) 
buffer_load_dwordx4 v[42:45], v58, s[64:67], 0 offen offset:192 // (op 241) Load value
v_add_u32 v58, v58, v64 // (op 241) 
buffer_load_dwordx4 v[46:49], v58, s[64:67], 0 offen offset:192 // (op 241) Load value
v_add_u32 v58, v58, v64 // (op 241) 
buffer_load_dwordx4 v[50:53], v58, s[64:67], 0 offen offset:192 // (op 241) Load value




v_add_u32 v17, 128, v17 // (op 2666) 
v_mov_b64 v[58:59], s[38:39] // (op 242) convert

v_mov_b32 v61, v62 // (op 242) 
buffer_load_ubyte v54, v61, s[72:75], 0 offen offset:12 // (op 242) Load value
v_add_u32 v61, v61, v58 // (op 242) 
buffer_load_ubyte v55, v61, s[72:75], 0 offen offset:12 // (op 242) Load value
v_add_u32 v61, v61, v58 // (op 242) 
buffer_load_ubyte v56, v61, s[72:75], 0 offen offset:12 // (op 242) Load value






v_add_u32 v20, 8, v20 // (op 2794) 
buffer_load_dwordx2 v[40:41], v60, s[52:55], 0 offen offset:192 // (op 228) Load value

s_barrier  // (op 3166) 
v_add_u32 v13, 128, v13 // (op 2602) 



s_waitcnt vmcnt(14)
ds_write_b128 v39, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v39, v[28:31] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v39, v[32:35] offset:128 // (op 13) Store local data 
s_barrier  // (op 3180) 



s_waitcnt vmcnt(11)
ds_write_b8 v12, v36 offset:12288 // (op 26) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v12, v37 offset:12292 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v12, v38 offset:12296 // (op 26) Store local data 
s_barrier  // (op 3153) 



s_waitcnt vmcnt(8)
ds_write_b64 v18, v[22:23] offset:13056 // (op 6) Store local data 
s_ashr_i64 s[24:25], s[42:43], 7 // (op 935) 
s_mov_b32 s1, s24 // (op 935) convert

s_and_b32 s18, s1, 4294967294 // (op 935) 

s_cmp_lt_u32 s0, s18 // (op 935) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopTop_KLoop_935 // (op 935) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopTop_KLoop_935 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopBottom_KLoop_935:















s_mul_hi_i32 s0, s16, s47 // (op 2560) 
v_lshrrev_b32 v12, 3, v1 // (op 2560) 
s_add_i32 s1, s0, s16 // (op 2560) 

v_and_b32 v13, -8, v1 // (op 2560) 
s_ashr_i32 s0, s1, 31 // (op 2560) 
v_sub_u32 v14, v1, v13 // (op 2560) 

s_cmp_eq_i32 s47, 0 // (op 2560) 
s_mov_b32 s18, scc // (op 2560) 
v_lshlrev_b32 v13, 4, v14 // (op 2560) 

s_cmp_eq_i32 s18, 0 // (op 2560) 
s_cselect_b32 s19, 0, -1 // (op 2560) 

s_and_b32 s18, s11, 31 // (op 2560) 
s_lshl_b32 s20, 1, s18 // (op 2560) 
s_add_i32 s21, s20, s19 // (op 2560) 


s_and_b32 s19, s0, s21 // (op 2560) 


s_add_i32 s0, s1, s19 // (op 2560) 


s_ashr_i32 s1, s0, s18 // (op 2560) 


s_lshl_b32 s0, s11, 24 // (op 2560) 
s_ashr_i32 s18, s0, 31 // (op 2560) 

s_xor_b32 s0, s1, s18 // (op 2560) 

s_sub_i32 s1, s0, s18 // (op 2560) 


s_mul_hi_i32 s0, s1, s12 // (op 2560) 
s_add_i32 s18, s0, s1 // (op 2560) 

s_ashr_i32 s0, s18, 31 // (op 2560) 
s_cmp_eq_i32 s12, 0 // (op 2560) 
s_mov_b32 s19, scc // (op 2560) 
s_cmp_eq_i32 s19, 0 // (op 2560) 
s_cselect_b32 s20, 0, -1 // (op 2560) 

s_and_b32 s19, s13, 31 // (op 2560) 
s_lshl_b32 s21, 1, s19 // (op 2560) 
s_add_i32 s24, s21, s20 // (op 2560) 


s_and_b32 s25, s0, s24 // (op 2560) 


s_add_i32 s0, s18, s25 // (op 2560) 


s_ashr_i32 s18, s0, s19 // (op 2560) 


s_lshl_b32 s0, s13, 24 // (op 2560) 
s_ashr_i32 s19, s0, 31 // (op 2560) 

s_xor_b32 s0, s18, s19 // (op 2560) 

s_sub_i32 s18, s0, s19 // (op 2560) 


s_mul_i32 s0, s1, s48 // (op 2560) 

s_sub_u32 s1, s16, s0 // (op 2560) 

s_mul_i32 s0, s18, s48 // (op 2560) 
s_add_u32 s0, s0, s1 // (op 2560) 


s_mul_hi_i32 s1, s16, s10 // (op 2560) 
s_add_i32 s18, s1, s16 // (op 2560) 

s_ashr_i32 s1, s18, 31 // (op 2560) 
s_cmp_eq_i32 s10, 0 // (op 2560) 
s_mov_b32 s19, scc // (op 2560) 
s_cmp_eq_i32 s19, 0 // (op 2560) 
s_cselect_b32 s20, 0, -1 // (op 2560) 

s_and_b32 s19, s44, 31 // (op 2560) 
s_lshl_b32 s21, 1, s19 // (op 2560) 
s_add_i32 s24, s21, s20 // (op 2560) 


s_and_b32 s25, s1, s24 // (op 2560) 


s_add_i32 s1, s18, s25 // (op 2560) 


s_ashr_i32 s18, s1, s19 // (op 2560) 


s_lshl_b32 s1, s44, 24 // (op 2560) 
s_ashr_i32 s19, s1, 31 // (op 2560) 

s_xor_b32 s1, s18, s19 // (op 2560) 

s_sub_i32 s18, s1, s19 // (op 2560) 


s_mul_i32 s1, s48, s6 // (op 2560) 
s_mul_i32 s19, s14, s1 // (op 2560) 

s_mul_i32 s1, s18, s19 // (op 2560) 

s_sub_u32 s19, s16, s1 // (op 2560) 

s_mov_b32 s1, s19 // (op 2560) 
s_mul_hi_i32 s20, s1, s45 // (op 2560) 
s_add_i32 s21, s20, s1 // (op 2560) 


s_ashr_i32 s1, s21, 31 // (op 2560) 
s_cmp_eq_i32 s45, 0 // (op 2560) 
s_mov_b32 s20, scc // (op 2560) 
s_cmp_eq_i32 s20, 0 // (op 2560) 
s_cselect_b32 s24, 0, -1 // (op 2560) 

s_and_b32 s20, s15, 31 // (op 2560) 
s_lshl_b32 s25, 1, s20 // (op 2560) 
s_add_i32 s49, s25, s24 // (op 2560) 


s_and_b32 s24, s1, s49 // (op 2560) 


s_add_i32 s1, s21, s24 // (op 2560) 


s_ashr_i32 s21, s1, s20 // (op 2560) 


s_lshl_b32 s1, s15, 24 // (op 2560) 
s_ashr_i32 s20, s1, 31 // (op 2560) 

s_xor_b32 s1, s21, s20 // (op 2560) 

s_sub_i32 s21, s1, s20 // (op 2560) 


s_mul_i32 s1, s46, s48 // (op 2560) 
s_add_u32 s24, s8, 32 // (op 2560) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 2560) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2560) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2560) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 2560) 

s_sub_i32 s20, s24, s1 // (op 2560) 

s_mul_i32 s24, s21, s20 // (op 2560) 


s_sub_u32 s25, s19, s24 // (op 2560) 


s_add_u32 s24, s25, s1 // (op 2560) 


s_cmp_eq_i32 s18, 0 // (op 2560) 
s_mov_b32 s1, scc // (op 2560) 

s_cmp_eq_i32 s1, 0 // (op 2560) 
s_cselect_b32 s25, s24, s0 // (op 2560) 



v_lshl_add_u32 v14, s25, 5, v12 // (op 2560) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2560) 
s_lshr_b32 s18, s0, 1 // (op 2560) 

v_lshl_add_u32 v12, s18, 8, v13 // (op 2560) 


v_mov_b32 v23, 0 // (op 2560) convert
v_mov_b32 v22, v12 // (op 2560) convert

v_mul_lo_u32 v13, v14, s27 // (op 2560) most significant: low of low * high
v_mul_hi_u32 v15, v14, s26 // (op 2560) most significant: high of low * low
v_mul_lo_u32 v12, v14, s26 // (op 2560) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 2560) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v22 // (op 2560) least significant half
v_addc_co_u32 v13, s[0:1], v13, v23, s[0:1] // (op 2560) most significant half



v_lshlrev_b64 v[14:15], 2, v[12:13] // (op 2560) 

v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 2560) 

v_mov_b32 v14, v12 // (op 2560) convert

s_mul_i32 s1, s26, s9 // (op 2562) most significant: low of low * high
s_mul_i32 s18, s27, s8 // (op 2562) most significant: low of high * low
s_mul_i32 s0, s26, s8 // (op 2562) least significant
s_mul_hi_u32 s19, s26, s8 // (op 2562) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2562) most significant: sum
s_add_u32 s1, s1, s19 // (op 2562) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 2562) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 2562) 

s_mov_b32 s52, s28 // (op 2562) reg expression
s_mov_b32 s53, s29 // (op 2562) reg expression
s_mov_b32 s54, s0 // (op 2562) reg expression
s_mov_b32 s55, 131072 // (op 2562) call()

s_mul_hi_i32 s0, s16, s47 // (op 2632) 
v_lshrrev_b32 v12, 2, v1 // (op 2632) 
s_add_i32 s1, s0, s16 // (op 2632) 

v_mul_lo_u32 v13, v12, 3 // (op 2632) 

s_ashr_i32 s0, s1, 31 // (op 2632) 
v_and_b32 v12, -4, v1 // (op 2632) 
s_cmp_eq_i32 s47, 0 // (op 2632) 
s_mov_b32 s18, scc // (op 2632) 
v_sub_u32 v15, v1, v12 // (op 2632) 

s_cmp_eq_i32 s18, 0 // (op 2632) 
s_cselect_b32 s19, 0, -1 // (op 2632) 

v_lshlrev_b32 v12, 5, v15 // (op 2632) 

s_and_b32 s18, s11, 31 // (op 2632) 
s_lshl_b32 s20, 1, s18 // (op 2632) 
s_add_i32 s21, s20, s19 // (op 2632) 


s_and_b32 s19, s0, s21 // (op 2632) 


s_add_i32 s0, s1, s19 // (op 2632) 


s_ashr_i32 s1, s0, s18 // (op 2632) 


s_lshl_b32 s0, s11, 24 // (op 2632) 
s_ashr_i32 s18, s0, 31 // (op 2632) 

s_xor_b32 s0, s1, s18 // (op 2632) 

s_sub_i32 s1, s0, s18 // (op 2632) 


s_mul_hi_i32 s0, s1, s12 // (op 2632) 
s_add_i32 s18, s0, s1 // (op 2632) 

s_ashr_i32 s0, s18, 31 // (op 2632) 
s_cmp_eq_i32 s12, 0 // (op 2632) 
s_mov_b32 s19, scc // (op 2632) 
s_cmp_eq_i32 s19, 0 // (op 2632) 
s_cselect_b32 s20, 0, -1 // (op 2632) 

s_and_b32 s19, s13, 31 // (op 2632) 
s_lshl_b32 s21, 1, s19 // (op 2632) 
s_add_i32 s24, s21, s20 // (op 2632) 


s_and_b32 s25, s0, s24 // (op 2632) 


s_add_i32 s0, s18, s25 // (op 2632) 


s_ashr_i32 s18, s0, s19 // (op 2632) 


s_lshl_b32 s0, s13, 24 // (op 2632) 
s_ashr_i32 s19, s0, 31 // (op 2632) 

s_xor_b32 s0, s18, s19 // (op 2632) 

s_sub_i32 s18, s0, s19 // (op 2632) 


s_mul_i32 s0, s18, s6 // (op 2632) 

s_sub_i32 s18, s1, s0 // (op 2632) 


s_mul_hi_i32 s0, s16, s10 // (op 2632) 
s_add_i32 s1, s0, s16 // (op 2632) 

s_ashr_i32 s0, s1, 31 // (op 2632) 
s_cmp_eq_i32 s10, 0 // (op 2632) 
s_mov_b32 s19, scc // (op 2632) 
s_cmp_eq_i32 s19, 0 // (op 2632) 
s_cselect_b32 s20, 0, -1 // (op 2632) 

s_and_b32 s19, s44, 31 // (op 2632) 
s_lshl_b32 s21, 1, s19 // (op 2632) 
s_add_i32 s24, s21, s20 // (op 2632) 


s_and_b32 s25, s0, s24 // (op 2632) 


s_add_i32 s0, s1, s25 // (op 2632) 


s_ashr_i32 s1, s0, s19 // (op 2632) 


s_lshl_b32 s0, s44, 24 // (op 2632) 
s_ashr_i32 s19, s0, 31 // (op 2632) 

s_xor_b32 s0, s1, s19 // (op 2632) 

s_sub_i32 s1, s0, s19 // (op 2632) 


s_mul_i32 s0, s48, s6 // (op 2632) 
s_mul_i32 s19, s14, s0 // (op 2632) 

s_mul_i32 s0, s1, s19 // (op 2632) 

s_sub_u32 s19, s16, s0 // (op 2632) 

s_mul_hi_i32 s0, s19, s45 // (op 2632) 
s_add_i32 s20, s0, s19 // (op 2632) 


s_ashr_i32 s0, s20, 31 // (op 2632) 
s_cmp_eq_i32 s45, 0 // (op 2632) 
s_mov_b32 s19, scc // (op 2632) 
s_cmp_eq_i32 s19, 0 // (op 2632) 
s_cselect_b32 s21, 0, -1 // (op 2632) 

s_and_b32 s19, s15, 31 // (op 2632) 
s_lshl_b32 s24, 1, s19 // (op 2632) 
s_add_i32 s25, s24, s21 // (op 2632) 


s_and_b32 s21, s0, s25 // (op 2632) 


s_add_i32 s0, s20, s21 // (op 2632) 


s_ashr_i32 s20, s0, s19 // (op 2632) 


s_lshl_b32 s0, s15, 24 // (op 2632) 
s_ashr_i32 s19, s0, 31 // (op 2632) 

s_xor_b32 s0, s20, s19 // (op 2632) 

s_sub_i32 s20, s0, s19 // (op 2632) 


s_cmp_eq_i32 s1, 0 // (op 2632) 
s_mov_b32 s0, scc // (op 2632) 

s_cmp_eq_i32 s0, 0 // (op 2632) 
s_cselect_b32 s1, s20, s18 // (op 2632) 



v_mov_b32 v15, 192 // (op 2632) 
v_mul_lo_u32 v16, s1, v15 // (op 2632) 

v_add_u32 v16, v16, v13 // (op 2632) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2632) 
v_mul_lo_u32 v23, v16, s31 // (op 2632) most significant: low of low * high
s_lshr_b32 s18, s0, 1 // (op 2632) 

v_mul_hi_u32 v13, v16, s30 // (op 2632) most significant: high of low * low
v_mul_lo_u32 v22, v16, s30 // (op 2632) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 2632) most significant: sum


v_lshl_add_u32 v13, s18, 8, v12 // (op 2632) 


v_mov_b32 v25, 0 // (op 2632) convert
v_mov_b32 v24, v13 // (op 2632) convert

v_add_co_u32 v12, s[0:1], v24, v22 // (op 2632) least significant half
v_addc_co_u32 v13, s[0:1], v25, v23, s[0:1] // (op 2632) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 2632) 


v_lshrrev_b64 v[22:23], 3, v[12:13] // (op 2632) 

v_mov_b32 v12, v22 // (op 2632) convert

s_mul_i32 s1, s30, s23 // (op 2634) most significant: low of low * high
s_mul_i32 s18, s31, s22 // (op 2634) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 2634) least significant
s_mul_hi_u32 s19, s30, s22 // (op 2634) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2634) most significant: sum
s_add_u32 s1, s1, s19 // (op 2634) most significant: sum


s_lshl_b64 s[24:25], s[0:1], 2 // (op 2634) 

s_ashr_i64 s[0:1], s[24:25], 3 // (op 2634) 

s_mov_b32 s64, s32 // (op 2634) reg expression
s_mov_b32 s65, s33 // (op 2634) reg expression
s_mov_b32 s66, s0 // (op 2634) reg expression
s_mov_b32 s67, 131072 // (op 2634) call()

s_mul_hi_i32 s0, s16, s47 // (op 2696) 
v_lshrrev_b32 v13, 7, v1 // (op 2696) 
s_add_i32 s1, s0, s16 // (op 2696) 

v_and_b32 v15, 15, v1 // (op 2696) 
s_ashr_i32 s0, s1, 31 // (op 2696) 
v_lshl_add_u32 v16, v13, 4, v15 // (op 2696) 


s_cmp_eq_i32 s47, 0 // (op 2696) 
s_mov_b32 s18, scc // (op 2696) 
v_and_b32 v13, 63, v1 // (op 2696) 
s_cmp_eq_i32 s18, 0 // (op 2696) 
s_cselect_b32 s19, 0, -1 // (op 2696) 

v_lshrrev_b32 v15, 4, v13 // (op 2696) 

s_and_b32 s18, s11, 31 // (op 2696) 
s_lshl_b32 s20, 1, s18 // (op 2696) 
s_add_i32 s21, s20, s19 // (op 2696) 


s_and_b32 s19, s0, s21 // (op 2696) 


s_add_i32 s0, s1, s19 // (op 2696) 


s_ashr_i32 s1, s0, s18 // (op 2696) 


s_lshl_b32 s0, s11, 24 // (op 2696) 
s_ashr_i32 s18, s0, 31 // (op 2696) 

s_xor_b32 s0, s1, s18 // (op 2696) 

s_sub_i32 s1, s0, s18 // (op 2696) 


s_mul_hi_i32 s0, s1, s12 // (op 2696) 
s_add_i32 s18, s0, s1 // (op 2696) 

s_ashr_i32 s0, s18, 31 // (op 2696) 
s_cmp_eq_i32 s12, 0 // (op 2696) 
s_mov_b32 s19, scc // (op 2696) 
s_cmp_eq_i32 s19, 0 // (op 2696) 
s_cselect_b32 s20, 0, -1 // (op 2696) 

s_and_b32 s19, s13, 31 // (op 2696) 
s_lshl_b32 s21, 1, s19 // (op 2696) 
s_add_i32 s24, s21, s20 // (op 2696) 


s_and_b32 s25, s0, s24 // (op 2696) 


s_add_i32 s0, s18, s25 // (op 2696) 


s_ashr_i32 s18, s0, s19 // (op 2696) 


s_lshl_b32 s0, s13, 24 // (op 2696) 
s_ashr_i32 s19, s0, 31 // (op 2696) 

s_xor_b32 s0, s18, s19 // (op 2696) 

s_sub_i32 s18, s0, s19 // (op 2696) 


s_mul_i32 s0, s1, s48 // (op 2696) 

s_sub_u32 s1, s16, s0 // (op 2696) 

s_mul_i32 s0, s18, s48 // (op 2696) 
s_add_u32 s0, s0, s1 // (op 2696) 


s_mul_hi_i32 s1, s16, s10 // (op 2696) 
s_add_i32 s18, s1, s16 // (op 2696) 

s_ashr_i32 s1, s18, 31 // (op 2696) 
s_cmp_eq_i32 s10, 0 // (op 2696) 
s_mov_b32 s19, scc // (op 2696) 
s_cmp_eq_i32 s19, 0 // (op 2696) 
s_cselect_b32 s20, 0, -1 // (op 2696) 

s_and_b32 s19, s44, 31 // (op 2696) 
s_lshl_b32 s21, 1, s19 // (op 2696) 
s_add_i32 s24, s21, s20 // (op 2696) 


s_and_b32 s25, s1, s24 // (op 2696) 


s_add_i32 s1, s18, s25 // (op 2696) 


s_ashr_i32 s18, s1, s19 // (op 2696) 


s_lshl_b32 s1, s44, 24 // (op 2696) 
s_ashr_i32 s19, s1, 31 // (op 2696) 

s_xor_b32 s1, s18, s19 // (op 2696) 

s_sub_i32 s18, s1, s19 // (op 2696) 


s_mul_i32 s1, s48, s6 // (op 2696) 
s_mul_i32 s19, s14, s1 // (op 2696) 

s_mul_i32 s1, s18, s19 // (op 2696) 

s_sub_u32 s19, s16, s1 // (op 2696) 

s_mov_b32 s1, s19 // (op 2696) 
s_mul_hi_i32 s20, s1, s45 // (op 2696) 
s_add_i32 s21, s20, s1 // (op 2696) 


s_ashr_i32 s1, s21, 31 // (op 2696) 
s_cmp_eq_i32 s45, 0 // (op 2696) 
s_mov_b32 s20, scc // (op 2696) 
s_cmp_eq_i32 s20, 0 // (op 2696) 
s_cselect_b32 s24, 0, -1 // (op 2696) 

s_and_b32 s20, s15, 31 // (op 2696) 
s_lshl_b32 s25, 1, s20 // (op 2696) 
s_add_i32 s49, s25, s24 // (op 2696) 


s_and_b32 s24, s1, s49 // (op 2696) 


s_add_i32 s1, s21, s24 // (op 2696) 


s_ashr_i32 s21, s1, s20 // (op 2696) 


s_lshl_b32 s1, s15, 24 // (op 2696) 
s_ashr_i32 s20, s1, 31 // (op 2696) 

s_xor_b32 s1, s21, s20 // (op 2696) 

s_sub_i32 s21, s1, s20 // (op 2696) 


s_mul_i32 s1, s46, s48 // (op 2696) 
s_add_u32 s24, s8, 32 // (op 2696) Start of Int64 add, locking SCC
s_addc_u32 s25, s9, 0 // (op 2696) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2696) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2696) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 2696) 

s_sub_i32 s20, s24, s1 // (op 2696) 

s_mul_i32 s24, s21, s20 // (op 2696) 


s_sub_u32 s25, s19, s24 // (op 2696) 


s_add_u32 s24, s25, s1 // (op 2696) 


s_cmp_eq_i32 s18, 0 // (op 2696) 
s_mov_b32 s1, scc // (op 2696) 

s_cmp_eq_i32 s1, 0 // (op 2696) 
s_cselect_b32 s25, s24, s0 // (op 2696) 



v_lshl_add_u32 v13, s25, 5, v16 // (op 2696) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2696) 
s_lshr_b32 s18, s0, 1 // (op 2696) 

v_lshl_add_u32 v16, s18, 3, v15 // (op 2696) 


v_mul_lo_u32 v15, v13, s34 // (op 2696) 
v_add_u32 v15, v15, v16 // (op 2696) 


s_mul_i32 s1, s34, s9 // (op 2698) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 2698) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2698) least significant
s_mul_hi_u32 s19, s34, s8 // (op 2698) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2698) most significant: sum
s_add_u32 s1, s1, s19 // (op 2698) most significant: sum


s_mov_b32 s68, s36 // (op 2698) reg expression
s_mov_b32 s69, s37 // (op 2698) reg expression
s_mov_b32 s70, s0 // (op 2698) reg expression
s_mov_b32 s71, 131072 // (op 2698) call()

s_mul_hi_i32 s0, s16, s47 // (op 2760) 
v_lshrrev_b32 v13, 2, v1 // (op 2760) 
s_add_i32 s1, s0, s16 // (op 2760) 

v_mul_lo_u32 v16, v13, 3 // (op 2760) 

s_ashr_i32 s0, s1, 31 // (op 2760) 
v_and_b32 v13, -4, v1 // (op 2760) 
s_cmp_eq_i32 s47, 0 // (op 2760) 
s_mov_b32 s18, scc // (op 2760) 
v_sub_u32 v17, v1, v13 // (op 2760) 

s_cmp_eq_i32 s18, 0 // (op 2760) 
s_cselect_b32 s19, 0, -1 // (op 2760) 

s_and_b32 s18, s11, 31 // (op 2760) 
s_lshl_b32 s20, 1, s18 // (op 2760) 
s_add_i32 s21, s20, s19 // (op 2760) 


s_and_b32 s19, s0, s21 // (op 2760) 


s_add_i32 s0, s1, s19 // (op 2760) 


s_ashr_i32 s1, s0, s18 // (op 2760) 


s_lshl_b32 s0, s11, 24 // (op 2760) 
s_ashr_i32 s18, s0, 31 // (op 2760) 

s_xor_b32 s0, s1, s18 // (op 2760) 

s_sub_i32 s1, s0, s18 // (op 2760) 


s_mul_hi_i32 s0, s1, s12 // (op 2760) 
s_add_i32 s18, s0, s1 // (op 2760) 

s_ashr_i32 s0, s18, 31 // (op 2760) 
s_cmp_eq_i32 s12, 0 // (op 2760) 
s_mov_b32 s19, scc // (op 2760) 
s_cmp_eq_i32 s19, 0 // (op 2760) 
s_cselect_b32 s20, 0, -1 // (op 2760) 

s_and_b32 s19, s13, 31 // (op 2760) 
s_lshl_b32 s21, 1, s19 // (op 2760) 
s_add_i32 s24, s21, s20 // (op 2760) 


s_and_b32 s25, s0, s24 // (op 2760) 


s_add_i32 s0, s18, s25 // (op 2760) 


s_ashr_i32 s18, s0, s19 // (op 2760) 


s_lshl_b32 s0, s13, 24 // (op 2760) 
s_ashr_i32 s19, s0, 31 // (op 2760) 

s_xor_b32 s0, s18, s19 // (op 2760) 

s_sub_i32 s18, s0, s19 // (op 2760) 


s_mul_i32 s0, s18, s6 // (op 2760) 

s_sub_i32 s18, s1, s0 // (op 2760) 


s_mul_hi_i32 s0, s16, s10 // (op 2760) 
s_add_i32 s1, s0, s16 // (op 2760) 

s_ashr_i32 s0, s1, 31 // (op 2760) 
s_cmp_eq_i32 s10, 0 // (op 2760) 
s_mov_b32 s19, scc // (op 2760) 
s_cmp_eq_i32 s19, 0 // (op 2760) 
s_cselect_b32 s20, 0, -1 // (op 2760) 

s_and_b32 s19, s44, 31 // (op 2760) 
s_lshl_b32 s21, 1, s19 // (op 2760) 
s_add_i32 s24, s21, s20 // (op 2760) 


s_and_b32 s25, s0, s24 // (op 2760) 


s_add_i32 s0, s1, s25 // (op 2760) 


s_ashr_i32 s1, s0, s19 // (op 2760) 


s_lshl_b32 s0, s44, 24 // (op 2760) 
s_ashr_i32 s19, s0, 31 // (op 2760) 

s_xor_b32 s0, s1, s19 // (op 2760) 

s_sub_i32 s1, s0, s19 // (op 2760) 


s_mul_i32 s0, s48, s6 // (op 2760) 
s_mul_i32 s19, s14, s0 // (op 2760) 

s_mul_i32 s0, s1, s19 // (op 2760) 

s_sub_u32 s19, s16, s0 // (op 2760) 

s_mul_hi_i32 s0, s19, s45 // (op 2760) 
s_add_i32 s20, s0, s19 // (op 2760) 


s_ashr_i32 s0, s20, 31 // (op 2760) 
s_cmp_eq_i32 s45, 0 // (op 2760) 
s_mov_b32 s19, scc // (op 2760) 
s_cmp_eq_i32 s19, 0 // (op 2760) 
s_cselect_b32 s21, 0, -1 // (op 2760) 

s_and_b32 s19, s15, 31 // (op 2760) 
s_lshl_b32 s24, 1, s19 // (op 2760) 
s_add_i32 s25, s24, s21 // (op 2760) 


s_and_b32 s21, s0, s25 // (op 2760) 


s_add_i32 s0, s20, s21 // (op 2760) 


s_ashr_i32 s20, s0, s19 // (op 2760) 


s_lshl_b32 s0, s15, 24 // (op 2760) 
s_ashr_i32 s19, s0, 31 // (op 2760) 

s_xor_b32 s0, s20, s19 // (op 2760) 

s_sub_i32 s20, s0, s19 // (op 2760) 


s_cmp_eq_i32 s1, 0 // (op 2760) 
s_mov_b32 s0, scc // (op 2760) 

s_cmp_eq_i32 s0, 0 // (op 2760) 
s_cselect_b32 s1, s20, s18 // (op 2760) 



v_mov_b32 v13, 192 // (op 2760) 
v_mul_lo_u32 v18, s1, v13 // (op 2760) 

v_add_u32 v18, v18, v16 // (op 2760) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2760) 
v_mul_lo_u32 v13, v18, s38 // (op 2760) 

s_lshr_b32 s18, s0, 1 // (op 2760) 

v_lshl_add_u32 v16, s18, 3, v17 // (op 2760) 


v_add_u32 v17, v16, v13 // (op 2760) 


s_mul_i32 s1, s38, s23 // (op 2762) most significant: low of low * high
s_mul_i32 s18, s39, s22 // (op 2762) most significant: low of high * low
s_mul_i32 s0, s38, s22 // (op 2762) least significant
s_mul_hi_u32 s19, s38, s22 // (op 2762) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2762) most significant: sum
s_add_u32 s1, s1, s19 // (op 2762) most significant: sum


s_mov_b32 s72, s40 // (op 2762) reg expression
s_mov_b32 s73, s41 // (op 2762) reg expression
s_mov_b32 s74, s0 // (op 2762) reg expression
s_mov_b32 s75, 131072 // (op 2762) call()

v_and_b32 v13, -8, v1 // (op 2853) 
v_sub_u32 v16, v1, v13 // (op 2853) 

v_lshlrev_b32 v13, 4, v16 // (op 2853) 

v_lshrrev_b32 v16, 3, v1 // (op 2853) 
v_lshl_add_u32 v18, v16, 7, v13 // (op 2853) 


v_lshlrev_b32 v13, 2, v18 // (op 2853) 

v_lshrrev_b32 v16, 3, v13 // (op 2853) 

v_and_b32 v13, -4, v1 // (op 2906) 
v_sub_u32 v18, v1, v13 // (op 2906) 

v_lshlrev_b32 v13, 5, v18 // (op 2906) 

v_lshrrev_b32 v18, 2, v1 // (op 2906) 
v_mul_lo_u32 v19, v18, 3 // (op 2906) 

v_lshl_add_u32 v18, v19, 7, v13 // (op 2906) 


v_lshlrev_b32 v13, 2, v18 // (op 2906) 

v_lshrrev_b32 v18, 3, v13 // (op 2906) 

v_and_b32 v13, -4, v1 // (op 2964) 
v_sub_u32 v19, v1, v13 // (op 2964) 

v_lshrrev_b32 v13, 2, v1 // (op 2964) 
v_mul_lo_u32 v20, v13, 3 // (op 2964) 

v_lshl_add_u32 v13, v20, 2, v19 // (op 2964) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 941) 
s_and_b32 s18, s0, 4294967294 // (op 941) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 940) 
s_mov_b32 s19, s0 // (op 940) convert

s_cmp_lt_u32 s18, s19 // (op 940) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopBottom_KLoopTail_940 // (op 940) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopBottom_KLoopTail_940 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopTop_KLoopTail_940:


s_add_u32 s18, s18, 1 // (op 942) 
v_mov_b32 v19, v14 // (op 170) 

buffer_load_dwordx2 v[20:21], v19, s[52:55], 0 offen // (op 170) Load value

v_add_u32 v14, 64, v14 // (op 2550) 

s_barrier  // (op 3156) 
s_lshl_b64 s[0:1], s[30:31], 2 // (op 183) 



v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 183) 


v_mov_b32 v19, v12 // (op 183) 
v_mov_b32 v24, v19 // (op 183) 
buffer_load_dwordx4 v[26:29], v24, s[64:67], 0 offen // (op 183) Load value
v_add_u32 v24, v24, v22 // (op 183) 
buffer_load_dwordx4 v[30:33], v24, s[64:67], 0 offen // (op 183) Load value
v_add_u32 v24, v24, v22 // (op 183) 
s_waitcnt vmcnt(2)
ds_write_b64 v16, v[20:21] offset:13056 // (op 171) Store local data 
buffer_load_dwordx4 v[34:37], v24, s[64:67], 0 offen // (op 183) Load value




v_add_u32 v12, 64, v12 // (op 2624) 

s_barrier  // (op 3169) 
v_mov_b64 v[24:25], s[38:39] // (op 184) convert




v_mov_b32 v19, v17 // (op 184) 
v_mov_b32 v20, v19 // (op 184) 
buffer_load_ubyte v38, v20, s[72:75], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v20, v20, v24 // (op 184) 
buffer_load_ubyte v39, v20, s[72:75], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v20, v20, v24 // (op 184) 
s_waitcnt vmcnt(4)
ds_write_b128 v18, v[26:29] // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v18, v[30:33] offset:64 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v18, v[34:37] offset:128 // (op 181) Store local data 
buffer_load_ubyte v40, v20, s[72:75], 0 offen offset: 0 // (op 184) Load value




v_add_u32 v17, 4, v17 // (op 2752) 
s_barrier  // (op 3183) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3159) 






ds_read_b128 v[34:37], v3 offset:13056 // (op 173) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v13, v38 offset:12288 // (op 182) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3173) 
s_waitcnt vmcnt(1)
ds_write_b8 v13, v39 offset:12292 // (op 182) Store local data 



ds_read_b128 v[30:33], v6 // (op 175) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v13, v40 offset:12296 // (op 182) Store local data 



v_mov_b32 v19, v15 // (op 177) 

buffer_load_ubyte v20, v19, s[68:71], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v15, 4, v15 // (op 2688) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3187) 



ds_read_u8 v19, v5 offset:12288 // (op 179) Load local data 



ds_read_b128 v[26:29], v6 offset:2048 // (op 324) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[30:33], a[0:3], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 180) 





ds_read_u8 v19, v5 offset:12416 // (op 327) Load local data 



ds_read_b128 v[22:25], v6 offset:4096 // (op 480) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[26:29], a[4:7], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 328) 





ds_read_u8 v19, v5 offset:12544 // (op 483) Load local data 



ds_read_b128 v[26:29], v6 offset:6144 // (op 636) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[22:25], a[8:11], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 484) 





ds_read_u8 v19, v5 offset:12672 // (op 639) Load local data 



ds_read_b128 v[22:25], v6 offset:8192 // (op 792) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[26:29], a[12:15], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 640) 





ds_read_u8 v19, v5 offset:12800 // (op 795) Load local data 



ds_read_b128 v[26:29], v6 offset:10240 // (op 948) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[34:37], v[22:25], a[16:19], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 796) 





ds_read_u8 v19, v5 offset:12928 // (op 951) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:37], v[26:29], a[20:23], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 952) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 940) 
s_mov_b32 s19, s0 // (op 940) convert

s_cmp_lt_u32 s18, s19 // (op 940) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopTop_KLoopTail_940 // (op 940) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopTop_KLoopTail_940 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ForLoopBottom_KLoopTail_940:





















s_barrier  // (op 3293) 




















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v7, v12 // (op 46) 
v_mul_f32 v17, v7, v13 // (op 46) 
v_mul_f32 v18, v7, v14 // (op 46) 
v_mul_f32 v19, v7, v15 // (op 46) 

v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 
v_cvt_f16_f32 v14, v18 // (op 52) 
v_cvt_f16_f32 v15, v19 // (op 52) 


v_pack_b32_f16 v8, v12, v13 // (op 54) 
v_pack_b32_f16 v9, v14, v15 // (op 54) 

v_mov_b32 v3, v10 // (op 54) reg expression

buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 54) Store value



v_accvgpr_read v12, a4 // (op 1220) 
v_accvgpr_read v13, a5 // (op 1220) 
v_accvgpr_read v14, a6 // (op 1220) 
v_accvgpr_read v15, a7 // (op 1220) 
v_mul_f32 v16, v7, v12 // (op 1220) 
v_mul_f32 v17, v7, v13 // (op 1220) 
v_mul_f32 v18, v7, v14 // (op 1220) 
v_mul_f32 v19, v7, v15 // (op 1220) 

v_cvt_f16_f32 v12, v16 // (op 1222) 
v_cvt_f16_f32 v13, v17 // (op 1222) 
v_cvt_f16_f32 v14, v18 // (op 1222) 
v_cvt_f16_f32 v15, v19 // (op 1222) 


v_pack_b32_f16 v8, v12, v13 // (op 1223) 
v_pack_b32_f16 v9, v14, v15 // (op 1223) 

s_lshl_b32 s0, s4, 6 // (op 1223) 
v_add_u32 v3, v10, s0 // (op 1223) 


buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 1223) Store value



v_accvgpr_read v12, a8 // (op 1234) 
v_accvgpr_read v13, a9 // (op 1234) 
v_accvgpr_read v14, a10 // (op 1234) 
v_accvgpr_read v15, a11 // (op 1234) 
v_mul_f32 v16, v7, v12 // (op 1234) 
v_mul_f32 v17, v7, v13 // (op 1234) 
v_mul_f32 v18, v7, v14 // (op 1234) 
v_mul_f32 v19, v7, v15 // (op 1234) 

v_cvt_f16_f32 v12, v16 // (op 1236) 
v_cvt_f16_f32 v13, v17 // (op 1236) 
v_cvt_f16_f32 v14, v18 // (op 1236) 
v_cvt_f16_f32 v15, v19 // (op 1236) 


v_pack_b32_f16 v8, v12, v13 // (op 1237) 
v_pack_b32_f16 v9, v14, v15 // (op 1237) 

s_lshl_b32 s0, s4, 7 // (op 1237) 
v_add_u32 v3, v10, s0 // (op 1237) 


buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 1237) Store value



v_accvgpr_read v12, a12 // (op 1248) 
v_accvgpr_read v13, a13 // (op 1248) 
v_accvgpr_read v14, a14 // (op 1248) 
v_accvgpr_read v15, a15 // (op 1248) 
v_mul_f32 v16, v7, v12 // (op 1248) 
v_mul_f32 v17, v7, v13 // (op 1248) 
v_mul_f32 v18, v7, v14 // (op 1248) 
v_mul_f32 v19, v7, v15 // (op 1248) 

v_cvt_f16_f32 v12, v16 // (op 1250) 
v_cvt_f16_f32 v13, v17 // (op 1250) 
v_cvt_f16_f32 v14, v18 // (op 1250) 
v_cvt_f16_f32 v15, v19 // (op 1250) 


v_pack_b32_f16 v8, v12, v13 // (op 1251) 
v_pack_b32_f16 v9, v14, v15 // (op 1251) 

s_lshl_b32 s0, s4, 6 // (op 1251) 
s_mul_i32 s1, 3, s0 // (op 1251) 

v_add_u32 v3, v10, s1 // (op 1251) 


buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 1251) Store value



v_accvgpr_read v12, a16 // (op 1262) 
v_accvgpr_read v13, a17 // (op 1262) 
v_accvgpr_read v14, a18 // (op 1262) 
v_accvgpr_read v15, a19 // (op 1262) 
v_mul_f32 v16, v7, v12 // (op 1262) 
v_mul_f32 v17, v7, v13 // (op 1262) 
v_mul_f32 v18, v7, v14 // (op 1262) 
v_mul_f32 v19, v7, v15 // (op 1262) 

v_cvt_f16_f32 v12, v16 // (op 1264) 
v_cvt_f16_f32 v13, v17 // (op 1264) 
v_cvt_f16_f32 v14, v18 // (op 1264) 
v_cvt_f16_f32 v15, v19 // (op 1264) 


v_pack_b32_f16 v8, v12, v13 // (op 1265) 
v_pack_b32_f16 v9, v14, v15 // (op 1265) 

s_lshl_b32 s0, s4, 8 // (op 1265) 
v_add_u32 v3, v10, s0 // (op 1265) 


buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 1265) Store value



v_accvgpr_read v12, a20 // (op 1276) 
v_accvgpr_read v13, a21 // (op 1276) 
v_accvgpr_read v14, a22 // (op 1276) 
v_accvgpr_read v15, a23 // (op 1276) 
v_mul_f32 v16, v7, v12 // (op 1276) 
v_mul_f32 v17, v7, v13 // (op 1276) 
v_mul_f32 v18, v7, v14 // (op 1276) 
v_mul_f32 v19, v7, v15 // (op 1276) 

v_cvt_f16_f32 v12, v16 // (op 1278) 
v_cvt_f16_f32 v13, v17 // (op 1278) 
v_cvt_f16_f32 v14, v18 // (op 1278) 
v_cvt_f16_f32 v15, v19 // (op 1278) 


v_pack_b32_f16 v8, v12, v13 // (op 1279) 
v_pack_b32_f16 v9, v14, v15 // (op 1279) 

s_lshl_b32 s0, s4, 6 // (op 1279) 
s_mul_i32 s1, 5, s0 // (op 1279) 

v_add_u32 v3, v10, s1 // (op 1279) 


buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 1279) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ConditionalFalse_0_beta_125:


v_mov_b32 v3, v11 // (op 138) reg expression

buffer_load_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 138) Load value

v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v7, v12 // (op 134) 
v_mul_f32 v17, v7, v13 // (op 134) 
v_mul_f32 v18, v7, v14 // (op 134) 
v_mul_f32 v19, v7, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 139) 
v_lshrrev_b32 v13, 16, v8 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v0, v12 // (op 139) 
v_mul_f32 v21, v0, v13 // (op 139) 
v_cvt_f32_f16 v12, v9 // (op 139) 
v_lshrrev_b32 v13, 16, v9 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v22, v0, v12 // (op 139) 
v_mul_f32 v23, v0, v13 // (op 139) 


v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 
v_add_f32 v14, v22, v18 // (op 135) 
v_add_f32 v15, v23, v19 // (op 135) 


v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 
v_cvt_f16_f32 v18, v14 // (op 136) 
v_cvt_f16_f32 v19, v15 // (op 136) 


v_pack_b32_f16 v8, v16, v17 // (op 137) 
v_pack_b32_f16 v9, v18, v19 // (op 137) 

v_mov_b32 v3, v10 // (op 137) reg expression

buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 1323) 
v_add_u32 v3, v11, s0 // (op 1323) 


buffer_load_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 1323) Load value

v_accvgpr_read v12, a4 // (op 1319) 
v_accvgpr_read v13, a5 // (op 1319) 
v_accvgpr_read v14, a6 // (op 1319) 
v_accvgpr_read v15, a7 // (op 1319) 
v_mul_f32 v16, v7, v12 // (op 1319) 
v_mul_f32 v17, v7, v13 // (op 1319) 
v_mul_f32 v18, v7, v14 // (op 1319) 
v_mul_f32 v19, v7, v15 // (op 1319) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 1324) 
v_lshrrev_b32 v13, 16, v8 // (op 1324) 
v_cvt_f32_f16 v13, v13 // (op 1324) 
v_mul_f32 v20, v0, v12 // (op 1324) 
v_mul_f32 v21, v0, v13 // (op 1324) 
v_cvt_f32_f16 v12, v9 // (op 1324) 
v_lshrrev_b32 v13, 16, v9 // (op 1324) 
v_cvt_f32_f16 v13, v13 // (op 1324) 
v_mul_f32 v22, v0, v12 // (op 1324) 
v_mul_f32 v23, v0, v13 // (op 1324) 


v_add_f32 v12, v20, v16 // (op 1320) 
v_add_f32 v13, v21, v17 // (op 1320) 
v_add_f32 v14, v22, v18 // (op 1320) 
v_add_f32 v15, v23, v19 // (op 1320) 


v_cvt_f16_f32 v16, v12 // (op 1321) 
v_cvt_f16_f32 v17, v13 // (op 1321) 
v_cvt_f16_f32 v18, v14 // (op 1321) 
v_cvt_f16_f32 v19, v15 // (op 1321) 


v_pack_b32_f16 v8, v16, v17 // (op 1322) 
v_pack_b32_f16 v9, v18, v19 // (op 1322) 

s_lshl_b32 s0, s4, 6 // (op 1322) 
v_add_u32 v3, v10, s0 // (op 1322) 


buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 1322) Store value





s_lshl_b32 s0, s2, 7 // (op 1337) 
v_add_u32 v3, v11, s0 // (op 1337) 


buffer_load_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 1337) Load value

v_accvgpr_read v12, a8 // (op 1333) 
v_accvgpr_read v13, a9 // (op 1333) 
v_accvgpr_read v14, a10 // (op 1333) 
v_accvgpr_read v15, a11 // (op 1333) 
v_mul_f32 v16, v7, v12 // (op 1333) 
v_mul_f32 v17, v7, v13 // (op 1333) 
v_mul_f32 v18, v7, v14 // (op 1333) 
v_mul_f32 v19, v7, v15 // (op 1333) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 1338) 
v_lshrrev_b32 v13, 16, v8 // (op 1338) 
v_cvt_f32_f16 v13, v13 // (op 1338) 
v_mul_f32 v20, v0, v12 // (op 1338) 
v_mul_f32 v21, v0, v13 // (op 1338) 
v_cvt_f32_f16 v12, v9 // (op 1338) 
v_lshrrev_b32 v13, 16, v9 // (op 1338) 
v_cvt_f32_f16 v13, v13 // (op 1338) 
v_mul_f32 v22, v0, v12 // (op 1338) 
v_mul_f32 v23, v0, v13 // (op 1338) 


v_add_f32 v12, v20, v16 // (op 1334) 
v_add_f32 v13, v21, v17 // (op 1334) 
v_add_f32 v14, v22, v18 // (op 1334) 
v_add_f32 v15, v23, v19 // (op 1334) 


v_cvt_f16_f32 v16, v12 // (op 1335) 
v_cvt_f16_f32 v17, v13 // (op 1335) 
v_cvt_f16_f32 v18, v14 // (op 1335) 
v_cvt_f16_f32 v19, v15 // (op 1335) 


v_pack_b32_f16 v8, v16, v17 // (op 1336) 
v_pack_b32_f16 v9, v18, v19 // (op 1336) 

s_lshl_b32 s0, s4, 7 // (op 1336) 
v_add_u32 v3, v10, s0 // (op 1336) 


buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 1336) Store value





s_lshl_b32 s0, s2, 6 // (op 1351) 
s_mul_i32 s1, 3, s0 // (op 1351) 

v_add_u32 v3, v11, s1 // (op 1351) 


buffer_load_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 1351) Load value

v_accvgpr_read v12, a12 // (op 1347) 
v_accvgpr_read v13, a13 // (op 1347) 
v_accvgpr_read v14, a14 // (op 1347) 
v_accvgpr_read v15, a15 // (op 1347) 
v_mul_f32 v16, v7, v12 // (op 1347) 
v_mul_f32 v17, v7, v13 // (op 1347) 
v_mul_f32 v18, v7, v14 // (op 1347) 
v_mul_f32 v19, v7, v15 // (op 1347) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 1352) 
v_lshrrev_b32 v13, 16, v8 // (op 1352) 
v_cvt_f32_f16 v13, v13 // (op 1352) 
v_mul_f32 v20, v0, v12 // (op 1352) 
v_mul_f32 v21, v0, v13 // (op 1352) 
v_cvt_f32_f16 v12, v9 // (op 1352) 
v_lshrrev_b32 v13, 16, v9 // (op 1352) 
v_cvt_f32_f16 v13, v13 // (op 1352) 
v_mul_f32 v22, v0, v12 // (op 1352) 
v_mul_f32 v23, v0, v13 // (op 1352) 


v_add_f32 v12, v20, v16 // (op 1348) 
v_add_f32 v13, v21, v17 // (op 1348) 
v_add_f32 v14, v22, v18 // (op 1348) 
v_add_f32 v15, v23, v19 // (op 1348) 


v_cvt_f16_f32 v16, v12 // (op 1349) 
v_cvt_f16_f32 v17, v13 // (op 1349) 
v_cvt_f16_f32 v18, v14 // (op 1349) 
v_cvt_f16_f32 v19, v15 // (op 1349) 


v_pack_b32_f16 v8, v16, v17 // (op 1350) 
v_pack_b32_f16 v9, v18, v19 // (op 1350) 

s_lshl_b32 s0, s4, 6 // (op 1350) 
s_mul_i32 s1, 3, s0 // (op 1350) 

v_add_u32 v3, v10, s1 // (op 1350) 


buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 1350) Store value





s_lshl_b32 s0, s2, 8 // (op 1365) 
v_add_u32 v3, v11, s0 // (op 1365) 


buffer_load_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 1365) Load value

v_accvgpr_read v12, a16 // (op 1361) 
v_accvgpr_read v13, a17 // (op 1361) 
v_accvgpr_read v14, a18 // (op 1361) 
v_accvgpr_read v15, a19 // (op 1361) 
v_mul_f32 v16, v7, v12 // (op 1361) 
v_mul_f32 v17, v7, v13 // (op 1361) 
v_mul_f32 v18, v7, v14 // (op 1361) 
v_mul_f32 v19, v7, v15 // (op 1361) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 1366) 
v_lshrrev_b32 v13, 16, v8 // (op 1366) 
v_cvt_f32_f16 v13, v13 // (op 1366) 
v_mul_f32 v20, v0, v12 // (op 1366) 
v_mul_f32 v21, v0, v13 // (op 1366) 
v_cvt_f32_f16 v12, v9 // (op 1366) 
v_lshrrev_b32 v13, 16, v9 // (op 1366) 
v_cvt_f32_f16 v13, v13 // (op 1366) 
v_mul_f32 v22, v0, v12 // (op 1366) 
v_mul_f32 v23, v0, v13 // (op 1366) 


v_add_f32 v12, v20, v16 // (op 1362) 
v_add_f32 v13, v21, v17 // (op 1362) 
v_add_f32 v14, v22, v18 // (op 1362) 
v_add_f32 v15, v23, v19 // (op 1362) 


v_cvt_f16_f32 v16, v12 // (op 1363) 
v_cvt_f16_f32 v17, v13 // (op 1363) 
v_cvt_f16_f32 v18, v14 // (op 1363) 
v_cvt_f16_f32 v19, v15 // (op 1363) 


v_pack_b32_f16 v8, v16, v17 // (op 1364) 
v_pack_b32_f16 v9, v18, v19 // (op 1364) 

s_lshl_b32 s0, s4, 8 // (op 1364) 
v_add_u32 v3, v10, s0 // (op 1364) 


buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 1364) Store value





s_lshl_b32 s0, s2, 6 // (op 1379) 
s_mul_i32 s1, 5, s0 // (op 1379) 

v_add_u32 v3, v11, s1 // (op 1379) 


buffer_load_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 1379) Load value

v_accvgpr_read v12, a20 // (op 1375) 
v_accvgpr_read v13, a21 // (op 1375) 
v_accvgpr_read v14, a22 // (op 1375) 
v_accvgpr_read v15, a23 // (op 1375) 
v_mul_f32 v16, v7, v12 // (op 1375) 
v_mul_f32 v17, v7, v13 // (op 1375) 
v_mul_f32 v18, v7, v14 // (op 1375) 
v_mul_f32 v19, v7, v15 // (op 1375) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 1380) 
v_lshrrev_b32 v13, 16, v8 // (op 1380) 
v_cvt_f32_f16 v13, v13 // (op 1380) 
v_mul_f32 v20, v0, v12 // (op 1380) 
v_mul_f32 v21, v0, v13 // (op 1380) 
v_cvt_f32_f16 v12, v9 // (op 1380) 
v_lshrrev_b32 v13, 16, v9 // (op 1380) 
v_cvt_f32_f16 v13, v13 // (op 1380) 
v_mul_f32 v22, v0, v12 // (op 1380) 
v_mul_f32 v23, v0, v13 // (op 1380) 


v_add_f32 v12, v20, v16 // (op 1376) 
v_add_f32 v13, v21, v17 // (op 1376) 
v_add_f32 v14, v22, v18 // (op 1376) 
v_add_f32 v15, v23, v19 // (op 1376) 


v_cvt_f16_f32 v16, v12 // (op 1377) 
v_cvt_f16_f32 v17, v13 // (op 1377) 
v_cvt_f16_f32 v18, v14 // (op 1377) 
v_cvt_f16_f32 v19, v15 // (op 1377) 


v_pack_b32_f16 v8, v16, v17 // (op 1378) 
v_pack_b32_f16 v9, v18, v19 // (op 1378) 

s_lshl_b32 s0, s4, 6 // (op 1378) 
s_mul_i32 s1, 5, s0 // (op 1378) 

v_add_u32 v3, v10, s1 // (op 1378) 


buffer_store_dwordx2 v[8:9], v3, s[60:63], 0 offen // (op 1378) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__ConditionalBottom_0_beta_125:














s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM_
.LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM_, .LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM__end-RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM_
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
            { type: int64, semantic: StrideD0 },
            { type: int64, semantic: RR_NumTilesN },
            { type: int64, semantic: SizeFree0 },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
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
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [32, 192, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 6]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_WGM_.kd
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
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 32
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 156, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}}
        .variableType: {dataType: Int32, pointerType: Value}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 44
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_22
        .size: 4
        .offset: 48
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_25, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 168, size: 4}, rhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 152
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_20, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 148, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_21, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}, offset: 152, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 156, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}, rhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 156
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_25, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 168, size: 4}, rhs: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_25
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_23, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 160, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_24, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 144, size: 4}}, offset: 164, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
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

