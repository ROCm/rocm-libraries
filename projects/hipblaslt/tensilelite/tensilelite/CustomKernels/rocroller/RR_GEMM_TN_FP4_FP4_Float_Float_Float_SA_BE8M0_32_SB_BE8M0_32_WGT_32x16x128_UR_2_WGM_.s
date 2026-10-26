

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 739) 
v_and_b32 v3, 15, v1 // (op 739) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 739) 


v_and_b32 v0, 63, v1 // (op 739) 
v_lshrrev_b32 v3, 4, v0 // (op 739) 

v_lshlrev_b32 v0, 5, v3 // (op 739) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 739) 


v_lshlrev_b32 v0, 2, v3 // (op 739) 

v_lshrrev_b32 v3, 3, v0 // (op 739) 

v_and_b32 v0, 15, v1 // (op 792) 
v_and_b32 v4, 63, v1 // (op 792) 
v_lshrrev_b32 v5, 4, v4 // (op 792) 

v_lshlrev_b32 v4, 5, v5 // (op 792) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 792) 


v_lshlrev_b32 v0, 2, v5 // (op 792) 

v_lshrrev_b32 v4, 3, v0 // (op 792) 

v_lshrrev_b32 v0, 6, v1 // (op 845) 
v_and_b32 v5, 15, v1 // (op 845) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 845) 


v_and_b32 v0, 63, v1 // (op 845) 
v_lshrrev_b32 v5, 4, v0 // (op 845) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 845) 


v_lshrrev_b32 v5, 6, v1 // (op 903) 
v_and_b32 v6, 15, v1 // (op 903) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 903) 


v_and_b32 v5, 63, v1 // (op 903) 
v_lshrrev_b32 v6, 4, v5 // (op 903) 

v_lshlrev_b32 v5, 5, v6 // (op 903) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 903) 


v_lshlrev_b32 v5, 2, v6 // (op 903) 

v_lshrrev_b32 v6, 3, v5 // (op 903) 

v_and_b32 v5, 15, v1 // (op 927) 
v_and_b32 v7, 63, v1 // (op 927) 
v_lshrrev_b32 v8, 4, v7 // (op 927) 

v_lshlrev_b32 v7, 5, v8 // (op 927) 

v_lshl_add_u32 v8, v5, 7, v7 // (op 927) 


v_lshlrev_b32 v5, 2, v8 // (op 927) 

v_lshrrev_b32 v7, 3, v5 // (op 927) 

v_lshrrev_b32 v5, 6, v1 // (op 951) 
v_and_b32 v8, 15, v1 // (op 951) 
v_lshl_add_u32 v9, v5, 4, v8 // (op 951) 


v_and_b32 v5, 63, v1 // (op 951) 
v_lshrrev_b32 v8, 4, v5 // (op 951) 

v_lshl_add_u32 v5, v9, 2, v8 // (op 951) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
s_mul_hi_i32 s0, s16, s12 // (op 449) 
v_lshrrev_b32 v10, 2, v1 // (op 449) 
s_add_i32 s1, s0, s16 // (op 449) 

v_and_b32 v11, -4, v1 // (op 449) 
s_ashr_i32 s0, s1, 31 // (op 449) 
v_sub_u32 v12, v1, v11 // (op 449) 

s_cmp_eq_i32 s12, 0 // (op 449) 
s_mov_b32 s8, scc // (op 449) 
v_lshlrev_b32 v11, 5, v12 // (op 449) 

s_cmp_eq_i32 s8, 0 // (op 449) 
s_cselect_b32 s9, 0, -1 // (op 449) 

v_mov_b32 v13, 0 // (op 449) convert
v_mov_b32 v12, v11 // (op 449) convert

s_and_b32 s8, s13, 31 // (op 449) 
s_lshl_b32 s18, 1, s8 // (op 449) 
s_add_i32 s19, s18, s9 // (op 449) 


s_and_b32 s9, s0, s19 // (op 449) 


s_add_i32 s0, s1, s9 // (op 449) 


s_ashr_i32 s1, s0, s8 // (op 449) 


s_lshl_b32 s0, s13, 24 // (op 449) 
s_ashr_i32 s8, s0, 31 // (op 449) 

s_xor_b32 s0, s1, s8 // (op 449) 

s_sub_i32 s1, s0, s8 // (op 449) 


s_mul_hi_i32 s0, s1, s14 // (op 449) 
s_add_i32 s8, s0, s1 // (op 449) 

s_ashr_i32 s0, s8, 31 // (op 449) 
s_cmp_eq_i32 s14, 0 // (op 449) 
s_mov_b32 s9, scc // (op 449) 
s_cmp_eq_i32 s9, 0 // (op 449) 
s_cselect_b32 s18, 0, -1 // (op 449) 

s_and_b32 s9, s15, 31 // (op 449) 
s_lshl_b32 s19, 1, s9 // (op 449) 
s_add_i32 s49, s19, s18 // (op 449) 


s_and_b32 s18, s0, s49 // (op 449) 


s_add_i32 s0, s8, s18 // (op 449) 


s_ashr_i32 s8, s0, s9 // (op 449) 


s_lshl_b32 s0, s15, 24 // (op 449) 
s_ashr_i32 s9, s0, 31 // (op 449) 

s_xor_b32 s0, s8, s9 // (op 449) 

s_sub_i32 s8, s0, s9 // (op 449) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 449) 

s_sub_u32 s1, s16, s0 // (op 449) 

s_mul_i32 s0, s8, s45 // (op 449) 
s_add_u32 s0, s0, s1 // (op 449) 


s_mul_hi_i32 s1, s16, s11 // (op 449) 
s_add_i32 s8, s1, s16 // (op 449) 

s_ashr_i32 s1, s8, 31 // (op 449) 
s_cmp_eq_i32 s11, 0 // (op 449) 
s_mov_b32 s9, scc // (op 449) 
s_cmp_eq_i32 s9, 0 // (op 449) 
s_cselect_b32 s18, 0, -1 // (op 449) 

s_and_b32 s9, s46, 31 // (op 449) 
s_lshl_b32 s19, 1, s9 // (op 449) 
s_add_i32 s49, s19, s18 // (op 449) 


s_and_b32 s18, s1, s49 // (op 449) 


s_add_i32 s1, s8, s18 // (op 449) 


s_ashr_i32 s8, s1, s9 // (op 449) 


s_lshl_b32 s1, s46, 24 // (op 449) 
s_ashr_i32 s9, s1, 31 // (op 449) 

s_xor_b32 s1, s8, s9 // (op 449) 

s_sub_i32 s8, s1, s9 // (op 449) 


s_add_u32 s18, s4, 16 // (op 449) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 449) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 449) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 449) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 449) 

s_mul_i32 s1, s45, s18 // (op 449) 

s_mul_i32 s9, s10, s1 // (op 449) 

s_mul_i32 s1, s8, s9 // (op 449) 

s_sub_u32 s9, s16, s1 // (op 449) 

s_mov_b32 s1, s9 // (op 449) 
s_mul_hi_i32 s18, s1, s44 // (op 449) 
s_add_i32 s19, s18, s1 // (op 449) 


s_ashr_i32 s1, s19, 31 // (op 449) 
s_cmp_eq_i32 s44, 0 // (op 449) 
s_mov_b32 s18, scc // (op 449) 
s_cmp_eq_i32 s18, 0 // (op 449) 
s_cselect_b32 s49, 0, -1 // (op 449) 

s_and_b32 s18, s48, 31 // (op 449) 
s_lshl_b32 s50, 1, s18 // (op 449) 
s_add_i32 s51, s50, s49 // (op 449) 


s_and_b32 s49, s1, s51 // (op 449) 


s_add_i32 s1, s19, s49 // (op 449) 


s_ashr_i32 s19, s1, s18 // (op 449) 


s_lshl_b32 s1, s48, 24 // (op 449) 
s_ashr_i32 s18, s1, 31 // (op 449) 

s_xor_b32 s1, s19, s18 // (op 449) 

s_sub_i32 s19, s1, s18 // (op 449) 


s_mul_i32 s1, s47, s45 // (op 449) 
s_add_u32 s50, s6, 32 // (op 449) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 449) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 449) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 449) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 5 // (op 449) 

s_sub_i32 s18, s50, s1 // (op 449) 

s_mul_i32 s49, s19, s18 // (op 449) 


s_sub_u32 s18, s9, s49 // (op 449) 


s_add_u32 s9, s18, s1 // (op 449) 


s_cmp_eq_i32 s8, 0 // (op 449) 
s_mov_b32 s1, scc // (op 449) 

s_cmp_eq_i32 s1, 0 // (op 449) 
s_cselect_b32 s8, s9, s0 // (op 449) 



v_lshl_add_u32 v11, s8, 5, v10 // (op 449) 


v_mul_lo_u32 v15, v11, s3 // (op 449) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 449) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 449) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 449) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 449) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 449) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 449) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 449) 

v_mov_b32 v10, v12 // (op 449) convert

s_mul_i32 s1, s2, s7 // (op 451) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 451) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 451) least significant
s_mul_hi_u32 s9, s2, s6 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 451) most significant: sum
s_add_u32 s1, s1, s9 // (op 451) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 451) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 451) 

s_mov_b32 s52, s20 // (op 451) reg expression
s_mov_b32 s53, s21 // (op 451) reg expression
s_mov_b32 s54, s0 // (op 451) reg expression
s_mov_b32 s55, 131072 // (op 451) call()

s_mul_hi_i32 s0, s16, s12 // (op 468) 
v_lshrrev_b32 v11, 2, v1 // (op 468) 
s_add_i32 s1, s0, s16 // (op 468) 

v_and_b32 v12, -4, v1 // (op 468) 
s_ashr_i32 s0, s1, 31 // (op 468) 
v_sub_u32 v13, v1, v12 // (op 468) 

s_cmp_eq_i32 s12, 0 // (op 468) 
s_mov_b32 s8, scc // (op 468) 
v_lshlrev_b32 v12, 5, v13 // (op 468) 

s_cmp_eq_i32 s8, 0 // (op 468) 
s_cselect_b32 s9, 0, -1 // (op 468) 

v_mov_b32 v15, 0 // (op 468) convert
v_mov_b32 v14, v12 // (op 468) convert

s_and_b32 s8, s13, 31 // (op 468) 
s_lshl_b32 s18, 1, s8 // (op 468) 
s_add_i32 s19, s18, s9 // (op 468) 


s_and_b32 s9, s0, s19 // (op 468) 


s_add_i32 s0, s1, s9 // (op 468) 


s_ashr_i32 s1, s0, s8 // (op 468) 


s_lshl_b32 s0, s13, 24 // (op 468) 
s_ashr_i32 s8, s0, 31 // (op 468) 

s_xor_b32 s0, s1, s8 // (op 468) 

s_sub_i32 s1, s0, s8 // (op 468) 


s_mul_hi_i32 s0, s1, s14 // (op 468) 
s_add_i32 s8, s0, s1 // (op 468) 

s_ashr_i32 s0, s8, 31 // (op 468) 
s_cmp_eq_i32 s14, 0 // (op 468) 
s_mov_b32 s9, scc // (op 468) 
s_cmp_eq_i32 s9, 0 // (op 468) 
s_cselect_b32 s18, 0, -1 // (op 468) 

s_and_b32 s9, s15, 31 // (op 468) 
s_lshl_b32 s19, 1, s9 // (op 468) 
s_add_i32 s49, s19, s18 // (op 468) 


s_and_b32 s18, s0, s49 // (op 468) 


s_add_i32 s0, s8, s18 // (op 468) 


s_ashr_i32 s8, s0, s9 // (op 468) 


s_lshl_b32 s0, s15, 24 // (op 468) 
s_ashr_i32 s9, s0, 31 // (op 468) 

s_xor_b32 s0, s8, s9 // (op 468) 

s_sub_i32 s8, s0, s9 // (op 468) 


s_mul_i32 s0, s1, s45 // (op 468) 

s_sub_u32 s1, s16, s0 // (op 468) 

s_mul_i32 s0, s8, s45 // (op 468) 
s_add_u32 s0, s0, s1 // (op 468) 


s_mul_hi_i32 s1, s16, s11 // (op 468) 
s_add_i32 s8, s1, s16 // (op 468) 

s_ashr_i32 s1, s8, 31 // (op 468) 
s_cmp_eq_i32 s11, 0 // (op 468) 
s_mov_b32 s9, scc // (op 468) 
s_cmp_eq_i32 s9, 0 // (op 468) 
s_cselect_b32 s18, 0, -1 // (op 468) 

s_and_b32 s9, s46, 31 // (op 468) 
s_lshl_b32 s19, 1, s9 // (op 468) 
s_add_i32 s49, s19, s18 // (op 468) 


s_and_b32 s18, s1, s49 // (op 468) 


s_add_i32 s1, s8, s18 // (op 468) 


s_ashr_i32 s8, s1, s9 // (op 468) 


s_lshl_b32 s1, s46, 24 // (op 468) 
s_ashr_i32 s9, s1, 31 // (op 468) 

s_xor_b32 s1, s8, s9 // (op 468) 

s_sub_i32 s8, s1, s9 // (op 468) 


s_add_u32 s18, s4, 16 // (op 468) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 468) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 468) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 468) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 468) 

s_mul_i32 s1, s45, s18 // (op 468) 

s_mul_i32 s9, s10, s1 // (op 468) 

s_mul_i32 s1, s8, s9 // (op 468) 

s_sub_u32 s9, s16, s1 // (op 468) 

s_mov_b32 s1, s9 // (op 468) 
s_mul_hi_i32 s18, s1, s44 // (op 468) 
s_add_i32 s19, s18, s1 // (op 468) 


s_ashr_i32 s1, s19, 31 // (op 468) 
s_cmp_eq_i32 s44, 0 // (op 468) 
s_mov_b32 s18, scc // (op 468) 
s_cmp_eq_i32 s18, 0 // (op 468) 
s_cselect_b32 s49, 0, -1 // (op 468) 

s_and_b32 s18, s48, 31 // (op 468) 
s_lshl_b32 s50, 1, s18 // (op 468) 
s_add_i32 s51, s50, s49 // (op 468) 


s_and_b32 s49, s1, s51 // (op 468) 


s_add_i32 s1, s19, s49 // (op 468) 


s_ashr_i32 s19, s1, s18 // (op 468) 


s_lshl_b32 s1, s48, 24 // (op 468) 
s_ashr_i32 s18, s1, 31 // (op 468) 

s_xor_b32 s1, s19, s18 // (op 468) 

s_sub_i32 s19, s1, s18 // (op 468) 


s_mul_i32 s1, s47, s45 // (op 468) 
s_add_u32 s50, s6, 32 // (op 468) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 468) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 468) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 468) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 5 // (op 468) 

s_sub_i32 s18, s50, s1 // (op 468) 

s_mul_i32 s49, s19, s18 // (op 468) 


s_sub_u32 s18, s9, s49 // (op 468) 


s_add_u32 s9, s18, s1 // (op 468) 


s_cmp_eq_i32 s8, 0 // (op 468) 
s_mov_b32 s1, scc // (op 468) 

s_cmp_eq_i32 s1, 0 // (op 468) 
s_cselect_b32 s8, s9, s0 // (op 468) 



v_lshl_add_u32 v12, s8, 5, v11 // (op 468) 


v_mul_lo_u32 v17, v12, s3 // (op 468) most significant: low of low * high
v_mul_hi_u32 v11, v12, s2 // (op 468) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 468) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 468) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 468) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 468) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 468) 

v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 468) 

v_mov_b32 v11, v12 // (op 468) convert

s_mul_i32 s1, s2, s7 // (op 470) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 470) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 470) least significant
s_mul_hi_u32 s9, s2, s6 // (op 470) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 470) most significant: sum
s_add_u32 s1, s1, s9 // (op 470) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 470) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 470) 

s_mov_b32 s52, s20 // (op 470) reg expression
s_mov_b32 s53, s21 // (op 470) reg expression
s_mov_b32 s54, s0 // (op 470) reg expression
s_mov_b32 s55, 131072 // (op 470) call()

s_mul_hi_i32 s0, s16, s12 // (op 513) 
v_lshrrev_b32 v12, 3, v1 // (op 513) 
s_add_i32 s1, s0, s16 // (op 513) 

v_and_b32 v13, -8, v1 // (op 513) 
s_ashr_i32 s0, s1, 31 // (op 513) 
v_sub_u32 v14, v1, v13 // (op 513) 

s_cmp_eq_i32 s12, 0 // (op 513) 
s_mov_b32 s8, scc // (op 513) 
v_lshlrev_b32 v13, 4, v14 // (op 513) 

s_cmp_eq_i32 s8, 0 // (op 513) 
s_cselect_b32 s9, 0, -1 // (op 513) 

v_mov_b32 v15, 0 // (op 513) convert
v_mov_b32 v14, v13 // (op 513) convert

s_and_b32 s8, s13, 31 // (op 513) 
s_lshl_b32 s18, 1, s8 // (op 513) 
s_add_i32 s19, s18, s9 // (op 513) 


s_and_b32 s9, s0, s19 // (op 513) 


s_add_i32 s0, s1, s9 // (op 513) 


s_ashr_i32 s1, s0, s8 // (op 513) 


s_lshl_b32 s0, s13, 24 // (op 513) 
s_ashr_i32 s8, s0, 31 // (op 513) 

s_xor_b32 s0, s1, s8 // (op 513) 

s_sub_i32 s1, s0, s8 // (op 513) 


s_mul_hi_i32 s0, s1, s14 // (op 513) 
s_add_i32 s8, s0, s1 // (op 513) 

s_ashr_i32 s0, s8, 31 // (op 513) 
s_cmp_eq_i32 s14, 0 // (op 513) 
s_mov_b32 s9, scc // (op 513) 
s_cmp_eq_i32 s9, 0 // (op 513) 
s_cselect_b32 s18, 0, -1 // (op 513) 

s_and_b32 s9, s15, 31 // (op 513) 
s_lshl_b32 s19, 1, s9 // (op 513) 
s_add_i32 s49, s19, s18 // (op 513) 


s_and_b32 s18, s0, s49 // (op 513) 


s_add_i32 s0, s8, s18 // (op 513) 


s_ashr_i32 s8, s0, s9 // (op 513) 


s_lshl_b32 s0, s15, 24 // (op 513) 
s_ashr_i32 s9, s0, 31 // (op 513) 

s_xor_b32 s0, s8, s9 // (op 513) 

s_sub_i32 s8, s0, s9 // (op 513) 


s_add_u32 s18, s4, 16 // (op 513) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 513) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 513) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 513) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 513) 

s_mul_i32 s0, s8, s18 // (op 513) 

s_sub_i32 s8, s1, s0 // (op 513) 


s_mul_hi_i32 s9, s16, s11 // (op 513) 
s_add_i32 s0, s9, s16 // (op 513) 

s_ashr_i32 s1, s0, 31 // (op 513) 
s_cmp_eq_i32 s11, 0 // (op 513) 
s_mov_b32 s9, scc // (op 513) 
s_cmp_eq_i32 s9, 0 // (op 513) 
s_cselect_b32 s49, 0, -1 // (op 513) 

s_and_b32 s9, s46, 31 // (op 513) 
s_lshl_b32 s50, 1, s9 // (op 513) 
s_add_i32 s51, s50, s49 // (op 513) 


s_and_b32 s49, s1, s51 // (op 513) 


s_add_i32 s1, s0, s49 // (op 513) 


s_ashr_i32 s0, s1, s9 // (op 513) 


s_lshl_b32 s1, s46, 24 // (op 513) 
s_ashr_i32 s9, s1, 31 // (op 513) 

s_xor_b32 s1, s0, s9 // (op 513) 

s_sub_i32 s0, s1, s9 // (op 513) 


s_mul_i32 s1, s45, s18 // (op 513) 

s_mul_i32 s9, s10, s1 // (op 513) 

s_mul_i32 s1, s0, s9 // (op 513) 

s_sub_u32 s9, s16, s1 // (op 513) 

s_mul_hi_i32 s1, s9, s44 // (op 513) 
s_add_i32 s18, s1, s9 // (op 513) 


s_ashr_i32 s1, s18, 31 // (op 513) 
s_cmp_eq_i32 s44, 0 // (op 513) 
s_mov_b32 s9, scc // (op 513) 
s_cmp_eq_i32 s9, 0 // (op 513) 
s_cselect_b32 s19, 0, -1 // (op 513) 

s_and_b32 s9, s48, 31 // (op 513) 
s_lshl_b32 s49, 1, s9 // (op 513) 
s_add_i32 s50, s49, s19 // (op 513) 


s_and_b32 s19, s1, s50 // (op 513) 


s_add_i32 s1, s18, s19 // (op 513) 


s_ashr_i32 s18, s1, s9 // (op 513) 


s_lshl_b32 s1, s48, 24 // (op 513) 
s_ashr_i32 s9, s1, 31 // (op 513) 

s_xor_b32 s1, s18, s9 // (op 513) 

s_sub_i32 s18, s1, s9 // (op 513) 


s_cmp_eq_i32 s0, 0 // (op 513) 
s_mov_b32 s1, scc // (op 513) 

s_cmp_eq_i32 s1, 0 // (op 513) 
s_cselect_b32 s0, s18, s8 // (op 513) 



v_lshl_add_u32 v13, s0, 4, v12 // (op 513) 


v_mul_lo_u32 v17, v13, s23 // (op 513) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 513) most significant: high of low * low
v_mul_lo_u32 v16, v13, s22 // (op 513) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 513) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 513) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 513) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 513) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 513) 

v_mov_b32 v12, v14 // (op 513) convert

s_mul_i32 s1, s22, s5 // (op 515) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 515) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 515) least significant
s_mul_hi_u32 s9, s22, s4 // (op 515) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 515) most significant: sum
s_add_u32 s1, s1, s9 // (op 515) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 515) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 515) 

s_mov_b32 s56, s24 // (op 515) reg expression
s_mov_b32 s57, s25 // (op 515) reg expression
s_mov_b32 s58, s0 // (op 515) reg expression
s_mov_b32 s59, 131072 // (op 515) call()

s_mul_hi_i32 s0, s16, s12 // (op 532) 
v_lshrrev_b32 v13, 3, v1 // (op 532) 
s_add_i32 s1, s0, s16 // (op 532) 

v_and_b32 v14, -8, v1 // (op 532) 
s_ashr_i32 s0, s1, 31 // (op 532) 
v_sub_u32 v15, v1, v14 // (op 532) 

s_cmp_eq_i32 s12, 0 // (op 532) 
s_mov_b32 s8, scc // (op 532) 
v_lshlrev_b32 v14, 4, v15 // (op 532) 

s_cmp_eq_i32 s8, 0 // (op 532) 
s_cselect_b32 s9, 0, -1 // (op 532) 

v_mov_b32 v17, 0 // (op 532) convert
v_mov_b32 v16, v14 // (op 532) convert

s_and_b32 s8, s13, 31 // (op 532) 
s_lshl_b32 s18, 1, s8 // (op 532) 
s_add_i32 s19, s18, s9 // (op 532) 


s_and_b32 s9, s0, s19 // (op 532) 


s_add_i32 s0, s1, s9 // (op 532) 


s_ashr_i32 s1, s0, s8 // (op 532) 


s_lshl_b32 s0, s13, 24 // (op 532) 
s_ashr_i32 s8, s0, 31 // (op 532) 

s_xor_b32 s0, s1, s8 // (op 532) 

s_sub_i32 s1, s0, s8 // (op 532) 


s_mul_hi_i32 s0, s1, s14 // (op 532) 
s_add_i32 s8, s0, s1 // (op 532) 

s_ashr_i32 s0, s8, 31 // (op 532) 
s_cmp_eq_i32 s14, 0 // (op 532) 
s_mov_b32 s9, scc // (op 532) 
s_cmp_eq_i32 s9, 0 // (op 532) 
s_cselect_b32 s18, 0, -1 // (op 532) 

s_and_b32 s9, s15, 31 // (op 532) 
s_lshl_b32 s19, 1, s9 // (op 532) 
s_add_i32 s49, s19, s18 // (op 532) 


s_and_b32 s18, s0, s49 // (op 532) 


s_add_i32 s0, s8, s18 // (op 532) 


s_ashr_i32 s8, s0, s9 // (op 532) 


s_lshl_b32 s0, s15, 24 // (op 532) 
s_ashr_i32 s9, s0, 31 // (op 532) 

s_xor_b32 s0, s8, s9 // (op 532) 

s_sub_i32 s8, s0, s9 // (op 532) 


s_add_u32 s18, s4, 16 // (op 532) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 532) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 532) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 532) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 532) 

s_mul_i32 s0, s8, s18 // (op 532) 

s_sub_i32 s8, s1, s0 // (op 532) 


s_mul_hi_i32 s9, s16, s11 // (op 532) 
s_add_i32 s0, s9, s16 // (op 532) 

s_ashr_i32 s1, s0, 31 // (op 532) 
s_cmp_eq_i32 s11, 0 // (op 532) 
s_mov_b32 s9, scc // (op 532) 
s_cmp_eq_i32 s9, 0 // (op 532) 
s_cselect_b32 s49, 0, -1 // (op 532) 

s_and_b32 s9, s46, 31 // (op 532) 
s_lshl_b32 s50, 1, s9 // (op 532) 
s_add_i32 s51, s50, s49 // (op 532) 


s_and_b32 s49, s1, s51 // (op 532) 


s_add_i32 s1, s0, s49 // (op 532) 


s_ashr_i32 s0, s1, s9 // (op 532) 


s_lshl_b32 s1, s46, 24 // (op 532) 
s_ashr_i32 s9, s1, 31 // (op 532) 

s_xor_b32 s1, s0, s9 // (op 532) 

s_sub_i32 s0, s1, s9 // (op 532) 


s_mul_i32 s1, s45, s18 // (op 532) 

s_mul_i32 s9, s10, s1 // (op 532) 

s_mul_i32 s1, s0, s9 // (op 532) 

s_sub_u32 s9, s16, s1 // (op 532) 

s_mul_hi_i32 s1, s9, s44 // (op 532) 
s_add_i32 s18, s1, s9 // (op 532) 


s_ashr_i32 s1, s18, 31 // (op 532) 
s_cmp_eq_i32 s44, 0 // (op 532) 
s_mov_b32 s9, scc // (op 532) 
s_cmp_eq_i32 s9, 0 // (op 532) 
s_cselect_b32 s19, 0, -1 // (op 532) 

s_and_b32 s9, s48, 31 // (op 532) 
s_lshl_b32 s49, 1, s9 // (op 532) 
s_add_i32 s50, s49, s19 // (op 532) 


s_and_b32 s19, s1, s50 // (op 532) 


s_add_i32 s1, s18, s19 // (op 532) 


s_ashr_i32 s18, s1, s9 // (op 532) 


s_lshl_b32 s1, s48, 24 // (op 532) 
s_ashr_i32 s9, s1, 31 // (op 532) 

s_xor_b32 s1, s18, s9 // (op 532) 

s_sub_i32 s18, s1, s9 // (op 532) 


s_cmp_eq_i32 s0, 0 // (op 532) 
s_mov_b32 s1, scc // (op 532) 

s_cmp_eq_i32 s1, 0 // (op 532) 
s_cselect_b32 s0, s18, s8 // (op 532) 



v_lshl_add_u32 v14, s0, 4, v13 // (op 532) 


v_mul_lo_u32 v19, v14, s23 // (op 532) most significant: low of low * high
v_mul_hi_u32 v13, v14, s22 // (op 532) most significant: high of low * low
v_mul_lo_u32 v18, v14, s22 // (op 532) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 532) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 532) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 532) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 532) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 532) 

v_mov_b32 v13, v16 // (op 532) convert

s_mul_i32 s1, s22, s5 // (op 534) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 534) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 534) least significant
s_mul_hi_u32 s9, s22, s4 // (op 534) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 534) most significant: sum
s_add_u32 s1, s1, s9 // (op 534) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 534) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 534) 

s_mov_b32 s56, s24 // (op 534) reg expression
s_mov_b32 s57, s25 // (op 534) reg expression
s_mov_b32 s58, s0 // (op 534) reg expression
s_mov_b32 s59, 131072 // (op 534) call()

s_mul_hi_i32 s0, s16, s12 // (op 577) 
v_lshrrev_b32 v14, 2, v1 // (op 577) 
s_add_i32 s1, s0, s16 // (op 577) 

v_and_b32 v15, -4, v1 // (op 577) 
s_ashr_i32 s0, s1, 31 // (op 577) 
v_sub_u32 v16, v1, v15 // (op 577) 

s_cmp_eq_i32 s12, 0 // (op 577) 
s_mov_b32 s8, scc // (op 577) 
s_cmp_eq_i32 s8, 0 // (op 577) 
s_cselect_b32 s9, 0, -1 // (op 577) 

s_and_b32 s8, s13, 31 // (op 577) 
s_lshl_b32 s18, 1, s8 // (op 577) 
s_add_i32 s19, s18, s9 // (op 577) 


s_and_b32 s9, s0, s19 // (op 577) 


s_add_i32 s0, s1, s9 // (op 577) 


s_ashr_i32 s1, s0, s8 // (op 577) 


s_lshl_b32 s0, s13, 24 // (op 577) 
s_ashr_i32 s8, s0, 31 // (op 577) 

s_xor_b32 s0, s1, s8 // (op 577) 

s_sub_i32 s1, s0, s8 // (op 577) 


s_mul_hi_i32 s0, s1, s14 // (op 577) 
s_add_i32 s8, s0, s1 // (op 577) 

s_ashr_i32 s0, s8, 31 // (op 577) 
s_cmp_eq_i32 s14, 0 // (op 577) 
s_mov_b32 s9, scc // (op 577) 
s_cmp_eq_i32 s9, 0 // (op 577) 
s_cselect_b32 s18, 0, -1 // (op 577) 

s_and_b32 s9, s15, 31 // (op 577) 
s_lshl_b32 s19, 1, s9 // (op 577) 
s_add_i32 s49, s19, s18 // (op 577) 


s_and_b32 s18, s0, s49 // (op 577) 


s_add_i32 s0, s8, s18 // (op 577) 


s_ashr_i32 s8, s0, s9 // (op 577) 


s_lshl_b32 s0, s15, 24 // (op 577) 
s_ashr_i32 s9, s0, 31 // (op 577) 

s_xor_b32 s0, s8, s9 // (op 577) 

s_sub_i32 s8, s0, s9 // (op 577) 


s_mul_i32 s0, s1, s45 // (op 577) 

s_sub_u32 s1, s16, s0 // (op 577) 

s_mul_i32 s0, s8, s45 // (op 577) 
s_add_u32 s0, s0, s1 // (op 577) 


s_mul_hi_i32 s1, s16, s11 // (op 577) 
s_add_i32 s8, s1, s16 // (op 577) 

s_ashr_i32 s1, s8, 31 // (op 577) 
s_cmp_eq_i32 s11, 0 // (op 577) 
s_mov_b32 s9, scc // (op 577) 
s_cmp_eq_i32 s9, 0 // (op 577) 
s_cselect_b32 s18, 0, -1 // (op 577) 

s_and_b32 s9, s46, 31 // (op 577) 
s_lshl_b32 s19, 1, s9 // (op 577) 
s_add_i32 s49, s19, s18 // (op 577) 


s_and_b32 s18, s1, s49 // (op 577) 


s_add_i32 s1, s8, s18 // (op 577) 


s_ashr_i32 s8, s1, s9 // (op 577) 


s_lshl_b32 s1, s46, 24 // (op 577) 
s_ashr_i32 s9, s1, 31 // (op 577) 

s_xor_b32 s1, s8, s9 // (op 577) 

s_sub_i32 s8, s1, s9 // (op 577) 


s_add_u32 s18, s4, 16 // (op 577) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 577) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 577) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 577) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 577) 

s_mul_i32 s1, s45, s18 // (op 577) 

s_mul_i32 s9, s10, s1 // (op 577) 

s_mul_i32 s1, s8, s9 // (op 577) 

s_sub_u32 s9, s16, s1 // (op 577) 

s_mov_b32 s1, s9 // (op 577) 
s_mul_hi_i32 s18, s1, s44 // (op 577) 
s_add_i32 s19, s18, s1 // (op 577) 


s_ashr_i32 s1, s19, 31 // (op 577) 
s_cmp_eq_i32 s44, 0 // (op 577) 
s_mov_b32 s18, scc // (op 577) 
s_cmp_eq_i32 s18, 0 // (op 577) 
s_cselect_b32 s49, 0, -1 // (op 577) 

s_and_b32 s18, s48, 31 // (op 577) 
s_lshl_b32 s50, 1, s18 // (op 577) 
s_add_i32 s51, s50, s49 // (op 577) 


s_and_b32 s49, s1, s51 // (op 577) 


s_add_i32 s1, s19, s49 // (op 577) 


s_ashr_i32 s19, s1, s18 // (op 577) 


s_lshl_b32 s1, s48, 24 // (op 577) 
s_ashr_i32 s18, s1, 31 // (op 577) 

s_xor_b32 s1, s19, s18 // (op 577) 

s_sub_i32 s19, s1, s18 // (op 577) 


s_mul_i32 s1, s47, s45 // (op 577) 
s_add_u32 s50, s6, 32 // (op 577) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 577) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 577) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 577) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 5 // (op 577) 

s_sub_i32 s18, s50, s1 // (op 577) 

s_mul_i32 s49, s19, s18 // (op 577) 


s_sub_u32 s18, s9, s49 // (op 577) 


s_add_u32 s9, s18, s1 // (op 577) 


s_cmp_eq_i32 s8, 0 // (op 577) 
s_mov_b32 s1, scc // (op 577) 

s_cmp_eq_i32 s1, 0 // (op 577) 
s_cselect_b32 s8, s9, s0 // (op 577) 



v_lshl_add_u32 v15, s8, 5, v14 // (op 577) 


v_mul_lo_u32 v14, v15, s26 // (op 577) 
v_add_u32 v14, v14, v16 // (op 577) 


s_mul_i32 s1, s26, s7 // (op 579) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 579) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 579) least significant
s_mul_hi_u32 s9, s26, s6 // (op 579) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 579) most significant: sum
s_add_u32 s1, s1, s9 // (op 579) most significant: sum


s_mov_b32 s60, s28 // (op 579) reg expression
s_mov_b32 s61, s29 // (op 579) reg expression
s_mov_b32 s62, s0 // (op 579) reg expression
s_mov_b32 s63, 131072 // (op 579) call()

s_mul_hi_i32 s0, s16, s12 // (op 596) 
v_lshrrev_b32 v15, 2, v1 // (op 596) 
s_add_i32 s1, s0, s16 // (op 596) 

v_and_b32 v16, -4, v1 // (op 596) 
s_ashr_i32 s0, s1, 31 // (op 596) 
v_sub_u32 v17, v1, v16 // (op 596) 

s_cmp_eq_i32 s12, 0 // (op 596) 
s_mov_b32 s8, scc // (op 596) 
s_cmp_eq_i32 s8, 0 // (op 596) 
s_cselect_b32 s9, 0, -1 // (op 596) 

s_and_b32 s8, s13, 31 // (op 596) 
s_lshl_b32 s18, 1, s8 // (op 596) 
s_add_i32 s19, s18, s9 // (op 596) 


s_and_b32 s9, s0, s19 // (op 596) 


s_add_i32 s0, s1, s9 // (op 596) 


s_ashr_i32 s1, s0, s8 // (op 596) 


s_lshl_b32 s0, s13, 24 // (op 596) 
s_ashr_i32 s8, s0, 31 // (op 596) 

s_xor_b32 s0, s1, s8 // (op 596) 

s_sub_i32 s1, s0, s8 // (op 596) 


s_mul_hi_i32 s0, s1, s14 // (op 596) 
s_add_i32 s8, s0, s1 // (op 596) 

s_ashr_i32 s0, s8, 31 // (op 596) 
s_cmp_eq_i32 s14, 0 // (op 596) 
s_mov_b32 s9, scc // (op 596) 
s_cmp_eq_i32 s9, 0 // (op 596) 
s_cselect_b32 s18, 0, -1 // (op 596) 

s_and_b32 s9, s15, 31 // (op 596) 
s_lshl_b32 s19, 1, s9 // (op 596) 
s_add_i32 s49, s19, s18 // (op 596) 


s_and_b32 s18, s0, s49 // (op 596) 


s_add_i32 s0, s8, s18 // (op 596) 


s_ashr_i32 s8, s0, s9 // (op 596) 


s_lshl_b32 s0, s15, 24 // (op 596) 
s_ashr_i32 s9, s0, 31 // (op 596) 

s_xor_b32 s0, s8, s9 // (op 596) 

s_sub_i32 s8, s0, s9 // (op 596) 


s_mul_i32 s0, s1, s45 // (op 596) 

s_sub_u32 s1, s16, s0 // (op 596) 

s_mul_i32 s0, s8, s45 // (op 596) 
s_add_u32 s0, s0, s1 // (op 596) 


s_mul_hi_i32 s1, s16, s11 // (op 596) 
s_add_i32 s8, s1, s16 // (op 596) 

s_ashr_i32 s1, s8, 31 // (op 596) 
s_cmp_eq_i32 s11, 0 // (op 596) 
s_mov_b32 s9, scc // (op 596) 
s_cmp_eq_i32 s9, 0 // (op 596) 
s_cselect_b32 s18, 0, -1 // (op 596) 

s_and_b32 s9, s46, 31 // (op 596) 
s_lshl_b32 s19, 1, s9 // (op 596) 
s_add_i32 s49, s19, s18 // (op 596) 


s_and_b32 s18, s1, s49 // (op 596) 


s_add_i32 s1, s8, s18 // (op 596) 


s_ashr_i32 s8, s1, s9 // (op 596) 


s_lshl_b32 s1, s46, 24 // (op 596) 
s_ashr_i32 s9, s1, 31 // (op 596) 

s_xor_b32 s1, s8, s9 // (op 596) 

s_sub_i32 s8, s1, s9 // (op 596) 


s_add_u32 s18, s4, 16 // (op 596) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 596) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 596) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 596) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 596) 

s_mul_i32 s1, s45, s18 // (op 596) 

s_mul_i32 s9, s10, s1 // (op 596) 

s_mul_i32 s1, s8, s9 // (op 596) 

s_sub_u32 s9, s16, s1 // (op 596) 

s_mov_b32 s1, s9 // (op 596) 
s_mul_hi_i32 s18, s1, s44 // (op 596) 
s_add_i32 s19, s18, s1 // (op 596) 


s_ashr_i32 s1, s19, 31 // (op 596) 
s_cmp_eq_i32 s44, 0 // (op 596) 
s_mov_b32 s18, scc // (op 596) 
s_cmp_eq_i32 s18, 0 // (op 596) 
s_cselect_b32 s49, 0, -1 // (op 596) 

s_and_b32 s18, s48, 31 // (op 596) 
s_lshl_b32 s50, 1, s18 // (op 596) 
s_add_i32 s51, s50, s49 // (op 596) 


s_and_b32 s49, s1, s51 // (op 596) 


s_add_i32 s1, s19, s49 // (op 596) 


s_ashr_i32 s19, s1, s18 // (op 596) 


s_lshl_b32 s1, s48, 24 // (op 596) 
s_ashr_i32 s18, s1, 31 // (op 596) 

s_xor_b32 s1, s19, s18 // (op 596) 

s_sub_i32 s19, s1, s18 // (op 596) 


s_mul_i32 s1, s47, s45 // (op 596) 
s_add_u32 s50, s6, 32 // (op 596) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 596) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 596) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 596) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 5 // (op 596) 

s_sub_i32 s18, s50, s1 // (op 596) 

s_mul_i32 s49, s19, s18 // (op 596) 


s_sub_u32 s18, s9, s49 // (op 596) 


s_add_u32 s9, s18, s1 // (op 596) 


s_cmp_eq_i32 s8, 0 // (op 596) 
s_mov_b32 s1, scc // (op 596) 

s_cmp_eq_i32 s1, 0 // (op 596) 
s_cselect_b32 s8, s9, s0 // (op 596) 



v_lshl_add_u32 v16, s8, 5, v15 // (op 596) 


v_mul_lo_u32 v15, v16, s26 // (op 596) 
v_add_u32 v15, v15, v17 // (op 596) 


s_mul_i32 s1, s26, s7 // (op 598) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 598) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 598) least significant
s_mul_hi_u32 s9, s26, s6 // (op 598) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 598) most significant: sum
s_add_u32 s1, s1, s9 // (op 598) most significant: sum


s_mov_b32 s60, s28 // (op 598) reg expression
s_mov_b32 s61, s29 // (op 598) reg expression
s_mov_b32 s62, s0 // (op 598) reg expression
s_mov_b32 s63, 131072 // (op 598) call()

s_mul_hi_i32 s0, s16, s12 // (op 641) 
v_and_b32 v16, 15, v1 // (op 641) 
s_add_i32 s1, s0, s16 // (op 641) 

v_and_b32 v17, 63, v1 // (op 641) 
s_ashr_i32 s0, s1, 31 // (op 641) 
v_lshrrev_b32 v18, 4, v17 // (op 641) 

s_cmp_eq_i32 s12, 0 // (op 641) 
s_mov_b32 s8, scc // (op 641) 
s_cmp_eq_i32 s8, 0 // (op 641) 
s_cselect_b32 s9, 0, -1 // (op 641) 

s_and_b32 s8, s13, 31 // (op 641) 
s_lshl_b32 s18, 1, s8 // (op 641) 
s_add_i32 s19, s18, s9 // (op 641) 


s_and_b32 s9, s0, s19 // (op 641) 


s_add_i32 s0, s1, s9 // (op 641) 


s_ashr_i32 s1, s0, s8 // (op 641) 


s_lshl_b32 s0, s13, 24 // (op 641) 
s_ashr_i32 s8, s0, 31 // (op 641) 

s_xor_b32 s0, s1, s8 // (op 641) 

s_sub_i32 s1, s0, s8 // (op 641) 


s_mul_hi_i32 s0, s1, s14 // (op 641) 
s_add_i32 s8, s0, s1 // (op 641) 

s_ashr_i32 s0, s8, 31 // (op 641) 
s_cmp_eq_i32 s14, 0 // (op 641) 
s_mov_b32 s9, scc // (op 641) 
s_cmp_eq_i32 s9, 0 // (op 641) 
s_cselect_b32 s18, 0, -1 // (op 641) 

s_and_b32 s9, s15, 31 // (op 641) 
s_lshl_b32 s19, 1, s9 // (op 641) 
s_add_i32 s49, s19, s18 // (op 641) 


s_and_b32 s18, s0, s49 // (op 641) 


s_add_i32 s0, s8, s18 // (op 641) 


s_ashr_i32 s8, s0, s9 // (op 641) 


s_lshl_b32 s0, s15, 24 // (op 641) 
s_ashr_i32 s9, s0, 31 // (op 641) 

s_xor_b32 s0, s8, s9 // (op 641) 

s_sub_i32 s8, s0, s9 // (op 641) 


s_add_u32 s18, s4, 16 // (op 641) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 641) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 641) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 641) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 641) 

s_mul_i32 s0, s8, s18 // (op 641) 

s_sub_i32 s8, s1, s0 // (op 641) 


s_mul_hi_i32 s9, s16, s11 // (op 641) 
s_add_i32 s0, s9, s16 // (op 641) 

s_ashr_i32 s1, s0, 31 // (op 641) 
s_cmp_eq_i32 s11, 0 // (op 641) 
s_mov_b32 s9, scc // (op 641) 
s_cmp_eq_i32 s9, 0 // (op 641) 
s_cselect_b32 s49, 0, -1 // (op 641) 

s_and_b32 s9, s46, 31 // (op 641) 
s_lshl_b32 s50, 1, s9 // (op 641) 
s_add_i32 s51, s50, s49 // (op 641) 


s_and_b32 s49, s1, s51 // (op 641) 


s_add_i32 s1, s0, s49 // (op 641) 


s_ashr_i32 s0, s1, s9 // (op 641) 


s_lshl_b32 s1, s46, 24 // (op 641) 
s_ashr_i32 s9, s1, 31 // (op 641) 

s_xor_b32 s1, s0, s9 // (op 641) 

s_sub_i32 s0, s1, s9 // (op 641) 


s_mul_i32 s1, s45, s18 // (op 641) 

s_mul_i32 s9, s10, s1 // (op 641) 

s_mul_i32 s1, s0, s9 // (op 641) 

s_sub_u32 s9, s16, s1 // (op 641) 

s_mul_hi_i32 s1, s9, s44 // (op 641) 
s_add_i32 s18, s1, s9 // (op 641) 


s_ashr_i32 s1, s18, 31 // (op 641) 
s_cmp_eq_i32 s44, 0 // (op 641) 
s_mov_b32 s9, scc // (op 641) 
s_cmp_eq_i32 s9, 0 // (op 641) 
s_cselect_b32 s19, 0, -1 // (op 641) 

s_and_b32 s9, s48, 31 // (op 641) 
s_lshl_b32 s49, 1, s9 // (op 641) 
s_add_i32 s50, s49, s19 // (op 641) 


s_and_b32 s19, s1, s50 // (op 641) 


s_add_i32 s1, s18, s19 // (op 641) 


s_ashr_i32 s18, s1, s9 // (op 641) 


s_lshl_b32 s1, s48, 24 // (op 641) 
s_ashr_i32 s9, s1, 31 // (op 641) 

s_xor_b32 s1, s18, s9 // (op 641) 

s_sub_i32 s18, s1, s9 // (op 641) 


s_cmp_eq_i32 s0, 0 // (op 641) 
s_mov_b32 s1, scc // (op 641) 

s_cmp_eq_i32 s1, 0 // (op 641) 
s_cselect_b32 s0, s18, s8 // (op 641) 



v_lshl_add_u32 v17, s0, 4, v16 // (op 641) 


v_mul_lo_u32 v16, v17, s30 // (op 641) 

v_add_u32 v17, v18, v16 // (op 641) 


s_mul_i32 s1, s30, s5 // (op 643) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 643) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 643) least significant
s_mul_hi_u32 s9, s30, s4 // (op 643) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 643) most significant: sum
s_add_u32 s1, s1, s9 // (op 643) most significant: sum


s_mov_b32 s64, s32 // (op 643) reg expression
s_mov_b32 s65, s33 // (op 643) reg expression
s_mov_b32 s66, s0 // (op 643) reg expression
s_mov_b32 s67, 131072 // (op 643) call()

s_mul_hi_i32 s0, s16, s12 // (op 660) 
v_and_b32 v16, 15, v1 // (op 660) 
s_add_i32 s1, s0, s16 // (op 660) 

v_and_b32 v18, 63, v1 // (op 660) 
s_ashr_i32 s0, s1, 31 // (op 660) 
v_lshrrev_b32 v19, 4, v18 // (op 660) 

s_cmp_eq_i32 s12, 0 // (op 660) 
s_mov_b32 s8, scc // (op 660) 
s_cmp_eq_i32 s8, 0 // (op 660) 
s_cselect_b32 s9, 0, -1 // (op 660) 

s_and_b32 s8, s13, 31 // (op 660) 
s_lshl_b32 s18, 1, s8 // (op 660) 
s_add_i32 s19, s18, s9 // (op 660) 


s_and_b32 s9, s0, s19 // (op 660) 


s_add_i32 s0, s1, s9 // (op 660) 


s_ashr_i32 s1, s0, s8 // (op 660) 


s_lshl_b32 s0, s13, 24 // (op 660) 
s_ashr_i32 s8, s0, 31 // (op 660) 

s_xor_b32 s0, s1, s8 // (op 660) 

s_sub_i32 s1, s0, s8 // (op 660) 


s_mul_hi_i32 s0, s1, s14 // (op 660) 
s_add_i32 s8, s0, s1 // (op 660) 

s_ashr_i32 s0, s8, 31 // (op 660) 
s_cmp_eq_i32 s14, 0 // (op 660) 
s_mov_b32 s9, scc // (op 660) 
s_cmp_eq_i32 s9, 0 // (op 660) 
s_cselect_b32 s18, 0, -1 // (op 660) 

s_and_b32 s9, s15, 31 // (op 660) 
s_lshl_b32 s19, 1, s9 // (op 660) 
s_add_i32 s49, s19, s18 // (op 660) 


s_and_b32 s18, s0, s49 // (op 660) 


s_add_i32 s0, s8, s18 // (op 660) 


s_ashr_i32 s8, s0, s9 // (op 660) 


s_lshl_b32 s0, s15, 24 // (op 660) 
s_ashr_i32 s9, s0, 31 // (op 660) 

s_xor_b32 s0, s8, s9 // (op 660) 

s_sub_i32 s8, s0, s9 // (op 660) 


s_add_u32 s18, s4, 16 // (op 660) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 660) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 660) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 660) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 660) 

s_mul_i32 s0, s8, s18 // (op 660) 

s_sub_i32 s8, s1, s0 // (op 660) 


s_mul_hi_i32 s9, s16, s11 // (op 660) 
s_add_i32 s0, s9, s16 // (op 660) 

s_ashr_i32 s1, s0, 31 // (op 660) 
s_cmp_eq_i32 s11, 0 // (op 660) 
s_mov_b32 s9, scc // (op 660) 
s_cmp_eq_i32 s9, 0 // (op 660) 
s_cselect_b32 s49, 0, -1 // (op 660) 

s_and_b32 s9, s46, 31 // (op 660) 
s_lshl_b32 s50, 1, s9 // (op 660) 
s_add_i32 s51, s50, s49 // (op 660) 


s_and_b32 s49, s1, s51 // (op 660) 


s_add_i32 s1, s0, s49 // (op 660) 


s_ashr_i32 s0, s1, s9 // (op 660) 


s_lshl_b32 s1, s46, 24 // (op 660) 
s_ashr_i32 s9, s1, 31 // (op 660) 

s_xor_b32 s1, s0, s9 // (op 660) 

s_sub_i32 s0, s1, s9 // (op 660) 


s_mul_i32 s1, s45, s18 // (op 660) 

s_mul_i32 s9, s10, s1 // (op 660) 

s_mul_i32 s1, s0, s9 // (op 660) 

s_sub_u32 s9, s16, s1 // (op 660) 

s_mul_hi_i32 s1, s9, s44 // (op 660) 
s_add_i32 s18, s1, s9 // (op 660) 


s_ashr_i32 s1, s18, 31 // (op 660) 
s_cmp_eq_i32 s44, 0 // (op 660) 
s_mov_b32 s9, scc // (op 660) 
s_cmp_eq_i32 s9, 0 // (op 660) 
s_cselect_b32 s19, 0, -1 // (op 660) 

s_and_b32 s9, s48, 31 // (op 660) 
s_lshl_b32 s49, 1, s9 // (op 660) 
s_add_i32 s50, s49, s19 // (op 660) 


s_and_b32 s19, s1, s50 // (op 660) 


s_add_i32 s1, s18, s19 // (op 660) 


s_ashr_i32 s18, s1, s9 // (op 660) 


s_lshl_b32 s1, s48, 24 // (op 660) 
s_ashr_i32 s9, s1, 31 // (op 660) 

s_xor_b32 s1, s18, s9 // (op 660) 

s_sub_i32 s18, s1, s9 // (op 660) 


s_cmp_eq_i32 s0, 0 // (op 660) 
s_mov_b32 s1, scc // (op 660) 

s_cmp_eq_i32 s1, 0 // (op 660) 
s_cselect_b32 s0, s18, s8 // (op 660) 



v_lshl_add_u32 v18, s0, 4, v16 // (op 660) 


v_mul_lo_u32 v16, v18, s30 // (op 660) 

v_add_u32 v18, v19, v16 // (op 660) 


s_mul_i32 s1, s30, s5 // (op 662) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 662) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 662) least significant
s_mul_hi_u32 s9, s30, s4 // (op 662) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 662) most significant: sum
s_add_u32 s1, s1, s9 // (op 662) most significant: sum


s_mov_b32 s64, s32 // (op 662) reg expression
s_mov_b32 s65, s33 // (op 662) reg expression
s_mov_b32 s66, s0 // (op 662) reg expression
s_mov_b32 s67, 131072 // (op 662) call()




buffer_load_ubyte v16, v17, s[64:67], 0 offen offset: 0 // (op 337) Load value



buffer_load_dwordx2 v[20:21], v12, s[56:59], 0 offen // (op 341) Load value



buffer_load_dwordx4 v[22:25], v10, s[52:55], 0 offen // (op 344) Load value



buffer_load_ubyte v19, v14, s[60:63], 0 offen offset: 0 // (op 347) Load value



buffer_load_ubyte v26, v17, s[64:67], 0 offen offset:4 // (op 351) Load value




buffer_load_dwordx2 v[28:29], v12, s[56:59], 0 offen offset:64 // (op 355) Load value




buffer_load_dwordx4 v[30:33], v10, s[52:55], 0 offen offset:64 // (op 358) Load value




buffer_load_ubyte v10, v14, s[60:63], 0 offen offset:4 // (op 361) Load value

v_and_b32 v12, -8, v1 // (op 780) 
v_sub_u32 v14, v1, v12 // (op 780) 

v_lshlrev_b32 v12, 4, v14 // (op 780) 

v_lshrrev_b32 v14, 3, v1 // (op 780) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 780) 


v_lshlrev_b32 v12, 2, v17 // (op 780) 

v_lshrrev_b32 v14, 3, v12 // (op 780) 




s_waitcnt vmcnt(6)
ds_write_b64 v14, v[20:21] // (op 363) Store local data 

v_and_b32 v12, -4, v1 // (op 727) 
v_sub_u32 v14, v1, v12 // (op 727) 

v_lshlrev_b32 v12, 5, v14 // (op 727) 

v_lshrrev_b32 v14, 2, v1 // (op 727) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 727) 


v_lshlrev_b32 v12, 2, v17 // (op 727) 

v_lshrrev_b32 v14, 3, v12 // (op 727) 




s_waitcnt vmcnt(5)
ds_write_b128 v14, v[22:25] offset:1024 // (op 364) Store local data 

v_and_b32 v12, -4, v1 // (op 833) 
v_sub_u32 v14, v1, v12 // (op 833) 

v_and_b32 v12, 4294967292, v1 // (op 833) 
v_add_u32 v17, v12, v14 // (op 833) 





s_waitcnt vmcnt(4)
ds_write_b8 v17, v19 offset:3072 // (op 365) Store local data 

v_and_b32 v12, -4, v1 // (op 698) 
v_sub_u32 v14, v1, v12 // (op 698) 

v_lshlrev_b32 v12, 5, v14 // (op 698) 

v_lshrrev_b32 v14, 2, v1 // (op 698) 
v_lshl_add_u32 v17, v14, 7, v12 // (op 698) 


v_lshlrev_b32 v12, 2, v17 // (op 698) 

v_lshrrev_b32 v14, 3, v12 // (op 698) 

v_and_b32 v12, -8, v1 // (op 751) 
v_sub_u32 v17, v1, v12 // (op 751) 

v_lshlrev_b32 v12, 4, v17 // (op 751) 

v_lshrrev_b32 v17, 3, v1 // (op 751) 
v_lshl_add_u32 v27, v17, 7, v12 // (op 751) 


v_lshlrev_b32 v12, 2, v27 // (op 751) 

v_lshrrev_b32 v17, 3, v12 // (op 751) 

v_and_b32 v12, -4, v1 // (op 804) 
v_sub_u32 v27, v1, v12 // (op 804) 

v_and_b32 v12, 4294967292, v1 // (op 804) 
v_add_u32 v34, v12, v27 // (op 804) 


v_and_b32 v12, -4, v1 // (op 891) 
v_sub_u32 v27, v1, v12 // (op 891) 

v_lshlrev_b32 v12, 5, v27 // (op 891) 

v_lshrrev_b32 v27, 2, v1 // (op 891) 
v_lshl_add_u32 v35, v27, 7, v12 // (op 891) 


v_lshlrev_b32 v12, 2, v35 // (op 891) 

v_lshrrev_b32 v27, 3, v12 // (op 891) 

v_and_b32 v12, -8, v1 // (op 915) 
v_sub_u32 v35, v1, v12 // (op 915) 

v_lshlrev_b32 v12, 4, v35 // (op 915) 

v_lshrrev_b32 v35, 3, v1 // (op 915) 
v_lshl_add_u32 v36, v35, 7, v12 // (op 915) 


v_lshlrev_b32 v12, 2, v36 // (op 915) 

v_lshrrev_b32 v35, 3, v12 // (op 915) 

v_and_b32 v12, -4, v1 // (op 939) 
v_sub_u32 v36, v1, v12 // (op 939) 

v_and_b32 v12, 4294967292, v1 // (op 939) 
v_add_u32 v37, v12, v36 // (op 939) 


s_mov_b32 s0, 0 // (op 55) call()
s_ashr_i64 s[8:9], s[34:35], 7 // (op 54) 
s_mov_b32 s1, s8 // (op 54) convert

s_and_b32 s8, s1, 4294967294 // (op 54) 

s_cmp_lt_u32 s0, s8 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_54 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_54:
s_add_u32 s0, s0, 2 // (op 56) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 984) 



ds_read_b128 v[38:41], v3 offset:1024 // (op 65) Load local data 
s_barrier  // (op 997) 



ds_read_b128 v[42:45], v4 // (op 70) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1011) 



ds_read_u8 v12, v0 offset:3072 // (op 75) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:41], v[42:45], a[0:3], v12, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 82) 





v_mov_b32 v12, v18 // (op 80) 

buffer_load_ubyte v16, v12, s[64:67], 0 offen offset:8 // (op 80) Load value


v_mov_b32 v36, v13 // (op 10) 

buffer_load_dwordx2 v[20:21], v36, s[56:59], 0 offen offset:128 // (op 10) Load value


v_mov_b32 v38, v11 // (op 4) 

buffer_load_dwordx4 v[22:25], v38, s[52:55], 0 offen offset:128 // (op 4) Load value


v_mov_b32 v39, v15 // (op 17) 

buffer_load_ubyte v19, v39, s[60:63], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1034) 



s_waitcnt vmcnt(6)
ds_write_b64 v35, v[28:29] offset:3200 // (op 235) Store local data 
s_barrier  // (op 1028) 



s_waitcnt vmcnt(5)
ds_write_b128 v27, v[30:33] offset:4224 // (op 225) Store local data 
s_barrier  // (op 1040) 



s_waitcnt vmcnt(4)
ds_write_b8 v37, v10 offset:6272 // (op 236) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1025) 



ds_read_b128 v[40:43], v6 offset:4224 // (op 227) Load local data 
s_barrier  // (op 1031) 
s_barrier  // (op 1091) 



ds_read_b128 v[44:47], v7 offset:3200 // (op 229) Load local data 
s_barrier  // (op 1097) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1037) 



ds_read_u8 v48, v5 offset:6272 // (op 231) Load local data 
s_barrier  // (op 1103) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:43], v[44:47], a[0:3], v48, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 234) 






buffer_load_ubyte v26, v12, s[64:67], 0 offen offset:12 // (op 233) Load value




v_add_u32 v18, 8, v18 // (op 652) 
buffer_load_dwordx2 v[28:29], v36, s[56:59], 0 offen offset:192 // (op 237) Load value




v_add_u32 v13, 128, v13 // (op 524) 
buffer_load_dwordx4 v[30:33], v38, s[52:55], 0 offen offset:192 // (op 224) Load value




v_add_u32 v11, 128, v11 // (op 460) 
buffer_load_ubyte v10, v39, s[60:63], 0 offen offset:12 // (op 238) Load value

s_barrier  // (op 1000) 
v_add_u32 v15, 8, v15 // (op 588) 



s_waitcnt vmcnt(6)
ds_write_b64 v17, v[20:21] // (op 13) Store local data 
s_barrier  // (op 987) 



s_waitcnt vmcnt(5)
ds_write_b128 v14, v[22:25] offset:1024 // (op 6) Store local data 
s_barrier  // (op 1014) 



s_waitcnt vmcnt(4)
ds_write_b8 v34, v19 offset:3072 // (op 21) Store local data 
s_ashr_i64 s[8:9], s[34:35], 7 // (op 54) 
s_mov_b32 s1, s8 // (op 54) convert

s_and_b32 s8, s1, 4294967294 // (op 54) 

s_cmp_lt_u32 s0, s8 // (op 54) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_54 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_54:















s_mul_hi_i32 s0, s16, s12 // (op 423) 
v_lshrrev_b32 v27, 2, v1 // (op 423) 
s_add_i32 s1, s0, s16 // (op 423) 

v_and_b32 v11, -4, v1 // (op 423) 
s_ashr_i32 s0, s1, 31 // (op 423) 
v_sub_u32 v12, v1, v11 // (op 423) 

s_cmp_eq_i32 s12, 0 // (op 423) 
s_mov_b32 s8, scc // (op 423) 
v_lshlrev_b32 v11, 5, v12 // (op 423) 

s_cmp_eq_i32 s8, 0 // (op 423) 
s_cselect_b32 s9, 0, -1 // (op 423) 

s_and_b32 s8, s13, 31 // (op 423) 
s_lshl_b32 s18, 1, s8 // (op 423) 
s_add_i32 s19, s18, s9 // (op 423) 


s_and_b32 s9, s0, s19 // (op 423) 


s_add_i32 s0, s1, s9 // (op 423) 


s_ashr_i32 s1, s0, s8 // (op 423) 


s_lshl_b32 s0, s13, 24 // (op 423) 
s_ashr_i32 s8, s0, 31 // (op 423) 

s_xor_b32 s0, s1, s8 // (op 423) 

s_sub_i32 s1, s0, s8 // (op 423) 


s_mul_hi_i32 s0, s1, s14 // (op 423) 
s_add_i32 s8, s0, s1 // (op 423) 

s_ashr_i32 s0, s8, 31 // (op 423) 
s_cmp_eq_i32 s14, 0 // (op 423) 
s_mov_b32 s9, scc // (op 423) 
s_cmp_eq_i32 s9, 0 // (op 423) 
s_cselect_b32 s18, 0, -1 // (op 423) 

s_and_b32 s9, s15, 31 // (op 423) 
s_lshl_b32 s19, 1, s9 // (op 423) 
s_add_i32 s49, s19, s18 // (op 423) 


s_and_b32 s18, s0, s49 // (op 423) 


s_add_i32 s0, s8, s18 // (op 423) 


s_ashr_i32 s8, s0, s9 // (op 423) 


s_lshl_b32 s0, s15, 24 // (op 423) 
s_ashr_i32 s9, s0, 31 // (op 423) 

s_xor_b32 s0, s8, s9 // (op 423) 

s_sub_i32 s8, s0, s9 // (op 423) 


s_mul_i32 s0, s1, s45 // (op 423) 

s_sub_u32 s1, s16, s0 // (op 423) 

s_mul_i32 s0, s8, s45 // (op 423) 
s_add_u32 s0, s0, s1 // (op 423) 


s_mul_hi_i32 s1, s16, s11 // (op 423) 
s_add_i32 s8, s1, s16 // (op 423) 

s_ashr_i32 s1, s8, 31 // (op 423) 
s_cmp_eq_i32 s11, 0 // (op 423) 
s_mov_b32 s9, scc // (op 423) 
s_cmp_eq_i32 s9, 0 // (op 423) 
s_cselect_b32 s18, 0, -1 // (op 423) 

s_and_b32 s9, s46, 31 // (op 423) 
s_lshl_b32 s19, 1, s9 // (op 423) 
s_add_i32 s49, s19, s18 // (op 423) 


s_and_b32 s18, s1, s49 // (op 423) 


s_add_i32 s1, s8, s18 // (op 423) 


s_ashr_i32 s8, s1, s9 // (op 423) 


s_lshl_b32 s1, s46, 24 // (op 423) 
s_ashr_i32 s9, s1, 31 // (op 423) 

s_xor_b32 s1, s8, s9 // (op 423) 

s_sub_i32 s8, s1, s9 // (op 423) 


s_add_u32 s18, s4, 16 // (op 423) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 423) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 423) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 423) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 423) 

s_mul_i32 s1, s45, s18 // (op 423) 

s_mul_i32 s9, s10, s1 // (op 423) 

s_mul_i32 s1, s8, s9 // (op 423) 

s_sub_u32 s9, s16, s1 // (op 423) 

s_mov_b32 s1, s9 // (op 423) 
s_mul_hi_i32 s18, s1, s44 // (op 423) 
s_add_i32 s19, s18, s1 // (op 423) 


s_ashr_i32 s1, s19, 31 // (op 423) 
s_cmp_eq_i32 s44, 0 // (op 423) 
s_mov_b32 s18, scc // (op 423) 
s_cmp_eq_i32 s18, 0 // (op 423) 
s_cselect_b32 s49, 0, -1 // (op 423) 

s_and_b32 s18, s48, 31 // (op 423) 
s_lshl_b32 s50, 1, s18 // (op 423) 
s_add_i32 s51, s50, s49 // (op 423) 


s_and_b32 s49, s1, s51 // (op 423) 


s_add_i32 s1, s19, s49 // (op 423) 


s_ashr_i32 s19, s1, s18 // (op 423) 


s_lshl_b32 s1, s48, 24 // (op 423) 
s_ashr_i32 s18, s1, 31 // (op 423) 

s_xor_b32 s1, s19, s18 // (op 423) 

s_sub_i32 s19, s1, s18 // (op 423) 


s_mul_i32 s1, s47, s45 // (op 423) 
s_add_u32 s50, s6, 32 // (op 423) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 423) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 423) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 423) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 423) 

s_sub_i32 s18, s50, s1 // (op 423) 

s_mul_i32 s49, s19, s18 // (op 423) 


s_sub_u32 s18, s9, s49 // (op 423) 


s_add_u32 s9, s18, s1 // (op 423) 


s_cmp_eq_i32 s8, 0 // (op 423) 
s_mov_b32 s1, scc // (op 423) 

s_cmp_eq_i32 s1, 0 // (op 423) 
s_cselect_b32 s8, s9, s0 // (op 423) 



v_lshl_add_u32 v12, s8, 5, v27 // (op 423) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 423) 
s_lshr_b32 s8, s0, 1 // (op 423) 

v_lshl_add_u32 v27, s8, 8, v11 // (op 423) 


v_mov_b32 v25, 0 // (op 423) convert
v_mov_b32 v24, v27 // (op 423) convert

v_mul_lo_u32 v23, v12, s3 // (op 423) most significant: low of low * high
v_mul_hi_u32 v11, v12, s2 // (op 423) most significant: high of low * low
v_mul_lo_u32 v22, v12, s2 // (op 423) least significant: low of low * low
v_add_u32 v23, v23, v11 // (op 423) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v24 // (op 423) least significant half
v_addc_co_u32 v23, s[0:1], v23, v25, s[0:1] // (op 423) most significant half



v_lshlrev_b64 v[24:25], 2, v[22:23] // (op 423) 

v_lshrrev_b64 v[22:23], 3, v[24:25] // (op 423) 

v_mov_b32 v27, v22 // (op 423) convert

s_mul_i32 s1, s2, s7 // (op 425) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 425) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 425) least significant
s_mul_hi_u32 s9, s2, s6 // (op 425) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 425) most significant: sum
s_add_u32 s1, s1, s9 // (op 425) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 425) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 425) 

s_mov_b32 s52, s20 // (op 425) reg expression
s_mov_b32 s53, s21 // (op 425) reg expression
s_mov_b32 s54, s0 // (op 425) reg expression
s_mov_b32 s55, 131072 // (op 425) call()

s_mul_hi_i32 s0, s16, s12 // (op 490) 
v_lshrrev_b32 v11, 3, v1 // (op 490) 
s_add_i32 s1, s0, s16 // (op 490) 

v_and_b32 v12, -8, v1 // (op 490) 
s_ashr_i32 s0, s1, 31 // (op 490) 
v_sub_u32 v13, v1, v12 // (op 490) 

s_cmp_eq_i32 s12, 0 // (op 490) 
s_mov_b32 s8, scc // (op 490) 
v_lshlrev_b32 v12, 4, v13 // (op 490) 

s_cmp_eq_i32 s8, 0 // (op 490) 
s_cselect_b32 s9, 0, -1 // (op 490) 

s_and_b32 s8, s13, 31 // (op 490) 
s_lshl_b32 s18, 1, s8 // (op 490) 
s_add_i32 s19, s18, s9 // (op 490) 


s_and_b32 s9, s0, s19 // (op 490) 


s_add_i32 s0, s1, s9 // (op 490) 


s_ashr_i32 s1, s0, s8 // (op 490) 


s_lshl_b32 s0, s13, 24 // (op 490) 
s_ashr_i32 s8, s0, 31 // (op 490) 

s_xor_b32 s0, s1, s8 // (op 490) 

s_sub_i32 s1, s0, s8 // (op 490) 


s_mul_hi_i32 s0, s1, s14 // (op 490) 
s_add_i32 s8, s0, s1 // (op 490) 

s_ashr_i32 s0, s8, 31 // (op 490) 
s_cmp_eq_i32 s14, 0 // (op 490) 
s_mov_b32 s9, scc // (op 490) 
s_cmp_eq_i32 s9, 0 // (op 490) 
s_cselect_b32 s18, 0, -1 // (op 490) 

s_and_b32 s9, s15, 31 // (op 490) 
s_lshl_b32 s19, 1, s9 // (op 490) 
s_add_i32 s49, s19, s18 // (op 490) 


s_and_b32 s18, s0, s49 // (op 490) 


s_add_i32 s0, s8, s18 // (op 490) 


s_ashr_i32 s8, s0, s9 // (op 490) 


s_lshl_b32 s0, s15, 24 // (op 490) 
s_ashr_i32 s9, s0, 31 // (op 490) 

s_xor_b32 s0, s8, s9 // (op 490) 

s_sub_i32 s8, s0, s9 // (op 490) 


s_add_u32 s18, s4, 16 // (op 490) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 490) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 490) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 490) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 490) 

s_mul_i32 s0, s8, s18 // (op 490) 

s_sub_i32 s8, s1, s0 // (op 490) 


s_mul_hi_i32 s9, s16, s11 // (op 490) 
s_add_i32 s0, s9, s16 // (op 490) 

s_ashr_i32 s1, s0, 31 // (op 490) 
s_cmp_eq_i32 s11, 0 // (op 490) 
s_mov_b32 s9, scc // (op 490) 
s_cmp_eq_i32 s9, 0 // (op 490) 
s_cselect_b32 s49, 0, -1 // (op 490) 

s_and_b32 s9, s46, 31 // (op 490) 
s_lshl_b32 s50, 1, s9 // (op 490) 
s_add_i32 s51, s50, s49 // (op 490) 


s_and_b32 s49, s1, s51 // (op 490) 


s_add_i32 s1, s0, s49 // (op 490) 


s_ashr_i32 s0, s1, s9 // (op 490) 


s_lshl_b32 s1, s46, 24 // (op 490) 
s_ashr_i32 s9, s1, 31 // (op 490) 

s_xor_b32 s1, s0, s9 // (op 490) 

s_sub_i32 s0, s1, s9 // (op 490) 


s_mul_i32 s1, s45, s18 // (op 490) 

s_mul_i32 s9, s10, s1 // (op 490) 

s_mul_i32 s1, s0, s9 // (op 490) 

s_sub_u32 s9, s16, s1 // (op 490) 

s_mul_hi_i32 s1, s9, s44 // (op 490) 
s_add_i32 s18, s1, s9 // (op 490) 


s_ashr_i32 s1, s18, 31 // (op 490) 
s_cmp_eq_i32 s44, 0 // (op 490) 
s_mov_b32 s9, scc // (op 490) 
s_cmp_eq_i32 s9, 0 // (op 490) 
s_cselect_b32 s19, 0, -1 // (op 490) 

s_and_b32 s9, s48, 31 // (op 490) 
s_lshl_b32 s49, 1, s9 // (op 490) 
s_add_i32 s50, s49, s19 // (op 490) 


s_and_b32 s19, s1, s50 // (op 490) 


s_add_i32 s1, s18, s19 // (op 490) 


s_ashr_i32 s18, s1, s9 // (op 490) 


s_lshl_b32 s1, s48, 24 // (op 490) 
s_ashr_i32 s9, s1, 31 // (op 490) 

s_xor_b32 s1, s18, s9 // (op 490) 

s_sub_i32 s18, s1, s9 // (op 490) 


s_cmp_eq_i32 s0, 0 // (op 490) 
s_mov_b32 s1, scc // (op 490) 

s_cmp_eq_i32 s1, 0 // (op 490) 
s_cselect_b32 s0, s18, s8 // (op 490) 



v_lshl_add_u32 v13, s0, 4, v11 // (op 490) 


v_mul_lo_u32 v15, v13, s23 // (op 490) most significant: low of low * high
v_mul_hi_u32 v11, v13, s22 // (op 490) most significant: high of low * low
v_mul_lo_u32 v14, v13, s22 // (op 490) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 490) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 490) 
s_lshr_b32 s8, s0, 1 // (op 490) 

v_lshl_add_u32 v11, s8, 8, v12 // (op 490) 


v_mov_b32 v13, 0 // (op 490) convert
v_mov_b32 v12, v11 // (op 490) convert

v_add_co_u32 v16, s[0:1], v12, v14 // (op 490) least significant half
v_addc_co_u32 v17, s[0:1], v13, v15, s[0:1] // (op 490) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 490) 


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 490) 

v_mov_b32 v11, v14 // (op 490) convert

s_mul_i32 s1, s22, s5 // (op 492) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 492) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 492) least significant
s_mul_hi_u32 s9, s22, s4 // (op 492) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 492) most significant: sum
s_add_u32 s1, s1, s9 // (op 492) most significant: sum


s_lshl_b64 s[8:9], s[0:1], 2 // (op 492) 

s_ashr_i64 s[0:1], s[8:9], 3 // (op 492) 

s_mov_b32 s56, s24 // (op 492) reg expression
s_mov_b32 s57, s25 // (op 492) reg expression
s_mov_b32 s58, s0 // (op 492) reg expression
s_mov_b32 s59, 131072 // (op 492) call()

s_mul_hi_i32 s0, s16, s12 // (op 554) 
v_lshrrev_b32 v12, 2, v1 // (op 554) 
s_add_i32 s1, s0, s16 // (op 554) 

v_and_b32 v13, -4, v1 // (op 554) 
s_ashr_i32 s0, s1, 31 // (op 554) 
v_sub_u32 v14, v1, v13 // (op 554) 

s_cmp_eq_i32 s12, 0 // (op 554) 
s_mov_b32 s8, scc // (op 554) 
s_cmp_eq_i32 s8, 0 // (op 554) 
s_cselect_b32 s9, 0, -1 // (op 554) 

s_and_b32 s8, s13, 31 // (op 554) 
s_lshl_b32 s18, 1, s8 // (op 554) 
s_add_i32 s19, s18, s9 // (op 554) 


s_and_b32 s9, s0, s19 // (op 554) 


s_add_i32 s0, s1, s9 // (op 554) 


s_ashr_i32 s1, s0, s8 // (op 554) 


s_lshl_b32 s0, s13, 24 // (op 554) 
s_ashr_i32 s8, s0, 31 // (op 554) 

s_xor_b32 s0, s1, s8 // (op 554) 

s_sub_i32 s1, s0, s8 // (op 554) 


s_mul_hi_i32 s0, s1, s14 // (op 554) 
s_add_i32 s8, s0, s1 // (op 554) 

s_ashr_i32 s0, s8, 31 // (op 554) 
s_cmp_eq_i32 s14, 0 // (op 554) 
s_mov_b32 s9, scc // (op 554) 
s_cmp_eq_i32 s9, 0 // (op 554) 
s_cselect_b32 s18, 0, -1 // (op 554) 

s_and_b32 s9, s15, 31 // (op 554) 
s_lshl_b32 s19, 1, s9 // (op 554) 
s_add_i32 s49, s19, s18 // (op 554) 


s_and_b32 s18, s0, s49 // (op 554) 


s_add_i32 s0, s8, s18 // (op 554) 


s_ashr_i32 s8, s0, s9 // (op 554) 


s_lshl_b32 s0, s15, 24 // (op 554) 
s_ashr_i32 s9, s0, 31 // (op 554) 

s_xor_b32 s0, s8, s9 // (op 554) 

s_sub_i32 s8, s0, s9 // (op 554) 


s_mul_i32 s0, s1, s45 // (op 554) 

s_sub_u32 s1, s16, s0 // (op 554) 

s_mul_i32 s0, s8, s45 // (op 554) 
s_add_u32 s0, s0, s1 // (op 554) 


s_mul_hi_i32 s1, s16, s11 // (op 554) 
s_add_i32 s8, s1, s16 // (op 554) 

s_ashr_i32 s1, s8, 31 // (op 554) 
s_cmp_eq_i32 s11, 0 // (op 554) 
s_mov_b32 s9, scc // (op 554) 
s_cmp_eq_i32 s9, 0 // (op 554) 
s_cselect_b32 s18, 0, -1 // (op 554) 

s_and_b32 s9, s46, 31 // (op 554) 
s_lshl_b32 s19, 1, s9 // (op 554) 
s_add_i32 s49, s19, s18 // (op 554) 


s_and_b32 s18, s1, s49 // (op 554) 


s_add_i32 s1, s8, s18 // (op 554) 


s_ashr_i32 s8, s1, s9 // (op 554) 


s_lshl_b32 s1, s46, 24 // (op 554) 
s_ashr_i32 s9, s1, 31 // (op 554) 

s_xor_b32 s1, s8, s9 // (op 554) 

s_sub_i32 s8, s1, s9 // (op 554) 


s_add_u32 s18, s4, 16 // (op 554) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 554) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 554) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 554) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 554) 

s_mul_i32 s1, s45, s18 // (op 554) 

s_mul_i32 s9, s10, s1 // (op 554) 

s_mul_i32 s1, s8, s9 // (op 554) 

s_sub_u32 s9, s16, s1 // (op 554) 

s_mov_b32 s1, s9 // (op 554) 
s_mul_hi_i32 s18, s1, s44 // (op 554) 
s_add_i32 s19, s18, s1 // (op 554) 


s_ashr_i32 s1, s19, 31 // (op 554) 
s_cmp_eq_i32 s44, 0 // (op 554) 
s_mov_b32 s18, scc // (op 554) 
s_cmp_eq_i32 s18, 0 // (op 554) 
s_cselect_b32 s49, 0, -1 // (op 554) 

s_and_b32 s18, s48, 31 // (op 554) 
s_lshl_b32 s50, 1, s18 // (op 554) 
s_add_i32 s51, s50, s49 // (op 554) 


s_and_b32 s49, s1, s51 // (op 554) 


s_add_i32 s1, s19, s49 // (op 554) 


s_ashr_i32 s19, s1, s18 // (op 554) 


s_lshl_b32 s1, s48, 24 // (op 554) 
s_ashr_i32 s18, s1, 31 // (op 554) 

s_xor_b32 s1, s19, s18 // (op 554) 

s_sub_i32 s19, s1, s18 // (op 554) 


s_mul_i32 s1, s47, s45 // (op 554) 
s_add_u32 s50, s6, 32 // (op 554) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 554) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 554) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 554) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 5 // (op 554) 

s_sub_i32 s18, s50, s1 // (op 554) 

s_mul_i32 s49, s19, s18 // (op 554) 


s_sub_u32 s18, s9, s49 // (op 554) 


s_add_u32 s9, s18, s1 // (op 554) 


s_cmp_eq_i32 s8, 0 // (op 554) 
s_mov_b32 s1, scc // (op 554) 

s_cmp_eq_i32 s1, 0 // (op 554) 
s_cselect_b32 s8, s9, s0 // (op 554) 



v_lshl_add_u32 v13, s8, 5, v12 // (op 554) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 554) 
s_lshr_b32 s8, s0, 1 // (op 554) 

v_lshl_add_u32 v12, s8, 3, v14 // (op 554) 


v_mul_lo_u32 v14, v13, s26 // (op 554) 
v_add_u32 v14, v14, v12 // (op 554) 


s_mul_i32 s1, s26, s7 // (op 556) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 556) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 556) least significant
s_mul_hi_u32 s9, s26, s6 // (op 556) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 556) most significant: sum
s_add_u32 s1, s1, s9 // (op 556) most significant: sum


s_mov_b32 s60, s28 // (op 556) reg expression
s_mov_b32 s61, s29 // (op 556) reg expression
s_mov_b32 s62, s0 // (op 556) reg expression
s_mov_b32 s63, 131072 // (op 556) call()

s_mul_hi_i32 s0, s16, s12 // (op 618) 
v_and_b32 v12, 15, v1 // (op 618) 
s_add_i32 s1, s0, s16 // (op 618) 

v_and_b32 v13, 63, v1 // (op 618) 
s_ashr_i32 s0, s1, 31 // (op 618) 
v_lshrrev_b32 v15, 4, v13 // (op 618) 

s_cmp_eq_i32 s12, 0 // (op 618) 
s_mov_b32 s8, scc // (op 618) 
s_cmp_eq_i32 s8, 0 // (op 618) 
s_cselect_b32 s9, 0, -1 // (op 618) 

s_and_b32 s8, s13, 31 // (op 618) 
s_lshl_b32 s18, 1, s8 // (op 618) 
s_add_i32 s19, s18, s9 // (op 618) 


s_and_b32 s9, s0, s19 // (op 618) 


s_add_i32 s0, s1, s9 // (op 618) 


s_ashr_i32 s1, s0, s8 // (op 618) 


s_lshl_b32 s0, s13, 24 // (op 618) 
s_ashr_i32 s8, s0, 31 // (op 618) 

s_xor_b32 s0, s1, s8 // (op 618) 

s_sub_i32 s1, s0, s8 // (op 618) 


s_mul_hi_i32 s0, s1, s14 // (op 618) 
s_add_i32 s8, s0, s1 // (op 618) 

s_ashr_i32 s0, s8, 31 // (op 618) 
s_cmp_eq_i32 s14, 0 // (op 618) 
s_mov_b32 s9, scc // (op 618) 
s_cmp_eq_i32 s9, 0 // (op 618) 
s_cselect_b32 s18, 0, -1 // (op 618) 

s_and_b32 s9, s15, 31 // (op 618) 
s_lshl_b32 s19, 1, s9 // (op 618) 
s_add_i32 s49, s19, s18 // (op 618) 


s_and_b32 s18, s0, s49 // (op 618) 


s_add_i32 s0, s8, s18 // (op 618) 


s_ashr_i32 s8, s0, s9 // (op 618) 


s_lshl_b32 s0, s15, 24 // (op 618) 
s_ashr_i32 s9, s0, 31 // (op 618) 

s_xor_b32 s0, s8, s9 // (op 618) 

s_sub_i32 s8, s0, s9 // (op 618) 


s_add_u32 s18, s4, 16 // (op 618) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 618) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 618) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 618) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 618) 

s_mul_i32 s0, s8, s18 // (op 618) 

s_sub_i32 s8, s1, s0 // (op 618) 


s_mul_hi_i32 s9, s16, s11 // (op 618) 
s_add_i32 s0, s9, s16 // (op 618) 

s_ashr_i32 s1, s0, 31 // (op 618) 
s_cmp_eq_i32 s11, 0 // (op 618) 
s_mov_b32 s9, scc // (op 618) 
s_cmp_eq_i32 s9, 0 // (op 618) 
s_cselect_b32 s49, 0, -1 // (op 618) 

s_and_b32 s9, s46, 31 // (op 618) 
s_lshl_b32 s50, 1, s9 // (op 618) 
s_add_i32 s51, s50, s49 // (op 618) 


s_and_b32 s49, s1, s51 // (op 618) 


s_add_i32 s1, s0, s49 // (op 618) 


s_ashr_i32 s0, s1, s9 // (op 618) 


s_lshl_b32 s1, s46, 24 // (op 618) 
s_ashr_i32 s9, s1, 31 // (op 618) 

s_xor_b32 s1, s0, s9 // (op 618) 

s_sub_i32 s0, s1, s9 // (op 618) 


s_mul_i32 s1, s45, s18 // (op 618) 

s_mul_i32 s9, s10, s1 // (op 618) 

s_mul_i32 s1, s0, s9 // (op 618) 

s_sub_u32 s9, s16, s1 // (op 618) 

s_mul_hi_i32 s1, s9, s44 // (op 618) 
s_add_i32 s18, s1, s9 // (op 618) 


s_ashr_i32 s1, s18, 31 // (op 618) 
s_cmp_eq_i32 s44, 0 // (op 618) 
s_mov_b32 s9, scc // (op 618) 
s_cmp_eq_i32 s9, 0 // (op 618) 
s_cselect_b32 s19, 0, -1 // (op 618) 

s_and_b32 s9, s48, 31 // (op 618) 
s_lshl_b32 s49, 1, s9 // (op 618) 
s_add_i32 s50, s49, s19 // (op 618) 


s_and_b32 s19, s1, s50 // (op 618) 


s_add_i32 s1, s18, s19 // (op 618) 


s_ashr_i32 s18, s1, s9 // (op 618) 


s_lshl_b32 s1, s48, 24 // (op 618) 
s_ashr_i32 s9, s1, 31 // (op 618) 

s_xor_b32 s1, s18, s9 // (op 618) 

s_sub_i32 s18, s1, s9 // (op 618) 


s_cmp_eq_i32 s0, 0 // (op 618) 
s_mov_b32 s1, scc // (op 618) 

s_cmp_eq_i32 s1, 0 // (op 618) 
s_cselect_b32 s0, s18, s8 // (op 618) 



v_lshl_add_u32 v13, s0, 4, v12 // (op 618) 


v_mul_lo_u32 v12, v13, s30 // (op 618) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 618) 
s_lshr_b32 s8, s0, 1 // (op 618) 

v_lshl_add_u32 v13, s8, 3, v15 // (op 618) 


v_add_u32 v15, v13, v12 // (op 618) 


s_mul_i32 s1, s30, s5 // (op 620) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 620) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 620) least significant
s_mul_hi_u32 s9, s30, s4 // (op 620) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 620) most significant: sum
s_add_u32 s1, s1, s9 // (op 620) most significant: sum


s_mov_b32 s64, s32 // (op 620) reg expression
s_mov_b32 s65, s33 // (op 620) reg expression
s_mov_b32 s66, s0 // (op 620) reg expression
s_mov_b32 s67, 131072 // (op 620) call()

v_and_b32 v12, -4, v1 // (op 711) 
v_sub_u32 v13, v1, v12 // (op 711) 

v_lshlrev_b32 v12, 5, v13 // (op 711) 

v_lshrrev_b32 v13, 2, v1 // (op 711) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 711) 


v_lshlrev_b32 v12, 2, v16 // (op 711) 

v_lshrrev_b32 v13, 3, v12 // (op 711) 

v_and_b32 v12, -8, v1 // (op 764) 
v_sub_u32 v16, v1, v12 // (op 764) 

v_lshlrev_b32 v12, 4, v16 // (op 764) 

v_lshrrev_b32 v16, 3, v1 // (op 764) 
v_lshl_add_u32 v17, v16, 7, v12 // (op 764) 


v_lshlrev_b32 v12, 2, v17 // (op 764) 

v_lshrrev_b32 v16, 3, v12 // (op 764) 

v_and_b32 v12, -4, v1 // (op 817) 
v_sub_u32 v17, v1, v12 // (op 817) 

v_and_b32 v12, 4294967292, v1 // (op 817) 
v_add_u32 v18, v12, v17 // (op 817) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 162) 
s_and_b32 s8, s0, 4294967294 // (op 162) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 161) 
s_mov_b32 s9, s0 // (op 161) convert

s_cmp_lt_u32 s8, s9 // (op 161) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_161:


s_add_u32 s8, s8, 1 // (op 163) 
v_mov_b32 v10, v27 // (op 166) 

buffer_load_dwordx4 v[20:23], v10, s[52:55], 0 offen // (op 166) Load value

v_add_u32 v27, 64, v27 // (op 413) 


s_barrier  // (op 990) 
v_mov_b32 v10, v11 // (op 179) 




buffer_load_dwordx2 v[24:25], v10, s[56:59], 0 offen // (op 179) Load value

v_add_u32 v11, 64, v11 // (op 482) 
s_waitcnt vmcnt(1)
ds_write_b128 v13, v[20:23] offset:1024 // (op 167) Store local data 



v_mov_b32 v10, v14 // (op 180) 

s_barrier  // (op 1003) 
buffer_load_ubyte v12, v10, s[60:63], 0 offen offset: 0 // (op 180) Load value




v_add_u32 v14, 4, v14 // (op 546) 
s_waitcnt vmcnt(1)
ds_write_b64 v16, v[24:25] // (op 177) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 993) 



s_barrier  // (op 1017) 
ds_read_b128 v[20:23], v3 offset:1024 // (op 169) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1007) 
s_waitcnt vmcnt(0)
ds_write_b8 v18, v12 offset:3072 // (op 178) Store local data 




ds_read_b128 v[28:31], v4 // (op 171) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1021) 



ds_read_u8 v10, v0 offset:3072 // (op 173) Load local data 


v_mov_b32 v12, v15 // (op 175) 

buffer_load_ubyte v17, v12, s[64:67], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v15, 4, v15 // (op 610) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[28:31], a[0:3], v10, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 176) 




s_ashr_i64 s[0:1], s[34:35], 7 // (op 161) 
s_mov_b32 s9, s0 // (op 161) convert

s_cmp_lt_u32 s8, s9 // (op 161) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_161:





















s_barrier  // (op 1085) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_123
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

s_mul_hi_i32 s0, s16, s12 // (op 859) 
v_lshrrev_b32 v0, 6, v1 // (op 859) 
s_add_i32 s1, s0, s16 // (op 859) 

v_and_b32 v3, 63, v1 // (op 859) 
s_ashr_i32 s0, s1, 31 // (op 859) 
v_lshrrev_b32 v4, 4, v3 // (op 859) 
s_cmp_eq_i32 s12, 0 // (op 859) 
s_mov_b32 s2, scc // (op 859) 
v_lshlrev_b32 v5, 2, v4 // (op 859) 

s_cmp_eq_i32 s2, 0 // (op 859) 
s_cselect_b32 s3, 0, -1 // (op 859) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 859) 


s_and_b32 s2, s13, 31 // (op 859) 
v_lshrrev_b32 v0, 2, v3 // (op 859) 

s_lshl_b32 s8, 1, s2 // (op 859) 
v_and_b32 v3, 3, v0 // (op 859) 

s_add_i32 s9, s8, s3 // (op 859) 


v_and_b32 v0, 3, v1 // (op 859) 
s_and_b32 s3, s0, s9 // (op 859) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 859) 


s_add_i32 s0, s1, s3 // (op 859) 


s_ashr_i32 s1, s0, s2 // (op 859) 


s_lshl_b32 s0, s13, 24 // (op 859) 
s_ashr_i32 s2, s0, 31 // (op 859) 

s_xor_b32 s0, s1, s2 // (op 859) 

s_sub_i32 s1, s0, s2 // (op 859) 


s_mul_hi_i32 s0, s1, s14 // (op 859) 
s_add_i32 s2, s0, s1 // (op 859) 

s_ashr_i32 s0, s2, 31 // (op 859) 
s_cmp_eq_i32 s14, 0 // (op 859) 
s_mov_b32 s3, scc // (op 859) 
s_cmp_eq_i32 s3, 0 // (op 859) 
s_cselect_b32 s8, 0, -1 // (op 859) 

s_and_b32 s3, s15, 31 // (op 859) 
s_lshl_b32 s9, 1, s3 // (op 859) 
s_add_i32 s18, s9, s8 // (op 859) 


s_and_b32 s8, s0, s18 // (op 859) 


s_add_i32 s0, s2, s8 // (op 859) 


s_ashr_i32 s2, s0, s3 // (op 859) 


s_lshl_b32 s0, s15, 24 // (op 859) 
s_ashr_i32 s3, s0, 31 // (op 859) 

s_xor_b32 s0, s2, s3 // (op 859) 

s_sub_i32 s2, s0, s3 // (op 859) 


s_mul_i32 s0, s1, s45 // (op 859) 
s_sub_u32 s3, s16, s0 // (op 859) 

s_mul_i32 s0, s2, s45 // (op 859) 
s_add_u32 s0, s0, s3 // (op 859) 

s_mul_hi_i32 s3, s16, s11 // (op 859) 
s_add_i32 s8, s3, s16 // (op 859) 

s_ashr_i32 s3, s8, 31 // (op 859) 
s_cmp_eq_i32 s11, 0 // (op 859) 
s_mov_b32 s9, scc // (op 859) 
s_cmp_eq_i32 s9, 0 // (op 859) 
s_cselect_b32 s18, 0, -1 // (op 859) 

s_and_b32 s9, s46, 31 // (op 859) 
s_lshl_b32 s19, 1, s9 // (op 859) 
s_add_i32 s20, s19, s18 // (op 859) 


s_and_b32 s18, s3, s20 // (op 859) 


s_add_i32 s3, s8, s18 // (op 859) 


s_ashr_i32 s8, s3, s9 // (op 859) 


s_lshl_b32 s3, s46, 24 // (op 859) 
s_ashr_i32 s9, s3, 31 // (op 859) 

s_xor_b32 s3, s8, s9 // (op 859) 

s_sub_i32 s8, s3, s9 // (op 859) 


s_add_u32 s18, s4, 16 // (op 859) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 859) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 859) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 859) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 859) 

s_mul_i32 s3, s45, s18 // (op 859) 
s_mul_i32 s9, s10, s3 // (op 859) 

s_mul_i32 s3, s8, s9 // (op 859) 

s_sub_u32 s9, s16, s3 // (op 859) 

s_mov_b32 s3, s9 // (op 859) 
s_mul_hi_i32 s20, s3, s44 // (op 859) 
s_add_i32 s21, s20, s3 // (op 859) 


s_ashr_i32 s3, s21, 31 // (op 859) 
s_cmp_eq_i32 s44, 0 // (op 859) 
s_mov_b32 s20, scc // (op 859) 
s_cmp_eq_i32 s20, 0 // (op 859) 
s_cselect_b32 s22, 0, -1 // (op 859) 

s_and_b32 s20, s48, 31 // (op 859) 
s_lshl_b32 s23, 1, s20 // (op 859) 
s_add_i32 s24, s23, s22 // (op 859) 


s_and_b32 s22, s3, s24 // (op 859) 


s_add_i32 s3, s21, s22 // (op 859) 


s_ashr_i32 s21, s3, s20 // (op 859) 


s_lshl_b32 s3, s48, 24 // (op 859) 
s_ashr_i32 s20, s3, 31 // (op 859) 

s_xor_b32 s3, s21, s20 // (op 859) 

s_sub_i32 s21, s3, s20 // (op 859) 


s_mul_i32 s3, s47, s45 // (op 859) 
s_add_u32 s22, s6, 32 // (op 859) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 859) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 859) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 859) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 859) 

s_sub_i32 s20, s22, s3 // (op 859) 

s_mul_i32 s22, s21, s20 // (op 859) 

s_sub_u32 s20, s9, s22 // (op 859) 


s_add_u32 s9, s20, s3 // (op 859) 


s_cmp_eq_i32 s8, 0 // (op 859) 
s_mov_b32 s3, scc // (op 859) 

s_cmp_eq_i32 s3, 0 // (op 859) 
s_cselect_b32 s8, s9, s0 // (op 859) 


s_mul_i32 s0, s2, s18 // (op 859) 


v_lshl_add_u32 v0, s8, 5, v4 // (op 859) 


s_sub_i32 s2, s1, s0 // (op 859) 


s_cmp_eq_i32 s3, 0 // (op 859) 
s_cselect_b32 s0, s21, s2 // (op 859) 



v_lshl_add_u32 v3, s0, 4, v5 // (op 859) 


v_mul_lo_u32 v4, v3, s36 // (op 859) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 859) 


s_mul_i32 s9, s36, s5 // (op 861) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 861) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 861) least significant
s_mul_hi_u32 s1, s36, s4 // (op 861) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 861) most significant: sum
s_add_u32 s9, s9, s1 // (op 861) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 861) 

s_mov_b32 s32, s38 // (op 861) reg expression
s_mov_b32 s33, s39 // (op 861) reg expression
s_mov_b32 s34, s0 // (op 861) reg expression
s_mov_b32 s35, 131072 // (op 861) call()




buffer_store_dwordx4 v[10:13], v3, s[32:35], 0 offen // (op 52) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_123:
s_mul_hi_i32 s0, s16, s12 // (op 681) 
v_lshrrev_b32 v0, 6, v1 // (op 681) 
s_add_i32 s1, s0, s16 // (op 681) 

v_and_b32 v3, 63, v1 // (op 681) 
s_ashr_i32 s0, s1, 31 // (op 681) 
v_lshrrev_b32 v4, 4, v3 // (op 681) 
s_cmp_eq_i32 s12, 0 // (op 681) 
s_mov_b32 s2, scc // (op 681) 
v_lshlrev_b32 v5, 2, v4 // (op 681) 

s_cmp_eq_i32 s2, 0 // (op 681) 
s_cselect_b32 s3, 0, -1 // (op 681) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 681) 


s_and_b32 s2, s13, 31 // (op 681) 
v_lshrrev_b32 v0, 2, v3 // (op 681) 

s_lshl_b32 s8, 1, s2 // (op 681) 
v_and_b32 v3, 3, v0 // (op 681) 

s_add_i32 s9, s8, s3 // (op 681) 


v_and_b32 v0, 3, v1 // (op 681) 
s_and_b32 s3, s0, s9 // (op 681) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 681) 


s_add_i32 s0, s1, s3 // (op 681) 


s_ashr_i32 s1, s0, s2 // (op 681) 


s_lshl_b32 s0, s13, 24 // (op 681) 
s_ashr_i32 s2, s0, 31 // (op 681) 

s_xor_b32 s0, s1, s2 // (op 681) 

s_sub_i32 s1, s0, s2 // (op 681) 


s_mul_hi_i32 s0, s1, s14 // (op 681) 
s_add_i32 s2, s0, s1 // (op 681) 

s_ashr_i32 s0, s2, 31 // (op 681) 
s_cmp_eq_i32 s14, 0 // (op 681) 
s_mov_b32 s3, scc // (op 681) 
s_cmp_eq_i32 s3, 0 // (op 681) 
s_cselect_b32 s8, 0, -1 // (op 681) 

s_and_b32 s3, s15, 31 // (op 681) 
s_lshl_b32 s9, 1, s3 // (op 681) 
s_add_i32 s18, s9, s8 // (op 681) 


s_and_b32 s8, s0, s18 // (op 681) 


s_add_i32 s0, s2, s8 // (op 681) 


s_ashr_i32 s2, s0, s3 // (op 681) 


s_lshl_b32 s0, s15, 24 // (op 681) 
s_ashr_i32 s3, s0, 31 // (op 681) 

s_xor_b32 s0, s2, s3 // (op 681) 

s_sub_i32 s2, s0, s3 // (op 681) 


s_mul_i32 s0, s1, s45 // (op 681) 
s_sub_u32 s3, s16, s0 // (op 681) 

s_mul_i32 s0, s2, s45 // (op 681) 
s_add_u32 s0, s0, s3 // (op 681) 

s_mul_hi_i32 s3, s16, s11 // (op 681) 
s_add_i32 s8, s3, s16 // (op 681) 

s_ashr_i32 s3, s8, 31 // (op 681) 
s_cmp_eq_i32 s11, 0 // (op 681) 
s_mov_b32 s9, scc // (op 681) 
s_cmp_eq_i32 s9, 0 // (op 681) 
s_cselect_b32 s18, 0, -1 // (op 681) 

s_and_b32 s9, s46, 31 // (op 681) 
s_lshl_b32 s19, 1, s9 // (op 681) 
s_add_i32 s20, s19, s18 // (op 681) 


s_and_b32 s18, s3, s20 // (op 681) 


s_add_i32 s3, s8, s18 // (op 681) 


s_ashr_i32 s8, s3, s9 // (op 681) 


s_lshl_b32 s3, s46, 24 // (op 681) 
s_ashr_i32 s9, s3, 31 // (op 681) 

s_xor_b32 s3, s8, s9 // (op 681) 

s_sub_i32 s8, s3, s9 // (op 681) 


s_add_u32 s18, s4, 16 // (op 681) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 681) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 681) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 681) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 681) 

s_mul_i32 s3, s45, s18 // (op 681) 
s_mul_i32 s9, s10, s3 // (op 681) 

s_mul_i32 s3, s8, s9 // (op 681) 

s_sub_u32 s9, s16, s3 // (op 681) 

s_mov_b32 s3, s9 // (op 681) 
s_mul_hi_i32 s20, s3, s44 // (op 681) 
s_add_i32 s21, s20, s3 // (op 681) 


s_ashr_i32 s3, s21, 31 // (op 681) 
s_cmp_eq_i32 s44, 0 // (op 681) 
s_mov_b32 s20, scc // (op 681) 
s_cmp_eq_i32 s20, 0 // (op 681) 
s_cselect_b32 s22, 0, -1 // (op 681) 

s_and_b32 s20, s48, 31 // (op 681) 
s_lshl_b32 s23, 1, s20 // (op 681) 
s_add_i32 s24, s23, s22 // (op 681) 


s_and_b32 s22, s3, s24 // (op 681) 


s_add_i32 s3, s21, s22 // (op 681) 


s_ashr_i32 s21, s3, s20 // (op 681) 


s_lshl_b32 s3, s48, 24 // (op 681) 
s_ashr_i32 s20, s3, 31 // (op 681) 

s_xor_b32 s3, s21, s20 // (op 681) 

s_sub_i32 s21, s3, s20 // (op 681) 


s_mul_i32 s3, s47, s45 // (op 681) 
s_add_u32 s22, s6, 32 // (op 681) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 681) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 681) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 681) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 681) 

s_sub_i32 s20, s22, s3 // (op 681) 

s_mul_i32 s22, s21, s20 // (op 681) 

s_sub_u32 s20, s9, s22 // (op 681) 


s_add_u32 s9, s20, s3 // (op 681) 


s_cmp_eq_i32 s8, 0 // (op 681) 
s_mov_b32 s3, scc // (op 681) 

s_cmp_eq_i32 s3, 0 // (op 681) 
s_cselect_b32 s8, s9, s0 // (op 681) 


s_mul_i32 s0, s2, s18 // (op 681) 


v_lshl_add_u32 v0, s8, 5, v4 // (op 681) 


s_sub_i32 s2, s1, s0 // (op 681) 


s_cmp_eq_i32 s3, 0 // (op 681) 
s_cselect_b32 s0, s21, s2 // (op 681) 



v_lshl_add_u32 v3, s0, 4, v5 // (op 681) 


v_mul_lo_u32 v4, v3, s40 // (op 681) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 681) 


s_mul_i32 s9, s40, s5 // (op 683) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 683) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 683) least significant
s_mul_hi_u32 s1, s40, s4 // (op 683) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 683) most significant: sum
s_add_u32 s9, s9, s1 // (op 683) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 683) 

s_mov_b32 s28, s42 // (op 683) reg expression
s_mov_b32 s29, s43 // (op 683) reg expression
s_mov_b32 s30, s0 // (op 683) reg expression
s_mov_b32 s31, 131072 // (op 683) call()





buffer_load_dwordx4 v[4:7], v3, s[28:31], 0 offen // (op 135) Load value



v_accvgpr_read v10, a0 // (op 132) 
v_accvgpr_read v11, a1 // (op 132) 
v_accvgpr_read v12, a2 // (op 132) 
v_accvgpr_read v13, a3 // (op 132) 
v_mul_f32 v14, v9, v10 // (op 132) 
v_mul_f32 v15, v9, v11 // (op 132) 
v_mul_f32 v16, v9, v12 // (op 132) 
v_mul_f32 v17, v9, v13 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v8, v4 // (op 136) 
v_mul_f32 v11, v8, v5 // (op 136) 
v_mul_f32 v12, v8, v6 // (op 136) 
v_mul_f32 v13, v8, v7 // (op 136) 

v_add_f32 v4, v10, v14 // (op 133) 
v_add_f32 v5, v11, v15 // (op 133) 
v_add_f32 v6, v12, v16 // (op 133) 
v_add_f32 v7, v13, v17 // (op 133) 


s_mul_hi_i32 s0, s16, s12 // (op 876) 
v_lshrrev_b32 v0, 6, v1 // (op 876) 
s_add_i32 s1, s0, s16 // (op 876) 

v_and_b32 v3, 63, v1 // (op 876) 
s_ashr_i32 s0, s1, 31 // (op 876) 
v_lshrrev_b32 v10, 4, v3 // (op 876) 
s_cmp_eq_i32 s12, 0 // (op 876) 
s_mov_b32 s2, scc // (op 876) 
v_lshlrev_b32 v11, 2, v10 // (op 876) 

s_cmp_eq_i32 s2, 0 // (op 876) 
s_cselect_b32 s3, 0, -1 // (op 876) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 876) 


s_and_b32 s2, s13, 31 // (op 876) 
v_lshrrev_b32 v0, 2, v3 // (op 876) 

s_lshl_b32 s8, 1, s2 // (op 876) 
v_and_b32 v3, 3, v0 // (op 876) 

s_add_i32 s9, s8, s3 // (op 876) 


v_and_b32 v0, 3, v1 // (op 876) 
s_and_b32 s3, s0, s9 // (op 876) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 876) 


s_add_i32 s0, s1, s3 // (op 876) 


s_ashr_i32 s1, s0, s2 // (op 876) 


s_lshl_b32 s0, s13, 24 // (op 876) 
s_ashr_i32 s2, s0, 31 // (op 876) 

s_xor_b32 s0, s1, s2 // (op 876) 

s_sub_i32 s1, s0, s2 // (op 876) 


s_mul_hi_i32 s0, s1, s14 // (op 876) 
s_add_i32 s2, s0, s1 // (op 876) 

s_ashr_i32 s0, s2, 31 // (op 876) 
s_cmp_eq_i32 s14, 0 // (op 876) 
s_mov_b32 s3, scc // (op 876) 
s_cmp_eq_i32 s3, 0 // (op 876) 
s_cselect_b32 s8, 0, -1 // (op 876) 

s_and_b32 s3, s15, 31 // (op 876) 
s_lshl_b32 s9, 1, s3 // (op 876) 
s_add_i32 s18, s9, s8 // (op 876) 


s_and_b32 s8, s0, s18 // (op 876) 


s_add_i32 s0, s2, s8 // (op 876) 


s_ashr_i32 s2, s0, s3 // (op 876) 


s_lshl_b32 s0, s15, 24 // (op 876) 
s_ashr_i32 s3, s0, 31 // (op 876) 

s_xor_b32 s0, s2, s3 // (op 876) 

s_sub_i32 s2, s0, s3 // (op 876) 


s_mul_i32 s0, s1, s45 // (op 876) 
s_sub_u32 s3, s16, s0 // (op 876) 

s_mul_i32 s0, s2, s45 // (op 876) 
s_add_u32 s0, s0, s3 // (op 876) 

s_mul_hi_i32 s3, s16, s11 // (op 876) 
s_add_i32 s8, s3, s16 // (op 876) 

s_ashr_i32 s3, s8, 31 // (op 876) 
s_cmp_eq_i32 s11, 0 // (op 876) 
s_mov_b32 s9, scc // (op 876) 
s_cmp_eq_i32 s9, 0 // (op 876) 
s_cselect_b32 s18, 0, -1 // (op 876) 

s_and_b32 s9, s46, 31 // (op 876) 
s_lshl_b32 s19, 1, s9 // (op 876) 
s_add_i32 s20, s19, s18 // (op 876) 


s_and_b32 s18, s3, s20 // (op 876) 


s_add_i32 s3, s8, s18 // (op 876) 


s_ashr_i32 s8, s3, s9 // (op 876) 


s_lshl_b32 s3, s46, 24 // (op 876) 
s_ashr_i32 s9, s3, 31 // (op 876) 

s_xor_b32 s3, s8, s9 // (op 876) 

s_sub_i32 s8, s3, s9 // (op 876) 


s_add_u32 s18, s4, 16 // (op 876) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 876) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 876) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 876) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 4 // (op 876) 

s_mul_i32 s3, s45, s18 // (op 876) 
s_mul_i32 s9, s10, s3 // (op 876) 

s_mul_i32 s3, s8, s9 // (op 876) 

s_sub_u32 s9, s16, s3 // (op 876) 

s_mov_b32 s3, s9 // (op 876) 
s_mul_hi_i32 s20, s3, s44 // (op 876) 
s_add_i32 s21, s20, s3 // (op 876) 


s_ashr_i32 s3, s21, 31 // (op 876) 
s_cmp_eq_i32 s44, 0 // (op 876) 
s_mov_b32 s20, scc // (op 876) 
s_cmp_eq_i32 s20, 0 // (op 876) 
s_cselect_b32 s22, 0, -1 // (op 876) 

s_and_b32 s20, s48, 31 // (op 876) 
s_lshl_b32 s23, 1, s20 // (op 876) 
s_add_i32 s24, s23, s22 // (op 876) 


s_and_b32 s22, s3, s24 // (op 876) 


s_add_i32 s3, s21, s22 // (op 876) 


s_ashr_i32 s21, s3, s20 // (op 876) 


s_lshl_b32 s3, s48, 24 // (op 876) 
s_ashr_i32 s20, s3, 31 // (op 876) 

s_xor_b32 s3, s21, s20 // (op 876) 

s_sub_i32 s21, s3, s20 // (op 876) 


s_mul_i32 s3, s47, s45 // (op 876) 
s_add_u32 s22, s6, 32 // (op 876) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 876) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 876) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 876) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 876) 

s_sub_i32 s20, s22, s3 // (op 876) 

s_mul_i32 s22, s21, s20 // (op 876) 

s_sub_u32 s20, s9, s22 // (op 876) 


s_add_u32 s9, s20, s3 // (op 876) 


s_cmp_eq_i32 s8, 0 // (op 876) 
s_mov_b32 s3, scc // (op 876) 

s_cmp_eq_i32 s3, 0 // (op 876) 
s_cselect_b32 s8, s9, s0 // (op 876) 


s_mul_i32 s0, s2, s18 // (op 876) 


v_lshl_add_u32 v0, s8, 5, v10 // (op 876) 


s_sub_i32 s2, s1, s0 // (op 876) 


s_cmp_eq_i32 s3, 0 // (op 876) 
s_cselect_b32 s0, s21, s2 // (op 876) 



v_lshl_add_u32 v3, s0, 4, v11 // (op 876) 


v_mul_lo_u32 v10, v3, s36 // (op 876) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 876) 


s_mul_i32 s9, s36, s5 // (op 878) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 878) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 878) least significant
s_mul_hi_u32 s1, s36, s4 // (op 878) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 878) most significant: sum
s_add_u32 s9, s9, s1 // (op 878) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 2 // (op 878) 

s_mov_b32 s32, s38 // (op 878) reg expression
s_mov_b32 s33, s39 // (op 878) reg expression
s_mov_b32 s34, s0 // (op 878) reg expression
s_mov_b32 s35, 131072 // (op 878) call()




buffer_store_dwordx4 v[4:7], v3, s[32:35], 0 offen // (op 134) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 6400 // lds bytes
  .amdhsa_accum_offset 52
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
    args: [ { type: int64, semantic: StrideA0 },
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: SizeFree0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail } ]
    macrotile: [32, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 6400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 49
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
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
      - .name: user_Float_Value_14
        .size: 4
        .offset: 24
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 28
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 32
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 40
        .expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
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
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 128
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 136
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 144
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicMultiple_30
        .size: 4
        .offset: 152
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 156
        .expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 160
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 164
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 168
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

