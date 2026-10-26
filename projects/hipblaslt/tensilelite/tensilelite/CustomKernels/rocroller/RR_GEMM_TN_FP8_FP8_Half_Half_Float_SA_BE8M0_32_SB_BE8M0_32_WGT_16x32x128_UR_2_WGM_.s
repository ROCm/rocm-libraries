

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 799) 
v_and_b32 v3, 15, v1 // (op 799) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 799) 


v_and_b32 v0, 48, v1 // (op 799) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 799) 


v_lshrrev_b32 v0, 6, v1 // (op 851) 
v_and_b32 v4, 1, v0 // (op 851) 

v_and_b32 v0, 15, v1 // (op 851) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 851) 


v_and_b32 v0, 48, v1 // (op 851) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 851) 


v_lshrrev_b32 v0, 6, v1 // (op 904) 
v_and_b32 v5, 1, v0 // (op 904) 

v_and_b32 v0, 15, v1 // (op 904) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 904) 


v_and_b32 v0, 63, v1 // (op 904) 
v_lshrrev_b32 v5, 4, v0 // (op 904) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 904) 


v_lshrrev_b32 v5, 7, v1 // (op 977) 
v_and_b32 v6, 15, v1 // (op 977) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 977) 


v_and_b32 v5, 48, v1 // (op 977) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 977) 


v_lshrrev_b32 v5, 6, v1 // (op 1016) 
v_and_b32 v7, 1, v5 // (op 1016) 

v_and_b32 v5, 15, v1 // (op 1016) 
v_lshl_add_u32 v8, v7, 4, v5 // (op 1016) 


v_and_b32 v5, 48, v1 // (op 1016) 
v_lshl_add_u32 v7, v8, 7, v5 // (op 1016) 


v_lshrrev_b32 v5, 6, v1 // (op 1040) 
v_and_b32 v8, 1, v5 // (op 1040) 

v_and_b32 v5, 15, v1 // (op 1040) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 1040) 


v_and_b32 v5, 63, v1 // (op 1040) 
v_lshrrev_b32 v8, 4, v5 // (op 1040) 

v_lshl_add_u32 v5, v9, 2, v8 // (op 1040) 


v_mov_b32 v8, s8 // (op 2) Move value
v_mov_b32 v9, s9 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
s_mul_hi_i32 s0, s16, s12 // (op 510) 
v_lshrrev_b32 v10, 3, v1 // (op 510) 
s_add_i32 s1, s0, s16 // (op 510) 

v_and_b32 v11, -8, v1 // (op 510) 
s_ashr_i32 s0, s1, 31 // (op 510) 
v_sub_u32 v12, v1, v11 // (op 510) 

s_cmp_eq_i32 s12, 0 // (op 510) 
s_mov_b32 s8, scc // (op 510) 
v_lshlrev_b32 v11, 4, v12 // (op 510) 

s_cmp_eq_i32 s8, 0 // (op 510) 
s_cselect_b32 s9, 0, -1 // (op 510) 

v_mov_b32 v13, 0 // (op 510) convert
v_mov_b32 v12, v11 // (op 510) convert

s_and_b32 s8, s13, 31 // (op 510) 
s_lshl_b32 s18, 1, s8 // (op 510) 
s_add_i32 s19, s18, s9 // (op 510) 


s_and_b32 s9, s0, s19 // (op 510) 


s_add_i32 s0, s1, s9 // (op 510) 


s_ashr_i32 s1, s0, s8 // (op 510) 


s_lshl_b32 s0, s13, 24 // (op 510) 
s_ashr_i32 s8, s0, 31 // (op 510) 

s_xor_b32 s0, s1, s8 // (op 510) 

s_sub_i32 s1, s0, s8 // (op 510) 


s_mul_hi_i32 s0, s1, s14 // (op 510) 
s_add_i32 s8, s0, s1 // (op 510) 

s_ashr_i32 s0, s8, 31 // (op 510) 
s_cmp_eq_i32 s14, 0 // (op 510) 
s_mov_b32 s9, scc // (op 510) 
s_cmp_eq_i32 s9, 0 // (op 510) 
s_cselect_b32 s18, 0, -1 // (op 510) 

s_and_b32 s9, s15, 31 // (op 510) 
s_lshl_b32 s19, 1, s9 // (op 510) 
s_add_i32 s49, s19, s18 // (op 510) 


s_and_b32 s18, s0, s49 // (op 510) 


s_add_i32 s0, s8, s18 // (op 510) 


s_ashr_i32 s8, s0, s9 // (op 510) 


s_lshl_b32 s0, s15, 24 // (op 510) 
s_ashr_i32 s9, s0, 31 // (op 510) 

s_xor_b32 s0, s8, s9 // (op 510) 

s_sub_i32 s8, s0, s9 // (op 510) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s0, s1, s45 // (op 510) 

s_sub_u32 s1, s16, s0 // (op 510) 

s_mul_i32 s0, s8, s45 // (op 510) 
s_add_u32 s0, s0, s1 // (op 510) 


s_mul_hi_i32 s1, s16, s11 // (op 510) 
s_add_i32 s8, s1, s16 // (op 510) 

s_ashr_i32 s1, s8, 31 // (op 510) 
s_cmp_eq_i32 s11, 0 // (op 510) 
s_mov_b32 s9, scc // (op 510) 
s_cmp_eq_i32 s9, 0 // (op 510) 
s_cselect_b32 s18, 0, -1 // (op 510) 

s_and_b32 s9, s46, 31 // (op 510) 
s_lshl_b32 s19, 1, s9 // (op 510) 
s_add_i32 s49, s19, s18 // (op 510) 


s_and_b32 s18, s1, s49 // (op 510) 


s_add_i32 s1, s8, s18 // (op 510) 


s_ashr_i32 s8, s1, s9 // (op 510) 


s_lshl_b32 s1, s46, 24 // (op 510) 
s_ashr_i32 s9, s1, 31 // (op 510) 

s_xor_b32 s1, s8, s9 // (op 510) 

s_sub_i32 s8, s1, s9 // (op 510) 


s_add_u32 s18, s4, 32 // (op 510) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 510) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 510) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 510) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 510) 

s_mul_i32 s1, s45, s18 // (op 510) 

s_mul_i32 s9, s10, s1 // (op 510) 

s_mul_i32 s1, s8, s9 // (op 510) 

s_sub_u32 s9, s16, s1 // (op 510) 

s_mov_b32 s1, s9 // (op 510) 
s_mul_hi_i32 s18, s1, s44 // (op 510) 
s_add_i32 s19, s18, s1 // (op 510) 


s_ashr_i32 s1, s19, 31 // (op 510) 
s_cmp_eq_i32 s44, 0 // (op 510) 
s_mov_b32 s18, scc // (op 510) 
s_cmp_eq_i32 s18, 0 // (op 510) 
s_cselect_b32 s49, 0, -1 // (op 510) 

s_and_b32 s18, s48, 31 // (op 510) 
s_lshl_b32 s50, 1, s18 // (op 510) 
s_add_i32 s51, s50, s49 // (op 510) 


s_and_b32 s49, s1, s51 // (op 510) 


s_add_i32 s1, s19, s49 // (op 510) 


s_ashr_i32 s19, s1, s18 // (op 510) 


s_lshl_b32 s1, s48, 24 // (op 510) 
s_ashr_i32 s18, s1, 31 // (op 510) 

s_xor_b32 s1, s19, s18 // (op 510) 

s_sub_i32 s19, s1, s18 // (op 510) 


s_mul_i32 s1, s47, s45 // (op 510) 
s_add_u32 s50, s6, 16 // (op 510) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 510) End of Int64 add, unlocking SCC
s_sub_u32 s52, s50, 1 // (op 510) least significant half; sets scc
s_subb_u32 s53, s51, 0 // (op 510) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 4 // (op 510) 

s_sub_i32 s18, s50, s1 // (op 510) 

s_mul_i32 s49, s19, s18 // (op 510) 


s_sub_u32 s18, s9, s49 // (op 510) 


s_add_u32 s9, s18, s1 // (op 510) 


s_cmp_eq_i32 s8, 0 // (op 510) 
s_mov_b32 s1, scc // (op 510) 

s_cmp_eq_i32 s1, 0 // (op 510) 
s_cselect_b32 s8, s9, s0 // (op 510) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 510) 


v_mul_lo_u32 v15, v11, s3 // (op 510) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 510) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 510) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 510) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 510) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 510) most significant half



s_mul_i32 s1, s2, s7 // (op 512) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 512) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 512) least significant
s_mul_hi_u32 s9, s2, s6 // (op 512) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 512) most significant: sum
s_add_u32 s1, s1, s9 // (op 512) most significant: sum


s_mov_b32 s52, s20 // (op 512) reg expression
s_mov_b32 s53, s21 // (op 512) reg expression
s_mov_b32 s54, s0 // (op 512) reg expression
s_mov_b32 s55, 131072 // (op 512) call()

s_mul_hi_i32 s0, s16, s12 // (op 529) 
v_lshrrev_b32 v10, 3, v1 // (op 529) 
s_add_i32 s1, s0, s16 // (op 529) 

v_and_b32 v11, -8, v1 // (op 529) 
s_ashr_i32 s0, s1, 31 // (op 529) 
v_sub_u32 v12, v1, v11 // (op 529) 

s_cmp_eq_i32 s12, 0 // (op 529) 
s_mov_b32 s8, scc // (op 529) 
v_lshlrev_b32 v11, 4, v12 // (op 529) 

s_cmp_eq_i32 s8, 0 // (op 529) 
s_cselect_b32 s9, 0, -1 // (op 529) 

v_mov_b32 v13, 0 // (op 529) convert
v_mov_b32 v12, v11 // (op 529) convert

s_and_b32 s8, s13, 31 // (op 529) 
s_lshl_b32 s18, 1, s8 // (op 529) 
s_add_i32 s19, s18, s9 // (op 529) 


s_and_b32 s9, s0, s19 // (op 529) 


s_add_i32 s0, s1, s9 // (op 529) 


s_ashr_i32 s1, s0, s8 // (op 529) 


s_lshl_b32 s0, s13, 24 // (op 529) 
s_ashr_i32 s8, s0, 31 // (op 529) 

s_xor_b32 s0, s1, s8 // (op 529) 

s_sub_i32 s1, s0, s8 // (op 529) 


s_mul_hi_i32 s0, s1, s14 // (op 529) 
s_add_i32 s8, s0, s1 // (op 529) 

s_ashr_i32 s0, s8, 31 // (op 529) 
s_cmp_eq_i32 s14, 0 // (op 529) 
s_mov_b32 s9, scc // (op 529) 
s_cmp_eq_i32 s9, 0 // (op 529) 
s_cselect_b32 s18, 0, -1 // (op 529) 

s_and_b32 s9, s15, 31 // (op 529) 
s_lshl_b32 s19, 1, s9 // (op 529) 
s_add_i32 s49, s19, s18 // (op 529) 


s_and_b32 s18, s0, s49 // (op 529) 


s_add_i32 s0, s8, s18 // (op 529) 


s_ashr_i32 s8, s0, s9 // (op 529) 


s_lshl_b32 s0, s15, 24 // (op 529) 
s_ashr_i32 s9, s0, 31 // (op 529) 

s_xor_b32 s0, s8, s9 // (op 529) 

s_sub_i32 s8, s0, s9 // (op 529) 


s_mul_i32 s0, s1, s45 // (op 529) 

s_sub_u32 s1, s16, s0 // (op 529) 

s_mul_i32 s0, s8, s45 // (op 529) 
s_add_u32 s0, s0, s1 // (op 529) 


s_mul_hi_i32 s1, s16, s11 // (op 529) 
s_add_i32 s8, s1, s16 // (op 529) 

s_ashr_i32 s1, s8, 31 // (op 529) 
s_cmp_eq_i32 s11, 0 // (op 529) 
s_mov_b32 s9, scc // (op 529) 
s_cmp_eq_i32 s9, 0 // (op 529) 
s_cselect_b32 s18, 0, -1 // (op 529) 

s_and_b32 s9, s46, 31 // (op 529) 
s_lshl_b32 s19, 1, s9 // (op 529) 
s_add_i32 s49, s19, s18 // (op 529) 


s_and_b32 s18, s1, s49 // (op 529) 


s_add_i32 s1, s8, s18 // (op 529) 


s_ashr_i32 s8, s1, s9 // (op 529) 


s_lshl_b32 s1, s46, 24 // (op 529) 
s_ashr_i32 s9, s1, 31 // (op 529) 

s_xor_b32 s1, s8, s9 // (op 529) 

s_sub_i32 s8, s1, s9 // (op 529) 


s_add_u32 s18, s4, 32 // (op 529) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 529) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 529) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 529) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 529) 

s_mul_i32 s1, s45, s18 // (op 529) 

s_mul_i32 s9, s10, s1 // (op 529) 

s_mul_i32 s1, s8, s9 // (op 529) 

s_sub_u32 s9, s16, s1 // (op 529) 

s_mov_b32 s1, s9 // (op 529) 
s_mul_hi_i32 s18, s1, s44 // (op 529) 
s_add_i32 s19, s18, s1 // (op 529) 


s_ashr_i32 s1, s19, 31 // (op 529) 
s_cmp_eq_i32 s44, 0 // (op 529) 
s_mov_b32 s18, scc // (op 529) 
s_cmp_eq_i32 s18, 0 // (op 529) 
s_cselect_b32 s49, 0, -1 // (op 529) 

s_and_b32 s18, s48, 31 // (op 529) 
s_lshl_b32 s50, 1, s18 // (op 529) 
s_add_i32 s51, s50, s49 // (op 529) 


s_and_b32 s49, s1, s51 // (op 529) 


s_add_i32 s1, s19, s49 // (op 529) 


s_ashr_i32 s19, s1, s18 // (op 529) 


s_lshl_b32 s1, s48, 24 // (op 529) 
s_ashr_i32 s18, s1, 31 // (op 529) 

s_xor_b32 s1, s19, s18 // (op 529) 

s_sub_i32 s19, s1, s18 // (op 529) 


s_mul_i32 s1, s47, s45 // (op 529) 
s_add_u32 s50, s6, 16 // (op 529) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 529) End of Int64 add, unlocking SCC
s_sub_u32 s56, s50, 1 // (op 529) least significant half; sets scc
s_subb_u32 s57, s51, 0 // (op 529) most significant half; uses scc

s_ashr_i64 s[50:51], s[56:57], 4 // (op 529) 

s_sub_i32 s18, s50, s1 // (op 529) 

s_mul_i32 s49, s19, s18 // (op 529) 


s_sub_u32 s18, s9, s49 // (op 529) 


s_add_u32 s9, s18, s1 // (op 529) 


s_cmp_eq_i32 s8, 0 // (op 529) 
s_mov_b32 s1, scc // (op 529) 

s_cmp_eq_i32 s1, 0 // (op 529) 
s_cselect_b32 s8, s9, s0 // (op 529) 



v_lshl_add_u32 v11, s8, 4, v10 // (op 529) 


v_mul_lo_u32 v17, v11, s3 // (op 529) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 529) most significant: high of low * low
v_mul_lo_u32 v16, v11, s2 // (op 529) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 529) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 529) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 529) most significant half



s_mul_i32 s1, s2, s7 // (op 531) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 531) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 531) least significant
s_mul_hi_u32 s9, s2, s6 // (op 531) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 531) most significant: sum
s_add_u32 s1, s1, s9 // (op 531) most significant: sum


s_mov_b32 s52, s20 // (op 531) reg expression
s_mov_b32 s53, s21 // (op 531) reg expression
s_mov_b32 s54, s0 // (op 531) reg expression
s_mov_b32 s55, 131072 // (op 531) call()

s_mul_hi_i32 s0, s16, s12 // (op 574) 
v_lshrrev_b32 v10, 3, v1 // (op 574) 
s_add_i32 s1, s0, s16 // (op 574) 

v_and_b32 v11, -8, v1 // (op 574) 
s_ashr_i32 s0, s1, 31 // (op 574) 
v_sub_u32 v12, v1, v11 // (op 574) 

s_cmp_eq_i32 s12, 0 // (op 574) 
s_mov_b32 s8, scc // (op 574) 
v_lshlrev_b32 v11, 4, v12 // (op 574) 

s_cmp_eq_i32 s8, 0 // (op 574) 
s_cselect_b32 s9, 0, -1 // (op 574) 

v_mov_b32 v13, 0 // (op 574) convert
v_mov_b32 v12, v11 // (op 574) convert

s_and_b32 s8, s13, 31 // (op 574) 
s_lshl_b32 s18, 1, s8 // (op 574) 
s_add_i32 s19, s18, s9 // (op 574) 


s_and_b32 s9, s0, s19 // (op 574) 


s_add_i32 s0, s1, s9 // (op 574) 


s_ashr_i32 s1, s0, s8 // (op 574) 


s_lshl_b32 s0, s13, 24 // (op 574) 
s_ashr_i32 s8, s0, 31 // (op 574) 

s_xor_b32 s0, s1, s8 // (op 574) 

s_sub_i32 s1, s0, s8 // (op 574) 


s_mul_hi_i32 s0, s1, s14 // (op 574) 
s_add_i32 s8, s0, s1 // (op 574) 

s_ashr_i32 s0, s8, 31 // (op 574) 
s_cmp_eq_i32 s14, 0 // (op 574) 
s_mov_b32 s9, scc // (op 574) 
s_cmp_eq_i32 s9, 0 // (op 574) 
s_cselect_b32 s18, 0, -1 // (op 574) 

s_and_b32 s9, s15, 31 // (op 574) 
s_lshl_b32 s19, 1, s9 // (op 574) 
s_add_i32 s49, s19, s18 // (op 574) 


s_and_b32 s18, s0, s49 // (op 574) 


s_add_i32 s0, s8, s18 // (op 574) 


s_ashr_i32 s8, s0, s9 // (op 574) 


s_lshl_b32 s0, s15, 24 // (op 574) 
s_ashr_i32 s9, s0, 31 // (op 574) 

s_xor_b32 s0, s8, s9 // (op 574) 

s_sub_i32 s8, s0, s9 // (op 574) 


s_add_u32 s18, s4, 32 // (op 574) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 574) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 574) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 574) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 574) 

s_mul_i32 s0, s8, s18 // (op 574) 

s_sub_i32 s8, s1, s0 // (op 574) 


s_mul_hi_i32 s9, s16, s11 // (op 574) 
s_add_i32 s0, s9, s16 // (op 574) 

s_ashr_i32 s1, s0, 31 // (op 574) 
s_cmp_eq_i32 s11, 0 // (op 574) 
s_mov_b32 s9, scc // (op 574) 
s_cmp_eq_i32 s9, 0 // (op 574) 
s_cselect_b32 s49, 0, -1 // (op 574) 

s_and_b32 s9, s46, 31 // (op 574) 
s_lshl_b32 s50, 1, s9 // (op 574) 
s_add_i32 s51, s50, s49 // (op 574) 


s_and_b32 s49, s1, s51 // (op 574) 


s_add_i32 s1, s0, s49 // (op 574) 


s_ashr_i32 s0, s1, s9 // (op 574) 


s_lshl_b32 s1, s46, 24 // (op 574) 
s_ashr_i32 s9, s1, 31 // (op 574) 

s_xor_b32 s1, s0, s9 // (op 574) 

s_sub_i32 s0, s1, s9 // (op 574) 


s_mul_i32 s1, s45, s18 // (op 574) 

s_mul_i32 s9, s10, s1 // (op 574) 

s_mul_i32 s1, s0, s9 // (op 574) 

s_sub_u32 s9, s16, s1 // (op 574) 

s_mul_hi_i32 s1, s9, s44 // (op 574) 
s_add_i32 s18, s1, s9 // (op 574) 


s_ashr_i32 s1, s18, 31 // (op 574) 
s_cmp_eq_i32 s44, 0 // (op 574) 
s_mov_b32 s9, scc // (op 574) 
s_cmp_eq_i32 s9, 0 // (op 574) 
s_cselect_b32 s19, 0, -1 // (op 574) 

s_and_b32 s9, s48, 31 // (op 574) 
s_lshl_b32 s49, 1, s9 // (op 574) 
s_add_i32 s50, s49, s19 // (op 574) 


s_and_b32 s19, s1, s50 // (op 574) 


s_add_i32 s1, s18, s19 // (op 574) 


s_ashr_i32 s18, s1, s9 // (op 574) 


s_lshl_b32 s1, s48, 24 // (op 574) 
s_ashr_i32 s9, s1, 31 // (op 574) 

s_xor_b32 s1, s18, s9 // (op 574) 

s_sub_i32 s18, s1, s9 // (op 574) 


s_cmp_eq_i32 s0, 0 // (op 574) 
s_mov_b32 s1, scc // (op 574) 

s_cmp_eq_i32 s1, 0 // (op 574) 
s_cselect_b32 s0, s18, s8 // (op 574) 



v_lshl_add_u32 v11, s0, 5, v10 // (op 574) 


v_mul_lo_u32 v19, v11, s23 // (op 574) most significant: low of low * high
v_mul_hi_u32 v10, v11, s22 // (op 574) most significant: high of low * low
v_mul_lo_u32 v18, v11, s22 // (op 574) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 574) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 574) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 574) most significant half



s_mul_i32 s1, s22, s5 // (op 576) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 576) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 576) least significant
s_mul_hi_u32 s9, s22, s4 // (op 576) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 576) most significant: sum
s_add_u32 s1, s1, s9 // (op 576) most significant: sum


s_mov_b32 s56, s24 // (op 576) reg expression
s_mov_b32 s57, s25 // (op 576) reg expression
s_mov_b32 s58, s0 // (op 576) reg expression
s_mov_b32 s59, 131072 // (op 576) call()

s_mul_hi_i32 s0, s16, s12 // (op 593) 
v_lshrrev_b32 v12, 3, v1 // (op 593) 
s_add_i32 s1, s0, s16 // (op 593) 

v_and_b32 v13, -8, v1 // (op 593) 
s_ashr_i32 s0, s1, 31 // (op 593) 
v_sub_u32 v18, v1, v13 // (op 593) 

s_cmp_eq_i32 s12, 0 // (op 593) 
s_mov_b32 s8, scc // (op 593) 
v_lshlrev_b32 v13, 4, v18 // (op 593) 

s_cmp_eq_i32 s8, 0 // (op 593) 
s_cselect_b32 s9, 0, -1 // (op 593) 

v_mov_b32 v19, 0 // (op 593) convert
v_mov_b32 v18, v13 // (op 593) convert

s_and_b32 s8, s13, 31 // (op 593) 
s_lshl_b32 s18, 1, s8 // (op 593) 
s_add_i32 s19, s18, s9 // (op 593) 


s_and_b32 s9, s0, s19 // (op 593) 


s_add_i32 s0, s1, s9 // (op 593) 


s_ashr_i32 s1, s0, s8 // (op 593) 


s_lshl_b32 s0, s13, 24 // (op 593) 
s_ashr_i32 s8, s0, 31 // (op 593) 

s_xor_b32 s0, s1, s8 // (op 593) 

s_sub_i32 s1, s0, s8 // (op 593) 


s_mul_hi_i32 s0, s1, s14 // (op 593) 
s_add_i32 s8, s0, s1 // (op 593) 

s_ashr_i32 s0, s8, 31 // (op 593) 
s_cmp_eq_i32 s14, 0 // (op 593) 
s_mov_b32 s9, scc // (op 593) 
s_cmp_eq_i32 s9, 0 // (op 593) 
s_cselect_b32 s18, 0, -1 // (op 593) 

s_and_b32 s9, s15, 31 // (op 593) 
s_lshl_b32 s19, 1, s9 // (op 593) 
s_add_i32 s49, s19, s18 // (op 593) 


s_and_b32 s18, s0, s49 // (op 593) 


s_add_i32 s0, s8, s18 // (op 593) 


s_ashr_i32 s8, s0, s9 // (op 593) 


s_lshl_b32 s0, s15, 24 // (op 593) 
s_ashr_i32 s9, s0, 31 // (op 593) 

s_xor_b32 s0, s8, s9 // (op 593) 

s_sub_i32 s8, s0, s9 // (op 593) 


s_add_u32 s18, s4, 32 // (op 593) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 593) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 593) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 593) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 593) 

s_mul_i32 s0, s8, s18 // (op 593) 

s_sub_i32 s8, s1, s0 // (op 593) 


s_mul_hi_i32 s9, s16, s11 // (op 593) 
s_add_i32 s0, s9, s16 // (op 593) 

s_ashr_i32 s1, s0, 31 // (op 593) 
s_cmp_eq_i32 s11, 0 // (op 593) 
s_mov_b32 s9, scc // (op 593) 
s_cmp_eq_i32 s9, 0 // (op 593) 
s_cselect_b32 s49, 0, -1 // (op 593) 

s_and_b32 s9, s46, 31 // (op 593) 
s_lshl_b32 s50, 1, s9 // (op 593) 
s_add_i32 s51, s50, s49 // (op 593) 


s_and_b32 s49, s1, s51 // (op 593) 


s_add_i32 s1, s0, s49 // (op 593) 


s_ashr_i32 s0, s1, s9 // (op 593) 


s_lshl_b32 s1, s46, 24 // (op 593) 
s_ashr_i32 s9, s1, 31 // (op 593) 

s_xor_b32 s1, s0, s9 // (op 593) 

s_sub_i32 s0, s1, s9 // (op 593) 


s_mul_i32 s1, s45, s18 // (op 593) 

s_mul_i32 s9, s10, s1 // (op 593) 

s_mul_i32 s1, s0, s9 // (op 593) 

s_sub_u32 s9, s16, s1 // (op 593) 

s_mul_hi_i32 s1, s9, s44 // (op 593) 
s_add_i32 s18, s1, s9 // (op 593) 


s_ashr_i32 s1, s18, 31 // (op 593) 
s_cmp_eq_i32 s44, 0 // (op 593) 
s_mov_b32 s9, scc // (op 593) 
s_cmp_eq_i32 s9, 0 // (op 593) 
s_cselect_b32 s19, 0, -1 // (op 593) 

s_and_b32 s9, s48, 31 // (op 593) 
s_lshl_b32 s49, 1, s9 // (op 593) 
s_add_i32 s50, s49, s19 // (op 593) 


s_and_b32 s19, s1, s50 // (op 593) 


s_add_i32 s1, s18, s19 // (op 593) 


s_ashr_i32 s18, s1, s9 // (op 593) 


s_lshl_b32 s1, s48, 24 // (op 593) 
s_ashr_i32 s9, s1, 31 // (op 593) 

s_xor_b32 s1, s18, s9 // (op 593) 

s_sub_i32 s18, s1, s9 // (op 593) 


s_cmp_eq_i32 s0, 0 // (op 593) 
s_mov_b32 s1, scc // (op 593) 

s_cmp_eq_i32 s1, 0 // (op 593) 
s_cselect_b32 s0, s18, s8 // (op 593) 



v_lshl_add_u32 v13, s0, 5, v12 // (op 593) 


v_mul_lo_u32 v21, v13, s23 // (op 593) most significant: low of low * high
v_mul_hi_u32 v12, v13, s22 // (op 593) most significant: high of low * low
v_mul_lo_u32 v20, v13, s22 // (op 593) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 593) most significant: sum


v_add_co_u32 v12, s[0:1], v18, v20 // (op 593) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 593) most significant half



s_mul_i32 s1, s22, s5 // (op 595) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 595) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 595) least significant
s_mul_hi_u32 s9, s22, s4 // (op 595) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 595) most significant: sum
s_add_u32 s1, s1, s9 // (op 595) most significant: sum


s_mov_b32 s56, s24 // (op 595) reg expression
s_mov_b32 s57, s25 // (op 595) reg expression
s_mov_b32 s58, s0 // (op 595) reg expression
s_mov_b32 s59, 131072 // (op 595) call()

s_mul_hi_i32 s0, s16, s12 // (op 638) 
v_lshrrev_b32 v18, 7, v1 // (op 638) 
s_add_i32 s1, s0, s16 // (op 638) 

v_and_b32 v19, 15, v1 // (op 638) 
s_ashr_i32 s0, s1, 31 // (op 638) 
v_lshl_add_u32 v20, v18, 4, v19 // (op 638) 


s_cmp_eq_i32 s12, 0 // (op 638) 
s_mov_b32 s8, scc // (op 638) 
v_and_b32 v18, 63, v1 // (op 638) 
s_cmp_eq_i32 s8, 0 // (op 638) 
s_cselect_b32 s9, 0, -1 // (op 638) 

v_lshrrev_b32 v19, 4, v18 // (op 638) 

s_and_b32 s8, s13, 31 // (op 638) 
s_lshl_b32 s18, 1, s8 // (op 638) 
s_add_i32 s19, s18, s9 // (op 638) 


s_and_b32 s9, s0, s19 // (op 638) 


s_add_i32 s0, s1, s9 // (op 638) 


s_ashr_i32 s1, s0, s8 // (op 638) 


s_lshl_b32 s0, s13, 24 // (op 638) 
s_ashr_i32 s8, s0, 31 // (op 638) 

s_xor_b32 s0, s1, s8 // (op 638) 

s_sub_i32 s1, s0, s8 // (op 638) 


s_mul_hi_i32 s0, s1, s14 // (op 638) 
s_add_i32 s8, s0, s1 // (op 638) 

s_ashr_i32 s0, s8, 31 // (op 638) 
s_cmp_eq_i32 s14, 0 // (op 638) 
s_mov_b32 s9, scc // (op 638) 
s_cmp_eq_i32 s9, 0 // (op 638) 
s_cselect_b32 s18, 0, -1 // (op 638) 

s_and_b32 s9, s15, 31 // (op 638) 
s_lshl_b32 s19, 1, s9 // (op 638) 
s_add_i32 s49, s19, s18 // (op 638) 


s_and_b32 s18, s0, s49 // (op 638) 


s_add_i32 s0, s8, s18 // (op 638) 


s_ashr_i32 s8, s0, s9 // (op 638) 


s_lshl_b32 s0, s15, 24 // (op 638) 
s_ashr_i32 s9, s0, 31 // (op 638) 

s_xor_b32 s0, s8, s9 // (op 638) 

s_sub_i32 s8, s0, s9 // (op 638) 


s_mul_i32 s0, s1, s45 // (op 638) 

s_sub_u32 s1, s16, s0 // (op 638) 

s_mul_i32 s0, s8, s45 // (op 638) 
s_add_u32 s0, s0, s1 // (op 638) 


s_mul_hi_i32 s1, s16, s11 // (op 638) 
s_add_i32 s8, s1, s16 // (op 638) 

s_ashr_i32 s1, s8, 31 // (op 638) 
s_cmp_eq_i32 s11, 0 // (op 638) 
s_mov_b32 s9, scc // (op 638) 
s_cmp_eq_i32 s9, 0 // (op 638) 
s_cselect_b32 s18, 0, -1 // (op 638) 

s_and_b32 s9, s46, 31 // (op 638) 
s_lshl_b32 s19, 1, s9 // (op 638) 
s_add_i32 s49, s19, s18 // (op 638) 


s_and_b32 s18, s1, s49 // (op 638) 


s_add_i32 s1, s8, s18 // (op 638) 


s_ashr_i32 s8, s1, s9 // (op 638) 


s_lshl_b32 s1, s46, 24 // (op 638) 
s_ashr_i32 s9, s1, 31 // (op 638) 

s_xor_b32 s1, s8, s9 // (op 638) 

s_sub_i32 s8, s1, s9 // (op 638) 


s_add_u32 s18, s4, 32 // (op 638) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 638) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 638) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 638) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 638) 

s_mul_i32 s1, s45, s18 // (op 638) 

s_mul_i32 s9, s10, s1 // (op 638) 

s_mul_i32 s1, s8, s9 // (op 638) 

s_sub_u32 s9, s16, s1 // (op 638) 

s_mov_b32 s1, s9 // (op 638) 
s_mul_hi_i32 s18, s1, s44 // (op 638) 
s_add_i32 s19, s18, s1 // (op 638) 


s_ashr_i32 s1, s19, 31 // (op 638) 
s_cmp_eq_i32 s44, 0 // (op 638) 
s_mov_b32 s18, scc // (op 638) 
s_cmp_eq_i32 s18, 0 // (op 638) 
s_cselect_b32 s49, 0, -1 // (op 638) 

s_and_b32 s18, s48, 31 // (op 638) 
s_lshl_b32 s50, 1, s18 // (op 638) 
s_add_i32 s51, s50, s49 // (op 638) 


s_and_b32 s49, s1, s51 // (op 638) 


s_add_i32 s1, s19, s49 // (op 638) 


s_ashr_i32 s19, s1, s18 // (op 638) 


s_lshl_b32 s1, s48, 24 // (op 638) 
s_ashr_i32 s18, s1, 31 // (op 638) 

s_xor_b32 s1, s19, s18 // (op 638) 

s_sub_i32 s19, s1, s18 // (op 638) 


s_mul_i32 s1, s47, s45 // (op 638) 
s_add_u32 s50, s6, 16 // (op 638) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 638) End of Int64 add, unlocking SCC
s_sub_u32 s60, s50, 1 // (op 638) least significant half; sets scc
s_subb_u32 s61, s51, 0 // (op 638) most significant half; uses scc

s_ashr_i64 s[50:51], s[60:61], 4 // (op 638) 

s_sub_i32 s18, s50, s1 // (op 638) 

s_mul_i32 s49, s19, s18 // (op 638) 


s_sub_u32 s18, s9, s49 // (op 638) 


s_add_u32 s9, s18, s1 // (op 638) 


s_cmp_eq_i32 s8, 0 // (op 638) 
s_mov_b32 s1, scc // (op 638) 

s_cmp_eq_i32 s1, 0 // (op 638) 
s_cselect_b32 s8, s9, s0 // (op 638) 



v_lshl_add_u32 v18, s8, 4, v20 // (op 638) 


v_mul_lo_u32 v20, v18, s26 // (op 638) 
v_add_u32 v20, v20, v19 // (op 638) 


s_mul_i32 s1, s26, s7 // (op 640) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 640) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 640) least significant
s_mul_hi_u32 s9, s26, s6 // (op 640) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 640) most significant: sum
s_add_u32 s1, s1, s9 // (op 640) most significant: sum


s_mov_b32 s60, s28 // (op 640) reg expression
s_mov_b32 s61, s29 // (op 640) reg expression
s_mov_b32 s62, s0 // (op 640) reg expression
s_mov_b32 s63, 131072 // (op 640) call()

s_mul_hi_i32 s0, s16, s12 // (op 657) 
v_lshrrev_b32 v18, 7, v1 // (op 657) 
s_add_i32 s1, s0, s16 // (op 657) 

v_and_b32 v19, 15, v1 // (op 657) 
s_ashr_i32 s0, s1, 31 // (op 657) 
v_lshl_add_u32 v21, v18, 4, v19 // (op 657) 


s_cmp_eq_i32 s12, 0 // (op 657) 
s_mov_b32 s8, scc // (op 657) 
v_and_b32 v18, 63, v1 // (op 657) 
s_cmp_eq_i32 s8, 0 // (op 657) 
s_cselect_b32 s9, 0, -1 // (op 657) 

v_lshrrev_b32 v19, 4, v18 // (op 657) 

s_and_b32 s8, s13, 31 // (op 657) 
s_lshl_b32 s18, 1, s8 // (op 657) 
s_add_i32 s19, s18, s9 // (op 657) 


s_and_b32 s9, s0, s19 // (op 657) 


s_add_i32 s0, s1, s9 // (op 657) 


s_ashr_i32 s1, s0, s8 // (op 657) 


s_lshl_b32 s0, s13, 24 // (op 657) 
s_ashr_i32 s8, s0, 31 // (op 657) 

s_xor_b32 s0, s1, s8 // (op 657) 

s_sub_i32 s1, s0, s8 // (op 657) 


s_mul_hi_i32 s0, s1, s14 // (op 657) 
s_add_i32 s8, s0, s1 // (op 657) 

s_ashr_i32 s0, s8, 31 // (op 657) 
s_cmp_eq_i32 s14, 0 // (op 657) 
s_mov_b32 s9, scc // (op 657) 
s_cmp_eq_i32 s9, 0 // (op 657) 
s_cselect_b32 s18, 0, -1 // (op 657) 

s_and_b32 s9, s15, 31 // (op 657) 
s_lshl_b32 s19, 1, s9 // (op 657) 
s_add_i32 s49, s19, s18 // (op 657) 


s_and_b32 s18, s0, s49 // (op 657) 


s_add_i32 s0, s8, s18 // (op 657) 


s_ashr_i32 s8, s0, s9 // (op 657) 


s_lshl_b32 s0, s15, 24 // (op 657) 
s_ashr_i32 s9, s0, 31 // (op 657) 

s_xor_b32 s0, s8, s9 // (op 657) 

s_sub_i32 s8, s0, s9 // (op 657) 


s_mul_i32 s0, s1, s45 // (op 657) 

s_sub_u32 s1, s16, s0 // (op 657) 

s_mul_i32 s0, s8, s45 // (op 657) 
s_add_u32 s0, s0, s1 // (op 657) 


s_mul_hi_i32 s1, s16, s11 // (op 657) 
s_add_i32 s8, s1, s16 // (op 657) 

s_ashr_i32 s1, s8, 31 // (op 657) 
s_cmp_eq_i32 s11, 0 // (op 657) 
s_mov_b32 s9, scc // (op 657) 
s_cmp_eq_i32 s9, 0 // (op 657) 
s_cselect_b32 s18, 0, -1 // (op 657) 

s_and_b32 s9, s46, 31 // (op 657) 
s_lshl_b32 s19, 1, s9 // (op 657) 
s_add_i32 s49, s19, s18 // (op 657) 


s_and_b32 s18, s1, s49 // (op 657) 


s_add_i32 s1, s8, s18 // (op 657) 


s_ashr_i32 s8, s1, s9 // (op 657) 


s_lshl_b32 s1, s46, 24 // (op 657) 
s_ashr_i32 s9, s1, 31 // (op 657) 

s_xor_b32 s1, s8, s9 // (op 657) 

s_sub_i32 s8, s1, s9 // (op 657) 


s_add_u32 s18, s4, 32 // (op 657) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 657) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 657) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 657) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 657) 

s_mul_i32 s1, s45, s18 // (op 657) 

s_mul_i32 s9, s10, s1 // (op 657) 

s_mul_i32 s1, s8, s9 // (op 657) 

s_sub_u32 s9, s16, s1 // (op 657) 

s_mov_b32 s1, s9 // (op 657) 
s_mul_hi_i32 s18, s1, s44 // (op 657) 
s_add_i32 s19, s18, s1 // (op 657) 


s_ashr_i32 s1, s19, 31 // (op 657) 
s_cmp_eq_i32 s44, 0 // (op 657) 
s_mov_b32 s18, scc // (op 657) 
s_cmp_eq_i32 s18, 0 // (op 657) 
s_cselect_b32 s49, 0, -1 // (op 657) 

s_and_b32 s18, s48, 31 // (op 657) 
s_lshl_b32 s50, 1, s18 // (op 657) 
s_add_i32 s51, s50, s49 // (op 657) 


s_and_b32 s49, s1, s51 // (op 657) 


s_add_i32 s1, s19, s49 // (op 657) 


s_ashr_i32 s19, s1, s18 // (op 657) 


s_lshl_b32 s1, s48, 24 // (op 657) 
s_ashr_i32 s18, s1, 31 // (op 657) 

s_xor_b32 s1, s19, s18 // (op 657) 

s_sub_i32 s19, s1, s18 // (op 657) 


s_mul_i32 s1, s47, s45 // (op 657) 
s_add_u32 s50, s6, 16 // (op 657) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 657) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 657) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 657) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 4 // (op 657) 

s_sub_i32 s18, s50, s1 // (op 657) 

s_mul_i32 s49, s19, s18 // (op 657) 


s_sub_u32 s18, s9, s49 // (op 657) 


s_add_u32 s9, s18, s1 // (op 657) 


s_cmp_eq_i32 s8, 0 // (op 657) 
s_mov_b32 s1, scc // (op 657) 

s_cmp_eq_i32 s1, 0 // (op 657) 
s_cselect_b32 s8, s9, s0 // (op 657) 



v_lshl_add_u32 v18, s8, 4, v21 // (op 657) 


v_mul_lo_u32 v21, v18, s26 // (op 657) 
v_add_u32 v21, v21, v19 // (op 657) 


s_mul_i32 s1, s26, s7 // (op 659) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 659) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 659) least significant
s_mul_hi_u32 s9, s26, s6 // (op 659) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 659) most significant: sum
s_add_u32 s1, s1, s9 // (op 659) most significant: sum


s_mov_b32 s60, s28 // (op 659) reg expression
s_mov_b32 s61, s29 // (op 659) reg expression
s_mov_b32 s62, s0 // (op 659) reg expression
s_mov_b32 s63, 131072 // (op 659) call()

s_mul_hi_i32 s0, s16, s12 // (op 702) 
v_lshrrev_b32 v18, 2, v1 // (op 702) 
s_add_i32 s1, s0, s16 // (op 702) 

v_and_b32 v19, -4, v1 // (op 702) 
s_ashr_i32 s0, s1, 31 // (op 702) 
v_sub_u32 v22, v1, v19 // (op 702) 

s_cmp_eq_i32 s12, 0 // (op 702) 
s_mov_b32 s8, scc // (op 702) 
s_cmp_eq_i32 s8, 0 // (op 702) 
s_cselect_b32 s9, 0, -1 // (op 702) 

s_and_b32 s8, s13, 31 // (op 702) 
s_lshl_b32 s18, 1, s8 // (op 702) 
s_add_i32 s19, s18, s9 // (op 702) 


s_and_b32 s9, s0, s19 // (op 702) 


s_add_i32 s0, s1, s9 // (op 702) 


s_ashr_i32 s1, s0, s8 // (op 702) 


s_lshl_b32 s0, s13, 24 // (op 702) 
s_ashr_i32 s8, s0, 31 // (op 702) 

s_xor_b32 s0, s1, s8 // (op 702) 

s_sub_i32 s1, s0, s8 // (op 702) 


s_mul_hi_i32 s0, s1, s14 // (op 702) 
s_add_i32 s8, s0, s1 // (op 702) 

s_ashr_i32 s0, s8, 31 // (op 702) 
s_cmp_eq_i32 s14, 0 // (op 702) 
s_mov_b32 s9, scc // (op 702) 
s_cmp_eq_i32 s9, 0 // (op 702) 
s_cselect_b32 s18, 0, -1 // (op 702) 

s_and_b32 s9, s15, 31 // (op 702) 
s_lshl_b32 s19, 1, s9 // (op 702) 
s_add_i32 s49, s19, s18 // (op 702) 


s_and_b32 s18, s0, s49 // (op 702) 


s_add_i32 s0, s8, s18 // (op 702) 


s_ashr_i32 s8, s0, s9 // (op 702) 


s_lshl_b32 s0, s15, 24 // (op 702) 
s_ashr_i32 s9, s0, 31 // (op 702) 

s_xor_b32 s0, s8, s9 // (op 702) 

s_sub_i32 s8, s0, s9 // (op 702) 


s_add_u32 s18, s4, 32 // (op 702) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 702) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 702) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 702) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 702) 

s_mul_i32 s0, s8, s18 // (op 702) 

s_sub_i32 s8, s1, s0 // (op 702) 


s_mul_hi_i32 s9, s16, s11 // (op 702) 
s_add_i32 s0, s9, s16 // (op 702) 

s_ashr_i32 s1, s0, 31 // (op 702) 
s_cmp_eq_i32 s11, 0 // (op 702) 
s_mov_b32 s9, scc // (op 702) 
s_cmp_eq_i32 s9, 0 // (op 702) 
s_cselect_b32 s49, 0, -1 // (op 702) 

s_and_b32 s9, s46, 31 // (op 702) 
s_lshl_b32 s50, 1, s9 // (op 702) 
s_add_i32 s51, s50, s49 // (op 702) 


s_and_b32 s49, s1, s51 // (op 702) 


s_add_i32 s1, s0, s49 // (op 702) 


s_ashr_i32 s0, s1, s9 // (op 702) 


s_lshl_b32 s1, s46, 24 // (op 702) 
s_ashr_i32 s9, s1, 31 // (op 702) 

s_xor_b32 s1, s0, s9 // (op 702) 

s_sub_i32 s0, s1, s9 // (op 702) 


s_mul_i32 s1, s45, s18 // (op 702) 

s_mul_i32 s9, s10, s1 // (op 702) 

s_mul_i32 s1, s0, s9 // (op 702) 

s_sub_u32 s9, s16, s1 // (op 702) 

s_mul_hi_i32 s1, s9, s44 // (op 702) 
s_add_i32 s18, s1, s9 // (op 702) 


s_ashr_i32 s1, s18, 31 // (op 702) 
s_cmp_eq_i32 s44, 0 // (op 702) 
s_mov_b32 s9, scc // (op 702) 
s_cmp_eq_i32 s9, 0 // (op 702) 
s_cselect_b32 s19, 0, -1 // (op 702) 

s_and_b32 s9, s48, 31 // (op 702) 
s_lshl_b32 s49, 1, s9 // (op 702) 
s_add_i32 s50, s49, s19 // (op 702) 


s_and_b32 s19, s1, s50 // (op 702) 


s_add_i32 s1, s18, s19 // (op 702) 


s_ashr_i32 s18, s1, s9 // (op 702) 


s_lshl_b32 s1, s48, 24 // (op 702) 
s_ashr_i32 s9, s1, 31 // (op 702) 

s_xor_b32 s1, s18, s9 // (op 702) 

s_sub_i32 s18, s1, s9 // (op 702) 


s_cmp_eq_i32 s0, 0 // (op 702) 
s_mov_b32 s1, scc // (op 702) 

s_cmp_eq_i32 s1, 0 // (op 702) 
s_cselect_b32 s0, s18, s8 // (op 702) 



v_lshl_add_u32 v19, s0, 5, v18 // (op 702) 


v_mul_lo_u32 v18, v19, s30 // (op 702) 

v_add_u32 v19, v22, v18 // (op 702) 


s_mul_i32 s1, s30, s5 // (op 704) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 704) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 704) least significant
s_mul_hi_u32 s9, s30, s4 // (op 704) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 704) most significant: sum
s_add_u32 s1, s1, s9 // (op 704) most significant: sum


s_mov_b32 s64, s32 // (op 704) reg expression
s_mov_b32 s65, s33 // (op 704) reg expression
s_mov_b32 s66, s0 // (op 704) reg expression
s_mov_b32 s67, 131072 // (op 704) call()

s_mul_hi_i32 s0, s16, s12 // (op 721) 
v_lshrrev_b32 v18, 2, v1 // (op 721) 
s_add_i32 s1, s0, s16 // (op 721) 

v_and_b32 v22, -4, v1 // (op 721) 
s_ashr_i32 s0, s1, 31 // (op 721) 
v_sub_u32 v23, v1, v22 // (op 721) 

s_cmp_eq_i32 s12, 0 // (op 721) 
s_mov_b32 s8, scc // (op 721) 
s_cmp_eq_i32 s8, 0 // (op 721) 
s_cselect_b32 s9, 0, -1 // (op 721) 

s_and_b32 s8, s13, 31 // (op 721) 
s_lshl_b32 s18, 1, s8 // (op 721) 
s_add_i32 s19, s18, s9 // (op 721) 


s_and_b32 s9, s0, s19 // (op 721) 


s_add_i32 s0, s1, s9 // (op 721) 


s_ashr_i32 s1, s0, s8 // (op 721) 


s_lshl_b32 s0, s13, 24 // (op 721) 
s_ashr_i32 s8, s0, 31 // (op 721) 

s_xor_b32 s0, s1, s8 // (op 721) 

s_sub_i32 s1, s0, s8 // (op 721) 


s_mul_hi_i32 s0, s1, s14 // (op 721) 
s_add_i32 s8, s0, s1 // (op 721) 

s_ashr_i32 s0, s8, 31 // (op 721) 
s_cmp_eq_i32 s14, 0 // (op 721) 
s_mov_b32 s9, scc // (op 721) 
s_cmp_eq_i32 s9, 0 // (op 721) 
s_cselect_b32 s18, 0, -1 // (op 721) 

s_and_b32 s9, s15, 31 // (op 721) 
s_lshl_b32 s19, 1, s9 // (op 721) 
s_add_i32 s49, s19, s18 // (op 721) 


s_and_b32 s18, s0, s49 // (op 721) 


s_add_i32 s0, s8, s18 // (op 721) 


s_ashr_i32 s8, s0, s9 // (op 721) 


s_lshl_b32 s0, s15, 24 // (op 721) 
s_ashr_i32 s9, s0, 31 // (op 721) 

s_xor_b32 s0, s8, s9 // (op 721) 

s_sub_i32 s8, s0, s9 // (op 721) 


s_add_u32 s18, s4, 32 // (op 721) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 721) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 721) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 721) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 721) 

s_mul_i32 s0, s8, s18 // (op 721) 

s_sub_i32 s8, s1, s0 // (op 721) 


s_mul_hi_i32 s9, s16, s11 // (op 721) 
s_add_i32 s0, s9, s16 // (op 721) 

s_ashr_i32 s1, s0, 31 // (op 721) 
s_cmp_eq_i32 s11, 0 // (op 721) 
s_mov_b32 s9, scc // (op 721) 
s_cmp_eq_i32 s9, 0 // (op 721) 
s_cselect_b32 s49, 0, -1 // (op 721) 

s_and_b32 s9, s46, 31 // (op 721) 
s_lshl_b32 s50, 1, s9 // (op 721) 
s_add_i32 s51, s50, s49 // (op 721) 


s_and_b32 s49, s1, s51 // (op 721) 


s_add_i32 s1, s0, s49 // (op 721) 


s_ashr_i32 s0, s1, s9 // (op 721) 


s_lshl_b32 s1, s46, 24 // (op 721) 
s_ashr_i32 s9, s1, 31 // (op 721) 

s_xor_b32 s1, s0, s9 // (op 721) 

s_sub_i32 s0, s1, s9 // (op 721) 


s_mul_i32 s1, s45, s18 // (op 721) 

s_mul_i32 s9, s10, s1 // (op 721) 

s_mul_i32 s1, s0, s9 // (op 721) 

s_sub_u32 s9, s16, s1 // (op 721) 

s_mul_hi_i32 s1, s9, s44 // (op 721) 
s_add_i32 s18, s1, s9 // (op 721) 


s_ashr_i32 s1, s18, 31 // (op 721) 
s_cmp_eq_i32 s44, 0 // (op 721) 
s_mov_b32 s9, scc // (op 721) 
s_cmp_eq_i32 s9, 0 // (op 721) 
s_cselect_b32 s19, 0, -1 // (op 721) 

s_and_b32 s9, s48, 31 // (op 721) 
s_lshl_b32 s49, 1, s9 // (op 721) 
s_add_i32 s50, s49, s19 // (op 721) 


s_and_b32 s19, s1, s50 // (op 721) 


s_add_i32 s1, s18, s19 // (op 721) 


s_ashr_i32 s18, s1, s9 // (op 721) 


s_lshl_b32 s1, s48, 24 // (op 721) 
s_ashr_i32 s9, s1, 31 // (op 721) 

s_xor_b32 s1, s18, s9 // (op 721) 

s_sub_i32 s18, s1, s9 // (op 721) 


s_cmp_eq_i32 s0, 0 // (op 721) 
s_mov_b32 s1, scc // (op 721) 

s_cmp_eq_i32 s1, 0 // (op 721) 
s_cselect_b32 s0, s18, s8 // (op 721) 



v_lshl_add_u32 v22, s0, 5, v18 // (op 721) 


v_mul_lo_u32 v18, v22, s30 // (op 721) 

v_add_u32 v22, v23, v18 // (op 721) 


s_mul_i32 s1, s30, s5 // (op 723) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 723) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 723) least significant
s_mul_hi_u32 s9, s30, s4 // (op 723) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 723) most significant: sum
s_add_u32 s1, s1, s9 // (op 723) most significant: sum


s_mov_b32 s64, s32 // (op 723) reg expression
s_mov_b32 s65, s33 // (op 723) reg expression
s_mov_b32 s66, s0 // (op 723) reg expression
s_mov_b32 s67, 131072 // (op 723) call()




buffer_load_ubyte v18, v20, s[60:63], 0 offen offset: 0 // (op 341) Load value
v_and_b32 v23, -8, v1 // (op 839) 
v_sub_u32 v24, v1, v23 // (op 839) 

v_lshlrev_b32 v23, 4, v24 // (op 839) 

v_lshrrev_b32 v24, 3, v1 // (op 839) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 839) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 839) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[22:23], 4 // (op 459) 
v_mov_b64 v[24:25], s[8:9] // (op 459) convert


v_mov_b64 v[26:27], v[10:11] // (op 459) 
s_mov_b32 m0, s0 // (op 459) reg expression
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 459) Load value direct to lds
v_add_co_u32 v26, s[8:9], v26, v24 // (op 459) least significant half
v_addc_co_u32 v27, s[8:9], 0, v27, s[8:9] // (op 459) most significant half

s_add_u32 m0, m0, 2048 // (op 459) 
buffer_load_dwordx4 v26, s[56:59], 0 offen offset: 0 lds // (op 459) Load value direct to lds






buffer_load_ubyte v23, v19, s[64:67], 0 offen offset: 0 // (op 348) Load value
v_and_b32 v24, -8, v1 // (op 787) 
v_sub_u32 v25, v1, v24 // (op 787) 

v_lshlrev_b32 v24, 4, v25 // (op 787) 

v_lshrrev_b32 v25, 3, v1 // (op 787) 
v_lshl_add_u32 v26, v25, 7, v24 // (op 787) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 787) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 464) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 464) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 464) Load value direct to lds





buffer_load_ubyte v24, v20, s[60:63], 0 offen offset:4 // (op 355) Load value

v_and_b32 v20, -8, v1 // (op 1004) 
v_sub_u32 v25, v1, v20 // (op 1004) 

v_lshlrev_b32 v20, 4, v25 // (op 1004) 

v_lshrrev_b32 v25, 3, v1 // (op 1004) 
v_lshl_add_u32 v26, v25, 7, v20 // (op 1004) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1004) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[8:9], s[22:23], 4 // (op 469) 
v_mov_b64 v[26:27], s[8:9] // (op 469) convert


v_mov_b64 v[28:29], v[10:11] // (op 469) 
s_add_u32 s1, 6144, s0 // (op 469) 
s_mov_b32 m0, s1 // (op 469) reg expression
s_mov_b32 s8, 128 // (op 469) call()
buffer_load_dwordx4 v28, s[56:59], s8 offen offset: 0 lds // (op 469) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 469) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 469) most significant half

s_add_u32 m0, m0, 2048 // (op 469) 
buffer_load_dwordx4 v28, s[56:59], s8 offen offset: 0 lds // (op 469) Load value direct to lds









buffer_load_ubyte v20, v19, s[64:67], 0 offen offset:4 // (op 362) Load value

v_and_b32 v19, -8, v1 // (op 965) 
v_sub_u32 v10, v1, v19 // (op 965) 

v_lshlrev_b32 v11, 4, v10 // (op 965) 

v_lshrrev_b32 v10, 3, v1 // (op 965) 
v_lshl_add_u32 v19, v10, 7, v11 // (op 965) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 965) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 10240, s0 // (op 471) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 471) reg expression
s_mov_b32 s8, 128 // (op 471) call()
buffer_load_dwordx4 v14, s[52:55], s8 offen offset: 0 lds // (op 471) Load value direct to lds




v_and_b32 v19, -4, v1 // (op 892) 
v_sub_u32 v10, v1, v19 // (op 892) 

v_and_b32 v11, 4294967292, v1 // (op 892) 
v_add_u32 v19, v11, v10 // (op 892) 





s_waitcnt vmcnt(6)
ds_write_b8 v19, v23 offset:12288 // (op 368) Store local data 

s_barrier  // (op 1089) 



ds_read_b128 v[26:29], v4 // (op 385) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 385) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1105) 



ds_read_u8 v19, v0 offset:12288 // (op 388) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1073) 



ds_read_b128 v[34:37], v3 offset:4096 // (op 382) Load local data 
ds_read_b128 v[38:41], v3 offset:4160 // (op 382) Load local data 
v_and_b32 v25, -8, v1 // (op 759) 
v_sub_u32 v10, v1, v25 // (op 759) 

v_lshlrev_b32 v11, 4, v10 // (op 759) 

v_lshrrev_b32 v10, 3, v1 // (op 759) 
v_lshl_add_u32 v25, v10, 7, v11 // (op 759) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 759) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -8, v1 // (op 811) 
v_sub_u32 v10, v1, v25 // (op 811) 

v_lshlrev_b32 v11, 4, v10 // (op 811) 

v_lshrrev_b32 v10, 3, v1 // (op 811) 
v_lshl_add_u32 v25, v10, 7, v11 // (op 811) 


s_nop 0
v_readfirstlane_b32 s1, v25 // (op 811) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v25, -4, v1 // (op 863) 
v_sub_u32 v10, v1, v25 // (op 863) 

v_and_b32 v11, 4294967292, v1 // (op 863) 
v_add_u32 v25, v11, v10 // (op 863) 


v_and_b32 v10, -8, v1 // (op 950) 
v_sub_u32 v11, v1, v10 // (op 950) 

v_lshlrev_b32 v10, 4, v11 // (op 950) 

v_lshrrev_b32 v11, 3, v1 // (op 950) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 950) 


s_nop 0
v_readfirstlane_b32 s8, v14 // (op 950) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 989) 
v_sub_u32 v11, v1, v10 // (op 989) 

v_lshlrev_b32 v10, 4, v11 // (op 989) 

v_lshrrev_b32 v11, 3, v1 // (op 989) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 989) 


s_nop 0
v_readfirstlane_b32 s9, v14 // (op 989) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 1028) 
v_sub_u32 v11, v1, v10 // (op 1028) 

v_and_b32 v10, 4294967292, v1 // (op 1028) 
v_add_u32 v14, v10, v11 // (op 1028) 


s_mov_b32 s18, 0 // (op 57) call()
s_ashr_i64 s[50:51], s[34:35], 7 // (op 56) 
s_mov_b32 s19, s50 // (op 56) convert

s_and_b32 s49, s19, 4294967294 // (op 56) 

s_cmp_lt_u32 s18, s49 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56 if false)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56:
s_add_u32 s18, s18, 2 // (op 58) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v18, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 84) 


v_mov_b32 v15, v21 // (op 77) 

buffer_load_ubyte v18, v15, s[60:63], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 1092) 

s_lshl_b64 s[50:51], s[22:23], 4 // (op 431) 
v_mov_b64 v[10:11], s[50:51] // (op 431) convert


v_mov_b64 v[42:43], v[12:13] // (op 431) 
v_mov_b64 v[44:45], v[42:43] // (op 431) 
s_mov_b32 m0, s1 // (op 431) reg expression
s_mov_b32 s19, 256 // (op 431) call()
buffer_load_dwordx4 v44, s[56:59], s19 offen offset: 0 lds // (op 431) Load value direct to lds
v_add_co_u32 v44, s[50:51], v44, v10 // (op 431) least significant half
v_addc_co_u32 v45, s[50:51], 0, v45, s[50:51] // (op 431) most significant half

s_add_u32 m0, m0, 2048 // (op 431) 
buffer_load_dwordx4 v44, s[56:59], s19 offen offset: 0 lds // (op 431) Load value direct to lds





v_mov_b32 v10, v22 // (op 21) 

buffer_load_ubyte v23, v10, s[64:67], 0 offen offset:8 // (op 21) Load value
s_barrier  // (op 1076) 


v_mov_b64 v[44:45], v[16:17] // (op 426) 
s_add_u32 s19, 4096, s0 // (op 426) 

s_mov_b32 m0, s19 // (op 426) reg expression
s_mov_b32 s49, 256 // (op 426) call()
buffer_load_dwordx4 v44, s[52:55], s49 offen offset: 0 lds // (op 426) Load value direct to lds


s_barrier  // (op 1137) 



s_waitcnt vmcnt(6)
ds_write_b8 v14, v20 offset:12416 // (op 240) Store local data 
s_barrier  // (op 1128) 



ds_read_b128 v[26:29], v7 offset:6144 // (op 233) Load local data 
ds_read_b128 v[30:33], v7 offset:6208 // (op 233) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1134) 



ds_read_u8 v19, v5 offset:12416 // (op 237) Load local data 
s_barrier  // (op 1184) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1122) 



ds_read_b128 v[34:37], v6 offset:10240 // (op 231) Load local data 
ds_read_b128 v[38:41], v6 offset:10304 // (op 231) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v24, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 238) 



buffer_load_ubyte v24, v15, s[60:63], 0 offen offset:12 // (op 235) Load value

s_barrier  // (op 1131) 
v_add_u32 v21, 8, v21 // (op 649) 

s_lshl_b64 s[50:51], s[22:23], 4 // (op 454) 
v_mov_b64 v[46:47], s[50:51] // (op 454) convert


v_mov_b64 v[48:49], v[42:43] // (op 454) 
s_add_u32 s19, 6144, s9 // (op 454) 
s_mov_b32 m0, s19 // (op 454) reg expression
s_mov_b32 s49, 384 // (op 454) call()
buffer_load_dwordx4 v48, s[56:59], s49 offen offset: 0 lds // (op 454) Load value direct to lds
v_add_co_u32 v48, s[50:51], v48, v46 // (op 454) least significant half
v_addc_co_u32 v49, s[50:51], 0, v49, s[50:51] // (op 454) most significant half

s_add_u32 m0, m0, 2048 // (op 454) 
buffer_load_dwordx4 v48, s[56:59], s49 offen offset: 0 lds // (op 454) Load value direct to lds








v_add_co_u32 v12, vcc, 256, v12 // (op 585) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 585) most significant half
buffer_load_ubyte v20, v10, s[64:67], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 1125) 
v_add_u32 v22, 8, v22 // (op 713) 


s_add_u32 s19, 10240, s8 // (op 449) 

s_mov_b32 m0, s19 // (op 449) reg expression
s_mov_b32 s49, 384 // (op 449) call()
buffer_load_dwordx4 v44, s[52:55], s49 offen offset: 0 lds // (op 449) Load value direct to lds



s_barrier  // (op 1108) 
v_add_co_u32 v16, vcc, 256, v16 // (op 521) least significant half



v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 521) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v25, v23 offset:12288 // (op 26) Store local data 
s_barrier  // (op 1098) 



ds_read_b128 v[26:29], v4 // (op 72) Load local data 
ds_read_b128 v[30:33], v4 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1115) 



ds_read_u8 v19, v0 offset:12288 // (op 81) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1082) 



ds_read_b128 v[34:37], v3 offset:4096 // (op 67) Load local data 
ds_read_b128 v[38:41], v3 offset:4160 // (op 67) Load local data 
s_ashr_i64 s[50:51], s[34:35], 7 // (op 56) 
s_mov_b32 s19, s50 // (op 56) convert

s_and_b32 s49, s19, 4294967294 // (op 56) 

s_cmp_lt_u32 s18, s49 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56 if true)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56:













s_barrier  // (op 1148) 
s_mul_hi_i32 s0, s16, s12 // (op 484) 
v_lshrrev_b32 v25, 3, v1 // (op 484) 
s_add_i32 s1, s0, s16 // (op 484) 

v_and_b32 v10, -8, v1 // (op 484) 
s_ashr_i32 s0, s1, 31 // (op 484) 
v_sub_u32 v11, v1, v10 // (op 484) 

s_cmp_eq_i32 s12, 0 // (op 484) 
s_mov_b32 s8, scc // (op 484) 
v_lshlrev_b32 v10, 4, v11 // (op 484) 

s_cmp_eq_i32 s8, 0 // (op 484) 
s_cselect_b32 s9, 0, -1 // (op 484) 

s_and_b32 s8, s13, 31 // (op 484) 
s_lshl_b32 s18, 1, s8 // (op 484) 
s_add_i32 s19, s18, s9 // (op 484) 


s_and_b32 s9, s0, s19 // (op 484) 


s_add_i32 s0, s1, s9 // (op 484) 


s_ashr_i32 s1, s0, s8 // (op 484) 


s_lshl_b32 s0, s13, 24 // (op 484) 
s_ashr_i32 s8, s0, 31 // (op 484) 

s_xor_b32 s0, s1, s8 // (op 484) 

s_sub_i32 s1, s0, s8 // (op 484) 


s_mul_hi_i32 s0, s1, s14 // (op 484) 
s_add_i32 s8, s0, s1 // (op 484) 

s_ashr_i32 s0, s8, 31 // (op 484) 
s_cmp_eq_i32 s14, 0 // (op 484) 
s_mov_b32 s9, scc // (op 484) 
s_cmp_eq_i32 s9, 0 // (op 484) 
s_cselect_b32 s18, 0, -1 // (op 484) 

s_and_b32 s9, s15, 31 // (op 484) 
s_lshl_b32 s19, 1, s9 // (op 484) 
s_add_i32 s49, s19, s18 // (op 484) 


s_and_b32 s18, s0, s49 // (op 484) 


s_add_i32 s0, s8, s18 // (op 484) 


s_ashr_i32 s8, s0, s9 // (op 484) 


s_lshl_b32 s0, s15, 24 // (op 484) 
s_ashr_i32 s9, s0, 31 // (op 484) 

s_xor_b32 s0, s8, s9 // (op 484) 

s_sub_i32 s8, s0, s9 // (op 484) 


s_mul_i32 s0, s1, s45 // (op 484) 

s_sub_u32 s1, s16, s0 // (op 484) 

s_mul_i32 s0, s8, s45 // (op 484) 
s_add_u32 s0, s0, s1 // (op 484) 


s_mul_hi_i32 s1, s16, s11 // (op 484) 
s_add_i32 s8, s1, s16 // (op 484) 

s_ashr_i32 s1, s8, 31 // (op 484) 
s_cmp_eq_i32 s11, 0 // (op 484) 
s_mov_b32 s9, scc // (op 484) 
s_cmp_eq_i32 s9, 0 // (op 484) 
s_cselect_b32 s18, 0, -1 // (op 484) 

s_and_b32 s9, s46, 31 // (op 484) 
s_lshl_b32 s19, 1, s9 // (op 484) 
s_add_i32 s49, s19, s18 // (op 484) 


s_and_b32 s18, s1, s49 // (op 484) 


s_add_i32 s1, s8, s18 // (op 484) 


s_ashr_i32 s8, s1, s9 // (op 484) 


s_lshl_b32 s1, s46, 24 // (op 484) 
s_ashr_i32 s9, s1, 31 // (op 484) 

s_xor_b32 s1, s8, s9 // (op 484) 

s_sub_i32 s8, s1, s9 // (op 484) 


s_add_u32 s18, s4, 32 // (op 484) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 484) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 484) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 484) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 484) 

s_mul_i32 s1, s45, s18 // (op 484) 

s_mul_i32 s9, s10, s1 // (op 484) 

s_mul_i32 s1, s8, s9 // (op 484) 

s_sub_u32 s9, s16, s1 // (op 484) 

s_mov_b32 s1, s9 // (op 484) 
s_mul_hi_i32 s18, s1, s44 // (op 484) 
s_add_i32 s19, s18, s1 // (op 484) 


s_ashr_i32 s1, s19, 31 // (op 484) 
s_cmp_eq_i32 s44, 0 // (op 484) 
s_mov_b32 s18, scc // (op 484) 
s_cmp_eq_i32 s18, 0 // (op 484) 
s_cselect_b32 s49, 0, -1 // (op 484) 

s_and_b32 s18, s48, 31 // (op 484) 
s_lshl_b32 s50, 1, s18 // (op 484) 
s_add_i32 s51, s50, s49 // (op 484) 


s_and_b32 s49, s1, s51 // (op 484) 


s_add_i32 s1, s19, s49 // (op 484) 


s_ashr_i32 s19, s1, s18 // (op 484) 


s_lshl_b32 s1, s48, 24 // (op 484) 
s_ashr_i32 s18, s1, 31 // (op 484) 

s_xor_b32 s1, s19, s18 // (op 484) 

s_sub_i32 s19, s1, s18 // (op 484) 


s_mul_i32 s1, s47, s45 // (op 484) 
s_add_u32 s50, s6, 16 // (op 484) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 484) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 484) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 484) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 484) 

s_sub_i32 s18, s50, s1 // (op 484) 

s_mul_i32 s49, s19, s18 // (op 484) 


s_sub_u32 s18, s9, s49 // (op 484) 


s_add_u32 s9, s18, s1 // (op 484) 


s_cmp_eq_i32 s8, 0 // (op 484) 
s_mov_b32 s1, scc // (op 484) 

s_cmp_eq_i32 s1, 0 // (op 484) 
s_cselect_b32 s8, s9, s0 // (op 484) 



v_lshl_add_u32 v11, s8, 4, v25 // (op 484) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 484) 
s_lshr_b32 s8, s0, 1 // (op 484) 

v_lshl_add_u32 v25, s8, 8, v10 // (op 484) 


v_mov_b32 v23, 0 // (op 484) convert
v_mov_b32 v22, v25 // (op 484) convert

v_mul_lo_u32 v13, v11, s3 // (op 484) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 484) most significant: high of low * low
v_mul_lo_u32 v12, v11, s2 // (op 484) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 484) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v22 // (op 484) least significant half
v_addc_co_u32 v13, s[0:1], v13, v23, s[0:1] // (op 484) most significant half



s_mul_i32 s1, s2, s7 // (op 486) most significant: low of low * high
s_mul_i32 s8, s3, s6 // (op 486) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 486) least significant
s_mul_hi_u32 s9, s2, s6 // (op 486) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 486) most significant: sum
s_add_u32 s1, s1, s9 // (op 486) most significant: sum


s_mov_b32 s52, s20 // (op 486) reg expression
s_mov_b32 s53, s21 // (op 486) reg expression
s_mov_b32 s54, s0 // (op 486) reg expression
s_mov_b32 s55, 131072 // (op 486) call()

s_mul_hi_i32 s0, s16, s12 // (op 551) 
v_lshrrev_b32 v25, 3, v1 // (op 551) 
s_add_i32 s1, s0, s16 // (op 551) 

v_and_b32 v10, -8, v1 // (op 551) 
s_ashr_i32 s0, s1, 31 // (op 551) 
v_sub_u32 v11, v1, v10 // (op 551) 

s_cmp_eq_i32 s12, 0 // (op 551) 
s_mov_b32 s8, scc // (op 551) 
v_lshlrev_b32 v10, 4, v11 // (op 551) 

s_cmp_eq_i32 s8, 0 // (op 551) 
s_cselect_b32 s9, 0, -1 // (op 551) 

s_and_b32 s8, s13, 31 // (op 551) 
s_lshl_b32 s18, 1, s8 // (op 551) 
s_add_i32 s19, s18, s9 // (op 551) 


s_and_b32 s9, s0, s19 // (op 551) 


s_add_i32 s0, s1, s9 // (op 551) 


s_ashr_i32 s1, s0, s8 // (op 551) 


s_lshl_b32 s0, s13, 24 // (op 551) 
s_ashr_i32 s8, s0, 31 // (op 551) 

s_xor_b32 s0, s1, s8 // (op 551) 

s_sub_i32 s1, s0, s8 // (op 551) 


s_mul_hi_i32 s0, s1, s14 // (op 551) 
s_add_i32 s8, s0, s1 // (op 551) 

s_ashr_i32 s0, s8, 31 // (op 551) 
s_cmp_eq_i32 s14, 0 // (op 551) 
s_mov_b32 s9, scc // (op 551) 
s_cmp_eq_i32 s9, 0 // (op 551) 
s_cselect_b32 s18, 0, -1 // (op 551) 

s_and_b32 s9, s15, 31 // (op 551) 
s_lshl_b32 s19, 1, s9 // (op 551) 
s_add_i32 s49, s19, s18 // (op 551) 


s_and_b32 s18, s0, s49 // (op 551) 


s_add_i32 s0, s8, s18 // (op 551) 


s_ashr_i32 s8, s0, s9 // (op 551) 


s_lshl_b32 s0, s15, 24 // (op 551) 
s_ashr_i32 s9, s0, 31 // (op 551) 

s_xor_b32 s0, s8, s9 // (op 551) 

s_sub_i32 s8, s0, s9 // (op 551) 


s_add_u32 s18, s4, 32 // (op 551) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 551) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 551) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 551) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 551) 

s_mul_i32 s0, s8, s18 // (op 551) 

s_sub_i32 s8, s1, s0 // (op 551) 


s_mul_hi_i32 s9, s16, s11 // (op 551) 
s_add_i32 s0, s9, s16 // (op 551) 

s_ashr_i32 s1, s0, 31 // (op 551) 
s_cmp_eq_i32 s11, 0 // (op 551) 
s_mov_b32 s9, scc // (op 551) 
s_cmp_eq_i32 s9, 0 // (op 551) 
s_cselect_b32 s49, 0, -1 // (op 551) 

s_and_b32 s9, s46, 31 // (op 551) 
s_lshl_b32 s50, 1, s9 // (op 551) 
s_add_i32 s51, s50, s49 // (op 551) 


s_and_b32 s49, s1, s51 // (op 551) 


s_add_i32 s1, s0, s49 // (op 551) 


s_ashr_i32 s0, s1, s9 // (op 551) 


s_lshl_b32 s1, s46, 24 // (op 551) 
s_ashr_i32 s9, s1, 31 // (op 551) 

s_xor_b32 s1, s0, s9 // (op 551) 

s_sub_i32 s0, s1, s9 // (op 551) 


s_mul_i32 s1, s45, s18 // (op 551) 

s_mul_i32 s9, s10, s1 // (op 551) 

s_mul_i32 s1, s0, s9 // (op 551) 

s_sub_u32 s9, s16, s1 // (op 551) 

s_mul_hi_i32 s1, s9, s44 // (op 551) 
s_add_i32 s18, s1, s9 // (op 551) 


s_ashr_i32 s1, s18, 31 // (op 551) 
s_cmp_eq_i32 s44, 0 // (op 551) 
s_mov_b32 s9, scc // (op 551) 
s_cmp_eq_i32 s9, 0 // (op 551) 
s_cselect_b32 s19, 0, -1 // (op 551) 

s_and_b32 s9, s48, 31 // (op 551) 
s_lshl_b32 s49, 1, s9 // (op 551) 
s_add_i32 s50, s49, s19 // (op 551) 


s_and_b32 s19, s1, s50 // (op 551) 


s_add_i32 s1, s18, s19 // (op 551) 


s_ashr_i32 s18, s1, s9 // (op 551) 


s_lshl_b32 s1, s48, 24 // (op 551) 
s_ashr_i32 s9, s1, 31 // (op 551) 

s_xor_b32 s1, s18, s9 // (op 551) 

s_sub_i32 s18, s1, s9 // (op 551) 


s_cmp_eq_i32 s0, 0 // (op 551) 
s_mov_b32 s1, scc // (op 551) 

s_cmp_eq_i32 s1, 0 // (op 551) 
s_cselect_b32 s0, s18, s8 // (op 551) 



v_lshl_add_u32 v11, s0, 5, v25 // (op 551) 


v_mul_lo_u32 v23, v11, s23 // (op 551) most significant: low of low * high
v_mul_hi_u32 v21, v11, s22 // (op 551) most significant: high of low * low
v_mul_lo_u32 v22, v11, s22 // (op 551) least significant: low of low * low
v_add_u32 v23, v23, v21 // (op 551) most significant: sum


s_ashr_i64 s[0:1], s[34:35], 7 // (op 551) 
s_lshr_b32 s8, s0, 1 // (op 551) 

v_lshl_add_u32 v11, s8, 8, v10 // (op 551) 


v_mov_b32 v15, 0 // (op 551) convert
v_mov_b32 v14, v11 // (op 551) convert

v_add_co_u32 v10, s[0:1], v14, v22 // (op 551) least significant half
v_addc_co_u32 v11, s[0:1], v15, v23, s[0:1] // (op 551) most significant half



s_mul_i32 s1, s22, s5 // (op 553) most significant: low of low * high
s_mul_i32 s8, s23, s4 // (op 553) most significant: low of high * low
s_mul_i32 s0, s22, s4 // (op 553) least significant
s_mul_hi_u32 s9, s22, s4 // (op 553) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 553) most significant: sum
s_add_u32 s1, s1, s9 // (op 553) most significant: sum


s_mov_b32 s56, s24 // (op 553) reg expression
s_mov_b32 s57, s25 // (op 553) reg expression
s_mov_b32 s58, s0 // (op 553) reg expression
s_mov_b32 s59, 131072 // (op 553) call()

s_mul_hi_i32 s0, s16, s12 // (op 615) 
v_lshrrev_b32 v25, 7, v1 // (op 615) 
s_add_i32 s1, s0, s16 // (op 615) 

v_and_b32 v14, 15, v1 // (op 615) 
s_ashr_i32 s0, s1, 31 // (op 615) 
v_lshl_add_u32 v15, v25, 4, v14 // (op 615) 


s_cmp_eq_i32 s12, 0 // (op 615) 
s_mov_b32 s8, scc // (op 615) 
v_and_b32 v14, 63, v1 // (op 615) 
s_cmp_eq_i32 s8, 0 // (op 615) 
s_cselect_b32 s9, 0, -1 // (op 615) 

v_lshrrev_b32 v25, 4, v14 // (op 615) 

s_and_b32 s8, s13, 31 // (op 615) 
s_lshl_b32 s18, 1, s8 // (op 615) 
s_add_i32 s19, s18, s9 // (op 615) 


s_and_b32 s9, s0, s19 // (op 615) 


s_add_i32 s0, s1, s9 // (op 615) 


s_ashr_i32 s1, s0, s8 // (op 615) 


s_lshl_b32 s0, s13, 24 // (op 615) 
s_ashr_i32 s8, s0, 31 // (op 615) 

s_xor_b32 s0, s1, s8 // (op 615) 

s_sub_i32 s1, s0, s8 // (op 615) 


s_mul_hi_i32 s0, s1, s14 // (op 615) 
s_add_i32 s8, s0, s1 // (op 615) 

s_ashr_i32 s0, s8, 31 // (op 615) 
s_cmp_eq_i32 s14, 0 // (op 615) 
s_mov_b32 s9, scc // (op 615) 
s_cmp_eq_i32 s9, 0 // (op 615) 
s_cselect_b32 s18, 0, -1 // (op 615) 

s_and_b32 s9, s15, 31 // (op 615) 
s_lshl_b32 s19, 1, s9 // (op 615) 
s_add_i32 s49, s19, s18 // (op 615) 


s_and_b32 s18, s0, s49 // (op 615) 


s_add_i32 s0, s8, s18 // (op 615) 


s_ashr_i32 s8, s0, s9 // (op 615) 


s_lshl_b32 s0, s15, 24 // (op 615) 
s_ashr_i32 s9, s0, 31 // (op 615) 

s_xor_b32 s0, s8, s9 // (op 615) 

s_sub_i32 s8, s0, s9 // (op 615) 


s_mul_i32 s0, s1, s45 // (op 615) 

s_sub_u32 s1, s16, s0 // (op 615) 

s_mul_i32 s0, s8, s45 // (op 615) 
s_add_u32 s0, s0, s1 // (op 615) 


s_mul_hi_i32 s1, s16, s11 // (op 615) 
s_add_i32 s8, s1, s16 // (op 615) 

s_ashr_i32 s1, s8, 31 // (op 615) 
s_cmp_eq_i32 s11, 0 // (op 615) 
s_mov_b32 s9, scc // (op 615) 
s_cmp_eq_i32 s9, 0 // (op 615) 
s_cselect_b32 s18, 0, -1 // (op 615) 

s_and_b32 s9, s46, 31 // (op 615) 
s_lshl_b32 s19, 1, s9 // (op 615) 
s_add_i32 s49, s19, s18 // (op 615) 


s_and_b32 s18, s1, s49 // (op 615) 


s_add_i32 s1, s8, s18 // (op 615) 


s_ashr_i32 s8, s1, s9 // (op 615) 


s_lshl_b32 s1, s46, 24 // (op 615) 
s_ashr_i32 s9, s1, 31 // (op 615) 

s_xor_b32 s1, s8, s9 // (op 615) 

s_sub_i32 s8, s1, s9 // (op 615) 


s_add_u32 s18, s4, 32 // (op 615) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 615) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 615) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 615) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 615) 

s_mul_i32 s1, s45, s18 // (op 615) 

s_mul_i32 s9, s10, s1 // (op 615) 

s_mul_i32 s1, s8, s9 // (op 615) 

s_sub_u32 s9, s16, s1 // (op 615) 

s_mov_b32 s1, s9 // (op 615) 
s_mul_hi_i32 s18, s1, s44 // (op 615) 
s_add_i32 s19, s18, s1 // (op 615) 


s_ashr_i32 s1, s19, 31 // (op 615) 
s_cmp_eq_i32 s44, 0 // (op 615) 
s_mov_b32 s18, scc // (op 615) 
s_cmp_eq_i32 s18, 0 // (op 615) 
s_cselect_b32 s49, 0, -1 // (op 615) 

s_and_b32 s18, s48, 31 // (op 615) 
s_lshl_b32 s50, 1, s18 // (op 615) 
s_add_i32 s51, s50, s49 // (op 615) 


s_and_b32 s49, s1, s51 // (op 615) 


s_add_i32 s1, s19, s49 // (op 615) 


s_ashr_i32 s19, s1, s18 // (op 615) 


s_lshl_b32 s1, s48, 24 // (op 615) 
s_ashr_i32 s18, s1, 31 // (op 615) 

s_xor_b32 s1, s19, s18 // (op 615) 

s_sub_i32 s19, s1, s18 // (op 615) 


s_mul_i32 s1, s47, s45 // (op 615) 
s_add_u32 s50, s6, 16 // (op 615) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 615) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 615) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 615) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 4 // (op 615) 

s_sub_i32 s18, s50, s1 // (op 615) 

s_mul_i32 s49, s19, s18 // (op 615) 


s_sub_u32 s18, s9, s49 // (op 615) 


s_add_u32 s9, s18, s1 // (op 615) 


s_cmp_eq_i32 s8, 0 // (op 615) 
s_mov_b32 s1, scc // (op 615) 

s_cmp_eq_i32 s1, 0 // (op 615) 
s_cselect_b32 s8, s9, s0 // (op 615) 



v_lshl_add_u32 v14, s8, 4, v15 // (op 615) 


s_ashr_i64 s[0:1], s[34:35], 7 // (op 615) 
s_lshr_b32 s8, s0, 1 // (op 615) 

v_lshl_add_u32 v15, s8, 3, v25 // (op 615) 


v_mul_lo_u32 v25, v14, s26 // (op 615) 
v_add_u32 v25, v25, v15 // (op 615) 


s_mul_i32 s1, s26, s7 // (op 617) most significant: low of low * high
s_mul_i32 s8, s27, s6 // (op 617) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 617) least significant
s_mul_hi_u32 s9, s26, s6 // (op 617) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 617) most significant: sum
s_add_u32 s1, s1, s9 // (op 617) most significant: sum


s_mov_b32 s60, s28 // (op 617) reg expression
s_mov_b32 s61, s29 // (op 617) reg expression
s_mov_b32 s62, s0 // (op 617) reg expression
s_mov_b32 s63, 131072 // (op 617) call()

s_mul_hi_i32 s0, s16, s12 // (op 679) 
v_lshrrev_b32 v14, 2, v1 // (op 679) 
s_add_i32 s1, s0, s16 // (op 679) 

v_and_b32 v15, -4, v1 // (op 679) 
s_ashr_i32 s0, s1, 31 // (op 679) 
v_sub_u32 v16, v1, v15 // (op 679) 

s_cmp_eq_i32 s12, 0 // (op 679) 
s_mov_b32 s8, scc // (op 679) 
s_cmp_eq_i32 s8, 0 // (op 679) 
s_cselect_b32 s9, 0, -1 // (op 679) 

s_and_b32 s8, s13, 31 // (op 679) 
s_lshl_b32 s18, 1, s8 // (op 679) 
s_add_i32 s19, s18, s9 // (op 679) 


s_and_b32 s9, s0, s19 // (op 679) 


s_add_i32 s0, s1, s9 // (op 679) 


s_ashr_i32 s1, s0, s8 // (op 679) 


s_lshl_b32 s0, s13, 24 // (op 679) 
s_ashr_i32 s8, s0, 31 // (op 679) 

s_xor_b32 s0, s1, s8 // (op 679) 

s_sub_i32 s1, s0, s8 // (op 679) 


s_mul_hi_i32 s0, s1, s14 // (op 679) 
s_add_i32 s8, s0, s1 // (op 679) 

s_ashr_i32 s0, s8, 31 // (op 679) 
s_cmp_eq_i32 s14, 0 // (op 679) 
s_mov_b32 s9, scc // (op 679) 
s_cmp_eq_i32 s9, 0 // (op 679) 
s_cselect_b32 s18, 0, -1 // (op 679) 

s_and_b32 s9, s15, 31 // (op 679) 
s_lshl_b32 s19, 1, s9 // (op 679) 
s_add_i32 s49, s19, s18 // (op 679) 


s_and_b32 s18, s0, s49 // (op 679) 


s_add_i32 s0, s8, s18 // (op 679) 


s_ashr_i32 s8, s0, s9 // (op 679) 


s_lshl_b32 s0, s15, 24 // (op 679) 
s_ashr_i32 s9, s0, 31 // (op 679) 

s_xor_b32 s0, s8, s9 // (op 679) 

s_sub_i32 s8, s0, s9 // (op 679) 


s_add_u32 s18, s4, 32 // (op 679) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 679) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 679) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 679) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 679) 

s_mul_i32 s0, s8, s18 // (op 679) 

s_sub_i32 s8, s1, s0 // (op 679) 


s_mul_hi_i32 s9, s16, s11 // (op 679) 
s_add_i32 s0, s9, s16 // (op 679) 

s_ashr_i32 s1, s0, 31 // (op 679) 
s_cmp_eq_i32 s11, 0 // (op 679) 
s_mov_b32 s9, scc // (op 679) 
s_cmp_eq_i32 s9, 0 // (op 679) 
s_cselect_b32 s49, 0, -1 // (op 679) 

s_and_b32 s9, s46, 31 // (op 679) 
s_lshl_b32 s50, 1, s9 // (op 679) 
s_add_i32 s51, s50, s49 // (op 679) 


s_and_b32 s49, s1, s51 // (op 679) 


s_add_i32 s1, s0, s49 // (op 679) 


s_ashr_i32 s0, s1, s9 // (op 679) 


s_lshl_b32 s1, s46, 24 // (op 679) 
s_ashr_i32 s9, s1, 31 // (op 679) 

s_xor_b32 s1, s0, s9 // (op 679) 

s_sub_i32 s0, s1, s9 // (op 679) 


s_mul_i32 s1, s45, s18 // (op 679) 

s_mul_i32 s9, s10, s1 // (op 679) 

s_mul_i32 s1, s0, s9 // (op 679) 

s_sub_u32 s9, s16, s1 // (op 679) 

s_mul_hi_i32 s1, s9, s44 // (op 679) 
s_add_i32 s18, s1, s9 // (op 679) 


s_ashr_i32 s1, s18, 31 // (op 679) 
s_cmp_eq_i32 s44, 0 // (op 679) 
s_mov_b32 s9, scc // (op 679) 
s_cmp_eq_i32 s9, 0 // (op 679) 
s_cselect_b32 s19, 0, -1 // (op 679) 

s_and_b32 s9, s48, 31 // (op 679) 
s_lshl_b32 s49, 1, s9 // (op 679) 
s_add_i32 s50, s49, s19 // (op 679) 


s_and_b32 s19, s1, s50 // (op 679) 


s_add_i32 s1, s18, s19 // (op 679) 


s_ashr_i32 s18, s1, s9 // (op 679) 


s_lshl_b32 s1, s48, 24 // (op 679) 
s_ashr_i32 s9, s1, 31 // (op 679) 

s_xor_b32 s1, s18, s9 // (op 679) 

s_sub_i32 s18, s1, s9 // (op 679) 


s_cmp_eq_i32 s0, 0 // (op 679) 
s_mov_b32 s1, scc // (op 679) 

s_cmp_eq_i32 s1, 0 // (op 679) 
s_cselect_b32 s0, s18, s8 // (op 679) 



v_lshl_add_u32 v15, s0, 5, v14 // (op 679) 


v_mul_lo_u32 v14, v15, s30 // (op 679) 

s_ashr_i64 s[0:1], s[34:35], 7 // (op 679) 
s_lshr_b32 s8, s0, 1 // (op 679) 

v_lshl_add_u32 v15, s8, 3, v16 // (op 679) 


v_add_u32 v16, v15, v14 // (op 679) 


s_mul_i32 s1, s30, s5 // (op 681) most significant: low of low * high
s_mul_i32 s8, s31, s4 // (op 681) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 681) least significant
s_mul_hi_u32 s9, s30, s4 // (op 681) most significant: high of low * low
s_add_u32 s1, s1, s8 // (op 681) most significant: sum
s_add_u32 s1, s1, s9 // (op 681) most significant: sum


s_mov_b32 s64, s32 // (op 681) reg expression
s_mov_b32 s65, s33 // (op 681) reg expression
s_mov_b32 s66, s0 // (op 681) reg expression
s_mov_b32 s67, 131072 // (op 681) call()

v_and_b32 v14, -8, v1 // (op 772) 
v_sub_u32 v15, v1, v14 // (op 772) 

v_lshlrev_b32 v14, 4, v15 // (op 772) 

v_lshrrev_b32 v15, 3, v1 // (op 772) 
v_lshl_add_u32 v17, v15, 7, v14 // (op 772) 


s_nop 0
v_readfirstlane_b32 s0, v17 // (op 772) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 824) 
v_sub_u32 v15, v1, v14 // (op 824) 

v_lshlrev_b32 v14, 4, v15 // (op 824) 

v_lshrrev_b32 v15, 3, v1 // (op 824) 
v_lshl_add_u32 v17, v15, 7, v14 // (op 824) 


s_nop 0
v_readfirstlane_b32 s1, v17 // (op 824) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 876) 
v_sub_u32 v15, v1, v14 // (op 876) 

v_and_b32 v14, 4294967292, v1 // (op 876) 
v_add_u32 v17, v14, v15 // (op 876) 


s_ashr_i64 s[8:9], s[34:35], 7 // (op 166) 
s_and_b32 s18, s8, 4294967294 // (op 166) 

s_ashr_i64 s[8:9], s[34:35], 7 // (op 165) 
s_mov_b32 s19, s8 // (op 165) convert

s_cmp_lt_u32 s18, s19 // (op 165) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165:
s_barrier  // (op 1079) 
s_add_u32 s18, s18, 1 // (op 167) 


v_mov_b64 v[14:15], v[12:13] // (op 436) 
s_add_u32 s19, 4096, s0 // (op 436) 

s_mov_b32 m0, s19 // (op 436) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 436) Load value direct to lds


v_add_co_u32 v12, vcc, 128, v12 // (op 474) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 474) most significant half
s_barrier  // (op 1095) 

s_lshl_b64 s[8:9], s[22:23], 4 // (op 442) 
v_mov_b64 v[14:15], s[8:9] // (op 442) convert


v_mov_b64 v[18:19], v[10:11] // (op 442) 
v_mov_b64 v[20:21], v[18:19] // (op 442) 
s_mov_b32 m0, s1 // (op 442) reg expression
buffer_load_dwordx4 v20, s[56:59], 0 offen offset: 0 lds // (op 442) Load value direct to lds
v_add_co_u32 v20, s[8:9], v20, v14 // (op 442) least significant half
v_addc_co_u32 v21, s[8:9], 0, v21, s[8:9] // (op 442) most significant half

s_add_u32 m0, m0, 2048 // (op 442) 
buffer_load_dwordx4 v20, s[56:59], 0 offen offset: 0 lds // (op 442) Load value direct to lds



v_add_co_u32 v10, vcc, 128, v10 // (op 543) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 543) most significant half


v_mov_b32 v14, v16 // (op 184) 

buffer_load_ubyte v15, v14, s[64:67], 0 offen offset: 0 // (op 184) Load value

v_add_u32 v16, 4, v16 // (op 671) 
s_barrier  // (op 1111) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1085) 






ds_read_b128 v[26:29], v3 offset:4096 // (op 173) Load local data 
ds_read_b128 v[30:33], v3 offset:4160 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v17, v15 offset:12288 // (op 182) Store local data 

s_barrier  // (op 1101) 



ds_read_b128 v[34:37], v4 // (op 175) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 175) Load local data 


v_mov_b32 v14, v25 // (op 177) 

buffer_load_ubyte v15, v14, s[60:63], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v25, 4, v25 // (op 607) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1118) 



ds_read_u8 v14, v0 offset:12288 // (op 179) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 180) 




s_ashr_i64 s[8:9], s[34:35], 7 // (op 165) 
s_mov_b32 s19, s8 // (op 165) convert

s_cmp_lt_u32 s18, s19 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165 if true)
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165:





















s_barrier  // (op 1174) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_cvt_f16_f32 v4, v10 // (op 52) 
v_cvt_f16_f32 v5, v11 // (op 52) 
v_cvt_f16_f32 v6, v12 // (op 52) 
v_cvt_f16_f32 v7, v13 // (op 52) 

s_mul_hi_i32 s0, s16, s12 // (op 918) 
v_lshrrev_b32 v0, 7, v1 // (op 918) 
s_add_i32 s1, s0, s16 // (op 918) 

v_and_b32 v3, 63, v1 // (op 918) 
s_ashr_i32 s0, s1, 31 // (op 918) 
v_lshrrev_b32 v10, 4, v3 // (op 918) 
s_cmp_eq_i32 s12, 0 // (op 918) 
s_mov_b32 s2, scc // (op 918) 
v_lshlrev_b32 v11, 2, v10 // (op 918) 

s_cmp_eq_i32 s2, 0 // (op 918) 
s_cselect_b32 s3, 0, -1 // (op 918) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 918) 


s_and_b32 s2, s13, 31 // (op 918) 
v_lshrrev_b32 v0, 2, v3 // (op 918) 

s_lshl_b32 s8, 1, s2 // (op 918) 
v_and_b32 v3, 3, v0 // (op 918) 

s_add_i32 s9, s8, s3 // (op 918) 


v_and_b32 v0, 3, v1 // (op 918) 
s_and_b32 s3, s0, s9 // (op 918) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 918) 


s_add_i32 s0, s1, s3 // (op 918) 


v_lshrrev_b32 v0, 6, v1 // (op 918) 
s_ashr_i32 s1, s0, s2 // (op 918) 


v_and_b32 v3, 1, v0 // (op 918) 

s_lshl_b32 s0, s13, 24 // (op 918) 
v_lshl_add_u32 v0, v3, 4, v11 // (op 918) 


s_ashr_i32 s2, s0, 31 // (op 918) 

s_xor_b32 s0, s1, s2 // (op 918) 

s_sub_i32 s1, s0, s2 // (op 918) 


s_mul_hi_i32 s0, s1, s14 // (op 918) 
s_add_i32 s2, s0, s1 // (op 918) 

s_ashr_i32 s0, s2, 31 // (op 918) 
s_cmp_eq_i32 s14, 0 // (op 918) 
s_mov_b32 s3, scc // (op 918) 
s_cmp_eq_i32 s3, 0 // (op 918) 
s_cselect_b32 s8, 0, -1 // (op 918) 

s_and_b32 s3, s15, 31 // (op 918) 
s_lshl_b32 s9, 1, s3 // (op 918) 
s_add_i32 s18, s9, s8 // (op 918) 


s_and_b32 s8, s0, s18 // (op 918) 


s_add_i32 s0, s2, s8 // (op 918) 


s_ashr_i32 s2, s0, s3 // (op 918) 


s_lshl_b32 s0, s15, 24 // (op 918) 
s_ashr_i32 s3, s0, 31 // (op 918) 

s_xor_b32 s0, s2, s3 // (op 918) 

s_sub_i32 s2, s0, s3 // (op 918) 


s_mul_i32 s0, s1, s45 // (op 918) 
s_sub_u32 s3, s16, s0 // (op 918) 

s_mul_i32 s0, s2, s45 // (op 918) 
s_add_u32 s0, s0, s3 // (op 918) 

s_mul_hi_i32 s3, s16, s11 // (op 918) 
s_add_i32 s8, s3, s16 // (op 918) 

s_ashr_i32 s3, s8, 31 // (op 918) 
s_cmp_eq_i32 s11, 0 // (op 918) 
s_mov_b32 s9, scc // (op 918) 
s_cmp_eq_i32 s9, 0 // (op 918) 
s_cselect_b32 s18, 0, -1 // (op 918) 

s_and_b32 s9, s46, 31 // (op 918) 
s_lshl_b32 s19, 1, s9 // (op 918) 
s_add_i32 s20, s19, s18 // (op 918) 


s_and_b32 s18, s3, s20 // (op 918) 


s_add_i32 s3, s8, s18 // (op 918) 


s_ashr_i32 s8, s3, s9 // (op 918) 


s_lshl_b32 s3, s46, 24 // (op 918) 
s_ashr_i32 s9, s3, 31 // (op 918) 

s_xor_b32 s3, s8, s9 // (op 918) 

s_sub_i32 s8, s3, s9 // (op 918) 


s_add_u32 s18, s4, 32 // (op 918) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 918) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 918) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 918) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 918) 

s_mul_i32 s3, s45, s18 // (op 918) 
s_mul_i32 s9, s10, s3 // (op 918) 

s_mul_i32 s3, s8, s9 // (op 918) 

s_sub_u32 s9, s16, s3 // (op 918) 

s_mov_b32 s3, s9 // (op 918) 
s_mul_hi_i32 s20, s3, s44 // (op 918) 
s_add_i32 s21, s20, s3 // (op 918) 


s_ashr_i32 s3, s21, 31 // (op 918) 
s_cmp_eq_i32 s44, 0 // (op 918) 
s_mov_b32 s20, scc // (op 918) 
s_cmp_eq_i32 s20, 0 // (op 918) 
s_cselect_b32 s22, 0, -1 // (op 918) 

s_and_b32 s20, s48, 31 // (op 918) 
s_lshl_b32 s23, 1, s20 // (op 918) 
s_add_i32 s24, s23, s22 // (op 918) 


s_and_b32 s22, s3, s24 // (op 918) 


s_add_i32 s3, s21, s22 // (op 918) 


s_ashr_i32 s21, s3, s20 // (op 918) 


s_lshl_b32 s3, s48, 24 // (op 918) 
s_ashr_i32 s20, s3, 31 // (op 918) 

s_xor_b32 s3, s21, s20 // (op 918) 

s_sub_i32 s21, s3, s20 // (op 918) 


s_mul_i32 s3, s47, s45 // (op 918) 
s_add_u32 s22, s6, 16 // (op 918) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 918) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 918) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 918) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 918) 

s_sub_i32 s20, s22, s3 // (op 918) 

s_mul_i32 s22, s21, s20 // (op 918) 

s_sub_u32 s20, s9, s22 // (op 918) 


s_add_u32 s9, s20, s3 // (op 918) 


s_cmp_eq_i32 s8, 0 // (op 918) 
s_mov_b32 s3, scc // (op 918) 

s_cmp_eq_i32 s3, 0 // (op 918) 
s_cselect_b32 s8, s9, s0 // (op 918) 


s_mul_i32 s0, s2, s18 // (op 918) 


v_lshl_add_u32 v3, s8, 4, v10 // (op 918) 


s_sub_i32 s2, s1, s0 // (op 918) 


s_cmp_eq_i32 s3, 0 // (op 918) 
s_cselect_b32 s0, s21, s2 // (op 918) 



v_lshl_add_u32 v10, s0, 5, v0 // (op 918) 


v_mul_lo_u32 v0, v10, s36 // (op 918) 

v_add_lshl_u32 v10, v3, v0, 1 // (op 918) 


s_mul_i32 s9, s36, s5 // (op 920) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 920) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 920) least significant
s_mul_hi_u32 s1, s36, s4 // (op 920) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 920) most significant: sum
s_add_u32 s9, s9, s1 // (op 920) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 920) 

s_mov_b32 s32, s38 // (op 920) reg expression
s_mov_b32 s33, s39 // (op 920) reg expression
s_mov_b32 s34, s0 // (op 920) reg expression
s_mov_b32 s35, 131072 // (op 920) call()


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v10, s[32:35], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:
s_mul_hi_i32 s0, s16, s12 // (op 742) 
v_lshrrev_b32 v0, 7, v1 // (op 742) 
s_add_i32 s1, s0, s16 // (op 742) 

v_and_b32 v3, 63, v1 // (op 742) 
s_ashr_i32 s0, s1, 31 // (op 742) 
v_lshrrev_b32 v4, 4, v3 // (op 742) 
s_cmp_eq_i32 s12, 0 // (op 742) 
s_mov_b32 s2, scc // (op 742) 
v_lshlrev_b32 v5, 2, v4 // (op 742) 

s_cmp_eq_i32 s2, 0 // (op 742) 
s_cselect_b32 s3, 0, -1 // (op 742) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 742) 


s_and_b32 s2, s13, 31 // (op 742) 
v_lshrrev_b32 v0, 2, v3 // (op 742) 

s_lshl_b32 s8, 1, s2 // (op 742) 
v_and_b32 v3, 3, v0 // (op 742) 

s_add_i32 s9, s8, s3 // (op 742) 


v_and_b32 v0, 3, v1 // (op 742) 
s_and_b32 s3, s0, s9 // (op 742) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 742) 


s_add_i32 s0, s1, s3 // (op 742) 


v_lshrrev_b32 v0, 6, v1 // (op 742) 
s_ashr_i32 s1, s0, s2 // (op 742) 


v_and_b32 v3, 1, v0 // (op 742) 

s_lshl_b32 s0, s13, 24 // (op 742) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 742) 


s_ashr_i32 s2, s0, 31 // (op 742) 

s_xor_b32 s0, s1, s2 // (op 742) 

s_sub_i32 s1, s0, s2 // (op 742) 


s_mul_hi_i32 s0, s1, s14 // (op 742) 
s_add_i32 s2, s0, s1 // (op 742) 

s_ashr_i32 s0, s2, 31 // (op 742) 
s_cmp_eq_i32 s14, 0 // (op 742) 
s_mov_b32 s3, scc // (op 742) 
s_cmp_eq_i32 s3, 0 // (op 742) 
s_cselect_b32 s8, 0, -1 // (op 742) 

s_and_b32 s3, s15, 31 // (op 742) 
s_lshl_b32 s9, 1, s3 // (op 742) 
s_add_i32 s18, s9, s8 // (op 742) 


s_and_b32 s8, s0, s18 // (op 742) 


s_add_i32 s0, s2, s8 // (op 742) 


s_ashr_i32 s2, s0, s3 // (op 742) 


s_lshl_b32 s0, s15, 24 // (op 742) 
s_ashr_i32 s3, s0, 31 // (op 742) 

s_xor_b32 s0, s2, s3 // (op 742) 

s_sub_i32 s2, s0, s3 // (op 742) 


s_mul_i32 s0, s1, s45 // (op 742) 
s_sub_u32 s3, s16, s0 // (op 742) 

s_mul_i32 s0, s2, s45 // (op 742) 
s_add_u32 s0, s0, s3 // (op 742) 

s_mul_hi_i32 s3, s16, s11 // (op 742) 
s_add_i32 s8, s3, s16 // (op 742) 

s_ashr_i32 s3, s8, 31 // (op 742) 
s_cmp_eq_i32 s11, 0 // (op 742) 
s_mov_b32 s9, scc // (op 742) 
s_cmp_eq_i32 s9, 0 // (op 742) 
s_cselect_b32 s18, 0, -1 // (op 742) 

s_and_b32 s9, s46, 31 // (op 742) 
s_lshl_b32 s19, 1, s9 // (op 742) 
s_add_i32 s20, s19, s18 // (op 742) 


s_and_b32 s18, s3, s20 // (op 742) 


s_add_i32 s3, s8, s18 // (op 742) 


s_ashr_i32 s8, s3, s9 // (op 742) 


s_lshl_b32 s3, s46, 24 // (op 742) 
s_ashr_i32 s9, s3, 31 // (op 742) 

s_xor_b32 s3, s8, s9 // (op 742) 

s_sub_i32 s8, s3, s9 // (op 742) 


s_add_u32 s18, s4, 32 // (op 742) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 742) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 742) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 742) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 742) 

s_mul_i32 s3, s45, s18 // (op 742) 
s_mul_i32 s9, s10, s3 // (op 742) 

s_mul_i32 s3, s8, s9 // (op 742) 

s_sub_u32 s9, s16, s3 // (op 742) 

s_mov_b32 s3, s9 // (op 742) 
s_mul_hi_i32 s20, s3, s44 // (op 742) 
s_add_i32 s21, s20, s3 // (op 742) 


s_ashr_i32 s3, s21, 31 // (op 742) 
s_cmp_eq_i32 s44, 0 // (op 742) 
s_mov_b32 s20, scc // (op 742) 
s_cmp_eq_i32 s20, 0 // (op 742) 
s_cselect_b32 s22, 0, -1 // (op 742) 

s_and_b32 s20, s48, 31 // (op 742) 
s_lshl_b32 s23, 1, s20 // (op 742) 
s_add_i32 s24, s23, s22 // (op 742) 


s_and_b32 s22, s3, s24 // (op 742) 


s_add_i32 s3, s21, s22 // (op 742) 


s_ashr_i32 s21, s3, s20 // (op 742) 


s_lshl_b32 s3, s48, 24 // (op 742) 
s_ashr_i32 s20, s3, 31 // (op 742) 

s_xor_b32 s3, s21, s20 // (op 742) 

s_sub_i32 s21, s3, s20 // (op 742) 


s_mul_i32 s3, s47, s45 // (op 742) 
s_add_u32 s22, s6, 16 // (op 742) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 742) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 742) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 742) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 742) 

s_sub_i32 s20, s22, s3 // (op 742) 

s_mul_i32 s22, s21, s20 // (op 742) 

s_sub_u32 s20, s9, s22 // (op 742) 


s_add_u32 s9, s20, s3 // (op 742) 


s_cmp_eq_i32 s8, 0 // (op 742) 
s_mov_b32 s3, scc // (op 742) 

s_cmp_eq_i32 s3, 0 // (op 742) 
s_cselect_b32 s8, s9, s0 // (op 742) 


s_mul_i32 s0, s2, s18 // (op 742) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 742) 


s_sub_i32 s2, s1, s0 // (op 742) 


s_cmp_eq_i32 s3, 0 // (op 742) 
s_cselect_b32 s0, s21, s2 // (op 742) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 742) 


v_mul_lo_u32 v0, v4, s40 // (op 742) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 742) 


s_mul_i32 s9, s40, s5 // (op 744) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 744) most significant: low of high * low
s_mul_i32 s8, s40, s4 // (op 744) least significant
s_mul_hi_u32 s1, s40, s4 // (op 744) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 744) most significant: sum
s_add_u32 s9, s9, s1 // (op 744) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 744) 

s_mov_b32 s28, s42 // (op 744) reg expression
s_mov_b32 s29, s43 // (op 744) reg expression
s_mov_b32 s30, s0 // (op 744) reg expression
s_mov_b32 s31, 131072 // (op 744) call()





buffer_load_dwordx2 v[6:7], v4, s[28:31], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v4, v6 // (op 139) 
v_lshrrev_b32 v5, 16, v6 // (op 139) 
v_cvt_f32_f16 v5, v5 // (op 139) 
v_mul_f32 v10, v8, v4 // (op 139) 
v_mul_f32 v11, v8, v5 // (op 139) 
v_cvt_f32_f16 v4, v7 // (op 139) 
v_lshrrev_b32 v5, 16, v7 // (op 139) 
v_cvt_f32_f16 v5, v5 // (op 139) 
v_mul_f32 v12, v8, v4 // (op 139) 
v_mul_f32 v13, v8, v5 // (op 139) 


v_add_f32 v4, v10, v14 // (op 135) 
v_add_f32 v5, v11, v15 // (op 135) 
v_add_f32 v6, v12, v16 // (op 135) 
v_add_f32 v7, v13, v17 // (op 135) 


v_cvt_f16_f32 v10, v4 // (op 136) 
v_cvt_f16_f32 v11, v5 // (op 136) 
v_cvt_f16_f32 v12, v6 // (op 136) 
v_cvt_f16_f32 v13, v7 // (op 136) 

s_mul_hi_i32 s0, s16, s12 // (op 935) 
v_lshrrev_b32 v0, 7, v1 // (op 935) 
s_add_i32 s1, s0, s16 // (op 935) 

v_and_b32 v3, 63, v1 // (op 935) 
s_ashr_i32 s0, s1, 31 // (op 935) 
v_lshrrev_b32 v4, 4, v3 // (op 935) 
s_cmp_eq_i32 s12, 0 // (op 935) 
s_mov_b32 s2, scc // (op 935) 
v_lshlrev_b32 v5, 2, v4 // (op 935) 

s_cmp_eq_i32 s2, 0 // (op 935) 
s_cselect_b32 s3, 0, -1 // (op 935) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 935) 


s_and_b32 s2, s13, 31 // (op 935) 
v_lshrrev_b32 v0, 2, v3 // (op 935) 

s_lshl_b32 s8, 1, s2 // (op 935) 
v_and_b32 v3, 3, v0 // (op 935) 

s_add_i32 s9, s8, s3 // (op 935) 


v_and_b32 v0, 3, v1 // (op 935) 
s_and_b32 s3, s0, s9 // (op 935) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 935) 


s_add_i32 s0, s1, s3 // (op 935) 


v_lshrrev_b32 v0, 6, v1 // (op 935) 
s_ashr_i32 s1, s0, s2 // (op 935) 


v_and_b32 v3, 1, v0 // (op 935) 

s_lshl_b32 s0, s13, 24 // (op 935) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 935) 


s_ashr_i32 s2, s0, 31 // (op 935) 

s_xor_b32 s0, s1, s2 // (op 935) 

s_sub_i32 s1, s0, s2 // (op 935) 


s_mul_hi_i32 s0, s1, s14 // (op 935) 
s_add_i32 s2, s0, s1 // (op 935) 

s_ashr_i32 s0, s2, 31 // (op 935) 
s_cmp_eq_i32 s14, 0 // (op 935) 
s_mov_b32 s3, scc // (op 935) 
s_cmp_eq_i32 s3, 0 // (op 935) 
s_cselect_b32 s8, 0, -1 // (op 935) 

s_and_b32 s3, s15, 31 // (op 935) 
s_lshl_b32 s9, 1, s3 // (op 935) 
s_add_i32 s18, s9, s8 // (op 935) 


s_and_b32 s8, s0, s18 // (op 935) 


s_add_i32 s0, s2, s8 // (op 935) 


s_ashr_i32 s2, s0, s3 // (op 935) 


s_lshl_b32 s0, s15, 24 // (op 935) 
s_ashr_i32 s3, s0, 31 // (op 935) 

s_xor_b32 s0, s2, s3 // (op 935) 

s_sub_i32 s2, s0, s3 // (op 935) 


s_mul_i32 s0, s1, s45 // (op 935) 
s_sub_u32 s3, s16, s0 // (op 935) 

s_mul_i32 s0, s2, s45 // (op 935) 
s_add_u32 s0, s0, s3 // (op 935) 

s_mul_hi_i32 s3, s16, s11 // (op 935) 
s_add_i32 s8, s3, s16 // (op 935) 

s_ashr_i32 s3, s8, 31 // (op 935) 
s_cmp_eq_i32 s11, 0 // (op 935) 
s_mov_b32 s9, scc // (op 935) 
s_cmp_eq_i32 s9, 0 // (op 935) 
s_cselect_b32 s18, 0, -1 // (op 935) 

s_and_b32 s9, s46, 31 // (op 935) 
s_lshl_b32 s19, 1, s9 // (op 935) 
s_add_i32 s20, s19, s18 // (op 935) 


s_and_b32 s18, s3, s20 // (op 935) 


s_add_i32 s3, s8, s18 // (op 935) 


s_ashr_i32 s8, s3, s9 // (op 935) 


s_lshl_b32 s3, s46, 24 // (op 935) 
s_ashr_i32 s9, s3, 31 // (op 935) 

s_xor_b32 s3, s8, s9 // (op 935) 

s_sub_i32 s8, s3, s9 // (op 935) 


s_add_u32 s18, s4, 32 // (op 935) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 935) End of Int64 add, unlocking SCC
s_sub_u32 s20, s18, 1 // (op 935) least significant half; sets scc
s_subb_u32 s21, s19, 0 // (op 935) most significant half; uses scc

s_ashr_i64 s[18:19], s[20:21], 5 // (op 935) 

s_mul_i32 s3, s45, s18 // (op 935) 
s_mul_i32 s9, s10, s3 // (op 935) 

s_mul_i32 s3, s8, s9 // (op 935) 

s_sub_u32 s9, s16, s3 // (op 935) 

s_mov_b32 s3, s9 // (op 935) 
s_mul_hi_i32 s20, s3, s44 // (op 935) 
s_add_i32 s21, s20, s3 // (op 935) 


s_ashr_i32 s3, s21, 31 // (op 935) 
s_cmp_eq_i32 s44, 0 // (op 935) 
s_mov_b32 s20, scc // (op 935) 
s_cmp_eq_i32 s20, 0 // (op 935) 
s_cselect_b32 s22, 0, -1 // (op 935) 

s_and_b32 s20, s48, 31 // (op 935) 
s_lshl_b32 s23, 1, s20 // (op 935) 
s_add_i32 s24, s23, s22 // (op 935) 


s_and_b32 s22, s3, s24 // (op 935) 


s_add_i32 s3, s21, s22 // (op 935) 


s_ashr_i32 s21, s3, s20 // (op 935) 


s_lshl_b32 s3, s48, 24 // (op 935) 
s_ashr_i32 s20, s3, 31 // (op 935) 

s_xor_b32 s3, s21, s20 // (op 935) 

s_sub_i32 s21, s3, s20 // (op 935) 


s_mul_i32 s3, s47, s45 // (op 935) 
s_add_u32 s22, s6, 16 // (op 935) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 935) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 935) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 935) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 935) 

s_sub_i32 s20, s22, s3 // (op 935) 

s_mul_i32 s22, s21, s20 // (op 935) 

s_sub_u32 s20, s9, s22 // (op 935) 


s_add_u32 s9, s20, s3 // (op 935) 


s_cmp_eq_i32 s8, 0 // (op 935) 
s_mov_b32 s3, scc // (op 935) 

s_cmp_eq_i32 s3, 0 // (op 935) 
s_cselect_b32 s8, s9, s0 // (op 935) 


s_mul_i32 s0, s2, s18 // (op 935) 


v_lshl_add_u32 v3, s8, 4, v4 // (op 935) 


s_sub_i32 s2, s1, s0 // (op 935) 


s_cmp_eq_i32 s3, 0 // (op 935) 
s_cselect_b32 s0, s21, s2 // (op 935) 



v_lshl_add_u32 v4, s0, 5, v0 // (op 935) 


v_mul_lo_u32 v0, v4, s36 // (op 935) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 935) 


s_mul_i32 s9, s36, s5 // (op 937) most significant: low of low * high
s_mul_i32 s0, s37, s4 // (op 937) most significant: low of high * low
s_mul_i32 s8, s36, s4 // (op 937) least significant
s_mul_hi_u32 s1, s36, s4 // (op 937) most significant: high of low * low
s_add_u32 s9, s9, s0 // (op 937) most significant: sum
s_add_u32 s9, s9, s1 // (op 937) most significant: sum


s_lshl_b64 s[0:1], s[8:9], 1 // (op 937) 

s_mov_b32 s32, s38 // (op 937) reg expression
s_mov_b32 s33, s39 // (op 937) reg expression
s_mov_b32 s34, s0 // (op 937) reg expression
s_mov_b32 s35, 131072 // (op 937) call()


v_pack_b32_f16 v6, v10, v11 // (op 137) 
v_pack_b32_f16 v7, v12, v13 // (op 137) 


buffer_store_dwordx2 v[6:7], v4, s[32:35], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
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
    macrotile: [16, 32, 128]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 70
    .vgpr_count: 50
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 36
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: MagicMultiple_30
        .size: 4
        .offset: 152
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 164
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 168
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

