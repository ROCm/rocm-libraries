// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_,@function
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 63, v1 // (op 708) 
v_and_b32 v3, 48, v1 // (op 708) 
v_sub_u32 v4, v0, v3 // (op 708) 

v_lshrrev_b32 v3, 4, v0 // (op 708) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 708) 


v_lshrrev_b32 v3, 7, v1 // (op 708) 
v_lshlrev_b32 v4, 4, v3 // (op 708) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 708) 


v_lshlrev_b32 v0, 2, v3 // (op 708) 

v_lshrrev_b32 v3, 3, v0 // (op 708) 

v_and_b32 v0, 48, v1 // (op 761) 
v_and_b32 v4, 63, v1 // (op 761) 
v_sub_u32 v5, v4, v0 // (op 761) 

v_lshrrev_b32 v4, 1, v5 // (op 761) 
v_add_lshl_u32 v6, v0, v4, 5 // (op 761) 


v_and_b32 v0, 1, v5 // (op 761) 

v_lshlrev_b32 v4, 3, v0 // (op 761) 

v_lshrrev_b32 v0, 6, v1 // (op 761) 
v_and_b32 v5, 1, v0 // (op 761) 

v_lshl_add_u32 v0, v5, 4, v4 // (op 761) 


v_add_u32 v4, v6, v0 // (op 761) 


v_and_b32 v0, 63, v1 // (op 819) 
v_and_b32 v5, 48, v1 // (op 819) 
v_sub_u32 v6, v0, v5 // (op 819) 

v_lshrrev_b32 v5, 4, v0 // (op 819) 

v_lshl_add_u32 v0, v5, 5, v6 // (op 819) 


v_lshrrev_b32 v5, 7, v1 // (op 819) 
v_lshlrev_b32 v6, 4, v5 // (op 819) 

v_lshl_add_u32 v5, v0, 5, v6 // (op 819) 


v_lshlrev_b32 v0, 2, v5 // (op 819) 

v_lshrrev_b32 v5, 3, v0 // (op 819) 

v_and_b32 v0, 48, v1 // (op 843) 
v_and_b32 v6, 63, v1 // (op 843) 
v_sub_u32 v7, v6, v0 // (op 843) 

v_lshrrev_b32 v6, 1, v7 // (op 843) 
v_add_lshl_u32 v8, v0, v6, 5 // (op 843) 


v_and_b32 v0, 1, v7 // (op 843) 

v_lshlrev_b32 v6, 3, v0 // (op 843) 

v_lshrrev_b32 v0, 6, v1 // (op 843) 
v_and_b32 v7, 1, v0 // (op 843) 

v_lshl_add_u32 v0, v7, 4, v6 // (op 843) 


v_add_u32 v6, v8, v0 // (op 843) 


v_mov_b32 v0, s6 // (op 2) Move value
v_mov_b32 v7, s7 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
s_mul_hi_i32 s0, s16, s9 // (op 418) 
v_and_b32 v8, -2, v1 // (op 418) 
s_add_i32 s1, s0, s16 // (op 418) 

v_sub_u32 v9, v1, v8 // (op 418) 

s_ashr_i32 s0, s1, 31 // (op 418) 
v_lshlrev_b32 v8, 4, v9 // (op 418) 

s_cmp_eq_i32 s9, 0 // (op 418) 
s_mov_b32 s6, scc // (op 418) 
v_lshrrev_b32 v9, 1, v1 // (op 418) 
s_cmp_eq_i32 s6, 0 // (op 418) 
s_cselect_b32 s7, 0, -1 // (op 418) 

s_and_b32 s6, s10, 31 // (op 418) 
s_lshl_b32 s18, 1, s6 // (op 418) 
s_add_i32 s19, s18, s7 // (op 418) 


s_and_b32 s7, s0, s19 // (op 418) 


s_add_i32 s0, s1, s7 // (op 418) 


s_ashr_i32 s1, s0, s6 // (op 418) 


s_lshl_b32 s0, s10, 24 // (op 418) 
s_ashr_i32 s6, s0, 31 // (op 418) 

s_xor_b32 s0, s1, s6 // (op 418) 

s_sub_i32 s1, s0, s6 // (op 418) 


s_mul_hi_i32 s0, s1, s11 // (op 418) 
s_add_i32 s6, s0, s1 // (op 418) 

s_ashr_i32 s0, s6, 31 // (op 418) 
s_cmp_eq_i32 s11, 0 // (op 418) 
s_mov_b32 s7, scc // (op 418) 
s_cmp_eq_i32 s7, 0 // (op 418) 
s_cselect_b32 s18, 0, -1 // (op 418) 

s_and_b32 s7, s12, 31 // (op 418) 
s_lshl_b32 s19, 1, s7 // (op 418) 
s_add_i32 s51, s19, s18 // (op 418) 


s_and_b32 s18, s0, s51 // (op 418) 


s_add_i32 s0, s6, s18 // (op 418) 


s_ashr_i32 s6, s0, s7 // (op 418) 


s_lshl_b32 s0, s12, 24 // (op 418) 
s_ashr_i32 s7, s0, 31 // (op 418) 

s_xor_b32 s0, s6, s7 // (op 418) 

s_sub_i32 s6, s0, s7 // (op 418) 


s_mul_i32 s0, s1, s14 // (op 418) 

s_sub_u32 s1, s16, s0 // (op 418) 

s_mul_i32 s0, s6, s14 // (op 418) 
s_add_u32 s0, s0, s1 // (op 418) 


s_mul_hi_i32 s1, s16, s8 // (op 418) 
s_add_i32 s6, s1, s16 // (op 418) 

s_ashr_i32 s1, s6, 31 // (op 418) 
s_cmp_eq_i32 s8, 0 // (op 418) 
s_mov_b32 s7, scc // (op 418) 
s_cmp_eq_i32 s7, 0 // (op 418) 
s_cselect_b32 s18, 0, -1 // (op 418) 

s_and_b32 s7, s15, 31 // (op 418) 
s_lshl_b32 s19, 1, s7 // (op 418) 
s_add_i32 s51, s19, s18 // (op 418) 


s_and_b32 s18, s1, s51 // (op 418) 


s_add_i32 s1, s6, s18 // (op 418) 


s_ashr_i32 s6, s1, s7 // (op 418) 


s_lshl_b32 s1, s15, 24 // (op 418) 
s_ashr_i32 s7, s1, 31 // (op 418) 

s_xor_b32 s1, s6, s7 // (op 418) 

s_sub_i32 s6, s1, s7 // (op 418) 


s_add_u32 s18, s4, 32 // (op 418) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 418) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 418) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 418) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 418) 

s_mul_i32 s1, s14, s18 // (op 418) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 418) 

s_mul_i32 s1, s6, s7 // (op 418) 

s_sub_u32 s7, s16, s1 // (op 418) 

s_mov_b32 s1, s7 // (op 418) 
s_mul_hi_i32 s18, s1, s13 // (op 418) 
s_add_i32 s19, s18, s1 // (op 418) 


s_ashr_i32 s1, s19, 31 // (op 418) 
s_cmp_eq_i32 s13, 0 // (op 418) 
s_mov_b32 s18, scc // (op 418) 
s_cmp_eq_i32 s18, 0 // (op 418) 
s_cselect_b32 s51, 0, -1 // (op 418) 

s_and_b32 s18, s49, 31 // (op 418) 
s_lshl_b32 s52, 1, s18 // (op 418) 
s_add_i32 s53, s52, s51 // (op 418) 


s_and_b32 s51, s1, s53 // (op 418) 


s_add_i32 s1, s19, s51 // (op 418) 


s_ashr_i32 s19, s1, s18 // (op 418) 


s_lshl_b32 s1, s49, 24 // (op 418) 
s_ashr_i32 s18, s1, 31 // (op 418) 

s_xor_b32 s1, s19, s18 // (op 418) 

s_sub_i32 s19, s1, s18 // (op 418) 


s_mul_i32 s1, s48, s14 // (op 418) 
s_add_u32 s52, s20, 32 // (op 418) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 418) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 418) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 418) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 5 // (op 418) 

s_sub_i32 s18, s52, s1 // (op 418) 

s_mul_i32 s51, s19, s18 // (op 418) 


s_sub_u32 s18, s7, s51 // (op 418) 


s_add_u32 s7, s18, s1 // (op 418) 


s_cmp_eq_i32 s6, 0 // (op 418) 
s_mov_b32 s1, scc // (op 418) 

s_cmp_eq_i32 s1, 0 // (op 418) 
s_cselect_b32 s6, s7, s0 // (op 418) 



v_lshl_add_u32 v10, s6, 5, v8 // (op 418) 


v_mov_b32 v13, 0 // (op 418) convert
v_mov_b32 v12, v10 // (op 418) convert

v_mul_lo_u32 v11, v9, s3 // (op 418) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 418) most significant: high of low * low
v_mul_lo_u32 v10, v9, s2 // (op 418) least significant: low of low * low
v_add_u32 v11, v11, v8 // (op 418) most significant: sum


v_add_co_u32 v8, s[0:1], v12, v10 // (op 418) least significant half
v_addc_co_u32 v9, s[0:1], v13, v11, s[0:1] // (op 418) most significant half

v_lshlrev_b64 v[8:9], 2, v[8:9] // (op 418) 


v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 418) 

v_mov_b32 v8, v10 // (op 418) convert

s_mul_i32 s1, s2, s25 // (op 420) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 420) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 420) least significant
s_mul_hi_u32 s7, s2, s24 // (op 420) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 420) most significant: sum
s_add_u32 s1, s1, s7 // (op 420) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 420) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 420) 

s_mov_b32 s52, s22 // (op 420) reg expression
s_mov_b32 s53, s23 // (op 420) reg expression
s_mov_b32 s54, s0 // (op 420) reg expression
s_mov_b32 s55, 131072 // (op 420) call()

s_mul_hi_i32 s51, s16, s9 // (op 437) 
v_and_b32 v9, -2, v1 // (op 437) 
s_add_i32 s0, s51, s16 // (op 437) 

v_sub_u32 v10, v1, v9 // (op 437) 

s_ashr_i32 s1, s0, 31 // (op 437) 
v_lshlrev_b32 v9, 4, v10 // (op 437) 

s_cmp_eq_i32 s9, 0 // (op 437) 
s_mov_b32 s51, scc // (op 437) 
v_lshrrev_b32 v10, 1, v1 // (op 437) 
s_cmp_eq_i32 s51, 0 // (op 437) 
s_cselect_b32 s6, 0, -1 // (op 437) 

s_and_b32 s7, s10, 31 // (op 437) 
s_lshl_b32 s51, 1, s7 // (op 437) 
s_add_i32 s18, s51, s6 // (op 437) 


s_and_b32 s6, s1, s18 // (op 437) 


s_add_i32 s1, s0, s6 // (op 437) 


s_ashr_i32 s0, s1, s7 // (op 437) 


s_lshl_b32 s1, s10, 24 // (op 437) 
s_ashr_i32 s51, s1, 31 // (op 437) 

s_xor_b32 s1, s0, s51 // (op 437) 

s_sub_i32 s0, s1, s51 // (op 437) 


s_mul_hi_i32 s1, s0, s11 // (op 437) 
s_add_i32 s51, s1, s0 // (op 437) 

s_ashr_i32 s1, s51, 31 // (op 437) 
s_cmp_eq_i32 s11, 0 // (op 437) 
s_mov_b32 s6, scc // (op 437) 
s_cmp_eq_i32 s6, 0 // (op 437) 
s_cselect_b32 s7, 0, -1 // (op 437) 

s_and_b32 s6, s12, 31 // (op 437) 
s_lshl_b32 s18, 1, s6 // (op 437) 
s_add_i32 s19, s18, s7 // (op 437) 


s_and_b32 s7, s1, s19 // (op 437) 


s_add_i32 s1, s51, s7 // (op 437) 


s_ashr_i32 s7, s1, s6 // (op 437) 


s_lshl_b32 s1, s12, 24 // (op 437) 
s_ashr_i32 s6, s1, 31 // (op 437) 

s_xor_b32 s1, s7, s6 // (op 437) 

s_sub_i32 s7, s1, s6 // (op 437) 


s_mul_i32 s1, s0, s14 // (op 437) 

s_sub_u32 s0, s16, s1 // (op 437) 

s_mul_i32 s1, s7, s14 // (op 437) 
s_add_u32 s1, s1, s0 // (op 437) 


s_mul_hi_i32 s0, s16, s8 // (op 437) 
s_add_i32 s51, s0, s16 // (op 437) 

s_ashr_i32 s0, s51, 31 // (op 437) 
s_cmp_eq_i32 s8, 0 // (op 437) 
s_mov_b32 s6, scc // (op 437) 
s_cmp_eq_i32 s6, 0 // (op 437) 
s_cselect_b32 s7, 0, -1 // (op 437) 

s_and_b32 s6, s15, 31 // (op 437) 
s_lshl_b32 s18, 1, s6 // (op 437) 
s_add_i32 s19, s18, s7 // (op 437) 


s_and_b32 s7, s0, s19 // (op 437) 


s_add_i32 s0, s51, s7 // (op 437) 


s_ashr_i32 s7, s0, s6 // (op 437) 


s_lshl_b32 s0, s15, 24 // (op 437) 
s_ashr_i32 s6, s0, 31 // (op 437) 

s_xor_b32 s0, s7, s6 // (op 437) 

s_sub_i32 s7, s0, s6 // (op 437) 


s_add_u32 s18, s4, 32 // (op 437) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 437) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 437) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 437) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 437) 

s_mul_i32 s0, s14, s18 // (op 437) 

s_mul_i32 s6, s50, s0 // (op 437) 

s_mul_i32 s0, s7, s6 // (op 437) 

s_sub_u32 s6, s16, s0 // (op 437) 

s_mov_b32 s0, s6 // (op 437) 
s_mul_hi_i32 s51, s0, s13 // (op 437) 
s_add_i32 s18, s51, s0 // (op 437) 


s_ashr_i32 s0, s18, 31 // (op 437) 
s_cmp_eq_i32 s13, 0 // (op 437) 
s_mov_b32 s19, scc // (op 437) 
s_cmp_eq_i32 s19, 0 // (op 437) 
s_cselect_b32 s51, 0, -1 // (op 437) 

s_and_b32 s19, s49, 31 // (op 437) 
s_lshl_b32 s56, 1, s19 // (op 437) 
s_add_i32 s57, s56, s51 // (op 437) 


s_and_b32 s51, s0, s57 // (op 437) 


s_add_i32 s0, s18, s51 // (op 437) 


s_ashr_i32 s18, s0, s19 // (op 437) 


s_lshl_b32 s0, s49, 24 // (op 437) 
s_ashr_i32 s19, s0, 31 // (op 437) 

s_xor_b32 s0, s18, s19 // (op 437) 

s_sub_i32 s18, s0, s19 // (op 437) 


s_mul_i32 s0, s48, s14 // (op 437) 
s_add_u32 s56, s20, 32 // (op 437) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 437) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 437) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 437) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 437) 

s_sub_i32 s19, s56, s0 // (op 437) 

s_mul_i32 s51, s18, s19 // (op 437) 


s_sub_u32 s18, s6, s51 // (op 437) 


s_add_u32 s6, s18, s0 // (op 437) 


s_cmp_eq_i32 s7, 0 // (op 437) 
s_mov_b32 s0, scc // (op 437) 

s_cmp_eq_i32 s0, 0 // (op 437) 
s_cselect_b32 s7, s6, s1 // (op 437) 



v_lshl_add_u32 v11, s7, 5, v9 // (op 437) 


v_mov_b32 v13, 0 // (op 437) convert
v_mov_b32 v12, v11 // (op 437) convert

v_mul_lo_u32 v15, v10, s3 // (op 437) most significant: low of low * high
v_mul_hi_u32 v9, v10, s2 // (op 437) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 437) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 437) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 437) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 437) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 437) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 437) 

v_mov_b32 v9, v12 // (op 437) convert

s_mul_i32 s1, s2, s25 // (op 439) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 439) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 439) least significant
s_mul_hi_u32 s6, s2, s24 // (op 439) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 439) most significant: sum
s_add_u32 s1, s1, s6 // (op 439) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 439) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 439) 

s_mov_b32 s52, s22 // (op 439) reg expression
s_mov_b32 s53, s23 // (op 439) reg expression
s_mov_b32 s54, s0 // (op 439) reg expression
s_mov_b32 s55, 131072 // (op 439) call()

s_mul_hi_i32 s51, s16, s9 // (op 482) 
v_and_b32 v10, -2, v1 // (op 482) 
s_add_i32 s0, s51, s16 // (op 482) 

v_sub_u32 v11, v1, v10 // (op 482) 

s_ashr_i32 s1, s0, 31 // (op 482) 
v_lshlrev_b32 v10, 4, v11 // (op 482) 

s_cmp_eq_i32 s9, 0 // (op 482) 
s_mov_b32 s51, scc // (op 482) 
v_lshrrev_b32 v11, 1, v1 // (op 482) 
s_cmp_eq_i32 s51, 0 // (op 482) 
s_cselect_b32 s6, 0, -1 // (op 482) 

s_and_b32 s7, s10, 31 // (op 482) 
s_lshl_b32 s51, 1, s7 // (op 482) 
s_add_i32 s18, s51, s6 // (op 482) 


s_and_b32 s6, s1, s18 // (op 482) 


s_add_i32 s1, s0, s6 // (op 482) 


s_ashr_i32 s0, s1, s7 // (op 482) 


s_lshl_b32 s1, s10, 24 // (op 482) 
s_ashr_i32 s51, s1, 31 // (op 482) 

s_xor_b32 s1, s0, s51 // (op 482) 

s_sub_i32 s0, s1, s51 // (op 482) 


s_mul_hi_i32 s1, s0, s11 // (op 482) 
s_add_i32 s51, s1, s0 // (op 482) 

s_ashr_i32 s1, s51, 31 // (op 482) 
s_cmp_eq_i32 s11, 0 // (op 482) 
s_mov_b32 s6, scc // (op 482) 
s_cmp_eq_i32 s6, 0 // (op 482) 
s_cselect_b32 s7, 0, -1 // (op 482) 

s_and_b32 s6, s12, 31 // (op 482) 
s_lshl_b32 s18, 1, s6 // (op 482) 
s_add_i32 s19, s18, s7 // (op 482) 


s_and_b32 s7, s1, s19 // (op 482) 


s_add_i32 s1, s51, s7 // (op 482) 


s_ashr_i32 s7, s1, s6 // (op 482) 


s_lshl_b32 s1, s12, 24 // (op 482) 
s_ashr_i32 s6, s1, 31 // (op 482) 

s_xor_b32 s1, s7, s6 // (op 482) 

s_sub_i32 s7, s1, s6 // (op 482) 


s_add_u32 s18, s4, 32 // (op 482) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 482) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 482) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 482) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 482) 

s_mul_i32 s1, s7, s18 // (op 482) 

s_sub_i32 s51, s0, s1 // (op 482) 


s_mul_hi_i32 s0, s16, s8 // (op 482) 
s_add_i32 s1, s0, s16 // (op 482) 

s_ashr_i32 s0, s1, 31 // (op 482) 
s_cmp_eq_i32 s8, 0 // (op 482) 
s_mov_b32 s6, scc // (op 482) 
s_cmp_eq_i32 s6, 0 // (op 482) 
s_cselect_b32 s7, 0, -1 // (op 482) 

s_and_b32 s6, s15, 31 // (op 482) 
s_lshl_b32 s56, 1, s6 // (op 482) 
s_add_i32 s57, s56, s7 // (op 482) 


s_and_b32 s7, s0, s57 // (op 482) 


s_add_i32 s0, s1, s7 // (op 482) 


s_ashr_i32 s1, s0, s6 // (op 482) 


s_lshl_b32 s0, s15, 24 // (op 482) 
s_ashr_i32 s6, s0, 31 // (op 482) 

s_xor_b32 s0, s1, s6 // (op 482) 

s_sub_i32 s1, s0, s6 // (op 482) 


s_mul_i32 s0, s14, s18 // (op 482) 

s_mul_i32 s6, s50, s0 // (op 482) 

s_mul_i32 s0, s1, s6 // (op 482) 

s_sub_u32 s6, s16, s0 // (op 482) 

s_mul_hi_i32 s0, s6, s13 // (op 482) 
s_add_i32 s7, s0, s6 // (op 482) 


s_ashr_i32 s0, s7, 31 // (op 482) 
s_cmp_eq_i32 s13, 0 // (op 482) 
s_mov_b32 s6, scc // (op 482) 
s_cmp_eq_i32 s6, 0 // (op 482) 
s_cselect_b32 s18, 0, -1 // (op 482) 

s_and_b32 s6, s49, 31 // (op 482) 
s_lshl_b32 s19, 1, s6 // (op 482) 
s_add_i32 s56, s19, s18 // (op 482) 


s_and_b32 s18, s0, s56 // (op 482) 


s_add_i32 s0, s7, s18 // (op 482) 


s_ashr_i32 s7, s0, s6 // (op 482) 


s_lshl_b32 s0, s49, 24 // (op 482) 
s_ashr_i32 s6, s0, 31 // (op 482) 

s_xor_b32 s0, s7, s6 // (op 482) 

s_sub_i32 s7, s0, s6 // (op 482) 


s_cmp_eq_i32 s1, 0 // (op 482) 
s_mov_b32 s0, scc // (op 482) 

s_cmp_eq_i32 s0, 0 // (op 482) 
s_cselect_b32 s1, s7, s51 // (op 482) 



v_lshl_add_u32 v12, s1, 5, v10 // (op 482) 


v_mul_lo_u32 v10, v11, s26 // (op 482) 
v_add_u32 v10, v10, v12 // (op 482) 


s_mul_i32 s1, s26, s25 // (op 484) most significant: low of low * high
s_mul_i32 s51, s27, s24 // (op 484) most significant: low of high * low
s_mul_i32 s0, s26, s24 // (op 484) least significant
s_mul_hi_u32 s6, s26, s24 // (op 484) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 484) most significant: sum
s_add_u32 s1, s1, s6 // (op 484) most significant: sum


s_mov_b32 s56, s28 // (op 484) reg expression
s_mov_b32 s57, s29 // (op 484) reg expression
s_mov_b32 s58, s0 // (op 484) reg expression
s_mov_b32 s59, 131072 // (op 484) call()

s_mul_hi_i32 s51, s16, s9 // (op 501) 
v_and_b32 v11, -2, v1 // (op 501) 
s_add_i32 s0, s51, s16 // (op 501) 

v_sub_u32 v12, v1, v11 // (op 501) 

s_ashr_i32 s1, s0, 31 // (op 501) 
v_lshlrev_b32 v11, 4, v12 // (op 501) 

s_cmp_eq_i32 s9, 0 // (op 501) 
s_mov_b32 s51, scc // (op 501) 
v_lshrrev_b32 v12, 1, v1 // (op 501) 
s_cmp_eq_i32 s51, 0 // (op 501) 
s_cselect_b32 s6, 0, -1 // (op 501) 

s_and_b32 s7, s10, 31 // (op 501) 
s_lshl_b32 s51, 1, s7 // (op 501) 
s_add_i32 s18, s51, s6 // (op 501) 


s_and_b32 s6, s1, s18 // (op 501) 


s_add_i32 s1, s0, s6 // (op 501) 


s_ashr_i32 s0, s1, s7 // (op 501) 


s_lshl_b32 s1, s10, 24 // (op 501) 
s_ashr_i32 s51, s1, 31 // (op 501) 

s_xor_b32 s1, s0, s51 // (op 501) 

s_sub_i32 s0, s1, s51 // (op 501) 


s_mul_hi_i32 s1, s0, s11 // (op 501) 
s_add_i32 s51, s1, s0 // (op 501) 

s_ashr_i32 s1, s51, 31 // (op 501) 
s_cmp_eq_i32 s11, 0 // (op 501) 
s_mov_b32 s6, scc // (op 501) 
s_cmp_eq_i32 s6, 0 // (op 501) 
s_cselect_b32 s7, 0, -1 // (op 501) 

s_and_b32 s6, s12, 31 // (op 501) 
s_lshl_b32 s18, 1, s6 // (op 501) 
s_add_i32 s19, s18, s7 // (op 501) 


s_and_b32 s7, s1, s19 // (op 501) 


s_add_i32 s1, s51, s7 // (op 501) 


s_ashr_i32 s7, s1, s6 // (op 501) 


s_lshl_b32 s1, s12, 24 // (op 501) 
s_ashr_i32 s6, s1, 31 // (op 501) 

s_xor_b32 s1, s7, s6 // (op 501) 

s_sub_i32 s7, s1, s6 // (op 501) 


s_add_u32 s18, s4, 32 // (op 501) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 501) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 501) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 501) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 501) 

s_mul_i32 s1, s7, s18 // (op 501) 

s_sub_i32 s51, s0, s1 // (op 501) 


s_mul_hi_i32 s0, s16, s8 // (op 501) 
s_add_i32 s1, s0, s16 // (op 501) 

s_ashr_i32 s0, s1, 31 // (op 501) 
s_cmp_eq_i32 s8, 0 // (op 501) 
s_mov_b32 s6, scc // (op 501) 
s_cmp_eq_i32 s6, 0 // (op 501) 
s_cselect_b32 s7, 0, -1 // (op 501) 

s_and_b32 s6, s15, 31 // (op 501) 
s_lshl_b32 s60, 1, s6 // (op 501) 
s_add_i32 s61, s60, s7 // (op 501) 


s_and_b32 s7, s0, s61 // (op 501) 


s_add_i32 s0, s1, s7 // (op 501) 


s_ashr_i32 s1, s0, s6 // (op 501) 


s_lshl_b32 s0, s15, 24 // (op 501) 
s_ashr_i32 s6, s0, 31 // (op 501) 

s_xor_b32 s0, s1, s6 // (op 501) 

s_sub_i32 s1, s0, s6 // (op 501) 


s_mul_i32 s0, s14, s18 // (op 501) 

s_mul_i32 s6, s50, s0 // (op 501) 

s_mul_i32 s0, s1, s6 // (op 501) 

s_sub_u32 s6, s16, s0 // (op 501) 

s_mul_hi_i32 s0, s6, s13 // (op 501) 
s_add_i32 s7, s0, s6 // (op 501) 


s_ashr_i32 s0, s7, 31 // (op 501) 
s_cmp_eq_i32 s13, 0 // (op 501) 
s_mov_b32 s6, scc // (op 501) 
s_cmp_eq_i32 s6, 0 // (op 501) 
s_cselect_b32 s18, 0, -1 // (op 501) 

s_and_b32 s6, s49, 31 // (op 501) 
s_lshl_b32 s19, 1, s6 // (op 501) 
s_add_i32 s60, s19, s18 // (op 501) 


s_and_b32 s18, s0, s60 // (op 501) 


s_add_i32 s0, s7, s18 // (op 501) 


s_ashr_i32 s7, s0, s6 // (op 501) 


s_lshl_b32 s0, s49, 24 // (op 501) 
s_ashr_i32 s6, s0, 31 // (op 501) 

s_xor_b32 s0, s7, s6 // (op 501) 

s_sub_i32 s7, s0, s6 // (op 501) 


s_cmp_eq_i32 s1, 0 // (op 501) 
s_mov_b32 s0, scc // (op 501) 

s_cmp_eq_i32 s0, 0 // (op 501) 
s_cselect_b32 s1, s7, s51 // (op 501) 



v_lshl_add_u32 v13, s1, 5, v11 // (op 501) 


v_mul_lo_u32 v11, v12, s26 // (op 501) 
v_add_u32 v11, v11, v13 // (op 501) 


s_mul_i32 s1, s26, s25 // (op 503) most significant: low of low * high
s_mul_i32 s51, s27, s24 // (op 503) most significant: low of high * low
s_mul_i32 s0, s26, s24 // (op 503) least significant
s_mul_hi_u32 s6, s26, s24 // (op 503) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 503) most significant: sum
s_add_u32 s1, s1, s6 // (op 503) most significant: sum


s_mov_b32 s56, s28 // (op 503) reg expression
s_mov_b32 s57, s29 // (op 503) reg expression
s_mov_b32 s58, s0 // (op 503) reg expression
s_mov_b32 s59, 131072 // (op 503) call()

s_mul_hi_i32 s51, s16, s9 // (op 546) 
v_lshrrev_b32 v12, 7, v1 // (op 546) 
s_add_i32 s0, s51, s16 // (op 546) 

v_and_b32 v13, 15, v1 // (op 546) 
s_ashr_i32 s1, s0, 31 // (op 546) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 546) 


s_cmp_eq_i32 s9, 0 // (op 546) 
s_mov_b32 s51, scc // (op 546) 
v_and_b32 v12, 63, v1 // (op 546) 
s_cmp_eq_i32 s51, 0 // (op 546) 
s_cselect_b32 s6, 0, -1 // (op 546) 

v_lshrrev_b32 v13, 4, v12 // (op 546) 

s_and_b32 s7, s10, 31 // (op 546) 
s_lshl_b32 s51, 1, s7 // (op 546) 
s_add_i32 s18, s51, s6 // (op 546) 


s_and_b32 s6, s1, s18 // (op 546) 


s_add_i32 s1, s0, s6 // (op 546) 


s_ashr_i32 s0, s1, s7 // (op 546) 


s_lshl_b32 s1, s10, 24 // (op 546) 
s_ashr_i32 s51, s1, 31 // (op 546) 

s_xor_b32 s1, s0, s51 // (op 546) 

s_sub_i32 s0, s1, s51 // (op 546) 


s_mul_hi_i32 s1, s0, s11 // (op 546) 
s_add_i32 s51, s1, s0 // (op 546) 

s_ashr_i32 s1, s51, 31 // (op 546) 
s_cmp_eq_i32 s11, 0 // (op 546) 
s_mov_b32 s6, scc // (op 546) 
s_cmp_eq_i32 s6, 0 // (op 546) 
s_cselect_b32 s7, 0, -1 // (op 546) 

s_and_b32 s6, s12, 31 // (op 546) 
s_lshl_b32 s18, 1, s6 // (op 546) 
s_add_i32 s19, s18, s7 // (op 546) 


s_and_b32 s7, s1, s19 // (op 546) 


s_add_i32 s1, s51, s7 // (op 546) 


s_ashr_i32 s7, s1, s6 // (op 546) 


s_lshl_b32 s1, s12, 24 // (op 546) 
s_ashr_i32 s6, s1, 31 // (op 546) 

s_xor_b32 s1, s7, s6 // (op 546) 

s_sub_i32 s7, s1, s6 // (op 546) 


s_mul_i32 s1, s0, s14 // (op 546) 

s_sub_u32 s0, s16, s1 // (op 546) 

s_mul_i32 s1, s7, s14 // (op 546) 
s_add_u32 s1, s1, s0 // (op 546) 


s_mul_hi_i32 s0, s16, s8 // (op 546) 
s_add_i32 s51, s0, s16 // (op 546) 

s_ashr_i32 s0, s51, 31 // (op 546) 
s_cmp_eq_i32 s8, 0 // (op 546) 
s_mov_b32 s6, scc // (op 546) 
s_cmp_eq_i32 s6, 0 // (op 546) 
s_cselect_b32 s7, 0, -1 // (op 546) 

s_and_b32 s6, s15, 31 // (op 546) 
s_lshl_b32 s18, 1, s6 // (op 546) 
s_add_i32 s19, s18, s7 // (op 546) 


s_and_b32 s7, s0, s19 // (op 546) 


s_add_i32 s0, s51, s7 // (op 546) 


s_ashr_i32 s7, s0, s6 // (op 546) 


s_lshl_b32 s0, s15, 24 // (op 546) 
s_ashr_i32 s6, s0, 31 // (op 546) 

s_xor_b32 s0, s7, s6 // (op 546) 

s_sub_i32 s7, s0, s6 // (op 546) 


s_add_u32 s18, s4, 32 // (op 546) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 546) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 546) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 546) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 546) 

s_mul_i32 s0, s14, s18 // (op 546) 

s_mul_i32 s6, s50, s0 // (op 546) 

s_mul_i32 s0, s7, s6 // (op 546) 

s_sub_u32 s6, s16, s0 // (op 546) 

s_mov_b32 s0, s6 // (op 546) 
s_mul_hi_i32 s51, s0, s13 // (op 546) 
s_add_i32 s18, s51, s0 // (op 546) 


s_ashr_i32 s0, s18, 31 // (op 546) 
s_cmp_eq_i32 s13, 0 // (op 546) 
s_mov_b32 s19, scc // (op 546) 
s_cmp_eq_i32 s19, 0 // (op 546) 
s_cselect_b32 s51, 0, -1 // (op 546) 

s_and_b32 s19, s49, 31 // (op 546) 
s_lshl_b32 s60, 1, s19 // (op 546) 
s_add_i32 s61, s60, s51 // (op 546) 


s_and_b32 s51, s0, s61 // (op 546) 


s_add_i32 s0, s18, s51 // (op 546) 


s_ashr_i32 s18, s0, s19 // (op 546) 


s_lshl_b32 s0, s49, 24 // (op 546) 
s_ashr_i32 s19, s0, 31 // (op 546) 

s_xor_b32 s0, s18, s19 // (op 546) 

s_sub_i32 s18, s0, s19 // (op 546) 


s_mul_i32 s0, s48, s14 // (op 546) 
s_add_u32 s60, s20, 32 // (op 546) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 546) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 546) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 546) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 5 // (op 546) 

s_sub_i32 s19, s60, s0 // (op 546) 

s_mul_i32 s51, s18, s19 // (op 546) 


s_sub_u32 s18, s6, s51 // (op 546) 


s_add_u32 s6, s18, s0 // (op 546) 


s_cmp_eq_i32 s7, 0 // (op 546) 
s_mov_b32 s0, scc // (op 546) 

s_cmp_eq_i32 s0, 0 // (op 546) 
s_cselect_b32 s7, s6, s1 // (op 546) 



v_lshl_add_u32 v12, s7, 5, v14 // (op 546) 


v_mul_lo_u32 v14, v13, s30 // (op 546) 

v_add_u32 v13, v12, v14 // (op 546) 


s_mul_i32 s1, s30, s35 // (op 548) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 548) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 548) least significant
s_mul_hi_u32 s6, s30, s34 // (op 548) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 548) most significant: sum
s_add_u32 s1, s1, s6 // (op 548) most significant: sum


s_mov_b32 s60, s32 // (op 548) reg expression
s_mov_b32 s61, s33 // (op 548) reg expression
s_mov_b32 s62, s0 // (op 548) reg expression
s_mov_b32 s63, 131072 // (op 548) call()

s_mul_hi_i32 s51, s16, s9 // (op 565) 
v_lshrrev_b32 v12, 7, v1 // (op 565) 
s_add_i32 s0, s51, s16 // (op 565) 

v_and_b32 v14, 15, v1 // (op 565) 
s_ashr_i32 s1, s0, 31 // (op 565) 
v_lshl_add_u32 v15, v12, 4, v14 // (op 565) 


s_cmp_eq_i32 s9, 0 // (op 565) 
s_mov_b32 s51, scc // (op 565) 
v_and_b32 v12, 63, v1 // (op 565) 
s_cmp_eq_i32 s51, 0 // (op 565) 
s_cselect_b32 s6, 0, -1 // (op 565) 

v_lshrrev_b32 v14, 4, v12 // (op 565) 

s_and_b32 s7, s10, 31 // (op 565) 
s_lshl_b32 s51, 1, s7 // (op 565) 
s_add_i32 s18, s51, s6 // (op 565) 


s_and_b32 s6, s1, s18 // (op 565) 


s_add_i32 s1, s0, s6 // (op 565) 


s_ashr_i32 s0, s1, s7 // (op 565) 


s_lshl_b32 s1, s10, 24 // (op 565) 
s_ashr_i32 s51, s1, 31 // (op 565) 

s_xor_b32 s1, s0, s51 // (op 565) 

s_sub_i32 s0, s1, s51 // (op 565) 


s_mul_hi_i32 s1, s0, s11 // (op 565) 
s_add_i32 s51, s1, s0 // (op 565) 

s_ashr_i32 s1, s51, 31 // (op 565) 
s_cmp_eq_i32 s11, 0 // (op 565) 
s_mov_b32 s6, scc // (op 565) 
s_cmp_eq_i32 s6, 0 // (op 565) 
s_cselect_b32 s7, 0, -1 // (op 565) 

s_and_b32 s6, s12, 31 // (op 565) 
s_lshl_b32 s18, 1, s6 // (op 565) 
s_add_i32 s19, s18, s7 // (op 565) 


s_and_b32 s7, s1, s19 // (op 565) 


s_add_i32 s1, s51, s7 // (op 565) 


s_ashr_i32 s7, s1, s6 // (op 565) 


s_lshl_b32 s1, s12, 24 // (op 565) 
s_ashr_i32 s6, s1, 31 // (op 565) 

s_xor_b32 s1, s7, s6 // (op 565) 

s_sub_i32 s7, s1, s6 // (op 565) 


s_mul_i32 s1, s0, s14 // (op 565) 

s_sub_u32 s0, s16, s1 // (op 565) 

s_mul_i32 s1, s7, s14 // (op 565) 
s_add_u32 s1, s1, s0 // (op 565) 


s_mul_hi_i32 s0, s16, s8 // (op 565) 
s_add_i32 s51, s0, s16 // (op 565) 

s_ashr_i32 s0, s51, 31 // (op 565) 
s_cmp_eq_i32 s8, 0 // (op 565) 
s_mov_b32 s6, scc // (op 565) 
s_cmp_eq_i32 s6, 0 // (op 565) 
s_cselect_b32 s7, 0, -1 // (op 565) 

s_and_b32 s6, s15, 31 // (op 565) 
s_lshl_b32 s18, 1, s6 // (op 565) 
s_add_i32 s19, s18, s7 // (op 565) 


s_and_b32 s7, s0, s19 // (op 565) 


s_add_i32 s0, s51, s7 // (op 565) 


s_ashr_i32 s7, s0, s6 // (op 565) 


s_lshl_b32 s0, s15, 24 // (op 565) 
s_ashr_i32 s6, s0, 31 // (op 565) 

s_xor_b32 s0, s7, s6 // (op 565) 

s_sub_i32 s7, s0, s6 // (op 565) 


s_add_u32 s18, s4, 32 // (op 565) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 565) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 565) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 565) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 565) 

s_mul_i32 s0, s14, s18 // (op 565) 

s_mul_i32 s6, s50, s0 // (op 565) 

s_mul_i32 s0, s7, s6 // (op 565) 

s_sub_u32 s6, s16, s0 // (op 565) 

s_mov_b32 s0, s6 // (op 565) 
s_mul_hi_i32 s51, s0, s13 // (op 565) 
s_add_i32 s18, s51, s0 // (op 565) 


s_ashr_i32 s0, s18, 31 // (op 565) 
s_cmp_eq_i32 s13, 0 // (op 565) 
s_mov_b32 s19, scc // (op 565) 
s_cmp_eq_i32 s19, 0 // (op 565) 
s_cselect_b32 s51, 0, -1 // (op 565) 

s_and_b32 s19, s49, 31 // (op 565) 
s_lshl_b32 s64, 1, s19 // (op 565) 
s_add_i32 s65, s64, s51 // (op 565) 


s_and_b32 s51, s0, s65 // (op 565) 


s_add_i32 s0, s18, s51 // (op 565) 


s_ashr_i32 s18, s0, s19 // (op 565) 


s_lshl_b32 s0, s49, 24 // (op 565) 
s_ashr_i32 s19, s0, 31 // (op 565) 

s_xor_b32 s0, s18, s19 // (op 565) 

s_sub_i32 s18, s0, s19 // (op 565) 


s_mul_i32 s0, s48, s14 // (op 565) 
s_add_u32 s64, s20, 32 // (op 565) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 565) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 565) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 565) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 5 // (op 565) 

s_sub_i32 s19, s64, s0 // (op 565) 

s_mul_i32 s51, s18, s19 // (op 565) 


s_sub_u32 s18, s6, s51 // (op 565) 


s_add_u32 s6, s18, s0 // (op 565) 


s_cmp_eq_i32 s7, 0 // (op 565) 
s_mov_b32 s0, scc // (op 565) 

s_cmp_eq_i32 s0, 0 // (op 565) 
s_cselect_b32 s7, s6, s1 // (op 565) 



v_lshl_add_u32 v12, s7, 5, v15 // (op 565) 


v_mul_lo_u32 v15, v14, s30 // (op 565) 

v_add_u32 v14, v12, v15 // (op 565) 


s_mul_i32 s1, s30, s35 // (op 567) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 567) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 567) least significant
s_mul_hi_u32 s6, s30, s34 // (op 567) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 567) most significant: sum
s_add_u32 s1, s1, s6 // (op 567) most significant: sum


s_mov_b32 s60, s32 // (op 567) reg expression
s_mov_b32 s61, s33 // (op 567) reg expression
s_mov_b32 s62, s0 // (op 567) reg expression
s_mov_b32 s63, 131072 // (op 567) call()

s_mul_hi_i32 s51, s16, s9 // (op 610) 
v_lshrrev_b32 v12, 6, v1 // (op 610) 
s_add_i32 s0, s51, s16 // (op 610) 

v_and_b32 v15, 1, v12 // (op 610) 

s_ashr_i32 s1, s0, 31 // (op 610) 
v_and_b32 v12, 15, v1 // (op 610) 
s_cmp_eq_i32 s9, 0 // (op 610) 
s_mov_b32 s51, scc // (op 610) 
v_lshl_add_u32 v16, v15, 4, v12 // (op 610) 


s_cmp_eq_i32 s51, 0 // (op 610) 
s_cselect_b32 s6, 0, -1 // (op 610) 

v_and_b32 v12, 63, v1 // (op 610) 
s_and_b32 s7, s10, 31 // (op 610) 
v_lshrrev_b32 v15, 4, v12 // (op 610) 

s_lshl_b32 s51, 1, s7 // (op 610) 
s_add_i32 s18, s51, s6 // (op 610) 


s_and_b32 s6, s1, s18 // (op 610) 


s_add_i32 s1, s0, s6 // (op 610) 


s_ashr_i32 s0, s1, s7 // (op 610) 


s_lshl_b32 s1, s10, 24 // (op 610) 
s_ashr_i32 s51, s1, 31 // (op 610) 

s_xor_b32 s1, s0, s51 // (op 610) 

s_sub_i32 s0, s1, s51 // (op 610) 


s_mul_hi_i32 s1, s0, s11 // (op 610) 
s_add_i32 s51, s1, s0 // (op 610) 

s_ashr_i32 s1, s51, 31 // (op 610) 
s_cmp_eq_i32 s11, 0 // (op 610) 
s_mov_b32 s6, scc // (op 610) 
s_cmp_eq_i32 s6, 0 // (op 610) 
s_cselect_b32 s7, 0, -1 // (op 610) 

s_and_b32 s6, s12, 31 // (op 610) 
s_lshl_b32 s18, 1, s6 // (op 610) 
s_add_i32 s19, s18, s7 // (op 610) 


s_and_b32 s7, s1, s19 // (op 610) 


s_add_i32 s1, s51, s7 // (op 610) 


s_ashr_i32 s7, s1, s6 // (op 610) 


s_lshl_b32 s1, s12, 24 // (op 610) 
s_ashr_i32 s6, s1, 31 // (op 610) 

s_xor_b32 s1, s7, s6 // (op 610) 

s_sub_i32 s7, s1, s6 // (op 610) 


s_add_u32 s18, s4, 32 // (op 610) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 610) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 610) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 610) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 610) 

s_mul_i32 s1, s7, s18 // (op 610) 

s_sub_i32 s51, s0, s1 // (op 610) 


s_mul_hi_i32 s0, s16, s8 // (op 610) 
s_add_i32 s1, s0, s16 // (op 610) 

s_ashr_i32 s0, s1, 31 // (op 610) 
s_cmp_eq_i32 s8, 0 // (op 610) 
s_mov_b32 s6, scc // (op 610) 
s_cmp_eq_i32 s6, 0 // (op 610) 
s_cselect_b32 s7, 0, -1 // (op 610) 

s_and_b32 s6, s15, 31 // (op 610) 
s_lshl_b32 s64, 1, s6 // (op 610) 
s_add_i32 s65, s64, s7 // (op 610) 


s_and_b32 s7, s0, s65 // (op 610) 


s_add_i32 s0, s1, s7 // (op 610) 


s_ashr_i32 s1, s0, s6 // (op 610) 


s_lshl_b32 s0, s15, 24 // (op 610) 
s_ashr_i32 s6, s0, 31 // (op 610) 

s_xor_b32 s0, s1, s6 // (op 610) 

s_sub_i32 s1, s0, s6 // (op 610) 


s_mul_i32 s0, s14, s18 // (op 610) 

s_mul_i32 s6, s50, s0 // (op 610) 

s_mul_i32 s0, s1, s6 // (op 610) 

s_sub_u32 s6, s16, s0 // (op 610) 

s_mul_hi_i32 s0, s6, s13 // (op 610) 
s_add_i32 s7, s0, s6 // (op 610) 


s_ashr_i32 s0, s7, 31 // (op 610) 
s_cmp_eq_i32 s13, 0 // (op 610) 
s_mov_b32 s6, scc // (op 610) 
s_cmp_eq_i32 s6, 0 // (op 610) 
s_cselect_b32 s18, 0, -1 // (op 610) 

s_and_b32 s6, s49, 31 // (op 610) 
s_lshl_b32 s19, 1, s6 // (op 610) 
s_add_i32 s64, s19, s18 // (op 610) 


s_and_b32 s18, s0, s64 // (op 610) 


s_add_i32 s0, s7, s18 // (op 610) 


s_ashr_i32 s7, s0, s6 // (op 610) 


s_lshl_b32 s0, s49, 24 // (op 610) 
s_ashr_i32 s6, s0, 31 // (op 610) 

s_xor_b32 s0, s7, s6 // (op 610) 

s_sub_i32 s7, s0, s6 // (op 610) 


s_cmp_eq_i32 s1, 0 // (op 610) 
s_mov_b32 s0, scc // (op 610) 

s_cmp_eq_i32 s0, 0 // (op 610) 
s_cselect_b32 s1, s7, s51 // (op 610) 



v_lshl_add_u32 v12, s1, 5, v16 // (op 610) 


v_mul_lo_u32 v16, v15, s36 // (op 610) 
v_add_u32 v16, v16, v12 // (op 610) 


s_mul_i32 s1, s36, s35 // (op 612) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 612) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 612) least significant
s_mul_hi_u32 s6, s36, s34 // (op 612) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 612) most significant: sum
s_add_u32 s1, s1, s6 // (op 612) most significant: sum


s_mov_b32 s64, s38 // (op 612) reg expression
s_mov_b32 s65, s39 // (op 612) reg expression
s_mov_b32 s66, s0 // (op 612) reg expression
s_mov_b32 s67, 131072 // (op 612) call()

s_mul_hi_i32 s51, s16, s9 // (op 629) 
v_lshrrev_b32 v12, 6, v1 // (op 629) 
s_add_i32 s0, s51, s16 // (op 629) 

v_and_b32 v15, 1, v12 // (op 629) 

s_ashr_i32 s1, s0, 31 // (op 629) 
v_and_b32 v12, 15, v1 // (op 629) 
s_cmp_eq_i32 s9, 0 // (op 629) 
s_mov_b32 s51, scc // (op 629) 
v_lshl_add_u32 v17, v15, 4, v12 // (op 629) 


s_cmp_eq_i32 s51, 0 // (op 629) 
s_cselect_b32 s6, 0, -1 // (op 629) 

v_and_b32 v12, 63, v1 // (op 629) 
s_and_b32 s7, s10, 31 // (op 629) 
v_lshrrev_b32 v15, 4, v12 // (op 629) 

s_lshl_b32 s51, 1, s7 // (op 629) 
s_add_i32 s18, s51, s6 // (op 629) 


s_and_b32 s6, s1, s18 // (op 629) 


s_add_i32 s1, s0, s6 // (op 629) 


s_ashr_i32 s0, s1, s7 // (op 629) 


s_lshl_b32 s1, s10, 24 // (op 629) 
s_ashr_i32 s51, s1, 31 // (op 629) 

s_xor_b32 s1, s0, s51 // (op 629) 

s_sub_i32 s0, s1, s51 // (op 629) 


s_mul_hi_i32 s1, s0, s11 // (op 629) 
s_add_i32 s51, s1, s0 // (op 629) 

s_ashr_i32 s1, s51, 31 // (op 629) 
s_cmp_eq_i32 s11, 0 // (op 629) 
s_mov_b32 s6, scc // (op 629) 
s_cmp_eq_i32 s6, 0 // (op 629) 
s_cselect_b32 s7, 0, -1 // (op 629) 

s_and_b32 s6, s12, 31 // (op 629) 
s_lshl_b32 s18, 1, s6 // (op 629) 
s_add_i32 s19, s18, s7 // (op 629) 


s_and_b32 s7, s1, s19 // (op 629) 


s_add_i32 s1, s51, s7 // (op 629) 


s_ashr_i32 s7, s1, s6 // (op 629) 


s_lshl_b32 s1, s12, 24 // (op 629) 
s_ashr_i32 s6, s1, 31 // (op 629) 

s_xor_b32 s1, s7, s6 // (op 629) 

s_sub_i32 s7, s1, s6 // (op 629) 


s_add_u32 s18, s4, 32 // (op 629) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 629) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 629) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 629) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 629) 

s_mul_i32 s1, s7, s18 // (op 629) 

s_sub_i32 s51, s0, s1 // (op 629) 


s_mul_hi_i32 s0, s16, s8 // (op 629) 
s_add_i32 s1, s0, s16 // (op 629) 

s_ashr_i32 s0, s1, 31 // (op 629) 
s_cmp_eq_i32 s8, 0 // (op 629) 
s_mov_b32 s6, scc // (op 629) 
s_cmp_eq_i32 s6, 0 // (op 629) 
s_cselect_b32 s7, 0, -1 // (op 629) 

s_and_b32 s6, s15, 31 // (op 629) 
s_lshl_b32 s68, 1, s6 // (op 629) 
s_add_i32 s69, s68, s7 // (op 629) 


s_and_b32 s7, s0, s69 // (op 629) 


s_add_i32 s0, s1, s7 // (op 629) 


s_ashr_i32 s1, s0, s6 // (op 629) 


s_lshl_b32 s0, s15, 24 // (op 629) 
s_ashr_i32 s6, s0, 31 // (op 629) 

s_xor_b32 s0, s1, s6 // (op 629) 

s_sub_i32 s1, s0, s6 // (op 629) 


s_mul_i32 s0, s14, s18 // (op 629) 

s_mul_i32 s6, s50, s0 // (op 629) 

s_mul_i32 s0, s1, s6 // (op 629) 

s_sub_u32 s6, s16, s0 // (op 629) 

s_mul_hi_i32 s0, s6, s13 // (op 629) 
s_add_i32 s7, s0, s6 // (op 629) 


s_ashr_i32 s0, s7, 31 // (op 629) 
s_cmp_eq_i32 s13, 0 // (op 629) 
s_mov_b32 s6, scc // (op 629) 
s_cmp_eq_i32 s6, 0 // (op 629) 
s_cselect_b32 s18, 0, -1 // (op 629) 

s_and_b32 s6, s49, 31 // (op 629) 
s_lshl_b32 s19, 1, s6 // (op 629) 
s_add_i32 s68, s19, s18 // (op 629) 


s_and_b32 s18, s0, s68 // (op 629) 


s_add_i32 s0, s7, s18 // (op 629) 


s_ashr_i32 s7, s0, s6 // (op 629) 


s_lshl_b32 s0, s49, 24 // (op 629) 
s_ashr_i32 s6, s0, 31 // (op 629) 

s_xor_b32 s0, s7, s6 // (op 629) 

s_sub_i32 s7, s0, s6 // (op 629) 


s_cmp_eq_i32 s1, 0 // (op 629) 
s_mov_b32 s0, scc // (op 629) 

s_cmp_eq_i32 s0, 0 // (op 629) 
s_cselect_b32 s1, s7, s51 // (op 629) 



v_lshl_add_u32 v12, s1, 5, v17 // (op 629) 


v_mul_lo_u32 v17, v15, s36 // (op 629) 
v_add_u32 v17, v17, v12 // (op 629) 


s_mul_i32 s1, s36, s35 // (op 631) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 631) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 631) least significant
s_mul_hi_u32 s6, s36, s34 // (op 631) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 631) most significant: sum
s_add_u32 s1, s1, s6 // (op 631) most significant: sum


s_mov_b32 s64, s38 // (op 631) reg expression
s_mov_b32 s65, s39 // (op 631) reg expression
s_mov_b32 s66, s0 // (op 631) reg expression
s_mov_b32 s67, 131072 // (op 631) call()


v_mov_b64 v[18:19], s[30:31] // (op 308) convert

v_mov_b32 v12, v13 // (op 308) reg expression
v_mov_b32 v15, v12 // (op 308) 
buffer_load_ubyte v20, v15, s[60:63], 0 offen offset: 0 // (op 308) Load value




v_mov_b64 v[18:19], s[36:37] // (op 313) convert

v_mov_b32 v12, v16 // (op 313) reg expression
v_mov_b32 v15, v12 // (op 313) 
buffer_load_ubyte v21, v15, s[64:67], 0 offen offset: 0 // (op 313) Load value





v_mov_b32 v12, v10 // (op 317) reg expression

buffer_load_dwordx4 v[22:25], v12, s[56:59], 0 offen // (op 317) Load value



v_mov_b32 v12, v8 // (op 320) reg expression

buffer_load_dwordx2 v[18:19], v12, s[52:55], 0 offen // (op 320) Load value


v_mov_b64 v[26:27], s[30:31] // (op 324) convert

s_lshl_b32 s51, s30, 2 // (op 324) 
v_add_u32 v12, v13, s51 // (op 324) 

v_mov_b32 v15, v12 // (op 324) 
buffer_load_ubyte v28, v15, s[60:63], 0 offen offset: 0 // (op 324) Load value





v_mov_b64 v[12:13], s[36:37] // (op 329) convert

s_lshl_b32 s51, s36, 2 // (op 329) 
v_add_u32 v15, v16, s51 // (op 329) 

v_mov_b32 v26, v15 // (op 329) 
buffer_load_ubyte v27, v26, s[64:67], 0 offen offset: 0 // (op 329) Load value






s_lshl_b32 s51, s26, 7 // (op 333) 
v_add_u32 v26, v10, s51 // (op 333) 


buffer_load_dwordx4 v[30:33], v26, s[56:59], 0 offen // (op 333) Load value




s_lshl_b64 s[0:1], s[2:3], 7 // (op 336) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 336) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 336) 

v_add_u32 v10, v8, s0 // (op 336) 


buffer_load_dwordx2 v[12:13], v10, s[52:55], 0 offen // (op 336) Load value


v_and_b32 v8, -2, v1 // (op 749) 
v_sub_u32 v10, v1, v8 // (op 749) 

v_lshlrev_b32 v8, 4, v10 // (op 749) 

v_lshrrev_b32 v10, 1, v1 // (op 749) 
v_lshl_add_u32 v26, v10, 5, v8 // (op 749) 





s_waitcnt vmcnt(5)
ds_write_b128 v26, v[22:25] // (op 338) Store local data 

v_and_b32 v8, -2, v1 // (op 696) 
v_sub_u32 v10, v1, v8 // (op 696) 

v_lshlrev_b32 v8, 4, v10 // (op 696) 

v_lshrrev_b32 v10, 1, v1 // (op 696) 
v_lshl_add_u32 v26, v10, 5, v8 // (op 696) 


v_lshlrev_b32 v8, 2, v26 // (op 696) 

v_lshrrev_b32 v10, 3, v8 // (op 696) 




s_waitcnt vmcnt(4)
ds_write_b64 v10, v[18:19] offset:4096 // (op 339) Store local data 

v_and_b32 v8, -2, v1 // (op 667) 
v_sub_u32 v10, v1, v8 // (op 667) 

v_lshlrev_b32 v8, 4, v10 // (op 667) 

v_lshrrev_b32 v10, 1, v1 // (op 667) 
v_lshl_add_u32 v26, v10, 5, v8 // (op 667) 


v_lshlrev_b32 v8, 2, v26 // (op 667) 

v_lshrrev_b32 v10, 3, v8 // (op 667) 

v_and_b32 v8, -2, v1 // (op 720) 
v_sub_u32 v26, v1, v8 // (op 720) 

v_lshlrev_b32 v8, 4, v26 // (op 720) 

v_lshrrev_b32 v26, 1, v1 // (op 720) 
v_lshl_add_u32 v29, v26, 5, v8 // (op 720) 


v_and_b32 v8, -2, v1 // (op 807) 
v_sub_u32 v26, v1, v8 // (op 807) 

v_lshlrev_b32 v8, 4, v26 // (op 807) 

v_lshrrev_b32 v26, 1, v1 // (op 807) 
v_lshl_add_u32 v15, v26, 5, v8 // (op 807) 


v_lshlrev_b32 v8, 2, v15 // (op 807) 

v_lshrrev_b32 v26, 3, v8 // (op 807) 

v_and_b32 v8, -2, v1 // (op 831) 
v_sub_u32 v15, v1, v8 // (op 831) 

v_lshlrev_b32 v8, 4, v15 // (op 831) 

v_lshrrev_b32 v15, 1, v1 // (op 831) 
v_lshl_add_u32 v16, v15, 5, v8 // (op 831) 


s_mov_b32 s51, 0 // (op 51) call()
s_ashr_i64 s[0:1], s[24:25], 7 // (op 50) 
s_mov_b32 s6, s0 // (op 50) convert

s_and_b32 s7, s6, 4294967294 // (op 50) 

s_cmp_lt_u32 s51, s7 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_50 // (op 50) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_50 if false)
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_50:
s_add_u32 s51, s51, 2 // (op 52) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 872) 



ds_read_b64_tr_b4 v[34:35], v3 offset:4096 // (op 60) Transpose load local data 
ds_read_b64_tr_b4 v[36:37], v3 offset:4352 // (op 60) Transpose load local data 
s_barrier  // (op 885) 



ds_read_b64_tr_b8 v[38:39], v4 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:256 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:2048 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:2304 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:45], a[0:3], v20, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 76) 



v_mov_b64 v[34:35], s[30:31] // (op 70) convert

v_mov_b32 v8, v14 // (op 70) 
s_lshl_b32 s0, s30, 3 // (op 70) 
v_add_u32 v15, v8, s0 // (op 70) 

v_mov_b32 v36, v15 // (op 70) 
buffer_load_ubyte v20, v36, s[60:63], 0 offen offset: 0 // (op 70) Load value




v_mov_b64 v[34:35], s[36:37] // (op 74) convert

v_mov_b32 v15, v17 // (op 74) 
s_lshl_b32 s0, s36, 3 // (op 74) 
v_add_u32 v36, v15, s0 // (op 74) 

v_mov_b32 v37, v36 // (op 74) 
buffer_load_ubyte v21, v37, s[64:67], 0 offen offset: 0 // (op 74) Load value





v_mov_b32 v34, v11 // (op 10) 
s_lshl_b32 s0, s26, 8 // (op 10) 
v_add_u32 v35, v34, s0 // (op 10) 


buffer_load_dwordx4 v[22:25], v35, s[56:59], 0 offen // (op 10) Load value



v_mov_b32 v35, v9 // (op 4) 
s_lshl_b64 s[0:1], s[2:3], 7 // (op 4) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 4) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 4) 

s_lshl_b32 s6, s0, 1 // (op 4) 

v_add_u32 v36, v35, s6 // (op 4) 


buffer_load_dwordx2 v[18:19], v36, s[52:55], 0 offen // (op 4) Load value

s_barrier  // (op 908) 



s_waitcnt vmcnt(5)
ds_write_b128 v16, v[30:33] offset:6144 // (op 218) Store local data 
s_barrier  // (op 902) 



s_waitcnt vmcnt(4)
ds_write_b64 v26, v[12:13] offset:10240 // (op 208) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 899) 



ds_read_b64_tr_b4 v[36:37], v5 offset:10240 // (op 210) Transpose load local data 
ds_read_b64_tr_b4 v[38:39], v5 offset:10496 // (op 210) Transpose load local data 
s_barrier  // (op 905) 
s_barrier  // (op 955) 



ds_read_b64_tr_b8 v[40:41], v6 offset:6144 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v6 offset:6400 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v6 offset:8192 // (op 212) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v6 offset:8448 // (op 212) Transpose load local data 
s_barrier  // (op 961) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:39], v[40:47], a[0:3], v28, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 217) 



v_mov_b64 v[36:37], s[30:31] // (op 214) convert

s_lshl_b32 s0, s30, 2 // (op 214) 
s_mul_i32 s1, 3, s0 // (op 214) 

v_add_u32 v38, v8, s1 // (op 214) 

v_mov_b32 v39, v38 // (op 214) 
buffer_load_ubyte v28, v39, s[60:63], 0 offen offset: 0 // (op 214) Load value





s_lshl_b32 s0, s30, 3 // (op 557) 
v_add_u32 v14, v14, s0 // (op 557) 

v_mov_b64 v[36:37], s[36:37] // (op 216) convert

s_lshl_b32 s0, s36, 2 // (op 216) 
s_mul_i32 s1, 3, s0 // (op 216) 

v_add_u32 v8, v15, s1 // (op 216) 

v_mov_b32 v38, v8 // (op 216) 
buffer_load_ubyte v27, v38, s[64:67], 0 offen offset: 0 // (op 216) Load value






s_lshl_b32 s0, s26, 7 // (op 219) 
s_mul_i32 s1, 3, s0 // (op 219) 

s_lshl_b32 s0, s36, 3 // (op 621) 
v_add_u32 v17, v17, s0 // (op 621) 

v_add_u32 v8, v34, s1 // (op 219) 


buffer_load_dwordx4 v[30:33], v8, s[56:59], 0 offen // (op 219) Load value




s_lshl_b32 s0, s26, 8 // (op 493) 
v_add_u32 v11, v11, s0 // (op 493) 

s_lshl_b64 s[0:1], s[2:3], 7 // (op 207) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 207) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 207) 

s_mul_i32 s6, 3, s0 // (op 207) 

v_add_u32 v8, v35, s6 // (op 207) 


buffer_load_dwordx2 v[12:13], v8, s[52:55], 0 offen // (op 207) Load value


s_barrier  // (op 888) 
s_lshl_b64 s[0:1], s[2:3], 7 // (op 429) 



s_lshl_b64 s[6:7], s[0:1], 2 // (op 429) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 429) 

s_lshl_b32 s6, s0, 1 // (op 429) 

v_add_u32 v9, v9, s6 // (op 429) 

s_waitcnt vmcnt(5)
ds_write_b128 v29, v[22:25] // (op 13) Store local data 
s_barrier  // (op 875) 



s_waitcnt vmcnt(4)
ds_write_b64 v10, v[18:19] offset:4096 // (op 6) Store local data 
s_ashr_i64 s[0:1], s[24:25], 7 // (op 50) 
s_mov_b32 s6, s0 // (op 50) convert

s_and_b32 s7, s6, 4294967294 // (op 50) 

s_cmp_lt_u32 s51, s7 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_50 // (op 50) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_50 if true)
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_50:













s_mul_hi_i32 s51, s16, s9 // (op 392) 
v_and_b32 v29, -2, v1 // (op 392) 
s_add_i32 s0, s51, s16 // (op 392) 

v_sub_u32 v8, v1, v29 // (op 392) 

s_ashr_i32 s1, s0, 31 // (op 392) 
v_lshlrev_b32 v29, 4, v8 // (op 392) 

s_cmp_eq_i32 s9, 0 // (op 392) 
s_mov_b32 s51, scc // (op 392) 
v_lshrrev_b32 v8, 1, v1 // (op 392) 
s_cmp_eq_i32 s51, 0 // (op 392) 
s_cselect_b32 s6, 0, -1 // (op 392) 

s_and_b32 s7, s10, 31 // (op 392) 
s_lshl_b32 s51, 1, s7 // (op 392) 
s_add_i32 s18, s51, s6 // (op 392) 


s_and_b32 s6, s1, s18 // (op 392) 


s_add_i32 s1, s0, s6 // (op 392) 


s_ashr_i32 s0, s1, s7 // (op 392) 


s_lshl_b32 s1, s10, 24 // (op 392) 
s_ashr_i32 s51, s1, 31 // (op 392) 

s_xor_b32 s1, s0, s51 // (op 392) 

s_sub_i32 s0, s1, s51 // (op 392) 


s_mul_hi_i32 s1, s0, s11 // (op 392) 
s_add_i32 s51, s1, s0 // (op 392) 

s_ashr_i32 s1, s51, 31 // (op 392) 
s_cmp_eq_i32 s11, 0 // (op 392) 
s_mov_b32 s6, scc // (op 392) 
s_cmp_eq_i32 s6, 0 // (op 392) 
s_cselect_b32 s7, 0, -1 // (op 392) 

s_and_b32 s6, s12, 31 // (op 392) 
s_lshl_b32 s18, 1, s6 // (op 392) 
s_add_i32 s19, s18, s7 // (op 392) 


s_and_b32 s7, s1, s19 // (op 392) 


s_add_i32 s1, s51, s7 // (op 392) 


s_ashr_i32 s7, s1, s6 // (op 392) 


s_lshl_b32 s1, s12, 24 // (op 392) 
s_ashr_i32 s6, s1, 31 // (op 392) 

s_xor_b32 s1, s7, s6 // (op 392) 

s_sub_i32 s7, s1, s6 // (op 392) 


s_mul_i32 s1, s0, s14 // (op 392) 

s_sub_u32 s0, s16, s1 // (op 392) 

s_mul_i32 s1, s7, s14 // (op 392) 
s_add_u32 s1, s1, s0 // (op 392) 


s_mul_hi_i32 s0, s16, s8 // (op 392) 
s_add_i32 s51, s0, s16 // (op 392) 

s_ashr_i32 s0, s51, 31 // (op 392) 
s_cmp_eq_i32 s8, 0 // (op 392) 
s_mov_b32 s6, scc // (op 392) 
s_cmp_eq_i32 s6, 0 // (op 392) 
s_cselect_b32 s7, 0, -1 // (op 392) 

s_and_b32 s6, s15, 31 // (op 392) 
s_lshl_b32 s18, 1, s6 // (op 392) 
s_add_i32 s19, s18, s7 // (op 392) 


s_and_b32 s7, s0, s19 // (op 392) 


s_add_i32 s0, s51, s7 // (op 392) 


s_ashr_i32 s7, s0, s6 // (op 392) 


s_lshl_b32 s0, s15, 24 // (op 392) 
s_ashr_i32 s6, s0, 31 // (op 392) 

s_xor_b32 s0, s7, s6 // (op 392) 

s_sub_i32 s7, s0, s6 // (op 392) 


s_add_u32 s18, s4, 32 // (op 392) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 392) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 392) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 392) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 392) 

s_mul_i32 s0, s14, s18 // (op 392) 

s_mul_i32 s6, s50, s0 // (op 392) 

s_mul_i32 s0, s7, s6 // (op 392) 

s_sub_u32 s6, s16, s0 // (op 392) 

s_mov_b32 s0, s6 // (op 392) 
s_mul_hi_i32 s51, s0, s13 // (op 392) 
s_add_i32 s18, s51, s0 // (op 392) 


s_ashr_i32 s0, s18, 31 // (op 392) 
s_cmp_eq_i32 s13, 0 // (op 392) 
s_mov_b32 s19, scc // (op 392) 
s_cmp_eq_i32 s19, 0 // (op 392) 
s_cselect_b32 s51, 0, -1 // (op 392) 

s_and_b32 s19, s49, 31 // (op 392) 
s_lshl_b32 s68, 1, s19 // (op 392) 
s_add_i32 s69, s68, s51 // (op 392) 


s_and_b32 s51, s0, s69 // (op 392) 


s_add_i32 s0, s18, s51 // (op 392) 


s_ashr_i32 s18, s0, s19 // (op 392) 


s_lshl_b32 s0, s49, 24 // (op 392) 
s_ashr_i32 s19, s0, 31 // (op 392) 

s_xor_b32 s0, s18, s19 // (op 392) 

s_sub_i32 s18, s0, s19 // (op 392) 


s_mul_i32 s0, s48, s14 // (op 392) 
s_add_u32 s68, s20, 32 // (op 392) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 392) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 392) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 392) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 392) 

s_sub_i32 s19, s68, s0 // (op 392) 

s_mul_i32 s51, s18, s19 // (op 392) 


s_sub_u32 s18, s6, s51 // (op 392) 


s_add_u32 s6, s18, s0 // (op 392) 


s_cmp_eq_i32 s7, 0 // (op 392) 
s_mov_b32 s0, scc // (op 392) 

s_cmp_eq_i32 s0, 0 // (op 392) 
s_cselect_b32 s7, s6, s1 // (op 392) 



v_lshl_add_u32 v9, s7, 5, v29 // (op 392) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 392) 
v_mov_b32 v11, 0 // (op 392) convert
v_mov_b32 v10, v9 // (op 392) convert

s_lshr_b32 s51, s0, 1 // (op 392) 

v_lshl_add_u32 v9, s51, 8, v8 // (op 392) 


v_mul_lo_u32 v15, v9, s3 // (op 392) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 392) most significant: high of low * low
v_mul_lo_u32 v14, v9, s2 // (op 392) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 392) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v14 // (op 392) least significant half
v_addc_co_u32 v9, s[0:1], v11, v15, s[0:1] // (op 392) most significant half

v_lshlrev_b64 v[8:9], 2, v[8:9] // (op 392) 


v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 392) 

v_mov_b32 v29, v10 // (op 392) convert

s_mul_i32 s1, s2, s25 // (op 394) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 394) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 394) least significant
s_mul_hi_u32 s6, s2, s24 // (op 394) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 394) most significant: sum
s_add_u32 s1, s1, s6 // (op 394) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 394) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 394) 

s_mov_b32 s52, s22 // (op 394) reg expression
s_mov_b32 s53, s23 // (op 394) reg expression
s_mov_b32 s54, s0 // (op 394) reg expression
s_mov_b32 s55, 131072 // (op 394) call()

s_mul_hi_i32 s51, s16, s9 // (op 459) 
v_and_b32 v8, -2, v1 // (op 459) 
s_add_i32 s0, s51, s16 // (op 459) 

v_sub_u32 v9, v1, v8 // (op 459) 

s_ashr_i32 s1, s0, 31 // (op 459) 
v_lshlrev_b32 v8, 4, v9 // (op 459) 

s_cmp_eq_i32 s9, 0 // (op 459) 
s_mov_b32 s51, scc // (op 459) 
v_lshrrev_b32 v9, 1, v1 // (op 459) 
s_cmp_eq_i32 s51, 0 // (op 459) 
s_cselect_b32 s6, 0, -1 // (op 459) 

s_and_b32 s7, s10, 31 // (op 459) 
s_lshl_b32 s51, 1, s7 // (op 459) 
s_add_i32 s18, s51, s6 // (op 459) 


s_and_b32 s6, s1, s18 // (op 459) 


s_add_i32 s1, s0, s6 // (op 459) 


s_ashr_i32 s0, s1, s7 // (op 459) 


s_lshl_b32 s1, s10, 24 // (op 459) 
s_ashr_i32 s51, s1, 31 // (op 459) 

s_xor_b32 s1, s0, s51 // (op 459) 

s_sub_i32 s0, s1, s51 // (op 459) 


s_mul_hi_i32 s1, s0, s11 // (op 459) 
s_add_i32 s51, s1, s0 // (op 459) 

s_ashr_i32 s1, s51, 31 // (op 459) 
s_cmp_eq_i32 s11, 0 // (op 459) 
s_mov_b32 s6, scc // (op 459) 
s_cmp_eq_i32 s6, 0 // (op 459) 
s_cselect_b32 s7, 0, -1 // (op 459) 

s_and_b32 s6, s12, 31 // (op 459) 
s_lshl_b32 s18, 1, s6 // (op 459) 
s_add_i32 s19, s18, s7 // (op 459) 


s_and_b32 s7, s1, s19 // (op 459) 


s_add_i32 s1, s51, s7 // (op 459) 


s_ashr_i32 s7, s1, s6 // (op 459) 


s_lshl_b32 s1, s12, 24 // (op 459) 
s_ashr_i32 s6, s1, 31 // (op 459) 

s_xor_b32 s1, s7, s6 // (op 459) 

s_sub_i32 s7, s1, s6 // (op 459) 


s_add_u32 s18, s4, 32 // (op 459) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 459) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 459) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 459) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 459) 

s_mul_i32 s1, s7, s18 // (op 459) 

s_sub_i32 s51, s0, s1 // (op 459) 


s_mul_hi_i32 s0, s16, s8 // (op 459) 
s_add_i32 s1, s0, s16 // (op 459) 

s_ashr_i32 s0, s1, 31 // (op 459) 
s_cmp_eq_i32 s8, 0 // (op 459) 
s_mov_b32 s6, scc // (op 459) 
s_cmp_eq_i32 s6, 0 // (op 459) 
s_cselect_b32 s7, 0, -1 // (op 459) 

s_and_b32 s6, s15, 31 // (op 459) 
s_lshl_b32 s68, 1, s6 // (op 459) 
s_add_i32 s69, s68, s7 // (op 459) 


s_and_b32 s7, s0, s69 // (op 459) 


s_add_i32 s0, s1, s7 // (op 459) 


s_ashr_i32 s1, s0, s6 // (op 459) 


s_lshl_b32 s0, s15, 24 // (op 459) 
s_ashr_i32 s6, s0, 31 // (op 459) 

s_xor_b32 s0, s1, s6 // (op 459) 

s_sub_i32 s1, s0, s6 // (op 459) 


s_mul_i32 s0, s14, s18 // (op 459) 

s_mul_i32 s6, s50, s0 // (op 459) 

s_mul_i32 s0, s1, s6 // (op 459) 

s_sub_u32 s6, s16, s0 // (op 459) 

s_mul_hi_i32 s0, s6, s13 // (op 459) 
s_add_i32 s7, s0, s6 // (op 459) 


s_ashr_i32 s0, s7, 31 // (op 459) 
s_cmp_eq_i32 s13, 0 // (op 459) 
s_mov_b32 s6, scc // (op 459) 
s_cmp_eq_i32 s6, 0 // (op 459) 
s_cselect_b32 s18, 0, -1 // (op 459) 

s_and_b32 s6, s49, 31 // (op 459) 
s_lshl_b32 s19, 1, s6 // (op 459) 
s_add_i32 s68, s19, s18 // (op 459) 


s_and_b32 s18, s0, s68 // (op 459) 


s_add_i32 s0, s7, s18 // (op 459) 


s_ashr_i32 s7, s0, s6 // (op 459) 


s_lshl_b32 s0, s49, 24 // (op 459) 
s_ashr_i32 s6, s0, 31 // (op 459) 

s_xor_b32 s0, s7, s6 // (op 459) 

s_sub_i32 s7, s0, s6 // (op 459) 


s_cmp_eq_i32 s1, 0 // (op 459) 
s_mov_b32 s0, scc // (op 459) 

s_cmp_eq_i32 s0, 0 // (op 459) 
s_cselect_b32 s1, s7, s51 // (op 459) 



v_lshl_add_u32 v10, s1, 5, v8 // (op 459) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 459) 
s_lshr_b32 s51, s0, 1 // (op 459) 

v_lshl_add_u32 v8, s51, 8, v9 // (op 459) 


v_mul_lo_u32 v9, v8, s26 // (op 459) 
v_add_u32 v9, v9, v10 // (op 459) 


s_mul_i32 s1, s26, s25 // (op 461) most significant: low of low * high
s_mul_i32 s51, s27, s24 // (op 461) most significant: low of high * low
s_mul_i32 s0, s26, s24 // (op 461) least significant
s_mul_hi_u32 s6, s26, s24 // (op 461) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 461) most significant: sum
s_add_u32 s1, s1, s6 // (op 461) most significant: sum


s_mov_b32 s56, s28 // (op 461) reg expression
s_mov_b32 s57, s29 // (op 461) reg expression
s_mov_b32 s58, s0 // (op 461) reg expression
s_mov_b32 s59, 131072 // (op 461) call()

s_mul_hi_i32 s51, s16, s9 // (op 523) 
v_lshrrev_b32 v8, 7, v1 // (op 523) 
s_add_i32 s0, s51, s16 // (op 523) 

v_and_b32 v10, 15, v1 // (op 523) 
s_ashr_i32 s1, s0, 31 // (op 523) 
v_lshl_add_u32 v11, v8, 4, v10 // (op 523) 


s_cmp_eq_i32 s9, 0 // (op 523) 
s_mov_b32 s51, scc // (op 523) 
v_and_b32 v8, 63, v1 // (op 523) 
s_cmp_eq_i32 s51, 0 // (op 523) 
s_cselect_b32 s6, 0, -1 // (op 523) 

v_lshrrev_b32 v10, 4, v8 // (op 523) 

s_and_b32 s7, s10, 31 // (op 523) 
s_lshl_b32 s51, 1, s7 // (op 523) 
s_add_i32 s18, s51, s6 // (op 523) 


s_and_b32 s6, s1, s18 // (op 523) 


s_add_i32 s1, s0, s6 // (op 523) 


s_ashr_i32 s0, s1, s7 // (op 523) 


s_lshl_b32 s1, s10, 24 // (op 523) 
s_ashr_i32 s51, s1, 31 // (op 523) 

s_xor_b32 s1, s0, s51 // (op 523) 

s_sub_i32 s0, s1, s51 // (op 523) 


s_mul_hi_i32 s1, s0, s11 // (op 523) 
s_add_i32 s51, s1, s0 // (op 523) 

s_ashr_i32 s1, s51, 31 // (op 523) 
s_cmp_eq_i32 s11, 0 // (op 523) 
s_mov_b32 s6, scc // (op 523) 
s_cmp_eq_i32 s6, 0 // (op 523) 
s_cselect_b32 s7, 0, -1 // (op 523) 

s_and_b32 s6, s12, 31 // (op 523) 
s_lshl_b32 s18, 1, s6 // (op 523) 
s_add_i32 s19, s18, s7 // (op 523) 


s_and_b32 s7, s1, s19 // (op 523) 


s_add_i32 s1, s51, s7 // (op 523) 


s_ashr_i32 s7, s1, s6 // (op 523) 


s_lshl_b32 s1, s12, 24 // (op 523) 
s_ashr_i32 s6, s1, 31 // (op 523) 

s_xor_b32 s1, s7, s6 // (op 523) 

s_sub_i32 s7, s1, s6 // (op 523) 


s_mul_i32 s1, s0, s14 // (op 523) 

s_sub_u32 s0, s16, s1 // (op 523) 

s_mul_i32 s1, s7, s14 // (op 523) 
s_add_u32 s1, s1, s0 // (op 523) 


s_mul_hi_i32 s0, s16, s8 // (op 523) 
s_add_i32 s51, s0, s16 // (op 523) 

s_ashr_i32 s0, s51, 31 // (op 523) 
s_cmp_eq_i32 s8, 0 // (op 523) 
s_mov_b32 s6, scc // (op 523) 
s_cmp_eq_i32 s6, 0 // (op 523) 
s_cselect_b32 s7, 0, -1 // (op 523) 

s_and_b32 s6, s15, 31 // (op 523) 
s_lshl_b32 s18, 1, s6 // (op 523) 
s_add_i32 s19, s18, s7 // (op 523) 


s_and_b32 s7, s0, s19 // (op 523) 


s_add_i32 s0, s51, s7 // (op 523) 


s_ashr_i32 s7, s0, s6 // (op 523) 


s_lshl_b32 s0, s15, 24 // (op 523) 
s_ashr_i32 s6, s0, 31 // (op 523) 

s_xor_b32 s0, s7, s6 // (op 523) 

s_sub_i32 s7, s0, s6 // (op 523) 


s_add_u32 s18, s4, 32 // (op 523) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 523) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 523) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 523) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 523) 

s_mul_i32 s0, s14, s18 // (op 523) 

s_mul_i32 s6, s50, s0 // (op 523) 

s_mul_i32 s0, s7, s6 // (op 523) 

s_sub_u32 s6, s16, s0 // (op 523) 

s_mov_b32 s0, s6 // (op 523) 
s_mul_hi_i32 s51, s0, s13 // (op 523) 
s_add_i32 s18, s51, s0 // (op 523) 


s_ashr_i32 s0, s18, 31 // (op 523) 
s_cmp_eq_i32 s13, 0 // (op 523) 
s_mov_b32 s19, scc // (op 523) 
s_cmp_eq_i32 s19, 0 // (op 523) 
s_cselect_b32 s51, 0, -1 // (op 523) 

s_and_b32 s19, s49, 31 // (op 523) 
s_lshl_b32 s68, 1, s19 // (op 523) 
s_add_i32 s69, s68, s51 // (op 523) 


s_and_b32 s51, s0, s69 // (op 523) 


s_add_i32 s0, s18, s51 // (op 523) 


s_ashr_i32 s18, s0, s19 // (op 523) 


s_lshl_b32 s0, s49, 24 // (op 523) 
s_ashr_i32 s19, s0, 31 // (op 523) 

s_xor_b32 s0, s18, s19 // (op 523) 

s_sub_i32 s18, s0, s19 // (op 523) 


s_mul_i32 s0, s48, s14 // (op 523) 
s_add_u32 s68, s20, 32 // (op 523) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 523) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 523) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 523) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 523) 

s_sub_i32 s19, s68, s0 // (op 523) 

s_mul_i32 s51, s18, s19 // (op 523) 


s_sub_u32 s18, s6, s51 // (op 523) 


s_add_u32 s6, s18, s0 // (op 523) 


s_cmp_eq_i32 s7, 0 // (op 523) 
s_mov_b32 s0, scc // (op 523) 

s_cmp_eq_i32 s0, 0 // (op 523) 
s_cselect_b32 s7, s6, s1 // (op 523) 



v_lshl_add_u32 v8, s7, 5, v11 // (op 523) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 523) 
s_lshr_b32 s51, s0, 1 // (op 523) 

v_lshl_add_u32 v11, s51, 3, v10 // (op 523) 


v_mul_lo_u32 v10, v11, s30 // (op 523) 

v_add_u32 v11, v8, v10 // (op 523) 


s_mul_i32 s1, s30, s35 // (op 525) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 525) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 525) least significant
s_mul_hi_u32 s6, s30, s34 // (op 525) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 525) most significant: sum
s_add_u32 s1, s1, s6 // (op 525) most significant: sum


s_mov_b32 s60, s32 // (op 525) reg expression
s_mov_b32 s61, s33 // (op 525) reg expression
s_mov_b32 s62, s0 // (op 525) reg expression
s_mov_b32 s63, 131072 // (op 525) call()

s_mul_hi_i32 s51, s16, s9 // (op 587) 
v_lshrrev_b32 v8, 6, v1 // (op 587) 
s_add_i32 s0, s51, s16 // (op 587) 

v_and_b32 v10, 1, v8 // (op 587) 

s_ashr_i32 s1, s0, 31 // (op 587) 
v_and_b32 v8, 15, v1 // (op 587) 
s_cmp_eq_i32 s9, 0 // (op 587) 
s_mov_b32 s51, scc // (op 587) 
v_lshl_add_u32 v14, v10, 4, v8 // (op 587) 


s_cmp_eq_i32 s51, 0 // (op 587) 
s_cselect_b32 s6, 0, -1 // (op 587) 

v_and_b32 v8, 63, v1 // (op 587) 
s_and_b32 s7, s10, 31 // (op 587) 
v_lshrrev_b32 v10, 4, v8 // (op 587) 

s_lshl_b32 s51, 1, s7 // (op 587) 
s_add_i32 s18, s51, s6 // (op 587) 


s_and_b32 s6, s1, s18 // (op 587) 


s_add_i32 s1, s0, s6 // (op 587) 


s_ashr_i32 s0, s1, s7 // (op 587) 


s_lshl_b32 s1, s10, 24 // (op 587) 
s_ashr_i32 s51, s1, 31 // (op 587) 

s_xor_b32 s1, s0, s51 // (op 587) 

s_sub_i32 s0, s1, s51 // (op 587) 


s_mul_hi_i32 s1, s0, s11 // (op 587) 
s_add_i32 s51, s1, s0 // (op 587) 

s_ashr_i32 s1, s51, 31 // (op 587) 
s_cmp_eq_i32 s11, 0 // (op 587) 
s_mov_b32 s6, scc // (op 587) 
s_cmp_eq_i32 s6, 0 // (op 587) 
s_cselect_b32 s7, 0, -1 // (op 587) 

s_and_b32 s6, s12, 31 // (op 587) 
s_lshl_b32 s18, 1, s6 // (op 587) 
s_add_i32 s19, s18, s7 // (op 587) 


s_and_b32 s7, s1, s19 // (op 587) 


s_add_i32 s1, s51, s7 // (op 587) 


s_ashr_i32 s7, s1, s6 // (op 587) 


s_lshl_b32 s1, s12, 24 // (op 587) 
s_ashr_i32 s6, s1, 31 // (op 587) 

s_xor_b32 s1, s7, s6 // (op 587) 

s_sub_i32 s7, s1, s6 // (op 587) 


s_add_u32 s18, s4, 32 // (op 587) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 587) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 587) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 587) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 587) 

s_mul_i32 s1, s7, s18 // (op 587) 

s_sub_i32 s51, s0, s1 // (op 587) 


s_mul_hi_i32 s0, s16, s8 // (op 587) 
s_add_i32 s1, s0, s16 // (op 587) 

s_ashr_i32 s0, s1, 31 // (op 587) 
s_cmp_eq_i32 s8, 0 // (op 587) 
s_mov_b32 s6, scc // (op 587) 
s_cmp_eq_i32 s6, 0 // (op 587) 
s_cselect_b32 s7, 0, -1 // (op 587) 

s_and_b32 s6, s15, 31 // (op 587) 
s_lshl_b32 s68, 1, s6 // (op 587) 
s_add_i32 s69, s68, s7 // (op 587) 


s_and_b32 s7, s0, s69 // (op 587) 


s_add_i32 s0, s1, s7 // (op 587) 


s_ashr_i32 s1, s0, s6 // (op 587) 


s_lshl_b32 s0, s15, 24 // (op 587) 
s_ashr_i32 s6, s0, 31 // (op 587) 

s_xor_b32 s0, s1, s6 // (op 587) 

s_sub_i32 s1, s0, s6 // (op 587) 


s_mul_i32 s0, s14, s18 // (op 587) 

s_mul_i32 s6, s50, s0 // (op 587) 

s_mul_i32 s0, s1, s6 // (op 587) 

s_sub_u32 s6, s16, s0 // (op 587) 

s_mul_hi_i32 s0, s6, s13 // (op 587) 
s_add_i32 s7, s0, s6 // (op 587) 


s_ashr_i32 s0, s7, 31 // (op 587) 
s_cmp_eq_i32 s13, 0 // (op 587) 
s_mov_b32 s6, scc // (op 587) 
s_cmp_eq_i32 s6, 0 // (op 587) 
s_cselect_b32 s18, 0, -1 // (op 587) 

s_and_b32 s6, s49, 31 // (op 587) 
s_lshl_b32 s19, 1, s6 // (op 587) 
s_add_i32 s68, s19, s18 // (op 587) 


s_and_b32 s18, s0, s68 // (op 587) 


s_add_i32 s0, s7, s18 // (op 587) 


s_ashr_i32 s7, s0, s6 // (op 587) 


s_lshl_b32 s0, s49, 24 // (op 587) 
s_ashr_i32 s6, s0, 31 // (op 587) 

s_xor_b32 s0, s7, s6 // (op 587) 

s_sub_i32 s7, s0, s6 // (op 587) 


s_cmp_eq_i32 s1, 0 // (op 587) 
s_mov_b32 s0, scc // (op 587) 

s_cmp_eq_i32 s0, 0 // (op 587) 
s_cselect_b32 s1, s7, s51 // (op 587) 



v_lshl_add_u32 v8, s1, 5, v14 // (op 587) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 587) 
s_lshr_b32 s51, s0, 1 // (op 587) 

v_lshl_add_u32 v14, s51, 3, v10 // (op 587) 


v_mul_lo_u32 v10, v14, s36 // (op 587) 
v_add_u32 v10, v10, v8 // (op 587) 


s_mul_i32 s1, s36, s35 // (op 589) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 589) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 589) least significant
s_mul_hi_u32 s6, s36, s34 // (op 589) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 589) most significant: sum
s_add_u32 s1, s1, s6 // (op 589) most significant: sum


s_mov_b32 s64, s38 // (op 589) reg expression
s_mov_b32 s65, s39 // (op 589) reg expression
s_mov_b32 s66, s0 // (op 589) reg expression
s_mov_b32 s67, 131072 // (op 589) call()

v_and_b32 v8, -2, v1 // (op 680) 
v_sub_u32 v14, v1, v8 // (op 680) 

v_lshlrev_b32 v8, 4, v14 // (op 680) 

v_lshrrev_b32 v14, 1, v1 // (op 680) 
v_lshl_add_u32 v15, v14, 5, v8 // (op 680) 


v_lshlrev_b32 v8, 2, v15 // (op 680) 

v_lshrrev_b32 v14, 3, v8 // (op 680) 

v_and_b32 v8, -2, v1 // (op 733) 
v_sub_u32 v15, v1, v8 // (op 733) 

v_lshlrev_b32 v8, 4, v15 // (op 733) 

v_lshrrev_b32 v15, 1, v1 // (op 733) 
v_lshl_add_u32 v16, v15, 5, v8 // (op 733) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 153) 
s_and_b32 s51, s0, 4294967294 // (op 153) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 152) 
s_mov_b32 s6, s0 // (op 152) convert

s_cmp_lt_u32 s51, s6 // (op 152) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_152 // (op 152) Condition: Top (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_152 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_152:


s_add_u32 s51, s51, 1 // (op 154) 
v_mov_b32 v8, v29 // (op 157) 
v_mov_b32 v15, v8 // (op 157) reg expression

buffer_load_dwordx2 v[12:13], v15, s[52:55], 0 offen // (op 157) Load value


s_lshl_b64 s[0:1], s[2:3], 7 // (op 382) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 382) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 382) 

v_add_u32 v29, v29, s0 // (op 382) 



s_barrier  // (op 878) 
v_mov_b32 v8, v9 // (op 169) 



v_mov_b32 v15, v8 // (op 169) reg expression

buffer_load_dwordx4 v[18:21], v15, s[56:59], 0 offen // (op 169) Load value


s_lshl_b32 s0, s26, 7 // (op 451) 
v_add_u32 v9, v9, s0 // (op 451) 

s_waitcnt vmcnt(1)
ds_write_b64 v14, v[12:13] offset:4096 // (op 158) Store local data 

s_barrier  // (op 891) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 881) 






ds_read_b64_tr_b4 v[22:23], v3 offset:4096 // (op 160) Transpose load local data 
ds_read_b64_tr_b4 v[24:25], v3 offset:4352 // (op 160) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b128 v16, v[18:21] // (op 168) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 895) 



ds_read_b64_tr_b8 v[30:31], v4 // (op 162) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v4 offset:256 // (op 162) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v4 offset:2048 // (op 162) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:2304 // (op 162) Transpose load local data 

v_mov_b64 v[12:13], s[30:31] // (op 164) convert

v_mov_b32 v8, v11 // (op 164) 
v_mov_b32 v15, v8 // (op 164) reg expression
v_mov_b32 v17, v15 // (op 164) 
buffer_load_ubyte v18, v17, s[60:63], 0 offen offset: 0 // (op 164) Load value




s_lshl_b32 s0, s30, 2 // (op 515) 
v_add_u32 v11, v11, s0 // (op 515) 


v_mov_b64 v[12:13], s[36:37] // (op 166) convert

v_mov_b32 v8, v10 // (op 166) 
v_mov_b32 v15, v8 // (op 166) reg expression
v_mov_b32 v17, v15 // (op 166) 
buffer_load_ubyte v19, v17, s[64:67], 0 offen offset: 0 // (op 166) Load value




s_lshl_b32 s0, s36, 2 // (op 579) 
v_add_u32 v10, v10, s0 // (op 579) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[30:37], a[0:3], v18, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b001 // (op 167) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 152) 
s_mov_b32 s6, s0 // (op 152) convert

s_cmp_lt_u32 s51, s6 // (op 152) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_152 // (op 152) Condition: Bottom (jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_152 if true)
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_152:





















s_barrier  // (op 949) 




v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_114
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

s_mul_hi_i32 s0, s16, s9 // (op 775) 
v_lshrrev_b32 v3, 7, v1 // (op 775) 
s_add_i32 s1, s0, s16 // (op 775) 

v_and_b32 v4, 63, v1 // (op 775) 
s_ashr_i32 s0, s1, 31 // (op 775) 
v_lshrrev_b32 v5, 4, v4 // (op 775) 
s_cmp_eq_i32 s9, 0 // (op 775) 
s_mov_b32 s2, scc // (op 775) 
v_lshlrev_b32 v6, 2, v5 // (op 775) 

s_cmp_eq_i32 s2, 0 // (op 775) 
s_cselect_b32 s3, 0, -1 // (op 775) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 775) 


s_and_b32 s2, s10, 31 // (op 775) 
v_lshrrev_b32 v3, 2, v4 // (op 775) 

s_lshl_b32 s6, 1, s2 // (op 775) 
v_and_b32 v4, 3, v3 // (op 775) 

s_add_i32 s7, s6, s3 // (op 775) 


v_and_b32 v3, 3, v1 // (op 775) 
s_and_b32 s3, s0, s7 // (op 775) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 775) 


s_add_i32 s0, s1, s3 // (op 775) 


v_lshrrev_b32 v3, 6, v1 // (op 775) 
s_ashr_i32 s1, s0, s2 // (op 775) 


v_and_b32 v4, 1, v3 // (op 775) 

s_lshl_b32 s0, s10, 24 // (op 775) 
v_lshl_add_u32 v3, v4, 4, v6 // (op 775) 


s_ashr_i32 s2, s0, 31 // (op 775) 

s_xor_b32 s0, s1, s2 // (op 775) 

s_sub_i32 s1, s0, s2 // (op 775) 


s_mul_hi_i32 s0, s1, s11 // (op 775) 
s_add_i32 s2, s0, s1 // (op 775) 

s_ashr_i32 s0, s2, 31 // (op 775) 
s_cmp_eq_i32 s11, 0 // (op 775) 
s_mov_b32 s3, scc // (op 775) 
s_cmp_eq_i32 s3, 0 // (op 775) 
s_cselect_b32 s6, 0, -1 // (op 775) 

s_and_b32 s3, s12, 31 // (op 775) 
s_lshl_b32 s7, 1, s3 // (op 775) 
s_add_i32 s18, s7, s6 // (op 775) 


s_and_b32 s19, s0, s18 // (op 775) 


s_add_i32 s0, s2, s19 // (op 775) 


s_ashr_i32 s2, s0, s3 // (op 775) 


s_lshl_b32 s0, s12, 24 // (op 775) 
s_ashr_i32 s3, s0, 31 // (op 775) 

s_xor_b32 s0, s2, s3 // (op 775) 

s_sub_i32 s2, s0, s3 // (op 775) 


s_mul_i32 s0, s1, s14 // (op 775) 
s_sub_u32 s3, s16, s0 // (op 775) 

s_mul_i32 s0, s2, s14 // (op 775) 
s_add_u32 s0, s0, s3 // (op 775) 

s_mul_hi_i32 s3, s16, s8 // (op 775) 
s_add_i32 s6, s3, s16 // (op 775) 

s_ashr_i32 s3, s6, 31 // (op 775) 
s_cmp_eq_i32 s8, 0 // (op 775) 
s_mov_b32 s7, scc // (op 775) 
s_cmp_eq_i32 s7, 0 // (op 775) 
s_cselect_b32 s18, 0, -1 // (op 775) 

s_and_b32 s7, s15, 31 // (op 775) 
s_lshl_b32 s19, 1, s7 // (op 775) 
s_add_i32 s22, s19, s18 // (op 775) 


s_and_b32 s18, s3, s22 // (op 775) 


s_add_i32 s3, s6, s18 // (op 775) 


s_ashr_i32 s6, s3, s7 // (op 775) 


s_lshl_b32 s3, s15, 24 // (op 775) 
s_ashr_i32 s7, s3, 31 // (op 775) 

s_xor_b32 s3, s6, s7 // (op 775) 

s_sub_i32 s6, s3, s7 // (op 775) 


s_add_u32 s18, s4, 32 // (op 775) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 775) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 775) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 775) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 5 // (op 775) 

s_mul_i32 s3, s14, s18 // (op 775) 
s_mul_i32 s7, s50, s3 // (op 775) 

s_mul_i32 s3, s6, s7 // (op 775) 

s_sub_u32 s7, s16, s3 // (op 775) 

s_mov_b32 s3, s7 // (op 775) 
s_mul_hi_i32 s22, s3, s13 // (op 775) 
s_add_i32 s23, s22, s3 // (op 775) 


s_ashr_i32 s3, s23, 31 // (op 775) 
s_cmp_eq_i32 s13, 0 // (op 775) 
s_mov_b32 s22, scc // (op 775) 
s_cmp_eq_i32 s22, 0 // (op 775) 
s_cselect_b32 s24, 0, -1 // (op 775) 

s_and_b32 s22, s49, 31 // (op 775) 
s_lshl_b32 s25, 1, s22 // (op 775) 
s_add_i32 s26, s25, s24 // (op 775) 


s_and_b32 s24, s3, s26 // (op 775) 


s_add_i32 s3, s23, s24 // (op 775) 


s_ashr_i32 s23, s3, s22 // (op 775) 


s_lshl_b32 s3, s49, 24 // (op 775) 
s_ashr_i32 s22, s3, 31 // (op 775) 

s_xor_b32 s3, s23, s22 // (op 775) 

s_sub_i32 s23, s3, s22 // (op 775) 


s_mul_i32 s3, s48, s14 // (op 775) 
s_add_u32 s24, s20, 32 // (op 775) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 775) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 775) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 775) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 775) 

s_sub_i32 s22, s24, s3 // (op 775) 

s_mul_i32 s24, s23, s22 // (op 775) 

s_sub_u32 s22, s7, s24 // (op 775) 


s_add_u32 s7, s22, s3 // (op 775) 


s_cmp_eq_i32 s6, 0 // (op 775) 
s_mov_b32 s3, scc // (op 775) 

s_cmp_eq_i32 s3, 0 // (op 775) 
s_cselect_b32 s6, s7, s0 // (op 775) 


s_mul_i32 s0, s2, s18 // (op 775) 


v_lshl_add_u32 v4, s6, 5, v5 // (op 775) 


s_sub_i32 s2, s1, s0 // (op 775) 


s_cmp_eq_i32 s3, 0 // (op 775) 
s_cselect_b32 s22, s23, s2 // (op 775) 



v_lshl_add_u32 v5, s22, 5, v3 // (op 775) 


v_mul_lo_u32 v3, v5, s40 // (op 775) 

v_add_lshl_u32 v5, v4, v3, 2 // (op 775) 


s_mul_i32 s7, s40, s5 // (op 777) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 777) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 777) least significant
s_mul_hi_u32 s1, s40, s4 // (op 777) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 777) most significant: sum
s_add_u32 s7, s7, s1 // (op 777) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 777) 

s_mov_b32 s0, s42 // (op 777) reg expression
s_mov_b32 s1, s43 // (op 777) reg expression
s_mov_b32 s2, s18 // (op 777) reg expression
s_mov_b32 s3, 131072 // (op 777) call()




buffer_store_dwordx4 v[12:15], v5, s[0:3], 0 offen // (op 48) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_114
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_114:
s_mul_hi_i32 s6, s16, s9 // (op 650) 
v_lshrrev_b32 v3, 7, v1 // (op 650) 
s_add_i32 s7, s6, s16 // (op 650) 

v_and_b32 v4, 63, v1 // (op 650) 
s_ashr_i32 s6, s7, 31 // (op 650) 
v_lshrrev_b32 v5, 4, v4 // (op 650) 
s_cmp_eq_i32 s9, 0 // (op 650) 
s_mov_b32 s18, scc // (op 650) 
v_lshlrev_b32 v6, 2, v5 // (op 650) 

s_cmp_eq_i32 s18, 0 // (op 650) 
s_cselect_b32 s19, 0, -1 // (op 650) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 650) 


s_and_b32 s18, s10, 31 // (op 650) 
v_lshrrev_b32 v3, 2, v4 // (op 650) 

s_lshl_b32 s22, 1, s18 // (op 650) 
v_and_b32 v4, 3, v3 // (op 650) 

s_add_i32 s23, s22, s19 // (op 650) 


v_and_b32 v3, 3, v1 // (op 650) 
s_and_b32 s19, s6, s23 // (op 650) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 650) 


s_add_i32 s6, s7, s19 // (op 650) 


v_lshrrev_b32 v3, 6, v1 // (op 650) 
s_ashr_i32 s7, s6, s18 // (op 650) 


v_and_b32 v4, 1, v3 // (op 650) 

s_lshl_b32 s6, s10, 24 // (op 650) 
v_lshl_add_u32 v3, v4, 4, v6 // (op 650) 


s_ashr_i32 s18, s6, 31 // (op 650) 

s_xor_b32 s6, s7, s18 // (op 650) 

s_sub_i32 s7, s6, s18 // (op 650) 


s_mul_hi_i32 s6, s7, s11 // (op 650) 
s_add_i32 s18, s6, s7 // (op 650) 

s_ashr_i32 s6, s18, 31 // (op 650) 
s_cmp_eq_i32 s11, 0 // (op 650) 
s_mov_b32 s19, scc // (op 650) 
s_cmp_eq_i32 s19, 0 // (op 650) 
s_cselect_b32 s22, 0, -1 // (op 650) 

s_and_b32 s19, s12, 31 // (op 650) 
s_lshl_b32 s23, 1, s19 // (op 650) 
s_add_i32 s24, s23, s22 // (op 650) 


s_and_b32 s22, s6, s24 // (op 650) 


s_add_i32 s6, s18, s22 // (op 650) 


s_ashr_i32 s18, s6, s19 // (op 650) 


s_lshl_b32 s6, s12, 24 // (op 650) 
s_ashr_i32 s19, s6, 31 // (op 650) 

s_xor_b32 s6, s18, s19 // (op 650) 

s_sub_i32 s18, s6, s19 // (op 650) 


s_mul_i32 s6, s7, s14 // (op 650) 
s_sub_u32 s19, s16, s6 // (op 650) 

s_mul_i32 s6, s18, s14 // (op 650) 
s_add_u32 s6, s6, s19 // (op 650) 

s_mul_hi_i32 s19, s16, s8 // (op 650) 
s_add_i32 s22, s19, s16 // (op 650) 

s_ashr_i32 s19, s22, 31 // (op 650) 
s_cmp_eq_i32 s8, 0 // (op 650) 
s_mov_b32 s23, scc // (op 650) 
s_cmp_eq_i32 s23, 0 // (op 650) 
s_cselect_b32 s24, 0, -1 // (op 650) 

s_and_b32 s23, s15, 31 // (op 650) 
s_lshl_b32 s25, 1, s23 // (op 650) 
s_add_i32 s26, s25, s24 // (op 650) 


s_and_b32 s24, s19, s26 // (op 650) 


s_add_i32 s19, s22, s24 // (op 650) 


s_ashr_i32 s22, s19, s23 // (op 650) 


s_lshl_b32 s19, s15, 24 // (op 650) 
s_ashr_i32 s23, s19, 31 // (op 650) 

s_xor_b32 s19, s22, s23 // (op 650) 

s_sub_i32 s22, s19, s23 // (op 650) 


s_add_u32 s38, s4, 32 // (op 650) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 650) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 650) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 650) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 5 // (op 650) 

s_mul_i32 s19, s14, s38 // (op 650) 
s_mul_i32 s23, s50, s19 // (op 650) 

s_mul_i32 s19, s22, s23 // (op 650) 

s_sub_u32 s23, s16, s19 // (op 650) 

s_mov_b32 s19, s23 // (op 650) 
s_mul_hi_i32 s24, s19, s13 // (op 650) 
s_add_i32 s25, s24, s19 // (op 650) 


s_ashr_i32 s19, s25, 31 // (op 650) 
s_cmp_eq_i32 s13, 0 // (op 650) 
s_mov_b32 s24, scc // (op 650) 
s_cmp_eq_i32 s24, 0 // (op 650) 
s_cselect_b32 s26, 0, -1 // (op 650) 

s_and_b32 s24, s49, 31 // (op 650) 
s_lshl_b32 s27, 1, s24 // (op 650) 
s_add_i32 s28, s27, s26 // (op 650) 


s_and_b32 s26, s19, s28 // (op 650) 


s_add_i32 s19, s25, s26 // (op 650) 


s_ashr_i32 s25, s19, s24 // (op 650) 


s_lshl_b32 s19, s49, 24 // (op 650) 
s_ashr_i32 s24, s19, 31 // (op 650) 

s_xor_b32 s19, s25, s24 // (op 650) 

s_sub_i32 s25, s19, s24 // (op 650) 


s_mul_i32 s19, s48, s14 // (op 650) 
s_add_u32 s26, s20, 32 // (op 650) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 650) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 650) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 650) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 650) 

s_sub_i32 s24, s26, s19 // (op 650) 

s_mul_i32 s26, s25, s24 // (op 650) 

s_sub_u32 s24, s23, s26 // (op 650) 


s_add_u32 s23, s24, s19 // (op 650) 


s_cmp_eq_i32 s22, 0 // (op 650) 
s_mov_b32 s19, scc // (op 650) 

s_cmp_eq_i32 s19, 0 // (op 650) 
s_cselect_b32 s22, s23, s6 // (op 650) 


s_mul_i32 s6, s18, s38 // (op 650) 


v_lshl_add_u32 v4, s22, 5, v5 // (op 650) 


s_sub_i32 s18, s7, s6 // (op 650) 


s_cmp_eq_i32 s19, 0 // (op 650) 
s_cselect_b32 s6, s25, s18 // (op 650) 



v_lshl_add_u32 v5, s6, 5, v3 // (op 650) 


v_mul_lo_u32 v3, v5, s44 // (op 650) 

v_add_lshl_u32 v5, v4, v3, 2 // (op 650) 


s_mul_i32 s7, s44, s5 // (op 652) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 652) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 652) least significant
s_mul_hi_u32 s19, s44, s4 // (op 652) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 652) most significant: sum
s_add_u32 s7, s7, s19 // (op 652) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 652) 

s_mov_b32 s36, s46 // (op 652) reg expression
s_mov_b32 s37, s47 // (op 652) reg expression
s_mov_b32 s38, s18 // (op 652) reg expression
s_mov_b32 s39, 131072 // (op 652) call()





buffer_load_dwordx4 v[8:11], v5, s[36:39], 0 offen // (op 126) Load value



v_accvgpr_read v12, a0 // (op 123) 
v_accvgpr_read v13, a1 // (op 123) 
v_accvgpr_read v14, a2 // (op 123) 
v_accvgpr_read v15, a3 // (op 123) 
v_mul_f32 v16, v7, v12 // (op 123) 
v_mul_f32 v17, v7, v13 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v0, v8 // (op 127) 
v_mul_f32 v13, v0, v9 // (op 127) 
v_mul_f32 v14, v0, v10 // (op 127) 
v_mul_f32 v15, v0, v11 // (op 127) 

v_add_f32 v8, v12, v16 // (op 124) 
v_add_f32 v9, v13, v17 // (op 124) 
v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 


s_mul_hi_i32 s6, s16, s9 // (op 792) 
v_lshrrev_b32 v3, 7, v1 // (op 792) 
s_add_i32 s7, s6, s16 // (op 792) 

v_and_b32 v4, 63, v1 // (op 792) 
s_ashr_i32 s6, s7, 31 // (op 792) 
v_lshrrev_b32 v5, 4, v4 // (op 792) 
s_cmp_eq_i32 s9, 0 // (op 792) 
s_mov_b32 s18, scc // (op 792) 
v_lshlrev_b32 v6, 2, v5 // (op 792) 

s_cmp_eq_i32 s18, 0 // (op 792) 
s_cselect_b32 s19, 0, -1 // (op 792) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 792) 


s_and_b32 s18, s10, 31 // (op 792) 
v_lshrrev_b32 v3, 2, v4 // (op 792) 

s_lshl_b32 s22, 1, s18 // (op 792) 
v_and_b32 v4, 3, v3 // (op 792) 

s_add_i32 s23, s22, s19 // (op 792) 


v_and_b32 v3, 3, v1 // (op 792) 
s_and_b32 s19, s6, s23 // (op 792) 


v_lshl_add_u32 v6, v4, 2, v3 // (op 792) 


s_add_i32 s6, s7, s19 // (op 792) 


v_lshrrev_b32 v3, 6, v1 // (op 792) 
s_ashr_i32 s7, s6, s18 // (op 792) 


v_and_b32 v4, 1, v3 // (op 792) 

s_lshl_b32 s6, s10, 24 // (op 792) 
v_lshl_add_u32 v3, v4, 4, v6 // (op 792) 


s_ashr_i32 s18, s6, 31 // (op 792) 

s_xor_b32 s6, s7, s18 // (op 792) 

s_sub_i32 s7, s6, s18 // (op 792) 


s_mul_hi_i32 s6, s7, s11 // (op 792) 
s_add_i32 s18, s6, s7 // (op 792) 

s_ashr_i32 s6, s18, 31 // (op 792) 
s_cmp_eq_i32 s11, 0 // (op 792) 
s_mov_b32 s19, scc // (op 792) 
s_cmp_eq_i32 s19, 0 // (op 792) 
s_cselect_b32 s22, 0, -1 // (op 792) 

s_and_b32 s19, s12, 31 // (op 792) 
s_lshl_b32 s23, 1, s19 // (op 792) 
s_add_i32 s24, s23, s22 // (op 792) 


s_and_b32 s22, s6, s24 // (op 792) 


s_add_i32 s6, s18, s22 // (op 792) 


s_ashr_i32 s18, s6, s19 // (op 792) 


s_lshl_b32 s6, s12, 24 // (op 792) 
s_ashr_i32 s19, s6, 31 // (op 792) 

s_xor_b32 s6, s18, s19 // (op 792) 

s_sub_i32 s18, s6, s19 // (op 792) 


s_mul_i32 s6, s7, s14 // (op 792) 
s_sub_u32 s19, s16, s6 // (op 792) 

s_mul_i32 s6, s18, s14 // (op 792) 
s_add_u32 s6, s6, s19 // (op 792) 

s_mul_hi_i32 s19, s16, s8 // (op 792) 
s_add_i32 s22, s19, s16 // (op 792) 

s_ashr_i32 s19, s22, 31 // (op 792) 
s_cmp_eq_i32 s8, 0 // (op 792) 
s_mov_b32 s23, scc // (op 792) 
s_cmp_eq_i32 s23, 0 // (op 792) 
s_cselect_b32 s24, 0, -1 // (op 792) 

s_and_b32 s23, s15, 31 // (op 792) 
s_lshl_b32 s25, 1, s23 // (op 792) 
s_add_i32 s26, s25, s24 // (op 792) 


s_and_b32 s24, s19, s26 // (op 792) 


s_add_i32 s19, s22, s24 // (op 792) 


s_ashr_i32 s22, s19, s23 // (op 792) 


s_lshl_b32 s19, s15, 24 // (op 792) 
s_ashr_i32 s23, s19, 31 // (op 792) 

s_xor_b32 s19, s22, s23 // (op 792) 

s_sub_i32 s22, s19, s23 // (op 792) 


s_add_u32 s38, s4, 32 // (op 792) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 792) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 792) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 792) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 5 // (op 792) 

s_mul_i32 s19, s14, s38 // (op 792) 
s_mul_i32 s23, s50, s19 // (op 792) 

s_mul_i32 s19, s22, s23 // (op 792) 

s_sub_u32 s23, s16, s19 // (op 792) 

s_mov_b32 s19, s23 // (op 792) 
s_mul_hi_i32 s24, s19, s13 // (op 792) 
s_add_i32 s25, s24, s19 // (op 792) 


s_ashr_i32 s19, s25, 31 // (op 792) 
s_cmp_eq_i32 s13, 0 // (op 792) 
s_mov_b32 s24, scc // (op 792) 
s_cmp_eq_i32 s24, 0 // (op 792) 
s_cselect_b32 s26, 0, -1 // (op 792) 

s_and_b32 s24, s49, 31 // (op 792) 
s_lshl_b32 s27, 1, s24 // (op 792) 
s_add_i32 s28, s27, s26 // (op 792) 


s_and_b32 s26, s19, s28 // (op 792) 


s_add_i32 s19, s25, s26 // (op 792) 


s_ashr_i32 s25, s19, s24 // (op 792) 


s_lshl_b32 s19, s49, 24 // (op 792) 
s_ashr_i32 s24, s19, 31 // (op 792) 

s_xor_b32 s19, s25, s24 // (op 792) 

s_sub_i32 s25, s19, s24 // (op 792) 


s_mul_i32 s19, s48, s14 // (op 792) 
s_add_u32 s26, s20, 32 // (op 792) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 792) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 792) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 792) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 792) 

s_sub_i32 s24, s26, s19 // (op 792) 

s_mul_i32 s26, s25, s24 // (op 792) 

s_sub_u32 s24, s23, s26 // (op 792) 


s_add_u32 s23, s24, s19 // (op 792) 


s_cmp_eq_i32 s22, 0 // (op 792) 
s_mov_b32 s19, scc // (op 792) 

s_cmp_eq_i32 s19, 0 // (op 792) 
s_cselect_b32 s22, s23, s6 // (op 792) 


s_mul_i32 s6, s18, s38 // (op 792) 


v_lshl_add_u32 v4, s22, 5, v5 // (op 792) 


s_sub_i32 s18, s7, s6 // (op 792) 


s_cmp_eq_i32 s19, 0 // (op 792) 
s_cselect_b32 s6, s25, s18 // (op 792) 



v_lshl_add_u32 v5, s6, 5, v3 // (op 792) 


v_mul_lo_u32 v3, v5, s40 // (op 792) 

v_add_lshl_u32 v5, v4, v3, 2 // (op 792) 


s_mul_i32 s7, s40, s5 // (op 794) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 794) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 794) least significant
s_mul_hi_u32 s19, s40, s4 // (op 794) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 794) most significant: sum
s_add_u32 s7, s7, s19 // (op 794) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 2 // (op 794) 

s_mov_b32 s0, s42 // (op 794) reg expression
s_mov_b32 s1, s43 // (op 794) reg expression
s_mov_b32 s2, s18 // (op 794) reg expression
s_mov_b32 s3, 131072 // (op 794) call()




buffer_store_dwordx4 v[8:11], v5, s[0:3], 0 offen // (op 125) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_114:
















s_endpgm  // End of RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.LRR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__end:
.size RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_, .LRR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__end-RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12288 // lds bytes
  .amdhsa_accum_offset 48
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DataTypeA: F4
    DataTypeB: B8
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: True
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
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int64, semantic: SizeFree0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: int32, semantic: RR_QuotientTilesByBlock } ]
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP4_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 12288
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 48
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_2_size_1_11
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 16
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 20
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_0_size_0_15
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 152
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Subtract_24
        .size: 4
        .offset: 168
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 172
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 176
        .value_kind: by_value
...

.end_amdgpu_metadata

