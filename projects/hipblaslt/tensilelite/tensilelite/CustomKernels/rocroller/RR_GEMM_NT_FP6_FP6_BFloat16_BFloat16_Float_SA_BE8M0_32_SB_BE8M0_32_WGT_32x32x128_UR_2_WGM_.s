// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_,@function
RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 63, v1 // (op 712) 
v_and_b32 v3, 48, v1 // (op 712) 
v_sub_u32 v4, v0, v3 // (op 712) 

v_lshrrev_b32 v3, 4, v0 // (op 712) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 712) 


v_lshrrev_b32 v3, 7, v1 // (op 712) 
v_lshlrev_b32 v4, 4, v3 // (op 712) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 712) 


v_mul_lo_u32 v0, v3, 6 // (op 712) 
v_lshrrev_b32 v4, 3, v0 // (op 712) 

v_lshrrev_b32 v0, 4, v3 // (op 712) 

v_lshlrev_b32 v3, 2, v0 // (op 712) 

v_add_u32 v0, v4, v3 // (op 712) 


v_and_b32 v3, 63, v1 // (op 765) 
v_and_b32 v4, 48, v1 // (op 765) 
v_sub_u32 v5, v3, v4 // (op 765) 

v_lshrrev_b32 v4, 4, v3 // (op 765) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 765) 


v_lshrrev_b32 v4, 6, v1 // (op 765) 
v_and_b32 v5, 1, v4 // (op 765) 

v_lshlrev_b32 v4, 4, v5 // (op 765) 

v_lshl_add_u32 v5, v3, 5, v4 // (op 765) 


v_mul_lo_u32 v3, v5, 6 // (op 765) 
v_lshrrev_b32 v4, 3, v3 // (op 765) 

v_lshrrev_b32 v3, 4, v5 // (op 765) 

v_lshlrev_b32 v5, 2, v3 // (op 765) 

v_add_u32 v3, v4, v5 // (op 765) 


v_and_b32 v4, 63, v1 // (op 823) 
v_and_b32 v5, 48, v1 // (op 823) 
v_sub_u32 v6, v4, v5 // (op 823) 

v_lshrrev_b32 v5, 4, v4 // (op 823) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 823) 


v_lshrrev_b32 v5, 7, v1 // (op 823) 
v_lshlrev_b32 v6, 4, v5 // (op 823) 

v_lshl_add_u32 v5, v4, 5, v6 // (op 823) 


v_mul_lo_u32 v4, v5, 6 // (op 823) 
v_lshrrev_b32 v6, 3, v4 // (op 823) 

v_lshrrev_b32 v4, 4, v5 // (op 823) 

v_lshlrev_b32 v5, 2, v4 // (op 823) 

v_add_u32 v4, v6, v5 // (op 823) 


v_and_b32 v5, 63, v1 // (op 847) 
v_and_b32 v6, 48, v1 // (op 847) 
v_sub_u32 v7, v5, v6 // (op 847) 

v_lshrrev_b32 v6, 4, v5 // (op 847) 

v_lshl_add_u32 v5, v6, 5, v7 // (op 847) 


v_lshrrev_b32 v6, 6, v1 // (op 847) 
v_and_b32 v7, 1, v6 // (op 847) 

v_lshlrev_b32 v6, 4, v7 // (op 847) 

v_lshl_add_u32 v7, v5, 5, v6 // (op 847) 


v_mul_lo_u32 v5, v7, 6 // (op 847) 
v_lshrrev_b32 v6, 3, v5 // (op 847) 

v_lshrrev_b32 v5, 4, v7 // (op 847) 

v_lshlrev_b32 v7, 2, v5 // (op 847) 

v_add_u32 v5, v6, v7 // (op 847) 


v_mov_b32 v6, s6 // (op 2) Move value
v_mov_b32 v7, s7 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
s_mul_hi_i32 s0, s16, s9 // (op 422) 
v_and_b32 v8, -2, v1 // (op 422) 
s_add_i32 s1, s0, s16 // (op 422) 

v_sub_u32 v9, v1, v8 // (op 422) 

s_ashr_i32 s0, s1, 31 // (op 422) 
v_lshlrev_b32 v8, 4, v9 // (op 422) 

s_cmp_eq_i32 s9, 0 // (op 422) 
s_mov_b32 s6, scc // (op 422) 
v_lshrrev_b32 v9, 1, v1 // (op 422) 
s_cmp_eq_i32 s6, 0 // (op 422) 
s_cselect_b32 s7, 0, -1 // (op 422) 

s_and_b32 s6, s10, 31 // (op 422) 
s_lshl_b32 s18, 1, s6 // (op 422) 
s_add_i32 s19, s18, s7 // (op 422) 


s_and_b32 s7, s0, s19 // (op 422) 


s_add_i32 s0, s1, s7 // (op 422) 


s_ashr_i32 s1, s0, s6 // (op 422) 


s_lshl_b32 s0, s10, 24 // (op 422) 
s_ashr_i32 s6, s0, 31 // (op 422) 

s_xor_b32 s0, s1, s6 // (op 422) 

s_sub_i32 s1, s0, s6 // (op 422) 


s_mul_hi_i32 s0, s1, s11 // (op 422) 
s_add_i32 s6, s0, s1 // (op 422) 

s_ashr_i32 s0, s6, 31 // (op 422) 
s_cmp_eq_i32 s11, 0 // (op 422) 
s_mov_b32 s7, scc // (op 422) 
s_cmp_eq_i32 s7, 0 // (op 422) 
s_cselect_b32 s18, 0, -1 // (op 422) 

s_and_b32 s7, s12, 31 // (op 422) 
s_lshl_b32 s19, 1, s7 // (op 422) 
s_add_i32 s51, s19, s18 // (op 422) 


s_and_b32 s18, s0, s51 // (op 422) 


s_add_i32 s0, s6, s18 // (op 422) 


s_ashr_i32 s6, s0, s7 // (op 422) 


s_lshl_b32 s0, s12, 24 // (op 422) 
s_ashr_i32 s7, s0, 31 // (op 422) 

s_xor_b32 s0, s6, s7 // (op 422) 

s_sub_i32 s6, s0, s7 // (op 422) 


s_mul_i32 s0, s1, s14 // (op 422) 

s_sub_u32 s1, s16, s0 // (op 422) 

s_mul_i32 s0, s6, s14 // (op 422) 
s_add_u32 s0, s0, s1 // (op 422) 


s_mul_hi_i32 s1, s16, s8 // (op 422) 
s_add_i32 s6, s1, s16 // (op 422) 

s_ashr_i32 s1, s6, 31 // (op 422) 
s_cmp_eq_i32 s8, 0 // (op 422) 
s_mov_b32 s7, scc // (op 422) 
s_cmp_eq_i32 s7, 0 // (op 422) 
s_cselect_b32 s18, 0, -1 // (op 422) 

s_and_b32 s7, s15, 31 // (op 422) 
s_lshl_b32 s19, 1, s7 // (op 422) 
s_add_i32 s51, s19, s18 // (op 422) 


s_and_b32 s18, s1, s51 // (op 422) 


s_add_i32 s1, s6, s18 // (op 422) 


s_ashr_i32 s6, s1, s7 // (op 422) 


s_lshl_b32 s1, s15, 24 // (op 422) 
s_ashr_i32 s7, s1, 31 // (op 422) 

s_xor_b32 s1, s6, s7 // (op 422) 

s_sub_i32 s6, s1, s7 // (op 422) 


s_add_u32 s18, s4, 32 // (op 422) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 422) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 422) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 422) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 422) 

s_mul_i32 s1, s14, s18 // (op 422) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 422) 

s_mul_i32 s1, s6, s7 // (op 422) 

s_sub_u32 s7, s16, s1 // (op 422) 

s_mov_b32 s1, s7 // (op 422) 
s_mul_hi_i32 s18, s1, s13 // (op 422) 
s_add_i32 s19, s18, s1 // (op 422) 


s_ashr_i32 s1, s19, 31 // (op 422) 
s_cmp_eq_i32 s13, 0 // (op 422) 
s_mov_b32 s18, scc // (op 422) 
s_cmp_eq_i32 s18, 0 // (op 422) 
s_cselect_b32 s51, 0, -1 // (op 422) 

s_and_b32 s18, s49, 31 // (op 422) 
s_lshl_b32 s52, 1, s18 // (op 422) 
s_add_i32 s53, s52, s51 // (op 422) 


s_and_b32 s51, s1, s53 // (op 422) 


s_add_i32 s1, s19, s51 // (op 422) 


s_ashr_i32 s19, s1, s18 // (op 422) 


s_lshl_b32 s1, s49, 24 // (op 422) 
s_ashr_i32 s18, s1, 31 // (op 422) 

s_xor_b32 s1, s19, s18 // (op 422) 

s_sub_i32 s19, s1, s18 // (op 422) 


s_mul_i32 s1, s48, s14 // (op 422) 
s_add_u32 s52, s20, 32 // (op 422) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 422) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 422) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 422) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 5 // (op 422) 

s_sub_i32 s18, s52, s1 // (op 422) 

s_mul_i32 s51, s19, s18 // (op 422) 


s_sub_u32 s18, s7, s51 // (op 422) 


s_add_u32 s7, s18, s1 // (op 422) 


s_cmp_eq_i32 s6, 0 // (op 422) 
s_mov_b32 s1, scc // (op 422) 

s_cmp_eq_i32 s1, 0 // (op 422) 
s_cselect_b32 s6, s7, s0 // (op 422) 



v_lshl_add_u32 v10, s6, 5, v8 // (op 422) 


v_mov_b32 v13, 0 // (op 422) convert
v_mov_b32 v12, v10 // (op 422) convert

v_mul_lo_u32 v11, v9, s3 // (op 422) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 422) most significant: high of low * low
v_mul_lo_u32 v10, v9, s2 // (op 422) least significant: low of low * low
v_add_u32 v11, v11, v8 // (op 422) most significant: sum


v_add_co_u32 v8, s[0:1], v12, v10 // (op 422) least significant half
v_addc_co_u32 v9, s[0:1], v13, v11, s[0:1] // (op 422) most significant half



v_mul_lo_u32 v11, v9, 6 // (op 422) most significant: low of high * low
v_mul_hi_u32 v12, v8, 6 // (op 422) most significant: high of low * low
v_mul_lo_u32 v10, v8, 6 // (op 422) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 422) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[10:11] // (op 422) 

v_mov_b32 v10, v8 // (op 422) convert

s_mul_i32 s1, s25, 6 // (op 424) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 424) least significant
s_mul_hi_u32 s6, s24, 6 // (op 424) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 424) most significant: sum

s_mul_i32 s7, s2, s1 // (op 424) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 424) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 424) least significant
s_mul_hi_u32 s19, s2, s0 // (op 424) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 424) most significant: sum
s_add_u32 s7, s7, s19 // (op 424) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 424) 

s_mov_b32 s52, s22 // (op 424) reg expression
s_mov_b32 s53, s23 // (op 424) reg expression
s_mov_b32 s54, s0 // (op 424) reg expression
s_mov_b32 s55, 131072 // (op 424) call()

s_mul_hi_i32 s51, s16, s9 // (op 441) 
v_and_b32 v8, -2, v1 // (op 441) 
s_add_i32 s0, s51, s16 // (op 441) 

v_sub_u32 v9, v1, v8 // (op 441) 

s_ashr_i32 s1, s0, 31 // (op 441) 
v_lshlrev_b32 v8, 4, v9 // (op 441) 

s_cmp_eq_i32 s9, 0 // (op 441) 
s_mov_b32 s51, scc // (op 441) 
v_lshrrev_b32 v9, 1, v1 // (op 441) 
s_cmp_eq_i32 s51, 0 // (op 441) 
s_cselect_b32 s6, 0, -1 // (op 441) 

s_and_b32 s7, s10, 31 // (op 441) 
s_lshl_b32 s51, 1, s7 // (op 441) 
s_add_i32 s18, s51, s6 // (op 441) 


s_and_b32 s6, s1, s18 // (op 441) 


s_add_i32 s1, s0, s6 // (op 441) 


s_ashr_i32 s0, s1, s7 // (op 441) 


s_lshl_b32 s1, s10, 24 // (op 441) 
s_ashr_i32 s51, s1, 31 // (op 441) 

s_xor_b32 s1, s0, s51 // (op 441) 

s_sub_i32 s0, s1, s51 // (op 441) 


s_mul_hi_i32 s1, s0, s11 // (op 441) 
s_add_i32 s51, s1, s0 // (op 441) 

s_ashr_i32 s1, s51, 31 // (op 441) 
s_cmp_eq_i32 s11, 0 // (op 441) 
s_mov_b32 s6, scc // (op 441) 
s_cmp_eq_i32 s6, 0 // (op 441) 
s_cselect_b32 s7, 0, -1 // (op 441) 

s_and_b32 s6, s12, 31 // (op 441) 
s_lshl_b32 s18, 1, s6 // (op 441) 
s_add_i32 s19, s18, s7 // (op 441) 


s_and_b32 s7, s1, s19 // (op 441) 


s_add_i32 s1, s51, s7 // (op 441) 


s_ashr_i32 s7, s1, s6 // (op 441) 


s_lshl_b32 s1, s12, 24 // (op 441) 
s_ashr_i32 s6, s1, 31 // (op 441) 

s_xor_b32 s1, s7, s6 // (op 441) 

s_sub_i32 s7, s1, s6 // (op 441) 


s_mul_i32 s1, s0, s14 // (op 441) 

s_sub_u32 s0, s16, s1 // (op 441) 

s_mul_i32 s1, s7, s14 // (op 441) 
s_add_u32 s1, s1, s0 // (op 441) 


s_mul_hi_i32 s0, s16, s8 // (op 441) 
s_add_i32 s51, s0, s16 // (op 441) 

s_ashr_i32 s0, s51, 31 // (op 441) 
s_cmp_eq_i32 s8, 0 // (op 441) 
s_mov_b32 s6, scc // (op 441) 
s_cmp_eq_i32 s6, 0 // (op 441) 
s_cselect_b32 s7, 0, -1 // (op 441) 

s_and_b32 s6, s15, 31 // (op 441) 
s_lshl_b32 s18, 1, s6 // (op 441) 
s_add_i32 s19, s18, s7 // (op 441) 


s_and_b32 s7, s0, s19 // (op 441) 


s_add_i32 s0, s51, s7 // (op 441) 


s_ashr_i32 s7, s0, s6 // (op 441) 


s_lshl_b32 s0, s15, 24 // (op 441) 
s_ashr_i32 s6, s0, 31 // (op 441) 

s_xor_b32 s0, s7, s6 // (op 441) 

s_sub_i32 s7, s0, s6 // (op 441) 


s_add_u32 s18, s4, 32 // (op 441) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 441) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 441) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 441) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 441) 

s_mul_i32 s0, s14, s18 // (op 441) 

s_mul_i32 s6, s50, s0 // (op 441) 

s_mul_i32 s0, s7, s6 // (op 441) 

s_sub_u32 s6, s16, s0 // (op 441) 

s_mov_b32 s0, s6 // (op 441) 
s_mul_hi_i32 s51, s0, s13 // (op 441) 
s_add_i32 s18, s51, s0 // (op 441) 


s_ashr_i32 s0, s18, 31 // (op 441) 
s_cmp_eq_i32 s13, 0 // (op 441) 
s_mov_b32 s19, scc // (op 441) 
s_cmp_eq_i32 s19, 0 // (op 441) 
s_cselect_b32 s51, 0, -1 // (op 441) 

s_and_b32 s19, s49, 31 // (op 441) 
s_lshl_b32 s56, 1, s19 // (op 441) 
s_add_i32 s57, s56, s51 // (op 441) 


s_and_b32 s51, s0, s57 // (op 441) 


s_add_i32 s0, s18, s51 // (op 441) 


s_ashr_i32 s18, s0, s19 // (op 441) 


s_lshl_b32 s0, s49, 24 // (op 441) 
s_ashr_i32 s19, s0, 31 // (op 441) 

s_xor_b32 s0, s18, s19 // (op 441) 

s_sub_i32 s18, s0, s19 // (op 441) 


s_mul_i32 s0, s48, s14 // (op 441) 
s_add_u32 s56, s20, 32 // (op 441) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 441) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 441) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 441) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 441) 

s_sub_i32 s19, s56, s0 // (op 441) 

s_mul_i32 s51, s18, s19 // (op 441) 


s_sub_u32 s18, s6, s51 // (op 441) 


s_add_u32 s6, s18, s0 // (op 441) 


s_cmp_eq_i32 s7, 0 // (op 441) 
s_mov_b32 s0, scc // (op 441) 

s_cmp_eq_i32 s0, 0 // (op 441) 
s_cselect_b32 s7, s6, s1 // (op 441) 



v_lshl_add_u32 v11, s7, 5, v8 // (op 441) 


v_mov_b32 v13, 0 // (op 441) convert
v_mov_b32 v12, v11 // (op 441) convert

v_mul_lo_u32 v15, v9, s3 // (op 441) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 441) most significant: high of low * low
v_mul_lo_u32 v14, v9, s2 // (op 441) least significant: low of low * low
v_add_u32 v15, v15, v8 // (op 441) most significant: sum


v_add_co_u32 v8, s[0:1], v12, v14 // (op 441) least significant half
v_addc_co_u32 v9, s[0:1], v13, v15, s[0:1] // (op 441) most significant half



v_mul_lo_u32 v13, v9, 6 // (op 441) most significant: low of high * low
v_mul_hi_u32 v11, v8, 6 // (op 441) most significant: high of low * low
v_mul_lo_u32 v12, v8, 6 // (op 441) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 441) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[12:13] // (op 441) 

v_mov_b32 v11, v8 // (op 441) convert

s_mul_i32 s1, s25, 6 // (op 443) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 443) least significant
s_mul_hi_u32 s51, s24, 6 // (op 443) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 443) most significant: sum

s_mul_i32 s7, s2, s1 // (op 443) most significant: low of low * high
s_mul_i32 s51, s3, s0 // (op 443) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 443) least significant
s_mul_hi_u32 s18, s2, s0 // (op 443) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 443) most significant: sum
s_add_u32 s7, s7, s18 // (op 443) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 443) 

s_mov_b32 s52, s22 // (op 443) reg expression
s_mov_b32 s53, s23 // (op 443) reg expression
s_mov_b32 s54, s0 // (op 443) reg expression
s_mov_b32 s55, 131072 // (op 443) call()

s_mul_hi_i32 s51, s16, s9 // (op 486) 
v_and_b32 v8, -2, v1 // (op 486) 
s_add_i32 s0, s51, s16 // (op 486) 

v_sub_u32 v9, v1, v8 // (op 486) 

s_ashr_i32 s1, s0, 31 // (op 486) 
v_lshlrev_b32 v8, 4, v9 // (op 486) 

s_cmp_eq_i32 s9, 0 // (op 486) 
s_mov_b32 s51, scc // (op 486) 
v_lshrrev_b32 v9, 1, v1 // (op 486) 
s_cmp_eq_i32 s51, 0 // (op 486) 
s_cselect_b32 s6, 0, -1 // (op 486) 

s_and_b32 s7, s10, 31 // (op 486) 
s_lshl_b32 s51, 1, s7 // (op 486) 
s_add_i32 s18, s51, s6 // (op 486) 


s_and_b32 s6, s1, s18 // (op 486) 


s_add_i32 s1, s0, s6 // (op 486) 


s_ashr_i32 s0, s1, s7 // (op 486) 


s_lshl_b32 s1, s10, 24 // (op 486) 
s_ashr_i32 s51, s1, 31 // (op 486) 

s_xor_b32 s1, s0, s51 // (op 486) 

s_sub_i32 s0, s1, s51 // (op 486) 


s_mul_hi_i32 s1, s0, s11 // (op 486) 
s_add_i32 s51, s1, s0 // (op 486) 

s_ashr_i32 s1, s51, 31 // (op 486) 
s_cmp_eq_i32 s11, 0 // (op 486) 
s_mov_b32 s6, scc // (op 486) 
s_cmp_eq_i32 s6, 0 // (op 486) 
s_cselect_b32 s7, 0, -1 // (op 486) 

s_and_b32 s6, s12, 31 // (op 486) 
s_lshl_b32 s18, 1, s6 // (op 486) 
s_add_i32 s19, s18, s7 // (op 486) 


s_and_b32 s7, s1, s19 // (op 486) 


s_add_i32 s1, s51, s7 // (op 486) 


s_ashr_i32 s7, s1, s6 // (op 486) 


s_lshl_b32 s1, s12, 24 // (op 486) 
s_ashr_i32 s6, s1, 31 // (op 486) 

s_xor_b32 s1, s7, s6 // (op 486) 

s_sub_i32 s7, s1, s6 // (op 486) 


s_add_u32 s18, s4, 32 // (op 486) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 486) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 486) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 486) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 486) 

s_mul_i32 s1, s7, s18 // (op 486) 

s_sub_i32 s51, s0, s1 // (op 486) 


s_mul_hi_i32 s0, s16, s8 // (op 486) 
s_add_i32 s1, s0, s16 // (op 486) 

s_ashr_i32 s0, s1, 31 // (op 486) 
s_cmp_eq_i32 s8, 0 // (op 486) 
s_mov_b32 s6, scc // (op 486) 
s_cmp_eq_i32 s6, 0 // (op 486) 
s_cselect_b32 s7, 0, -1 // (op 486) 

s_and_b32 s6, s15, 31 // (op 486) 
s_lshl_b32 s56, 1, s6 // (op 486) 
s_add_i32 s57, s56, s7 // (op 486) 


s_and_b32 s7, s0, s57 // (op 486) 


s_add_i32 s0, s1, s7 // (op 486) 


s_ashr_i32 s1, s0, s6 // (op 486) 


s_lshl_b32 s0, s15, 24 // (op 486) 
s_ashr_i32 s6, s0, 31 // (op 486) 

s_xor_b32 s0, s1, s6 // (op 486) 

s_sub_i32 s1, s0, s6 // (op 486) 


s_mul_i32 s0, s14, s18 // (op 486) 

s_mul_i32 s6, s50, s0 // (op 486) 

s_mul_i32 s0, s1, s6 // (op 486) 

s_sub_u32 s6, s16, s0 // (op 486) 

s_mul_hi_i32 s0, s6, s13 // (op 486) 
s_add_i32 s7, s0, s6 // (op 486) 


s_ashr_i32 s0, s7, 31 // (op 486) 
s_cmp_eq_i32 s13, 0 // (op 486) 
s_mov_b32 s6, scc // (op 486) 
s_cmp_eq_i32 s6, 0 // (op 486) 
s_cselect_b32 s18, 0, -1 // (op 486) 

s_and_b32 s6, s49, 31 // (op 486) 
s_lshl_b32 s19, 1, s6 // (op 486) 
s_add_i32 s56, s19, s18 // (op 486) 


s_and_b32 s18, s0, s56 // (op 486) 


s_add_i32 s0, s7, s18 // (op 486) 


s_ashr_i32 s7, s0, s6 // (op 486) 


s_lshl_b32 s0, s49, 24 // (op 486) 
s_ashr_i32 s6, s0, 31 // (op 486) 

s_xor_b32 s0, s7, s6 // (op 486) 

s_sub_i32 s7, s0, s6 // (op 486) 


s_cmp_eq_i32 s1, 0 // (op 486) 
s_mov_b32 s0, scc // (op 486) 

s_cmp_eq_i32 s0, 0 // (op 486) 
s_cselect_b32 s1, s7, s51 // (op 486) 



v_lshl_add_u32 v12, s1, 5, v8 // (op 486) 


v_mov_b32 v15, 0 // (op 486) convert
v_mov_b32 v14, v12 // (op 486) convert

v_mul_lo_u32 v13, v9, s27 // (op 486) most significant: low of low * high
v_mul_hi_u32 v8, v9, s26 // (op 486) most significant: high of low * low
v_mul_lo_u32 v12, v9, s26 // (op 486) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 486) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v14 // (op 486) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 486) most significant half



v_mul_lo_u32 v9, v13, 6 // (op 486) most significant: low of high * low
v_mul_hi_u32 v14, v12, 6 // (op 486) most significant: high of low * low
v_mul_lo_u32 v8, v12, 6 // (op 486) least significant: low of low * low
v_add_u32 v9, v9, v14 // (op 486) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[8:9] // (op 486) 

v_mov_b32 v8, v12 // (op 486) convert

s_mul_i32 s1, s25, 6 // (op 488) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 488) least significant
s_mul_hi_u32 s51, s24, 6 // (op 488) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 488) most significant: sum

s_mul_i32 s7, s26, s1 // (op 488) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 488) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 488) least significant
s_mul_hi_u32 s18, s26, s0 // (op 488) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 488) most significant: sum
s_add_u32 s7, s7, s18 // (op 488) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 488) 

s_mov_b32 s56, s28 // (op 488) reg expression
s_mov_b32 s57, s29 // (op 488) reg expression
s_mov_b32 s58, s0 // (op 488) reg expression
s_mov_b32 s59, 131072 // (op 488) call()

s_mul_hi_i32 s51, s16, s9 // (op 505) 
v_and_b32 v9, -2, v1 // (op 505) 
s_add_i32 s0, s51, s16 // (op 505) 

v_sub_u32 v12, v1, v9 // (op 505) 

s_ashr_i32 s1, s0, 31 // (op 505) 
v_lshlrev_b32 v9, 4, v12 // (op 505) 

s_cmp_eq_i32 s9, 0 // (op 505) 
s_mov_b32 s51, scc // (op 505) 
v_lshrrev_b32 v12, 1, v1 // (op 505) 
s_cmp_eq_i32 s51, 0 // (op 505) 
s_cselect_b32 s6, 0, -1 // (op 505) 

s_and_b32 s7, s10, 31 // (op 505) 
s_lshl_b32 s51, 1, s7 // (op 505) 
s_add_i32 s18, s51, s6 // (op 505) 


s_and_b32 s6, s1, s18 // (op 505) 


s_add_i32 s1, s0, s6 // (op 505) 


s_ashr_i32 s0, s1, s7 // (op 505) 


s_lshl_b32 s1, s10, 24 // (op 505) 
s_ashr_i32 s51, s1, 31 // (op 505) 

s_xor_b32 s1, s0, s51 // (op 505) 

s_sub_i32 s0, s1, s51 // (op 505) 


s_mul_hi_i32 s1, s0, s11 // (op 505) 
s_add_i32 s51, s1, s0 // (op 505) 

s_ashr_i32 s1, s51, 31 // (op 505) 
s_cmp_eq_i32 s11, 0 // (op 505) 
s_mov_b32 s6, scc // (op 505) 
s_cmp_eq_i32 s6, 0 // (op 505) 
s_cselect_b32 s7, 0, -1 // (op 505) 

s_and_b32 s6, s12, 31 // (op 505) 
s_lshl_b32 s18, 1, s6 // (op 505) 
s_add_i32 s19, s18, s7 // (op 505) 


s_and_b32 s7, s1, s19 // (op 505) 


s_add_i32 s1, s51, s7 // (op 505) 


s_ashr_i32 s7, s1, s6 // (op 505) 


s_lshl_b32 s1, s12, 24 // (op 505) 
s_ashr_i32 s6, s1, 31 // (op 505) 

s_xor_b32 s1, s7, s6 // (op 505) 

s_sub_i32 s7, s1, s6 // (op 505) 


s_add_u32 s18, s4, 32 // (op 505) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 505) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 505) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 505) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 505) 

s_mul_i32 s1, s7, s18 // (op 505) 

s_sub_i32 s51, s0, s1 // (op 505) 


s_mul_hi_i32 s0, s16, s8 // (op 505) 
s_add_i32 s1, s0, s16 // (op 505) 

s_ashr_i32 s0, s1, 31 // (op 505) 
s_cmp_eq_i32 s8, 0 // (op 505) 
s_mov_b32 s6, scc // (op 505) 
s_cmp_eq_i32 s6, 0 // (op 505) 
s_cselect_b32 s7, 0, -1 // (op 505) 

s_and_b32 s6, s15, 31 // (op 505) 
s_lshl_b32 s60, 1, s6 // (op 505) 
s_add_i32 s61, s60, s7 // (op 505) 


s_and_b32 s7, s0, s61 // (op 505) 


s_add_i32 s0, s1, s7 // (op 505) 


s_ashr_i32 s1, s0, s6 // (op 505) 


s_lshl_b32 s0, s15, 24 // (op 505) 
s_ashr_i32 s6, s0, 31 // (op 505) 

s_xor_b32 s0, s1, s6 // (op 505) 

s_sub_i32 s1, s0, s6 // (op 505) 


s_mul_i32 s0, s14, s18 // (op 505) 

s_mul_i32 s6, s50, s0 // (op 505) 

s_mul_i32 s0, s1, s6 // (op 505) 

s_sub_u32 s6, s16, s0 // (op 505) 

s_mul_hi_i32 s0, s6, s13 // (op 505) 
s_add_i32 s7, s0, s6 // (op 505) 


s_ashr_i32 s0, s7, 31 // (op 505) 
s_cmp_eq_i32 s13, 0 // (op 505) 
s_mov_b32 s6, scc // (op 505) 
s_cmp_eq_i32 s6, 0 // (op 505) 
s_cselect_b32 s18, 0, -1 // (op 505) 

s_and_b32 s6, s49, 31 // (op 505) 
s_lshl_b32 s19, 1, s6 // (op 505) 
s_add_i32 s60, s19, s18 // (op 505) 


s_and_b32 s18, s0, s60 // (op 505) 


s_add_i32 s0, s7, s18 // (op 505) 


s_ashr_i32 s7, s0, s6 // (op 505) 


s_lshl_b32 s0, s49, 24 // (op 505) 
s_ashr_i32 s6, s0, 31 // (op 505) 

s_xor_b32 s0, s7, s6 // (op 505) 

s_sub_i32 s7, s0, s6 // (op 505) 


s_cmp_eq_i32 s1, 0 // (op 505) 
s_mov_b32 s0, scc // (op 505) 

s_cmp_eq_i32 s0, 0 // (op 505) 
s_cselect_b32 s1, s7, s51 // (op 505) 



v_lshl_add_u32 v13, s1, 5, v9 // (op 505) 


v_mov_b32 v15, 0 // (op 505) convert
v_mov_b32 v14, v13 // (op 505) convert

v_mul_lo_u32 v17, v12, s27 // (op 505) most significant: low of low * high
v_mul_hi_u32 v9, v12, s26 // (op 505) most significant: high of low * low
v_mul_lo_u32 v16, v12, s26 // (op 505) least significant: low of low * low
v_add_u32 v17, v17, v9 // (op 505) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 505) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 505) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 505) most significant: low of high * low
v_mul_hi_u32 v9, v16, 6 // (op 505) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 505) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 505) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 505) 

v_mov_b32 v9, v14 // (op 505) convert

s_mul_i32 s1, s25, 6 // (op 507) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 507) least significant
s_mul_hi_u32 s51, s24, 6 // (op 507) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 507) most significant: sum

s_mul_i32 s7, s26, s1 // (op 507) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 507) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 507) least significant
s_mul_hi_u32 s18, s26, s0 // (op 507) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 507) most significant: sum
s_add_u32 s7, s7, s18 // (op 507) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 507) 

s_mov_b32 s56, s28 // (op 507) reg expression
s_mov_b32 s57, s29 // (op 507) reg expression
s_mov_b32 s58, s0 // (op 507) reg expression
s_mov_b32 s59, 131072 // (op 507) call()

s_mul_hi_i32 s51, s16, s9 // (op 550) 
v_lshrrev_b32 v12, 7, v1 // (op 550) 
s_add_i32 s0, s51, s16 // (op 550) 

v_and_b32 v13, 15, v1 // (op 550) 
s_ashr_i32 s1, s0, 31 // (op 550) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 550) 


s_cmp_eq_i32 s9, 0 // (op 550) 
s_mov_b32 s51, scc // (op 550) 
v_and_b32 v12, 63, v1 // (op 550) 
s_cmp_eq_i32 s51, 0 // (op 550) 
s_cselect_b32 s6, 0, -1 // (op 550) 

v_lshrrev_b32 v13, 4, v12 // (op 550) 

s_and_b32 s7, s10, 31 // (op 550) 
s_lshl_b32 s51, 1, s7 // (op 550) 
s_add_i32 s18, s51, s6 // (op 550) 


s_and_b32 s6, s1, s18 // (op 550) 


s_add_i32 s1, s0, s6 // (op 550) 


s_ashr_i32 s0, s1, s7 // (op 550) 


s_lshl_b32 s1, s10, 24 // (op 550) 
s_ashr_i32 s51, s1, 31 // (op 550) 

s_xor_b32 s1, s0, s51 // (op 550) 

s_sub_i32 s0, s1, s51 // (op 550) 


s_mul_hi_i32 s1, s0, s11 // (op 550) 
s_add_i32 s51, s1, s0 // (op 550) 

s_ashr_i32 s1, s51, 31 // (op 550) 
s_cmp_eq_i32 s11, 0 // (op 550) 
s_mov_b32 s6, scc // (op 550) 
s_cmp_eq_i32 s6, 0 // (op 550) 
s_cselect_b32 s7, 0, -1 // (op 550) 

s_and_b32 s6, s12, 31 // (op 550) 
s_lshl_b32 s18, 1, s6 // (op 550) 
s_add_i32 s19, s18, s7 // (op 550) 


s_and_b32 s7, s1, s19 // (op 550) 


s_add_i32 s1, s51, s7 // (op 550) 


s_ashr_i32 s7, s1, s6 // (op 550) 


s_lshl_b32 s1, s12, 24 // (op 550) 
s_ashr_i32 s6, s1, 31 // (op 550) 

s_xor_b32 s1, s7, s6 // (op 550) 

s_sub_i32 s7, s1, s6 // (op 550) 


s_mul_i32 s1, s0, s14 // (op 550) 

s_sub_u32 s0, s16, s1 // (op 550) 

s_mul_i32 s1, s7, s14 // (op 550) 
s_add_u32 s1, s1, s0 // (op 550) 


s_mul_hi_i32 s0, s16, s8 // (op 550) 
s_add_i32 s51, s0, s16 // (op 550) 

s_ashr_i32 s0, s51, 31 // (op 550) 
s_cmp_eq_i32 s8, 0 // (op 550) 
s_mov_b32 s6, scc // (op 550) 
s_cmp_eq_i32 s6, 0 // (op 550) 
s_cselect_b32 s7, 0, -1 // (op 550) 

s_and_b32 s6, s15, 31 // (op 550) 
s_lshl_b32 s18, 1, s6 // (op 550) 
s_add_i32 s19, s18, s7 // (op 550) 


s_and_b32 s7, s0, s19 // (op 550) 


s_add_i32 s0, s51, s7 // (op 550) 


s_ashr_i32 s7, s0, s6 // (op 550) 


s_lshl_b32 s0, s15, 24 // (op 550) 
s_ashr_i32 s6, s0, 31 // (op 550) 

s_xor_b32 s0, s7, s6 // (op 550) 

s_sub_i32 s7, s0, s6 // (op 550) 


s_add_u32 s18, s4, 32 // (op 550) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 550) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 550) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 550) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 550) 

s_mul_i32 s0, s14, s18 // (op 550) 

s_mul_i32 s6, s50, s0 // (op 550) 

s_mul_i32 s0, s7, s6 // (op 550) 

s_sub_u32 s6, s16, s0 // (op 550) 

s_mov_b32 s0, s6 // (op 550) 
s_mul_hi_i32 s51, s0, s13 // (op 550) 
s_add_i32 s18, s51, s0 // (op 550) 


s_ashr_i32 s0, s18, 31 // (op 550) 
s_cmp_eq_i32 s13, 0 // (op 550) 
s_mov_b32 s19, scc // (op 550) 
s_cmp_eq_i32 s19, 0 // (op 550) 
s_cselect_b32 s51, 0, -1 // (op 550) 

s_and_b32 s19, s49, 31 // (op 550) 
s_lshl_b32 s60, 1, s19 // (op 550) 
s_add_i32 s61, s60, s51 // (op 550) 


s_and_b32 s51, s0, s61 // (op 550) 


s_add_i32 s0, s18, s51 // (op 550) 


s_ashr_i32 s18, s0, s19 // (op 550) 


s_lshl_b32 s0, s49, 24 // (op 550) 
s_ashr_i32 s19, s0, 31 // (op 550) 

s_xor_b32 s0, s18, s19 // (op 550) 

s_sub_i32 s18, s0, s19 // (op 550) 


s_mul_i32 s0, s48, s14 // (op 550) 
s_add_u32 s60, s20, 32 // (op 550) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 550) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 550) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 550) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 5 // (op 550) 

s_sub_i32 s19, s60, s0 // (op 550) 

s_mul_i32 s51, s18, s19 // (op 550) 


s_sub_u32 s18, s6, s51 // (op 550) 


s_add_u32 s6, s18, s0 // (op 550) 


s_cmp_eq_i32 s7, 0 // (op 550) 
s_mov_b32 s0, scc // (op 550) 

s_cmp_eq_i32 s0, 0 // (op 550) 
s_cselect_b32 s7, s6, s1 // (op 550) 



v_lshl_add_u32 v12, s7, 5, v14 // (op 550) 


v_mul_lo_u32 v14, v13, s30 // (op 550) 

v_add_u32 v13, v12, v14 // (op 550) 


s_mul_i32 s1, s30, s35 // (op 552) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 552) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 552) least significant
s_mul_hi_u32 s6, s30, s34 // (op 552) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 552) most significant: sum
s_add_u32 s1, s1, s6 // (op 552) most significant: sum


s_mov_b32 s60, s32 // (op 552) reg expression
s_mov_b32 s61, s33 // (op 552) reg expression
s_mov_b32 s62, s0 // (op 552) reg expression
s_mov_b32 s63, 131072 // (op 552) call()

s_mul_hi_i32 s51, s16, s9 // (op 569) 
v_lshrrev_b32 v12, 7, v1 // (op 569) 
s_add_i32 s0, s51, s16 // (op 569) 

v_and_b32 v14, 15, v1 // (op 569) 
s_ashr_i32 s1, s0, 31 // (op 569) 
v_lshl_add_u32 v15, v12, 4, v14 // (op 569) 


s_cmp_eq_i32 s9, 0 // (op 569) 
s_mov_b32 s51, scc // (op 569) 
v_and_b32 v12, 63, v1 // (op 569) 
s_cmp_eq_i32 s51, 0 // (op 569) 
s_cselect_b32 s6, 0, -1 // (op 569) 

v_lshrrev_b32 v14, 4, v12 // (op 569) 

s_and_b32 s7, s10, 31 // (op 569) 
s_lshl_b32 s51, 1, s7 // (op 569) 
s_add_i32 s18, s51, s6 // (op 569) 


s_and_b32 s6, s1, s18 // (op 569) 


s_add_i32 s1, s0, s6 // (op 569) 


s_ashr_i32 s0, s1, s7 // (op 569) 


s_lshl_b32 s1, s10, 24 // (op 569) 
s_ashr_i32 s51, s1, 31 // (op 569) 

s_xor_b32 s1, s0, s51 // (op 569) 

s_sub_i32 s0, s1, s51 // (op 569) 


s_mul_hi_i32 s1, s0, s11 // (op 569) 
s_add_i32 s51, s1, s0 // (op 569) 

s_ashr_i32 s1, s51, 31 // (op 569) 
s_cmp_eq_i32 s11, 0 // (op 569) 
s_mov_b32 s6, scc // (op 569) 
s_cmp_eq_i32 s6, 0 // (op 569) 
s_cselect_b32 s7, 0, -1 // (op 569) 

s_and_b32 s6, s12, 31 // (op 569) 
s_lshl_b32 s18, 1, s6 // (op 569) 
s_add_i32 s19, s18, s7 // (op 569) 


s_and_b32 s7, s1, s19 // (op 569) 


s_add_i32 s1, s51, s7 // (op 569) 


s_ashr_i32 s7, s1, s6 // (op 569) 


s_lshl_b32 s1, s12, 24 // (op 569) 
s_ashr_i32 s6, s1, 31 // (op 569) 

s_xor_b32 s1, s7, s6 // (op 569) 

s_sub_i32 s7, s1, s6 // (op 569) 


s_mul_i32 s1, s0, s14 // (op 569) 

s_sub_u32 s0, s16, s1 // (op 569) 

s_mul_i32 s1, s7, s14 // (op 569) 
s_add_u32 s1, s1, s0 // (op 569) 


s_mul_hi_i32 s0, s16, s8 // (op 569) 
s_add_i32 s51, s0, s16 // (op 569) 

s_ashr_i32 s0, s51, 31 // (op 569) 
s_cmp_eq_i32 s8, 0 // (op 569) 
s_mov_b32 s6, scc // (op 569) 
s_cmp_eq_i32 s6, 0 // (op 569) 
s_cselect_b32 s7, 0, -1 // (op 569) 

s_and_b32 s6, s15, 31 // (op 569) 
s_lshl_b32 s18, 1, s6 // (op 569) 
s_add_i32 s19, s18, s7 // (op 569) 


s_and_b32 s7, s0, s19 // (op 569) 


s_add_i32 s0, s51, s7 // (op 569) 


s_ashr_i32 s7, s0, s6 // (op 569) 


s_lshl_b32 s0, s15, 24 // (op 569) 
s_ashr_i32 s6, s0, 31 // (op 569) 

s_xor_b32 s0, s7, s6 // (op 569) 

s_sub_i32 s7, s0, s6 // (op 569) 


s_add_u32 s18, s4, 32 // (op 569) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 569) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 569) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 569) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 569) 

s_mul_i32 s0, s14, s18 // (op 569) 

s_mul_i32 s6, s50, s0 // (op 569) 

s_mul_i32 s0, s7, s6 // (op 569) 

s_sub_u32 s6, s16, s0 // (op 569) 

s_mov_b32 s0, s6 // (op 569) 
s_mul_hi_i32 s51, s0, s13 // (op 569) 
s_add_i32 s18, s51, s0 // (op 569) 


s_ashr_i32 s0, s18, 31 // (op 569) 
s_cmp_eq_i32 s13, 0 // (op 569) 
s_mov_b32 s19, scc // (op 569) 
s_cmp_eq_i32 s19, 0 // (op 569) 
s_cselect_b32 s51, 0, -1 // (op 569) 

s_and_b32 s19, s49, 31 // (op 569) 
s_lshl_b32 s64, 1, s19 // (op 569) 
s_add_i32 s65, s64, s51 // (op 569) 


s_and_b32 s51, s0, s65 // (op 569) 


s_add_i32 s0, s18, s51 // (op 569) 


s_ashr_i32 s18, s0, s19 // (op 569) 


s_lshl_b32 s0, s49, 24 // (op 569) 
s_ashr_i32 s19, s0, 31 // (op 569) 

s_xor_b32 s0, s18, s19 // (op 569) 

s_sub_i32 s18, s0, s19 // (op 569) 


s_mul_i32 s0, s48, s14 // (op 569) 
s_add_u32 s64, s20, 32 // (op 569) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 569) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 569) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 569) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 5 // (op 569) 

s_sub_i32 s19, s64, s0 // (op 569) 

s_mul_i32 s51, s18, s19 // (op 569) 


s_sub_u32 s18, s6, s51 // (op 569) 


s_add_u32 s6, s18, s0 // (op 569) 


s_cmp_eq_i32 s7, 0 // (op 569) 
s_mov_b32 s0, scc // (op 569) 

s_cmp_eq_i32 s0, 0 // (op 569) 
s_cselect_b32 s7, s6, s1 // (op 569) 



v_lshl_add_u32 v12, s7, 5, v15 // (op 569) 


v_mul_lo_u32 v15, v14, s30 // (op 569) 

v_add_u32 v14, v12, v15 // (op 569) 


s_mul_i32 s1, s30, s35 // (op 571) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 571) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 571) least significant
s_mul_hi_u32 s6, s30, s34 // (op 571) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 571) most significant: sum
s_add_u32 s1, s1, s6 // (op 571) most significant: sum


s_mov_b32 s60, s32 // (op 571) reg expression
s_mov_b32 s61, s33 // (op 571) reg expression
s_mov_b32 s62, s0 // (op 571) reg expression
s_mov_b32 s63, 131072 // (op 571) call()

s_mul_hi_i32 s51, s16, s9 // (op 614) 
v_lshrrev_b32 v12, 6, v1 // (op 614) 
s_add_i32 s0, s51, s16 // (op 614) 

v_and_b32 v15, 1, v12 // (op 614) 

s_ashr_i32 s1, s0, 31 // (op 614) 
v_and_b32 v12, 15, v1 // (op 614) 
s_cmp_eq_i32 s9, 0 // (op 614) 
s_mov_b32 s51, scc // (op 614) 
v_lshl_add_u32 v16, v15, 4, v12 // (op 614) 


s_cmp_eq_i32 s51, 0 // (op 614) 
s_cselect_b32 s6, 0, -1 // (op 614) 

v_and_b32 v12, 63, v1 // (op 614) 
s_and_b32 s7, s10, 31 // (op 614) 
v_lshrrev_b32 v15, 4, v12 // (op 614) 

s_lshl_b32 s51, 1, s7 // (op 614) 
s_add_i32 s18, s51, s6 // (op 614) 


s_and_b32 s6, s1, s18 // (op 614) 


s_add_i32 s1, s0, s6 // (op 614) 


s_ashr_i32 s0, s1, s7 // (op 614) 


s_lshl_b32 s1, s10, 24 // (op 614) 
s_ashr_i32 s51, s1, 31 // (op 614) 

s_xor_b32 s1, s0, s51 // (op 614) 

s_sub_i32 s0, s1, s51 // (op 614) 


s_mul_hi_i32 s1, s0, s11 // (op 614) 
s_add_i32 s51, s1, s0 // (op 614) 

s_ashr_i32 s1, s51, 31 // (op 614) 
s_cmp_eq_i32 s11, 0 // (op 614) 
s_mov_b32 s6, scc // (op 614) 
s_cmp_eq_i32 s6, 0 // (op 614) 
s_cselect_b32 s7, 0, -1 // (op 614) 

s_and_b32 s6, s12, 31 // (op 614) 
s_lshl_b32 s18, 1, s6 // (op 614) 
s_add_i32 s19, s18, s7 // (op 614) 


s_and_b32 s7, s1, s19 // (op 614) 


s_add_i32 s1, s51, s7 // (op 614) 


s_ashr_i32 s7, s1, s6 // (op 614) 


s_lshl_b32 s1, s12, 24 // (op 614) 
s_ashr_i32 s6, s1, 31 // (op 614) 

s_xor_b32 s1, s7, s6 // (op 614) 

s_sub_i32 s7, s1, s6 // (op 614) 


s_add_u32 s18, s4, 32 // (op 614) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 614) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 614) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 614) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 614) 

s_mul_i32 s1, s7, s18 // (op 614) 

s_sub_i32 s51, s0, s1 // (op 614) 


s_mul_hi_i32 s0, s16, s8 // (op 614) 
s_add_i32 s1, s0, s16 // (op 614) 

s_ashr_i32 s0, s1, 31 // (op 614) 
s_cmp_eq_i32 s8, 0 // (op 614) 
s_mov_b32 s6, scc // (op 614) 
s_cmp_eq_i32 s6, 0 // (op 614) 
s_cselect_b32 s7, 0, -1 // (op 614) 

s_and_b32 s6, s15, 31 // (op 614) 
s_lshl_b32 s64, 1, s6 // (op 614) 
s_add_i32 s65, s64, s7 // (op 614) 


s_and_b32 s7, s0, s65 // (op 614) 


s_add_i32 s0, s1, s7 // (op 614) 


s_ashr_i32 s1, s0, s6 // (op 614) 


s_lshl_b32 s0, s15, 24 // (op 614) 
s_ashr_i32 s6, s0, 31 // (op 614) 

s_xor_b32 s0, s1, s6 // (op 614) 

s_sub_i32 s1, s0, s6 // (op 614) 


s_mul_i32 s0, s14, s18 // (op 614) 

s_mul_i32 s6, s50, s0 // (op 614) 

s_mul_i32 s0, s1, s6 // (op 614) 

s_sub_u32 s6, s16, s0 // (op 614) 

s_mul_hi_i32 s0, s6, s13 // (op 614) 
s_add_i32 s7, s0, s6 // (op 614) 


s_ashr_i32 s0, s7, 31 // (op 614) 
s_cmp_eq_i32 s13, 0 // (op 614) 
s_mov_b32 s6, scc // (op 614) 
s_cmp_eq_i32 s6, 0 // (op 614) 
s_cselect_b32 s18, 0, -1 // (op 614) 

s_and_b32 s6, s49, 31 // (op 614) 
s_lshl_b32 s19, 1, s6 // (op 614) 
s_add_i32 s64, s19, s18 // (op 614) 


s_and_b32 s18, s0, s64 // (op 614) 


s_add_i32 s0, s7, s18 // (op 614) 


s_ashr_i32 s7, s0, s6 // (op 614) 


s_lshl_b32 s0, s49, 24 // (op 614) 
s_ashr_i32 s6, s0, 31 // (op 614) 

s_xor_b32 s0, s7, s6 // (op 614) 

s_sub_i32 s7, s0, s6 // (op 614) 


s_cmp_eq_i32 s1, 0 // (op 614) 
s_mov_b32 s0, scc // (op 614) 

s_cmp_eq_i32 s0, 0 // (op 614) 
s_cselect_b32 s1, s7, s51 // (op 614) 



v_lshl_add_u32 v12, s1, 5, v16 // (op 614) 


v_mul_lo_u32 v16, v15, s36 // (op 614) 
v_add_u32 v16, v16, v12 // (op 614) 


s_mul_i32 s1, s36, s35 // (op 616) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 616) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 616) least significant
s_mul_hi_u32 s6, s36, s34 // (op 616) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 616) most significant: sum
s_add_u32 s1, s1, s6 // (op 616) most significant: sum


s_mov_b32 s64, s38 // (op 616) reg expression
s_mov_b32 s65, s39 // (op 616) reg expression
s_mov_b32 s66, s0 // (op 616) reg expression
s_mov_b32 s67, 131072 // (op 616) call()

s_mul_hi_i32 s51, s16, s9 // (op 633) 
v_lshrrev_b32 v12, 6, v1 // (op 633) 
s_add_i32 s0, s51, s16 // (op 633) 

v_and_b32 v15, 1, v12 // (op 633) 

s_ashr_i32 s1, s0, 31 // (op 633) 
v_and_b32 v12, 15, v1 // (op 633) 
s_cmp_eq_i32 s9, 0 // (op 633) 
s_mov_b32 s51, scc // (op 633) 
v_lshl_add_u32 v17, v15, 4, v12 // (op 633) 


s_cmp_eq_i32 s51, 0 // (op 633) 
s_cselect_b32 s6, 0, -1 // (op 633) 

v_and_b32 v12, 63, v1 // (op 633) 
s_and_b32 s7, s10, 31 // (op 633) 
v_lshrrev_b32 v15, 4, v12 // (op 633) 

s_lshl_b32 s51, 1, s7 // (op 633) 
s_add_i32 s18, s51, s6 // (op 633) 


s_and_b32 s6, s1, s18 // (op 633) 


s_add_i32 s1, s0, s6 // (op 633) 


s_ashr_i32 s0, s1, s7 // (op 633) 


s_lshl_b32 s1, s10, 24 // (op 633) 
s_ashr_i32 s51, s1, 31 // (op 633) 

s_xor_b32 s1, s0, s51 // (op 633) 

s_sub_i32 s0, s1, s51 // (op 633) 


s_mul_hi_i32 s1, s0, s11 // (op 633) 
s_add_i32 s51, s1, s0 // (op 633) 

s_ashr_i32 s1, s51, 31 // (op 633) 
s_cmp_eq_i32 s11, 0 // (op 633) 
s_mov_b32 s6, scc // (op 633) 
s_cmp_eq_i32 s6, 0 // (op 633) 
s_cselect_b32 s7, 0, -1 // (op 633) 

s_and_b32 s6, s12, 31 // (op 633) 
s_lshl_b32 s18, 1, s6 // (op 633) 
s_add_i32 s19, s18, s7 // (op 633) 


s_and_b32 s7, s1, s19 // (op 633) 


s_add_i32 s1, s51, s7 // (op 633) 


s_ashr_i32 s7, s1, s6 // (op 633) 


s_lshl_b32 s1, s12, 24 // (op 633) 
s_ashr_i32 s6, s1, 31 // (op 633) 

s_xor_b32 s1, s7, s6 // (op 633) 

s_sub_i32 s7, s1, s6 // (op 633) 


s_add_u32 s18, s4, 32 // (op 633) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 633) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 633) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 633) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 633) 

s_mul_i32 s1, s7, s18 // (op 633) 

s_sub_i32 s51, s0, s1 // (op 633) 


s_mul_hi_i32 s0, s16, s8 // (op 633) 
s_add_i32 s1, s0, s16 // (op 633) 

s_ashr_i32 s0, s1, 31 // (op 633) 
s_cmp_eq_i32 s8, 0 // (op 633) 
s_mov_b32 s6, scc // (op 633) 
s_cmp_eq_i32 s6, 0 // (op 633) 
s_cselect_b32 s7, 0, -1 // (op 633) 

s_and_b32 s6, s15, 31 // (op 633) 
s_lshl_b32 s68, 1, s6 // (op 633) 
s_add_i32 s69, s68, s7 // (op 633) 


s_and_b32 s7, s0, s69 // (op 633) 


s_add_i32 s0, s1, s7 // (op 633) 


s_ashr_i32 s1, s0, s6 // (op 633) 


s_lshl_b32 s0, s15, 24 // (op 633) 
s_ashr_i32 s6, s0, 31 // (op 633) 

s_xor_b32 s0, s1, s6 // (op 633) 

s_sub_i32 s1, s0, s6 // (op 633) 


s_mul_i32 s0, s14, s18 // (op 633) 

s_mul_i32 s6, s50, s0 // (op 633) 

s_mul_i32 s0, s1, s6 // (op 633) 

s_sub_u32 s6, s16, s0 // (op 633) 

s_mul_hi_i32 s0, s6, s13 // (op 633) 
s_add_i32 s7, s0, s6 // (op 633) 


s_ashr_i32 s0, s7, 31 // (op 633) 
s_cmp_eq_i32 s13, 0 // (op 633) 
s_mov_b32 s6, scc // (op 633) 
s_cmp_eq_i32 s6, 0 // (op 633) 
s_cselect_b32 s18, 0, -1 // (op 633) 

s_and_b32 s6, s49, 31 // (op 633) 
s_lshl_b32 s19, 1, s6 // (op 633) 
s_add_i32 s68, s19, s18 // (op 633) 


s_and_b32 s18, s0, s68 // (op 633) 


s_add_i32 s0, s7, s18 // (op 633) 


s_ashr_i32 s7, s0, s6 // (op 633) 


s_lshl_b32 s0, s49, 24 // (op 633) 
s_ashr_i32 s6, s0, 31 // (op 633) 

s_xor_b32 s0, s7, s6 // (op 633) 

s_sub_i32 s7, s0, s6 // (op 633) 


s_cmp_eq_i32 s1, 0 // (op 633) 
s_mov_b32 s0, scc // (op 633) 

s_cmp_eq_i32 s0, 0 // (op 633) 
s_cselect_b32 s1, s7, s51 // (op 633) 



v_lshl_add_u32 v12, s1, 5, v17 // (op 633) 


v_mul_lo_u32 v17, v15, s36 // (op 633) 
v_add_u32 v17, v17, v12 // (op 633) 


s_mul_i32 s1, s36, s35 // (op 635) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 635) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 635) least significant
s_mul_hi_u32 s6, s36, s34 // (op 635) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 635) most significant: sum
s_add_u32 s1, s1, s6 // (op 635) most significant: sum


s_mov_b32 s64, s38 // (op 635) reg expression
s_mov_b32 s65, s39 // (op 635) reg expression
s_mov_b32 s66, s0 // (op 635) reg expression
s_mov_b32 s67, 131072 // (op 635) call()


v_mov_b64 v[18:19], s[30:31] // (op 312) convert

v_mov_b32 v12, v13 // (op 312) reg expression
v_mov_b32 v15, v12 // (op 312) 
buffer_load_ubyte v20, v15, s[60:63], 0 offen offset: 0 // (op 312) Load value




v_mov_b64 v[18:19], s[36:37] // (op 317) convert

v_mov_b32 v12, v16 // (op 317) reg expression
v_mov_b32 v15, v12 // (op 317) 
buffer_load_ubyte v21, v15, s[64:67], 0 offen offset: 0 // (op 317) Load value





v_mov_b32 v12, v8 // (op 321) reg expression

buffer_load_dwordx3 v[22:24], v12, s[56:59], 0 offen // (op 321) Load value



v_mov_b32 v12, v10 // (op 324) reg expression

buffer_load_dwordx3 v[26:28], v12, s[52:55], 0 offen // (op 324) Load value


v_mov_b64 v[18:19], s[30:31] // (op 328) convert

s_lshl_b32 s51, s30, 2 // (op 328) 
v_add_u32 v12, v13, s51 // (op 328) 

v_mov_b32 v15, v12 // (op 328) 
buffer_load_ubyte v25, v15, s[60:63], 0 offen offset: 0 // (op 328) Load value





v_mov_b64 v[12:13], s[36:37] // (op 333) convert

s_lshl_b32 s51, s36, 2 // (op 333) 
v_add_u32 v15, v16, s51 // (op 333) 

v_mov_b32 v18, v15 // (op 333) 
buffer_load_ubyte v19, v18, s[64:67], 0 offen offset: 0 // (op 333) Load value






s_lshl_b64 s[0:1], s[26:27], 7 // (op 337) 
s_mul_i32 s7, s1, 6 // (op 337) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 337) least significant
s_mul_hi_u32 s51, s0, 6 // (op 337) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 337) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 337) 

v_add_u32 v18, v8, s0 // (op 337) 


buffer_load_dwordx3 v[30:32], v18, s[56:59], 0 offen // (op 337) Load value




s_lshl_b64 s[0:1], s[2:3], 7 // (op 340) 
s_mul_i32 s7, s1, 6 // (op 340) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 340) least significant
s_mul_hi_u32 s51, s0, 6 // (op 340) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 340) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 340) 

v_add_u32 v8, v10, s0 // (op 340) 


buffer_load_dwordx3 v[34:36], v8, s[52:55], 0 offen // (op 340) Load value


v_and_b32 v8, -2, v1 // (op 753) 
v_sub_u32 v10, v1, v8 // (op 753) 

v_lshlrev_b32 v8, 4, v10 // (op 753) 

v_lshrrev_b32 v10, 1, v1 // (op 753) 
v_lshl_add_u32 v18, v10, 5, v8 // (op 753) 


v_mul_lo_u32 v8, v18, 6 // (op 753) 
v_lshrrev_b32 v10, 3, v8 // (op 753) 

v_lshrrev_b32 v8, 4, v18 // (op 753) 

v_lshlrev_b32 v18, 2, v8 // (op 753) 

v_add_u32 v8, v10, v18 // (op 753) 





s_waitcnt vmcnt(5)
ds_write_b96 v8, v[22:24] // (op 342) Store local data 

v_and_b32 v8, -2, v1 // (op 700) 
v_sub_u32 v10, v1, v8 // (op 700) 

v_lshlrev_b32 v8, 4, v10 // (op 700) 

v_lshrrev_b32 v10, 1, v1 // (op 700) 
v_lshl_add_u32 v18, v10, 5, v8 // (op 700) 


v_mul_lo_u32 v8, v18, 6 // (op 700) 
v_lshrrev_b32 v10, 3, v8 // (op 700) 

v_lshrrev_b32 v8, 4, v18 // (op 700) 

v_lshlrev_b32 v18, 2, v8 // (op 700) 

v_add_u32 v8, v10, v18 // (op 700) 





s_waitcnt vmcnt(4)
ds_write_b96 v8, v[26:28] offset:4096 // (op 343) Store local data 

v_and_b32 v8, -2, v1 // (op 671) 
v_sub_u32 v10, v1, v8 // (op 671) 

v_lshlrev_b32 v8, 4, v10 // (op 671) 

v_lshrrev_b32 v10, 1, v1 // (op 671) 
v_lshl_add_u32 v18, v10, 5, v8 // (op 671) 


v_mul_lo_u32 v8, v18, 6 // (op 671) 
v_lshrrev_b32 v10, 3, v8 // (op 671) 

v_lshrrev_b32 v8, 4, v18 // (op 671) 

v_lshlrev_b32 v18, 2, v8 // (op 671) 

v_add_u32 v8, v10, v18 // (op 671) 


v_and_b32 v10, -2, v1 // (op 724) 
v_sub_u32 v18, v1, v10 // (op 724) 

v_lshlrev_b32 v10, 4, v18 // (op 724) 

v_lshrrev_b32 v18, 1, v1 // (op 724) 
v_lshl_add_u32 v29, v18, 5, v10 // (op 724) 


v_mul_lo_u32 v10, v29, 6 // (op 724) 
v_lshrrev_b32 v18, 3, v10 // (op 724) 

v_lshrrev_b32 v10, 4, v29 // (op 724) 

v_lshlrev_b32 v29, 2, v10 // (op 724) 

v_add_u32 v10, v18, v29 // (op 724) 


v_and_b32 v18, -2, v1 // (op 811) 
v_sub_u32 v29, v1, v18 // (op 811) 

v_lshlrev_b32 v18, 4, v29 // (op 811) 

v_lshrrev_b32 v29, 1, v1 // (op 811) 
v_lshl_add_u32 v33, v29, 5, v18 // (op 811) 


v_mul_lo_u32 v18, v33, 6 // (op 811) 
v_lshrrev_b32 v29, 3, v18 // (op 811) 

v_lshrrev_b32 v18, 4, v33 // (op 811) 

v_lshlrev_b32 v33, 2, v18 // (op 811) 

v_add_u32 v18, v29, v33 // (op 811) 


v_and_b32 v29, -2, v1 // (op 835) 
v_sub_u32 v33, v1, v29 // (op 835) 

v_lshlrev_b32 v29, 4, v33 // (op 835) 

v_lshrrev_b32 v33, 1, v1 // (op 835) 
v_lshl_add_u32 v12, v33, 5, v29 // (op 835) 


v_mul_lo_u32 v13, v12, 6 // (op 835) 
v_lshrrev_b32 v29, 3, v13 // (op 835) 

v_lshrrev_b32 v13, 4, v12 // (op 835) 

v_lshlrev_b32 v12, 2, v13 // (op 835) 

v_add_u32 v13, v29, v12 // (op 835) 


s_mov_b32 s51, 0 // (op 53) call()
s_ashr_i64 s[0:1], s[24:25], 7 // (op 52) 
s_mov_b32 s6, s0 // (op 52) convert

s_and_b32 s7, s6, 4294967294 // (op 52) 

s_cmp_lt_u32 s51, s7 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_52 // (op 52) Condition: Top (jump to RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_52 if false)
RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_52:
s_add_u32 s51, s51, 2 // (op 54) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 876) 



ds_read_b96_tr_b6 v[38:40], v0 offset:4096 // (op 62) Transpose load local data 
ds_read_b96_tr_b6 v[41:43], v0 offset:4608 // (op 62) Transpose load local data 
s_barrier  // (op 889) 



ds_read_b96_tr_b6 v[44:46], v3 // (op 67) Transpose load local data 
ds_read_b96_tr_b6 v[47:49], v3 offset:512 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:43], v[44:49], a[0:3], v20, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 78) 



v_mov_b64 v[38:39], s[30:31] // (op 72) convert

v_mov_b32 v12, v14 // (op 72) 
s_lshl_b32 s0, s30, 3 // (op 72) 
v_add_u32 v29, v12, s0 // (op 72) 

v_mov_b32 v33, v29 // (op 72) 
buffer_load_ubyte v20, v33, s[60:63], 0 offen offset: 0 // (op 72) Load value




v_mov_b64 v[38:39], s[36:37] // (op 76) convert

v_mov_b32 v29, v17 // (op 76) 
s_lshl_b32 s0, s36, 3 // (op 76) 
v_add_u32 v33, v29, s0 // (op 76) 

v_mov_b32 v37, v33 // (op 76) 
buffer_load_ubyte v21, v37, s[64:67], 0 offen offset: 0 // (op 76) Load value





v_mov_b32 v33, v9 // (op 10) 
s_lshl_b64 s[0:1], s[26:27], 7 // (op 10) 
s_mul_i32 s7, s1, 6 // (op 10) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 10) least significant
s_mul_hi_u32 s18, s0, 6 // (op 10) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 10) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 10) 

s_lshl_b32 s6, s0, 1 // (op 10) 

v_add_u32 v15, v33, s6 // (op 10) 


buffer_load_dwordx3 v[22:24], v15, s[56:59], 0 offen // (op 10) Load value



v_mov_b32 v15, v11 // (op 4) 
s_lshl_b64 s[0:1], s[2:3], 7 // (op 4) 
s_mul_i32 s7, s1, 6 // (op 4) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 4) least significant
s_mul_hi_u32 s18, s0, 6 // (op 4) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 4) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 4) 

s_lshl_b32 s6, s0, 1 // (op 4) 

v_add_u32 v16, v15, s6 // (op 4) 


buffer_load_dwordx3 v[26:28], v16, s[52:55], 0 offen // (op 4) Load value

s_barrier  // (op 912) 



s_waitcnt vmcnt(5)
ds_write_b96 v13, v[30:32] offset:8192 // (op 222) Store local data 
s_barrier  // (op 906) 



s_waitcnt vmcnt(4)
ds_write_b96 v18, v[34:36] offset:12288 // (op 212) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 903) 



ds_read_b96_tr_b6 v[38:40], v4 offset:12288 // (op 214) Transpose load local data 
ds_read_b96_tr_b6 v[41:43], v4 offset:12800 // (op 214) Transpose load local data 
s_barrier  // (op 909) 
s_barrier  // (op 963) 



ds_read_b96_tr_b6 v[44:46], v5 offset:8192 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[47:49], v5 offset:8704 // (op 216) Transpose load local data 
s_barrier  // (op 969) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:43], v[44:49], a[0:3], v25, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 221) 



v_mov_b64 v[38:39], s[30:31] // (op 218) convert

s_lshl_b32 s0, s30, 2 // (op 218) 
s_mul_i32 s1, 3, s0 // (op 218) 

v_add_u32 v16, v12, s1 // (op 218) 

v_mov_b32 v37, v16 // (op 218) 
buffer_load_ubyte v25, v37, s[60:63], 0 offen offset: 0 // (op 218) Load value





s_lshl_b32 s0, s30, 3 // (op 561) 
v_add_u32 v14, v14, s0 // (op 561) 

v_mov_b64 v[38:39], s[36:37] // (op 220) convert

s_lshl_b32 s0, s36, 2 // (op 220) 
s_mul_i32 s1, 3, s0 // (op 220) 

v_add_u32 v12, v29, s1 // (op 220) 

v_mov_b32 v16, v12 // (op 220) 
buffer_load_ubyte v19, v16, s[64:67], 0 offen offset: 0 // (op 220) Load value






s_lshl_b32 s0, s36, 3 // (op 625) 
v_add_u32 v17, v17, s0 // (op 625) 

s_lshl_b64 s[0:1], s[26:27], 7 // (op 223) 
s_mul_i32 s7, s1, 6 // (op 223) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 223) least significant
s_mul_hi_u32 s18, s0, 6 // (op 223) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 223) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 223) 

s_mul_i32 s6, 3, s0 // (op 223) 

v_add_u32 v12, v33, s6 // (op 223) 


buffer_load_dwordx3 v[30:32], v12, s[56:59], 0 offen // (op 223) Load value




s_lshl_b64 s[0:1], s[2:3], 7 // (op 211) 
s_mul_i32 s7, s1, 6 // (op 211) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 211) least significant
s_mul_hi_u32 s18, s0, 6 // (op 211) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 211) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 211) 

s_mul_i32 s6, 3, s0 // (op 211) 

v_add_u32 v12, v15, s6 // (op 211) 


s_lshl_b64 s[0:1], s[26:27], 7 // (op 497) 
s_mul_i32 s7, s1, 6 // (op 497) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 497) least significant
s_mul_hi_u32 s18, s0, 6 // (op 497) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 497) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 497) 

s_lshl_b32 s6, s0, 1 // (op 497) 

v_add_u32 v9, v9, s6 // (op 497) 

buffer_load_dwordx3 v[34:36], v12, s[52:55], 0 offen // (op 211) Load value


s_barrier  // (op 892) 
s_lshl_b64 s[0:1], s[2:3], 7 // (op 433) 



s_mul_i32 s7, s1, 6 // (op 433) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 433) least significant
s_mul_hi_u32 s18, s0, 6 // (op 433) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 433) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 433) 

s_lshl_b32 s6, s0, 1 // (op 433) 

v_add_u32 v11, v11, s6 // (op 433) 

s_waitcnt vmcnt(5)
ds_write_b96 v10, v[22:24] // (op 13) Store local data 
s_barrier  // (op 879) 



s_waitcnt vmcnt(4)
ds_write_b96 v8, v[26:28] offset:4096 // (op 6) Store local data 
s_ashr_i64 s[0:1], s[24:25], 7 // (op 52) 
s_mov_b32 s6, s0 // (op 52) convert

s_and_b32 s7, s6, 4294967294 // (op 52) 

s_cmp_lt_u32 s51, s7 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_52 // (op 52) Condition: Bottom (jump to RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoop_52 if true)
RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoop_52:













s_mul_hi_i32 s51, s16, s9 // (op 396) 
v_and_b32 v33, -2, v1 // (op 396) 
s_add_i32 s0, s51, s16 // (op 396) 

v_sub_u32 v8, v1, v33 // (op 396) 

s_ashr_i32 s1, s0, 31 // (op 396) 
v_lshlrev_b32 v33, 4, v8 // (op 396) 

s_cmp_eq_i32 s9, 0 // (op 396) 
s_mov_b32 s51, scc // (op 396) 
v_lshrrev_b32 v8, 1, v1 // (op 396) 
s_cmp_eq_i32 s51, 0 // (op 396) 
s_cselect_b32 s6, 0, -1 // (op 396) 

s_and_b32 s7, s10, 31 // (op 396) 
s_lshl_b32 s51, 1, s7 // (op 396) 
s_add_i32 s18, s51, s6 // (op 396) 


s_and_b32 s6, s1, s18 // (op 396) 


s_add_i32 s1, s0, s6 // (op 396) 


s_ashr_i32 s0, s1, s7 // (op 396) 


s_lshl_b32 s1, s10, 24 // (op 396) 
s_ashr_i32 s51, s1, 31 // (op 396) 

s_xor_b32 s1, s0, s51 // (op 396) 

s_sub_i32 s0, s1, s51 // (op 396) 


s_mul_hi_i32 s1, s0, s11 // (op 396) 
s_add_i32 s51, s1, s0 // (op 396) 

s_ashr_i32 s1, s51, 31 // (op 396) 
s_cmp_eq_i32 s11, 0 // (op 396) 
s_mov_b32 s6, scc // (op 396) 
s_cmp_eq_i32 s6, 0 // (op 396) 
s_cselect_b32 s7, 0, -1 // (op 396) 

s_and_b32 s6, s12, 31 // (op 396) 
s_lshl_b32 s18, 1, s6 // (op 396) 
s_add_i32 s19, s18, s7 // (op 396) 


s_and_b32 s7, s1, s19 // (op 396) 


s_add_i32 s1, s51, s7 // (op 396) 


s_ashr_i32 s7, s1, s6 // (op 396) 


s_lshl_b32 s1, s12, 24 // (op 396) 
s_ashr_i32 s6, s1, 31 // (op 396) 

s_xor_b32 s1, s7, s6 // (op 396) 

s_sub_i32 s7, s1, s6 // (op 396) 


s_mul_i32 s1, s0, s14 // (op 396) 

s_sub_u32 s0, s16, s1 // (op 396) 

s_mul_i32 s1, s7, s14 // (op 396) 
s_add_u32 s1, s1, s0 // (op 396) 


s_mul_hi_i32 s0, s16, s8 // (op 396) 
s_add_i32 s51, s0, s16 // (op 396) 

s_ashr_i32 s0, s51, 31 // (op 396) 
s_cmp_eq_i32 s8, 0 // (op 396) 
s_mov_b32 s6, scc // (op 396) 
s_cmp_eq_i32 s6, 0 // (op 396) 
s_cselect_b32 s7, 0, -1 // (op 396) 

s_and_b32 s6, s15, 31 // (op 396) 
s_lshl_b32 s18, 1, s6 // (op 396) 
s_add_i32 s19, s18, s7 // (op 396) 


s_and_b32 s7, s0, s19 // (op 396) 


s_add_i32 s0, s51, s7 // (op 396) 


s_ashr_i32 s7, s0, s6 // (op 396) 


s_lshl_b32 s0, s15, 24 // (op 396) 
s_ashr_i32 s6, s0, 31 // (op 396) 

s_xor_b32 s0, s7, s6 // (op 396) 

s_sub_i32 s7, s0, s6 // (op 396) 


s_add_u32 s18, s4, 32 // (op 396) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 396) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 396) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 396) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 396) 

s_mul_i32 s0, s14, s18 // (op 396) 

s_mul_i32 s6, s50, s0 // (op 396) 

s_mul_i32 s0, s7, s6 // (op 396) 

s_sub_u32 s6, s16, s0 // (op 396) 

s_mov_b32 s0, s6 // (op 396) 
s_mul_hi_i32 s51, s0, s13 // (op 396) 
s_add_i32 s18, s51, s0 // (op 396) 


s_ashr_i32 s0, s18, 31 // (op 396) 
s_cmp_eq_i32 s13, 0 // (op 396) 
s_mov_b32 s19, scc // (op 396) 
s_cmp_eq_i32 s19, 0 // (op 396) 
s_cselect_b32 s51, 0, -1 // (op 396) 

s_and_b32 s19, s49, 31 // (op 396) 
s_lshl_b32 s68, 1, s19 // (op 396) 
s_add_i32 s69, s68, s51 // (op 396) 


s_and_b32 s51, s0, s69 // (op 396) 


s_add_i32 s0, s18, s51 // (op 396) 


s_ashr_i32 s18, s0, s19 // (op 396) 


s_lshl_b32 s0, s49, 24 // (op 396) 
s_ashr_i32 s19, s0, 31 // (op 396) 

s_xor_b32 s0, s18, s19 // (op 396) 

s_sub_i32 s18, s0, s19 // (op 396) 


s_mul_i32 s0, s48, s14 // (op 396) 
s_add_u32 s68, s20, 32 // (op 396) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 396) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 396) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 396) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 396) 

s_sub_i32 s19, s68, s0 // (op 396) 

s_mul_i32 s51, s18, s19 // (op 396) 


s_sub_u32 s18, s6, s51 // (op 396) 


s_add_u32 s6, s18, s0 // (op 396) 


s_cmp_eq_i32 s7, 0 // (op 396) 
s_mov_b32 s0, scc // (op 396) 

s_cmp_eq_i32 s0, 0 // (op 396) 
s_cselect_b32 s7, s6, s1 // (op 396) 



v_lshl_add_u32 v9, s7, 5, v33 // (op 396) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 396) 
v_mov_b32 v11, 0 // (op 396) convert
v_mov_b32 v10, v9 // (op 396) convert

s_lshr_b32 s51, s0, 1 // (op 396) 

v_lshl_add_u32 v9, s51, 8, v8 // (op 396) 


v_mul_lo_u32 v13, v9, s3 // (op 396) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 396) most significant: high of low * low
v_mul_lo_u32 v12, v9, s2 // (op 396) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 396) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v12 // (op 396) least significant half
v_addc_co_u32 v9, s[0:1], v11, v13, s[0:1] // (op 396) most significant half



v_mul_lo_u32 v11, v9, 6 // (op 396) most significant: low of high * low
v_mul_hi_u32 v33, v8, 6 // (op 396) most significant: high of low * low
v_mul_lo_u32 v10, v8, 6 // (op 396) least significant: low of low * low
v_add_u32 v11, v11, v33 // (op 396) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[10:11] // (op 396) 

v_mov_b32 v33, v8 // (op 396) convert

s_mul_i32 s1, s25, 6 // (op 398) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 398) least significant
s_mul_hi_u32 s51, s24, 6 // (op 398) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 398) most significant: sum

s_mul_i32 s7, s2, s1 // (op 398) most significant: low of low * high
s_mul_i32 s51, s3, s0 // (op 398) most significant: low of high * low
s_mul_i32 s6, s2, s0 // (op 398) least significant
s_mul_hi_u32 s18, s2, s0 // (op 398) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 398) most significant: sum
s_add_u32 s7, s7, s18 // (op 398) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 398) 

s_mov_b32 s52, s22 // (op 398) reg expression
s_mov_b32 s53, s23 // (op 398) reg expression
s_mov_b32 s54, s0 // (op 398) reg expression
s_mov_b32 s55, 131072 // (op 398) call()

s_mul_hi_i32 s51, s16, s9 // (op 463) 
v_and_b32 v8, -2, v1 // (op 463) 
s_add_i32 s0, s51, s16 // (op 463) 

v_sub_u32 v9, v1, v8 // (op 463) 

s_ashr_i32 s1, s0, 31 // (op 463) 
v_lshlrev_b32 v8, 4, v9 // (op 463) 

s_cmp_eq_i32 s9, 0 // (op 463) 
s_mov_b32 s51, scc // (op 463) 
v_lshrrev_b32 v9, 1, v1 // (op 463) 
s_cmp_eq_i32 s51, 0 // (op 463) 
s_cselect_b32 s6, 0, -1 // (op 463) 

s_and_b32 s7, s10, 31 // (op 463) 
s_lshl_b32 s51, 1, s7 // (op 463) 
s_add_i32 s18, s51, s6 // (op 463) 


s_and_b32 s6, s1, s18 // (op 463) 


s_add_i32 s1, s0, s6 // (op 463) 


s_ashr_i32 s0, s1, s7 // (op 463) 


s_lshl_b32 s1, s10, 24 // (op 463) 
s_ashr_i32 s51, s1, 31 // (op 463) 

s_xor_b32 s1, s0, s51 // (op 463) 

s_sub_i32 s0, s1, s51 // (op 463) 


s_mul_hi_i32 s1, s0, s11 // (op 463) 
s_add_i32 s51, s1, s0 // (op 463) 

s_ashr_i32 s1, s51, 31 // (op 463) 
s_cmp_eq_i32 s11, 0 // (op 463) 
s_mov_b32 s6, scc // (op 463) 
s_cmp_eq_i32 s6, 0 // (op 463) 
s_cselect_b32 s7, 0, -1 // (op 463) 

s_and_b32 s6, s12, 31 // (op 463) 
s_lshl_b32 s18, 1, s6 // (op 463) 
s_add_i32 s19, s18, s7 // (op 463) 


s_and_b32 s7, s1, s19 // (op 463) 


s_add_i32 s1, s51, s7 // (op 463) 


s_ashr_i32 s7, s1, s6 // (op 463) 


s_lshl_b32 s1, s12, 24 // (op 463) 
s_ashr_i32 s6, s1, 31 // (op 463) 

s_xor_b32 s1, s7, s6 // (op 463) 

s_sub_i32 s7, s1, s6 // (op 463) 


s_add_u32 s18, s4, 32 // (op 463) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 463) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 463) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 463) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 463) 

s_mul_i32 s1, s7, s18 // (op 463) 

s_sub_i32 s51, s0, s1 // (op 463) 


s_mul_hi_i32 s0, s16, s8 // (op 463) 
s_add_i32 s1, s0, s16 // (op 463) 

s_ashr_i32 s0, s1, 31 // (op 463) 
s_cmp_eq_i32 s8, 0 // (op 463) 
s_mov_b32 s6, scc // (op 463) 
s_cmp_eq_i32 s6, 0 // (op 463) 
s_cselect_b32 s7, 0, -1 // (op 463) 

s_and_b32 s6, s15, 31 // (op 463) 
s_lshl_b32 s68, 1, s6 // (op 463) 
s_add_i32 s69, s68, s7 // (op 463) 


s_and_b32 s7, s0, s69 // (op 463) 


s_add_i32 s0, s1, s7 // (op 463) 


s_ashr_i32 s1, s0, s6 // (op 463) 


s_lshl_b32 s0, s15, 24 // (op 463) 
s_ashr_i32 s6, s0, 31 // (op 463) 

s_xor_b32 s0, s1, s6 // (op 463) 

s_sub_i32 s1, s0, s6 // (op 463) 


s_mul_i32 s0, s14, s18 // (op 463) 

s_mul_i32 s6, s50, s0 // (op 463) 

s_mul_i32 s0, s1, s6 // (op 463) 

s_sub_u32 s6, s16, s0 // (op 463) 

s_mul_hi_i32 s0, s6, s13 // (op 463) 
s_add_i32 s7, s0, s6 // (op 463) 


s_ashr_i32 s0, s7, 31 // (op 463) 
s_cmp_eq_i32 s13, 0 // (op 463) 
s_mov_b32 s6, scc // (op 463) 
s_cmp_eq_i32 s6, 0 // (op 463) 
s_cselect_b32 s18, 0, -1 // (op 463) 

s_and_b32 s6, s49, 31 // (op 463) 
s_lshl_b32 s19, 1, s6 // (op 463) 
s_add_i32 s68, s19, s18 // (op 463) 


s_and_b32 s18, s0, s68 // (op 463) 


s_add_i32 s0, s7, s18 // (op 463) 


s_ashr_i32 s7, s0, s6 // (op 463) 


s_lshl_b32 s0, s49, 24 // (op 463) 
s_ashr_i32 s6, s0, 31 // (op 463) 

s_xor_b32 s0, s7, s6 // (op 463) 

s_sub_i32 s7, s0, s6 // (op 463) 


s_cmp_eq_i32 s1, 0 // (op 463) 
s_mov_b32 s0, scc // (op 463) 

s_cmp_eq_i32 s0, 0 // (op 463) 
s_cselect_b32 s1, s7, s51 // (op 463) 



v_lshl_add_u32 v10, s1, 5, v8 // (op 463) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 463) 
v_mov_b32 v21, 0 // (op 463) convert
v_mov_b32 v20, v10 // (op 463) convert

s_lshr_b32 s51, s0, 1 // (op 463) 

v_lshl_add_u32 v8, s51, 8, v9 // (op 463) 


v_mul_lo_u32 v23, v8, s27 // (op 463) most significant: low of low * high
v_mul_hi_u32 v24, v8, s26 // (op 463) most significant: high of low * low
v_mul_lo_u32 v22, v8, s26 // (op 463) least significant: low of low * low
v_add_u32 v23, v23, v24 // (op 463) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v20 // (op 463) least significant half
v_addc_co_u32 v23, s[0:1], v23, v21, s[0:1] // (op 463) most significant half



v_mul_lo_u32 v21, v23, 6 // (op 463) most significant: low of high * low
v_mul_hi_u32 v24, v22, 6 // (op 463) most significant: high of low * low
v_mul_lo_u32 v20, v22, 6 // (op 463) least significant: low of low * low
v_add_u32 v21, v21, v24 // (op 463) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[20:21] // (op 463) 

v_mov_b32 v10, v8 // (op 463) convert

s_mul_i32 s1, s25, 6 // (op 465) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 465) least significant
s_mul_hi_u32 s51, s24, 6 // (op 465) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 465) most significant: sum

s_mul_i32 s7, s26, s1 // (op 465) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 465) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 465) least significant
s_mul_hi_u32 s18, s26, s0 // (op 465) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 465) most significant: sum
s_add_u32 s7, s7, s18 // (op 465) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 465) 

s_mov_b32 s56, s28 // (op 465) reg expression
s_mov_b32 s57, s29 // (op 465) reg expression
s_mov_b32 s58, s0 // (op 465) reg expression
s_mov_b32 s59, 131072 // (op 465) call()

s_mul_hi_i32 s51, s16, s9 // (op 527) 
v_lshrrev_b32 v8, 7, v1 // (op 527) 
s_add_i32 s0, s51, s16 // (op 527) 

v_and_b32 v9, 15, v1 // (op 527) 
s_ashr_i32 s1, s0, 31 // (op 527) 
v_lshl_add_u32 v11, v8, 4, v9 // (op 527) 


s_cmp_eq_i32 s9, 0 // (op 527) 
s_mov_b32 s51, scc // (op 527) 
v_and_b32 v8, 63, v1 // (op 527) 
s_cmp_eq_i32 s51, 0 // (op 527) 
s_cselect_b32 s6, 0, -1 // (op 527) 

v_lshrrev_b32 v9, 4, v8 // (op 527) 

s_and_b32 s7, s10, 31 // (op 527) 
s_lshl_b32 s51, 1, s7 // (op 527) 
s_add_i32 s18, s51, s6 // (op 527) 


s_and_b32 s6, s1, s18 // (op 527) 


s_add_i32 s1, s0, s6 // (op 527) 


s_ashr_i32 s0, s1, s7 // (op 527) 


s_lshl_b32 s1, s10, 24 // (op 527) 
s_ashr_i32 s51, s1, 31 // (op 527) 

s_xor_b32 s1, s0, s51 // (op 527) 

s_sub_i32 s0, s1, s51 // (op 527) 


s_mul_hi_i32 s1, s0, s11 // (op 527) 
s_add_i32 s51, s1, s0 // (op 527) 

s_ashr_i32 s1, s51, 31 // (op 527) 
s_cmp_eq_i32 s11, 0 // (op 527) 
s_mov_b32 s6, scc // (op 527) 
s_cmp_eq_i32 s6, 0 // (op 527) 
s_cselect_b32 s7, 0, -1 // (op 527) 

s_and_b32 s6, s12, 31 // (op 527) 
s_lshl_b32 s18, 1, s6 // (op 527) 
s_add_i32 s19, s18, s7 // (op 527) 


s_and_b32 s7, s1, s19 // (op 527) 


s_add_i32 s1, s51, s7 // (op 527) 


s_ashr_i32 s7, s1, s6 // (op 527) 


s_lshl_b32 s1, s12, 24 // (op 527) 
s_ashr_i32 s6, s1, 31 // (op 527) 

s_xor_b32 s1, s7, s6 // (op 527) 

s_sub_i32 s7, s1, s6 // (op 527) 


s_mul_i32 s1, s0, s14 // (op 527) 

s_sub_u32 s0, s16, s1 // (op 527) 

s_mul_i32 s1, s7, s14 // (op 527) 
s_add_u32 s1, s1, s0 // (op 527) 


s_mul_hi_i32 s0, s16, s8 // (op 527) 
s_add_i32 s51, s0, s16 // (op 527) 

s_ashr_i32 s0, s51, 31 // (op 527) 
s_cmp_eq_i32 s8, 0 // (op 527) 
s_mov_b32 s6, scc // (op 527) 
s_cmp_eq_i32 s6, 0 // (op 527) 
s_cselect_b32 s7, 0, -1 // (op 527) 

s_and_b32 s6, s15, 31 // (op 527) 
s_lshl_b32 s18, 1, s6 // (op 527) 
s_add_i32 s19, s18, s7 // (op 527) 


s_and_b32 s7, s0, s19 // (op 527) 


s_add_i32 s0, s51, s7 // (op 527) 


s_ashr_i32 s7, s0, s6 // (op 527) 


s_lshl_b32 s0, s15, 24 // (op 527) 
s_ashr_i32 s6, s0, 31 // (op 527) 

s_xor_b32 s0, s7, s6 // (op 527) 

s_sub_i32 s7, s0, s6 // (op 527) 


s_add_u32 s18, s4, 32 // (op 527) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 527) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 527) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 527) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 527) 

s_mul_i32 s0, s14, s18 // (op 527) 

s_mul_i32 s6, s50, s0 // (op 527) 

s_mul_i32 s0, s7, s6 // (op 527) 

s_sub_u32 s6, s16, s0 // (op 527) 

s_mov_b32 s0, s6 // (op 527) 
s_mul_hi_i32 s51, s0, s13 // (op 527) 
s_add_i32 s18, s51, s0 // (op 527) 


s_ashr_i32 s0, s18, 31 // (op 527) 
s_cmp_eq_i32 s13, 0 // (op 527) 
s_mov_b32 s19, scc // (op 527) 
s_cmp_eq_i32 s19, 0 // (op 527) 
s_cselect_b32 s51, 0, -1 // (op 527) 

s_and_b32 s19, s49, 31 // (op 527) 
s_lshl_b32 s68, 1, s19 // (op 527) 
s_add_i32 s69, s68, s51 // (op 527) 


s_and_b32 s51, s0, s69 // (op 527) 


s_add_i32 s0, s18, s51 // (op 527) 


s_ashr_i32 s18, s0, s19 // (op 527) 


s_lshl_b32 s0, s49, 24 // (op 527) 
s_ashr_i32 s19, s0, 31 // (op 527) 

s_xor_b32 s0, s18, s19 // (op 527) 

s_sub_i32 s18, s0, s19 // (op 527) 


s_mul_i32 s0, s48, s14 // (op 527) 
s_add_u32 s68, s20, 32 // (op 527) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 527) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 527) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 527) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 527) 

s_sub_i32 s19, s68, s0 // (op 527) 

s_mul_i32 s51, s18, s19 // (op 527) 


s_sub_u32 s18, s6, s51 // (op 527) 


s_add_u32 s6, s18, s0 // (op 527) 


s_cmp_eq_i32 s7, 0 // (op 527) 
s_mov_b32 s0, scc // (op 527) 

s_cmp_eq_i32 s0, 0 // (op 527) 
s_cselect_b32 s7, s6, s1 // (op 527) 



v_lshl_add_u32 v8, s7, 5, v11 // (op 527) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 527) 
s_lshr_b32 s51, s0, 1 // (op 527) 

v_lshl_add_u32 v11, s51, 3, v9 // (op 527) 


v_mul_lo_u32 v9, v11, s30 // (op 527) 

v_add_u32 v11, v8, v9 // (op 527) 


s_mul_i32 s1, s30, s35 // (op 529) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 529) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 529) least significant
s_mul_hi_u32 s6, s30, s34 // (op 529) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 529) most significant: sum
s_add_u32 s1, s1, s6 // (op 529) most significant: sum


s_mov_b32 s60, s32 // (op 529) reg expression
s_mov_b32 s61, s33 // (op 529) reg expression
s_mov_b32 s62, s0 // (op 529) reg expression
s_mov_b32 s63, 131072 // (op 529) call()

s_mul_hi_i32 s51, s16, s9 // (op 591) 
v_lshrrev_b32 v8, 6, v1 // (op 591) 
s_add_i32 s0, s51, s16 // (op 591) 

v_and_b32 v9, 1, v8 // (op 591) 

s_ashr_i32 s1, s0, 31 // (op 591) 
v_and_b32 v8, 15, v1 // (op 591) 
s_cmp_eq_i32 s9, 0 // (op 591) 
s_mov_b32 s51, scc // (op 591) 
v_lshl_add_u32 v12, v9, 4, v8 // (op 591) 


s_cmp_eq_i32 s51, 0 // (op 591) 
s_cselect_b32 s6, 0, -1 // (op 591) 

v_and_b32 v8, 63, v1 // (op 591) 
s_and_b32 s7, s10, 31 // (op 591) 
v_lshrrev_b32 v9, 4, v8 // (op 591) 

s_lshl_b32 s51, 1, s7 // (op 591) 
s_add_i32 s18, s51, s6 // (op 591) 


s_and_b32 s6, s1, s18 // (op 591) 


s_add_i32 s1, s0, s6 // (op 591) 


s_ashr_i32 s0, s1, s7 // (op 591) 


s_lshl_b32 s1, s10, 24 // (op 591) 
s_ashr_i32 s51, s1, 31 // (op 591) 

s_xor_b32 s1, s0, s51 // (op 591) 

s_sub_i32 s0, s1, s51 // (op 591) 


s_mul_hi_i32 s1, s0, s11 // (op 591) 
s_add_i32 s51, s1, s0 // (op 591) 

s_ashr_i32 s1, s51, 31 // (op 591) 
s_cmp_eq_i32 s11, 0 // (op 591) 
s_mov_b32 s6, scc // (op 591) 
s_cmp_eq_i32 s6, 0 // (op 591) 
s_cselect_b32 s7, 0, -1 // (op 591) 

s_and_b32 s6, s12, 31 // (op 591) 
s_lshl_b32 s18, 1, s6 // (op 591) 
s_add_i32 s19, s18, s7 // (op 591) 


s_and_b32 s7, s1, s19 // (op 591) 


s_add_i32 s1, s51, s7 // (op 591) 


s_ashr_i32 s7, s1, s6 // (op 591) 


s_lshl_b32 s1, s12, 24 // (op 591) 
s_ashr_i32 s6, s1, 31 // (op 591) 

s_xor_b32 s1, s7, s6 // (op 591) 

s_sub_i32 s7, s1, s6 // (op 591) 


s_add_u32 s18, s4, 32 // (op 591) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 591) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 591) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 591) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 591) 

s_mul_i32 s1, s7, s18 // (op 591) 

s_sub_i32 s51, s0, s1 // (op 591) 


s_mul_hi_i32 s0, s16, s8 // (op 591) 
s_add_i32 s1, s0, s16 // (op 591) 

s_ashr_i32 s0, s1, 31 // (op 591) 
s_cmp_eq_i32 s8, 0 // (op 591) 
s_mov_b32 s6, scc // (op 591) 
s_cmp_eq_i32 s6, 0 // (op 591) 
s_cselect_b32 s7, 0, -1 // (op 591) 

s_and_b32 s6, s15, 31 // (op 591) 
s_lshl_b32 s68, 1, s6 // (op 591) 
s_add_i32 s69, s68, s7 // (op 591) 


s_and_b32 s7, s0, s69 // (op 591) 


s_add_i32 s0, s1, s7 // (op 591) 


s_ashr_i32 s1, s0, s6 // (op 591) 


s_lshl_b32 s0, s15, 24 // (op 591) 
s_ashr_i32 s6, s0, 31 // (op 591) 

s_xor_b32 s0, s1, s6 // (op 591) 

s_sub_i32 s1, s0, s6 // (op 591) 


s_mul_i32 s0, s14, s18 // (op 591) 

s_mul_i32 s6, s50, s0 // (op 591) 

s_mul_i32 s0, s1, s6 // (op 591) 

s_sub_u32 s6, s16, s0 // (op 591) 

s_mul_hi_i32 s0, s6, s13 // (op 591) 
s_add_i32 s7, s0, s6 // (op 591) 


s_ashr_i32 s0, s7, 31 // (op 591) 
s_cmp_eq_i32 s13, 0 // (op 591) 
s_mov_b32 s6, scc // (op 591) 
s_cmp_eq_i32 s6, 0 // (op 591) 
s_cselect_b32 s18, 0, -1 // (op 591) 

s_and_b32 s6, s49, 31 // (op 591) 
s_lshl_b32 s19, 1, s6 // (op 591) 
s_add_i32 s68, s19, s18 // (op 591) 


s_and_b32 s18, s0, s68 // (op 591) 


s_add_i32 s0, s7, s18 // (op 591) 


s_ashr_i32 s7, s0, s6 // (op 591) 


s_lshl_b32 s0, s49, 24 // (op 591) 
s_ashr_i32 s6, s0, 31 // (op 591) 

s_xor_b32 s0, s7, s6 // (op 591) 

s_sub_i32 s7, s0, s6 // (op 591) 


s_cmp_eq_i32 s1, 0 // (op 591) 
s_mov_b32 s0, scc // (op 591) 

s_cmp_eq_i32 s0, 0 // (op 591) 
s_cselect_b32 s1, s7, s51 // (op 591) 



v_lshl_add_u32 v8, s1, 5, v12 // (op 591) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 591) 
s_lshr_b32 s51, s0, 1 // (op 591) 

v_lshl_add_u32 v12, s51, 3, v9 // (op 591) 


v_mul_lo_u32 v9, v12, s36 // (op 591) 
v_add_u32 v9, v9, v8 // (op 591) 


s_mul_i32 s1, s36, s35 // (op 593) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 593) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 593) least significant
s_mul_hi_u32 s6, s36, s34 // (op 593) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 593) most significant: sum
s_add_u32 s1, s1, s6 // (op 593) most significant: sum


s_mov_b32 s64, s38 // (op 593) reg expression
s_mov_b32 s65, s39 // (op 593) reg expression
s_mov_b32 s66, s0 // (op 593) reg expression
s_mov_b32 s67, 131072 // (op 593) call()

v_and_b32 v8, -2, v1 // (op 684) 
v_sub_u32 v12, v1, v8 // (op 684) 

v_lshlrev_b32 v8, 4, v12 // (op 684) 

v_lshrrev_b32 v12, 1, v1 // (op 684) 
v_lshl_add_u32 v13, v12, 5, v8 // (op 684) 


v_mul_lo_u32 v8, v13, 6 // (op 684) 
v_lshrrev_b32 v12, 3, v8 // (op 684) 

v_lshrrev_b32 v8, 4, v13 // (op 684) 

v_lshlrev_b32 v13, 2, v8 // (op 684) 

v_add_u32 v8, v12, v13 // (op 684) 


v_and_b32 v12, -2, v1 // (op 737) 
v_sub_u32 v13, v1, v12 // (op 737) 

v_lshlrev_b32 v12, 4, v13 // (op 737) 

v_lshrrev_b32 v13, 1, v1 // (op 737) 
v_lshl_add_u32 v14, v13, 5, v12 // (op 737) 


v_mul_lo_u32 v12, v14, 6 // (op 737) 
v_lshrrev_b32 v13, 3, v12 // (op 737) 

v_lshrrev_b32 v12, 4, v14 // (op 737) 

v_lshlrev_b32 v14, 2, v12 // (op 737) 

v_add_u32 v12, v13, v14 // (op 737) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 157) 
s_and_b32 s51, s0, 4294967294 // (op 157) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 156) 
s_mov_b32 s6, s0 // (op 156) convert

s_cmp_lt_u32 s51, s6 // (op 156) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_156:


s_add_u32 s51, s51, 1 // (op 158) 
v_mov_b32 v13, v33 // (op 161) 
v_mov_b32 v14, v13 // (op 161) reg expression

buffer_load_dwordx3 v[16:18], v14, s[52:55], 0 offen // (op 161) Load value


s_lshl_b64 s[0:1], s[2:3], 7 // (op 386) 
s_mul_i32 s7, s1, 6 // (op 386) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 386) least significant
s_mul_hi_u32 s18, s0, 6 // (op 386) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 386) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 386) 

v_add_u32 v33, v33, s0 // (op 386) 



s_barrier  // (op 882) 
v_mov_b32 v13, v10 // (op 173) 



v_mov_b32 v14, v13 // (op 173) reg expression

buffer_load_dwordx3 v[20:22], v14, s[56:59], 0 offen // (op 173) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 455) 
s_mul_i32 s7, s1, 6 // (op 455) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 455) least significant
s_mul_hi_u32 s18, s0, 6 // (op 455) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 455) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 455) 

v_add_u32 v10, v10, s0 // (op 455) 

s_waitcnt vmcnt(1)
ds_write_b96 v8, v[16:18] offset:4096 // (op 162) Store local data 

s_barrier  // (op 895) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 885) 






ds_read_b96_tr_b6 v[14:16], v0 offset:4096 // (op 164) Transpose load local data 
ds_read_b96_tr_b6 v[17:19], v0 offset:4608 // (op 164) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b96 v12, v[20:22] // (op 172) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 899) 



ds_read_b96_tr_b6 v[20:22], v3 // (op 166) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v3 offset:512 // (op 166) Transpose load local data 

v_mov_b64 v[26:27], s[30:31] // (op 168) convert

v_mov_b32 v13, v11 // (op 168) 
v_mov_b32 v28, v13 // (op 168) reg expression
v_mov_b32 v29, v28 // (op 168) 
buffer_load_ubyte v30, v29, s[60:63], 0 offen offset: 0 // (op 168) Load value




s_lshl_b32 s0, s30, 2 // (op 519) 
v_add_u32 v11, v11, s0 // (op 519) 


v_mov_b64 v[26:27], s[36:37] // (op 170) convert

v_mov_b32 v13, v9 // (op 170) 
v_mov_b32 v28, v13 // (op 170) reg expression
v_mov_b32 v29, v28 // (op 170) 
buffer_load_ubyte v31, v29, s[64:67], 0 offen offset: 0 // (op 170) Load value




s_lshl_b32 s0, s36, 2 // (op 583) 
v_add_u32 v9, v9, s0 // (op 583) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[14:19], v[20:25], a[0:3], v30, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 171) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 156) 
s_mov_b32 s6, s0 // (op 156) convert

s_cmp_lt_u32 s51, s6 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopTop_KLoopTail_156 if true)
RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_156:





















s_barrier  // (op 957) 




v_cmp_eq_f32 vcc, 0.00000, v6 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_116
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v8, 16, v12 // (op 48) 
v_lshrrev_b32 v9, 16, v13 // (op 48) 
v_lshrrev_b32 v10, 16, v14 // (op 48) 
v_lshrrev_b32 v11, 16, v15 // (op 48) 

s_mul_hi_i32 s0, s16, s9 // (op 779) 
v_lshrrev_b32 v0, 7, v1 // (op 779) 
s_add_i32 s1, s0, s16 // (op 779) 

v_and_b32 v3, 63, v1 // (op 779) 
s_ashr_i32 s0, s1, 31 // (op 779) 
v_lshrrev_b32 v4, 4, v3 // (op 779) 
s_cmp_eq_i32 s9, 0 // (op 779) 
s_mov_b32 s2, scc // (op 779) 
v_lshlrev_b32 v5, 2, v4 // (op 779) 

s_cmp_eq_i32 s2, 0 // (op 779) 
s_cselect_b32 s3, 0, -1 // (op 779) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 779) 


s_and_b32 s2, s10, 31 // (op 779) 
v_lshrrev_b32 v0, 2, v3 // (op 779) 

s_lshl_b32 s6, 1, s2 // (op 779) 
v_and_b32 v3, 3, v0 // (op 779) 

s_add_i32 s7, s6, s3 // (op 779) 


v_and_b32 v0, 3, v1 // (op 779) 
s_and_b32 s3, s0, s7 // (op 779) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 779) 


s_add_i32 s0, s1, s3 // (op 779) 


v_lshrrev_b32 v0, 6, v1 // (op 779) 
s_ashr_i32 s1, s0, s2 // (op 779) 


v_and_b32 v3, 1, v0 // (op 779) 

s_lshl_b32 s0, s10, 24 // (op 779) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 779) 


s_ashr_i32 s2, s0, 31 // (op 779) 

s_xor_b32 s0, s1, s2 // (op 779) 

s_sub_i32 s1, s0, s2 // (op 779) 


s_mul_hi_i32 s0, s1, s11 // (op 779) 
s_add_i32 s2, s0, s1 // (op 779) 

s_ashr_i32 s0, s2, 31 // (op 779) 
s_cmp_eq_i32 s11, 0 // (op 779) 
s_mov_b32 s3, scc // (op 779) 
s_cmp_eq_i32 s3, 0 // (op 779) 
s_cselect_b32 s6, 0, -1 // (op 779) 

s_and_b32 s3, s12, 31 // (op 779) 
s_lshl_b32 s7, 1, s3 // (op 779) 
s_add_i32 s18, s7, s6 // (op 779) 


s_and_b32 s19, s0, s18 // (op 779) 


s_add_i32 s0, s2, s19 // (op 779) 


s_ashr_i32 s2, s0, s3 // (op 779) 


s_lshl_b32 s0, s12, 24 // (op 779) 
s_ashr_i32 s3, s0, 31 // (op 779) 

s_xor_b32 s0, s2, s3 // (op 779) 

s_sub_i32 s2, s0, s3 // (op 779) 


s_mul_i32 s0, s1, s14 // (op 779) 
s_sub_u32 s3, s16, s0 // (op 779) 

s_mul_i32 s0, s2, s14 // (op 779) 
s_add_u32 s0, s0, s3 // (op 779) 

s_mul_hi_i32 s3, s16, s8 // (op 779) 
s_add_i32 s6, s3, s16 // (op 779) 

s_ashr_i32 s3, s6, 31 // (op 779) 
s_cmp_eq_i32 s8, 0 // (op 779) 
s_mov_b32 s7, scc // (op 779) 
s_cmp_eq_i32 s7, 0 // (op 779) 
s_cselect_b32 s18, 0, -1 // (op 779) 

s_and_b32 s7, s15, 31 // (op 779) 
s_lshl_b32 s19, 1, s7 // (op 779) 
s_add_i32 s22, s19, s18 // (op 779) 


s_and_b32 s18, s3, s22 // (op 779) 


s_add_i32 s3, s6, s18 // (op 779) 


s_ashr_i32 s6, s3, s7 // (op 779) 


s_lshl_b32 s3, s15, 24 // (op 779) 
s_ashr_i32 s7, s3, 31 // (op 779) 

s_xor_b32 s3, s6, s7 // (op 779) 

s_sub_i32 s6, s3, s7 // (op 779) 


s_add_u32 s18, s4, 32 // (op 779) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 779) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 779) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 779) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 5 // (op 779) 

s_mul_i32 s3, s14, s18 // (op 779) 
s_mul_i32 s7, s50, s3 // (op 779) 

s_mul_i32 s3, s6, s7 // (op 779) 

s_sub_u32 s7, s16, s3 // (op 779) 

s_mov_b32 s3, s7 // (op 779) 
s_mul_hi_i32 s22, s3, s13 // (op 779) 
s_add_i32 s23, s22, s3 // (op 779) 


s_ashr_i32 s3, s23, 31 // (op 779) 
s_cmp_eq_i32 s13, 0 // (op 779) 
s_mov_b32 s22, scc // (op 779) 
s_cmp_eq_i32 s22, 0 // (op 779) 
s_cselect_b32 s24, 0, -1 // (op 779) 

s_and_b32 s22, s49, 31 // (op 779) 
s_lshl_b32 s25, 1, s22 // (op 779) 
s_add_i32 s26, s25, s24 // (op 779) 


s_and_b32 s24, s3, s26 // (op 779) 


s_add_i32 s3, s23, s24 // (op 779) 


s_ashr_i32 s23, s3, s22 // (op 779) 


s_lshl_b32 s3, s49, 24 // (op 779) 
s_ashr_i32 s22, s3, 31 // (op 779) 

s_xor_b32 s3, s23, s22 // (op 779) 

s_sub_i32 s23, s3, s22 // (op 779) 


s_mul_i32 s3, s48, s14 // (op 779) 
s_add_u32 s24, s20, 32 // (op 779) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 779) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 779) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 779) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 779) 

s_sub_i32 s22, s24, s3 // (op 779) 

s_mul_i32 s24, s23, s22 // (op 779) 

s_sub_u32 s22, s7, s24 // (op 779) 


s_add_u32 s7, s22, s3 // (op 779) 


s_cmp_eq_i32 s6, 0 // (op 779) 
s_mov_b32 s3, scc // (op 779) 

s_cmp_eq_i32 s3, 0 // (op 779) 
s_cselect_b32 s6, s7, s0 // (op 779) 


s_mul_i32 s0, s2, s18 // (op 779) 


v_lshl_add_u32 v3, s6, 5, v4 // (op 779) 


s_sub_i32 s2, s1, s0 // (op 779) 


s_cmp_eq_i32 s3, 0 // (op 779) 
s_cselect_b32 s22, s23, s2 // (op 779) 



v_lshl_add_u32 v4, s22, 5, v0 // (op 779) 


v_mul_lo_u32 v0, v4, s40 // (op 779) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 779) 


s_mul_i32 s7, s40, s5 // (op 781) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 781) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 781) least significant
s_mul_hi_u32 s1, s40, s4 // (op 781) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 781) most significant: sum
s_add_u32 s7, s7, s1 // (op 781) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 781) 

s_mov_b32 s0, s42 // (op 781) reg expression
s_mov_b32 s1, s43 // (op 781) reg expression
s_mov_b32 s2, s18 // (op 781) reg expression
s_mov_b32 s3, 131072 // (op 781) call()


v_pack_b32_f16 v12, v8, v9 // (op 50) 
v_pack_b32_f16 v13, v10, v11 // (op 50) 


buffer_store_dwordx2 v[12:13], v4, s[0:3], 0 offen // (op 50) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_116
RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalFalse_0_beta_116:
s_mul_hi_i32 s6, s16, s9 // (op 654) 
v_lshrrev_b32 v0, 7, v1 // (op 654) 
s_add_i32 s7, s6, s16 // (op 654) 

v_and_b32 v3, 63, v1 // (op 654) 
s_ashr_i32 s6, s7, 31 // (op 654) 
v_lshrrev_b32 v4, 4, v3 // (op 654) 
s_cmp_eq_i32 s9, 0 // (op 654) 
s_mov_b32 s18, scc // (op 654) 
v_lshlrev_b32 v5, 2, v4 // (op 654) 

s_cmp_eq_i32 s18, 0 // (op 654) 
s_cselect_b32 s19, 0, -1 // (op 654) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 654) 


s_and_b32 s18, s10, 31 // (op 654) 
v_lshrrev_b32 v0, 2, v3 // (op 654) 

s_lshl_b32 s22, 1, s18 // (op 654) 
v_and_b32 v3, 3, v0 // (op 654) 

s_add_i32 s23, s22, s19 // (op 654) 


v_and_b32 v0, 3, v1 // (op 654) 
s_and_b32 s19, s6, s23 // (op 654) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 654) 


s_add_i32 s6, s7, s19 // (op 654) 


v_lshrrev_b32 v0, 6, v1 // (op 654) 
s_ashr_i32 s7, s6, s18 // (op 654) 


v_and_b32 v3, 1, v0 // (op 654) 

s_lshl_b32 s6, s10, 24 // (op 654) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 654) 


s_ashr_i32 s18, s6, 31 // (op 654) 

s_xor_b32 s6, s7, s18 // (op 654) 

s_sub_i32 s7, s6, s18 // (op 654) 


s_mul_hi_i32 s6, s7, s11 // (op 654) 
s_add_i32 s18, s6, s7 // (op 654) 

s_ashr_i32 s6, s18, 31 // (op 654) 
s_cmp_eq_i32 s11, 0 // (op 654) 
s_mov_b32 s19, scc // (op 654) 
s_cmp_eq_i32 s19, 0 // (op 654) 
s_cselect_b32 s22, 0, -1 // (op 654) 

s_and_b32 s19, s12, 31 // (op 654) 
s_lshl_b32 s23, 1, s19 // (op 654) 
s_add_i32 s24, s23, s22 // (op 654) 


s_and_b32 s22, s6, s24 // (op 654) 


s_add_i32 s6, s18, s22 // (op 654) 


s_ashr_i32 s18, s6, s19 // (op 654) 


s_lshl_b32 s6, s12, 24 // (op 654) 
s_ashr_i32 s19, s6, 31 // (op 654) 

s_xor_b32 s6, s18, s19 // (op 654) 

s_sub_i32 s18, s6, s19 // (op 654) 


s_mul_i32 s6, s7, s14 // (op 654) 
s_sub_u32 s19, s16, s6 // (op 654) 

s_mul_i32 s6, s18, s14 // (op 654) 
s_add_u32 s6, s6, s19 // (op 654) 

s_mul_hi_i32 s19, s16, s8 // (op 654) 
s_add_i32 s22, s19, s16 // (op 654) 

s_ashr_i32 s19, s22, 31 // (op 654) 
s_cmp_eq_i32 s8, 0 // (op 654) 
s_mov_b32 s23, scc // (op 654) 
s_cmp_eq_i32 s23, 0 // (op 654) 
s_cselect_b32 s24, 0, -1 // (op 654) 

s_and_b32 s23, s15, 31 // (op 654) 
s_lshl_b32 s25, 1, s23 // (op 654) 
s_add_i32 s26, s25, s24 // (op 654) 


s_and_b32 s24, s19, s26 // (op 654) 


s_add_i32 s19, s22, s24 // (op 654) 


s_ashr_i32 s22, s19, s23 // (op 654) 


s_lshl_b32 s19, s15, 24 // (op 654) 
s_ashr_i32 s23, s19, 31 // (op 654) 

s_xor_b32 s19, s22, s23 // (op 654) 

s_sub_i32 s22, s19, s23 // (op 654) 


s_add_u32 s38, s4, 32 // (op 654) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 654) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 654) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 654) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 5 // (op 654) 

s_mul_i32 s19, s14, s38 // (op 654) 
s_mul_i32 s23, s50, s19 // (op 654) 

s_mul_i32 s19, s22, s23 // (op 654) 

s_sub_u32 s23, s16, s19 // (op 654) 

s_mov_b32 s19, s23 // (op 654) 
s_mul_hi_i32 s24, s19, s13 // (op 654) 
s_add_i32 s25, s24, s19 // (op 654) 


s_ashr_i32 s19, s25, 31 // (op 654) 
s_cmp_eq_i32 s13, 0 // (op 654) 
s_mov_b32 s24, scc // (op 654) 
s_cmp_eq_i32 s24, 0 // (op 654) 
s_cselect_b32 s26, 0, -1 // (op 654) 

s_and_b32 s24, s49, 31 // (op 654) 
s_lshl_b32 s27, 1, s24 // (op 654) 
s_add_i32 s28, s27, s26 // (op 654) 


s_and_b32 s26, s19, s28 // (op 654) 


s_add_i32 s19, s25, s26 // (op 654) 


s_ashr_i32 s25, s19, s24 // (op 654) 


s_lshl_b32 s19, s49, 24 // (op 654) 
s_ashr_i32 s24, s19, 31 // (op 654) 

s_xor_b32 s19, s25, s24 // (op 654) 

s_sub_i32 s25, s19, s24 // (op 654) 


s_mul_i32 s19, s48, s14 // (op 654) 
s_add_u32 s26, s20, 32 // (op 654) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 654) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 654) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 654) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 654) 

s_sub_i32 s24, s26, s19 // (op 654) 

s_mul_i32 s26, s25, s24 // (op 654) 

s_sub_u32 s24, s23, s26 // (op 654) 


s_add_u32 s23, s24, s19 // (op 654) 


s_cmp_eq_i32 s22, 0 // (op 654) 
s_mov_b32 s19, scc // (op 654) 

s_cmp_eq_i32 s19, 0 // (op 654) 
s_cselect_b32 s22, s23, s6 // (op 654) 


s_mul_i32 s6, s18, s38 // (op 654) 


v_lshl_add_u32 v3, s22, 5, v4 // (op 654) 


s_sub_i32 s18, s7, s6 // (op 654) 


s_cmp_eq_i32 s19, 0 // (op 654) 
s_cselect_b32 s6, s25, s18 // (op 654) 



v_lshl_add_u32 v4, s6, 5, v0 // (op 654) 


v_mul_lo_u32 v0, v4, s44 // (op 654) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 654) 


s_mul_i32 s7, s44, s5 // (op 656) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 656) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 656) least significant
s_mul_hi_u32 s19, s44, s4 // (op 656) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 656) most significant: sum
s_add_u32 s7, s7, s19 // (op 656) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 656) 

s_mov_b32 s36, s46 // (op 656) reg expression
s_mov_b32 s37, s47 // (op 656) reg expression
s_mov_b32 s38, s18 // (op 656) reg expression
s_mov_b32 s39, 131072 // (op 656) call()





buffer_load_dwordx2 v[8:9], v4, s[36:39], 0 offen // (op 129) Load value



v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v4, 65535, v8 // (op 130) 
v_lshrrev_b32 v5, 16, v8 // (op 130) 
v_lshlrev_b32 v4, 16, v4 // (op 130) 
v_lshlrev_b32 v5, 16, v5 // (op 130) 
v_mul_f32 v10, v6, v4 // (op 130) 
v_mul_f32 v11, v6, v5 // (op 130) 
v_and_b32 v4, 65535, v9 // (op 130) 
v_lshrrev_b32 v5, 16, v9 // (op 130) 
v_lshlrev_b32 v4, 16, v4 // (op 130) 
v_lshlrev_b32 v5, 16, v5 // (op 130) 
v_mul_f32 v12, v6, v4 // (op 130) 
v_mul_f32 v13, v6, v5 // (op 130) 


v_add_f32 v18, v10, v14 // (op 126) 
v_add_f32 v19, v11, v15 // (op 126) 
v_add_f32 v20, v12, v16 // (op 126) 
v_add_f32 v21, v13, v17 // (op 126) 


v_lshrrev_b32 v8, 16, v18 // (op 127) 
v_lshrrev_b32 v9, 16, v19 // (op 127) 
v_lshrrev_b32 v10, 16, v20 // (op 127) 
v_lshrrev_b32 v11, 16, v21 // (op 127) 

s_mul_hi_i32 s6, s16, s9 // (op 796) 
v_lshrrev_b32 v0, 7, v1 // (op 796) 
s_add_i32 s7, s6, s16 // (op 796) 

v_and_b32 v3, 63, v1 // (op 796) 
s_ashr_i32 s6, s7, 31 // (op 796) 
v_lshrrev_b32 v4, 4, v3 // (op 796) 
s_cmp_eq_i32 s9, 0 // (op 796) 
s_mov_b32 s18, scc // (op 796) 
v_lshlrev_b32 v5, 2, v4 // (op 796) 

s_cmp_eq_i32 s18, 0 // (op 796) 
s_cselect_b32 s19, 0, -1 // (op 796) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 796) 


s_and_b32 s18, s10, 31 // (op 796) 
v_lshrrev_b32 v0, 2, v3 // (op 796) 

s_lshl_b32 s22, 1, s18 // (op 796) 
v_and_b32 v3, 3, v0 // (op 796) 

s_add_i32 s23, s22, s19 // (op 796) 


v_and_b32 v0, 3, v1 // (op 796) 
s_and_b32 s19, s6, s23 // (op 796) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 796) 


s_add_i32 s6, s7, s19 // (op 796) 


v_lshrrev_b32 v0, 6, v1 // (op 796) 
s_ashr_i32 s7, s6, s18 // (op 796) 


v_and_b32 v3, 1, v0 // (op 796) 

s_lshl_b32 s6, s10, 24 // (op 796) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 796) 


s_ashr_i32 s18, s6, 31 // (op 796) 

s_xor_b32 s6, s7, s18 // (op 796) 

s_sub_i32 s7, s6, s18 // (op 796) 


s_mul_hi_i32 s6, s7, s11 // (op 796) 
s_add_i32 s18, s6, s7 // (op 796) 

s_ashr_i32 s6, s18, 31 // (op 796) 
s_cmp_eq_i32 s11, 0 // (op 796) 
s_mov_b32 s19, scc // (op 796) 
s_cmp_eq_i32 s19, 0 // (op 796) 
s_cselect_b32 s22, 0, -1 // (op 796) 

s_and_b32 s19, s12, 31 // (op 796) 
s_lshl_b32 s23, 1, s19 // (op 796) 
s_add_i32 s24, s23, s22 // (op 796) 


s_and_b32 s22, s6, s24 // (op 796) 


s_add_i32 s6, s18, s22 // (op 796) 


s_ashr_i32 s18, s6, s19 // (op 796) 


s_lshl_b32 s6, s12, 24 // (op 796) 
s_ashr_i32 s19, s6, 31 // (op 796) 

s_xor_b32 s6, s18, s19 // (op 796) 

s_sub_i32 s18, s6, s19 // (op 796) 


s_mul_i32 s6, s7, s14 // (op 796) 
s_sub_u32 s19, s16, s6 // (op 796) 

s_mul_i32 s6, s18, s14 // (op 796) 
s_add_u32 s6, s6, s19 // (op 796) 

s_mul_hi_i32 s19, s16, s8 // (op 796) 
s_add_i32 s22, s19, s16 // (op 796) 

s_ashr_i32 s19, s22, 31 // (op 796) 
s_cmp_eq_i32 s8, 0 // (op 796) 
s_mov_b32 s23, scc // (op 796) 
s_cmp_eq_i32 s23, 0 // (op 796) 
s_cselect_b32 s24, 0, -1 // (op 796) 

s_and_b32 s23, s15, 31 // (op 796) 
s_lshl_b32 s25, 1, s23 // (op 796) 
s_add_i32 s26, s25, s24 // (op 796) 


s_and_b32 s24, s19, s26 // (op 796) 


s_add_i32 s19, s22, s24 // (op 796) 


s_ashr_i32 s22, s19, s23 // (op 796) 


s_lshl_b32 s19, s15, 24 // (op 796) 
s_ashr_i32 s23, s19, 31 // (op 796) 

s_xor_b32 s19, s22, s23 // (op 796) 

s_sub_i32 s22, s19, s23 // (op 796) 


s_add_u32 s38, s4, 32 // (op 796) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 796) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 796) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 796) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 5 // (op 796) 

s_mul_i32 s19, s14, s38 // (op 796) 
s_mul_i32 s23, s50, s19 // (op 796) 

s_mul_i32 s19, s22, s23 // (op 796) 

s_sub_u32 s23, s16, s19 // (op 796) 

s_mov_b32 s19, s23 // (op 796) 
s_mul_hi_i32 s24, s19, s13 // (op 796) 
s_add_i32 s25, s24, s19 // (op 796) 


s_ashr_i32 s19, s25, 31 // (op 796) 
s_cmp_eq_i32 s13, 0 // (op 796) 
s_mov_b32 s24, scc // (op 796) 
s_cmp_eq_i32 s24, 0 // (op 796) 
s_cselect_b32 s26, 0, -1 // (op 796) 

s_and_b32 s24, s49, 31 // (op 796) 
s_lshl_b32 s27, 1, s24 // (op 796) 
s_add_i32 s28, s27, s26 // (op 796) 


s_and_b32 s26, s19, s28 // (op 796) 


s_add_i32 s19, s25, s26 // (op 796) 


s_ashr_i32 s25, s19, s24 // (op 796) 


s_lshl_b32 s19, s49, 24 // (op 796) 
s_ashr_i32 s24, s19, 31 // (op 796) 

s_xor_b32 s19, s25, s24 // (op 796) 

s_sub_i32 s25, s19, s24 // (op 796) 


s_mul_i32 s19, s48, s14 // (op 796) 
s_add_u32 s26, s20, 32 // (op 796) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 796) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 796) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 796) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 796) 

s_sub_i32 s24, s26, s19 // (op 796) 

s_mul_i32 s26, s25, s24 // (op 796) 

s_sub_u32 s24, s23, s26 // (op 796) 


s_add_u32 s23, s24, s19 // (op 796) 


s_cmp_eq_i32 s22, 0 // (op 796) 
s_mov_b32 s19, scc // (op 796) 

s_cmp_eq_i32 s19, 0 // (op 796) 
s_cselect_b32 s22, s23, s6 // (op 796) 


s_mul_i32 s6, s18, s38 // (op 796) 


v_lshl_add_u32 v3, s22, 5, v4 // (op 796) 


s_sub_i32 s18, s7, s6 // (op 796) 


s_cmp_eq_i32 s19, 0 // (op 796) 
s_cselect_b32 s6, s25, s18 // (op 796) 



v_lshl_add_u32 v4, s6, 5, v0 // (op 796) 


v_mul_lo_u32 v0, v4, s40 // (op 796) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 796) 


s_mul_i32 s7, s40, s5 // (op 798) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 798) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 798) least significant
s_mul_hi_u32 s19, s40, s4 // (op 798) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 798) most significant: sum
s_add_u32 s7, s7, s19 // (op 798) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 798) 

s_mov_b32 s0, s42 // (op 798) reg expression
s_mov_b32 s1, s43 // (op 798) reg expression
s_mov_b32 s2, s18 // (op 798) reg expression
s_mov_b32 s3, 131072 // (op 798) call()


v_pack_b32_f16 v12, v8, v9 // (op 128) 
v_pack_b32_f16 v13, v10, v11 // (op 128) 


buffer_store_dwordx2 v[12:13], v4, s[0:3], 0 offen // (op 128) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__ConditionalBottom_0_beta_116:
















s_endpgm  // End of RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.LRR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__end:
.size RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_, .LRR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM__end-RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: F6
    DataTypeA: F6
    DataTypeB: F6
    DestDataType: b
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
  - .name: RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP6_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 50
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

