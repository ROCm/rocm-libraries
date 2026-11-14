// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_,@function
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 48, v1 // (op 765) 
v_and_b32 v3, 63, v1 // (op 765) 
v_sub_u32 v4, v3, v0 // (op 765) 

v_lshrrev_b32 v3, 1, v4 // (op 765) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 765) 


v_and_b32 v0, 1, v4 // (op 765) 

v_lshlrev_b32 v3, 3, v0 // (op 765) 

v_lshrrev_b32 v0, 6, v1 // (op 765) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 765) 


v_add_u32 v0, v5, v4 // (op 765) 


v_and_b32 v3, 63, v1 // (op 818) 
v_and_b32 v4, 48, v1 // (op 818) 
v_sub_u32 v5, v3, v4 // (op 818) 

v_lshrrev_b32 v4, 4, v3 // (op 818) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 818) 


v_lshlrev_b32 v4, 4, v3 // (op 818) 
v_mul_lo_u32 v5, v4, 6 // (op 818) 

v_lshrrev_b32 v4, 3, v5 // (op 818) 

v_and_b32 v5, 268435455, v3 // (op 818) 

v_lshlrev_b32 v3, 2, v5 // (op 818) 

v_add_u32 v5, v4, v3 // (op 818) 


v_lshrrev_b32 v3, 6, v1 // (op 871) 
v_and_b32 v4, 15, v1 // (op 871) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 871) 


v_and_b32 v3, 63, v1 // (op 871) 
v_lshrrev_b32 v4, 4, v3 // (op 871) 

v_lshl_add_u32 v3, v4, 5, v6 // (op 871) 


v_and_b32 v4, 48, v1 // (op 944) 
v_and_b32 v6, 63, v1 // (op 944) 
v_sub_u32 v7, v6, v4 // (op 944) 

v_lshrrev_b32 v6, 1, v7 // (op 944) 
v_add_lshl_u32 v8, v4, v6, 5 // (op 944) 


v_and_b32 v4, 1, v7 // (op 944) 

v_lshlrev_b32 v6, 3, v4 // (op 944) 

v_lshrrev_b32 v4, 6, v1 // (op 944) 
v_lshl_add_u32 v7, v4, 4, v6 // (op 944) 


v_add_u32 v4, v8, v7 // (op 944) 


v_and_b32 v6, 63, v1 // (op 968) 
v_and_b32 v7, 48, v1 // (op 968) 
v_sub_u32 v8, v6, v7 // (op 968) 

v_lshrrev_b32 v7, 4, v6 // (op 968) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 968) 


v_lshlrev_b32 v7, 4, v6 // (op 968) 
v_mul_lo_u32 v8, v7, 6 // (op 968) 

v_lshrrev_b32 v7, 3, v8 // (op 968) 

v_and_b32 v8, 268435455, v6 // (op 968) 

v_lshlrev_b32 v6, 2, v8 // (op 968) 

v_add_u32 v8, v7, v6 // (op 968) 


v_lshrrev_b32 v6, 6, v1 // (op 992) 
v_and_b32 v7, 15, v1 // (op 992) 
v_lshl_add_u32 v9, v6, 4, v7 // (op 992) 


v_and_b32 v6, 63, v1 // (op 992) 
v_lshrrev_b32 v7, 4, v6 // (op 992) 

v_lshl_add_u32 v6, v7, 5, v9 // (op 992) 


v_mov_b32 v7, s6 // (op 2) Move value
v_mov_b32 v9, s7 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
s_mul_hi_i32 s0, s16, s9 // (op 476) 
v_and_b32 v10, -2, v1 // (op 476) 
s_add_i32 s1, s0, s16 // (op 476) 

v_sub_u32 v11, v1, v10 // (op 476) 

s_ashr_i32 s0, s1, 31 // (op 476) 
v_lshlrev_b32 v10, 4, v11 // (op 476) 

s_cmp_eq_i32 s9, 0 // (op 476) 
s_mov_b32 s6, scc // (op 476) 
v_lshrrev_b32 v11, 1, v1 // (op 476) 
s_cmp_eq_i32 s6, 0 // (op 476) 
s_cselect_b32 s7, 0, -1 // (op 476) 

s_and_b32 s6, s10, 31 // (op 476) 
s_lshl_b32 s18, 1, s6 // (op 476) 
s_add_i32 s19, s18, s7 // (op 476) 


s_and_b32 s7, s0, s19 // (op 476) 


s_add_i32 s0, s1, s7 // (op 476) 


s_ashr_i32 s1, s0, s6 // (op 476) 


s_lshl_b32 s0, s10, 24 // (op 476) 
s_ashr_i32 s6, s0, 31 // (op 476) 

s_xor_b32 s0, s1, s6 // (op 476) 

s_sub_i32 s1, s0, s6 // (op 476) 


s_mul_hi_i32 s0, s1, s11 // (op 476) 
s_add_i32 s6, s0, s1 // (op 476) 

s_ashr_i32 s0, s6, 31 // (op 476) 
s_cmp_eq_i32 s11, 0 // (op 476) 
s_mov_b32 s7, scc // (op 476) 
s_cmp_eq_i32 s7, 0 // (op 476) 
s_cselect_b32 s18, 0, -1 // (op 476) 

s_and_b32 s7, s12, 31 // (op 476) 
s_lshl_b32 s19, 1, s7 // (op 476) 
s_add_i32 s51, s19, s18 // (op 476) 


s_and_b32 s18, s0, s51 // (op 476) 


s_add_i32 s0, s6, s18 // (op 476) 


s_ashr_i32 s6, s0, s7 // (op 476) 


s_lshl_b32 s0, s12, 24 // (op 476) 
s_ashr_i32 s7, s0, 31 // (op 476) 

s_xor_b32 s0, s6, s7 // (op 476) 

s_sub_i32 s6, s0, s7 // (op 476) 


s_mul_i32 s0, s1, s14 // (op 476) 

s_sub_u32 s1, s16, s0 // (op 476) 

s_mul_i32 s0, s6, s14 // (op 476) 
s_add_u32 s0, s0, s1 // (op 476) 


s_mul_hi_i32 s1, s16, s8 // (op 476) 
s_add_i32 s6, s1, s16 // (op 476) 

s_ashr_i32 s1, s6, 31 // (op 476) 
s_cmp_eq_i32 s8, 0 // (op 476) 
s_mov_b32 s7, scc // (op 476) 
s_cmp_eq_i32 s7, 0 // (op 476) 
s_cselect_b32 s18, 0, -1 // (op 476) 

s_and_b32 s7, s15, 31 // (op 476) 
s_lshl_b32 s19, 1, s7 // (op 476) 
s_add_i32 s51, s19, s18 // (op 476) 


s_and_b32 s18, s1, s51 // (op 476) 


s_add_i32 s1, s6, s18 // (op 476) 


s_ashr_i32 s6, s1, s7 // (op 476) 


s_lshl_b32 s1, s15, 24 // (op 476) 
s_ashr_i32 s7, s1, 31 // (op 476) 

s_xor_b32 s1, s6, s7 // (op 476) 

s_sub_i32 s6, s1, s7 // (op 476) 


s_add_u32 s18, s4, 16 // (op 476) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 476) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 476) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 476) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 4 // (op 476) 

s_mul_i32 s1, s14, s18 // (op 476) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 476) 

s_mul_i32 s1, s6, s7 // (op 476) 

s_sub_u32 s7, s16, s1 // (op 476) 

s_mov_b32 s1, s7 // (op 476) 
s_mul_hi_i32 s18, s1, s13 // (op 476) 
s_add_i32 s19, s18, s1 // (op 476) 


s_ashr_i32 s1, s19, 31 // (op 476) 
s_cmp_eq_i32 s13, 0 // (op 476) 
s_mov_b32 s18, scc // (op 476) 
s_cmp_eq_i32 s18, 0 // (op 476) 
s_cselect_b32 s51, 0, -1 // (op 476) 

s_and_b32 s18, s49, 31 // (op 476) 
s_lshl_b32 s52, 1, s18 // (op 476) 
s_add_i32 s53, s52, s51 // (op 476) 


s_and_b32 s51, s1, s53 // (op 476) 


s_add_i32 s1, s19, s51 // (op 476) 


s_ashr_i32 s19, s1, s18 // (op 476) 


s_lshl_b32 s1, s49, 24 // (op 476) 
s_ashr_i32 s18, s1, 31 // (op 476) 

s_xor_b32 s1, s19, s18 // (op 476) 

s_sub_i32 s19, s1, s18 // (op 476) 


s_mul_i32 s1, s48, s14 // (op 476) 
s_add_u32 s52, s20, 32 // (op 476) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 476) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 476) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 476) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 5 // (op 476) 

s_sub_i32 s18, s52, s1 // (op 476) 

s_mul_i32 s51, s19, s18 // (op 476) 


s_sub_u32 s18, s7, s51 // (op 476) 


s_add_u32 s7, s18, s1 // (op 476) 


s_cmp_eq_i32 s6, 0 // (op 476) 
s_mov_b32 s1, scc // (op 476) 

s_cmp_eq_i32 s1, 0 // (op 476) 
s_cselect_b32 s6, s7, s0 // (op 476) 



v_lshl_add_u32 v12, s6, 5, v10 // (op 476) 


v_mov_b32 v15, 0 // (op 476) convert
v_mov_b32 v14, v12 // (op 476) convert

v_mul_lo_u32 v13, v11, s3 // (op 476) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 476) most significant: high of low * low
v_mul_lo_u32 v12, v11, s2 // (op 476) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 476) most significant: sum


v_add_co_u32 v10, s[0:1], v14, v12 // (op 476) least significant half
v_addc_co_u32 v11, s[0:1], v15, v13, s[0:1] // (op 476) most significant half



s_mul_i32 s1, s2, s25 // (op 478) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 478) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 478) least significant
s_mul_hi_u32 s7, s2, s24 // (op 478) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 478) most significant: sum
s_add_u32 s1, s1, s7 // (op 478) most significant: sum


s_mov_b32 s52, s22 // (op 478) reg expression
s_mov_b32 s53, s23 // (op 478) reg expression
s_mov_b32 s54, s0 // (op 478) reg expression
s_mov_b32 s55, 131072 // (op 478) call()

s_mul_hi_i32 s51, s16, s9 // (op 495) 
v_and_b32 v12, -2, v1 // (op 495) 
s_add_i32 s0, s51, s16 // (op 495) 

v_sub_u32 v13, v1, v12 // (op 495) 

s_ashr_i32 s1, s0, 31 // (op 495) 
v_lshlrev_b32 v12, 4, v13 // (op 495) 

s_cmp_eq_i32 s9, 0 // (op 495) 
s_mov_b32 s51, scc // (op 495) 
v_lshrrev_b32 v13, 1, v1 // (op 495) 
s_cmp_eq_i32 s51, 0 // (op 495) 
s_cselect_b32 s6, 0, -1 // (op 495) 

s_and_b32 s7, s10, 31 // (op 495) 
s_lshl_b32 s51, 1, s7 // (op 495) 
s_add_i32 s18, s51, s6 // (op 495) 


s_and_b32 s6, s1, s18 // (op 495) 


s_add_i32 s1, s0, s6 // (op 495) 


s_ashr_i32 s0, s1, s7 // (op 495) 


s_lshl_b32 s1, s10, 24 // (op 495) 
s_ashr_i32 s51, s1, 31 // (op 495) 

s_xor_b32 s1, s0, s51 // (op 495) 

s_sub_i32 s0, s1, s51 // (op 495) 


s_mul_hi_i32 s1, s0, s11 // (op 495) 
s_add_i32 s51, s1, s0 // (op 495) 

s_ashr_i32 s1, s51, 31 // (op 495) 
s_cmp_eq_i32 s11, 0 // (op 495) 
s_mov_b32 s6, scc // (op 495) 
s_cmp_eq_i32 s6, 0 // (op 495) 
s_cselect_b32 s7, 0, -1 // (op 495) 

s_and_b32 s6, s12, 31 // (op 495) 
s_lshl_b32 s18, 1, s6 // (op 495) 
s_add_i32 s19, s18, s7 // (op 495) 


s_and_b32 s7, s1, s19 // (op 495) 


s_add_i32 s1, s51, s7 // (op 495) 


s_ashr_i32 s7, s1, s6 // (op 495) 


s_lshl_b32 s1, s12, 24 // (op 495) 
s_ashr_i32 s6, s1, 31 // (op 495) 

s_xor_b32 s1, s7, s6 // (op 495) 

s_sub_i32 s7, s1, s6 // (op 495) 


s_mul_i32 s1, s0, s14 // (op 495) 

s_sub_u32 s0, s16, s1 // (op 495) 

s_mul_i32 s1, s7, s14 // (op 495) 
s_add_u32 s1, s1, s0 // (op 495) 


s_mul_hi_i32 s0, s16, s8 // (op 495) 
s_add_i32 s51, s0, s16 // (op 495) 

s_ashr_i32 s0, s51, 31 // (op 495) 
s_cmp_eq_i32 s8, 0 // (op 495) 
s_mov_b32 s6, scc // (op 495) 
s_cmp_eq_i32 s6, 0 // (op 495) 
s_cselect_b32 s7, 0, -1 // (op 495) 

s_and_b32 s6, s15, 31 // (op 495) 
s_lshl_b32 s18, 1, s6 // (op 495) 
s_add_i32 s19, s18, s7 // (op 495) 


s_and_b32 s7, s0, s19 // (op 495) 


s_add_i32 s0, s51, s7 // (op 495) 


s_ashr_i32 s7, s0, s6 // (op 495) 


s_lshl_b32 s0, s15, 24 // (op 495) 
s_ashr_i32 s6, s0, 31 // (op 495) 

s_xor_b32 s0, s7, s6 // (op 495) 

s_sub_i32 s7, s0, s6 // (op 495) 


s_add_u32 s18, s4, 16 // (op 495) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 495) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 495) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 495) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 495) 

s_mul_i32 s0, s14, s18 // (op 495) 

s_mul_i32 s6, s50, s0 // (op 495) 

s_mul_i32 s0, s7, s6 // (op 495) 

s_sub_u32 s6, s16, s0 // (op 495) 

s_mov_b32 s0, s6 // (op 495) 
s_mul_hi_i32 s51, s0, s13 // (op 495) 
s_add_i32 s18, s51, s0 // (op 495) 


s_ashr_i32 s0, s18, 31 // (op 495) 
s_cmp_eq_i32 s13, 0 // (op 495) 
s_mov_b32 s19, scc // (op 495) 
s_cmp_eq_i32 s19, 0 // (op 495) 
s_cselect_b32 s51, 0, -1 // (op 495) 

s_and_b32 s19, s49, 31 // (op 495) 
s_lshl_b32 s56, 1, s19 // (op 495) 
s_add_i32 s57, s56, s51 // (op 495) 


s_and_b32 s51, s0, s57 // (op 495) 


s_add_i32 s0, s18, s51 // (op 495) 


s_ashr_i32 s18, s0, s19 // (op 495) 


s_lshl_b32 s0, s49, 24 // (op 495) 
s_ashr_i32 s19, s0, 31 // (op 495) 

s_xor_b32 s0, s18, s19 // (op 495) 

s_sub_i32 s18, s0, s19 // (op 495) 


s_mul_i32 s0, s48, s14 // (op 495) 
s_add_u32 s56, s20, 32 // (op 495) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 495) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 495) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 495) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 495) 

s_sub_i32 s19, s56, s0 // (op 495) 

s_mul_i32 s51, s18, s19 // (op 495) 


s_sub_u32 s18, s6, s51 // (op 495) 


s_add_u32 s6, s18, s0 // (op 495) 


s_cmp_eq_i32 s7, 0 // (op 495) 
s_mov_b32 s0, scc // (op 495) 

s_cmp_eq_i32 s0, 0 // (op 495) 
s_cselect_b32 s7, s6, s1 // (op 495) 



v_lshl_add_u32 v14, s7, 5, v12 // (op 495) 


v_mov_b32 v17, 0 // (op 495) convert
v_mov_b32 v16, v14 // (op 495) convert

v_mul_lo_u32 v15, v13, s3 // (op 495) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 495) most significant: high of low * low
v_mul_lo_u32 v14, v13, s2 // (op 495) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 495) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v14 // (op 495) least significant half
v_addc_co_u32 v13, s[0:1], v17, v15, s[0:1] // (op 495) most significant half



s_mul_i32 s1, s2, s25 // (op 497) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 497) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 497) least significant
s_mul_hi_u32 s6, s2, s24 // (op 497) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 497) most significant: sum
s_add_u32 s1, s1, s6 // (op 497) most significant: sum


s_mov_b32 s52, s22 // (op 497) reg expression
s_mov_b32 s53, s23 // (op 497) reg expression
s_mov_b32 s54, s0 // (op 497) reg expression
s_mov_b32 s55, 131072 // (op 497) call()

s_mul_hi_i32 s51, s16, s9 // (op 540) 
s_add_i32 s0, s51, s16 // (op 540) 

s_ashr_i32 s1, s0, 31 // (op 540) 
s_cmp_eq_i32 s9, 0 // (op 540) 
s_mov_b32 s51, scc // (op 540) 
s_cmp_eq_i32 s51, 0 // (op 540) 
s_cselect_b32 s6, 0, -1 // (op 540) 

s_and_b32 s7, s10, 31 // (op 540) 
s_lshl_b32 s51, 1, s7 // (op 540) 
s_add_i32 s18, s51, s6 // (op 540) 


s_and_b32 s6, s1, s18 // (op 540) 


s_add_i32 s1, s0, s6 // (op 540) 


s_ashr_i32 s0, s1, s7 // (op 540) 


s_lshl_b32 s1, s10, 24 // (op 540) 
s_ashr_i32 s51, s1, 31 // (op 540) 

s_xor_b32 s1, s0, s51 // (op 540) 

s_sub_i32 s0, s1, s51 // (op 540) 


s_mul_hi_i32 s1, s0, s11 // (op 540) 
s_add_i32 s51, s1, s0 // (op 540) 

s_ashr_i32 s1, s51, 31 // (op 540) 
s_cmp_eq_i32 s11, 0 // (op 540) 
s_mov_b32 s6, scc // (op 540) 
s_cmp_eq_i32 s6, 0 // (op 540) 
s_cselect_b32 s7, 0, -1 // (op 540) 

s_and_b32 s6, s12, 31 // (op 540) 
s_lshl_b32 s18, 1, s6 // (op 540) 
s_add_i32 s19, s18, s7 // (op 540) 


s_and_b32 s7, s1, s19 // (op 540) 


s_add_i32 s1, s51, s7 // (op 540) 


s_ashr_i32 s7, s1, s6 // (op 540) 


s_lshl_b32 s1, s12, 24 // (op 540) 
s_ashr_i32 s6, s1, 31 // (op 540) 

s_xor_b32 s1, s7, s6 // (op 540) 

s_sub_i32 s7, s1, s6 // (op 540) 


s_add_u32 s18, s4, 16 // (op 540) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 540) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 540) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 540) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 4 // (op 540) 

s_mul_i32 s1, s7, s18 // (op 540) 

s_sub_i32 s51, s0, s1 // (op 540) 


s_mul_hi_i32 s0, s16, s8 // (op 540) 
s_add_i32 s1, s0, s16 // (op 540) 

s_ashr_i32 s0, s1, 31 // (op 540) 
s_cmp_eq_i32 s8, 0 // (op 540) 
s_mov_b32 s6, scc // (op 540) 
s_cmp_eq_i32 s6, 0 // (op 540) 
s_cselect_b32 s7, 0, -1 // (op 540) 

s_and_b32 s6, s15, 31 // (op 540) 
s_lshl_b32 s56, 1, s6 // (op 540) 
s_add_i32 s57, s56, s7 // (op 540) 


s_and_b32 s7, s0, s57 // (op 540) 


s_add_i32 s0, s1, s7 // (op 540) 


s_ashr_i32 s1, s0, s6 // (op 540) 


s_lshl_b32 s0, s15, 24 // (op 540) 
s_ashr_i32 s6, s0, 31 // (op 540) 

s_xor_b32 s0, s1, s6 // (op 540) 

s_sub_i32 s1, s0, s6 // (op 540) 


s_mul_i32 s0, s14, s18 // (op 540) 

s_mul_i32 s6, s50, s0 // (op 540) 

s_mul_i32 s0, s1, s6 // (op 540) 

s_sub_u32 s6, s16, s0 // (op 540) 

s_mul_hi_i32 s0, s6, s13 // (op 540) 
s_add_i32 s7, s0, s6 // (op 540) 


s_ashr_i32 s0, s7, 31 // (op 540) 
s_cmp_eq_i32 s13, 0 // (op 540) 
s_mov_b32 s6, scc // (op 540) 
s_cmp_eq_i32 s6, 0 // (op 540) 
s_cselect_b32 s18, 0, -1 // (op 540) 

s_and_b32 s6, s49, 31 // (op 540) 
s_lshl_b32 s19, 1, s6 // (op 540) 
s_add_i32 s56, s19, s18 // (op 540) 


s_and_b32 s18, s0, s56 // (op 540) 


s_add_i32 s0, s7, s18 // (op 540) 


s_ashr_i32 s7, s0, s6 // (op 540) 


s_lshl_b32 s0, s49, 24 // (op 540) 
s_ashr_i32 s6, s0, 31 // (op 540) 

s_xor_b32 s0, s7, s6 // (op 540) 

s_sub_i32 s7, s0, s6 // (op 540) 


s_cmp_eq_i32 s1, 0 // (op 540) 
s_mov_b32 s0, scc // (op 540) 

s_cmp_eq_i32 s0, 0 // (op 540) 
s_cselect_b32 s1, s7, s51 // (op 540) 



s_lshl_b32 s0, s1, 4 // (op 540) 

s_mov_b32 s7, 0 // (op 540) convert
s_mov_b32 s6, s0 // (op 540) convert

v_mul_lo_u32 v15, v1, s27 // (op 540) most significant: low of low * high
v_mul_hi_u32 v16, v1, s26 // (op 540) most significant: high of low * low
v_mul_lo_u32 v14, v1, s26 // (op 540) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 540) most significant: sum

v_mov_b32 v16, s6 // (op 540) 
v_mov_b32 v17, s7 // (op 540) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 540) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 540) most significant half




v_mul_lo_u32 v17, v15, 6 // (op 540) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 540) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 540) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 540) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 540) 

v_mov_b32 v16, v14 // (op 540) convert

s_mul_i32 s1, s25, 6 // (op 542) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 542) least significant
s_mul_hi_u32 s51, s24, 6 // (op 542) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 542) most significant: sum

s_mul_i32 s7, s26, s1 // (op 542) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 542) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 542) least significant
s_mul_hi_u32 s18, s26, s0 // (op 542) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 542) most significant: sum
s_add_u32 s7, s7, s18 // (op 542) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 542) 

s_mov_b32 s56, s28 // (op 542) reg expression
s_mov_b32 s57, s29 // (op 542) reg expression
s_mov_b32 s58, s0 // (op 542) reg expression
s_mov_b32 s59, 131072 // (op 542) call()

s_mul_hi_i32 s51, s16, s9 // (op 559) 
s_add_i32 s0, s51, s16 // (op 559) 

s_ashr_i32 s1, s0, 31 // (op 559) 
s_cmp_eq_i32 s9, 0 // (op 559) 
s_mov_b32 s51, scc // (op 559) 
s_cmp_eq_i32 s51, 0 // (op 559) 
s_cselect_b32 s6, 0, -1 // (op 559) 

s_and_b32 s7, s10, 31 // (op 559) 
s_lshl_b32 s51, 1, s7 // (op 559) 
s_add_i32 s18, s51, s6 // (op 559) 


s_and_b32 s6, s1, s18 // (op 559) 


s_add_i32 s1, s0, s6 // (op 559) 


s_ashr_i32 s0, s1, s7 // (op 559) 


s_lshl_b32 s1, s10, 24 // (op 559) 
s_ashr_i32 s51, s1, 31 // (op 559) 

s_xor_b32 s1, s0, s51 // (op 559) 

s_sub_i32 s0, s1, s51 // (op 559) 


s_mul_hi_i32 s1, s0, s11 // (op 559) 
s_add_i32 s51, s1, s0 // (op 559) 

s_ashr_i32 s1, s51, 31 // (op 559) 
s_cmp_eq_i32 s11, 0 // (op 559) 
s_mov_b32 s6, scc // (op 559) 
s_cmp_eq_i32 s6, 0 // (op 559) 
s_cselect_b32 s7, 0, -1 // (op 559) 

s_and_b32 s6, s12, 31 // (op 559) 
s_lshl_b32 s18, 1, s6 // (op 559) 
s_add_i32 s19, s18, s7 // (op 559) 


s_and_b32 s7, s1, s19 // (op 559) 


s_add_i32 s1, s51, s7 // (op 559) 


s_ashr_i32 s7, s1, s6 // (op 559) 


s_lshl_b32 s1, s12, 24 // (op 559) 
s_ashr_i32 s6, s1, 31 // (op 559) 

s_xor_b32 s1, s7, s6 // (op 559) 

s_sub_i32 s7, s1, s6 // (op 559) 


s_add_u32 s18, s4, 16 // (op 559) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 559) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 559) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 559) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 559) 

s_mul_i32 s1, s7, s18 // (op 559) 

s_sub_i32 s51, s0, s1 // (op 559) 


s_mul_hi_i32 s0, s16, s8 // (op 559) 
s_add_i32 s1, s0, s16 // (op 559) 

s_ashr_i32 s0, s1, 31 // (op 559) 
s_cmp_eq_i32 s8, 0 // (op 559) 
s_mov_b32 s6, scc // (op 559) 
s_cmp_eq_i32 s6, 0 // (op 559) 
s_cselect_b32 s7, 0, -1 // (op 559) 

s_and_b32 s6, s15, 31 // (op 559) 
s_lshl_b32 s60, 1, s6 // (op 559) 
s_add_i32 s61, s60, s7 // (op 559) 


s_and_b32 s7, s0, s61 // (op 559) 


s_add_i32 s0, s1, s7 // (op 559) 


s_ashr_i32 s1, s0, s6 // (op 559) 


s_lshl_b32 s0, s15, 24 // (op 559) 
s_ashr_i32 s6, s0, 31 // (op 559) 

s_xor_b32 s0, s1, s6 // (op 559) 

s_sub_i32 s1, s0, s6 // (op 559) 


s_mul_i32 s0, s14, s18 // (op 559) 

s_mul_i32 s6, s50, s0 // (op 559) 

s_mul_i32 s0, s1, s6 // (op 559) 

s_sub_u32 s6, s16, s0 // (op 559) 

s_mul_hi_i32 s0, s6, s13 // (op 559) 
s_add_i32 s7, s0, s6 // (op 559) 


s_ashr_i32 s0, s7, 31 // (op 559) 
s_cmp_eq_i32 s13, 0 // (op 559) 
s_mov_b32 s6, scc // (op 559) 
s_cmp_eq_i32 s6, 0 // (op 559) 
s_cselect_b32 s18, 0, -1 // (op 559) 

s_and_b32 s6, s49, 31 // (op 559) 
s_lshl_b32 s19, 1, s6 // (op 559) 
s_add_i32 s60, s19, s18 // (op 559) 


s_and_b32 s18, s0, s60 // (op 559) 


s_add_i32 s0, s7, s18 // (op 559) 


s_ashr_i32 s7, s0, s6 // (op 559) 


s_lshl_b32 s0, s49, 24 // (op 559) 
s_ashr_i32 s6, s0, 31 // (op 559) 

s_xor_b32 s0, s7, s6 // (op 559) 

s_sub_i32 s7, s0, s6 // (op 559) 


s_cmp_eq_i32 s1, 0 // (op 559) 
s_mov_b32 s0, scc // (op 559) 

s_cmp_eq_i32 s0, 0 // (op 559) 
s_cselect_b32 s1, s7, s51 // (op 559) 



s_lshl_b32 s0, s1, 4 // (op 559) 

s_mov_b32 s7, 0 // (op 559) convert
s_mov_b32 s6, s0 // (op 559) convert

v_mul_lo_u32 v15, v1, s27 // (op 559) most significant: low of low * high
v_mul_hi_u32 v17, v1, s26 // (op 559) most significant: high of low * low
v_mul_lo_u32 v14, v1, s26 // (op 559) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 559) most significant: sum

v_mov_b32 v17, s6 // (op 559) 
v_mov_b32 v18, s7 // (op 559) 
v_add_co_u32 v14, s[0:1], v14, v17 // (op 559) least significant half
v_addc_co_u32 v15, s[0:1], v15, v18, s[0:1] // (op 559) most significant half




v_mul_lo_u32 v19, v15, 6 // (op 559) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 559) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 559) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 559) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 559) 

v_mov_b32 v17, v14 // (op 559) convert

s_mul_i32 s1, s25, 6 // (op 561) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 561) least significant
s_mul_hi_u32 s51, s24, 6 // (op 561) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 561) most significant: sum

s_mul_i32 s7, s26, s1 // (op 561) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 561) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 561) least significant
s_mul_hi_u32 s18, s26, s0 // (op 561) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 561) most significant: sum
s_add_u32 s7, s7, s18 // (op 561) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 561) 

s_mov_b32 s56, s28 // (op 561) reg expression
s_mov_b32 s57, s29 // (op 561) reg expression
s_mov_b32 s58, s0 // (op 561) reg expression
s_mov_b32 s59, 131072 // (op 561) call()

s_mul_hi_i32 s51, s16, s9 // (op 604) 
v_and_b32 v14, -32, v1 // (op 604) 
s_add_i32 s0, s51, s16 // (op 604) 

v_sub_u32 v15, v1, v14 // (op 604) 

s_ashr_i32 s1, s0, 31 // (op 604) 
v_lshrrev_b32 v14, 5, v1 // (op 604) 
s_cmp_eq_i32 s9, 0 // (op 604) 
s_mov_b32 s51, scc // (op 604) 
s_cmp_eq_i32 s51, 0 // (op 604) 
s_cselect_b32 s6, 0, -1 // (op 604) 

s_and_b32 s7, s10, 31 // (op 604) 
s_lshl_b32 s51, 1, s7 // (op 604) 
s_add_i32 s18, s51, s6 // (op 604) 


s_and_b32 s6, s1, s18 // (op 604) 


s_add_i32 s1, s0, s6 // (op 604) 


s_ashr_i32 s0, s1, s7 // (op 604) 


s_lshl_b32 s1, s10, 24 // (op 604) 
s_ashr_i32 s51, s1, 31 // (op 604) 

s_xor_b32 s1, s0, s51 // (op 604) 

s_sub_i32 s0, s1, s51 // (op 604) 


s_mul_hi_i32 s1, s0, s11 // (op 604) 
s_add_i32 s51, s1, s0 // (op 604) 

s_ashr_i32 s1, s51, 31 // (op 604) 
s_cmp_eq_i32 s11, 0 // (op 604) 
s_mov_b32 s6, scc // (op 604) 
s_cmp_eq_i32 s6, 0 // (op 604) 
s_cselect_b32 s7, 0, -1 // (op 604) 

s_and_b32 s6, s12, 31 // (op 604) 
s_lshl_b32 s18, 1, s6 // (op 604) 
s_add_i32 s19, s18, s7 // (op 604) 


s_and_b32 s7, s1, s19 // (op 604) 


s_add_i32 s1, s51, s7 // (op 604) 


s_ashr_i32 s7, s1, s6 // (op 604) 


s_lshl_b32 s1, s12, 24 // (op 604) 
s_ashr_i32 s6, s1, 31 // (op 604) 

s_xor_b32 s1, s7, s6 // (op 604) 

s_sub_i32 s7, s1, s6 // (op 604) 


s_mul_i32 s1, s0, s14 // (op 604) 

s_sub_u32 s0, s16, s1 // (op 604) 

s_mul_i32 s1, s7, s14 // (op 604) 
s_add_u32 s1, s1, s0 // (op 604) 


s_mul_hi_i32 s0, s16, s8 // (op 604) 
s_add_i32 s51, s0, s16 // (op 604) 

s_ashr_i32 s0, s51, 31 // (op 604) 
s_cmp_eq_i32 s8, 0 // (op 604) 
s_mov_b32 s6, scc // (op 604) 
s_cmp_eq_i32 s6, 0 // (op 604) 
s_cselect_b32 s7, 0, -1 // (op 604) 

s_and_b32 s6, s15, 31 // (op 604) 
s_lshl_b32 s18, 1, s6 // (op 604) 
s_add_i32 s19, s18, s7 // (op 604) 


s_and_b32 s7, s0, s19 // (op 604) 


s_add_i32 s0, s51, s7 // (op 604) 


s_ashr_i32 s7, s0, s6 // (op 604) 


s_lshl_b32 s0, s15, 24 // (op 604) 
s_ashr_i32 s6, s0, 31 // (op 604) 

s_xor_b32 s0, s7, s6 // (op 604) 

s_sub_i32 s7, s0, s6 // (op 604) 


s_add_u32 s18, s4, 16 // (op 604) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 604) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 604) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 604) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 4 // (op 604) 

s_mul_i32 s0, s14, s18 // (op 604) 

s_mul_i32 s6, s50, s0 // (op 604) 

s_mul_i32 s0, s7, s6 // (op 604) 

s_sub_u32 s6, s16, s0 // (op 604) 

s_mov_b32 s0, s6 // (op 604) 
s_mul_hi_i32 s51, s0, s13 // (op 604) 
s_add_i32 s18, s51, s0 // (op 604) 


s_ashr_i32 s0, s18, 31 // (op 604) 
s_cmp_eq_i32 s13, 0 // (op 604) 
s_mov_b32 s19, scc // (op 604) 
s_cmp_eq_i32 s19, 0 // (op 604) 
s_cselect_b32 s51, 0, -1 // (op 604) 

s_and_b32 s19, s49, 31 // (op 604) 
s_lshl_b32 s60, 1, s19 // (op 604) 
s_add_i32 s61, s60, s51 // (op 604) 


s_and_b32 s51, s0, s61 // (op 604) 


s_add_i32 s0, s18, s51 // (op 604) 


s_ashr_i32 s18, s0, s19 // (op 604) 


s_lshl_b32 s0, s49, 24 // (op 604) 
s_ashr_i32 s19, s0, 31 // (op 604) 

s_xor_b32 s0, s18, s19 // (op 604) 

s_sub_i32 s18, s0, s19 // (op 604) 


s_mul_i32 s0, s48, s14 // (op 604) 
s_add_u32 s60, s20, 32 // (op 604) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 604) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 604) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 604) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 5 // (op 604) 

s_sub_i32 s19, s60, s0 // (op 604) 

s_mul_i32 s51, s18, s19 // (op 604) 


s_sub_u32 s18, s6, s51 // (op 604) 


s_add_u32 s6, s18, s0 // (op 604) 


s_cmp_eq_i32 s7, 0 // (op 604) 
s_mov_b32 s0, scc // (op 604) 

s_cmp_eq_i32 s0, 0 // (op 604) 
s_cselect_b32 s7, s6, s1 // (op 604) 



v_lshl_add_u32 v18, s7, 5, v15 // (op 604) 


v_mul_lo_u32 v15, v14, s30 // (op 604) 

v_add_u32 v14, v18, v15 // (op 604) 


s_mul_i32 s1, s30, s35 // (op 606) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 606) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 606) least significant
s_mul_hi_u32 s6, s30, s34 // (op 606) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 606) most significant: sum
s_add_u32 s1, s1, s6 // (op 606) most significant: sum


s_mov_b32 s60, s32 // (op 606) reg expression
s_mov_b32 s61, s33 // (op 606) reg expression
s_mov_b32 s62, s0 // (op 606) reg expression
s_mov_b32 s63, 131072 // (op 606) call()

s_mul_hi_i32 s51, s16, s9 // (op 623) 
v_and_b32 v15, -32, v1 // (op 623) 
s_add_i32 s0, s51, s16 // (op 623) 

v_sub_u32 v18, v1, v15 // (op 623) 

s_ashr_i32 s1, s0, 31 // (op 623) 
v_lshrrev_b32 v15, 5, v1 // (op 623) 
s_cmp_eq_i32 s9, 0 // (op 623) 
s_mov_b32 s51, scc // (op 623) 
s_cmp_eq_i32 s51, 0 // (op 623) 
s_cselect_b32 s6, 0, -1 // (op 623) 

s_and_b32 s7, s10, 31 // (op 623) 
s_lshl_b32 s51, 1, s7 // (op 623) 
s_add_i32 s18, s51, s6 // (op 623) 


s_and_b32 s6, s1, s18 // (op 623) 


s_add_i32 s1, s0, s6 // (op 623) 


s_ashr_i32 s0, s1, s7 // (op 623) 


s_lshl_b32 s1, s10, 24 // (op 623) 
s_ashr_i32 s51, s1, 31 // (op 623) 

s_xor_b32 s1, s0, s51 // (op 623) 

s_sub_i32 s0, s1, s51 // (op 623) 


s_mul_hi_i32 s1, s0, s11 // (op 623) 
s_add_i32 s51, s1, s0 // (op 623) 

s_ashr_i32 s1, s51, 31 // (op 623) 
s_cmp_eq_i32 s11, 0 // (op 623) 
s_mov_b32 s6, scc // (op 623) 
s_cmp_eq_i32 s6, 0 // (op 623) 
s_cselect_b32 s7, 0, -1 // (op 623) 

s_and_b32 s6, s12, 31 // (op 623) 
s_lshl_b32 s18, 1, s6 // (op 623) 
s_add_i32 s19, s18, s7 // (op 623) 


s_and_b32 s7, s1, s19 // (op 623) 


s_add_i32 s1, s51, s7 // (op 623) 


s_ashr_i32 s7, s1, s6 // (op 623) 


s_lshl_b32 s1, s12, 24 // (op 623) 
s_ashr_i32 s6, s1, 31 // (op 623) 

s_xor_b32 s1, s7, s6 // (op 623) 

s_sub_i32 s7, s1, s6 // (op 623) 


s_mul_i32 s1, s0, s14 // (op 623) 

s_sub_u32 s0, s16, s1 // (op 623) 

s_mul_i32 s1, s7, s14 // (op 623) 
s_add_u32 s1, s1, s0 // (op 623) 


s_mul_hi_i32 s0, s16, s8 // (op 623) 
s_add_i32 s51, s0, s16 // (op 623) 

s_ashr_i32 s0, s51, 31 // (op 623) 
s_cmp_eq_i32 s8, 0 // (op 623) 
s_mov_b32 s6, scc // (op 623) 
s_cmp_eq_i32 s6, 0 // (op 623) 
s_cselect_b32 s7, 0, -1 // (op 623) 

s_and_b32 s6, s15, 31 // (op 623) 
s_lshl_b32 s18, 1, s6 // (op 623) 
s_add_i32 s19, s18, s7 // (op 623) 


s_and_b32 s7, s0, s19 // (op 623) 


s_add_i32 s0, s51, s7 // (op 623) 


s_ashr_i32 s7, s0, s6 // (op 623) 


s_lshl_b32 s0, s15, 24 // (op 623) 
s_ashr_i32 s6, s0, 31 // (op 623) 

s_xor_b32 s0, s7, s6 // (op 623) 

s_sub_i32 s7, s0, s6 // (op 623) 


s_add_u32 s18, s4, 16 // (op 623) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 623) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 623) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 623) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 623) 

s_mul_i32 s0, s14, s18 // (op 623) 

s_mul_i32 s6, s50, s0 // (op 623) 

s_mul_i32 s0, s7, s6 // (op 623) 

s_sub_u32 s6, s16, s0 // (op 623) 

s_mov_b32 s0, s6 // (op 623) 
s_mul_hi_i32 s51, s0, s13 // (op 623) 
s_add_i32 s18, s51, s0 // (op 623) 


s_ashr_i32 s0, s18, 31 // (op 623) 
s_cmp_eq_i32 s13, 0 // (op 623) 
s_mov_b32 s19, scc // (op 623) 
s_cmp_eq_i32 s19, 0 // (op 623) 
s_cselect_b32 s51, 0, -1 // (op 623) 

s_and_b32 s19, s49, 31 // (op 623) 
s_lshl_b32 s64, 1, s19 // (op 623) 
s_add_i32 s65, s64, s51 // (op 623) 


s_and_b32 s51, s0, s65 // (op 623) 


s_add_i32 s0, s18, s51 // (op 623) 


s_ashr_i32 s18, s0, s19 // (op 623) 


s_lshl_b32 s0, s49, 24 // (op 623) 
s_ashr_i32 s19, s0, 31 // (op 623) 

s_xor_b32 s0, s18, s19 // (op 623) 

s_sub_i32 s18, s0, s19 // (op 623) 


s_mul_i32 s0, s48, s14 // (op 623) 
s_add_u32 s64, s20, 32 // (op 623) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 623) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 623) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 623) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 5 // (op 623) 

s_sub_i32 s19, s64, s0 // (op 623) 

s_mul_i32 s51, s18, s19 // (op 623) 


s_sub_u32 s18, s6, s51 // (op 623) 


s_add_u32 s6, s18, s0 // (op 623) 


s_cmp_eq_i32 s7, 0 // (op 623) 
s_mov_b32 s0, scc // (op 623) 

s_cmp_eq_i32 s0, 0 // (op 623) 
s_cselect_b32 s7, s6, s1 // (op 623) 



v_lshl_add_u32 v19, s7, 5, v18 // (op 623) 


v_mul_lo_u32 v18, v15, s30 // (op 623) 

v_add_u32 v15, v19, v18 // (op 623) 


s_mul_i32 s1, s30, s35 // (op 625) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 625) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 625) least significant
s_mul_hi_u32 s6, s30, s34 // (op 625) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 625) most significant: sum
s_add_u32 s1, s1, s6 // (op 625) most significant: sum


s_mov_b32 s60, s32 // (op 625) reg expression
s_mov_b32 s61, s33 // (op 625) reg expression
s_mov_b32 s62, s0 // (op 625) reg expression
s_mov_b32 s63, 131072 // (op 625) call()

s_mul_hi_i32 s51, s16, s9 // (op 668) 
v_and_b32 v18, 15, v1 // (op 668) 
s_add_i32 s0, s51, s16 // (op 668) 

v_and_b32 v19, 63, v1 // (op 668) 
s_ashr_i32 s1, s0, 31 // (op 668) 
v_lshrrev_b32 v20, 4, v19 // (op 668) 

s_cmp_eq_i32 s9, 0 // (op 668) 
s_mov_b32 s51, scc // (op 668) 
s_cmp_eq_i32 s51, 0 // (op 668) 
s_cselect_b32 s6, 0, -1 // (op 668) 

s_and_b32 s7, s10, 31 // (op 668) 
s_lshl_b32 s51, 1, s7 // (op 668) 
s_add_i32 s18, s51, s6 // (op 668) 


s_and_b32 s6, s1, s18 // (op 668) 


s_add_i32 s1, s0, s6 // (op 668) 


s_ashr_i32 s0, s1, s7 // (op 668) 


s_lshl_b32 s1, s10, 24 // (op 668) 
s_ashr_i32 s51, s1, 31 // (op 668) 

s_xor_b32 s1, s0, s51 // (op 668) 

s_sub_i32 s0, s1, s51 // (op 668) 


s_mul_hi_i32 s1, s0, s11 // (op 668) 
s_add_i32 s51, s1, s0 // (op 668) 

s_ashr_i32 s1, s51, 31 // (op 668) 
s_cmp_eq_i32 s11, 0 // (op 668) 
s_mov_b32 s6, scc // (op 668) 
s_cmp_eq_i32 s6, 0 // (op 668) 
s_cselect_b32 s7, 0, -1 // (op 668) 

s_and_b32 s6, s12, 31 // (op 668) 
s_lshl_b32 s18, 1, s6 // (op 668) 
s_add_i32 s19, s18, s7 // (op 668) 


s_and_b32 s7, s1, s19 // (op 668) 


s_add_i32 s1, s51, s7 // (op 668) 


s_ashr_i32 s7, s1, s6 // (op 668) 


s_lshl_b32 s1, s12, 24 // (op 668) 
s_ashr_i32 s6, s1, 31 // (op 668) 

s_xor_b32 s1, s7, s6 // (op 668) 

s_sub_i32 s7, s1, s6 // (op 668) 


s_add_u32 s18, s4, 16 // (op 668) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 668) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 668) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 668) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 4 // (op 668) 

s_mul_i32 s1, s7, s18 // (op 668) 

s_sub_i32 s51, s0, s1 // (op 668) 


s_mul_hi_i32 s0, s16, s8 // (op 668) 
s_add_i32 s1, s0, s16 // (op 668) 

s_ashr_i32 s0, s1, 31 // (op 668) 
s_cmp_eq_i32 s8, 0 // (op 668) 
s_mov_b32 s6, scc // (op 668) 
s_cmp_eq_i32 s6, 0 // (op 668) 
s_cselect_b32 s7, 0, -1 // (op 668) 

s_and_b32 s6, s15, 31 // (op 668) 
s_lshl_b32 s64, 1, s6 // (op 668) 
s_add_i32 s65, s64, s7 // (op 668) 


s_and_b32 s7, s0, s65 // (op 668) 


s_add_i32 s0, s1, s7 // (op 668) 


s_ashr_i32 s1, s0, s6 // (op 668) 


s_lshl_b32 s0, s15, 24 // (op 668) 
s_ashr_i32 s6, s0, 31 // (op 668) 

s_xor_b32 s0, s1, s6 // (op 668) 

s_sub_i32 s1, s0, s6 // (op 668) 


s_mul_i32 s0, s14, s18 // (op 668) 

s_mul_i32 s6, s50, s0 // (op 668) 

s_mul_i32 s0, s1, s6 // (op 668) 

s_sub_u32 s6, s16, s0 // (op 668) 

s_mul_hi_i32 s0, s6, s13 // (op 668) 
s_add_i32 s7, s0, s6 // (op 668) 


s_ashr_i32 s0, s7, 31 // (op 668) 
s_cmp_eq_i32 s13, 0 // (op 668) 
s_mov_b32 s6, scc // (op 668) 
s_cmp_eq_i32 s6, 0 // (op 668) 
s_cselect_b32 s18, 0, -1 // (op 668) 

s_and_b32 s6, s49, 31 // (op 668) 
s_lshl_b32 s19, 1, s6 // (op 668) 
s_add_i32 s64, s19, s18 // (op 668) 


s_and_b32 s18, s0, s64 // (op 668) 


s_add_i32 s0, s7, s18 // (op 668) 


s_ashr_i32 s7, s0, s6 // (op 668) 


s_lshl_b32 s0, s49, 24 // (op 668) 
s_ashr_i32 s6, s0, 31 // (op 668) 

s_xor_b32 s0, s7, s6 // (op 668) 

s_sub_i32 s7, s0, s6 // (op 668) 


s_cmp_eq_i32 s1, 0 // (op 668) 
s_mov_b32 s0, scc // (op 668) 

s_cmp_eq_i32 s0, 0 // (op 668) 
s_cselect_b32 s1, s7, s51 // (op 668) 



v_lshl_add_u32 v19, s1, 4, v18 // (op 668) 


v_mul_lo_u32 v18, v20, s36 // (op 668) 
v_add_u32 v18, v18, v19 // (op 668) 


s_mul_i32 s1, s36, s35 // (op 670) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 670) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 670) least significant
s_mul_hi_u32 s6, s36, s34 // (op 670) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 670) most significant: sum
s_add_u32 s1, s1, s6 // (op 670) most significant: sum


s_mov_b32 s64, s38 // (op 670) reg expression
s_mov_b32 s65, s39 // (op 670) reg expression
s_mov_b32 s66, s0 // (op 670) reg expression
s_mov_b32 s67, 131072 // (op 670) call()

s_mul_hi_i32 s51, s16, s9 // (op 687) 
v_and_b32 v19, 15, v1 // (op 687) 
s_add_i32 s0, s51, s16 // (op 687) 

v_and_b32 v20, 63, v1 // (op 687) 
s_ashr_i32 s1, s0, 31 // (op 687) 
v_lshrrev_b32 v21, 4, v20 // (op 687) 

s_cmp_eq_i32 s9, 0 // (op 687) 
s_mov_b32 s51, scc // (op 687) 
s_cmp_eq_i32 s51, 0 // (op 687) 
s_cselect_b32 s6, 0, -1 // (op 687) 

s_and_b32 s7, s10, 31 // (op 687) 
s_lshl_b32 s51, 1, s7 // (op 687) 
s_add_i32 s18, s51, s6 // (op 687) 


s_and_b32 s6, s1, s18 // (op 687) 


s_add_i32 s1, s0, s6 // (op 687) 


s_ashr_i32 s0, s1, s7 // (op 687) 


s_lshl_b32 s1, s10, 24 // (op 687) 
s_ashr_i32 s51, s1, 31 // (op 687) 

s_xor_b32 s1, s0, s51 // (op 687) 

s_sub_i32 s0, s1, s51 // (op 687) 


s_mul_hi_i32 s1, s0, s11 // (op 687) 
s_add_i32 s51, s1, s0 // (op 687) 

s_ashr_i32 s1, s51, 31 // (op 687) 
s_cmp_eq_i32 s11, 0 // (op 687) 
s_mov_b32 s6, scc // (op 687) 
s_cmp_eq_i32 s6, 0 // (op 687) 
s_cselect_b32 s7, 0, -1 // (op 687) 

s_and_b32 s6, s12, 31 // (op 687) 
s_lshl_b32 s18, 1, s6 // (op 687) 
s_add_i32 s19, s18, s7 // (op 687) 


s_and_b32 s7, s1, s19 // (op 687) 


s_add_i32 s1, s51, s7 // (op 687) 


s_ashr_i32 s7, s1, s6 // (op 687) 


s_lshl_b32 s1, s12, 24 // (op 687) 
s_ashr_i32 s6, s1, 31 // (op 687) 

s_xor_b32 s1, s7, s6 // (op 687) 

s_sub_i32 s7, s1, s6 // (op 687) 


s_add_u32 s18, s4, 16 // (op 687) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 687) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 687) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 687) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 687) 

s_mul_i32 s1, s7, s18 // (op 687) 

s_sub_i32 s51, s0, s1 // (op 687) 


s_mul_hi_i32 s0, s16, s8 // (op 687) 
s_add_i32 s1, s0, s16 // (op 687) 

s_ashr_i32 s0, s1, 31 // (op 687) 
s_cmp_eq_i32 s8, 0 // (op 687) 
s_mov_b32 s6, scc // (op 687) 
s_cmp_eq_i32 s6, 0 // (op 687) 
s_cselect_b32 s7, 0, -1 // (op 687) 

s_and_b32 s6, s15, 31 // (op 687) 
s_lshl_b32 s68, 1, s6 // (op 687) 
s_add_i32 s69, s68, s7 // (op 687) 


s_and_b32 s7, s0, s69 // (op 687) 


s_add_i32 s0, s1, s7 // (op 687) 


s_ashr_i32 s1, s0, s6 // (op 687) 


s_lshl_b32 s0, s15, 24 // (op 687) 
s_ashr_i32 s6, s0, 31 // (op 687) 

s_xor_b32 s0, s1, s6 // (op 687) 

s_sub_i32 s1, s0, s6 // (op 687) 


s_mul_i32 s0, s14, s18 // (op 687) 

s_mul_i32 s6, s50, s0 // (op 687) 

s_mul_i32 s0, s1, s6 // (op 687) 

s_sub_u32 s6, s16, s0 // (op 687) 

s_mul_hi_i32 s0, s6, s13 // (op 687) 
s_add_i32 s7, s0, s6 // (op 687) 


s_ashr_i32 s0, s7, 31 // (op 687) 
s_cmp_eq_i32 s13, 0 // (op 687) 
s_mov_b32 s6, scc // (op 687) 
s_cmp_eq_i32 s6, 0 // (op 687) 
s_cselect_b32 s18, 0, -1 // (op 687) 

s_and_b32 s6, s49, 31 // (op 687) 
s_lshl_b32 s19, 1, s6 // (op 687) 
s_add_i32 s68, s19, s18 // (op 687) 


s_and_b32 s18, s0, s68 // (op 687) 


s_add_i32 s0, s7, s18 // (op 687) 


s_ashr_i32 s7, s0, s6 // (op 687) 


s_lshl_b32 s0, s49, 24 // (op 687) 
s_ashr_i32 s6, s0, 31 // (op 687) 

s_xor_b32 s0, s7, s6 // (op 687) 

s_sub_i32 s7, s0, s6 // (op 687) 


s_cmp_eq_i32 s1, 0 // (op 687) 
s_mov_b32 s0, scc // (op 687) 

s_cmp_eq_i32 s0, 0 // (op 687) 
s_cselect_b32 s1, s7, s51 // (op 687) 



v_lshl_add_u32 v20, s1, 4, v19 // (op 687) 


v_mul_lo_u32 v19, v21, s36 // (op 687) 
v_add_u32 v19, v19, v20 // (op 687) 


s_mul_i32 s1, s36, s35 // (op 689) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 689) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 689) least significant
s_mul_hi_u32 s6, s36, s34 // (op 689) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 689) most significant: sum
s_add_u32 s1, s1, s6 // (op 689) most significant: sum


s_mov_b32 s64, s38 // (op 689) reg expression
s_mov_b32 s65, s39 // (op 689) reg expression
s_mov_b32 s66, s0 // (op 689) reg expression
s_mov_b32 s67, 131072 // (op 689) call()


v_mov_b64 v[20:21], s[36:37] // (op 341) convert

v_mov_b32 v22, v18 // (op 341) reg expression
v_mov_b32 v23, v22 // (op 341) 
buffer_load_ubyte v24, v23, s[64:67], 0 offen offset: 0 // (op 341) Load value





v_mov_b32 v20, v16 // (op 345) reg expression

buffer_load_dwordx3 v[26:28], v20, s[56:59], 0 offen // (op 345) Load value

v_and_b32 v25, -2, v1 // (op 753) 
v_sub_u32 v20, v1, v25 // (op 753) 

v_lshlrev_b32 v25, 4, v20 // (op 753) 

v_lshrrev_b32 v20, 1, v1 // (op 753) 
v_lshl_add_u32 v21, v20, 5, v25 // (op 753) 


s_nop 0
v_readfirstlane_b32 s51, v21 // (op 753) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[2:3], 6 // (op 432) 
v_mov_b64 v[20:21], s[0:1] // (op 432) convert


v_mov_b64 v[22:23], v[10:11] // (op 432) reg expression
s_mov_b32 m0, s51 // (op 432) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 432) Load value direct to lds
v_add_co_u32 v22, s[0:1], v22, v20 // (op 432) least significant half
v_addc_co_u32 v23, s[0:1], 0, v23, s[0:1] // (op 432) most significant half

s_add_u32 m0, m0, 2048 // (op 432) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 432) Load value direct to lds





v_mov_b32 v25, v14 // (op 351) reg expression

buffer_load_ubyte v20, v25, s[60:63], 0 offen offset: 0 // (op 351) Load value


v_mov_b64 v[22:23], s[36:37] // (op 355) convert

s_lshl_b32 s51, s36, 2 // (op 355) 
v_add_u32 v21, v18, s51 // (op 355) 

v_mov_b32 v25, v21 // (op 355) 
buffer_load_ubyte v29, v25, s[64:67], 0 offen offset: 0 // (op 355) Load value






s_lshl_b64 s[0:1], s[26:27], 7 // (op 359) 
s_mul_i32 s7, s1, 6 // (op 359) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 359) least significant
s_mul_hi_u32 s51, s0, 6 // (op 359) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 359) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 359) 

v_add_u32 v18, v16, s0 // (op 359) 


buffer_load_dwordx3 v[30:32], v18, s[56:59], 0 offen // (op 359) Load value


v_and_b32 v16, -2, v1 // (op 932) 
v_sub_u32 v18, v1, v16 // (op 932) 

v_lshlrev_b32 v16, 4, v18 // (op 932) 

v_lshrrev_b32 v18, 1, v1 // (op 932) 
v_lshl_add_u32 v25, v18, 5, v16 // (op 932) 


s_nop 0
v_readfirstlane_b32 s51, v25 // (op 932) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[2:3], 6 // (op 437) 
v_mov_b64 v[22:23], s[0:1] // (op 437) convert


s_lshl_b64 s[0:1], s[2:3], 7 // (op 437) 
v_mov_b32 v16, s0 // (op 437) 
v_mov_b32 v18, s1 // (op 437) 
v_add_co_u32 v34, s[6:7], v10, v16 // (op 437) least significant half
v_addc_co_u32 v35, s[6:7], v11, v18, s[6:7] // (op 437) most significant half




s_add_u32 s0, 4096, s51 // (op 437) 
s_mov_b32 m0, s0 // (op 437) reg expression
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 437) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v22 // (op 437) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 437) most significant half

s_add_u32 m0, m0, 2048 // (op 437) 
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 437) Load value direct to lds







s_lshl_b32 s51, s30, 2 // (op 365) 
v_add_u32 v16, v14, s51 // (op 365) 


buffer_load_ubyte v18, v16, s[60:63], 0 offen offset: 0 // (op 365) Load value


v_lshlrev_b32 v14, 4, v1 // (op 806) 
v_mul_lo_u32 v16, v14, 6 // (op 806) 

v_lshrrev_b32 v14, 3, v16 // (op 806) 

v_and_b32 v16, 268435455, v1 // (op 806) 
v_lshlrev_b32 v25, 2, v16 // (op 806) 

v_add_u32 v16, v14, v25 // (op 806) 





s_waitcnt vmcnt(8)
ds_write_b96 v16, v[26:28] offset:8192 // (op 367) Store local data 

v_and_b32 v14, -32, v1 // (op 859) 
v_sub_u32 v16, v1, v14 // (op 859) 

v_and_b32 v14, 4294967264, v1 // (op 859) 
v_add_u32 v25, v14, v16 // (op 859) 





s_waitcnt vmcnt(5)
ds_write_b8 v25, v20 offset:10240 // (op 369) Store local data 

v_and_b32 v14, -2, v1 // (op 725) 
v_sub_u32 v16, v1, v14 // (op 725) 

v_lshlrev_b32 v14, 4, v16 // (op 725) 

v_lshrrev_b32 v16, 1, v1 // (op 725) 
v_lshl_add_u32 v25, v16, 5, v14 // (op 725) 


s_nop 0
v_readfirstlane_b32 s51, v25 // (op 725) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 4, v1 // (op 777) 
v_mul_lo_u32 v16, v14, 6 // (op 777) 

v_lshrrev_b32 v14, 3, v16 // (op 777) 

v_and_b32 v16, 268435455, v1 // (op 777) 
v_lshlrev_b32 v25, 2, v16 // (op 777) 

v_add_u32 v16, v14, v25 // (op 777) 


v_and_b32 v14, -32, v1 // (op 830) 
v_sub_u32 v25, v1, v14 // (op 830) 

v_and_b32 v14, 4294967264, v1 // (op 830) 
v_add_u32 v10, v14, v25 // (op 830) 


v_and_b32 v11, -2, v1 // (op 917) 
v_sub_u32 v14, v1, v11 // (op 917) 

v_lshlrev_b32 v11, 4, v14 // (op 917) 

v_lshrrev_b32 v14, 1, v1 // (op 917) 
v_lshl_add_u32 v25, v14, 5, v11 // (op 917) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 917) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v11, 4, v1 // (op 956) 
v_mul_lo_u32 v14, v11, 6 // (op 956) 

v_lshrrev_b32 v11, 3, v14 // (op 956) 

v_and_b32 v14, 268435455, v1 // (op 956) 
v_lshlrev_b32 v25, 2, v14 // (op 956) 

v_add_u32 v14, v11, v25 // (op 956) 


v_and_b32 v11, -32, v1 // (op 980) 
v_sub_u32 v25, v1, v11 // (op 980) 

v_and_b32 v11, 4294967264, v1 // (op 980) 
v_add_u32 v21, v11, v25 // (op 980) 


s_mov_b32 s1, 0 // (op 57) call()
s_ashr_i64 s[6:7], s[24:25], 7 // (op 56) 
s_mov_b32 s18, s6 // (op 56) convert

s_and_b32 s19, s18, 4294967294 // (op 56) 

s_cmp_lt_u32 s1, s19 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_56 if false)
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_56:
s_barrier  // (op 1025) 
s_add_u32 s1, s1, 2 // (op 58) 



ds_read_b64_tr_b8 v[34:35], v0 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:256 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:2048 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:2304 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1038) 



ds_read_b96_tr_b6 v[42:44], v5 offset:8192 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[45:47], v5 offset:8448 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1052) 


ds_read_u8 v11, v3 offset:10240 // (op 77) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:47], a[0:3], v11, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 84) 




v_mov_b64 v[22:23], s[36:37] // (op 82) convert

v_mov_b32 v11, v19 // (op 82) 
s_lshl_b32 s6, s36, 3 // (op 82) 
v_add_u32 v25, v11, s6 // (op 82) 

v_mov_b32 v33, v25 // (op 82) 
buffer_load_ubyte v24, v33, s[64:67], 0 offen offset: 0 // (op 82) Load value





v_mov_b32 v25, v17 // (op 10) 
s_lshl_b64 s[6:7], s[26:27], 7 // (op 10) 
s_mul_i32 s19, s7, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 10) least significant
s_mul_hi_u32 s68, s6, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 10) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 10) 

s_lshl_b32 s18, s6, 1 // (op 10) 

v_add_u32 v22, v25, s18 // (op 10) 


buffer_load_dwordx3 v[26:28], v22, s[56:59], 0 offen // (op 10) Load value

s_barrier  // (op 1028) 

s_lshl_b64 s[6:7], s[2:3], 6 // (op 416) 
v_mov_b64 v[22:23], s[6:7] // (op 416) convert


v_mov_b64 v[34:35], v[12:13] // (op 416) 
s_lshl_b64 s[6:7], s[2:3], 7 // (op 416) 
s_lshl_b64 s[18:19], s[6:7], 1 // (op 416) 

v_mov_b32 v33, s18 // (op 416) 
v_mov_b32 v36, s19 // (op 416) 
v_add_co_u32 v38, s[6:7], v34, v33 // (op 416) least significant half
v_addc_co_u32 v39, s[6:7], v35, v36, s[6:7] // (op 416) most significant half




s_mov_b32 m0, s51 // (op 416) reg expression
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 416) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v22 // (op 416) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 416) most significant half

s_add_u32 m0, m0, 2048 // (op 416) 
buffer_load_dwordx4 v38, s[52:55], 0 offen offset: 0 lds // (op 416) Load value direct to lds




v_mov_b32 v33, v15 // (op 17) 
s_lshl_b32 s6, s30, 3 // (op 17) 
v_add_u32 v22, v33, s6 // (op 17) 


buffer_load_ubyte v20, v22, s[60:63], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1075) 



s_waitcnt vmcnt(8)
ds_write_b96 v14, v[30:32] offset:10368 // (op 239) Store local data 
s_barrier  // (op 1081) 



s_waitcnt vmcnt(5)
ds_write_b8 v21, v18 offset:12416 // (op 240) Store local data 
s_barrier  // (op 1066) 



ds_read_b64_tr_b8 v[36:37], v4 offset:4096 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:4352 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:6144 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v4 offset:6400 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1072) 



ds_read_b96_tr_b6 v[44:46], v8 offset:10368 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[47:49], v8 offset:10624 // (op 233) Transpose load local data 
s_barrier  // (op 1132) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1078) 


ds_read_u8 v22, v6 offset:12416 // (op 235) Load local data 
s_barrier  // (op 1138) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:49], a[0:3], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 238) 




v_mov_b64 v[22:23], s[36:37] // (op 237) convert

s_lshl_b32 s6, s36, 2 // (op 237) 
s_mul_i32 s7, 3, s6 // (op 237) 

v_add_u32 v36, v11, s7 // (op 237) 

v_mov_b32 v37, v36 // (op 237) 
buffer_load_ubyte v29, v37, s[64:67], 0 offen offset: 0 // (op 237) Load value






s_lshl_b32 s6, s36, 3 // (op 679) 
v_add_u32 v19, v19, s6 // (op 679) 

s_lshl_b64 s[6:7], s[26:27], 7 // (op 241) 
s_mul_i32 s19, s7, 6 // (op 241) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 241) least significant
s_mul_hi_u32 s68, s6, 6 // (op 241) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 241) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 241) 

s_mul_i32 s18, 3, s6 // (op 241) 

v_add_u32 v11, v25, s18 // (op 241) 


buffer_load_dwordx3 v[30:32], v11, s[56:59], 0 offen // (op 241) Load value


s_barrier  // (op 1069) 
s_lshl_b64 s[6:7], s[26:27], 7 // (op 551) 

s_mul_i32 s19, s7, 6 // (op 551) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 551) least significant
s_mul_hi_u32 s68, s6, 6 // (op 551) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 551) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 551) 

s_lshl_b32 s18, s6, 1 // (op 551) 

v_add_u32 v17, v17, s18 // (op 551) 

s_lshl_b64 s[6:7], s[2:3], 6 // (op 427) 
v_mov_b64 v[22:23], s[6:7] // (op 427) convert


s_lshl_b64 s[6:7], s[2:3], 7 // (op 427) 
s_mul_i32 s19, 3, s7 // (op 427) most significant: low of low * high
s_mul_i32 s18, 3, s6 // (op 427) least significant
s_mul_hi_u32 s68, 3, s6 // (op 427) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 427) most significant: sum


v_mov_b32 v11, s18 // (op 427) 
v_mov_b32 v25, s19 // (op 427) 
v_add_co_u32 v36, s[6:7], v34, v11 // (op 427) least significant half
v_addc_co_u32 v37, s[6:7], v35, v25, s[6:7] // (op 427) most significant half




s_add_u32 s6, 4096, s0 // (op 427) 
s_nop 0
s_mov_b32 m0, s6 // (op 427) reg expression
buffer_load_dwordx4 v36, s[52:55], 0 offen offset: 0 lds // (op 427) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v22 // (op 427) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 427) most significant half

s_add_u32 m0, m0, 2048 // (op 427) 
buffer_load_dwordx4 v36, s[52:55], 0 offen offset: 0 lds // (op 427) Load value direct to lds






s_lshl_b32 s6, s30, 2 // (op 242) 
s_mul_i32 s7, 3, s6 // (op 242) 

v_add_u32 v11, v33, s7 // (op 242) 


s_lshl_b64 s[6:7], s[2:3], 7 // (op 487) 
s_lshl_b64 s[18:19], s[6:7], 1 // (op 487) 

v_mov_b32 v25, s18 // (op 487) 
v_mov_b32 v22, s19 // (op 487) 
v_add_co_u32 v12, s[6:7], v12, v25 // (op 487) least significant half
v_addc_co_u32 v13, s[6:7], v13, v22, s[6:7] // (op 487) most significant half




buffer_load_ubyte v18, v11, s[60:63], 0 offen offset: 0 // (op 242) Load value


s_barrier  // (op 1041) 
s_lshl_b32 s6, s30, 3 // (op 615) 



v_add_u32 v15, v15, s6 // (op 615) 

s_waitcnt vmcnt(8)
ds_write_b96 v16, v[26:28] offset:8192 // (op 13) Store local data 
s_barrier  // (op 1055) 



s_waitcnt vmcnt(5)
ds_write_b8 v10, v20 offset:10240 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[24:25], 7 // (op 56) 
s_mov_b32 s18, s6 // (op 56) convert

s_and_b32 s19, s18, 4294967294 // (op 56) 

s_cmp_lt_u32 s1, s19 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoop_56 if true)
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoop_56:














s_barrier  // (op 1094) 
s_mul_hi_i32 s51, s16, s9 // (op 450) 
v_and_b32 v10, -2, v1 // (op 450) 
s_add_i32 s0, s51, s16 // (op 450) 

v_sub_u32 v11, v1, v10 // (op 450) 

s_ashr_i32 s1, s0, 31 // (op 450) 
v_lshlrev_b32 v10, 4, v11 // (op 450) 

s_cmp_eq_i32 s9, 0 // (op 450) 
s_mov_b32 s51, scc // (op 450) 
v_lshrrev_b32 v11, 1, v1 // (op 450) 
s_cmp_eq_i32 s51, 0 // (op 450) 
s_cselect_b32 s6, 0, -1 // (op 450) 

s_and_b32 s7, s10, 31 // (op 450) 
s_lshl_b32 s51, 1, s7 // (op 450) 
s_add_i32 s18, s51, s6 // (op 450) 


s_and_b32 s6, s1, s18 // (op 450) 


s_add_i32 s1, s0, s6 // (op 450) 


s_ashr_i32 s0, s1, s7 // (op 450) 


s_lshl_b32 s1, s10, 24 // (op 450) 
s_ashr_i32 s51, s1, 31 // (op 450) 

s_xor_b32 s1, s0, s51 // (op 450) 

s_sub_i32 s0, s1, s51 // (op 450) 


s_mul_hi_i32 s1, s0, s11 // (op 450) 
s_add_i32 s51, s1, s0 // (op 450) 

s_ashr_i32 s1, s51, 31 // (op 450) 
s_cmp_eq_i32 s11, 0 // (op 450) 
s_mov_b32 s6, scc // (op 450) 
s_cmp_eq_i32 s6, 0 // (op 450) 
s_cselect_b32 s7, 0, -1 // (op 450) 

s_and_b32 s6, s12, 31 // (op 450) 
s_lshl_b32 s18, 1, s6 // (op 450) 
s_add_i32 s19, s18, s7 // (op 450) 


s_and_b32 s7, s1, s19 // (op 450) 


s_add_i32 s1, s51, s7 // (op 450) 


s_ashr_i32 s7, s1, s6 // (op 450) 


s_lshl_b32 s1, s12, 24 // (op 450) 
s_ashr_i32 s6, s1, 31 // (op 450) 

s_xor_b32 s1, s7, s6 // (op 450) 

s_sub_i32 s7, s1, s6 // (op 450) 


s_mul_i32 s1, s0, s14 // (op 450) 

s_sub_u32 s0, s16, s1 // (op 450) 

s_mul_i32 s1, s7, s14 // (op 450) 
s_add_u32 s1, s1, s0 // (op 450) 


s_mul_hi_i32 s0, s16, s8 // (op 450) 
s_add_i32 s51, s0, s16 // (op 450) 

s_ashr_i32 s0, s51, 31 // (op 450) 
s_cmp_eq_i32 s8, 0 // (op 450) 
s_mov_b32 s6, scc // (op 450) 
s_cmp_eq_i32 s6, 0 // (op 450) 
s_cselect_b32 s7, 0, -1 // (op 450) 

s_and_b32 s6, s15, 31 // (op 450) 
s_lshl_b32 s18, 1, s6 // (op 450) 
s_add_i32 s19, s18, s7 // (op 450) 


s_and_b32 s7, s0, s19 // (op 450) 


s_add_i32 s0, s51, s7 // (op 450) 


s_ashr_i32 s7, s0, s6 // (op 450) 


s_lshl_b32 s0, s15, 24 // (op 450) 
s_ashr_i32 s6, s0, 31 // (op 450) 

s_xor_b32 s0, s7, s6 // (op 450) 

s_sub_i32 s7, s0, s6 // (op 450) 


s_add_u32 s18, s4, 16 // (op 450) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 450) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 450) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 450) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 450) 

s_mul_i32 s0, s14, s18 // (op 450) 

s_mul_i32 s6, s50, s0 // (op 450) 

s_mul_i32 s0, s7, s6 // (op 450) 

s_sub_u32 s6, s16, s0 // (op 450) 

s_mov_b32 s0, s6 // (op 450) 
s_mul_hi_i32 s51, s0, s13 // (op 450) 
s_add_i32 s18, s51, s0 // (op 450) 


s_ashr_i32 s0, s18, 31 // (op 450) 
s_cmp_eq_i32 s13, 0 // (op 450) 
s_mov_b32 s19, scc // (op 450) 
s_cmp_eq_i32 s19, 0 // (op 450) 
s_cselect_b32 s51, 0, -1 // (op 450) 

s_and_b32 s19, s49, 31 // (op 450) 
s_lshl_b32 s68, 1, s19 // (op 450) 
s_add_i32 s69, s68, s51 // (op 450) 


s_and_b32 s51, s0, s69 // (op 450) 


s_add_i32 s0, s18, s51 // (op 450) 


s_ashr_i32 s18, s0, s19 // (op 450) 


s_lshl_b32 s0, s49, 24 // (op 450) 
s_ashr_i32 s19, s0, 31 // (op 450) 

s_xor_b32 s0, s18, s19 // (op 450) 

s_sub_i32 s18, s0, s19 // (op 450) 


s_mul_i32 s0, s48, s14 // (op 450) 
s_add_u32 s68, s20, 32 // (op 450) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 450) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 450) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 450) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 450) 

s_sub_i32 s19, s68, s0 // (op 450) 

s_mul_i32 s51, s18, s19 // (op 450) 


s_sub_u32 s18, s6, s51 // (op 450) 


s_add_u32 s6, s18, s0 // (op 450) 


s_cmp_eq_i32 s7, 0 // (op 450) 
s_mov_b32 s0, scc // (op 450) 

s_cmp_eq_i32 s0, 0 // (op 450) 
s_cselect_b32 s7, s6, s1 // (op 450) 



v_lshl_add_u32 v12, s7, 5, v10 // (op 450) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 450) 
v_mov_b32 v17, 0 // (op 450) convert
v_mov_b32 v16, v12 // (op 450) convert

s_lshr_b32 s51, s0, 1 // (op 450) 

v_lshl_add_u32 v10, s51, 8, v11 // (op 450) 


v_mul_lo_u32 v15, v10, s3 // (op 450) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 450) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 450) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 450) most significant: sum


v_add_co_u32 v10, s[0:1], v16, v14 // (op 450) least significant half
v_addc_co_u32 v11, s[0:1], v17, v15, s[0:1] // (op 450) most significant half



s_mul_i32 s1, s2, s25 // (op 452) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 452) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 452) least significant
s_mul_hi_u32 s6, s2, s24 // (op 452) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 452) most significant: sum
s_add_u32 s1, s1, s6 // (op 452) most significant: sum


s_mov_b32 s52, s22 // (op 452) reg expression
s_mov_b32 s53, s23 // (op 452) reg expression
s_mov_b32 s54, s0 // (op 452) reg expression
s_mov_b32 s55, 131072 // (op 452) call()

s_mul_hi_i32 s51, s16, s9 // (op 517) 
s_add_i32 s0, s51, s16 // (op 517) 

s_ashr_i32 s1, s0, 31 // (op 517) 
s_cmp_eq_i32 s9, 0 // (op 517) 
s_mov_b32 s51, scc // (op 517) 
s_cmp_eq_i32 s51, 0 // (op 517) 
s_cselect_b32 s6, 0, -1 // (op 517) 

s_and_b32 s7, s10, 31 // (op 517) 
s_lshl_b32 s51, 1, s7 // (op 517) 
s_add_i32 s18, s51, s6 // (op 517) 


s_and_b32 s6, s1, s18 // (op 517) 


s_add_i32 s1, s0, s6 // (op 517) 


s_ashr_i32 s0, s1, s7 // (op 517) 


s_lshl_b32 s1, s10, 24 // (op 517) 
s_ashr_i32 s51, s1, 31 // (op 517) 

s_xor_b32 s1, s0, s51 // (op 517) 

s_sub_i32 s0, s1, s51 // (op 517) 


s_mul_hi_i32 s1, s0, s11 // (op 517) 
s_add_i32 s51, s1, s0 // (op 517) 

s_ashr_i32 s1, s51, 31 // (op 517) 
s_cmp_eq_i32 s11, 0 // (op 517) 
s_mov_b32 s6, scc // (op 517) 
s_cmp_eq_i32 s6, 0 // (op 517) 
s_cselect_b32 s7, 0, -1 // (op 517) 

s_and_b32 s6, s12, 31 // (op 517) 
s_lshl_b32 s18, 1, s6 // (op 517) 
s_add_i32 s19, s18, s7 // (op 517) 


s_and_b32 s7, s1, s19 // (op 517) 


s_add_i32 s1, s51, s7 // (op 517) 


s_ashr_i32 s7, s1, s6 // (op 517) 


s_lshl_b32 s1, s12, 24 // (op 517) 
s_ashr_i32 s6, s1, 31 // (op 517) 

s_xor_b32 s1, s7, s6 // (op 517) 

s_sub_i32 s7, s1, s6 // (op 517) 


s_add_u32 s18, s4, 16 // (op 517) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 517) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 517) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 517) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 517) 

s_mul_i32 s1, s7, s18 // (op 517) 

s_sub_i32 s51, s0, s1 // (op 517) 


s_mul_hi_i32 s0, s16, s8 // (op 517) 
s_add_i32 s1, s0, s16 // (op 517) 

s_ashr_i32 s0, s1, 31 // (op 517) 
s_cmp_eq_i32 s8, 0 // (op 517) 
s_mov_b32 s6, scc // (op 517) 
s_cmp_eq_i32 s6, 0 // (op 517) 
s_cselect_b32 s7, 0, -1 // (op 517) 

s_and_b32 s6, s15, 31 // (op 517) 
s_lshl_b32 s68, 1, s6 // (op 517) 
s_add_i32 s69, s68, s7 // (op 517) 


s_and_b32 s7, s0, s69 // (op 517) 


s_add_i32 s0, s1, s7 // (op 517) 


s_ashr_i32 s1, s0, s6 // (op 517) 


s_lshl_b32 s0, s15, 24 // (op 517) 
s_ashr_i32 s6, s0, 31 // (op 517) 

s_xor_b32 s0, s1, s6 // (op 517) 

s_sub_i32 s1, s0, s6 // (op 517) 


s_mul_i32 s0, s14, s18 // (op 517) 

s_mul_i32 s6, s50, s0 // (op 517) 

s_mul_i32 s0, s1, s6 // (op 517) 

s_sub_u32 s6, s16, s0 // (op 517) 

s_mul_hi_i32 s0, s6, s13 // (op 517) 
s_add_i32 s7, s0, s6 // (op 517) 


s_ashr_i32 s0, s7, 31 // (op 517) 
s_cmp_eq_i32 s13, 0 // (op 517) 
s_mov_b32 s6, scc // (op 517) 
s_cmp_eq_i32 s6, 0 // (op 517) 
s_cselect_b32 s18, 0, -1 // (op 517) 

s_and_b32 s6, s49, 31 // (op 517) 
s_lshl_b32 s19, 1, s6 // (op 517) 
s_add_i32 s68, s19, s18 // (op 517) 


s_and_b32 s18, s0, s68 // (op 517) 


s_add_i32 s0, s7, s18 // (op 517) 


s_ashr_i32 s7, s0, s6 // (op 517) 


s_lshl_b32 s0, s49, 24 // (op 517) 
s_ashr_i32 s6, s0, 31 // (op 517) 

s_xor_b32 s0, s7, s6 // (op 517) 

s_sub_i32 s7, s0, s6 // (op 517) 


s_cmp_eq_i32 s1, 0 // (op 517) 
s_mov_b32 s0, scc // (op 517) 

s_cmp_eq_i32 s0, 0 // (op 517) 
s_cselect_b32 s1, s7, s51 // (op 517) 



s_lshl_b32 s0, s1, 4 // (op 517) 

s_mov_b32 s7, 0 // (op 517) convert
s_mov_b32 s6, s0 // (op 517) convert

s_ashr_i64 s[0:1], s[24:25], 7 // (op 517) 
s_lshr_b32 s51, s0, 1 // (op 517) 

v_lshl_add_u32 v12, s51, 8, v1 // (op 517) 

v_mul_lo_u32 v17, v12, s27 // (op 517) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 517) most significant: high of low * low
v_mul_lo_u32 v16, v12, s26 // (op 517) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 517) most significant: sum

v_mov_b32 v13, s6 // (op 517) 
v_mov_b32 v14, s7 // (op 517) 
v_add_co_u32 v16, s[0:1], v16, v13 // (op 517) least significant half
v_addc_co_u32 v17, s[0:1], v17, v14, s[0:1] // (op 517) most significant half





v_mul_lo_u32 v13, v17, 6 // (op 517) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 517) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 517) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 517) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 517) 

v_mov_b32 v12, v14 // (op 517) convert

s_mul_i32 s1, s25, 6 // (op 519) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 519) least significant
s_mul_hi_u32 s51, s24, 6 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 519) most significant: sum

s_mul_i32 s7, s26, s1 // (op 519) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 519) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 519) least significant
s_mul_hi_u32 s18, s26, s0 // (op 519) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 519) most significant: sum
s_add_u32 s7, s7, s18 // (op 519) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 519) 

s_mov_b32 s56, s28 // (op 519) reg expression
s_mov_b32 s57, s29 // (op 519) reg expression
s_mov_b32 s58, s0 // (op 519) reg expression
s_mov_b32 s59, 131072 // (op 519) call()

s_mul_hi_i32 s51, s16, s9 // (op 581) 
v_and_b32 v13, -32, v1 // (op 581) 
s_add_i32 s0, s51, s16 // (op 581) 

v_sub_u32 v14, v1, v13 // (op 581) 

s_ashr_i32 s1, s0, 31 // (op 581) 
v_lshrrev_b32 v13, 5, v1 // (op 581) 
s_cmp_eq_i32 s9, 0 // (op 581) 
s_mov_b32 s51, scc // (op 581) 
s_cmp_eq_i32 s51, 0 // (op 581) 
s_cselect_b32 s6, 0, -1 // (op 581) 

s_and_b32 s7, s10, 31 // (op 581) 
s_lshl_b32 s51, 1, s7 // (op 581) 
s_add_i32 s18, s51, s6 // (op 581) 


s_and_b32 s6, s1, s18 // (op 581) 


s_add_i32 s1, s0, s6 // (op 581) 


s_ashr_i32 s0, s1, s7 // (op 581) 


s_lshl_b32 s1, s10, 24 // (op 581) 
s_ashr_i32 s51, s1, 31 // (op 581) 

s_xor_b32 s1, s0, s51 // (op 581) 

s_sub_i32 s0, s1, s51 // (op 581) 


s_mul_hi_i32 s1, s0, s11 // (op 581) 
s_add_i32 s51, s1, s0 // (op 581) 

s_ashr_i32 s1, s51, 31 // (op 581) 
s_cmp_eq_i32 s11, 0 // (op 581) 
s_mov_b32 s6, scc // (op 581) 
s_cmp_eq_i32 s6, 0 // (op 581) 
s_cselect_b32 s7, 0, -1 // (op 581) 

s_and_b32 s6, s12, 31 // (op 581) 
s_lshl_b32 s18, 1, s6 // (op 581) 
s_add_i32 s19, s18, s7 // (op 581) 


s_and_b32 s7, s1, s19 // (op 581) 


s_add_i32 s1, s51, s7 // (op 581) 


s_ashr_i32 s7, s1, s6 // (op 581) 


s_lshl_b32 s1, s12, 24 // (op 581) 
s_ashr_i32 s6, s1, 31 // (op 581) 

s_xor_b32 s1, s7, s6 // (op 581) 

s_sub_i32 s7, s1, s6 // (op 581) 


s_mul_i32 s1, s0, s14 // (op 581) 

s_sub_u32 s0, s16, s1 // (op 581) 

s_mul_i32 s1, s7, s14 // (op 581) 
s_add_u32 s1, s1, s0 // (op 581) 


s_mul_hi_i32 s0, s16, s8 // (op 581) 
s_add_i32 s51, s0, s16 // (op 581) 

s_ashr_i32 s0, s51, 31 // (op 581) 
s_cmp_eq_i32 s8, 0 // (op 581) 
s_mov_b32 s6, scc // (op 581) 
s_cmp_eq_i32 s6, 0 // (op 581) 
s_cselect_b32 s7, 0, -1 // (op 581) 

s_and_b32 s6, s15, 31 // (op 581) 
s_lshl_b32 s18, 1, s6 // (op 581) 
s_add_i32 s19, s18, s7 // (op 581) 


s_and_b32 s7, s0, s19 // (op 581) 


s_add_i32 s0, s51, s7 // (op 581) 


s_ashr_i32 s7, s0, s6 // (op 581) 


s_lshl_b32 s0, s15, 24 // (op 581) 
s_ashr_i32 s6, s0, 31 // (op 581) 

s_xor_b32 s0, s7, s6 // (op 581) 

s_sub_i32 s7, s0, s6 // (op 581) 


s_add_u32 s18, s4, 16 // (op 581) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 581) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 581) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 581) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 581) 

s_mul_i32 s0, s14, s18 // (op 581) 

s_mul_i32 s6, s50, s0 // (op 581) 

s_mul_i32 s0, s7, s6 // (op 581) 

s_sub_u32 s6, s16, s0 // (op 581) 

s_mov_b32 s0, s6 // (op 581) 
s_mul_hi_i32 s51, s0, s13 // (op 581) 
s_add_i32 s18, s51, s0 // (op 581) 


s_ashr_i32 s0, s18, 31 // (op 581) 
s_cmp_eq_i32 s13, 0 // (op 581) 
s_mov_b32 s19, scc // (op 581) 
s_cmp_eq_i32 s19, 0 // (op 581) 
s_cselect_b32 s51, 0, -1 // (op 581) 

s_and_b32 s19, s49, 31 // (op 581) 
s_lshl_b32 s68, 1, s19 // (op 581) 
s_add_i32 s69, s68, s51 // (op 581) 


s_and_b32 s51, s0, s69 // (op 581) 


s_add_i32 s0, s18, s51 // (op 581) 


s_ashr_i32 s18, s0, s19 // (op 581) 


s_lshl_b32 s0, s49, 24 // (op 581) 
s_ashr_i32 s19, s0, 31 // (op 581) 

s_xor_b32 s0, s18, s19 // (op 581) 

s_sub_i32 s18, s0, s19 // (op 581) 


s_mul_i32 s0, s48, s14 // (op 581) 
s_add_u32 s68, s20, 32 // (op 581) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 581) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 581) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 581) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 581) 

s_sub_i32 s19, s68, s0 // (op 581) 

s_mul_i32 s51, s18, s19 // (op 581) 


s_sub_u32 s18, s6, s51 // (op 581) 


s_add_u32 s6, s18, s0 // (op 581) 


s_cmp_eq_i32 s7, 0 // (op 581) 
s_mov_b32 s0, scc // (op 581) 

s_cmp_eq_i32 s0, 0 // (op 581) 
s_cselect_b32 s7, s6, s1 // (op 581) 



v_lshl_add_u32 v15, s7, 5, v14 // (op 581) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 581) 
s_lshr_b32 s51, s0, 1 // (op 581) 

v_lshl_add_u32 v14, s51, 3, v13 // (op 581) 


v_mul_lo_u32 v13, v14, s30 // (op 581) 

v_add_u32 v14, v15, v13 // (op 581) 


s_mul_i32 s1, s30, s35 // (op 583) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 583) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 583) least significant
s_mul_hi_u32 s6, s30, s34 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 583) most significant: sum
s_add_u32 s1, s1, s6 // (op 583) most significant: sum


s_mov_b32 s60, s32 // (op 583) reg expression
s_mov_b32 s61, s33 // (op 583) reg expression
s_mov_b32 s62, s0 // (op 583) reg expression
s_mov_b32 s63, 131072 // (op 583) call()

s_mul_hi_i32 s51, s16, s9 // (op 645) 
v_and_b32 v13, 15, v1 // (op 645) 
s_add_i32 s0, s51, s16 // (op 645) 

v_and_b32 v15, 63, v1 // (op 645) 
s_ashr_i32 s1, s0, 31 // (op 645) 
v_lshrrev_b32 v16, 4, v15 // (op 645) 

s_cmp_eq_i32 s9, 0 // (op 645) 
s_mov_b32 s51, scc // (op 645) 
s_cmp_eq_i32 s51, 0 // (op 645) 
s_cselect_b32 s6, 0, -1 // (op 645) 

s_and_b32 s7, s10, 31 // (op 645) 
s_lshl_b32 s51, 1, s7 // (op 645) 
s_add_i32 s18, s51, s6 // (op 645) 


s_and_b32 s6, s1, s18 // (op 645) 


s_add_i32 s1, s0, s6 // (op 645) 


s_ashr_i32 s0, s1, s7 // (op 645) 


s_lshl_b32 s1, s10, 24 // (op 645) 
s_ashr_i32 s51, s1, 31 // (op 645) 

s_xor_b32 s1, s0, s51 // (op 645) 

s_sub_i32 s0, s1, s51 // (op 645) 


s_mul_hi_i32 s1, s0, s11 // (op 645) 
s_add_i32 s51, s1, s0 // (op 645) 

s_ashr_i32 s1, s51, 31 // (op 645) 
s_cmp_eq_i32 s11, 0 // (op 645) 
s_mov_b32 s6, scc // (op 645) 
s_cmp_eq_i32 s6, 0 // (op 645) 
s_cselect_b32 s7, 0, -1 // (op 645) 

s_and_b32 s6, s12, 31 // (op 645) 
s_lshl_b32 s18, 1, s6 // (op 645) 
s_add_i32 s19, s18, s7 // (op 645) 


s_and_b32 s7, s1, s19 // (op 645) 


s_add_i32 s1, s51, s7 // (op 645) 


s_ashr_i32 s7, s1, s6 // (op 645) 


s_lshl_b32 s1, s12, 24 // (op 645) 
s_ashr_i32 s6, s1, 31 // (op 645) 

s_xor_b32 s1, s7, s6 // (op 645) 

s_sub_i32 s7, s1, s6 // (op 645) 


s_add_u32 s18, s4, 16 // (op 645) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 645) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 645) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 645) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 4 // (op 645) 

s_mul_i32 s1, s7, s18 // (op 645) 

s_sub_i32 s51, s0, s1 // (op 645) 


s_mul_hi_i32 s0, s16, s8 // (op 645) 
s_add_i32 s1, s0, s16 // (op 645) 

s_ashr_i32 s0, s1, 31 // (op 645) 
s_cmp_eq_i32 s8, 0 // (op 645) 
s_mov_b32 s6, scc // (op 645) 
s_cmp_eq_i32 s6, 0 // (op 645) 
s_cselect_b32 s7, 0, -1 // (op 645) 

s_and_b32 s6, s15, 31 // (op 645) 
s_lshl_b32 s68, 1, s6 // (op 645) 
s_add_i32 s69, s68, s7 // (op 645) 


s_and_b32 s7, s0, s69 // (op 645) 


s_add_i32 s0, s1, s7 // (op 645) 


s_ashr_i32 s1, s0, s6 // (op 645) 


s_lshl_b32 s0, s15, 24 // (op 645) 
s_ashr_i32 s6, s0, 31 // (op 645) 

s_xor_b32 s0, s1, s6 // (op 645) 

s_sub_i32 s1, s0, s6 // (op 645) 


s_mul_i32 s0, s14, s18 // (op 645) 

s_mul_i32 s6, s50, s0 // (op 645) 

s_mul_i32 s0, s1, s6 // (op 645) 

s_sub_u32 s6, s16, s0 // (op 645) 

s_mul_hi_i32 s0, s6, s13 // (op 645) 
s_add_i32 s7, s0, s6 // (op 645) 


s_ashr_i32 s0, s7, 31 // (op 645) 
s_cmp_eq_i32 s13, 0 // (op 645) 
s_mov_b32 s6, scc // (op 645) 
s_cmp_eq_i32 s6, 0 // (op 645) 
s_cselect_b32 s18, 0, -1 // (op 645) 

s_and_b32 s6, s49, 31 // (op 645) 
s_lshl_b32 s19, 1, s6 // (op 645) 
s_add_i32 s68, s19, s18 // (op 645) 


s_and_b32 s18, s0, s68 // (op 645) 


s_add_i32 s0, s7, s18 // (op 645) 


s_ashr_i32 s7, s0, s6 // (op 645) 


s_lshl_b32 s0, s49, 24 // (op 645) 
s_ashr_i32 s6, s0, 31 // (op 645) 

s_xor_b32 s0, s7, s6 // (op 645) 

s_sub_i32 s7, s0, s6 // (op 645) 


s_cmp_eq_i32 s1, 0 // (op 645) 
s_mov_b32 s0, scc // (op 645) 

s_cmp_eq_i32 s0, 0 // (op 645) 
s_cselect_b32 s1, s7, s51 // (op 645) 



v_lshl_add_u32 v15, s1, 4, v13 // (op 645) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 645) 
s_lshr_b32 s51, s0, 1 // (op 645) 

v_lshl_add_u32 v13, s51, 3, v16 // (op 645) 


v_mul_lo_u32 v16, v13, s36 // (op 645) 
v_add_u32 v16, v16, v15 // (op 645) 


s_mul_i32 s1, s36, s35 // (op 647) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 647) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 647) least significant
s_mul_hi_u32 s6, s36, s34 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 647) most significant: sum
s_add_u32 s1, s1, s6 // (op 647) most significant: sum


s_mov_b32 s64, s38 // (op 647) reg expression
s_mov_b32 s65, s39 // (op 647) reg expression
s_mov_b32 s66, s0 // (op 647) reg expression
s_mov_b32 s67, 131072 // (op 647) call()

v_and_b32 v13, -2, v1 // (op 738) 
v_sub_u32 v15, v1, v13 // (op 738) 

v_lshlrev_b32 v13, 4, v15 // (op 738) 

v_lshrrev_b32 v15, 1, v1 // (op 738) 
v_lshl_add_u32 v17, v15, 5, v13 // (op 738) 


s_nop 0
v_readfirstlane_b32 s51, v17 // (op 738) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v13, 4, v1 // (op 790) 
v_mul_lo_u32 v15, v13, 6 // (op 790) 

v_lshrrev_b32 v13, 3, v15 // (op 790) 

v_and_b32 v15, 268435455, v1 // (op 790) 
v_lshlrev_b32 v17, 2, v15 // (op 790) 

v_add_u32 v15, v13, v17 // (op 790) 


v_and_b32 v13, -32, v1 // (op 843) 
v_sub_u32 v17, v1, v13 // (op 843) 

v_and_b32 v13, 4294967264, v1 // (op 843) 
v_add_u32 v19, v13, v17 // (op 843) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 166) 
s_and_b32 s6, s0, 4294967294 // (op 166) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 165) 
s_mov_b32 s7, s0 // (op 165) convert

s_cmp_lt_u32 s6, s7 // (op 165) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_165:
s_barrier  // (op 1031) 
s_add_u32 s6, s6, 1 // (op 167) 

s_lshl_b64 s[0:1], s[2:3], 6 // (op 421) 
v_mov_b64 v[20:21], s[0:1] // (op 421) convert


v_mov_b64 v[22:23], v[10:11] // (op 421) 
v_mov_b64 v[24:25], v[22:23] // (op 421) reg expression
s_mov_b32 m0, s51 // (op 421) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 421) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v20 // (op 421) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 421) most significant half

s_add_u32 m0, m0, 2048 // (op 421) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 421) Load value direct to lds



s_lshl_b64 s[0:1], s[2:3], 7 // (op 440) 
v_mov_b32 v13, s0 // (op 440) 
v_mov_b32 v17, s1 // (op 440) 
v_add_co_u32 v10, s[18:19], v10, v13 // (op 440) least significant half
v_addc_co_u32 v11, s[18:19], v11, v17, s[18:19] // (op 440) most significant half






v_mov_b32 v13, v12 // (op 183) 
v_mov_b32 v17, v13 // (op 183) reg expression

buffer_load_dwordx3 v[20:22], v17, s[56:59], 0 offen // (op 183) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 509) 
s_mul_i32 s19, s1, 6 // (op 509) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 509) least significant
s_mul_hi_u32 s7, s0, 6 // (op 509) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 509) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 509) 

v_add_u32 v12, v12, s0 // (op 509) 



s_barrier  // (op 1044) 
v_mov_b32 v13, v14 // (op 184) 



v_mov_b32 v17, v13 // (op 184) reg expression

s_waitcnt vmcnt(0)
ds_write_b96 v15, v[20:22] offset:8192 // (op 181) Store local data 
buffer_load_ubyte v18, v17, s[60:63], 0 offen offset: 0 // (op 184) Load value



s_lshl_b32 s7, s30, 2 // (op 573) 
v_add_u32 v14, v14, s7 // (op 573) 

s_barrier  // (op 1034) 
s_barrier  // (op 1058) 






ds_read_b64_tr_b8 v[20:21], v0 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v0 offset:256 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:2048 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:2304 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v18 offset:10240 // (op 182) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1048) 



ds_read_b96_tr_b6 v[28:30], v5 offset:8192 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v5 offset:8448 // (op 175) Transpose load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1062) 


ds_read_u8 v13, v3 offset:10240 // (op 177) Load local data 

v_mov_b64 v[34:35], s[36:37] // (op 179) convert

v_mov_b32 v17, v16 // (op 179) 
v_mov_b32 v18, v17 // (op 179) reg expression
v_mov_b32 v36, v18 // (op 179) 
buffer_load_ubyte v37, v36, s[64:67], 0 offen offset: 0 // (op 179) Load value




s_lshl_b32 s7, s36, 2 // (op 637) 
v_add_u32 v16, v16, s7 // (op 637) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:33], a[0:3], v13, v37 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 180) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 165) 
s_mov_b32 s7, s0 // (op 165) convert

s_cmp_lt_u32 s6, s7 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopTop_KLoopTail_165 if true)
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1126) 






v_cmp_eq_f32 vcc, 0.00000, v7 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v10, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 

v_cvt_f16_f32 v10, v14 // (op 52) 
v_cvt_f16_f32 v11, v15 // (op 52) 
v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 

s_mul_hi_i32 s0, s16, s9 // (op 885) 
v_lshrrev_b32 v0, 6, v1 // (op 885) 
s_add_i32 s1, s0, s16 // (op 885) 

v_and_b32 v8, 63, v1 // (op 885) 
s_ashr_i32 s0, s1, 31 // (op 885) 
v_lshrrev_b32 v3, 4, v8 // (op 885) 
s_cmp_eq_i32 s9, 0 // (op 885) 
s_mov_b32 s2, scc // (op 885) 
v_lshlrev_b32 v4, 2, v3 // (op 885) 

s_cmp_eq_i32 s2, 0 // (op 885) 
s_cselect_b32 s3, 0, -1 // (op 885) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 885) 


s_and_b32 s2, s10, 31 // (op 885) 
v_lshrrev_b32 v0, 2, v8 // (op 885) 

s_lshl_b32 s6, 1, s2 // (op 885) 
v_and_b32 v8, 3, v0 // (op 885) 

s_add_i32 s7, s6, s3 // (op 885) 


v_and_b32 v0, 3, v1 // (op 885) 
s_and_b32 s3, s0, s7 // (op 885) 


v_lshl_add_u32 v4, v8, 2, v0 // (op 885) 


s_add_i32 s0, s1, s3 // (op 885) 


s_ashr_i32 s1, s0, s2 // (op 885) 


s_lshl_b32 s0, s10, 24 // (op 885) 
s_ashr_i32 s2, s0, 31 // (op 885) 

s_xor_b32 s0, s1, s2 // (op 885) 

s_sub_i32 s1, s0, s2 // (op 885) 


s_mul_hi_i32 s0, s1, s11 // (op 885) 
s_add_i32 s2, s0, s1 // (op 885) 

s_ashr_i32 s0, s2, 31 // (op 885) 
s_cmp_eq_i32 s11, 0 // (op 885) 
s_mov_b32 s3, scc // (op 885) 
s_cmp_eq_i32 s3, 0 // (op 885) 
s_cselect_b32 s6, 0, -1 // (op 885) 

s_and_b32 s3, s12, 31 // (op 885) 
s_lshl_b32 s7, 1, s3 // (op 885) 
s_add_i32 s18, s7, s6 // (op 885) 


s_and_b32 s19, s0, s18 // (op 885) 


s_add_i32 s0, s2, s19 // (op 885) 


s_ashr_i32 s2, s0, s3 // (op 885) 


s_lshl_b32 s0, s12, 24 // (op 885) 
s_ashr_i32 s3, s0, 31 // (op 885) 

s_xor_b32 s0, s2, s3 // (op 885) 

s_sub_i32 s2, s0, s3 // (op 885) 


s_mul_i32 s0, s1, s14 // (op 885) 
s_sub_u32 s3, s16, s0 // (op 885) 

s_mul_i32 s0, s2, s14 // (op 885) 
s_add_u32 s0, s0, s3 // (op 885) 

s_mul_hi_i32 s3, s16, s8 // (op 885) 
s_add_i32 s6, s3, s16 // (op 885) 

s_ashr_i32 s3, s6, 31 // (op 885) 
s_cmp_eq_i32 s8, 0 // (op 885) 
s_mov_b32 s7, scc // (op 885) 
s_cmp_eq_i32 s7, 0 // (op 885) 
s_cselect_b32 s18, 0, -1 // (op 885) 

s_and_b32 s7, s15, 31 // (op 885) 
s_lshl_b32 s19, 1, s7 // (op 885) 
s_add_i32 s22, s19, s18 // (op 885) 


s_and_b32 s18, s3, s22 // (op 885) 


s_add_i32 s3, s6, s18 // (op 885) 


s_ashr_i32 s6, s3, s7 // (op 885) 


s_lshl_b32 s3, s15, 24 // (op 885) 
s_ashr_i32 s7, s3, 31 // (op 885) 

s_xor_b32 s3, s6, s7 // (op 885) 

s_sub_i32 s6, s3, s7 // (op 885) 


s_add_u32 s18, s4, 16 // (op 885) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 885) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 885) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 885) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 4 // (op 885) 

s_mul_i32 s3, s14, s18 // (op 885) 
s_mul_i32 s7, s50, s3 // (op 885) 

s_mul_i32 s3, s6, s7 // (op 885) 

s_sub_u32 s7, s16, s3 // (op 885) 

s_mov_b32 s3, s7 // (op 885) 
s_mul_hi_i32 s22, s3, s13 // (op 885) 
s_add_i32 s23, s22, s3 // (op 885) 


s_ashr_i32 s3, s23, 31 // (op 885) 
s_cmp_eq_i32 s13, 0 // (op 885) 
s_mov_b32 s22, scc // (op 885) 
s_cmp_eq_i32 s22, 0 // (op 885) 
s_cselect_b32 s24, 0, -1 // (op 885) 

s_and_b32 s22, s49, 31 // (op 885) 
s_lshl_b32 s25, 1, s22 // (op 885) 
s_add_i32 s26, s25, s24 // (op 885) 


s_and_b32 s24, s3, s26 // (op 885) 


s_add_i32 s3, s23, s24 // (op 885) 


s_ashr_i32 s23, s3, s22 // (op 885) 


s_lshl_b32 s3, s49, 24 // (op 885) 
s_ashr_i32 s22, s3, 31 // (op 885) 

s_xor_b32 s3, s23, s22 // (op 885) 

s_sub_i32 s23, s3, s22 // (op 885) 


s_mul_i32 s3, s48, s14 // (op 885) 
s_add_u32 s24, s20, 32 // (op 885) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 885) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 885) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 885) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 5 // (op 885) 

s_sub_i32 s22, s24, s3 // (op 885) 

s_mul_i32 s24, s23, s22 // (op 885) 

s_sub_u32 s22, s7, s24 // (op 885) 


s_add_u32 s7, s22, s3 // (op 885) 


s_cmp_eq_i32 s6, 0 // (op 885) 
s_mov_b32 s3, scc // (op 885) 

s_cmp_eq_i32 s3, 0 // (op 885) 
s_cselect_b32 s6, s7, s0 // (op 885) 


s_mul_i32 s0, s2, s18 // (op 885) 


v_lshl_add_u32 v0, s6, 5, v3 // (op 885) 


s_sub_i32 s2, s1, s0 // (op 885) 


s_cmp_eq_i32 s3, 0 // (op 885) 
s_cselect_b32 s22, s23, s2 // (op 885) 



v_lshl_add_u32 v3, s22, 4, v4 // (op 885) 


v_mul_lo_u32 v8, v3, s40 // (op 885) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 885) 


s_mul_i32 s7, s40, s5 // (op 887) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 887) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 887) least significant
s_mul_hi_u32 s1, s40, s4 // (op 887) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 887) most significant: sum
s_add_u32 s7, s7, s1 // (op 887) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 887) 

s_mov_b32 s0, s42 // (op 887) reg expression
s_mov_b32 s1, s43 // (op 887) reg expression
s_mov_b32 s2, s18 // (op 887) reg expression
s_mov_b32 s3, 131072 // (op 887) call()


v_pack_b32_f16 v4, v10, v11 // (op 54) 
v_pack_b32_f16 v5, v12, v13 // (op 54) 


buffer_store_dwordx2 v[4:5], v3, s[0:3], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalFalse_0_beta_125:
s_mul_hi_i32 s6, s16, s9 // (op 708) 
v_lshrrev_b32 v0, 6, v1 // (op 708) 
s_add_i32 s7, s6, s16 // (op 708) 

v_and_b32 v8, 63, v1 // (op 708) 
s_ashr_i32 s6, s7, 31 // (op 708) 
v_lshrrev_b32 v3, 4, v8 // (op 708) 
s_cmp_eq_i32 s9, 0 // (op 708) 
s_mov_b32 s18, scc // (op 708) 
v_lshlrev_b32 v4, 2, v3 // (op 708) 

s_cmp_eq_i32 s18, 0 // (op 708) 
s_cselect_b32 s19, 0, -1 // (op 708) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 708) 


s_and_b32 s18, s10, 31 // (op 708) 
v_lshrrev_b32 v0, 2, v8 // (op 708) 

s_lshl_b32 s22, 1, s18 // (op 708) 
v_and_b32 v8, 3, v0 // (op 708) 

s_add_i32 s23, s22, s19 // (op 708) 


v_and_b32 v0, 3, v1 // (op 708) 
s_and_b32 s19, s6, s23 // (op 708) 


v_lshl_add_u32 v4, v8, 2, v0 // (op 708) 


s_add_i32 s6, s7, s19 // (op 708) 


s_ashr_i32 s7, s6, s18 // (op 708) 


s_lshl_b32 s6, s10, 24 // (op 708) 
s_ashr_i32 s18, s6, 31 // (op 708) 

s_xor_b32 s6, s7, s18 // (op 708) 

s_sub_i32 s7, s6, s18 // (op 708) 


s_mul_hi_i32 s6, s7, s11 // (op 708) 
s_add_i32 s18, s6, s7 // (op 708) 

s_ashr_i32 s6, s18, 31 // (op 708) 
s_cmp_eq_i32 s11, 0 // (op 708) 
s_mov_b32 s19, scc // (op 708) 
s_cmp_eq_i32 s19, 0 // (op 708) 
s_cselect_b32 s22, 0, -1 // (op 708) 

s_and_b32 s19, s12, 31 // (op 708) 
s_lshl_b32 s23, 1, s19 // (op 708) 
s_add_i32 s24, s23, s22 // (op 708) 


s_and_b32 s22, s6, s24 // (op 708) 


s_add_i32 s6, s18, s22 // (op 708) 


s_ashr_i32 s18, s6, s19 // (op 708) 


s_lshl_b32 s6, s12, 24 // (op 708) 
s_ashr_i32 s19, s6, 31 // (op 708) 

s_xor_b32 s6, s18, s19 // (op 708) 

s_sub_i32 s18, s6, s19 // (op 708) 


s_mul_i32 s6, s7, s14 // (op 708) 
s_sub_u32 s19, s16, s6 // (op 708) 

s_mul_i32 s6, s18, s14 // (op 708) 
s_add_u32 s6, s6, s19 // (op 708) 

s_mul_hi_i32 s19, s16, s8 // (op 708) 
s_add_i32 s22, s19, s16 // (op 708) 

s_ashr_i32 s19, s22, 31 // (op 708) 
s_cmp_eq_i32 s8, 0 // (op 708) 
s_mov_b32 s23, scc // (op 708) 
s_cmp_eq_i32 s23, 0 // (op 708) 
s_cselect_b32 s24, 0, -1 // (op 708) 

s_and_b32 s23, s15, 31 // (op 708) 
s_lshl_b32 s25, 1, s23 // (op 708) 
s_add_i32 s26, s25, s24 // (op 708) 


s_and_b32 s24, s19, s26 // (op 708) 


s_add_i32 s19, s22, s24 // (op 708) 


s_ashr_i32 s22, s19, s23 // (op 708) 


s_lshl_b32 s19, s15, 24 // (op 708) 
s_ashr_i32 s23, s19, 31 // (op 708) 

s_xor_b32 s19, s22, s23 // (op 708) 

s_sub_i32 s22, s19, s23 // (op 708) 


s_add_u32 s38, s4, 16 // (op 708) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 708) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 708) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 708) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 708) 

s_mul_i32 s19, s14, s38 // (op 708) 
s_mul_i32 s23, s50, s19 // (op 708) 

s_mul_i32 s19, s22, s23 // (op 708) 

s_sub_u32 s23, s16, s19 // (op 708) 

s_mov_b32 s19, s23 // (op 708) 
s_mul_hi_i32 s24, s19, s13 // (op 708) 
s_add_i32 s25, s24, s19 // (op 708) 


s_ashr_i32 s19, s25, 31 // (op 708) 
s_cmp_eq_i32 s13, 0 // (op 708) 
s_mov_b32 s24, scc // (op 708) 
s_cmp_eq_i32 s24, 0 // (op 708) 
s_cselect_b32 s26, 0, -1 // (op 708) 

s_and_b32 s24, s49, 31 // (op 708) 
s_lshl_b32 s27, 1, s24 // (op 708) 
s_add_i32 s28, s27, s26 // (op 708) 


s_and_b32 s26, s19, s28 // (op 708) 


s_add_i32 s19, s25, s26 // (op 708) 


s_ashr_i32 s25, s19, s24 // (op 708) 


s_lshl_b32 s19, s49, 24 // (op 708) 
s_ashr_i32 s24, s19, 31 // (op 708) 

s_xor_b32 s19, s25, s24 // (op 708) 

s_sub_i32 s25, s19, s24 // (op 708) 


s_mul_i32 s19, s48, s14 // (op 708) 
s_add_u32 s26, s20, 32 // (op 708) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 708) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 708) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 708) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 708) 

s_sub_i32 s24, s26, s19 // (op 708) 

s_mul_i32 s26, s25, s24 // (op 708) 

s_sub_u32 s24, s23, s26 // (op 708) 


s_add_u32 s23, s24, s19 // (op 708) 


s_cmp_eq_i32 s22, 0 // (op 708) 
s_mov_b32 s19, scc // (op 708) 

s_cmp_eq_i32 s19, 0 // (op 708) 
s_cselect_b32 s22, s23, s6 // (op 708) 


s_mul_i32 s6, s18, s38 // (op 708) 


v_lshl_add_u32 v0, s22, 5, v3 // (op 708) 


s_sub_i32 s18, s7, s6 // (op 708) 


s_cmp_eq_i32 s19, 0 // (op 708) 
s_cselect_b32 s6, s25, s18 // (op 708) 



v_lshl_add_u32 v3, s6, 4, v4 // (op 708) 


v_mul_lo_u32 v8, v3, s44 // (op 708) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 708) 


s_mul_i32 s7, s44, s5 // (op 710) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 710) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 710) least significant
s_mul_hi_u32 s19, s44, s4 // (op 710) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 710) most significant: sum
s_add_u32 s7, s7, s19 // (op 710) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 710) 

s_mov_b32 s36, s46 // (op 710) reg expression
s_mov_b32 s37, s47 // (op 710) reg expression
s_mov_b32 s38, s18 // (op 710) reg expression
s_mov_b32 s39, 131072 // (op 710) call()





buffer_load_dwordx2 v[4:5], v3, s[36:39], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 139) 
v_lshrrev_b32 v11, 16, v4 // (op 139) 
v_cvt_f32_f16 v11, v11 // (op 139) 
v_mul_f32 v18, v7, v10 // (op 139) 
v_mul_f32 v19, v7, v11 // (op 139) 
v_cvt_f32_f16 v10, v5 // (op 139) 
v_lshrrev_b32 v11, 16, v5 // (op 139) 
v_cvt_f32_f16 v11, v11 // (op 139) 
v_mul_f32 v20, v7, v10 // (op 139) 
v_mul_f32 v21, v7, v11 // (op 139) 


v_add_f32 v10, v18, v14 // (op 135) 
v_add_f32 v11, v19, v15 // (op 135) 
v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 


v_cvt_f16_f32 v14, v10 // (op 136) 
v_cvt_f16_f32 v15, v11 // (op 136) 
v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 

s_mul_hi_i32 s6, s16, s9 // (op 902) 
v_lshrrev_b32 v0, 6, v1 // (op 902) 
s_add_i32 s7, s6, s16 // (op 902) 

v_and_b32 v8, 63, v1 // (op 902) 
s_ashr_i32 s6, s7, 31 // (op 902) 
v_lshrrev_b32 v3, 4, v8 // (op 902) 
s_cmp_eq_i32 s9, 0 // (op 902) 
s_mov_b32 s18, scc // (op 902) 
v_lshlrev_b32 v4, 2, v3 // (op 902) 

s_cmp_eq_i32 s18, 0 // (op 902) 
s_cselect_b32 s19, 0, -1 // (op 902) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 902) 


s_and_b32 s18, s10, 31 // (op 902) 
v_lshrrev_b32 v0, 2, v8 // (op 902) 

s_lshl_b32 s22, 1, s18 // (op 902) 
v_and_b32 v8, 3, v0 // (op 902) 

s_add_i32 s23, s22, s19 // (op 902) 


v_and_b32 v0, 3, v1 // (op 902) 
s_and_b32 s19, s6, s23 // (op 902) 


v_lshl_add_u32 v4, v8, 2, v0 // (op 902) 


s_add_i32 s6, s7, s19 // (op 902) 


s_ashr_i32 s7, s6, s18 // (op 902) 


s_lshl_b32 s6, s10, 24 // (op 902) 
s_ashr_i32 s18, s6, 31 // (op 902) 

s_xor_b32 s6, s7, s18 // (op 902) 

s_sub_i32 s7, s6, s18 // (op 902) 


s_mul_hi_i32 s6, s7, s11 // (op 902) 
s_add_i32 s18, s6, s7 // (op 902) 

s_ashr_i32 s6, s18, 31 // (op 902) 
s_cmp_eq_i32 s11, 0 // (op 902) 
s_mov_b32 s19, scc // (op 902) 
s_cmp_eq_i32 s19, 0 // (op 902) 
s_cselect_b32 s22, 0, -1 // (op 902) 

s_and_b32 s19, s12, 31 // (op 902) 
s_lshl_b32 s23, 1, s19 // (op 902) 
s_add_i32 s24, s23, s22 // (op 902) 


s_and_b32 s22, s6, s24 // (op 902) 


s_add_i32 s6, s18, s22 // (op 902) 


s_ashr_i32 s18, s6, s19 // (op 902) 


s_lshl_b32 s6, s12, 24 // (op 902) 
s_ashr_i32 s19, s6, 31 // (op 902) 

s_xor_b32 s6, s18, s19 // (op 902) 

s_sub_i32 s18, s6, s19 // (op 902) 


s_mul_i32 s6, s7, s14 // (op 902) 
s_sub_u32 s19, s16, s6 // (op 902) 

s_mul_i32 s6, s18, s14 // (op 902) 
s_add_u32 s6, s6, s19 // (op 902) 

s_mul_hi_i32 s19, s16, s8 // (op 902) 
s_add_i32 s22, s19, s16 // (op 902) 

s_ashr_i32 s19, s22, 31 // (op 902) 
s_cmp_eq_i32 s8, 0 // (op 902) 
s_mov_b32 s23, scc // (op 902) 
s_cmp_eq_i32 s23, 0 // (op 902) 
s_cselect_b32 s24, 0, -1 // (op 902) 

s_and_b32 s23, s15, 31 // (op 902) 
s_lshl_b32 s25, 1, s23 // (op 902) 
s_add_i32 s26, s25, s24 // (op 902) 


s_and_b32 s24, s19, s26 // (op 902) 


s_add_i32 s19, s22, s24 // (op 902) 


s_ashr_i32 s22, s19, s23 // (op 902) 


s_lshl_b32 s19, s15, 24 // (op 902) 
s_ashr_i32 s23, s19, 31 // (op 902) 

s_xor_b32 s19, s22, s23 // (op 902) 

s_sub_i32 s22, s19, s23 // (op 902) 


s_add_u32 s38, s4, 16 // (op 902) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 902) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 902) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 902) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 4 // (op 902) 

s_mul_i32 s19, s14, s38 // (op 902) 
s_mul_i32 s23, s50, s19 // (op 902) 

s_mul_i32 s19, s22, s23 // (op 902) 

s_sub_u32 s23, s16, s19 // (op 902) 

s_mov_b32 s19, s23 // (op 902) 
s_mul_hi_i32 s24, s19, s13 // (op 902) 
s_add_i32 s25, s24, s19 // (op 902) 


s_ashr_i32 s19, s25, 31 // (op 902) 
s_cmp_eq_i32 s13, 0 // (op 902) 
s_mov_b32 s24, scc // (op 902) 
s_cmp_eq_i32 s24, 0 // (op 902) 
s_cselect_b32 s26, 0, -1 // (op 902) 

s_and_b32 s24, s49, 31 // (op 902) 
s_lshl_b32 s27, 1, s24 // (op 902) 
s_add_i32 s28, s27, s26 // (op 902) 


s_and_b32 s26, s19, s28 // (op 902) 


s_add_i32 s19, s25, s26 // (op 902) 


s_ashr_i32 s25, s19, s24 // (op 902) 


s_lshl_b32 s19, s49, 24 // (op 902) 
s_ashr_i32 s24, s19, 31 // (op 902) 

s_xor_b32 s19, s25, s24 // (op 902) 

s_sub_i32 s25, s19, s24 // (op 902) 


s_mul_i32 s19, s48, s14 // (op 902) 
s_add_u32 s26, s20, 32 // (op 902) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 902) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 902) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 902) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 5 // (op 902) 

s_sub_i32 s24, s26, s19 // (op 902) 

s_mul_i32 s26, s25, s24 // (op 902) 

s_sub_u32 s24, s23, s26 // (op 902) 


s_add_u32 s23, s24, s19 // (op 902) 


s_cmp_eq_i32 s22, 0 // (op 902) 
s_mov_b32 s19, scc // (op 902) 

s_cmp_eq_i32 s19, 0 // (op 902) 
s_cselect_b32 s22, s23, s6 // (op 902) 


s_mul_i32 s6, s18, s38 // (op 902) 


v_lshl_add_u32 v0, s22, 5, v3 // (op 902) 


s_sub_i32 s18, s7, s6 // (op 902) 


s_cmp_eq_i32 s19, 0 // (op 902) 
s_cselect_b32 s6, s25, s18 // (op 902) 



v_lshl_add_u32 v3, s6, 4, v4 // (op 902) 


v_mul_lo_u32 v8, v3, s40 // (op 902) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 902) 


s_mul_i32 s7, s40, s5 // (op 904) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 904) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 904) least significant
s_mul_hi_u32 s19, s40, s4 // (op 904) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 904) most significant: sum
s_add_u32 s7, s7, s19 // (op 904) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 904) 

s_mov_b32 s0, s42 // (op 904) reg expression
s_mov_b32 s1, s43 // (op 904) reg expression
s_mov_b32 s2, s18 // (op 904) reg expression
s_mov_b32 s3, 131072 // (op 904) call()


v_pack_b32_f16 v4, v14, v15 // (op 137) 
v_pack_b32_f16 v5, v16, v17 // (op 137) 


buffer_store_dwordx2 v[4:5], v3, s[0:3], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.LRR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end:
.size RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_, .LRR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM__end-RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: B8
    DataTypeA: B8
    DataTypeB: B6
    DestDataType: h
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
    macrotile: [32, 16, 128]
    threads: [128, 1, 1]
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
  - .name: RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 50
    .agpr_count: 4
    .max_flat_workgroup_size: 128
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

