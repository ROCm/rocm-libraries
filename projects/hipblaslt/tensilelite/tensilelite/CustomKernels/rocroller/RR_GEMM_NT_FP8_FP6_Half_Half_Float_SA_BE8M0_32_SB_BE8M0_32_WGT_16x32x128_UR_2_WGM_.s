// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_,@function
RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__exec_begin:


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


v_and_b32 v0, 48, v1 // (op 769) 
v_and_b32 v3, 63, v1 // (op 769) 
v_sub_u32 v4, v3, v0 // (op 769) 

v_lshrrev_b32 v3, 1, v4 // (op 769) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 769) 


v_and_b32 v0, 1, v4 // (op 769) 

v_lshlrev_b32 v3, 3, v0 // (op 769) 

v_lshrrev_b32 v0, 7, v1 // (op 769) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 769) 


v_add_u32 v0, v5, v4 // (op 769) 


v_and_b32 v3, 63, v1 // (op 822) 
v_and_b32 v4, 48, v1 // (op 822) 
v_sub_u32 v5, v3, v4 // (op 822) 

v_lshrrev_b32 v4, 4, v3 // (op 822) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 822) 


v_lshrrev_b32 v4, 6, v1 // (op 822) 
v_and_b32 v5, 1, v4 // (op 822) 

v_lshlrev_b32 v4, 4, v5 // (op 822) 

v_lshl_add_u32 v5, v3, 5, v4 // (op 822) 


v_mul_lo_u32 v3, v5, 6 // (op 822) 
v_lshrrev_b32 v4, 3, v3 // (op 822) 

v_lshrrev_b32 v3, 4, v5 // (op 822) 

v_lshlrev_b32 v5, 2, v3 // (op 822) 

v_add_u32 v3, v4, v5 // (op 822) 


v_lshrrev_b32 v4, 6, v1 // (op 875) 
v_and_b32 v5, 1, v4 // (op 875) 

v_and_b32 v4, 15, v1 // (op 875) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 875) 


v_and_b32 v4, 63, v1 // (op 875) 
v_lshrrev_b32 v5, 4, v4 // (op 875) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 875) 


v_and_b32 v5, 48, v1 // (op 948) 
v_and_b32 v6, 63, v1 // (op 948) 
v_sub_u32 v7, v6, v5 // (op 948) 

v_lshrrev_b32 v6, 1, v7 // (op 948) 
v_add_lshl_u32 v8, v5, v6, 4 // (op 948) 


v_and_b32 v5, 1, v7 // (op 948) 

v_lshlrev_b32 v6, 3, v5 // (op 948) 

v_lshrrev_b32 v5, 7, v1 // (op 948) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 948) 


v_add_u32 v5, v8, v7 // (op 948) 


v_and_b32 v6, 63, v1 // (op 972) 
v_and_b32 v7, 48, v1 // (op 972) 
v_sub_u32 v8, v6, v7 // (op 972) 

v_lshrrev_b32 v7, 4, v6 // (op 972) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 972) 


v_lshrrev_b32 v7, 6, v1 // (op 972) 
v_and_b32 v8, 1, v7 // (op 972) 

v_lshlrev_b32 v7, 4, v8 // (op 972) 

v_lshl_add_u32 v8, v6, 5, v7 // (op 972) 


v_mul_lo_u32 v6, v8, 6 // (op 972) 
v_lshrrev_b32 v7, 3, v6 // (op 972) 

v_lshrrev_b32 v6, 4, v8 // (op 972) 

v_lshlrev_b32 v8, 2, v6 // (op 972) 

v_add_u32 v6, v7, v8 // (op 972) 


v_lshrrev_b32 v7, 6, v1 // (op 996) 
v_and_b32 v8, 1, v7 // (op 996) 

v_and_b32 v7, 15, v1 // (op 996) 
v_lshl_add_u32 v9, v8, 4, v7 // (op 996) 


v_and_b32 v7, 63, v1 // (op 996) 
v_lshrrev_b32 v8, 4, v7 // (op 996) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 996) 


v_mov_b32 v8, s6 // (op 2) Move value
v_mov_b32 v9, s7 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
s_mul_hi_i32 s0, s16, s9 // (op 480) 
s_add_i32 s1, s0, s16 // (op 480) 

s_ashr_i32 s0, s1, 31 // (op 480) 
s_cmp_eq_i32 s9, 0 // (op 480) 
s_mov_b32 s6, scc // (op 480) 
s_cmp_eq_i32 s6, 0 // (op 480) 
s_cselect_b32 s7, 0, -1 // (op 480) 

s_and_b32 s6, s10, 31 // (op 480) 
s_lshl_b32 s18, 1, s6 // (op 480) 
s_add_i32 s19, s18, s7 // (op 480) 


s_and_b32 s7, s0, s19 // (op 480) 


s_add_i32 s0, s1, s7 // (op 480) 


s_ashr_i32 s1, s0, s6 // (op 480) 


s_lshl_b32 s0, s10, 24 // (op 480) 
s_ashr_i32 s6, s0, 31 // (op 480) 

s_xor_b32 s0, s1, s6 // (op 480) 

s_sub_i32 s1, s0, s6 // (op 480) 


s_mul_hi_i32 s0, s1, s11 // (op 480) 
s_add_i32 s6, s0, s1 // (op 480) 

s_ashr_i32 s0, s6, 31 // (op 480) 
s_cmp_eq_i32 s11, 0 // (op 480) 
s_mov_b32 s7, scc // (op 480) 
s_cmp_eq_i32 s7, 0 // (op 480) 
s_cselect_b32 s18, 0, -1 // (op 480) 

s_and_b32 s7, s12, 31 // (op 480) 
s_lshl_b32 s19, 1, s7 // (op 480) 
s_add_i32 s51, s19, s18 // (op 480) 


s_and_b32 s18, s0, s51 // (op 480) 


s_add_i32 s0, s6, s18 // (op 480) 


s_ashr_i32 s6, s0, s7 // (op 480) 


s_lshl_b32 s0, s12, 24 // (op 480) 
s_ashr_i32 s7, s0, 31 // (op 480) 

s_xor_b32 s0, s6, s7 // (op 480) 

s_sub_i32 s6, s0, s7 // (op 480) 


s_mul_i32 s0, s1, s14 // (op 480) 

s_sub_u32 s1, s16, s0 // (op 480) 

s_mul_i32 s0, s6, s14 // (op 480) 
s_add_u32 s0, s0, s1 // (op 480) 


s_mul_hi_i32 s1, s16, s8 // (op 480) 
s_add_i32 s6, s1, s16 // (op 480) 

s_ashr_i32 s1, s6, 31 // (op 480) 
s_cmp_eq_i32 s8, 0 // (op 480) 
s_mov_b32 s7, scc // (op 480) 
s_cmp_eq_i32 s7, 0 // (op 480) 
s_cselect_b32 s18, 0, -1 // (op 480) 

s_and_b32 s7, s15, 31 // (op 480) 
s_lshl_b32 s19, 1, s7 // (op 480) 
s_add_i32 s51, s19, s18 // (op 480) 


s_and_b32 s18, s1, s51 // (op 480) 


s_add_i32 s1, s6, s18 // (op 480) 


s_ashr_i32 s6, s1, s7 // (op 480) 


s_lshl_b32 s1, s15, 24 // (op 480) 
s_ashr_i32 s7, s1, 31 // (op 480) 

s_xor_b32 s1, s6, s7 // (op 480) 

s_sub_i32 s6, s1, s7 // (op 480) 


s_add_u32 s18, s4, 32 // (op 480) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 480) End of Int64 add, unlocking SCC
s_sub_u32 s52, s18, 1 // (op 480) least significant half; sets scc
s_subb_u32 s53, s19, 0 // (op 480) most significant half; uses scc

s_ashr_i64 s[18:19], s[52:53], 5 // (op 480) 

s_mul_i32 s1, s14, s18 // (op 480) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s7, s50, s1 // (op 480) 

s_mul_i32 s1, s6, s7 // (op 480) 

s_sub_u32 s7, s16, s1 // (op 480) 

s_mov_b32 s1, s7 // (op 480) 
s_mul_hi_i32 s18, s1, s13 // (op 480) 
s_add_i32 s19, s18, s1 // (op 480) 


s_ashr_i32 s1, s19, 31 // (op 480) 
s_cmp_eq_i32 s13, 0 // (op 480) 
s_mov_b32 s18, scc // (op 480) 
s_cmp_eq_i32 s18, 0 // (op 480) 
s_cselect_b32 s51, 0, -1 // (op 480) 

s_and_b32 s18, s49, 31 // (op 480) 
s_lshl_b32 s52, 1, s18 // (op 480) 
s_add_i32 s53, s52, s51 // (op 480) 


s_and_b32 s51, s1, s53 // (op 480) 


s_add_i32 s1, s19, s51 // (op 480) 


s_ashr_i32 s19, s1, s18 // (op 480) 


s_lshl_b32 s1, s49, 24 // (op 480) 
s_ashr_i32 s18, s1, 31 // (op 480) 

s_xor_b32 s1, s19, s18 // (op 480) 

s_sub_i32 s19, s1, s18 // (op 480) 


s_mul_i32 s1, s48, s14 // (op 480) 
s_add_u32 s52, s20, 16 // (op 480) Start of Int64 add, locking SCC
s_addc_u32 s53, s21, 0 // (op 480) End of Int64 add, unlocking SCC
s_sub_u32 s54, s52, 1 // (op 480) least significant half; sets scc
s_subb_u32 s55, s53, 0 // (op 480) most significant half; uses scc

s_ashr_i64 s[52:53], s[54:55], 4 // (op 480) 

s_sub_i32 s18, s52, s1 // (op 480) 

s_mul_i32 s51, s19, s18 // (op 480) 


s_sub_u32 s18, s7, s51 // (op 480) 


s_add_u32 s7, s18, s1 // (op 480) 


s_cmp_eq_i32 s6, 0 // (op 480) 
s_mov_b32 s1, scc // (op 480) 

s_cmp_eq_i32 s1, 0 // (op 480) 
s_cselect_b32 s6, s7, s0 // (op 480) 



s_lshl_b32 s7, s6, 4 // (op 480) 

s_mov_b32 s1, 0 // (op 480) convert
s_mov_b32 s0, s7 // (op 480) convert

v_mul_lo_u32 v11, v1, s3 // (op 480) most significant: low of low * high
v_mul_hi_u32 v12, v1, s2 // (op 480) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 480) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 480) most significant: sum

v_mov_b32 v12, s0 // (op 480) 
v_mov_b32 v13, s1 // (op 480) 
v_add_co_u32 v14, s[6:7], v12, v10 // (op 480) least significant half
v_addc_co_u32 v15, s[6:7], v13, v11, s[6:7] // (op 480) most significant half





s_mul_i32 s1, s2, s25 // (op 482) most significant: low of low * high
s_mul_i32 s6, s3, s24 // (op 482) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 482) least significant
s_mul_hi_u32 s7, s2, s24 // (op 482) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 482) most significant: sum
s_add_u32 s1, s1, s7 // (op 482) most significant: sum


s_mov_b32 s52, s22 // (op 482) reg expression
s_mov_b32 s53, s23 // (op 482) reg expression
s_mov_b32 s54, s0 // (op 482) reg expression
s_mov_b32 s55, 131072 // (op 482) call()

s_mul_hi_i32 s51, s16, s9 // (op 499) 
s_add_i32 s0, s51, s16 // (op 499) 

s_ashr_i32 s1, s0, 31 // (op 499) 
s_cmp_eq_i32 s9, 0 // (op 499) 
s_mov_b32 s51, scc // (op 499) 
s_cmp_eq_i32 s51, 0 // (op 499) 
s_cselect_b32 s6, 0, -1 // (op 499) 

s_and_b32 s7, s10, 31 // (op 499) 
s_lshl_b32 s51, 1, s7 // (op 499) 
s_add_i32 s18, s51, s6 // (op 499) 


s_and_b32 s6, s1, s18 // (op 499) 


s_add_i32 s1, s0, s6 // (op 499) 


s_ashr_i32 s0, s1, s7 // (op 499) 


s_lshl_b32 s1, s10, 24 // (op 499) 
s_ashr_i32 s51, s1, 31 // (op 499) 

s_xor_b32 s1, s0, s51 // (op 499) 

s_sub_i32 s0, s1, s51 // (op 499) 


s_mul_hi_i32 s1, s0, s11 // (op 499) 
s_add_i32 s51, s1, s0 // (op 499) 

s_ashr_i32 s1, s51, 31 // (op 499) 
s_cmp_eq_i32 s11, 0 // (op 499) 
s_mov_b32 s6, scc // (op 499) 
s_cmp_eq_i32 s6, 0 // (op 499) 
s_cselect_b32 s7, 0, -1 // (op 499) 

s_and_b32 s6, s12, 31 // (op 499) 
s_lshl_b32 s18, 1, s6 // (op 499) 
s_add_i32 s19, s18, s7 // (op 499) 


s_and_b32 s7, s1, s19 // (op 499) 


s_add_i32 s1, s51, s7 // (op 499) 


s_ashr_i32 s7, s1, s6 // (op 499) 


s_lshl_b32 s1, s12, 24 // (op 499) 
s_ashr_i32 s6, s1, 31 // (op 499) 

s_xor_b32 s1, s7, s6 // (op 499) 

s_sub_i32 s7, s1, s6 // (op 499) 


s_mul_i32 s1, s0, s14 // (op 499) 

s_sub_u32 s0, s16, s1 // (op 499) 

s_mul_i32 s1, s7, s14 // (op 499) 
s_add_u32 s1, s1, s0 // (op 499) 


s_mul_hi_i32 s0, s16, s8 // (op 499) 
s_add_i32 s51, s0, s16 // (op 499) 

s_ashr_i32 s0, s51, 31 // (op 499) 
s_cmp_eq_i32 s8, 0 // (op 499) 
s_mov_b32 s6, scc // (op 499) 
s_cmp_eq_i32 s6, 0 // (op 499) 
s_cselect_b32 s7, 0, -1 // (op 499) 

s_and_b32 s6, s15, 31 // (op 499) 
s_lshl_b32 s18, 1, s6 // (op 499) 
s_add_i32 s19, s18, s7 // (op 499) 


s_and_b32 s7, s0, s19 // (op 499) 


s_add_i32 s0, s51, s7 // (op 499) 


s_ashr_i32 s7, s0, s6 // (op 499) 


s_lshl_b32 s0, s15, 24 // (op 499) 
s_ashr_i32 s6, s0, 31 // (op 499) 

s_xor_b32 s0, s7, s6 // (op 499) 

s_sub_i32 s7, s0, s6 // (op 499) 


s_add_u32 s18, s4, 32 // (op 499) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 499) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 499) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 499) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 499) 

s_mul_i32 s0, s14, s18 // (op 499) 

s_mul_i32 s6, s50, s0 // (op 499) 

s_mul_i32 s0, s7, s6 // (op 499) 

s_sub_u32 s6, s16, s0 // (op 499) 

s_mov_b32 s0, s6 // (op 499) 
s_mul_hi_i32 s51, s0, s13 // (op 499) 
s_add_i32 s18, s51, s0 // (op 499) 


s_ashr_i32 s0, s18, 31 // (op 499) 
s_cmp_eq_i32 s13, 0 // (op 499) 
s_mov_b32 s19, scc // (op 499) 
s_cmp_eq_i32 s19, 0 // (op 499) 
s_cselect_b32 s51, 0, -1 // (op 499) 

s_and_b32 s19, s49, 31 // (op 499) 
s_lshl_b32 s56, 1, s19 // (op 499) 
s_add_i32 s57, s56, s51 // (op 499) 


s_and_b32 s51, s0, s57 // (op 499) 


s_add_i32 s0, s18, s51 // (op 499) 


s_ashr_i32 s18, s0, s19 // (op 499) 


s_lshl_b32 s0, s49, 24 // (op 499) 
s_ashr_i32 s19, s0, 31 // (op 499) 

s_xor_b32 s0, s18, s19 // (op 499) 

s_sub_i32 s18, s0, s19 // (op 499) 


s_mul_i32 s0, s48, s14 // (op 499) 
s_add_u32 s56, s20, 16 // (op 499) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 499) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 499) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 499) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 499) 

s_sub_i32 s19, s56, s0 // (op 499) 

s_mul_i32 s51, s18, s19 // (op 499) 


s_sub_u32 s18, s6, s51 // (op 499) 


s_add_u32 s6, s18, s0 // (op 499) 


s_cmp_eq_i32 s7, 0 // (op 499) 
s_mov_b32 s0, scc // (op 499) 

s_cmp_eq_i32 s0, 0 // (op 499) 
s_cselect_b32 s7, s6, s1 // (op 499) 



s_lshl_b32 s6, s7, 4 // (op 499) 

s_mov_b32 s1, 0 // (op 499) convert
s_mov_b32 s0, s6 // (op 499) convert

v_mul_lo_u32 v11, v1, s3 // (op 499) most significant: low of low * high
v_mul_hi_u32 v12, v1, s2 // (op 499) most significant: high of low * low
v_mul_lo_u32 v10, v1, s2 // (op 499) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 499) most significant: sum

v_mov_b32 v12, s0 // (op 499) 
v_mov_b32 v13, s1 // (op 499) 
v_add_co_u32 v16, s[6:7], v12, v10 // (op 499) least significant half
v_addc_co_u32 v17, s[6:7], v13, v11, s[6:7] // (op 499) most significant half





s_mul_i32 s1, s2, s25 // (op 501) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 501) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 501) least significant
s_mul_hi_u32 s6, s2, s24 // (op 501) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 501) most significant: sum
s_add_u32 s1, s1, s6 // (op 501) most significant: sum


s_mov_b32 s52, s22 // (op 501) reg expression
s_mov_b32 s53, s23 // (op 501) reg expression
s_mov_b32 s54, s0 // (op 501) reg expression
s_mov_b32 s55, 131072 // (op 501) call()

s_mul_hi_i32 s51, s16, s9 // (op 544) 
s_add_i32 s0, s51, s16 // (op 544) 

s_ashr_i32 s1, s0, 31 // (op 544) 
s_cmp_eq_i32 s9, 0 // (op 544) 
s_mov_b32 s51, scc // (op 544) 
s_cmp_eq_i32 s51, 0 // (op 544) 
s_cselect_b32 s6, 0, -1 // (op 544) 

s_and_b32 s7, s10, 31 // (op 544) 
s_lshl_b32 s51, 1, s7 // (op 544) 
s_add_i32 s18, s51, s6 // (op 544) 


s_and_b32 s6, s1, s18 // (op 544) 


s_add_i32 s1, s0, s6 // (op 544) 


s_ashr_i32 s0, s1, s7 // (op 544) 


s_lshl_b32 s1, s10, 24 // (op 544) 
s_ashr_i32 s51, s1, 31 // (op 544) 

s_xor_b32 s1, s0, s51 // (op 544) 

s_sub_i32 s0, s1, s51 // (op 544) 


s_mul_hi_i32 s1, s0, s11 // (op 544) 
s_add_i32 s51, s1, s0 // (op 544) 

s_ashr_i32 s1, s51, 31 // (op 544) 
s_cmp_eq_i32 s11, 0 // (op 544) 
s_mov_b32 s6, scc // (op 544) 
s_cmp_eq_i32 s6, 0 // (op 544) 
s_cselect_b32 s7, 0, -1 // (op 544) 

s_and_b32 s6, s12, 31 // (op 544) 
s_lshl_b32 s18, 1, s6 // (op 544) 
s_add_i32 s19, s18, s7 // (op 544) 


s_and_b32 s7, s1, s19 // (op 544) 


s_add_i32 s1, s51, s7 // (op 544) 


s_ashr_i32 s7, s1, s6 // (op 544) 


s_lshl_b32 s1, s12, 24 // (op 544) 
s_ashr_i32 s6, s1, 31 // (op 544) 

s_xor_b32 s1, s7, s6 // (op 544) 

s_sub_i32 s7, s1, s6 // (op 544) 


s_add_u32 s18, s4, 32 // (op 544) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 544) End of Int64 add, unlocking SCC
s_sub_u32 s56, s18, 1 // (op 544) least significant half; sets scc
s_subb_u32 s57, s19, 0 // (op 544) most significant half; uses scc

s_ashr_i64 s[18:19], s[56:57], 5 // (op 544) 

s_mul_i32 s1, s7, s18 // (op 544) 

s_sub_i32 s51, s0, s1 // (op 544) 


s_mul_hi_i32 s0, s16, s8 // (op 544) 
s_add_i32 s1, s0, s16 // (op 544) 

s_ashr_i32 s0, s1, 31 // (op 544) 
s_cmp_eq_i32 s8, 0 // (op 544) 
s_mov_b32 s6, scc // (op 544) 
s_cmp_eq_i32 s6, 0 // (op 544) 
s_cselect_b32 s7, 0, -1 // (op 544) 

s_and_b32 s6, s15, 31 // (op 544) 
s_lshl_b32 s56, 1, s6 // (op 544) 
s_add_i32 s57, s56, s7 // (op 544) 


s_and_b32 s7, s0, s57 // (op 544) 


s_add_i32 s0, s1, s7 // (op 544) 


s_ashr_i32 s1, s0, s6 // (op 544) 


s_lshl_b32 s0, s15, 24 // (op 544) 
s_ashr_i32 s6, s0, 31 // (op 544) 

s_xor_b32 s0, s1, s6 // (op 544) 

s_sub_i32 s1, s0, s6 // (op 544) 


s_mul_i32 s0, s14, s18 // (op 544) 

s_mul_i32 s6, s50, s0 // (op 544) 

s_mul_i32 s0, s1, s6 // (op 544) 

s_sub_u32 s6, s16, s0 // (op 544) 

s_mul_hi_i32 s0, s6, s13 // (op 544) 
s_add_i32 s7, s0, s6 // (op 544) 


s_ashr_i32 s0, s7, 31 // (op 544) 
s_cmp_eq_i32 s13, 0 // (op 544) 
s_mov_b32 s6, scc // (op 544) 
s_cmp_eq_i32 s6, 0 // (op 544) 
s_cselect_b32 s18, 0, -1 // (op 544) 

s_and_b32 s6, s49, 31 // (op 544) 
s_lshl_b32 s19, 1, s6 // (op 544) 
s_add_i32 s56, s19, s18 // (op 544) 


s_and_b32 s18, s0, s56 // (op 544) 


s_add_i32 s0, s7, s18 // (op 544) 


s_ashr_i32 s7, s0, s6 // (op 544) 


s_lshl_b32 s0, s49, 24 // (op 544) 
s_ashr_i32 s6, s0, 31 // (op 544) 

s_xor_b32 s0, s7, s6 // (op 544) 

s_sub_i32 s7, s0, s6 // (op 544) 


s_cmp_eq_i32 s1, 0 // (op 544) 
s_mov_b32 s0, scc // (op 544) 

s_cmp_eq_i32 s0, 0 // (op 544) 
s_cselect_b32 s1, s7, s51 // (op 544) 



s_lshl_b32 s0, s1, 5 // (op 544) 

s_mov_b32 s7, 0 // (op 544) convert
s_mov_b32 s6, s0 // (op 544) convert

v_mul_lo_u32 v11, v1, s27 // (op 544) most significant: low of low * high
v_mul_hi_u32 v12, v1, s26 // (op 544) most significant: high of low * low
v_mul_lo_u32 v10, v1, s26 // (op 544) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 544) most significant: sum

v_mov_b32 v12, s6 // (op 544) 
v_mov_b32 v13, s7 // (op 544) 
v_add_co_u32 v10, s[0:1], v10, v12 // (op 544) least significant half
v_addc_co_u32 v11, s[0:1], v11, v13, s[0:1] // (op 544) most significant half




v_mul_lo_u32 v13, v11, 6 // (op 544) most significant: low of high * low
v_mul_hi_u32 v18, v10, 6 // (op 544) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 544) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 544) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 544) 

v_mov_b32 v12, v10 // (op 544) convert

s_mul_i32 s1, s25, 6 // (op 546) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 546) least significant
s_mul_hi_u32 s51, s24, 6 // (op 546) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 546) most significant: sum

s_mul_i32 s7, s26, s1 // (op 546) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 546) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 546) least significant
s_mul_hi_u32 s18, s26, s0 // (op 546) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 546) most significant: sum
s_add_u32 s7, s7, s18 // (op 546) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 546) 

s_mov_b32 s56, s28 // (op 546) reg expression
s_mov_b32 s57, s29 // (op 546) reg expression
s_mov_b32 s58, s0 // (op 546) reg expression
s_mov_b32 s59, 131072 // (op 546) call()

s_mul_hi_i32 s51, s16, s9 // (op 563) 
s_add_i32 s0, s51, s16 // (op 563) 

s_ashr_i32 s1, s0, 31 // (op 563) 
s_cmp_eq_i32 s9, 0 // (op 563) 
s_mov_b32 s51, scc // (op 563) 
s_cmp_eq_i32 s51, 0 // (op 563) 
s_cselect_b32 s6, 0, -1 // (op 563) 

s_and_b32 s7, s10, 31 // (op 563) 
s_lshl_b32 s51, 1, s7 // (op 563) 
s_add_i32 s18, s51, s6 // (op 563) 


s_and_b32 s6, s1, s18 // (op 563) 


s_add_i32 s1, s0, s6 // (op 563) 


s_ashr_i32 s0, s1, s7 // (op 563) 


s_lshl_b32 s1, s10, 24 // (op 563) 
s_ashr_i32 s51, s1, 31 // (op 563) 

s_xor_b32 s1, s0, s51 // (op 563) 

s_sub_i32 s0, s1, s51 // (op 563) 


s_mul_hi_i32 s1, s0, s11 // (op 563) 
s_add_i32 s51, s1, s0 // (op 563) 

s_ashr_i32 s1, s51, 31 // (op 563) 
s_cmp_eq_i32 s11, 0 // (op 563) 
s_mov_b32 s6, scc // (op 563) 
s_cmp_eq_i32 s6, 0 // (op 563) 
s_cselect_b32 s7, 0, -1 // (op 563) 

s_and_b32 s6, s12, 31 // (op 563) 
s_lshl_b32 s18, 1, s6 // (op 563) 
s_add_i32 s19, s18, s7 // (op 563) 


s_and_b32 s7, s1, s19 // (op 563) 


s_add_i32 s1, s51, s7 // (op 563) 


s_ashr_i32 s7, s1, s6 // (op 563) 


s_lshl_b32 s1, s12, 24 // (op 563) 
s_ashr_i32 s6, s1, 31 // (op 563) 

s_xor_b32 s1, s7, s6 // (op 563) 

s_sub_i32 s7, s1, s6 // (op 563) 


s_add_u32 s18, s4, 32 // (op 563) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 563) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 563) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 563) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 563) 

s_mul_i32 s1, s7, s18 // (op 563) 

s_sub_i32 s51, s0, s1 // (op 563) 


s_mul_hi_i32 s0, s16, s8 // (op 563) 
s_add_i32 s1, s0, s16 // (op 563) 

s_ashr_i32 s0, s1, 31 // (op 563) 
s_cmp_eq_i32 s8, 0 // (op 563) 
s_mov_b32 s6, scc // (op 563) 
s_cmp_eq_i32 s6, 0 // (op 563) 
s_cselect_b32 s7, 0, -1 // (op 563) 

s_and_b32 s6, s15, 31 // (op 563) 
s_lshl_b32 s60, 1, s6 // (op 563) 
s_add_i32 s61, s60, s7 // (op 563) 


s_and_b32 s7, s0, s61 // (op 563) 


s_add_i32 s0, s1, s7 // (op 563) 


s_ashr_i32 s1, s0, s6 // (op 563) 


s_lshl_b32 s0, s15, 24 // (op 563) 
s_ashr_i32 s6, s0, 31 // (op 563) 

s_xor_b32 s0, s1, s6 // (op 563) 

s_sub_i32 s1, s0, s6 // (op 563) 


s_mul_i32 s0, s14, s18 // (op 563) 

s_mul_i32 s6, s50, s0 // (op 563) 

s_mul_i32 s0, s1, s6 // (op 563) 

s_sub_u32 s6, s16, s0 // (op 563) 

s_mul_hi_i32 s0, s6, s13 // (op 563) 
s_add_i32 s7, s0, s6 // (op 563) 


s_ashr_i32 s0, s7, 31 // (op 563) 
s_cmp_eq_i32 s13, 0 // (op 563) 
s_mov_b32 s6, scc // (op 563) 
s_cmp_eq_i32 s6, 0 // (op 563) 
s_cselect_b32 s18, 0, -1 // (op 563) 

s_and_b32 s6, s49, 31 // (op 563) 
s_lshl_b32 s19, 1, s6 // (op 563) 
s_add_i32 s60, s19, s18 // (op 563) 


s_and_b32 s18, s0, s60 // (op 563) 


s_add_i32 s0, s7, s18 // (op 563) 


s_ashr_i32 s7, s0, s6 // (op 563) 


s_lshl_b32 s0, s49, 24 // (op 563) 
s_ashr_i32 s6, s0, 31 // (op 563) 

s_xor_b32 s0, s7, s6 // (op 563) 

s_sub_i32 s7, s0, s6 // (op 563) 


s_cmp_eq_i32 s1, 0 // (op 563) 
s_mov_b32 s0, scc // (op 563) 

s_cmp_eq_i32 s0, 0 // (op 563) 
s_cselect_b32 s1, s7, s51 // (op 563) 



s_lshl_b32 s0, s1, 5 // (op 563) 

s_mov_b32 s7, 0 // (op 563) convert
s_mov_b32 s6, s0 // (op 563) convert

v_mul_lo_u32 v11, v1, s27 // (op 563) most significant: low of low * high
v_mul_hi_u32 v13, v1, s26 // (op 563) most significant: high of low * low
v_mul_lo_u32 v10, v1, s26 // (op 563) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 563) most significant: sum

v_mov_b32 v13, s6 // (op 563) 
v_mov_b32 v18, s7 // (op 563) 
v_add_co_u32 v10, s[0:1], v10, v13 // (op 563) least significant half
v_addc_co_u32 v11, s[0:1], v11, v18, s[0:1] // (op 563) most significant half




v_mul_lo_u32 v19, v11, 6 // (op 563) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 563) most significant: high of low * low
v_mul_lo_u32 v18, v10, 6 // (op 563) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 563) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[18:19] // (op 563) 

v_mov_b32 v13, v10 // (op 563) convert

s_mul_i32 s1, s25, 6 // (op 565) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 565) least significant
s_mul_hi_u32 s51, s24, 6 // (op 565) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 565) most significant: sum

s_mul_i32 s7, s26, s1 // (op 565) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 565) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 565) least significant
s_mul_hi_u32 s18, s26, s0 // (op 565) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 565) most significant: sum
s_add_u32 s7, s7, s18 // (op 565) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 565) 

s_mov_b32 s56, s28 // (op 565) reg expression
s_mov_b32 s57, s29 // (op 565) reg expression
s_mov_b32 s58, s0 // (op 565) reg expression
s_mov_b32 s59, 131072 // (op 565) call()

s_mul_hi_i32 s51, s16, s9 // (op 608) 
v_lshrrev_b32 v10, 7, v1 // (op 608) 
s_add_i32 s0, s51, s16 // (op 608) 

v_and_b32 v11, 15, v1 // (op 608) 
s_ashr_i32 s1, s0, 31 // (op 608) 
v_lshl_add_u32 v18, v10, 4, v11 // (op 608) 


s_cmp_eq_i32 s9, 0 // (op 608) 
s_mov_b32 s51, scc // (op 608) 
v_and_b32 v10, 63, v1 // (op 608) 
s_cmp_eq_i32 s51, 0 // (op 608) 
s_cselect_b32 s6, 0, -1 // (op 608) 

v_lshrrev_b32 v11, 4, v10 // (op 608) 

s_and_b32 s7, s10, 31 // (op 608) 
s_lshl_b32 s51, 1, s7 // (op 608) 
s_add_i32 s18, s51, s6 // (op 608) 


s_and_b32 s6, s1, s18 // (op 608) 


s_add_i32 s1, s0, s6 // (op 608) 


s_ashr_i32 s0, s1, s7 // (op 608) 


s_lshl_b32 s1, s10, 24 // (op 608) 
s_ashr_i32 s51, s1, 31 // (op 608) 

s_xor_b32 s1, s0, s51 // (op 608) 

s_sub_i32 s0, s1, s51 // (op 608) 


s_mul_hi_i32 s1, s0, s11 // (op 608) 
s_add_i32 s51, s1, s0 // (op 608) 

s_ashr_i32 s1, s51, 31 // (op 608) 
s_cmp_eq_i32 s11, 0 // (op 608) 
s_mov_b32 s6, scc // (op 608) 
s_cmp_eq_i32 s6, 0 // (op 608) 
s_cselect_b32 s7, 0, -1 // (op 608) 

s_and_b32 s6, s12, 31 // (op 608) 
s_lshl_b32 s18, 1, s6 // (op 608) 
s_add_i32 s19, s18, s7 // (op 608) 


s_and_b32 s7, s1, s19 // (op 608) 


s_add_i32 s1, s51, s7 // (op 608) 


s_ashr_i32 s7, s1, s6 // (op 608) 


s_lshl_b32 s1, s12, 24 // (op 608) 
s_ashr_i32 s6, s1, 31 // (op 608) 

s_xor_b32 s1, s7, s6 // (op 608) 

s_sub_i32 s7, s1, s6 // (op 608) 


s_mul_i32 s1, s0, s14 // (op 608) 

s_sub_u32 s0, s16, s1 // (op 608) 

s_mul_i32 s1, s7, s14 // (op 608) 
s_add_u32 s1, s1, s0 // (op 608) 


s_mul_hi_i32 s0, s16, s8 // (op 608) 
s_add_i32 s51, s0, s16 // (op 608) 

s_ashr_i32 s0, s51, 31 // (op 608) 
s_cmp_eq_i32 s8, 0 // (op 608) 
s_mov_b32 s6, scc // (op 608) 
s_cmp_eq_i32 s6, 0 // (op 608) 
s_cselect_b32 s7, 0, -1 // (op 608) 

s_and_b32 s6, s15, 31 // (op 608) 
s_lshl_b32 s18, 1, s6 // (op 608) 
s_add_i32 s19, s18, s7 // (op 608) 


s_and_b32 s7, s0, s19 // (op 608) 


s_add_i32 s0, s51, s7 // (op 608) 


s_ashr_i32 s7, s0, s6 // (op 608) 


s_lshl_b32 s0, s15, 24 // (op 608) 
s_ashr_i32 s6, s0, 31 // (op 608) 

s_xor_b32 s0, s7, s6 // (op 608) 

s_sub_i32 s7, s0, s6 // (op 608) 


s_add_u32 s18, s4, 32 // (op 608) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 608) End of Int64 add, unlocking SCC
s_sub_u32 s60, s18, 1 // (op 608) least significant half; sets scc
s_subb_u32 s61, s19, 0 // (op 608) most significant half; uses scc

s_ashr_i64 s[18:19], s[60:61], 5 // (op 608) 

s_mul_i32 s0, s14, s18 // (op 608) 

s_mul_i32 s6, s50, s0 // (op 608) 

s_mul_i32 s0, s7, s6 // (op 608) 

s_sub_u32 s6, s16, s0 // (op 608) 

s_mov_b32 s0, s6 // (op 608) 
s_mul_hi_i32 s51, s0, s13 // (op 608) 
s_add_i32 s18, s51, s0 // (op 608) 


s_ashr_i32 s0, s18, 31 // (op 608) 
s_cmp_eq_i32 s13, 0 // (op 608) 
s_mov_b32 s19, scc // (op 608) 
s_cmp_eq_i32 s19, 0 // (op 608) 
s_cselect_b32 s51, 0, -1 // (op 608) 

s_and_b32 s19, s49, 31 // (op 608) 
s_lshl_b32 s60, 1, s19 // (op 608) 
s_add_i32 s61, s60, s51 // (op 608) 


s_and_b32 s51, s0, s61 // (op 608) 


s_add_i32 s0, s18, s51 // (op 608) 


s_ashr_i32 s18, s0, s19 // (op 608) 


s_lshl_b32 s0, s49, 24 // (op 608) 
s_ashr_i32 s19, s0, 31 // (op 608) 

s_xor_b32 s0, s18, s19 // (op 608) 

s_sub_i32 s18, s0, s19 // (op 608) 


s_mul_i32 s0, s48, s14 // (op 608) 
s_add_u32 s60, s20, 16 // (op 608) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 608) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 608) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 608) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 4 // (op 608) 

s_sub_i32 s19, s60, s0 // (op 608) 

s_mul_i32 s51, s18, s19 // (op 608) 


s_sub_u32 s18, s6, s51 // (op 608) 


s_add_u32 s6, s18, s0 // (op 608) 


s_cmp_eq_i32 s7, 0 // (op 608) 
s_mov_b32 s0, scc // (op 608) 

s_cmp_eq_i32 s0, 0 // (op 608) 
s_cselect_b32 s7, s6, s1 // (op 608) 



v_lshl_add_u32 v10, s7, 4, v18 // (op 608) 


v_mul_lo_u32 v18, v11, s30 // (op 608) 

v_add_u32 v11, v10, v18 // (op 608) 


s_mul_i32 s1, s30, s35 // (op 610) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 610) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 610) least significant
s_mul_hi_u32 s6, s30, s34 // (op 610) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 610) most significant: sum
s_add_u32 s1, s1, s6 // (op 610) most significant: sum


s_mov_b32 s60, s32 // (op 610) reg expression
s_mov_b32 s61, s33 // (op 610) reg expression
s_mov_b32 s62, s0 // (op 610) reg expression
s_mov_b32 s63, 131072 // (op 610) call()

s_mul_hi_i32 s51, s16, s9 // (op 627) 
v_lshrrev_b32 v10, 7, v1 // (op 627) 
s_add_i32 s0, s51, s16 // (op 627) 

v_and_b32 v18, 15, v1 // (op 627) 
s_ashr_i32 s1, s0, 31 // (op 627) 
v_lshl_add_u32 v19, v10, 4, v18 // (op 627) 


s_cmp_eq_i32 s9, 0 // (op 627) 
s_mov_b32 s51, scc // (op 627) 
v_and_b32 v10, 63, v1 // (op 627) 
s_cmp_eq_i32 s51, 0 // (op 627) 
s_cselect_b32 s6, 0, -1 // (op 627) 

v_lshrrev_b32 v18, 4, v10 // (op 627) 

s_and_b32 s7, s10, 31 // (op 627) 
s_lshl_b32 s51, 1, s7 // (op 627) 
s_add_i32 s18, s51, s6 // (op 627) 


s_and_b32 s6, s1, s18 // (op 627) 


s_add_i32 s1, s0, s6 // (op 627) 


s_ashr_i32 s0, s1, s7 // (op 627) 


s_lshl_b32 s1, s10, 24 // (op 627) 
s_ashr_i32 s51, s1, 31 // (op 627) 

s_xor_b32 s1, s0, s51 // (op 627) 

s_sub_i32 s0, s1, s51 // (op 627) 


s_mul_hi_i32 s1, s0, s11 // (op 627) 
s_add_i32 s51, s1, s0 // (op 627) 

s_ashr_i32 s1, s51, 31 // (op 627) 
s_cmp_eq_i32 s11, 0 // (op 627) 
s_mov_b32 s6, scc // (op 627) 
s_cmp_eq_i32 s6, 0 // (op 627) 
s_cselect_b32 s7, 0, -1 // (op 627) 

s_and_b32 s6, s12, 31 // (op 627) 
s_lshl_b32 s18, 1, s6 // (op 627) 
s_add_i32 s19, s18, s7 // (op 627) 


s_and_b32 s7, s1, s19 // (op 627) 


s_add_i32 s1, s51, s7 // (op 627) 


s_ashr_i32 s7, s1, s6 // (op 627) 


s_lshl_b32 s1, s12, 24 // (op 627) 
s_ashr_i32 s6, s1, 31 // (op 627) 

s_xor_b32 s1, s7, s6 // (op 627) 

s_sub_i32 s7, s1, s6 // (op 627) 


s_mul_i32 s1, s0, s14 // (op 627) 

s_sub_u32 s0, s16, s1 // (op 627) 

s_mul_i32 s1, s7, s14 // (op 627) 
s_add_u32 s1, s1, s0 // (op 627) 


s_mul_hi_i32 s0, s16, s8 // (op 627) 
s_add_i32 s51, s0, s16 // (op 627) 

s_ashr_i32 s0, s51, 31 // (op 627) 
s_cmp_eq_i32 s8, 0 // (op 627) 
s_mov_b32 s6, scc // (op 627) 
s_cmp_eq_i32 s6, 0 // (op 627) 
s_cselect_b32 s7, 0, -1 // (op 627) 

s_and_b32 s6, s15, 31 // (op 627) 
s_lshl_b32 s18, 1, s6 // (op 627) 
s_add_i32 s19, s18, s7 // (op 627) 


s_and_b32 s7, s0, s19 // (op 627) 


s_add_i32 s0, s51, s7 // (op 627) 


s_ashr_i32 s7, s0, s6 // (op 627) 


s_lshl_b32 s0, s15, 24 // (op 627) 
s_ashr_i32 s6, s0, 31 // (op 627) 

s_xor_b32 s0, s7, s6 // (op 627) 

s_sub_i32 s7, s0, s6 // (op 627) 


s_add_u32 s18, s4, 32 // (op 627) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 627) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 627) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 627) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 627) 

s_mul_i32 s0, s14, s18 // (op 627) 

s_mul_i32 s6, s50, s0 // (op 627) 

s_mul_i32 s0, s7, s6 // (op 627) 

s_sub_u32 s6, s16, s0 // (op 627) 

s_mov_b32 s0, s6 // (op 627) 
s_mul_hi_i32 s51, s0, s13 // (op 627) 
s_add_i32 s18, s51, s0 // (op 627) 


s_ashr_i32 s0, s18, 31 // (op 627) 
s_cmp_eq_i32 s13, 0 // (op 627) 
s_mov_b32 s19, scc // (op 627) 
s_cmp_eq_i32 s19, 0 // (op 627) 
s_cselect_b32 s51, 0, -1 // (op 627) 

s_and_b32 s19, s49, 31 // (op 627) 
s_lshl_b32 s64, 1, s19 // (op 627) 
s_add_i32 s65, s64, s51 // (op 627) 


s_and_b32 s51, s0, s65 // (op 627) 


s_add_i32 s0, s18, s51 // (op 627) 


s_ashr_i32 s18, s0, s19 // (op 627) 


s_lshl_b32 s0, s49, 24 // (op 627) 
s_ashr_i32 s19, s0, 31 // (op 627) 

s_xor_b32 s0, s18, s19 // (op 627) 

s_sub_i32 s18, s0, s19 // (op 627) 


s_mul_i32 s0, s48, s14 // (op 627) 
s_add_u32 s64, s20, 16 // (op 627) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 627) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 627) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 627) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 4 // (op 627) 

s_sub_i32 s19, s64, s0 // (op 627) 

s_mul_i32 s51, s18, s19 // (op 627) 


s_sub_u32 s18, s6, s51 // (op 627) 


s_add_u32 s6, s18, s0 // (op 627) 


s_cmp_eq_i32 s7, 0 // (op 627) 
s_mov_b32 s0, scc // (op 627) 

s_cmp_eq_i32 s0, 0 // (op 627) 
s_cselect_b32 s7, s6, s1 // (op 627) 



v_lshl_add_u32 v10, s7, 4, v19 // (op 627) 


v_mul_lo_u32 v19, v18, s30 // (op 627) 

v_add_u32 v18, v10, v19 // (op 627) 


s_mul_i32 s1, s30, s35 // (op 629) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 629) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 629) least significant
s_mul_hi_u32 s6, s30, s34 // (op 629) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 629) most significant: sum
s_add_u32 s1, s1, s6 // (op 629) most significant: sum


s_mov_b32 s60, s32 // (op 629) reg expression
s_mov_b32 s61, s33 // (op 629) reg expression
s_mov_b32 s62, s0 // (op 629) reg expression
s_mov_b32 s63, 131072 // (op 629) call()

s_mul_hi_i32 s51, s16, s9 // (op 672) 
v_and_b32 v10, -32, v1 // (op 672) 
s_add_i32 s0, s51, s16 // (op 672) 

v_sub_u32 v19, v1, v10 // (op 672) 

s_ashr_i32 s1, s0, 31 // (op 672) 
v_lshrrev_b32 v10, 5, v1 // (op 672) 
s_cmp_eq_i32 s9, 0 // (op 672) 
s_mov_b32 s51, scc // (op 672) 
s_cmp_eq_i32 s51, 0 // (op 672) 
s_cselect_b32 s6, 0, -1 // (op 672) 

s_and_b32 s7, s10, 31 // (op 672) 
s_lshl_b32 s51, 1, s7 // (op 672) 
s_add_i32 s18, s51, s6 // (op 672) 


s_and_b32 s6, s1, s18 // (op 672) 


s_add_i32 s1, s0, s6 // (op 672) 


s_ashr_i32 s0, s1, s7 // (op 672) 


s_lshl_b32 s1, s10, 24 // (op 672) 
s_ashr_i32 s51, s1, 31 // (op 672) 

s_xor_b32 s1, s0, s51 // (op 672) 

s_sub_i32 s0, s1, s51 // (op 672) 


s_mul_hi_i32 s1, s0, s11 // (op 672) 
s_add_i32 s51, s1, s0 // (op 672) 

s_ashr_i32 s1, s51, 31 // (op 672) 
s_cmp_eq_i32 s11, 0 // (op 672) 
s_mov_b32 s6, scc // (op 672) 
s_cmp_eq_i32 s6, 0 // (op 672) 
s_cselect_b32 s7, 0, -1 // (op 672) 

s_and_b32 s6, s12, 31 // (op 672) 
s_lshl_b32 s18, 1, s6 // (op 672) 
s_add_i32 s19, s18, s7 // (op 672) 


s_and_b32 s7, s1, s19 // (op 672) 


s_add_i32 s1, s51, s7 // (op 672) 


s_ashr_i32 s7, s1, s6 // (op 672) 


s_lshl_b32 s1, s12, 24 // (op 672) 
s_ashr_i32 s6, s1, 31 // (op 672) 

s_xor_b32 s1, s7, s6 // (op 672) 

s_sub_i32 s7, s1, s6 // (op 672) 


s_add_u32 s18, s4, 32 // (op 672) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 672) End of Int64 add, unlocking SCC
s_sub_u32 s64, s18, 1 // (op 672) least significant half; sets scc
s_subb_u32 s65, s19, 0 // (op 672) most significant half; uses scc

s_ashr_i64 s[18:19], s[64:65], 5 // (op 672) 

s_mul_i32 s1, s7, s18 // (op 672) 

s_sub_i32 s51, s0, s1 // (op 672) 


s_mul_hi_i32 s0, s16, s8 // (op 672) 
s_add_i32 s1, s0, s16 // (op 672) 

s_ashr_i32 s0, s1, 31 // (op 672) 
s_cmp_eq_i32 s8, 0 // (op 672) 
s_mov_b32 s6, scc // (op 672) 
s_cmp_eq_i32 s6, 0 // (op 672) 
s_cselect_b32 s7, 0, -1 // (op 672) 

s_and_b32 s6, s15, 31 // (op 672) 
s_lshl_b32 s64, 1, s6 // (op 672) 
s_add_i32 s65, s64, s7 // (op 672) 


s_and_b32 s7, s0, s65 // (op 672) 


s_add_i32 s0, s1, s7 // (op 672) 


s_ashr_i32 s1, s0, s6 // (op 672) 


s_lshl_b32 s0, s15, 24 // (op 672) 
s_ashr_i32 s6, s0, 31 // (op 672) 

s_xor_b32 s0, s1, s6 // (op 672) 

s_sub_i32 s1, s0, s6 // (op 672) 


s_mul_i32 s0, s14, s18 // (op 672) 

s_mul_i32 s6, s50, s0 // (op 672) 

s_mul_i32 s0, s1, s6 // (op 672) 

s_sub_u32 s6, s16, s0 // (op 672) 

s_mul_hi_i32 s0, s6, s13 // (op 672) 
s_add_i32 s7, s0, s6 // (op 672) 


s_ashr_i32 s0, s7, 31 // (op 672) 
s_cmp_eq_i32 s13, 0 // (op 672) 
s_mov_b32 s6, scc // (op 672) 
s_cmp_eq_i32 s6, 0 // (op 672) 
s_cselect_b32 s18, 0, -1 // (op 672) 

s_and_b32 s6, s49, 31 // (op 672) 
s_lshl_b32 s19, 1, s6 // (op 672) 
s_add_i32 s64, s19, s18 // (op 672) 


s_and_b32 s18, s0, s64 // (op 672) 


s_add_i32 s0, s7, s18 // (op 672) 


s_ashr_i32 s7, s0, s6 // (op 672) 


s_lshl_b32 s0, s49, 24 // (op 672) 
s_ashr_i32 s6, s0, 31 // (op 672) 

s_xor_b32 s0, s7, s6 // (op 672) 

s_sub_i32 s7, s0, s6 // (op 672) 


s_cmp_eq_i32 s1, 0 // (op 672) 
s_mov_b32 s0, scc // (op 672) 

s_cmp_eq_i32 s0, 0 // (op 672) 
s_cselect_b32 s1, s7, s51 // (op 672) 



v_lshl_add_u32 v20, s1, 5, v19 // (op 672) 


v_mul_lo_u32 v19, v10, s36 // (op 672) 
v_add_u32 v19, v19, v20 // (op 672) 


s_mul_i32 s1, s36, s35 // (op 674) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 674) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 674) least significant
s_mul_hi_u32 s6, s36, s34 // (op 674) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 674) most significant: sum
s_add_u32 s1, s1, s6 // (op 674) most significant: sum


s_mov_b32 s64, s38 // (op 674) reg expression
s_mov_b32 s65, s39 // (op 674) reg expression
s_mov_b32 s66, s0 // (op 674) reg expression
s_mov_b32 s67, 131072 // (op 674) call()

s_mul_hi_i32 s51, s16, s9 // (op 691) 
v_and_b32 v10, -32, v1 // (op 691) 
s_add_i32 s0, s51, s16 // (op 691) 

v_sub_u32 v20, v1, v10 // (op 691) 

s_ashr_i32 s1, s0, 31 // (op 691) 
v_lshrrev_b32 v10, 5, v1 // (op 691) 
s_cmp_eq_i32 s9, 0 // (op 691) 
s_mov_b32 s51, scc // (op 691) 
s_cmp_eq_i32 s51, 0 // (op 691) 
s_cselect_b32 s6, 0, -1 // (op 691) 

s_and_b32 s7, s10, 31 // (op 691) 
s_lshl_b32 s51, 1, s7 // (op 691) 
s_add_i32 s18, s51, s6 // (op 691) 


s_and_b32 s6, s1, s18 // (op 691) 


s_add_i32 s1, s0, s6 // (op 691) 


s_ashr_i32 s0, s1, s7 // (op 691) 


s_lshl_b32 s1, s10, 24 // (op 691) 
s_ashr_i32 s51, s1, 31 // (op 691) 

s_xor_b32 s1, s0, s51 // (op 691) 

s_sub_i32 s0, s1, s51 // (op 691) 


s_mul_hi_i32 s1, s0, s11 // (op 691) 
s_add_i32 s51, s1, s0 // (op 691) 

s_ashr_i32 s1, s51, 31 // (op 691) 
s_cmp_eq_i32 s11, 0 // (op 691) 
s_mov_b32 s6, scc // (op 691) 
s_cmp_eq_i32 s6, 0 // (op 691) 
s_cselect_b32 s7, 0, -1 // (op 691) 

s_and_b32 s6, s12, 31 // (op 691) 
s_lshl_b32 s18, 1, s6 // (op 691) 
s_add_i32 s19, s18, s7 // (op 691) 


s_and_b32 s7, s1, s19 // (op 691) 


s_add_i32 s1, s51, s7 // (op 691) 


s_ashr_i32 s7, s1, s6 // (op 691) 


s_lshl_b32 s1, s12, 24 // (op 691) 
s_ashr_i32 s6, s1, 31 // (op 691) 

s_xor_b32 s1, s7, s6 // (op 691) 

s_sub_i32 s7, s1, s6 // (op 691) 


s_add_u32 s18, s4, 32 // (op 691) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 691) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 691) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 691) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 691) 

s_mul_i32 s1, s7, s18 // (op 691) 

s_sub_i32 s51, s0, s1 // (op 691) 


s_mul_hi_i32 s0, s16, s8 // (op 691) 
s_add_i32 s1, s0, s16 // (op 691) 

s_ashr_i32 s0, s1, 31 // (op 691) 
s_cmp_eq_i32 s8, 0 // (op 691) 
s_mov_b32 s6, scc // (op 691) 
s_cmp_eq_i32 s6, 0 // (op 691) 
s_cselect_b32 s7, 0, -1 // (op 691) 

s_and_b32 s6, s15, 31 // (op 691) 
s_lshl_b32 s68, 1, s6 // (op 691) 
s_add_i32 s69, s68, s7 // (op 691) 


s_and_b32 s7, s0, s69 // (op 691) 


s_add_i32 s0, s1, s7 // (op 691) 


s_ashr_i32 s1, s0, s6 // (op 691) 


s_lshl_b32 s0, s15, 24 // (op 691) 
s_ashr_i32 s6, s0, 31 // (op 691) 

s_xor_b32 s0, s1, s6 // (op 691) 

s_sub_i32 s1, s0, s6 // (op 691) 


s_mul_i32 s0, s14, s18 // (op 691) 

s_mul_i32 s6, s50, s0 // (op 691) 

s_mul_i32 s0, s1, s6 // (op 691) 

s_sub_u32 s6, s16, s0 // (op 691) 

s_mul_hi_i32 s0, s6, s13 // (op 691) 
s_add_i32 s7, s0, s6 // (op 691) 


s_ashr_i32 s0, s7, 31 // (op 691) 
s_cmp_eq_i32 s13, 0 // (op 691) 
s_mov_b32 s6, scc // (op 691) 
s_cmp_eq_i32 s6, 0 // (op 691) 
s_cselect_b32 s18, 0, -1 // (op 691) 

s_and_b32 s6, s49, 31 // (op 691) 
s_lshl_b32 s19, 1, s6 // (op 691) 
s_add_i32 s68, s19, s18 // (op 691) 


s_and_b32 s18, s0, s68 // (op 691) 


s_add_i32 s0, s7, s18 // (op 691) 


s_ashr_i32 s7, s0, s6 // (op 691) 


s_lshl_b32 s0, s49, 24 // (op 691) 
s_ashr_i32 s6, s0, 31 // (op 691) 

s_xor_b32 s0, s7, s6 // (op 691) 

s_sub_i32 s7, s0, s6 // (op 691) 


s_cmp_eq_i32 s1, 0 // (op 691) 
s_mov_b32 s0, scc // (op 691) 

s_cmp_eq_i32 s0, 0 // (op 691) 
s_cselect_b32 s1, s7, s51 // (op 691) 



v_lshl_add_u32 v21, s1, 5, v20 // (op 691) 


v_mul_lo_u32 v20, v10, s36 // (op 691) 
v_add_u32 v20, v20, v21 // (op 691) 


s_mul_i32 s1, s36, s35 // (op 693) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 693) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 693) least significant
s_mul_hi_u32 s6, s36, s34 // (op 693) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 693) most significant: sum
s_add_u32 s1, s1, s6 // (op 693) most significant: sum


s_mov_b32 s64, s38 // (op 693) reg expression
s_mov_b32 s65, s39 // (op 693) reg expression
s_mov_b32 s66, s0 // (op 693) reg expression
s_mov_b32 s67, 131072 // (op 693) call()


v_mov_b64 v[22:23], s[30:31] // (op 341) convert

v_mov_b32 v10, v11 // (op 341) reg expression
v_mov_b32 v21, v10 // (op 341) 
buffer_load_ubyte v24, v21, s[60:63], 0 offen offset: 0 // (op 341) Load value





v_mov_b32 v10, v12 // (op 345) reg expression

buffer_load_dwordx3 v[26:28], v10, s[56:59], 0 offen // (op 345) Load value
buffer_load_dwordx3 v[30:32], v10, s[56:59], 0 offen offset:12 // (op 345) Load value



v_mov_b32 v10, v19 // (op 348) reg expression

buffer_load_ubyte v25, v10, s[64:67], 0 offen offset: 0 // (op 348) Load value

v_lshlrev_b32 v10, 4, v1 // (op 757) 
s_nop 0
v_readfirstlane_b32 s51, v10 // (op 757) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[22:23], v[14:15] // (op 436) reg expression

s_nop 0
s_mov_b32 m0, s51 // (op 436) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 436) Load value direct to lds



v_mov_b64 v[22:23], s[30:31] // (op 355) convert

s_lshl_b32 s51, s30, 2 // (op 355) 
v_add_u32 v10, v11, s51 // (op 355) 

v_mov_b32 v21, v10 // (op 355) 
buffer_load_ubyte v29, v21, s[60:63], 0 offen offset: 0 // (op 355) Load value






s_lshl_b64 s[0:1], s[26:27], 7 // (op 359) 
s_mul_i32 s7, s1, 6 // (op 359) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 359) least significant
s_mul_hi_u32 s51, s0, 6 // (op 359) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 359) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 359) 

v_add_u32 v10, v12, s0 // (op 359) 


buffer_load_dwordx3 v[34:36], v10, s[56:59], 0 offen // (op 359) Load value
buffer_load_dwordx3 v[38:40], v10, s[56:59], 0 offen offset:12 // (op 359) Load value




s_lshl_b32 s51, s36, 2 // (op 362) 
v_add_u32 v33, v19, s51 // (op 362) 


buffer_load_ubyte v37, v33, s[64:67], 0 offen offset: 0 // (op 362) Load value


v_lshlrev_b32 v19, 4, v1 // (op 936) 
s_nop 0
v_readfirstlane_b32 s51, v19 // (op 936) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[0:1], s[2:3], 7 // (op 441) 
v_mov_b32 v19, s0 // (op 441) 
v_mov_b32 v33, s1 // (op 441) 
v_add_co_u32 v22, s[6:7], v14, v19 // (op 441) least significant half
v_addc_co_u32 v23, s[6:7], v15, v33, s[6:7] // (op 441) most significant half




s_add_u32 s0, 2048, s51 // (op 441) 

s_mov_b32 m0, s0 // (op 441) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 441) Load value direct to lds




v_lshlrev_b32 v19, 5, v1 // (op 810) 
v_mul_lo_u32 v33, v19, 6 // (op 810) 
v_lshrrev_b32 v10, 3, v33 // (op 810) 

v_lshrrev_b32 v33, 4, v19 // (op 810) 

v_lshlrev_b32 v19, 2, v33 // (op 810) 

v_add_u32 v33, v10, v19 // (op 810) 





s_waitcnt vmcnt(8)
ds_write_b96 v33, v[26:28] offset:4096 // (op 367) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v33, v[30:32] offset:4112 // (op 367) Store local data 

v_and_b32 v19, -32, v1 // (op 863) 
v_sub_u32 v33, v1, v19 // (op 863) 

v_and_b32 v19, 4294967264, v1 // (op 863) 
v_add_u32 v10, v19, v33 // (op 863) 





s_waitcnt vmcnt(6)
ds_write_b8 v10, v25 offset:8192 // (op 368) Store local data 

v_lshlrev_b32 v19, 4, v1 // (op 729) 
s_nop 0
v_readfirstlane_b32 s51, v19 // (op 729) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 5, v1 // (op 781) 
v_mul_lo_u32 v33, v19, 6 // (op 781) 
v_lshrrev_b32 v10, 3, v33 // (op 781) 

v_lshrrev_b32 v33, 4, v19 // (op 781) 

v_lshlrev_b32 v19, 2, v33 // (op 781) 

v_add_u32 v33, v10, v19 // (op 781) 


v_and_b32 v19, -32, v1 // (op 834) 
v_sub_u32 v10, v1, v19 // (op 834) 

v_and_b32 v19, 4294967264, v1 // (op 834) 
v_add_u32 v11, v19, v10 // (op 834) 


v_lshlrev_b32 v10, 4, v1 // (op 921) 
s_nop 0
v_readfirstlane_b32 s0, v10 // (op 921) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v10, 5, v1 // (op 960) 
v_mul_lo_u32 v12, v10, 6 // (op 960) 
v_lshrrev_b32 v19, 3, v12 // (op 960) 

v_lshrrev_b32 v12, 4, v10 // (op 960) 

v_lshlrev_b32 v10, 2, v12 // (op 960) 

v_add_u32 v12, v19, v10 // (op 960) 


v_and_b32 v10, -32, v1 // (op 984) 
v_sub_u32 v19, v1, v10 // (op 984) 

v_and_b32 v10, 4294967264, v1 // (op 984) 
v_add_u32 v14, v10, v19 // (op 984) 


s_mov_b32 s1, 0 // (op 57) call()
s_ashr_i64 s[6:7], s[24:25], 7 // (op 56) 
s_mov_b32 s18, s6 // (op 56) convert

s_and_b32 s19, s18, 4294967294 // (op 56) 

s_cmp_lt_u32 s1, s19 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56 if false)
RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56:
s_add_u32 s1, s1, 2 // (op 58) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1029) 



ds_read_b64_tr_b8 v[42:43], v0 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:128 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:1024 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:1152 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1042) 



ds_read_b96_tr_b6 v[50:52], v3 offset:4096 // (op 72) Transpose load local data 
ds_read_b96_tr_b6 v[53:55], v3 offset:4608 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1056) 


ds_read_u8 v10, v4 offset:8192 // (op 81) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:55], a[0:3], v24, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 84) 




v_mov_b64 v[22:23], s[30:31] // (op 77) convert

v_mov_b32 v10, v18 // (op 77) 
s_lshl_b32 s6, s30, 3 // (op 77) 
v_add_u32 v15, v10, s6 // (op 77) 

v_mov_b32 v19, v15 // (op 77) 
buffer_load_ubyte v24, v19, s[60:63], 0 offen offset: 0 // (op 77) Load value





v_mov_b32 v15, v13 // (op 10) 
s_lshl_b64 s[6:7], s[26:27], 7 // (op 10) 
s_mul_i32 s19, s7, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 10) least significant
s_mul_hi_u32 s68, s6, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 10) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 10) 

s_lshl_b32 s18, s6, 1 // (op 10) 

v_add_u32 v19, v15, s18 // (op 10) 


buffer_load_dwordx3 v[26:28], v19, s[56:59], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[30:32], v19, s[56:59], 0 offen offset:12 // (op 10) Load value



v_mov_b32 v19, v20 // (op 21) 
s_lshl_b32 s6, s36, 3 // (op 21) 
v_add_u32 v21, v19, s6 // (op 21) 


buffer_load_ubyte v25, v21, s[64:67], 0 offen offset: 0 // (op 21) Load value

s_barrier  // (op 1032) 


v_mov_b64 v[22:23], v[16:17] // (op 420) 
s_lshl_b64 s[6:7], s[2:3], 7 // (op 420) 
s_lshl_b64 s[18:19], s[6:7], 1 // (op 420) 

v_mov_b32 v21, s18 // (op 420) 
v_mov_b32 v41, s19 // (op 420) 
v_add_co_u32 v42, s[6:7], v22, v21 // (op 420) least significant half
v_addc_co_u32 v43, s[6:7], v23, v41, s[6:7] // (op 420) most significant half





s_mov_b32 m0, s51 // (op 420) reg expression
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 420) Load value direct to lds

s_barrier  // (op 1079) 



s_waitcnt vmcnt(8)
ds_write_b96 v12, v[34:36] offset:8320 // (op 239) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v12, v[38:40] offset:8336 // (op 239) Store local data 
s_barrier  // (op 1085) 



s_waitcnt vmcnt(6)
ds_write_b8 v14, v37 offset:12416 // (op 240) Store local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1070) 



ds_read_b64_tr_b8 v[42:43], v5 offset:2048 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:2176 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v5 offset:3072 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v5 offset:3200 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1076) 



ds_read_b96_tr_b6 v[50:52], v6 offset:8320 // (op 233) Transpose load local data 
ds_read_b96_tr_b6 v[53:55], v6 offset:8832 // (op 233) Transpose load local data 
s_barrier  // (op 1136) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1082) 


ds_read_u8 v21, v7 offset:12416 // (op 237) Load local data 
s_barrier  // (op 1144) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:55], a[0:3], v29, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 238) 




v_mov_b64 v[42:43], s[30:31] // (op 235) convert

s_lshl_b32 s6, s30, 2 // (op 235) 
s_mul_i32 s7, 3, s6 // (op 235) 

v_add_u32 v21, v10, s7 // (op 235) 

v_mov_b32 v41, v21 // (op 235) 
buffer_load_ubyte v29, v41, s[60:63], 0 offen offset: 0 // (op 235) Load value






s_lshl_b32 s6, s30, 3 // (op 619) 
v_add_u32 v18, v18, s6 // (op 619) 

s_lshl_b64 s[6:7], s[26:27], 7 // (op 241) 
s_mul_i32 s19, s7, 6 // (op 241) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 241) least significant
s_mul_hi_u32 s68, s6, 6 // (op 241) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 241) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 241) 

s_mul_i32 s18, 3, s6 // (op 241) 

v_add_u32 v10, v15, s18 // (op 241) 


buffer_load_dwordx3 v[34:36], v10, s[56:59], 0 offen // (op 241) Load value
buffer_load_dwordx3 v[38:40], v10, s[56:59], 0 offen offset:12 // (op 241) Load value




s_lshl_b32 s6, s36, 2 // (op 242) 
s_mul_i32 s7, 3, s6 // (op 242) 

v_add_u32 v10, v19, s7 // (op 242) 


s_lshl_b64 s[6:7], s[26:27], 7 // (op 555) 
s_mul_i32 s19, s7, 6 // (op 555) most significant: low of high * low
s_mul_i32 s18, s6, 6 // (op 555) least significant
s_mul_hi_u32 s68, s6, 6 // (op 555) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 555) most significant: sum


s_lshr_b64 s[6:7], s[18:19], 3 // (op 555) 

s_lshl_b32 s18, s6, 1 // (op 555) 

v_add_u32 v13, v13, s18 // (op 555) 

buffer_load_ubyte v37, v10, s[64:67], 0 offen offset: 0 // (op 242) Load value


s_barrier  // (op 1073) 
s_lshl_b32 s6, s36, 3 // (op 683) 


v_add_u32 v20, v20, s6 // (op 683) 

s_lshl_b64 s[6:7], s[2:3], 7 // (op 431) 
s_mul_i32 s19, 3, s7 // (op 431) most significant: low of low * high
s_mul_i32 s18, 3, s6 // (op 431) least significant
s_mul_hi_u32 s68, 3, s6 // (op 431) most significant: high of low * low
s_add_u32 s19, s19, s68 // (op 431) most significant: sum


v_mov_b32 v10, s18 // (op 431) 
v_mov_b32 v15, s19 // (op 431) 
v_add_co_u32 v42, s[6:7], v22, v10 // (op 431) least significant half
v_addc_co_u32 v43, s[6:7], v23, v15, s[6:7] // (op 431) most significant half




s_add_u32 s6, 2048, s0 // (op 431) 

s_nop 0
s_mov_b32 m0, s6 // (op 431) reg expression
buffer_load_dwordx4 v42, s[52:55], 0 offen offset: 0 lds // (op 431) Load value direct to lds



s_barrier  // (op 1045) 
s_lshl_b64 s[6:7], s[2:3], 7 // (op 491) 



s_lshl_b64 s[18:19], s[6:7], 1 // (op 491) 

v_mov_b32 v10, s18 // (op 491) 
v_mov_b32 v15, s19 // (op 491) 
v_add_co_u32 v16, s[6:7], v16, v10 // (op 491) least significant half
v_addc_co_u32 v17, s[6:7], v17, v15, s[6:7] // (op 491) most significant half




s_waitcnt vmcnt(8)
ds_write_b96 v33, v[26:28] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v33, v[30:32] offset:4112 // (op 13) Store local data 
s_barrier  // (op 1059) 



s_waitcnt vmcnt(6)
ds_write_b8 v11, v25 offset:8192 // (op 26) Store local data 
s_ashr_i64 s[6:7], s[24:25], 7 // (op 56) 
s_mov_b32 s18, s6 // (op 56) convert

s_and_b32 s19, s18, 4294967294 // (op 56) 

s_cmp_lt_u32 s1, s19 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoop_56 if true)
RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoop_56:














s_barrier  // (op 1098) 
s_mul_hi_i32 s51, s16, s9 // (op 454) 
s_add_i32 s0, s51, s16 // (op 454) 

s_ashr_i32 s1, s0, 31 // (op 454) 
s_cmp_eq_i32 s9, 0 // (op 454) 
s_mov_b32 s51, scc // (op 454) 
s_cmp_eq_i32 s51, 0 // (op 454) 
s_cselect_b32 s6, 0, -1 // (op 454) 

s_and_b32 s7, s10, 31 // (op 454) 
s_lshl_b32 s51, 1, s7 // (op 454) 
s_add_i32 s18, s51, s6 // (op 454) 


s_and_b32 s6, s1, s18 // (op 454) 


s_add_i32 s1, s0, s6 // (op 454) 


s_ashr_i32 s0, s1, s7 // (op 454) 


s_lshl_b32 s1, s10, 24 // (op 454) 
s_ashr_i32 s51, s1, 31 // (op 454) 

s_xor_b32 s1, s0, s51 // (op 454) 

s_sub_i32 s0, s1, s51 // (op 454) 


s_mul_hi_i32 s1, s0, s11 // (op 454) 
s_add_i32 s51, s1, s0 // (op 454) 

s_ashr_i32 s1, s51, 31 // (op 454) 
s_cmp_eq_i32 s11, 0 // (op 454) 
s_mov_b32 s6, scc // (op 454) 
s_cmp_eq_i32 s6, 0 // (op 454) 
s_cselect_b32 s7, 0, -1 // (op 454) 

s_and_b32 s6, s12, 31 // (op 454) 
s_lshl_b32 s18, 1, s6 // (op 454) 
s_add_i32 s19, s18, s7 // (op 454) 


s_and_b32 s7, s1, s19 // (op 454) 


s_add_i32 s1, s51, s7 // (op 454) 


s_ashr_i32 s7, s1, s6 // (op 454) 


s_lshl_b32 s1, s12, 24 // (op 454) 
s_ashr_i32 s6, s1, 31 // (op 454) 

s_xor_b32 s1, s7, s6 // (op 454) 

s_sub_i32 s7, s1, s6 // (op 454) 


s_mul_i32 s1, s0, s14 // (op 454) 

s_sub_u32 s0, s16, s1 // (op 454) 

s_mul_i32 s1, s7, s14 // (op 454) 
s_add_u32 s1, s1, s0 // (op 454) 


s_mul_hi_i32 s0, s16, s8 // (op 454) 
s_add_i32 s51, s0, s16 // (op 454) 

s_ashr_i32 s0, s51, 31 // (op 454) 
s_cmp_eq_i32 s8, 0 // (op 454) 
s_mov_b32 s6, scc // (op 454) 
s_cmp_eq_i32 s6, 0 // (op 454) 
s_cselect_b32 s7, 0, -1 // (op 454) 

s_and_b32 s6, s15, 31 // (op 454) 
s_lshl_b32 s18, 1, s6 // (op 454) 
s_add_i32 s19, s18, s7 // (op 454) 


s_and_b32 s7, s0, s19 // (op 454) 


s_add_i32 s0, s51, s7 // (op 454) 


s_ashr_i32 s7, s0, s6 // (op 454) 


s_lshl_b32 s0, s15, 24 // (op 454) 
s_ashr_i32 s6, s0, 31 // (op 454) 

s_xor_b32 s0, s7, s6 // (op 454) 

s_sub_i32 s7, s0, s6 // (op 454) 


s_add_u32 s18, s4, 32 // (op 454) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 454) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 454) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 454) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 454) 

s_mul_i32 s0, s14, s18 // (op 454) 

s_mul_i32 s6, s50, s0 // (op 454) 

s_mul_i32 s0, s7, s6 // (op 454) 

s_sub_u32 s6, s16, s0 // (op 454) 

s_mov_b32 s0, s6 // (op 454) 
s_mul_hi_i32 s51, s0, s13 // (op 454) 
s_add_i32 s18, s51, s0 // (op 454) 


s_ashr_i32 s0, s18, 31 // (op 454) 
s_cmp_eq_i32 s13, 0 // (op 454) 
s_mov_b32 s19, scc // (op 454) 
s_cmp_eq_i32 s19, 0 // (op 454) 
s_cselect_b32 s51, 0, -1 // (op 454) 

s_and_b32 s19, s49, 31 // (op 454) 
s_lshl_b32 s68, 1, s19 // (op 454) 
s_add_i32 s69, s68, s51 // (op 454) 


s_and_b32 s51, s0, s69 // (op 454) 


s_add_i32 s0, s18, s51 // (op 454) 


s_ashr_i32 s18, s0, s19 // (op 454) 


s_lshl_b32 s0, s49, 24 // (op 454) 
s_ashr_i32 s19, s0, 31 // (op 454) 

s_xor_b32 s0, s18, s19 // (op 454) 

s_sub_i32 s18, s0, s19 // (op 454) 


s_mul_i32 s0, s48, s14 // (op 454) 
s_add_u32 s68, s20, 16 // (op 454) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 454) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 454) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 454) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 454) 

s_sub_i32 s19, s68, s0 // (op 454) 

s_mul_i32 s51, s18, s19 // (op 454) 


s_sub_u32 s18, s6, s51 // (op 454) 


s_add_u32 s6, s18, s0 // (op 454) 


s_cmp_eq_i32 s7, 0 // (op 454) 
s_mov_b32 s0, scc // (op 454) 

s_cmp_eq_i32 s0, 0 // (op 454) 
s_cselect_b32 s7, s6, s1 // (op 454) 



s_lshl_b32 s6, s7, 4 // (op 454) 

s_mov_b32 s1, 0 // (op 454) convert
s_mov_b32 s0, s6 // (op 454) convert

s_ashr_i64 s[6:7], s[24:25], 7 // (op 454) 
s_lshr_b32 s51, s6, 1 // (op 454) 

v_lshl_add_u32 v10, s51, 8, v1 // (op 454) 

v_mul_lo_u32 v31, v10, s3 // (op 454) most significant: low of low * high
v_mul_hi_u32 v11, v10, s2 // (op 454) most significant: high of low * low
v_mul_lo_u32 v30, v10, s2 // (op 454) least significant: low of low * low
v_add_u32 v31, v31, v11 // (op 454) most significant: sum


v_mov_b32 v10, s0 // (op 454) 
v_mov_b32 v11, s1 // (op 454) 
v_add_co_u32 v32, s[6:7], v10, v30 // (op 454) least significant half
v_addc_co_u32 v33, s[6:7], v11, v31, s[6:7] // (op 454) most significant half





s_mul_i32 s1, s2, s25 // (op 456) most significant: low of low * high
s_mul_i32 s51, s3, s24 // (op 456) most significant: low of high * low
s_mul_i32 s0, s2, s24 // (op 456) least significant
s_mul_hi_u32 s6, s2, s24 // (op 456) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 456) most significant: sum
s_add_u32 s1, s1, s6 // (op 456) most significant: sum


s_mov_b32 s52, s22 // (op 456) reg expression
s_mov_b32 s53, s23 // (op 456) reg expression
s_mov_b32 s54, s0 // (op 456) reg expression
s_mov_b32 s55, 131072 // (op 456) call()

s_mul_hi_i32 s51, s16, s9 // (op 521) 
s_add_i32 s0, s51, s16 // (op 521) 

s_ashr_i32 s1, s0, 31 // (op 521) 
s_cmp_eq_i32 s9, 0 // (op 521) 
s_mov_b32 s51, scc // (op 521) 
s_cmp_eq_i32 s51, 0 // (op 521) 
s_cselect_b32 s6, 0, -1 // (op 521) 

s_and_b32 s7, s10, 31 // (op 521) 
s_lshl_b32 s51, 1, s7 // (op 521) 
s_add_i32 s18, s51, s6 // (op 521) 


s_and_b32 s6, s1, s18 // (op 521) 


s_add_i32 s1, s0, s6 // (op 521) 


s_ashr_i32 s0, s1, s7 // (op 521) 


s_lshl_b32 s1, s10, 24 // (op 521) 
s_ashr_i32 s51, s1, 31 // (op 521) 

s_xor_b32 s1, s0, s51 // (op 521) 

s_sub_i32 s0, s1, s51 // (op 521) 


s_mul_hi_i32 s1, s0, s11 // (op 521) 
s_add_i32 s51, s1, s0 // (op 521) 

s_ashr_i32 s1, s51, 31 // (op 521) 
s_cmp_eq_i32 s11, 0 // (op 521) 
s_mov_b32 s6, scc // (op 521) 
s_cmp_eq_i32 s6, 0 // (op 521) 
s_cselect_b32 s7, 0, -1 // (op 521) 

s_and_b32 s6, s12, 31 // (op 521) 
s_lshl_b32 s18, 1, s6 // (op 521) 
s_add_i32 s19, s18, s7 // (op 521) 


s_and_b32 s7, s1, s19 // (op 521) 


s_add_i32 s1, s51, s7 // (op 521) 


s_ashr_i32 s7, s1, s6 // (op 521) 


s_lshl_b32 s1, s12, 24 // (op 521) 
s_ashr_i32 s6, s1, 31 // (op 521) 

s_xor_b32 s1, s7, s6 // (op 521) 

s_sub_i32 s7, s1, s6 // (op 521) 


s_add_u32 s18, s4, 32 // (op 521) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 521) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 521) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 521) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 521) 

s_mul_i32 s1, s7, s18 // (op 521) 

s_sub_i32 s51, s0, s1 // (op 521) 


s_mul_hi_i32 s0, s16, s8 // (op 521) 
s_add_i32 s1, s0, s16 // (op 521) 

s_ashr_i32 s0, s1, 31 // (op 521) 
s_cmp_eq_i32 s8, 0 // (op 521) 
s_mov_b32 s6, scc // (op 521) 
s_cmp_eq_i32 s6, 0 // (op 521) 
s_cselect_b32 s7, 0, -1 // (op 521) 

s_and_b32 s6, s15, 31 // (op 521) 
s_lshl_b32 s68, 1, s6 // (op 521) 
s_add_i32 s69, s68, s7 // (op 521) 


s_and_b32 s7, s0, s69 // (op 521) 


s_add_i32 s0, s1, s7 // (op 521) 


s_ashr_i32 s1, s0, s6 // (op 521) 


s_lshl_b32 s0, s15, 24 // (op 521) 
s_ashr_i32 s6, s0, 31 // (op 521) 

s_xor_b32 s0, s1, s6 // (op 521) 

s_sub_i32 s1, s0, s6 // (op 521) 


s_mul_i32 s0, s14, s18 // (op 521) 

s_mul_i32 s6, s50, s0 // (op 521) 

s_mul_i32 s0, s1, s6 // (op 521) 

s_sub_u32 s6, s16, s0 // (op 521) 

s_mul_hi_i32 s0, s6, s13 // (op 521) 
s_add_i32 s7, s0, s6 // (op 521) 


s_ashr_i32 s0, s7, 31 // (op 521) 
s_cmp_eq_i32 s13, 0 // (op 521) 
s_mov_b32 s6, scc // (op 521) 
s_cmp_eq_i32 s6, 0 // (op 521) 
s_cselect_b32 s18, 0, -1 // (op 521) 

s_and_b32 s6, s49, 31 // (op 521) 
s_lshl_b32 s19, 1, s6 // (op 521) 
s_add_i32 s68, s19, s18 // (op 521) 


s_and_b32 s18, s0, s68 // (op 521) 


s_add_i32 s0, s7, s18 // (op 521) 


s_ashr_i32 s7, s0, s6 // (op 521) 


s_lshl_b32 s0, s49, 24 // (op 521) 
s_ashr_i32 s6, s0, 31 // (op 521) 

s_xor_b32 s0, s7, s6 // (op 521) 

s_sub_i32 s7, s0, s6 // (op 521) 


s_cmp_eq_i32 s1, 0 // (op 521) 
s_mov_b32 s0, scc // (op 521) 

s_cmp_eq_i32 s0, 0 // (op 521) 
s_cselect_b32 s1, s7, s51 // (op 521) 



s_lshl_b32 s0, s1, 5 // (op 521) 

s_mov_b32 s7, 0 // (op 521) convert
s_mov_b32 s6, s0 // (op 521) convert

s_ashr_i64 s[0:1], s[24:25], 7 // (op 521) 
s_lshr_b32 s51, s0, 1 // (op 521) 

v_lshl_add_u32 v10, s51, 8, v1 // (op 521) 

v_mul_lo_u32 v31, v10, s27 // (op 521) most significant: low of low * high
v_mul_hi_u32 v11, v10, s26 // (op 521) most significant: high of low * low
v_mul_lo_u32 v30, v10, s26 // (op 521) least significant: low of low * low
v_add_u32 v31, v31, v11 // (op 521) most significant: sum

v_mov_b32 v11, s6 // (op 521) 
v_mov_b32 v12, s7 // (op 521) 
v_add_co_u32 v30, s[0:1], v30, v11 // (op 521) least significant half
v_addc_co_u32 v31, s[0:1], v31, v12, s[0:1] // (op 521) most significant half





v_mul_lo_u32 v11, v31, 6 // (op 521) most significant: low of high * low
v_mul_hi_u32 v12, v30, 6 // (op 521) most significant: high of low * low
v_mul_lo_u32 v10, v30, 6 // (op 521) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 521) most significant: sum


v_lshrrev_b64 v[30:31], 3, v[10:11] // (op 521) 

v_mov_b32 v10, v30 // (op 521) convert

s_mul_i32 s1, s25, 6 // (op 523) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 523) least significant
s_mul_hi_u32 s51, s24, 6 // (op 523) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 523) most significant: sum

s_mul_i32 s7, s26, s1 // (op 523) most significant: low of low * high
s_mul_i32 s51, s27, s0 // (op 523) most significant: low of high * low
s_mul_i32 s6, s26, s0 // (op 523) least significant
s_mul_hi_u32 s18, s26, s0 // (op 523) most significant: high of low * low
s_add_u32 s7, s7, s51 // (op 523) most significant: sum
s_add_u32 s7, s7, s18 // (op 523) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 523) 

s_mov_b32 s56, s28 // (op 523) reg expression
s_mov_b32 s57, s29 // (op 523) reg expression
s_mov_b32 s58, s0 // (op 523) reg expression
s_mov_b32 s59, 131072 // (op 523) call()

s_mul_hi_i32 s51, s16, s9 // (op 585) 
v_lshrrev_b32 v11, 7, v1 // (op 585) 
s_add_i32 s0, s51, s16 // (op 585) 

v_and_b32 v12, 15, v1 // (op 585) 
s_ashr_i32 s1, s0, 31 // (op 585) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 585) 


s_cmp_eq_i32 s9, 0 // (op 585) 
s_mov_b32 s51, scc // (op 585) 
v_and_b32 v11, 63, v1 // (op 585) 
s_cmp_eq_i32 s51, 0 // (op 585) 
s_cselect_b32 s6, 0, -1 // (op 585) 

v_lshrrev_b32 v12, 4, v11 // (op 585) 

s_and_b32 s7, s10, 31 // (op 585) 
s_lshl_b32 s51, 1, s7 // (op 585) 
s_add_i32 s18, s51, s6 // (op 585) 


s_and_b32 s6, s1, s18 // (op 585) 


s_add_i32 s1, s0, s6 // (op 585) 


s_ashr_i32 s0, s1, s7 // (op 585) 


s_lshl_b32 s1, s10, 24 // (op 585) 
s_ashr_i32 s51, s1, 31 // (op 585) 

s_xor_b32 s1, s0, s51 // (op 585) 

s_sub_i32 s0, s1, s51 // (op 585) 


s_mul_hi_i32 s1, s0, s11 // (op 585) 
s_add_i32 s51, s1, s0 // (op 585) 

s_ashr_i32 s1, s51, 31 // (op 585) 
s_cmp_eq_i32 s11, 0 // (op 585) 
s_mov_b32 s6, scc // (op 585) 
s_cmp_eq_i32 s6, 0 // (op 585) 
s_cselect_b32 s7, 0, -1 // (op 585) 

s_and_b32 s6, s12, 31 // (op 585) 
s_lshl_b32 s18, 1, s6 // (op 585) 
s_add_i32 s19, s18, s7 // (op 585) 


s_and_b32 s7, s1, s19 // (op 585) 


s_add_i32 s1, s51, s7 // (op 585) 


s_ashr_i32 s7, s1, s6 // (op 585) 


s_lshl_b32 s1, s12, 24 // (op 585) 
s_ashr_i32 s6, s1, 31 // (op 585) 

s_xor_b32 s1, s7, s6 // (op 585) 

s_sub_i32 s7, s1, s6 // (op 585) 


s_mul_i32 s1, s0, s14 // (op 585) 

s_sub_u32 s0, s16, s1 // (op 585) 

s_mul_i32 s1, s7, s14 // (op 585) 
s_add_u32 s1, s1, s0 // (op 585) 


s_mul_hi_i32 s0, s16, s8 // (op 585) 
s_add_i32 s51, s0, s16 // (op 585) 

s_ashr_i32 s0, s51, 31 // (op 585) 
s_cmp_eq_i32 s8, 0 // (op 585) 
s_mov_b32 s6, scc // (op 585) 
s_cmp_eq_i32 s6, 0 // (op 585) 
s_cselect_b32 s7, 0, -1 // (op 585) 

s_and_b32 s6, s15, 31 // (op 585) 
s_lshl_b32 s18, 1, s6 // (op 585) 
s_add_i32 s19, s18, s7 // (op 585) 


s_and_b32 s7, s0, s19 // (op 585) 


s_add_i32 s0, s51, s7 // (op 585) 


s_ashr_i32 s7, s0, s6 // (op 585) 


s_lshl_b32 s0, s15, 24 // (op 585) 
s_ashr_i32 s6, s0, 31 // (op 585) 

s_xor_b32 s0, s7, s6 // (op 585) 

s_sub_i32 s7, s0, s6 // (op 585) 


s_add_u32 s18, s4, 32 // (op 585) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 585) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 585) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 585) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 585) 

s_mul_i32 s0, s14, s18 // (op 585) 

s_mul_i32 s6, s50, s0 // (op 585) 

s_mul_i32 s0, s7, s6 // (op 585) 

s_sub_u32 s6, s16, s0 // (op 585) 

s_mov_b32 s0, s6 // (op 585) 
s_mul_hi_i32 s51, s0, s13 // (op 585) 
s_add_i32 s18, s51, s0 // (op 585) 


s_ashr_i32 s0, s18, 31 // (op 585) 
s_cmp_eq_i32 s13, 0 // (op 585) 
s_mov_b32 s19, scc // (op 585) 
s_cmp_eq_i32 s19, 0 // (op 585) 
s_cselect_b32 s51, 0, -1 // (op 585) 

s_and_b32 s19, s49, 31 // (op 585) 
s_lshl_b32 s68, 1, s19 // (op 585) 
s_add_i32 s69, s68, s51 // (op 585) 


s_and_b32 s51, s0, s69 // (op 585) 


s_add_i32 s0, s18, s51 // (op 585) 


s_ashr_i32 s18, s0, s19 // (op 585) 


s_lshl_b32 s0, s49, 24 // (op 585) 
s_ashr_i32 s19, s0, 31 // (op 585) 

s_xor_b32 s0, s18, s19 // (op 585) 

s_sub_i32 s18, s0, s19 // (op 585) 


s_mul_i32 s0, s48, s14 // (op 585) 
s_add_u32 s68, s20, 16 // (op 585) Start of Int64 add, locking SCC
s_addc_u32 s69, s21, 0 // (op 585) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 585) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 585) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 4 // (op 585) 

s_sub_i32 s19, s68, s0 // (op 585) 

s_mul_i32 s51, s18, s19 // (op 585) 


s_sub_u32 s18, s6, s51 // (op 585) 


s_add_u32 s6, s18, s0 // (op 585) 


s_cmp_eq_i32 s7, 0 // (op 585) 
s_mov_b32 s0, scc // (op 585) 

s_cmp_eq_i32 s0, 0 // (op 585) 
s_cselect_b32 s7, s6, s1 // (op 585) 



v_lshl_add_u32 v11, s7, 4, v13 // (op 585) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 585) 
s_lshr_b32 s51, s0, 1 // (op 585) 

v_lshl_add_u32 v13, s51, 3, v12 // (op 585) 


v_mul_lo_u32 v12, v13, s30 // (op 585) 

v_add_u32 v13, v11, v12 // (op 585) 


s_mul_i32 s1, s30, s35 // (op 587) most significant: low of low * high
s_mul_i32 s51, s31, s34 // (op 587) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 587) least significant
s_mul_hi_u32 s6, s30, s34 // (op 587) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 587) most significant: sum
s_add_u32 s1, s1, s6 // (op 587) most significant: sum


s_mov_b32 s60, s32 // (op 587) reg expression
s_mov_b32 s61, s33 // (op 587) reg expression
s_mov_b32 s62, s0 // (op 587) reg expression
s_mov_b32 s63, 131072 // (op 587) call()

s_mul_hi_i32 s51, s16, s9 // (op 649) 
v_and_b32 v11, -32, v1 // (op 649) 
s_add_i32 s0, s51, s16 // (op 649) 

v_sub_u32 v12, v1, v11 // (op 649) 

s_ashr_i32 s1, s0, 31 // (op 649) 
v_lshrrev_b32 v11, 5, v1 // (op 649) 
s_cmp_eq_i32 s9, 0 // (op 649) 
s_mov_b32 s51, scc // (op 649) 
s_cmp_eq_i32 s51, 0 // (op 649) 
s_cselect_b32 s6, 0, -1 // (op 649) 

s_and_b32 s7, s10, 31 // (op 649) 
s_lshl_b32 s51, 1, s7 // (op 649) 
s_add_i32 s18, s51, s6 // (op 649) 


s_and_b32 s6, s1, s18 // (op 649) 


s_add_i32 s1, s0, s6 // (op 649) 


s_ashr_i32 s0, s1, s7 // (op 649) 


s_lshl_b32 s1, s10, 24 // (op 649) 
s_ashr_i32 s51, s1, 31 // (op 649) 

s_xor_b32 s1, s0, s51 // (op 649) 

s_sub_i32 s0, s1, s51 // (op 649) 


s_mul_hi_i32 s1, s0, s11 // (op 649) 
s_add_i32 s51, s1, s0 // (op 649) 

s_ashr_i32 s1, s51, 31 // (op 649) 
s_cmp_eq_i32 s11, 0 // (op 649) 
s_mov_b32 s6, scc // (op 649) 
s_cmp_eq_i32 s6, 0 // (op 649) 
s_cselect_b32 s7, 0, -1 // (op 649) 

s_and_b32 s6, s12, 31 // (op 649) 
s_lshl_b32 s18, 1, s6 // (op 649) 
s_add_i32 s19, s18, s7 // (op 649) 


s_and_b32 s7, s1, s19 // (op 649) 


s_add_i32 s1, s51, s7 // (op 649) 


s_ashr_i32 s7, s1, s6 // (op 649) 


s_lshl_b32 s1, s12, 24 // (op 649) 
s_ashr_i32 s6, s1, 31 // (op 649) 

s_xor_b32 s1, s7, s6 // (op 649) 

s_sub_i32 s7, s1, s6 // (op 649) 


s_add_u32 s18, s4, 32 // (op 649) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 649) End of Int64 add, unlocking SCC
s_sub_u32 s68, s18, 1 // (op 649) least significant half; sets scc
s_subb_u32 s69, s19, 0 // (op 649) most significant half; uses scc

s_ashr_i64 s[18:19], s[68:69], 5 // (op 649) 

s_mul_i32 s1, s7, s18 // (op 649) 

s_sub_i32 s51, s0, s1 // (op 649) 


s_mul_hi_i32 s0, s16, s8 // (op 649) 
s_add_i32 s1, s0, s16 // (op 649) 

s_ashr_i32 s0, s1, 31 // (op 649) 
s_cmp_eq_i32 s8, 0 // (op 649) 
s_mov_b32 s6, scc // (op 649) 
s_cmp_eq_i32 s6, 0 // (op 649) 
s_cselect_b32 s7, 0, -1 // (op 649) 

s_and_b32 s6, s15, 31 // (op 649) 
s_lshl_b32 s68, 1, s6 // (op 649) 
s_add_i32 s69, s68, s7 // (op 649) 


s_and_b32 s7, s0, s69 // (op 649) 


s_add_i32 s0, s1, s7 // (op 649) 


s_ashr_i32 s1, s0, s6 // (op 649) 


s_lshl_b32 s0, s15, 24 // (op 649) 
s_ashr_i32 s6, s0, 31 // (op 649) 

s_xor_b32 s0, s1, s6 // (op 649) 

s_sub_i32 s1, s0, s6 // (op 649) 


s_mul_i32 s0, s14, s18 // (op 649) 

s_mul_i32 s6, s50, s0 // (op 649) 

s_mul_i32 s0, s1, s6 // (op 649) 

s_sub_u32 s6, s16, s0 // (op 649) 

s_mul_hi_i32 s0, s6, s13 // (op 649) 
s_add_i32 s7, s0, s6 // (op 649) 


s_ashr_i32 s0, s7, 31 // (op 649) 
s_cmp_eq_i32 s13, 0 // (op 649) 
s_mov_b32 s6, scc // (op 649) 
s_cmp_eq_i32 s6, 0 // (op 649) 
s_cselect_b32 s18, 0, -1 // (op 649) 

s_and_b32 s6, s49, 31 // (op 649) 
s_lshl_b32 s19, 1, s6 // (op 649) 
s_add_i32 s68, s19, s18 // (op 649) 


s_and_b32 s18, s0, s68 // (op 649) 


s_add_i32 s0, s7, s18 // (op 649) 


s_ashr_i32 s7, s0, s6 // (op 649) 


s_lshl_b32 s0, s49, 24 // (op 649) 
s_ashr_i32 s6, s0, 31 // (op 649) 

s_xor_b32 s0, s7, s6 // (op 649) 

s_sub_i32 s7, s0, s6 // (op 649) 


s_cmp_eq_i32 s1, 0 // (op 649) 
s_mov_b32 s0, scc // (op 649) 

s_cmp_eq_i32 s0, 0 // (op 649) 
s_cselect_b32 s1, s7, s51 // (op 649) 



v_lshl_add_u32 v14, s1, 5, v12 // (op 649) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 649) 
s_lshr_b32 s51, s0, 1 // (op 649) 

v_lshl_add_u32 v12, s51, 3, v11 // (op 649) 


v_mul_lo_u32 v11, v12, s36 // (op 649) 
v_add_u32 v11, v11, v14 // (op 649) 


s_mul_i32 s1, s36, s35 // (op 651) most significant: low of low * high
s_mul_i32 s51, s37, s34 // (op 651) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 651) least significant
s_mul_hi_u32 s6, s36, s34 // (op 651) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 651) most significant: sum
s_add_u32 s1, s1, s6 // (op 651) most significant: sum


s_mov_b32 s64, s38 // (op 651) reg expression
s_mov_b32 s65, s39 // (op 651) reg expression
s_mov_b32 s66, s0 // (op 651) reg expression
s_mov_b32 s67, 131072 // (op 651) call()

v_lshlrev_b32 v12, 4, v1 // (op 742) 
s_nop 0
v_readfirstlane_b32 s51, v12 // (op 742) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 5, v1 // (op 794) 
v_mul_lo_u32 v14, v12, 6 // (op 794) 
v_lshrrev_b32 v15, 3, v14 // (op 794) 

v_lshrrev_b32 v14, 4, v12 // (op 794) 

v_lshlrev_b32 v12, 2, v14 // (op 794) 

v_add_u32 v14, v15, v12 // (op 794) 


v_and_b32 v12, -32, v1 // (op 847) 
v_sub_u32 v15, v1, v12 // (op 847) 

v_and_b32 v12, 4294967264, v1 // (op 847) 
v_add_u32 v16, v12, v15 // (op 847) 


s_ashr_i64 s[0:1], s[24:25], 7 // (op 166) 
s_and_b32 s6, s0, 4294967294 // (op 166) 

s_ashr_i64 s[0:1], s[24:25], 7 // (op 165) 
s_mov_b32 s7, s0 // (op 165) convert

s_cmp_lt_u32 s6, s7 // (op 165) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165:
s_barrier  // (op 1035) 
s_add_u32 s6, s6, 1 // (op 167) 


v_mov_b64 v[30:31], v[32:33] // (op 425) 
v_mov_b64 v[28:29], v[30:31] // (op 425) reg expression

s_mov_b32 m0, s51 // (op 425) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 425) Load value direct to lds


s_lshl_b64 s[0:1], s[2:3], 7 // (op 444) 
v_mov_b32 v12, s0 // (op 444) 
v_mov_b32 v15, s1 // (op 444) 
v_add_co_u32 v32, s[18:19], v32, v12 // (op 444) least significant half
v_addc_co_u32 v33, s[18:19], v33, v15, s[18:19] // (op 444) most significant half






v_mov_b32 v12, v10 // (op 183) 
v_mov_b32 v15, v12 // (op 183) reg expression

buffer_load_dwordx3 v[18:20], v15, s[56:59], 0 offen // (op 183) Load value
buffer_load_dwordx3 v[22:24], v15, s[56:59], 0 offen offset:12 // (op 183) Load value


s_lshl_b64 s[0:1], s[26:27], 7 // (op 513) 
s_mul_i32 s19, s1, 6 // (op 513) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 513) least significant
s_mul_hi_u32 s7, s0, 6 // (op 513) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 513) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 513) 

v_add_u32 v10, v10, s0 // (op 513) 



s_barrier  // (op 1048) 
v_mov_b32 v12, v11 // (op 184) 



v_mov_b32 v15, v12 // (op 184) reg expression

s_waitcnt vmcnt(1)
ds_write_b96 v14, v[18:20] offset:4096 // (op 181) Store local data 
buffer_load_ubyte v17, v15, s[64:67], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s7, s36, 2 // (op 641) 
v_add_u32 v11, v11, s7 // (op 641) 

s_waitcnt vmcnt(1)
ds_write_b96 v14, v[22:24] offset:4112 // (op 181) Store local data 

s_barrier  // (op 1038) 



s_barrier  // (op 1062) 
ds_read_b64_tr_b8 v[18:19], v0 // (op 173) Transpose load local data 



ds_read_b64_tr_b8 v[20:21], v0 offset:128 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v0 offset:1024 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:1152 // (op 173) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v17 offset:8192 // (op 182) Store local data 

s_waitcnt lgkmcnt(5)
s_barrier  // (op 1052) 



ds_read_b96_tr_b6 v[26:28], v3 offset:4096 // (op 175) Transpose load local data 
ds_read_b96_tr_b6 v[29:31], v3 offset:4608 // (op 175) Transpose load local data 

v_mov_b64 v[34:35], s[30:31] // (op 177) convert

v_mov_b32 v12, v13 // (op 177) 
v_mov_b32 v15, v12 // (op 177) reg expression
v_mov_b32 v17, v15 // (op 177) 
buffer_load_ubyte v36, v17, s[60:63], 0 offen offset: 0 // (op 177) Load value




s_lshl_b32 s7, s30, 2 // (op 577) 
v_add_u32 v13, v13, s7 // (op 577) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1066) 


ds_read_u8 v12, v4 offset:8192 // (op 179) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:31], a[0:3], v36, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 180) 




s_ashr_i64 s[0:1], s[24:25], 7 // (op 165) 
s_mov_b32 s7, s0 // (op 165) convert

s_cmp_lt_u32 s6, s7 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopTop_KLoopTail_165 if true)
RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1130) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
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

s_mul_hi_i32 s0, s16, s9 // (op 889) 
v_lshrrev_b32 v0, 7, v1 // (op 889) 
s_add_i32 s1, s0, s16 // (op 889) 

v_and_b32 v3, 63, v1 // (op 889) 
s_ashr_i32 s0, s1, 31 // (op 889) 
v_lshrrev_b32 v10, 4, v3 // (op 889) 
s_cmp_eq_i32 s9, 0 // (op 889) 
s_mov_b32 s2, scc // (op 889) 
v_lshlrev_b32 v11, 2, v10 // (op 889) 

s_cmp_eq_i32 s2, 0 // (op 889) 
s_cselect_b32 s3, 0, -1 // (op 889) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 889) 


s_and_b32 s2, s10, 31 // (op 889) 
v_lshrrev_b32 v0, 2, v3 // (op 889) 

s_lshl_b32 s6, 1, s2 // (op 889) 
v_and_b32 v3, 3, v0 // (op 889) 

s_add_i32 s7, s6, s3 // (op 889) 


v_and_b32 v0, 3, v1 // (op 889) 
s_and_b32 s3, s0, s7 // (op 889) 


v_lshl_add_u32 v11, v3, 2, v0 // (op 889) 


s_add_i32 s0, s1, s3 // (op 889) 


v_lshrrev_b32 v0, 6, v1 // (op 889) 
s_ashr_i32 s1, s0, s2 // (op 889) 


v_and_b32 v3, 1, v0 // (op 889) 

s_lshl_b32 s0, s10, 24 // (op 889) 
v_lshl_add_u32 v0, v3, 4, v11 // (op 889) 


s_ashr_i32 s2, s0, 31 // (op 889) 

s_xor_b32 s0, s1, s2 // (op 889) 

s_sub_i32 s1, s0, s2 // (op 889) 


s_mul_hi_i32 s0, s1, s11 // (op 889) 
s_add_i32 s2, s0, s1 // (op 889) 

s_ashr_i32 s0, s2, 31 // (op 889) 
s_cmp_eq_i32 s11, 0 // (op 889) 
s_mov_b32 s3, scc // (op 889) 
s_cmp_eq_i32 s3, 0 // (op 889) 
s_cselect_b32 s6, 0, -1 // (op 889) 

s_and_b32 s3, s12, 31 // (op 889) 
s_lshl_b32 s7, 1, s3 // (op 889) 
s_add_i32 s18, s7, s6 // (op 889) 


s_and_b32 s19, s0, s18 // (op 889) 


s_add_i32 s0, s2, s19 // (op 889) 


s_ashr_i32 s2, s0, s3 // (op 889) 


s_lshl_b32 s0, s12, 24 // (op 889) 
s_ashr_i32 s3, s0, 31 // (op 889) 

s_xor_b32 s0, s2, s3 // (op 889) 

s_sub_i32 s2, s0, s3 // (op 889) 


s_mul_i32 s0, s1, s14 // (op 889) 
s_sub_u32 s3, s16, s0 // (op 889) 

s_mul_i32 s0, s2, s14 // (op 889) 
s_add_u32 s0, s0, s3 // (op 889) 

s_mul_hi_i32 s3, s16, s8 // (op 889) 
s_add_i32 s6, s3, s16 // (op 889) 

s_ashr_i32 s3, s6, 31 // (op 889) 
s_cmp_eq_i32 s8, 0 // (op 889) 
s_mov_b32 s7, scc // (op 889) 
s_cmp_eq_i32 s7, 0 // (op 889) 
s_cselect_b32 s18, 0, -1 // (op 889) 

s_and_b32 s7, s15, 31 // (op 889) 
s_lshl_b32 s19, 1, s7 // (op 889) 
s_add_i32 s22, s19, s18 // (op 889) 


s_and_b32 s18, s3, s22 // (op 889) 


s_add_i32 s3, s6, s18 // (op 889) 


s_ashr_i32 s6, s3, s7 // (op 889) 


s_lshl_b32 s3, s15, 24 // (op 889) 
s_ashr_i32 s7, s3, 31 // (op 889) 

s_xor_b32 s3, s6, s7 // (op 889) 

s_sub_i32 s6, s3, s7 // (op 889) 


s_add_u32 s18, s4, 32 // (op 889) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 889) End of Int64 add, unlocking SCC
s_sub_u32 s22, s18, 1 // (op 889) least significant half; sets scc
s_subb_u32 s23, s19, 0 // (op 889) most significant half; uses scc

s_ashr_i64 s[18:19], s[22:23], 5 // (op 889) 

s_mul_i32 s3, s14, s18 // (op 889) 
s_mul_i32 s7, s50, s3 // (op 889) 

s_mul_i32 s3, s6, s7 // (op 889) 

s_sub_u32 s7, s16, s3 // (op 889) 

s_mov_b32 s3, s7 // (op 889) 
s_mul_hi_i32 s22, s3, s13 // (op 889) 
s_add_i32 s23, s22, s3 // (op 889) 


s_ashr_i32 s3, s23, 31 // (op 889) 
s_cmp_eq_i32 s13, 0 // (op 889) 
s_mov_b32 s22, scc // (op 889) 
s_cmp_eq_i32 s22, 0 // (op 889) 
s_cselect_b32 s24, 0, -1 // (op 889) 

s_and_b32 s22, s49, 31 // (op 889) 
s_lshl_b32 s25, 1, s22 // (op 889) 
s_add_i32 s26, s25, s24 // (op 889) 


s_and_b32 s24, s3, s26 // (op 889) 


s_add_i32 s3, s23, s24 // (op 889) 


s_ashr_i32 s23, s3, s22 // (op 889) 


s_lshl_b32 s3, s49, 24 // (op 889) 
s_ashr_i32 s22, s3, 31 // (op 889) 

s_xor_b32 s3, s23, s22 // (op 889) 

s_sub_i32 s23, s3, s22 // (op 889) 


s_mul_i32 s3, s48, s14 // (op 889) 
s_add_u32 s24, s20, 16 // (op 889) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 889) End of Int64 add, unlocking SCC
s_sub_u32 s26, s24, 1 // (op 889) least significant half; sets scc
s_subb_u32 s27, s25, 0 // (op 889) most significant half; uses scc

s_ashr_i64 s[24:25], s[26:27], 4 // (op 889) 

s_sub_i32 s22, s24, s3 // (op 889) 

s_mul_i32 s24, s23, s22 // (op 889) 

s_sub_u32 s22, s7, s24 // (op 889) 


s_add_u32 s7, s22, s3 // (op 889) 


s_cmp_eq_i32 s6, 0 // (op 889) 
s_mov_b32 s3, scc // (op 889) 

s_cmp_eq_i32 s3, 0 // (op 889) 
s_cselect_b32 s6, s7, s0 // (op 889) 


s_mul_i32 s0, s2, s18 // (op 889) 


v_lshl_add_u32 v3, s6, 4, v10 // (op 889) 


s_sub_i32 s2, s1, s0 // (op 889) 


s_cmp_eq_i32 s3, 0 // (op 889) 
s_cselect_b32 s22, s23, s2 // (op 889) 



v_lshl_add_u32 v10, s22, 5, v0 // (op 889) 


v_mul_lo_u32 v0, v10, s40 // (op 889) 

v_add_lshl_u32 v10, v3, v0, 1 // (op 889) 


s_mul_i32 s7, s40, s5 // (op 891) most significant: low of low * high
s_mul_i32 s0, s41, s4 // (op 891) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 891) least significant
s_mul_hi_u32 s1, s40, s4 // (op 891) most significant: high of low * low
s_add_u32 s7, s7, s0 // (op 891) most significant: sum
s_add_u32 s7, s7, s1 // (op 891) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 891) 

s_mov_b32 s0, s42 // (op 891) reg expression
s_mov_b32 s1, s43 // (op 891) reg expression
s_mov_b32 s2, s18 // (op 891) reg expression
s_mov_b32 s3, 131072 // (op 891) call()


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v10, s[0:3], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:
s_mul_hi_i32 s6, s16, s9 // (op 712) 
v_lshrrev_b32 v0, 7, v1 // (op 712) 
s_add_i32 s7, s6, s16 // (op 712) 

v_and_b32 v3, 63, v1 // (op 712) 
s_ashr_i32 s6, s7, 31 // (op 712) 
v_lshrrev_b32 v4, 4, v3 // (op 712) 
s_cmp_eq_i32 s9, 0 // (op 712) 
s_mov_b32 s18, scc // (op 712) 
v_lshlrev_b32 v5, 2, v4 // (op 712) 

s_cmp_eq_i32 s18, 0 // (op 712) 
s_cselect_b32 s19, 0, -1 // (op 712) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 712) 


s_and_b32 s18, s10, 31 // (op 712) 
v_lshrrev_b32 v0, 2, v3 // (op 712) 

s_lshl_b32 s22, 1, s18 // (op 712) 
v_and_b32 v3, 3, v0 // (op 712) 

s_add_i32 s23, s22, s19 // (op 712) 


v_and_b32 v0, 3, v1 // (op 712) 
s_and_b32 s19, s6, s23 // (op 712) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 712) 


s_add_i32 s6, s7, s19 // (op 712) 


v_lshrrev_b32 v0, 6, v1 // (op 712) 
s_ashr_i32 s7, s6, s18 // (op 712) 


v_and_b32 v3, 1, v0 // (op 712) 

s_lshl_b32 s6, s10, 24 // (op 712) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 712) 


s_ashr_i32 s18, s6, 31 // (op 712) 

s_xor_b32 s6, s7, s18 // (op 712) 

s_sub_i32 s7, s6, s18 // (op 712) 


s_mul_hi_i32 s6, s7, s11 // (op 712) 
s_add_i32 s18, s6, s7 // (op 712) 

s_ashr_i32 s6, s18, 31 // (op 712) 
s_cmp_eq_i32 s11, 0 // (op 712) 
s_mov_b32 s19, scc // (op 712) 
s_cmp_eq_i32 s19, 0 // (op 712) 
s_cselect_b32 s22, 0, -1 // (op 712) 

s_and_b32 s19, s12, 31 // (op 712) 
s_lshl_b32 s23, 1, s19 // (op 712) 
s_add_i32 s24, s23, s22 // (op 712) 


s_and_b32 s22, s6, s24 // (op 712) 


s_add_i32 s6, s18, s22 // (op 712) 


s_ashr_i32 s18, s6, s19 // (op 712) 


s_lshl_b32 s6, s12, 24 // (op 712) 
s_ashr_i32 s19, s6, 31 // (op 712) 

s_xor_b32 s6, s18, s19 // (op 712) 

s_sub_i32 s18, s6, s19 // (op 712) 


s_mul_i32 s6, s7, s14 // (op 712) 
s_sub_u32 s19, s16, s6 // (op 712) 

s_mul_i32 s6, s18, s14 // (op 712) 
s_add_u32 s6, s6, s19 // (op 712) 

s_mul_hi_i32 s19, s16, s8 // (op 712) 
s_add_i32 s22, s19, s16 // (op 712) 

s_ashr_i32 s19, s22, 31 // (op 712) 
s_cmp_eq_i32 s8, 0 // (op 712) 
s_mov_b32 s23, scc // (op 712) 
s_cmp_eq_i32 s23, 0 // (op 712) 
s_cselect_b32 s24, 0, -1 // (op 712) 

s_and_b32 s23, s15, 31 // (op 712) 
s_lshl_b32 s25, 1, s23 // (op 712) 
s_add_i32 s26, s25, s24 // (op 712) 


s_and_b32 s24, s19, s26 // (op 712) 


s_add_i32 s19, s22, s24 // (op 712) 


s_ashr_i32 s22, s19, s23 // (op 712) 


s_lshl_b32 s19, s15, 24 // (op 712) 
s_ashr_i32 s23, s19, 31 // (op 712) 

s_xor_b32 s19, s22, s23 // (op 712) 

s_sub_i32 s22, s19, s23 // (op 712) 


s_add_u32 s38, s4, 32 // (op 712) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 712) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 712) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 712) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 5 // (op 712) 

s_mul_i32 s19, s14, s38 // (op 712) 
s_mul_i32 s23, s50, s19 // (op 712) 

s_mul_i32 s19, s22, s23 // (op 712) 

s_sub_u32 s23, s16, s19 // (op 712) 

s_mov_b32 s19, s23 // (op 712) 
s_mul_hi_i32 s24, s19, s13 // (op 712) 
s_add_i32 s25, s24, s19 // (op 712) 


s_ashr_i32 s19, s25, 31 // (op 712) 
s_cmp_eq_i32 s13, 0 // (op 712) 
s_mov_b32 s24, scc // (op 712) 
s_cmp_eq_i32 s24, 0 // (op 712) 
s_cselect_b32 s26, 0, -1 // (op 712) 

s_and_b32 s24, s49, 31 // (op 712) 
s_lshl_b32 s27, 1, s24 // (op 712) 
s_add_i32 s28, s27, s26 // (op 712) 


s_and_b32 s26, s19, s28 // (op 712) 


s_add_i32 s19, s25, s26 // (op 712) 


s_ashr_i32 s25, s19, s24 // (op 712) 


s_lshl_b32 s19, s49, 24 // (op 712) 
s_ashr_i32 s24, s19, 31 // (op 712) 

s_xor_b32 s19, s25, s24 // (op 712) 

s_sub_i32 s25, s19, s24 // (op 712) 


s_mul_i32 s19, s48, s14 // (op 712) 
s_add_u32 s26, s20, 16 // (op 712) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 712) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 712) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 712) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 712) 

s_sub_i32 s24, s26, s19 // (op 712) 

s_mul_i32 s26, s25, s24 // (op 712) 

s_sub_u32 s24, s23, s26 // (op 712) 


s_add_u32 s23, s24, s19 // (op 712) 


s_cmp_eq_i32 s22, 0 // (op 712) 
s_mov_b32 s19, scc // (op 712) 

s_cmp_eq_i32 s19, 0 // (op 712) 
s_cselect_b32 s22, s23, s6 // (op 712) 


s_mul_i32 s6, s18, s38 // (op 712) 


v_lshl_add_u32 v3, s22, 4, v4 // (op 712) 


s_sub_i32 s18, s7, s6 // (op 712) 


s_cmp_eq_i32 s19, 0 // (op 712) 
s_cselect_b32 s6, s25, s18 // (op 712) 



v_lshl_add_u32 v4, s6, 5, v0 // (op 712) 


v_mul_lo_u32 v0, v4, s44 // (op 712) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 712) 


s_mul_i32 s7, s44, s5 // (op 714) most significant: low of low * high
s_mul_i32 s18, s45, s4 // (op 714) most significant: low of high * low
s_mul_i32 s6, s44, s4 // (op 714) least significant
s_mul_hi_u32 s19, s44, s4 // (op 714) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 714) most significant: sum
s_add_u32 s7, s7, s19 // (op 714) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 714) 

s_mov_b32 s36, s46 // (op 714) reg expression
s_mov_b32 s37, s47 // (op 714) reg expression
s_mov_b32 s38, s18 // (op 714) reg expression
s_mov_b32 s39, 131072 // (op 714) call()





buffer_load_dwordx2 v[6:7], v4, s[36:39], 0 offen // (op 138) Load value



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

s_mul_hi_i32 s6, s16, s9 // (op 906) 
v_lshrrev_b32 v0, 7, v1 // (op 906) 
s_add_i32 s7, s6, s16 // (op 906) 

v_and_b32 v3, 63, v1 // (op 906) 
s_ashr_i32 s6, s7, 31 // (op 906) 
v_lshrrev_b32 v4, 4, v3 // (op 906) 
s_cmp_eq_i32 s9, 0 // (op 906) 
s_mov_b32 s18, scc // (op 906) 
v_lshlrev_b32 v5, 2, v4 // (op 906) 

s_cmp_eq_i32 s18, 0 // (op 906) 
s_cselect_b32 s19, 0, -1 // (op 906) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 906) 


s_and_b32 s18, s10, 31 // (op 906) 
v_lshrrev_b32 v0, 2, v3 // (op 906) 

s_lshl_b32 s22, 1, s18 // (op 906) 
v_and_b32 v3, 3, v0 // (op 906) 

s_add_i32 s23, s22, s19 // (op 906) 


v_and_b32 v0, 3, v1 // (op 906) 
s_and_b32 s19, s6, s23 // (op 906) 


v_lshl_add_u32 v5, v3, 2, v0 // (op 906) 


s_add_i32 s6, s7, s19 // (op 906) 


v_lshrrev_b32 v0, 6, v1 // (op 906) 
s_ashr_i32 s7, s6, s18 // (op 906) 


v_and_b32 v3, 1, v0 // (op 906) 

s_lshl_b32 s6, s10, 24 // (op 906) 
v_lshl_add_u32 v0, v3, 4, v5 // (op 906) 


s_ashr_i32 s18, s6, 31 // (op 906) 

s_xor_b32 s6, s7, s18 // (op 906) 

s_sub_i32 s7, s6, s18 // (op 906) 


s_mul_hi_i32 s6, s7, s11 // (op 906) 
s_add_i32 s18, s6, s7 // (op 906) 

s_ashr_i32 s6, s18, 31 // (op 906) 
s_cmp_eq_i32 s11, 0 // (op 906) 
s_mov_b32 s19, scc // (op 906) 
s_cmp_eq_i32 s19, 0 // (op 906) 
s_cselect_b32 s22, 0, -1 // (op 906) 

s_and_b32 s19, s12, 31 // (op 906) 
s_lshl_b32 s23, 1, s19 // (op 906) 
s_add_i32 s24, s23, s22 // (op 906) 


s_and_b32 s22, s6, s24 // (op 906) 


s_add_i32 s6, s18, s22 // (op 906) 


s_ashr_i32 s18, s6, s19 // (op 906) 


s_lshl_b32 s6, s12, 24 // (op 906) 
s_ashr_i32 s19, s6, 31 // (op 906) 

s_xor_b32 s6, s18, s19 // (op 906) 

s_sub_i32 s18, s6, s19 // (op 906) 


s_mul_i32 s6, s7, s14 // (op 906) 
s_sub_u32 s19, s16, s6 // (op 906) 

s_mul_i32 s6, s18, s14 // (op 906) 
s_add_u32 s6, s6, s19 // (op 906) 

s_mul_hi_i32 s19, s16, s8 // (op 906) 
s_add_i32 s22, s19, s16 // (op 906) 

s_ashr_i32 s19, s22, 31 // (op 906) 
s_cmp_eq_i32 s8, 0 // (op 906) 
s_mov_b32 s23, scc // (op 906) 
s_cmp_eq_i32 s23, 0 // (op 906) 
s_cselect_b32 s24, 0, -1 // (op 906) 

s_and_b32 s23, s15, 31 // (op 906) 
s_lshl_b32 s25, 1, s23 // (op 906) 
s_add_i32 s26, s25, s24 // (op 906) 


s_and_b32 s24, s19, s26 // (op 906) 


s_add_i32 s19, s22, s24 // (op 906) 


s_ashr_i32 s22, s19, s23 // (op 906) 


s_lshl_b32 s19, s15, 24 // (op 906) 
s_ashr_i32 s23, s19, 31 // (op 906) 

s_xor_b32 s19, s22, s23 // (op 906) 

s_sub_i32 s22, s19, s23 // (op 906) 


s_add_u32 s38, s4, 32 // (op 906) Start of Int64 add, locking SCC
s_addc_u32 s39, s5, 0 // (op 906) End of Int64 add, unlocking SCC
s_sub_u32 s36, s38, 1 // (op 906) least significant half; sets scc
s_subb_u32 s37, s39, 0 // (op 906) most significant half; uses scc

s_ashr_i64 s[38:39], s[36:37], 5 // (op 906) 

s_mul_i32 s19, s14, s38 // (op 906) 
s_mul_i32 s23, s50, s19 // (op 906) 

s_mul_i32 s19, s22, s23 // (op 906) 

s_sub_u32 s23, s16, s19 // (op 906) 

s_mov_b32 s19, s23 // (op 906) 
s_mul_hi_i32 s24, s19, s13 // (op 906) 
s_add_i32 s25, s24, s19 // (op 906) 


s_ashr_i32 s19, s25, 31 // (op 906) 
s_cmp_eq_i32 s13, 0 // (op 906) 
s_mov_b32 s24, scc // (op 906) 
s_cmp_eq_i32 s24, 0 // (op 906) 
s_cselect_b32 s26, 0, -1 // (op 906) 

s_and_b32 s24, s49, 31 // (op 906) 
s_lshl_b32 s27, 1, s24 // (op 906) 
s_add_i32 s28, s27, s26 // (op 906) 


s_and_b32 s26, s19, s28 // (op 906) 


s_add_i32 s19, s25, s26 // (op 906) 


s_ashr_i32 s25, s19, s24 // (op 906) 


s_lshl_b32 s19, s49, 24 // (op 906) 
s_ashr_i32 s24, s19, 31 // (op 906) 

s_xor_b32 s19, s25, s24 // (op 906) 

s_sub_i32 s25, s19, s24 // (op 906) 


s_mul_i32 s19, s48, s14 // (op 906) 
s_add_u32 s26, s20, 16 // (op 906) Start of Int64 add, locking SCC
s_addc_u32 s27, s21, 0 // (op 906) End of Int64 add, unlocking SCC
s_sub_u32 s28, s26, 1 // (op 906) least significant half; sets scc
s_subb_u32 s29, s27, 0 // (op 906) most significant half; uses scc

s_ashr_i64 s[26:27], s[28:29], 4 // (op 906) 

s_sub_i32 s24, s26, s19 // (op 906) 

s_mul_i32 s26, s25, s24 // (op 906) 

s_sub_u32 s24, s23, s26 // (op 906) 


s_add_u32 s23, s24, s19 // (op 906) 


s_cmp_eq_i32 s22, 0 // (op 906) 
s_mov_b32 s19, scc // (op 906) 

s_cmp_eq_i32 s19, 0 // (op 906) 
s_cselect_b32 s22, s23, s6 // (op 906) 


s_mul_i32 s6, s18, s38 // (op 906) 


v_lshl_add_u32 v3, s22, 4, v4 // (op 906) 


s_sub_i32 s18, s7, s6 // (op 906) 


s_cmp_eq_i32 s19, 0 // (op 906) 
s_cselect_b32 s6, s25, s18 // (op 906) 



v_lshl_add_u32 v4, s6, 5, v0 // (op 906) 


v_mul_lo_u32 v0, v4, s40 // (op 906) 

v_add_lshl_u32 v4, v3, v0, 1 // (op 906) 


s_mul_i32 s7, s40, s5 // (op 908) most significant: low of low * high
s_mul_i32 s18, s41, s4 // (op 908) most significant: low of high * low
s_mul_i32 s6, s40, s4 // (op 908) least significant
s_mul_hi_u32 s19, s40, s4 // (op 908) most significant: high of low * low
s_add_u32 s7, s7, s18 // (op 908) most significant: sum
s_add_u32 s7, s7, s19 // (op 908) most significant: sum


s_lshl_b64 s[18:19], s[6:7], 1 // (op 908) 

s_mov_b32 s0, s42 // (op 908) reg expression
s_mov_b32 s1, s43 // (op 908) reg expression
s_mov_b32 s2, s18 // (op 908) reg expression
s_mov_b32 s3, 131072 // (op 908) call()


v_pack_b32_f16 v6, v10, v11 // (op 137) 
v_pack_b32_f16 v7, v12, v13 // (op 137) 


buffer_store_dwordx2 v[6:7], v4, s[0:3], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.LRR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end:
.size RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_, .LRR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM__end-RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 60
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12544 // lds bytes
  .amdhsa_accum_offset 56
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F6
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
    macrotile: [16, 32, 128]
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
  - .name: RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_
    .symbol: RR_GEMM_NT_FP8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 12544
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 56
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

