// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_exec_begin:


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
s_load_dwordx2 s[50:51], s[0:1], 176 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_mov_b32 v0, s6 // (op 2) Move value
v_mov_b32 v3, s7 // (op 27) Move value


s_mov_b32 s0, 0 // (op 338) call()
s_mov_b32 s1, 0 // (op 339) call()
s_mov_b32 s6, 0 // (op 340) call()
s_mov_b32 s7, 0 // (op 341) call()
s_mov_b32 s18, 0 // (op 342) call()
s_mov_b32 s19, 0 // (op 343) call()
s_mov_b32 s0, 0 // (op 238) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 237) 
s_cmp_gt_u32 s9, s52 // (op 237) 
s_mov_b32 s53, scc // (op 237) 
s_sub_u32 s54, s9, s52 // (op 237) 

s_cmp_eq_i32 s53, 0 // (op 237) 
s_cselect_b32 s52, 0, s54 // (op 237) 


s_cmp_lt_u32 s8, s52 // (op 237) 
s_mov_b32 s53, scc // (op 237) 
s_cmp_eq_i32 s53, 0 // (op 237) 
s_cselect_b32 s54, s52, s8 // (op 237) 


s_cmp_lt_u32 s0, s54 // (op 237) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_237 // (op 237) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_237 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_237:
v_lshrrev_b32 v4, 7, v1 // (op 1001) 
s_mul_i32 s52, s16, s8 // (op 1001) 
s_add_u32 s52, s52, s0 // (op 1001) 
v_and_b32 v5, 63, v1 // (op 1001) 
s_mul_hi_u32 s53, s52, s13 // (op 1001) 
v_lshrrev_b32 v6, 4, v5 // (op 1001) 
s_sub_u32 s54, s52, s53 // (op 1001) 
v_lshlrev_b32 v7, 2, v6 // (op 1001) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1001) 


s_lshr_b32 s55, s54, 1 // (op 1001) 

s_add_u32 s54, s55, s53 // (op 1001) 


v_lshrrev_b32 v4, 2, v5 // (op 1001) 

s_lshr_b32 s53, s54, s11 // (op 1001) 

v_and_b32 v5, 3, v4 // (op 1001) 

s_and_b32 s54, s11, 2147483648 // (op 1001) 
v_and_b32 v4, 3, v1 // (op 1001) 
s_cmp_eq_i32 s54, 0 // (op 1001) 
s_mov_b32 s55, scc // (op 1001) 

v_lshl_add_u32 v7, v5, 2, v4 // (op 1001) 


s_cmp_eq_i32 s55, 0 // (op 1001) 
s_cselect_b32 s54, s52, s53 // (op 1001) 



v_lshrrev_b32 v4, 6, v1 // (op 1001) 
s_mul_hi_u32 s52, s54, s12 // (op 1001) 
v_and_b32 v5, 1, v4 // (op 1001) 

s_sub_u32 s53, s54, s52 // (op 1001) 
v_lshl_add_u32 v4, v5, 4, v7 // (op 1001) 


s_lshr_b32 s55, s53, 1 // (op 1001) 

s_add_u32 s53, s55, s52 // (op 1001) 


s_lshr_b32 s52, s53, s10 // (op 1001) 

s_and_b32 s53, s10, 2147483648 // (op 1001) 
s_cmp_eq_i32 s53, 0 // (op 1001) 
s_mov_b32 s55, scc // (op 1001) 

s_cmp_eq_i32 s55, 0 // (op 1001) 
s_cselect_b32 s53, s54, s52 // (op 1001) 


s_mul_i32 s52, s53, s14 // (op 1001) 
v_lshl_add_u32 v5, s53, 5, v6 // (op 1001) 


s_sub_u32 s53, s54, s52 // (op 1001) 


v_lshl_add_u32 v6, s53, 5, v4 // (op 1001) 


v_mul_lo_u32 v4, v6, s4 // (op 1001) 

v_add_lshl_u32 v6, v5, v4, 1 // (op 1001) 


s_mul_i32 s53, s4, s23 // (op 1003) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1003) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1003) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1003) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1003) most significant: sum
s_add_u32 s53, s53, s55 // (op 1003) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1003) 

s_mov_b32 s56, s20 // (op 1003) reg expression
s_mov_b32 s57, s21 // (op 1003) reg expression
s_mov_b32 s58, s54 // (op 1003) reg expression
s_mov_b32 s59, 131072 // (op 1003) call()

s_mul_i32 s6, s8, s16 // (op 242) 
s_add_u32 s6, s6, s0 // (op 242) 
s_mul_hi_u32 s52, s6, s13 // (op 243) 
s_sub_u32 s53, s6, s52 // (op 243) 
s_lshr_b32 s54, s53, 1 // (op 243) 

s_add_u32 s53, s54, s52 // (op 243) 


s_lshr_b32 s52, s53, s11 // (op 243) 

s_and_b32 s53, s11, 2147483648 // (op 243) 
s_cmp_eq_i32 s53, 0 // (op 243) 
s_mov_b32 s54, scc // (op 243) 

s_cmp_eq_i32 s54, 0 // (op 243) 
s_cselect_b32 s53, s6, s52 // (op 243) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 243) 
s_mul_i32 s52, s53, s54 // (op 243) 


s_sub_u32 s18, s6, s52 // (op 243) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 244) 
s_cmp_gt_u32 s52, s18 // (op 244) 
s_mov_b32 s54, scc // (op 244) 
s_sub_u32 s55, s52, s18 // (op 244) 

s_cmp_eq_i32 s54, 0 // (op 244) 
s_cselect_b32 s52, 0, s55 // (op 244) 


s_cmp_gt_u32 s8, s0 // (op 244) 
s_mov_b32 s53, scc // (op 244) 
s_sub_u32 s54, s8, s0 // (op 244) 
s_cmp_eq_i32 s53, 0 // (op 244) 
s_cselect_b32 s55, 0, s54 // (op 244) 


s_cmp_lt_u32 s52, s55 // (op 244) 
s_mov_b32 s53, scc // (op 244) 
s_cmp_eq_i32 s53, 0 // (op 244) 
s_cselect_b32 s54, s55, s52 // (op 244) 



s_cmp_gt_u32 s9, s6 // (op 244) 
s_mov_b32 s55, scc // (op 244) 
s_sub_u32 s52, s9, s6 // (op 244) 
s_cmp_eq_i32 s55, 0 // (op 244) 
s_cselect_b32 s53, 0, s52 // (op 244) 


s_cmp_lt_u32 s54, s53 // (op 244) 
s_mov_b32 s52, scc // (op 244) 
s_cmp_eq_i32 s52, 0 // (op 244) 
s_cselect_b32 s7, s53, s54 // (op 244) 



s_add_u32 s52, s18, s7 // (op 245) 
s_sub_u32 s19, s52, 1 // (op 245) 

v_lshrrev_b32 v4, 7, v1 // (op 693) 
s_mul_i32 s52, s16, s8 // (op 693) 
s_add_u32 s52, s52, s0 // (op 693) 
v_and_b32 v5, 15, v1 // (op 693) 
s_mul_hi_u32 s53, s52, s13 // (op 693) 
v_lshl_add_u32 v7, v4, 4, v5 // (op 693) 


s_sub_u32 s54, s52, s53 // (op 693) 
v_and_b32 v4, 63, v1 // (op 693) 
s_lshr_b32 s55, s54, 1 // (op 693) 

v_lshrrev_b32 v5, 4, v4 // (op 693) 

s_add_u32 s54, s55, s53 // (op 693) 


v_lshlrev_b32 v4, 5, v5 // (op 693) 

s_lshr_b32 s53, s54, s11 // (op 693) 

s_and_b32 s54, s11, 2147483648 // (op 693) 
s_cmp_eq_i32 s54, 0 // (op 693) 
s_mov_b32 s55, scc // (op 693) 

s_cmp_eq_i32 s55, 0 // (op 693) 
s_cselect_b32 s54, s52, s53 // (op 693) 


s_mul_hi_u32 s53, s54, s12 // (op 693) 
s_sub_u32 s55, s54, s53 // (op 693) 
s_lshr_b32 s60, s55, 1 // (op 693) 

s_add_u32 s55, s60, s53 // (op 693) 


s_lshr_b32 s53, s55, s10 // (op 693) 

s_and_b32 s55, s10, 2147483648 // (op 693) 
s_cmp_eq_i32 s55, 0 // (op 693) 
s_mov_b32 s60, scc // (op 693) 

s_cmp_eq_i32 s60, 0 // (op 693) 
s_cselect_b32 s55, s54, s53 // (op 693) 


v_lshl_add_u32 v5, s55, 5, v7 // (op 693) 


s_ashr_i64 s[60:61], s[2:3], 7 // (op 693) 
s_mul_i32 s53, s54, s60 // (op 693) 


s_sub_u32 s54, s52, s53 // (op 693) 


v_lshl_add_u32 v7, s54, 7, v4 // (op 693) 


v_mov_b32 v9, 0 // (op 693) convert
v_mov_b32 v8, v7 // (op 693) convert

v_mul_lo_u32 v11, v5, s25 // (op 693) most significant: low of low * high
v_mul_hi_u32 v4, v5, s24 // (op 693) most significant: high of low * low
v_mul_lo_u32 v10, v5, s24 // (op 693) least significant: low of low * low
v_add_u32 v11, v11, v4 // (op 693) most significant: sum

v_add_co_u32 v10, s[52:53], v10, v8 // (op 693) least significant half
v_addc_co_u32 v11, s[52:53], v11, v9, s[52:53] // (op 693) most significant half



v_lshlrev_b64 v[4:5], 2, v[10:11] // (op 693) 

v_lshrrev_b64 v[8:9], 3, v[4:5] // (op 693) 

v_mov_b32 v7, v8 // (op 693) convert

s_mul_i32 s53, s24, s29 // (op 695) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 695) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 695) least significant
s_mul_hi_u32 s55, s24, s28 // (op 695) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 695) most significant: sum
s_add_u32 s53, s53, s55 // (op 695) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 695) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 695) 

s_mov_b32 s60, s26 // (op 695) reg expression
s_mov_b32 s61, s27 // (op 695) reg expression
s_mov_b32 s62, s52 // (op 695) reg expression
s_mov_b32 s63, 131072 // (op 695) call()

v_lshrrev_b32 v4, 6, v1 // (op 783) 
s_mul_i32 s52, s16, s8 // (op 783) 
s_add_u32 s52, s52, s0 // (op 783) 
v_and_b32 v5, 1, v4 // (op 783) 

s_mul_hi_u32 s53, s52, s13 // (op 783) 
v_and_b32 v4, 15, v1 // (op 783) 
s_sub_u32 s54, s52, s53 // (op 783) 
v_lshl_add_u32 v8, v5, 4, v4 // (op 783) 


s_lshr_b32 s55, s54, 1 // (op 783) 

v_and_b32 v4, 63, v1 // (op 783) 
s_add_u32 s54, s55, s53 // (op 783) 


v_lshrrev_b32 v5, 4, v4 // (op 783) 

s_lshr_b32 s53, s54, s11 // (op 783) 

v_lshlrev_b32 v4, 5, v5 // (op 783) 

s_and_b32 s54, s11, 2147483648 // (op 783) 
s_cmp_eq_i32 s54, 0 // (op 783) 
s_mov_b32 s55, scc // (op 783) 

s_cmp_eq_i32 s55, 0 // (op 783) 
s_cselect_b32 s54, s52, s53 // (op 783) 


s_mul_hi_u32 s53, s54, s12 // (op 783) 
s_sub_u32 s55, s54, s53 // (op 783) 
s_lshr_b32 s64, s55, 1 // (op 783) 

s_add_u32 s55, s64, s53 // (op 783) 


s_lshr_b32 s53, s55, s10 // (op 783) 

s_and_b32 s55, s10, 2147483648 // (op 783) 
s_cmp_eq_i32 s55, 0 // (op 783) 
s_mov_b32 s64, scc // (op 783) 

s_cmp_eq_i32 s64, 0 // (op 783) 
s_cselect_b32 s55, s54, s53 // (op 783) 


s_mul_i32 s53, s55, s14 // (op 783) 

s_sub_u32 s55, s54, s53 // (op 783) 

v_lshl_add_u32 v5, s55, 5, v8 // (op 783) 


v_mul_lo_u32 v9, v5, s31 // (op 783) most significant: low of low * high
v_mul_hi_u32 v10, v5, s30 // (op 783) most significant: high of low * low
v_mul_lo_u32 v8, v5, s30 // (op 783) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 783) most significant: sum


s_ashr_i64 s[64:65], s[2:3], 7 // (op 783) 
s_mul_i32 s53, s54, s64 // (op 783) 


s_sub_u32 s54, s52, s53 // (op 783) 


v_lshl_add_u32 v5, s54, 7, v4 // (op 783) 


v_mov_b32 v11, 0 // (op 783) convert
v_mov_b32 v10, v5 // (op 783) convert

v_add_co_u32 v4, s[52:53], v10, v8 // (op 783) least significant half
v_addc_co_u32 v5, s[52:53], v11, v9, s[52:53] // (op 783) most significant half

v_lshlrev_b64 v[4:5], 2, v[4:5] // (op 783) 


v_lshrrev_b64 v[8:9], 3, v[4:5] // (op 783) 

v_mov_b32 v4, v8 // (op 783) convert

s_mul_i32 s53, s30, s23 // (op 785) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 785) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 785) least significant
s_mul_hi_u32 s55, s30, s22 // (op 785) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 785) most significant: sum
s_add_u32 s53, s53, s55 // (op 785) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 785) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 785) 

s_mov_b32 s64, s32 // (op 785) reg expression
s_mov_b32 s65, s33 // (op 785) reg expression
s_mov_b32 s66, s52 // (op 785) reg expression
s_mov_b32 s67, 131072 // (op 785) call()

v_lshrrev_b32 v5, 7, v1 // (op 871) 
s_mul_i32 s52, s16, s8 // (op 871) 
s_add_u32 s52, s52, s0 // (op 871) 
v_and_b32 v8, 15, v1 // (op 871) 
s_mul_hi_u32 s53, s52, s13 // (op 871) 
v_lshl_add_u32 v9, v5, 4, v8 // (op 871) 


s_sub_u32 s54, s52, s53 // (op 871) 
v_and_b32 v5, 63, v1 // (op 871) 
s_lshr_b32 s55, s54, 1 // (op 871) 

v_lshrrev_b32 v8, 4, v5 // (op 871) 

s_add_u32 s54, s55, s53 // (op 871) 


s_lshr_b32 s53, s54, s11 // (op 871) 

s_and_b32 s54, s11, 2147483648 // (op 871) 
s_cmp_eq_i32 s54, 0 // (op 871) 
s_mov_b32 s55, scc // (op 871) 

s_cmp_eq_i32 s55, 0 // (op 871) 
s_cselect_b32 s54, s52, s53 // (op 871) 


s_mul_hi_u32 s53, s54, s12 // (op 871) 
s_sub_u32 s55, s54, s53 // (op 871) 
s_lshr_b32 s68, s55, 1 // (op 871) 

s_add_u32 s55, s68, s53 // (op 871) 


s_lshr_b32 s53, s55, s10 // (op 871) 

s_and_b32 s55, s10, 2147483648 // (op 871) 
s_cmp_eq_i32 s55, 0 // (op 871) 
s_mov_b32 s68, scc // (op 871) 

s_cmp_eq_i32 s68, 0 // (op 871) 
s_cselect_b32 s55, s54, s53 // (op 871) 


v_lshl_add_u32 v5, s55, 5, v9 // (op 871) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 871) 
s_mul_i32 s53, s54, s68 // (op 871) 


s_sub_u32 s54, s52, s53 // (op 871) 


v_lshl_add_u32 v9, s54, 2, v8 // (op 871) 


v_mul_lo_u32 v8, v5, s34 // (op 871) 
v_add_u32 v8, v8, v9 // (op 871) 


s_mul_i32 s53, s34, s29 // (op 873) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 873) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 873) least significant
s_mul_hi_u32 s55, s34, s28 // (op 873) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 873) most significant: sum
s_add_u32 s53, s53, s55 // (op 873) most significant: sum


s_mov_b32 s68, s36 // (op 873) reg expression
s_mov_b32 s69, s37 // (op 873) reg expression
s_mov_b32 s70, s52 // (op 873) reg expression
s_mov_b32 s71, 131072 // (op 873) call()

v_lshrrev_b32 v5, 6, v1 // (op 959) 
s_mul_i32 s52, s16, s8 // (op 959) 
s_add_u32 s52, s52, s0 // (op 959) 
v_and_b32 v9, 1, v5 // (op 959) 

s_mul_hi_u32 s53, s52, s13 // (op 959) 
v_and_b32 v5, 15, v1 // (op 959) 
s_sub_u32 s54, s52, s53 // (op 959) 
v_lshl_add_u32 v10, v9, 4, v5 // (op 959) 


s_lshr_b32 s55, s54, 1 // (op 959) 

v_and_b32 v5, 63, v1 // (op 959) 
s_add_u32 s54, s55, s53 // (op 959) 


v_lshrrev_b32 v9, 4, v5 // (op 959) 

s_lshr_b32 s53, s54, s11 // (op 959) 

s_and_b32 s54, s11, 2147483648 // (op 959) 
s_cmp_eq_i32 s54, 0 // (op 959) 
s_mov_b32 s55, scc // (op 959) 

s_cmp_eq_i32 s55, 0 // (op 959) 
s_cselect_b32 s54, s52, s53 // (op 959) 


s_mul_hi_u32 s53, s54, s12 // (op 959) 
s_sub_u32 s55, s54, s53 // (op 959) 
s_lshr_b32 s72, s55, 1 // (op 959) 

s_add_u32 s55, s72, s53 // (op 959) 


s_lshr_b32 s53, s55, s10 // (op 959) 

s_and_b32 s55, s10, 2147483648 // (op 959) 
s_cmp_eq_i32 s55, 0 // (op 959) 
s_mov_b32 s72, scc // (op 959) 

s_cmp_eq_i32 s72, 0 // (op 959) 
s_cselect_b32 s55, s54, s53 // (op 959) 


s_mul_i32 s53, s55, s14 // (op 959) 

s_sub_u32 s55, s54, s53 // (op 959) 

v_lshl_add_u32 v5, s55, 5, v10 // (op 959) 


v_mul_lo_u32 v10, v5, s38 // (op 959) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 959) 
s_mul_i32 s53, s54, s72 // (op 959) 


s_sub_u32 s54, s52, s53 // (op 959) 


v_lshl_add_u32 v5, s54, 2, v9 // (op 959) 


v_add_u32 v9, v5, v10 // (op 959) 


s_mul_i32 s53, s38, s23 // (op 961) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 961) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 961) least significant
s_mul_hi_u32 s55, s38, s22 // (op 961) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 961) most significant: sum
s_add_u32 s53, s53, s55 // (op 961) most significant: sum


s_mov_b32 s72, s40 // (op 961) reg expression
s_mov_b32 s73, s41 // (op 961) reg expression
s_mov_b32 s74, s52 // (op 961) reg expression
s_mov_b32 s75, 131072 // (op 961) call()

v_accvgpr_write a0, 0.00000 // (op 49) call()
v_accvgpr_write a1, 0.00000 // (op 49) call()
v_accvgpr_write a2, 0.00000 // (op 49) call()
v_accvgpr_write a3, 0.00000 // (op 49) call()
s_mov_b32 s1, 0 // (op 45) call()
s_and_b32 s52, s7, 4294967294 // (op 44) 
s_cmp_lt_u32 s1, s52 // (op 44) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_44 // (op 44) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_44 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_44:


s_add_u32 s1, s1, 2 // (op 46) 
v_mov_b32 v5, v7 // (op 52) 

buffer_load_dwordx4 v[10:13], v5, s[60:63], 0 offen // (op 52) Load value


v_mov_b32 v14, v4 // (op 56) 

buffer_load_dwordx4 v[16:19], v14, s[64:67], 0 offen // (op 56) Load value


v_mov_b32 v15, v8 // (op 60) 

buffer_load_ubyte v20, v15, s[68:71], 0 offen offset: 0 // (op 60) Load value


v_mov_b32 v21, v9 // (op 64) 

buffer_load_ubyte v22, v21, s[72:75], 0 offen offset: 0 // (op 64) Load value



buffer_load_dwordx4 v[24:27], v5, s[60:63], 0 offen offset:64 // (op 526) Load value
s_waitcnt vmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:13], v[16:19], a[0:3], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 66) 








v_add_u32 v7, 128, v7 // (op 683) 
buffer_load_dwordx4 v[10:13], v14, s[64:67], 0 offen offset:64 // (op 528) Load value




v_add_u32 v4, 128, v4 // (op 775) 
buffer_load_ubyte v5, v15, s[68:71], 0 offen offset:4 // (op 530) Load value




v_add_u32 v8, 8, v8 // (op 863) 
buffer_load_ubyte v14, v21, s[72:75], 0 offen offset:4 // (op 532) Load value

v_add_u32 v9, 8, v9 // (op 951) 
s_waitcnt vmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[10:13], a[0:3], v5, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 533) 




s_and_b32 s52, s7, 4294967294 // (op 44) 
s_cmp_lt_u32 s1, s52 // (op 44) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_44 // (op 44) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_44 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_44:




s_and_b32 s52, s7, 4294967294 // (op 669) 
v_lshrrev_b32 v4, 7, v1 // (op 669) 
s_add_u32 s53, s0, s52 // (op 669) 

v_and_b32 v5, 15, v1 // (op 669) 
v_lshl_add_u32 v7, v4, 4, v5 // (op 669) 


s_mul_i32 s52, s16, s8 // (op 669) 
s_add_u32 s52, s52, s53 // (op 669) 

v_and_b32 v4, 63, v1 // (op 669) 
s_mul_hi_u32 s53, s52, s13 // (op 669) 
v_lshrrev_b32 v5, 4, v4 // (op 669) 

s_sub_u32 s54, s52, s53 // (op 669) 
v_lshlrev_b32 v4, 5, v5 // (op 669) 

s_lshr_b32 s55, s54, 1 // (op 669) 

s_add_u32 s54, s55, s53 // (op 669) 


s_lshr_b32 s53, s54, s11 // (op 669) 

s_and_b32 s54, s11, 2147483648 // (op 669) 
s_cmp_eq_i32 s54, 0 // (op 669) 
s_mov_b32 s55, scc // (op 669) 

s_cmp_eq_i32 s55, 0 // (op 669) 
s_cselect_b32 s54, s52, s53 // (op 669) 


s_mul_hi_u32 s53, s54, s12 // (op 669) 
s_sub_u32 s55, s54, s53 // (op 669) 
s_lshr_b32 s76, s55, 1 // (op 669) 

s_add_u32 s55, s76, s53 // (op 669) 


s_lshr_b32 s53, s55, s10 // (op 669) 

s_and_b32 s55, s10, 2147483648 // (op 669) 
s_cmp_eq_i32 s55, 0 // (op 669) 
s_mov_b32 s76, scc // (op 669) 

s_cmp_eq_i32 s76, 0 // (op 669) 
s_cselect_b32 s55, s54, s53 // (op 669) 


v_lshl_add_u32 v5, s55, 5, v7 // (op 669) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 669) 
s_mul_i32 s53, s54, s76 // (op 669) 


s_sub_u32 s54, s52, s53 // (op 669) 


v_lshl_add_u32 v7, s54, 7, v4 // (op 669) 


v_mov_b32 v9, 0 // (op 669) convert
v_mov_b32 v8, v7 // (op 669) convert

v_mul_lo_u32 v11, v5, s25 // (op 669) most significant: low of low * high
v_mul_hi_u32 v4, v5, s24 // (op 669) most significant: high of low * low
v_mul_lo_u32 v10, v5, s24 // (op 669) least significant: low of low * low
v_add_u32 v11, v11, v4 // (op 669) most significant: sum

v_add_co_u32 v10, s[52:53], v10, v8 // (op 669) least significant half
v_addc_co_u32 v11, s[52:53], v11, v9, s[52:53] // (op 669) most significant half



v_lshlrev_b64 v[4:5], 2, v[10:11] // (op 669) 

v_lshrrev_b64 v[8:9], 3, v[4:5] // (op 669) 

v_mov_b32 v7, v8 // (op 669) convert

s_mul_i32 s53, s24, s29 // (op 671) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 671) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 671) least significant
s_mul_hi_u32 s55, s24, s28 // (op 671) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 671) most significant: sum
s_add_u32 s53, s53, s55 // (op 671) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 671) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 671) 

s_mov_b32 s60, s26 // (op 671) reg expression
s_mov_b32 s61, s27 // (op 671) reg expression
s_mov_b32 s62, s52 // (op 671) reg expression
s_mov_b32 s63, 131072 // (op 671) call()

s_and_b32 s52, s7, 4294967294 // (op 761) 
v_lshrrev_b32 v4, 6, v1 // (op 761) 
s_add_u32 s53, s0, s52 // (op 761) 

v_and_b32 v5, 1, v4 // (op 761) 

v_and_b32 v4, 15, v1 // (op 761) 
s_mul_i32 s52, s16, s8 // (op 761) 
s_add_u32 s52, s52, s53 // (op 761) 

v_lshl_add_u32 v8, v5, 4, v4 // (op 761) 


s_mul_hi_u32 s53, s52, s13 // (op 761) 
v_and_b32 v4, 63, v1 // (op 761) 
s_sub_u32 s54, s52, s53 // (op 761) 
v_lshrrev_b32 v5, 4, v4 // (op 761) 

s_lshr_b32 s55, s54, 1 // (op 761) 

v_lshlrev_b32 v4, 5, v5 // (op 761) 

s_add_u32 s54, s55, s53 // (op 761) 


s_lshr_b32 s53, s54, s11 // (op 761) 

s_and_b32 s54, s11, 2147483648 // (op 761) 
s_cmp_eq_i32 s54, 0 // (op 761) 
s_mov_b32 s55, scc // (op 761) 

s_cmp_eq_i32 s55, 0 // (op 761) 
s_cselect_b32 s54, s52, s53 // (op 761) 


s_mul_hi_u32 s53, s54, s12 // (op 761) 
s_sub_u32 s55, s54, s53 // (op 761) 
s_lshr_b32 s76, s55, 1 // (op 761) 

s_add_u32 s55, s76, s53 // (op 761) 


s_lshr_b32 s53, s55, s10 // (op 761) 

s_and_b32 s55, s10, 2147483648 // (op 761) 
s_cmp_eq_i32 s55, 0 // (op 761) 
s_mov_b32 s76, scc // (op 761) 

s_cmp_eq_i32 s76, 0 // (op 761) 
s_cselect_b32 s55, s54, s53 // (op 761) 


s_mul_i32 s53, s55, s14 // (op 761) 

s_sub_u32 s55, s54, s53 // (op 761) 

v_lshl_add_u32 v5, s55, 5, v8 // (op 761) 


v_mul_lo_u32 v9, v5, s31 // (op 761) most significant: low of low * high
v_mul_hi_u32 v10, v5, s30 // (op 761) most significant: high of low * low
v_mul_lo_u32 v8, v5, s30 // (op 761) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 761) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 761) 
s_mul_i32 s53, s54, s76 // (op 761) 


s_sub_u32 s54, s52, s53 // (op 761) 


v_lshl_add_u32 v5, s54, 7, v4 // (op 761) 


v_mov_b32 v11, 0 // (op 761) convert
v_mov_b32 v10, v5 // (op 761) convert

v_add_co_u32 v4, s[52:53], v10, v8 // (op 761) least significant half
v_addc_co_u32 v5, s[52:53], v11, v9, s[52:53] // (op 761) most significant half

v_lshlrev_b64 v[4:5], 2, v[4:5] // (op 761) 


v_lshrrev_b64 v[8:9], 3, v[4:5] // (op 761) 

v_mov_b32 v4, v8 // (op 761) convert

s_mul_i32 s53, s30, s23 // (op 763) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 763) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 763) least significant
s_mul_hi_u32 s55, s30, s22 // (op 763) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 763) most significant: sum
s_add_u32 s53, s53, s55 // (op 763) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 763) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 763) 

s_mov_b32 s64, s32 // (op 763) reg expression
s_mov_b32 s65, s33 // (op 763) reg expression
s_mov_b32 s66, s52 // (op 763) reg expression
s_mov_b32 s67, 131072 // (op 763) call()

s_and_b32 s52, s7, 4294967294 // (op 849) 
v_lshrrev_b32 v5, 7, v1 // (op 849) 
s_add_u32 s53, s0, s52 // (op 849) 

v_and_b32 v8, 15, v1 // (op 849) 
v_lshl_add_u32 v9, v5, 4, v8 // (op 849) 


s_mul_i32 s52, s16, s8 // (op 849) 
s_add_u32 s52, s52, s53 // (op 849) 

v_and_b32 v5, 63, v1 // (op 849) 
s_mul_hi_u32 s53, s52, s13 // (op 849) 
v_lshrrev_b32 v8, 4, v5 // (op 849) 

s_sub_u32 s54, s52, s53 // (op 849) 
s_lshr_b32 s55, s54, 1 // (op 849) 

s_add_u32 s54, s55, s53 // (op 849) 


s_lshr_b32 s53, s54, s11 // (op 849) 

s_and_b32 s54, s11, 2147483648 // (op 849) 
s_cmp_eq_i32 s54, 0 // (op 849) 
s_mov_b32 s55, scc // (op 849) 

s_cmp_eq_i32 s55, 0 // (op 849) 
s_cselect_b32 s54, s52, s53 // (op 849) 


s_mul_hi_u32 s53, s54, s12 // (op 849) 
s_sub_u32 s55, s54, s53 // (op 849) 
s_lshr_b32 s76, s55, 1 // (op 849) 

s_add_u32 s55, s76, s53 // (op 849) 


s_lshr_b32 s53, s55, s10 // (op 849) 

s_and_b32 s55, s10, 2147483648 // (op 849) 
s_cmp_eq_i32 s55, 0 // (op 849) 
s_mov_b32 s76, scc // (op 849) 

s_cmp_eq_i32 s76, 0 // (op 849) 
s_cselect_b32 s55, s54, s53 // (op 849) 


v_lshl_add_u32 v5, s55, 5, v9 // (op 849) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 849) 
s_mul_i32 s53, s54, s76 // (op 849) 


s_sub_u32 s54, s52, s53 // (op 849) 


v_lshl_add_u32 v9, s54, 2, v8 // (op 849) 


v_mul_lo_u32 v8, v5, s34 // (op 849) 
v_add_u32 v8, v8, v9 // (op 849) 


s_mul_i32 s53, s34, s29 // (op 851) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 851) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 851) least significant
s_mul_hi_u32 s55, s34, s28 // (op 851) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 851) most significant: sum
s_add_u32 s53, s53, s55 // (op 851) most significant: sum


s_mov_b32 s68, s36 // (op 851) reg expression
s_mov_b32 s69, s37 // (op 851) reg expression
s_mov_b32 s70, s52 // (op 851) reg expression
s_mov_b32 s71, 131072 // (op 851) call()

s_and_b32 s52, s7, 4294967294 // (op 937) 
v_lshrrev_b32 v5, 6, v1 // (op 937) 
s_add_u32 s53, s0, s52 // (op 937) 

v_and_b32 v9, 1, v5 // (op 937) 

v_and_b32 v5, 15, v1 // (op 937) 
s_mul_i32 s52, s16, s8 // (op 937) 
s_add_u32 s52, s52, s53 // (op 937) 

v_lshl_add_u32 v10, v9, 4, v5 // (op 937) 


s_mul_hi_u32 s53, s52, s13 // (op 937) 
v_and_b32 v5, 63, v1 // (op 937) 
s_sub_u32 s54, s52, s53 // (op 937) 
v_lshrrev_b32 v9, 4, v5 // (op 937) 

s_lshr_b32 s55, s54, 1 // (op 937) 

s_add_u32 s54, s55, s53 // (op 937) 


s_lshr_b32 s53, s54, s11 // (op 937) 

s_and_b32 s54, s11, 2147483648 // (op 937) 
s_cmp_eq_i32 s54, 0 // (op 937) 
s_mov_b32 s55, scc // (op 937) 

s_cmp_eq_i32 s55, 0 // (op 937) 
s_cselect_b32 s54, s52, s53 // (op 937) 


s_mul_hi_u32 s53, s54, s12 // (op 937) 
s_sub_u32 s55, s54, s53 // (op 937) 
s_lshr_b32 s76, s55, 1 // (op 937) 

s_add_u32 s55, s76, s53 // (op 937) 


s_lshr_b32 s53, s55, s10 // (op 937) 

s_and_b32 s55, s10, 2147483648 // (op 937) 
s_cmp_eq_i32 s55, 0 // (op 937) 
s_mov_b32 s76, scc // (op 937) 

s_cmp_eq_i32 s76, 0 // (op 937) 
s_cselect_b32 s55, s54, s53 // (op 937) 


s_mul_i32 s53, s55, s14 // (op 937) 

s_sub_u32 s55, s54, s53 // (op 937) 

v_lshl_add_u32 v5, s55, 5, v10 // (op 937) 


v_mul_lo_u32 v10, v5, s38 // (op 937) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 937) 
s_mul_i32 s53, s54, s76 // (op 937) 


s_sub_u32 s54, s52, s53 // (op 937) 


v_lshl_add_u32 v5, s54, 2, v9 // (op 937) 


v_add_u32 v9, v5, v10 // (op 937) 


s_mul_i32 s53, s38, s23 // (op 939) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 939) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 939) least significant
s_mul_hi_u32 s55, s38, s22 // (op 939) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 939) most significant: sum
s_add_u32 s53, s53, s55 // (op 939) most significant: sum


s_mov_b32 s72, s40 // (op 939) reg expression
s_mov_b32 s73, s41 // (op 939) reg expression
s_mov_b32 s74, s52 // (op 939) reg expression
s_mov_b32 s75, 131072 // (op 939) call()

s_and_b32 s52, s7, 4294967294 // (op 486) 
s_cmp_lt_u32 s52, s7 // (op 485) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_485 // (op 485) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_485 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_485:


s_add_u32 s52, s52, 1 // (op 487) 
v_mov_b32 v5, v7 // (op 491) 

buffer_load_dwordx4 v[10:13], v5, s[60:63], 0 offen // (op 491) Load value

v_add_u32 v7, 64, v7 // (op 659) 


v_mov_b32 v5, v4 // (op 493) 

buffer_load_dwordx4 v[14:17], v5, s[64:67], 0 offen // (op 493) Load value

v_add_u32 v4, 64, v4 // (op 753) 


v_mov_b32 v5, v8 // (op 495) 

buffer_load_ubyte v18, v5, s[68:71], 0 offen offset: 0 // (op 495) Load value

v_add_u32 v8, 4, v8 // (op 841) 


v_mov_b32 v5, v9 // (op 497) 

buffer_load_ubyte v19, v5, s[72:75], 0 offen offset: 0 // (op 497) Load value

v_add_u32 v9, 4, v9 // (op 929) 
s_waitcnt vmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:13], v[14:17], a[0:3], v18, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 498) 




s_cmp_lt_u32 s52, s7 // (op 485) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_485 // (op 485) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_485 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_485:





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 265) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_265 // (op 265) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_265
v_lshrrev_b32 v4, 5, v1 // (op 1103) 
v_lshlrev_b32 v5, 2, v4 // (op 1103) 

v_lshl_add_u32 v4, s16, 5, v5 // (op 1103) 

v_and_b32 v5, -32, v1 // (op 1103) 
v_sub_u32 v7, v1, v5 // (op 1103) 

v_lshl_add_u32 v5, v4, 5, v7 // (op 1103) 


v_lshlrev_b32 v4, 2, v5 // (op 1103) 

s_lshl_b32 s52, s15, 12 // (op 1105) 
s_mov_b32 s76, s42 // (op 1105) reg expression
s_mov_b32 s77, s43 // (op 1105) reg expression
s_mov_b32 s78, s52 // (op 1105) reg expression
s_mov_b32 s79, 131072 // (op 1105) call()




buffer_store_dword a0, v4, s[76:79], 0 offen sc1 // (op 262) Store value
buffer_store_dword a1, v4, s[76:79], 0 offen offset:128 sc1 // (op 262) Store value
buffer_store_dword a2, v4, s[76:79], 0 offen offset:256 sc1 // (op 262) Store value
buffer_store_dword a3, v4, s[76:79], 0 offen offset:384 sc1 // (op 262) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 266) 
v_cmp_eq_i32 vcc, v1, 0 // (op 270) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_270 // (op 270) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_270
s_mov_b32 s52, 1 // (op 268) call()
s_mov_b32 s55, 0 // (op 269) convert
s_mov_b32 s54, s16 // (op 269) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 269) 
s_add_u32 s80, s44, s54 // (op 269) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 269) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 269) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_270 // (op 270) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_270
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_270:
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_270:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_265 // (op 265) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_265
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_265:
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_265:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 281) 
s_mov_b32 s52, scc // (op 281) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 281) 
s_mov_b32 s53, s54 // (op 281) convert

s_sub_u32 s54, s53, 1 // (op 281) 

s_cmp_lt_u32 s19, s54 // (op 281) 
s_mov_b32 s53, scc // (op 281) 

s_and_b32 s54, s52, s53 // (op 281) 
s_cmp_lg_u32 s54, 0 // (op 281) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_281 // (op 281) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_281
s_mov_b32 s52, 0 // (op 247) call()
s_mul_i32 s53, s52, s8 // (op 246) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 246) 
s_mov_b32 s80, s54 // (op 246) convert

s_sub_u32 s54, s80, s19 // (op 246) 

s_sub_u32 s55, s54, 1 // (op 246) 

s_cmp_lt_u32 s53, s55 // (op 246) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_246 // (op 246) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_246 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_246:
s_add_u32 s53, s16, 1 // (op 1120) 
v_lshrrev_b32 v4, 5, v1 // (op 1120) 
s_add_u32 s54, s53, s52 // (op 1120) 
s_lshl_b32 s54, s54, 5 // (op 1120) 

v_lshlrev_b32 v5, 2, v4 // (op 1120) 

v_add_lshl_u32 v4, s54, v5, 5 // (op 1120) 


v_and_b32 v5, -32, v1 // (op 1120) 
v_sub_u32 v7, v1, v5 // (op 1120) 

v_add_lshl_u32 v5, v4, v7, 2 // (op 1120) 


s_lshl_b32 s53, s15, 12 // (op 1122) 
s_mov_b32 s76, s42 // (op 1122) reg expression
s_mov_b32 s77, s43 // (op 1122) reg expression
s_mov_b32 s78, s53 // (op 1122) reg expression
s_mov_b32 s79, 131072 // (op 1122) call()

RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_285:
s_add_u32 s53, s16, 1 // (op 283) 
s_add_u32 s54, s53, s52 // (op 283) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 283) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 283) 

s_add_u32 s80, s44, s54 // (op 283) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 283) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 283) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 285) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_285 // (op 285) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_285 if true)
s_barrier  // (op 289) 
v_cmp_eq_i32 vcc, v1, 0 // (op 288) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_288 // (op 288) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_288
s_mov_b32 s53, 0 // (op 286) call()
s_add_u32 s54, s16, 1 // (op 287) 
s_add_u32 s80, s54, s52 // (op 287) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 287) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 287) 

s_add_u32 s54, s44, s80 // (op 287) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 287) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 287) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_288 // (op 288) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_288
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_288:
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_288:




buffer_load_dword v8, v5, s[76:79], 0 offen // (op 300) Load value
buffer_load_dword v9, v5, s[76:79], 0 offen offset:128 // (op 300) Load value
buffer_load_dword v10, v5, s[76:79], 0 offen offset:256 // (op 300) Load value
buffer_load_dword v11, v5, s[76:79], 0 offen offset:384 // (op 300) Load value
v_accvgpr_read v12, a0 // (op 303) reg expression
v_accvgpr_read v13, a1 // (op 303) reg expression
v_accvgpr_read v14, a2 // (op 303) reg expression
v_accvgpr_read v15, a3 // (op 303) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v12, v12, v8 // (op 305) 
s_waitcnt vmcnt(2)
v_add_f32 v13, v13, v9 // (op 305) 
s_waitcnt vmcnt(1)
v_add_f32 v14, v14, v10 // (op 305) 
s_waitcnt vmcnt(0)
v_add_f32 v15, v15, v11 // (op 305) 

v_accvgpr_write a0, v12 // (op 307) reg expression
v_accvgpr_write a1, v13 // (op 307) reg expression
v_accvgpr_write a2, v14 // (op 307) reg expression
v_accvgpr_write a3, v15 // (op 307) reg expression

s_add_u32 s52, s52, 1 // (op 248) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 246) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 246) 
s_mov_b32 s80, s54 // (op 246) convert

s_sub_u32 s54, s80, s19 // (op 246) 

s_sub_u32 s55, s54, 1 // (op 246) 

s_cmp_lt_u32 s53, s55 // (op 246) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_246 // (op 246) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_246 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_246:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_281 // (op 281) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_281
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_281:
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_281:

s_cmp_eq_i32 s18, 0 // (op 324) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_324 // (op 324) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_324
v_cmp_eq_f32 vcc, 0.00000, v0 // (op 101) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_101 // (op 101) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_101
v_accvgpr_read v8, a0 // (op 34) 
v_accvgpr_read v9, a1 // (op 34) 
v_accvgpr_read v10, a2 // (op 34) 
v_accvgpr_read v11, a3 // (op 34) 
v_mul_f32 v12, v3, v8 // (op 34) 
v_mul_f32 v13, v3, v9 // (op 34) 
v_mul_f32 v14, v3, v10 // (op 34) 
v_mul_f32 v15, v3, v11 // (op 34) 

v_lshrrev_b32 v8, 16, v12 // (op 40) 
v_lshrrev_b32 v9, 16, v13 // (op 40) 
v_lshrrev_b32 v10, 16, v14 // (op 40) 
v_lshrrev_b32 v11, 16, v15 // (op 40) 

v_lshrrev_b32 v7, 7, v1 // (op 1035) 
s_mul_i32 s52, s16, s8 // (op 1035) 
s_add_u32 s52, s52, s0 // (op 1035) 
v_and_b32 v4, 63, v1 // (op 1035) 
s_mul_hi_u32 s53, s52, s13 // (op 1035) 
v_lshrrev_b32 v5, 4, v4 // (op 1035) 
s_sub_u32 s54, s52, s53 // (op 1035) 
v_lshlrev_b32 v12, 2, v5 // (op 1035) 

s_lshr_b32 s55, s54, 1 // (op 1035) 

v_lshl_add_u32 v5, v7, 4, v12 // (op 1035) 


s_add_u32 s54, s55, s53 // (op 1035) 


v_lshrrev_b32 v7, 2, v4 // (op 1035) 

s_lshr_b32 s53, s54, s11 // (op 1035) 

v_and_b32 v4, 3, v7 // (op 1035) 

s_and_b32 s54, s11, 2147483648 // (op 1035) 
v_and_b32 v7, 3, v1 // (op 1035) 
s_cmp_eq_i32 s54, 0 // (op 1035) 
s_mov_b32 s55, scc // (op 1035) 

v_lshl_add_u32 v12, v4, 2, v7 // (op 1035) 


s_cmp_eq_i32 s55, 0 // (op 1035) 
s_cselect_b32 s54, s52, s53 // (op 1035) 



v_lshrrev_b32 v4, 6, v1 // (op 1035) 
s_mul_hi_u32 s55, s54, s12 // (op 1035) 
v_and_b32 v7, 1, v4 // (op 1035) 

s_sub_u32 s52, s54, s55 // (op 1035) 
v_lshl_add_u32 v4, v7, 4, v12 // (op 1035) 


s_lshr_b32 s53, s52, 1 // (op 1035) 

s_add_u32 s52, s53, s55 // (op 1035) 


s_lshr_b32 s53, s52, s10 // (op 1035) 

s_and_b32 s52, s10, 2147483648 // (op 1035) 
s_cmp_eq_i32 s52, 0 // (op 1035) 
s_mov_b32 s55, scc // (op 1035) 

s_cmp_eq_i32 s55, 0 // (op 1035) 
s_cselect_b32 s52, s54, s53 // (op 1035) 


s_mul_i32 s53, s52, s14 // (op 1035) 
v_lshl_add_u32 v7, s52, 5, v5 // (op 1035) 


s_sub_u32 s52, s54, s53 // (op 1035) 


v_lshl_add_u32 v5, s52, 5, v4 // (op 1035) 


v_mul_lo_u32 v4, v5, s46 // (op 1035) 

v_add_lshl_u32 v5, v7, v4, 1 // (op 1035) 


s_mul_i32 s53, s46, s23 // (op 1037) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1037) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1037) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1037) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1037) most significant: sum
s_add_u32 s53, s53, s55 // (op 1037) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1037) 

s_mov_b32 s76, s48 // (op 1037) reg expression
s_mov_b32 s77, s49 // (op 1037) reg expression
s_mov_b32 s78, s54 // (op 1037) reg expression
s_mov_b32 s79, 131072 // (op 1037) call()


v_pack_b32_f16 v12, v8, v9 // (op 42) 
v_pack_b32_f16 v13, v10, v11 // (op 42) 


buffer_store_dwordx2 v[12:13], v5, s[76:79], 0 offen // (op 42) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_101 // (op 101) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_101
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_101:



buffer_load_dwordx2 v[4:5], v6, s[56:59], 0 offen // (op 114) Load value
v_accvgpr_read v8, a0 // (op 110) 
v_accvgpr_read v9, a1 // (op 110) 
v_accvgpr_read v10, a2 // (op 110) 
v_accvgpr_read v11, a3 // (op 110) 
v_mul_f32 v12, v3, v8 // (op 110) 
v_mul_f32 v13, v3, v9 // (op 110) 
v_mul_f32 v14, v3, v10 // (op 110) 
v_mul_f32 v15, v3, v11 // (op 110) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 115) 
v_lshrrev_b32 v11, 16, v4 // (op 115) 
v_lshlrev_b32 v10, 16, v10 // (op 115) 
v_lshlrev_b32 v11, 16, v11 // (op 115) 
v_mul_f32 v16, v0, v10 // (op 115) 
v_mul_f32 v17, v0, v11 // (op 115) 
v_and_b32 v10, 65535, v5 // (op 115) 
v_lshrrev_b32 v11, 16, v5 // (op 115) 
v_lshlrev_b32 v10, 16, v10 // (op 115) 
v_lshlrev_b32 v11, 16, v11 // (op 115) 
v_mul_f32 v18, v0, v10 // (op 115) 
v_mul_f32 v19, v0, v11 // (op 115) 


v_add_f32 v8, v16, v12 // (op 111) 
v_add_f32 v9, v17, v13 // (op 111) 
v_add_f32 v10, v18, v14 // (op 111) 
v_add_f32 v11, v19, v15 // (op 111) 


v_lshrrev_b32 v12, 16, v8 // (op 112) 
v_lshrrev_b32 v13, 16, v9 // (op 112) 
v_lshrrev_b32 v14, 16, v10 // (op 112) 
v_lshrrev_b32 v15, 16, v11 // (op 112) 

v_lshrrev_b32 v4, 7, v1 // (op 1052) 
s_mul_i32 s52, s16, s8 // (op 1052) 
s_add_u32 s52, s52, s0 // (op 1052) 
v_and_b32 v5, 63, v1 // (op 1052) 
s_mul_hi_u32 s53, s52, s13 // (op 1052) 
v_lshrrev_b32 v7, 4, v5 // (op 1052) 
s_sub_u32 s54, s52, s53 // (op 1052) 
v_lshlrev_b32 v8, 2, v7 // (op 1052) 

s_lshr_b32 s55, s54, 1 // (op 1052) 

v_lshl_add_u32 v7, v4, 4, v8 // (op 1052) 


s_add_u32 s54, s55, s53 // (op 1052) 


v_lshrrev_b32 v4, 2, v5 // (op 1052) 

s_lshr_b32 s53, s54, s11 // (op 1052) 

v_and_b32 v5, 3, v4 // (op 1052) 

s_and_b32 s54, s11, 2147483648 // (op 1052) 
v_and_b32 v4, 3, v1 // (op 1052) 
s_cmp_eq_i32 s54, 0 // (op 1052) 
s_mov_b32 s55, scc // (op 1052) 

v_lshl_add_u32 v8, v5, 2, v4 // (op 1052) 


s_cmp_eq_i32 s55, 0 // (op 1052) 
s_cselect_b32 s54, s52, s53 // (op 1052) 



v_lshrrev_b32 v4, 6, v1 // (op 1052) 
s_mul_hi_u32 s55, s54, s12 // (op 1052) 
v_and_b32 v5, 1, v4 // (op 1052) 

s_sub_u32 s52, s54, s55 // (op 1052) 
v_lshl_add_u32 v4, v5, 4, v8 // (op 1052) 


s_lshr_b32 s53, s52, 1 // (op 1052) 

s_add_u32 s52, s53, s55 // (op 1052) 


s_lshr_b32 s53, s52, s10 // (op 1052) 

s_and_b32 s52, s10, 2147483648 // (op 1052) 
s_cmp_eq_i32 s52, 0 // (op 1052) 
s_mov_b32 s55, scc // (op 1052) 

s_cmp_eq_i32 s55, 0 // (op 1052) 
s_cselect_b32 s52, s54, s53 // (op 1052) 


s_mul_i32 s53, s52, s14 // (op 1052) 
v_lshl_add_u32 v5, s52, 5, v7 // (op 1052) 


s_sub_u32 s52, s54, s53 // (op 1052) 


v_lshl_add_u32 v7, s52, 5, v4 // (op 1052) 


v_mul_lo_u32 v4, v7, s46 // (op 1052) 

v_add_lshl_u32 v7, v5, v4, 1 // (op 1052) 


s_mul_i32 s53, s46, s23 // (op 1054) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1054) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1054) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1054) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1054) most significant: sum
s_add_u32 s53, s53, s55 // (op 1054) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1054) 

s_mov_b32 s76, s48 // (op 1054) reg expression
s_mov_b32 s77, s49 // (op 1054) reg expression
s_mov_b32 s78, s54 // (op 1054) reg expression
s_mov_b32 s79, 131072 // (op 1054) call()


v_pack_b32_f16 v4, v12, v13 // (op 113) 
v_pack_b32_f16 v5, v14, v15 // (op 113) 


buffer_store_dwordx2 v[4:5], v7, s[76:79], 0 offen // (op 113) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_101:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_324 // (op 324) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_324
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_324:
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_324:

s_add_u32 s0, s0, s7 // (op 239) 
s_mul_i32 s52, s8, s16 // (op 237) 
s_cmp_gt_u32 s9, s52 // (op 237) 
s_mov_b32 s53, scc // (op 237) 
s_sub_u32 s54, s9, s52 // (op 237) 

s_cmp_eq_i32 s53, 0 // (op 237) 
s_cselect_b32 s52, 0, s54 // (op 237) 


s_cmp_lt_u32 s8, s52 // (op 237) 
s_mov_b32 s53, scc // (op 237) 
s_cmp_eq_i32 s53, 0 // (op 237) 
s_cselect_b32 s54, s52, s8 // (op 237) 


s_cmp_lt_u32 s0, s54 // (op 237) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_237 // (op 237) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_237 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_237:









s_mov_b32 s0, 0 // (op 378) call()
s_mov_b32 s1, 0 // (op 379) call()
s_mov_b32 s0, 0 // (op 393) call()
s_mul_i32 s15, s50, s16 // (op 392) 
s_cmp_gt_u32 s51, s15 // (op 392) 
s_mov_b32 s6, scc // (op 392) 
s_sub_u32 s7, s51, s15 // (op 392) 

s_cmp_eq_i32 s6, 0 // (op 392) 
s_cselect_b32 s8, 0, s7 // (op 392) 


s_cmp_lt_u32 s50, s8 // (op 392) 
s_mov_b32 s15, scc // (op 392) 
s_cmp_eq_i32 s15, 0 // (op 392) 
s_cselect_b32 s6, s8, s50 // (op 392) 


s_cmp_lt_u32 s0, s6 // (op 392) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_392 // (op 392) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_392 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_392:
v_lshrrev_b32 v4, 7, v1 // (op 1018) 
s_mul_i32 s15, s16, s50 // (op 1018) 
s_add_u32 s15, s15, s0 // (op 1018) 
v_and_b32 v5, 63, v1 // (op 1018) 
s_add_u32 s6, s15, s9 // (op 1018) 

v_lshrrev_b32 v6, 4, v5 // (op 1018) 
s_mul_hi_u32 s15, s6, s13 // (op 1018) 
v_lshlrev_b32 v7, 2, v6 // (op 1018) 

s_sub_u32 s7, s6, s15 // (op 1018) 
v_lshl_add_u32 v6, v4, 4, v7 // (op 1018) 


s_lshr_b32 s8, s7, 1 // (op 1018) 

v_lshrrev_b32 v4, 2, v5 // (op 1018) 

s_add_u32 s7, s8, s15 // (op 1018) 


v_and_b32 v5, 3, v4 // (op 1018) 

s_lshr_b32 s8, s7, s11 // (op 1018) 

v_and_b32 v4, 3, v1 // (op 1018) 
s_and_b32 s7, s11, 2147483648 // (op 1018) 
v_lshl_add_u32 v7, v5, 2, v4 // (op 1018) 


s_cmp_eq_i32 s7, 0 // (op 1018) 
s_mov_b32 s15, scc // (op 1018) 

v_lshrrev_b32 v4, 6, v1 // (op 1018) 
s_cmp_eq_i32 s15, 0 // (op 1018) 
s_cselect_b32 s7, s6, s8 // (op 1018) 



v_and_b32 v5, 1, v4 // (op 1018) 

s_mul_hi_u32 s6, s7, s12 // (op 1018) 
v_lshl_add_u32 v4, v5, 4, v7 // (op 1018) 


s_sub_u32 s8, s7, s6 // (op 1018) 
s_lshr_b32 s15, s8, 1 // (op 1018) 

s_add_u32 s8, s15, s6 // (op 1018) 


s_lshr_b32 s6, s8, s10 // (op 1018) 

s_and_b32 s8, s10, 2147483648 // (op 1018) 
s_cmp_eq_i32 s8, 0 // (op 1018) 
s_mov_b32 s15, scc // (op 1018) 

s_cmp_eq_i32 s15, 0 // (op 1018) 
s_cselect_b32 s8, s7, s6 // (op 1018) 


s_mul_i32 s6, s8, s14 // (op 1018) 
v_lshl_add_u32 v5, s8, 5, v6 // (op 1018) 


s_sub_u32 s8, s7, s6 // (op 1018) 


v_lshl_add_u32 v6, s8, 5, v4 // (op 1018) 


v_mul_lo_u32 v4, v6, s4 // (op 1018) 

v_add_lshl_u32 v6, v5, v4, 1 // (op 1018) 


s_mul_i32 s19, s4, s23 // (op 1020) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1020) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1020) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1020) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1020) most significant: sum
s_add_u32 s19, s19, s6 // (op 1020) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1020) 

s_mov_b32 s56, s20 // (op 1020) reg expression
s_mov_b32 s57, s21 // (op 1020) reg expression
s_mov_b32 s58, s6 // (op 1020) reg expression
s_mov_b32 s59, 131072 // (op 1020) call()

v_lshrrev_b32 v4, 7, v1 // (op 717) 
s_mul_i32 s15, s16, s50 // (op 717) 
s_add_u32 s15, s15, s0 // (op 717) 
v_and_b32 v5, 15, v1 // (op 717) 
s_add_u32 s6, s15, s9 // (op 717) 

v_lshl_add_u32 v7, v4, 4, v5 // (op 717) 


s_mul_hi_u32 s15, s6, s13 // (op 717) 
v_and_b32 v4, 63, v1 // (op 717) 
s_sub_u32 s7, s6, s15 // (op 717) 
v_lshrrev_b32 v5, 4, v4 // (op 717) 

s_lshr_b32 s8, s7, 1 // (op 717) 

v_lshlrev_b32 v4, 5, v5 // (op 717) 

s_add_u32 s7, s8, s15 // (op 717) 


s_lshr_b32 s8, s7, s11 // (op 717) 

s_and_b32 s7, s11, 2147483648 // (op 717) 
s_cmp_eq_i32 s7, 0 // (op 717) 
s_mov_b32 s15, scc // (op 717) 

s_cmp_eq_i32 s15, 0 // (op 717) 
s_cselect_b32 s7, s6, s8 // (op 717) 


s_mul_hi_u32 s8, s7, s12 // (op 717) 
s_sub_u32 s15, s7, s8 // (op 717) 
s_lshr_b32 s18, s15, 1 // (op 717) 

s_add_u32 s15, s18, s8 // (op 717) 


s_lshr_b32 s8, s15, s10 // (op 717) 

s_and_b32 s15, s10, 2147483648 // (op 717) 
s_cmp_eq_i32 s15, 0 // (op 717) 
s_mov_b32 s18, scc // (op 717) 

s_cmp_eq_i32 s18, 0 // (op 717) 
s_cselect_b32 s15, s7, s8 // (op 717) 


v_lshl_add_u32 v5, s15, 5, v7 // (op 717) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 717) 
s_mul_i32 s8, s7, s18 // (op 717) 


s_sub_u32 s7, s6, s8 // (op 717) 


v_lshl_add_u32 v7, s7, 7, v4 // (op 717) 


v_mov_b32 v9, 0 // (op 717) convert
v_mov_b32 v8, v7 // (op 717) convert

v_mul_lo_u32 v11, v5, s25 // (op 717) most significant: low of low * high
v_mul_hi_u32 v4, v5, s24 // (op 717) most significant: high of low * low
v_mul_lo_u32 v10, v5, s24 // (op 717) least significant: low of low * low
v_add_u32 v11, v11, v4 // (op 717) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v8 // (op 717) least significant half
v_addc_co_u32 v11, s[18:19], v11, v9, s[18:19] // (op 717) most significant half



v_lshlrev_b64 v[4:5], 2, v[10:11] // (op 717) 

v_lshrrev_b64 v[8:9], 3, v[4:5] // (op 717) 

v_mov_b32 v7, v8 // (op 717) convert

s_mul_i32 s19, s24, s29 // (op 719) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 719) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 719) least significant
s_mul_hi_u32 s6, s24, s28 // (op 719) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 719) most significant: sum
s_add_u32 s19, s19, s6 // (op 719) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 719) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 719) 

s_mov_b32 s60, s26 // (op 719) reg expression
s_mov_b32 s61, s27 // (op 719) reg expression
s_mov_b32 s62, s18 // (op 719) reg expression
s_mov_b32 s63, 131072 // (op 719) call()

v_lshrrev_b32 v4, 6, v1 // (op 805) 
s_mul_i32 s15, s16, s50 // (op 805) 
s_add_u32 s15, s15, s0 // (op 805) 
v_and_b32 v5, 1, v4 // (op 805) 

s_add_u32 s6, s15, s9 // (op 805) 

v_and_b32 v4, 15, v1 // (op 805) 
s_mul_hi_u32 s15, s6, s13 // (op 805) 
v_lshl_add_u32 v8, v5, 4, v4 // (op 805) 


s_sub_u32 s7, s6, s15 // (op 805) 
v_and_b32 v4, 63, v1 // (op 805) 
s_lshr_b32 s8, s7, 1 // (op 805) 

v_lshrrev_b32 v5, 4, v4 // (op 805) 

s_add_u32 s7, s8, s15 // (op 805) 


v_lshlrev_b32 v4, 5, v5 // (op 805) 

s_lshr_b32 s8, s7, s11 // (op 805) 

s_and_b32 s7, s11, 2147483648 // (op 805) 
s_cmp_eq_i32 s7, 0 // (op 805) 
s_mov_b32 s15, scc // (op 805) 

s_cmp_eq_i32 s15, 0 // (op 805) 
s_cselect_b32 s7, s6, s8 // (op 805) 


s_mul_hi_u32 s8, s7, s12 // (op 805) 
s_sub_u32 s15, s7, s8 // (op 805) 
s_lshr_b32 s18, s15, 1 // (op 805) 

s_add_u32 s15, s18, s8 // (op 805) 


s_lshr_b32 s8, s15, s10 // (op 805) 

s_and_b32 s15, s10, 2147483648 // (op 805) 
s_cmp_eq_i32 s15, 0 // (op 805) 
s_mov_b32 s18, scc // (op 805) 

s_cmp_eq_i32 s18, 0 // (op 805) 
s_cselect_b32 s15, s7, s8 // (op 805) 


s_mul_i32 s8, s15, s14 // (op 805) 

s_sub_u32 s15, s7, s8 // (op 805) 

v_lshl_add_u32 v5, s15, 5, v8 // (op 805) 


v_mul_lo_u32 v9, v5, s31 // (op 805) most significant: low of low * high
v_mul_hi_u32 v10, v5, s30 // (op 805) most significant: high of low * low
v_mul_lo_u32 v8, v5, s30 // (op 805) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 805) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 805) 
s_mul_i32 s8, s7, s18 // (op 805) 


s_sub_u32 s7, s6, s8 // (op 805) 


v_lshl_add_u32 v5, s7, 7, v4 // (op 805) 


v_mov_b32 v11, 0 // (op 805) convert
v_mov_b32 v10, v5 // (op 805) convert

v_add_co_u32 v4, s[18:19], v10, v8 // (op 805) least significant half
v_addc_co_u32 v5, s[18:19], v11, v9, s[18:19] // (op 805) most significant half

v_lshlrev_b64 v[4:5], 2, v[4:5] // (op 805) 


v_lshrrev_b64 v[8:9], 3, v[4:5] // (op 805) 

v_mov_b32 v4, v8 // (op 805) convert

s_mul_i32 s19, s30, s23 // (op 807) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 807) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 807) least significant
s_mul_hi_u32 s6, s30, s22 // (op 807) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 807) most significant: sum
s_add_u32 s19, s19, s6 // (op 807) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 807) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 807) 

s_mov_b32 s64, s32 // (op 807) reg expression
s_mov_b32 s65, s33 // (op 807) reg expression
s_mov_b32 s66, s18 // (op 807) reg expression
s_mov_b32 s67, 131072 // (op 807) call()

v_lshrrev_b32 v5, 7, v1 // (op 893) 
s_mul_i32 s15, s16, s50 // (op 893) 
s_add_u32 s15, s15, s0 // (op 893) 
v_and_b32 v8, 15, v1 // (op 893) 
s_add_u32 s6, s15, s9 // (op 893) 

v_lshl_add_u32 v9, v5, 4, v8 // (op 893) 


s_mul_hi_u32 s15, s6, s13 // (op 893) 
v_and_b32 v5, 63, v1 // (op 893) 
s_sub_u32 s7, s6, s15 // (op 893) 
v_lshrrev_b32 v8, 4, v5 // (op 893) 

s_lshr_b32 s8, s7, 1 // (op 893) 

s_add_u32 s7, s8, s15 // (op 893) 


s_lshr_b32 s8, s7, s11 // (op 893) 

s_and_b32 s7, s11, 2147483648 // (op 893) 
s_cmp_eq_i32 s7, 0 // (op 893) 
s_mov_b32 s15, scc // (op 893) 

s_cmp_eq_i32 s15, 0 // (op 893) 
s_cselect_b32 s7, s6, s8 // (op 893) 


s_mul_hi_u32 s8, s7, s12 // (op 893) 
s_sub_u32 s15, s7, s8 // (op 893) 
s_lshr_b32 s18, s15, 1 // (op 893) 

s_add_u32 s15, s18, s8 // (op 893) 


s_lshr_b32 s8, s15, s10 // (op 893) 

s_and_b32 s15, s10, 2147483648 // (op 893) 
s_cmp_eq_i32 s15, 0 // (op 893) 
s_mov_b32 s18, scc // (op 893) 

s_cmp_eq_i32 s18, 0 // (op 893) 
s_cselect_b32 s15, s7, s8 // (op 893) 


v_lshl_add_u32 v5, s15, 5, v9 // (op 893) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 893) 
s_mul_i32 s8, s7, s18 // (op 893) 


s_sub_u32 s7, s6, s8 // (op 893) 


v_lshl_add_u32 v9, s7, 2, v8 // (op 893) 


v_mul_lo_u32 v8, v5, s34 // (op 893) 
v_add_u32 v8, v8, v9 // (op 893) 


s_mul_i32 s19, s34, s29 // (op 895) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 895) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 895) least significant
s_mul_hi_u32 s6, s34, s28 // (op 895) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 895) most significant: sum
s_add_u32 s19, s19, s6 // (op 895) most significant: sum


s_mov_b32 s68, s36 // (op 895) reg expression
s_mov_b32 s69, s37 // (op 895) reg expression
s_mov_b32 s70, s18 // (op 895) reg expression
s_mov_b32 s71, 131072 // (op 895) call()

v_lshrrev_b32 v5, 6, v1 // (op 981) 
s_mul_i32 s15, s16, s50 // (op 981) 
s_add_u32 s15, s15, s0 // (op 981) 
v_and_b32 v9, 1, v5 // (op 981) 

s_add_u32 s6, s15, s9 // (op 981) 

v_and_b32 v5, 15, v1 // (op 981) 
s_mul_hi_u32 s15, s6, s13 // (op 981) 
v_lshl_add_u32 v10, v9, 4, v5 // (op 981) 


s_sub_u32 s7, s6, s15 // (op 981) 
v_and_b32 v5, 63, v1 // (op 981) 
s_lshr_b32 s8, s7, 1 // (op 981) 

v_lshrrev_b32 v9, 4, v5 // (op 981) 

s_add_u32 s7, s8, s15 // (op 981) 


s_lshr_b32 s8, s7, s11 // (op 981) 

s_and_b32 s7, s11, 2147483648 // (op 981) 
s_cmp_eq_i32 s7, 0 // (op 981) 
s_mov_b32 s15, scc // (op 981) 

s_cmp_eq_i32 s15, 0 // (op 981) 
s_cselect_b32 s7, s6, s8 // (op 981) 


s_mul_hi_u32 s8, s7, s12 // (op 981) 
s_sub_u32 s15, s7, s8 // (op 981) 
s_lshr_b32 s18, s15, 1 // (op 981) 

s_add_u32 s15, s18, s8 // (op 981) 


s_lshr_b32 s8, s15, s10 // (op 981) 

s_and_b32 s15, s10, 2147483648 // (op 981) 
s_cmp_eq_i32 s15, 0 // (op 981) 
s_mov_b32 s18, scc // (op 981) 

s_cmp_eq_i32 s18, 0 // (op 981) 
s_cselect_b32 s15, s7, s8 // (op 981) 


s_mul_i32 s8, s15, s14 // (op 981) 

s_sub_u32 s15, s7, s8 // (op 981) 

v_lshl_add_u32 v5, s15, 5, v10 // (op 981) 


v_mul_lo_u32 v10, v5, s38 // (op 981) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 981) 
s_mul_i32 s8, s7, s18 // (op 981) 


s_sub_u32 s7, s6, s8 // (op 981) 


v_lshl_add_u32 v5, s7, 2, v9 // (op 981) 


v_add_u32 v9, v5, v10 // (op 981) 


s_mul_i32 s19, s38, s23 // (op 983) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 983) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 983) least significant
s_mul_hi_u32 s6, s38, s22 // (op 983) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 983) most significant: sum
s_add_u32 s19, s19, s6 // (op 983) most significant: sum


s_mov_b32 s72, s40 // (op 983) reg expression
s_mov_b32 s73, s41 // (op 983) reg expression
s_mov_b32 s74, s18 // (op 983) reg expression
s_mov_b32 s75, 131072 // (op 983) call()

v_accvgpr_write a0, 0.00000 // (op 150) call()
v_accvgpr_write a1, 0.00000 // (op 150) call()
v_accvgpr_write a2, 0.00000 // (op 150) call()
v_accvgpr_write a3, 0.00000 // (op 150) call()
s_mov_b32 s1, 0 // (op 152) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 151) 
s_mov_b32 s15, s18 // (op 151) convert

s_and_b32 s6, s15, 4294967294 // (op 151) 

s_cmp_lt_u32 s1, s6 // (op 151) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_151 // (op 151) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_151 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_151:


s_add_u32 s1, s1, 2 // (op 153) 
v_mov_b32 v5, v7 // (op 155) 

buffer_load_dwordx4 v[10:13], v5, s[60:63], 0 offen // (op 155) Load value


v_mov_b32 v14, v4 // (op 157) 

buffer_load_dwordx4 v[16:19], v14, s[64:67], 0 offen // (op 157) Load value


v_mov_b32 v15, v8 // (op 159) 

buffer_load_ubyte v20, v15, s[68:71], 0 offen offset: 0 // (op 159) Load value


v_mov_b32 v21, v9 // (op 161) 

buffer_load_ubyte v22, v21, s[72:75], 0 offen offset: 0 // (op 161) Load value



buffer_load_dwordx4 v[24:27], v5, s[60:63], 0 offen offset:64 // (op 441) Load value
s_waitcnt vmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:13], v[16:19], a[0:3], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 162) 








v_add_u32 v7, 128, v7 // (op 707) 
buffer_load_dwordx4 v[10:13], v14, s[64:67], 0 offen offset:64 // (op 443) Load value




v_add_u32 v4, 128, v4 // (op 797) 
buffer_load_ubyte v5, v15, s[68:71], 0 offen offset:4 // (op 445) Load value




v_add_u32 v8, 8, v8 // (op 885) 
buffer_load_ubyte v14, v21, s[72:75], 0 offen offset:4 // (op 447) Load value

v_add_u32 v9, 8, v9 // (op 973) 
s_waitcnt vmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[10:13], a[0:3], v5, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 448) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 151) 
s_mov_b32 s15, s18 // (op 151) convert

s_and_b32 s6, s15, 4294967294 // (op 151) 

s_cmp_lt_u32 s1, s6 // (op 151) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_151 // (op 151) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_151 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_151:




v_lshrrev_b32 v4, 7, v1 // (op 645) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 645) 
v_and_b32 v5, 15, v1 // (op 645) 
s_and_b32 s15, s18, 4294967294 // (op 645) 
v_lshl_add_u32 v7, v4, 4, v5 // (op 645) 


s_add_u32 s6, s0, s15 // (op 645) 

v_and_b32 v4, 63, v1 // (op 645) 
v_lshrrev_b32 v5, 4, v4 // (op 645) 

s_mul_i32 s15, s16, s50 // (op 645) 
s_add_u32 s15, s15, s6 // (op 645) 

v_lshlrev_b32 v4, 5, v5 // (op 645) 

s_add_u32 s6, s15, s9 // (op 645) 

s_mul_hi_u32 s15, s6, s13 // (op 645) 
s_sub_u32 s7, s6, s15 // (op 645) 
s_lshr_b32 s8, s7, 1 // (op 645) 

s_add_u32 s7, s8, s15 // (op 645) 


s_lshr_b32 s8, s7, s11 // (op 645) 

s_and_b32 s7, s11, 2147483648 // (op 645) 
s_cmp_eq_i32 s7, 0 // (op 645) 
s_mov_b32 s15, scc // (op 645) 

s_cmp_eq_i32 s15, 0 // (op 645) 
s_cselect_b32 s7, s6, s8 // (op 645) 


s_mul_hi_u32 s8, s7, s12 // (op 645) 
s_sub_u32 s15, s7, s8 // (op 645) 
s_lshr_b32 s42, s15, 1 // (op 645) 

s_add_u32 s15, s42, s8 // (op 645) 


s_lshr_b32 s8, s15, s10 // (op 645) 

s_and_b32 s15, s10, 2147483648 // (op 645) 
s_cmp_eq_i32 s15, 0 // (op 645) 
s_mov_b32 s42, scc // (op 645) 

s_cmp_eq_i32 s42, 0 // (op 645) 
s_cselect_b32 s15, s7, s8 // (op 645) 


s_mul_i32 s8, s7, s18 // (op 645) 


v_lshl_add_u32 v5, s15, 5, v7 // (op 645) 


s_sub_u32 s7, s6, s8 // (op 645) 


v_lshl_add_u32 v7, s7, 7, v4 // (op 645) 


v_mov_b32 v9, 0 // (op 645) convert
v_mov_b32 v8, v7 // (op 645) convert

v_mul_lo_u32 v11, v5, s25 // (op 645) most significant: low of low * high
v_mul_hi_u32 v4, v5, s24 // (op 645) most significant: high of low * low
v_mul_lo_u32 v10, v5, s24 // (op 645) least significant: low of low * low
v_add_u32 v11, v11, v4 // (op 645) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v8 // (op 645) least significant half
v_addc_co_u32 v11, s[18:19], v11, v9, s[18:19] // (op 645) most significant half



v_lshlrev_b64 v[4:5], 2, v[10:11] // (op 645) 

v_lshrrev_b64 v[8:9], 3, v[4:5] // (op 645) 

v_mov_b32 v7, v8 // (op 645) convert

s_mul_i32 s19, s24, s29 // (op 647) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 647) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 647) least significant
s_mul_hi_u32 s6, s24, s28 // (op 647) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 647) most significant: sum
s_add_u32 s19, s19, s6 // (op 647) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 647) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 647) 

s_mov_b32 s60, s26 // (op 647) reg expression
s_mov_b32 s61, s27 // (op 647) reg expression
s_mov_b32 s62, s18 // (op 647) reg expression
s_mov_b32 s63, 131072 // (op 647) call()

v_lshrrev_b32 v4, 6, v1 // (op 739) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 739) 
v_and_b32 v5, 1, v4 // (op 739) 

s_and_b32 s15, s18, 4294967294 // (op 739) 
v_and_b32 v4, 15, v1 // (op 739) 
s_add_u32 s6, s0, s15 // (op 739) 

v_lshl_add_u32 v8, v5, 4, v4 // (op 739) 


v_and_b32 v4, 63, v1 // (op 739) 
s_mul_i32 s15, s16, s50 // (op 739) 
s_add_u32 s15, s15, s6 // (op 739) 

v_lshrrev_b32 v5, 4, v4 // (op 739) 

s_add_u32 s6, s15, s9 // (op 739) 

v_lshlrev_b32 v4, 5, v5 // (op 739) 

s_mul_hi_u32 s15, s6, s13 // (op 739) 
s_sub_u32 s7, s6, s15 // (op 739) 
s_lshr_b32 s8, s7, 1 // (op 739) 

s_add_u32 s7, s8, s15 // (op 739) 


s_lshr_b32 s8, s7, s11 // (op 739) 

s_and_b32 s7, s11, 2147483648 // (op 739) 
s_cmp_eq_i32 s7, 0 // (op 739) 
s_mov_b32 s15, scc // (op 739) 

s_cmp_eq_i32 s15, 0 // (op 739) 
s_cselect_b32 s7, s6, s8 // (op 739) 


s_mul_hi_u32 s8, s7, s12 // (op 739) 
s_sub_u32 s15, s7, s8 // (op 739) 
s_lshr_b32 s42, s15, 1 // (op 739) 

s_add_u32 s15, s42, s8 // (op 739) 


s_lshr_b32 s8, s15, s10 // (op 739) 

s_and_b32 s15, s10, 2147483648 // (op 739) 
s_cmp_eq_i32 s15, 0 // (op 739) 
s_mov_b32 s42, scc // (op 739) 

s_cmp_eq_i32 s42, 0 // (op 739) 
s_cselect_b32 s15, s7, s8 // (op 739) 


s_mul_i32 s8, s15, s14 // (op 739) 

s_sub_u32 s15, s7, s8 // (op 739) 

s_mul_i32 s8, s7, s18 // (op 739) 


v_lshl_add_u32 v5, s15, 5, v8 // (op 739) 


s_sub_u32 s7, s6, s8 // (op 739) 


v_mul_lo_u32 v9, v5, s31 // (op 739) most significant: low of low * high
v_mul_hi_u32 v10, v5, s30 // (op 739) most significant: high of low * low
v_mul_lo_u32 v8, v5, s30 // (op 739) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 739) most significant: sum


v_lshl_add_u32 v5, s7, 7, v4 // (op 739) 


v_mov_b32 v11, 0 // (op 739) convert
v_mov_b32 v10, v5 // (op 739) convert

v_add_co_u32 v4, s[18:19], v10, v8 // (op 739) least significant half
v_addc_co_u32 v5, s[18:19], v11, v9, s[18:19] // (op 739) most significant half

v_lshlrev_b64 v[4:5], 2, v[4:5] // (op 739) 


v_lshrrev_b64 v[8:9], 3, v[4:5] // (op 739) 

v_mov_b32 v4, v8 // (op 739) convert

s_mul_i32 s19, s30, s23 // (op 741) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 741) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 741) least significant
s_mul_hi_u32 s6, s30, s22 // (op 741) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 741) most significant: sum
s_add_u32 s19, s19, s6 // (op 741) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 741) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 741) 

s_mov_b32 s64, s32 // (op 741) reg expression
s_mov_b32 s65, s33 // (op 741) reg expression
s_mov_b32 s66, s18 // (op 741) reg expression
s_mov_b32 s67, 131072 // (op 741) call()

v_lshrrev_b32 v5, 7, v1 // (op 827) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 827) 
v_and_b32 v8, 15, v1 // (op 827) 
s_and_b32 s15, s18, 4294967294 // (op 827) 
v_lshl_add_u32 v9, v5, 4, v8 // (op 827) 


s_add_u32 s6, s0, s15 // (op 827) 

v_and_b32 v5, 63, v1 // (op 827) 
v_lshrrev_b32 v8, 4, v5 // (op 827) 

s_mul_i32 s15, s16, s50 // (op 827) 
s_add_u32 s15, s15, s6 // (op 827) 

s_add_u32 s6, s15, s9 // (op 827) 

s_mul_hi_u32 s15, s6, s13 // (op 827) 
s_sub_u32 s7, s6, s15 // (op 827) 
s_lshr_b32 s8, s7, 1 // (op 827) 

s_add_u32 s7, s8, s15 // (op 827) 


s_lshr_b32 s8, s7, s11 // (op 827) 

s_and_b32 s7, s11, 2147483648 // (op 827) 
s_cmp_eq_i32 s7, 0 // (op 827) 
s_mov_b32 s15, scc // (op 827) 

s_cmp_eq_i32 s15, 0 // (op 827) 
s_cselect_b32 s7, s6, s8 // (op 827) 


s_mul_hi_u32 s8, s7, s12 // (op 827) 
s_sub_u32 s15, s7, s8 // (op 827) 
s_lshr_b32 s42, s15, 1 // (op 827) 

s_add_u32 s15, s42, s8 // (op 827) 


s_lshr_b32 s8, s15, s10 // (op 827) 

s_and_b32 s15, s10, 2147483648 // (op 827) 
s_cmp_eq_i32 s15, 0 // (op 827) 
s_mov_b32 s42, scc // (op 827) 

s_cmp_eq_i32 s42, 0 // (op 827) 
s_cselect_b32 s15, s7, s8 // (op 827) 


s_mul_i32 s8, s7, s18 // (op 827) 


v_lshl_add_u32 v5, s15, 5, v9 // (op 827) 


s_sub_u32 s7, s6, s8 // (op 827) 


v_lshl_add_u32 v9, s7, 2, v8 // (op 827) 


v_mul_lo_u32 v8, v5, s34 // (op 827) 
v_add_u32 v8, v8, v9 // (op 827) 


s_mul_i32 s19, s34, s29 // (op 829) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 829) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 829) least significant
s_mul_hi_u32 s6, s34, s28 // (op 829) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 829) most significant: sum
s_add_u32 s19, s19, s6 // (op 829) most significant: sum


s_mov_b32 s68, s36 // (op 829) reg expression
s_mov_b32 s69, s37 // (op 829) reg expression
s_mov_b32 s70, s18 // (op 829) reg expression
s_mov_b32 s71, 131072 // (op 829) call()

v_lshrrev_b32 v5, 6, v1 // (op 915) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 915) 
v_and_b32 v9, 1, v5 // (op 915) 

s_and_b32 s15, s18, 4294967294 // (op 915) 
v_and_b32 v5, 15, v1 // (op 915) 
s_add_u32 s6, s0, s15 // (op 915) 

v_lshl_add_u32 v10, v9, 4, v5 // (op 915) 


v_and_b32 v5, 63, v1 // (op 915) 
s_mul_i32 s15, s16, s50 // (op 915) 
s_add_u32 s15, s15, s6 // (op 915) 

v_lshrrev_b32 v9, 4, v5 // (op 915) 

s_add_u32 s6, s15, s9 // (op 915) 

s_mul_hi_u32 s15, s6, s13 // (op 915) 
s_sub_u32 s7, s6, s15 // (op 915) 
s_lshr_b32 s8, s7, 1 // (op 915) 

s_add_u32 s7, s8, s15 // (op 915) 


s_lshr_b32 s8, s7, s11 // (op 915) 

s_and_b32 s7, s11, 2147483648 // (op 915) 
s_cmp_eq_i32 s7, 0 // (op 915) 
s_mov_b32 s15, scc // (op 915) 

s_cmp_eq_i32 s15, 0 // (op 915) 
s_cselect_b32 s7, s6, s8 // (op 915) 


s_mul_hi_u32 s8, s7, s12 // (op 915) 
s_sub_u32 s15, s7, s8 // (op 915) 
s_lshr_b32 s42, s15, 1 // (op 915) 

s_add_u32 s15, s42, s8 // (op 915) 


s_lshr_b32 s8, s15, s10 // (op 915) 

s_and_b32 s15, s10, 2147483648 // (op 915) 
s_cmp_eq_i32 s15, 0 // (op 915) 
s_mov_b32 s42, scc // (op 915) 

s_cmp_eq_i32 s42, 0 // (op 915) 
s_cselect_b32 s15, s7, s8 // (op 915) 


s_mul_i32 s8, s15, s14 // (op 915) 

s_sub_u32 s15, s7, s8 // (op 915) 

v_lshl_add_u32 v5, s15, 5, v10 // (op 915) 


s_mul_i32 s8, s7, s18 // (op 915) 


v_mul_lo_u32 v10, v5, s38 // (op 915) 

s_sub_u32 s7, s6, s8 // (op 915) 


v_lshl_add_u32 v5, s7, 2, v9 // (op 915) 


v_add_u32 v9, v5, v10 // (op 915) 


s_mul_i32 s19, s38, s23 // (op 917) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 917) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 917) least significant
s_mul_hi_u32 s6, s38, s22 // (op 917) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 917) most significant: sum
s_add_u32 s19, s19, s6 // (op 917) most significant: sum


s_mov_b32 s72, s40 // (op 917) reg expression
s_mov_b32 s73, s41 // (op 917) reg expression
s_mov_b32 s74, s18 // (op 917) reg expression
s_mov_b32 s75, 131072 // (op 917) call()

s_ashr_i64 s[18:19], s[2:3], 7 // (op 401) 
s_and_b32 s15, s18, 4294967294 // (op 401) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 400) 
s_mov_b32 s6, s18 // (op 400) convert

s_cmp_lt_u32 s15, s6 // (op 400) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_400 // (op 400) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_400 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_400:


s_add_u32 s15, s15, 1 // (op 402) 
v_mov_b32 v5, v7 // (op 406) 

buffer_load_dwordx4 v[10:13], v5, s[60:63], 0 offen // (op 406) Load value

v_add_u32 v7, 64, v7 // (op 635) 


v_mov_b32 v5, v4 // (op 408) 

buffer_load_dwordx4 v[14:17], v5, s[64:67], 0 offen // (op 408) Load value

v_add_u32 v4, 64, v4 // (op 731) 


v_mov_b32 v5, v8 // (op 410) 

buffer_load_ubyte v18, v5, s[68:71], 0 offen offset: 0 // (op 410) Load value

v_add_u32 v8, 4, v8 // (op 819) 


v_mov_b32 v5, v9 // (op 412) 

buffer_load_ubyte v19, v5, s[72:75], 0 offen offset: 0 // (op 412) Load value

v_add_u32 v9, 4, v9 // (op 907) 
s_waitcnt vmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:13], v[14:17], a[0:3], v18, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 413) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 400) 
s_mov_b32 s6, s18 // (op 400) convert

s_cmp_lt_u32 s15, s6 // (op 400) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_400 // (op 400) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_400 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_400:





v_cmp_eq_f32 vcc, 0.00000, v0 // (op 181) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_181 // (op 181) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_181
s_nop 1
v_accvgpr_read v8, a0 // (op 189) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 189) 
v_accvgpr_read v10, a2 // (op 189) 
v_accvgpr_read v11, a3 // (op 189) 
v_mul_f32 v12, v3, v8 // (op 189) 
v_mul_f32 v13, v3, v9 // (op 189) 
v_mul_f32 v14, v3, v10 // (op 189) 
v_mul_f32 v15, v3, v11 // (op 189) 

v_lshrrev_b32 v8, 16, v12 // (op 191) 
v_lshrrev_b32 v9, 16, v13 // (op 191) 
v_lshrrev_b32 v10, 16, v14 // (op 191) 
v_lshrrev_b32 v11, 16, v15 // (op 191) 

v_lshrrev_b32 v7, 7, v1 // (op 1069) 
s_mul_i32 s15, s16, s50 // (op 1069) 
s_add_u32 s15, s15, s0 // (op 1069) 
v_and_b32 v4, 63, v1 // (op 1069) 
s_add_u32 s6, s15, s9 // (op 1069) 

v_lshrrev_b32 v5, 4, v4 // (op 1069) 
s_mul_hi_u32 s15, s6, s13 // (op 1069) 
v_lshlrev_b32 v12, 2, v5 // (op 1069) 

s_sub_u32 s7, s6, s15 // (op 1069) 
v_lshl_add_u32 v5, v7, 4, v12 // (op 1069) 


s_lshr_b32 s8, s7, 1 // (op 1069) 

v_lshrrev_b32 v7, 2, v4 // (op 1069) 

s_add_u32 s7, s8, s15 // (op 1069) 


v_and_b32 v4, 3, v7 // (op 1069) 

s_lshr_b32 s8, s7, s11 // (op 1069) 

v_and_b32 v7, 3, v1 // (op 1069) 
s_and_b32 s7, s11, 2147483648 // (op 1069) 
v_lshl_add_u32 v12, v4, 2, v7 // (op 1069) 


s_cmp_eq_i32 s7, 0 // (op 1069) 
s_mov_b32 s15, scc // (op 1069) 

v_lshrrev_b32 v4, 6, v1 // (op 1069) 
s_cmp_eq_i32 s15, 0 // (op 1069) 
s_cselect_b32 s7, s6, s8 // (op 1069) 



v_and_b32 v7, 1, v4 // (op 1069) 

s_mul_hi_u32 s6, s7, s12 // (op 1069) 
v_lshl_add_u32 v4, v7, 4, v12 // (op 1069) 


s_sub_u32 s8, s7, s6 // (op 1069) 
s_lshr_b32 s15, s8, 1 // (op 1069) 

s_add_u32 s8, s15, s6 // (op 1069) 


s_lshr_b32 s6, s8, s10 // (op 1069) 

s_and_b32 s8, s10, 2147483648 // (op 1069) 
s_cmp_eq_i32 s8, 0 // (op 1069) 
s_mov_b32 s15, scc // (op 1069) 

s_cmp_eq_i32 s15, 0 // (op 1069) 
s_cselect_b32 s8, s7, s6 // (op 1069) 


s_mul_i32 s6, s8, s14 // (op 1069) 
v_lshl_add_u32 v7, s8, 5, v5 // (op 1069) 


s_sub_u32 s8, s7, s6 // (op 1069) 


v_lshl_add_u32 v5, s8, 5, v4 // (op 1069) 


v_mul_lo_u32 v4, v5, s46 // (op 1069) 

v_add_lshl_u32 v5, v7, v4, 1 // (op 1069) 


s_mul_i32 s19, s46, s23 // (op 1071) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1071) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1071) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1071) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1071) most significant: sum
s_add_u32 s19, s19, s6 // (op 1071) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1071) 

s_mov_b32 s76, s48 // (op 1071) reg expression
s_mov_b32 s77, s49 // (op 1071) reg expression
s_mov_b32 s78, s6 // (op 1071) reg expression
s_mov_b32 s79, 131072 // (op 1071) call()


v_pack_b32_f16 v12, v8, v9 // (op 192) 
v_pack_b32_f16 v13, v10, v11 // (op 192) 


buffer_store_dwordx2 v[12:13], v5, s[76:79], 0 offen // (op 192) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_181 // (op 181) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_181
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_181:



buffer_load_dwordx2 v[4:5], v6, s[56:59], 0 offen // (op 206) Load value
v_accvgpr_read v8, a0 // (op 202) 
v_accvgpr_read v9, a1 // (op 202) 
v_accvgpr_read v10, a2 // (op 202) 
v_accvgpr_read v11, a3 // (op 202) 
v_mul_f32 v12, v3, v8 // (op 202) 
v_mul_f32 v13, v3, v9 // (op 202) 
v_mul_f32 v14, v3, v10 // (op 202) 
v_mul_f32 v15, v3, v11 // (op 202) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 207) 
v_lshrrev_b32 v11, 16, v4 // (op 207) 
v_lshlrev_b32 v10, 16, v10 // (op 207) 
v_lshlrev_b32 v11, 16, v11 // (op 207) 
v_mul_f32 v16, v0, v10 // (op 207) 
v_mul_f32 v17, v0, v11 // (op 207) 
v_and_b32 v10, 65535, v5 // (op 207) 
v_lshrrev_b32 v11, 16, v5 // (op 207) 
v_lshlrev_b32 v10, 16, v10 // (op 207) 
v_lshlrev_b32 v11, 16, v11 // (op 207) 
v_mul_f32 v18, v0, v10 // (op 207) 
v_mul_f32 v19, v0, v11 // (op 207) 


v_add_f32 v8, v16, v12 // (op 203) 
v_add_f32 v9, v17, v13 // (op 203) 
v_add_f32 v10, v18, v14 // (op 203) 
v_add_f32 v11, v19, v15 // (op 203) 


v_lshrrev_b32 v12, 16, v8 // (op 204) 
v_lshrrev_b32 v13, 16, v9 // (op 204) 
v_lshrrev_b32 v14, 16, v10 // (op 204) 
v_lshrrev_b32 v15, 16, v11 // (op 204) 

v_lshrrev_b32 v4, 7, v1 // (op 1086) 
s_mul_i32 s15, s16, s50 // (op 1086) 
s_add_u32 s15, s15, s0 // (op 1086) 
v_and_b32 v5, 63, v1 // (op 1086) 
s_add_u32 s6, s15, s9 // (op 1086) 

v_lshrrev_b32 v7, 4, v5 // (op 1086) 
s_mul_hi_u32 s15, s6, s13 // (op 1086) 
v_lshlrev_b32 v8, 2, v7 // (op 1086) 

s_sub_u32 s7, s6, s15 // (op 1086) 
v_lshl_add_u32 v7, v4, 4, v8 // (op 1086) 


s_lshr_b32 s8, s7, 1 // (op 1086) 

v_lshrrev_b32 v4, 2, v5 // (op 1086) 

s_add_u32 s7, s8, s15 // (op 1086) 


v_and_b32 v5, 3, v4 // (op 1086) 

s_lshr_b32 s8, s7, s11 // (op 1086) 

v_and_b32 v4, 3, v1 // (op 1086) 
s_and_b32 s7, s11, 2147483648 // (op 1086) 
v_lshl_add_u32 v8, v5, 2, v4 // (op 1086) 


s_cmp_eq_i32 s7, 0 // (op 1086) 
s_mov_b32 s15, scc // (op 1086) 

v_lshrrev_b32 v4, 6, v1 // (op 1086) 
s_cmp_eq_i32 s15, 0 // (op 1086) 
s_cselect_b32 s7, s6, s8 // (op 1086) 



v_and_b32 v5, 1, v4 // (op 1086) 

s_mul_hi_u32 s6, s7, s12 // (op 1086) 
v_lshl_add_u32 v4, v5, 4, v8 // (op 1086) 


s_sub_u32 s8, s7, s6 // (op 1086) 
s_lshr_b32 s15, s8, 1 // (op 1086) 

s_add_u32 s8, s15, s6 // (op 1086) 


s_lshr_b32 s6, s8, s10 // (op 1086) 

s_and_b32 s8, s10, 2147483648 // (op 1086) 
s_cmp_eq_i32 s8, 0 // (op 1086) 
s_mov_b32 s15, scc // (op 1086) 

s_cmp_eq_i32 s15, 0 // (op 1086) 
s_cselect_b32 s8, s7, s6 // (op 1086) 


s_mul_i32 s6, s8, s14 // (op 1086) 
v_lshl_add_u32 v5, s8, 5, v7 // (op 1086) 


s_sub_u32 s8, s7, s6 // (op 1086) 


v_lshl_add_u32 v7, s8, 5, v4 // (op 1086) 


v_mul_lo_u32 v4, v7, s46 // (op 1086) 

v_add_lshl_u32 v7, v5, v4, 1 // (op 1086) 


s_mul_i32 s19, s46, s23 // (op 1088) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1088) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1088) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1088) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1088) most significant: sum
s_add_u32 s19, s19, s6 // (op 1088) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1088) 

s_mov_b32 s76, s48 // (op 1088) reg expression
s_mov_b32 s77, s49 // (op 1088) reg expression
s_mov_b32 s78, s6 // (op 1088) reg expression
s_mov_b32 s79, 131072 // (op 1088) call()


v_pack_b32_f16 v4, v12, v13 // (op 205) 
v_pack_b32_f16 v5, v14, v15 // (op 205) 


buffer_store_dwordx2 v[4:5], v7, s[76:79], 0 offen // (op 205) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_181:

s_add_u32 s0, s0, s1 // (op 394) 
s_mul_i32 s15, s50, s16 // (op 392) 
s_cmp_gt_u32 s51, s15 // (op 392) 
s_mov_b32 s6, scc // (op 392) 
s_sub_u32 s7, s51, s15 // (op 392) 

s_cmp_eq_i32 s6, 0 // (op 392) 
s_cselect_b32 s8, 0, s7 // (op 392) 


s_cmp_lt_u32 s50, s8 // (op 392) 
s_mov_b32 s15, scc // (op 392) 
s_cmp_eq_i32 s15, 0 // (op 392) 
s_cselect_b32 s6, s8, s50 // (op 392) 


s_cmp_lt_u32 s0, s6 // (op 392) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_392 // (op 392) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_392 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_392:


































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2, .LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_end-RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 0 // lds bytes
  .amdhsa_accum_offset 28
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
    args: [ { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideC0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: uint32, semantic: RR_SKNumSKTilesPerWG },
            { type: uint32, semantic: RR_SKNumSKTiles },
            { type: int32, semantic: RR_SKMagicShiftsNumTilesN },
            { type: int32, semantic: RR_SKMagicShiftsIters },
            { type: uint32, semantic: RR_SKMagicMultipleNumTilesN },
            { type: uint32, semantic: RR_SKMagicMultipleIters },
            { type: uint32, semantic: RR_SKNumTilesN },
            { type: uint32, semantic: RR_SKNumWGs },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: address, semantic: RR_SKAddressPartials },
            { type: address, semantic: RR_SKAddressFlags },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: uint32, semantic: RR_SKNumDPTilesPerWG },
            { type: uint32, semantic: RR_SKNumDPTiles } ]
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 8
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 2
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 82
    .vgpr_count: 28
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_22
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_10_stride_1_18
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
      - .name: numSKTilesPerWG
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: numSKTiles
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicShifts_31
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicShifts_29
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: numTiles1
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: numWGs
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_0_stride_0_8
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_9
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_stride_1_11
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_14
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_16
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_None
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_ZeroedBeforeAndAfter
        .size: 8
        .offset: 152
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_stride_1_24
        .size: 8
        .offset: 160
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 168
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: numDPTilesPerWG
        .size: 4
        .offset: 176
        .value_kind: by_value
      - .name: numDPTiles
        .size: 4
        .offset: 180
        .value_kind: by_value
...

.end_amdgpu_metadata

