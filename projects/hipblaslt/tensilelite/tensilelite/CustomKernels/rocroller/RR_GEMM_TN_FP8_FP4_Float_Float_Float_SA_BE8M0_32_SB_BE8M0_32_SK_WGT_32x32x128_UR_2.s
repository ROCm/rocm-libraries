// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2,@function
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_exec_begin:


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


s_mov_b32 s0, 0 // (op 330) call()
s_mov_b32 s1, 0 // (op 331) call()
s_mov_b32 s6, 0 // (op 332) call()
s_mov_b32 s7, 0 // (op 333) call()
s_mov_b32 s18, 0 // (op 334) call()
s_mov_b32 s19, 0 // (op 335) call()
s_mov_b32 s0, 0 // (op 230) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 229) 
s_cmp_gt_u32 s9, s52 // (op 229) 
s_mov_b32 s53, scc // (op 229) 
s_sub_u32 s54, s9, s52 // (op 229) 

s_cmp_eq_i32 s53, 0 // (op 229) 
s_cselect_b32 s52, 0, s54 // (op 229) 


s_cmp_lt_u32 s8, s52 // (op 229) 
s_mov_b32 s53, scc // (op 229) 
s_cmp_eq_i32 s53, 0 // (op 229) 
s_cselect_b32 s54, s52, s8 // (op 229) 


s_cmp_lt_u32 s0, s54 // (op 229) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_229 // (op 229) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_229 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_229:
v_lshrrev_b32 v4, 7, v1 // (op 993) 
s_mul_i32 s52, s16, s8 // (op 993) 
s_add_u32 s52, s52, s0 // (op 993) 
v_and_b32 v5, 63, v1 // (op 993) 
s_mul_hi_u32 s53, s52, s13 // (op 993) 
v_lshrrev_b32 v6, 4, v5 // (op 993) 
s_sub_u32 s54, s52, s53 // (op 993) 
v_lshlrev_b32 v7, 2, v6 // (op 993) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 993) 


s_lshr_b32 s55, s54, 1 // (op 993) 

s_add_u32 s54, s55, s53 // (op 993) 


v_lshrrev_b32 v4, 2, v5 // (op 993) 

s_lshr_b32 s53, s54, s11 // (op 993) 

v_and_b32 v5, 3, v4 // (op 993) 

s_and_b32 s54, s11, 2147483648 // (op 993) 
v_and_b32 v4, 3, v1 // (op 993) 
s_cmp_eq_i32 s54, 0 // (op 993) 
s_mov_b32 s55, scc // (op 993) 

v_lshl_add_u32 v7, v5, 2, v4 // (op 993) 


s_cmp_eq_i32 s55, 0 // (op 993) 
s_cselect_b32 s54, s52, s53 // (op 993) 



v_lshrrev_b32 v4, 6, v1 // (op 993) 
s_mul_hi_u32 s52, s54, s12 // (op 993) 
v_and_b32 v5, 1, v4 // (op 993) 

s_sub_u32 s53, s54, s52 // (op 993) 
v_lshl_add_u32 v4, v5, 4, v7 // (op 993) 


s_lshr_b32 s55, s53, 1 // (op 993) 

s_add_u32 s53, s55, s52 // (op 993) 


s_lshr_b32 s52, s53, s10 // (op 993) 

s_and_b32 s53, s10, 2147483648 // (op 993) 
s_cmp_eq_i32 s53, 0 // (op 993) 
s_mov_b32 s55, scc // (op 993) 

s_cmp_eq_i32 s55, 0 // (op 993) 
s_cselect_b32 s53, s54, s52 // (op 993) 


s_mul_i32 s52, s53, s14 // (op 993) 
v_lshl_add_u32 v5, s53, 5, v6 // (op 993) 


s_sub_u32 s53, s54, s52 // (op 993) 


v_lshl_add_u32 v6, s53, 5, v4 // (op 993) 


v_mul_lo_u32 v4, v6, s4 // (op 993) 

v_add_lshl_u32 v6, v5, v4, 2 // (op 993) 


s_mul_i32 s53, s4, s23 // (op 995) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 995) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 995) least significant
s_mul_hi_u32 s55, s4, s22 // (op 995) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 995) most significant: sum
s_add_u32 s53, s53, s55 // (op 995) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 995) 

s_mov_b32 s56, s20 // (op 995) reg expression
s_mov_b32 s57, s21 // (op 995) reg expression
s_mov_b32 s58, s54 // (op 995) reg expression
s_mov_b32 s59, 131072 // (op 995) call()

s_mul_i32 s6, s8, s16 // (op 234) 
s_add_u32 s6, s6, s0 // (op 234) 
s_mul_hi_u32 s52, s6, s13 // (op 235) 
s_sub_u32 s53, s6, s52 // (op 235) 
s_lshr_b32 s54, s53, 1 // (op 235) 

s_add_u32 s53, s54, s52 // (op 235) 


s_lshr_b32 s52, s53, s11 // (op 235) 

s_and_b32 s53, s11, 2147483648 // (op 235) 
s_cmp_eq_i32 s53, 0 // (op 235) 
s_mov_b32 s54, scc // (op 235) 

s_cmp_eq_i32 s54, 0 // (op 235) 
s_cselect_b32 s53, s6, s52 // (op 235) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 235) 
s_mul_i32 s52, s53, s54 // (op 235) 


s_sub_u32 s18, s6, s52 // (op 235) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 236) 
s_cmp_gt_u32 s52, s18 // (op 236) 
s_mov_b32 s54, scc // (op 236) 
s_sub_u32 s55, s52, s18 // (op 236) 

s_cmp_eq_i32 s54, 0 // (op 236) 
s_cselect_b32 s52, 0, s55 // (op 236) 


s_cmp_gt_u32 s8, s0 // (op 236) 
s_mov_b32 s53, scc // (op 236) 
s_sub_u32 s54, s8, s0 // (op 236) 
s_cmp_eq_i32 s53, 0 // (op 236) 
s_cselect_b32 s55, 0, s54 // (op 236) 


s_cmp_lt_u32 s52, s55 // (op 236) 
s_mov_b32 s53, scc // (op 236) 
s_cmp_eq_i32 s53, 0 // (op 236) 
s_cselect_b32 s54, s55, s52 // (op 236) 



s_cmp_gt_u32 s9, s6 // (op 236) 
s_mov_b32 s55, scc // (op 236) 
s_sub_u32 s52, s9, s6 // (op 236) 
s_cmp_eq_i32 s55, 0 // (op 236) 
s_cselect_b32 s53, 0, s52 // (op 236) 


s_cmp_lt_u32 s54, s53 // (op 236) 
s_mov_b32 s52, scc // (op 236) 
s_cmp_eq_i32 s52, 0 // (op 236) 
s_cselect_b32 s7, s53, s54 // (op 236) 



s_add_u32 s52, s18, s7 // (op 237) 
s_sub_u32 s19, s52, 1 // (op 237) 

v_lshrrev_b32 v4, 7, v1 // (op 685) 
s_mul_i32 s52, s16, s8 // (op 685) 
s_add_u32 s52, s52, s0 // (op 685) 
v_and_b32 v5, 15, v1 // (op 685) 
s_mul_hi_u32 s53, s52, s13 // (op 685) 
v_lshl_add_u32 v7, v4, 4, v5 // (op 685) 


s_sub_u32 s54, s52, s53 // (op 685) 
v_and_b32 v4, 48, v1 // (op 685) 
s_lshr_b32 s55, s54, 1 // (op 685) 

s_add_u32 s54, s55, s53 // (op 685) 


s_lshr_b32 s53, s54, s11 // (op 685) 

s_and_b32 s54, s11, 2147483648 // (op 685) 
s_cmp_eq_i32 s54, 0 // (op 685) 
s_mov_b32 s55, scc // (op 685) 

s_cmp_eq_i32 s55, 0 // (op 685) 
s_cselect_b32 s54, s52, s53 // (op 685) 


s_mul_hi_u32 s53, s54, s12 // (op 685) 
s_sub_u32 s55, s54, s53 // (op 685) 
s_lshr_b32 s60, s55, 1 // (op 685) 

s_add_u32 s55, s60, s53 // (op 685) 


s_lshr_b32 s53, s55, s10 // (op 685) 

s_and_b32 s55, s10, 2147483648 // (op 685) 
s_cmp_eq_i32 s55, 0 // (op 685) 
s_mov_b32 s60, scc // (op 685) 

s_cmp_eq_i32 s60, 0 // (op 685) 
s_cselect_b32 s55, s54, s53 // (op 685) 


v_lshl_add_u32 v5, s55, 5, v7 // (op 685) 


s_ashr_i64 s[60:61], s[2:3], 7 // (op 685) 
s_mul_i32 s53, s54, s60 // (op 685) 


s_sub_u32 s54, s52, s53 // (op 685) 


v_lshl_add_u32 v7, s54, 7, v4 // (op 685) 


v_mul_lo_u32 v4, v5, s24 // (op 685) 
v_add_u32 v4, v4, v7 // (op 685) 


s_mul_i32 s53, s24, s29 // (op 687) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 687) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 687) least significant
s_mul_hi_u32 s55, s24, s28 // (op 687) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 687) most significant: sum
s_add_u32 s53, s53, s55 // (op 687) most significant: sum


s_mov_b32 s60, s26 // (op 687) reg expression
s_mov_b32 s61, s27 // (op 687) reg expression
s_mov_b32 s62, s52 // (op 687) reg expression
s_mov_b32 s63, 131072 // (op 687) call()

v_lshrrev_b32 v5, 6, v1 // (op 775) 
s_mul_i32 s52, s16, s8 // (op 775) 
s_add_u32 s52, s52, s0 // (op 775) 
v_and_b32 v7, 1, v5 // (op 775) 

s_mul_hi_u32 s53, s52, s13 // (op 775) 
v_and_b32 v5, 15, v1 // (op 775) 
s_sub_u32 s54, s52, s53 // (op 775) 
v_lshl_add_u32 v8, v7, 4, v5 // (op 775) 


s_lshr_b32 s55, s54, 1 // (op 775) 

v_and_b32 v5, 63, v1 // (op 775) 
s_add_u32 s54, s55, s53 // (op 775) 


v_lshrrev_b32 v7, 4, v5 // (op 775) 

s_lshr_b32 s53, s54, s11 // (op 775) 

v_lshlrev_b32 v5, 5, v7 // (op 775) 

s_and_b32 s54, s11, 2147483648 // (op 775) 
s_cmp_eq_i32 s54, 0 // (op 775) 
s_mov_b32 s55, scc // (op 775) 

s_cmp_eq_i32 s55, 0 // (op 775) 
s_cselect_b32 s54, s52, s53 // (op 775) 


s_mul_hi_u32 s53, s54, s12 // (op 775) 
s_sub_u32 s55, s54, s53 // (op 775) 
s_lshr_b32 s64, s55, 1 // (op 775) 

s_add_u32 s55, s64, s53 // (op 775) 


s_lshr_b32 s53, s55, s10 // (op 775) 

s_and_b32 s55, s10, 2147483648 // (op 775) 
s_cmp_eq_i32 s55, 0 // (op 775) 
s_mov_b32 s64, scc // (op 775) 

s_cmp_eq_i32 s64, 0 // (op 775) 
s_cselect_b32 s55, s54, s53 // (op 775) 


s_mul_i32 s53, s55, s14 // (op 775) 

s_sub_u32 s55, s54, s53 // (op 775) 

v_lshl_add_u32 v7, s55, 5, v8 // (op 775) 


v_mul_lo_u32 v9, v7, s31 // (op 775) most significant: low of low * high
v_mul_hi_u32 v10, v7, s30 // (op 775) most significant: high of low * low
v_mul_lo_u32 v8, v7, s30 // (op 775) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 775) most significant: sum


s_ashr_i64 s[64:65], s[2:3], 7 // (op 775) 
s_mul_i32 s53, s54, s64 // (op 775) 


s_sub_u32 s54, s52, s53 // (op 775) 


v_lshl_add_u32 v7, s54, 7, v5 // (op 775) 


v_mov_b32 v11, 0 // (op 775) convert
v_mov_b32 v10, v7 // (op 775) convert

v_add_co_u32 v12, s[52:53], v10, v8 // (op 775) least significant half
v_addc_co_u32 v13, s[52:53], v11, v9, s[52:53] // (op 775) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 775) 


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 775) 

v_mov_b32 v5, v10 // (op 775) convert

s_mul_i32 s53, s30, s23 // (op 777) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 777) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 777) least significant
s_mul_hi_u32 s55, s30, s22 // (op 777) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 777) most significant: sum
s_add_u32 s53, s53, s55 // (op 777) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 777) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 777) 

s_mov_b32 s64, s32 // (op 777) reg expression
s_mov_b32 s65, s33 // (op 777) reg expression
s_mov_b32 s66, s52 // (op 777) reg expression
s_mov_b32 s67, 131072 // (op 777) call()

v_lshrrev_b32 v7, 7, v1 // (op 863) 
s_mul_i32 s52, s16, s8 // (op 863) 
s_add_u32 s52, s52, s0 // (op 863) 
v_and_b32 v8, 15, v1 // (op 863) 
s_mul_hi_u32 s53, s52, s13 // (op 863) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 863) 


s_sub_u32 s54, s52, s53 // (op 863) 
v_and_b32 v7, 63, v1 // (op 863) 
s_lshr_b32 s55, s54, 1 // (op 863) 

v_lshrrev_b32 v8, 4, v7 // (op 863) 

s_add_u32 s54, s55, s53 // (op 863) 


s_lshr_b32 s53, s54, s11 // (op 863) 

s_and_b32 s54, s11, 2147483648 // (op 863) 
s_cmp_eq_i32 s54, 0 // (op 863) 
s_mov_b32 s55, scc // (op 863) 

s_cmp_eq_i32 s55, 0 // (op 863) 
s_cselect_b32 s54, s52, s53 // (op 863) 


s_mul_hi_u32 s53, s54, s12 // (op 863) 
s_sub_u32 s55, s54, s53 // (op 863) 
s_lshr_b32 s68, s55, 1 // (op 863) 

s_add_u32 s55, s68, s53 // (op 863) 


s_lshr_b32 s53, s55, s10 // (op 863) 

s_and_b32 s55, s10, 2147483648 // (op 863) 
s_cmp_eq_i32 s55, 0 // (op 863) 
s_mov_b32 s68, scc // (op 863) 

s_cmp_eq_i32 s68, 0 // (op 863) 
s_cselect_b32 s55, s54, s53 // (op 863) 


v_lshl_add_u32 v7, s55, 5, v9 // (op 863) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 863) 
s_mul_i32 s53, s54, s68 // (op 863) 


s_sub_u32 s54, s52, s53 // (op 863) 


v_lshl_add_u32 v9, s54, 2, v8 // (op 863) 


v_mul_lo_u32 v8, v7, s34 // (op 863) 
v_add_u32 v8, v8, v9 // (op 863) 


s_mul_i32 s53, s34, s29 // (op 865) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 865) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 865) least significant
s_mul_hi_u32 s55, s34, s28 // (op 865) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 865) most significant: sum
s_add_u32 s53, s53, s55 // (op 865) most significant: sum


s_mov_b32 s68, s36 // (op 865) reg expression
s_mov_b32 s69, s37 // (op 865) reg expression
s_mov_b32 s70, s52 // (op 865) reg expression
s_mov_b32 s71, 131072 // (op 865) call()

v_lshrrev_b32 v7, 6, v1 // (op 951) 
s_mul_i32 s52, s16, s8 // (op 951) 
s_add_u32 s52, s52, s0 // (op 951) 
v_and_b32 v9, 1, v7 // (op 951) 

s_mul_hi_u32 s53, s52, s13 // (op 951) 
v_and_b32 v7, 15, v1 // (op 951) 
s_sub_u32 s54, s52, s53 // (op 951) 
v_lshl_add_u32 v10, v9, 4, v7 // (op 951) 


s_lshr_b32 s55, s54, 1 // (op 951) 

v_and_b32 v7, 63, v1 // (op 951) 
s_add_u32 s54, s55, s53 // (op 951) 


v_lshrrev_b32 v9, 4, v7 // (op 951) 

s_lshr_b32 s53, s54, s11 // (op 951) 

s_and_b32 s54, s11, 2147483648 // (op 951) 
s_cmp_eq_i32 s54, 0 // (op 951) 
s_mov_b32 s55, scc // (op 951) 

s_cmp_eq_i32 s55, 0 // (op 951) 
s_cselect_b32 s54, s52, s53 // (op 951) 


s_mul_hi_u32 s53, s54, s12 // (op 951) 
s_sub_u32 s55, s54, s53 // (op 951) 
s_lshr_b32 s72, s55, 1 // (op 951) 

s_add_u32 s55, s72, s53 // (op 951) 


s_lshr_b32 s53, s55, s10 // (op 951) 

s_and_b32 s55, s10, 2147483648 // (op 951) 
s_cmp_eq_i32 s55, 0 // (op 951) 
s_mov_b32 s72, scc // (op 951) 

s_cmp_eq_i32 s72, 0 // (op 951) 
s_cselect_b32 s55, s54, s53 // (op 951) 


s_mul_i32 s53, s55, s14 // (op 951) 

s_sub_u32 s55, s54, s53 // (op 951) 

v_lshl_add_u32 v7, s55, 5, v10 // (op 951) 


v_mul_lo_u32 v10, v7, s38 // (op 951) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 951) 
s_mul_i32 s53, s54, s72 // (op 951) 


s_sub_u32 s54, s52, s53 // (op 951) 


v_lshl_add_u32 v7, s54, 2, v9 // (op 951) 


v_add_u32 v9, v7, v10 // (op 951) 


s_mul_i32 s53, s38, s23 // (op 953) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 953) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 953) least significant
s_mul_hi_u32 s55, s38, s22 // (op 953) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 953) most significant: sum
s_add_u32 s53, s53, s55 // (op 953) most significant: sum


s_mov_b32 s72, s40 // (op 953) reg expression
s_mov_b32 s73, s41 // (op 953) reg expression
s_mov_b32 s74, s52 // (op 953) reg expression
s_mov_b32 s75, 131072 // (op 953) call()

v_accvgpr_write a0, 0.00000 // (op 47) call()
v_accvgpr_write a1, 0.00000 // (op 47) call()
v_accvgpr_write a2, 0.00000 // (op 47) call()
v_accvgpr_write a3, 0.00000 // (op 47) call()
s_mov_b32 s1, 0 // (op 43) call()
s_and_b32 s52, s7, 4294967294 // (op 42) 
s_cmp_lt_u32 s1, s52 // (op 42) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_42 // (op 42) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_42 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_42:


s_add_u32 s1, s1, 2 // (op 44) 
v_mov_b32 v7, v4 // (op 50) 

buffer_load_dwordx4 v[10:13], v7, s[60:63], 0 offen // (op 50) Load value
buffer_load_dwordx4 v[14:17], v7, s[60:63], 0 offen offset:64 // (op 50) Load value


v_mov_b32 v18, v5 // (op 54) 

buffer_load_dwordx4 v[20:23], v18, s[64:67], 0 offen // (op 54) Load value


v_mov_b32 v19, v8 // (op 58) 

buffer_load_ubyte v24, v19, s[68:71], 0 offen offset: 0 // (op 58) Load value


v_mov_b32 v25, v9 // (op 62) 

buffer_load_ubyte v26, v25, s[72:75], 0 offen offset: 0 // (op 62) Load value



buffer_load_dwordx4 v[28:31], v7, s[60:63], 0 offen offset:128 // (op 518) Load value
s_waitcnt vmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:17], v[20:23], a[0:3], v24, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 64) 
buffer_load_dwordx4 v[32:35], v7, s[60:63], 0 offen offset:192 // (op 518) Load value








v_add_u32 v4, 256, v4 // (op 675) 
buffer_load_dwordx4 v[10:13], v18, s[64:67], 0 offen offset:64 // (op 520) Load value




v_add_u32 v5, 128, v5 // (op 767) 
buffer_load_ubyte v7, v19, s[68:71], 0 offen offset:4 // (op 522) Load value




v_add_u32 v8, 8, v8 // (op 855) 
buffer_load_ubyte v14, v25, s[72:75], 0 offen offset:4 // (op 524) Load value

v_add_u32 v9, 8, v9 // (op 943) 
s_waitcnt vmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[10:13], a[0:3], v7, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 525) 




s_and_b32 s52, s7, 4294967294 // (op 42) 
s_cmp_lt_u32 s1, s52 // (op 42) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_42 // (op 42) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_42 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_42:




s_and_b32 s52, s7, 4294967294 // (op 661) 
v_lshrrev_b32 v4, 7, v1 // (op 661) 
s_add_u32 s53, s0, s52 // (op 661) 

v_and_b32 v5, 15, v1 // (op 661) 
v_lshl_add_u32 v7, v4, 4, v5 // (op 661) 


s_mul_i32 s52, s16, s8 // (op 661) 
s_add_u32 s52, s52, s53 // (op 661) 

v_and_b32 v4, 48, v1 // (op 661) 
s_mul_hi_u32 s53, s52, s13 // (op 661) 
s_sub_u32 s54, s52, s53 // (op 661) 
s_lshr_b32 s55, s54, 1 // (op 661) 

s_add_u32 s54, s55, s53 // (op 661) 


s_lshr_b32 s53, s54, s11 // (op 661) 

s_and_b32 s54, s11, 2147483648 // (op 661) 
s_cmp_eq_i32 s54, 0 // (op 661) 
s_mov_b32 s55, scc // (op 661) 

s_cmp_eq_i32 s55, 0 // (op 661) 
s_cselect_b32 s54, s52, s53 // (op 661) 


s_mul_hi_u32 s53, s54, s12 // (op 661) 
s_sub_u32 s55, s54, s53 // (op 661) 
s_lshr_b32 s76, s55, 1 // (op 661) 

s_add_u32 s55, s76, s53 // (op 661) 


s_lshr_b32 s53, s55, s10 // (op 661) 

s_and_b32 s55, s10, 2147483648 // (op 661) 
s_cmp_eq_i32 s55, 0 // (op 661) 
s_mov_b32 s76, scc // (op 661) 

s_cmp_eq_i32 s76, 0 // (op 661) 
s_cselect_b32 s55, s54, s53 // (op 661) 


v_lshl_add_u32 v5, s55, 5, v7 // (op 661) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 661) 
s_mul_i32 s53, s54, s76 // (op 661) 


s_sub_u32 s54, s52, s53 // (op 661) 


v_lshl_add_u32 v7, s54, 7, v4 // (op 661) 


v_mul_lo_u32 v4, v5, s24 // (op 661) 
v_add_u32 v4, v4, v7 // (op 661) 


s_mul_i32 s53, s24, s29 // (op 663) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 663) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 663) least significant
s_mul_hi_u32 s55, s24, s28 // (op 663) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 663) most significant: sum
s_add_u32 s53, s53, s55 // (op 663) most significant: sum


s_mov_b32 s60, s26 // (op 663) reg expression
s_mov_b32 s61, s27 // (op 663) reg expression
s_mov_b32 s62, s52 // (op 663) reg expression
s_mov_b32 s63, 131072 // (op 663) call()

s_and_b32 s52, s7, 4294967294 // (op 753) 
v_lshrrev_b32 v5, 6, v1 // (op 753) 
s_add_u32 s53, s0, s52 // (op 753) 

v_and_b32 v7, 1, v5 // (op 753) 

v_and_b32 v5, 15, v1 // (op 753) 
s_mul_i32 s52, s16, s8 // (op 753) 
s_add_u32 s52, s52, s53 // (op 753) 

v_lshl_add_u32 v8, v7, 4, v5 // (op 753) 


s_mul_hi_u32 s53, s52, s13 // (op 753) 
v_and_b32 v5, 63, v1 // (op 753) 
s_sub_u32 s54, s52, s53 // (op 753) 
v_lshrrev_b32 v7, 4, v5 // (op 753) 

s_lshr_b32 s55, s54, 1 // (op 753) 

v_lshlrev_b32 v5, 5, v7 // (op 753) 

s_add_u32 s54, s55, s53 // (op 753) 


s_lshr_b32 s53, s54, s11 // (op 753) 

s_and_b32 s54, s11, 2147483648 // (op 753) 
s_cmp_eq_i32 s54, 0 // (op 753) 
s_mov_b32 s55, scc // (op 753) 

s_cmp_eq_i32 s55, 0 // (op 753) 
s_cselect_b32 s54, s52, s53 // (op 753) 


s_mul_hi_u32 s53, s54, s12 // (op 753) 
s_sub_u32 s55, s54, s53 // (op 753) 
s_lshr_b32 s76, s55, 1 // (op 753) 

s_add_u32 s55, s76, s53 // (op 753) 


s_lshr_b32 s53, s55, s10 // (op 753) 

s_and_b32 s55, s10, 2147483648 // (op 753) 
s_cmp_eq_i32 s55, 0 // (op 753) 
s_mov_b32 s76, scc // (op 753) 

s_cmp_eq_i32 s76, 0 // (op 753) 
s_cselect_b32 s55, s54, s53 // (op 753) 


s_mul_i32 s53, s55, s14 // (op 753) 

s_sub_u32 s55, s54, s53 // (op 753) 

v_lshl_add_u32 v7, s55, 5, v8 // (op 753) 


v_mul_lo_u32 v9, v7, s31 // (op 753) most significant: low of low * high
v_mul_hi_u32 v10, v7, s30 // (op 753) most significant: high of low * low
v_mul_lo_u32 v8, v7, s30 // (op 753) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 753) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 753) 
s_mul_i32 s53, s54, s76 // (op 753) 


s_sub_u32 s54, s52, s53 // (op 753) 


v_lshl_add_u32 v7, s54, 7, v5 // (op 753) 


v_mov_b32 v11, 0 // (op 753) convert
v_mov_b32 v10, v7 // (op 753) convert

v_add_co_u32 v12, s[52:53], v10, v8 // (op 753) least significant half
v_addc_co_u32 v13, s[52:53], v11, v9, s[52:53] // (op 753) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 753) 


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 753) 

v_mov_b32 v5, v10 // (op 753) convert

s_mul_i32 s53, s30, s23 // (op 755) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 755) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 755) least significant
s_mul_hi_u32 s55, s30, s22 // (op 755) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 755) most significant: sum
s_add_u32 s53, s53, s55 // (op 755) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 755) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 755) 

s_mov_b32 s64, s32 // (op 755) reg expression
s_mov_b32 s65, s33 // (op 755) reg expression
s_mov_b32 s66, s52 // (op 755) reg expression
s_mov_b32 s67, 131072 // (op 755) call()

s_and_b32 s52, s7, 4294967294 // (op 841) 
v_lshrrev_b32 v7, 7, v1 // (op 841) 
s_add_u32 s53, s0, s52 // (op 841) 

v_and_b32 v8, 15, v1 // (op 841) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 841) 


s_mul_i32 s52, s16, s8 // (op 841) 
s_add_u32 s52, s52, s53 // (op 841) 

v_and_b32 v7, 63, v1 // (op 841) 
s_mul_hi_u32 s53, s52, s13 // (op 841) 
v_lshrrev_b32 v8, 4, v7 // (op 841) 

s_sub_u32 s54, s52, s53 // (op 841) 
s_lshr_b32 s55, s54, 1 // (op 841) 

s_add_u32 s54, s55, s53 // (op 841) 


s_lshr_b32 s53, s54, s11 // (op 841) 

s_and_b32 s54, s11, 2147483648 // (op 841) 
s_cmp_eq_i32 s54, 0 // (op 841) 
s_mov_b32 s55, scc // (op 841) 

s_cmp_eq_i32 s55, 0 // (op 841) 
s_cselect_b32 s54, s52, s53 // (op 841) 


s_mul_hi_u32 s53, s54, s12 // (op 841) 
s_sub_u32 s55, s54, s53 // (op 841) 
s_lshr_b32 s76, s55, 1 // (op 841) 

s_add_u32 s55, s76, s53 // (op 841) 


s_lshr_b32 s53, s55, s10 // (op 841) 

s_and_b32 s55, s10, 2147483648 // (op 841) 
s_cmp_eq_i32 s55, 0 // (op 841) 
s_mov_b32 s76, scc // (op 841) 

s_cmp_eq_i32 s76, 0 // (op 841) 
s_cselect_b32 s55, s54, s53 // (op 841) 


v_lshl_add_u32 v7, s55, 5, v9 // (op 841) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 841) 
s_mul_i32 s53, s54, s76 // (op 841) 


s_sub_u32 s54, s52, s53 // (op 841) 


v_lshl_add_u32 v9, s54, 2, v8 // (op 841) 


v_mul_lo_u32 v8, v7, s34 // (op 841) 
v_add_u32 v8, v8, v9 // (op 841) 


s_mul_i32 s53, s34, s29 // (op 843) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 843) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 843) least significant
s_mul_hi_u32 s55, s34, s28 // (op 843) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 843) most significant: sum
s_add_u32 s53, s53, s55 // (op 843) most significant: sum


s_mov_b32 s68, s36 // (op 843) reg expression
s_mov_b32 s69, s37 // (op 843) reg expression
s_mov_b32 s70, s52 // (op 843) reg expression
s_mov_b32 s71, 131072 // (op 843) call()

s_and_b32 s52, s7, 4294967294 // (op 929) 
v_lshrrev_b32 v7, 6, v1 // (op 929) 
s_add_u32 s53, s0, s52 // (op 929) 

v_and_b32 v9, 1, v7 // (op 929) 

v_and_b32 v7, 15, v1 // (op 929) 
s_mul_i32 s52, s16, s8 // (op 929) 
s_add_u32 s52, s52, s53 // (op 929) 

v_lshl_add_u32 v10, v9, 4, v7 // (op 929) 


s_mul_hi_u32 s53, s52, s13 // (op 929) 
v_and_b32 v7, 63, v1 // (op 929) 
s_sub_u32 s54, s52, s53 // (op 929) 
v_lshrrev_b32 v9, 4, v7 // (op 929) 

s_lshr_b32 s55, s54, 1 // (op 929) 

s_add_u32 s54, s55, s53 // (op 929) 


s_lshr_b32 s53, s54, s11 // (op 929) 

s_and_b32 s54, s11, 2147483648 // (op 929) 
s_cmp_eq_i32 s54, 0 // (op 929) 
s_mov_b32 s55, scc // (op 929) 

s_cmp_eq_i32 s55, 0 // (op 929) 
s_cselect_b32 s54, s52, s53 // (op 929) 


s_mul_hi_u32 s53, s54, s12 // (op 929) 
s_sub_u32 s55, s54, s53 // (op 929) 
s_lshr_b32 s76, s55, 1 // (op 929) 

s_add_u32 s55, s76, s53 // (op 929) 


s_lshr_b32 s53, s55, s10 // (op 929) 

s_and_b32 s55, s10, 2147483648 // (op 929) 
s_cmp_eq_i32 s55, 0 // (op 929) 
s_mov_b32 s76, scc // (op 929) 

s_cmp_eq_i32 s76, 0 // (op 929) 
s_cselect_b32 s55, s54, s53 // (op 929) 


s_mul_i32 s53, s55, s14 // (op 929) 

s_sub_u32 s55, s54, s53 // (op 929) 

v_lshl_add_u32 v7, s55, 5, v10 // (op 929) 


v_mul_lo_u32 v10, v7, s38 // (op 929) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 929) 
s_mul_i32 s53, s54, s76 // (op 929) 


s_sub_u32 s54, s52, s53 // (op 929) 


v_lshl_add_u32 v7, s54, 2, v9 // (op 929) 


v_add_u32 v9, v7, v10 // (op 929) 


s_mul_i32 s53, s38, s23 // (op 931) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 931) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 931) least significant
s_mul_hi_u32 s55, s38, s22 // (op 931) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 931) most significant: sum
s_add_u32 s53, s53, s55 // (op 931) most significant: sum


s_mov_b32 s72, s40 // (op 931) reg expression
s_mov_b32 s73, s41 // (op 931) reg expression
s_mov_b32 s74, s52 // (op 931) reg expression
s_mov_b32 s75, 131072 // (op 931) call()

s_and_b32 s52, s7, 4294967294 // (op 478) 
s_cmp_lt_u32 s52, s7 // (op 477) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_477 // (op 477) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_477 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_477:


s_add_u32 s52, s52, 1 // (op 479) 
v_mov_b32 v7, v4 // (op 483) 

buffer_load_dwordx4 v[10:13], v7, s[60:63], 0 offen // (op 483) Load value
buffer_load_dwordx4 v[14:17], v7, s[60:63], 0 offen offset:64 // (op 483) Load value

v_add_u32 v4, 128, v4 // (op 651) 


v_mov_b32 v7, v5 // (op 485) 

buffer_load_dwordx4 v[18:21], v7, s[64:67], 0 offen // (op 485) Load value

v_add_u32 v5, 64, v5 // (op 745) 


v_mov_b32 v7, v8 // (op 487) 

buffer_load_ubyte v22, v7, s[68:71], 0 offen offset: 0 // (op 487) Load value

v_add_u32 v8, 4, v8 // (op 833) 


v_mov_b32 v7, v9 // (op 489) 

buffer_load_ubyte v23, v7, s[72:75], 0 offen offset: 0 // (op 489) Load value

v_add_u32 v9, 4, v9 // (op 921) 
s_waitcnt vmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:17], v[18:21], a[0:3], v22, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 490) 




s_cmp_lt_u32 s52, s7 // (op 477) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_477 // (op 477) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_477 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_477:





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 257) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_257 // (op 257) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_257
v_lshrrev_b32 v4, 5, v1 // (op 1095) 
v_lshlrev_b32 v5, 2, v4 // (op 1095) 

v_lshl_add_u32 v4, s16, 5, v5 // (op 1095) 

v_and_b32 v5, -32, v1 // (op 1095) 
v_sub_u32 v7, v1, v5 // (op 1095) 

v_lshl_add_u32 v5, v4, 5, v7 // (op 1095) 


v_lshlrev_b32 v4, 2, v5 // (op 1095) 

s_lshl_b32 s52, s15, 12 // (op 1097) 
s_mov_b32 s76, s42 // (op 1097) reg expression
s_mov_b32 s77, s43 // (op 1097) reg expression
s_mov_b32 s78, s52 // (op 1097) reg expression
s_mov_b32 s79, 131072 // (op 1097) call()




buffer_store_dword a0, v4, s[76:79], 0 offen sc1 // (op 254) Store value
buffer_store_dword a1, v4, s[76:79], 0 offen offset:128 sc1 // (op 254) Store value
buffer_store_dword a2, v4, s[76:79], 0 offen offset:256 sc1 // (op 254) Store value
buffer_store_dword a3, v4, s[76:79], 0 offen offset:384 sc1 // (op 254) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 258) 
v_cmp_eq_i32 vcc, v1, 0 // (op 262) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_262 // (op 262) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_262
s_mov_b32 s52, 1 // (op 260) call()
s_mov_b32 s55, 0 // (op 261) convert
s_mov_b32 s54, s16 // (op 261) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 261) 
s_add_u32 s80, s44, s54 // (op 261) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 261) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 261) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_262 // (op 262) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_262
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_262:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_262:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_257 // (op 257) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_257
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Send_Tile_257:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Send_Tile_257:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 273) 
s_mov_b32 s52, scc // (op 273) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 273) 
s_mov_b32 s53, s54 // (op 273) convert

s_sub_u32 s54, s53, 1 // (op 273) 

s_cmp_lt_u32 s19, s54 // (op 273) 
s_mov_b32 s53, scc // (op 273) 

s_and_b32 s54, s52, s53 // (op 273) 
s_cmp_lg_u32 s54, 0 // (op 273) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_273 // (op 273) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_273
s_mov_b32 s52, 0 // (op 239) call()
s_mul_i32 s53, s52, s8 // (op 238) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 238) 
s_mov_b32 s80, s54 // (op 238) convert

s_sub_u32 s54, s80, s19 // (op 238) 

s_sub_u32 s55, s54, 1 // (op 238) 

s_cmp_lt_u32 s53, s55 // (op 238) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_238 // (op 238) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_238 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_238:
s_add_u32 s53, s16, 1 // (op 1112) 
v_lshrrev_b32 v4, 5, v1 // (op 1112) 
s_add_u32 s54, s53, s52 // (op 1112) 
s_lshl_b32 s54, s54, 5 // (op 1112) 

v_lshlrev_b32 v5, 2, v4 // (op 1112) 

v_add_lshl_u32 v4, s54, v5, 5 // (op 1112) 


v_and_b32 v5, -32, v1 // (op 1112) 
v_sub_u32 v7, v1, v5 // (op 1112) 

v_add_lshl_u32 v5, v4, v7, 2 // (op 1112) 


s_lshl_b32 s53, s15, 12 // (op 1114) 
s_mov_b32 s76, s42 // (op 1114) reg expression
s_mov_b32 s77, s43 // (op 1114) reg expression
s_mov_b32 s78, s53 // (op 1114) reg expression
s_mov_b32 s79, 131072 // (op 1114) call()

RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_277:
s_add_u32 s53, s16, 1 // (op 275) 
s_add_u32 s54, s53, s52 // (op 275) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 275) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 275) 

s_add_u32 s80, s44, s54 // (op 275) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 275) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 275) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 277) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_277 // (op 277) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_277 if true)
s_barrier  // (op 281) 
v_cmp_eq_i32 vcc, v1, 0 // (op 280) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_280 // (op 280) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_280
s_mov_b32 s53, 0 // (op 278) call()
s_add_u32 s54, s16, 1 // (op 279) 
s_add_u32 s80, s54, s52 // (op 279) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 279) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 279) 

s_add_u32 s54, s44, s80 // (op 279) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 279) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 279) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_280 // (op 280) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_280
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_280:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_280:




buffer_load_dword v8, v5, s[76:79], 0 offen // (op 292) Load value
buffer_load_dword v9, v5, s[76:79], 0 offen offset:128 // (op 292) Load value
buffer_load_dword v10, v5, s[76:79], 0 offen offset:256 // (op 292) Load value
buffer_load_dword v11, v5, s[76:79], 0 offen offset:384 // (op 292) Load value
v_accvgpr_read v12, a0 // (op 295) reg expression
v_accvgpr_read v13, a1 // (op 295) reg expression
v_accvgpr_read v14, a2 // (op 295) reg expression
v_accvgpr_read v15, a3 // (op 295) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v12, v12, v8 // (op 297) 
s_waitcnt vmcnt(2)
v_add_f32 v13, v13, v9 // (op 297) 
s_waitcnt vmcnt(1)
v_add_f32 v14, v14, v10 // (op 297) 
s_waitcnt vmcnt(0)
v_add_f32 v15, v15, v11 // (op 297) 

v_accvgpr_write a0, v12 // (op 299) reg expression
v_accvgpr_write a1, v13 // (op 299) reg expression
v_accvgpr_write a2, v14 // (op 299) reg expression
v_accvgpr_write a3, v15 // (op 299) reg expression

s_add_u32 s52, s52, 1 // (op 240) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 238) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 238) 
s_mov_b32 s80, s54 // (op 238) convert

s_sub_u32 s54, s80, s19 // (op 238) 

s_sub_u32 s55, s54, 1 // (op 238) 

s_cmp_lt_u32 s53, s55 // (op 238) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_238 // (op 238) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_ReceiveTileLoop_238 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_238:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_273 // (op 273) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_273
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Receive_Tile_273:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Receive_Tile_273:

s_cmp_eq_i32 s18, 0 // (op 316) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_316 // (op 316) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_316
v_cmp_eq_f32 vcc, 0.00000, v0 // (op 99) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_99 // (op 99) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_99
v_accvgpr_read v8, a0 // (op 34) 
v_accvgpr_read v9, a1 // (op 34) 
v_accvgpr_read v10, a2 // (op 34) 
v_accvgpr_read v11, a3 // (op 34) 
v_mul_f32 v12, v3, v8 // (op 34) 
v_mul_f32 v13, v3, v9 // (op 34) 
v_mul_f32 v14, v3, v10 // (op 34) 
v_mul_f32 v15, v3, v11 // (op 34) 

v_lshrrev_b32 v4, 7, v1 // (op 1027) 
s_mul_i32 s52, s16, s8 // (op 1027) 
s_add_u32 s52, s52, s0 // (op 1027) 
v_and_b32 v5, 63, v1 // (op 1027) 
s_mul_hi_u32 s53, s52, s13 // (op 1027) 
v_lshrrev_b32 v7, 4, v5 // (op 1027) 
s_sub_u32 s54, s52, s53 // (op 1027) 
v_lshlrev_b32 v8, 2, v7 // (op 1027) 

s_lshr_b32 s55, s54, 1 // (op 1027) 

v_lshl_add_u32 v7, v4, 4, v8 // (op 1027) 


s_add_u32 s54, s55, s53 // (op 1027) 


v_lshrrev_b32 v4, 2, v5 // (op 1027) 

s_lshr_b32 s53, s54, s11 // (op 1027) 

v_and_b32 v5, 3, v4 // (op 1027) 

s_and_b32 s54, s11, 2147483648 // (op 1027) 
v_and_b32 v4, 3, v1 // (op 1027) 
s_cmp_eq_i32 s54, 0 // (op 1027) 
s_mov_b32 s55, scc // (op 1027) 

v_lshl_add_u32 v8, v5, 2, v4 // (op 1027) 


s_cmp_eq_i32 s55, 0 // (op 1027) 
s_cselect_b32 s54, s52, s53 // (op 1027) 



v_lshrrev_b32 v4, 6, v1 // (op 1027) 
s_mul_hi_u32 s55, s54, s12 // (op 1027) 
v_and_b32 v5, 1, v4 // (op 1027) 

s_sub_u32 s52, s54, s55 // (op 1027) 
v_lshl_add_u32 v4, v5, 4, v8 // (op 1027) 


s_lshr_b32 s53, s52, 1 // (op 1027) 

s_add_u32 s52, s53, s55 // (op 1027) 


s_lshr_b32 s53, s52, s10 // (op 1027) 

s_and_b32 s52, s10, 2147483648 // (op 1027) 
s_cmp_eq_i32 s52, 0 // (op 1027) 
s_mov_b32 s55, scc // (op 1027) 

s_cmp_eq_i32 s55, 0 // (op 1027) 
s_cselect_b32 s52, s54, s53 // (op 1027) 


s_mul_i32 s53, s52, s14 // (op 1027) 
v_lshl_add_u32 v5, s52, 5, v7 // (op 1027) 


s_sub_u32 s52, s54, s53 // (op 1027) 


v_lshl_add_u32 v7, s52, 5, v4 // (op 1027) 


v_mul_lo_u32 v4, v7, s46 // (op 1027) 

v_add_lshl_u32 v7, v5, v4, 2 // (op 1027) 


s_mul_i32 s53, s46, s23 // (op 1029) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1029) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1029) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1029) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1029) most significant: sum
s_add_u32 s53, s53, s55 // (op 1029) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1029) 

s_mov_b32 s76, s48 // (op 1029) reg expression
s_mov_b32 s77, s49 // (op 1029) reg expression
s_mov_b32 s78, s54 // (op 1029) reg expression
s_mov_b32 s79, 131072 // (op 1029) call()




buffer_store_dwordx4 v[12:15], v7, s[76:79], 0 offen // (op 40) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_99 // (op 99) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_99
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_99:



buffer_load_dwordx4 v[8:11], v6, s[56:59], 0 offen // (op 111) Load value
v_accvgpr_read v12, a0 // (op 108) 
v_accvgpr_read v13, a1 // (op 108) 
v_accvgpr_read v14, a2 // (op 108) 
v_accvgpr_read v15, a3 // (op 108) 
v_mul_f32 v16, v3, v12 // (op 108) 
v_mul_f32 v17, v3, v13 // (op 108) 
v_mul_f32 v18, v3, v14 // (op 108) 
v_mul_f32 v19, v3, v15 // (op 108) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v0, v8 // (op 112) 
v_mul_f32 v13, v0, v9 // (op 112) 
v_mul_f32 v14, v0, v10 // (op 112) 
v_mul_f32 v15, v0, v11 // (op 112) 

v_add_f32 v8, v12, v16 // (op 109) 
v_add_f32 v9, v13, v17 // (op 109) 
v_add_f32 v10, v14, v18 // (op 109) 
v_add_f32 v11, v15, v19 // (op 109) 


v_lshrrev_b32 v7, 7, v1 // (op 1044) 
s_mul_i32 s52, s16, s8 // (op 1044) 
s_add_u32 s52, s52, s0 // (op 1044) 
v_and_b32 v4, 63, v1 // (op 1044) 
s_mul_hi_u32 s53, s52, s13 // (op 1044) 
v_lshrrev_b32 v5, 4, v4 // (op 1044) 
s_sub_u32 s54, s52, s53 // (op 1044) 
v_lshlrev_b32 v12, 2, v5 // (op 1044) 

s_lshr_b32 s55, s54, 1 // (op 1044) 

v_lshl_add_u32 v5, v7, 4, v12 // (op 1044) 


s_add_u32 s54, s55, s53 // (op 1044) 


v_lshrrev_b32 v7, 2, v4 // (op 1044) 

s_lshr_b32 s53, s54, s11 // (op 1044) 

v_and_b32 v4, 3, v7 // (op 1044) 

s_and_b32 s54, s11, 2147483648 // (op 1044) 
v_and_b32 v7, 3, v1 // (op 1044) 
s_cmp_eq_i32 s54, 0 // (op 1044) 
s_mov_b32 s55, scc // (op 1044) 

v_lshl_add_u32 v12, v4, 2, v7 // (op 1044) 


s_cmp_eq_i32 s55, 0 // (op 1044) 
s_cselect_b32 s54, s52, s53 // (op 1044) 



v_lshrrev_b32 v4, 6, v1 // (op 1044) 
s_mul_hi_u32 s55, s54, s12 // (op 1044) 
v_and_b32 v7, 1, v4 // (op 1044) 

s_sub_u32 s52, s54, s55 // (op 1044) 
v_lshl_add_u32 v4, v7, 4, v12 // (op 1044) 


s_lshr_b32 s53, s52, 1 // (op 1044) 

s_add_u32 s52, s53, s55 // (op 1044) 


s_lshr_b32 s53, s52, s10 // (op 1044) 

s_and_b32 s52, s10, 2147483648 // (op 1044) 
s_cmp_eq_i32 s52, 0 // (op 1044) 
s_mov_b32 s55, scc // (op 1044) 

s_cmp_eq_i32 s55, 0 // (op 1044) 
s_cselect_b32 s52, s54, s53 // (op 1044) 


s_mul_i32 s53, s52, s14 // (op 1044) 
v_lshl_add_u32 v7, s52, 5, v5 // (op 1044) 


s_sub_u32 s52, s54, s53 // (op 1044) 


v_lshl_add_u32 v5, s52, 5, v4 // (op 1044) 


v_mul_lo_u32 v4, v5, s46 // (op 1044) 

v_add_lshl_u32 v5, v7, v4, 2 // (op 1044) 


s_mul_i32 s53, s46, s23 // (op 1046) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1046) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1046) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1046) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1046) most significant: sum
s_add_u32 s53, s53, s55 // (op 1046) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1046) 

s_mov_b32 s76, s48 // (op 1046) reg expression
s_mov_b32 s77, s49 // (op 1046) reg expression
s_mov_b32 s78, s54 // (op 1046) reg expression
s_mov_b32 s79, 131072 // (op 1046) call()




buffer_store_dwordx4 v[8:11], v5, s[76:79], 0 offen // (op 110) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_99:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_316 // (op 316) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_316
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_316:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_316:

s_add_u32 s0, s0, s7 // (op 231) 
s_mul_i32 s52, s8, s16 // (op 229) 
s_cmp_gt_u32 s9, s52 // (op 229) 
s_mov_b32 s53, scc // (op 229) 
s_sub_u32 s54, s9, s52 // (op 229) 

s_cmp_eq_i32 s53, 0 // (op 229) 
s_cselect_b32 s52, 0, s54 // (op 229) 


s_cmp_lt_u32 s8, s52 // (op 229) 
s_mov_b32 s53, scc // (op 229) 
s_cmp_eq_i32 s53, 0 // (op 229) 
s_cselect_b32 s54, s52, s8 // (op 229) 


s_cmp_lt_u32 s0, s54 // (op 229) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_229 // (op 229) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_SKStreamTileLoop_229 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_229:









s_mov_b32 s0, 0 // (op 370) call()
s_mov_b32 s1, 0 // (op 371) call()
s_mov_b32 s0, 0 // (op 385) call()
s_mul_i32 s15, s50, s16 // (op 384) 
s_cmp_gt_u32 s51, s15 // (op 384) 
s_mov_b32 s6, scc // (op 384) 
s_sub_u32 s7, s51, s15 // (op 384) 

s_cmp_eq_i32 s6, 0 // (op 384) 
s_cselect_b32 s8, 0, s7 // (op 384) 


s_cmp_lt_u32 s50, s8 // (op 384) 
s_mov_b32 s15, scc // (op 384) 
s_cmp_eq_i32 s15, 0 // (op 384) 
s_cselect_b32 s6, s8, s50 // (op 384) 


s_cmp_lt_u32 s0, s6 // (op 384) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_384 // (op 384) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_384 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_384:
v_lshrrev_b32 v4, 7, v1 // (op 1010) 
s_mul_i32 s15, s16, s50 // (op 1010) 
s_add_u32 s15, s15, s0 // (op 1010) 
v_and_b32 v5, 63, v1 // (op 1010) 
s_add_u32 s6, s15, s9 // (op 1010) 

v_lshrrev_b32 v6, 4, v5 // (op 1010) 
s_mul_hi_u32 s15, s6, s13 // (op 1010) 
v_lshlrev_b32 v7, 2, v6 // (op 1010) 

s_sub_u32 s7, s6, s15 // (op 1010) 
v_lshl_add_u32 v6, v4, 4, v7 // (op 1010) 


s_lshr_b32 s8, s7, 1 // (op 1010) 

v_lshrrev_b32 v4, 2, v5 // (op 1010) 

s_add_u32 s7, s8, s15 // (op 1010) 


v_and_b32 v5, 3, v4 // (op 1010) 

s_lshr_b32 s8, s7, s11 // (op 1010) 

v_and_b32 v4, 3, v1 // (op 1010) 
s_and_b32 s7, s11, 2147483648 // (op 1010) 
v_lshl_add_u32 v7, v5, 2, v4 // (op 1010) 


s_cmp_eq_i32 s7, 0 // (op 1010) 
s_mov_b32 s15, scc // (op 1010) 

v_lshrrev_b32 v4, 6, v1 // (op 1010) 
s_cmp_eq_i32 s15, 0 // (op 1010) 
s_cselect_b32 s7, s6, s8 // (op 1010) 



v_and_b32 v5, 1, v4 // (op 1010) 

s_mul_hi_u32 s6, s7, s12 // (op 1010) 
v_lshl_add_u32 v4, v5, 4, v7 // (op 1010) 


s_sub_u32 s8, s7, s6 // (op 1010) 
s_lshr_b32 s15, s8, 1 // (op 1010) 

s_add_u32 s8, s15, s6 // (op 1010) 


s_lshr_b32 s6, s8, s10 // (op 1010) 

s_and_b32 s8, s10, 2147483648 // (op 1010) 
s_cmp_eq_i32 s8, 0 // (op 1010) 
s_mov_b32 s15, scc // (op 1010) 

s_cmp_eq_i32 s15, 0 // (op 1010) 
s_cselect_b32 s8, s7, s6 // (op 1010) 


s_mul_i32 s6, s8, s14 // (op 1010) 
v_lshl_add_u32 v5, s8, 5, v6 // (op 1010) 


s_sub_u32 s8, s7, s6 // (op 1010) 


v_lshl_add_u32 v6, s8, 5, v4 // (op 1010) 


v_mul_lo_u32 v4, v6, s4 // (op 1010) 

v_add_lshl_u32 v6, v5, v4, 2 // (op 1010) 


s_mul_i32 s19, s4, s23 // (op 1012) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1012) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1012) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1012) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1012) most significant: sum
s_add_u32 s19, s19, s6 // (op 1012) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1012) 

s_mov_b32 s56, s20 // (op 1012) reg expression
s_mov_b32 s57, s21 // (op 1012) reg expression
s_mov_b32 s58, s6 // (op 1012) reg expression
s_mov_b32 s59, 131072 // (op 1012) call()

v_lshrrev_b32 v4, 7, v1 // (op 709) 
s_mul_i32 s15, s16, s50 // (op 709) 
s_add_u32 s15, s15, s0 // (op 709) 
v_and_b32 v5, 15, v1 // (op 709) 
s_add_u32 s6, s15, s9 // (op 709) 

v_lshl_add_u32 v7, v4, 4, v5 // (op 709) 


s_mul_hi_u32 s15, s6, s13 // (op 709) 
v_and_b32 v4, 48, v1 // (op 709) 
s_sub_u32 s7, s6, s15 // (op 709) 
s_lshr_b32 s8, s7, 1 // (op 709) 

s_add_u32 s7, s8, s15 // (op 709) 


s_lshr_b32 s8, s7, s11 // (op 709) 

s_and_b32 s7, s11, 2147483648 // (op 709) 
s_cmp_eq_i32 s7, 0 // (op 709) 
s_mov_b32 s15, scc // (op 709) 

s_cmp_eq_i32 s15, 0 // (op 709) 
s_cselect_b32 s7, s6, s8 // (op 709) 


s_mul_hi_u32 s8, s7, s12 // (op 709) 
s_sub_u32 s15, s7, s8 // (op 709) 
s_lshr_b32 s18, s15, 1 // (op 709) 

s_add_u32 s15, s18, s8 // (op 709) 


s_lshr_b32 s8, s15, s10 // (op 709) 

s_and_b32 s15, s10, 2147483648 // (op 709) 
s_cmp_eq_i32 s15, 0 // (op 709) 
s_mov_b32 s18, scc // (op 709) 

s_cmp_eq_i32 s18, 0 // (op 709) 
s_cselect_b32 s15, s7, s8 // (op 709) 


v_lshl_add_u32 v5, s15, 5, v7 // (op 709) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 709) 
s_mul_i32 s8, s7, s18 // (op 709) 


s_sub_u32 s7, s6, s8 // (op 709) 


v_lshl_add_u32 v7, s7, 7, v4 // (op 709) 


v_mul_lo_u32 v4, v5, s24 // (op 709) 
v_add_u32 v4, v4, v7 // (op 709) 


s_mul_i32 s19, s24, s29 // (op 711) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 711) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 711) least significant
s_mul_hi_u32 s6, s24, s28 // (op 711) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 711) most significant: sum
s_add_u32 s19, s19, s6 // (op 711) most significant: sum


s_mov_b32 s60, s26 // (op 711) reg expression
s_mov_b32 s61, s27 // (op 711) reg expression
s_mov_b32 s62, s18 // (op 711) reg expression
s_mov_b32 s63, 131072 // (op 711) call()

v_lshrrev_b32 v5, 6, v1 // (op 797) 
s_mul_i32 s15, s16, s50 // (op 797) 
s_add_u32 s15, s15, s0 // (op 797) 
v_and_b32 v7, 1, v5 // (op 797) 

s_add_u32 s6, s15, s9 // (op 797) 

v_and_b32 v5, 15, v1 // (op 797) 
s_mul_hi_u32 s15, s6, s13 // (op 797) 
v_lshl_add_u32 v8, v7, 4, v5 // (op 797) 


s_sub_u32 s7, s6, s15 // (op 797) 
v_and_b32 v5, 63, v1 // (op 797) 
s_lshr_b32 s8, s7, 1 // (op 797) 

v_lshrrev_b32 v7, 4, v5 // (op 797) 

s_add_u32 s7, s8, s15 // (op 797) 


v_lshlrev_b32 v5, 5, v7 // (op 797) 

s_lshr_b32 s8, s7, s11 // (op 797) 

s_and_b32 s7, s11, 2147483648 // (op 797) 
s_cmp_eq_i32 s7, 0 // (op 797) 
s_mov_b32 s15, scc // (op 797) 

s_cmp_eq_i32 s15, 0 // (op 797) 
s_cselect_b32 s7, s6, s8 // (op 797) 


s_mul_hi_u32 s8, s7, s12 // (op 797) 
s_sub_u32 s15, s7, s8 // (op 797) 
s_lshr_b32 s18, s15, 1 // (op 797) 

s_add_u32 s15, s18, s8 // (op 797) 


s_lshr_b32 s8, s15, s10 // (op 797) 

s_and_b32 s15, s10, 2147483648 // (op 797) 
s_cmp_eq_i32 s15, 0 // (op 797) 
s_mov_b32 s18, scc // (op 797) 

s_cmp_eq_i32 s18, 0 // (op 797) 
s_cselect_b32 s15, s7, s8 // (op 797) 


s_mul_i32 s8, s15, s14 // (op 797) 

s_sub_u32 s15, s7, s8 // (op 797) 

v_lshl_add_u32 v7, s15, 5, v8 // (op 797) 


v_mul_lo_u32 v9, v7, s31 // (op 797) most significant: low of low * high
v_mul_hi_u32 v10, v7, s30 // (op 797) most significant: high of low * low
v_mul_lo_u32 v8, v7, s30 // (op 797) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 797) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 797) 
s_mul_i32 s8, s7, s18 // (op 797) 


s_sub_u32 s7, s6, s8 // (op 797) 


v_lshl_add_u32 v7, s7, 7, v5 // (op 797) 


v_mov_b32 v11, 0 // (op 797) convert
v_mov_b32 v10, v7 // (op 797) convert

v_add_co_u32 v12, s[18:19], v10, v8 // (op 797) least significant half
v_addc_co_u32 v13, s[18:19], v11, v9, s[18:19] // (op 797) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 797) 


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 797) 

v_mov_b32 v5, v10 // (op 797) convert

s_mul_i32 s19, s30, s23 // (op 799) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 799) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 799) least significant
s_mul_hi_u32 s6, s30, s22 // (op 799) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 799) most significant: sum
s_add_u32 s19, s19, s6 // (op 799) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 799) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 799) 

s_mov_b32 s64, s32 // (op 799) reg expression
s_mov_b32 s65, s33 // (op 799) reg expression
s_mov_b32 s66, s18 // (op 799) reg expression
s_mov_b32 s67, 131072 // (op 799) call()

v_lshrrev_b32 v7, 7, v1 // (op 885) 
s_mul_i32 s15, s16, s50 // (op 885) 
s_add_u32 s15, s15, s0 // (op 885) 
v_and_b32 v8, 15, v1 // (op 885) 
s_add_u32 s6, s15, s9 // (op 885) 

v_lshl_add_u32 v9, v7, 4, v8 // (op 885) 


s_mul_hi_u32 s15, s6, s13 // (op 885) 
v_and_b32 v7, 63, v1 // (op 885) 
s_sub_u32 s7, s6, s15 // (op 885) 
v_lshrrev_b32 v8, 4, v7 // (op 885) 

s_lshr_b32 s8, s7, 1 // (op 885) 

s_add_u32 s7, s8, s15 // (op 885) 


s_lshr_b32 s8, s7, s11 // (op 885) 

s_and_b32 s7, s11, 2147483648 // (op 885) 
s_cmp_eq_i32 s7, 0 // (op 885) 
s_mov_b32 s15, scc // (op 885) 

s_cmp_eq_i32 s15, 0 // (op 885) 
s_cselect_b32 s7, s6, s8 // (op 885) 


s_mul_hi_u32 s8, s7, s12 // (op 885) 
s_sub_u32 s15, s7, s8 // (op 885) 
s_lshr_b32 s18, s15, 1 // (op 885) 

s_add_u32 s15, s18, s8 // (op 885) 


s_lshr_b32 s8, s15, s10 // (op 885) 

s_and_b32 s15, s10, 2147483648 // (op 885) 
s_cmp_eq_i32 s15, 0 // (op 885) 
s_mov_b32 s18, scc // (op 885) 

s_cmp_eq_i32 s18, 0 // (op 885) 
s_cselect_b32 s15, s7, s8 // (op 885) 


v_lshl_add_u32 v7, s15, 5, v9 // (op 885) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 885) 
s_mul_i32 s8, s7, s18 // (op 885) 


s_sub_u32 s7, s6, s8 // (op 885) 


v_lshl_add_u32 v9, s7, 2, v8 // (op 885) 


v_mul_lo_u32 v8, v7, s34 // (op 885) 
v_add_u32 v8, v8, v9 // (op 885) 


s_mul_i32 s19, s34, s29 // (op 887) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 887) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 887) least significant
s_mul_hi_u32 s6, s34, s28 // (op 887) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 887) most significant: sum
s_add_u32 s19, s19, s6 // (op 887) most significant: sum


s_mov_b32 s68, s36 // (op 887) reg expression
s_mov_b32 s69, s37 // (op 887) reg expression
s_mov_b32 s70, s18 // (op 887) reg expression
s_mov_b32 s71, 131072 // (op 887) call()

v_lshrrev_b32 v7, 6, v1 // (op 973) 
s_mul_i32 s15, s16, s50 // (op 973) 
s_add_u32 s15, s15, s0 // (op 973) 
v_and_b32 v9, 1, v7 // (op 973) 

s_add_u32 s6, s15, s9 // (op 973) 

v_and_b32 v7, 15, v1 // (op 973) 
s_mul_hi_u32 s15, s6, s13 // (op 973) 
v_lshl_add_u32 v10, v9, 4, v7 // (op 973) 


s_sub_u32 s7, s6, s15 // (op 973) 
v_and_b32 v7, 63, v1 // (op 973) 
s_lshr_b32 s8, s7, 1 // (op 973) 

v_lshrrev_b32 v9, 4, v7 // (op 973) 

s_add_u32 s7, s8, s15 // (op 973) 


s_lshr_b32 s8, s7, s11 // (op 973) 

s_and_b32 s7, s11, 2147483648 // (op 973) 
s_cmp_eq_i32 s7, 0 // (op 973) 
s_mov_b32 s15, scc // (op 973) 

s_cmp_eq_i32 s15, 0 // (op 973) 
s_cselect_b32 s7, s6, s8 // (op 973) 


s_mul_hi_u32 s8, s7, s12 // (op 973) 
s_sub_u32 s15, s7, s8 // (op 973) 
s_lshr_b32 s18, s15, 1 // (op 973) 

s_add_u32 s15, s18, s8 // (op 973) 


s_lshr_b32 s8, s15, s10 // (op 973) 

s_and_b32 s15, s10, 2147483648 // (op 973) 
s_cmp_eq_i32 s15, 0 // (op 973) 
s_mov_b32 s18, scc // (op 973) 

s_cmp_eq_i32 s18, 0 // (op 973) 
s_cselect_b32 s15, s7, s8 // (op 973) 


s_mul_i32 s8, s15, s14 // (op 973) 

s_sub_u32 s15, s7, s8 // (op 973) 

v_lshl_add_u32 v7, s15, 5, v10 // (op 973) 


v_mul_lo_u32 v10, v7, s38 // (op 973) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 973) 
s_mul_i32 s8, s7, s18 // (op 973) 


s_sub_u32 s7, s6, s8 // (op 973) 


v_lshl_add_u32 v7, s7, 2, v9 // (op 973) 


v_add_u32 v9, v7, v10 // (op 973) 


s_mul_i32 s19, s38, s23 // (op 975) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 975) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 975) least significant
s_mul_hi_u32 s6, s38, s22 // (op 975) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 975) most significant: sum
s_add_u32 s19, s19, s6 // (op 975) most significant: sum


s_mov_b32 s72, s40 // (op 975) reg expression
s_mov_b32 s73, s41 // (op 975) reg expression
s_mov_b32 s74, s18 // (op 975) reg expression
s_mov_b32 s75, 131072 // (op 975) call()

v_accvgpr_write a0, 0.00000 // (op 146) call()
v_accvgpr_write a1, 0.00000 // (op 146) call()
v_accvgpr_write a2, 0.00000 // (op 146) call()
v_accvgpr_write a3, 0.00000 // (op 146) call()
s_mov_b32 s1, 0 // (op 148) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 147) 
s_mov_b32 s15, s18 // (op 147) convert

s_and_b32 s6, s15, 4294967294 // (op 147) 

s_cmp_lt_u32 s1, s6 // (op 147) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_147 // (op 147) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_147 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_147:


s_add_u32 s1, s1, 2 // (op 149) 
v_mov_b32 v7, v4 // (op 151) 

buffer_load_dwordx4 v[10:13], v7, s[60:63], 0 offen // (op 151) Load value
buffer_load_dwordx4 v[14:17], v7, s[60:63], 0 offen offset:64 // (op 151) Load value


v_mov_b32 v18, v5 // (op 153) 

buffer_load_dwordx4 v[20:23], v18, s[64:67], 0 offen // (op 153) Load value


v_mov_b32 v19, v8 // (op 155) 

buffer_load_ubyte v24, v19, s[68:71], 0 offen offset: 0 // (op 155) Load value


v_mov_b32 v25, v9 // (op 157) 

buffer_load_ubyte v26, v25, s[72:75], 0 offen offset: 0 // (op 157) Load value



buffer_load_dwordx4 v[28:31], v7, s[60:63], 0 offen offset:128 // (op 433) Load value
s_waitcnt vmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:17], v[20:23], a[0:3], v24, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 158) 
buffer_load_dwordx4 v[32:35], v7, s[60:63], 0 offen offset:192 // (op 433) Load value








v_add_u32 v4, 256, v4 // (op 699) 
buffer_load_dwordx4 v[10:13], v18, s[64:67], 0 offen offset:64 // (op 435) Load value




v_add_u32 v5, 128, v5 // (op 789) 
buffer_load_ubyte v7, v19, s[68:71], 0 offen offset:4 // (op 437) Load value




v_add_u32 v8, 8, v8 // (op 877) 
buffer_load_ubyte v14, v25, s[72:75], 0 offen offset:4 // (op 439) Load value

v_add_u32 v9, 8, v9 // (op 965) 
s_waitcnt vmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[10:13], a[0:3], v7, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 440) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 147) 
s_mov_b32 s15, s18 // (op 147) convert

s_and_b32 s6, s15, 4294967294 // (op 147) 

s_cmp_lt_u32 s1, s6 // (op 147) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_147 // (op 147) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoop_147 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_147:




v_lshrrev_b32 v4, 7, v1 // (op 637) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 637) 
v_and_b32 v5, 15, v1 // (op 637) 
s_and_b32 s15, s18, 4294967294 // (op 637) 
v_lshl_add_u32 v7, v4, 4, v5 // (op 637) 


s_add_u32 s6, s0, s15 // (op 637) 

v_and_b32 v4, 48, v1 // (op 637) 
s_mul_i32 s15, s16, s50 // (op 637) 
s_add_u32 s15, s15, s6 // (op 637) 

s_add_u32 s6, s15, s9 // (op 637) 

s_mul_hi_u32 s15, s6, s13 // (op 637) 
s_sub_u32 s7, s6, s15 // (op 637) 
s_lshr_b32 s8, s7, 1 // (op 637) 

s_add_u32 s7, s8, s15 // (op 637) 


s_lshr_b32 s8, s7, s11 // (op 637) 

s_and_b32 s7, s11, 2147483648 // (op 637) 
s_cmp_eq_i32 s7, 0 // (op 637) 
s_mov_b32 s15, scc // (op 637) 

s_cmp_eq_i32 s15, 0 // (op 637) 
s_cselect_b32 s7, s6, s8 // (op 637) 


s_mul_hi_u32 s8, s7, s12 // (op 637) 
s_sub_u32 s15, s7, s8 // (op 637) 
s_lshr_b32 s42, s15, 1 // (op 637) 

s_add_u32 s15, s42, s8 // (op 637) 


s_lshr_b32 s8, s15, s10 // (op 637) 

s_and_b32 s15, s10, 2147483648 // (op 637) 
s_cmp_eq_i32 s15, 0 // (op 637) 
s_mov_b32 s42, scc // (op 637) 

s_cmp_eq_i32 s42, 0 // (op 637) 
s_cselect_b32 s15, s7, s8 // (op 637) 


s_mul_i32 s8, s7, s18 // (op 637) 


v_lshl_add_u32 v5, s15, 5, v7 // (op 637) 


s_sub_u32 s7, s6, s8 // (op 637) 


v_lshl_add_u32 v7, s7, 7, v4 // (op 637) 


v_mul_lo_u32 v4, v5, s24 // (op 637) 
v_add_u32 v4, v4, v7 // (op 637) 


s_mul_i32 s19, s24, s29 // (op 639) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 639) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 639) least significant
s_mul_hi_u32 s6, s24, s28 // (op 639) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 639) most significant: sum
s_add_u32 s19, s19, s6 // (op 639) most significant: sum


s_mov_b32 s60, s26 // (op 639) reg expression
s_mov_b32 s61, s27 // (op 639) reg expression
s_mov_b32 s62, s18 // (op 639) reg expression
s_mov_b32 s63, 131072 // (op 639) call()

v_lshrrev_b32 v5, 6, v1 // (op 731) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 731) 
v_and_b32 v7, 1, v5 // (op 731) 

s_and_b32 s15, s18, 4294967294 // (op 731) 
v_and_b32 v5, 15, v1 // (op 731) 
s_add_u32 s6, s0, s15 // (op 731) 

v_lshl_add_u32 v8, v7, 4, v5 // (op 731) 


v_and_b32 v5, 63, v1 // (op 731) 
s_mul_i32 s15, s16, s50 // (op 731) 
s_add_u32 s15, s15, s6 // (op 731) 

v_lshrrev_b32 v7, 4, v5 // (op 731) 

s_add_u32 s6, s15, s9 // (op 731) 

v_lshlrev_b32 v5, 5, v7 // (op 731) 

s_mul_hi_u32 s15, s6, s13 // (op 731) 
s_sub_u32 s7, s6, s15 // (op 731) 
s_lshr_b32 s8, s7, 1 // (op 731) 

s_add_u32 s7, s8, s15 // (op 731) 


s_lshr_b32 s8, s7, s11 // (op 731) 

s_and_b32 s7, s11, 2147483648 // (op 731) 
s_cmp_eq_i32 s7, 0 // (op 731) 
s_mov_b32 s15, scc // (op 731) 

s_cmp_eq_i32 s15, 0 // (op 731) 
s_cselect_b32 s7, s6, s8 // (op 731) 


s_mul_hi_u32 s8, s7, s12 // (op 731) 
s_sub_u32 s15, s7, s8 // (op 731) 
s_lshr_b32 s42, s15, 1 // (op 731) 

s_add_u32 s15, s42, s8 // (op 731) 


s_lshr_b32 s8, s15, s10 // (op 731) 

s_and_b32 s15, s10, 2147483648 // (op 731) 
s_cmp_eq_i32 s15, 0 // (op 731) 
s_mov_b32 s42, scc // (op 731) 

s_cmp_eq_i32 s42, 0 // (op 731) 
s_cselect_b32 s15, s7, s8 // (op 731) 


s_mul_i32 s8, s15, s14 // (op 731) 

s_sub_u32 s15, s7, s8 // (op 731) 

s_mul_i32 s8, s7, s18 // (op 731) 


v_lshl_add_u32 v7, s15, 5, v8 // (op 731) 


s_sub_u32 s7, s6, s8 // (op 731) 


v_mul_lo_u32 v9, v7, s31 // (op 731) most significant: low of low * high
v_mul_hi_u32 v10, v7, s30 // (op 731) most significant: high of low * low
v_mul_lo_u32 v8, v7, s30 // (op 731) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 731) most significant: sum


v_lshl_add_u32 v7, s7, 7, v5 // (op 731) 


v_mov_b32 v11, 0 // (op 731) convert
v_mov_b32 v10, v7 // (op 731) convert

v_add_co_u32 v12, s[18:19], v10, v8 // (op 731) least significant half
v_addc_co_u32 v13, s[18:19], v11, v9, s[18:19] // (op 731) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 731) 


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 731) 

v_mov_b32 v5, v10 // (op 731) convert

s_mul_i32 s19, s30, s23 // (op 733) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 733) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 733) least significant
s_mul_hi_u32 s6, s30, s22 // (op 733) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 733) most significant: sum
s_add_u32 s19, s19, s6 // (op 733) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 733) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 733) 

s_mov_b32 s64, s32 // (op 733) reg expression
s_mov_b32 s65, s33 // (op 733) reg expression
s_mov_b32 s66, s18 // (op 733) reg expression
s_mov_b32 s67, 131072 // (op 733) call()

v_lshrrev_b32 v7, 7, v1 // (op 819) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 819) 
v_and_b32 v8, 15, v1 // (op 819) 
s_and_b32 s15, s18, 4294967294 // (op 819) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 819) 


s_add_u32 s6, s0, s15 // (op 819) 

v_and_b32 v7, 63, v1 // (op 819) 
v_lshrrev_b32 v8, 4, v7 // (op 819) 

s_mul_i32 s15, s16, s50 // (op 819) 
s_add_u32 s15, s15, s6 // (op 819) 

s_add_u32 s6, s15, s9 // (op 819) 

s_mul_hi_u32 s15, s6, s13 // (op 819) 
s_sub_u32 s7, s6, s15 // (op 819) 
s_lshr_b32 s8, s7, 1 // (op 819) 

s_add_u32 s7, s8, s15 // (op 819) 


s_lshr_b32 s8, s7, s11 // (op 819) 

s_and_b32 s7, s11, 2147483648 // (op 819) 
s_cmp_eq_i32 s7, 0 // (op 819) 
s_mov_b32 s15, scc // (op 819) 

s_cmp_eq_i32 s15, 0 // (op 819) 
s_cselect_b32 s7, s6, s8 // (op 819) 


s_mul_hi_u32 s8, s7, s12 // (op 819) 
s_sub_u32 s15, s7, s8 // (op 819) 
s_lshr_b32 s42, s15, 1 // (op 819) 

s_add_u32 s15, s42, s8 // (op 819) 


s_lshr_b32 s8, s15, s10 // (op 819) 

s_and_b32 s15, s10, 2147483648 // (op 819) 
s_cmp_eq_i32 s15, 0 // (op 819) 
s_mov_b32 s42, scc // (op 819) 

s_cmp_eq_i32 s42, 0 // (op 819) 
s_cselect_b32 s15, s7, s8 // (op 819) 


s_mul_i32 s8, s7, s18 // (op 819) 


v_lshl_add_u32 v7, s15, 5, v9 // (op 819) 


s_sub_u32 s7, s6, s8 // (op 819) 


v_lshl_add_u32 v9, s7, 2, v8 // (op 819) 


v_mul_lo_u32 v8, v7, s34 // (op 819) 
v_add_u32 v8, v8, v9 // (op 819) 


s_mul_i32 s19, s34, s29 // (op 821) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 821) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 821) least significant
s_mul_hi_u32 s6, s34, s28 // (op 821) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 821) most significant: sum
s_add_u32 s19, s19, s6 // (op 821) most significant: sum


s_mov_b32 s68, s36 // (op 821) reg expression
s_mov_b32 s69, s37 // (op 821) reg expression
s_mov_b32 s70, s18 // (op 821) reg expression
s_mov_b32 s71, 131072 // (op 821) call()

v_lshrrev_b32 v7, 6, v1 // (op 907) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 907) 
v_and_b32 v9, 1, v7 // (op 907) 

s_and_b32 s15, s18, 4294967294 // (op 907) 
v_and_b32 v7, 15, v1 // (op 907) 
s_add_u32 s6, s0, s15 // (op 907) 

v_lshl_add_u32 v10, v9, 4, v7 // (op 907) 


v_and_b32 v7, 63, v1 // (op 907) 
s_mul_i32 s15, s16, s50 // (op 907) 
s_add_u32 s15, s15, s6 // (op 907) 

v_lshrrev_b32 v9, 4, v7 // (op 907) 

s_add_u32 s6, s15, s9 // (op 907) 

s_mul_hi_u32 s15, s6, s13 // (op 907) 
s_sub_u32 s7, s6, s15 // (op 907) 
s_lshr_b32 s8, s7, 1 // (op 907) 

s_add_u32 s7, s8, s15 // (op 907) 


s_lshr_b32 s8, s7, s11 // (op 907) 

s_and_b32 s7, s11, 2147483648 // (op 907) 
s_cmp_eq_i32 s7, 0 // (op 907) 
s_mov_b32 s15, scc // (op 907) 

s_cmp_eq_i32 s15, 0 // (op 907) 
s_cselect_b32 s7, s6, s8 // (op 907) 


s_mul_hi_u32 s8, s7, s12 // (op 907) 
s_sub_u32 s15, s7, s8 // (op 907) 
s_lshr_b32 s42, s15, 1 // (op 907) 

s_add_u32 s15, s42, s8 // (op 907) 


s_lshr_b32 s8, s15, s10 // (op 907) 

s_and_b32 s15, s10, 2147483648 // (op 907) 
s_cmp_eq_i32 s15, 0 // (op 907) 
s_mov_b32 s42, scc // (op 907) 

s_cmp_eq_i32 s42, 0 // (op 907) 
s_cselect_b32 s15, s7, s8 // (op 907) 


s_mul_i32 s8, s15, s14 // (op 907) 

s_sub_u32 s15, s7, s8 // (op 907) 

v_lshl_add_u32 v7, s15, 5, v10 // (op 907) 


s_mul_i32 s8, s7, s18 // (op 907) 


v_mul_lo_u32 v10, v7, s38 // (op 907) 

s_sub_u32 s7, s6, s8 // (op 907) 


v_lshl_add_u32 v7, s7, 2, v9 // (op 907) 


v_add_u32 v9, v7, v10 // (op 907) 


s_mul_i32 s19, s38, s23 // (op 909) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 909) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 909) least significant
s_mul_hi_u32 s6, s38, s22 // (op 909) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 909) most significant: sum
s_add_u32 s19, s19, s6 // (op 909) most significant: sum


s_mov_b32 s72, s40 // (op 909) reg expression
s_mov_b32 s73, s41 // (op 909) reg expression
s_mov_b32 s74, s18 // (op 909) reg expression
s_mov_b32 s75, 131072 // (op 909) call()

s_ashr_i64 s[18:19], s[2:3], 7 // (op 393) 
s_and_b32 s15, s18, 4294967294 // (op 393) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 392) 
s_mov_b32 s6, s18 // (op 392) convert

s_cmp_lt_u32 s15, s6 // (op 392) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_392 // (op 392) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_392 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_392:


s_add_u32 s15, s15, 1 // (op 394) 
v_mov_b32 v7, v4 // (op 398) 

buffer_load_dwordx4 v[10:13], v7, s[60:63], 0 offen // (op 398) Load value
buffer_load_dwordx4 v[14:17], v7, s[60:63], 0 offen offset:64 // (op 398) Load value

v_add_u32 v4, 128, v4 // (op 627) 


v_mov_b32 v7, v5 // (op 400) 

buffer_load_dwordx4 v[18:21], v7, s[64:67], 0 offen // (op 400) Load value

v_add_u32 v5, 64, v5 // (op 723) 


v_mov_b32 v7, v8 // (op 402) 

buffer_load_ubyte v22, v7, s[68:71], 0 offen offset: 0 // (op 402) Load value

v_add_u32 v8, 4, v8 // (op 811) 


v_mov_b32 v7, v9 // (op 404) 

buffer_load_ubyte v23, v7, s[72:75], 0 offen offset: 0 // (op 404) Load value

v_add_u32 v9, 4, v9 // (op 899) 
s_waitcnt vmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:17], v[18:21], a[0:3], v22, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 405) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 392) 
s_mov_b32 s6, s18 // (op 392) convert

s_cmp_lt_u32 s15, s6 // (op 392) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_392 // (op 392) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_392 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_392:





v_cmp_eq_f32 vcc, 0.00000, v0 // (op 177) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_177 // (op 177) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_177
s_nop 1
v_accvgpr_read v8, a0 // (op 185) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 185) 
v_accvgpr_read v10, a2 // (op 185) 
v_accvgpr_read v11, a3 // (op 185) 
v_mul_f32 v12, v3, v8 // (op 185) 
v_mul_f32 v13, v3, v9 // (op 185) 
v_mul_f32 v14, v3, v10 // (op 185) 
v_mul_f32 v15, v3, v11 // (op 185) 

v_lshrrev_b32 v4, 7, v1 // (op 1061) 
s_mul_i32 s15, s16, s50 // (op 1061) 
s_add_u32 s15, s15, s0 // (op 1061) 
v_and_b32 v5, 63, v1 // (op 1061) 
s_add_u32 s6, s15, s9 // (op 1061) 

v_lshrrev_b32 v7, 4, v5 // (op 1061) 
s_mul_hi_u32 s15, s6, s13 // (op 1061) 
v_lshlrev_b32 v8, 2, v7 // (op 1061) 

s_sub_u32 s7, s6, s15 // (op 1061) 
v_lshl_add_u32 v7, v4, 4, v8 // (op 1061) 


s_lshr_b32 s8, s7, 1 // (op 1061) 

v_lshrrev_b32 v4, 2, v5 // (op 1061) 

s_add_u32 s7, s8, s15 // (op 1061) 


v_and_b32 v5, 3, v4 // (op 1061) 

s_lshr_b32 s8, s7, s11 // (op 1061) 

v_and_b32 v4, 3, v1 // (op 1061) 
s_and_b32 s7, s11, 2147483648 // (op 1061) 
v_lshl_add_u32 v8, v5, 2, v4 // (op 1061) 


s_cmp_eq_i32 s7, 0 // (op 1061) 
s_mov_b32 s15, scc // (op 1061) 

v_lshrrev_b32 v4, 6, v1 // (op 1061) 
s_cmp_eq_i32 s15, 0 // (op 1061) 
s_cselect_b32 s7, s6, s8 // (op 1061) 



v_and_b32 v5, 1, v4 // (op 1061) 

s_mul_hi_u32 s6, s7, s12 // (op 1061) 
v_lshl_add_u32 v4, v5, 4, v8 // (op 1061) 


s_sub_u32 s8, s7, s6 // (op 1061) 
s_lshr_b32 s15, s8, 1 // (op 1061) 

s_add_u32 s8, s15, s6 // (op 1061) 


s_lshr_b32 s6, s8, s10 // (op 1061) 

s_and_b32 s8, s10, 2147483648 // (op 1061) 
s_cmp_eq_i32 s8, 0 // (op 1061) 
s_mov_b32 s15, scc // (op 1061) 

s_cmp_eq_i32 s15, 0 // (op 1061) 
s_cselect_b32 s8, s7, s6 // (op 1061) 


s_mul_i32 s6, s8, s14 // (op 1061) 
v_lshl_add_u32 v5, s8, 5, v7 // (op 1061) 


s_sub_u32 s8, s7, s6 // (op 1061) 


v_lshl_add_u32 v7, s8, 5, v4 // (op 1061) 


v_mul_lo_u32 v4, v7, s46 // (op 1061) 

v_add_lshl_u32 v7, v5, v4, 2 // (op 1061) 


s_mul_i32 s19, s46, s23 // (op 1063) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1063) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1063) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1063) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1063) most significant: sum
s_add_u32 s19, s19, s6 // (op 1063) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1063) 

s_mov_b32 s76, s48 // (op 1063) reg expression
s_mov_b32 s77, s49 // (op 1063) reg expression
s_mov_b32 s78, s6 // (op 1063) reg expression
s_mov_b32 s79, 131072 // (op 1063) call()




buffer_store_dwordx4 v[12:15], v7, s[76:79], 0 offen // (op 187) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_177 // (op 177) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_177
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_177:



buffer_load_dwordx4 v[8:11], v6, s[56:59], 0 offen // (op 200) Load value
v_accvgpr_read v12, a0 // (op 197) 
v_accvgpr_read v13, a1 // (op 197) 
v_accvgpr_read v14, a2 // (op 197) 
v_accvgpr_read v15, a3 // (op 197) 
v_mul_f32 v16, v3, v12 // (op 197) 
v_mul_f32 v17, v3, v13 // (op 197) 
v_mul_f32 v18, v3, v14 // (op 197) 
v_mul_f32 v19, v3, v15 // (op 197) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v0, v8 // (op 201) 
v_mul_f32 v13, v0, v9 // (op 201) 
v_mul_f32 v14, v0, v10 // (op 201) 
v_mul_f32 v15, v0, v11 // (op 201) 

v_add_f32 v8, v12, v16 // (op 198) 
v_add_f32 v9, v13, v17 // (op 198) 
v_add_f32 v10, v14, v18 // (op 198) 
v_add_f32 v11, v15, v19 // (op 198) 


v_lshrrev_b32 v7, 7, v1 // (op 1078) 
s_mul_i32 s15, s16, s50 // (op 1078) 
s_add_u32 s15, s15, s0 // (op 1078) 
v_and_b32 v4, 63, v1 // (op 1078) 
s_add_u32 s6, s15, s9 // (op 1078) 

v_lshrrev_b32 v5, 4, v4 // (op 1078) 
s_mul_hi_u32 s15, s6, s13 // (op 1078) 
v_lshlrev_b32 v12, 2, v5 // (op 1078) 

s_sub_u32 s7, s6, s15 // (op 1078) 
v_lshl_add_u32 v5, v7, 4, v12 // (op 1078) 


s_lshr_b32 s8, s7, 1 // (op 1078) 

v_lshrrev_b32 v7, 2, v4 // (op 1078) 

s_add_u32 s7, s8, s15 // (op 1078) 


v_and_b32 v4, 3, v7 // (op 1078) 

s_lshr_b32 s8, s7, s11 // (op 1078) 

v_and_b32 v7, 3, v1 // (op 1078) 
s_and_b32 s7, s11, 2147483648 // (op 1078) 
v_lshl_add_u32 v12, v4, 2, v7 // (op 1078) 


s_cmp_eq_i32 s7, 0 // (op 1078) 
s_mov_b32 s15, scc // (op 1078) 

v_lshrrev_b32 v4, 6, v1 // (op 1078) 
s_cmp_eq_i32 s15, 0 // (op 1078) 
s_cselect_b32 s7, s6, s8 // (op 1078) 



v_and_b32 v7, 1, v4 // (op 1078) 

s_mul_hi_u32 s6, s7, s12 // (op 1078) 
v_lshl_add_u32 v4, v7, 4, v12 // (op 1078) 


s_sub_u32 s8, s7, s6 // (op 1078) 
s_lshr_b32 s15, s8, 1 // (op 1078) 

s_add_u32 s8, s15, s6 // (op 1078) 


s_lshr_b32 s6, s8, s10 // (op 1078) 

s_and_b32 s8, s10, 2147483648 // (op 1078) 
s_cmp_eq_i32 s8, 0 // (op 1078) 
s_mov_b32 s15, scc // (op 1078) 

s_cmp_eq_i32 s15, 0 // (op 1078) 
s_cselect_b32 s8, s7, s6 // (op 1078) 


s_mul_i32 s6, s8, s14 // (op 1078) 
v_lshl_add_u32 v7, s8, 5, v5 // (op 1078) 


s_sub_u32 s8, s7, s6 // (op 1078) 


v_lshl_add_u32 v5, s8, 5, v4 // (op 1078) 


v_mul_lo_u32 v4, v5, s46 // (op 1078) 

v_add_lshl_u32 v5, v7, v4, 2 // (op 1078) 


s_mul_i32 s19, s46, s23 // (op 1080) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1080) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1080) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1080) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1080) most significant: sum
s_add_u32 s19, s19, s6 // (op 1080) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1080) 

s_mov_b32 s76, s48 // (op 1080) reg expression
s_mov_b32 s77, s49 // (op 1080) reg expression
s_mov_b32 s78, s6 // (op 1080) reg expression
s_mov_b32 s79, 131072 // (op 1080) call()




buffer_store_dwordx4 v[8:11], v5, s[76:79], 0 offen // (op 199) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_177:

s_add_u32 s0, s0, s1 // (op 386) 
s_mul_i32 s15, s50, s16 // (op 384) 
s_cmp_gt_u32 s51, s15 // (op 384) 
s_mov_b32 s6, scc // (op 384) 
s_sub_u32 s7, s51, s15 // (op 384) 

s_cmp_eq_i32 s6, 0 // (op 384) 
s_cselect_b32 s8, 0, s7 // (op 384) 


s_cmp_lt_u32 s50, s8 // (op 384) 
s_mov_b32 s15, scc // (op 384) 
s_cmp_eq_i32 s15, 0 // (op 384) 
s_cselect_b32 s6, s8, s50 // (op 384) 


s_cmp_lt_u32 s0, s6 // (op 384) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_384 // (op 384) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopTop_DPStreamTileLoop_384 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_384:


































s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2, .LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2_end-RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 40
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 0 // lds bytes
  .amdhsa_accum_offset 36
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
  - .name: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 82
    .vgpr_count: 36
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

