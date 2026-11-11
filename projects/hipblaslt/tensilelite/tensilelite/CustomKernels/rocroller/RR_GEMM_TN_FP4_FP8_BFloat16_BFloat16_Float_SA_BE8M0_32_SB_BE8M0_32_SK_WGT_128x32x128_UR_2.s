// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2,@function
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 3661) 
v_and_b32 v3, 15, v1 // (op 3661) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3661) 


v_and_b32 v0, 63, v1 // (op 3661) 
v_lshrrev_b32 v3, 4, v0 // (op 3661) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 3661) 


v_lshrrev_b32 v3, 7, v1 // (op 3761) 
v_and_b32 v4, 15, v1 // (op 3761) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 3761) 


v_and_b32 v3, 63, v1 // (op 3761) 
v_lshrrev_b32 v4, 4, v3 // (op 3761) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 3761) 


v_lshrrev_b32 v4, 7, v1 // (op 3832) 
v_and_b32 v5, 15, v1 // (op 3832) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 3832) 


v_and_b32 v4, 63, v1 // (op 3832) 
v_lshrrev_b32 v5, 4, v4 // (op 3832) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 3832) 


v_lshrrev_b32 v5, 7, v1 // (op 3861) 
v_and_b32 v6, 15, v1 // (op 3861) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 3861) 


v_and_b32 v5, 63, v1 // (op 3861) 
v_lshrrev_b32 v6, 4, v5 // (op 3861) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 3861) 


v_mov_b32 v6, s6 // (op 2) Move value
v_mov_b32 v7, s7 // (op 31) Move value


s_mov_b32 s0, 0 // (op 350) call()
s_mov_b32 s1, 0 // (op 351) call()
s_mov_b32 s6, 0 // (op 352) call()
s_mov_b32 s7, 0 // (op 353) call()
s_mov_b32 s18, 0 // (op 354) call()
s_mov_b32 s19, 0 // (op 355) call()
s_mov_b32 s0, 0 // (op 250) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 249) 
s_cmp_gt_u32 s9, s52 // (op 249) 
s_mov_b32 s53, scc // (op 249) 
s_sub_u32 s54, s9, s52 // (op 249) 

s_cmp_eq_i32 s53, 0 // (op 249) 
s_cselect_b32 s52, 0, s54 // (op 249) 


s_cmp_lt_u32 s8, s52 // (op 249) 
s_mov_b32 s53, scc // (op 249) 
s_cmp_eq_i32 s53, 0 // (op 249) 
s_cselect_b32 s54, s52, s8 // (op 249) 


s_cmp_lt_u32 s0, s54 // (op 249) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 // (op 249) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_249:
v_lshrrev_b32 v8, 7, v1 // (op 3576) 
s_mul_i32 s52, s16, s8 // (op 3576) 
s_add_u32 s52, s52, s0 // (op 3576) 
v_and_b32 v9, 63, v1 // (op 3576) 
s_mul_hi_u32 s53, s52, s13 // (op 3576) 
v_lshrrev_b32 v10, 4, v9 // (op 3576) 
s_sub_u32 s54, s52, s53 // (op 3576) 
v_lshlrev_b32 v11, 2, v10 // (op 3576) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 3576) 


s_lshr_b32 s55, s54, 1 // (op 3576) 

s_add_u32 s54, s55, s53 // (op 3576) 


v_lshrrev_b32 v8, 2, v9 // (op 3576) 

s_lshr_b32 s53, s54, s11 // (op 3576) 

v_and_b32 v9, 3, v8 // (op 3576) 

s_and_b32 s54, s11, 2147483648 // (op 3576) 
v_and_b32 v8, 3, v1 // (op 3576) 
s_cmp_eq_i32 s54, 0 // (op 3576) 
s_mov_b32 s55, scc // (op 3576) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 3576) 


s_cmp_eq_i32 s55, 0 // (op 3576) 
s_cselect_b32 s54, s52, s53 // (op 3576) 



v_lshrrev_b32 v8, 6, v1 // (op 3576) 
s_mul_hi_u32 s52, s54, s12 // (op 3576) 
v_and_b32 v9, 1, v8 // (op 3576) 

s_sub_u32 s53, s54, s52 // (op 3576) 
v_lshl_add_u32 v8, v9, 4, v11 // (op 3576) 


s_lshr_b32 s55, s53, 1 // (op 3576) 

s_add_u32 s53, s55, s52 // (op 3576) 


s_lshr_b32 s52, s53, s10 // (op 3576) 

s_and_b32 s53, s10, 2147483648 // (op 3576) 
s_cmp_eq_i32 s53, 0 // (op 3576) 
s_mov_b32 s55, scc // (op 3576) 

s_cmp_eq_i32 s55, 0 // (op 3576) 
s_cselect_b32 s53, s54, s52 // (op 3576) 


s_mul_i32 s52, s53, s14 // (op 3576) 
v_lshl_add_u32 v9, s53, 7, v10 // (op 3576) 


s_sub_u32 s53, s54, s52 // (op 3576) 


v_lshl_add_u32 v10, s53, 5, v8 // (op 3576) 


v_mul_lo_u32 v8, v10, s4 // (op 3576) 

v_add_lshl_u32 v10, v9, v8, 1 // (op 3576) 


s_mul_i32 s53, s4, s23 // (op 3578) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 3578) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 3578) least significant
s_mul_hi_u32 s55, s4, s22 // (op 3578) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3578) most significant: sum
s_add_u32 s53, s53, s55 // (op 3578) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 3578) 

s_mov_b32 s56, s20 // (op 3578) reg expression
s_mov_b32 s57, s21 // (op 3578) reg expression
s_mov_b32 s58, s54 // (op 3578) reg expression
s_mov_b32 s59, 131072 // (op 3578) call()

v_lshrrev_b32 v8, 7, v1 // (op 3678) 
s_mul_i32 s52, s16, s8 // (op 3678) 
s_add_u32 s52, s52, s0 // (op 3678) 
v_and_b32 v9, 63, v1 // (op 3678) 
s_mul_hi_u32 s53, s52, s13 // (op 3678) 
v_lshrrev_b32 v11, 4, v9 // (op 3678) 
s_sub_u32 s54, s52, s53 // (op 3678) 
v_lshlrev_b32 v12, 2, v11 // (op 3678) 

s_lshr_b32 s55, s54, 1 // (op 3678) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 3678) 


s_add_u32 s54, s55, s53 // (op 3678) 


v_lshrrev_b32 v8, 2, v9 // (op 3678) 

s_lshr_b32 s53, s54, s11 // (op 3678) 

v_and_b32 v9, 3, v8 // (op 3678) 

s_and_b32 s54, s11, 2147483648 // (op 3678) 
v_and_b32 v8, 3, v1 // (op 3678) 
s_cmp_eq_i32 s54, 0 // (op 3678) 
s_mov_b32 s55, scc // (op 3678) 

v_lshl_add_u32 v12, v9, 2, v8 // (op 3678) 


s_cmp_eq_i32 s55, 0 // (op 3678) 
s_cselect_b32 s54, s52, s53 // (op 3678) 



v_lshrrev_b32 v8, 6, v1 // (op 3678) 
s_mul_hi_u32 s55, s54, s12 // (op 3678) 
v_and_b32 v9, 1, v8 // (op 3678) 

s_sub_u32 s52, s54, s55 // (op 3678) 
v_lshl_add_u32 v8, v9, 4, v12 // (op 3678) 


s_lshr_b32 s53, s52, 1 // (op 3678) 

s_add_u32 s52, s53, s55 // (op 3678) 


s_lshr_b32 s53, s52, s10 // (op 3678) 

s_and_b32 s52, s10, 2147483648 // (op 3678) 
s_cmp_eq_i32 s52, 0 // (op 3678) 
s_mov_b32 s55, scc // (op 3678) 

s_cmp_eq_i32 s55, 0 // (op 3678) 
s_cselect_b32 s52, s54, s53 // (op 3678) 


s_mul_i32 s53, s52, s14 // (op 3678) 
v_lshl_add_u32 v9, s52, 7, v11 // (op 3678) 


s_sub_u32 s52, s54, s53 // (op 3678) 


v_lshl_add_u32 v11, s52, 5, v8 // (op 3678) 


v_mul_lo_u32 v8, v11, s24 // (op 3678) 

v_add_lshl_u32 v11, v9, v8, 1 // (op 3678) 


s_mul_i32 s53, s24, s23 // (op 3680) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 3680) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 3680) least significant
s_mul_hi_u32 s55, s24, s22 // (op 3680) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3680) most significant: sum
s_add_u32 s53, s53, s55 // (op 3680) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 3680) 

s_mov_b32 s60, s26 // (op 3680) reg expression
s_mov_b32 s61, s27 // (op 3680) reg expression
s_mov_b32 s62, s54 // (op 3680) reg expression
s_mov_b32 s63, 131072 // (op 3680) call()

v_lshrrev_b32 v8, 5, v1 // (op 3778) 
v_lshlrev_b32 v9, 2, v8 // (op 3778) 

v_lshl_add_u32 v8, s16, 7, v9 // (op 3778) 

v_and_b32 v9, -32, v1 // (op 3778) 
v_sub_u32 v12, v1, v9 // (op 3778) 

v_lshl_add_u32 v9, v8, 5, v12 // (op 3778) 


v_lshlrev_b32 v8, 2, v9 // (op 3778) 

s_lshl_b32 s52, s15, 14 // (op 3780) 
s_mov_b32 s64, s28 // (op 3780) reg expression
s_mov_b32 s65, s29 // (op 3780) reg expression
s_mov_b32 s66, s52 // (op 3780) reg expression
s_mov_b32 s67, 131072 // (op 3780) call()

s_mul_i32 s6, s8, s16 // (op 254) 
s_add_u32 s6, s6, s0 // (op 254) 
s_mul_hi_u32 s52, s6, s13 // (op 255) 
s_sub_u32 s53, s6, s52 // (op 255) 
s_lshr_b32 s54, s53, 1 // (op 255) 

s_add_u32 s53, s54, s52 // (op 255) 


s_lshr_b32 s52, s53, s11 // (op 255) 

s_and_b32 s53, s11, 2147483648 // (op 255) 
s_cmp_eq_i32 s53, 0 // (op 255) 
s_mov_b32 s54, scc // (op 255) 

s_cmp_eq_i32 s54, 0 // (op 255) 
s_cselect_b32 s53, s6, s52 // (op 255) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 255) 
s_mul_i32 s52, s53, s54 // (op 255) 


s_sub_u32 s18, s6, s52 // (op 255) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 256) 
s_cmp_gt_u32 s52, s18 // (op 256) 
s_mov_b32 s54, scc // (op 256) 
s_sub_u32 s55, s52, s18 // (op 256) 

s_cmp_eq_i32 s54, 0 // (op 256) 
s_cselect_b32 s52, 0, s55 // (op 256) 


s_cmp_gt_u32 s8, s0 // (op 256) 
s_mov_b32 s53, scc // (op 256) 
s_sub_u32 s54, s8, s0 // (op 256) 
s_cmp_eq_i32 s53, 0 // (op 256) 
s_cselect_b32 s55, 0, s54 // (op 256) 


s_cmp_lt_u32 s52, s55 // (op 256) 
s_mov_b32 s53, scc // (op 256) 
s_cmp_eq_i32 s53, 0 // (op 256) 
s_cselect_b32 s54, s55, s52 // (op 256) 



s_cmp_gt_u32 s9, s6 // (op 256) 
s_mov_b32 s55, scc // (op 256) 
s_sub_u32 s52, s9, s6 // (op 256) 
s_cmp_eq_i32 s55, 0 // (op 256) 
s_cselect_b32 s53, 0, s52 // (op 256) 


s_cmp_lt_u32 s54, s53 // (op 256) 
s_mov_b32 s52, scc // (op 256) 
s_cmp_eq_i32 s52, 0 // (op 256) 
s_cselect_b32 s7, s53, s54 // (op 256) 



s_add_u32 s52, s18, s7 // (op 257) 
s_sub_u32 s19, s52, 1 // (op 257) 

v_accvgpr_write a0, 0.00000 // (op 53) call()
v_accvgpr_write a1, 0.00000 // (op 53) call()
v_accvgpr_write a2, 0.00000 // (op 53) call()
v_accvgpr_write a3, 0.00000 // (op 53) call()
v_accvgpr_write a4, 0.00000 // (op 1253) call()
v_accvgpr_write a5, 0.00000 // (op 1253) call()
v_accvgpr_write a6, 0.00000 // (op 1253) call()
v_accvgpr_write a7, 0.00000 // (op 1253) call()
v_accvgpr_write a8, 0.00000 // (op 1375) call()
v_accvgpr_write a9, 0.00000 // (op 1375) call()
v_accvgpr_write a10, 0.00000 // (op 1375) call()
v_accvgpr_write a11, 0.00000 // (op 1375) call()
v_accvgpr_write a12, 0.00000 // (op 1497) call()
v_accvgpr_write a13, 0.00000 // (op 1497) call()
v_accvgpr_write a14, 0.00000 // (op 1497) call()
v_accvgpr_write a15, 0.00000 // (op 1497) call()
v_lshrrev_b32 v9, 7, v1 // (op 3143) 
s_mul_i32 s52, s16, s8 // (op 3143) 
s_add_u32 s52, s52, s0 // (op 3143) 
v_and_b32 v12, 15, v1 // (op 3143) 
s_mul_hi_u32 s53, s52, s13 // (op 3143) 
v_lshl_add_u32 v13, v9, 4, v12 // (op 3143) 


s_sub_u32 s54, s52, s53 // (op 3143) 
v_and_b32 v9, 63, v1 // (op 3143) 
s_lshr_b32 s55, s54, 1 // (op 3143) 

v_lshrrev_b32 v12, 4, v9 // (op 3143) 

s_add_u32 s54, s55, s53 // (op 3143) 


v_lshlrev_b32 v9, 5, v12 // (op 3143) 

s_lshr_b32 s53, s54, s11 // (op 3143) 

s_and_b32 s54, s11, 2147483648 // (op 3143) 
s_cmp_eq_i32 s54, 0 // (op 3143) 
s_mov_b32 s55, scc // (op 3143) 

s_cmp_eq_i32 s55, 0 // (op 3143) 
s_cselect_b32 s54, s52, s53 // (op 3143) 


s_mul_hi_u32 s53, s54, s12 // (op 3143) 
s_sub_u32 s55, s54, s53 // (op 3143) 
s_lshr_b32 s68, s55, 1 // (op 3143) 

s_add_u32 s55, s68, s53 // (op 3143) 


s_lshr_b32 s53, s55, s10 // (op 3143) 

s_and_b32 s55, s10, 2147483648 // (op 3143) 
s_cmp_eq_i32 s55, 0 // (op 3143) 
s_mov_b32 s68, scc // (op 3143) 

s_cmp_eq_i32 s68, 0 // (op 3143) 
s_cselect_b32 s55, s54, s53 // (op 3143) 


v_lshl_add_u32 v12, s55, 7, v13 // (op 3143) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 3143) 
s_mul_i32 s53, s54, s68 // (op 3143) 


s_sub_u32 s54, s52, s53 // (op 3143) 


v_lshl_add_u32 v13, s54, 7, v9 // (op 3143) 


v_mov_b32 v15, 0 // (op 3143) convert
v_mov_b32 v14, v13 // (op 3143) convert

v_mul_lo_u32 v17, v12, s31 // (op 3143) most significant: low of low * high
v_mul_hi_u32 v9, v12, s30 // (op 3143) most significant: high of low * low
v_mul_lo_u32 v16, v12, s30 // (op 3143) least significant: low of low * low
v_add_u32 v17, v17, v9 // (op 3143) most significant: sum

v_add_co_u32 v16, s[52:53], v16, v14 // (op 3143) least significant half
v_addc_co_u32 v17, s[52:53], v17, v15, s[52:53] // (op 3143) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 3143) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3143) 

v_mov_b32 v9, v14 // (op 3143) convert

s_mul_i32 s53, s30, s35 // (op 3145) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3145) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3145) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3145) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3145) most significant: sum
s_add_u32 s53, s53, s55 // (op 3145) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3145) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 3145) 

s_mov_b32 s68, s32 // (op 3145) reg expression
s_mov_b32 s69, s33 // (op 3145) reg expression
s_mov_b32 s70, s52 // (op 3145) reg expression
s_mov_b32 s71, 131072 // (op 3145) call()

v_lshrrev_b32 v12, 7, v1 // (op 3164) 
s_mul_i32 s52, s16, s8 // (op 3164) 
s_add_u32 s52, s52, s0 // (op 3164) 
v_and_b32 v13, 15, v1 // (op 3164) 
s_mul_hi_u32 s53, s52, s13 // (op 3164) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 3164) 


s_sub_u32 s54, s52, s53 // (op 3164) 
v_and_b32 v12, 63, v1 // (op 3164) 
s_lshr_b32 s55, s54, 1 // (op 3164) 

v_lshrrev_b32 v13, 4, v12 // (op 3164) 

s_add_u32 s54, s55, s53 // (op 3164) 


v_lshlrev_b32 v12, 5, v13 // (op 3164) 

s_lshr_b32 s53, s54, s11 // (op 3164) 

s_and_b32 s54, s11, 2147483648 // (op 3164) 
s_cmp_eq_i32 s54, 0 // (op 3164) 
s_mov_b32 s55, scc // (op 3164) 

s_cmp_eq_i32 s55, 0 // (op 3164) 
s_cselect_b32 s54, s52, s53 // (op 3164) 


s_mul_hi_u32 s53, s54, s12 // (op 3164) 
s_sub_u32 s55, s54, s53 // (op 3164) 
s_lshr_b32 s72, s55, 1 // (op 3164) 

s_add_u32 s55, s72, s53 // (op 3164) 


s_lshr_b32 s53, s55, s10 // (op 3164) 

s_and_b32 s55, s10, 2147483648 // (op 3164) 
s_cmp_eq_i32 s55, 0 // (op 3164) 
s_mov_b32 s72, scc // (op 3164) 

s_cmp_eq_i32 s72, 0 // (op 3164) 
s_cselect_b32 s55, s54, s53 // (op 3164) 


v_lshl_add_u32 v13, s55, 7, v14 // (op 3164) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 3164) 
s_mul_i32 s53, s54, s72 // (op 3164) 


s_sub_u32 s54, s52, s53 // (op 3164) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 3164) 


v_mov_b32 v17, 0 // (op 3164) convert
v_mov_b32 v16, v14 // (op 3164) convert

v_mul_lo_u32 v15, v13, s31 // (op 3164) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 3164) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 3164) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 3164) most significant: sum

v_add_co_u32 v14, s[52:53], v14, v16 // (op 3164) least significant half
v_addc_co_u32 v15, s[52:53], v15, v17, s[52:53] // (op 3164) most significant half



v_lshlrev_b64 v[12:13], 2, v[14:15] // (op 3164) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3164) 

v_mov_b32 v12, v14 // (op 3164) convert

s_mul_i32 s53, s30, s35 // (op 3166) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3166) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3166) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3166) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3166) most significant: sum
s_add_u32 s53, s53, s55 // (op 3166) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3166) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 3166) 

s_mov_b32 s68, s32 // (op 3166) reg expression
s_mov_b32 s69, s33 // (op 3166) reg expression
s_mov_b32 s70, s52 // (op 3166) reg expression
s_mov_b32 s71, 131072 // (op 3166) call()

v_lshrrev_b32 v13, 6, v1 // (op 3276) 
s_mul_i32 s52, s16, s8 // (op 3276) 
s_add_u32 s52, s52, s0 // (op 3276) 
v_and_b32 v14, 1, v13 // (op 3276) 

s_mul_hi_u32 s53, s52, s13 // (op 3276) 
v_and_b32 v13, 15, v1 // (op 3276) 
s_sub_u32 s54, s52, s53 // (op 3276) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 3276) 


s_lshr_b32 s55, s54, 1 // (op 3276) 

v_and_b32 v13, 48, v1 // (op 3276) 
s_add_u32 s54, s55, s53 // (op 3276) 


s_lshr_b32 s53, s54, s11 // (op 3276) 

s_and_b32 s54, s11, 2147483648 // (op 3276) 
s_cmp_eq_i32 s54, 0 // (op 3276) 
s_mov_b32 s55, scc // (op 3276) 

s_cmp_eq_i32 s55, 0 // (op 3276) 
s_cselect_b32 s54, s52, s53 // (op 3276) 


s_mul_hi_u32 s53, s54, s12 // (op 3276) 
s_sub_u32 s55, s54, s53 // (op 3276) 
s_lshr_b32 s72, s55, 1 // (op 3276) 

s_add_u32 s55, s72, s53 // (op 3276) 


s_lshr_b32 s53, s55, s10 // (op 3276) 

s_and_b32 s55, s10, 2147483648 // (op 3276) 
s_cmp_eq_i32 s55, 0 // (op 3276) 
s_mov_b32 s72, scc // (op 3276) 

s_cmp_eq_i32 s72, 0 // (op 3276) 
s_cselect_b32 s55, s54, s53 // (op 3276) 


s_mul_i32 s53, s55, s14 // (op 3276) 

s_sub_u32 s55, s54, s53 // (op 3276) 

v_lshl_add_u32 v14, s55, 5, v15 // (op 3276) 


v_mul_lo_u32 v15, v14, s36 // (op 3276) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 3276) 
s_mul_i32 s53, s54, s72 // (op 3276) 


s_sub_u32 s54, s52, s53 // (op 3276) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 3276) 


v_add_u32 v13, v14, v15 // (op 3276) 


s_mul_i32 s53, s36, s23 // (op 3278) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3278) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3278) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3278) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3278) most significant: sum
s_add_u32 s53, s53, s55 // (op 3278) most significant: sum


s_mov_b32 s72, s38 // (op 3278) reg expression
s_mov_b32 s73, s39 // (op 3278) reg expression
s_mov_b32 s74, s52 // (op 3278) reg expression
s_mov_b32 s75, 131072 // (op 3278) call()

v_lshrrev_b32 v14, 6, v1 // (op 3295) 
s_mul_i32 s52, s16, s8 // (op 3295) 
s_add_u32 s52, s52, s0 // (op 3295) 
v_and_b32 v15, 1, v14 // (op 3295) 

s_mul_hi_u32 s53, s52, s13 // (op 3295) 
v_and_b32 v14, 15, v1 // (op 3295) 
s_sub_u32 s54, s52, s53 // (op 3295) 
v_lshl_add_u32 v16, v15, 4, v14 // (op 3295) 


s_lshr_b32 s55, s54, 1 // (op 3295) 

v_and_b32 v14, 48, v1 // (op 3295) 
s_add_u32 s54, s55, s53 // (op 3295) 


s_lshr_b32 s53, s54, s11 // (op 3295) 

s_and_b32 s54, s11, 2147483648 // (op 3295) 
s_cmp_eq_i32 s54, 0 // (op 3295) 
s_mov_b32 s55, scc // (op 3295) 

s_cmp_eq_i32 s55, 0 // (op 3295) 
s_cselect_b32 s54, s52, s53 // (op 3295) 


s_mul_hi_u32 s53, s54, s12 // (op 3295) 
s_sub_u32 s55, s54, s53 // (op 3295) 
s_lshr_b32 s76, s55, 1 // (op 3295) 

s_add_u32 s55, s76, s53 // (op 3295) 


s_lshr_b32 s53, s55, s10 // (op 3295) 

s_and_b32 s55, s10, 2147483648 // (op 3295) 
s_cmp_eq_i32 s55, 0 // (op 3295) 
s_mov_b32 s76, scc // (op 3295) 

s_cmp_eq_i32 s76, 0 // (op 3295) 
s_cselect_b32 s55, s54, s53 // (op 3295) 


s_mul_i32 s53, s55, s14 // (op 3295) 

s_sub_u32 s55, s54, s53 // (op 3295) 

v_lshl_add_u32 v15, s55, 5, v16 // (op 3295) 


v_mul_lo_u32 v16, v15, s36 // (op 3295) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 3295) 
s_mul_i32 s53, s54, s76 // (op 3295) 


s_sub_u32 s54, s52, s53 // (op 3295) 


v_lshl_add_u32 v15, s54, 7, v14 // (op 3295) 


v_add_u32 v14, v15, v16 // (op 3295) 


s_mul_i32 s53, s36, s23 // (op 3297) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3297) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3297) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3297) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3297) most significant: sum
s_add_u32 s53, s53, s55 // (op 3297) most significant: sum


s_mov_b32 s72, s38 // (op 3297) reg expression
s_mov_b32 s73, s39 // (op 3297) reg expression
s_mov_b32 s74, s52 // (op 3297) reg expression
s_mov_b32 s75, 131072 // (op 3297) call()

v_lshrrev_b32 v15, 2, v1 // (op 3404) 
s_mul_i32 s52, s16, s8 // (op 3404) 
s_add_u32 s52, s52, s0 // (op 3404) 
v_lshlrev_b32 v16, 1, v15 // (op 3404) 

s_mul_hi_u32 s53, s52, s13 // (op 3404) 
v_and_b32 v15, -4, v1 // (op 3404) 
s_sub_u32 s54, s52, s53 // (op 3404) 
v_sub_u32 v17, v1, v15 // (op 3404) 

s_lshr_b32 s55, s54, 1 // (op 3404) 

s_add_u32 s54, s55, s53 // (op 3404) 


s_lshr_b32 s53, s54, s11 // (op 3404) 

s_and_b32 s54, s11, 2147483648 // (op 3404) 
s_cmp_eq_i32 s54, 0 // (op 3404) 
s_mov_b32 s55, scc // (op 3404) 

s_cmp_eq_i32 s55, 0 // (op 3404) 
s_cselect_b32 s54, s52, s53 // (op 3404) 


s_mul_hi_u32 s53, s54, s12 // (op 3404) 
s_sub_u32 s55, s54, s53 // (op 3404) 
s_lshr_b32 s76, s55, 1 // (op 3404) 

s_add_u32 s55, s76, s53 // (op 3404) 


s_lshr_b32 s53, s55, s10 // (op 3404) 

s_and_b32 s55, s10, 2147483648 // (op 3404) 
s_cmp_eq_i32 s55, 0 // (op 3404) 
s_mov_b32 s76, scc // (op 3404) 

s_cmp_eq_i32 s76, 0 // (op 3404) 
s_cselect_b32 s55, s54, s53 // (op 3404) 


v_lshl_add_u32 v15, s55, 7, v16 // (op 3404) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 3404) 
s_mul_i32 s53, s54, s76 // (op 3404) 


s_sub_u32 s54, s52, s53 // (op 3404) 


v_lshl_add_u32 v16, s54, 2, v17 // (op 3404) 


v_mul_lo_u32 v17, v15, s40 // (op 3404) 
v_add_u32 v17, v17, v16 // (op 3404) 


s_mul_i32 s53, s40, s35 // (op 3406) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3406) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3406) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3406) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3406) most significant: sum
s_add_u32 s53, s53, s55 // (op 3406) most significant: sum


s_mov_b32 s76, s42 // (op 3406) reg expression
s_mov_b32 s77, s43 // (op 3406) reg expression
s_mov_b32 s78, s52 // (op 3406) reg expression
s_mov_b32 s79, 131072 // (op 3406) call()

v_lshrrev_b32 v15, 2, v1 // (op 3423) 
s_mul_i32 s52, s16, s8 // (op 3423) 
s_add_u32 s52, s52, s0 // (op 3423) 
v_lshlrev_b32 v16, 1, v15 // (op 3423) 

s_mul_hi_u32 s53, s52, s13 // (op 3423) 
v_and_b32 v15, -4, v1 // (op 3423) 
s_sub_u32 s54, s52, s53 // (op 3423) 
v_sub_u32 v18, v1, v15 // (op 3423) 

s_lshr_b32 s55, s54, 1 // (op 3423) 

s_add_u32 s54, s55, s53 // (op 3423) 


s_lshr_b32 s53, s54, s11 // (op 3423) 

s_and_b32 s54, s11, 2147483648 // (op 3423) 
s_cmp_eq_i32 s54, 0 // (op 3423) 
s_mov_b32 s55, scc // (op 3423) 

s_cmp_eq_i32 s55, 0 // (op 3423) 
s_cselect_b32 s54, s52, s53 // (op 3423) 


s_mul_hi_u32 s53, s54, s12 // (op 3423) 
s_sub_u32 s55, s54, s53 // (op 3423) 
s_lshr_b32 s80, s55, 1 // (op 3423) 

s_add_u32 s55, s80, s53 // (op 3423) 


s_lshr_b32 s53, s55, s10 // (op 3423) 

s_and_b32 s55, s10, 2147483648 // (op 3423) 
s_cmp_eq_i32 s55, 0 // (op 3423) 
s_mov_b32 s80, scc // (op 3423) 

s_cmp_eq_i32 s80, 0 // (op 3423) 
s_cselect_b32 s55, s54, s53 // (op 3423) 


v_lshl_add_u32 v15, s55, 7, v16 // (op 3423) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 3423) 
s_mul_i32 s53, s54, s80 // (op 3423) 


s_sub_u32 s54, s52, s53 // (op 3423) 


v_lshl_add_u32 v16, s54, 2, v18 // (op 3423) 


v_mul_lo_u32 v18, v15, s40 // (op 3423) 
v_add_u32 v18, v18, v16 // (op 3423) 


s_mul_i32 s53, s40, s35 // (op 3425) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3425) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3425) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3425) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3425) most significant: sum
s_add_u32 s53, s53, s55 // (op 3425) most significant: sum


s_mov_b32 s76, s42 // (op 3425) reg expression
s_mov_b32 s77, s43 // (op 3425) reg expression
s_mov_b32 s78, s52 // (op 3425) reg expression
s_mov_b32 s79, 131072 // (op 3425) call()

v_lshrrev_b32 v15, 6, v1 // (op 3532) 
s_mul_i32 s52, s16, s8 // (op 3532) 
s_add_u32 s52, s52, s0 // (op 3532) 
v_and_b32 v16, 1, v15 // (op 3532) 

s_mul_hi_u32 s53, s52, s13 // (op 3532) 
v_and_b32 v15, 15, v1 // (op 3532) 
s_sub_u32 s54, s52, s53 // (op 3532) 
v_lshl_add_u32 v19, v16, 4, v15 // (op 3532) 


s_lshr_b32 s55, s54, 1 // (op 3532) 

v_and_b32 v15, 63, v1 // (op 3532) 
s_add_u32 s54, s55, s53 // (op 3532) 


v_lshrrev_b32 v16, 4, v15 // (op 3532) 

s_lshr_b32 s53, s54, s11 // (op 3532) 

s_and_b32 s54, s11, 2147483648 // (op 3532) 
s_cmp_eq_i32 s54, 0 // (op 3532) 
s_mov_b32 s55, scc // (op 3532) 

s_cmp_eq_i32 s55, 0 // (op 3532) 
s_cselect_b32 s54, s52, s53 // (op 3532) 


s_mul_hi_u32 s53, s54, s12 // (op 3532) 
s_sub_u32 s55, s54, s53 // (op 3532) 
s_lshr_b32 s80, s55, 1 // (op 3532) 

s_add_u32 s55, s80, s53 // (op 3532) 


s_lshr_b32 s53, s55, s10 // (op 3532) 

s_and_b32 s55, s10, 2147483648 // (op 3532) 
s_cmp_eq_i32 s55, 0 // (op 3532) 
s_mov_b32 s80, scc // (op 3532) 

s_cmp_eq_i32 s80, 0 // (op 3532) 
s_cselect_b32 s55, s54, s53 // (op 3532) 


s_mul_i32 s53, s55, s14 // (op 3532) 

s_sub_u32 s55, s54, s53 // (op 3532) 

v_lshl_add_u32 v15, s55, 5, v19 // (op 3532) 


v_mul_lo_u32 v19, v15, s44 // (op 3532) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 3532) 
s_mul_i32 s53, s54, s80 // (op 3532) 


s_sub_u32 s54, s52, s53 // (op 3532) 


v_lshl_add_u32 v15, s54, 2, v16 // (op 3532) 


v_add_u32 v16, v15, v19 // (op 3532) 


s_mul_i32 s53, s44, s23 // (op 3534) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3534) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3534) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3534) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3534) most significant: sum
s_add_u32 s53, s53, s55 // (op 3534) most significant: sum


s_mov_b32 s80, s46 // (op 3534) reg expression
s_mov_b32 s81, s47 // (op 3534) reg expression
s_mov_b32 s82, s52 // (op 3534) reg expression
s_mov_b32 s83, 131072 // (op 3534) call()

v_lshrrev_b32 v15, 6, v1 // (op 3551) 
s_mul_i32 s52, s16, s8 // (op 3551) 
s_add_u32 s52, s52, s0 // (op 3551) 
v_and_b32 v19, 1, v15 // (op 3551) 

s_mul_hi_u32 s53, s52, s13 // (op 3551) 
v_and_b32 v15, 15, v1 // (op 3551) 
s_sub_u32 s54, s52, s53 // (op 3551) 
v_lshl_add_u32 v20, v19, 4, v15 // (op 3551) 


s_lshr_b32 s55, s54, 1 // (op 3551) 

v_and_b32 v15, 63, v1 // (op 3551) 
s_add_u32 s54, s55, s53 // (op 3551) 


v_lshrrev_b32 v19, 4, v15 // (op 3551) 

s_lshr_b32 s53, s54, s11 // (op 3551) 

s_and_b32 s54, s11, 2147483648 // (op 3551) 
s_cmp_eq_i32 s54, 0 // (op 3551) 
s_mov_b32 s55, scc // (op 3551) 

s_cmp_eq_i32 s55, 0 // (op 3551) 
s_cselect_b32 s54, s52, s53 // (op 3551) 


s_mul_hi_u32 s53, s54, s12 // (op 3551) 
s_sub_u32 s55, s54, s53 // (op 3551) 
s_lshr_b32 s84, s55, 1 // (op 3551) 

s_add_u32 s55, s84, s53 // (op 3551) 


s_lshr_b32 s53, s55, s10 // (op 3551) 

s_and_b32 s55, s10, 2147483648 // (op 3551) 
s_cmp_eq_i32 s55, 0 // (op 3551) 
s_mov_b32 s84, scc // (op 3551) 

s_cmp_eq_i32 s84, 0 // (op 3551) 
s_cselect_b32 s55, s54, s53 // (op 3551) 


s_mul_i32 s53, s55, s14 // (op 3551) 

s_sub_u32 s55, s54, s53 // (op 3551) 

v_lshl_add_u32 v15, s55, 5, v20 // (op 3551) 


v_mul_lo_u32 v20, v15, s44 // (op 3551) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 3551) 
s_mul_i32 s53, s54, s84 // (op 3551) 


s_sub_u32 s54, s52, s53 // (op 3551) 


v_lshl_add_u32 v15, s54, 2, v19 // (op 3551) 


v_add_u32 v19, v15, v20 // (op 3551) 


s_mul_i32 s53, s44, s23 // (op 3553) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3553) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3553) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3553) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3553) most significant: sum
s_add_u32 s53, s53, s55 // (op 3553) most significant: sum


s_mov_b32 s80, s46 // (op 3553) reg expression
s_mov_b32 s81, s47 // (op 3553) reg expression
s_mov_b32 s82, s52 // (op 3553) reg expression
s_mov_b32 s83, 131072 // (op 3553) call()



v_mov_b32 v15, v9 // (op 2848) reg expression

buffer_load_dwordx4 v[20:23], v15, s[68:71], 0 offen // (op 2848) Load value




buffer_load_dwordx4 v[24:27], v13, s[72:75], 0 offen // (op 2854) Load value
buffer_load_dwordx4 v[28:31], v13, s[72:75], 0 offen offset:64 // (op 2854) Load value



buffer_load_ubyte v15, v16, s[80:83], 0 offen offset: 0 // (op 2859) Load value


s_lshl_b64 s[52:53], s[30:31], 5 // (op 2865) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2865) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2865) 

v_add_u32 v32, v9, s52 // (op 2865) 


buffer_load_dwordx4 v[34:37], v32, s[68:71], 0 offen // (op 2865) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 2872) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2872) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2872) 

s_lshl_b32 s54, s52, 1 // (op 2872) 

v_add_u32 v32, v9, s54 // (op 2872) 


buffer_load_dwordx4 v[38:41], v32, s[68:71], 0 offen // (op 2872) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 2879) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2879) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2879) 

s_mul_i32 s54, 3, s52 // (op 2879) 

v_add_u32 v32, v9, s54 // (op 2879) 


buffer_load_dwordx4 v[42:45], v32, s[68:71], 0 offen // (op 2879) Load value


v_mov_b64 v[32:33], s[40:41] // (op 2884) convert

v_mov_b32 v46, v17 // (op 2884) 
buffer_load_ubyte v48, v46, s[76:79], 0 offen offset: 0 // (op 2884) Load value
v_add_u32 v46, v46, v32 // (op 2884) 
buffer_load_ubyte v49, v46, s[76:79], 0 offen offset: 0 // (op 2884) Load value





buffer_load_ubyte v32, v16, s[80:83], 0 offen offset:4 // (op 2888) Load value



v_add_u32 v16, 64, v9 // (op 2894) 

buffer_load_dwordx4 v[50:53], v16, s[68:71], 0 offen // (op 2894) Load value




buffer_load_dwordx4 v[54:57], v13, s[72:75], 0 offen offset:128 // (op 2900) Load value
buffer_load_dwordx4 v[58:61], v13, s[72:75], 0 offen offset:192 // (op 2900) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 2906) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2906) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2906) 

s_add_u32 s54, s52, 64 // (op 2906) 

v_add_u32 v13, v9, s54 // (op 2906) 


buffer_load_dwordx4 v[62:65], v13, s[68:71], 0 offen // (op 2906) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 2913) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2913) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2913) 

s_lshl_b32 s54, s52, 1 // (op 2913) 
s_add_u32 s54, s54, 64 // (op 2913) 

v_add_u32 v13, v9, s54 // (op 2913) 


buffer_load_dwordx4 v[66:69], v13, s[68:71], 0 offen // (op 2913) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 2920) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2920) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2920) 

s_mul_i32 s54, 3, s52 // (op 2920) 
s_add_u32 s54, s54, 64 // (op 2920) 

v_add_u32 v13, v9, s54 // (op 2920) 


buffer_load_dwordx4 v[70:73], v13, s[68:71], 0 offen // (op 2920) Load value



v_mov_b64 v[46:47], s[40:41] // (op 2925) convert

v_mov_b32 v9, v17 // (op 2925) 
buffer_load_ubyte v74, v9, s[76:79], 0 offen offset:4 // (op 2925) Load value
v_add_u32 v9, v9, v46 // (op 2925) 
buffer_load_ubyte v75, v9, s[76:79], 0 offen offset:4 // (op 2925) Load value



v_and_b32 v9, -4, v1 // (op 3644) 
v_sub_u32 v13, v1, v9 // (op 3644) 

v_lshrrev_b32 v9, 2, v1 // (op 3644) 
v_lshl_add_u32 v33, v9, 3, v13 // (op 3644) 


s_barrier  // (op 3956) 



s_waitcnt vmcnt(10)
ds_write_b8 v33, v48 // (op 2927) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v33, v49 offset:4 // (op 2927) Store local data 

v_and_b32 v9, -4, v1 // (op 3615) 
v_sub_u32 v13, v1, v9 // (op 3615) 

v_lshrrev_b32 v9, 2, v1 // (op 3615) 
v_lshl_add_u32 v33, v9, 3, v13 // (op 3615) 


v_and_b32 v9, -4, v1 // (op 3844) 
v_sub_u32 v13, v1, v9 // (op 3844) 

v_lshrrev_b32 v9, 2, v1 // (op 3844) 
v_lshl_add_u32 v16, v9, 3, v13 // (op 3844) 


s_mov_b32 s1, 0 // (op 1499) call()
s_and_b32 s52, s7, 4294967294 // (op 1498) 
s_cmp_lt_u32 s1, s52 // (op 1498) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1498 // (op 1498) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1498 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1498:
s_add_u32 s1, s1, 2 // (op 1500) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3953) 



ds_read_u8 v9, v0 // (op 65) Load local data 



ds_read_u8 v13, v0 offset:128 // (op 1280) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:31], a[0:3], v9, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 72) 




ds_read_u8 v9, v0 offset:256 // (op 1402) Load local data 



ds_read_u8 v17, v0 offset:384 // (op 1524) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[24:31], a[4:7], v13, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1282) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:41], v[24:31], a[8:11], v9, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1404) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:45], v[24:31], a[12:15], v17, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1526) 



v_mov_b32 v9, v12 // (op 57) 
v_add_u32 v13, 128, v9 // (op 57) 

buffer_load_dwordx4 v[20:23], v13, s[68:71], 0 offen // (op 57) Load value



v_mov_b32 v13, v14 // (op 61) 

buffer_load_dwordx4 v[24:27], v13, s[72:75], 0 offen offset:256 // (op 61) Load value
buffer_load_dwordx4 v[28:31], v13, s[72:75], 0 offen offset:320 // (op 61) Load value


v_mov_b32 v17, v19 // (op 70) 

buffer_load_ubyte v15, v17, s[80:83], 0 offen offset:8 // (op 70) Load value


s_lshl_b64 s[52:53], s[30:31], 5 // (op 1305) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1305) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1305) 

s_add_u32 s54, s52, 128 // (op 1305) 

v_add_u32 v46, v9, s54 // (op 1305) 


buffer_load_dwordx4 v[34:37], v46, s[68:71], 0 offen // (op 1305) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 1427) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1427) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1427) 

s_lshl_b32 s54, s52, 1 // (op 1427) 
s_add_u32 s54, s54, 128 // (op 1427) 

v_add_u32 v46, v9, s54 // (op 1427) 


buffer_load_dwordx4 v[38:41], v46, s[68:71], 0 offen // (op 1427) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 1549) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1549) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1549) 

s_mul_i32 s54, 3, s52 // (op 1549) 
s_add_u32 s54, s54, 128 // (op 1549) 

v_add_u32 v46, v9, s54 // (op 1549) 


buffer_load_dwordx4 v[42:45], v46, s[68:71], 0 offen // (op 1549) Load value


v_mov_b64 v[46:47], s[40:41] // (op 9) convert

v_mov_b32 v76, v18 // (op 9) 
v_mov_b32 v77, v76 // (op 9) 
buffer_load_ubyte v48, v77, s[76:79], 0 offen offset:8 // (op 9) Load value
v_add_u32 v77, v77, v46 // (op 9) 
buffer_load_ubyte v49, v77, s[76:79], 0 offen offset:8 // (op 9) Load value


s_barrier  // (op 3994) 



s_waitcnt vmcnt(10)
ds_write_b8 v16, v74 offset:512 // (op 1193) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v16, v75 offset:516 // (op 1193) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3991) 



ds_read_u8 v46, v5 offset:512 // (op 1195) Load local data 



ds_read_u8 v47, v5 offset:640 // (op 1289) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[54:61], a[0:3], v46, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1198) 




ds_read_u8 v46, v5 offset:768 // (op 1411) Load local data 



ds_read_u8 v77, v5 offset:896 // (op 1533) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[54:61], a[4:7], v47, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1283) 

s_barrier  // (op 4199) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[54:61], a[8:11], v46, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1405) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[54:61], a[12:15], v77, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1527) 




buffer_load_ubyte v32, v17, s[80:83], 0 offen offset:12 // (op 1197) Load value



v_add_u32 v19, 8, v19 // (op 3543) 
v_add_u32 v17, 192, v9 // (op 1200) 

buffer_load_dwordx4 v[50:53], v17, s[68:71], 0 offen // (op 1200) Load value




buffer_load_dwordx4 v[54:57], v13, s[72:75], 0 offen offset:384 // (op 1202) Load value
buffer_load_dwordx4 v[58:61], v13, s[72:75], 0 offen offset:448 // (op 1202) Load value



v_add_u32 v14, 256, v14 // (op 3287) 
s_lshl_b64 s[52:53], s[30:31], 5 // (op 1298) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1298) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1298) 

s_add_u32 s54, s52, 192 // (op 1298) 

v_add_u32 v13, v9, s54 // (op 1298) 


buffer_load_dwordx4 v[62:65], v13, s[68:71], 0 offen // (op 1298) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 1420) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1420) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1420) 

s_lshl_b32 s54, s52, 1 // (op 1420) 
s_add_u32 s54, s54, 192 // (op 1420) 

v_add_u32 v13, v9, s54 // (op 1420) 


buffer_load_dwordx4 v[66:69], v13, s[68:71], 0 offen // (op 1420) Load value



s_lshl_b64 s[52:53], s[30:31], 5 // (op 1542) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1542) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1542) 

s_mul_i32 s54, 3, s52 // (op 1542) 
s_add_u32 s54, s54, 192 // (op 1542) 

v_add_u32 v13, v9, s54 // (op 1542) 


buffer_load_dwordx4 v[70:73], v13, s[68:71], 0 offen // (op 1542) Load value



v_add_u32 v12, 128, v12 // (op 3154) 
v_mov_b64 v[46:47], s[40:41] // (op 1192) convert

v_mov_b32 v9, v76 // (op 1192) 
buffer_load_ubyte v74, v9, s[76:79], 0 offen offset:12 // (op 1192) Load value
v_add_u32 v9, v9, v46 // (op 1192) 
buffer_load_ubyte v75, v9, s[76:79], 0 offen offset:12 // (op 1192) Load value



s_barrier  // (op 3959) 
v_add_u32 v18, 8, v18 // (op 3415) 



s_waitcnt vmcnt(10)
ds_write_b8 v33, v48 // (op 13) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v33, v49 offset:4 // (op 13) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 1498) 
s_cmp_lt_u32 s1, s52 // (op 1498) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1498 // (op 1498) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1498 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1498:













s_and_b32 s52, s7, 4294967294 // (op 3048) 
v_lshrrev_b32 v9, 7, v1 // (op 3048) 
s_add_u32 s53, s0, s52 // (op 3048) 

v_and_b32 v12, 15, v1 // (op 3048) 
v_lshl_add_u32 v13, v9, 4, v12 // (op 3048) 


s_mul_i32 s52, s16, s8 // (op 3048) 
s_add_u32 s52, s52, s53 // (op 3048) 

v_and_b32 v9, 63, v1 // (op 3048) 
s_mul_hi_u32 s53, s52, s13 // (op 3048) 
v_lshrrev_b32 v12, 4, v9 // (op 3048) 

s_sub_u32 s54, s52, s53 // (op 3048) 
v_lshlrev_b32 v9, 5, v12 // (op 3048) 

s_lshr_b32 s55, s54, 1 // (op 3048) 

s_add_u32 s54, s55, s53 // (op 3048) 


s_lshr_b32 s53, s54, s11 // (op 3048) 

s_and_b32 s54, s11, 2147483648 // (op 3048) 
s_cmp_eq_i32 s54, 0 // (op 3048) 
s_mov_b32 s55, scc // (op 3048) 

s_cmp_eq_i32 s55, 0 // (op 3048) 
s_cselect_b32 s54, s52, s53 // (op 3048) 


s_mul_hi_u32 s53, s54, s12 // (op 3048) 
s_sub_u32 s55, s54, s53 // (op 3048) 
s_lshr_b32 s84, s55, 1 // (op 3048) 

s_add_u32 s55, s84, s53 // (op 3048) 


s_lshr_b32 s53, s55, s10 // (op 3048) 

s_and_b32 s55, s10, 2147483648 // (op 3048) 
s_cmp_eq_i32 s55, 0 // (op 3048) 
s_mov_b32 s84, scc // (op 3048) 

s_cmp_eq_i32 s84, 0 // (op 3048) 
s_cselect_b32 s55, s54, s53 // (op 3048) 


v_lshl_add_u32 v12, s55, 7, v13 // (op 3048) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 3048) 
s_mul_i32 s53, s54, s84 // (op 3048) 


s_sub_u32 s54, s52, s53 // (op 3048) 


v_lshl_add_u32 v13, s54, 7, v9 // (op 3048) 


v_mov_b32 v15, 0 // (op 3048) convert
v_mov_b32 v14, v13 // (op 3048) convert

v_mul_lo_u32 v17, v12, s31 // (op 3048) most significant: low of low * high
v_mul_hi_u32 v9, v12, s30 // (op 3048) most significant: high of low * low
v_mul_lo_u32 v16, v12, s30 // (op 3048) least significant: low of low * low
v_add_u32 v17, v17, v9 // (op 3048) most significant: sum

v_add_co_u32 v16, s[52:53], v16, v14 // (op 3048) least significant half
v_addc_co_u32 v17, s[52:53], v17, v15, s[52:53] // (op 3048) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 3048) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3048) 

v_mov_b32 v9, v14 // (op 3048) convert

s_mul_i32 s53, s30, s35 // (op 3050) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3050) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3050) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3050) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3050) most significant: sum
s_add_u32 s53, s53, s55 // (op 3050) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3050) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 3050) 

s_mov_b32 s68, s32 // (op 3050) reg expression
s_mov_b32 s69, s33 // (op 3050) reg expression
s_mov_b32 s70, s52 // (op 3050) reg expression
s_mov_b32 s71, 131072 // (op 3050) call()

s_and_b32 s52, s7, 4294967294 // (op 3211) 
v_lshrrev_b32 v12, 6, v1 // (op 3211) 
s_add_u32 s53, s0, s52 // (op 3211) 

v_and_b32 v13, 1, v12 // (op 3211) 

v_and_b32 v12, 15, v1 // (op 3211) 
s_mul_i32 s52, s16, s8 // (op 3211) 
s_add_u32 s52, s52, s53 // (op 3211) 

v_lshl_add_u32 v14, v13, 4, v12 // (op 3211) 


s_mul_hi_u32 s53, s52, s13 // (op 3211) 
v_and_b32 v12, 48, v1 // (op 3211) 
s_sub_u32 s54, s52, s53 // (op 3211) 
s_lshr_b32 s55, s54, 1 // (op 3211) 

s_add_u32 s54, s55, s53 // (op 3211) 


s_lshr_b32 s53, s54, s11 // (op 3211) 

s_and_b32 s54, s11, 2147483648 // (op 3211) 
s_cmp_eq_i32 s54, 0 // (op 3211) 
s_mov_b32 s55, scc // (op 3211) 

s_cmp_eq_i32 s55, 0 // (op 3211) 
s_cselect_b32 s54, s52, s53 // (op 3211) 


s_mul_hi_u32 s53, s54, s12 // (op 3211) 
s_sub_u32 s55, s54, s53 // (op 3211) 
s_lshr_b32 s84, s55, 1 // (op 3211) 

s_add_u32 s55, s84, s53 // (op 3211) 


s_lshr_b32 s53, s55, s10 // (op 3211) 

s_and_b32 s55, s10, 2147483648 // (op 3211) 
s_cmp_eq_i32 s55, 0 // (op 3211) 
s_mov_b32 s84, scc // (op 3211) 

s_cmp_eq_i32 s84, 0 // (op 3211) 
s_cselect_b32 s55, s54, s53 // (op 3211) 


s_mul_i32 s53, s55, s14 // (op 3211) 

s_sub_u32 s55, s54, s53 // (op 3211) 

v_lshl_add_u32 v13, s55, 5, v14 // (op 3211) 


v_mul_lo_u32 v14, v13, s36 // (op 3211) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 3211) 
s_mul_i32 s53, s54, s84 // (op 3211) 


s_sub_u32 s54, s52, s53 // (op 3211) 


v_lshl_add_u32 v13, s54, 7, v12 // (op 3211) 


v_add_u32 v12, v13, v14 // (op 3211) 


s_mul_i32 s53, s36, s23 // (op 3213) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3213) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3213) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3213) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3213) most significant: sum
s_add_u32 s53, s53, s55 // (op 3213) most significant: sum


s_mov_b32 s72, s38 // (op 3213) reg expression
s_mov_b32 s73, s39 // (op 3213) reg expression
s_mov_b32 s74, s52 // (op 3213) reg expression
s_mov_b32 s75, 131072 // (op 3213) call()

s_and_b32 s52, s7, 4294967294 // (op 3339) 
v_lshrrev_b32 v13, 2, v1 // (op 3339) 
s_add_u32 s53, s0, s52 // (op 3339) 

v_lshlrev_b32 v14, 1, v13 // (op 3339) 

v_and_b32 v13, -4, v1 // (op 3339) 
s_mul_i32 s52, s16, s8 // (op 3339) 
s_add_u32 s52, s52, s53 // (op 3339) 

v_sub_u32 v15, v1, v13 // (op 3339) 

s_mul_hi_u32 s53, s52, s13 // (op 3339) 
s_sub_u32 s54, s52, s53 // (op 3339) 
s_lshr_b32 s55, s54, 1 // (op 3339) 

s_add_u32 s54, s55, s53 // (op 3339) 


s_lshr_b32 s53, s54, s11 // (op 3339) 

s_and_b32 s54, s11, 2147483648 // (op 3339) 
s_cmp_eq_i32 s54, 0 // (op 3339) 
s_mov_b32 s55, scc // (op 3339) 

s_cmp_eq_i32 s55, 0 // (op 3339) 
s_cselect_b32 s54, s52, s53 // (op 3339) 


s_mul_hi_u32 s53, s54, s12 // (op 3339) 
s_sub_u32 s55, s54, s53 // (op 3339) 
s_lshr_b32 s84, s55, 1 // (op 3339) 

s_add_u32 s55, s84, s53 // (op 3339) 


s_lshr_b32 s53, s55, s10 // (op 3339) 

s_and_b32 s55, s10, 2147483648 // (op 3339) 
s_cmp_eq_i32 s55, 0 // (op 3339) 
s_mov_b32 s84, scc // (op 3339) 

s_cmp_eq_i32 s84, 0 // (op 3339) 
s_cselect_b32 s55, s54, s53 // (op 3339) 


v_lshl_add_u32 v13, s55, 7, v14 // (op 3339) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 3339) 
s_mul_i32 s53, s54, s84 // (op 3339) 


s_sub_u32 s54, s52, s53 // (op 3339) 


v_lshl_add_u32 v14, s54, 2, v15 // (op 3339) 


v_mul_lo_u32 v15, v13, s40 // (op 3339) 
v_add_u32 v15, v15, v14 // (op 3339) 


s_mul_i32 s53, s40, s35 // (op 3341) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3341) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3341) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3341) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3341) most significant: sum
s_add_u32 s53, s53, s55 // (op 3341) most significant: sum


s_mov_b32 s76, s42 // (op 3341) reg expression
s_mov_b32 s77, s43 // (op 3341) reg expression
s_mov_b32 s78, s52 // (op 3341) reg expression
s_mov_b32 s79, 131072 // (op 3341) call()

s_and_b32 s52, s7, 4294967294 // (op 3467) 
v_lshrrev_b32 v13, 6, v1 // (op 3467) 
s_add_u32 s53, s0, s52 // (op 3467) 

v_and_b32 v14, 1, v13 // (op 3467) 

v_and_b32 v13, 15, v1 // (op 3467) 
s_mul_i32 s52, s16, s8 // (op 3467) 
s_add_u32 s52, s52, s53 // (op 3467) 

v_lshl_add_u32 v16, v14, 4, v13 // (op 3467) 


s_mul_hi_u32 s53, s52, s13 // (op 3467) 
v_and_b32 v13, 63, v1 // (op 3467) 
s_sub_u32 s54, s52, s53 // (op 3467) 
v_lshrrev_b32 v14, 4, v13 // (op 3467) 

s_lshr_b32 s55, s54, 1 // (op 3467) 

s_add_u32 s54, s55, s53 // (op 3467) 


s_lshr_b32 s53, s54, s11 // (op 3467) 

s_and_b32 s54, s11, 2147483648 // (op 3467) 
s_cmp_eq_i32 s54, 0 // (op 3467) 
s_mov_b32 s55, scc // (op 3467) 

s_cmp_eq_i32 s55, 0 // (op 3467) 
s_cselect_b32 s54, s52, s53 // (op 3467) 


s_mul_hi_u32 s53, s54, s12 // (op 3467) 
s_sub_u32 s55, s54, s53 // (op 3467) 
s_lshr_b32 s84, s55, 1 // (op 3467) 

s_add_u32 s55, s84, s53 // (op 3467) 


s_lshr_b32 s53, s55, s10 // (op 3467) 

s_and_b32 s55, s10, 2147483648 // (op 3467) 
s_cmp_eq_i32 s55, 0 // (op 3467) 
s_mov_b32 s84, scc // (op 3467) 

s_cmp_eq_i32 s84, 0 // (op 3467) 
s_cselect_b32 s55, s54, s53 // (op 3467) 


s_mul_i32 s53, s55, s14 // (op 3467) 

s_sub_u32 s55, s54, s53 // (op 3467) 

v_lshl_add_u32 v13, s55, 5, v16 // (op 3467) 


v_mul_lo_u32 v16, v13, s44 // (op 3467) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 3467) 
s_mul_i32 s53, s54, s84 // (op 3467) 


s_sub_u32 s54, s52, s53 // (op 3467) 


v_lshl_add_u32 v13, s54, 2, v14 // (op 3467) 


v_add_u32 v14, v13, v16 // (op 3467) 


s_mul_i32 s53, s44, s23 // (op 3469) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3469) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3469) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3469) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3469) most significant: sum
s_add_u32 s53, s53, s55 // (op 3469) most significant: sum


s_mov_b32 s80, s46 // (op 3469) reg expression
s_mov_b32 s81, s47 // (op 3469) reg expression
s_mov_b32 s82, s52 // (op 3469) reg expression
s_mov_b32 s83, 131072 // (op 3469) call()

v_and_b32 v13, -4, v1 // (op 3628) 
v_sub_u32 v16, v1, v13 // (op 3628) 

v_lshrrev_b32 v13, 2, v1 // (op 3628) 
v_lshl_add_u32 v17, v13, 3, v16 // (op 3628) 


s_and_b32 s52, s7, 4294967294 // (op 1504) 
s_cmp_lt_u32 s52, s7 // (op 1503) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1503 // (op 1503) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1503 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1503:

s_add_u32 s52, s52, 1 // (op 1505) 
v_mov_b64 v[18:19], s[40:41] // (op 1150) convert

v_mov_b32 v13, v15 // (op 1150) 
v_mov_b32 v16, v13 // (op 1150) 
buffer_load_ubyte v20, v16, s[76:79], 0 offen offset: 0 // (op 1150) Load value
v_add_u32 v16, v16, v18 // (op 1150) 
buffer_load_ubyte v21, v16, s[76:79], 0 offen offset: 0 // (op 1150) Load value



v_add_u32 v15, 4, v15 // (op 3331) 


s_barrier  // (op 3962) 
v_mov_b32 v13, v9 // (op 1158) 



v_mov_b32 v16, v13 // (op 1158) reg expression

buffer_load_dwordx4 v[22:25], v16, s[68:71], 0 offen // (op 1158) Load value

s_waitcnt vmcnt(2)
ds_write_b8 v17, v20 // (op 1151) Store local data 


v_mov_b32 v16, v12 // (op 1160) 

buffer_load_dwordx4 v[26:29], v16, s[72:75], 0 offen // (op 1160) Load value
buffer_load_dwordx4 v[30:33], v16, s[72:75], 0 offen offset:64 // (op 1160) Load value

v_add_u32 v12, 128, v12 // (op 3203) 
s_waitcnt vmcnt(3)
ds_write_b8 v17, v21 offset:4 // (op 1151) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3965) 



ds_read_u8 v16, v0 // (op 1153) Load local data 


v_mov_b32 v18, v14 // (op 1155) 

buffer_load_ubyte v19, v18, s[80:83], 0 offen offset: 0 // (op 1155) Load value

v_add_u32 v14, 4, v14 // (op 3459) 


s_lshl_b64 s[54:55], s[30:31], 5 // (op 1275) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1275) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1275) 

v_add_u32 v18, v13, s54 // (op 1275) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:33], a[0:3], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1156) 
buffer_load_dwordx4 v[34:37], v18, s[68:71], 0 offen // (op 1275) Load value






ds_read_u8 v16, v0 offset:128 // (op 1265) Load local data 


s_lshl_b64 s[54:55], s[30:31], 5 // (op 1397) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1397) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1397) 

s_lshl_b32 s53, s54, 1 // (op 1397) 

v_add_u32 v18, v13, s53 // (op 1397) 


buffer_load_dwordx4 v[20:23], v18, s[68:71], 0 offen // (op 1397) Load value

s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[26:33], a[4:7], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1267) 





ds_read_u8 v16, v0 offset:256 // (op 1387) Load local data 


s_lshl_b64 s[54:55], s[30:31], 5 // (op 1519) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1519) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1519) 

s_mul_i32 s53, 3, s54 // (op 1519) 

v_add_u32 v18, v13, s53 // (op 1519) 


buffer_load_dwordx4 v[34:37], v18, s[68:71], 0 offen // (op 1519) Load value

s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[26:33], a[8:11], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1389) 






v_add_u32 v9, 64, v9 // (op 3036) 
ds_read_u8 v13, v0 offset:384 // (op 1509) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[26:33], a[12:15], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1511) 




s_cmp_lt_u32 s52, s7 // (op 1503) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1503 // (op 1503) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1503 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1503:






s_barrier  // (op 4185) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 251) 
s_cmp_lg_u32 s18, 0 // (op 277) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_277 // (op 277) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_277



v_add_u32 v9, 12288, v8 // (op 1718) 
s_nop 0
buffer_store_dword a12, v9, s[64:67], 0 offen sc1 // (op 1718) Store value

v_add_u32 v9, 12416, v8 // (op 1718) 
buffer_store_dword a13, v9, s[64:67], 0 offen sc1 // (op 1718) Store value

v_add_u32 v9, 12544, v8 // (op 1718) 
buffer_store_dword a14, v9, s[64:67], 0 offen sc1 // (op 1718) Store value

v_add_u32 v9, 12672, v8 // (op 1718) 
buffer_store_dword a15, v9, s[64:67], 0 offen sc1 // (op 1718) Store value




v_add_u32 v9, 8192, v8 // (op 1711) 
buffer_store_dword a8, v9, s[64:67], 0 offen sc1 // (op 1711) Store value

v_add_u32 v9, 8320, v8 // (op 1711) 
buffer_store_dword a9, v9, s[64:67], 0 offen sc1 // (op 1711) Store value

v_add_u32 v9, 8448, v8 // (op 1711) 
buffer_store_dword a10, v9, s[64:67], 0 offen sc1 // (op 1711) Store value

v_add_u32 v9, 8576, v8 // (op 1711) 
buffer_store_dword a11, v9, s[64:67], 0 offen sc1 // (op 1711) Store value




v_add_u32 v9, 4096, v8 // (op 1704) 
buffer_store_dword a4, v9, s[64:67], 0 offen sc1 // (op 1704) Store value

v_add_u32 v9, 4224, v8 // (op 1704) 
buffer_store_dword a5, v9, s[64:67], 0 offen sc1 // (op 1704) Store value

v_add_u32 v9, 4352, v8 // (op 1704) 
buffer_store_dword a6, v9, s[64:67], 0 offen sc1 // (op 1704) Store value

v_add_u32 v9, 4480, v8 // (op 1704) 
buffer_store_dword a7, v9, s[64:67], 0 offen sc1 // (op 1704) Store value




buffer_store_dword a0, v8, s[64:67], 0 offen sc1 // (op 274) Store value
buffer_store_dword a1, v8, s[64:67], 0 offen offset:128 sc1 // (op 274) Store value
buffer_store_dword a2, v8, s[64:67], 0 offen offset:256 sc1 // (op 274) Store value
buffer_store_dword a3, v8, s[64:67], 0 offen offset:384 sc1 // (op 274) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 278) 
v_cmp_eq_i32 vcc, v1, 0 // (op 282) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282 // (op 282) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282
s_mov_b32 s52, 1 // (op 280) call()
s_mov_b32 s55, 0 // (op 281) convert
s_mov_b32 s54, s16 // (op 281) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 281) 
s_add_u32 s84, s48, s54 // (op 281) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 281) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 281) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282 // (op 282) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282:
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_277 // (op 277) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_277
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_277:
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_277:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 293) 
s_mov_b32 s52, scc // (op 293) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 293) 
s_mov_b32 s53, s54 // (op 293) convert

s_sub_u32 s54, s53, 1 // (op 293) 

s_cmp_lt_u32 s19, s54 // (op 293) 
s_mov_b32 s53, scc // (op 293) 

s_and_b32 s54, s52, s53 // (op 293) 
s_cmp_lg_u32 s54, 0 // (op 293) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_293 // (op 293) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_293
s_mov_b32 s52, 0 // (op 259) call()
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s84, s54 // (op 258) convert

s_sub_u32 s54, s84, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 // (op 258) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_258:
s_add_u32 s53, s16, 1 // (op 3800) 
v_lshrrev_b32 v8, 5, v1 // (op 3800) 
s_add_u32 s54, s53, s52 // (op 3800) 
s_lshl_b32 s54, s54, 7 // (op 3800) 

v_lshlrev_b32 v9, 2, v8 // (op 3800) 

v_add_lshl_u32 v8, s54, v9, 5 // (op 3800) 


v_and_b32 v9, -32, v1 // (op 3800) 
v_sub_u32 v12, v1, v9 // (op 3800) 

v_add_lshl_u32 v9, v8, v12, 2 // (op 3800) 


s_lshl_b32 s53, s15, 14 // (op 3802) 
s_mov_b32 s64, s28 // (op 3802) reg expression
s_mov_b32 s65, s29 // (op 3802) reg expression
s_mov_b32 s66, s53 // (op 3802) reg expression
s_mov_b32 s67, 131072 // (op 3802) call()

RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_297:
s_add_u32 s53, s16, 1 // (op 295) 
s_add_u32 s54, s53, s52 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 295) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 295) 

s_add_u32 s84, s48, s54 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 295) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 295) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 297) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 // (op 297) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 if true)
s_barrier  // (op 301) 
v_cmp_eq_i32 vcc, v1, 0 // (op 300) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300 // (op 300) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300
s_mov_b32 s53, 0 // (op 298) call()
s_add_u32 s54, s16, 1 // (op 299) 
s_add_u32 s84, s54, s52 // (op 299) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 299) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 299) 

s_add_u32 s54, s48, s84 // (op 299) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 299) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 299) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300 // (op 300) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300:
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300:




buffer_load_dword v12, v9, s[64:67], 0 offen // (op 312) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:128 // (op 312) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:256 // (op 312) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:384 // (op 312) Load value
v_accvgpr_read v16, a0 // (op 315) reg expression
v_accvgpr_read v17, a1 // (op 315) reg expression
v_accvgpr_read v18, a2 // (op 315) reg expression
v_accvgpr_read v19, a3 // (op 315) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 317) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 317) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 317) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 317) 

v_accvgpr_write a0, v16 // (op 319) reg expression
v_accvgpr_write a1, v17 // (op 319) reg expression
v_accvgpr_write a2, v18 // (op 319) reg expression
v_accvgpr_write a3, v19 // (op 319) reg expression




v_mov_b32 v8, 4096 // (op 1744) 
v_add_i32 v12, v9, v8 // (op 1744) 

buffer_load_dword v14, v12, s[64:67], 0 offen // (op 1744) Load value

v_mov_b32 v8, 4224 // (op 1744) 
v_add_i32 v12, v9, v8 // (op 1744) 

buffer_load_dword v15, v12, s[64:67], 0 offen // (op 1744) Load value

v_mov_b32 v8, 4352 // (op 1744) 
v_add_i32 v12, v9, v8 // (op 1744) 

buffer_load_dword v16, v12, s[64:67], 0 offen // (op 1744) Load value

v_mov_b32 v8, 4480 // (op 1744) 
v_add_i32 v12, v9, v8 // (op 1744) 

buffer_load_dword v17, v12, s[64:67], 0 offen // (op 1744) Load value

v_accvgpr_read v18, a4 // (op 1745) reg expression
v_accvgpr_read v19, a5 // (op 1745) reg expression
v_accvgpr_read v20, a6 // (op 1745) reg expression
v_accvgpr_read v21, a7 // (op 1745) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v18, v18, v14 // (op 1746) 
s_waitcnt vmcnt(2)
v_add_f32 v19, v19, v15 // (op 1746) 
s_waitcnt vmcnt(1)
v_add_f32 v20, v20, v16 // (op 1746) 
s_waitcnt vmcnt(0)
v_add_f32 v21, v21, v17 // (op 1746) 

v_accvgpr_write a4, v18 // (op 1747) reg expression
v_accvgpr_write a5, v19 // (op 1747) reg expression
v_accvgpr_write a6, v20 // (op 1747) reg expression
v_accvgpr_write a7, v21 // (op 1747) reg expression




v_mov_b32 v8, 8192 // (op 1757) 
v_add_i32 v12, v9, v8 // (op 1757) 

buffer_load_dword v14, v12, s[64:67], 0 offen // (op 1757) Load value

v_mov_b32 v8, 8320 // (op 1757) 
v_add_i32 v12, v9, v8 // (op 1757) 

buffer_load_dword v15, v12, s[64:67], 0 offen // (op 1757) Load value

v_mov_b32 v8, 8448 // (op 1757) 
v_add_i32 v12, v9, v8 // (op 1757) 

buffer_load_dword v16, v12, s[64:67], 0 offen // (op 1757) Load value

v_mov_b32 v8, 8576 // (op 1757) 
v_add_i32 v12, v9, v8 // (op 1757) 

buffer_load_dword v17, v12, s[64:67], 0 offen // (op 1757) Load value

v_accvgpr_read v18, a8 // (op 1758) reg expression
v_accvgpr_read v19, a9 // (op 1758) reg expression
v_accvgpr_read v20, a10 // (op 1758) reg expression
v_accvgpr_read v21, a11 // (op 1758) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v18, v18, v14 // (op 1759) 
s_waitcnt vmcnt(2)
v_add_f32 v19, v19, v15 // (op 1759) 
s_waitcnt vmcnt(1)
v_add_f32 v20, v20, v16 // (op 1759) 
s_waitcnt vmcnt(0)
v_add_f32 v21, v21, v17 // (op 1759) 

v_accvgpr_write a8, v18 // (op 1760) reg expression
v_accvgpr_write a9, v19 // (op 1760) reg expression
v_accvgpr_write a10, v20 // (op 1760) reg expression
v_accvgpr_write a11, v21 // (op 1760) reg expression




v_mov_b32 v8, 12288 // (op 1770) 
v_add_i32 v12, v9, v8 // (op 1770) 

buffer_load_dword v14, v12, s[64:67], 0 offen // (op 1770) Load value

v_mov_b32 v8, 12416 // (op 1770) 
v_add_i32 v12, v9, v8 // (op 1770) 

buffer_load_dword v15, v12, s[64:67], 0 offen // (op 1770) Load value

v_mov_b32 v8, 12544 // (op 1770) 
v_add_i32 v12, v9, v8 // (op 1770) 

buffer_load_dword v16, v12, s[64:67], 0 offen // (op 1770) Load value

v_mov_b32 v8, 12672 // (op 1770) 
v_add_i32 v12, v9, v8 // (op 1770) 

buffer_load_dword v17, v12, s[64:67], 0 offen // (op 1770) Load value

v_accvgpr_read v18, a12 // (op 1771) reg expression
v_accvgpr_read v19, a13 // (op 1771) reg expression
v_accvgpr_read v20, a14 // (op 1771) reg expression
v_accvgpr_read v21, a15 // (op 1771) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v18, v18, v14 // (op 1772) 
s_waitcnt vmcnt(2)
v_add_f32 v19, v19, v15 // (op 1772) 
s_waitcnt vmcnt(1)
v_add_f32 v20, v20, v16 // (op 1772) 
s_waitcnt vmcnt(0)
v_add_f32 v21, v21, v17 // (op 1772) 

v_accvgpr_write a12, v18 // (op 1773) reg expression
v_accvgpr_write a13, v19 // (op 1773) reg expression
v_accvgpr_write a14, v20 // (op 1773) reg expression
v_accvgpr_write a15, v21 // (op 1773) reg expression

s_add_u32 s52, s52, 1 // (op 260) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s84, s54 // (op 258) convert

s_sub_u32 s54, s84, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_258 // (op 258) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_258 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_258:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_293 // (op 293) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_293
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_293:
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_293:

s_cmp_eq_i32 s18, 0 // (op 336) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336 // (op 336) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 108) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_108 // (op 108) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_108
v_accvgpr_read v12, a0 // (op 38) 
v_accvgpr_read v13, a1 // (op 38) 
v_accvgpr_read v14, a2 // (op 38) 
v_accvgpr_read v15, a3 // (op 38) 
v_mul_f32 v16, v7, v12 // (op 38) 
v_mul_f32 v17, v7, v13 // (op 38) 
v_mul_f32 v18, v7, v14 // (op 38) 
v_mul_f32 v19, v7, v15 // (op 38) 

v_lshrrev_b32 v12, 16, v16 // (op 44) 
v_lshrrev_b32 v13, 16, v17 // (op 44) 
v_lshrrev_b32 v14, 16, v18 // (op 44) 
v_lshrrev_b32 v15, 16, v19 // (op 44) 


v_pack_b32_f16 v8, v12, v13 // (op 46) 
v_pack_b32_f16 v9, v14, v15 // (op 46) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen // (op 46) Store value


v_accvgpr_read v12, a4 // (op 1807) 
v_accvgpr_read v13, a5 // (op 1807) 
v_accvgpr_read v14, a6 // (op 1807) 
v_accvgpr_read v15, a7 // (op 1807) 
v_mul_f32 v16, v7, v12 // (op 1807) 
v_mul_f32 v17, v7, v13 // (op 1807) 
v_mul_f32 v18, v7, v14 // (op 1807) 
v_mul_f32 v19, v7, v15 // (op 1807) 

v_lshrrev_b32 v12, 16, v16 // (op 1809) 
v_lshrrev_b32 v13, 16, v17 // (op 1809) 
v_lshrrev_b32 v14, 16, v18 // (op 1809) 
v_lshrrev_b32 v15, 16, v19 // (op 1809) 


v_pack_b32_f16 v8, v12, v13 // (op 1810) 
v_pack_b32_f16 v9, v14, v15 // (op 1810) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen offset:64 // (op 1810) Store value


v_accvgpr_read v12, a8 // (op 1827) 
v_accvgpr_read v13, a9 // (op 1827) 
v_accvgpr_read v14, a10 // (op 1827) 
v_accvgpr_read v15, a11 // (op 1827) 
v_mul_f32 v16, v7, v12 // (op 1827) 
v_mul_f32 v17, v7, v13 // (op 1827) 
v_mul_f32 v18, v7, v14 // (op 1827) 
v_mul_f32 v19, v7, v15 // (op 1827) 

v_lshrrev_b32 v12, 16, v16 // (op 1829) 
v_lshrrev_b32 v13, 16, v17 // (op 1829) 
v_lshrrev_b32 v14, 16, v18 // (op 1829) 
v_lshrrev_b32 v15, 16, v19 // (op 1829) 


v_pack_b32_f16 v8, v12, v13 // (op 1830) 
v_pack_b32_f16 v9, v14, v15 // (op 1830) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen offset:128 // (op 1830) Store value


v_accvgpr_read v12, a12 // (op 1847) 
v_accvgpr_read v13, a13 // (op 1847) 
v_accvgpr_read v14, a14 // (op 1847) 
v_accvgpr_read v15, a15 // (op 1847) 
v_mul_f32 v16, v7, v12 // (op 1847) 
v_mul_f32 v17, v7, v13 // (op 1847) 
v_mul_f32 v18, v7, v14 // (op 1847) 
v_mul_f32 v19, v7, v15 // (op 1847) 

v_lshrrev_b32 v12, 16, v16 // (op 1849) 
v_lshrrev_b32 v13, 16, v17 // (op 1849) 
v_lshrrev_b32 v14, 16, v18 // (op 1849) 
v_lshrrev_b32 v15, 16, v19 // (op 1849) 


v_pack_b32_f16 v8, v12, v13 // (op 1850) 
v_pack_b32_f16 v9, v14, v15 // (op 1850) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen offset:192 // (op 1850) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_108 // (op 108) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_108
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_108:



buffer_load_dwordx2 v[8:9], v10, s[56:59], 0 offen offset:192 // (op 1930) Load value
v_accvgpr_read v12, a12 // (op 1926) 
v_accvgpr_read v13, a13 // (op 1926) 
v_accvgpr_read v14, a14 // (op 1926) 
v_accvgpr_read v15, a15 // (op 1926) 
v_mul_f32 v16, v7, v12 // (op 1926) 
v_mul_f32 v17, v7, v13 // (op 1926) 
v_mul_f32 v18, v7, v14 // (op 1926) 
v_mul_f32 v19, v7, v15 // (op 1926) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 1931) 
v_lshrrev_b32 v13, 16, v8 // (op 1931) 
v_lshlrev_b32 v12, 16, v12 // (op 1931) 
v_lshlrev_b32 v13, 16, v13 // (op 1931) 
v_mul_f32 v20, v6, v12 // (op 1931) 
v_mul_f32 v21, v6, v13 // (op 1931) 
v_and_b32 v12, 65535, v9 // (op 1931) 
v_lshrrev_b32 v13, 16, v9 // (op 1931) 
v_lshlrev_b32 v12, 16, v12 // (op 1931) 
v_lshlrev_b32 v13, 16, v13 // (op 1931) 
v_mul_f32 v22, v6, v12 // (op 1931) 
v_mul_f32 v23, v6, v13 // (op 1931) 


v_add_f32 v12, v20, v16 // (op 1927) 
v_add_f32 v13, v21, v17 // (op 1927) 
v_add_f32 v14, v22, v18 // (op 1927) 
v_add_f32 v15, v23, v19 // (op 1927) 


v_lshrrev_b32 v16, 16, v12 // (op 1928) 
v_lshrrev_b32 v17, 16, v13 // (op 1928) 
v_lshrrev_b32 v18, 16, v14 // (op 1928) 
v_lshrrev_b32 v19, 16, v15 // (op 1928) 


v_pack_b32_f16 v8, v16, v17 // (op 1929) 
v_pack_b32_f16 v9, v18, v19 // (op 1929) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen offset:192 // (op 1929) Store value





buffer_load_dwordx2 v[8:9], v10, s[56:59], 0 offen offset:128 // (op 1910) Load value
v_accvgpr_read v12, a8 // (op 1906) 
v_accvgpr_read v13, a9 // (op 1906) 
v_accvgpr_read v14, a10 // (op 1906) 
v_accvgpr_read v15, a11 // (op 1906) 
v_mul_f32 v16, v7, v12 // (op 1906) 
v_mul_f32 v17, v7, v13 // (op 1906) 
v_mul_f32 v18, v7, v14 // (op 1906) 
v_mul_f32 v19, v7, v15 // (op 1906) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 1911) 
v_lshrrev_b32 v13, 16, v8 // (op 1911) 
v_lshlrev_b32 v12, 16, v12 // (op 1911) 
v_lshlrev_b32 v13, 16, v13 // (op 1911) 
v_mul_f32 v20, v6, v12 // (op 1911) 
v_mul_f32 v21, v6, v13 // (op 1911) 
v_and_b32 v12, 65535, v9 // (op 1911) 
v_lshrrev_b32 v13, 16, v9 // (op 1911) 
v_lshlrev_b32 v12, 16, v12 // (op 1911) 
v_lshlrev_b32 v13, 16, v13 // (op 1911) 
v_mul_f32 v22, v6, v12 // (op 1911) 
v_mul_f32 v23, v6, v13 // (op 1911) 


v_add_f32 v12, v20, v16 // (op 1907) 
v_add_f32 v13, v21, v17 // (op 1907) 
v_add_f32 v14, v22, v18 // (op 1907) 
v_add_f32 v15, v23, v19 // (op 1907) 


v_lshrrev_b32 v16, 16, v12 // (op 1908) 
v_lshrrev_b32 v17, 16, v13 // (op 1908) 
v_lshrrev_b32 v18, 16, v14 // (op 1908) 
v_lshrrev_b32 v19, 16, v15 // (op 1908) 


v_pack_b32_f16 v8, v16, v17 // (op 1909) 
v_pack_b32_f16 v9, v18, v19 // (op 1909) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen offset:128 // (op 1909) Store value





buffer_load_dwordx2 v[8:9], v10, s[56:59], 0 offen offset:64 // (op 1890) Load value
v_accvgpr_read v12, a4 // (op 1886) 
v_accvgpr_read v13, a5 // (op 1886) 
v_accvgpr_read v14, a6 // (op 1886) 
v_accvgpr_read v15, a7 // (op 1886) 
v_mul_f32 v16, v7, v12 // (op 1886) 
v_mul_f32 v17, v7, v13 // (op 1886) 
v_mul_f32 v18, v7, v14 // (op 1886) 
v_mul_f32 v19, v7, v15 // (op 1886) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 1891) 
v_lshrrev_b32 v13, 16, v8 // (op 1891) 
v_lshlrev_b32 v12, 16, v12 // (op 1891) 
v_lshlrev_b32 v13, 16, v13 // (op 1891) 
v_mul_f32 v20, v6, v12 // (op 1891) 
v_mul_f32 v21, v6, v13 // (op 1891) 
v_and_b32 v12, 65535, v9 // (op 1891) 
v_lshrrev_b32 v13, 16, v9 // (op 1891) 
v_lshlrev_b32 v12, 16, v12 // (op 1891) 
v_lshlrev_b32 v13, 16, v13 // (op 1891) 
v_mul_f32 v22, v6, v12 // (op 1891) 
v_mul_f32 v23, v6, v13 // (op 1891) 


v_add_f32 v12, v20, v16 // (op 1887) 
v_add_f32 v13, v21, v17 // (op 1887) 
v_add_f32 v14, v22, v18 // (op 1887) 
v_add_f32 v15, v23, v19 // (op 1887) 


v_lshrrev_b32 v16, 16, v12 // (op 1888) 
v_lshrrev_b32 v17, 16, v13 // (op 1888) 
v_lshrrev_b32 v18, 16, v14 // (op 1888) 
v_lshrrev_b32 v19, 16, v15 // (op 1888) 


v_pack_b32_f16 v8, v16, v17 // (op 1889) 
v_pack_b32_f16 v9, v18, v19 // (op 1889) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen offset:64 // (op 1889) Store value





buffer_load_dwordx2 v[8:9], v10, s[56:59], 0 offen // (op 121) Load value
v_accvgpr_read v12, a0 // (op 117) 
v_accvgpr_read v13, a1 // (op 117) 
v_accvgpr_read v14, a2 // (op 117) 
v_accvgpr_read v15, a3 // (op 117) 
v_mul_f32 v16, v7, v12 // (op 117) 
v_mul_f32 v17, v7, v13 // (op 117) 
v_mul_f32 v18, v7, v14 // (op 117) 
v_mul_f32 v19, v7, v15 // (op 117) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v8 // (op 122) 
v_lshrrev_b32 v13, 16, v8 // (op 122) 
v_lshlrev_b32 v12, 16, v12 // (op 122) 
v_lshlrev_b32 v13, 16, v13 // (op 122) 
v_mul_f32 v20, v6, v12 // (op 122) 
v_mul_f32 v21, v6, v13 // (op 122) 
v_and_b32 v12, 65535, v9 // (op 122) 
v_lshrrev_b32 v13, 16, v9 // (op 122) 
v_lshlrev_b32 v12, 16, v12 // (op 122) 
v_lshlrev_b32 v13, 16, v13 // (op 122) 
v_mul_f32 v22, v6, v12 // (op 122) 
v_mul_f32 v23, v6, v13 // (op 122) 


v_add_f32 v12, v20, v16 // (op 118) 
v_add_f32 v13, v21, v17 // (op 118) 
v_add_f32 v14, v22, v18 // (op 118) 
v_add_f32 v15, v23, v19 // (op 118) 


v_lshrrev_b32 v16, 16, v12 // (op 119) 
v_lshrrev_b32 v17, 16, v13 // (op 119) 
v_lshrrev_b32 v18, 16, v14 // (op 119) 
v_lshrrev_b32 v19, 16, v15 // (op 119) 


v_pack_b32_f16 v8, v16, v17 // (op 120) 
v_pack_b32_f16 v9, v18, v19 // (op 120) 


buffer_store_dwordx2 v[8:9], v11, s[60:63], 0 offen // (op 120) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_108:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336 // (op 336) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336:
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336:


s_mul_i32 s52, s8, s16 // (op 249) 
s_cmp_gt_u32 s9, s52 // (op 249) 
s_mov_b32 s53, scc // (op 249) 
s_sub_u32 s54, s9, s52 // (op 249) 

s_cmp_eq_i32 s53, 0 // (op 249) 
s_cselect_b32 s52, 0, s54 // (op 249) 


s_cmp_lt_u32 s8, s52 // (op 249) 
s_mov_b32 s53, scc // (op 249) 
s_cmp_eq_i32 s53, 0 // (op 249) 
s_cselect_b32 s54, s52, s8 // (op 249) 


s_cmp_lt_u32 s0, s54 // (op 249) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_249 // (op 249) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_249 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_249:











s_mov_b32 s0, 0 // (op 390) call()
s_mov_b32 s1, 0 // (op 391) call()
s_mov_b32 s0, 0 // (op 405) call()
s_mul_i32 s15, s50, s16 // (op 404) 
s_cmp_gt_u32 s51, s15 // (op 404) 
s_mov_b32 s6, scc // (op 404) 
s_sub_u32 s7, s51, s15 // (op 404) 

s_cmp_eq_i32 s6, 0 // (op 404) 
s_cselect_b32 s8, 0, s7 // (op 404) 


s_cmp_lt_u32 s50, s8 // (op 404) 
s_mov_b32 s15, scc // (op 404) 
s_cmp_eq_i32 s15, 0 // (op 404) 
s_cselect_b32 s6, s8, s50 // (op 404) 


s_cmp_lt_u32 s0, s6 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 // (op 404) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_404:
v_lshrrev_b32 v0, 7, v1 // (op 3598) 
s_mul_i32 s15, s16, s50 // (op 3598) 
s_add_u32 s15, s15, s0 // (op 3598) 
v_and_b32 v5, 63, v1 // (op 3598) 
s_add_u32 s6, s15, s9 // (op 3598) 

v_lshrrev_b32 v8, 4, v5 // (op 3598) 
s_mul_hi_u32 s15, s6, s13 // (op 3598) 
v_lshlrev_b32 v9, 2, v8 // (op 3598) 

s_sub_u32 s7, s6, s15 // (op 3598) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 3598) 


s_lshr_b32 s8, s7, 1 // (op 3598) 

v_lshrrev_b32 v0, 2, v5 // (op 3598) 

s_add_u32 s7, s8, s15 // (op 3598) 


v_and_b32 v5, 3, v0 // (op 3598) 

s_lshr_b32 s8, s7, s11 // (op 3598) 

v_and_b32 v0, 3, v1 // (op 3598) 
s_and_b32 s7, s11, 2147483648 // (op 3598) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 3598) 


s_cmp_eq_i32 s7, 0 // (op 3598) 
s_mov_b32 s15, scc // (op 3598) 

v_lshrrev_b32 v0, 6, v1 // (op 3598) 
s_cmp_eq_i32 s15, 0 // (op 3598) 
s_cselect_b32 s7, s6, s8 // (op 3598) 



v_and_b32 v5, 1, v0 // (op 3598) 

s_mul_hi_u32 s6, s7, s12 // (op 3598) 
v_lshl_add_u32 v0, v5, 4, v9 // (op 3598) 


s_sub_u32 s8, s7, s6 // (op 3598) 
s_lshr_b32 s15, s8, 1 // (op 3598) 

s_add_u32 s8, s15, s6 // (op 3598) 


s_lshr_b32 s6, s8, s10 // (op 3598) 

s_and_b32 s8, s10, 2147483648 // (op 3598) 
s_cmp_eq_i32 s8, 0 // (op 3598) 
s_mov_b32 s15, scc // (op 3598) 

s_cmp_eq_i32 s15, 0 // (op 3598) 
s_cselect_b32 s8, s7, s6 // (op 3598) 


s_mul_i32 s6, s8, s14 // (op 3598) 
v_lshl_add_u32 v5, s8, 7, v8 // (op 3598) 


s_sub_u32 s8, s7, s6 // (op 3598) 


v_lshl_add_u32 v8, s8, 5, v0 // (op 3598) 


v_mul_lo_u32 v0, v8, s4 // (op 3598) 

v_add_lshl_u32 v8, v5, v0, 1 // (op 3598) 


s_mul_i32 s19, s4, s23 // (op 3600) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 3600) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 3600) least significant
s_mul_hi_u32 s6, s4, s22 // (op 3600) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3600) most significant: sum
s_add_u32 s19, s19, s6 // (op 3600) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 3600) 

s_mov_b32 s56, s20 // (op 3600) reg expression
s_mov_b32 s57, s21 // (op 3600) reg expression
s_mov_b32 s58, s28 // (op 3600) reg expression
s_mov_b32 s59, 131072 // (op 3600) call()

v_lshrrev_b32 v0, 7, v1 // (op 3698) 
s_mul_i32 s15, s16, s50 // (op 3698) 
s_add_u32 s15, s15, s0 // (op 3698) 
v_and_b32 v5, 63, v1 // (op 3698) 
s_add_u32 s6, s15, s9 // (op 3698) 

v_lshrrev_b32 v9, 4, v5 // (op 3698) 
s_mul_hi_u32 s15, s6, s13 // (op 3698) 
v_lshlrev_b32 v10, 2, v9 // (op 3698) 

s_sub_u32 s7, s6, s15 // (op 3698) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 3698) 


s_lshr_b32 s8, s7, 1 // (op 3698) 

v_lshrrev_b32 v0, 2, v5 // (op 3698) 

s_add_u32 s7, s8, s15 // (op 3698) 


v_and_b32 v5, 3, v0 // (op 3698) 

s_lshr_b32 s8, s7, s11 // (op 3698) 

v_and_b32 v0, 3, v1 // (op 3698) 
s_and_b32 s7, s11, 2147483648 // (op 3698) 
v_lshl_add_u32 v10, v5, 2, v0 // (op 3698) 


s_cmp_eq_i32 s7, 0 // (op 3698) 
s_mov_b32 s15, scc // (op 3698) 

v_lshrrev_b32 v0, 6, v1 // (op 3698) 
s_cmp_eq_i32 s15, 0 // (op 3698) 
s_cselect_b32 s7, s6, s8 // (op 3698) 



v_and_b32 v5, 1, v0 // (op 3698) 

s_mul_hi_u32 s6, s7, s12 // (op 3698) 
v_lshl_add_u32 v0, v5, 4, v10 // (op 3698) 


s_sub_u32 s8, s7, s6 // (op 3698) 
s_lshr_b32 s15, s8, 1 // (op 3698) 

s_add_u32 s8, s15, s6 // (op 3698) 


s_lshr_b32 s6, s8, s10 // (op 3698) 

s_and_b32 s8, s10, 2147483648 // (op 3698) 
s_cmp_eq_i32 s8, 0 // (op 3698) 
s_mov_b32 s15, scc // (op 3698) 

s_cmp_eq_i32 s15, 0 // (op 3698) 
s_cselect_b32 s8, s7, s6 // (op 3698) 


s_mul_i32 s6, s8, s14 // (op 3698) 
v_lshl_add_u32 v5, s8, 7, v9 // (op 3698) 


s_sub_u32 s8, s7, s6 // (op 3698) 


v_lshl_add_u32 v9, s8, 5, v0 // (op 3698) 


v_mul_lo_u32 v0, v9, s24 // (op 3698) 

v_add_lshl_u32 v9, v5, v0, 1 // (op 3698) 


s_mul_i32 s19, s24, s23 // (op 3700) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 3700) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 3700) least significant
s_mul_hi_u32 s6, s24, s22 // (op 3700) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3700) most significant: sum
s_add_u32 s19, s19, s6 // (op 3700) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 3700) 

s_mov_b32 s60, s26 // (op 3700) reg expression
s_mov_b32 s61, s27 // (op 3700) reg expression
s_mov_b32 s62, s28 // (op 3700) reg expression
s_mov_b32 s63, 131072 // (op 3700) call()

v_accvgpr_write a0, 0.00000 // (op 157) call()
v_accvgpr_write a1, 0.00000 // (op 157) call()
v_accvgpr_write a2, 0.00000 // (op 157) call()
v_accvgpr_write a3, 0.00000 // (op 157) call()
v_accvgpr_write a4, 0.00000 // (op 520) call()
v_accvgpr_write a5, 0.00000 // (op 520) call()
v_accvgpr_write a6, 0.00000 // (op 520) call()
v_accvgpr_write a7, 0.00000 // (op 520) call()
v_accvgpr_write a8, 0.00000 // (op 642) call()
v_accvgpr_write a9, 0.00000 // (op 642) call()
v_accvgpr_write a10, 0.00000 // (op 642) call()
v_accvgpr_write a11, 0.00000 // (op 642) call()
v_accvgpr_write a12, 0.00000 // (op 764) call()
v_accvgpr_write a13, 0.00000 // (op 764) call()
v_accvgpr_write a14, 0.00000 // (op 764) call()
v_accvgpr_write a15, 0.00000 // (op 764) call()
v_lshrrev_b32 v0, 7, v1 // (op 3085) 
s_mul_i32 s15, s16, s50 // (op 3085) 
s_add_u32 s15, s15, s0 // (op 3085) 
v_and_b32 v5, 15, v1 // (op 3085) 
s_add_u32 s6, s15, s9 // (op 3085) 

v_lshl_add_u32 v10, v0, 4, v5 // (op 3085) 


s_mul_hi_u32 s15, s6, s13 // (op 3085) 
v_and_b32 v0, 63, v1 // (op 3085) 
s_sub_u32 s7, s6, s15 // (op 3085) 
v_lshrrev_b32 v5, 4, v0 // (op 3085) 

s_lshr_b32 s8, s7, 1 // (op 3085) 

v_lshlrev_b32 v0, 5, v5 // (op 3085) 

s_add_u32 s7, s8, s15 // (op 3085) 


s_lshr_b32 s8, s7, s11 // (op 3085) 

s_and_b32 s7, s11, 2147483648 // (op 3085) 
s_cmp_eq_i32 s7, 0 // (op 3085) 
s_mov_b32 s15, scc // (op 3085) 

s_cmp_eq_i32 s15, 0 // (op 3085) 
s_cselect_b32 s7, s6, s8 // (op 3085) 


s_mul_hi_u32 s8, s7, s12 // (op 3085) 
s_sub_u32 s15, s7, s8 // (op 3085) 
s_lshr_b32 s18, s15, 1 // (op 3085) 

s_add_u32 s15, s18, s8 // (op 3085) 


s_lshr_b32 s8, s15, s10 // (op 3085) 

s_and_b32 s15, s10, 2147483648 // (op 3085) 
s_cmp_eq_i32 s15, 0 // (op 3085) 
s_mov_b32 s18, scc // (op 3085) 

s_cmp_eq_i32 s18, 0 // (op 3085) 
s_cselect_b32 s15, s7, s8 // (op 3085) 


v_lshl_add_u32 v5, s15, 7, v10 // (op 3085) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3085) 
s_mul_i32 s8, s7, s18 // (op 3085) 


s_sub_u32 s7, s6, s8 // (op 3085) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 3085) 


v_mov_b32 v13, 0 // (op 3085) convert
v_mov_b32 v12, v10 // (op 3085) convert

v_mul_lo_u32 v11, v5, s31 // (op 3085) most significant: low of low * high
v_mul_hi_u32 v0, v5, s30 // (op 3085) most significant: high of low * low
v_mul_lo_u32 v10, v5, s30 // (op 3085) least significant: low of low * low
v_add_u32 v11, v11, v0 // (op 3085) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v12 // (op 3085) least significant half
v_addc_co_u32 v11, s[18:19], v11, v13, s[18:19] // (op 3085) most significant half



v_lshlrev_b64 v[12:13], 2, v[10:11] // (op 3085) 

v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 3085) 

v_mov_b32 v0, v10 // (op 3085) convert

s_mul_i32 s19, s30, s35 // (op 3087) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3087) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3087) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3087) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3087) most significant: sum
s_add_u32 s19, s19, s6 // (op 3087) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3087) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 3087) 

s_mov_b32 s68, s32 // (op 3087) reg expression
s_mov_b32 s69, s33 // (op 3087) reg expression
s_mov_b32 s70, s18 // (op 3087) reg expression
s_mov_b32 s71, 131072 // (op 3087) call()

v_lshrrev_b32 v5, 7, v1 // (op 3106) 
s_mul_i32 s15, s16, s50 // (op 3106) 
s_add_u32 s15, s15, s0 // (op 3106) 
v_and_b32 v10, 15, v1 // (op 3106) 
s_add_u32 s6, s15, s9 // (op 3106) 

v_lshl_add_u32 v11, v5, 4, v10 // (op 3106) 


s_mul_hi_u32 s15, s6, s13 // (op 3106) 
v_and_b32 v5, 63, v1 // (op 3106) 
s_sub_u32 s7, s6, s15 // (op 3106) 
v_lshrrev_b32 v10, 4, v5 // (op 3106) 

s_lshr_b32 s8, s7, 1 // (op 3106) 

v_lshlrev_b32 v5, 5, v10 // (op 3106) 

s_add_u32 s7, s8, s15 // (op 3106) 


s_lshr_b32 s8, s7, s11 // (op 3106) 

s_and_b32 s7, s11, 2147483648 // (op 3106) 
s_cmp_eq_i32 s7, 0 // (op 3106) 
s_mov_b32 s15, scc // (op 3106) 

s_cmp_eq_i32 s15, 0 // (op 3106) 
s_cselect_b32 s7, s6, s8 // (op 3106) 


s_mul_hi_u32 s8, s7, s12 // (op 3106) 
s_sub_u32 s15, s7, s8 // (op 3106) 
s_lshr_b32 s18, s15, 1 // (op 3106) 

s_add_u32 s15, s18, s8 // (op 3106) 


s_lshr_b32 s8, s15, s10 // (op 3106) 

s_and_b32 s15, s10, 2147483648 // (op 3106) 
s_cmp_eq_i32 s15, 0 // (op 3106) 
s_mov_b32 s18, scc // (op 3106) 

s_cmp_eq_i32 s18, 0 // (op 3106) 
s_cselect_b32 s15, s7, s8 // (op 3106) 


v_lshl_add_u32 v10, s15, 7, v11 // (op 3106) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3106) 
s_mul_i32 s8, s7, s18 // (op 3106) 


s_sub_u32 s7, s6, s8 // (op 3106) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 3106) 


v_mov_b32 v13, 0 // (op 3106) convert
v_mov_b32 v12, v11 // (op 3106) convert

v_mul_lo_u32 v15, v10, s31 // (op 3106) most significant: low of low * high
v_mul_hi_u32 v5, v10, s30 // (op 3106) most significant: high of low * low
v_mul_lo_u32 v14, v10, s30 // (op 3106) least significant: low of low * low
v_add_u32 v15, v15, v5 // (op 3106) most significant: sum

v_add_co_u32 v14, s[18:19], v14, v12 // (op 3106) least significant half
v_addc_co_u32 v15, s[18:19], v15, v13, s[18:19] // (op 3106) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 3106) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 3106) 

v_mov_b32 v5, v12 // (op 3106) convert

s_mul_i32 s19, s30, s35 // (op 3108) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3108) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3108) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3108) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3108) most significant: sum
s_add_u32 s19, s19, s6 // (op 3108) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3108) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 3108) 

s_mov_b32 s68, s32 // (op 3108) reg expression
s_mov_b32 s69, s33 // (op 3108) reg expression
s_mov_b32 s70, s18 // (op 3108) reg expression
s_mov_b32 s71, 131072 // (op 3108) call()

v_lshrrev_b32 v10, 6, v1 // (op 3234) 
s_mul_i32 s15, s16, s50 // (op 3234) 
s_add_u32 s15, s15, s0 // (op 3234) 
v_and_b32 v11, 1, v10 // (op 3234) 

s_add_u32 s6, s15, s9 // (op 3234) 

v_and_b32 v10, 15, v1 // (op 3234) 
s_mul_hi_u32 s15, s6, s13 // (op 3234) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 3234) 


s_sub_u32 s7, s6, s15 // (op 3234) 
v_and_b32 v10, 48, v1 // (op 3234) 
s_lshr_b32 s8, s7, 1 // (op 3234) 

s_add_u32 s7, s8, s15 // (op 3234) 


s_lshr_b32 s8, s7, s11 // (op 3234) 

s_and_b32 s7, s11, 2147483648 // (op 3234) 
s_cmp_eq_i32 s7, 0 // (op 3234) 
s_mov_b32 s15, scc // (op 3234) 

s_cmp_eq_i32 s15, 0 // (op 3234) 
s_cselect_b32 s7, s6, s8 // (op 3234) 


s_mul_hi_u32 s8, s7, s12 // (op 3234) 
s_sub_u32 s15, s7, s8 // (op 3234) 
s_lshr_b32 s18, s15, 1 // (op 3234) 

s_add_u32 s15, s18, s8 // (op 3234) 


s_lshr_b32 s8, s15, s10 // (op 3234) 

s_and_b32 s15, s10, 2147483648 // (op 3234) 
s_cmp_eq_i32 s15, 0 // (op 3234) 
s_mov_b32 s18, scc // (op 3234) 

s_cmp_eq_i32 s18, 0 // (op 3234) 
s_cselect_b32 s15, s7, s8 // (op 3234) 


s_mul_i32 s8, s15, s14 // (op 3234) 

s_sub_u32 s15, s7, s8 // (op 3234) 

v_lshl_add_u32 v11, s15, 5, v12 // (op 3234) 


v_mul_lo_u32 v12, v11, s36 // (op 3234) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3234) 
s_mul_i32 s8, s7, s18 // (op 3234) 


s_sub_u32 s7, s6, s8 // (op 3234) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 3234) 


v_add_u32 v10, v11, v12 // (op 3234) 


s_mul_i32 s19, s36, s23 // (op 3236) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3236) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3236) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3236) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3236) most significant: sum
s_add_u32 s19, s19, s6 // (op 3236) most significant: sum


s_mov_b32 s72, s38 // (op 3236) reg expression
s_mov_b32 s73, s39 // (op 3236) reg expression
s_mov_b32 s74, s18 // (op 3236) reg expression
s_mov_b32 s75, 131072 // (op 3236) call()

v_lshrrev_b32 v11, 6, v1 // (op 3253) 
s_mul_i32 s15, s16, s50 // (op 3253) 
s_add_u32 s15, s15, s0 // (op 3253) 
v_and_b32 v12, 1, v11 // (op 3253) 

s_add_u32 s6, s15, s9 // (op 3253) 

v_and_b32 v11, 15, v1 // (op 3253) 
s_mul_hi_u32 s15, s6, s13 // (op 3253) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 3253) 


s_sub_u32 s7, s6, s15 // (op 3253) 
v_and_b32 v11, 48, v1 // (op 3253) 
s_lshr_b32 s8, s7, 1 // (op 3253) 

s_add_u32 s7, s8, s15 // (op 3253) 


s_lshr_b32 s8, s7, s11 // (op 3253) 

s_and_b32 s7, s11, 2147483648 // (op 3253) 
s_cmp_eq_i32 s7, 0 // (op 3253) 
s_mov_b32 s15, scc // (op 3253) 

s_cmp_eq_i32 s15, 0 // (op 3253) 
s_cselect_b32 s7, s6, s8 // (op 3253) 


s_mul_hi_u32 s8, s7, s12 // (op 3253) 
s_sub_u32 s15, s7, s8 // (op 3253) 
s_lshr_b32 s18, s15, 1 // (op 3253) 

s_add_u32 s15, s18, s8 // (op 3253) 


s_lshr_b32 s8, s15, s10 // (op 3253) 

s_and_b32 s15, s10, 2147483648 // (op 3253) 
s_cmp_eq_i32 s15, 0 // (op 3253) 
s_mov_b32 s18, scc // (op 3253) 

s_cmp_eq_i32 s18, 0 // (op 3253) 
s_cselect_b32 s15, s7, s8 // (op 3253) 


s_mul_i32 s8, s15, s14 // (op 3253) 

s_sub_u32 s15, s7, s8 // (op 3253) 

v_lshl_add_u32 v12, s15, 5, v13 // (op 3253) 


v_mul_lo_u32 v13, v12, s36 // (op 3253) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3253) 
s_mul_i32 s8, s7, s18 // (op 3253) 


s_sub_u32 s7, s6, s8 // (op 3253) 


v_lshl_add_u32 v12, s7, 7, v11 // (op 3253) 


v_add_u32 v11, v12, v13 // (op 3253) 


s_mul_i32 s19, s36, s23 // (op 3255) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3255) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3255) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3255) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3255) most significant: sum
s_add_u32 s19, s19, s6 // (op 3255) most significant: sum


s_mov_b32 s72, s38 // (op 3255) reg expression
s_mov_b32 s73, s39 // (op 3255) reg expression
s_mov_b32 s74, s18 // (op 3255) reg expression
s_mov_b32 s75, 131072 // (op 3255) call()

v_lshrrev_b32 v12, 2, v1 // (op 3362) 
s_mul_i32 s15, s16, s50 // (op 3362) 
s_add_u32 s15, s15, s0 // (op 3362) 
v_lshlrev_b32 v13, 1, v12 // (op 3362) 

s_add_u32 s6, s15, s9 // (op 3362) 

v_and_b32 v12, -4, v1 // (op 3362) 
s_mul_hi_u32 s15, s6, s13 // (op 3362) 
v_sub_u32 v14, v1, v12 // (op 3362) 

s_sub_u32 s7, s6, s15 // (op 3362) 
s_lshr_b32 s8, s7, 1 // (op 3362) 

s_add_u32 s7, s8, s15 // (op 3362) 


s_lshr_b32 s8, s7, s11 // (op 3362) 

s_and_b32 s7, s11, 2147483648 // (op 3362) 
s_cmp_eq_i32 s7, 0 // (op 3362) 
s_mov_b32 s15, scc // (op 3362) 

s_cmp_eq_i32 s15, 0 // (op 3362) 
s_cselect_b32 s7, s6, s8 // (op 3362) 


s_mul_hi_u32 s8, s7, s12 // (op 3362) 
s_sub_u32 s15, s7, s8 // (op 3362) 
s_lshr_b32 s18, s15, 1 // (op 3362) 

s_add_u32 s15, s18, s8 // (op 3362) 


s_lshr_b32 s8, s15, s10 // (op 3362) 

s_and_b32 s15, s10, 2147483648 // (op 3362) 
s_cmp_eq_i32 s15, 0 // (op 3362) 
s_mov_b32 s18, scc // (op 3362) 

s_cmp_eq_i32 s18, 0 // (op 3362) 
s_cselect_b32 s15, s7, s8 // (op 3362) 


v_lshl_add_u32 v12, s15, 7, v13 // (op 3362) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3362) 
s_mul_i32 s8, s7, s18 // (op 3362) 


s_sub_u32 s7, s6, s8 // (op 3362) 


v_lshl_add_u32 v13, s7, 2, v14 // (op 3362) 


v_mul_lo_u32 v14, v12, s40 // (op 3362) 
v_add_u32 v14, v14, v13 // (op 3362) 


s_mul_i32 s19, s40, s35 // (op 3364) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3364) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3364) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3364) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3364) most significant: sum
s_add_u32 s19, s19, s6 // (op 3364) most significant: sum


s_mov_b32 s76, s42 // (op 3364) reg expression
s_mov_b32 s77, s43 // (op 3364) reg expression
s_mov_b32 s78, s18 // (op 3364) reg expression
s_mov_b32 s79, 131072 // (op 3364) call()

v_lshrrev_b32 v12, 2, v1 // (op 3381) 
s_mul_i32 s15, s16, s50 // (op 3381) 
s_add_u32 s15, s15, s0 // (op 3381) 
v_lshlrev_b32 v13, 1, v12 // (op 3381) 

s_add_u32 s6, s15, s9 // (op 3381) 

v_and_b32 v12, -4, v1 // (op 3381) 
s_mul_hi_u32 s15, s6, s13 // (op 3381) 
v_sub_u32 v15, v1, v12 // (op 3381) 

s_sub_u32 s7, s6, s15 // (op 3381) 
s_lshr_b32 s8, s7, 1 // (op 3381) 

s_add_u32 s7, s8, s15 // (op 3381) 


s_lshr_b32 s8, s7, s11 // (op 3381) 

s_and_b32 s7, s11, 2147483648 // (op 3381) 
s_cmp_eq_i32 s7, 0 // (op 3381) 
s_mov_b32 s15, scc // (op 3381) 

s_cmp_eq_i32 s15, 0 // (op 3381) 
s_cselect_b32 s7, s6, s8 // (op 3381) 


s_mul_hi_u32 s8, s7, s12 // (op 3381) 
s_sub_u32 s15, s7, s8 // (op 3381) 
s_lshr_b32 s18, s15, 1 // (op 3381) 

s_add_u32 s15, s18, s8 // (op 3381) 


s_lshr_b32 s8, s15, s10 // (op 3381) 

s_and_b32 s15, s10, 2147483648 // (op 3381) 
s_cmp_eq_i32 s15, 0 // (op 3381) 
s_mov_b32 s18, scc // (op 3381) 

s_cmp_eq_i32 s18, 0 // (op 3381) 
s_cselect_b32 s15, s7, s8 // (op 3381) 


v_lshl_add_u32 v12, s15, 7, v13 // (op 3381) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3381) 
s_mul_i32 s8, s7, s18 // (op 3381) 


s_sub_u32 s7, s6, s8 // (op 3381) 


v_lshl_add_u32 v13, s7, 2, v15 // (op 3381) 


v_mul_lo_u32 v15, v12, s40 // (op 3381) 
v_add_u32 v15, v15, v13 // (op 3381) 


s_mul_i32 s19, s40, s35 // (op 3383) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3383) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3383) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3383) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3383) most significant: sum
s_add_u32 s19, s19, s6 // (op 3383) most significant: sum


s_mov_b32 s76, s42 // (op 3383) reg expression
s_mov_b32 s77, s43 // (op 3383) reg expression
s_mov_b32 s78, s18 // (op 3383) reg expression
s_mov_b32 s79, 131072 // (op 3383) call()

v_lshrrev_b32 v12, 6, v1 // (op 3490) 
s_mul_i32 s15, s16, s50 // (op 3490) 
s_add_u32 s15, s15, s0 // (op 3490) 
v_and_b32 v13, 1, v12 // (op 3490) 

s_add_u32 s6, s15, s9 // (op 3490) 

v_and_b32 v12, 15, v1 // (op 3490) 
s_mul_hi_u32 s15, s6, s13 // (op 3490) 
v_lshl_add_u32 v16, v13, 4, v12 // (op 3490) 


s_sub_u32 s7, s6, s15 // (op 3490) 
v_and_b32 v12, 63, v1 // (op 3490) 
s_lshr_b32 s8, s7, 1 // (op 3490) 

v_lshrrev_b32 v13, 4, v12 // (op 3490) 

s_add_u32 s7, s8, s15 // (op 3490) 


s_lshr_b32 s8, s7, s11 // (op 3490) 

s_and_b32 s7, s11, 2147483648 // (op 3490) 
s_cmp_eq_i32 s7, 0 // (op 3490) 
s_mov_b32 s15, scc // (op 3490) 

s_cmp_eq_i32 s15, 0 // (op 3490) 
s_cselect_b32 s7, s6, s8 // (op 3490) 


s_mul_hi_u32 s8, s7, s12 // (op 3490) 
s_sub_u32 s15, s7, s8 // (op 3490) 
s_lshr_b32 s18, s15, 1 // (op 3490) 

s_add_u32 s15, s18, s8 // (op 3490) 


s_lshr_b32 s8, s15, s10 // (op 3490) 

s_and_b32 s15, s10, 2147483648 // (op 3490) 
s_cmp_eq_i32 s15, 0 // (op 3490) 
s_mov_b32 s18, scc // (op 3490) 

s_cmp_eq_i32 s18, 0 // (op 3490) 
s_cselect_b32 s15, s7, s8 // (op 3490) 


s_mul_i32 s8, s15, s14 // (op 3490) 

s_sub_u32 s15, s7, s8 // (op 3490) 

v_lshl_add_u32 v12, s15, 5, v16 // (op 3490) 


v_mul_lo_u32 v16, v12, s44 // (op 3490) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3490) 
s_mul_i32 s8, s7, s18 // (op 3490) 


s_sub_u32 s7, s6, s8 // (op 3490) 


v_lshl_add_u32 v12, s7, 2, v13 // (op 3490) 


v_add_u32 v13, v12, v16 // (op 3490) 


s_mul_i32 s19, s44, s23 // (op 3492) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3492) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3492) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3492) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3492) most significant: sum
s_add_u32 s19, s19, s6 // (op 3492) most significant: sum


s_mov_b32 s80, s46 // (op 3492) reg expression
s_mov_b32 s81, s47 // (op 3492) reg expression
s_mov_b32 s82, s18 // (op 3492) reg expression
s_mov_b32 s83, 131072 // (op 3492) call()

v_lshrrev_b32 v12, 6, v1 // (op 3509) 
s_mul_i32 s15, s16, s50 // (op 3509) 
s_add_u32 s15, s15, s0 // (op 3509) 
v_and_b32 v16, 1, v12 // (op 3509) 

s_add_u32 s6, s15, s9 // (op 3509) 

v_and_b32 v12, 15, v1 // (op 3509) 
s_mul_hi_u32 s15, s6, s13 // (op 3509) 
v_lshl_add_u32 v17, v16, 4, v12 // (op 3509) 


s_sub_u32 s7, s6, s15 // (op 3509) 
v_and_b32 v12, 63, v1 // (op 3509) 
s_lshr_b32 s8, s7, 1 // (op 3509) 

v_lshrrev_b32 v16, 4, v12 // (op 3509) 

s_add_u32 s7, s8, s15 // (op 3509) 


s_lshr_b32 s8, s7, s11 // (op 3509) 

s_and_b32 s7, s11, 2147483648 // (op 3509) 
s_cmp_eq_i32 s7, 0 // (op 3509) 
s_mov_b32 s15, scc // (op 3509) 

s_cmp_eq_i32 s15, 0 // (op 3509) 
s_cselect_b32 s7, s6, s8 // (op 3509) 


s_mul_hi_u32 s8, s7, s12 // (op 3509) 
s_sub_u32 s15, s7, s8 // (op 3509) 
s_lshr_b32 s18, s15, 1 // (op 3509) 

s_add_u32 s15, s18, s8 // (op 3509) 


s_lshr_b32 s8, s15, s10 // (op 3509) 

s_and_b32 s15, s10, 2147483648 // (op 3509) 
s_cmp_eq_i32 s15, 0 // (op 3509) 
s_mov_b32 s18, scc // (op 3509) 

s_cmp_eq_i32 s18, 0 // (op 3509) 
s_cselect_b32 s15, s7, s8 // (op 3509) 


s_mul_i32 s8, s15, s14 // (op 3509) 

s_sub_u32 s15, s7, s8 // (op 3509) 

v_lshl_add_u32 v12, s15, 5, v17 // (op 3509) 


v_mul_lo_u32 v17, v12, s44 // (op 3509) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3509) 
s_mul_i32 s8, s7, s18 // (op 3509) 


s_sub_u32 s7, s6, s8 // (op 3509) 


v_lshl_add_u32 v12, s7, 2, v16 // (op 3509) 


v_add_u32 v16, v12, v17 // (op 3509) 


s_mul_i32 s19, s44, s23 // (op 3511) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3511) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3511) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3511) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3511) most significant: sum
s_add_u32 s19, s19, s6 // (op 3511) most significant: sum


s_mov_b32 s80, s46 // (op 3511) reg expression
s_mov_b32 s81, s47 // (op 3511) reg expression
s_mov_b32 s82, s18 // (op 3511) reg expression
s_mov_b32 s83, 131072 // (op 3511) call()




buffer_load_ubyte v12, v13, s[80:83], 0 offen offset: 0 // (op 2678) Load value


v_mov_b32 v17, v0 // (op 2684) reg expression

buffer_load_dwordx4 v[18:21], v17, s[68:71], 0 offen // (op 2684) Load value




buffer_load_dwordx4 v[22:25], v10, s[72:75], 0 offen // (op 2690) Load value
buffer_load_dwordx4 v[26:29], v10, s[72:75], 0 offen offset:64 // (op 2690) Load value


s_lshl_b64 s[18:19], s[30:31], 5 // (op 2696) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2696) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2696) 

v_add_u32 v17, v0, s18 // (op 2696) 


buffer_load_dwordx4 v[30:33], v17, s[68:71], 0 offen // (op 2696) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 2703) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2703) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2703) 

s_lshl_b32 s15, s18, 1 // (op 2703) 

v_add_u32 v17, v0, s15 // (op 2703) 


buffer_load_dwordx4 v[34:37], v17, s[68:71], 0 offen // (op 2703) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 2710) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2710) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2710) 

s_mul_i32 s15, 3, s18 // (op 2710) 

v_add_u32 v17, v0, s15 // (op 2710) 


buffer_load_dwordx4 v[38:41], v17, s[68:71], 0 offen // (op 2710) Load value


v_mov_b64 v[42:43], s[40:41] // (op 2715) convert

v_mov_b32 v17, v14 // (op 2715) 
buffer_load_ubyte v44, v17, s[76:79], 0 offen offset: 0 // (op 2715) Load value
v_add_u32 v17, v17, v42 // (op 2715) 
buffer_load_ubyte v45, v17, s[76:79], 0 offen offset: 0 // (op 2715) Load value





buffer_load_ubyte v17, v13, s[80:83], 0 offen offset:4 // (op 2719) Load value



v_add_u32 v13, 64, v0 // (op 2725) 

buffer_load_dwordx4 v[46:49], v13, s[68:71], 0 offen // (op 2725) Load value




buffer_load_dwordx4 v[50:53], v10, s[72:75], 0 offen offset:128 // (op 2731) Load value
buffer_load_dwordx4 v[54:57], v10, s[72:75], 0 offen offset:192 // (op 2731) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 2737) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2737) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2737) 

s_add_u32 s15, s18, 64 // (op 2737) 

v_add_u32 v10, v0, s15 // (op 2737) 


buffer_load_dwordx4 v[58:61], v10, s[68:71], 0 offen // (op 2737) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 2744) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2744) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2744) 

s_lshl_b32 s15, s18, 1 // (op 2744) 
s_add_u32 s15, s15, 64 // (op 2744) 

v_add_u32 v10, v0, s15 // (op 2744) 


buffer_load_dwordx4 v[62:65], v10, s[68:71], 0 offen // (op 2744) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 2751) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2751) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2751) 

s_mul_i32 s15, 3, s18 // (op 2751) 
s_add_u32 s15, s15, 64 // (op 2751) 

v_add_u32 v10, v0, s15 // (op 2751) 


buffer_load_dwordx4 v[66:69], v10, s[68:71], 0 offen // (op 2751) Load value



v_mov_b64 v[42:43], s[40:41] // (op 2756) convert

v_mov_b32 v0, v14 // (op 2756) 
buffer_load_ubyte v70, v0, s[76:79], 0 offen offset:4 // (op 2756) Load value
v_add_u32 v0, v0, v42 // (op 2756) 
buffer_load_ubyte v71, v0, s[76:79], 0 offen offset:4 // (op 2756) Load value



v_and_b32 v0, -4, v1 // (op 3744) 
v_sub_u32 v10, v1, v0 // (op 3744) 

v_lshrrev_b32 v0, 2, v1 // (op 3744) 
v_lshl_add_u32 v13, v0, 3, v10 // (op 3744) 


s_barrier  // (op 3972) 



s_waitcnt vmcnt(10)
ds_write_b8 v13, v44 // (op 2758) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v13, v45 offset:4 // (op 2758) Store local data 

v_and_b32 v0, -4, v1 // (op 3715) 
v_sub_u32 v10, v1, v0 // (op 3715) 

v_lshrrev_b32 v0, 2, v1 // (op 3715) 
v_lshl_add_u32 v13, v0, 3, v10 // (op 3715) 


v_and_b32 v0, -4, v1 // (op 3815) 
v_sub_u32 v10, v1, v0 // (op 3815) 

v_lshrrev_b32 v0, 2, v1 // (op 3815) 
v_lshl_add_u32 v14, v0, 3, v10 // (op 3815) 


s_mov_b32 s1, 0 // (op 766) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 765) 
s_mov_b32 s15, s18 // (op 765) convert

s_and_b32 s6, s15, 4294967294 // (op 765) 

s_cmp_lt_u32 s1, s6 // (op 765) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_765 // (op 765) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_765 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_765:
s_add_u32 s1, s1, 2 // (op 767) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3969) 



ds_read_u8 v0, v3 // (op 164) Load local data 



ds_read_u8 v10, v3 offset:128 // (op 547) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[22:29], a[0:3], v0, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 167) 




ds_read_u8 v0, v3 offset:256 // (op 669) Load local data 



ds_read_u8 v42, v3 offset:384 // (op 791) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[22:29], a[4:7], v10, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 549) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[22:29], a[8:11], v0, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 671) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:41], v[22:29], a[12:15], v42, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 793) 



v_mov_b32 v0, v16 // (op 166) 

buffer_load_ubyte v12, v0, s[80:83], 0 offen offset:8 // (op 166) Load value


v_mov_b32 v10, v5 // (op 169) 
v_add_u32 v42, 128, v10 // (op 169) 

buffer_load_dwordx4 v[18:21], v42, s[68:71], 0 offen // (op 169) Load value



v_mov_b32 v42, v11 // (op 171) 

buffer_load_dwordx4 v[22:25], v42, s[72:75], 0 offen offset:256 // (op 171) Load value
buffer_load_dwordx4 v[26:29], v42, s[72:75], 0 offen offset:320 // (op 171) Load value


s_lshl_b64 s[18:19], s[30:31], 5 // (op 572) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 572) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 572) 

s_add_u32 s15, s18, 128 // (op 572) 

v_add_u32 v43, v10, s15 // (op 572) 


buffer_load_dwordx4 v[30:33], v43, s[68:71], 0 offen // (op 572) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 694) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 694) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 694) 

s_lshl_b32 s15, s18, 1 // (op 694) 
s_add_u32 s15, s15, 128 // (op 694) 

v_add_u32 v43, v10, s15 // (op 694) 


buffer_load_dwordx4 v[34:37], v43, s[68:71], 0 offen // (op 694) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 816) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 816) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 816) 

s_mul_i32 s15, 3, s18 // (op 816) 
s_add_u32 s15, s15, 128 // (op 816) 

v_add_u32 v43, v10, s15 // (op 816) 


buffer_load_dwordx4 v[38:41], v43, s[68:71], 0 offen // (op 816) Load value


v_mov_b64 v[72:73], s[40:41] // (op 161) convert

v_mov_b32 v43, v15 // (op 161) 
v_mov_b32 v74, v43 // (op 161) 
buffer_load_ubyte v44, v74, s[76:79], 0 offen offset:8 // (op 161) Load value
v_add_u32 v74, v74, v72 // (op 161) 
buffer_load_ubyte v45, v74, s[76:79], 0 offen offset:8 // (op 161) Load value


s_barrier  // (op 3988) 



s_waitcnt vmcnt(10)
ds_write_b8 v14, v70 offset:512 // (op 460) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v14, v71 offset:516 // (op 460) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3985) 



ds_read_u8 v72, v4 offset:512 // (op 462) Load local data 



ds_read_u8 v73, v4 offset:640 // (op 556) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[50:57], a[0:3], v72, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 465) 




ds_read_u8 v72, v4 offset:768 // (op 678) Load local data 



ds_read_u8 v74, v4 offset:896 // (op 800) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:61], v[50:57], a[4:7], v73, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 550) 

s_barrier  // (op 4103) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[50:57], a[8:11], v72, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 672) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[50:57], a[12:15], v74, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 794) 




buffer_load_ubyte v17, v0, s[80:83], 0 offen offset:12 // (op 464) Load value



v_add_u32 v16, 8, v16 // (op 3501) 
v_add_u32 v0, 192, v10 // (op 467) 

buffer_load_dwordx4 v[46:49], v0, s[68:71], 0 offen // (op 467) Load value




buffer_load_dwordx4 v[50:53], v42, s[72:75], 0 offen offset:384 // (op 469) Load value
buffer_load_dwordx4 v[54:57], v42, s[72:75], 0 offen offset:448 // (op 469) Load value



v_add_u32 v11, 256, v11 // (op 3245) 
s_lshl_b64 s[18:19], s[30:31], 5 // (op 565) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 565) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 565) 

s_add_u32 s15, s18, 192 // (op 565) 

v_add_u32 v0, v10, s15 // (op 565) 


buffer_load_dwordx4 v[58:61], v0, s[68:71], 0 offen // (op 565) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 687) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 687) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 687) 

s_lshl_b32 s15, s18, 1 // (op 687) 
s_add_u32 s15, s15, 192 // (op 687) 

v_add_u32 v0, v10, s15 // (op 687) 


buffer_load_dwordx4 v[62:65], v0, s[68:71], 0 offen // (op 687) Load value



s_lshl_b64 s[18:19], s[30:31], 5 // (op 809) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 809) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 809) 

s_mul_i32 s15, 3, s18 // (op 809) 
s_add_u32 s15, s15, 192 // (op 809) 

v_add_u32 v0, v10, s15 // (op 809) 


buffer_load_dwordx4 v[66:69], v0, s[68:71], 0 offen // (op 809) Load value



v_add_u32 v5, 128, v5 // (op 3096) 
v_mov_b64 v[72:73], s[40:41] // (op 459) convert

v_mov_b32 v0, v43 // (op 459) 
buffer_load_ubyte v70, v0, s[76:79], 0 offen offset:12 // (op 459) Load value
v_add_u32 v0, v0, v72 // (op 459) 
buffer_load_ubyte v71, v0, s[76:79], 0 offen offset:12 // (op 459) Load value



s_barrier  // (op 3975) 
v_add_u32 v15, 8, v15 // (op 3373) 



s_waitcnt vmcnt(10)
ds_write_b8 v13, v44 // (op 162) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v13, v45 offset:4 // (op 162) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 765) 
s_mov_b32 s15, s18 // (op 765) convert

s_and_b32 s6, s15, 4294967294 // (op 765) 

s_cmp_lt_u32 s1, s6 // (op 765) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_765 // (op 765) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_765 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_765:













v_lshrrev_b32 v0, 7, v1 // (op 3019) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3019) 
v_and_b32 v5, 15, v1 // (op 3019) 
s_and_b32 s15, s18, 4294967294 // (op 3019) 
v_lshl_add_u32 v10, v0, 4, v5 // (op 3019) 


s_add_u32 s6, s0, s15 // (op 3019) 

v_and_b32 v0, 63, v1 // (op 3019) 
v_lshrrev_b32 v5, 4, v0 // (op 3019) 

s_mul_i32 s15, s16, s50 // (op 3019) 
s_add_u32 s15, s15, s6 // (op 3019) 

v_lshlrev_b32 v0, 5, v5 // (op 3019) 

s_add_u32 s6, s15, s9 // (op 3019) 

s_mul_hi_u32 s15, s6, s13 // (op 3019) 
s_sub_u32 s7, s6, s15 // (op 3019) 
s_lshr_b32 s8, s7, 1 // (op 3019) 

s_add_u32 s7, s8, s15 // (op 3019) 


s_lshr_b32 s8, s7, s11 // (op 3019) 

s_and_b32 s7, s11, 2147483648 // (op 3019) 
s_cmp_eq_i32 s7, 0 // (op 3019) 
s_mov_b32 s15, scc // (op 3019) 

s_cmp_eq_i32 s15, 0 // (op 3019) 
s_cselect_b32 s7, s6, s8 // (op 3019) 


s_mul_hi_u32 s8, s7, s12 // (op 3019) 
s_sub_u32 s15, s7, s8 // (op 3019) 
s_lshr_b32 s28, s15, 1 // (op 3019) 

s_add_u32 s15, s28, s8 // (op 3019) 


s_lshr_b32 s8, s15, s10 // (op 3019) 

s_and_b32 s15, s10, 2147483648 // (op 3019) 
s_cmp_eq_i32 s15, 0 // (op 3019) 
s_mov_b32 s28, scc // (op 3019) 

s_cmp_eq_i32 s28, 0 // (op 3019) 
s_cselect_b32 s15, s7, s8 // (op 3019) 


s_mul_i32 s8, s7, s18 // (op 3019) 


v_lshl_add_u32 v5, s15, 7, v10 // (op 3019) 


s_sub_u32 s7, s6, s8 // (op 3019) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 3019) 


v_mov_b32 v19, 0 // (op 3019) convert
v_mov_b32 v18, v10 // (op 3019) convert

v_mul_lo_u32 v11, v5, s31 // (op 3019) most significant: low of low * high
v_mul_hi_u32 v0, v5, s30 // (op 3019) most significant: high of low * low
v_mul_lo_u32 v10, v5, s30 // (op 3019) least significant: low of low * low
v_add_u32 v11, v11, v0 // (op 3019) most significant: sum

v_add_co_u32 v10, s[18:19], v10, v18 // (op 3019) least significant half
v_addc_co_u32 v11, s[18:19], v11, v19, s[18:19] // (op 3019) most significant half



v_lshlrev_b64 v[12:13], 2, v[10:11] // (op 3019) 

v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 3019) 

v_mov_b32 v0, v10 // (op 3019) convert

s_mul_i32 s19, s30, s35 // (op 3021) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3021) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3021) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3021) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3021) most significant: sum
s_add_u32 s19, s19, s6 // (op 3021) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3021) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 3021) 

s_mov_b32 s68, s32 // (op 3021) reg expression
s_mov_b32 s69, s33 // (op 3021) reg expression
s_mov_b32 s70, s18 // (op 3021) reg expression
s_mov_b32 s71, 131072 // (op 3021) call()

v_lshrrev_b32 v5, 6, v1 // (op 3189) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3189) 
v_and_b32 v10, 1, v5 // (op 3189) 

s_and_b32 s15, s18, 4294967294 // (op 3189) 
v_and_b32 v5, 15, v1 // (op 3189) 
s_add_u32 s6, s0, s15 // (op 3189) 

v_lshl_add_u32 v11, v10, 4, v5 // (op 3189) 


v_and_b32 v5, 48, v1 // (op 3189) 
s_mul_i32 s15, s16, s50 // (op 3189) 
s_add_u32 s15, s15, s6 // (op 3189) 

s_add_u32 s6, s15, s9 // (op 3189) 

s_mul_hi_u32 s15, s6, s13 // (op 3189) 
s_sub_u32 s7, s6, s15 // (op 3189) 
s_lshr_b32 s8, s7, 1 // (op 3189) 

s_add_u32 s7, s8, s15 // (op 3189) 


s_lshr_b32 s8, s7, s11 // (op 3189) 

s_and_b32 s7, s11, 2147483648 // (op 3189) 
s_cmp_eq_i32 s7, 0 // (op 3189) 
s_mov_b32 s15, scc // (op 3189) 

s_cmp_eq_i32 s15, 0 // (op 3189) 
s_cselect_b32 s7, s6, s8 // (op 3189) 


s_mul_hi_u32 s8, s7, s12 // (op 3189) 
s_sub_u32 s15, s7, s8 // (op 3189) 
s_lshr_b32 s28, s15, 1 // (op 3189) 

s_add_u32 s15, s28, s8 // (op 3189) 


s_lshr_b32 s8, s15, s10 // (op 3189) 

s_and_b32 s15, s10, 2147483648 // (op 3189) 
s_cmp_eq_i32 s15, 0 // (op 3189) 
s_mov_b32 s28, scc // (op 3189) 

s_cmp_eq_i32 s28, 0 // (op 3189) 
s_cselect_b32 s15, s7, s8 // (op 3189) 


s_mul_i32 s8, s15, s14 // (op 3189) 

s_sub_u32 s15, s7, s8 // (op 3189) 

v_lshl_add_u32 v10, s15, 5, v11 // (op 3189) 


s_mul_i32 s8, s7, s18 // (op 3189) 


v_mul_lo_u32 v11, v10, s36 // (op 3189) 

s_sub_u32 s7, s6, s8 // (op 3189) 


v_lshl_add_u32 v10, s7, 7, v5 // (op 3189) 


v_add_u32 v5, v10, v11 // (op 3189) 


s_mul_i32 s19, s36, s23 // (op 3191) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3191) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3191) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3191) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3191) most significant: sum
s_add_u32 s19, s19, s6 // (op 3191) most significant: sum


s_mov_b32 s72, s38 // (op 3191) reg expression
s_mov_b32 s73, s39 // (op 3191) reg expression
s_mov_b32 s74, s18 // (op 3191) reg expression
s_mov_b32 s75, 131072 // (op 3191) call()

v_lshrrev_b32 v10, 2, v1 // (op 3317) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3317) 
v_lshlrev_b32 v11, 1, v10 // (op 3317) 

s_and_b32 s15, s18, 4294967294 // (op 3317) 
v_and_b32 v10, -4, v1 // (op 3317) 
s_add_u32 s6, s0, s15 // (op 3317) 

v_sub_u32 v12, v1, v10 // (op 3317) 

s_mul_i32 s15, s16, s50 // (op 3317) 
s_add_u32 s15, s15, s6 // (op 3317) 

s_add_u32 s6, s15, s9 // (op 3317) 

s_mul_hi_u32 s15, s6, s13 // (op 3317) 
s_sub_u32 s7, s6, s15 // (op 3317) 
s_lshr_b32 s8, s7, 1 // (op 3317) 

s_add_u32 s7, s8, s15 // (op 3317) 


s_lshr_b32 s8, s7, s11 // (op 3317) 

s_and_b32 s7, s11, 2147483648 // (op 3317) 
s_cmp_eq_i32 s7, 0 // (op 3317) 
s_mov_b32 s15, scc // (op 3317) 

s_cmp_eq_i32 s15, 0 // (op 3317) 
s_cselect_b32 s7, s6, s8 // (op 3317) 


s_mul_hi_u32 s8, s7, s12 // (op 3317) 
s_sub_u32 s15, s7, s8 // (op 3317) 
s_lshr_b32 s28, s15, 1 // (op 3317) 

s_add_u32 s15, s28, s8 // (op 3317) 


s_lshr_b32 s8, s15, s10 // (op 3317) 

s_and_b32 s15, s10, 2147483648 // (op 3317) 
s_cmp_eq_i32 s15, 0 // (op 3317) 
s_mov_b32 s28, scc // (op 3317) 

s_cmp_eq_i32 s28, 0 // (op 3317) 
s_cselect_b32 s15, s7, s8 // (op 3317) 


s_mul_i32 s8, s7, s18 // (op 3317) 


v_lshl_add_u32 v10, s15, 7, v11 // (op 3317) 


s_sub_u32 s7, s6, s8 // (op 3317) 


v_lshl_add_u32 v11, s7, 2, v12 // (op 3317) 


v_mul_lo_u32 v12, v10, s40 // (op 3317) 
v_add_u32 v12, v12, v11 // (op 3317) 


s_mul_i32 s19, s40, s35 // (op 3319) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3319) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3319) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3319) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3319) most significant: sum
s_add_u32 s19, s19, s6 // (op 3319) most significant: sum


s_mov_b32 s76, s42 // (op 3319) reg expression
s_mov_b32 s77, s43 // (op 3319) reg expression
s_mov_b32 s78, s18 // (op 3319) reg expression
s_mov_b32 s79, 131072 // (op 3319) call()

v_lshrrev_b32 v10, 6, v1 // (op 3445) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3445) 
v_and_b32 v11, 1, v10 // (op 3445) 

s_and_b32 s15, s18, 4294967294 // (op 3445) 
v_and_b32 v10, 15, v1 // (op 3445) 
s_add_u32 s6, s0, s15 // (op 3445) 

v_lshl_add_u32 v13, v11, 4, v10 // (op 3445) 


v_and_b32 v10, 63, v1 // (op 3445) 
s_mul_i32 s15, s16, s50 // (op 3445) 
s_add_u32 s15, s15, s6 // (op 3445) 

v_lshrrev_b32 v11, 4, v10 // (op 3445) 

s_add_u32 s6, s15, s9 // (op 3445) 

s_mul_hi_u32 s15, s6, s13 // (op 3445) 
s_sub_u32 s7, s6, s15 // (op 3445) 
s_lshr_b32 s8, s7, 1 // (op 3445) 

s_add_u32 s7, s8, s15 // (op 3445) 


s_lshr_b32 s8, s7, s11 // (op 3445) 

s_and_b32 s7, s11, 2147483648 // (op 3445) 
s_cmp_eq_i32 s7, 0 // (op 3445) 
s_mov_b32 s15, scc // (op 3445) 

s_cmp_eq_i32 s15, 0 // (op 3445) 
s_cselect_b32 s7, s6, s8 // (op 3445) 


s_mul_hi_u32 s8, s7, s12 // (op 3445) 
s_sub_u32 s15, s7, s8 // (op 3445) 
s_lshr_b32 s28, s15, 1 // (op 3445) 

s_add_u32 s15, s28, s8 // (op 3445) 


s_lshr_b32 s8, s15, s10 // (op 3445) 

s_and_b32 s15, s10, 2147483648 // (op 3445) 
s_cmp_eq_i32 s15, 0 // (op 3445) 
s_mov_b32 s28, scc // (op 3445) 

s_cmp_eq_i32 s28, 0 // (op 3445) 
s_cselect_b32 s15, s7, s8 // (op 3445) 


s_mul_i32 s8, s15, s14 // (op 3445) 

s_sub_u32 s15, s7, s8 // (op 3445) 

v_lshl_add_u32 v10, s15, 5, v13 // (op 3445) 


s_mul_i32 s8, s7, s18 // (op 3445) 


v_mul_lo_u32 v13, v10, s44 // (op 3445) 

s_sub_u32 s7, s6, s8 // (op 3445) 


v_lshl_add_u32 v10, s7, 2, v11 // (op 3445) 


v_add_u32 v11, v10, v13 // (op 3445) 


s_mul_i32 s19, s44, s23 // (op 3447) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3447) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3447) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3447) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3447) most significant: sum
s_add_u32 s19, s19, s6 // (op 3447) most significant: sum


s_mov_b32 s80, s46 // (op 3447) reg expression
s_mov_b32 s81, s47 // (op 3447) reg expression
s_mov_b32 s82, s18 // (op 3447) reg expression
s_mov_b32 s83, 131072 // (op 3447) call()

v_and_b32 v10, -4, v1 // (op 3728) 
v_sub_u32 v13, v1, v10 // (op 3728) 

v_lshrrev_b32 v10, 2, v1 // (op 3728) 
v_lshl_add_u32 v14, v10, 3, v13 // (op 3728) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 771) 
s_and_b32 s15, s18, 4294967294 // (op 771) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 770) 
s_mov_b32 s6, s18 // (op 770) convert

s_cmp_lt_u32 s15, s6 // (op 770) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_770 // (op 770) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_770 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_770:

s_add_u32 s15, s15, 1 // (op 772) 
v_mov_b64 v[16:17], s[40:41] // (op 417) convert

v_mov_b32 v10, v12 // (op 417) 
v_mov_b32 v13, v10 // (op 417) 
buffer_load_ubyte v18, v13, s[76:79], 0 offen offset: 0 // (op 417) Load value
v_add_u32 v13, v13, v16 // (op 417) 
buffer_load_ubyte v19, v13, s[76:79], 0 offen offset: 0 // (op 417) Load value



v_add_u32 v12, 4, v12 // (op 3309) 


s_barrier  // (op 3978) 
v_mov_b32 v10, v0 // (op 425) 



v_mov_b32 v13, v10 // (op 425) reg expression

buffer_load_dwordx4 v[20:23], v13, s[68:71], 0 offen // (op 425) Load value

s_waitcnt vmcnt(2)
ds_write_b8 v14, v18 // (op 418) Store local data 


v_mov_b32 v13, v5 // (op 427) 

buffer_load_dwordx4 v[24:27], v13, s[72:75], 0 offen // (op 427) Load value
buffer_load_dwordx4 v[28:31], v13, s[72:75], 0 offen offset:64 // (op 427) Load value

v_add_u32 v5, 128, v5 // (op 3181) 
s_waitcnt vmcnt(3)
ds_write_b8 v14, v19 offset:4 // (op 418) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3981) 



ds_read_u8 v13, v3 // (op 420) Load local data 


v_mov_b32 v15, v11 // (op 422) 

buffer_load_ubyte v16, v15, s[80:83], 0 offen offset: 0 // (op 422) Load value

v_add_u32 v11, 4, v11 // (op 3437) 


s_lshl_b64 s[18:19], s[30:31], 5 // (op 542) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 542) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 542) 

v_add_u32 v15, v10, s18 // (op 542) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:31], a[0:3], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 423) 
buffer_load_dwordx4 v[32:35], v15, s[68:71], 0 offen // (op 542) Load value






ds_read_u8 v13, v3 offset:128 // (op 532) Load local data 


s_lshl_b64 s[18:19], s[30:31], 5 // (op 664) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 664) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 664) 

s_lshl_b32 s6, s18, 1 // (op 664) 

v_add_u32 v15, v10, s6 // (op 664) 


buffer_load_dwordx4 v[20:23], v15, s[68:71], 0 offen // (op 664) Load value

s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:35], v[24:31], a[4:7], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 534) 





ds_read_u8 v13, v3 offset:256 // (op 654) Load local data 


s_lshl_b64 s[18:19], s[30:31], 5 // (op 786) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 786) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 786) 

s_mul_i32 s6, 3, s18 // (op 786) 

v_add_u32 v15, v10, s6 // (op 786) 


buffer_load_dwordx4 v[32:35], v15, s[68:71], 0 offen // (op 786) Load value

s_waitcnt vmcnt(1) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[24:31], a[8:11], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 656) 






v_add_u32 v0, 64, v0 // (op 3007) 
ds_read_u8 v10, v3 offset:384 // (op 776) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:35], v[24:31], a[12:15], v10, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 778) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 770) 
s_mov_b32 s6, s18 // (op 770) convert

s_cmp_lt_u32 s15, s6 // (op 770) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_770 // (op 770) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_770 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_770:






s_barrier  // (op 4089) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 193) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_193 // (op 193) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_193
v_accvgpr_read v10, a0 // (op 201) 
v_accvgpr_read v11, a1 // (op 201) 
v_accvgpr_read v12, a2 // (op 201) 
v_accvgpr_read v13, a3 // (op 201) 
v_mul_f32 v14, v7, v10 // (op 201) 
v_mul_f32 v15, v7, v11 // (op 201) 
v_mul_f32 v16, v7, v12 // (op 201) 
v_mul_f32 v17, v7, v13 // (op 201) 

v_lshrrev_b32 v10, 16, v14 // (op 203) 
v_lshrrev_b32 v11, 16, v15 // (op 203) 
v_lshrrev_b32 v12, 16, v16 // (op 203) 
v_lshrrev_b32 v13, 16, v17 // (op 203) 


v_pack_b32_f16 v14, v10, v11 // (op 204) 
v_pack_b32_f16 v15, v12, v13 // (op 204) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen // (op 204) Store value


v_accvgpr_read v10, a4 // (op 972) 
v_accvgpr_read v11, a5 // (op 972) 
v_accvgpr_read v12, a6 // (op 972) 
v_accvgpr_read v13, a7 // (op 972) 
v_mul_f32 v14, v7, v10 // (op 972) 
v_mul_f32 v15, v7, v11 // (op 972) 
v_mul_f32 v16, v7, v12 // (op 972) 
v_mul_f32 v17, v7, v13 // (op 972) 

v_lshrrev_b32 v10, 16, v14 // (op 974) 
v_lshrrev_b32 v11, 16, v15 // (op 974) 
v_lshrrev_b32 v12, 16, v16 // (op 974) 
v_lshrrev_b32 v13, 16, v17 // (op 974) 


v_pack_b32_f16 v14, v10, v11 // (op 975) 
v_pack_b32_f16 v15, v12, v13 // (op 975) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:64 // (op 975) Store value


v_accvgpr_read v10, a8 // (op 992) 
v_accvgpr_read v11, a9 // (op 992) 
v_accvgpr_read v12, a10 // (op 992) 
v_accvgpr_read v13, a11 // (op 992) 
v_mul_f32 v14, v7, v10 // (op 992) 
v_mul_f32 v15, v7, v11 // (op 992) 
v_mul_f32 v16, v7, v12 // (op 992) 
v_mul_f32 v17, v7, v13 // (op 992) 

v_lshrrev_b32 v10, 16, v14 // (op 994) 
v_lshrrev_b32 v11, 16, v15 // (op 994) 
v_lshrrev_b32 v12, 16, v16 // (op 994) 
v_lshrrev_b32 v13, 16, v17 // (op 994) 


v_pack_b32_f16 v14, v10, v11 // (op 995) 
v_pack_b32_f16 v15, v12, v13 // (op 995) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:128 // (op 995) Store value


v_accvgpr_read v10, a12 // (op 1012) 
v_accvgpr_read v11, a13 // (op 1012) 
v_accvgpr_read v12, a14 // (op 1012) 
v_accvgpr_read v13, a15 // (op 1012) 
v_mul_f32 v14, v7, v10 // (op 1012) 
v_mul_f32 v15, v7, v11 // (op 1012) 
v_mul_f32 v16, v7, v12 // (op 1012) 
v_mul_f32 v17, v7, v13 // (op 1012) 

v_lshrrev_b32 v10, 16, v14 // (op 1014) 
v_lshrrev_b32 v11, 16, v15 // (op 1014) 
v_lshrrev_b32 v12, 16, v16 // (op 1014) 
v_lshrrev_b32 v13, 16, v17 // (op 1014) 


v_pack_b32_f16 v14, v10, v11 // (op 1015) 
v_pack_b32_f16 v15, v12, v13 // (op 1015) 


buffer_store_dwordx2 v[14:15], v9, s[60:63], 0 offen offset:192 // (op 1015) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_193 // (op 193) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_193
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_193:



buffer_load_dwordx2 v[10:11], v8, s[56:59], 0 offen offset:192 // (op 1095) Load value
v_accvgpr_read v12, a12 // (op 1091) 
v_accvgpr_read v13, a13 // (op 1091) 
v_accvgpr_read v14, a14 // (op 1091) 
v_accvgpr_read v15, a15 // (op 1091) 
v_mul_f32 v16, v7, v12 // (op 1091) 
v_mul_f32 v17, v7, v13 // (op 1091) 
v_mul_f32 v18, v7, v14 // (op 1091) 
v_mul_f32 v19, v7, v15 // (op 1091) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1096) 
v_lshrrev_b32 v13, 16, v10 // (op 1096) 
v_lshlrev_b32 v12, 16, v12 // (op 1096) 
v_lshlrev_b32 v13, 16, v13 // (op 1096) 
v_mul_f32 v20, v6, v12 // (op 1096) 
v_mul_f32 v21, v6, v13 // (op 1096) 
v_and_b32 v12, 65535, v11 // (op 1096) 
v_lshrrev_b32 v13, 16, v11 // (op 1096) 
v_lshlrev_b32 v12, 16, v12 // (op 1096) 
v_lshlrev_b32 v13, 16, v13 // (op 1096) 
v_mul_f32 v22, v6, v12 // (op 1096) 
v_mul_f32 v23, v6, v13 // (op 1096) 


v_add_f32 v10, v20, v16 // (op 1092) 
v_add_f32 v11, v21, v17 // (op 1092) 
v_add_f32 v12, v22, v18 // (op 1092) 
v_add_f32 v13, v23, v19 // (op 1092) 


v_lshrrev_b32 v14, 16, v10 // (op 1093) 
v_lshrrev_b32 v15, 16, v11 // (op 1093) 
v_lshrrev_b32 v16, 16, v12 // (op 1093) 
v_lshrrev_b32 v17, 16, v13 // (op 1093) 


v_pack_b32_f16 v10, v14, v15 // (op 1094) 
v_pack_b32_f16 v11, v16, v17 // (op 1094) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:192 // (op 1094) Store value





buffer_load_dwordx2 v[10:11], v8, s[56:59], 0 offen offset:128 // (op 1075) Load value
v_accvgpr_read v12, a8 // (op 1071) 
v_accvgpr_read v13, a9 // (op 1071) 
v_accvgpr_read v14, a10 // (op 1071) 
v_accvgpr_read v15, a11 // (op 1071) 
v_mul_f32 v16, v7, v12 // (op 1071) 
v_mul_f32 v17, v7, v13 // (op 1071) 
v_mul_f32 v18, v7, v14 // (op 1071) 
v_mul_f32 v19, v7, v15 // (op 1071) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1076) 
v_lshrrev_b32 v13, 16, v10 // (op 1076) 
v_lshlrev_b32 v12, 16, v12 // (op 1076) 
v_lshlrev_b32 v13, 16, v13 // (op 1076) 
v_mul_f32 v20, v6, v12 // (op 1076) 
v_mul_f32 v21, v6, v13 // (op 1076) 
v_and_b32 v12, 65535, v11 // (op 1076) 
v_lshrrev_b32 v13, 16, v11 // (op 1076) 
v_lshlrev_b32 v12, 16, v12 // (op 1076) 
v_lshlrev_b32 v13, 16, v13 // (op 1076) 
v_mul_f32 v22, v6, v12 // (op 1076) 
v_mul_f32 v23, v6, v13 // (op 1076) 


v_add_f32 v10, v20, v16 // (op 1072) 
v_add_f32 v11, v21, v17 // (op 1072) 
v_add_f32 v12, v22, v18 // (op 1072) 
v_add_f32 v13, v23, v19 // (op 1072) 


v_lshrrev_b32 v14, 16, v10 // (op 1073) 
v_lshrrev_b32 v15, 16, v11 // (op 1073) 
v_lshrrev_b32 v16, 16, v12 // (op 1073) 
v_lshrrev_b32 v17, 16, v13 // (op 1073) 


v_pack_b32_f16 v10, v14, v15 // (op 1074) 
v_pack_b32_f16 v11, v16, v17 // (op 1074) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:128 // (op 1074) Store value





buffer_load_dwordx2 v[10:11], v8, s[56:59], 0 offen offset:64 // (op 1055) Load value
v_accvgpr_read v12, a4 // (op 1051) 
v_accvgpr_read v13, a5 // (op 1051) 
v_accvgpr_read v14, a6 // (op 1051) 
v_accvgpr_read v15, a7 // (op 1051) 
v_mul_f32 v16, v7, v12 // (op 1051) 
v_mul_f32 v17, v7, v13 // (op 1051) 
v_mul_f32 v18, v7, v14 // (op 1051) 
v_mul_f32 v19, v7, v15 // (op 1051) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 1056) 
v_lshrrev_b32 v13, 16, v10 // (op 1056) 
v_lshlrev_b32 v12, 16, v12 // (op 1056) 
v_lshlrev_b32 v13, 16, v13 // (op 1056) 
v_mul_f32 v20, v6, v12 // (op 1056) 
v_mul_f32 v21, v6, v13 // (op 1056) 
v_and_b32 v12, 65535, v11 // (op 1056) 
v_lshrrev_b32 v13, 16, v11 // (op 1056) 
v_lshlrev_b32 v12, 16, v12 // (op 1056) 
v_lshlrev_b32 v13, 16, v13 // (op 1056) 
v_mul_f32 v22, v6, v12 // (op 1056) 
v_mul_f32 v23, v6, v13 // (op 1056) 


v_add_f32 v10, v20, v16 // (op 1052) 
v_add_f32 v11, v21, v17 // (op 1052) 
v_add_f32 v12, v22, v18 // (op 1052) 
v_add_f32 v13, v23, v19 // (op 1052) 


v_lshrrev_b32 v14, 16, v10 // (op 1053) 
v_lshrrev_b32 v15, 16, v11 // (op 1053) 
v_lshrrev_b32 v16, 16, v12 // (op 1053) 
v_lshrrev_b32 v17, 16, v13 // (op 1053) 


v_pack_b32_f16 v10, v14, v15 // (op 1054) 
v_pack_b32_f16 v11, v16, v17 // (op 1054) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen offset:64 // (op 1054) Store value





buffer_load_dwordx2 v[10:11], v8, s[56:59], 0 offen // (op 218) Load value
v_accvgpr_read v12, a0 // (op 214) 
v_accvgpr_read v13, a1 // (op 214) 
v_accvgpr_read v14, a2 // (op 214) 
v_accvgpr_read v15, a3 // (op 214) 
v_mul_f32 v16, v7, v12 // (op 214) 
v_mul_f32 v17, v7, v13 // (op 214) 
v_mul_f32 v18, v7, v14 // (op 214) 
v_mul_f32 v19, v7, v15 // (op 214) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 219) 
v_lshrrev_b32 v13, 16, v10 // (op 219) 
v_lshlrev_b32 v12, 16, v12 // (op 219) 
v_lshlrev_b32 v13, 16, v13 // (op 219) 
v_mul_f32 v20, v6, v12 // (op 219) 
v_mul_f32 v21, v6, v13 // (op 219) 
v_and_b32 v12, 65535, v11 // (op 219) 
v_lshrrev_b32 v13, 16, v11 // (op 219) 
v_lshlrev_b32 v12, 16, v12 // (op 219) 
v_lshlrev_b32 v13, 16, v13 // (op 219) 
v_mul_f32 v22, v6, v12 // (op 219) 
v_mul_f32 v23, v6, v13 // (op 219) 


v_add_f32 v10, v20, v16 // (op 215) 
v_add_f32 v11, v21, v17 // (op 215) 
v_add_f32 v12, v22, v18 // (op 215) 
v_add_f32 v13, v23, v19 // (op 215) 


v_lshrrev_b32 v14, 16, v10 // (op 216) 
v_lshrrev_b32 v15, 16, v11 // (op 216) 
v_lshrrev_b32 v16, 16, v12 // (op 216) 
v_lshrrev_b32 v17, 16, v13 // (op 216) 


v_pack_b32_f16 v10, v14, v15 // (op 217) 
v_pack_b32_f16 v11, v16, v17 // (op 217) 


buffer_store_dwordx2 v[10:11], v9, s[60:63], 0 offen // (op 217) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_193:
s_add_u32 s0, s0, s1 // (op 406) 


s_mul_i32 s15, s50, s16 // (op 404) 
s_cmp_gt_u32 s51, s15 // (op 404) 
s_mov_b32 s6, scc // (op 404) 
s_sub_u32 s7, s51, s15 // (op 404) 

s_cmp_eq_i32 s6, 0 // (op 404) 
s_cselect_b32 s8, 0, s7 // (op 404) 


s_cmp_lt_u32 s50, s8 // (op 404) 
s_mov_b32 s15, scc // (op 404) 
s_cmp_eq_i32 s15, 0 // (op 404) 
s_cselect_b32 s6, s8, s50 // (op 404) 


s_cmp_lt_u32 s0, s6 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_404 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_404:







































s_endpgm  // End of RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.LRR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2, .LRR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_end-RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 1024 // lds bytes
  .amdhsa_accum_offset 80
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
    DataTypeB: F8
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
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: address, semantic: RR_SKAddressPartials },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: address, semantic: RR_SKAddressFlags },
            { type: uint32, semantic: RR_SKNumDPTilesPerWG },
            { type: uint32, semantic: RR_SKNumDPTiles } ]
    macrotile: [128, 32, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 6
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 2
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 86
    .vgpr_count: 78
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_23
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
      - .name: Tensor_21_stride_1_24
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_None
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_8
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_9
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_2_stride_1_11
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_14
        .size: 8
        .offset: 136
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 144
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_16
        .size: 8
        .offset: 152
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: SCRATCH_ZeroedBeforeAndAfter
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

