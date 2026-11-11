// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2,@function
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 3446) 
v_and_b32 v3, 1, v0 // (op 3446) 

v_and_b32 v0, 15, v1 // (op 3446) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3446) 


v_and_b32 v0, 63, v1 // (op 3446) 
v_lshrrev_b32 v3, 4, v0 // (op 3446) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 3446) 


v_lshrrev_b32 v3, 6, v1 // (op 3546) 
v_and_b32 v4, 1, v3 // (op 3546) 

v_and_b32 v3, 15, v1 // (op 3546) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 3546) 


v_and_b32 v3, 63, v1 // (op 3546) 
v_lshrrev_b32 v4, 4, v3 // (op 3546) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 3546) 


v_lshrrev_b32 v4, 6, v1 // (op 3617) 
v_and_b32 v5, 1, v4 // (op 3617) 

v_and_b32 v4, 15, v1 // (op 3617) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 3617) 


v_and_b32 v4, 63, v1 // (op 3617) 
v_lshrrev_b32 v5, 4, v4 // (op 3617) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 3617) 


v_lshrrev_b32 v5, 6, v1 // (op 3646) 
v_and_b32 v6, 1, v5 // (op 3646) 

v_and_b32 v5, 15, v1 // (op 3646) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 3646) 


v_and_b32 v5, 63, v1 // (op 3646) 
v_lshrrev_b32 v6, 4, v5 // (op 3646) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 3646) 


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
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 // (op 249) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 if false)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_249:
v_lshrrev_b32 v8, 7, v1 // (op 3361) 
s_mul_i32 s52, s16, s8 // (op 3361) 
s_add_u32 s52, s52, s0 // (op 3361) 
v_and_b32 v9, 63, v1 // (op 3361) 
s_mul_hi_u32 s53, s52, s13 // (op 3361) 
v_lshrrev_b32 v10, 4, v9 // (op 3361) 
s_sub_u32 s54, s52, s53 // (op 3361) 
v_lshlrev_b32 v11, 2, v10 // (op 3361) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 3361) 


s_lshr_b32 s55, s54, 1 // (op 3361) 

s_add_u32 s54, s55, s53 // (op 3361) 


v_lshrrev_b32 v8, 2, v9 // (op 3361) 

s_lshr_b32 s53, s54, s11 // (op 3361) 

v_and_b32 v9, 3, v8 // (op 3361) 

s_and_b32 s54, s11, 2147483648 // (op 3361) 
v_and_b32 v8, 3, v1 // (op 3361) 
s_cmp_eq_i32 s54, 0 // (op 3361) 
s_mov_b32 s55, scc // (op 3361) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 3361) 


s_cmp_eq_i32 s55, 0 // (op 3361) 
s_cselect_b32 s54, s52, s53 // (op 3361) 



v_lshrrev_b32 v8, 6, v1 // (op 3361) 
s_mul_hi_u32 s52, s54, s12 // (op 3361) 
v_and_b32 v9, 1, v8 // (op 3361) 

s_sub_u32 s53, s54, s52 // (op 3361) 
v_lshl_add_u32 v8, v9, 4, v11 // (op 3361) 


s_lshr_b32 s55, s53, 1 // (op 3361) 

s_add_u32 s53, s55, s52 // (op 3361) 


s_lshr_b32 s52, s53, s10 // (op 3361) 

s_and_b32 s53, s10, 2147483648 // (op 3361) 
s_cmp_eq_i32 s53, 0 // (op 3361) 
s_mov_b32 s55, scc // (op 3361) 

s_cmp_eq_i32 s55, 0 // (op 3361) 
s_cselect_b32 s53, s54, s52 // (op 3361) 


s_mul_i32 s52, s53, s14 // (op 3361) 
v_lshl_add_u32 v9, s53, 5, v10 // (op 3361) 


s_sub_u32 s53, s54, s52 // (op 3361) 


v_lshl_add_u32 v10, s53, 7, v8 // (op 3361) 


v_mul_lo_u32 v8, v10, s4 // (op 3361) 

v_add_lshl_u32 v10, v9, v8, 1 // (op 3361) 


s_mul_i32 s53, s4, s23 // (op 3363) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 3363) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 3363) least significant
s_mul_hi_u32 s55, s4, s22 // (op 3363) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3363) most significant: sum
s_add_u32 s53, s53, s55 // (op 3363) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 3363) 

s_mov_b32 s56, s20 // (op 3363) reg expression
s_mov_b32 s57, s21 // (op 3363) reg expression
s_mov_b32 s58, s54 // (op 3363) reg expression
s_mov_b32 s59, 131072 // (op 3363) call()

v_lshrrev_b32 v8, 7, v1 // (op 3463) 
s_mul_i32 s52, s16, s8 // (op 3463) 
s_add_u32 s52, s52, s0 // (op 3463) 
v_and_b32 v9, 63, v1 // (op 3463) 
s_mul_hi_u32 s53, s52, s13 // (op 3463) 
v_lshrrev_b32 v11, 4, v9 // (op 3463) 
s_sub_u32 s54, s52, s53 // (op 3463) 
v_lshlrev_b32 v12, 2, v11 // (op 3463) 

s_lshr_b32 s55, s54, 1 // (op 3463) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 3463) 


s_add_u32 s54, s55, s53 // (op 3463) 


v_lshrrev_b32 v8, 2, v9 // (op 3463) 

s_lshr_b32 s53, s54, s11 // (op 3463) 

v_and_b32 v9, 3, v8 // (op 3463) 

s_and_b32 s54, s11, 2147483648 // (op 3463) 
v_and_b32 v8, 3, v1 // (op 3463) 
s_cmp_eq_i32 s54, 0 // (op 3463) 
s_mov_b32 s55, scc // (op 3463) 

v_lshl_add_u32 v12, v9, 2, v8 // (op 3463) 


s_cmp_eq_i32 s55, 0 // (op 3463) 
s_cselect_b32 s54, s52, s53 // (op 3463) 



v_lshrrev_b32 v8, 6, v1 // (op 3463) 
s_mul_hi_u32 s55, s54, s12 // (op 3463) 
v_and_b32 v9, 1, v8 // (op 3463) 

s_sub_u32 s52, s54, s55 // (op 3463) 
v_lshl_add_u32 v8, v9, 4, v12 // (op 3463) 


s_lshr_b32 s53, s52, 1 // (op 3463) 

s_add_u32 s52, s53, s55 // (op 3463) 


s_lshr_b32 s53, s52, s10 // (op 3463) 

s_and_b32 s52, s10, 2147483648 // (op 3463) 
s_cmp_eq_i32 s52, 0 // (op 3463) 
s_mov_b32 s55, scc // (op 3463) 

s_cmp_eq_i32 s55, 0 // (op 3463) 
s_cselect_b32 s52, s54, s53 // (op 3463) 


s_mul_i32 s53, s52, s14 // (op 3463) 
v_lshl_add_u32 v9, s52, 5, v11 // (op 3463) 


s_sub_u32 s52, s54, s53 // (op 3463) 


v_lshl_add_u32 v11, s52, 7, v8 // (op 3463) 


v_mul_lo_u32 v8, v11, s24 // (op 3463) 

v_add_lshl_u32 v11, v9, v8, 1 // (op 3463) 


s_mul_i32 s53, s24, s23 // (op 3465) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 3465) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 3465) least significant
s_mul_hi_u32 s55, s24, s22 // (op 3465) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3465) most significant: sum
s_add_u32 s53, s53, s55 // (op 3465) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 3465) 

s_mov_b32 s60, s26 // (op 3465) reg expression
s_mov_b32 s61, s27 // (op 3465) reg expression
s_mov_b32 s62, s54 // (op 3465) reg expression
s_mov_b32 s63, 131072 // (op 3465) call()

v_lshrrev_b32 v8, 5, v1 // (op 3563) 
v_lshlrev_b32 v9, 2, v8 // (op 3563) 

v_lshl_add_u32 v8, s16, 5, v9 // (op 3563) 

v_and_b32 v9, -32, v1 // (op 3563) 
v_sub_u32 v12, v1, v9 // (op 3563) 

v_lshl_add_u32 v9, v8, 7, v12 // (op 3563) 


v_lshlrev_b32 v8, 2, v9 // (op 3563) 

s_lshl_b32 s52, s15, 14 // (op 3565) 
s_mov_b32 s64, s28 // (op 3565) reg expression
s_mov_b32 s65, s29 // (op 3565) reg expression
s_mov_b32 s66, s52 // (op 3565) reg expression
s_mov_b32 s67, 131072 // (op 3565) call()

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
v_accvgpr_write a4, 0.00000 // (op 1196) call()
v_accvgpr_write a5, 0.00000 // (op 1196) call()
v_accvgpr_write a6, 0.00000 // (op 1196) call()
v_accvgpr_write a7, 0.00000 // (op 1196) call()
v_accvgpr_write a8, 0.00000 // (op 1312) call()
v_accvgpr_write a9, 0.00000 // (op 1312) call()
v_accvgpr_write a10, 0.00000 // (op 1312) call()
v_accvgpr_write a11, 0.00000 // (op 1312) call()
v_accvgpr_write a12, 0.00000 // (op 1428) call()
v_accvgpr_write a13, 0.00000 // (op 1428) call()
v_accvgpr_write a14, 0.00000 // (op 1428) call()
v_accvgpr_write a15, 0.00000 // (op 1428) call()
v_lshrrev_b32 v9, 7, v1 // (op 2903) 
s_mul_i32 s52, s16, s8 // (op 2903) 
s_add_u32 s52, s52, s0 // (op 2903) 
v_and_b32 v12, 15, v1 // (op 2903) 
s_mul_hi_u32 s53, s52, s13 // (op 2903) 
v_lshl_add_u32 v13, v9, 4, v12 // (op 2903) 


s_sub_u32 s54, s52, s53 // (op 2903) 
v_and_b32 v9, 48, v1 // (op 2903) 
s_lshr_b32 s55, s54, 1 // (op 2903) 

s_add_u32 s54, s55, s53 // (op 2903) 


s_lshr_b32 s53, s54, s11 // (op 2903) 

s_and_b32 s54, s11, 2147483648 // (op 2903) 
s_cmp_eq_i32 s54, 0 // (op 2903) 
s_mov_b32 s55, scc // (op 2903) 

s_cmp_eq_i32 s55, 0 // (op 2903) 
s_cselect_b32 s54, s52, s53 // (op 2903) 


s_mul_hi_u32 s53, s54, s12 // (op 2903) 
s_sub_u32 s55, s54, s53 // (op 2903) 
s_lshr_b32 s68, s55, 1 // (op 2903) 

s_add_u32 s55, s68, s53 // (op 2903) 


s_lshr_b32 s53, s55, s10 // (op 2903) 

s_and_b32 s55, s10, 2147483648 // (op 2903) 
s_cmp_eq_i32 s55, 0 // (op 2903) 
s_mov_b32 s68, scc // (op 2903) 

s_cmp_eq_i32 s68, 0 // (op 2903) 
s_cselect_b32 s55, s54, s53 // (op 2903) 


v_lshl_add_u32 v12, s55, 5, v13 // (op 2903) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 2903) 
s_mul_i32 s53, s54, s68 // (op 2903) 


s_sub_u32 s54, s52, s53 // (op 2903) 


v_lshl_add_u32 v13, s54, 7, v9 // (op 2903) 


v_mul_lo_u32 v9, v12, s30 // (op 2903) 
v_add_u32 v9, v9, v13 // (op 2903) 


s_mul_i32 s53, s30, s35 // (op 2905) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 2905) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 2905) least significant
s_mul_hi_u32 s55, s30, s34 // (op 2905) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2905) most significant: sum
s_add_u32 s53, s53, s55 // (op 2905) most significant: sum


s_mov_b32 s68, s32 // (op 2905) reg expression
s_mov_b32 s69, s33 // (op 2905) reg expression
s_mov_b32 s70, s52 // (op 2905) reg expression
s_mov_b32 s71, 131072 // (op 2905) call()

v_lshrrev_b32 v12, 7, v1 // (op 2922) 
s_mul_i32 s52, s16, s8 // (op 2922) 
s_add_u32 s52, s52, s0 // (op 2922) 
v_and_b32 v13, 15, v1 // (op 2922) 
s_mul_hi_u32 s53, s52, s13 // (op 2922) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2922) 


s_sub_u32 s54, s52, s53 // (op 2922) 
v_and_b32 v12, 48, v1 // (op 2922) 
s_lshr_b32 s55, s54, 1 // (op 2922) 

s_add_u32 s54, s55, s53 // (op 2922) 


s_lshr_b32 s53, s54, s11 // (op 2922) 

s_and_b32 s54, s11, 2147483648 // (op 2922) 
s_cmp_eq_i32 s54, 0 // (op 2922) 
s_mov_b32 s55, scc // (op 2922) 

s_cmp_eq_i32 s55, 0 // (op 2922) 
s_cselect_b32 s54, s52, s53 // (op 2922) 


s_mul_hi_u32 s53, s54, s12 // (op 2922) 
s_sub_u32 s55, s54, s53 // (op 2922) 
s_lshr_b32 s72, s55, 1 // (op 2922) 

s_add_u32 s55, s72, s53 // (op 2922) 


s_lshr_b32 s53, s55, s10 // (op 2922) 

s_and_b32 s55, s10, 2147483648 // (op 2922) 
s_cmp_eq_i32 s55, 0 // (op 2922) 
s_mov_b32 s72, scc // (op 2922) 

s_cmp_eq_i32 s72, 0 // (op 2922) 
s_cselect_b32 s55, s54, s53 // (op 2922) 


v_lshl_add_u32 v13, s55, 5, v14 // (op 2922) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 2922) 
s_mul_i32 s53, s54, s72 // (op 2922) 


s_sub_u32 s54, s52, s53 // (op 2922) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 2922) 


v_mul_lo_u32 v12, v13, s30 // (op 2922) 
v_add_u32 v12, v12, v14 // (op 2922) 


s_mul_i32 s53, s30, s35 // (op 2924) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 2924) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 2924) least significant
s_mul_hi_u32 s55, s30, s34 // (op 2924) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2924) most significant: sum
s_add_u32 s53, s53, s55 // (op 2924) most significant: sum


s_mov_b32 s68, s32 // (op 2924) reg expression
s_mov_b32 s69, s33 // (op 2924) reg expression
s_mov_b32 s70, s52 // (op 2924) reg expression
s_mov_b32 s71, 131072 // (op 2924) call()

v_lshrrev_b32 v13, 6, v1 // (op 3056) 
s_mul_i32 s52, s16, s8 // (op 3056) 
s_add_u32 s52, s52, s0 // (op 3056) 
v_and_b32 v14, 1, v13 // (op 3056) 

s_mul_hi_u32 s53, s52, s13 // (op 3056) 
v_and_b32 v13, 15, v1 // (op 3056) 
s_sub_u32 s54, s52, s53 // (op 3056) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 3056) 


s_lshr_b32 s55, s54, 1 // (op 3056) 

v_and_b32 v13, 63, v1 // (op 3056) 
s_add_u32 s54, s55, s53 // (op 3056) 


v_lshrrev_b32 v14, 4, v13 // (op 3056) 

s_lshr_b32 s53, s54, s11 // (op 3056) 

v_lshlrev_b32 v13, 5, v14 // (op 3056) 

s_and_b32 s54, s11, 2147483648 // (op 3056) 
s_cmp_eq_i32 s54, 0 // (op 3056) 
s_mov_b32 s55, scc // (op 3056) 

s_cmp_eq_i32 s55, 0 // (op 3056) 
s_cselect_b32 s54, s52, s53 // (op 3056) 


s_mul_hi_u32 s53, s54, s12 // (op 3056) 
s_sub_u32 s55, s54, s53 // (op 3056) 
s_lshr_b32 s72, s55, 1 // (op 3056) 

s_add_u32 s55, s72, s53 // (op 3056) 


s_lshr_b32 s53, s55, s10 // (op 3056) 

s_and_b32 s55, s10, 2147483648 // (op 3056) 
s_cmp_eq_i32 s55, 0 // (op 3056) 
s_mov_b32 s72, scc // (op 3056) 

s_cmp_eq_i32 s72, 0 // (op 3056) 
s_cselect_b32 s55, s54, s53 // (op 3056) 


s_mul_i32 s53, s55, s14 // (op 3056) 

s_sub_u32 s55, s54, s53 // (op 3056) 

v_lshl_add_u32 v14, s55, 7, v15 // (op 3056) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 3056) 
v_mul_lo_u32 v17, v14, s37 // (op 3056) most significant: low of low * high
v_mul_hi_u32 v15, v14, s36 // (op 3056) most significant: high of low * low
v_mul_lo_u32 v16, v14, s36 // (op 3056) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 3056) most significant: sum


s_mul_i32 s53, s54, s72 // (op 3056) 


s_sub_u32 s54, s52, s53 // (op 3056) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 3056) 


v_mov_b32 v19, 0 // (op 3056) convert
v_mov_b32 v18, v14 // (op 3056) convert

v_add_co_u32 v14, s[52:53], v18, v16 // (op 3056) least significant half
v_addc_co_u32 v15, s[52:53], v19, v17, s[52:53] // (op 3056) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 3056) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 3056) 

v_mov_b32 v13, v16 // (op 3056) convert

s_mul_i32 s53, s36, s23 // (op 3058) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3058) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3058) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3058) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3058) most significant: sum
s_add_u32 s53, s53, s55 // (op 3058) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3058) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 3058) 

s_mov_b32 s72, s38 // (op 3058) reg expression
s_mov_b32 s73, s39 // (op 3058) reg expression
s_mov_b32 s74, s52 // (op 3058) reg expression
s_mov_b32 s75, 131072 // (op 3058) call()

v_lshrrev_b32 v14, 6, v1 // (op 3077) 
s_mul_i32 s52, s16, s8 // (op 3077) 
s_add_u32 s52, s52, s0 // (op 3077) 
v_and_b32 v15, 1, v14 // (op 3077) 

s_mul_hi_u32 s53, s52, s13 // (op 3077) 
v_and_b32 v14, 15, v1 // (op 3077) 
s_sub_u32 s54, s52, s53 // (op 3077) 
v_lshl_add_u32 v16, v15, 4, v14 // (op 3077) 


s_lshr_b32 s55, s54, 1 // (op 3077) 

v_and_b32 v14, 63, v1 // (op 3077) 
s_add_u32 s54, s55, s53 // (op 3077) 


v_lshrrev_b32 v15, 4, v14 // (op 3077) 

s_lshr_b32 s53, s54, s11 // (op 3077) 

v_lshlrev_b32 v14, 5, v15 // (op 3077) 

s_and_b32 s54, s11, 2147483648 // (op 3077) 
s_cmp_eq_i32 s54, 0 // (op 3077) 
s_mov_b32 s55, scc // (op 3077) 

s_cmp_eq_i32 s55, 0 // (op 3077) 
s_cselect_b32 s54, s52, s53 // (op 3077) 


s_mul_hi_u32 s53, s54, s12 // (op 3077) 
s_sub_u32 s55, s54, s53 // (op 3077) 
s_lshr_b32 s76, s55, 1 // (op 3077) 

s_add_u32 s55, s76, s53 // (op 3077) 


s_lshr_b32 s53, s55, s10 // (op 3077) 

s_and_b32 s55, s10, 2147483648 // (op 3077) 
s_cmp_eq_i32 s55, 0 // (op 3077) 
s_mov_b32 s76, scc // (op 3077) 

s_cmp_eq_i32 s76, 0 // (op 3077) 
s_cselect_b32 s55, s54, s53 // (op 3077) 


s_mul_i32 s53, s55, s14 // (op 3077) 

s_sub_u32 s55, s54, s53 // (op 3077) 

v_lshl_add_u32 v15, s55, 7, v16 // (op 3077) 


v_mul_lo_u32 v17, v15, s37 // (op 3077) most significant: low of low * high
v_mul_hi_u32 v18, v15, s36 // (op 3077) most significant: high of low * low
v_mul_lo_u32 v16, v15, s36 // (op 3077) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 3077) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 3077) 
s_mul_i32 s53, s54, s76 // (op 3077) 


s_sub_u32 s54, s52, s53 // (op 3077) 


v_lshl_add_u32 v15, s54, 7, v14 // (op 3077) 


v_mov_b32 v19, 0 // (op 3077) convert
v_mov_b32 v18, v15 // (op 3077) convert

v_add_co_u32 v14, s[52:53], v18, v16 // (op 3077) least significant half
v_addc_co_u32 v15, s[52:53], v19, v17, s[52:53] // (op 3077) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 3077) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 3077) 

v_mov_b32 v14, v16 // (op 3077) convert

s_mul_i32 s53, s36, s23 // (op 3079) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 3079) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 3079) least significant
s_mul_hi_u32 s55, s36, s22 // (op 3079) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3079) most significant: sum
s_add_u32 s53, s53, s55 // (op 3079) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 3079) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 3079) 

s_mov_b32 s72, s38 // (op 3079) reg expression
s_mov_b32 s73, s39 // (op 3079) reg expression
s_mov_b32 s74, s52 // (op 3079) reg expression
s_mov_b32 s75, 131072 // (op 3079) call()

v_lshrrev_b32 v15, 7, v1 // (op 3189) 
s_mul_i32 s52, s16, s8 // (op 3189) 
s_add_u32 s52, s52, s0 // (op 3189) 
v_and_b32 v16, 15, v1 // (op 3189) 
s_mul_hi_u32 s53, s52, s13 // (op 3189) 
v_lshl_add_u32 v17, v15, 4, v16 // (op 3189) 


s_sub_u32 s54, s52, s53 // (op 3189) 
v_and_b32 v15, 63, v1 // (op 3189) 
s_lshr_b32 s55, s54, 1 // (op 3189) 

v_lshrrev_b32 v16, 4, v15 // (op 3189) 

s_add_u32 s54, s55, s53 // (op 3189) 


s_lshr_b32 s53, s54, s11 // (op 3189) 

s_and_b32 s54, s11, 2147483648 // (op 3189) 
s_cmp_eq_i32 s54, 0 // (op 3189) 
s_mov_b32 s55, scc // (op 3189) 

s_cmp_eq_i32 s55, 0 // (op 3189) 
s_cselect_b32 s54, s52, s53 // (op 3189) 


s_mul_hi_u32 s53, s54, s12 // (op 3189) 
s_sub_u32 s55, s54, s53 // (op 3189) 
s_lshr_b32 s76, s55, 1 // (op 3189) 

s_add_u32 s55, s76, s53 // (op 3189) 


s_lshr_b32 s53, s55, s10 // (op 3189) 

s_and_b32 s55, s10, 2147483648 // (op 3189) 
s_cmp_eq_i32 s55, 0 // (op 3189) 
s_mov_b32 s76, scc // (op 3189) 

s_cmp_eq_i32 s76, 0 // (op 3189) 
s_cselect_b32 s55, s54, s53 // (op 3189) 


v_lshl_add_u32 v15, s55, 5, v17 // (op 3189) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 3189) 
s_mul_i32 s53, s54, s76 // (op 3189) 


s_sub_u32 s54, s52, s53 // (op 3189) 


v_lshl_add_u32 v17, s54, 2, v16 // (op 3189) 


v_mul_lo_u32 v16, v15, s40 // (op 3189) 
v_add_u32 v16, v16, v17 // (op 3189) 


s_mul_i32 s53, s40, s35 // (op 3191) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3191) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3191) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3191) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3191) most significant: sum
s_add_u32 s53, s53, s55 // (op 3191) most significant: sum


s_mov_b32 s76, s42 // (op 3191) reg expression
s_mov_b32 s77, s43 // (op 3191) reg expression
s_mov_b32 s78, s52 // (op 3191) reg expression
s_mov_b32 s79, 131072 // (op 3191) call()

v_lshrrev_b32 v15, 7, v1 // (op 3208) 
s_mul_i32 s52, s16, s8 // (op 3208) 
s_add_u32 s52, s52, s0 // (op 3208) 
v_and_b32 v17, 15, v1 // (op 3208) 
s_mul_hi_u32 s53, s52, s13 // (op 3208) 
v_lshl_add_u32 v18, v15, 4, v17 // (op 3208) 


s_sub_u32 s54, s52, s53 // (op 3208) 
v_and_b32 v15, 63, v1 // (op 3208) 
s_lshr_b32 s55, s54, 1 // (op 3208) 

v_lshrrev_b32 v17, 4, v15 // (op 3208) 

s_add_u32 s54, s55, s53 // (op 3208) 


s_lshr_b32 s53, s54, s11 // (op 3208) 

s_and_b32 s54, s11, 2147483648 // (op 3208) 
s_cmp_eq_i32 s54, 0 // (op 3208) 
s_mov_b32 s55, scc // (op 3208) 

s_cmp_eq_i32 s55, 0 // (op 3208) 
s_cselect_b32 s54, s52, s53 // (op 3208) 


s_mul_hi_u32 s53, s54, s12 // (op 3208) 
s_sub_u32 s55, s54, s53 // (op 3208) 
s_lshr_b32 s80, s55, 1 // (op 3208) 

s_add_u32 s55, s80, s53 // (op 3208) 


s_lshr_b32 s53, s55, s10 // (op 3208) 

s_and_b32 s55, s10, 2147483648 // (op 3208) 
s_cmp_eq_i32 s55, 0 // (op 3208) 
s_mov_b32 s80, scc // (op 3208) 

s_cmp_eq_i32 s80, 0 // (op 3208) 
s_cselect_b32 s55, s54, s53 // (op 3208) 


v_lshl_add_u32 v15, s55, 5, v18 // (op 3208) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 3208) 
s_mul_i32 s53, s54, s80 // (op 3208) 


s_sub_u32 s54, s52, s53 // (op 3208) 


v_lshl_add_u32 v18, s54, 2, v17 // (op 3208) 


v_mul_lo_u32 v17, v15, s40 // (op 3208) 
v_add_u32 v17, v17, v18 // (op 3208) 


s_mul_i32 s53, s40, s35 // (op 3210) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3210) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3210) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3210) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3210) most significant: sum
s_add_u32 s53, s53, s55 // (op 3210) most significant: sum


s_mov_b32 s76, s42 // (op 3210) reg expression
s_mov_b32 s77, s43 // (op 3210) reg expression
s_mov_b32 s78, s52 // (op 3210) reg expression
s_mov_b32 s79, 131072 // (op 3210) call()

v_lshrrev_b32 v15, 2, v1 // (op 3317) 
s_mul_i32 s52, s16, s8 // (op 3317) 
s_add_u32 s52, s52, s0 // (op 3317) 
v_lshlrev_b32 v18, 1, v15 // (op 3317) 

s_mul_hi_u32 s53, s52, s13 // (op 3317) 
v_and_b32 v15, -4, v1 // (op 3317) 
s_sub_u32 s54, s52, s53 // (op 3317) 
v_sub_u32 v19, v1, v15 // (op 3317) 

s_lshr_b32 s55, s54, 1 // (op 3317) 

s_add_u32 s54, s55, s53 // (op 3317) 


s_lshr_b32 s53, s54, s11 // (op 3317) 

s_and_b32 s54, s11, 2147483648 // (op 3317) 
s_cmp_eq_i32 s54, 0 // (op 3317) 
s_mov_b32 s55, scc // (op 3317) 

s_cmp_eq_i32 s55, 0 // (op 3317) 
s_cselect_b32 s54, s52, s53 // (op 3317) 


s_mul_hi_u32 s53, s54, s12 // (op 3317) 
s_sub_u32 s55, s54, s53 // (op 3317) 
s_lshr_b32 s80, s55, 1 // (op 3317) 

s_add_u32 s55, s80, s53 // (op 3317) 


s_lshr_b32 s53, s55, s10 // (op 3317) 

s_and_b32 s55, s10, 2147483648 // (op 3317) 
s_cmp_eq_i32 s55, 0 // (op 3317) 
s_mov_b32 s80, scc // (op 3317) 

s_cmp_eq_i32 s80, 0 // (op 3317) 
s_cselect_b32 s55, s54, s53 // (op 3317) 


s_mul_i32 s53, s55, s14 // (op 3317) 

s_sub_u32 s55, s54, s53 // (op 3317) 

v_lshl_add_u32 v15, s55, 7, v18 // (op 3317) 


v_mul_lo_u32 v18, v15, s44 // (op 3317) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 3317) 
s_mul_i32 s53, s54, s80 // (op 3317) 


s_sub_u32 s54, s52, s53 // (op 3317) 


v_lshl_add_u32 v15, s54, 2, v19 // (op 3317) 


v_add_u32 v19, v15, v18 // (op 3317) 


s_mul_i32 s53, s44, s23 // (op 3319) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3319) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3319) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3319) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3319) most significant: sum
s_add_u32 s53, s53, s55 // (op 3319) most significant: sum


s_mov_b32 s80, s46 // (op 3319) reg expression
s_mov_b32 s81, s47 // (op 3319) reg expression
s_mov_b32 s82, s52 // (op 3319) reg expression
s_mov_b32 s83, 131072 // (op 3319) call()

v_lshrrev_b32 v15, 2, v1 // (op 3336) 
s_mul_i32 s52, s16, s8 // (op 3336) 
s_add_u32 s52, s52, s0 // (op 3336) 
v_lshlrev_b32 v18, 1, v15 // (op 3336) 

s_mul_hi_u32 s53, s52, s13 // (op 3336) 
v_and_b32 v15, -4, v1 // (op 3336) 
s_sub_u32 s54, s52, s53 // (op 3336) 
v_sub_u32 v20, v1, v15 // (op 3336) 

s_lshr_b32 s55, s54, 1 // (op 3336) 

s_add_u32 s54, s55, s53 // (op 3336) 


s_lshr_b32 s53, s54, s11 // (op 3336) 

s_and_b32 s54, s11, 2147483648 // (op 3336) 
s_cmp_eq_i32 s54, 0 // (op 3336) 
s_mov_b32 s55, scc // (op 3336) 

s_cmp_eq_i32 s55, 0 // (op 3336) 
s_cselect_b32 s54, s52, s53 // (op 3336) 


s_mul_hi_u32 s53, s54, s12 // (op 3336) 
s_sub_u32 s55, s54, s53 // (op 3336) 
s_lshr_b32 s84, s55, 1 // (op 3336) 

s_add_u32 s55, s84, s53 // (op 3336) 


s_lshr_b32 s53, s55, s10 // (op 3336) 

s_and_b32 s55, s10, 2147483648 // (op 3336) 
s_cmp_eq_i32 s55, 0 // (op 3336) 
s_mov_b32 s84, scc // (op 3336) 

s_cmp_eq_i32 s84, 0 // (op 3336) 
s_cselect_b32 s55, s54, s53 // (op 3336) 


s_mul_i32 s53, s55, s14 // (op 3336) 

s_sub_u32 s55, s54, s53 // (op 3336) 

v_lshl_add_u32 v15, s55, 7, v18 // (op 3336) 


v_mul_lo_u32 v18, v15, s44 // (op 3336) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 3336) 
s_mul_i32 s53, s54, s84 // (op 3336) 


s_sub_u32 s54, s52, s53 // (op 3336) 


v_lshl_add_u32 v15, s54, 2, v20 // (op 3336) 


v_add_u32 v20, v15, v18 // (op 3336) 


s_mul_i32 s53, s44, s23 // (op 3338) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3338) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3338) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3338) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3338) most significant: sum
s_add_u32 s53, s53, s55 // (op 3338) most significant: sum


s_mov_b32 s80, s46 // (op 3338) reg expression
s_mov_b32 s81, s47 // (op 3338) reg expression
s_mov_b32 s82, s52 // (op 3338) reg expression
s_mov_b32 s83, 131072 // (op 3338) call()




buffer_load_dwordx4 v[22:25], v9, s[68:71], 0 offen // (op 2634) Load value
buffer_load_dwordx4 v[26:29], v9, s[68:71], 0 offen offset:64 // (op 2634) Load value


v_mov_b32 v15, v13 // (op 2640) reg expression

buffer_load_dwordx4 v[30:33], v15, s[72:75], 0 offen // (op 2640) Load value




buffer_load_ubyte v15, v16, s[76:79], 0 offen offset: 0 // (op 2646) Load value


s_lshl_b64 s[52:53], s[36:37], 5 // (op 2652) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2652) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2652) 

v_add_u32 v18, v13, s52 // (op 2652) 


buffer_load_dwordx4 v[34:37], v18, s[72:75], 0 offen // (op 2652) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 2659) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2659) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2659) 

s_lshl_b32 s54, s52, 1 // (op 2659) 

v_add_u32 v18, v13, s54 // (op 2659) 


buffer_load_dwordx4 v[38:41], v18, s[72:75], 0 offen // (op 2659) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 2666) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2666) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2666) 

s_mul_i32 s54, 3, s52 // (op 2666) 

v_add_u32 v18, v13, s54 // (op 2666) 


buffer_load_dwordx4 v[42:45], v18, s[72:75], 0 offen // (op 2666) Load value


v_mov_b64 v[46:47], s[44:45] // (op 2671) convert

v_mov_b32 v18, v19 // (op 2671) 
buffer_load_ubyte v48, v18, s[80:83], 0 offen offset: 0 // (op 2671) Load value
v_add_u32 v18, v18, v46 // (op 2671) 
buffer_load_ubyte v49, v18, s[80:83], 0 offen offset: 0 // (op 2671) Load value





buffer_load_dwordx4 v[50:53], v9, s[68:71], 0 offen offset:128 // (op 2675) Load value
buffer_load_dwordx4 v[54:57], v9, s[68:71], 0 offen offset:192 // (op 2675) Load value



v_add_u32 v9, 64, v13 // (op 2681) 

buffer_load_dwordx4 v[58:61], v9, s[72:75], 0 offen // (op 2681) Load value




buffer_load_ubyte v9, v16, s[76:79], 0 offen offset:4 // (op 2687) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 2693) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2693) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2693) 

s_add_u32 s54, s52, 64 // (op 2693) 

v_add_u32 v16, v13, s54 // (op 2693) 


buffer_load_dwordx4 v[62:65], v16, s[72:75], 0 offen // (op 2693) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 2700) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2700) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2700) 

s_lshl_b32 s54, s52, 1 // (op 2700) 
s_add_u32 s54, s54, 64 // (op 2700) 

v_add_u32 v16, v13, s54 // (op 2700) 


buffer_load_dwordx4 v[66:69], v16, s[72:75], 0 offen // (op 2700) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 2707) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 2707) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 2707) 

s_mul_i32 s54, 3, s52 // (op 2707) 
s_add_u32 s54, s54, 64 // (op 2707) 

v_add_u32 v16, v13, s54 // (op 2707) 


buffer_load_dwordx4 v[70:73], v16, s[72:75], 0 offen // (op 2707) Load value



v_mov_b64 v[46:47], s[44:45] // (op 2712) convert

v_mov_b32 v13, v19 // (op 2712) 
buffer_load_ubyte v74, v13, s[80:83], 0 offen offset:4 // (op 2712) Load value
v_add_u32 v13, v13, v46 // (op 2712) 
buffer_load_ubyte v75, v13, s[80:83], 0 offen offset:4 // (op 2712) Load value



v_and_b32 v13, -4, v1 // (op 3429) 
v_sub_u32 v16, v1, v13 // (op 3429) 

v_lshrrev_b32 v13, 2, v1 // (op 3429) 
v_lshl_add_u32 v21, v13, 3, v16 // (op 3429) 


s_barrier  // (op 3741) 



s_waitcnt vmcnt(10)
ds_write_b8 v21, v48 // (op 2714) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v21, v49 offset:4 // (op 2714) Store local data 

v_and_b32 v13, -4, v1 // (op 3400) 
v_sub_u32 v16, v1, v13 // (op 3400) 

v_lshrrev_b32 v13, 2, v1 // (op 3400) 
v_lshl_add_u32 v21, v13, 3, v16 // (op 3400) 


v_and_b32 v13, -4, v1 // (op 3629) 
v_sub_u32 v16, v1, v13 // (op 3629) 

v_lshrrev_b32 v13, 2, v1 // (op 3629) 
v_lshl_add_u32 v18, v13, 3, v16 // (op 3629) 


s_mov_b32 s1, 0 // (op 1430) call()
s_and_b32 s52, s7, 4294967294 // (op 1429) 
s_cmp_lt_u32 s1, s52 // (op 1429) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1429 // (op 1429) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1429 if false)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1429:
s_add_u32 s1, s1, 2 // (op 1431) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3738) 



ds_read_u8 v13, v0 // (op 69) Load local data 



ds_read_u8 v16, v0 offset:128 // (op 1223) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:33], a[0:3], v15, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 72) 




ds_read_u8 v13, v0 offset:256 // (op 1339) Load local data 



ds_read_u8 v19, v0 offset:384 // (op 1455) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:29], v[34:37], a[4:7], v15, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1224) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:29], v[38:41], a[8:11], v15, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1340) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:29], v[42:45], a[12:15], v15, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1456) 



v_mov_b32 v13, v12 // (op 57) 

buffer_load_dwordx4 v[22:25], v13, s[68:71], 0 offen offset:256 // (op 57) Load value
buffer_load_dwordx4 v[26:29], v13, s[68:71], 0 offen offset:320 // (op 57) Load value


v_mov_b32 v16, v14 // (op 61) 
v_add_u32 v19, 128, v16 // (op 61) 

buffer_load_dwordx4 v[30:33], v19, s[72:75], 0 offen // (op 61) Load value



v_mov_b32 v19, v17 // (op 65) 

buffer_load_ubyte v15, v19, s[76:79], 0 offen offset:8 // (op 65) Load value


s_lshl_b64 s[52:53], s[36:37], 5 // (op 1246) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1246) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1246) 

s_add_u32 s54, s52, 128 // (op 1246) 

v_add_u32 v46, v16, s54 // (op 1246) 


buffer_load_dwordx4 v[34:37], v46, s[72:75], 0 offen // (op 1246) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 1362) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1362) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1362) 

s_lshl_b32 s54, s52, 1 // (op 1362) 
s_add_u32 s54, s54, 128 // (op 1362) 

v_add_u32 v46, v16, s54 // (op 1362) 


buffer_load_dwordx4 v[38:41], v46, s[72:75], 0 offen // (op 1362) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 1478) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1478) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1478) 

s_mul_i32 s54, 3, s52 // (op 1478) 
s_add_u32 s54, s54, 128 // (op 1478) 

v_add_u32 v46, v16, s54 // (op 1478) 


buffer_load_dwordx4 v[42:45], v46, s[72:75], 0 offen // (op 1478) Load value


v_mov_b64 v[46:47], s[44:45] // (op 13) convert

v_mov_b32 v76, v20 // (op 13) 
v_mov_b32 v77, v76 // (op 13) 
buffer_load_ubyte v48, v77, s[80:83], 0 offen offset:8 // (op 13) Load value
v_add_u32 v77, v77, v46 // (op 13) 
buffer_load_ubyte v49, v77, s[80:83], 0 offen offset:8 // (op 13) Load value


s_barrier  // (op 3779) 



s_waitcnt vmcnt(10)
ds_write_b8 v18, v74 offset:512 // (op 1139) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v18, v75 offset:516 // (op 1139) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3776) 



ds_read_u8 v46, v5 offset:512 // (op 1141) Load local data 



ds_read_u8 v47, v5 offset:640 // (op 1227) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[58:61], a[0:3], v9, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1142) 




ds_read_u8 v46, v5 offset:768 // (op 1343) Load local data 



ds_read_u8 v77, v5 offset:896 // (op 1459) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[62:65], a[4:7], v9, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1225) 

s_barrier  // (op 3984) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[66:69], a[8:11], v9, v46 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1341) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[70:73], a[12:15], v9, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1457) 




buffer_load_dwordx4 v[50:53], v13, s[68:71], 0 offen offset:384 // (op 1144) Load value
buffer_load_dwordx4 v[54:57], v13, s[68:71], 0 offen offset:448 // (op 1144) Load value



v_add_u32 v12, 256, v12 // (op 2914) 
v_add_u32 v13, 192, v16 // (op 1146) 

buffer_load_dwordx4 v[58:61], v13, s[72:75], 0 offen // (op 1146) Load value




buffer_load_ubyte v9, v19, s[76:79], 0 offen offset:12 // (op 1148) Load value



v_add_u32 v17, 8, v17 // (op 3200) 
s_lshl_b64 s[52:53], s[36:37], 5 // (op 1241) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1241) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1241) 

s_add_u32 s54, s52, 192 // (op 1241) 

v_add_u32 v13, v16, s54 // (op 1241) 


buffer_load_dwordx4 v[62:65], v13, s[72:75], 0 offen // (op 1241) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 1357) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1357) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1357) 

s_lshl_b32 s54, s52, 1 // (op 1357) 
s_add_u32 s54, s54, 192 // (op 1357) 

v_add_u32 v13, v16, s54 // (op 1357) 


buffer_load_dwordx4 v[66:69], v13, s[72:75], 0 offen // (op 1357) Load value



s_lshl_b64 s[52:53], s[36:37], 5 // (op 1473) 
s_lshl_b64 s[54:55], s[52:53], 2 // (op 1473) 

s_lshr_b64 s[52:53], s[54:55], 3 // (op 1473) 

s_mul_i32 s54, 3, s52 // (op 1473) 
s_add_u32 s54, s54, 192 // (op 1473) 

v_add_u32 v13, v16, s54 // (op 1473) 


buffer_load_dwordx4 v[70:73], v13, s[72:75], 0 offen // (op 1473) Load value



v_add_u32 v14, 128, v14 // (op 3067) 
v_mov_b64 v[46:47], s[44:45] // (op 1138) convert

v_mov_b32 v13, v76 // (op 1138) 
buffer_load_ubyte v74, v13, s[80:83], 0 offen offset:12 // (op 1138) Load value
v_add_u32 v13, v13, v46 // (op 1138) 
buffer_load_ubyte v75, v13, s[80:83], 0 offen offset:12 // (op 1138) Load value



s_barrier  // (op 3744) 
v_add_u32 v20, 8, v20 // (op 3328) 



s_waitcnt vmcnt(10)
ds_write_b8 v21, v48 // (op 18) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v21, v49 offset:4 // (op 18) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 1429) 
s_cmp_lt_u32 s1, s52 // (op 1429) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1429 // (op 1429) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1429 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1429:













s_and_b32 s52, s7, 4294967294 // (op 2826) 
v_lshrrev_b32 v12, 7, v1 // (op 2826) 
s_add_u32 s53, s0, s52 // (op 2826) 

v_and_b32 v13, 15, v1 // (op 2826) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2826) 


s_mul_i32 s52, s16, s8 // (op 2826) 
s_add_u32 s52, s52, s53 // (op 2826) 

v_and_b32 v12, 48, v1 // (op 2826) 
s_mul_hi_u32 s53, s52, s13 // (op 2826) 
s_sub_u32 s54, s52, s53 // (op 2826) 
s_lshr_b32 s55, s54, 1 // (op 2826) 

s_add_u32 s54, s55, s53 // (op 2826) 


s_lshr_b32 s53, s54, s11 // (op 2826) 

s_and_b32 s54, s11, 2147483648 // (op 2826) 
s_cmp_eq_i32 s54, 0 // (op 2826) 
s_mov_b32 s55, scc // (op 2826) 

s_cmp_eq_i32 s55, 0 // (op 2826) 
s_cselect_b32 s54, s52, s53 // (op 2826) 


s_mul_hi_u32 s53, s54, s12 // (op 2826) 
s_sub_u32 s55, s54, s53 // (op 2826) 
s_lshr_b32 s84, s55, 1 // (op 2826) 

s_add_u32 s55, s84, s53 // (op 2826) 


s_lshr_b32 s53, s55, s10 // (op 2826) 

s_and_b32 s55, s10, 2147483648 // (op 2826) 
s_cmp_eq_i32 s55, 0 // (op 2826) 
s_mov_b32 s84, scc // (op 2826) 

s_cmp_eq_i32 s84, 0 // (op 2826) 
s_cselect_b32 s55, s54, s53 // (op 2826) 


v_lshl_add_u32 v13, s55, 5, v14 // (op 2826) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2826) 
s_mul_i32 s53, s54, s84 // (op 2826) 


s_sub_u32 s54, s52, s53 // (op 2826) 


v_lshl_add_u32 v14, s54, 7, v12 // (op 2826) 


v_mul_lo_u32 v12, v13, s30 // (op 2826) 
v_add_u32 v12, v12, v14 // (op 2826) 


s_mul_i32 s53, s30, s35 // (op 2828) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 2828) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 2828) least significant
s_mul_hi_u32 s55, s30, s34 // (op 2828) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2828) most significant: sum
s_add_u32 s53, s53, s55 // (op 2828) most significant: sum


s_mov_b32 s68, s32 // (op 2828) reg expression
s_mov_b32 s69, s33 // (op 2828) reg expression
s_mov_b32 s70, s52 // (op 2828) reg expression
s_mov_b32 s71, 131072 // (op 2828) call()

s_and_b32 s52, s7, 4294967294 // (op 2973) 
v_lshrrev_b32 v13, 6, v1 // (op 2973) 
s_add_u32 s53, s0, s52 // (op 2973) 

v_and_b32 v14, 1, v13 // (op 2973) 

v_and_b32 v13, 15, v1 // (op 2973) 
s_mul_i32 s52, s16, s8 // (op 2973) 
s_add_u32 s52, s52, s53 // (op 2973) 

v_lshl_add_u32 v15, v14, 4, v13 // (op 2973) 


s_mul_hi_u32 s53, s52, s13 // (op 2973) 
v_and_b32 v13, 63, v1 // (op 2973) 
s_sub_u32 s54, s52, s53 // (op 2973) 
v_lshrrev_b32 v14, 4, v13 // (op 2973) 

s_lshr_b32 s55, s54, 1 // (op 2973) 

v_lshlrev_b32 v13, 5, v14 // (op 2973) 

s_add_u32 s54, s55, s53 // (op 2973) 


s_lshr_b32 s53, s54, s11 // (op 2973) 

s_and_b32 s54, s11, 2147483648 // (op 2973) 
s_cmp_eq_i32 s54, 0 // (op 2973) 
s_mov_b32 s55, scc // (op 2973) 

s_cmp_eq_i32 s55, 0 // (op 2973) 
s_cselect_b32 s54, s52, s53 // (op 2973) 


s_mul_hi_u32 s53, s54, s12 // (op 2973) 
s_sub_u32 s55, s54, s53 // (op 2973) 
s_lshr_b32 s84, s55, 1 // (op 2973) 

s_add_u32 s55, s84, s53 // (op 2973) 


s_lshr_b32 s53, s55, s10 // (op 2973) 

s_and_b32 s55, s10, 2147483648 // (op 2973) 
s_cmp_eq_i32 s55, 0 // (op 2973) 
s_mov_b32 s84, scc // (op 2973) 

s_cmp_eq_i32 s84, 0 // (op 2973) 
s_cselect_b32 s55, s54, s53 // (op 2973) 


s_mul_i32 s53, s55, s14 // (op 2973) 

s_sub_u32 s55, s54, s53 // (op 2973) 

v_lshl_add_u32 v14, s55, 7, v15 // (op 2973) 


v_mul_lo_u32 v49, v14, s37 // (op 2973) most significant: low of low * high
v_mul_hi_u32 v15, v14, s36 // (op 2973) most significant: high of low * low
v_mul_lo_u32 v48, v14, s36 // (op 2973) least significant: low of low * low
v_add_u32 v49, v49, v15 // (op 2973) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 2973) 
s_mul_i32 s53, s54, s84 // (op 2973) 


s_sub_u32 s54, s52, s53 // (op 2973) 


v_lshl_add_u32 v14, s54, 7, v13 // (op 2973) 


v_mov_b32 v47, 0 // (op 2973) convert
v_mov_b32 v46, v14 // (op 2973) convert

v_add_co_u32 v44, s[52:53], v46, v48 // (op 2973) least significant half
v_addc_co_u32 v45, s[52:53], v47, v49, s[52:53] // (op 2973) most significant half

v_lshlrev_b64 v[44:45], 2, v[44:45] // (op 2973) 


v_lshrrev_b64 v[42:43], 3, v[44:45] // (op 2973) 

v_mov_b32 v13, v42 // (op 2973) convert

s_mul_i32 s53, s36, s23 // (op 2975) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 2975) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 2975) least significant
s_mul_hi_u32 s55, s36, s22 // (op 2975) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2975) most significant: sum
s_add_u32 s53, s53, s55 // (op 2975) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 2975) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 2975) 

s_mov_b32 s72, s38 // (op 2975) reg expression
s_mov_b32 s73, s39 // (op 2975) reg expression
s_mov_b32 s74, s52 // (op 2975) reg expression
s_mov_b32 s75, 131072 // (op 2975) call()

s_and_b32 s52, s7, 4294967294 // (op 3124) 
v_lshrrev_b32 v14, 7, v1 // (op 3124) 
s_add_u32 s53, s0, s52 // (op 3124) 

v_and_b32 v15, 15, v1 // (op 3124) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 3124) 


s_mul_i32 s52, s16, s8 // (op 3124) 
s_add_u32 s52, s52, s53 // (op 3124) 

v_and_b32 v14, 63, v1 // (op 3124) 
s_mul_hi_u32 s53, s52, s13 // (op 3124) 
v_lshrrev_b32 v15, 4, v14 // (op 3124) 

s_sub_u32 s54, s52, s53 // (op 3124) 
s_lshr_b32 s55, s54, 1 // (op 3124) 

s_add_u32 s54, s55, s53 // (op 3124) 


s_lshr_b32 s53, s54, s11 // (op 3124) 

s_and_b32 s54, s11, 2147483648 // (op 3124) 
s_cmp_eq_i32 s54, 0 // (op 3124) 
s_mov_b32 s55, scc // (op 3124) 

s_cmp_eq_i32 s55, 0 // (op 3124) 
s_cselect_b32 s54, s52, s53 // (op 3124) 


s_mul_hi_u32 s53, s54, s12 // (op 3124) 
s_sub_u32 s55, s54, s53 // (op 3124) 
s_lshr_b32 s84, s55, 1 // (op 3124) 

s_add_u32 s55, s84, s53 // (op 3124) 


s_lshr_b32 s53, s55, s10 // (op 3124) 

s_and_b32 s55, s10, 2147483648 // (op 3124) 
s_cmp_eq_i32 s55, 0 // (op 3124) 
s_mov_b32 s84, scc // (op 3124) 

s_cmp_eq_i32 s84, 0 // (op 3124) 
s_cselect_b32 s55, s54, s53 // (op 3124) 


v_lshl_add_u32 v14, s55, 5, v16 // (op 3124) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 3124) 
s_mul_i32 s53, s54, s84 // (op 3124) 


s_sub_u32 s54, s52, s53 // (op 3124) 


v_lshl_add_u32 v16, s54, 2, v15 // (op 3124) 


v_mul_lo_u32 v15, v14, s40 // (op 3124) 
v_add_u32 v15, v15, v16 // (op 3124) 


s_mul_i32 s53, s40, s35 // (op 3126) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 3126) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 3126) least significant
s_mul_hi_u32 s55, s40, s34 // (op 3126) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3126) most significant: sum
s_add_u32 s53, s53, s55 // (op 3126) most significant: sum


s_mov_b32 s76, s42 // (op 3126) reg expression
s_mov_b32 s77, s43 // (op 3126) reg expression
s_mov_b32 s78, s52 // (op 3126) reg expression
s_mov_b32 s79, 131072 // (op 3126) call()

s_and_b32 s52, s7, 4294967294 // (op 3252) 
v_lshrrev_b32 v14, 2, v1 // (op 3252) 
s_add_u32 s53, s0, s52 // (op 3252) 

v_lshlrev_b32 v16, 1, v14 // (op 3252) 

v_and_b32 v14, -4, v1 // (op 3252) 
s_mul_i32 s52, s16, s8 // (op 3252) 
s_add_u32 s52, s52, s53 // (op 3252) 

v_sub_u32 v17, v1, v14 // (op 3252) 

s_mul_hi_u32 s53, s52, s13 // (op 3252) 
s_sub_u32 s54, s52, s53 // (op 3252) 
s_lshr_b32 s55, s54, 1 // (op 3252) 

s_add_u32 s54, s55, s53 // (op 3252) 


s_lshr_b32 s53, s54, s11 // (op 3252) 

s_and_b32 s54, s11, 2147483648 // (op 3252) 
s_cmp_eq_i32 s54, 0 // (op 3252) 
s_mov_b32 s55, scc // (op 3252) 

s_cmp_eq_i32 s55, 0 // (op 3252) 
s_cselect_b32 s54, s52, s53 // (op 3252) 


s_mul_hi_u32 s53, s54, s12 // (op 3252) 
s_sub_u32 s55, s54, s53 // (op 3252) 
s_lshr_b32 s84, s55, 1 // (op 3252) 

s_add_u32 s55, s84, s53 // (op 3252) 


s_lshr_b32 s53, s55, s10 // (op 3252) 

s_and_b32 s55, s10, 2147483648 // (op 3252) 
s_cmp_eq_i32 s55, 0 // (op 3252) 
s_mov_b32 s84, scc // (op 3252) 

s_cmp_eq_i32 s84, 0 // (op 3252) 
s_cselect_b32 s55, s54, s53 // (op 3252) 


s_mul_i32 s53, s55, s14 // (op 3252) 

s_sub_u32 s55, s54, s53 // (op 3252) 

v_lshl_add_u32 v14, s55, 7, v16 // (op 3252) 


v_mul_lo_u32 v16, v14, s44 // (op 3252) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 3252) 
s_mul_i32 s53, s54, s84 // (op 3252) 


s_sub_u32 s54, s52, s53 // (op 3252) 


v_lshl_add_u32 v14, s54, 2, v17 // (op 3252) 


v_add_u32 v17, v14, v16 // (op 3252) 


s_mul_i32 s53, s44, s23 // (op 3254) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 3254) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 3254) least significant
s_mul_hi_u32 s55, s44, s22 // (op 3254) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3254) most significant: sum
s_add_u32 s53, s53, s55 // (op 3254) most significant: sum


s_mov_b32 s80, s46 // (op 3254) reg expression
s_mov_b32 s81, s47 // (op 3254) reg expression
s_mov_b32 s82, s52 // (op 3254) reg expression
s_mov_b32 s83, 131072 // (op 3254) call()

v_and_b32 v14, -4, v1 // (op 3413) 
v_sub_u32 v16, v1, v14 // (op 3413) 

v_lshrrev_b32 v14, 2, v1 // (op 3413) 
v_lshl_add_u32 v18, v14, 3, v16 // (op 3413) 


s_and_b32 s52, s7, 4294967294 // (op 1435) 
s_cmp_lt_u32 s52, s7 // (op 1434) 







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1434 // (op 1434) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1434 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1434:

s_add_u32 s52, s52, 1 // (op 1436) 
v_mov_b64 v[20:21], s[44:45] // (op 1096) convert

v_mov_b32 v9, v17 // (op 1096) 
v_mov_b32 v14, v9 // (op 1096) 
buffer_load_ubyte v22, v14, s[80:83], 0 offen offset: 0 // (op 1096) Load value
v_add_u32 v14, v14, v20 // (op 1096) 
buffer_load_ubyte v23, v14, s[80:83], 0 offen offset: 0 // (op 1096) Load value



v_add_u32 v17, 4, v17 // (op 3244) 


s_barrier  // (op 3747) 
v_mov_b32 v9, v12 // (op 1102) 




buffer_load_dwordx4 v[24:27], v9, s[68:71], 0 offen // (op 1102) Load value
s_waitcnt vmcnt(2)
ds_write_b8 v18, v22 // (op 1097) Store local data 
buffer_load_dwordx4 v[28:31], v9, s[68:71], 0 offen offset:64 // (op 1102) Load value

v_add_u32 v12, 128, v12 // (op 2816) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v23 offset:4 // (op 1097) Store local data 



v_mov_b32 v9, v13 // (op 1104) 
v_mov_b32 v14, v9 // (op 1104) reg expression

buffer_load_dwordx4 v[20:23], v14, s[72:75], 0 offen // (op 1104) Load value



v_mov_b32 v14, v15 // (op 1106) 

buffer_load_ubyte v16, v14, s[76:79], 0 offen offset: 0 // (op 1106) Load value

v_add_u32 v15, 4, v15 // (op 3116) 


s_lshl_b64 s[54:55], s[36:37], 5 // (op 1216) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1216) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1216) 

v_add_u32 v14, v9, s54 // (op 1216) 


s_waitcnt lgkmcnt(0)
s_barrier  // (op 3750) 
buffer_load_dwordx4 v[32:35], v14, s[72:75], 0 offen // (op 1216) Load value






ds_read_u8 v14, v0 // (op 1099) Load local data 
s_lshl_b64 s[54:55], s[36:37], 5 // (op 1332) 



ds_read_u8 v19, v0 offset:128 // (op 1208) Load local data 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1332) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1332) 

s_lshl_b32 s53, s54, 1 // (op 1332) 

v_add_u32 v36, v9, s53 // (op 1332) 


buffer_load_dwordx4 v[38:41], v36, s[72:75], 0 offen // (op 1332) Load value

s_waitcnt vmcnt(2) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[20:23], a[0:3], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1100) 







ds_read_u8 v14, v0 offset:256 // (op 1324) Load local data 
s_lshl_b64 s[54:55], s[36:37], 5 // (op 1448) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 1448) 

s_lshr_b64 s[54:55], s[84:85], 3 // (op 1448) 

s_mul_i32 s53, 3, s54 // (op 1448) 

v_add_u32 v20, v9, s53 // (op 1448) 


buffer_load_dwordx4 v[42:45], v20, s[72:75], 0 offen // (op 1448) Load value

s_waitcnt vmcnt(2) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:31], v[32:35], a[4:7], v16, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1209) 






v_add_u32 v13, 64, v13 // (op 2963) 
ds_read_u8 v9, v0 offset:384 // (op 1440) Load local data 
s_waitcnt vmcnt(1) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:31], v[38:41], a[8:11], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1325) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:31], v[42:45], a[12:15], v16, v9 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 1441) 




s_cmp_lt_u32 s52, s7 // (op 1434) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1434 // (op 1434) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1434 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1434:






s_barrier  // (op 3970) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 251) 
s_cmp_lg_u32 s18, 0 // (op 277) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_277 // (op 277) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_277



buffer_store_dword a0, v8, s[64:67], 0 offen sc1 // (op 274) Store value
buffer_store_dword a1, v8, s[64:67], 0 offen offset:512 sc1 // (op 274) Store value
buffer_store_dword a2, v8, s[64:67], 0 offen offset:1024 sc1 // (op 274) Store value
buffer_store_dword a3, v8, s[64:67], 0 offen offset:1536 sc1 // (op 274) Store value



buffer_store_dword a4, v8, s[64:67], 0 offen offset:128 sc1 // (op 1629) Store value
buffer_store_dword a5, v8, s[64:67], 0 offen offset:640 sc1 // (op 1629) Store value
buffer_store_dword a6, v8, s[64:67], 0 offen offset:1152 sc1 // (op 1629) Store value
buffer_store_dword a7, v8, s[64:67], 0 offen offset:1664 sc1 // (op 1629) Store value



buffer_store_dword a8, v8, s[64:67], 0 offen offset:256 sc1 // (op 1630) Store value
buffer_store_dword a9, v8, s[64:67], 0 offen offset:768 sc1 // (op 1630) Store value
buffer_store_dword a10, v8, s[64:67], 0 offen offset:1280 sc1 // (op 1630) Store value
buffer_store_dword a11, v8, s[64:67], 0 offen offset:1792 sc1 // (op 1630) Store value



buffer_store_dword a12, v8, s[64:67], 0 offen offset:384 sc1 // (op 1631) Store value
buffer_store_dword a13, v8, s[64:67], 0 offen offset:896 sc1 // (op 1631) Store value
buffer_store_dword a14, v8, s[64:67], 0 offen offset:1408 sc1 // (op 1631) Store value
buffer_store_dword a15, v8, s[64:67], 0 offen offset:1920 sc1 // (op 1631) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 278) 
v_cmp_eq_i32 vcc, v1, 0 // (op 282) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282 // (op 282) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282
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
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282 // (op 282) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282:
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_277 // (op 277) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_277
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_277:
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_277:

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
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_293 // (op 293) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_293
s_mov_b32 s52, 0 // (op 259) call()
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s84, s54 // (op 258) convert

s_sub_u32 s54, s84, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 // (op 258) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 if false)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_258:
s_add_u32 s53, s16, 1 // (op 3585) 
v_lshrrev_b32 v8, 5, v1 // (op 3585) 
s_add_u32 s54, s53, s52 // (op 3585) 
s_lshl_b32 s54, s54, 5 // (op 3585) 

v_lshlrev_b32 v9, 2, v8 // (op 3585) 

v_add_lshl_u32 v8, s54, v9, 7 // (op 3585) 


v_and_b32 v9, -32, v1 // (op 3585) 
v_sub_u32 v12, v1, v9 // (op 3585) 

v_add_lshl_u32 v9, v8, v12, 2 // (op 3585) 


s_lshl_b32 s53, s15, 14 // (op 3587) 
s_mov_b32 s64, s28 // (op 3587) reg expression
s_mov_b32 s65, s29 // (op 3587) reg expression
s_mov_b32 s66, s53 // (op 3587) reg expression
s_mov_b32 s67, 131072 // (op 3587) call()

RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_297:
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
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 // (op 297) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 if true)
s_barrier  // (op 301) 
v_cmp_eq_i32 vcc, v1, 0 // (op 300) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300 // (op 300) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300
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
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300 // (op 300) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300:
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300:




buffer_load_dword v12, v9, s[64:67], 0 offen // (op 312) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:512 // (op 312) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:1024 // (op 312) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:1536 // (op 312) Load value
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




buffer_load_dword v12, v9, s[64:67], 0 offen offset:128 // (op 1651) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:640 // (op 1651) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:1152 // (op 1651) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:1664 // (op 1651) Load value
v_accvgpr_read v16, a4 // (op 1652) reg expression
v_accvgpr_read v17, a5 // (op 1652) reg expression
v_accvgpr_read v18, a6 // (op 1652) reg expression
v_accvgpr_read v19, a7 // (op 1652) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 1653) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 1653) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 1653) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 1653) 

v_accvgpr_write a4, v16 // (op 1654) reg expression
v_accvgpr_write a5, v17 // (op 1654) reg expression
v_accvgpr_write a6, v18 // (op 1654) reg expression
v_accvgpr_write a7, v19 // (op 1654) reg expression




buffer_load_dword v12, v9, s[64:67], 0 offen offset:256 // (op 1658) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:768 // (op 1658) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:1280 // (op 1658) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:1792 // (op 1658) Load value
v_accvgpr_read v16, a8 // (op 1659) reg expression
v_accvgpr_read v17, a9 // (op 1659) reg expression
v_accvgpr_read v18, a10 // (op 1659) reg expression
v_accvgpr_read v19, a11 // (op 1659) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 1660) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 1660) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 1660) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 1660) 

v_accvgpr_write a8, v16 // (op 1661) reg expression
v_accvgpr_write a9, v17 // (op 1661) reg expression
v_accvgpr_write a10, v18 // (op 1661) reg expression
v_accvgpr_write a11, v19 // (op 1661) reg expression




buffer_load_dword v12, v9, s[64:67], 0 offen offset:384 // (op 1665) Load value
buffer_load_dword v13, v9, s[64:67], 0 offen offset:896 // (op 1665) Load value
buffer_load_dword v14, v9, s[64:67], 0 offen offset:1408 // (op 1665) Load value
buffer_load_dword v15, v9, s[64:67], 0 offen offset:1920 // (op 1665) Load value
v_accvgpr_read v16, a12 // (op 1666) reg expression
v_accvgpr_read v17, a13 // (op 1666) reg expression
v_accvgpr_read v18, a14 // (op 1666) reg expression
v_accvgpr_read v19, a15 // (op 1666) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v16, v16, v12 // (op 1667) 
s_waitcnt vmcnt(2)
v_add_f32 v17, v17, v13 // (op 1667) 
s_waitcnt vmcnt(1)
v_add_f32 v18, v18, v14 // (op 1667) 
s_waitcnt vmcnt(0)
v_add_f32 v19, v19, v15 // (op 1667) 

v_accvgpr_write a12, v16 // (op 1668) reg expression
v_accvgpr_write a13, v17 // (op 1668) reg expression
v_accvgpr_write a14, v18 // (op 1668) reg expression
v_accvgpr_write a15, v19 // (op 1668) reg expression

s_add_u32 s52, s52, 1 // (op 260) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s84, s54 // (op 258) convert

s_sub_u32 s54, s84, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_258 // (op 258) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_258 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_258:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_293 // (op 293) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_293
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_293:
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_293:

s_cmp_eq_i32 s18, 0 // (op 336) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336 // (op 336) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 108) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_108 // (op 108) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_108
v_accvgpr_read v12, a0 // (op 38) 
v_accvgpr_read v13, a1 // (op 38) 
v_accvgpr_read v14, a2 // (op 38) 
v_accvgpr_read v15, a3 // (op 38) 
v_mul_f32 v16, v7, v12 // (op 38) 
v_mul_f32 v17, v7, v13 // (op 38) 
v_mul_f32 v18, v7, v14 // (op 38) 
v_mul_f32 v19, v7, v15 // (op 38) 

v_cvt_f16_f32 v12, v16 // (op 44) 
v_cvt_f16_f32 v13, v17 // (op 44) 
v_cvt_f16_f32 v14, v18 // (op 44) 
v_cvt_f16_f32 v15, v19 // (op 44) 


v_pack_b32_f16 v8, v12, v13 // (op 46) 
v_pack_b32_f16 v9, v14, v15 // (op 46) 

v_mov_b32 v16, v11 // (op 46) reg expression

buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 46) Store value



v_accvgpr_read v12, a4 // (op 1696) 
v_accvgpr_read v13, a5 // (op 1696) 
v_accvgpr_read v14, a6 // (op 1696) 
v_accvgpr_read v15, a7 // (op 1696) 
v_mul_f32 v16, v7, v12 // (op 1696) 
v_mul_f32 v17, v7, v13 // (op 1696) 
v_mul_f32 v18, v7, v14 // (op 1696) 
v_mul_f32 v19, v7, v15 // (op 1696) 

v_cvt_f16_f32 v12, v16 // (op 1698) 
v_cvt_f16_f32 v13, v17 // (op 1698) 
v_cvt_f16_f32 v14, v18 // (op 1698) 
v_cvt_f16_f32 v15, v19 // (op 1698) 


v_pack_b32_f16 v8, v12, v13 // (op 1699) 
v_pack_b32_f16 v9, v14, v15 // (op 1699) 

s_lshl_b32 s52, s24, 6 // (op 1699) 
v_add_u32 v16, v11, s52 // (op 1699) 


buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 1699) Store value



v_accvgpr_read v12, a8 // (op 1710) 
v_accvgpr_read v13, a9 // (op 1710) 
v_accvgpr_read v14, a10 // (op 1710) 
v_accvgpr_read v15, a11 // (op 1710) 
v_mul_f32 v16, v7, v12 // (op 1710) 
v_mul_f32 v17, v7, v13 // (op 1710) 
v_mul_f32 v18, v7, v14 // (op 1710) 
v_mul_f32 v19, v7, v15 // (op 1710) 

v_cvt_f16_f32 v12, v16 // (op 1712) 
v_cvt_f16_f32 v13, v17 // (op 1712) 
v_cvt_f16_f32 v14, v18 // (op 1712) 
v_cvt_f16_f32 v15, v19 // (op 1712) 


v_pack_b32_f16 v8, v12, v13 // (op 1713) 
v_pack_b32_f16 v9, v14, v15 // (op 1713) 

s_lshl_b32 s52, s24, 7 // (op 1713) 
v_add_u32 v16, v11, s52 // (op 1713) 


buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 1713) Store value



v_accvgpr_read v12, a12 // (op 1724) 
v_accvgpr_read v13, a13 // (op 1724) 
v_accvgpr_read v14, a14 // (op 1724) 
v_accvgpr_read v15, a15 // (op 1724) 
v_mul_f32 v16, v7, v12 // (op 1724) 
v_mul_f32 v17, v7, v13 // (op 1724) 
v_mul_f32 v18, v7, v14 // (op 1724) 
v_mul_f32 v19, v7, v15 // (op 1724) 

v_cvt_f16_f32 v12, v16 // (op 1726) 
v_cvt_f16_f32 v13, v17 // (op 1726) 
v_cvt_f16_f32 v14, v18 // (op 1726) 
v_cvt_f16_f32 v15, v19 // (op 1726) 


v_pack_b32_f16 v8, v12, v13 // (op 1727) 
v_pack_b32_f16 v9, v14, v15 // (op 1727) 

s_lshl_b32 s52, s24, 6 // (op 1727) 
s_mul_i32 s53, 3, s52 // (op 1727) 

v_add_u32 v16, v11, s53 // (op 1727) 


buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 1727) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_108 // (op 108) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_108
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_108:


v_mov_b32 v8, v10 // (op 121) reg expression

buffer_load_dwordx2 v[12:13], v8, s[56:59], 0 offen // (op 121) Load value

v_accvgpr_read v14, a0 // (op 117) 
v_accvgpr_read v15, a1 // (op 117) 
v_accvgpr_read v16, a2 // (op 117) 
v_accvgpr_read v17, a3 // (op 117) 
v_mul_f32 v18, v7, v14 // (op 117) 
v_mul_f32 v19, v7, v15 // (op 117) 
v_mul_f32 v20, v7, v16 // (op 117) 
v_mul_f32 v21, v7, v17 // (op 117) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v8, v12 // (op 122) 
v_lshrrev_b32 v9, 16, v12 // (op 122) 
v_cvt_f32_f16 v9, v9 // (op 122) 
v_mul_f32 v14, v6, v8 // (op 122) 
v_mul_f32 v15, v6, v9 // (op 122) 
v_cvt_f32_f16 v8, v13 // (op 122) 
v_lshrrev_b32 v9, 16, v13 // (op 122) 
v_cvt_f32_f16 v9, v9 // (op 122) 
v_mul_f32 v16, v6, v8 // (op 122) 
v_mul_f32 v17, v6, v9 // (op 122) 


v_add_f32 v22, v14, v18 // (op 118) 
v_add_f32 v23, v15, v19 // (op 118) 
v_add_f32 v24, v16, v20 // (op 118) 
v_add_f32 v25, v17, v21 // (op 118) 


v_cvt_f16_f32 v12, v22 // (op 119) 
v_cvt_f16_f32 v13, v23 // (op 119) 
v_cvt_f16_f32 v14, v24 // (op 119) 
v_cvt_f16_f32 v15, v25 // (op 119) 


v_pack_b32_f16 v8, v12, v13 // (op 120) 
v_pack_b32_f16 v9, v14, v15 // (op 120) 

v_mov_b32 v16, v11 // (op 120) reg expression

buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 120) Store value





s_lshl_b32 s52, s4, 6 // (op 1761) 
v_add_u32 v8, v10, s52 // (op 1761) 


buffer_load_dwordx2 v[12:13], v8, s[56:59], 0 offen // (op 1761) Load value

v_accvgpr_read v14, a4 // (op 1757) 
v_accvgpr_read v15, a5 // (op 1757) 
v_accvgpr_read v16, a6 // (op 1757) 
v_accvgpr_read v17, a7 // (op 1757) 
v_mul_f32 v18, v7, v14 // (op 1757) 
v_mul_f32 v19, v7, v15 // (op 1757) 
v_mul_f32 v20, v7, v16 // (op 1757) 
v_mul_f32 v21, v7, v17 // (op 1757) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v8, v12 // (op 1762) 
v_lshrrev_b32 v9, 16, v12 // (op 1762) 
v_cvt_f32_f16 v9, v9 // (op 1762) 
v_mul_f32 v14, v6, v8 // (op 1762) 
v_mul_f32 v15, v6, v9 // (op 1762) 
v_cvt_f32_f16 v8, v13 // (op 1762) 
v_lshrrev_b32 v9, 16, v13 // (op 1762) 
v_cvt_f32_f16 v9, v9 // (op 1762) 
v_mul_f32 v16, v6, v8 // (op 1762) 
v_mul_f32 v17, v6, v9 // (op 1762) 


v_add_f32 v22, v14, v18 // (op 1758) 
v_add_f32 v23, v15, v19 // (op 1758) 
v_add_f32 v24, v16, v20 // (op 1758) 
v_add_f32 v25, v17, v21 // (op 1758) 


v_cvt_f16_f32 v12, v22 // (op 1759) 
v_cvt_f16_f32 v13, v23 // (op 1759) 
v_cvt_f16_f32 v14, v24 // (op 1759) 
v_cvt_f16_f32 v15, v25 // (op 1759) 


v_pack_b32_f16 v8, v12, v13 // (op 1760) 
v_pack_b32_f16 v9, v14, v15 // (op 1760) 

s_lshl_b32 s52, s24, 6 // (op 1760) 
v_add_u32 v16, v11, s52 // (op 1760) 


buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 1760) Store value





s_lshl_b32 s52, s4, 7 // (op 1775) 
v_add_u32 v8, v10, s52 // (op 1775) 


buffer_load_dwordx2 v[12:13], v8, s[56:59], 0 offen // (op 1775) Load value

v_accvgpr_read v14, a8 // (op 1771) 
v_accvgpr_read v15, a9 // (op 1771) 
v_accvgpr_read v16, a10 // (op 1771) 
v_accvgpr_read v17, a11 // (op 1771) 
v_mul_f32 v18, v7, v14 // (op 1771) 
v_mul_f32 v19, v7, v15 // (op 1771) 
v_mul_f32 v20, v7, v16 // (op 1771) 
v_mul_f32 v21, v7, v17 // (op 1771) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v8, v12 // (op 1776) 
v_lshrrev_b32 v9, 16, v12 // (op 1776) 
v_cvt_f32_f16 v9, v9 // (op 1776) 
v_mul_f32 v14, v6, v8 // (op 1776) 
v_mul_f32 v15, v6, v9 // (op 1776) 
v_cvt_f32_f16 v8, v13 // (op 1776) 
v_lshrrev_b32 v9, 16, v13 // (op 1776) 
v_cvt_f32_f16 v9, v9 // (op 1776) 
v_mul_f32 v16, v6, v8 // (op 1776) 
v_mul_f32 v17, v6, v9 // (op 1776) 


v_add_f32 v22, v14, v18 // (op 1772) 
v_add_f32 v23, v15, v19 // (op 1772) 
v_add_f32 v24, v16, v20 // (op 1772) 
v_add_f32 v25, v17, v21 // (op 1772) 


v_cvt_f16_f32 v12, v22 // (op 1773) 
v_cvt_f16_f32 v13, v23 // (op 1773) 
v_cvt_f16_f32 v14, v24 // (op 1773) 
v_cvt_f16_f32 v15, v25 // (op 1773) 


v_pack_b32_f16 v8, v12, v13 // (op 1774) 
v_pack_b32_f16 v9, v14, v15 // (op 1774) 

s_lshl_b32 s52, s24, 7 // (op 1774) 
v_add_u32 v16, v11, s52 // (op 1774) 


buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 1774) Store value





s_lshl_b32 s52, s4, 6 // (op 1789) 
s_mul_i32 s53, 3, s52 // (op 1789) 

v_add_u32 v8, v10, s53 // (op 1789) 


buffer_load_dwordx2 v[12:13], v8, s[56:59], 0 offen // (op 1789) Load value

v_accvgpr_read v14, a12 // (op 1785) 
v_accvgpr_read v15, a13 // (op 1785) 
v_accvgpr_read v16, a14 // (op 1785) 
v_accvgpr_read v17, a15 // (op 1785) 
v_mul_f32 v18, v7, v14 // (op 1785) 
v_mul_f32 v19, v7, v15 // (op 1785) 
v_mul_f32 v20, v7, v16 // (op 1785) 
v_mul_f32 v21, v7, v17 // (op 1785) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v8, v12 // (op 1790) 
v_lshrrev_b32 v9, 16, v12 // (op 1790) 
v_cvt_f32_f16 v9, v9 // (op 1790) 
v_mul_f32 v14, v6, v8 // (op 1790) 
v_mul_f32 v15, v6, v9 // (op 1790) 
v_cvt_f32_f16 v8, v13 // (op 1790) 
v_lshrrev_b32 v9, 16, v13 // (op 1790) 
v_cvt_f32_f16 v9, v9 // (op 1790) 
v_mul_f32 v16, v6, v8 // (op 1790) 
v_mul_f32 v17, v6, v9 // (op 1790) 


v_add_f32 v22, v14, v18 // (op 1786) 
v_add_f32 v23, v15, v19 // (op 1786) 
v_add_f32 v24, v16, v20 // (op 1786) 
v_add_f32 v25, v17, v21 // (op 1786) 


v_cvt_f16_f32 v12, v22 // (op 1787) 
v_cvt_f16_f32 v13, v23 // (op 1787) 
v_cvt_f16_f32 v14, v24 // (op 1787) 
v_cvt_f16_f32 v15, v25 // (op 1787) 


v_pack_b32_f16 v8, v12, v13 // (op 1788) 
v_pack_b32_f16 v9, v14, v15 // (op 1788) 

s_lshl_b32 s52, s24, 6 // (op 1788) 
s_mul_i32 s53, 3, s52 // (op 1788) 

v_add_u32 v16, v11, s53 // (op 1788) 


buffer_store_dwordx2 v[8:9], v16, s[60:63], 0 offen // (op 1788) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_108:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336 // (op 336) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336:
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336:


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
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_249 // (op 249) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_249 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_249:











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
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 // (op 404) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 if false)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_404:
v_lshrrev_b32 v0, 7, v1 // (op 3383) 
s_mul_i32 s15, s16, s50 // (op 3383) 
s_add_u32 s15, s15, s0 // (op 3383) 
v_and_b32 v5, 63, v1 // (op 3383) 
s_add_u32 s6, s15, s9 // (op 3383) 

v_lshrrev_b32 v8, 4, v5 // (op 3383) 
s_mul_hi_u32 s15, s6, s13 // (op 3383) 
v_lshlrev_b32 v9, 2, v8 // (op 3383) 

s_sub_u32 s7, s6, s15 // (op 3383) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 3383) 


s_lshr_b32 s8, s7, 1 // (op 3383) 

v_lshrrev_b32 v0, 2, v5 // (op 3383) 

s_add_u32 s7, s8, s15 // (op 3383) 


v_and_b32 v5, 3, v0 // (op 3383) 

s_lshr_b32 s8, s7, s11 // (op 3383) 

v_and_b32 v0, 3, v1 // (op 3383) 
s_and_b32 s7, s11, 2147483648 // (op 3383) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 3383) 


s_cmp_eq_i32 s7, 0 // (op 3383) 
s_mov_b32 s15, scc // (op 3383) 

v_lshrrev_b32 v0, 6, v1 // (op 3383) 
s_cmp_eq_i32 s15, 0 // (op 3383) 
s_cselect_b32 s7, s6, s8 // (op 3383) 



v_and_b32 v5, 1, v0 // (op 3383) 

s_mul_hi_u32 s6, s7, s12 // (op 3383) 
v_lshl_add_u32 v0, v5, 4, v9 // (op 3383) 


s_sub_u32 s8, s7, s6 // (op 3383) 
s_lshr_b32 s15, s8, 1 // (op 3383) 

s_add_u32 s8, s15, s6 // (op 3383) 


s_lshr_b32 s6, s8, s10 // (op 3383) 

s_and_b32 s8, s10, 2147483648 // (op 3383) 
s_cmp_eq_i32 s8, 0 // (op 3383) 
s_mov_b32 s15, scc // (op 3383) 

s_cmp_eq_i32 s15, 0 // (op 3383) 
s_cselect_b32 s8, s7, s6 // (op 3383) 


s_mul_i32 s6, s8, s14 // (op 3383) 
v_lshl_add_u32 v5, s8, 5, v8 // (op 3383) 


s_sub_u32 s8, s7, s6 // (op 3383) 


v_lshl_add_u32 v8, s8, 7, v0 // (op 3383) 


v_mul_lo_u32 v0, v8, s4 // (op 3383) 

v_add_lshl_u32 v8, v5, v0, 1 // (op 3383) 


s_mul_i32 s19, s4, s23 // (op 3385) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 3385) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 3385) least significant
s_mul_hi_u32 s6, s4, s22 // (op 3385) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3385) most significant: sum
s_add_u32 s19, s19, s6 // (op 3385) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 3385) 

s_mov_b32 s56, s20 // (op 3385) reg expression
s_mov_b32 s57, s21 // (op 3385) reg expression
s_mov_b32 s58, s28 // (op 3385) reg expression
s_mov_b32 s59, 131072 // (op 3385) call()

v_lshrrev_b32 v0, 7, v1 // (op 3483) 
s_mul_i32 s15, s16, s50 // (op 3483) 
s_add_u32 s15, s15, s0 // (op 3483) 
v_and_b32 v5, 63, v1 // (op 3483) 
s_add_u32 s6, s15, s9 // (op 3483) 

v_lshrrev_b32 v9, 4, v5 // (op 3483) 
s_mul_hi_u32 s15, s6, s13 // (op 3483) 
v_lshlrev_b32 v10, 2, v9 // (op 3483) 

s_sub_u32 s7, s6, s15 // (op 3483) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 3483) 


s_lshr_b32 s8, s7, 1 // (op 3483) 

v_lshrrev_b32 v0, 2, v5 // (op 3483) 

s_add_u32 s7, s8, s15 // (op 3483) 


v_and_b32 v5, 3, v0 // (op 3483) 

s_lshr_b32 s8, s7, s11 // (op 3483) 

v_and_b32 v0, 3, v1 // (op 3483) 
s_and_b32 s7, s11, 2147483648 // (op 3483) 
v_lshl_add_u32 v10, v5, 2, v0 // (op 3483) 


s_cmp_eq_i32 s7, 0 // (op 3483) 
s_mov_b32 s15, scc // (op 3483) 

v_lshrrev_b32 v0, 6, v1 // (op 3483) 
s_cmp_eq_i32 s15, 0 // (op 3483) 
s_cselect_b32 s7, s6, s8 // (op 3483) 



v_and_b32 v5, 1, v0 // (op 3483) 

s_mul_hi_u32 s6, s7, s12 // (op 3483) 
v_lshl_add_u32 v0, v5, 4, v10 // (op 3483) 


s_sub_u32 s8, s7, s6 // (op 3483) 
s_lshr_b32 s15, s8, 1 // (op 3483) 

s_add_u32 s8, s15, s6 // (op 3483) 


s_lshr_b32 s6, s8, s10 // (op 3483) 

s_and_b32 s8, s10, 2147483648 // (op 3483) 
s_cmp_eq_i32 s8, 0 // (op 3483) 
s_mov_b32 s15, scc // (op 3483) 

s_cmp_eq_i32 s15, 0 // (op 3483) 
s_cselect_b32 s8, s7, s6 // (op 3483) 


s_mul_i32 s6, s8, s14 // (op 3483) 
v_lshl_add_u32 v5, s8, 5, v9 // (op 3483) 


s_sub_u32 s8, s7, s6 // (op 3483) 


v_lshl_add_u32 v9, s8, 7, v0 // (op 3483) 


v_mul_lo_u32 v0, v9, s24 // (op 3483) 

v_add_lshl_u32 v9, v5, v0, 1 // (op 3483) 


s_mul_i32 s19, s24, s23 // (op 3485) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 3485) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 3485) least significant
s_mul_hi_u32 s6, s24, s22 // (op 3485) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3485) most significant: sum
s_add_u32 s19, s19, s6 // (op 3485) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 3485) 

s_mov_b32 s60, s26 // (op 3485) reg expression
s_mov_b32 s61, s27 // (op 3485) reg expression
s_mov_b32 s62, s28 // (op 3485) reg expression
s_mov_b32 s63, 131072 // (op 3485) call()

v_accvgpr_write a0, 0.00000 // (op 157) call()
v_accvgpr_write a1, 0.00000 // (op 157) call()
v_accvgpr_write a2, 0.00000 // (op 157) call()
v_accvgpr_write a3, 0.00000 // (op 157) call()
v_accvgpr_write a4, 0.00000 // (op 517) call()
v_accvgpr_write a5, 0.00000 // (op 517) call()
v_accvgpr_write a6, 0.00000 // (op 517) call()
v_accvgpr_write a7, 0.00000 // (op 517) call()
v_accvgpr_write a8, 0.00000 // (op 633) call()
v_accvgpr_write a9, 0.00000 // (op 633) call()
v_accvgpr_write a10, 0.00000 // (op 633) call()
v_accvgpr_write a11, 0.00000 // (op 633) call()
v_accvgpr_write a12, 0.00000 // (op 749) call()
v_accvgpr_write a13, 0.00000 // (op 749) call()
v_accvgpr_write a14, 0.00000 // (op 749) call()
v_accvgpr_write a15, 0.00000 // (op 749) call()
v_lshrrev_b32 v0, 7, v1 // (op 2855) 
s_mul_i32 s15, s16, s50 // (op 2855) 
s_add_u32 s15, s15, s0 // (op 2855) 
v_and_b32 v5, 15, v1 // (op 2855) 
s_add_u32 s6, s15, s9 // (op 2855) 

v_lshl_add_u32 v10, v0, 4, v5 // (op 2855) 


s_mul_hi_u32 s15, s6, s13 // (op 2855) 
v_and_b32 v0, 48, v1 // (op 2855) 
s_sub_u32 s7, s6, s15 // (op 2855) 
s_lshr_b32 s8, s7, 1 // (op 2855) 

s_add_u32 s7, s8, s15 // (op 2855) 


s_lshr_b32 s8, s7, s11 // (op 2855) 

s_and_b32 s7, s11, 2147483648 // (op 2855) 
s_cmp_eq_i32 s7, 0 // (op 2855) 
s_mov_b32 s15, scc // (op 2855) 

s_cmp_eq_i32 s15, 0 // (op 2855) 
s_cselect_b32 s7, s6, s8 // (op 2855) 


s_mul_hi_u32 s8, s7, s12 // (op 2855) 
s_sub_u32 s15, s7, s8 // (op 2855) 
s_lshr_b32 s18, s15, 1 // (op 2855) 

s_add_u32 s15, s18, s8 // (op 2855) 


s_lshr_b32 s8, s15, s10 // (op 2855) 

s_and_b32 s15, s10, 2147483648 // (op 2855) 
s_cmp_eq_i32 s15, 0 // (op 2855) 
s_mov_b32 s18, scc // (op 2855) 

s_cmp_eq_i32 s18, 0 // (op 2855) 
s_cselect_b32 s15, s7, s8 // (op 2855) 


v_lshl_add_u32 v5, s15, 5, v10 // (op 2855) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2855) 
s_mul_i32 s8, s7, s18 // (op 2855) 


s_sub_u32 s7, s6, s8 // (op 2855) 


v_lshl_add_u32 v10, s7, 7, v0 // (op 2855) 


v_mul_lo_u32 v0, v5, s30 // (op 2855) 
v_add_u32 v0, v0, v10 // (op 2855) 


s_mul_i32 s19, s30, s35 // (op 2857) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2857) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2857) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2857) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2857) most significant: sum
s_add_u32 s19, s19, s6 // (op 2857) most significant: sum


s_mov_b32 s68, s32 // (op 2857) reg expression
s_mov_b32 s69, s33 // (op 2857) reg expression
s_mov_b32 s70, s18 // (op 2857) reg expression
s_mov_b32 s71, 131072 // (op 2857) call()

v_lshrrev_b32 v5, 7, v1 // (op 2874) 
s_mul_i32 s15, s16, s50 // (op 2874) 
s_add_u32 s15, s15, s0 // (op 2874) 
v_and_b32 v10, 15, v1 // (op 2874) 
s_add_u32 s6, s15, s9 // (op 2874) 

v_lshl_add_u32 v11, v5, 4, v10 // (op 2874) 


s_mul_hi_u32 s15, s6, s13 // (op 2874) 
v_and_b32 v5, 48, v1 // (op 2874) 
s_sub_u32 s7, s6, s15 // (op 2874) 
s_lshr_b32 s8, s7, 1 // (op 2874) 

s_add_u32 s7, s8, s15 // (op 2874) 


s_lshr_b32 s8, s7, s11 // (op 2874) 

s_and_b32 s7, s11, 2147483648 // (op 2874) 
s_cmp_eq_i32 s7, 0 // (op 2874) 
s_mov_b32 s15, scc // (op 2874) 

s_cmp_eq_i32 s15, 0 // (op 2874) 
s_cselect_b32 s7, s6, s8 // (op 2874) 


s_mul_hi_u32 s8, s7, s12 // (op 2874) 
s_sub_u32 s15, s7, s8 // (op 2874) 
s_lshr_b32 s18, s15, 1 // (op 2874) 

s_add_u32 s15, s18, s8 // (op 2874) 


s_lshr_b32 s8, s15, s10 // (op 2874) 

s_and_b32 s15, s10, 2147483648 // (op 2874) 
s_cmp_eq_i32 s15, 0 // (op 2874) 
s_mov_b32 s18, scc // (op 2874) 

s_cmp_eq_i32 s18, 0 // (op 2874) 
s_cselect_b32 s15, s7, s8 // (op 2874) 


v_lshl_add_u32 v10, s15, 5, v11 // (op 2874) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 2874) 
s_mul_i32 s8, s7, s18 // (op 2874) 


s_sub_u32 s7, s6, s8 // (op 2874) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 2874) 


v_mul_lo_u32 v5, v10, s30 // (op 2874) 
v_add_u32 v5, v5, v11 // (op 2874) 


s_mul_i32 s19, s30, s35 // (op 2876) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2876) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2876) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2876) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2876) most significant: sum
s_add_u32 s19, s19, s6 // (op 2876) most significant: sum


s_mov_b32 s68, s32 // (op 2876) reg expression
s_mov_b32 s69, s33 // (op 2876) reg expression
s_mov_b32 s70, s18 // (op 2876) reg expression
s_mov_b32 s71, 131072 // (op 2876) call()

v_lshrrev_b32 v10, 6, v1 // (op 3004) 
s_mul_i32 s15, s16, s50 // (op 3004) 
s_add_u32 s15, s15, s0 // (op 3004) 
v_and_b32 v11, 1, v10 // (op 3004) 

s_add_u32 s6, s15, s9 // (op 3004) 

v_and_b32 v10, 15, v1 // (op 3004) 
s_mul_hi_u32 s15, s6, s13 // (op 3004) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 3004) 


s_sub_u32 s7, s6, s15 // (op 3004) 
v_and_b32 v10, 63, v1 // (op 3004) 
s_lshr_b32 s8, s7, 1 // (op 3004) 

v_lshrrev_b32 v11, 4, v10 // (op 3004) 

s_add_u32 s7, s8, s15 // (op 3004) 


v_lshlrev_b32 v10, 5, v11 // (op 3004) 

s_lshr_b32 s8, s7, s11 // (op 3004) 

s_and_b32 s7, s11, 2147483648 // (op 3004) 
s_cmp_eq_i32 s7, 0 // (op 3004) 
s_mov_b32 s15, scc // (op 3004) 

s_cmp_eq_i32 s15, 0 // (op 3004) 
s_cselect_b32 s7, s6, s8 // (op 3004) 


s_mul_hi_u32 s8, s7, s12 // (op 3004) 
s_sub_u32 s15, s7, s8 // (op 3004) 
s_lshr_b32 s18, s15, 1 // (op 3004) 

s_add_u32 s15, s18, s8 // (op 3004) 


s_lshr_b32 s8, s15, s10 // (op 3004) 

s_and_b32 s15, s10, 2147483648 // (op 3004) 
s_cmp_eq_i32 s15, 0 // (op 3004) 
s_mov_b32 s18, scc // (op 3004) 

s_cmp_eq_i32 s18, 0 // (op 3004) 
s_cselect_b32 s15, s7, s8 // (op 3004) 


s_mul_i32 s8, s15, s14 // (op 3004) 

s_sub_u32 s15, s7, s8 // (op 3004) 

v_lshl_add_u32 v11, s15, 7, v12 // (op 3004) 


v_mul_lo_u32 v13, v11, s37 // (op 3004) most significant: low of low * high
v_mul_hi_u32 v14, v11, s36 // (op 3004) most significant: high of low * low
v_mul_lo_u32 v12, v11, s36 // (op 3004) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 3004) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3004) 
s_mul_i32 s8, s7, s18 // (op 3004) 


s_sub_u32 s7, s6, s8 // (op 3004) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 3004) 


v_mov_b32 v15, 0 // (op 3004) convert
v_mov_b32 v14, v11 // (op 3004) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 3004) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 3004) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 3004) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 3004) 

v_mov_b32 v10, v12 // (op 3004) convert

s_mul_i32 s19, s36, s23 // (op 3006) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3006) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3006) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3006) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3006) most significant: sum
s_add_u32 s19, s19, s6 // (op 3006) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3006) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 3006) 

s_mov_b32 s72, s38 // (op 3006) reg expression
s_mov_b32 s73, s39 // (op 3006) reg expression
s_mov_b32 s74, s18 // (op 3006) reg expression
s_mov_b32 s75, 131072 // (op 3006) call()

v_lshrrev_b32 v11, 6, v1 // (op 3025) 
s_mul_i32 s15, s16, s50 // (op 3025) 
s_add_u32 s15, s15, s0 // (op 3025) 
v_and_b32 v12, 1, v11 // (op 3025) 

s_add_u32 s6, s15, s9 // (op 3025) 

v_and_b32 v11, 15, v1 // (op 3025) 
s_mul_hi_u32 s15, s6, s13 // (op 3025) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 3025) 


s_sub_u32 s7, s6, s15 // (op 3025) 
v_and_b32 v11, 63, v1 // (op 3025) 
s_lshr_b32 s8, s7, 1 // (op 3025) 

v_lshrrev_b32 v12, 4, v11 // (op 3025) 

s_add_u32 s7, s8, s15 // (op 3025) 


v_lshlrev_b32 v11, 5, v12 // (op 3025) 

s_lshr_b32 s8, s7, s11 // (op 3025) 

s_and_b32 s7, s11, 2147483648 // (op 3025) 
s_cmp_eq_i32 s7, 0 // (op 3025) 
s_mov_b32 s15, scc // (op 3025) 

s_cmp_eq_i32 s15, 0 // (op 3025) 
s_cselect_b32 s7, s6, s8 // (op 3025) 


s_mul_hi_u32 s8, s7, s12 // (op 3025) 
s_sub_u32 s15, s7, s8 // (op 3025) 
s_lshr_b32 s18, s15, 1 // (op 3025) 

s_add_u32 s15, s18, s8 // (op 3025) 


s_lshr_b32 s8, s15, s10 // (op 3025) 

s_and_b32 s15, s10, 2147483648 // (op 3025) 
s_cmp_eq_i32 s15, 0 // (op 3025) 
s_mov_b32 s18, scc // (op 3025) 

s_cmp_eq_i32 s18, 0 // (op 3025) 
s_cselect_b32 s15, s7, s8 // (op 3025) 


s_mul_i32 s8, s15, s14 // (op 3025) 

s_sub_u32 s15, s7, s8 // (op 3025) 

v_lshl_add_u32 v12, s15, 7, v13 // (op 3025) 


v_mul_lo_u32 v15, v12, s37 // (op 3025) most significant: low of low * high
v_mul_hi_u32 v13, v12, s36 // (op 3025) most significant: high of low * low
v_mul_lo_u32 v14, v12, s36 // (op 3025) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 3025) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3025) 
s_mul_i32 s8, s7, s18 // (op 3025) 


s_sub_u32 s7, s6, s8 // (op 3025) 


v_lshl_add_u32 v12, s7, 7, v11 // (op 3025) 


v_mov_b32 v17, 0 // (op 3025) convert
v_mov_b32 v16, v12 // (op 3025) convert

v_add_co_u32 v12, s[18:19], v16, v14 // (op 3025) least significant half
v_addc_co_u32 v13, s[18:19], v17, v15, s[18:19] // (op 3025) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 3025) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3025) 

v_mov_b32 v11, v14 // (op 3025) convert

s_mul_i32 s19, s36, s23 // (op 3027) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 3027) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 3027) least significant
s_mul_hi_u32 s6, s36, s22 // (op 3027) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3027) most significant: sum
s_add_u32 s19, s19, s6 // (op 3027) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 3027) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 3027) 

s_mov_b32 s72, s38 // (op 3027) reg expression
s_mov_b32 s73, s39 // (op 3027) reg expression
s_mov_b32 s74, s18 // (op 3027) reg expression
s_mov_b32 s75, 131072 // (op 3027) call()

v_lshrrev_b32 v12, 7, v1 // (op 3147) 
s_mul_i32 s15, s16, s50 // (op 3147) 
s_add_u32 s15, s15, s0 // (op 3147) 
v_and_b32 v13, 15, v1 // (op 3147) 
s_add_u32 s6, s15, s9 // (op 3147) 

v_lshl_add_u32 v14, v12, 4, v13 // (op 3147) 


s_mul_hi_u32 s15, s6, s13 // (op 3147) 
v_and_b32 v12, 63, v1 // (op 3147) 
s_sub_u32 s7, s6, s15 // (op 3147) 
v_lshrrev_b32 v13, 4, v12 // (op 3147) 

s_lshr_b32 s8, s7, 1 // (op 3147) 

s_add_u32 s7, s8, s15 // (op 3147) 


s_lshr_b32 s8, s7, s11 // (op 3147) 

s_and_b32 s7, s11, 2147483648 // (op 3147) 
s_cmp_eq_i32 s7, 0 // (op 3147) 
s_mov_b32 s15, scc // (op 3147) 

s_cmp_eq_i32 s15, 0 // (op 3147) 
s_cselect_b32 s7, s6, s8 // (op 3147) 


s_mul_hi_u32 s8, s7, s12 // (op 3147) 
s_sub_u32 s15, s7, s8 // (op 3147) 
s_lshr_b32 s18, s15, 1 // (op 3147) 

s_add_u32 s15, s18, s8 // (op 3147) 


s_lshr_b32 s8, s15, s10 // (op 3147) 

s_and_b32 s15, s10, 2147483648 // (op 3147) 
s_cmp_eq_i32 s15, 0 // (op 3147) 
s_mov_b32 s18, scc // (op 3147) 

s_cmp_eq_i32 s18, 0 // (op 3147) 
s_cselect_b32 s15, s7, s8 // (op 3147) 


v_lshl_add_u32 v12, s15, 5, v14 // (op 3147) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3147) 
s_mul_i32 s8, s7, s18 // (op 3147) 


s_sub_u32 s7, s6, s8 // (op 3147) 


v_lshl_add_u32 v14, s7, 2, v13 // (op 3147) 


v_mul_lo_u32 v13, v12, s40 // (op 3147) 
v_add_u32 v13, v13, v14 // (op 3147) 


s_mul_i32 s19, s40, s35 // (op 3149) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3149) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3149) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3149) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3149) most significant: sum
s_add_u32 s19, s19, s6 // (op 3149) most significant: sum


s_mov_b32 s76, s42 // (op 3149) reg expression
s_mov_b32 s77, s43 // (op 3149) reg expression
s_mov_b32 s78, s18 // (op 3149) reg expression
s_mov_b32 s79, 131072 // (op 3149) call()

v_lshrrev_b32 v12, 7, v1 // (op 3166) 
s_mul_i32 s15, s16, s50 // (op 3166) 
s_add_u32 s15, s15, s0 // (op 3166) 
v_and_b32 v14, 15, v1 // (op 3166) 
s_add_u32 s6, s15, s9 // (op 3166) 

v_lshl_add_u32 v15, v12, 4, v14 // (op 3166) 


s_mul_hi_u32 s15, s6, s13 // (op 3166) 
v_and_b32 v12, 63, v1 // (op 3166) 
s_sub_u32 s7, s6, s15 // (op 3166) 
v_lshrrev_b32 v14, 4, v12 // (op 3166) 

s_lshr_b32 s8, s7, 1 // (op 3166) 

s_add_u32 s7, s8, s15 // (op 3166) 


s_lshr_b32 s8, s7, s11 // (op 3166) 

s_and_b32 s7, s11, 2147483648 // (op 3166) 
s_cmp_eq_i32 s7, 0 // (op 3166) 
s_mov_b32 s15, scc // (op 3166) 

s_cmp_eq_i32 s15, 0 // (op 3166) 
s_cselect_b32 s7, s6, s8 // (op 3166) 


s_mul_hi_u32 s8, s7, s12 // (op 3166) 
s_sub_u32 s15, s7, s8 // (op 3166) 
s_lshr_b32 s18, s15, 1 // (op 3166) 

s_add_u32 s15, s18, s8 // (op 3166) 


s_lshr_b32 s8, s15, s10 // (op 3166) 

s_and_b32 s15, s10, 2147483648 // (op 3166) 
s_cmp_eq_i32 s15, 0 // (op 3166) 
s_mov_b32 s18, scc // (op 3166) 

s_cmp_eq_i32 s18, 0 // (op 3166) 
s_cselect_b32 s15, s7, s8 // (op 3166) 


v_lshl_add_u32 v12, s15, 5, v15 // (op 3166) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3166) 
s_mul_i32 s8, s7, s18 // (op 3166) 


s_sub_u32 s7, s6, s8 // (op 3166) 


v_lshl_add_u32 v15, s7, 2, v14 // (op 3166) 


v_mul_lo_u32 v14, v12, s40 // (op 3166) 
v_add_u32 v14, v14, v15 // (op 3166) 


s_mul_i32 s19, s40, s35 // (op 3168) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3168) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3168) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3168) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3168) most significant: sum
s_add_u32 s19, s19, s6 // (op 3168) most significant: sum


s_mov_b32 s76, s42 // (op 3168) reg expression
s_mov_b32 s77, s43 // (op 3168) reg expression
s_mov_b32 s78, s18 // (op 3168) reg expression
s_mov_b32 s79, 131072 // (op 3168) call()

v_lshrrev_b32 v12, 2, v1 // (op 3275) 
s_mul_i32 s15, s16, s50 // (op 3275) 
s_add_u32 s15, s15, s0 // (op 3275) 
v_lshlrev_b32 v15, 1, v12 // (op 3275) 

s_add_u32 s6, s15, s9 // (op 3275) 

v_and_b32 v12, -4, v1 // (op 3275) 
s_mul_hi_u32 s15, s6, s13 // (op 3275) 
v_sub_u32 v16, v1, v12 // (op 3275) 

s_sub_u32 s7, s6, s15 // (op 3275) 
s_lshr_b32 s8, s7, 1 // (op 3275) 

s_add_u32 s7, s8, s15 // (op 3275) 


s_lshr_b32 s8, s7, s11 // (op 3275) 

s_and_b32 s7, s11, 2147483648 // (op 3275) 
s_cmp_eq_i32 s7, 0 // (op 3275) 
s_mov_b32 s15, scc // (op 3275) 

s_cmp_eq_i32 s15, 0 // (op 3275) 
s_cselect_b32 s7, s6, s8 // (op 3275) 


s_mul_hi_u32 s8, s7, s12 // (op 3275) 
s_sub_u32 s15, s7, s8 // (op 3275) 
s_lshr_b32 s18, s15, 1 // (op 3275) 

s_add_u32 s15, s18, s8 // (op 3275) 


s_lshr_b32 s8, s15, s10 // (op 3275) 

s_and_b32 s15, s10, 2147483648 // (op 3275) 
s_cmp_eq_i32 s15, 0 // (op 3275) 
s_mov_b32 s18, scc // (op 3275) 

s_cmp_eq_i32 s18, 0 // (op 3275) 
s_cselect_b32 s15, s7, s8 // (op 3275) 


s_mul_i32 s8, s15, s14 // (op 3275) 

s_sub_u32 s15, s7, s8 // (op 3275) 

v_lshl_add_u32 v12, s15, 7, v15 // (op 3275) 


v_mul_lo_u32 v15, v12, s44 // (op 3275) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3275) 
s_mul_i32 s8, s7, s18 // (op 3275) 


s_sub_u32 s7, s6, s8 // (op 3275) 


v_lshl_add_u32 v12, s7, 2, v16 // (op 3275) 


v_add_u32 v16, v12, v15 // (op 3275) 


s_mul_i32 s19, s44, s23 // (op 3277) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3277) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3277) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3277) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3277) most significant: sum
s_add_u32 s19, s19, s6 // (op 3277) most significant: sum


s_mov_b32 s80, s46 // (op 3277) reg expression
s_mov_b32 s81, s47 // (op 3277) reg expression
s_mov_b32 s82, s18 // (op 3277) reg expression
s_mov_b32 s83, 131072 // (op 3277) call()

v_lshrrev_b32 v12, 2, v1 // (op 3294) 
s_mul_i32 s15, s16, s50 // (op 3294) 
s_add_u32 s15, s15, s0 // (op 3294) 
v_lshlrev_b32 v15, 1, v12 // (op 3294) 

s_add_u32 s6, s15, s9 // (op 3294) 

v_and_b32 v12, -4, v1 // (op 3294) 
s_mul_hi_u32 s15, s6, s13 // (op 3294) 
v_sub_u32 v17, v1, v12 // (op 3294) 

s_sub_u32 s7, s6, s15 // (op 3294) 
s_lshr_b32 s8, s7, 1 // (op 3294) 

s_add_u32 s7, s8, s15 // (op 3294) 


s_lshr_b32 s8, s7, s11 // (op 3294) 

s_and_b32 s7, s11, 2147483648 // (op 3294) 
s_cmp_eq_i32 s7, 0 // (op 3294) 
s_mov_b32 s15, scc // (op 3294) 

s_cmp_eq_i32 s15, 0 // (op 3294) 
s_cselect_b32 s7, s6, s8 // (op 3294) 


s_mul_hi_u32 s8, s7, s12 // (op 3294) 
s_sub_u32 s15, s7, s8 // (op 3294) 
s_lshr_b32 s18, s15, 1 // (op 3294) 

s_add_u32 s15, s18, s8 // (op 3294) 


s_lshr_b32 s8, s15, s10 // (op 3294) 

s_and_b32 s15, s10, 2147483648 // (op 3294) 
s_cmp_eq_i32 s15, 0 // (op 3294) 
s_mov_b32 s18, scc // (op 3294) 

s_cmp_eq_i32 s18, 0 // (op 3294) 
s_cselect_b32 s15, s7, s8 // (op 3294) 


s_mul_i32 s8, s15, s14 // (op 3294) 

s_sub_u32 s15, s7, s8 // (op 3294) 

v_lshl_add_u32 v12, s15, 7, v15 // (op 3294) 


v_mul_lo_u32 v15, v12, s44 // (op 3294) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3294) 
s_mul_i32 s8, s7, s18 // (op 3294) 


s_sub_u32 s7, s6, s8 // (op 3294) 


v_lshl_add_u32 v12, s7, 2, v17 // (op 3294) 


v_add_u32 v17, v12, v15 // (op 3294) 


s_mul_i32 s19, s44, s23 // (op 3296) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3296) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3296) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3296) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3296) most significant: sum
s_add_u32 s19, s19, s6 // (op 3296) most significant: sum


s_mov_b32 s80, s46 // (op 3296) reg expression
s_mov_b32 s81, s47 // (op 3296) reg expression
s_mov_b32 s82, s18 // (op 3296) reg expression
s_mov_b32 s83, 131072 // (op 3296) call()




buffer_load_dwordx4 v[18:21], v0, s[68:71], 0 offen // (op 2467) Load value
buffer_load_dwordx4 v[22:25], v0, s[68:71], 0 offen offset:64 // (op 2467) Load value


v_mov_b32 v12, v10 // (op 2473) reg expression

buffer_load_dwordx4 v[26:29], v12, s[72:75], 0 offen // (op 2473) Load value




buffer_load_ubyte v12, v13, s[76:79], 0 offen offset: 0 // (op 2479) Load value


s_lshl_b64 s[18:19], s[36:37], 5 // (op 2485) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2485) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2485) 

v_add_u32 v15, v10, s18 // (op 2485) 


buffer_load_dwordx4 v[30:33], v15, s[72:75], 0 offen // (op 2485) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 2492) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2492) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2492) 

s_lshl_b32 s15, s18, 1 // (op 2492) 

v_add_u32 v15, v10, s15 // (op 2492) 


buffer_load_dwordx4 v[34:37], v15, s[72:75], 0 offen // (op 2492) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 2499) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2499) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2499) 

s_mul_i32 s15, 3, s18 // (op 2499) 

v_add_u32 v15, v10, s15 // (op 2499) 


buffer_load_dwordx4 v[38:41], v15, s[72:75], 0 offen // (op 2499) Load value


v_mov_b64 v[42:43], s[44:45] // (op 2504) convert

v_mov_b32 v15, v16 // (op 2504) 
buffer_load_ubyte v44, v15, s[80:83], 0 offen offset: 0 // (op 2504) Load value
v_add_u32 v15, v15, v42 // (op 2504) 
buffer_load_ubyte v45, v15, s[80:83], 0 offen offset: 0 // (op 2504) Load value





buffer_load_dwordx4 v[46:49], v0, s[68:71], 0 offen offset:128 // (op 2508) Load value
buffer_load_dwordx4 v[50:53], v0, s[68:71], 0 offen offset:192 // (op 2508) Load value



v_add_u32 v0, 64, v10 // (op 2514) 

buffer_load_dwordx4 v[54:57], v0, s[72:75], 0 offen // (op 2514) Load value




buffer_load_ubyte v0, v13, s[76:79], 0 offen offset:4 // (op 2520) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 2526) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2526) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2526) 

s_add_u32 s15, s18, 64 // (op 2526) 

v_add_u32 v13, v10, s15 // (op 2526) 


buffer_load_dwordx4 v[58:61], v13, s[72:75], 0 offen // (op 2526) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 2533) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2533) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2533) 

s_lshl_b32 s15, s18, 1 // (op 2533) 
s_add_u32 s15, s15, 64 // (op 2533) 

v_add_u32 v13, v10, s15 // (op 2533) 


buffer_load_dwordx4 v[62:65], v13, s[72:75], 0 offen // (op 2533) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 2540) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 2540) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 2540) 

s_mul_i32 s15, 3, s18 // (op 2540) 
s_add_u32 s15, s15, 64 // (op 2540) 

v_add_u32 v13, v10, s15 // (op 2540) 


buffer_load_dwordx4 v[66:69], v13, s[72:75], 0 offen // (op 2540) Load value



v_mov_b64 v[42:43], s[44:45] // (op 2545) convert

v_mov_b32 v10, v16 // (op 2545) 
buffer_load_ubyte v70, v10, s[80:83], 0 offen offset:4 // (op 2545) Load value
v_add_u32 v10, v10, v42 // (op 2545) 
buffer_load_ubyte v71, v10, s[80:83], 0 offen offset:4 // (op 2545) Load value



v_and_b32 v10, -4, v1 // (op 3529) 
v_sub_u32 v13, v1, v10 // (op 3529) 

v_lshrrev_b32 v10, 2, v1 // (op 3529) 
v_lshl_add_u32 v15, v10, 3, v13 // (op 3529) 


s_barrier  // (op 3757) 



s_waitcnt vmcnt(10)
ds_write_b8 v15, v44 // (op 2547) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v15, v45 offset:4 // (op 2547) Store local data 

v_and_b32 v10, -4, v1 // (op 3500) 
v_sub_u32 v13, v1, v10 // (op 3500) 

v_lshrrev_b32 v10, 2, v1 // (op 3500) 
v_lshl_add_u32 v15, v10, 3, v13 // (op 3500) 


v_and_b32 v10, -4, v1 // (op 3600) 
v_sub_u32 v13, v1, v10 // (op 3600) 

v_lshrrev_b32 v10, 2, v1 // (op 3600) 
v_lshl_add_u32 v16, v10, 3, v13 // (op 3600) 


s_mov_b32 s1, 0 // (op 751) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 750) 
s_mov_b32 s15, s18 // (op 750) convert

s_and_b32 s6, s15, 4294967294 // (op 750) 

s_cmp_lt_u32 s1, s6 // (op 750) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_750 // (op 750) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_750 if false)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_750:
s_add_u32 s1, s1, 2 // (op 752) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3754) 



ds_read_u8 v10, v3 // (op 164) Load local data 



ds_read_u8 v13, v3 offset:128 // (op 544) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:29], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 165) 




ds_read_u8 v10, v3 offset:256 // (op 660) Load local data 



ds_read_u8 v42, v3 offset:384 // (op 776) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:25], v[30:33], a[4:7], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 545) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:25], v[34:37], a[8:11], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 661) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:25], v[38:41], a[12:15], v12, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 777) 



v_mov_b32 v10, v5 // (op 167) 

buffer_load_dwordx4 v[18:21], v10, s[68:71], 0 offen offset:256 // (op 167) Load value
buffer_load_dwordx4 v[22:25], v10, s[68:71], 0 offen offset:320 // (op 167) Load value


v_mov_b32 v13, v11 // (op 169) 
v_add_u32 v42, 128, v13 // (op 169) 

buffer_load_dwordx4 v[26:29], v42, s[72:75], 0 offen // (op 169) Load value



v_mov_b32 v42, v14 // (op 171) 

buffer_load_ubyte v12, v42, s[76:79], 0 offen offset:8 // (op 171) Load value


s_lshl_b64 s[18:19], s[36:37], 5 // (op 567) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 567) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 567) 

s_add_u32 s15, s18, 128 // (op 567) 

v_add_u32 v43, v13, s15 // (op 567) 


buffer_load_dwordx4 v[30:33], v43, s[72:75], 0 offen // (op 567) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 683) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 683) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 683) 

s_lshl_b32 s15, s18, 1 // (op 683) 
s_add_u32 s15, s15, 128 // (op 683) 

v_add_u32 v43, v13, s15 // (op 683) 


buffer_load_dwordx4 v[34:37], v43, s[72:75], 0 offen // (op 683) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 799) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 799) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 799) 

s_mul_i32 s15, 3, s18 // (op 799) 
s_add_u32 s15, s15, 128 // (op 799) 

v_add_u32 v43, v13, s15 // (op 799) 


buffer_load_dwordx4 v[38:41], v43, s[72:75], 0 offen // (op 799) Load value


v_mov_b64 v[72:73], s[44:45] // (op 161) convert

v_mov_b32 v43, v17 // (op 161) 
v_mov_b32 v74, v43 // (op 161) 
buffer_load_ubyte v44, v74, s[80:83], 0 offen offset:8 // (op 161) Load value
v_add_u32 v74, v74, v72 // (op 161) 
buffer_load_ubyte v45, v74, s[80:83], 0 offen offset:8 // (op 161) Load value


s_barrier  // (op 3773) 



s_waitcnt vmcnt(10)
ds_write_b8 v16, v70 offset:512 // (op 460) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v16, v71 offset:516 // (op 460) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3770) 



ds_read_u8 v72, v4 offset:512 // (op 462) Load local data 



ds_read_u8 v73, v4 offset:640 // (op 548) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[54:57], a[0:3], v0, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 463) 




ds_read_u8 v72, v4 offset:768 // (op 664) Load local data 



ds_read_u8 v74, v4 offset:896 // (op 780) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[58:61], a[4:7], v0, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 546) 

s_barrier  // (op 3888) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:53], v[62:65], a[8:11], v0, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 662) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:53], v[66:69], a[12:15], v0, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 778) 




buffer_load_dwordx4 v[46:49], v10, s[68:71], 0 offen offset:384 // (op 465) Load value
buffer_load_dwordx4 v[50:53], v10, s[68:71], 0 offen offset:448 // (op 465) Load value



v_add_u32 v5, 256, v5 // (op 2866) 
v_add_u32 v10, 192, v13 // (op 467) 

buffer_load_dwordx4 v[54:57], v10, s[72:75], 0 offen // (op 467) Load value




buffer_load_ubyte v0, v42, s[76:79], 0 offen offset:12 // (op 469) Load value



v_add_u32 v14, 8, v14 // (op 3158) 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 562) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 562) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 562) 

s_add_u32 s15, s18, 192 // (op 562) 

v_add_u32 v10, v13, s15 // (op 562) 


buffer_load_dwordx4 v[58:61], v10, s[72:75], 0 offen // (op 562) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 678) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 678) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 678) 

s_lshl_b32 s15, s18, 1 // (op 678) 
s_add_u32 s15, s15, 192 // (op 678) 

v_add_u32 v10, v13, s15 // (op 678) 


buffer_load_dwordx4 v[62:65], v10, s[72:75], 0 offen // (op 678) Load value



s_lshl_b64 s[18:19], s[36:37], 5 // (op 794) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 794) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 794) 

s_mul_i32 s15, 3, s18 // (op 794) 
s_add_u32 s15, s15, 192 // (op 794) 

v_add_u32 v10, v13, s15 // (op 794) 


buffer_load_dwordx4 v[66:69], v10, s[72:75], 0 offen // (op 794) Load value



v_add_u32 v11, 128, v11 // (op 3015) 
v_mov_b64 v[72:73], s[44:45] // (op 459) convert

v_mov_b32 v10, v43 // (op 459) 
buffer_load_ubyte v70, v10, s[80:83], 0 offen offset:12 // (op 459) Load value
v_add_u32 v10, v10, v72 // (op 459) 
buffer_load_ubyte v71, v10, s[80:83], 0 offen offset:12 // (op 459) Load value



s_barrier  // (op 3760) 
v_add_u32 v17, 8, v17 // (op 3286) 



s_waitcnt vmcnt(10)
ds_write_b8 v15, v44 // (op 162) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v15, v45 offset:4 // (op 162) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 750) 
s_mov_b32 s15, s18 // (op 750) convert

s_and_b32 s6, s15, 4294967294 // (op 750) 

s_cmp_lt_u32 s1, s6 // (op 750) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_750 // (op 750) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_750 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_750:













v_lshrrev_b32 v5, 7, v1 // (op 2802) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2802) 
v_and_b32 v10, 15, v1 // (op 2802) 
s_and_b32 s15, s18, 4294967294 // (op 2802) 
v_lshl_add_u32 v11, v5, 4, v10 // (op 2802) 


s_add_u32 s6, s0, s15 // (op 2802) 

v_and_b32 v5, 48, v1 // (op 2802) 
s_mul_i32 s15, s16, s50 // (op 2802) 
s_add_u32 s15, s15, s6 // (op 2802) 

s_add_u32 s6, s15, s9 // (op 2802) 

s_mul_hi_u32 s15, s6, s13 // (op 2802) 
s_sub_u32 s7, s6, s15 // (op 2802) 
s_lshr_b32 s8, s7, 1 // (op 2802) 

s_add_u32 s7, s8, s15 // (op 2802) 


s_lshr_b32 s8, s7, s11 // (op 2802) 

s_and_b32 s7, s11, 2147483648 // (op 2802) 
s_cmp_eq_i32 s7, 0 // (op 2802) 
s_mov_b32 s15, scc // (op 2802) 

s_cmp_eq_i32 s15, 0 // (op 2802) 
s_cselect_b32 s7, s6, s8 // (op 2802) 


s_mul_hi_u32 s8, s7, s12 // (op 2802) 
s_sub_u32 s15, s7, s8 // (op 2802) 
s_lshr_b32 s28, s15, 1 // (op 2802) 

s_add_u32 s15, s28, s8 // (op 2802) 


s_lshr_b32 s8, s15, s10 // (op 2802) 

s_and_b32 s15, s10, 2147483648 // (op 2802) 
s_cmp_eq_i32 s15, 0 // (op 2802) 
s_mov_b32 s28, scc // (op 2802) 

s_cmp_eq_i32 s28, 0 // (op 2802) 
s_cselect_b32 s15, s7, s8 // (op 2802) 


s_mul_i32 s8, s7, s18 // (op 2802) 


v_lshl_add_u32 v10, s15, 5, v11 // (op 2802) 


s_sub_u32 s7, s6, s8 // (op 2802) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 2802) 


v_mul_lo_u32 v5, v10, s30 // (op 2802) 
v_add_u32 v5, v5, v11 // (op 2802) 


s_mul_i32 s19, s30, s35 // (op 2804) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 2804) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 2804) least significant
s_mul_hi_u32 s6, s30, s34 // (op 2804) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2804) most significant: sum
s_add_u32 s19, s19, s6 // (op 2804) most significant: sum


s_mov_b32 s68, s32 // (op 2804) reg expression
s_mov_b32 s69, s33 // (op 2804) reg expression
s_mov_b32 s70, s18 // (op 2804) reg expression
s_mov_b32 s71, 131072 // (op 2804) call()

v_lshrrev_b32 v10, 6, v1 // (op 2946) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 2946) 
v_and_b32 v11, 1, v10 // (op 2946) 

s_and_b32 s15, s18, 4294967294 // (op 2946) 
v_and_b32 v10, 15, v1 // (op 2946) 
s_add_u32 s6, s0, s15 // (op 2946) 

v_lshl_add_u32 v12, v11, 4, v10 // (op 2946) 


v_and_b32 v10, 63, v1 // (op 2946) 
s_mul_i32 s15, s16, s50 // (op 2946) 
s_add_u32 s15, s15, s6 // (op 2946) 

v_lshrrev_b32 v11, 4, v10 // (op 2946) 

s_add_u32 s6, s15, s9 // (op 2946) 

v_lshlrev_b32 v10, 5, v11 // (op 2946) 

s_mul_hi_u32 s15, s6, s13 // (op 2946) 
s_sub_u32 s7, s6, s15 // (op 2946) 
s_lshr_b32 s8, s7, 1 // (op 2946) 

s_add_u32 s7, s8, s15 // (op 2946) 


s_lshr_b32 s8, s7, s11 // (op 2946) 

s_and_b32 s7, s11, 2147483648 // (op 2946) 
s_cmp_eq_i32 s7, 0 // (op 2946) 
s_mov_b32 s15, scc // (op 2946) 

s_cmp_eq_i32 s15, 0 // (op 2946) 
s_cselect_b32 s7, s6, s8 // (op 2946) 


s_mul_hi_u32 s8, s7, s12 // (op 2946) 
s_sub_u32 s15, s7, s8 // (op 2946) 
s_lshr_b32 s28, s15, 1 // (op 2946) 

s_add_u32 s15, s28, s8 // (op 2946) 


s_lshr_b32 s8, s15, s10 // (op 2946) 

s_and_b32 s15, s10, 2147483648 // (op 2946) 
s_cmp_eq_i32 s15, 0 // (op 2946) 
s_mov_b32 s28, scc // (op 2946) 

s_cmp_eq_i32 s28, 0 // (op 2946) 
s_cselect_b32 s15, s7, s8 // (op 2946) 


s_mul_i32 s8, s15, s14 // (op 2946) 

s_sub_u32 s15, s7, s8 // (op 2946) 

s_mul_i32 s8, s7, s18 // (op 2946) 


v_lshl_add_u32 v11, s15, 7, v12 // (op 2946) 


s_sub_u32 s7, s6, s8 // (op 2946) 


v_mul_lo_u32 v13, v11, s37 // (op 2946) most significant: low of low * high
v_mul_hi_u32 v14, v11, s36 // (op 2946) most significant: high of low * low
v_mul_lo_u32 v12, v11, s36 // (op 2946) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 2946) most significant: sum


v_lshl_add_u32 v11, s7, 7, v10 // (op 2946) 


v_mov_b32 v15, 0 // (op 2946) convert
v_mov_b32 v14, v11 // (op 2946) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 2946) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 2946) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 2946) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 2946) 

v_mov_b32 v10, v12 // (op 2946) convert

s_mul_i32 s19, s36, s23 // (op 2948) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 2948) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 2948) least significant
s_mul_hi_u32 s6, s36, s22 // (op 2948) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2948) most significant: sum
s_add_u32 s19, s19, s6 // (op 2948) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 2948) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 2948) 

s_mov_b32 s72, s38 // (op 2948) reg expression
s_mov_b32 s73, s39 // (op 2948) reg expression
s_mov_b32 s74, s18 // (op 2948) reg expression
s_mov_b32 s75, 131072 // (op 2948) call()

v_lshrrev_b32 v11, 7, v1 // (op 3102) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3102) 
v_and_b32 v12, 15, v1 // (op 3102) 
s_and_b32 s15, s18, 4294967294 // (op 3102) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 3102) 


s_add_u32 s6, s0, s15 // (op 3102) 

v_and_b32 v11, 63, v1 // (op 3102) 
v_lshrrev_b32 v12, 4, v11 // (op 3102) 

s_mul_i32 s15, s16, s50 // (op 3102) 
s_add_u32 s15, s15, s6 // (op 3102) 

s_add_u32 s6, s15, s9 // (op 3102) 

s_mul_hi_u32 s15, s6, s13 // (op 3102) 
s_sub_u32 s7, s6, s15 // (op 3102) 
s_lshr_b32 s8, s7, 1 // (op 3102) 

s_add_u32 s7, s8, s15 // (op 3102) 


s_lshr_b32 s8, s7, s11 // (op 3102) 

s_and_b32 s7, s11, 2147483648 // (op 3102) 
s_cmp_eq_i32 s7, 0 // (op 3102) 
s_mov_b32 s15, scc // (op 3102) 

s_cmp_eq_i32 s15, 0 // (op 3102) 
s_cselect_b32 s7, s6, s8 // (op 3102) 


s_mul_hi_u32 s8, s7, s12 // (op 3102) 
s_sub_u32 s15, s7, s8 // (op 3102) 
s_lshr_b32 s28, s15, 1 // (op 3102) 

s_add_u32 s15, s28, s8 // (op 3102) 


s_lshr_b32 s8, s15, s10 // (op 3102) 

s_and_b32 s15, s10, 2147483648 // (op 3102) 
s_cmp_eq_i32 s15, 0 // (op 3102) 
s_mov_b32 s28, scc // (op 3102) 

s_cmp_eq_i32 s28, 0 // (op 3102) 
s_cselect_b32 s15, s7, s8 // (op 3102) 


s_mul_i32 s8, s7, s18 // (op 3102) 


v_lshl_add_u32 v11, s15, 5, v13 // (op 3102) 


s_sub_u32 s7, s6, s8 // (op 3102) 


v_lshl_add_u32 v13, s7, 2, v12 // (op 3102) 


v_mul_lo_u32 v12, v11, s40 // (op 3102) 
v_add_u32 v12, v12, v13 // (op 3102) 


s_mul_i32 s19, s40, s35 // (op 3104) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3104) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3104) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3104) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3104) most significant: sum
s_add_u32 s19, s19, s6 // (op 3104) most significant: sum


s_mov_b32 s76, s42 // (op 3104) reg expression
s_mov_b32 s77, s43 // (op 3104) reg expression
s_mov_b32 s78, s18 // (op 3104) reg expression
s_mov_b32 s79, 131072 // (op 3104) call()

v_lshrrev_b32 v11, 2, v1 // (op 3230) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3230) 
v_lshlrev_b32 v13, 1, v11 // (op 3230) 

s_and_b32 s15, s18, 4294967294 // (op 3230) 
v_and_b32 v11, -4, v1 // (op 3230) 
s_add_u32 s6, s0, s15 // (op 3230) 

v_sub_u32 v14, v1, v11 // (op 3230) 

s_mul_i32 s15, s16, s50 // (op 3230) 
s_add_u32 s15, s15, s6 // (op 3230) 

s_add_u32 s6, s15, s9 // (op 3230) 

s_mul_hi_u32 s15, s6, s13 // (op 3230) 
s_sub_u32 s7, s6, s15 // (op 3230) 
s_lshr_b32 s8, s7, 1 // (op 3230) 

s_add_u32 s7, s8, s15 // (op 3230) 


s_lshr_b32 s8, s7, s11 // (op 3230) 

s_and_b32 s7, s11, 2147483648 // (op 3230) 
s_cmp_eq_i32 s7, 0 // (op 3230) 
s_mov_b32 s15, scc // (op 3230) 

s_cmp_eq_i32 s15, 0 // (op 3230) 
s_cselect_b32 s7, s6, s8 // (op 3230) 


s_mul_hi_u32 s8, s7, s12 // (op 3230) 
s_sub_u32 s15, s7, s8 // (op 3230) 
s_lshr_b32 s28, s15, 1 // (op 3230) 

s_add_u32 s15, s28, s8 // (op 3230) 


s_lshr_b32 s8, s15, s10 // (op 3230) 

s_and_b32 s15, s10, 2147483648 // (op 3230) 
s_cmp_eq_i32 s15, 0 // (op 3230) 
s_mov_b32 s28, scc // (op 3230) 

s_cmp_eq_i32 s28, 0 // (op 3230) 
s_cselect_b32 s15, s7, s8 // (op 3230) 


s_mul_i32 s8, s15, s14 // (op 3230) 

s_sub_u32 s15, s7, s8 // (op 3230) 

v_lshl_add_u32 v11, s15, 7, v13 // (op 3230) 


s_mul_i32 s8, s7, s18 // (op 3230) 


v_mul_lo_u32 v13, v11, s44 // (op 3230) 

s_sub_u32 s7, s6, s8 // (op 3230) 


v_lshl_add_u32 v11, s7, 2, v14 // (op 3230) 


v_add_u32 v14, v11, v13 // (op 3230) 


s_mul_i32 s19, s44, s23 // (op 3232) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 3232) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 3232) least significant
s_mul_hi_u32 s6, s44, s22 // (op 3232) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3232) most significant: sum
s_add_u32 s19, s19, s6 // (op 3232) most significant: sum


s_mov_b32 s80, s46 // (op 3232) reg expression
s_mov_b32 s81, s47 // (op 3232) reg expression
s_mov_b32 s82, s18 // (op 3232) reg expression
s_mov_b32 s83, 131072 // (op 3232) call()

v_and_b32 v11, -4, v1 // (op 3513) 
v_sub_u32 v13, v1, v11 // (op 3513) 

v_lshrrev_b32 v11, 2, v1 // (op 3513) 
v_lshl_add_u32 v15, v11, 3, v13 // (op 3513) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 756) 
s_and_b32 s15, s18, 4294967294 // (op 756) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 755) 
s_mov_b32 s6, s18 // (op 755) convert

s_cmp_lt_u32 s15, s6 // (op 755) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_755 // (op 755) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_755 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_755:

s_add_u32 s15, s15, 1 // (op 757) 
v_mov_b64 v[16:17], s[44:45] // (op 417) convert

v_mov_b32 v0, v14 // (op 417) 
v_mov_b32 v11, v0 // (op 417) 
buffer_load_ubyte v18, v11, s[80:83], 0 offen offset: 0 // (op 417) Load value
v_add_u32 v11, v11, v16 // (op 417) 
buffer_load_ubyte v19, v11, s[80:83], 0 offen offset: 0 // (op 417) Load value



v_add_u32 v14, 4, v14 // (op 3222) 


s_barrier  // (op 3763) 
v_mov_b32 v0, v5 // (op 423) 




buffer_load_dwordx4 v[20:23], v0, s[68:71], 0 offen // (op 423) Load value
s_waitcnt vmcnt(2)
ds_write_b8 v15, v18 // (op 418) Store local data 
buffer_load_dwordx4 v[24:27], v0, s[68:71], 0 offen offset:64 // (op 423) Load value

v_add_u32 v5, 128, v5 // (op 2792) 
s_waitcnt vmcnt(2)
ds_write_b8 v15, v19 offset:4 // (op 418) Store local data 



v_mov_b32 v0, v10 // (op 425) 
v_mov_b32 v11, v0 // (op 425) reg expression

buffer_load_dwordx4 v[16:19], v11, s[72:75], 0 offen // (op 425) Load value



v_mov_b32 v11, v12 // (op 427) 

buffer_load_ubyte v13, v11, s[76:79], 0 offen offset: 0 // (op 427) Load value

v_add_u32 v12, 4, v12 // (op 3094) 


s_lshl_b64 s[18:19], s[36:37], 5 // (op 537) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 537) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 537) 

v_add_u32 v11, v0, s18 // (op 537) 


s_waitcnt lgkmcnt(0)
s_barrier  // (op 3766) 
buffer_load_dwordx4 v[28:31], v11, s[72:75], 0 offen // (op 537) Load value






ds_read_u8 v11, v3 // (op 420) Load local data 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 653) 



ds_read_u8 v32, v3 offset:128 // (op 529) Load local data 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 653) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 653) 

s_lshl_b32 s6, s18, 1 // (op 653) 

v_add_u32 v33, v0, s6 // (op 653) 


buffer_load_dwordx4 v[34:37], v33, s[72:75], 0 offen // (op 653) Load value

s_waitcnt vmcnt(2) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[16:19], a[0:3], v13, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 421) 







ds_read_u8 v11, v3 offset:256 // (op 645) Load local data 
s_lshl_b64 s[18:19], s[36:37], 5 // (op 769) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 769) 

s_lshr_b64 s[18:19], s[28:29], 3 // (op 769) 

s_mul_i32 s6, 3, s18 // (op 769) 

v_add_u32 v33, v0, s6 // (op 769) 


buffer_load_dwordx4 v[16:19], v33, s[72:75], 0 offen // (op 769) Load value

s_waitcnt vmcnt(2) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:27], v[28:31], a[4:7], v13, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 530) 






v_add_u32 v10, 64, v10 // (op 2936) 
ds_read_u8 v0, v3 offset:384 // (op 761) Load local data 
s_waitcnt vmcnt(1) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:27], v[34:37], a[8:11], v13, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 646) 


s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:27], v[16:19], a[12:15], v13, v0 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 762) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 755) 
s_mov_b32 s6, s18 // (op 755) convert

s_cmp_lt_u32 s15, s6 // (op 755) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_755 // (op 755) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_755 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_755:






s_barrier  // (op 3874) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 193) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_193 // (op 193) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_193
v_accvgpr_read v10, a0 // (op 201) 
v_accvgpr_read v11, a1 // (op 201) 
v_accvgpr_read v12, a2 // (op 201) 
v_accvgpr_read v13, a3 // (op 201) 
v_mul_f32 v14, v7, v10 // (op 201) 
v_mul_f32 v15, v7, v11 // (op 201) 
v_mul_f32 v16, v7, v12 // (op 201) 
v_mul_f32 v17, v7, v13 // (op 201) 

v_cvt_f16_f32 v10, v14 // (op 203) 
v_cvt_f16_f32 v11, v15 // (op 203) 
v_cvt_f16_f32 v12, v16 // (op 203) 
v_cvt_f16_f32 v13, v17 // (op 203) 


v_pack_b32_f16 v14, v10, v11 // (op 204) 
v_pack_b32_f16 v15, v12, v13 // (op 204) 

v_mov_b32 v0, v9 // (op 204) reg expression

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 204) Store value



v_accvgpr_read v10, a4 // (op 951) 
v_accvgpr_read v11, a5 // (op 951) 
v_accvgpr_read v12, a6 // (op 951) 
v_accvgpr_read v13, a7 // (op 951) 
v_mul_f32 v14, v7, v10 // (op 951) 
v_mul_f32 v15, v7, v11 // (op 951) 
v_mul_f32 v16, v7, v12 // (op 951) 
v_mul_f32 v17, v7, v13 // (op 951) 

v_cvt_f16_f32 v10, v14 // (op 953) 
v_cvt_f16_f32 v11, v15 // (op 953) 
v_cvt_f16_f32 v12, v16 // (op 953) 
v_cvt_f16_f32 v13, v17 // (op 953) 


v_pack_b32_f16 v14, v10, v11 // (op 954) 
v_pack_b32_f16 v15, v12, v13 // (op 954) 

s_lshl_b32 s15, s24, 6 // (op 954) 
v_add_u32 v0, v9, s15 // (op 954) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 954) Store value



v_accvgpr_read v10, a8 // (op 965) 
v_accvgpr_read v11, a9 // (op 965) 
v_accvgpr_read v12, a10 // (op 965) 
v_accvgpr_read v13, a11 // (op 965) 
v_mul_f32 v14, v7, v10 // (op 965) 
v_mul_f32 v15, v7, v11 // (op 965) 
v_mul_f32 v16, v7, v12 // (op 965) 
v_mul_f32 v17, v7, v13 // (op 965) 

v_cvt_f16_f32 v10, v14 // (op 967) 
v_cvt_f16_f32 v11, v15 // (op 967) 
v_cvt_f16_f32 v12, v16 // (op 967) 
v_cvt_f16_f32 v13, v17 // (op 967) 


v_pack_b32_f16 v14, v10, v11 // (op 968) 
v_pack_b32_f16 v15, v12, v13 // (op 968) 

s_lshl_b32 s15, s24, 7 // (op 968) 
v_add_u32 v0, v9, s15 // (op 968) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 968) Store value



v_accvgpr_read v10, a12 // (op 979) 
v_accvgpr_read v11, a13 // (op 979) 
v_accvgpr_read v12, a14 // (op 979) 
v_accvgpr_read v13, a15 // (op 979) 
v_mul_f32 v14, v7, v10 // (op 979) 
v_mul_f32 v15, v7, v11 // (op 979) 
v_mul_f32 v16, v7, v12 // (op 979) 
v_mul_f32 v17, v7, v13 // (op 979) 

v_cvt_f16_f32 v10, v14 // (op 981) 
v_cvt_f16_f32 v11, v15 // (op 981) 
v_cvt_f16_f32 v12, v16 // (op 981) 
v_cvt_f16_f32 v13, v17 // (op 981) 


v_pack_b32_f16 v14, v10, v11 // (op 982) 
v_pack_b32_f16 v15, v12, v13 // (op 982) 

s_lshl_b32 s15, s24, 6 // (op 982) 
s_mul_i32 s6, 3, s15 // (op 982) 

v_add_u32 v0, v9, s6 // (op 982) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 982) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_193 // (op 193) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_193
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_193:


v_mov_b32 v0, v8 // (op 218) reg expression

buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 218) Load value

v_accvgpr_read v12, a0 // (op 214) 
v_accvgpr_read v13, a1 // (op 214) 
v_accvgpr_read v14, a2 // (op 214) 
v_accvgpr_read v15, a3 // (op 214) 
v_mul_f32 v16, v7, v12 // (op 214) 
v_mul_f32 v17, v7, v13 // (op 214) 
v_mul_f32 v18, v7, v14 // (op 214) 
v_mul_f32 v19, v7, v15 // (op 214) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 219) 
v_lshrrev_b32 v13, 16, v10 // (op 219) 
v_cvt_f32_f16 v13, v13 // (op 219) 
v_mul_f32 v20, v6, v12 // (op 219) 
v_mul_f32 v21, v6, v13 // (op 219) 
v_cvt_f32_f16 v12, v11 // (op 219) 
v_lshrrev_b32 v13, 16, v11 // (op 219) 
v_cvt_f32_f16 v13, v13 // (op 219) 
v_mul_f32 v22, v6, v12 // (op 219) 
v_mul_f32 v23, v6, v13 // (op 219) 


v_add_f32 v10, v20, v16 // (op 215) 
v_add_f32 v11, v21, v17 // (op 215) 
v_add_f32 v12, v22, v18 // (op 215) 
v_add_f32 v13, v23, v19 // (op 215) 


v_cvt_f16_f32 v14, v10 // (op 216) 
v_cvt_f16_f32 v15, v11 // (op 216) 
v_cvt_f16_f32 v16, v12 // (op 216) 
v_cvt_f16_f32 v17, v13 // (op 216) 


v_pack_b32_f16 v10, v14, v15 // (op 217) 
v_pack_b32_f16 v11, v16, v17 // (op 217) 

v_mov_b32 v0, v9 // (op 217) reg expression

buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 217) Store value





s_lshl_b32 s15, s4, 6 // (op 1016) 
v_add_u32 v0, v8, s15 // (op 1016) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1016) Load value

v_accvgpr_read v12, a4 // (op 1012) 
v_accvgpr_read v13, a5 // (op 1012) 
v_accvgpr_read v14, a6 // (op 1012) 
v_accvgpr_read v15, a7 // (op 1012) 
v_mul_f32 v16, v7, v12 // (op 1012) 
v_mul_f32 v17, v7, v13 // (op 1012) 
v_mul_f32 v18, v7, v14 // (op 1012) 
v_mul_f32 v19, v7, v15 // (op 1012) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1017) 
v_lshrrev_b32 v13, 16, v10 // (op 1017) 
v_cvt_f32_f16 v13, v13 // (op 1017) 
v_mul_f32 v20, v6, v12 // (op 1017) 
v_mul_f32 v21, v6, v13 // (op 1017) 
v_cvt_f32_f16 v12, v11 // (op 1017) 
v_lshrrev_b32 v13, 16, v11 // (op 1017) 
v_cvt_f32_f16 v13, v13 // (op 1017) 
v_mul_f32 v22, v6, v12 // (op 1017) 
v_mul_f32 v23, v6, v13 // (op 1017) 


v_add_f32 v10, v20, v16 // (op 1013) 
v_add_f32 v11, v21, v17 // (op 1013) 
v_add_f32 v12, v22, v18 // (op 1013) 
v_add_f32 v13, v23, v19 // (op 1013) 


v_cvt_f16_f32 v14, v10 // (op 1014) 
v_cvt_f16_f32 v15, v11 // (op 1014) 
v_cvt_f16_f32 v16, v12 // (op 1014) 
v_cvt_f16_f32 v17, v13 // (op 1014) 


v_pack_b32_f16 v10, v14, v15 // (op 1015) 
v_pack_b32_f16 v11, v16, v17 // (op 1015) 

s_lshl_b32 s15, s24, 6 // (op 1015) 
v_add_u32 v0, v9, s15 // (op 1015) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1015) Store value





s_lshl_b32 s15, s4, 7 // (op 1030) 
v_add_u32 v0, v8, s15 // (op 1030) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1030) Load value

v_accvgpr_read v12, a8 // (op 1026) 
v_accvgpr_read v13, a9 // (op 1026) 
v_accvgpr_read v14, a10 // (op 1026) 
v_accvgpr_read v15, a11 // (op 1026) 
v_mul_f32 v16, v7, v12 // (op 1026) 
v_mul_f32 v17, v7, v13 // (op 1026) 
v_mul_f32 v18, v7, v14 // (op 1026) 
v_mul_f32 v19, v7, v15 // (op 1026) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1031) 
v_lshrrev_b32 v13, 16, v10 // (op 1031) 
v_cvt_f32_f16 v13, v13 // (op 1031) 
v_mul_f32 v20, v6, v12 // (op 1031) 
v_mul_f32 v21, v6, v13 // (op 1031) 
v_cvt_f32_f16 v12, v11 // (op 1031) 
v_lshrrev_b32 v13, 16, v11 // (op 1031) 
v_cvt_f32_f16 v13, v13 // (op 1031) 
v_mul_f32 v22, v6, v12 // (op 1031) 
v_mul_f32 v23, v6, v13 // (op 1031) 


v_add_f32 v10, v20, v16 // (op 1027) 
v_add_f32 v11, v21, v17 // (op 1027) 
v_add_f32 v12, v22, v18 // (op 1027) 
v_add_f32 v13, v23, v19 // (op 1027) 


v_cvt_f16_f32 v14, v10 // (op 1028) 
v_cvt_f16_f32 v15, v11 // (op 1028) 
v_cvt_f16_f32 v16, v12 // (op 1028) 
v_cvt_f16_f32 v17, v13 // (op 1028) 


v_pack_b32_f16 v10, v14, v15 // (op 1029) 
v_pack_b32_f16 v11, v16, v17 // (op 1029) 

s_lshl_b32 s15, s24, 7 // (op 1029) 
v_add_u32 v0, v9, s15 // (op 1029) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1029) Store value





s_lshl_b32 s15, s4, 6 // (op 1044) 
s_mul_i32 s6, 3, s15 // (op 1044) 

v_add_u32 v0, v8, s6 // (op 1044) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1044) Load value

v_accvgpr_read v12, a12 // (op 1040) 
v_accvgpr_read v13, a13 // (op 1040) 
v_accvgpr_read v14, a14 // (op 1040) 
v_accvgpr_read v15, a15 // (op 1040) 
v_mul_f32 v16, v7, v12 // (op 1040) 
v_mul_f32 v17, v7, v13 // (op 1040) 
v_mul_f32 v18, v7, v14 // (op 1040) 
v_mul_f32 v19, v7, v15 // (op 1040) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 1045) 
v_lshrrev_b32 v13, 16, v10 // (op 1045) 
v_cvt_f32_f16 v13, v13 // (op 1045) 
v_mul_f32 v20, v6, v12 // (op 1045) 
v_mul_f32 v21, v6, v13 // (op 1045) 
v_cvt_f32_f16 v12, v11 // (op 1045) 
v_lshrrev_b32 v13, 16, v11 // (op 1045) 
v_cvt_f32_f16 v13, v13 // (op 1045) 
v_mul_f32 v22, v6, v12 // (op 1045) 
v_mul_f32 v23, v6, v13 // (op 1045) 


v_add_f32 v10, v20, v16 // (op 1041) 
v_add_f32 v11, v21, v17 // (op 1041) 
v_add_f32 v12, v22, v18 // (op 1041) 
v_add_f32 v13, v23, v19 // (op 1041) 


v_cvt_f16_f32 v14, v10 // (op 1042) 
v_cvt_f16_f32 v15, v11 // (op 1042) 
v_cvt_f16_f32 v16, v12 // (op 1042) 
v_cvt_f16_f32 v17, v13 // (op 1042) 


v_pack_b32_f16 v10, v14, v15 // (op 1043) 
v_pack_b32_f16 v11, v16, v17 // (op 1043) 

s_lshl_b32 s15, s24, 6 // (op 1043) 
s_mul_i32 s6, 3, s15 // (op 1043) 

v_add_u32 v0, v9, s6 // (op 1043) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1043) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_193:
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
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_404 if true)
RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_404:







































s_endpgm  // End of RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.LRR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2, .LRR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_end-RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
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
    DataType: B8
    DataTypeA: B8
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
    macrotile: [32, 128, 128]
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
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TN_BF8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2.kd
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

