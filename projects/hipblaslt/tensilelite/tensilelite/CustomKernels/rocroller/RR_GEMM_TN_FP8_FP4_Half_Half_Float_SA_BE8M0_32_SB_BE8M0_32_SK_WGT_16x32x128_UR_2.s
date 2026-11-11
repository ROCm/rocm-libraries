// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2,@function
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 1442) 
v_and_b32 v3, 1, v0 // (op 1442) 

v_and_b32 v0, 15, v1 // (op 1442) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 1442) 


v_and_b32 v0, 63, v1 // (op 1442) 
v_lshrrev_b32 v3, 4, v0 // (op 1442) 

v_lshl_add_u32 v0, v4, 2, v3 // (op 1442) 


v_lshrrev_b32 v3, 6, v1 // (op 1565) 
v_and_b32 v4, 1, v3 // (op 1565) 

v_and_b32 v3, 15, v1 // (op 1565) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 1565) 


v_and_b32 v3, 63, v1 // (op 1565) 
v_lshrrev_b32 v4, 4, v3 // (op 1565) 

v_lshl_add_u32 v3, v5, 2, v4 // (op 1565) 


v_lshrrev_b32 v4, 6, v1 // (op 1623) 
v_and_b32 v5, 1, v4 // (op 1623) 

v_and_b32 v4, 15, v1 // (op 1623) 
v_lshl_add_u32 v6, v5, 4, v4 // (op 1623) 


v_and_b32 v4, 63, v1 // (op 1623) 
v_lshrrev_b32 v5, 4, v4 // (op 1623) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 1623) 


v_lshrrev_b32 v5, 6, v1 // (op 1647) 
v_and_b32 v6, 1, v5 // (op 1647) 

v_and_b32 v5, 15, v1 // (op 1647) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 1647) 


v_and_b32 v5, 63, v1 // (op 1647) 
v_lshrrev_b32 v6, 4, v5 // (op 1647) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 1647) 


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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 // (op 249) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_249 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_SKStreamTileLoop_249:
v_lshrrev_b32 v8, 7, v1 // (op 1367) 
s_mul_i32 s52, s16, s8 // (op 1367) 
s_add_u32 s52, s52, s0 // (op 1367) 
v_and_b32 v9, 63, v1 // (op 1367) 
s_mul_hi_u32 s53, s52, s13 // (op 1367) 
v_lshrrev_b32 v10, 4, v9 // (op 1367) 
s_sub_u32 s54, s52, s53 // (op 1367) 
v_lshlrev_b32 v11, 2, v10 // (op 1367) 

v_lshl_add_u32 v10, v8, 4, v11 // (op 1367) 


s_lshr_b32 s55, s54, 1 // (op 1367) 

s_add_u32 s54, s55, s53 // (op 1367) 


v_lshrrev_b32 v8, 2, v9 // (op 1367) 

s_lshr_b32 s53, s54, s11 // (op 1367) 

v_and_b32 v9, 3, v8 // (op 1367) 

s_and_b32 s54, s11, 2147483648 // (op 1367) 
v_and_b32 v8, 3, v1 // (op 1367) 
s_cmp_eq_i32 s54, 0 // (op 1367) 
s_mov_b32 s55, scc // (op 1367) 

v_lshl_add_u32 v11, v9, 2, v8 // (op 1367) 


s_cmp_eq_i32 s55, 0 // (op 1367) 
s_cselect_b32 s54, s52, s53 // (op 1367) 



v_lshrrev_b32 v8, 6, v1 // (op 1367) 
s_mul_hi_u32 s52, s54, s12 // (op 1367) 
v_and_b32 v9, 1, v8 // (op 1367) 

s_sub_u32 s53, s54, s52 // (op 1367) 
v_lshl_add_u32 v8, v9, 4, v11 // (op 1367) 


s_lshr_b32 s55, s53, 1 // (op 1367) 

s_add_u32 s53, s55, s52 // (op 1367) 


s_lshr_b32 s52, s53, s10 // (op 1367) 

s_and_b32 s53, s10, 2147483648 // (op 1367) 
s_cmp_eq_i32 s53, 0 // (op 1367) 
s_mov_b32 s55, scc // (op 1367) 

s_cmp_eq_i32 s55, 0 // (op 1367) 
s_cselect_b32 s53, s54, s52 // (op 1367) 


s_mul_i32 s52, s53, s14 // (op 1367) 
v_lshl_add_u32 v9, s53, 4, v10 // (op 1367) 


s_sub_u32 s53, s54, s52 // (op 1367) 


v_lshl_add_u32 v10, s53, 5, v8 // (op 1367) 


v_mul_lo_u32 v8, v10, s4 // (op 1367) 

v_add_lshl_u32 v10, v9, v8, 1 // (op 1367) 


s_mul_i32 s53, s4, s23 // (op 1369) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 1369) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 1369) least significant
s_mul_hi_u32 s55, s4, s22 // (op 1369) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1369) most significant: sum
s_add_u32 s53, s53, s55 // (op 1369) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1369) 

s_mov_b32 s56, s20 // (op 1369) reg expression
s_mov_b32 s57, s21 // (op 1369) reg expression
s_mov_b32 s58, s54 // (op 1369) reg expression
s_mov_b32 s59, 131072 // (op 1369) call()

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
v_lshrrev_b32 v8, 7, v1 // (op 899) 
s_mul_i32 s52, s16, s8 // (op 899) 
s_add_u32 s52, s52, s0 // (op 899) 
v_and_b32 v9, 15, v1 // (op 899) 
s_mul_hi_u32 s53, s52, s13 // (op 899) 
v_lshl_add_u32 v11, v8, 4, v9 // (op 899) 


s_sub_u32 s54, s52, s53 // (op 899) 
v_and_b32 v8, 48, v1 // (op 899) 
s_lshr_b32 s55, s54, 1 // (op 899) 

s_add_u32 s54, s55, s53 // (op 899) 


s_lshr_b32 s53, s54, s11 // (op 899) 

s_and_b32 s54, s11, 2147483648 // (op 899) 
s_cmp_eq_i32 s54, 0 // (op 899) 
s_mov_b32 s55, scc // (op 899) 

s_cmp_eq_i32 s55, 0 // (op 899) 
s_cselect_b32 s54, s52, s53 // (op 899) 


s_mul_hi_u32 s53, s54, s12 // (op 899) 
s_sub_u32 s55, s54, s53 // (op 899) 
s_lshr_b32 s60, s55, 1 // (op 899) 

s_add_u32 s55, s60, s53 // (op 899) 


s_lshr_b32 s53, s55, s10 // (op 899) 

s_and_b32 s55, s10, 2147483648 // (op 899) 
s_cmp_eq_i32 s55, 0 // (op 899) 
s_mov_b32 s60, scc // (op 899) 

s_cmp_eq_i32 s60, 0 // (op 899) 
s_cselect_b32 s55, s54, s53 // (op 899) 


v_lshl_add_u32 v9, s55, 4, v11 // (op 899) 


s_ashr_i64 s[60:61], s[2:3], 7 // (op 899) 
s_mul_i32 s53, s54, s60 // (op 899) 


s_sub_u32 s54, s52, s53 // (op 899) 


v_lshl_add_u32 v11, s54, 7, v8 // (op 899) 


v_mul_lo_u32 v8, v9, s24 // (op 899) 
v_add_u32 v8, v8, v11 // (op 899) 


s_mul_i32 s53, s24, s29 // (op 901) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 901) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 901) least significant
s_mul_hi_u32 s55, s24, s28 // (op 901) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 901) most significant: sum
s_add_u32 s53, s53, s55 // (op 901) most significant: sum


s_mov_b32 s60, s26 // (op 901) reg expression
s_mov_b32 s61, s27 // (op 901) reg expression
s_mov_b32 s62, s52 // (op 901) reg expression
s_mov_b32 s63, 131072 // (op 901) call()

v_lshrrev_b32 v9, 7, v1 // (op 918) 
s_mul_i32 s52, s16, s8 // (op 918) 
s_add_u32 s52, s52, s0 // (op 918) 
v_and_b32 v11, 15, v1 // (op 918) 
s_mul_hi_u32 s53, s52, s13 // (op 918) 
v_lshl_add_u32 v12, v9, 4, v11 // (op 918) 


s_sub_u32 s54, s52, s53 // (op 918) 
v_and_b32 v9, 48, v1 // (op 918) 
s_lshr_b32 s55, s54, 1 // (op 918) 

s_add_u32 s54, s55, s53 // (op 918) 


s_lshr_b32 s53, s54, s11 // (op 918) 

s_and_b32 s54, s11, 2147483648 // (op 918) 
s_cmp_eq_i32 s54, 0 // (op 918) 
s_mov_b32 s55, scc // (op 918) 

s_cmp_eq_i32 s55, 0 // (op 918) 
s_cselect_b32 s54, s52, s53 // (op 918) 


s_mul_hi_u32 s53, s54, s12 // (op 918) 
s_sub_u32 s55, s54, s53 // (op 918) 
s_lshr_b32 s64, s55, 1 // (op 918) 

s_add_u32 s55, s64, s53 // (op 918) 


s_lshr_b32 s53, s55, s10 // (op 918) 

s_and_b32 s55, s10, 2147483648 // (op 918) 
s_cmp_eq_i32 s55, 0 // (op 918) 
s_mov_b32 s64, scc // (op 918) 

s_cmp_eq_i32 s64, 0 // (op 918) 
s_cselect_b32 s55, s54, s53 // (op 918) 


v_lshl_add_u32 v11, s55, 4, v12 // (op 918) 


s_ashr_i64 s[64:65], s[2:3], 7 // (op 918) 
s_mul_i32 s53, s54, s64 // (op 918) 


s_sub_u32 s54, s52, s53 // (op 918) 


v_lshl_add_u32 v12, s54, 7, v9 // (op 918) 


v_mul_lo_u32 v9, v11, s24 // (op 918) 
v_add_u32 v9, v9, v12 // (op 918) 


s_mul_i32 s53, s24, s29 // (op 920) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 920) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 920) least significant
s_mul_hi_u32 s55, s24, s28 // (op 920) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 920) most significant: sum
s_add_u32 s53, s53, s55 // (op 920) most significant: sum


s_mov_b32 s60, s26 // (op 920) reg expression
s_mov_b32 s61, s27 // (op 920) reg expression
s_mov_b32 s62, s52 // (op 920) reg expression
s_mov_b32 s63, 131072 // (op 920) call()

v_lshrrev_b32 v11, 6, v1 // (op 1030) 
s_mul_i32 s52, s16, s8 // (op 1030) 
s_add_u32 s52, s52, s0 // (op 1030) 
v_and_b32 v12, 1, v11 // (op 1030) 

s_mul_hi_u32 s53, s52, s13 // (op 1030) 
v_and_b32 v11, 15, v1 // (op 1030) 
s_sub_u32 s54, s52, s53 // (op 1030) 
v_lshl_add_u32 v13, v12, 4, v11 // (op 1030) 


s_lshr_b32 s55, s54, 1 // (op 1030) 

v_and_b32 v11, 63, v1 // (op 1030) 
s_add_u32 s54, s55, s53 // (op 1030) 


v_lshrrev_b32 v12, 4, v11 // (op 1030) 

s_lshr_b32 s53, s54, s11 // (op 1030) 

v_lshlrev_b32 v11, 5, v12 // (op 1030) 

s_and_b32 s54, s11, 2147483648 // (op 1030) 
s_cmp_eq_i32 s54, 0 // (op 1030) 
s_mov_b32 s55, scc // (op 1030) 

s_cmp_eq_i32 s55, 0 // (op 1030) 
s_cselect_b32 s54, s52, s53 // (op 1030) 


s_mul_hi_u32 s53, s54, s12 // (op 1030) 
s_sub_u32 s55, s54, s53 // (op 1030) 
s_lshr_b32 s64, s55, 1 // (op 1030) 

s_add_u32 s55, s64, s53 // (op 1030) 


s_lshr_b32 s53, s55, s10 // (op 1030) 

s_and_b32 s55, s10, 2147483648 // (op 1030) 
s_cmp_eq_i32 s55, 0 // (op 1030) 
s_mov_b32 s64, scc // (op 1030) 

s_cmp_eq_i32 s64, 0 // (op 1030) 
s_cselect_b32 s55, s54, s53 // (op 1030) 


s_mul_i32 s53, s55, s14 // (op 1030) 

s_sub_u32 s55, s54, s53 // (op 1030) 

v_lshl_add_u32 v12, s55, 5, v13 // (op 1030) 


s_ashr_i64 s[64:65], s[2:3], 7 // (op 1030) 
v_mul_lo_u32 v15, v12, s31 // (op 1030) most significant: low of low * high
v_mul_hi_u32 v13, v12, s30 // (op 1030) most significant: high of low * low
v_mul_lo_u32 v14, v12, s30 // (op 1030) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 1030) most significant: sum


s_mul_i32 s53, s54, s64 // (op 1030) 


s_sub_u32 s54, s52, s53 // (op 1030) 


v_lshl_add_u32 v12, s54, 7, v11 // (op 1030) 


v_mov_b32 v17, 0 // (op 1030) convert
v_mov_b32 v16, v12 // (op 1030) convert

v_add_co_u32 v12, s[52:53], v16, v14 // (op 1030) least significant half
v_addc_co_u32 v13, s[52:53], v17, v15, s[52:53] // (op 1030) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 1030) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1030) 

v_mov_b32 v11, v14 // (op 1030) convert

s_mul_i32 s53, s30, s23 // (op 1032) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1032) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1032) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1032) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1032) most significant: sum
s_add_u32 s53, s53, s55 // (op 1032) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1032) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1032) 

s_mov_b32 s64, s32 // (op 1032) reg expression
s_mov_b32 s65, s33 // (op 1032) reg expression
s_mov_b32 s66, s52 // (op 1032) reg expression
s_mov_b32 s67, 131072 // (op 1032) call()

v_lshrrev_b32 v12, 6, v1 // (op 1049) 
s_mul_i32 s52, s16, s8 // (op 1049) 
s_add_u32 s52, s52, s0 // (op 1049) 
v_and_b32 v13, 1, v12 // (op 1049) 

s_mul_hi_u32 s53, s52, s13 // (op 1049) 
v_and_b32 v12, 15, v1 // (op 1049) 
s_sub_u32 s54, s52, s53 // (op 1049) 
v_lshl_add_u32 v14, v13, 4, v12 // (op 1049) 


s_lshr_b32 s55, s54, 1 // (op 1049) 

v_and_b32 v12, 63, v1 // (op 1049) 
s_add_u32 s54, s55, s53 // (op 1049) 


v_lshrrev_b32 v13, 4, v12 // (op 1049) 

s_lshr_b32 s53, s54, s11 // (op 1049) 

v_lshlrev_b32 v12, 5, v13 // (op 1049) 

s_and_b32 s54, s11, 2147483648 // (op 1049) 
s_cmp_eq_i32 s54, 0 // (op 1049) 
s_mov_b32 s55, scc // (op 1049) 

s_cmp_eq_i32 s55, 0 // (op 1049) 
s_cselect_b32 s54, s52, s53 // (op 1049) 


s_mul_hi_u32 s53, s54, s12 // (op 1049) 
s_sub_u32 s55, s54, s53 // (op 1049) 
s_lshr_b32 s68, s55, 1 // (op 1049) 

s_add_u32 s55, s68, s53 // (op 1049) 


s_lshr_b32 s53, s55, s10 // (op 1049) 

s_and_b32 s55, s10, 2147483648 // (op 1049) 
s_cmp_eq_i32 s55, 0 // (op 1049) 
s_mov_b32 s68, scc // (op 1049) 

s_cmp_eq_i32 s68, 0 // (op 1049) 
s_cselect_b32 s55, s54, s53 // (op 1049) 


s_mul_i32 s53, s55, s14 // (op 1049) 

s_sub_u32 s55, s54, s53 // (op 1049) 

v_lshl_add_u32 v13, s55, 5, v14 // (op 1049) 


v_mul_lo_u32 v15, v13, s31 // (op 1049) most significant: low of low * high
v_mul_hi_u32 v16, v13, s30 // (op 1049) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 1049) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1049) most significant: sum


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1049) 
s_mul_i32 s53, s54, s68 // (op 1049) 


s_sub_u32 s54, s52, s53 // (op 1049) 


v_lshl_add_u32 v13, s54, 7, v12 // (op 1049) 


v_mov_b32 v17, 0 // (op 1049) convert
v_mov_b32 v16, v13 // (op 1049) convert

v_add_co_u32 v12, s[52:53], v16, v14 // (op 1049) least significant half
v_addc_co_u32 v13, s[52:53], v17, v15, s[52:53] // (op 1049) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 1049) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1049) 

v_mov_b32 v12, v14 // (op 1049) convert

s_mul_i32 s53, s30, s23 // (op 1051) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1051) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1051) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1051) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1051) most significant: sum
s_add_u32 s53, s53, s55 // (op 1051) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1051) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1051) 

s_mov_b32 s64, s32 // (op 1051) reg expression
s_mov_b32 s65, s33 // (op 1051) reg expression
s_mov_b32 s66, s52 // (op 1051) reg expression
s_mov_b32 s67, 131072 // (op 1051) call()

v_lshrrev_b32 v13, 7, v1 // (op 1158) 
s_mul_i32 s52, s16, s8 // (op 1158) 
s_add_u32 s52, s52, s0 // (op 1158) 
v_and_b32 v14, 15, v1 // (op 1158) 
s_mul_hi_u32 s53, s52, s13 // (op 1158) 
v_lshl_add_u32 v15, v13, 4, v14 // (op 1158) 


s_sub_u32 s54, s52, s53 // (op 1158) 
v_and_b32 v13, 63, v1 // (op 1158) 
s_lshr_b32 s55, s54, 1 // (op 1158) 

v_lshrrev_b32 v14, 4, v13 // (op 1158) 

s_add_u32 s54, s55, s53 // (op 1158) 


s_lshr_b32 s53, s54, s11 // (op 1158) 

s_and_b32 s54, s11, 2147483648 // (op 1158) 
s_cmp_eq_i32 s54, 0 // (op 1158) 
s_mov_b32 s55, scc // (op 1158) 

s_cmp_eq_i32 s55, 0 // (op 1158) 
s_cselect_b32 s54, s52, s53 // (op 1158) 


s_mul_hi_u32 s53, s54, s12 // (op 1158) 
s_sub_u32 s55, s54, s53 // (op 1158) 
s_lshr_b32 s68, s55, 1 // (op 1158) 

s_add_u32 s55, s68, s53 // (op 1158) 


s_lshr_b32 s53, s55, s10 // (op 1158) 

s_and_b32 s55, s10, 2147483648 // (op 1158) 
s_cmp_eq_i32 s55, 0 // (op 1158) 
s_mov_b32 s68, scc // (op 1158) 

s_cmp_eq_i32 s68, 0 // (op 1158) 
s_cselect_b32 s55, s54, s53 // (op 1158) 


v_lshl_add_u32 v13, s55, 4, v15 // (op 1158) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 1158) 
s_mul_i32 s53, s54, s68 // (op 1158) 


s_sub_u32 s54, s52, s53 // (op 1158) 


v_lshl_add_u32 v15, s54, 2, v14 // (op 1158) 


v_mul_lo_u32 v14, v13, s34 // (op 1158) 
v_add_u32 v14, v14, v15 // (op 1158) 


s_mul_i32 s53, s34, s29 // (op 1160) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1160) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1160) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1160) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1160) most significant: sum
s_add_u32 s53, s53, s55 // (op 1160) most significant: sum


s_mov_b32 s68, s36 // (op 1160) reg expression
s_mov_b32 s69, s37 // (op 1160) reg expression
s_mov_b32 s70, s52 // (op 1160) reg expression
s_mov_b32 s71, 131072 // (op 1160) call()

v_lshrrev_b32 v13, 7, v1 // (op 1177) 
s_mul_i32 s52, s16, s8 // (op 1177) 
s_add_u32 s52, s52, s0 // (op 1177) 
v_and_b32 v15, 15, v1 // (op 1177) 
s_mul_hi_u32 s53, s52, s13 // (op 1177) 
v_lshl_add_u32 v16, v13, 4, v15 // (op 1177) 


s_sub_u32 s54, s52, s53 // (op 1177) 
v_and_b32 v13, 63, v1 // (op 1177) 
s_lshr_b32 s55, s54, 1 // (op 1177) 

v_lshrrev_b32 v15, 4, v13 // (op 1177) 

s_add_u32 s54, s55, s53 // (op 1177) 


s_lshr_b32 s53, s54, s11 // (op 1177) 

s_and_b32 s54, s11, 2147483648 // (op 1177) 
s_cmp_eq_i32 s54, 0 // (op 1177) 
s_mov_b32 s55, scc // (op 1177) 

s_cmp_eq_i32 s55, 0 // (op 1177) 
s_cselect_b32 s54, s52, s53 // (op 1177) 


s_mul_hi_u32 s53, s54, s12 // (op 1177) 
s_sub_u32 s55, s54, s53 // (op 1177) 
s_lshr_b32 s72, s55, 1 // (op 1177) 

s_add_u32 s55, s72, s53 // (op 1177) 


s_lshr_b32 s53, s55, s10 // (op 1177) 

s_and_b32 s55, s10, 2147483648 // (op 1177) 
s_cmp_eq_i32 s55, 0 // (op 1177) 
s_mov_b32 s72, scc // (op 1177) 

s_cmp_eq_i32 s72, 0 // (op 1177) 
s_cselect_b32 s55, s54, s53 // (op 1177) 


v_lshl_add_u32 v13, s55, 4, v16 // (op 1177) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 1177) 
s_mul_i32 s53, s54, s72 // (op 1177) 


s_sub_u32 s54, s52, s53 // (op 1177) 


v_lshl_add_u32 v16, s54, 2, v15 // (op 1177) 


v_mul_lo_u32 v15, v13, s34 // (op 1177) 
v_add_u32 v15, v15, v16 // (op 1177) 


s_mul_i32 s53, s34, s29 // (op 1179) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1179) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1179) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1179) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1179) most significant: sum
s_add_u32 s53, s53, s55 // (op 1179) most significant: sum


s_mov_b32 s68, s36 // (op 1179) reg expression
s_mov_b32 s69, s37 // (op 1179) reg expression
s_mov_b32 s70, s52 // (op 1179) reg expression
s_mov_b32 s71, 131072 // (op 1179) call()

v_lshrrev_b32 v13, 2, v1 // (op 1286) 
s_mul_i32 s52, s16, s8 // (op 1286) 
s_add_u32 s52, s52, s0 // (op 1286) 
v_and_b32 v16, -4, v1 // (op 1286) 
s_mul_hi_u32 s53, s52, s13 // (op 1286) 
v_sub_u32 v17, v1, v16 // (op 1286) 

s_sub_u32 s54, s52, s53 // (op 1286) 
s_lshr_b32 s55, s54, 1 // (op 1286) 

s_add_u32 s54, s55, s53 // (op 1286) 


s_lshr_b32 s53, s54, s11 // (op 1286) 

s_and_b32 s54, s11, 2147483648 // (op 1286) 
s_cmp_eq_i32 s54, 0 // (op 1286) 
s_mov_b32 s55, scc // (op 1286) 

s_cmp_eq_i32 s55, 0 // (op 1286) 
s_cselect_b32 s54, s52, s53 // (op 1286) 


s_mul_hi_u32 s53, s54, s12 // (op 1286) 
s_sub_u32 s55, s54, s53 // (op 1286) 
s_lshr_b32 s72, s55, 1 // (op 1286) 

s_add_u32 s55, s72, s53 // (op 1286) 


s_lshr_b32 s53, s55, s10 // (op 1286) 

s_and_b32 s55, s10, 2147483648 // (op 1286) 
s_cmp_eq_i32 s55, 0 // (op 1286) 
s_mov_b32 s72, scc // (op 1286) 

s_cmp_eq_i32 s72, 0 // (op 1286) 
s_cselect_b32 s55, s54, s53 // (op 1286) 


s_mul_i32 s53, s55, s14 // (op 1286) 

s_sub_u32 s55, s54, s53 // (op 1286) 

v_lshl_add_u32 v16, s55, 5, v13 // (op 1286) 


v_mul_lo_u32 v13, v16, s38 // (op 1286) 

s_ashr_i64 s[72:73], s[2:3], 7 // (op 1286) 
s_mul_i32 s53, s54, s72 // (op 1286) 


s_sub_u32 s54, s52, s53 // (op 1286) 


v_lshl_add_u32 v16, s54, 2, v17 // (op 1286) 


v_add_u32 v17, v16, v13 // (op 1286) 


s_mul_i32 s53, s38, s23 // (op 1288) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1288) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1288) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1288) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1288) most significant: sum
s_add_u32 s53, s53, s55 // (op 1288) most significant: sum


s_mov_b32 s72, s40 // (op 1288) reg expression
s_mov_b32 s73, s41 // (op 1288) reg expression
s_mov_b32 s74, s52 // (op 1288) reg expression
s_mov_b32 s75, 131072 // (op 1288) call()

v_lshrrev_b32 v13, 2, v1 // (op 1305) 
s_mul_i32 s52, s16, s8 // (op 1305) 
s_add_u32 s52, s52, s0 // (op 1305) 
v_and_b32 v16, -4, v1 // (op 1305) 
s_mul_hi_u32 s53, s52, s13 // (op 1305) 
v_sub_u32 v18, v1, v16 // (op 1305) 

s_sub_u32 s54, s52, s53 // (op 1305) 
s_lshr_b32 s55, s54, 1 // (op 1305) 

s_add_u32 s54, s55, s53 // (op 1305) 


s_lshr_b32 s53, s54, s11 // (op 1305) 

s_and_b32 s54, s11, 2147483648 // (op 1305) 
s_cmp_eq_i32 s54, 0 // (op 1305) 
s_mov_b32 s55, scc // (op 1305) 

s_cmp_eq_i32 s55, 0 // (op 1305) 
s_cselect_b32 s54, s52, s53 // (op 1305) 


s_mul_hi_u32 s53, s54, s12 // (op 1305) 
s_sub_u32 s55, s54, s53 // (op 1305) 
s_lshr_b32 s76, s55, 1 // (op 1305) 

s_add_u32 s55, s76, s53 // (op 1305) 


s_lshr_b32 s53, s55, s10 // (op 1305) 

s_and_b32 s55, s10, 2147483648 // (op 1305) 
s_cmp_eq_i32 s55, 0 // (op 1305) 
s_mov_b32 s76, scc // (op 1305) 

s_cmp_eq_i32 s76, 0 // (op 1305) 
s_cselect_b32 s55, s54, s53 // (op 1305) 


s_mul_i32 s53, s55, s14 // (op 1305) 

s_sub_u32 s55, s54, s53 // (op 1305) 

v_lshl_add_u32 v16, s55, 5, v13 // (op 1305) 


v_mul_lo_u32 v13, v16, s38 // (op 1305) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1305) 
s_mul_i32 s53, s54, s76 // (op 1305) 


s_sub_u32 s54, s52, s53 // (op 1305) 


v_lshl_add_u32 v16, s54, 2, v18 // (op 1305) 


v_add_u32 v18, v16, v13 // (op 1305) 


s_mul_i32 s53, s38, s23 // (op 1307) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1307) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1307) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1307) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1307) most significant: sum
s_add_u32 s53, s53, s55 // (op 1307) most significant: sum


s_mov_b32 s72, s40 // (op 1307) reg expression
s_mov_b32 s73, s41 // (op 1307) reg expression
s_mov_b32 s74, s52 // (op 1307) reg expression
s_mov_b32 s75, 131072 // (op 1307) call()




buffer_load_dwordx4 v[20:23], v8, s[60:63], 0 offen // (op 693) Load value
buffer_load_dwordx4 v[24:27], v8, s[60:63], 0 offen offset:64 // (op 693) Load value



buffer_load_dwordx4 v[28:31], v11, s[64:67], 0 offen // (op 698) Load value



buffer_load_ubyte v13, v14, s[68:71], 0 offen offset: 0 // (op 703) Load value



buffer_load_ubyte v16, v17, s[72:75], 0 offen offset: 0 // (op 707) Load value



buffer_load_dwordx4 v[32:35], v8, s[60:63], 0 offen offset:128 // (op 711) Load value
buffer_load_dwordx4 v[36:39], v8, s[60:63], 0 offen offset:192 // (op 711) Load value




buffer_load_dwordx4 v[40:43], v11, s[64:67], 0 offen offset:64 // (op 716) Load value




buffer_load_ubyte v8, v14, s[68:71], 0 offen offset:4 // (op 721) Load value




buffer_load_ubyte v11, v17, s[72:75], 0 offen offset:4 // (op 725) Load value

v_and_b32 v14, -4, v1 // (op 1430) 
v_sub_u32 v17, v1, v14 // (op 1430) 

v_and_b32 v14, 4294967292, v1 // (op 1430) 
v_add_u32 v19, v14, v17 // (op 1430) 


s_barrier  // (op 1704) 



s_waitcnt vmcnt(5)
ds_write_b8 v19, v16 // (op 727) Store local data 

v_and_b32 v14, -4, v1 // (op 1401) 
v_sub_u32 v17, v1, v14 // (op 1401) 

v_and_b32 v14, 4294967292, v1 // (op 1401) 
v_add_u32 v19, v14, v17 // (op 1401) 


v_and_b32 v14, -4, v1 // (op 1635) 
v_sub_u32 v17, v1, v14 // (op 1635) 

v_and_b32 v14, 4294967292, v1 // (op 1635) 
v_add_u32 v44, v14, v17 // (op 1635) 


s_mov_b32 s1, 0 // (op 49) call()
s_and_b32 s52, s7, 4294967294 // (op 48) 
s_cmp_lt_u32 s1, s52 // (op 48) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_48 // (op 48) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_48 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_48:
s_add_u32 s1, s1, 2 // (op 50) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1701) 



ds_read_u8 v14, v0 // (op 69) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[28:31], a[0:3], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 72) 



v_mov_b32 v14, v9 // (op 57) 

buffer_load_dwordx4 v[20:23], v14, s[60:63], 0 offen offset:256 // (op 57) Load value
buffer_load_dwordx4 v[24:27], v14, s[60:63], 0 offen offset:320 // (op 57) Load value


v_mov_b32 v17, v12 // (op 61) 

buffer_load_dwordx4 v[28:31], v17, s[64:67], 0 offen offset:128 // (op 61) Load value


v_mov_b32 v45, v15 // (op 65) 

buffer_load_ubyte v13, v45, s[68:71], 0 offen offset:8 // (op 65) Load value


v_mov_b32 v46, v18 // (op 13) 

buffer_load_ubyte v16, v46, s[72:75], 0 offen offset:8 // (op 13) Load value
s_barrier  // (op 1742) 



s_waitcnt vmcnt(5)
ds_write_b8 v44, v11 offset:128 // (op 565) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1739) 



ds_read_u8 v47, v5 offset:128 // (op 567) Load local data 
s_barrier  // (op 1875) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:43], a[0:3], v8, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 568) 




buffer_load_dwordx4 v[32:35], v14, s[60:63], 0 offen offset:384 // (op 570) Load value
buffer_load_dwordx4 v[36:39], v14, s[60:63], 0 offen offset:448 // (op 570) Load value




v_add_u32 v9, 256, v9 // (op 910) 
buffer_load_dwordx4 v[40:43], v17, s[64:67], 0 offen offset:192 // (op 572) Load value




v_add_u32 v12, 128, v12 // (op 1041) 
buffer_load_ubyte v8, v45, s[68:71], 0 offen offset:12 // (op 574) Load value




v_add_u32 v15, 8, v15 // (op 1169) 
buffer_load_ubyte v11, v46, s[72:75], 0 offen offset:12 // (op 564) Load value

s_barrier  // (op 1707) 
v_add_u32 v18, 8, v18 // (op 1297) 



s_waitcnt vmcnt(5)
ds_write_b8 v19, v16 // (op 18) Store local data 
s_and_b32 s52, s7, 4294967294 // (op 48) 
s_cmp_lt_u32 s1, s52 // (op 48) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_48 // (op 48) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_48 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_48:










s_and_b32 s52, s7, 4294967294 // (op 873) 
v_lshrrev_b32 v9, 7, v1 // (op 873) 
s_add_u32 s53, s0, s52 // (op 873) 

v_and_b32 v12, 15, v1 // (op 873) 
v_lshl_add_u32 v13, v9, 4, v12 // (op 873) 


s_mul_i32 s52, s16, s8 // (op 873) 
s_add_u32 s52, s52, s53 // (op 873) 

v_and_b32 v9, 48, v1 // (op 873) 
s_mul_hi_u32 s53, s52, s13 // (op 873) 
s_sub_u32 s54, s52, s53 // (op 873) 
s_lshr_b32 s55, s54, 1 // (op 873) 

s_add_u32 s54, s55, s53 // (op 873) 


s_lshr_b32 s53, s54, s11 // (op 873) 

s_and_b32 s54, s11, 2147483648 // (op 873) 
s_cmp_eq_i32 s54, 0 // (op 873) 
s_mov_b32 s55, scc // (op 873) 

s_cmp_eq_i32 s55, 0 // (op 873) 
s_cselect_b32 s54, s52, s53 // (op 873) 


s_mul_hi_u32 s53, s54, s12 // (op 873) 
s_sub_u32 s55, s54, s53 // (op 873) 
s_lshr_b32 s76, s55, 1 // (op 873) 

s_add_u32 s55, s76, s53 // (op 873) 


s_lshr_b32 s53, s55, s10 // (op 873) 

s_and_b32 s55, s10, 2147483648 // (op 873) 
s_cmp_eq_i32 s55, 0 // (op 873) 
s_mov_b32 s76, scc // (op 873) 

s_cmp_eq_i32 s76, 0 // (op 873) 
s_cselect_b32 s55, s54, s53 // (op 873) 


v_lshl_add_u32 v12, s55, 4, v13 // (op 873) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 873) 
s_mul_i32 s53, s54, s76 // (op 873) 


s_sub_u32 s54, s52, s53 // (op 873) 


v_lshl_add_u32 v13, s54, 7, v9 // (op 873) 


v_mul_lo_u32 v9, v12, s24 // (op 873) 
v_add_u32 v9, v9, v13 // (op 873) 


s_mul_i32 s53, s24, s29 // (op 875) most significant: low of low * high
s_mul_i32 s54, s25, s28 // (op 875) most significant: low of high * low
s_mul_i32 s52, s24, s28 // (op 875) least significant
s_mul_hi_u32 s55, s24, s28 // (op 875) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 875) most significant: sum
s_add_u32 s53, s53, s55 // (op 875) most significant: sum


s_mov_b32 s60, s26 // (op 875) reg expression
s_mov_b32 s61, s27 // (op 875) reg expression
s_mov_b32 s62, s52 // (op 875) reg expression
s_mov_b32 s63, 131072 // (op 875) call()

s_and_b32 s52, s7, 4294967294 // (op 1007) 
v_lshrrev_b32 v12, 6, v1 // (op 1007) 
s_add_u32 s53, s0, s52 // (op 1007) 

v_and_b32 v13, 1, v12 // (op 1007) 

v_and_b32 v12, 15, v1 // (op 1007) 
s_mul_i32 s52, s16, s8 // (op 1007) 
s_add_u32 s52, s52, s53 // (op 1007) 

v_lshl_add_u32 v14, v13, 4, v12 // (op 1007) 


s_mul_hi_u32 s53, s52, s13 // (op 1007) 
v_and_b32 v12, 63, v1 // (op 1007) 
s_sub_u32 s54, s52, s53 // (op 1007) 
v_lshrrev_b32 v13, 4, v12 // (op 1007) 

s_lshr_b32 s55, s54, 1 // (op 1007) 

v_lshlrev_b32 v12, 5, v13 // (op 1007) 

s_add_u32 s54, s55, s53 // (op 1007) 


s_lshr_b32 s53, s54, s11 // (op 1007) 

s_and_b32 s54, s11, 2147483648 // (op 1007) 
s_cmp_eq_i32 s54, 0 // (op 1007) 
s_mov_b32 s55, scc // (op 1007) 

s_cmp_eq_i32 s55, 0 // (op 1007) 
s_cselect_b32 s54, s52, s53 // (op 1007) 


s_mul_hi_u32 s53, s54, s12 // (op 1007) 
s_sub_u32 s55, s54, s53 // (op 1007) 
s_lshr_b32 s76, s55, 1 // (op 1007) 

s_add_u32 s55, s76, s53 // (op 1007) 


s_lshr_b32 s53, s55, s10 // (op 1007) 

s_and_b32 s55, s10, 2147483648 // (op 1007) 
s_cmp_eq_i32 s55, 0 // (op 1007) 
s_mov_b32 s76, scc // (op 1007) 

s_cmp_eq_i32 s76, 0 // (op 1007) 
s_cselect_b32 s55, s54, s53 // (op 1007) 


s_mul_i32 s53, s55, s14 // (op 1007) 

s_sub_u32 s55, s54, s53 // (op 1007) 

v_lshl_add_u32 v13, s55, 5, v14 // (op 1007) 


v_mul_lo_u32 v15, v13, s31 // (op 1007) most significant: low of low * high
v_mul_hi_u32 v16, v13, s30 // (op 1007) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 1007) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1007) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1007) 
s_mul_i32 s53, s54, s76 // (op 1007) 


s_sub_u32 s54, s52, s53 // (op 1007) 


v_lshl_add_u32 v13, s54, 7, v12 // (op 1007) 


v_mov_b32 v17, 0 // (op 1007) convert
v_mov_b32 v16, v13 // (op 1007) convert

v_add_co_u32 v12, s[52:53], v16, v14 // (op 1007) least significant half
v_addc_co_u32 v13, s[52:53], v17, v15, s[52:53] // (op 1007) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 1007) 


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1007) 

v_mov_b32 v12, v14 // (op 1007) convert

s_mul_i32 s53, s30, s23 // (op 1009) most significant: low of low * high
s_mul_i32 s54, s31, s22 // (op 1009) most significant: low of high * low
s_mul_i32 s52, s30, s22 // (op 1009) least significant
s_mul_hi_u32 s55, s30, s22 // (op 1009) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1009) most significant: sum
s_add_u32 s53, s53, s55 // (op 1009) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 1009) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 1009) 

s_mov_b32 s64, s32 // (op 1009) reg expression
s_mov_b32 s65, s33 // (op 1009) reg expression
s_mov_b32 s66, s52 // (op 1009) reg expression
s_mov_b32 s67, 131072 // (op 1009) call()

s_and_b32 s52, s7, 4294967294 // (op 1135) 
v_lshrrev_b32 v13, 7, v1 // (op 1135) 
s_add_u32 s53, s0, s52 // (op 1135) 

v_and_b32 v14, 15, v1 // (op 1135) 
v_lshl_add_u32 v15, v13, 4, v14 // (op 1135) 


s_mul_i32 s52, s16, s8 // (op 1135) 
s_add_u32 s52, s52, s53 // (op 1135) 

v_and_b32 v13, 63, v1 // (op 1135) 
s_mul_hi_u32 s53, s52, s13 // (op 1135) 
v_lshrrev_b32 v14, 4, v13 // (op 1135) 

s_sub_u32 s54, s52, s53 // (op 1135) 
s_lshr_b32 s55, s54, 1 // (op 1135) 

s_add_u32 s54, s55, s53 // (op 1135) 


s_lshr_b32 s53, s54, s11 // (op 1135) 

s_and_b32 s54, s11, 2147483648 // (op 1135) 
s_cmp_eq_i32 s54, 0 // (op 1135) 
s_mov_b32 s55, scc // (op 1135) 

s_cmp_eq_i32 s55, 0 // (op 1135) 
s_cselect_b32 s54, s52, s53 // (op 1135) 


s_mul_hi_u32 s53, s54, s12 // (op 1135) 
s_sub_u32 s55, s54, s53 // (op 1135) 
s_lshr_b32 s76, s55, 1 // (op 1135) 

s_add_u32 s55, s76, s53 // (op 1135) 


s_lshr_b32 s53, s55, s10 // (op 1135) 

s_and_b32 s55, s10, 2147483648 // (op 1135) 
s_cmp_eq_i32 s55, 0 // (op 1135) 
s_mov_b32 s76, scc // (op 1135) 

s_cmp_eq_i32 s76, 0 // (op 1135) 
s_cselect_b32 s55, s54, s53 // (op 1135) 


v_lshl_add_u32 v13, s55, 4, v15 // (op 1135) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 1135) 
s_mul_i32 s53, s54, s76 // (op 1135) 


s_sub_u32 s54, s52, s53 // (op 1135) 


v_lshl_add_u32 v15, s54, 2, v14 // (op 1135) 


v_mul_lo_u32 v14, v13, s34 // (op 1135) 
v_add_u32 v14, v14, v15 // (op 1135) 


s_mul_i32 s53, s34, s29 // (op 1137) most significant: low of low * high
s_mul_i32 s54, s35, s28 // (op 1137) most significant: low of high * low
s_mul_i32 s52, s34, s28 // (op 1137) least significant
s_mul_hi_u32 s55, s34, s28 // (op 1137) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1137) most significant: sum
s_add_u32 s53, s53, s55 // (op 1137) most significant: sum


s_mov_b32 s68, s36 // (op 1137) reg expression
s_mov_b32 s69, s37 // (op 1137) reg expression
s_mov_b32 s70, s52 // (op 1137) reg expression
s_mov_b32 s71, 131072 // (op 1137) call()

s_and_b32 s52, s7, 4294967294 // (op 1263) 
v_lshrrev_b32 v13, 2, v1 // (op 1263) 
s_add_u32 s53, s0, s52 // (op 1263) 

v_and_b32 v15, -4, v1 // (op 1263) 
v_sub_u32 v16, v1, v15 // (op 1263) 

s_mul_i32 s52, s16, s8 // (op 1263) 
s_add_u32 s52, s52, s53 // (op 1263) 

s_mul_hi_u32 s53, s52, s13 // (op 1263) 
s_sub_u32 s54, s52, s53 // (op 1263) 
s_lshr_b32 s55, s54, 1 // (op 1263) 

s_add_u32 s54, s55, s53 // (op 1263) 


s_lshr_b32 s53, s54, s11 // (op 1263) 

s_and_b32 s54, s11, 2147483648 // (op 1263) 
s_cmp_eq_i32 s54, 0 // (op 1263) 
s_mov_b32 s55, scc // (op 1263) 

s_cmp_eq_i32 s55, 0 // (op 1263) 
s_cselect_b32 s54, s52, s53 // (op 1263) 


s_mul_hi_u32 s53, s54, s12 // (op 1263) 
s_sub_u32 s55, s54, s53 // (op 1263) 
s_lshr_b32 s76, s55, 1 // (op 1263) 

s_add_u32 s55, s76, s53 // (op 1263) 


s_lshr_b32 s53, s55, s10 // (op 1263) 

s_and_b32 s55, s10, 2147483648 // (op 1263) 
s_cmp_eq_i32 s55, 0 // (op 1263) 
s_mov_b32 s76, scc // (op 1263) 

s_cmp_eq_i32 s76, 0 // (op 1263) 
s_cselect_b32 s55, s54, s53 // (op 1263) 


s_mul_i32 s53, s55, s14 // (op 1263) 

s_sub_u32 s55, s54, s53 // (op 1263) 

v_lshl_add_u32 v15, s55, 5, v13 // (op 1263) 


v_mul_lo_u32 v13, v15, s38 // (op 1263) 

s_ashr_i64 s[76:77], s[2:3], 7 // (op 1263) 
s_mul_i32 s53, s54, s76 // (op 1263) 


s_sub_u32 s54, s52, s53 // (op 1263) 


v_lshl_add_u32 v15, s54, 2, v16 // (op 1263) 


v_add_u32 v16, v15, v13 // (op 1263) 


s_mul_i32 s53, s38, s23 // (op 1265) most significant: low of low * high
s_mul_i32 s54, s39, s22 // (op 1265) most significant: low of high * low
s_mul_i32 s52, s38, s22 // (op 1265) least significant
s_mul_hi_u32 s55, s38, s22 // (op 1265) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1265) most significant: sum
s_add_u32 s53, s53, s55 // (op 1265) most significant: sum


s_mov_b32 s72, s40 // (op 1265) reg expression
s_mov_b32 s73, s41 // (op 1265) reg expression
s_mov_b32 s74, s52 // (op 1265) reg expression
s_mov_b32 s75, 131072 // (op 1265) call()

v_and_b32 v13, -4, v1 // (op 1414) 
v_sub_u32 v15, v1, v13 // (op 1414) 

v_and_b32 v13, 4294967292, v1 // (op 1414) 
v_add_u32 v17, v13, v15 // (op 1414) 


s_and_b32 s52, s7, 4294967294 // (op 518) 
s_cmp_lt_u32 s52, s7 // (op 517) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_517 // (op 517) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_517 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_517:


s_add_u32 s52, s52, 1 // (op 519) 
v_mov_b32 v8, v16 // (op 522) 

buffer_load_ubyte v11, v8, s[72:75], 0 offen offset: 0 // (op 522) Load value

v_add_u32 v16, 4, v16 // (op 1255) 


s_barrier  // (op 1710) 
v_mov_b32 v8, v9 // (op 528) 




buffer_load_dwordx4 v[18:21], v8, s[60:63], 0 offen // (op 528) Load value
buffer_load_dwordx4 v[22:25], v8, s[60:63], 0 offen offset:64 // (op 528) Load value

v_add_u32 v9, 128, v9 // (op 863) 
s_waitcnt vmcnt(2)
ds_write_b8 v17, v11 // (op 523) Store local data 



v_mov_b32 v8, v12 // (op 530) 

buffer_load_dwordx4 v[26:29], v8, s[64:67], 0 offen // (op 530) Load value

v_add_u32 v12, 64, v12 // (op 999) 


v_mov_b32 v8, v14 // (op 532) 

buffer_load_ubyte v11, v8, s[68:71], 0 offen offset: 0 // (op 532) Load value

v_add_u32 v14, 4, v14 // (op 1127) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1713) 



ds_read_u8 v8, v0 // (op 525) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:29], a[0:3], v11, v8 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 526) 




s_cmp_lt_u32 s52, s7 // (op 517) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_517 // (op 517) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_517 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_517:






s_barrier  // (op 1869) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_lg_u32 s18, 0 // (op 277) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Send_Tile_277 // (op 277) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Send_Tile_277
v_lshrrev_b32 v8, 5, v1 // (op 1579) 
v_lshlrev_b32 v9, 2, v8 // (op 1579) 

v_lshl_add_u32 v8, s16, 4, v9 // (op 1579) 

v_and_b32 v9, -32, v1 // (op 1579) 
v_sub_u32 v11, v1, v9 // (op 1579) 

v_lshl_add_u32 v9, v8, 5, v11 // (op 1579) 


v_lshlrev_b32 v8, 2, v9 // (op 1579) 

s_lshl_b32 s52, s15, 11 // (op 1581) 
s_mov_b32 s76, s42 // (op 1581) reg expression
s_mov_b32 s77, s43 // (op 1581) reg expression
s_mov_b32 s78, s52 // (op 1581) reg expression
s_mov_b32 s79, 131072 // (op 1581) call()




buffer_store_dword a0, v8, s[76:79], 0 offen sc1 // (op 274) Store value
buffer_store_dword a1, v8, s[76:79], 0 offen offset:128 sc1 // (op 274) Store value
buffer_store_dword a2, v8, s[76:79], 0 offen offset:256 sc1 // (op 274) Store value
buffer_store_dword a3, v8, s[76:79], 0 offen offset:384 sc1 // (op 274) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 278) 
v_cmp_eq_i32 vcc, v1, 0 // (op 282) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282 // (op 282) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282
s_mov_b32 s52, 1 // (op 280) call()
s_mov_b32 s55, 0 // (op 281) convert
s_mov_b32 s54, s16 // (op 281) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 281) 
s_add_u32 s80, s44, s54 // (op 281) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 281) End of Int64 add, unlocking SCC
s_store_dword s52, s[80:81], 0 glc // (op 281) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282 // (op 282) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_282:
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_282:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Send_Tile_277 // (op 277) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Send_Tile_277
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Send_Tile_277:
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Send_Tile_277:
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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Receive_Tile_293 // (op 293) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Receive_Tile_293
s_mov_b32 s52, 0 // (op 259) call()
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s80, s54 // (op 258) convert

s_sub_u32 s54, s80, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 // (op 258) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_258 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_ReceiveTileLoop_258:
s_add_u32 s53, s16, 1 // (op 1596) 
v_lshrrev_b32 v8, 5, v1 // (op 1596) 
s_add_u32 s54, s53, s52 // (op 1596) 
s_lshl_b32 s54, s54, 4 // (op 1596) 

v_lshlrev_b32 v9, 2, v8 // (op 1596) 

v_add_lshl_u32 v8, s54, v9, 5 // (op 1596) 


v_and_b32 v9, -32, v1 // (op 1596) 
v_sub_u32 v11, v1, v9 // (op 1596) 

v_add_lshl_u32 v9, v8, v11, 2 // (op 1596) 


s_lshl_b32 s53, s15, 11 // (op 1598) 
s_mov_b32 s76, s42 // (op 1598) reg expression
s_mov_b32 s77, s43 // (op 1598) reg expression
s_mov_b32 s78, s53 // (op 1598) reg expression
s_mov_b32 s79, 131072 // (op 1598) call()

RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_297:
s_add_u32 s53, s16, 1 // (op 295) 
s_add_u32 s54, s53, s52 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 295) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 295) 

s_add_u32 s80, s44, s54 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s81, s45, s55 // (op 295) End of Int64 add, unlocking SCC
s_load_dword s53, s[80:81], 0 glc // (op 295) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 297) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 // (op 297) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_297 if true)
s_barrier  // (op 301) 
v_cmp_eq_i32 vcc, v1, 0 // (op 300) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300 // (op 300) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300
s_mov_b32 s53, 0 // (op 298) call()
s_add_u32 s54, s16, 1 // (op 299) 
s_add_u32 s80, s54, s52 // (op 299) Start of Int64 add, locking SCC
s_addc_u32 s81, 0, 0 // (op 299) End of Int64 add, unlocking SCC
s_lshl_b64 s[80:81], s[80:81], 2 // (op 299) 

s_add_u32 s54, s44, s80 // (op 299) Start of Int64 add, locking SCC
s_addc_u32 s55, s45, s81 // (op 299) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 299) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300 // (op 300) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_300:
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_300:




buffer_load_dword v12, v9, s[76:79], 0 offen // (op 312) Load value
buffer_load_dword v13, v9, s[76:79], 0 offen offset:128 // (op 312) Load value
buffer_load_dword v14, v9, s[76:79], 0 offen offset:256 // (op 312) Load value
buffer_load_dword v15, v9, s[76:79], 0 offen offset:384 // (op 312) Load value
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

s_add_u32 s52, s52, 1 // (op 260) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 258) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 258) 
s_mov_b32 s80, s54 // (op 258) convert

s_sub_u32 s54, s80, s19 // (op 258) 

s_sub_u32 s55, s54, 1 // (op 258) 

s_cmp_lt_u32 s53, s55 // (op 258) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_ReceiveTileLoop_258 // (op 258) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_ReceiveTileLoop_258 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_258:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Receive_Tile_293 // (op 293) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Receive_Tile_293
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Receive_Tile_293:
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Receive_Tile_293:

s_cmp_eq_i32 s18, 0 // (op 336) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336 // (op 336) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 108) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_108 // (op 108) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_108
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

v_lshrrev_b32 v11, 7, v1 // (op 1456) 
s_mul_i32 s52, s16, s8 // (op 1456) 
s_add_u32 s52, s52, s0 // (op 1456) 
v_and_b32 v8, 63, v1 // (op 1456) 
s_mul_hi_u32 s53, s52, s13 // (op 1456) 
v_lshrrev_b32 v9, 4, v8 // (op 1456) 
s_sub_u32 s54, s52, s53 // (op 1456) 
v_lshlrev_b32 v16, 2, v9 // (op 1456) 

s_lshr_b32 s55, s54, 1 // (op 1456) 

v_lshl_add_u32 v9, v11, 4, v16 // (op 1456) 


s_add_u32 s54, s55, s53 // (op 1456) 


v_lshrrev_b32 v11, 2, v8 // (op 1456) 

s_lshr_b32 s53, s54, s11 // (op 1456) 

v_and_b32 v8, 3, v11 // (op 1456) 

s_and_b32 s54, s11, 2147483648 // (op 1456) 
v_and_b32 v11, 3, v1 // (op 1456) 
s_cmp_eq_i32 s54, 0 // (op 1456) 
s_mov_b32 s55, scc // (op 1456) 

v_lshl_add_u32 v16, v8, 2, v11 // (op 1456) 


s_cmp_eq_i32 s55, 0 // (op 1456) 
s_cselect_b32 s54, s52, s53 // (op 1456) 



v_lshrrev_b32 v8, 6, v1 // (op 1456) 
s_mul_hi_u32 s55, s54, s12 // (op 1456) 
v_and_b32 v11, 1, v8 // (op 1456) 

s_sub_u32 s52, s54, s55 // (op 1456) 
v_lshl_add_u32 v8, v11, 4, v16 // (op 1456) 


s_lshr_b32 s53, s52, 1 // (op 1456) 

s_add_u32 s52, s53, s55 // (op 1456) 


s_lshr_b32 s53, s52, s10 // (op 1456) 

s_and_b32 s52, s10, 2147483648 // (op 1456) 
s_cmp_eq_i32 s52, 0 // (op 1456) 
s_mov_b32 s55, scc // (op 1456) 

s_cmp_eq_i32 s55, 0 // (op 1456) 
s_cselect_b32 s52, s54, s53 // (op 1456) 


s_mul_i32 s53, s52, s14 // (op 1456) 
v_lshl_add_u32 v11, s52, 4, v9 // (op 1456) 


s_sub_u32 s52, s54, s53 // (op 1456) 


v_lshl_add_u32 v9, s52, 5, v8 // (op 1456) 


v_mul_lo_u32 v8, v9, s46 // (op 1456) 

v_add_lshl_u32 v9, v11, v8, 1 // (op 1456) 


s_mul_i32 s53, s46, s23 // (op 1458) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1458) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1458) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1458) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1458) most significant: sum
s_add_u32 s53, s53, s55 // (op 1458) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1458) 

s_mov_b32 s76, s48 // (op 1458) reg expression
s_mov_b32 s77, s49 // (op 1458) reg expression
s_mov_b32 s78, s54 // (op 1458) reg expression
s_mov_b32 s79, 131072 // (op 1458) call()


v_pack_b32_f16 v16, v12, v13 // (op 46) 
v_pack_b32_f16 v17, v14, v15 // (op 46) 


buffer_store_dwordx2 v[16:17], v9, s[76:79], 0 offen // (op 46) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_108 // (op 108) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_108
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_108:



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
v_cvt_f32_f16 v14, v8 // (op 122) 
v_lshrrev_b32 v15, 16, v8 // (op 122) 
v_cvt_f32_f16 v15, v15 // (op 122) 
v_mul_f32 v20, v6, v14 // (op 122) 
v_mul_f32 v21, v6, v15 // (op 122) 
v_cvt_f32_f16 v14, v9 // (op 122) 
v_lshrrev_b32 v15, 16, v9 // (op 122) 
v_cvt_f32_f16 v15, v15 // (op 122) 
v_mul_f32 v22, v6, v14 // (op 122) 
v_mul_f32 v23, v6, v15 // (op 122) 


v_add_f32 v12, v20, v16 // (op 118) 
v_add_f32 v13, v21, v17 // (op 118) 
v_add_f32 v14, v22, v18 // (op 118) 
v_add_f32 v15, v23, v19 // (op 118) 


v_cvt_f16_f32 v16, v12 // (op 119) 
v_cvt_f16_f32 v17, v13 // (op 119) 
v_cvt_f16_f32 v18, v14 // (op 119) 
v_cvt_f16_f32 v19, v15 // (op 119) 

v_lshrrev_b32 v8, 7, v1 // (op 1473) 
s_mul_i32 s52, s16, s8 // (op 1473) 
s_add_u32 s52, s52, s0 // (op 1473) 
v_and_b32 v9, 63, v1 // (op 1473) 
s_mul_hi_u32 s53, s52, s13 // (op 1473) 
v_lshrrev_b32 v11, 4, v9 // (op 1473) 
s_sub_u32 s54, s52, s53 // (op 1473) 
v_lshlrev_b32 v12, 2, v11 // (op 1473) 

s_lshr_b32 s55, s54, 1 // (op 1473) 

v_lshl_add_u32 v11, v8, 4, v12 // (op 1473) 


s_add_u32 s54, s55, s53 // (op 1473) 


v_lshrrev_b32 v8, 2, v9 // (op 1473) 

s_lshr_b32 s53, s54, s11 // (op 1473) 

v_and_b32 v9, 3, v8 // (op 1473) 

s_and_b32 s54, s11, 2147483648 // (op 1473) 
v_and_b32 v8, 3, v1 // (op 1473) 
s_cmp_eq_i32 s54, 0 // (op 1473) 
s_mov_b32 s55, scc // (op 1473) 

v_lshl_add_u32 v12, v9, 2, v8 // (op 1473) 


s_cmp_eq_i32 s55, 0 // (op 1473) 
s_cselect_b32 s54, s52, s53 // (op 1473) 



v_lshrrev_b32 v8, 6, v1 // (op 1473) 
s_mul_hi_u32 s55, s54, s12 // (op 1473) 
v_and_b32 v9, 1, v8 // (op 1473) 

s_sub_u32 s52, s54, s55 // (op 1473) 
v_lshl_add_u32 v8, v9, 4, v12 // (op 1473) 


s_lshr_b32 s53, s52, 1 // (op 1473) 

s_add_u32 s52, s53, s55 // (op 1473) 


s_lshr_b32 s53, s52, s10 // (op 1473) 

s_and_b32 s52, s10, 2147483648 // (op 1473) 
s_cmp_eq_i32 s52, 0 // (op 1473) 
s_mov_b32 s55, scc // (op 1473) 

s_cmp_eq_i32 s55, 0 // (op 1473) 
s_cselect_b32 s52, s54, s53 // (op 1473) 


s_mul_i32 s53, s52, s14 // (op 1473) 
v_lshl_add_u32 v9, s52, 4, v11 // (op 1473) 


s_sub_u32 s52, s54, s53 // (op 1473) 


v_lshl_add_u32 v11, s52, 5, v8 // (op 1473) 


v_mul_lo_u32 v8, v11, s46 // (op 1473) 

v_add_lshl_u32 v11, v9, v8, 1 // (op 1473) 


s_mul_i32 s53, s46, s23 // (op 1475) most significant: low of low * high
s_mul_i32 s54, s47, s22 // (op 1475) most significant: low of high * low
s_mul_i32 s52, s46, s22 // (op 1475) least significant
s_mul_hi_u32 s55, s46, s22 // (op 1475) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 1475) most significant: sum
s_add_u32 s53, s53, s55 // (op 1475) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 1475) 

s_mov_b32 s76, s48 // (op 1475) reg expression
s_mov_b32 s77, s49 // (op 1475) reg expression
s_mov_b32 s78, s54 // (op 1475) reg expression
s_mov_b32 s79, 131072 // (op 1475) call()


v_pack_b32_f16 v8, v16, v17 // (op 120) 
v_pack_b32_f16 v9, v18, v19 // (op 120) 


buffer_store_dwordx2 v[8:9], v11, s[76:79], 0 offen // (op 120) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_108:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336 // (op 336) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_336:
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_336:

s_add_u32 s0, s0, s7 // (op 251) 
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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_SKStreamTileLoop_249 // (op 249) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_SKStreamTileLoop_249 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_249:











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
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 // (op 404) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_404 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_DPStreamTileLoop_404:
v_lshrrev_b32 v0, 7, v1 // (op 1384) 
s_mul_i32 s15, s16, s50 // (op 1384) 
s_add_u32 s15, s15, s0 // (op 1384) 
v_and_b32 v5, 63, v1 // (op 1384) 
s_add_u32 s6, s15, s9 // (op 1384) 

v_lshrrev_b32 v8, 4, v5 // (op 1384) 
s_mul_hi_u32 s15, s6, s13 // (op 1384) 
v_lshlrev_b32 v9, 2, v8 // (op 1384) 

s_sub_u32 s7, s6, s15 // (op 1384) 
v_lshl_add_u32 v8, v0, 4, v9 // (op 1384) 


s_lshr_b32 s8, s7, 1 // (op 1384) 

v_lshrrev_b32 v0, 2, v5 // (op 1384) 

s_add_u32 s7, s8, s15 // (op 1384) 


v_and_b32 v5, 3, v0 // (op 1384) 

s_lshr_b32 s8, s7, s11 // (op 1384) 

v_and_b32 v0, 3, v1 // (op 1384) 
s_and_b32 s7, s11, 2147483648 // (op 1384) 
v_lshl_add_u32 v9, v5, 2, v0 // (op 1384) 


s_cmp_eq_i32 s7, 0 // (op 1384) 
s_mov_b32 s15, scc // (op 1384) 

v_lshrrev_b32 v0, 6, v1 // (op 1384) 
s_cmp_eq_i32 s15, 0 // (op 1384) 
s_cselect_b32 s7, s6, s8 // (op 1384) 



v_and_b32 v5, 1, v0 // (op 1384) 

s_mul_hi_u32 s6, s7, s12 // (op 1384) 
v_lshl_add_u32 v0, v5, 4, v9 // (op 1384) 


s_sub_u32 s8, s7, s6 // (op 1384) 
s_lshr_b32 s15, s8, 1 // (op 1384) 

s_add_u32 s8, s15, s6 // (op 1384) 


s_lshr_b32 s6, s8, s10 // (op 1384) 

s_and_b32 s8, s10, 2147483648 // (op 1384) 
s_cmp_eq_i32 s8, 0 // (op 1384) 
s_mov_b32 s15, scc // (op 1384) 

s_cmp_eq_i32 s15, 0 // (op 1384) 
s_cselect_b32 s8, s7, s6 // (op 1384) 


s_mul_i32 s6, s8, s14 // (op 1384) 
v_lshl_add_u32 v5, s8, 4, v8 // (op 1384) 


s_sub_u32 s8, s7, s6 // (op 1384) 


v_lshl_add_u32 v8, s8, 5, v0 // (op 1384) 


v_mul_lo_u32 v0, v8, s4 // (op 1384) 

v_add_lshl_u32 v8, v5, v0, 1 // (op 1384) 


s_mul_i32 s19, s4, s23 // (op 1386) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 1386) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 1386) least significant
s_mul_hi_u32 s6, s4, s22 // (op 1386) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1386) most significant: sum
s_add_u32 s19, s19, s6 // (op 1386) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1386) 

s_mov_b32 s56, s20 // (op 1386) reg expression
s_mov_b32 s57, s21 // (op 1386) reg expression
s_mov_b32 s58, s6 // (op 1386) reg expression
s_mov_b32 s59, 131072 // (op 1386) call()

v_accvgpr_write a0, 0.00000 // (op 157) call()
v_accvgpr_write a1, 0.00000 // (op 157) call()
v_accvgpr_write a2, 0.00000 // (op 157) call()
v_accvgpr_write a3, 0.00000 // (op 157) call()
v_lshrrev_b32 v0, 7, v1 // (op 944) 
s_mul_i32 s15, s16, s50 // (op 944) 
s_add_u32 s15, s15, s0 // (op 944) 
v_and_b32 v5, 15, v1 // (op 944) 
s_add_u32 s6, s15, s9 // (op 944) 

v_lshl_add_u32 v9, v0, 4, v5 // (op 944) 


s_mul_hi_u32 s15, s6, s13 // (op 944) 
v_and_b32 v0, 48, v1 // (op 944) 
s_sub_u32 s7, s6, s15 // (op 944) 
s_lshr_b32 s8, s7, 1 // (op 944) 

s_add_u32 s7, s8, s15 // (op 944) 


s_lshr_b32 s8, s7, s11 // (op 944) 

s_and_b32 s7, s11, 2147483648 // (op 944) 
s_cmp_eq_i32 s7, 0 // (op 944) 
s_mov_b32 s15, scc // (op 944) 

s_cmp_eq_i32 s15, 0 // (op 944) 
s_cselect_b32 s7, s6, s8 // (op 944) 


s_mul_hi_u32 s8, s7, s12 // (op 944) 
s_sub_u32 s15, s7, s8 // (op 944) 
s_lshr_b32 s18, s15, 1 // (op 944) 

s_add_u32 s15, s18, s8 // (op 944) 


s_lshr_b32 s8, s15, s10 // (op 944) 

s_and_b32 s15, s10, 2147483648 // (op 944) 
s_cmp_eq_i32 s15, 0 // (op 944) 
s_mov_b32 s18, scc // (op 944) 

s_cmp_eq_i32 s18, 0 // (op 944) 
s_cselect_b32 s15, s7, s8 // (op 944) 


v_lshl_add_u32 v5, s15, 4, v9 // (op 944) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 944) 
s_mul_i32 s8, s7, s18 // (op 944) 


s_sub_u32 s7, s6, s8 // (op 944) 


v_lshl_add_u32 v9, s7, 7, v0 // (op 944) 


v_mul_lo_u32 v0, v5, s24 // (op 944) 
v_add_u32 v0, v0, v9 // (op 944) 


s_mul_i32 s19, s24, s29 // (op 946) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 946) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 946) least significant
s_mul_hi_u32 s6, s24, s28 // (op 946) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 946) most significant: sum
s_add_u32 s19, s19, s6 // (op 946) most significant: sum


s_mov_b32 s60, s26 // (op 946) reg expression
s_mov_b32 s61, s27 // (op 946) reg expression
s_mov_b32 s62, s18 // (op 946) reg expression
s_mov_b32 s63, 131072 // (op 946) call()

v_lshrrev_b32 v5, 7, v1 // (op 963) 
s_mul_i32 s15, s16, s50 // (op 963) 
s_add_u32 s15, s15, s0 // (op 963) 
v_and_b32 v9, 15, v1 // (op 963) 
s_add_u32 s6, s15, s9 // (op 963) 

v_lshl_add_u32 v10, v5, 4, v9 // (op 963) 


s_mul_hi_u32 s15, s6, s13 // (op 963) 
v_and_b32 v5, 48, v1 // (op 963) 
s_sub_u32 s7, s6, s15 // (op 963) 
s_lshr_b32 s8, s7, 1 // (op 963) 

s_add_u32 s7, s8, s15 // (op 963) 


s_lshr_b32 s8, s7, s11 // (op 963) 

s_and_b32 s7, s11, 2147483648 // (op 963) 
s_cmp_eq_i32 s7, 0 // (op 963) 
s_mov_b32 s15, scc // (op 963) 

s_cmp_eq_i32 s15, 0 // (op 963) 
s_cselect_b32 s7, s6, s8 // (op 963) 


s_mul_hi_u32 s8, s7, s12 // (op 963) 
s_sub_u32 s15, s7, s8 // (op 963) 
s_lshr_b32 s18, s15, 1 // (op 963) 

s_add_u32 s15, s18, s8 // (op 963) 


s_lshr_b32 s8, s15, s10 // (op 963) 

s_and_b32 s15, s10, 2147483648 // (op 963) 
s_cmp_eq_i32 s15, 0 // (op 963) 
s_mov_b32 s18, scc // (op 963) 

s_cmp_eq_i32 s18, 0 // (op 963) 
s_cselect_b32 s15, s7, s8 // (op 963) 


v_lshl_add_u32 v9, s15, 4, v10 // (op 963) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 963) 
s_mul_i32 s8, s7, s18 // (op 963) 


s_sub_u32 s7, s6, s8 // (op 963) 


v_lshl_add_u32 v10, s7, 7, v5 // (op 963) 


v_mul_lo_u32 v5, v9, s24 // (op 963) 
v_add_u32 v5, v5, v10 // (op 963) 


s_mul_i32 s19, s24, s29 // (op 965) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 965) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 965) least significant
s_mul_hi_u32 s6, s24, s28 // (op 965) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 965) most significant: sum
s_add_u32 s19, s19, s6 // (op 965) most significant: sum


s_mov_b32 s60, s26 // (op 965) reg expression
s_mov_b32 s61, s27 // (op 965) reg expression
s_mov_b32 s62, s18 // (op 965) reg expression
s_mov_b32 s63, 131072 // (op 965) call()

v_lshrrev_b32 v9, 6, v1 // (op 1072) 
s_mul_i32 s15, s16, s50 // (op 1072) 
s_add_u32 s15, s15, s0 // (op 1072) 
v_and_b32 v10, 1, v9 // (op 1072) 

s_add_u32 s6, s15, s9 // (op 1072) 

v_and_b32 v9, 15, v1 // (op 1072) 
s_mul_hi_u32 s15, s6, s13 // (op 1072) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 1072) 


s_sub_u32 s7, s6, s15 // (op 1072) 
v_and_b32 v9, 63, v1 // (op 1072) 
s_lshr_b32 s8, s7, 1 // (op 1072) 

v_lshrrev_b32 v10, 4, v9 // (op 1072) 

s_add_u32 s7, s8, s15 // (op 1072) 


v_lshlrev_b32 v9, 5, v10 // (op 1072) 

s_lshr_b32 s8, s7, s11 // (op 1072) 

s_and_b32 s7, s11, 2147483648 // (op 1072) 
s_cmp_eq_i32 s7, 0 // (op 1072) 
s_mov_b32 s15, scc // (op 1072) 

s_cmp_eq_i32 s15, 0 // (op 1072) 
s_cselect_b32 s7, s6, s8 // (op 1072) 


s_mul_hi_u32 s8, s7, s12 // (op 1072) 
s_sub_u32 s15, s7, s8 // (op 1072) 
s_lshr_b32 s18, s15, 1 // (op 1072) 

s_add_u32 s15, s18, s8 // (op 1072) 


s_lshr_b32 s8, s15, s10 // (op 1072) 

s_and_b32 s15, s10, 2147483648 // (op 1072) 
s_cmp_eq_i32 s15, 0 // (op 1072) 
s_mov_b32 s18, scc // (op 1072) 

s_cmp_eq_i32 s18, 0 // (op 1072) 
s_cselect_b32 s15, s7, s8 // (op 1072) 


s_mul_i32 s8, s15, s14 // (op 1072) 

s_sub_u32 s15, s7, s8 // (op 1072) 

v_lshl_add_u32 v10, s15, 5, v11 // (op 1072) 


v_mul_lo_u32 v13, v10, s31 // (op 1072) most significant: low of low * high
v_mul_hi_u32 v11, v10, s30 // (op 1072) most significant: high of low * low
v_mul_lo_u32 v12, v10, s30 // (op 1072) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 1072) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1072) 
s_mul_i32 s8, s7, s18 // (op 1072) 


s_sub_u32 s7, s6, s8 // (op 1072) 


v_lshl_add_u32 v10, s7, 7, v9 // (op 1072) 


v_mov_b32 v15, 0 // (op 1072) convert
v_mov_b32 v14, v10 // (op 1072) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 1072) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 1072) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 1072) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 1072) 

v_mov_b32 v9, v12 // (op 1072) convert

s_mul_i32 s19, s30, s23 // (op 1074) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1074) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1074) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1074) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1074) most significant: sum
s_add_u32 s19, s19, s6 // (op 1074) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1074) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 1074) 

s_mov_b32 s64, s32 // (op 1074) reg expression
s_mov_b32 s65, s33 // (op 1074) reg expression
s_mov_b32 s66, s18 // (op 1074) reg expression
s_mov_b32 s67, 131072 // (op 1074) call()

v_lshrrev_b32 v10, 6, v1 // (op 1091) 
s_mul_i32 s15, s16, s50 // (op 1091) 
s_add_u32 s15, s15, s0 // (op 1091) 
v_and_b32 v11, 1, v10 // (op 1091) 

s_add_u32 s6, s15, s9 // (op 1091) 

v_and_b32 v10, 15, v1 // (op 1091) 
s_mul_hi_u32 s15, s6, s13 // (op 1091) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 1091) 


s_sub_u32 s7, s6, s15 // (op 1091) 
v_and_b32 v10, 63, v1 // (op 1091) 
s_lshr_b32 s8, s7, 1 // (op 1091) 

v_lshrrev_b32 v11, 4, v10 // (op 1091) 

s_add_u32 s7, s8, s15 // (op 1091) 


v_lshlrev_b32 v10, 5, v11 // (op 1091) 

s_lshr_b32 s8, s7, s11 // (op 1091) 

s_and_b32 s7, s11, 2147483648 // (op 1091) 
s_cmp_eq_i32 s7, 0 // (op 1091) 
s_mov_b32 s15, scc // (op 1091) 

s_cmp_eq_i32 s15, 0 // (op 1091) 
s_cselect_b32 s7, s6, s8 // (op 1091) 


s_mul_hi_u32 s8, s7, s12 // (op 1091) 
s_sub_u32 s15, s7, s8 // (op 1091) 
s_lshr_b32 s18, s15, 1 // (op 1091) 

s_add_u32 s15, s18, s8 // (op 1091) 


s_lshr_b32 s8, s15, s10 // (op 1091) 

s_and_b32 s15, s10, 2147483648 // (op 1091) 
s_cmp_eq_i32 s15, 0 // (op 1091) 
s_mov_b32 s18, scc // (op 1091) 

s_cmp_eq_i32 s18, 0 // (op 1091) 
s_cselect_b32 s15, s7, s8 // (op 1091) 


s_mul_i32 s8, s15, s14 // (op 1091) 

s_sub_u32 s15, s7, s8 // (op 1091) 

v_lshl_add_u32 v11, s15, 5, v12 // (op 1091) 


v_mul_lo_u32 v13, v11, s31 // (op 1091) most significant: low of low * high
v_mul_hi_u32 v14, v11, s30 // (op 1091) most significant: high of low * low
v_mul_lo_u32 v12, v11, s30 // (op 1091) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 1091) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1091) 
s_mul_i32 s8, s7, s18 // (op 1091) 


s_sub_u32 s7, s6, s8 // (op 1091) 


v_lshl_add_u32 v11, s7, 7, v10 // (op 1091) 


v_mov_b32 v15, 0 // (op 1091) convert
v_mov_b32 v14, v11 // (op 1091) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 1091) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 1091) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 1091) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 1091) 

v_mov_b32 v10, v12 // (op 1091) convert

s_mul_i32 s19, s30, s23 // (op 1093) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 1093) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 1093) least significant
s_mul_hi_u32 s6, s30, s22 // (op 1093) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1093) most significant: sum
s_add_u32 s19, s19, s6 // (op 1093) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 1093) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 1093) 

s_mov_b32 s64, s32 // (op 1093) reg expression
s_mov_b32 s65, s33 // (op 1093) reg expression
s_mov_b32 s66, s18 // (op 1093) reg expression
s_mov_b32 s67, 131072 // (op 1093) call()

v_lshrrev_b32 v11, 7, v1 // (op 1200) 
s_mul_i32 s15, s16, s50 // (op 1200) 
s_add_u32 s15, s15, s0 // (op 1200) 
v_and_b32 v12, 15, v1 // (op 1200) 
s_add_u32 s6, s15, s9 // (op 1200) 

v_lshl_add_u32 v13, v11, 4, v12 // (op 1200) 


s_mul_hi_u32 s15, s6, s13 // (op 1200) 
v_and_b32 v11, 63, v1 // (op 1200) 
s_sub_u32 s7, s6, s15 // (op 1200) 
v_lshrrev_b32 v12, 4, v11 // (op 1200) 

s_lshr_b32 s8, s7, 1 // (op 1200) 

s_add_u32 s7, s8, s15 // (op 1200) 


s_lshr_b32 s8, s7, s11 // (op 1200) 

s_and_b32 s7, s11, 2147483648 // (op 1200) 
s_cmp_eq_i32 s7, 0 // (op 1200) 
s_mov_b32 s15, scc // (op 1200) 

s_cmp_eq_i32 s15, 0 // (op 1200) 
s_cselect_b32 s7, s6, s8 // (op 1200) 


s_mul_hi_u32 s8, s7, s12 // (op 1200) 
s_sub_u32 s15, s7, s8 // (op 1200) 
s_lshr_b32 s18, s15, 1 // (op 1200) 

s_add_u32 s15, s18, s8 // (op 1200) 


s_lshr_b32 s8, s15, s10 // (op 1200) 

s_and_b32 s15, s10, 2147483648 // (op 1200) 
s_cmp_eq_i32 s15, 0 // (op 1200) 
s_mov_b32 s18, scc // (op 1200) 

s_cmp_eq_i32 s18, 0 // (op 1200) 
s_cselect_b32 s15, s7, s8 // (op 1200) 


v_lshl_add_u32 v11, s15, 4, v13 // (op 1200) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1200) 
s_mul_i32 s8, s7, s18 // (op 1200) 


s_sub_u32 s7, s6, s8 // (op 1200) 


v_lshl_add_u32 v13, s7, 2, v12 // (op 1200) 


v_mul_lo_u32 v12, v11, s34 // (op 1200) 
v_add_u32 v12, v12, v13 // (op 1200) 


s_mul_i32 s19, s34, s29 // (op 1202) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1202) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1202) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1202) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1202) most significant: sum
s_add_u32 s19, s19, s6 // (op 1202) most significant: sum


s_mov_b32 s68, s36 // (op 1202) reg expression
s_mov_b32 s69, s37 // (op 1202) reg expression
s_mov_b32 s70, s18 // (op 1202) reg expression
s_mov_b32 s71, 131072 // (op 1202) call()

v_lshrrev_b32 v11, 7, v1 // (op 1219) 
s_mul_i32 s15, s16, s50 // (op 1219) 
s_add_u32 s15, s15, s0 // (op 1219) 
v_and_b32 v13, 15, v1 // (op 1219) 
s_add_u32 s6, s15, s9 // (op 1219) 

v_lshl_add_u32 v14, v11, 4, v13 // (op 1219) 


s_mul_hi_u32 s15, s6, s13 // (op 1219) 
v_and_b32 v11, 63, v1 // (op 1219) 
s_sub_u32 s7, s6, s15 // (op 1219) 
v_lshrrev_b32 v13, 4, v11 // (op 1219) 

s_lshr_b32 s8, s7, 1 // (op 1219) 

s_add_u32 s7, s8, s15 // (op 1219) 


s_lshr_b32 s8, s7, s11 // (op 1219) 

s_and_b32 s7, s11, 2147483648 // (op 1219) 
s_cmp_eq_i32 s7, 0 // (op 1219) 
s_mov_b32 s15, scc // (op 1219) 

s_cmp_eq_i32 s15, 0 // (op 1219) 
s_cselect_b32 s7, s6, s8 // (op 1219) 


s_mul_hi_u32 s8, s7, s12 // (op 1219) 
s_sub_u32 s15, s7, s8 // (op 1219) 
s_lshr_b32 s18, s15, 1 // (op 1219) 

s_add_u32 s15, s18, s8 // (op 1219) 


s_lshr_b32 s8, s15, s10 // (op 1219) 

s_and_b32 s15, s10, 2147483648 // (op 1219) 
s_cmp_eq_i32 s15, 0 // (op 1219) 
s_mov_b32 s18, scc // (op 1219) 

s_cmp_eq_i32 s18, 0 // (op 1219) 
s_cselect_b32 s15, s7, s8 // (op 1219) 


v_lshl_add_u32 v11, s15, 4, v14 // (op 1219) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 1219) 
s_mul_i32 s8, s7, s18 // (op 1219) 


s_sub_u32 s7, s6, s8 // (op 1219) 


v_lshl_add_u32 v14, s7, 2, v13 // (op 1219) 


v_mul_lo_u32 v13, v11, s34 // (op 1219) 
v_add_u32 v13, v13, v14 // (op 1219) 


s_mul_i32 s19, s34, s29 // (op 1221) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1221) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1221) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1221) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1221) most significant: sum
s_add_u32 s19, s19, s6 // (op 1221) most significant: sum


s_mov_b32 s68, s36 // (op 1221) reg expression
s_mov_b32 s69, s37 // (op 1221) reg expression
s_mov_b32 s70, s18 // (op 1221) reg expression
s_mov_b32 s71, 131072 // (op 1221) call()

v_lshrrev_b32 v11, 2, v1 // (op 1328) 
s_mul_i32 s15, s16, s50 // (op 1328) 
s_add_u32 s15, s15, s0 // (op 1328) 
v_and_b32 v14, -4, v1 // (op 1328) 
s_add_u32 s6, s15, s9 // (op 1328) 

v_sub_u32 v15, v1, v14 // (op 1328) 

s_mul_hi_u32 s15, s6, s13 // (op 1328) 
s_sub_u32 s7, s6, s15 // (op 1328) 
s_lshr_b32 s8, s7, 1 // (op 1328) 

s_add_u32 s7, s8, s15 // (op 1328) 


s_lshr_b32 s8, s7, s11 // (op 1328) 

s_and_b32 s7, s11, 2147483648 // (op 1328) 
s_cmp_eq_i32 s7, 0 // (op 1328) 
s_mov_b32 s15, scc // (op 1328) 

s_cmp_eq_i32 s15, 0 // (op 1328) 
s_cselect_b32 s7, s6, s8 // (op 1328) 


s_mul_hi_u32 s8, s7, s12 // (op 1328) 
s_sub_u32 s15, s7, s8 // (op 1328) 
s_lshr_b32 s18, s15, 1 // (op 1328) 

s_add_u32 s15, s18, s8 // (op 1328) 


s_lshr_b32 s8, s15, s10 // (op 1328) 

s_and_b32 s15, s10, 2147483648 // (op 1328) 
s_cmp_eq_i32 s15, 0 // (op 1328) 
s_mov_b32 s18, scc // (op 1328) 

s_cmp_eq_i32 s18, 0 // (op 1328) 
s_cselect_b32 s15, s7, s8 // (op 1328) 


s_mul_i32 s8, s15, s14 // (op 1328) 

s_sub_u32 s15, s7, s8 // (op 1328) 

v_lshl_add_u32 v14, s15, 5, v11 // (op 1328) 


v_mul_lo_u32 v11, v14, s38 // (op 1328) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1328) 
s_mul_i32 s8, s7, s18 // (op 1328) 


s_sub_u32 s7, s6, s8 // (op 1328) 


v_lshl_add_u32 v14, s7, 2, v15 // (op 1328) 


v_add_u32 v15, v14, v11 // (op 1328) 


s_mul_i32 s19, s38, s23 // (op 1330) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1330) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1330) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1330) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1330) most significant: sum
s_add_u32 s19, s19, s6 // (op 1330) most significant: sum


s_mov_b32 s72, s40 // (op 1330) reg expression
s_mov_b32 s73, s41 // (op 1330) reg expression
s_mov_b32 s74, s18 // (op 1330) reg expression
s_mov_b32 s75, 131072 // (op 1330) call()

v_lshrrev_b32 v11, 2, v1 // (op 1347) 
s_mul_i32 s15, s16, s50 // (op 1347) 
s_add_u32 s15, s15, s0 // (op 1347) 
v_and_b32 v14, -4, v1 // (op 1347) 
s_add_u32 s6, s15, s9 // (op 1347) 

v_sub_u32 v16, v1, v14 // (op 1347) 

s_mul_hi_u32 s15, s6, s13 // (op 1347) 
s_sub_u32 s7, s6, s15 // (op 1347) 
s_lshr_b32 s8, s7, 1 // (op 1347) 

s_add_u32 s7, s8, s15 // (op 1347) 


s_lshr_b32 s8, s7, s11 // (op 1347) 

s_and_b32 s7, s11, 2147483648 // (op 1347) 
s_cmp_eq_i32 s7, 0 // (op 1347) 
s_mov_b32 s15, scc // (op 1347) 

s_cmp_eq_i32 s15, 0 // (op 1347) 
s_cselect_b32 s7, s6, s8 // (op 1347) 


s_mul_hi_u32 s8, s7, s12 // (op 1347) 
s_sub_u32 s15, s7, s8 // (op 1347) 
s_lshr_b32 s18, s15, 1 // (op 1347) 

s_add_u32 s15, s18, s8 // (op 1347) 


s_lshr_b32 s8, s15, s10 // (op 1347) 

s_and_b32 s15, s10, 2147483648 // (op 1347) 
s_cmp_eq_i32 s15, 0 // (op 1347) 
s_mov_b32 s18, scc // (op 1347) 

s_cmp_eq_i32 s18, 0 // (op 1347) 
s_cselect_b32 s15, s7, s8 // (op 1347) 


s_mul_i32 s8, s15, s14 // (op 1347) 

s_sub_u32 s15, s7, s8 // (op 1347) 

v_lshl_add_u32 v14, s15, 5, v11 // (op 1347) 


v_mul_lo_u32 v11, v14, s38 // (op 1347) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 1347) 
s_mul_i32 s8, s7, s18 // (op 1347) 


s_sub_u32 s7, s6, s8 // (op 1347) 


v_lshl_add_u32 v14, s7, 2, v16 // (op 1347) 


v_add_u32 v16, v14, v11 // (op 1347) 


s_mul_i32 s19, s38, s23 // (op 1349) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1349) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1349) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1349) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1349) most significant: sum
s_add_u32 s19, s19, s6 // (op 1349) most significant: sum


s_mov_b32 s72, s40 // (op 1349) reg expression
s_mov_b32 s73, s41 // (op 1349) reg expression
s_mov_b32 s74, s18 // (op 1349) reg expression
s_mov_b32 s75, 131072 // (op 1349) call()




buffer_load_dwordx4 v[18:21], v0, s[60:63], 0 offen // (op 769) Load value
buffer_load_dwordx4 v[22:25], v0, s[60:63], 0 offen offset:64 // (op 769) Load value



buffer_load_dwordx4 v[26:29], v9, s[64:67], 0 offen // (op 774) Load value



buffer_load_ubyte v11, v12, s[68:71], 0 offen offset: 0 // (op 779) Load value



buffer_load_ubyte v14, v15, s[72:75], 0 offen offset: 0 // (op 783) Load value



buffer_load_dwordx4 v[30:33], v0, s[60:63], 0 offen offset:128 // (op 787) Load value
buffer_load_dwordx4 v[34:37], v0, s[60:63], 0 offen offset:192 // (op 787) Load value




buffer_load_dwordx4 v[38:41], v9, s[64:67], 0 offen offset:64 // (op 792) Load value




buffer_load_ubyte v0, v12, s[68:71], 0 offen offset:4 // (op 797) Load value




buffer_load_ubyte v9, v15, s[72:75], 0 offen offset:4 // (op 801) Load value

v_and_b32 v12, -4, v1 // (op 1553) 
v_sub_u32 v15, v1, v12 // (op 1553) 

v_and_b32 v12, 4294967292, v1 // (op 1553) 
v_add_u32 v17, v12, v15 // (op 1553) 


s_barrier  // (op 1720) 



s_waitcnt vmcnt(5)
ds_write_b8 v17, v14 // (op 803) Store local data 

v_and_b32 v12, -4, v1 // (op 1524) 
v_sub_u32 v15, v1, v12 // (op 1524) 

v_and_b32 v12, 4294967292, v1 // (op 1524) 
v_add_u32 v17, v12, v15 // (op 1524) 


v_and_b32 v12, -4, v1 // (op 1611) 
v_sub_u32 v15, v1, v12 // (op 1611) 

v_and_b32 v12, 4294967292, v1 // (op 1611) 
v_add_u32 v42, v12, v15 // (op 1611) 


s_mov_b32 s1, 0 // (op 159) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 158) 
s_mov_b32 s15, s18 // (op 158) convert

s_and_b32 s6, s15, 4294967294 // (op 158) 

s_cmp_lt_u32 s1, s6 // (op 158) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_158 // (op 158) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_158 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_158:
s_add_u32 s1, s1, 2 // (op 160) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1717) 



ds_read_u8 v12, v3 // (op 164) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:29], a[0:3], v11, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 165) 



v_mov_b32 v12, v5 // (op 167) 

buffer_load_dwordx4 v[18:21], v12, s[60:63], 0 offen offset:256 // (op 167) Load value
buffer_load_dwordx4 v[22:25], v12, s[60:63], 0 offen offset:320 // (op 167) Load value


v_mov_b32 v15, v10 // (op 169) 

buffer_load_dwordx4 v[26:29], v15, s[64:67], 0 offen offset:128 // (op 169) Load value


v_mov_b32 v43, v13 // (op 171) 

buffer_load_ubyte v11, v43, s[68:71], 0 offen offset:8 // (op 171) Load value


v_mov_b32 v44, v16 // (op 161) 

buffer_load_ubyte v14, v44, s[72:75], 0 offen offset:8 // (op 161) Load value
s_barrier  // (op 1736) 



s_waitcnt vmcnt(5)
ds_write_b8 v42, v9 offset:128 // (op 460) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1733) 



ds_read_u8 v45, v4 offset:128 // (op 462) Load local data 
s_barrier  // (op 1839) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:41], a[0:3], v0, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 463) 




buffer_load_dwordx4 v[30:33], v12, s[60:63], 0 offen offset:384 // (op 465) Load value
buffer_load_dwordx4 v[34:37], v12, s[60:63], 0 offen offset:448 // (op 465) Load value




v_add_u32 v5, 256, v5 // (op 955) 
buffer_load_dwordx4 v[38:41], v15, s[64:67], 0 offen offset:192 // (op 467) Load value




v_add_u32 v10, 128, v10 // (op 1083) 
buffer_load_ubyte v0, v43, s[68:71], 0 offen offset:12 // (op 469) Load value




v_add_u32 v13, 8, v13 // (op 1211) 
buffer_load_ubyte v9, v44, s[72:75], 0 offen offset:12 // (op 459) Load value

s_barrier  // (op 1723) 
v_add_u32 v16, 8, v16 // (op 1339) 



s_waitcnt vmcnt(5)
ds_write_b8 v17, v14 // (op 162) Store local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 158) 
s_mov_b32 s15, s18 // (op 158) convert

s_and_b32 s6, s15, 4294967294 // (op 158) 

s_cmp_lt_u32 s1, s6 // (op 158) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_158 // (op 158) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoop_158 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_158:










v_lshrrev_b32 v5, 7, v1 // (op 849) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 849) 
v_and_b32 v10, 15, v1 // (op 849) 
s_and_b32 s15, s18, 4294967294 // (op 849) 
v_lshl_add_u32 v11, v5, 4, v10 // (op 849) 


s_add_u32 s6, s0, s15 // (op 849) 

v_and_b32 v5, 48, v1 // (op 849) 
s_mul_i32 s15, s16, s50 // (op 849) 
s_add_u32 s15, s15, s6 // (op 849) 

s_add_u32 s6, s15, s9 // (op 849) 

s_mul_hi_u32 s15, s6, s13 // (op 849) 
s_sub_u32 s7, s6, s15 // (op 849) 
s_lshr_b32 s8, s7, 1 // (op 849) 

s_add_u32 s7, s8, s15 // (op 849) 


s_lshr_b32 s8, s7, s11 // (op 849) 

s_and_b32 s7, s11, 2147483648 // (op 849) 
s_cmp_eq_i32 s7, 0 // (op 849) 
s_mov_b32 s15, scc // (op 849) 

s_cmp_eq_i32 s15, 0 // (op 849) 
s_cselect_b32 s7, s6, s8 // (op 849) 


s_mul_hi_u32 s8, s7, s12 // (op 849) 
s_sub_u32 s15, s7, s8 // (op 849) 
s_lshr_b32 s42, s15, 1 // (op 849) 

s_add_u32 s15, s42, s8 // (op 849) 


s_lshr_b32 s8, s15, s10 // (op 849) 

s_and_b32 s15, s10, 2147483648 // (op 849) 
s_cmp_eq_i32 s15, 0 // (op 849) 
s_mov_b32 s42, scc // (op 849) 

s_cmp_eq_i32 s42, 0 // (op 849) 
s_cselect_b32 s15, s7, s8 // (op 849) 


s_mul_i32 s8, s7, s18 // (op 849) 


v_lshl_add_u32 v10, s15, 4, v11 // (op 849) 


s_sub_u32 s7, s6, s8 // (op 849) 


v_lshl_add_u32 v11, s7, 7, v5 // (op 849) 


v_mul_lo_u32 v5, v10, s24 // (op 849) 
v_add_u32 v5, v5, v11 // (op 849) 


s_mul_i32 s19, s24, s29 // (op 851) most significant: low of low * high
s_mul_i32 s15, s25, s28 // (op 851) most significant: low of high * low
s_mul_i32 s18, s24, s28 // (op 851) least significant
s_mul_hi_u32 s6, s24, s28 // (op 851) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 851) most significant: sum
s_add_u32 s19, s19, s6 // (op 851) most significant: sum


s_mov_b32 s60, s26 // (op 851) reg expression
s_mov_b32 s61, s27 // (op 851) reg expression
s_mov_b32 s62, s18 // (op 851) reg expression
s_mov_b32 s63, 131072 // (op 851) call()

v_lshrrev_b32 v10, 6, v1 // (op 985) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 985) 
v_and_b32 v11, 1, v10 // (op 985) 

s_and_b32 s15, s18, 4294967294 // (op 985) 
v_and_b32 v10, 15, v1 // (op 985) 
s_add_u32 s6, s0, s15 // (op 985) 

v_lshl_add_u32 v12, v11, 4, v10 // (op 985) 


v_and_b32 v10, 63, v1 // (op 985) 
s_mul_i32 s15, s16, s50 // (op 985) 
s_add_u32 s15, s15, s6 // (op 985) 

v_lshrrev_b32 v11, 4, v10 // (op 985) 

s_add_u32 s6, s15, s9 // (op 985) 

v_lshlrev_b32 v10, 5, v11 // (op 985) 

s_mul_hi_u32 s15, s6, s13 // (op 985) 
s_sub_u32 s7, s6, s15 // (op 985) 
s_lshr_b32 s8, s7, 1 // (op 985) 

s_add_u32 s7, s8, s15 // (op 985) 


s_lshr_b32 s8, s7, s11 // (op 985) 

s_and_b32 s7, s11, 2147483648 // (op 985) 
s_cmp_eq_i32 s7, 0 // (op 985) 
s_mov_b32 s15, scc // (op 985) 

s_cmp_eq_i32 s15, 0 // (op 985) 
s_cselect_b32 s7, s6, s8 // (op 985) 


s_mul_hi_u32 s8, s7, s12 // (op 985) 
s_sub_u32 s15, s7, s8 // (op 985) 
s_lshr_b32 s42, s15, 1 // (op 985) 

s_add_u32 s15, s42, s8 // (op 985) 


s_lshr_b32 s8, s15, s10 // (op 985) 

s_and_b32 s15, s10, 2147483648 // (op 985) 
s_cmp_eq_i32 s15, 0 // (op 985) 
s_mov_b32 s42, scc // (op 985) 

s_cmp_eq_i32 s42, 0 // (op 985) 
s_cselect_b32 s15, s7, s8 // (op 985) 


s_mul_i32 s8, s15, s14 // (op 985) 

s_sub_u32 s15, s7, s8 // (op 985) 

s_mul_i32 s8, s7, s18 // (op 985) 


v_lshl_add_u32 v11, s15, 5, v12 // (op 985) 


s_sub_u32 s7, s6, s8 // (op 985) 


v_mul_lo_u32 v13, v11, s31 // (op 985) most significant: low of low * high
v_mul_hi_u32 v14, v11, s30 // (op 985) most significant: high of low * low
v_mul_lo_u32 v12, v11, s30 // (op 985) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 985) most significant: sum


v_lshl_add_u32 v11, s7, 7, v10 // (op 985) 


v_mov_b32 v15, 0 // (op 985) convert
v_mov_b32 v14, v11 // (op 985) convert

v_add_co_u32 v10, s[18:19], v14, v12 // (op 985) least significant half
v_addc_co_u32 v11, s[18:19], v15, v13, s[18:19] // (op 985) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 985) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 985) 

v_mov_b32 v10, v12 // (op 985) convert

s_mul_i32 s19, s30, s23 // (op 987) most significant: low of low * high
s_mul_i32 s15, s31, s22 // (op 987) most significant: low of high * low
s_mul_i32 s18, s30, s22 // (op 987) least significant
s_mul_hi_u32 s6, s30, s22 // (op 987) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 987) most significant: sum
s_add_u32 s19, s19, s6 // (op 987) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 2 // (op 987) 

s_ashr_i64 s[18:19], s[6:7], 3 // (op 987) 

s_mov_b32 s64, s32 // (op 987) reg expression
s_mov_b32 s65, s33 // (op 987) reg expression
s_mov_b32 s66, s18 // (op 987) reg expression
s_mov_b32 s67, 131072 // (op 987) call()

v_lshrrev_b32 v11, 7, v1 // (op 1113) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1113) 
v_and_b32 v12, 15, v1 // (op 1113) 
s_and_b32 s15, s18, 4294967294 // (op 1113) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 1113) 


s_add_u32 s6, s0, s15 // (op 1113) 

v_and_b32 v11, 63, v1 // (op 1113) 
v_lshrrev_b32 v12, 4, v11 // (op 1113) 

s_mul_i32 s15, s16, s50 // (op 1113) 
s_add_u32 s15, s15, s6 // (op 1113) 

s_add_u32 s6, s15, s9 // (op 1113) 

s_mul_hi_u32 s15, s6, s13 // (op 1113) 
s_sub_u32 s7, s6, s15 // (op 1113) 
s_lshr_b32 s8, s7, 1 // (op 1113) 

s_add_u32 s7, s8, s15 // (op 1113) 


s_lshr_b32 s8, s7, s11 // (op 1113) 

s_and_b32 s7, s11, 2147483648 // (op 1113) 
s_cmp_eq_i32 s7, 0 // (op 1113) 
s_mov_b32 s15, scc // (op 1113) 

s_cmp_eq_i32 s15, 0 // (op 1113) 
s_cselect_b32 s7, s6, s8 // (op 1113) 


s_mul_hi_u32 s8, s7, s12 // (op 1113) 
s_sub_u32 s15, s7, s8 // (op 1113) 
s_lshr_b32 s42, s15, 1 // (op 1113) 

s_add_u32 s15, s42, s8 // (op 1113) 


s_lshr_b32 s8, s15, s10 // (op 1113) 

s_and_b32 s15, s10, 2147483648 // (op 1113) 
s_cmp_eq_i32 s15, 0 // (op 1113) 
s_mov_b32 s42, scc // (op 1113) 

s_cmp_eq_i32 s42, 0 // (op 1113) 
s_cselect_b32 s15, s7, s8 // (op 1113) 


s_mul_i32 s8, s7, s18 // (op 1113) 


v_lshl_add_u32 v11, s15, 4, v13 // (op 1113) 


s_sub_u32 s7, s6, s8 // (op 1113) 


v_lshl_add_u32 v13, s7, 2, v12 // (op 1113) 


v_mul_lo_u32 v12, v11, s34 // (op 1113) 
v_add_u32 v12, v12, v13 // (op 1113) 


s_mul_i32 s19, s34, s29 // (op 1115) most significant: low of low * high
s_mul_i32 s15, s35, s28 // (op 1115) most significant: low of high * low
s_mul_i32 s18, s34, s28 // (op 1115) least significant
s_mul_hi_u32 s6, s34, s28 // (op 1115) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1115) most significant: sum
s_add_u32 s19, s19, s6 // (op 1115) most significant: sum


s_mov_b32 s68, s36 // (op 1115) reg expression
s_mov_b32 s69, s37 // (op 1115) reg expression
s_mov_b32 s70, s18 // (op 1115) reg expression
s_mov_b32 s71, 131072 // (op 1115) call()

v_lshrrev_b32 v11, 2, v1 // (op 1241) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1241) 
v_and_b32 v13, -4, v1 // (op 1241) 
s_and_b32 s15, s18, 4294967294 // (op 1241) 
v_sub_u32 v14, v1, v13 // (op 1241) 

s_add_u32 s6, s0, s15 // (op 1241) 

s_mul_i32 s15, s16, s50 // (op 1241) 
s_add_u32 s15, s15, s6 // (op 1241) 

s_add_u32 s6, s15, s9 // (op 1241) 

s_mul_hi_u32 s15, s6, s13 // (op 1241) 
s_sub_u32 s7, s6, s15 // (op 1241) 
s_lshr_b32 s8, s7, 1 // (op 1241) 

s_add_u32 s7, s8, s15 // (op 1241) 


s_lshr_b32 s8, s7, s11 // (op 1241) 

s_and_b32 s7, s11, 2147483648 // (op 1241) 
s_cmp_eq_i32 s7, 0 // (op 1241) 
s_mov_b32 s15, scc // (op 1241) 

s_cmp_eq_i32 s15, 0 // (op 1241) 
s_cselect_b32 s7, s6, s8 // (op 1241) 


s_mul_hi_u32 s8, s7, s12 // (op 1241) 
s_sub_u32 s15, s7, s8 // (op 1241) 
s_lshr_b32 s42, s15, 1 // (op 1241) 

s_add_u32 s15, s42, s8 // (op 1241) 


s_lshr_b32 s8, s15, s10 // (op 1241) 

s_and_b32 s15, s10, 2147483648 // (op 1241) 
s_cmp_eq_i32 s15, 0 // (op 1241) 
s_mov_b32 s42, scc // (op 1241) 

s_cmp_eq_i32 s42, 0 // (op 1241) 
s_cselect_b32 s15, s7, s8 // (op 1241) 


s_mul_i32 s8, s15, s14 // (op 1241) 

s_sub_u32 s15, s7, s8 // (op 1241) 

v_lshl_add_u32 v13, s15, 5, v11 // (op 1241) 


s_mul_i32 s8, s7, s18 // (op 1241) 


v_mul_lo_u32 v11, v13, s38 // (op 1241) 

s_sub_u32 s7, s6, s8 // (op 1241) 


v_lshl_add_u32 v13, s7, 2, v14 // (op 1241) 


v_add_u32 v14, v13, v11 // (op 1241) 


s_mul_i32 s19, s38, s23 // (op 1243) most significant: low of low * high
s_mul_i32 s15, s39, s22 // (op 1243) most significant: low of high * low
s_mul_i32 s18, s38, s22 // (op 1243) least significant
s_mul_hi_u32 s6, s38, s22 // (op 1243) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1243) most significant: sum
s_add_u32 s19, s19, s6 // (op 1243) most significant: sum


s_mov_b32 s72, s40 // (op 1243) reg expression
s_mov_b32 s73, s41 // (op 1243) reg expression
s_mov_b32 s74, s18 // (op 1243) reg expression
s_mov_b32 s75, 131072 // (op 1243) call()

v_and_b32 v11, -4, v1 // (op 1537) 
v_sub_u32 v13, v1, v11 // (op 1537) 

v_and_b32 v11, 4294967292, v1 // (op 1537) 
v_add_u32 v15, v11, v13 // (op 1537) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 413) 
s_and_b32 s15, s18, 4294967294 // (op 413) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 412) 
s_mov_b32 s6, s18 // (op 412) convert

s_cmp_lt_u32 s15, s6 // (op 412) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_412 // (op 412) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_412 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_412:


s_add_u32 s15, s15, 1 // (op 414) 
v_mov_b32 v0, v14 // (op 417) 

buffer_load_ubyte v9, v0, s[72:75], 0 offen offset: 0 // (op 417) Load value

v_add_u32 v14, 4, v14 // (op 1233) 


s_barrier  // (op 1726) 
v_mov_b32 v0, v5 // (op 423) 




buffer_load_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 423) Load value
buffer_load_dwordx4 v[20:23], v0, s[60:63], 0 offen offset:64 // (op 423) Load value

v_add_u32 v5, 128, v5 // (op 839) 
s_waitcnt vmcnt(2)
ds_write_b8 v15, v9 // (op 418) Store local data 



v_mov_b32 v0, v10 // (op 425) 

buffer_load_dwordx4 v[24:27], v0, s[64:67], 0 offen // (op 425) Load value

v_add_u32 v10, 64, v10 // (op 977) 


v_mov_b32 v0, v12 // (op 427) 

buffer_load_ubyte v9, v0, s[68:71], 0 offen offset: 0 // (op 427) Load value

v_add_u32 v12, 4, v12 // (op 1105) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1729) 



ds_read_u8 v0, v3 // (op 420) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:23], v[24:27], a[0:3], v9, v0 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 421) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 412) 
s_mov_b32 s6, s18 // (op 412) convert

s_cmp_lt_u32 s15, s6 // (op 412) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_412 // (op 412) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_412 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_412:






s_barrier  // (op 1833) 
v_cmp_eq_f32 vcc, 0.00000, v6 // (op 193) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_193 // (op 193) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_193
s_nop 0
v_accvgpr_read v10, a0 // (op 201) Wait state hazard: XDL Write Hazard
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

v_lshrrev_b32 v0, 7, v1 // (op 1490) 
s_mul_i32 s15, s16, s50 // (op 1490) 
s_add_u32 s15, s15, s0 // (op 1490) 
v_and_b32 v5, 63, v1 // (op 1490) 
s_add_u32 s6, s15, s9 // (op 1490) 

v_lshrrev_b32 v9, 4, v5 // (op 1490) 
s_mul_hi_u32 s15, s6, s13 // (op 1490) 
v_lshlrev_b32 v14, 2, v9 // (op 1490) 

s_sub_u32 s7, s6, s15 // (op 1490) 
v_lshl_add_u32 v9, v0, 4, v14 // (op 1490) 


s_lshr_b32 s8, s7, 1 // (op 1490) 

v_lshrrev_b32 v0, 2, v5 // (op 1490) 

s_add_u32 s7, s8, s15 // (op 1490) 


v_and_b32 v5, 3, v0 // (op 1490) 

s_lshr_b32 s8, s7, s11 // (op 1490) 

v_and_b32 v0, 3, v1 // (op 1490) 
s_and_b32 s7, s11, 2147483648 // (op 1490) 
v_lshl_add_u32 v14, v5, 2, v0 // (op 1490) 


s_cmp_eq_i32 s7, 0 // (op 1490) 
s_mov_b32 s15, scc // (op 1490) 

v_lshrrev_b32 v0, 6, v1 // (op 1490) 
s_cmp_eq_i32 s15, 0 // (op 1490) 
s_cselect_b32 s7, s6, s8 // (op 1490) 



v_and_b32 v5, 1, v0 // (op 1490) 

s_mul_hi_u32 s6, s7, s12 // (op 1490) 
v_lshl_add_u32 v0, v5, 4, v14 // (op 1490) 


s_sub_u32 s8, s7, s6 // (op 1490) 
s_lshr_b32 s15, s8, 1 // (op 1490) 

s_add_u32 s8, s15, s6 // (op 1490) 


s_lshr_b32 s6, s8, s10 // (op 1490) 

s_and_b32 s8, s10, 2147483648 // (op 1490) 
s_cmp_eq_i32 s8, 0 // (op 1490) 
s_mov_b32 s15, scc // (op 1490) 

s_cmp_eq_i32 s15, 0 // (op 1490) 
s_cselect_b32 s8, s7, s6 // (op 1490) 


s_mul_i32 s6, s8, s14 // (op 1490) 
v_lshl_add_u32 v5, s8, 4, v9 // (op 1490) 


s_sub_u32 s8, s7, s6 // (op 1490) 


v_lshl_add_u32 v9, s8, 5, v0 // (op 1490) 


v_mul_lo_u32 v0, v9, s46 // (op 1490) 

v_add_lshl_u32 v9, v5, v0, 1 // (op 1490) 


s_mul_i32 s19, s46, s23 // (op 1492) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1492) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1492) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1492) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1492) most significant: sum
s_add_u32 s19, s19, s6 // (op 1492) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1492) 

s_mov_b32 s76, s48 // (op 1492) reg expression
s_mov_b32 s77, s49 // (op 1492) reg expression
s_mov_b32 s78, s6 // (op 1492) reg expression
s_mov_b32 s79, 131072 // (op 1492) call()


v_pack_b32_f16 v14, v10, v11 // (op 204) 
v_pack_b32_f16 v15, v12, v13 // (op 204) 


buffer_store_dwordx2 v[14:15], v9, s[76:79], 0 offen // (op 204) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_193 // (op 193) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_193
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_193:



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


v_add_f32 v12, v20, v16 // (op 215) 
v_add_f32 v13, v21, v17 // (op 215) 
v_add_f32 v14, v22, v18 // (op 215) 
v_add_f32 v15, v23, v19 // (op 215) 


v_cvt_f16_f32 v16, v12 // (op 216) 
v_cvt_f16_f32 v17, v13 // (op 216) 
v_cvt_f16_f32 v18, v14 // (op 216) 
v_cvt_f16_f32 v19, v15 // (op 216) 

v_lshrrev_b32 v0, 7, v1 // (op 1507) 
s_mul_i32 s15, s16, s50 // (op 1507) 
s_add_u32 s15, s15, s0 // (op 1507) 
v_and_b32 v5, 63, v1 // (op 1507) 
s_add_u32 s6, s15, s9 // (op 1507) 

v_lshrrev_b32 v9, 4, v5 // (op 1507) 
s_mul_hi_u32 s15, s6, s13 // (op 1507) 
v_lshlrev_b32 v10, 2, v9 // (op 1507) 

s_sub_u32 s7, s6, s15 // (op 1507) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 1507) 


s_lshr_b32 s8, s7, 1 // (op 1507) 

v_lshrrev_b32 v0, 2, v5 // (op 1507) 

s_add_u32 s7, s8, s15 // (op 1507) 


v_and_b32 v5, 3, v0 // (op 1507) 

s_lshr_b32 s8, s7, s11 // (op 1507) 

v_and_b32 v0, 3, v1 // (op 1507) 
s_and_b32 s7, s11, 2147483648 // (op 1507) 
v_lshl_add_u32 v10, v5, 2, v0 // (op 1507) 


s_cmp_eq_i32 s7, 0 // (op 1507) 
s_mov_b32 s15, scc // (op 1507) 

v_lshrrev_b32 v0, 6, v1 // (op 1507) 
s_cmp_eq_i32 s15, 0 // (op 1507) 
s_cselect_b32 s7, s6, s8 // (op 1507) 



v_and_b32 v5, 1, v0 // (op 1507) 

s_mul_hi_u32 s6, s7, s12 // (op 1507) 
v_lshl_add_u32 v0, v5, 4, v10 // (op 1507) 


s_sub_u32 s8, s7, s6 // (op 1507) 
s_lshr_b32 s15, s8, 1 // (op 1507) 

s_add_u32 s8, s15, s6 // (op 1507) 


s_lshr_b32 s6, s8, s10 // (op 1507) 

s_and_b32 s8, s10, 2147483648 // (op 1507) 
s_cmp_eq_i32 s8, 0 // (op 1507) 
s_mov_b32 s15, scc // (op 1507) 

s_cmp_eq_i32 s15, 0 // (op 1507) 
s_cselect_b32 s8, s7, s6 // (op 1507) 


s_mul_i32 s6, s8, s14 // (op 1507) 
v_lshl_add_u32 v5, s8, 4, v9 // (op 1507) 


s_sub_u32 s8, s7, s6 // (op 1507) 


v_lshl_add_u32 v9, s8, 5, v0 // (op 1507) 


v_mul_lo_u32 v0, v9, s46 // (op 1507) 

v_add_lshl_u32 v9, v5, v0, 1 // (op 1507) 


s_mul_i32 s19, s46, s23 // (op 1509) most significant: low of low * high
s_mul_i32 s15, s47, s22 // (op 1509) most significant: low of high * low
s_mul_i32 s18, s46, s22 // (op 1509) least significant
s_mul_hi_u32 s6, s46, s22 // (op 1509) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 1509) most significant: sum
s_add_u32 s19, s19, s6 // (op 1509) most significant: sum


s_lshl_b64 s[6:7], s[18:19], 1 // (op 1509) 

s_mov_b32 s76, s48 // (op 1509) reg expression
s_mov_b32 s77, s49 // (op 1509) reg expression
s_mov_b32 s78, s6 // (op 1509) reg expression
s_mov_b32 s79, 131072 // (op 1509) call()


v_pack_b32_f16 v10, v16, v17 // (op 217) 
v_pack_b32_f16 v11, v18, v19 // (op 217) 


buffer_store_dwordx2 v[10:11], v9, s[76:79], 0 offen // (op 217) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_193:

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
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_DPStreamTileLoop_404 // (op 404) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopTop_DPStreamTileLoop_404 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_404:




































s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
.LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2, .LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2_end-RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 256 // lds bytes
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
    DataType: F8
    DataTypeA: F8
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
    macrotile: [16, 32, 128]
    threads: [128, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 16
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
  - .name: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 82
    .vgpr_count: 48
    .agpr_count: 4
    .max_flat_workgroup_size: 128
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

