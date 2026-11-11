// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2,@function
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 14434) 
v_and_b32 v3, 15, v1 // (op 14434) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 14434) 


v_and_b32 v0, 48, v1 // (op 14434) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 14434) 


v_lshrrev_b32 v0, 6, v1 // (op 14509) 
v_and_b32 v4, 1, v0 // (op 14509) 

v_and_b32 v0, 15, v1 // (op 14509) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 14509) 


v_and_b32 v0, 63, v1 // (op 14509) 
v_lshrrev_b32 v4, 4, v0 // (op 14509) 

v_lshlrev_b32 v0, 5, v4 // (op 14509) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 14509) 


v_lshlrev_b32 v0, 2, v4 // (op 14509) 

v_lshrrev_b32 v4, 3, v0 // (op 14509) 

v_lshrrev_b32 v0, 7, v1 // (op 14636) 
v_and_b32 v5, 15, v1 // (op 14636) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 14636) 


v_and_b32 v0, 48, v1 // (op 14636) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 14636) 


v_lshrrev_b32 v0, 6, v1 // (op 14711) 
v_and_b32 v6, 1, v0 // (op 14711) 

v_and_b32 v0, 15, v1 // (op 14711) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 14711) 


v_and_b32 v0, 63, v1 // (op 14711) 
v_lshrrev_b32 v6, 4, v0 // (op 14711) 

v_lshlrev_b32 v0, 5, v6 // (op 14711) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 14711) 


v_lshlrev_b32 v0, 2, v6 // (op 14711) 

v_lshrrev_b32 v6, 3, v0 // (op 14711) 

v_lshrrev_b32 v0, 7, v1 // (op 14825) 
v_and_b32 v7, 15, v1 // (op 14825) 
v_lshl_add_u32 v8, v0, 6, v7 // (op 14825) 


v_and_b32 v0, 48, v1 // (op 14825) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 14825) 


v_lshrrev_b32 v0, 6, v1 // (op 14869) 
v_and_b32 v8, 1, v0 // (op 14869) 

v_and_b32 v0, 15, v1 // (op 14869) 
v_lshl_add_u32 v9, v8, 6, v0 // (op 14869) 


v_and_b32 v0, 63, v1 // (op 14869) 
v_lshrrev_b32 v8, 4, v0 // (op 14869) 

v_lshlrev_b32 v0, 5, v8 // (op 14869) 

v_lshl_add_u32 v8, v9, 7, v0 // (op 14869) 


v_lshlrev_b32 v0, 2, v8 // (op 14869) 

v_lshrrev_b32 v8, 3, v0 // (op 14869) 

v_lshrrev_b32 v0, 7, v1 // (op 14913) 
v_and_b32 v9, 15, v1 // (op 14913) 
v_lshl_add_u32 v10, v0, 6, v9 // (op 14913) 


v_and_b32 v0, 48, v1 // (op 14913) 
v_lshl_add_u32 v9, v10, 7, v0 // (op 14913) 


v_lshrrev_b32 v0, 6, v1 // (op 14957) 
v_and_b32 v10, 1, v0 // (op 14957) 

v_and_b32 v0, 15, v1 // (op 14957) 
v_lshl_add_u32 v11, v10, 6, v0 // (op 14957) 


v_and_b32 v0, 63, v1 // (op 14957) 
v_lshrrev_b32 v10, 4, v0 // (op 14957) 

v_lshlrev_b32 v0, 5, v10 // (op 14957) 

v_lshl_add_u32 v10, v11, 7, v0 // (op 14957) 


v_lshlrev_b32 v0, 2, v10 // (op 14957) 

v_lshrrev_b32 v10, 3, v0 // (op 14957) 

v_mov_b32 v0, s6 // (op 2) Move value
v_mov_b32 v11, s7 // (op 35) Move value


s_mov_b32 s0, 0 // (op 356) call()
s_mov_b32 s1, 0 // (op 357) call()
s_mov_b32 s6, 0 // (op 358) call()
s_mov_b32 s7, 0 // (op 359) call()
s_mov_b32 s18, 0 // (op 360) call()
s_mov_b32 s19, 0 // (op 361) call()
s_mov_b32 s0, 0 // (op 256) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 255) 
s_cmp_gt_u32 s9, s52 // (op 255) 
s_mov_b32 s53, scc // (op 255) 
s_sub_u32 s54, s9, s52 // (op 255) 

s_cmp_eq_i32 s53, 0 // (op 255) 
s_cselect_b32 s52, 0, s54 // (op 255) 


s_cmp_lt_u32 s8, s52 // (op 255) 
s_mov_b32 s53, scc // (op 255) 
s_cmp_eq_i32 s53, 0 // (op 255) 
s_cselect_b32 s54, s52, s8 // (op 255) 


s_cmp_lt_u32 s0, s54 // (op 255) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_255 // (op 255) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_255 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_255:
v_lshrrev_b32 v12, 7, v1 // (op 14345) 
s_mul_i32 s52, s16, s8 // (op 14345) 
s_add_u32 s52, s52, s0 // (op 14345) 
v_and_b32 v13, 63, v1 // (op 14345) 
s_mul_hi_u32 s53, s52, s13 // (op 14345) 
v_lshrrev_b32 v14, 4, v13 // (op 14345) 
s_sub_u32 s54, s52, s53 // (op 14345) 
v_lshlrev_b32 v15, 2, v14 // (op 14345) 

v_lshl_add_u32 v14, v12, 6, v15 // (op 14345) 


s_lshr_b32 s55, s54, 1 // (op 14345) 

s_add_u32 s54, s55, s53 // (op 14345) 


v_lshrrev_b32 v12, 2, v13 // (op 14345) 

s_lshr_b32 s53, s54, s11 // (op 14345) 

v_and_b32 v13, 3, v12 // (op 14345) 

s_and_b32 s54, s11, 2147483648 // (op 14345) 
v_and_b32 v12, 3, v1 // (op 14345) 
s_cmp_eq_i32 s54, 0 // (op 14345) 
s_mov_b32 s55, scc // (op 14345) 

v_lshl_add_u32 v15, v13, 2, v12 // (op 14345) 


s_cmp_eq_i32 s55, 0 // (op 14345) 
s_cselect_b32 s54, s52, s53 // (op 14345) 



v_lshrrev_b32 v12, 6, v1 // (op 14345) 
s_mul_hi_u32 s52, s54, s12 // (op 14345) 
v_and_b32 v13, 1, v12 // (op 14345) 

s_sub_u32 s53, s54, s52 // (op 14345) 
v_lshl_add_u32 v12, v13, 6, v15 // (op 14345) 


s_lshr_b32 s55, s53, 1 // (op 14345) 

s_add_u32 s53, s55, s52 // (op 14345) 


s_lshr_b32 s52, s53, s10 // (op 14345) 

s_and_b32 s53, s10, 2147483648 // (op 14345) 
s_cmp_eq_i32 s53, 0 // (op 14345) 
s_mov_b32 s55, scc // (op 14345) 

s_cmp_eq_i32 s55, 0 // (op 14345) 
s_cselect_b32 s53, s54, s52 // (op 14345) 


s_mul_i32 s52, s53, s14 // (op 14345) 
v_lshl_add_u32 v13, s53, 7, v14 // (op 14345) 


s_sub_u32 s53, s54, s52 // (op 14345) 


v_lshl_add_u32 v14, s53, 7, v12 // (op 14345) 


v_mul_lo_u32 v12, v14, s4 // (op 14345) 

v_add_lshl_u32 v14, v13, v12, 2 // (op 14345) 


s_mul_i32 s53, s4, s23 // (op 14347) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 14347) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 14347) least significant
s_mul_hi_u32 s55, s4, s22 // (op 14347) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14347) most significant: sum
s_add_u32 s53, s53, s55 // (op 14347) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 14347) 

s_mov_b32 s56, s20 // (op 14347) reg expression
s_mov_b32 s57, s21 // (op 14347) reg expression
s_mov_b32 s58, s54 // (op 14347) reg expression
s_mov_b32 s59, 131072 // (op 14347) call()

v_lshrrev_b32 v12, 7, v1 // (op 14451) 
v_and_b32 v13, 15, v1 // (op 14451) 
v_lshl_add_u32 v15, v12, 6, v13 // (op 14451) 


v_and_b32 v12, 48, v1 // (op 14451) 
v_lshl_add_u32 v13, v15, 7, v12 // (op 14451) 


v_lshrrev_b32 v12, 6, v1 // (op 14526) 
v_and_b32 v15, 1, v12 // (op 14526) 

v_and_b32 v12, 15, v1 // (op 14526) 
v_lshl_add_u32 v16, v15, 6, v12 // (op 14526) 


v_and_b32 v12, 63, v1 // (op 14526) 
v_lshrrev_b32 v15, 4, v12 // (op 14526) 

v_lshlrev_b32 v12, 5, v15 // (op 14526) 

v_lshl_add_u32 v15, v16, 7, v12 // (op 14526) 


v_lshlrev_b32 v12, 2, v15 // (op 14526) 

v_lshrrev_b32 v15, 3, v12 // (op 14526) 

v_lshrrev_b32 v12, 7, v1 // (op 14549) 
s_mul_i32 s52, s16, s8 // (op 14549) 
s_add_u32 s52, s52, s0 // (op 14549) 
v_and_b32 v16, 63, v1 // (op 14549) 
s_mul_hi_u32 s53, s52, s13 // (op 14549) 
v_lshrrev_b32 v17, 4, v16 // (op 14549) 
s_sub_u32 s54, s52, s53 // (op 14549) 
v_lshlrev_b32 v18, 2, v17 // (op 14549) 

s_lshr_b32 s55, s54, 1 // (op 14549) 

v_lshl_add_u32 v17, v12, 6, v18 // (op 14549) 


s_add_u32 s54, s55, s53 // (op 14549) 


v_lshrrev_b32 v12, 2, v16 // (op 14549) 

s_lshr_b32 s53, s54, s11 // (op 14549) 

v_and_b32 v16, 3, v12 // (op 14549) 

s_and_b32 s54, s11, 2147483648 // (op 14549) 
v_and_b32 v12, 3, v1 // (op 14549) 
s_cmp_eq_i32 s54, 0 // (op 14549) 
s_mov_b32 s55, scc // (op 14549) 

v_lshl_add_u32 v18, v16, 2, v12 // (op 14549) 


s_cmp_eq_i32 s55, 0 // (op 14549) 
s_cselect_b32 s54, s52, s53 // (op 14549) 



v_lshrrev_b32 v12, 6, v1 // (op 14549) 
s_mul_hi_u32 s55, s54, s12 // (op 14549) 
v_and_b32 v16, 1, v12 // (op 14549) 

s_sub_u32 s52, s54, s55 // (op 14549) 
v_lshl_add_u32 v12, v16, 6, v18 // (op 14549) 


s_lshr_b32 s53, s52, 1 // (op 14549) 

s_add_u32 s52, s53, s55 // (op 14549) 


s_lshr_b32 s53, s52, s10 // (op 14549) 

s_and_b32 s52, s10, 2147483648 // (op 14549) 
s_cmp_eq_i32 s52, 0 // (op 14549) 
s_mov_b32 s55, scc // (op 14549) 

s_cmp_eq_i32 s55, 0 // (op 14549) 
s_cselect_b32 s52, s54, s53 // (op 14549) 


s_mul_i32 s53, s52, s14 // (op 14549) 
v_lshl_add_u32 v16, s52, 7, v17 // (op 14549) 


s_sub_u32 s52, s54, s53 // (op 14549) 


v_lshl_add_u32 v17, s52, 7, v12 // (op 14549) 


v_mul_lo_u32 v12, v17, s24 // (op 14549) 

v_add_lshl_u32 v17, v16, v12, 2 // (op 14549) 


s_mul_i32 s53, s24, s23 // (op 14551) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 14551) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 14551) least significant
s_mul_hi_u32 s55, s24, s22 // (op 14551) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14551) most significant: sum
s_add_u32 s53, s53, s55 // (op 14551) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 14551) 

s_mov_b32 s60, s26 // (op 14551) reg expression
s_mov_b32 s61, s27 // (op 14551) reg expression
s_mov_b32 s62, s54 // (op 14551) reg expression
s_mov_b32 s63, 131072 // (op 14551) call()

v_lshrrev_b32 v12, 5, v1 // (op 14751) 
v_lshlrev_b32 v16, 2, v12 // (op 14751) 

v_lshl_add_u32 v12, s16, 7, v16 // (op 14751) 

v_and_b32 v16, -32, v1 // (op 14751) 
v_sub_u32 v18, v1, v16 // (op 14751) 

v_lshl_add_u32 v16, v12, 7, v18 // (op 14751) 


v_lshlrev_b32 v12, 2, v16 // (op 14751) 

s_lshl_b32 s52, s15, 16 // (op 14753) 
s_mov_b32 s64, s28 // (op 14753) reg expression
s_mov_b32 s65, s29 // (op 14753) reg expression
s_mov_b32 s66, s52 // (op 14753) reg expression
s_mov_b32 s67, 131072 // (op 14753) call()

s_mul_i32 s6, s8, s16 // (op 260) 
s_add_u32 s6, s6, s0 // (op 260) 
s_mul_hi_u32 s52, s6, s13 // (op 261) 
s_sub_u32 s53, s6, s52 // (op 261) 
s_lshr_b32 s54, s53, 1 // (op 261) 

s_add_u32 s53, s54, s52 // (op 261) 


s_lshr_b32 s52, s53, s11 // (op 261) 

s_and_b32 s53, s11, 2147483648 // (op 261) 
s_cmp_eq_i32 s53, 0 // (op 261) 
s_mov_b32 s54, scc // (op 261) 

s_cmp_eq_i32 s54, 0 // (op 261) 
s_cselect_b32 s53, s6, s52 // (op 261) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 261) 
s_mul_i32 s52, s53, s54 // (op 261) 


s_sub_u32 s18, s6, s52 // (op 261) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 262) 
s_cmp_gt_u32 s52, s18 // (op 262) 
s_mov_b32 s54, scc // (op 262) 
s_sub_u32 s55, s52, s18 // (op 262) 

s_cmp_eq_i32 s54, 0 // (op 262) 
s_cselect_b32 s52, 0, s55 // (op 262) 


s_cmp_gt_u32 s8, s0 // (op 262) 
s_mov_b32 s53, scc // (op 262) 
s_sub_u32 s54, s8, s0 // (op 262) 
s_cmp_eq_i32 s53, 0 // (op 262) 
s_cselect_b32 s55, 0, s54 // (op 262) 


s_cmp_lt_u32 s52, s55 // (op 262) 
s_mov_b32 s53, scc // (op 262) 
s_cmp_eq_i32 s53, 0 // (op 262) 
s_cselect_b32 s54, s55, s52 // (op 262) 



s_cmp_gt_u32 s9, s6 // (op 262) 
s_mov_b32 s55, scc // (op 262) 
s_sub_u32 s52, s9, s6 // (op 262) 
s_cmp_eq_i32 s55, 0 // (op 262) 
s_cselect_b32 s53, 0, s52 // (op 262) 


s_cmp_lt_u32 s54, s53 // (op 262) 
s_mov_b32 s52, scc // (op 262) 
s_cmp_eq_i32 s52, 0 // (op 262) 
s_cselect_b32 s7, s53, s54 // (op 262) 



s_add_u32 s52, s18, s7 // (op 263) 
s_sub_u32 s19, s52, 1 // (op 263) 

v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_accvgpr_write a4, 0.00000 // (op 4023) call()
v_accvgpr_write a5, 0.00000 // (op 4023) call()
v_accvgpr_write a6, 0.00000 // (op 4023) call()
v_accvgpr_write a7, 0.00000 // (op 4023) call()
v_accvgpr_write a8, 0.00000 // (op 4159) call()
v_accvgpr_write a9, 0.00000 // (op 4159) call()
v_accvgpr_write a10, 0.00000 // (op 4159) call()
v_accvgpr_write a11, 0.00000 // (op 4159) call()
v_accvgpr_write a12, 0.00000 // (op 4295) call()
v_accvgpr_write a13, 0.00000 // (op 4295) call()
v_accvgpr_write a14, 0.00000 // (op 4295) call()
v_accvgpr_write a15, 0.00000 // (op 4295) call()
v_accvgpr_write a16, 0.00000 // (op 4519) call()
v_accvgpr_write a17, 0.00000 // (op 4519) call()
v_accvgpr_write a18, 0.00000 // (op 4519) call()
v_accvgpr_write a19, 0.00000 // (op 4519) call()
v_accvgpr_write a20, 0.00000 // (op 4780) call()
v_accvgpr_write a21, 0.00000 // (op 4780) call()
v_accvgpr_write a22, 0.00000 // (op 4780) call()
v_accvgpr_write a23, 0.00000 // (op 4780) call()
v_accvgpr_write a24, 0.00000 // (op 4781) call()
v_accvgpr_write a25, 0.00000 // (op 4781) call()
v_accvgpr_write a26, 0.00000 // (op 4781) call()
v_accvgpr_write a27, 0.00000 // (op 4781) call()
v_accvgpr_write a28, 0.00000 // (op 4782) call()
v_accvgpr_write a29, 0.00000 // (op 4782) call()
v_accvgpr_write a30, 0.00000 // (op 4782) call()
v_accvgpr_write a31, 0.00000 // (op 4782) call()
v_accvgpr_write a32, 0.00000 // (op 5129) call()
v_accvgpr_write a33, 0.00000 // (op 5129) call()
v_accvgpr_write a34, 0.00000 // (op 5129) call()
v_accvgpr_write a35, 0.00000 // (op 5129) call()
v_accvgpr_write a36, 0.00000 // (op 5390) call()
v_accvgpr_write a37, 0.00000 // (op 5390) call()
v_accvgpr_write a38, 0.00000 // (op 5390) call()
v_accvgpr_write a39, 0.00000 // (op 5390) call()
v_accvgpr_write a40, 0.00000 // (op 5391) call()
v_accvgpr_write a41, 0.00000 // (op 5391) call()
v_accvgpr_write a42, 0.00000 // (op 5391) call()
v_accvgpr_write a43, 0.00000 // (op 5391) call()
v_accvgpr_write a44, 0.00000 // (op 5392) call()
v_accvgpr_write a45, 0.00000 // (op 5392) call()
v_accvgpr_write a46, 0.00000 // (op 5392) call()
v_accvgpr_write a47, 0.00000 // (op 5392) call()
v_accvgpr_write a48, 0.00000 // (op 5739) call()
v_accvgpr_write a49, 0.00000 // (op 5739) call()
v_accvgpr_write a50, 0.00000 // (op 5739) call()
v_accvgpr_write a51, 0.00000 // (op 5739) call()
v_accvgpr_write a52, 0.00000 // (op 6000) call()
v_accvgpr_write a53, 0.00000 // (op 6000) call()
v_accvgpr_write a54, 0.00000 // (op 6000) call()
v_accvgpr_write a55, 0.00000 // (op 6000) call()
v_accvgpr_write a56, 0.00000 // (op 6001) call()
v_accvgpr_write a57, 0.00000 // (op 6001) call()
v_accvgpr_write a58, 0.00000 // (op 6001) call()
v_accvgpr_write a59, 0.00000 // (op 6001) call()
v_accvgpr_write a60, 0.00000 // (op 6002) call()
v_accvgpr_write a61, 0.00000 // (op 6002) call()
v_accvgpr_write a62, 0.00000 // (op 6002) call()
v_accvgpr_write a63, 0.00000 // (op 6002) call()
v_lshrrev_b32 v16, 3, v1 // (op 14168) 
s_mul_i32 s52, s16, s8 // (op 14168) 
s_add_u32 s52, s52, s0 // (op 14168) 
v_and_b32 v18, -8, v1 // (op 14168) 
s_mul_hi_u32 s53, s52, s13 // (op 14168) 
v_sub_u32 v19, v1, v18 // (op 14168) 

s_sub_u32 s54, s52, s53 // (op 14168) 
v_lshlrev_b32 v18, 4, v19 // (op 14168) 

s_lshr_b32 s55, s54, 1 // (op 14168) 

s_add_u32 s54, s55, s53 // (op 14168) 


s_lshr_b32 s53, s54, s11 // (op 14168) 

s_and_b32 s54, s11, 2147483648 // (op 14168) 
s_cmp_eq_i32 s54, 0 // (op 14168) 
s_mov_b32 s55, scc // (op 14168) 

s_cmp_eq_i32 s55, 0 // (op 14168) 
s_cselect_b32 s54, s52, s53 // (op 14168) 


s_mul_hi_u32 s53, s54, s12 // (op 14168) 
s_sub_u32 s55, s54, s53 // (op 14168) 
s_lshr_b32 s68, s55, 1 // (op 14168) 

s_add_u32 s55, s68, s53 // (op 14168) 


s_lshr_b32 s53, s55, s10 // (op 14168) 

s_and_b32 s55, s10, 2147483648 // (op 14168) 
s_cmp_eq_i32 s55, 0 // (op 14168) 
s_mov_b32 s68, scc // (op 14168) 

s_cmp_eq_i32 s68, 0 // (op 14168) 
s_cselect_b32 s55, s54, s53 // (op 14168) 


v_lshl_add_u32 v19, s55, 7, v16 // (op 14168) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 14168) 
s_mul_i32 s53, s54, s68 // (op 14168) 


s_sub_u32 s54, s52, s53 // (op 14168) 


v_lshl_add_u32 v16, s54, 7, v18 // (op 14168) 


v_mov_b32 v21, 0 // (op 14168) convert
v_mov_b32 v20, v16 // (op 14168) convert

v_mul_lo_u32 v23, v19, s31 // (op 14168) most significant: low of low * high
v_mul_hi_u32 v16, v19, s30 // (op 14168) most significant: high of low * low
v_mul_lo_u32 v22, v19, s30 // (op 14168) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 14168) most significant: sum

v_add_co_u32 v22, s[52:53], v22, v20 // (op 14168) least significant half
v_addc_co_u32 v23, s[52:53], v23, v21, s[52:53] // (op 14168) most significant half



s_mul_i32 s53, s30, s35 // (op 14170) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 14170) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 14170) least significant
s_mul_hi_u32 s55, s30, s34 // (op 14170) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14170) most significant: sum
s_add_u32 s53, s53, s55 // (op 14170) most significant: sum


s_mov_b32 s68, s32 // (op 14170) reg expression
s_mov_b32 s69, s33 // (op 14170) reg expression
s_mov_b32 s70, s52 // (op 14170) reg expression
s_mov_b32 s71, 131072 // (op 14170) call()

v_lshrrev_b32 v16, 3, v1 // (op 14187) 
s_mul_i32 s52, s16, s8 // (op 14187) 
s_add_u32 s52, s52, s0 // (op 14187) 
v_and_b32 v18, -8, v1 // (op 14187) 
s_mul_hi_u32 s53, s52, s13 // (op 14187) 
v_sub_u32 v19, v1, v18 // (op 14187) 

s_sub_u32 s54, s52, s53 // (op 14187) 
v_lshlrev_b32 v18, 4, v19 // (op 14187) 

s_lshr_b32 s55, s54, 1 // (op 14187) 

s_add_u32 s54, s55, s53 // (op 14187) 


s_lshr_b32 s53, s54, s11 // (op 14187) 

s_and_b32 s54, s11, 2147483648 // (op 14187) 
s_cmp_eq_i32 s54, 0 // (op 14187) 
s_mov_b32 s55, scc // (op 14187) 

s_cmp_eq_i32 s55, 0 // (op 14187) 
s_cselect_b32 s54, s52, s53 // (op 14187) 


s_mul_hi_u32 s53, s54, s12 // (op 14187) 
s_sub_u32 s55, s54, s53 // (op 14187) 
s_lshr_b32 s72, s55, 1 // (op 14187) 

s_add_u32 s55, s72, s53 // (op 14187) 


s_lshr_b32 s53, s55, s10 // (op 14187) 

s_and_b32 s55, s10, 2147483648 // (op 14187) 
s_cmp_eq_i32 s55, 0 // (op 14187) 
s_mov_b32 s72, scc // (op 14187) 

s_cmp_eq_i32 s72, 0 // (op 14187) 
s_cselect_b32 s55, s54, s53 // (op 14187) 


v_lshl_add_u32 v19, s55, 7, v16 // (op 14187) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 14187) 
s_mul_i32 s53, s54, s72 // (op 14187) 


s_sub_u32 s54, s52, s53 // (op 14187) 


v_lshl_add_u32 v16, s54, 7, v18 // (op 14187) 


v_mov_b32 v21, 0 // (op 14187) convert
v_mov_b32 v20, v16 // (op 14187) convert

v_mul_lo_u32 v25, v19, s31 // (op 14187) most significant: low of low * high
v_mul_hi_u32 v16, v19, s30 // (op 14187) most significant: high of low * low
v_mul_lo_u32 v24, v19, s30 // (op 14187) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 14187) most significant: sum

v_add_co_u32 v24, s[52:53], v24, v20 // (op 14187) least significant half
v_addc_co_u32 v25, s[52:53], v25, v21, s[52:53] // (op 14187) most significant half



s_mul_i32 s53, s30, s35 // (op 14189) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 14189) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 14189) least significant
s_mul_hi_u32 s55, s30, s34 // (op 14189) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14189) most significant: sum
s_add_u32 s53, s53, s55 // (op 14189) most significant: sum


s_mov_b32 s68, s32 // (op 14189) reg expression
s_mov_b32 s69, s33 // (op 14189) reg expression
s_mov_b32 s70, s52 // (op 14189) reg expression
s_mov_b32 s71, 131072 // (op 14189) call()

v_lshrrev_b32 v16, 2, v1 // (op 14296) 
s_mul_i32 s52, s16, s8 // (op 14296) 
s_add_u32 s52, s52, s0 // (op 14296) 
v_and_b32 v18, -4, v1 // (op 14296) 
s_mul_hi_u32 s53, s52, s13 // (op 14296) 
v_sub_u32 v19, v1, v18 // (op 14296) 

s_sub_u32 s54, s52, s53 // (op 14296) 
v_lshlrev_b32 v18, 5, v19 // (op 14296) 

s_lshr_b32 s55, s54, 1 // (op 14296) 

s_add_u32 s54, s55, s53 // (op 14296) 


s_lshr_b32 s53, s54, s11 // (op 14296) 

s_and_b32 s54, s11, 2147483648 // (op 14296) 
s_cmp_eq_i32 s54, 0 // (op 14296) 
s_mov_b32 s55, scc // (op 14296) 

s_cmp_eq_i32 s55, 0 // (op 14296) 
s_cselect_b32 s54, s52, s53 // (op 14296) 


s_mul_hi_u32 s53, s54, s12 // (op 14296) 
s_sub_u32 s55, s54, s53 // (op 14296) 
s_lshr_b32 s72, s55, 1 // (op 14296) 

s_add_u32 s55, s72, s53 // (op 14296) 


s_lshr_b32 s53, s55, s10 // (op 14296) 

s_and_b32 s55, s10, 2147483648 // (op 14296) 
s_cmp_eq_i32 s55, 0 // (op 14296) 
s_mov_b32 s72, scc // (op 14296) 

s_cmp_eq_i32 s72, 0 // (op 14296) 
s_cselect_b32 s55, s54, s53 // (op 14296) 


s_mul_i32 s53, s55, s14 // (op 14296) 

s_sub_u32 s55, s54, s53 // (op 14296) 

v_lshl_add_u32 v19, s55, 7, v16 // (op 14296) 


v_mul_lo_u32 v21, v19, s37 // (op 14296) most significant: low of low * high
v_mul_hi_u32 v16, v19, s36 // (op 14296) most significant: high of low * low
v_mul_lo_u32 v20, v19, s36 // (op 14296) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 14296) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 14296) 
s_mul_i32 s53, s54, s72 // (op 14296) 


s_sub_u32 s54, s52, s53 // (op 14296) 


v_lshl_add_u32 v16, s54, 7, v18 // (op 14296) 


v_mov_b32 v19, 0 // (op 14296) convert
v_mov_b32 v18, v16 // (op 14296) convert

v_add_co_u32 v26, s[52:53], v18, v20 // (op 14296) least significant half
v_addc_co_u32 v27, s[52:53], v19, v21, s[52:53] // (op 14296) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 14296) 


v_lshrrev_b64 v[18:19], 3, v[26:27] // (op 14296) 

s_mul_i32 s53, s36, s23 // (op 14298) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 14298) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 14298) least significant
s_mul_hi_u32 s55, s36, s22 // (op 14298) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14298) most significant: sum
s_add_u32 s53, s53, s55 // (op 14298) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 14298) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 14298) 

s_mov_b32 s72, s38 // (op 14298) reg expression
s_mov_b32 s73, s39 // (op 14298) reg expression
s_mov_b32 s74, s52 // (op 14298) reg expression
s_mov_b32 s75, 131072 // (op 14298) call()

v_lshrrev_b32 v16, 2, v1 // (op 14315) 
s_mul_i32 s52, s16, s8 // (op 14315) 
s_add_u32 s52, s52, s0 // (op 14315) 
v_and_b32 v20, -4, v1 // (op 14315) 
s_mul_hi_u32 s53, s52, s13 // (op 14315) 
v_sub_u32 v21, v1, v20 // (op 14315) 

s_sub_u32 s54, s52, s53 // (op 14315) 
v_lshlrev_b32 v20, 5, v21 // (op 14315) 

s_lshr_b32 s55, s54, 1 // (op 14315) 

s_add_u32 s54, s55, s53 // (op 14315) 


s_lshr_b32 s53, s54, s11 // (op 14315) 

s_and_b32 s54, s11, 2147483648 // (op 14315) 
s_cmp_eq_i32 s54, 0 // (op 14315) 
s_mov_b32 s55, scc // (op 14315) 

s_cmp_eq_i32 s55, 0 // (op 14315) 
s_cselect_b32 s54, s52, s53 // (op 14315) 


s_mul_hi_u32 s53, s54, s12 // (op 14315) 
s_sub_u32 s55, s54, s53 // (op 14315) 
s_lshr_b32 s76, s55, 1 // (op 14315) 

s_add_u32 s55, s76, s53 // (op 14315) 


s_lshr_b32 s53, s55, s10 // (op 14315) 

s_and_b32 s55, s10, 2147483648 // (op 14315) 
s_cmp_eq_i32 s55, 0 // (op 14315) 
s_mov_b32 s76, scc // (op 14315) 

s_cmp_eq_i32 s76, 0 // (op 14315) 
s_cselect_b32 s55, s54, s53 // (op 14315) 


s_mul_i32 s53, s55, s14 // (op 14315) 

s_sub_u32 s55, s54, s53 // (op 14315) 

v_lshl_add_u32 v21, s55, 7, v16 // (op 14315) 


v_mul_lo_u32 v27, v21, s37 // (op 14315) most significant: low of low * high
v_mul_hi_u32 v16, v21, s36 // (op 14315) most significant: high of low * low
v_mul_lo_u32 v26, v21, s36 // (op 14315) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 14315) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 14315) 
s_mul_i32 s53, s54, s76 // (op 14315) 


s_sub_u32 s54, s52, s53 // (op 14315) 


v_lshl_add_u32 v16, s54, 7, v20 // (op 14315) 


v_mov_b32 v21, 0 // (op 14315) convert
v_mov_b32 v20, v16 // (op 14315) convert

v_add_co_u32 v28, s[52:53], v20, v26 // (op 14315) least significant half
v_addc_co_u32 v29, s[52:53], v21, v27, s[52:53] // (op 14315) most significant half

v_lshlrev_b64 v[28:29], 2, v[28:29] // (op 14315) 


v_lshrrev_b64 v[20:21], 3, v[28:29] // (op 14315) 

s_mul_i32 s53, s36, s23 // (op 14317) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 14317) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 14317) least significant
s_mul_hi_u32 s55, s36, s22 // (op 14317) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14317) most significant: sum
s_add_u32 s53, s53, s55 // (op 14317) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 14317) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 14317) 

s_mov_b32 s72, s38 // (op 14317) reg expression
s_mov_b32 s73, s39 // (op 14317) reg expression
s_mov_b32 s74, s52 // (op 14317) reg expression
s_mov_b32 s75, 131072 // (op 14317) call()

v_and_b32 v16, 63, v1 // (op 15133) 
s_mul_i32 s52, s16, s8 // (op 15133) 
s_add_u32 s52, s52, s0 // (op 15133) 
v_lshrrev_b32 v26, 4, v16 // (op 15133) 
s_mul_hi_u32 s53, s52, s13 // (op 15133) 
v_and_b32 v27, 3, v26 // (op 15133) 

s_sub_u32 s54, s52, s53 // (op 15133) 
v_and_b32 v26, 15, v1 // (op 15133) 
s_lshr_b32 s55, s54, 1 // (op 15133) 

v_lshl_add_u32 v28, v27, 4, v26 // (op 15133) 


s_add_u32 s54, s55, s53 // (op 15133) 


v_lshrrev_b32 v26, 7, v1 // (op 15133) 
s_lshr_b32 s53, s54, s11 // (op 15133) 

v_lshl_add_u32 v27, v26, 6, v28 // (op 15133) 


s_and_b32 s54, s11, 2147483648 // (op 15133) 
v_lshrrev_b32 v26, 6, v16 // (op 15133) 

s_cmp_eq_i32 s54, 0 // (op 15133) 
s_mov_b32 s55, scc // (op 15133) 

v_lshlrev_b32 v16, 2, v26 // (op 15133) 

s_cmp_eq_i32 s55, 0 // (op 15133) 
s_cselect_b32 s54, s52, s53 // (op 15133) 


s_mul_hi_u32 s53, s54, s12 // (op 15133) 
s_sub_u32 s55, s54, s53 // (op 15133) 
s_lshr_b32 s76, s55, 1 // (op 15133) 

s_add_u32 s55, s76, s53 // (op 15133) 


s_lshr_b32 s53, s55, s10 // (op 15133) 

s_and_b32 s55, s10, 2147483648 // (op 15133) 
s_cmp_eq_i32 s55, 0 // (op 15133) 
s_mov_b32 s76, scc // (op 15133) 

s_cmp_eq_i32 s76, 0 // (op 15133) 
s_cselect_b32 s55, s54, s53 // (op 15133) 


v_lshl_add_u32 v26, s55, 7, v27 // (op 15133) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 15133) 
s_mul_i32 s53, s54, s76 // (op 15133) 


s_sub_u32 s54, s52, s53 // (op 15133) 


v_lshl_add_u32 v27, s54, 2, v16 // (op 15133) 


v_mul_lo_u32 v16, v26, s40 // (op 15133) 
v_add_u32 v16, v16, v27 // (op 15133) 


s_mul_i32 s53, s40, s35 // (op 15135) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 15135) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 15135) least significant
s_mul_hi_u32 s55, s40, s34 // (op 15135) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 15135) most significant: sum
s_add_u32 s53, s53, s55 // (op 15135) most significant: sum


s_mov_b32 s76, s42 // (op 15135) reg expression
s_mov_b32 s77, s43 // (op 15135) reg expression
s_mov_b32 s78, s52 // (op 15135) reg expression
s_mov_b32 s79, 131072 // (op 15135) call()

v_and_b32 v26, 63, v1 // (op 15160) 
s_mul_i32 s52, s16, s8 // (op 15160) 
s_add_u32 s52, s52, s0 // (op 15160) 
v_lshrrev_b32 v27, 4, v26 // (op 15160) 
s_mul_hi_u32 s53, s52, s13 // (op 15160) 
v_and_b32 v28, 3, v27 // (op 15160) 

s_sub_u32 s54, s52, s53 // (op 15160) 
v_and_b32 v27, 15, v1 // (op 15160) 
s_lshr_b32 s55, s54, 1 // (op 15160) 

v_lshl_add_u32 v29, v28, 4, v27 // (op 15160) 


s_add_u32 s54, s55, s53 // (op 15160) 


v_lshrrev_b32 v27, 6, v1 // (op 15160) 
s_lshr_b32 s53, s54, s11 // (op 15160) 

v_and_b32 v28, 1, v27 // (op 15160) 

s_and_b32 s54, s11, 2147483648 // (op 15160) 
v_lshl_add_u32 v27, v28, 6, v29 // (op 15160) 


s_cmp_eq_i32 s54, 0 // (op 15160) 
s_mov_b32 s55, scc // (op 15160) 

v_lshrrev_b32 v28, 6, v26 // (op 15160) 

s_cmp_eq_i32 s55, 0 // (op 15160) 
s_cselect_b32 s54, s52, s53 // (op 15160) 


v_lshlrev_b32 v26, 2, v28 // (op 15160) 

s_mul_hi_u32 s53, s54, s12 // (op 15160) 
s_sub_u32 s55, s54, s53 // (op 15160) 
s_lshr_b32 s80, s55, 1 // (op 15160) 

s_add_u32 s55, s80, s53 // (op 15160) 


s_lshr_b32 s53, s55, s10 // (op 15160) 

s_and_b32 s55, s10, 2147483648 // (op 15160) 
s_cmp_eq_i32 s55, 0 // (op 15160) 
s_mov_b32 s80, scc // (op 15160) 

s_cmp_eq_i32 s80, 0 // (op 15160) 
s_cselect_b32 s55, s54, s53 // (op 15160) 


s_mul_i32 s53, s55, s14 // (op 15160) 

s_sub_u32 s55, s54, s53 // (op 15160) 

v_lshl_add_u32 v28, s55, 7, v27 // (op 15160) 


v_mul_lo_u32 v27, v28, s44 // (op 15160) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 15160) 
s_mul_i32 s53, s54, s80 // (op 15160) 


s_sub_u32 s54, s52, s53 // (op 15160) 


v_lshl_add_u32 v28, s54, 2, v26 // (op 15160) 


v_add_u32 v26, v28, v27 // (op 15160) 


s_mul_i32 s53, s44, s23 // (op 15162) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 15162) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 15162) least significant
s_mul_hi_u32 s55, s44, s22 // (op 15162) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 15162) most significant: sum
s_add_u32 s53, s53, s55 // (op 15162) most significant: sum


s_mov_b32 s80, s46 // (op 15162) reg expression
s_mov_b32 s81, s47 // (op 15162) reg expression
s_mov_b32 s82, s52 // (op 15162) reg expression
s_mov_b32 s83, 131072 // (op 15162) call()

v_and_b32 v27, -4, v1 // (op 14492) 
v_sub_u32 v28, v1, v27 // (op 14492) 

v_lshlrev_b32 v27, 5, v28 // (op 14492) 

v_lshrrev_b32 v28, 2, v1 // (op 14492) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 14492) 


v_lshlrev_b32 v27, 2, v29 // (op 14492) 

v_lshrrev_b32 v28, 3, v27 // (op 14492) 

s_nop 0
v_readfirstlane_b32 s52, v28 // (op 14492) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15461) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 14018) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 14018) 

v_lshrrev_b64 v[28:29], 3, s[84:85] // (op 14018) 


v_mov_b64 v[30:31], v[18:19] // (op 14018) 
s_mov_b32 m0, s52 // (op 14018) reg expression
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 14018) Load value direct to lds
v_add_co_u32 v30, s[54:55], v30, v28 // (op 14018) least significant half
v_addc_co_u32 v31, s[54:55], 0, v31, s[54:55] // (op 14018) most significant half

s_add_u32 m0, m0, 4096 // (op 14018) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 14018) Load value direct to lds



v_and_b32 v27, -8, v1 // (op 14417) 
v_sub_u32 v28, v1, v27 // (op 14417) 

v_lshlrev_b32 v27, 4, v28 // (op 14417) 

v_lshrrev_b32 v28, 3, v1 // (op 14417) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 14417) 


s_nop 0
v_readfirstlane_b32 s52, v29 // (op 14417) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15442) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 14023) 
v_mov_b64 v[28:29], s[54:55] // (op 14023) convert


v_mov_b64 v[30:31], v[22:23] // (op 14023) 
s_add_u32 s53, 8192, s52 // (op 14023) 
s_mov_b32 m0, s53 // (op 14023) reg expression
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 14023) Load value direct to lds
v_add_co_u32 v30, s[54:55], v30, v28 // (op 14023) least significant half
v_addc_co_u32 v31, s[54:55], 0, v31, s[54:55] // (op 14023) most significant half

s_add_u32 m0, m0, 4096 // (op 14023) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 14023) Load value direct to lds
v_add_co_u32 v30, s[54:55], v30, v28 // (op 14023) least significant half
v_addc_co_u32 v31, s[54:55], 0, v31, s[54:55] // (op 14023) most significant half

s_add_u32 m0, m0, 4096 // (op 14023) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 14023) Load value direct to lds
v_add_co_u32 v30, s[54:55], v30, v28 // (op 14023) least significant half
v_addc_co_u32 v31, s[54:55], 0, v31, s[54:55] // (op 14023) most significant half

s_add_u32 m0, m0, 4096 // (op 14023) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 14023) Load value direct to lds




v_and_b32 v27, -4, v1 // (op 14940) 
v_sub_u32 v28, v1, v27 // (op 14940) 

v_lshlrev_b32 v27, 5, v28 // (op 14940) 

v_lshrrev_b32 v28, 2, v1 // (op 14940) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 14940) 


v_lshlrev_b32 v27, 2, v29 // (op 14940) 

v_lshrrev_b32 v28, 3, v27 // (op 14940) 

s_nop 0
v_readfirstlane_b32 s52, v28 // (op 14940) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15548) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 14028) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 14028) 

v_lshrrev_b64 v[28:29], 3, s[84:85] // (op 14028) 


v_mov_b64 v[30:31], v[18:19] // (op 14028) 
s_add_u32 s53, 24576, s52 // (op 14028) 
s_mov_b32 m0, s53 // (op 14028) reg expression
buffer_load_dwordx4 v30, s[72:75], 64 offen offset: 0 lds // (op 14028) Load value direct to lds
v_add_co_u32 v30, s[54:55], v30, v28 // (op 14028) least significant half
v_addc_co_u32 v31, s[54:55], 0, v31, s[54:55] // (op 14028) most significant half

s_add_u32 m0, m0, 4096 // (op 14028) 
buffer_load_dwordx4 v30, s[72:75], 64 offen offset: 0 lds // (op 14028) Load value direct to lds





v_and_b32 v18, -8, v1 // (op 14896) 
v_sub_u32 v19, v1, v18 // (op 14896) 

v_lshlrev_b32 v18, 4, v19 // (op 14896) 

v_lshrrev_b32 v19, 3, v1 // (op 14896) 
v_lshl_add_u32 v27, v19, 7, v18 // (op 14896) 


s_nop 0
v_readfirstlane_b32 s52, v27 // (op 14896) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15538) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 14030) 
v_mov_b64 v[18:19], s[54:55] // (op 14030) convert


v_mov_b64 v[28:29], v[22:23] // (op 14030) 
s_add_u32 s53, 32768, s52 // (op 14030) 
s_mov_b32 m0, s53 // (op 14030) reg expression
s_mov_b32 s54, 128 // (op 14030) call()
buffer_load_dwordx4 v28, s[68:71], s54 offen offset: 0 lds // (op 14030) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v18 // (op 14030) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 14030) most significant half

s_add_u32 m0, m0, 4096 // (op 14030) 
buffer_load_dwordx4 v28, s[68:71], s54 offen offset: 0 lds // (op 14030) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v18 // (op 14030) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 14030) most significant half

s_add_u32 m0, m0, 4096 // (op 14030) 
buffer_load_dwordx4 v28, s[68:71], s54 offen offset: 0 lds // (op 14030) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v18 // (op 14030) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 14030) most significant half

s_add_u32 m0, m0, 4096 // (op 14030) 
buffer_load_dwordx4 v28, s[68:71], s54 offen offset: 0 lds // (op 14030) Load value direct to lds






s_waitcnt vmcnt(10)
s_barrier  // (op 15458) 



ds_read_b128 v[28:31], v15 // (op 13718) Load local data 



ds_read_b128 v[32:35], v15 offset:1024 // (op 13723) Load local data 



ds_read_b128 v[36:39], v15 offset:2048 // (op 13728) Load local data 



ds_read_b128 v[40:43], v15 offset:3072 // (op 13733) Load local data 
s_waitcnt vmcnt(6)
s_barrier  // (op 15439) 



ds_read_b128 v[44:47], v13 offset:8192 // (op 13698) Load local data 
ds_read_b128 v[48:51], v13 offset:8256 // (op 13698) Load local data 



ds_read_b128 v[52:55], v13 offset:10240 // (op 13703) Load local data 
ds_read_b128 v[56:59], v13 offset:10304 // (op 13703) Load local data 



ds_read_b128 v[60:63], v13 offset:12288 // (op 13708) Load local data 
ds_read_b128 v[64:67], v13 offset:12352 // (op 13708) Load local data 



ds_read_b128 v[68:71], v13 offset:14336 // (op 13713) Load local data 
ds_read_b128 v[72:75], v13 offset:14400 // (op 13713) Load local data 
v_and_b32 v27, -8, v1 // (op 14402) 
v_sub_u32 v18, v1, v27 // (op 14402) 

v_lshlrev_b32 v19, 4, v18 // (op 14402) 

v_lshrrev_b32 v18, 3, v1 // (op 14402) 
v_lshl_add_u32 v27, v18, 7, v19 // (op 14402) 


s_nop 0
v_readfirstlane_b32 s52, v27 // (op 14402) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -4, v1 // (op 14477) 
v_sub_u32 v18, v1, v27 // (op 14477) 

v_lshlrev_b32 v19, 5, v18 // (op 14477) 

v_lshrrev_b32 v18, 2, v1 // (op 14477) 
v_lshl_add_u32 v27, v18, 7, v19 // (op 14477) 


v_lshlrev_b32 v18, 2, v27 // (op 14477) 

v_lshrrev_b32 v19, 3, v18 // (op 14477) 

s_nop 0
v_readfirstlane_b32 s53, v19 // (op 14477) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -8, v1 // (op 14881) 
v_sub_u32 v18, v1, v27 // (op 14881) 

v_lshlrev_b32 v19, 4, v18 // (op 14881) 

v_lshrrev_b32 v18, 3, v1 // (op 14881) 
v_lshl_add_u32 v27, v18, 7, v19 // (op 14881) 


s_nop 0
v_readfirstlane_b32 s54, v27 // (op 14881) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -4, v1 // (op 14925) 
v_sub_u32 v18, v1, v27 // (op 14925) 

v_lshlrev_b32 v19, 5, v18 // (op 14925) 

v_lshrrev_b32 v18, 2, v1 // (op 14925) 
v_lshl_add_u32 v27, v18, 7, v19 // (op 14925) 


v_lshlrev_b32 v18, 2, v27 // (op 14925) 

v_lshrrev_b32 v19, 3, v18 // (op 14925) 

s_nop 0
v_readfirstlane_b32 s55, v19 // (op 14925) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 5741) call()
s_and_b32 s84, s7, 4294967294 // (op 5740) 
s_cmp_lt_u32 s1, s84 // (op 5740) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_5740 // (op 5740) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_5740 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_5740:


s_add_u32 s1, s1, 2 // (op 5742) 
v_mov_b32 v27, v16 // (op 70) 
v_mov_b32 v18, v27 // (op 70) reg expression

buffer_load_dwordx2 v[22:23], v18, s[76:79], 0 offen // (op 70) Load value


v_add_u32 v16, 8, v16 // (op 15123) 


v_mov_b32 v27, v26 // (op 74) 
v_mov_b32 v18, v27 // (op 74) reg expression

buffer_load_dwordx2 v[76:77], v18, s[80:83], 0 offen // (op 74) Load value


v_add_u32 v26, 8, v26 // (op 15150) 
s_waitcnt vmcnt(1)
v_bfe_u32 v78, v22, 0, 8 // (op 12947) 
v_bfe_u32 v79, v22, 8, 8 // (op 12947) 
v_bfe_u32 v80, v22, 16, 8 // (op 12947) 
v_bfe_u32 v81, v22, 24, 8 // (op 12947) 
v_permlane16_swap_b32 v78, v79 // (op 12947) 
s_nop 0
v_permlane16_swap_b32 v80, v81 // (op 12947) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v78, v80 // (op 12947) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v79, v81 // (op 12947) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 15464) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 13945) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 13945) 

v_lshrrev_b64 v[18:19], 3, s[86:87] // (op 13945) 


v_mov_b64 v[82:83], v[20:21] // (op 13945) 
v_mov_b64 v[84:85], v[82:83] // (op 13945) 
s_mov_b32 m0, s53 // (op 13945) reg expression
s_mov_b32 s84, 128 // (op 13945) call()
buffer_load_dwordx4 v84, s[72:75], s84 offen offset: 0 lds // (op 13945) Load value direct to lds
v_add_co_u32 v84, s[86:87], v84, v18 // (op 13945) least significant half
v_addc_co_u32 v85, s[86:87], 0, v85, s[86:87] // (op 13945) most significant half

s_add_u32 m0, m0, 4096 // (op 13945) 
buffer_load_dwordx4 v84, s[72:75], s84 offen offset: 0 lds // (op 13945) Load value direct to lds



s_waitcnt vmcnt(2)
v_bfe_u32 v84, v76, 0, 8 // (op 13010) 
v_bfe_u32 v85, v76, 8, 8 // (op 13010) 
v_bfe_u32 v86, v76, 16, 8 // (op 13010) 
v_bfe_u32 v87, v76, 24, 8 // (op 13010) 
v_permlane16_swap_b32 v84, v85 // (op 13010) 
s_nop 0
v_permlane16_swap_b32 v86, v87 // (op 13010) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v84, v86 // (op 13010) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v85, v87 // (op 13010) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 15445) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:31], a[0:3], v78, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 76) 
s_lshl_b64 s[84:85], s[30:31], 5 // (op 13940) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[32:35], a[4:7], v78, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4057) 
v_mov_b64 v[18:19], s[84:85] // (op 13940) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:51], v[36:39], a[8:11], v78, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4193) 
v_mov_b64 v[88:89], v[24:25] // (op 13940) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[44:51], v[40:43], a[12:15], v78, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4329) 
v_mov_b64 v[90:91], v[88:89] // (op 13940) 
s_add_u32 s84, 8192, s52 // (op 13940) 
s_barrier  // (op 15544) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:59], v[28:31], a[16:19], v79, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4749) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:59], v[32:35], a[20:23], v79, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4711) 
s_mov_b32 m0, s84 // (op 13940) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:59], v[36:39], a[24:27], v79, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4673) 
s_mov_b32 s85, 256 // (op 13940) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:59], v[40:43], a[28:31], v79, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4635) 
buffer_load_dwordx4 v90, s[68:71], s85 offen offset: 0 lds // (op 13940) Load value direct to lds
v_add_co_u32 v90, s[86:87], v90, v18 // (op 13940) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:67], v[28:31], a[32:35], v80, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5359) 
v_addc_co_u32 v91, s[86:87], 0, v91, s[86:87] // (op 13940) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:67], v[32:35], a[36:39], v80, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5321) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:67], v[36:39], a[40:43], v80, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5283) 
s_add_u32 m0, m0, 4096 // (op 13940) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:67], v[40:43], a[44:47], v80, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5245) 
ds_read_b128 v[92:95], v10 offset:24576 // (op 3958) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:75], v[28:31], a[48:51], v81, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5969) 
buffer_load_dwordx4 v90, s[68:71], s85 offen offset: 0 lds // (op 13940) Load value direct to lds
v_add_co_u32 v90, s[86:87], v90, v18 // (op 13940) least significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:75], v[32:35], a[52:55], v81, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5931) 
v_addc_co_u32 v91, s[86:87], 0, v91, s[86:87] // (op 13940) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:75], v[36:39], a[56:59], v81, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5893) 
s_add_u32 m0, m0, 4096 // (op 13940) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:75], v[40:43], a[60:63], v81, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5855) 
buffer_load_dwordx4 v90, s[68:71], s85 offen offset: 0 lds // (op 13940) Load value direct to lds


ds_read_b128 v[78:81], v10 offset:25600 // (op 4066) Load local data 
v_add_co_u32 v90, s[86:87], v90, v18 // (op 13940) least significant half



v_addc_co_u32 v91, s[86:87], 0, v91, s[86:87] // (op 13940) most significant half

s_add_u32 m0, m0, 4096 // (op 13940) 
ds_read_b128 v[84:87], v10 offset:26624 // (op 4202) Load local data 
buffer_load_dwordx4 v90, s[68:71], s85 offen offset: 0 lds // (op 13940) Load value direct to lds







ds_read_b128 v[96:99], v10 offset:27648 // (op 4338) Load local data 
s_barrier  // (op 15535) 



ds_read_b128 v[44:47], v9 offset:32768 // (op 3956) Load local data 
ds_read_b128 v[48:51], v9 offset:32832 // (op 3956) Load local data 



ds_read_b128 v[52:55], v9 offset:34816 // (op 4643) Load local data 
ds_read_b128 v[56:59], v9 offset:34880 // (op 4643) Load local data 



ds_read_b128 v[60:63], v9 offset:36864 // (op 5253) Load local data 
ds_read_b128 v[64:67], v9 offset:36928 // (op 5253) Load local data 



ds_read_b128 v[100:103], v9 offset:38912 // (op 5863) Load local data 
ds_read_b128 v[104:107], v9 offset:38976 // (op 5863) Load local data 
v_bfe_u32 v108, v23, 0, 8 // (op 12962) 
v_bfe_u32 v109, v23, 8, 8 // (op 12962) 
v_bfe_u32 v110, v23, 16, 8 // (op 12962) 
v_bfe_u32 v111, v23, 24, 8 // (op 12962) 
v_permlane16_swap_b32 v108, v109 // (op 12962) 
s_nop 0
v_permlane16_swap_b32 v110, v111 // (op 12962) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 12962) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 12962) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 15551) 


s_lshl_b64 s[84:85], s[36:37], 6 // (op 13999) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 13999) 

v_lshrrev_b64 v[18:19], 3, s[86:87] // (op 13999) 


v_mov_b64 v[22:23], v[82:83] // (op 13999) 
s_add_u32 s84, 24576, s55 // (op 13999) 
s_mov_b32 m0, s84 // (op 13999) reg expression
s_mov_b32 s85, 192 // (op 13999) call()
v_bfe_u32 v112, v77, 0, 8 // (op 13025) 
v_bfe_u32 v113, v77, 8, 8 // (op 13025) 
v_bfe_u32 v114, v77, 16, 8 // (op 13025) 
v_bfe_u32 v115, v77, 24, 8 // (op 13025) 
v_permlane16_swap_b32 v112, v113 // (op 13025) 
s_nop 0
v_permlane16_swap_b32 v114, v115 // (op 13025) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v112, v114 // (op 13025) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v113, v115 // (op 13025) 
buffer_load_dwordx4 v22, s[72:75], s85 offen offset: 0 lds // (op 13999) Load value direct to lds

v_add_co_u32 v22, s[86:87], v22, v18 // (op 13999) least significant half
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[92:95], a[0:3], v108, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3963) 
v_addc_co_u32 v23, s[86:87], 0, v23, s[86:87] // (op 13999) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[78:81], a[4:7], v108, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4058) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:51], v[84:87], a[8:11], v108, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4194) 
s_add_u32 m0, m0, 4096 // (op 13999) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[44:51], v[96:99], a[12:15], v108, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4330) 
buffer_load_dwordx4 v22, s[72:75], s85 offen offset: 0 lds // (op 13999) Load value direct to lds





s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:59], v[92:95], a[16:19], v109, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4750) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:59], v[78:81], a[20:23], v109, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4712) 
v_add_co_u32 v20, vcc, 128, v20 // (op 14307) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:59], v[84:87], a[24:27], v109, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4674) 
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 14307) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:59], v[96:99], a[28:31], v109, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4636) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 15541) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:67], v[92:95], a[32:35], v110, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5360) 
s_lshl_b64 s[84:85], s[30:31], 5 // (op 13994) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:67], v[78:81], a[36:39], v110, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5322) 
v_mov_b64 v[18:19], s[84:85] // (op 13994) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:67], v[84:87], a[40:43], v110, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5284) 
v_mov_b64 v[22:23], v[88:89] // (op 13994) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:67], v[96:99], a[44:47], v110, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5246) 
s_add_u32 s84, 32768, s54 // (op 13994) 
s_mov_b32 m0, s84 // (op 13994) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[100:107], v[92:95], a[48:51], v111, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5970) 
s_mov_b32 s85, 384 // (op 13994) call()

buffer_load_dwordx4 v22, s[68:71], s85 offen offset: 0 lds // (op 13994) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[100:107], v[78:81], a[52:55], v111, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5932) 
v_add_co_u32 v22, s[86:87], v22, v18 // (op 13994) least significant half

v_addc_co_u32 v23, s[86:87], 0, v23, s[86:87] // (op 13994) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:107], v[84:87], a[56:59], v111, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5894) 

s_add_u32 m0, m0, 4096 // (op 13994) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:107], v[96:99], a[60:63], v111, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5856) 




s_waitcnt vmcnt(7)
s_barrier  // (op 15470) 
buffer_load_dwordx4 v22, s[68:71], s85 offen offset: 0 lds // (op 13994) Load value direct to lds



ds_read_b128 v[28:31], v4 // (op 65) Load local data 
v_add_co_u32 v22, s[86:87], v22, v18 // (op 13994) least significant half



ds_read_b128 v[32:35], v4 offset:1024 // (op 4054) Load local data 
v_addc_co_u32 v23, s[86:87], 0, v23, s[86:87] // (op 13994) most significant half




s_add_u32 m0, m0, 4096 // (op 13994) 
ds_read_b128 v[36:39], v4 offset:2048 // (op 4190) Load local data 
buffer_load_dwordx4 v22, s[68:71], s85 offen offset: 0 lds // (op 13994) Load value direct to lds



v_add_co_u32 v22, s[86:87], v22, v18 // (op 13994) least significant half
v_addc_co_u32 v23, s[86:87], 0, v23, s[86:87] // (op 13994) most significant half

s_add_u32 m0, m0, 4096 // (op 13994) 
ds_read_b128 v[40:43], v4 offset:3072 // (op 4326) Load local data 
buffer_load_dwordx4 v22, s[68:71], s85 offen offset: 0 lds // (op 13994) Load value direct to lds





s_waitcnt vmcnt(6)
s_barrier  // (op 15451) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 60) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 60) Load local data 
v_add_co_u32 v24, vcc, 256, v24 // (op 14179) least significant half



v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 14179) most significant half
ds_read_b128 v[52:55], v3 offset:10240 // (op 4631) Load local data 
ds_read_b128 v[56:59], v3 offset:10304 // (op 4631) Load local data 



ds_read_b128 v[60:63], v3 offset:12288 // (op 5241) Load local data 
ds_read_b128 v[64:67], v3 offset:12352 // (op 5241) Load local data 



ds_read_b128 v[68:71], v3 offset:14336 // (op 5851) Load local data 
ds_read_b128 v[72:75], v3 offset:14400 // (op 5851) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 5740) 
s_cmp_lt_u32 s1, s84 // (op 5740) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_5740 // (op 5740) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_5740 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_5740:










s_barrier  // (op 15832) 
s_and_b32 s52, s7, 4294967294 // (op 14067) 
v_lshrrev_b32 v16, 3, v1 // (op 14067) 
s_add_u32 s53, s0, s52 // (op 14067) 

v_and_b32 v18, -8, v1 // (op 14067) 
v_sub_u32 v19, v1, v18 // (op 14067) 

s_mul_i32 s52, s16, s8 // (op 14067) 
s_add_u32 s52, s52, s53 // (op 14067) 

v_lshlrev_b32 v18, 4, v19 // (op 14067) 

s_mul_hi_u32 s53, s52, s13 // (op 14067) 
s_sub_u32 s54, s52, s53 // (op 14067) 
s_lshr_b32 s55, s54, 1 // (op 14067) 

s_add_u32 s54, s55, s53 // (op 14067) 


s_lshr_b32 s53, s54, s11 // (op 14067) 

s_and_b32 s54, s11, 2147483648 // (op 14067) 
s_cmp_eq_i32 s54, 0 // (op 14067) 
s_mov_b32 s55, scc // (op 14067) 

s_cmp_eq_i32 s55, 0 // (op 14067) 
s_cselect_b32 s54, s52, s53 // (op 14067) 


s_mul_hi_u32 s53, s54, s12 // (op 14067) 
s_sub_u32 s55, s54, s53 // (op 14067) 
s_lshr_b32 s76, s55, 1 // (op 14067) 

s_add_u32 s55, s76, s53 // (op 14067) 


s_lshr_b32 s53, s55, s10 // (op 14067) 

s_and_b32 s55, s10, 2147483648 // (op 14067) 
s_cmp_eq_i32 s55, 0 // (op 14067) 
s_mov_b32 s76, scc // (op 14067) 

s_cmp_eq_i32 s76, 0 // (op 14067) 
s_cselect_b32 s55, s54, s53 // (op 14067) 


v_lshl_add_u32 v19, s55, 7, v16 // (op 14067) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 14067) 
s_mul_i32 s53, s54, s76 // (op 14067) 


s_sub_u32 s54, s52, s53 // (op 14067) 


v_lshl_add_u32 v16, s54, 7, v18 // (op 14067) 


v_mov_b32 v21, 0 // (op 14067) convert
v_mov_b32 v20, v16 // (op 14067) convert

v_mul_lo_u32 v23, v19, s31 // (op 14067) most significant: low of low * high
v_mul_hi_u32 v16, v19, s30 // (op 14067) most significant: high of low * low
v_mul_lo_u32 v22, v19, s30 // (op 14067) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 14067) most significant: sum

v_add_co_u32 v22, s[52:53], v22, v20 // (op 14067) least significant half
v_addc_co_u32 v23, s[52:53], v23, v21, s[52:53] // (op 14067) most significant half



s_mul_i32 s53, s30, s35 // (op 14069) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 14069) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 14069) least significant
s_mul_hi_u32 s55, s30, s34 // (op 14069) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14069) most significant: sum
s_add_u32 s53, s53, s55 // (op 14069) most significant: sum


s_mov_b32 s68, s32 // (op 14069) reg expression
s_mov_b32 s69, s33 // (op 14069) reg expression
s_mov_b32 s70, s52 // (op 14069) reg expression
s_mov_b32 s71, 131072 // (op 14069) call()

s_and_b32 s52, s7, 4294967294 // (op 14231) 
v_lshrrev_b32 v16, 2, v1 // (op 14231) 
s_add_u32 s53, s0, s52 // (op 14231) 

v_and_b32 v18, -4, v1 // (op 14231) 
v_sub_u32 v19, v1, v18 // (op 14231) 

s_mul_i32 s52, s16, s8 // (op 14231) 
s_add_u32 s52, s52, s53 // (op 14231) 

v_lshlrev_b32 v18, 5, v19 // (op 14231) 

s_mul_hi_u32 s53, s52, s13 // (op 14231) 
s_sub_u32 s54, s52, s53 // (op 14231) 
s_lshr_b32 s55, s54, 1 // (op 14231) 

s_add_u32 s54, s55, s53 // (op 14231) 


s_lshr_b32 s53, s54, s11 // (op 14231) 

s_and_b32 s54, s11, 2147483648 // (op 14231) 
s_cmp_eq_i32 s54, 0 // (op 14231) 
s_mov_b32 s55, scc // (op 14231) 

s_cmp_eq_i32 s55, 0 // (op 14231) 
s_cselect_b32 s54, s52, s53 // (op 14231) 


s_mul_hi_u32 s53, s54, s12 // (op 14231) 
s_sub_u32 s55, s54, s53 // (op 14231) 
s_lshr_b32 s76, s55, 1 // (op 14231) 

s_add_u32 s55, s76, s53 // (op 14231) 


s_lshr_b32 s53, s55, s10 // (op 14231) 

s_and_b32 s55, s10, 2147483648 // (op 14231) 
s_cmp_eq_i32 s55, 0 // (op 14231) 
s_mov_b32 s76, scc // (op 14231) 

s_cmp_eq_i32 s76, 0 // (op 14231) 
s_cselect_b32 s55, s54, s53 // (op 14231) 


s_mul_i32 s53, s55, s14 // (op 14231) 

s_sub_u32 s55, s54, s53 // (op 14231) 

v_lshl_add_u32 v19, s55, 7, v16 // (op 14231) 


v_mul_lo_u32 v21, v19, s37 // (op 14231) most significant: low of low * high
v_mul_hi_u32 v16, v19, s36 // (op 14231) most significant: high of low * low
v_mul_lo_u32 v20, v19, s36 // (op 14231) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 14231) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 14231) 
s_mul_i32 s53, s54, s76 // (op 14231) 


s_sub_u32 s54, s52, s53 // (op 14231) 


v_lshl_add_u32 v16, s54, 7, v18 // (op 14231) 


v_mov_b32 v19, 0 // (op 14231) convert
v_mov_b32 v18, v16 // (op 14231) convert

v_add_co_u32 v24, s[52:53], v18, v20 // (op 14231) least significant half
v_addc_co_u32 v25, s[52:53], v19, v21, s[52:53] // (op 14231) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 14231) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 14231) 

s_mul_i32 s53, s36, s23 // (op 14233) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 14233) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 14233) least significant
s_mul_hi_u32 s55, s36, s22 // (op 14233) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 14233) most significant: sum
s_add_u32 s53, s53, s55 // (op 14233) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 14233) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 14233) 

s_mov_b32 s72, s38 // (op 14233) reg expression
s_mov_b32 s73, s39 // (op 14233) reg expression
s_mov_b32 s74, s52 // (op 14233) reg expression
s_mov_b32 s75, 131072 // (op 14233) call()

v_and_b32 v16, -8, v1 // (op 14387) 
v_sub_u32 v18, v1, v16 // (op 14387) 

v_lshlrev_b32 v16, 4, v18 // (op 14387) 

v_lshrrev_b32 v18, 3, v1 // (op 14387) 
v_lshl_add_u32 v19, v18, 7, v16 // (op 14387) 


s_nop 0
v_readfirstlane_b32 s52, v19 // (op 14387) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 14464) 
v_sub_u32 v18, v1, v16 // (op 14464) 

v_lshlrev_b32 v16, 5, v18 // (op 14464) 

v_lshrrev_b32 v18, 2, v1 // (op 14464) 
v_lshl_add_u32 v19, v18, 7, v16 // (op 14464) 


v_lshlrev_b32 v16, 2, v19 // (op 14464) 

v_lshrrev_b32 v18, 3, v16 // (op 14464) 

s_nop 0
v_readfirstlane_b32 s53, v18 // (op 14464) Wait state hazard: VALU Write Readlane Hazard

s_and_b32 s54, s7, 4294967294 // (op 15187) 
v_and_b32 v16, 63, v1 // (op 15187) 
s_add_u32 s55, s0, s54 // (op 15187) 

v_lshrrev_b32 v18, 4, v16 // (op 15187) 
v_and_b32 v19, 3, v18 // (op 15187) 

s_mul_i32 s54, s16, s8 // (op 15187) 
s_add_u32 s54, s54, s55 // (op 15187) 

v_and_b32 v18, 15, v1 // (op 15187) 
s_mul_hi_u32 s55, s54, s13 // (op 15187) 
v_lshl_add_u32 v20, v19, 4, v18 // (op 15187) 


s_sub_u32 s76, s54, s55 // (op 15187) 
v_lshrrev_b32 v21, 7, v1 // (op 15187) 
s_lshr_b32 s77, s76, 1 // (op 15187) 

v_lshl_add_u32 v18, v21, 6, v20 // (op 15187) 


s_add_u32 s76, s77, s55 // (op 15187) 


v_lshrrev_b32 v19, 6, v16 // (op 15187) 

s_lshr_b32 s55, s76, s11 // (op 15187) 

v_lshlrev_b32 v16, 2, v19 // (op 15187) 

s_and_b32 s76, s11, 2147483648 // (op 15187) 
s_cmp_eq_i32 s76, 0 // (op 15187) 
s_mov_b32 s77, scc // (op 15187) 

s_cmp_eq_i32 s77, 0 // (op 15187) 
s_cselect_b32 s76, s54, s55 // (op 15187) 


s_mul_hi_u32 s55, s76, s12 // (op 15187) 
s_sub_u32 s77, s76, s55 // (op 15187) 
s_lshr_b32 s78, s77, 1 // (op 15187) 

s_add_u32 s77, s78, s55 // (op 15187) 


s_lshr_b32 s55, s77, s10 // (op 15187) 

s_and_b32 s77, s10, 2147483648 // (op 15187) 
s_cmp_eq_i32 s77, 0 // (op 15187) 
s_mov_b32 s78, scc // (op 15187) 

s_cmp_eq_i32 s78, 0 // (op 15187) 
s_cselect_b32 s77, s76, s55 // (op 15187) 


v_lshl_add_u32 v19, s77, 7, v18 // (op 15187) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 15187) 
s_mul_i32 s55, s76, s78 // (op 15187) 


s_sub_u32 s76, s54, s55 // (op 15187) 


v_lshl_add_u32 v18, s76, 2, v16 // (op 15187) 


v_mul_lo_u32 v16, v19, s40 // (op 15187) 
v_add_u32 v16, v16, v18 // (op 15187) 


s_mul_i32 s55, s40, s35 // (op 15189) most significant: low of low * high
s_mul_i32 s76, s41, s34 // (op 15189) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 15189) least significant
s_mul_hi_u32 s77, s40, s34 // (op 15189) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 15189) most significant: sum
s_add_u32 s55, s55, s77 // (op 15189) most significant: sum


s_mov_b32 s76, s42 // (op 15189) reg expression
s_mov_b32 s77, s43 // (op 15189) reg expression
s_mov_b32 s78, s54 // (op 15189) reg expression
s_mov_b32 s79, 131072 // (op 15189) call()

s_and_b32 s54, s7, 4294967294 // (op 15214) 
v_and_b32 v18, 63, v1 // (op 15214) 
s_add_u32 s55, s0, s54 // (op 15214) 

v_lshrrev_b32 v19, 4, v18 // (op 15214) 
v_and_b32 v20, 3, v19 // (op 15214) 

s_mul_i32 s54, s16, s8 // (op 15214) 
s_add_u32 s54, s54, s55 // (op 15214) 

v_and_b32 v19, 15, v1 // (op 15214) 
s_mul_hi_u32 s55, s54, s13 // (op 15214) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 15214) 


s_sub_u32 s80, s54, s55 // (op 15214) 
v_lshrrev_b32 v19, 6, v1 // (op 15214) 
s_lshr_b32 s81, s80, 1 // (op 15214) 

v_and_b32 v20, 1, v19 // (op 15214) 

s_add_u32 s80, s81, s55 // (op 15214) 


v_lshl_add_u32 v19, v20, 6, v21 // (op 15214) 


s_lshr_b32 s55, s80, s11 // (op 15214) 

v_lshrrev_b32 v20, 6, v18 // (op 15214) 

s_and_b32 s80, s11, 2147483648 // (op 15214) 
v_lshlrev_b32 v18, 2, v20 // (op 15214) 

s_cmp_eq_i32 s80, 0 // (op 15214) 
s_mov_b32 s81, scc // (op 15214) 

s_cmp_eq_i32 s81, 0 // (op 15214) 
s_cselect_b32 s80, s54, s55 // (op 15214) 


s_mul_hi_u32 s55, s80, s12 // (op 15214) 
s_sub_u32 s81, s80, s55 // (op 15214) 
s_lshr_b32 s82, s81, 1 // (op 15214) 

s_add_u32 s81, s82, s55 // (op 15214) 


s_lshr_b32 s55, s81, s10 // (op 15214) 

s_and_b32 s81, s10, 2147483648 // (op 15214) 
s_cmp_eq_i32 s81, 0 // (op 15214) 
s_mov_b32 s82, scc // (op 15214) 

s_cmp_eq_i32 s82, 0 // (op 15214) 
s_cselect_b32 s81, s80, s55 // (op 15214) 


s_mul_i32 s55, s81, s14 // (op 15214) 

s_sub_u32 s81, s80, s55 // (op 15214) 

v_lshl_add_u32 v20, s81, 7, v19 // (op 15214) 


v_mul_lo_u32 v19, v20, s44 // (op 15214) 

s_ashr_i64 s[82:83], s[2:3], 7 // (op 15214) 
s_mul_i32 s55, s80, s82 // (op 15214) 


s_sub_u32 s80, s54, s55 // (op 15214) 


v_lshl_add_u32 v20, s80, 2, v18 // (op 15214) 


v_add_u32 v18, v20, v19 // (op 15214) 


s_mul_i32 s55, s44, s23 // (op 15216) most significant: low of low * high
s_mul_i32 s80, s45, s22 // (op 15216) most significant: low of high * low
s_mul_i32 s54, s44, s22 // (op 15216) least significant
s_mul_hi_u32 s81, s44, s22 // (op 15216) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 15216) most significant: sum
s_add_u32 s55, s55, s81 // (op 15216) most significant: sum


s_mov_b32 s80, s46 // (op 15216) reg expression
s_mov_b32 s81, s47 // (op 15216) reg expression
s_mov_b32 s82, s54 // (op 15216) reg expression
s_mov_b32 s83, 131072 // (op 15216) call()

s_and_b32 s54, s7, 4294967294 // (op 3899) 
s_cmp_lt_u32 s54, s7 // (op 3898) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_3898 // (op 3898) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_3898 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_3898:
s_barrier  // (op 15448) 
s_add_u32 s54, s54, 1 // (op 3900) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 13982) 
v_mov_b64 v[20:21], s[84:85] // (op 13982) convert


v_mov_b64 v[24:25], v[22:23] // (op 13982) 
v_mov_b64 v[28:29], v[24:25] // (op 13982) 
s_add_u32 s55, 8192, s52 // (op 13982) 
s_mov_b32 m0, s55 // (op 13982) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 13982) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v20 // (op 13982) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 13982) most significant half

s_add_u32 m0, m0, 4096 // (op 13982) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 13982) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v20 // (op 13982) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 13982) most significant half

s_add_u32 m0, m0, 4096 // (op 13982) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 13982) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v20 // (op 13982) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 13982) most significant half

s_add_u32 m0, m0, 4096 // (op 13982) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 13982) Load value direct to lds




v_add_co_u32 v22, vcc, 128, v22 // (op 14057) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 14057) most significant half
s_barrier  // (op 15467) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 13988) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 13988) 

v_lshrrev_b64 v[20:21], 3, s[86:87] // (op 13988) 


v_mov_b64 v[24:25], v[26:27] // (op 13988) 
v_mov_b64 v[28:29], v[24:25] // (op 13988) 
s_mov_b32 m0, s53 // (op 13988) reg expression
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 13988) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v20 // (op 13988) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 13988) most significant half

s_add_u32 m0, m0, 4096 // (op 13988) 
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 13988) Load value direct to lds



v_add_co_u32 v26, s[84:85], 64, v26 // (op 14223) least significant half
v_addc_co_u32 v27, s[84:85], 0, v27, s[84:85] // (op 14223) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 15454) 



ds_read_b128 v[28:31], v3 offset:8192 // (op 3906) Load local data 
ds_read_b128 v[32:35], v3 offset:8256 // (op 3906) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 15473) 



ds_read_b128 v[36:39], v4 // (op 3908) Load local data 


v_mov_b32 v19, v16 // (op 3910) 
v_mov_b32 v20, v19 // (op 3910) reg expression

buffer_load_dwordx2 v[24:25], v20, s[76:79], 0 offen // (op 3910) Load value


v_add_u32 v16, 4, v16 // (op 15177) 


v_mov_b32 v19, v18 // (op 3912) 
v_mov_b32 v20, v19 // (op 3912) reg expression

buffer_load_dwordx2 v[40:41], v20, s[80:83], 0 offen // (op 3912) Load value


v_add_u32 v18, 4, v18 // (op 15204) 
s_waitcnt vmcnt(1)
v_bfe_u32 v42, v24, 0, 8 // (op 13074) 



v_bfe_u32 v43, v24, 8, 8 // (op 13074) 
v_bfe_u32 v44, v24, 16, 8 // (op 13074) 
v_bfe_u32 v45, v24, 24, 8 // (op 13074) 
v_permlane16_swap_b32 v42, v43 // (op 13074) 
s_nop 0
v_permlane16_swap_b32 v44, v45 // (op 13074) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v42, v44 // (op 13074) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v43, v45 // (op 13074) 
ds_read_b128 v[46:49], v4 offset:1024 // (op 4037) Load local data 




ds_read_b128 v[50:53], v4 offset:2048 // (op 4173) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v54, v40, 0, 8 // (op 13104) 



v_bfe_u32 v55, v40, 8, 8 // (op 13104) 
v_bfe_u32 v56, v40, 16, 8 // (op 13104) 
v_bfe_u32 v57, v40, 24, 8 // (op 13104) 
v_permlane16_swap_b32 v54, v55 // (op 13104) 
ds_read_b128 v[58:61], v4 offset:3072 // (op 4309) Load local data 
v_permlane16_swap_b32 v56, v57 // (op 13104) 



s_nop 1
v_permlane32_swap_b32 v54, v56 // (op 13104) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v55, v57 // (op 13104) 
ds_read_b128 v[62:65], v3 offset:10240 // (op 4531) Load local data 

ds_read_b128 v[66:69], v3 offset:10304 // (op 4531) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:39], a[0:3], v42, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 3913) 
ds_read_b128 v[70:73], v3 offset:12288 // (op 5141) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:35], v[46:49], a[4:7], v42, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4040) 
ds_read_b128 v[74:77], v3 offset:12352 // (op 5141) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:35], v[50:53], a[8:11], v42, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4176) 
ds_read_b128 v[78:81], v3 offset:14336 // (op 5751) Load local data 
ds_read_b128 v[82:85], v3 offset:14400 // (op 5751) Load local data 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[28:35], v[58:61], a[12:15], v42, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4312) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:69], v[36:39], a[16:19], v43, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4535) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[62:69], v[50:53], a[24:27], v43, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4589) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[62:69], v[58:61], a[28:31], v43, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:69], v[46:49], a[20:23], v43, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 4562) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:77], v[36:39], a[32:35], v44, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5145) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[70:77], v[50:53], a[40:43], v44, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5199) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[70:77], v[58:61], a[44:47], v44, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5226) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:77], v[46:49], a[36:39], v44, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5172) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[78:85], v[36:39], a[48:51], v45, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5755) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[78:85], v[50:53], a[56:59], v45, v56 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5809) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[78:85], v[58:61], a[60:63], v45, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5836) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[78:85], v[46:49], a[52:55], v45, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 5782) 




s_cmp_lt_u32 s54, s7 // (op 3898) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_3898 // (op 3898) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_3898 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_3898:









s_barrier  // (op 15804) 


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 257) 
s_cmp_lg_u32 s18, 0 // (op 283) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_283 // (op 283) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_283



v_add_u32 v13, 49152, v12 // (op 6721) 
buffer_store_dword a48, v13, s[64:67], 0 offen sc1 // (op 6721) Store value

v_add_u32 v13, 49664, v12 // (op 6721) 
buffer_store_dword a49, v13, s[64:67], 0 offen sc1 // (op 6721) Store value

v_add_u32 v13, 50176, v12 // (op 6721) 
buffer_store_dword a50, v13, s[64:67], 0 offen sc1 // (op 6721) Store value

v_add_u32 v13, 50688, v12 // (op 6721) 
buffer_store_dword a51, v13, s[64:67], 0 offen sc1 // (op 6721) Store value




v_add_u32 v13, 49280, v12 // (op 6723) 
buffer_store_dword a52, v13, s[64:67], 0 offen sc1 // (op 6723) Store value

v_add_u32 v13, 49792, v12 // (op 6723) 
buffer_store_dword a53, v13, s[64:67], 0 offen sc1 // (op 6723) Store value

v_add_u32 v13, 50304, v12 // (op 6723) 
buffer_store_dword a54, v13, s[64:67], 0 offen sc1 // (op 6723) Store value

v_add_u32 v13, 50816, v12 // (op 6723) 
buffer_store_dword a55, v13, s[64:67], 0 offen sc1 // (op 6723) Store value




v_add_u32 v13, 49408, v12 // (op 6725) 
buffer_store_dword a56, v13, s[64:67], 0 offen sc1 // (op 6725) Store value

v_add_u32 v13, 49920, v12 // (op 6725) 
buffer_store_dword a57, v13, s[64:67], 0 offen sc1 // (op 6725) Store value

v_add_u32 v13, 50432, v12 // (op 6725) 
buffer_store_dword a58, v13, s[64:67], 0 offen sc1 // (op 6725) Store value

v_add_u32 v13, 50944, v12 // (op 6725) 
buffer_store_dword a59, v13, s[64:67], 0 offen sc1 // (op 6725) Store value




v_add_u32 v13, 49536, v12 // (op 6727) 
buffer_store_dword a60, v13, s[64:67], 0 offen sc1 // (op 6727) Store value

v_add_u32 v13, 50048, v12 // (op 6727) 
buffer_store_dword a61, v13, s[64:67], 0 offen sc1 // (op 6727) Store value

v_add_u32 v13, 50560, v12 // (op 6727) 
buffer_store_dword a62, v13, s[64:67], 0 offen sc1 // (op 6727) Store value

v_add_u32 v13, 51072, v12 // (op 6727) 
buffer_store_dword a63, v13, s[64:67], 0 offen sc1 // (op 6727) Store value




v_add_u32 v13, 32768, v12 // (op 6697) 
buffer_store_dword a32, v13, s[64:67], 0 offen sc1 // (op 6697) Store value

v_add_u32 v13, 33280, v12 // (op 6697) 
buffer_store_dword a33, v13, s[64:67], 0 offen sc1 // (op 6697) Store value

v_add_u32 v13, 33792, v12 // (op 6697) 
buffer_store_dword a34, v13, s[64:67], 0 offen sc1 // (op 6697) Store value

v_add_u32 v13, 34304, v12 // (op 6697) 
buffer_store_dword a35, v13, s[64:67], 0 offen sc1 // (op 6697) Store value




v_add_u32 v13, 32896, v12 // (op 6699) 
buffer_store_dword a36, v13, s[64:67], 0 offen sc1 // (op 6699) Store value

v_add_u32 v13, 33408, v12 // (op 6699) 
buffer_store_dword a37, v13, s[64:67], 0 offen sc1 // (op 6699) Store value

v_add_u32 v13, 33920, v12 // (op 6699) 
buffer_store_dword a38, v13, s[64:67], 0 offen sc1 // (op 6699) Store value

v_add_u32 v13, 34432, v12 // (op 6699) 
buffer_store_dword a39, v13, s[64:67], 0 offen sc1 // (op 6699) Store value




v_add_u32 v13, 33024, v12 // (op 6701) 
buffer_store_dword a40, v13, s[64:67], 0 offen sc1 // (op 6701) Store value

v_add_u32 v13, 33536, v12 // (op 6701) 
buffer_store_dword a41, v13, s[64:67], 0 offen sc1 // (op 6701) Store value

v_add_u32 v13, 34048, v12 // (op 6701) 
buffer_store_dword a42, v13, s[64:67], 0 offen sc1 // (op 6701) Store value

v_add_u32 v13, 34560, v12 // (op 6701) 
buffer_store_dword a43, v13, s[64:67], 0 offen sc1 // (op 6701) Store value




v_add_u32 v13, 33152, v12 // (op 6703) 
buffer_store_dword a44, v13, s[64:67], 0 offen sc1 // (op 6703) Store value

v_add_u32 v13, 33664, v12 // (op 6703) 
buffer_store_dword a45, v13, s[64:67], 0 offen sc1 // (op 6703) Store value

v_add_u32 v13, 34176, v12 // (op 6703) 
buffer_store_dword a46, v13, s[64:67], 0 offen sc1 // (op 6703) Store value

v_add_u32 v13, 34688, v12 // (op 6703) 
buffer_store_dword a47, v13, s[64:67], 0 offen sc1 // (op 6703) Store value




v_add_u32 v13, 16384, v12 // (op 6673) 
buffer_store_dword a16, v13, s[64:67], 0 offen sc1 // (op 6673) Store value

v_add_u32 v13, 16896, v12 // (op 6673) 
buffer_store_dword a17, v13, s[64:67], 0 offen sc1 // (op 6673) Store value

v_add_u32 v13, 17408, v12 // (op 6673) 
buffer_store_dword a18, v13, s[64:67], 0 offen sc1 // (op 6673) Store value

v_add_u32 v13, 17920, v12 // (op 6673) 
buffer_store_dword a19, v13, s[64:67], 0 offen sc1 // (op 6673) Store value




v_add_u32 v13, 16512, v12 // (op 6675) 
buffer_store_dword a20, v13, s[64:67], 0 offen sc1 // (op 6675) Store value

v_add_u32 v13, 17024, v12 // (op 6675) 
buffer_store_dword a21, v13, s[64:67], 0 offen sc1 // (op 6675) Store value

v_add_u32 v13, 17536, v12 // (op 6675) 
buffer_store_dword a22, v13, s[64:67], 0 offen sc1 // (op 6675) Store value

v_add_u32 v13, 18048, v12 // (op 6675) 
buffer_store_dword a23, v13, s[64:67], 0 offen sc1 // (op 6675) Store value




v_add_u32 v13, 16640, v12 // (op 6677) 
buffer_store_dword a24, v13, s[64:67], 0 offen sc1 // (op 6677) Store value

v_add_u32 v13, 17152, v12 // (op 6677) 
buffer_store_dword a25, v13, s[64:67], 0 offen sc1 // (op 6677) Store value

v_add_u32 v13, 17664, v12 // (op 6677) 
buffer_store_dword a26, v13, s[64:67], 0 offen sc1 // (op 6677) Store value

v_add_u32 v13, 18176, v12 // (op 6677) 
buffer_store_dword a27, v13, s[64:67], 0 offen sc1 // (op 6677) Store value




v_add_u32 v13, 16768, v12 // (op 6679) 
buffer_store_dword a28, v13, s[64:67], 0 offen sc1 // (op 6679) Store value

v_add_u32 v13, 17280, v12 // (op 6679) 
buffer_store_dword a29, v13, s[64:67], 0 offen sc1 // (op 6679) Store value

v_add_u32 v13, 17792, v12 // (op 6679) 
buffer_store_dword a30, v13, s[64:67], 0 offen sc1 // (op 6679) Store value

v_add_u32 v13, 18304, v12 // (op 6679) 
buffer_store_dword a31, v13, s[64:67], 0 offen sc1 // (op 6679) Store value




buffer_store_dword a0, v12, s[64:67], 0 offen sc1 // (op 280) Store value
buffer_store_dword a1, v12, s[64:67], 0 offen offset:512 sc1 // (op 280) Store value
buffer_store_dword a2, v12, s[64:67], 0 offen offset:1024 sc1 // (op 280) Store value
buffer_store_dword a3, v12, s[64:67], 0 offen offset:1536 sc1 // (op 280) Store value



buffer_store_dword a4, v12, s[64:67], 0 offen offset:128 sc1 // (op 6647) Store value
buffer_store_dword a5, v12, s[64:67], 0 offen offset:640 sc1 // (op 6647) Store value
buffer_store_dword a6, v12, s[64:67], 0 offen offset:1152 sc1 // (op 6647) Store value
buffer_store_dword a7, v12, s[64:67], 0 offen offset:1664 sc1 // (op 6647) Store value



buffer_store_dword a8, v12, s[64:67], 0 offen offset:256 sc1 // (op 6648) Store value
buffer_store_dword a9, v12, s[64:67], 0 offen offset:768 sc1 // (op 6648) Store value
buffer_store_dword a10, v12, s[64:67], 0 offen offset:1280 sc1 // (op 6648) Store value
buffer_store_dword a11, v12, s[64:67], 0 offen offset:1792 sc1 // (op 6648) Store value



buffer_store_dword a12, v12, s[64:67], 0 offen offset:384 sc1 // (op 6649) Store value
buffer_store_dword a13, v12, s[64:67], 0 offen offset:896 sc1 // (op 6649) Store value
buffer_store_dword a14, v12, s[64:67], 0 offen offset:1408 sc1 // (op 6649) Store value
buffer_store_dword a15, v12, s[64:67], 0 offen offset:1920 sc1 // (op 6649) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 284) 
v_cmp_eq_i32 vcc, v1, 0 // (op 288) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_288 // (op 288) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_288
s_mov_b32 s52, 1 // (op 286) call()
s_mov_b32 s55, 0 // (op 287) convert
s_mov_b32 s54, s16 // (op 287) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 287) 
s_add_u32 s76, s48, s54 // (op 287) Start of Int64 add, locking SCC
s_addc_u32 s77, s49, s55 // (op 287) End of Int64 add, unlocking SCC
s_store_dword s52, s[76:77], 0 glc // (op 287) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_288 // (op 288) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_288
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_288:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_288:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_283 // (op 283) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_283
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Send_Tile_283:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Send_Tile_283:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 299) 
s_mov_b32 s52, scc // (op 299) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 299) 
s_mov_b32 s53, s54 // (op 299) convert

s_sub_u32 s54, s53, 1 // (op 299) 

s_cmp_lt_u32 s19, s54 // (op 299) 
s_mov_b32 s53, scc // (op 299) 

s_and_b32 s54, s52, s53 // (op 299) 
s_cmp_lg_u32 s54, 0 // (op 299) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_299 // (op 299) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_299
s_mov_b32 s52, 0 // (op 265) call()
s_mul_i32 s53, s52, s8 // (op 264) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 264) 
s_mov_b32 s76, s54 // (op 264) convert

s_sub_u32 s54, s76, s19 // (op 264) 

s_sub_u32 s55, s54, 1 // (op 264) 

s_cmp_lt_u32 s53, s55 // (op 264) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_264 // (op 264) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_264 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_264:
s_add_u32 s53, s16, 1 // (op 14778) 
v_lshrrev_b32 v12, 5, v1 // (op 14778) 
s_add_u32 s54, s53, s52 // (op 14778) 
s_lshl_b32 s54, s54, 7 // (op 14778) 

v_lshlrev_b32 v13, 2, v12 // (op 14778) 

v_add_lshl_u32 v12, s54, v13, 7 // (op 14778) 


v_and_b32 v13, -32, v1 // (op 14778) 
v_sub_u32 v15, v1, v13 // (op 14778) 

v_add_lshl_u32 v13, v12, v15, 2 // (op 14778) 


s_lshl_b32 s53, s15, 16 // (op 14780) 
s_mov_b32 s64, s28 // (op 14780) reg expression
s_mov_b32 s65, s29 // (op 14780) reg expression
s_mov_b32 s66, s53 // (op 14780) reg expression
s_mov_b32 s67, 131072 // (op 14780) call()

RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_303:
s_add_u32 s53, s16, 1 // (op 301) 
s_add_u32 s54, s53, s52 // (op 301) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 301) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 301) 

s_add_u32 s76, s48, s54 // (op 301) Start of Int64 add, locking SCC
s_addc_u32 s77, s49, s55 // (op 301) End of Int64 add, unlocking SCC
s_load_dword s53, s[76:77], 0 glc // (op 301) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 303) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_303 // (op 303) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_303 if true)
s_barrier  // (op 307) 
v_cmp_eq_i32 vcc, v1, 0 // (op 306) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_306 // (op 306) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_306
s_mov_b32 s53, 0 // (op 304) call()
s_add_u32 s54, s16, 1 // (op 305) 
s_add_u32 s76, s54, s52 // (op 305) Start of Int64 add, locking SCC
s_addc_u32 s77, 0, 0 // (op 305) End of Int64 add, unlocking SCC
s_lshl_b64 s[76:77], s[76:77], 2 // (op 305) 

s_add_u32 s54, s48, s76 // (op 305) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s77 // (op 305) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 305) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_306 // (op 306) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_306
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_306:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_306:




buffer_load_dword v18, v13, s[64:67], 0 offen // (op 318) Load value
buffer_load_dword v19, v13, s[64:67], 0 offen offset:512 // (op 318) Load value
buffer_load_dword v20, v13, s[64:67], 0 offen offset:1024 // (op 318) Load value
buffer_load_dword v21, v13, s[64:67], 0 offen offset:1536 // (op 318) Load value
v_accvgpr_read v22, a0 // (op 321) reg expression
v_accvgpr_read v23, a1 // (op 321) reg expression
v_accvgpr_read v24, a2 // (op 321) reg expression
v_accvgpr_read v25, a3 // (op 321) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 323) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 323) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 323) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 323) 

v_accvgpr_write a0, v22 // (op 325) reg expression
v_accvgpr_write a1, v23 // (op 325) reg expression
v_accvgpr_write a2, v24 // (op 325) reg expression
v_accvgpr_write a3, v25 // (op 325) reg expression




buffer_load_dword v18, v13, s[64:67], 0 offen offset:128 // (op 6796) Load value
buffer_load_dword v19, v13, s[64:67], 0 offen offset:640 // (op 6796) Load value
buffer_load_dword v20, v13, s[64:67], 0 offen offset:1152 // (op 6796) Load value
buffer_load_dword v21, v13, s[64:67], 0 offen offset:1664 // (op 6796) Load value
v_accvgpr_read v22, a4 // (op 6797) reg expression
v_accvgpr_read v23, a5 // (op 6797) reg expression
v_accvgpr_read v24, a6 // (op 6797) reg expression
v_accvgpr_read v25, a7 // (op 6797) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6798) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6798) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6798) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6798) 

v_accvgpr_write a4, v22 // (op 6799) reg expression
v_accvgpr_write a5, v23 // (op 6799) reg expression
v_accvgpr_write a6, v24 // (op 6799) reg expression
v_accvgpr_write a7, v25 // (op 6799) reg expression




buffer_load_dword v18, v13, s[64:67], 0 offen offset:256 // (op 6803) Load value
buffer_load_dword v19, v13, s[64:67], 0 offen offset:768 // (op 6803) Load value
buffer_load_dword v20, v13, s[64:67], 0 offen offset:1280 // (op 6803) Load value
buffer_load_dword v21, v13, s[64:67], 0 offen offset:1792 // (op 6803) Load value
v_accvgpr_read v22, a8 // (op 6804) reg expression
v_accvgpr_read v23, a9 // (op 6804) reg expression
v_accvgpr_read v24, a10 // (op 6804) reg expression
v_accvgpr_read v25, a11 // (op 6804) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6805) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6805) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6805) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6805) 

v_accvgpr_write a8, v22 // (op 6806) reg expression
v_accvgpr_write a9, v23 // (op 6806) reg expression
v_accvgpr_write a10, v24 // (op 6806) reg expression
v_accvgpr_write a11, v25 // (op 6806) reg expression




buffer_load_dword v18, v13, s[64:67], 0 offen offset:384 // (op 6810) Load value
buffer_load_dword v19, v13, s[64:67], 0 offen offset:896 // (op 6810) Load value
buffer_load_dword v20, v13, s[64:67], 0 offen offset:1408 // (op 6810) Load value
buffer_load_dword v21, v13, s[64:67], 0 offen offset:1920 // (op 6810) Load value
v_accvgpr_read v22, a12 // (op 6811) reg expression
v_accvgpr_read v23, a13 // (op 6811) reg expression
v_accvgpr_read v24, a14 // (op 6811) reg expression
v_accvgpr_read v25, a15 // (op 6811) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6812) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6812) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6812) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6812) 

v_accvgpr_write a12, v22 // (op 6813) reg expression
v_accvgpr_write a13, v23 // (op 6813) reg expression
v_accvgpr_write a14, v24 // (op 6813) reg expression
v_accvgpr_write a15, v25 // (op 6813) reg expression




v_mov_b32 v12, 16384 // (op 6847) 
v_add_i32 v15, v13, v12 // (op 6847) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6847) Load value

v_mov_b32 v12, 16896 // (op 6847) 
v_add_i32 v15, v13, v12 // (op 6847) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6847) Load value

v_mov_b32 v12, 17408 // (op 6847) 
v_add_i32 v15, v13, v12 // (op 6847) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6847) Load value

v_mov_b32 v12, 17920 // (op 6847) 
v_add_i32 v15, v13, v12 // (op 6847) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6847) Load value

v_accvgpr_read v22, a16 // (op 6844) reg expression
v_accvgpr_read v23, a17 // (op 6844) reg expression
v_accvgpr_read v24, a18 // (op 6844) reg expression
v_accvgpr_read v25, a19 // (op 6844) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6845) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6845) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6845) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6845) 

v_accvgpr_write a16, v22 // (op 6846) reg expression
v_accvgpr_write a17, v23 // (op 6846) reg expression
v_accvgpr_write a18, v24 // (op 6846) reg expression
v_accvgpr_write a19, v25 // (op 6846) reg expression




v_mov_b32 v12, 16512 // (op 6852) 
v_add_i32 v15, v13, v12 // (op 6852) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6852) Load value

v_mov_b32 v12, 17024 // (op 6852) 
v_add_i32 v15, v13, v12 // (op 6852) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6852) Load value

v_mov_b32 v12, 17536 // (op 6852) 
v_add_i32 v15, v13, v12 // (op 6852) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6852) Load value

v_mov_b32 v12, 18048 // (op 6852) 
v_add_i32 v15, v13, v12 // (op 6852) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6852) Load value

v_accvgpr_read v22, a20 // (op 6849) reg expression
v_accvgpr_read v23, a21 // (op 6849) reg expression
v_accvgpr_read v24, a22 // (op 6849) reg expression
v_accvgpr_read v25, a23 // (op 6849) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6850) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6850) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6850) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6850) 

v_accvgpr_write a20, v22 // (op 6851) reg expression
v_accvgpr_write a21, v23 // (op 6851) reg expression
v_accvgpr_write a22, v24 // (op 6851) reg expression
v_accvgpr_write a23, v25 // (op 6851) reg expression




v_mov_b32 v12, 16640 // (op 6857) 
v_add_i32 v15, v13, v12 // (op 6857) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6857) Load value

v_mov_b32 v12, 17152 // (op 6857) 
v_add_i32 v15, v13, v12 // (op 6857) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6857) Load value

v_mov_b32 v12, 17664 // (op 6857) 
v_add_i32 v15, v13, v12 // (op 6857) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6857) Load value

v_mov_b32 v12, 18176 // (op 6857) 
v_add_i32 v15, v13, v12 // (op 6857) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6857) Load value

v_accvgpr_read v22, a24 // (op 6854) reg expression
v_accvgpr_read v23, a25 // (op 6854) reg expression
v_accvgpr_read v24, a26 // (op 6854) reg expression
v_accvgpr_read v25, a27 // (op 6854) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6855) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6855) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6855) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6855) 

v_accvgpr_write a24, v22 // (op 6856) reg expression
v_accvgpr_write a25, v23 // (op 6856) reg expression
v_accvgpr_write a26, v24 // (op 6856) reg expression
v_accvgpr_write a27, v25 // (op 6856) reg expression




v_mov_b32 v12, 16768 // (op 6862) 
v_add_i32 v15, v13, v12 // (op 6862) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6862) Load value

v_mov_b32 v12, 17280 // (op 6862) 
v_add_i32 v15, v13, v12 // (op 6862) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6862) Load value

v_mov_b32 v12, 17792 // (op 6862) 
v_add_i32 v15, v13, v12 // (op 6862) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6862) Load value

v_mov_b32 v12, 18304 // (op 6862) 
v_add_i32 v15, v13, v12 // (op 6862) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6862) Load value

v_accvgpr_read v22, a28 // (op 6859) reg expression
v_accvgpr_read v23, a29 // (op 6859) reg expression
v_accvgpr_read v24, a30 // (op 6859) reg expression
v_accvgpr_read v25, a31 // (op 6859) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6860) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6860) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6860) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6860) 

v_accvgpr_write a28, v22 // (op 6861) reg expression
v_accvgpr_write a29, v23 // (op 6861) reg expression
v_accvgpr_write a30, v24 // (op 6861) reg expression
v_accvgpr_write a31, v25 // (op 6861) reg expression




v_mov_b32 v12, 32768 // (op 6895) 
v_add_i32 v15, v13, v12 // (op 6895) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6895) Load value

v_mov_b32 v12, 33280 // (op 6895) 
v_add_i32 v15, v13, v12 // (op 6895) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6895) Load value

v_mov_b32 v12, 33792 // (op 6895) 
v_add_i32 v15, v13, v12 // (op 6895) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6895) Load value

v_mov_b32 v12, 34304 // (op 6895) 
v_add_i32 v15, v13, v12 // (op 6895) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6895) Load value

v_accvgpr_read v22, a32 // (op 6892) reg expression
v_accvgpr_read v23, a33 // (op 6892) reg expression
v_accvgpr_read v24, a34 // (op 6892) reg expression
v_accvgpr_read v25, a35 // (op 6892) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6893) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6893) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6893) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6893) 

v_accvgpr_write a32, v22 // (op 6894) reg expression
v_accvgpr_write a33, v23 // (op 6894) reg expression
v_accvgpr_write a34, v24 // (op 6894) reg expression
v_accvgpr_write a35, v25 // (op 6894) reg expression




v_mov_b32 v12, 32896 // (op 6900) 
v_add_i32 v15, v13, v12 // (op 6900) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6900) Load value

v_mov_b32 v12, 33408 // (op 6900) 
v_add_i32 v15, v13, v12 // (op 6900) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6900) Load value

v_mov_b32 v12, 33920 // (op 6900) 
v_add_i32 v15, v13, v12 // (op 6900) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6900) Load value

v_mov_b32 v12, 34432 // (op 6900) 
v_add_i32 v15, v13, v12 // (op 6900) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6900) Load value

v_accvgpr_read v22, a36 // (op 6897) reg expression
v_accvgpr_read v23, a37 // (op 6897) reg expression
v_accvgpr_read v24, a38 // (op 6897) reg expression
v_accvgpr_read v25, a39 // (op 6897) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6898) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6898) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6898) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6898) 

v_accvgpr_write a36, v22 // (op 6899) reg expression
v_accvgpr_write a37, v23 // (op 6899) reg expression
v_accvgpr_write a38, v24 // (op 6899) reg expression
v_accvgpr_write a39, v25 // (op 6899) reg expression




v_mov_b32 v12, 33024 // (op 6905) 
v_add_i32 v15, v13, v12 // (op 6905) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6905) Load value

v_mov_b32 v12, 33536 // (op 6905) 
v_add_i32 v15, v13, v12 // (op 6905) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6905) Load value

v_mov_b32 v12, 34048 // (op 6905) 
v_add_i32 v15, v13, v12 // (op 6905) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6905) Load value

v_mov_b32 v12, 34560 // (op 6905) 
v_add_i32 v15, v13, v12 // (op 6905) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6905) Load value

v_accvgpr_read v22, a40 // (op 6902) reg expression
v_accvgpr_read v23, a41 // (op 6902) reg expression
v_accvgpr_read v24, a42 // (op 6902) reg expression
v_accvgpr_read v25, a43 // (op 6902) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6903) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6903) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6903) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6903) 

v_accvgpr_write a40, v22 // (op 6904) reg expression
v_accvgpr_write a41, v23 // (op 6904) reg expression
v_accvgpr_write a42, v24 // (op 6904) reg expression
v_accvgpr_write a43, v25 // (op 6904) reg expression




v_mov_b32 v12, 33152 // (op 6910) 
v_add_i32 v15, v13, v12 // (op 6910) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6910) Load value

v_mov_b32 v12, 33664 // (op 6910) 
v_add_i32 v15, v13, v12 // (op 6910) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6910) Load value

v_mov_b32 v12, 34176 // (op 6910) 
v_add_i32 v15, v13, v12 // (op 6910) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6910) Load value

v_mov_b32 v12, 34688 // (op 6910) 
v_add_i32 v15, v13, v12 // (op 6910) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6910) Load value

v_accvgpr_read v22, a44 // (op 6907) reg expression
v_accvgpr_read v23, a45 // (op 6907) reg expression
v_accvgpr_read v24, a46 // (op 6907) reg expression
v_accvgpr_read v25, a47 // (op 6907) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6908) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6908) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6908) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6908) 

v_accvgpr_write a44, v22 // (op 6909) reg expression
v_accvgpr_write a45, v23 // (op 6909) reg expression
v_accvgpr_write a46, v24 // (op 6909) reg expression
v_accvgpr_write a47, v25 // (op 6909) reg expression




v_mov_b32 v12, 49152 // (op 6943) 
v_add_i32 v15, v13, v12 // (op 6943) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6943) Load value

v_mov_b32 v12, 49664 // (op 6943) 
v_add_i32 v15, v13, v12 // (op 6943) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6943) Load value

v_mov_b32 v12, 50176 // (op 6943) 
v_add_i32 v15, v13, v12 // (op 6943) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6943) Load value

v_mov_b32 v12, 50688 // (op 6943) 
v_add_i32 v15, v13, v12 // (op 6943) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6943) Load value

v_accvgpr_read v22, a48 // (op 6940) reg expression
v_accvgpr_read v23, a49 // (op 6940) reg expression
v_accvgpr_read v24, a50 // (op 6940) reg expression
v_accvgpr_read v25, a51 // (op 6940) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6941) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6941) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6941) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6941) 

v_accvgpr_write a48, v22 // (op 6942) reg expression
v_accvgpr_write a49, v23 // (op 6942) reg expression
v_accvgpr_write a50, v24 // (op 6942) reg expression
v_accvgpr_write a51, v25 // (op 6942) reg expression




v_mov_b32 v12, 49280 // (op 6948) 
v_add_i32 v15, v13, v12 // (op 6948) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6948) Load value

v_mov_b32 v12, 49792 // (op 6948) 
v_add_i32 v15, v13, v12 // (op 6948) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6948) Load value

v_mov_b32 v12, 50304 // (op 6948) 
v_add_i32 v15, v13, v12 // (op 6948) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6948) Load value

v_mov_b32 v12, 50816 // (op 6948) 
v_add_i32 v15, v13, v12 // (op 6948) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6948) Load value

v_accvgpr_read v22, a52 // (op 6945) reg expression
v_accvgpr_read v23, a53 // (op 6945) reg expression
v_accvgpr_read v24, a54 // (op 6945) reg expression
v_accvgpr_read v25, a55 // (op 6945) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6946) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6946) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6946) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6946) 

v_accvgpr_write a52, v22 // (op 6947) reg expression
v_accvgpr_write a53, v23 // (op 6947) reg expression
v_accvgpr_write a54, v24 // (op 6947) reg expression
v_accvgpr_write a55, v25 // (op 6947) reg expression




v_mov_b32 v12, 49408 // (op 6953) 
v_add_i32 v15, v13, v12 // (op 6953) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6953) Load value

v_mov_b32 v12, 49920 // (op 6953) 
v_add_i32 v15, v13, v12 // (op 6953) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6953) Load value

v_mov_b32 v12, 50432 // (op 6953) 
v_add_i32 v15, v13, v12 // (op 6953) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6953) Load value

v_mov_b32 v12, 50944 // (op 6953) 
v_add_i32 v15, v13, v12 // (op 6953) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6953) Load value

v_accvgpr_read v22, a56 // (op 6950) reg expression
v_accvgpr_read v23, a57 // (op 6950) reg expression
v_accvgpr_read v24, a58 // (op 6950) reg expression
v_accvgpr_read v25, a59 // (op 6950) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6951) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6951) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6951) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6951) 

v_accvgpr_write a56, v22 // (op 6952) reg expression
v_accvgpr_write a57, v23 // (op 6952) reg expression
v_accvgpr_write a58, v24 // (op 6952) reg expression
v_accvgpr_write a59, v25 // (op 6952) reg expression




v_mov_b32 v12, 49536 // (op 6958) 
v_add_i32 v15, v13, v12 // (op 6958) 

buffer_load_dword v18, v15, s[64:67], 0 offen // (op 6958) Load value

v_mov_b32 v12, 50048 // (op 6958) 
v_add_i32 v15, v13, v12 // (op 6958) 

buffer_load_dword v19, v15, s[64:67], 0 offen // (op 6958) Load value

v_mov_b32 v12, 50560 // (op 6958) 
v_add_i32 v15, v13, v12 // (op 6958) 

buffer_load_dword v20, v15, s[64:67], 0 offen // (op 6958) Load value

v_mov_b32 v12, 51072 // (op 6958) 
v_add_i32 v15, v13, v12 // (op 6958) 

buffer_load_dword v21, v15, s[64:67], 0 offen // (op 6958) Load value

v_accvgpr_read v22, a60 // (op 6955) reg expression
v_accvgpr_read v23, a61 // (op 6955) reg expression
v_accvgpr_read v24, a62 // (op 6955) reg expression
v_accvgpr_read v25, a63 // (op 6955) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v22, v22, v18 // (op 6956) 
s_waitcnt vmcnt(2)
v_add_f32 v23, v23, v19 // (op 6956) 
s_waitcnt vmcnt(1)
v_add_f32 v24, v24, v20 // (op 6956) 
s_waitcnt vmcnt(0)
v_add_f32 v25, v25, v21 // (op 6956) 

v_accvgpr_write a60, v22 // (op 6957) reg expression
v_accvgpr_write a61, v23 // (op 6957) reg expression
v_accvgpr_write a62, v24 // (op 6957) reg expression
v_accvgpr_write a63, v25 // (op 6957) reg expression

s_add_u32 s52, s52, 1 // (op 266) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 264) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 264) 
s_mov_b32 s76, s54 // (op 264) convert

s_sub_u32 s54, s76, s19 // (op 264) 

s_sub_u32 s55, s54, 1 // (op 264) 

s_cmp_lt_u32 s53, s55 // (op 264) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_264 // (op 264) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_ReceiveTileLoop_264 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_264:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_299 // (op 299) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_299
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Receive_Tile_299:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Receive_Tile_299:

s_cmp_eq_i32 s18, 0 // (op 342) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_342 // (op 342) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_342
v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_114
v_accvgpr_read v18, a0 // (op 42) 
v_accvgpr_read v19, a1 // (op 42) 
v_accvgpr_read v20, a2 // (op 42) 
v_accvgpr_read v21, a3 // (op 42) 
v_mul_f32 v22, v11, v18 // (op 42) 
v_mul_f32 v23, v11, v19 // (op 42) 
v_mul_f32 v24, v11, v20 // (op 42) 
v_mul_f32 v25, v11, v21 // (op 42) 



v_mov_b32 v12, v17 // (op 48) reg expression

buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 48) Store value


v_accvgpr_read v18, a4 // (op 7040) 
v_accvgpr_read v19, a5 // (op 7040) 
v_accvgpr_read v20, a6 // (op 7040) 
v_accvgpr_read v21, a7 // (op 7040) 
v_mul_f32 v22, v11, v18 // (op 7040) 
v_mul_f32 v23, v11, v19 // (op 7040) 
v_mul_f32 v24, v11, v20 // (op 7040) 
v_mul_f32 v25, v11, v21 // (op 7040) 



s_lshl_b32 s52, s24, 6 // (op 7042) 
v_add_u32 v12, v17, s52 // (op 7042) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7042) Store value


v_accvgpr_read v18, a8 // (op 7052) 
v_accvgpr_read v19, a9 // (op 7052) 
v_accvgpr_read v20, a10 // (op 7052) 
v_accvgpr_read v21, a11 // (op 7052) 
v_mul_f32 v22, v11, v18 // (op 7052) 
v_mul_f32 v23, v11, v19 // (op 7052) 
v_mul_f32 v24, v11, v20 // (op 7052) 
v_mul_f32 v25, v11, v21 // (op 7052) 



s_lshl_b32 s52, s24, 7 // (op 7054) 
v_add_u32 v12, v17, s52 // (op 7054) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7054) Store value


v_accvgpr_read v18, a12 // (op 7064) 
v_accvgpr_read v19, a13 // (op 7064) 
v_accvgpr_read v20, a14 // (op 7064) 
v_accvgpr_read v21, a15 // (op 7064) 
v_mul_f32 v22, v11, v18 // (op 7064) 
v_mul_f32 v23, v11, v19 // (op 7064) 
v_mul_f32 v24, v11, v20 // (op 7064) 
v_mul_f32 v25, v11, v21 // (op 7064) 



s_lshl_b32 s52, s24, 6 // (op 7066) 
s_mul_i32 s53, 3, s52 // (op 7066) 

v_add_u32 v12, v17, s53 // (op 7066) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7066) Store value


v_accvgpr_read v18, a16 // (op 7098) 
v_accvgpr_read v19, a17 // (op 7098) 
v_accvgpr_read v20, a18 // (op 7098) 
v_accvgpr_read v21, a19 // (op 7098) 
v_mul_f32 v22, v11, v18 // (op 7098) 
v_mul_f32 v23, v11, v19 // (op 7098) 
v_mul_f32 v24, v11, v20 // (op 7098) 
v_mul_f32 v25, v11, v21 // (op 7098) 



v_add_u32 v12, 64, v17 // (op 7101) 

buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7101) Store value


v_accvgpr_read v18, a20 // (op 7111) 
v_accvgpr_read v19, a21 // (op 7111) 
v_accvgpr_read v20, a22 // (op 7111) 
v_accvgpr_read v21, a23 // (op 7111) 
v_mul_f32 v22, v11, v18 // (op 7111) 
v_mul_f32 v23, v11, v19 // (op 7111) 
v_mul_f32 v24, v11, v20 // (op 7111) 
v_mul_f32 v25, v11, v21 // (op 7111) 



s_lshl_b32 s52, s24, 6 // (op 7103) 
s_add_u32 s52, s52, 64 // (op 7103) 
v_add_u32 v12, v17, s52 // (op 7103) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7103) Store value


v_accvgpr_read v18, a24 // (op 7116) 
v_accvgpr_read v19, a25 // (op 7116) 
v_accvgpr_read v20, a26 // (op 7116) 
v_accvgpr_read v21, a27 // (op 7116) 
v_mul_f32 v22, v11, v18 // (op 7116) 
v_mul_f32 v23, v11, v19 // (op 7116) 
v_mul_f32 v24, v11, v20 // (op 7116) 
v_mul_f32 v25, v11, v21 // (op 7116) 



s_lshl_b32 s52, s24, 7 // (op 7105) 
s_add_u32 s52, s52, 64 // (op 7105) 
v_add_u32 v12, v17, s52 // (op 7105) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7105) Store value


v_accvgpr_read v18, a28 // (op 7121) 
v_accvgpr_read v19, a29 // (op 7121) 
v_accvgpr_read v20, a30 // (op 7121) 
v_accvgpr_read v21, a31 // (op 7121) 
v_mul_f32 v22, v11, v18 // (op 7121) 
v_mul_f32 v23, v11, v19 // (op 7121) 
v_mul_f32 v24, v11, v20 // (op 7121) 
v_mul_f32 v25, v11, v21 // (op 7121) 



s_lshl_b32 s52, s24, 6 // (op 7107) 
s_mul_i32 s53, 3, s52 // (op 7107) 
s_add_u32 s53, s53, 64 // (op 7107) 

v_add_u32 v12, v17, s53 // (op 7107) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7107) Store value


v_accvgpr_read v18, a32 // (op 7166) 
v_accvgpr_read v19, a33 // (op 7166) 
v_accvgpr_read v20, a34 // (op 7166) 
v_accvgpr_read v21, a35 // (op 7166) 
v_mul_f32 v22, v11, v18 // (op 7166) 
v_mul_f32 v23, v11, v19 // (op 7166) 
v_mul_f32 v24, v11, v20 // (op 7166) 
v_mul_f32 v25, v11, v21 // (op 7166) 



v_add_u32 v12, 128, v17 // (op 7169) 

buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7169) Store value


v_accvgpr_read v18, a36 // (op 7179) 
v_accvgpr_read v19, a37 // (op 7179) 
v_accvgpr_read v20, a38 // (op 7179) 
v_accvgpr_read v21, a39 // (op 7179) 
v_mul_f32 v22, v11, v18 // (op 7179) 
v_mul_f32 v23, v11, v19 // (op 7179) 
v_mul_f32 v24, v11, v20 // (op 7179) 
v_mul_f32 v25, v11, v21 // (op 7179) 



s_lshl_b32 s52, s24, 6 // (op 7171) 
s_add_u32 s52, s52, 128 // (op 7171) 
v_add_u32 v12, v17, s52 // (op 7171) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7171) Store value


v_accvgpr_read v18, a40 // (op 7184) 
v_accvgpr_read v19, a41 // (op 7184) 
v_accvgpr_read v20, a42 // (op 7184) 
v_accvgpr_read v21, a43 // (op 7184) 
v_mul_f32 v22, v11, v18 // (op 7184) 
v_mul_f32 v23, v11, v19 // (op 7184) 
v_mul_f32 v24, v11, v20 // (op 7184) 
v_mul_f32 v25, v11, v21 // (op 7184) 



s_lshl_b32 s52, s24, 7 // (op 7173) 
s_add_u32 s52, s52, 128 // (op 7173) 
v_add_u32 v12, v17, s52 // (op 7173) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7173) Store value


v_accvgpr_read v18, a44 // (op 7189) 
v_accvgpr_read v19, a45 // (op 7189) 
v_accvgpr_read v20, a46 // (op 7189) 
v_accvgpr_read v21, a47 // (op 7189) 
v_mul_f32 v22, v11, v18 // (op 7189) 
v_mul_f32 v23, v11, v19 // (op 7189) 
v_mul_f32 v24, v11, v20 // (op 7189) 
v_mul_f32 v25, v11, v21 // (op 7189) 



s_lshl_b32 s52, s24, 6 // (op 7175) 
s_mul_i32 s53, 3, s52 // (op 7175) 
s_add_u32 s53, s53, 128 // (op 7175) 

v_add_u32 v12, v17, s53 // (op 7175) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7175) Store value


v_accvgpr_read v18, a48 // (op 7234) 
v_accvgpr_read v19, a49 // (op 7234) 
v_accvgpr_read v20, a50 // (op 7234) 
v_accvgpr_read v21, a51 // (op 7234) 
v_mul_f32 v22, v11, v18 // (op 7234) 
v_mul_f32 v23, v11, v19 // (op 7234) 
v_mul_f32 v24, v11, v20 // (op 7234) 
v_mul_f32 v25, v11, v21 // (op 7234) 



v_add_u32 v12, 192, v17 // (op 7237) 

buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7237) Store value


v_accvgpr_read v18, a52 // (op 7247) 
v_accvgpr_read v19, a53 // (op 7247) 
v_accvgpr_read v20, a54 // (op 7247) 
v_accvgpr_read v21, a55 // (op 7247) 
v_mul_f32 v22, v11, v18 // (op 7247) 
v_mul_f32 v23, v11, v19 // (op 7247) 
v_mul_f32 v24, v11, v20 // (op 7247) 
v_mul_f32 v25, v11, v21 // (op 7247) 



s_lshl_b32 s52, s24, 6 // (op 7239) 
s_add_u32 s52, s52, 192 // (op 7239) 
v_add_u32 v12, v17, s52 // (op 7239) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7239) Store value


v_accvgpr_read v18, a56 // (op 7252) 
v_accvgpr_read v19, a57 // (op 7252) 
v_accvgpr_read v20, a58 // (op 7252) 
v_accvgpr_read v21, a59 // (op 7252) 
v_mul_f32 v22, v11, v18 // (op 7252) 
v_mul_f32 v23, v11, v19 // (op 7252) 
v_mul_f32 v24, v11, v20 // (op 7252) 
v_mul_f32 v25, v11, v21 // (op 7252) 



s_lshl_b32 s52, s24, 7 // (op 7241) 
s_add_u32 s52, s52, 192 // (op 7241) 
v_add_u32 v12, v17, s52 // (op 7241) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7241) Store value


v_accvgpr_read v18, a60 // (op 7257) 
v_accvgpr_read v19, a61 // (op 7257) 
v_accvgpr_read v20, a62 // (op 7257) 
v_accvgpr_read v21, a63 // (op 7257) 
v_mul_f32 v22, v11, v18 // (op 7257) 
v_mul_f32 v23, v11, v19 // (op 7257) 
v_mul_f32 v24, v11, v20 // (op 7257) 
v_mul_f32 v25, v11, v21 // (op 7257) 



s_lshl_b32 s52, s24, 6 // (op 7243) 
s_mul_i32 s53, 3, s52 // (op 7243) 
s_add_u32 s53, s53, 192 // (op 7243) 

v_add_u32 v12, v17, s53 // (op 7243) 


buffer_store_dwordx4 v[22:25], v12, s[60:63], 0 offen // (op 7243) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_114:


v_add_u32 v12, 192, v14 // (op 7601) 

buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7601) Load value

v_accvgpr_read v22, a48 // (op 7589) 
v_accvgpr_read v23, a49 // (op 7589) 
v_accvgpr_read v24, a50 // (op 7589) 
v_accvgpr_read v25, a51 // (op 7589) 
v_mul_f32 v26, v11, v22 // (op 7589) 
v_mul_f32 v27, v11, v23 // (op 7589) 
v_mul_f32 v28, v11, v24 // (op 7589) 
v_mul_f32 v29, v11, v25 // (op 7589) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7600) 
v_mul_f32 v23, v0, v19 // (op 7600) 
v_mul_f32 v24, v0, v20 // (op 7600) 
v_mul_f32 v25, v0, v21 // (op 7600) 

v_add_f32 v18, v22, v26 // (op 7590) 
v_add_f32 v19, v23, v27 // (op 7590) 
v_add_f32 v20, v24, v28 // (op 7590) 
v_add_f32 v21, v25, v29 // (op 7590) 




v_add_u32 v12, 192, v17 // (op 7592) 

buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7592) Store value




s_lshl_b32 s52, s4, 6 // (op 7605) 
s_add_u32 s52, s52, 192 // (op 7605) 
v_add_u32 v12, v14, s52 // (op 7605) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7605) Load value

v_accvgpr_read v22, a52 // (op 7615) 
v_accvgpr_read v23, a53 // (op 7615) 
v_accvgpr_read v24, a54 // (op 7615) 
v_accvgpr_read v25, a55 // (op 7615) 
v_mul_f32 v26, v11, v22 // (op 7615) 
v_mul_f32 v27, v11, v23 // (op 7615) 
v_mul_f32 v28, v11, v24 // (op 7615) 
v_mul_f32 v29, v11, v25 // (op 7615) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7603) 
v_mul_f32 v23, v0, v19 // (op 7603) 
v_mul_f32 v24, v0, v20 // (op 7603) 
v_mul_f32 v25, v0, v21 // (op 7603) 

v_add_f32 v18, v22, v26 // (op 7604) 
v_add_f32 v19, v23, v27 // (op 7604) 
v_add_f32 v20, v24, v28 // (op 7604) 
v_add_f32 v21, v25, v29 // (op 7604) 




s_lshl_b32 s52, s24, 6 // (op 7594) 
s_add_u32 s52, s52, 192 // (op 7594) 
v_add_u32 v12, v17, s52 // (op 7594) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7594) Store value




s_lshl_b32 s52, s4, 7 // (op 7609) 
s_add_u32 s52, s52, 192 // (op 7609) 
v_add_u32 v12, v14, s52 // (op 7609) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7609) Load value

v_accvgpr_read v22, a56 // (op 7617) 
v_accvgpr_read v23, a57 // (op 7617) 
v_accvgpr_read v24, a58 // (op 7617) 
v_accvgpr_read v25, a59 // (op 7617) 
v_mul_f32 v26, v11, v22 // (op 7617) 
v_mul_f32 v27, v11, v23 // (op 7617) 
v_mul_f32 v28, v11, v24 // (op 7617) 
v_mul_f32 v29, v11, v25 // (op 7617) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7607) 
v_mul_f32 v23, v0, v19 // (op 7607) 
v_mul_f32 v24, v0, v20 // (op 7607) 
v_mul_f32 v25, v0, v21 // (op 7607) 

v_add_f32 v18, v22, v26 // (op 7608) 
v_add_f32 v19, v23, v27 // (op 7608) 
v_add_f32 v20, v24, v28 // (op 7608) 
v_add_f32 v21, v25, v29 // (op 7608) 




s_lshl_b32 s52, s24, 7 // (op 7596) 
s_add_u32 s52, s52, 192 // (op 7596) 
v_add_u32 v12, v17, s52 // (op 7596) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7596) Store value




s_lshl_b32 s52, s4, 6 // (op 7613) 
s_mul_i32 s53, 3, s52 // (op 7613) 
s_add_u32 s53, s53, 192 // (op 7613) 

v_add_u32 v12, v14, s53 // (op 7613) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7613) Load value

v_accvgpr_read v22, a60 // (op 7619) 
v_accvgpr_read v23, a61 // (op 7619) 
v_accvgpr_read v24, a62 // (op 7619) 
v_accvgpr_read v25, a63 // (op 7619) 
v_mul_f32 v26, v11, v22 // (op 7619) 
v_mul_f32 v27, v11, v23 // (op 7619) 
v_mul_f32 v28, v11, v24 // (op 7619) 
v_mul_f32 v29, v11, v25 // (op 7619) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7611) 
v_mul_f32 v23, v0, v19 // (op 7611) 
v_mul_f32 v24, v0, v20 // (op 7611) 
v_mul_f32 v25, v0, v21 // (op 7611) 

v_add_f32 v18, v22, v26 // (op 7612) 
v_add_f32 v19, v23, v27 // (op 7612) 
v_add_f32 v20, v24, v28 // (op 7612) 
v_add_f32 v21, v25, v29 // (op 7612) 




s_lshl_b32 s52, s24, 6 // (op 7598) 
s_mul_i32 s53, 3, s52 // (op 7598) 
s_add_u32 s53, s53, 192 // (op 7598) 

v_add_u32 v12, v17, s53 // (op 7598) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7598) Store value




v_add_u32 v12, 128, v14 // (op 7522) 

s_nop 0
buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7522) Load value

v_accvgpr_read v22, a32 // (op 7510) 
v_accvgpr_read v23, a33 // (op 7510) 
v_accvgpr_read v24, a34 // (op 7510) 
v_accvgpr_read v25, a35 // (op 7510) 
v_mul_f32 v26, v11, v22 // (op 7510) 
v_mul_f32 v27, v11, v23 // (op 7510) 
v_mul_f32 v28, v11, v24 // (op 7510) 
v_mul_f32 v29, v11, v25 // (op 7510) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7521) 
v_mul_f32 v23, v0, v19 // (op 7521) 
v_mul_f32 v24, v0, v20 // (op 7521) 
v_mul_f32 v25, v0, v21 // (op 7521) 

v_add_f32 v18, v22, v26 // (op 7511) 
v_add_f32 v19, v23, v27 // (op 7511) 
v_add_f32 v20, v24, v28 // (op 7511) 
v_add_f32 v21, v25, v29 // (op 7511) 




v_add_u32 v12, 128, v17 // (op 7513) 

buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7513) Store value




s_lshl_b32 s52, s4, 6 // (op 7526) 
s_add_u32 s52, s52, 128 // (op 7526) 
v_add_u32 v12, v14, s52 // (op 7526) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7526) Load value

v_accvgpr_read v22, a36 // (op 7536) 
v_accvgpr_read v23, a37 // (op 7536) 
v_accvgpr_read v24, a38 // (op 7536) 
v_accvgpr_read v25, a39 // (op 7536) 
v_mul_f32 v26, v11, v22 // (op 7536) 
v_mul_f32 v27, v11, v23 // (op 7536) 
v_mul_f32 v28, v11, v24 // (op 7536) 
v_mul_f32 v29, v11, v25 // (op 7536) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7524) 
v_mul_f32 v23, v0, v19 // (op 7524) 
v_mul_f32 v24, v0, v20 // (op 7524) 
v_mul_f32 v25, v0, v21 // (op 7524) 

v_add_f32 v18, v22, v26 // (op 7525) 
v_add_f32 v19, v23, v27 // (op 7525) 
v_add_f32 v20, v24, v28 // (op 7525) 
v_add_f32 v21, v25, v29 // (op 7525) 




s_lshl_b32 s52, s24, 6 // (op 7515) 
s_add_u32 s52, s52, 128 // (op 7515) 
v_add_u32 v12, v17, s52 // (op 7515) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7515) Store value




s_lshl_b32 s52, s4, 7 // (op 7530) 
s_add_u32 s52, s52, 128 // (op 7530) 
v_add_u32 v12, v14, s52 // (op 7530) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7530) Load value

v_accvgpr_read v22, a40 // (op 7538) 
v_accvgpr_read v23, a41 // (op 7538) 
v_accvgpr_read v24, a42 // (op 7538) 
v_accvgpr_read v25, a43 // (op 7538) 
v_mul_f32 v26, v11, v22 // (op 7538) 
v_mul_f32 v27, v11, v23 // (op 7538) 
v_mul_f32 v28, v11, v24 // (op 7538) 
v_mul_f32 v29, v11, v25 // (op 7538) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7528) 
v_mul_f32 v23, v0, v19 // (op 7528) 
v_mul_f32 v24, v0, v20 // (op 7528) 
v_mul_f32 v25, v0, v21 // (op 7528) 

v_add_f32 v18, v22, v26 // (op 7529) 
v_add_f32 v19, v23, v27 // (op 7529) 
v_add_f32 v20, v24, v28 // (op 7529) 
v_add_f32 v21, v25, v29 // (op 7529) 




s_lshl_b32 s52, s24, 7 // (op 7517) 
s_add_u32 s52, s52, 128 // (op 7517) 
v_add_u32 v12, v17, s52 // (op 7517) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7517) Store value




s_lshl_b32 s52, s4, 6 // (op 7534) 
s_mul_i32 s53, 3, s52 // (op 7534) 
s_add_u32 s53, s53, 128 // (op 7534) 

v_add_u32 v12, v14, s53 // (op 7534) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7534) Load value

v_accvgpr_read v22, a44 // (op 7540) 
v_accvgpr_read v23, a45 // (op 7540) 
v_accvgpr_read v24, a46 // (op 7540) 
v_accvgpr_read v25, a47 // (op 7540) 
v_mul_f32 v26, v11, v22 // (op 7540) 
v_mul_f32 v27, v11, v23 // (op 7540) 
v_mul_f32 v28, v11, v24 // (op 7540) 
v_mul_f32 v29, v11, v25 // (op 7540) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7532) 
v_mul_f32 v23, v0, v19 // (op 7532) 
v_mul_f32 v24, v0, v20 // (op 7532) 
v_mul_f32 v25, v0, v21 // (op 7532) 

v_add_f32 v18, v22, v26 // (op 7533) 
v_add_f32 v19, v23, v27 // (op 7533) 
v_add_f32 v20, v24, v28 // (op 7533) 
v_add_f32 v21, v25, v29 // (op 7533) 




s_lshl_b32 s52, s24, 6 // (op 7519) 
s_mul_i32 s53, 3, s52 // (op 7519) 
s_add_u32 s53, s53, 128 // (op 7519) 

v_add_u32 v12, v17, s53 // (op 7519) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7519) Store value




v_add_u32 v12, 64, v14 // (op 7443) 

s_nop 0
buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7443) Load value

v_accvgpr_read v22, a16 // (op 7431) 
v_accvgpr_read v23, a17 // (op 7431) 
v_accvgpr_read v24, a18 // (op 7431) 
v_accvgpr_read v25, a19 // (op 7431) 
v_mul_f32 v26, v11, v22 // (op 7431) 
v_mul_f32 v27, v11, v23 // (op 7431) 
v_mul_f32 v28, v11, v24 // (op 7431) 
v_mul_f32 v29, v11, v25 // (op 7431) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7442) 
v_mul_f32 v23, v0, v19 // (op 7442) 
v_mul_f32 v24, v0, v20 // (op 7442) 
v_mul_f32 v25, v0, v21 // (op 7442) 

v_add_f32 v18, v22, v26 // (op 7432) 
v_add_f32 v19, v23, v27 // (op 7432) 
v_add_f32 v20, v24, v28 // (op 7432) 
v_add_f32 v21, v25, v29 // (op 7432) 




v_add_u32 v12, 64, v17 // (op 7434) 

buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7434) Store value




s_lshl_b32 s52, s4, 6 // (op 7447) 
s_add_u32 s52, s52, 64 // (op 7447) 
v_add_u32 v12, v14, s52 // (op 7447) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7447) Load value

v_accvgpr_read v22, a20 // (op 7457) 
v_accvgpr_read v23, a21 // (op 7457) 
v_accvgpr_read v24, a22 // (op 7457) 
v_accvgpr_read v25, a23 // (op 7457) 
v_mul_f32 v26, v11, v22 // (op 7457) 
v_mul_f32 v27, v11, v23 // (op 7457) 
v_mul_f32 v28, v11, v24 // (op 7457) 
v_mul_f32 v29, v11, v25 // (op 7457) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7445) 
v_mul_f32 v23, v0, v19 // (op 7445) 
v_mul_f32 v24, v0, v20 // (op 7445) 
v_mul_f32 v25, v0, v21 // (op 7445) 

v_add_f32 v18, v22, v26 // (op 7446) 
v_add_f32 v19, v23, v27 // (op 7446) 
v_add_f32 v20, v24, v28 // (op 7446) 
v_add_f32 v21, v25, v29 // (op 7446) 




s_lshl_b32 s52, s24, 6 // (op 7436) 
s_add_u32 s52, s52, 64 // (op 7436) 
v_add_u32 v12, v17, s52 // (op 7436) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7436) Store value




s_lshl_b32 s52, s4, 7 // (op 7451) 
s_add_u32 s52, s52, 64 // (op 7451) 
v_add_u32 v12, v14, s52 // (op 7451) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7451) Load value

v_accvgpr_read v22, a24 // (op 7459) 
v_accvgpr_read v23, a25 // (op 7459) 
v_accvgpr_read v24, a26 // (op 7459) 
v_accvgpr_read v25, a27 // (op 7459) 
v_mul_f32 v26, v11, v22 // (op 7459) 
v_mul_f32 v27, v11, v23 // (op 7459) 
v_mul_f32 v28, v11, v24 // (op 7459) 
v_mul_f32 v29, v11, v25 // (op 7459) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7449) 
v_mul_f32 v23, v0, v19 // (op 7449) 
v_mul_f32 v24, v0, v20 // (op 7449) 
v_mul_f32 v25, v0, v21 // (op 7449) 

v_add_f32 v18, v22, v26 // (op 7450) 
v_add_f32 v19, v23, v27 // (op 7450) 
v_add_f32 v20, v24, v28 // (op 7450) 
v_add_f32 v21, v25, v29 // (op 7450) 




s_lshl_b32 s52, s24, 7 // (op 7438) 
s_add_u32 s52, s52, 64 // (op 7438) 
v_add_u32 v12, v17, s52 // (op 7438) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7438) Store value




s_lshl_b32 s52, s4, 6 // (op 7455) 
s_mul_i32 s53, 3, s52 // (op 7455) 
s_add_u32 s53, s53, 64 // (op 7455) 

v_add_u32 v12, v14, s53 // (op 7455) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7455) Load value

v_accvgpr_read v22, a28 // (op 7461) 
v_accvgpr_read v23, a29 // (op 7461) 
v_accvgpr_read v24, a30 // (op 7461) 
v_accvgpr_read v25, a31 // (op 7461) 
v_mul_f32 v26, v11, v22 // (op 7461) 
v_mul_f32 v27, v11, v23 // (op 7461) 
v_mul_f32 v28, v11, v24 // (op 7461) 
v_mul_f32 v29, v11, v25 // (op 7461) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7453) 
v_mul_f32 v23, v0, v19 // (op 7453) 
v_mul_f32 v24, v0, v20 // (op 7453) 
v_mul_f32 v25, v0, v21 // (op 7453) 

v_add_f32 v18, v22, v26 // (op 7454) 
v_add_f32 v19, v23, v27 // (op 7454) 
v_add_f32 v20, v24, v28 // (op 7454) 
v_add_f32 v21, v25, v29 // (op 7454) 




s_lshl_b32 s52, s24, 6 // (op 7440) 
s_mul_i32 s53, 3, s52 // (op 7440) 
s_add_u32 s53, s53, 64 // (op 7440) 

v_add_u32 v12, v17, s53 // (op 7440) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7440) Store value




v_mov_b32 v12, v14 // (op 126) reg expression

s_nop 0
buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 126) Load value

v_accvgpr_read v22, a0 // (op 123) 
v_accvgpr_read v23, a1 // (op 123) 
v_accvgpr_read v24, a2 // (op 123) 
v_accvgpr_read v25, a3 // (op 123) 
v_mul_f32 v26, v11, v22 // (op 123) 
v_mul_f32 v27, v11, v23 // (op 123) 
v_mul_f32 v28, v11, v24 // (op 123) 
v_mul_f32 v29, v11, v25 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 127) 
v_mul_f32 v23, v0, v19 // (op 127) 
v_mul_f32 v24, v0, v20 // (op 127) 
v_mul_f32 v25, v0, v21 // (op 127) 

v_add_f32 v18, v22, v26 // (op 124) 
v_add_f32 v19, v23, v27 // (op 124) 
v_add_f32 v20, v24, v28 // (op 124) 
v_add_f32 v21, v25, v29 // (op 124) 




v_mov_b32 v12, v17 // (op 125) reg expression

buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 125) Store value




s_lshl_b32 s52, s4, 6 // (op 7357) 
v_add_u32 v12, v14, s52 // (op 7357) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7357) Load value

v_accvgpr_read v22, a4 // (op 7354) 
v_accvgpr_read v23, a5 // (op 7354) 
v_accvgpr_read v24, a6 // (op 7354) 
v_accvgpr_read v25, a7 // (op 7354) 
v_mul_f32 v26, v11, v22 // (op 7354) 
v_mul_f32 v27, v11, v23 // (op 7354) 
v_mul_f32 v28, v11, v24 // (op 7354) 
v_mul_f32 v29, v11, v25 // (op 7354) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7358) 
v_mul_f32 v23, v0, v19 // (op 7358) 
v_mul_f32 v24, v0, v20 // (op 7358) 
v_mul_f32 v25, v0, v21 // (op 7358) 

v_add_f32 v18, v22, v26 // (op 7355) 
v_add_f32 v19, v23, v27 // (op 7355) 
v_add_f32 v20, v24, v28 // (op 7355) 
v_add_f32 v21, v25, v29 // (op 7355) 




s_lshl_b32 s52, s24, 6 // (op 7356) 
v_add_u32 v12, v17, s52 // (op 7356) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7356) Store value




s_lshl_b32 s52, s4, 7 // (op 7369) 
v_add_u32 v12, v14, s52 // (op 7369) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7369) Load value

v_accvgpr_read v22, a8 // (op 7366) 
v_accvgpr_read v23, a9 // (op 7366) 
v_accvgpr_read v24, a10 // (op 7366) 
v_accvgpr_read v25, a11 // (op 7366) 
v_mul_f32 v26, v11, v22 // (op 7366) 
v_mul_f32 v27, v11, v23 // (op 7366) 
v_mul_f32 v28, v11, v24 // (op 7366) 
v_mul_f32 v29, v11, v25 // (op 7366) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7370) 
v_mul_f32 v23, v0, v19 // (op 7370) 
v_mul_f32 v24, v0, v20 // (op 7370) 
v_mul_f32 v25, v0, v21 // (op 7370) 

v_add_f32 v18, v22, v26 // (op 7367) 
v_add_f32 v19, v23, v27 // (op 7367) 
v_add_f32 v20, v24, v28 // (op 7367) 
v_add_f32 v21, v25, v29 // (op 7367) 




s_lshl_b32 s52, s24, 7 // (op 7368) 
v_add_u32 v12, v17, s52 // (op 7368) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7368) Store value




s_lshl_b32 s52, s4, 6 // (op 7381) 
s_mul_i32 s53, 3, s52 // (op 7381) 

v_add_u32 v12, v14, s53 // (op 7381) 


buffer_load_dwordx4 v[18:21], v12, s[56:59], 0 offen // (op 7381) Load value

v_accvgpr_read v22, a12 // (op 7378) 
v_accvgpr_read v23, a13 // (op 7378) 
v_accvgpr_read v24, a14 // (op 7378) 
v_accvgpr_read v25, a15 // (op 7378) 
v_mul_f32 v26, v11, v22 // (op 7378) 
v_mul_f32 v27, v11, v23 // (op 7378) 
v_mul_f32 v28, v11, v24 // (op 7378) 
v_mul_f32 v29, v11, v25 // (op 7378) 

s_waitcnt vmcnt(0)
v_mul_f32 v22, v0, v18 // (op 7382) 
v_mul_f32 v23, v0, v19 // (op 7382) 
v_mul_f32 v24, v0, v20 // (op 7382) 
v_mul_f32 v25, v0, v21 // (op 7382) 

v_add_f32 v18, v22, v26 // (op 7379) 
v_add_f32 v19, v23, v27 // (op 7379) 
v_add_f32 v20, v24, v28 // (op 7379) 
v_add_f32 v21, v25, v29 // (op 7379) 




s_lshl_b32 s52, s24, 6 // (op 7380) 
s_mul_i32 s53, 3, s52 // (op 7380) 

v_add_u32 v12, v17, s53 // (op 7380) 


buffer_store_dwordx4 v[18:21], v12, s[60:63], 0 offen // (op 7380) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_114:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_342 // (op 342) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_342
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_342:
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_342:


s_mul_i32 s52, s8, s16 // (op 255) 
s_cmp_gt_u32 s9, s52 // (op 255) 
s_mov_b32 s53, scc // (op 255) 
s_sub_u32 s54, s9, s52 // (op 255) 

s_cmp_eq_i32 s53, 0 // (op 255) 
s_cselect_b32 s52, 0, s54 // (op 255) 


s_cmp_lt_u32 s8, s52 // (op 255) 
s_mov_b32 s53, scc // (op 255) 
s_cmp_eq_i32 s53, 0 // (op 255) 
s_cselect_b32 s54, s52, s8 // (op 255) 


s_cmp_lt_u32 s0, s54 // (op 255) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_255 // (op 255) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_SKStreamTileLoop_255 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_255:













s_mov_b32 s0, 0 // (op 396) call()
s_mov_b32 s1, 0 // (op 397) call()
s_mov_b32 s0, 0 // (op 411) call()
s_mul_i32 s15, s50, s16 // (op 410) 
s_cmp_gt_u32 s51, s15 // (op 410) 
s_mov_b32 s6, scc // (op 410) 
s_sub_u32 s7, s51, s15 // (op 410) 

s_cmp_eq_i32 s6, 0 // (op 410) 
s_cselect_b32 s8, 0, s7 // (op 410) 


s_cmp_lt_u32 s50, s8 // (op 410) 
s_mov_b32 s15, scc // (op 410) 
s_cmp_eq_i32 s15, 0 // (op 410) 
s_cselect_b32 s6, s8, s50 // (op 410) 


s_cmp_lt_u32 s0, s6 // (op 410) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_410 // (op 410) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_410 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_410:
v_lshrrev_b32 v3, 7, v1 // (op 14372) 
s_mul_i32 s15, s16, s50 // (op 14372) 
s_add_u32 s15, s15, s0 // (op 14372) 
v_and_b32 v4, 63, v1 // (op 14372) 
s_add_u32 s6, s15, s9 // (op 14372) 

v_lshrrev_b32 v9, 4, v4 // (op 14372) 
s_mul_hi_u32 s15, s6, s13 // (op 14372) 
v_lshlrev_b32 v10, 2, v9 // (op 14372) 

s_sub_u32 s7, s6, s15 // (op 14372) 
v_lshl_add_u32 v9, v3, 6, v10 // (op 14372) 


s_lshr_b32 s8, s7, 1 // (op 14372) 

v_lshrrev_b32 v3, 2, v4 // (op 14372) 

s_add_u32 s7, s8, s15 // (op 14372) 


v_and_b32 v4, 3, v3 // (op 14372) 

s_lshr_b32 s8, s7, s11 // (op 14372) 

v_and_b32 v3, 3, v1 // (op 14372) 
s_and_b32 s7, s11, 2147483648 // (op 14372) 
v_lshl_add_u32 v10, v4, 2, v3 // (op 14372) 


s_cmp_eq_i32 s7, 0 // (op 14372) 
s_mov_b32 s15, scc // (op 14372) 

v_lshrrev_b32 v3, 6, v1 // (op 14372) 
s_cmp_eq_i32 s15, 0 // (op 14372) 
s_cselect_b32 s7, s6, s8 // (op 14372) 



v_and_b32 v4, 1, v3 // (op 14372) 

s_mul_hi_u32 s6, s7, s12 // (op 14372) 
v_lshl_add_u32 v3, v4, 6, v10 // (op 14372) 


s_sub_u32 s8, s7, s6 // (op 14372) 
s_lshr_b32 s15, s8, 1 // (op 14372) 

s_add_u32 s8, s15, s6 // (op 14372) 


s_lshr_b32 s6, s8, s10 // (op 14372) 

s_and_b32 s8, s10, 2147483648 // (op 14372) 
s_cmp_eq_i32 s8, 0 // (op 14372) 
s_mov_b32 s15, scc // (op 14372) 

s_cmp_eq_i32 s15, 0 // (op 14372) 
s_cselect_b32 s8, s7, s6 // (op 14372) 


s_mul_i32 s6, s8, s14 // (op 14372) 
v_lshl_add_u32 v4, s8, 7, v9 // (op 14372) 


s_sub_u32 s8, s7, s6 // (op 14372) 


v_lshl_add_u32 v9, s8, 7, v3 // (op 14372) 


v_mul_lo_u32 v3, v9, s4 // (op 14372) 

v_add_lshl_u32 v9, v4, v3, 2 // (op 14372) 


s_mul_i32 s19, s4, s23 // (op 14374) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 14374) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 14374) least significant
s_mul_hi_u32 s6, s4, s22 // (op 14374) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14374) most significant: sum
s_add_u32 s19, s19, s6 // (op 14374) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 14374) 

s_mov_b32 s56, s20 // (op 14374) reg expression
s_mov_b32 s57, s21 // (op 14374) reg expression
s_mov_b32 s58, s28 // (op 14374) reg expression
s_mov_b32 s59, 131072 // (op 14374) call()

v_lshrrev_b32 v10, 7, v1 // (op 14574) 
s_mul_i32 s15, s16, s50 // (op 14574) 
s_add_u32 s15, s15, s0 // (op 14574) 
v_and_b32 v3, 63, v1 // (op 14574) 
s_add_u32 s6, s15, s9 // (op 14574) 

v_lshrrev_b32 v4, 4, v3 // (op 14574) 
s_mul_hi_u32 s15, s6, s13 // (op 14574) 
v_lshlrev_b32 v12, 2, v4 // (op 14574) 

s_sub_u32 s7, s6, s15 // (op 14574) 
v_lshl_add_u32 v4, v10, 6, v12 // (op 14574) 


s_lshr_b32 s8, s7, 1 // (op 14574) 

v_lshrrev_b32 v10, 2, v3 // (op 14574) 

s_add_u32 s7, s8, s15 // (op 14574) 


v_and_b32 v3, 3, v10 // (op 14574) 

s_lshr_b32 s8, s7, s11 // (op 14574) 

v_and_b32 v10, 3, v1 // (op 14574) 
s_and_b32 s7, s11, 2147483648 // (op 14574) 
v_lshl_add_u32 v12, v3, 2, v10 // (op 14574) 


s_cmp_eq_i32 s7, 0 // (op 14574) 
s_mov_b32 s15, scc // (op 14574) 

v_lshrrev_b32 v3, 6, v1 // (op 14574) 
s_cmp_eq_i32 s15, 0 // (op 14574) 
s_cselect_b32 s7, s6, s8 // (op 14574) 



v_and_b32 v10, 1, v3 // (op 14574) 

s_mul_hi_u32 s6, s7, s12 // (op 14574) 
v_lshl_add_u32 v3, v10, 6, v12 // (op 14574) 


s_sub_u32 s8, s7, s6 // (op 14574) 
s_lshr_b32 s15, s8, 1 // (op 14574) 

s_add_u32 s8, s15, s6 // (op 14574) 


s_lshr_b32 s6, s8, s10 // (op 14574) 

s_and_b32 s8, s10, 2147483648 // (op 14574) 
s_cmp_eq_i32 s8, 0 // (op 14574) 
s_mov_b32 s15, scc // (op 14574) 

s_cmp_eq_i32 s15, 0 // (op 14574) 
s_cselect_b32 s8, s7, s6 // (op 14574) 


s_mul_i32 s6, s8, s14 // (op 14574) 
v_lshl_add_u32 v10, s8, 7, v4 // (op 14574) 


s_sub_u32 s8, s7, s6 // (op 14574) 


v_lshl_add_u32 v4, s8, 7, v3 // (op 14574) 


v_mul_lo_u32 v3, v4, s24 // (op 14574) 

v_add_lshl_u32 v4, v10, v3, 2 // (op 14574) 


s_mul_i32 s19, s24, s23 // (op 14576) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 14576) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 14576) least significant
s_mul_hi_u32 s6, s24, s22 // (op 14576) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14576) most significant: sum
s_add_u32 s19, s19, s6 // (op 14576) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 14576) 

s_mov_b32 s60, s26 // (op 14576) reg expression
s_mov_b32 s61, s27 // (op 14576) reg expression
s_mov_b32 s62, s28 // (op 14576) reg expression
s_mov_b32 s63, 131072 // (op 14576) call()

v_lshrrev_b32 v3, 7, v1 // (op 14653) 
v_and_b32 v10, 15, v1 // (op 14653) 
v_lshl_add_u32 v12, v3, 6, v10 // (op 14653) 


v_and_b32 v3, 48, v1 // (op 14653) 
v_lshl_add_u32 v10, v12, 7, v3 // (op 14653) 


v_lshrrev_b32 v3, 6, v1 // (op 14728) 
v_and_b32 v12, 1, v3 // (op 14728) 

v_and_b32 v3, 15, v1 // (op 14728) 
v_lshl_add_u32 v13, v12, 6, v3 // (op 14728) 


v_and_b32 v3, 63, v1 // (op 14728) 
v_lshrrev_b32 v12, 4, v3 // (op 14728) 

v_lshlrev_b32 v3, 5, v12 // (op 14728) 

v_lshl_add_u32 v12, v13, 7, v3 // (op 14728) 


v_lshlrev_b32 v3, 2, v12 // (op 14728) 

v_lshrrev_b32 v12, 3, v3 // (op 14728) 

v_accvgpr_write a0, 0.00000 // (op 161) call()
v_accvgpr_write a1, 0.00000 // (op 161) call()
v_accvgpr_write a2, 0.00000 // (op 161) call()
v_accvgpr_write a3, 0.00000 // (op 161) call()
v_accvgpr_write a4, 0.00000 // (op 543) call()
v_accvgpr_write a5, 0.00000 // (op 543) call()
v_accvgpr_write a6, 0.00000 // (op 543) call()
v_accvgpr_write a7, 0.00000 // (op 543) call()
v_accvgpr_write a8, 0.00000 // (op 679) call()
v_accvgpr_write a9, 0.00000 // (op 679) call()
v_accvgpr_write a10, 0.00000 // (op 679) call()
v_accvgpr_write a11, 0.00000 // (op 679) call()
v_accvgpr_write a12, 0.00000 // (op 815) call()
v_accvgpr_write a13, 0.00000 // (op 815) call()
v_accvgpr_write a14, 0.00000 // (op 815) call()
v_accvgpr_write a15, 0.00000 // (op 815) call()
v_accvgpr_write a16, 0.00000 // (op 1039) call()
v_accvgpr_write a17, 0.00000 // (op 1039) call()
v_accvgpr_write a18, 0.00000 // (op 1039) call()
v_accvgpr_write a19, 0.00000 // (op 1039) call()
v_accvgpr_write a20, 0.00000 // (op 1300) call()
v_accvgpr_write a21, 0.00000 // (op 1300) call()
v_accvgpr_write a22, 0.00000 // (op 1300) call()
v_accvgpr_write a23, 0.00000 // (op 1300) call()
v_accvgpr_write a24, 0.00000 // (op 1301) call()
v_accvgpr_write a25, 0.00000 // (op 1301) call()
v_accvgpr_write a26, 0.00000 // (op 1301) call()
v_accvgpr_write a27, 0.00000 // (op 1301) call()
v_accvgpr_write a28, 0.00000 // (op 1302) call()
v_accvgpr_write a29, 0.00000 // (op 1302) call()
v_accvgpr_write a30, 0.00000 // (op 1302) call()
v_accvgpr_write a31, 0.00000 // (op 1302) call()
v_accvgpr_write a32, 0.00000 // (op 1649) call()
v_accvgpr_write a33, 0.00000 // (op 1649) call()
v_accvgpr_write a34, 0.00000 // (op 1649) call()
v_accvgpr_write a35, 0.00000 // (op 1649) call()
v_accvgpr_write a36, 0.00000 // (op 1910) call()
v_accvgpr_write a37, 0.00000 // (op 1910) call()
v_accvgpr_write a38, 0.00000 // (op 1910) call()
v_accvgpr_write a39, 0.00000 // (op 1910) call()
v_accvgpr_write a40, 0.00000 // (op 1911) call()
v_accvgpr_write a41, 0.00000 // (op 1911) call()
v_accvgpr_write a42, 0.00000 // (op 1911) call()
v_accvgpr_write a43, 0.00000 // (op 1911) call()
v_accvgpr_write a44, 0.00000 // (op 1912) call()
v_accvgpr_write a45, 0.00000 // (op 1912) call()
v_accvgpr_write a46, 0.00000 // (op 1912) call()
v_accvgpr_write a47, 0.00000 // (op 1912) call()
v_accvgpr_write a48, 0.00000 // (op 2259) call()
v_accvgpr_write a49, 0.00000 // (op 2259) call()
v_accvgpr_write a50, 0.00000 // (op 2259) call()
v_accvgpr_write a51, 0.00000 // (op 2259) call()
v_accvgpr_write a52, 0.00000 // (op 2520) call()
v_accvgpr_write a53, 0.00000 // (op 2520) call()
v_accvgpr_write a54, 0.00000 // (op 2520) call()
v_accvgpr_write a55, 0.00000 // (op 2520) call()
v_accvgpr_write a56, 0.00000 // (op 2521) call()
v_accvgpr_write a57, 0.00000 // (op 2521) call()
v_accvgpr_write a58, 0.00000 // (op 2521) call()
v_accvgpr_write a59, 0.00000 // (op 2521) call()
v_accvgpr_write a60, 0.00000 // (op 2522) call()
v_accvgpr_write a61, 0.00000 // (op 2522) call()
v_accvgpr_write a62, 0.00000 // (op 2522) call()
v_accvgpr_write a63, 0.00000 // (op 2522) call()
v_lshrrev_b32 v3, 3, v1 // (op 14108) 
s_mul_i32 s15, s16, s50 // (op 14108) 
s_add_u32 s15, s15, s0 // (op 14108) 
v_and_b32 v13, -8, v1 // (op 14108) 
s_add_u32 s6, s15, s9 // (op 14108) 

v_sub_u32 v14, v1, v13 // (op 14108) 

s_mul_hi_u32 s15, s6, s13 // (op 14108) 
v_lshlrev_b32 v13, 4, v14 // (op 14108) 

s_sub_u32 s7, s6, s15 // (op 14108) 
s_lshr_b32 s8, s7, 1 // (op 14108) 

s_add_u32 s7, s8, s15 // (op 14108) 


s_lshr_b32 s8, s7, s11 // (op 14108) 

s_and_b32 s7, s11, 2147483648 // (op 14108) 
s_cmp_eq_i32 s7, 0 // (op 14108) 
s_mov_b32 s15, scc // (op 14108) 

s_cmp_eq_i32 s15, 0 // (op 14108) 
s_cselect_b32 s7, s6, s8 // (op 14108) 


s_mul_hi_u32 s8, s7, s12 // (op 14108) 
s_sub_u32 s15, s7, s8 // (op 14108) 
s_lshr_b32 s18, s15, 1 // (op 14108) 

s_add_u32 s15, s18, s8 // (op 14108) 


s_lshr_b32 s8, s15, s10 // (op 14108) 

s_and_b32 s15, s10, 2147483648 // (op 14108) 
s_cmp_eq_i32 s15, 0 // (op 14108) 
s_mov_b32 s18, scc // (op 14108) 

s_cmp_eq_i32 s18, 0 // (op 14108) 
s_cselect_b32 s15, s7, s8 // (op 14108) 


v_lshl_add_u32 v14, s15, 7, v3 // (op 14108) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 14108) 
s_mul_i32 s8, s7, s18 // (op 14108) 


s_sub_u32 s7, s6, s8 // (op 14108) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 14108) 


v_mov_b32 v17, 0 // (op 14108) convert
v_mov_b32 v16, v3 // (op 14108) convert

v_mul_lo_u32 v19, v14, s31 // (op 14108) most significant: low of low * high
v_mul_hi_u32 v3, v14, s30 // (op 14108) most significant: high of low * low
v_mul_lo_u32 v18, v14, s30 // (op 14108) least significant: low of low * low
v_add_u32 v19, v19, v3 // (op 14108) most significant: sum

v_add_co_u32 v18, s[18:19], v18, v16 // (op 14108) least significant half
v_addc_co_u32 v19, s[18:19], v19, v17, s[18:19] // (op 14108) most significant half



s_mul_i32 s19, s30, s35 // (op 14110) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 14110) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 14110) least significant
s_mul_hi_u32 s6, s30, s34 // (op 14110) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14110) most significant: sum
s_add_u32 s19, s19, s6 // (op 14110) most significant: sum


s_mov_b32 s68, s32 // (op 14110) reg expression
s_mov_b32 s69, s33 // (op 14110) reg expression
s_mov_b32 s70, s18 // (op 14110) reg expression
s_mov_b32 s71, 131072 // (op 14110) call()

v_lshrrev_b32 v3, 3, v1 // (op 14127) 
s_mul_i32 s15, s16, s50 // (op 14127) 
s_add_u32 s15, s15, s0 // (op 14127) 
v_and_b32 v13, -8, v1 // (op 14127) 
s_add_u32 s6, s15, s9 // (op 14127) 

v_sub_u32 v14, v1, v13 // (op 14127) 

s_mul_hi_u32 s15, s6, s13 // (op 14127) 
v_lshlrev_b32 v13, 4, v14 // (op 14127) 

s_sub_u32 s7, s6, s15 // (op 14127) 
s_lshr_b32 s8, s7, 1 // (op 14127) 

s_add_u32 s7, s8, s15 // (op 14127) 


s_lshr_b32 s8, s7, s11 // (op 14127) 

s_and_b32 s7, s11, 2147483648 // (op 14127) 
s_cmp_eq_i32 s7, 0 // (op 14127) 
s_mov_b32 s15, scc // (op 14127) 

s_cmp_eq_i32 s15, 0 // (op 14127) 
s_cselect_b32 s7, s6, s8 // (op 14127) 


s_mul_hi_u32 s8, s7, s12 // (op 14127) 
s_sub_u32 s15, s7, s8 // (op 14127) 
s_lshr_b32 s18, s15, 1 // (op 14127) 

s_add_u32 s15, s18, s8 // (op 14127) 


s_lshr_b32 s8, s15, s10 // (op 14127) 

s_and_b32 s15, s10, 2147483648 // (op 14127) 
s_cmp_eq_i32 s15, 0 // (op 14127) 
s_mov_b32 s18, scc // (op 14127) 

s_cmp_eq_i32 s18, 0 // (op 14127) 
s_cselect_b32 s15, s7, s8 // (op 14127) 


v_lshl_add_u32 v14, s15, 7, v3 // (op 14127) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 14127) 
s_mul_i32 s8, s7, s18 // (op 14127) 


s_sub_u32 s7, s6, s8 // (op 14127) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 14127) 


v_mov_b32 v17, 0 // (op 14127) convert
v_mov_b32 v16, v3 // (op 14127) convert

v_mul_lo_u32 v21, v14, s31 // (op 14127) most significant: low of low * high
v_mul_hi_u32 v3, v14, s30 // (op 14127) most significant: high of low * low
v_mul_lo_u32 v20, v14, s30 // (op 14127) least significant: low of low * low
v_add_u32 v21, v21, v3 // (op 14127) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v16 // (op 14127) least significant half
v_addc_co_u32 v21, s[18:19], v21, v17, s[18:19] // (op 14127) most significant half



s_mul_i32 s19, s30, s35 // (op 14129) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 14129) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 14129) least significant
s_mul_hi_u32 s6, s30, s34 // (op 14129) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14129) most significant: sum
s_add_u32 s19, s19, s6 // (op 14129) most significant: sum


s_mov_b32 s68, s32 // (op 14129) reg expression
s_mov_b32 s69, s33 // (op 14129) reg expression
s_mov_b32 s70, s18 // (op 14129) reg expression
s_mov_b32 s71, 131072 // (op 14129) call()

v_lshrrev_b32 v3, 2, v1 // (op 14254) 
s_mul_i32 s15, s16, s50 // (op 14254) 
s_add_u32 s15, s15, s0 // (op 14254) 
v_and_b32 v13, -4, v1 // (op 14254) 
s_add_u32 s6, s15, s9 // (op 14254) 

v_sub_u32 v14, v1, v13 // (op 14254) 

s_mul_hi_u32 s15, s6, s13 // (op 14254) 
v_lshlrev_b32 v13, 5, v14 // (op 14254) 

s_sub_u32 s7, s6, s15 // (op 14254) 
s_lshr_b32 s8, s7, 1 // (op 14254) 

s_add_u32 s7, s8, s15 // (op 14254) 


s_lshr_b32 s8, s7, s11 // (op 14254) 

s_and_b32 s7, s11, 2147483648 // (op 14254) 
s_cmp_eq_i32 s7, 0 // (op 14254) 
s_mov_b32 s15, scc // (op 14254) 

s_cmp_eq_i32 s15, 0 // (op 14254) 
s_cselect_b32 s7, s6, s8 // (op 14254) 


s_mul_hi_u32 s8, s7, s12 // (op 14254) 
s_sub_u32 s15, s7, s8 // (op 14254) 
s_lshr_b32 s18, s15, 1 // (op 14254) 

s_add_u32 s15, s18, s8 // (op 14254) 


s_lshr_b32 s8, s15, s10 // (op 14254) 

s_and_b32 s15, s10, 2147483648 // (op 14254) 
s_cmp_eq_i32 s15, 0 // (op 14254) 
s_mov_b32 s18, scc // (op 14254) 

s_cmp_eq_i32 s18, 0 // (op 14254) 
s_cselect_b32 s15, s7, s8 // (op 14254) 


s_mul_i32 s8, s15, s14 // (op 14254) 

s_sub_u32 s15, s7, s8 // (op 14254) 

v_lshl_add_u32 v14, s15, 7, v3 // (op 14254) 


v_mul_lo_u32 v17, v14, s37 // (op 14254) most significant: low of low * high
v_mul_hi_u32 v3, v14, s36 // (op 14254) most significant: high of low * low
v_mul_lo_u32 v16, v14, s36 // (op 14254) least significant: low of low * low
v_add_u32 v17, v17, v3 // (op 14254) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 14254) 
s_mul_i32 s8, s7, s18 // (op 14254) 


s_sub_u32 s7, s6, s8 // (op 14254) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 14254) 


v_mov_b32 v15, 0 // (op 14254) convert
v_mov_b32 v14, v3 // (op 14254) convert

v_add_co_u32 v22, s[18:19], v14, v16 // (op 14254) least significant half
v_addc_co_u32 v23, s[18:19], v15, v17, s[18:19] // (op 14254) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 14254) 


v_lshrrev_b64 v[16:17], 3, v[22:23] // (op 14254) 

s_mul_i32 s19, s36, s23 // (op 14256) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 14256) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 14256) least significant
s_mul_hi_u32 s6, s36, s22 // (op 14256) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14256) most significant: sum
s_add_u32 s19, s19, s6 // (op 14256) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 14256) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 14256) 

s_mov_b32 s72, s38 // (op 14256) reg expression
s_mov_b32 s73, s39 // (op 14256) reg expression
s_mov_b32 s74, s18 // (op 14256) reg expression
s_mov_b32 s75, 131072 // (op 14256) call()

v_lshrrev_b32 v3, 2, v1 // (op 14273) 
s_mul_i32 s15, s16, s50 // (op 14273) 
s_add_u32 s15, s15, s0 // (op 14273) 
v_and_b32 v13, -4, v1 // (op 14273) 
s_add_u32 s6, s15, s9 // (op 14273) 

v_sub_u32 v14, v1, v13 // (op 14273) 

s_mul_hi_u32 s15, s6, s13 // (op 14273) 
v_lshlrev_b32 v13, 5, v14 // (op 14273) 

s_sub_u32 s7, s6, s15 // (op 14273) 
s_lshr_b32 s8, s7, 1 // (op 14273) 

s_add_u32 s7, s8, s15 // (op 14273) 


s_lshr_b32 s8, s7, s11 // (op 14273) 

s_and_b32 s7, s11, 2147483648 // (op 14273) 
s_cmp_eq_i32 s7, 0 // (op 14273) 
s_mov_b32 s15, scc // (op 14273) 

s_cmp_eq_i32 s15, 0 // (op 14273) 
s_cselect_b32 s7, s6, s8 // (op 14273) 


s_mul_hi_u32 s8, s7, s12 // (op 14273) 
s_sub_u32 s15, s7, s8 // (op 14273) 
s_lshr_b32 s18, s15, 1 // (op 14273) 

s_add_u32 s15, s18, s8 // (op 14273) 


s_lshr_b32 s8, s15, s10 // (op 14273) 

s_and_b32 s15, s10, 2147483648 // (op 14273) 
s_cmp_eq_i32 s15, 0 // (op 14273) 
s_mov_b32 s18, scc // (op 14273) 

s_cmp_eq_i32 s18, 0 // (op 14273) 
s_cselect_b32 s15, s7, s8 // (op 14273) 


s_mul_i32 s8, s15, s14 // (op 14273) 

s_sub_u32 s15, s7, s8 // (op 14273) 

v_lshl_add_u32 v14, s15, 7, v3 // (op 14273) 


v_mul_lo_u32 v23, v14, s37 // (op 14273) most significant: low of low * high
v_mul_hi_u32 v3, v14, s36 // (op 14273) most significant: high of low * low
v_mul_lo_u32 v22, v14, s36 // (op 14273) least significant: low of low * low
v_add_u32 v23, v23, v3 // (op 14273) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 14273) 
s_mul_i32 s8, s7, s18 // (op 14273) 


s_sub_u32 s7, s6, s8 // (op 14273) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 14273) 


v_mov_b32 v15, 0 // (op 14273) convert
v_mov_b32 v14, v3 // (op 14273) convert

v_add_co_u32 v24, s[18:19], v14, v22 // (op 14273) least significant half
v_addc_co_u32 v25, s[18:19], v15, v23, s[18:19] // (op 14273) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 14273) 


v_lshrrev_b64 v[14:15], 3, v[24:25] // (op 14273) 

s_mul_i32 s19, s36, s23 // (op 14275) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 14275) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 14275) least significant
s_mul_hi_u32 s6, s36, s22 // (op 14275) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14275) most significant: sum
s_add_u32 s19, s19, s6 // (op 14275) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 14275) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 14275) 

s_mov_b32 s72, s38 // (op 14275) reg expression
s_mov_b32 s73, s39 // (op 14275) reg expression
s_mov_b32 s74, s18 // (op 14275) reg expression
s_mov_b32 s75, 131072 // (op 14275) call()

v_and_b32 v3, 63, v1 // (op 14979) 
s_mul_i32 s15, s16, s50 // (op 14979) 
s_add_u32 s15, s15, s0 // (op 14979) 
v_lshrrev_b32 v13, 4, v3 // (op 14979) 
s_add_u32 s6, s15, s9 // (op 14979) 

v_and_b32 v22, 3, v13 // (op 14979) 

s_mul_hi_u32 s15, s6, s13 // (op 14979) 
v_and_b32 v13, 15, v1 // (op 14979) 
s_sub_u32 s7, s6, s15 // (op 14979) 
v_lshl_add_u32 v23, v22, 4, v13 // (op 14979) 


s_lshr_b32 s8, s7, 1 // (op 14979) 

v_lshrrev_b32 v13, 7, v1 // (op 14979) 
s_add_u32 s7, s8, s15 // (op 14979) 


v_lshl_add_u32 v22, v13, 6, v23 // (op 14979) 


s_lshr_b32 s8, s7, s11 // (op 14979) 

v_lshrrev_b32 v13, 6, v3 // (op 14979) 

s_and_b32 s7, s11, 2147483648 // (op 14979) 
v_lshlrev_b32 v3, 2, v13 // (op 14979) 

s_cmp_eq_i32 s7, 0 // (op 14979) 
s_mov_b32 s15, scc // (op 14979) 

s_cmp_eq_i32 s15, 0 // (op 14979) 
s_cselect_b32 s7, s6, s8 // (op 14979) 


s_mul_hi_u32 s8, s7, s12 // (op 14979) 
s_sub_u32 s15, s7, s8 // (op 14979) 
s_lshr_b32 s18, s15, 1 // (op 14979) 

s_add_u32 s15, s18, s8 // (op 14979) 


s_lshr_b32 s8, s15, s10 // (op 14979) 

s_and_b32 s15, s10, 2147483648 // (op 14979) 
s_cmp_eq_i32 s15, 0 // (op 14979) 
s_mov_b32 s18, scc // (op 14979) 

s_cmp_eq_i32 s18, 0 // (op 14979) 
s_cselect_b32 s15, s7, s8 // (op 14979) 


v_lshl_add_u32 v13, s15, 7, v22 // (op 14979) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 14979) 
s_mul_i32 s8, s7, s18 // (op 14979) 


s_sub_u32 s7, s6, s8 // (op 14979) 


v_lshl_add_u32 v22, s7, 2, v3 // (op 14979) 


v_mul_lo_u32 v3, v13, s40 // (op 14979) 
v_add_u32 v3, v3, v22 // (op 14979) 


s_mul_i32 s19, s40, s35 // (op 14981) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 14981) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 14981) least significant
s_mul_hi_u32 s6, s40, s34 // (op 14981) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14981) most significant: sum
s_add_u32 s19, s19, s6 // (op 14981) most significant: sum


s_mov_b32 s52, s42 // (op 14981) reg expression
s_mov_b32 s53, s43 // (op 14981) reg expression
s_mov_b32 s54, s18 // (op 14981) reg expression
s_mov_b32 s55, 131072 // (op 14981) call()

v_and_b32 v13, 63, v1 // (op 15000) 
s_mul_i32 s15, s16, s50 // (op 15000) 
s_add_u32 s15, s15, s0 // (op 15000) 
v_lshrrev_b32 v22, 4, v13 // (op 15000) 
s_add_u32 s6, s15, s9 // (op 15000) 

v_and_b32 v23, 3, v22 // (op 15000) 

s_mul_hi_u32 s15, s6, s13 // (op 15000) 
v_and_b32 v22, 15, v1 // (op 15000) 
s_sub_u32 s7, s6, s15 // (op 15000) 
v_lshl_add_u32 v24, v23, 4, v22 // (op 15000) 


s_lshr_b32 s8, s7, 1 // (op 15000) 

v_lshrrev_b32 v22, 7, v1 // (op 15000) 
s_add_u32 s7, s8, s15 // (op 15000) 


v_lshl_add_u32 v23, v22, 6, v24 // (op 15000) 


s_lshr_b32 s8, s7, s11 // (op 15000) 

v_lshrrev_b32 v22, 6, v13 // (op 15000) 

s_and_b32 s7, s11, 2147483648 // (op 15000) 
v_lshlrev_b32 v13, 2, v22 // (op 15000) 

s_cmp_eq_i32 s7, 0 // (op 15000) 
s_mov_b32 s15, scc // (op 15000) 

s_cmp_eq_i32 s15, 0 // (op 15000) 
s_cselect_b32 s7, s6, s8 // (op 15000) 


s_mul_hi_u32 s8, s7, s12 // (op 15000) 
s_sub_u32 s15, s7, s8 // (op 15000) 
s_lshr_b32 s18, s15, 1 // (op 15000) 

s_add_u32 s15, s18, s8 // (op 15000) 


s_lshr_b32 s8, s15, s10 // (op 15000) 

s_and_b32 s15, s10, 2147483648 // (op 15000) 
s_cmp_eq_i32 s15, 0 // (op 15000) 
s_mov_b32 s18, scc // (op 15000) 

s_cmp_eq_i32 s18, 0 // (op 15000) 
s_cselect_b32 s15, s7, s8 // (op 15000) 


v_lshl_add_u32 v22, s15, 7, v23 // (op 15000) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 15000) 
s_mul_i32 s8, s7, s18 // (op 15000) 


s_sub_u32 s7, s6, s8 // (op 15000) 


v_lshl_add_u32 v23, s7, 2, v13 // (op 15000) 


v_mul_lo_u32 v13, v22, s40 // (op 15000) 
v_add_u32 v13, v13, v23 // (op 15000) 


s_mul_i32 s19, s40, s35 // (op 15002) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 15002) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 15002) least significant
s_mul_hi_u32 s6, s40, s34 // (op 15002) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 15002) most significant: sum
s_add_u32 s19, s19, s6 // (op 15002) most significant: sum


s_mov_b32 s52, s42 // (op 15002) reg expression
s_mov_b32 s53, s43 // (op 15002) reg expression
s_mov_b32 s54, s18 // (op 15002) reg expression
s_mov_b32 s55, 131072 // (op 15002) call()

v_and_b32 v22, 63, v1 // (op 15031) 
s_mul_i32 s15, s16, s50 // (op 15031) 
s_add_u32 s15, s15, s0 // (op 15031) 
v_lshrrev_b32 v23, 4, v22 // (op 15031) 
s_add_u32 s6, s15, s9 // (op 15031) 

v_and_b32 v24, 3, v23 // (op 15031) 

s_mul_hi_u32 s15, s6, s13 // (op 15031) 
v_and_b32 v23, 15, v1 // (op 15031) 
s_sub_u32 s7, s6, s15 // (op 15031) 
v_lshl_add_u32 v25, v24, 4, v23 // (op 15031) 


s_lshr_b32 s8, s7, 1 // (op 15031) 

v_lshrrev_b32 v23, 6, v1 // (op 15031) 
s_add_u32 s7, s8, s15 // (op 15031) 


v_and_b32 v24, 1, v23 // (op 15031) 

s_lshr_b32 s8, s7, s11 // (op 15031) 

v_lshl_add_u32 v23, v24, 6, v25 // (op 15031) 


s_and_b32 s7, s11, 2147483648 // (op 15031) 
v_lshrrev_b32 v24, 6, v22 // (op 15031) 

s_cmp_eq_i32 s7, 0 // (op 15031) 
s_mov_b32 s15, scc // (op 15031) 

v_lshlrev_b32 v22, 2, v24 // (op 15031) 

s_cmp_eq_i32 s15, 0 // (op 15031) 
s_cselect_b32 s7, s6, s8 // (op 15031) 


s_mul_hi_u32 s8, s7, s12 // (op 15031) 
s_sub_u32 s15, s7, s8 // (op 15031) 
s_lshr_b32 s18, s15, 1 // (op 15031) 

s_add_u32 s15, s18, s8 // (op 15031) 


s_lshr_b32 s8, s15, s10 // (op 15031) 

s_and_b32 s15, s10, 2147483648 // (op 15031) 
s_cmp_eq_i32 s15, 0 // (op 15031) 
s_mov_b32 s18, scc // (op 15031) 

s_cmp_eq_i32 s18, 0 // (op 15031) 
s_cselect_b32 s15, s7, s8 // (op 15031) 


s_mul_i32 s8, s15, s14 // (op 15031) 

s_sub_u32 s15, s7, s8 // (op 15031) 

v_lshl_add_u32 v24, s15, 7, v23 // (op 15031) 


v_mul_lo_u32 v23, v24, s44 // (op 15031) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 15031) 
s_mul_i32 s8, s7, s18 // (op 15031) 


s_sub_u32 s7, s6, s8 // (op 15031) 


v_lshl_add_u32 v24, s7, 2, v22 // (op 15031) 


v_add_u32 v22, v24, v23 // (op 15031) 


s_mul_i32 s19, s44, s23 // (op 15033) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 15033) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 15033) least significant
s_mul_hi_u32 s6, s44, s22 // (op 15033) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 15033) most significant: sum
s_add_u32 s19, s19, s6 // (op 15033) most significant: sum


s_mov_b32 s64, s46 // (op 15033) reg expression
s_mov_b32 s65, s47 // (op 15033) reg expression
s_mov_b32 s66, s18 // (op 15033) reg expression
s_mov_b32 s67, 131072 // (op 15033) call()

v_and_b32 v23, 63, v1 // (op 15052) 
s_mul_i32 s15, s16, s50 // (op 15052) 
s_add_u32 s15, s15, s0 // (op 15052) 
v_lshrrev_b32 v24, 4, v23 // (op 15052) 
s_add_u32 s6, s15, s9 // (op 15052) 

v_and_b32 v25, 3, v24 // (op 15052) 

s_mul_hi_u32 s15, s6, s13 // (op 15052) 
v_and_b32 v24, 15, v1 // (op 15052) 
s_sub_u32 s7, s6, s15 // (op 15052) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 15052) 


s_lshr_b32 s8, s7, 1 // (op 15052) 

v_lshrrev_b32 v24, 6, v1 // (op 15052) 
s_add_u32 s7, s8, s15 // (op 15052) 


v_and_b32 v25, 1, v24 // (op 15052) 

s_lshr_b32 s8, s7, s11 // (op 15052) 

v_lshl_add_u32 v24, v25, 6, v26 // (op 15052) 


s_and_b32 s7, s11, 2147483648 // (op 15052) 
v_lshrrev_b32 v25, 6, v23 // (op 15052) 

s_cmp_eq_i32 s7, 0 // (op 15052) 
s_mov_b32 s15, scc // (op 15052) 

v_lshlrev_b32 v23, 2, v25 // (op 15052) 

s_cmp_eq_i32 s15, 0 // (op 15052) 
s_cselect_b32 s7, s6, s8 // (op 15052) 


s_mul_hi_u32 s8, s7, s12 // (op 15052) 
s_sub_u32 s15, s7, s8 // (op 15052) 
s_lshr_b32 s18, s15, 1 // (op 15052) 

s_add_u32 s15, s18, s8 // (op 15052) 


s_lshr_b32 s8, s15, s10 // (op 15052) 

s_and_b32 s15, s10, 2147483648 // (op 15052) 
s_cmp_eq_i32 s15, 0 // (op 15052) 
s_mov_b32 s18, scc // (op 15052) 

s_cmp_eq_i32 s18, 0 // (op 15052) 
s_cselect_b32 s15, s7, s8 // (op 15052) 


s_mul_i32 s8, s15, s14 // (op 15052) 

s_sub_u32 s15, s7, s8 // (op 15052) 

v_lshl_add_u32 v25, s15, 7, v24 // (op 15052) 


v_mul_lo_u32 v24, v25, s44 // (op 15052) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 15052) 
s_mul_i32 s8, s7, s18 // (op 15052) 


s_sub_u32 s7, s6, s8 // (op 15052) 


v_lshl_add_u32 v25, s7, 2, v23 // (op 15052) 


v_add_u32 v23, v25, v24 // (op 15052) 


s_mul_i32 s19, s44, s23 // (op 15054) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 15054) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 15054) least significant
s_mul_hi_u32 s6, s44, s22 // (op 15054) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 15054) most significant: sum
s_add_u32 s19, s19, s6 // (op 15054) most significant: sum


s_mov_b32 s64, s46 // (op 15054) reg expression
s_mov_b32 s65, s47 // (op 15054) reg expression
s_mov_b32 s66, s18 // (op 15054) reg expression
s_mov_b32 s67, 131072 // (op 15054) call()

v_and_b32 v24, -4, v1 // (op 14694) 
v_sub_u32 v25, v1, v24 // (op 14694) 

v_lshlrev_b32 v24, 5, v25 // (op 14694) 

v_lshrrev_b32 v25, 2, v1 // (op 14694) 
v_lshl_add_u32 v26, v25, 7, v24 // (op 14694) 


v_lshlrev_b32 v24, 2, v26 // (op 14694) 

v_lshrrev_b32 v25, 3, v24 // (op 14694) 

s_nop 0
v_readfirstlane_b32 s15, v25 // (op 14694) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15499) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 14004) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 14004) 

v_lshrrev_b64 v[24:25], 3, s[28:29] // (op 14004) 


v_mov_b64 v[26:27], v[16:17] // (op 14004) 
s_mov_b32 m0, s15 // (op 14004) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 14004) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 14004) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 14004) most significant half

s_add_u32 m0, m0, 4096 // (op 14004) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 14004) Load value direct to lds



v_and_b32 v24, -8, v1 // (op 14619) 
v_sub_u32 v25, v1, v24 // (op 14619) 

v_lshlrev_b32 v24, 4, v25 // (op 14619) 

v_lshrrev_b32 v25, 3, v1 // (op 14619) 
v_lshl_add_u32 v26, v25, 7, v24 // (op 14619) 


s_nop 0
v_readfirstlane_b32 s15, v26 // (op 14619) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15480) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 14009) 
v_mov_b64 v[24:25], s[18:19] // (op 14009) convert


v_mov_b64 v[26:27], v[18:19] // (op 14009) 
s_add_u32 s6, 8192, s15 // (op 14009) 
s_mov_b32 m0, s6 // (op 14009) reg expression
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 14009) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 14009) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 14009) most significant half

s_add_u32 m0, m0, 4096 // (op 14009) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 14009) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 14009) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 14009) most significant half

s_add_u32 m0, m0, 4096 // (op 14009) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 14009) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 14009) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 14009) most significant half

s_add_u32 m0, m0, 4096 // (op 14009) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 14009) Load value direct to lds






v_mov_b32 v24, v22 // (op 174) reg expression

buffer_load_dwordx2 v[26:27], v24, s[64:67], 0 offen // (op 174) Load value




v_mov_b32 v22, v3 // (op 172) reg expression

buffer_load_dwordx2 v[24:25], v22, s[52:55], 0 offen // (op 172) Load value


v_and_b32 v3, -4, v1 // (op 14852) 
v_sub_u32 v22, v1, v3 // (op 14852) 

v_lshlrev_b32 v3, 5, v22 // (op 14852) 

v_lshrrev_b32 v22, 2, v1 // (op 14852) 
v_lshl_add_u32 v28, v22, 7, v3 // (op 14852) 


v_lshlrev_b32 v3, 2, v28 // (op 14852) 

v_lshrrev_b32 v22, 3, v3 // (op 14852) 

s_nop 0
v_readfirstlane_b32 s15, v22 // (op 14852) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15528) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 14014) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 14014) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 14014) 


v_mov_b64 v[30:31], v[16:17] // (op 14014) 
s_add_u32 s6, 24576, s15 // (op 14014) 
s_mov_b32 m0, s6 // (op 14014) reg expression
buffer_load_dwordx4 v30, s[72:75], 64 offen offset: 0 lds // (op 14014) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 14014) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 14014) most significant half

s_add_u32 m0, m0, 4096 // (op 14014) 
buffer_load_dwordx4 v30, s[72:75], 64 offen offset: 0 lds // (op 14014) Load value direct to lds





v_and_b32 v3, -8, v1 // (op 14808) 
v_sub_u32 v22, v1, v3 // (op 14808) 

v_lshlrev_b32 v3, 4, v22 // (op 14808) 

v_lshrrev_b32 v22, 3, v1 // (op 14808) 
v_lshl_add_u32 v16, v22, 7, v3 // (op 14808) 


s_nop 0
v_readfirstlane_b32 s15, v16 // (op 14808) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 15518) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 14016) 
v_mov_b64 v[16:17], s[18:19] // (op 14016) convert


v_mov_b64 v[28:29], v[18:19] // (op 14016) 
s_add_u32 s6, 32768, s15 // (op 14016) 
s_mov_b32 m0, s6 // (op 14016) reg expression
s_mov_b32 s7, 128 // (op 14016) call()
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 14016) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v16 // (op 14016) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 14016) most significant half

s_add_u32 m0, m0, 4096 // (op 14016) 
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 14016) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v16 // (op 14016) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 14016) most significant half

s_add_u32 m0, m0, 4096 // (op 14016) 
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 14016) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v16 // (op 14016) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 14016) most significant half

s_add_u32 m0, m0, 4096 // (op 14016) 
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 14016) Load value direct to lds






s_waitcnt vmcnt(12)
s_barrier  // (op 15496) 



ds_read_b128 v[16:19], v12 // (op 13498) Load local data 



ds_read_b128 v[28:31], v12 offset:1024 // (op 13503) Load local data 



ds_read_b128 v[32:35], v12 offset:2048 // (op 13508) Load local data 



ds_read_b128 v[36:39], v12 offset:3072 // (op 13513) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 15477) 



ds_read_b128 v[40:43], v10 offset:8192 // (op 13478) Load local data 
ds_read_b128 v[44:47], v10 offset:8256 // (op 13478) Load local data 



ds_read_b128 v[48:51], v10 offset:10240 // (op 13483) Load local data 
ds_read_b128 v[52:55], v10 offset:10304 // (op 13483) Load local data 



ds_read_b128 v[56:59], v10 offset:12288 // (op 13488) Load local data 
ds_read_b128 v[60:63], v10 offset:12352 // (op 13488) Load local data 



ds_read_b128 v[64:67], v10 offset:14336 // (op 13493) Load local data 
ds_read_b128 v[68:71], v10 offset:14400 // (op 13493) Load local data 
v_and_b32 v3, -8, v1 // (op 14604) 
v_sub_u32 v22, v1, v3 // (op 14604) 

v_lshlrev_b32 v3, 4, v22 // (op 14604) 

v_lshrrev_b32 v22, 3, v1 // (op 14604) 
v_lshl_add_u32 v72, v22, 7, v3 // (op 14604) 


s_nop 0
v_readfirstlane_b32 s15, v72 // (op 14604) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 14679) 
v_sub_u32 v22, v1, v3 // (op 14679) 

v_lshlrev_b32 v3, 5, v22 // (op 14679) 

v_lshrrev_b32 v22, 2, v1 // (op 14679) 
v_lshl_add_u32 v72, v22, 7, v3 // (op 14679) 


v_lshlrev_b32 v3, 2, v72 // (op 14679) 

v_lshrrev_b32 v22, 3, v3 // (op 14679) 

s_nop 0
v_readfirstlane_b32 s6, v22 // (op 14679) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -8, v1 // (op 14793) 
v_sub_u32 v22, v1, v3 // (op 14793) 

v_lshlrev_b32 v3, 4, v22 // (op 14793) 

v_lshrrev_b32 v22, 3, v1 // (op 14793) 
v_lshl_add_u32 v72, v22, 7, v3 // (op 14793) 


s_nop 0
v_readfirstlane_b32 s7, v72 // (op 14793) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 14837) 
v_sub_u32 v22, v1, v3 // (op 14837) 

v_lshlrev_b32 v3, 5, v22 // (op 14837) 

v_lshrrev_b32 v22, 2, v1 // (op 14837) 
v_lshl_add_u32 v72, v22, 7, v3 // (op 14837) 


v_lshlrev_b32 v3, 2, v72 // (op 14837) 

v_lshrrev_b32 v22, 3, v3 // (op 14837) 

s_nop 0
v_readfirstlane_b32 s8, v22 // (op 14837) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 1122) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1121) 
s_mov_b32 s28, s18 // (op 1121) convert

s_and_b32 s29, s28, 4294967294 // (op 1121) 

s_cmp_lt_u32 s1, s29 // (op 1121) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1121 // (op 1121) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1121 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1121:
s_add_u32 s1, s1, 2 // (op 1123) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 15502) 
s_waitcnt vmcnt(7)
v_mov_b32 v72, v26 // (op 13894) reg expression

v_mov_b32 v73, v27 // (op 13894) reg expression
s_lshl_b64 s[18:19], s[36:37], 6 // (op 13955) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 13955) 

v_lshrrev_b64 v[74:75], 3, s[28:29] // (op 13955) 


v_mov_b64 v[76:77], v[14:15] // (op 13955) 
v_mov_b64 v[78:79], v[76:77] // (op 13955) 
s_mov_b32 m0, s6 // (op 13955) reg expression
s_mov_b32 s18, 128 // (op 13955) call()
buffer_load_dwordx4 v78, s[72:75], s18 offen offset: 0 lds // (op 13955) Load value direct to lds
v_add_co_u32 v78, s[28:29], v78, v74 // (op 13955) least significant half
v_addc_co_u32 v79, s[28:29], 0, v79, s[28:29] // (op 13955) most significant half

s_add_u32 m0, m0, 4096 // (op 13955) 
v_bfe_u32 v80, v73, 0, 8 // (op 12838) 
v_bfe_u32 v81, v73, 8, 8 // (op 12838) 
v_bfe_u32 v82, v73, 16, 8 // (op 12838) 
v_bfe_u32 v83, v73, 24, 8 // (op 12838) 
v_permlane16_swap_b32 v80, v81 // (op 12838) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 12838) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 12838) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 12838) 
s_waitcnt vmcnt(7)
v_mov_b32 v84, v24 // (op 13908) reg expression
v_mov_b32 v85, v25 // (op 13908) reg expression
buffer_load_dwordx4 v78, s[72:75], s18 offen offset: 0 lds // (op 13955) Load value direct to lds



v_bfe_u32 v86, v84, 0, 8 // (op 12760) 
v_bfe_u32 v87, v84, 8, 8 // (op 12760) 
v_bfe_u32 v88, v84, 16, 8 // (op 12760) 
v_bfe_u32 v89, v84, 24, 8 // (op 12760) 
v_permlane16_swap_b32 v86, v87 // (op 12760) 
v_bfe_u32 v90, v85, 0, 8 // (op 12775) 
v_permlane16_swap_b32 v88, v89 // (op 12760) 
v_bfe_u32 v91, v85, 8, 8 // (op 12775) 
v_bfe_u32 v92, v85, 16, 8 // (op 12775) 
v_permlane32_swap_b32 v86, v88 // (op 12760) 
v_permlane32_swap_b32 v87, v89 // (op 12760) 
v_bfe_u32 v93, v85, 24, 8 // (op 12775) 
v_permlane16_swap_b32 v90, v91 // (op 12775) 
s_nop 0
v_permlane16_swap_b32 v92, v93 // (op 12775) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v90, v92 // (op 12775) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v91, v93 // (op 12775) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 15483) 


s_lshl_b64 s[18:19], s[30:31], 5 // (op 13950) 
v_mov_b64 v[74:75], s[18:19] // (op 13950) convert


v_mov_b64 v[78:79], v[20:21] // (op 13950) 
v_mov_b64 v[84:85], v[78:79] // (op 13950) 
s_add_u32 s18, 8192, s15 // (op 13950) 
s_mov_b32 m0, s18 // (op 13950) reg expression
s_mov_b32 s19, 256 // (op 13950) call()
buffer_load_dwordx4 v84, s[68:71], s19 offen offset: 0 lds // (op 13950) Load value direct to lds
v_add_co_u32 v84, s[28:29], v84, v74 // (op 13950) least significant half
v_addc_co_u32 v85, s[28:29], 0, v85, s[28:29] // (op 13950) most significant half

s_add_u32 m0, m0, 4096 // (op 13950) 
v_bfe_u32 v94, v72, 0, 8 // (op 12823) 
v_bfe_u32 v95, v72, 8, 8 // (op 12823) 
v_bfe_u32 v96, v72, 16, 8 // (op 12823) 
v_bfe_u32 v97, v72, 24, 8 // (op 12823) 
v_permlane16_swap_b32 v94, v95 // (op 12823) 
s_nop 0
v_permlane16_swap_b32 v96, v97 // (op 12823) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v94, v96 // (op 12823) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v95, v97 // (op 12823) 
buffer_load_dwordx4 v84, s[68:71], s19 offen offset: 0 lds // (op 13950) Load value direct to lds

v_add_co_u32 v84, s[28:29], v84, v74 // (op 13950) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[16:19], a[0:3], v86, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 175) 
v_addc_co_u32 v85, s[28:29], 0, v85, s[28:29] // (op 13950) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:47], v[28:31], a[4:7], v86, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 577) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[32:35], a[8:11], v86, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 713) 
s_add_u32 m0, m0, 4096 // (op 13950) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:47], v[36:39], a[12:15], v86, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 849) 
buffer_load_dwordx4 v84, s[68:71], s19 offen offset: 0 lds // (op 13950) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:55], v[16:19], a[16:19], v87, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1269) 
v_add_co_u32 v84, s[28:29], v84, v74 // (op 13950) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:55], v[28:31], a[20:23], v87, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1231) 
v_addc_co_u32 v85, s[28:29], 0, v85, s[28:29] // (op 13950) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:55], v[32:35], a[24:27], v87, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1193) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:55], v[36:39], a[28:31], v87, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1155) 
s_add_u32 m0, m0, 4096 // (op 13950) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:63], v[16:19], a[32:35], v88, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1879) 
buffer_load_dwordx4 v84, s[68:71], s19 offen offset: 0 lds // (op 13950) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[56:63], v[28:31], a[36:39], v88, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1841) 
v_mov_b32 v3, v23 // (op 13903) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[56:63], v[32:35], a[40:43], v88, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1803) 
v_add_u32 v22, 8, v3 // (op 13903) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:63], v[36:39], a[44:47], v88, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1765) 
s_waitcnt vmcnt(10)
s_barrier  // (op 15524) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[64:71], v[16:19], a[48:51], v89, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2489) 
ds_read_b128 v[72:75], v8 offset:24576 // (op 478) Load local data 
buffer_load_dwordx2 v[26:27], v22, s[64:67], 0 offen // (op 13903) Load value





v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[64:71], v[28:31], a[52:55], v89, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2451) 
v_add_u32 v23, 8, v23 // (op 15042) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:71], v[32:35], a[56:59], v89, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2413) 
v_mov_b32 v3, v13 // (op 13917) 
v_add_u32 v22, 8, v3 // (op 13917) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[64:71], v[36:39], a[60:63], v89, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2375) 
ds_read_b128 v[98:101], v8 offset:25600 // (op 586) Load local data 





ds_read_b128 v[94:97], v8 offset:26624 // (op 722) Load local data 
buffer_load_dwordx2 v[24:25], v22, s[52:55], 0 offen // (op 13917) Load value





v_add_u32 v13, 8, v13 // (op 14990) 
ds_read_b128 v[84:87], v8 offset:27648 // (op 858) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 15515) 



ds_read_b128 v[40:43], v7 offset:32768 // (op 476) Load local data 
ds_read_b128 v[44:47], v7 offset:32832 // (op 476) Load local data 



ds_read_b128 v[48:51], v7 offset:34816 // (op 1163) Load local data 
ds_read_b128 v[52:55], v7 offset:34880 // (op 1163) Load local data 



ds_read_b128 v[56:59], v7 offset:36864 // (op 1773) Load local data 
ds_read_b128 v[60:63], v7 offset:36928 // (op 1773) Load local data 



ds_read_b128 v[102:105], v7 offset:38912 // (op 2383) Load local data 
ds_read_b128 v[106:109], v7 offset:38976 // (op 2383) Load local data 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[72:75], a[0:3], v90, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 483) 
s_barrier  // (op 15531) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:47], v[98:101], a[4:7], v90, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 578) 
s_lshl_b64 s[18:19], s[36:37], 6 // (op 13977) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:47], v[84:87], a[12:15], v90, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 850) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 13977) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[94:97], a[8:11], v90, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 714) 
v_lshrrev_b64 v[88:89], 3, s[28:29] // (op 13977) 


v_mov_b64 v[110:111], v[76:77] // (op 13977) 
s_add_u32 s18, 24576, s8 // (op 13977) 
s_mov_b32 m0, s18 // (op 13977) reg expression
s_mov_b32 s19, 192 // (op 13977) call()
buffer_load_dwordx4 v110, s[72:75], s19 offen offset: 0 lds // (op 13977) Load value direct to lds
v_add_co_u32 v110, s[28:29], v110, v88 // (op 13977) least significant half
v_addc_co_u32 v111, s[28:29], 0, v111, s[28:29] // (op 13977) most significant half

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[48:55], v[72:75], a[16:19], v91, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1270) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[48:55], v[98:101], a[20:23], v91, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1232) 
s_add_u32 m0, m0, 4096 // (op 13977) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[48:55], v[94:97], a[24:27], v91, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1194) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[48:55], v[84:87], a[28:31], v91, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1156) 
s_waitcnt vmcnt(7)
s_barrier  // (op 15508) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:63], v[72:75], a[32:35], v92, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1880) 
ds_read_b128 v[16:19], v6 // (op 170) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[56:63], v[98:101], a[36:39], v92, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1842) 
buffer_load_dwordx4 v110, s[72:75], s19 offen offset: 0 lds // (op 13977) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[56:63], v[94:97], a[40:43], v92, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1804) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:63], v[84:87], a[44:47], v92, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1766) 
v_add_co_u32 v14, vcc, 128, v14 // (op 14265) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 14265) most significant half
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:109], v[72:75], a[48:51], v93, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2490) 
s_barrier  // (op 15521) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[102:109], v[98:101], a[52:55], v93, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2452) 
ds_read_b128 v[28:31], v6 offset:1024 // (op 574) Load local data 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 13972) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:109], v[94:97], a[56:59], v93, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2414) 
ds_read_b128 v[32:35], v6 offset:2048 // (op 710) Load local data 

v_mov_b64 v[88:89], s[18:19] // (op 13972) convert





v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:109], v[84:87], a[60:63], v93, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2376) 
ds_read_b128 v[36:39], v6 offset:3072 // (op 846) Load local data 




v_mov_b64 v[72:73], v[78:79] // (op 13972) 
s_add_u32 s18, 32768, s7 // (op 13972) 
s_mov_b32 m0, s18 // (op 13972) reg expression
s_mov_b32 s19, 384 // (op 13972) call()
s_waitcnt vmcnt(4)
s_barrier  // (op 15489) 
buffer_load_dwordx4 v72, s[68:71], s19 offen offset: 0 lds // (op 13972) Load value direct to lds



v_add_co_u32 v72, s[28:29], v72, v88 // (op 13972) least significant half
v_addc_co_u32 v73, s[28:29], 0, v73, s[28:29] // (op 13972) most significant half

s_add_u32 m0, m0, 4096 // (op 13972) 
ds_read_b128 v[40:43], v5 offset:8192 // (op 168) Load local data 
ds_read_b128 v[44:47], v5 offset:8256 // (op 168) Load local data 
buffer_load_dwordx4 v72, s[68:71], s19 offen offset: 0 lds // (op 13972) Load value direct to lds



ds_read_b128 v[48:51], v5 offset:10240 // (op 1151) Load local data 
ds_read_b128 v[52:55], v5 offset:10304 // (op 1151) Load local data 
v_add_co_u32 v72, s[28:29], v72, v88 // (op 13972) least significant half



ds_read_b128 v[56:59], v5 offset:12288 // (op 1761) Load local data 
v_addc_co_u32 v73, s[28:29], 0, v73, s[28:29] // (op 13972) most significant half

s_add_u32 m0, m0, 4096 // (op 13972) 
ds_read_b128 v[60:63], v5 offset:12352 // (op 1761) Load local data 
buffer_load_dwordx4 v72, s[68:71], s19 offen offset: 0 lds // (op 13972) Load value direct to lds



ds_read_b128 v[64:67], v5 offset:14336 // (op 2371) Load local data 
ds_read_b128 v[68:71], v5 offset:14400 // (op 2371) Load local data 
v_add_co_u32 v72, s[28:29], v72, v88 // (op 13972) least significant half
v_addc_co_u32 v73, s[28:29], 0, v73, s[28:29] // (op 13972) most significant half

s_add_u32 m0, m0, 4096 // (op 13972) 
buffer_load_dwordx4 v72, s[68:71], s19 offen offset: 0 lds // (op 13972) Load value direct to lds





v_add_co_u32 v20, vcc, 256, v20 // (op 14119) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 14119) most significant half
s_ashr_i64 s[18:19], s[2:3], 7 // (op 1121) 
s_mov_b32 s28, s18 // (op 1121) convert

s_and_b32 s29, s28, 4294967294 // (op 1121) 

s_cmp_lt_u32 s1, s29 // (op 1121) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1121 // (op 1121) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoop_1121 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_1121:










s_barrier  // (op 15632) 
v_lshrrev_b32 v3, 3, v1 // (op 14043) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 14043) 
v_and_b32 v13, -8, v1 // (op 14043) 
s_and_b32 s15, s18, 4294967294 // (op 14043) 
v_sub_u32 v14, v1, v13 // (op 14043) 

s_add_u32 s6, s0, s15 // (op 14043) 

v_lshlrev_b32 v13, 4, v14 // (op 14043) 

s_mul_i32 s15, s16, s50 // (op 14043) 
s_add_u32 s15, s15, s6 // (op 14043) 

s_add_u32 s6, s15, s9 // (op 14043) 

s_mul_hi_u32 s15, s6, s13 // (op 14043) 
s_sub_u32 s7, s6, s15 // (op 14043) 
s_lshr_b32 s8, s7, 1 // (op 14043) 

s_add_u32 s7, s8, s15 // (op 14043) 


s_lshr_b32 s8, s7, s11 // (op 14043) 

s_and_b32 s7, s11, 2147483648 // (op 14043) 
s_cmp_eq_i32 s7, 0 // (op 14043) 
s_mov_b32 s15, scc // (op 14043) 

s_cmp_eq_i32 s15, 0 // (op 14043) 
s_cselect_b32 s7, s6, s8 // (op 14043) 


s_mul_hi_u32 s8, s7, s12 // (op 14043) 
s_sub_u32 s15, s7, s8 // (op 14043) 
s_lshr_b32 s28, s15, 1 // (op 14043) 

s_add_u32 s15, s28, s8 // (op 14043) 


s_lshr_b32 s8, s15, s10 // (op 14043) 

s_and_b32 s15, s10, 2147483648 // (op 14043) 
s_cmp_eq_i32 s15, 0 // (op 14043) 
s_mov_b32 s28, scc // (op 14043) 

s_cmp_eq_i32 s28, 0 // (op 14043) 
s_cselect_b32 s15, s7, s8 // (op 14043) 


s_mul_i32 s8, s7, s18 // (op 14043) 


v_lshl_add_u32 v14, s15, 7, v3 // (op 14043) 


s_sub_u32 s7, s6, s8 // (op 14043) 


v_lshl_add_u32 v3, s7, 7, v13 // (op 14043) 


v_mov_b32 v21, 0 // (op 14043) convert
v_mov_b32 v20, v3 // (op 14043) convert

v_mul_lo_u32 v23, v14, s31 // (op 14043) most significant: low of low * high
v_mul_hi_u32 v3, v14, s30 // (op 14043) most significant: high of low * low
v_mul_lo_u32 v22, v14, s30 // (op 14043) least significant: low of low * low
v_add_u32 v23, v23, v3 // (op 14043) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v20 // (op 14043) least significant half
v_addc_co_u32 v23, s[18:19], v23, v21, s[18:19] // (op 14043) most significant half



s_mul_i32 s19, s30, s35 // (op 14045) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 14045) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 14045) least significant
s_mul_hi_u32 s6, s30, s34 // (op 14045) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14045) most significant: sum
s_add_u32 s19, s19, s6 // (op 14045) most significant: sum


s_mov_b32 s68, s32 // (op 14045) reg expression
s_mov_b32 s69, s33 // (op 14045) reg expression
s_mov_b32 s70, s18 // (op 14045) reg expression
s_mov_b32 s71, 131072 // (op 14045) call()

v_lshrrev_b32 v3, 2, v1 // (op 14209) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 14209) 
v_and_b32 v13, -4, v1 // (op 14209) 
s_and_b32 s15, s18, 4294967294 // (op 14209) 
v_sub_u32 v14, v1, v13 // (op 14209) 

s_add_u32 s6, s0, s15 // (op 14209) 

v_lshlrev_b32 v13, 5, v14 // (op 14209) 

s_mul_i32 s15, s16, s50 // (op 14209) 
s_add_u32 s15, s15, s6 // (op 14209) 

s_add_u32 s6, s15, s9 // (op 14209) 

s_mul_hi_u32 s15, s6, s13 // (op 14209) 
s_sub_u32 s7, s6, s15 // (op 14209) 
s_lshr_b32 s8, s7, 1 // (op 14209) 

s_add_u32 s7, s8, s15 // (op 14209) 


s_lshr_b32 s8, s7, s11 // (op 14209) 

s_and_b32 s7, s11, 2147483648 // (op 14209) 
s_cmp_eq_i32 s7, 0 // (op 14209) 
s_mov_b32 s15, scc // (op 14209) 

s_cmp_eq_i32 s15, 0 // (op 14209) 
s_cselect_b32 s7, s6, s8 // (op 14209) 


s_mul_hi_u32 s8, s7, s12 // (op 14209) 
s_sub_u32 s15, s7, s8 // (op 14209) 
s_lshr_b32 s28, s15, 1 // (op 14209) 

s_add_u32 s15, s28, s8 // (op 14209) 


s_lshr_b32 s8, s15, s10 // (op 14209) 

s_and_b32 s15, s10, 2147483648 // (op 14209) 
s_cmp_eq_i32 s15, 0 // (op 14209) 
s_mov_b32 s28, scc // (op 14209) 

s_cmp_eq_i32 s28, 0 // (op 14209) 
s_cselect_b32 s15, s7, s8 // (op 14209) 


s_mul_i32 s8, s15, s14 // (op 14209) 

s_sub_u32 s15, s7, s8 // (op 14209) 

s_mul_i32 s8, s7, s18 // (op 14209) 


v_lshl_add_u32 v14, s15, 7, v3 // (op 14209) 


s_sub_u32 s7, s6, s8 // (op 14209) 


v_mul_lo_u32 v21, v14, s37 // (op 14209) most significant: low of low * high
v_mul_hi_u32 v3, v14, s36 // (op 14209) most significant: high of low * low
v_mul_lo_u32 v20, v14, s36 // (op 14209) least significant: low of low * low
v_add_u32 v21, v21, v3 // (op 14209) most significant: sum


v_lshl_add_u32 v3, s7, 7, v13 // (op 14209) 


v_mov_b32 v15, 0 // (op 14209) convert
v_mov_b32 v14, v3 // (op 14209) convert

v_add_co_u32 v72, s[18:19], v14, v20 // (op 14209) least significant half
v_addc_co_u32 v73, s[18:19], v15, v21, s[18:19] // (op 14209) most significant half

v_lshlrev_b64 v[72:73], 2, v[72:73] // (op 14209) 


v_lshrrev_b64 v[14:15], 3, v[72:73] // (op 14209) 

s_mul_i32 s19, s36, s23 // (op 14211) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 14211) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 14211) least significant
s_mul_hi_u32 s6, s36, s22 // (op 14211) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 14211) most significant: sum
s_add_u32 s19, s19, s6 // (op 14211) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 14211) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 14211) 

s_mov_b32 s72, s38 // (op 14211) reg expression
s_mov_b32 s73, s39 // (op 14211) reg expression
s_mov_b32 s74, s18 // (op 14211) reg expression
s_mov_b32 s75, 131072 // (op 14211) call()

v_and_b32 v3, -8, v1 // (op 14589) 
v_sub_u32 v13, v1, v3 // (op 14589) 

v_lshlrev_b32 v3, 4, v13 // (op 14589) 

v_lshrrev_b32 v13, 3, v1 // (op 14589) 
v_lshl_add_u32 v20, v13, 7, v3 // (op 14589) 


s_nop 0
v_readfirstlane_b32 s15, v20 // (op 14589) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 14666) 
v_sub_u32 v13, v1, v3 // (op 14666) 

v_lshlrev_b32 v3, 5, v13 // (op 14666) 

v_lshrrev_b32 v13, 2, v1 // (op 14666) 
v_lshl_add_u32 v20, v13, 7, v3 // (op 14666) 


v_lshlrev_b32 v3, 2, v20 // (op 14666) 

v_lshrrev_b32 v13, 3, v3 // (op 14666) 

s_nop 0
v_readfirstlane_b32 s6, v13 // (op 14666) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, 63, v1 // (op 15079) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 15079) 
v_lshrrev_b32 v13, 4, v3 // (op 15079) 
s_and_b32 s7, s18, 4294967294 // (op 15079) 
v_and_b32 v20, 3, v13 // (op 15079) 

s_add_u32 s8, s0, s7 // (op 15079) 

v_and_b32 v13, 15, v1 // (op 15079) 
v_lshl_add_u32 v21, v20, 4, v13 // (op 15079) 


s_mul_i32 s7, s16, s50 // (op 15079) 
s_add_u32 s7, s7, s8 // (op 15079) 

v_lshrrev_b32 v13, 7, v1 // (op 15079) 
s_add_u32 s8, s7, s9 // (op 15079) 

v_lshl_add_u32 v20, v13, 6, v21 // (op 15079) 


s_mul_hi_u32 s7, s8, s13 // (op 15079) 
v_lshrrev_b32 v13, 6, v3 // (op 15079) 

s_sub_u32 s28, s8, s7 // (op 15079) 
v_lshlrev_b32 v3, 2, v13 // (op 15079) 

s_lshr_b32 s29, s28, 1 // (op 15079) 

s_add_u32 s28, s29, s7 // (op 15079) 


s_lshr_b32 s7, s28, s11 // (op 15079) 

s_and_b32 s28, s11, 2147483648 // (op 15079) 
s_cmp_eq_i32 s28, 0 // (op 15079) 
s_mov_b32 s29, scc // (op 15079) 

s_cmp_eq_i32 s29, 0 // (op 15079) 
s_cselect_b32 s28, s8, s7 // (op 15079) 


s_mul_hi_u32 s7, s28, s12 // (op 15079) 
s_sub_u32 s29, s28, s7 // (op 15079) 
s_lshr_b32 s48, s29, 1 // (op 15079) 

s_add_u32 s29, s48, s7 // (op 15079) 


s_lshr_b32 s7, s29, s10 // (op 15079) 

s_and_b32 s29, s10, 2147483648 // (op 15079) 
s_cmp_eq_i32 s29, 0 // (op 15079) 
s_mov_b32 s48, scc // (op 15079) 

s_cmp_eq_i32 s48, 0 // (op 15079) 
s_cselect_b32 s29, s28, s7 // (op 15079) 


s_mul_i32 s7, s28, s18 // (op 15079) 


v_lshl_add_u32 v13, s29, 7, v20 // (op 15079) 


s_sub_u32 s18, s8, s7 // (op 15079) 


v_lshl_add_u32 v20, s18, 2, v3 // (op 15079) 


v_mul_lo_u32 v3, v13, s40 // (op 15079) 
v_add_u32 v3, v3, v20 // (op 15079) 


s_mul_i32 s19, s40, s35 // (op 15081) most significant: low of low * high
s_mul_i32 s7, s41, s34 // (op 15081) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 15081) least significant
s_mul_hi_u32 s8, s40, s34 // (op 15081) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 15081) most significant: sum
s_add_u32 s19, s19, s8 // (op 15081) most significant: sum


s_mov_b32 s52, s42 // (op 15081) reg expression
s_mov_b32 s53, s43 // (op 15081) reg expression
s_mov_b32 s54, s18 // (op 15081) reg expression
s_mov_b32 s55, 131072 // (op 15081) call()

v_and_b32 v13, 63, v1 // (op 15106) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 15106) 
v_lshrrev_b32 v20, 4, v13 // (op 15106) 
s_and_b32 s7, s18, 4294967294 // (op 15106) 
v_and_b32 v21, 3, v20 // (op 15106) 

s_add_u32 s8, s0, s7 // (op 15106) 

v_and_b32 v20, 15, v1 // (op 15106) 
v_lshl_add_u32 v72, v21, 4, v20 // (op 15106) 


s_mul_i32 s7, s16, s50 // (op 15106) 
s_add_u32 s7, s7, s8 // (op 15106) 

v_lshrrev_b32 v20, 6, v1 // (op 15106) 
s_add_u32 s8, s7, s9 // (op 15106) 

v_and_b32 v21, 1, v20 // (op 15106) 

s_mul_hi_u32 s7, s8, s13 // (op 15106) 
v_lshl_add_u32 v20, v21, 6, v72 // (op 15106) 


s_sub_u32 s28, s8, s7 // (op 15106) 
v_lshrrev_b32 v21, 6, v13 // (op 15106) 

s_lshr_b32 s29, s28, 1 // (op 15106) 

v_lshlrev_b32 v13, 2, v21 // (op 15106) 

s_add_u32 s28, s29, s7 // (op 15106) 


s_lshr_b32 s7, s28, s11 // (op 15106) 

s_and_b32 s28, s11, 2147483648 // (op 15106) 
s_cmp_eq_i32 s28, 0 // (op 15106) 
s_mov_b32 s29, scc // (op 15106) 

s_cmp_eq_i32 s29, 0 // (op 15106) 
s_cselect_b32 s28, s8, s7 // (op 15106) 


s_mul_hi_u32 s7, s28, s12 // (op 15106) 
s_sub_u32 s29, s28, s7 // (op 15106) 
s_lshr_b32 s48, s29, 1 // (op 15106) 

s_add_u32 s29, s48, s7 // (op 15106) 


s_lshr_b32 s7, s29, s10 // (op 15106) 

s_and_b32 s29, s10, 2147483648 // (op 15106) 
s_cmp_eq_i32 s29, 0 // (op 15106) 
s_mov_b32 s48, scc // (op 15106) 

s_cmp_eq_i32 s48, 0 // (op 15106) 
s_cselect_b32 s29, s28, s7 // (op 15106) 


s_mul_i32 s7, s29, s14 // (op 15106) 

s_sub_u32 s29, s28, s7 // (op 15106) 

v_lshl_add_u32 v21, s29, 7, v20 // (op 15106) 


s_mul_i32 s7, s28, s18 // (op 15106) 


v_mul_lo_u32 v20, v21, s44 // (op 15106) 

s_sub_u32 s18, s8, s7 // (op 15106) 


v_lshl_add_u32 v21, s18, 2, v13 // (op 15106) 


v_add_u32 v13, v21, v20 // (op 15106) 


s_mul_i32 s19, s44, s23 // (op 15108) most significant: low of low * high
s_mul_i32 s7, s45, s22 // (op 15108) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 15108) least significant
s_mul_hi_u32 s8, s44, s22 // (op 15108) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 15108) most significant: sum
s_add_u32 s19, s19, s8 // (op 15108) most significant: sum


s_mov_b32 s64, s46 // (op 15108) reg expression
s_mov_b32 s65, s47 // (op 15108) reg expression
s_mov_b32 s66, s18 // (op 15108) reg expression
s_mov_b32 s67, 131072 // (op 15108) call()

s_ashr_i64 s[18:19], s[2:3], 7 // (op 419) 
s_and_b32 s7, s18, 4294967294 // (op 419) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 418) 
s_mov_b32 s8, s18 // (op 418) convert

s_cmp_lt_u32 s7, s8 // (op 418) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_418 // (op 418) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_418 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_418:
s_barrier  // (op 15486) 
s_add_u32 s7, s7, 1 // (op 420) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 13960) 
v_mov_b64 v[16:17], s[18:19] // (op 13960) convert


v_mov_b64 v[18:19], v[22:23] // (op 13960) 
v_mov_b64 v[20:21], v[18:19] // (op 13960) 
s_add_u32 s8, 8192, s15 // (op 13960) 
s_mov_b32 m0, s8 // (op 13960) reg expression
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 13960) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13960) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13960) most significant half

s_add_u32 m0, m0, 4096 // (op 13960) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 13960) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13960) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13960) most significant half

s_add_u32 m0, m0, 4096 // (op 13960) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 13960) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13960) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13960) most significant half

s_add_u32 m0, m0, 4096 // (op 13960) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 13960) Load value direct to lds




v_add_co_u32 v22, vcc, 128, v22 // (op 14033) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 14033) most significant half
s_barrier  // (op 15505) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 13966) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 13966) 

v_lshrrev_b64 v[16:17], 3, s[28:29] // (op 13966) 


v_mov_b64 v[18:19], v[14:15] // (op 13966) 
v_mov_b64 v[20:21], v[18:19] // (op 13966) 
s_mov_b32 m0, s6 // (op 13966) reg expression
buffer_load_dwordx4 v20, s[72:75], 0 offen offset: 0 lds // (op 13966) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 13966) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 13966) most significant half

s_add_u32 m0, m0, 4096 // (op 13966) 
buffer_load_dwordx4 v20, s[72:75], 0 offen offset: 0 lds // (op 13966) Load value direct to lds



v_add_co_u32 v14, s[18:19], 64, v14 // (op 14201) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 14201) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 15492) 



ds_read_b128 v[24:27], v5 offset:8192 // (op 426) Load local data 
ds_read_b128 v[28:31], v5 offset:8256 // (op 426) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 15511) 



ds_read_b128 v[16:19], v6 // (op 428) Load local data 


v_mov_b32 v20, v3 // (op 430) 
v_mov_b32 v21, v20 // (op 430) reg expression

buffer_load_dwordx2 v[32:33], v21, s[52:55], 0 offen // (op 430) Load value


v_add_u32 v3, 4, v3 // (op 15069) 


v_mov_b32 v20, v13 // (op 432) 
v_mov_b32 v21, v20 // (op 432) reg expression

buffer_load_dwordx2 v[34:35], v21, s[64:67], 0 offen // (op 432) Load value


v_add_u32 v13, 4, v13 // (op 15096) 
s_waitcnt vmcnt(1)
v_bfe_u32 v36, v32, 0, 8 // (op 12887) 



v_bfe_u32 v37, v32, 8, 8 // (op 12887) 
v_bfe_u32 v38, v32, 16, 8 // (op 12887) 
v_bfe_u32 v39, v32, 24, 8 // (op 12887) 
v_permlane16_swap_b32 v36, v37 // (op 12887) 
s_nop 0
v_permlane16_swap_b32 v38, v39 // (op 12887) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v36, v38 // (op 12887) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v37, v39 // (op 12887) 
ds_read_b128 v[40:43], v6 offset:1024 // (op 557) Load local data 




ds_read_b128 v[44:47], v6 offset:2048 // (op 693) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v48, v34, 0, 8 // (op 12917) 



v_bfe_u32 v49, v34, 8, 8 // (op 12917) 
v_bfe_u32 v50, v34, 16, 8 // (op 12917) 
v_bfe_u32 v51, v34, 24, 8 // (op 12917) 
v_permlane16_swap_b32 v48, v49 // (op 12917) 
ds_read_b128 v[52:55], v6 offset:3072 // (op 829) Load local data 
v_permlane16_swap_b32 v50, v51 // (op 12917) 



s_nop 1
v_permlane32_swap_b32 v48, v50 // (op 12917) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v49, v51 // (op 12917) 
ds_read_b128 v[56:59], v5 offset:10240 // (op 1051) Load local data 

ds_read_b128 v[60:63], v5 offset:10304 // (op 1051) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[16:19], a[0:3], v36, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 433) 
ds_read_b128 v[64:67], v5 offset:12288 // (op 1661) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:31], v[40:43], a[4:7], v36, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 560) 
ds_read_b128 v[68:71], v5 offset:12352 // (op 1661) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:31], v[44:47], a[8:11], v36, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 696) 
ds_read_b128 v[72:75], v5 offset:14336 // (op 2271) Load local data 
ds_read_b128 v[76:79], v5 offset:14400 // (op 2271) Load local data 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:31], v[52:55], a[12:15], v36, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 832) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:63], v[16:19], a[16:19], v37, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1055) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:63], v[44:47], a[24:27], v37, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1109) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:63], v[52:55], a[28:31], v37, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1136) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:63], v[40:43], a[20:23], v37, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1082) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:71], v[16:19], a[32:35], v38, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1665) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:71], v[44:47], a[40:43], v38, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1719) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:71], v[52:55], a[44:47], v38, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1746) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:71], v[40:43], a[36:39], v38, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1692) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:79], v[16:19], a[48:51], v39, v48 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2275) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:79], v[44:47], a[56:59], v39, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2329) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[72:79], v[52:55], a[60:63], v39, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2356) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[72:79], v[40:43], a[52:55], v39, v49 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2302) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 418) 
s_mov_b32 s8, s18 // (op 418) convert

s_cmp_lt_u32 s7, s8 // (op 418) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_418 // (op 418) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_418 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_418:









s_barrier  // (op 15614) 


v_cmp_eq_f32 vcc, 0.00000, v0 // (op 203) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_203 // (op 203) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_203
v_accvgpr_read v12, a0 // (op 211) 
v_accvgpr_read v13, a1 // (op 211) 
v_accvgpr_read v14, a2 // (op 211) 
v_accvgpr_read v15, a3 // (op 211) 
v_mul_f32 v16, v11, v12 // (op 211) 
v_mul_f32 v17, v11, v13 // (op 211) 
v_mul_f32 v18, v11, v14 // (op 211) 
v_mul_f32 v19, v11, v15 // (op 211) 



v_mov_b32 v3, v4 // (op 213) reg expression

buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 213) Store value


v_accvgpr_read v12, a4 // (op 3168) 
v_accvgpr_read v13, a5 // (op 3168) 
v_accvgpr_read v14, a6 // (op 3168) 
v_accvgpr_read v15, a7 // (op 3168) 
v_mul_f32 v16, v11, v12 // (op 3168) 
v_mul_f32 v17, v11, v13 // (op 3168) 
v_mul_f32 v18, v11, v14 // (op 3168) 
v_mul_f32 v19, v11, v15 // (op 3168) 



s_lshl_b32 s15, s24, 6 // (op 3170) 
v_add_u32 v3, v4, s15 // (op 3170) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3170) Store value


v_accvgpr_read v12, a8 // (op 3180) 
v_accvgpr_read v13, a9 // (op 3180) 
v_accvgpr_read v14, a10 // (op 3180) 
v_accvgpr_read v15, a11 // (op 3180) 
v_mul_f32 v16, v11, v12 // (op 3180) 
v_mul_f32 v17, v11, v13 // (op 3180) 
v_mul_f32 v18, v11, v14 // (op 3180) 
v_mul_f32 v19, v11, v15 // (op 3180) 



s_lshl_b32 s15, s24, 7 // (op 3182) 
v_add_u32 v3, v4, s15 // (op 3182) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3182) Store value


v_accvgpr_read v12, a12 // (op 3192) 
v_accvgpr_read v13, a13 // (op 3192) 
v_accvgpr_read v14, a14 // (op 3192) 
v_accvgpr_read v15, a15 // (op 3192) 
v_mul_f32 v16, v11, v12 // (op 3192) 
v_mul_f32 v17, v11, v13 // (op 3192) 
v_mul_f32 v18, v11, v14 // (op 3192) 
v_mul_f32 v19, v11, v15 // (op 3192) 



s_lshl_b32 s15, s24, 6 // (op 3194) 
s_mul_i32 s6, 3, s15 // (op 3194) 

v_add_u32 v3, v4, s6 // (op 3194) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3194) Store value


v_accvgpr_read v12, a16 // (op 3226) 
v_accvgpr_read v13, a17 // (op 3226) 
v_accvgpr_read v14, a18 // (op 3226) 
v_accvgpr_read v15, a19 // (op 3226) 
v_mul_f32 v16, v11, v12 // (op 3226) 
v_mul_f32 v17, v11, v13 // (op 3226) 
v_mul_f32 v18, v11, v14 // (op 3226) 
v_mul_f32 v19, v11, v15 // (op 3226) 



v_add_u32 v3, 64, v4 // (op 3229) 

buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3229) Store value


v_accvgpr_read v12, a20 // (op 3239) 
v_accvgpr_read v13, a21 // (op 3239) 
v_accvgpr_read v14, a22 // (op 3239) 
v_accvgpr_read v15, a23 // (op 3239) 
v_mul_f32 v16, v11, v12 // (op 3239) 
v_mul_f32 v17, v11, v13 // (op 3239) 
v_mul_f32 v18, v11, v14 // (op 3239) 
v_mul_f32 v19, v11, v15 // (op 3239) 



s_lshl_b32 s15, s24, 6 // (op 3231) 
s_add_u32 s15, s15, 64 // (op 3231) 
v_add_u32 v3, v4, s15 // (op 3231) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3231) Store value


v_accvgpr_read v12, a24 // (op 3244) 
v_accvgpr_read v13, a25 // (op 3244) 
v_accvgpr_read v14, a26 // (op 3244) 
v_accvgpr_read v15, a27 // (op 3244) 
v_mul_f32 v16, v11, v12 // (op 3244) 
v_mul_f32 v17, v11, v13 // (op 3244) 
v_mul_f32 v18, v11, v14 // (op 3244) 
v_mul_f32 v19, v11, v15 // (op 3244) 



s_lshl_b32 s15, s24, 7 // (op 3233) 
s_add_u32 s15, s15, 64 // (op 3233) 
v_add_u32 v3, v4, s15 // (op 3233) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3233) Store value


v_accvgpr_read v12, a28 // (op 3249) 
v_accvgpr_read v13, a29 // (op 3249) 
v_accvgpr_read v14, a30 // (op 3249) 
v_accvgpr_read v15, a31 // (op 3249) 
v_mul_f32 v16, v11, v12 // (op 3249) 
v_mul_f32 v17, v11, v13 // (op 3249) 
v_mul_f32 v18, v11, v14 // (op 3249) 
v_mul_f32 v19, v11, v15 // (op 3249) 



s_lshl_b32 s15, s24, 6 // (op 3235) 
s_mul_i32 s6, 3, s15 // (op 3235) 
s_add_u32 s6, s6, 64 // (op 3235) 

v_add_u32 v3, v4, s6 // (op 3235) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3235) Store value


v_accvgpr_read v12, a32 // (op 3294) 
v_accvgpr_read v13, a33 // (op 3294) 
v_accvgpr_read v14, a34 // (op 3294) 
v_accvgpr_read v15, a35 // (op 3294) 
v_mul_f32 v16, v11, v12 // (op 3294) 
v_mul_f32 v17, v11, v13 // (op 3294) 
v_mul_f32 v18, v11, v14 // (op 3294) 
v_mul_f32 v19, v11, v15 // (op 3294) 



v_add_u32 v3, 128, v4 // (op 3297) 

buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3297) Store value


v_accvgpr_read v12, a36 // (op 3307) 
v_accvgpr_read v13, a37 // (op 3307) 
v_accvgpr_read v14, a38 // (op 3307) 
v_accvgpr_read v15, a39 // (op 3307) 
v_mul_f32 v16, v11, v12 // (op 3307) 
v_mul_f32 v17, v11, v13 // (op 3307) 
v_mul_f32 v18, v11, v14 // (op 3307) 
v_mul_f32 v19, v11, v15 // (op 3307) 



s_lshl_b32 s15, s24, 6 // (op 3299) 
s_add_u32 s15, s15, 128 // (op 3299) 
v_add_u32 v3, v4, s15 // (op 3299) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3299) Store value


v_accvgpr_read v12, a40 // (op 3312) 
v_accvgpr_read v13, a41 // (op 3312) 
v_accvgpr_read v14, a42 // (op 3312) 
v_accvgpr_read v15, a43 // (op 3312) 
v_mul_f32 v16, v11, v12 // (op 3312) 
v_mul_f32 v17, v11, v13 // (op 3312) 
v_mul_f32 v18, v11, v14 // (op 3312) 
v_mul_f32 v19, v11, v15 // (op 3312) 



s_lshl_b32 s15, s24, 7 // (op 3301) 
s_add_u32 s15, s15, 128 // (op 3301) 
v_add_u32 v3, v4, s15 // (op 3301) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3301) Store value


v_accvgpr_read v12, a44 // (op 3317) 
v_accvgpr_read v13, a45 // (op 3317) 
v_accvgpr_read v14, a46 // (op 3317) 
v_accvgpr_read v15, a47 // (op 3317) 
v_mul_f32 v16, v11, v12 // (op 3317) 
v_mul_f32 v17, v11, v13 // (op 3317) 
v_mul_f32 v18, v11, v14 // (op 3317) 
v_mul_f32 v19, v11, v15 // (op 3317) 



s_lshl_b32 s15, s24, 6 // (op 3303) 
s_mul_i32 s6, 3, s15 // (op 3303) 
s_add_u32 s6, s6, 128 // (op 3303) 

v_add_u32 v3, v4, s6 // (op 3303) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3303) Store value


v_accvgpr_read v12, a48 // (op 3362) 
v_accvgpr_read v13, a49 // (op 3362) 
v_accvgpr_read v14, a50 // (op 3362) 
v_accvgpr_read v15, a51 // (op 3362) 
v_mul_f32 v16, v11, v12 // (op 3362) 
v_mul_f32 v17, v11, v13 // (op 3362) 
v_mul_f32 v18, v11, v14 // (op 3362) 
v_mul_f32 v19, v11, v15 // (op 3362) 



v_add_u32 v3, 192, v4 // (op 3365) 

buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3365) Store value


v_accvgpr_read v12, a52 // (op 3375) 
v_accvgpr_read v13, a53 // (op 3375) 
v_accvgpr_read v14, a54 // (op 3375) 
v_accvgpr_read v15, a55 // (op 3375) 
v_mul_f32 v16, v11, v12 // (op 3375) 
v_mul_f32 v17, v11, v13 // (op 3375) 
v_mul_f32 v18, v11, v14 // (op 3375) 
v_mul_f32 v19, v11, v15 // (op 3375) 



s_lshl_b32 s15, s24, 6 // (op 3367) 
s_add_u32 s15, s15, 192 // (op 3367) 
v_add_u32 v3, v4, s15 // (op 3367) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3367) Store value


v_accvgpr_read v12, a56 // (op 3380) 
v_accvgpr_read v13, a57 // (op 3380) 
v_accvgpr_read v14, a58 // (op 3380) 
v_accvgpr_read v15, a59 // (op 3380) 
v_mul_f32 v16, v11, v12 // (op 3380) 
v_mul_f32 v17, v11, v13 // (op 3380) 
v_mul_f32 v18, v11, v14 // (op 3380) 
v_mul_f32 v19, v11, v15 // (op 3380) 



s_lshl_b32 s15, s24, 7 // (op 3369) 
s_add_u32 s15, s15, 192 // (op 3369) 
v_add_u32 v3, v4, s15 // (op 3369) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3369) Store value


v_accvgpr_read v12, a60 // (op 3385) 
v_accvgpr_read v13, a61 // (op 3385) 
v_accvgpr_read v14, a62 // (op 3385) 
v_accvgpr_read v15, a63 // (op 3385) 
v_mul_f32 v16, v11, v12 // (op 3385) 
v_mul_f32 v17, v11, v13 // (op 3385) 
v_mul_f32 v18, v11, v14 // (op 3385) 
v_mul_f32 v19, v11, v15 // (op 3385) 



s_lshl_b32 s15, s24, 6 // (op 3371) 
s_mul_i32 s6, 3, s15 // (op 3371) 
s_add_u32 s6, s6, 192 // (op 3371) 

v_add_u32 v3, v4, s6 // (op 3371) 


buffer_store_dwordx4 v[16:19], v3, s[60:63], 0 offen // (op 3371) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_203 // (op 203) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_203
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_203:


v_add_u32 v3, 192, v9 // (op 3729) 

buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3729) Load value

v_accvgpr_read v16, a48 // (op 3717) 
v_accvgpr_read v17, a49 // (op 3717) 
v_accvgpr_read v18, a50 // (op 3717) 
v_accvgpr_read v19, a51 // (op 3717) 
v_mul_f32 v20, v11, v16 // (op 3717) 
v_mul_f32 v21, v11, v17 // (op 3717) 
v_mul_f32 v22, v11, v18 // (op 3717) 
v_mul_f32 v23, v11, v19 // (op 3717) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3728) 
v_mul_f32 v17, v0, v13 // (op 3728) 
v_mul_f32 v18, v0, v14 // (op 3728) 
v_mul_f32 v19, v0, v15 // (op 3728) 

v_add_f32 v12, v16, v20 // (op 3718) 
v_add_f32 v13, v17, v21 // (op 3718) 
v_add_f32 v14, v18, v22 // (op 3718) 
v_add_f32 v15, v19, v23 // (op 3718) 




v_add_u32 v3, 192, v4 // (op 3720) 

buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3720) Store value




s_lshl_b32 s15, s4, 6 // (op 3733) 
s_add_u32 s15, s15, 192 // (op 3733) 
v_add_u32 v3, v9, s15 // (op 3733) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3733) Load value

v_accvgpr_read v16, a52 // (op 3743) 
v_accvgpr_read v17, a53 // (op 3743) 
v_accvgpr_read v18, a54 // (op 3743) 
v_accvgpr_read v19, a55 // (op 3743) 
v_mul_f32 v20, v11, v16 // (op 3743) 
v_mul_f32 v21, v11, v17 // (op 3743) 
v_mul_f32 v22, v11, v18 // (op 3743) 
v_mul_f32 v23, v11, v19 // (op 3743) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3731) 
v_mul_f32 v17, v0, v13 // (op 3731) 
v_mul_f32 v18, v0, v14 // (op 3731) 
v_mul_f32 v19, v0, v15 // (op 3731) 

v_add_f32 v12, v16, v20 // (op 3732) 
v_add_f32 v13, v17, v21 // (op 3732) 
v_add_f32 v14, v18, v22 // (op 3732) 
v_add_f32 v15, v19, v23 // (op 3732) 




s_lshl_b32 s15, s24, 6 // (op 3722) 
s_add_u32 s15, s15, 192 // (op 3722) 
v_add_u32 v3, v4, s15 // (op 3722) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3722) Store value




s_lshl_b32 s15, s4, 7 // (op 3737) 
s_add_u32 s15, s15, 192 // (op 3737) 
v_add_u32 v3, v9, s15 // (op 3737) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3737) Load value

v_accvgpr_read v16, a56 // (op 3745) 
v_accvgpr_read v17, a57 // (op 3745) 
v_accvgpr_read v18, a58 // (op 3745) 
v_accvgpr_read v19, a59 // (op 3745) 
v_mul_f32 v20, v11, v16 // (op 3745) 
v_mul_f32 v21, v11, v17 // (op 3745) 
v_mul_f32 v22, v11, v18 // (op 3745) 
v_mul_f32 v23, v11, v19 // (op 3745) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3735) 
v_mul_f32 v17, v0, v13 // (op 3735) 
v_mul_f32 v18, v0, v14 // (op 3735) 
v_mul_f32 v19, v0, v15 // (op 3735) 

v_add_f32 v12, v16, v20 // (op 3736) 
v_add_f32 v13, v17, v21 // (op 3736) 
v_add_f32 v14, v18, v22 // (op 3736) 
v_add_f32 v15, v19, v23 // (op 3736) 




s_lshl_b32 s15, s24, 7 // (op 3724) 
s_add_u32 s15, s15, 192 // (op 3724) 
v_add_u32 v3, v4, s15 // (op 3724) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3724) Store value




s_lshl_b32 s15, s4, 6 // (op 3741) 
s_mul_i32 s6, 3, s15 // (op 3741) 
s_add_u32 s6, s6, 192 // (op 3741) 

v_add_u32 v3, v9, s6 // (op 3741) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3741) Load value

v_accvgpr_read v16, a60 // (op 3747) 
v_accvgpr_read v17, a61 // (op 3747) 
v_accvgpr_read v18, a62 // (op 3747) 
v_accvgpr_read v19, a63 // (op 3747) 
v_mul_f32 v20, v11, v16 // (op 3747) 
v_mul_f32 v21, v11, v17 // (op 3747) 
v_mul_f32 v22, v11, v18 // (op 3747) 
v_mul_f32 v23, v11, v19 // (op 3747) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3739) 
v_mul_f32 v17, v0, v13 // (op 3739) 
v_mul_f32 v18, v0, v14 // (op 3739) 
v_mul_f32 v19, v0, v15 // (op 3739) 

v_add_f32 v12, v16, v20 // (op 3740) 
v_add_f32 v13, v17, v21 // (op 3740) 
v_add_f32 v14, v18, v22 // (op 3740) 
v_add_f32 v15, v19, v23 // (op 3740) 




s_lshl_b32 s15, s24, 6 // (op 3726) 
s_mul_i32 s6, 3, s15 // (op 3726) 
s_add_u32 s6, s6, 192 // (op 3726) 

v_add_u32 v3, v4, s6 // (op 3726) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3726) Store value




v_add_u32 v3, 128, v9 // (op 3650) 

s_nop 0
buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3650) Load value

v_accvgpr_read v16, a32 // (op 3638) 
v_accvgpr_read v17, a33 // (op 3638) 
v_accvgpr_read v18, a34 // (op 3638) 
v_accvgpr_read v19, a35 // (op 3638) 
v_mul_f32 v20, v11, v16 // (op 3638) 
v_mul_f32 v21, v11, v17 // (op 3638) 
v_mul_f32 v22, v11, v18 // (op 3638) 
v_mul_f32 v23, v11, v19 // (op 3638) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3649) 
v_mul_f32 v17, v0, v13 // (op 3649) 
v_mul_f32 v18, v0, v14 // (op 3649) 
v_mul_f32 v19, v0, v15 // (op 3649) 

v_add_f32 v12, v16, v20 // (op 3639) 
v_add_f32 v13, v17, v21 // (op 3639) 
v_add_f32 v14, v18, v22 // (op 3639) 
v_add_f32 v15, v19, v23 // (op 3639) 




v_add_u32 v3, 128, v4 // (op 3641) 

buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3641) Store value




s_lshl_b32 s15, s4, 6 // (op 3654) 
s_add_u32 s15, s15, 128 // (op 3654) 
v_add_u32 v3, v9, s15 // (op 3654) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3654) Load value

v_accvgpr_read v16, a36 // (op 3664) 
v_accvgpr_read v17, a37 // (op 3664) 
v_accvgpr_read v18, a38 // (op 3664) 
v_accvgpr_read v19, a39 // (op 3664) 
v_mul_f32 v20, v11, v16 // (op 3664) 
v_mul_f32 v21, v11, v17 // (op 3664) 
v_mul_f32 v22, v11, v18 // (op 3664) 
v_mul_f32 v23, v11, v19 // (op 3664) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3652) 
v_mul_f32 v17, v0, v13 // (op 3652) 
v_mul_f32 v18, v0, v14 // (op 3652) 
v_mul_f32 v19, v0, v15 // (op 3652) 

v_add_f32 v12, v16, v20 // (op 3653) 
v_add_f32 v13, v17, v21 // (op 3653) 
v_add_f32 v14, v18, v22 // (op 3653) 
v_add_f32 v15, v19, v23 // (op 3653) 




s_lshl_b32 s15, s24, 6 // (op 3643) 
s_add_u32 s15, s15, 128 // (op 3643) 
v_add_u32 v3, v4, s15 // (op 3643) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3643) Store value




s_lshl_b32 s15, s4, 7 // (op 3658) 
s_add_u32 s15, s15, 128 // (op 3658) 
v_add_u32 v3, v9, s15 // (op 3658) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3658) Load value

v_accvgpr_read v16, a40 // (op 3666) 
v_accvgpr_read v17, a41 // (op 3666) 
v_accvgpr_read v18, a42 // (op 3666) 
v_accvgpr_read v19, a43 // (op 3666) 
v_mul_f32 v20, v11, v16 // (op 3666) 
v_mul_f32 v21, v11, v17 // (op 3666) 
v_mul_f32 v22, v11, v18 // (op 3666) 
v_mul_f32 v23, v11, v19 // (op 3666) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3656) 
v_mul_f32 v17, v0, v13 // (op 3656) 
v_mul_f32 v18, v0, v14 // (op 3656) 
v_mul_f32 v19, v0, v15 // (op 3656) 

v_add_f32 v12, v16, v20 // (op 3657) 
v_add_f32 v13, v17, v21 // (op 3657) 
v_add_f32 v14, v18, v22 // (op 3657) 
v_add_f32 v15, v19, v23 // (op 3657) 




s_lshl_b32 s15, s24, 7 // (op 3645) 
s_add_u32 s15, s15, 128 // (op 3645) 
v_add_u32 v3, v4, s15 // (op 3645) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3645) Store value




s_lshl_b32 s15, s4, 6 // (op 3662) 
s_mul_i32 s6, 3, s15 // (op 3662) 
s_add_u32 s6, s6, 128 // (op 3662) 

v_add_u32 v3, v9, s6 // (op 3662) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3662) Load value

v_accvgpr_read v16, a44 // (op 3668) 
v_accvgpr_read v17, a45 // (op 3668) 
v_accvgpr_read v18, a46 // (op 3668) 
v_accvgpr_read v19, a47 // (op 3668) 
v_mul_f32 v20, v11, v16 // (op 3668) 
v_mul_f32 v21, v11, v17 // (op 3668) 
v_mul_f32 v22, v11, v18 // (op 3668) 
v_mul_f32 v23, v11, v19 // (op 3668) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3660) 
v_mul_f32 v17, v0, v13 // (op 3660) 
v_mul_f32 v18, v0, v14 // (op 3660) 
v_mul_f32 v19, v0, v15 // (op 3660) 

v_add_f32 v12, v16, v20 // (op 3661) 
v_add_f32 v13, v17, v21 // (op 3661) 
v_add_f32 v14, v18, v22 // (op 3661) 
v_add_f32 v15, v19, v23 // (op 3661) 




s_lshl_b32 s15, s24, 6 // (op 3647) 
s_mul_i32 s6, 3, s15 // (op 3647) 
s_add_u32 s6, s6, 128 // (op 3647) 

v_add_u32 v3, v4, s6 // (op 3647) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3647) Store value




v_add_u32 v3, 64, v9 // (op 3571) 

s_nop 0
buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3571) Load value

v_accvgpr_read v16, a16 // (op 3559) 
v_accvgpr_read v17, a17 // (op 3559) 
v_accvgpr_read v18, a18 // (op 3559) 
v_accvgpr_read v19, a19 // (op 3559) 
v_mul_f32 v20, v11, v16 // (op 3559) 
v_mul_f32 v21, v11, v17 // (op 3559) 
v_mul_f32 v22, v11, v18 // (op 3559) 
v_mul_f32 v23, v11, v19 // (op 3559) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3570) 
v_mul_f32 v17, v0, v13 // (op 3570) 
v_mul_f32 v18, v0, v14 // (op 3570) 
v_mul_f32 v19, v0, v15 // (op 3570) 

v_add_f32 v12, v16, v20 // (op 3560) 
v_add_f32 v13, v17, v21 // (op 3560) 
v_add_f32 v14, v18, v22 // (op 3560) 
v_add_f32 v15, v19, v23 // (op 3560) 




v_add_u32 v3, 64, v4 // (op 3562) 

buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3562) Store value




s_lshl_b32 s15, s4, 6 // (op 3575) 
s_add_u32 s15, s15, 64 // (op 3575) 
v_add_u32 v3, v9, s15 // (op 3575) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3575) Load value

v_accvgpr_read v16, a20 // (op 3585) 
v_accvgpr_read v17, a21 // (op 3585) 
v_accvgpr_read v18, a22 // (op 3585) 
v_accvgpr_read v19, a23 // (op 3585) 
v_mul_f32 v20, v11, v16 // (op 3585) 
v_mul_f32 v21, v11, v17 // (op 3585) 
v_mul_f32 v22, v11, v18 // (op 3585) 
v_mul_f32 v23, v11, v19 // (op 3585) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3573) 
v_mul_f32 v17, v0, v13 // (op 3573) 
v_mul_f32 v18, v0, v14 // (op 3573) 
v_mul_f32 v19, v0, v15 // (op 3573) 

v_add_f32 v12, v16, v20 // (op 3574) 
v_add_f32 v13, v17, v21 // (op 3574) 
v_add_f32 v14, v18, v22 // (op 3574) 
v_add_f32 v15, v19, v23 // (op 3574) 




s_lshl_b32 s15, s24, 6 // (op 3564) 
s_add_u32 s15, s15, 64 // (op 3564) 
v_add_u32 v3, v4, s15 // (op 3564) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3564) Store value




s_lshl_b32 s15, s4, 7 // (op 3579) 
s_add_u32 s15, s15, 64 // (op 3579) 
v_add_u32 v3, v9, s15 // (op 3579) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3579) Load value

v_accvgpr_read v16, a24 // (op 3587) 
v_accvgpr_read v17, a25 // (op 3587) 
v_accvgpr_read v18, a26 // (op 3587) 
v_accvgpr_read v19, a27 // (op 3587) 
v_mul_f32 v20, v11, v16 // (op 3587) 
v_mul_f32 v21, v11, v17 // (op 3587) 
v_mul_f32 v22, v11, v18 // (op 3587) 
v_mul_f32 v23, v11, v19 // (op 3587) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3577) 
v_mul_f32 v17, v0, v13 // (op 3577) 
v_mul_f32 v18, v0, v14 // (op 3577) 
v_mul_f32 v19, v0, v15 // (op 3577) 

v_add_f32 v12, v16, v20 // (op 3578) 
v_add_f32 v13, v17, v21 // (op 3578) 
v_add_f32 v14, v18, v22 // (op 3578) 
v_add_f32 v15, v19, v23 // (op 3578) 




s_lshl_b32 s15, s24, 7 // (op 3566) 
s_add_u32 s15, s15, 64 // (op 3566) 
v_add_u32 v3, v4, s15 // (op 3566) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3566) Store value




s_lshl_b32 s15, s4, 6 // (op 3583) 
s_mul_i32 s6, 3, s15 // (op 3583) 
s_add_u32 s6, s6, 64 // (op 3583) 

v_add_u32 v3, v9, s6 // (op 3583) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3583) Load value

v_accvgpr_read v16, a28 // (op 3589) 
v_accvgpr_read v17, a29 // (op 3589) 
v_accvgpr_read v18, a30 // (op 3589) 
v_accvgpr_read v19, a31 // (op 3589) 
v_mul_f32 v20, v11, v16 // (op 3589) 
v_mul_f32 v21, v11, v17 // (op 3589) 
v_mul_f32 v22, v11, v18 // (op 3589) 
v_mul_f32 v23, v11, v19 // (op 3589) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3581) 
v_mul_f32 v17, v0, v13 // (op 3581) 
v_mul_f32 v18, v0, v14 // (op 3581) 
v_mul_f32 v19, v0, v15 // (op 3581) 

v_add_f32 v12, v16, v20 // (op 3582) 
v_add_f32 v13, v17, v21 // (op 3582) 
v_add_f32 v14, v18, v22 // (op 3582) 
v_add_f32 v15, v19, v23 // (op 3582) 




s_lshl_b32 s15, s24, 6 // (op 3568) 
s_mul_i32 s6, 3, s15 // (op 3568) 
s_add_u32 s6, s6, 64 // (op 3568) 

v_add_u32 v3, v4, s6 // (op 3568) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3568) Store value




v_mov_b32 v3, v9 // (op 226) reg expression

s_nop 0
buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 226) Load value

v_accvgpr_read v16, a0 // (op 223) 
v_accvgpr_read v17, a1 // (op 223) 
v_accvgpr_read v18, a2 // (op 223) 
v_accvgpr_read v19, a3 // (op 223) 
v_mul_f32 v20, v11, v16 // (op 223) 
v_mul_f32 v21, v11, v17 // (op 223) 
v_mul_f32 v22, v11, v18 // (op 223) 
v_mul_f32 v23, v11, v19 // (op 223) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 227) 
v_mul_f32 v17, v0, v13 // (op 227) 
v_mul_f32 v18, v0, v14 // (op 227) 
v_mul_f32 v19, v0, v15 // (op 227) 

v_add_f32 v12, v16, v20 // (op 224) 
v_add_f32 v13, v17, v21 // (op 224) 
v_add_f32 v14, v18, v22 // (op 224) 
v_add_f32 v15, v19, v23 // (op 224) 




v_mov_b32 v3, v4 // (op 225) reg expression

buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 225) Store value




s_lshl_b32 s15, s4, 6 // (op 3485) 
v_add_u32 v3, v9, s15 // (op 3485) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3485) Load value

v_accvgpr_read v16, a4 // (op 3482) 
v_accvgpr_read v17, a5 // (op 3482) 
v_accvgpr_read v18, a6 // (op 3482) 
v_accvgpr_read v19, a7 // (op 3482) 
v_mul_f32 v20, v11, v16 // (op 3482) 
v_mul_f32 v21, v11, v17 // (op 3482) 
v_mul_f32 v22, v11, v18 // (op 3482) 
v_mul_f32 v23, v11, v19 // (op 3482) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3486) 
v_mul_f32 v17, v0, v13 // (op 3486) 
v_mul_f32 v18, v0, v14 // (op 3486) 
v_mul_f32 v19, v0, v15 // (op 3486) 

v_add_f32 v12, v16, v20 // (op 3483) 
v_add_f32 v13, v17, v21 // (op 3483) 
v_add_f32 v14, v18, v22 // (op 3483) 
v_add_f32 v15, v19, v23 // (op 3483) 




s_lshl_b32 s15, s24, 6 // (op 3484) 
v_add_u32 v3, v4, s15 // (op 3484) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3484) Store value




s_lshl_b32 s15, s4, 7 // (op 3497) 
v_add_u32 v3, v9, s15 // (op 3497) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3497) Load value

v_accvgpr_read v16, a8 // (op 3494) 
v_accvgpr_read v17, a9 // (op 3494) 
v_accvgpr_read v18, a10 // (op 3494) 
v_accvgpr_read v19, a11 // (op 3494) 
v_mul_f32 v20, v11, v16 // (op 3494) 
v_mul_f32 v21, v11, v17 // (op 3494) 
v_mul_f32 v22, v11, v18 // (op 3494) 
v_mul_f32 v23, v11, v19 // (op 3494) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3498) 
v_mul_f32 v17, v0, v13 // (op 3498) 
v_mul_f32 v18, v0, v14 // (op 3498) 
v_mul_f32 v19, v0, v15 // (op 3498) 

v_add_f32 v12, v16, v20 // (op 3495) 
v_add_f32 v13, v17, v21 // (op 3495) 
v_add_f32 v14, v18, v22 // (op 3495) 
v_add_f32 v15, v19, v23 // (op 3495) 




s_lshl_b32 s15, s24, 7 // (op 3496) 
v_add_u32 v3, v4, s15 // (op 3496) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3496) Store value




s_lshl_b32 s15, s4, 6 // (op 3509) 
s_mul_i32 s6, 3, s15 // (op 3509) 

v_add_u32 v3, v9, s6 // (op 3509) 


buffer_load_dwordx4 v[12:15], v3, s[56:59], 0 offen // (op 3509) Load value

v_accvgpr_read v16, a12 // (op 3506) 
v_accvgpr_read v17, a13 // (op 3506) 
v_accvgpr_read v18, a14 // (op 3506) 
v_accvgpr_read v19, a15 // (op 3506) 
v_mul_f32 v20, v11, v16 // (op 3506) 
v_mul_f32 v21, v11, v17 // (op 3506) 
v_mul_f32 v22, v11, v18 // (op 3506) 
v_mul_f32 v23, v11, v19 // (op 3506) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v0, v12 // (op 3510) 
v_mul_f32 v17, v0, v13 // (op 3510) 
v_mul_f32 v18, v0, v14 // (op 3510) 
v_mul_f32 v19, v0, v15 // (op 3510) 

v_add_f32 v12, v16, v20 // (op 3507) 
v_add_f32 v13, v17, v21 // (op 3507) 
v_add_f32 v14, v18, v22 // (op 3507) 
v_add_f32 v15, v19, v23 // (op 3507) 




s_lshl_b32 s15, s24, 6 // (op 3508) 
s_mul_i32 s6, 3, s15 // (op 3508) 

v_add_u32 v3, v4, s6 // (op 3508) 


buffer_store_dwordx4 v[12:15], v3, s[60:63], 0 offen // (op 3508) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_203:
s_add_u32 s0, s0, s1 // (op 412) 


s_mul_i32 s15, s50, s16 // (op 410) 
s_cmp_gt_u32 s51, s15 // (op 410) 
s_mov_b32 s6, scc // (op 410) 
s_sub_u32 s7, s51, s15 // (op 410) 

s_cmp_eq_i32 s6, 0 // (op 410) 
s_cselect_b32 s8, 0, s7 // (op 410) 


s_cmp_lt_u32 s50, s8 // (op 410) 
s_mov_b32 s15, scc // (op 410) 
s_cmp_eq_i32 s15, 0 // (op 410) 
s_cselect_b32 s6, s8, s50 // (op 410) 


s_cmp_lt_u32 s0, s6 // (op 410) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_410 // (op 410) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopTop_DPStreamTileLoop_410 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_410:



















































s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2, .LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2_end-RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
  .amdhsa_next_free_vgpr 180
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 49152 // lds bytes
  .amdhsa_accum_offset 116
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
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 3
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 2
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 49152
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 88
    .vgpr_count: 116
    .agpr_count: 64
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

