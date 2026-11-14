// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2,@function
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 4576) 
v_and_b32 v3, 1, v0 // (op 4576) 

v_and_b32 v0, 15, v1 // (op 4576) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 4576) 


v_and_b32 v0, 48, v1 // (op 4576) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 4576) 


v_lshrrev_b32 v0, 6, v1 // (op 4834) 
v_and_b32 v4, 1, v0 // (op 4834) 

v_and_b32 v0, 15, v1 // (op 4834) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4834) 


v_and_b32 v0, 48, v1 // (op 4834) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4834) 


v_lshrrev_b32 v0, 6, v1 // (op 5048) 
v_and_b32 v5, 1, v0 // (op 5048) 

v_and_b32 v0, 15, v1 // (op 5048) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 5048) 


v_and_b32 v0, 48, v1 // (op 5048) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 5048) 


v_lshrrev_b32 v0, 6, v1 // (op 5160) 
v_and_b32 v6, 1, v0 // (op 5160) 

v_and_b32 v0, 15, v1 // (op 5160) 
v_lshl_add_u32 v7, v6, 4, v0 // (op 5160) 


v_and_b32 v0, 48, v1 // (op 5160) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 5160) 


v_lshrrev_b32 v0, 7, v1 // (op 4506) 
v_and_b32 v7, 15, v1 // (op 4506) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 4506) 


v_and_b32 v0, 48, v1 // (op 4506) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 4506) 


v_lshrrev_b32 v0, 7, v1 // (op 4647) 
v_and_b32 v8, 15, v1 // (op 4647) 
v_lshl_add_u32 v9, v0, 4, v8 // (op 4647) 


v_and_b32 v0, 63, v1 // (op 4647) 
v_lshrrev_b32 v8, 4, v0 // (op 4647) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 4647) 


v_lshrrev_b32 v8, 7, v1 // (op 4764) 
v_and_b32 v9, 15, v1 // (op 4764) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 4764) 


v_and_b32 v8, 48, v1 // (op 4764) 
v_lshl_add_u32 v9, v10, 7, v8 // (op 4764) 


v_lshrrev_b32 v8, 7, v1 // (op 4905) 
v_and_b32 v10, 15, v1 // (op 4905) 
v_lshl_add_u32 v11, v8, 4, v10 // (op 4905) 


v_and_b32 v8, 63, v1 // (op 4905) 
v_lshrrev_b32 v10, 4, v8 // (op 4905) 

v_lshl_add_u32 v8, v11, 2, v10 // (op 4905) 


v_lshrrev_b32 v10, 7, v1 // (op 5009) 
v_and_b32 v11, 15, v1 // (op 5009) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 5009) 


v_and_b32 v10, 48, v1 // (op 5009) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 5009) 


v_lshrrev_b32 v10, 7, v1 // (op 5077) 
v_and_b32 v12, 15, v1 // (op 5077) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 5077) 


v_and_b32 v10, 63, v1 // (op 5077) 
v_lshrrev_b32 v12, 4, v10 // (op 5077) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 5077) 


v_lshrrev_b32 v12, 7, v1 // (op 5121) 
v_and_b32 v13, 15, v1 // (op 5121) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 5121) 


v_and_b32 v12, 48, v1 // (op 5121) 
v_lshl_add_u32 v13, v14, 7, v12 // (op 5121) 


v_lshrrev_b32 v12, 7, v1 // (op 5189) 
v_and_b32 v14, 15, v1 // (op 5189) 
v_lshl_add_u32 v15, v12, 4, v14 // (op 5189) 


v_and_b32 v12, 63, v1 // (op 5189) 
v_lshrrev_b32 v14, 4, v12 // (op 5189) 

v_lshl_add_u32 v12, v15, 2, v14 // (op 5189) 


v_mov_b32 v14, s6 // (op 2) Move value
v_mov_b32 v15, s7 // (op 39) Move value


s_mov_b32 s0, 0 // (op 371) call()
s_mov_b32 s1, 0 // (op 372) call()
s_mov_b32 s6, 0 // (op 373) call()
s_mov_b32 s7, 0 // (op 374) call()
s_mov_b32 s18, 0 // (op 375) call()
s_mov_b32 s19, 0 // (op 376) call()
s_mov_b32 s0, 0 // (op 271) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 270) 
s_cmp_gt_u32 s9, s52 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_sub_u32 s54, s9, s52 // (op 270) 

s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s52, 0, s54 // (op 270) 


s_cmp_lt_u32 s8, s52 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s54, s52, s8 // (op 270) 


s_cmp_lt_u32 s0, s54 // (op 270) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 // (op 270) Condition: Top (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 if false)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_270:
v_lshrrev_b32 v16, 7, v1 // (op 4422) 
s_mul_i32 s52, s16, s8 // (op 4422) 
s_add_u32 s52, s52, s0 // (op 4422) 
v_and_b32 v17, 63, v1 // (op 4422) 
s_mul_hi_u32 s53, s52, s13 // (op 4422) 
v_lshrrev_b32 v18, 4, v17 // (op 4422) 
s_sub_u32 s54, s52, s53 // (op 4422) 
v_lshlrev_b32 v19, 2, v18 // (op 4422) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 4422) 


s_lshr_b32 s55, s54, 1 // (op 4422) 

s_add_u32 s54, s55, s53 // (op 4422) 


v_lshrrev_b32 v16, 2, v17 // (op 4422) 

s_lshr_b32 s53, s54, s11 // (op 4422) 

v_and_b32 v17, 3, v16 // (op 4422) 

s_and_b32 s54, s11, 2147483648 // (op 4422) 
v_and_b32 v16, 3, v1 // (op 4422) 
s_cmp_eq_i32 s54, 0 // (op 4422) 
s_mov_b32 s55, scc // (op 4422) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 4422) 


s_cmp_eq_i32 s55, 0 // (op 4422) 
s_cselect_b32 s54, s52, s53 // (op 4422) 



v_lshrrev_b32 v16, 6, v1 // (op 4422) 
s_mul_hi_u32 s52, s54, s12 // (op 4422) 
v_and_b32 v17, 1, v16 // (op 4422) 

s_sub_u32 s53, s54, s52 // (op 4422) 
v_lshl_add_u32 v16, v17, 4, v19 // (op 4422) 


s_lshr_b32 s55, s53, 1 // (op 4422) 

s_add_u32 s53, s55, s52 // (op 4422) 


s_lshr_b32 s52, s53, s10 // (op 4422) 

s_and_b32 s53, s10, 2147483648 // (op 4422) 
s_cmp_eq_i32 s53, 0 // (op 4422) 
s_mov_b32 s55, scc // (op 4422) 

s_cmp_eq_i32 s55, 0 // (op 4422) 
s_cselect_b32 s53, s54, s52 // (op 4422) 


s_mul_i32 s52, s53, s14 // (op 4422) 
v_lshl_add_u32 v17, s53, 7, v18 // (op 4422) 


s_sub_u32 s53, s54, s52 // (op 4422) 


v_lshl_add_u32 v18, s53, 5, v16 // (op 4422) 


v_mul_lo_u32 v16, v18, s4 // (op 4422) 

v_add_lshl_u32 v18, v17, v16, 2 // (op 4422) 


s_mul_i32 s53, s4, s23 // (op 4424) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 4424) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 4424) least significant
s_mul_hi_u32 s55, s4, s22 // (op 4424) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4424) most significant: sum
s_add_u32 s53, s53, s55 // (op 4424) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 4424) 

s_mov_b32 s56, s20 // (op 4424) reg expression
s_mov_b32 s57, s21 // (op 4424) reg expression
s_mov_b32 s58, s54 // (op 4424) reg expression
s_mov_b32 s59, 131072 // (op 4424) call()

v_lshrrev_b32 v16, 7, v1 // (op 4523) 
v_and_b32 v17, 15, v1 // (op 4523) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 4523) 


v_and_b32 v16, 48, v1 // (op 4523) 
v_lshl_add_u32 v17, v19, 7, v16 // (op 4523) 


v_lshrrev_b32 v16, 6, v1 // (op 4588) 
v_and_b32 v19, 1, v16 // (op 4588) 

v_and_b32 v16, 15, v1 // (op 4588) 
v_lshl_add_u32 v20, v19, 4, v16 // (op 4588) 


v_and_b32 v16, 48, v1 // (op 4588) 
v_lshl_add_u32 v19, v20, 7, v16 // (op 4588) 


v_lshrrev_b32 v16, 7, v1 // (op 4664) 
v_and_b32 v20, 15, v1 // (op 4664) 
v_lshl_add_u32 v21, v16, 4, v20 // (op 4664) 


v_and_b32 v16, 63, v1 // (op 4664) 
v_lshrrev_b32 v20, 4, v16 // (op 4664) 

v_lshl_add_u32 v16, v21, 2, v20 // (op 4664) 


v_lshrrev_b32 v20, 7, v1 // (op 4682) 
s_mul_i32 s52, s16, s8 // (op 4682) 
s_add_u32 s52, s52, s0 // (op 4682) 
v_and_b32 v21, 63, v1 // (op 4682) 
s_mul_hi_u32 s53, s52, s13 // (op 4682) 
v_lshrrev_b32 v22, 4, v21 // (op 4682) 
s_sub_u32 s54, s52, s53 // (op 4682) 
v_lshlrev_b32 v23, 2, v22 // (op 4682) 

s_lshr_b32 s55, s54, 1 // (op 4682) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 4682) 


s_add_u32 s54, s55, s53 // (op 4682) 


v_lshrrev_b32 v20, 2, v21 // (op 4682) 

s_lshr_b32 s53, s54, s11 // (op 4682) 

v_and_b32 v21, 3, v20 // (op 4682) 

s_and_b32 s54, s11, 2147483648 // (op 4682) 
v_and_b32 v20, 3, v1 // (op 4682) 
s_cmp_eq_i32 s54, 0 // (op 4682) 
s_mov_b32 s55, scc // (op 4682) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 4682) 


s_cmp_eq_i32 s55, 0 // (op 4682) 
s_cselect_b32 s54, s52, s53 // (op 4682) 



v_lshrrev_b32 v20, 6, v1 // (op 4682) 
s_mul_hi_u32 s55, s54, s12 // (op 4682) 
v_and_b32 v21, 1, v20 // (op 4682) 

s_sub_u32 s52, s54, s55 // (op 4682) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 4682) 


s_lshr_b32 s53, s52, 1 // (op 4682) 

s_add_u32 s52, s53, s55 // (op 4682) 


s_lshr_b32 s53, s52, s10 // (op 4682) 

s_and_b32 s52, s10, 2147483648 // (op 4682) 
s_cmp_eq_i32 s52, 0 // (op 4682) 
s_mov_b32 s55, scc // (op 4682) 

s_cmp_eq_i32 s55, 0 // (op 4682) 
s_cselect_b32 s52, s54, s53 // (op 4682) 


s_mul_i32 s53, s52, s14 // (op 4682) 
v_lshl_add_u32 v21, s52, 7, v22 // (op 4682) 


s_sub_u32 s52, s54, s53 // (op 4682) 


v_lshl_add_u32 v22, s52, 5, v20 // (op 4682) 


v_mul_lo_u32 v20, v22, s24 // (op 4682) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 4682) 


s_mul_i32 s53, s24, s23 // (op 4684) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 4684) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 4684) least significant
s_mul_hi_u32 s55, s24, s22 // (op 4684) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4684) most significant: sum
s_add_u32 s53, s53, s55 // (op 4684) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 4684) 

s_mov_b32 s60, s26 // (op 4684) reg expression
s_mov_b32 s61, s27 // (op 4684) reg expression
s_mov_b32 s62, s54 // (op 4684) reg expression
s_mov_b32 s63, 131072 // (op 4684) call()

v_lshrrev_b32 v20, 5, v1 // (op 4940) 
v_lshlrev_b32 v21, 2, v20 // (op 4940) 

v_lshl_add_u32 v20, s16, 7, v21 // (op 4940) 

v_and_b32 v21, -32, v1 // (op 4940) 
v_sub_u32 v23, v1, v21 // (op 4940) 

v_lshl_add_u32 v21, v20, 5, v23 // (op 4940) 


v_lshlrev_b32 v20, 2, v21 // (op 4940) 

s_lshl_b32 s52, s15, 14 // (op 4942) 
s_mov_b32 s64, s28 // (op 4942) reg expression
s_mov_b32 s65, s29 // (op 4942) reg expression
s_mov_b32 s66, s52 // (op 4942) reg expression
s_mov_b32 s67, 131072 // (op 4942) call()

s_mul_i32 s6, s8, s16 // (op 275) 
s_add_u32 s6, s6, s0 // (op 275) 
s_mul_hi_u32 s52, s6, s13 // (op 276) 
s_sub_u32 s53, s6, s52 // (op 276) 
s_lshr_b32 s54, s53, 1 // (op 276) 

s_add_u32 s53, s54, s52 // (op 276) 


s_lshr_b32 s52, s53, s11 // (op 276) 

s_and_b32 s53, s11, 2147483648 // (op 276) 
s_cmp_eq_i32 s53, 0 // (op 276) 
s_mov_b32 s54, scc // (op 276) 

s_cmp_eq_i32 s54, 0 // (op 276) 
s_cselect_b32 s53, s6, s52 // (op 276) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 276) 
s_mul_i32 s52, s53, s54 // (op 276) 


s_sub_u32 s18, s6, s52 // (op 276) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 277) 
s_cmp_gt_u32 s52, s18 // (op 277) 
s_mov_b32 s54, scc // (op 277) 
s_sub_u32 s55, s52, s18 // (op 277) 

s_cmp_eq_i32 s54, 0 // (op 277) 
s_cselect_b32 s52, 0, s55 // (op 277) 


s_cmp_gt_u32 s8, s0 // (op 277) 
s_mov_b32 s53, scc // (op 277) 
s_sub_u32 s54, s8, s0 // (op 277) 
s_cmp_eq_i32 s53, 0 // (op 277) 
s_cselect_b32 s55, 0, s54 // (op 277) 


s_cmp_lt_u32 s52, s55 // (op 277) 
s_mov_b32 s53, scc // (op 277) 
s_cmp_eq_i32 s53, 0 // (op 277) 
s_cselect_b32 s54, s55, s52 // (op 277) 



s_cmp_gt_u32 s9, s6 // (op 277) 
s_mov_b32 s55, scc // (op 277) 
s_sub_u32 s52, s9, s6 // (op 277) 
s_cmp_eq_i32 s55, 0 // (op 277) 
s_cselect_b32 s53, 0, s52 // (op 277) 


s_cmp_lt_u32 s54, s53 // (op 277) 
s_mov_b32 s52, scc // (op 277) 
s_cmp_eq_i32 s52, 0 // (op 277) 
s_cselect_b32 s7, s53, s54 // (op 277) 



s_add_u32 s52, s18, s7 // (op 278) 
s_sub_u32 s19, s52, 1 // (op 278) 

v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 1462) call()
v_accvgpr_write a5, 0.00000 // (op 1462) call()
v_accvgpr_write a6, 0.00000 // (op 1462) call()
v_accvgpr_write a7, 0.00000 // (op 1462) call()
v_accvgpr_write a8, 0.00000 // (op 1624) call()
v_accvgpr_write a9, 0.00000 // (op 1624) call()
v_accvgpr_write a10, 0.00000 // (op 1624) call()
v_accvgpr_write a11, 0.00000 // (op 1624) call()
v_accvgpr_write a12, 0.00000 // (op 1786) call()
v_accvgpr_write a13, 0.00000 // (op 1786) call()
v_accvgpr_write a14, 0.00000 // (op 1786) call()
v_accvgpr_write a15, 0.00000 // (op 1786) call()
v_lshrrev_b32 v21, 3, v1 // (op 3994) 
s_mul_i32 s52, s16, s8 // (op 3994) 
s_add_u32 s52, s52, s0 // (op 3994) 
v_and_b32 v23, -8, v1 // (op 3994) 
s_mul_hi_u32 s53, s52, s13 // (op 3994) 
v_sub_u32 v24, v1, v23 // (op 3994) 

s_sub_u32 s54, s52, s53 // (op 3994) 
v_lshlrev_b32 v23, 4, v24 // (op 3994) 

s_lshr_b32 s55, s54, 1 // (op 3994) 

s_add_u32 s54, s55, s53 // (op 3994) 


s_lshr_b32 s53, s54, s11 // (op 3994) 

s_and_b32 s54, s11, 2147483648 // (op 3994) 
s_cmp_eq_i32 s54, 0 // (op 3994) 
s_mov_b32 s55, scc // (op 3994) 

s_cmp_eq_i32 s55, 0 // (op 3994) 
s_cselect_b32 s54, s52, s53 // (op 3994) 


s_mul_hi_u32 s53, s54, s12 // (op 3994) 
s_sub_u32 s55, s54, s53 // (op 3994) 
s_lshr_b32 s68, s55, 1 // (op 3994) 

s_add_u32 s55, s68, s53 // (op 3994) 


s_lshr_b32 s53, s55, s10 // (op 3994) 

s_and_b32 s55, s10, 2147483648 // (op 3994) 
s_cmp_eq_i32 s55, 0 // (op 3994) 
s_mov_b32 s68, scc // (op 3994) 

s_cmp_eq_i32 s68, 0 // (op 3994) 
s_cselect_b32 s55, s54, s53 // (op 3994) 


v_lshl_add_u32 v24, s55, 7, v21 // (op 3994) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 3994) 
s_mul_i32 s53, s54, s68 // (op 3994) 


s_sub_u32 s54, s52, s53 // (op 3994) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 3994) 


v_mov_b32 v27, 0 // (op 3994) convert
v_mov_b32 v26, v21 // (op 3994) convert

v_mul_lo_u32 v29, v24, s31 // (op 3994) most significant: low of low * high
v_mul_hi_u32 v21, v24, s30 // (op 3994) most significant: high of low * low
v_mul_lo_u32 v28, v24, s30 // (op 3994) least significant: low of low * low
v_add_u32 v29, v29, v21 // (op 3994) most significant: sum

v_add_co_u32 v28, s[52:53], v28, v26 // (op 3994) least significant half
v_addc_co_u32 v29, s[52:53], v29, v27, s[52:53] // (op 3994) most significant half



s_mul_i32 s53, s30, s35 // (op 3996) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3996) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3996) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3996) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3996) most significant: sum
s_add_u32 s53, s53, s55 // (op 3996) most significant: sum


s_mov_b32 s68, s32 // (op 3996) reg expression
s_mov_b32 s69, s33 // (op 3996) reg expression
s_mov_b32 s70, s52 // (op 3996) reg expression
s_mov_b32 s71, 131072 // (op 3996) call()

v_lshrrev_b32 v21, 3, v1 // (op 4013) 
s_mul_i32 s52, s16, s8 // (op 4013) 
s_add_u32 s52, s52, s0 // (op 4013) 
v_and_b32 v23, -8, v1 // (op 4013) 
s_mul_hi_u32 s53, s52, s13 // (op 4013) 
v_sub_u32 v24, v1, v23 // (op 4013) 

s_sub_u32 s54, s52, s53 // (op 4013) 
v_lshlrev_b32 v23, 4, v24 // (op 4013) 

s_lshr_b32 s55, s54, 1 // (op 4013) 

s_add_u32 s54, s55, s53 // (op 4013) 


s_lshr_b32 s53, s54, s11 // (op 4013) 

s_and_b32 s54, s11, 2147483648 // (op 4013) 
s_cmp_eq_i32 s54, 0 // (op 4013) 
s_mov_b32 s55, scc // (op 4013) 

s_cmp_eq_i32 s55, 0 // (op 4013) 
s_cselect_b32 s54, s52, s53 // (op 4013) 


s_mul_hi_u32 s53, s54, s12 // (op 4013) 
s_sub_u32 s55, s54, s53 // (op 4013) 
s_lshr_b32 s72, s55, 1 // (op 4013) 

s_add_u32 s55, s72, s53 // (op 4013) 


s_lshr_b32 s53, s55, s10 // (op 4013) 

s_and_b32 s55, s10, 2147483648 // (op 4013) 
s_cmp_eq_i32 s55, 0 // (op 4013) 
s_mov_b32 s72, scc // (op 4013) 

s_cmp_eq_i32 s72, 0 // (op 4013) 
s_cselect_b32 s55, s54, s53 // (op 4013) 


v_lshl_add_u32 v24, s55, 7, v21 // (op 4013) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 4013) 
s_mul_i32 s53, s54, s72 // (op 4013) 


s_sub_u32 s54, s52, s53 // (op 4013) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 4013) 


v_mov_b32 v27, 0 // (op 4013) convert
v_mov_b32 v26, v21 // (op 4013) convert

v_mul_lo_u32 v31, v24, s31 // (op 4013) most significant: low of low * high
v_mul_hi_u32 v21, v24, s30 // (op 4013) most significant: high of low * low
v_mul_lo_u32 v30, v24, s30 // (op 4013) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 4013) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v26 // (op 4013) least significant half
v_addc_co_u32 v31, s[52:53], v31, v27, s[52:53] // (op 4013) most significant half



s_mul_i32 s53, s30, s35 // (op 4015) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 4015) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 4015) least significant
s_mul_hi_u32 s55, s30, s34 // (op 4015) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4015) most significant: sum
s_add_u32 s53, s53, s55 // (op 4015) most significant: sum


s_mov_b32 s68, s32 // (op 4015) reg expression
s_mov_b32 s69, s33 // (op 4015) reg expression
s_mov_b32 s70, s52 // (op 4015) reg expression
s_mov_b32 s71, 131072 // (op 4015) call()

v_lshrrev_b32 v21, 3, v1 // (op 4122) 
s_mul_i32 s52, s16, s8 // (op 4122) 
s_add_u32 s52, s52, s0 // (op 4122) 
v_and_b32 v23, -8, v1 // (op 4122) 
s_mul_hi_u32 s53, s52, s13 // (op 4122) 
v_sub_u32 v24, v1, v23 // (op 4122) 

s_sub_u32 s54, s52, s53 // (op 4122) 
v_lshlrev_b32 v23, 4, v24 // (op 4122) 

s_lshr_b32 s55, s54, 1 // (op 4122) 

s_add_u32 s54, s55, s53 // (op 4122) 


s_lshr_b32 s53, s54, s11 // (op 4122) 

s_and_b32 s54, s11, 2147483648 // (op 4122) 
s_cmp_eq_i32 s54, 0 // (op 4122) 
s_mov_b32 s55, scc // (op 4122) 

s_cmp_eq_i32 s55, 0 // (op 4122) 
s_cselect_b32 s54, s52, s53 // (op 4122) 


s_mul_hi_u32 s53, s54, s12 // (op 4122) 
s_sub_u32 s55, s54, s53 // (op 4122) 
s_lshr_b32 s72, s55, 1 // (op 4122) 

s_add_u32 s55, s72, s53 // (op 4122) 


s_lshr_b32 s53, s55, s10 // (op 4122) 

s_and_b32 s55, s10, 2147483648 // (op 4122) 
s_cmp_eq_i32 s55, 0 // (op 4122) 
s_mov_b32 s72, scc // (op 4122) 

s_cmp_eq_i32 s72, 0 // (op 4122) 
s_cselect_b32 s55, s54, s53 // (op 4122) 


s_mul_i32 s53, s55, s14 // (op 4122) 

s_sub_u32 s55, s54, s53 // (op 4122) 

v_lshl_add_u32 v24, s55, 5, v21 // (op 4122) 


v_mul_lo_u32 v27, v24, s37 // (op 4122) most significant: low of low * high
v_mul_hi_u32 v21, v24, s36 // (op 4122) most significant: high of low * low
v_mul_lo_u32 v26, v24, s36 // (op 4122) least significant: low of low * low
v_add_u32 v27, v27, v21 // (op 4122) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 4122) 
s_mul_i32 s53, s54, s72 // (op 4122) 


s_sub_u32 s54, s52, s53 // (op 4122) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 4122) 


v_mov_b32 v25, 0 // (op 4122) convert
v_mov_b32 v24, v21 // (op 4122) convert

v_add_co_u32 v32, s[52:53], v24, v26 // (op 4122) least significant half
v_addc_co_u32 v33, s[52:53], v25, v27, s[52:53] // (op 4122) most significant half



s_mul_i32 s53, s36, s23 // (op 4124) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 4124) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 4124) least significant
s_mul_hi_u32 s55, s36, s22 // (op 4124) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4124) most significant: sum
s_add_u32 s53, s53, s55 // (op 4124) most significant: sum


s_mov_b32 s72, s38 // (op 4124) reg expression
s_mov_b32 s73, s39 // (op 4124) reg expression
s_mov_b32 s74, s52 // (op 4124) reg expression
s_mov_b32 s75, 131072 // (op 4124) call()

v_lshrrev_b32 v21, 3, v1 // (op 4141) 
s_mul_i32 s52, s16, s8 // (op 4141) 
s_add_u32 s52, s52, s0 // (op 4141) 
v_and_b32 v23, -8, v1 // (op 4141) 
s_mul_hi_u32 s53, s52, s13 // (op 4141) 
v_sub_u32 v24, v1, v23 // (op 4141) 

s_sub_u32 s54, s52, s53 // (op 4141) 
v_lshlrev_b32 v23, 4, v24 // (op 4141) 

s_lshr_b32 s55, s54, 1 // (op 4141) 

s_add_u32 s54, s55, s53 // (op 4141) 


s_lshr_b32 s53, s54, s11 // (op 4141) 

s_and_b32 s54, s11, 2147483648 // (op 4141) 
s_cmp_eq_i32 s54, 0 // (op 4141) 
s_mov_b32 s55, scc // (op 4141) 

s_cmp_eq_i32 s55, 0 // (op 4141) 
s_cselect_b32 s54, s52, s53 // (op 4141) 


s_mul_hi_u32 s53, s54, s12 // (op 4141) 
s_sub_u32 s55, s54, s53 // (op 4141) 
s_lshr_b32 s76, s55, 1 // (op 4141) 

s_add_u32 s55, s76, s53 // (op 4141) 


s_lshr_b32 s53, s55, s10 // (op 4141) 

s_and_b32 s55, s10, 2147483648 // (op 4141) 
s_cmp_eq_i32 s55, 0 // (op 4141) 
s_mov_b32 s76, scc // (op 4141) 

s_cmp_eq_i32 s76, 0 // (op 4141) 
s_cselect_b32 s55, s54, s53 // (op 4141) 


s_mul_i32 s53, s55, s14 // (op 4141) 

s_sub_u32 s55, s54, s53 // (op 4141) 

v_lshl_add_u32 v24, s55, 5, v21 // (op 4141) 


v_mul_lo_u32 v27, v24, s37 // (op 4141) most significant: low of low * high
v_mul_hi_u32 v21, v24, s36 // (op 4141) most significant: high of low * low
v_mul_lo_u32 v26, v24, s36 // (op 4141) least significant: low of low * low
v_add_u32 v27, v27, v21 // (op 4141) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 4141) 
s_mul_i32 s53, s54, s76 // (op 4141) 


s_sub_u32 s54, s52, s53 // (op 4141) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 4141) 


v_mov_b32 v25, 0 // (op 4141) convert
v_mov_b32 v24, v21 // (op 4141) convert

v_add_co_u32 v34, s[52:53], v24, v26 // (op 4141) least significant half
v_addc_co_u32 v35, s[52:53], v25, v27, s[52:53] // (op 4141) most significant half



s_mul_i32 s53, s36, s23 // (op 4143) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 4143) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 4143) least significant
s_mul_hi_u32 s55, s36, s22 // (op 4143) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4143) most significant: sum
s_add_u32 s53, s53, s55 // (op 4143) most significant: sum


s_mov_b32 s72, s38 // (op 4143) reg expression
s_mov_b32 s73, s39 // (op 4143) reg expression
s_mov_b32 s74, s52 // (op 4143) reg expression
s_mov_b32 s75, 131072 // (op 4143) call()

v_lshrrev_b32 v21, 2, v1 // (op 4250) 
s_mul_i32 s52, s16, s8 // (op 4250) 
s_add_u32 s52, s52, s0 // (op 4250) 
v_lshlrev_b32 v23, 1, v21 // (op 4250) 

s_mul_hi_u32 s53, s52, s13 // (op 4250) 
v_and_b32 v21, -4, v1 // (op 4250) 
s_sub_u32 s54, s52, s53 // (op 4250) 
v_sub_u32 v24, v1, v21 // (op 4250) 

s_lshr_b32 s55, s54, 1 // (op 4250) 

s_add_u32 s54, s55, s53 // (op 4250) 


s_lshr_b32 s53, s54, s11 // (op 4250) 

s_and_b32 s54, s11, 2147483648 // (op 4250) 
s_cmp_eq_i32 s54, 0 // (op 4250) 
s_mov_b32 s55, scc // (op 4250) 

s_cmp_eq_i32 s55, 0 // (op 4250) 
s_cselect_b32 s54, s52, s53 // (op 4250) 


s_mul_hi_u32 s53, s54, s12 // (op 4250) 
s_sub_u32 s55, s54, s53 // (op 4250) 
s_lshr_b32 s76, s55, 1 // (op 4250) 

s_add_u32 s55, s76, s53 // (op 4250) 


s_lshr_b32 s53, s55, s10 // (op 4250) 

s_and_b32 s55, s10, 2147483648 // (op 4250) 
s_cmp_eq_i32 s55, 0 // (op 4250) 
s_mov_b32 s76, scc // (op 4250) 

s_cmp_eq_i32 s76, 0 // (op 4250) 
s_cselect_b32 s55, s54, s53 // (op 4250) 


v_lshl_add_u32 v21, s55, 7, v23 // (op 4250) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 4250) 
s_mul_i32 s53, s54, s76 // (op 4250) 


s_sub_u32 s54, s52, s53 // (op 4250) 


v_lshl_add_u32 v23, s54, 2, v24 // (op 4250) 


v_mul_lo_u32 v24, v21, s40 // (op 4250) 
v_add_u32 v24, v24, v23 // (op 4250) 


s_mul_i32 s53, s40, s35 // (op 4252) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 4252) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 4252) least significant
s_mul_hi_u32 s55, s40, s34 // (op 4252) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4252) most significant: sum
s_add_u32 s53, s53, s55 // (op 4252) most significant: sum


s_mov_b32 s76, s42 // (op 4252) reg expression
s_mov_b32 s77, s43 // (op 4252) reg expression
s_mov_b32 s78, s52 // (op 4252) reg expression
s_mov_b32 s79, 131072 // (op 4252) call()

v_lshrrev_b32 v21, 2, v1 // (op 4269) 
s_mul_i32 s52, s16, s8 // (op 4269) 
s_add_u32 s52, s52, s0 // (op 4269) 
v_lshlrev_b32 v23, 1, v21 // (op 4269) 

s_mul_hi_u32 s53, s52, s13 // (op 4269) 
v_and_b32 v21, -4, v1 // (op 4269) 
s_sub_u32 s54, s52, s53 // (op 4269) 
v_sub_u32 v25, v1, v21 // (op 4269) 

s_lshr_b32 s55, s54, 1 // (op 4269) 

s_add_u32 s54, s55, s53 // (op 4269) 


s_lshr_b32 s53, s54, s11 // (op 4269) 

s_and_b32 s54, s11, 2147483648 // (op 4269) 
s_cmp_eq_i32 s54, 0 // (op 4269) 
s_mov_b32 s55, scc // (op 4269) 

s_cmp_eq_i32 s55, 0 // (op 4269) 
s_cselect_b32 s54, s52, s53 // (op 4269) 


s_mul_hi_u32 s53, s54, s12 // (op 4269) 
s_sub_u32 s55, s54, s53 // (op 4269) 
s_lshr_b32 s80, s55, 1 // (op 4269) 

s_add_u32 s55, s80, s53 // (op 4269) 


s_lshr_b32 s53, s55, s10 // (op 4269) 

s_and_b32 s55, s10, 2147483648 // (op 4269) 
s_cmp_eq_i32 s55, 0 // (op 4269) 
s_mov_b32 s80, scc // (op 4269) 

s_cmp_eq_i32 s80, 0 // (op 4269) 
s_cselect_b32 s55, s54, s53 // (op 4269) 


v_lshl_add_u32 v21, s55, 7, v23 // (op 4269) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 4269) 
s_mul_i32 s53, s54, s80 // (op 4269) 


s_sub_u32 s54, s52, s53 // (op 4269) 


v_lshl_add_u32 v23, s54, 2, v25 // (op 4269) 


v_mul_lo_u32 v25, v21, s40 // (op 4269) 
v_add_u32 v25, v25, v23 // (op 4269) 


s_mul_i32 s53, s40, s35 // (op 4271) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 4271) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 4271) least significant
s_mul_hi_u32 s55, s40, s34 // (op 4271) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4271) most significant: sum
s_add_u32 s53, s53, s55 // (op 4271) most significant: sum


s_mov_b32 s76, s42 // (op 4271) reg expression
s_mov_b32 s77, s43 // (op 4271) reg expression
s_mov_b32 s78, s52 // (op 4271) reg expression
s_mov_b32 s79, 131072 // (op 4271) call()

v_lshrrev_b32 v21, 6, v1 // (op 4378) 
s_mul_i32 s52, s16, s8 // (op 4378) 
s_add_u32 s52, s52, s0 // (op 4378) 
v_and_b32 v23, 1, v21 // (op 4378) 

s_mul_hi_u32 s53, s52, s13 // (op 4378) 
v_and_b32 v21, 15, v1 // (op 4378) 
s_sub_u32 s54, s52, s53 // (op 4378) 
v_lshl_add_u32 v26, v23, 4, v21 // (op 4378) 


s_lshr_b32 s55, s54, 1 // (op 4378) 

v_and_b32 v21, 63, v1 // (op 4378) 
s_add_u32 s54, s55, s53 // (op 4378) 


v_lshrrev_b32 v23, 4, v21 // (op 4378) 

s_lshr_b32 s53, s54, s11 // (op 4378) 

s_and_b32 s54, s11, 2147483648 // (op 4378) 
s_cmp_eq_i32 s54, 0 // (op 4378) 
s_mov_b32 s55, scc // (op 4378) 

s_cmp_eq_i32 s55, 0 // (op 4378) 
s_cselect_b32 s54, s52, s53 // (op 4378) 


s_mul_hi_u32 s53, s54, s12 // (op 4378) 
s_sub_u32 s55, s54, s53 // (op 4378) 
s_lshr_b32 s80, s55, 1 // (op 4378) 

s_add_u32 s55, s80, s53 // (op 4378) 


s_lshr_b32 s53, s55, s10 // (op 4378) 

s_and_b32 s55, s10, 2147483648 // (op 4378) 
s_cmp_eq_i32 s55, 0 // (op 4378) 
s_mov_b32 s80, scc // (op 4378) 

s_cmp_eq_i32 s80, 0 // (op 4378) 
s_cselect_b32 s55, s54, s53 // (op 4378) 


s_mul_i32 s53, s55, s14 // (op 4378) 

s_sub_u32 s55, s54, s53 // (op 4378) 

v_lshl_add_u32 v21, s55, 5, v26 // (op 4378) 


v_mul_lo_u32 v26, v21, s44 // (op 4378) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 4378) 
s_mul_i32 s53, s54, s80 // (op 4378) 


s_sub_u32 s54, s52, s53 // (op 4378) 


v_lshl_add_u32 v21, s54, 2, v23 // (op 4378) 


v_add_u32 v23, v21, v26 // (op 4378) 


s_mul_i32 s53, s44, s23 // (op 4380) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4380) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4380) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4380) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4380) most significant: sum
s_add_u32 s53, s53, s55 // (op 4380) most significant: sum


s_mov_b32 s80, s46 // (op 4380) reg expression
s_mov_b32 s81, s47 // (op 4380) reg expression
s_mov_b32 s82, s52 // (op 4380) reg expression
s_mov_b32 s83, 131072 // (op 4380) call()

v_lshrrev_b32 v21, 6, v1 // (op 4397) 
s_mul_i32 s52, s16, s8 // (op 4397) 
s_add_u32 s52, s52, s0 // (op 4397) 
v_and_b32 v26, 1, v21 // (op 4397) 

s_mul_hi_u32 s53, s52, s13 // (op 4397) 
v_and_b32 v21, 15, v1 // (op 4397) 
s_sub_u32 s54, s52, s53 // (op 4397) 
v_lshl_add_u32 v27, v26, 4, v21 // (op 4397) 


s_lshr_b32 s55, s54, 1 // (op 4397) 

v_and_b32 v21, 63, v1 // (op 4397) 
s_add_u32 s54, s55, s53 // (op 4397) 


v_lshrrev_b32 v26, 4, v21 // (op 4397) 

s_lshr_b32 s53, s54, s11 // (op 4397) 

s_and_b32 s54, s11, 2147483648 // (op 4397) 
s_cmp_eq_i32 s54, 0 // (op 4397) 
s_mov_b32 s55, scc // (op 4397) 

s_cmp_eq_i32 s55, 0 // (op 4397) 
s_cselect_b32 s54, s52, s53 // (op 4397) 


s_mul_hi_u32 s53, s54, s12 // (op 4397) 
s_sub_u32 s55, s54, s53 // (op 4397) 
s_lshr_b32 s84, s55, 1 // (op 4397) 

s_add_u32 s55, s84, s53 // (op 4397) 


s_lshr_b32 s53, s55, s10 // (op 4397) 

s_and_b32 s55, s10, 2147483648 // (op 4397) 
s_cmp_eq_i32 s55, 0 // (op 4397) 
s_mov_b32 s84, scc // (op 4397) 

s_cmp_eq_i32 s84, 0 // (op 4397) 
s_cselect_b32 s55, s54, s53 // (op 4397) 


s_mul_i32 s53, s55, s14 // (op 4397) 

s_sub_u32 s55, s54, s53 // (op 4397) 

v_lshl_add_u32 v21, s55, 5, v27 // (op 4397) 


v_mul_lo_u32 v27, v21, s44 // (op 4397) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 4397) 
s_mul_i32 s53, s54, s84 // (op 4397) 


s_sub_u32 s54, s52, s53 // (op 4397) 


v_lshl_add_u32 v21, s54, 2, v26 // (op 4397) 


v_add_u32 v26, v21, v27 // (op 4397) 


s_mul_i32 s53, s44, s23 // (op 4399) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4399) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4399) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4399) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4399) most significant: sum
s_add_u32 s53, s53, s55 // (op 4399) most significant: sum


s_mov_b32 s80, s46 // (op 4399) reg expression
s_mov_b32 s81, s47 // (op 4399) reg expression
s_mov_b32 s82, s52 // (op 4399) reg expression
s_mov_b32 s83, 131072 // (op 4399) call()




buffer_load_ubyte v21, v23, s[80:83], 0 offen offset: 0 // (op 3639) Load value
v_and_b32 v27, -8, v1 // (op 4564) 
v_sub_u32 v36, v1, v27 // (op 4564) 

v_lshlrev_b32 v27, 4, v36 // (op 4564) 

v_lshrrev_b32 v36, 3, v1 // (op 4564) 
v_lshl_add_u32 v37, v36, 7, v27 // (op 4564) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 4564) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5301) 



s_nop 0
s_mov_b32 m0, s52 // (op 3868) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3868) Load value direct to lds

v_and_b32 v27, -8, v1 // (op 4489) 
v_sub_u32 v36, v1, v27 // (op 4489) 

v_lshlrev_b32 v27, 4, v36 // (op 4489) 

v_lshrrev_b32 v36, 3, v1 // (op 4489) 
v_lshl_add_u32 v37, v36, 7, v27 // (op 4489) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 4489) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5282) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 3873) 
v_mov_b64 v[36:37], s[54:55] // (op 3873) convert


v_mov_b64 v[38:39], v[28:29] // (op 3873) 
s_add_u32 s53, 4096, s52 // (op 3873) 
s_mov_b32 m0, s53 // (op 3873) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 3873) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 3873) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 3873) most significant half

s_add_u32 m0, m0, 4096 // (op 3873) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 3873) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 3873) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 3873) most significant half

s_add_u32 m0, m0, 4096 // (op 3873) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 3873) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 3873) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 3873) most significant half

s_add_u32 m0, m0, 4096 // (op 3873) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 3873) Load value direct to lds





v_mov_b64 v[36:37], s[40:41] // (op 3649) convert

v_mov_b32 v27, v24 // (op 3649) 
buffer_load_ubyte v38, v27, s[76:79], 0 offen offset: 0 // (op 3649) Load value
v_add_u32 v27, v27, v36 // (op 3649) 
buffer_load_ubyte v39, v27, s[76:79], 0 offen offset: 0 // (op 3649) Load value





buffer_load_ubyte v27, v23, s[80:83], 0 offen offset:4 // (op 3653) Load value

v_and_b32 v23, -8, v1 // (op 5148) 
v_sub_u32 v36, v1, v23 // (op 5148) 

v_lshlrev_b32 v23, 4, v36 // (op 5148) 

v_lshrrev_b32 v36, 3, v1 // (op 5148) 
v_lshl_add_u32 v37, v36, 7, v23 // (op 5148) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 5148) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5431) 


s_nop 0
s_add_u32 s53, 20480, s52 // (op 3878) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s53 // (op 3878) reg expression
s_mov_b32 s54, 128 // (op 3878) call()
buffer_load_dwordx4 v32, s[72:75], s54 offen offset: 0 lds // (op 3878) Load value direct to lds




v_and_b32 v23, -8, v1 // (op 5104) 
v_sub_u32 v32, v1, v23 // (op 5104) 

v_lshlrev_b32 v23, 4, v32 // (op 5104) 

v_lshrrev_b32 v32, 3, v1 // (op 5104) 
v_lshl_add_u32 v33, v32, 7, v23 // (op 5104) 


s_nop 0
v_readfirstlane_b32 s52, v33 // (op 5104) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5422) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 3880) 
v_mov_b64 v[32:33], s[54:55] // (op 3880) convert


v_mov_b64 v[36:37], v[28:29] // (op 3880) 
s_add_u32 s53, 24576, s52 // (op 3880) 
s_mov_b32 m0, s53 // (op 3880) reg expression
s_mov_b32 s54, 128 // (op 3880) call()
buffer_load_dwordx4 v36, s[68:71], s54 offen offset: 0 lds // (op 3880) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v32 // (op 3880) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 3880) most significant half

s_add_u32 m0, m0, 4096 // (op 3880) 
buffer_load_dwordx4 v36, s[68:71], s54 offen offset: 0 lds // (op 3880) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v32 // (op 3880) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 3880) most significant half

s_add_u32 m0, m0, 4096 // (op 3880) 
buffer_load_dwordx4 v36, s[68:71], s54 offen offset: 0 lds // (op 3880) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v32 // (op 3880) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 3880) most significant half

s_add_u32 m0, m0, 4096 // (op 3880) 
buffer_load_dwordx4 v36, s[68:71], s54 offen offset: 0 lds // (op 3880) Load value direct to lds







v_mov_b64 v[28:29], s[40:41] // (op 3663) convert

v_mov_b32 v23, v24 // (op 3663) 
buffer_load_ubyte v32, v23, s[76:79], 0 offen offset:4 // (op 3663) Load value
v_add_u32 v23, v23, v28 // (op 3663) 
buffer_load_ubyte v33, v23, s[76:79], 0 offen offset:4 // (op 3663) Load value



v_and_b32 v23, -4, v1 // (op 4630) 
v_sub_u32 v24, v1, v23 // (op 4630) 

v_lshrrev_b32 v23, 2, v1 // (op 4630) 
v_lshl_add_u32 v28, v23, 3, v24 // (op 4630) 


s_barrier  // (op 5320) 



s_waitcnt vmcnt(9)
ds_write_b8 v28, v38 offset:40960 // (op 3667) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v28, v39 offset:40964 // (op 3667) Store local data 

s_barrier  // (op 5298) 



ds_read_b128 v[40:43], v19 // (op 3700) Load local data 
ds_read_b128 v[44:47], v19 offset:64 // (op 3700) Load local data 
s_barrier  // (op 5279) 



ds_read_b128 v[48:51], v17 offset:4096 // (op 3681) Load local data 
ds_read_b128 v[52:55], v17 offset:4160 // (op 3681) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5317) 



ds_read_u8 v23, v16 offset:40960 // (op 3704) Load local data 



ds_read_b128 v[56:59], v17 offset:8192 // (op 3686) Load local data 
ds_read_b128 v[60:63], v17 offset:8256 // (op 3686) Load local data 



ds_read_u8 v24, v16 offset:41088 // (op 3709) Load local data 



ds_read_b128 v[64:67], v17 offset:12288 // (op 3691) Load local data 
ds_read_b128 v[68:71], v17 offset:12352 // (op 3691) Load local data 



ds_read_u8 v28, v16 offset:41216 // (op 3714) Load local data 



ds_read_b128 v[72:75], v17 offset:16384 // (op 3696) Load local data 
ds_read_b128 v[76:79], v17 offset:16448 // (op 3696) Load local data 



ds_read_u8 v29, v16 offset:41344 // (op 3719) Load local data 
v_and_b32 v36, -8, v1 // (op 4461) 
v_sub_u32 v37, v1, v36 // (op 4461) 

v_lshlrev_b32 v36, 4, v37 // (op 4461) 

v_lshrrev_b32 v37, 3, v1 // (op 4461) 
v_lshl_add_u32 v80, v37, 7, v36 // (op 4461) 


s_nop 0
v_readfirstlane_b32 s52, v80 // (op 4461) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v36, -8, v1 // (op 4536) 
v_sub_u32 v37, v1, v36 // (op 4536) 

v_lshlrev_b32 v36, 4, v37 // (op 4536) 

v_lshrrev_b32 v37, 3, v1 // (op 4536) 
v_lshl_add_u32 v80, v37, 7, v36 // (op 4536) 


s_nop 0
v_readfirstlane_b32 s53, v80 // (op 4536) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v36, -4, v1 // (op 4601) 
v_sub_u32 v37, v1, v36 // (op 4601) 

v_lshrrev_b32 v36, 2, v1 // (op 4601) 
v_lshl_add_u32 v80, v36, 3, v37 // (op 4601) 


v_and_b32 v36, -8, v1 // (op 5089) 
v_sub_u32 v37, v1, v36 // (op 5089) 

v_lshlrev_b32 v36, 4, v37 // (op 5089) 

v_lshrrev_b32 v37, 3, v1 // (op 5089) 
v_lshl_add_u32 v81, v37, 7, v36 // (op 5089) 


s_nop 0
v_readfirstlane_b32 s54, v81 // (op 5089) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v36, -8, v1 // (op 5133) 
v_sub_u32 v37, v1, v36 // (op 5133) 

v_lshlrev_b32 v36, 4, v37 // (op 5133) 

v_lshrrev_b32 v37, 3, v1 // (op 5133) 
v_lshl_add_u32 v81, v37, 7, v36 // (op 5133) 


s_nop 0
v_readfirstlane_b32 s55, v81 // (op 5133) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v36, -4, v1 // (op 5172) 
v_sub_u32 v37, v1, v36 // (op 5172) 

v_lshrrev_b32 v36, 2, v1 // (op 5172) 
v_lshl_add_u32 v81, v36, 3, v37 // (op 5172) 


s_mov_b32 s1, 0 // (op 1788) call()
s_and_b32 s84, s7, 4294967294 // (op 1787) 
s_cmp_lt_u32 s1, s84 // (op 1787) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1787 // (op 1787) Condition: Top (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1787 if false)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1787:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[40:47], a[0:3], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 82) 
s_add_u32 s1, s1, 2 // (op 1789) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[40:47], a[4:7], v24, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1499) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:71], v[40:47], a[8:11], v28, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1661) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:79], v[40:47], a[12:15], v29, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1823) 


v_mov_b32 v36, v26 // (op 80) 

buffer_load_ubyte v21, v36, s[80:83], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 5304) 


v_mov_b64 v[82:83], v[34:35] // (op 3793) 

s_mov_b32 m0, s53 // (op 3793) reg expression
s_mov_b32 s84, 256 // (op 3793) call()
buffer_load_dwordx4 v82, s[72:75], s84 offen offset: 0 lds // (op 3793) Load value direct to lds

s_barrier  // (op 5285) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 3788) 
v_mov_b64 v[84:85], s[84:85] // (op 3788) convert


v_mov_b64 v[86:87], v[30:31] // (op 3788) 
v_mov_b64 v[88:89], v[86:87] // (op 3788) 
s_add_u32 s84, 4096, s52 // (op 3788) 
s_mov_b32 m0, s84 // (op 3788) reg expression
s_mov_b32 s85, 256 // (op 3788) call()
buffer_load_dwordx4 v88, s[68:71], s85 offen offset: 0 lds // (op 3788) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v84 // (op 3788) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3788) most significant half

s_add_u32 m0, m0, 4096 // (op 3788) 
buffer_load_dwordx4 v88, s[68:71], s85 offen offset: 0 lds // (op 3788) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v84 // (op 3788) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3788) most significant half

s_add_u32 m0, m0, 4096 // (op 3788) 
buffer_load_dwordx4 v88, s[68:71], s85 offen offset: 0 lds // (op 3788) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v84 // (op 3788) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3788) most significant half

s_add_u32 m0, m0, 4096 // (op 3788) 
buffer_load_dwordx4 v88, s[68:71], s85 offen offset: 0 lds // (op 3788) Load value direct to lds





v_mov_b64 v[84:85], s[40:41] // (op 17) convert

v_mov_b32 v37, v25 // (op 17) 
v_mov_b32 v88, v37 // (op 17) 
buffer_load_ubyte v38, v88, s[76:79], 0 offen offset:8 // (op 17) Load value
v_add_u32 v88, v88, v84 // (op 17) 
buffer_load_ubyte v39, v88, s[76:79], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 5440) 



s_waitcnt vmcnt(9)
ds_write_b8 v81, v32 offset:41472 // (op 1389) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v81, v33 offset:41476 // (op 1389) Store local data 
s_barrier  // (op 5428) 



ds_read_b128 v[40:43], v6 offset:20480 // (op 1382) Load local data 
ds_read_b128 v[44:47], v6 offset:20544 // (op 1382) Load local data 
s_barrier  // (op 5419) 



ds_read_b128 v[48:51], v13 offset:24576 // (op 1380) Load local data 
ds_read_b128 v[52:55], v13 offset:24640 // (op 1380) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5437) 



ds_read_u8 v23, v12 offset:41472 // (op 1384) Load local data 



ds_read_b128 v[56:59], v13 offset:28672 // (op 1506) Load local data 
ds_read_b128 v[60:63], v13 offset:28736 // (op 1506) Load local data 



ds_read_u8 v24, v12 offset:41600 // (op 1510) Load local data 



ds_read_b128 v[64:67], v13 offset:32768 // (op 1668) Load local data 
ds_read_b128 v[68:71], v13 offset:32832 // (op 1668) Load local data 



ds_read_u8 v28, v12 offset:41728 // (op 1672) Load local data 



ds_read_b128 v[72:75], v13 offset:36864 // (op 1830) Load local data 
ds_read_b128 v[76:79], v13 offset:36928 // (op 1830) Load local data 



ds_read_u8 v29, v12 offset:41856 // (op 1834) Load local data 
s_barrier  // (op 5615) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[40:47], a[0:3], v23, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1387) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[40:47], a[4:7], v24, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1500) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:71], v[40:47], a[8:11], v28, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1662) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:79], v[40:47], a[12:15], v29, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1824) 



buffer_load_ubyte v27, v36, s[80:83], 0 offen offset:12 // (op 1386) Load value

s_barrier  // (op 5434) 
v_add_u32 v26, 8, v26 // (op 4389) 


s_add_u32 s84, 20480, s55 // (op 3849) 

s_mov_b32 m0, s84 // (op 3849) reg expression
s_mov_b32 s85, 384 // (op 3849) call()
buffer_load_dwordx4 v82, s[72:75], s85 offen offset: 0 lds // (op 3849) Load value direct to lds



s_barrier  // (op 5425) 
v_add_co_u32 v34, vcc, 256, v34 // (op 4133) least significant half

v_addc_co_u32 v35, vcc, 0, v35, vcc // (op 4133) most significant half
s_lshl_b64 s[84:85], s[30:31], 5 // (op 3844) 
v_mov_b64 v[82:83], s[84:85] // (op 3844) convert


v_mov_b64 v[84:85], v[86:87] // (op 3844) 
s_add_u32 s84, 24576, s54 // (op 3844) 
s_mov_b32 m0, s84 // (op 3844) reg expression
s_mov_b32 s85, 384 // (op 3844) call()
buffer_load_dwordx4 v84, s[68:71], s85 offen offset: 0 lds // (op 3844) Load value direct to lds
v_add_co_u32 v84, s[86:87], v84, v82 // (op 3844) least significant half
v_addc_co_u32 v85, s[86:87], 0, v85, s[86:87] // (op 3844) most significant half

s_add_u32 m0, m0, 4096 // (op 3844) 
buffer_load_dwordx4 v84, s[68:71], s85 offen offset: 0 lds // (op 3844) Load value direct to lds
v_add_co_u32 v84, s[86:87], v84, v82 // (op 3844) least significant half
v_addc_co_u32 v85, s[86:87], 0, v85, s[86:87] // (op 3844) most significant half

s_add_u32 m0, m0, 4096 // (op 3844) 
buffer_load_dwordx4 v84, s[68:71], s85 offen offset: 0 lds // (op 3844) Load value direct to lds
v_add_co_u32 v84, s[86:87], v84, v82 // (op 3844) least significant half
v_addc_co_u32 v85, s[86:87], 0, v85, s[86:87] // (op 3844) most significant half

s_add_u32 m0, m0, 4096 // (op 3844) 
buffer_load_dwordx4 v84, s[68:71], s85 offen offset: 0 lds // (op 3844) Load value direct to lds






v_add_co_u32 v30, vcc, 256, v30 // (op 4005) least significant half
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 4005) most significant half
v_mov_b64 v[82:83], s[40:41] // (op 1391) convert

v_mov_b32 v36, v37 // (op 1391) 
buffer_load_ubyte v32, v36, s[76:79], 0 offen offset:12 // (op 1391) Load value
v_add_u32 v36, v36, v82 // (op 1391) 
buffer_load_ubyte v33, v36, s[76:79], 0 offen offset:12 // (op 1391) Load value



v_add_u32 v25, 8, v25 // (op 4261) 
s_barrier  // (op 5323) 



s_waitcnt vmcnt(9)
ds_write_b8 v80, v38 offset:40960 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v80, v39 offset:40964 // (op 21) Store local data 
s_barrier  // (op 5310) 



ds_read_b128 v[40:43], v3 // (op 70) Load local data 
ds_read_b128 v[44:47], v3 offset:64 // (op 70) Load local data 
s_barrier  // (op 5291) 



ds_read_b128 v[48:51], v7 offset:4096 // (op 65) Load local data 
ds_read_b128 v[52:55], v7 offset:4160 // (op 65) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5330) 



ds_read_u8 v23, v0 offset:40960 // (op 75) Load local data 



ds_read_b128 v[56:59], v7 offset:8192 // (op 1493) Load local data 
ds_read_b128 v[60:63], v7 offset:8256 // (op 1493) Load local data 



ds_read_u8 v24, v0 offset:41088 // (op 1497) Load local data 



ds_read_b128 v[64:67], v7 offset:12288 // (op 1655) Load local data 
ds_read_b128 v[68:71], v7 offset:12352 // (op 1655) Load local data 



ds_read_u8 v28, v0 offset:41216 // (op 1659) Load local data 



ds_read_b128 v[72:75], v7 offset:16384 // (op 1817) Load local data 
ds_read_b128 v[76:79], v7 offset:16448 // (op 1817) Load local data 



ds_read_u8 v29, v0 offset:41344 // (op 1821) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 1787) 
s_cmp_lt_u32 s1, s84 // (op 1787) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1787 // (op 1787) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1787 if true)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1787:












s_barrier  // (op 5599) 
s_and_b32 s52, s7, 4294967294 // (op 3917) 
v_lshrrev_b32 v21, 3, v1 // (op 3917) 
s_add_u32 s53, s0, s52 // (op 3917) 

v_and_b32 v25, -8, v1 // (op 3917) 
v_sub_u32 v26, v1, v25 // (op 3917) 

s_mul_i32 s52, s16, s8 // (op 3917) 
s_add_u32 s52, s52, s53 // (op 3917) 

v_lshlrev_b32 v25, 4, v26 // (op 3917) 

s_mul_hi_u32 s53, s52, s13 // (op 3917) 
s_sub_u32 s54, s52, s53 // (op 3917) 
s_lshr_b32 s55, s54, 1 // (op 3917) 

s_add_u32 s54, s55, s53 // (op 3917) 


s_lshr_b32 s53, s54, s11 // (op 3917) 

s_and_b32 s54, s11, 2147483648 // (op 3917) 
s_cmp_eq_i32 s54, 0 // (op 3917) 
s_mov_b32 s55, scc // (op 3917) 

s_cmp_eq_i32 s55, 0 // (op 3917) 
s_cselect_b32 s54, s52, s53 // (op 3917) 


s_mul_hi_u32 s53, s54, s12 // (op 3917) 
s_sub_u32 s55, s54, s53 // (op 3917) 
s_lshr_b32 s84, s55, 1 // (op 3917) 

s_add_u32 s55, s84, s53 // (op 3917) 


s_lshr_b32 s53, s55, s10 // (op 3917) 

s_and_b32 s55, s10, 2147483648 // (op 3917) 
s_cmp_eq_i32 s55, 0 // (op 3917) 
s_mov_b32 s84, scc // (op 3917) 

s_cmp_eq_i32 s84, 0 // (op 3917) 
s_cselect_b32 s55, s54, s53 // (op 3917) 


v_lshl_add_u32 v26, s55, 7, v21 // (op 3917) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 3917) 
s_mul_i32 s53, s54, s84 // (op 3917) 


s_sub_u32 s54, s52, s53 // (op 3917) 


v_lshl_add_u32 v21, s54, 7, v25 // (op 3917) 


v_mov_b32 v31, 0 // (op 3917) convert
v_mov_b32 v30, v21 // (op 3917) convert

v_mul_lo_u32 v35, v26, s31 // (op 3917) most significant: low of low * high
v_mul_hi_u32 v21, v26, s30 // (op 3917) most significant: high of low * low
v_mul_lo_u32 v34, v26, s30 // (op 3917) least significant: low of low * low
v_add_u32 v35, v35, v21 // (op 3917) most significant: sum

v_add_co_u32 v34, s[52:53], v34, v30 // (op 3917) least significant half
v_addc_co_u32 v35, s[52:53], v35, v31, s[52:53] // (op 3917) most significant half



s_mul_i32 s53, s30, s35 // (op 3919) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 3919) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 3919) least significant
s_mul_hi_u32 s55, s30, s34 // (op 3919) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3919) most significant: sum
s_add_u32 s53, s53, s55 // (op 3919) most significant: sum


s_mov_b32 s68, s32 // (op 3919) reg expression
s_mov_b32 s69, s33 // (op 3919) reg expression
s_mov_b32 s70, s52 // (op 3919) reg expression
s_mov_b32 s71, 131072 // (op 3919) call()

s_and_b32 s52, s7, 4294967294 // (op 4057) 
v_lshrrev_b32 v21, 3, v1 // (op 4057) 
s_add_u32 s53, s0, s52 // (op 4057) 

v_and_b32 v25, -8, v1 // (op 4057) 
v_sub_u32 v26, v1, v25 // (op 4057) 

s_mul_i32 s52, s16, s8 // (op 4057) 
s_add_u32 s52, s52, s53 // (op 4057) 

v_lshlrev_b32 v25, 4, v26 // (op 4057) 

s_mul_hi_u32 s53, s52, s13 // (op 4057) 
s_sub_u32 s54, s52, s53 // (op 4057) 
s_lshr_b32 s55, s54, 1 // (op 4057) 

s_add_u32 s54, s55, s53 // (op 4057) 


s_lshr_b32 s53, s54, s11 // (op 4057) 

s_and_b32 s54, s11, 2147483648 // (op 4057) 
s_cmp_eq_i32 s54, 0 // (op 4057) 
s_mov_b32 s55, scc // (op 4057) 

s_cmp_eq_i32 s55, 0 // (op 4057) 
s_cselect_b32 s54, s52, s53 // (op 4057) 


s_mul_hi_u32 s53, s54, s12 // (op 4057) 
s_sub_u32 s55, s54, s53 // (op 4057) 
s_lshr_b32 s84, s55, 1 // (op 4057) 

s_add_u32 s55, s84, s53 // (op 4057) 


s_lshr_b32 s53, s55, s10 // (op 4057) 

s_and_b32 s55, s10, 2147483648 // (op 4057) 
s_cmp_eq_i32 s55, 0 // (op 4057) 
s_mov_b32 s84, scc // (op 4057) 

s_cmp_eq_i32 s84, 0 // (op 4057) 
s_cselect_b32 s55, s54, s53 // (op 4057) 


s_mul_i32 s53, s55, s14 // (op 4057) 

s_sub_u32 s55, s54, s53 // (op 4057) 

v_lshl_add_u32 v26, s55, 5, v21 // (op 4057) 


v_mul_lo_u32 v31, v26, s37 // (op 4057) most significant: low of low * high
v_mul_hi_u32 v21, v26, s36 // (op 4057) most significant: high of low * low
v_mul_lo_u32 v30, v26, s36 // (op 4057) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 4057) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4057) 
s_mul_i32 s53, s54, s84 // (op 4057) 


s_sub_u32 s54, s52, s53 // (op 4057) 


v_lshl_add_u32 v21, s54, 7, v25 // (op 4057) 


v_mov_b32 v37, 0 // (op 4057) convert
v_mov_b32 v36, v21 // (op 4057) convert

v_add_co_u32 v38, s[52:53], v36, v30 // (op 4057) least significant half
v_addc_co_u32 v39, s[52:53], v37, v31, s[52:53] // (op 4057) most significant half



s_mul_i32 s53, s36, s23 // (op 4059) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 4059) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 4059) least significant
s_mul_hi_u32 s55, s36, s22 // (op 4059) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4059) most significant: sum
s_add_u32 s53, s53, s55 // (op 4059) most significant: sum


s_mov_b32 s72, s38 // (op 4059) reg expression
s_mov_b32 s73, s39 // (op 4059) reg expression
s_mov_b32 s74, s52 // (op 4059) reg expression
s_mov_b32 s75, 131072 // (op 4059) call()

s_and_b32 s52, s7, 4294967294 // (op 4185) 
v_lshrrev_b32 v21, 2, v1 // (op 4185) 
s_add_u32 s53, s0, s52 // (op 4185) 

v_lshlrev_b32 v25, 1, v21 // (op 4185) 

v_and_b32 v21, -4, v1 // (op 4185) 
s_mul_i32 s52, s16, s8 // (op 4185) 
s_add_u32 s52, s52, s53 // (op 4185) 

v_sub_u32 v26, v1, v21 // (op 4185) 

s_mul_hi_u32 s53, s52, s13 // (op 4185) 
s_sub_u32 s54, s52, s53 // (op 4185) 
s_lshr_b32 s55, s54, 1 // (op 4185) 

s_add_u32 s54, s55, s53 // (op 4185) 


s_lshr_b32 s53, s54, s11 // (op 4185) 

s_and_b32 s54, s11, 2147483648 // (op 4185) 
s_cmp_eq_i32 s54, 0 // (op 4185) 
s_mov_b32 s55, scc // (op 4185) 

s_cmp_eq_i32 s55, 0 // (op 4185) 
s_cselect_b32 s54, s52, s53 // (op 4185) 


s_mul_hi_u32 s53, s54, s12 // (op 4185) 
s_sub_u32 s55, s54, s53 // (op 4185) 
s_lshr_b32 s84, s55, 1 // (op 4185) 

s_add_u32 s55, s84, s53 // (op 4185) 


s_lshr_b32 s53, s55, s10 // (op 4185) 

s_and_b32 s55, s10, 2147483648 // (op 4185) 
s_cmp_eq_i32 s55, 0 // (op 4185) 
s_mov_b32 s84, scc // (op 4185) 

s_cmp_eq_i32 s84, 0 // (op 4185) 
s_cselect_b32 s55, s54, s53 // (op 4185) 


v_lshl_add_u32 v21, s55, 7, v25 // (op 4185) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4185) 
s_mul_i32 s53, s54, s84 // (op 4185) 


s_sub_u32 s54, s52, s53 // (op 4185) 


v_lshl_add_u32 v25, s54, 2, v26 // (op 4185) 


v_mul_lo_u32 v26, v21, s40 // (op 4185) 
v_add_u32 v26, v26, v25 // (op 4185) 


s_mul_i32 s53, s40, s35 // (op 4187) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 4187) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 4187) least significant
s_mul_hi_u32 s55, s40, s34 // (op 4187) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4187) most significant: sum
s_add_u32 s53, s53, s55 // (op 4187) most significant: sum


s_mov_b32 s76, s42 // (op 4187) reg expression
s_mov_b32 s77, s43 // (op 4187) reg expression
s_mov_b32 s78, s52 // (op 4187) reg expression
s_mov_b32 s79, 131072 // (op 4187) call()

s_and_b32 s52, s7, 4294967294 // (op 4313) 
v_lshrrev_b32 v21, 6, v1 // (op 4313) 
s_add_u32 s53, s0, s52 // (op 4313) 

v_and_b32 v25, 1, v21 // (op 4313) 

v_and_b32 v21, 15, v1 // (op 4313) 
s_mul_i32 s52, s16, s8 // (op 4313) 
s_add_u32 s52, s52, s53 // (op 4313) 

v_lshl_add_u32 v30, v25, 4, v21 // (op 4313) 


s_mul_hi_u32 s53, s52, s13 // (op 4313) 
v_and_b32 v21, 63, v1 // (op 4313) 
s_sub_u32 s54, s52, s53 // (op 4313) 
v_lshrrev_b32 v25, 4, v21 // (op 4313) 

s_lshr_b32 s55, s54, 1 // (op 4313) 

s_add_u32 s54, s55, s53 // (op 4313) 


s_lshr_b32 s53, s54, s11 // (op 4313) 

s_and_b32 s54, s11, 2147483648 // (op 4313) 
s_cmp_eq_i32 s54, 0 // (op 4313) 
s_mov_b32 s55, scc // (op 4313) 

s_cmp_eq_i32 s55, 0 // (op 4313) 
s_cselect_b32 s54, s52, s53 // (op 4313) 


s_mul_hi_u32 s53, s54, s12 // (op 4313) 
s_sub_u32 s55, s54, s53 // (op 4313) 
s_lshr_b32 s84, s55, 1 // (op 4313) 

s_add_u32 s55, s84, s53 // (op 4313) 


s_lshr_b32 s53, s55, s10 // (op 4313) 

s_and_b32 s55, s10, 2147483648 // (op 4313) 
s_cmp_eq_i32 s55, 0 // (op 4313) 
s_mov_b32 s84, scc // (op 4313) 

s_cmp_eq_i32 s84, 0 // (op 4313) 
s_cselect_b32 s55, s54, s53 // (op 4313) 


s_mul_i32 s53, s55, s14 // (op 4313) 

s_sub_u32 s55, s54, s53 // (op 4313) 

v_lshl_add_u32 v21, s55, 5, v30 // (op 4313) 


v_mul_lo_u32 v30, v21, s44 // (op 4313) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 4313) 
s_mul_i32 s53, s54, s84 // (op 4313) 


s_sub_u32 s54, s52, s53 // (op 4313) 


v_lshl_add_u32 v21, s54, 2, v25 // (op 4313) 


v_add_u32 v25, v21, v30 // (op 4313) 


s_mul_i32 s53, s44, s23 // (op 4315) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4315) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4315) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4315) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4315) most significant: sum
s_add_u32 s53, s53, s55 // (op 4315) most significant: sum


s_mov_b32 s80, s46 // (op 4315) reg expression
s_mov_b32 s81, s47 // (op 4315) reg expression
s_mov_b32 s82, s52 // (op 4315) reg expression
s_mov_b32 s83, 131072 // (op 4315) call()

v_and_b32 v21, -8, v1 // (op 4474) 
v_sub_u32 v30, v1, v21 // (op 4474) 

v_lshlrev_b32 v21, 4, v30 // (op 4474) 

v_lshrrev_b32 v30, 3, v1 // (op 4474) 
v_lshl_add_u32 v31, v30, 7, v21 // (op 4474) 


s_nop 0
v_readfirstlane_b32 s52, v31 // (op 4474) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 4549) 
v_sub_u32 v30, v1, v21 // (op 4549) 

v_lshlrev_b32 v21, 4, v30 // (op 4549) 

v_lshrrev_b32 v30, 3, v1 // (op 4549) 
v_lshl_add_u32 v31, v30, 7, v21 // (op 4549) 


s_nop 0
v_readfirstlane_b32 s53, v31 // (op 4549) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -4, v1 // (op 4614) 
v_sub_u32 v30, v1, v21 // (op 4614) 

v_lshrrev_b32 v21, 2, v1 // (op 4614) 
v_lshl_add_u32 v31, v21, 3, v30 // (op 4614) 


s_and_b32 s54, s7, 4294967294 // (op 1793) 
s_cmp_lt_u32 s54, s7 // (op 1792) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1792 // (op 1792) Condition: Top (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1792 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1792:
s_barrier  // (op 5288) 
s_add_u32 s54, s54, 1 // (op 1794) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 3831) 
v_mov_b64 v[32:33], s[84:85] // (op 3831) convert


v_mov_b64 v[36:37], v[34:35] // (op 3831) 
v_mov_b64 v[28:29], v[36:37] // (op 3831) 
s_add_u32 s55, 4096, s52 // (op 3831) 
s_mov_b32 m0, s55 // (op 3831) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 3831) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v32 // (op 3831) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 3831) most significant half

s_add_u32 m0, m0, 4096 // (op 3831) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 3831) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v32 // (op 3831) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 3831) most significant half

s_add_u32 m0, m0, 4096 // (op 3831) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 3831) Load value direct to lds
v_add_co_u32 v28, s[84:85], v28, v32 // (op 3831) least significant half
v_addc_co_u32 v29, s[84:85], 0, v29, s[84:85] // (op 3831) most significant half

s_add_u32 m0, m0, 4096 // (op 3831) 
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 3831) Load value direct to lds




v_add_co_u32 v34, vcc, 128, v34 // (op 3907) least significant half
v_addc_co_u32 v35, vcc, 0, v35, vcc // (op 3907) most significant half
s_barrier  // (op 5307) 


v_mov_b64 v[32:33], v[38:39] // (op 3837) 

s_mov_b32 m0, s53 // (op 3837) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3837) Load value direct to lds

v_add_co_u32 v38, vcc, 128, v38 // (op 4049) least significant half
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 4049) most significant half

v_mov_b64 v[32:33], s[40:41] // (op 1333) convert

v_mov_b32 v21, v26 // (op 1333) 
v_mov_b32 v23, v21 // (op 1333) 
buffer_load_ubyte v36, v23, s[76:79], 0 offen offset: 0 // (op 1333) Load value
v_add_u32 v23, v23, v32 // (op 1333) 
buffer_load_ubyte v37, v23, s[76:79], 0 offen offset: 0 // (op 1333) Load value



v_add_u32 v26, 4, v26 // (op 4177) 
s_barrier  // (op 5326) 
s_waitcnt vmcnt(3)
s_barrier  // (op 5294) 






ds_read_b128 v[40:43], v7 offset:4096 // (op 1322) Load local data 
ds_read_b128 v[44:47], v7 offset:4160 // (op 1322) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v31, v36 offset:40960 // (op 1331) Store local data 
s_barrier  // (op 5313) 
s_waitcnt vmcnt(0)
ds_write_b8 v31, v37 offset:40964 // (op 1331) Store local data 




ds_read_b128 v[48:51], v3 // (op 1324) Load local data 
ds_read_b128 v[52:55], v3 offset:64 // (op 1324) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5333) 



ds_read_u8 v21, v0 offset:40960 // (op 1326) Load local data 


v_mov_b32 v23, v25 // (op 1328) 

buffer_load_ubyte v24, v23, s[80:83], 0 offen offset: 0 // (op 1328) Load value

v_add_u32 v25, 4, v25 // (op 4305) 



ds_read_b128 v[56:59], v7 offset:8192 // (op 1474) Load local data 
ds_read_b128 v[60:63], v7 offset:8256 // (op 1474) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:55], a[0:3], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1329) 





ds_read_u8 v21, v0 offset:41088 // (op 1478) Load local data 



ds_read_b128 v[40:43], v7 offset:12288 // (op 1636) Load local data 
ds_read_b128 v[44:47], v7 offset:12352 // (op 1636) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[48:55], a[4:7], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1480) 





ds_read_u8 v21, v0 offset:41216 // (op 1640) Load local data 



ds_read_b128 v[56:59], v7 offset:16384 // (op 1798) Load local data 
ds_read_b128 v[60:63], v7 offset:16448 // (op 1798) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[48:55], a[8:11], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1642) 





ds_read_u8 v21, v0 offset:41344 // (op 1802) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[48:55], a[12:15], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 1804) 




s_cmp_lt_u32 s54, s7 // (op 1792) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1792 // (op 1792) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1792 if true)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1792:








s_barrier  // (op 5603) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 272) 
s_cmp_lg_u32 s18, 0 // (op 298) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_298 // (op 298) Condition: False, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_298



v_add_u32 v19, 12288, v20 // (op 2047) 
s_nop 0
buffer_store_dword a12, v19, s[64:67], 0 offen sc1 // (op 2047) Store value

v_add_u32 v19, 12416, v20 // (op 2047) 
buffer_store_dword a13, v19, s[64:67], 0 offen sc1 // (op 2047) Store value

v_add_u32 v19, 12544, v20 // (op 2047) 
buffer_store_dword a14, v19, s[64:67], 0 offen sc1 // (op 2047) Store value

v_add_u32 v19, 12672, v20 // (op 2047) 
buffer_store_dword a15, v19, s[64:67], 0 offen sc1 // (op 2047) Store value




v_add_u32 v19, 8192, v20 // (op 2040) 
buffer_store_dword a8, v19, s[64:67], 0 offen sc1 // (op 2040) Store value

v_add_u32 v19, 8320, v20 // (op 2040) 
buffer_store_dword a9, v19, s[64:67], 0 offen sc1 // (op 2040) Store value

v_add_u32 v19, 8448, v20 // (op 2040) 
buffer_store_dword a10, v19, s[64:67], 0 offen sc1 // (op 2040) Store value

v_add_u32 v19, 8576, v20 // (op 2040) 
buffer_store_dword a11, v19, s[64:67], 0 offen sc1 // (op 2040) Store value




v_add_u32 v19, 4096, v20 // (op 2033) 
buffer_store_dword a4, v19, s[64:67], 0 offen sc1 // (op 2033) Store value

v_add_u32 v19, 4224, v20 // (op 2033) 
buffer_store_dword a5, v19, s[64:67], 0 offen sc1 // (op 2033) Store value

v_add_u32 v19, 4352, v20 // (op 2033) 
buffer_store_dword a6, v19, s[64:67], 0 offen sc1 // (op 2033) Store value

v_add_u32 v19, 4480, v20 // (op 2033) 
buffer_store_dword a7, v19, s[64:67], 0 offen sc1 // (op 2033) Store value




buffer_store_dword a0, v20, s[64:67], 0 offen sc1 // (op 295) Store value
buffer_store_dword a1, v20, s[64:67], 0 offen offset:128 sc1 // (op 295) Store value
buffer_store_dword a2, v20, s[64:67], 0 offen offset:256 sc1 // (op 295) Store value
buffer_store_dword a3, v20, s[64:67], 0 offen offset:384 sc1 // (op 295) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 299) 
v_cmp_eq_i32 vcc, v1, 0 // (op 303) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303 // (op 303) Condition: False, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303
s_mov_b32 s52, 1 // (op 301) call()
s_mov_b32 s55, 0 // (op 302) convert
s_mov_b32 s54, s16 // (op 302) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 302) 
s_add_u32 s84, s48, s54 // (op 302) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 302) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 302) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303 // (op 303) Condition: Done, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303:
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_298 // (op 298) Condition: Done, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_298
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_298:
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_298:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 314) 
s_mov_b32 s52, scc // (op 314) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 314) 
s_mov_b32 s53, s54 // (op 314) convert

s_sub_u32 s54, s53, 1 // (op 314) 

s_cmp_lt_u32 s19, s54 // (op 314) 
s_mov_b32 s53, scc // (op 314) 

s_and_b32 s54, s52, s53 // (op 314) 
s_cmp_lg_u32 s54, 0 // (op 314) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_314
s_mov_b32 s52, 0 // (op 280) call()
s_mul_i32 s53, s52, s8 // (op 279) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 279) 
s_mov_b32 s84, s54 // (op 279) convert

s_sub_u32 s54, s84, s19 // (op 279) 

s_sub_u32 s55, s54, 1 // (op 279) 

s_cmp_lt_u32 s53, s55 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 // (op 279) Condition: Top (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 if false)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_279:
s_add_u32 s53, s16, 1 // (op 4962) 
v_lshrrev_b32 v16, 5, v1 // (op 4962) 
s_add_u32 s54, s53, s52 // (op 4962) 
s_lshl_b32 s54, s54, 7 // (op 4962) 

v_lshlrev_b32 v17, 2, v16 // (op 4962) 

v_add_lshl_u32 v16, s54, v17, 5 // (op 4962) 


v_and_b32 v17, -32, v1 // (op 4962) 
v_sub_u32 v19, v1, v17 // (op 4962) 

v_add_lshl_u32 v17, v16, v19, 2 // (op 4962) 


s_lshl_b32 s53, s15, 14 // (op 4964) 
s_mov_b32 s64, s28 // (op 4964) reg expression
s_mov_b32 s65, s29 // (op 4964) reg expression
s_mov_b32 s66, s53 // (op 4964) reg expression
s_mov_b32 s67, 131072 // (op 4964) call()

RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_318:
s_add_u32 s53, s16, 1 // (op 316) 
s_add_u32 s54, s53, s52 // (op 316) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 316) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 316) 

s_add_u32 s84, s48, s54 // (op 316) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 316) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 316) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 318) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 // (op 318) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 if true)
s_barrier  // (op 322) 
v_cmp_eq_i32 vcc, v1, 0 // (op 321) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321 // (op 321) Condition: False, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321
s_mov_b32 s53, 0 // (op 319) call()
s_add_u32 s54, s16, 1 // (op 320) 
s_add_u32 s84, s54, s52 // (op 320) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 320) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 320) 

s_add_u32 s54, s48, s84 // (op 320) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 320) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 320) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321 // (op 321) Condition: Done, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321:
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321:




buffer_load_dword v24, v17, s[64:67], 0 offen // (op 333) Load value
buffer_load_dword v25, v17, s[64:67], 0 offen offset:128 // (op 333) Load value
buffer_load_dword v26, v17, s[64:67], 0 offen offset:256 // (op 333) Load value
buffer_load_dword v27, v17, s[64:67], 0 offen offset:384 // (op 333) Load value
v_accvgpr_read v28, a0 // (op 336) reg expression
v_accvgpr_read v29, a1 // (op 336) reg expression
v_accvgpr_read v30, a2 // (op 336) reg expression
v_accvgpr_read v31, a3 // (op 336) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 338) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 338) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 338) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 338) 

v_accvgpr_write a0, v28 // (op 340) reg expression
v_accvgpr_write a1, v29 // (op 340) reg expression
v_accvgpr_write a2, v30 // (op 340) reg expression
v_accvgpr_write a3, v31 // (op 340) reg expression




v_mov_b32 v16, 4096 // (op 2073) 
v_add_i32 v19, v17, v16 // (op 2073) 

buffer_load_dword v24, v19, s[64:67], 0 offen // (op 2073) Load value

v_mov_b32 v16, 4224 // (op 2073) 
v_add_i32 v23, v17, v16 // (op 2073) 

buffer_load_dword v25, v23, s[64:67], 0 offen // (op 2073) Load value

v_mov_b32 v16, 4352 // (op 2073) 
v_add_i32 v23, v17, v16 // (op 2073) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 2073) Load value

v_mov_b32 v16, 4480 // (op 2073) 
v_add_i32 v23, v17, v16 // (op 2073) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 2073) Load value

v_accvgpr_read v28, a4 // (op 2074) reg expression
v_accvgpr_read v29, a5 // (op 2074) reg expression
v_accvgpr_read v30, a6 // (op 2074) reg expression
v_accvgpr_read v31, a7 // (op 2074) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 2075) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 2075) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 2075) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 2075) 

v_accvgpr_write a4, v28 // (op 2076) reg expression
v_accvgpr_write a5, v29 // (op 2076) reg expression
v_accvgpr_write a6, v30 // (op 2076) reg expression
v_accvgpr_write a7, v31 // (op 2076) reg expression




v_mov_b32 v16, 8192 // (op 2086) 
v_add_i32 v19, v17, v16 // (op 2086) 

buffer_load_dword v24, v19, s[64:67], 0 offen // (op 2086) Load value

v_mov_b32 v16, 8320 // (op 2086) 
v_add_i32 v23, v17, v16 // (op 2086) 

buffer_load_dword v25, v23, s[64:67], 0 offen // (op 2086) Load value

v_mov_b32 v16, 8448 // (op 2086) 
v_add_i32 v23, v17, v16 // (op 2086) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 2086) Load value

v_mov_b32 v16, 8576 // (op 2086) 
v_add_i32 v23, v17, v16 // (op 2086) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 2086) Load value

v_accvgpr_read v28, a8 // (op 2087) reg expression
v_accvgpr_read v29, a9 // (op 2087) reg expression
v_accvgpr_read v30, a10 // (op 2087) reg expression
v_accvgpr_read v31, a11 // (op 2087) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 2088) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 2088) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 2088) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 2088) 

v_accvgpr_write a8, v28 // (op 2089) reg expression
v_accvgpr_write a9, v29 // (op 2089) reg expression
v_accvgpr_write a10, v30 // (op 2089) reg expression
v_accvgpr_write a11, v31 // (op 2089) reg expression




v_mov_b32 v16, 12288 // (op 2099) 
v_add_i32 v19, v17, v16 // (op 2099) 

buffer_load_dword v24, v19, s[64:67], 0 offen // (op 2099) Load value

v_mov_b32 v16, 12416 // (op 2099) 
v_add_i32 v23, v17, v16 // (op 2099) 

buffer_load_dword v25, v23, s[64:67], 0 offen // (op 2099) Load value

v_mov_b32 v16, 12544 // (op 2099) 
v_add_i32 v23, v17, v16 // (op 2099) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 2099) Load value

v_mov_b32 v16, 12672 // (op 2099) 
v_add_i32 v23, v17, v16 // (op 2099) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 2099) Load value

v_accvgpr_read v28, a12 // (op 2100) reg expression
v_accvgpr_read v29, a13 // (op 2100) reg expression
v_accvgpr_read v30, a14 // (op 2100) reg expression
v_accvgpr_read v31, a15 // (op 2100) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 2101) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 2101) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 2101) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 2101) 

v_accvgpr_write a12, v28 // (op 2102) reg expression
v_accvgpr_write a13, v29 // (op 2102) reg expression
v_accvgpr_write a14, v30 // (op 2102) reg expression
v_accvgpr_write a15, v31 // (op 2102) reg expression

s_add_u32 s52, s52, 1 // (op 281) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 279) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 279) 
s_mov_b32 s84, s54 // (op 279) convert

s_sub_u32 s54, s84, s19 // (op 279) 

s_sub_u32 s55, s54, 1 // (op 279) 

s_cmp_lt_u32 s53, s55 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_279 // (op 279) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_279 if true)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_279:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_314
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_314:
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_314:

s_cmp_eq_i32 s18, 0 // (op 357) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357 // (op 357) Condition: False, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357
v_cmp_eq_f32 vcc, 0.00000, v14 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v24, a0 // (op 46) 
v_accvgpr_read v25, a1 // (op 46) 
v_accvgpr_read v26, a2 // (op 46) 
v_accvgpr_read v27, a3 // (op 46) 
v_mul_f32 v28, v15, v24 // (op 46) 
v_mul_f32 v29, v15, v25 // (op 46) 
v_mul_f32 v30, v15, v26 // (op 46) 
v_mul_f32 v31, v15, v27 // (op 46) 




buffer_store_dwordx4 v[28:31], v22, s[60:63], 0 offen // (op 52) Store value

v_accvgpr_read v24, a4 // (op 2136) 
v_accvgpr_read v25, a5 // (op 2136) 
v_accvgpr_read v26, a6 // (op 2136) 
v_accvgpr_read v27, a7 // (op 2136) 
v_mul_f32 v28, v15, v24 // (op 2136) 
v_mul_f32 v29, v15, v25 // (op 2136) 
v_mul_f32 v30, v15, v26 // (op 2136) 
v_mul_f32 v31, v15, v27 // (op 2136) 




buffer_store_dwordx4 v[28:31], v22, s[60:63], 0 offen offset:128 // (op 2138) Store value

v_accvgpr_read v24, a8 // (op 2154) 
v_accvgpr_read v25, a9 // (op 2154) 
v_accvgpr_read v26, a10 // (op 2154) 
v_accvgpr_read v27, a11 // (op 2154) 
v_mul_f32 v28, v15, v24 // (op 2154) 
v_mul_f32 v29, v15, v25 // (op 2154) 
v_mul_f32 v30, v15, v26 // (op 2154) 
v_mul_f32 v31, v15, v27 // (op 2154) 




buffer_store_dwordx4 v[28:31], v22, s[60:63], 0 offen offset:256 // (op 2156) Store value

v_accvgpr_read v24, a12 // (op 2172) 
v_accvgpr_read v25, a13 // (op 2172) 
v_accvgpr_read v26, a14 // (op 2172) 
v_accvgpr_read v27, a15 // (op 2172) 
v_mul_f32 v28, v15, v24 // (op 2172) 
v_mul_f32 v29, v15, v25 // (op 2172) 
v_mul_f32 v30, v15, v26 // (op 2172) 
v_mul_f32 v31, v15, v27 // (op 2172) 




buffer_store_dwordx4 v[28:31], v22, s[60:63], 0 offen offset:384 // (op 2174) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[24:27], v18, s[56:59], 0 offen offset:384 // (op 2248) Load value
v_accvgpr_read v28, a12 // (op 2245) 
v_accvgpr_read v29, a13 // (op 2245) 
v_accvgpr_read v30, a14 // (op 2245) 
v_accvgpr_read v31, a15 // (op 2245) 
v_mul_f32 v32, v15, v28 // (op 2245) 
v_mul_f32 v33, v15, v29 // (op 2245) 
v_mul_f32 v34, v15, v30 // (op 2245) 
v_mul_f32 v35, v15, v31 // (op 2245) 

s_waitcnt vmcnt(0)
v_mul_f32 v28, v14, v24 // (op 2249) 
v_mul_f32 v29, v14, v25 // (op 2249) 
v_mul_f32 v30, v14, v26 // (op 2249) 
v_mul_f32 v31, v14, v27 // (op 2249) 

v_add_f32 v24, v28, v32 // (op 2246) 
v_add_f32 v25, v29, v33 // (op 2246) 
v_add_f32 v26, v30, v34 // (op 2246) 
v_add_f32 v27, v31, v35 // (op 2246) 





buffer_store_dwordx4 v[24:27], v22, s[60:63], 0 offen offset:384 // (op 2247) Store value




s_nop 1
buffer_load_dwordx4 v[24:27], v18, s[56:59], 0 offen offset:256 // (op 2230) Load value
v_accvgpr_read v28, a8 // (op 2227) 
v_accvgpr_read v29, a9 // (op 2227) 
v_accvgpr_read v30, a10 // (op 2227) 
v_accvgpr_read v31, a11 // (op 2227) 
v_mul_f32 v32, v15, v28 // (op 2227) 
v_mul_f32 v33, v15, v29 // (op 2227) 
v_mul_f32 v34, v15, v30 // (op 2227) 
v_mul_f32 v35, v15, v31 // (op 2227) 

s_waitcnt vmcnt(0)
v_mul_f32 v28, v14, v24 // (op 2231) 
v_mul_f32 v29, v14, v25 // (op 2231) 
v_mul_f32 v30, v14, v26 // (op 2231) 
v_mul_f32 v31, v14, v27 // (op 2231) 

v_add_f32 v24, v28, v32 // (op 2228) 
v_add_f32 v25, v29, v33 // (op 2228) 
v_add_f32 v26, v30, v34 // (op 2228) 
v_add_f32 v27, v31, v35 // (op 2228) 





buffer_store_dwordx4 v[24:27], v22, s[60:63], 0 offen offset:256 // (op 2229) Store value




s_nop 1
buffer_load_dwordx4 v[24:27], v18, s[56:59], 0 offen offset:128 // (op 2212) Load value
v_accvgpr_read v28, a4 // (op 2209) 
v_accvgpr_read v29, a5 // (op 2209) 
v_accvgpr_read v30, a6 // (op 2209) 
v_accvgpr_read v31, a7 // (op 2209) 
v_mul_f32 v32, v15, v28 // (op 2209) 
v_mul_f32 v33, v15, v29 // (op 2209) 
v_mul_f32 v34, v15, v30 // (op 2209) 
v_mul_f32 v35, v15, v31 // (op 2209) 

s_waitcnt vmcnt(0)
v_mul_f32 v28, v14, v24 // (op 2213) 
v_mul_f32 v29, v14, v25 // (op 2213) 
v_mul_f32 v30, v14, v26 // (op 2213) 
v_mul_f32 v31, v14, v27 // (op 2213) 

v_add_f32 v24, v28, v32 // (op 2210) 
v_add_f32 v25, v29, v33 // (op 2210) 
v_add_f32 v26, v30, v34 // (op 2210) 
v_add_f32 v27, v31, v35 // (op 2210) 





buffer_store_dwordx4 v[24:27], v22, s[60:63], 0 offen offset:128 // (op 2211) Store value




s_nop 1
buffer_load_dwordx4 v[24:27], v18, s[56:59], 0 offen // (op 135) Load value
v_accvgpr_read v28, a0 // (op 132) 
v_accvgpr_read v29, a1 // (op 132) 
v_accvgpr_read v30, a2 // (op 132) 
v_accvgpr_read v31, a3 // (op 132) 
v_mul_f32 v32, v15, v28 // (op 132) 
v_mul_f32 v33, v15, v29 // (op 132) 
v_mul_f32 v34, v15, v30 // (op 132) 
v_mul_f32 v35, v15, v31 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v28, v14, v24 // (op 136) 
v_mul_f32 v29, v14, v25 // (op 136) 
v_mul_f32 v30, v14, v26 // (op 136) 
v_mul_f32 v31, v14, v27 // (op 136) 

v_add_f32 v24, v28, v32 // (op 133) 
v_add_f32 v25, v29, v33 // (op 133) 
v_add_f32 v26, v30, v34 // (op 133) 
v_add_f32 v27, v31, v35 // (op 133) 





buffer_store_dwordx4 v[24:27], v22, s[60:63], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_123:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357 // (op 357) Condition: Done, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357:
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357:


s_mul_i32 s52, s8, s16 // (op 270) 
s_cmp_gt_u32 s9, s52 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_sub_u32 s54, s9, s52 // (op 270) 

s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s52, 0, s54 // (op 270) 


s_cmp_lt_u32 s8, s52 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s54, s52, s8 // (op 270) 


s_cmp_lt_u32 s0, s54 // (op 270) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_270 // (op 270) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_270 if true)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_270:















s_mov_b32 s0, 0 // (op 411) call()
s_mov_b32 s1, 0 // (op 412) call()
s_mov_b32 s0, 0 // (op 426) call()
s_mul_i32 s15, s50, s16 // (op 425) 
s_cmp_gt_u32 s51, s15 // (op 425) 
s_mov_b32 s6, scc // (op 425) 
s_sub_u32 s7, s51, s15 // (op 425) 

s_cmp_eq_i32 s6, 0 // (op 425) 
s_cselect_b32 s8, 0, s7 // (op 425) 


s_cmp_lt_u32 s50, s8 // (op 425) 
s_mov_b32 s15, scc // (op 425) 
s_cmp_eq_i32 s15, 0 // (op 425) 
s_cselect_b32 s6, s8, s50 // (op 425) 


s_cmp_lt_u32 s0, s6 // (op 425) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 // (op 425) Condition: Top (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 if false)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_425:
v_lshrrev_b32 v0, 7, v1 // (op 4444) 
s_mul_i32 s15, s16, s50 // (op 4444) 
s_add_u32 s15, s15, s0 // (op 4444) 
v_and_b32 v3, 63, v1 // (op 4444) 
s_add_u32 s6, s15, s9 // (op 4444) 

v_lshrrev_b32 v6, 4, v3 // (op 4444) 
s_mul_hi_u32 s15, s6, s13 // (op 4444) 
v_lshlrev_b32 v7, 2, v6 // (op 4444) 

s_sub_u32 s7, s6, s15 // (op 4444) 
v_lshl_add_u32 v6, v0, 4, v7 // (op 4444) 


s_lshr_b32 s8, s7, 1 // (op 4444) 

v_lshrrev_b32 v0, 2, v3 // (op 4444) 

s_add_u32 s7, s8, s15 // (op 4444) 


v_and_b32 v3, 3, v0 // (op 4444) 

s_lshr_b32 s8, s7, s11 // (op 4444) 

v_and_b32 v0, 3, v1 // (op 4444) 
s_and_b32 s7, s11, 2147483648 // (op 4444) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 4444) 


s_cmp_eq_i32 s7, 0 // (op 4444) 
s_mov_b32 s15, scc // (op 4444) 

v_lshrrev_b32 v0, 6, v1 // (op 4444) 
s_cmp_eq_i32 s15, 0 // (op 4444) 
s_cselect_b32 s7, s6, s8 // (op 4444) 



v_and_b32 v3, 1, v0 // (op 4444) 

s_mul_hi_u32 s6, s7, s12 // (op 4444) 
v_lshl_add_u32 v0, v3, 4, v7 // (op 4444) 


s_sub_u32 s8, s7, s6 // (op 4444) 
s_lshr_b32 s15, s8, 1 // (op 4444) 

s_add_u32 s8, s15, s6 // (op 4444) 


s_lshr_b32 s6, s8, s10 // (op 4444) 

s_and_b32 s8, s10, 2147483648 // (op 4444) 
s_cmp_eq_i32 s8, 0 // (op 4444) 
s_mov_b32 s15, scc // (op 4444) 

s_cmp_eq_i32 s15, 0 // (op 4444) 
s_cselect_b32 s8, s7, s6 // (op 4444) 


s_mul_i32 s6, s8, s14 // (op 4444) 
v_lshl_add_u32 v3, s8, 7, v6 // (op 4444) 


s_sub_u32 s8, s7, s6 // (op 4444) 


v_lshl_add_u32 v6, s8, 5, v0 // (op 4444) 


v_mul_lo_u32 v0, v6, s4 // (op 4444) 

v_add_lshl_u32 v6, v3, v0, 2 // (op 4444) 


s_mul_i32 s19, s4, s23 // (op 4446) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 4446) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 4446) least significant
s_mul_hi_u32 s6, s4, s22 // (op 4446) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4446) most significant: sum
s_add_u32 s19, s19, s6 // (op 4446) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 4446) 

s_mov_b32 s56, s20 // (op 4446) reg expression
s_mov_b32 s57, s21 // (op 4446) reg expression
s_mov_b32 s58, s28 // (op 4446) reg expression
s_mov_b32 s59, 131072 // (op 4446) call()

v_lshrrev_b32 v0, 7, v1 // (op 4702) 
s_mul_i32 s15, s16, s50 // (op 4702) 
s_add_u32 s15, s15, s0 // (op 4702) 
v_and_b32 v3, 63, v1 // (op 4702) 
s_add_u32 s6, s15, s9 // (op 4702) 

v_lshrrev_b32 v7, 4, v3 // (op 4702) 
s_mul_hi_u32 s15, s6, s13 // (op 4702) 
v_lshlrev_b32 v12, 2, v7 // (op 4702) 

s_sub_u32 s7, s6, s15 // (op 4702) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 4702) 


s_lshr_b32 s8, s7, 1 // (op 4702) 

v_lshrrev_b32 v0, 2, v3 // (op 4702) 

s_add_u32 s7, s8, s15 // (op 4702) 


v_and_b32 v3, 3, v0 // (op 4702) 

s_lshr_b32 s8, s7, s11 // (op 4702) 

v_and_b32 v0, 3, v1 // (op 4702) 
s_and_b32 s7, s11, 2147483648 // (op 4702) 
v_lshl_add_u32 v12, v3, 2, v0 // (op 4702) 


s_cmp_eq_i32 s7, 0 // (op 4702) 
s_mov_b32 s15, scc // (op 4702) 

v_lshrrev_b32 v0, 6, v1 // (op 4702) 
s_cmp_eq_i32 s15, 0 // (op 4702) 
s_cselect_b32 s7, s6, s8 // (op 4702) 



v_and_b32 v3, 1, v0 // (op 4702) 

s_mul_hi_u32 s6, s7, s12 // (op 4702) 
v_lshl_add_u32 v0, v3, 4, v12 // (op 4702) 


s_sub_u32 s8, s7, s6 // (op 4702) 
s_lshr_b32 s15, s8, 1 // (op 4702) 

s_add_u32 s8, s15, s6 // (op 4702) 


s_lshr_b32 s6, s8, s10 // (op 4702) 

s_and_b32 s8, s10, 2147483648 // (op 4702) 
s_cmp_eq_i32 s8, 0 // (op 4702) 
s_mov_b32 s15, scc // (op 4702) 

s_cmp_eq_i32 s15, 0 // (op 4702) 
s_cselect_b32 s8, s7, s6 // (op 4702) 


s_mul_i32 s6, s8, s14 // (op 4702) 
v_lshl_add_u32 v3, s8, 7, v7 // (op 4702) 


s_sub_u32 s8, s7, s6 // (op 4702) 


v_lshl_add_u32 v7, s8, 5, v0 // (op 4702) 


v_mul_lo_u32 v0, v7, s24 // (op 4702) 

v_add_lshl_u32 v7, v3, v0, 2 // (op 4702) 


s_mul_i32 s19, s24, s23 // (op 4704) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 4704) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 4704) least significant
s_mul_hi_u32 s6, s24, s22 // (op 4704) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4704) most significant: sum
s_add_u32 s19, s19, s6 // (op 4704) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 4704) 

s_mov_b32 s60, s26 // (op 4704) reg expression
s_mov_b32 s61, s27 // (op 4704) reg expression
s_mov_b32 s62, s28 // (op 4704) reg expression
s_mov_b32 s63, 131072 // (op 4704) call()

v_lshrrev_b32 v0, 7, v1 // (op 4781) 
v_and_b32 v3, 15, v1 // (op 4781) 
v_lshl_add_u32 v12, v0, 4, v3 // (op 4781) 


v_and_b32 v0, 48, v1 // (op 4781) 
v_lshl_add_u32 v3, v12, 7, v0 // (op 4781) 


v_lshrrev_b32 v0, 6, v1 // (op 4846) 
v_and_b32 v12, 1, v0 // (op 4846) 

v_and_b32 v0, 15, v1 // (op 4846) 
v_lshl_add_u32 v13, v12, 4, v0 // (op 4846) 


v_and_b32 v0, 48, v1 // (op 4846) 
v_lshl_add_u32 v12, v13, 7, v0 // (op 4846) 


v_lshrrev_b32 v0, 7, v1 // (op 4922) 
v_and_b32 v13, 15, v1 // (op 4922) 
v_lshl_add_u32 v16, v0, 4, v13 // (op 4922) 


v_and_b32 v0, 63, v1 // (op 4922) 
v_lshrrev_b32 v13, 4, v0 // (op 4922) 

v_lshl_add_u32 v0, v16, 2, v13 // (op 4922) 


v_accvgpr_write a0, 0.00000 // (op 170) call()
v_accvgpr_write a1, 0.00000 // (op 170) call()
v_accvgpr_write a2, 0.00000 // (op 170) call()
v_accvgpr_write a3, 0.00000 // (op 170) call()
v_accvgpr_write a4, 0.00000 // (op 581) call()
v_accvgpr_write a5, 0.00000 // (op 581) call()
v_accvgpr_write a6, 0.00000 // (op 581) call()
v_accvgpr_write a7, 0.00000 // (op 581) call()
v_accvgpr_write a8, 0.00000 // (op 743) call()
v_accvgpr_write a9, 0.00000 // (op 743) call()
v_accvgpr_write a10, 0.00000 // (op 743) call()
v_accvgpr_write a11, 0.00000 // (op 743) call()
v_accvgpr_write a12, 0.00000 // (op 905) call()
v_accvgpr_write a13, 0.00000 // (op 905) call()
v_accvgpr_write a14, 0.00000 // (op 905) call()
v_accvgpr_write a15, 0.00000 // (op 905) call()
v_lshrrev_b32 v13, 3, v1 // (op 3946) 
s_mul_i32 s15, s16, s50 // (op 3946) 
s_add_u32 s15, s15, s0 // (op 3946) 
v_and_b32 v16, -8, v1 // (op 3946) 
s_add_u32 s6, s15, s9 // (op 3946) 

v_sub_u32 v17, v1, v16 // (op 3946) 

s_mul_hi_u32 s15, s6, s13 // (op 3946) 
v_lshlrev_b32 v16, 4, v17 // (op 3946) 

s_sub_u32 s7, s6, s15 // (op 3946) 
s_lshr_b32 s8, s7, 1 // (op 3946) 

s_add_u32 s7, s8, s15 // (op 3946) 


s_lshr_b32 s8, s7, s11 // (op 3946) 

s_and_b32 s7, s11, 2147483648 // (op 3946) 
s_cmp_eq_i32 s7, 0 // (op 3946) 
s_mov_b32 s15, scc // (op 3946) 

s_cmp_eq_i32 s15, 0 // (op 3946) 
s_cselect_b32 s7, s6, s8 // (op 3946) 


s_mul_hi_u32 s8, s7, s12 // (op 3946) 
s_sub_u32 s15, s7, s8 // (op 3946) 
s_lshr_b32 s18, s15, 1 // (op 3946) 

s_add_u32 s15, s18, s8 // (op 3946) 


s_lshr_b32 s8, s15, s10 // (op 3946) 

s_and_b32 s15, s10, 2147483648 // (op 3946) 
s_cmp_eq_i32 s15, 0 // (op 3946) 
s_mov_b32 s18, scc // (op 3946) 

s_cmp_eq_i32 s18, 0 // (op 3946) 
s_cselect_b32 s15, s7, s8 // (op 3946) 


v_lshl_add_u32 v17, s15, 7, v13 // (op 3946) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3946) 
s_mul_i32 s8, s7, s18 // (op 3946) 


s_sub_u32 s7, s6, s8 // (op 3946) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3946) 


v_mov_b32 v19, 0 // (op 3946) convert
v_mov_b32 v18, v13 // (op 3946) convert

v_mul_lo_u32 v21, v17, s31 // (op 3946) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 3946) most significant: high of low * low
v_mul_lo_u32 v20, v17, s30 // (op 3946) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 3946) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v18 // (op 3946) least significant half
v_addc_co_u32 v21, s[18:19], v21, v19, s[18:19] // (op 3946) most significant half



s_mul_i32 s19, s30, s35 // (op 3948) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3948) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3948) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3948) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3948) most significant: sum
s_add_u32 s19, s19, s6 // (op 3948) most significant: sum


s_mov_b32 s68, s32 // (op 3948) reg expression
s_mov_b32 s69, s33 // (op 3948) reg expression
s_mov_b32 s70, s18 // (op 3948) reg expression
s_mov_b32 s71, 131072 // (op 3948) call()

v_lshrrev_b32 v13, 3, v1 // (op 3965) 
s_mul_i32 s15, s16, s50 // (op 3965) 
s_add_u32 s15, s15, s0 // (op 3965) 
v_and_b32 v16, -8, v1 // (op 3965) 
s_add_u32 s6, s15, s9 // (op 3965) 

v_sub_u32 v17, v1, v16 // (op 3965) 

s_mul_hi_u32 s15, s6, s13 // (op 3965) 
v_lshlrev_b32 v16, 4, v17 // (op 3965) 

s_sub_u32 s7, s6, s15 // (op 3965) 
s_lshr_b32 s8, s7, 1 // (op 3965) 

s_add_u32 s7, s8, s15 // (op 3965) 


s_lshr_b32 s8, s7, s11 // (op 3965) 

s_and_b32 s7, s11, 2147483648 // (op 3965) 
s_cmp_eq_i32 s7, 0 // (op 3965) 
s_mov_b32 s15, scc // (op 3965) 

s_cmp_eq_i32 s15, 0 // (op 3965) 
s_cselect_b32 s7, s6, s8 // (op 3965) 


s_mul_hi_u32 s8, s7, s12 // (op 3965) 
s_sub_u32 s15, s7, s8 // (op 3965) 
s_lshr_b32 s18, s15, 1 // (op 3965) 

s_add_u32 s15, s18, s8 // (op 3965) 


s_lshr_b32 s8, s15, s10 // (op 3965) 

s_and_b32 s15, s10, 2147483648 // (op 3965) 
s_cmp_eq_i32 s15, 0 // (op 3965) 
s_mov_b32 s18, scc // (op 3965) 

s_cmp_eq_i32 s18, 0 // (op 3965) 
s_cselect_b32 s15, s7, s8 // (op 3965) 


v_lshl_add_u32 v17, s15, 7, v13 // (op 3965) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3965) 
s_mul_i32 s8, s7, s18 // (op 3965) 


s_sub_u32 s7, s6, s8 // (op 3965) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3965) 


v_mov_b32 v19, 0 // (op 3965) convert
v_mov_b32 v18, v13 // (op 3965) convert

v_mul_lo_u32 v23, v17, s31 // (op 3965) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 3965) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 3965) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 3965) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v18 // (op 3965) least significant half
v_addc_co_u32 v23, s[18:19], v23, v19, s[18:19] // (op 3965) most significant half



s_mul_i32 s19, s30, s35 // (op 3967) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3967) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3967) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3967) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3967) most significant: sum
s_add_u32 s19, s19, s6 // (op 3967) most significant: sum


s_mov_b32 s68, s32 // (op 3967) reg expression
s_mov_b32 s69, s33 // (op 3967) reg expression
s_mov_b32 s70, s18 // (op 3967) reg expression
s_mov_b32 s71, 131072 // (op 3967) call()

v_lshrrev_b32 v13, 3, v1 // (op 4080) 
s_mul_i32 s15, s16, s50 // (op 4080) 
s_add_u32 s15, s15, s0 // (op 4080) 
v_and_b32 v16, -8, v1 // (op 4080) 
s_add_u32 s6, s15, s9 // (op 4080) 

v_sub_u32 v17, v1, v16 // (op 4080) 

s_mul_hi_u32 s15, s6, s13 // (op 4080) 
v_lshlrev_b32 v16, 4, v17 // (op 4080) 

s_sub_u32 s7, s6, s15 // (op 4080) 
s_lshr_b32 s8, s7, 1 // (op 4080) 

s_add_u32 s7, s8, s15 // (op 4080) 


s_lshr_b32 s8, s7, s11 // (op 4080) 

s_and_b32 s7, s11, 2147483648 // (op 4080) 
s_cmp_eq_i32 s7, 0 // (op 4080) 
s_mov_b32 s15, scc // (op 4080) 

s_cmp_eq_i32 s15, 0 // (op 4080) 
s_cselect_b32 s7, s6, s8 // (op 4080) 


s_mul_hi_u32 s8, s7, s12 // (op 4080) 
s_sub_u32 s15, s7, s8 // (op 4080) 
s_lshr_b32 s18, s15, 1 // (op 4080) 

s_add_u32 s15, s18, s8 // (op 4080) 


s_lshr_b32 s8, s15, s10 // (op 4080) 

s_and_b32 s15, s10, 2147483648 // (op 4080) 
s_cmp_eq_i32 s15, 0 // (op 4080) 
s_mov_b32 s18, scc // (op 4080) 

s_cmp_eq_i32 s18, 0 // (op 4080) 
s_cselect_b32 s15, s7, s8 // (op 4080) 


s_mul_i32 s8, s15, s14 // (op 4080) 

s_sub_u32 s15, s7, s8 // (op 4080) 

v_lshl_add_u32 v17, s15, 5, v13 // (op 4080) 


v_mul_lo_u32 v19, v17, s37 // (op 4080) most significant: low of low * high
v_mul_hi_u32 v13, v17, s36 // (op 4080) most significant: high of low * low
v_mul_lo_u32 v18, v17, s36 // (op 4080) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 4080) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4080) 
s_mul_i32 s8, s7, s18 // (op 4080) 


s_sub_u32 s7, s6, s8 // (op 4080) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 4080) 


v_mov_b32 v17, 0 // (op 4080) convert
v_mov_b32 v16, v13 // (op 4080) convert

v_add_co_u32 v24, s[18:19], v16, v18 // (op 4080) least significant half
v_addc_co_u32 v25, s[18:19], v17, v19, s[18:19] // (op 4080) most significant half



s_mul_i32 s19, s36, s23 // (op 4082) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 4082) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 4082) least significant
s_mul_hi_u32 s6, s36, s22 // (op 4082) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4082) most significant: sum
s_add_u32 s19, s19, s6 // (op 4082) most significant: sum


s_mov_b32 s72, s38 // (op 4082) reg expression
s_mov_b32 s73, s39 // (op 4082) reg expression
s_mov_b32 s74, s18 // (op 4082) reg expression
s_mov_b32 s75, 131072 // (op 4082) call()

v_lshrrev_b32 v13, 3, v1 // (op 4099) 
s_mul_i32 s15, s16, s50 // (op 4099) 
s_add_u32 s15, s15, s0 // (op 4099) 
v_and_b32 v16, -8, v1 // (op 4099) 
s_add_u32 s6, s15, s9 // (op 4099) 

v_sub_u32 v17, v1, v16 // (op 4099) 

s_mul_hi_u32 s15, s6, s13 // (op 4099) 
v_lshlrev_b32 v16, 4, v17 // (op 4099) 

s_sub_u32 s7, s6, s15 // (op 4099) 
s_lshr_b32 s8, s7, 1 // (op 4099) 

s_add_u32 s7, s8, s15 // (op 4099) 


s_lshr_b32 s8, s7, s11 // (op 4099) 

s_and_b32 s7, s11, 2147483648 // (op 4099) 
s_cmp_eq_i32 s7, 0 // (op 4099) 
s_mov_b32 s15, scc // (op 4099) 

s_cmp_eq_i32 s15, 0 // (op 4099) 
s_cselect_b32 s7, s6, s8 // (op 4099) 


s_mul_hi_u32 s8, s7, s12 // (op 4099) 
s_sub_u32 s15, s7, s8 // (op 4099) 
s_lshr_b32 s18, s15, 1 // (op 4099) 

s_add_u32 s15, s18, s8 // (op 4099) 


s_lshr_b32 s8, s15, s10 // (op 4099) 

s_and_b32 s15, s10, 2147483648 // (op 4099) 
s_cmp_eq_i32 s15, 0 // (op 4099) 
s_mov_b32 s18, scc // (op 4099) 

s_cmp_eq_i32 s18, 0 // (op 4099) 
s_cselect_b32 s15, s7, s8 // (op 4099) 


s_mul_i32 s8, s15, s14 // (op 4099) 

s_sub_u32 s15, s7, s8 // (op 4099) 

v_lshl_add_u32 v17, s15, 5, v13 // (op 4099) 


v_mul_lo_u32 v19, v17, s37 // (op 4099) most significant: low of low * high
v_mul_hi_u32 v13, v17, s36 // (op 4099) most significant: high of low * low
v_mul_lo_u32 v18, v17, s36 // (op 4099) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 4099) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4099) 
s_mul_i32 s8, s7, s18 // (op 4099) 


s_sub_u32 s7, s6, s8 // (op 4099) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 4099) 


v_mov_b32 v17, 0 // (op 4099) convert
v_mov_b32 v16, v13 // (op 4099) convert

v_add_co_u32 v26, s[18:19], v16, v18 // (op 4099) least significant half
v_addc_co_u32 v27, s[18:19], v17, v19, s[18:19] // (op 4099) most significant half



s_mul_i32 s19, s36, s23 // (op 4101) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 4101) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 4101) least significant
s_mul_hi_u32 s6, s36, s22 // (op 4101) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4101) most significant: sum
s_add_u32 s19, s19, s6 // (op 4101) most significant: sum


s_mov_b32 s72, s38 // (op 4101) reg expression
s_mov_b32 s73, s39 // (op 4101) reg expression
s_mov_b32 s74, s18 // (op 4101) reg expression
s_mov_b32 s75, 131072 // (op 4101) call()

v_lshrrev_b32 v13, 2, v1 // (op 4208) 
s_mul_i32 s15, s16, s50 // (op 4208) 
s_add_u32 s15, s15, s0 // (op 4208) 
v_lshlrev_b32 v16, 1, v13 // (op 4208) 

s_add_u32 s6, s15, s9 // (op 4208) 

v_and_b32 v13, -4, v1 // (op 4208) 
s_mul_hi_u32 s15, s6, s13 // (op 4208) 
v_sub_u32 v17, v1, v13 // (op 4208) 

s_sub_u32 s7, s6, s15 // (op 4208) 
s_lshr_b32 s8, s7, 1 // (op 4208) 

s_add_u32 s7, s8, s15 // (op 4208) 


s_lshr_b32 s8, s7, s11 // (op 4208) 

s_and_b32 s7, s11, 2147483648 // (op 4208) 
s_cmp_eq_i32 s7, 0 // (op 4208) 
s_mov_b32 s15, scc // (op 4208) 

s_cmp_eq_i32 s15, 0 // (op 4208) 
s_cselect_b32 s7, s6, s8 // (op 4208) 


s_mul_hi_u32 s8, s7, s12 // (op 4208) 
s_sub_u32 s15, s7, s8 // (op 4208) 
s_lshr_b32 s18, s15, 1 // (op 4208) 

s_add_u32 s15, s18, s8 // (op 4208) 


s_lshr_b32 s8, s15, s10 // (op 4208) 

s_and_b32 s15, s10, 2147483648 // (op 4208) 
s_cmp_eq_i32 s15, 0 // (op 4208) 
s_mov_b32 s18, scc // (op 4208) 

s_cmp_eq_i32 s18, 0 // (op 4208) 
s_cselect_b32 s15, s7, s8 // (op 4208) 


v_lshl_add_u32 v13, s15, 7, v16 // (op 4208) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4208) 
s_mul_i32 s8, s7, s18 // (op 4208) 


s_sub_u32 s7, s6, s8 // (op 4208) 


v_lshl_add_u32 v16, s7, 2, v17 // (op 4208) 


v_mul_lo_u32 v17, v13, s40 // (op 4208) 
v_add_u32 v17, v17, v16 // (op 4208) 


s_mul_i32 s19, s40, s35 // (op 4210) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4210) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4210) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4210) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4210) most significant: sum
s_add_u32 s19, s19, s6 // (op 4210) most significant: sum


s_mov_b32 s76, s42 // (op 4210) reg expression
s_mov_b32 s77, s43 // (op 4210) reg expression
s_mov_b32 s78, s18 // (op 4210) reg expression
s_mov_b32 s79, 131072 // (op 4210) call()

v_lshrrev_b32 v13, 2, v1 // (op 4227) 
s_mul_i32 s15, s16, s50 // (op 4227) 
s_add_u32 s15, s15, s0 // (op 4227) 
v_lshlrev_b32 v16, 1, v13 // (op 4227) 

s_add_u32 s6, s15, s9 // (op 4227) 

v_and_b32 v13, -4, v1 // (op 4227) 
s_mul_hi_u32 s15, s6, s13 // (op 4227) 
v_sub_u32 v18, v1, v13 // (op 4227) 

s_sub_u32 s7, s6, s15 // (op 4227) 
s_lshr_b32 s8, s7, 1 // (op 4227) 

s_add_u32 s7, s8, s15 // (op 4227) 


s_lshr_b32 s8, s7, s11 // (op 4227) 

s_and_b32 s7, s11, 2147483648 // (op 4227) 
s_cmp_eq_i32 s7, 0 // (op 4227) 
s_mov_b32 s15, scc // (op 4227) 

s_cmp_eq_i32 s15, 0 // (op 4227) 
s_cselect_b32 s7, s6, s8 // (op 4227) 


s_mul_hi_u32 s8, s7, s12 // (op 4227) 
s_sub_u32 s15, s7, s8 // (op 4227) 
s_lshr_b32 s18, s15, 1 // (op 4227) 

s_add_u32 s15, s18, s8 // (op 4227) 


s_lshr_b32 s8, s15, s10 // (op 4227) 

s_and_b32 s15, s10, 2147483648 // (op 4227) 
s_cmp_eq_i32 s15, 0 // (op 4227) 
s_mov_b32 s18, scc // (op 4227) 

s_cmp_eq_i32 s18, 0 // (op 4227) 
s_cselect_b32 s15, s7, s8 // (op 4227) 


v_lshl_add_u32 v13, s15, 7, v16 // (op 4227) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4227) 
s_mul_i32 s8, s7, s18 // (op 4227) 


s_sub_u32 s7, s6, s8 // (op 4227) 


v_lshl_add_u32 v16, s7, 2, v18 // (op 4227) 


v_mul_lo_u32 v18, v13, s40 // (op 4227) 
v_add_u32 v18, v18, v16 // (op 4227) 


s_mul_i32 s19, s40, s35 // (op 4229) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4229) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4229) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4229) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4229) most significant: sum
s_add_u32 s19, s19, s6 // (op 4229) most significant: sum


s_mov_b32 s76, s42 // (op 4229) reg expression
s_mov_b32 s77, s43 // (op 4229) reg expression
s_mov_b32 s78, s18 // (op 4229) reg expression
s_mov_b32 s79, 131072 // (op 4229) call()

v_lshrrev_b32 v13, 6, v1 // (op 4336) 
s_mul_i32 s15, s16, s50 // (op 4336) 
s_add_u32 s15, s15, s0 // (op 4336) 
v_and_b32 v16, 1, v13 // (op 4336) 

s_add_u32 s6, s15, s9 // (op 4336) 

v_and_b32 v13, 15, v1 // (op 4336) 
s_mul_hi_u32 s15, s6, s13 // (op 4336) 
v_lshl_add_u32 v19, v16, 4, v13 // (op 4336) 


s_sub_u32 s7, s6, s15 // (op 4336) 
v_and_b32 v13, 63, v1 // (op 4336) 
s_lshr_b32 s8, s7, 1 // (op 4336) 

v_lshrrev_b32 v16, 4, v13 // (op 4336) 

s_add_u32 s7, s8, s15 // (op 4336) 


s_lshr_b32 s8, s7, s11 // (op 4336) 

s_and_b32 s7, s11, 2147483648 // (op 4336) 
s_cmp_eq_i32 s7, 0 // (op 4336) 
s_mov_b32 s15, scc // (op 4336) 

s_cmp_eq_i32 s15, 0 // (op 4336) 
s_cselect_b32 s7, s6, s8 // (op 4336) 


s_mul_hi_u32 s8, s7, s12 // (op 4336) 
s_sub_u32 s15, s7, s8 // (op 4336) 
s_lshr_b32 s18, s15, 1 // (op 4336) 

s_add_u32 s15, s18, s8 // (op 4336) 


s_lshr_b32 s8, s15, s10 // (op 4336) 

s_and_b32 s15, s10, 2147483648 // (op 4336) 
s_cmp_eq_i32 s15, 0 // (op 4336) 
s_mov_b32 s18, scc // (op 4336) 

s_cmp_eq_i32 s18, 0 // (op 4336) 
s_cselect_b32 s15, s7, s8 // (op 4336) 


s_mul_i32 s8, s15, s14 // (op 4336) 

s_sub_u32 s15, s7, s8 // (op 4336) 

v_lshl_add_u32 v13, s15, 5, v19 // (op 4336) 


v_mul_lo_u32 v19, v13, s44 // (op 4336) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 4336) 
s_mul_i32 s8, s7, s18 // (op 4336) 


s_sub_u32 s7, s6, s8 // (op 4336) 


v_lshl_add_u32 v13, s7, 2, v16 // (op 4336) 


v_add_u32 v16, v13, v19 // (op 4336) 


s_mul_i32 s19, s44, s23 // (op 4338) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4338) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4338) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4338) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4338) most significant: sum
s_add_u32 s19, s19, s6 // (op 4338) most significant: sum


s_mov_b32 s80, s46 // (op 4338) reg expression
s_mov_b32 s81, s47 // (op 4338) reg expression
s_mov_b32 s82, s18 // (op 4338) reg expression
s_mov_b32 s83, 131072 // (op 4338) call()

v_lshrrev_b32 v13, 6, v1 // (op 4355) 
s_mul_i32 s15, s16, s50 // (op 4355) 
s_add_u32 s15, s15, s0 // (op 4355) 
v_and_b32 v19, 1, v13 // (op 4355) 

s_add_u32 s6, s15, s9 // (op 4355) 

v_and_b32 v13, 15, v1 // (op 4355) 
s_mul_hi_u32 s15, s6, s13 // (op 4355) 
v_lshl_add_u32 v28, v19, 4, v13 // (op 4355) 


s_sub_u32 s7, s6, s15 // (op 4355) 
v_and_b32 v13, 63, v1 // (op 4355) 
s_lshr_b32 s8, s7, 1 // (op 4355) 

v_lshrrev_b32 v19, 4, v13 // (op 4355) 

s_add_u32 s7, s8, s15 // (op 4355) 


s_lshr_b32 s8, s7, s11 // (op 4355) 

s_and_b32 s7, s11, 2147483648 // (op 4355) 
s_cmp_eq_i32 s7, 0 // (op 4355) 
s_mov_b32 s15, scc // (op 4355) 

s_cmp_eq_i32 s15, 0 // (op 4355) 
s_cselect_b32 s7, s6, s8 // (op 4355) 


s_mul_hi_u32 s8, s7, s12 // (op 4355) 
s_sub_u32 s15, s7, s8 // (op 4355) 
s_lshr_b32 s18, s15, 1 // (op 4355) 

s_add_u32 s15, s18, s8 // (op 4355) 


s_lshr_b32 s8, s15, s10 // (op 4355) 

s_and_b32 s15, s10, 2147483648 // (op 4355) 
s_cmp_eq_i32 s15, 0 // (op 4355) 
s_mov_b32 s18, scc // (op 4355) 

s_cmp_eq_i32 s18, 0 // (op 4355) 
s_cselect_b32 s15, s7, s8 // (op 4355) 


s_mul_i32 s8, s15, s14 // (op 4355) 

s_sub_u32 s15, s7, s8 // (op 4355) 

v_lshl_add_u32 v13, s15, 5, v28 // (op 4355) 


v_mul_lo_u32 v28, v13, s44 // (op 4355) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 4355) 
s_mul_i32 s8, s7, s18 // (op 4355) 


s_sub_u32 s7, s6, s8 // (op 4355) 


v_lshl_add_u32 v13, s7, 2, v19 // (op 4355) 


v_add_u32 v19, v13, v28 // (op 4355) 


s_mul_i32 s19, s44, s23 // (op 4357) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4357) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4357) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4357) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4357) most significant: sum
s_add_u32 s19, s19, s6 // (op 4357) most significant: sum


s_mov_b32 s80, s46 // (op 4357) reg expression
s_mov_b32 s81, s47 // (op 4357) reg expression
s_mov_b32 s82, s18 // (op 4357) reg expression
s_mov_b32 s83, 131072 // (op 4357) call()




buffer_load_ubyte v13, v16, s[80:83], 0 offen offset: 0 // (op 3480) Load value
v_and_b32 v28, -8, v1 // (op 4822) 
v_sub_u32 v29, v1, v28 // (op 4822) 

v_lshlrev_b32 v28, 4, v29 // (op 4822) 

v_lshrrev_b32 v29, 3, v1 // (op 4822) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 4822) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 4822) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5359) 



s_nop 0
s_mov_b32 m0, s15 // (op 3854) reg expression
buffer_load_dwordx4 v24, s[72:75], 0 offen offset: 0 lds // (op 3854) Load value direct to lds

v_and_b32 v28, -8, v1 // (op 4747) 
v_sub_u32 v29, v1, v28 // (op 4747) 

v_lshlrev_b32 v28, 4, v29 // (op 4747) 

v_lshrrev_b32 v29, 3, v1 // (op 4747) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 4747) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 4747) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5340) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 3859) 
v_mov_b64 v[28:29], s[18:19] // (op 3859) convert


v_mov_b64 v[30:31], v[20:21] // (op 3859) 
s_add_u32 s6, 4096, s15 // (op 3859) 
s_mov_b32 m0, s6 // (op 3859) reg expression
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3859) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 3859) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3859) most significant half

s_add_u32 m0, m0, 4096 // (op 3859) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3859) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 3859) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3859) most significant half

s_add_u32 m0, m0, 4096 // (op 3859) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3859) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 3859) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3859) most significant half

s_add_u32 m0, m0, 4096 // (op 3859) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3859) Load value direct to lds





v_mov_b64 v[28:29], s[40:41] // (op 3490) convert

v_mov_b32 v30, v17 // (op 3490) 
buffer_load_ubyte v32, v30, s[76:79], 0 offen offset: 0 // (op 3490) Load value
v_add_u32 v30, v30, v28 // (op 3490) 
buffer_load_ubyte v33, v30, s[76:79], 0 offen offset: 0 // (op 3490) Load value





buffer_load_ubyte v28, v16, s[80:83], 0 offen offset:4 // (op 3494) Load value

v_and_b32 v16, -8, v1 // (op 5036) 
v_sub_u32 v29, v1, v16 // (op 5036) 

v_lshlrev_b32 v16, 4, v29 // (op 5036) 

v_lshrrev_b32 v29, 3, v1 // (op 5036) 
v_lshl_add_u32 v30, v29, 7, v16 // (op 5036) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 5036) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5407) 


s_nop 0
s_add_u32 s6, 20480, s15 // (op 3864) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 3864) reg expression
s_mov_b32 s7, 128 // (op 3864) call()
buffer_load_dwordx4 v24, s[72:75], s7 offen offset: 0 lds // (op 3864) Load value direct to lds




v_and_b32 v16, -8, v1 // (op 4992) 
v_sub_u32 v24, v1, v16 // (op 4992) 

v_lshlrev_b32 v16, 4, v24 // (op 4992) 

v_lshrrev_b32 v24, 3, v1 // (op 4992) 
v_lshl_add_u32 v25, v24, 7, v16 // (op 4992) 


s_nop 0
v_readfirstlane_b32 s15, v25 // (op 4992) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5398) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 3866) 
v_mov_b64 v[24:25], s[18:19] // (op 3866) convert


v_mov_b64 v[30:31], v[20:21] // (op 3866) 
s_add_u32 s6, 24576, s15 // (op 3866) 
s_mov_b32 m0, s6 // (op 3866) reg expression
s_mov_b32 s7, 128 // (op 3866) call()
buffer_load_dwordx4 v30, s[68:71], s7 offen offset: 0 lds // (op 3866) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 3866) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3866) most significant half

s_add_u32 m0, m0, 4096 // (op 3866) 
buffer_load_dwordx4 v30, s[68:71], s7 offen offset: 0 lds // (op 3866) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 3866) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3866) most significant half

s_add_u32 m0, m0, 4096 // (op 3866) 
buffer_load_dwordx4 v30, s[68:71], s7 offen offset: 0 lds // (op 3866) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 3866) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3866) most significant half

s_add_u32 m0, m0, 4096 // (op 3866) 
buffer_load_dwordx4 v30, s[68:71], s7 offen offset: 0 lds // (op 3866) Load value direct to lds







v_mov_b64 v[20:21], s[40:41] // (op 3504) convert

v_mov_b32 v16, v17 // (op 3504) 
buffer_load_ubyte v24, v16, s[76:79], 0 offen offset:4 // (op 3504) Load value
v_add_u32 v16, v16, v20 // (op 3504) 
buffer_load_ubyte v25, v16, s[76:79], 0 offen offset:4 // (op 3504) Load value



v_and_b32 v16, -4, v1 // (op 4888) 
v_sub_u32 v17, v1, v16 // (op 4888) 

v_lshrrev_b32 v16, 2, v1 // (op 4888) 
v_lshl_add_u32 v20, v16, 3, v17 // (op 4888) 


s_barrier  // (op 5378) 



s_waitcnt vmcnt(9)
ds_write_b8 v20, v32 offset:40960 // (op 3508) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v33 offset:40964 // (op 3508) Store local data 

s_barrier  // (op 5356) 



ds_read_b128 v[34:37], v12 // (op 3541) Load local data 
ds_read_b128 v[38:41], v12 offset:64 // (op 3541) Load local data 
s_barrier  // (op 5337) 



ds_read_b128 v[42:45], v3 offset:4096 // (op 3522) Load local data 
ds_read_b128 v[46:49], v3 offset:4160 // (op 3522) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5375) 



ds_read_u8 v16, v0 offset:40960 // (op 3545) Load local data 



ds_read_b128 v[50:53], v3 offset:8192 // (op 3527) Load local data 
ds_read_b128 v[54:57], v3 offset:8256 // (op 3527) Load local data 



ds_read_u8 v17, v0 offset:41088 // (op 3550) Load local data 



ds_read_b128 v[58:61], v3 offset:12288 // (op 3532) Load local data 
ds_read_b128 v[62:65], v3 offset:12352 // (op 3532) Load local data 



ds_read_u8 v20, v0 offset:41216 // (op 3555) Load local data 



ds_read_b128 v[66:69], v3 offset:16384 // (op 3537) Load local data 
ds_read_b128 v[70:73], v3 offset:16448 // (op 3537) Load local data 



ds_read_u8 v21, v0 offset:41344 // (op 3560) Load local data 
v_and_b32 v29, -8, v1 // (op 4719) 
v_sub_u32 v30, v1, v29 // (op 4719) 

v_lshlrev_b32 v29, 4, v30 // (op 4719) 

v_lshrrev_b32 v30, 3, v1 // (op 4719) 
v_lshl_add_u32 v31, v30, 7, v29 // (op 4719) 


s_nop 0
v_readfirstlane_b32 s15, v31 // (op 4719) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -8, v1 // (op 4794) 
v_sub_u32 v30, v1, v29 // (op 4794) 

v_lshlrev_b32 v29, 4, v30 // (op 4794) 

v_lshrrev_b32 v30, 3, v1 // (op 4794) 
v_lshl_add_u32 v31, v30, 7, v29 // (op 4794) 


s_nop 0
v_readfirstlane_b32 s6, v31 // (op 4794) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -4, v1 // (op 4859) 
v_sub_u32 v30, v1, v29 // (op 4859) 

v_lshrrev_b32 v29, 2, v1 // (op 4859) 
v_lshl_add_u32 v31, v29, 3, v30 // (op 4859) 


v_and_b32 v29, -8, v1 // (op 4977) 
v_sub_u32 v30, v1, v29 // (op 4977) 

v_lshlrev_b32 v29, 4, v30 // (op 4977) 

v_lshrrev_b32 v30, 3, v1 // (op 4977) 
v_lshl_add_u32 v74, v30, 7, v29 // (op 4977) 


s_nop 0
v_readfirstlane_b32 s7, v74 // (op 4977) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -8, v1 // (op 5021) 
v_sub_u32 v30, v1, v29 // (op 5021) 

v_lshlrev_b32 v29, 4, v30 // (op 5021) 

v_lshrrev_b32 v30, 3, v1 // (op 5021) 
v_lshl_add_u32 v74, v30, 7, v29 // (op 5021) 


s_nop 0
v_readfirstlane_b32 s8, v74 // (op 5021) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -4, v1 // (op 5060) 
v_sub_u32 v30, v1, v29 // (op 5060) 

v_lshrrev_b32 v29, 2, v1 // (op 5060) 
v_lshl_add_u32 v74, v29, 3, v30 // (op 5060) 


s_mov_b32 s1, 0 // (op 907) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 906) 
s_mov_b32 s28, s18 // (op 906) convert

s_and_b32 s29, s28, 4294967294 // (op 906) 

s_cmp_lt_u32 s1, s29 // (op 906) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_906 // (op 906) Condition: Top (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_906 if false)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_906:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[34:41], a[0:3], v16, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 184) 
s_add_u32 s1, s1, 2 // (op 908) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[34:41], a[4:7], v17, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 618) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[34:41], a[8:11], v20, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 780) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[34:41], a[12:15], v21, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 942) 


v_mov_b32 v29, v19 // (op 183) 

buffer_load_ubyte v13, v29, s[80:83], 0 offen offset:8 // (op 183) Load value
s_barrier  // (op 5362) 


v_mov_b64 v[76:77], v[26:27] // (op 3803) 

s_mov_b32 m0, s6 // (op 3803) reg expression
s_mov_b32 s18, 256 // (op 3803) call()
buffer_load_dwordx4 v76, s[72:75], s18 offen offset: 0 lds // (op 3803) Load value direct to lds

s_barrier  // (op 5343) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 3798) 
v_mov_b64 v[78:79], s[18:19] // (op 3798) convert


v_mov_b64 v[80:81], v[22:23] // (op 3798) 
v_mov_b64 v[82:83], v[80:81] // (op 3798) 
s_add_u32 s18, 4096, s15 // (op 3798) 
s_mov_b32 m0, s18 // (op 3798) reg expression
s_mov_b32 s19, 256 // (op 3798) call()
buffer_load_dwordx4 v82, s[68:71], s19 offen offset: 0 lds // (op 3798) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v78 // (op 3798) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3798) most significant half

s_add_u32 m0, m0, 4096 // (op 3798) 
buffer_load_dwordx4 v82, s[68:71], s19 offen offset: 0 lds // (op 3798) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v78 // (op 3798) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3798) most significant half

s_add_u32 m0, m0, 4096 // (op 3798) 
buffer_load_dwordx4 v82, s[68:71], s19 offen offset: 0 lds // (op 3798) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v78 // (op 3798) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3798) most significant half

s_add_u32 m0, m0, 4096 // (op 3798) 
buffer_load_dwordx4 v82, s[68:71], s19 offen offset: 0 lds // (op 3798) Load value direct to lds





v_mov_b64 v[78:79], s[40:41] // (op 188) convert

v_mov_b32 v30, v18 // (op 188) 
v_mov_b32 v75, v30 // (op 188) 
buffer_load_ubyte v32, v75, s[76:79], 0 offen offset:8 // (op 188) Load value
v_add_u32 v75, v75, v78 // (op 188) 
buffer_load_ubyte v33, v75, s[76:79], 0 offen offset:8 // (op 188) Load value


s_barrier  // (op 5416) 



s_waitcnt vmcnt(9)
ds_write_b8 v74, v24 offset:41472 // (op 508) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v74, v25 offset:41476 // (op 508) Store local data 
s_barrier  // (op 5404) 



ds_read_b128 v[34:37], v5 offset:20480 // (op 501) Load local data 
ds_read_b128 v[38:41], v5 offset:20544 // (op 501) Load local data 
s_barrier  // (op 5395) 



ds_read_b128 v[42:45], v11 offset:24576 // (op 499) Load local data 
ds_read_b128 v[46:49], v11 offset:24640 // (op 499) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5413) 



ds_read_u8 v16, v10 offset:41472 // (op 503) Load local data 



ds_read_b128 v[50:53], v11 offset:28672 // (op 625) Load local data 
ds_read_b128 v[54:57], v11 offset:28736 // (op 625) Load local data 



ds_read_u8 v17, v10 offset:41600 // (op 629) Load local data 



ds_read_b128 v[58:61], v11 offset:32768 // (op 787) Load local data 
ds_read_b128 v[62:65], v11 offset:32832 // (op 787) Load local data 



ds_read_u8 v20, v10 offset:41728 // (op 791) Load local data 



ds_read_b128 v[66:69], v11 offset:36864 // (op 949) Load local data 
ds_read_b128 v[70:73], v11 offset:36928 // (op 949) Load local data 



ds_read_u8 v21, v10 offset:41856 // (op 953) Load local data 
s_barrier  // (op 5535) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[34:41], a[0:3], v16, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 506) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[34:41], a[4:7], v17, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 619) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[34:41], a[8:11], v20, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 781) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[34:41], a[12:15], v21, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 943) 



buffer_load_ubyte v28, v29, s[80:83], 0 offen offset:12 // (op 505) Load value

s_barrier  // (op 5410) 
v_add_u32 v19, 8, v19 // (op 4347) 


s_add_u32 s18, 20480, s8 // (op 3826) 

s_mov_b32 m0, s18 // (op 3826) reg expression
s_mov_b32 s19, 384 // (op 3826) call()
buffer_load_dwordx4 v76, s[72:75], s19 offen offset: 0 lds // (op 3826) Load value direct to lds



s_barrier  // (op 5401) 
v_add_co_u32 v26, vcc, 256, v26 // (op 4091) least significant half

v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 4091) most significant half
s_lshl_b64 s[18:19], s[30:31], 5 // (op 3821) 
v_mov_b64 v[78:79], s[18:19] // (op 3821) convert


v_mov_b64 v[76:77], v[80:81] // (op 3821) 
s_add_u32 s18, 24576, s7 // (op 3821) 
s_mov_b32 m0, s18 // (op 3821) reg expression
s_mov_b32 s19, 384 // (op 3821) call()
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 3821) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v78 // (op 3821) least significant half
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 3821) most significant half

s_add_u32 m0, m0, 4096 // (op 3821) 
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 3821) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v78 // (op 3821) least significant half
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 3821) most significant half

s_add_u32 m0, m0, 4096 // (op 3821) 
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 3821) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v78 // (op 3821) least significant half
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 3821) most significant half

s_add_u32 m0, m0, 4096 // (op 3821) 
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 3821) Load value direct to lds






v_add_co_u32 v22, vcc, 256, v22 // (op 3957) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 3957) most significant half
v_mov_b64 v[76:77], s[40:41] // (op 510) convert

v_mov_b32 v29, v30 // (op 510) 
buffer_load_ubyte v24, v29, s[76:79], 0 offen offset:12 // (op 510) Load value
v_add_u32 v29, v29, v76 // (op 510) 
buffer_load_ubyte v25, v29, s[76:79], 0 offen offset:12 // (op 510) Load value



v_add_u32 v18, 8, v18 // (op 4219) 
s_barrier  // (op 5381) 



s_waitcnt vmcnt(9)
ds_write_b8 v31, v32 offset:40960 // (op 186) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v31, v33 offset:40964 // (op 186) Store local data 
s_barrier  // (op 5368) 



ds_read_b128 v[34:37], v4 // (op 179) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 179) Load local data 
s_barrier  // (op 5349) 



ds_read_b128 v[42:45], v9 offset:4096 // (op 177) Load local data 
ds_read_b128 v[46:49], v9 offset:4160 // (op 177) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5388) 



ds_read_u8 v16, v8 offset:40960 // (op 181) Load local data 



ds_read_b128 v[50:53], v9 offset:8192 // (op 612) Load local data 
ds_read_b128 v[54:57], v9 offset:8256 // (op 612) Load local data 



ds_read_u8 v17, v8 offset:41088 // (op 616) Load local data 



ds_read_b128 v[58:61], v9 offset:12288 // (op 774) Load local data 
ds_read_b128 v[62:65], v9 offset:12352 // (op 774) Load local data 



ds_read_u8 v20, v8 offset:41216 // (op 778) Load local data 



ds_read_b128 v[66:69], v9 offset:16384 // (op 936) Load local data 
ds_read_b128 v[70:73], v9 offset:16448 // (op 936) Load local data 



ds_read_u8 v21, v8 offset:41344 // (op 940) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 906) 
s_mov_b32 s28, s18 // (op 906) convert

s_and_b32 s29, s28, 4294967294 // (op 906) 

s_cmp_lt_u32 s1, s29 // (op 906) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_906 // (op 906) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_906 if true)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_906:












s_barrier  // (op 5519) 
v_lshrrev_b32 v13, 3, v1 // (op 3893) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3893) 
v_and_b32 v18, -8, v1 // (op 3893) 
s_and_b32 s15, s18, 4294967294 // (op 3893) 
v_sub_u32 v19, v1, v18 // (op 3893) 

s_add_u32 s6, s0, s15 // (op 3893) 

v_lshlrev_b32 v18, 4, v19 // (op 3893) 

s_mul_i32 s15, s16, s50 // (op 3893) 
s_add_u32 s15, s15, s6 // (op 3893) 

s_add_u32 s6, s15, s9 // (op 3893) 

s_mul_hi_u32 s15, s6, s13 // (op 3893) 
s_sub_u32 s7, s6, s15 // (op 3893) 
s_lshr_b32 s8, s7, 1 // (op 3893) 

s_add_u32 s7, s8, s15 // (op 3893) 


s_lshr_b32 s8, s7, s11 // (op 3893) 

s_and_b32 s7, s11, 2147483648 // (op 3893) 
s_cmp_eq_i32 s7, 0 // (op 3893) 
s_mov_b32 s15, scc // (op 3893) 

s_cmp_eq_i32 s15, 0 // (op 3893) 
s_cselect_b32 s7, s6, s8 // (op 3893) 


s_mul_hi_u32 s8, s7, s12 // (op 3893) 
s_sub_u32 s15, s7, s8 // (op 3893) 
s_lshr_b32 s28, s15, 1 // (op 3893) 

s_add_u32 s15, s28, s8 // (op 3893) 


s_lshr_b32 s8, s15, s10 // (op 3893) 

s_and_b32 s15, s10, 2147483648 // (op 3893) 
s_cmp_eq_i32 s15, 0 // (op 3893) 
s_mov_b32 s28, scc // (op 3893) 

s_cmp_eq_i32 s28, 0 // (op 3893) 
s_cselect_b32 s15, s7, s8 // (op 3893) 


s_mul_i32 s8, s7, s18 // (op 3893) 


v_lshl_add_u32 v19, s15, 7, v13 // (op 3893) 


s_sub_u32 s7, s6, s8 // (op 3893) 


v_lshl_add_u32 v13, s7, 7, v18 // (op 3893) 


v_mov_b32 v23, 0 // (op 3893) convert
v_mov_b32 v22, v13 // (op 3893) convert

v_mul_lo_u32 v27, v19, s31 // (op 3893) most significant: low of low * high
v_mul_hi_u32 v13, v19, s30 // (op 3893) most significant: high of low * low
v_mul_lo_u32 v26, v19, s30 // (op 3893) least significant: low of low * low
v_add_u32 v27, v27, v13 // (op 3893) most significant: sum

v_add_co_u32 v26, s[18:19], v26, v22 // (op 3893) least significant half
v_addc_co_u32 v27, s[18:19], v27, v23, s[18:19] // (op 3893) most significant half



s_mul_i32 s19, s30, s35 // (op 3895) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3895) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3895) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3895) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3895) most significant: sum
s_add_u32 s19, s19, s6 // (op 3895) most significant: sum


s_mov_b32 s68, s32 // (op 3895) reg expression
s_mov_b32 s69, s33 // (op 3895) reg expression
s_mov_b32 s70, s18 // (op 3895) reg expression
s_mov_b32 s71, 131072 // (op 3895) call()

v_lshrrev_b32 v13, 3, v1 // (op 4035) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4035) 
v_and_b32 v18, -8, v1 // (op 4035) 
s_and_b32 s15, s18, 4294967294 // (op 4035) 
v_sub_u32 v19, v1, v18 // (op 4035) 

s_add_u32 s6, s0, s15 // (op 4035) 

v_lshlrev_b32 v18, 4, v19 // (op 4035) 

s_mul_i32 s15, s16, s50 // (op 4035) 
s_add_u32 s15, s15, s6 // (op 4035) 

s_add_u32 s6, s15, s9 // (op 4035) 

s_mul_hi_u32 s15, s6, s13 // (op 4035) 
s_sub_u32 s7, s6, s15 // (op 4035) 
s_lshr_b32 s8, s7, 1 // (op 4035) 

s_add_u32 s7, s8, s15 // (op 4035) 


s_lshr_b32 s8, s7, s11 // (op 4035) 

s_and_b32 s7, s11, 2147483648 // (op 4035) 
s_cmp_eq_i32 s7, 0 // (op 4035) 
s_mov_b32 s15, scc // (op 4035) 

s_cmp_eq_i32 s15, 0 // (op 4035) 
s_cselect_b32 s7, s6, s8 // (op 4035) 


s_mul_hi_u32 s8, s7, s12 // (op 4035) 
s_sub_u32 s15, s7, s8 // (op 4035) 
s_lshr_b32 s28, s15, 1 // (op 4035) 

s_add_u32 s15, s28, s8 // (op 4035) 


s_lshr_b32 s8, s15, s10 // (op 4035) 

s_and_b32 s15, s10, 2147483648 // (op 4035) 
s_cmp_eq_i32 s15, 0 // (op 4035) 
s_mov_b32 s28, scc // (op 4035) 

s_cmp_eq_i32 s28, 0 // (op 4035) 
s_cselect_b32 s15, s7, s8 // (op 4035) 


s_mul_i32 s8, s15, s14 // (op 4035) 

s_sub_u32 s15, s7, s8 // (op 4035) 

s_mul_i32 s8, s7, s18 // (op 4035) 


v_lshl_add_u32 v19, s15, 5, v13 // (op 4035) 


s_sub_u32 s7, s6, s8 // (op 4035) 


v_mul_lo_u32 v23, v19, s37 // (op 4035) most significant: low of low * high
v_mul_hi_u32 v13, v19, s36 // (op 4035) most significant: high of low * low
v_mul_lo_u32 v22, v19, s36 // (op 4035) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 4035) most significant: sum


v_lshl_add_u32 v13, s7, 7, v18 // (op 4035) 


v_mov_b32 v19, 0 // (op 4035) convert
v_mov_b32 v18, v13 // (op 4035) convert

v_add_co_u32 v32, s[18:19], v18, v22 // (op 4035) least significant half
v_addc_co_u32 v33, s[18:19], v19, v23, s[18:19] // (op 4035) most significant half



s_mul_i32 s19, s36, s23 // (op 4037) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 4037) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 4037) least significant
s_mul_hi_u32 s6, s36, s22 // (op 4037) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4037) most significant: sum
s_add_u32 s19, s19, s6 // (op 4037) most significant: sum


s_mov_b32 s72, s38 // (op 4037) reg expression
s_mov_b32 s73, s39 // (op 4037) reg expression
s_mov_b32 s74, s18 // (op 4037) reg expression
s_mov_b32 s75, 131072 // (op 4037) call()

v_lshrrev_b32 v13, 2, v1 // (op 4163) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4163) 
v_lshlrev_b32 v18, 1, v13 // (op 4163) 

s_and_b32 s15, s18, 4294967294 // (op 4163) 
v_and_b32 v13, -4, v1 // (op 4163) 
s_add_u32 s6, s0, s15 // (op 4163) 

v_sub_u32 v19, v1, v13 // (op 4163) 

s_mul_i32 s15, s16, s50 // (op 4163) 
s_add_u32 s15, s15, s6 // (op 4163) 

s_add_u32 s6, s15, s9 // (op 4163) 

s_mul_hi_u32 s15, s6, s13 // (op 4163) 
s_sub_u32 s7, s6, s15 // (op 4163) 
s_lshr_b32 s8, s7, 1 // (op 4163) 

s_add_u32 s7, s8, s15 // (op 4163) 


s_lshr_b32 s8, s7, s11 // (op 4163) 

s_and_b32 s7, s11, 2147483648 // (op 4163) 
s_cmp_eq_i32 s7, 0 // (op 4163) 
s_mov_b32 s15, scc // (op 4163) 

s_cmp_eq_i32 s15, 0 // (op 4163) 
s_cselect_b32 s7, s6, s8 // (op 4163) 


s_mul_hi_u32 s8, s7, s12 // (op 4163) 
s_sub_u32 s15, s7, s8 // (op 4163) 
s_lshr_b32 s28, s15, 1 // (op 4163) 

s_add_u32 s15, s28, s8 // (op 4163) 


s_lshr_b32 s8, s15, s10 // (op 4163) 

s_and_b32 s15, s10, 2147483648 // (op 4163) 
s_cmp_eq_i32 s15, 0 // (op 4163) 
s_mov_b32 s28, scc // (op 4163) 

s_cmp_eq_i32 s28, 0 // (op 4163) 
s_cselect_b32 s15, s7, s8 // (op 4163) 


s_mul_i32 s8, s7, s18 // (op 4163) 


v_lshl_add_u32 v13, s15, 7, v18 // (op 4163) 


s_sub_u32 s7, s6, s8 // (op 4163) 


v_lshl_add_u32 v18, s7, 2, v19 // (op 4163) 


v_mul_lo_u32 v19, v13, s40 // (op 4163) 
v_add_u32 v19, v19, v18 // (op 4163) 


s_mul_i32 s19, s40, s35 // (op 4165) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4165) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4165) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4165) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4165) most significant: sum
s_add_u32 s19, s19, s6 // (op 4165) most significant: sum


s_mov_b32 s76, s42 // (op 4165) reg expression
s_mov_b32 s77, s43 // (op 4165) reg expression
s_mov_b32 s78, s18 // (op 4165) reg expression
s_mov_b32 s79, 131072 // (op 4165) call()

v_lshrrev_b32 v13, 6, v1 // (op 4291) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4291) 
v_and_b32 v18, 1, v13 // (op 4291) 

s_and_b32 s15, s18, 4294967294 // (op 4291) 
v_and_b32 v13, 15, v1 // (op 4291) 
s_add_u32 s6, s0, s15 // (op 4291) 

v_lshl_add_u32 v22, v18, 4, v13 // (op 4291) 


v_and_b32 v13, 63, v1 // (op 4291) 
s_mul_i32 s15, s16, s50 // (op 4291) 
s_add_u32 s15, s15, s6 // (op 4291) 

v_lshrrev_b32 v18, 4, v13 // (op 4291) 

s_add_u32 s6, s15, s9 // (op 4291) 

s_mul_hi_u32 s15, s6, s13 // (op 4291) 
s_sub_u32 s7, s6, s15 // (op 4291) 
s_lshr_b32 s8, s7, 1 // (op 4291) 

s_add_u32 s7, s8, s15 // (op 4291) 


s_lshr_b32 s8, s7, s11 // (op 4291) 

s_and_b32 s7, s11, 2147483648 // (op 4291) 
s_cmp_eq_i32 s7, 0 // (op 4291) 
s_mov_b32 s15, scc // (op 4291) 

s_cmp_eq_i32 s15, 0 // (op 4291) 
s_cselect_b32 s7, s6, s8 // (op 4291) 


s_mul_hi_u32 s8, s7, s12 // (op 4291) 
s_sub_u32 s15, s7, s8 // (op 4291) 
s_lshr_b32 s28, s15, 1 // (op 4291) 

s_add_u32 s15, s28, s8 // (op 4291) 


s_lshr_b32 s8, s15, s10 // (op 4291) 

s_and_b32 s15, s10, 2147483648 // (op 4291) 
s_cmp_eq_i32 s15, 0 // (op 4291) 
s_mov_b32 s28, scc // (op 4291) 

s_cmp_eq_i32 s28, 0 // (op 4291) 
s_cselect_b32 s15, s7, s8 // (op 4291) 


s_mul_i32 s8, s15, s14 // (op 4291) 

s_sub_u32 s15, s7, s8 // (op 4291) 

v_lshl_add_u32 v13, s15, 5, v22 // (op 4291) 


s_mul_i32 s8, s7, s18 // (op 4291) 


v_mul_lo_u32 v22, v13, s44 // (op 4291) 

s_sub_u32 s7, s6, s8 // (op 4291) 


v_lshl_add_u32 v13, s7, 2, v18 // (op 4291) 


v_add_u32 v18, v13, v22 // (op 4291) 


s_mul_i32 s19, s44, s23 // (op 4293) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4293) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4293) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4293) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4293) most significant: sum
s_add_u32 s19, s19, s6 // (op 4293) most significant: sum


s_mov_b32 s80, s46 // (op 4293) reg expression
s_mov_b32 s81, s47 // (op 4293) reg expression
s_mov_b32 s82, s18 // (op 4293) reg expression
s_mov_b32 s83, 131072 // (op 4293) call()

v_and_b32 v13, -8, v1 // (op 4732) 
v_sub_u32 v22, v1, v13 // (op 4732) 

v_lshlrev_b32 v13, 4, v22 // (op 4732) 

v_lshrrev_b32 v22, 3, v1 // (op 4732) 
v_lshl_add_u32 v23, v22, 7, v13 // (op 4732) 


s_nop 0
v_readfirstlane_b32 s15, v23 // (op 4732) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 4807) 
v_sub_u32 v22, v1, v13 // (op 4807) 

v_lshlrev_b32 v13, 4, v22 // (op 4807) 

v_lshrrev_b32 v22, 3, v1 // (op 4807) 
v_lshl_add_u32 v23, v22, 7, v13 // (op 4807) 


s_nop 0
v_readfirstlane_b32 s6, v23 // (op 4807) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -4, v1 // (op 4872) 
v_sub_u32 v22, v1, v13 // (op 4872) 

v_lshrrev_b32 v13, 2, v1 // (op 4872) 
v_lshl_add_u32 v23, v13, 3, v22 // (op 4872) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 912) 
s_and_b32 s7, s18, 4294967294 // (op 912) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 911) 
s_mov_b32 s8, s18 // (op 911) convert

s_cmp_lt_u32 s7, s8 // (op 911) 












s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_911 // (op 911) Condition: Top (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_911 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_911:
s_barrier  // (op 5346) 
s_add_u32 s7, s7, 1 // (op 913) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 3808) 
v_mov_b64 v[16:17], s[18:19] // (op 3808) convert


v_mov_b64 v[24:25], v[26:27] // (op 3808) 
v_mov_b64 v[20:21], v[24:25] // (op 3808) 
s_add_u32 s8, 4096, s15 // (op 3808) 
s_mov_b32 m0, s8 // (op 3808) reg expression
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 3808) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 3808) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 3808) most significant half

s_add_u32 m0, m0, 4096 // (op 3808) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 3808) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 3808) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 3808) most significant half

s_add_u32 m0, m0, 4096 // (op 3808) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 3808) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v16 // (op 3808) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 3808) most significant half

s_add_u32 m0, m0, 4096 // (op 3808) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 3808) Load value direct to lds




v_add_co_u32 v26, vcc, 128, v26 // (op 3883) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 3883) most significant half
s_barrier  // (op 5365) 


v_mov_b64 v[16:17], v[32:33] // (op 3814) 

s_mov_b32 m0, s6 // (op 3814) reg expression
buffer_load_dwordx4 v16, s[72:75], 0 offen offset: 0 lds // (op 3814) Load value direct to lds

v_add_co_u32 v32, vcc, 128, v32 // (op 4027) least significant half
v_addc_co_u32 v33, vcc, 0, v33, vcc // (op 4027) most significant half

v_mov_b64 v[16:17], s[40:41] // (op 452) convert

v_mov_b32 v13, v19 // (op 452) 
v_mov_b32 v20, v13 // (op 452) 
buffer_load_ubyte v24, v20, s[76:79], 0 offen offset: 0 // (op 452) Load value
v_add_u32 v20, v20, v16 // (op 452) 
buffer_load_ubyte v25, v20, s[76:79], 0 offen offset: 0 // (op 452) Load value



v_add_u32 v19, 4, v19 // (op 4155) 
s_barrier  // (op 5384) 
s_waitcnt vmcnt(3)
s_barrier  // (op 5352) 






ds_read_b128 v[34:37], v9 offset:4096 // (op 441) Load local data 
ds_read_b128 v[38:41], v9 offset:4160 // (op 441) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v24 offset:40960 // (op 450) Store local data 
s_barrier  // (op 5371) 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v25 offset:40964 // (op 450) Store local data 




ds_read_b128 v[42:45], v4 // (op 443) Load local data 
ds_read_b128 v[46:49], v4 offset:64 // (op 443) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5391) 



ds_read_u8 v13, v8 offset:40960 // (op 445) Load local data 


v_mov_b32 v16, v18 // (op 447) 

buffer_load_ubyte v17, v16, s[80:83], 0 offen offset: 0 // (op 447) Load value

v_add_u32 v18, 4, v18 // (op 4283) 



ds_read_b128 v[50:53], v9 offset:8192 // (op 593) Load local data 
ds_read_b128 v[54:57], v9 offset:8256 // (op 593) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:49], a[0:3], v13, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 448) 





ds_read_u8 v13, v8 offset:41088 // (op 597) Load local data 



ds_read_b128 v[34:37], v9 offset:12288 // (op 755) Load local data 
ds_read_b128 v[38:41], v9 offset:12352 // (op 755) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[42:49], a[4:7], v13, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 599) 





ds_read_u8 v13, v8 offset:41216 // (op 759) Load local data 



ds_read_b128 v[50:53], v9 offset:16384 // (op 917) Load local data 
ds_read_b128 v[54:57], v9 offset:16448 // (op 917) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:41], v[42:49], a[8:11], v13, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 761) 





ds_read_u8 v13, v8 offset:41344 // (op 921) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[42:49], a[12:15], v13, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 923) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 911) 
s_mov_b32 s8, s18 // (op 911) convert

s_cmp_lt_u32 s7, s8 // (op 911) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_911 // (op 911) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_911 if true)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_911:








s_barrier  // (op 5523) 



v_cmp_eq_f32 vcc, 0.00000, v14 // (op 218) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_218 // (op 218) Condition: False, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_218
v_accvgpr_read v16, a0 // (op 226) 
v_accvgpr_read v17, a1 // (op 226) 
v_accvgpr_read v18, a2 // (op 226) 
v_accvgpr_read v19, a3 // (op 226) 
v_mul_f32 v20, v15, v16 // (op 226) 
v_mul_f32 v21, v15, v17 // (op 226) 
v_mul_f32 v22, v15, v18 // (op 226) 
v_mul_f32 v23, v15, v19 // (op 226) 




buffer_store_dwordx4 v[20:23], v7, s[60:63], 0 offen // (op 228) Store value

v_accvgpr_read v16, a4 // (op 1153) 
v_accvgpr_read v17, a5 // (op 1153) 
v_accvgpr_read v18, a6 // (op 1153) 
v_accvgpr_read v19, a7 // (op 1153) 
v_mul_f32 v20, v15, v16 // (op 1153) 
v_mul_f32 v21, v15, v17 // (op 1153) 
v_mul_f32 v22, v15, v18 // (op 1153) 
v_mul_f32 v23, v15, v19 // (op 1153) 




buffer_store_dwordx4 v[20:23], v7, s[60:63], 0 offen offset:128 // (op 1155) Store value

v_accvgpr_read v16, a8 // (op 1171) 
v_accvgpr_read v17, a9 // (op 1171) 
v_accvgpr_read v18, a10 // (op 1171) 
v_accvgpr_read v19, a11 // (op 1171) 
v_mul_f32 v20, v15, v16 // (op 1171) 
v_mul_f32 v21, v15, v17 // (op 1171) 
v_mul_f32 v22, v15, v18 // (op 1171) 
v_mul_f32 v23, v15, v19 // (op 1171) 




buffer_store_dwordx4 v[20:23], v7, s[60:63], 0 offen offset:256 // (op 1173) Store value

v_accvgpr_read v16, a12 // (op 1189) 
v_accvgpr_read v17, a13 // (op 1189) 
v_accvgpr_read v18, a14 // (op 1189) 
v_accvgpr_read v19, a15 // (op 1189) 
v_mul_f32 v20, v15, v16 // (op 1189) 
v_mul_f32 v21, v15, v17 // (op 1189) 
v_mul_f32 v22, v15, v18 // (op 1189) 
v_mul_f32 v23, v15, v19 // (op 1189) 




buffer_store_dwordx4 v[20:23], v7, s[60:63], 0 offen offset:384 // (op 1191) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_218 // (op 218) Condition: Done, jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_218
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_218:



buffer_load_dwordx4 v[16:19], v6, s[56:59], 0 offen offset:384 // (op 1265) Load value
v_accvgpr_read v20, a12 // (op 1262) 
v_accvgpr_read v21, a13 // (op 1262) 
v_accvgpr_read v22, a14 // (op 1262) 
v_accvgpr_read v23, a15 // (op 1262) 
v_mul_f32 v24, v15, v20 // (op 1262) 
v_mul_f32 v25, v15, v21 // (op 1262) 
v_mul_f32 v26, v15, v22 // (op 1262) 
v_mul_f32 v27, v15, v23 // (op 1262) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v14, v16 // (op 1266) 
v_mul_f32 v21, v14, v17 // (op 1266) 
v_mul_f32 v22, v14, v18 // (op 1266) 
v_mul_f32 v23, v14, v19 // (op 1266) 

v_add_f32 v16, v20, v24 // (op 1263) 
v_add_f32 v17, v21, v25 // (op 1263) 
v_add_f32 v18, v22, v26 // (op 1263) 
v_add_f32 v19, v23, v27 // (op 1263) 





buffer_store_dwordx4 v[16:19], v7, s[60:63], 0 offen offset:384 // (op 1264) Store value




s_nop 1
buffer_load_dwordx4 v[16:19], v6, s[56:59], 0 offen offset:256 // (op 1247) Load value
v_accvgpr_read v20, a8 // (op 1244) 
v_accvgpr_read v21, a9 // (op 1244) 
v_accvgpr_read v22, a10 // (op 1244) 
v_accvgpr_read v23, a11 // (op 1244) 
v_mul_f32 v24, v15, v20 // (op 1244) 
v_mul_f32 v25, v15, v21 // (op 1244) 
v_mul_f32 v26, v15, v22 // (op 1244) 
v_mul_f32 v27, v15, v23 // (op 1244) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v14, v16 // (op 1248) 
v_mul_f32 v21, v14, v17 // (op 1248) 
v_mul_f32 v22, v14, v18 // (op 1248) 
v_mul_f32 v23, v14, v19 // (op 1248) 

v_add_f32 v16, v20, v24 // (op 1245) 
v_add_f32 v17, v21, v25 // (op 1245) 
v_add_f32 v18, v22, v26 // (op 1245) 
v_add_f32 v19, v23, v27 // (op 1245) 





buffer_store_dwordx4 v[16:19], v7, s[60:63], 0 offen offset:256 // (op 1246) Store value




s_nop 1
buffer_load_dwordx4 v[16:19], v6, s[56:59], 0 offen offset:128 // (op 1229) Load value
v_accvgpr_read v20, a4 // (op 1226) 
v_accvgpr_read v21, a5 // (op 1226) 
v_accvgpr_read v22, a6 // (op 1226) 
v_accvgpr_read v23, a7 // (op 1226) 
v_mul_f32 v24, v15, v20 // (op 1226) 
v_mul_f32 v25, v15, v21 // (op 1226) 
v_mul_f32 v26, v15, v22 // (op 1226) 
v_mul_f32 v27, v15, v23 // (op 1226) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v14, v16 // (op 1230) 
v_mul_f32 v21, v14, v17 // (op 1230) 
v_mul_f32 v22, v14, v18 // (op 1230) 
v_mul_f32 v23, v14, v19 // (op 1230) 

v_add_f32 v16, v20, v24 // (op 1227) 
v_add_f32 v17, v21, v25 // (op 1227) 
v_add_f32 v18, v22, v26 // (op 1227) 
v_add_f32 v19, v23, v27 // (op 1227) 





buffer_store_dwordx4 v[16:19], v7, s[60:63], 0 offen offset:128 // (op 1228) Store value




s_nop 1
buffer_load_dwordx4 v[16:19], v6, s[56:59], 0 offen // (op 241) Load value
v_accvgpr_read v20, a0 // (op 238) 
v_accvgpr_read v21, a1 // (op 238) 
v_accvgpr_read v22, a2 // (op 238) 
v_accvgpr_read v23, a3 // (op 238) 
v_mul_f32 v24, v15, v20 // (op 238) 
v_mul_f32 v25, v15, v21 // (op 238) 
v_mul_f32 v26, v15, v22 // (op 238) 
v_mul_f32 v27, v15, v23 // (op 238) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v14, v16 // (op 242) 
v_mul_f32 v21, v14, v17 // (op 242) 
v_mul_f32 v22, v14, v18 // (op 242) 
v_mul_f32 v23, v14, v19 // (op 242) 

v_add_f32 v16, v20, v24 // (op 239) 
v_add_f32 v17, v21, v25 // (op 239) 
v_add_f32 v18, v22, v26 // (op 239) 
v_add_f32 v19, v23, v27 // (op 239) 





buffer_store_dwordx4 v[16:19], v7, s[60:63], 0 offen // (op 240) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_218:
s_add_u32 s0, s0, s1 // (op 427) 


s_mul_i32 s15, s50, s16 // (op 425) 
s_cmp_gt_u32 s51, s15 // (op 425) 
s_mov_b32 s6, scc // (op 425) 
s_sub_u32 s7, s51, s15 // (op 425) 

s_cmp_eq_i32 s6, 0 // (op 425) 
s_cselect_b32 s8, 0, s7 // (op 425) 


s_cmp_lt_u32 s50, s8 // (op 425) 
s_mov_b32 s15, scc // (op 425) 
s_cmp_eq_i32 s15, 0 // (op 425) 
s_cselect_b32 s6, s8, s50 // (op 425) 


s_cmp_lt_u32 s0, s6 // (op 425) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_425 // (op 425) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_425 if true)
RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_425:











































s_endpgm  // End of RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.LRR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_end:
.size RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2, .LRR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_end-RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
  .amdhsa_next_free_vgpr 108
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 41984 // lds bytes
  .amdhsa_accum_offset 92
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
    DataTypeB: F8
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
    macrotile: [128, 32, 128]
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
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_TN_BF8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 41984
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 88
    .vgpr_count: 90
    .agpr_count: 16
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

