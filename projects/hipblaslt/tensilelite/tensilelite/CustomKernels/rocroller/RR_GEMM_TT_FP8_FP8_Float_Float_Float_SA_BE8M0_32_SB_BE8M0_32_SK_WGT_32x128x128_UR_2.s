// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2,@function
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_exec_begin:


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
s_load_dwordx2 s[52:53], s[0:1], 184 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_lshrrev_b32 v0, 7, v1 // (op 4331) 
v_and_b32 v3, 15, v1 // (op 4331) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4331) 


v_and_b32 v0, 48, v1 // (op 4331) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 4331) 


v_lshrrev_b32 v0, 7, v1 // (op 4589) 
v_and_b32 v4, 15, v1 // (op 4589) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 4589) 


v_and_b32 v0, 48, v1 // (op 4589) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4589) 


v_lshrrev_b32 v0, 7, v1 // (op 4834) 
v_and_b32 v5, 15, v1 // (op 4834) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 4834) 


v_and_b32 v0, 48, v1 // (op 4834) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 4834) 


v_lshrrev_b32 v0, 7, v1 // (op 4946) 
v_and_b32 v6, 15, v1 // (op 4946) 
v_lshl_add_u32 v7, v0, 4, v6 // (op 4946) 


v_and_b32 v0, 48, v1 // (op 4946) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 4946) 


v_and_b32 v0, 48, v1 // (op 4401) 
v_and_b32 v7, 63, v1 // (op 4401) 
v_sub_u32 v8, v7, v0 // (op 4401) 

v_lshrrev_b32 v7, 1, v8 // (op 4401) 
v_add_lshl_u32 v9, v0, v7, 7 // (op 4401) 


v_and_b32 v0, 1, v8 // (op 4401) 

v_lshlrev_b32 v7, 3, v0 // (op 4401) 

v_lshrrev_b32 v0, 6, v1 // (op 4401) 
v_and_b32 v8, 1, v0 // (op 4401) 

v_lshl_add_u32 v0, v8, 4, v7 // (op 4401) 


v_add_u32 v7, v9, v0 // (op 4401) 


v_lshrrev_b32 v0, 6, v1 // (op 4477) 
v_and_b32 v8, 1, v0 // (op 4477) 

v_and_b32 v0, 15, v1 // (op 4477) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 4477) 


v_and_b32 v0, 63, v1 // (op 4477) 
v_lshrrev_b32 v8, 4, v0 // (op 4477) 

v_lshl_add_u32 v0, v8, 7, v9 // (op 4477) 


v_and_b32 v8, 48, v1 // (op 4659) 
v_and_b32 v9, 63, v1 // (op 4659) 
v_sub_u32 v10, v9, v8 // (op 4659) 

v_lshrrev_b32 v9, 1, v10 // (op 4659) 
v_add_lshl_u32 v11, v8, v9, 7 // (op 4659) 


v_and_b32 v8, 1, v10 // (op 4659) 

v_lshlrev_b32 v9, 3, v8 // (op 4659) 

v_lshrrev_b32 v8, 6, v1 // (op 4659) 
v_and_b32 v10, 1, v8 // (op 4659) 

v_lshl_add_u32 v8, v10, 4, v9 // (op 4659) 


v_add_u32 v9, v11, v8 // (op 4659) 


v_lshrrev_b32 v8, 6, v1 // (op 4735) 
v_and_b32 v10, 1, v8 // (op 4735) 

v_and_b32 v8, 15, v1 // (op 4735) 
v_lshl_add_u32 v11, v10, 4, v8 // (op 4735) 


v_and_b32 v8, 63, v1 // (op 4735) 
v_lshrrev_b32 v10, 4, v8 // (op 4735) 

v_lshl_add_u32 v8, v10, 7, v11 // (op 4735) 


v_and_b32 v10, 48, v1 // (op 4878) 
v_and_b32 v11, 63, v1 // (op 4878) 
v_sub_u32 v12, v11, v10 // (op 4878) 

v_lshrrev_b32 v11, 1, v12 // (op 4878) 
v_add_lshl_u32 v13, v10, v11, 7 // (op 4878) 


v_and_b32 v10, 1, v12 // (op 4878) 

v_lshlrev_b32 v11, 3, v10 // (op 4878) 

v_lshrrev_b32 v10, 6, v1 // (op 4878) 
v_and_b32 v12, 1, v10 // (op 4878) 

v_lshl_add_u32 v10, v12, 4, v11 // (op 4878) 


v_add_u32 v11, v13, v10 // (op 4878) 


v_lshrrev_b32 v10, 6, v1 // (op 4907) 
v_and_b32 v12, 1, v10 // (op 4907) 

v_and_b32 v10, 15, v1 // (op 4907) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 4907) 


v_and_b32 v10, 63, v1 // (op 4907) 
v_lshrrev_b32 v12, 4, v10 // (op 4907) 

v_lshl_add_u32 v10, v12, 7, v13 // (op 4907) 


v_and_b32 v12, 48, v1 // (op 4990) 
v_and_b32 v13, 63, v1 // (op 4990) 
v_sub_u32 v14, v13, v12 // (op 4990) 

v_lshrrev_b32 v13, 1, v14 // (op 4990) 
v_add_lshl_u32 v15, v12, v13, 7 // (op 4990) 


v_and_b32 v12, 1, v14 // (op 4990) 

v_lshlrev_b32 v13, 3, v12 // (op 4990) 

v_lshrrev_b32 v12, 6, v1 // (op 4990) 
v_and_b32 v14, 1, v12 // (op 4990) 

v_lshl_add_u32 v12, v14, 4, v13 // (op 4990) 


v_add_u32 v13, v15, v12 // (op 4990) 


v_lshrrev_b32 v12, 6, v1 // (op 5019) 
v_and_b32 v14, 1, v12 // (op 5019) 

v_and_b32 v12, 15, v1 // (op 5019) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 5019) 


v_and_b32 v12, 63, v1 // (op 5019) 
v_lshrrev_b32 v14, 4, v12 // (op 5019) 

v_lshl_add_u32 v12, v14, 7, v15 // (op 5019) 


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
s_mul_i32 s54, s8, s16 // (op 270) 
s_cmp_gt_u32 s9, s54 // (op 270) 
s_mov_b32 s55, scc // (op 270) 
s_sub_u32 s56, s9, s54 // (op 270) 

s_cmp_eq_i32 s55, 0 // (op 270) 
s_cselect_b32 s54, 0, s56 // (op 270) 


s_cmp_lt_u32 s8, s54 // (op 270) 
s_mov_b32 s55, scc // (op 270) 
s_cmp_eq_i32 s55, 0 // (op 270) 
s_cselect_b32 s56, s54, s8 // (op 270) 


s_cmp_lt_u32 s0, s56 // (op 270) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 // (op 270) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_270 if false)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_270:
v_lshrrev_b32 v16, 7, v1 // (op 4252) 
s_mul_i32 s54, s16, s8 // (op 4252) 
s_add_u32 s54, s54, s0 // (op 4252) 
v_and_b32 v17, 63, v1 // (op 4252) 
s_mul_hi_u32 s55, s54, s13 // (op 4252) 
v_lshrrev_b32 v18, 4, v17 // (op 4252) 
s_sub_u32 s56, s54, s55 // (op 4252) 
v_lshlrev_b32 v19, 2, v18 // (op 4252) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 4252) 


s_lshr_b32 s57, s56, 1 // (op 4252) 

s_add_u32 s56, s57, s55 // (op 4252) 


v_lshrrev_b32 v16, 2, v17 // (op 4252) 

s_lshr_b32 s55, s56, s11 // (op 4252) 

v_and_b32 v17, 3, v16 // (op 4252) 

s_and_b32 s56, s11, 2147483648 // (op 4252) 
v_and_b32 v16, 3, v1 // (op 4252) 
s_cmp_eq_i32 s56, 0 // (op 4252) 
s_mov_b32 s57, scc // (op 4252) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 4252) 


s_cmp_eq_i32 s57, 0 // (op 4252) 
s_cselect_b32 s56, s54, s55 // (op 4252) 



v_lshrrev_b32 v16, 6, v1 // (op 4252) 
s_mul_hi_u32 s54, s56, s12 // (op 4252) 
v_and_b32 v17, 1, v16 // (op 4252) 

s_sub_u32 s55, s56, s54 // (op 4252) 
v_lshl_add_u32 v16, v17, 4, v19 // (op 4252) 


s_lshr_b32 s57, s55, 1 // (op 4252) 

s_add_u32 s55, s57, s54 // (op 4252) 


s_lshr_b32 s54, s55, s10 // (op 4252) 

s_and_b32 s55, s10, 2147483648 // (op 4252) 
s_cmp_eq_i32 s55, 0 // (op 4252) 
s_mov_b32 s57, scc // (op 4252) 

s_cmp_eq_i32 s57, 0 // (op 4252) 
s_cselect_b32 s55, s56, s54 // (op 4252) 


s_mul_i32 s54, s55, s14 // (op 4252) 
v_lshl_add_u32 v17, s55, 5, v18 // (op 4252) 


s_sub_u32 s55, s56, s54 // (op 4252) 


v_lshl_add_u32 v18, s55, 7, v16 // (op 4252) 


v_mul_lo_u32 v16, v18, s4 // (op 4252) 

v_add_lshl_u32 v18, v17, v16, 2 // (op 4252) 


s_mul_i32 s55, s4, s23 // (op 4254) most significant: low of low * high
s_mul_i32 s56, s5, s22 // (op 4254) most significant: low of high * low
s_mul_i32 s54, s4, s22 // (op 4254) least significant
s_mul_hi_u32 s57, s4, s22 // (op 4254) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 4254) most significant: sum
s_add_u32 s55, s55, s57 // (op 4254) most significant: sum


s_lshl_b64 s[56:57], s[54:55], 2 // (op 4254) 

s_mov_b32 s60, s20 // (op 4254) reg expression
s_mov_b32 s61, s21 // (op 4254) reg expression
s_mov_b32 s62, s56 // (op 4254) reg expression
s_mov_b32 s63, 131072 // (op 4254) call()

v_lshrrev_b32 v16, 7, v1 // (op 4343) 
v_and_b32 v17, 15, v1 // (op 4343) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 4343) 


v_and_b32 v16, 48, v1 // (op 4343) 
v_lshl_add_u32 v17, v19, 7, v16 // (op 4343) 


v_and_b32 v16, 48, v1 // (op 4418) 
v_and_b32 v19, 63, v1 // (op 4418) 
v_sub_u32 v20, v19, v16 // (op 4418) 

v_lshrrev_b32 v19, 1, v20 // (op 4418) 
v_add_lshl_u32 v21, v16, v19, 7 // (op 4418) 


v_and_b32 v16, 1, v20 // (op 4418) 

v_lshlrev_b32 v19, 3, v16 // (op 4418) 

v_lshrrev_b32 v16, 6, v1 // (op 4418) 
v_and_b32 v20, 1, v16 // (op 4418) 

v_lshl_add_u32 v16, v20, 4, v19 // (op 4418) 


v_add_u32 v19, v21, v16 // (op 4418) 


v_lshrrev_b32 v16, 6, v1 // (op 4494) 
v_and_b32 v20, 1, v16 // (op 4494) 

v_and_b32 v16, 15, v1 // (op 4494) 
v_lshl_add_u32 v21, v20, 4, v16 // (op 4494) 


v_and_b32 v16, 63, v1 // (op 4494) 
v_lshrrev_b32 v20, 4, v16 // (op 4494) 

v_lshl_add_u32 v16, v20, 7, v21 // (op 4494) 


v_lshrrev_b32 v20, 7, v1 // (op 4512) 
s_mul_i32 s54, s16, s8 // (op 4512) 
s_add_u32 s54, s54, s0 // (op 4512) 
v_and_b32 v21, 63, v1 // (op 4512) 
s_mul_hi_u32 s55, s54, s13 // (op 4512) 
v_lshrrev_b32 v22, 4, v21 // (op 4512) 
s_sub_u32 s56, s54, s55 // (op 4512) 
v_lshlrev_b32 v23, 2, v22 // (op 4512) 

s_lshr_b32 s57, s56, 1 // (op 4512) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 4512) 


s_add_u32 s56, s57, s55 // (op 4512) 


v_lshrrev_b32 v20, 2, v21 // (op 4512) 

s_lshr_b32 s55, s56, s11 // (op 4512) 

v_and_b32 v21, 3, v20 // (op 4512) 

s_and_b32 s56, s11, 2147483648 // (op 4512) 
v_and_b32 v20, 3, v1 // (op 4512) 
s_cmp_eq_i32 s56, 0 // (op 4512) 
s_mov_b32 s57, scc // (op 4512) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 4512) 


s_cmp_eq_i32 s57, 0 // (op 4512) 
s_cselect_b32 s56, s54, s55 // (op 4512) 



v_lshrrev_b32 v20, 6, v1 // (op 4512) 
s_mul_hi_u32 s54, s56, s12 // (op 4512) 
v_and_b32 v21, 1, v20 // (op 4512) 

s_sub_u32 s55, s56, s54 // (op 4512) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 4512) 


s_lshr_b32 s57, s55, 1 // (op 4512) 

s_add_u32 s55, s57, s54 // (op 4512) 


s_lshr_b32 s54, s55, s10 // (op 4512) 

s_and_b32 s55, s10, 2147483648 // (op 4512) 
s_cmp_eq_i32 s55, 0 // (op 4512) 
s_mov_b32 s57, scc // (op 4512) 

s_cmp_eq_i32 s57, 0 // (op 4512) 
s_cselect_b32 s55, s56, s54 // (op 4512) 


s_mul_i32 s54, s55, s14 // (op 4512) 
v_lshl_add_u32 v21, s55, 5, v22 // (op 4512) 


s_sub_u32 s55, s56, s54 // (op 4512) 


v_lshl_add_u32 v22, s55, 7, v20 // (op 4512) 


v_mul_lo_u32 v20, v22, s24 // (op 4512) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 4512) 


s_mul_i32 s55, s24, s23 // (op 4514) most significant: low of low * high
s_mul_i32 s56, s25, s22 // (op 4514) most significant: low of high * low
s_mul_i32 s54, s24, s22 // (op 4514) least significant
s_mul_hi_u32 s57, s24, s22 // (op 4514) most significant: high of low * low
s_add_u32 s55, s55, s56 // (op 4514) most significant: sum
s_add_u32 s55, s55, s57 // (op 4514) most significant: sum


s_lshl_b64 s[56:57], s[54:55], 2 // (op 4514) 

s_mov_b32 s64, s26 // (op 4514) reg expression
s_mov_b32 s65, s27 // (op 4514) reg expression
s_mov_b32 s66, s56 // (op 4514) reg expression
s_mov_b32 s67, 131072 // (op 4514) call()

v_lshrrev_b32 v20, 5, v1 // (op 4770) 
v_lshlrev_b32 v21, 2, v20 // (op 4770) 

v_lshl_add_u32 v20, s16, 5, v21 // (op 4770) 

v_and_b32 v21, -32, v1 // (op 4770) 
v_sub_u32 v23, v1, v21 // (op 4770) 

v_lshl_add_u32 v21, v20, 7, v23 // (op 4770) 


v_lshlrev_b32 v20, 2, v21 // (op 4770) 

s_lshl_b32 s54, s15, 14 // (op 4772) 
s_mov_b32 s56, s28 // (op 4772) reg expression
s_mov_b32 s57, s29 // (op 4772) reg expression
s_mov_b32 s58, s54 // (op 4772) reg expression
s_mov_b32 s59, 131072 // (op 4772) call()

s_mul_i32 s6, s8, s16 // (op 275) 
s_add_u32 s6, s6, s0 // (op 275) 
s_mul_hi_u32 s54, s6, s13 // (op 276) 
s_sub_u32 s55, s6, s54 // (op 276) 
s_lshr_b32 s68, s55, 1 // (op 276) 

s_add_u32 s55, s68, s54 // (op 276) 


s_lshr_b32 s54, s55, s11 // (op 276) 

s_and_b32 s55, s11, 2147483648 // (op 276) 
s_cmp_eq_i32 s55, 0 // (op 276) 
s_mov_b32 s68, scc // (op 276) 

s_cmp_eq_i32 s68, 0 // (op 276) 
s_cselect_b32 s55, s6, s54 // (op 276) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 276) 
s_mul_i32 s54, s55, s68 // (op 276) 


s_sub_u32 s18, s6, s54 // (op 276) 

s_ashr_i64 s[54:55], s[2:3], 7 // (op 277) 
s_cmp_gt_u32 s54, s18 // (op 277) 
s_mov_b32 s68, scc // (op 277) 
s_sub_u32 s69, s54, s18 // (op 277) 

s_cmp_eq_i32 s68, 0 // (op 277) 
s_cselect_b32 s54, 0, s69 // (op 277) 


s_cmp_gt_u32 s8, s0 // (op 277) 
s_mov_b32 s55, scc // (op 277) 
s_sub_u32 s68, s8, s0 // (op 277) 
s_cmp_eq_i32 s55, 0 // (op 277) 
s_cselect_b32 s69, 0, s68 // (op 277) 


s_cmp_lt_u32 s54, s69 // (op 277) 
s_mov_b32 s55, scc // (op 277) 
s_cmp_eq_i32 s55, 0 // (op 277) 
s_cselect_b32 s68, s69, s54 // (op 277) 



s_cmp_gt_u32 s9, s6 // (op 277) 
s_mov_b32 s54, scc // (op 277) 
s_sub_u32 s55, s9, s6 // (op 277) 
s_cmp_eq_i32 s54, 0 // (op 277) 
s_cselect_b32 s69, 0, s55 // (op 277) 


s_cmp_lt_u32 s68, s69 // (op 277) 
s_mov_b32 s54, scc // (op 277) 
s_cmp_eq_i32 s54, 0 // (op 277) 
s_cselect_b32 s7, s69, s68 // (op 277) 



s_add_u32 s54, s18, s7 // (op 278) 
s_sub_u32 s19, s54, 1 // (op 278) 

v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 1405) call()
v_accvgpr_write a5, 0.00000 // (op 1405) call()
v_accvgpr_write a6, 0.00000 // (op 1405) call()
v_accvgpr_write a7, 0.00000 // (op 1405) call()
v_accvgpr_write a8, 0.00000 // (op 1561) call()
v_accvgpr_write a9, 0.00000 // (op 1561) call()
v_accvgpr_write a10, 0.00000 // (op 1561) call()
v_accvgpr_write a11, 0.00000 // (op 1561) call()
v_accvgpr_write a12, 0.00000 // (op 1717) call()
v_accvgpr_write a13, 0.00000 // (op 1717) call()
v_accvgpr_write a14, 0.00000 // (op 1717) call()
v_accvgpr_write a15, 0.00000 // (op 1717) call()
v_lshrrev_b32 v21, 3, v1 // (op 3824) 
s_mul_i32 s54, s16, s8 // (op 3824) 
s_add_u32 s54, s54, s0 // (op 3824) 
v_and_b32 v23, -8, v1 // (op 3824) 
s_mul_hi_u32 s55, s54, s13 // (op 3824) 
v_sub_u32 v24, v1, v23 // (op 3824) 

s_sub_u32 s68, s54, s55 // (op 3824) 
v_lshlrev_b32 v23, 4, v24 // (op 3824) 

s_lshr_b32 s69, s68, 1 // (op 3824) 

s_add_u32 s68, s69, s55 // (op 3824) 


s_lshr_b32 s55, s68, s11 // (op 3824) 

s_and_b32 s68, s11, 2147483648 // (op 3824) 
s_cmp_eq_i32 s68, 0 // (op 3824) 
s_mov_b32 s69, scc // (op 3824) 

s_cmp_eq_i32 s69, 0 // (op 3824) 
s_cselect_b32 s68, s54, s55 // (op 3824) 


s_mul_hi_u32 s55, s68, s12 // (op 3824) 
s_sub_u32 s69, s68, s55 // (op 3824) 
s_lshr_b32 s70, s69, 1 // (op 3824) 

s_add_u32 s69, s70, s55 // (op 3824) 


s_lshr_b32 s55, s69, s10 // (op 3824) 

s_and_b32 s69, s10, 2147483648 // (op 3824) 
s_cmp_eq_i32 s69, 0 // (op 3824) 
s_mov_b32 s70, scc // (op 3824) 

s_cmp_eq_i32 s70, 0 // (op 3824) 
s_cselect_b32 s69, s68, s55 // (op 3824) 


v_lshl_add_u32 v24, s69, 5, v21 // (op 3824) 


s_ashr_i64 s[70:71], s[2:3], 7 // (op 3824) 
s_mul_i32 s55, s68, s70 // (op 3824) 


s_sub_u32 s68, s54, s55 // (op 3824) 


v_lshl_add_u32 v21, s68, 7, v23 // (op 3824) 


v_mov_b32 v27, 0 // (op 3824) convert
v_mov_b32 v26, v21 // (op 3824) convert

v_mul_lo_u32 v29, v24, s31 // (op 3824) most significant: low of low * high
v_mul_hi_u32 v21, v24, s30 // (op 3824) most significant: high of low * low
v_mul_lo_u32 v28, v24, s30 // (op 3824) least significant: low of low * low
v_add_u32 v29, v29, v21 // (op 3824) most significant: sum

v_add_co_u32 v28, s[54:55], v28, v26 // (op 3824) least significant half
v_addc_co_u32 v29, s[54:55], v29, v27, s[54:55] // (op 3824) most significant half



s_mul_i32 s55, s30, s35 // (op 3826) most significant: low of low * high
s_mul_i32 s68, s31, s34 // (op 3826) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 3826) least significant
s_mul_hi_u32 s69, s30, s34 // (op 3826) most significant: high of low * low
s_add_u32 s55, s55, s68 // (op 3826) most significant: sum
s_add_u32 s55, s55, s69 // (op 3826) most significant: sum


s_mov_b32 s68, s32 // (op 3826) reg expression
s_mov_b32 s69, s33 // (op 3826) reg expression
s_mov_b32 s70, s54 // (op 3826) reg expression
s_mov_b32 s71, 131072 // (op 3826) call()

v_lshrrev_b32 v21, 3, v1 // (op 3843) 
s_mul_i32 s54, s16, s8 // (op 3843) 
s_add_u32 s54, s54, s0 // (op 3843) 
v_and_b32 v23, -8, v1 // (op 3843) 
s_mul_hi_u32 s55, s54, s13 // (op 3843) 
v_sub_u32 v24, v1, v23 // (op 3843) 

v_lshlrev_b32 v23, 4, v24 // (op 3843) 

s_sub_u32 s72, s54, s55 // (op 3843) 
s_lshr_b32 s73, s72, 1 // (op 3843) 

s_add_u32 s72, s73, s55 // (op 3843) 


s_lshr_b32 s55, s72, s11 // (op 3843) 

s_and_b32 s72, s11, 2147483648 // (op 3843) 
s_cmp_eq_i32 s72, 0 // (op 3843) 
s_mov_b32 s73, scc // (op 3843) 

s_cmp_eq_i32 s73, 0 // (op 3843) 
s_cselect_b32 s72, s54, s55 // (op 3843) 


s_mul_hi_u32 s55, s72, s12 // (op 3843) 
s_sub_u32 s73, s72, s55 // (op 3843) 
s_lshr_b32 s74, s73, 1 // (op 3843) 

s_add_u32 s73, s74, s55 // (op 3843) 


s_lshr_b32 s55, s73, s10 // (op 3843) 

s_and_b32 s73, s10, 2147483648 // (op 3843) 
s_cmp_eq_i32 s73, 0 // (op 3843) 
s_mov_b32 s74, scc // (op 3843) 

s_cmp_eq_i32 s74, 0 // (op 3843) 
s_cselect_b32 s73, s72, s55 // (op 3843) 


v_lshl_add_u32 v24, s73, 5, v21 // (op 3843) 


s_ashr_i64 s[74:75], s[2:3], 7 // (op 3843) 
s_mul_i32 s55, s72, s74 // (op 3843) 


s_sub_u32 s72, s54, s55 // (op 3843) 


v_lshl_add_u32 v21, s72, 7, v23 // (op 3843) 


v_mov_b32 v27, 0 // (op 3843) convert
v_mov_b32 v26, v21 // (op 3843) convert

v_mul_lo_u32 v31, v24, s31 // (op 3843) most significant: low of low * high
v_mul_hi_u32 v21, v24, s30 // (op 3843) most significant: high of low * low
v_mul_lo_u32 v30, v24, s30 // (op 3843) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 3843) most significant: sum

v_add_co_u32 v30, s[54:55], v30, v26 // (op 3843) least significant half
v_addc_co_u32 v31, s[54:55], v31, v27, s[54:55] // (op 3843) most significant half



s_mul_i32 s55, s30, s35 // (op 3845) most significant: low of low * high
s_mul_i32 s72, s31, s34 // (op 3845) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 3845) least significant
s_mul_hi_u32 s73, s30, s34 // (op 3845) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 3845) most significant: sum
s_add_u32 s55, s55, s73 // (op 3845) most significant: sum


s_mov_b32 s68, s32 // (op 3845) reg expression
s_mov_b32 s69, s33 // (op 3845) reg expression
s_mov_b32 s70, s54 // (op 3845) reg expression
s_mov_b32 s71, 131072 // (op 3845) call()

v_and_b32 v21, -8, v1 // (op 3952) 
s_mul_i32 s54, s16, s8 // (op 3952) 
s_add_u32 s54, s54, s0 // (op 3952) 
v_sub_u32 v23, v1, v21 // (op 3952) 

s_mul_hi_u32 s55, s54, s13 // (op 3952) 
v_lshlrev_b32 v21, 4, v23 // (op 3952) 

s_sub_u32 s72, s54, s55 // (op 3952) 
v_lshrrev_b32 v23, 3, v1 // (op 3952) 
s_lshr_b32 s73, s72, 1 // (op 3952) 

s_add_u32 s72, s73, s55 // (op 3952) 


s_lshr_b32 s55, s72, s11 // (op 3952) 

s_and_b32 s72, s11, 2147483648 // (op 3952) 
s_cmp_eq_i32 s72, 0 // (op 3952) 
s_mov_b32 s73, scc // (op 3952) 

s_cmp_eq_i32 s73, 0 // (op 3952) 
s_cselect_b32 s72, s54, s55 // (op 3952) 


s_mul_hi_u32 s55, s72, s12 // (op 3952) 
s_sub_u32 s73, s72, s55 // (op 3952) 
s_lshr_b32 s74, s73, 1 // (op 3952) 

s_add_u32 s73, s74, s55 // (op 3952) 


s_lshr_b32 s55, s73, s10 // (op 3952) 

s_and_b32 s73, s10, 2147483648 // (op 3952) 
s_cmp_eq_i32 s73, 0 // (op 3952) 
s_mov_b32 s74, scc // (op 3952) 

s_cmp_eq_i32 s74, 0 // (op 3952) 
s_cselect_b32 s73, s72, s55 // (op 3952) 


s_mul_i32 s55, s73, s14 // (op 3952) 

s_sub_u32 s73, s72, s55 // (op 3952) 

v_lshl_add_u32 v24, s73, 7, v21 // (op 3952) 


s_ashr_i64 s[74:75], s[2:3], 7 // (op 3952) 
v_mov_b32 v27, 0 // (op 3952) convert
v_mov_b32 v26, v24 // (op 3952) convert

s_mul_i32 s55, s72, s74 // (op 3952) 


s_sub_u32 s72, s54, s55 // (op 3952) 


v_lshl_add_u32 v21, s72, 7, v23 // (op 3952) 


v_mul_lo_u32 v25, v21, s37 // (op 3952) most significant: low of low * high
v_mul_hi_u32 v23, v21, s36 // (op 3952) most significant: high of low * low
v_mul_lo_u32 v24, v21, s36 // (op 3952) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 3952) most significant: sum

v_add_co_u32 v24, s[54:55], v24, v26 // (op 3952) least significant half
v_addc_co_u32 v25, s[54:55], v25, v27, s[54:55] // (op 3952) most significant half



s_mul_i32 s55, s36, s3 // (op 3954) most significant: low of low * high
s_mul_i32 s72, s37, s2 // (op 3954) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 3954) least significant
s_mul_hi_u32 s73, s36, s2 // (op 3954) most significant: high of low * low
s_add_u32 s55, s55, s72 // (op 3954) most significant: sum
s_add_u32 s55, s55, s73 // (op 3954) most significant: sum


s_mov_b32 s72, s38 // (op 3954) reg expression
s_mov_b32 s73, s39 // (op 3954) reg expression
s_mov_b32 s74, s54 // (op 3954) reg expression
s_mov_b32 s75, 131072 // (op 3954) call()

v_and_b32 v21, -8, v1 // (op 3971) 
s_mul_i32 s54, s16, s8 // (op 3971) 
s_add_u32 s54, s54, s0 // (op 3971) 
v_sub_u32 v23, v1, v21 // (op 3971) 

s_mul_hi_u32 s55, s54, s13 // (op 3971) 
v_lshlrev_b32 v21, 4, v23 // (op 3971) 

v_lshrrev_b32 v23, 3, v1 // (op 3971) 
s_sub_u32 s76, s54, s55 // (op 3971) 
s_lshr_b32 s77, s76, 1 // (op 3971) 

s_add_u32 s76, s77, s55 // (op 3971) 


s_lshr_b32 s55, s76, s11 // (op 3971) 

s_and_b32 s76, s11, 2147483648 // (op 3971) 
s_cmp_eq_i32 s76, 0 // (op 3971) 
s_mov_b32 s77, scc // (op 3971) 

s_cmp_eq_i32 s77, 0 // (op 3971) 
s_cselect_b32 s76, s54, s55 // (op 3971) 


s_mul_hi_u32 s55, s76, s12 // (op 3971) 
s_sub_u32 s77, s76, s55 // (op 3971) 
s_lshr_b32 s78, s77, 1 // (op 3971) 

s_add_u32 s77, s78, s55 // (op 3971) 


s_lshr_b32 s55, s77, s10 // (op 3971) 

s_and_b32 s77, s10, 2147483648 // (op 3971) 
s_cmp_eq_i32 s77, 0 // (op 3971) 
s_mov_b32 s78, scc // (op 3971) 

s_cmp_eq_i32 s78, 0 // (op 3971) 
s_cselect_b32 s77, s76, s55 // (op 3971) 


s_mul_i32 s55, s77, s14 // (op 3971) 

s_sub_u32 s77, s76, s55 // (op 3971) 

v_lshl_add_u32 v26, s77, 7, v21 // (op 3971) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 3971) 
v_mov_b32 v33, 0 // (op 3971) convert
v_mov_b32 v32, v26 // (op 3971) convert

s_mul_i32 s55, s76, s78 // (op 3971) 


s_sub_u32 s76, s54, s55 // (op 3971) 


v_lshl_add_u32 v21, s76, 7, v23 // (op 3971) 


v_mul_lo_u32 v27, v21, s37 // (op 3971) most significant: low of low * high
v_mul_hi_u32 v23, v21, s36 // (op 3971) most significant: high of low * low
v_mul_lo_u32 v26, v21, s36 // (op 3971) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 3971) most significant: sum

v_add_co_u32 v26, s[54:55], v26, v32 // (op 3971) least significant half
v_addc_co_u32 v27, s[54:55], v27, v33, s[54:55] // (op 3971) most significant half



s_mul_i32 s55, s36, s3 // (op 3973) most significant: low of low * high
s_mul_i32 s76, s37, s2 // (op 3973) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 3973) least significant
s_mul_hi_u32 s77, s36, s2 // (op 3973) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 3973) most significant: sum
s_add_u32 s55, s55, s77 // (op 3973) most significant: sum


s_mov_b32 s72, s38 // (op 3973) reg expression
s_mov_b32 s73, s39 // (op 3973) reg expression
s_mov_b32 s74, s54 // (op 3973) reg expression
s_mov_b32 s75, 131072 // (op 3973) call()

v_lshrrev_b32 v21, 7, v1 // (op 4080) 
s_mul_i32 s54, s16, s8 // (op 4080) 
s_add_u32 s54, s54, s0 // (op 4080) 
v_and_b32 v23, 15, v1 // (op 4080) 
s_mul_hi_u32 s55, s54, s13 // (op 4080) 
v_lshl_add_u32 v32, v21, 4, v23 // (op 4080) 


s_sub_u32 s76, s54, s55 // (op 4080) 
v_and_b32 v21, 63, v1 // (op 4080) 
s_lshr_b32 s77, s76, 1 // (op 4080) 

v_lshrrev_b32 v23, 4, v21 // (op 4080) 

s_add_u32 s76, s77, s55 // (op 4080) 


s_lshr_b32 s55, s76, s11 // (op 4080) 

s_and_b32 s76, s11, 2147483648 // (op 4080) 
s_cmp_eq_i32 s76, 0 // (op 4080) 
s_mov_b32 s77, scc // (op 4080) 

s_cmp_eq_i32 s77, 0 // (op 4080) 
s_cselect_b32 s76, s54, s55 // (op 4080) 


s_mul_hi_u32 s55, s76, s12 // (op 4080) 
s_sub_u32 s77, s76, s55 // (op 4080) 
s_lshr_b32 s78, s77, 1 // (op 4080) 

s_add_u32 s77, s78, s55 // (op 4080) 


s_lshr_b32 s55, s77, s10 // (op 4080) 

s_and_b32 s77, s10, 2147483648 // (op 4080) 
s_cmp_eq_i32 s77, 0 // (op 4080) 
s_mov_b32 s78, scc // (op 4080) 

s_cmp_eq_i32 s78, 0 // (op 4080) 
s_cselect_b32 s77, s76, s55 // (op 4080) 


v_lshl_add_u32 v21, s77, 5, v32 // (op 4080) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 4080) 
s_mul_i32 s55, s76, s78 // (op 4080) 


s_sub_u32 s76, s54, s55 // (op 4080) 


v_lshl_add_u32 v32, s76, 2, v23 // (op 4080) 


v_mul_lo_u32 v23, v21, s40 // (op 4080) 
v_add_u32 v23, v23, v32 // (op 4080) 


s_mul_i32 s55, s40, s35 // (op 4082) most significant: low of low * high
s_mul_i32 s76, s41, s34 // (op 4082) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 4082) least significant
s_mul_hi_u32 s77, s40, s34 // (op 4082) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 4082) most significant: sum
s_add_u32 s55, s55, s77 // (op 4082) most significant: sum


s_mov_b32 s76, s42 // (op 4082) reg expression
s_mov_b32 s77, s43 // (op 4082) reg expression
s_mov_b32 s78, s54 // (op 4082) reg expression
s_mov_b32 s79, 131072 // (op 4082) call()

v_lshrrev_b32 v21, 7, v1 // (op 4099) 
s_mul_i32 s54, s16, s8 // (op 4099) 
s_add_u32 s54, s54, s0 // (op 4099) 
v_and_b32 v32, 15, v1 // (op 4099) 
s_mul_hi_u32 s55, s54, s13 // (op 4099) 
v_lshl_add_u32 v33, v21, 4, v32 // (op 4099) 


v_and_b32 v21, 63, v1 // (op 4099) 
s_sub_u32 s80, s54, s55 // (op 4099) 
v_lshrrev_b32 v32, 4, v21 // (op 4099) 

s_lshr_b32 s81, s80, 1 // (op 4099) 

s_add_u32 s80, s81, s55 // (op 4099) 


s_lshr_b32 s55, s80, s11 // (op 4099) 

s_and_b32 s80, s11, 2147483648 // (op 4099) 
s_cmp_eq_i32 s80, 0 // (op 4099) 
s_mov_b32 s81, scc // (op 4099) 

s_cmp_eq_i32 s81, 0 // (op 4099) 
s_cselect_b32 s80, s54, s55 // (op 4099) 


s_mul_hi_u32 s55, s80, s12 // (op 4099) 
s_sub_u32 s81, s80, s55 // (op 4099) 
s_lshr_b32 s82, s81, 1 // (op 4099) 

s_add_u32 s81, s82, s55 // (op 4099) 


s_lshr_b32 s55, s81, s10 // (op 4099) 

s_and_b32 s81, s10, 2147483648 // (op 4099) 
s_cmp_eq_i32 s81, 0 // (op 4099) 
s_mov_b32 s82, scc // (op 4099) 

s_cmp_eq_i32 s82, 0 // (op 4099) 
s_cselect_b32 s81, s80, s55 // (op 4099) 


v_lshl_add_u32 v21, s81, 5, v33 // (op 4099) 


s_ashr_i64 s[82:83], s[2:3], 7 // (op 4099) 
s_mul_i32 s55, s80, s82 // (op 4099) 


s_sub_u32 s80, s54, s55 // (op 4099) 


v_lshl_add_u32 v33, s80, 2, v32 // (op 4099) 


v_mul_lo_u32 v32, v21, s40 // (op 4099) 
v_add_u32 v32, v32, v33 // (op 4099) 


s_mul_i32 s55, s40, s35 // (op 4101) most significant: low of low * high
s_mul_i32 s80, s41, s34 // (op 4101) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 4101) least significant
s_mul_hi_u32 s81, s40, s34 // (op 4101) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 4101) most significant: sum
s_add_u32 s55, s55, s81 // (op 4101) most significant: sum


s_mov_b32 s76, s42 // (op 4101) reg expression
s_mov_b32 s77, s43 // (op 4101) reg expression
s_mov_b32 s78, s54 // (op 4101) reg expression
s_mov_b32 s79, 131072 // (op 4101) call()

v_and_b32 v21, -128, v1 // (op 4208) 
s_mul_i32 s54, s16, s8 // (op 4208) 
s_add_u32 s54, s54, s0 // (op 4208) 
v_sub_u32 v33, v1, v21 // (op 4208) 

s_mul_hi_u32 s55, s54, s13 // (op 4208) 
v_lshrrev_b32 v21, 7, v1 // (op 4208) 
s_sub_u32 s80, s54, s55 // (op 4208) 
v_lshlrev_b32 v34, 1, v21 // (op 4208) 

s_lshr_b32 s81, s80, 1 // (op 4208) 

s_add_u32 s80, s81, s55 // (op 4208) 


s_lshr_b32 s55, s80, s11 // (op 4208) 

s_and_b32 s80, s11, 2147483648 // (op 4208) 
s_cmp_eq_i32 s80, 0 // (op 4208) 
s_mov_b32 s81, scc // (op 4208) 

s_cmp_eq_i32 s81, 0 // (op 4208) 
s_cselect_b32 s80, s54, s55 // (op 4208) 


s_mul_hi_u32 s55, s80, s12 // (op 4208) 
s_sub_u32 s81, s80, s55 // (op 4208) 
s_lshr_b32 s82, s81, 1 // (op 4208) 

s_add_u32 s81, s82, s55 // (op 4208) 


s_lshr_b32 s55, s81, s10 // (op 4208) 

s_and_b32 s81, s10, 2147483648 // (op 4208) 
s_cmp_eq_i32 s81, 0 // (op 4208) 
s_mov_b32 s82, scc // (op 4208) 

s_cmp_eq_i32 s82, 0 // (op 4208) 
s_cselect_b32 s81, s80, s55 // (op 4208) 


s_mul_i32 s55, s81, s14 // (op 4208) 

s_sub_u32 s81, s80, s55 // (op 4208) 

v_lshl_add_u32 v21, s81, 7, v33 // (op 4208) 


s_ashr_i64 s[82:83], s[2:3], 7 // (op 4208) 
s_mul_i32 s55, s80, s82 // (op 4208) 


s_sub_u32 s80, s54, s55 // (op 4208) 


v_lshl_add_u32 v33, s80, 2, v34 // (op 4208) 


v_mul_lo_u32 v34, v33, s44 // (op 4208) 
v_add_u32 v34, v34, v21 // (op 4208) 


s_mul_i32 s55, s44, s49 // (op 4210) most significant: low of low * high
s_mul_i32 s80, s45, s48 // (op 4210) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 4210) least significant
s_mul_hi_u32 s81, s44, s48 // (op 4210) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 4210) most significant: sum
s_add_u32 s55, s55, s81 // (op 4210) most significant: sum


s_mov_b32 s80, s46 // (op 4210) reg expression
s_mov_b32 s81, s47 // (op 4210) reg expression
s_mov_b32 s82, s54 // (op 4210) reg expression
s_mov_b32 s83, 131072 // (op 4210) call()

v_and_b32 v21, -128, v1 // (op 4227) 
s_mul_i32 s54, s16, s8 // (op 4227) 
s_add_u32 s54, s54, s0 // (op 4227) 
v_sub_u32 v33, v1, v21 // (op 4227) 

s_mul_hi_u32 s55, s54, s13 // (op 4227) 
v_lshrrev_b32 v21, 7, v1 // (op 4227) 
v_lshlrev_b32 v35, 1, v21 // (op 4227) 

s_sub_u32 s84, s54, s55 // (op 4227) 
s_lshr_b32 s85, s84, 1 // (op 4227) 

s_add_u32 s84, s85, s55 // (op 4227) 


s_lshr_b32 s55, s84, s11 // (op 4227) 

s_and_b32 s84, s11, 2147483648 // (op 4227) 
s_cmp_eq_i32 s84, 0 // (op 4227) 
s_mov_b32 s85, scc // (op 4227) 

s_cmp_eq_i32 s85, 0 // (op 4227) 
s_cselect_b32 s84, s54, s55 // (op 4227) 


s_mul_hi_u32 s55, s84, s12 // (op 4227) 
s_sub_u32 s85, s84, s55 // (op 4227) 
s_lshr_b32 s86, s85, 1 // (op 4227) 

s_add_u32 s85, s86, s55 // (op 4227) 


s_lshr_b32 s55, s85, s10 // (op 4227) 

s_and_b32 s85, s10, 2147483648 // (op 4227) 
s_cmp_eq_i32 s85, 0 // (op 4227) 
s_mov_b32 s86, scc // (op 4227) 

s_cmp_eq_i32 s86, 0 // (op 4227) 
s_cselect_b32 s85, s84, s55 // (op 4227) 


s_mul_i32 s55, s85, s14 // (op 4227) 

s_sub_u32 s85, s84, s55 // (op 4227) 

v_lshl_add_u32 v21, s85, 7, v33 // (op 4227) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 4227) 
s_mul_i32 s55, s84, s86 // (op 4227) 


s_sub_u32 s84, s54, s55 // (op 4227) 


v_lshl_add_u32 v33, s84, 2, v35 // (op 4227) 


v_mul_lo_u32 v35, v33, s44 // (op 4227) 
v_add_u32 v35, v35, v21 // (op 4227) 


s_mul_i32 s55, s44, s49 // (op 4229) most significant: low of low * high
s_mul_i32 s84, s45, s48 // (op 4229) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 4229) least significant
s_mul_hi_u32 s85, s44, s48 // (op 4229) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 4229) most significant: sum
s_add_u32 s55, s55, s85 // (op 4229) most significant: sum


s_mov_b32 s80, s46 // (op 4229) reg expression
s_mov_b32 s81, s47 // (op 4229) reg expression
s_mov_b32 s82, s54 // (op 4229) reg expression
s_mov_b32 s83, 131072 // (op 4229) call()




buffer_load_ubyte v21, v23, s[76:79], 0 offen offset: 0 // (op 3469) Load value
v_and_b32 v33, -8, v1 // (op 4384) 
v_sub_u32 v36, v1, v33 // (op 4384) 

v_lshlrev_b32 v33, 4, v36 // (op 4384) 

v_lshrrev_b32 v36, 3, v1 // (op 4384) 
v_lshl_add_u32 v37, v36, 7, v33 // (op 4384) 


s_nop 0
v_readfirstlane_b32 s54, v37 // (op 4384) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5131) 

s_lshl_b64 s[84:85], s[36:37], 5 // (op 3698) 
v_mov_b64 v[36:37], s[84:85] // (op 3698) convert


v_mov_b64 v[38:39], v[24:25] // (op 3698) reg expression
s_mov_b32 m0, s54 // (op 3698) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3698) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 3698) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 3698) most significant half

s_add_u32 m0, m0, 4096 // (op 3698) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3698) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 3698) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 3698) most significant half

s_add_u32 m0, m0, 4096 // (op 3698) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3698) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 3698) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 3698) most significant half

s_add_u32 m0, m0, 4096 // (op 3698) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3698) Load value direct to lds




v_mov_b64 v[36:37], s[44:45] // (op 3476) convert

v_mov_b32 v33, v34 // (op 3476) reg expression
buffer_load_ubyte v38, v33, s[80:83], 0 offen offset: 0 // (op 3476) Load value
v_add_u32 v33, v33, v36 // (op 3476) 
buffer_load_ubyte v39, v33, s[80:83], 0 offen offset: 0 // (op 3476) Load value


v_and_b32 v33, -8, v1 // (op 4319) 
v_sub_u32 v36, v1, v33 // (op 4319) 

v_lshlrev_b32 v33, 4, v36 // (op 4319) 

v_lshrrev_b32 v36, 3, v1 // (op 4319) 
v_lshl_add_u32 v37, v36, 7, v33 // (op 4319) 


s_nop 0
v_readfirstlane_b32 s54, v37 // (op 4319) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5112) 


s_nop 0
s_add_u32 s55, 16384, s54 // (op 3703) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s55 // (op 3703) reg expression
buffer_load_dwordx4 v28, s[68:71], 0 offen offset: 0 lds // (op 3703) Load value direct to lds





buffer_load_ubyte v33, v23, s[76:79], 0 offen offset:4 // (op 3483) Load value

v_and_b32 v23, -8, v1 // (op 4973) 
v_sub_u32 v36, v1, v23 // (op 4973) 

v_lshlrev_b32 v23, 4, v36 // (op 4973) 

v_lshrrev_b32 v36, 3, v1 // (op 4973) 
v_lshl_add_u32 v37, v36, 7, v23 // (op 4973) 


s_nop 0
v_readfirstlane_b32 s54, v37 // (op 4973) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5261) 

s_lshl_b64 s[84:85], s[36:37], 5 // (op 3708) 
v_mov_b64 v[36:37], s[84:85] // (op 3708) convert


s_lshl_b64 s[84:85], s[36:37], 7 // (op 3708) 
v_mov_b32 v23, s84 // (op 3708) 
v_mov_b32 v40, s85 // (op 3708) 
v_add_co_u32 v42, s[86:87], v24, v23 // (op 3708) least significant half
v_addc_co_u32 v43, s[86:87], v25, v40, s[86:87] // (op 3708) most significant half




s_add_u32 s55, 20480, s54 // (op 3708) 
s_mov_b32 m0, s55 // (op 3708) reg expression
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 3708) Load value direct to lds
v_add_co_u32 v42, s[84:85], v42, v36 // (op 3708) least significant half
v_addc_co_u32 v43, s[84:85], 0, v43, s[84:85] // (op 3708) most significant half

s_add_u32 m0, m0, 4096 // (op 3708) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 3708) Load value direct to lds
v_add_co_u32 v42, s[84:85], v42, v36 // (op 3708) least significant half
v_addc_co_u32 v43, s[84:85], 0, v43, s[84:85] // (op 3708) most significant half

s_add_u32 m0, m0, 4096 // (op 3708) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 3708) Load value direct to lds
v_add_co_u32 v42, s[84:85], v42, v36 // (op 3708) least significant half
v_addc_co_u32 v43, s[84:85], 0, v43, s[84:85] // (op 3708) most significant half

s_add_u32 m0, m0, 4096 // (op 3708) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 3708) Load value direct to lds






v_mov_b64 v[24:25], s[44:45] // (op 3490) convert

s_lshl_b32 s54, s44, 2 // (op 3490) 
v_add_u32 v23, v34, s54 // (op 3490) 

buffer_load_ubyte v36, v23, s[80:83], 0 offen offset: 0 // (op 3490) Load value
v_add_u32 v23, v23, v24 // (op 3490) 
buffer_load_ubyte v37, v23, s[80:83], 0 offen offset: 0 // (op 3490) Load value



v_and_b32 v34, -8, v1 // (op 4934) 
v_sub_u32 v23, v1, v34 // (op 4934) 

v_lshlrev_b32 v34, 4, v23 // (op 4934) 

v_lshrrev_b32 v23, 3, v1 // (op 4934) 
v_lshl_add_u32 v24, v23, 7, v34 // (op 4934) 


s_nop 0
v_readfirstlane_b32 s54, v24 // (op 4934) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5252) 


s_nop 0
s_add_u32 s55, 36864, s54 // (op 3710) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s55 // (op 3710) reg expression
s_mov_b32 s84, 128 // (op 3710) call()
buffer_load_dwordx4 v28, s[68:71], s84 offen offset: 0 lds // (op 3710) Load value direct to lds




v_and_b32 v34, -128, v1 // (op 4460) 
v_sub_u32 v23, v1, v34 // (op 4460) 

v_lshrrev_b32 v34, 7, v1 // (op 4460) 
v_lshl_add_u32 v24, v34, 8, v23 // (op 4460) 


s_barrier  // (op 5150) 



s_waitcnt vmcnt(10)
ds_write_b8 v24, v38 offset:40960 // (op 3496) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v24, v39 offset:41088 // (op 3496) Store local data 

s_barrier  // (op 5128) 



ds_read_b64_tr_b8 v[40:41], v19 // (op 3514) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v19 offset:1024 // (op 3514) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v19 offset:8192 // (op 3514) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v19 offset:9216 // (op 3514) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5147) 


ds_read_u8 v34, v16 offset:40960 // (op 3534) Load local data 



ds_read_b64_tr_b8 v[48:49], v19 offset:32 // (op 3519) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v19 offset:1056 // (op 3519) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v19 offset:8224 // (op 3519) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v19 offset:9248 // (op 3519) Transpose load local data 


ds_read_u8 v23, v16 offset:40992 // (op 3539) Load local data 



ds_read_b64_tr_b8 v[56:57], v19 offset:64 // (op 3524) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v19 offset:1088 // (op 3524) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v19 offset:8256 // (op 3524) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v19 offset:9280 // (op 3524) Transpose load local data 


ds_read_u8 v24, v16 offset:41024 // (op 3544) Load local data 



ds_read_b64_tr_b8 v[64:65], v19 offset:96 // (op 3529) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v19 offset:1120 // (op 3529) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v19 offset:8288 // (op 3529) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v19 offset:9312 // (op 3529) Transpose load local data 


ds_read_u8 v25, v16 offset:41056 // (op 3549) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 5109) 



ds_read_b128 v[72:75], v17 offset:16384 // (op 3510) Load local data 
ds_read_b128 v[76:79], v17 offset:16448 // (op 3510) Load local data 
v_and_b32 v28, -8, v1 // (op 4291) 
v_sub_u32 v29, v1, v28 // (op 4291) 

v_lshlrev_b32 v28, 4, v29 // (op 4291) 

v_lshrrev_b32 v29, 3, v1 // (op 4291) 
v_lshl_add_u32 v80, v29, 7, v28 // (op 4291) 


s_nop 0
v_readfirstlane_b32 s54, v80 // (op 4291) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v28, -8, v1 // (op 4356) 
v_sub_u32 v29, v1, v28 // (op 4356) 

v_lshlrev_b32 v28, 4, v29 // (op 4356) 

v_lshrrev_b32 v29, 3, v1 // (op 4356) 
v_lshl_add_u32 v80, v29, 7, v28 // (op 4356) 


s_nop 0
v_readfirstlane_b32 s55, v80 // (op 4356) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v28, -128, v1 // (op 4431) 
v_sub_u32 v29, v1, v28 // (op 4431) 

v_lshrrev_b32 v28, 7, v1 // (op 4431) 
v_lshl_add_u32 v80, v28, 8, v29 // (op 4431) 


v_and_b32 v28, -8, v1 // (op 4919) 
v_sub_u32 v29, v1, v28 // (op 4919) 

v_lshlrev_b32 v28, 4, v29 // (op 4919) 

v_lshrrev_b32 v29, 3, v1 // (op 4919) 
v_lshl_add_u32 v81, v29, 7, v28 // (op 4919) 


s_nop 0
v_readfirstlane_b32 s84, v81 // (op 4919) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v28, -8, v1 // (op 4958) 
v_sub_u32 v29, v1, v28 // (op 4958) 

v_lshlrev_b32 v28, 4, v29 // (op 4958) 

v_lshrrev_b32 v29, 3, v1 // (op 4958) 
v_lshl_add_u32 v81, v29, 7, v28 // (op 4958) 


s_nop 0
v_readfirstlane_b32 s85, v81 // (op 4958) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v28, -128, v1 // (op 5002) 
v_sub_u32 v29, v1, v28 // (op 5002) 

v_lshrrev_b32 v28, 7, v1 // (op 5002) 
v_lshl_add_u32 v81, v28, 8, v29 // (op 5002) 


s_mov_b32 s1, 0 // (op 1719) call()
s_and_b32 s86, s7, 4294967294 // (op 1718) 
s_cmp_lt_u32 s1, s86 // (op 1718) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1718 // (op 1718) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1718 if false)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1718:
s_add_u32 s1, s1, 2 // (op 1720) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:79], v[40:47], a[0:3], v21, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 82) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:79], v[48:55], a[4:7], v21, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1442) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:79], v[56:63], a[8:11], v21, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1598) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:79], v[64:71], a[12:15], v21, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1754) 


v_mov_b32 v28, v32 // (op 75) 

buffer_load_ubyte v21, v28, s[76:79], 0 offen offset:8 // (op 75) Load value
s_barrier  // (op 5134) 

s_lshl_b64 s[86:87], s[36:37], 5 // (op 3623) 
v_mov_b64 v[82:83], s[86:87] // (op 3623) convert


v_mov_b64 v[84:85], v[26:27] // (op 3623) 
s_lshl_b64 s[86:87], s[36:37], 7 // (op 3623) 
s_lshl_b64 s[88:89], s[86:87], 1 // (op 3623) 

v_mov_b32 v29, s88 // (op 3623) 
v_mov_b32 v86, s89 // (op 3623) 
v_add_co_u32 v88, s[86:87], v84, v29 // (op 3623) least significant half
v_addc_co_u32 v89, s[86:87], v85, v86, s[86:87] // (op 3623) most significant half




s_mov_b32 m0, s55 // (op 3623) reg expression
buffer_load_dwordx4 v88, s[72:75], 0 offen offset: 0 lds // (op 3623) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v82 // (op 3623) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3623) most significant half

s_add_u32 m0, m0, 4096 // (op 3623) 
buffer_load_dwordx4 v88, s[72:75], 0 offen offset: 0 lds // (op 3623) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v82 // (op 3623) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3623) most significant half

s_add_u32 m0, m0, 4096 // (op 3623) 
buffer_load_dwordx4 v88, s[72:75], 0 offen offset: 0 lds // (op 3623) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v82 // (op 3623) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3623) most significant half

s_add_u32 m0, m0, 4096 // (op 3623) 
buffer_load_dwordx4 v88, s[72:75], 0 offen offset: 0 lds // (op 3623) Load value direct to lds



v_mov_b64 v[82:83], s[44:45] // (op 21) convert

v_mov_b32 v29, v35 // (op 21) 
s_lshl_b32 s86, s44, 3 // (op 21) 
v_add_u32 v86, v29, s86 // (op 21) 

buffer_load_ubyte v38, v86, s[80:83], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v86, v86, v82 // (op 21) 
buffer_load_ubyte v39, v86, s[80:83], 0 offen offset: 0 // (op 21) Load value


s_barrier  // (op 5115) 


v_mov_b64 v[82:83], v[30:31] // (op 3618) 
s_add_u32 s86, 16384, s54 // (op 3618) 

s_mov_b32 m0, s86 // (op 3618) reg expression
s_mov_b32 s87, 256 // (op 3618) call()
buffer_load_dwordx4 v82, s[68:71], s87 offen offset: 0 lds // (op 3618) Load value direct to lds


s_barrier  // (op 5270) 



s_waitcnt vmcnt(10)
ds_write_b8 v81, v36 offset:41472 // (op 1335) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v81, v37 offset:41600 // (op 1335) Store local data 
s_barrier  // (op 5258) 



ds_read_b64_tr_b8 v[40:41], v13 offset:20480 // (op 1328) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v13 offset:21504 // (op 1328) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v13 offset:28672 // (op 1328) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v13 offset:29696 // (op 1328) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5267) 


ds_read_u8 v34, v12 offset:41472 // (op 1332) Load local data 



ds_read_b64_tr_b8 v[48:49], v13 offset:20512 // (op 1447) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v13 offset:21536 // (op 1447) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v13 offset:28704 // (op 1447) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v13 offset:29728 // (op 1447) Transpose load local data 


ds_read_u8 v23, v12 offset:41504 // (op 1450) Load local data 



ds_read_b64_tr_b8 v[56:57], v13 offset:20544 // (op 1603) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v13 offset:21568 // (op 1603) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v13 offset:28736 // (op 1603) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v13 offset:29760 // (op 1603) Transpose load local data 


ds_read_u8 v24, v12 offset:41536 // (op 1606) Load local data 



ds_read_b64_tr_b8 v[64:65], v13 offset:20576 // (op 1759) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v13 offset:21600 // (op 1759) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v13 offset:28768 // (op 1759) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v13 offset:29792 // (op 1759) Transpose load local data 


ds_read_u8 v25, v12 offset:41568 // (op 1762) Load local data 
s_barrier  // (op 5445) 
s_waitcnt vmcnt(8)
s_barrier  // (op 5249) 



ds_read_b128 v[72:75], v6 offset:36864 // (op 1326) Load local data 
ds_read_b128 v[76:79], v6 offset:36928 // (op 1326) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:79], v[40:47], a[0:3], v33, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1333) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:79], v[48:55], a[4:7], v33, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1443) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:79], v[56:63], a[8:11], v33, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1599) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:79], v[64:71], a[12:15], v33, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1755) 



buffer_load_ubyte v33, v28, s[76:79], 0 offen offset:12 // (op 1330) Load value

s_barrier  // (op 5264) 
v_add_u32 v32, 8, v32 // (op 4091) 

s_lshl_b64 s[86:87], s[36:37], 5 // (op 3679) 
v_mov_b64 v[86:87], s[86:87] // (op 3679) convert


s_lshl_b64 s[86:87], s[36:37], 7 // (op 3679) 
s_mul_i32 s89, 3, s87 // (op 3679) most significant: low of low * high
s_mul_i32 s88, 3, s86 // (op 3679) least significant
s_mul_hi_u32 s90, 3, s86 // (op 3679) most significant: high of low * low
s_add_u32 s89, s89, s90 // (op 3679) most significant: sum


v_mov_b32 v28, s88 // (op 3679) 
v_mov_b32 v88, s89 // (op 3679) 
v_add_co_u32 v90, s[86:87], v84, v28 // (op 3679) least significant half
v_addc_co_u32 v91, s[86:87], v85, v88, s[86:87] // (op 3679) most significant half




s_add_u32 s86, 20480, s85 // (op 3679) 
s_nop 0
s_mov_b32 m0, s86 // (op 3679) reg expression
buffer_load_dwordx4 v90, s[72:75], 0 offen offset: 0 lds // (op 3679) Load value direct to lds
v_add_co_u32 v90, s[88:89], v90, v86 // (op 3679) least significant half
v_addc_co_u32 v91, s[88:89], 0, v91, s[88:89] // (op 3679) most significant half

s_add_u32 m0, m0, 4096 // (op 3679) 
buffer_load_dwordx4 v90, s[72:75], 0 offen offset: 0 lds // (op 3679) Load value direct to lds
v_add_co_u32 v90, s[88:89], v90, v86 // (op 3679) least significant half
v_addc_co_u32 v91, s[88:89], 0, v91, s[88:89] // (op 3679) most significant half

s_add_u32 m0, m0, 4096 // (op 3679) 
buffer_load_dwordx4 v90, s[72:75], 0 offen offset: 0 lds // (op 3679) Load value direct to lds
v_add_co_u32 v90, s[88:89], v90, v86 // (op 3679) least significant half
v_addc_co_u32 v91, s[88:89], 0, v91, s[88:89] // (op 3679) most significant half

s_add_u32 m0, m0, 4096 // (op 3679) 
buffer_load_dwordx4 v90, s[72:75], 0 offen offset: 0 lds // (op 3679) Load value direct to lds





v_mov_b64 v[84:85], s[44:45] // (op 1337) convert

s_lshl_b32 s86, s44, 2 // (op 1337) 
s_mul_i32 s87, 3, s86 // (op 1337) 

v_add_u32 v28, v29, s87 // (op 1337) 

buffer_load_ubyte v36, v28, s[80:83], 0 offen offset: 0 // (op 1337) Load value
v_add_u32 v28, v28, v84 // (op 1337) 
s_lshl_b64 s[86:87], s[36:37], 7 // (op 3963) 
s_lshl_b64 s[88:89], s[86:87], 1 // (op 3963) 

v_mov_b32 v86, s88 // (op 3963) 
v_mov_b32 v87, s89 // (op 3963) 
v_add_co_u32 v26, s[86:87], v26, v86 // (op 3963) least significant half
v_addc_co_u32 v27, s[86:87], v27, v87, s[86:87] // (op 3963) most significant half




buffer_load_ubyte v37, v28, s[80:83], 0 offen offset: 0 // (op 1337) Load value



s_barrier  // (op 5255) 
s_lshl_b32 s86, s44, 3 // (op 4219) 


v_add_u32 v35, v35, s86 // (op 4219) 

s_add_u32 s86, 36864, s84 // (op 3674) 

s_mov_b32 m0, s86 // (op 3674) reg expression
s_mov_b32 s87, 384 // (op 3674) call()
buffer_load_dwordx4 v82, s[68:71], s87 offen offset: 0 lds // (op 3674) Load value direct to lds



s_barrier  // (op 5153) 
v_add_co_u32 v30, vcc, 256, v30 // (op 3835) least significant half



v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 3835) most significant half
s_waitcnt vmcnt(10)
ds_write_b8 v80, v38 offset:40960 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v80, v39 offset:41088 // (op 26) Store local data 
s_barrier  // (op 5140) 



ds_read_b64_tr_b8 v[40:41], v7 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:1024 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:8192 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:9216 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5160) 


ds_read_u8 v34, v0 offset:40960 // (op 79) Load local data 



ds_read_b64_tr_b8 v[48:49], v7 offset:32 // (op 1438) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:1056 // (op 1438) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v7 offset:8224 // (op 1438) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v7 offset:9248 // (op 1438) Transpose load local data 


ds_read_u8 v23, v0 offset:40992 // (op 1441) Load local data 



ds_read_b64_tr_b8 v[56:57], v7 offset:64 // (op 1594) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v7 offset:1088 // (op 1594) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v7 offset:8256 // (op 1594) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v7 offset:9280 // (op 1594) Transpose load local data 


ds_read_u8 v24, v0 offset:41024 // (op 1597) Load local data 



ds_read_b64_tr_b8 v[64:65], v7 offset:96 // (op 1750) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v7 offset:1120 // (op 1750) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v7 offset:8288 // (op 1750) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v7 offset:9312 // (op 1750) Transpose load local data 


ds_read_u8 v25, v0 offset:41056 // (op 1753) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 5121) 



ds_read_b128 v[72:75], v3 offset:16384 // (op 65) Load local data 
ds_read_b128 v[76:79], v3 offset:16448 // (op 65) Load local data 
s_and_b32 s86, s7, 4294967294 // (op 1718) 
s_cmp_lt_u32 s1, s86 // (op 1718) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1718 // (op 1718) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_1718 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_1718:














s_barrier  // (op 5429) 
s_and_b32 s54, s7, 4294967294 // (op 3747) 
v_lshrrev_b32 v21, 3, v1 // (op 3747) 
s_add_u32 s55, s0, s54 // (op 3747) 

v_and_b32 v26, -8, v1 // (op 3747) 
v_sub_u32 v27, v1, v26 // (op 3747) 

s_mul_i32 s54, s16, s8 // (op 3747) 
s_add_u32 s54, s54, s55 // (op 3747) 

v_lshlrev_b32 v26, 4, v27 // (op 3747) 

s_mul_hi_u32 s55, s54, s13 // (op 3747) 
s_sub_u32 s84, s54, s55 // (op 3747) 
s_lshr_b32 s85, s84, 1 // (op 3747) 

s_add_u32 s84, s85, s55 // (op 3747) 


s_lshr_b32 s55, s84, s11 // (op 3747) 

s_and_b32 s84, s11, 2147483648 // (op 3747) 
s_cmp_eq_i32 s84, 0 // (op 3747) 
s_mov_b32 s85, scc // (op 3747) 

s_cmp_eq_i32 s85, 0 // (op 3747) 
s_cselect_b32 s84, s54, s55 // (op 3747) 


s_mul_hi_u32 s55, s84, s12 // (op 3747) 
s_sub_u32 s85, s84, s55 // (op 3747) 
s_lshr_b32 s86, s85, 1 // (op 3747) 

s_add_u32 s85, s86, s55 // (op 3747) 


s_lshr_b32 s55, s85, s10 // (op 3747) 

s_and_b32 s85, s10, 2147483648 // (op 3747) 
s_cmp_eq_i32 s85, 0 // (op 3747) 
s_mov_b32 s86, scc // (op 3747) 

s_cmp_eq_i32 s86, 0 // (op 3747) 
s_cselect_b32 s85, s84, s55 // (op 3747) 


v_lshl_add_u32 v27, s85, 5, v21 // (op 3747) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 3747) 
s_mul_i32 s55, s84, s86 // (op 3747) 


s_sub_u32 s84, s54, s55 // (op 3747) 


v_lshl_add_u32 v21, s84, 7, v26 // (op 3747) 


v_mov_b32 v35, 0 // (op 3747) convert
v_mov_b32 v34, v21 // (op 3747) convert

v_mul_lo_u32 v29, v27, s31 // (op 3747) most significant: low of low * high
v_mul_hi_u32 v21, v27, s30 // (op 3747) most significant: high of low * low
v_mul_lo_u32 v28, v27, s30 // (op 3747) least significant: low of low * low
v_add_u32 v29, v29, v21 // (op 3747) most significant: sum

v_add_co_u32 v28, s[54:55], v28, v34 // (op 3747) least significant half
v_addc_co_u32 v29, s[54:55], v29, v35, s[54:55] // (op 3747) most significant half



s_mul_i32 s55, s30, s35 // (op 3749) most significant: low of low * high
s_mul_i32 s84, s31, s34 // (op 3749) most significant: low of high * low
s_mul_i32 s54, s30, s34 // (op 3749) least significant
s_mul_hi_u32 s85, s30, s34 // (op 3749) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 3749) most significant: sum
s_add_u32 s55, s55, s85 // (op 3749) most significant: sum


s_mov_b32 s68, s32 // (op 3749) reg expression
s_mov_b32 s69, s33 // (op 3749) reg expression
s_mov_b32 s70, s54 // (op 3749) reg expression
s_mov_b32 s71, 131072 // (op 3749) call()

s_and_b32 s54, s7, 4294967294 // (op 3887) 
v_and_b32 v21, -8, v1 // (op 3887) 
s_add_u32 s55, s0, s54 // (op 3887) 

v_sub_u32 v26, v1, v21 // (op 3887) 

v_lshlrev_b32 v21, 4, v26 // (op 3887) 

s_mul_i32 s54, s16, s8 // (op 3887) 
s_add_u32 s54, s54, s55 // (op 3887) 

v_lshrrev_b32 v26, 3, v1 // (op 3887) 
s_mul_hi_u32 s55, s54, s13 // (op 3887) 
s_sub_u32 s84, s54, s55 // (op 3887) 
s_lshr_b32 s85, s84, 1 // (op 3887) 

s_add_u32 s84, s85, s55 // (op 3887) 


s_lshr_b32 s55, s84, s11 // (op 3887) 

s_and_b32 s84, s11, 2147483648 // (op 3887) 
s_cmp_eq_i32 s84, 0 // (op 3887) 
s_mov_b32 s85, scc // (op 3887) 

s_cmp_eq_i32 s85, 0 // (op 3887) 
s_cselect_b32 s84, s54, s55 // (op 3887) 


s_mul_hi_u32 s55, s84, s12 // (op 3887) 
s_sub_u32 s85, s84, s55 // (op 3887) 
s_lshr_b32 s86, s85, 1 // (op 3887) 

s_add_u32 s85, s86, s55 // (op 3887) 


s_lshr_b32 s55, s85, s10 // (op 3887) 

s_and_b32 s85, s10, 2147483648 // (op 3887) 
s_cmp_eq_i32 s85, 0 // (op 3887) 
s_mov_b32 s86, scc // (op 3887) 

s_cmp_eq_i32 s86, 0 // (op 3887) 
s_cselect_b32 s85, s84, s55 // (op 3887) 


s_mul_i32 s55, s85, s14 // (op 3887) 

s_sub_u32 s85, s84, s55 // (op 3887) 

v_lshl_add_u32 v27, s85, 7, v21 // (op 3887) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 3887) 
v_mov_b32 v35, 0 // (op 3887) convert
v_mov_b32 v34, v27 // (op 3887) convert

s_mul_i32 s55, s84, s86 // (op 3887) 


s_sub_u32 s84, s54, s55 // (op 3887) 


v_lshl_add_u32 v21, s84, 7, v26 // (op 3887) 


v_mul_lo_u32 v27, v21, s37 // (op 3887) most significant: low of low * high
v_mul_hi_u32 v30, v21, s36 // (op 3887) most significant: high of low * low
v_mul_lo_u32 v26, v21, s36 // (op 3887) least significant: low of low * low
v_add_u32 v27, v27, v30 // (op 3887) most significant: sum

v_add_co_u32 v26, s[54:55], v26, v34 // (op 3887) least significant half
v_addc_co_u32 v27, s[54:55], v27, v35, s[54:55] // (op 3887) most significant half



s_mul_i32 s55, s36, s3 // (op 3889) most significant: low of low * high
s_mul_i32 s84, s37, s2 // (op 3889) most significant: low of high * low
s_mul_i32 s54, s36, s2 // (op 3889) least significant
s_mul_hi_u32 s85, s36, s2 // (op 3889) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 3889) most significant: sum
s_add_u32 s55, s55, s85 // (op 3889) most significant: sum


s_mov_b32 s72, s38 // (op 3889) reg expression
s_mov_b32 s73, s39 // (op 3889) reg expression
s_mov_b32 s74, s54 // (op 3889) reg expression
s_mov_b32 s75, 131072 // (op 3889) call()

s_and_b32 s54, s7, 4294967294 // (op 4015) 
v_lshrrev_b32 v21, 7, v1 // (op 4015) 
s_add_u32 s55, s0, s54 // (op 4015) 

v_and_b32 v30, 15, v1 // (op 4015) 
v_lshl_add_u32 v31, v21, 4, v30 // (op 4015) 


s_mul_i32 s54, s16, s8 // (op 4015) 
s_add_u32 s54, s54, s55 // (op 4015) 

v_and_b32 v21, 63, v1 // (op 4015) 
s_mul_hi_u32 s55, s54, s13 // (op 4015) 
v_lshrrev_b32 v30, 4, v21 // (op 4015) 

s_sub_u32 s84, s54, s55 // (op 4015) 
s_lshr_b32 s85, s84, 1 // (op 4015) 

s_add_u32 s84, s85, s55 // (op 4015) 


s_lshr_b32 s55, s84, s11 // (op 4015) 

s_and_b32 s84, s11, 2147483648 // (op 4015) 
s_cmp_eq_i32 s84, 0 // (op 4015) 
s_mov_b32 s85, scc // (op 4015) 

s_cmp_eq_i32 s85, 0 // (op 4015) 
s_cselect_b32 s84, s54, s55 // (op 4015) 


s_mul_hi_u32 s55, s84, s12 // (op 4015) 
s_sub_u32 s85, s84, s55 // (op 4015) 
s_lshr_b32 s86, s85, 1 // (op 4015) 

s_add_u32 s85, s86, s55 // (op 4015) 


s_lshr_b32 s55, s85, s10 // (op 4015) 

s_and_b32 s85, s10, 2147483648 // (op 4015) 
s_cmp_eq_i32 s85, 0 // (op 4015) 
s_mov_b32 s86, scc // (op 4015) 

s_cmp_eq_i32 s86, 0 // (op 4015) 
s_cselect_b32 s85, s84, s55 // (op 4015) 


v_lshl_add_u32 v21, s85, 5, v31 // (op 4015) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 4015) 
s_mul_i32 s55, s84, s86 // (op 4015) 


s_sub_u32 s84, s54, s55 // (op 4015) 


v_lshl_add_u32 v31, s84, 2, v30 // (op 4015) 


v_mul_lo_u32 v30, v21, s40 // (op 4015) 
v_add_u32 v30, v30, v31 // (op 4015) 


s_mul_i32 s55, s40, s35 // (op 4017) most significant: low of low * high
s_mul_i32 s84, s41, s34 // (op 4017) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 4017) least significant
s_mul_hi_u32 s85, s40, s34 // (op 4017) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 4017) most significant: sum
s_add_u32 s55, s55, s85 // (op 4017) most significant: sum


s_mov_b32 s76, s42 // (op 4017) reg expression
s_mov_b32 s77, s43 // (op 4017) reg expression
s_mov_b32 s78, s54 // (op 4017) reg expression
s_mov_b32 s79, 131072 // (op 4017) call()

s_and_b32 s54, s7, 4294967294 // (op 4143) 
v_and_b32 v21, -128, v1 // (op 4143) 
s_add_u32 s55, s0, s54 // (op 4143) 

v_sub_u32 v31, v1, v21 // (op 4143) 

v_lshrrev_b32 v21, 7, v1 // (op 4143) 
s_mul_i32 s54, s16, s8 // (op 4143) 
s_add_u32 s54, s54, s55 // (op 4143) 

v_lshlrev_b32 v32, 1, v21 // (op 4143) 

s_mul_hi_u32 s55, s54, s13 // (op 4143) 
s_sub_u32 s84, s54, s55 // (op 4143) 
s_lshr_b32 s85, s84, 1 // (op 4143) 

s_add_u32 s84, s85, s55 // (op 4143) 


s_lshr_b32 s55, s84, s11 // (op 4143) 

s_and_b32 s84, s11, 2147483648 // (op 4143) 
s_cmp_eq_i32 s84, 0 // (op 4143) 
s_mov_b32 s85, scc // (op 4143) 

s_cmp_eq_i32 s85, 0 // (op 4143) 
s_cselect_b32 s84, s54, s55 // (op 4143) 


s_mul_hi_u32 s55, s84, s12 // (op 4143) 
s_sub_u32 s85, s84, s55 // (op 4143) 
s_lshr_b32 s86, s85, 1 // (op 4143) 

s_add_u32 s85, s86, s55 // (op 4143) 


s_lshr_b32 s55, s85, s10 // (op 4143) 

s_and_b32 s85, s10, 2147483648 // (op 4143) 
s_cmp_eq_i32 s85, 0 // (op 4143) 
s_mov_b32 s86, scc // (op 4143) 

s_cmp_eq_i32 s86, 0 // (op 4143) 
s_cselect_b32 s85, s84, s55 // (op 4143) 


s_mul_i32 s55, s85, s14 // (op 4143) 

s_sub_u32 s85, s84, s55 // (op 4143) 

v_lshl_add_u32 v21, s85, 7, v31 // (op 4143) 


s_ashr_i64 s[86:87], s[2:3], 7 // (op 4143) 
s_mul_i32 s55, s84, s86 // (op 4143) 


s_sub_u32 s84, s54, s55 // (op 4143) 


v_lshl_add_u32 v31, s84, 2, v32 // (op 4143) 


v_mul_lo_u32 v32, v31, s44 // (op 4143) 
v_add_u32 v32, v32, v21 // (op 4143) 


s_mul_i32 s55, s44, s49 // (op 4145) most significant: low of low * high
s_mul_i32 s84, s45, s48 // (op 4145) most significant: low of high * low
s_mul_i32 s54, s44, s48 // (op 4145) least significant
s_mul_hi_u32 s85, s44, s48 // (op 4145) most significant: high of low * low
s_add_u32 s55, s55, s84 // (op 4145) most significant: sum
s_add_u32 s55, s55, s85 // (op 4145) most significant: sum


s_mov_b32 s80, s46 // (op 4145) reg expression
s_mov_b32 s81, s47 // (op 4145) reg expression
s_mov_b32 s82, s54 // (op 4145) reg expression
s_mov_b32 s83, 131072 // (op 4145) call()

v_and_b32 v21, -8, v1 // (op 4304) 
v_sub_u32 v31, v1, v21 // (op 4304) 

v_lshlrev_b32 v21, 4, v31 // (op 4304) 

v_lshrrev_b32 v31, 3, v1 // (op 4304) 
v_lshl_add_u32 v34, v31, 7, v21 // (op 4304) 


s_nop 0
v_readfirstlane_b32 s54, v34 // (op 4304) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 4369) 
v_sub_u32 v31, v1, v21 // (op 4369) 

v_lshlrev_b32 v21, 4, v31 // (op 4369) 

v_lshrrev_b32 v31, 3, v1 // (op 4369) 
v_lshl_add_u32 v34, v31, 7, v21 // (op 4369) 


s_nop 0
v_readfirstlane_b32 s55, v34 // (op 4369) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -128, v1 // (op 4444) 
v_sub_u32 v31, v1, v21 // (op 4444) 

v_lshrrev_b32 v21, 7, v1 // (op 4444) 
v_lshl_add_u32 v34, v21, 8, v31 // (op 4444) 


s_and_b32 s84, s7, 4294967294 // (op 1724) 
s_cmp_lt_u32 s84, s7 // (op 1723) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1723 // (op 1723) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1723 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1723:
s_barrier  // (op 5118) 
s_add_u32 s84, s84, 1 // (op 1725) 


v_mov_b64 v[24:25], v[28:29] // (op 3661) 
s_add_u32 s85, 16384, s54 // (op 3661) 

s_mov_b32 m0, s85 // (op 3661) reg expression
buffer_load_dwordx4 v24, s[68:71], 0 offen offset: 0 lds // (op 3661) Load value direct to lds


v_add_co_u32 v28, vcc, 128, v28 // (op 3737) least significant half
v_addc_co_u32 v29, vcc, 0, v29, vcc // (op 3737) most significant half
s_barrier  // (op 5137) 

s_lshl_b64 s[86:87], s[36:37], 5 // (op 3667) 
v_mov_b64 v[24:25], s[86:87] // (op 3667) convert


v_mov_b64 v[36:37], v[26:27] // (op 3667) 
v_mov_b64 v[38:39], v[36:37] // (op 3667) reg expression
s_mov_b32 m0, s55 // (op 3667) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3667) Load value direct to lds
v_add_co_u32 v38, s[86:87], v38, v24 // (op 3667) least significant half
v_addc_co_u32 v39, s[86:87], 0, v39, s[86:87] // (op 3667) most significant half

s_add_u32 m0, m0, 4096 // (op 3667) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3667) Load value direct to lds
v_add_co_u32 v38, s[86:87], v38, v24 // (op 3667) least significant half
v_addc_co_u32 v39, s[86:87], 0, v39, s[86:87] // (op 3667) most significant half

s_add_u32 m0, m0, 4096 // (op 3667) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3667) Load value direct to lds
v_add_co_u32 v38, s[86:87], v38, v24 // (op 3667) least significant half
v_addc_co_u32 v39, s[86:87], 0, v39, s[86:87] // (op 3667) most significant half

s_add_u32 m0, m0, 4096 // (op 3667) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 3667) Load value direct to lds



s_lshl_b64 s[86:87], s[36:37], 7 // (op 3879) 
v_mov_b32 v21, s86 // (op 3879) 
v_mov_b32 v31, s87 // (op 3879) 
v_add_co_u32 v26, s[88:89], v26, v21 // (op 3879) least significant half
v_addc_co_u32 v27, s[88:89], v27, v31, s[88:89] // (op 3879) most significant half





v_mov_b64 v[24:25], s[44:45] // (op 1279) convert

v_mov_b32 v21, v32 // (op 1279) 
v_mov_b32 v23, v21 // (op 1279) reg expression
buffer_load_ubyte v36, v23, s[80:83], 0 offen offset: 0 // (op 1279) Load value
v_add_u32 v23, v23, v24 // (op 1279) 
buffer_load_ubyte v37, v23, s[80:83], 0 offen offset: 0 // (op 1279) Load value



s_lshl_b32 s85, s44, 2 // (op 4135) 
v_add_u32 v32, v32, s85 // (op 4135) 

s_barrier  // (op 5156) 
s_waitcnt vmcnt(6)
s_barrier  // (op 5124) 






ds_read_b128 v[38:41], v3 offset:16384 // (op 1268) Load local data 
ds_read_b128 v[42:45], v3 offset:16448 // (op 1268) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v34, v36 offset:40960 // (op 1277) Store local data 
s_barrier  // (op 5143) 
s_waitcnt vmcnt(0)
ds_write_b8 v34, v37 offset:41088 // (op 1277) Store local data 




ds_read_b64_tr_b8 v[46:47], v7 // (op 1270) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v7 offset:1024 // (op 1270) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v7 offset:8192 // (op 1270) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v7 offset:9216 // (op 1270) Transpose load local data 


v_mov_b32 v21, v30 // (op 1272) 

buffer_load_ubyte v31, v21, s[76:79], 0 offen offset: 0 // (op 1272) Load value

v_add_u32 v30, 4, v30 // (op 4007) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5163) 


ds_read_u8 v21, v0 offset:40960 // (op 1274) Load local data 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:53], a[0:3], v31, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1275) 
ds_read_b64_tr_b8 v[54:55], v7 offset:32 // (op 1419) Transpose load local data 


ds_read_b64_tr_b8 v[56:57], v7 offset:1056 // (op 1419) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v7 offset:8224 // (op 1419) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v7 offset:9248 // (op 1419) Transpose load local data 


ds_read_u8 v21, v0 offset:40992 // (op 1422) Load local data 



ds_read_b64_tr_b8 v[46:47], v7 offset:64 // (op 1575) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[54:61], a[4:7], v31, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1423) 
ds_read_b64_tr_b8 v[48:49], v7 offset:1088 // (op 1575) Transpose load local data 


ds_read_b64_tr_b8 v[50:51], v7 offset:8256 // (op 1575) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v7 offset:9280 // (op 1575) Transpose load local data 


ds_read_u8 v21, v0 offset:41024 // (op 1578) Load local data 



ds_read_b64_tr_b8 v[54:55], v7 offset:96 // (op 1731) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:1120 // (op 1731) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:45], v[46:53], a[8:11], v31, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1579) 
ds_read_b64_tr_b8 v[58:59], v7 offset:8288 // (op 1731) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v7 offset:9312 // (op 1731) Transpose load local data 


ds_read_u8 v21, v0 offset:41056 // (op 1734) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:45], v[54:61], a[12:15], v31, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1735) 




s_cmp_lt_u32 s84, s7 // (op 1723) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1723 // (op 1723) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_1723 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_1723:








s_barrier  // (op 5433) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 272) 
s_cmp_lg_u32 s18, 0 // (op 298) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_298 // (op 298) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_298



buffer_store_dword a0, v20, s[56:59], 0 offen sc1 // (op 295) Store value
buffer_store_dword a1, v20, s[56:59], 0 offen offset:512 sc1 // (op 295) Store value
buffer_store_dword a2, v20, s[56:59], 0 offen offset:1024 sc1 // (op 295) Store value
buffer_store_dword a3, v20, s[56:59], 0 offen offset:1536 sc1 // (op 295) Store value



buffer_store_dword a4, v20, s[56:59], 0 offen offset:128 sc1 // (op 1958) Store value
buffer_store_dword a5, v20, s[56:59], 0 offen offset:640 sc1 // (op 1958) Store value
buffer_store_dword a6, v20, s[56:59], 0 offen offset:1152 sc1 // (op 1958) Store value
buffer_store_dword a7, v20, s[56:59], 0 offen offset:1664 sc1 // (op 1958) Store value



buffer_store_dword a8, v20, s[56:59], 0 offen offset:256 sc1 // (op 1959) Store value
buffer_store_dword a9, v20, s[56:59], 0 offen offset:768 sc1 // (op 1959) Store value
buffer_store_dword a10, v20, s[56:59], 0 offen offset:1280 sc1 // (op 1959) Store value
buffer_store_dword a11, v20, s[56:59], 0 offen offset:1792 sc1 // (op 1959) Store value



buffer_store_dword a12, v20, s[56:59], 0 offen offset:384 sc1 // (op 1960) Store value
buffer_store_dword a13, v20, s[56:59], 0 offen offset:896 sc1 // (op 1960) Store value
buffer_store_dword a14, v20, s[56:59], 0 offen offset:1408 sc1 // (op 1960) Store value
buffer_store_dword a15, v20, s[56:59], 0 offen offset:1920 sc1 // (op 1960) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 299) 
v_cmp_eq_i32 vcc, v1, 0 // (op 303) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303 // (op 303) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303
s_mov_b32 s54, 1 // (op 301) call()
s_mov_b32 s85, 0 // (op 302) convert
s_mov_b32 s84, s16 // (op 302) convert
s_lshl_b64 s[84:85], s[84:85], 2 // (op 302) 
s_add_u32 s86, s50, s84 // (op 302) Start of Int64 add, locking SCC
s_addc_u32 s87, s51, s85 // (op 302) End of Int64 add, unlocking SCC
s_store_dword s54, s[86:87], 0 glc // (op 302) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303 // (op 303) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Store_Flag_303:
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Store_Flag_303:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_298 // (op 298) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_298
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Send_Tile_298:
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Send_Tile_298:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 314) 
s_mov_b32 s54, scc // (op 314) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 314) 
s_mov_b32 s55, s84 // (op 314) convert

s_sub_u32 s84, s55, 1 // (op 314) 

s_cmp_lt_u32 s19, s84 // (op 314) 
s_mov_b32 s55, scc // (op 314) 

s_and_b32 s84, s54, s55 // (op 314) 
s_cmp_lg_u32 s84, 0 // (op 314) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_314 // (op 314) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_314
s_mov_b32 s54, 0 // (op 280) call()
s_mul_i32 s55, s54, s8 // (op 279) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 279) 
s_mov_b32 s86, s84 // (op 279) convert

s_sub_u32 s84, s86, s19 // (op 279) 

s_sub_u32 s85, s84, 1 // (op 279) 

s_cmp_lt_u32 s55, s85 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 // (op 279) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_279 if false)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_279:
s_add_u32 s55, s16, 1 // (op 4792) 
v_lshrrev_b32 v16, 5, v1 // (op 4792) 
s_add_u32 s84, s55, s54 // (op 4792) 
s_lshl_b32 s84, s84, 5 // (op 4792) 

v_lshlrev_b32 v17, 2, v16 // (op 4792) 

v_add_lshl_u32 v16, s84, v17, 7 // (op 4792) 


v_and_b32 v17, -32, v1 // (op 4792) 
v_sub_u32 v19, v1, v17 // (op 4792) 

v_add_lshl_u32 v17, v16, v19, 2 // (op 4792) 


s_lshl_b32 s55, s15, 14 // (op 4794) 
s_mov_b32 s56, s28 // (op 4794) reg expression
s_mov_b32 s57, s29 // (op 4794) reg expression
s_mov_b32 s58, s55 // (op 4794) reg expression
s_mov_b32 s59, 131072 // (op 4794) call()

RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_318:
s_add_u32 s55, s16, 1 // (op 316) 
s_add_u32 s84, s55, s54 // (op 316) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 316) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 316) 

s_add_u32 s86, s50, s84 // (op 316) Start of Int64 add, locking SCC
s_addc_u32 s87, s51, s85 // (op 316) End of Int64 add, unlocking SCC
s_load_dword s55, s[86:87], 0 glc // (op 316) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s55, 0 // (op 318) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 // (op 318) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_DoWhileTop_Global_sync_spin_loop_318 if true)
s_barrier  // (op 322) 
v_cmp_eq_i32 vcc, v1, 0 // (op 321) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321 // (op 321) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321
s_mov_b32 s55, 0 // (op 319) call()
s_add_u32 s84, s16, 1 // (op 320) 
s_add_u32 s86, s84, s54 // (op 320) Start of Int64 add, locking SCC
s_addc_u32 s87, 0, 0 // (op 320) End of Int64 add, unlocking SCC
s_lshl_b64 s[86:87], s[86:87], 2 // (op 320) 

s_add_u32 s84, s50, s86 // (op 320) Start of Int64 add, locking SCC
s_addc_u32 s85, s51, s87 // (op 320) End of Int64 add, unlocking SCC
s_store_dword s55, s[84:85], 0 glc // (op 320) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321 // (op 321) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_321:
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_321:




buffer_load_dword v24, v17, s[56:59], 0 offen // (op 333) Load value
buffer_load_dword v25, v17, s[56:59], 0 offen offset:512 // (op 333) Load value
buffer_load_dword v26, v17, s[56:59], 0 offen offset:1024 // (op 333) Load value
buffer_load_dword v27, v17, s[56:59], 0 offen offset:1536 // (op 333) Load value
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




buffer_load_dword v24, v17, s[56:59], 0 offen offset:128 // (op 1980) Load value
buffer_load_dword v25, v17, s[56:59], 0 offen offset:640 // (op 1980) Load value
buffer_load_dword v26, v17, s[56:59], 0 offen offset:1152 // (op 1980) Load value
buffer_load_dword v27, v17, s[56:59], 0 offen offset:1664 // (op 1980) Load value
v_accvgpr_read v28, a4 // (op 1981) reg expression
v_accvgpr_read v29, a5 // (op 1981) reg expression
v_accvgpr_read v30, a6 // (op 1981) reg expression
v_accvgpr_read v31, a7 // (op 1981) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 1982) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 1982) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 1982) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 1982) 

v_accvgpr_write a4, v28 // (op 1983) reg expression
v_accvgpr_write a5, v29 // (op 1983) reg expression
v_accvgpr_write a6, v30 // (op 1983) reg expression
v_accvgpr_write a7, v31 // (op 1983) reg expression




buffer_load_dword v24, v17, s[56:59], 0 offen offset:256 // (op 1987) Load value
buffer_load_dword v25, v17, s[56:59], 0 offen offset:768 // (op 1987) Load value
buffer_load_dword v26, v17, s[56:59], 0 offen offset:1280 // (op 1987) Load value
buffer_load_dword v27, v17, s[56:59], 0 offen offset:1792 // (op 1987) Load value
v_accvgpr_read v28, a8 // (op 1988) reg expression
v_accvgpr_read v29, a9 // (op 1988) reg expression
v_accvgpr_read v30, a10 // (op 1988) reg expression
v_accvgpr_read v31, a11 // (op 1988) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 1989) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 1989) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 1989) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 1989) 

v_accvgpr_write a8, v28 // (op 1990) reg expression
v_accvgpr_write a9, v29 // (op 1990) reg expression
v_accvgpr_write a10, v30 // (op 1990) reg expression
v_accvgpr_write a11, v31 // (op 1990) reg expression




buffer_load_dword v24, v17, s[56:59], 0 offen offset:384 // (op 1994) Load value
buffer_load_dword v25, v17, s[56:59], 0 offen offset:896 // (op 1994) Load value
buffer_load_dword v26, v17, s[56:59], 0 offen offset:1408 // (op 1994) Load value
buffer_load_dword v27, v17, s[56:59], 0 offen offset:1920 // (op 1994) Load value
v_accvgpr_read v28, a12 // (op 1995) reg expression
v_accvgpr_read v29, a13 // (op 1995) reg expression
v_accvgpr_read v30, a14 // (op 1995) reg expression
v_accvgpr_read v31, a15 // (op 1995) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 1996) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 1996) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 1996) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 1996) 

v_accvgpr_write a12, v28 // (op 1997) reg expression
v_accvgpr_write a13, v29 // (op 1997) reg expression
v_accvgpr_write a14, v30 // (op 1997) reg expression
v_accvgpr_write a15, v31 // (op 1997) reg expression

s_add_u32 s54, s54, 1 // (op 281) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s55, s54, s8 // (op 279) 
s_ashr_i64 s[84:85], s[2:3], 7 // (op 279) 
s_mov_b32 s86, s84 // (op 279) convert

s_sub_u32 s84, s86, s19 // (op 279) 

s_sub_u32 s85, s84, 1 // (op 279) 

s_cmp_lt_u32 s55, s85 // (op 279) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_279 // (op 279) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_ReceiveTileLoop_279 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_ReceiveTileLoop_279:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_314 // (op 314) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_314
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Receive_Tile_314:
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Receive_Tile_314:

s_cmp_eq_i32 s18, 0 // (op 357) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357 // (op 357) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357
v_cmp_eq_f32 vcc, 0.00000, v14 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v24, a0 // (op 46) 
v_accvgpr_read v25, a1 // (op 46) 
v_accvgpr_read v26, a2 // (op 46) 
v_accvgpr_read v27, a3 // (op 46) 
v_mul_f32 v28, v15, v24 // (op 46) 
v_mul_f32 v29, v15, v25 // (op 46) 
v_mul_f32 v30, v15, v26 // (op 46) 
v_mul_f32 v31, v15, v27 // (op 46) 



v_mov_b32 v16, v22 // (op 52) reg expression

buffer_store_dwordx4 v[28:31], v16, s[64:67], 0 offen // (op 52) Store value


v_accvgpr_read v24, a4 // (op 2025) 
v_accvgpr_read v25, a5 // (op 2025) 
v_accvgpr_read v26, a6 // (op 2025) 
v_accvgpr_read v27, a7 // (op 2025) 
v_mul_f32 v28, v15, v24 // (op 2025) 
v_mul_f32 v29, v15, v25 // (op 2025) 
v_mul_f32 v30, v15, v26 // (op 2025) 
v_mul_f32 v31, v15, v27 // (op 2025) 



s_lshl_b32 s54, s24, 7 // (op 2027) 
v_add_u32 v16, v22, s54 // (op 2027) 


buffer_store_dwordx4 v[28:31], v16, s[64:67], 0 offen // (op 2027) Store value


v_accvgpr_read v24, a8 // (op 2037) 
v_accvgpr_read v25, a9 // (op 2037) 
v_accvgpr_read v26, a10 // (op 2037) 
v_accvgpr_read v27, a11 // (op 2037) 
v_mul_f32 v28, v15, v24 // (op 2037) 
v_mul_f32 v29, v15, v25 // (op 2037) 
v_mul_f32 v30, v15, v26 // (op 2037) 
v_mul_f32 v31, v15, v27 // (op 2037) 



s_lshl_b32 s54, s24, 8 // (op 2039) 
v_add_u32 v16, v22, s54 // (op 2039) 


buffer_store_dwordx4 v[28:31], v16, s[64:67], 0 offen // (op 2039) Store value


v_accvgpr_read v24, a12 // (op 2049) 
v_accvgpr_read v25, a13 // (op 2049) 
v_accvgpr_read v26, a14 // (op 2049) 
v_accvgpr_read v27, a15 // (op 2049) 
v_mul_f32 v28, v15, v24 // (op 2049) 
v_mul_f32 v29, v15, v25 // (op 2049) 
v_mul_f32 v30, v15, v26 // (op 2049) 
v_mul_f32 v31, v15, v27 // (op 2049) 



s_lshl_b32 s54, s24, 7 // (op 2051) 
s_mul_i32 s55, 3, s54 // (op 2051) 

v_add_u32 v16, v22, s55 // (op 2051) 


buffer_store_dwordx4 v[28:31], v16, s[64:67], 0 offen // (op 2051) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123:


v_mov_b32 v16, v18 // (op 135) reg expression

buffer_load_dwordx4 v[24:27], v16, s[60:63], 0 offen // (op 135) Load value

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




v_mov_b32 v23, v22 // (op 134) reg expression

buffer_store_dwordx4 v[24:27], v23, s[64:67], 0 offen // (op 134) Store value




s_lshl_b32 s54, s4, 7 // (op 2083) 
v_add_u32 v16, v18, s54 // (op 2083) 


buffer_load_dwordx4 v[24:27], v16, s[60:63], 0 offen // (op 2083) Load value

v_accvgpr_read v28, a4 // (op 2080) 
v_accvgpr_read v29, a5 // (op 2080) 
v_accvgpr_read v30, a6 // (op 2080) 
v_accvgpr_read v31, a7 // (op 2080) 
v_mul_f32 v32, v15, v28 // (op 2080) 
v_mul_f32 v33, v15, v29 // (op 2080) 
v_mul_f32 v34, v15, v30 // (op 2080) 
v_mul_f32 v35, v15, v31 // (op 2080) 

s_waitcnt vmcnt(0)
v_mul_f32 v28, v14, v24 // (op 2084) 
v_mul_f32 v29, v14, v25 // (op 2084) 
v_mul_f32 v30, v14, v26 // (op 2084) 
v_mul_f32 v31, v14, v27 // (op 2084) 

v_add_f32 v24, v28, v32 // (op 2081) 
v_add_f32 v25, v29, v33 // (op 2081) 
v_add_f32 v26, v30, v34 // (op 2081) 
v_add_f32 v27, v31, v35 // (op 2081) 




s_lshl_b32 s54, s24, 7 // (op 2082) 
v_add_u32 v23, v22, s54 // (op 2082) 


buffer_store_dwordx4 v[24:27], v23, s[64:67], 0 offen // (op 2082) Store value




s_lshl_b32 s54, s4, 8 // (op 2095) 
v_add_u32 v16, v18, s54 // (op 2095) 


buffer_load_dwordx4 v[24:27], v16, s[60:63], 0 offen // (op 2095) Load value

v_accvgpr_read v28, a8 // (op 2092) 
v_accvgpr_read v29, a9 // (op 2092) 
v_accvgpr_read v30, a10 // (op 2092) 
v_accvgpr_read v31, a11 // (op 2092) 
v_mul_f32 v32, v15, v28 // (op 2092) 
v_mul_f32 v33, v15, v29 // (op 2092) 
v_mul_f32 v34, v15, v30 // (op 2092) 
v_mul_f32 v35, v15, v31 // (op 2092) 

s_waitcnt vmcnt(0)
v_mul_f32 v28, v14, v24 // (op 2096) 
v_mul_f32 v29, v14, v25 // (op 2096) 
v_mul_f32 v30, v14, v26 // (op 2096) 
v_mul_f32 v31, v14, v27 // (op 2096) 

v_add_f32 v24, v28, v32 // (op 2093) 
v_add_f32 v25, v29, v33 // (op 2093) 
v_add_f32 v26, v30, v34 // (op 2093) 
v_add_f32 v27, v31, v35 // (op 2093) 




s_lshl_b32 s54, s24, 8 // (op 2094) 
v_add_u32 v23, v22, s54 // (op 2094) 


buffer_store_dwordx4 v[24:27], v23, s[64:67], 0 offen // (op 2094) Store value




s_lshl_b32 s54, s4, 7 // (op 2107) 
s_mul_i32 s55, 3, s54 // (op 2107) 

v_add_u32 v16, v18, s55 // (op 2107) 


buffer_load_dwordx4 v[24:27], v16, s[60:63], 0 offen // (op 2107) Load value

v_accvgpr_read v28, a12 // (op 2104) 
v_accvgpr_read v29, a13 // (op 2104) 
v_accvgpr_read v30, a14 // (op 2104) 
v_accvgpr_read v31, a15 // (op 2104) 
v_mul_f32 v32, v15, v28 // (op 2104) 
v_mul_f32 v33, v15, v29 // (op 2104) 
v_mul_f32 v34, v15, v30 // (op 2104) 
v_mul_f32 v35, v15, v31 // (op 2104) 

s_waitcnt vmcnt(0)
v_mul_f32 v28, v14, v24 // (op 2108) 
v_mul_f32 v29, v14, v25 // (op 2108) 
v_mul_f32 v30, v14, v26 // (op 2108) 
v_mul_f32 v31, v14, v27 // (op 2108) 

v_add_f32 v24, v28, v32 // (op 2105) 
v_add_f32 v25, v29, v33 // (op 2105) 
v_add_f32 v26, v30, v34 // (op 2105) 
v_add_f32 v27, v31, v35 // (op 2105) 




s_lshl_b32 s54, s24, 7 // (op 2106) 
s_mul_i32 s55, 3, s54 // (op 2106) 

v_add_u32 v23, v22, s55 // (op 2106) 


buffer_store_dwordx4 v[24:27], v23, s[64:67], 0 offen // (op 2106) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357 // (op 357) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_357:
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_357:


s_mul_i32 s54, s8, s16 // (op 270) 
s_cmp_gt_u32 s9, s54 // (op 270) 
s_mov_b32 s55, scc // (op 270) 
s_sub_u32 s56, s9, s54 // (op 270) 

s_cmp_eq_i32 s55, 0 // (op 270) 
s_cselect_b32 s54, 0, s56 // (op 270) 


s_cmp_lt_u32 s8, s54 // (op 270) 
s_mov_b32 s55, scc // (op 270) 
s_cmp_eq_i32 s55, 0 // (op 270) 
s_cselect_b32 s56, s54, s8 // (op 270) 


s_cmp_lt_u32 s0, s56 // (op 270) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_270 // (op 270) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_SKStreamTileLoop_270 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_SKStreamTileLoop_270:















s_mov_b32 s0, 0 // (op 411) call()
s_mov_b32 s1, 0 // (op 412) call()
s_mov_b32 s0, 0 // (op 426) call()
s_mul_i32 s15, s52, s16 // (op 425) 
s_cmp_gt_u32 s53, s15 // (op 425) 
s_mov_b32 s6, scc // (op 425) 
s_sub_u32 s7, s53, s15 // (op 425) 

s_cmp_eq_i32 s6, 0 // (op 425) 
s_cselect_b32 s8, 0, s7 // (op 425) 


s_cmp_lt_u32 s52, s8 // (op 425) 
s_mov_b32 s15, scc // (op 425) 
s_cmp_eq_i32 s15, 0 // (op 425) 
s_cselect_b32 s6, s8, s52 // (op 425) 


s_cmp_lt_u32 s0, s6 // (op 425) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 // (op 425) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_425 if false)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_425:
v_lshrrev_b32 v0, 7, v1 // (op 4274) 
s_mul_i32 s15, s16, s52 // (op 4274) 
s_add_u32 s15, s15, s0 // (op 4274) 
v_and_b32 v3, 63, v1 // (op 4274) 
s_add_u32 s6, s15, s9 // (op 4274) 

v_lshrrev_b32 v6, 4, v3 // (op 4274) 
s_mul_hi_u32 s15, s6, s13 // (op 4274) 
v_lshlrev_b32 v7, 2, v6 // (op 4274) 

s_sub_u32 s7, s6, s15 // (op 4274) 
v_lshl_add_u32 v6, v0, 4, v7 // (op 4274) 


s_lshr_b32 s8, s7, 1 // (op 4274) 

v_lshrrev_b32 v0, 2, v3 // (op 4274) 

s_add_u32 s7, s8, s15 // (op 4274) 


v_and_b32 v3, 3, v0 // (op 4274) 

s_lshr_b32 s8, s7, s11 // (op 4274) 

v_and_b32 v0, 3, v1 // (op 4274) 
s_and_b32 s7, s11, 2147483648 // (op 4274) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 4274) 


s_cmp_eq_i32 s7, 0 // (op 4274) 
s_mov_b32 s15, scc // (op 4274) 

v_lshrrev_b32 v0, 6, v1 // (op 4274) 
s_cmp_eq_i32 s15, 0 // (op 4274) 
s_cselect_b32 s7, s6, s8 // (op 4274) 



v_and_b32 v3, 1, v0 // (op 4274) 

s_mul_hi_u32 s6, s7, s12 // (op 4274) 
v_lshl_add_u32 v0, v3, 4, v7 // (op 4274) 


s_sub_u32 s8, s7, s6 // (op 4274) 
s_lshr_b32 s15, s8, 1 // (op 4274) 

s_add_u32 s8, s15, s6 // (op 4274) 


s_lshr_b32 s6, s8, s10 // (op 4274) 

s_and_b32 s8, s10, 2147483648 // (op 4274) 
s_cmp_eq_i32 s8, 0 // (op 4274) 
s_mov_b32 s15, scc // (op 4274) 

s_cmp_eq_i32 s15, 0 // (op 4274) 
s_cselect_b32 s8, s7, s6 // (op 4274) 


s_mul_i32 s6, s8, s14 // (op 4274) 
v_lshl_add_u32 v3, s8, 5, v6 // (op 4274) 


s_sub_u32 s8, s7, s6 // (op 4274) 


v_lshl_add_u32 v6, s8, 7, v0 // (op 4274) 


v_mul_lo_u32 v0, v6, s4 // (op 4274) 

v_add_lshl_u32 v6, v3, v0, 2 // (op 4274) 


s_mul_i32 s19, s4, s23 // (op 4276) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 4276) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 4276) least significant
s_mul_hi_u32 s6, s4, s22 // (op 4276) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4276) most significant: sum
s_add_u32 s19, s19, s6 // (op 4276) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 4276) 

s_mov_b32 s60, s20 // (op 4276) reg expression
s_mov_b32 s61, s21 // (op 4276) reg expression
s_mov_b32 s62, s28 // (op 4276) reg expression
s_mov_b32 s63, 131072 // (op 4276) call()

v_lshrrev_b32 v0, 7, v1 // (op 4532) 
s_mul_i32 s15, s16, s52 // (op 4532) 
s_add_u32 s15, s15, s0 // (op 4532) 
v_and_b32 v3, 63, v1 // (op 4532) 
s_add_u32 s6, s15, s9 // (op 4532) 

v_lshrrev_b32 v7, 4, v3 // (op 4532) 
s_mul_hi_u32 s15, s6, s13 // (op 4532) 
v_lshlrev_b32 v12, 2, v7 // (op 4532) 

s_sub_u32 s7, s6, s15 // (op 4532) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 4532) 


s_lshr_b32 s8, s7, 1 // (op 4532) 

v_lshrrev_b32 v0, 2, v3 // (op 4532) 

s_add_u32 s7, s8, s15 // (op 4532) 


v_and_b32 v3, 3, v0 // (op 4532) 

s_lshr_b32 s8, s7, s11 // (op 4532) 

v_and_b32 v0, 3, v1 // (op 4532) 
s_and_b32 s7, s11, 2147483648 // (op 4532) 
v_lshl_add_u32 v12, v3, 2, v0 // (op 4532) 


s_cmp_eq_i32 s7, 0 // (op 4532) 
s_mov_b32 s15, scc // (op 4532) 

v_lshrrev_b32 v0, 6, v1 // (op 4532) 
s_cmp_eq_i32 s15, 0 // (op 4532) 
s_cselect_b32 s7, s6, s8 // (op 4532) 



v_and_b32 v3, 1, v0 // (op 4532) 

s_mul_hi_u32 s6, s7, s12 // (op 4532) 
v_lshl_add_u32 v0, v3, 4, v12 // (op 4532) 


s_sub_u32 s8, s7, s6 // (op 4532) 
s_lshr_b32 s15, s8, 1 // (op 4532) 

s_add_u32 s8, s15, s6 // (op 4532) 


s_lshr_b32 s6, s8, s10 // (op 4532) 

s_and_b32 s8, s10, 2147483648 // (op 4532) 
s_cmp_eq_i32 s8, 0 // (op 4532) 
s_mov_b32 s15, scc // (op 4532) 

s_cmp_eq_i32 s15, 0 // (op 4532) 
s_cselect_b32 s8, s7, s6 // (op 4532) 


s_mul_i32 s6, s8, s14 // (op 4532) 
v_lshl_add_u32 v3, s8, 5, v7 // (op 4532) 


s_sub_u32 s8, s7, s6 // (op 4532) 


v_lshl_add_u32 v7, s8, 7, v0 // (op 4532) 


v_mul_lo_u32 v0, v7, s24 // (op 4532) 

v_add_lshl_u32 v7, v3, v0, 2 // (op 4532) 


s_mul_i32 s19, s24, s23 // (op 4534) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 4534) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 4534) least significant
s_mul_hi_u32 s6, s24, s22 // (op 4534) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4534) most significant: sum
s_add_u32 s19, s19, s6 // (op 4534) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 4534) 

s_mov_b32 s64, s26 // (op 4534) reg expression
s_mov_b32 s65, s27 // (op 4534) reg expression
s_mov_b32 s66, s28 // (op 4534) reg expression
s_mov_b32 s67, 131072 // (op 4534) call()

v_lshrrev_b32 v0, 7, v1 // (op 4601) 
v_and_b32 v3, 15, v1 // (op 4601) 
v_lshl_add_u32 v12, v0, 4, v3 // (op 4601) 


v_and_b32 v0, 48, v1 // (op 4601) 
v_lshl_add_u32 v3, v12, 7, v0 // (op 4601) 


v_and_b32 v0, 48, v1 // (op 4676) 
v_and_b32 v12, 63, v1 // (op 4676) 
v_sub_u32 v13, v12, v0 // (op 4676) 

v_lshrrev_b32 v12, 1, v13 // (op 4676) 
v_add_lshl_u32 v16, v0, v12, 7 // (op 4676) 


v_and_b32 v0, 1, v13 // (op 4676) 

v_lshlrev_b32 v12, 3, v0 // (op 4676) 

v_lshrrev_b32 v0, 6, v1 // (op 4676) 
v_and_b32 v13, 1, v0 // (op 4676) 

v_lshl_add_u32 v0, v13, 4, v12 // (op 4676) 


v_add_u32 v12, v16, v0 // (op 4676) 


v_lshrrev_b32 v0, 6, v1 // (op 4752) 
v_and_b32 v13, 1, v0 // (op 4752) 

v_and_b32 v0, 15, v1 // (op 4752) 
v_lshl_add_u32 v16, v13, 4, v0 // (op 4752) 


v_and_b32 v0, 63, v1 // (op 4752) 
v_lshrrev_b32 v13, 4, v0 // (op 4752) 

v_lshl_add_u32 v0, v13, 7, v16 // (op 4752) 


v_accvgpr_write a0, 0.00000 // (op 170) call()
v_accvgpr_write a1, 0.00000 // (op 170) call()
v_accvgpr_write a2, 0.00000 // (op 170) call()
v_accvgpr_write a3, 0.00000 // (op 170) call()
v_accvgpr_write a4, 0.00000 // (op 578) call()
v_accvgpr_write a5, 0.00000 // (op 578) call()
v_accvgpr_write a6, 0.00000 // (op 578) call()
v_accvgpr_write a7, 0.00000 // (op 578) call()
v_accvgpr_write a8, 0.00000 // (op 734) call()
v_accvgpr_write a9, 0.00000 // (op 734) call()
v_accvgpr_write a10, 0.00000 // (op 734) call()
v_accvgpr_write a11, 0.00000 // (op 734) call()
v_accvgpr_write a12, 0.00000 // (op 890) call()
v_accvgpr_write a13, 0.00000 // (op 890) call()
v_accvgpr_write a14, 0.00000 // (op 890) call()
v_accvgpr_write a15, 0.00000 // (op 890) call()
v_lshrrev_b32 v13, 3, v1 // (op 3776) 
s_mul_i32 s15, s16, s52 // (op 3776) 
s_add_u32 s15, s15, s0 // (op 3776) 
v_and_b32 v16, -8, v1 // (op 3776) 
s_add_u32 s6, s15, s9 // (op 3776) 

v_sub_u32 v17, v1, v16 // (op 3776) 

s_mul_hi_u32 s15, s6, s13 // (op 3776) 
v_lshlrev_b32 v16, 4, v17 // (op 3776) 

s_sub_u32 s7, s6, s15 // (op 3776) 
s_lshr_b32 s8, s7, 1 // (op 3776) 

s_add_u32 s7, s8, s15 // (op 3776) 


s_lshr_b32 s8, s7, s11 // (op 3776) 

s_and_b32 s7, s11, 2147483648 // (op 3776) 
s_cmp_eq_i32 s7, 0 // (op 3776) 
s_mov_b32 s15, scc // (op 3776) 

s_cmp_eq_i32 s15, 0 // (op 3776) 
s_cselect_b32 s7, s6, s8 // (op 3776) 


s_mul_hi_u32 s8, s7, s12 // (op 3776) 
s_sub_u32 s15, s7, s8 // (op 3776) 
s_lshr_b32 s18, s15, 1 // (op 3776) 

s_add_u32 s15, s18, s8 // (op 3776) 


s_lshr_b32 s8, s15, s10 // (op 3776) 

s_and_b32 s15, s10, 2147483648 // (op 3776) 
s_cmp_eq_i32 s15, 0 // (op 3776) 
s_mov_b32 s18, scc // (op 3776) 

s_cmp_eq_i32 s18, 0 // (op 3776) 
s_cselect_b32 s15, s7, s8 // (op 3776) 


v_lshl_add_u32 v17, s15, 5, v13 // (op 3776) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3776) 
s_mul_i32 s8, s7, s18 // (op 3776) 


s_sub_u32 s7, s6, s8 // (op 3776) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3776) 


v_mov_b32 v19, 0 // (op 3776) convert
v_mov_b32 v18, v13 // (op 3776) convert

v_mul_lo_u32 v21, v17, s31 // (op 3776) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 3776) most significant: high of low * low
v_mul_lo_u32 v20, v17, s30 // (op 3776) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 3776) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v18 // (op 3776) least significant half
v_addc_co_u32 v21, s[18:19], v21, v19, s[18:19] // (op 3776) most significant half



s_mul_i32 s19, s30, s35 // (op 3778) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3778) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3778) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3778) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3778) most significant: sum
s_add_u32 s19, s19, s6 // (op 3778) most significant: sum


s_mov_b32 s68, s32 // (op 3778) reg expression
s_mov_b32 s69, s33 // (op 3778) reg expression
s_mov_b32 s70, s18 // (op 3778) reg expression
s_mov_b32 s71, 131072 // (op 3778) call()

v_lshrrev_b32 v13, 3, v1 // (op 3795) 
s_mul_i32 s15, s16, s52 // (op 3795) 
s_add_u32 s15, s15, s0 // (op 3795) 
v_and_b32 v16, -8, v1 // (op 3795) 
s_add_u32 s6, s15, s9 // (op 3795) 

v_sub_u32 v17, v1, v16 // (op 3795) 

s_mul_hi_u32 s15, s6, s13 // (op 3795) 
v_lshlrev_b32 v16, 4, v17 // (op 3795) 

s_sub_u32 s7, s6, s15 // (op 3795) 
s_lshr_b32 s8, s7, 1 // (op 3795) 

s_add_u32 s7, s8, s15 // (op 3795) 


s_lshr_b32 s8, s7, s11 // (op 3795) 

s_and_b32 s7, s11, 2147483648 // (op 3795) 
s_cmp_eq_i32 s7, 0 // (op 3795) 
s_mov_b32 s15, scc // (op 3795) 

s_cmp_eq_i32 s15, 0 // (op 3795) 
s_cselect_b32 s7, s6, s8 // (op 3795) 


s_mul_hi_u32 s8, s7, s12 // (op 3795) 
s_sub_u32 s15, s7, s8 // (op 3795) 
s_lshr_b32 s18, s15, 1 // (op 3795) 

s_add_u32 s15, s18, s8 // (op 3795) 


s_lshr_b32 s8, s15, s10 // (op 3795) 

s_and_b32 s15, s10, 2147483648 // (op 3795) 
s_cmp_eq_i32 s15, 0 // (op 3795) 
s_mov_b32 s18, scc // (op 3795) 

s_cmp_eq_i32 s18, 0 // (op 3795) 
s_cselect_b32 s15, s7, s8 // (op 3795) 


v_lshl_add_u32 v17, s15, 5, v13 // (op 3795) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3795) 
s_mul_i32 s8, s7, s18 // (op 3795) 


s_sub_u32 s7, s6, s8 // (op 3795) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3795) 


v_mov_b32 v19, 0 // (op 3795) convert
v_mov_b32 v18, v13 // (op 3795) convert

v_mul_lo_u32 v23, v17, s31 // (op 3795) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 3795) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 3795) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 3795) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v18 // (op 3795) least significant half
v_addc_co_u32 v23, s[18:19], v23, v19, s[18:19] // (op 3795) most significant half



s_mul_i32 s19, s30, s35 // (op 3797) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3797) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3797) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3797) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3797) most significant: sum
s_add_u32 s19, s19, s6 // (op 3797) most significant: sum


s_mov_b32 s68, s32 // (op 3797) reg expression
s_mov_b32 s69, s33 // (op 3797) reg expression
s_mov_b32 s70, s18 // (op 3797) reg expression
s_mov_b32 s71, 131072 // (op 3797) call()

v_and_b32 v13, -8, v1 // (op 3910) 
s_mul_i32 s15, s16, s52 // (op 3910) 
s_add_u32 s15, s15, s0 // (op 3910) 
v_sub_u32 v16, v1, v13 // (op 3910) 

s_add_u32 s6, s15, s9 // (op 3910) 

v_lshlrev_b32 v13, 4, v16 // (op 3910) 

s_mul_hi_u32 s15, s6, s13 // (op 3910) 
v_lshrrev_b32 v16, 3, v1 // (op 3910) 
s_sub_u32 s7, s6, s15 // (op 3910) 
s_lshr_b32 s8, s7, 1 // (op 3910) 

s_add_u32 s7, s8, s15 // (op 3910) 


s_lshr_b32 s8, s7, s11 // (op 3910) 

s_and_b32 s7, s11, 2147483648 // (op 3910) 
s_cmp_eq_i32 s7, 0 // (op 3910) 
s_mov_b32 s15, scc // (op 3910) 

s_cmp_eq_i32 s15, 0 // (op 3910) 
s_cselect_b32 s7, s6, s8 // (op 3910) 


s_mul_hi_u32 s8, s7, s12 // (op 3910) 
s_sub_u32 s15, s7, s8 // (op 3910) 
s_lshr_b32 s18, s15, 1 // (op 3910) 

s_add_u32 s15, s18, s8 // (op 3910) 


s_lshr_b32 s8, s15, s10 // (op 3910) 

s_and_b32 s15, s10, 2147483648 // (op 3910) 
s_cmp_eq_i32 s15, 0 // (op 3910) 
s_mov_b32 s18, scc // (op 3910) 

s_cmp_eq_i32 s18, 0 // (op 3910) 
s_cselect_b32 s15, s7, s8 // (op 3910) 


s_mul_i32 s8, s15, s14 // (op 3910) 

s_sub_u32 s15, s7, s8 // (op 3910) 

v_lshl_add_u32 v17, s15, 7, v13 // (op 3910) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3910) 
v_mov_b32 v19, 0 // (op 3910) convert
v_mov_b32 v18, v17 // (op 3910) convert

s_mul_i32 s8, s7, s18 // (op 3910) 


s_sub_u32 s7, s6, s8 // (op 3910) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3910) 


v_mul_lo_u32 v17, v13, s37 // (op 3910) most significant: low of low * high
v_mul_hi_u32 v24, v13, s36 // (op 3910) most significant: high of low * low
v_mul_lo_u32 v16, v13, s36 // (op 3910) least significant: low of low * low
v_add_u32 v17, v17, v24 // (op 3910) most significant: sum

v_add_co_u32 v16, s[18:19], v16, v18 // (op 3910) least significant half
v_addc_co_u32 v17, s[18:19], v17, v19, s[18:19] // (op 3910) most significant half



s_mul_i32 s19, s36, s3 // (op 3912) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 3912) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 3912) least significant
s_mul_hi_u32 s6, s36, s2 // (op 3912) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3912) most significant: sum
s_add_u32 s19, s19, s6 // (op 3912) most significant: sum


s_mov_b32 s72, s38 // (op 3912) reg expression
s_mov_b32 s73, s39 // (op 3912) reg expression
s_mov_b32 s74, s18 // (op 3912) reg expression
s_mov_b32 s75, 131072 // (op 3912) call()

v_and_b32 v13, -8, v1 // (op 3929) 
s_mul_i32 s15, s16, s52 // (op 3929) 
s_add_u32 s15, s15, s0 // (op 3929) 
v_sub_u32 v18, v1, v13 // (op 3929) 

s_add_u32 s6, s15, s9 // (op 3929) 

v_lshlrev_b32 v13, 4, v18 // (op 3929) 

s_mul_hi_u32 s15, s6, s13 // (op 3929) 
v_lshrrev_b32 v18, 3, v1 // (op 3929) 
s_sub_u32 s7, s6, s15 // (op 3929) 
s_lshr_b32 s8, s7, 1 // (op 3929) 

s_add_u32 s7, s8, s15 // (op 3929) 


s_lshr_b32 s8, s7, s11 // (op 3929) 

s_and_b32 s7, s11, 2147483648 // (op 3929) 
s_cmp_eq_i32 s7, 0 // (op 3929) 
s_mov_b32 s15, scc // (op 3929) 

s_cmp_eq_i32 s15, 0 // (op 3929) 
s_cselect_b32 s7, s6, s8 // (op 3929) 


s_mul_hi_u32 s8, s7, s12 // (op 3929) 
s_sub_u32 s15, s7, s8 // (op 3929) 
s_lshr_b32 s18, s15, 1 // (op 3929) 

s_add_u32 s15, s18, s8 // (op 3929) 


s_lshr_b32 s8, s15, s10 // (op 3929) 

s_and_b32 s15, s10, 2147483648 // (op 3929) 
s_cmp_eq_i32 s15, 0 // (op 3929) 
s_mov_b32 s18, scc // (op 3929) 

s_cmp_eq_i32 s18, 0 // (op 3929) 
s_cselect_b32 s15, s7, s8 // (op 3929) 


s_mul_i32 s8, s15, s14 // (op 3929) 

s_sub_u32 s15, s7, s8 // (op 3929) 

v_lshl_add_u32 v19, s15, 7, v13 // (op 3929) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3929) 
v_mov_b32 v25, 0 // (op 3929) convert
v_mov_b32 v24, v19 // (op 3929) convert

s_mul_i32 s8, s7, s18 // (op 3929) 


s_sub_u32 s7, s6, s8 // (op 3929) 


v_lshl_add_u32 v13, s7, 7, v18 // (op 3929) 


v_mul_lo_u32 v19, v13, s37 // (op 3929) most significant: low of low * high
v_mul_hi_u32 v26, v13, s36 // (op 3929) most significant: high of low * low
v_mul_lo_u32 v18, v13, s36 // (op 3929) least significant: low of low * low
v_add_u32 v19, v19, v26 // (op 3929) most significant: sum

v_add_co_u32 v18, s[18:19], v18, v24 // (op 3929) least significant half
v_addc_co_u32 v19, s[18:19], v19, v25, s[18:19] // (op 3929) most significant half



s_mul_i32 s19, s36, s3 // (op 3931) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 3931) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 3931) least significant
s_mul_hi_u32 s6, s36, s2 // (op 3931) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3931) most significant: sum
s_add_u32 s19, s19, s6 // (op 3931) most significant: sum


s_mov_b32 s72, s38 // (op 3931) reg expression
s_mov_b32 s73, s39 // (op 3931) reg expression
s_mov_b32 s74, s18 // (op 3931) reg expression
s_mov_b32 s75, 131072 // (op 3931) call()

v_lshrrev_b32 v13, 7, v1 // (op 4038) 
s_mul_i32 s15, s16, s52 // (op 4038) 
s_add_u32 s15, s15, s0 // (op 4038) 
v_and_b32 v24, 15, v1 // (op 4038) 
s_add_u32 s6, s15, s9 // (op 4038) 

v_lshl_add_u32 v25, v13, 4, v24 // (op 4038) 


s_mul_hi_u32 s15, s6, s13 // (op 4038) 
v_and_b32 v13, 63, v1 // (op 4038) 
s_sub_u32 s7, s6, s15 // (op 4038) 
v_lshrrev_b32 v24, 4, v13 // (op 4038) 

s_lshr_b32 s8, s7, 1 // (op 4038) 

s_add_u32 s7, s8, s15 // (op 4038) 


s_lshr_b32 s8, s7, s11 // (op 4038) 

s_and_b32 s7, s11, 2147483648 // (op 4038) 
s_cmp_eq_i32 s7, 0 // (op 4038) 
s_mov_b32 s15, scc // (op 4038) 

s_cmp_eq_i32 s15, 0 // (op 4038) 
s_cselect_b32 s7, s6, s8 // (op 4038) 


s_mul_hi_u32 s8, s7, s12 // (op 4038) 
s_sub_u32 s15, s7, s8 // (op 4038) 
s_lshr_b32 s18, s15, 1 // (op 4038) 

s_add_u32 s15, s18, s8 // (op 4038) 


s_lshr_b32 s8, s15, s10 // (op 4038) 

s_and_b32 s15, s10, 2147483648 // (op 4038) 
s_cmp_eq_i32 s15, 0 // (op 4038) 
s_mov_b32 s18, scc // (op 4038) 

s_cmp_eq_i32 s18, 0 // (op 4038) 
s_cselect_b32 s15, s7, s8 // (op 4038) 


v_lshl_add_u32 v13, s15, 5, v25 // (op 4038) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4038) 
s_mul_i32 s8, s7, s18 // (op 4038) 


s_sub_u32 s7, s6, s8 // (op 4038) 


v_lshl_add_u32 v25, s7, 2, v24 // (op 4038) 


v_mul_lo_u32 v24, v13, s40 // (op 4038) 
v_add_u32 v24, v24, v25 // (op 4038) 


s_mul_i32 s19, s40, s35 // (op 4040) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4040) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4040) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4040) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4040) most significant: sum
s_add_u32 s19, s19, s6 // (op 4040) most significant: sum


s_mov_b32 s76, s42 // (op 4040) reg expression
s_mov_b32 s77, s43 // (op 4040) reg expression
s_mov_b32 s78, s18 // (op 4040) reg expression
s_mov_b32 s79, 131072 // (op 4040) call()

v_lshrrev_b32 v13, 7, v1 // (op 4057) 
s_mul_i32 s15, s16, s52 // (op 4057) 
s_add_u32 s15, s15, s0 // (op 4057) 
v_and_b32 v25, 15, v1 // (op 4057) 
s_add_u32 s6, s15, s9 // (op 4057) 

v_lshl_add_u32 v26, v13, 4, v25 // (op 4057) 


s_mul_hi_u32 s15, s6, s13 // (op 4057) 
v_and_b32 v13, 63, v1 // (op 4057) 
s_sub_u32 s7, s6, s15 // (op 4057) 
v_lshrrev_b32 v25, 4, v13 // (op 4057) 

s_lshr_b32 s8, s7, 1 // (op 4057) 

s_add_u32 s7, s8, s15 // (op 4057) 


s_lshr_b32 s8, s7, s11 // (op 4057) 

s_and_b32 s7, s11, 2147483648 // (op 4057) 
s_cmp_eq_i32 s7, 0 // (op 4057) 
s_mov_b32 s15, scc // (op 4057) 

s_cmp_eq_i32 s15, 0 // (op 4057) 
s_cselect_b32 s7, s6, s8 // (op 4057) 


s_mul_hi_u32 s8, s7, s12 // (op 4057) 
s_sub_u32 s15, s7, s8 // (op 4057) 
s_lshr_b32 s18, s15, 1 // (op 4057) 

s_add_u32 s15, s18, s8 // (op 4057) 


s_lshr_b32 s8, s15, s10 // (op 4057) 

s_and_b32 s15, s10, 2147483648 // (op 4057) 
s_cmp_eq_i32 s15, 0 // (op 4057) 
s_mov_b32 s18, scc // (op 4057) 

s_cmp_eq_i32 s18, 0 // (op 4057) 
s_cselect_b32 s15, s7, s8 // (op 4057) 


v_lshl_add_u32 v13, s15, 5, v26 // (op 4057) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4057) 
s_mul_i32 s8, s7, s18 // (op 4057) 


s_sub_u32 s7, s6, s8 // (op 4057) 


v_lshl_add_u32 v26, s7, 2, v25 // (op 4057) 


v_mul_lo_u32 v25, v13, s40 // (op 4057) 
v_add_u32 v25, v25, v26 // (op 4057) 


s_mul_i32 s19, s40, s35 // (op 4059) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 4059) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 4059) least significant
s_mul_hi_u32 s6, s40, s34 // (op 4059) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4059) most significant: sum
s_add_u32 s19, s19, s6 // (op 4059) most significant: sum


s_mov_b32 s76, s42 // (op 4059) reg expression
s_mov_b32 s77, s43 // (op 4059) reg expression
s_mov_b32 s78, s18 // (op 4059) reg expression
s_mov_b32 s79, 131072 // (op 4059) call()

v_and_b32 v13, -128, v1 // (op 4166) 
s_mul_i32 s15, s16, s52 // (op 4166) 
s_add_u32 s15, s15, s0 // (op 4166) 
v_sub_u32 v26, v1, v13 // (op 4166) 

s_add_u32 s6, s15, s9 // (op 4166) 

v_lshrrev_b32 v13, 7, v1 // (op 4166) 
s_mul_hi_u32 s15, s6, s13 // (op 4166) 
v_lshlrev_b32 v27, 1, v13 // (op 4166) 

s_sub_u32 s7, s6, s15 // (op 4166) 
s_lshr_b32 s8, s7, 1 // (op 4166) 

s_add_u32 s7, s8, s15 // (op 4166) 


s_lshr_b32 s8, s7, s11 // (op 4166) 

s_and_b32 s7, s11, 2147483648 // (op 4166) 
s_cmp_eq_i32 s7, 0 // (op 4166) 
s_mov_b32 s15, scc // (op 4166) 

s_cmp_eq_i32 s15, 0 // (op 4166) 
s_cselect_b32 s7, s6, s8 // (op 4166) 


s_mul_hi_u32 s8, s7, s12 // (op 4166) 
s_sub_u32 s15, s7, s8 // (op 4166) 
s_lshr_b32 s18, s15, 1 // (op 4166) 

s_add_u32 s15, s18, s8 // (op 4166) 


s_lshr_b32 s8, s15, s10 // (op 4166) 

s_and_b32 s15, s10, 2147483648 // (op 4166) 
s_cmp_eq_i32 s15, 0 // (op 4166) 
s_mov_b32 s18, scc // (op 4166) 

s_cmp_eq_i32 s18, 0 // (op 4166) 
s_cselect_b32 s15, s7, s8 // (op 4166) 


s_mul_i32 s8, s15, s14 // (op 4166) 

s_sub_u32 s15, s7, s8 // (op 4166) 

v_lshl_add_u32 v13, s15, 7, v26 // (op 4166) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4166) 
s_mul_i32 s8, s7, s18 // (op 4166) 


s_sub_u32 s7, s6, s8 // (op 4166) 


v_lshl_add_u32 v26, s7, 2, v27 // (op 4166) 


v_mul_lo_u32 v27, v26, s44 // (op 4166) 
v_add_u32 v27, v27, v13 // (op 4166) 


s_mul_i32 s19, s44, s49 // (op 4168) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 4168) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 4168) least significant
s_mul_hi_u32 s6, s44, s48 // (op 4168) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4168) most significant: sum
s_add_u32 s19, s19, s6 // (op 4168) most significant: sum


s_mov_b32 s80, s46 // (op 4168) reg expression
s_mov_b32 s81, s47 // (op 4168) reg expression
s_mov_b32 s82, s18 // (op 4168) reg expression
s_mov_b32 s83, 131072 // (op 4168) call()

v_and_b32 v13, -128, v1 // (op 4185) 
s_mul_i32 s15, s16, s52 // (op 4185) 
s_add_u32 s15, s15, s0 // (op 4185) 
v_sub_u32 v26, v1, v13 // (op 4185) 

s_add_u32 s6, s15, s9 // (op 4185) 

v_lshrrev_b32 v13, 7, v1 // (op 4185) 
s_mul_hi_u32 s15, s6, s13 // (op 4185) 
v_lshlrev_b32 v28, 1, v13 // (op 4185) 

s_sub_u32 s7, s6, s15 // (op 4185) 
s_lshr_b32 s8, s7, 1 // (op 4185) 

s_add_u32 s7, s8, s15 // (op 4185) 


s_lshr_b32 s8, s7, s11 // (op 4185) 

s_and_b32 s7, s11, 2147483648 // (op 4185) 
s_cmp_eq_i32 s7, 0 // (op 4185) 
s_mov_b32 s15, scc // (op 4185) 

s_cmp_eq_i32 s15, 0 // (op 4185) 
s_cselect_b32 s7, s6, s8 // (op 4185) 


s_mul_hi_u32 s8, s7, s12 // (op 4185) 
s_sub_u32 s15, s7, s8 // (op 4185) 
s_lshr_b32 s18, s15, 1 // (op 4185) 

s_add_u32 s15, s18, s8 // (op 4185) 


s_lshr_b32 s8, s15, s10 // (op 4185) 

s_and_b32 s15, s10, 2147483648 // (op 4185) 
s_cmp_eq_i32 s15, 0 // (op 4185) 
s_mov_b32 s18, scc // (op 4185) 

s_cmp_eq_i32 s18, 0 // (op 4185) 
s_cselect_b32 s15, s7, s8 // (op 4185) 


s_mul_i32 s8, s15, s14 // (op 4185) 

s_sub_u32 s15, s7, s8 // (op 4185) 

v_lshl_add_u32 v13, s15, 7, v26 // (op 4185) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4185) 
s_mul_i32 s8, s7, s18 // (op 4185) 


s_sub_u32 s7, s6, s8 // (op 4185) 


v_lshl_add_u32 v26, s7, 2, v28 // (op 4185) 


v_mul_lo_u32 v28, v26, s44 // (op 4185) 
v_add_u32 v28, v28, v13 // (op 4185) 


s_mul_i32 s19, s44, s49 // (op 4187) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 4187) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 4187) least significant
s_mul_hi_u32 s6, s44, s48 // (op 4187) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4187) most significant: sum
s_add_u32 s19, s19, s6 // (op 4187) most significant: sum


s_mov_b32 s80, s46 // (op 4187) reg expression
s_mov_b32 s81, s47 // (op 4187) reg expression
s_mov_b32 s82, s18 // (op 4187) reg expression
s_mov_b32 s83, 131072 // (op 4187) call()




buffer_load_ubyte v13, v24, s[76:79], 0 offen offset: 0 // (op 3310) Load value
v_and_b32 v26, -8, v1 // (op 4642) 
v_sub_u32 v29, v1, v26 // (op 4642) 

v_lshlrev_b32 v26, 4, v29 // (op 4642) 

v_lshrrev_b32 v29, 3, v1 // (op 4642) 
v_lshl_add_u32 v30, v29, 7, v26 // (op 4642) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 4642) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5189) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 3684) 
v_mov_b64 v[30:31], s[18:19] // (op 3684) convert


v_mov_b64 v[32:33], v[16:17] // (op 3684) reg expression
s_mov_b32 m0, s15 // (op 3684) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3684) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 3684) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 3684) most significant half

s_add_u32 m0, m0, 4096 // (op 3684) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3684) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 3684) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 3684) most significant half

s_add_u32 m0, m0, 4096 // (op 3684) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3684) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v30 // (op 3684) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 3684) most significant half

s_add_u32 m0, m0, 4096 // (op 3684) 
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3684) Load value direct to lds




v_mov_b64 v[30:31], s[44:45] // (op 3317) convert

v_mov_b32 v26, v27 // (op 3317) reg expression
buffer_load_ubyte v32, v26, s[80:83], 0 offen offset: 0 // (op 3317) Load value
v_add_u32 v26, v26, v30 // (op 3317) 
buffer_load_ubyte v33, v26, s[80:83], 0 offen offset: 0 // (op 3317) Load value


v_and_b32 v26, -8, v1 // (op 4577) 
v_sub_u32 v29, v1, v26 // (op 4577) 

v_lshlrev_b32 v26, 4, v29 // (op 4577) 

v_lshrrev_b32 v29, 3, v1 // (op 4577) 
v_lshl_add_u32 v30, v29, 7, v26 // (op 4577) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 4577) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5170) 


s_nop 0
s_add_u32 s6, 16384, s15 // (op 3689) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 3689) reg expression
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 3689) Load value direct to lds





buffer_load_ubyte v26, v24, s[76:79], 0 offen offset:4 // (op 3324) Load value

v_and_b32 v24, -8, v1 // (op 4861) 
v_sub_u32 v29, v1, v24 // (op 4861) 

v_lshlrev_b32 v24, 4, v29 // (op 4861) 

v_lshrrev_b32 v29, 3, v1 // (op 4861) 
v_lshl_add_u32 v30, v29, 7, v24 // (op 4861) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 4861) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5237) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 3694) 
v_mov_b64 v[30:31], s[18:19] // (op 3694) convert


s_lshl_b64 s[18:19], s[36:37], 7 // (op 3694) 
v_mov_b32 v24, s18 // (op 3694) 
v_mov_b32 v29, s19 // (op 3694) 
v_add_co_u32 v34, s[28:29], v16, v24 // (op 3694) least significant half
v_addc_co_u32 v35, s[28:29], v17, v29, s[28:29] // (op 3694) most significant half




s_add_u32 s6, 20480, s15 // (op 3694) 
s_mov_b32 m0, s6 // (op 3694) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 3694) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v30 // (op 3694) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 3694) most significant half

s_add_u32 m0, m0, 4096 // (op 3694) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 3694) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v30 // (op 3694) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 3694) most significant half

s_add_u32 m0, m0, 4096 // (op 3694) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 3694) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v30 // (op 3694) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 3694) most significant half

s_add_u32 m0, m0, 4096 // (op 3694) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 3694) Load value direct to lds






v_mov_b64 v[16:17], s[44:45] // (op 3331) convert

s_lshl_b32 s15, s44, 2 // (op 3331) 
v_add_u32 v24, v27, s15 // (op 3331) 

buffer_load_ubyte v30, v24, s[80:83], 0 offen offset: 0 // (op 3331) Load value
v_add_u32 v24, v24, v16 // (op 3331) 
buffer_load_ubyte v31, v24, s[80:83], 0 offen offset: 0 // (op 3331) Load value



v_and_b32 v24, -8, v1 // (op 4822) 
v_sub_u32 v27, v1, v24 // (op 4822) 

v_lshlrev_b32 v24, 4, v27 // (op 4822) 

v_lshrrev_b32 v27, 3, v1 // (op 4822) 
v_lshl_add_u32 v29, v27, 7, v24 // (op 4822) 


s_nop 0
v_readfirstlane_b32 s15, v29 // (op 4822) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5228) 


s_nop 0
s_add_u32 s6, 36864, s15 // (op 3696) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 3696) reg expression
s_mov_b32 s7, 128 // (op 3696) call()
buffer_load_dwordx4 v20, s[68:71], s7 offen offset: 0 lds // (op 3696) Load value direct to lds




v_and_b32 v24, -128, v1 // (op 4718) 
v_sub_u32 v27, v1, v24 // (op 4718) 

v_lshrrev_b32 v24, 7, v1 // (op 4718) 
v_lshl_add_u32 v29, v24, 8, v27 // (op 4718) 


s_barrier  // (op 5208) 



s_waitcnt vmcnt(10)
ds_write_b8 v29, v32 offset:40960 // (op 3337) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v29, v33 offset:41088 // (op 3337) Store local data 

s_barrier  // (op 5186) 



ds_read_b64_tr_b8 v[34:35], v12 // (op 3355) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v12 offset:1024 // (op 3355) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v12 offset:8192 // (op 3355) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v12 offset:9216 // (op 3355) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5205) 


ds_read_u8 v24, v0 offset:40960 // (op 3375) Load local data 



ds_read_b64_tr_b8 v[42:43], v12 offset:32 // (op 3360) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v12 offset:1056 // (op 3360) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v12 offset:8224 // (op 3360) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v12 offset:9248 // (op 3360) Transpose load local data 


ds_read_u8 v27, v0 offset:40992 // (op 3380) Load local data 



ds_read_b64_tr_b8 v[50:51], v12 offset:64 // (op 3365) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v12 offset:1088 // (op 3365) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v12 offset:8256 // (op 3365) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v12 offset:9280 // (op 3365) Transpose load local data 


ds_read_u8 v29, v0 offset:41024 // (op 3385) Load local data 



ds_read_b64_tr_b8 v[58:59], v12 offset:96 // (op 3370) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v12 offset:1120 // (op 3370) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v12 offset:8288 // (op 3370) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v12 offset:9312 // (op 3370) Transpose load local data 


ds_read_u8 v16, v0 offset:41056 // (op 3390) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 5167) 



ds_read_b128 v[66:69], v3 offset:16384 // (op 3351) Load local data 
ds_read_b128 v[70:73], v3 offset:16448 // (op 3351) Load local data 
v_and_b32 v17, -8, v1 // (op 4549) 
v_sub_u32 v20, v1, v17 // (op 4549) 

v_lshlrev_b32 v17, 4, v20 // (op 4549) 

v_lshrrev_b32 v20, 3, v1 // (op 4549) 
v_lshl_add_u32 v21, v20, 7, v17 // (op 4549) 


s_nop 0
v_readfirstlane_b32 s15, v21 // (op 4549) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 4614) 
v_sub_u32 v20, v1, v17 // (op 4614) 

v_lshlrev_b32 v17, 4, v20 // (op 4614) 

v_lshrrev_b32 v20, 3, v1 // (op 4614) 
v_lshl_add_u32 v21, v20, 7, v17 // (op 4614) 


s_nop 0
v_readfirstlane_b32 s6, v21 // (op 4614) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -128, v1 // (op 4689) 
v_sub_u32 v20, v1, v17 // (op 4689) 

v_lshrrev_b32 v17, 7, v1 // (op 4689) 
v_lshl_add_u32 v21, v17, 8, v20 // (op 4689) 


v_and_b32 v17, -8, v1 // (op 4807) 
v_sub_u32 v20, v1, v17 // (op 4807) 

v_lshlrev_b32 v17, 4, v20 // (op 4807) 

v_lshrrev_b32 v20, 3, v1 // (op 4807) 
v_lshl_add_u32 v74, v20, 7, v17 // (op 4807) 


s_nop 0
v_readfirstlane_b32 s7, v74 // (op 4807) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 4846) 
v_sub_u32 v20, v1, v17 // (op 4846) 

v_lshlrev_b32 v17, 4, v20 // (op 4846) 

v_lshrrev_b32 v20, 3, v1 // (op 4846) 
v_lshl_add_u32 v74, v20, 7, v17 // (op 4846) 


s_nop 0
v_readfirstlane_b32 s8, v74 // (op 4846) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -128, v1 // (op 4890) 
v_sub_u32 v20, v1, v17 // (op 4890) 

v_lshrrev_b32 v17, 7, v1 // (op 4890) 
v_lshl_add_u32 v74, v17, 8, v20 // (op 4890) 


s_mov_b32 s1, 0 // (op 892) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 891) 
s_mov_b32 s28, s18 // (op 891) convert

s_and_b32 s29, s28, 4294967294 // (op 891) 

s_cmp_lt_u32 s1, s29 // (op 891) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_891 // (op 891) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_891 if false)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_891:
s_add_u32 s1, s1, 2 // (op 893) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[34:41], a[0:3], v13, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 184) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[42:49], a[4:7], v13, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 615) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[50:57], a[8:11], v13, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 771) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[58:65], a[12:15], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 927) 


v_mov_b32 v17, v25 // (op 181) 

buffer_load_ubyte v13, v17, s[76:79], 0 offen offset:8 // (op 181) Load value
s_barrier  // (op 5192) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 3633) 
v_mov_b64 v[76:77], s[18:19] // (op 3633) convert


v_mov_b64 v[78:79], v[18:19] // (op 3633) 
s_lshl_b64 s[18:19], s[36:37], 7 // (op 3633) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 3633) 

v_mov_b32 v20, s28 // (op 3633) 
v_mov_b32 v75, s29 // (op 3633) 
v_add_co_u32 v80, s[18:19], v78, v20 // (op 3633) least significant half
v_addc_co_u32 v81, s[18:19], v79, v75, s[18:19] // (op 3633) most significant half




s_mov_b32 m0, s6 // (op 3633) reg expression
buffer_load_dwordx4 v80, s[72:75], 0 offen offset: 0 lds // (op 3633) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v76 // (op 3633) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 3633) most significant half

s_add_u32 m0, m0, 4096 // (op 3633) 
buffer_load_dwordx4 v80, s[72:75], 0 offen offset: 0 lds // (op 3633) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v76 // (op 3633) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 3633) most significant half

s_add_u32 m0, m0, 4096 // (op 3633) 
buffer_load_dwordx4 v80, s[72:75], 0 offen offset: 0 lds // (op 3633) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v76 // (op 3633) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 3633) most significant half

s_add_u32 m0, m0, 4096 // (op 3633) 
buffer_load_dwordx4 v80, s[72:75], 0 offen offset: 0 lds // (op 3633) Load value direct to lds



v_mov_b64 v[76:77], s[44:45] // (op 188) convert

v_mov_b32 v20, v28 // (op 188) 
s_lshl_b32 s18, s44, 3 // (op 188) 
v_add_u32 v75, v20, s18 // (op 188) 

buffer_load_ubyte v32, v75, s[80:83], 0 offen offset: 0 // (op 188) Load value
v_add_u32 v75, v75, v76 // (op 188) 
buffer_load_ubyte v33, v75, s[80:83], 0 offen offset: 0 // (op 188) Load value


s_barrier  // (op 5173) 


v_mov_b64 v[76:77], v[22:23] // (op 3628) 
s_add_u32 s18, 16384, s15 // (op 3628) 

s_mov_b32 m0, s18 // (op 3628) reg expression
s_mov_b32 s19, 256 // (op 3628) call()
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 3628) Load value direct to lds


s_barrier  // (op 5246) 



s_waitcnt vmcnt(10)
ds_write_b8 v74, v30 offset:41472 // (op 508) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v74, v31 offset:41600 // (op 508) Store local data 
s_barrier  // (op 5234) 



ds_read_b64_tr_b8 v[34:35], v11 offset:20480 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v11 offset:21504 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v11 offset:28672 // (op 501) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v11 offset:29696 // (op 501) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5243) 


ds_read_u8 v24, v10 offset:41472 // (op 505) Load local data 



ds_read_b64_tr_b8 v[42:43], v11 offset:20512 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v11 offset:21536 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v11 offset:28704 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v11 offset:29728 // (op 620) Transpose load local data 


ds_read_u8 v27, v10 offset:41504 // (op 623) Load local data 



ds_read_b64_tr_b8 v[50:51], v11 offset:20544 // (op 776) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v11 offset:21568 // (op 776) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v11 offset:28736 // (op 776) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v11 offset:29760 // (op 776) Transpose load local data 


ds_read_u8 v29, v10 offset:41536 // (op 779) Load local data 



ds_read_b64_tr_b8 v[58:59], v11 offset:20576 // (op 932) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v11 offset:21600 // (op 932) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v11 offset:28768 // (op 932) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v11 offset:29792 // (op 932) Transpose load local data 


ds_read_u8 v16, v10 offset:41568 // (op 935) Load local data 
s_barrier  // (op 5365) 
s_waitcnt vmcnt(8)
s_barrier  // (op 5225) 



ds_read_b128 v[66:69], v5 offset:36864 // (op 499) Load local data 
ds_read_b128 v[70:73], v5 offset:36928 // (op 499) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[34:41], a[0:3], v26, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 506) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[42:49], a[4:7], v26, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[50:57], a[8:11], v26, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 772) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[58:65], a[12:15], v26, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 928) 



buffer_load_ubyte v26, v17, s[76:79], 0 offen offset:12 // (op 503) Load value

s_barrier  // (op 5240) 
v_add_u32 v25, 8, v25 // (op 4049) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 3656) 
v_mov_b64 v[80:81], s[18:19] // (op 3656) convert


s_lshl_b64 s[18:19], s[36:37], 7 // (op 3656) 
s_mul_i32 s29, 3, s19 // (op 3656) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 3656) least significant
s_mul_hi_u32 s50, 3, s18 // (op 3656) most significant: high of low * low
s_add_u32 s29, s29, s50 // (op 3656) most significant: sum


v_mov_b32 v17, s28 // (op 3656) 
v_mov_b32 v75, s29 // (op 3656) 
v_add_co_u32 v82, s[18:19], v78, v17 // (op 3656) least significant half
v_addc_co_u32 v83, s[18:19], v79, v75, s[18:19] // (op 3656) most significant half




s_add_u32 s18, 20480, s8 // (op 3656) 
s_nop 0
s_mov_b32 m0, s18 // (op 3656) reg expression
buffer_load_dwordx4 v82, s[72:75], 0 offen offset: 0 lds // (op 3656) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v80 // (op 3656) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3656) most significant half

s_add_u32 m0, m0, 4096 // (op 3656) 
buffer_load_dwordx4 v82, s[72:75], 0 offen offset: 0 lds // (op 3656) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v80 // (op 3656) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3656) most significant half

s_add_u32 m0, m0, 4096 // (op 3656) 
buffer_load_dwordx4 v82, s[72:75], 0 offen offset: 0 lds // (op 3656) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v80 // (op 3656) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3656) most significant half

s_add_u32 m0, m0, 4096 // (op 3656) 
buffer_load_dwordx4 v82, s[72:75], 0 offen offset: 0 lds // (op 3656) Load value direct to lds





v_mov_b64 v[78:79], s[44:45] // (op 510) convert

s_lshl_b32 s18, s44, 2 // (op 510) 
s_mul_i32 s19, 3, s18 // (op 510) 

v_add_u32 v17, v20, s19 // (op 510) 

buffer_load_ubyte v30, v17, s[80:83], 0 offen offset: 0 // (op 510) Load value
v_add_u32 v17, v17, v78 // (op 510) 
s_lshl_b64 s[18:19], s[36:37], 7 // (op 3921) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 3921) 

v_mov_b32 v75, s28 // (op 3921) 
v_mov_b32 v80, s29 // (op 3921) 
v_add_co_u32 v18, s[18:19], v18, v75 // (op 3921) least significant half
v_addc_co_u32 v19, s[18:19], v19, v80, s[18:19] // (op 3921) most significant half




buffer_load_ubyte v31, v17, s[80:83], 0 offen offset: 0 // (op 510) Load value



s_barrier  // (op 5231) 
s_lshl_b32 s18, s44, 3 // (op 4177) 


v_add_u32 v28, v28, s18 // (op 4177) 

s_add_u32 s18, 36864, s7 // (op 3651) 

s_mov_b32 m0, s18 // (op 3651) reg expression
s_mov_b32 s19, 384 // (op 3651) call()
buffer_load_dwordx4 v76, s[68:71], s19 offen offset: 0 lds // (op 3651) Load value direct to lds



s_barrier  // (op 5211) 
v_add_co_u32 v22, vcc, 256, v22 // (op 3787) least significant half



v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 3787) most significant half
s_waitcnt vmcnt(10)
ds_write_b8 v21, v32 offset:40960 // (op 186) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v21, v33 offset:41088 // (op 186) Store local data 
s_barrier  // (op 5198) 



ds_read_b64_tr_b8 v[34:35], v9 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v9 offset:1024 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v9 offset:8192 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:9216 // (op 179) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5218) 


ds_read_u8 v24, v8 offset:40960 // (op 183) Load local data 



ds_read_b64_tr_b8 v[42:43], v9 offset:32 // (op 611) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v9 offset:1056 // (op 611) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v9 offset:8224 // (op 611) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v9 offset:9248 // (op 611) Transpose load local data 


ds_read_u8 v27, v8 offset:40992 // (op 614) Load local data 



ds_read_b64_tr_b8 v[50:51], v9 offset:64 // (op 767) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v9 offset:1088 // (op 767) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v9 offset:8256 // (op 767) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v9 offset:9280 // (op 767) Transpose load local data 


ds_read_u8 v29, v8 offset:41024 // (op 770) Load local data 



ds_read_b64_tr_b8 v[58:59], v9 offset:96 // (op 923) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v9 offset:1120 // (op 923) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v9 offset:8288 // (op 923) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v9 offset:9312 // (op 923) Transpose load local data 


ds_read_u8 v16, v8 offset:41056 // (op 926) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 5179) 



ds_read_b128 v[66:69], v4 offset:16384 // (op 177) Load local data 
ds_read_b128 v[70:73], v4 offset:16448 // (op 177) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 891) 
s_mov_b32 s28, s18 // (op 891) convert

s_and_b32 s29, s28, 4294967294 // (op 891) 

s_cmp_lt_u32 s1, s29 // (op 891) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_891 // (op 891) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoop_891 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_891:














s_barrier  // (op 5349) 
v_lshrrev_b32 v13, 3, v1 // (op 3723) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3723) 
v_and_b32 v28, -8, v1 // (op 3723) 
s_and_b32 s15, s18, 4294967294 // (op 3723) 
v_sub_u32 v17, v1, v28 // (op 3723) 

s_add_u32 s6, s0, s15 // (op 3723) 

v_lshlrev_b32 v28, 4, v17 // (op 3723) 

s_mul_i32 s15, s16, s52 // (op 3723) 
s_add_u32 s15, s15, s6 // (op 3723) 

s_add_u32 s6, s15, s9 // (op 3723) 

s_mul_hi_u32 s15, s6, s13 // (op 3723) 
s_sub_u32 s7, s6, s15 // (op 3723) 
s_lshr_b32 s8, s7, 1 // (op 3723) 

s_add_u32 s7, s8, s15 // (op 3723) 


s_lshr_b32 s8, s7, s11 // (op 3723) 

s_and_b32 s7, s11, 2147483648 // (op 3723) 
s_cmp_eq_i32 s7, 0 // (op 3723) 
s_mov_b32 s15, scc // (op 3723) 

s_cmp_eq_i32 s15, 0 // (op 3723) 
s_cselect_b32 s7, s6, s8 // (op 3723) 


s_mul_hi_u32 s8, s7, s12 // (op 3723) 
s_sub_u32 s15, s7, s8 // (op 3723) 
s_lshr_b32 s28, s15, 1 // (op 3723) 

s_add_u32 s15, s28, s8 // (op 3723) 


s_lshr_b32 s8, s15, s10 // (op 3723) 

s_and_b32 s15, s10, 2147483648 // (op 3723) 
s_cmp_eq_i32 s15, 0 // (op 3723) 
s_mov_b32 s28, scc // (op 3723) 

s_cmp_eq_i32 s28, 0 // (op 3723) 
s_cselect_b32 s15, s7, s8 // (op 3723) 


s_mul_i32 s8, s7, s18 // (op 3723) 


v_lshl_add_u32 v17, s15, 5, v13 // (op 3723) 


s_sub_u32 s7, s6, s8 // (op 3723) 


v_lshl_add_u32 v13, s7, 7, v28 // (op 3723) 


v_mov_b32 v19, 0 // (op 3723) convert
v_mov_b32 v18, v13 // (op 3723) convert

v_mul_lo_u32 v21, v17, s31 // (op 3723) most significant: low of low * high
v_mul_hi_u32 v13, v17, s30 // (op 3723) most significant: high of low * low
v_mul_lo_u32 v20, v17, s30 // (op 3723) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 3723) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v18 // (op 3723) least significant half
v_addc_co_u32 v21, s[18:19], v21, v19, s[18:19] // (op 3723) most significant half



s_mul_i32 s19, s30, s35 // (op 3725) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 3725) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 3725) least significant
s_mul_hi_u32 s6, s30, s34 // (op 3725) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3725) most significant: sum
s_add_u32 s19, s19, s6 // (op 3725) most significant: sum


s_mov_b32 s68, s32 // (op 3725) reg expression
s_mov_b32 s69, s33 // (op 3725) reg expression
s_mov_b32 s70, s18 // (op 3725) reg expression
s_mov_b32 s71, 131072 // (op 3725) call()

v_and_b32 v13, -8, v1 // (op 3865) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3865) 
v_sub_u32 v28, v1, v13 // (op 3865) 

s_and_b32 s15, s18, 4294967294 // (op 3865) 
v_lshlrev_b32 v13, 4, v28 // (op 3865) 

s_add_u32 s6, s0, s15 // (op 3865) 

v_lshrrev_b32 v28, 3, v1 // (op 3865) 
s_mul_i32 s15, s16, s52 // (op 3865) 
s_add_u32 s15, s15, s6 // (op 3865) 

s_add_u32 s6, s15, s9 // (op 3865) 

s_mul_hi_u32 s15, s6, s13 // (op 3865) 
s_sub_u32 s7, s6, s15 // (op 3865) 
s_lshr_b32 s8, s7, 1 // (op 3865) 

s_add_u32 s7, s8, s15 // (op 3865) 


s_lshr_b32 s8, s7, s11 // (op 3865) 

s_and_b32 s7, s11, 2147483648 // (op 3865) 
s_cmp_eq_i32 s7, 0 // (op 3865) 
s_mov_b32 s15, scc // (op 3865) 

s_cmp_eq_i32 s15, 0 // (op 3865) 
s_cselect_b32 s7, s6, s8 // (op 3865) 


s_mul_hi_u32 s8, s7, s12 // (op 3865) 
s_sub_u32 s15, s7, s8 // (op 3865) 
s_lshr_b32 s28, s15, 1 // (op 3865) 

s_add_u32 s15, s28, s8 // (op 3865) 


s_lshr_b32 s8, s15, s10 // (op 3865) 

s_and_b32 s15, s10, 2147483648 // (op 3865) 
s_cmp_eq_i32 s15, 0 // (op 3865) 
s_mov_b32 s28, scc // (op 3865) 

s_cmp_eq_i32 s28, 0 // (op 3865) 
s_cselect_b32 s15, s7, s8 // (op 3865) 


s_mul_i32 s8, s15, s14 // (op 3865) 

s_sub_u32 s15, s7, s8 // (op 3865) 

s_mul_i32 s8, s7, s18 // (op 3865) 


v_lshl_add_u32 v17, s15, 7, v13 // (op 3865) 


s_sub_u32 s7, s6, s8 // (op 3865) 


v_mov_b32 v19, 0 // (op 3865) convert
v_mov_b32 v18, v17 // (op 3865) convert

v_lshl_add_u32 v13, s7, 7, v28 // (op 3865) 


v_mul_lo_u32 v23, v13, s37 // (op 3865) most significant: low of low * high
v_mul_hi_u32 v17, v13, s36 // (op 3865) most significant: high of low * low
v_mul_lo_u32 v22, v13, s36 // (op 3865) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 3865) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v18 // (op 3865) least significant half
v_addc_co_u32 v23, s[18:19], v23, v19, s[18:19] // (op 3865) most significant half



s_mul_i32 s19, s36, s3 // (op 3867) most significant: low of low * high
s_mul_i32 s15, s37, s2 // (op 3867) most significant: low of high * low
s_mul_i32 s18, s36, s2 // (op 3867) least significant
s_mul_hi_u32 s6, s36, s2 // (op 3867) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3867) most significant: sum
s_add_u32 s19, s19, s6 // (op 3867) most significant: sum


s_mov_b32 s72, s38 // (op 3867) reg expression
s_mov_b32 s73, s39 // (op 3867) reg expression
s_mov_b32 s74, s18 // (op 3867) reg expression
s_mov_b32 s75, 131072 // (op 3867) call()

v_lshrrev_b32 v13, 7, v1 // (op 3993) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3993) 
v_and_b32 v28, 15, v1 // (op 3993) 
s_and_b32 s15, s18, 4294967294 // (op 3993) 
v_lshl_add_u32 v17, v13, 4, v28 // (op 3993) 


s_add_u32 s6, s0, s15 // (op 3993) 

v_and_b32 v13, 63, v1 // (op 3993) 
v_lshrrev_b32 v28, 4, v13 // (op 3993) 

s_mul_i32 s15, s16, s52 // (op 3993) 
s_add_u32 s15, s15, s6 // (op 3993) 

s_add_u32 s6, s15, s9 // (op 3993) 

s_mul_hi_u32 s15, s6, s13 // (op 3993) 
s_sub_u32 s7, s6, s15 // (op 3993) 
s_lshr_b32 s8, s7, 1 // (op 3993) 

s_add_u32 s7, s8, s15 // (op 3993) 


s_lshr_b32 s8, s7, s11 // (op 3993) 

s_and_b32 s7, s11, 2147483648 // (op 3993) 
s_cmp_eq_i32 s7, 0 // (op 3993) 
s_mov_b32 s15, scc // (op 3993) 

s_cmp_eq_i32 s15, 0 // (op 3993) 
s_cselect_b32 s7, s6, s8 // (op 3993) 


s_mul_hi_u32 s8, s7, s12 // (op 3993) 
s_sub_u32 s15, s7, s8 // (op 3993) 
s_lshr_b32 s28, s15, 1 // (op 3993) 

s_add_u32 s15, s28, s8 // (op 3993) 


s_lshr_b32 s8, s15, s10 // (op 3993) 

s_and_b32 s15, s10, 2147483648 // (op 3993) 
s_cmp_eq_i32 s15, 0 // (op 3993) 
s_mov_b32 s28, scc // (op 3993) 

s_cmp_eq_i32 s28, 0 // (op 3993) 
s_cselect_b32 s15, s7, s8 // (op 3993) 


s_mul_i32 s8, s7, s18 // (op 3993) 


v_lshl_add_u32 v13, s15, 5, v17 // (op 3993) 


s_sub_u32 s7, s6, s8 // (op 3993) 


v_lshl_add_u32 v17, s7, 2, v28 // (op 3993) 


v_mul_lo_u32 v28, v13, s40 // (op 3993) 
v_add_u32 v28, v28, v17 // (op 3993) 


s_mul_i32 s19, s40, s35 // (op 3995) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 3995) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 3995) least significant
s_mul_hi_u32 s6, s40, s34 // (op 3995) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3995) most significant: sum
s_add_u32 s19, s19, s6 // (op 3995) most significant: sum


s_mov_b32 s76, s42 // (op 3995) reg expression
s_mov_b32 s77, s43 // (op 3995) reg expression
s_mov_b32 s78, s18 // (op 3995) reg expression
s_mov_b32 s79, 131072 // (op 3995) call()

v_and_b32 v13, -128, v1 // (op 4121) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4121) 
v_sub_u32 v17, v1, v13 // (op 4121) 

s_and_b32 s15, s18, 4294967294 // (op 4121) 
v_lshrrev_b32 v13, 7, v1 // (op 4121) 
s_add_u32 s6, s0, s15 // (op 4121) 

v_lshlrev_b32 v18, 1, v13 // (op 4121) 

s_mul_i32 s15, s16, s52 // (op 4121) 
s_add_u32 s15, s15, s6 // (op 4121) 

s_add_u32 s6, s15, s9 // (op 4121) 

s_mul_hi_u32 s15, s6, s13 // (op 4121) 
s_sub_u32 s7, s6, s15 // (op 4121) 
s_lshr_b32 s8, s7, 1 // (op 4121) 

s_add_u32 s7, s8, s15 // (op 4121) 


s_lshr_b32 s8, s7, s11 // (op 4121) 

s_and_b32 s7, s11, 2147483648 // (op 4121) 
s_cmp_eq_i32 s7, 0 // (op 4121) 
s_mov_b32 s15, scc // (op 4121) 

s_cmp_eq_i32 s15, 0 // (op 4121) 
s_cselect_b32 s7, s6, s8 // (op 4121) 


s_mul_hi_u32 s8, s7, s12 // (op 4121) 
s_sub_u32 s15, s7, s8 // (op 4121) 
s_lshr_b32 s28, s15, 1 // (op 4121) 

s_add_u32 s15, s28, s8 // (op 4121) 


s_lshr_b32 s8, s15, s10 // (op 4121) 

s_and_b32 s15, s10, 2147483648 // (op 4121) 
s_cmp_eq_i32 s15, 0 // (op 4121) 
s_mov_b32 s28, scc // (op 4121) 

s_cmp_eq_i32 s28, 0 // (op 4121) 
s_cselect_b32 s15, s7, s8 // (op 4121) 


s_mul_i32 s8, s15, s14 // (op 4121) 

s_sub_u32 s15, s7, s8 // (op 4121) 

s_mul_i32 s8, s7, s18 // (op 4121) 


v_lshl_add_u32 v13, s15, 7, v17 // (op 4121) 


s_sub_u32 s7, s6, s8 // (op 4121) 


v_lshl_add_u32 v17, s7, 2, v18 // (op 4121) 


v_mul_lo_u32 v18, v17, s44 // (op 4121) 
v_add_u32 v18, v18, v13 // (op 4121) 


s_mul_i32 s19, s44, s49 // (op 4123) most significant: low of low * high
s_mul_i32 s15, s45, s48 // (op 4123) most significant: low of high * low
s_mul_i32 s18, s44, s48 // (op 4123) least significant
s_mul_hi_u32 s6, s44, s48 // (op 4123) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4123) most significant: sum
s_add_u32 s19, s19, s6 // (op 4123) most significant: sum


s_mov_b32 s80, s46 // (op 4123) reg expression
s_mov_b32 s81, s47 // (op 4123) reg expression
s_mov_b32 s82, s18 // (op 4123) reg expression
s_mov_b32 s83, 131072 // (op 4123) call()

v_and_b32 v13, -8, v1 // (op 4562) 
v_sub_u32 v17, v1, v13 // (op 4562) 

v_lshlrev_b32 v13, 4, v17 // (op 4562) 

v_lshrrev_b32 v17, 3, v1 // (op 4562) 
v_lshl_add_u32 v19, v17, 7, v13 // (op 4562) 


s_nop 0
v_readfirstlane_b32 s15, v19 // (op 4562) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 4627) 
v_sub_u32 v17, v1, v13 // (op 4627) 

v_lshlrev_b32 v13, 4, v17 // (op 4627) 

v_lshrrev_b32 v17, 3, v1 // (op 4627) 
v_lshl_add_u32 v19, v17, 7, v13 // (op 4627) 


s_nop 0
v_readfirstlane_b32 s6, v19 // (op 4627) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -128, v1 // (op 4702) 
v_sub_u32 v17, v1, v13 // (op 4702) 

v_lshrrev_b32 v13, 7, v1 // (op 4702) 
v_lshl_add_u32 v19, v13, 8, v17 // (op 4702) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 897) 
s_and_b32 s7, s18, 4294967294 // (op 897) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 896) 
s_mov_b32 s8, s18 // (op 896) convert

s_cmp_lt_u32 s7, s8 // (op 896) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_896 // (op 896) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_896 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_896:
s_barrier  // (op 5176) 
s_add_u32 s7, s7, 1 // (op 898) 


v_mov_b64 v[16:17], v[20:21] // (op 3638) 
s_add_u32 s8, 16384, s15 // (op 3638) 

s_mov_b32 m0, s8 // (op 3638) reg expression
buffer_load_dwordx4 v16, s[68:71], 0 offen offset: 0 lds // (op 3638) Load value direct to lds


v_add_co_u32 v20, vcc, 128, v20 // (op 3713) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 3713) most significant half
s_barrier  // (op 5195) 

s_lshl_b64 s[18:19], s[36:37], 5 // (op 3644) 
v_mov_b64 v[16:17], s[18:19] // (op 3644) convert


v_mov_b64 v[24:25], v[22:23] // (op 3644) 
v_mov_b64 v[26:27], v[24:25] // (op 3644) reg expression
s_mov_b32 m0, s6 // (op 3644) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 3644) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 3644) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3644) most significant half

s_add_u32 m0, m0, 4096 // (op 3644) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 3644) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 3644) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3644) most significant half

s_add_u32 m0, m0, 4096 // (op 3644) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 3644) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v16 // (op 3644) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3644) most significant half

s_add_u32 m0, m0, 4096 // (op 3644) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 3644) Load value direct to lds



s_lshl_b64 s[18:19], s[36:37], 7 // (op 3857) 
v_mov_b32 v13, s18 // (op 3857) 
v_mov_b32 v16, s19 // (op 3857) 
v_add_co_u32 v22, s[28:29], v22, v13 // (op 3857) least significant half
v_addc_co_u32 v23, s[28:29], v23, v16, s[28:29] // (op 3857) most significant half





v_mov_b64 v[16:17], s[44:45] // (op 452) convert

v_mov_b32 v13, v18 // (op 452) 
v_mov_b32 v24, v13 // (op 452) reg expression
buffer_load_ubyte v26, v24, s[80:83], 0 offen offset: 0 // (op 452) Load value
v_add_u32 v24, v24, v16 // (op 452) 
buffer_load_ubyte v27, v24, s[80:83], 0 offen offset: 0 // (op 452) Load value



s_lshl_b32 s8, s44, 2 // (op 4113) 
v_add_u32 v18, v18, s8 // (op 4113) 

s_barrier  // (op 5214) 
s_waitcnt vmcnt(6)
s_barrier  // (op 5182) 






ds_read_b128 v[30:33], v4 offset:16384 // (op 441) Load local data 
ds_read_b128 v[34:37], v4 offset:16448 // (op 441) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v26 offset:40960 // (op 450) Store local data 
s_barrier  // (op 5201) 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v27 offset:41088 // (op 450) Store local data 




ds_read_b64_tr_b8 v[38:39], v9 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v9 offset:1024 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v9 offset:8192 // (op 443) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v9 offset:9216 // (op 443) Transpose load local data 


v_mov_b32 v13, v28 // (op 445) 

buffer_load_ubyte v29, v13, s[76:79], 0 offen offset: 0 // (op 445) Load value

v_add_u32 v28, 4, v28 // (op 3985) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 5221) 


ds_read_u8 v13, v8 offset:40960 // (op 447) Load local data 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v29, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 448) 
ds_read_b64_tr_b8 v[46:47], v9 offset:32 // (op 592) Transpose load local data 


ds_read_b64_tr_b8 v[48:49], v9 offset:1056 // (op 592) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v9 offset:8224 // (op 592) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v9 offset:9248 // (op 592) Transpose load local data 


ds_read_u8 v13, v8 offset:40992 // (op 595) Load local data 



ds_read_b64_tr_b8 v[38:39], v9 offset:64 // (op 748) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:37], v[46:53], a[4:7], v29, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 596) 
ds_read_b64_tr_b8 v[40:41], v9 offset:1088 // (op 748) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v9 offset:8256 // (op 748) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v9 offset:9280 // (op 748) Transpose load local data 


ds_read_u8 v13, v8 offset:41024 // (op 751) Load local data 



ds_read_b64_tr_b8 v[46:47], v9 offset:96 // (op 904) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v9 offset:1120 // (op 904) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:37], v[38:45], a[8:11], v29, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 752) 
ds_read_b64_tr_b8 v[50:51], v9 offset:8288 // (op 904) Transpose load local data 


ds_read_b64_tr_b8 v[52:53], v9 offset:9312 // (op 904) Transpose load local data 


ds_read_u8 v13, v8 offset:41056 // (op 907) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:37], v[46:53], a[12:15], v29, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 908) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 896) 
s_mov_b32 s8, s18 // (op 896) convert

s_cmp_lt_u32 s7, s8 // (op 896) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_896 // (op 896) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_896 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_896:








s_barrier  // (op 5353) 



v_cmp_eq_f32 vcc, 0.00000, v14 // (op 218) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_218 // (op 218) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_218
v_accvgpr_read v16, a0 // (op 226) 
v_accvgpr_read v17, a1 // (op 226) 
v_accvgpr_read v18, a2 // (op 226) 
v_accvgpr_read v19, a3 // (op 226) 
v_mul_f32 v20, v15, v16 // (op 226) 
v_mul_f32 v21, v15, v17 // (op 226) 
v_mul_f32 v22, v15, v18 // (op 226) 
v_mul_f32 v23, v15, v19 // (op 226) 



v_mov_b32 v0, v7 // (op 228) reg expression

buffer_store_dwordx4 v[20:23], v0, s[64:67], 0 offen // (op 228) Store value


v_accvgpr_read v16, a4 // (op 1132) 
v_accvgpr_read v17, a5 // (op 1132) 
v_accvgpr_read v18, a6 // (op 1132) 
v_accvgpr_read v19, a7 // (op 1132) 
v_mul_f32 v20, v15, v16 // (op 1132) 
v_mul_f32 v21, v15, v17 // (op 1132) 
v_mul_f32 v22, v15, v18 // (op 1132) 
v_mul_f32 v23, v15, v19 // (op 1132) 



s_lshl_b32 s15, s24, 7 // (op 1134) 
v_add_u32 v0, v7, s15 // (op 1134) 


buffer_store_dwordx4 v[20:23], v0, s[64:67], 0 offen // (op 1134) Store value


v_accvgpr_read v16, a8 // (op 1144) 
v_accvgpr_read v17, a9 // (op 1144) 
v_accvgpr_read v18, a10 // (op 1144) 
v_accvgpr_read v19, a11 // (op 1144) 
v_mul_f32 v20, v15, v16 // (op 1144) 
v_mul_f32 v21, v15, v17 // (op 1144) 
v_mul_f32 v22, v15, v18 // (op 1144) 
v_mul_f32 v23, v15, v19 // (op 1144) 



s_lshl_b32 s15, s24, 8 // (op 1146) 
v_add_u32 v0, v7, s15 // (op 1146) 


buffer_store_dwordx4 v[20:23], v0, s[64:67], 0 offen // (op 1146) Store value


v_accvgpr_read v16, a12 // (op 1156) 
v_accvgpr_read v17, a13 // (op 1156) 
v_accvgpr_read v18, a14 // (op 1156) 
v_accvgpr_read v19, a15 // (op 1156) 
v_mul_f32 v20, v15, v16 // (op 1156) 
v_mul_f32 v21, v15, v17 // (op 1156) 
v_mul_f32 v22, v15, v18 // (op 1156) 
v_mul_f32 v23, v15, v19 // (op 1156) 



s_lshl_b32 s15, s24, 7 // (op 1158) 
s_mul_i32 s6, 3, s15 // (op 1158) 

v_add_u32 v0, v7, s6 // (op 1158) 


buffer_store_dwordx4 v[20:23], v0, s[64:67], 0 offen // (op 1158) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_218 // (op 218) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_218
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_218:


v_mov_b32 v0, v6 // (op 241) reg expression

buffer_load_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 241) Load value

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




v_mov_b32 v0, v7 // (op 240) reg expression

buffer_store_dwordx4 v[16:19], v0, s[64:67], 0 offen // (op 240) Store value




s_lshl_b32 s15, s4, 7 // (op 1190) 
v_add_u32 v0, v6, s15 // (op 1190) 


buffer_load_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 1190) Load value

v_accvgpr_read v20, a4 // (op 1187) 
v_accvgpr_read v21, a5 // (op 1187) 
v_accvgpr_read v22, a6 // (op 1187) 
v_accvgpr_read v23, a7 // (op 1187) 
v_mul_f32 v24, v15, v20 // (op 1187) 
v_mul_f32 v25, v15, v21 // (op 1187) 
v_mul_f32 v26, v15, v22 // (op 1187) 
v_mul_f32 v27, v15, v23 // (op 1187) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v14, v16 // (op 1191) 
v_mul_f32 v21, v14, v17 // (op 1191) 
v_mul_f32 v22, v14, v18 // (op 1191) 
v_mul_f32 v23, v14, v19 // (op 1191) 

v_add_f32 v16, v20, v24 // (op 1188) 
v_add_f32 v17, v21, v25 // (op 1188) 
v_add_f32 v18, v22, v26 // (op 1188) 
v_add_f32 v19, v23, v27 // (op 1188) 




s_lshl_b32 s15, s24, 7 // (op 1189) 
v_add_u32 v0, v7, s15 // (op 1189) 


buffer_store_dwordx4 v[16:19], v0, s[64:67], 0 offen // (op 1189) Store value




s_lshl_b32 s15, s4, 8 // (op 1202) 
v_add_u32 v0, v6, s15 // (op 1202) 


buffer_load_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 1202) Load value

v_accvgpr_read v20, a8 // (op 1199) 
v_accvgpr_read v21, a9 // (op 1199) 
v_accvgpr_read v22, a10 // (op 1199) 
v_accvgpr_read v23, a11 // (op 1199) 
v_mul_f32 v24, v15, v20 // (op 1199) 
v_mul_f32 v25, v15, v21 // (op 1199) 
v_mul_f32 v26, v15, v22 // (op 1199) 
v_mul_f32 v27, v15, v23 // (op 1199) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v14, v16 // (op 1203) 
v_mul_f32 v21, v14, v17 // (op 1203) 
v_mul_f32 v22, v14, v18 // (op 1203) 
v_mul_f32 v23, v14, v19 // (op 1203) 

v_add_f32 v16, v20, v24 // (op 1200) 
v_add_f32 v17, v21, v25 // (op 1200) 
v_add_f32 v18, v22, v26 // (op 1200) 
v_add_f32 v19, v23, v27 // (op 1200) 




s_lshl_b32 s15, s24, 8 // (op 1201) 
v_add_u32 v0, v7, s15 // (op 1201) 


buffer_store_dwordx4 v[16:19], v0, s[64:67], 0 offen // (op 1201) Store value




s_lshl_b32 s15, s4, 7 // (op 1214) 
s_mul_i32 s6, 3, s15 // (op 1214) 

v_add_u32 v0, v6, s6 // (op 1214) 


buffer_load_dwordx4 v[16:19], v0, s[60:63], 0 offen // (op 1214) Load value

v_accvgpr_read v20, a12 // (op 1211) 
v_accvgpr_read v21, a13 // (op 1211) 
v_accvgpr_read v22, a14 // (op 1211) 
v_accvgpr_read v23, a15 // (op 1211) 
v_mul_f32 v24, v15, v20 // (op 1211) 
v_mul_f32 v25, v15, v21 // (op 1211) 
v_mul_f32 v26, v15, v22 // (op 1211) 
v_mul_f32 v27, v15, v23 // (op 1211) 

s_waitcnt vmcnt(0)
v_mul_f32 v20, v14, v16 // (op 1215) 
v_mul_f32 v21, v14, v17 // (op 1215) 
v_mul_f32 v22, v14, v18 // (op 1215) 
v_mul_f32 v23, v14, v19 // (op 1215) 

v_add_f32 v16, v20, v24 // (op 1212) 
v_add_f32 v17, v21, v25 // (op 1212) 
v_add_f32 v18, v22, v26 // (op 1212) 
v_add_f32 v19, v23, v27 // (op 1212) 




s_lshl_b32 s15, s24, 7 // (op 1213) 
s_mul_i32 s6, 3, s15 // (op 1213) 

v_add_u32 v0, v7, s6 // (op 1213) 


buffer_store_dwordx4 v[16:19], v0, s[64:67], 0 offen // (op 1213) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_218:
s_add_u32 s0, s0, s1 // (op 427) 


s_mul_i32 s15, s52, s16 // (op 425) 
s_cmp_gt_u32 s53, s15 // (op 425) 
s_mov_b32 s6, scc // (op 425) 
s_sub_u32 s7, s53, s15 // (op 425) 

s_cmp_eq_i32 s6, 0 // (op 425) 
s_cselect_b32 s8, 0, s7 // (op 425) 


s_cmp_lt_u32 s52, s8 // (op 425) 
s_mov_b32 s15, scc // (op 425) 
s_cmp_eq_i32 s15, 0 // (op 425) 
s_cselect_b32 s6, s8, s52 // (op 425) 


s_cmp_lt_u32 s0, s6 // (op 425) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_425 // (op 425) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopTop_DPStreamTileLoop_425 if true)
RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_ForLoopBottom_DPStreamTileLoop_425:












































s_endpgm  // End of RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.LRR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2, .LRR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2_end-RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: True
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
            { type: int64, semantic: SizeSumScaleA },
            { type: address, semantic: RR_SKAddressFlags },
            { type: uint32, semantic: RR_SKNumDPTilesPerWG },
            { type: uint32, semantic: RR_SKNumDPTiles } ]
    macrotile: [32, 128, 128]
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
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TT_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_32x128x128_UR_2.kd
    .kernarg_segment_size: 192
    .group_segment_fixed_size: 41984
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 91
    .vgpr_count: 92
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_12
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_10_stride_1_19
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
      - .name: Tensor_2_size_1_20
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
      - .name: Tensor_2_stride_0_11
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
      - .name: Tensor_7_stride_0_16
        .size: 8
        .offset: 152
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 160
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_17
        .size: 8
        .offset: 168
        .value_kind: by_value
      - .name: SCRATCH_ZeroedBeforeAndAfter
        .size: 8
        .offset: 176
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: numDPTilesPerWG
        .size: 4
        .offset: 184
        .value_kind: by_value
      - .name: numDPTiles
        .size: 4
        .offset: 188
        .value_kind: by_value
...

.end_amdgpu_metadata

