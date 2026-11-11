// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2,@function
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 4608) 
v_and_b32 v3, 1, v0 // (op 4608) 

v_and_b32 v0, 15, v1 // (op 4608) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 4608) 


v_and_b32 v0, 48, v1 // (op 4608) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 4608) 


v_lshrrev_b32 v0, 6, v1 // (op 4866) 
v_and_b32 v4, 1, v0 // (op 4866) 

v_and_b32 v0, 15, v1 // (op 4866) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4866) 


v_and_b32 v0, 48, v1 // (op 4866) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4866) 


v_lshrrev_b32 v0, 6, v1 // (op 5080) 
v_and_b32 v5, 1, v0 // (op 5080) 

v_and_b32 v0, 15, v1 // (op 5080) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 5080) 


v_and_b32 v0, 48, v1 // (op 5080) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 5080) 


v_lshrrev_b32 v0, 6, v1 // (op 5192) 
v_and_b32 v6, 1, v0 // (op 5192) 

v_and_b32 v0, 15, v1 // (op 5192) 
v_lshl_add_u32 v7, v6, 4, v0 // (op 5192) 


v_and_b32 v0, 48, v1 // (op 5192) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 5192) 


v_and_b32 v0, 48, v1 // (op 4538) 
v_and_b32 v7, 63, v1 // (op 4538) 
v_sub_u32 v8, v7, v0 // (op 4538) 

v_lshrrev_b32 v7, 1, v8 // (op 4538) 
v_add_lshl_u32 v9, v0, v7, 7 // (op 4538) 


v_and_b32 v0, 1, v8 // (op 4538) 

v_lshlrev_b32 v7, 3, v0 // (op 4538) 

v_lshrrev_b32 v0, 7, v1 // (op 4538) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 4538) 


v_add_u32 v0, v9, v8 // (op 4538) 


v_lshrrev_b32 v7, 7, v1 // (op 4679) 
v_and_b32 v8, 15, v1 // (op 4679) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 4679) 


v_and_b32 v7, 63, v1 // (op 4679) 
v_lshrrev_b32 v8, 4, v7 // (op 4679) 

v_lshl_add_u32 v7, v8, 7, v9 // (op 4679) 


v_and_b32 v8, 48, v1 // (op 4796) 
v_and_b32 v9, 63, v1 // (op 4796) 
v_sub_u32 v10, v9, v8 // (op 4796) 

v_lshrrev_b32 v9, 1, v10 // (op 4796) 
v_add_lshl_u32 v11, v8, v9, 7 // (op 4796) 


v_and_b32 v8, 1, v10 // (op 4796) 

v_lshlrev_b32 v9, 3, v8 // (op 4796) 

v_lshrrev_b32 v8, 7, v1 // (op 4796) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 4796) 


v_add_u32 v8, v11, v10 // (op 4796) 


v_lshrrev_b32 v9, 7, v1 // (op 4937) 
v_and_b32 v10, 15, v1 // (op 4937) 
v_lshl_add_u32 v11, v9, 4, v10 // (op 4937) 


v_and_b32 v9, 63, v1 // (op 4937) 
v_lshrrev_b32 v10, 4, v9 // (op 4937) 

v_lshl_add_u32 v9, v10, 7, v11 // (op 4937) 


v_and_b32 v10, 48, v1 // (op 5041) 
v_and_b32 v11, 63, v1 // (op 5041) 
v_sub_u32 v12, v11, v10 // (op 5041) 

v_lshrrev_b32 v11, 1, v12 // (op 5041) 
v_add_lshl_u32 v13, v10, v11, 7 // (op 5041) 


v_and_b32 v10, 1, v12 // (op 5041) 

v_lshlrev_b32 v11, 3, v10 // (op 5041) 

v_lshrrev_b32 v10, 7, v1 // (op 5041) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 5041) 


v_add_u32 v10, v13, v12 // (op 5041) 


v_lshrrev_b32 v11, 7, v1 // (op 5109) 
v_and_b32 v12, 15, v1 // (op 5109) 
v_lshl_add_u32 v13, v11, 4, v12 // (op 5109) 


v_and_b32 v11, 63, v1 // (op 5109) 
v_lshrrev_b32 v12, 4, v11 // (op 5109) 

v_lshl_add_u32 v11, v12, 7, v13 // (op 5109) 


v_and_b32 v12, 48, v1 // (op 5153) 
v_and_b32 v13, 63, v1 // (op 5153) 
v_sub_u32 v14, v13, v12 // (op 5153) 

v_lshrrev_b32 v13, 1, v14 // (op 5153) 
v_add_lshl_u32 v15, v12, v13, 7 // (op 5153) 


v_and_b32 v12, 1, v14 // (op 5153) 

v_lshlrev_b32 v13, 3, v12 // (op 5153) 

v_lshrrev_b32 v12, 7, v1 // (op 5153) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 5153) 


v_add_u32 v12, v15, v14 // (op 5153) 


v_lshrrev_b32 v13, 7, v1 // (op 5221) 
v_and_b32 v14, 15, v1 // (op 5221) 
v_lshl_add_u32 v15, v13, 4, v14 // (op 5221) 


v_and_b32 v13, 63, v1 // (op 5221) 
v_lshrrev_b32 v14, 4, v13 // (op 5221) 

v_lshl_add_u32 v13, v14, 7, v15 // (op 5221) 


v_mov_b32 v14, s6 // (op 2) Move value
v_mov_b32 v15, s7 // (op 39) Move value


s_mov_b32 s0, 0 // (op 379) call()
s_mov_b32 s1, 0 // (op 380) call()
s_mov_b32 s6, 0 // (op 381) call()
s_mov_b32 s7, 0 // (op 382) call()
s_mov_b32 s18, 0 // (op 383) call()
s_mov_b32 s19, 0 // (op 384) call()
s_mov_b32 s0, 0 // (op 279) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 278) 
s_cmp_gt_u32 s9, s52 // (op 278) 
s_mov_b32 s53, scc // (op 278) 
s_sub_u32 s54, s9, s52 // (op 278) 

s_cmp_eq_i32 s53, 0 // (op 278) 
s_cselect_b32 s52, 0, s54 // (op 278) 


s_cmp_lt_u32 s8, s52 // (op 278) 
s_mov_b32 s53, scc // (op 278) 
s_cmp_eq_i32 s53, 0 // (op 278) 
s_cselect_b32 s54, s52, s8 // (op 278) 


s_cmp_lt_u32 s0, s54 // (op 278) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_278 // (op 278) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_278 if false)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_278:
v_lshrrev_b32 v16, 7, v1 // (op 4454) 
s_mul_i32 s52, s16, s8 // (op 4454) 
s_add_u32 s52, s52, s0 // (op 4454) 
v_and_b32 v17, 63, v1 // (op 4454) 
s_mul_hi_u32 s53, s52, s13 // (op 4454) 
v_lshrrev_b32 v18, 4, v17 // (op 4454) 
s_sub_u32 s54, s52, s53 // (op 4454) 
v_lshlrev_b32 v19, 2, v18 // (op 4454) 

v_lshl_add_u32 v18, v16, 4, v19 // (op 4454) 


s_lshr_b32 s55, s54, 1 // (op 4454) 

s_add_u32 s54, s55, s53 // (op 4454) 


v_lshrrev_b32 v16, 2, v17 // (op 4454) 

s_lshr_b32 s53, s54, s11 // (op 4454) 

v_and_b32 v17, 3, v16 // (op 4454) 

s_and_b32 s54, s11, 2147483648 // (op 4454) 
v_and_b32 v16, 3, v1 // (op 4454) 
s_cmp_eq_i32 s54, 0 // (op 4454) 
s_mov_b32 s55, scc // (op 4454) 

v_lshl_add_u32 v19, v17, 2, v16 // (op 4454) 


s_cmp_eq_i32 s55, 0 // (op 4454) 
s_cselect_b32 s54, s52, s53 // (op 4454) 



v_lshrrev_b32 v16, 6, v1 // (op 4454) 
s_mul_hi_u32 s52, s54, s12 // (op 4454) 
v_and_b32 v17, 1, v16 // (op 4454) 

s_sub_u32 s53, s54, s52 // (op 4454) 
v_lshl_add_u32 v16, v17, 4, v19 // (op 4454) 


s_lshr_b32 s55, s53, 1 // (op 4454) 

s_add_u32 s53, s55, s52 // (op 4454) 


s_lshr_b32 s52, s53, s10 // (op 4454) 

s_and_b32 s53, s10, 2147483648 // (op 4454) 
s_cmp_eq_i32 s53, 0 // (op 4454) 
s_mov_b32 s55, scc // (op 4454) 

s_cmp_eq_i32 s55, 0 // (op 4454) 
s_cselect_b32 s53, s54, s52 // (op 4454) 


s_mul_i32 s52, s53, s14 // (op 4454) 
v_lshl_add_u32 v17, s53, 7, v18 // (op 4454) 


s_sub_u32 s53, s54, s52 // (op 4454) 


v_lshl_add_u32 v18, s53, 5, v16 // (op 4454) 


v_mul_lo_u32 v16, v18, s4 // (op 4454) 

v_add_lshl_u32 v18, v17, v16, 1 // (op 4454) 


s_mul_i32 s53, s4, s23 // (op 4456) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 4456) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 4456) least significant
s_mul_hi_u32 s55, s4, s22 // (op 4456) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4456) most significant: sum
s_add_u32 s53, s53, s55 // (op 4456) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 4456) 

s_mov_b32 s56, s20 // (op 4456) reg expression
s_mov_b32 s57, s21 // (op 4456) reg expression
s_mov_b32 s58, s54 // (op 4456) reg expression
s_mov_b32 s59, 131072 // (op 4456) call()

v_and_b32 v16, 48, v1 // (op 4555) 
v_and_b32 v17, 63, v1 // (op 4555) 
v_sub_u32 v19, v17, v16 // (op 4555) 

v_lshrrev_b32 v17, 1, v19 // (op 4555) 
v_add_lshl_u32 v20, v16, v17, 7 // (op 4555) 


v_and_b32 v16, 1, v19 // (op 4555) 

v_lshlrev_b32 v17, 3, v16 // (op 4555) 

v_lshrrev_b32 v16, 7, v1 // (op 4555) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 4555) 


v_add_u32 v16, v20, v19 // (op 4555) 


v_lshrrev_b32 v17, 6, v1 // (op 4620) 
v_and_b32 v19, 1, v17 // (op 4620) 

v_and_b32 v17, 15, v1 // (op 4620) 
v_lshl_add_u32 v20, v19, 4, v17 // (op 4620) 


v_and_b32 v17, 48, v1 // (op 4620) 
v_lshl_add_u32 v19, v20, 7, v17 // (op 4620) 


v_lshrrev_b32 v17, 7, v1 // (op 4696) 
v_and_b32 v20, 15, v1 // (op 4696) 
v_lshl_add_u32 v21, v17, 4, v20 // (op 4696) 


v_and_b32 v17, 63, v1 // (op 4696) 
v_lshrrev_b32 v20, 4, v17 // (op 4696) 

v_lshl_add_u32 v17, v20, 7, v21 // (op 4696) 


v_lshrrev_b32 v20, 7, v1 // (op 4714) 
s_mul_i32 s52, s16, s8 // (op 4714) 
s_add_u32 s52, s52, s0 // (op 4714) 
v_and_b32 v21, 63, v1 // (op 4714) 
s_mul_hi_u32 s53, s52, s13 // (op 4714) 
v_lshrrev_b32 v22, 4, v21 // (op 4714) 
s_sub_u32 s54, s52, s53 // (op 4714) 
v_lshlrev_b32 v23, 2, v22 // (op 4714) 

s_lshr_b32 s55, s54, 1 // (op 4714) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 4714) 


s_add_u32 s54, s55, s53 // (op 4714) 


v_lshrrev_b32 v20, 2, v21 // (op 4714) 

s_lshr_b32 s53, s54, s11 // (op 4714) 

v_and_b32 v21, 3, v20 // (op 4714) 

s_and_b32 s54, s11, 2147483648 // (op 4714) 
v_and_b32 v20, 3, v1 // (op 4714) 
s_cmp_eq_i32 s54, 0 // (op 4714) 
s_mov_b32 s55, scc // (op 4714) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 4714) 


s_cmp_eq_i32 s55, 0 // (op 4714) 
s_cselect_b32 s54, s52, s53 // (op 4714) 



v_lshrrev_b32 v20, 6, v1 // (op 4714) 
s_mul_hi_u32 s55, s54, s12 // (op 4714) 
v_and_b32 v21, 1, v20 // (op 4714) 

s_sub_u32 s52, s54, s55 // (op 4714) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 4714) 


s_lshr_b32 s53, s52, 1 // (op 4714) 

s_add_u32 s52, s53, s55 // (op 4714) 


s_lshr_b32 s53, s52, s10 // (op 4714) 

s_and_b32 s52, s10, 2147483648 // (op 4714) 
s_cmp_eq_i32 s52, 0 // (op 4714) 
s_mov_b32 s55, scc // (op 4714) 

s_cmp_eq_i32 s55, 0 // (op 4714) 
s_cselect_b32 s52, s54, s53 // (op 4714) 


s_mul_i32 s53, s52, s14 // (op 4714) 
v_lshl_add_u32 v21, s52, 7, v22 // (op 4714) 


s_sub_u32 s52, s54, s53 // (op 4714) 


v_lshl_add_u32 v22, s52, 5, v20 // (op 4714) 


v_mul_lo_u32 v20, v22, s24 // (op 4714) 

v_add_lshl_u32 v22, v21, v20, 1 // (op 4714) 


s_mul_i32 s53, s24, s23 // (op 4716) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 4716) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 4716) least significant
s_mul_hi_u32 s55, s24, s22 // (op 4716) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4716) most significant: sum
s_add_u32 s53, s53, s55 // (op 4716) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 4716) 

s_mov_b32 s60, s26 // (op 4716) reg expression
s_mov_b32 s61, s27 // (op 4716) reg expression
s_mov_b32 s62, s54 // (op 4716) reg expression
s_mov_b32 s63, 131072 // (op 4716) call()

v_lshrrev_b32 v20, 5, v1 // (op 4972) 
v_lshlrev_b32 v21, 2, v20 // (op 4972) 

v_lshl_add_u32 v20, s16, 7, v21 // (op 4972) 

v_and_b32 v21, -32, v1 // (op 4972) 
v_sub_u32 v23, v1, v21 // (op 4972) 

v_lshl_add_u32 v21, v20, 5, v23 // (op 4972) 


v_lshlrev_b32 v20, 2, v21 // (op 4972) 

s_lshl_b32 s52, s15, 14 // (op 4974) 
s_mov_b32 s64, s28 // (op 4974) reg expression
s_mov_b32 s65, s29 // (op 4974) reg expression
s_mov_b32 s66, s52 // (op 4974) reg expression
s_mov_b32 s67, 131072 // (op 4974) call()

s_mul_i32 s6, s8, s16 // (op 283) 
s_add_u32 s6, s6, s0 // (op 283) 
s_mul_hi_u32 s52, s6, s13 // (op 284) 
s_sub_u32 s53, s6, s52 // (op 284) 
s_lshr_b32 s54, s53, 1 // (op 284) 

s_add_u32 s53, s54, s52 // (op 284) 


s_lshr_b32 s52, s53, s11 // (op 284) 

s_and_b32 s53, s11, 2147483648 // (op 284) 
s_cmp_eq_i32 s53, 0 // (op 284) 
s_mov_b32 s54, scc // (op 284) 

s_cmp_eq_i32 s54, 0 // (op 284) 
s_cselect_b32 s53, s6, s52 // (op 284) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 284) 
s_mul_i32 s52, s53, s54 // (op 284) 


s_sub_u32 s18, s6, s52 // (op 284) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 285) 
s_cmp_gt_u32 s52, s18 // (op 285) 
s_mov_b32 s54, scc // (op 285) 
s_sub_u32 s55, s52, s18 // (op 285) 

s_cmp_eq_i32 s54, 0 // (op 285) 
s_cselect_b32 s52, 0, s55 // (op 285) 


s_cmp_gt_u32 s8, s0 // (op 285) 
s_mov_b32 s53, scc // (op 285) 
s_sub_u32 s54, s8, s0 // (op 285) 
s_cmp_eq_i32 s53, 0 // (op 285) 
s_cselect_b32 s55, 0, s54 // (op 285) 


s_cmp_lt_u32 s52, s55 // (op 285) 
s_mov_b32 s53, scc // (op 285) 
s_cmp_eq_i32 s53, 0 // (op 285) 
s_cselect_b32 s54, s55, s52 // (op 285) 



s_cmp_gt_u32 s9, s6 // (op 285) 
s_mov_b32 s55, scc // (op 285) 
s_sub_u32 s52, s9, s6 // (op 285) 
s_cmp_eq_i32 s55, 0 // (op 285) 
s_cselect_b32 s53, 0, s52 // (op 285) 


s_cmp_lt_u32 s54, s53 // (op 285) 
s_mov_b32 s52, scc // (op 285) 
s_cmp_eq_i32 s52, 0 // (op 285) 
s_cselect_b32 s7, s53, s54 // (op 285) 



s_add_u32 s52, s18, s7 // (op 286) 
s_sub_u32 s19, s52, 1 // (op 286) 

v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 1482) call()
v_accvgpr_write a5, 0.00000 // (op 1482) call()
v_accvgpr_write a6, 0.00000 // (op 1482) call()
v_accvgpr_write a7, 0.00000 // (op 1482) call()
v_accvgpr_write a8, 0.00000 // (op 1644) call()
v_accvgpr_write a9, 0.00000 // (op 1644) call()
v_accvgpr_write a10, 0.00000 // (op 1644) call()
v_accvgpr_write a11, 0.00000 // (op 1644) call()
v_accvgpr_write a12, 0.00000 // (op 1806) call()
v_accvgpr_write a13, 0.00000 // (op 1806) call()
v_accvgpr_write a14, 0.00000 // (op 1806) call()
v_accvgpr_write a15, 0.00000 // (op 1806) call()
v_and_b32 v21, -8, v1 // (op 4026) 
s_mul_i32 s52, s16, s8 // (op 4026) 
s_add_u32 s52, s52, s0 // (op 4026) 
v_sub_u32 v23, v1, v21 // (op 4026) 

s_mul_hi_u32 s53, s52, s13 // (op 4026) 
v_lshlrev_b32 v21, 4, v23 // (op 4026) 

s_sub_u32 s54, s52, s53 // (op 4026) 
v_lshrrev_b32 v23, 3, v1 // (op 4026) 
s_lshr_b32 s55, s54, 1 // (op 4026) 

s_add_u32 s54, s55, s53 // (op 4026) 


s_lshr_b32 s53, s54, s11 // (op 4026) 

s_and_b32 s54, s11, 2147483648 // (op 4026) 
s_cmp_eq_i32 s54, 0 // (op 4026) 
s_mov_b32 s55, scc // (op 4026) 

s_cmp_eq_i32 s55, 0 // (op 4026) 
s_cselect_b32 s54, s52, s53 // (op 4026) 


s_mul_hi_u32 s53, s54, s12 // (op 4026) 
s_sub_u32 s55, s54, s53 // (op 4026) 
s_lshr_b32 s68, s55, 1 // (op 4026) 

s_add_u32 s55, s68, s53 // (op 4026) 


s_lshr_b32 s53, s55, s10 // (op 4026) 

s_and_b32 s55, s10, 2147483648 // (op 4026) 
s_cmp_eq_i32 s55, 0 // (op 4026) 
s_mov_b32 s68, scc // (op 4026) 

s_cmp_eq_i32 s68, 0 // (op 4026) 
s_cselect_b32 s55, s54, s53 // (op 4026) 


v_lshl_add_u32 v24, s55, 7, v21 // (op 4026) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 4026) 
v_mov_b32 v27, 0 // (op 4026) convert
v_mov_b32 v26, v24 // (op 4026) convert

s_mul_i32 s53, s54, s68 // (op 4026) 


s_sub_u32 s54, s52, s53 // (op 4026) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 4026) 


v_mul_lo_u32 v25, v21, s31 // (op 4026) most significant: low of low * high
v_mul_hi_u32 v23, v21, s30 // (op 4026) most significant: high of low * low
v_mul_lo_u32 v24, v21, s30 // (op 4026) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 4026) most significant: sum


v_add_co_u32 v28, s[52:53], v26, v24 // (op 4026) least significant half
v_addc_co_u32 v29, s[52:53], v27, v25, s[52:53] // (op 4026) most significant half



s_mul_i32 s53, s30, s3 // (op 4028) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 4028) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 4028) least significant
s_mul_hi_u32 s55, s30, s2 // (op 4028) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4028) most significant: sum
s_add_u32 s53, s53, s55 // (op 4028) most significant: sum


s_mov_b32 s68, s32 // (op 4028) reg expression
s_mov_b32 s69, s33 // (op 4028) reg expression
s_mov_b32 s70, s52 // (op 4028) reg expression
s_mov_b32 s71, 131072 // (op 4028) call()

v_and_b32 v21, -8, v1 // (op 4045) 
s_mul_i32 s52, s16, s8 // (op 4045) 
s_add_u32 s52, s52, s0 // (op 4045) 
v_sub_u32 v23, v1, v21 // (op 4045) 

s_mul_hi_u32 s53, s52, s13 // (op 4045) 
v_lshlrev_b32 v21, 4, v23 // (op 4045) 

s_sub_u32 s54, s52, s53 // (op 4045) 
v_lshrrev_b32 v23, 3, v1 // (op 4045) 
s_lshr_b32 s55, s54, 1 // (op 4045) 

s_add_u32 s54, s55, s53 // (op 4045) 


s_lshr_b32 s53, s54, s11 // (op 4045) 

s_and_b32 s54, s11, 2147483648 // (op 4045) 
s_cmp_eq_i32 s54, 0 // (op 4045) 
s_mov_b32 s55, scc // (op 4045) 

s_cmp_eq_i32 s55, 0 // (op 4045) 
s_cselect_b32 s54, s52, s53 // (op 4045) 


s_mul_hi_u32 s53, s54, s12 // (op 4045) 
s_sub_u32 s55, s54, s53 // (op 4045) 
s_lshr_b32 s72, s55, 1 // (op 4045) 

s_add_u32 s55, s72, s53 // (op 4045) 


s_lshr_b32 s53, s55, s10 // (op 4045) 

s_and_b32 s55, s10, 2147483648 // (op 4045) 
s_cmp_eq_i32 s55, 0 // (op 4045) 
s_mov_b32 s72, scc // (op 4045) 

s_cmp_eq_i32 s72, 0 // (op 4045) 
s_cselect_b32 s55, s54, s53 // (op 4045) 


v_lshl_add_u32 v24, s55, 7, v21 // (op 4045) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 4045) 
v_mov_b32 v27, 0 // (op 4045) convert
v_mov_b32 v26, v24 // (op 4045) convert

s_mul_i32 s53, s54, s72 // (op 4045) 


s_sub_u32 s54, s52, s53 // (op 4045) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 4045) 


v_mul_lo_u32 v25, v21, s31 // (op 4045) most significant: low of low * high
v_mul_hi_u32 v23, v21, s30 // (op 4045) most significant: high of low * low
v_mul_lo_u32 v24, v21, s30 // (op 4045) least significant: low of low * low
v_add_u32 v25, v25, v23 // (op 4045) most significant: sum


v_add_co_u32 v30, s[52:53], v26, v24 // (op 4045) least significant half
v_addc_co_u32 v31, s[52:53], v27, v25, s[52:53] // (op 4045) most significant half



s_mul_i32 s53, s30, s3 // (op 4047) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 4047) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 4047) least significant
s_mul_hi_u32 s55, s30, s2 // (op 4047) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4047) most significant: sum
s_add_u32 s53, s53, s55 // (op 4047) most significant: sum


s_mov_b32 s68, s32 // (op 4047) reg expression
s_mov_b32 s69, s33 // (op 4047) reg expression
s_mov_b32 s70, s52 // (op 4047) reg expression
s_mov_b32 s71, 131072 // (op 4047) call()

v_lshrrev_b32 v21, 3, v1 // (op 4154) 
s_mul_i32 s52, s16, s8 // (op 4154) 
s_add_u32 s52, s52, s0 // (op 4154) 
v_and_b32 v23, -8, v1 // (op 4154) 
s_mul_hi_u32 s53, s52, s13 // (op 4154) 
v_sub_u32 v24, v1, v23 // (op 4154) 

s_sub_u32 s54, s52, s53 // (op 4154) 
v_lshlrev_b32 v23, 4, v24 // (op 4154) 

s_lshr_b32 s55, s54, 1 // (op 4154) 

s_add_u32 s54, s55, s53 // (op 4154) 


s_lshr_b32 s53, s54, s11 // (op 4154) 

s_and_b32 s54, s11, 2147483648 // (op 4154) 
s_cmp_eq_i32 s54, 0 // (op 4154) 
s_mov_b32 s55, scc // (op 4154) 

s_cmp_eq_i32 s55, 0 // (op 4154) 
s_cselect_b32 s54, s52, s53 // (op 4154) 


s_mul_hi_u32 s53, s54, s12 // (op 4154) 
s_sub_u32 s55, s54, s53 // (op 4154) 
s_lshr_b32 s72, s55, 1 // (op 4154) 

s_add_u32 s55, s72, s53 // (op 4154) 


s_lshr_b32 s53, s55, s10 // (op 4154) 

s_and_b32 s55, s10, 2147483648 // (op 4154) 
s_cmp_eq_i32 s55, 0 // (op 4154) 
s_mov_b32 s72, scc // (op 4154) 

s_cmp_eq_i32 s72, 0 // (op 4154) 
s_cselect_b32 s55, s54, s53 // (op 4154) 


s_mul_i32 s53, s55, s14 // (op 4154) 

s_sub_u32 s55, s54, s53 // (op 4154) 

v_lshl_add_u32 v24, s55, 5, v21 // (op 4154) 


v_mul_lo_u32 v27, v24, s35 // (op 4154) most significant: low of low * high
v_mul_hi_u32 v21, v24, s34 // (op 4154) most significant: high of low * low
v_mul_lo_u32 v26, v24, s34 // (op 4154) least significant: low of low * low
v_add_u32 v27, v27, v21 // (op 4154) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 4154) 
s_mul_i32 s53, s54, s72 // (op 4154) 


s_sub_u32 s54, s52, s53 // (op 4154) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 4154) 


v_mov_b32 v25, 0 // (op 4154) convert
v_mov_b32 v24, v21 // (op 4154) convert

v_add_co_u32 v32, s[52:53], v24, v26 // (op 4154) least significant half
v_addc_co_u32 v33, s[52:53], v25, v27, s[52:53] // (op 4154) most significant half



s_mul_i32 s53, s34, s23 // (op 4156) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 4156) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 4156) least significant
s_mul_hi_u32 s55, s34, s22 // (op 4156) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4156) most significant: sum
s_add_u32 s53, s53, s55 // (op 4156) most significant: sum


s_mov_b32 s72, s36 // (op 4156) reg expression
s_mov_b32 s73, s37 // (op 4156) reg expression
s_mov_b32 s74, s52 // (op 4156) reg expression
s_mov_b32 s75, 131072 // (op 4156) call()

v_lshrrev_b32 v21, 3, v1 // (op 4173) 
s_mul_i32 s52, s16, s8 // (op 4173) 
s_add_u32 s52, s52, s0 // (op 4173) 
v_and_b32 v23, -8, v1 // (op 4173) 
s_mul_hi_u32 s53, s52, s13 // (op 4173) 
v_sub_u32 v24, v1, v23 // (op 4173) 

s_sub_u32 s54, s52, s53 // (op 4173) 
v_lshlrev_b32 v23, 4, v24 // (op 4173) 

s_lshr_b32 s55, s54, 1 // (op 4173) 

s_add_u32 s54, s55, s53 // (op 4173) 


s_lshr_b32 s53, s54, s11 // (op 4173) 

s_and_b32 s54, s11, 2147483648 // (op 4173) 
s_cmp_eq_i32 s54, 0 // (op 4173) 
s_mov_b32 s55, scc // (op 4173) 

s_cmp_eq_i32 s55, 0 // (op 4173) 
s_cselect_b32 s54, s52, s53 // (op 4173) 


s_mul_hi_u32 s53, s54, s12 // (op 4173) 
s_sub_u32 s55, s54, s53 // (op 4173) 
s_lshr_b32 s76, s55, 1 // (op 4173) 

s_add_u32 s55, s76, s53 // (op 4173) 


s_lshr_b32 s53, s55, s10 // (op 4173) 

s_and_b32 s55, s10, 2147483648 // (op 4173) 
s_cmp_eq_i32 s55, 0 // (op 4173) 
s_mov_b32 s76, scc // (op 4173) 

s_cmp_eq_i32 s76, 0 // (op 4173) 
s_cselect_b32 s55, s54, s53 // (op 4173) 


s_mul_i32 s53, s55, s14 // (op 4173) 

s_sub_u32 s55, s54, s53 // (op 4173) 

v_lshl_add_u32 v24, s55, 5, v21 // (op 4173) 


v_mul_lo_u32 v27, v24, s35 // (op 4173) most significant: low of low * high
v_mul_hi_u32 v21, v24, s34 // (op 4173) most significant: high of low * low
v_mul_lo_u32 v26, v24, s34 // (op 4173) least significant: low of low * low
v_add_u32 v27, v27, v21 // (op 4173) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 4173) 
s_mul_i32 s53, s54, s76 // (op 4173) 


s_sub_u32 s54, s52, s53 // (op 4173) 


v_lshl_add_u32 v21, s54, 7, v23 // (op 4173) 


v_mov_b32 v25, 0 // (op 4173) convert
v_mov_b32 v24, v21 // (op 4173) convert

v_add_co_u32 v34, s[52:53], v24, v26 // (op 4173) least significant half
v_addc_co_u32 v35, s[52:53], v25, v27, s[52:53] // (op 4173) most significant half



s_mul_i32 s53, s34, s23 // (op 4175) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 4175) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 4175) least significant
s_mul_hi_u32 s55, s34, s22 // (op 4175) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4175) most significant: sum
s_add_u32 s53, s53, s55 // (op 4175) most significant: sum


s_mov_b32 s72, s36 // (op 4175) reg expression
s_mov_b32 s73, s37 // (op 4175) reg expression
s_mov_b32 s74, s52 // (op 4175) reg expression
s_mov_b32 s75, 131072 // (op 4175) call()

v_and_b32 v21, -128, v1 // (op 4282) 
s_mul_i32 s52, s16, s8 // (op 4282) 
s_add_u32 s52, s52, s0 // (op 4282) 
v_sub_u32 v23, v1, v21 // (op 4282) 

s_mul_hi_u32 s53, s52, s13 // (op 4282) 
v_lshrrev_b32 v21, 7, v1 // (op 4282) 
s_sub_u32 s54, s52, s53 // (op 4282) 
v_lshlrev_b32 v24, 1, v21 // (op 4282) 

s_lshr_b32 s55, s54, 1 // (op 4282) 

s_add_u32 s54, s55, s53 // (op 4282) 


s_lshr_b32 s53, s54, s11 // (op 4282) 

s_and_b32 s54, s11, 2147483648 // (op 4282) 
s_cmp_eq_i32 s54, 0 // (op 4282) 
s_mov_b32 s55, scc // (op 4282) 

s_cmp_eq_i32 s55, 0 // (op 4282) 
s_cselect_b32 s54, s52, s53 // (op 4282) 


s_mul_hi_u32 s53, s54, s12 // (op 4282) 
s_sub_u32 s55, s54, s53 // (op 4282) 
s_lshr_b32 s76, s55, 1 // (op 4282) 

s_add_u32 s55, s76, s53 // (op 4282) 


s_lshr_b32 s53, s55, s10 // (op 4282) 

s_and_b32 s55, s10, 2147483648 // (op 4282) 
s_cmp_eq_i32 s55, 0 // (op 4282) 
s_mov_b32 s76, scc // (op 4282) 

s_cmp_eq_i32 s76, 0 // (op 4282) 
s_cselect_b32 s55, s54, s53 // (op 4282) 


v_lshl_add_u32 v21, s55, 7, v23 // (op 4282) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 4282) 
s_mul_i32 s53, s54, s76 // (op 4282) 


s_sub_u32 s54, s52, s53 // (op 4282) 


v_lshl_add_u32 v23, s54, 2, v24 // (op 4282) 


v_mul_lo_u32 v24, v23, s38 // (op 4282) 

v_add_u32 v23, v21, v24 // (op 4282) 


s_mul_i32 s53, s38, s43 // (op 4284) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 4284) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 4284) least significant
s_mul_hi_u32 s55, s38, s42 // (op 4284) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4284) most significant: sum
s_add_u32 s53, s53, s55 // (op 4284) most significant: sum


s_mov_b32 s76, s40 // (op 4284) reg expression
s_mov_b32 s77, s41 // (op 4284) reg expression
s_mov_b32 s78, s52 // (op 4284) reg expression
s_mov_b32 s79, 131072 // (op 4284) call()

v_and_b32 v21, -128, v1 // (op 4301) 
s_mul_i32 s52, s16, s8 // (op 4301) 
s_add_u32 s52, s52, s0 // (op 4301) 
v_sub_u32 v24, v1, v21 // (op 4301) 

s_mul_hi_u32 s53, s52, s13 // (op 4301) 
v_lshrrev_b32 v21, 7, v1 // (op 4301) 
s_sub_u32 s54, s52, s53 // (op 4301) 
v_lshlrev_b32 v25, 1, v21 // (op 4301) 

s_lshr_b32 s55, s54, 1 // (op 4301) 

s_add_u32 s54, s55, s53 // (op 4301) 


s_lshr_b32 s53, s54, s11 // (op 4301) 

s_and_b32 s54, s11, 2147483648 // (op 4301) 
s_cmp_eq_i32 s54, 0 // (op 4301) 
s_mov_b32 s55, scc // (op 4301) 

s_cmp_eq_i32 s55, 0 // (op 4301) 
s_cselect_b32 s54, s52, s53 // (op 4301) 


s_mul_hi_u32 s53, s54, s12 // (op 4301) 
s_sub_u32 s55, s54, s53 // (op 4301) 
s_lshr_b32 s80, s55, 1 // (op 4301) 

s_add_u32 s55, s80, s53 // (op 4301) 


s_lshr_b32 s53, s55, s10 // (op 4301) 

s_and_b32 s55, s10, 2147483648 // (op 4301) 
s_cmp_eq_i32 s55, 0 // (op 4301) 
s_mov_b32 s80, scc // (op 4301) 

s_cmp_eq_i32 s80, 0 // (op 4301) 
s_cselect_b32 s55, s54, s53 // (op 4301) 


v_lshl_add_u32 v21, s55, 7, v24 // (op 4301) 


s_ashr_i64 s[80:81], s[2:3], 7 // (op 4301) 
s_mul_i32 s53, s54, s80 // (op 4301) 


s_sub_u32 s54, s52, s53 // (op 4301) 


v_lshl_add_u32 v24, s54, 2, v25 // (op 4301) 


v_mul_lo_u32 v25, v24, s38 // (op 4301) 

v_add_u32 v24, v21, v25 // (op 4301) 


s_mul_i32 s53, s38, s43 // (op 4303) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 4303) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 4303) least significant
s_mul_hi_u32 s55, s38, s42 // (op 4303) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4303) most significant: sum
s_add_u32 s53, s53, s55 // (op 4303) most significant: sum


s_mov_b32 s76, s40 // (op 4303) reg expression
s_mov_b32 s77, s41 // (op 4303) reg expression
s_mov_b32 s78, s52 // (op 4303) reg expression
s_mov_b32 s79, 131072 // (op 4303) call()

v_lshrrev_b32 v21, 6, v1 // (op 4410) 
s_mul_i32 s52, s16, s8 // (op 4410) 
s_add_u32 s52, s52, s0 // (op 4410) 
v_and_b32 v25, 1, v21 // (op 4410) 

s_mul_hi_u32 s53, s52, s13 // (op 4410) 
v_and_b32 v21, 15, v1 // (op 4410) 
s_sub_u32 s54, s52, s53 // (op 4410) 
v_lshl_add_u32 v26, v25, 4, v21 // (op 4410) 


s_lshr_b32 s55, s54, 1 // (op 4410) 

v_and_b32 v21, 63, v1 // (op 4410) 
s_add_u32 s54, s55, s53 // (op 4410) 


v_lshrrev_b32 v25, 4, v21 // (op 4410) 

s_lshr_b32 s53, s54, s11 // (op 4410) 

s_and_b32 s54, s11, 2147483648 // (op 4410) 
s_cmp_eq_i32 s54, 0 // (op 4410) 
s_mov_b32 s55, scc // (op 4410) 

s_cmp_eq_i32 s55, 0 // (op 4410) 
s_cselect_b32 s54, s52, s53 // (op 4410) 


s_mul_hi_u32 s53, s54, s12 // (op 4410) 
s_sub_u32 s55, s54, s53 // (op 4410) 
s_lshr_b32 s80, s55, 1 // (op 4410) 

s_add_u32 s55, s80, s53 // (op 4410) 


s_lshr_b32 s53, s55, s10 // (op 4410) 

s_and_b32 s55, s10, 2147483648 // (op 4410) 
s_cmp_eq_i32 s55, 0 // (op 4410) 
s_mov_b32 s80, scc // (op 4410) 

s_cmp_eq_i32 s80, 0 // (op 4410) 
s_cselect_b32 s55, s54, s53 // (op 4410) 


s_mul_i32 s53, s55, s14 // (op 4410) 

s_sub_u32 s55, s54, s53 // (op 4410) 

v_lshl_add_u32 v21, s55, 5, v26 // (op 4410) 


v_mul_lo_u32 v26, v21, s44 // (op 4410) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 4410) 
s_mul_i32 s53, s54, s80 // (op 4410) 


s_sub_u32 s54, s52, s53 // (op 4410) 


v_lshl_add_u32 v21, s54, 2, v25 // (op 4410) 


v_add_u32 v25, v21, v26 // (op 4410) 


s_mul_i32 s53, s44, s23 // (op 4412) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4412) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4412) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4412) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4412) most significant: sum
s_add_u32 s53, s53, s55 // (op 4412) most significant: sum


s_mov_b32 s80, s46 // (op 4412) reg expression
s_mov_b32 s81, s47 // (op 4412) reg expression
s_mov_b32 s82, s52 // (op 4412) reg expression
s_mov_b32 s83, 131072 // (op 4412) call()

v_lshrrev_b32 v21, 6, v1 // (op 4429) 
s_mul_i32 s52, s16, s8 // (op 4429) 
s_add_u32 s52, s52, s0 // (op 4429) 
v_and_b32 v26, 1, v21 // (op 4429) 

s_mul_hi_u32 s53, s52, s13 // (op 4429) 
v_and_b32 v21, 15, v1 // (op 4429) 
s_sub_u32 s54, s52, s53 // (op 4429) 
v_lshl_add_u32 v27, v26, 4, v21 // (op 4429) 


s_lshr_b32 s55, s54, 1 // (op 4429) 

v_and_b32 v21, 63, v1 // (op 4429) 
s_add_u32 s54, s55, s53 // (op 4429) 


v_lshrrev_b32 v26, 4, v21 // (op 4429) 

s_lshr_b32 s53, s54, s11 // (op 4429) 

s_and_b32 s54, s11, 2147483648 // (op 4429) 
s_cmp_eq_i32 s54, 0 // (op 4429) 
s_mov_b32 s55, scc // (op 4429) 

s_cmp_eq_i32 s55, 0 // (op 4429) 
s_cselect_b32 s54, s52, s53 // (op 4429) 


s_mul_hi_u32 s53, s54, s12 // (op 4429) 
s_sub_u32 s55, s54, s53 // (op 4429) 
s_lshr_b32 s84, s55, 1 // (op 4429) 

s_add_u32 s55, s84, s53 // (op 4429) 


s_lshr_b32 s53, s55, s10 // (op 4429) 

s_and_b32 s55, s10, 2147483648 // (op 4429) 
s_cmp_eq_i32 s55, 0 // (op 4429) 
s_mov_b32 s84, scc // (op 4429) 

s_cmp_eq_i32 s84, 0 // (op 4429) 
s_cselect_b32 s55, s54, s53 // (op 4429) 


s_mul_i32 s53, s55, s14 // (op 4429) 

s_sub_u32 s55, s54, s53 // (op 4429) 

v_lshl_add_u32 v21, s55, 5, v27 // (op 4429) 


v_mul_lo_u32 v27, v21, s44 // (op 4429) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 4429) 
s_mul_i32 s53, s54, s84 // (op 4429) 


s_sub_u32 s54, s52, s53 // (op 4429) 


v_lshl_add_u32 v21, s54, 2, v26 // (op 4429) 


v_add_u32 v26, v21, v27 // (op 4429) 


s_mul_i32 s53, s44, s23 // (op 4431) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4431) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4431) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4431) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4431) most significant: sum
s_add_u32 s53, s53, s55 // (op 4431) most significant: sum


s_mov_b32 s80, s46 // (op 4431) reg expression
s_mov_b32 s81, s47 // (op 4431) reg expression
s_mov_b32 s82, s52 // (op 4431) reg expression
s_mov_b32 s83, 131072 // (op 4431) call()




buffer_load_ubyte v21, v25, s[80:83], 0 offen offset: 0 // (op 3671) Load value
v_and_b32 v27, -8, v1 // (op 4596) 
v_sub_u32 v36, v1, v27 // (op 4596) 

v_lshlrev_b32 v27, 4, v36 // (op 4596) 

v_lshrrev_b32 v36, 3, v1 // (op 4596) 
v_lshl_add_u32 v37, v36, 7, v27 // (op 4596) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 4596) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5333) 



s_nop 0
s_mov_b32 m0, s52 // (op 3900) reg expression
buffer_load_dwordx4 v32, s[72:75], 0 offen offset: 0 lds // (op 3900) Load value direct to lds

v_and_b32 v27, -8, v1 // (op 4521) 
v_sub_u32 v36, v1, v27 // (op 4521) 

v_lshlrev_b32 v27, 4, v36 // (op 4521) 

v_lshrrev_b32 v36, 3, v1 // (op 4521) 
v_lshl_add_u32 v37, v36, 7, v27 // (op 4521) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 4521) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5314) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 3905) 
v_mov_b64 v[36:37], s[54:55] // (op 3905) convert


v_mov_b64 v[38:39], v[28:29] // (op 3905) reg expression
s_add_u32 s53, 4096, s52 // (op 3905) 
s_mov_b32 m0, s53 // (op 3905) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 3905) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 3905) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 3905) most significant half

s_add_u32 m0, m0, 4096 // (op 3905) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 3905) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 3905) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 3905) most significant half

s_add_u32 m0, m0, 4096 // (op 3905) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 3905) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 3905) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 3905) most significant half

s_add_u32 m0, m0, 4096 // (op 3905) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 3905) Load value direct to lds





v_mov_b64 v[36:37], s[38:39] // (op 3681) convert

v_mov_b32 v27, v23 // (op 3681) reg expression
buffer_load_ubyte v38, v27, s[76:79], 0 offen offset: 0 // (op 3681) Load value
v_add_u32 v27, v27, v36 // (op 3681) 
buffer_load_ubyte v39, v27, s[76:79], 0 offen offset: 0 // (op 3681) Load value





buffer_load_ubyte v27, v25, s[80:83], 0 offen offset:4 // (op 3685) Load value

v_and_b32 v25, -8, v1 // (op 5180) 
v_sub_u32 v36, v1, v25 // (op 5180) 

v_lshlrev_b32 v25, 4, v36 // (op 5180) 

v_lshrrev_b32 v36, 3, v1 // (op 5180) 
v_lshl_add_u32 v37, v36, 7, v25 // (op 5180) 


s_nop 0
v_readfirstlane_b32 s52, v37 // (op 5180) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5463) 


s_nop 0
s_add_u32 s53, 20480, s52 // (op 3910) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s53 // (op 3910) reg expression
s_mov_b32 s54, 128 // (op 3910) call()
buffer_load_dwordx4 v32, s[72:75], s54 offen offset: 0 lds // (op 3910) Load value direct to lds




v_and_b32 v25, -8, v1 // (op 5136) 
v_sub_u32 v32, v1, v25 // (op 5136) 

v_lshlrev_b32 v25, 4, v32 // (op 5136) 

v_lshrrev_b32 v32, 3, v1 // (op 5136) 
v_lshl_add_u32 v33, v32, 7, v25 // (op 5136) 


s_nop 0
v_readfirstlane_b32 s52, v33 // (op 5136) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5454) 

s_lshl_b64 s[54:55], s[30:31], 5 // (op 3912) 
v_mov_b64 v[32:33], s[54:55] // (op 3912) convert


s_lshl_b64 s[54:55], s[30:31], 7 // (op 3912) 
v_mov_b32 v25, s54 // (op 3912) 
v_mov_b32 v36, s55 // (op 3912) 
v_add_co_u32 v40, s[84:85], v28, v25 // (op 3912) least significant half
v_addc_co_u32 v41, s[84:85], v29, v36, s[84:85] // (op 3912) most significant half




s_add_u32 s53, 24576, s52 // (op 3912) 
s_mov_b32 m0, s53 // (op 3912) reg expression
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 3912) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v32 // (op 3912) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 3912) most significant half

s_add_u32 m0, m0, 4096 // (op 3912) 
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 3912) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v32 // (op 3912) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 3912) most significant half

s_add_u32 m0, m0, 4096 // (op 3912) 
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 3912) Load value direct to lds
v_add_co_u32 v40, s[54:55], v40, v32 // (op 3912) least significant half
v_addc_co_u32 v41, s[54:55], 0, v41, s[54:55] // (op 3912) most significant half

s_add_u32 m0, m0, 4096 // (op 3912) 
buffer_load_dwordx4 v40, s[68:71], 0 offen offset: 0 lds // (op 3912) Load value direct to lds






v_mov_b64 v[28:29], s[38:39] // (op 3695) convert

s_lshl_b32 s52, s38, 2 // (op 3695) 
v_add_u32 v25, v23, s52 // (op 3695) 

buffer_load_ubyte v32, v25, s[76:79], 0 offen offset: 0 // (op 3695) Load value
v_add_u32 v25, v25, v28 // (op 3695) 
buffer_load_ubyte v33, v25, s[76:79], 0 offen offset: 0 // (op 3695) Load value



v_and_b32 v23, -128, v1 // (op 4662) 
v_sub_u32 v25, v1, v23 // (op 4662) 

v_lshrrev_b32 v23, 7, v1 // (op 4662) 
v_lshl_add_u32 v28, v23, 8, v25 // (op 4662) 


s_barrier  // (op 5352) 



s_waitcnt vmcnt(9)
ds_write_b8 v28, v38 offset:40960 // (op 3699) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v28, v39 offset:41088 // (op 3699) Store local data 

s_barrier  // (op 5330) 



ds_read_b128 v[40:43], v19 // (op 3732) Load local data 
ds_read_b128 v[44:47], v19 offset:64 // (op 3732) Load local data 
s_barrier  // (op 5311) 



ds_read_b64_tr_b8 v[48:49], v16 offset:4096 // (op 3713) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v16 offset:5120 // (op 3713) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v16 offset:12288 // (op 3713) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v16 offset:13312 // (op 3713) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 5349) 


ds_read_u8 v23, v17 offset:40960 // (op 3736) Load local data 



ds_read_b64_tr_b8 v[56:57], v16 offset:4128 // (op 3718) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v16 offset:5152 // (op 3718) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v16 offset:12320 // (op 3718) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v16 offset:13344 // (op 3718) Transpose load local data 


ds_read_u8 v25, v17 offset:40992 // (op 3741) Load local data 



ds_read_b64_tr_b8 v[64:65], v16 offset:4160 // (op 3723) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v16 offset:5184 // (op 3723) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v16 offset:12352 // (op 3723) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v16 offset:13376 // (op 3723) Transpose load local data 


ds_read_u8 v28, v17 offset:41024 // (op 3746) Load local data 



ds_read_b64_tr_b8 v[72:73], v16 offset:4192 // (op 3728) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v16 offset:5216 // (op 3728) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v16 offset:12384 // (op 3728) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v16 offset:13408 // (op 3728) Transpose load local data 


ds_read_u8 v29, v17 offset:41056 // (op 3751) Load local data 
v_and_b32 v36, -8, v1 // (op 4493) 
v_sub_u32 v37, v1, v36 // (op 4493) 

v_lshlrev_b32 v36, 4, v37 // (op 4493) 

v_lshrrev_b32 v37, 3, v1 // (op 4493) 
v_lshl_add_u32 v80, v37, 7, v36 // (op 4493) 


s_nop 0
v_readfirstlane_b32 s52, v80 // (op 4493) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v36, -8, v1 // (op 4568) 
v_sub_u32 v37, v1, v36 // (op 4568) 

v_lshlrev_b32 v36, 4, v37 // (op 4568) 

v_lshrrev_b32 v37, 3, v1 // (op 4568) 
v_lshl_add_u32 v80, v37, 7, v36 // (op 4568) 


s_nop 0
v_readfirstlane_b32 s53, v80 // (op 4568) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v36, -128, v1 // (op 4633) 
v_sub_u32 v37, v1, v36 // (op 4633) 

v_lshrrev_b32 v36, 7, v1 // (op 4633) 
v_lshl_add_u32 v80, v36, 8, v37 // (op 4633) 


v_and_b32 v36, -8, v1 // (op 5121) 
v_sub_u32 v37, v1, v36 // (op 5121) 

v_lshlrev_b32 v36, 4, v37 // (op 5121) 

v_lshrrev_b32 v37, 3, v1 // (op 5121) 
v_lshl_add_u32 v81, v37, 7, v36 // (op 5121) 


s_nop 0
v_readfirstlane_b32 s54, v81 // (op 5121) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v36, -8, v1 // (op 5165) 
v_sub_u32 v37, v1, v36 // (op 5165) 

v_lshlrev_b32 v36, 4, v37 // (op 5165) 

v_lshrrev_b32 v37, 3, v1 // (op 5165) 
v_lshl_add_u32 v81, v37, 7, v36 // (op 5165) 


s_nop 0
v_readfirstlane_b32 s55, v81 // (op 5165) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v36, -128, v1 // (op 5204) 
v_sub_u32 v37, v1, v36 // (op 5204) 

v_lshrrev_b32 v36, 7, v1 // (op 5204) 
v_lshl_add_u32 v81, v36, 8, v37 // (op 5204) 


s_mov_b32 s1, 0 // (op 1808) call()
s_and_b32 s84, s7, 4294967294 // (op 1807) 
s_cmp_lt_u32 s1, s84 // (op 1807) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1807 // (op 1807) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1807 if false)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1807:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[40:47], a[0:3], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 84) 
s_add_u32 s1, s1, 2 // (op 1809) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[40:47], a[4:7], v25, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1519) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:71], v[40:47], a[8:11], v28, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1681) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:79], v[40:47], a[12:15], v29, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1843) 


v_mov_b32 v36, v26 // (op 82) 

buffer_load_ubyte v21, v36, s[80:83], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 5336) 


v_mov_b64 v[82:83], v[34:35] // (op 3825) 

s_mov_b32 m0, s53 // (op 3825) reg expression
s_mov_b32 s84, 256 // (op 3825) call()
buffer_load_dwordx4 v82, s[72:75], s84 offen offset: 0 lds // (op 3825) Load value direct to lds

s_barrier  // (op 5317) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 3820) 
v_mov_b64 v[84:85], s[84:85] // (op 3820) convert


v_mov_b64 v[86:87], v[30:31] // (op 3820) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 3820) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 3820) 

v_mov_b32 v37, s86 // (op 3820) 
v_mov_b32 v88, s87 // (op 3820) 
v_add_co_u32 v90, s[84:85], v86, v37 // (op 3820) least significant half
v_addc_co_u32 v91, s[84:85], v87, v88, s[84:85] // (op 3820) most significant half




s_add_u32 s84, 4096, s52 // (op 3820) 
s_nop 0
s_mov_b32 m0, s84 // (op 3820) reg expression
buffer_load_dwordx4 v90, s[68:71], 0 offen offset: 0 lds // (op 3820) Load value direct to lds
v_add_co_u32 v90, s[86:87], v90, v84 // (op 3820) least significant half
v_addc_co_u32 v91, s[86:87], 0, v91, s[86:87] // (op 3820) most significant half

s_add_u32 m0, m0, 4096 // (op 3820) 
buffer_load_dwordx4 v90, s[68:71], 0 offen offset: 0 lds // (op 3820) Load value direct to lds
v_add_co_u32 v90, s[86:87], v90, v84 // (op 3820) least significant half
v_addc_co_u32 v91, s[86:87], 0, v91, s[86:87] // (op 3820) most significant half

s_add_u32 m0, m0, 4096 // (op 3820) 
buffer_load_dwordx4 v90, s[68:71], 0 offen offset: 0 lds // (op 3820) Load value direct to lds
v_add_co_u32 v90, s[86:87], v90, v84 // (op 3820) least significant half
v_addc_co_u32 v91, s[86:87], 0, v91, s[86:87] // (op 3820) most significant half

s_add_u32 m0, m0, 4096 // (op 3820) 
buffer_load_dwordx4 v90, s[68:71], 0 offen offset: 0 lds // (op 3820) Load value direct to lds




v_mov_b64 v[84:85], s[38:39] // (op 17) convert

v_mov_b32 v37, v24 // (op 17) 
s_lshl_b32 s84, s38, 3 // (op 17) 
v_add_u32 v88, v37, s84 // (op 17) 

buffer_load_ubyte v38, v88, s[76:79], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v88, v88, v84 // (op 17) 
buffer_load_ubyte v39, v88, s[76:79], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 5472) 



s_waitcnt vmcnt(9)
ds_write_b8 v81, v32 offset:41472 // (op 1409) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v81, v33 offset:41600 // (op 1409) Store local data 
s_barrier  // (op 5460) 



ds_read_b128 v[40:43], v6 offset:20480 // (op 1402) Load local data 
ds_read_b128 v[44:47], v6 offset:20544 // (op 1402) Load local data 
s_barrier  // (op 5451) 



ds_read_b64_tr_b8 v[48:49], v12 offset:24576 // (op 1400) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v12 offset:25600 // (op 1400) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v12 offset:32768 // (op 1400) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v12 offset:33792 // (op 1400) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 5469) 


ds_read_u8 v23, v13 offset:41472 // (op 1404) Load local data 



ds_read_b64_tr_b8 v[56:57], v12 offset:24608 // (op 1526) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v12 offset:25632 // (op 1526) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v12 offset:32800 // (op 1526) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v12 offset:33824 // (op 1526) Transpose load local data 


ds_read_u8 v25, v13 offset:41504 // (op 1530) Load local data 



ds_read_b64_tr_b8 v[64:65], v12 offset:24640 // (op 1688) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v12 offset:25664 // (op 1688) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v12 offset:32832 // (op 1688) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v12 offset:33856 // (op 1688) Transpose load local data 


ds_read_u8 v28, v13 offset:41536 // (op 1692) Load local data 



ds_read_b64_tr_b8 v[72:73], v12 offset:24672 // (op 1850) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v12 offset:25696 // (op 1850) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v12 offset:32864 // (op 1850) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v12 offset:33888 // (op 1850) Transpose load local data 


ds_read_u8 v29, v13 offset:41568 // (op 1854) Load local data 
s_barrier  // (op 5667) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[40:47], a[0:3], v23, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1407) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[40:47], a[4:7], v25, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1520) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:71], v[40:47], a[8:11], v28, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1682) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:79], v[40:47], a[12:15], v29, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1844) 



buffer_load_ubyte v27, v36, s[80:83], 0 offen offset:12 // (op 1406) Load value

s_barrier  // (op 5466) 
v_add_u32 v26, 8, v26 // (op 4421) 


s_add_u32 s84, 20480, s55 // (op 3881) 

s_mov_b32 m0, s84 // (op 3881) reg expression
s_mov_b32 s85, 384 // (op 3881) call()
buffer_load_dwordx4 v82, s[72:75], s85 offen offset: 0 lds // (op 3881) Load value direct to lds



s_barrier  // (op 5457) 
v_add_co_u32 v34, vcc, 256, v34 // (op 4165) least significant half

v_addc_co_u32 v35, vcc, 0, v35, vcc // (op 4165) most significant half
s_lshl_b64 s[84:85], s[30:31], 5 // (op 3876) 
v_mov_b64 v[82:83], s[84:85] // (op 3876) convert


s_lshl_b64 s[84:85], s[30:31], 7 // (op 3876) 
s_mul_i32 s87, 3, s85 // (op 3876) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 3876) least significant
s_mul_hi_u32 s88, 3, s84 // (op 3876) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 3876) most significant: sum


v_mov_b32 v36, s86 // (op 3876) 
v_mov_b32 v84, s87 // (op 3876) 
v_add_co_u32 v88, s[84:85], v86, v36 // (op 3876) least significant half
v_addc_co_u32 v89, s[84:85], v87, v84, s[84:85] // (op 3876) most significant half




s_add_u32 s84, 24576, s54 // (op 3876) 
s_nop 0
s_mov_b32 m0, s84 // (op 3876) reg expression
buffer_load_dwordx4 v88, s[68:71], 0 offen offset: 0 lds // (op 3876) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v82 // (op 3876) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3876) most significant half

s_add_u32 m0, m0, 4096 // (op 3876) 
buffer_load_dwordx4 v88, s[68:71], 0 offen offset: 0 lds // (op 3876) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v82 // (op 3876) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3876) most significant half

s_add_u32 m0, m0, 4096 // (op 3876) 
buffer_load_dwordx4 v88, s[68:71], 0 offen offset: 0 lds // (op 3876) Load value direct to lds
v_add_co_u32 v88, s[86:87], v88, v82 // (op 3876) least significant half
v_addc_co_u32 v89, s[86:87], 0, v89, s[86:87] // (op 3876) most significant half

s_add_u32 m0, m0, 4096 // (op 3876) 
buffer_load_dwordx4 v88, s[68:71], 0 offen offset: 0 lds // (op 3876) Load value direct to lds





v_mov_b64 v[82:83], s[38:39] // (op 1411) convert

s_lshl_b32 s84, s38, 2 // (op 1411) 
s_mul_i32 s85, 3, s84 // (op 1411) 

v_add_u32 v36, v37, s85 // (op 1411) 

s_lshl_b64 s[84:85], s[30:31], 7 // (op 4037) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 4037) 

v_mov_b32 v84, s86 // (op 4037) 
v_mov_b32 v85, s87 // (op 4037) 
v_add_co_u32 v30, s[84:85], v30, v84 // (op 4037) least significant half
v_addc_co_u32 v31, s[84:85], v31, v85, s[84:85] // (op 4037) most significant half




buffer_load_ubyte v32, v36, s[76:79], 0 offen offset: 0 // (op 1411) Load value
v_add_u32 v36, v36, v82 // (op 1411) 
buffer_load_ubyte v33, v36, s[76:79], 0 offen offset: 0 // (op 1411) Load value



s_lshl_b32 s84, s38, 3 // (op 4293) 
s_barrier  // (op 5355) 
v_add_u32 v24, v24, s84 // (op 4293) 




s_waitcnt vmcnt(9)
ds_write_b8 v80, v38 offset:40960 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v80, v39 offset:41088 // (op 21) Store local data 
s_barrier  // (op 5342) 



ds_read_b128 v[40:43], v3 // (op 72) Load local data 
ds_read_b128 v[44:47], v3 offset:64 // (op 72) Load local data 
s_barrier  // (op 5323) 



ds_read_b64_tr_b8 v[48:49], v0 offset:4096 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:5120 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:12288 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:13312 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 5362) 


ds_read_u8 v23, v7 offset:40960 // (op 77) Load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:4128 // (op 1513) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:5152 // (op 1513) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:12320 // (op 1513) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:13344 // (op 1513) Transpose load local data 


ds_read_u8 v25, v7 offset:40992 // (op 1517) Load local data 



ds_read_b64_tr_b8 v[64:65], v0 offset:4160 // (op 1675) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v0 offset:5184 // (op 1675) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:12352 // (op 1675) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:13376 // (op 1675) Transpose load local data 


ds_read_u8 v28, v7 offset:41024 // (op 1679) Load local data 



ds_read_b64_tr_b8 v[72:73], v0 offset:4192 // (op 1837) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:5216 // (op 1837) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:12384 // (op 1837) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v0 offset:13408 // (op 1837) Transpose load local data 


ds_read_u8 v29, v7 offset:41056 // (op 1841) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 1807) 
s_cmp_lt_u32 s1, s84 // (op 1807) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1807 // (op 1807) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_1807 if true)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_1807:















s_barrier  // (op 5651) 
s_and_b32 s52, s7, 4294967294 // (op 3949) 
v_and_b32 v21, -8, v1 // (op 3949) 
s_add_u32 s53, s0, s52 // (op 3949) 

v_sub_u32 v26, v1, v21 // (op 3949) 

v_lshlrev_b32 v21, 4, v26 // (op 3949) 

s_mul_i32 s52, s16, s8 // (op 3949) 
s_add_u32 s52, s52, s53 // (op 3949) 

v_lshrrev_b32 v26, 3, v1 // (op 3949) 
s_mul_hi_u32 s53, s52, s13 // (op 3949) 
s_sub_u32 s54, s52, s53 // (op 3949) 
s_lshr_b32 s55, s54, 1 // (op 3949) 

s_add_u32 s54, s55, s53 // (op 3949) 


s_lshr_b32 s53, s54, s11 // (op 3949) 

s_and_b32 s54, s11, 2147483648 // (op 3949) 
s_cmp_eq_i32 s54, 0 // (op 3949) 
s_mov_b32 s55, scc // (op 3949) 

s_cmp_eq_i32 s55, 0 // (op 3949) 
s_cselect_b32 s54, s52, s53 // (op 3949) 


s_mul_hi_u32 s53, s54, s12 // (op 3949) 
s_sub_u32 s55, s54, s53 // (op 3949) 
s_lshr_b32 s84, s55, 1 // (op 3949) 

s_add_u32 s55, s84, s53 // (op 3949) 


s_lshr_b32 s53, s55, s10 // (op 3949) 

s_and_b32 s55, s10, 2147483648 // (op 3949) 
s_cmp_eq_i32 s55, 0 // (op 3949) 
s_mov_b32 s84, scc // (op 3949) 

s_cmp_eq_i32 s84, 0 // (op 3949) 
s_cselect_b32 s55, s54, s53 // (op 3949) 


v_lshl_add_u32 v23, s55, 7, v21 // (op 3949) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 3949) 
v_mov_b32 v31, 0 // (op 3949) convert
v_mov_b32 v30, v23 // (op 3949) convert

s_mul_i32 s53, s54, s84 // (op 3949) 


s_sub_u32 s54, s52, s53 // (op 3949) 


v_lshl_add_u32 v21, s54, 7, v26 // (op 3949) 


v_mul_lo_u32 v35, v21, s31 // (op 3949) most significant: low of low * high
v_mul_hi_u32 v26, v21, s30 // (op 3949) most significant: high of low * low
v_mul_lo_u32 v34, v21, s30 // (op 3949) least significant: low of low * low
v_add_u32 v35, v35, v26 // (op 3949) most significant: sum


v_add_co_u32 v36, s[52:53], v30, v34 // (op 3949) least significant half
v_addc_co_u32 v37, s[52:53], v31, v35, s[52:53] // (op 3949) most significant half



s_mul_i32 s53, s30, s3 // (op 3951) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 3951) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 3951) least significant
s_mul_hi_u32 s55, s30, s2 // (op 3951) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3951) most significant: sum
s_add_u32 s53, s53, s55 // (op 3951) most significant: sum


s_mov_b32 s68, s32 // (op 3951) reg expression
s_mov_b32 s69, s33 // (op 3951) reg expression
s_mov_b32 s70, s52 // (op 3951) reg expression
s_mov_b32 s71, 131072 // (op 3951) call()

s_and_b32 s52, s7, 4294967294 // (op 4089) 
v_lshrrev_b32 v21, 3, v1 // (op 4089) 
s_add_u32 s53, s0, s52 // (op 4089) 

v_and_b32 v26, -8, v1 // (op 4089) 
v_sub_u32 v23, v1, v26 // (op 4089) 

s_mul_i32 s52, s16, s8 // (op 4089) 
s_add_u32 s52, s52, s53 // (op 4089) 

v_lshlrev_b32 v24, 4, v23 // (op 4089) 

s_mul_hi_u32 s53, s52, s13 // (op 4089) 
s_sub_u32 s54, s52, s53 // (op 4089) 
s_lshr_b32 s55, s54, 1 // (op 4089) 

s_add_u32 s54, s55, s53 // (op 4089) 


s_lshr_b32 s53, s54, s11 // (op 4089) 

s_and_b32 s54, s11, 2147483648 // (op 4089) 
s_cmp_eq_i32 s54, 0 // (op 4089) 
s_mov_b32 s55, scc // (op 4089) 

s_cmp_eq_i32 s55, 0 // (op 4089) 
s_cselect_b32 s54, s52, s53 // (op 4089) 


s_mul_hi_u32 s53, s54, s12 // (op 4089) 
s_sub_u32 s55, s54, s53 // (op 4089) 
s_lshr_b32 s84, s55, 1 // (op 4089) 

s_add_u32 s55, s84, s53 // (op 4089) 


s_lshr_b32 s53, s55, s10 // (op 4089) 

s_and_b32 s55, s10, 2147483648 // (op 4089) 
s_cmp_eq_i32 s55, 0 // (op 4089) 
s_mov_b32 s84, scc // (op 4089) 

s_cmp_eq_i32 s84, 0 // (op 4089) 
s_cselect_b32 s55, s54, s53 // (op 4089) 


s_mul_i32 s53, s55, s14 // (op 4089) 

s_sub_u32 s55, s54, s53 // (op 4089) 

v_lshl_add_u32 v23, s55, 5, v21 // (op 4089) 


v_mul_lo_u32 v31, v23, s35 // (op 4089) most significant: low of low * high
v_mul_hi_u32 v21, v23, s34 // (op 4089) most significant: high of low * low
v_mul_lo_u32 v30, v23, s34 // (op 4089) least significant: low of low * low
v_add_u32 v31, v31, v21 // (op 4089) most significant: sum


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4089) 
s_mul_i32 s53, s54, s84 // (op 4089) 


s_sub_u32 s54, s52, s53 // (op 4089) 


v_lshl_add_u32 v21, s54, 7, v24 // (op 4089) 


v_mov_b32 v35, 0 // (op 4089) convert
v_mov_b32 v34, v21 // (op 4089) convert

v_add_co_u32 v38, s[52:53], v34, v30 // (op 4089) least significant half
v_addc_co_u32 v39, s[52:53], v35, v31, s[52:53] // (op 4089) most significant half



s_mul_i32 s53, s34, s23 // (op 4091) most significant: low of low * high
s_mul_i32 s54, s35, s22 // (op 4091) most significant: low of high * low
s_mul_i32 s52, s34, s22 // (op 4091) least significant
s_mul_hi_u32 s55, s34, s22 // (op 4091) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4091) most significant: sum
s_add_u32 s53, s53, s55 // (op 4091) most significant: sum


s_mov_b32 s72, s36 // (op 4091) reg expression
s_mov_b32 s73, s37 // (op 4091) reg expression
s_mov_b32 s74, s52 // (op 4091) reg expression
s_mov_b32 s75, 131072 // (op 4091) call()

s_and_b32 s52, s7, 4294967294 // (op 4217) 
v_and_b32 v21, -128, v1 // (op 4217) 
s_add_u32 s53, s0, s52 // (op 4217) 

v_sub_u32 v26, v1, v21 // (op 4217) 

v_lshrrev_b32 v21, 7, v1 // (op 4217) 
s_mul_i32 s52, s16, s8 // (op 4217) 
s_add_u32 s52, s52, s53 // (op 4217) 

v_lshlrev_b32 v23, 1, v21 // (op 4217) 

s_mul_hi_u32 s53, s52, s13 // (op 4217) 
s_sub_u32 s54, s52, s53 // (op 4217) 
s_lshr_b32 s55, s54, 1 // (op 4217) 

s_add_u32 s54, s55, s53 // (op 4217) 


s_lshr_b32 s53, s54, s11 // (op 4217) 

s_and_b32 s54, s11, 2147483648 // (op 4217) 
s_cmp_eq_i32 s54, 0 // (op 4217) 
s_mov_b32 s55, scc // (op 4217) 

s_cmp_eq_i32 s55, 0 // (op 4217) 
s_cselect_b32 s54, s52, s53 // (op 4217) 


s_mul_hi_u32 s53, s54, s12 // (op 4217) 
s_sub_u32 s55, s54, s53 // (op 4217) 
s_lshr_b32 s84, s55, 1 // (op 4217) 

s_add_u32 s55, s84, s53 // (op 4217) 


s_lshr_b32 s53, s55, s10 // (op 4217) 

s_and_b32 s55, s10, 2147483648 // (op 4217) 
s_cmp_eq_i32 s55, 0 // (op 4217) 
s_mov_b32 s84, scc // (op 4217) 

s_cmp_eq_i32 s84, 0 // (op 4217) 
s_cselect_b32 s55, s54, s53 // (op 4217) 


v_lshl_add_u32 v21, s55, 7, v26 // (op 4217) 


s_ashr_i64 s[84:85], s[2:3], 7 // (op 4217) 
s_mul_i32 s53, s54, s84 // (op 4217) 


s_sub_u32 s54, s52, s53 // (op 4217) 


v_lshl_add_u32 v24, s54, 2, v23 // (op 4217) 


v_mul_lo_u32 v23, v24, s38 // (op 4217) 

v_add_u32 v24, v21, v23 // (op 4217) 


s_mul_i32 s53, s38, s43 // (op 4219) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 4219) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 4219) least significant
s_mul_hi_u32 s55, s38, s42 // (op 4219) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4219) most significant: sum
s_add_u32 s53, s53, s55 // (op 4219) most significant: sum


s_mov_b32 s76, s40 // (op 4219) reg expression
s_mov_b32 s77, s41 // (op 4219) reg expression
s_mov_b32 s78, s52 // (op 4219) reg expression
s_mov_b32 s79, 131072 // (op 4219) call()

s_and_b32 s52, s7, 4294967294 // (op 4345) 
v_lshrrev_b32 v21, 6, v1 // (op 4345) 
s_add_u32 s53, s0, s52 // (op 4345) 

v_and_b32 v23, 1, v21 // (op 4345) 

v_and_b32 v21, 15, v1 // (op 4345) 
s_mul_i32 s52, s16, s8 // (op 4345) 
s_add_u32 s52, s52, s53 // (op 4345) 

v_lshl_add_u32 v26, v23, 4, v21 // (op 4345) 


s_mul_hi_u32 s53, s52, s13 // (op 4345) 
v_and_b32 v21, 63, v1 // (op 4345) 
s_sub_u32 s54, s52, s53 // (op 4345) 
v_lshrrev_b32 v23, 4, v21 // (op 4345) 

s_lshr_b32 s55, s54, 1 // (op 4345) 

s_add_u32 s54, s55, s53 // (op 4345) 


s_lshr_b32 s53, s54, s11 // (op 4345) 

s_and_b32 s54, s11, 2147483648 // (op 4345) 
s_cmp_eq_i32 s54, 0 // (op 4345) 
s_mov_b32 s55, scc // (op 4345) 

s_cmp_eq_i32 s55, 0 // (op 4345) 
s_cselect_b32 s54, s52, s53 // (op 4345) 


s_mul_hi_u32 s53, s54, s12 // (op 4345) 
s_sub_u32 s55, s54, s53 // (op 4345) 
s_lshr_b32 s84, s55, 1 // (op 4345) 

s_add_u32 s55, s84, s53 // (op 4345) 


s_lshr_b32 s53, s55, s10 // (op 4345) 

s_and_b32 s55, s10, 2147483648 // (op 4345) 
s_cmp_eq_i32 s55, 0 // (op 4345) 
s_mov_b32 s84, scc // (op 4345) 

s_cmp_eq_i32 s84, 0 // (op 4345) 
s_cselect_b32 s55, s54, s53 // (op 4345) 


s_mul_i32 s53, s55, s14 // (op 4345) 

s_sub_u32 s55, s54, s53 // (op 4345) 

v_lshl_add_u32 v21, s55, 5, v26 // (op 4345) 


v_mul_lo_u32 v26, v21, s44 // (op 4345) 

s_ashr_i64 s[84:85], s[2:3], 7 // (op 4345) 
s_mul_i32 s53, s54, s84 // (op 4345) 


s_sub_u32 s54, s52, s53 // (op 4345) 


v_lshl_add_u32 v21, s54, 2, v23 // (op 4345) 


v_add_u32 v23, v21, v26 // (op 4345) 


s_mul_i32 s53, s44, s23 // (op 4347) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 4347) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 4347) least significant
s_mul_hi_u32 s55, s44, s22 // (op 4347) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 4347) most significant: sum
s_add_u32 s53, s53, s55 // (op 4347) most significant: sum


s_mov_b32 s80, s46 // (op 4347) reg expression
s_mov_b32 s81, s47 // (op 4347) reg expression
s_mov_b32 s82, s52 // (op 4347) reg expression
s_mov_b32 s83, 131072 // (op 4347) call()

v_and_b32 v21, -8, v1 // (op 4506) 
v_sub_u32 v26, v1, v21 // (op 4506) 

v_lshlrev_b32 v21, 4, v26 // (op 4506) 

v_lshrrev_b32 v26, 3, v1 // (op 4506) 
v_lshl_add_u32 v30, v26, 7, v21 // (op 4506) 


s_nop 0
v_readfirstlane_b32 s52, v30 // (op 4506) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 4581) 
v_sub_u32 v26, v1, v21 // (op 4581) 

v_lshlrev_b32 v21, 4, v26 // (op 4581) 

v_lshrrev_b32 v26, 3, v1 // (op 4581) 
v_lshl_add_u32 v30, v26, 7, v21 // (op 4581) 


s_nop 0
v_readfirstlane_b32 s53, v30 // (op 4581) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -128, v1 // (op 4646) 
v_sub_u32 v26, v1, v21 // (op 4646) 

v_lshrrev_b32 v21, 7, v1 // (op 4646) 
v_lshl_add_u32 v30, v21, 8, v26 // (op 4646) 


s_and_b32 s54, s7, 4294967294 // (op 1813) 
s_cmp_lt_u32 s54, s7 // (op 1812) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1812 // (op 1812) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1812 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1812:
s_barrier  // (op 5320) 
s_add_u32 s54, s54, 1 // (op 1814) 

s_lshl_b64 s[84:85], s[30:31], 5 // (op 3863) 
v_mov_b64 v[28:29], s[84:85] // (op 3863) convert


v_mov_b64 v[26:27], v[36:37] // (op 3863) 
v_mov_b64 v[34:35], v[26:27] // (op 3863) reg expression
s_add_u32 s55, 4096, s52 // (op 3863) 
s_mov_b32 m0, s55 // (op 3863) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 3863) Load value direct to lds
v_add_co_u32 v34, s[84:85], v34, v28 // (op 3863) least significant half
v_addc_co_u32 v35, s[84:85], 0, v35, s[84:85] // (op 3863) most significant half

s_add_u32 m0, m0, 4096 // (op 3863) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 3863) Load value direct to lds
v_add_co_u32 v34, s[84:85], v34, v28 // (op 3863) least significant half
v_addc_co_u32 v35, s[84:85], 0, v35, s[84:85] // (op 3863) most significant half

s_add_u32 m0, m0, 4096 // (op 3863) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 3863) Load value direct to lds
v_add_co_u32 v34, s[84:85], v34, v28 // (op 3863) least significant half
v_addc_co_u32 v35, s[84:85], 0, v35, s[84:85] // (op 3863) most significant half

s_add_u32 m0, m0, 4096 // (op 3863) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 3863) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 7 // (op 3939) 
v_mov_b32 v21, s84 // (op 3939) 
v_mov_b32 v25, s85 // (op 3939) 
v_add_co_u32 v36, s[86:87], v36, v21 // (op 3939) least significant half
v_addc_co_u32 v37, s[86:87], v37, v25, s[86:87] // (op 3939) most significant half




s_barrier  // (op 5339) 


v_mov_b64 v[28:29], v[38:39] // (op 3869) 

s_mov_b32 m0, s53 // (op 3869) reg expression
buffer_load_dwordx4 v28, s[72:75], 0 offen offset: 0 lds // (op 3869) Load value direct to lds

v_add_co_u32 v38, vcc, 128, v38 // (op 4081) least significant half
v_addc_co_u32 v39, vcc, 0, v39, vcc // (op 4081) most significant half

v_mov_b64 v[28:29], s[38:39] // (op 1353) convert

v_mov_b32 v21, v24 // (op 1353) 
v_mov_b32 v25, v21 // (op 1353) reg expression
buffer_load_ubyte v26, v25, s[76:79], 0 offen offset: 0 // (op 1353) Load value
v_add_u32 v25, v25, v28 // (op 1353) 
buffer_load_ubyte v27, v25, s[76:79], 0 offen offset: 0 // (op 1353) Load value



s_lshl_b32 s55, s38, 2 // (op 4209) 
v_add_u32 v24, v24, s55 // (op 4209) 

s_barrier  // (op 5358) 
s_waitcnt vmcnt(3)
s_barrier  // (op 5326) 






ds_read_b64_tr_b8 v[40:41], v0 offset:4096 // (op 1342) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:5120 // (op 1342) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:12288 // (op 1342) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:13312 // (op 1342) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v30, v26 offset:40960 // (op 1351) Store local data 
s_barrier  // (op 5345) 
s_waitcnt vmcnt(0)
ds_write_b8 v30, v27 offset:41088 // (op 1351) Store local data 




ds_read_b128 v[48:51], v3 // (op 1344) Load local data 
ds_read_b128 v[52:55], v3 offset:64 // (op 1344) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5365) 


ds_read_u8 v21, v7 offset:40960 // (op 1346) Load local data 


v_mov_b32 v25, v23 // (op 1348) 

buffer_load_ubyte v26, v25, s[80:83], 0 offen offset: 0 // (op 1348) Load value

v_add_u32 v23, 4, v23 // (op 4337) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:55], a[0:3], v21, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1349) 
ds_read_b64_tr_b8 v[56:57], v0 offset:4128 // (op 1494) Transpose load local data 


ds_read_b64_tr_b8 v[58:59], v0 offset:5152 // (op 1494) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:12320 // (op 1494) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:13344 // (op 1494) Transpose load local data 


ds_read_u8 v21, v7 offset:40992 // (op 1498) Load local data 



ds_read_b64_tr_b8 v[40:41], v0 offset:4160 // (op 1656) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:63], v[48:55], a[4:7], v21, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1500) 
ds_read_b64_tr_b8 v[42:43], v0 offset:5184 // (op 1656) Transpose load local data 


ds_read_b64_tr_b8 v[44:45], v0 offset:12352 // (op 1656) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:13376 // (op 1656) Transpose load local data 


ds_read_u8 v21, v7 offset:41024 // (op 1660) Load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:4192 // (op 1818) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:5216 // (op 1818) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:47], v[48:55], a[8:11], v21, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1662) 
ds_read_b64_tr_b8 v[60:61], v0 offset:12384 // (op 1818) Transpose load local data 


ds_read_b64_tr_b8 v[62:63], v0 offset:13408 // (op 1818) Transpose load local data 


ds_read_u8 v21, v7 offset:41056 // (op 1822) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:63], v[48:55], a[12:15], v21, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 1824) 




s_cmp_lt_u32 s54, s7 // (op 1812) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1812 // (op 1812) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_1812 if true)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_1812:








s_barrier  // (op 5655) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 280) 
s_cmp_lg_u32 s18, 0 // (op 306) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_306 // (op 306) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_306



v_add_u32 v19, 12288, v20 // (op 2067) 
s_nop 0
buffer_store_dword a12, v19, s[64:67], 0 offen sc1 // (op 2067) Store value

v_add_u32 v19, 12416, v20 // (op 2067) 
buffer_store_dword a13, v19, s[64:67], 0 offen sc1 // (op 2067) Store value

v_add_u32 v19, 12544, v20 // (op 2067) 
buffer_store_dword a14, v19, s[64:67], 0 offen sc1 // (op 2067) Store value

v_add_u32 v19, 12672, v20 // (op 2067) 
buffer_store_dword a15, v19, s[64:67], 0 offen sc1 // (op 2067) Store value




v_add_u32 v19, 8192, v20 // (op 2060) 
buffer_store_dword a8, v19, s[64:67], 0 offen sc1 // (op 2060) Store value

v_add_u32 v19, 8320, v20 // (op 2060) 
buffer_store_dword a9, v19, s[64:67], 0 offen sc1 // (op 2060) Store value

v_add_u32 v19, 8448, v20 // (op 2060) 
buffer_store_dword a10, v19, s[64:67], 0 offen sc1 // (op 2060) Store value

v_add_u32 v19, 8576, v20 // (op 2060) 
buffer_store_dword a11, v19, s[64:67], 0 offen sc1 // (op 2060) Store value




v_add_u32 v19, 4096, v20 // (op 2053) 
buffer_store_dword a4, v19, s[64:67], 0 offen sc1 // (op 2053) Store value

v_add_u32 v19, 4224, v20 // (op 2053) 
buffer_store_dword a5, v19, s[64:67], 0 offen sc1 // (op 2053) Store value

v_add_u32 v19, 4352, v20 // (op 2053) 
buffer_store_dword a6, v19, s[64:67], 0 offen sc1 // (op 2053) Store value

v_add_u32 v19, 4480, v20 // (op 2053) 
buffer_store_dword a7, v19, s[64:67], 0 offen sc1 // (op 2053) Store value




buffer_store_dword a0, v20, s[64:67], 0 offen sc1 // (op 303) Store value
buffer_store_dword a1, v20, s[64:67], 0 offen offset:128 sc1 // (op 303) Store value
buffer_store_dword a2, v20, s[64:67], 0 offen offset:256 sc1 // (op 303) Store value
buffer_store_dword a3, v20, s[64:67], 0 offen offset:384 sc1 // (op 303) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 307) 
v_cmp_eq_i32 vcc, v1, 0 // (op 311) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311 // (op 311) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311
s_mov_b32 s52, 1 // (op 309) call()
s_mov_b32 s55, 0 // (op 310) convert
s_mov_b32 s54, s16 // (op 310) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 310) 
s_add_u32 s84, s48, s54 // (op 310) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 310) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 310) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311 // (op 311) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Store_Flag_311:
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Store_Flag_311:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_306 // (op 306) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_306
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Send_Tile_306:
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Send_Tile_306:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 322) 
s_mov_b32 s52, scc // (op 322) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 322) 
s_mov_b32 s53, s54 // (op 322) convert

s_sub_u32 s54, s53, 1 // (op 322) 

s_cmp_lt_u32 s19, s54 // (op 322) 
s_mov_b32 s53, scc // (op 322) 

s_and_b32 s54, s52, s53 // (op 322) 
s_cmp_lg_u32 s54, 0 // (op 322) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_322 // (op 322) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_322
s_mov_b32 s52, 0 // (op 288) call()
s_mul_i32 s53, s52, s8 // (op 287) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 287) 
s_mov_b32 s84, s54 // (op 287) convert

s_sub_u32 s54, s84, s19 // (op 287) 

s_sub_u32 s55, s54, 1 // (op 287) 

s_cmp_lt_u32 s53, s55 // (op 287) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_287 // (op 287) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_287 if false)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_287:
s_add_u32 s53, s16, 1 // (op 4994) 
v_lshrrev_b32 v16, 5, v1 // (op 4994) 
s_add_u32 s54, s53, s52 // (op 4994) 
s_lshl_b32 s54, s54, 7 // (op 4994) 

v_lshlrev_b32 v17, 2, v16 // (op 4994) 

v_add_lshl_u32 v16, s54, v17, 5 // (op 4994) 


v_and_b32 v17, -32, v1 // (op 4994) 
v_sub_u32 v19, v1, v17 // (op 4994) 

v_add_lshl_u32 v17, v16, v19, 2 // (op 4994) 


s_lshl_b32 s53, s15, 14 // (op 4996) 
s_mov_b32 s64, s28 // (op 4996) reg expression
s_mov_b32 s65, s29 // (op 4996) reg expression
s_mov_b32 s66, s53 // (op 4996) reg expression
s_mov_b32 s67, 131072 // (op 4996) call()

RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_326:
s_add_u32 s53, s16, 1 // (op 324) 
s_add_u32 s54, s53, s52 // (op 324) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 324) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 324) 

s_add_u32 s84, s48, s54 // (op 324) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 324) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 324) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 326) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_326 // (op 326) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_DoWhileTop_Global_sync_spin_loop_326 if true)
s_barrier  // (op 330) 
v_cmp_eq_i32 vcc, v1, 0 // (op 329) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329 // (op 329) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329
s_mov_b32 s53, 0 // (op 327) call()
s_add_u32 s54, s16, 1 // (op 328) 
s_add_u32 s84, s54, s52 // (op 328) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 328) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 328) 

s_add_u32 s54, s48, s84 // (op 328) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 328) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 328) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329 // (op 329) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Wave0_Reset_Flag_329:
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Wave0_Reset_Flag_329:




buffer_load_dword v24, v17, s[64:67], 0 offen // (op 341) Load value
buffer_load_dword v25, v17, s[64:67], 0 offen offset:128 // (op 341) Load value
buffer_load_dword v26, v17, s[64:67], 0 offen offset:256 // (op 341) Load value
buffer_load_dword v27, v17, s[64:67], 0 offen offset:384 // (op 341) Load value
v_accvgpr_read v28, a0 // (op 344) reg expression
v_accvgpr_read v29, a1 // (op 344) reg expression
v_accvgpr_read v30, a2 // (op 344) reg expression
v_accvgpr_read v31, a3 // (op 344) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 346) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 346) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 346) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 346) 

v_accvgpr_write a0, v28 // (op 348) reg expression
v_accvgpr_write a1, v29 // (op 348) reg expression
v_accvgpr_write a2, v30 // (op 348) reg expression
v_accvgpr_write a3, v31 // (op 348) reg expression




v_mov_b32 v16, 4096 // (op 2093) 
v_add_i32 v19, v17, v16 // (op 2093) 

buffer_load_dword v24, v19, s[64:67], 0 offen // (op 2093) Load value

v_mov_b32 v16, 4224 // (op 2093) 
v_add_i32 v23, v17, v16 // (op 2093) 

buffer_load_dword v25, v23, s[64:67], 0 offen // (op 2093) Load value

v_mov_b32 v16, 4352 // (op 2093) 
v_add_i32 v23, v17, v16 // (op 2093) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 2093) Load value

v_mov_b32 v16, 4480 // (op 2093) 
v_add_i32 v23, v17, v16 // (op 2093) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 2093) Load value

v_accvgpr_read v28, a4 // (op 2094) reg expression
v_accvgpr_read v29, a5 // (op 2094) reg expression
v_accvgpr_read v30, a6 // (op 2094) reg expression
v_accvgpr_read v31, a7 // (op 2094) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 2095) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 2095) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 2095) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 2095) 

v_accvgpr_write a4, v28 // (op 2096) reg expression
v_accvgpr_write a5, v29 // (op 2096) reg expression
v_accvgpr_write a6, v30 // (op 2096) reg expression
v_accvgpr_write a7, v31 // (op 2096) reg expression




v_mov_b32 v16, 8192 // (op 2106) 
v_add_i32 v19, v17, v16 // (op 2106) 

buffer_load_dword v24, v19, s[64:67], 0 offen // (op 2106) Load value

v_mov_b32 v16, 8320 // (op 2106) 
v_add_i32 v23, v17, v16 // (op 2106) 

buffer_load_dword v25, v23, s[64:67], 0 offen // (op 2106) Load value

v_mov_b32 v16, 8448 // (op 2106) 
v_add_i32 v23, v17, v16 // (op 2106) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 2106) Load value

v_mov_b32 v16, 8576 // (op 2106) 
v_add_i32 v23, v17, v16 // (op 2106) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 2106) Load value

v_accvgpr_read v28, a8 // (op 2107) reg expression
v_accvgpr_read v29, a9 // (op 2107) reg expression
v_accvgpr_read v30, a10 // (op 2107) reg expression
v_accvgpr_read v31, a11 // (op 2107) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 2108) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 2108) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 2108) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 2108) 

v_accvgpr_write a8, v28 // (op 2109) reg expression
v_accvgpr_write a9, v29 // (op 2109) reg expression
v_accvgpr_write a10, v30 // (op 2109) reg expression
v_accvgpr_write a11, v31 // (op 2109) reg expression




v_mov_b32 v16, 12288 // (op 2119) 
v_add_i32 v19, v17, v16 // (op 2119) 

buffer_load_dword v24, v19, s[64:67], 0 offen // (op 2119) Load value

v_mov_b32 v16, 12416 // (op 2119) 
v_add_i32 v23, v17, v16 // (op 2119) 

buffer_load_dword v25, v23, s[64:67], 0 offen // (op 2119) Load value

v_mov_b32 v16, 12544 // (op 2119) 
v_add_i32 v23, v17, v16 // (op 2119) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 2119) Load value

v_mov_b32 v16, 12672 // (op 2119) 
v_add_i32 v23, v17, v16 // (op 2119) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 2119) Load value

v_accvgpr_read v28, a12 // (op 2120) reg expression
v_accvgpr_read v29, a13 // (op 2120) reg expression
v_accvgpr_read v30, a14 // (op 2120) reg expression
v_accvgpr_read v31, a15 // (op 2120) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v28, v28, v24 // (op 2121) 
s_waitcnt vmcnt(2)
v_add_f32 v29, v29, v25 // (op 2121) 
s_waitcnt vmcnt(1)
v_add_f32 v30, v30, v26 // (op 2121) 
s_waitcnt vmcnt(0)
v_add_f32 v31, v31, v27 // (op 2121) 

v_accvgpr_write a12, v28 // (op 2122) reg expression
v_accvgpr_write a13, v29 // (op 2122) reg expression
v_accvgpr_write a14, v30 // (op 2122) reg expression
v_accvgpr_write a15, v31 // (op 2122) reg expression

s_add_u32 s52, s52, 1 // (op 289) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 287) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 287) 
s_mov_b32 s84, s54 // (op 287) convert

s_sub_u32 s54, s84, s19 // (op 287) 

s_sub_u32 s55, s54, 1 // (op 287) 

s_cmp_lt_u32 s53, s55 // (op 287) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_287 // (op 287) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_ReceiveTileLoop_287 if true)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_ReceiveTileLoop_287:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_322 // (op 322) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_322
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Receive_Tile_322:
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Receive_Tile_322:

s_cmp_eq_i32 s18, 0 // (op 365) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365 // (op 365) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365
v_cmp_eq_f32 vcc, 0.00000, v14 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v24, a0 // (op 46) 
v_accvgpr_read v25, a1 // (op 46) 
v_accvgpr_read v26, a2 // (op 46) 
v_accvgpr_read v27, a3 // (op 46) 
v_mul_f32 v28, v15, v24 // (op 46) 
v_mul_f32 v29, v15, v25 // (op 46) 
v_mul_f32 v30, v15, v26 // (op 46) 
v_mul_f32 v31, v15, v27 // (op 46) 

v_cvt_f16_f32 v24, v28 // (op 52) 
v_cvt_f16_f32 v25, v29 // (op 52) 
v_cvt_f16_f32 v26, v30 // (op 52) 
v_cvt_f16_f32 v27, v31 // (op 52) 


v_pack_b32_f16 v16, v24, v25 // (op 54) 
v_pack_b32_f16 v17, v26, v27 // (op 54) 


buffer_store_dwordx2 v[16:17], v22, s[60:63], 0 offen // (op 54) Store value


v_accvgpr_read v24, a4 // (op 2156) 
v_accvgpr_read v25, a5 // (op 2156) 
v_accvgpr_read v26, a6 // (op 2156) 
v_accvgpr_read v27, a7 // (op 2156) 
v_mul_f32 v28, v15, v24 // (op 2156) 
v_mul_f32 v29, v15, v25 // (op 2156) 
v_mul_f32 v30, v15, v26 // (op 2156) 
v_mul_f32 v31, v15, v27 // (op 2156) 

v_cvt_f16_f32 v24, v28 // (op 2158) 
v_cvt_f16_f32 v25, v29 // (op 2158) 
v_cvt_f16_f32 v26, v30 // (op 2158) 
v_cvt_f16_f32 v27, v31 // (op 2158) 


v_pack_b32_f16 v16, v24, v25 // (op 2159) 
v_pack_b32_f16 v17, v26, v27 // (op 2159) 


buffer_store_dwordx2 v[16:17], v22, s[60:63], 0 offen offset:64 // (op 2159) Store value


v_accvgpr_read v24, a8 // (op 2176) 
v_accvgpr_read v25, a9 // (op 2176) 
v_accvgpr_read v26, a10 // (op 2176) 
v_accvgpr_read v27, a11 // (op 2176) 
v_mul_f32 v28, v15, v24 // (op 2176) 
v_mul_f32 v29, v15, v25 // (op 2176) 
v_mul_f32 v30, v15, v26 // (op 2176) 
v_mul_f32 v31, v15, v27 // (op 2176) 

v_cvt_f16_f32 v24, v28 // (op 2178) 
v_cvt_f16_f32 v25, v29 // (op 2178) 
v_cvt_f16_f32 v26, v30 // (op 2178) 
v_cvt_f16_f32 v27, v31 // (op 2178) 


v_pack_b32_f16 v16, v24, v25 // (op 2179) 
v_pack_b32_f16 v17, v26, v27 // (op 2179) 


buffer_store_dwordx2 v[16:17], v22, s[60:63], 0 offen offset:128 // (op 2179) Store value


v_accvgpr_read v24, a12 // (op 2196) 
v_accvgpr_read v25, a13 // (op 2196) 
v_accvgpr_read v26, a14 // (op 2196) 
v_accvgpr_read v27, a15 // (op 2196) 
v_mul_f32 v28, v15, v24 // (op 2196) 
v_mul_f32 v29, v15, v25 // (op 2196) 
v_mul_f32 v30, v15, v26 // (op 2196) 
v_mul_f32 v31, v15, v27 // (op 2196) 

v_cvt_f16_f32 v24, v28 // (op 2198) 
v_cvt_f16_f32 v25, v29 // (op 2198) 
v_cvt_f16_f32 v26, v30 // (op 2198) 
v_cvt_f16_f32 v27, v31 // (op 2198) 


v_pack_b32_f16 v16, v24, v25 // (op 2199) 
v_pack_b32_f16 v17, v26, v27 // (op 2199) 


buffer_store_dwordx2 v[16:17], v22, s[60:63], 0 offen offset:192 // (op 2199) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[16:17], v18, s[56:59], 0 offen offset:192 // (op 2279) Load value
v_accvgpr_read v24, a12 // (op 2275) 
v_accvgpr_read v25, a13 // (op 2275) 
v_accvgpr_read v26, a14 // (op 2275) 
v_accvgpr_read v27, a15 // (op 2275) 
v_mul_f32 v28, v15, v24 // (op 2275) 
v_mul_f32 v29, v15, v25 // (op 2275) 
v_mul_f32 v30, v15, v26 // (op 2275) 
v_mul_f32 v31, v15, v27 // (op 2275) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 2280) 
v_lshrrev_b32 v21, 16, v16 // (op 2280) 
v_cvt_f32_f16 v21, v21 // (op 2280) 
v_mul_f32 v24, v14, v20 // (op 2280) 
v_mul_f32 v25, v14, v21 // (op 2280) 
v_cvt_f32_f16 v20, v17 // (op 2280) 
v_lshrrev_b32 v21, 16, v17 // (op 2280) 
v_cvt_f32_f16 v21, v21 // (op 2280) 
v_mul_f32 v26, v14, v20 // (op 2280) 
v_mul_f32 v27, v14, v21 // (op 2280) 


v_add_f32 v32, v24, v28 // (op 2276) 
v_add_f32 v33, v25, v29 // (op 2276) 
v_add_f32 v34, v26, v30 // (op 2276) 
v_add_f32 v35, v27, v31 // (op 2276) 


v_cvt_f16_f32 v28, v32 // (op 2277) 
v_cvt_f16_f32 v29, v33 // (op 2277) 
v_cvt_f16_f32 v30, v34 // (op 2277) 
v_cvt_f16_f32 v31, v35 // (op 2277) 


v_pack_b32_f16 v16, v28, v29 // (op 2278) 
v_pack_b32_f16 v17, v30, v31 // (op 2278) 


buffer_store_dwordx2 v[16:17], v22, s[60:63], 0 offen offset:192 // (op 2278) Store value





buffer_load_dwordx2 v[16:17], v18, s[56:59], 0 offen offset:128 // (op 2259) Load value
v_accvgpr_read v24, a8 // (op 2255) 
v_accvgpr_read v25, a9 // (op 2255) 
v_accvgpr_read v26, a10 // (op 2255) 
v_accvgpr_read v27, a11 // (op 2255) 
v_mul_f32 v28, v15, v24 // (op 2255) 
v_mul_f32 v29, v15, v25 // (op 2255) 
v_mul_f32 v30, v15, v26 // (op 2255) 
v_mul_f32 v31, v15, v27 // (op 2255) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 2260) 
v_lshrrev_b32 v21, 16, v16 // (op 2260) 
v_cvt_f32_f16 v21, v21 // (op 2260) 
v_mul_f32 v24, v14, v20 // (op 2260) 
v_mul_f32 v25, v14, v21 // (op 2260) 
v_cvt_f32_f16 v20, v17 // (op 2260) 
v_lshrrev_b32 v21, 16, v17 // (op 2260) 
v_cvt_f32_f16 v21, v21 // (op 2260) 
v_mul_f32 v26, v14, v20 // (op 2260) 
v_mul_f32 v27, v14, v21 // (op 2260) 


v_add_f32 v32, v24, v28 // (op 2256) 
v_add_f32 v33, v25, v29 // (op 2256) 
v_add_f32 v34, v26, v30 // (op 2256) 
v_add_f32 v35, v27, v31 // (op 2256) 


v_cvt_f16_f32 v28, v32 // (op 2257) 
v_cvt_f16_f32 v29, v33 // (op 2257) 
v_cvt_f16_f32 v30, v34 // (op 2257) 
v_cvt_f16_f32 v31, v35 // (op 2257) 


v_pack_b32_f16 v16, v28, v29 // (op 2258) 
v_pack_b32_f16 v17, v30, v31 // (op 2258) 


buffer_store_dwordx2 v[16:17], v22, s[60:63], 0 offen offset:128 // (op 2258) Store value





buffer_load_dwordx2 v[16:17], v18, s[56:59], 0 offen offset:64 // (op 2239) Load value
v_accvgpr_read v24, a4 // (op 2235) 
v_accvgpr_read v25, a5 // (op 2235) 
v_accvgpr_read v26, a6 // (op 2235) 
v_accvgpr_read v27, a7 // (op 2235) 
v_mul_f32 v28, v15, v24 // (op 2235) 
v_mul_f32 v29, v15, v25 // (op 2235) 
v_mul_f32 v30, v15, v26 // (op 2235) 
v_mul_f32 v31, v15, v27 // (op 2235) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 2240) 
v_lshrrev_b32 v21, 16, v16 // (op 2240) 
v_cvt_f32_f16 v21, v21 // (op 2240) 
v_mul_f32 v24, v14, v20 // (op 2240) 
v_mul_f32 v25, v14, v21 // (op 2240) 
v_cvt_f32_f16 v20, v17 // (op 2240) 
v_lshrrev_b32 v21, 16, v17 // (op 2240) 
v_cvt_f32_f16 v21, v21 // (op 2240) 
v_mul_f32 v26, v14, v20 // (op 2240) 
v_mul_f32 v27, v14, v21 // (op 2240) 


v_add_f32 v32, v24, v28 // (op 2236) 
v_add_f32 v33, v25, v29 // (op 2236) 
v_add_f32 v34, v26, v30 // (op 2236) 
v_add_f32 v35, v27, v31 // (op 2236) 


v_cvt_f16_f32 v28, v32 // (op 2237) 
v_cvt_f16_f32 v29, v33 // (op 2237) 
v_cvt_f16_f32 v30, v34 // (op 2237) 
v_cvt_f16_f32 v31, v35 // (op 2237) 


v_pack_b32_f16 v16, v28, v29 // (op 2238) 
v_pack_b32_f16 v17, v30, v31 // (op 2238) 


buffer_store_dwordx2 v[16:17], v22, s[60:63], 0 offen offset:64 // (op 2238) Store value





buffer_load_dwordx2 v[16:17], v18, s[56:59], 0 offen // (op 138) Load value
v_accvgpr_read v24, a0 // (op 134) 
v_accvgpr_read v25, a1 // (op 134) 
v_accvgpr_read v26, a2 // (op 134) 
v_accvgpr_read v27, a3 // (op 134) 
v_mul_f32 v28, v15, v24 // (op 134) 
v_mul_f32 v29, v15, v25 // (op 134) 
v_mul_f32 v30, v15, v26 // (op 134) 
v_mul_f32 v31, v15, v27 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 139) 
v_lshrrev_b32 v21, 16, v16 // (op 139) 
v_cvt_f32_f16 v21, v21 // (op 139) 
v_mul_f32 v24, v14, v20 // (op 139) 
v_mul_f32 v25, v14, v21 // (op 139) 
v_cvt_f32_f16 v20, v17 // (op 139) 
v_lshrrev_b32 v21, 16, v17 // (op 139) 
v_cvt_f32_f16 v21, v21 // (op 139) 
v_mul_f32 v26, v14, v20 // (op 139) 
v_mul_f32 v27, v14, v21 // (op 139) 


v_add_f32 v32, v24, v28 // (op 135) 
v_add_f32 v33, v25, v29 // (op 135) 
v_add_f32 v34, v26, v30 // (op 135) 
v_add_f32 v35, v27, v31 // (op 135) 


v_cvt_f16_f32 v28, v32 // (op 136) 
v_cvt_f16_f32 v29, v33 // (op 136) 
v_cvt_f16_f32 v30, v34 // (op 136) 
v_cvt_f16_f32 v31, v35 // (op 136) 


v_pack_b32_f16 v16, v28, v29 // (op 137) 
v_pack_b32_f16 v17, v30, v31 // (op 137) 


buffer_store_dwordx2 v[16:17], v22, s[60:63], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_125:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365 // (op 365) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_Postmaccumulation_Condition_365:
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_Postmaccumulation_Condition_365:


s_mul_i32 s52, s8, s16 // (op 278) 
s_cmp_gt_u32 s9, s52 // (op 278) 
s_mov_b32 s53, scc // (op 278) 
s_sub_u32 s54, s9, s52 // (op 278) 

s_cmp_eq_i32 s53, 0 // (op 278) 
s_cselect_b32 s52, 0, s54 // (op 278) 


s_cmp_lt_u32 s8, s52 // (op 278) 
s_mov_b32 s53, scc // (op 278) 
s_cmp_eq_i32 s53, 0 // (op 278) 
s_cselect_b32 s54, s52, s8 // (op 278) 


s_cmp_lt_u32 s0, s54 // (op 278) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_278 // (op 278) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_SKStreamTileLoop_278 if true)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_SKStreamTileLoop_278:















s_mov_b32 s0, 0 // (op 419) call()
s_mov_b32 s1, 0 // (op 420) call()
s_mov_b32 s0, 0 // (op 434) call()
s_mul_i32 s15, s50, s16 // (op 433) 
s_cmp_gt_u32 s51, s15 // (op 433) 
s_mov_b32 s6, scc // (op 433) 
s_sub_u32 s7, s51, s15 // (op 433) 

s_cmp_eq_i32 s6, 0 // (op 433) 
s_cselect_b32 s8, 0, s7 // (op 433) 


s_cmp_lt_u32 s50, s8 // (op 433) 
s_mov_b32 s15, scc // (op 433) 
s_cmp_eq_i32 s15, 0 // (op 433) 
s_cselect_b32 s6, s8, s50 // (op 433) 


s_cmp_lt_u32 s0, s6 // (op 433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_433 // (op 433) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_433 if false)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_433:
v_lshrrev_b32 v0, 7, v1 // (op 4476) 
s_mul_i32 s15, s16, s50 // (op 4476) 
s_add_u32 s15, s15, s0 // (op 4476) 
v_and_b32 v3, 63, v1 // (op 4476) 
s_add_u32 s6, s15, s9 // (op 4476) 

v_lshrrev_b32 v6, 4, v3 // (op 4476) 
s_mul_hi_u32 s15, s6, s13 // (op 4476) 
v_lshlrev_b32 v7, 2, v6 // (op 4476) 

s_sub_u32 s7, s6, s15 // (op 4476) 
v_lshl_add_u32 v6, v0, 4, v7 // (op 4476) 


s_lshr_b32 s8, s7, 1 // (op 4476) 

v_lshrrev_b32 v0, 2, v3 // (op 4476) 

s_add_u32 s7, s8, s15 // (op 4476) 


v_and_b32 v3, 3, v0 // (op 4476) 

s_lshr_b32 s8, s7, s11 // (op 4476) 

v_and_b32 v0, 3, v1 // (op 4476) 
s_and_b32 s7, s11, 2147483648 // (op 4476) 
v_lshl_add_u32 v7, v3, 2, v0 // (op 4476) 


s_cmp_eq_i32 s7, 0 // (op 4476) 
s_mov_b32 s15, scc // (op 4476) 

v_lshrrev_b32 v0, 6, v1 // (op 4476) 
s_cmp_eq_i32 s15, 0 // (op 4476) 
s_cselect_b32 s7, s6, s8 // (op 4476) 



v_and_b32 v3, 1, v0 // (op 4476) 

s_mul_hi_u32 s6, s7, s12 // (op 4476) 
v_lshl_add_u32 v0, v3, 4, v7 // (op 4476) 


s_sub_u32 s8, s7, s6 // (op 4476) 
s_lshr_b32 s15, s8, 1 // (op 4476) 

s_add_u32 s8, s15, s6 // (op 4476) 


s_lshr_b32 s6, s8, s10 // (op 4476) 

s_and_b32 s8, s10, 2147483648 // (op 4476) 
s_cmp_eq_i32 s8, 0 // (op 4476) 
s_mov_b32 s15, scc // (op 4476) 

s_cmp_eq_i32 s15, 0 // (op 4476) 
s_cselect_b32 s8, s7, s6 // (op 4476) 


s_mul_i32 s6, s8, s14 // (op 4476) 
v_lshl_add_u32 v3, s8, 7, v6 // (op 4476) 


s_sub_u32 s8, s7, s6 // (op 4476) 


v_lshl_add_u32 v6, s8, 5, v0 // (op 4476) 


v_mul_lo_u32 v0, v6, s4 // (op 4476) 

v_add_lshl_u32 v6, v3, v0, 1 // (op 4476) 


s_mul_i32 s19, s4, s23 // (op 4478) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 4478) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 4478) least significant
s_mul_hi_u32 s6, s4, s22 // (op 4478) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4478) most significant: sum
s_add_u32 s19, s19, s6 // (op 4478) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 4478) 

s_mov_b32 s56, s20 // (op 4478) reg expression
s_mov_b32 s57, s21 // (op 4478) reg expression
s_mov_b32 s58, s28 // (op 4478) reg expression
s_mov_b32 s59, 131072 // (op 4478) call()

v_lshrrev_b32 v0, 7, v1 // (op 4734) 
s_mul_i32 s15, s16, s50 // (op 4734) 
s_add_u32 s15, s15, s0 // (op 4734) 
v_and_b32 v3, 63, v1 // (op 4734) 
s_add_u32 s6, s15, s9 // (op 4734) 

v_lshrrev_b32 v7, 4, v3 // (op 4734) 
s_mul_hi_u32 s15, s6, s13 // (op 4734) 
v_lshlrev_b32 v12, 2, v7 // (op 4734) 

s_sub_u32 s7, s6, s15 // (op 4734) 
v_lshl_add_u32 v7, v0, 4, v12 // (op 4734) 


s_lshr_b32 s8, s7, 1 // (op 4734) 

v_lshrrev_b32 v0, 2, v3 // (op 4734) 

s_add_u32 s7, s8, s15 // (op 4734) 


v_and_b32 v3, 3, v0 // (op 4734) 

s_lshr_b32 s8, s7, s11 // (op 4734) 

v_and_b32 v0, 3, v1 // (op 4734) 
s_and_b32 s7, s11, 2147483648 // (op 4734) 
v_lshl_add_u32 v12, v3, 2, v0 // (op 4734) 


s_cmp_eq_i32 s7, 0 // (op 4734) 
s_mov_b32 s15, scc // (op 4734) 

v_lshrrev_b32 v0, 6, v1 // (op 4734) 
s_cmp_eq_i32 s15, 0 // (op 4734) 
s_cselect_b32 s7, s6, s8 // (op 4734) 



v_and_b32 v3, 1, v0 // (op 4734) 

s_mul_hi_u32 s6, s7, s12 // (op 4734) 
v_lshl_add_u32 v0, v3, 4, v12 // (op 4734) 


s_sub_u32 s8, s7, s6 // (op 4734) 
s_lshr_b32 s15, s8, 1 // (op 4734) 

s_add_u32 s8, s15, s6 // (op 4734) 


s_lshr_b32 s6, s8, s10 // (op 4734) 

s_and_b32 s8, s10, 2147483648 // (op 4734) 
s_cmp_eq_i32 s8, 0 // (op 4734) 
s_mov_b32 s15, scc // (op 4734) 

s_cmp_eq_i32 s15, 0 // (op 4734) 
s_cselect_b32 s8, s7, s6 // (op 4734) 


s_mul_i32 s6, s8, s14 // (op 4734) 
v_lshl_add_u32 v3, s8, 7, v7 // (op 4734) 


s_sub_u32 s8, s7, s6 // (op 4734) 


v_lshl_add_u32 v7, s8, 5, v0 // (op 4734) 


v_mul_lo_u32 v0, v7, s24 // (op 4734) 

v_add_lshl_u32 v7, v3, v0, 1 // (op 4734) 


s_mul_i32 s19, s24, s23 // (op 4736) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 4736) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 4736) least significant
s_mul_hi_u32 s6, s24, s22 // (op 4736) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4736) most significant: sum
s_add_u32 s19, s19, s6 // (op 4736) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 4736) 

s_mov_b32 s60, s26 // (op 4736) reg expression
s_mov_b32 s61, s27 // (op 4736) reg expression
s_mov_b32 s62, s28 // (op 4736) reg expression
s_mov_b32 s63, 131072 // (op 4736) call()

v_and_b32 v0, 48, v1 // (op 4813) 
v_and_b32 v3, 63, v1 // (op 4813) 
v_sub_u32 v12, v3, v0 // (op 4813) 

v_lshrrev_b32 v3, 1, v12 // (op 4813) 
v_add_lshl_u32 v13, v0, v3, 7 // (op 4813) 


v_and_b32 v0, 1, v12 // (op 4813) 

v_lshlrev_b32 v3, 3, v0 // (op 4813) 

v_lshrrev_b32 v0, 7, v1 // (op 4813) 
v_lshl_add_u32 v12, v0, 4, v3 // (op 4813) 


v_add_u32 v0, v13, v12 // (op 4813) 


v_lshrrev_b32 v3, 6, v1 // (op 4878) 
v_and_b32 v12, 1, v3 // (op 4878) 

v_and_b32 v3, 15, v1 // (op 4878) 
v_lshl_add_u32 v13, v12, 4, v3 // (op 4878) 


v_and_b32 v3, 48, v1 // (op 4878) 
v_lshl_add_u32 v12, v13, 7, v3 // (op 4878) 


v_lshrrev_b32 v3, 7, v1 // (op 4954) 
v_and_b32 v13, 15, v1 // (op 4954) 
v_lshl_add_u32 v16, v3, 4, v13 // (op 4954) 


v_and_b32 v3, 63, v1 // (op 4954) 
v_lshrrev_b32 v13, 4, v3 // (op 4954) 

v_lshl_add_u32 v3, v13, 7, v16 // (op 4954) 


v_accvgpr_write a0, 0.00000 // (op 174) call()
v_accvgpr_write a1, 0.00000 // (op 174) call()
v_accvgpr_write a2, 0.00000 // (op 174) call()
v_accvgpr_write a3, 0.00000 // (op 174) call()
v_accvgpr_write a4, 0.00000 // (op 589) call()
v_accvgpr_write a5, 0.00000 // (op 589) call()
v_accvgpr_write a6, 0.00000 // (op 589) call()
v_accvgpr_write a7, 0.00000 // (op 589) call()
v_accvgpr_write a8, 0.00000 // (op 751) call()
v_accvgpr_write a9, 0.00000 // (op 751) call()
v_accvgpr_write a10, 0.00000 // (op 751) call()
v_accvgpr_write a11, 0.00000 // (op 751) call()
v_accvgpr_write a12, 0.00000 // (op 913) call()
v_accvgpr_write a13, 0.00000 // (op 913) call()
v_accvgpr_write a14, 0.00000 // (op 913) call()
v_accvgpr_write a15, 0.00000 // (op 913) call()
v_and_b32 v13, -8, v1 // (op 3978) 
s_mul_i32 s15, s16, s50 // (op 3978) 
s_add_u32 s15, s15, s0 // (op 3978) 
v_sub_u32 v16, v1, v13 // (op 3978) 

s_add_u32 s6, s15, s9 // (op 3978) 

v_lshlrev_b32 v13, 4, v16 // (op 3978) 

s_mul_hi_u32 s15, s6, s13 // (op 3978) 
v_lshrrev_b32 v16, 3, v1 // (op 3978) 
s_sub_u32 s7, s6, s15 // (op 3978) 
s_lshr_b32 s8, s7, 1 // (op 3978) 

s_add_u32 s7, s8, s15 // (op 3978) 


s_lshr_b32 s8, s7, s11 // (op 3978) 

s_and_b32 s7, s11, 2147483648 // (op 3978) 
s_cmp_eq_i32 s7, 0 // (op 3978) 
s_mov_b32 s15, scc // (op 3978) 

s_cmp_eq_i32 s15, 0 // (op 3978) 
s_cselect_b32 s7, s6, s8 // (op 3978) 


s_mul_hi_u32 s8, s7, s12 // (op 3978) 
s_sub_u32 s15, s7, s8 // (op 3978) 
s_lshr_b32 s18, s15, 1 // (op 3978) 

s_add_u32 s15, s18, s8 // (op 3978) 


s_lshr_b32 s8, s15, s10 // (op 3978) 

s_and_b32 s15, s10, 2147483648 // (op 3978) 
s_cmp_eq_i32 s15, 0 // (op 3978) 
s_mov_b32 s18, scc // (op 3978) 

s_cmp_eq_i32 s18, 0 // (op 3978) 
s_cselect_b32 s15, s7, s8 // (op 3978) 


v_lshl_add_u32 v17, s15, 7, v13 // (op 3978) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3978) 
v_mov_b32 v19, 0 // (op 3978) convert
v_mov_b32 v18, v17 // (op 3978) convert

s_mul_i32 s8, s7, s18 // (op 3978) 


s_sub_u32 s7, s6, s8 // (op 3978) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3978) 


v_mul_lo_u32 v17, v13, s31 // (op 3978) most significant: low of low * high
v_mul_hi_u32 v20, v13, s30 // (op 3978) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 3978) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 3978) most significant: sum


v_add_co_u32 v20, s[18:19], v18, v16 // (op 3978) least significant half
v_addc_co_u32 v21, s[18:19], v19, v17, s[18:19] // (op 3978) most significant half



s_mul_i32 s19, s30, s3 // (op 3980) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 3980) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 3980) least significant
s_mul_hi_u32 s6, s30, s2 // (op 3980) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3980) most significant: sum
s_add_u32 s19, s19, s6 // (op 3980) most significant: sum


s_mov_b32 s68, s32 // (op 3980) reg expression
s_mov_b32 s69, s33 // (op 3980) reg expression
s_mov_b32 s70, s18 // (op 3980) reg expression
s_mov_b32 s71, 131072 // (op 3980) call()

v_and_b32 v13, -8, v1 // (op 3997) 
s_mul_i32 s15, s16, s50 // (op 3997) 
s_add_u32 s15, s15, s0 // (op 3997) 
v_sub_u32 v16, v1, v13 // (op 3997) 

s_add_u32 s6, s15, s9 // (op 3997) 

v_lshlrev_b32 v13, 4, v16 // (op 3997) 

s_mul_hi_u32 s15, s6, s13 // (op 3997) 
v_lshrrev_b32 v16, 3, v1 // (op 3997) 
s_sub_u32 s7, s6, s15 // (op 3997) 
s_lshr_b32 s8, s7, 1 // (op 3997) 

s_add_u32 s7, s8, s15 // (op 3997) 


s_lshr_b32 s8, s7, s11 // (op 3997) 

s_and_b32 s7, s11, 2147483648 // (op 3997) 
s_cmp_eq_i32 s7, 0 // (op 3997) 
s_mov_b32 s15, scc // (op 3997) 

s_cmp_eq_i32 s15, 0 // (op 3997) 
s_cselect_b32 s7, s6, s8 // (op 3997) 


s_mul_hi_u32 s8, s7, s12 // (op 3997) 
s_sub_u32 s15, s7, s8 // (op 3997) 
s_lshr_b32 s18, s15, 1 // (op 3997) 

s_add_u32 s15, s18, s8 // (op 3997) 


s_lshr_b32 s8, s15, s10 // (op 3997) 

s_and_b32 s15, s10, 2147483648 // (op 3997) 
s_cmp_eq_i32 s15, 0 // (op 3997) 
s_mov_b32 s18, scc // (op 3997) 

s_cmp_eq_i32 s18, 0 // (op 3997) 
s_cselect_b32 s15, s7, s8 // (op 3997) 


v_lshl_add_u32 v17, s15, 7, v13 // (op 3997) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 3997) 
v_mov_b32 v19, 0 // (op 3997) convert
v_mov_b32 v18, v17 // (op 3997) convert

s_mul_i32 s8, s7, s18 // (op 3997) 


s_sub_u32 s7, s6, s8 // (op 3997) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 3997) 


v_mul_lo_u32 v17, v13, s31 // (op 3997) most significant: low of low * high
v_mul_hi_u32 v22, v13, s30 // (op 3997) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 3997) least significant: low of low * low
v_add_u32 v17, v17, v22 // (op 3997) most significant: sum


v_add_co_u32 v22, s[18:19], v18, v16 // (op 3997) least significant half
v_addc_co_u32 v23, s[18:19], v19, v17, s[18:19] // (op 3997) most significant half



s_mul_i32 s19, s30, s3 // (op 3999) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 3999) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 3999) least significant
s_mul_hi_u32 s6, s30, s2 // (op 3999) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3999) most significant: sum
s_add_u32 s19, s19, s6 // (op 3999) most significant: sum


s_mov_b32 s68, s32 // (op 3999) reg expression
s_mov_b32 s69, s33 // (op 3999) reg expression
s_mov_b32 s70, s18 // (op 3999) reg expression
s_mov_b32 s71, 131072 // (op 3999) call()

v_lshrrev_b32 v13, 3, v1 // (op 4112) 
s_mul_i32 s15, s16, s50 // (op 4112) 
s_add_u32 s15, s15, s0 // (op 4112) 
v_and_b32 v16, -8, v1 // (op 4112) 
s_add_u32 s6, s15, s9 // (op 4112) 

v_sub_u32 v17, v1, v16 // (op 4112) 

s_mul_hi_u32 s15, s6, s13 // (op 4112) 
v_lshlrev_b32 v16, 4, v17 // (op 4112) 

s_sub_u32 s7, s6, s15 // (op 4112) 
s_lshr_b32 s8, s7, 1 // (op 4112) 

s_add_u32 s7, s8, s15 // (op 4112) 


s_lshr_b32 s8, s7, s11 // (op 4112) 

s_and_b32 s7, s11, 2147483648 // (op 4112) 
s_cmp_eq_i32 s7, 0 // (op 4112) 
s_mov_b32 s15, scc // (op 4112) 

s_cmp_eq_i32 s15, 0 // (op 4112) 
s_cselect_b32 s7, s6, s8 // (op 4112) 


s_mul_hi_u32 s8, s7, s12 // (op 4112) 
s_sub_u32 s15, s7, s8 // (op 4112) 
s_lshr_b32 s18, s15, 1 // (op 4112) 

s_add_u32 s15, s18, s8 // (op 4112) 


s_lshr_b32 s8, s15, s10 // (op 4112) 

s_and_b32 s15, s10, 2147483648 // (op 4112) 
s_cmp_eq_i32 s15, 0 // (op 4112) 
s_mov_b32 s18, scc // (op 4112) 

s_cmp_eq_i32 s18, 0 // (op 4112) 
s_cselect_b32 s15, s7, s8 // (op 4112) 


s_mul_i32 s8, s15, s14 // (op 4112) 

s_sub_u32 s15, s7, s8 // (op 4112) 

v_lshl_add_u32 v17, s15, 5, v13 // (op 4112) 


v_mul_lo_u32 v19, v17, s35 // (op 4112) most significant: low of low * high
v_mul_hi_u32 v13, v17, s34 // (op 4112) most significant: high of low * low
v_mul_lo_u32 v18, v17, s34 // (op 4112) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 4112) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4112) 
s_mul_i32 s8, s7, s18 // (op 4112) 


s_sub_u32 s7, s6, s8 // (op 4112) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 4112) 


v_mov_b32 v17, 0 // (op 4112) convert
v_mov_b32 v16, v13 // (op 4112) convert

v_add_co_u32 v24, s[18:19], v16, v18 // (op 4112) least significant half
v_addc_co_u32 v25, s[18:19], v17, v19, s[18:19] // (op 4112) most significant half



s_mul_i32 s19, s34, s23 // (op 4114) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 4114) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 4114) least significant
s_mul_hi_u32 s6, s34, s22 // (op 4114) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4114) most significant: sum
s_add_u32 s19, s19, s6 // (op 4114) most significant: sum


s_mov_b32 s72, s36 // (op 4114) reg expression
s_mov_b32 s73, s37 // (op 4114) reg expression
s_mov_b32 s74, s18 // (op 4114) reg expression
s_mov_b32 s75, 131072 // (op 4114) call()

v_lshrrev_b32 v13, 3, v1 // (op 4131) 
s_mul_i32 s15, s16, s50 // (op 4131) 
s_add_u32 s15, s15, s0 // (op 4131) 
v_and_b32 v16, -8, v1 // (op 4131) 
s_add_u32 s6, s15, s9 // (op 4131) 

v_sub_u32 v17, v1, v16 // (op 4131) 

s_mul_hi_u32 s15, s6, s13 // (op 4131) 
v_lshlrev_b32 v16, 4, v17 // (op 4131) 

s_sub_u32 s7, s6, s15 // (op 4131) 
s_lshr_b32 s8, s7, 1 // (op 4131) 

s_add_u32 s7, s8, s15 // (op 4131) 


s_lshr_b32 s8, s7, s11 // (op 4131) 

s_and_b32 s7, s11, 2147483648 // (op 4131) 
s_cmp_eq_i32 s7, 0 // (op 4131) 
s_mov_b32 s15, scc // (op 4131) 

s_cmp_eq_i32 s15, 0 // (op 4131) 
s_cselect_b32 s7, s6, s8 // (op 4131) 


s_mul_hi_u32 s8, s7, s12 // (op 4131) 
s_sub_u32 s15, s7, s8 // (op 4131) 
s_lshr_b32 s18, s15, 1 // (op 4131) 

s_add_u32 s15, s18, s8 // (op 4131) 


s_lshr_b32 s8, s15, s10 // (op 4131) 

s_and_b32 s15, s10, 2147483648 // (op 4131) 
s_cmp_eq_i32 s15, 0 // (op 4131) 
s_mov_b32 s18, scc // (op 4131) 

s_cmp_eq_i32 s18, 0 // (op 4131) 
s_cselect_b32 s15, s7, s8 // (op 4131) 


s_mul_i32 s8, s15, s14 // (op 4131) 

s_sub_u32 s15, s7, s8 // (op 4131) 

v_lshl_add_u32 v17, s15, 5, v13 // (op 4131) 


v_mul_lo_u32 v19, v17, s35 // (op 4131) most significant: low of low * high
v_mul_hi_u32 v13, v17, s34 // (op 4131) most significant: high of low * low
v_mul_lo_u32 v18, v17, s34 // (op 4131) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 4131) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4131) 
s_mul_i32 s8, s7, s18 // (op 4131) 


s_sub_u32 s7, s6, s8 // (op 4131) 


v_lshl_add_u32 v13, s7, 7, v16 // (op 4131) 


v_mov_b32 v17, 0 // (op 4131) convert
v_mov_b32 v16, v13 // (op 4131) convert

v_add_co_u32 v26, s[18:19], v16, v18 // (op 4131) least significant half
v_addc_co_u32 v27, s[18:19], v17, v19, s[18:19] // (op 4131) most significant half



s_mul_i32 s19, s34, s23 // (op 4133) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 4133) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 4133) least significant
s_mul_hi_u32 s6, s34, s22 // (op 4133) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4133) most significant: sum
s_add_u32 s19, s19, s6 // (op 4133) most significant: sum


s_mov_b32 s72, s36 // (op 4133) reg expression
s_mov_b32 s73, s37 // (op 4133) reg expression
s_mov_b32 s74, s18 // (op 4133) reg expression
s_mov_b32 s75, 131072 // (op 4133) call()

v_and_b32 v13, -128, v1 // (op 4240) 
s_mul_i32 s15, s16, s50 // (op 4240) 
s_add_u32 s15, s15, s0 // (op 4240) 
v_sub_u32 v16, v1, v13 // (op 4240) 

s_add_u32 s6, s15, s9 // (op 4240) 

v_lshrrev_b32 v13, 7, v1 // (op 4240) 
s_mul_hi_u32 s15, s6, s13 // (op 4240) 
v_lshlrev_b32 v17, 1, v13 // (op 4240) 

s_sub_u32 s7, s6, s15 // (op 4240) 
s_lshr_b32 s8, s7, 1 // (op 4240) 

s_add_u32 s7, s8, s15 // (op 4240) 


s_lshr_b32 s8, s7, s11 // (op 4240) 

s_and_b32 s7, s11, 2147483648 // (op 4240) 
s_cmp_eq_i32 s7, 0 // (op 4240) 
s_mov_b32 s15, scc // (op 4240) 

s_cmp_eq_i32 s15, 0 // (op 4240) 
s_cselect_b32 s7, s6, s8 // (op 4240) 


s_mul_hi_u32 s8, s7, s12 // (op 4240) 
s_sub_u32 s15, s7, s8 // (op 4240) 
s_lshr_b32 s18, s15, 1 // (op 4240) 

s_add_u32 s15, s18, s8 // (op 4240) 


s_lshr_b32 s8, s15, s10 // (op 4240) 

s_and_b32 s15, s10, 2147483648 // (op 4240) 
s_cmp_eq_i32 s15, 0 // (op 4240) 
s_mov_b32 s18, scc // (op 4240) 

s_cmp_eq_i32 s18, 0 // (op 4240) 
s_cselect_b32 s15, s7, s8 // (op 4240) 


v_lshl_add_u32 v13, s15, 7, v16 // (op 4240) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4240) 
s_mul_i32 s8, s7, s18 // (op 4240) 


s_sub_u32 s7, s6, s8 // (op 4240) 


v_lshl_add_u32 v16, s7, 2, v17 // (op 4240) 


v_mul_lo_u32 v17, v16, s38 // (op 4240) 

v_add_u32 v16, v13, v17 // (op 4240) 


s_mul_i32 s19, s38, s43 // (op 4242) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 4242) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 4242) least significant
s_mul_hi_u32 s6, s38, s42 // (op 4242) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4242) most significant: sum
s_add_u32 s19, s19, s6 // (op 4242) most significant: sum


s_mov_b32 s76, s40 // (op 4242) reg expression
s_mov_b32 s77, s41 // (op 4242) reg expression
s_mov_b32 s78, s18 // (op 4242) reg expression
s_mov_b32 s79, 131072 // (op 4242) call()

v_and_b32 v13, -128, v1 // (op 4259) 
s_mul_i32 s15, s16, s50 // (op 4259) 
s_add_u32 s15, s15, s0 // (op 4259) 
v_sub_u32 v17, v1, v13 // (op 4259) 

s_add_u32 s6, s15, s9 // (op 4259) 

v_lshrrev_b32 v13, 7, v1 // (op 4259) 
s_mul_hi_u32 s15, s6, s13 // (op 4259) 
v_lshlrev_b32 v18, 1, v13 // (op 4259) 

s_sub_u32 s7, s6, s15 // (op 4259) 
s_lshr_b32 s8, s7, 1 // (op 4259) 

s_add_u32 s7, s8, s15 // (op 4259) 


s_lshr_b32 s8, s7, s11 // (op 4259) 

s_and_b32 s7, s11, 2147483648 // (op 4259) 
s_cmp_eq_i32 s7, 0 // (op 4259) 
s_mov_b32 s15, scc // (op 4259) 

s_cmp_eq_i32 s15, 0 // (op 4259) 
s_cselect_b32 s7, s6, s8 // (op 4259) 


s_mul_hi_u32 s8, s7, s12 // (op 4259) 
s_sub_u32 s15, s7, s8 // (op 4259) 
s_lshr_b32 s18, s15, 1 // (op 4259) 

s_add_u32 s15, s18, s8 // (op 4259) 


s_lshr_b32 s8, s15, s10 // (op 4259) 

s_and_b32 s15, s10, 2147483648 // (op 4259) 
s_cmp_eq_i32 s15, 0 // (op 4259) 
s_mov_b32 s18, scc // (op 4259) 

s_cmp_eq_i32 s18, 0 // (op 4259) 
s_cselect_b32 s15, s7, s8 // (op 4259) 


v_lshl_add_u32 v13, s15, 7, v17 // (op 4259) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 4259) 
s_mul_i32 s8, s7, s18 // (op 4259) 


s_sub_u32 s7, s6, s8 // (op 4259) 


v_lshl_add_u32 v17, s7, 2, v18 // (op 4259) 


v_mul_lo_u32 v18, v17, s38 // (op 4259) 

v_add_u32 v17, v13, v18 // (op 4259) 


s_mul_i32 s19, s38, s43 // (op 4261) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 4261) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 4261) least significant
s_mul_hi_u32 s6, s38, s42 // (op 4261) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4261) most significant: sum
s_add_u32 s19, s19, s6 // (op 4261) most significant: sum


s_mov_b32 s76, s40 // (op 4261) reg expression
s_mov_b32 s77, s41 // (op 4261) reg expression
s_mov_b32 s78, s18 // (op 4261) reg expression
s_mov_b32 s79, 131072 // (op 4261) call()

v_lshrrev_b32 v13, 6, v1 // (op 4368) 
s_mul_i32 s15, s16, s50 // (op 4368) 
s_add_u32 s15, s15, s0 // (op 4368) 
v_and_b32 v18, 1, v13 // (op 4368) 

s_add_u32 s6, s15, s9 // (op 4368) 

v_and_b32 v13, 15, v1 // (op 4368) 
s_mul_hi_u32 s15, s6, s13 // (op 4368) 
v_lshl_add_u32 v19, v18, 4, v13 // (op 4368) 


s_sub_u32 s7, s6, s15 // (op 4368) 
v_and_b32 v13, 63, v1 // (op 4368) 
s_lshr_b32 s8, s7, 1 // (op 4368) 

v_lshrrev_b32 v18, 4, v13 // (op 4368) 

s_add_u32 s7, s8, s15 // (op 4368) 


s_lshr_b32 s8, s7, s11 // (op 4368) 

s_and_b32 s7, s11, 2147483648 // (op 4368) 
s_cmp_eq_i32 s7, 0 // (op 4368) 
s_mov_b32 s15, scc // (op 4368) 

s_cmp_eq_i32 s15, 0 // (op 4368) 
s_cselect_b32 s7, s6, s8 // (op 4368) 


s_mul_hi_u32 s8, s7, s12 // (op 4368) 
s_sub_u32 s15, s7, s8 // (op 4368) 
s_lshr_b32 s18, s15, 1 // (op 4368) 

s_add_u32 s15, s18, s8 // (op 4368) 


s_lshr_b32 s8, s15, s10 // (op 4368) 

s_and_b32 s15, s10, 2147483648 // (op 4368) 
s_cmp_eq_i32 s15, 0 // (op 4368) 
s_mov_b32 s18, scc // (op 4368) 

s_cmp_eq_i32 s18, 0 // (op 4368) 
s_cselect_b32 s15, s7, s8 // (op 4368) 


s_mul_i32 s8, s15, s14 // (op 4368) 

s_sub_u32 s15, s7, s8 // (op 4368) 

v_lshl_add_u32 v13, s15, 5, v19 // (op 4368) 


v_mul_lo_u32 v19, v13, s44 // (op 4368) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 4368) 
s_mul_i32 s8, s7, s18 // (op 4368) 


s_sub_u32 s7, s6, s8 // (op 4368) 


v_lshl_add_u32 v13, s7, 2, v18 // (op 4368) 


v_add_u32 v18, v13, v19 // (op 4368) 


s_mul_i32 s19, s44, s23 // (op 4370) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4370) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4370) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4370) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4370) most significant: sum
s_add_u32 s19, s19, s6 // (op 4370) most significant: sum


s_mov_b32 s80, s46 // (op 4370) reg expression
s_mov_b32 s81, s47 // (op 4370) reg expression
s_mov_b32 s82, s18 // (op 4370) reg expression
s_mov_b32 s83, 131072 // (op 4370) call()

v_lshrrev_b32 v13, 6, v1 // (op 4387) 
s_mul_i32 s15, s16, s50 // (op 4387) 
s_add_u32 s15, s15, s0 // (op 4387) 
v_and_b32 v19, 1, v13 // (op 4387) 

s_add_u32 s6, s15, s9 // (op 4387) 

v_and_b32 v13, 15, v1 // (op 4387) 
s_mul_hi_u32 s15, s6, s13 // (op 4387) 
v_lshl_add_u32 v28, v19, 4, v13 // (op 4387) 


s_sub_u32 s7, s6, s15 // (op 4387) 
v_and_b32 v13, 63, v1 // (op 4387) 
s_lshr_b32 s8, s7, 1 // (op 4387) 

v_lshrrev_b32 v19, 4, v13 // (op 4387) 

s_add_u32 s7, s8, s15 // (op 4387) 


s_lshr_b32 s8, s7, s11 // (op 4387) 

s_and_b32 s7, s11, 2147483648 // (op 4387) 
s_cmp_eq_i32 s7, 0 // (op 4387) 
s_mov_b32 s15, scc // (op 4387) 

s_cmp_eq_i32 s15, 0 // (op 4387) 
s_cselect_b32 s7, s6, s8 // (op 4387) 


s_mul_hi_u32 s8, s7, s12 // (op 4387) 
s_sub_u32 s15, s7, s8 // (op 4387) 
s_lshr_b32 s18, s15, 1 // (op 4387) 

s_add_u32 s15, s18, s8 // (op 4387) 


s_lshr_b32 s8, s15, s10 // (op 4387) 

s_and_b32 s15, s10, 2147483648 // (op 4387) 
s_cmp_eq_i32 s15, 0 // (op 4387) 
s_mov_b32 s18, scc // (op 4387) 

s_cmp_eq_i32 s18, 0 // (op 4387) 
s_cselect_b32 s15, s7, s8 // (op 4387) 


s_mul_i32 s8, s15, s14 // (op 4387) 

s_sub_u32 s15, s7, s8 // (op 4387) 

v_lshl_add_u32 v13, s15, 5, v28 // (op 4387) 


v_mul_lo_u32 v28, v13, s44 // (op 4387) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 4387) 
s_mul_i32 s8, s7, s18 // (op 4387) 


s_sub_u32 s7, s6, s8 // (op 4387) 


v_lshl_add_u32 v13, s7, 2, v19 // (op 4387) 


v_add_u32 v19, v13, v28 // (op 4387) 


s_mul_i32 s19, s44, s23 // (op 4389) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4389) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4389) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4389) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4389) most significant: sum
s_add_u32 s19, s19, s6 // (op 4389) most significant: sum


s_mov_b32 s80, s46 // (op 4389) reg expression
s_mov_b32 s81, s47 // (op 4389) reg expression
s_mov_b32 s82, s18 // (op 4389) reg expression
s_mov_b32 s83, 131072 // (op 4389) call()




buffer_load_ubyte v13, v18, s[80:83], 0 offen offset: 0 // (op 3512) Load value
v_and_b32 v28, -8, v1 // (op 4854) 
v_sub_u32 v29, v1, v28 // (op 4854) 

v_lshlrev_b32 v28, 4, v29 // (op 4854) 

v_lshrrev_b32 v29, 3, v1 // (op 4854) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 4854) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 4854) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5391) 



s_nop 0
s_mov_b32 m0, s15 // (op 3886) reg expression
buffer_load_dwordx4 v24, s[72:75], 0 offen offset: 0 lds // (op 3886) Load value direct to lds

v_and_b32 v28, -8, v1 // (op 4779) 
v_sub_u32 v29, v1, v28 // (op 4779) 

v_lshlrev_b32 v28, 4, v29 // (op 4779) 

v_lshrrev_b32 v29, 3, v1 // (op 4779) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 4779) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 4779) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5372) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 3891) 
v_mov_b64 v[28:29], s[18:19] // (op 3891) convert


v_mov_b64 v[30:31], v[20:21] // (op 3891) reg expression
s_add_u32 s6, 4096, s15 // (op 3891) 
s_mov_b32 m0, s6 // (op 3891) reg expression
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3891) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 3891) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3891) most significant half

s_add_u32 m0, m0, 4096 // (op 3891) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3891) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 3891) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3891) most significant half

s_add_u32 m0, m0, 4096 // (op 3891) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3891) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 3891) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3891) most significant half

s_add_u32 m0, m0, 4096 // (op 3891) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3891) Load value direct to lds





v_mov_b64 v[28:29], s[38:39] // (op 3522) convert

v_mov_b32 v30, v16 // (op 3522) reg expression
buffer_load_ubyte v32, v30, s[76:79], 0 offen offset: 0 // (op 3522) Load value
v_add_u32 v30, v30, v28 // (op 3522) 
buffer_load_ubyte v33, v30, s[76:79], 0 offen offset: 0 // (op 3522) Load value





buffer_load_ubyte v28, v18, s[80:83], 0 offen offset:4 // (op 3526) Load value

v_and_b32 v18, -8, v1 // (op 5068) 
v_sub_u32 v29, v1, v18 // (op 5068) 

v_lshlrev_b32 v18, 4, v29 // (op 5068) 

v_lshrrev_b32 v29, 3, v1 // (op 5068) 
v_lshl_add_u32 v30, v29, 7, v18 // (op 5068) 


s_nop 0
v_readfirstlane_b32 s15, v30 // (op 5068) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5439) 


s_nop 0
s_add_u32 s6, 20480, s15 // (op 3896) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s6 // (op 3896) reg expression
s_mov_b32 s7, 128 // (op 3896) call()
buffer_load_dwordx4 v24, s[72:75], s7 offen offset: 0 lds // (op 3896) Load value direct to lds




v_and_b32 v18, -8, v1 // (op 5024) 
v_sub_u32 v24, v1, v18 // (op 5024) 

v_lshlrev_b32 v18, 4, v24 // (op 5024) 

v_lshrrev_b32 v24, 3, v1 // (op 5024) 
v_lshl_add_u32 v25, v24, 7, v18 // (op 5024) 


s_nop 0
v_readfirstlane_b32 s15, v25 // (op 5024) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 5430) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 3898) 
v_mov_b64 v[24:25], s[18:19] // (op 3898) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 3898) 
v_mov_b32 v18, s18 // (op 3898) 
v_mov_b32 v29, s19 // (op 3898) 
v_add_co_u32 v30, s[28:29], v20, v18 // (op 3898) least significant half
v_addc_co_u32 v31, s[28:29], v21, v29, s[28:29] // (op 3898) most significant half




s_add_u32 s6, 24576, s15 // (op 3898) 
s_mov_b32 m0, s6 // (op 3898) reg expression
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3898) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 3898) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3898) most significant half

s_add_u32 m0, m0, 4096 // (op 3898) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3898) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 3898) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3898) most significant half

s_add_u32 m0, m0, 4096 // (op 3898) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3898) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v24 // (op 3898) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 3898) most significant half

s_add_u32 m0, m0, 4096 // (op 3898) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 3898) Load value direct to lds






v_mov_b64 v[20:21], s[38:39] // (op 3536) convert

s_lshl_b32 s15, s38, 2 // (op 3536) 
v_add_u32 v18, v16, s15 // (op 3536) 

buffer_load_ubyte v24, v18, s[76:79], 0 offen offset: 0 // (op 3536) Load value
v_add_u32 v18, v18, v20 // (op 3536) 
buffer_load_ubyte v25, v18, s[76:79], 0 offen offset: 0 // (op 3536) Load value



v_and_b32 v16, -128, v1 // (op 4920) 
v_sub_u32 v18, v1, v16 // (op 4920) 

v_lshrrev_b32 v16, 7, v1 // (op 4920) 
v_lshl_add_u32 v20, v16, 8, v18 // (op 4920) 


s_barrier  // (op 5410) 



s_waitcnt vmcnt(9)
ds_write_b8 v20, v32 offset:40960 // (op 3540) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v20, v33 offset:41088 // (op 3540) Store local data 

s_barrier  // (op 5388) 



ds_read_b128 v[34:37], v12 // (op 3573) Load local data 
ds_read_b128 v[38:41], v12 offset:64 // (op 3573) Load local data 
s_barrier  // (op 5369) 



ds_read_b64_tr_b8 v[42:43], v0 offset:4096 // (op 3554) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:5120 // (op 3554) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:12288 // (op 3554) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:13312 // (op 3554) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 5407) 


ds_read_u8 v16, v3 offset:40960 // (op 3577) Load local data 



ds_read_b64_tr_b8 v[50:51], v0 offset:4128 // (op 3559) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:5152 // (op 3559) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:12320 // (op 3559) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:13344 // (op 3559) Transpose load local data 


ds_read_u8 v18, v3 offset:40992 // (op 3582) Load local data 



ds_read_b64_tr_b8 v[58:59], v0 offset:4160 // (op 3564) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:5184 // (op 3564) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:12352 // (op 3564) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:13376 // (op 3564) Transpose load local data 


ds_read_u8 v20, v3 offset:41024 // (op 3587) Load local data 



ds_read_b64_tr_b8 v[66:67], v0 offset:4192 // (op 3569) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:5216 // (op 3569) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:12384 // (op 3569) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:13408 // (op 3569) Transpose load local data 


ds_read_u8 v21, v3 offset:41056 // (op 3592) Load local data 
v_and_b32 v29, -8, v1 // (op 4751) 
v_sub_u32 v30, v1, v29 // (op 4751) 

v_lshlrev_b32 v29, 4, v30 // (op 4751) 

v_lshrrev_b32 v30, 3, v1 // (op 4751) 
v_lshl_add_u32 v31, v30, 7, v29 // (op 4751) 


s_nop 0
v_readfirstlane_b32 s15, v31 // (op 4751) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -8, v1 // (op 4826) 
v_sub_u32 v30, v1, v29 // (op 4826) 

v_lshlrev_b32 v29, 4, v30 // (op 4826) 

v_lshrrev_b32 v30, 3, v1 // (op 4826) 
v_lshl_add_u32 v31, v30, 7, v29 // (op 4826) 


s_nop 0
v_readfirstlane_b32 s6, v31 // (op 4826) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -128, v1 // (op 4891) 
v_sub_u32 v30, v1, v29 // (op 4891) 

v_lshrrev_b32 v29, 7, v1 // (op 4891) 
v_lshl_add_u32 v31, v29, 8, v30 // (op 4891) 


v_and_b32 v29, -8, v1 // (op 5009) 
v_sub_u32 v30, v1, v29 // (op 5009) 

v_lshlrev_b32 v29, 4, v30 // (op 5009) 

v_lshrrev_b32 v30, 3, v1 // (op 5009) 
v_lshl_add_u32 v74, v30, 7, v29 // (op 5009) 


s_nop 0
v_readfirstlane_b32 s7, v74 // (op 5009) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -8, v1 // (op 5053) 
v_sub_u32 v30, v1, v29 // (op 5053) 

v_lshlrev_b32 v29, 4, v30 // (op 5053) 

v_lshrrev_b32 v30, 3, v1 // (op 5053) 
v_lshl_add_u32 v74, v30, 7, v29 // (op 5053) 


s_nop 0
v_readfirstlane_b32 s8, v74 // (op 5053) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v29, -128, v1 // (op 5092) 
v_sub_u32 v30, v1, v29 // (op 5092) 

v_lshrrev_b32 v29, 7, v1 // (op 5092) 
v_lshl_add_u32 v74, v29, 8, v30 // (op 5092) 


s_mov_b32 s1, 0 // (op 915) call()
s_ashr_i64 s[18:19], s[2:3], 7 // (op 914) 
s_mov_b32 s28, s18 // (op 914) convert

s_and_b32 s29, s28, 4294967294 // (op 914) 

s_cmp_lt_u32 s1, s29 // (op 914) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_914 // (op 914) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_914 if false)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_914:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[34:41], a[0:3], v16, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 188) 
s_add_u32 s1, s1, 2 // (op 916) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[34:41], a[4:7], v18, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 626) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[34:41], a[8:11], v20, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 788) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[34:41], a[12:15], v21, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 950) 


v_mov_b32 v29, v19 // (op 187) 

buffer_load_ubyte v13, v29, s[80:83], 0 offen offset:8 // (op 187) Load value
s_barrier  // (op 5394) 


v_mov_b64 v[76:77], v[26:27] // (op 3835) 

s_mov_b32 m0, s6 // (op 3835) reg expression
s_mov_b32 s18, 256 // (op 3835) call()
buffer_load_dwordx4 v76, s[72:75], s18 offen offset: 0 lds // (op 3835) Load value direct to lds

s_barrier  // (op 5375) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 3830) 
v_mov_b64 v[78:79], s[18:19] // (op 3830) convert


v_mov_b64 v[80:81], v[22:23] // (op 3830) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 3830) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 3830) 

v_mov_b32 v30, s28 // (op 3830) 
v_mov_b32 v75, s29 // (op 3830) 
v_add_co_u32 v82, s[18:19], v80, v30 // (op 3830) least significant half
v_addc_co_u32 v83, s[18:19], v81, v75, s[18:19] // (op 3830) most significant half




s_add_u32 s18, 4096, s15 // (op 3830) 
s_nop 0
s_mov_b32 m0, s18 // (op 3830) reg expression
buffer_load_dwordx4 v82, s[68:71], 0 offen offset: 0 lds // (op 3830) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v78 // (op 3830) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3830) most significant half

s_add_u32 m0, m0, 4096 // (op 3830) 
buffer_load_dwordx4 v82, s[68:71], 0 offen offset: 0 lds // (op 3830) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v78 // (op 3830) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3830) most significant half

s_add_u32 m0, m0, 4096 // (op 3830) 
buffer_load_dwordx4 v82, s[68:71], 0 offen offset: 0 lds // (op 3830) Load value direct to lds
v_add_co_u32 v82, s[28:29], v82, v78 // (op 3830) least significant half
v_addc_co_u32 v83, s[28:29], 0, v83, s[28:29] // (op 3830) most significant half

s_add_u32 m0, m0, 4096 // (op 3830) 
buffer_load_dwordx4 v82, s[68:71], 0 offen offset: 0 lds // (op 3830) Load value direct to lds




v_mov_b64 v[78:79], s[38:39] // (op 192) convert

v_mov_b32 v30, v17 // (op 192) 
s_lshl_b32 s18, s38, 3 // (op 192) 
v_add_u32 v75, v30, s18 // (op 192) 

buffer_load_ubyte v32, v75, s[76:79], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v75, v75, v78 // (op 192) 
buffer_load_ubyte v33, v75, s[76:79], 0 offen offset: 0 // (op 192) Load value


s_barrier  // (op 5448) 



s_waitcnt vmcnt(9)
ds_write_b8 v74, v24 offset:41472 // (op 516) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v74, v25 offset:41600 // (op 516) Store local data 
s_barrier  // (op 5436) 



ds_read_b128 v[34:37], v5 offset:20480 // (op 509) Load local data 
ds_read_b128 v[38:41], v5 offset:20544 // (op 509) Load local data 
s_barrier  // (op 5427) 



ds_read_b64_tr_b8 v[42:43], v10 offset:24576 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v10 offset:25600 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v10 offset:32768 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v10 offset:33792 // (op 507) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 5445) 


ds_read_u8 v16, v11 offset:41472 // (op 511) Load local data 



ds_read_b64_tr_b8 v[50:51], v10 offset:24608 // (op 633) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v10 offset:25632 // (op 633) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v10 offset:32800 // (op 633) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v10 offset:33824 // (op 633) Transpose load local data 


ds_read_u8 v18, v11 offset:41504 // (op 637) Load local data 



ds_read_b64_tr_b8 v[58:59], v10 offset:24640 // (op 795) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v10 offset:25664 // (op 795) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v10 offset:32832 // (op 795) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v10 offset:33856 // (op 795) Transpose load local data 


ds_read_u8 v20, v11 offset:41536 // (op 799) Load local data 



ds_read_b64_tr_b8 v[66:67], v10 offset:24672 // (op 957) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v10 offset:25696 // (op 957) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v10 offset:32864 // (op 957) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v10 offset:33888 // (op 957) Transpose load local data 


ds_read_u8 v21, v11 offset:41568 // (op 961) Load local data 
s_barrier  // (op 5575) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[34:41], a[0:3], v16, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 514) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[34:41], a[4:7], v18, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 627) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:65], v[34:41], a[8:11], v20, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 789) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[34:41], a[12:15], v21, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 951) 



buffer_load_ubyte v28, v29, s[80:83], 0 offen offset:12 // (op 513) Load value

s_barrier  // (op 5442) 
v_add_u32 v19, 8, v19 // (op 4379) 


s_add_u32 s18, 20480, s8 // (op 3858) 

s_mov_b32 m0, s18 // (op 3858) reg expression
s_mov_b32 s19, 384 // (op 3858) call()
buffer_load_dwordx4 v76, s[72:75], s19 offen offset: 0 lds // (op 3858) Load value direct to lds



s_barrier  // (op 5433) 
v_add_co_u32 v26, vcc, 256, v26 // (op 4123) least significant half

v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 4123) most significant half
s_lshl_b64 s[18:19], s[30:31], 5 // (op 3853) 
v_mov_b64 v[78:79], s[18:19] // (op 3853) convert


s_lshl_b64 s[18:19], s[30:31], 7 // (op 3853) 
s_mul_i32 s29, 3, s19 // (op 3853) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 3853) least significant
s_mul_hi_u32 s48, 3, s18 // (op 3853) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 3853) most significant: sum


v_mov_b32 v29, s28 // (op 3853) 
v_mov_b32 v75, s29 // (op 3853) 
v_add_co_u32 v76, s[18:19], v80, v29 // (op 3853) least significant half
v_addc_co_u32 v77, s[18:19], v81, v75, s[18:19] // (op 3853) most significant half




s_add_u32 s18, 24576, s7 // (op 3853) 
s_nop 0
s_mov_b32 m0, s18 // (op 3853) reg expression
buffer_load_dwordx4 v76, s[68:71], 0 offen offset: 0 lds // (op 3853) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v78 // (op 3853) least significant half
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 3853) most significant half

s_add_u32 m0, m0, 4096 // (op 3853) 
buffer_load_dwordx4 v76, s[68:71], 0 offen offset: 0 lds // (op 3853) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v78 // (op 3853) least significant half
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 3853) most significant half

s_add_u32 m0, m0, 4096 // (op 3853) 
buffer_load_dwordx4 v76, s[68:71], 0 offen offset: 0 lds // (op 3853) Load value direct to lds
v_add_co_u32 v76, s[28:29], v76, v78 // (op 3853) least significant half
v_addc_co_u32 v77, s[28:29], 0, v77, s[28:29] // (op 3853) most significant half

s_add_u32 m0, m0, 4096 // (op 3853) 
buffer_load_dwordx4 v76, s[68:71], 0 offen offset: 0 lds // (op 3853) Load value direct to lds





v_mov_b64 v[76:77], s[38:39] // (op 518) convert

s_lshl_b32 s18, s38, 2 // (op 518) 
s_mul_i32 s19, 3, s18 // (op 518) 

v_add_u32 v29, v30, s19 // (op 518) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 3989) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 3989) 

v_mov_b32 v75, s28 // (op 3989) 
v_mov_b32 v78, s29 // (op 3989) 
v_add_co_u32 v22, s[18:19], v22, v75 // (op 3989) least significant half
v_addc_co_u32 v23, s[18:19], v23, v78, s[18:19] // (op 3989) most significant half




buffer_load_ubyte v24, v29, s[76:79], 0 offen offset: 0 // (op 518) Load value
v_add_u32 v29, v29, v76 // (op 518) 
buffer_load_ubyte v25, v29, s[76:79], 0 offen offset: 0 // (op 518) Load value



s_lshl_b32 s18, s38, 3 // (op 4251) 
s_barrier  // (op 5413) 
v_add_u32 v17, v17, s18 // (op 4251) 




s_waitcnt vmcnt(9)
ds_write_b8 v31, v32 offset:40960 // (op 190) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v31, v33 offset:41088 // (op 190) Store local data 
s_barrier  // (op 5400) 



ds_read_b128 v[34:37], v4 // (op 183) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 183) Load local data 
s_barrier  // (op 5381) 



ds_read_b64_tr_b8 v[42:43], v8 offset:4096 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v8 offset:5120 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v8 offset:12288 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v8 offset:13312 // (op 181) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 5420) 


ds_read_u8 v16, v9 offset:40960 // (op 185) Load local data 



ds_read_b64_tr_b8 v[50:51], v8 offset:4128 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v8 offset:5152 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v8 offset:12320 // (op 620) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v8 offset:13344 // (op 620) Transpose load local data 


ds_read_u8 v18, v9 offset:40992 // (op 624) Load local data 



ds_read_b64_tr_b8 v[58:59], v8 offset:4160 // (op 782) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v8 offset:5184 // (op 782) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v8 offset:12352 // (op 782) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v8 offset:13376 // (op 782) Transpose load local data 


ds_read_u8 v20, v9 offset:41024 // (op 786) Load local data 



ds_read_b64_tr_b8 v[66:67], v8 offset:4192 // (op 944) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v8 offset:5216 // (op 944) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v8 offset:12384 // (op 944) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v8 offset:13408 // (op 944) Transpose load local data 


ds_read_u8 v21, v9 offset:41056 // (op 948) Load local data 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 914) 
s_mov_b32 s28, s18 // (op 914) convert

s_and_b32 s29, s28, 4294967294 // (op 914) 

s_cmp_lt_u32 s1, s29 // (op 914) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_914 // (op 914) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoop_914 if true)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoop_914:















s_barrier  // (op 5559) 
v_and_b32 v13, -8, v1 // (op 3925) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 3925) 
v_sub_u32 v19, v1, v13 // (op 3925) 

s_and_b32 s15, s18, 4294967294 // (op 3925) 
v_lshlrev_b32 v13, 4, v19 // (op 3925) 

s_add_u32 s6, s0, s15 // (op 3925) 

v_lshrrev_b32 v19, 3, v1 // (op 3925) 
s_mul_i32 s15, s16, s50 // (op 3925) 
s_add_u32 s15, s15, s6 // (op 3925) 

s_add_u32 s6, s15, s9 // (op 3925) 

s_mul_hi_u32 s15, s6, s13 // (op 3925) 
s_sub_u32 s7, s6, s15 // (op 3925) 
s_lshr_b32 s8, s7, 1 // (op 3925) 

s_add_u32 s7, s8, s15 // (op 3925) 


s_lshr_b32 s8, s7, s11 // (op 3925) 

s_and_b32 s7, s11, 2147483648 // (op 3925) 
s_cmp_eq_i32 s7, 0 // (op 3925) 
s_mov_b32 s15, scc // (op 3925) 

s_cmp_eq_i32 s15, 0 // (op 3925) 
s_cselect_b32 s7, s6, s8 // (op 3925) 


s_mul_hi_u32 s8, s7, s12 // (op 3925) 
s_sub_u32 s15, s7, s8 // (op 3925) 
s_lshr_b32 s28, s15, 1 // (op 3925) 

s_add_u32 s15, s28, s8 // (op 3925) 


s_lshr_b32 s8, s15, s10 // (op 3925) 

s_and_b32 s15, s10, 2147483648 // (op 3925) 
s_cmp_eq_i32 s15, 0 // (op 3925) 
s_mov_b32 s28, scc // (op 3925) 

s_cmp_eq_i32 s28, 0 // (op 3925) 
s_cselect_b32 s15, s7, s8 // (op 3925) 


s_mul_i32 s8, s7, s18 // (op 3925) 


v_lshl_add_u32 v16, s15, 7, v13 // (op 3925) 


s_sub_u32 s7, s6, s8 // (op 3925) 


v_mov_b32 v23, 0 // (op 3925) convert
v_mov_b32 v22, v16 // (op 3925) convert

v_lshl_add_u32 v13, s7, 7, v19 // (op 3925) 


v_mul_lo_u32 v17, v13, s31 // (op 3925) most significant: low of low * high
v_mul_hi_u32 v19, v13, s30 // (op 3925) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 3925) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 3925) most significant: sum


v_add_co_u32 v26, s[18:19], v22, v16 // (op 3925) least significant half
v_addc_co_u32 v27, s[18:19], v23, v17, s[18:19] // (op 3925) most significant half



s_mul_i32 s19, s30, s3 // (op 3927) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 3927) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 3927) least significant
s_mul_hi_u32 s6, s30, s2 // (op 3927) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3927) most significant: sum
s_add_u32 s19, s19, s6 // (op 3927) most significant: sum


s_mov_b32 s68, s32 // (op 3927) reg expression
s_mov_b32 s69, s33 // (op 3927) reg expression
s_mov_b32 s70, s18 // (op 3927) reg expression
s_mov_b32 s71, 131072 // (op 3927) call()

v_lshrrev_b32 v13, 3, v1 // (op 4067) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4067) 
v_and_b32 v19, -8, v1 // (op 4067) 
s_and_b32 s15, s18, 4294967294 // (op 4067) 
v_sub_u32 v16, v1, v19 // (op 4067) 

s_add_u32 s6, s0, s15 // (op 4067) 

v_lshlrev_b32 v17, 4, v16 // (op 4067) 

s_mul_i32 s15, s16, s50 // (op 4067) 
s_add_u32 s15, s15, s6 // (op 4067) 

s_add_u32 s6, s15, s9 // (op 4067) 

s_mul_hi_u32 s15, s6, s13 // (op 4067) 
s_sub_u32 s7, s6, s15 // (op 4067) 
s_lshr_b32 s8, s7, 1 // (op 4067) 

s_add_u32 s7, s8, s15 // (op 4067) 


s_lshr_b32 s8, s7, s11 // (op 4067) 

s_and_b32 s7, s11, 2147483648 // (op 4067) 
s_cmp_eq_i32 s7, 0 // (op 4067) 
s_mov_b32 s15, scc // (op 4067) 

s_cmp_eq_i32 s15, 0 // (op 4067) 
s_cselect_b32 s7, s6, s8 // (op 4067) 


s_mul_hi_u32 s8, s7, s12 // (op 4067) 
s_sub_u32 s15, s7, s8 // (op 4067) 
s_lshr_b32 s28, s15, 1 // (op 4067) 

s_add_u32 s15, s28, s8 // (op 4067) 


s_lshr_b32 s8, s15, s10 // (op 4067) 

s_and_b32 s15, s10, 2147483648 // (op 4067) 
s_cmp_eq_i32 s15, 0 // (op 4067) 
s_mov_b32 s28, scc // (op 4067) 

s_cmp_eq_i32 s28, 0 // (op 4067) 
s_cselect_b32 s15, s7, s8 // (op 4067) 


s_mul_i32 s8, s15, s14 // (op 4067) 

s_sub_u32 s15, s7, s8 // (op 4067) 

s_mul_i32 s8, s7, s18 // (op 4067) 


v_lshl_add_u32 v16, s15, 5, v13 // (op 4067) 


s_sub_u32 s7, s6, s8 // (op 4067) 


v_mul_lo_u32 v23, v16, s35 // (op 4067) most significant: low of low * high
v_mul_hi_u32 v13, v16, s34 // (op 4067) most significant: high of low * low
v_mul_lo_u32 v22, v16, s34 // (op 4067) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 4067) most significant: sum


v_lshl_add_u32 v13, s7, 7, v17 // (op 4067) 


v_mov_b32 v17, 0 // (op 4067) convert
v_mov_b32 v16, v13 // (op 4067) convert

v_add_co_u32 v48, s[18:19], v16, v22 // (op 4067) least significant half
v_addc_co_u32 v49, s[18:19], v17, v23, s[18:19] // (op 4067) most significant half



s_mul_i32 s19, s34, s23 // (op 4069) most significant: low of low * high
s_mul_i32 s15, s35, s22 // (op 4069) most significant: low of high * low
s_mul_i32 s18, s34, s22 // (op 4069) least significant
s_mul_hi_u32 s6, s34, s22 // (op 4069) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4069) most significant: sum
s_add_u32 s19, s19, s6 // (op 4069) most significant: sum


s_mov_b32 s72, s36 // (op 4069) reg expression
s_mov_b32 s73, s37 // (op 4069) reg expression
s_mov_b32 s74, s18 // (op 4069) reg expression
s_mov_b32 s75, 131072 // (op 4069) call()

v_and_b32 v13, -128, v1 // (op 4195) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4195) 
v_sub_u32 v19, v1, v13 // (op 4195) 

s_and_b32 s15, s18, 4294967294 // (op 4195) 
v_lshrrev_b32 v13, 7, v1 // (op 4195) 
s_add_u32 s6, s0, s15 // (op 4195) 

v_lshlrev_b32 v16, 1, v13 // (op 4195) 

s_mul_i32 s15, s16, s50 // (op 4195) 
s_add_u32 s15, s15, s6 // (op 4195) 

s_add_u32 s6, s15, s9 // (op 4195) 

s_mul_hi_u32 s15, s6, s13 // (op 4195) 
s_sub_u32 s7, s6, s15 // (op 4195) 
s_lshr_b32 s8, s7, 1 // (op 4195) 

s_add_u32 s7, s8, s15 // (op 4195) 


s_lshr_b32 s8, s7, s11 // (op 4195) 

s_and_b32 s7, s11, 2147483648 // (op 4195) 
s_cmp_eq_i32 s7, 0 // (op 4195) 
s_mov_b32 s15, scc // (op 4195) 

s_cmp_eq_i32 s15, 0 // (op 4195) 
s_cselect_b32 s7, s6, s8 // (op 4195) 


s_mul_hi_u32 s8, s7, s12 // (op 4195) 
s_sub_u32 s15, s7, s8 // (op 4195) 
s_lshr_b32 s28, s15, 1 // (op 4195) 

s_add_u32 s15, s28, s8 // (op 4195) 


s_lshr_b32 s8, s15, s10 // (op 4195) 

s_and_b32 s15, s10, 2147483648 // (op 4195) 
s_cmp_eq_i32 s15, 0 // (op 4195) 
s_mov_b32 s28, scc // (op 4195) 

s_cmp_eq_i32 s28, 0 // (op 4195) 
s_cselect_b32 s15, s7, s8 // (op 4195) 


s_mul_i32 s8, s7, s18 // (op 4195) 


v_lshl_add_u32 v13, s15, 7, v19 // (op 4195) 


s_sub_u32 s7, s6, s8 // (op 4195) 


v_lshl_add_u32 v17, s7, 2, v16 // (op 4195) 


v_mul_lo_u32 v16, v17, s38 // (op 4195) 

v_add_u32 v17, v13, v16 // (op 4195) 


s_mul_i32 s19, s38, s43 // (op 4197) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 4197) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 4197) least significant
s_mul_hi_u32 s6, s38, s42 // (op 4197) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4197) most significant: sum
s_add_u32 s19, s19, s6 // (op 4197) most significant: sum


s_mov_b32 s76, s40 // (op 4197) reg expression
s_mov_b32 s77, s41 // (op 4197) reg expression
s_mov_b32 s78, s18 // (op 4197) reg expression
s_mov_b32 s79, 131072 // (op 4197) call()

v_lshrrev_b32 v13, 6, v1 // (op 4323) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 4323) 
v_and_b32 v16, 1, v13 // (op 4323) 

s_and_b32 s15, s18, 4294967294 // (op 4323) 
v_and_b32 v13, 15, v1 // (op 4323) 
s_add_u32 s6, s0, s15 // (op 4323) 

v_lshl_add_u32 v19, v16, 4, v13 // (op 4323) 


v_and_b32 v13, 63, v1 // (op 4323) 
s_mul_i32 s15, s16, s50 // (op 4323) 
s_add_u32 s15, s15, s6 // (op 4323) 

v_lshrrev_b32 v16, 4, v13 // (op 4323) 

s_add_u32 s6, s15, s9 // (op 4323) 

s_mul_hi_u32 s15, s6, s13 // (op 4323) 
s_sub_u32 s7, s6, s15 // (op 4323) 
s_lshr_b32 s8, s7, 1 // (op 4323) 

s_add_u32 s7, s8, s15 // (op 4323) 


s_lshr_b32 s8, s7, s11 // (op 4323) 

s_and_b32 s7, s11, 2147483648 // (op 4323) 
s_cmp_eq_i32 s7, 0 // (op 4323) 
s_mov_b32 s15, scc // (op 4323) 

s_cmp_eq_i32 s15, 0 // (op 4323) 
s_cselect_b32 s7, s6, s8 // (op 4323) 


s_mul_hi_u32 s8, s7, s12 // (op 4323) 
s_sub_u32 s15, s7, s8 // (op 4323) 
s_lshr_b32 s28, s15, 1 // (op 4323) 

s_add_u32 s15, s28, s8 // (op 4323) 


s_lshr_b32 s8, s15, s10 // (op 4323) 

s_and_b32 s15, s10, 2147483648 // (op 4323) 
s_cmp_eq_i32 s15, 0 // (op 4323) 
s_mov_b32 s28, scc // (op 4323) 

s_cmp_eq_i32 s28, 0 // (op 4323) 
s_cselect_b32 s15, s7, s8 // (op 4323) 


s_mul_i32 s8, s15, s14 // (op 4323) 

s_sub_u32 s15, s7, s8 // (op 4323) 

v_lshl_add_u32 v13, s15, 5, v19 // (op 4323) 


s_mul_i32 s8, s7, s18 // (op 4323) 


v_mul_lo_u32 v19, v13, s44 // (op 4323) 

s_sub_u32 s7, s6, s8 // (op 4323) 


v_lshl_add_u32 v13, s7, 2, v16 // (op 4323) 


v_add_u32 v16, v13, v19 // (op 4323) 


s_mul_i32 s19, s44, s23 // (op 4325) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 4325) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 4325) least significant
s_mul_hi_u32 s6, s44, s22 // (op 4325) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 4325) most significant: sum
s_add_u32 s19, s19, s6 // (op 4325) most significant: sum


s_mov_b32 s80, s46 // (op 4325) reg expression
s_mov_b32 s81, s47 // (op 4325) reg expression
s_mov_b32 s82, s18 // (op 4325) reg expression
s_mov_b32 s83, 131072 // (op 4325) call()

v_and_b32 v13, -8, v1 // (op 4764) 
v_sub_u32 v19, v1, v13 // (op 4764) 

v_lshlrev_b32 v13, 4, v19 // (op 4764) 

v_lshrrev_b32 v19, 3, v1 // (op 4764) 
v_lshl_add_u32 v22, v19, 7, v13 // (op 4764) 


s_nop 0
v_readfirstlane_b32 s15, v22 // (op 4764) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 4839) 
v_sub_u32 v19, v1, v13 // (op 4839) 

v_lshlrev_b32 v13, 4, v19 // (op 4839) 

v_lshrrev_b32 v19, 3, v1 // (op 4839) 
v_lshl_add_u32 v22, v19, 7, v13 // (op 4839) 


s_nop 0
v_readfirstlane_b32 s6, v22 // (op 4839) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -128, v1 // (op 4904) 
v_sub_u32 v19, v1, v13 // (op 4904) 

v_lshrrev_b32 v13, 7, v1 // (op 4904) 
v_lshl_add_u32 v22, v13, 8, v19 // (op 4904) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 920) 
s_and_b32 s7, s18, 4294967294 // (op 920) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 919) 
s_mov_b32 s8, s18 // (op 919) convert

s_cmp_lt_u32 s7, s8 // (op 919) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_919 // (op 919) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_919 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_919:
s_barrier  // (op 5378) 
s_add_u32 s7, s7, 1 // (op 921) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 3840) 
v_mov_b64 v[24:25], s[18:19] // (op 3840) convert


v_mov_b64 v[18:19], v[26:27] // (op 3840) 
v_mov_b64 v[20:21], v[18:19] // (op 3840) reg expression
s_add_u32 s8, 4096, s15 // (op 3840) 
s_mov_b32 m0, s8 // (op 3840) reg expression
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 3840) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v24 // (op 3840) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 3840) most significant half

s_add_u32 m0, m0, 4096 // (op 3840) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 3840) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v24 // (op 3840) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 3840) most significant half

s_add_u32 m0, m0, 4096 // (op 3840) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 3840) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v24 // (op 3840) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 3840) most significant half

s_add_u32 m0, m0, 4096 // (op 3840) 
buffer_load_dwordx4 v20, s[68:71], 0 offen offset: 0 lds // (op 3840) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 7 // (op 3915) 
v_mov_b32 v13, s18 // (op 3915) 
v_mov_b32 v18, s19 // (op 3915) 
v_add_co_u32 v26, s[28:29], v26, v13 // (op 3915) least significant half
v_addc_co_u32 v27, s[28:29], v27, v18, s[28:29] // (op 3915) most significant half




s_barrier  // (op 5397) 


v_mov_b64 v[24:25], v[48:49] // (op 3846) 

s_mov_b32 m0, s6 // (op 3846) reg expression
buffer_load_dwordx4 v24, s[72:75], 0 offen offset: 0 lds // (op 3846) Load value direct to lds

v_add_co_u32 v48, vcc, 128, v48 // (op 4059) least significant half
v_addc_co_u32 v49, vcc, 0, v49, vcc // (op 4059) most significant half

v_mov_b64 v[24:25], s[38:39] // (op 460) convert

v_mov_b32 v13, v17 // (op 460) 
v_mov_b32 v23, v13 // (op 460) reg expression
buffer_load_ubyte v18, v23, s[76:79], 0 offen offset: 0 // (op 460) Load value
v_add_u32 v23, v23, v24 // (op 460) 
buffer_load_ubyte v19, v23, s[76:79], 0 offen offset: 0 // (op 460) Load value



s_lshl_b32 s8, s38, 2 // (op 4187) 
v_add_u32 v17, v17, s8 // (op 4187) 

s_barrier  // (op 5416) 
s_waitcnt vmcnt(3)
s_barrier  // (op 5384) 






ds_read_b64_tr_b8 v[28:29], v8 offset:4096 // (op 449) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v8 offset:5120 // (op 449) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v8 offset:12288 // (op 449) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v8 offset:13312 // (op 449) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v18 offset:40960 // (op 458) Store local data 
s_barrier  // (op 5403) 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v19 offset:41088 // (op 458) Store local data 




ds_read_b128 v[36:39], v4 // (op 451) Load local data 
ds_read_b128 v[40:43], v4 offset:64 // (op 451) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 5423) 


ds_read_u8 v13, v9 offset:40960 // (op 453) Load local data 


v_mov_b32 v18, v16 // (op 455) 

buffer_load_ubyte v19, v18, s[80:83], 0 offen offset: 0 // (op 455) Load value

v_add_u32 v16, 4, v16 // (op 4315) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:43], a[0:3], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 456) 
ds_read_b64_tr_b8 v[50:51], v8 offset:4128 // (op 601) Transpose load local data 


ds_read_b64_tr_b8 v[52:53], v8 offset:5152 // (op 601) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v8 offset:12320 // (op 601) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v8 offset:13344 // (op 601) Transpose load local data 


ds_read_u8 v13, v9 offset:40992 // (op 605) Load local data 



ds_read_b64_tr_b8 v[28:29], v8 offset:4160 // (op 763) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[36:43], a[4:7], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 607) 
ds_read_b64_tr_b8 v[30:31], v8 offset:5184 // (op 763) Transpose load local data 


ds_read_b64_tr_b8 v[32:33], v8 offset:12352 // (op 763) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v8 offset:13376 // (op 763) Transpose load local data 


ds_read_u8 v13, v9 offset:41024 // (op 767) Load local data 



ds_read_b64_tr_b8 v[50:51], v8 offset:4192 // (op 925) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v8 offset:5216 // (op 925) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:35], v[36:43], a[8:11], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 769) 
ds_read_b64_tr_b8 v[54:55], v8 offset:12384 // (op 925) Transpose load local data 


ds_read_b64_tr_b8 v[56:57], v8 offset:13408 // (op 925) Transpose load local data 


ds_read_u8 v13, v9 offset:41056 // (op 929) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[36:43], a[12:15], v13, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 931) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 919) 
s_mov_b32 s8, s18 // (op 919) convert

s_cmp_lt_u32 s7, s8 // (op 919) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_919 // (op 919) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_KLoopTail_919 if true)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_KLoopTail_919:








s_barrier  // (op 5563) 



v_cmp_eq_f32 vcc, 0.00000, v14 // (op 222) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_222 // (op 222) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_222
v_accvgpr_read v16, a0 // (op 230) 
v_accvgpr_read v17, a1 // (op 230) 
v_accvgpr_read v18, a2 // (op 230) 
v_accvgpr_read v19, a3 // (op 230) 
v_mul_f32 v20, v15, v16 // (op 230) 
v_mul_f32 v21, v15, v17 // (op 230) 
v_mul_f32 v22, v15, v18 // (op 230) 
v_mul_f32 v23, v15, v19 // (op 230) 

v_cvt_f16_f32 v16, v20 // (op 232) 
v_cvt_f16_f32 v17, v21 // (op 232) 
v_cvt_f16_f32 v18, v22 // (op 232) 
v_cvt_f16_f32 v19, v23 // (op 232) 


v_pack_b32_f16 v12, v16, v17 // (op 233) 
v_pack_b32_f16 v13, v18, v19 // (op 233) 


buffer_store_dwordx2 v[12:13], v7, s[60:63], 0 offen // (op 233) Store value


v_accvgpr_read v16, a4 // (op 1161) 
v_accvgpr_read v17, a5 // (op 1161) 
v_accvgpr_read v18, a6 // (op 1161) 
v_accvgpr_read v19, a7 // (op 1161) 
v_mul_f32 v20, v15, v16 // (op 1161) 
v_mul_f32 v21, v15, v17 // (op 1161) 
v_mul_f32 v22, v15, v18 // (op 1161) 
v_mul_f32 v23, v15, v19 // (op 1161) 

v_cvt_f16_f32 v16, v20 // (op 1163) 
v_cvt_f16_f32 v17, v21 // (op 1163) 
v_cvt_f16_f32 v18, v22 // (op 1163) 
v_cvt_f16_f32 v19, v23 // (op 1163) 


v_pack_b32_f16 v12, v16, v17 // (op 1164) 
v_pack_b32_f16 v13, v18, v19 // (op 1164) 


buffer_store_dwordx2 v[12:13], v7, s[60:63], 0 offen offset:64 // (op 1164) Store value


v_accvgpr_read v16, a8 // (op 1181) 
v_accvgpr_read v17, a9 // (op 1181) 
v_accvgpr_read v18, a10 // (op 1181) 
v_accvgpr_read v19, a11 // (op 1181) 
v_mul_f32 v20, v15, v16 // (op 1181) 
v_mul_f32 v21, v15, v17 // (op 1181) 
v_mul_f32 v22, v15, v18 // (op 1181) 
v_mul_f32 v23, v15, v19 // (op 1181) 

v_cvt_f16_f32 v16, v20 // (op 1183) 
v_cvt_f16_f32 v17, v21 // (op 1183) 
v_cvt_f16_f32 v18, v22 // (op 1183) 
v_cvt_f16_f32 v19, v23 // (op 1183) 


v_pack_b32_f16 v12, v16, v17 // (op 1184) 
v_pack_b32_f16 v13, v18, v19 // (op 1184) 


buffer_store_dwordx2 v[12:13], v7, s[60:63], 0 offen offset:128 // (op 1184) Store value


v_accvgpr_read v16, a12 // (op 1201) 
v_accvgpr_read v17, a13 // (op 1201) 
v_accvgpr_read v18, a14 // (op 1201) 
v_accvgpr_read v19, a15 // (op 1201) 
v_mul_f32 v20, v15, v16 // (op 1201) 
v_mul_f32 v21, v15, v17 // (op 1201) 
v_mul_f32 v22, v15, v18 // (op 1201) 
v_mul_f32 v23, v15, v19 // (op 1201) 

v_cvt_f16_f32 v16, v20 // (op 1203) 
v_cvt_f16_f32 v17, v21 // (op 1203) 
v_cvt_f16_f32 v18, v22 // (op 1203) 
v_cvt_f16_f32 v19, v23 // (op 1203) 


v_pack_b32_f16 v12, v16, v17 // (op 1204) 
v_pack_b32_f16 v13, v18, v19 // (op 1204) 


buffer_store_dwordx2 v[12:13], v7, s[60:63], 0 offen offset:192 // (op 1204) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_222 // (op 222) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_222
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalFalse_0_beta_222:



buffer_load_dwordx2 v[12:13], v6, s[56:59], 0 offen offset:192 // (op 1284) Load value
v_accvgpr_read v16, a12 // (op 1280) 
v_accvgpr_read v17, a13 // (op 1280) 
v_accvgpr_read v18, a14 // (op 1280) 
v_accvgpr_read v19, a15 // (op 1280) 
v_mul_f32 v20, v15, v16 // (op 1280) 
v_mul_f32 v21, v15, v17 // (op 1280) 
v_mul_f32 v22, v15, v18 // (op 1280) 
v_mul_f32 v23, v15, v19 // (op 1280) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v12 // (op 1285) 
v_lshrrev_b32 v17, 16, v12 // (op 1285) 
v_cvt_f32_f16 v17, v17 // (op 1285) 
v_mul_f32 v24, v14, v16 // (op 1285) 
v_mul_f32 v25, v14, v17 // (op 1285) 
v_cvt_f32_f16 v16, v13 // (op 1285) 
v_lshrrev_b32 v17, 16, v13 // (op 1285) 
v_cvt_f32_f16 v17, v17 // (op 1285) 
v_mul_f32 v26, v14, v16 // (op 1285) 
v_mul_f32 v27, v14, v17 // (op 1285) 


v_add_f32 v16, v24, v20 // (op 1281) 
v_add_f32 v17, v25, v21 // (op 1281) 
v_add_f32 v18, v26, v22 // (op 1281) 
v_add_f32 v19, v27, v23 // (op 1281) 


v_cvt_f16_f32 v20, v16 // (op 1282) 
v_cvt_f16_f32 v21, v17 // (op 1282) 
v_cvt_f16_f32 v22, v18 // (op 1282) 
v_cvt_f16_f32 v23, v19 // (op 1282) 


v_pack_b32_f16 v12, v20, v21 // (op 1283) 
v_pack_b32_f16 v13, v22, v23 // (op 1283) 


buffer_store_dwordx2 v[12:13], v7, s[60:63], 0 offen offset:192 // (op 1283) Store value





buffer_load_dwordx2 v[12:13], v6, s[56:59], 0 offen offset:128 // (op 1264) Load value
v_accvgpr_read v16, a8 // (op 1260) 
v_accvgpr_read v17, a9 // (op 1260) 
v_accvgpr_read v18, a10 // (op 1260) 
v_accvgpr_read v19, a11 // (op 1260) 
v_mul_f32 v20, v15, v16 // (op 1260) 
v_mul_f32 v21, v15, v17 // (op 1260) 
v_mul_f32 v22, v15, v18 // (op 1260) 
v_mul_f32 v23, v15, v19 // (op 1260) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v12 // (op 1265) 
v_lshrrev_b32 v17, 16, v12 // (op 1265) 
v_cvt_f32_f16 v17, v17 // (op 1265) 
v_mul_f32 v24, v14, v16 // (op 1265) 
v_mul_f32 v25, v14, v17 // (op 1265) 
v_cvt_f32_f16 v16, v13 // (op 1265) 
v_lshrrev_b32 v17, 16, v13 // (op 1265) 
v_cvt_f32_f16 v17, v17 // (op 1265) 
v_mul_f32 v26, v14, v16 // (op 1265) 
v_mul_f32 v27, v14, v17 // (op 1265) 


v_add_f32 v16, v24, v20 // (op 1261) 
v_add_f32 v17, v25, v21 // (op 1261) 
v_add_f32 v18, v26, v22 // (op 1261) 
v_add_f32 v19, v27, v23 // (op 1261) 


v_cvt_f16_f32 v20, v16 // (op 1262) 
v_cvt_f16_f32 v21, v17 // (op 1262) 
v_cvt_f16_f32 v22, v18 // (op 1262) 
v_cvt_f16_f32 v23, v19 // (op 1262) 


v_pack_b32_f16 v12, v20, v21 // (op 1263) 
v_pack_b32_f16 v13, v22, v23 // (op 1263) 


buffer_store_dwordx2 v[12:13], v7, s[60:63], 0 offen offset:128 // (op 1263) Store value





buffer_load_dwordx2 v[12:13], v6, s[56:59], 0 offen offset:64 // (op 1244) Load value
v_accvgpr_read v16, a4 // (op 1240) 
v_accvgpr_read v17, a5 // (op 1240) 
v_accvgpr_read v18, a6 // (op 1240) 
v_accvgpr_read v19, a7 // (op 1240) 
v_mul_f32 v20, v15, v16 // (op 1240) 
v_mul_f32 v21, v15, v17 // (op 1240) 
v_mul_f32 v22, v15, v18 // (op 1240) 
v_mul_f32 v23, v15, v19 // (op 1240) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v12 // (op 1245) 
v_lshrrev_b32 v17, 16, v12 // (op 1245) 
v_cvt_f32_f16 v17, v17 // (op 1245) 
v_mul_f32 v24, v14, v16 // (op 1245) 
v_mul_f32 v25, v14, v17 // (op 1245) 
v_cvt_f32_f16 v16, v13 // (op 1245) 
v_lshrrev_b32 v17, 16, v13 // (op 1245) 
v_cvt_f32_f16 v17, v17 // (op 1245) 
v_mul_f32 v26, v14, v16 // (op 1245) 
v_mul_f32 v27, v14, v17 // (op 1245) 


v_add_f32 v16, v24, v20 // (op 1241) 
v_add_f32 v17, v25, v21 // (op 1241) 
v_add_f32 v18, v26, v22 // (op 1241) 
v_add_f32 v19, v27, v23 // (op 1241) 


v_cvt_f16_f32 v20, v16 // (op 1242) 
v_cvt_f16_f32 v21, v17 // (op 1242) 
v_cvt_f16_f32 v22, v18 // (op 1242) 
v_cvt_f16_f32 v23, v19 // (op 1242) 


v_pack_b32_f16 v12, v20, v21 // (op 1243) 
v_pack_b32_f16 v13, v22, v23 // (op 1243) 


buffer_store_dwordx2 v[12:13], v7, s[60:63], 0 offen offset:64 // (op 1243) Store value





buffer_load_dwordx2 v[12:13], v6, s[56:59], 0 offen // (op 247) Load value
v_accvgpr_read v16, a0 // (op 243) 
v_accvgpr_read v17, a1 // (op 243) 
v_accvgpr_read v18, a2 // (op 243) 
v_accvgpr_read v19, a3 // (op 243) 
v_mul_f32 v20, v15, v16 // (op 243) 
v_mul_f32 v21, v15, v17 // (op 243) 
v_mul_f32 v22, v15, v18 // (op 243) 
v_mul_f32 v23, v15, v19 // (op 243) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v12 // (op 248) 
v_lshrrev_b32 v17, 16, v12 // (op 248) 
v_cvt_f32_f16 v17, v17 // (op 248) 
v_mul_f32 v24, v14, v16 // (op 248) 
v_mul_f32 v25, v14, v17 // (op 248) 
v_cvt_f32_f16 v16, v13 // (op 248) 
v_lshrrev_b32 v17, 16, v13 // (op 248) 
v_cvt_f32_f16 v17, v17 // (op 248) 
v_mul_f32 v26, v14, v16 // (op 248) 
v_mul_f32 v27, v14, v17 // (op 248) 


v_add_f32 v16, v24, v20 // (op 244) 
v_add_f32 v17, v25, v21 // (op 244) 
v_add_f32 v18, v26, v22 // (op 244) 
v_add_f32 v19, v27, v23 // (op 244) 


v_cvt_f16_f32 v20, v16 // (op 245) 
v_cvt_f16_f32 v21, v17 // (op 245) 
v_cvt_f16_f32 v22, v18 // (op 245) 
v_cvt_f16_f32 v23, v19 // (op 245) 


v_pack_b32_f16 v12, v20, v21 // (op 246) 
v_pack_b32_f16 v13, v22, v23 // (op 246) 


buffer_store_dwordx2 v[12:13], v7, s[60:63], 0 offen // (op 246) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ConditionalBottom_0_beta_222:
s_add_u32 s0, s0, s1 // (op 435) 


s_mul_i32 s15, s50, s16 // (op 433) 
s_cmp_gt_u32 s51, s15 // (op 433) 
s_mov_b32 s6, scc // (op 433) 
s_sub_u32 s7, s51, s15 // (op 433) 

s_cmp_eq_i32 s6, 0 // (op 433) 
s_cselect_b32 s8, 0, s7 // (op 433) 


s_cmp_lt_u32 s50, s8 // (op 433) 
s_mov_b32 s15, scc // (op 433) 
s_cmp_eq_i32 s15, 0 // (op 433) 
s_cselect_b32 s6, s8, s50 // (op 433) 


s_cmp_lt_u32 s0, s6 // (op 433) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_433 // (op 433) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopTop_DPStreamTileLoop_433 if true)
RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_ForLoopBottom_DPStreamTileLoop_433:











































s_endpgm  // End of RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.LRR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_end:
.size RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2, .LRR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2_end-RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
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
    DataTypeB: B8
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
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
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
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
  - .name: RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2
    .symbol: RR_GEMM_NN_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x32x128_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 41984
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 89
    .vgpr_count: 92
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_9
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
      - .name: Tensor_0_stride_1_8
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_11
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_14
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_15
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: Tensor_7_stride_1_17
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

