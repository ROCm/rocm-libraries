// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4,@function
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 21575) 
v_and_b32 v3, 15, v1 // (op 21575) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 21575) 


v_and_b32 v0, 63, v1 // (op 21575) 
v_lshrrev_b32 v3, 4, v0 // (op 21575) 

v_lshlrev_b32 v0, 5, v3 // (op 21575) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 21575) 


v_lshlrev_b32 v0, 2, v3 // (op 21575) 

v_lshrrev_b32 v3, 3, v0 // (op 21575) 

v_lshrrev_b32 v0, 6, v1 // (op 21650) 
v_and_b32 v4, 1, v0 // (op 21650) 

v_and_b32 v0, 15, v1 // (op 21650) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 21650) 


v_and_b32 v0, 63, v1 // (op 21650) 
v_lshrrev_b32 v4, 4, v0 // (op 21650) 

v_lshlrev_b32 v0, 5, v4 // (op 21650) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 21650) 


v_lshlrev_b32 v0, 2, v4 // (op 21650) 

v_lshrrev_b32 v4, 3, v0 // (op 21650) 

v_lshrrev_b32 v0, 7, v1 // (op 21777) 
v_and_b32 v5, 15, v1 // (op 21777) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 21777) 


v_and_b32 v0, 63, v1 // (op 21777) 
v_lshrrev_b32 v5, 4, v0 // (op 21777) 

v_lshlrev_b32 v0, 5, v5 // (op 21777) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 21777) 


v_lshlrev_b32 v0, 2, v5 // (op 21777) 

v_lshrrev_b32 v5, 3, v0 // (op 21777) 

v_lshrrev_b32 v0, 6, v1 // (op 21852) 
v_and_b32 v6, 1, v0 // (op 21852) 

v_and_b32 v0, 15, v1 // (op 21852) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 21852) 


v_and_b32 v0, 63, v1 // (op 21852) 
v_lshrrev_b32 v6, 4, v0 // (op 21852) 

v_lshlrev_b32 v0, 5, v6 // (op 21852) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 21852) 


v_lshlrev_b32 v0, 2, v6 // (op 21852) 

v_lshrrev_b32 v6, 3, v0 // (op 21852) 

v_lshrrev_b32 v0, 7, v1 // (op 21966) 
v_and_b32 v7, 15, v1 // (op 21966) 
v_lshl_add_u32 v8, v0, 6, v7 // (op 21966) 


v_and_b32 v0, 63, v1 // (op 21966) 
v_lshrrev_b32 v7, 4, v0 // (op 21966) 

v_lshlrev_b32 v0, 5, v7 // (op 21966) 

v_lshl_add_u32 v7, v8, 7, v0 // (op 21966) 


v_lshlrev_b32 v0, 2, v7 // (op 21966) 

v_lshrrev_b32 v7, 3, v0 // (op 21966) 

v_lshrrev_b32 v0, 6, v1 // (op 22010) 
v_and_b32 v8, 1, v0 // (op 22010) 

v_and_b32 v0, 15, v1 // (op 22010) 
v_lshl_add_u32 v9, v8, 6, v0 // (op 22010) 


v_and_b32 v0, 63, v1 // (op 22010) 
v_lshrrev_b32 v8, 4, v0 // (op 22010) 

v_lshlrev_b32 v0, 5, v8 // (op 22010) 

v_lshl_add_u32 v8, v9, 7, v0 // (op 22010) 


v_lshlrev_b32 v0, 2, v8 // (op 22010) 

v_lshrrev_b32 v8, 3, v0 // (op 22010) 

v_lshrrev_b32 v0, 7, v1 // (op 22054) 
v_and_b32 v9, 15, v1 // (op 22054) 
v_lshl_add_u32 v10, v0, 6, v9 // (op 22054) 


v_and_b32 v0, 63, v1 // (op 22054) 
v_lshrrev_b32 v9, 4, v0 // (op 22054) 

v_lshlrev_b32 v0, 5, v9 // (op 22054) 

v_lshl_add_u32 v9, v10, 7, v0 // (op 22054) 


v_lshlrev_b32 v0, 2, v9 // (op 22054) 

v_lshrrev_b32 v9, 3, v0 // (op 22054) 

v_lshrrev_b32 v0, 6, v1 // (op 22098) 
v_and_b32 v10, 1, v0 // (op 22098) 

v_and_b32 v0, 15, v1 // (op 22098) 
v_lshl_add_u32 v11, v10, 6, v0 // (op 22098) 


v_and_b32 v0, 63, v1 // (op 22098) 
v_lshrrev_b32 v10, 4, v0 // (op 22098) 

v_lshlrev_b32 v0, 5, v10 // (op 22098) 

v_lshl_add_u32 v10, v11, 7, v0 // (op 22098) 


v_lshlrev_b32 v0, 2, v10 // (op 22098) 

v_lshrrev_b32 v10, 3, v0 // (op 22098) 

v_lshrrev_b32 v0, 7, v1 // (op 22142) 
v_and_b32 v11, 15, v1 // (op 22142) 
v_lshl_add_u32 v12, v0, 6, v11 // (op 22142) 


v_and_b32 v0, 63, v1 // (op 22142) 
v_lshrrev_b32 v11, 4, v0 // (op 22142) 

v_lshlrev_b32 v0, 5, v11 // (op 22142) 

v_lshl_add_u32 v11, v12, 7, v0 // (op 22142) 


v_lshlrev_b32 v0, 2, v11 // (op 22142) 

v_lshrrev_b32 v11, 3, v0 // (op 22142) 

v_lshrrev_b32 v0, 6, v1 // (op 22186) 
v_and_b32 v12, 1, v0 // (op 22186) 

v_and_b32 v0, 15, v1 // (op 22186) 
v_lshl_add_u32 v13, v12, 6, v0 // (op 22186) 


v_and_b32 v0, 63, v1 // (op 22186) 
v_lshrrev_b32 v12, 4, v0 // (op 22186) 

v_lshlrev_b32 v0, 5, v12 // (op 22186) 

v_lshl_add_u32 v12, v13, 7, v0 // (op 22186) 


v_lshlrev_b32 v0, 2, v12 // (op 22186) 

v_lshrrev_b32 v12, 3, v0 // (op 22186) 

v_lshrrev_b32 v0, 7, v1 // (op 22230) 
v_and_b32 v13, 15, v1 // (op 22230) 
v_lshl_add_u32 v14, v0, 6, v13 // (op 22230) 


v_and_b32 v0, 63, v1 // (op 22230) 
v_lshrrev_b32 v13, 4, v0 // (op 22230) 

v_lshlrev_b32 v0, 5, v13 // (op 22230) 

v_lshl_add_u32 v13, v14, 7, v0 // (op 22230) 


v_lshlrev_b32 v0, 2, v13 // (op 22230) 

v_lshrrev_b32 v13, 3, v0 // (op 22230) 

v_lshrrev_b32 v0, 6, v1 // (op 22274) 
v_and_b32 v14, 1, v0 // (op 22274) 

v_and_b32 v0, 15, v1 // (op 22274) 
v_lshl_add_u32 v15, v14, 6, v0 // (op 22274) 


v_and_b32 v0, 63, v1 // (op 22274) 
v_lshrrev_b32 v14, 4, v0 // (op 22274) 

v_lshlrev_b32 v0, 5, v14 // (op 22274) 

v_lshl_add_u32 v14, v15, 7, v0 // (op 22274) 


v_lshlrev_b32 v0, 2, v14 // (op 22274) 

v_lshrrev_b32 v14, 3, v0 // (op 22274) 

v_lshrrev_b32 v0, 7, v1 // (op 22318) 
v_and_b32 v15, 15, v1 // (op 22318) 
v_lshl_add_u32 v16, v0, 6, v15 // (op 22318) 


v_and_b32 v0, 63, v1 // (op 22318) 
v_lshrrev_b32 v15, 4, v0 // (op 22318) 

v_lshlrev_b32 v0, 5, v15 // (op 22318) 

v_lshl_add_u32 v15, v16, 7, v0 // (op 22318) 


v_lshlrev_b32 v0, 2, v15 // (op 22318) 

v_lshrrev_b32 v15, 3, v0 // (op 22318) 

v_lshrrev_b32 v0, 6, v1 // (op 22362) 
v_and_b32 v16, 1, v0 // (op 22362) 

v_and_b32 v0, 15, v1 // (op 22362) 
v_lshl_add_u32 v17, v16, 6, v0 // (op 22362) 


v_and_b32 v0, 63, v1 // (op 22362) 
v_lshrrev_b32 v16, 4, v0 // (op 22362) 

v_lshlrev_b32 v0, 5, v16 // (op 22362) 

v_lshl_add_u32 v16, v17, 7, v0 // (op 22362) 


v_lshlrev_b32 v0, 2, v16 // (op 22362) 

v_lshrrev_b32 v16, 3, v0 // (op 22362) 

v_lshrrev_b32 v0, 7, v1 // (op 22406) 
v_and_b32 v17, 15, v1 // (op 22406) 
v_lshl_add_u32 v18, v0, 6, v17 // (op 22406) 


v_and_b32 v0, 63, v1 // (op 22406) 
v_lshrrev_b32 v17, 4, v0 // (op 22406) 

v_lshlrev_b32 v0, 5, v17 // (op 22406) 

v_lshl_add_u32 v17, v18, 7, v0 // (op 22406) 


v_lshlrev_b32 v0, 2, v17 // (op 22406) 

v_lshrrev_b32 v17, 3, v0 // (op 22406) 

v_lshrrev_b32 v0, 6, v1 // (op 22450) 
v_and_b32 v18, 1, v0 // (op 22450) 

v_and_b32 v0, 15, v1 // (op 22450) 
v_lshl_add_u32 v19, v18, 6, v0 // (op 22450) 


v_and_b32 v0, 63, v1 // (op 22450) 
v_lshrrev_b32 v18, 4, v0 // (op 22450) 

v_lshlrev_b32 v0, 5, v18 // (op 22450) 

v_lshl_add_u32 v18, v19, 7, v0 // (op 22450) 


v_lshlrev_b32 v0, 2, v18 // (op 22450) 

v_lshrrev_b32 v18, 3, v0 // (op 22450) 

v_mov_b32 v0, s6 // (op 2) Move value
v_mov_b32 v19, s7 // (op 35) Move value


s_mov_b32 s0, 0 // (op 364) call()
s_mov_b32 s1, 0 // (op 365) call()
s_mov_b32 s6, 0 // (op 366) call()
s_mov_b32 s7, 0 // (op 367) call()
s_mov_b32 s18, 0 // (op 368) call()
s_mov_b32 s19, 0 // (op 369) call()
s_mov_b32 s0, 0 // (op 264) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 263) 
s_cmp_gt_u32 s9, s52 // (op 263) 
s_mov_b32 s53, scc // (op 263) 
s_sub_u32 s54, s9, s52 // (op 263) 

s_cmp_eq_i32 s53, 0 // (op 263) 
s_cselect_b32 s52, 0, s54 // (op 263) 


s_cmp_lt_u32 s8, s52 // (op 263) 
s_mov_b32 s53, scc // (op 263) 
s_cmp_eq_i32 s53, 0 // (op 263) 
s_cselect_b32 s54, s52, s8 // (op 263) 


s_cmp_lt_u32 s0, s54 // (op 263) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_263 // (op 263) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_263 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_263:
v_lshrrev_b32 v20, 7, v1 // (op 21486) 
s_mul_i32 s52, s16, s8 // (op 21486) 
s_add_u32 s52, s52, s0 // (op 21486) 
v_and_b32 v21, 63, v1 // (op 21486) 
s_mul_hi_u32 s53, s52, s13 // (op 21486) 
v_lshrrev_b32 v22, 4, v21 // (op 21486) 
s_sub_u32 s54, s52, s53 // (op 21486) 
v_lshlrev_b32 v23, 2, v22 // (op 21486) 

v_lshl_add_u32 v22, v20, 6, v23 // (op 21486) 


s_lshr_b32 s55, s54, 1 // (op 21486) 

s_add_u32 s54, s55, s53 // (op 21486) 


v_lshrrev_b32 v20, 2, v21 // (op 21486) 

s_lshr_b32 s53, s54, s11 // (op 21486) 

v_and_b32 v21, 3, v20 // (op 21486) 

s_and_b32 s54, s11, 2147483648 // (op 21486) 
v_and_b32 v20, 3, v1 // (op 21486) 
s_cmp_eq_i32 s54, 0 // (op 21486) 
s_mov_b32 s55, scc // (op 21486) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 21486) 


s_cmp_eq_i32 s55, 0 // (op 21486) 
s_cselect_b32 s54, s52, s53 // (op 21486) 



v_lshrrev_b32 v20, 6, v1 // (op 21486) 
s_mul_hi_u32 s52, s54, s12 // (op 21486) 
v_and_b32 v21, 1, v20 // (op 21486) 

s_sub_u32 s53, s54, s52 // (op 21486) 
v_lshl_add_u32 v20, v21, 6, v23 // (op 21486) 


s_lshr_b32 s55, s53, 1 // (op 21486) 

s_add_u32 s53, s55, s52 // (op 21486) 


s_lshr_b32 s52, s53, s10 // (op 21486) 

s_and_b32 s53, s10, 2147483648 // (op 21486) 
s_cmp_eq_i32 s53, 0 // (op 21486) 
s_mov_b32 s55, scc // (op 21486) 

s_cmp_eq_i32 s55, 0 // (op 21486) 
s_cselect_b32 s53, s54, s52 // (op 21486) 


s_mul_i32 s52, s53, s14 // (op 21486) 
v_lshl_add_u32 v21, s53, 7, v22 // (op 21486) 


s_sub_u32 s53, s54, s52 // (op 21486) 


v_lshl_add_u32 v22, s53, 7, v20 // (op 21486) 


v_mul_lo_u32 v20, v22, s4 // (op 21486) 

v_add_lshl_u32 v22, v21, v20, 1 // (op 21486) 


s_mul_i32 s53, s4, s23 // (op 21488) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 21488) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 21488) least significant
s_mul_hi_u32 s55, s4, s22 // (op 21488) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21488) most significant: sum
s_add_u32 s53, s53, s55 // (op 21488) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 21488) 

s_mov_b32 s56, s20 // (op 21488) reg expression
s_mov_b32 s57, s21 // (op 21488) reg expression
s_mov_b32 s58, s54 // (op 21488) reg expression
s_mov_b32 s59, 131072 // (op 21488) call()

v_lshrrev_b32 v20, 7, v1 // (op 21592) 
v_and_b32 v21, 15, v1 // (op 21592) 
v_lshl_add_u32 v23, v20, 6, v21 // (op 21592) 


v_and_b32 v20, 63, v1 // (op 21592) 
v_lshrrev_b32 v21, 4, v20 // (op 21592) 

v_lshlrev_b32 v20, 5, v21 // (op 21592) 

v_lshl_add_u32 v21, v23, 7, v20 // (op 21592) 


v_lshlrev_b32 v20, 2, v21 // (op 21592) 

v_lshrrev_b32 v21, 3, v20 // (op 21592) 

v_lshrrev_b32 v20, 6, v1 // (op 21667) 
v_and_b32 v23, 1, v20 // (op 21667) 

v_and_b32 v20, 15, v1 // (op 21667) 
v_lshl_add_u32 v24, v23, 6, v20 // (op 21667) 


v_and_b32 v20, 63, v1 // (op 21667) 
v_lshrrev_b32 v23, 4, v20 // (op 21667) 

v_lshlrev_b32 v20, 5, v23 // (op 21667) 

v_lshl_add_u32 v23, v24, 7, v20 // (op 21667) 


v_lshlrev_b32 v20, 2, v23 // (op 21667) 

v_lshrrev_b32 v23, 3, v20 // (op 21667) 

v_lshrrev_b32 v20, 7, v1 // (op 21690) 
s_mul_i32 s52, s16, s8 // (op 21690) 
s_add_u32 s52, s52, s0 // (op 21690) 
v_and_b32 v24, 63, v1 // (op 21690) 
s_mul_hi_u32 s53, s52, s13 // (op 21690) 
v_lshrrev_b32 v25, 4, v24 // (op 21690) 
s_sub_u32 s54, s52, s53 // (op 21690) 
v_lshlrev_b32 v26, 2, v25 // (op 21690) 

s_lshr_b32 s55, s54, 1 // (op 21690) 

v_lshl_add_u32 v25, v20, 6, v26 // (op 21690) 


s_add_u32 s54, s55, s53 // (op 21690) 


v_lshrrev_b32 v20, 2, v24 // (op 21690) 

s_lshr_b32 s53, s54, s11 // (op 21690) 

v_and_b32 v24, 3, v20 // (op 21690) 

s_and_b32 s54, s11, 2147483648 // (op 21690) 
v_and_b32 v20, 3, v1 // (op 21690) 
s_cmp_eq_i32 s54, 0 // (op 21690) 
s_mov_b32 s55, scc // (op 21690) 

v_lshl_add_u32 v26, v24, 2, v20 // (op 21690) 


s_cmp_eq_i32 s55, 0 // (op 21690) 
s_cselect_b32 s54, s52, s53 // (op 21690) 



v_lshrrev_b32 v20, 6, v1 // (op 21690) 
s_mul_hi_u32 s55, s54, s12 // (op 21690) 
v_and_b32 v24, 1, v20 // (op 21690) 

s_sub_u32 s52, s54, s55 // (op 21690) 
v_lshl_add_u32 v20, v24, 6, v26 // (op 21690) 


s_lshr_b32 s53, s52, 1 // (op 21690) 

s_add_u32 s52, s53, s55 // (op 21690) 


s_lshr_b32 s53, s52, s10 // (op 21690) 

s_and_b32 s52, s10, 2147483648 // (op 21690) 
s_cmp_eq_i32 s52, 0 // (op 21690) 
s_mov_b32 s55, scc // (op 21690) 

s_cmp_eq_i32 s55, 0 // (op 21690) 
s_cselect_b32 s52, s54, s53 // (op 21690) 


s_mul_i32 s53, s52, s14 // (op 21690) 
v_lshl_add_u32 v24, s52, 7, v25 // (op 21690) 


s_sub_u32 s52, s54, s53 // (op 21690) 


v_lshl_add_u32 v25, s52, 7, v20 // (op 21690) 


v_mul_lo_u32 v20, v25, s24 // (op 21690) 

v_add_lshl_u32 v25, v24, v20, 1 // (op 21690) 


s_mul_i32 s53, s24, s23 // (op 21692) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 21692) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 21692) least significant
s_mul_hi_u32 s55, s24, s22 // (op 21692) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21692) most significant: sum
s_add_u32 s53, s53, s55 // (op 21692) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 21692) 

s_mov_b32 s60, s26 // (op 21692) reg expression
s_mov_b32 s61, s27 // (op 21692) reg expression
s_mov_b32 s62, s54 // (op 21692) reg expression
s_mov_b32 s63, 131072 // (op 21692) call()

v_lshrrev_b32 v20, 5, v1 // (op 21892) 
v_lshlrev_b32 v24, 2, v20 // (op 21892) 

v_lshl_add_u32 v20, s16, 7, v24 // (op 21892) 

v_and_b32 v24, -32, v1 // (op 21892) 
v_sub_u32 v26, v1, v24 // (op 21892) 

v_lshl_add_u32 v24, v20, 7, v26 // (op 21892) 


v_lshlrev_b32 v20, 2, v24 // (op 21892) 

s_lshl_b32 s52, s15, 16 // (op 21894) 
s_mov_b32 s64, s28 // (op 21894) reg expression
s_mov_b32 s65, s29 // (op 21894) reg expression
s_mov_b32 s66, s52 // (op 21894) reg expression
s_mov_b32 s67, 131072 // (op 21894) call()

s_mul_i32 s6, s8, s16 // (op 268) 
s_add_u32 s6, s6, s0 // (op 268) 
s_mul_hi_u32 s52, s6, s13 // (op 269) 
s_sub_u32 s53, s6, s52 // (op 269) 
s_lshr_b32 s54, s53, 1 // (op 269) 

s_add_u32 s53, s54, s52 // (op 269) 


s_lshr_b32 s52, s53, s11 // (op 269) 

s_and_b32 s53, s11, 2147483648 // (op 269) 
s_cmp_eq_i32 s53, 0 // (op 269) 
s_mov_b32 s54, scc // (op 269) 

s_cmp_eq_i32 s54, 0 // (op 269) 
s_cselect_b32 s53, s6, s52 // (op 269) 


s_ashr_i64 s[54:55], s[2:3], 7 // (op 269) 
s_mul_i32 s52, s53, s54 // (op 269) 


s_sub_u32 s18, s6, s52 // (op 269) 

s_ashr_i64 s[52:53], s[2:3], 7 // (op 270) 
s_cmp_gt_u32 s52, s18 // (op 270) 
s_mov_b32 s54, scc // (op 270) 
s_sub_u32 s55, s52, s18 // (op 270) 

s_cmp_eq_i32 s54, 0 // (op 270) 
s_cselect_b32 s52, 0, s55 // (op 270) 


s_cmp_gt_u32 s8, s0 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_sub_u32 s54, s8, s0 // (op 270) 
s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s55, 0, s54 // (op 270) 


s_cmp_lt_u32 s52, s55 // (op 270) 
s_mov_b32 s53, scc // (op 270) 
s_cmp_eq_i32 s53, 0 // (op 270) 
s_cselect_b32 s54, s55, s52 // (op 270) 



s_cmp_gt_u32 s9, s6 // (op 270) 
s_mov_b32 s55, scc // (op 270) 
s_sub_u32 s52, s9, s6 // (op 270) 
s_cmp_eq_i32 s55, 0 // (op 270) 
s_cselect_b32 s53, 0, s52 // (op 270) 


s_cmp_lt_u32 s54, s53 // (op 270) 
s_mov_b32 s52, scc // (op 270) 
s_cmp_eq_i32 s52, 0 // (op 270) 
s_cselect_b32 s7, s53, s54 // (op 270) 



s_add_u32 s52, s18, s7 // (op 271) 
s_sub_u32 s19, s52, 1 // (op 271) 

v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_accvgpr_write a4, 0.00000 // (op 5989) call()
v_accvgpr_write a5, 0.00000 // (op 5989) call()
v_accvgpr_write a6, 0.00000 // (op 5989) call()
v_accvgpr_write a7, 0.00000 // (op 5989) call()
v_accvgpr_write a8, 0.00000 // (op 6215) call()
v_accvgpr_write a9, 0.00000 // (op 6215) call()
v_accvgpr_write a10, 0.00000 // (op 6215) call()
v_accvgpr_write a11, 0.00000 // (op 6215) call()
v_accvgpr_write a12, 0.00000 // (op 6441) call()
v_accvgpr_write a13, 0.00000 // (op 6441) call()
v_accvgpr_write a14, 0.00000 // (op 6441) call()
v_accvgpr_write a15, 0.00000 // (op 6441) call()
v_accvgpr_write a16, 0.00000 // (op 6803) call()
v_accvgpr_write a17, 0.00000 // (op 6803) call()
v_accvgpr_write a18, 0.00000 // (op 6803) call()
v_accvgpr_write a19, 0.00000 // (op 6803) call()
v_accvgpr_write a20, 0.00000 // (op 7216) call()
v_accvgpr_write a21, 0.00000 // (op 7216) call()
v_accvgpr_write a22, 0.00000 // (op 7216) call()
v_accvgpr_write a23, 0.00000 // (op 7216) call()
v_accvgpr_write a24, 0.00000 // (op 7217) call()
v_accvgpr_write a25, 0.00000 // (op 7217) call()
v_accvgpr_write a26, 0.00000 // (op 7217) call()
v_accvgpr_write a27, 0.00000 // (op 7217) call()
v_accvgpr_write a28, 0.00000 // (op 7218) call()
v_accvgpr_write a29, 0.00000 // (op 7218) call()
v_accvgpr_write a30, 0.00000 // (op 7218) call()
v_accvgpr_write a31, 0.00000 // (op 7218) call()
v_accvgpr_write a32, 0.00000 // (op 7805) call()
v_accvgpr_write a33, 0.00000 // (op 7805) call()
v_accvgpr_write a34, 0.00000 // (op 7805) call()
v_accvgpr_write a35, 0.00000 // (op 7805) call()
v_accvgpr_write a36, 0.00000 // (op 8218) call()
v_accvgpr_write a37, 0.00000 // (op 8218) call()
v_accvgpr_write a38, 0.00000 // (op 8218) call()
v_accvgpr_write a39, 0.00000 // (op 8218) call()
v_accvgpr_write a40, 0.00000 // (op 8219) call()
v_accvgpr_write a41, 0.00000 // (op 8219) call()
v_accvgpr_write a42, 0.00000 // (op 8219) call()
v_accvgpr_write a43, 0.00000 // (op 8219) call()
v_accvgpr_write a44, 0.00000 // (op 8220) call()
v_accvgpr_write a45, 0.00000 // (op 8220) call()
v_accvgpr_write a46, 0.00000 // (op 8220) call()
v_accvgpr_write a47, 0.00000 // (op 8220) call()
v_accvgpr_write a48, 0.00000 // (op 8807) call()
v_accvgpr_write a49, 0.00000 // (op 8807) call()
v_accvgpr_write a50, 0.00000 // (op 8807) call()
v_accvgpr_write a51, 0.00000 // (op 8807) call()
v_accvgpr_write a52, 0.00000 // (op 9220) call()
v_accvgpr_write a53, 0.00000 // (op 9220) call()
v_accvgpr_write a54, 0.00000 // (op 9220) call()
v_accvgpr_write a55, 0.00000 // (op 9220) call()
v_accvgpr_write a56, 0.00000 // (op 9221) call()
v_accvgpr_write a57, 0.00000 // (op 9221) call()
v_accvgpr_write a58, 0.00000 // (op 9221) call()
v_accvgpr_write a59, 0.00000 // (op 9221) call()
v_accvgpr_write a60, 0.00000 // (op 9222) call()
v_accvgpr_write a61, 0.00000 // (op 9222) call()
v_accvgpr_write a62, 0.00000 // (op 9222) call()
v_accvgpr_write a63, 0.00000 // (op 9222) call()
v_lshrrev_b32 v24, 2, v1 // (op 21309) 
s_mul_i32 s52, s16, s8 // (op 21309) 
s_add_u32 s52, s52, s0 // (op 21309) 
v_and_b32 v26, -4, v1 // (op 21309) 
s_mul_hi_u32 s53, s52, s13 // (op 21309) 
v_sub_u32 v27, v1, v26 // (op 21309) 

s_sub_u32 s54, s52, s53 // (op 21309) 
v_lshlrev_b32 v26, 5, v27 // (op 21309) 

s_lshr_b32 s55, s54, 1 // (op 21309) 

s_add_u32 s54, s55, s53 // (op 21309) 


s_lshr_b32 s53, s54, s11 // (op 21309) 

s_and_b32 s54, s11, 2147483648 // (op 21309) 
s_cmp_eq_i32 s54, 0 // (op 21309) 
s_mov_b32 s55, scc // (op 21309) 

s_cmp_eq_i32 s55, 0 // (op 21309) 
s_cselect_b32 s54, s52, s53 // (op 21309) 


s_mul_hi_u32 s53, s54, s12 // (op 21309) 
s_sub_u32 s55, s54, s53 // (op 21309) 
s_lshr_b32 s68, s55, 1 // (op 21309) 

s_add_u32 s55, s68, s53 // (op 21309) 


s_lshr_b32 s53, s55, s10 // (op 21309) 

s_and_b32 s55, s10, 2147483648 // (op 21309) 
s_cmp_eq_i32 s55, 0 // (op 21309) 
s_mov_b32 s68, scc // (op 21309) 

s_cmp_eq_i32 s68, 0 // (op 21309) 
s_cselect_b32 s55, s54, s53 // (op 21309) 


v_lshl_add_u32 v27, s55, 7, v24 // (op 21309) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 21309) 
s_mul_i32 s53, s54, s68 // (op 21309) 


s_sub_u32 s54, s52, s53 // (op 21309) 


v_lshl_add_u32 v24, s54, 7, v26 // (op 21309) 


v_mov_b32 v29, 0 // (op 21309) convert
v_mov_b32 v28, v24 // (op 21309) convert

v_mul_lo_u32 v31, v27, s31 // (op 21309) most significant: low of low * high
v_mul_hi_u32 v24, v27, s30 // (op 21309) most significant: high of low * low
v_mul_lo_u32 v30, v27, s30 // (op 21309) least significant: low of low * low
v_add_u32 v31, v31, v24 // (op 21309) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v28 // (op 21309) least significant half
v_addc_co_u32 v31, s[52:53], v31, v29, s[52:53] // (op 21309) most significant half



v_lshlrev_b64 v[26:27], 2, v[30:31] // (op 21309) 

v_lshrrev_b64 v[28:29], 3, v[26:27] // (op 21309) 

s_mul_i32 s53, s30, s35 // (op 21311) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 21311) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 21311) least significant
s_mul_hi_u32 s55, s30, s34 // (op 21311) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21311) most significant: sum
s_add_u32 s53, s53, s55 // (op 21311) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21311) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21311) 

s_mov_b32 s68, s32 // (op 21311) reg expression
s_mov_b32 s69, s33 // (op 21311) reg expression
s_mov_b32 s70, s52 // (op 21311) reg expression
s_mov_b32 s71, 131072 // (op 21311) call()

v_lshrrev_b32 v24, 2, v1 // (op 21328) 
s_mul_i32 s52, s16, s8 // (op 21328) 
s_add_u32 s52, s52, s0 // (op 21328) 
v_and_b32 v26, -4, v1 // (op 21328) 
s_mul_hi_u32 s53, s52, s13 // (op 21328) 
v_sub_u32 v27, v1, v26 // (op 21328) 

s_sub_u32 s54, s52, s53 // (op 21328) 
v_lshlrev_b32 v26, 5, v27 // (op 21328) 

s_lshr_b32 s55, s54, 1 // (op 21328) 

s_add_u32 s54, s55, s53 // (op 21328) 


s_lshr_b32 s53, s54, s11 // (op 21328) 

s_and_b32 s54, s11, 2147483648 // (op 21328) 
s_cmp_eq_i32 s54, 0 // (op 21328) 
s_mov_b32 s55, scc // (op 21328) 

s_cmp_eq_i32 s55, 0 // (op 21328) 
s_cselect_b32 s54, s52, s53 // (op 21328) 


s_mul_hi_u32 s53, s54, s12 // (op 21328) 
s_sub_u32 s55, s54, s53 // (op 21328) 
s_lshr_b32 s72, s55, 1 // (op 21328) 

s_add_u32 s55, s72, s53 // (op 21328) 


s_lshr_b32 s53, s55, s10 // (op 21328) 

s_and_b32 s55, s10, 2147483648 // (op 21328) 
s_cmp_eq_i32 s55, 0 // (op 21328) 
s_mov_b32 s72, scc // (op 21328) 

s_cmp_eq_i32 s72, 0 // (op 21328) 
s_cselect_b32 s55, s54, s53 // (op 21328) 


v_lshl_add_u32 v27, s55, 7, v24 // (op 21328) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 21328) 
s_mul_i32 s53, s54, s72 // (op 21328) 


s_sub_u32 s54, s52, s53 // (op 21328) 


v_lshl_add_u32 v24, s54, 7, v26 // (op 21328) 


v_mov_b32 v31, 0 // (op 21328) convert
v_mov_b32 v30, v24 // (op 21328) convert

v_mul_lo_u32 v33, v27, s31 // (op 21328) most significant: low of low * high
v_mul_hi_u32 v24, v27, s30 // (op 21328) most significant: high of low * low
v_mul_lo_u32 v32, v27, s30 // (op 21328) least significant: low of low * low
v_add_u32 v33, v33, v24 // (op 21328) most significant: sum

v_add_co_u32 v32, s[52:53], v32, v30 // (op 21328) least significant half
v_addc_co_u32 v33, s[52:53], v33, v31, s[52:53] // (op 21328) most significant half



v_lshlrev_b64 v[26:27], 2, v[32:33] // (op 21328) 

v_lshrrev_b64 v[30:31], 3, v[26:27] // (op 21328) 

s_mul_i32 s53, s30, s35 // (op 21330) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 21330) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 21330) least significant
s_mul_hi_u32 s55, s30, s34 // (op 21330) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21330) most significant: sum
s_add_u32 s53, s53, s55 // (op 21330) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21330) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21330) 

s_mov_b32 s68, s32 // (op 21330) reg expression
s_mov_b32 s69, s33 // (op 21330) reg expression
s_mov_b32 s70, s52 // (op 21330) reg expression
s_mov_b32 s71, 131072 // (op 21330) call()

v_lshrrev_b32 v24, 2, v1 // (op 21437) 
s_mul_i32 s52, s16, s8 // (op 21437) 
s_add_u32 s52, s52, s0 // (op 21437) 
v_and_b32 v26, -4, v1 // (op 21437) 
s_mul_hi_u32 s53, s52, s13 // (op 21437) 
v_sub_u32 v27, v1, v26 // (op 21437) 

s_sub_u32 s54, s52, s53 // (op 21437) 
v_lshlrev_b32 v26, 5, v27 // (op 21437) 

s_lshr_b32 s55, s54, 1 // (op 21437) 

s_add_u32 s54, s55, s53 // (op 21437) 


s_lshr_b32 s53, s54, s11 // (op 21437) 

s_and_b32 s54, s11, 2147483648 // (op 21437) 
s_cmp_eq_i32 s54, 0 // (op 21437) 
s_mov_b32 s55, scc // (op 21437) 

s_cmp_eq_i32 s55, 0 // (op 21437) 
s_cselect_b32 s54, s52, s53 // (op 21437) 


s_mul_hi_u32 s53, s54, s12 // (op 21437) 
s_sub_u32 s55, s54, s53 // (op 21437) 
s_lshr_b32 s72, s55, 1 // (op 21437) 

s_add_u32 s55, s72, s53 // (op 21437) 


s_lshr_b32 s53, s55, s10 // (op 21437) 

s_and_b32 s55, s10, 2147483648 // (op 21437) 
s_cmp_eq_i32 s55, 0 // (op 21437) 
s_mov_b32 s72, scc // (op 21437) 

s_cmp_eq_i32 s72, 0 // (op 21437) 
s_cselect_b32 s55, s54, s53 // (op 21437) 


s_mul_i32 s53, s55, s14 // (op 21437) 

s_sub_u32 s55, s54, s53 // (op 21437) 

v_lshl_add_u32 v27, s55, 7, v24 // (op 21437) 


v_mul_lo_u32 v33, v27, s37 // (op 21437) most significant: low of low * high
v_mul_hi_u32 v24, v27, s36 // (op 21437) most significant: high of low * low
v_mul_lo_u32 v32, v27, s36 // (op 21437) least significant: low of low * low
v_add_u32 v33, v33, v24 // (op 21437) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 21437) 
s_mul_i32 s53, s54, s72 // (op 21437) 


s_sub_u32 s54, s52, s53 // (op 21437) 


v_lshl_add_u32 v24, s54, 7, v26 // (op 21437) 


v_mov_b32 v27, 0 // (op 21437) convert
v_mov_b32 v26, v24 // (op 21437) convert

v_add_co_u32 v34, s[52:53], v26, v32 // (op 21437) least significant half
v_addc_co_u32 v35, s[52:53], v27, v33, s[52:53] // (op 21437) most significant half

v_lshlrev_b64 v[34:35], 2, v[34:35] // (op 21437) 


v_lshrrev_b64 v[26:27], 3, v[34:35] // (op 21437) 

s_mul_i32 s53, s36, s23 // (op 21439) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 21439) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 21439) least significant
s_mul_hi_u32 s55, s36, s22 // (op 21439) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21439) most significant: sum
s_add_u32 s53, s53, s55 // (op 21439) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21439) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21439) 

s_mov_b32 s72, s38 // (op 21439) reg expression
s_mov_b32 s73, s39 // (op 21439) reg expression
s_mov_b32 s74, s52 // (op 21439) reg expression
s_mov_b32 s75, 131072 // (op 21439) call()

v_lshrrev_b32 v24, 2, v1 // (op 21456) 
s_mul_i32 s52, s16, s8 // (op 21456) 
s_add_u32 s52, s52, s0 // (op 21456) 
v_and_b32 v32, -4, v1 // (op 21456) 
s_mul_hi_u32 s53, s52, s13 // (op 21456) 
v_sub_u32 v33, v1, v32 // (op 21456) 

s_sub_u32 s54, s52, s53 // (op 21456) 
v_lshlrev_b32 v32, 5, v33 // (op 21456) 

s_lshr_b32 s55, s54, 1 // (op 21456) 

s_add_u32 s54, s55, s53 // (op 21456) 


s_lshr_b32 s53, s54, s11 // (op 21456) 

s_and_b32 s54, s11, 2147483648 // (op 21456) 
s_cmp_eq_i32 s54, 0 // (op 21456) 
s_mov_b32 s55, scc // (op 21456) 

s_cmp_eq_i32 s55, 0 // (op 21456) 
s_cselect_b32 s54, s52, s53 // (op 21456) 


s_mul_hi_u32 s53, s54, s12 // (op 21456) 
s_sub_u32 s55, s54, s53 // (op 21456) 
s_lshr_b32 s76, s55, 1 // (op 21456) 

s_add_u32 s55, s76, s53 // (op 21456) 


s_lshr_b32 s53, s55, s10 // (op 21456) 

s_and_b32 s55, s10, 2147483648 // (op 21456) 
s_cmp_eq_i32 s55, 0 // (op 21456) 
s_mov_b32 s76, scc // (op 21456) 

s_cmp_eq_i32 s76, 0 // (op 21456) 
s_cselect_b32 s55, s54, s53 // (op 21456) 


s_mul_i32 s53, s55, s14 // (op 21456) 

s_sub_u32 s55, s54, s53 // (op 21456) 

v_lshl_add_u32 v33, s55, 7, v24 // (op 21456) 


v_mul_lo_u32 v35, v33, s37 // (op 21456) most significant: low of low * high
v_mul_hi_u32 v24, v33, s36 // (op 21456) most significant: high of low * low
v_mul_lo_u32 v34, v33, s36 // (op 21456) least significant: low of low * low
v_add_u32 v35, v35, v24 // (op 21456) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 21456) 
s_mul_i32 s53, s54, s76 // (op 21456) 


s_sub_u32 s54, s52, s53 // (op 21456) 


v_lshl_add_u32 v24, s54, 7, v32 // (op 21456) 


v_mov_b32 v33, 0 // (op 21456) convert
v_mov_b32 v32, v24 // (op 21456) convert

v_add_co_u32 v36, s[52:53], v32, v34 // (op 21456) least significant half
v_addc_co_u32 v37, s[52:53], v33, v35, s[52:53] // (op 21456) most significant half

v_lshlrev_b64 v[36:37], 2, v[36:37] // (op 21456) 


v_lshrrev_b64 v[32:33], 3, v[36:37] // (op 21456) 

s_mul_i32 s53, s36, s23 // (op 21458) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 21458) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 21458) least significant
s_mul_hi_u32 s55, s36, s22 // (op 21458) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21458) most significant: sum
s_add_u32 s53, s53, s55 // (op 21458) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21458) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21458) 

s_mov_b32 s72, s38 // (op 21458) reg expression
s_mov_b32 s73, s39 // (op 21458) reg expression
s_mov_b32 s74, s52 // (op 21458) reg expression
s_mov_b32 s75, 131072 // (op 21458) call()

v_and_b32 v24, 63, v1 // (op 22626) 
s_mul_i32 s52, s16, s8 // (op 22626) 
s_add_u32 s52, s52, s0 // (op 22626) 
v_lshrrev_b32 v34, 4, v24 // (op 22626) 
s_mul_hi_u32 s53, s52, s13 // (op 22626) 
v_and_b32 v35, 3, v34 // (op 22626) 

s_sub_u32 s54, s52, s53 // (op 22626) 
v_and_b32 v34, 15, v1 // (op 22626) 
s_lshr_b32 s55, s54, 1 // (op 22626) 

v_lshl_add_u32 v36, v35, 4, v34 // (op 22626) 


s_add_u32 s54, s55, s53 // (op 22626) 


v_lshrrev_b32 v34, 7, v1 // (op 22626) 
s_lshr_b32 s53, s54, s11 // (op 22626) 

v_lshl_add_u32 v35, v34, 6, v36 // (op 22626) 


s_and_b32 s54, s11, 2147483648 // (op 22626) 
v_lshrrev_b32 v34, 6, v24 // (op 22626) 

s_cmp_eq_i32 s54, 0 // (op 22626) 
s_mov_b32 s55, scc // (op 22626) 

v_lshlrev_b32 v24, 2, v34 // (op 22626) 

s_cmp_eq_i32 s55, 0 // (op 22626) 
s_cselect_b32 s54, s52, s53 // (op 22626) 


s_mul_hi_u32 s53, s54, s12 // (op 22626) 
s_sub_u32 s55, s54, s53 // (op 22626) 
s_lshr_b32 s76, s55, 1 // (op 22626) 

s_add_u32 s55, s76, s53 // (op 22626) 


s_lshr_b32 s53, s55, s10 // (op 22626) 

s_and_b32 s55, s10, 2147483648 // (op 22626) 
s_cmp_eq_i32 s55, 0 // (op 22626) 
s_mov_b32 s76, scc // (op 22626) 

s_cmp_eq_i32 s76, 0 // (op 22626) 
s_cselect_b32 s55, s54, s53 // (op 22626) 


v_lshl_add_u32 v34, s55, 7, v35 // (op 22626) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 22626) 
s_mul_i32 s53, s54, s76 // (op 22626) 


s_sub_u32 s54, s52, s53 // (op 22626) 


v_lshl_add_u32 v35, s54, 2, v24 // (op 22626) 


v_mul_lo_u32 v24, v34, s40 // (op 22626) 
v_add_u32 v24, v24, v35 // (op 22626) 


s_mul_i32 s53, s40, s35 // (op 22628) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 22628) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 22628) least significant
s_mul_hi_u32 s55, s40, s34 // (op 22628) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 22628) most significant: sum
s_add_u32 s53, s53, s55 // (op 22628) most significant: sum


s_mov_b32 s76, s42 // (op 22628) reg expression
s_mov_b32 s77, s43 // (op 22628) reg expression
s_mov_b32 s78, s52 // (op 22628) reg expression
s_mov_b32 s79, 131072 // (op 22628) call()

v_and_b32 v34, 63, v1 // (op 22653) 
s_mul_i32 s52, s16, s8 // (op 22653) 
s_add_u32 s52, s52, s0 // (op 22653) 
v_lshrrev_b32 v35, 4, v34 // (op 22653) 
s_mul_hi_u32 s53, s52, s13 // (op 22653) 
v_and_b32 v36, 3, v35 // (op 22653) 

s_sub_u32 s54, s52, s53 // (op 22653) 
v_and_b32 v35, 15, v1 // (op 22653) 
s_lshr_b32 s55, s54, 1 // (op 22653) 

v_lshl_add_u32 v37, v36, 4, v35 // (op 22653) 


s_add_u32 s54, s55, s53 // (op 22653) 


v_lshrrev_b32 v35, 6, v1 // (op 22653) 
s_lshr_b32 s53, s54, s11 // (op 22653) 

v_and_b32 v36, 1, v35 // (op 22653) 

s_and_b32 s54, s11, 2147483648 // (op 22653) 
v_lshl_add_u32 v35, v36, 6, v37 // (op 22653) 


s_cmp_eq_i32 s54, 0 // (op 22653) 
s_mov_b32 s55, scc // (op 22653) 

v_lshrrev_b32 v36, 6, v34 // (op 22653) 

s_cmp_eq_i32 s55, 0 // (op 22653) 
s_cselect_b32 s54, s52, s53 // (op 22653) 


v_lshlrev_b32 v34, 2, v36 // (op 22653) 

s_mul_hi_u32 s53, s54, s12 // (op 22653) 
s_sub_u32 s55, s54, s53 // (op 22653) 
s_lshr_b32 s80, s55, 1 // (op 22653) 

s_add_u32 s55, s80, s53 // (op 22653) 


s_lshr_b32 s53, s55, s10 // (op 22653) 

s_and_b32 s55, s10, 2147483648 // (op 22653) 
s_cmp_eq_i32 s55, 0 // (op 22653) 
s_mov_b32 s80, scc // (op 22653) 

s_cmp_eq_i32 s80, 0 // (op 22653) 
s_cselect_b32 s55, s54, s53 // (op 22653) 


s_mul_i32 s53, s55, s14 // (op 22653) 

s_sub_u32 s55, s54, s53 // (op 22653) 

v_lshl_add_u32 v36, s55, 7, v35 // (op 22653) 


v_mul_lo_u32 v35, v36, s44 // (op 22653) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 22653) 
s_mul_i32 s53, s54, s80 // (op 22653) 


s_sub_u32 s54, s52, s53 // (op 22653) 


v_lshl_add_u32 v36, s54, 2, v34 // (op 22653) 


v_add_u32 v34, v36, v35 // (op 22653) 


s_mul_i32 s53, s44, s23 // (op 22655) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 22655) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 22655) least significant
s_mul_hi_u32 s55, s44, s22 // (op 22655) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 22655) most significant: sum
s_add_u32 s53, s53, s55 // (op 22655) most significant: sum


s_mov_b32 s80, s46 // (op 22655) reg expression
s_mov_b32 s81, s47 // (op 22655) reg expression
s_mov_b32 s82, s52 // (op 22655) reg expression
s_mov_b32 s83, 131072 // (op 22655) call()

v_and_b32 v35, -4, v1 // (op 21633) 
v_sub_u32 v36, v1, v35 // (op 21633) 

v_lshlrev_b32 v35, 5, v36 // (op 21633) 

v_lshrrev_b32 v36, 2, v1 // (op 21633) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 21633) 


v_lshlrev_b32 v35, 2, v37 // (op 21633) 

v_lshrrev_b32 v36, 3, v35 // (op 21633) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 21633) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23147) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 21151) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21151) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21151) 


v_mov_b64 v[38:39], v[26:27] // (op 21151) 
s_mov_b32 m0, s52 // (op 21151) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 21151) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 21151) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 21151) most significant half

s_add_u32 m0, m0, 4096 // (op 21151) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 21151) Load value direct to lds



v_and_b32 v35, -4, v1 // (op 21558) 
v_sub_u32 v36, v1, v35 // (op 21558) 

v_lshlrev_b32 v35, 5, v36 // (op 21558) 

v_lshrrev_b32 v36, 2, v1 // (op 21558) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 21558) 


v_lshlrev_b32 v35, 2, v37 // (op 21558) 

v_lshrrev_b32 v36, 3, v35 // (op 21558) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 21558) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23127) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 21156) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21156) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21156) 


v_mov_b64 v[38:39], v[28:29] // (op 21156) 
s_add_u32 s53, 8192, s52 // (op 21156) 
s_mov_b32 m0, s53 // (op 21156) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 21156) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 21156) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 21156) most significant half

s_add_u32 m0, m0, 4096 // (op 21156) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 21156) Load value direct to lds




v_and_b32 v35, -4, v1 // (op 22257) 
v_sub_u32 v36, v1, v35 // (op 22257) 

v_lshlrev_b32 v35, 5, v36 // (op 22257) 

v_lshrrev_b32 v36, 2, v1 // (op 22257) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 22257) 


v_lshlrev_b32 v35, 2, v37 // (op 22257) 

v_lshrrev_b32 v36, 3, v35 // (op 22257) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 22257) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23275) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 21161) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21161) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21161) 


v_mov_b64 v[38:39], v[26:27] // (op 21161) 
s_add_u32 s53, 16384, s52 // (op 21161) 
s_mov_b32 m0, s53 // (op 21161) reg expression
buffer_load_dwordx4 v38, s[72:75], 64 offen offset: 0 lds // (op 21161) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 21161) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 21161) most significant half

s_add_u32 m0, m0, 4096 // (op 21161) 
buffer_load_dwordx4 v38, s[72:75], 64 offen offset: 0 lds // (op 21161) Load value direct to lds




v_and_b32 v35, -4, v1 // (op 22213) 
v_sub_u32 v36, v1, v35 // (op 22213) 

v_lshlrev_b32 v35, 5, v36 // (op 22213) 

v_lshrrev_b32 v36, 2, v1 // (op 22213) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 22213) 


v_lshlrev_b32 v35, 2, v37 // (op 22213) 

v_lshrrev_b32 v36, 3, v35 // (op 22213) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 22213) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23265) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 21163) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21163) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21163) 


v_mov_b64 v[38:39], v[28:29] // (op 21163) 
s_add_u32 s53, 24576, s52 // (op 21163) 
s_mov_b32 m0, s53 // (op 21163) reg expression
buffer_load_dwordx4 v38, s[68:71], 64 offen offset: 0 lds // (op 21163) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 21163) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 21163) most significant half

s_add_u32 m0, m0, 4096 // (op 21163) 
buffer_load_dwordx4 v38, s[68:71], 64 offen offset: 0 lds // (op 21163) Load value direct to lds




v_and_b32 v35, -4, v1 // (op 22345) 
v_sub_u32 v36, v1, v35 // (op 22345) 

v_lshlrev_b32 v35, 5, v36 // (op 22345) 

v_lshrrev_b32 v36, 2, v1 // (op 22345) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 22345) 


v_lshlrev_b32 v35, 2, v37 // (op 22345) 

v_lshrrev_b32 v36, 3, v35 // (op 22345) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 22345) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23294) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 21165) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21165) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21165) 


v_mov_b64 v[38:39], v[26:27] // (op 21165) 
s_add_u32 s53, 32768, s52 // (op 21165) 
s_mov_b32 m0, s53 // (op 21165) reg expression
s_mov_b32 s54, 128 // (op 21165) call()
buffer_load_dwordx4 v38, s[72:75], s54 offen offset: 0 lds // (op 21165) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 21165) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 21165) most significant half

s_add_u32 m0, m0, 4096 // (op 21165) 
buffer_load_dwordx4 v38, s[72:75], s54 offen offset: 0 lds // (op 21165) Load value direct to lds





v_and_b32 v35, -4, v1 // (op 22301) 
v_sub_u32 v36, v1, v35 // (op 22301) 

v_lshlrev_b32 v35, 5, v36 // (op 22301) 

v_lshrrev_b32 v36, 2, v1 // (op 22301) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 22301) 


v_lshlrev_b32 v35, 2, v37 // (op 22301) 

v_lshrrev_b32 v36, 3, v35 // (op 22301) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 22301) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23285) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 21167) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21167) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21167) 


v_mov_b64 v[38:39], v[28:29] // (op 21167) 
s_add_u32 s53, 40960, s52 // (op 21167) 
s_mov_b32 m0, s53 // (op 21167) reg expression
s_mov_b32 s54, 128 // (op 21167) call()
buffer_load_dwordx4 v38, s[68:71], s54 offen offset: 0 lds // (op 21167) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 21167) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 21167) most significant half

s_add_u32 m0, m0, 4096 // (op 21167) 
buffer_load_dwordx4 v38, s[68:71], s54 offen offset: 0 lds // (op 21167) Load value direct to lds





v_and_b32 v35, -4, v1 // (op 22433) 
v_sub_u32 v36, v1, v35 // (op 22433) 

v_lshlrev_b32 v35, 5, v36 // (op 22433) 

v_lshrrev_b32 v36, 2, v1 // (op 22433) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 22433) 


v_lshlrev_b32 v35, 2, v37 // (op 22433) 

v_lshrrev_b32 v36, 3, v35 // (op 22433) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 22433) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23313) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 21169) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21169) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21169) 


v_mov_b64 v[38:39], v[26:27] // (op 21169) 
s_add_u32 s53, 49152, s52 // (op 21169) 
s_mov_b32 m0, s53 // (op 21169) reg expression
s_mov_b32 s54, 192 // (op 21169) call()
buffer_load_dwordx4 v38, s[72:75], s54 offen offset: 0 lds // (op 21169) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 21169) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 21169) most significant half

s_add_u32 m0, m0, 4096 // (op 21169) 
buffer_load_dwordx4 v38, s[72:75], s54 offen offset: 0 lds // (op 21169) Load value direct to lds






v_and_b32 v26, -4, v1 // (op 22389) 
v_sub_u32 v27, v1, v26 // (op 22389) 

v_lshlrev_b32 v26, 5, v27 // (op 22389) 

v_lshrrev_b32 v27, 2, v1 // (op 22389) 
v_lshl_add_u32 v35, v27, 7, v26 // (op 22389) 


v_lshlrev_b32 v26, 2, v35 // (op 22389) 

v_lshrrev_b32 v27, 3, v26 // (op 22389) 

s_nop 0
v_readfirstlane_b32 s52, v27 // (op 22389) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23304) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 21171) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21171) 

v_lshrrev_b64 v[26:27], 3, s[84:85] // (op 21171) 


v_mov_b64 v[36:37], v[28:29] // (op 21171) 
s_add_u32 s53, 57344, s52 // (op 21171) 
s_mov_b32 m0, s53 // (op 21171) reg expression
s_mov_b32 s54, 192 // (op 21171) call()
buffer_load_dwordx4 v36, s[68:71], s54 offen offset: 0 lds // (op 21171) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v26 // (op 21171) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 21171) most significant half

s_add_u32 m0, m0, 4096 // (op 21171) 
buffer_load_dwordx4 v36, s[68:71], s54 offen offset: 0 lds // (op 21171) Load value direct to lds






s_waitcnt vmcnt(14)
s_barrier  // (op 23144) 



ds_read_b128 v[26:29], v23 // (op 20653) Load local data 



ds_read_b128 v[36:39], v23 offset:1024 // (op 20658) Load local data 



ds_read_b128 v[40:43], v23 offset:2048 // (op 20663) Load local data 



ds_read_b128 v[44:47], v23 offset:3072 // (op 20668) Load local data 
s_waitcnt vmcnt(12)
s_barrier  // (op 23124) 



ds_read_b128 v[48:51], v21 offset:8192 // (op 20633) Load local data 



ds_read_b128 v[52:55], v21 offset:9216 // (op 20638) Load local data 



ds_read_b128 v[56:59], v21 offset:10240 // (op 20643) Load local data 



ds_read_b128 v[60:63], v21 offset:11264 // (op 20648) Load local data 
v_and_b32 v35, -4, v1 // (op 21530) 
v_sub_u32 v64, v1, v35 // (op 21530) 

v_lshlrev_b32 v35, 5, v64 // (op 21530) 

v_lshrrev_b32 v64, 2, v1 // (op 21530) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 21530) 


v_lshlrev_b32 v35, 2, v65 // (op 21530) 

v_lshrrev_b32 v64, 3, v35 // (op 21530) 

s_nop 0
v_readfirstlane_b32 s52, v64 // (op 21530) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 21605) 
v_sub_u32 v64, v1, v35 // (op 21605) 

v_lshlrev_b32 v35, 5, v64 // (op 21605) 

v_lshrrev_b32 v64, 2, v1 // (op 21605) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 21605) 


v_lshlrev_b32 v35, 2, v65 // (op 21605) 

v_lshrrev_b32 v64, 3, v35 // (op 21605) 

s_nop 0
v_readfirstlane_b32 s53, v64 // (op 21605) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22198) 
v_sub_u32 v64, v1, v35 // (op 22198) 

v_lshlrev_b32 v35, 5, v64 // (op 22198) 

v_lshrrev_b32 v64, 2, v1 // (op 22198) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22198) 


v_lshlrev_b32 v35, 2, v65 // (op 22198) 

v_lshrrev_b32 v64, 3, v35 // (op 22198) 

s_nop 0
v_readfirstlane_b32 s54, v64 // (op 22198) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22242) 
v_sub_u32 v64, v1, v35 // (op 22242) 

v_lshlrev_b32 v35, 5, v64 // (op 22242) 

v_lshrrev_b32 v64, 2, v1 // (op 22242) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22242) 


v_lshlrev_b32 v35, 2, v65 // (op 22242) 

v_lshrrev_b32 v64, 3, v35 // (op 22242) 

s_nop 0
v_readfirstlane_b32 s55, v64 // (op 22242) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22286) 
v_sub_u32 v64, v1, v35 // (op 22286) 

v_lshlrev_b32 v35, 5, v64 // (op 22286) 

v_lshrrev_b32 v64, 2, v1 // (op 22286) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22286) 


v_lshlrev_b32 v35, 2, v65 // (op 22286) 

v_lshrrev_b32 v64, 3, v35 // (op 22286) 

s_nop 0
v_readfirstlane_b32 s84, v64 // (op 22286) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22330) 
v_sub_u32 v64, v1, v35 // (op 22330) 

v_lshlrev_b32 v35, 5, v64 // (op 22330) 

v_lshrrev_b32 v64, 2, v1 // (op 22330) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22330) 


v_lshlrev_b32 v35, 2, v65 // (op 22330) 

v_lshrrev_b32 v64, 3, v35 // (op 22330) 

s_nop 0
v_readfirstlane_b32 s85, v64 // (op 22330) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22374) 
v_sub_u32 v64, v1, v35 // (op 22374) 

v_lshlrev_b32 v35, 5, v64 // (op 22374) 

v_lshrrev_b32 v64, 2, v1 // (op 22374) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22374) 


v_lshlrev_b32 v35, 2, v65 // (op 22374) 

v_lshrrev_b32 v64, 3, v35 // (op 22374) 

s_nop 0
v_readfirstlane_b32 s86, v64 // (op 22374) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22418) 
v_sub_u32 v64, v1, v35 // (op 22418) 

v_lshlrev_b32 v35, 5, v64 // (op 22418) 

v_lshrrev_b32 v64, 2, v1 // (op 22418) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22418) 


v_lshlrev_b32 v35, 2, v65 // (op 22418) 

v_lshrrev_b32 v64, 3, v35 // (op 22418) 

s_nop 0
v_readfirstlane_b32 s87, v64 // (op 22418) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 7888) call()
s_and_b32 s88, s7, 4294967292 // (op 7887) 
s_cmp_lt_u32 s1, s88 // (op 7887) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_7887 // (op 7887) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_7887 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_7887:


s_add_u32 s1, s1, 4 // (op 7889) 
v_mov_b32 v35, v24 // (op 72) 
v_mov_b32 v64, v35 // (op 72) reg expression

buffer_load_dwordx4 v[66:69], v64, s[76:79], 0 offen // (op 72) Load value


v_add_u32 v24, 16, v24 // (op 22616) 


v_mov_b32 v35, v34 // (op 76) 
v_mov_b32 v64, v35 // (op 76) reg expression

buffer_load_dwordx4 v[70:73], v64, s[80:83], 0 offen // (op 76) Load value


v_add_u32 v34, 16, v34 // (op 22643) 
s_waitcnt vmcnt(1)
v_bfe_u32 v74, v66, 0, 8 // (op 19383) 
v_bfe_u32 v75, v66, 8, 8 // (op 19383) 
v_bfe_u32 v76, v66, 16, 8 // (op 19383) 
v_bfe_u32 v77, v66, 24, 8 // (op 19383) 
v_permlane16_swap_b32 v74, v75 // (op 19383) 
s_nop 0
v_permlane16_swap_b32 v76, v77 // (op 19383) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v74, v76 // (op 19383) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v75, v77 // (op 19383) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 23150) 

s_lshl_b64 s[88:89], s[36:37], 6 // (op 21030) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 21030) 

v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 21030) 


v_mov_b64 v[78:79], v[32:33] // (op 21030) 
v_mov_b64 v[80:81], v[78:79] // (op 21030) 
s_mov_b32 m0, s53 // (op 21030) reg expression
s_mov_b32 s88, 256 // (op 21030) call()
buffer_load_dwordx4 v80, s[72:75], s88 offen offset: 0 lds // (op 21030) Load value direct to lds
v_add_co_u32 v80, s[90:91], v80, v64 // (op 21030) least significant half
v_addc_co_u32 v81, s[90:91], 0, v81, s[90:91] // (op 21030) most significant half

s_add_u32 m0, m0, 4096 // (op 21030) 
buffer_load_dwordx4 v80, s[72:75], s88 offen offset: 0 lds // (op 21030) Load value direct to lds



s_waitcnt vmcnt(2)
v_bfe_u32 v80, v70, 0, 8 // (op 19512) 
v_bfe_u32 v81, v70, 8, 8 // (op 19512) 
v_bfe_u32 v82, v70, 16, 8 // (op 19512) 
v_bfe_u32 v83, v70, 24, 8 // (op 19512) 
v_permlane16_swap_b32 v80, v81 // (op 19512) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 19512) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 19512) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 19512) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23130) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[26:29], a[0:3], v74, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 78) 
s_lshl_b64 s[88:89], s[30:31], 6 // (op 21025) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:51], v[36:39], a[4:7], v74, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6023) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 21025) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:51], v[40:43], a[8:11], v74, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6249) 
v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 21025) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[44:47], a[12:15], v74, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6475) 
v_mov_b64 v[84:85], v[30:31] // (op 21025) 
v_mov_b64 v[86:87], v[84:85] // (op 21025) 
s_barrier  // (op 23271) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:55], v[26:29], a[16:19], v75, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7147) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:55], v[36:39], a[20:23], v75, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7071) 
s_add_u32 s88, 8192, s52 // (op 21025) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:55], v[40:43], a[24:27], v75, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6995) 
s_mov_b32 m0, s88 // (op 21025) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:55], v[44:47], a[28:31], v75, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6919) 
s_mov_b32 s89, 256 // (op 21025) call()
buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 21025) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:59], v[26:29], a[32:35], v76, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8149) 
v_add_co_u32 v86, s[90:91], v86, v64 // (op 21025) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[56:59], v[36:39], a[36:39], v76, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8073) 
v_addc_co_u32 v87, s[90:91], 0, v87, s[90:91] // (op 21025) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[56:59], v[40:43], a[40:43], v76, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7997) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:59], v[44:47], a[44:47], v76, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7921) 
s_add_u32 m0, m0, 4096 // (op 21025) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:63], v[26:29], a[48:51], v77, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9151) 
ds_read_b128 v[88:91], v14 offset:16384 // (op 5818) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:63], v[36:39], a[52:55], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9075) 
ds_read_b128 v[92:95], v14 offset:17408 // (op 6034) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:63], v[40:43], a[56:59], v77, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8999) 
ds_read_b128 v[96:99], v14 offset:18432 // (op 6260) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[60:63], v[44:47], a[60:63], v77, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8923) 
ds_read_b128 v[100:103], v14 offset:19456 // (op 6486) Load local data 


buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 21025) Load value direct to lds




s_barrier  // (op 23261) 



ds_read_b128 v[48:51], v13 offset:24576 // (op 5816) Load local data 



ds_read_b128 v[52:55], v13 offset:25600 // (op 6929) Load local data 



ds_read_b128 v[56:59], v13 offset:26624 // (op 7931) Load local data 



ds_read_b128 v[74:77], v13 offset:27648 // (op 8933) Load local data 
v_bfe_u32 v80, v67, 0, 8 // (op 19398) 
v_bfe_u32 v81, v67, 8, 8 // (op 19398) 
v_bfe_u32 v82, v67, 16, 8 // (op 19398) 
v_bfe_u32 v83, v67, 24, 8 // (op 19398) 
v_permlane16_swap_b32 v80, v81 // (op 19398) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 19398) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 19398) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 19398) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 23278) 

s_lshl_b64 s[88:89], s[36:37], 6 // (op 21104) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 21104) 

v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 21104) 


v_mov_b64 v[86:87], v[78:79] // (op 21104) 
s_add_u32 s88, 16384, s55 // (op 21104) 
s_mov_b32 m0, s88 // (op 21104) reg expression
s_mov_b32 s89, 320 // (op 21104) call()
buffer_load_dwordx4 v86, s[72:75], s89 offen offset: 0 lds // (op 21104) Load value direct to lds
v_add_co_u32 v86, s[90:91], v86, v64 // (op 21104) least significant half
v_addc_co_u32 v87, s[90:91], 0, v87, s[90:91] // (op 21104) most significant half

s_add_u32 m0, m0, 4096 // (op 21104) 
buffer_load_dwordx4 v86, s[72:75], s89 offen offset: 0 lds // (op 21104) Load value direct to lds




v_bfe_u32 v104, v71, 0, 8 // (op 19527) 
v_bfe_u32 v105, v71, 8, 8 // (op 19527) 
v_bfe_u32 v106, v71, 16, 8 // (op 19527) 
v_bfe_u32 v107, v71, 24, 8 // (op 19527) 
v_permlane16_swap_b32 v104, v105 // (op 19527) 
s_nop 0
v_permlane16_swap_b32 v106, v107 // (op 19527) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v104, v106 // (op 19527) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v105, v107 // (op 19527) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23268) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[88:91], a[0:3], v80, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5823) 
s_lshl_b64 s[88:89], s[30:31], 6 // (op 21099) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:51], v[92:95], a[4:7], v80, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6024) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 21099) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:51], v[96:99], a[8:11], v80, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6250) 
v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 21099) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[100:103], a[12:15], v80, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6476) 
v_mov_b64 v[86:87], v[84:85] // (op 21099) 
s_add_u32 s88, 24576, s54 // (op 21099) 
s_barrier  // (op 23291) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:55], v[88:91], a[16:19], v81, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7148) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:55], v[92:95], a[20:23], v81, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7072) 
s_mov_b32 m0, s88 // (op 21099) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:55], v[96:99], a[24:27], v81, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6996) 
ds_read_b128 v[26:29], v16 offset:32768 // (op 5846) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:55], v[100:103], a[28:31], v81, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6920) 
ds_read_b128 v[36:39], v16 offset:33792 // (op 6044) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:59], v[88:91], a[32:35], v82, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8150) 
ds_read_b128 v[40:43], v16 offset:34816 // (op 6270) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[56:59], v[92:95], a[36:39], v82, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8074) 
s_mov_b32 s89, 320 // (op 21099) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[56:59], v[96:99], a[40:43], v82, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7998) 
ds_read_b128 v[44:47], v16 offset:35840 // (op 6496) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:59], v[100:103], a[44:47], v82, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7922) 
s_barrier  // (op 23281) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:77], v[88:91], a[48:51], v83, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9152) 
ds_read_b128 v[48:51], v15 offset:40960 // (op 5844) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:77], v[92:95], a[52:55], v83, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9076) 
ds_read_b128 v[52:55], v15 offset:41984 // (op 6939) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:77], v[96:99], a[56:59], v83, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9000) 
ds_read_b128 v[88:91], v15 offset:43008 // (op 7941) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[100:103], a[60:63], v83, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8924) 
ds_read_b128 v[92:95], v15 offset:44032 // (op 8943) Load local data 




buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 21099) Load value direct to lds
v_add_co_u32 v86, s[90:91], v86, v64 // (op 21099) least significant half
v_addc_co_u32 v87, s[90:91], 0, v87, s[90:91] // (op 21099) most significant half

s_add_u32 m0, m0, 4096 // (op 21099) 
buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 21099) Load value direct to lds




v_bfe_u32 v74, v68, 0, 8 // (op 19415) 
v_bfe_u32 v75, v68, 8, 8 // (op 19415) 
v_bfe_u32 v76, v68, 16, 8 // (op 19415) 
v_bfe_u32 v77, v68, 24, 8 // (op 19415) 
v_permlane16_swap_b32 v74, v75 // (op 19415) 
s_nop 0
v_permlane16_swap_b32 v76, v77 // (op 19415) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v74, v76 // (op 19415) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v75, v77 // (op 19415) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 23297) 

s_lshl_b64 s[88:89], s[36:37], 6 // (op 21114) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 21114) 

v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 21114) 


v_mov_b64 v[86:87], v[78:79] // (op 21114) 
s_add_u32 s88, 32768, s85 // (op 21114) 
s_mov_b32 m0, s88 // (op 21114) reg expression
s_mov_b32 s89, 384 // (op 21114) call()
buffer_load_dwordx4 v86, s[72:75], s89 offen offset: 0 lds // (op 21114) Load value direct to lds
v_add_co_u32 v86, s[90:91], v86, v64 // (op 21114) least significant half
v_addc_co_u32 v87, s[90:91], 0, v87, s[90:91] // (op 21114) most significant half

s_add_u32 m0, m0, 4096 // (op 21114) 
v_bfe_u32 v80, v72, 0, 8 // (op 19545) 
v_bfe_u32 v81, v72, 8, 8 // (op 19545) 
v_bfe_u32 v82, v72, 16, 8 // (op 19545) 
v_bfe_u32 v83, v72, 24, 8 // (op 19545) 
v_permlane16_swap_b32 v80, v81 // (op 19545) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 19545) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 19545) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 19545) 
buffer_load_dwordx4 v86, s[72:75], s89 offen offset: 0 lds // (op 21114) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[26:29], a[0:3], v74, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5851) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23288) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:51], v[36:39], a[4:7], v74, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6025) 
s_lshl_b64 s[88:89], s[30:31], 6 // (op 21109) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:51], v[40:43], a[8:11], v74, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6251) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 21109) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[44:47], a[12:15], v74, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6477) 
v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 21109) 


v_mov_b64 v[86:87], v[84:85] // (op 21109) 
s_barrier  // (op 23310) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:55], v[26:29], a[16:19], v75, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7149) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:55], v[36:39], a[20:23], v75, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7073) 
ds_read_b128 v[56:59], v18 offset:49152 // (op 5874) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:55], v[40:43], a[24:27], v75, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6997) 
ds_read_b128 v[60:63], v18 offset:50176 // (op 6054) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:55], v[44:47], a[28:31], v75, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6921) 
s_add_u32 s88, 40960, s84 // (op 21109) 
s_mov_b32 m0, s88 // (op 21109) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[26:29], a[32:35], v76, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8151) 
s_mov_b32 s89, 384 // (op 21109) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[36:39], a[36:39], v76, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8075) 
ds_read_b128 v[96:99], v18 offset:51200 // (op 6280) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[40:43], a[40:43], v76, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7999) 
buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 21109) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[44:47], a[44:47], v76, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7923) 
v_add_co_u32 v86, s[90:91], v86, v64 // (op 21109) least significant half

v_addc_co_u32 v87, s[90:91], 0, v87, s[90:91] // (op 21109) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[26:29], a[48:51], v77, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9153) 
s_add_u32 m0, m0, 4096 // (op 21109) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[36:39], a[52:55], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9077) 
ds_read_b128 v[88:91], v18 offset:52224 // (op 6506) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[40:43], a[56:59], v77, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9001) 
s_barrier  // (op 23300) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[44:47], a[60:63], v77, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8925) 
ds_read_b128 v[48:51], v17 offset:57344 // (op 5872) Load local data 






buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 21109) Load value direct to lds




ds_read_b128 v[52:55], v17 offset:58368 // (op 6949) Load local data 



ds_read_b128 v[74:77], v17 offset:59392 // (op 7951) Load local data 



ds_read_b128 v[80:83], v17 offset:60416 // (op 8953) Load local data 
v_bfe_u32 v92, v69, 0, 8 // (op 19432) 
v_bfe_u32 v93, v69, 8, 8 // (op 19432) 
v_bfe_u32 v94, v69, 16, 8 // (op 19432) 
v_bfe_u32 v95, v69, 24, 8 // (op 19432) 
v_permlane16_swap_b32 v92, v93 // (op 19432) 
s_nop 0
v_permlane16_swap_b32 v94, v95 // (op 19432) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v92, v94 // (op 19432) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v93, v95 // (op 19432) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 23316) 


s_lshl_b64 s[88:89], s[36:37], 6 // (op 21124) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 21124) 

v_lshrrev_b64 v[86:87], 3, s[90:91] // (op 21124) 


v_mov_b64 v[64:65], v[78:79] // (op 21124) 
s_add_u32 s88, 49152, s87 // (op 21124) 
s_mov_b32 m0, s88 // (op 21124) reg expression
s_mov_b32 s89, 448 // (op 21124) call()
v_bfe_u32 v66, v73, 0, 8 // (op 19563) 
v_bfe_u32 v67, v73, 8, 8 // (op 19563) 
v_bfe_u32 v68, v73, 16, 8 // (op 19563) 
v_bfe_u32 v69, v73, 24, 8 // (op 19563) 
v_permlane16_swap_b32 v66, v67 // (op 19563) 
s_nop 0
v_permlane16_swap_b32 v68, v69 // (op 19563) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v66, v68 // (op 19563) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 19563) 
buffer_load_dwordx4 v64, s[72:75], s89 offen offset: 0 lds // (op 21124) Load value direct to lds

v_add_co_u32 v64, s[90:91], v64, v86 // (op 21124) least significant half
v_addc_co_u32 v65, s[90:91], 0, v65, s[90:91] // (op 21124) most significant half

s_add_u32 m0, m0, 4096 // (op 21124) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[56:59], a[0:3], v92, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5879) 
buffer_load_dwordx4 v64, s[72:75], s89 offen offset: 0 lds // (op 21124) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:51], v[60:63], a[4:7], v92, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6026) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:51], v[96:99], a[8:11], v92, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6252) 
v_add_co_u32 v32, vcc, 256, v32 // (op 21448) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[88:91], a[12:15], v92, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6478) 
v_addc_co_u32 v33, vcc, 0, v33, vcc // (op 21448) most significant half
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23307) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:55], v[56:59], a[16:19], v93, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7150) 
s_lshl_b64 s[88:89], s[30:31], 6 // (op 21119) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:55], v[60:63], a[20:23], v93, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7074) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 21119) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:55], v[96:99], a[24:27], v93, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6998) 
v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 21119) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:55], v[88:91], a[28:31], v93, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6922) 
v_mov_b64 v[78:79], v[84:85] // (op 21119) 
s_add_u32 s88, 57344, s86 // (op 21119) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[56:59], a[32:35], v94, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8152) 
s_mov_b32 m0, s88 // (op 21119) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[60:63], a[36:39], v94, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8076) 
s_mov_b32 s89, 448 // (op 21119) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:77], v[96:99], a[40:43], v94, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8000) 
buffer_load_dwordx4 v78, s[68:71], s89 offen offset: 0 lds // (op 21119) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:77], v[88:91], a[44:47], v94, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7924) 
v_add_co_u32 v78, s[90:91], v78, v64 // (op 21119) least significant half

v_addc_co_u32 v79, s[90:91], 0, v79, s[90:91] // (op 21119) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[56:59], a[48:51], v95, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9154) 
s_add_u32 m0, m0, 4096 // (op 21119) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[60:63], a[52:55], v95, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9078) 
s_waitcnt vmcnt(13)
s_barrier  // (op 23156) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[96:99], a[56:59], v95, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9002) 
ds_read_b128 v[26:29], v4 // (op 67) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[88:91], a[60:63], v95, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8926) 
ds_read_b128 v[36:39], v4 offset:1024 // (op 6020) Load local data 







ds_read_b128 v[40:43], v4 offset:2048 // (op 6246) Load local data 
buffer_load_dwordx4 v78, s[68:71], s89 offen offset: 0 lds // (op 21119) Load value direct to lds








ds_read_b128 v[44:47], v4 offset:3072 // (op 6472) Load local data 
v_add_co_u32 v30, vcc, 256, v30 // (op 21320) least significant half
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 21320) most significant half
s_waitcnt vmcnt(12)
s_barrier  // (op 23136) 



ds_read_b128 v[48:51], v3 offset:8192 // (op 62) Load local data 



ds_read_b128 v[52:55], v3 offset:9216 // (op 6915) Load local data 



ds_read_b128 v[56:59], v3 offset:10240 // (op 7917) Load local data 



ds_read_b128 v[60:63], v3 offset:11264 // (op 8919) Load local data 
s_and_b32 s88, s7, 4294967292 // (op 7887) 
s_cmp_lt_u32 s1, s88 // (op 7887) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_7887 // (op 7887) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_7887 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_7887:














s_barrier  // (op 23703) 
s_and_b32 s52, s7, 4294967292 // (op 21208) 
v_lshrrev_b32 v24, 2, v1 // (op 21208) 
s_add_u32 s53, s0, s52 // (op 21208) 

v_and_b32 v30, -4, v1 // (op 21208) 
v_sub_u32 v31, v1, v30 // (op 21208) 

s_mul_i32 s52, s16, s8 // (op 21208) 
s_add_u32 s52, s52, s53 // (op 21208) 

v_lshlrev_b32 v30, 5, v31 // (op 21208) 

s_mul_hi_u32 s53, s52, s13 // (op 21208) 
s_sub_u32 s54, s52, s53 // (op 21208) 
s_lshr_b32 s55, s54, 1 // (op 21208) 

s_add_u32 s54, s55, s53 // (op 21208) 


s_lshr_b32 s53, s54, s11 // (op 21208) 

s_and_b32 s54, s11, 2147483648 // (op 21208) 
s_cmp_eq_i32 s54, 0 // (op 21208) 
s_mov_b32 s55, scc // (op 21208) 

s_cmp_eq_i32 s55, 0 // (op 21208) 
s_cselect_b32 s54, s52, s53 // (op 21208) 


s_mul_hi_u32 s53, s54, s12 // (op 21208) 
s_sub_u32 s55, s54, s53 // (op 21208) 
s_lshr_b32 s76, s55, 1 // (op 21208) 

s_add_u32 s55, s76, s53 // (op 21208) 


s_lshr_b32 s53, s55, s10 // (op 21208) 

s_and_b32 s55, s10, 2147483648 // (op 21208) 
s_cmp_eq_i32 s55, 0 // (op 21208) 
s_mov_b32 s76, scc // (op 21208) 

s_cmp_eq_i32 s76, 0 // (op 21208) 
s_cselect_b32 s55, s54, s53 // (op 21208) 


v_lshl_add_u32 v31, s55, 7, v24 // (op 21208) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 21208) 
s_mul_i32 s53, s54, s76 // (op 21208) 


s_sub_u32 s54, s52, s53 // (op 21208) 


v_lshl_add_u32 v24, s54, 7, v30 // (op 21208) 


v_mov_b32 v33, 0 // (op 21208) convert
v_mov_b32 v32, v24 // (op 21208) convert

v_mul_lo_u32 v35, v31, s31 // (op 21208) most significant: low of low * high
v_mul_hi_u32 v24, v31, s30 // (op 21208) most significant: high of low * low
v_mul_lo_u32 v34, v31, s30 // (op 21208) least significant: low of low * low
v_add_u32 v35, v35, v24 // (op 21208) most significant: sum

v_add_co_u32 v34, s[52:53], v34, v32 // (op 21208) least significant half
v_addc_co_u32 v35, s[52:53], v35, v33, s[52:53] // (op 21208) most significant half



v_lshlrev_b64 v[30:31], 2, v[34:35] // (op 21208) 

v_lshrrev_b64 v[32:33], 3, v[30:31] // (op 21208) 

s_mul_i32 s53, s30, s35 // (op 21210) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 21210) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 21210) least significant
s_mul_hi_u32 s55, s30, s34 // (op 21210) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21210) most significant: sum
s_add_u32 s53, s53, s55 // (op 21210) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21210) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21210) 

s_mov_b32 s68, s32 // (op 21210) reg expression
s_mov_b32 s69, s33 // (op 21210) reg expression
s_mov_b32 s70, s52 // (op 21210) reg expression
s_mov_b32 s71, 131072 // (op 21210) call()

s_and_b32 s52, s7, 4294967292 // (op 21372) 
v_lshrrev_b32 v24, 2, v1 // (op 21372) 
s_add_u32 s53, s0, s52 // (op 21372) 

v_and_b32 v30, -4, v1 // (op 21372) 
v_sub_u32 v31, v1, v30 // (op 21372) 

s_mul_i32 s52, s16, s8 // (op 21372) 
s_add_u32 s52, s52, s53 // (op 21372) 

v_lshlrev_b32 v30, 5, v31 // (op 21372) 

s_mul_hi_u32 s53, s52, s13 // (op 21372) 
s_sub_u32 s54, s52, s53 // (op 21372) 
s_lshr_b32 s55, s54, 1 // (op 21372) 

s_add_u32 s54, s55, s53 // (op 21372) 


s_lshr_b32 s53, s54, s11 // (op 21372) 

s_and_b32 s54, s11, 2147483648 // (op 21372) 
s_cmp_eq_i32 s54, 0 // (op 21372) 
s_mov_b32 s55, scc // (op 21372) 

s_cmp_eq_i32 s55, 0 // (op 21372) 
s_cselect_b32 s54, s52, s53 // (op 21372) 


s_mul_hi_u32 s53, s54, s12 // (op 21372) 
s_sub_u32 s55, s54, s53 // (op 21372) 
s_lshr_b32 s76, s55, 1 // (op 21372) 

s_add_u32 s55, s76, s53 // (op 21372) 


s_lshr_b32 s53, s55, s10 // (op 21372) 

s_and_b32 s55, s10, 2147483648 // (op 21372) 
s_cmp_eq_i32 s55, 0 // (op 21372) 
s_mov_b32 s76, scc // (op 21372) 

s_cmp_eq_i32 s76, 0 // (op 21372) 
s_cselect_b32 s55, s54, s53 // (op 21372) 


s_mul_i32 s53, s55, s14 // (op 21372) 

s_sub_u32 s55, s54, s53 // (op 21372) 

v_lshl_add_u32 v31, s55, 7, v24 // (op 21372) 


v_mul_lo_u32 v35, v31, s37 // (op 21372) most significant: low of low * high
v_mul_hi_u32 v24, v31, s36 // (op 21372) most significant: high of low * low
v_mul_lo_u32 v34, v31, s36 // (op 21372) least significant: low of low * low
v_add_u32 v35, v35, v24 // (op 21372) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 21372) 
s_mul_i32 s53, s54, s76 // (op 21372) 


s_sub_u32 s54, s52, s53 // (op 21372) 


v_lshl_add_u32 v24, s54, 7, v30 // (op 21372) 


v_mov_b32 v31, 0 // (op 21372) convert
v_mov_b32 v30, v24 // (op 21372) convert

v_add_co_u32 v64, s[52:53], v30, v34 // (op 21372) least significant half
v_addc_co_u32 v65, s[52:53], v31, v35, s[52:53] // (op 21372) most significant half

v_lshlrev_b64 v[64:65], 2, v[64:65] // (op 21372) 


v_lshrrev_b64 v[30:31], 3, v[64:65] // (op 21372) 

s_mul_i32 s53, s36, s23 // (op 21374) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 21374) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 21374) least significant
s_mul_hi_u32 s55, s36, s22 // (op 21374) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21374) most significant: sum
s_add_u32 s53, s53, s55 // (op 21374) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21374) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21374) 

s_mov_b32 s72, s38 // (op 21374) reg expression
s_mov_b32 s73, s39 // (op 21374) reg expression
s_mov_b32 s74, s52 // (op 21374) reg expression
s_mov_b32 s75, 131072 // (op 21374) call()

v_and_b32 v24, -4, v1 // (op 21543) 
v_sub_u32 v34, v1, v24 // (op 21543) 

v_lshlrev_b32 v24, 5, v34 // (op 21543) 

v_lshrrev_b32 v34, 2, v1 // (op 21543) 
v_lshl_add_u32 v35, v34, 7, v24 // (op 21543) 


v_lshlrev_b32 v24, 2, v35 // (op 21543) 

v_lshrrev_b32 v34, 3, v24 // (op 21543) 

s_nop 0
v_readfirstlane_b32 s52, v34 // (op 21543) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 21618) 
v_sub_u32 v34, v1, v24 // (op 21618) 

v_lshlrev_b32 v24, 5, v34 // (op 21618) 

v_lshrrev_b32 v34, 2, v1 // (op 21618) 
v_lshl_add_u32 v35, v34, 7, v24 // (op 21618) 


v_lshlrev_b32 v24, 2, v35 // (op 21618) 

v_lshrrev_b32 v34, 3, v24 // (op 21618) 

s_nop 0
v_readfirstlane_b32 s53, v34 // (op 21618) Wait state hazard: VALU Write Readlane Hazard

s_and_b32 s54, s7, 4294967292 // (op 22680) 
v_and_b32 v24, 63, v1 // (op 22680) 
s_add_u32 s55, s0, s54 // (op 22680) 

v_lshrrev_b32 v34, 4, v24 // (op 22680) 
v_and_b32 v35, 3, v34 // (op 22680) 

s_mul_i32 s54, s16, s8 // (op 22680) 
s_add_u32 s54, s54, s55 // (op 22680) 

v_and_b32 v34, 15, v1 // (op 22680) 
s_mul_hi_u32 s55, s54, s13 // (op 22680) 
v_lshl_add_u32 v64, v35, 4, v34 // (op 22680) 


s_sub_u32 s76, s54, s55 // (op 22680) 
v_lshrrev_b32 v34, 7, v1 // (op 22680) 
s_lshr_b32 s77, s76, 1 // (op 22680) 

v_lshl_add_u32 v35, v34, 6, v64 // (op 22680) 


s_add_u32 s76, s77, s55 // (op 22680) 


v_lshrrev_b32 v34, 6, v24 // (op 22680) 

s_lshr_b32 s55, s76, s11 // (op 22680) 

v_lshlrev_b32 v24, 2, v34 // (op 22680) 

s_and_b32 s76, s11, 2147483648 // (op 22680) 
s_cmp_eq_i32 s76, 0 // (op 22680) 
s_mov_b32 s77, scc // (op 22680) 

s_cmp_eq_i32 s77, 0 // (op 22680) 
s_cselect_b32 s76, s54, s55 // (op 22680) 


s_mul_hi_u32 s55, s76, s12 // (op 22680) 
s_sub_u32 s77, s76, s55 // (op 22680) 
s_lshr_b32 s78, s77, 1 // (op 22680) 

s_add_u32 s77, s78, s55 // (op 22680) 


s_lshr_b32 s55, s77, s10 // (op 22680) 

s_and_b32 s77, s10, 2147483648 // (op 22680) 
s_cmp_eq_i32 s77, 0 // (op 22680) 
s_mov_b32 s78, scc // (op 22680) 

s_cmp_eq_i32 s78, 0 // (op 22680) 
s_cselect_b32 s77, s76, s55 // (op 22680) 


v_lshl_add_u32 v34, s77, 7, v35 // (op 22680) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 22680) 
s_mul_i32 s55, s76, s78 // (op 22680) 


s_sub_u32 s76, s54, s55 // (op 22680) 


v_lshl_add_u32 v35, s76, 2, v24 // (op 22680) 


v_mul_lo_u32 v24, v34, s40 // (op 22680) 
v_add_u32 v24, v24, v35 // (op 22680) 


s_mul_i32 s55, s40, s35 // (op 22682) most significant: low of low * high
s_mul_i32 s76, s41, s34 // (op 22682) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 22682) least significant
s_mul_hi_u32 s77, s40, s34 // (op 22682) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 22682) most significant: sum
s_add_u32 s55, s55, s77 // (op 22682) most significant: sum


s_mov_b32 s76, s42 // (op 22682) reg expression
s_mov_b32 s77, s43 // (op 22682) reg expression
s_mov_b32 s78, s54 // (op 22682) reg expression
s_mov_b32 s79, 131072 // (op 22682) call()

s_and_b32 s54, s7, 4294967292 // (op 22707) 
v_and_b32 v34, 63, v1 // (op 22707) 
s_add_u32 s55, s0, s54 // (op 22707) 

v_lshrrev_b32 v35, 4, v34 // (op 22707) 
v_and_b32 v64, 3, v35 // (op 22707) 

s_mul_i32 s54, s16, s8 // (op 22707) 
s_add_u32 s54, s54, s55 // (op 22707) 

v_and_b32 v35, 15, v1 // (op 22707) 
s_mul_hi_u32 s55, s54, s13 // (op 22707) 
v_lshl_add_u32 v65, v64, 4, v35 // (op 22707) 


s_sub_u32 s80, s54, s55 // (op 22707) 
v_lshrrev_b32 v35, 6, v1 // (op 22707) 
s_lshr_b32 s81, s80, 1 // (op 22707) 

v_and_b32 v64, 1, v35 // (op 22707) 

s_add_u32 s80, s81, s55 // (op 22707) 


v_lshl_add_u32 v35, v64, 6, v65 // (op 22707) 


s_lshr_b32 s55, s80, s11 // (op 22707) 

v_lshrrev_b32 v64, 6, v34 // (op 22707) 

s_and_b32 s80, s11, 2147483648 // (op 22707) 
v_lshlrev_b32 v34, 2, v64 // (op 22707) 

s_cmp_eq_i32 s80, 0 // (op 22707) 
s_mov_b32 s81, scc // (op 22707) 

s_cmp_eq_i32 s81, 0 // (op 22707) 
s_cselect_b32 s80, s54, s55 // (op 22707) 


s_mul_hi_u32 s55, s80, s12 // (op 22707) 
s_sub_u32 s81, s80, s55 // (op 22707) 
s_lshr_b32 s82, s81, 1 // (op 22707) 

s_add_u32 s81, s82, s55 // (op 22707) 


s_lshr_b32 s55, s81, s10 // (op 22707) 

s_and_b32 s81, s10, 2147483648 // (op 22707) 
s_cmp_eq_i32 s81, 0 // (op 22707) 
s_mov_b32 s82, scc // (op 22707) 

s_cmp_eq_i32 s82, 0 // (op 22707) 
s_cselect_b32 s81, s80, s55 // (op 22707) 


s_mul_i32 s55, s81, s14 // (op 22707) 

s_sub_u32 s81, s80, s55 // (op 22707) 

v_lshl_add_u32 v64, s81, 7, v35 // (op 22707) 


v_mul_lo_u32 v35, v64, s44 // (op 22707) 

s_ashr_i64 s[82:83], s[2:3], 7 // (op 22707) 
s_mul_i32 s55, s80, s82 // (op 22707) 


s_sub_u32 s80, s54, s55 // (op 22707) 


v_lshl_add_u32 v64, s80, 2, v34 // (op 22707) 


v_add_u32 v34, v64, v35 // (op 22707) 


s_mul_i32 s55, s44, s23 // (op 22709) most significant: low of low * high
s_mul_i32 s80, s45, s22 // (op 22709) most significant: low of high * low
s_mul_i32 s54, s44, s22 // (op 22709) least significant
s_mul_hi_u32 s81, s44, s22 // (op 22709) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 22709) most significant: sum
s_add_u32 s55, s55, s81 // (op 22709) most significant: sum


s_mov_b32 s80, s46 // (op 22709) reg expression
s_mov_b32 s81, s47 // (op 22709) reg expression
s_mov_b32 s82, s54 // (op 22709) reg expression
s_mov_b32 s83, 131072 // (op 22709) call()

s_and_b32 s54, s7, 4294967292 // (op 8895) 
s_cmp_lt_u32 s54, s7 // (op 8894) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_8894 // (op 8894) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_8894 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_8894:
s_barrier  // (op 23133) 
s_add_u32 s54, s54, 1 // (op 8896) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 21087) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 21087) 

v_lshrrev_b64 v[26:27], 3, s[86:87] // (op 21087) 


v_mov_b64 v[28:29], v[32:33] // (op 21087) 
v_mov_b64 v[36:37], v[28:29] // (op 21087) 
s_add_u32 s55, 8192, s52 // (op 21087) 
s_mov_b32 m0, s55 // (op 21087) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 21087) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v26 // (op 21087) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 21087) most significant half

s_add_u32 m0, m0, 4096 // (op 21087) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 21087) Load value direct to lds




v_add_co_u32 v32, s[84:85], 64, v32 // (op 21198) least significant half
v_addc_co_u32 v33, s[84:85], 0, v33, s[84:85] // (op 21198) most significant half

s_barrier  // (op 23153) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 21093) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 21093) 

v_lshrrev_b64 v[26:27], 3, s[86:87] // (op 21093) 


v_mov_b64 v[28:29], v[30:31] // (op 21093) 
v_mov_b64 v[36:37], v[28:29] // (op 21093) 
s_mov_b32 m0, s53 // (op 21093) reg expression
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 21093) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v26 // (op 21093) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 21093) most significant half

s_add_u32 m0, m0, 4096 // (op 21093) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 21093) Load value direct to lds



v_add_co_u32 v30, s[84:85], 64, v30 // (op 21364) least significant half
v_addc_co_u32 v31, s[84:85], 0, v31, s[84:85] // (op 21364) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 23140) 



ds_read_b128 v[26:29], v3 offset:8192 // (op 5766) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 23159) 



ds_read_b128 v[36:39], v4 // (op 5768) Load local data 


v_mov_b32 v35, v24 // (op 5770) 
v_mov_b32 v40, v35 // (op 5770) reg expression

buffer_load_dwordx4 v[42:45], v40, s[76:79], 0 offen // (op 5770) Load value


v_add_u32 v24, 4, v24 // (op 22670) 


v_mov_b32 v35, v34 // (op 5772) 
v_mov_b32 v40, v35 // (op 5772) reg expression

buffer_load_dwordx4 v[46:49], v40, s[80:83], 0 offen // (op 5772) Load value


v_add_u32 v34, 4, v34 // (op 22697) 
s_waitcnt vmcnt(1)
v_bfe_u32 v50, v42, 0, 8 // (op 19644) 



v_bfe_u32 v51, v42, 8, 8 // (op 19644) 
v_bfe_u32 v52, v42, 16, 8 // (op 19644) 
v_bfe_u32 v53, v42, 24, 8 // (op 19644) 
v_permlane16_swap_b32 v50, v51 // (op 19644) 
ds_read_b128 v[54:57], v4 offset:1024 // (op 6003) Load local data 
v_permlane16_swap_b32 v52, v53 // (op 19644) 



s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 19644) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 19644) 
ds_read_b128 v[58:61], v4 offset:2048 // (op 6229) Load local data 




ds_read_b128 v[40:43], v4 offset:3072 // (op 6455) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v62, v46, 0, 8 // (op 19674) 



v_bfe_u32 v63, v46, 8, 8 // (op 19674) 
v_bfe_u32 v64, v46, 16, 8 // (op 19674) 
v_bfe_u32 v65, v46, 24, 8 // (op 19674) 
v_permlane16_swap_b32 v62, v63 // (op 19674) 
s_nop 0
v_permlane16_swap_b32 v64, v65 // (op 19674) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v62, v64 // (op 19674) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v63, v65 // (op 19674) 
ds_read_b128 v[66:69], v3 offset:9216 // (op 6815) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[36:39], a[0:3], v50, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5773) 
ds_read_b128 v[44:47], v3 offset:10240 // (op 7817) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[54:57], a[4:7], v50, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6006) 
ds_read_b128 v[70:73], v3 offset:11264 // (op 8819) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:29], v[58:61], a[8:11], v50, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6232) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:29], v[40:43], a[12:15], v50, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6458) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[66:69], v[36:39], a[16:19], v51, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6819) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:69], v[58:61], a[24:27], v51, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6873) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[66:69], v[40:43], a[28:31], v51, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6900) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[66:69], v[54:57], a[20:23], v51, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6846) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[44:47], v[36:39], a[32:35], v52, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7821) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[44:47], v[58:61], a[40:43], v52, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7875) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[44:47], v[40:43], a[44:47], v52, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7902) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[44:47], v[54:57], a[36:39], v52, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7848) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[70:73], v[36:39], a[48:51], v53, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8823) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:73], v[58:61], a[56:59], v53, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8877) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[70:73], v[40:43], a[60:63], v53, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8904) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[70:73], v[54:57], a[52:55], v53, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8850) 




s_cmp_lt_u32 s54, s7 // (op 8894) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_8894 // (op 8894) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_8894 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_8894:









s_barrier  // (op 23721) 


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 265) 
s_cmp_lg_u32 s18, 0 // (op 291) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_291 // (op 291) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_291



v_add_u32 v21, 49152, v20 // (op 10373) 
buffer_store_dword a48, v21, s[64:67], 0 offen sc1 // (op 10373) Store value

v_add_u32 v21, 49664, v20 // (op 10373) 
buffer_store_dword a49, v21, s[64:67], 0 offen sc1 // (op 10373) Store value

v_add_u32 v21, 50176, v20 // (op 10373) 
buffer_store_dword a50, v21, s[64:67], 0 offen sc1 // (op 10373) Store value

v_add_u32 v21, 50688, v20 // (op 10373) 
buffer_store_dword a51, v21, s[64:67], 0 offen sc1 // (op 10373) Store value




v_add_u32 v21, 49280, v20 // (op 10375) 
buffer_store_dword a52, v21, s[64:67], 0 offen sc1 // (op 10375) Store value

v_add_u32 v21, 49792, v20 // (op 10375) 
buffer_store_dword a53, v21, s[64:67], 0 offen sc1 // (op 10375) Store value

v_add_u32 v21, 50304, v20 // (op 10375) 
buffer_store_dword a54, v21, s[64:67], 0 offen sc1 // (op 10375) Store value

v_add_u32 v21, 50816, v20 // (op 10375) 
buffer_store_dword a55, v21, s[64:67], 0 offen sc1 // (op 10375) Store value




v_add_u32 v21, 49408, v20 // (op 10377) 
buffer_store_dword a56, v21, s[64:67], 0 offen sc1 // (op 10377) Store value

v_add_u32 v21, 49920, v20 // (op 10377) 
buffer_store_dword a57, v21, s[64:67], 0 offen sc1 // (op 10377) Store value

v_add_u32 v21, 50432, v20 // (op 10377) 
buffer_store_dword a58, v21, s[64:67], 0 offen sc1 // (op 10377) Store value

v_add_u32 v21, 50944, v20 // (op 10377) 
buffer_store_dword a59, v21, s[64:67], 0 offen sc1 // (op 10377) Store value




v_add_u32 v21, 49536, v20 // (op 10379) 
buffer_store_dword a60, v21, s[64:67], 0 offen sc1 // (op 10379) Store value

v_add_u32 v21, 50048, v20 // (op 10379) 
buffer_store_dword a61, v21, s[64:67], 0 offen sc1 // (op 10379) Store value

v_add_u32 v21, 50560, v20 // (op 10379) 
buffer_store_dword a62, v21, s[64:67], 0 offen sc1 // (op 10379) Store value

v_add_u32 v21, 51072, v20 // (op 10379) 
buffer_store_dword a63, v21, s[64:67], 0 offen sc1 // (op 10379) Store value




v_add_u32 v21, 16384, v20 // (op 10325) 
buffer_store_dword a16, v21, s[64:67], 0 offen sc1 // (op 10325) Store value

v_add_u32 v21, 16896, v20 // (op 10325) 
buffer_store_dword a17, v21, s[64:67], 0 offen sc1 // (op 10325) Store value

v_add_u32 v21, 17408, v20 // (op 10325) 
buffer_store_dword a18, v21, s[64:67], 0 offen sc1 // (op 10325) Store value

v_add_u32 v21, 17920, v20 // (op 10325) 
buffer_store_dword a19, v21, s[64:67], 0 offen sc1 // (op 10325) Store value




v_add_u32 v21, 16512, v20 // (op 10327) 
buffer_store_dword a20, v21, s[64:67], 0 offen sc1 // (op 10327) Store value

v_add_u32 v21, 17024, v20 // (op 10327) 
buffer_store_dword a21, v21, s[64:67], 0 offen sc1 // (op 10327) Store value

v_add_u32 v21, 17536, v20 // (op 10327) 
buffer_store_dword a22, v21, s[64:67], 0 offen sc1 // (op 10327) Store value

v_add_u32 v21, 18048, v20 // (op 10327) 
buffer_store_dword a23, v21, s[64:67], 0 offen sc1 // (op 10327) Store value




v_add_u32 v21, 16640, v20 // (op 10329) 
buffer_store_dword a24, v21, s[64:67], 0 offen sc1 // (op 10329) Store value

v_add_u32 v21, 17152, v20 // (op 10329) 
buffer_store_dword a25, v21, s[64:67], 0 offen sc1 // (op 10329) Store value

v_add_u32 v21, 17664, v20 // (op 10329) 
buffer_store_dword a26, v21, s[64:67], 0 offen sc1 // (op 10329) Store value

v_add_u32 v21, 18176, v20 // (op 10329) 
buffer_store_dword a27, v21, s[64:67], 0 offen sc1 // (op 10329) Store value




v_add_u32 v21, 16768, v20 // (op 10331) 
buffer_store_dword a28, v21, s[64:67], 0 offen sc1 // (op 10331) Store value

v_add_u32 v21, 17280, v20 // (op 10331) 
buffer_store_dword a29, v21, s[64:67], 0 offen sc1 // (op 10331) Store value

v_add_u32 v21, 17792, v20 // (op 10331) 
buffer_store_dword a30, v21, s[64:67], 0 offen sc1 // (op 10331) Store value

v_add_u32 v21, 18304, v20 // (op 10331) 
buffer_store_dword a31, v21, s[64:67], 0 offen sc1 // (op 10331) Store value




v_add_u32 v21, 32768, v20 // (op 10349) 
buffer_store_dword a32, v21, s[64:67], 0 offen sc1 // (op 10349) Store value

v_add_u32 v21, 33280, v20 // (op 10349) 
buffer_store_dword a33, v21, s[64:67], 0 offen sc1 // (op 10349) Store value

v_add_u32 v21, 33792, v20 // (op 10349) 
buffer_store_dword a34, v21, s[64:67], 0 offen sc1 // (op 10349) Store value

v_add_u32 v21, 34304, v20 // (op 10349) 
buffer_store_dword a35, v21, s[64:67], 0 offen sc1 // (op 10349) Store value




v_add_u32 v21, 32896, v20 // (op 10351) 
buffer_store_dword a36, v21, s[64:67], 0 offen sc1 // (op 10351) Store value

v_add_u32 v21, 33408, v20 // (op 10351) 
buffer_store_dword a37, v21, s[64:67], 0 offen sc1 // (op 10351) Store value

v_add_u32 v21, 33920, v20 // (op 10351) 
buffer_store_dword a38, v21, s[64:67], 0 offen sc1 // (op 10351) Store value

v_add_u32 v21, 34432, v20 // (op 10351) 
buffer_store_dword a39, v21, s[64:67], 0 offen sc1 // (op 10351) Store value




v_add_u32 v21, 33024, v20 // (op 10353) 
buffer_store_dword a40, v21, s[64:67], 0 offen sc1 // (op 10353) Store value

v_add_u32 v21, 33536, v20 // (op 10353) 
buffer_store_dword a41, v21, s[64:67], 0 offen sc1 // (op 10353) Store value

v_add_u32 v21, 34048, v20 // (op 10353) 
buffer_store_dword a42, v21, s[64:67], 0 offen sc1 // (op 10353) Store value

v_add_u32 v21, 34560, v20 // (op 10353) 
buffer_store_dword a43, v21, s[64:67], 0 offen sc1 // (op 10353) Store value




v_add_u32 v21, 33152, v20 // (op 10355) 
buffer_store_dword a44, v21, s[64:67], 0 offen sc1 // (op 10355) Store value

v_add_u32 v21, 33664, v20 // (op 10355) 
buffer_store_dword a45, v21, s[64:67], 0 offen sc1 // (op 10355) Store value

v_add_u32 v21, 34176, v20 // (op 10355) 
buffer_store_dword a46, v21, s[64:67], 0 offen sc1 // (op 10355) Store value

v_add_u32 v21, 34688, v20 // (op 10355) 
buffer_store_dword a47, v21, s[64:67], 0 offen sc1 // (op 10355) Store value




buffer_store_dword a0, v20, s[64:67], 0 offen sc1 // (op 288) Store value
buffer_store_dword a1, v20, s[64:67], 0 offen offset:512 sc1 // (op 288) Store value
buffer_store_dword a2, v20, s[64:67], 0 offen offset:1024 sc1 // (op 288) Store value
buffer_store_dword a3, v20, s[64:67], 0 offen offset:1536 sc1 // (op 288) Store value



buffer_store_dword a4, v20, s[64:67], 0 offen offset:128 sc1 // (op 10299) Store value
buffer_store_dword a5, v20, s[64:67], 0 offen offset:640 sc1 // (op 10299) Store value
buffer_store_dword a6, v20, s[64:67], 0 offen offset:1152 sc1 // (op 10299) Store value
buffer_store_dword a7, v20, s[64:67], 0 offen offset:1664 sc1 // (op 10299) Store value



buffer_store_dword a8, v20, s[64:67], 0 offen offset:256 sc1 // (op 10300) Store value
buffer_store_dword a9, v20, s[64:67], 0 offen offset:768 sc1 // (op 10300) Store value
buffer_store_dword a10, v20, s[64:67], 0 offen offset:1280 sc1 // (op 10300) Store value
buffer_store_dword a11, v20, s[64:67], 0 offen offset:1792 sc1 // (op 10300) Store value



buffer_store_dword a12, v20, s[64:67], 0 offen offset:384 sc1 // (op 10301) Store value
buffer_store_dword a13, v20, s[64:67], 0 offen offset:896 sc1 // (op 10301) Store value
buffer_store_dword a14, v20, s[64:67], 0 offen offset:1408 sc1 // (op 10301) Store value
buffer_store_dword a15, v20, s[64:67], 0 offen offset:1920 sc1 // (op 10301) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 292) 
v_cmp_eq_i32 vcc, v1, 0 // (op 296) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_296 // (op 296) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_296
s_mov_b32 s52, 1 // (op 294) call()
s_mov_b32 s55, 0 // (op 295) convert
s_mov_b32 s54, s16 // (op 295) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 295) 
s_add_u32 s76, s48, s54 // (op 295) Start of Int64 add, locking SCC
s_addc_u32 s77, s49, s55 // (op 295) End of Int64 add, unlocking SCC
s_store_dword s52, s[76:77], 0 glc // (op 295) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_296 // (op 296) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_296
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_296:
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_296:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_291 // (op 291) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_291
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_291:
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_291:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 307) 
s_mov_b32 s52, scc // (op 307) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 307) 
s_mov_b32 s53, s54 // (op 307) convert

s_sub_u32 s54, s53, 1 // (op 307) 

s_cmp_lt_u32 s19, s54 // (op 307) 
s_mov_b32 s53, scc // (op 307) 

s_and_b32 s54, s52, s53 // (op 307) 
s_cmp_lg_u32 s54, 0 // (op 307) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_307 // (op 307) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_307
s_mov_b32 s52, 0 // (op 273) call()
s_mul_i32 s53, s52, s8 // (op 272) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 272) 
s_mov_b32 s76, s54 // (op 272) convert

s_sub_u32 s54, s76, s19 // (op 272) 

s_sub_u32 s55, s54, 1 // (op 272) 

s_cmp_lt_u32 s53, s55 // (op 272) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_272 // (op 272) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_272 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_272:
s_add_u32 s53, s16, 1 // (op 21919) 
v_lshrrev_b32 v20, 5, v1 // (op 21919) 
s_add_u32 s54, s53, s52 // (op 21919) 
s_lshl_b32 s54, s54, 7 // (op 21919) 

v_lshlrev_b32 v21, 2, v20 // (op 21919) 

v_add_lshl_u32 v20, s54, v21, 7 // (op 21919) 


v_and_b32 v21, -32, v1 // (op 21919) 
v_sub_u32 v23, v1, v21 // (op 21919) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 21919) 


s_lshl_b32 s53, s15, 16 // (op 21921) 
s_mov_b32 s64, s28 // (op 21921) reg expression
s_mov_b32 s65, s29 // (op 21921) reg expression
s_mov_b32 s66, s53 // (op 21921) reg expression
s_mov_b32 s67, 131072 // (op 21921) call()

RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_311:
s_add_u32 s53, s16, 1 // (op 309) 
s_add_u32 s54, s53, s52 // (op 309) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 309) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 309) 

s_add_u32 s76, s48, s54 // (op 309) Start of Int64 add, locking SCC
s_addc_u32 s77, s49, s55 // (op 309) End of Int64 add, unlocking SCC
s_load_dword s53, s[76:77], 0 glc // (op 309) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 311) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_311 if true)
s_barrier  // (op 315) 
v_cmp_eq_i32 vcc, v1, 0 // (op 314) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_314 // (op 314) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_314
s_mov_b32 s53, 0 // (op 312) call()
s_add_u32 s54, s16, 1 // (op 313) 
s_add_u32 s76, s54, s52 // (op 313) Start of Int64 add, locking SCC
s_addc_u32 s77, 0, 0 // (op 313) End of Int64 add, unlocking SCC
s_lshl_b64 s[76:77], s[76:77], 2 // (op 313) 

s_add_u32 s54, s48, s76 // (op 313) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s77 // (op 313) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 313) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_314 // (op 314) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_314
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_314:
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_314:




buffer_load_dword v26, v21, s[64:67], 0 offen // (op 326) Load value
buffer_load_dword v27, v21, s[64:67], 0 offen offset:512 // (op 326) Load value
buffer_load_dword v28, v21, s[64:67], 0 offen offset:1024 // (op 326) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:1536 // (op 326) Load value
v_accvgpr_read v30, a0 // (op 329) reg expression
v_accvgpr_read v31, a1 // (op 329) reg expression
v_accvgpr_read v32, a2 // (op 329) reg expression
v_accvgpr_read v33, a3 // (op 329) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 331) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 331) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 331) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 331) 

v_accvgpr_write a0, v30 // (op 333) reg expression
v_accvgpr_write a1, v31 // (op 333) reg expression
v_accvgpr_write a2, v32 // (op 333) reg expression
v_accvgpr_write a3, v33 // (op 333) reg expression




buffer_load_dword v26, v21, s[64:67], 0 offen offset:128 // (op 10448) Load value
buffer_load_dword v27, v21, s[64:67], 0 offen offset:640 // (op 10448) Load value
buffer_load_dword v28, v21, s[64:67], 0 offen offset:1152 // (op 10448) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:1664 // (op 10448) Load value
v_accvgpr_read v30, a4 // (op 10449) reg expression
v_accvgpr_read v31, a5 // (op 10449) reg expression
v_accvgpr_read v32, a6 // (op 10449) reg expression
v_accvgpr_read v33, a7 // (op 10449) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10450) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10450) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10450) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10450) 

v_accvgpr_write a4, v30 // (op 10451) reg expression
v_accvgpr_write a5, v31 // (op 10451) reg expression
v_accvgpr_write a6, v32 // (op 10451) reg expression
v_accvgpr_write a7, v33 // (op 10451) reg expression




buffer_load_dword v26, v21, s[64:67], 0 offen offset:256 // (op 10455) Load value
buffer_load_dword v27, v21, s[64:67], 0 offen offset:768 // (op 10455) Load value
buffer_load_dword v28, v21, s[64:67], 0 offen offset:1280 // (op 10455) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:1792 // (op 10455) Load value
v_accvgpr_read v30, a8 // (op 10456) reg expression
v_accvgpr_read v31, a9 // (op 10456) reg expression
v_accvgpr_read v32, a10 // (op 10456) reg expression
v_accvgpr_read v33, a11 // (op 10456) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10457) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10457) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10457) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10457) 

v_accvgpr_write a8, v30 // (op 10458) reg expression
v_accvgpr_write a9, v31 // (op 10458) reg expression
v_accvgpr_write a10, v32 // (op 10458) reg expression
v_accvgpr_write a11, v33 // (op 10458) reg expression




buffer_load_dword v26, v21, s[64:67], 0 offen offset:384 // (op 10462) Load value
buffer_load_dword v27, v21, s[64:67], 0 offen offset:896 // (op 10462) Load value
buffer_load_dword v28, v21, s[64:67], 0 offen offset:1408 // (op 10462) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:1920 // (op 10462) Load value
v_accvgpr_read v30, a12 // (op 10463) reg expression
v_accvgpr_read v31, a13 // (op 10463) reg expression
v_accvgpr_read v32, a14 // (op 10463) reg expression
v_accvgpr_read v33, a15 // (op 10463) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10464) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10464) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10464) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10464) 

v_accvgpr_write a12, v30 // (op 10465) reg expression
v_accvgpr_write a13, v31 // (op 10465) reg expression
v_accvgpr_write a14, v32 // (op 10465) reg expression
v_accvgpr_write a15, v33 // (op 10465) reg expression




v_mov_b32 v20, 16384 // (op 10499) 
v_add_i32 v23, v21, v20 // (op 10499) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10499) Load value

v_mov_b32 v20, 16896 // (op 10499) 
v_add_i32 v23, v21, v20 // (op 10499) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10499) Load value

v_mov_b32 v20, 17408 // (op 10499) 
v_add_i32 v23, v21, v20 // (op 10499) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10499) Load value

v_mov_b32 v20, 17920 // (op 10499) 
v_add_i32 v23, v21, v20 // (op 10499) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10499) Load value

v_accvgpr_read v30, a16 // (op 10496) reg expression
v_accvgpr_read v31, a17 // (op 10496) reg expression
v_accvgpr_read v32, a18 // (op 10496) reg expression
v_accvgpr_read v33, a19 // (op 10496) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10497) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10497) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10497) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10497) 

v_accvgpr_write a16, v30 // (op 10498) reg expression
v_accvgpr_write a17, v31 // (op 10498) reg expression
v_accvgpr_write a18, v32 // (op 10498) reg expression
v_accvgpr_write a19, v33 // (op 10498) reg expression




v_mov_b32 v20, 16512 // (op 10504) 
v_add_i32 v23, v21, v20 // (op 10504) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10504) Load value

v_mov_b32 v20, 17024 // (op 10504) 
v_add_i32 v23, v21, v20 // (op 10504) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10504) Load value

v_mov_b32 v20, 17536 // (op 10504) 
v_add_i32 v23, v21, v20 // (op 10504) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10504) Load value

v_mov_b32 v20, 18048 // (op 10504) 
v_add_i32 v23, v21, v20 // (op 10504) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10504) Load value

v_accvgpr_read v30, a20 // (op 10501) reg expression
v_accvgpr_read v31, a21 // (op 10501) reg expression
v_accvgpr_read v32, a22 // (op 10501) reg expression
v_accvgpr_read v33, a23 // (op 10501) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10502) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10502) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10502) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10502) 

v_accvgpr_write a20, v30 // (op 10503) reg expression
v_accvgpr_write a21, v31 // (op 10503) reg expression
v_accvgpr_write a22, v32 // (op 10503) reg expression
v_accvgpr_write a23, v33 // (op 10503) reg expression




v_mov_b32 v20, 16640 // (op 10509) 
v_add_i32 v23, v21, v20 // (op 10509) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10509) Load value

v_mov_b32 v20, 17152 // (op 10509) 
v_add_i32 v23, v21, v20 // (op 10509) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10509) Load value

v_mov_b32 v20, 17664 // (op 10509) 
v_add_i32 v23, v21, v20 // (op 10509) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10509) Load value

v_mov_b32 v20, 18176 // (op 10509) 
v_add_i32 v23, v21, v20 // (op 10509) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10509) Load value

v_accvgpr_read v30, a24 // (op 10506) reg expression
v_accvgpr_read v31, a25 // (op 10506) reg expression
v_accvgpr_read v32, a26 // (op 10506) reg expression
v_accvgpr_read v33, a27 // (op 10506) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10507) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10507) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10507) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10507) 

v_accvgpr_write a24, v30 // (op 10508) reg expression
v_accvgpr_write a25, v31 // (op 10508) reg expression
v_accvgpr_write a26, v32 // (op 10508) reg expression
v_accvgpr_write a27, v33 // (op 10508) reg expression




v_mov_b32 v20, 16768 // (op 10514) 
v_add_i32 v23, v21, v20 // (op 10514) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10514) Load value

v_mov_b32 v20, 17280 // (op 10514) 
v_add_i32 v23, v21, v20 // (op 10514) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10514) Load value

v_mov_b32 v20, 17792 // (op 10514) 
v_add_i32 v23, v21, v20 // (op 10514) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10514) Load value

v_mov_b32 v20, 18304 // (op 10514) 
v_add_i32 v23, v21, v20 // (op 10514) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10514) Load value

v_accvgpr_read v30, a28 // (op 10511) reg expression
v_accvgpr_read v31, a29 // (op 10511) reg expression
v_accvgpr_read v32, a30 // (op 10511) reg expression
v_accvgpr_read v33, a31 // (op 10511) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10512) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10512) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10512) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10512) 

v_accvgpr_write a28, v30 // (op 10513) reg expression
v_accvgpr_write a29, v31 // (op 10513) reg expression
v_accvgpr_write a30, v32 // (op 10513) reg expression
v_accvgpr_write a31, v33 // (op 10513) reg expression




v_mov_b32 v20, 32768 // (op 10547) 
v_add_i32 v23, v21, v20 // (op 10547) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10547) Load value

v_mov_b32 v20, 33280 // (op 10547) 
v_add_i32 v23, v21, v20 // (op 10547) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10547) Load value

v_mov_b32 v20, 33792 // (op 10547) 
v_add_i32 v23, v21, v20 // (op 10547) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10547) Load value

v_mov_b32 v20, 34304 // (op 10547) 
v_add_i32 v23, v21, v20 // (op 10547) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10547) Load value

v_accvgpr_read v30, a32 // (op 10544) reg expression
v_accvgpr_read v31, a33 // (op 10544) reg expression
v_accvgpr_read v32, a34 // (op 10544) reg expression
v_accvgpr_read v33, a35 // (op 10544) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10545) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10545) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10545) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10545) 

v_accvgpr_write a32, v30 // (op 10546) reg expression
v_accvgpr_write a33, v31 // (op 10546) reg expression
v_accvgpr_write a34, v32 // (op 10546) reg expression
v_accvgpr_write a35, v33 // (op 10546) reg expression




v_mov_b32 v20, 32896 // (op 10552) 
v_add_i32 v23, v21, v20 // (op 10552) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10552) Load value

v_mov_b32 v20, 33408 // (op 10552) 
v_add_i32 v23, v21, v20 // (op 10552) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10552) Load value

v_mov_b32 v20, 33920 // (op 10552) 
v_add_i32 v23, v21, v20 // (op 10552) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10552) Load value

v_mov_b32 v20, 34432 // (op 10552) 
v_add_i32 v23, v21, v20 // (op 10552) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10552) Load value

v_accvgpr_read v30, a36 // (op 10549) reg expression
v_accvgpr_read v31, a37 // (op 10549) reg expression
v_accvgpr_read v32, a38 // (op 10549) reg expression
v_accvgpr_read v33, a39 // (op 10549) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10550) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10550) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10550) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10550) 

v_accvgpr_write a36, v30 // (op 10551) reg expression
v_accvgpr_write a37, v31 // (op 10551) reg expression
v_accvgpr_write a38, v32 // (op 10551) reg expression
v_accvgpr_write a39, v33 // (op 10551) reg expression




v_mov_b32 v20, 33024 // (op 10557) 
v_add_i32 v23, v21, v20 // (op 10557) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10557) Load value

v_mov_b32 v20, 33536 // (op 10557) 
v_add_i32 v23, v21, v20 // (op 10557) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10557) Load value

v_mov_b32 v20, 34048 // (op 10557) 
v_add_i32 v23, v21, v20 // (op 10557) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10557) Load value

v_mov_b32 v20, 34560 // (op 10557) 
v_add_i32 v23, v21, v20 // (op 10557) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10557) Load value

v_accvgpr_read v30, a40 // (op 10554) reg expression
v_accvgpr_read v31, a41 // (op 10554) reg expression
v_accvgpr_read v32, a42 // (op 10554) reg expression
v_accvgpr_read v33, a43 // (op 10554) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10555) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10555) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10555) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10555) 

v_accvgpr_write a40, v30 // (op 10556) reg expression
v_accvgpr_write a41, v31 // (op 10556) reg expression
v_accvgpr_write a42, v32 // (op 10556) reg expression
v_accvgpr_write a43, v33 // (op 10556) reg expression




v_mov_b32 v20, 33152 // (op 10562) 
v_add_i32 v23, v21, v20 // (op 10562) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10562) Load value

v_mov_b32 v20, 33664 // (op 10562) 
v_add_i32 v23, v21, v20 // (op 10562) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10562) Load value

v_mov_b32 v20, 34176 // (op 10562) 
v_add_i32 v23, v21, v20 // (op 10562) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10562) Load value

v_mov_b32 v20, 34688 // (op 10562) 
v_add_i32 v23, v21, v20 // (op 10562) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10562) Load value

v_accvgpr_read v30, a44 // (op 10559) reg expression
v_accvgpr_read v31, a45 // (op 10559) reg expression
v_accvgpr_read v32, a46 // (op 10559) reg expression
v_accvgpr_read v33, a47 // (op 10559) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10560) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10560) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10560) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10560) 

v_accvgpr_write a44, v30 // (op 10561) reg expression
v_accvgpr_write a45, v31 // (op 10561) reg expression
v_accvgpr_write a46, v32 // (op 10561) reg expression
v_accvgpr_write a47, v33 // (op 10561) reg expression




v_mov_b32 v20, 49152 // (op 10595) 
v_add_i32 v23, v21, v20 // (op 10595) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10595) Load value

v_mov_b32 v20, 49664 // (op 10595) 
v_add_i32 v23, v21, v20 // (op 10595) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10595) Load value

v_mov_b32 v20, 50176 // (op 10595) 
v_add_i32 v23, v21, v20 // (op 10595) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10595) Load value

v_mov_b32 v20, 50688 // (op 10595) 
v_add_i32 v23, v21, v20 // (op 10595) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10595) Load value

v_accvgpr_read v30, a48 // (op 10592) reg expression
v_accvgpr_read v31, a49 // (op 10592) reg expression
v_accvgpr_read v32, a50 // (op 10592) reg expression
v_accvgpr_read v33, a51 // (op 10592) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10593) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10593) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10593) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10593) 

v_accvgpr_write a48, v30 // (op 10594) reg expression
v_accvgpr_write a49, v31 // (op 10594) reg expression
v_accvgpr_write a50, v32 // (op 10594) reg expression
v_accvgpr_write a51, v33 // (op 10594) reg expression




v_mov_b32 v20, 49280 // (op 10600) 
v_add_i32 v23, v21, v20 // (op 10600) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10600) Load value

v_mov_b32 v20, 49792 // (op 10600) 
v_add_i32 v23, v21, v20 // (op 10600) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10600) Load value

v_mov_b32 v20, 50304 // (op 10600) 
v_add_i32 v23, v21, v20 // (op 10600) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10600) Load value

v_mov_b32 v20, 50816 // (op 10600) 
v_add_i32 v23, v21, v20 // (op 10600) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10600) Load value

v_accvgpr_read v30, a52 // (op 10597) reg expression
v_accvgpr_read v31, a53 // (op 10597) reg expression
v_accvgpr_read v32, a54 // (op 10597) reg expression
v_accvgpr_read v33, a55 // (op 10597) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10598) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10598) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10598) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10598) 

v_accvgpr_write a52, v30 // (op 10599) reg expression
v_accvgpr_write a53, v31 // (op 10599) reg expression
v_accvgpr_write a54, v32 // (op 10599) reg expression
v_accvgpr_write a55, v33 // (op 10599) reg expression




v_mov_b32 v20, 49408 // (op 10605) 
v_add_i32 v23, v21, v20 // (op 10605) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10605) Load value

v_mov_b32 v20, 49920 // (op 10605) 
v_add_i32 v23, v21, v20 // (op 10605) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10605) Load value

v_mov_b32 v20, 50432 // (op 10605) 
v_add_i32 v23, v21, v20 // (op 10605) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10605) Load value

v_mov_b32 v20, 50944 // (op 10605) 
v_add_i32 v23, v21, v20 // (op 10605) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10605) Load value

v_accvgpr_read v30, a56 // (op 10602) reg expression
v_accvgpr_read v31, a57 // (op 10602) reg expression
v_accvgpr_read v32, a58 // (op 10602) reg expression
v_accvgpr_read v33, a59 // (op 10602) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10603) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10603) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10603) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10603) 

v_accvgpr_write a56, v30 // (op 10604) reg expression
v_accvgpr_write a57, v31 // (op 10604) reg expression
v_accvgpr_write a58, v32 // (op 10604) reg expression
v_accvgpr_write a59, v33 // (op 10604) reg expression




v_mov_b32 v20, 49536 // (op 10610) 
v_add_i32 v23, v21, v20 // (op 10610) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10610) Load value

v_mov_b32 v20, 50048 // (op 10610) 
v_add_i32 v23, v21, v20 // (op 10610) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10610) Load value

v_mov_b32 v20, 50560 // (op 10610) 
v_add_i32 v23, v21, v20 // (op 10610) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10610) Load value

v_mov_b32 v20, 51072 // (op 10610) 
v_add_i32 v23, v21, v20 // (op 10610) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10610) Load value

v_accvgpr_read v30, a60 // (op 10607) reg expression
v_accvgpr_read v31, a61 // (op 10607) reg expression
v_accvgpr_read v32, a62 // (op 10607) reg expression
v_accvgpr_read v33, a63 // (op 10607) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10608) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10608) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10608) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10608) 

v_accvgpr_write a60, v30 // (op 10609) reg expression
v_accvgpr_write a61, v31 // (op 10609) reg expression
v_accvgpr_write a62, v32 // (op 10609) reg expression
v_accvgpr_write a63, v33 // (op 10609) reg expression

s_add_u32 s52, s52, 1 // (op 274) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 272) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 272) 
s_mov_b32 s76, s54 // (op 272) convert

s_sub_u32 s54, s76, s19 // (op 272) 

s_sub_u32 s55, s54, 1 // (op 272) 

s_cmp_lt_u32 s53, s55 // (op 272) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_272 // (op 272) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_272 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_272:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_307 // (op 307) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_307
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_307:
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_307:

s_cmp_eq_i32 s18, 0 // (op 350) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_350 // (op 350) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_350
v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_116
v_accvgpr_read v26, a0 // (op 42) 
v_accvgpr_read v27, a1 // (op 42) 
v_accvgpr_read v28, a2 // (op 42) 
v_accvgpr_read v29, a3 // (op 42) 
v_mul_f32 v30, v19, v26 // (op 42) 
v_mul_f32 v31, v19, v27 // (op 42) 
v_mul_f32 v32, v19, v28 // (op 42) 
v_mul_f32 v33, v19, v29 // (op 42) 

v_cvt_f16_f32 v26, v30 // (op 48) 
v_cvt_f16_f32 v27, v31 // (op 48) 
v_cvt_f16_f32 v28, v32 // (op 48) 
v_cvt_f16_f32 v29, v33 // (op 48) 


v_pack_b32_f16 v20, v26, v27 // (op 50) 
v_pack_b32_f16 v21, v28, v29 // (op 50) 

v_mov_b32 v23, v25 // (op 50) reg expression

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 50) Store value



v_accvgpr_read v26, a4 // (op 10692) 
v_accvgpr_read v27, a5 // (op 10692) 
v_accvgpr_read v28, a6 // (op 10692) 
v_accvgpr_read v29, a7 // (op 10692) 
v_mul_f32 v30, v19, v26 // (op 10692) 
v_mul_f32 v31, v19, v27 // (op 10692) 
v_mul_f32 v32, v19, v28 // (op 10692) 
v_mul_f32 v33, v19, v29 // (op 10692) 

v_cvt_f16_f32 v26, v30 // (op 10694) 
v_cvt_f16_f32 v27, v31 // (op 10694) 
v_cvt_f16_f32 v28, v32 // (op 10694) 
v_cvt_f16_f32 v29, v33 // (op 10694) 


v_pack_b32_f16 v20, v26, v27 // (op 10695) 
v_pack_b32_f16 v21, v28, v29 // (op 10695) 

s_lshl_b32 s52, s24, 5 // (op 10695) 
v_add_u32 v23, v25, s52 // (op 10695) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10695) Store value



v_accvgpr_read v26, a8 // (op 10706) 
v_accvgpr_read v27, a9 // (op 10706) 
v_accvgpr_read v28, a10 // (op 10706) 
v_accvgpr_read v29, a11 // (op 10706) 
v_mul_f32 v30, v19, v26 // (op 10706) 
v_mul_f32 v31, v19, v27 // (op 10706) 
v_mul_f32 v32, v19, v28 // (op 10706) 
v_mul_f32 v33, v19, v29 // (op 10706) 

v_cvt_f16_f32 v26, v30 // (op 10708) 
v_cvt_f16_f32 v27, v31 // (op 10708) 
v_cvt_f16_f32 v28, v32 // (op 10708) 
v_cvt_f16_f32 v29, v33 // (op 10708) 


v_pack_b32_f16 v20, v26, v27 // (op 10709) 
v_pack_b32_f16 v21, v28, v29 // (op 10709) 

s_lshl_b32 s52, s24, 6 // (op 10709) 
v_add_u32 v23, v25, s52 // (op 10709) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10709) Store value



v_accvgpr_read v26, a12 // (op 10720) 
v_accvgpr_read v27, a13 // (op 10720) 
v_accvgpr_read v28, a14 // (op 10720) 
v_accvgpr_read v29, a15 // (op 10720) 
v_mul_f32 v30, v19, v26 // (op 10720) 
v_mul_f32 v31, v19, v27 // (op 10720) 
v_mul_f32 v32, v19, v28 // (op 10720) 
v_mul_f32 v33, v19, v29 // (op 10720) 

v_cvt_f16_f32 v26, v30 // (op 10722) 
v_cvt_f16_f32 v27, v31 // (op 10722) 
v_cvt_f16_f32 v28, v32 // (op 10722) 
v_cvt_f16_f32 v29, v33 // (op 10722) 


v_pack_b32_f16 v20, v26, v27 // (op 10723) 
v_pack_b32_f16 v21, v28, v29 // (op 10723) 

s_lshl_b32 s52, s24, 5 // (op 10723) 
s_mul_i32 s53, 3, s52 // (op 10723) 

v_add_u32 v23, v25, s53 // (op 10723) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10723) Store value



v_accvgpr_read v26, a16 // (op 10756) 
v_accvgpr_read v27, a17 // (op 10756) 
v_accvgpr_read v28, a18 // (op 10756) 
v_accvgpr_read v29, a19 // (op 10756) 
v_mul_f32 v30, v19, v26 // (op 10756) 
v_mul_f32 v31, v19, v27 // (op 10756) 
v_mul_f32 v32, v19, v28 // (op 10756) 
v_mul_f32 v33, v19, v29 // (op 10756) 

v_cvt_f16_f32 v26, v30 // (op 10758) 
v_cvt_f16_f32 v27, v31 // (op 10758) 
v_cvt_f16_f32 v28, v32 // (op 10758) 
v_cvt_f16_f32 v29, v33 // (op 10758) 


v_pack_b32_f16 v20, v26, v27 // (op 10760) 
v_pack_b32_f16 v21, v28, v29 // (op 10760) 

v_add_u32 v23, 32, v25 // (op 10760) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10760) Store value



v_accvgpr_read v26, a20 // (op 10770) 
v_accvgpr_read v27, a21 // (op 10770) 
v_accvgpr_read v28, a22 // (op 10770) 
v_accvgpr_read v29, a23 // (op 10770) 
v_mul_f32 v30, v19, v26 // (op 10770) 
v_mul_f32 v31, v19, v27 // (op 10770) 
v_mul_f32 v32, v19, v28 // (op 10770) 
v_mul_f32 v33, v19, v29 // (op 10770) 

v_cvt_f16_f32 v26, v30 // (op 10772) 
v_cvt_f16_f32 v27, v31 // (op 10772) 
v_cvt_f16_f32 v28, v32 // (op 10772) 
v_cvt_f16_f32 v29, v33 // (op 10772) 


v_pack_b32_f16 v20, v26, v27 // (op 10762) 
v_pack_b32_f16 v21, v28, v29 // (op 10762) 

s_lshl_b32 s52, s24, 5 // (op 10762) 
s_add_u32 s52, s52, 32 // (op 10762) 
v_add_u32 v23, v25, s52 // (op 10762) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10762) Store value



v_accvgpr_read v26, a24 // (op 10776) 
v_accvgpr_read v27, a25 // (op 10776) 
v_accvgpr_read v28, a26 // (op 10776) 
v_accvgpr_read v29, a27 // (op 10776) 
v_mul_f32 v30, v19, v26 // (op 10776) 
v_mul_f32 v31, v19, v27 // (op 10776) 
v_mul_f32 v32, v19, v28 // (op 10776) 
v_mul_f32 v33, v19, v29 // (op 10776) 

v_cvt_f16_f32 v26, v30 // (op 10778) 
v_cvt_f16_f32 v27, v31 // (op 10778) 
v_cvt_f16_f32 v28, v32 // (op 10778) 
v_cvt_f16_f32 v29, v33 // (op 10778) 


v_pack_b32_f16 v20, v26, v27 // (op 10764) 
v_pack_b32_f16 v21, v28, v29 // (op 10764) 

s_lshl_b32 s52, s24, 6 // (op 10764) 
s_add_u32 s52, s52, 32 // (op 10764) 
v_add_u32 v23, v25, s52 // (op 10764) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10764) Store value



v_accvgpr_read v26, a28 // (op 10782) 
v_accvgpr_read v27, a29 // (op 10782) 
v_accvgpr_read v28, a30 // (op 10782) 
v_accvgpr_read v29, a31 // (op 10782) 
v_mul_f32 v30, v19, v26 // (op 10782) 
v_mul_f32 v31, v19, v27 // (op 10782) 
v_mul_f32 v32, v19, v28 // (op 10782) 
v_mul_f32 v33, v19, v29 // (op 10782) 

v_cvt_f16_f32 v26, v30 // (op 10784) 
v_cvt_f16_f32 v27, v31 // (op 10784) 
v_cvt_f16_f32 v28, v32 // (op 10784) 
v_cvt_f16_f32 v29, v33 // (op 10784) 


v_pack_b32_f16 v20, v26, v27 // (op 10766) 
v_pack_b32_f16 v21, v28, v29 // (op 10766) 

s_lshl_b32 s52, s24, 5 // (op 10766) 
s_mul_i32 s53, 3, s52 // (op 10766) 
s_add_u32 s53, s53, 32 // (op 10766) 

v_add_u32 v23, v25, s53 // (op 10766) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10766) Store value



v_accvgpr_read v26, a32 // (op 10832) 
v_accvgpr_read v27, a33 // (op 10832) 
v_accvgpr_read v28, a34 // (op 10832) 
v_accvgpr_read v29, a35 // (op 10832) 
v_mul_f32 v30, v19, v26 // (op 10832) 
v_mul_f32 v31, v19, v27 // (op 10832) 
v_mul_f32 v32, v19, v28 // (op 10832) 
v_mul_f32 v33, v19, v29 // (op 10832) 

v_cvt_f16_f32 v26, v30 // (op 10834) 
v_cvt_f16_f32 v27, v31 // (op 10834) 
v_cvt_f16_f32 v28, v32 // (op 10834) 
v_cvt_f16_f32 v29, v33 // (op 10834) 


v_pack_b32_f16 v20, v26, v27 // (op 10836) 
v_pack_b32_f16 v21, v28, v29 // (op 10836) 

v_add_u32 v23, 64, v25 // (op 10836) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10836) Store value



v_accvgpr_read v26, a36 // (op 10846) 
v_accvgpr_read v27, a37 // (op 10846) 
v_accvgpr_read v28, a38 // (op 10846) 
v_accvgpr_read v29, a39 // (op 10846) 
v_mul_f32 v30, v19, v26 // (op 10846) 
v_mul_f32 v31, v19, v27 // (op 10846) 
v_mul_f32 v32, v19, v28 // (op 10846) 
v_mul_f32 v33, v19, v29 // (op 10846) 

v_cvt_f16_f32 v26, v30 // (op 10848) 
v_cvt_f16_f32 v27, v31 // (op 10848) 
v_cvt_f16_f32 v28, v32 // (op 10848) 
v_cvt_f16_f32 v29, v33 // (op 10848) 


v_pack_b32_f16 v20, v26, v27 // (op 10838) 
v_pack_b32_f16 v21, v28, v29 // (op 10838) 

s_lshl_b32 s52, s24, 5 // (op 10838) 
s_add_u32 s52, s52, 64 // (op 10838) 
v_add_u32 v23, v25, s52 // (op 10838) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10838) Store value



v_accvgpr_read v26, a40 // (op 10852) 
v_accvgpr_read v27, a41 // (op 10852) 
v_accvgpr_read v28, a42 // (op 10852) 
v_accvgpr_read v29, a43 // (op 10852) 
v_mul_f32 v30, v19, v26 // (op 10852) 
v_mul_f32 v31, v19, v27 // (op 10852) 
v_mul_f32 v32, v19, v28 // (op 10852) 
v_mul_f32 v33, v19, v29 // (op 10852) 

v_cvt_f16_f32 v26, v30 // (op 10854) 
v_cvt_f16_f32 v27, v31 // (op 10854) 
v_cvt_f16_f32 v28, v32 // (op 10854) 
v_cvt_f16_f32 v29, v33 // (op 10854) 


v_pack_b32_f16 v20, v26, v27 // (op 10840) 
v_pack_b32_f16 v21, v28, v29 // (op 10840) 

s_lshl_b32 s52, s24, 6 // (op 10840) 
s_add_u32 s52, s52, 64 // (op 10840) 
v_add_u32 v23, v25, s52 // (op 10840) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10840) Store value



v_accvgpr_read v26, a44 // (op 10858) 
v_accvgpr_read v27, a45 // (op 10858) 
v_accvgpr_read v28, a46 // (op 10858) 
v_accvgpr_read v29, a47 // (op 10858) 
v_mul_f32 v30, v19, v26 // (op 10858) 
v_mul_f32 v31, v19, v27 // (op 10858) 
v_mul_f32 v32, v19, v28 // (op 10858) 
v_mul_f32 v33, v19, v29 // (op 10858) 

v_cvt_f16_f32 v26, v30 // (op 10860) 
v_cvt_f16_f32 v27, v31 // (op 10860) 
v_cvt_f16_f32 v28, v32 // (op 10860) 
v_cvt_f16_f32 v29, v33 // (op 10860) 


v_pack_b32_f16 v20, v26, v27 // (op 10842) 
v_pack_b32_f16 v21, v28, v29 // (op 10842) 

s_lshl_b32 s52, s24, 5 // (op 10842) 
s_mul_i32 s53, 3, s52 // (op 10842) 
s_add_u32 s53, s53, 64 // (op 10842) 

v_add_u32 v23, v25, s53 // (op 10842) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10842) Store value



v_accvgpr_read v26, a48 // (op 10908) 
v_accvgpr_read v27, a49 // (op 10908) 
v_accvgpr_read v28, a50 // (op 10908) 
v_accvgpr_read v29, a51 // (op 10908) 
v_mul_f32 v30, v19, v26 // (op 10908) 
v_mul_f32 v31, v19, v27 // (op 10908) 
v_mul_f32 v32, v19, v28 // (op 10908) 
v_mul_f32 v33, v19, v29 // (op 10908) 

v_cvt_f16_f32 v26, v30 // (op 10910) 
v_cvt_f16_f32 v27, v31 // (op 10910) 
v_cvt_f16_f32 v28, v32 // (op 10910) 
v_cvt_f16_f32 v29, v33 // (op 10910) 


v_pack_b32_f16 v20, v26, v27 // (op 10912) 
v_pack_b32_f16 v21, v28, v29 // (op 10912) 

v_add_u32 v23, 96, v25 // (op 10912) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10912) Store value



v_accvgpr_read v26, a52 // (op 10922) 
v_accvgpr_read v27, a53 // (op 10922) 
v_accvgpr_read v28, a54 // (op 10922) 
v_accvgpr_read v29, a55 // (op 10922) 
v_mul_f32 v30, v19, v26 // (op 10922) 
v_mul_f32 v31, v19, v27 // (op 10922) 
v_mul_f32 v32, v19, v28 // (op 10922) 
v_mul_f32 v33, v19, v29 // (op 10922) 

v_cvt_f16_f32 v26, v30 // (op 10924) 
v_cvt_f16_f32 v27, v31 // (op 10924) 
v_cvt_f16_f32 v28, v32 // (op 10924) 
v_cvt_f16_f32 v29, v33 // (op 10924) 


v_pack_b32_f16 v20, v26, v27 // (op 10914) 
v_pack_b32_f16 v21, v28, v29 // (op 10914) 

s_lshl_b32 s52, s24, 5 // (op 10914) 
s_add_u32 s52, s52, 96 // (op 10914) 
v_add_u32 v23, v25, s52 // (op 10914) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10914) Store value



v_accvgpr_read v26, a56 // (op 10928) 
v_accvgpr_read v27, a57 // (op 10928) 
v_accvgpr_read v28, a58 // (op 10928) 
v_accvgpr_read v29, a59 // (op 10928) 
v_mul_f32 v30, v19, v26 // (op 10928) 
v_mul_f32 v31, v19, v27 // (op 10928) 
v_mul_f32 v32, v19, v28 // (op 10928) 
v_mul_f32 v33, v19, v29 // (op 10928) 

v_cvt_f16_f32 v26, v30 // (op 10930) 
v_cvt_f16_f32 v27, v31 // (op 10930) 
v_cvt_f16_f32 v28, v32 // (op 10930) 
v_cvt_f16_f32 v29, v33 // (op 10930) 


v_pack_b32_f16 v20, v26, v27 // (op 10916) 
v_pack_b32_f16 v21, v28, v29 // (op 10916) 

s_lshl_b32 s52, s24, 6 // (op 10916) 
s_add_u32 s52, s52, 96 // (op 10916) 
v_add_u32 v23, v25, s52 // (op 10916) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10916) Store value



v_accvgpr_read v26, a60 // (op 10934) 
v_accvgpr_read v27, a61 // (op 10934) 
v_accvgpr_read v28, a62 // (op 10934) 
v_accvgpr_read v29, a63 // (op 10934) 
v_mul_f32 v30, v19, v26 // (op 10934) 
v_mul_f32 v31, v19, v27 // (op 10934) 
v_mul_f32 v32, v19, v28 // (op 10934) 
v_mul_f32 v33, v19, v29 // (op 10934) 

v_cvt_f16_f32 v26, v30 // (op 10936) 
v_cvt_f16_f32 v27, v31 // (op 10936) 
v_cvt_f16_f32 v28, v32 // (op 10936) 
v_cvt_f16_f32 v29, v33 // (op 10936) 


v_pack_b32_f16 v20, v26, v27 // (op 10918) 
v_pack_b32_f16 v21, v28, v29 // (op 10918) 

s_lshl_b32 s52, s24, 5 // (op 10918) 
s_mul_i32 s53, 3, s52 // (op 10918) 
s_add_u32 s53, s53, 96 // (op 10918) 

v_add_u32 v23, v25, s53 // (op 10918) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 10918) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_116
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_116:


v_add_u32 v20, 96, v22 // (op 11306) 

buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11306) Load value

v_accvgpr_read v28, a48 // (op 11293) 
v_accvgpr_read v29, a49 // (op 11293) 
v_accvgpr_read v30, a50 // (op 11293) 
v_accvgpr_read v31, a51 // (op 11293) 
v_mul_f32 v32, v19, v28 // (op 11293) 
v_mul_f32 v33, v19, v29 // (op 11293) 
v_mul_f32 v34, v19, v30 // (op 11293) 
v_mul_f32 v35, v19, v31 // (op 11293) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11305) 
v_lshrrev_b32 v21, 16, v26 // (op 11305) 
v_cvt_f32_f16 v21, v21 // (op 11305) 
v_mul_f32 v28, v0, v20 // (op 11305) 
v_mul_f32 v29, v0, v21 // (op 11305) 
v_cvt_f32_f16 v20, v27 // (op 11305) 
v_lshrrev_b32 v21, 16, v27 // (op 11305) 
v_cvt_f32_f16 v21, v21 // (op 11305) 
v_mul_f32 v30, v0, v20 // (op 11305) 
v_mul_f32 v31, v0, v21 // (op 11305) 


v_add_f32 v36, v28, v32 // (op 11294) 
v_add_f32 v37, v29, v33 // (op 11294) 
v_add_f32 v38, v30, v34 // (op 11294) 
v_add_f32 v39, v31, v35 // (op 11294) 


v_cvt_f16_f32 v26, v36 // (op 11295) 
v_cvt_f16_f32 v27, v37 // (op 11295) 
v_cvt_f16_f32 v28, v38 // (op 11295) 
v_cvt_f16_f32 v29, v39 // (op 11295) 


v_pack_b32_f16 v20, v26, v27 // (op 11297) 
v_pack_b32_f16 v21, v28, v29 // (op 11297) 

v_add_u32 v23, 96, v25 // (op 11297) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11297) Store value





s_lshl_b32 s52, s4, 5 // (op 11311) 
s_add_u32 s52, s52, 96 // (op 11311) 
v_add_u32 v20, v22, s52 // (op 11311) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11311) Load value

v_accvgpr_read v28, a52 // (op 11323) 
v_accvgpr_read v29, a53 // (op 11323) 
v_accvgpr_read v30, a54 // (op 11323) 
v_accvgpr_read v31, a55 // (op 11323) 
v_mul_f32 v32, v19, v28 // (op 11323) 
v_mul_f32 v33, v19, v29 // (op 11323) 
v_mul_f32 v34, v19, v30 // (op 11323) 
v_mul_f32 v35, v19, v31 // (op 11323) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11308) 
v_lshrrev_b32 v21, 16, v26 // (op 11308) 
v_cvt_f32_f16 v21, v21 // (op 11308) 
v_mul_f32 v28, v0, v20 // (op 11308) 
v_mul_f32 v29, v0, v21 // (op 11308) 
v_cvt_f32_f16 v20, v27 // (op 11308) 
v_lshrrev_b32 v21, 16, v27 // (op 11308) 
v_cvt_f32_f16 v21, v21 // (op 11308) 
v_mul_f32 v30, v0, v20 // (op 11308) 
v_mul_f32 v31, v0, v21 // (op 11308) 


v_add_f32 v36, v28, v32 // (op 11309) 
v_add_f32 v37, v29, v33 // (op 11309) 
v_add_f32 v38, v30, v34 // (op 11309) 
v_add_f32 v39, v31, v35 // (op 11309) 


v_cvt_f16_f32 v26, v36 // (op 11310) 
v_cvt_f16_f32 v27, v37 // (op 11310) 
v_cvt_f16_f32 v28, v38 // (op 11310) 
v_cvt_f16_f32 v29, v39 // (op 11310) 


v_pack_b32_f16 v20, v26, v27 // (op 11299) 
v_pack_b32_f16 v21, v28, v29 // (op 11299) 

s_lshl_b32 s52, s24, 5 // (op 11299) 
s_add_u32 s52, s52, 96 // (op 11299) 
v_add_u32 v23, v25, s52 // (op 11299) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11299) Store value





s_lshl_b32 s52, s4, 6 // (op 11316) 
s_add_u32 s52, s52, 96 // (op 11316) 
v_add_u32 v20, v22, s52 // (op 11316) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11316) Load value

v_accvgpr_read v28, a56 // (op 11325) 
v_accvgpr_read v29, a57 // (op 11325) 
v_accvgpr_read v30, a58 // (op 11325) 
v_accvgpr_read v31, a59 // (op 11325) 
v_mul_f32 v32, v19, v28 // (op 11325) 
v_mul_f32 v33, v19, v29 // (op 11325) 
v_mul_f32 v34, v19, v30 // (op 11325) 
v_mul_f32 v35, v19, v31 // (op 11325) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11313) 
v_lshrrev_b32 v21, 16, v26 // (op 11313) 
v_cvt_f32_f16 v21, v21 // (op 11313) 
v_mul_f32 v28, v0, v20 // (op 11313) 
v_mul_f32 v29, v0, v21 // (op 11313) 
v_cvt_f32_f16 v20, v27 // (op 11313) 
v_lshrrev_b32 v21, 16, v27 // (op 11313) 
v_cvt_f32_f16 v21, v21 // (op 11313) 
v_mul_f32 v30, v0, v20 // (op 11313) 
v_mul_f32 v31, v0, v21 // (op 11313) 


v_add_f32 v36, v28, v32 // (op 11314) 
v_add_f32 v37, v29, v33 // (op 11314) 
v_add_f32 v38, v30, v34 // (op 11314) 
v_add_f32 v39, v31, v35 // (op 11314) 


v_cvt_f16_f32 v26, v36 // (op 11315) 
v_cvt_f16_f32 v27, v37 // (op 11315) 
v_cvt_f16_f32 v28, v38 // (op 11315) 
v_cvt_f16_f32 v29, v39 // (op 11315) 


v_pack_b32_f16 v20, v26, v27 // (op 11301) 
v_pack_b32_f16 v21, v28, v29 // (op 11301) 

s_lshl_b32 s52, s24, 6 // (op 11301) 
s_add_u32 s52, s52, 96 // (op 11301) 
v_add_u32 v23, v25, s52 // (op 11301) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11301) Store value





s_lshl_b32 s52, s4, 5 // (op 11321) 
s_mul_i32 s53, 3, s52 // (op 11321) 
s_add_u32 s53, s53, 96 // (op 11321) 

v_add_u32 v20, v22, s53 // (op 11321) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11321) Load value

v_accvgpr_read v28, a60 // (op 11327) 
v_accvgpr_read v29, a61 // (op 11327) 
v_accvgpr_read v30, a62 // (op 11327) 
v_accvgpr_read v31, a63 // (op 11327) 
v_mul_f32 v32, v19, v28 // (op 11327) 
v_mul_f32 v33, v19, v29 // (op 11327) 
v_mul_f32 v34, v19, v30 // (op 11327) 
v_mul_f32 v35, v19, v31 // (op 11327) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11318) 
v_lshrrev_b32 v21, 16, v26 // (op 11318) 
v_cvt_f32_f16 v21, v21 // (op 11318) 
v_mul_f32 v28, v0, v20 // (op 11318) 
v_mul_f32 v29, v0, v21 // (op 11318) 
v_cvt_f32_f16 v20, v27 // (op 11318) 
v_lshrrev_b32 v21, 16, v27 // (op 11318) 
v_cvt_f32_f16 v21, v21 // (op 11318) 
v_mul_f32 v30, v0, v20 // (op 11318) 
v_mul_f32 v31, v0, v21 // (op 11318) 


v_add_f32 v36, v28, v32 // (op 11319) 
v_add_f32 v37, v29, v33 // (op 11319) 
v_add_f32 v38, v30, v34 // (op 11319) 
v_add_f32 v39, v31, v35 // (op 11319) 


v_cvt_f16_f32 v26, v36 // (op 11320) 
v_cvt_f16_f32 v27, v37 // (op 11320) 
v_cvt_f16_f32 v28, v38 // (op 11320) 
v_cvt_f16_f32 v29, v39 // (op 11320) 


v_pack_b32_f16 v20, v26, v27 // (op 11303) 
v_pack_b32_f16 v21, v28, v29 // (op 11303) 

s_lshl_b32 s52, s24, 5 // (op 11303) 
s_mul_i32 s53, 3, s52 // (op 11303) 
s_add_u32 s53, s53, 96 // (op 11303) 

v_add_u32 v23, v25, s53 // (op 11303) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11303) Store value





v_add_u32 v20, 64, v22 // (op 11219) 

buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11219) Load value

v_accvgpr_read v28, a32 // (op 11206) 
v_accvgpr_read v29, a33 // (op 11206) 
v_accvgpr_read v30, a34 // (op 11206) 
v_accvgpr_read v31, a35 // (op 11206) 
v_mul_f32 v32, v19, v28 // (op 11206) 
v_mul_f32 v33, v19, v29 // (op 11206) 
v_mul_f32 v34, v19, v30 // (op 11206) 
v_mul_f32 v35, v19, v31 // (op 11206) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11218) 
v_lshrrev_b32 v21, 16, v26 // (op 11218) 
v_cvt_f32_f16 v21, v21 // (op 11218) 
v_mul_f32 v28, v0, v20 // (op 11218) 
v_mul_f32 v29, v0, v21 // (op 11218) 
v_cvt_f32_f16 v20, v27 // (op 11218) 
v_lshrrev_b32 v21, 16, v27 // (op 11218) 
v_cvt_f32_f16 v21, v21 // (op 11218) 
v_mul_f32 v30, v0, v20 // (op 11218) 
v_mul_f32 v31, v0, v21 // (op 11218) 


v_add_f32 v36, v28, v32 // (op 11207) 
v_add_f32 v37, v29, v33 // (op 11207) 
v_add_f32 v38, v30, v34 // (op 11207) 
v_add_f32 v39, v31, v35 // (op 11207) 


v_cvt_f16_f32 v26, v36 // (op 11208) 
v_cvt_f16_f32 v27, v37 // (op 11208) 
v_cvt_f16_f32 v28, v38 // (op 11208) 
v_cvt_f16_f32 v29, v39 // (op 11208) 


v_pack_b32_f16 v20, v26, v27 // (op 11210) 
v_pack_b32_f16 v21, v28, v29 // (op 11210) 

v_add_u32 v23, 64, v25 // (op 11210) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11210) Store value





s_lshl_b32 s52, s4, 5 // (op 11224) 
s_add_u32 s52, s52, 64 // (op 11224) 
v_add_u32 v20, v22, s52 // (op 11224) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11224) Load value

v_accvgpr_read v28, a36 // (op 11236) 
v_accvgpr_read v29, a37 // (op 11236) 
v_accvgpr_read v30, a38 // (op 11236) 
v_accvgpr_read v31, a39 // (op 11236) 
v_mul_f32 v32, v19, v28 // (op 11236) 
v_mul_f32 v33, v19, v29 // (op 11236) 
v_mul_f32 v34, v19, v30 // (op 11236) 
v_mul_f32 v35, v19, v31 // (op 11236) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11221) 
v_lshrrev_b32 v21, 16, v26 // (op 11221) 
v_cvt_f32_f16 v21, v21 // (op 11221) 
v_mul_f32 v28, v0, v20 // (op 11221) 
v_mul_f32 v29, v0, v21 // (op 11221) 
v_cvt_f32_f16 v20, v27 // (op 11221) 
v_lshrrev_b32 v21, 16, v27 // (op 11221) 
v_cvt_f32_f16 v21, v21 // (op 11221) 
v_mul_f32 v30, v0, v20 // (op 11221) 
v_mul_f32 v31, v0, v21 // (op 11221) 


v_add_f32 v36, v28, v32 // (op 11222) 
v_add_f32 v37, v29, v33 // (op 11222) 
v_add_f32 v38, v30, v34 // (op 11222) 
v_add_f32 v39, v31, v35 // (op 11222) 


v_cvt_f16_f32 v26, v36 // (op 11223) 
v_cvt_f16_f32 v27, v37 // (op 11223) 
v_cvt_f16_f32 v28, v38 // (op 11223) 
v_cvt_f16_f32 v29, v39 // (op 11223) 


v_pack_b32_f16 v20, v26, v27 // (op 11212) 
v_pack_b32_f16 v21, v28, v29 // (op 11212) 

s_lshl_b32 s52, s24, 5 // (op 11212) 
s_add_u32 s52, s52, 64 // (op 11212) 
v_add_u32 v23, v25, s52 // (op 11212) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11212) Store value





s_lshl_b32 s52, s4, 6 // (op 11229) 
s_add_u32 s52, s52, 64 // (op 11229) 
v_add_u32 v20, v22, s52 // (op 11229) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11229) Load value

v_accvgpr_read v28, a40 // (op 11238) 
v_accvgpr_read v29, a41 // (op 11238) 
v_accvgpr_read v30, a42 // (op 11238) 
v_accvgpr_read v31, a43 // (op 11238) 
v_mul_f32 v32, v19, v28 // (op 11238) 
v_mul_f32 v33, v19, v29 // (op 11238) 
v_mul_f32 v34, v19, v30 // (op 11238) 
v_mul_f32 v35, v19, v31 // (op 11238) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11226) 
v_lshrrev_b32 v21, 16, v26 // (op 11226) 
v_cvt_f32_f16 v21, v21 // (op 11226) 
v_mul_f32 v28, v0, v20 // (op 11226) 
v_mul_f32 v29, v0, v21 // (op 11226) 
v_cvt_f32_f16 v20, v27 // (op 11226) 
v_lshrrev_b32 v21, 16, v27 // (op 11226) 
v_cvt_f32_f16 v21, v21 // (op 11226) 
v_mul_f32 v30, v0, v20 // (op 11226) 
v_mul_f32 v31, v0, v21 // (op 11226) 


v_add_f32 v36, v28, v32 // (op 11227) 
v_add_f32 v37, v29, v33 // (op 11227) 
v_add_f32 v38, v30, v34 // (op 11227) 
v_add_f32 v39, v31, v35 // (op 11227) 


v_cvt_f16_f32 v26, v36 // (op 11228) 
v_cvt_f16_f32 v27, v37 // (op 11228) 
v_cvt_f16_f32 v28, v38 // (op 11228) 
v_cvt_f16_f32 v29, v39 // (op 11228) 


v_pack_b32_f16 v20, v26, v27 // (op 11214) 
v_pack_b32_f16 v21, v28, v29 // (op 11214) 

s_lshl_b32 s52, s24, 6 // (op 11214) 
s_add_u32 s52, s52, 64 // (op 11214) 
v_add_u32 v23, v25, s52 // (op 11214) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11214) Store value





s_lshl_b32 s52, s4, 5 // (op 11234) 
s_mul_i32 s53, 3, s52 // (op 11234) 
s_add_u32 s53, s53, 64 // (op 11234) 

v_add_u32 v20, v22, s53 // (op 11234) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11234) Load value

v_accvgpr_read v28, a44 // (op 11240) 
v_accvgpr_read v29, a45 // (op 11240) 
v_accvgpr_read v30, a46 // (op 11240) 
v_accvgpr_read v31, a47 // (op 11240) 
v_mul_f32 v32, v19, v28 // (op 11240) 
v_mul_f32 v33, v19, v29 // (op 11240) 
v_mul_f32 v34, v19, v30 // (op 11240) 
v_mul_f32 v35, v19, v31 // (op 11240) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11231) 
v_lshrrev_b32 v21, 16, v26 // (op 11231) 
v_cvt_f32_f16 v21, v21 // (op 11231) 
v_mul_f32 v28, v0, v20 // (op 11231) 
v_mul_f32 v29, v0, v21 // (op 11231) 
v_cvt_f32_f16 v20, v27 // (op 11231) 
v_lshrrev_b32 v21, 16, v27 // (op 11231) 
v_cvt_f32_f16 v21, v21 // (op 11231) 
v_mul_f32 v30, v0, v20 // (op 11231) 
v_mul_f32 v31, v0, v21 // (op 11231) 


v_add_f32 v36, v28, v32 // (op 11232) 
v_add_f32 v37, v29, v33 // (op 11232) 
v_add_f32 v38, v30, v34 // (op 11232) 
v_add_f32 v39, v31, v35 // (op 11232) 


v_cvt_f16_f32 v26, v36 // (op 11233) 
v_cvt_f16_f32 v27, v37 // (op 11233) 
v_cvt_f16_f32 v28, v38 // (op 11233) 
v_cvt_f16_f32 v29, v39 // (op 11233) 


v_pack_b32_f16 v20, v26, v27 // (op 11216) 
v_pack_b32_f16 v21, v28, v29 // (op 11216) 

s_lshl_b32 s52, s24, 5 // (op 11216) 
s_mul_i32 s53, 3, s52 // (op 11216) 
s_add_u32 s53, s53, 64 // (op 11216) 

v_add_u32 v23, v25, s53 // (op 11216) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11216) Store value





v_add_u32 v20, 32, v22 // (op 11132) 

buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11132) Load value

v_accvgpr_read v28, a16 // (op 11119) 
v_accvgpr_read v29, a17 // (op 11119) 
v_accvgpr_read v30, a18 // (op 11119) 
v_accvgpr_read v31, a19 // (op 11119) 
v_mul_f32 v32, v19, v28 // (op 11119) 
v_mul_f32 v33, v19, v29 // (op 11119) 
v_mul_f32 v34, v19, v30 // (op 11119) 
v_mul_f32 v35, v19, v31 // (op 11119) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11131) 
v_lshrrev_b32 v21, 16, v26 // (op 11131) 
v_cvt_f32_f16 v21, v21 // (op 11131) 
v_mul_f32 v28, v0, v20 // (op 11131) 
v_mul_f32 v29, v0, v21 // (op 11131) 
v_cvt_f32_f16 v20, v27 // (op 11131) 
v_lshrrev_b32 v21, 16, v27 // (op 11131) 
v_cvt_f32_f16 v21, v21 // (op 11131) 
v_mul_f32 v30, v0, v20 // (op 11131) 
v_mul_f32 v31, v0, v21 // (op 11131) 


v_add_f32 v36, v28, v32 // (op 11120) 
v_add_f32 v37, v29, v33 // (op 11120) 
v_add_f32 v38, v30, v34 // (op 11120) 
v_add_f32 v39, v31, v35 // (op 11120) 


v_cvt_f16_f32 v26, v36 // (op 11121) 
v_cvt_f16_f32 v27, v37 // (op 11121) 
v_cvt_f16_f32 v28, v38 // (op 11121) 
v_cvt_f16_f32 v29, v39 // (op 11121) 


v_pack_b32_f16 v20, v26, v27 // (op 11123) 
v_pack_b32_f16 v21, v28, v29 // (op 11123) 

v_add_u32 v23, 32, v25 // (op 11123) 

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11123) Store value





s_lshl_b32 s52, s4, 5 // (op 11137) 
s_add_u32 s52, s52, 32 // (op 11137) 
v_add_u32 v20, v22, s52 // (op 11137) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11137) Load value

v_accvgpr_read v28, a20 // (op 11149) 
v_accvgpr_read v29, a21 // (op 11149) 
v_accvgpr_read v30, a22 // (op 11149) 
v_accvgpr_read v31, a23 // (op 11149) 
v_mul_f32 v32, v19, v28 // (op 11149) 
v_mul_f32 v33, v19, v29 // (op 11149) 
v_mul_f32 v34, v19, v30 // (op 11149) 
v_mul_f32 v35, v19, v31 // (op 11149) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11134) 
v_lshrrev_b32 v21, 16, v26 // (op 11134) 
v_cvt_f32_f16 v21, v21 // (op 11134) 
v_mul_f32 v28, v0, v20 // (op 11134) 
v_mul_f32 v29, v0, v21 // (op 11134) 
v_cvt_f32_f16 v20, v27 // (op 11134) 
v_lshrrev_b32 v21, 16, v27 // (op 11134) 
v_cvt_f32_f16 v21, v21 // (op 11134) 
v_mul_f32 v30, v0, v20 // (op 11134) 
v_mul_f32 v31, v0, v21 // (op 11134) 


v_add_f32 v36, v28, v32 // (op 11135) 
v_add_f32 v37, v29, v33 // (op 11135) 
v_add_f32 v38, v30, v34 // (op 11135) 
v_add_f32 v39, v31, v35 // (op 11135) 


v_cvt_f16_f32 v26, v36 // (op 11136) 
v_cvt_f16_f32 v27, v37 // (op 11136) 
v_cvt_f16_f32 v28, v38 // (op 11136) 
v_cvt_f16_f32 v29, v39 // (op 11136) 


v_pack_b32_f16 v20, v26, v27 // (op 11125) 
v_pack_b32_f16 v21, v28, v29 // (op 11125) 

s_lshl_b32 s52, s24, 5 // (op 11125) 
s_add_u32 s52, s52, 32 // (op 11125) 
v_add_u32 v23, v25, s52 // (op 11125) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11125) Store value





s_lshl_b32 s52, s4, 6 // (op 11142) 
s_add_u32 s52, s52, 32 // (op 11142) 
v_add_u32 v20, v22, s52 // (op 11142) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11142) Load value

v_accvgpr_read v28, a24 // (op 11151) 
v_accvgpr_read v29, a25 // (op 11151) 
v_accvgpr_read v30, a26 // (op 11151) 
v_accvgpr_read v31, a27 // (op 11151) 
v_mul_f32 v32, v19, v28 // (op 11151) 
v_mul_f32 v33, v19, v29 // (op 11151) 
v_mul_f32 v34, v19, v30 // (op 11151) 
v_mul_f32 v35, v19, v31 // (op 11151) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11139) 
v_lshrrev_b32 v21, 16, v26 // (op 11139) 
v_cvt_f32_f16 v21, v21 // (op 11139) 
v_mul_f32 v28, v0, v20 // (op 11139) 
v_mul_f32 v29, v0, v21 // (op 11139) 
v_cvt_f32_f16 v20, v27 // (op 11139) 
v_lshrrev_b32 v21, 16, v27 // (op 11139) 
v_cvt_f32_f16 v21, v21 // (op 11139) 
v_mul_f32 v30, v0, v20 // (op 11139) 
v_mul_f32 v31, v0, v21 // (op 11139) 


v_add_f32 v36, v28, v32 // (op 11140) 
v_add_f32 v37, v29, v33 // (op 11140) 
v_add_f32 v38, v30, v34 // (op 11140) 
v_add_f32 v39, v31, v35 // (op 11140) 


v_cvt_f16_f32 v26, v36 // (op 11141) 
v_cvt_f16_f32 v27, v37 // (op 11141) 
v_cvt_f16_f32 v28, v38 // (op 11141) 
v_cvt_f16_f32 v29, v39 // (op 11141) 


v_pack_b32_f16 v20, v26, v27 // (op 11127) 
v_pack_b32_f16 v21, v28, v29 // (op 11127) 

s_lshl_b32 s52, s24, 6 // (op 11127) 
s_add_u32 s52, s52, 32 // (op 11127) 
v_add_u32 v23, v25, s52 // (op 11127) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11127) Store value





s_lshl_b32 s52, s4, 5 // (op 11147) 
s_mul_i32 s53, 3, s52 // (op 11147) 
s_add_u32 s53, s53, 32 // (op 11147) 

v_add_u32 v20, v22, s53 // (op 11147) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11147) Load value

v_accvgpr_read v28, a28 // (op 11153) 
v_accvgpr_read v29, a29 // (op 11153) 
v_accvgpr_read v30, a30 // (op 11153) 
v_accvgpr_read v31, a31 // (op 11153) 
v_mul_f32 v32, v19, v28 // (op 11153) 
v_mul_f32 v33, v19, v29 // (op 11153) 
v_mul_f32 v34, v19, v30 // (op 11153) 
v_mul_f32 v35, v19, v31 // (op 11153) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11144) 
v_lshrrev_b32 v21, 16, v26 // (op 11144) 
v_cvt_f32_f16 v21, v21 // (op 11144) 
v_mul_f32 v28, v0, v20 // (op 11144) 
v_mul_f32 v29, v0, v21 // (op 11144) 
v_cvt_f32_f16 v20, v27 // (op 11144) 
v_lshrrev_b32 v21, 16, v27 // (op 11144) 
v_cvt_f32_f16 v21, v21 // (op 11144) 
v_mul_f32 v30, v0, v20 // (op 11144) 
v_mul_f32 v31, v0, v21 // (op 11144) 


v_add_f32 v36, v28, v32 // (op 11145) 
v_add_f32 v37, v29, v33 // (op 11145) 
v_add_f32 v38, v30, v34 // (op 11145) 
v_add_f32 v39, v31, v35 // (op 11145) 


v_cvt_f16_f32 v26, v36 // (op 11146) 
v_cvt_f16_f32 v27, v37 // (op 11146) 
v_cvt_f16_f32 v28, v38 // (op 11146) 
v_cvt_f16_f32 v29, v39 // (op 11146) 


v_pack_b32_f16 v20, v26, v27 // (op 11129) 
v_pack_b32_f16 v21, v28, v29 // (op 11129) 

s_lshl_b32 s52, s24, 5 // (op 11129) 
s_mul_i32 s53, 3, s52 // (op 11129) 
s_add_u32 s53, s53, 32 // (op 11129) 

v_add_u32 v23, v25, s53 // (op 11129) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11129) Store value





v_mov_b32 v20, v22 // (op 129) reg expression

buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 129) Load value

v_accvgpr_read v28, a0 // (op 125) 
v_accvgpr_read v29, a1 // (op 125) 
v_accvgpr_read v30, a2 // (op 125) 
v_accvgpr_read v31, a3 // (op 125) 
v_mul_f32 v32, v19, v28 // (op 125) 
v_mul_f32 v33, v19, v29 // (op 125) 
v_mul_f32 v34, v19, v30 // (op 125) 
v_mul_f32 v35, v19, v31 // (op 125) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 130) 
v_lshrrev_b32 v21, 16, v26 // (op 130) 
v_cvt_f32_f16 v21, v21 // (op 130) 
v_mul_f32 v28, v0, v20 // (op 130) 
v_mul_f32 v29, v0, v21 // (op 130) 
v_cvt_f32_f16 v20, v27 // (op 130) 
v_lshrrev_b32 v21, 16, v27 // (op 130) 
v_cvt_f32_f16 v21, v21 // (op 130) 
v_mul_f32 v30, v0, v20 // (op 130) 
v_mul_f32 v31, v0, v21 // (op 130) 


v_add_f32 v36, v28, v32 // (op 126) 
v_add_f32 v37, v29, v33 // (op 126) 
v_add_f32 v38, v30, v34 // (op 126) 
v_add_f32 v39, v31, v35 // (op 126) 


v_cvt_f16_f32 v26, v36 // (op 127) 
v_cvt_f16_f32 v27, v37 // (op 127) 
v_cvt_f16_f32 v28, v38 // (op 127) 
v_cvt_f16_f32 v29, v39 // (op 127) 


v_pack_b32_f16 v20, v26, v27 // (op 128) 
v_pack_b32_f16 v21, v28, v29 // (op 128) 

v_mov_b32 v23, v25 // (op 128) reg expression

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 128) Store value





s_lshl_b32 s52, s4, 5 // (op 11040) 
v_add_u32 v20, v22, s52 // (op 11040) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11040) Load value

v_accvgpr_read v28, a4 // (op 11036) 
v_accvgpr_read v29, a5 // (op 11036) 
v_accvgpr_read v30, a6 // (op 11036) 
v_accvgpr_read v31, a7 // (op 11036) 
v_mul_f32 v32, v19, v28 // (op 11036) 
v_mul_f32 v33, v19, v29 // (op 11036) 
v_mul_f32 v34, v19, v30 // (op 11036) 
v_mul_f32 v35, v19, v31 // (op 11036) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11041) 
v_lshrrev_b32 v21, 16, v26 // (op 11041) 
v_cvt_f32_f16 v21, v21 // (op 11041) 
v_mul_f32 v28, v0, v20 // (op 11041) 
v_mul_f32 v29, v0, v21 // (op 11041) 
v_cvt_f32_f16 v20, v27 // (op 11041) 
v_lshrrev_b32 v21, 16, v27 // (op 11041) 
v_cvt_f32_f16 v21, v21 // (op 11041) 
v_mul_f32 v30, v0, v20 // (op 11041) 
v_mul_f32 v31, v0, v21 // (op 11041) 


v_add_f32 v36, v28, v32 // (op 11037) 
v_add_f32 v37, v29, v33 // (op 11037) 
v_add_f32 v38, v30, v34 // (op 11037) 
v_add_f32 v39, v31, v35 // (op 11037) 


v_cvt_f16_f32 v26, v36 // (op 11038) 
v_cvt_f16_f32 v27, v37 // (op 11038) 
v_cvt_f16_f32 v28, v38 // (op 11038) 
v_cvt_f16_f32 v29, v39 // (op 11038) 


v_pack_b32_f16 v20, v26, v27 // (op 11039) 
v_pack_b32_f16 v21, v28, v29 // (op 11039) 

s_lshl_b32 s52, s24, 5 // (op 11039) 
v_add_u32 v23, v25, s52 // (op 11039) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11039) Store value





s_lshl_b32 s52, s4, 6 // (op 11054) 
v_add_u32 v20, v22, s52 // (op 11054) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11054) Load value

v_accvgpr_read v28, a8 // (op 11050) 
v_accvgpr_read v29, a9 // (op 11050) 
v_accvgpr_read v30, a10 // (op 11050) 
v_accvgpr_read v31, a11 // (op 11050) 
v_mul_f32 v32, v19, v28 // (op 11050) 
v_mul_f32 v33, v19, v29 // (op 11050) 
v_mul_f32 v34, v19, v30 // (op 11050) 
v_mul_f32 v35, v19, v31 // (op 11050) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11055) 
v_lshrrev_b32 v21, 16, v26 // (op 11055) 
v_cvt_f32_f16 v21, v21 // (op 11055) 
v_mul_f32 v28, v0, v20 // (op 11055) 
v_mul_f32 v29, v0, v21 // (op 11055) 
v_cvt_f32_f16 v20, v27 // (op 11055) 
v_lshrrev_b32 v21, 16, v27 // (op 11055) 
v_cvt_f32_f16 v21, v21 // (op 11055) 
v_mul_f32 v30, v0, v20 // (op 11055) 
v_mul_f32 v31, v0, v21 // (op 11055) 


v_add_f32 v36, v28, v32 // (op 11051) 
v_add_f32 v37, v29, v33 // (op 11051) 
v_add_f32 v38, v30, v34 // (op 11051) 
v_add_f32 v39, v31, v35 // (op 11051) 


v_cvt_f16_f32 v26, v36 // (op 11052) 
v_cvt_f16_f32 v27, v37 // (op 11052) 
v_cvt_f16_f32 v28, v38 // (op 11052) 
v_cvt_f16_f32 v29, v39 // (op 11052) 


v_pack_b32_f16 v20, v26, v27 // (op 11053) 
v_pack_b32_f16 v21, v28, v29 // (op 11053) 

s_lshl_b32 s52, s24, 6 // (op 11053) 
v_add_u32 v23, v25, s52 // (op 11053) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11053) Store value





s_lshl_b32 s52, s4, 5 // (op 11068) 
s_mul_i32 s53, 3, s52 // (op 11068) 

v_add_u32 v20, v22, s53 // (op 11068) 


buffer_load_dwordx2 v[26:27], v20, s[56:59], 0 offen // (op 11068) Load value

v_accvgpr_read v28, a12 // (op 11064) 
v_accvgpr_read v29, a13 // (op 11064) 
v_accvgpr_read v30, a14 // (op 11064) 
v_accvgpr_read v31, a15 // (op 11064) 
v_mul_f32 v32, v19, v28 // (op 11064) 
v_mul_f32 v33, v19, v29 // (op 11064) 
v_mul_f32 v34, v19, v30 // (op 11064) 
v_mul_f32 v35, v19, v31 // (op 11064) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v26 // (op 11069) 
v_lshrrev_b32 v21, 16, v26 // (op 11069) 
v_cvt_f32_f16 v21, v21 // (op 11069) 
v_mul_f32 v28, v0, v20 // (op 11069) 
v_mul_f32 v29, v0, v21 // (op 11069) 
v_cvt_f32_f16 v20, v27 // (op 11069) 
v_lshrrev_b32 v21, 16, v27 // (op 11069) 
v_cvt_f32_f16 v21, v21 // (op 11069) 
v_mul_f32 v30, v0, v20 // (op 11069) 
v_mul_f32 v31, v0, v21 // (op 11069) 


v_add_f32 v36, v28, v32 // (op 11065) 
v_add_f32 v37, v29, v33 // (op 11065) 
v_add_f32 v38, v30, v34 // (op 11065) 
v_add_f32 v39, v31, v35 // (op 11065) 


v_cvt_f16_f32 v26, v36 // (op 11066) 
v_cvt_f16_f32 v27, v37 // (op 11066) 
v_cvt_f16_f32 v28, v38 // (op 11066) 
v_cvt_f16_f32 v29, v39 // (op 11066) 


v_pack_b32_f16 v20, v26, v27 // (op 11067) 
v_pack_b32_f16 v21, v28, v29 // (op 11067) 

s_lshl_b32 s52, s24, 5 // (op 11067) 
s_mul_i32 s53, 3, s52 // (op 11067) 

v_add_u32 v23, v25, s53 // (op 11067) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 11067) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_116:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_350 // (op 350) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_350
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_350:
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_350:


s_mul_i32 s52, s8, s16 // (op 263) 
s_cmp_gt_u32 s9, s52 // (op 263) 
s_mov_b32 s53, scc // (op 263) 
s_sub_u32 s54, s9, s52 // (op 263) 

s_cmp_eq_i32 s53, 0 // (op 263) 
s_cselect_b32 s52, 0, s54 // (op 263) 


s_cmp_lt_u32 s8, s52 // (op 263) 
s_mov_b32 s53, scc // (op 263) 
s_cmp_eq_i32 s53, 0 // (op 263) 
s_cselect_b32 s54, s52, s8 // (op 263) 


s_cmp_lt_u32 s0, s54 // (op 263) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_263 // (op 263) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_263 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_263:

















s_mov_b32 s0, 0 // (op 404) call()
s_mov_b32 s1, 0 // (op 405) call()
s_mov_b32 s0, 0 // (op 419) call()
s_mul_i32 s15, s50, s16 // (op 418) 
s_cmp_gt_u32 s51, s15 // (op 418) 
s_mov_b32 s6, scc // (op 418) 
s_sub_u32 s7, s51, s15 // (op 418) 

s_cmp_eq_i32 s6, 0 // (op 418) 
s_cselect_b32 s8, 0, s7 // (op 418) 


s_cmp_lt_u32 s50, s8 // (op 418) 
s_mov_b32 s15, scc // (op 418) 
s_cmp_eq_i32 s15, 0 // (op 418) 
s_cselect_b32 s6, s8, s50 // (op 418) 


s_cmp_lt_u32 s0, s6 // (op 418) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_418 // (op 418) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_418 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_418:
v_lshrrev_b32 v3, 7, v1 // (op 21513) 
s_mul_i32 s15, s16, s50 // (op 21513) 
s_add_u32 s15, s15, s0 // (op 21513) 
v_and_b32 v4, 63, v1 // (op 21513) 
s_add_u32 s6, s15, s9 // (op 21513) 

v_lshrrev_b32 v13, 4, v4 // (op 21513) 
s_mul_hi_u32 s15, s6, s13 // (op 21513) 
v_lshlrev_b32 v14, 2, v13 // (op 21513) 

s_sub_u32 s7, s6, s15 // (op 21513) 
v_lshl_add_u32 v13, v3, 6, v14 // (op 21513) 


s_lshr_b32 s8, s7, 1 // (op 21513) 

v_lshrrev_b32 v3, 2, v4 // (op 21513) 

s_add_u32 s7, s8, s15 // (op 21513) 


v_and_b32 v4, 3, v3 // (op 21513) 

s_lshr_b32 s8, s7, s11 // (op 21513) 

v_and_b32 v3, 3, v1 // (op 21513) 
s_and_b32 s7, s11, 2147483648 // (op 21513) 
v_lshl_add_u32 v14, v4, 2, v3 // (op 21513) 


s_cmp_eq_i32 s7, 0 // (op 21513) 
s_mov_b32 s15, scc // (op 21513) 

v_lshrrev_b32 v3, 6, v1 // (op 21513) 
s_cmp_eq_i32 s15, 0 // (op 21513) 
s_cselect_b32 s7, s6, s8 // (op 21513) 



v_and_b32 v4, 1, v3 // (op 21513) 

s_mul_hi_u32 s6, s7, s12 // (op 21513) 
v_lshl_add_u32 v3, v4, 6, v14 // (op 21513) 


s_sub_u32 s8, s7, s6 // (op 21513) 
s_lshr_b32 s15, s8, 1 // (op 21513) 

s_add_u32 s8, s15, s6 // (op 21513) 


s_lshr_b32 s6, s8, s10 // (op 21513) 

s_and_b32 s8, s10, 2147483648 // (op 21513) 
s_cmp_eq_i32 s8, 0 // (op 21513) 
s_mov_b32 s15, scc // (op 21513) 

s_cmp_eq_i32 s15, 0 // (op 21513) 
s_cselect_b32 s8, s7, s6 // (op 21513) 


s_mul_i32 s6, s8, s14 // (op 21513) 
v_lshl_add_u32 v4, s8, 7, v13 // (op 21513) 


s_sub_u32 s8, s7, s6 // (op 21513) 


v_lshl_add_u32 v13, s8, 7, v3 // (op 21513) 


v_mul_lo_u32 v3, v13, s4 // (op 21513) 

v_add_lshl_u32 v13, v4, v3, 1 // (op 21513) 


s_mul_i32 s19, s4, s23 // (op 21515) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 21515) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 21515) least significant
s_mul_hi_u32 s6, s4, s22 // (op 21515) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21515) most significant: sum
s_add_u32 s19, s19, s6 // (op 21515) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 21515) 

s_mov_b32 s56, s20 // (op 21515) reg expression
s_mov_b32 s57, s21 // (op 21515) reg expression
s_mov_b32 s58, s28 // (op 21515) reg expression
s_mov_b32 s59, 131072 // (op 21515) call()

v_lshrrev_b32 v3, 7, v1 // (op 21715) 
s_mul_i32 s15, s16, s50 // (op 21715) 
s_add_u32 s15, s15, s0 // (op 21715) 
v_and_b32 v4, 63, v1 // (op 21715) 
s_add_u32 s6, s15, s9 // (op 21715) 

v_lshrrev_b32 v14, 4, v4 // (op 21715) 
s_mul_hi_u32 s15, s6, s13 // (op 21715) 
v_lshlrev_b32 v15, 2, v14 // (op 21715) 

s_sub_u32 s7, s6, s15 // (op 21715) 
v_lshl_add_u32 v14, v3, 6, v15 // (op 21715) 


s_lshr_b32 s8, s7, 1 // (op 21715) 

v_lshrrev_b32 v3, 2, v4 // (op 21715) 

s_add_u32 s7, s8, s15 // (op 21715) 


v_and_b32 v4, 3, v3 // (op 21715) 

s_lshr_b32 s8, s7, s11 // (op 21715) 

v_and_b32 v3, 3, v1 // (op 21715) 
s_and_b32 s7, s11, 2147483648 // (op 21715) 
v_lshl_add_u32 v15, v4, 2, v3 // (op 21715) 


s_cmp_eq_i32 s7, 0 // (op 21715) 
s_mov_b32 s15, scc // (op 21715) 

v_lshrrev_b32 v3, 6, v1 // (op 21715) 
s_cmp_eq_i32 s15, 0 // (op 21715) 
s_cselect_b32 s7, s6, s8 // (op 21715) 



v_and_b32 v4, 1, v3 // (op 21715) 

s_mul_hi_u32 s6, s7, s12 // (op 21715) 
v_lshl_add_u32 v3, v4, 6, v15 // (op 21715) 


s_sub_u32 s8, s7, s6 // (op 21715) 
s_lshr_b32 s15, s8, 1 // (op 21715) 

s_add_u32 s8, s15, s6 // (op 21715) 


s_lshr_b32 s6, s8, s10 // (op 21715) 

s_and_b32 s8, s10, 2147483648 // (op 21715) 
s_cmp_eq_i32 s8, 0 // (op 21715) 
s_mov_b32 s15, scc // (op 21715) 

s_cmp_eq_i32 s15, 0 // (op 21715) 
s_cselect_b32 s8, s7, s6 // (op 21715) 


s_mul_i32 s6, s8, s14 // (op 21715) 
v_lshl_add_u32 v4, s8, 7, v14 // (op 21715) 


s_sub_u32 s8, s7, s6 // (op 21715) 


v_lshl_add_u32 v14, s8, 7, v3 // (op 21715) 


v_mul_lo_u32 v3, v14, s24 // (op 21715) 

v_add_lshl_u32 v14, v4, v3, 1 // (op 21715) 


s_mul_i32 s19, s24, s23 // (op 21717) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 21717) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 21717) least significant
s_mul_hi_u32 s6, s24, s22 // (op 21717) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21717) most significant: sum
s_add_u32 s19, s19, s6 // (op 21717) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 21717) 

s_mov_b32 s60, s26 // (op 21717) reg expression
s_mov_b32 s61, s27 // (op 21717) reg expression
s_mov_b32 s62, s28 // (op 21717) reg expression
s_mov_b32 s63, 131072 // (op 21717) call()

v_lshrrev_b32 v3, 7, v1 // (op 21794) 
v_and_b32 v4, 15, v1 // (op 21794) 
v_lshl_add_u32 v15, v3, 6, v4 // (op 21794) 


v_and_b32 v3, 63, v1 // (op 21794) 
v_lshrrev_b32 v4, 4, v3 // (op 21794) 

v_lshlrev_b32 v3, 5, v4 // (op 21794) 

v_lshl_add_u32 v4, v15, 7, v3 // (op 21794) 


v_lshlrev_b32 v3, 2, v4 // (op 21794) 

v_lshrrev_b32 v4, 3, v3 // (op 21794) 

v_lshrrev_b32 v3, 6, v1 // (op 21869) 
v_and_b32 v15, 1, v3 // (op 21869) 

v_and_b32 v3, 15, v1 // (op 21869) 
v_lshl_add_u32 v16, v15, 6, v3 // (op 21869) 


v_and_b32 v3, 63, v1 // (op 21869) 
v_lshrrev_b32 v15, 4, v3 // (op 21869) 

v_lshlrev_b32 v3, 5, v15 // (op 21869) 

v_lshl_add_u32 v15, v16, 7, v3 // (op 21869) 


v_lshlrev_b32 v3, 2, v15 // (op 21869) 

v_lshrrev_b32 v15, 3, v3 // (op 21869) 

v_accvgpr_write a0, 0.00000 // (op 165) call()
v_accvgpr_write a1, 0.00000 // (op 165) call()
v_accvgpr_write a2, 0.00000 // (op 165) call()
v_accvgpr_write a3, 0.00000 // (op 165) call()
v_accvgpr_write a4, 0.00000 // (op 657) call()
v_accvgpr_write a5, 0.00000 // (op 657) call()
v_accvgpr_write a6, 0.00000 // (op 657) call()
v_accvgpr_write a7, 0.00000 // (op 657) call()
v_accvgpr_write a8, 0.00000 // (op 883) call()
v_accvgpr_write a9, 0.00000 // (op 883) call()
v_accvgpr_write a10, 0.00000 // (op 883) call()
v_accvgpr_write a11, 0.00000 // (op 883) call()
v_accvgpr_write a12, 0.00000 // (op 1109) call()
v_accvgpr_write a13, 0.00000 // (op 1109) call()
v_accvgpr_write a14, 0.00000 // (op 1109) call()
v_accvgpr_write a15, 0.00000 // (op 1109) call()
v_accvgpr_write a16, 0.00000 // (op 1471) call()
v_accvgpr_write a17, 0.00000 // (op 1471) call()
v_accvgpr_write a18, 0.00000 // (op 1471) call()
v_accvgpr_write a19, 0.00000 // (op 1471) call()
v_accvgpr_write a20, 0.00000 // (op 1884) call()
v_accvgpr_write a21, 0.00000 // (op 1884) call()
v_accvgpr_write a22, 0.00000 // (op 1884) call()
v_accvgpr_write a23, 0.00000 // (op 1884) call()
v_accvgpr_write a24, 0.00000 // (op 1885) call()
v_accvgpr_write a25, 0.00000 // (op 1885) call()
v_accvgpr_write a26, 0.00000 // (op 1885) call()
v_accvgpr_write a27, 0.00000 // (op 1885) call()
v_accvgpr_write a28, 0.00000 // (op 1886) call()
v_accvgpr_write a29, 0.00000 // (op 1886) call()
v_accvgpr_write a30, 0.00000 // (op 1886) call()
v_accvgpr_write a31, 0.00000 // (op 1886) call()
v_accvgpr_write a32, 0.00000 // (op 2473) call()
v_accvgpr_write a33, 0.00000 // (op 2473) call()
v_accvgpr_write a34, 0.00000 // (op 2473) call()
v_accvgpr_write a35, 0.00000 // (op 2473) call()
v_accvgpr_write a36, 0.00000 // (op 2886) call()
v_accvgpr_write a37, 0.00000 // (op 2886) call()
v_accvgpr_write a38, 0.00000 // (op 2886) call()
v_accvgpr_write a39, 0.00000 // (op 2886) call()
v_accvgpr_write a40, 0.00000 // (op 2887) call()
v_accvgpr_write a41, 0.00000 // (op 2887) call()
v_accvgpr_write a42, 0.00000 // (op 2887) call()
v_accvgpr_write a43, 0.00000 // (op 2887) call()
v_accvgpr_write a44, 0.00000 // (op 2888) call()
v_accvgpr_write a45, 0.00000 // (op 2888) call()
v_accvgpr_write a46, 0.00000 // (op 2888) call()
v_accvgpr_write a47, 0.00000 // (op 2888) call()
v_accvgpr_write a48, 0.00000 // (op 3475) call()
v_accvgpr_write a49, 0.00000 // (op 3475) call()
v_accvgpr_write a50, 0.00000 // (op 3475) call()
v_accvgpr_write a51, 0.00000 // (op 3475) call()
v_accvgpr_write a52, 0.00000 // (op 3888) call()
v_accvgpr_write a53, 0.00000 // (op 3888) call()
v_accvgpr_write a54, 0.00000 // (op 3888) call()
v_accvgpr_write a55, 0.00000 // (op 3888) call()
v_accvgpr_write a56, 0.00000 // (op 3889) call()
v_accvgpr_write a57, 0.00000 // (op 3889) call()
v_accvgpr_write a58, 0.00000 // (op 3889) call()
v_accvgpr_write a59, 0.00000 // (op 3889) call()
v_accvgpr_write a60, 0.00000 // (op 3890) call()
v_accvgpr_write a61, 0.00000 // (op 3890) call()
v_accvgpr_write a62, 0.00000 // (op 3890) call()
v_accvgpr_write a63, 0.00000 // (op 3890) call()
v_lshrrev_b32 v3, 2, v1 // (op 21249) 
s_mul_i32 s15, s16, s50 // (op 21249) 
s_add_u32 s15, s15, s0 // (op 21249) 
v_and_b32 v16, -4, v1 // (op 21249) 
s_add_u32 s6, s15, s9 // (op 21249) 

v_sub_u32 v17, v1, v16 // (op 21249) 

s_mul_hi_u32 s15, s6, s13 // (op 21249) 
v_lshlrev_b32 v16, 5, v17 // (op 21249) 

s_sub_u32 s7, s6, s15 // (op 21249) 
s_lshr_b32 s8, s7, 1 // (op 21249) 

s_add_u32 s7, s8, s15 // (op 21249) 


s_lshr_b32 s8, s7, s11 // (op 21249) 

s_and_b32 s7, s11, 2147483648 // (op 21249) 
s_cmp_eq_i32 s7, 0 // (op 21249) 
s_mov_b32 s15, scc // (op 21249) 

s_cmp_eq_i32 s15, 0 // (op 21249) 
s_cselect_b32 s7, s6, s8 // (op 21249) 


s_mul_hi_u32 s8, s7, s12 // (op 21249) 
s_sub_u32 s15, s7, s8 // (op 21249) 
s_lshr_b32 s18, s15, 1 // (op 21249) 

s_add_u32 s15, s18, s8 // (op 21249) 


s_lshr_b32 s8, s15, s10 // (op 21249) 

s_and_b32 s15, s10, 2147483648 // (op 21249) 
s_cmp_eq_i32 s15, 0 // (op 21249) 
s_mov_b32 s18, scc // (op 21249) 

s_cmp_eq_i32 s18, 0 // (op 21249) 
s_cselect_b32 s15, s7, s8 // (op 21249) 


v_lshl_add_u32 v17, s15, 7, v3 // (op 21249) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 21249) 
s_mul_i32 s8, s7, s18 // (op 21249) 


s_sub_u32 s7, s6, s8 // (op 21249) 


v_lshl_add_u32 v3, s7, 7, v16 // (op 21249) 


v_mov_b32 v21, 0 // (op 21249) convert
v_mov_b32 v20, v3 // (op 21249) convert

v_mul_lo_u32 v23, v17, s31 // (op 21249) most significant: low of low * high
v_mul_hi_u32 v3, v17, s30 // (op 21249) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 21249) least significant: low of low * low
v_add_u32 v23, v23, v3 // (op 21249) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v20 // (op 21249) least significant half
v_addc_co_u32 v23, s[18:19], v23, v21, s[18:19] // (op 21249) most significant half



v_lshlrev_b64 v[20:21], 2, v[22:23] // (op 21249) 

v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 21249) 

s_mul_i32 s19, s30, s35 // (op 21251) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 21251) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 21251) least significant
s_mul_hi_u32 s6, s30, s34 // (op 21251) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21251) most significant: sum
s_add_u32 s19, s19, s6 // (op 21251) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21251) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21251) 

s_mov_b32 s68, s32 // (op 21251) reg expression
s_mov_b32 s69, s33 // (op 21251) reg expression
s_mov_b32 s70, s18 // (op 21251) reg expression
s_mov_b32 s71, 131072 // (op 21251) call()

v_lshrrev_b32 v3, 2, v1 // (op 21268) 
s_mul_i32 s15, s16, s50 // (op 21268) 
s_add_u32 s15, s15, s0 // (op 21268) 
v_and_b32 v18, -4, v1 // (op 21268) 
s_add_u32 s6, s15, s9 // (op 21268) 

v_sub_u32 v20, v1, v18 // (op 21268) 

s_mul_hi_u32 s15, s6, s13 // (op 21268) 
v_lshlrev_b32 v18, 5, v20 // (op 21268) 

s_sub_u32 s7, s6, s15 // (op 21268) 
s_lshr_b32 s8, s7, 1 // (op 21268) 

s_add_u32 s7, s8, s15 // (op 21268) 


s_lshr_b32 s8, s7, s11 // (op 21268) 

s_and_b32 s7, s11, 2147483648 // (op 21268) 
s_cmp_eq_i32 s7, 0 // (op 21268) 
s_mov_b32 s15, scc // (op 21268) 

s_cmp_eq_i32 s15, 0 // (op 21268) 
s_cselect_b32 s7, s6, s8 // (op 21268) 


s_mul_hi_u32 s8, s7, s12 // (op 21268) 
s_sub_u32 s15, s7, s8 // (op 21268) 
s_lshr_b32 s18, s15, 1 // (op 21268) 

s_add_u32 s15, s18, s8 // (op 21268) 


s_lshr_b32 s8, s15, s10 // (op 21268) 

s_and_b32 s15, s10, 2147483648 // (op 21268) 
s_cmp_eq_i32 s15, 0 // (op 21268) 
s_mov_b32 s18, scc // (op 21268) 

s_cmp_eq_i32 s18, 0 // (op 21268) 
s_cselect_b32 s15, s7, s8 // (op 21268) 


v_lshl_add_u32 v20, s15, 7, v3 // (op 21268) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 21268) 
s_mul_i32 s8, s7, s18 // (op 21268) 


s_sub_u32 s7, s6, s8 // (op 21268) 


v_lshl_add_u32 v3, s7, 7, v18 // (op 21268) 


v_mov_b32 v23, 0 // (op 21268) convert
v_mov_b32 v22, v3 // (op 21268) convert

v_mul_lo_u32 v25, v20, s31 // (op 21268) most significant: low of low * high
v_mul_hi_u32 v3, v20, s30 // (op 21268) most significant: high of low * low
v_mul_lo_u32 v24, v20, s30 // (op 21268) least significant: low of low * low
v_add_u32 v25, v25, v3 // (op 21268) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v22 // (op 21268) least significant half
v_addc_co_u32 v25, s[18:19], v25, v23, s[18:19] // (op 21268) most significant half



v_lshlrev_b64 v[20:21], 2, v[24:25] // (op 21268) 

v_lshrrev_b64 v[22:23], 3, v[20:21] // (op 21268) 

s_mul_i32 s19, s30, s35 // (op 21270) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 21270) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 21270) least significant
s_mul_hi_u32 s6, s30, s34 // (op 21270) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21270) most significant: sum
s_add_u32 s19, s19, s6 // (op 21270) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21270) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21270) 

s_mov_b32 s68, s32 // (op 21270) reg expression
s_mov_b32 s69, s33 // (op 21270) reg expression
s_mov_b32 s70, s18 // (op 21270) reg expression
s_mov_b32 s71, 131072 // (op 21270) call()

v_lshrrev_b32 v3, 2, v1 // (op 21395) 
s_mul_i32 s15, s16, s50 // (op 21395) 
s_add_u32 s15, s15, s0 // (op 21395) 
v_and_b32 v18, -4, v1 // (op 21395) 
s_add_u32 s6, s15, s9 // (op 21395) 

v_sub_u32 v20, v1, v18 // (op 21395) 

s_mul_hi_u32 s15, s6, s13 // (op 21395) 
v_lshlrev_b32 v18, 5, v20 // (op 21395) 

s_sub_u32 s7, s6, s15 // (op 21395) 
s_lshr_b32 s8, s7, 1 // (op 21395) 

s_add_u32 s7, s8, s15 // (op 21395) 


s_lshr_b32 s8, s7, s11 // (op 21395) 

s_and_b32 s7, s11, 2147483648 // (op 21395) 
s_cmp_eq_i32 s7, 0 // (op 21395) 
s_mov_b32 s15, scc // (op 21395) 

s_cmp_eq_i32 s15, 0 // (op 21395) 
s_cselect_b32 s7, s6, s8 // (op 21395) 


s_mul_hi_u32 s8, s7, s12 // (op 21395) 
s_sub_u32 s15, s7, s8 // (op 21395) 
s_lshr_b32 s18, s15, 1 // (op 21395) 

s_add_u32 s15, s18, s8 // (op 21395) 


s_lshr_b32 s8, s15, s10 // (op 21395) 

s_and_b32 s15, s10, 2147483648 // (op 21395) 
s_cmp_eq_i32 s15, 0 // (op 21395) 
s_mov_b32 s18, scc // (op 21395) 

s_cmp_eq_i32 s18, 0 // (op 21395) 
s_cselect_b32 s15, s7, s8 // (op 21395) 


s_mul_i32 s8, s15, s14 // (op 21395) 

s_sub_u32 s15, s7, s8 // (op 21395) 

v_lshl_add_u32 v20, s15, 7, v3 // (op 21395) 


v_mul_lo_u32 v25, v20, s37 // (op 21395) most significant: low of low * high
v_mul_hi_u32 v3, v20, s36 // (op 21395) most significant: high of low * low
v_mul_lo_u32 v24, v20, s36 // (op 21395) least significant: low of low * low
v_add_u32 v25, v25, v3 // (op 21395) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 21395) 
s_mul_i32 s8, s7, s18 // (op 21395) 


s_sub_u32 s7, s6, s8 // (op 21395) 


v_lshl_add_u32 v3, s7, 7, v18 // (op 21395) 


v_mov_b32 v21, 0 // (op 21395) convert
v_mov_b32 v20, v3 // (op 21395) convert

v_add_co_u32 v26, s[18:19], v20, v24 // (op 21395) least significant half
v_addc_co_u32 v27, s[18:19], v21, v25, s[18:19] // (op 21395) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 21395) 


v_lshrrev_b64 v[20:21], 3, v[26:27] // (op 21395) 

s_mul_i32 s19, s36, s23 // (op 21397) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 21397) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 21397) least significant
s_mul_hi_u32 s6, s36, s22 // (op 21397) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21397) most significant: sum
s_add_u32 s19, s19, s6 // (op 21397) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21397) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21397) 

s_mov_b32 s72, s38 // (op 21397) reg expression
s_mov_b32 s73, s39 // (op 21397) reg expression
s_mov_b32 s74, s18 // (op 21397) reg expression
s_mov_b32 s75, 131072 // (op 21397) call()

v_lshrrev_b32 v3, 2, v1 // (op 21414) 
s_mul_i32 s15, s16, s50 // (op 21414) 
s_add_u32 s15, s15, s0 // (op 21414) 
v_and_b32 v18, -4, v1 // (op 21414) 
s_add_u32 s6, s15, s9 // (op 21414) 

v_sub_u32 v24, v1, v18 // (op 21414) 

s_mul_hi_u32 s15, s6, s13 // (op 21414) 
v_lshlrev_b32 v18, 5, v24 // (op 21414) 

s_sub_u32 s7, s6, s15 // (op 21414) 
s_lshr_b32 s8, s7, 1 // (op 21414) 

s_add_u32 s7, s8, s15 // (op 21414) 


s_lshr_b32 s8, s7, s11 // (op 21414) 

s_and_b32 s7, s11, 2147483648 // (op 21414) 
s_cmp_eq_i32 s7, 0 // (op 21414) 
s_mov_b32 s15, scc // (op 21414) 

s_cmp_eq_i32 s15, 0 // (op 21414) 
s_cselect_b32 s7, s6, s8 // (op 21414) 


s_mul_hi_u32 s8, s7, s12 // (op 21414) 
s_sub_u32 s15, s7, s8 // (op 21414) 
s_lshr_b32 s18, s15, 1 // (op 21414) 

s_add_u32 s15, s18, s8 // (op 21414) 


s_lshr_b32 s8, s15, s10 // (op 21414) 

s_and_b32 s15, s10, 2147483648 // (op 21414) 
s_cmp_eq_i32 s15, 0 // (op 21414) 
s_mov_b32 s18, scc // (op 21414) 

s_cmp_eq_i32 s18, 0 // (op 21414) 
s_cselect_b32 s15, s7, s8 // (op 21414) 


s_mul_i32 s8, s15, s14 // (op 21414) 

s_sub_u32 s15, s7, s8 // (op 21414) 

v_lshl_add_u32 v24, s15, 7, v3 // (op 21414) 


v_mul_lo_u32 v27, v24, s37 // (op 21414) most significant: low of low * high
v_mul_hi_u32 v3, v24, s36 // (op 21414) most significant: high of low * low
v_mul_lo_u32 v26, v24, s36 // (op 21414) least significant: low of low * low
v_add_u32 v27, v27, v3 // (op 21414) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 21414) 
s_mul_i32 s8, s7, s18 // (op 21414) 


s_sub_u32 s7, s6, s8 // (op 21414) 


v_lshl_add_u32 v3, s7, 7, v18 // (op 21414) 


v_mov_b32 v25, 0 // (op 21414) convert
v_mov_b32 v24, v3 // (op 21414) convert

v_add_co_u32 v28, s[18:19], v24, v26 // (op 21414) least significant half
v_addc_co_u32 v29, s[18:19], v25, v27, s[18:19] // (op 21414) most significant half

v_lshlrev_b64 v[28:29], 2, v[28:29] // (op 21414) 


v_lshrrev_b64 v[24:25], 3, v[28:29] // (op 21414) 

s_mul_i32 s19, s36, s23 // (op 21416) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 21416) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 21416) least significant
s_mul_hi_u32 s6, s36, s22 // (op 21416) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21416) most significant: sum
s_add_u32 s19, s19, s6 // (op 21416) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21416) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21416) 

s_mov_b32 s72, s38 // (op 21416) reg expression
s_mov_b32 s73, s39 // (op 21416) reg expression
s_mov_b32 s74, s18 // (op 21416) reg expression
s_mov_b32 s75, 131072 // (op 21416) call()

v_and_b32 v3, 63, v1 // (op 22472) 
s_mul_i32 s15, s16, s50 // (op 22472) 
s_add_u32 s15, s15, s0 // (op 22472) 
v_lshrrev_b32 v18, 4, v3 // (op 22472) 
s_add_u32 s6, s15, s9 // (op 22472) 

v_and_b32 v26, 3, v18 // (op 22472) 

s_mul_hi_u32 s15, s6, s13 // (op 22472) 
v_and_b32 v18, 15, v1 // (op 22472) 
s_sub_u32 s7, s6, s15 // (op 22472) 
v_lshl_add_u32 v27, v26, 4, v18 // (op 22472) 


s_lshr_b32 s8, s7, 1 // (op 22472) 

v_lshrrev_b32 v18, 7, v1 // (op 22472) 
s_add_u32 s7, s8, s15 // (op 22472) 


v_lshl_add_u32 v26, v18, 6, v27 // (op 22472) 


s_lshr_b32 s8, s7, s11 // (op 22472) 

v_lshrrev_b32 v18, 6, v3 // (op 22472) 

s_and_b32 s7, s11, 2147483648 // (op 22472) 
v_lshlrev_b32 v3, 2, v18 // (op 22472) 

s_cmp_eq_i32 s7, 0 // (op 22472) 
s_mov_b32 s15, scc // (op 22472) 

s_cmp_eq_i32 s15, 0 // (op 22472) 
s_cselect_b32 s7, s6, s8 // (op 22472) 


s_mul_hi_u32 s8, s7, s12 // (op 22472) 
s_sub_u32 s15, s7, s8 // (op 22472) 
s_lshr_b32 s18, s15, 1 // (op 22472) 

s_add_u32 s15, s18, s8 // (op 22472) 


s_lshr_b32 s8, s15, s10 // (op 22472) 

s_and_b32 s15, s10, 2147483648 // (op 22472) 
s_cmp_eq_i32 s15, 0 // (op 22472) 
s_mov_b32 s18, scc // (op 22472) 

s_cmp_eq_i32 s18, 0 // (op 22472) 
s_cselect_b32 s15, s7, s8 // (op 22472) 


v_lshl_add_u32 v18, s15, 7, v26 // (op 22472) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 22472) 
s_mul_i32 s8, s7, s18 // (op 22472) 


s_sub_u32 s7, s6, s8 // (op 22472) 


v_lshl_add_u32 v26, s7, 2, v3 // (op 22472) 


v_mul_lo_u32 v3, v18, s40 // (op 22472) 
v_add_u32 v3, v3, v26 // (op 22472) 


s_mul_i32 s19, s40, s35 // (op 22474) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 22474) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 22474) least significant
s_mul_hi_u32 s6, s40, s34 // (op 22474) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 22474) most significant: sum
s_add_u32 s19, s19, s6 // (op 22474) most significant: sum


s_mov_b32 s52, s42 // (op 22474) reg expression
s_mov_b32 s53, s43 // (op 22474) reg expression
s_mov_b32 s54, s18 // (op 22474) reg expression
s_mov_b32 s55, 131072 // (op 22474) call()

v_and_b32 v18, 63, v1 // (op 22493) 
s_mul_i32 s15, s16, s50 // (op 22493) 
s_add_u32 s15, s15, s0 // (op 22493) 
v_lshrrev_b32 v26, 4, v18 // (op 22493) 
s_add_u32 s6, s15, s9 // (op 22493) 

v_and_b32 v27, 3, v26 // (op 22493) 

s_mul_hi_u32 s15, s6, s13 // (op 22493) 
v_and_b32 v26, 15, v1 // (op 22493) 
s_sub_u32 s7, s6, s15 // (op 22493) 
v_lshl_add_u32 v28, v27, 4, v26 // (op 22493) 


s_lshr_b32 s8, s7, 1 // (op 22493) 

v_lshrrev_b32 v26, 7, v1 // (op 22493) 
s_add_u32 s7, s8, s15 // (op 22493) 


v_lshl_add_u32 v27, v26, 6, v28 // (op 22493) 


s_lshr_b32 s8, s7, s11 // (op 22493) 

v_lshrrev_b32 v26, 6, v18 // (op 22493) 

s_and_b32 s7, s11, 2147483648 // (op 22493) 
v_lshlrev_b32 v18, 2, v26 // (op 22493) 

s_cmp_eq_i32 s7, 0 // (op 22493) 
s_mov_b32 s15, scc // (op 22493) 

s_cmp_eq_i32 s15, 0 // (op 22493) 
s_cselect_b32 s7, s6, s8 // (op 22493) 


s_mul_hi_u32 s8, s7, s12 // (op 22493) 
s_sub_u32 s15, s7, s8 // (op 22493) 
s_lshr_b32 s18, s15, 1 // (op 22493) 

s_add_u32 s15, s18, s8 // (op 22493) 


s_lshr_b32 s8, s15, s10 // (op 22493) 

s_and_b32 s15, s10, 2147483648 // (op 22493) 
s_cmp_eq_i32 s15, 0 // (op 22493) 
s_mov_b32 s18, scc // (op 22493) 

s_cmp_eq_i32 s18, 0 // (op 22493) 
s_cselect_b32 s15, s7, s8 // (op 22493) 


v_lshl_add_u32 v26, s15, 7, v27 // (op 22493) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 22493) 
s_mul_i32 s8, s7, s18 // (op 22493) 


s_sub_u32 s7, s6, s8 // (op 22493) 


v_lshl_add_u32 v27, s7, 2, v18 // (op 22493) 


v_mul_lo_u32 v18, v26, s40 // (op 22493) 
v_add_u32 v18, v18, v27 // (op 22493) 


s_mul_i32 s19, s40, s35 // (op 22495) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 22495) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 22495) least significant
s_mul_hi_u32 s6, s40, s34 // (op 22495) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 22495) most significant: sum
s_add_u32 s19, s19, s6 // (op 22495) most significant: sum


s_mov_b32 s52, s42 // (op 22495) reg expression
s_mov_b32 s53, s43 // (op 22495) reg expression
s_mov_b32 s54, s18 // (op 22495) reg expression
s_mov_b32 s55, 131072 // (op 22495) call()

v_and_b32 v26, 63, v1 // (op 22524) 
s_mul_i32 s15, s16, s50 // (op 22524) 
s_add_u32 s15, s15, s0 // (op 22524) 
v_lshrrev_b32 v27, 4, v26 // (op 22524) 
s_add_u32 s6, s15, s9 // (op 22524) 

v_and_b32 v28, 3, v27 // (op 22524) 

s_mul_hi_u32 s15, s6, s13 // (op 22524) 
v_and_b32 v27, 15, v1 // (op 22524) 
s_sub_u32 s7, s6, s15 // (op 22524) 
v_lshl_add_u32 v29, v28, 4, v27 // (op 22524) 


s_lshr_b32 s8, s7, 1 // (op 22524) 

v_lshrrev_b32 v27, 6, v1 // (op 22524) 
s_add_u32 s7, s8, s15 // (op 22524) 


v_and_b32 v28, 1, v27 // (op 22524) 

s_lshr_b32 s8, s7, s11 // (op 22524) 

v_lshl_add_u32 v27, v28, 6, v29 // (op 22524) 


s_and_b32 s7, s11, 2147483648 // (op 22524) 
v_lshrrev_b32 v28, 6, v26 // (op 22524) 

s_cmp_eq_i32 s7, 0 // (op 22524) 
s_mov_b32 s15, scc // (op 22524) 

v_lshlrev_b32 v26, 2, v28 // (op 22524) 

s_cmp_eq_i32 s15, 0 // (op 22524) 
s_cselect_b32 s7, s6, s8 // (op 22524) 


s_mul_hi_u32 s8, s7, s12 // (op 22524) 
s_sub_u32 s15, s7, s8 // (op 22524) 
s_lshr_b32 s18, s15, 1 // (op 22524) 

s_add_u32 s15, s18, s8 // (op 22524) 


s_lshr_b32 s8, s15, s10 // (op 22524) 

s_and_b32 s15, s10, 2147483648 // (op 22524) 
s_cmp_eq_i32 s15, 0 // (op 22524) 
s_mov_b32 s18, scc // (op 22524) 

s_cmp_eq_i32 s18, 0 // (op 22524) 
s_cselect_b32 s15, s7, s8 // (op 22524) 


s_mul_i32 s8, s15, s14 // (op 22524) 

s_sub_u32 s15, s7, s8 // (op 22524) 

v_lshl_add_u32 v28, s15, 7, v27 // (op 22524) 


v_mul_lo_u32 v27, v28, s44 // (op 22524) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 22524) 
s_mul_i32 s8, s7, s18 // (op 22524) 


s_sub_u32 s7, s6, s8 // (op 22524) 


v_lshl_add_u32 v28, s7, 2, v26 // (op 22524) 


v_add_u32 v26, v28, v27 // (op 22524) 


s_mul_i32 s19, s44, s23 // (op 22526) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 22526) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 22526) least significant
s_mul_hi_u32 s6, s44, s22 // (op 22526) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 22526) most significant: sum
s_add_u32 s19, s19, s6 // (op 22526) most significant: sum


s_mov_b32 s64, s46 // (op 22526) reg expression
s_mov_b32 s65, s47 // (op 22526) reg expression
s_mov_b32 s66, s18 // (op 22526) reg expression
s_mov_b32 s67, 131072 // (op 22526) call()

v_and_b32 v27, 63, v1 // (op 22545) 
s_mul_i32 s15, s16, s50 // (op 22545) 
s_add_u32 s15, s15, s0 // (op 22545) 
v_lshrrev_b32 v28, 4, v27 // (op 22545) 
s_add_u32 s6, s15, s9 // (op 22545) 

v_and_b32 v29, 3, v28 // (op 22545) 

s_mul_hi_u32 s15, s6, s13 // (op 22545) 
v_and_b32 v28, 15, v1 // (op 22545) 
s_sub_u32 s7, s6, s15 // (op 22545) 
v_lshl_add_u32 v30, v29, 4, v28 // (op 22545) 


s_lshr_b32 s8, s7, 1 // (op 22545) 

v_lshrrev_b32 v28, 6, v1 // (op 22545) 
s_add_u32 s7, s8, s15 // (op 22545) 


v_and_b32 v29, 1, v28 // (op 22545) 

s_lshr_b32 s8, s7, s11 // (op 22545) 

v_lshl_add_u32 v28, v29, 6, v30 // (op 22545) 


s_and_b32 s7, s11, 2147483648 // (op 22545) 
v_lshrrev_b32 v29, 6, v27 // (op 22545) 

s_cmp_eq_i32 s7, 0 // (op 22545) 
s_mov_b32 s15, scc // (op 22545) 

v_lshlrev_b32 v27, 2, v29 // (op 22545) 

s_cmp_eq_i32 s15, 0 // (op 22545) 
s_cselect_b32 s7, s6, s8 // (op 22545) 


s_mul_hi_u32 s8, s7, s12 // (op 22545) 
s_sub_u32 s15, s7, s8 // (op 22545) 
s_lshr_b32 s18, s15, 1 // (op 22545) 

s_add_u32 s15, s18, s8 // (op 22545) 


s_lshr_b32 s8, s15, s10 // (op 22545) 

s_and_b32 s15, s10, 2147483648 // (op 22545) 
s_cmp_eq_i32 s15, 0 // (op 22545) 
s_mov_b32 s18, scc // (op 22545) 

s_cmp_eq_i32 s18, 0 // (op 22545) 
s_cselect_b32 s15, s7, s8 // (op 22545) 


s_mul_i32 s8, s15, s14 // (op 22545) 

s_sub_u32 s15, s7, s8 // (op 22545) 

v_lshl_add_u32 v29, s15, 7, v28 // (op 22545) 


v_mul_lo_u32 v28, v29, s44 // (op 22545) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 22545) 
s_mul_i32 s8, s7, s18 // (op 22545) 


s_sub_u32 s7, s6, s8 // (op 22545) 


v_lshl_add_u32 v29, s7, 2, v27 // (op 22545) 


v_add_u32 v27, v29, v28 // (op 22545) 


s_mul_i32 s19, s44, s23 // (op 22547) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 22547) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 22547) least significant
s_mul_hi_u32 s6, s44, s22 // (op 22547) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 22547) most significant: sum
s_add_u32 s19, s19, s6 // (op 22547) most significant: sum


s_mov_b32 s64, s46 // (op 22547) reg expression
s_mov_b32 s65, s47 // (op 22547) reg expression
s_mov_b32 s66, s18 // (op 22547) reg expression
s_mov_b32 s67, 131072 // (op 22547) call()

v_and_b32 v28, -4, v1 // (op 21835) 
v_sub_u32 v29, v1, v28 // (op 21835) 

v_lshlrev_b32 v28, 5, v29 // (op 21835) 

v_lshrrev_b32 v29, 2, v1 // (op 21835) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 21835) 


v_lshlrev_b32 v28, 2, v30 // (op 21835) 

v_lshrrev_b32 v29, 3, v28 // (op 21835) 

s_nop 0
v_readfirstlane_b32 s15, v29 // (op 21835) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23186) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 21129) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21129) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21129) 


v_mov_b64 v[30:31], v[20:21] // (op 21129) 
s_mov_b32 m0, s15 // (op 21129) reg expression
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 21129) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 21129) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 21129) most significant half

s_add_u32 m0, m0, 4096 // (op 21129) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 21129) Load value direct to lds



v_and_b32 v28, -4, v1 // (op 21760) 
v_sub_u32 v29, v1, v28 // (op 21760) 

v_lshlrev_b32 v28, 5, v29 // (op 21760) 

v_lshrrev_b32 v29, 2, v1 // (op 21760) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 21760) 


v_lshlrev_b32 v28, 2, v30 // (op 21760) 

v_lshrrev_b32 v29, 3, v28 // (op 21760) 

s_nop 0
v_readfirstlane_b32 s15, v29 // (op 21760) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23166) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 21134) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21134) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21134) 


v_mov_b64 v[30:31], v[16:17] // (op 21134) 
s_add_u32 s6, 8192, s15 // (op 21134) 
s_mov_b32 m0, s6 // (op 21134) reg expression
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 21134) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 21134) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 21134) most significant half

s_add_u32 m0, m0, 4096 // (op 21134) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 21134) Load value direct to lds






v_mov_b32 v28, v26 // (op 178) reg expression

buffer_load_dwordx4 v[30:33], v28, s[64:67], 0 offen // (op 178) Load value




v_mov_b32 v26, v3 // (op 176) reg expression

buffer_load_dwordx4 v[34:37], v26, s[52:55], 0 offen // (op 176) Load value


v_and_b32 v3, -4, v1 // (op 21993) 
v_sub_u32 v26, v1, v3 // (op 21993) 

v_lshlrev_b32 v3, 5, v26 // (op 21993) 

v_lshrrev_b32 v26, 2, v1 // (op 21993) 
v_lshl_add_u32 v28, v26, 7, v3 // (op 21993) 


v_lshlrev_b32 v3, 2, v28 // (op 21993) 

v_lshrrev_b32 v26, 3, v3 // (op 21993) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 21993) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23216) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 21139) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21139) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21139) 


v_mov_b64 v[38:39], v[20:21] // (op 21139) 
s_add_u32 s6, 16384, s15 // (op 21139) 
s_mov_b32 m0, s6 // (op 21139) reg expression
buffer_load_dwordx4 v38, s[72:75], 64 offen offset: 0 lds // (op 21139) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v28 // (op 21139) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 21139) most significant half

s_add_u32 m0, m0, 4096 // (op 21139) 
buffer_load_dwordx4 v38, s[72:75], 64 offen offset: 0 lds // (op 21139) Load value direct to lds




v_and_b32 v3, -4, v1 // (op 21949) 
v_sub_u32 v26, v1, v3 // (op 21949) 

v_lshlrev_b32 v3, 5, v26 // (op 21949) 

v_lshrrev_b32 v26, 2, v1 // (op 21949) 
v_lshl_add_u32 v28, v26, 7, v3 // (op 21949) 


v_lshlrev_b32 v3, 2, v28 // (op 21949) 

v_lshrrev_b32 v26, 3, v3 // (op 21949) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 21949) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23206) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 21141) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21141) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21141) 


v_mov_b64 v[38:39], v[16:17] // (op 21141) 
s_add_u32 s6, 24576, s15 // (op 21141) 
s_mov_b32 m0, s6 // (op 21141) reg expression
buffer_load_dwordx4 v38, s[68:71], 64 offen offset: 0 lds // (op 21141) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v28 // (op 21141) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 21141) most significant half

s_add_u32 m0, m0, 4096 // (op 21141) 
buffer_load_dwordx4 v38, s[68:71], 64 offen offset: 0 lds // (op 21141) Load value direct to lds




v_and_b32 v3, -4, v1 // (op 22081) 
v_sub_u32 v26, v1, v3 // (op 22081) 

v_lshlrev_b32 v3, 5, v26 // (op 22081) 

v_lshrrev_b32 v26, 2, v1 // (op 22081) 
v_lshl_add_u32 v28, v26, 7, v3 // (op 22081) 


v_lshlrev_b32 v3, 2, v28 // (op 22081) 

v_lshrrev_b32 v26, 3, v3 // (op 22081) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 22081) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23236) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 21143) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21143) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21143) 


v_mov_b64 v[38:39], v[20:21] // (op 21143) 
s_add_u32 s6, 32768, s15 // (op 21143) 
s_mov_b32 m0, s6 // (op 21143) reg expression
s_mov_b32 s7, 128 // (op 21143) call()
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 21143) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v28 // (op 21143) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 21143) most significant half

s_add_u32 m0, m0, 4096 // (op 21143) 
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 21143) Load value direct to lds





v_and_b32 v3, -4, v1 // (op 22037) 
v_sub_u32 v26, v1, v3 // (op 22037) 

v_lshlrev_b32 v3, 5, v26 // (op 22037) 

v_lshrrev_b32 v26, 2, v1 // (op 22037) 
v_lshl_add_u32 v28, v26, 7, v3 // (op 22037) 


v_lshlrev_b32 v3, 2, v28 // (op 22037) 

v_lshrrev_b32 v26, 3, v3 // (op 22037) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 22037) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23227) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 21145) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21145) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21145) 


v_mov_b64 v[38:39], v[16:17] // (op 21145) 
s_add_u32 s6, 40960, s15 // (op 21145) 
s_mov_b32 m0, s6 // (op 21145) reg expression
s_mov_b32 s7, 128 // (op 21145) call()
buffer_load_dwordx4 v38, s[68:71], s7 offen offset: 0 lds // (op 21145) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v28 // (op 21145) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 21145) most significant half

s_add_u32 m0, m0, 4096 // (op 21145) 
buffer_load_dwordx4 v38, s[68:71], s7 offen offset: 0 lds // (op 21145) Load value direct to lds





v_and_b32 v3, -4, v1 // (op 22169) 
v_sub_u32 v26, v1, v3 // (op 22169) 

v_lshlrev_b32 v3, 5, v26 // (op 22169) 

v_lshrrev_b32 v26, 2, v1 // (op 22169) 
v_lshl_add_u32 v28, v26, 7, v3 // (op 22169) 


v_lshlrev_b32 v3, 2, v28 // (op 22169) 

v_lshrrev_b32 v26, 3, v3 // (op 22169) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 22169) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23255) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 21147) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21147) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21147) 


v_mov_b64 v[38:39], v[20:21] // (op 21147) 
s_add_u32 s6, 49152, s15 // (op 21147) 
s_mov_b32 m0, s6 // (op 21147) reg expression
s_mov_b32 s7, 192 // (op 21147) call()
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 21147) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v28 // (op 21147) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 21147) most significant half

s_add_u32 m0, m0, 4096 // (op 21147) 
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 21147) Load value direct to lds






v_and_b32 v3, -4, v1 // (op 22125) 
v_sub_u32 v26, v1, v3 // (op 22125) 

v_lshlrev_b32 v3, 5, v26 // (op 22125) 

v_lshrrev_b32 v26, 2, v1 // (op 22125) 
v_lshl_add_u32 v20, v26, 7, v3 // (op 22125) 


v_lshlrev_b32 v3, 2, v20 // (op 22125) 

v_lshrrev_b32 v26, 3, v3 // (op 22125) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 22125) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23246) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 21149) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21149) 

v_lshrrev_b64 v[20:21], 3, s[28:29] // (op 21149) 


v_mov_b64 v[28:29], v[16:17] // (op 21149) 
s_add_u32 s6, 57344, s15 // (op 21149) 
s_mov_b32 m0, s6 // (op 21149) reg expression
s_mov_b32 s7, 192 // (op 21149) call()
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 21149) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v20 // (op 21149) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 21149) most significant half

s_add_u32 m0, m0, 4096 // (op 21149) 
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 21149) Load value direct to lds






s_waitcnt vmcnt(16)
s_barrier  // (op 23183) 



ds_read_b128 v[38:41], v15 // (op 20311) Load local data 



ds_read_b128 v[42:45], v15 offset:1024 // (op 20316) Load local data 



ds_read_b128 v[46:49], v15 offset:2048 // (op 20321) Load local data 



ds_read_b128 v[50:53], v15 offset:3072 // (op 20326) Load local data 
s_waitcnt vmcnt(14)
s_barrier  // (op 23163) 



ds_read_b128 v[54:57], v4 offset:8192 // (op 20291) Load local data 



ds_read_b128 v[58:61], v4 offset:9216 // (op 20296) Load local data 



ds_read_b128 v[62:65], v4 offset:10240 // (op 20301) Load local data 



ds_read_b128 v[66:69], v4 offset:11264 // (op 20306) Load local data 
v_and_b32 v3, -4, v1 // (op 21732) 
v_sub_u32 v26, v1, v3 // (op 21732) 

v_lshlrev_b32 v3, 5, v26 // (op 21732) 

v_lshrrev_b32 v26, 2, v1 // (op 21732) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21732) 


v_lshlrev_b32 v3, 2, v16 // (op 21732) 

v_lshrrev_b32 v26, 3, v3 // (op 21732) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 21732) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21807) 
v_sub_u32 v26, v1, v3 // (op 21807) 

v_lshlrev_b32 v3, 5, v26 // (op 21807) 

v_lshrrev_b32 v26, 2, v1 // (op 21807) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21807) 


v_lshlrev_b32 v3, 2, v16 // (op 21807) 

v_lshrrev_b32 v26, 3, v3 // (op 21807) 

s_nop 0
v_readfirstlane_b32 s6, v26 // (op 21807) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21934) 
v_sub_u32 v26, v1, v3 // (op 21934) 

v_lshlrev_b32 v3, 5, v26 // (op 21934) 

v_lshrrev_b32 v26, 2, v1 // (op 21934) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21934) 


v_lshlrev_b32 v3, 2, v16 // (op 21934) 

v_lshrrev_b32 v26, 3, v3 // (op 21934) 

s_nop 0
v_readfirstlane_b32 s7, v26 // (op 21934) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21978) 
v_sub_u32 v26, v1, v3 // (op 21978) 

v_lshlrev_b32 v3, 5, v26 // (op 21978) 

v_lshrrev_b32 v26, 2, v1 // (op 21978) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21978) 


v_lshlrev_b32 v3, 2, v16 // (op 21978) 

v_lshrrev_b32 v26, 3, v3 // (op 21978) 

s_nop 0
v_readfirstlane_b32 s8, v26 // (op 21978) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 22022) 
v_sub_u32 v26, v1, v3 // (op 22022) 

v_lshlrev_b32 v3, 5, v26 // (op 22022) 

v_lshrrev_b32 v26, 2, v1 // (op 22022) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 22022) 


v_lshlrev_b32 v3, 2, v16 // (op 22022) 

v_lshrrev_b32 v26, 3, v3 // (op 22022) 

s_nop 0
v_readfirstlane_b32 s18, v26 // (op 22022) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 22066) 
v_sub_u32 v26, v1, v3 // (op 22066) 

v_lshlrev_b32 v3, 5, v26 // (op 22066) 

v_lshrrev_b32 v26, 2, v1 // (op 22066) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 22066) 


v_lshlrev_b32 v3, 2, v16 // (op 22066) 

v_lshrrev_b32 v26, 3, v3 // (op 22066) 

s_nop 0
v_readfirstlane_b32 s19, v26 // (op 22066) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 22110) 
v_sub_u32 v26, v1, v3 // (op 22110) 

v_lshlrev_b32 v3, 5, v26 // (op 22110) 

v_lshrrev_b32 v26, 2, v1 // (op 22110) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 22110) 


v_lshlrev_b32 v3, 2, v16 // (op 22110) 

v_lshrrev_b32 v26, 3, v3 // (op 22110) 

s_nop 0
v_readfirstlane_b32 s28, v26 // (op 22110) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 22154) 
v_sub_u32 v26, v1, v3 // (op 22154) 

v_lshlrev_b32 v3, 5, v26 // (op 22154) 

v_lshrrev_b32 v26, 2, v1 // (op 22154) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 22154) 


v_lshlrev_b32 v3, 2, v16 // (op 22154) 

v_lshrrev_b32 v26, 3, v3 // (op 22154) 

s_nop 0
v_readfirstlane_b32 s29, v26 // (op 22154) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 2556) call()
s_ashr_i64 s[48:49], s[2:3], 7 // (op 2555) 
s_mov_b32 s76, s48 // (op 2555) convert

s_and_b32 s48, s76, 4294967292 // (op 2555) 

s_cmp_lt_u32 s1, s48 // (op 2555) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_2555 // (op 2555) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_2555 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_2555:
s_add_u32 s1, s1, 4 // (op 2557) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 23189) 
s_waitcnt vmcnt(13)
v_mov_b32 v70, v30 // (op 20971) reg expression

v_mov_b32 v71, v31 // (op 20971) reg expression
v_mov_b32 v72, v32 // (op 20971) reg expression
v_mov_b32 v73, v33 // (op 20971) reg expression
s_lshl_b64 s[48:49], s[36:37], 6 // (op 21040) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 21040) 

v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 21040) 


v_mov_b64 v[20:21], v[24:25] // (op 21040) 
v_mov_b64 v[28:29], v[20:21] // (op 21040) 
s_mov_b32 m0, s6 // (op 21040) reg expression
s_mov_b32 s48, 256 // (op 21040) call()
v_bfe_u32 v74, v71, 0, 8 // (op 19206) 
v_bfe_u32 v75, v71, 8, 8 // (op 19206) 
v_bfe_u32 v76, v71, 16, 8 // (op 19206) 
v_bfe_u32 v77, v71, 24, 8 // (op 19206) 
v_permlane16_swap_b32 v74, v75 // (op 19206) 
v_bfe_u32 v78, v72, 0, 8 // (op 19224) 
v_permlane16_swap_b32 v76, v77 // (op 19206) 
v_bfe_u32 v79, v72, 8, 8 // (op 19224) 
v_bfe_u32 v80, v72, 16, 8 // (op 19224) 
v_permlane32_swap_b32 v74, v76 // (op 19206) 
v_permlane32_swap_b32 v75, v77 // (op 19206) 
v_bfe_u32 v81, v72, 24, 8 // (op 19224) 
v_permlane16_swap_b32 v78, v79 // (op 19224) 
v_bfe_u32 v82, v73, 0, 8 // (op 19242) 
v_permlane16_swap_b32 v80, v81 // (op 19224) 
v_bfe_u32 v83, v73, 8, 8 // (op 19242) 
v_bfe_u32 v84, v73, 16, 8 // (op 19242) 
v_permlane32_swap_b32 v78, v80 // (op 19224) 
v_permlane32_swap_b32 v79, v81 // (op 19224) 
v_bfe_u32 v85, v73, 24, 8 // (op 19242) 
v_permlane16_swap_b32 v82, v83 // (op 19242) 
s_nop 0
v_permlane16_swap_b32 v84, v85 // (op 19242) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v82, v84 // (op 19242) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v83, v85 // (op 19242) 
s_waitcnt vmcnt(12)
v_mov_b32 v86, v34 // (op 20987) reg expression
v_mov_b32 v87, v35 // (op 20987) reg expression
v_mov_b32 v88, v36 // (op 20987) reg expression
v_mov_b32 v89, v37 // (op 20987) reg expression
buffer_load_dwordx4 v28, s[72:75], s48 offen offset: 0 lds // (op 21040) Load value direct to lds
v_add_co_u32 v28, s[76:77], v28, v16 // (op 21040) least significant half
v_addc_co_u32 v29, s[76:77], 0, v29, s[76:77] // (op 21040) most significant half

s_add_u32 m0, m0, 4096 // (op 21040) 
v_bfe_u32 v90, v86, 0, 8 // (op 19062) 
v_bfe_u32 v91, v86, 8, 8 // (op 19062) 
v_bfe_u32 v92, v86, 16, 8 // (op 19062) 
v_bfe_u32 v93, v86, 24, 8 // (op 19062) 
v_permlane16_swap_b32 v90, v91 // (op 19062) 
v_bfe_u32 v94, v87, 0, 8 // (op 19077) 
v_permlane16_swap_b32 v92, v93 // (op 19062) 
v_bfe_u32 v95, v87, 8, 8 // (op 19077) 
v_bfe_u32 v96, v87, 16, 8 // (op 19077) 
v_permlane32_swap_b32 v90, v92 // (op 19062) 
v_permlane32_swap_b32 v91, v93 // (op 19062) 
v_bfe_u32 v97, v87, 24, 8 // (op 19077) 
v_permlane16_swap_b32 v94, v95 // (op 19077) 
v_bfe_u32 v98, v88, 0, 8 // (op 19094) 
buffer_load_dwordx4 v28, s[72:75], s48 offen offset: 0 lds // (op 21040) Load value direct to lds



v_permlane16_swap_b32 v96, v97 // (op 19077) 
v_bfe_u32 v99, v88, 8, 8 // (op 19094) 
v_bfe_u32 v100, v88, 16, 8 // (op 19094) 
v_permlane32_swap_b32 v94, v96 // (op 19077) 
v_permlane32_swap_b32 v95, v97 // (op 19077) 
v_bfe_u32 v101, v88, 24, 8 // (op 19094) 
v_permlane16_swap_b32 v98, v99 // (op 19094) 
v_bfe_u32 v102, v89, 0, 8 // (op 19111) 
v_permlane16_swap_b32 v100, v101 // (op 19094) 
v_bfe_u32 v103, v89, 8, 8 // (op 19111) 
v_bfe_u32 v104, v89, 16, 8 // (op 19111) 
v_permlane32_swap_b32 v98, v100 // (op 19094) 
v_permlane32_swap_b32 v99, v101 // (op 19094) 
v_bfe_u32 v105, v89, 24, 8 // (op 19111) 
v_permlane16_swap_b32 v102, v103 // (op 19111) 
s_nop 0
v_permlane16_swap_b32 v104, v105 // (op 19111) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v102, v104 // (op 19111) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v103, v105 // (op 19111) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23169) 


s_lshl_b64 s[48:49], s[30:31], 6 // (op 21035) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 21035) 

v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 21035) 


v_mov_b64 v[28:29], v[22:23] // (op 21035) 
v_mov_b64 v[86:87], v[28:29] // (op 21035) 
s_add_u32 s48, 8192, s15 // (op 21035) 
s_mov_b32 m0, s48 // (op 21035) reg expression
s_mov_b32 s49, 256 // (op 21035) call()
buffer_load_dwordx4 v86, s[68:71], s49 offen offset: 0 lds // (op 21035) Load value direct to lds
v_add_co_u32 v86, s[76:77], v86, v16 // (op 21035) least significant half
v_addc_co_u32 v87, s[76:77], 0, v87, s[76:77] // (op 21035) most significant half

s_add_u32 m0, m0, 4096 // (op 21035) 
v_bfe_u32 v106, v70, 0, 8 // (op 19191) 
v_bfe_u32 v107, v70, 8, 8 // (op 19191) 
v_bfe_u32 v108, v70, 16, 8 // (op 19191) 
v_bfe_u32 v109, v70, 24, 8 // (op 19191) 
v_permlane16_swap_b32 v106, v107 // (op 19191) 
s_nop 0
v_permlane16_swap_b32 v108, v109 // (op 19191) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v106, v108 // (op 19191) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 19191) 
buffer_load_dwordx4 v86, s[68:71], s49 offen offset: 0 lds // (op 21035) Load value direct to lds







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[38:41], a[0:3], v90, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 179) 
v_mov_b32 v3, v27 // (op 20982) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:57], v[42:45], a[4:7], v90, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 691) 
v_add_u32 v26, 16, v3 // (op 20982) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[46:49], a[8:11], v90, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 917) 
buffer_load_dwordx4 v[30:33], v26, s[64:67], 0 offen // (op 20982) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:57], v[50:53], a[12:15], v90, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1143) 
v_add_u32 v27, 16, v27 // (op 22535) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:61], v[38:41], a[16:19], v91, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1815) 
v_mov_b32 v3, v18 // (op 20998) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:61], v[42:45], a[20:23], v91, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1739) 
v_add_u32 v26, 16, v3 // (op 20998) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[46:49], a[24:27], v91, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1663) 
buffer_load_dwordx4 v[34:37], v26, s[52:55], 0 offen // (op 20998) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:61], v[50:53], a[28:31], v91, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1587) 
v_add_u32 v18, 16, v18 // (op 22483) 
s_waitcnt vmcnt(16)
s_barrier  // (op 23212) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:65], v[38:41], a[32:35], v92, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2817) 
ds_read_b128 v[70:73], v8 offset:16384 // (op 486) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[62:65], v[42:45], a[36:39], v92, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2741) 
ds_read_b128 v[86:89], v8 offset:17408 // (op 702) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:65], v[46:49], a[40:43], v92, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2665) 
ds_read_b128 v[110:113], v8 offset:18432 // (op 928) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[62:65], v[50:53], a[44:47], v92, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2589) 
ds_read_b128 v[114:117], v8 offset:19456 // (op 1154) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:69], v[38:41], a[48:51], v93, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3819) 
s_waitcnt vmcnt(14)
s_barrier  // (op 23202) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:69], v[42:45], a[52:55], v93, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3743) 
ds_read_b128 v[54:57], v7 offset:24576 // (op 484) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[66:69], v[46:49], a[56:59], v93, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3667) 
ds_read_b128 v[58:61], v7 offset:25600 // (op 1597) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:69], v[50:53], a[60:63], v93, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3591) 
ds_read_b128 v[62:65], v7 offset:26624 // (op 2599) Load local data 





ds_read_b128 v[90:93], v7 offset:27648 // (op 3601) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[70:73], a[0:3], v94, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 491) 
s_barrier  // (op 23219) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:57], v[86:89], a[4:7], v94, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 692) 
s_lshl_b64 s[48:49], s[36:37], 6 // (op 21062) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:57], v[114:117], a[12:15], v94, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1144) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 21062) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[110:113], a[8:11], v94, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 918) 
v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 21062) 


v_mov_b64 v[106:107], v[20:21] // (op 21062) 
s_add_u32 s48, 16384, s8 // (op 21062) 
s_mov_b32 m0, s48 // (op 21062) reg expression
s_mov_b32 s49, 320 // (op 21062) call()
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:61], v[70:73], a[16:19], v95, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1816) 
s_waitcnt vmcnt(12)
s_barrier  // (op 23233) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:61], v[86:89], a[20:23], v95, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1740) 
ds_read_b128 v[38:41], v10 offset:32768 // (op 514) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[110:113], a[24:27], v95, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1664) 
ds_read_b128 v[42:45], v10 offset:33792 // (op 712) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:61], v[114:117], a[28:31], v95, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1588) 
ds_read_b128 v[46:49], v10 offset:34816 // (op 938) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:65], v[70:73], a[32:35], v96, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2818) 
ds_read_b128 v[50:53], v10 offset:35840 // (op 1164) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[62:65], v[86:89], a[36:39], v96, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2742) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:65], v[110:113], a[40:43], v96, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2666) 
s_waitcnt vmcnt(10)
s_barrier  // (op 23223) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[62:65], v[114:117], a[44:47], v96, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2590) 
ds_read_b128 v[54:57], v9 offset:40960 // (op 512) Load local data 
buffer_load_dwordx4 v106, s[72:75], s49 offen offset: 0 lds // (op 21062) Load value direct to lds



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[70:73], a[48:51], v97, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3820) 
ds_read_b128 v[58:61], v9 offset:41984 // (op 1607) Load local data 

v_add_co_u32 v106, s[76:77], v106, v16 // (op 21062) least significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[86:89], a[52:55], v97, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3744) 
v_addc_co_u32 v107, s[76:77], 0, v107, s[76:77] // (op 21062) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[110:113], a[56:59], v97, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3668) 
s_add_u32 m0, m0, 4096 // (op 21062) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[114:117], a[60:63], v97, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3592) 
buffer_load_dwordx4 v106, s[72:75], s49 offen offset: 0 lds // (op 21062) Load value direct to lds








s_barrier  // (op 23209) 
ds_read_b128 v[70:73], v9 offset:43008 // (op 2609) Load local data 




s_lshl_b64 s[48:49], s[30:31], 6 // (op 21057) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 21057) 

v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 21057) 


v_mov_b64 v[74:75], v[28:29] // (op 21057) 
s_add_u32 s48, 24576, s7 // (op 21057) 
s_mov_b32 m0, s48 // (op 21057) reg expression
s_mov_b32 s49, 320 // (op 21057) call()
buffer_load_dwordx4 v74, s[68:71], s49 offen offset: 0 lds // (op 21057) Load value direct to lds
v_add_co_u32 v74, s[76:77], v74, v16 // (op 21057) least significant half
v_addc_co_u32 v75, s[76:77], 0, v75, s[76:77] // (op 21057) most significant half

s_add_u32 m0, m0, 4096 // (op 21057) 
ds_read_b128 v[86:89], v9 offset:44032 // (op 3611) Load local data 
buffer_load_dwordx4 v74, s[68:71], s49 offen offset: 0 lds // (op 21057) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[38:41], a[0:3], v98, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 519) 
s_barrier  // (op 23239) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:57], v[42:45], a[4:7], v98, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 693) 
s_lshl_b64 s[48:49], s[36:37], 6 // (op 21072) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[46:49], a[8:11], v98, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 919) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 21072) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:57], v[50:53], a[12:15], v98, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1145) 
v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 21072) 


v_mov_b64 v[74:75], v[20:21] // (op 21072) 
s_add_u32 s48, 32768, s19 // (op 21072) 
s_mov_b32 m0, s48 // (op 21072) reg expression
s_mov_b32 s49, 384 // (op 21072) call()
buffer_load_dwordx4 v74, s[72:75], s49 offen offset: 0 lds // (op 21072) Load value direct to lds
v_add_co_u32 v74, s[76:77], v74, v16 // (op 21072) least significant half
v_addc_co_u32 v75, s[76:77], 0, v75, s[76:77] // (op 21072) most significant half

s_add_u32 m0, m0, 4096 // (op 21072) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:61], v[38:41], a[16:19], v99, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1817) 
buffer_load_dwordx4 v74, s[72:75], s49 offen offset: 0 lds // (op 21072) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:61], v[42:45], a[20:23], v99, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1741) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23230) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[46:49], a[24:27], v99, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1665) 
s_lshl_b64 s[48:49], s[30:31], 6 // (op 21067) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:61], v[50:53], a[28:31], v99, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1589) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 21067) 

v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 21067) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:73], v[38:41], a[32:35], v100, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2819) 
v_mov_b64 v[74:75], v[28:29] // (op 21067) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[42:45], a[36:39], v100, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2743) 
s_add_u32 s48, 40960, s18 // (op 21067) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[70:73], v[46:49], a[40:43], v100, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2667) 
s_mov_b32 m0, s48 // (op 21067) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[70:73], v[50:53], a[44:47], v100, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2591) 
s_mov_b32 s49, 384 // (op 21067) call()

s_waitcnt vmcnt(14)
s_barrier  // (op 23252) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[38:41], a[48:51], v101, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3821) 
ds_read_b128 v[62:65], v12 offset:49152 // (op 542) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[42:45], a[52:55], v101, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3745) 
ds_read_b128 v[66:69], v12 offset:50176 // (op 722) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v101, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3669) 
ds_read_b128 v[70:73], v12 offset:51200 // (op 948) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[50:53], a[60:63], v101, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3593) 
ds_read_b128 v[90:93], v12 offset:52224 // (op 1174) Load local data 



buffer_load_dwordx4 v74, s[68:71], s49 offen offset: 0 lds // (op 21067) Load value direct to lds
v_add_co_u32 v74, s[76:77], v74, v16 // (op 21067) least significant half
v_addc_co_u32 v75, s[76:77], 0, v75, s[76:77] // (op 21067) most significant half

s_add_u32 m0, m0, 4096 // (op 21067) 
buffer_load_dwordx4 v74, s[68:71], s49 offen offset: 0 lds // (op 21067) Load value direct to lds




s_waitcnt vmcnt(14)
s_barrier  // (op 23242) 



ds_read_b128 v[54:57], v11 offset:57344 // (op 540) Load local data 



ds_read_b128 v[58:61], v11 offset:58368 // (op 1617) Load local data 



ds_read_b128 v[86:89], v11 offset:59392 // (op 2619) Load local data 



ds_read_b128 v[74:77], v11 offset:60416 // (op 3621) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[62:65], a[0:3], v102, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 547) 
s_barrier  // (op 23258) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:57], v[66:69], a[4:7], v102, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 694) 
s_lshl_b64 s[48:49], s[36:37], 6 // (op 21082) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[70:73], a[8:11], v102, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 920) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 21082) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:57], v[90:93], a[12:15], v102, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1146) 
v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 21082) 


v_mov_b64 v[78:79], v[20:21] // (op 21082) 
s_add_u32 s48, 49152, s29 // (op 21082) 
s_mov_b32 m0, s48 // (op 21082) reg expression
s_mov_b32 s49, 448 // (op 21082) call()
buffer_load_dwordx4 v78, s[72:75], s49 offen offset: 0 lds // (op 21082) Load value direct to lds
v_add_co_u32 v78, s[76:77], v78, v16 // (op 21082) least significant half
v_addc_co_u32 v79, s[76:77], 0, v79, s[76:77] // (op 21082) most significant half

s_add_u32 m0, m0, 4096 // (op 21082) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:61], v[62:65], a[16:19], v103, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1818) 
s_waitcnt vmcnt(13)
s_barrier  // (op 23195) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:61], v[66:69], a[20:23], v103, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1742) 
ds_read_b128 v[38:41], v6 // (op 174) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[70:73], a[24:27], v103, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1666) 
ds_read_b128 v[42:45], v6 offset:1024 // (op 688) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:61], v[90:93], a[28:31], v103, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1590) 
ds_read_b128 v[46:49], v6 offset:2048 // (op 914) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[86:89], v[62:65], a[32:35], v104, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2820) 
ds_read_b128 v[50:53], v6 offset:3072 // (op 1140) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[86:89], v[66:69], a[36:39], v104, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2744) 
s_waitcnt vmcnt(11)
s_barrier  // (op 23175) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[86:89], v[70:73], a[40:43], v104, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2668) 
ds_read_b128 v[54:57], v5 offset:8192 // (op 172) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[90:93], a[44:47], v104, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2592) 
buffer_load_dwordx4 v78, s[72:75], s49 offen offset: 0 lds // (op 21082) Load value direct to lds









s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:77], v[62:65], a[48:51], v105, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3822) 
s_barrier  // (op 23249) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:77], v[66:69], a[52:55], v105, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3746) 
ds_read_b128 v[58:61], v5 offset:9216 // (op 1583) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:77], v[70:73], a[56:59], v105, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3670) 
ds_read_b128 v[62:65], v5 offset:10240 // (op 2585) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[90:93], a[60:63], v105, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3594) 
ds_read_b128 v[66:69], v5 offset:11264 // (op 3587) Load local data 




v_add_co_u32 v24, vcc, 256, v24 // (op 21406) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 21406) most significant half
s_lshl_b64 s[48:49], s[30:31], 6 // (op 21077) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 21077) 

v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 21077) 


v_mov_b64 v[20:21], v[28:29] // (op 21077) 
s_add_u32 s48, 57344, s28 // (op 21077) 
s_mov_b32 m0, s48 // (op 21077) reg expression
s_mov_b32 s49, 448 // (op 21077) call()
buffer_load_dwordx4 v20, s[68:71], s49 offen offset: 0 lds // (op 21077) Load value direct to lds
v_add_co_u32 v20, s[76:77], v20, v16 // (op 21077) least significant half
v_addc_co_u32 v21, s[76:77], 0, v21, s[76:77] // (op 21077) most significant half

s_add_u32 m0, m0, 4096 // (op 21077) 
buffer_load_dwordx4 v20, s[68:71], s49 offen offset: 0 lds // (op 21077) Load value direct to lds





v_add_co_u32 v22, vcc, 256, v22 // (op 21260) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 21260) most significant half
s_ashr_i64 s[48:49], s[2:3], 7 // (op 2555) 
s_mov_b32 s76, s48 // (op 2555) convert

s_and_b32 s48, s76, 4294967292 // (op 2555) 

s_cmp_lt_u32 s1, s48 // (op 2555) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_2555 // (op 2555) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_2555 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_2555:














s_barrier  // (op 23417) 
v_lshrrev_b32 v3, 2, v1 // (op 21184) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 21184) 
v_and_b32 v16, -4, v1 // (op 21184) 
s_and_b32 s15, s18, 4294967292 // (op 21184) 
v_sub_u32 v17, v1, v16 // (op 21184) 

s_add_u32 s6, s0, s15 // (op 21184) 

v_lshlrev_b32 v16, 5, v17 // (op 21184) 

s_mul_i32 s15, s16, s50 // (op 21184) 
s_add_u32 s15, s15, s6 // (op 21184) 

s_add_u32 s6, s15, s9 // (op 21184) 

s_mul_hi_u32 s15, s6, s13 // (op 21184) 
s_sub_u32 s7, s6, s15 // (op 21184) 
s_lshr_b32 s8, s7, 1 // (op 21184) 

s_add_u32 s7, s8, s15 // (op 21184) 


s_lshr_b32 s8, s7, s11 // (op 21184) 

s_and_b32 s7, s11, 2147483648 // (op 21184) 
s_cmp_eq_i32 s7, 0 // (op 21184) 
s_mov_b32 s15, scc // (op 21184) 

s_cmp_eq_i32 s15, 0 // (op 21184) 
s_cselect_b32 s7, s6, s8 // (op 21184) 


s_mul_hi_u32 s8, s7, s12 // (op 21184) 
s_sub_u32 s15, s7, s8 // (op 21184) 
s_lshr_b32 s28, s15, 1 // (op 21184) 

s_add_u32 s15, s28, s8 // (op 21184) 


s_lshr_b32 s8, s15, s10 // (op 21184) 

s_and_b32 s15, s10, 2147483648 // (op 21184) 
s_cmp_eq_i32 s15, 0 // (op 21184) 
s_mov_b32 s28, scc // (op 21184) 

s_cmp_eq_i32 s28, 0 // (op 21184) 
s_cselect_b32 s15, s7, s8 // (op 21184) 


s_mul_i32 s8, s7, s18 // (op 21184) 


v_lshl_add_u32 v17, s15, 7, v3 // (op 21184) 


s_sub_u32 s7, s6, s8 // (op 21184) 


v_lshl_add_u32 v3, s7, 7, v16 // (op 21184) 


v_mov_b32 v21, 0 // (op 21184) convert
v_mov_b32 v20, v3 // (op 21184) convert

v_mul_lo_u32 v23, v17, s31 // (op 21184) most significant: low of low * high
v_mul_hi_u32 v3, v17, s30 // (op 21184) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 21184) least significant: low of low * low
v_add_u32 v23, v23, v3 // (op 21184) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v20 // (op 21184) least significant half
v_addc_co_u32 v23, s[18:19], v23, v21, s[18:19] // (op 21184) most significant half



v_lshlrev_b64 v[20:21], 2, v[22:23] // (op 21184) 

v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 21184) 

s_mul_i32 s19, s30, s35 // (op 21186) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 21186) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 21186) least significant
s_mul_hi_u32 s6, s30, s34 // (op 21186) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21186) most significant: sum
s_add_u32 s19, s19, s6 // (op 21186) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21186) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21186) 

s_mov_b32 s68, s32 // (op 21186) reg expression
s_mov_b32 s69, s33 // (op 21186) reg expression
s_mov_b32 s70, s18 // (op 21186) reg expression
s_mov_b32 s71, 131072 // (op 21186) call()

v_lshrrev_b32 v3, 2, v1 // (op 21350) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 21350) 
v_and_b32 v18, -4, v1 // (op 21350) 
s_and_b32 s15, s18, 4294967292 // (op 21350) 
v_sub_u32 v20, v1, v18 // (op 21350) 

s_add_u32 s6, s0, s15 // (op 21350) 

v_lshlrev_b32 v18, 5, v20 // (op 21350) 

s_mul_i32 s15, s16, s50 // (op 21350) 
s_add_u32 s15, s15, s6 // (op 21350) 

s_add_u32 s6, s15, s9 // (op 21350) 

s_mul_hi_u32 s15, s6, s13 // (op 21350) 
s_sub_u32 s7, s6, s15 // (op 21350) 
s_lshr_b32 s8, s7, 1 // (op 21350) 

s_add_u32 s7, s8, s15 // (op 21350) 


s_lshr_b32 s8, s7, s11 // (op 21350) 

s_and_b32 s7, s11, 2147483648 // (op 21350) 
s_cmp_eq_i32 s7, 0 // (op 21350) 
s_mov_b32 s15, scc // (op 21350) 

s_cmp_eq_i32 s15, 0 // (op 21350) 
s_cselect_b32 s7, s6, s8 // (op 21350) 


s_mul_hi_u32 s8, s7, s12 // (op 21350) 
s_sub_u32 s15, s7, s8 // (op 21350) 
s_lshr_b32 s28, s15, 1 // (op 21350) 

s_add_u32 s15, s28, s8 // (op 21350) 


s_lshr_b32 s8, s15, s10 // (op 21350) 

s_and_b32 s15, s10, 2147483648 // (op 21350) 
s_cmp_eq_i32 s15, 0 // (op 21350) 
s_mov_b32 s28, scc // (op 21350) 

s_cmp_eq_i32 s28, 0 // (op 21350) 
s_cselect_b32 s15, s7, s8 // (op 21350) 


s_mul_i32 s8, s15, s14 // (op 21350) 

s_sub_u32 s15, s7, s8 // (op 21350) 

s_mul_i32 s8, s7, s18 // (op 21350) 


v_lshl_add_u32 v20, s15, 7, v3 // (op 21350) 


s_sub_u32 s7, s6, s8 // (op 21350) 


v_mul_lo_u32 v29, v20, s37 // (op 21350) most significant: low of low * high
v_mul_hi_u32 v3, v20, s36 // (op 21350) most significant: high of low * low
v_mul_lo_u32 v28, v20, s36 // (op 21350) least significant: low of low * low
v_add_u32 v29, v29, v3 // (op 21350) most significant: sum


v_lshl_add_u32 v3, s7, 7, v18 // (op 21350) 


v_mov_b32 v21, 0 // (op 21350) convert
v_mov_b32 v20, v3 // (op 21350) convert

v_add_co_u32 v22, s[18:19], v20, v28 // (op 21350) least significant half
v_addc_co_u32 v23, s[18:19], v21, v29, s[18:19] // (op 21350) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 21350) 


v_lshrrev_b64 v[20:21], 3, v[22:23] // (op 21350) 

s_mul_i32 s19, s36, s23 // (op 21352) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 21352) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 21352) least significant
s_mul_hi_u32 s6, s36, s22 // (op 21352) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21352) most significant: sum
s_add_u32 s19, s19, s6 // (op 21352) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21352) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21352) 

s_mov_b32 s72, s38 // (op 21352) reg expression
s_mov_b32 s73, s39 // (op 21352) reg expression
s_mov_b32 s74, s18 // (op 21352) reg expression
s_mov_b32 s75, 131072 // (op 21352) call()

v_and_b32 v3, -4, v1 // (op 21745) 
v_sub_u32 v18, v1, v3 // (op 21745) 

v_lshlrev_b32 v3, 5, v18 // (op 21745) 

v_lshrrev_b32 v18, 2, v1 // (op 21745) 
v_lshl_add_u32 v22, v18, 7, v3 // (op 21745) 


v_lshlrev_b32 v3, 2, v22 // (op 21745) 

v_lshrrev_b32 v18, 3, v3 // (op 21745) 

s_nop 0
v_readfirstlane_b32 s15, v18 // (op 21745) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21820) 
v_sub_u32 v18, v1, v3 // (op 21820) 

v_lshlrev_b32 v3, 5, v18 // (op 21820) 

v_lshrrev_b32 v18, 2, v1 // (op 21820) 
v_lshl_add_u32 v22, v18, 7, v3 // (op 21820) 


v_lshlrev_b32 v3, 2, v22 // (op 21820) 

v_lshrrev_b32 v18, 3, v3 // (op 21820) 

s_nop 0
v_readfirstlane_b32 s6, v18 // (op 21820) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, 63, v1 // (op 22572) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 22572) 
v_lshrrev_b32 v18, 4, v3 // (op 22572) 
s_and_b32 s7, s18, 4294967292 // (op 22572) 
v_and_b32 v22, 3, v18 // (op 22572) 

s_add_u32 s8, s0, s7 // (op 22572) 

v_and_b32 v18, 15, v1 // (op 22572) 
v_lshl_add_u32 v23, v22, 4, v18 // (op 22572) 


s_mul_i32 s7, s16, s50 // (op 22572) 
s_add_u32 s7, s7, s8 // (op 22572) 

v_lshrrev_b32 v18, 7, v1 // (op 22572) 
s_add_u32 s8, s7, s9 // (op 22572) 

v_lshl_add_u32 v22, v18, 6, v23 // (op 22572) 


s_mul_hi_u32 s7, s8, s13 // (op 22572) 
v_lshrrev_b32 v18, 6, v3 // (op 22572) 

s_sub_u32 s28, s8, s7 // (op 22572) 
v_lshlrev_b32 v3, 2, v18 // (op 22572) 

s_lshr_b32 s29, s28, 1 // (op 22572) 

s_add_u32 s28, s29, s7 // (op 22572) 


s_lshr_b32 s7, s28, s11 // (op 22572) 

s_and_b32 s28, s11, 2147483648 // (op 22572) 
s_cmp_eq_i32 s28, 0 // (op 22572) 
s_mov_b32 s29, scc // (op 22572) 

s_cmp_eq_i32 s29, 0 // (op 22572) 
s_cselect_b32 s28, s8, s7 // (op 22572) 


s_mul_hi_u32 s7, s28, s12 // (op 22572) 
s_sub_u32 s29, s28, s7 // (op 22572) 
s_lshr_b32 s48, s29, 1 // (op 22572) 

s_add_u32 s29, s48, s7 // (op 22572) 


s_lshr_b32 s7, s29, s10 // (op 22572) 

s_and_b32 s29, s10, 2147483648 // (op 22572) 
s_cmp_eq_i32 s29, 0 // (op 22572) 
s_mov_b32 s48, scc // (op 22572) 

s_cmp_eq_i32 s48, 0 // (op 22572) 
s_cselect_b32 s29, s28, s7 // (op 22572) 


s_mul_i32 s7, s28, s18 // (op 22572) 


v_lshl_add_u32 v18, s29, 7, v22 // (op 22572) 


s_sub_u32 s18, s8, s7 // (op 22572) 


v_lshl_add_u32 v22, s18, 2, v3 // (op 22572) 


v_mul_lo_u32 v3, v18, s40 // (op 22572) 
v_add_u32 v3, v3, v22 // (op 22572) 


s_mul_i32 s19, s40, s35 // (op 22574) most significant: low of low * high
s_mul_i32 s7, s41, s34 // (op 22574) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 22574) least significant
s_mul_hi_u32 s8, s40, s34 // (op 22574) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 22574) most significant: sum
s_add_u32 s19, s19, s8 // (op 22574) most significant: sum


s_mov_b32 s52, s42 // (op 22574) reg expression
s_mov_b32 s53, s43 // (op 22574) reg expression
s_mov_b32 s54, s18 // (op 22574) reg expression
s_mov_b32 s55, 131072 // (op 22574) call()

v_and_b32 v18, 63, v1 // (op 22599) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 22599) 
v_lshrrev_b32 v22, 4, v18 // (op 22599) 
s_and_b32 s7, s18, 4294967292 // (op 22599) 
v_and_b32 v23, 3, v22 // (op 22599) 

s_add_u32 s8, s0, s7 // (op 22599) 

v_and_b32 v22, 15, v1 // (op 22599) 
v_lshl_add_u32 v24, v23, 4, v22 // (op 22599) 


s_mul_i32 s7, s16, s50 // (op 22599) 
s_add_u32 s7, s7, s8 // (op 22599) 

v_lshrrev_b32 v22, 6, v1 // (op 22599) 
s_add_u32 s8, s7, s9 // (op 22599) 

v_and_b32 v23, 1, v22 // (op 22599) 

s_mul_hi_u32 s7, s8, s13 // (op 22599) 
v_lshl_add_u32 v22, v23, 6, v24 // (op 22599) 


s_sub_u32 s28, s8, s7 // (op 22599) 
v_lshrrev_b32 v23, 6, v18 // (op 22599) 

s_lshr_b32 s29, s28, 1 // (op 22599) 

v_lshlrev_b32 v18, 2, v23 // (op 22599) 

s_add_u32 s28, s29, s7 // (op 22599) 


s_lshr_b32 s7, s28, s11 // (op 22599) 

s_and_b32 s28, s11, 2147483648 // (op 22599) 
s_cmp_eq_i32 s28, 0 // (op 22599) 
s_mov_b32 s29, scc // (op 22599) 

s_cmp_eq_i32 s29, 0 // (op 22599) 
s_cselect_b32 s28, s8, s7 // (op 22599) 


s_mul_hi_u32 s7, s28, s12 // (op 22599) 
s_sub_u32 s29, s28, s7 // (op 22599) 
s_lshr_b32 s48, s29, 1 // (op 22599) 

s_add_u32 s29, s48, s7 // (op 22599) 


s_lshr_b32 s7, s29, s10 // (op 22599) 

s_and_b32 s29, s10, 2147483648 // (op 22599) 
s_cmp_eq_i32 s29, 0 // (op 22599) 
s_mov_b32 s48, scc // (op 22599) 

s_cmp_eq_i32 s48, 0 // (op 22599) 
s_cselect_b32 s29, s28, s7 // (op 22599) 


s_mul_i32 s7, s29, s14 // (op 22599) 

s_sub_u32 s29, s28, s7 // (op 22599) 

v_lshl_add_u32 v23, s29, 7, v22 // (op 22599) 


s_mul_i32 s7, s28, s18 // (op 22599) 


v_mul_lo_u32 v22, v23, s44 // (op 22599) 

s_sub_u32 s18, s8, s7 // (op 22599) 


v_lshl_add_u32 v23, s18, 2, v18 // (op 22599) 


v_add_u32 v18, v23, v22 // (op 22599) 


s_mul_i32 s19, s44, s23 // (op 22601) most significant: low of low * high
s_mul_i32 s7, s45, s22 // (op 22601) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 22601) least significant
s_mul_hi_u32 s8, s44, s22 // (op 22601) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 22601) most significant: sum
s_add_u32 s19, s19, s8 // (op 22601) most significant: sum


s_mov_b32 s64, s46 // (op 22601) reg expression
s_mov_b32 s65, s47 // (op 22601) reg expression
s_mov_b32 s66, s18 // (op 22601) reg expression
s_mov_b32 s67, 131072 // (op 22601) call()

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3563) 
s_and_b32 s7, s18, 4294967292 // (op 3563) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3562) 
s_mov_b32 s8, s18 // (op 3562) convert

s_cmp_lt_u32 s7, s8 // (op 3562) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_3562 // (op 3562) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_3562 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_3562:
s_barrier  // (op 23172) 
s_add_u32 s7, s7, 1 // (op 3564) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 21045) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21045) 

v_lshrrev_b64 v[22:23], 3, s[28:29] // (op 21045) 


v_mov_b64 v[24:25], v[16:17] // (op 21045) 
v_mov_b64 v[26:27], v[24:25] // (op 21045) 
s_add_u32 s8, 8192, s15 // (op 21045) 
s_mov_b32 m0, s8 // (op 21045) reg expression
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 21045) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v22 // (op 21045) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 21045) most significant half

s_add_u32 m0, m0, 4096 // (op 21045) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 21045) Load value direct to lds




v_add_co_u32 v16, s[18:19], 64, v16 // (op 21174) least significant half
v_addc_co_u32 v17, s[18:19], 0, v17, s[18:19] // (op 21174) most significant half

s_barrier  // (op 23192) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 21051) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21051) 

v_lshrrev_b64 v[22:23], 3, s[28:29] // (op 21051) 


v_mov_b64 v[24:25], v[20:21] // (op 21051) 
v_mov_b64 v[26:27], v[24:25] // (op 21051) 
s_mov_b32 m0, s6 // (op 21051) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 21051) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v22 // (op 21051) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 21051) most significant half

s_add_u32 m0, m0, 4096 // (op 21051) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 21051) Load value direct to lds



v_add_co_u32 v20, s[18:19], 64, v20 // (op 21342) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 21342) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 23179) 



ds_read_b128 v[22:25], v5 offset:8192 // (op 434) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 23198) 



ds_read_b128 v[26:29], v6 // (op 436) Load local data 


v_mov_b32 v30, v3 // (op 438) 
v_mov_b32 v31, v30 // (op 438) reg expression

buffer_load_dwordx4 v[32:35], v31, s[52:55], 0 offen // (op 438) Load value


v_add_u32 v3, 4, v3 // (op 22562) 


v_mov_b32 v30, v18 // (op 440) 
v_mov_b32 v31, v30 // (op 440) reg expression

buffer_load_dwordx4 v[36:39], v31, s[64:67], 0 offen // (op 440) Load value


v_add_u32 v18, 4, v18 // (op 22589) 
s_waitcnt vmcnt(1)
v_bfe_u32 v40, v32, 0, 8 // (op 19323) 



v_bfe_u32 v41, v32, 8, 8 // (op 19323) 
v_bfe_u32 v42, v32, 16, 8 // (op 19323) 
v_bfe_u32 v43, v32, 24, 8 // (op 19323) 
v_permlane16_swap_b32 v40, v41 // (op 19323) 
ds_read_b128 v[44:47], v6 offset:1024 // (op 671) Load local data 
v_permlane16_swap_b32 v42, v43 // (op 19323) 



s_nop 1
v_permlane32_swap_b32 v40, v42 // (op 19323) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v41, v43 // (op 19323) 
ds_read_b128 v[48:51], v6 offset:2048 // (op 897) Load local data 




ds_read_b128 v[30:33], v6 offset:3072 // (op 1123) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v52, v36, 0, 8 // (op 19353) 



v_bfe_u32 v53, v36, 8, 8 // (op 19353) 
v_bfe_u32 v54, v36, 16, 8 // (op 19353) 
v_bfe_u32 v55, v36, 24, 8 // (op 19353) 
v_permlane16_swap_b32 v52, v53 // (op 19353) 
s_nop 0
v_permlane16_swap_b32 v54, v55 // (op 19353) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v52, v54 // (op 19353) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v53, v55 // (op 19353) 
ds_read_b128 v[56:59], v5 offset:9216 // (op 1483) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 441) 
ds_read_b128 v[34:37], v5 offset:10240 // (op 2485) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[44:47], a[4:7], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 674) 
ds_read_b128 v[60:63], v5 offset:11264 // (op 3487) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[48:51], a[8:11], v40, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 900) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:25], v[30:33], a[12:15], v40, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1126) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[26:29], a[16:19], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1487) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v41, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1541) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[30:33], a[28:31], v41, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1568) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[44:47], a[20:23], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1514) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[34:37], v[26:29], a[32:35], v42, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2489) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[34:37], v[48:51], a[40:43], v42, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2543) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[34:37], v[30:33], a[44:47], v42, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2570) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[34:37], v[44:47], a[36:39], v42, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2516) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:63], v[26:29], a[48:51], v43, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3491) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:63], v[48:51], a[56:59], v43, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3545) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[60:63], v[30:33], a[60:63], v43, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3572) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:63], v[44:47], a[52:55], v43, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3518) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 3562) 
s_mov_b32 s8, s18 // (op 3562) convert

s_cmp_lt_u32 s7, s8 // (op 3562) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_3562 // (op 3562) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_3562 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_3562:









s_barrier  // (op 23435) 


v_cmp_eq_f32 vcc, 0.00000, v0 // (op 207) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_207 // (op 207) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_207
v_accvgpr_read v20, a0 // (op 215) 
v_accvgpr_read v21, a1 // (op 215) 
v_accvgpr_read v22, a2 // (op 215) 
v_accvgpr_read v23, a3 // (op 215) 
v_mul_f32 v24, v19, v20 // (op 215) 
v_mul_f32 v25, v19, v21 // (op 215) 
v_mul_f32 v26, v19, v22 // (op 215) 
v_mul_f32 v27, v19, v23 // (op 215) 

v_cvt_f16_f32 v20, v24 // (op 217) 
v_cvt_f16_f32 v21, v25 // (op 217) 
v_cvt_f16_f32 v22, v26 // (op 217) 
v_cvt_f16_f32 v23, v27 // (op 217) 


v_pack_b32_f16 v16, v20, v21 // (op 218) 
v_pack_b32_f16 v17, v22, v23 // (op 218) 

v_mov_b32 v15, v14 // (op 218) reg expression

buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 218) Store value



v_accvgpr_read v20, a4 // (op 4968) 
v_accvgpr_read v21, a5 // (op 4968) 
v_accvgpr_read v22, a6 // (op 4968) 
v_accvgpr_read v23, a7 // (op 4968) 
v_mul_f32 v24, v19, v20 // (op 4968) 
v_mul_f32 v25, v19, v21 // (op 4968) 
v_mul_f32 v26, v19, v22 // (op 4968) 
v_mul_f32 v27, v19, v23 // (op 4968) 

v_cvt_f16_f32 v20, v24 // (op 4970) 
v_cvt_f16_f32 v21, v25 // (op 4970) 
v_cvt_f16_f32 v22, v26 // (op 4970) 
v_cvt_f16_f32 v23, v27 // (op 4970) 


v_pack_b32_f16 v16, v20, v21 // (op 4971) 
v_pack_b32_f16 v17, v22, v23 // (op 4971) 

s_lshl_b32 s15, s24, 5 // (op 4971) 
v_add_u32 v15, v14, s15 // (op 4971) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 4971) Store value



v_accvgpr_read v20, a8 // (op 4982) 
v_accvgpr_read v21, a9 // (op 4982) 
v_accvgpr_read v22, a10 // (op 4982) 
v_accvgpr_read v23, a11 // (op 4982) 
v_mul_f32 v24, v19, v20 // (op 4982) 
v_mul_f32 v25, v19, v21 // (op 4982) 
v_mul_f32 v26, v19, v22 // (op 4982) 
v_mul_f32 v27, v19, v23 // (op 4982) 

v_cvt_f16_f32 v20, v24 // (op 4984) 
v_cvt_f16_f32 v21, v25 // (op 4984) 
v_cvt_f16_f32 v22, v26 // (op 4984) 
v_cvt_f16_f32 v23, v27 // (op 4984) 


v_pack_b32_f16 v16, v20, v21 // (op 4985) 
v_pack_b32_f16 v17, v22, v23 // (op 4985) 

s_lshl_b32 s15, s24, 6 // (op 4985) 
v_add_u32 v15, v14, s15 // (op 4985) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 4985) Store value



v_accvgpr_read v20, a12 // (op 4996) 
v_accvgpr_read v21, a13 // (op 4996) 
v_accvgpr_read v22, a14 // (op 4996) 
v_accvgpr_read v23, a15 // (op 4996) 
v_mul_f32 v24, v19, v20 // (op 4996) 
v_mul_f32 v25, v19, v21 // (op 4996) 
v_mul_f32 v26, v19, v22 // (op 4996) 
v_mul_f32 v27, v19, v23 // (op 4996) 

v_cvt_f16_f32 v20, v24 // (op 4998) 
v_cvt_f16_f32 v21, v25 // (op 4998) 
v_cvt_f16_f32 v22, v26 // (op 4998) 
v_cvt_f16_f32 v23, v27 // (op 4998) 


v_pack_b32_f16 v16, v20, v21 // (op 4999) 
v_pack_b32_f16 v17, v22, v23 // (op 4999) 

s_lshl_b32 s15, s24, 5 // (op 4999) 
s_mul_i32 s6, 3, s15 // (op 4999) 

v_add_u32 v15, v14, s6 // (op 4999) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 4999) Store value



v_accvgpr_read v20, a16 // (op 5032) 
v_accvgpr_read v21, a17 // (op 5032) 
v_accvgpr_read v22, a18 // (op 5032) 
v_accvgpr_read v23, a19 // (op 5032) 
v_mul_f32 v24, v19, v20 // (op 5032) 
v_mul_f32 v25, v19, v21 // (op 5032) 
v_mul_f32 v26, v19, v22 // (op 5032) 
v_mul_f32 v27, v19, v23 // (op 5032) 

v_cvt_f16_f32 v20, v24 // (op 5034) 
v_cvt_f16_f32 v21, v25 // (op 5034) 
v_cvt_f16_f32 v22, v26 // (op 5034) 
v_cvt_f16_f32 v23, v27 // (op 5034) 


v_pack_b32_f16 v16, v20, v21 // (op 5036) 
v_pack_b32_f16 v17, v22, v23 // (op 5036) 

v_add_u32 v15, 32, v14 // (op 5036) 

buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5036) Store value



v_accvgpr_read v20, a20 // (op 5046) 
v_accvgpr_read v21, a21 // (op 5046) 
v_accvgpr_read v22, a22 // (op 5046) 
v_accvgpr_read v23, a23 // (op 5046) 
v_mul_f32 v24, v19, v20 // (op 5046) 
v_mul_f32 v25, v19, v21 // (op 5046) 
v_mul_f32 v26, v19, v22 // (op 5046) 
v_mul_f32 v27, v19, v23 // (op 5046) 

v_cvt_f16_f32 v20, v24 // (op 5048) 
v_cvt_f16_f32 v21, v25 // (op 5048) 
v_cvt_f16_f32 v22, v26 // (op 5048) 
v_cvt_f16_f32 v23, v27 // (op 5048) 


v_pack_b32_f16 v16, v20, v21 // (op 5038) 
v_pack_b32_f16 v17, v22, v23 // (op 5038) 

s_lshl_b32 s15, s24, 5 // (op 5038) 
s_add_u32 s15, s15, 32 // (op 5038) 
v_add_u32 v15, v14, s15 // (op 5038) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5038) Store value



v_accvgpr_read v20, a24 // (op 5052) 
v_accvgpr_read v21, a25 // (op 5052) 
v_accvgpr_read v22, a26 // (op 5052) 
v_accvgpr_read v23, a27 // (op 5052) 
v_mul_f32 v24, v19, v20 // (op 5052) 
v_mul_f32 v25, v19, v21 // (op 5052) 
v_mul_f32 v26, v19, v22 // (op 5052) 
v_mul_f32 v27, v19, v23 // (op 5052) 

v_cvt_f16_f32 v20, v24 // (op 5054) 
v_cvt_f16_f32 v21, v25 // (op 5054) 
v_cvt_f16_f32 v22, v26 // (op 5054) 
v_cvt_f16_f32 v23, v27 // (op 5054) 


v_pack_b32_f16 v16, v20, v21 // (op 5040) 
v_pack_b32_f16 v17, v22, v23 // (op 5040) 

s_lshl_b32 s15, s24, 6 // (op 5040) 
s_add_u32 s15, s15, 32 // (op 5040) 
v_add_u32 v15, v14, s15 // (op 5040) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5040) Store value



v_accvgpr_read v20, a28 // (op 5058) 
v_accvgpr_read v21, a29 // (op 5058) 
v_accvgpr_read v22, a30 // (op 5058) 
v_accvgpr_read v23, a31 // (op 5058) 
v_mul_f32 v24, v19, v20 // (op 5058) 
v_mul_f32 v25, v19, v21 // (op 5058) 
v_mul_f32 v26, v19, v22 // (op 5058) 
v_mul_f32 v27, v19, v23 // (op 5058) 

v_cvt_f16_f32 v20, v24 // (op 5060) 
v_cvt_f16_f32 v21, v25 // (op 5060) 
v_cvt_f16_f32 v22, v26 // (op 5060) 
v_cvt_f16_f32 v23, v27 // (op 5060) 


v_pack_b32_f16 v16, v20, v21 // (op 5042) 
v_pack_b32_f16 v17, v22, v23 // (op 5042) 

s_lshl_b32 s15, s24, 5 // (op 5042) 
s_mul_i32 s6, 3, s15 // (op 5042) 
s_add_u32 s6, s6, 32 // (op 5042) 

v_add_u32 v15, v14, s6 // (op 5042) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5042) Store value



v_accvgpr_read v20, a32 // (op 5108) 
v_accvgpr_read v21, a33 // (op 5108) 
v_accvgpr_read v22, a34 // (op 5108) 
v_accvgpr_read v23, a35 // (op 5108) 
v_mul_f32 v24, v19, v20 // (op 5108) 
v_mul_f32 v25, v19, v21 // (op 5108) 
v_mul_f32 v26, v19, v22 // (op 5108) 
v_mul_f32 v27, v19, v23 // (op 5108) 

v_cvt_f16_f32 v20, v24 // (op 5110) 
v_cvt_f16_f32 v21, v25 // (op 5110) 
v_cvt_f16_f32 v22, v26 // (op 5110) 
v_cvt_f16_f32 v23, v27 // (op 5110) 


v_pack_b32_f16 v16, v20, v21 // (op 5112) 
v_pack_b32_f16 v17, v22, v23 // (op 5112) 

v_add_u32 v15, 64, v14 // (op 5112) 

buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5112) Store value



v_accvgpr_read v20, a36 // (op 5122) 
v_accvgpr_read v21, a37 // (op 5122) 
v_accvgpr_read v22, a38 // (op 5122) 
v_accvgpr_read v23, a39 // (op 5122) 
v_mul_f32 v24, v19, v20 // (op 5122) 
v_mul_f32 v25, v19, v21 // (op 5122) 
v_mul_f32 v26, v19, v22 // (op 5122) 
v_mul_f32 v27, v19, v23 // (op 5122) 

v_cvt_f16_f32 v20, v24 // (op 5124) 
v_cvt_f16_f32 v21, v25 // (op 5124) 
v_cvt_f16_f32 v22, v26 // (op 5124) 
v_cvt_f16_f32 v23, v27 // (op 5124) 


v_pack_b32_f16 v16, v20, v21 // (op 5114) 
v_pack_b32_f16 v17, v22, v23 // (op 5114) 

s_lshl_b32 s15, s24, 5 // (op 5114) 
s_add_u32 s15, s15, 64 // (op 5114) 
v_add_u32 v15, v14, s15 // (op 5114) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5114) Store value



v_accvgpr_read v20, a40 // (op 5128) 
v_accvgpr_read v21, a41 // (op 5128) 
v_accvgpr_read v22, a42 // (op 5128) 
v_accvgpr_read v23, a43 // (op 5128) 
v_mul_f32 v24, v19, v20 // (op 5128) 
v_mul_f32 v25, v19, v21 // (op 5128) 
v_mul_f32 v26, v19, v22 // (op 5128) 
v_mul_f32 v27, v19, v23 // (op 5128) 

v_cvt_f16_f32 v20, v24 // (op 5130) 
v_cvt_f16_f32 v21, v25 // (op 5130) 
v_cvt_f16_f32 v22, v26 // (op 5130) 
v_cvt_f16_f32 v23, v27 // (op 5130) 


v_pack_b32_f16 v16, v20, v21 // (op 5116) 
v_pack_b32_f16 v17, v22, v23 // (op 5116) 

s_lshl_b32 s15, s24, 6 // (op 5116) 
s_add_u32 s15, s15, 64 // (op 5116) 
v_add_u32 v15, v14, s15 // (op 5116) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5116) Store value



v_accvgpr_read v20, a44 // (op 5134) 
v_accvgpr_read v21, a45 // (op 5134) 
v_accvgpr_read v22, a46 // (op 5134) 
v_accvgpr_read v23, a47 // (op 5134) 
v_mul_f32 v24, v19, v20 // (op 5134) 
v_mul_f32 v25, v19, v21 // (op 5134) 
v_mul_f32 v26, v19, v22 // (op 5134) 
v_mul_f32 v27, v19, v23 // (op 5134) 

v_cvt_f16_f32 v20, v24 // (op 5136) 
v_cvt_f16_f32 v21, v25 // (op 5136) 
v_cvt_f16_f32 v22, v26 // (op 5136) 
v_cvt_f16_f32 v23, v27 // (op 5136) 


v_pack_b32_f16 v16, v20, v21 // (op 5118) 
v_pack_b32_f16 v17, v22, v23 // (op 5118) 

s_lshl_b32 s15, s24, 5 // (op 5118) 
s_mul_i32 s6, 3, s15 // (op 5118) 
s_add_u32 s6, s6, 64 // (op 5118) 

v_add_u32 v15, v14, s6 // (op 5118) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5118) Store value



v_accvgpr_read v20, a48 // (op 5184) 
v_accvgpr_read v21, a49 // (op 5184) 
v_accvgpr_read v22, a50 // (op 5184) 
v_accvgpr_read v23, a51 // (op 5184) 
v_mul_f32 v24, v19, v20 // (op 5184) 
v_mul_f32 v25, v19, v21 // (op 5184) 
v_mul_f32 v26, v19, v22 // (op 5184) 
v_mul_f32 v27, v19, v23 // (op 5184) 

v_cvt_f16_f32 v20, v24 // (op 5186) 
v_cvt_f16_f32 v21, v25 // (op 5186) 
v_cvt_f16_f32 v22, v26 // (op 5186) 
v_cvt_f16_f32 v23, v27 // (op 5186) 


v_pack_b32_f16 v16, v20, v21 // (op 5188) 
v_pack_b32_f16 v17, v22, v23 // (op 5188) 

v_add_u32 v15, 96, v14 // (op 5188) 

buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5188) Store value



v_accvgpr_read v20, a52 // (op 5198) 
v_accvgpr_read v21, a53 // (op 5198) 
v_accvgpr_read v22, a54 // (op 5198) 
v_accvgpr_read v23, a55 // (op 5198) 
v_mul_f32 v24, v19, v20 // (op 5198) 
v_mul_f32 v25, v19, v21 // (op 5198) 
v_mul_f32 v26, v19, v22 // (op 5198) 
v_mul_f32 v27, v19, v23 // (op 5198) 

v_cvt_f16_f32 v20, v24 // (op 5200) 
v_cvt_f16_f32 v21, v25 // (op 5200) 
v_cvt_f16_f32 v22, v26 // (op 5200) 
v_cvt_f16_f32 v23, v27 // (op 5200) 


v_pack_b32_f16 v16, v20, v21 // (op 5190) 
v_pack_b32_f16 v17, v22, v23 // (op 5190) 

s_lshl_b32 s15, s24, 5 // (op 5190) 
s_add_u32 s15, s15, 96 // (op 5190) 
v_add_u32 v15, v14, s15 // (op 5190) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5190) Store value



v_accvgpr_read v20, a56 // (op 5204) 
v_accvgpr_read v21, a57 // (op 5204) 
v_accvgpr_read v22, a58 // (op 5204) 
v_accvgpr_read v23, a59 // (op 5204) 
v_mul_f32 v24, v19, v20 // (op 5204) 
v_mul_f32 v25, v19, v21 // (op 5204) 
v_mul_f32 v26, v19, v22 // (op 5204) 
v_mul_f32 v27, v19, v23 // (op 5204) 

v_cvt_f16_f32 v20, v24 // (op 5206) 
v_cvt_f16_f32 v21, v25 // (op 5206) 
v_cvt_f16_f32 v22, v26 // (op 5206) 
v_cvt_f16_f32 v23, v27 // (op 5206) 


v_pack_b32_f16 v16, v20, v21 // (op 5192) 
v_pack_b32_f16 v17, v22, v23 // (op 5192) 

s_lshl_b32 s15, s24, 6 // (op 5192) 
s_add_u32 s15, s15, 96 // (op 5192) 
v_add_u32 v15, v14, s15 // (op 5192) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5192) Store value



v_accvgpr_read v20, a60 // (op 5210) 
v_accvgpr_read v21, a61 // (op 5210) 
v_accvgpr_read v22, a62 // (op 5210) 
v_accvgpr_read v23, a63 // (op 5210) 
v_mul_f32 v24, v19, v20 // (op 5210) 
v_mul_f32 v25, v19, v21 // (op 5210) 
v_mul_f32 v26, v19, v22 // (op 5210) 
v_mul_f32 v27, v19, v23 // (op 5210) 

v_cvt_f16_f32 v20, v24 // (op 5212) 
v_cvt_f16_f32 v21, v25 // (op 5212) 
v_cvt_f16_f32 v22, v26 // (op 5212) 
v_cvt_f16_f32 v23, v27 // (op 5212) 


v_pack_b32_f16 v16, v20, v21 // (op 5194) 
v_pack_b32_f16 v17, v22, v23 // (op 5194) 

s_lshl_b32 s15, s24, 5 // (op 5194) 
s_mul_i32 s6, 3, s15 // (op 5194) 
s_add_u32 s6, s6, 96 // (op 5194) 

v_add_u32 v15, v14, s6 // (op 5194) 


buffer_store_dwordx2 v[16:17], v15, s[60:63], 0 offen // (op 5194) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_207 // (op 207) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_207
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_207:


v_add_u32 v3, 96, v13 // (op 5582) 

buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5582) Load value

v_accvgpr_read v20, a48 // (op 5569) 
v_accvgpr_read v21, a49 // (op 5569) 
v_accvgpr_read v22, a50 // (op 5569) 
v_accvgpr_read v23, a51 // (op 5569) 
v_mul_f32 v24, v19, v20 // (op 5569) 
v_mul_f32 v25, v19, v21 // (op 5569) 
v_mul_f32 v26, v19, v22 // (op 5569) 
v_mul_f32 v27, v19, v23 // (op 5569) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5581) 
v_lshrrev_b32 v21, 16, v16 // (op 5581) 
v_cvt_f32_f16 v21, v21 // (op 5581) 
v_mul_f32 v28, v0, v20 // (op 5581) 
v_mul_f32 v29, v0, v21 // (op 5581) 
v_cvt_f32_f16 v20, v17 // (op 5581) 
v_lshrrev_b32 v21, 16, v17 // (op 5581) 
v_cvt_f32_f16 v21, v21 // (op 5581) 
v_mul_f32 v30, v0, v20 // (op 5581) 
v_mul_f32 v31, v0, v21 // (op 5581) 


v_add_f32 v20, v28, v24 // (op 5570) 
v_add_f32 v21, v29, v25 // (op 5570) 
v_add_f32 v22, v30, v26 // (op 5570) 
v_add_f32 v23, v31, v27 // (op 5570) 


v_cvt_f16_f32 v24, v20 // (op 5571) 
v_cvt_f16_f32 v25, v21 // (op 5571) 
v_cvt_f16_f32 v26, v22 // (op 5571) 
v_cvt_f16_f32 v27, v23 // (op 5571) 


v_pack_b32_f16 v20, v24, v25 // (op 5573) 
v_pack_b32_f16 v21, v26, v27 // (op 5573) 

v_add_u32 v3, 96, v14 // (op 5573) 

buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5573) Store value





s_lshl_b32 s15, s4, 5 // (op 5587) 
s_add_u32 s15, s15, 96 // (op 5587) 
v_add_u32 v3, v13, s15 // (op 5587) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5587) Load value

v_accvgpr_read v20, a52 // (op 5599) 
v_accvgpr_read v21, a53 // (op 5599) 
v_accvgpr_read v22, a54 // (op 5599) 
v_accvgpr_read v23, a55 // (op 5599) 
v_mul_f32 v24, v19, v20 // (op 5599) 
v_mul_f32 v25, v19, v21 // (op 5599) 
v_mul_f32 v26, v19, v22 // (op 5599) 
v_mul_f32 v27, v19, v23 // (op 5599) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5584) 
v_lshrrev_b32 v21, 16, v16 // (op 5584) 
v_cvt_f32_f16 v21, v21 // (op 5584) 
v_mul_f32 v28, v0, v20 // (op 5584) 
v_mul_f32 v29, v0, v21 // (op 5584) 
v_cvt_f32_f16 v20, v17 // (op 5584) 
v_lshrrev_b32 v21, 16, v17 // (op 5584) 
v_cvt_f32_f16 v21, v21 // (op 5584) 
v_mul_f32 v30, v0, v20 // (op 5584) 
v_mul_f32 v31, v0, v21 // (op 5584) 


v_add_f32 v20, v28, v24 // (op 5585) 
v_add_f32 v21, v29, v25 // (op 5585) 
v_add_f32 v22, v30, v26 // (op 5585) 
v_add_f32 v23, v31, v27 // (op 5585) 


v_cvt_f16_f32 v24, v20 // (op 5586) 
v_cvt_f16_f32 v25, v21 // (op 5586) 
v_cvt_f16_f32 v26, v22 // (op 5586) 
v_cvt_f16_f32 v27, v23 // (op 5586) 


v_pack_b32_f16 v20, v24, v25 // (op 5575) 
v_pack_b32_f16 v21, v26, v27 // (op 5575) 

s_lshl_b32 s15, s24, 5 // (op 5575) 
s_add_u32 s15, s15, 96 // (op 5575) 
v_add_u32 v3, v14, s15 // (op 5575) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5575) Store value





s_lshl_b32 s15, s4, 6 // (op 5592) 
s_add_u32 s15, s15, 96 // (op 5592) 
v_add_u32 v3, v13, s15 // (op 5592) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5592) Load value

v_accvgpr_read v20, a56 // (op 5601) 
v_accvgpr_read v21, a57 // (op 5601) 
v_accvgpr_read v22, a58 // (op 5601) 
v_accvgpr_read v23, a59 // (op 5601) 
v_mul_f32 v24, v19, v20 // (op 5601) 
v_mul_f32 v25, v19, v21 // (op 5601) 
v_mul_f32 v26, v19, v22 // (op 5601) 
v_mul_f32 v27, v19, v23 // (op 5601) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5589) 
v_lshrrev_b32 v21, 16, v16 // (op 5589) 
v_cvt_f32_f16 v21, v21 // (op 5589) 
v_mul_f32 v28, v0, v20 // (op 5589) 
v_mul_f32 v29, v0, v21 // (op 5589) 
v_cvt_f32_f16 v20, v17 // (op 5589) 
v_lshrrev_b32 v21, 16, v17 // (op 5589) 
v_cvt_f32_f16 v21, v21 // (op 5589) 
v_mul_f32 v30, v0, v20 // (op 5589) 
v_mul_f32 v31, v0, v21 // (op 5589) 


v_add_f32 v20, v28, v24 // (op 5590) 
v_add_f32 v21, v29, v25 // (op 5590) 
v_add_f32 v22, v30, v26 // (op 5590) 
v_add_f32 v23, v31, v27 // (op 5590) 


v_cvt_f16_f32 v24, v20 // (op 5591) 
v_cvt_f16_f32 v25, v21 // (op 5591) 
v_cvt_f16_f32 v26, v22 // (op 5591) 
v_cvt_f16_f32 v27, v23 // (op 5591) 


v_pack_b32_f16 v20, v24, v25 // (op 5577) 
v_pack_b32_f16 v21, v26, v27 // (op 5577) 

s_lshl_b32 s15, s24, 6 // (op 5577) 
s_add_u32 s15, s15, 96 // (op 5577) 
v_add_u32 v3, v14, s15 // (op 5577) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5577) Store value





s_lshl_b32 s15, s4, 5 // (op 5597) 
s_mul_i32 s6, 3, s15 // (op 5597) 
s_add_u32 s6, s6, 96 // (op 5597) 

v_add_u32 v3, v13, s6 // (op 5597) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5597) Load value

v_accvgpr_read v20, a60 // (op 5603) 
v_accvgpr_read v21, a61 // (op 5603) 
v_accvgpr_read v22, a62 // (op 5603) 
v_accvgpr_read v23, a63 // (op 5603) 
v_mul_f32 v24, v19, v20 // (op 5603) 
v_mul_f32 v25, v19, v21 // (op 5603) 
v_mul_f32 v26, v19, v22 // (op 5603) 
v_mul_f32 v27, v19, v23 // (op 5603) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5594) 
v_lshrrev_b32 v21, 16, v16 // (op 5594) 
v_cvt_f32_f16 v21, v21 // (op 5594) 
v_mul_f32 v28, v0, v20 // (op 5594) 
v_mul_f32 v29, v0, v21 // (op 5594) 
v_cvt_f32_f16 v20, v17 // (op 5594) 
v_lshrrev_b32 v21, 16, v17 // (op 5594) 
v_cvt_f32_f16 v21, v21 // (op 5594) 
v_mul_f32 v30, v0, v20 // (op 5594) 
v_mul_f32 v31, v0, v21 // (op 5594) 


v_add_f32 v20, v28, v24 // (op 5595) 
v_add_f32 v21, v29, v25 // (op 5595) 
v_add_f32 v22, v30, v26 // (op 5595) 
v_add_f32 v23, v31, v27 // (op 5595) 


v_cvt_f16_f32 v24, v20 // (op 5596) 
v_cvt_f16_f32 v25, v21 // (op 5596) 
v_cvt_f16_f32 v26, v22 // (op 5596) 
v_cvt_f16_f32 v27, v23 // (op 5596) 


v_pack_b32_f16 v20, v24, v25 // (op 5579) 
v_pack_b32_f16 v21, v26, v27 // (op 5579) 

s_lshl_b32 s15, s24, 5 // (op 5579) 
s_mul_i32 s6, 3, s15 // (op 5579) 
s_add_u32 s6, s6, 96 // (op 5579) 

v_add_u32 v3, v14, s6 // (op 5579) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5579) Store value





v_add_u32 v3, 64, v13 // (op 5495) 

buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5495) Load value

v_accvgpr_read v20, a32 // (op 5482) 
v_accvgpr_read v21, a33 // (op 5482) 
v_accvgpr_read v22, a34 // (op 5482) 
v_accvgpr_read v23, a35 // (op 5482) 
v_mul_f32 v24, v19, v20 // (op 5482) 
v_mul_f32 v25, v19, v21 // (op 5482) 
v_mul_f32 v26, v19, v22 // (op 5482) 
v_mul_f32 v27, v19, v23 // (op 5482) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5494) 
v_lshrrev_b32 v21, 16, v16 // (op 5494) 
v_cvt_f32_f16 v21, v21 // (op 5494) 
v_mul_f32 v28, v0, v20 // (op 5494) 
v_mul_f32 v29, v0, v21 // (op 5494) 
v_cvt_f32_f16 v20, v17 // (op 5494) 
v_lshrrev_b32 v21, 16, v17 // (op 5494) 
v_cvt_f32_f16 v21, v21 // (op 5494) 
v_mul_f32 v30, v0, v20 // (op 5494) 
v_mul_f32 v31, v0, v21 // (op 5494) 


v_add_f32 v20, v28, v24 // (op 5483) 
v_add_f32 v21, v29, v25 // (op 5483) 
v_add_f32 v22, v30, v26 // (op 5483) 
v_add_f32 v23, v31, v27 // (op 5483) 


v_cvt_f16_f32 v24, v20 // (op 5484) 
v_cvt_f16_f32 v25, v21 // (op 5484) 
v_cvt_f16_f32 v26, v22 // (op 5484) 
v_cvt_f16_f32 v27, v23 // (op 5484) 


v_pack_b32_f16 v20, v24, v25 // (op 5486) 
v_pack_b32_f16 v21, v26, v27 // (op 5486) 

v_add_u32 v3, 64, v14 // (op 5486) 

buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5486) Store value





s_lshl_b32 s15, s4, 5 // (op 5500) 
s_add_u32 s15, s15, 64 // (op 5500) 
v_add_u32 v3, v13, s15 // (op 5500) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5500) Load value

v_accvgpr_read v20, a36 // (op 5512) 
v_accvgpr_read v21, a37 // (op 5512) 
v_accvgpr_read v22, a38 // (op 5512) 
v_accvgpr_read v23, a39 // (op 5512) 
v_mul_f32 v24, v19, v20 // (op 5512) 
v_mul_f32 v25, v19, v21 // (op 5512) 
v_mul_f32 v26, v19, v22 // (op 5512) 
v_mul_f32 v27, v19, v23 // (op 5512) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5497) 
v_lshrrev_b32 v21, 16, v16 // (op 5497) 
v_cvt_f32_f16 v21, v21 // (op 5497) 
v_mul_f32 v28, v0, v20 // (op 5497) 
v_mul_f32 v29, v0, v21 // (op 5497) 
v_cvt_f32_f16 v20, v17 // (op 5497) 
v_lshrrev_b32 v21, 16, v17 // (op 5497) 
v_cvt_f32_f16 v21, v21 // (op 5497) 
v_mul_f32 v30, v0, v20 // (op 5497) 
v_mul_f32 v31, v0, v21 // (op 5497) 


v_add_f32 v20, v28, v24 // (op 5498) 
v_add_f32 v21, v29, v25 // (op 5498) 
v_add_f32 v22, v30, v26 // (op 5498) 
v_add_f32 v23, v31, v27 // (op 5498) 


v_cvt_f16_f32 v24, v20 // (op 5499) 
v_cvt_f16_f32 v25, v21 // (op 5499) 
v_cvt_f16_f32 v26, v22 // (op 5499) 
v_cvt_f16_f32 v27, v23 // (op 5499) 


v_pack_b32_f16 v20, v24, v25 // (op 5488) 
v_pack_b32_f16 v21, v26, v27 // (op 5488) 

s_lshl_b32 s15, s24, 5 // (op 5488) 
s_add_u32 s15, s15, 64 // (op 5488) 
v_add_u32 v3, v14, s15 // (op 5488) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5488) Store value





s_lshl_b32 s15, s4, 6 // (op 5505) 
s_add_u32 s15, s15, 64 // (op 5505) 
v_add_u32 v3, v13, s15 // (op 5505) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5505) Load value

v_accvgpr_read v20, a40 // (op 5514) 
v_accvgpr_read v21, a41 // (op 5514) 
v_accvgpr_read v22, a42 // (op 5514) 
v_accvgpr_read v23, a43 // (op 5514) 
v_mul_f32 v24, v19, v20 // (op 5514) 
v_mul_f32 v25, v19, v21 // (op 5514) 
v_mul_f32 v26, v19, v22 // (op 5514) 
v_mul_f32 v27, v19, v23 // (op 5514) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5502) 
v_lshrrev_b32 v21, 16, v16 // (op 5502) 
v_cvt_f32_f16 v21, v21 // (op 5502) 
v_mul_f32 v28, v0, v20 // (op 5502) 
v_mul_f32 v29, v0, v21 // (op 5502) 
v_cvt_f32_f16 v20, v17 // (op 5502) 
v_lshrrev_b32 v21, 16, v17 // (op 5502) 
v_cvt_f32_f16 v21, v21 // (op 5502) 
v_mul_f32 v30, v0, v20 // (op 5502) 
v_mul_f32 v31, v0, v21 // (op 5502) 


v_add_f32 v20, v28, v24 // (op 5503) 
v_add_f32 v21, v29, v25 // (op 5503) 
v_add_f32 v22, v30, v26 // (op 5503) 
v_add_f32 v23, v31, v27 // (op 5503) 


v_cvt_f16_f32 v24, v20 // (op 5504) 
v_cvt_f16_f32 v25, v21 // (op 5504) 
v_cvt_f16_f32 v26, v22 // (op 5504) 
v_cvt_f16_f32 v27, v23 // (op 5504) 


v_pack_b32_f16 v20, v24, v25 // (op 5490) 
v_pack_b32_f16 v21, v26, v27 // (op 5490) 

s_lshl_b32 s15, s24, 6 // (op 5490) 
s_add_u32 s15, s15, 64 // (op 5490) 
v_add_u32 v3, v14, s15 // (op 5490) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5490) Store value





s_lshl_b32 s15, s4, 5 // (op 5510) 
s_mul_i32 s6, 3, s15 // (op 5510) 
s_add_u32 s6, s6, 64 // (op 5510) 

v_add_u32 v3, v13, s6 // (op 5510) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5510) Load value

v_accvgpr_read v20, a44 // (op 5516) 
v_accvgpr_read v21, a45 // (op 5516) 
v_accvgpr_read v22, a46 // (op 5516) 
v_accvgpr_read v23, a47 // (op 5516) 
v_mul_f32 v24, v19, v20 // (op 5516) 
v_mul_f32 v25, v19, v21 // (op 5516) 
v_mul_f32 v26, v19, v22 // (op 5516) 
v_mul_f32 v27, v19, v23 // (op 5516) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5507) 
v_lshrrev_b32 v21, 16, v16 // (op 5507) 
v_cvt_f32_f16 v21, v21 // (op 5507) 
v_mul_f32 v28, v0, v20 // (op 5507) 
v_mul_f32 v29, v0, v21 // (op 5507) 
v_cvt_f32_f16 v20, v17 // (op 5507) 
v_lshrrev_b32 v21, 16, v17 // (op 5507) 
v_cvt_f32_f16 v21, v21 // (op 5507) 
v_mul_f32 v30, v0, v20 // (op 5507) 
v_mul_f32 v31, v0, v21 // (op 5507) 


v_add_f32 v20, v28, v24 // (op 5508) 
v_add_f32 v21, v29, v25 // (op 5508) 
v_add_f32 v22, v30, v26 // (op 5508) 
v_add_f32 v23, v31, v27 // (op 5508) 


v_cvt_f16_f32 v24, v20 // (op 5509) 
v_cvt_f16_f32 v25, v21 // (op 5509) 
v_cvt_f16_f32 v26, v22 // (op 5509) 
v_cvt_f16_f32 v27, v23 // (op 5509) 


v_pack_b32_f16 v20, v24, v25 // (op 5492) 
v_pack_b32_f16 v21, v26, v27 // (op 5492) 

s_lshl_b32 s15, s24, 5 // (op 5492) 
s_mul_i32 s6, 3, s15 // (op 5492) 
s_add_u32 s6, s6, 64 // (op 5492) 

v_add_u32 v3, v14, s6 // (op 5492) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5492) Store value





v_add_u32 v3, 32, v13 // (op 5408) 

buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5408) Load value

v_accvgpr_read v20, a16 // (op 5395) 
v_accvgpr_read v21, a17 // (op 5395) 
v_accvgpr_read v22, a18 // (op 5395) 
v_accvgpr_read v23, a19 // (op 5395) 
v_mul_f32 v24, v19, v20 // (op 5395) 
v_mul_f32 v25, v19, v21 // (op 5395) 
v_mul_f32 v26, v19, v22 // (op 5395) 
v_mul_f32 v27, v19, v23 // (op 5395) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5407) 
v_lshrrev_b32 v21, 16, v16 // (op 5407) 
v_cvt_f32_f16 v21, v21 // (op 5407) 
v_mul_f32 v28, v0, v20 // (op 5407) 
v_mul_f32 v29, v0, v21 // (op 5407) 
v_cvt_f32_f16 v20, v17 // (op 5407) 
v_lshrrev_b32 v21, 16, v17 // (op 5407) 
v_cvt_f32_f16 v21, v21 // (op 5407) 
v_mul_f32 v30, v0, v20 // (op 5407) 
v_mul_f32 v31, v0, v21 // (op 5407) 


v_add_f32 v20, v28, v24 // (op 5396) 
v_add_f32 v21, v29, v25 // (op 5396) 
v_add_f32 v22, v30, v26 // (op 5396) 
v_add_f32 v23, v31, v27 // (op 5396) 


v_cvt_f16_f32 v24, v20 // (op 5397) 
v_cvt_f16_f32 v25, v21 // (op 5397) 
v_cvt_f16_f32 v26, v22 // (op 5397) 
v_cvt_f16_f32 v27, v23 // (op 5397) 


v_pack_b32_f16 v20, v24, v25 // (op 5399) 
v_pack_b32_f16 v21, v26, v27 // (op 5399) 

v_add_u32 v3, 32, v14 // (op 5399) 

buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5399) Store value





s_lshl_b32 s15, s4, 5 // (op 5413) 
s_add_u32 s15, s15, 32 // (op 5413) 
v_add_u32 v3, v13, s15 // (op 5413) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5413) Load value

v_accvgpr_read v20, a20 // (op 5425) 
v_accvgpr_read v21, a21 // (op 5425) 
v_accvgpr_read v22, a22 // (op 5425) 
v_accvgpr_read v23, a23 // (op 5425) 
v_mul_f32 v24, v19, v20 // (op 5425) 
v_mul_f32 v25, v19, v21 // (op 5425) 
v_mul_f32 v26, v19, v22 // (op 5425) 
v_mul_f32 v27, v19, v23 // (op 5425) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5410) 
v_lshrrev_b32 v21, 16, v16 // (op 5410) 
v_cvt_f32_f16 v21, v21 // (op 5410) 
v_mul_f32 v28, v0, v20 // (op 5410) 
v_mul_f32 v29, v0, v21 // (op 5410) 
v_cvt_f32_f16 v20, v17 // (op 5410) 
v_lshrrev_b32 v21, 16, v17 // (op 5410) 
v_cvt_f32_f16 v21, v21 // (op 5410) 
v_mul_f32 v30, v0, v20 // (op 5410) 
v_mul_f32 v31, v0, v21 // (op 5410) 


v_add_f32 v20, v28, v24 // (op 5411) 
v_add_f32 v21, v29, v25 // (op 5411) 
v_add_f32 v22, v30, v26 // (op 5411) 
v_add_f32 v23, v31, v27 // (op 5411) 


v_cvt_f16_f32 v24, v20 // (op 5412) 
v_cvt_f16_f32 v25, v21 // (op 5412) 
v_cvt_f16_f32 v26, v22 // (op 5412) 
v_cvt_f16_f32 v27, v23 // (op 5412) 


v_pack_b32_f16 v20, v24, v25 // (op 5401) 
v_pack_b32_f16 v21, v26, v27 // (op 5401) 

s_lshl_b32 s15, s24, 5 // (op 5401) 
s_add_u32 s15, s15, 32 // (op 5401) 
v_add_u32 v3, v14, s15 // (op 5401) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5401) Store value





s_lshl_b32 s15, s4, 6 // (op 5418) 
s_add_u32 s15, s15, 32 // (op 5418) 
v_add_u32 v3, v13, s15 // (op 5418) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5418) Load value

v_accvgpr_read v20, a24 // (op 5427) 
v_accvgpr_read v21, a25 // (op 5427) 
v_accvgpr_read v22, a26 // (op 5427) 
v_accvgpr_read v23, a27 // (op 5427) 
v_mul_f32 v24, v19, v20 // (op 5427) 
v_mul_f32 v25, v19, v21 // (op 5427) 
v_mul_f32 v26, v19, v22 // (op 5427) 
v_mul_f32 v27, v19, v23 // (op 5427) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5415) 
v_lshrrev_b32 v21, 16, v16 // (op 5415) 
v_cvt_f32_f16 v21, v21 // (op 5415) 
v_mul_f32 v28, v0, v20 // (op 5415) 
v_mul_f32 v29, v0, v21 // (op 5415) 
v_cvt_f32_f16 v20, v17 // (op 5415) 
v_lshrrev_b32 v21, 16, v17 // (op 5415) 
v_cvt_f32_f16 v21, v21 // (op 5415) 
v_mul_f32 v30, v0, v20 // (op 5415) 
v_mul_f32 v31, v0, v21 // (op 5415) 


v_add_f32 v20, v28, v24 // (op 5416) 
v_add_f32 v21, v29, v25 // (op 5416) 
v_add_f32 v22, v30, v26 // (op 5416) 
v_add_f32 v23, v31, v27 // (op 5416) 


v_cvt_f16_f32 v24, v20 // (op 5417) 
v_cvt_f16_f32 v25, v21 // (op 5417) 
v_cvt_f16_f32 v26, v22 // (op 5417) 
v_cvt_f16_f32 v27, v23 // (op 5417) 


v_pack_b32_f16 v20, v24, v25 // (op 5403) 
v_pack_b32_f16 v21, v26, v27 // (op 5403) 

s_lshl_b32 s15, s24, 6 // (op 5403) 
s_add_u32 s15, s15, 32 // (op 5403) 
v_add_u32 v3, v14, s15 // (op 5403) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5403) Store value





s_lshl_b32 s15, s4, 5 // (op 5423) 
s_mul_i32 s6, 3, s15 // (op 5423) 
s_add_u32 s6, s6, 32 // (op 5423) 

v_add_u32 v3, v13, s6 // (op 5423) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5423) Load value

v_accvgpr_read v20, a28 // (op 5429) 
v_accvgpr_read v21, a29 // (op 5429) 
v_accvgpr_read v22, a30 // (op 5429) 
v_accvgpr_read v23, a31 // (op 5429) 
v_mul_f32 v24, v19, v20 // (op 5429) 
v_mul_f32 v25, v19, v21 // (op 5429) 
v_mul_f32 v26, v19, v22 // (op 5429) 
v_mul_f32 v27, v19, v23 // (op 5429) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5420) 
v_lshrrev_b32 v21, 16, v16 // (op 5420) 
v_cvt_f32_f16 v21, v21 // (op 5420) 
v_mul_f32 v28, v0, v20 // (op 5420) 
v_mul_f32 v29, v0, v21 // (op 5420) 
v_cvt_f32_f16 v20, v17 // (op 5420) 
v_lshrrev_b32 v21, 16, v17 // (op 5420) 
v_cvt_f32_f16 v21, v21 // (op 5420) 
v_mul_f32 v30, v0, v20 // (op 5420) 
v_mul_f32 v31, v0, v21 // (op 5420) 


v_add_f32 v20, v28, v24 // (op 5421) 
v_add_f32 v21, v29, v25 // (op 5421) 
v_add_f32 v22, v30, v26 // (op 5421) 
v_add_f32 v23, v31, v27 // (op 5421) 


v_cvt_f16_f32 v24, v20 // (op 5422) 
v_cvt_f16_f32 v25, v21 // (op 5422) 
v_cvt_f16_f32 v26, v22 // (op 5422) 
v_cvt_f16_f32 v27, v23 // (op 5422) 


v_pack_b32_f16 v20, v24, v25 // (op 5405) 
v_pack_b32_f16 v21, v26, v27 // (op 5405) 

s_lshl_b32 s15, s24, 5 // (op 5405) 
s_mul_i32 s6, 3, s15 // (op 5405) 
s_add_u32 s6, s6, 32 // (op 5405) 

v_add_u32 v3, v14, s6 // (op 5405) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5405) Store value





v_mov_b32 v3, v13 // (op 232) reg expression

buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 232) Load value

v_accvgpr_read v20, a0 // (op 228) 
v_accvgpr_read v21, a1 // (op 228) 
v_accvgpr_read v22, a2 // (op 228) 
v_accvgpr_read v23, a3 // (op 228) 
v_mul_f32 v24, v19, v20 // (op 228) 
v_mul_f32 v25, v19, v21 // (op 228) 
v_mul_f32 v26, v19, v22 // (op 228) 
v_mul_f32 v27, v19, v23 // (op 228) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 233) 
v_lshrrev_b32 v21, 16, v16 // (op 233) 
v_cvt_f32_f16 v21, v21 // (op 233) 
v_mul_f32 v28, v0, v20 // (op 233) 
v_mul_f32 v29, v0, v21 // (op 233) 
v_cvt_f32_f16 v20, v17 // (op 233) 
v_lshrrev_b32 v21, 16, v17 // (op 233) 
v_cvt_f32_f16 v21, v21 // (op 233) 
v_mul_f32 v30, v0, v20 // (op 233) 
v_mul_f32 v31, v0, v21 // (op 233) 


v_add_f32 v20, v28, v24 // (op 229) 
v_add_f32 v21, v29, v25 // (op 229) 
v_add_f32 v22, v30, v26 // (op 229) 
v_add_f32 v23, v31, v27 // (op 229) 


v_cvt_f16_f32 v24, v20 // (op 230) 
v_cvt_f16_f32 v25, v21 // (op 230) 
v_cvt_f16_f32 v26, v22 // (op 230) 
v_cvt_f16_f32 v27, v23 // (op 230) 


v_pack_b32_f16 v20, v24, v25 // (op 231) 
v_pack_b32_f16 v21, v26, v27 // (op 231) 

v_mov_b32 v3, v14 // (op 231) reg expression

buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 231) Store value





s_lshl_b32 s15, s4, 5 // (op 5316) 
v_add_u32 v3, v13, s15 // (op 5316) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5316) Load value

v_accvgpr_read v20, a4 // (op 5312) 
v_accvgpr_read v21, a5 // (op 5312) 
v_accvgpr_read v22, a6 // (op 5312) 
v_accvgpr_read v23, a7 // (op 5312) 
v_mul_f32 v24, v19, v20 // (op 5312) 
v_mul_f32 v25, v19, v21 // (op 5312) 
v_mul_f32 v26, v19, v22 // (op 5312) 
v_mul_f32 v27, v19, v23 // (op 5312) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5317) 
v_lshrrev_b32 v21, 16, v16 // (op 5317) 
v_cvt_f32_f16 v21, v21 // (op 5317) 
v_mul_f32 v28, v0, v20 // (op 5317) 
v_mul_f32 v29, v0, v21 // (op 5317) 
v_cvt_f32_f16 v20, v17 // (op 5317) 
v_lshrrev_b32 v21, 16, v17 // (op 5317) 
v_cvt_f32_f16 v21, v21 // (op 5317) 
v_mul_f32 v30, v0, v20 // (op 5317) 
v_mul_f32 v31, v0, v21 // (op 5317) 


v_add_f32 v20, v28, v24 // (op 5313) 
v_add_f32 v21, v29, v25 // (op 5313) 
v_add_f32 v22, v30, v26 // (op 5313) 
v_add_f32 v23, v31, v27 // (op 5313) 


v_cvt_f16_f32 v24, v20 // (op 5314) 
v_cvt_f16_f32 v25, v21 // (op 5314) 
v_cvt_f16_f32 v26, v22 // (op 5314) 
v_cvt_f16_f32 v27, v23 // (op 5314) 


v_pack_b32_f16 v20, v24, v25 // (op 5315) 
v_pack_b32_f16 v21, v26, v27 // (op 5315) 

s_lshl_b32 s15, s24, 5 // (op 5315) 
v_add_u32 v3, v14, s15 // (op 5315) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5315) Store value





s_lshl_b32 s15, s4, 6 // (op 5330) 
v_add_u32 v3, v13, s15 // (op 5330) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5330) Load value

v_accvgpr_read v20, a8 // (op 5326) 
v_accvgpr_read v21, a9 // (op 5326) 
v_accvgpr_read v22, a10 // (op 5326) 
v_accvgpr_read v23, a11 // (op 5326) 
v_mul_f32 v24, v19, v20 // (op 5326) 
v_mul_f32 v25, v19, v21 // (op 5326) 
v_mul_f32 v26, v19, v22 // (op 5326) 
v_mul_f32 v27, v19, v23 // (op 5326) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5331) 
v_lshrrev_b32 v21, 16, v16 // (op 5331) 
v_cvt_f32_f16 v21, v21 // (op 5331) 
v_mul_f32 v28, v0, v20 // (op 5331) 
v_mul_f32 v29, v0, v21 // (op 5331) 
v_cvt_f32_f16 v20, v17 // (op 5331) 
v_lshrrev_b32 v21, 16, v17 // (op 5331) 
v_cvt_f32_f16 v21, v21 // (op 5331) 
v_mul_f32 v30, v0, v20 // (op 5331) 
v_mul_f32 v31, v0, v21 // (op 5331) 


v_add_f32 v20, v28, v24 // (op 5327) 
v_add_f32 v21, v29, v25 // (op 5327) 
v_add_f32 v22, v30, v26 // (op 5327) 
v_add_f32 v23, v31, v27 // (op 5327) 


v_cvt_f16_f32 v24, v20 // (op 5328) 
v_cvt_f16_f32 v25, v21 // (op 5328) 
v_cvt_f16_f32 v26, v22 // (op 5328) 
v_cvt_f16_f32 v27, v23 // (op 5328) 


v_pack_b32_f16 v20, v24, v25 // (op 5329) 
v_pack_b32_f16 v21, v26, v27 // (op 5329) 

s_lshl_b32 s15, s24, 6 // (op 5329) 
v_add_u32 v3, v14, s15 // (op 5329) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5329) Store value





s_lshl_b32 s15, s4, 5 // (op 5344) 
s_mul_i32 s6, 3, s15 // (op 5344) 

v_add_u32 v3, v13, s6 // (op 5344) 


buffer_load_dwordx2 v[16:17], v3, s[56:59], 0 offen // (op 5344) Load value

v_accvgpr_read v20, a12 // (op 5340) 
v_accvgpr_read v21, a13 // (op 5340) 
v_accvgpr_read v22, a14 // (op 5340) 
v_accvgpr_read v23, a15 // (op 5340) 
v_mul_f32 v24, v19, v20 // (op 5340) 
v_mul_f32 v25, v19, v21 // (op 5340) 
v_mul_f32 v26, v19, v22 // (op 5340) 
v_mul_f32 v27, v19, v23 // (op 5340) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v16 // (op 5345) 
v_lshrrev_b32 v21, 16, v16 // (op 5345) 
v_cvt_f32_f16 v21, v21 // (op 5345) 
v_mul_f32 v28, v0, v20 // (op 5345) 
v_mul_f32 v29, v0, v21 // (op 5345) 
v_cvt_f32_f16 v20, v17 // (op 5345) 
v_lshrrev_b32 v21, 16, v17 // (op 5345) 
v_cvt_f32_f16 v21, v21 // (op 5345) 
v_mul_f32 v30, v0, v20 // (op 5345) 
v_mul_f32 v31, v0, v21 // (op 5345) 


v_add_f32 v20, v28, v24 // (op 5341) 
v_add_f32 v21, v29, v25 // (op 5341) 
v_add_f32 v22, v30, v26 // (op 5341) 
v_add_f32 v23, v31, v27 // (op 5341) 


v_cvt_f16_f32 v24, v20 // (op 5342) 
v_cvt_f16_f32 v25, v21 // (op 5342) 
v_cvt_f16_f32 v26, v22 // (op 5342) 
v_cvt_f16_f32 v27, v23 // (op 5342) 


v_pack_b32_f16 v20, v24, v25 // (op 5343) 
v_pack_b32_f16 v21, v26, v27 // (op 5343) 

s_lshl_b32 s15, s24, 5 // (op 5343) 
s_mul_i32 s6, 3, s15 // (op 5343) 

v_add_u32 v3, v14, s6 // (op 5343) 


buffer_store_dwordx2 v[20:21], v3, s[60:63], 0 offen // (op 5343) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_207:
s_add_u32 s0, s0, s1 // (op 420) 


s_mul_i32 s15, s50, s16 // (op 418) 
s_cmp_gt_u32 s51, s15 // (op 418) 
s_mov_b32 s6, scc // (op 418) 
s_sub_u32 s7, s51, s15 // (op 418) 

s_cmp_eq_i32 s6, 0 // (op 418) 
s_cselect_b32 s8, 0, s7 // (op 418) 


s_cmp_lt_u32 s50, s8 // (op 418) 
s_mov_b32 s15, scc // (op 418) 
s_cmp_eq_i32 s15, 0 // (op 418) 
s_cselect_b32 s6, s8, s50 // (op 418) 


s_cmp_lt_u32 s0, s6 // (op 418) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_418 // (op 418) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_418 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_418:























































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_end:
.size RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4, .LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_end-RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
  .amdhsa_next_free_vgpr 184
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 65536 // lds bytes
  .amdhsa_accum_offset 120
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
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 2
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 4
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
    .symbol: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 92
    .vgpr_count: 118
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

