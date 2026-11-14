// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 21457) 
v_and_b32 v3, 15, v1 // (op 21457) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 21457) 


v_and_b32 v0, 63, v1 // (op 21457) 
v_lshrrev_b32 v3, 4, v0 // (op 21457) 

v_lshlrev_b32 v0, 5, v3 // (op 21457) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 21457) 


v_lshlrev_b32 v0, 2, v3 // (op 21457) 

v_lshrrev_b32 v3, 3, v0 // (op 21457) 

v_lshrrev_b32 v0, 6, v1 // (op 21532) 
v_and_b32 v4, 1, v0 // (op 21532) 

v_and_b32 v0, 15, v1 // (op 21532) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 21532) 


v_and_b32 v0, 63, v1 // (op 21532) 
v_lshrrev_b32 v4, 4, v0 // (op 21532) 

v_lshlrev_b32 v0, 5, v4 // (op 21532) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 21532) 


v_lshlrev_b32 v0, 2, v4 // (op 21532) 

v_lshrrev_b32 v4, 3, v0 // (op 21532) 

v_lshrrev_b32 v0, 7, v1 // (op 21659) 
v_and_b32 v5, 15, v1 // (op 21659) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 21659) 


v_and_b32 v0, 63, v1 // (op 21659) 
v_lshrrev_b32 v5, 4, v0 // (op 21659) 

v_lshlrev_b32 v0, 5, v5 // (op 21659) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 21659) 


v_lshlrev_b32 v0, 2, v5 // (op 21659) 

v_lshrrev_b32 v5, 3, v0 // (op 21659) 

v_lshrrev_b32 v0, 6, v1 // (op 21734) 
v_and_b32 v6, 1, v0 // (op 21734) 

v_and_b32 v0, 15, v1 // (op 21734) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 21734) 


v_and_b32 v0, 63, v1 // (op 21734) 
v_lshrrev_b32 v6, 4, v0 // (op 21734) 

v_lshlrev_b32 v0, 5, v6 // (op 21734) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 21734) 


v_lshlrev_b32 v0, 2, v6 // (op 21734) 

v_lshrrev_b32 v6, 3, v0 // (op 21734) 

v_lshrrev_b32 v0, 7, v1 // (op 21848) 
v_and_b32 v7, 15, v1 // (op 21848) 
v_lshl_add_u32 v8, v0, 6, v7 // (op 21848) 


v_and_b32 v0, 63, v1 // (op 21848) 
v_lshrrev_b32 v7, 4, v0 // (op 21848) 

v_lshlrev_b32 v0, 5, v7 // (op 21848) 

v_lshl_add_u32 v7, v8, 7, v0 // (op 21848) 


v_lshlrev_b32 v0, 2, v7 // (op 21848) 

v_lshrrev_b32 v7, 3, v0 // (op 21848) 

v_lshrrev_b32 v0, 6, v1 // (op 21892) 
v_and_b32 v8, 1, v0 // (op 21892) 

v_and_b32 v0, 15, v1 // (op 21892) 
v_lshl_add_u32 v9, v8, 6, v0 // (op 21892) 


v_and_b32 v0, 63, v1 // (op 21892) 
v_lshrrev_b32 v8, 4, v0 // (op 21892) 

v_lshlrev_b32 v0, 5, v8 // (op 21892) 

v_lshl_add_u32 v8, v9, 7, v0 // (op 21892) 


v_lshlrev_b32 v0, 2, v8 // (op 21892) 

v_lshrrev_b32 v8, 3, v0 // (op 21892) 

v_lshrrev_b32 v0, 7, v1 // (op 21936) 
v_and_b32 v9, 15, v1 // (op 21936) 
v_lshl_add_u32 v10, v0, 6, v9 // (op 21936) 


v_and_b32 v0, 63, v1 // (op 21936) 
v_lshrrev_b32 v9, 4, v0 // (op 21936) 

v_lshlrev_b32 v0, 5, v9 // (op 21936) 

v_lshl_add_u32 v9, v10, 7, v0 // (op 21936) 


v_lshlrev_b32 v0, 2, v9 // (op 21936) 

v_lshrrev_b32 v9, 3, v0 // (op 21936) 

v_lshrrev_b32 v0, 6, v1 // (op 21980) 
v_and_b32 v10, 1, v0 // (op 21980) 

v_and_b32 v0, 15, v1 // (op 21980) 
v_lshl_add_u32 v11, v10, 6, v0 // (op 21980) 


v_and_b32 v0, 63, v1 // (op 21980) 
v_lshrrev_b32 v10, 4, v0 // (op 21980) 

v_lshlrev_b32 v0, 5, v10 // (op 21980) 

v_lshl_add_u32 v10, v11, 7, v0 // (op 21980) 


v_lshlrev_b32 v0, 2, v10 // (op 21980) 

v_lshrrev_b32 v10, 3, v0 // (op 21980) 

v_lshrrev_b32 v0, 7, v1 // (op 22024) 
v_and_b32 v11, 15, v1 // (op 22024) 
v_lshl_add_u32 v12, v0, 6, v11 // (op 22024) 


v_and_b32 v0, 63, v1 // (op 22024) 
v_lshrrev_b32 v11, 4, v0 // (op 22024) 

v_lshlrev_b32 v0, 5, v11 // (op 22024) 

v_lshl_add_u32 v11, v12, 7, v0 // (op 22024) 


v_lshlrev_b32 v0, 2, v11 // (op 22024) 

v_lshrrev_b32 v11, 3, v0 // (op 22024) 

v_lshrrev_b32 v0, 6, v1 // (op 22068) 
v_and_b32 v12, 1, v0 // (op 22068) 

v_and_b32 v0, 15, v1 // (op 22068) 
v_lshl_add_u32 v13, v12, 6, v0 // (op 22068) 


v_and_b32 v0, 63, v1 // (op 22068) 
v_lshrrev_b32 v12, 4, v0 // (op 22068) 

v_lshlrev_b32 v0, 5, v12 // (op 22068) 

v_lshl_add_u32 v12, v13, 7, v0 // (op 22068) 


v_lshlrev_b32 v0, 2, v12 // (op 22068) 

v_lshrrev_b32 v12, 3, v0 // (op 22068) 

v_lshrrev_b32 v0, 7, v1 // (op 22112) 
v_and_b32 v13, 15, v1 // (op 22112) 
v_lshl_add_u32 v14, v0, 6, v13 // (op 22112) 


v_and_b32 v0, 63, v1 // (op 22112) 
v_lshrrev_b32 v13, 4, v0 // (op 22112) 

v_lshlrev_b32 v0, 5, v13 // (op 22112) 

v_lshl_add_u32 v13, v14, 7, v0 // (op 22112) 


v_lshlrev_b32 v0, 2, v13 // (op 22112) 

v_lshrrev_b32 v13, 3, v0 // (op 22112) 

v_lshrrev_b32 v0, 6, v1 // (op 22156) 
v_and_b32 v14, 1, v0 // (op 22156) 

v_and_b32 v0, 15, v1 // (op 22156) 
v_lshl_add_u32 v15, v14, 6, v0 // (op 22156) 


v_and_b32 v0, 63, v1 // (op 22156) 
v_lshrrev_b32 v14, 4, v0 // (op 22156) 

v_lshlrev_b32 v0, 5, v14 // (op 22156) 

v_lshl_add_u32 v14, v15, 7, v0 // (op 22156) 


v_lshlrev_b32 v0, 2, v14 // (op 22156) 

v_lshrrev_b32 v14, 3, v0 // (op 22156) 

v_lshrrev_b32 v0, 7, v1 // (op 22200) 
v_and_b32 v15, 15, v1 // (op 22200) 
v_lshl_add_u32 v16, v0, 6, v15 // (op 22200) 


v_and_b32 v0, 63, v1 // (op 22200) 
v_lshrrev_b32 v15, 4, v0 // (op 22200) 

v_lshlrev_b32 v0, 5, v15 // (op 22200) 

v_lshl_add_u32 v15, v16, 7, v0 // (op 22200) 


v_lshlrev_b32 v0, 2, v15 // (op 22200) 

v_lshrrev_b32 v15, 3, v0 // (op 22200) 

v_lshrrev_b32 v0, 6, v1 // (op 22244) 
v_and_b32 v16, 1, v0 // (op 22244) 

v_and_b32 v0, 15, v1 // (op 22244) 
v_lshl_add_u32 v17, v16, 6, v0 // (op 22244) 


v_and_b32 v0, 63, v1 // (op 22244) 
v_lshrrev_b32 v16, 4, v0 // (op 22244) 

v_lshlrev_b32 v0, 5, v16 // (op 22244) 

v_lshl_add_u32 v16, v17, 7, v0 // (op 22244) 


v_lshlrev_b32 v0, 2, v16 // (op 22244) 

v_lshrrev_b32 v16, 3, v0 // (op 22244) 

v_lshrrev_b32 v0, 7, v1 // (op 22288) 
v_and_b32 v17, 15, v1 // (op 22288) 
v_lshl_add_u32 v18, v0, 6, v17 // (op 22288) 


v_and_b32 v0, 63, v1 // (op 22288) 
v_lshrrev_b32 v17, 4, v0 // (op 22288) 

v_lshlrev_b32 v0, 5, v17 // (op 22288) 

v_lshl_add_u32 v17, v18, 7, v0 // (op 22288) 


v_lshlrev_b32 v0, 2, v17 // (op 22288) 

v_lshrrev_b32 v17, 3, v0 // (op 22288) 

v_lshrrev_b32 v0, 6, v1 // (op 22332) 
v_and_b32 v18, 1, v0 // (op 22332) 

v_and_b32 v0, 15, v1 // (op 22332) 
v_lshl_add_u32 v19, v18, 6, v0 // (op 22332) 


v_and_b32 v0, 63, v1 // (op 22332) 
v_lshrrev_b32 v18, 4, v0 // (op 22332) 

v_lshlrev_b32 v0, 5, v18 // (op 22332) 

v_lshl_add_u32 v18, v19, 7, v0 // (op 22332) 


v_lshlrev_b32 v0, 2, v18 // (op 22332) 

v_lshrrev_b32 v18, 3, v0 // (op 22332) 

v_mov_b32 v0, s6 // (op 2) Move value
v_mov_b32 v19, s7 // (op 35) Move value


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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_255 // (op 255) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_255 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_255:
v_lshrrev_b32 v20, 7, v1 // (op 21368) 
s_mul_i32 s52, s16, s8 // (op 21368) 
s_add_u32 s52, s52, s0 // (op 21368) 
v_and_b32 v21, 63, v1 // (op 21368) 
s_mul_hi_u32 s53, s52, s13 // (op 21368) 
v_lshrrev_b32 v22, 4, v21 // (op 21368) 
s_sub_u32 s54, s52, s53 // (op 21368) 
v_lshlrev_b32 v23, 2, v22 // (op 21368) 

v_lshl_add_u32 v22, v20, 6, v23 // (op 21368) 


s_lshr_b32 s55, s54, 1 // (op 21368) 

s_add_u32 s54, s55, s53 // (op 21368) 


v_lshrrev_b32 v20, 2, v21 // (op 21368) 

s_lshr_b32 s53, s54, s11 // (op 21368) 

v_and_b32 v21, 3, v20 // (op 21368) 

s_and_b32 s54, s11, 2147483648 // (op 21368) 
v_and_b32 v20, 3, v1 // (op 21368) 
s_cmp_eq_i32 s54, 0 // (op 21368) 
s_mov_b32 s55, scc // (op 21368) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 21368) 


s_cmp_eq_i32 s55, 0 // (op 21368) 
s_cselect_b32 s54, s52, s53 // (op 21368) 



v_lshrrev_b32 v20, 6, v1 // (op 21368) 
s_mul_hi_u32 s52, s54, s12 // (op 21368) 
v_and_b32 v21, 1, v20 // (op 21368) 

s_sub_u32 s53, s54, s52 // (op 21368) 
v_lshl_add_u32 v20, v21, 6, v23 // (op 21368) 


s_lshr_b32 s55, s53, 1 // (op 21368) 

s_add_u32 s53, s55, s52 // (op 21368) 


s_lshr_b32 s52, s53, s10 // (op 21368) 

s_and_b32 s53, s10, 2147483648 // (op 21368) 
s_cmp_eq_i32 s53, 0 // (op 21368) 
s_mov_b32 s55, scc // (op 21368) 

s_cmp_eq_i32 s55, 0 // (op 21368) 
s_cselect_b32 s53, s54, s52 // (op 21368) 


s_mul_i32 s52, s53, s14 // (op 21368) 
v_lshl_add_u32 v21, s53, 7, v22 // (op 21368) 


s_sub_u32 s53, s54, s52 // (op 21368) 


v_lshl_add_u32 v22, s53, 7, v20 // (op 21368) 


v_mul_lo_u32 v20, v22, s4 // (op 21368) 

v_add_lshl_u32 v22, v21, v20, 2 // (op 21368) 


s_mul_i32 s53, s4, s23 // (op 21370) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 21370) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 21370) least significant
s_mul_hi_u32 s55, s4, s22 // (op 21370) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21370) most significant: sum
s_add_u32 s53, s53, s55 // (op 21370) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21370) 

s_mov_b32 s56, s20 // (op 21370) reg expression
s_mov_b32 s57, s21 // (op 21370) reg expression
s_mov_b32 s58, s54 // (op 21370) reg expression
s_mov_b32 s59, 131072 // (op 21370) call()

v_lshrrev_b32 v20, 7, v1 // (op 21474) 
v_and_b32 v21, 15, v1 // (op 21474) 
v_lshl_add_u32 v23, v20, 6, v21 // (op 21474) 


v_and_b32 v20, 63, v1 // (op 21474) 
v_lshrrev_b32 v21, 4, v20 // (op 21474) 

v_lshlrev_b32 v20, 5, v21 // (op 21474) 

v_lshl_add_u32 v21, v23, 7, v20 // (op 21474) 


v_lshlrev_b32 v20, 2, v21 // (op 21474) 

v_lshrrev_b32 v21, 3, v20 // (op 21474) 

v_lshrrev_b32 v20, 6, v1 // (op 21549) 
v_and_b32 v23, 1, v20 // (op 21549) 

v_and_b32 v20, 15, v1 // (op 21549) 
v_lshl_add_u32 v24, v23, 6, v20 // (op 21549) 


v_and_b32 v20, 63, v1 // (op 21549) 
v_lshrrev_b32 v23, 4, v20 // (op 21549) 

v_lshlrev_b32 v20, 5, v23 // (op 21549) 

v_lshl_add_u32 v23, v24, 7, v20 // (op 21549) 


v_lshlrev_b32 v20, 2, v23 // (op 21549) 

v_lshrrev_b32 v23, 3, v20 // (op 21549) 

v_lshrrev_b32 v20, 7, v1 // (op 21572) 
s_mul_i32 s52, s16, s8 // (op 21572) 
s_add_u32 s52, s52, s0 // (op 21572) 
v_and_b32 v24, 63, v1 // (op 21572) 
s_mul_hi_u32 s53, s52, s13 // (op 21572) 
v_lshrrev_b32 v25, 4, v24 // (op 21572) 
s_sub_u32 s54, s52, s53 // (op 21572) 
v_lshlrev_b32 v26, 2, v25 // (op 21572) 

s_lshr_b32 s55, s54, 1 // (op 21572) 

v_lshl_add_u32 v25, v20, 6, v26 // (op 21572) 


s_add_u32 s54, s55, s53 // (op 21572) 


v_lshrrev_b32 v20, 2, v24 // (op 21572) 

s_lshr_b32 s53, s54, s11 // (op 21572) 

v_and_b32 v24, 3, v20 // (op 21572) 

s_and_b32 s54, s11, 2147483648 // (op 21572) 
v_and_b32 v20, 3, v1 // (op 21572) 
s_cmp_eq_i32 s54, 0 // (op 21572) 
s_mov_b32 s55, scc // (op 21572) 

v_lshl_add_u32 v26, v24, 2, v20 // (op 21572) 


s_cmp_eq_i32 s55, 0 // (op 21572) 
s_cselect_b32 s54, s52, s53 // (op 21572) 



v_lshrrev_b32 v20, 6, v1 // (op 21572) 
s_mul_hi_u32 s55, s54, s12 // (op 21572) 
v_and_b32 v24, 1, v20 // (op 21572) 

s_sub_u32 s52, s54, s55 // (op 21572) 
v_lshl_add_u32 v20, v24, 6, v26 // (op 21572) 


s_lshr_b32 s53, s52, 1 // (op 21572) 

s_add_u32 s52, s53, s55 // (op 21572) 


s_lshr_b32 s53, s52, s10 // (op 21572) 

s_and_b32 s52, s10, 2147483648 // (op 21572) 
s_cmp_eq_i32 s52, 0 // (op 21572) 
s_mov_b32 s55, scc // (op 21572) 

s_cmp_eq_i32 s55, 0 // (op 21572) 
s_cselect_b32 s52, s54, s53 // (op 21572) 


s_mul_i32 s53, s52, s14 // (op 21572) 
v_lshl_add_u32 v24, s52, 7, v25 // (op 21572) 


s_sub_u32 s52, s54, s53 // (op 21572) 


v_lshl_add_u32 v25, s52, 7, v20 // (op 21572) 


v_mul_lo_u32 v20, v25, s24 // (op 21572) 

v_add_lshl_u32 v25, v24, v20, 2 // (op 21572) 


s_mul_i32 s53, s24, s23 // (op 21574) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 21574) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 21574) least significant
s_mul_hi_u32 s55, s24, s22 // (op 21574) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21574) most significant: sum
s_add_u32 s53, s53, s55 // (op 21574) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21574) 

s_mov_b32 s60, s26 // (op 21574) reg expression
s_mov_b32 s61, s27 // (op 21574) reg expression
s_mov_b32 s62, s54 // (op 21574) reg expression
s_mov_b32 s63, 131072 // (op 21574) call()

v_lshrrev_b32 v20, 5, v1 // (op 21774) 
v_lshlrev_b32 v24, 2, v20 // (op 21774) 

v_lshl_add_u32 v20, s16, 7, v24 // (op 21774) 

v_and_b32 v24, -32, v1 // (op 21774) 
v_sub_u32 v26, v1, v24 // (op 21774) 

v_lshl_add_u32 v24, v20, 7, v26 // (op 21774) 


v_lshlrev_b32 v20, 2, v24 // (op 21774) 

s_lshl_b32 s52, s15, 16 // (op 21776) 
s_mov_b32 s64, s28 // (op 21776) reg expression
s_mov_b32 s65, s29 // (op 21776) reg expression
s_mov_b32 s66, s52 // (op 21776) reg expression
s_mov_b32 s67, 131072 // (op 21776) call()

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
v_accvgpr_write a4, 0.00000 // (op 5921) call()
v_accvgpr_write a5, 0.00000 // (op 5921) call()
v_accvgpr_write a6, 0.00000 // (op 5921) call()
v_accvgpr_write a7, 0.00000 // (op 5921) call()
v_accvgpr_write a8, 0.00000 // (op 6147) call()
v_accvgpr_write a9, 0.00000 // (op 6147) call()
v_accvgpr_write a10, 0.00000 // (op 6147) call()
v_accvgpr_write a11, 0.00000 // (op 6147) call()
v_accvgpr_write a12, 0.00000 // (op 6373) call()
v_accvgpr_write a13, 0.00000 // (op 6373) call()
v_accvgpr_write a14, 0.00000 // (op 6373) call()
v_accvgpr_write a15, 0.00000 // (op 6373) call()
v_accvgpr_write a16, 0.00000 // (op 6735) call()
v_accvgpr_write a17, 0.00000 // (op 6735) call()
v_accvgpr_write a18, 0.00000 // (op 6735) call()
v_accvgpr_write a19, 0.00000 // (op 6735) call()
v_accvgpr_write a20, 0.00000 // (op 7148) call()
v_accvgpr_write a21, 0.00000 // (op 7148) call()
v_accvgpr_write a22, 0.00000 // (op 7148) call()
v_accvgpr_write a23, 0.00000 // (op 7148) call()
v_accvgpr_write a24, 0.00000 // (op 7149) call()
v_accvgpr_write a25, 0.00000 // (op 7149) call()
v_accvgpr_write a26, 0.00000 // (op 7149) call()
v_accvgpr_write a27, 0.00000 // (op 7149) call()
v_accvgpr_write a28, 0.00000 // (op 7150) call()
v_accvgpr_write a29, 0.00000 // (op 7150) call()
v_accvgpr_write a30, 0.00000 // (op 7150) call()
v_accvgpr_write a31, 0.00000 // (op 7150) call()
v_accvgpr_write a32, 0.00000 // (op 7737) call()
v_accvgpr_write a33, 0.00000 // (op 7737) call()
v_accvgpr_write a34, 0.00000 // (op 7737) call()
v_accvgpr_write a35, 0.00000 // (op 7737) call()
v_accvgpr_write a36, 0.00000 // (op 8150) call()
v_accvgpr_write a37, 0.00000 // (op 8150) call()
v_accvgpr_write a38, 0.00000 // (op 8150) call()
v_accvgpr_write a39, 0.00000 // (op 8150) call()
v_accvgpr_write a40, 0.00000 // (op 8151) call()
v_accvgpr_write a41, 0.00000 // (op 8151) call()
v_accvgpr_write a42, 0.00000 // (op 8151) call()
v_accvgpr_write a43, 0.00000 // (op 8151) call()
v_accvgpr_write a44, 0.00000 // (op 8152) call()
v_accvgpr_write a45, 0.00000 // (op 8152) call()
v_accvgpr_write a46, 0.00000 // (op 8152) call()
v_accvgpr_write a47, 0.00000 // (op 8152) call()
v_accvgpr_write a48, 0.00000 // (op 8739) call()
v_accvgpr_write a49, 0.00000 // (op 8739) call()
v_accvgpr_write a50, 0.00000 // (op 8739) call()
v_accvgpr_write a51, 0.00000 // (op 8739) call()
v_accvgpr_write a52, 0.00000 // (op 9152) call()
v_accvgpr_write a53, 0.00000 // (op 9152) call()
v_accvgpr_write a54, 0.00000 // (op 9152) call()
v_accvgpr_write a55, 0.00000 // (op 9152) call()
v_accvgpr_write a56, 0.00000 // (op 9153) call()
v_accvgpr_write a57, 0.00000 // (op 9153) call()
v_accvgpr_write a58, 0.00000 // (op 9153) call()
v_accvgpr_write a59, 0.00000 // (op 9153) call()
v_accvgpr_write a60, 0.00000 // (op 9154) call()
v_accvgpr_write a61, 0.00000 // (op 9154) call()
v_accvgpr_write a62, 0.00000 // (op 9154) call()
v_accvgpr_write a63, 0.00000 // (op 9154) call()
v_lshrrev_b32 v24, 2, v1 // (op 21191) 
s_mul_i32 s52, s16, s8 // (op 21191) 
s_add_u32 s52, s52, s0 // (op 21191) 
v_and_b32 v26, -4, v1 // (op 21191) 
s_mul_hi_u32 s53, s52, s13 // (op 21191) 
v_sub_u32 v27, v1, v26 // (op 21191) 

s_sub_u32 s54, s52, s53 // (op 21191) 
v_lshlrev_b32 v26, 5, v27 // (op 21191) 

s_lshr_b32 s55, s54, 1 // (op 21191) 

s_add_u32 s54, s55, s53 // (op 21191) 


s_lshr_b32 s53, s54, s11 // (op 21191) 

s_and_b32 s54, s11, 2147483648 // (op 21191) 
s_cmp_eq_i32 s54, 0 // (op 21191) 
s_mov_b32 s55, scc // (op 21191) 

s_cmp_eq_i32 s55, 0 // (op 21191) 
s_cselect_b32 s54, s52, s53 // (op 21191) 


s_mul_hi_u32 s53, s54, s12 // (op 21191) 
s_sub_u32 s55, s54, s53 // (op 21191) 
s_lshr_b32 s68, s55, 1 // (op 21191) 

s_add_u32 s55, s68, s53 // (op 21191) 


s_lshr_b32 s53, s55, s10 // (op 21191) 

s_and_b32 s55, s10, 2147483648 // (op 21191) 
s_cmp_eq_i32 s55, 0 // (op 21191) 
s_mov_b32 s68, scc // (op 21191) 

s_cmp_eq_i32 s68, 0 // (op 21191) 
s_cselect_b32 s55, s54, s53 // (op 21191) 


v_lshl_add_u32 v27, s55, 7, v24 // (op 21191) 


s_ashr_i64 s[68:69], s[2:3], 7 // (op 21191) 
s_mul_i32 s53, s54, s68 // (op 21191) 


s_sub_u32 s54, s52, s53 // (op 21191) 


v_lshl_add_u32 v24, s54, 7, v26 // (op 21191) 


v_mov_b32 v29, 0 // (op 21191) convert
v_mov_b32 v28, v24 // (op 21191) convert

v_mul_lo_u32 v31, v27, s31 // (op 21191) most significant: low of low * high
v_mul_hi_u32 v24, v27, s30 // (op 21191) most significant: high of low * low
v_mul_lo_u32 v30, v27, s30 // (op 21191) least significant: low of low * low
v_add_u32 v31, v31, v24 // (op 21191) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v28 // (op 21191) least significant half
v_addc_co_u32 v31, s[52:53], v31, v29, s[52:53] // (op 21191) most significant half



v_lshlrev_b64 v[26:27], 2, v[30:31] // (op 21191) 

v_lshrrev_b64 v[28:29], 3, v[26:27] // (op 21191) 

s_mul_i32 s53, s30, s35 // (op 21193) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 21193) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 21193) least significant
s_mul_hi_u32 s55, s30, s34 // (op 21193) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21193) most significant: sum
s_add_u32 s53, s53, s55 // (op 21193) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21193) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21193) 

s_mov_b32 s68, s32 // (op 21193) reg expression
s_mov_b32 s69, s33 // (op 21193) reg expression
s_mov_b32 s70, s52 // (op 21193) reg expression
s_mov_b32 s71, 131072 // (op 21193) call()

v_lshrrev_b32 v24, 2, v1 // (op 21210) 
s_mul_i32 s52, s16, s8 // (op 21210) 
s_add_u32 s52, s52, s0 // (op 21210) 
v_and_b32 v26, -4, v1 // (op 21210) 
s_mul_hi_u32 s53, s52, s13 // (op 21210) 
v_sub_u32 v27, v1, v26 // (op 21210) 

s_sub_u32 s54, s52, s53 // (op 21210) 
v_lshlrev_b32 v26, 5, v27 // (op 21210) 

s_lshr_b32 s55, s54, 1 // (op 21210) 

s_add_u32 s54, s55, s53 // (op 21210) 


s_lshr_b32 s53, s54, s11 // (op 21210) 

s_and_b32 s54, s11, 2147483648 // (op 21210) 
s_cmp_eq_i32 s54, 0 // (op 21210) 
s_mov_b32 s55, scc // (op 21210) 

s_cmp_eq_i32 s55, 0 // (op 21210) 
s_cselect_b32 s54, s52, s53 // (op 21210) 


s_mul_hi_u32 s53, s54, s12 // (op 21210) 
s_sub_u32 s55, s54, s53 // (op 21210) 
s_lshr_b32 s72, s55, 1 // (op 21210) 

s_add_u32 s55, s72, s53 // (op 21210) 


s_lshr_b32 s53, s55, s10 // (op 21210) 

s_and_b32 s55, s10, 2147483648 // (op 21210) 
s_cmp_eq_i32 s55, 0 // (op 21210) 
s_mov_b32 s72, scc // (op 21210) 

s_cmp_eq_i32 s72, 0 // (op 21210) 
s_cselect_b32 s55, s54, s53 // (op 21210) 


v_lshl_add_u32 v27, s55, 7, v24 // (op 21210) 


s_ashr_i64 s[72:73], s[2:3], 7 // (op 21210) 
s_mul_i32 s53, s54, s72 // (op 21210) 


s_sub_u32 s54, s52, s53 // (op 21210) 


v_lshl_add_u32 v24, s54, 7, v26 // (op 21210) 


v_mov_b32 v31, 0 // (op 21210) convert
v_mov_b32 v30, v24 // (op 21210) convert

v_mul_lo_u32 v33, v27, s31 // (op 21210) most significant: low of low * high
v_mul_hi_u32 v24, v27, s30 // (op 21210) most significant: high of low * low
v_mul_lo_u32 v32, v27, s30 // (op 21210) least significant: low of low * low
v_add_u32 v33, v33, v24 // (op 21210) most significant: sum

v_add_co_u32 v32, s[52:53], v32, v30 // (op 21210) least significant half
v_addc_co_u32 v33, s[52:53], v33, v31, s[52:53] // (op 21210) most significant half



v_lshlrev_b64 v[26:27], 2, v[32:33] // (op 21210) 

v_lshrrev_b64 v[30:31], 3, v[26:27] // (op 21210) 

s_mul_i32 s53, s30, s35 // (op 21212) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 21212) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 21212) least significant
s_mul_hi_u32 s55, s30, s34 // (op 21212) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21212) most significant: sum
s_add_u32 s53, s53, s55 // (op 21212) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21212) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21212) 

s_mov_b32 s68, s32 // (op 21212) reg expression
s_mov_b32 s69, s33 // (op 21212) reg expression
s_mov_b32 s70, s52 // (op 21212) reg expression
s_mov_b32 s71, 131072 // (op 21212) call()

v_lshrrev_b32 v24, 2, v1 // (op 21319) 
s_mul_i32 s52, s16, s8 // (op 21319) 
s_add_u32 s52, s52, s0 // (op 21319) 
v_and_b32 v26, -4, v1 // (op 21319) 
s_mul_hi_u32 s53, s52, s13 // (op 21319) 
v_sub_u32 v27, v1, v26 // (op 21319) 

s_sub_u32 s54, s52, s53 // (op 21319) 
v_lshlrev_b32 v26, 5, v27 // (op 21319) 

s_lshr_b32 s55, s54, 1 // (op 21319) 

s_add_u32 s54, s55, s53 // (op 21319) 


s_lshr_b32 s53, s54, s11 // (op 21319) 

s_and_b32 s54, s11, 2147483648 // (op 21319) 
s_cmp_eq_i32 s54, 0 // (op 21319) 
s_mov_b32 s55, scc // (op 21319) 

s_cmp_eq_i32 s55, 0 // (op 21319) 
s_cselect_b32 s54, s52, s53 // (op 21319) 


s_mul_hi_u32 s53, s54, s12 // (op 21319) 
s_sub_u32 s55, s54, s53 // (op 21319) 
s_lshr_b32 s72, s55, 1 // (op 21319) 

s_add_u32 s55, s72, s53 // (op 21319) 


s_lshr_b32 s53, s55, s10 // (op 21319) 

s_and_b32 s55, s10, 2147483648 // (op 21319) 
s_cmp_eq_i32 s55, 0 // (op 21319) 
s_mov_b32 s72, scc // (op 21319) 

s_cmp_eq_i32 s72, 0 // (op 21319) 
s_cselect_b32 s55, s54, s53 // (op 21319) 


s_mul_i32 s53, s55, s14 // (op 21319) 

s_sub_u32 s55, s54, s53 // (op 21319) 

v_lshl_add_u32 v27, s55, 7, v24 // (op 21319) 


v_mul_lo_u32 v33, v27, s37 // (op 21319) most significant: low of low * high
v_mul_hi_u32 v24, v27, s36 // (op 21319) most significant: high of low * low
v_mul_lo_u32 v32, v27, s36 // (op 21319) least significant: low of low * low
v_add_u32 v33, v33, v24 // (op 21319) most significant: sum


s_ashr_i64 s[72:73], s[2:3], 7 // (op 21319) 
s_mul_i32 s53, s54, s72 // (op 21319) 


s_sub_u32 s54, s52, s53 // (op 21319) 


v_lshl_add_u32 v24, s54, 7, v26 // (op 21319) 


v_mov_b32 v27, 0 // (op 21319) convert
v_mov_b32 v26, v24 // (op 21319) convert

v_add_co_u32 v34, s[52:53], v26, v32 // (op 21319) least significant half
v_addc_co_u32 v35, s[52:53], v27, v33, s[52:53] // (op 21319) most significant half

v_lshlrev_b64 v[34:35], 2, v[34:35] // (op 21319) 


v_lshrrev_b64 v[26:27], 3, v[34:35] // (op 21319) 

s_mul_i32 s53, s36, s23 // (op 21321) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 21321) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 21321) least significant
s_mul_hi_u32 s55, s36, s22 // (op 21321) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21321) most significant: sum
s_add_u32 s53, s53, s55 // (op 21321) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21321) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21321) 

s_mov_b32 s72, s38 // (op 21321) reg expression
s_mov_b32 s73, s39 // (op 21321) reg expression
s_mov_b32 s74, s52 // (op 21321) reg expression
s_mov_b32 s75, 131072 // (op 21321) call()

v_lshrrev_b32 v24, 2, v1 // (op 21338) 
s_mul_i32 s52, s16, s8 // (op 21338) 
s_add_u32 s52, s52, s0 // (op 21338) 
v_and_b32 v32, -4, v1 // (op 21338) 
s_mul_hi_u32 s53, s52, s13 // (op 21338) 
v_sub_u32 v33, v1, v32 // (op 21338) 

s_sub_u32 s54, s52, s53 // (op 21338) 
v_lshlrev_b32 v32, 5, v33 // (op 21338) 

s_lshr_b32 s55, s54, 1 // (op 21338) 

s_add_u32 s54, s55, s53 // (op 21338) 


s_lshr_b32 s53, s54, s11 // (op 21338) 

s_and_b32 s54, s11, 2147483648 // (op 21338) 
s_cmp_eq_i32 s54, 0 // (op 21338) 
s_mov_b32 s55, scc // (op 21338) 

s_cmp_eq_i32 s55, 0 // (op 21338) 
s_cselect_b32 s54, s52, s53 // (op 21338) 


s_mul_hi_u32 s53, s54, s12 // (op 21338) 
s_sub_u32 s55, s54, s53 // (op 21338) 
s_lshr_b32 s76, s55, 1 // (op 21338) 

s_add_u32 s55, s76, s53 // (op 21338) 


s_lshr_b32 s53, s55, s10 // (op 21338) 

s_and_b32 s55, s10, 2147483648 // (op 21338) 
s_cmp_eq_i32 s55, 0 // (op 21338) 
s_mov_b32 s76, scc // (op 21338) 

s_cmp_eq_i32 s76, 0 // (op 21338) 
s_cselect_b32 s55, s54, s53 // (op 21338) 


s_mul_i32 s53, s55, s14 // (op 21338) 

s_sub_u32 s55, s54, s53 // (op 21338) 

v_lshl_add_u32 v33, s55, 7, v24 // (op 21338) 


v_mul_lo_u32 v35, v33, s37 // (op 21338) most significant: low of low * high
v_mul_hi_u32 v24, v33, s36 // (op 21338) most significant: high of low * low
v_mul_lo_u32 v34, v33, s36 // (op 21338) least significant: low of low * low
v_add_u32 v35, v35, v24 // (op 21338) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 21338) 
s_mul_i32 s53, s54, s76 // (op 21338) 


s_sub_u32 s54, s52, s53 // (op 21338) 


v_lshl_add_u32 v24, s54, 7, v32 // (op 21338) 


v_mov_b32 v33, 0 // (op 21338) convert
v_mov_b32 v32, v24 // (op 21338) convert

v_add_co_u32 v36, s[52:53], v32, v34 // (op 21338) least significant half
v_addc_co_u32 v37, s[52:53], v33, v35, s[52:53] // (op 21338) most significant half

v_lshlrev_b64 v[36:37], 2, v[36:37] // (op 21338) 


v_lshrrev_b64 v[32:33], 3, v[36:37] // (op 21338) 

s_mul_i32 s53, s36, s23 // (op 21340) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 21340) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 21340) least significant
s_mul_hi_u32 s55, s36, s22 // (op 21340) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21340) most significant: sum
s_add_u32 s53, s53, s55 // (op 21340) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21340) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21340) 

s_mov_b32 s72, s38 // (op 21340) reg expression
s_mov_b32 s73, s39 // (op 21340) reg expression
s_mov_b32 s74, s52 // (op 21340) reg expression
s_mov_b32 s75, 131072 // (op 21340) call()

v_and_b32 v24, 63, v1 // (op 22508) 
s_mul_i32 s52, s16, s8 // (op 22508) 
s_add_u32 s52, s52, s0 // (op 22508) 
v_lshrrev_b32 v34, 4, v24 // (op 22508) 
s_mul_hi_u32 s53, s52, s13 // (op 22508) 
v_and_b32 v35, 3, v34 // (op 22508) 

s_sub_u32 s54, s52, s53 // (op 22508) 
v_and_b32 v34, 15, v1 // (op 22508) 
s_lshr_b32 s55, s54, 1 // (op 22508) 

v_lshl_add_u32 v36, v35, 4, v34 // (op 22508) 


s_add_u32 s54, s55, s53 // (op 22508) 


v_lshrrev_b32 v34, 7, v1 // (op 22508) 
s_lshr_b32 s53, s54, s11 // (op 22508) 

v_lshl_add_u32 v35, v34, 6, v36 // (op 22508) 


s_and_b32 s54, s11, 2147483648 // (op 22508) 
v_lshrrev_b32 v34, 6, v24 // (op 22508) 

s_cmp_eq_i32 s54, 0 // (op 22508) 
s_mov_b32 s55, scc // (op 22508) 

v_lshlrev_b32 v24, 2, v34 // (op 22508) 

s_cmp_eq_i32 s55, 0 // (op 22508) 
s_cselect_b32 s54, s52, s53 // (op 22508) 


s_mul_hi_u32 s53, s54, s12 // (op 22508) 
s_sub_u32 s55, s54, s53 // (op 22508) 
s_lshr_b32 s76, s55, 1 // (op 22508) 

s_add_u32 s55, s76, s53 // (op 22508) 


s_lshr_b32 s53, s55, s10 // (op 22508) 

s_and_b32 s55, s10, 2147483648 // (op 22508) 
s_cmp_eq_i32 s55, 0 // (op 22508) 
s_mov_b32 s76, scc // (op 22508) 

s_cmp_eq_i32 s76, 0 // (op 22508) 
s_cselect_b32 s55, s54, s53 // (op 22508) 


v_lshl_add_u32 v34, s55, 7, v35 // (op 22508) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 22508) 
s_mul_i32 s53, s54, s76 // (op 22508) 


s_sub_u32 s54, s52, s53 // (op 22508) 


v_lshl_add_u32 v35, s54, 2, v24 // (op 22508) 


v_mul_lo_u32 v24, v34, s40 // (op 22508) 
v_add_u32 v24, v24, v35 // (op 22508) 


s_mul_i32 s53, s40, s35 // (op 22510) most significant: low of low * high
s_mul_i32 s54, s41, s34 // (op 22510) most significant: low of high * low
s_mul_i32 s52, s40, s34 // (op 22510) least significant
s_mul_hi_u32 s55, s40, s34 // (op 22510) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 22510) most significant: sum
s_add_u32 s53, s53, s55 // (op 22510) most significant: sum


s_mov_b32 s76, s42 // (op 22510) reg expression
s_mov_b32 s77, s43 // (op 22510) reg expression
s_mov_b32 s78, s52 // (op 22510) reg expression
s_mov_b32 s79, 131072 // (op 22510) call()

v_and_b32 v34, 63, v1 // (op 22535) 
s_mul_i32 s52, s16, s8 // (op 22535) 
s_add_u32 s52, s52, s0 // (op 22535) 
v_lshrrev_b32 v35, 4, v34 // (op 22535) 
s_mul_hi_u32 s53, s52, s13 // (op 22535) 
v_and_b32 v36, 3, v35 // (op 22535) 

s_sub_u32 s54, s52, s53 // (op 22535) 
v_and_b32 v35, 15, v1 // (op 22535) 
s_lshr_b32 s55, s54, 1 // (op 22535) 

v_lshl_add_u32 v37, v36, 4, v35 // (op 22535) 


s_add_u32 s54, s55, s53 // (op 22535) 


v_lshrrev_b32 v35, 6, v1 // (op 22535) 
s_lshr_b32 s53, s54, s11 // (op 22535) 

v_and_b32 v36, 1, v35 // (op 22535) 

s_and_b32 s54, s11, 2147483648 // (op 22535) 
v_lshl_add_u32 v35, v36, 6, v37 // (op 22535) 


s_cmp_eq_i32 s54, 0 // (op 22535) 
s_mov_b32 s55, scc // (op 22535) 

v_lshrrev_b32 v36, 6, v34 // (op 22535) 

s_cmp_eq_i32 s55, 0 // (op 22535) 
s_cselect_b32 s54, s52, s53 // (op 22535) 


v_lshlrev_b32 v34, 2, v36 // (op 22535) 

s_mul_hi_u32 s53, s54, s12 // (op 22535) 
s_sub_u32 s55, s54, s53 // (op 22535) 
s_lshr_b32 s80, s55, 1 // (op 22535) 

s_add_u32 s55, s80, s53 // (op 22535) 


s_lshr_b32 s53, s55, s10 // (op 22535) 

s_and_b32 s55, s10, 2147483648 // (op 22535) 
s_cmp_eq_i32 s55, 0 // (op 22535) 
s_mov_b32 s80, scc // (op 22535) 

s_cmp_eq_i32 s80, 0 // (op 22535) 
s_cselect_b32 s55, s54, s53 // (op 22535) 


s_mul_i32 s53, s55, s14 // (op 22535) 

s_sub_u32 s55, s54, s53 // (op 22535) 

v_lshl_add_u32 v36, s55, 7, v35 // (op 22535) 


v_mul_lo_u32 v35, v36, s44 // (op 22535) 

s_ashr_i64 s[80:81], s[2:3], 7 // (op 22535) 
s_mul_i32 s53, s54, s80 // (op 22535) 


s_sub_u32 s54, s52, s53 // (op 22535) 


v_lshl_add_u32 v36, s54, 2, v34 // (op 22535) 


v_add_u32 v34, v36, v35 // (op 22535) 


s_mul_i32 s53, s44, s23 // (op 22537) most significant: low of low * high
s_mul_i32 s54, s45, s22 // (op 22537) most significant: low of high * low
s_mul_i32 s52, s44, s22 // (op 22537) least significant
s_mul_hi_u32 s55, s44, s22 // (op 22537) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 22537) most significant: sum
s_add_u32 s53, s53, s55 // (op 22537) most significant: sum


s_mov_b32 s80, s46 // (op 22537) reg expression
s_mov_b32 s81, s47 // (op 22537) reg expression
s_mov_b32 s82, s52 // (op 22537) reg expression
s_mov_b32 s83, 131072 // (op 22537) call()

v_and_b32 v35, -4, v1 // (op 21515) 
v_sub_u32 v36, v1, v35 // (op 21515) 

v_lshlrev_b32 v35, 5, v36 // (op 21515) 

v_lshrrev_b32 v36, 2, v1 // (op 21515) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 21515) 


v_lshlrev_b32 v35, 2, v37 // (op 21515) 

v_lshrrev_b32 v36, 3, v35 // (op 21515) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 21515) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23029) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 21033) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21033) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21033) 


v_mov_b64 v[38:39], v[26:27] // (op 21033) 
s_mov_b32 m0, s52 // (op 21033) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 21033) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 21033) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 21033) most significant half

s_add_u32 m0, m0, 4096 // (op 21033) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 21033) Load value direct to lds



v_and_b32 v35, -4, v1 // (op 21440) 
v_sub_u32 v36, v1, v35 // (op 21440) 

v_lshlrev_b32 v35, 5, v36 // (op 21440) 

v_lshrrev_b32 v36, 2, v1 // (op 21440) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 21440) 


v_lshlrev_b32 v35, 2, v37 // (op 21440) 

v_lshrrev_b32 v36, 3, v35 // (op 21440) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 21440) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23009) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 21038) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21038) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21038) 


v_mov_b64 v[38:39], v[28:29] // (op 21038) 
s_add_u32 s53, 8192, s52 // (op 21038) 
s_mov_b32 m0, s53 // (op 21038) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 21038) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 21038) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 21038) most significant half

s_add_u32 m0, m0, 4096 // (op 21038) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 21038) Load value direct to lds




v_and_b32 v35, -4, v1 // (op 22139) 
v_sub_u32 v36, v1, v35 // (op 22139) 

v_lshlrev_b32 v35, 5, v36 // (op 22139) 

v_lshrrev_b32 v36, 2, v1 // (op 22139) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 22139) 


v_lshlrev_b32 v35, 2, v37 // (op 22139) 

v_lshrrev_b32 v36, 3, v35 // (op 22139) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 22139) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23157) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 21043) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21043) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21043) 


v_mov_b64 v[38:39], v[26:27] // (op 21043) 
s_add_u32 s53, 16384, s52 // (op 21043) 
s_mov_b32 m0, s53 // (op 21043) reg expression
buffer_load_dwordx4 v38, s[72:75], 64 offen offset: 0 lds // (op 21043) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 21043) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 21043) most significant half

s_add_u32 m0, m0, 4096 // (op 21043) 
buffer_load_dwordx4 v38, s[72:75], 64 offen offset: 0 lds // (op 21043) Load value direct to lds




v_and_b32 v35, -4, v1 // (op 22095) 
v_sub_u32 v36, v1, v35 // (op 22095) 

v_lshlrev_b32 v35, 5, v36 // (op 22095) 

v_lshrrev_b32 v36, 2, v1 // (op 22095) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 22095) 


v_lshlrev_b32 v35, 2, v37 // (op 22095) 

v_lshrrev_b32 v36, 3, v35 // (op 22095) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 22095) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23147) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 21045) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21045) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21045) 


v_mov_b64 v[38:39], v[28:29] // (op 21045) 
s_add_u32 s53, 24576, s52 // (op 21045) 
s_mov_b32 m0, s53 // (op 21045) reg expression
buffer_load_dwordx4 v38, s[68:71], 64 offen offset: 0 lds // (op 21045) Load value direct to lds
v_add_co_u32 v38, s[54:55], v38, v36 // (op 21045) least significant half
v_addc_co_u32 v39, s[54:55], 0, v39, s[54:55] // (op 21045) most significant half

s_add_u32 m0, m0, 4096 // (op 21045) 
buffer_load_dwordx4 v38, s[68:71], 64 offen offset: 0 lds // (op 21045) Load value direct to lds




v_and_b32 v35, -4, v1 // (op 22227) 
v_sub_u32 v36, v1, v35 // (op 22227) 

v_lshlrev_b32 v35, 5, v36 // (op 22227) 

v_lshrrev_b32 v36, 2, v1 // (op 22227) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 22227) 


v_lshlrev_b32 v35, 2, v37 // (op 22227) 

v_lshrrev_b32 v36, 3, v35 // (op 22227) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 22227) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23176) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 21047) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21047) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21047) 


v_mov_b64 v[38:39], v[26:27] // (op 21047) 
s_add_u32 s53, 32768, s52 // (op 21047) 
s_mov_b32 m0, s53 // (op 21047) reg expression
s_mov_b32 s54, 128 // (op 21047) call()
buffer_load_dwordx4 v38, s[72:75], s54 offen offset: 0 lds // (op 21047) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 21047) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 21047) most significant half

s_add_u32 m0, m0, 4096 // (op 21047) 
buffer_load_dwordx4 v38, s[72:75], s54 offen offset: 0 lds // (op 21047) Load value direct to lds





v_and_b32 v35, -4, v1 // (op 22183) 
v_sub_u32 v36, v1, v35 // (op 22183) 

v_lshlrev_b32 v35, 5, v36 // (op 22183) 

v_lshrrev_b32 v36, 2, v1 // (op 22183) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 22183) 


v_lshlrev_b32 v35, 2, v37 // (op 22183) 

v_lshrrev_b32 v36, 3, v35 // (op 22183) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 22183) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23167) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 21049) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21049) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21049) 


v_mov_b64 v[38:39], v[28:29] // (op 21049) 
s_add_u32 s53, 40960, s52 // (op 21049) 
s_mov_b32 m0, s53 // (op 21049) reg expression
s_mov_b32 s54, 128 // (op 21049) call()
buffer_load_dwordx4 v38, s[68:71], s54 offen offset: 0 lds // (op 21049) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 21049) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 21049) most significant half

s_add_u32 m0, m0, 4096 // (op 21049) 
buffer_load_dwordx4 v38, s[68:71], s54 offen offset: 0 lds // (op 21049) Load value direct to lds





v_and_b32 v35, -4, v1 // (op 22315) 
v_sub_u32 v36, v1, v35 // (op 22315) 

v_lshlrev_b32 v35, 5, v36 // (op 22315) 

v_lshrrev_b32 v36, 2, v1 // (op 22315) 
v_lshl_add_u32 v37, v36, 7, v35 // (op 22315) 


v_lshlrev_b32 v35, 2, v37 // (op 22315) 

v_lshrrev_b32 v36, 3, v35 // (op 22315) 

s_nop 0
v_readfirstlane_b32 s52, v36 // (op 22315) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23195) 

s_lshl_b64 s[54:55], s[36:37], 6 // (op 21051) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21051) 

v_lshrrev_b64 v[36:37], 3, s[84:85] // (op 21051) 


v_mov_b64 v[38:39], v[26:27] // (op 21051) 
s_add_u32 s53, 49152, s52 // (op 21051) 
s_mov_b32 m0, s53 // (op 21051) reg expression
s_mov_b32 s54, 192 // (op 21051) call()
buffer_load_dwordx4 v38, s[72:75], s54 offen offset: 0 lds // (op 21051) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v36 // (op 21051) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 21051) most significant half

s_add_u32 m0, m0, 4096 // (op 21051) 
buffer_load_dwordx4 v38, s[72:75], s54 offen offset: 0 lds // (op 21051) Load value direct to lds






v_and_b32 v26, -4, v1 // (op 22271) 
v_sub_u32 v27, v1, v26 // (op 22271) 

v_lshlrev_b32 v26, 5, v27 // (op 22271) 

v_lshrrev_b32 v27, 2, v1 // (op 22271) 
v_lshl_add_u32 v35, v27, 7, v26 // (op 22271) 


v_lshlrev_b32 v26, 2, v35 // (op 22271) 

v_lshrrev_b32 v27, 3, v26 // (op 22271) 

s_nop 0
v_readfirstlane_b32 s52, v27 // (op 22271) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23186) 

s_lshl_b64 s[54:55], s[30:31], 6 // (op 21053) 
s_lshl_b64 s[84:85], s[54:55], 2 // (op 21053) 

v_lshrrev_b64 v[26:27], 3, s[84:85] // (op 21053) 


v_mov_b64 v[36:37], v[28:29] // (op 21053) 
s_add_u32 s53, 57344, s52 // (op 21053) 
s_mov_b32 m0, s53 // (op 21053) reg expression
s_mov_b32 s54, 192 // (op 21053) call()
buffer_load_dwordx4 v36, s[68:71], s54 offen offset: 0 lds // (op 21053) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v26 // (op 21053) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 21053) most significant half

s_add_u32 m0, m0, 4096 // (op 21053) 
buffer_load_dwordx4 v36, s[68:71], s54 offen offset: 0 lds // (op 21053) Load value direct to lds






s_waitcnt vmcnt(14)
s_barrier  // (op 23026) 



ds_read_b128 v[26:29], v23 // (op 20535) Load local data 



ds_read_b128 v[36:39], v23 offset:1024 // (op 20540) Load local data 



ds_read_b128 v[40:43], v23 offset:2048 // (op 20545) Load local data 



ds_read_b128 v[44:47], v23 offset:3072 // (op 20550) Load local data 
s_waitcnt vmcnt(12)
s_barrier  // (op 23006) 



ds_read_b128 v[48:51], v21 offset:8192 // (op 20515) Load local data 



ds_read_b128 v[52:55], v21 offset:9216 // (op 20520) Load local data 



ds_read_b128 v[56:59], v21 offset:10240 // (op 20525) Load local data 



ds_read_b128 v[60:63], v21 offset:11264 // (op 20530) Load local data 
v_and_b32 v35, -4, v1 // (op 21412) 
v_sub_u32 v64, v1, v35 // (op 21412) 

v_lshlrev_b32 v35, 5, v64 // (op 21412) 

v_lshrrev_b32 v64, 2, v1 // (op 21412) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 21412) 


v_lshlrev_b32 v35, 2, v65 // (op 21412) 

v_lshrrev_b32 v64, 3, v35 // (op 21412) 

s_nop 0
v_readfirstlane_b32 s52, v64 // (op 21412) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 21487) 
v_sub_u32 v64, v1, v35 // (op 21487) 

v_lshlrev_b32 v35, 5, v64 // (op 21487) 

v_lshrrev_b32 v64, 2, v1 // (op 21487) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 21487) 


v_lshlrev_b32 v35, 2, v65 // (op 21487) 

v_lshrrev_b32 v64, 3, v35 // (op 21487) 

s_nop 0
v_readfirstlane_b32 s53, v64 // (op 21487) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22080) 
v_sub_u32 v64, v1, v35 // (op 22080) 

v_lshlrev_b32 v35, 5, v64 // (op 22080) 

v_lshrrev_b32 v64, 2, v1 // (op 22080) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22080) 


v_lshlrev_b32 v35, 2, v65 // (op 22080) 

v_lshrrev_b32 v64, 3, v35 // (op 22080) 

s_nop 0
v_readfirstlane_b32 s54, v64 // (op 22080) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22124) 
v_sub_u32 v64, v1, v35 // (op 22124) 

v_lshlrev_b32 v35, 5, v64 // (op 22124) 

v_lshrrev_b32 v64, 2, v1 // (op 22124) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22124) 


v_lshlrev_b32 v35, 2, v65 // (op 22124) 

v_lshrrev_b32 v64, 3, v35 // (op 22124) 

s_nop 0
v_readfirstlane_b32 s55, v64 // (op 22124) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22168) 
v_sub_u32 v64, v1, v35 // (op 22168) 

v_lshlrev_b32 v35, 5, v64 // (op 22168) 

v_lshrrev_b32 v64, 2, v1 // (op 22168) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22168) 


v_lshlrev_b32 v35, 2, v65 // (op 22168) 

v_lshrrev_b32 v64, 3, v35 // (op 22168) 

s_nop 0
v_readfirstlane_b32 s84, v64 // (op 22168) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22212) 
v_sub_u32 v64, v1, v35 // (op 22212) 

v_lshlrev_b32 v35, 5, v64 // (op 22212) 

v_lshrrev_b32 v64, 2, v1 // (op 22212) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22212) 


v_lshlrev_b32 v35, 2, v65 // (op 22212) 

v_lshrrev_b32 v64, 3, v35 // (op 22212) 

s_nop 0
v_readfirstlane_b32 s85, v64 // (op 22212) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22256) 
v_sub_u32 v64, v1, v35 // (op 22256) 

v_lshlrev_b32 v35, 5, v64 // (op 22256) 

v_lshrrev_b32 v64, 2, v1 // (op 22256) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22256) 


v_lshlrev_b32 v35, 2, v65 // (op 22256) 

v_lshrrev_b32 v64, 3, v35 // (op 22256) 

s_nop 0
v_readfirstlane_b32 s86, v64 // (op 22256) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v35, -4, v1 // (op 22300) 
v_sub_u32 v64, v1, v35 // (op 22300) 

v_lshlrev_b32 v35, 5, v64 // (op 22300) 

v_lshrrev_b32 v64, 2, v1 // (op 22300) 
v_lshl_add_u32 v65, v64, 7, v35 // (op 22300) 


v_lshlrev_b32 v35, 2, v65 // (op 22300) 

v_lshrrev_b32 v64, 3, v35 // (op 22300) 

s_nop 0
v_readfirstlane_b32 s87, v64 // (op 22300) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 8822) call()
s_and_b32 s88, s7, 4294967292 // (op 8821) 
s_cmp_lt_u32 s1, s88 // (op 8821) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_8821 // (op 8821) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_8821 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_8821:


s_add_u32 s1, s1, 4 // (op 8823) 
v_mov_b32 v35, v24 // (op 70) 
v_mov_b32 v64, v35 // (op 70) reg expression

buffer_load_dwordx4 v[66:69], v64, s[76:79], 0 offen // (op 70) Load value


v_add_u32 v24, 16, v24 // (op 22498) 


v_mov_b32 v35, v34 // (op 74) 
v_mov_b32 v64, v35 // (op 74) reg expression

buffer_load_dwordx4 v[70:73], v64, s[80:83], 0 offen // (op 74) Load value


v_add_u32 v34, 16, v34 // (op 22525) 
s_waitcnt vmcnt(1)
v_bfe_u32 v74, v66, 0, 8 // (op 19268) 
v_bfe_u32 v75, v66, 8, 8 // (op 19268) 
v_bfe_u32 v76, v66, 16, 8 // (op 19268) 
v_bfe_u32 v77, v66, 24, 8 // (op 19268) 
v_permlane16_swap_b32 v74, v75 // (op 19268) 
s_nop 0
v_permlane16_swap_b32 v76, v77 // (op 19268) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v74, v76 // (op 19268) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v75, v77 // (op 19268) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 23032) 

s_lshl_b64 s[88:89], s[36:37], 6 // (op 20912) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 20912) 

v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 20912) 


v_mov_b64 v[78:79], v[32:33] // (op 20912) 
v_mov_b64 v[80:81], v[78:79] // (op 20912) 
s_mov_b32 m0, s53 // (op 20912) reg expression
s_mov_b32 s88, 256 // (op 20912) call()
buffer_load_dwordx4 v80, s[72:75], s88 offen offset: 0 lds // (op 20912) Load value direct to lds
v_add_co_u32 v80, s[90:91], v80, v64 // (op 20912) least significant half
v_addc_co_u32 v81, s[90:91], 0, v81, s[90:91] // (op 20912) most significant half

s_add_u32 m0, m0, 4096 // (op 20912) 
buffer_load_dwordx4 v80, s[72:75], s88 offen offset: 0 lds // (op 20912) Load value direct to lds



s_waitcnt vmcnt(2)
v_bfe_u32 v80, v70, 0, 8 // (op 19397) 
v_bfe_u32 v81, v70, 8, 8 // (op 19397) 
v_bfe_u32 v82, v70, 16, 8 // (op 19397) 
v_bfe_u32 v83, v70, 24, 8 // (op 19397) 
v_permlane16_swap_b32 v80, v81 // (op 19397) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 19397) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 19397) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 19397) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23012) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[26:29], a[0:3], v74, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 76) 
s_lshl_b64 s[88:89], s[30:31], 6 // (op 20907) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:51], v[36:39], a[4:7], v74, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5955) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 20907) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:51], v[40:43], a[8:11], v74, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6181) 
v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 20907) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[44:47], a[12:15], v74, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6407) 
v_mov_b64 v[84:85], v[30:31] // (op 20907) 
v_mov_b64 v[86:87], v[84:85] // (op 20907) 
s_barrier  // (op 23153) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:55], v[26:29], a[16:19], v75, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7079) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:55], v[36:39], a[20:23], v75, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7003) 
s_add_u32 s88, 8192, s52 // (op 20907) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:55], v[40:43], a[24:27], v75, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6927) 
s_mov_b32 m0, s88 // (op 20907) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:55], v[44:47], a[28:31], v75, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6851) 
s_mov_b32 s89, 256 // (op 20907) call()
buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 20907) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:59], v[26:29], a[32:35], v76, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8081) 
v_add_co_u32 v86, s[90:91], v86, v64 // (op 20907) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[56:59], v[36:39], a[36:39], v76, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8005) 
v_addc_co_u32 v87, s[90:91], 0, v87, s[90:91] // (op 20907) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[56:59], v[40:43], a[40:43], v76, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7929) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:59], v[44:47], a[44:47], v76, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7853) 
s_add_u32 m0, m0, 4096 // (op 20907) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:63], v[26:29], a[48:51], v77, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9083) 
ds_read_b128 v[88:91], v14 offset:16384 // (op 5750) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:63], v[36:39], a[52:55], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9007) 
ds_read_b128 v[92:95], v14 offset:17408 // (op 5966) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:63], v[40:43], a[56:59], v77, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8931) 
ds_read_b128 v[96:99], v14 offset:18432 // (op 6192) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[60:63], v[44:47], a[60:63], v77, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8855) 
ds_read_b128 v[100:103], v14 offset:19456 // (op 6418) Load local data 


buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 20907) Load value direct to lds




s_barrier  // (op 23143) 



ds_read_b128 v[48:51], v13 offset:24576 // (op 5748) Load local data 



ds_read_b128 v[52:55], v13 offset:25600 // (op 6861) Load local data 



ds_read_b128 v[56:59], v13 offset:26624 // (op 7863) Load local data 



ds_read_b128 v[74:77], v13 offset:27648 // (op 8865) Load local data 
v_bfe_u32 v80, v67, 0, 8 // (op 19283) 
v_bfe_u32 v81, v67, 8, 8 // (op 19283) 
v_bfe_u32 v82, v67, 16, 8 // (op 19283) 
v_bfe_u32 v83, v67, 24, 8 // (op 19283) 
v_permlane16_swap_b32 v80, v81 // (op 19283) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 19283) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 19283) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 19283) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 23160) 

s_lshl_b64 s[88:89], s[36:37], 6 // (op 20986) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 20986) 

v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 20986) 


v_mov_b64 v[86:87], v[78:79] // (op 20986) 
s_add_u32 s88, 16384, s55 // (op 20986) 
s_mov_b32 m0, s88 // (op 20986) reg expression
s_mov_b32 s89, 320 // (op 20986) call()
buffer_load_dwordx4 v86, s[72:75], s89 offen offset: 0 lds // (op 20986) Load value direct to lds
v_add_co_u32 v86, s[90:91], v86, v64 // (op 20986) least significant half
v_addc_co_u32 v87, s[90:91], 0, v87, s[90:91] // (op 20986) most significant half

s_add_u32 m0, m0, 4096 // (op 20986) 
buffer_load_dwordx4 v86, s[72:75], s89 offen offset: 0 lds // (op 20986) Load value direct to lds




v_bfe_u32 v104, v71, 0, 8 // (op 19412) 
v_bfe_u32 v105, v71, 8, 8 // (op 19412) 
v_bfe_u32 v106, v71, 16, 8 // (op 19412) 
v_bfe_u32 v107, v71, 24, 8 // (op 19412) 
v_permlane16_swap_b32 v104, v105 // (op 19412) 
s_nop 0
v_permlane16_swap_b32 v106, v107 // (op 19412) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v104, v106 // (op 19412) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v105, v107 // (op 19412) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23150) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[88:91], a[0:3], v80, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5755) 
s_lshl_b64 s[88:89], s[30:31], 6 // (op 20981) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:51], v[92:95], a[4:7], v80, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5956) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 20981) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:51], v[96:99], a[8:11], v80, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6182) 
v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 20981) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[100:103], a[12:15], v80, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6408) 
v_mov_b64 v[86:87], v[84:85] // (op 20981) 
s_add_u32 s88, 24576, s54 // (op 20981) 
s_barrier  // (op 23173) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:55], v[88:91], a[16:19], v81, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7080) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:55], v[92:95], a[20:23], v81, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7004) 
s_mov_b32 m0, s88 // (op 20981) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:55], v[96:99], a[24:27], v81, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6928) 
ds_read_b128 v[26:29], v16 offset:32768 // (op 5778) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:55], v[100:103], a[28:31], v81, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6852) 
ds_read_b128 v[36:39], v16 offset:33792 // (op 5976) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:59], v[88:91], a[32:35], v82, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8082) 
ds_read_b128 v[40:43], v16 offset:34816 // (op 6202) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[56:59], v[92:95], a[36:39], v82, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8006) 
s_mov_b32 s89, 320 // (op 20981) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[56:59], v[96:99], a[40:43], v82, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7930) 
ds_read_b128 v[44:47], v16 offset:35840 // (op 6428) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[56:59], v[100:103], a[44:47], v82, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7854) 
s_barrier  // (op 23163) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:77], v[88:91], a[48:51], v83, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9084) 
ds_read_b128 v[48:51], v15 offset:40960 // (op 5776) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:77], v[92:95], a[52:55], v83, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9008) 
ds_read_b128 v[52:55], v15 offset:41984 // (op 6871) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:77], v[96:99], a[56:59], v83, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8932) 
ds_read_b128 v[88:91], v15 offset:43008 // (op 7873) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[100:103], a[60:63], v83, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8856) 
ds_read_b128 v[92:95], v15 offset:44032 // (op 8875) Load local data 




buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 20981) Load value direct to lds
v_add_co_u32 v86, s[90:91], v86, v64 // (op 20981) least significant half
v_addc_co_u32 v87, s[90:91], 0, v87, s[90:91] // (op 20981) most significant half

s_add_u32 m0, m0, 4096 // (op 20981) 
buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 20981) Load value direct to lds




v_bfe_u32 v74, v68, 0, 8 // (op 19300) 
v_bfe_u32 v75, v68, 8, 8 // (op 19300) 
v_bfe_u32 v76, v68, 16, 8 // (op 19300) 
v_bfe_u32 v77, v68, 24, 8 // (op 19300) 
v_permlane16_swap_b32 v74, v75 // (op 19300) 
s_nop 0
v_permlane16_swap_b32 v76, v77 // (op 19300) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v74, v76 // (op 19300) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v75, v77 // (op 19300) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 23179) 

s_lshl_b64 s[88:89], s[36:37], 6 // (op 20996) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 20996) 

v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 20996) 


v_mov_b64 v[86:87], v[78:79] // (op 20996) 
s_add_u32 s88, 32768, s85 // (op 20996) 
s_mov_b32 m0, s88 // (op 20996) reg expression
s_mov_b32 s89, 384 // (op 20996) call()
buffer_load_dwordx4 v86, s[72:75], s89 offen offset: 0 lds // (op 20996) Load value direct to lds
v_add_co_u32 v86, s[90:91], v86, v64 // (op 20996) least significant half
v_addc_co_u32 v87, s[90:91], 0, v87, s[90:91] // (op 20996) most significant half

s_add_u32 m0, m0, 4096 // (op 20996) 
v_bfe_u32 v80, v72, 0, 8 // (op 19430) 
v_bfe_u32 v81, v72, 8, 8 // (op 19430) 
v_bfe_u32 v82, v72, 16, 8 // (op 19430) 
v_bfe_u32 v83, v72, 24, 8 // (op 19430) 
v_permlane16_swap_b32 v80, v81 // (op 19430) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 19430) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 19430) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 19430) 
buffer_load_dwordx4 v86, s[72:75], s89 offen offset: 0 lds // (op 20996) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[26:29], a[0:3], v74, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5783) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23170) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:51], v[36:39], a[4:7], v74, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5957) 
s_lshl_b64 s[88:89], s[30:31], 6 // (op 20991) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:51], v[40:43], a[8:11], v74, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6183) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 20991) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[44:47], a[12:15], v74, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6409) 
v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 20991) 


v_mov_b64 v[86:87], v[84:85] // (op 20991) 
s_barrier  // (op 23192) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:55], v[26:29], a[16:19], v75, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7081) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:55], v[36:39], a[20:23], v75, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7005) 
ds_read_b128 v[56:59], v18 offset:49152 // (op 5806) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:55], v[40:43], a[24:27], v75, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6929) 
ds_read_b128 v[60:63], v18 offset:50176 // (op 5986) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:55], v[44:47], a[28:31], v75, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6853) 
s_add_u32 s88, 40960, s84 // (op 20991) 
s_mov_b32 m0, s88 // (op 20991) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[26:29], a[32:35], v76, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8083) 
s_mov_b32 s89, 384 // (op 20991) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[36:39], a[36:39], v76, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8007) 
ds_read_b128 v[96:99], v18 offset:51200 // (op 6212) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[40:43], a[40:43], v76, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7931) 
buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 20991) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[44:47], a[44:47], v76, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7855) 
v_add_co_u32 v86, s[90:91], v86, v64 // (op 20991) least significant half

v_addc_co_u32 v87, s[90:91], 0, v87, s[90:91] // (op 20991) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[26:29], a[48:51], v77, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9085) 
s_add_u32 m0, m0, 4096 // (op 20991) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[36:39], a[52:55], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9009) 
ds_read_b128 v[88:91], v18 offset:52224 // (op 6438) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[40:43], a[56:59], v77, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8933) 
s_barrier  // (op 23182) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[44:47], a[60:63], v77, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8857) 
ds_read_b128 v[48:51], v17 offset:57344 // (op 5804) Load local data 






buffer_load_dwordx4 v86, s[68:71], s89 offen offset: 0 lds // (op 20991) Load value direct to lds




ds_read_b128 v[52:55], v17 offset:58368 // (op 6881) Load local data 



ds_read_b128 v[74:77], v17 offset:59392 // (op 7883) Load local data 



ds_read_b128 v[80:83], v17 offset:60416 // (op 8885) Load local data 
v_bfe_u32 v92, v69, 0, 8 // (op 19317) 
v_bfe_u32 v93, v69, 8, 8 // (op 19317) 
v_bfe_u32 v94, v69, 16, 8 // (op 19317) 
v_bfe_u32 v95, v69, 24, 8 // (op 19317) 
v_permlane16_swap_b32 v92, v93 // (op 19317) 
s_nop 0
v_permlane16_swap_b32 v94, v95 // (op 19317) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v92, v94 // (op 19317) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v93, v95 // (op 19317) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 23198) 


s_lshl_b64 s[88:89], s[36:37], 6 // (op 21006) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 21006) 

v_lshrrev_b64 v[86:87], 3, s[90:91] // (op 21006) 


v_mov_b64 v[64:65], v[78:79] // (op 21006) 
s_add_u32 s88, 49152, s87 // (op 21006) 
s_mov_b32 m0, s88 // (op 21006) reg expression
s_mov_b32 s89, 448 // (op 21006) call()
v_bfe_u32 v66, v73, 0, 8 // (op 19448) 
v_bfe_u32 v67, v73, 8, 8 // (op 19448) 
v_bfe_u32 v68, v73, 16, 8 // (op 19448) 
v_bfe_u32 v69, v73, 24, 8 // (op 19448) 
v_permlane16_swap_b32 v66, v67 // (op 19448) 
s_nop 0
v_permlane16_swap_b32 v68, v69 // (op 19448) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v66, v68 // (op 19448) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 19448) 
buffer_load_dwordx4 v64, s[72:75], s89 offen offset: 0 lds // (op 21006) Load value direct to lds

v_add_co_u32 v64, s[90:91], v64, v86 // (op 21006) least significant half
v_addc_co_u32 v65, s[90:91], 0, v65, s[90:91] // (op 21006) most significant half

s_add_u32 m0, m0, 4096 // (op 21006) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[56:59], a[0:3], v92, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5811) 
buffer_load_dwordx4 v64, s[72:75], s89 offen offset: 0 lds // (op 21006) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:51], v[60:63], a[4:7], v92, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5958) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:51], v[96:99], a[8:11], v92, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6184) 
v_add_co_u32 v32, vcc, 256, v32 // (op 21330) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[88:91], a[12:15], v92, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6410) 
v_addc_co_u32 v33, vcc, 0, v33, vcc // (op 21330) most significant half
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23189) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:55], v[56:59], a[16:19], v93, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7082) 
s_lshl_b64 s[88:89], s[30:31], 6 // (op 21001) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:55], v[60:63], a[20:23], v93, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7006) 
s_lshl_b64 s[90:91], s[88:89], 2 // (op 21001) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[52:55], v[96:99], a[24:27], v93, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6930) 
v_lshrrev_b64 v[64:65], 3, s[90:91] // (op 21001) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[52:55], v[88:91], a[28:31], v93, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6854) 
v_mov_b64 v[78:79], v[84:85] // (op 21001) 
s_add_u32 s88, 57344, s86 // (op 21001) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[56:59], a[32:35], v94, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8084) 
s_mov_b32 m0, s88 // (op 21001) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[60:63], a[36:39], v94, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8008) 
s_mov_b32 s89, 448 // (op 21001) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:77], v[96:99], a[40:43], v94, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7932) 
buffer_load_dwordx4 v78, s[68:71], s89 offen offset: 0 lds // (op 21001) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:77], v[88:91], a[44:47], v94, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7856) 
v_add_co_u32 v78, s[90:91], v78, v64 // (op 21001) least significant half

v_addc_co_u32 v79, s[90:91], 0, v79, s[90:91] // (op 21001) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[56:59], a[48:51], v95, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9086) 
s_add_u32 m0, m0, 4096 // (op 21001) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[60:63], a[52:55], v95, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9010) 
s_waitcnt vmcnt(13)
s_barrier  // (op 23038) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[96:99], a[56:59], v95, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8934) 
ds_read_b128 v[26:29], v4 // (op 65) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[88:91], a[60:63], v95, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8858) 
ds_read_b128 v[36:39], v4 offset:1024 // (op 5952) Load local data 







ds_read_b128 v[40:43], v4 offset:2048 // (op 6178) Load local data 
buffer_load_dwordx4 v78, s[68:71], s89 offen offset: 0 lds // (op 21001) Load value direct to lds








ds_read_b128 v[44:47], v4 offset:3072 // (op 6404) Load local data 
v_add_co_u32 v30, vcc, 256, v30 // (op 21202) least significant half
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 21202) most significant half
s_waitcnt vmcnt(12)
s_barrier  // (op 23018) 



ds_read_b128 v[48:51], v3 offset:8192 // (op 60) Load local data 



ds_read_b128 v[52:55], v3 offset:9216 // (op 6847) Load local data 



ds_read_b128 v[56:59], v3 offset:10240 // (op 7849) Load local data 



ds_read_b128 v[60:63], v3 offset:11264 // (op 8851) Load local data 
s_and_b32 s88, s7, 4294967292 // (op 8821) 
s_cmp_lt_u32 s1, s88 // (op 8821) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_8821 // (op 8821) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_8821 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_8821:














s_barrier  // (op 23531) 
s_and_b32 s52, s7, 4294967292 // (op 21090) 
v_lshrrev_b32 v24, 2, v1 // (op 21090) 
s_add_u32 s53, s0, s52 // (op 21090) 

v_and_b32 v30, -4, v1 // (op 21090) 
v_sub_u32 v31, v1, v30 // (op 21090) 

s_mul_i32 s52, s16, s8 // (op 21090) 
s_add_u32 s52, s52, s53 // (op 21090) 

v_lshlrev_b32 v30, 5, v31 // (op 21090) 

s_mul_hi_u32 s53, s52, s13 // (op 21090) 
s_sub_u32 s54, s52, s53 // (op 21090) 
s_lshr_b32 s55, s54, 1 // (op 21090) 

s_add_u32 s54, s55, s53 // (op 21090) 


s_lshr_b32 s53, s54, s11 // (op 21090) 

s_and_b32 s54, s11, 2147483648 // (op 21090) 
s_cmp_eq_i32 s54, 0 // (op 21090) 
s_mov_b32 s55, scc // (op 21090) 

s_cmp_eq_i32 s55, 0 // (op 21090) 
s_cselect_b32 s54, s52, s53 // (op 21090) 


s_mul_hi_u32 s53, s54, s12 // (op 21090) 
s_sub_u32 s55, s54, s53 // (op 21090) 
s_lshr_b32 s76, s55, 1 // (op 21090) 

s_add_u32 s55, s76, s53 // (op 21090) 


s_lshr_b32 s53, s55, s10 // (op 21090) 

s_and_b32 s55, s10, 2147483648 // (op 21090) 
s_cmp_eq_i32 s55, 0 // (op 21090) 
s_mov_b32 s76, scc // (op 21090) 

s_cmp_eq_i32 s76, 0 // (op 21090) 
s_cselect_b32 s55, s54, s53 // (op 21090) 


v_lshl_add_u32 v31, s55, 7, v24 // (op 21090) 


s_ashr_i64 s[76:77], s[2:3], 7 // (op 21090) 
s_mul_i32 s53, s54, s76 // (op 21090) 


s_sub_u32 s54, s52, s53 // (op 21090) 


v_lshl_add_u32 v24, s54, 7, v30 // (op 21090) 


v_mov_b32 v33, 0 // (op 21090) convert
v_mov_b32 v32, v24 // (op 21090) convert

v_mul_lo_u32 v35, v31, s31 // (op 21090) most significant: low of low * high
v_mul_hi_u32 v24, v31, s30 // (op 21090) most significant: high of low * low
v_mul_lo_u32 v34, v31, s30 // (op 21090) least significant: low of low * low
v_add_u32 v35, v35, v24 // (op 21090) most significant: sum

v_add_co_u32 v34, s[52:53], v34, v32 // (op 21090) least significant half
v_addc_co_u32 v35, s[52:53], v35, v33, s[52:53] // (op 21090) most significant half



v_lshlrev_b64 v[30:31], 2, v[34:35] // (op 21090) 

v_lshrrev_b64 v[32:33], 3, v[30:31] // (op 21090) 

s_mul_i32 s53, s30, s35 // (op 21092) most significant: low of low * high
s_mul_i32 s54, s31, s34 // (op 21092) most significant: low of high * low
s_mul_i32 s52, s30, s34 // (op 21092) least significant
s_mul_hi_u32 s55, s30, s34 // (op 21092) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21092) most significant: sum
s_add_u32 s53, s53, s55 // (op 21092) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21092) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21092) 

s_mov_b32 s68, s32 // (op 21092) reg expression
s_mov_b32 s69, s33 // (op 21092) reg expression
s_mov_b32 s70, s52 // (op 21092) reg expression
s_mov_b32 s71, 131072 // (op 21092) call()

s_and_b32 s52, s7, 4294967292 // (op 21254) 
v_lshrrev_b32 v24, 2, v1 // (op 21254) 
s_add_u32 s53, s0, s52 // (op 21254) 

v_and_b32 v30, -4, v1 // (op 21254) 
v_sub_u32 v31, v1, v30 // (op 21254) 

s_mul_i32 s52, s16, s8 // (op 21254) 
s_add_u32 s52, s52, s53 // (op 21254) 

v_lshlrev_b32 v30, 5, v31 // (op 21254) 

s_mul_hi_u32 s53, s52, s13 // (op 21254) 
s_sub_u32 s54, s52, s53 // (op 21254) 
s_lshr_b32 s55, s54, 1 // (op 21254) 

s_add_u32 s54, s55, s53 // (op 21254) 


s_lshr_b32 s53, s54, s11 // (op 21254) 

s_and_b32 s54, s11, 2147483648 // (op 21254) 
s_cmp_eq_i32 s54, 0 // (op 21254) 
s_mov_b32 s55, scc // (op 21254) 

s_cmp_eq_i32 s55, 0 // (op 21254) 
s_cselect_b32 s54, s52, s53 // (op 21254) 


s_mul_hi_u32 s53, s54, s12 // (op 21254) 
s_sub_u32 s55, s54, s53 // (op 21254) 
s_lshr_b32 s76, s55, 1 // (op 21254) 

s_add_u32 s55, s76, s53 // (op 21254) 


s_lshr_b32 s53, s55, s10 // (op 21254) 

s_and_b32 s55, s10, 2147483648 // (op 21254) 
s_cmp_eq_i32 s55, 0 // (op 21254) 
s_mov_b32 s76, scc // (op 21254) 

s_cmp_eq_i32 s76, 0 // (op 21254) 
s_cselect_b32 s55, s54, s53 // (op 21254) 


s_mul_i32 s53, s55, s14 // (op 21254) 

s_sub_u32 s55, s54, s53 // (op 21254) 

v_lshl_add_u32 v31, s55, 7, v24 // (op 21254) 


v_mul_lo_u32 v35, v31, s37 // (op 21254) most significant: low of low * high
v_mul_hi_u32 v24, v31, s36 // (op 21254) most significant: high of low * low
v_mul_lo_u32 v34, v31, s36 // (op 21254) least significant: low of low * low
v_add_u32 v35, v35, v24 // (op 21254) most significant: sum


s_ashr_i64 s[76:77], s[2:3], 7 // (op 21254) 
s_mul_i32 s53, s54, s76 // (op 21254) 


s_sub_u32 s54, s52, s53 // (op 21254) 


v_lshl_add_u32 v24, s54, 7, v30 // (op 21254) 


v_mov_b32 v31, 0 // (op 21254) convert
v_mov_b32 v30, v24 // (op 21254) convert

v_add_co_u32 v64, s[52:53], v30, v34 // (op 21254) least significant half
v_addc_co_u32 v65, s[52:53], v31, v35, s[52:53] // (op 21254) most significant half

v_lshlrev_b64 v[64:65], 2, v[64:65] // (op 21254) 


v_lshrrev_b64 v[30:31], 3, v[64:65] // (op 21254) 

s_mul_i32 s53, s36, s23 // (op 21256) most significant: low of low * high
s_mul_i32 s54, s37, s22 // (op 21256) most significant: low of high * low
s_mul_i32 s52, s36, s22 // (op 21256) least significant
s_mul_hi_u32 s55, s36, s22 // (op 21256) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 21256) most significant: sum
s_add_u32 s53, s53, s55 // (op 21256) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 2 // (op 21256) 

s_ashr_i64 s[52:53], s[54:55], 3 // (op 21256) 

s_mov_b32 s72, s38 // (op 21256) reg expression
s_mov_b32 s73, s39 // (op 21256) reg expression
s_mov_b32 s74, s52 // (op 21256) reg expression
s_mov_b32 s75, 131072 // (op 21256) call()

v_and_b32 v24, -4, v1 // (op 21425) 
v_sub_u32 v34, v1, v24 // (op 21425) 

v_lshlrev_b32 v24, 5, v34 // (op 21425) 

v_lshrrev_b32 v34, 2, v1 // (op 21425) 
v_lshl_add_u32 v35, v34, 7, v24 // (op 21425) 


v_lshlrev_b32 v24, 2, v35 // (op 21425) 

v_lshrrev_b32 v34, 3, v24 // (op 21425) 

s_nop 0
v_readfirstlane_b32 s52, v34 // (op 21425) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 21500) 
v_sub_u32 v34, v1, v24 // (op 21500) 

v_lshlrev_b32 v24, 5, v34 // (op 21500) 

v_lshrrev_b32 v34, 2, v1 // (op 21500) 
v_lshl_add_u32 v35, v34, 7, v24 // (op 21500) 


v_lshlrev_b32 v24, 2, v35 // (op 21500) 

v_lshrrev_b32 v34, 3, v24 // (op 21500) 

s_nop 0
v_readfirstlane_b32 s53, v34 // (op 21500) Wait state hazard: VALU Write Readlane Hazard

s_and_b32 s54, s7, 4294967292 // (op 22562) 
v_and_b32 v24, 63, v1 // (op 22562) 
s_add_u32 s55, s0, s54 // (op 22562) 

v_lshrrev_b32 v34, 4, v24 // (op 22562) 
v_and_b32 v35, 3, v34 // (op 22562) 

s_mul_i32 s54, s16, s8 // (op 22562) 
s_add_u32 s54, s54, s55 // (op 22562) 

v_and_b32 v34, 15, v1 // (op 22562) 
s_mul_hi_u32 s55, s54, s13 // (op 22562) 
v_lshl_add_u32 v64, v35, 4, v34 // (op 22562) 


s_sub_u32 s76, s54, s55 // (op 22562) 
v_lshrrev_b32 v34, 7, v1 // (op 22562) 
s_lshr_b32 s77, s76, 1 // (op 22562) 

v_lshl_add_u32 v35, v34, 6, v64 // (op 22562) 


s_add_u32 s76, s77, s55 // (op 22562) 


v_lshrrev_b32 v34, 6, v24 // (op 22562) 

s_lshr_b32 s55, s76, s11 // (op 22562) 

v_lshlrev_b32 v24, 2, v34 // (op 22562) 

s_and_b32 s76, s11, 2147483648 // (op 22562) 
s_cmp_eq_i32 s76, 0 // (op 22562) 
s_mov_b32 s77, scc // (op 22562) 

s_cmp_eq_i32 s77, 0 // (op 22562) 
s_cselect_b32 s76, s54, s55 // (op 22562) 


s_mul_hi_u32 s55, s76, s12 // (op 22562) 
s_sub_u32 s77, s76, s55 // (op 22562) 
s_lshr_b32 s78, s77, 1 // (op 22562) 

s_add_u32 s77, s78, s55 // (op 22562) 


s_lshr_b32 s55, s77, s10 // (op 22562) 

s_and_b32 s77, s10, 2147483648 // (op 22562) 
s_cmp_eq_i32 s77, 0 // (op 22562) 
s_mov_b32 s78, scc // (op 22562) 

s_cmp_eq_i32 s78, 0 // (op 22562) 
s_cselect_b32 s77, s76, s55 // (op 22562) 


v_lshl_add_u32 v34, s77, 7, v35 // (op 22562) 


s_ashr_i64 s[78:79], s[2:3], 7 // (op 22562) 
s_mul_i32 s55, s76, s78 // (op 22562) 


s_sub_u32 s76, s54, s55 // (op 22562) 


v_lshl_add_u32 v35, s76, 2, v24 // (op 22562) 


v_mul_lo_u32 v24, v34, s40 // (op 22562) 
v_add_u32 v24, v24, v35 // (op 22562) 


s_mul_i32 s55, s40, s35 // (op 22564) most significant: low of low * high
s_mul_i32 s76, s41, s34 // (op 22564) most significant: low of high * low
s_mul_i32 s54, s40, s34 // (op 22564) least significant
s_mul_hi_u32 s77, s40, s34 // (op 22564) most significant: high of low * low
s_add_u32 s55, s55, s76 // (op 22564) most significant: sum
s_add_u32 s55, s55, s77 // (op 22564) most significant: sum


s_mov_b32 s76, s42 // (op 22564) reg expression
s_mov_b32 s77, s43 // (op 22564) reg expression
s_mov_b32 s78, s54 // (op 22564) reg expression
s_mov_b32 s79, 131072 // (op 22564) call()

s_and_b32 s54, s7, 4294967292 // (op 22589) 
v_and_b32 v34, 63, v1 // (op 22589) 
s_add_u32 s55, s0, s54 // (op 22589) 

v_lshrrev_b32 v35, 4, v34 // (op 22589) 
v_and_b32 v64, 3, v35 // (op 22589) 

s_mul_i32 s54, s16, s8 // (op 22589) 
s_add_u32 s54, s54, s55 // (op 22589) 

v_and_b32 v35, 15, v1 // (op 22589) 
s_mul_hi_u32 s55, s54, s13 // (op 22589) 
v_lshl_add_u32 v65, v64, 4, v35 // (op 22589) 


s_sub_u32 s80, s54, s55 // (op 22589) 
v_lshrrev_b32 v35, 6, v1 // (op 22589) 
s_lshr_b32 s81, s80, 1 // (op 22589) 

v_and_b32 v64, 1, v35 // (op 22589) 

s_add_u32 s80, s81, s55 // (op 22589) 


v_lshl_add_u32 v35, v64, 6, v65 // (op 22589) 


s_lshr_b32 s55, s80, s11 // (op 22589) 

v_lshrrev_b32 v64, 6, v34 // (op 22589) 

s_and_b32 s80, s11, 2147483648 // (op 22589) 
v_lshlrev_b32 v34, 2, v64 // (op 22589) 

s_cmp_eq_i32 s80, 0 // (op 22589) 
s_mov_b32 s81, scc // (op 22589) 

s_cmp_eq_i32 s81, 0 // (op 22589) 
s_cselect_b32 s80, s54, s55 // (op 22589) 


s_mul_hi_u32 s55, s80, s12 // (op 22589) 
s_sub_u32 s81, s80, s55 // (op 22589) 
s_lshr_b32 s82, s81, 1 // (op 22589) 

s_add_u32 s81, s82, s55 // (op 22589) 


s_lshr_b32 s55, s81, s10 // (op 22589) 

s_and_b32 s81, s10, 2147483648 // (op 22589) 
s_cmp_eq_i32 s81, 0 // (op 22589) 
s_mov_b32 s82, scc // (op 22589) 

s_cmp_eq_i32 s82, 0 // (op 22589) 
s_cselect_b32 s81, s80, s55 // (op 22589) 


s_mul_i32 s55, s81, s14 // (op 22589) 

s_sub_u32 s81, s80, s55 // (op 22589) 

v_lshl_add_u32 v64, s81, 7, v35 // (op 22589) 


v_mul_lo_u32 v35, v64, s44 // (op 22589) 

s_ashr_i64 s[82:83], s[2:3], 7 // (op 22589) 
s_mul_i32 s55, s80, s82 // (op 22589) 


s_sub_u32 s80, s54, s55 // (op 22589) 


v_lshl_add_u32 v64, s80, 2, v34 // (op 22589) 


v_add_u32 v34, v64, v35 // (op 22589) 


s_mul_i32 s55, s44, s23 // (op 22591) most significant: low of low * high
s_mul_i32 s80, s45, s22 // (op 22591) most significant: low of high * low
s_mul_i32 s54, s44, s22 // (op 22591) least significant
s_mul_hi_u32 s81, s44, s22 // (op 22591) most significant: high of low * low
s_add_u32 s55, s55, s80 // (op 22591) most significant: sum
s_add_u32 s55, s55, s81 // (op 22591) most significant: sum


s_mov_b32 s80, s46 // (op 22591) reg expression
s_mov_b32 s81, s47 // (op 22591) reg expression
s_mov_b32 s82, s54 // (op 22591) reg expression
s_mov_b32 s83, 131072 // (op 22591) call()

s_and_b32 s54, s7, 4294967292 // (op 8827) 
s_cmp_lt_u32 s54, s7 // (op 8826) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_8826 // (op 8826) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_8826 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_8826:
s_barrier  // (op 23015) 
s_add_u32 s54, s54, 1 // (op 8828) 

s_lshl_b64 s[84:85], s[30:31], 6 // (op 20969) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 20969) 

v_lshrrev_b64 v[26:27], 3, s[86:87] // (op 20969) 


v_mov_b64 v[28:29], v[32:33] // (op 20969) 
v_mov_b64 v[36:37], v[28:29] // (op 20969) 
s_add_u32 s55, 8192, s52 // (op 20969) 
s_mov_b32 m0, s55 // (op 20969) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 20969) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v26 // (op 20969) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 20969) most significant half

s_add_u32 m0, m0, 4096 // (op 20969) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 20969) Load value direct to lds




v_add_co_u32 v32, s[84:85], 64, v32 // (op 21080) least significant half
v_addc_co_u32 v33, s[84:85], 0, v33, s[84:85] // (op 21080) most significant half

s_barrier  // (op 23035) 

s_lshl_b64 s[84:85], s[36:37], 6 // (op 20975) 
s_lshl_b64 s[86:87], s[84:85], 2 // (op 20975) 

v_lshrrev_b64 v[26:27], 3, s[86:87] // (op 20975) 


v_mov_b64 v[28:29], v[30:31] // (op 20975) 
v_mov_b64 v[36:37], v[28:29] // (op 20975) 
s_mov_b32 m0, s53 // (op 20975) reg expression
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 20975) Load value direct to lds
v_add_co_u32 v36, s[84:85], v36, v26 // (op 20975) least significant half
v_addc_co_u32 v37, s[84:85], 0, v37, s[84:85] // (op 20975) most significant half

s_add_u32 m0, m0, 4096 // (op 20975) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 20975) Load value direct to lds



v_add_co_u32 v30, s[84:85], 64, v30 // (op 21246) least significant half
v_addc_co_u32 v31, s[84:85], 0, v31, s[84:85] // (op 21246) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 23022) 



ds_read_b128 v[26:29], v3 offset:8192 // (op 5698) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 23041) 



ds_read_b128 v[36:39], v4 // (op 5700) Load local data 


v_mov_b32 v35, v24 // (op 5702) 
v_mov_b32 v40, v35 // (op 5702) reg expression

buffer_load_dwordx4 v[42:45], v40, s[76:79], 0 offen // (op 5702) Load value


v_add_u32 v24, 4, v24 // (op 22552) 


v_mov_b32 v35, v34 // (op 5704) 
v_mov_b32 v40, v35 // (op 5704) reg expression

buffer_load_dwordx4 v[46:49], v40, s[80:83], 0 offen // (op 5704) Load value


v_add_u32 v34, 4, v34 // (op 22579) 
s_waitcnt vmcnt(1)
v_bfe_u32 v50, v42, 0, 8 // (op 19529) 



v_bfe_u32 v51, v42, 8, 8 // (op 19529) 
v_bfe_u32 v52, v42, 16, 8 // (op 19529) 
v_bfe_u32 v53, v42, 24, 8 // (op 19529) 
v_permlane16_swap_b32 v50, v51 // (op 19529) 
ds_read_b128 v[54:57], v4 offset:1024 // (op 5935) Load local data 
v_permlane16_swap_b32 v52, v53 // (op 19529) 



s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 19529) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 19529) 
ds_read_b128 v[58:61], v4 offset:2048 // (op 6161) Load local data 




ds_read_b128 v[40:43], v4 offset:3072 // (op 6387) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v62, v46, 0, 8 // (op 19559) 



v_bfe_u32 v63, v46, 8, 8 // (op 19559) 
v_bfe_u32 v64, v46, 16, 8 // (op 19559) 
v_bfe_u32 v65, v46, 24, 8 // (op 19559) 
v_permlane16_swap_b32 v62, v63 // (op 19559) 
s_nop 0
v_permlane16_swap_b32 v64, v65 // (op 19559) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v62, v64 // (op 19559) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v63, v65 // (op 19559) 
ds_read_b128 v[66:69], v3 offset:9216 // (op 6747) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[36:39], a[0:3], v50, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5705) 
ds_read_b128 v[44:47], v3 offset:10240 // (op 7749) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[54:57], a[4:7], v50, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5938) 
ds_read_b128 v[70:73], v3 offset:11264 // (op 8751) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:29], v[58:61], a[8:11], v50, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6164) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:29], v[40:43], a[12:15], v50, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6390) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[66:69], v[36:39], a[16:19], v51, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6751) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:69], v[58:61], a[24:27], v51, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6805) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[66:69], v[40:43], a[28:31], v51, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6832) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[66:69], v[54:57], a[20:23], v51, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6778) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[44:47], v[36:39], a[32:35], v52, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7753) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[44:47], v[58:61], a[40:43], v52, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7807) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[44:47], v[40:43], a[44:47], v52, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7834) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[44:47], v[54:57], a[36:39], v52, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7780) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[70:73], v[36:39], a[48:51], v53, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8755) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:73], v[58:61], a[56:59], v53, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8809) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[70:73], v[40:43], a[60:63], v53, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8836) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[70:73], v[54:57], a[52:55], v53, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8782) 




s_cmp_lt_u32 s54, s7 // (op 8826) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_8826 // (op 8826) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_8826 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_8826:









s_barrier  // (op 23535) 


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 257) 
s_cmp_lg_u32 s18, 0 // (op 283) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_283 // (op 283) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_283



v_add_u32 v21, 49152, v20 // (op 10305) 
buffer_store_dword a48, v21, s[64:67], 0 offen sc1 // (op 10305) Store value

v_add_u32 v21, 49664, v20 // (op 10305) 
buffer_store_dword a49, v21, s[64:67], 0 offen sc1 // (op 10305) Store value

v_add_u32 v21, 50176, v20 // (op 10305) 
buffer_store_dword a50, v21, s[64:67], 0 offen sc1 // (op 10305) Store value

v_add_u32 v21, 50688, v20 // (op 10305) 
buffer_store_dword a51, v21, s[64:67], 0 offen sc1 // (op 10305) Store value




v_add_u32 v21, 49280, v20 // (op 10307) 
buffer_store_dword a52, v21, s[64:67], 0 offen sc1 // (op 10307) Store value

v_add_u32 v21, 49792, v20 // (op 10307) 
buffer_store_dword a53, v21, s[64:67], 0 offen sc1 // (op 10307) Store value

v_add_u32 v21, 50304, v20 // (op 10307) 
buffer_store_dword a54, v21, s[64:67], 0 offen sc1 // (op 10307) Store value

v_add_u32 v21, 50816, v20 // (op 10307) 
buffer_store_dword a55, v21, s[64:67], 0 offen sc1 // (op 10307) Store value




v_add_u32 v21, 49408, v20 // (op 10309) 
buffer_store_dword a56, v21, s[64:67], 0 offen sc1 // (op 10309) Store value

v_add_u32 v21, 49920, v20 // (op 10309) 
buffer_store_dword a57, v21, s[64:67], 0 offen sc1 // (op 10309) Store value

v_add_u32 v21, 50432, v20 // (op 10309) 
buffer_store_dword a58, v21, s[64:67], 0 offen sc1 // (op 10309) Store value

v_add_u32 v21, 50944, v20 // (op 10309) 
buffer_store_dword a59, v21, s[64:67], 0 offen sc1 // (op 10309) Store value




v_add_u32 v21, 49536, v20 // (op 10311) 
buffer_store_dword a60, v21, s[64:67], 0 offen sc1 // (op 10311) Store value

v_add_u32 v21, 50048, v20 // (op 10311) 
buffer_store_dword a61, v21, s[64:67], 0 offen sc1 // (op 10311) Store value

v_add_u32 v21, 50560, v20 // (op 10311) 
buffer_store_dword a62, v21, s[64:67], 0 offen sc1 // (op 10311) Store value

v_add_u32 v21, 51072, v20 // (op 10311) 
buffer_store_dword a63, v21, s[64:67], 0 offen sc1 // (op 10311) Store value




v_add_u32 v21, 32768, v20 // (op 10281) 
buffer_store_dword a32, v21, s[64:67], 0 offen sc1 // (op 10281) Store value

v_add_u32 v21, 33280, v20 // (op 10281) 
buffer_store_dword a33, v21, s[64:67], 0 offen sc1 // (op 10281) Store value

v_add_u32 v21, 33792, v20 // (op 10281) 
buffer_store_dword a34, v21, s[64:67], 0 offen sc1 // (op 10281) Store value

v_add_u32 v21, 34304, v20 // (op 10281) 
buffer_store_dword a35, v21, s[64:67], 0 offen sc1 // (op 10281) Store value




v_add_u32 v21, 32896, v20 // (op 10283) 
buffer_store_dword a36, v21, s[64:67], 0 offen sc1 // (op 10283) Store value

v_add_u32 v21, 33408, v20 // (op 10283) 
buffer_store_dword a37, v21, s[64:67], 0 offen sc1 // (op 10283) Store value

v_add_u32 v21, 33920, v20 // (op 10283) 
buffer_store_dword a38, v21, s[64:67], 0 offen sc1 // (op 10283) Store value

v_add_u32 v21, 34432, v20 // (op 10283) 
buffer_store_dword a39, v21, s[64:67], 0 offen sc1 // (op 10283) Store value




v_add_u32 v21, 33024, v20 // (op 10285) 
buffer_store_dword a40, v21, s[64:67], 0 offen sc1 // (op 10285) Store value

v_add_u32 v21, 33536, v20 // (op 10285) 
buffer_store_dword a41, v21, s[64:67], 0 offen sc1 // (op 10285) Store value

v_add_u32 v21, 34048, v20 // (op 10285) 
buffer_store_dword a42, v21, s[64:67], 0 offen sc1 // (op 10285) Store value

v_add_u32 v21, 34560, v20 // (op 10285) 
buffer_store_dword a43, v21, s[64:67], 0 offen sc1 // (op 10285) Store value




v_add_u32 v21, 33152, v20 // (op 10287) 
buffer_store_dword a44, v21, s[64:67], 0 offen sc1 // (op 10287) Store value

v_add_u32 v21, 33664, v20 // (op 10287) 
buffer_store_dword a45, v21, s[64:67], 0 offen sc1 // (op 10287) Store value

v_add_u32 v21, 34176, v20 // (op 10287) 
buffer_store_dword a46, v21, s[64:67], 0 offen sc1 // (op 10287) Store value

v_add_u32 v21, 34688, v20 // (op 10287) 
buffer_store_dword a47, v21, s[64:67], 0 offen sc1 // (op 10287) Store value




v_add_u32 v21, 16384, v20 // (op 10257) 
buffer_store_dword a16, v21, s[64:67], 0 offen sc1 // (op 10257) Store value

v_add_u32 v21, 16896, v20 // (op 10257) 
buffer_store_dword a17, v21, s[64:67], 0 offen sc1 // (op 10257) Store value

v_add_u32 v21, 17408, v20 // (op 10257) 
buffer_store_dword a18, v21, s[64:67], 0 offen sc1 // (op 10257) Store value

v_add_u32 v21, 17920, v20 // (op 10257) 
buffer_store_dword a19, v21, s[64:67], 0 offen sc1 // (op 10257) Store value




v_add_u32 v21, 16512, v20 // (op 10259) 
buffer_store_dword a20, v21, s[64:67], 0 offen sc1 // (op 10259) Store value

v_add_u32 v21, 17024, v20 // (op 10259) 
buffer_store_dword a21, v21, s[64:67], 0 offen sc1 // (op 10259) Store value

v_add_u32 v21, 17536, v20 // (op 10259) 
buffer_store_dword a22, v21, s[64:67], 0 offen sc1 // (op 10259) Store value

v_add_u32 v21, 18048, v20 // (op 10259) 
buffer_store_dword a23, v21, s[64:67], 0 offen sc1 // (op 10259) Store value




v_add_u32 v21, 16640, v20 // (op 10261) 
buffer_store_dword a24, v21, s[64:67], 0 offen sc1 // (op 10261) Store value

v_add_u32 v21, 17152, v20 // (op 10261) 
buffer_store_dword a25, v21, s[64:67], 0 offen sc1 // (op 10261) Store value

v_add_u32 v21, 17664, v20 // (op 10261) 
buffer_store_dword a26, v21, s[64:67], 0 offen sc1 // (op 10261) Store value

v_add_u32 v21, 18176, v20 // (op 10261) 
buffer_store_dword a27, v21, s[64:67], 0 offen sc1 // (op 10261) Store value




v_add_u32 v21, 16768, v20 // (op 10263) 
buffer_store_dword a28, v21, s[64:67], 0 offen sc1 // (op 10263) Store value

v_add_u32 v21, 17280, v20 // (op 10263) 
buffer_store_dword a29, v21, s[64:67], 0 offen sc1 // (op 10263) Store value

v_add_u32 v21, 17792, v20 // (op 10263) 
buffer_store_dword a30, v21, s[64:67], 0 offen sc1 // (op 10263) Store value

v_add_u32 v21, 18304, v20 // (op 10263) 
buffer_store_dword a31, v21, s[64:67], 0 offen sc1 // (op 10263) Store value




buffer_store_dword a0, v20, s[64:67], 0 offen sc1 // (op 280) Store value
buffer_store_dword a1, v20, s[64:67], 0 offen offset:512 sc1 // (op 280) Store value
buffer_store_dword a2, v20, s[64:67], 0 offen offset:1024 sc1 // (op 280) Store value
buffer_store_dword a3, v20, s[64:67], 0 offen offset:1536 sc1 // (op 280) Store value



buffer_store_dword a4, v20, s[64:67], 0 offen offset:128 sc1 // (op 10231) Store value
buffer_store_dword a5, v20, s[64:67], 0 offen offset:640 sc1 // (op 10231) Store value
buffer_store_dword a6, v20, s[64:67], 0 offen offset:1152 sc1 // (op 10231) Store value
buffer_store_dword a7, v20, s[64:67], 0 offen offset:1664 sc1 // (op 10231) Store value



buffer_store_dword a8, v20, s[64:67], 0 offen offset:256 sc1 // (op 10232) Store value
buffer_store_dword a9, v20, s[64:67], 0 offen offset:768 sc1 // (op 10232) Store value
buffer_store_dword a10, v20, s[64:67], 0 offen offset:1280 sc1 // (op 10232) Store value
buffer_store_dword a11, v20, s[64:67], 0 offen offset:1792 sc1 // (op 10232) Store value



buffer_store_dword a12, v20, s[64:67], 0 offen offset:384 sc1 // (op 10233) Store value
buffer_store_dword a13, v20, s[64:67], 0 offen offset:896 sc1 // (op 10233) Store value
buffer_store_dword a14, v20, s[64:67], 0 offen offset:1408 sc1 // (op 10233) Store value
buffer_store_dword a15, v20, s[64:67], 0 offen offset:1920 sc1 // (op 10233) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 284) 
v_cmp_eq_i32 vcc, v1, 0 // (op 288) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_288 // (op 288) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_288
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
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_288 // (op 288) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_288
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Store_Flag_288:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Store_Flag_288:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_283 // (op 283) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_283
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Send_Tile_283:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Send_Tile_283:

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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_299 // (op 299) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_299
s_mov_b32 s52, 0 // (op 265) call()
s_mul_i32 s53, s52, s8 // (op 264) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 264) 
s_mov_b32 s76, s54 // (op 264) convert

s_sub_u32 s54, s76, s19 // (op 264) 

s_sub_u32 s55, s54, 1 // (op 264) 

s_cmp_lt_u32 s53, s55 // (op 264) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_264 // (op 264) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_264 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_264:
s_add_u32 s53, s16, 1 // (op 21801) 
v_lshrrev_b32 v20, 5, v1 // (op 21801) 
s_add_u32 s54, s53, s52 // (op 21801) 
s_lshl_b32 s54, s54, 7 // (op 21801) 

v_lshlrev_b32 v21, 2, v20 // (op 21801) 

v_add_lshl_u32 v20, s54, v21, 7 // (op 21801) 


v_and_b32 v21, -32, v1 // (op 21801) 
v_sub_u32 v23, v1, v21 // (op 21801) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 21801) 


s_lshl_b32 s53, s15, 16 // (op 21803) 
s_mov_b32 s64, s28 // (op 21803) reg expression
s_mov_b32 s65, s29 // (op 21803) reg expression
s_mov_b32 s66, s53 // (op 21803) reg expression
s_mov_b32 s67, 131072 // (op 21803) call()

RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_303:
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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_303 // (op 303) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_DoWhileTop_Global_sync_spin_loop_303 if true)
s_barrier  // (op 307) 
v_cmp_eq_i32 vcc, v1, 0 // (op 306) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_306 // (op 306) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_306
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
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_306 // (op 306) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_306
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Wave0_Reset_Flag_306:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Wave0_Reset_Flag_306:




buffer_load_dword v26, v21, s[64:67], 0 offen // (op 318) Load value
buffer_load_dword v27, v21, s[64:67], 0 offen offset:512 // (op 318) Load value
buffer_load_dword v28, v21, s[64:67], 0 offen offset:1024 // (op 318) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:1536 // (op 318) Load value
v_accvgpr_read v30, a0 // (op 321) reg expression
v_accvgpr_read v31, a1 // (op 321) reg expression
v_accvgpr_read v32, a2 // (op 321) reg expression
v_accvgpr_read v33, a3 // (op 321) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 323) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 323) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 323) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 323) 

v_accvgpr_write a0, v30 // (op 325) reg expression
v_accvgpr_write a1, v31 // (op 325) reg expression
v_accvgpr_write a2, v32 // (op 325) reg expression
v_accvgpr_write a3, v33 // (op 325) reg expression




buffer_load_dword v26, v21, s[64:67], 0 offen offset:128 // (op 10380) Load value
buffer_load_dword v27, v21, s[64:67], 0 offen offset:640 // (op 10380) Load value
buffer_load_dword v28, v21, s[64:67], 0 offen offset:1152 // (op 10380) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:1664 // (op 10380) Load value
v_accvgpr_read v30, a4 // (op 10381) reg expression
v_accvgpr_read v31, a5 // (op 10381) reg expression
v_accvgpr_read v32, a6 // (op 10381) reg expression
v_accvgpr_read v33, a7 // (op 10381) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10382) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10382) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10382) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10382) 

v_accvgpr_write a4, v30 // (op 10383) reg expression
v_accvgpr_write a5, v31 // (op 10383) reg expression
v_accvgpr_write a6, v32 // (op 10383) reg expression
v_accvgpr_write a7, v33 // (op 10383) reg expression




buffer_load_dword v26, v21, s[64:67], 0 offen offset:256 // (op 10387) Load value
buffer_load_dword v27, v21, s[64:67], 0 offen offset:768 // (op 10387) Load value
buffer_load_dword v28, v21, s[64:67], 0 offen offset:1280 // (op 10387) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:1792 // (op 10387) Load value
v_accvgpr_read v30, a8 // (op 10388) reg expression
v_accvgpr_read v31, a9 // (op 10388) reg expression
v_accvgpr_read v32, a10 // (op 10388) reg expression
v_accvgpr_read v33, a11 // (op 10388) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10389) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10389) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10389) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10389) 

v_accvgpr_write a8, v30 // (op 10390) reg expression
v_accvgpr_write a9, v31 // (op 10390) reg expression
v_accvgpr_write a10, v32 // (op 10390) reg expression
v_accvgpr_write a11, v33 // (op 10390) reg expression




buffer_load_dword v26, v21, s[64:67], 0 offen offset:384 // (op 10394) Load value
buffer_load_dword v27, v21, s[64:67], 0 offen offset:896 // (op 10394) Load value
buffer_load_dword v28, v21, s[64:67], 0 offen offset:1408 // (op 10394) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:1920 // (op 10394) Load value
v_accvgpr_read v30, a12 // (op 10395) reg expression
v_accvgpr_read v31, a13 // (op 10395) reg expression
v_accvgpr_read v32, a14 // (op 10395) reg expression
v_accvgpr_read v33, a15 // (op 10395) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10396) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10396) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10396) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10396) 

v_accvgpr_write a12, v30 // (op 10397) reg expression
v_accvgpr_write a13, v31 // (op 10397) reg expression
v_accvgpr_write a14, v32 // (op 10397) reg expression
v_accvgpr_write a15, v33 // (op 10397) reg expression




v_mov_b32 v20, 16384 // (op 10431) 
v_add_i32 v23, v21, v20 // (op 10431) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10431) Load value

v_mov_b32 v20, 16896 // (op 10431) 
v_add_i32 v23, v21, v20 // (op 10431) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10431) Load value

v_mov_b32 v20, 17408 // (op 10431) 
v_add_i32 v23, v21, v20 // (op 10431) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10431) Load value

v_mov_b32 v20, 17920 // (op 10431) 
v_add_i32 v23, v21, v20 // (op 10431) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10431) Load value

v_accvgpr_read v30, a16 // (op 10428) reg expression
v_accvgpr_read v31, a17 // (op 10428) reg expression
v_accvgpr_read v32, a18 // (op 10428) reg expression
v_accvgpr_read v33, a19 // (op 10428) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10429) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10429) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10429) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10429) 

v_accvgpr_write a16, v30 // (op 10430) reg expression
v_accvgpr_write a17, v31 // (op 10430) reg expression
v_accvgpr_write a18, v32 // (op 10430) reg expression
v_accvgpr_write a19, v33 // (op 10430) reg expression




v_mov_b32 v20, 16512 // (op 10436) 
v_add_i32 v23, v21, v20 // (op 10436) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10436) Load value

v_mov_b32 v20, 17024 // (op 10436) 
v_add_i32 v23, v21, v20 // (op 10436) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10436) Load value

v_mov_b32 v20, 17536 // (op 10436) 
v_add_i32 v23, v21, v20 // (op 10436) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10436) Load value

v_mov_b32 v20, 18048 // (op 10436) 
v_add_i32 v23, v21, v20 // (op 10436) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10436) Load value

v_accvgpr_read v30, a20 // (op 10433) reg expression
v_accvgpr_read v31, a21 // (op 10433) reg expression
v_accvgpr_read v32, a22 // (op 10433) reg expression
v_accvgpr_read v33, a23 // (op 10433) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10434) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10434) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10434) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10434) 

v_accvgpr_write a20, v30 // (op 10435) reg expression
v_accvgpr_write a21, v31 // (op 10435) reg expression
v_accvgpr_write a22, v32 // (op 10435) reg expression
v_accvgpr_write a23, v33 // (op 10435) reg expression




v_mov_b32 v20, 16640 // (op 10441) 
v_add_i32 v23, v21, v20 // (op 10441) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10441) Load value

v_mov_b32 v20, 17152 // (op 10441) 
v_add_i32 v23, v21, v20 // (op 10441) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10441) Load value

v_mov_b32 v20, 17664 // (op 10441) 
v_add_i32 v23, v21, v20 // (op 10441) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10441) Load value

v_mov_b32 v20, 18176 // (op 10441) 
v_add_i32 v23, v21, v20 // (op 10441) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10441) Load value

v_accvgpr_read v30, a24 // (op 10438) reg expression
v_accvgpr_read v31, a25 // (op 10438) reg expression
v_accvgpr_read v32, a26 // (op 10438) reg expression
v_accvgpr_read v33, a27 // (op 10438) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10439) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10439) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10439) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10439) 

v_accvgpr_write a24, v30 // (op 10440) reg expression
v_accvgpr_write a25, v31 // (op 10440) reg expression
v_accvgpr_write a26, v32 // (op 10440) reg expression
v_accvgpr_write a27, v33 // (op 10440) reg expression




v_mov_b32 v20, 16768 // (op 10446) 
v_add_i32 v23, v21, v20 // (op 10446) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10446) Load value

v_mov_b32 v20, 17280 // (op 10446) 
v_add_i32 v23, v21, v20 // (op 10446) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10446) Load value

v_mov_b32 v20, 17792 // (op 10446) 
v_add_i32 v23, v21, v20 // (op 10446) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10446) Load value

v_mov_b32 v20, 18304 // (op 10446) 
v_add_i32 v23, v21, v20 // (op 10446) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10446) Load value

v_accvgpr_read v30, a28 // (op 10443) reg expression
v_accvgpr_read v31, a29 // (op 10443) reg expression
v_accvgpr_read v32, a30 // (op 10443) reg expression
v_accvgpr_read v33, a31 // (op 10443) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10444) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10444) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10444) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10444) 

v_accvgpr_write a28, v30 // (op 10445) reg expression
v_accvgpr_write a29, v31 // (op 10445) reg expression
v_accvgpr_write a30, v32 // (op 10445) reg expression
v_accvgpr_write a31, v33 // (op 10445) reg expression




v_mov_b32 v20, 32768 // (op 10479) 
v_add_i32 v23, v21, v20 // (op 10479) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10479) Load value

v_mov_b32 v20, 33280 // (op 10479) 
v_add_i32 v23, v21, v20 // (op 10479) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10479) Load value

v_mov_b32 v20, 33792 // (op 10479) 
v_add_i32 v23, v21, v20 // (op 10479) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10479) Load value

v_mov_b32 v20, 34304 // (op 10479) 
v_add_i32 v23, v21, v20 // (op 10479) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10479) Load value

v_accvgpr_read v30, a32 // (op 10476) reg expression
v_accvgpr_read v31, a33 // (op 10476) reg expression
v_accvgpr_read v32, a34 // (op 10476) reg expression
v_accvgpr_read v33, a35 // (op 10476) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10477) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10477) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10477) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10477) 

v_accvgpr_write a32, v30 // (op 10478) reg expression
v_accvgpr_write a33, v31 // (op 10478) reg expression
v_accvgpr_write a34, v32 // (op 10478) reg expression
v_accvgpr_write a35, v33 // (op 10478) reg expression




v_mov_b32 v20, 32896 // (op 10484) 
v_add_i32 v23, v21, v20 // (op 10484) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10484) Load value

v_mov_b32 v20, 33408 // (op 10484) 
v_add_i32 v23, v21, v20 // (op 10484) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10484) Load value

v_mov_b32 v20, 33920 // (op 10484) 
v_add_i32 v23, v21, v20 // (op 10484) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10484) Load value

v_mov_b32 v20, 34432 // (op 10484) 
v_add_i32 v23, v21, v20 // (op 10484) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10484) Load value

v_accvgpr_read v30, a36 // (op 10481) reg expression
v_accvgpr_read v31, a37 // (op 10481) reg expression
v_accvgpr_read v32, a38 // (op 10481) reg expression
v_accvgpr_read v33, a39 // (op 10481) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10482) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10482) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10482) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10482) 

v_accvgpr_write a36, v30 // (op 10483) reg expression
v_accvgpr_write a37, v31 // (op 10483) reg expression
v_accvgpr_write a38, v32 // (op 10483) reg expression
v_accvgpr_write a39, v33 // (op 10483) reg expression




v_mov_b32 v20, 33024 // (op 10489) 
v_add_i32 v23, v21, v20 // (op 10489) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10489) Load value

v_mov_b32 v20, 33536 // (op 10489) 
v_add_i32 v23, v21, v20 // (op 10489) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10489) Load value

v_mov_b32 v20, 34048 // (op 10489) 
v_add_i32 v23, v21, v20 // (op 10489) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10489) Load value

v_mov_b32 v20, 34560 // (op 10489) 
v_add_i32 v23, v21, v20 // (op 10489) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10489) Load value

v_accvgpr_read v30, a40 // (op 10486) reg expression
v_accvgpr_read v31, a41 // (op 10486) reg expression
v_accvgpr_read v32, a42 // (op 10486) reg expression
v_accvgpr_read v33, a43 // (op 10486) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10487) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10487) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10487) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10487) 

v_accvgpr_write a40, v30 // (op 10488) reg expression
v_accvgpr_write a41, v31 // (op 10488) reg expression
v_accvgpr_write a42, v32 // (op 10488) reg expression
v_accvgpr_write a43, v33 // (op 10488) reg expression




v_mov_b32 v20, 33152 // (op 10494) 
v_add_i32 v23, v21, v20 // (op 10494) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10494) Load value

v_mov_b32 v20, 33664 // (op 10494) 
v_add_i32 v23, v21, v20 // (op 10494) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10494) Load value

v_mov_b32 v20, 34176 // (op 10494) 
v_add_i32 v23, v21, v20 // (op 10494) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10494) Load value

v_mov_b32 v20, 34688 // (op 10494) 
v_add_i32 v23, v21, v20 // (op 10494) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10494) Load value

v_accvgpr_read v30, a44 // (op 10491) reg expression
v_accvgpr_read v31, a45 // (op 10491) reg expression
v_accvgpr_read v32, a46 // (op 10491) reg expression
v_accvgpr_read v33, a47 // (op 10491) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10492) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10492) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10492) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10492) 

v_accvgpr_write a44, v30 // (op 10493) reg expression
v_accvgpr_write a45, v31 // (op 10493) reg expression
v_accvgpr_write a46, v32 // (op 10493) reg expression
v_accvgpr_write a47, v33 // (op 10493) reg expression




v_mov_b32 v20, 49152 // (op 10527) 
v_add_i32 v23, v21, v20 // (op 10527) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10527) Load value

v_mov_b32 v20, 49664 // (op 10527) 
v_add_i32 v23, v21, v20 // (op 10527) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10527) Load value

v_mov_b32 v20, 50176 // (op 10527) 
v_add_i32 v23, v21, v20 // (op 10527) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10527) Load value

v_mov_b32 v20, 50688 // (op 10527) 
v_add_i32 v23, v21, v20 // (op 10527) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10527) Load value

v_accvgpr_read v30, a48 // (op 10524) reg expression
v_accvgpr_read v31, a49 // (op 10524) reg expression
v_accvgpr_read v32, a50 // (op 10524) reg expression
v_accvgpr_read v33, a51 // (op 10524) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10525) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10525) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10525) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10525) 

v_accvgpr_write a48, v30 // (op 10526) reg expression
v_accvgpr_write a49, v31 // (op 10526) reg expression
v_accvgpr_write a50, v32 // (op 10526) reg expression
v_accvgpr_write a51, v33 // (op 10526) reg expression




v_mov_b32 v20, 49280 // (op 10532) 
v_add_i32 v23, v21, v20 // (op 10532) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10532) Load value

v_mov_b32 v20, 49792 // (op 10532) 
v_add_i32 v23, v21, v20 // (op 10532) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10532) Load value

v_mov_b32 v20, 50304 // (op 10532) 
v_add_i32 v23, v21, v20 // (op 10532) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10532) Load value

v_mov_b32 v20, 50816 // (op 10532) 
v_add_i32 v23, v21, v20 // (op 10532) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10532) Load value

v_accvgpr_read v30, a52 // (op 10529) reg expression
v_accvgpr_read v31, a53 // (op 10529) reg expression
v_accvgpr_read v32, a54 // (op 10529) reg expression
v_accvgpr_read v33, a55 // (op 10529) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10530) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10530) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10530) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10530) 

v_accvgpr_write a52, v30 // (op 10531) reg expression
v_accvgpr_write a53, v31 // (op 10531) reg expression
v_accvgpr_write a54, v32 // (op 10531) reg expression
v_accvgpr_write a55, v33 // (op 10531) reg expression




v_mov_b32 v20, 49408 // (op 10537) 
v_add_i32 v23, v21, v20 // (op 10537) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10537) Load value

v_mov_b32 v20, 49920 // (op 10537) 
v_add_i32 v23, v21, v20 // (op 10537) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10537) Load value

v_mov_b32 v20, 50432 // (op 10537) 
v_add_i32 v23, v21, v20 // (op 10537) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10537) Load value

v_mov_b32 v20, 50944 // (op 10537) 
v_add_i32 v23, v21, v20 // (op 10537) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10537) Load value

v_accvgpr_read v30, a56 // (op 10534) reg expression
v_accvgpr_read v31, a57 // (op 10534) reg expression
v_accvgpr_read v32, a58 // (op 10534) reg expression
v_accvgpr_read v33, a59 // (op 10534) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10535) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10535) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10535) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10535) 

v_accvgpr_write a56, v30 // (op 10536) reg expression
v_accvgpr_write a57, v31 // (op 10536) reg expression
v_accvgpr_write a58, v32 // (op 10536) reg expression
v_accvgpr_write a59, v33 // (op 10536) reg expression




v_mov_b32 v20, 49536 // (op 10542) 
v_add_i32 v23, v21, v20 // (op 10542) 

buffer_load_dword v26, v23, s[64:67], 0 offen // (op 10542) Load value

v_mov_b32 v20, 50048 // (op 10542) 
v_add_i32 v23, v21, v20 // (op 10542) 

buffer_load_dword v27, v23, s[64:67], 0 offen // (op 10542) Load value

v_mov_b32 v20, 50560 // (op 10542) 
v_add_i32 v23, v21, v20 // (op 10542) 

buffer_load_dword v28, v23, s[64:67], 0 offen // (op 10542) Load value

v_mov_b32 v20, 51072 // (op 10542) 
v_add_i32 v23, v21, v20 // (op 10542) 

buffer_load_dword v29, v23, s[64:67], 0 offen // (op 10542) Load value

v_accvgpr_read v30, a60 // (op 10539) reg expression
v_accvgpr_read v31, a61 // (op 10539) reg expression
v_accvgpr_read v32, a62 // (op 10539) reg expression
v_accvgpr_read v33, a63 // (op 10539) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v30, v30, v26 // (op 10540) 
s_waitcnt vmcnt(2)
v_add_f32 v31, v31, v27 // (op 10540) 
s_waitcnt vmcnt(1)
v_add_f32 v32, v32, v28 // (op 10540) 
s_waitcnt vmcnt(0)
v_add_f32 v33, v33, v29 // (op 10540) 

v_accvgpr_write a60, v30 // (op 10541) reg expression
v_accvgpr_write a61, v31 // (op 10541) reg expression
v_accvgpr_write a62, v32 // (op 10541) reg expression
v_accvgpr_write a63, v33 // (op 10541) reg expression

s_add_u32 s52, s52, 1 // (op 266) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 264) 
s_ashr_i64 s[54:55], s[2:3], 7 // (op 264) 
s_mov_b32 s76, s54 // (op 264) convert

s_sub_u32 s54, s76, s19 // (op 264) 

s_sub_u32 s55, s54, 1 // (op 264) 

s_cmp_lt_u32 s53, s55 // (op 264) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_264 // (op 264) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_ReceiveTileLoop_264 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_ReceiveTileLoop_264:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_299 // (op 299) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_299
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Receive_Tile_299:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Receive_Tile_299:

s_cmp_eq_i32 s18, 0 // (op 342) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_342 // (op 342) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_342
v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_114
v_accvgpr_read v26, a0 // (op 42) 
v_accvgpr_read v27, a1 // (op 42) 
v_accvgpr_read v28, a2 // (op 42) 
v_accvgpr_read v29, a3 // (op 42) 
v_mul_f32 v30, v19, v26 // (op 42) 
v_mul_f32 v31, v19, v27 // (op 42) 
v_mul_f32 v32, v19, v28 // (op 42) 
v_mul_f32 v33, v19, v29 // (op 42) 



v_mov_b32 v20, v25 // (op 48) reg expression

buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 48) Store value


v_accvgpr_read v26, a4 // (op 10624) 
v_accvgpr_read v27, a5 // (op 10624) 
v_accvgpr_read v28, a6 // (op 10624) 
v_accvgpr_read v29, a7 // (op 10624) 
v_mul_f32 v30, v19, v26 // (op 10624) 
v_mul_f32 v31, v19, v27 // (op 10624) 
v_mul_f32 v32, v19, v28 // (op 10624) 
v_mul_f32 v33, v19, v29 // (op 10624) 



s_lshl_b32 s52, s24, 6 // (op 10626) 
v_add_u32 v20, v25, s52 // (op 10626) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10626) Store value


v_accvgpr_read v26, a8 // (op 10636) 
v_accvgpr_read v27, a9 // (op 10636) 
v_accvgpr_read v28, a10 // (op 10636) 
v_accvgpr_read v29, a11 // (op 10636) 
v_mul_f32 v30, v19, v26 // (op 10636) 
v_mul_f32 v31, v19, v27 // (op 10636) 
v_mul_f32 v32, v19, v28 // (op 10636) 
v_mul_f32 v33, v19, v29 // (op 10636) 



s_lshl_b32 s52, s24, 7 // (op 10638) 
v_add_u32 v20, v25, s52 // (op 10638) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10638) Store value


v_accvgpr_read v26, a12 // (op 10648) 
v_accvgpr_read v27, a13 // (op 10648) 
v_accvgpr_read v28, a14 // (op 10648) 
v_accvgpr_read v29, a15 // (op 10648) 
v_mul_f32 v30, v19, v26 // (op 10648) 
v_mul_f32 v31, v19, v27 // (op 10648) 
v_mul_f32 v32, v19, v28 // (op 10648) 
v_mul_f32 v33, v19, v29 // (op 10648) 



s_lshl_b32 s52, s24, 6 // (op 10650) 
s_mul_i32 s53, 3, s52 // (op 10650) 

v_add_u32 v20, v25, s53 // (op 10650) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10650) Store value


v_accvgpr_read v26, a16 // (op 10682) 
v_accvgpr_read v27, a17 // (op 10682) 
v_accvgpr_read v28, a18 // (op 10682) 
v_accvgpr_read v29, a19 // (op 10682) 
v_mul_f32 v30, v19, v26 // (op 10682) 
v_mul_f32 v31, v19, v27 // (op 10682) 
v_mul_f32 v32, v19, v28 // (op 10682) 
v_mul_f32 v33, v19, v29 // (op 10682) 



v_add_u32 v20, 64, v25 // (op 10685) 

buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10685) Store value


v_accvgpr_read v26, a20 // (op 10695) 
v_accvgpr_read v27, a21 // (op 10695) 
v_accvgpr_read v28, a22 // (op 10695) 
v_accvgpr_read v29, a23 // (op 10695) 
v_mul_f32 v30, v19, v26 // (op 10695) 
v_mul_f32 v31, v19, v27 // (op 10695) 
v_mul_f32 v32, v19, v28 // (op 10695) 
v_mul_f32 v33, v19, v29 // (op 10695) 



s_lshl_b32 s52, s24, 6 // (op 10687) 
s_add_u32 s52, s52, 64 // (op 10687) 
v_add_u32 v20, v25, s52 // (op 10687) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10687) Store value


v_accvgpr_read v26, a24 // (op 10700) 
v_accvgpr_read v27, a25 // (op 10700) 
v_accvgpr_read v28, a26 // (op 10700) 
v_accvgpr_read v29, a27 // (op 10700) 
v_mul_f32 v30, v19, v26 // (op 10700) 
v_mul_f32 v31, v19, v27 // (op 10700) 
v_mul_f32 v32, v19, v28 // (op 10700) 
v_mul_f32 v33, v19, v29 // (op 10700) 



s_lshl_b32 s52, s24, 7 // (op 10689) 
s_add_u32 s52, s52, 64 // (op 10689) 
v_add_u32 v20, v25, s52 // (op 10689) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10689) Store value


v_accvgpr_read v26, a28 // (op 10705) 
v_accvgpr_read v27, a29 // (op 10705) 
v_accvgpr_read v28, a30 // (op 10705) 
v_accvgpr_read v29, a31 // (op 10705) 
v_mul_f32 v30, v19, v26 // (op 10705) 
v_mul_f32 v31, v19, v27 // (op 10705) 
v_mul_f32 v32, v19, v28 // (op 10705) 
v_mul_f32 v33, v19, v29 // (op 10705) 



s_lshl_b32 s52, s24, 6 // (op 10691) 
s_mul_i32 s53, 3, s52 // (op 10691) 
s_add_u32 s53, s53, 64 // (op 10691) 

v_add_u32 v20, v25, s53 // (op 10691) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10691) Store value


v_accvgpr_read v26, a32 // (op 10750) 
v_accvgpr_read v27, a33 // (op 10750) 
v_accvgpr_read v28, a34 // (op 10750) 
v_accvgpr_read v29, a35 // (op 10750) 
v_mul_f32 v30, v19, v26 // (op 10750) 
v_mul_f32 v31, v19, v27 // (op 10750) 
v_mul_f32 v32, v19, v28 // (op 10750) 
v_mul_f32 v33, v19, v29 // (op 10750) 



v_add_u32 v20, 128, v25 // (op 10753) 

buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10753) Store value


v_accvgpr_read v26, a36 // (op 10763) 
v_accvgpr_read v27, a37 // (op 10763) 
v_accvgpr_read v28, a38 // (op 10763) 
v_accvgpr_read v29, a39 // (op 10763) 
v_mul_f32 v30, v19, v26 // (op 10763) 
v_mul_f32 v31, v19, v27 // (op 10763) 
v_mul_f32 v32, v19, v28 // (op 10763) 
v_mul_f32 v33, v19, v29 // (op 10763) 



s_lshl_b32 s52, s24, 6 // (op 10755) 
s_add_u32 s52, s52, 128 // (op 10755) 
v_add_u32 v20, v25, s52 // (op 10755) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10755) Store value


v_accvgpr_read v26, a40 // (op 10768) 
v_accvgpr_read v27, a41 // (op 10768) 
v_accvgpr_read v28, a42 // (op 10768) 
v_accvgpr_read v29, a43 // (op 10768) 
v_mul_f32 v30, v19, v26 // (op 10768) 
v_mul_f32 v31, v19, v27 // (op 10768) 
v_mul_f32 v32, v19, v28 // (op 10768) 
v_mul_f32 v33, v19, v29 // (op 10768) 



s_lshl_b32 s52, s24, 7 // (op 10757) 
s_add_u32 s52, s52, 128 // (op 10757) 
v_add_u32 v20, v25, s52 // (op 10757) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10757) Store value


v_accvgpr_read v26, a44 // (op 10773) 
v_accvgpr_read v27, a45 // (op 10773) 
v_accvgpr_read v28, a46 // (op 10773) 
v_accvgpr_read v29, a47 // (op 10773) 
v_mul_f32 v30, v19, v26 // (op 10773) 
v_mul_f32 v31, v19, v27 // (op 10773) 
v_mul_f32 v32, v19, v28 // (op 10773) 
v_mul_f32 v33, v19, v29 // (op 10773) 



s_lshl_b32 s52, s24, 6 // (op 10759) 
s_mul_i32 s53, 3, s52 // (op 10759) 
s_add_u32 s53, s53, 128 // (op 10759) 

v_add_u32 v20, v25, s53 // (op 10759) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10759) Store value


v_accvgpr_read v26, a48 // (op 10818) 
v_accvgpr_read v27, a49 // (op 10818) 
v_accvgpr_read v28, a50 // (op 10818) 
v_accvgpr_read v29, a51 // (op 10818) 
v_mul_f32 v30, v19, v26 // (op 10818) 
v_mul_f32 v31, v19, v27 // (op 10818) 
v_mul_f32 v32, v19, v28 // (op 10818) 
v_mul_f32 v33, v19, v29 // (op 10818) 



v_add_u32 v20, 192, v25 // (op 10821) 

buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10821) Store value


v_accvgpr_read v26, a52 // (op 10831) 
v_accvgpr_read v27, a53 // (op 10831) 
v_accvgpr_read v28, a54 // (op 10831) 
v_accvgpr_read v29, a55 // (op 10831) 
v_mul_f32 v30, v19, v26 // (op 10831) 
v_mul_f32 v31, v19, v27 // (op 10831) 
v_mul_f32 v32, v19, v28 // (op 10831) 
v_mul_f32 v33, v19, v29 // (op 10831) 



s_lshl_b32 s52, s24, 6 // (op 10823) 
s_add_u32 s52, s52, 192 // (op 10823) 
v_add_u32 v20, v25, s52 // (op 10823) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10823) Store value


v_accvgpr_read v26, a56 // (op 10836) 
v_accvgpr_read v27, a57 // (op 10836) 
v_accvgpr_read v28, a58 // (op 10836) 
v_accvgpr_read v29, a59 // (op 10836) 
v_mul_f32 v30, v19, v26 // (op 10836) 
v_mul_f32 v31, v19, v27 // (op 10836) 
v_mul_f32 v32, v19, v28 // (op 10836) 
v_mul_f32 v33, v19, v29 // (op 10836) 



s_lshl_b32 s52, s24, 7 // (op 10825) 
s_add_u32 s52, s52, 192 // (op 10825) 
v_add_u32 v20, v25, s52 // (op 10825) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10825) Store value


v_accvgpr_read v26, a60 // (op 10841) 
v_accvgpr_read v27, a61 // (op 10841) 
v_accvgpr_read v28, a62 // (op 10841) 
v_accvgpr_read v29, a63 // (op 10841) 
v_mul_f32 v30, v19, v26 // (op 10841) 
v_mul_f32 v31, v19, v27 // (op 10841) 
v_mul_f32 v32, v19, v28 // (op 10841) 
v_mul_f32 v33, v19, v29 // (op 10841) 



s_lshl_b32 s52, s24, 6 // (op 10827) 
s_mul_i32 s53, 3, s52 // (op 10827) 
s_add_u32 s53, s53, 192 // (op 10827) 

v_add_u32 v20, v25, s53 // (op 10827) 


buffer_store_dwordx4 v[30:33], v20, s[60:63], 0 offen // (op 10827) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_114
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_114:


v_add_u32 v20, 128, v22 // (op 11106) 

buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11106) Load value

v_accvgpr_read v30, a32 // (op 11094) 
v_accvgpr_read v31, a33 // (op 11094) 
v_accvgpr_read v32, a34 // (op 11094) 
v_accvgpr_read v33, a35 // (op 11094) 
v_mul_f32 v34, v19, v30 // (op 11094) 
v_mul_f32 v35, v19, v31 // (op 11094) 
v_mul_f32 v36, v19, v32 // (op 11094) 
v_mul_f32 v37, v19, v33 // (op 11094) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11105) 
v_mul_f32 v31, v0, v27 // (op 11105) 
v_mul_f32 v32, v0, v28 // (op 11105) 
v_mul_f32 v33, v0, v29 // (op 11105) 

v_add_f32 v26, v30, v34 // (op 11095) 
v_add_f32 v27, v31, v35 // (op 11095) 
v_add_f32 v28, v32, v36 // (op 11095) 
v_add_f32 v29, v33, v37 // (op 11095) 




v_add_u32 v20, 128, v25 // (op 11097) 

buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11097) Store value




s_lshl_b32 s52, s4, 6 // (op 11110) 
s_add_u32 s52, s52, 128 // (op 11110) 
v_add_u32 v20, v22, s52 // (op 11110) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11110) Load value

v_accvgpr_read v30, a36 // (op 11120) 
v_accvgpr_read v31, a37 // (op 11120) 
v_accvgpr_read v32, a38 // (op 11120) 
v_accvgpr_read v33, a39 // (op 11120) 
v_mul_f32 v34, v19, v30 // (op 11120) 
v_mul_f32 v35, v19, v31 // (op 11120) 
v_mul_f32 v36, v19, v32 // (op 11120) 
v_mul_f32 v37, v19, v33 // (op 11120) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11108) 
v_mul_f32 v31, v0, v27 // (op 11108) 
v_mul_f32 v32, v0, v28 // (op 11108) 
v_mul_f32 v33, v0, v29 // (op 11108) 

v_add_f32 v26, v30, v34 // (op 11109) 
v_add_f32 v27, v31, v35 // (op 11109) 
v_add_f32 v28, v32, v36 // (op 11109) 
v_add_f32 v29, v33, v37 // (op 11109) 




s_lshl_b32 s52, s24, 6 // (op 11099) 
s_add_u32 s52, s52, 128 // (op 11099) 
v_add_u32 v20, v25, s52 // (op 11099) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11099) Store value




s_lshl_b32 s52, s4, 7 // (op 11114) 
s_add_u32 s52, s52, 128 // (op 11114) 
v_add_u32 v20, v22, s52 // (op 11114) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11114) Load value

v_accvgpr_read v30, a40 // (op 11122) 
v_accvgpr_read v31, a41 // (op 11122) 
v_accvgpr_read v32, a42 // (op 11122) 
v_accvgpr_read v33, a43 // (op 11122) 
v_mul_f32 v34, v19, v30 // (op 11122) 
v_mul_f32 v35, v19, v31 // (op 11122) 
v_mul_f32 v36, v19, v32 // (op 11122) 
v_mul_f32 v37, v19, v33 // (op 11122) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11112) 
v_mul_f32 v31, v0, v27 // (op 11112) 
v_mul_f32 v32, v0, v28 // (op 11112) 
v_mul_f32 v33, v0, v29 // (op 11112) 

v_add_f32 v26, v30, v34 // (op 11113) 
v_add_f32 v27, v31, v35 // (op 11113) 
v_add_f32 v28, v32, v36 // (op 11113) 
v_add_f32 v29, v33, v37 // (op 11113) 




s_lshl_b32 s52, s24, 7 // (op 11101) 
s_add_u32 s52, s52, 128 // (op 11101) 
v_add_u32 v20, v25, s52 // (op 11101) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11101) Store value




s_lshl_b32 s52, s4, 6 // (op 11118) 
s_mul_i32 s53, 3, s52 // (op 11118) 
s_add_u32 s53, s53, 128 // (op 11118) 

v_add_u32 v20, v22, s53 // (op 11118) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11118) Load value

v_accvgpr_read v30, a44 // (op 11124) 
v_accvgpr_read v31, a45 // (op 11124) 
v_accvgpr_read v32, a46 // (op 11124) 
v_accvgpr_read v33, a47 // (op 11124) 
v_mul_f32 v34, v19, v30 // (op 11124) 
v_mul_f32 v35, v19, v31 // (op 11124) 
v_mul_f32 v36, v19, v32 // (op 11124) 
v_mul_f32 v37, v19, v33 // (op 11124) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11116) 
v_mul_f32 v31, v0, v27 // (op 11116) 
v_mul_f32 v32, v0, v28 // (op 11116) 
v_mul_f32 v33, v0, v29 // (op 11116) 

v_add_f32 v26, v30, v34 // (op 11117) 
v_add_f32 v27, v31, v35 // (op 11117) 
v_add_f32 v28, v32, v36 // (op 11117) 
v_add_f32 v29, v33, v37 // (op 11117) 




s_lshl_b32 s52, s24, 6 // (op 11103) 
s_mul_i32 s53, 3, s52 // (op 11103) 
s_add_u32 s53, s53, 128 // (op 11103) 

v_add_u32 v20, v25, s53 // (op 11103) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11103) Store value




v_add_u32 v20, 64, v22 // (op 11027) 

s_nop 0
buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11027) Load value

v_accvgpr_read v30, a16 // (op 11015) 
v_accvgpr_read v31, a17 // (op 11015) 
v_accvgpr_read v32, a18 // (op 11015) 
v_accvgpr_read v33, a19 // (op 11015) 
v_mul_f32 v34, v19, v30 // (op 11015) 
v_mul_f32 v35, v19, v31 // (op 11015) 
v_mul_f32 v36, v19, v32 // (op 11015) 
v_mul_f32 v37, v19, v33 // (op 11015) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11026) 
v_mul_f32 v31, v0, v27 // (op 11026) 
v_mul_f32 v32, v0, v28 // (op 11026) 
v_mul_f32 v33, v0, v29 // (op 11026) 

v_add_f32 v26, v30, v34 // (op 11016) 
v_add_f32 v27, v31, v35 // (op 11016) 
v_add_f32 v28, v32, v36 // (op 11016) 
v_add_f32 v29, v33, v37 // (op 11016) 




v_add_u32 v20, 64, v25 // (op 11018) 

buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11018) Store value




s_lshl_b32 s52, s4, 6 // (op 11031) 
s_add_u32 s52, s52, 64 // (op 11031) 
v_add_u32 v20, v22, s52 // (op 11031) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11031) Load value

v_accvgpr_read v30, a20 // (op 11041) 
v_accvgpr_read v31, a21 // (op 11041) 
v_accvgpr_read v32, a22 // (op 11041) 
v_accvgpr_read v33, a23 // (op 11041) 
v_mul_f32 v34, v19, v30 // (op 11041) 
v_mul_f32 v35, v19, v31 // (op 11041) 
v_mul_f32 v36, v19, v32 // (op 11041) 
v_mul_f32 v37, v19, v33 // (op 11041) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11029) 
v_mul_f32 v31, v0, v27 // (op 11029) 
v_mul_f32 v32, v0, v28 // (op 11029) 
v_mul_f32 v33, v0, v29 // (op 11029) 

v_add_f32 v26, v30, v34 // (op 11030) 
v_add_f32 v27, v31, v35 // (op 11030) 
v_add_f32 v28, v32, v36 // (op 11030) 
v_add_f32 v29, v33, v37 // (op 11030) 




s_lshl_b32 s52, s24, 6 // (op 11020) 
s_add_u32 s52, s52, 64 // (op 11020) 
v_add_u32 v20, v25, s52 // (op 11020) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11020) Store value




s_lshl_b32 s52, s4, 7 // (op 11035) 
s_add_u32 s52, s52, 64 // (op 11035) 
v_add_u32 v20, v22, s52 // (op 11035) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11035) Load value

v_accvgpr_read v30, a24 // (op 11043) 
v_accvgpr_read v31, a25 // (op 11043) 
v_accvgpr_read v32, a26 // (op 11043) 
v_accvgpr_read v33, a27 // (op 11043) 
v_mul_f32 v34, v19, v30 // (op 11043) 
v_mul_f32 v35, v19, v31 // (op 11043) 
v_mul_f32 v36, v19, v32 // (op 11043) 
v_mul_f32 v37, v19, v33 // (op 11043) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11033) 
v_mul_f32 v31, v0, v27 // (op 11033) 
v_mul_f32 v32, v0, v28 // (op 11033) 
v_mul_f32 v33, v0, v29 // (op 11033) 

v_add_f32 v26, v30, v34 // (op 11034) 
v_add_f32 v27, v31, v35 // (op 11034) 
v_add_f32 v28, v32, v36 // (op 11034) 
v_add_f32 v29, v33, v37 // (op 11034) 




s_lshl_b32 s52, s24, 7 // (op 11022) 
s_add_u32 s52, s52, 64 // (op 11022) 
v_add_u32 v20, v25, s52 // (op 11022) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11022) Store value




s_lshl_b32 s52, s4, 6 // (op 11039) 
s_mul_i32 s53, 3, s52 // (op 11039) 
s_add_u32 s53, s53, 64 // (op 11039) 

v_add_u32 v20, v22, s53 // (op 11039) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11039) Load value

v_accvgpr_read v30, a28 // (op 11045) 
v_accvgpr_read v31, a29 // (op 11045) 
v_accvgpr_read v32, a30 // (op 11045) 
v_accvgpr_read v33, a31 // (op 11045) 
v_mul_f32 v34, v19, v30 // (op 11045) 
v_mul_f32 v35, v19, v31 // (op 11045) 
v_mul_f32 v36, v19, v32 // (op 11045) 
v_mul_f32 v37, v19, v33 // (op 11045) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11037) 
v_mul_f32 v31, v0, v27 // (op 11037) 
v_mul_f32 v32, v0, v28 // (op 11037) 
v_mul_f32 v33, v0, v29 // (op 11037) 

v_add_f32 v26, v30, v34 // (op 11038) 
v_add_f32 v27, v31, v35 // (op 11038) 
v_add_f32 v28, v32, v36 // (op 11038) 
v_add_f32 v29, v33, v37 // (op 11038) 




s_lshl_b32 s52, s24, 6 // (op 11024) 
s_mul_i32 s53, 3, s52 // (op 11024) 
s_add_u32 s53, s53, 64 // (op 11024) 

v_add_u32 v20, v25, s53 // (op 11024) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11024) Store value




v_add_u32 v20, 192, v22 // (op 11185) 

s_nop 0
buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11185) Load value

v_accvgpr_read v30, a48 // (op 11173) 
v_accvgpr_read v31, a49 // (op 11173) 
v_accvgpr_read v32, a50 // (op 11173) 
v_accvgpr_read v33, a51 // (op 11173) 
v_mul_f32 v34, v19, v30 // (op 11173) 
v_mul_f32 v35, v19, v31 // (op 11173) 
v_mul_f32 v36, v19, v32 // (op 11173) 
v_mul_f32 v37, v19, v33 // (op 11173) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11184) 
v_mul_f32 v31, v0, v27 // (op 11184) 
v_mul_f32 v32, v0, v28 // (op 11184) 
v_mul_f32 v33, v0, v29 // (op 11184) 

v_add_f32 v26, v30, v34 // (op 11174) 
v_add_f32 v27, v31, v35 // (op 11174) 
v_add_f32 v28, v32, v36 // (op 11174) 
v_add_f32 v29, v33, v37 // (op 11174) 




v_add_u32 v20, 192, v25 // (op 11176) 

buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11176) Store value




s_lshl_b32 s52, s4, 6 // (op 11189) 
s_add_u32 s52, s52, 192 // (op 11189) 
v_add_u32 v20, v22, s52 // (op 11189) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11189) Load value

v_accvgpr_read v30, a52 // (op 11199) 
v_accvgpr_read v31, a53 // (op 11199) 
v_accvgpr_read v32, a54 // (op 11199) 
v_accvgpr_read v33, a55 // (op 11199) 
v_mul_f32 v34, v19, v30 // (op 11199) 
v_mul_f32 v35, v19, v31 // (op 11199) 
v_mul_f32 v36, v19, v32 // (op 11199) 
v_mul_f32 v37, v19, v33 // (op 11199) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11187) 
v_mul_f32 v31, v0, v27 // (op 11187) 
v_mul_f32 v32, v0, v28 // (op 11187) 
v_mul_f32 v33, v0, v29 // (op 11187) 

v_add_f32 v26, v30, v34 // (op 11188) 
v_add_f32 v27, v31, v35 // (op 11188) 
v_add_f32 v28, v32, v36 // (op 11188) 
v_add_f32 v29, v33, v37 // (op 11188) 




s_lshl_b32 s52, s24, 6 // (op 11178) 
s_add_u32 s52, s52, 192 // (op 11178) 
v_add_u32 v20, v25, s52 // (op 11178) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11178) Store value




s_lshl_b32 s52, s4, 7 // (op 11193) 
s_add_u32 s52, s52, 192 // (op 11193) 
v_add_u32 v20, v22, s52 // (op 11193) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11193) Load value

v_accvgpr_read v30, a56 // (op 11201) 
v_accvgpr_read v31, a57 // (op 11201) 
v_accvgpr_read v32, a58 // (op 11201) 
v_accvgpr_read v33, a59 // (op 11201) 
v_mul_f32 v34, v19, v30 // (op 11201) 
v_mul_f32 v35, v19, v31 // (op 11201) 
v_mul_f32 v36, v19, v32 // (op 11201) 
v_mul_f32 v37, v19, v33 // (op 11201) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11191) 
v_mul_f32 v31, v0, v27 // (op 11191) 
v_mul_f32 v32, v0, v28 // (op 11191) 
v_mul_f32 v33, v0, v29 // (op 11191) 

v_add_f32 v26, v30, v34 // (op 11192) 
v_add_f32 v27, v31, v35 // (op 11192) 
v_add_f32 v28, v32, v36 // (op 11192) 
v_add_f32 v29, v33, v37 // (op 11192) 




s_lshl_b32 s52, s24, 7 // (op 11180) 
s_add_u32 s52, s52, 192 // (op 11180) 
v_add_u32 v20, v25, s52 // (op 11180) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11180) Store value




s_lshl_b32 s52, s4, 6 // (op 11197) 
s_mul_i32 s53, 3, s52 // (op 11197) 
s_add_u32 s53, s53, 192 // (op 11197) 

v_add_u32 v20, v22, s53 // (op 11197) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 11197) Load value

v_accvgpr_read v30, a60 // (op 11203) 
v_accvgpr_read v31, a61 // (op 11203) 
v_accvgpr_read v32, a62 // (op 11203) 
v_accvgpr_read v33, a63 // (op 11203) 
v_mul_f32 v34, v19, v30 // (op 11203) 
v_mul_f32 v35, v19, v31 // (op 11203) 
v_mul_f32 v36, v19, v32 // (op 11203) 
v_mul_f32 v37, v19, v33 // (op 11203) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 11195) 
v_mul_f32 v31, v0, v27 // (op 11195) 
v_mul_f32 v32, v0, v28 // (op 11195) 
v_mul_f32 v33, v0, v29 // (op 11195) 

v_add_f32 v26, v30, v34 // (op 11196) 
v_add_f32 v27, v31, v35 // (op 11196) 
v_add_f32 v28, v32, v36 // (op 11196) 
v_add_f32 v29, v33, v37 // (op 11196) 




s_lshl_b32 s52, s24, 6 // (op 11182) 
s_mul_i32 s53, 3, s52 // (op 11182) 
s_add_u32 s53, s53, 192 // (op 11182) 

v_add_u32 v20, v25, s53 // (op 11182) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 11182) Store value




v_mov_b32 v20, v22 // (op 126) reg expression

s_nop 0
buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 126) Load value

v_accvgpr_read v30, a0 // (op 123) 
v_accvgpr_read v31, a1 // (op 123) 
v_accvgpr_read v32, a2 // (op 123) 
v_accvgpr_read v33, a3 // (op 123) 
v_mul_f32 v34, v19, v30 // (op 123) 
v_mul_f32 v35, v19, v31 // (op 123) 
v_mul_f32 v36, v19, v32 // (op 123) 
v_mul_f32 v37, v19, v33 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 127) 
v_mul_f32 v31, v0, v27 // (op 127) 
v_mul_f32 v32, v0, v28 // (op 127) 
v_mul_f32 v33, v0, v29 // (op 127) 

v_add_f32 v26, v30, v34 // (op 124) 
v_add_f32 v27, v31, v35 // (op 124) 
v_add_f32 v28, v32, v36 // (op 124) 
v_add_f32 v29, v33, v37 // (op 124) 




v_mov_b32 v20, v25 // (op 125) reg expression

buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 125) Store value




s_lshl_b32 s52, s4, 6 // (op 10941) 
v_add_u32 v20, v22, s52 // (op 10941) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 10941) Load value

v_accvgpr_read v30, a4 // (op 10938) 
v_accvgpr_read v31, a5 // (op 10938) 
v_accvgpr_read v32, a6 // (op 10938) 
v_accvgpr_read v33, a7 // (op 10938) 
v_mul_f32 v34, v19, v30 // (op 10938) 
v_mul_f32 v35, v19, v31 // (op 10938) 
v_mul_f32 v36, v19, v32 // (op 10938) 
v_mul_f32 v37, v19, v33 // (op 10938) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 10942) 
v_mul_f32 v31, v0, v27 // (op 10942) 
v_mul_f32 v32, v0, v28 // (op 10942) 
v_mul_f32 v33, v0, v29 // (op 10942) 

v_add_f32 v26, v30, v34 // (op 10939) 
v_add_f32 v27, v31, v35 // (op 10939) 
v_add_f32 v28, v32, v36 // (op 10939) 
v_add_f32 v29, v33, v37 // (op 10939) 




s_lshl_b32 s52, s24, 6 // (op 10940) 
v_add_u32 v20, v25, s52 // (op 10940) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 10940) Store value




s_lshl_b32 s52, s4, 7 // (op 10953) 
v_add_u32 v20, v22, s52 // (op 10953) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 10953) Load value

v_accvgpr_read v30, a8 // (op 10950) 
v_accvgpr_read v31, a9 // (op 10950) 
v_accvgpr_read v32, a10 // (op 10950) 
v_accvgpr_read v33, a11 // (op 10950) 
v_mul_f32 v34, v19, v30 // (op 10950) 
v_mul_f32 v35, v19, v31 // (op 10950) 
v_mul_f32 v36, v19, v32 // (op 10950) 
v_mul_f32 v37, v19, v33 // (op 10950) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 10954) 
v_mul_f32 v31, v0, v27 // (op 10954) 
v_mul_f32 v32, v0, v28 // (op 10954) 
v_mul_f32 v33, v0, v29 // (op 10954) 

v_add_f32 v26, v30, v34 // (op 10951) 
v_add_f32 v27, v31, v35 // (op 10951) 
v_add_f32 v28, v32, v36 // (op 10951) 
v_add_f32 v29, v33, v37 // (op 10951) 




s_lshl_b32 s52, s24, 7 // (op 10952) 
v_add_u32 v20, v25, s52 // (op 10952) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 10952) Store value




s_lshl_b32 s52, s4, 6 // (op 10965) 
s_mul_i32 s53, 3, s52 // (op 10965) 

v_add_u32 v20, v22, s53 // (op 10965) 


buffer_load_dwordx4 v[26:29], v20, s[56:59], 0 offen // (op 10965) Load value

v_accvgpr_read v30, a12 // (op 10962) 
v_accvgpr_read v31, a13 // (op 10962) 
v_accvgpr_read v32, a14 // (op 10962) 
v_accvgpr_read v33, a15 // (op 10962) 
v_mul_f32 v34, v19, v30 // (op 10962) 
v_mul_f32 v35, v19, v31 // (op 10962) 
v_mul_f32 v36, v19, v32 // (op 10962) 
v_mul_f32 v37, v19, v33 // (op 10962) 

s_waitcnt vmcnt(0)
v_mul_f32 v30, v0, v26 // (op 10966) 
v_mul_f32 v31, v0, v27 // (op 10966) 
v_mul_f32 v32, v0, v28 // (op 10966) 
v_mul_f32 v33, v0, v29 // (op 10966) 

v_add_f32 v26, v30, v34 // (op 10963) 
v_add_f32 v27, v31, v35 // (op 10963) 
v_add_f32 v28, v32, v36 // (op 10963) 
v_add_f32 v29, v33, v37 // (op 10963) 




s_lshl_b32 s52, s24, 6 // (op 10964) 
s_mul_i32 s53, 3, s52 // (op 10964) 

v_add_u32 v20, v25, s53 // (op 10964) 


buffer_store_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 10964) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_114:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_342 // (op 342) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_342
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_Postmaccumulation_Condition_342:
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_Postmaccumulation_Condition_342:


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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_255 // (op 255) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_SKStreamTileLoop_255 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_SKStreamTileLoop_255:

















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
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_410 // (op 410) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_410 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_410:
v_lshrrev_b32 v3, 7, v1 // (op 21395) 
s_mul_i32 s15, s16, s50 // (op 21395) 
s_add_u32 s15, s15, s0 // (op 21395) 
v_and_b32 v4, 63, v1 // (op 21395) 
s_add_u32 s6, s15, s9 // (op 21395) 

v_lshrrev_b32 v13, 4, v4 // (op 21395) 
s_mul_hi_u32 s15, s6, s13 // (op 21395) 
v_lshlrev_b32 v14, 2, v13 // (op 21395) 

s_sub_u32 s7, s6, s15 // (op 21395) 
v_lshl_add_u32 v13, v3, 6, v14 // (op 21395) 


s_lshr_b32 s8, s7, 1 // (op 21395) 

v_lshrrev_b32 v3, 2, v4 // (op 21395) 

s_add_u32 s7, s8, s15 // (op 21395) 


v_and_b32 v4, 3, v3 // (op 21395) 

s_lshr_b32 s8, s7, s11 // (op 21395) 

v_and_b32 v3, 3, v1 // (op 21395) 
s_and_b32 s7, s11, 2147483648 // (op 21395) 
v_lshl_add_u32 v14, v4, 2, v3 // (op 21395) 


s_cmp_eq_i32 s7, 0 // (op 21395) 
s_mov_b32 s15, scc // (op 21395) 

v_lshrrev_b32 v3, 6, v1 // (op 21395) 
s_cmp_eq_i32 s15, 0 // (op 21395) 
s_cselect_b32 s7, s6, s8 // (op 21395) 



v_and_b32 v4, 1, v3 // (op 21395) 

s_mul_hi_u32 s6, s7, s12 // (op 21395) 
v_lshl_add_u32 v3, v4, 6, v14 // (op 21395) 


s_sub_u32 s8, s7, s6 // (op 21395) 
s_lshr_b32 s15, s8, 1 // (op 21395) 

s_add_u32 s8, s15, s6 // (op 21395) 


s_lshr_b32 s6, s8, s10 // (op 21395) 

s_and_b32 s8, s10, 2147483648 // (op 21395) 
s_cmp_eq_i32 s8, 0 // (op 21395) 
s_mov_b32 s15, scc // (op 21395) 

s_cmp_eq_i32 s15, 0 // (op 21395) 
s_cselect_b32 s8, s7, s6 // (op 21395) 


s_mul_i32 s6, s8, s14 // (op 21395) 
v_lshl_add_u32 v4, s8, 7, v13 // (op 21395) 


s_sub_u32 s8, s7, s6 // (op 21395) 


v_lshl_add_u32 v13, s8, 7, v3 // (op 21395) 


v_mul_lo_u32 v3, v13, s4 // (op 21395) 

v_add_lshl_u32 v13, v4, v3, 2 // (op 21395) 


s_mul_i32 s19, s4, s23 // (op 21397) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 21397) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 21397) least significant
s_mul_hi_u32 s6, s4, s22 // (op 21397) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21397) most significant: sum
s_add_u32 s19, s19, s6 // (op 21397) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21397) 

s_mov_b32 s56, s20 // (op 21397) reg expression
s_mov_b32 s57, s21 // (op 21397) reg expression
s_mov_b32 s58, s28 // (op 21397) reg expression
s_mov_b32 s59, 131072 // (op 21397) call()

v_lshrrev_b32 v3, 7, v1 // (op 21597) 
s_mul_i32 s15, s16, s50 // (op 21597) 
s_add_u32 s15, s15, s0 // (op 21597) 
v_and_b32 v4, 63, v1 // (op 21597) 
s_add_u32 s6, s15, s9 // (op 21597) 

v_lshrrev_b32 v14, 4, v4 // (op 21597) 
s_mul_hi_u32 s15, s6, s13 // (op 21597) 
v_lshlrev_b32 v15, 2, v14 // (op 21597) 

s_sub_u32 s7, s6, s15 // (op 21597) 
v_lshl_add_u32 v14, v3, 6, v15 // (op 21597) 


s_lshr_b32 s8, s7, 1 // (op 21597) 

v_lshrrev_b32 v3, 2, v4 // (op 21597) 

s_add_u32 s7, s8, s15 // (op 21597) 


v_and_b32 v4, 3, v3 // (op 21597) 

s_lshr_b32 s8, s7, s11 // (op 21597) 

v_and_b32 v3, 3, v1 // (op 21597) 
s_and_b32 s7, s11, 2147483648 // (op 21597) 
v_lshl_add_u32 v15, v4, 2, v3 // (op 21597) 


s_cmp_eq_i32 s7, 0 // (op 21597) 
s_mov_b32 s15, scc // (op 21597) 

v_lshrrev_b32 v3, 6, v1 // (op 21597) 
s_cmp_eq_i32 s15, 0 // (op 21597) 
s_cselect_b32 s7, s6, s8 // (op 21597) 



v_and_b32 v4, 1, v3 // (op 21597) 

s_mul_hi_u32 s6, s7, s12 // (op 21597) 
v_lshl_add_u32 v3, v4, 6, v15 // (op 21597) 


s_sub_u32 s8, s7, s6 // (op 21597) 
s_lshr_b32 s15, s8, 1 // (op 21597) 

s_add_u32 s8, s15, s6 // (op 21597) 


s_lshr_b32 s6, s8, s10 // (op 21597) 

s_and_b32 s8, s10, 2147483648 // (op 21597) 
s_cmp_eq_i32 s8, 0 // (op 21597) 
s_mov_b32 s15, scc // (op 21597) 

s_cmp_eq_i32 s15, 0 // (op 21597) 
s_cselect_b32 s8, s7, s6 // (op 21597) 


s_mul_i32 s6, s8, s14 // (op 21597) 
v_lshl_add_u32 v4, s8, 7, v14 // (op 21597) 


s_sub_u32 s8, s7, s6 // (op 21597) 


v_lshl_add_u32 v14, s8, 7, v3 // (op 21597) 


v_mul_lo_u32 v3, v14, s24 // (op 21597) 

v_add_lshl_u32 v14, v4, v3, 2 // (op 21597) 


s_mul_i32 s19, s24, s23 // (op 21599) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 21599) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 21599) least significant
s_mul_hi_u32 s6, s24, s22 // (op 21599) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21599) most significant: sum
s_add_u32 s19, s19, s6 // (op 21599) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21599) 

s_mov_b32 s60, s26 // (op 21599) reg expression
s_mov_b32 s61, s27 // (op 21599) reg expression
s_mov_b32 s62, s28 // (op 21599) reg expression
s_mov_b32 s63, 131072 // (op 21599) call()

v_lshrrev_b32 v3, 7, v1 // (op 21676) 
v_and_b32 v4, 15, v1 // (op 21676) 
v_lshl_add_u32 v15, v3, 6, v4 // (op 21676) 


v_and_b32 v3, 63, v1 // (op 21676) 
v_lshrrev_b32 v4, 4, v3 // (op 21676) 

v_lshlrev_b32 v3, 5, v4 // (op 21676) 

v_lshl_add_u32 v4, v15, 7, v3 // (op 21676) 


v_lshlrev_b32 v3, 2, v4 // (op 21676) 

v_lshrrev_b32 v4, 3, v3 // (op 21676) 

v_lshrrev_b32 v3, 6, v1 // (op 21751) 
v_and_b32 v15, 1, v3 // (op 21751) 

v_and_b32 v3, 15, v1 // (op 21751) 
v_lshl_add_u32 v16, v15, 6, v3 // (op 21751) 


v_and_b32 v3, 63, v1 // (op 21751) 
v_lshrrev_b32 v15, 4, v3 // (op 21751) 

v_lshlrev_b32 v3, 5, v15 // (op 21751) 

v_lshl_add_u32 v15, v16, 7, v3 // (op 21751) 


v_lshlrev_b32 v3, 2, v15 // (op 21751) 

v_lshrrev_b32 v15, 3, v3 // (op 21751) 

v_accvgpr_write a0, 0.00000 // (op 161) call()
v_accvgpr_write a1, 0.00000 // (op 161) call()
v_accvgpr_write a2, 0.00000 // (op 161) call()
v_accvgpr_write a3, 0.00000 // (op 161) call()
v_accvgpr_write a4, 0.00000 // (op 649) call()
v_accvgpr_write a5, 0.00000 // (op 649) call()
v_accvgpr_write a6, 0.00000 // (op 649) call()
v_accvgpr_write a7, 0.00000 // (op 649) call()
v_accvgpr_write a8, 0.00000 // (op 875) call()
v_accvgpr_write a9, 0.00000 // (op 875) call()
v_accvgpr_write a10, 0.00000 // (op 875) call()
v_accvgpr_write a11, 0.00000 // (op 875) call()
v_accvgpr_write a12, 0.00000 // (op 1101) call()
v_accvgpr_write a13, 0.00000 // (op 1101) call()
v_accvgpr_write a14, 0.00000 // (op 1101) call()
v_accvgpr_write a15, 0.00000 // (op 1101) call()
v_accvgpr_write a16, 0.00000 // (op 1463) call()
v_accvgpr_write a17, 0.00000 // (op 1463) call()
v_accvgpr_write a18, 0.00000 // (op 1463) call()
v_accvgpr_write a19, 0.00000 // (op 1463) call()
v_accvgpr_write a20, 0.00000 // (op 1876) call()
v_accvgpr_write a21, 0.00000 // (op 1876) call()
v_accvgpr_write a22, 0.00000 // (op 1876) call()
v_accvgpr_write a23, 0.00000 // (op 1876) call()
v_accvgpr_write a24, 0.00000 // (op 1877) call()
v_accvgpr_write a25, 0.00000 // (op 1877) call()
v_accvgpr_write a26, 0.00000 // (op 1877) call()
v_accvgpr_write a27, 0.00000 // (op 1877) call()
v_accvgpr_write a28, 0.00000 // (op 1878) call()
v_accvgpr_write a29, 0.00000 // (op 1878) call()
v_accvgpr_write a30, 0.00000 // (op 1878) call()
v_accvgpr_write a31, 0.00000 // (op 1878) call()
v_accvgpr_write a32, 0.00000 // (op 2465) call()
v_accvgpr_write a33, 0.00000 // (op 2465) call()
v_accvgpr_write a34, 0.00000 // (op 2465) call()
v_accvgpr_write a35, 0.00000 // (op 2465) call()
v_accvgpr_write a36, 0.00000 // (op 2878) call()
v_accvgpr_write a37, 0.00000 // (op 2878) call()
v_accvgpr_write a38, 0.00000 // (op 2878) call()
v_accvgpr_write a39, 0.00000 // (op 2878) call()
v_accvgpr_write a40, 0.00000 // (op 2879) call()
v_accvgpr_write a41, 0.00000 // (op 2879) call()
v_accvgpr_write a42, 0.00000 // (op 2879) call()
v_accvgpr_write a43, 0.00000 // (op 2879) call()
v_accvgpr_write a44, 0.00000 // (op 2880) call()
v_accvgpr_write a45, 0.00000 // (op 2880) call()
v_accvgpr_write a46, 0.00000 // (op 2880) call()
v_accvgpr_write a47, 0.00000 // (op 2880) call()
v_accvgpr_write a48, 0.00000 // (op 3467) call()
v_accvgpr_write a49, 0.00000 // (op 3467) call()
v_accvgpr_write a50, 0.00000 // (op 3467) call()
v_accvgpr_write a51, 0.00000 // (op 3467) call()
v_accvgpr_write a52, 0.00000 // (op 3880) call()
v_accvgpr_write a53, 0.00000 // (op 3880) call()
v_accvgpr_write a54, 0.00000 // (op 3880) call()
v_accvgpr_write a55, 0.00000 // (op 3880) call()
v_accvgpr_write a56, 0.00000 // (op 3881) call()
v_accvgpr_write a57, 0.00000 // (op 3881) call()
v_accvgpr_write a58, 0.00000 // (op 3881) call()
v_accvgpr_write a59, 0.00000 // (op 3881) call()
v_accvgpr_write a60, 0.00000 // (op 3882) call()
v_accvgpr_write a61, 0.00000 // (op 3882) call()
v_accvgpr_write a62, 0.00000 // (op 3882) call()
v_accvgpr_write a63, 0.00000 // (op 3882) call()
v_lshrrev_b32 v3, 2, v1 // (op 21131) 
s_mul_i32 s15, s16, s50 // (op 21131) 
s_add_u32 s15, s15, s0 // (op 21131) 
v_and_b32 v16, -4, v1 // (op 21131) 
s_add_u32 s6, s15, s9 // (op 21131) 

v_sub_u32 v17, v1, v16 // (op 21131) 

s_mul_hi_u32 s15, s6, s13 // (op 21131) 
v_lshlrev_b32 v16, 5, v17 // (op 21131) 

s_sub_u32 s7, s6, s15 // (op 21131) 
s_lshr_b32 s8, s7, 1 // (op 21131) 

s_add_u32 s7, s8, s15 // (op 21131) 


s_lshr_b32 s8, s7, s11 // (op 21131) 

s_and_b32 s7, s11, 2147483648 // (op 21131) 
s_cmp_eq_i32 s7, 0 // (op 21131) 
s_mov_b32 s15, scc // (op 21131) 

s_cmp_eq_i32 s15, 0 // (op 21131) 
s_cselect_b32 s7, s6, s8 // (op 21131) 


s_mul_hi_u32 s8, s7, s12 // (op 21131) 
s_sub_u32 s15, s7, s8 // (op 21131) 
s_lshr_b32 s18, s15, 1 // (op 21131) 

s_add_u32 s15, s18, s8 // (op 21131) 


s_lshr_b32 s8, s15, s10 // (op 21131) 

s_and_b32 s15, s10, 2147483648 // (op 21131) 
s_cmp_eq_i32 s15, 0 // (op 21131) 
s_mov_b32 s18, scc // (op 21131) 

s_cmp_eq_i32 s18, 0 // (op 21131) 
s_cselect_b32 s15, s7, s8 // (op 21131) 


v_lshl_add_u32 v17, s15, 7, v3 // (op 21131) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 21131) 
s_mul_i32 s8, s7, s18 // (op 21131) 


s_sub_u32 s7, s6, s8 // (op 21131) 


v_lshl_add_u32 v3, s7, 7, v16 // (op 21131) 


v_mov_b32 v21, 0 // (op 21131) convert
v_mov_b32 v20, v3 // (op 21131) convert

v_mul_lo_u32 v23, v17, s31 // (op 21131) most significant: low of low * high
v_mul_hi_u32 v3, v17, s30 // (op 21131) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 21131) least significant: low of low * low
v_add_u32 v23, v23, v3 // (op 21131) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v20 // (op 21131) least significant half
v_addc_co_u32 v23, s[18:19], v23, v21, s[18:19] // (op 21131) most significant half



v_lshlrev_b64 v[20:21], 2, v[22:23] // (op 21131) 

v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 21131) 

s_mul_i32 s19, s30, s35 // (op 21133) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 21133) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 21133) least significant
s_mul_hi_u32 s6, s30, s34 // (op 21133) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21133) most significant: sum
s_add_u32 s19, s19, s6 // (op 21133) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21133) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21133) 

s_mov_b32 s68, s32 // (op 21133) reg expression
s_mov_b32 s69, s33 // (op 21133) reg expression
s_mov_b32 s70, s18 // (op 21133) reg expression
s_mov_b32 s71, 131072 // (op 21133) call()

v_lshrrev_b32 v3, 2, v1 // (op 21150) 
s_mul_i32 s15, s16, s50 // (op 21150) 
s_add_u32 s15, s15, s0 // (op 21150) 
v_and_b32 v18, -4, v1 // (op 21150) 
s_add_u32 s6, s15, s9 // (op 21150) 

v_sub_u32 v20, v1, v18 // (op 21150) 

s_mul_hi_u32 s15, s6, s13 // (op 21150) 
v_lshlrev_b32 v18, 5, v20 // (op 21150) 

s_sub_u32 s7, s6, s15 // (op 21150) 
s_lshr_b32 s8, s7, 1 // (op 21150) 

s_add_u32 s7, s8, s15 // (op 21150) 


s_lshr_b32 s8, s7, s11 // (op 21150) 

s_and_b32 s7, s11, 2147483648 // (op 21150) 
s_cmp_eq_i32 s7, 0 // (op 21150) 
s_mov_b32 s15, scc // (op 21150) 

s_cmp_eq_i32 s15, 0 // (op 21150) 
s_cselect_b32 s7, s6, s8 // (op 21150) 


s_mul_hi_u32 s8, s7, s12 // (op 21150) 
s_sub_u32 s15, s7, s8 // (op 21150) 
s_lshr_b32 s18, s15, 1 // (op 21150) 

s_add_u32 s15, s18, s8 // (op 21150) 


s_lshr_b32 s8, s15, s10 // (op 21150) 

s_and_b32 s15, s10, 2147483648 // (op 21150) 
s_cmp_eq_i32 s15, 0 // (op 21150) 
s_mov_b32 s18, scc // (op 21150) 

s_cmp_eq_i32 s18, 0 // (op 21150) 
s_cselect_b32 s15, s7, s8 // (op 21150) 


v_lshl_add_u32 v20, s15, 7, v3 // (op 21150) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 21150) 
s_mul_i32 s8, s7, s18 // (op 21150) 


s_sub_u32 s7, s6, s8 // (op 21150) 


v_lshl_add_u32 v3, s7, 7, v18 // (op 21150) 


v_mov_b32 v23, 0 // (op 21150) convert
v_mov_b32 v22, v3 // (op 21150) convert

v_mul_lo_u32 v25, v20, s31 // (op 21150) most significant: low of low * high
v_mul_hi_u32 v3, v20, s30 // (op 21150) most significant: high of low * low
v_mul_lo_u32 v24, v20, s30 // (op 21150) least significant: low of low * low
v_add_u32 v25, v25, v3 // (op 21150) most significant: sum

v_add_co_u32 v24, s[18:19], v24, v22 // (op 21150) least significant half
v_addc_co_u32 v25, s[18:19], v25, v23, s[18:19] // (op 21150) most significant half



v_lshlrev_b64 v[20:21], 2, v[24:25] // (op 21150) 

v_lshrrev_b64 v[22:23], 3, v[20:21] // (op 21150) 

s_mul_i32 s19, s30, s35 // (op 21152) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 21152) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 21152) least significant
s_mul_hi_u32 s6, s30, s34 // (op 21152) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21152) most significant: sum
s_add_u32 s19, s19, s6 // (op 21152) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21152) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21152) 

s_mov_b32 s68, s32 // (op 21152) reg expression
s_mov_b32 s69, s33 // (op 21152) reg expression
s_mov_b32 s70, s18 // (op 21152) reg expression
s_mov_b32 s71, 131072 // (op 21152) call()

v_lshrrev_b32 v3, 2, v1 // (op 21277) 
s_mul_i32 s15, s16, s50 // (op 21277) 
s_add_u32 s15, s15, s0 // (op 21277) 
v_and_b32 v18, -4, v1 // (op 21277) 
s_add_u32 s6, s15, s9 // (op 21277) 

v_sub_u32 v20, v1, v18 // (op 21277) 

s_mul_hi_u32 s15, s6, s13 // (op 21277) 
v_lshlrev_b32 v18, 5, v20 // (op 21277) 

s_sub_u32 s7, s6, s15 // (op 21277) 
s_lshr_b32 s8, s7, 1 // (op 21277) 

s_add_u32 s7, s8, s15 // (op 21277) 


s_lshr_b32 s8, s7, s11 // (op 21277) 

s_and_b32 s7, s11, 2147483648 // (op 21277) 
s_cmp_eq_i32 s7, 0 // (op 21277) 
s_mov_b32 s15, scc // (op 21277) 

s_cmp_eq_i32 s15, 0 // (op 21277) 
s_cselect_b32 s7, s6, s8 // (op 21277) 


s_mul_hi_u32 s8, s7, s12 // (op 21277) 
s_sub_u32 s15, s7, s8 // (op 21277) 
s_lshr_b32 s18, s15, 1 // (op 21277) 

s_add_u32 s15, s18, s8 // (op 21277) 


s_lshr_b32 s8, s15, s10 // (op 21277) 

s_and_b32 s15, s10, 2147483648 // (op 21277) 
s_cmp_eq_i32 s15, 0 // (op 21277) 
s_mov_b32 s18, scc // (op 21277) 

s_cmp_eq_i32 s18, 0 // (op 21277) 
s_cselect_b32 s15, s7, s8 // (op 21277) 


s_mul_i32 s8, s15, s14 // (op 21277) 

s_sub_u32 s15, s7, s8 // (op 21277) 

v_lshl_add_u32 v20, s15, 7, v3 // (op 21277) 


v_mul_lo_u32 v25, v20, s37 // (op 21277) most significant: low of low * high
v_mul_hi_u32 v3, v20, s36 // (op 21277) most significant: high of low * low
v_mul_lo_u32 v24, v20, s36 // (op 21277) least significant: low of low * low
v_add_u32 v25, v25, v3 // (op 21277) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 21277) 
s_mul_i32 s8, s7, s18 // (op 21277) 


s_sub_u32 s7, s6, s8 // (op 21277) 


v_lshl_add_u32 v3, s7, 7, v18 // (op 21277) 


v_mov_b32 v21, 0 // (op 21277) convert
v_mov_b32 v20, v3 // (op 21277) convert

v_add_co_u32 v26, s[18:19], v20, v24 // (op 21277) least significant half
v_addc_co_u32 v27, s[18:19], v21, v25, s[18:19] // (op 21277) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 21277) 


v_lshrrev_b64 v[20:21], 3, v[26:27] // (op 21277) 

s_mul_i32 s19, s36, s23 // (op 21279) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 21279) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 21279) least significant
s_mul_hi_u32 s6, s36, s22 // (op 21279) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21279) most significant: sum
s_add_u32 s19, s19, s6 // (op 21279) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21279) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21279) 

s_mov_b32 s72, s38 // (op 21279) reg expression
s_mov_b32 s73, s39 // (op 21279) reg expression
s_mov_b32 s74, s18 // (op 21279) reg expression
s_mov_b32 s75, 131072 // (op 21279) call()

v_lshrrev_b32 v3, 2, v1 // (op 21296) 
s_mul_i32 s15, s16, s50 // (op 21296) 
s_add_u32 s15, s15, s0 // (op 21296) 
v_and_b32 v18, -4, v1 // (op 21296) 
s_add_u32 s6, s15, s9 // (op 21296) 

v_sub_u32 v24, v1, v18 // (op 21296) 

s_mul_hi_u32 s15, s6, s13 // (op 21296) 
v_lshlrev_b32 v18, 5, v24 // (op 21296) 

s_sub_u32 s7, s6, s15 // (op 21296) 
s_lshr_b32 s8, s7, 1 // (op 21296) 

s_add_u32 s7, s8, s15 // (op 21296) 


s_lshr_b32 s8, s7, s11 // (op 21296) 

s_and_b32 s7, s11, 2147483648 // (op 21296) 
s_cmp_eq_i32 s7, 0 // (op 21296) 
s_mov_b32 s15, scc // (op 21296) 

s_cmp_eq_i32 s15, 0 // (op 21296) 
s_cselect_b32 s7, s6, s8 // (op 21296) 


s_mul_hi_u32 s8, s7, s12 // (op 21296) 
s_sub_u32 s15, s7, s8 // (op 21296) 
s_lshr_b32 s18, s15, 1 // (op 21296) 

s_add_u32 s15, s18, s8 // (op 21296) 


s_lshr_b32 s8, s15, s10 // (op 21296) 

s_and_b32 s15, s10, 2147483648 // (op 21296) 
s_cmp_eq_i32 s15, 0 // (op 21296) 
s_mov_b32 s18, scc // (op 21296) 

s_cmp_eq_i32 s18, 0 // (op 21296) 
s_cselect_b32 s15, s7, s8 // (op 21296) 


s_mul_i32 s8, s15, s14 // (op 21296) 

s_sub_u32 s15, s7, s8 // (op 21296) 

v_lshl_add_u32 v24, s15, 7, v3 // (op 21296) 


v_mul_lo_u32 v27, v24, s37 // (op 21296) most significant: low of low * high
v_mul_hi_u32 v3, v24, s36 // (op 21296) most significant: high of low * low
v_mul_lo_u32 v26, v24, s36 // (op 21296) least significant: low of low * low
v_add_u32 v27, v27, v3 // (op 21296) most significant: sum


s_ashr_i64 s[18:19], s[2:3], 7 // (op 21296) 
s_mul_i32 s8, s7, s18 // (op 21296) 


s_sub_u32 s7, s6, s8 // (op 21296) 


v_lshl_add_u32 v3, s7, 7, v18 // (op 21296) 


v_mov_b32 v25, 0 // (op 21296) convert
v_mov_b32 v24, v3 // (op 21296) convert

v_add_co_u32 v28, s[18:19], v24, v26 // (op 21296) least significant half
v_addc_co_u32 v29, s[18:19], v25, v27, s[18:19] // (op 21296) most significant half

v_lshlrev_b64 v[28:29], 2, v[28:29] // (op 21296) 


v_lshrrev_b64 v[24:25], 3, v[28:29] // (op 21296) 

s_mul_i32 s19, s36, s23 // (op 21298) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 21298) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 21298) least significant
s_mul_hi_u32 s6, s36, s22 // (op 21298) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21298) most significant: sum
s_add_u32 s19, s19, s6 // (op 21298) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21298) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21298) 

s_mov_b32 s72, s38 // (op 21298) reg expression
s_mov_b32 s73, s39 // (op 21298) reg expression
s_mov_b32 s74, s18 // (op 21298) reg expression
s_mov_b32 s75, 131072 // (op 21298) call()

v_and_b32 v3, 63, v1 // (op 22354) 
s_mul_i32 s15, s16, s50 // (op 22354) 
s_add_u32 s15, s15, s0 // (op 22354) 
v_lshrrev_b32 v18, 4, v3 // (op 22354) 
s_add_u32 s6, s15, s9 // (op 22354) 

v_and_b32 v26, 3, v18 // (op 22354) 

s_mul_hi_u32 s15, s6, s13 // (op 22354) 
v_and_b32 v18, 15, v1 // (op 22354) 
s_sub_u32 s7, s6, s15 // (op 22354) 
v_lshl_add_u32 v27, v26, 4, v18 // (op 22354) 


s_lshr_b32 s8, s7, 1 // (op 22354) 

v_lshrrev_b32 v18, 7, v1 // (op 22354) 
s_add_u32 s7, s8, s15 // (op 22354) 


v_lshl_add_u32 v26, v18, 6, v27 // (op 22354) 


s_lshr_b32 s8, s7, s11 // (op 22354) 

v_lshrrev_b32 v18, 6, v3 // (op 22354) 

s_and_b32 s7, s11, 2147483648 // (op 22354) 
v_lshlrev_b32 v3, 2, v18 // (op 22354) 

s_cmp_eq_i32 s7, 0 // (op 22354) 
s_mov_b32 s15, scc // (op 22354) 

s_cmp_eq_i32 s15, 0 // (op 22354) 
s_cselect_b32 s7, s6, s8 // (op 22354) 


s_mul_hi_u32 s8, s7, s12 // (op 22354) 
s_sub_u32 s15, s7, s8 // (op 22354) 
s_lshr_b32 s18, s15, 1 // (op 22354) 

s_add_u32 s15, s18, s8 // (op 22354) 


s_lshr_b32 s8, s15, s10 // (op 22354) 

s_and_b32 s15, s10, 2147483648 // (op 22354) 
s_cmp_eq_i32 s15, 0 // (op 22354) 
s_mov_b32 s18, scc // (op 22354) 

s_cmp_eq_i32 s18, 0 // (op 22354) 
s_cselect_b32 s15, s7, s8 // (op 22354) 


v_lshl_add_u32 v18, s15, 7, v26 // (op 22354) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 22354) 
s_mul_i32 s8, s7, s18 // (op 22354) 


s_sub_u32 s7, s6, s8 // (op 22354) 


v_lshl_add_u32 v26, s7, 2, v3 // (op 22354) 


v_mul_lo_u32 v3, v18, s40 // (op 22354) 
v_add_u32 v3, v3, v26 // (op 22354) 


s_mul_i32 s19, s40, s35 // (op 22356) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 22356) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 22356) least significant
s_mul_hi_u32 s6, s40, s34 // (op 22356) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 22356) most significant: sum
s_add_u32 s19, s19, s6 // (op 22356) most significant: sum


s_mov_b32 s52, s42 // (op 22356) reg expression
s_mov_b32 s53, s43 // (op 22356) reg expression
s_mov_b32 s54, s18 // (op 22356) reg expression
s_mov_b32 s55, 131072 // (op 22356) call()

v_and_b32 v18, 63, v1 // (op 22375) 
s_mul_i32 s15, s16, s50 // (op 22375) 
s_add_u32 s15, s15, s0 // (op 22375) 
v_lshrrev_b32 v26, 4, v18 // (op 22375) 
s_add_u32 s6, s15, s9 // (op 22375) 

v_and_b32 v27, 3, v26 // (op 22375) 

s_mul_hi_u32 s15, s6, s13 // (op 22375) 
v_and_b32 v26, 15, v1 // (op 22375) 
s_sub_u32 s7, s6, s15 // (op 22375) 
v_lshl_add_u32 v28, v27, 4, v26 // (op 22375) 


s_lshr_b32 s8, s7, 1 // (op 22375) 

v_lshrrev_b32 v26, 7, v1 // (op 22375) 
s_add_u32 s7, s8, s15 // (op 22375) 


v_lshl_add_u32 v27, v26, 6, v28 // (op 22375) 


s_lshr_b32 s8, s7, s11 // (op 22375) 

v_lshrrev_b32 v26, 6, v18 // (op 22375) 

s_and_b32 s7, s11, 2147483648 // (op 22375) 
v_lshlrev_b32 v18, 2, v26 // (op 22375) 

s_cmp_eq_i32 s7, 0 // (op 22375) 
s_mov_b32 s15, scc // (op 22375) 

s_cmp_eq_i32 s15, 0 // (op 22375) 
s_cselect_b32 s7, s6, s8 // (op 22375) 


s_mul_hi_u32 s8, s7, s12 // (op 22375) 
s_sub_u32 s15, s7, s8 // (op 22375) 
s_lshr_b32 s18, s15, 1 // (op 22375) 

s_add_u32 s15, s18, s8 // (op 22375) 


s_lshr_b32 s8, s15, s10 // (op 22375) 

s_and_b32 s15, s10, 2147483648 // (op 22375) 
s_cmp_eq_i32 s15, 0 // (op 22375) 
s_mov_b32 s18, scc // (op 22375) 

s_cmp_eq_i32 s18, 0 // (op 22375) 
s_cselect_b32 s15, s7, s8 // (op 22375) 


v_lshl_add_u32 v26, s15, 7, v27 // (op 22375) 


s_ashr_i64 s[18:19], s[2:3], 7 // (op 22375) 
s_mul_i32 s8, s7, s18 // (op 22375) 


s_sub_u32 s7, s6, s8 // (op 22375) 


v_lshl_add_u32 v27, s7, 2, v18 // (op 22375) 


v_mul_lo_u32 v18, v26, s40 // (op 22375) 
v_add_u32 v18, v18, v27 // (op 22375) 


s_mul_i32 s19, s40, s35 // (op 22377) most significant: low of low * high
s_mul_i32 s15, s41, s34 // (op 22377) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 22377) least significant
s_mul_hi_u32 s6, s40, s34 // (op 22377) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 22377) most significant: sum
s_add_u32 s19, s19, s6 // (op 22377) most significant: sum


s_mov_b32 s52, s42 // (op 22377) reg expression
s_mov_b32 s53, s43 // (op 22377) reg expression
s_mov_b32 s54, s18 // (op 22377) reg expression
s_mov_b32 s55, 131072 // (op 22377) call()

v_and_b32 v26, 63, v1 // (op 22406) 
s_mul_i32 s15, s16, s50 // (op 22406) 
s_add_u32 s15, s15, s0 // (op 22406) 
v_lshrrev_b32 v27, 4, v26 // (op 22406) 
s_add_u32 s6, s15, s9 // (op 22406) 

v_and_b32 v28, 3, v27 // (op 22406) 

s_mul_hi_u32 s15, s6, s13 // (op 22406) 
v_and_b32 v27, 15, v1 // (op 22406) 
s_sub_u32 s7, s6, s15 // (op 22406) 
v_lshl_add_u32 v29, v28, 4, v27 // (op 22406) 


s_lshr_b32 s8, s7, 1 // (op 22406) 

v_lshrrev_b32 v27, 6, v1 // (op 22406) 
s_add_u32 s7, s8, s15 // (op 22406) 


v_and_b32 v28, 1, v27 // (op 22406) 

s_lshr_b32 s8, s7, s11 // (op 22406) 

v_lshl_add_u32 v27, v28, 6, v29 // (op 22406) 


s_and_b32 s7, s11, 2147483648 // (op 22406) 
v_lshrrev_b32 v28, 6, v26 // (op 22406) 

s_cmp_eq_i32 s7, 0 // (op 22406) 
s_mov_b32 s15, scc // (op 22406) 

v_lshlrev_b32 v26, 2, v28 // (op 22406) 

s_cmp_eq_i32 s15, 0 // (op 22406) 
s_cselect_b32 s7, s6, s8 // (op 22406) 


s_mul_hi_u32 s8, s7, s12 // (op 22406) 
s_sub_u32 s15, s7, s8 // (op 22406) 
s_lshr_b32 s18, s15, 1 // (op 22406) 

s_add_u32 s15, s18, s8 // (op 22406) 


s_lshr_b32 s8, s15, s10 // (op 22406) 

s_and_b32 s15, s10, 2147483648 // (op 22406) 
s_cmp_eq_i32 s15, 0 // (op 22406) 
s_mov_b32 s18, scc // (op 22406) 

s_cmp_eq_i32 s18, 0 // (op 22406) 
s_cselect_b32 s15, s7, s8 // (op 22406) 


s_mul_i32 s8, s15, s14 // (op 22406) 

s_sub_u32 s15, s7, s8 // (op 22406) 

v_lshl_add_u32 v28, s15, 7, v27 // (op 22406) 


v_mul_lo_u32 v27, v28, s44 // (op 22406) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 22406) 
s_mul_i32 s8, s7, s18 // (op 22406) 


s_sub_u32 s7, s6, s8 // (op 22406) 


v_lshl_add_u32 v28, s7, 2, v26 // (op 22406) 


v_add_u32 v26, v28, v27 // (op 22406) 


s_mul_i32 s19, s44, s23 // (op 22408) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 22408) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 22408) least significant
s_mul_hi_u32 s6, s44, s22 // (op 22408) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 22408) most significant: sum
s_add_u32 s19, s19, s6 // (op 22408) most significant: sum


s_mov_b32 s64, s46 // (op 22408) reg expression
s_mov_b32 s65, s47 // (op 22408) reg expression
s_mov_b32 s66, s18 // (op 22408) reg expression
s_mov_b32 s67, 131072 // (op 22408) call()

v_and_b32 v27, 63, v1 // (op 22427) 
s_mul_i32 s15, s16, s50 // (op 22427) 
s_add_u32 s15, s15, s0 // (op 22427) 
v_lshrrev_b32 v28, 4, v27 // (op 22427) 
s_add_u32 s6, s15, s9 // (op 22427) 

v_and_b32 v29, 3, v28 // (op 22427) 

s_mul_hi_u32 s15, s6, s13 // (op 22427) 
v_and_b32 v28, 15, v1 // (op 22427) 
s_sub_u32 s7, s6, s15 // (op 22427) 
v_lshl_add_u32 v30, v29, 4, v28 // (op 22427) 


s_lshr_b32 s8, s7, 1 // (op 22427) 

v_lshrrev_b32 v28, 6, v1 // (op 22427) 
s_add_u32 s7, s8, s15 // (op 22427) 


v_and_b32 v29, 1, v28 // (op 22427) 

s_lshr_b32 s8, s7, s11 // (op 22427) 

v_lshl_add_u32 v28, v29, 6, v30 // (op 22427) 


s_and_b32 s7, s11, 2147483648 // (op 22427) 
v_lshrrev_b32 v29, 6, v27 // (op 22427) 

s_cmp_eq_i32 s7, 0 // (op 22427) 
s_mov_b32 s15, scc // (op 22427) 

v_lshlrev_b32 v27, 2, v29 // (op 22427) 

s_cmp_eq_i32 s15, 0 // (op 22427) 
s_cselect_b32 s7, s6, s8 // (op 22427) 


s_mul_hi_u32 s8, s7, s12 // (op 22427) 
s_sub_u32 s15, s7, s8 // (op 22427) 
s_lshr_b32 s18, s15, 1 // (op 22427) 

s_add_u32 s15, s18, s8 // (op 22427) 


s_lshr_b32 s8, s15, s10 // (op 22427) 

s_and_b32 s15, s10, 2147483648 // (op 22427) 
s_cmp_eq_i32 s15, 0 // (op 22427) 
s_mov_b32 s18, scc // (op 22427) 

s_cmp_eq_i32 s18, 0 // (op 22427) 
s_cselect_b32 s15, s7, s8 // (op 22427) 


s_mul_i32 s8, s15, s14 // (op 22427) 

s_sub_u32 s15, s7, s8 // (op 22427) 

v_lshl_add_u32 v29, s15, 7, v28 // (op 22427) 


v_mul_lo_u32 v28, v29, s44 // (op 22427) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 22427) 
s_mul_i32 s8, s7, s18 // (op 22427) 


s_sub_u32 s7, s6, s8 // (op 22427) 


v_lshl_add_u32 v29, s7, 2, v27 // (op 22427) 


v_add_u32 v27, v29, v28 // (op 22427) 


s_mul_i32 s19, s44, s23 // (op 22429) most significant: low of low * high
s_mul_i32 s15, s45, s22 // (op 22429) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 22429) least significant
s_mul_hi_u32 s6, s44, s22 // (op 22429) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 22429) most significant: sum
s_add_u32 s19, s19, s6 // (op 22429) most significant: sum


s_mov_b32 s64, s46 // (op 22429) reg expression
s_mov_b32 s65, s47 // (op 22429) reg expression
s_mov_b32 s66, s18 // (op 22429) reg expression
s_mov_b32 s67, 131072 // (op 22429) call()

v_and_b32 v28, -4, v1 // (op 21717) 
v_sub_u32 v29, v1, v28 // (op 21717) 

v_lshlrev_b32 v28, 5, v29 // (op 21717) 

v_lshrrev_b32 v29, 2, v1 // (op 21717) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 21717) 


v_lshlrev_b32 v28, 2, v30 // (op 21717) 

v_lshrrev_b32 v29, 3, v28 // (op 21717) 

s_nop 0
v_readfirstlane_b32 s15, v29 // (op 21717) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23068) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 21011) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21011) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21011) 


v_mov_b64 v[30:31], v[20:21] // (op 21011) 
s_mov_b32 m0, s15 // (op 21011) reg expression
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 21011) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 21011) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 21011) most significant half

s_add_u32 m0, m0, 4096 // (op 21011) 
buffer_load_dwordx4 v30, s[72:75], 0 offen offset: 0 lds // (op 21011) Load value direct to lds



v_and_b32 v28, -4, v1 // (op 21642) 
v_sub_u32 v29, v1, v28 // (op 21642) 

v_lshlrev_b32 v28, 5, v29 // (op 21642) 

v_lshrrev_b32 v29, 2, v1 // (op 21642) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 21642) 


v_lshlrev_b32 v28, 2, v30 // (op 21642) 

v_lshrrev_b32 v29, 3, v28 // (op 21642) 

s_nop 0
v_readfirstlane_b32 s15, v29 // (op 21642) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23048) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 21016) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21016) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21016) 


v_mov_b64 v[30:31], v[16:17] // (op 21016) 
s_add_u32 s6, 8192, s15 // (op 21016) 
s_mov_b32 m0, s6 // (op 21016) reg expression
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 21016) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 21016) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 21016) most significant half

s_add_u32 m0, m0, 4096 // (op 21016) 
buffer_load_dwordx4 v30, s[68:71], 0 offen offset: 0 lds // (op 21016) Load value direct to lds






v_mov_b32 v28, v26 // (op 174) reg expression

buffer_load_dwordx4 v[30:33], v28, s[64:67], 0 offen // (op 174) Load value




v_mov_b32 v26, v3 // (op 172) reg expression

buffer_load_dwordx4 v[34:37], v26, s[52:55], 0 offen // (op 172) Load value


v_and_b32 v3, -4, v1 // (op 21875) 
v_sub_u32 v26, v1, v3 // (op 21875) 

v_lshlrev_b32 v3, 5, v26 // (op 21875) 

v_lshrrev_b32 v26, 2, v1 // (op 21875) 
v_lshl_add_u32 v28, v26, 7, v3 // (op 21875) 


v_lshlrev_b32 v3, 2, v28 // (op 21875) 

v_lshrrev_b32 v26, 3, v3 // (op 21875) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 21875) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23098) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 21021) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21021) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21021) 


v_mov_b64 v[38:39], v[20:21] // (op 21021) 
s_add_u32 s6, 16384, s15 // (op 21021) 
s_mov_b32 m0, s6 // (op 21021) reg expression
buffer_load_dwordx4 v38, s[72:75], 64 offen offset: 0 lds // (op 21021) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v28 // (op 21021) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 21021) most significant half

s_add_u32 m0, m0, 4096 // (op 21021) 
buffer_load_dwordx4 v38, s[72:75], 64 offen offset: 0 lds // (op 21021) Load value direct to lds




v_and_b32 v3, -4, v1 // (op 21831) 
v_sub_u32 v26, v1, v3 // (op 21831) 

v_lshlrev_b32 v3, 5, v26 // (op 21831) 

v_lshrrev_b32 v26, 2, v1 // (op 21831) 
v_lshl_add_u32 v28, v26, 7, v3 // (op 21831) 


v_lshlrev_b32 v3, 2, v28 // (op 21831) 

v_lshrrev_b32 v26, 3, v3 // (op 21831) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 21831) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23088) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 21023) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21023) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21023) 


v_mov_b64 v[38:39], v[16:17] // (op 21023) 
s_add_u32 s6, 24576, s15 // (op 21023) 
s_mov_b32 m0, s6 // (op 21023) reg expression
buffer_load_dwordx4 v38, s[68:71], 64 offen offset: 0 lds // (op 21023) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v28 // (op 21023) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 21023) most significant half

s_add_u32 m0, m0, 4096 // (op 21023) 
buffer_load_dwordx4 v38, s[68:71], 64 offen offset: 0 lds // (op 21023) Load value direct to lds




v_and_b32 v3, -4, v1 // (op 21963) 
v_sub_u32 v26, v1, v3 // (op 21963) 

v_lshlrev_b32 v3, 5, v26 // (op 21963) 

v_lshrrev_b32 v26, 2, v1 // (op 21963) 
v_lshl_add_u32 v28, v26, 7, v3 // (op 21963) 


v_lshlrev_b32 v3, 2, v28 // (op 21963) 

v_lshrrev_b32 v26, 3, v3 // (op 21963) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 21963) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23118) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 21025) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21025) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21025) 


v_mov_b64 v[38:39], v[20:21] // (op 21025) 
s_add_u32 s6, 32768, s15 // (op 21025) 
s_mov_b32 m0, s6 // (op 21025) reg expression
s_mov_b32 s7, 128 // (op 21025) call()
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 21025) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v28 // (op 21025) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 21025) most significant half

s_add_u32 m0, m0, 4096 // (op 21025) 
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 21025) Load value direct to lds





v_and_b32 v3, -4, v1 // (op 21919) 
v_sub_u32 v26, v1, v3 // (op 21919) 

v_lshlrev_b32 v3, 5, v26 // (op 21919) 

v_lshrrev_b32 v26, 2, v1 // (op 21919) 
v_lshl_add_u32 v28, v26, 7, v3 // (op 21919) 


v_lshlrev_b32 v3, 2, v28 // (op 21919) 

v_lshrrev_b32 v26, 3, v3 // (op 21919) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 21919) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23109) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 21027) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21027) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21027) 


v_mov_b64 v[38:39], v[16:17] // (op 21027) 
s_add_u32 s6, 40960, s15 // (op 21027) 
s_mov_b32 m0, s6 // (op 21027) reg expression
s_mov_b32 s7, 128 // (op 21027) call()
buffer_load_dwordx4 v38, s[68:71], s7 offen offset: 0 lds // (op 21027) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v28 // (op 21027) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 21027) most significant half

s_add_u32 m0, m0, 4096 // (op 21027) 
buffer_load_dwordx4 v38, s[68:71], s7 offen offset: 0 lds // (op 21027) Load value direct to lds





v_and_b32 v3, -4, v1 // (op 22051) 
v_sub_u32 v26, v1, v3 // (op 22051) 

v_lshlrev_b32 v3, 5, v26 // (op 22051) 

v_lshrrev_b32 v26, 2, v1 // (op 22051) 
v_lshl_add_u32 v28, v26, 7, v3 // (op 22051) 


v_lshlrev_b32 v3, 2, v28 // (op 22051) 

v_lshrrev_b32 v26, 3, v3 // (op 22051) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 22051) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23137) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 21029) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21029) 

v_lshrrev_b64 v[28:29], 3, s[28:29] // (op 21029) 


v_mov_b64 v[38:39], v[20:21] // (op 21029) 
s_add_u32 s6, 49152, s15 // (op 21029) 
s_mov_b32 m0, s6 // (op 21029) reg expression
s_mov_b32 s7, 192 // (op 21029) call()
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 21029) Load value direct to lds
v_add_co_u32 v38, s[18:19], v38, v28 // (op 21029) least significant half
v_addc_co_u32 v39, s[18:19], 0, v39, s[18:19] // (op 21029) most significant half

s_add_u32 m0, m0, 4096 // (op 21029) 
buffer_load_dwordx4 v38, s[72:75], s7 offen offset: 0 lds // (op 21029) Load value direct to lds






v_and_b32 v3, -4, v1 // (op 22007) 
v_sub_u32 v26, v1, v3 // (op 22007) 

v_lshlrev_b32 v3, 5, v26 // (op 22007) 

v_lshrrev_b32 v26, 2, v1 // (op 22007) 
v_lshl_add_u32 v20, v26, 7, v3 // (op 22007) 


v_lshlrev_b32 v3, 2, v20 // (op 22007) 

v_lshrrev_b32 v26, 3, v3 // (op 22007) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 22007) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 23128) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 21031) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 21031) 

v_lshrrev_b64 v[20:21], 3, s[28:29] // (op 21031) 


v_mov_b64 v[28:29], v[16:17] // (op 21031) 
s_add_u32 s6, 57344, s15 // (op 21031) 
s_mov_b32 m0, s6 // (op 21031) reg expression
s_mov_b32 s7, 192 // (op 21031) call()
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 21031) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v20 // (op 21031) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 21031) most significant half

s_add_u32 m0, m0, 4096 // (op 21031) 
buffer_load_dwordx4 v28, s[68:71], s7 offen offset: 0 lds // (op 21031) Load value direct to lds






s_waitcnt vmcnt(16)
s_barrier  // (op 23065) 



ds_read_b128 v[38:41], v15 // (op 20193) Load local data 



ds_read_b128 v[42:45], v15 offset:1024 // (op 20198) Load local data 



ds_read_b128 v[46:49], v15 offset:2048 // (op 20203) Load local data 



ds_read_b128 v[50:53], v15 offset:3072 // (op 20208) Load local data 
s_waitcnt vmcnt(14)
s_barrier  // (op 23045) 



ds_read_b128 v[54:57], v4 offset:8192 // (op 20173) Load local data 



ds_read_b128 v[58:61], v4 offset:9216 // (op 20178) Load local data 



ds_read_b128 v[62:65], v4 offset:10240 // (op 20183) Load local data 



ds_read_b128 v[66:69], v4 offset:11264 // (op 20188) Load local data 
v_and_b32 v3, -4, v1 // (op 21614) 
v_sub_u32 v26, v1, v3 // (op 21614) 

v_lshlrev_b32 v3, 5, v26 // (op 21614) 

v_lshrrev_b32 v26, 2, v1 // (op 21614) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21614) 


v_lshlrev_b32 v3, 2, v16 // (op 21614) 

v_lshrrev_b32 v26, 3, v3 // (op 21614) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 21614) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21689) 
v_sub_u32 v26, v1, v3 // (op 21689) 

v_lshlrev_b32 v3, 5, v26 // (op 21689) 

v_lshrrev_b32 v26, 2, v1 // (op 21689) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21689) 


v_lshlrev_b32 v3, 2, v16 // (op 21689) 

v_lshrrev_b32 v26, 3, v3 // (op 21689) 

s_nop 0
v_readfirstlane_b32 s6, v26 // (op 21689) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21816) 
v_sub_u32 v26, v1, v3 // (op 21816) 

v_lshlrev_b32 v3, 5, v26 // (op 21816) 

v_lshrrev_b32 v26, 2, v1 // (op 21816) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21816) 


v_lshlrev_b32 v3, 2, v16 // (op 21816) 

v_lshrrev_b32 v26, 3, v3 // (op 21816) 

s_nop 0
v_readfirstlane_b32 s7, v26 // (op 21816) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21860) 
v_sub_u32 v26, v1, v3 // (op 21860) 

v_lshlrev_b32 v3, 5, v26 // (op 21860) 

v_lshrrev_b32 v26, 2, v1 // (op 21860) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21860) 


v_lshlrev_b32 v3, 2, v16 // (op 21860) 

v_lshrrev_b32 v26, 3, v3 // (op 21860) 

s_nop 0
v_readfirstlane_b32 s8, v26 // (op 21860) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21904) 
v_sub_u32 v26, v1, v3 // (op 21904) 

v_lshlrev_b32 v3, 5, v26 // (op 21904) 

v_lshrrev_b32 v26, 2, v1 // (op 21904) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21904) 


v_lshlrev_b32 v3, 2, v16 // (op 21904) 

v_lshrrev_b32 v26, 3, v3 // (op 21904) 

s_nop 0
v_readfirstlane_b32 s18, v26 // (op 21904) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21948) 
v_sub_u32 v26, v1, v3 // (op 21948) 

v_lshlrev_b32 v3, 5, v26 // (op 21948) 

v_lshrrev_b32 v26, 2, v1 // (op 21948) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21948) 


v_lshlrev_b32 v3, 2, v16 // (op 21948) 

v_lshrrev_b32 v26, 3, v3 // (op 21948) 

s_nop 0
v_readfirstlane_b32 s19, v26 // (op 21948) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21992) 
v_sub_u32 v26, v1, v3 // (op 21992) 

v_lshlrev_b32 v3, 5, v26 // (op 21992) 

v_lshrrev_b32 v26, 2, v1 // (op 21992) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 21992) 


v_lshlrev_b32 v3, 2, v16 // (op 21992) 

v_lshrrev_b32 v26, 3, v3 // (op 21992) 

s_nop 0
v_readfirstlane_b32 s28, v26 // (op 21992) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 22036) 
v_sub_u32 v26, v1, v3 // (op 22036) 

v_lshlrev_b32 v3, 5, v26 // (op 22036) 

v_lshrrev_b32 v26, 2, v1 // (op 22036) 
v_lshl_add_u32 v16, v26, 7, v3 // (op 22036) 


v_lshlrev_b32 v3, 2, v16 // (op 22036) 

v_lshrrev_b32 v26, 3, v3 // (op 22036) 

s_nop 0
v_readfirstlane_b32 s29, v26 // (op 22036) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s1, 0 // (op 3523) call()
s_ashr_i64 s[48:49], s[2:3], 7 // (op 3522) 
s_mov_b32 s76, s48 // (op 3522) convert

s_and_b32 s48, s76, 4294967292 // (op 3522) 

s_cmp_lt_u32 s1, s48 // (op 3522) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_3522 // (op 3522) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_3522 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_3522:
s_add_u32 s1, s1, 4 // (op 3524) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 23071) 
s_waitcnt vmcnt(13)
v_mov_b32 v70, v30 // (op 20853) reg expression

v_mov_b32 v71, v31 // (op 20853) reg expression
v_mov_b32 v72, v32 // (op 20853) reg expression
v_mov_b32 v73, v33 // (op 20853) reg expression
s_lshl_b64 s[48:49], s[36:37], 6 // (op 20922) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 20922) 

v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 20922) 


v_mov_b64 v[20:21], v[24:25] // (op 20922) 
v_mov_b64 v[28:29], v[20:21] // (op 20922) 
s_mov_b32 m0, s6 // (op 20922) reg expression
s_mov_b32 s48, 256 // (op 20922) call()
v_bfe_u32 v74, v71, 0, 8 // (op 19091) 
v_bfe_u32 v75, v71, 8, 8 // (op 19091) 
v_bfe_u32 v76, v71, 16, 8 // (op 19091) 
v_bfe_u32 v77, v71, 24, 8 // (op 19091) 
v_permlane16_swap_b32 v74, v75 // (op 19091) 
v_bfe_u32 v78, v72, 0, 8 // (op 19109) 
v_permlane16_swap_b32 v76, v77 // (op 19091) 
v_bfe_u32 v79, v72, 8, 8 // (op 19109) 
v_bfe_u32 v80, v72, 16, 8 // (op 19109) 
v_permlane32_swap_b32 v74, v76 // (op 19091) 
v_permlane32_swap_b32 v75, v77 // (op 19091) 
v_bfe_u32 v81, v72, 24, 8 // (op 19109) 
v_permlane16_swap_b32 v78, v79 // (op 19109) 
v_bfe_u32 v82, v73, 0, 8 // (op 19127) 
v_permlane16_swap_b32 v80, v81 // (op 19109) 
v_bfe_u32 v83, v73, 8, 8 // (op 19127) 
v_bfe_u32 v84, v73, 16, 8 // (op 19127) 
v_permlane32_swap_b32 v78, v80 // (op 19109) 
v_permlane32_swap_b32 v79, v81 // (op 19109) 
v_bfe_u32 v85, v73, 24, 8 // (op 19127) 
v_permlane16_swap_b32 v82, v83 // (op 19127) 
s_nop 0
v_permlane16_swap_b32 v84, v85 // (op 19127) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v82, v84 // (op 19127) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v83, v85 // (op 19127) 
s_waitcnt vmcnt(12)
v_mov_b32 v86, v34 // (op 20869) reg expression
v_mov_b32 v87, v35 // (op 20869) reg expression
v_mov_b32 v88, v36 // (op 20869) reg expression
v_mov_b32 v89, v37 // (op 20869) reg expression
buffer_load_dwordx4 v28, s[72:75], s48 offen offset: 0 lds // (op 20922) Load value direct to lds
v_add_co_u32 v28, s[76:77], v28, v16 // (op 20922) least significant half
v_addc_co_u32 v29, s[76:77], 0, v29, s[76:77] // (op 20922) most significant half

s_add_u32 m0, m0, 4096 // (op 20922) 
v_bfe_u32 v90, v86, 0, 8 // (op 18947) 
v_bfe_u32 v91, v86, 8, 8 // (op 18947) 
v_bfe_u32 v92, v86, 16, 8 // (op 18947) 
v_bfe_u32 v93, v86, 24, 8 // (op 18947) 
v_permlane16_swap_b32 v90, v91 // (op 18947) 
v_bfe_u32 v94, v87, 0, 8 // (op 18962) 
v_permlane16_swap_b32 v92, v93 // (op 18947) 
v_bfe_u32 v95, v87, 8, 8 // (op 18962) 
v_bfe_u32 v96, v87, 16, 8 // (op 18962) 
v_permlane32_swap_b32 v90, v92 // (op 18947) 
v_permlane32_swap_b32 v91, v93 // (op 18947) 
v_bfe_u32 v97, v87, 24, 8 // (op 18962) 
v_permlane16_swap_b32 v94, v95 // (op 18962) 
v_bfe_u32 v98, v88, 0, 8 // (op 18979) 
buffer_load_dwordx4 v28, s[72:75], s48 offen offset: 0 lds // (op 20922) Load value direct to lds



v_permlane16_swap_b32 v96, v97 // (op 18962) 
v_bfe_u32 v99, v88, 8, 8 // (op 18979) 
v_bfe_u32 v100, v88, 16, 8 // (op 18979) 
v_permlane32_swap_b32 v94, v96 // (op 18962) 
v_permlane32_swap_b32 v95, v97 // (op 18962) 
v_bfe_u32 v101, v88, 24, 8 // (op 18979) 
v_permlane16_swap_b32 v98, v99 // (op 18979) 
v_bfe_u32 v102, v89, 0, 8 // (op 18996) 
v_permlane16_swap_b32 v100, v101 // (op 18979) 
v_bfe_u32 v103, v89, 8, 8 // (op 18996) 
v_bfe_u32 v104, v89, 16, 8 // (op 18996) 
v_permlane32_swap_b32 v98, v100 // (op 18979) 
v_permlane32_swap_b32 v99, v101 // (op 18979) 
v_bfe_u32 v105, v89, 24, 8 // (op 18996) 
v_permlane16_swap_b32 v102, v103 // (op 18996) 
s_nop 0
v_permlane16_swap_b32 v104, v105 // (op 18996) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v102, v104 // (op 18996) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v103, v105 // (op 18996) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23051) 


s_lshl_b64 s[48:49], s[30:31], 6 // (op 20917) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 20917) 

v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 20917) 


v_mov_b64 v[28:29], v[22:23] // (op 20917) 
v_mov_b64 v[86:87], v[28:29] // (op 20917) 
s_add_u32 s48, 8192, s15 // (op 20917) 
s_mov_b32 m0, s48 // (op 20917) reg expression
s_mov_b32 s49, 256 // (op 20917) call()
buffer_load_dwordx4 v86, s[68:71], s49 offen offset: 0 lds // (op 20917) Load value direct to lds
v_add_co_u32 v86, s[76:77], v86, v16 // (op 20917) least significant half
v_addc_co_u32 v87, s[76:77], 0, v87, s[76:77] // (op 20917) most significant half

s_add_u32 m0, m0, 4096 // (op 20917) 
v_bfe_u32 v106, v70, 0, 8 // (op 19076) 
v_bfe_u32 v107, v70, 8, 8 // (op 19076) 
v_bfe_u32 v108, v70, 16, 8 // (op 19076) 
v_bfe_u32 v109, v70, 24, 8 // (op 19076) 
v_permlane16_swap_b32 v106, v107 // (op 19076) 
s_nop 0
v_permlane16_swap_b32 v108, v109 // (op 19076) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v106, v108 // (op 19076) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 19076) 
buffer_load_dwordx4 v86, s[68:71], s49 offen offset: 0 lds // (op 20917) Load value direct to lds







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[38:41], a[0:3], v90, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 175) 
v_mov_b32 v3, v27 // (op 20864) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:57], v[42:45], a[4:7], v90, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 683) 
v_add_u32 v26, 16, v3 // (op 20864) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[46:49], a[8:11], v90, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 909) 
buffer_load_dwordx4 v[30:33], v26, s[64:67], 0 offen // (op 20864) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:57], v[50:53], a[12:15], v90, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1135) 
v_add_u32 v27, 16, v27 // (op 22417) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:61], v[38:41], a[16:19], v91, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1807) 
v_mov_b32 v3, v18 // (op 20880) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:61], v[42:45], a[20:23], v91, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1731) 
v_add_u32 v26, 16, v3 // (op 20880) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[46:49], a[24:27], v91, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1655) 
buffer_load_dwordx4 v[34:37], v26, s[52:55], 0 offen // (op 20880) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:61], v[50:53], a[28:31], v91, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1579) 
v_add_u32 v18, 16, v18 // (op 22365) 
s_waitcnt vmcnt(16)
s_barrier  // (op 23094) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:65], v[38:41], a[32:35], v92, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2809) 
ds_read_b128 v[70:73], v8 offset:16384 // (op 478) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[62:65], v[42:45], a[36:39], v92, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2733) 
ds_read_b128 v[86:89], v8 offset:17408 // (op 694) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:65], v[46:49], a[40:43], v92, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2657) 
ds_read_b128 v[110:113], v8 offset:18432 // (op 920) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[62:65], v[50:53], a[44:47], v92, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2581) 
ds_read_b128 v[114:117], v8 offset:19456 // (op 1146) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:69], v[38:41], a[48:51], v93, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3811) 
s_waitcnt vmcnt(14)
s_barrier  // (op 23084) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:69], v[42:45], a[52:55], v93, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3735) 
ds_read_b128 v[54:57], v7 offset:24576 // (op 476) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[66:69], v[46:49], a[56:59], v93, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3659) 
ds_read_b128 v[58:61], v7 offset:25600 // (op 1589) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:69], v[50:53], a[60:63], v93, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3583) 
ds_read_b128 v[62:65], v7 offset:26624 // (op 2591) Load local data 





ds_read_b128 v[90:93], v7 offset:27648 // (op 3593) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[70:73], a[0:3], v94, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 483) 
s_barrier  // (op 23101) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:57], v[86:89], a[4:7], v94, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 684) 
s_lshl_b64 s[48:49], s[36:37], 6 // (op 20944) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:57], v[114:117], a[12:15], v94, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1136) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 20944) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[110:113], a[8:11], v94, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 910) 
v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 20944) 


v_mov_b64 v[106:107], v[20:21] // (op 20944) 
s_add_u32 s48, 16384, s8 // (op 20944) 
s_mov_b32 m0, s48 // (op 20944) reg expression
s_mov_b32 s49, 320 // (op 20944) call()
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:61], v[70:73], a[16:19], v95, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1808) 
s_waitcnt vmcnt(12)
s_barrier  // (op 23115) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:61], v[86:89], a[20:23], v95, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1732) 
ds_read_b128 v[38:41], v10 offset:32768 // (op 506) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[110:113], a[24:27], v95, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1656) 
ds_read_b128 v[42:45], v10 offset:33792 // (op 704) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:61], v[114:117], a[28:31], v95, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1580) 
ds_read_b128 v[46:49], v10 offset:34816 // (op 930) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:65], v[70:73], a[32:35], v96, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2810) 
ds_read_b128 v[50:53], v10 offset:35840 // (op 1156) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[62:65], v[86:89], a[36:39], v96, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2734) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:65], v[110:113], a[40:43], v96, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2658) 
s_waitcnt vmcnt(10)
s_barrier  // (op 23105) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[62:65], v[114:117], a[44:47], v96, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2582) 
ds_read_b128 v[54:57], v9 offset:40960 // (op 504) Load local data 
buffer_load_dwordx4 v106, s[72:75], s49 offen offset: 0 lds // (op 20944) Load value direct to lds



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[70:73], a[48:51], v97, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3812) 
ds_read_b128 v[58:61], v9 offset:41984 // (op 1599) Load local data 

v_add_co_u32 v106, s[76:77], v106, v16 // (op 20944) least significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[86:89], a[52:55], v97, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3736) 
v_addc_co_u32 v107, s[76:77], 0, v107, s[76:77] // (op 20944) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[110:113], a[56:59], v97, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3660) 
s_add_u32 m0, m0, 4096 // (op 20944) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[114:117], a[60:63], v97, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3584) 
buffer_load_dwordx4 v106, s[72:75], s49 offen offset: 0 lds // (op 20944) Load value direct to lds








s_barrier  // (op 23091) 
ds_read_b128 v[70:73], v9 offset:43008 // (op 2601) Load local data 




s_lshl_b64 s[48:49], s[30:31], 6 // (op 20939) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 20939) 

v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 20939) 


v_mov_b64 v[74:75], v[28:29] // (op 20939) 
s_add_u32 s48, 24576, s7 // (op 20939) 
s_mov_b32 m0, s48 // (op 20939) reg expression
s_mov_b32 s49, 320 // (op 20939) call()
buffer_load_dwordx4 v74, s[68:71], s49 offen offset: 0 lds // (op 20939) Load value direct to lds
v_add_co_u32 v74, s[76:77], v74, v16 // (op 20939) least significant half
v_addc_co_u32 v75, s[76:77], 0, v75, s[76:77] // (op 20939) most significant half

s_add_u32 m0, m0, 4096 // (op 20939) 
ds_read_b128 v[86:89], v9 offset:44032 // (op 3603) Load local data 
buffer_load_dwordx4 v74, s[68:71], s49 offen offset: 0 lds // (op 20939) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[38:41], a[0:3], v98, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 511) 
s_barrier  // (op 23121) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:57], v[42:45], a[4:7], v98, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 685) 
s_lshl_b64 s[48:49], s[36:37], 6 // (op 20954) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[46:49], a[8:11], v98, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 911) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 20954) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:57], v[50:53], a[12:15], v98, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1137) 
v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 20954) 


v_mov_b64 v[74:75], v[20:21] // (op 20954) 
s_add_u32 s48, 32768, s19 // (op 20954) 
s_mov_b32 m0, s48 // (op 20954) reg expression
s_mov_b32 s49, 384 // (op 20954) call()
buffer_load_dwordx4 v74, s[72:75], s49 offen offset: 0 lds // (op 20954) Load value direct to lds
v_add_co_u32 v74, s[76:77], v74, v16 // (op 20954) least significant half
v_addc_co_u32 v75, s[76:77], 0, v75, s[76:77] // (op 20954) most significant half

s_add_u32 m0, m0, 4096 // (op 20954) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:61], v[38:41], a[16:19], v99, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1809) 
buffer_load_dwordx4 v74, s[72:75], s49 offen offset: 0 lds // (op 20954) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:61], v[42:45], a[20:23], v99, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1733) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 23112) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[46:49], a[24:27], v99, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1657) 
s_lshl_b64 s[48:49], s[30:31], 6 // (op 20949) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:61], v[50:53], a[28:31], v99, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1581) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 20949) 

v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 20949) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:73], v[38:41], a[32:35], v100, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2811) 
v_mov_b64 v[74:75], v[28:29] // (op 20949) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[42:45], a[36:39], v100, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2735) 
s_add_u32 s48, 40960, s18 // (op 20949) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[70:73], v[46:49], a[40:43], v100, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2659) 
s_mov_b32 m0, s48 // (op 20949) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[70:73], v[50:53], a[44:47], v100, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2583) 
s_mov_b32 s49, 384 // (op 20949) call()

s_waitcnt vmcnt(14)
s_barrier  // (op 23134) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[38:41], a[48:51], v101, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3813) 
ds_read_b128 v[62:65], v12 offset:49152 // (op 534) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[42:45], a[52:55], v101, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3737) 
ds_read_b128 v[66:69], v12 offset:50176 // (op 714) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v101, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3661) 
ds_read_b128 v[70:73], v12 offset:51200 // (op 940) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[50:53], a[60:63], v101, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3585) 
ds_read_b128 v[90:93], v12 offset:52224 // (op 1166) Load local data 



buffer_load_dwordx4 v74, s[68:71], s49 offen offset: 0 lds // (op 20949) Load value direct to lds
v_add_co_u32 v74, s[76:77], v74, v16 // (op 20949) least significant half
v_addc_co_u32 v75, s[76:77], 0, v75, s[76:77] // (op 20949) most significant half

s_add_u32 m0, m0, 4096 // (op 20949) 
buffer_load_dwordx4 v74, s[68:71], s49 offen offset: 0 lds // (op 20949) Load value direct to lds




s_waitcnt vmcnt(14)
s_barrier  // (op 23124) 



ds_read_b128 v[54:57], v11 offset:57344 // (op 532) Load local data 



ds_read_b128 v[58:61], v11 offset:58368 // (op 1609) Load local data 



ds_read_b128 v[86:89], v11 offset:59392 // (op 2611) Load local data 



ds_read_b128 v[74:77], v11 offset:60416 // (op 3613) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[62:65], a[0:3], v102, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 539) 
s_barrier  // (op 23140) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:57], v[66:69], a[4:7], v102, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 686) 
s_lshl_b64 s[48:49], s[36:37], 6 // (op 20964) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[70:73], a[8:11], v102, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 912) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 20964) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:57], v[90:93], a[12:15], v102, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1138) 
v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 20964) 


v_mov_b64 v[78:79], v[20:21] // (op 20964) 
s_add_u32 s48, 49152, s29 // (op 20964) 
s_mov_b32 m0, s48 // (op 20964) reg expression
s_mov_b32 s49, 448 // (op 20964) call()
buffer_load_dwordx4 v78, s[72:75], s49 offen offset: 0 lds // (op 20964) Load value direct to lds
v_add_co_u32 v78, s[76:77], v78, v16 // (op 20964) least significant half
v_addc_co_u32 v79, s[76:77], 0, v79, s[76:77] // (op 20964) most significant half

s_add_u32 m0, m0, 4096 // (op 20964) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:61], v[62:65], a[16:19], v103, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1810) 
s_waitcnt vmcnt(13)
s_barrier  // (op 23077) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:61], v[66:69], a[20:23], v103, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1734) 
ds_read_b128 v[38:41], v6 // (op 170) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[70:73], a[24:27], v103, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1658) 
ds_read_b128 v[42:45], v6 offset:1024 // (op 680) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:61], v[90:93], a[28:31], v103, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1582) 
ds_read_b128 v[46:49], v6 offset:2048 // (op 906) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[86:89], v[62:65], a[32:35], v104, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2812) 
ds_read_b128 v[50:53], v6 offset:3072 // (op 1132) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[86:89], v[66:69], a[36:39], v104, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2736) 
s_waitcnt vmcnt(11)
s_barrier  // (op 23057) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[86:89], v[70:73], a[40:43], v104, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2660) 
ds_read_b128 v[54:57], v5 offset:8192 // (op 168) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[90:93], a[44:47], v104, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2584) 
buffer_load_dwordx4 v78, s[72:75], s49 offen offset: 0 lds // (op 20964) Load value direct to lds









s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:77], v[62:65], a[48:51], v105, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3814) 
s_barrier  // (op 23131) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:77], v[66:69], a[52:55], v105, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3738) 
ds_read_b128 v[58:61], v5 offset:9216 // (op 1575) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:77], v[70:73], a[56:59], v105, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3662) 
ds_read_b128 v[62:65], v5 offset:10240 // (op 2577) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[90:93], a[60:63], v105, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3586) 
ds_read_b128 v[66:69], v5 offset:11264 // (op 3579) Load local data 




v_add_co_u32 v24, vcc, 256, v24 // (op 21288) least significant half
v_addc_co_u32 v25, vcc, 0, v25, vcc // (op 21288) most significant half
s_lshl_b64 s[48:49], s[30:31], 6 // (op 20959) 
s_lshl_b64 s[76:77], s[48:49], 2 // (op 20959) 

v_lshrrev_b64 v[16:17], 3, s[76:77] // (op 20959) 


v_mov_b64 v[20:21], v[28:29] // (op 20959) 
s_add_u32 s48, 57344, s28 // (op 20959) 
s_mov_b32 m0, s48 // (op 20959) reg expression
s_mov_b32 s49, 448 // (op 20959) call()
buffer_load_dwordx4 v20, s[68:71], s49 offen offset: 0 lds // (op 20959) Load value direct to lds
v_add_co_u32 v20, s[76:77], v20, v16 // (op 20959) least significant half
v_addc_co_u32 v21, s[76:77], 0, v21, s[76:77] // (op 20959) most significant half

s_add_u32 m0, m0, 4096 // (op 20959) 
buffer_load_dwordx4 v20, s[68:71], s49 offen offset: 0 lds // (op 20959) Load value direct to lds





v_add_co_u32 v22, vcc, 256, v22 // (op 21142) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 21142) most significant half
s_ashr_i64 s[48:49], s[2:3], 7 // (op 3522) 
s_mov_b32 s76, s48 // (op 3522) convert

s_and_b32 s48, s76, 4294967292 // (op 3522) 

s_cmp_lt_u32 s1, s48 // (op 3522) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_3522 // (op 3522) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoop_3522 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_3522:














s_barrier  // (op 23303) 
v_lshrrev_b32 v3, 2, v1 // (op 21066) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 21066) 
v_and_b32 v16, -4, v1 // (op 21066) 
s_and_b32 s15, s18, 4294967292 // (op 21066) 
v_sub_u32 v17, v1, v16 // (op 21066) 

s_add_u32 s6, s0, s15 // (op 21066) 

v_lshlrev_b32 v16, 5, v17 // (op 21066) 

s_mul_i32 s15, s16, s50 // (op 21066) 
s_add_u32 s15, s15, s6 // (op 21066) 

s_add_u32 s6, s15, s9 // (op 21066) 

s_mul_hi_u32 s15, s6, s13 // (op 21066) 
s_sub_u32 s7, s6, s15 // (op 21066) 
s_lshr_b32 s8, s7, 1 // (op 21066) 

s_add_u32 s7, s8, s15 // (op 21066) 


s_lshr_b32 s8, s7, s11 // (op 21066) 

s_and_b32 s7, s11, 2147483648 // (op 21066) 
s_cmp_eq_i32 s7, 0 // (op 21066) 
s_mov_b32 s15, scc // (op 21066) 

s_cmp_eq_i32 s15, 0 // (op 21066) 
s_cselect_b32 s7, s6, s8 // (op 21066) 


s_mul_hi_u32 s8, s7, s12 // (op 21066) 
s_sub_u32 s15, s7, s8 // (op 21066) 
s_lshr_b32 s28, s15, 1 // (op 21066) 

s_add_u32 s15, s28, s8 // (op 21066) 


s_lshr_b32 s8, s15, s10 // (op 21066) 

s_and_b32 s15, s10, 2147483648 // (op 21066) 
s_cmp_eq_i32 s15, 0 // (op 21066) 
s_mov_b32 s28, scc // (op 21066) 

s_cmp_eq_i32 s28, 0 // (op 21066) 
s_cselect_b32 s15, s7, s8 // (op 21066) 


s_mul_i32 s8, s7, s18 // (op 21066) 


v_lshl_add_u32 v17, s15, 7, v3 // (op 21066) 


s_sub_u32 s7, s6, s8 // (op 21066) 


v_lshl_add_u32 v3, s7, 7, v16 // (op 21066) 


v_mov_b32 v21, 0 // (op 21066) convert
v_mov_b32 v20, v3 // (op 21066) convert

v_mul_lo_u32 v23, v17, s31 // (op 21066) most significant: low of low * high
v_mul_hi_u32 v3, v17, s30 // (op 21066) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 21066) least significant: low of low * low
v_add_u32 v23, v23, v3 // (op 21066) most significant: sum

v_add_co_u32 v22, s[18:19], v22, v20 // (op 21066) least significant half
v_addc_co_u32 v23, s[18:19], v23, v21, s[18:19] // (op 21066) most significant half



v_lshlrev_b64 v[20:21], 2, v[22:23] // (op 21066) 

v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 21066) 

s_mul_i32 s19, s30, s35 // (op 21068) most significant: low of low * high
s_mul_i32 s15, s31, s34 // (op 21068) most significant: low of high * low
s_mul_i32 s18, s30, s34 // (op 21068) least significant
s_mul_hi_u32 s6, s30, s34 // (op 21068) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21068) most significant: sum
s_add_u32 s19, s19, s6 // (op 21068) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21068) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21068) 

s_mov_b32 s68, s32 // (op 21068) reg expression
s_mov_b32 s69, s33 // (op 21068) reg expression
s_mov_b32 s70, s18 // (op 21068) reg expression
s_mov_b32 s71, 131072 // (op 21068) call()

v_lshrrev_b32 v3, 2, v1 // (op 21232) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 21232) 
v_and_b32 v18, -4, v1 // (op 21232) 
s_and_b32 s15, s18, 4294967292 // (op 21232) 
v_sub_u32 v20, v1, v18 // (op 21232) 

s_add_u32 s6, s0, s15 // (op 21232) 

v_lshlrev_b32 v18, 5, v20 // (op 21232) 

s_mul_i32 s15, s16, s50 // (op 21232) 
s_add_u32 s15, s15, s6 // (op 21232) 

s_add_u32 s6, s15, s9 // (op 21232) 

s_mul_hi_u32 s15, s6, s13 // (op 21232) 
s_sub_u32 s7, s6, s15 // (op 21232) 
s_lshr_b32 s8, s7, 1 // (op 21232) 

s_add_u32 s7, s8, s15 // (op 21232) 


s_lshr_b32 s8, s7, s11 // (op 21232) 

s_and_b32 s7, s11, 2147483648 // (op 21232) 
s_cmp_eq_i32 s7, 0 // (op 21232) 
s_mov_b32 s15, scc // (op 21232) 

s_cmp_eq_i32 s15, 0 // (op 21232) 
s_cselect_b32 s7, s6, s8 // (op 21232) 


s_mul_hi_u32 s8, s7, s12 // (op 21232) 
s_sub_u32 s15, s7, s8 // (op 21232) 
s_lshr_b32 s28, s15, 1 // (op 21232) 

s_add_u32 s15, s28, s8 // (op 21232) 


s_lshr_b32 s8, s15, s10 // (op 21232) 

s_and_b32 s15, s10, 2147483648 // (op 21232) 
s_cmp_eq_i32 s15, 0 // (op 21232) 
s_mov_b32 s28, scc // (op 21232) 

s_cmp_eq_i32 s28, 0 // (op 21232) 
s_cselect_b32 s15, s7, s8 // (op 21232) 


s_mul_i32 s8, s15, s14 // (op 21232) 

s_sub_u32 s15, s7, s8 // (op 21232) 

s_mul_i32 s8, s7, s18 // (op 21232) 


v_lshl_add_u32 v20, s15, 7, v3 // (op 21232) 


s_sub_u32 s7, s6, s8 // (op 21232) 


v_mul_lo_u32 v29, v20, s37 // (op 21232) most significant: low of low * high
v_mul_hi_u32 v3, v20, s36 // (op 21232) most significant: high of low * low
v_mul_lo_u32 v28, v20, s36 // (op 21232) least significant: low of low * low
v_add_u32 v29, v29, v3 // (op 21232) most significant: sum


v_lshl_add_u32 v3, s7, 7, v18 // (op 21232) 


v_mov_b32 v21, 0 // (op 21232) convert
v_mov_b32 v20, v3 // (op 21232) convert

v_add_co_u32 v22, s[18:19], v20, v28 // (op 21232) least significant half
v_addc_co_u32 v23, s[18:19], v21, v29, s[18:19] // (op 21232) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 21232) 


v_lshrrev_b64 v[20:21], 3, v[22:23] // (op 21232) 

s_mul_i32 s19, s36, s23 // (op 21234) most significant: low of low * high
s_mul_i32 s15, s37, s22 // (op 21234) most significant: low of high * low
s_mul_i32 s18, s36, s22 // (op 21234) least significant
s_mul_hi_u32 s6, s36, s22 // (op 21234) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 21234) most significant: sum
s_add_u32 s19, s19, s6 // (op 21234) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 2 // (op 21234) 

s_ashr_i64 s[18:19], s[28:29], 3 // (op 21234) 

s_mov_b32 s72, s38 // (op 21234) reg expression
s_mov_b32 s73, s39 // (op 21234) reg expression
s_mov_b32 s74, s18 // (op 21234) reg expression
s_mov_b32 s75, 131072 // (op 21234) call()

v_and_b32 v3, -4, v1 // (op 21627) 
v_sub_u32 v18, v1, v3 // (op 21627) 

v_lshlrev_b32 v3, 5, v18 // (op 21627) 

v_lshrrev_b32 v18, 2, v1 // (op 21627) 
v_lshl_add_u32 v22, v18, 7, v3 // (op 21627) 


v_lshlrev_b32 v3, 2, v22 // (op 21627) 

v_lshrrev_b32 v18, 3, v3 // (op 21627) 

s_nop 0
v_readfirstlane_b32 s15, v18 // (op 21627) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, -4, v1 // (op 21702) 
v_sub_u32 v18, v1, v3 // (op 21702) 

v_lshlrev_b32 v3, 5, v18 // (op 21702) 

v_lshrrev_b32 v18, 2, v1 // (op 21702) 
v_lshl_add_u32 v22, v18, 7, v3 // (op 21702) 


v_lshlrev_b32 v3, 2, v22 // (op 21702) 

v_lshrrev_b32 v18, 3, v3 // (op 21702) 

s_nop 0
v_readfirstlane_b32 s6, v18 // (op 21702) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v3, 63, v1 // (op 22454) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 22454) 
v_lshrrev_b32 v18, 4, v3 // (op 22454) 
s_and_b32 s7, s18, 4294967292 // (op 22454) 
v_and_b32 v22, 3, v18 // (op 22454) 

s_add_u32 s8, s0, s7 // (op 22454) 

v_and_b32 v18, 15, v1 // (op 22454) 
v_lshl_add_u32 v23, v22, 4, v18 // (op 22454) 


s_mul_i32 s7, s16, s50 // (op 22454) 
s_add_u32 s7, s7, s8 // (op 22454) 

v_lshrrev_b32 v18, 7, v1 // (op 22454) 
s_add_u32 s8, s7, s9 // (op 22454) 

v_lshl_add_u32 v22, v18, 6, v23 // (op 22454) 


s_mul_hi_u32 s7, s8, s13 // (op 22454) 
v_lshrrev_b32 v18, 6, v3 // (op 22454) 

s_sub_u32 s28, s8, s7 // (op 22454) 
v_lshlrev_b32 v3, 2, v18 // (op 22454) 

s_lshr_b32 s29, s28, 1 // (op 22454) 

s_add_u32 s28, s29, s7 // (op 22454) 


s_lshr_b32 s7, s28, s11 // (op 22454) 

s_and_b32 s28, s11, 2147483648 // (op 22454) 
s_cmp_eq_i32 s28, 0 // (op 22454) 
s_mov_b32 s29, scc // (op 22454) 

s_cmp_eq_i32 s29, 0 // (op 22454) 
s_cselect_b32 s28, s8, s7 // (op 22454) 


s_mul_hi_u32 s7, s28, s12 // (op 22454) 
s_sub_u32 s29, s28, s7 // (op 22454) 
s_lshr_b32 s48, s29, 1 // (op 22454) 

s_add_u32 s29, s48, s7 // (op 22454) 


s_lshr_b32 s7, s29, s10 // (op 22454) 

s_and_b32 s29, s10, 2147483648 // (op 22454) 
s_cmp_eq_i32 s29, 0 // (op 22454) 
s_mov_b32 s48, scc // (op 22454) 

s_cmp_eq_i32 s48, 0 // (op 22454) 
s_cselect_b32 s29, s28, s7 // (op 22454) 


s_mul_i32 s7, s28, s18 // (op 22454) 


v_lshl_add_u32 v18, s29, 7, v22 // (op 22454) 


s_sub_u32 s18, s8, s7 // (op 22454) 


v_lshl_add_u32 v22, s18, 2, v3 // (op 22454) 


v_mul_lo_u32 v3, v18, s40 // (op 22454) 
v_add_u32 v3, v3, v22 // (op 22454) 


s_mul_i32 s19, s40, s35 // (op 22456) most significant: low of low * high
s_mul_i32 s7, s41, s34 // (op 22456) most significant: low of high * low
s_mul_i32 s18, s40, s34 // (op 22456) least significant
s_mul_hi_u32 s8, s40, s34 // (op 22456) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 22456) most significant: sum
s_add_u32 s19, s19, s8 // (op 22456) most significant: sum


s_mov_b32 s52, s42 // (op 22456) reg expression
s_mov_b32 s53, s43 // (op 22456) reg expression
s_mov_b32 s54, s18 // (op 22456) reg expression
s_mov_b32 s55, 131072 // (op 22456) call()

v_and_b32 v18, 63, v1 // (op 22481) 
s_ashr_i64 s[18:19], s[2:3], 7 // (op 22481) 
v_lshrrev_b32 v22, 4, v18 // (op 22481) 
s_and_b32 s7, s18, 4294967292 // (op 22481) 
v_and_b32 v23, 3, v22 // (op 22481) 

s_add_u32 s8, s0, s7 // (op 22481) 

v_and_b32 v22, 15, v1 // (op 22481) 
v_lshl_add_u32 v24, v23, 4, v22 // (op 22481) 


s_mul_i32 s7, s16, s50 // (op 22481) 
s_add_u32 s7, s7, s8 // (op 22481) 

v_lshrrev_b32 v22, 6, v1 // (op 22481) 
s_add_u32 s8, s7, s9 // (op 22481) 

v_and_b32 v23, 1, v22 // (op 22481) 

s_mul_hi_u32 s7, s8, s13 // (op 22481) 
v_lshl_add_u32 v22, v23, 6, v24 // (op 22481) 


s_sub_u32 s28, s8, s7 // (op 22481) 
v_lshrrev_b32 v23, 6, v18 // (op 22481) 

s_lshr_b32 s29, s28, 1 // (op 22481) 

v_lshlrev_b32 v18, 2, v23 // (op 22481) 

s_add_u32 s28, s29, s7 // (op 22481) 


s_lshr_b32 s7, s28, s11 // (op 22481) 

s_and_b32 s28, s11, 2147483648 // (op 22481) 
s_cmp_eq_i32 s28, 0 // (op 22481) 
s_mov_b32 s29, scc // (op 22481) 

s_cmp_eq_i32 s29, 0 // (op 22481) 
s_cselect_b32 s28, s8, s7 // (op 22481) 


s_mul_hi_u32 s7, s28, s12 // (op 22481) 
s_sub_u32 s29, s28, s7 // (op 22481) 
s_lshr_b32 s48, s29, 1 // (op 22481) 

s_add_u32 s29, s48, s7 // (op 22481) 


s_lshr_b32 s7, s29, s10 // (op 22481) 

s_and_b32 s29, s10, 2147483648 // (op 22481) 
s_cmp_eq_i32 s29, 0 // (op 22481) 
s_mov_b32 s48, scc // (op 22481) 

s_cmp_eq_i32 s48, 0 // (op 22481) 
s_cselect_b32 s29, s28, s7 // (op 22481) 


s_mul_i32 s7, s29, s14 // (op 22481) 

s_sub_u32 s29, s28, s7 // (op 22481) 

v_lshl_add_u32 v23, s29, 7, v22 // (op 22481) 


s_mul_i32 s7, s28, s18 // (op 22481) 


v_mul_lo_u32 v22, v23, s44 // (op 22481) 

s_sub_u32 s18, s8, s7 // (op 22481) 


v_lshl_add_u32 v23, s18, 2, v18 // (op 22481) 


v_add_u32 v18, v23, v22 // (op 22481) 


s_mul_i32 s19, s44, s23 // (op 22483) most significant: low of low * high
s_mul_i32 s7, s45, s22 // (op 22483) most significant: low of high * low
s_mul_i32 s18, s44, s22 // (op 22483) least significant
s_mul_hi_u32 s8, s44, s22 // (op 22483) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 22483) most significant: sum
s_add_u32 s19, s19, s8 // (op 22483) most significant: sum


s_mov_b32 s64, s46 // (op 22483) reg expression
s_mov_b32 s65, s47 // (op 22483) reg expression
s_mov_b32 s66, s18 // (op 22483) reg expression
s_mov_b32 s67, 131072 // (op 22483) call()

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3555) 
s_and_b32 s7, s18, 4294967292 // (op 3555) 

s_ashr_i64 s[18:19], s[2:3], 7 // (op 3554) 
s_mov_b32 s8, s18 // (op 3554) convert

s_cmp_lt_u32 s7, s8 // (op 3554) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_3554 // (op 3554) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_3554 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_3554:
s_barrier  // (op 23054) 
s_add_u32 s7, s7, 1 // (op 3556) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 20927) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 20927) 

v_lshrrev_b64 v[22:23], 3, s[28:29] // (op 20927) 


v_mov_b64 v[24:25], v[16:17] // (op 20927) 
v_mov_b64 v[26:27], v[24:25] // (op 20927) 
s_add_u32 s8, 8192, s15 // (op 20927) 
s_mov_b32 m0, s8 // (op 20927) reg expression
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 20927) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v22 // (op 20927) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 20927) most significant half

s_add_u32 m0, m0, 4096 // (op 20927) 
buffer_load_dwordx4 v26, s[68:71], 0 offen offset: 0 lds // (op 20927) Load value direct to lds




v_add_co_u32 v16, s[18:19], 64, v16 // (op 21056) least significant half
v_addc_co_u32 v17, s[18:19], 0, v17, s[18:19] // (op 21056) most significant half

s_barrier  // (op 23074) 

s_lshl_b64 s[18:19], s[36:37], 6 // (op 20933) 
s_lshl_b64 s[28:29], s[18:19], 2 // (op 20933) 

v_lshrrev_b64 v[22:23], 3, s[28:29] // (op 20933) 


v_mov_b64 v[24:25], v[20:21] // (op 20933) 
v_mov_b64 v[26:27], v[24:25] // (op 20933) 
s_mov_b32 m0, s6 // (op 20933) reg expression
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 20933) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v22 // (op 20933) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 20933) most significant half

s_add_u32 m0, m0, 4096 // (op 20933) 
buffer_load_dwordx4 v26, s[72:75], 0 offen offset: 0 lds // (op 20933) Load value direct to lds



v_add_co_u32 v20, s[18:19], 64, v20 // (op 21224) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 21224) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 23061) 



ds_read_b128 v[22:25], v5 offset:8192 // (op 426) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 23080) 



ds_read_b128 v[26:29], v6 // (op 428) Load local data 


v_mov_b32 v30, v3 // (op 430) 
v_mov_b32 v31, v30 // (op 430) reg expression

buffer_load_dwordx4 v[32:35], v31, s[52:55], 0 offen // (op 430) Load value


v_add_u32 v3, 4, v3 // (op 22444) 


v_mov_b32 v30, v18 // (op 432) 
v_mov_b32 v31, v30 // (op 432) reg expression

buffer_load_dwordx4 v[36:39], v31, s[64:67], 0 offen // (op 432) Load value


v_add_u32 v18, 4, v18 // (op 22471) 
s_waitcnt vmcnt(1)
v_bfe_u32 v40, v32, 0, 8 // (op 19208) 



v_bfe_u32 v41, v32, 8, 8 // (op 19208) 
v_bfe_u32 v42, v32, 16, 8 // (op 19208) 
v_bfe_u32 v43, v32, 24, 8 // (op 19208) 
v_permlane16_swap_b32 v40, v41 // (op 19208) 
ds_read_b128 v[44:47], v6 offset:1024 // (op 663) Load local data 
v_permlane16_swap_b32 v42, v43 // (op 19208) 



s_nop 1
v_permlane32_swap_b32 v40, v42 // (op 19208) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v41, v43 // (op 19208) 
ds_read_b128 v[48:51], v6 offset:2048 // (op 889) Load local data 




ds_read_b128 v[30:33], v6 offset:3072 // (op 1115) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v52, v36, 0, 8 // (op 19238) 



v_bfe_u32 v53, v36, 8, 8 // (op 19238) 
v_bfe_u32 v54, v36, 16, 8 // (op 19238) 
v_bfe_u32 v55, v36, 24, 8 // (op 19238) 
v_permlane16_swap_b32 v52, v53 // (op 19238) 
s_nop 0
v_permlane16_swap_b32 v54, v55 // (op 19238) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v52, v54 // (op 19238) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v53, v55 // (op 19238) 
ds_read_b128 v[56:59], v5 offset:9216 // (op 1475) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 433) 
ds_read_b128 v[34:37], v5 offset:10240 // (op 2477) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[44:47], a[4:7], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 666) 
ds_read_b128 v[60:63], v5 offset:11264 // (op 3479) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[48:51], a[8:11], v40, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 892) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:25], v[30:33], a[12:15], v40, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1118) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[26:29], a[16:19], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1479) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v41, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1533) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[30:33], a[28:31], v41, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1560) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[44:47], a[20:23], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1506) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[34:37], v[26:29], a[32:35], v42, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2481) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[34:37], v[48:51], a[40:43], v42, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2535) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[34:37], v[30:33], a[44:47], v42, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2562) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[34:37], v[44:47], a[36:39], v42, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2508) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:63], v[26:29], a[48:51], v43, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3483) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:63], v[48:51], a[56:59], v43, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3537) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[60:63], v[30:33], a[60:63], v43, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3564) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:63], v[44:47], a[52:55], v43, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3510) 




s_ashr_i64 s[18:19], s[2:3], 7 // (op 3554) 
s_mov_b32 s8, s18 // (op 3554) convert

s_cmp_lt_u32 s7, s8 // (op 3554) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_3554 // (op 3554) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_3554 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_3554:









s_barrier  // (op 23309) 


v_cmp_eq_f32 vcc, 0.00000, v0 // (op 203) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_203 // (op 203) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_203
v_accvgpr_read v20, a0 // (op 211) 
v_accvgpr_read v21, a1 // (op 211) 
v_accvgpr_read v22, a2 // (op 211) 
v_accvgpr_read v23, a3 // (op 211) 
v_mul_f32 v24, v19, v20 // (op 211) 
v_mul_f32 v25, v19, v21 // (op 211) 
v_mul_f32 v26, v19, v22 // (op 211) 
v_mul_f32 v27, v19, v23 // (op 211) 



v_mov_b32 v3, v14 // (op 213) reg expression

buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 213) Store value


v_accvgpr_read v20, a4 // (op 4960) 
v_accvgpr_read v21, a5 // (op 4960) 
v_accvgpr_read v22, a6 // (op 4960) 
v_accvgpr_read v23, a7 // (op 4960) 
v_mul_f32 v24, v19, v20 // (op 4960) 
v_mul_f32 v25, v19, v21 // (op 4960) 
v_mul_f32 v26, v19, v22 // (op 4960) 
v_mul_f32 v27, v19, v23 // (op 4960) 



s_lshl_b32 s15, s24, 6 // (op 4962) 
v_add_u32 v3, v14, s15 // (op 4962) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 4962) Store value


v_accvgpr_read v20, a8 // (op 4972) 
v_accvgpr_read v21, a9 // (op 4972) 
v_accvgpr_read v22, a10 // (op 4972) 
v_accvgpr_read v23, a11 // (op 4972) 
v_mul_f32 v24, v19, v20 // (op 4972) 
v_mul_f32 v25, v19, v21 // (op 4972) 
v_mul_f32 v26, v19, v22 // (op 4972) 
v_mul_f32 v27, v19, v23 // (op 4972) 



s_lshl_b32 s15, s24, 7 // (op 4974) 
v_add_u32 v3, v14, s15 // (op 4974) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 4974) Store value


v_accvgpr_read v20, a12 // (op 4984) 
v_accvgpr_read v21, a13 // (op 4984) 
v_accvgpr_read v22, a14 // (op 4984) 
v_accvgpr_read v23, a15 // (op 4984) 
v_mul_f32 v24, v19, v20 // (op 4984) 
v_mul_f32 v25, v19, v21 // (op 4984) 
v_mul_f32 v26, v19, v22 // (op 4984) 
v_mul_f32 v27, v19, v23 // (op 4984) 



s_lshl_b32 s15, s24, 6 // (op 4986) 
s_mul_i32 s6, 3, s15 // (op 4986) 

v_add_u32 v3, v14, s6 // (op 4986) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 4986) Store value


v_accvgpr_read v20, a16 // (op 5018) 
v_accvgpr_read v21, a17 // (op 5018) 
v_accvgpr_read v22, a18 // (op 5018) 
v_accvgpr_read v23, a19 // (op 5018) 
v_mul_f32 v24, v19, v20 // (op 5018) 
v_mul_f32 v25, v19, v21 // (op 5018) 
v_mul_f32 v26, v19, v22 // (op 5018) 
v_mul_f32 v27, v19, v23 // (op 5018) 



v_add_u32 v3, 64, v14 // (op 5021) 

buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5021) Store value


v_accvgpr_read v20, a20 // (op 5031) 
v_accvgpr_read v21, a21 // (op 5031) 
v_accvgpr_read v22, a22 // (op 5031) 
v_accvgpr_read v23, a23 // (op 5031) 
v_mul_f32 v24, v19, v20 // (op 5031) 
v_mul_f32 v25, v19, v21 // (op 5031) 
v_mul_f32 v26, v19, v22 // (op 5031) 
v_mul_f32 v27, v19, v23 // (op 5031) 



s_lshl_b32 s15, s24, 6 // (op 5023) 
s_add_u32 s15, s15, 64 // (op 5023) 
v_add_u32 v3, v14, s15 // (op 5023) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5023) Store value


v_accvgpr_read v20, a24 // (op 5036) 
v_accvgpr_read v21, a25 // (op 5036) 
v_accvgpr_read v22, a26 // (op 5036) 
v_accvgpr_read v23, a27 // (op 5036) 
v_mul_f32 v24, v19, v20 // (op 5036) 
v_mul_f32 v25, v19, v21 // (op 5036) 
v_mul_f32 v26, v19, v22 // (op 5036) 
v_mul_f32 v27, v19, v23 // (op 5036) 



s_lshl_b32 s15, s24, 7 // (op 5025) 
s_add_u32 s15, s15, 64 // (op 5025) 
v_add_u32 v3, v14, s15 // (op 5025) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5025) Store value


v_accvgpr_read v20, a28 // (op 5041) 
v_accvgpr_read v21, a29 // (op 5041) 
v_accvgpr_read v22, a30 // (op 5041) 
v_accvgpr_read v23, a31 // (op 5041) 
v_mul_f32 v24, v19, v20 // (op 5041) 
v_mul_f32 v25, v19, v21 // (op 5041) 
v_mul_f32 v26, v19, v22 // (op 5041) 
v_mul_f32 v27, v19, v23 // (op 5041) 



s_lshl_b32 s15, s24, 6 // (op 5027) 
s_mul_i32 s6, 3, s15 // (op 5027) 
s_add_u32 s6, s6, 64 // (op 5027) 

v_add_u32 v3, v14, s6 // (op 5027) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5027) Store value


v_accvgpr_read v20, a32 // (op 5086) 
v_accvgpr_read v21, a33 // (op 5086) 
v_accvgpr_read v22, a34 // (op 5086) 
v_accvgpr_read v23, a35 // (op 5086) 
v_mul_f32 v24, v19, v20 // (op 5086) 
v_mul_f32 v25, v19, v21 // (op 5086) 
v_mul_f32 v26, v19, v22 // (op 5086) 
v_mul_f32 v27, v19, v23 // (op 5086) 



v_add_u32 v3, 128, v14 // (op 5089) 

buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5089) Store value


v_accvgpr_read v20, a36 // (op 5099) 
v_accvgpr_read v21, a37 // (op 5099) 
v_accvgpr_read v22, a38 // (op 5099) 
v_accvgpr_read v23, a39 // (op 5099) 
v_mul_f32 v24, v19, v20 // (op 5099) 
v_mul_f32 v25, v19, v21 // (op 5099) 
v_mul_f32 v26, v19, v22 // (op 5099) 
v_mul_f32 v27, v19, v23 // (op 5099) 



s_lshl_b32 s15, s24, 6 // (op 5091) 
s_add_u32 s15, s15, 128 // (op 5091) 
v_add_u32 v3, v14, s15 // (op 5091) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5091) Store value


v_accvgpr_read v20, a40 // (op 5104) 
v_accvgpr_read v21, a41 // (op 5104) 
v_accvgpr_read v22, a42 // (op 5104) 
v_accvgpr_read v23, a43 // (op 5104) 
v_mul_f32 v24, v19, v20 // (op 5104) 
v_mul_f32 v25, v19, v21 // (op 5104) 
v_mul_f32 v26, v19, v22 // (op 5104) 
v_mul_f32 v27, v19, v23 // (op 5104) 



s_lshl_b32 s15, s24, 7 // (op 5093) 
s_add_u32 s15, s15, 128 // (op 5093) 
v_add_u32 v3, v14, s15 // (op 5093) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5093) Store value


v_accvgpr_read v20, a44 // (op 5109) 
v_accvgpr_read v21, a45 // (op 5109) 
v_accvgpr_read v22, a46 // (op 5109) 
v_accvgpr_read v23, a47 // (op 5109) 
v_mul_f32 v24, v19, v20 // (op 5109) 
v_mul_f32 v25, v19, v21 // (op 5109) 
v_mul_f32 v26, v19, v22 // (op 5109) 
v_mul_f32 v27, v19, v23 // (op 5109) 



s_lshl_b32 s15, s24, 6 // (op 5095) 
s_mul_i32 s6, 3, s15 // (op 5095) 
s_add_u32 s6, s6, 128 // (op 5095) 

v_add_u32 v3, v14, s6 // (op 5095) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5095) Store value


v_accvgpr_read v20, a48 // (op 5154) 
v_accvgpr_read v21, a49 // (op 5154) 
v_accvgpr_read v22, a50 // (op 5154) 
v_accvgpr_read v23, a51 // (op 5154) 
v_mul_f32 v24, v19, v20 // (op 5154) 
v_mul_f32 v25, v19, v21 // (op 5154) 
v_mul_f32 v26, v19, v22 // (op 5154) 
v_mul_f32 v27, v19, v23 // (op 5154) 



v_add_u32 v3, 192, v14 // (op 5157) 

buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5157) Store value


v_accvgpr_read v20, a52 // (op 5167) 
v_accvgpr_read v21, a53 // (op 5167) 
v_accvgpr_read v22, a54 // (op 5167) 
v_accvgpr_read v23, a55 // (op 5167) 
v_mul_f32 v24, v19, v20 // (op 5167) 
v_mul_f32 v25, v19, v21 // (op 5167) 
v_mul_f32 v26, v19, v22 // (op 5167) 
v_mul_f32 v27, v19, v23 // (op 5167) 



s_lshl_b32 s15, s24, 6 // (op 5159) 
s_add_u32 s15, s15, 192 // (op 5159) 
v_add_u32 v3, v14, s15 // (op 5159) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5159) Store value


v_accvgpr_read v20, a56 // (op 5172) 
v_accvgpr_read v21, a57 // (op 5172) 
v_accvgpr_read v22, a58 // (op 5172) 
v_accvgpr_read v23, a59 // (op 5172) 
v_mul_f32 v24, v19, v20 // (op 5172) 
v_mul_f32 v25, v19, v21 // (op 5172) 
v_mul_f32 v26, v19, v22 // (op 5172) 
v_mul_f32 v27, v19, v23 // (op 5172) 



s_lshl_b32 s15, s24, 7 // (op 5161) 
s_add_u32 s15, s15, 192 // (op 5161) 
v_add_u32 v3, v14, s15 // (op 5161) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5161) Store value


v_accvgpr_read v20, a60 // (op 5177) 
v_accvgpr_read v21, a61 // (op 5177) 
v_accvgpr_read v22, a62 // (op 5177) 
v_accvgpr_read v23, a63 // (op 5177) 
v_mul_f32 v24, v19, v20 // (op 5177) 
v_mul_f32 v25, v19, v21 // (op 5177) 
v_mul_f32 v26, v19, v22 // (op 5177) 
v_mul_f32 v27, v19, v23 // (op 5177) 



s_lshl_b32 s15, s24, 6 // (op 5163) 
s_mul_i32 s6, 3, s15 // (op 5163) 
s_add_u32 s6, s6, 192 // (op 5163) 

v_add_u32 v3, v14, s6 // (op 5163) 


buffer_store_dwordx4 v[24:27], v3, s[60:63], 0 offen // (op 5163) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_203 // (op 203) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_203
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_203:


v_add_u32 v3, 192, v13 // (op 5521) 

buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5521) Load value

v_accvgpr_read v24, a48 // (op 5509) 
v_accvgpr_read v25, a49 // (op 5509) 
v_accvgpr_read v26, a50 // (op 5509) 
v_accvgpr_read v27, a51 // (op 5509) 
v_mul_f32 v28, v19, v24 // (op 5509) 
v_mul_f32 v29, v19, v25 // (op 5509) 
v_mul_f32 v30, v19, v26 // (op 5509) 
v_mul_f32 v31, v19, v27 // (op 5509) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5520) 
v_mul_f32 v25, v0, v21 // (op 5520) 
v_mul_f32 v26, v0, v22 // (op 5520) 
v_mul_f32 v27, v0, v23 // (op 5520) 

v_add_f32 v20, v24, v28 // (op 5510) 
v_add_f32 v21, v25, v29 // (op 5510) 
v_add_f32 v22, v26, v30 // (op 5510) 
v_add_f32 v23, v27, v31 // (op 5510) 




v_add_u32 v3, 192, v14 // (op 5512) 

buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5512) Store value




s_lshl_b32 s15, s4, 6 // (op 5525) 
s_add_u32 s15, s15, 192 // (op 5525) 
v_add_u32 v3, v13, s15 // (op 5525) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5525) Load value

v_accvgpr_read v24, a52 // (op 5535) 
v_accvgpr_read v25, a53 // (op 5535) 
v_accvgpr_read v26, a54 // (op 5535) 
v_accvgpr_read v27, a55 // (op 5535) 
v_mul_f32 v28, v19, v24 // (op 5535) 
v_mul_f32 v29, v19, v25 // (op 5535) 
v_mul_f32 v30, v19, v26 // (op 5535) 
v_mul_f32 v31, v19, v27 // (op 5535) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5523) 
v_mul_f32 v25, v0, v21 // (op 5523) 
v_mul_f32 v26, v0, v22 // (op 5523) 
v_mul_f32 v27, v0, v23 // (op 5523) 

v_add_f32 v20, v24, v28 // (op 5524) 
v_add_f32 v21, v25, v29 // (op 5524) 
v_add_f32 v22, v26, v30 // (op 5524) 
v_add_f32 v23, v27, v31 // (op 5524) 




s_lshl_b32 s15, s24, 6 // (op 5514) 
s_add_u32 s15, s15, 192 // (op 5514) 
v_add_u32 v3, v14, s15 // (op 5514) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5514) Store value




s_lshl_b32 s15, s4, 7 // (op 5529) 
s_add_u32 s15, s15, 192 // (op 5529) 
v_add_u32 v3, v13, s15 // (op 5529) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5529) Load value

v_accvgpr_read v24, a56 // (op 5537) 
v_accvgpr_read v25, a57 // (op 5537) 
v_accvgpr_read v26, a58 // (op 5537) 
v_accvgpr_read v27, a59 // (op 5537) 
v_mul_f32 v28, v19, v24 // (op 5537) 
v_mul_f32 v29, v19, v25 // (op 5537) 
v_mul_f32 v30, v19, v26 // (op 5537) 
v_mul_f32 v31, v19, v27 // (op 5537) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5527) 
v_mul_f32 v25, v0, v21 // (op 5527) 
v_mul_f32 v26, v0, v22 // (op 5527) 
v_mul_f32 v27, v0, v23 // (op 5527) 

v_add_f32 v20, v24, v28 // (op 5528) 
v_add_f32 v21, v25, v29 // (op 5528) 
v_add_f32 v22, v26, v30 // (op 5528) 
v_add_f32 v23, v27, v31 // (op 5528) 




s_lshl_b32 s15, s24, 7 // (op 5516) 
s_add_u32 s15, s15, 192 // (op 5516) 
v_add_u32 v3, v14, s15 // (op 5516) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5516) Store value




s_lshl_b32 s15, s4, 6 // (op 5533) 
s_mul_i32 s6, 3, s15 // (op 5533) 
s_add_u32 s6, s6, 192 // (op 5533) 

v_add_u32 v3, v13, s6 // (op 5533) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5533) Load value

v_accvgpr_read v24, a60 // (op 5539) 
v_accvgpr_read v25, a61 // (op 5539) 
v_accvgpr_read v26, a62 // (op 5539) 
v_accvgpr_read v27, a63 // (op 5539) 
v_mul_f32 v28, v19, v24 // (op 5539) 
v_mul_f32 v29, v19, v25 // (op 5539) 
v_mul_f32 v30, v19, v26 // (op 5539) 
v_mul_f32 v31, v19, v27 // (op 5539) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5531) 
v_mul_f32 v25, v0, v21 // (op 5531) 
v_mul_f32 v26, v0, v22 // (op 5531) 
v_mul_f32 v27, v0, v23 // (op 5531) 

v_add_f32 v20, v24, v28 // (op 5532) 
v_add_f32 v21, v25, v29 // (op 5532) 
v_add_f32 v22, v26, v30 // (op 5532) 
v_add_f32 v23, v27, v31 // (op 5532) 




s_lshl_b32 s15, s24, 6 // (op 5518) 
s_mul_i32 s6, 3, s15 // (op 5518) 
s_add_u32 s6, s6, 192 // (op 5518) 

v_add_u32 v3, v14, s6 // (op 5518) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5518) Store value




v_add_u32 v3, 128, v13 // (op 5442) 

s_nop 0
buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5442) Load value

v_accvgpr_read v24, a32 // (op 5430) 
v_accvgpr_read v25, a33 // (op 5430) 
v_accvgpr_read v26, a34 // (op 5430) 
v_accvgpr_read v27, a35 // (op 5430) 
v_mul_f32 v28, v19, v24 // (op 5430) 
v_mul_f32 v29, v19, v25 // (op 5430) 
v_mul_f32 v30, v19, v26 // (op 5430) 
v_mul_f32 v31, v19, v27 // (op 5430) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5441) 
v_mul_f32 v25, v0, v21 // (op 5441) 
v_mul_f32 v26, v0, v22 // (op 5441) 
v_mul_f32 v27, v0, v23 // (op 5441) 

v_add_f32 v20, v24, v28 // (op 5431) 
v_add_f32 v21, v25, v29 // (op 5431) 
v_add_f32 v22, v26, v30 // (op 5431) 
v_add_f32 v23, v27, v31 // (op 5431) 




v_add_u32 v3, 128, v14 // (op 5433) 

buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5433) Store value




s_lshl_b32 s15, s4, 6 // (op 5446) 
s_add_u32 s15, s15, 128 // (op 5446) 
v_add_u32 v3, v13, s15 // (op 5446) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5446) Load value

v_accvgpr_read v24, a36 // (op 5456) 
v_accvgpr_read v25, a37 // (op 5456) 
v_accvgpr_read v26, a38 // (op 5456) 
v_accvgpr_read v27, a39 // (op 5456) 
v_mul_f32 v28, v19, v24 // (op 5456) 
v_mul_f32 v29, v19, v25 // (op 5456) 
v_mul_f32 v30, v19, v26 // (op 5456) 
v_mul_f32 v31, v19, v27 // (op 5456) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5444) 
v_mul_f32 v25, v0, v21 // (op 5444) 
v_mul_f32 v26, v0, v22 // (op 5444) 
v_mul_f32 v27, v0, v23 // (op 5444) 

v_add_f32 v20, v24, v28 // (op 5445) 
v_add_f32 v21, v25, v29 // (op 5445) 
v_add_f32 v22, v26, v30 // (op 5445) 
v_add_f32 v23, v27, v31 // (op 5445) 




s_lshl_b32 s15, s24, 6 // (op 5435) 
s_add_u32 s15, s15, 128 // (op 5435) 
v_add_u32 v3, v14, s15 // (op 5435) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5435) Store value




s_lshl_b32 s15, s4, 7 // (op 5450) 
s_add_u32 s15, s15, 128 // (op 5450) 
v_add_u32 v3, v13, s15 // (op 5450) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5450) Load value

v_accvgpr_read v24, a40 // (op 5458) 
v_accvgpr_read v25, a41 // (op 5458) 
v_accvgpr_read v26, a42 // (op 5458) 
v_accvgpr_read v27, a43 // (op 5458) 
v_mul_f32 v28, v19, v24 // (op 5458) 
v_mul_f32 v29, v19, v25 // (op 5458) 
v_mul_f32 v30, v19, v26 // (op 5458) 
v_mul_f32 v31, v19, v27 // (op 5458) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5448) 
v_mul_f32 v25, v0, v21 // (op 5448) 
v_mul_f32 v26, v0, v22 // (op 5448) 
v_mul_f32 v27, v0, v23 // (op 5448) 

v_add_f32 v20, v24, v28 // (op 5449) 
v_add_f32 v21, v25, v29 // (op 5449) 
v_add_f32 v22, v26, v30 // (op 5449) 
v_add_f32 v23, v27, v31 // (op 5449) 




s_lshl_b32 s15, s24, 7 // (op 5437) 
s_add_u32 s15, s15, 128 // (op 5437) 
v_add_u32 v3, v14, s15 // (op 5437) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5437) Store value




s_lshl_b32 s15, s4, 6 // (op 5454) 
s_mul_i32 s6, 3, s15 // (op 5454) 
s_add_u32 s6, s6, 128 // (op 5454) 

v_add_u32 v3, v13, s6 // (op 5454) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5454) Load value

v_accvgpr_read v24, a44 // (op 5460) 
v_accvgpr_read v25, a45 // (op 5460) 
v_accvgpr_read v26, a46 // (op 5460) 
v_accvgpr_read v27, a47 // (op 5460) 
v_mul_f32 v28, v19, v24 // (op 5460) 
v_mul_f32 v29, v19, v25 // (op 5460) 
v_mul_f32 v30, v19, v26 // (op 5460) 
v_mul_f32 v31, v19, v27 // (op 5460) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5452) 
v_mul_f32 v25, v0, v21 // (op 5452) 
v_mul_f32 v26, v0, v22 // (op 5452) 
v_mul_f32 v27, v0, v23 // (op 5452) 

v_add_f32 v20, v24, v28 // (op 5453) 
v_add_f32 v21, v25, v29 // (op 5453) 
v_add_f32 v22, v26, v30 // (op 5453) 
v_add_f32 v23, v27, v31 // (op 5453) 




s_lshl_b32 s15, s24, 6 // (op 5439) 
s_mul_i32 s6, 3, s15 // (op 5439) 
s_add_u32 s6, s6, 128 // (op 5439) 

v_add_u32 v3, v14, s6 // (op 5439) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5439) Store value




v_add_u32 v3, 64, v13 // (op 5363) 

s_nop 0
buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5363) Load value

v_accvgpr_read v24, a16 // (op 5351) 
v_accvgpr_read v25, a17 // (op 5351) 
v_accvgpr_read v26, a18 // (op 5351) 
v_accvgpr_read v27, a19 // (op 5351) 
v_mul_f32 v28, v19, v24 // (op 5351) 
v_mul_f32 v29, v19, v25 // (op 5351) 
v_mul_f32 v30, v19, v26 // (op 5351) 
v_mul_f32 v31, v19, v27 // (op 5351) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5362) 
v_mul_f32 v25, v0, v21 // (op 5362) 
v_mul_f32 v26, v0, v22 // (op 5362) 
v_mul_f32 v27, v0, v23 // (op 5362) 

v_add_f32 v20, v24, v28 // (op 5352) 
v_add_f32 v21, v25, v29 // (op 5352) 
v_add_f32 v22, v26, v30 // (op 5352) 
v_add_f32 v23, v27, v31 // (op 5352) 




v_add_u32 v3, 64, v14 // (op 5354) 

buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5354) Store value




s_lshl_b32 s15, s4, 6 // (op 5367) 
s_add_u32 s15, s15, 64 // (op 5367) 
v_add_u32 v3, v13, s15 // (op 5367) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5367) Load value

v_accvgpr_read v24, a20 // (op 5377) 
v_accvgpr_read v25, a21 // (op 5377) 
v_accvgpr_read v26, a22 // (op 5377) 
v_accvgpr_read v27, a23 // (op 5377) 
v_mul_f32 v28, v19, v24 // (op 5377) 
v_mul_f32 v29, v19, v25 // (op 5377) 
v_mul_f32 v30, v19, v26 // (op 5377) 
v_mul_f32 v31, v19, v27 // (op 5377) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5365) 
v_mul_f32 v25, v0, v21 // (op 5365) 
v_mul_f32 v26, v0, v22 // (op 5365) 
v_mul_f32 v27, v0, v23 // (op 5365) 

v_add_f32 v20, v24, v28 // (op 5366) 
v_add_f32 v21, v25, v29 // (op 5366) 
v_add_f32 v22, v26, v30 // (op 5366) 
v_add_f32 v23, v27, v31 // (op 5366) 




s_lshl_b32 s15, s24, 6 // (op 5356) 
s_add_u32 s15, s15, 64 // (op 5356) 
v_add_u32 v3, v14, s15 // (op 5356) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5356) Store value




s_lshl_b32 s15, s4, 7 // (op 5371) 
s_add_u32 s15, s15, 64 // (op 5371) 
v_add_u32 v3, v13, s15 // (op 5371) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5371) Load value

v_accvgpr_read v24, a24 // (op 5379) 
v_accvgpr_read v25, a25 // (op 5379) 
v_accvgpr_read v26, a26 // (op 5379) 
v_accvgpr_read v27, a27 // (op 5379) 
v_mul_f32 v28, v19, v24 // (op 5379) 
v_mul_f32 v29, v19, v25 // (op 5379) 
v_mul_f32 v30, v19, v26 // (op 5379) 
v_mul_f32 v31, v19, v27 // (op 5379) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5369) 
v_mul_f32 v25, v0, v21 // (op 5369) 
v_mul_f32 v26, v0, v22 // (op 5369) 
v_mul_f32 v27, v0, v23 // (op 5369) 

v_add_f32 v20, v24, v28 // (op 5370) 
v_add_f32 v21, v25, v29 // (op 5370) 
v_add_f32 v22, v26, v30 // (op 5370) 
v_add_f32 v23, v27, v31 // (op 5370) 




s_lshl_b32 s15, s24, 7 // (op 5358) 
s_add_u32 s15, s15, 64 // (op 5358) 
v_add_u32 v3, v14, s15 // (op 5358) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5358) Store value




s_lshl_b32 s15, s4, 6 // (op 5375) 
s_mul_i32 s6, 3, s15 // (op 5375) 
s_add_u32 s6, s6, 64 // (op 5375) 

v_add_u32 v3, v13, s6 // (op 5375) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5375) Load value

v_accvgpr_read v24, a28 // (op 5381) 
v_accvgpr_read v25, a29 // (op 5381) 
v_accvgpr_read v26, a30 // (op 5381) 
v_accvgpr_read v27, a31 // (op 5381) 
v_mul_f32 v28, v19, v24 // (op 5381) 
v_mul_f32 v29, v19, v25 // (op 5381) 
v_mul_f32 v30, v19, v26 // (op 5381) 
v_mul_f32 v31, v19, v27 // (op 5381) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5373) 
v_mul_f32 v25, v0, v21 // (op 5373) 
v_mul_f32 v26, v0, v22 // (op 5373) 
v_mul_f32 v27, v0, v23 // (op 5373) 

v_add_f32 v20, v24, v28 // (op 5374) 
v_add_f32 v21, v25, v29 // (op 5374) 
v_add_f32 v22, v26, v30 // (op 5374) 
v_add_f32 v23, v27, v31 // (op 5374) 




s_lshl_b32 s15, s24, 6 // (op 5360) 
s_mul_i32 s6, 3, s15 // (op 5360) 
s_add_u32 s6, s6, 64 // (op 5360) 

v_add_u32 v3, v14, s6 // (op 5360) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5360) Store value




v_mov_b32 v3, v13 // (op 226) reg expression

s_nop 0
buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 226) Load value

v_accvgpr_read v24, a0 // (op 223) 
v_accvgpr_read v25, a1 // (op 223) 
v_accvgpr_read v26, a2 // (op 223) 
v_accvgpr_read v27, a3 // (op 223) 
v_mul_f32 v28, v19, v24 // (op 223) 
v_mul_f32 v29, v19, v25 // (op 223) 
v_mul_f32 v30, v19, v26 // (op 223) 
v_mul_f32 v31, v19, v27 // (op 223) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 227) 
v_mul_f32 v25, v0, v21 // (op 227) 
v_mul_f32 v26, v0, v22 // (op 227) 
v_mul_f32 v27, v0, v23 // (op 227) 

v_add_f32 v20, v24, v28 // (op 224) 
v_add_f32 v21, v25, v29 // (op 224) 
v_add_f32 v22, v26, v30 // (op 224) 
v_add_f32 v23, v27, v31 // (op 224) 




v_mov_b32 v3, v14 // (op 225) reg expression

buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 225) Store value




s_lshl_b32 s15, s4, 6 // (op 5277) 
v_add_u32 v3, v13, s15 // (op 5277) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5277) Load value

v_accvgpr_read v24, a4 // (op 5274) 
v_accvgpr_read v25, a5 // (op 5274) 
v_accvgpr_read v26, a6 // (op 5274) 
v_accvgpr_read v27, a7 // (op 5274) 
v_mul_f32 v28, v19, v24 // (op 5274) 
v_mul_f32 v29, v19, v25 // (op 5274) 
v_mul_f32 v30, v19, v26 // (op 5274) 
v_mul_f32 v31, v19, v27 // (op 5274) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5278) 
v_mul_f32 v25, v0, v21 // (op 5278) 
v_mul_f32 v26, v0, v22 // (op 5278) 
v_mul_f32 v27, v0, v23 // (op 5278) 

v_add_f32 v20, v24, v28 // (op 5275) 
v_add_f32 v21, v25, v29 // (op 5275) 
v_add_f32 v22, v26, v30 // (op 5275) 
v_add_f32 v23, v27, v31 // (op 5275) 




s_lshl_b32 s15, s24, 6 // (op 5276) 
v_add_u32 v3, v14, s15 // (op 5276) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5276) Store value




s_lshl_b32 s15, s4, 7 // (op 5289) 
v_add_u32 v3, v13, s15 // (op 5289) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5289) Load value

v_accvgpr_read v24, a8 // (op 5286) 
v_accvgpr_read v25, a9 // (op 5286) 
v_accvgpr_read v26, a10 // (op 5286) 
v_accvgpr_read v27, a11 // (op 5286) 
v_mul_f32 v28, v19, v24 // (op 5286) 
v_mul_f32 v29, v19, v25 // (op 5286) 
v_mul_f32 v30, v19, v26 // (op 5286) 
v_mul_f32 v31, v19, v27 // (op 5286) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5290) 
v_mul_f32 v25, v0, v21 // (op 5290) 
v_mul_f32 v26, v0, v22 // (op 5290) 
v_mul_f32 v27, v0, v23 // (op 5290) 

v_add_f32 v20, v24, v28 // (op 5287) 
v_add_f32 v21, v25, v29 // (op 5287) 
v_add_f32 v22, v26, v30 // (op 5287) 
v_add_f32 v23, v27, v31 // (op 5287) 




s_lshl_b32 s15, s24, 7 // (op 5288) 
v_add_u32 v3, v14, s15 // (op 5288) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5288) Store value




s_lshl_b32 s15, s4, 6 // (op 5301) 
s_mul_i32 s6, 3, s15 // (op 5301) 

v_add_u32 v3, v13, s6 // (op 5301) 


buffer_load_dwordx4 v[20:23], v3, s[56:59], 0 offen // (op 5301) Load value

v_accvgpr_read v24, a12 // (op 5298) 
v_accvgpr_read v25, a13 // (op 5298) 
v_accvgpr_read v26, a14 // (op 5298) 
v_accvgpr_read v27, a15 // (op 5298) 
v_mul_f32 v28, v19, v24 // (op 5298) 
v_mul_f32 v29, v19, v25 // (op 5298) 
v_mul_f32 v30, v19, v26 // (op 5298) 
v_mul_f32 v31, v19, v27 // (op 5298) 

s_waitcnt vmcnt(0)
v_mul_f32 v24, v0, v20 // (op 5302) 
v_mul_f32 v25, v0, v21 // (op 5302) 
v_mul_f32 v26, v0, v22 // (op 5302) 
v_mul_f32 v27, v0, v23 // (op 5302) 

v_add_f32 v20, v24, v28 // (op 5299) 
v_add_f32 v21, v25, v29 // (op 5299) 
v_add_f32 v22, v26, v30 // (op 5299) 
v_add_f32 v23, v27, v31 // (op 5299) 




s_lshl_b32 s15, s24, 6 // (op 5300) 
s_mul_i32 s6, 3, s15 // (op 5300) 

v_add_u32 v3, v14, s6 // (op 5300) 


buffer_store_dwordx4 v[20:23], v3, s[60:63], 0 offen // (op 5300) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_203:
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
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_410 // (op 410) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopTop_DPStreamTileLoop_410 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_ForLoopBottom_DPStreamTileLoop_410:























































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
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
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_128x128x128_UR_4.kd
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

