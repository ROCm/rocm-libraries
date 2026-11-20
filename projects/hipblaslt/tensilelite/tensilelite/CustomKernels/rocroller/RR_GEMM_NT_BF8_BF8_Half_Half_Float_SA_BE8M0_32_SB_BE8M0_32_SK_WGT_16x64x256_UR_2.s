// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2,@function
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 3276) 
v_and_b32 v3, 63, v1 // (op 3276) 
v_sub_u32 v4, v3, v0 // (op 3276) 

v_lshrrev_b32 v3, 1, v4 // (op 3276) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 3276) 


v_and_b32 v0, 1, v4 // (op 3276) 

v_lshlrev_b32 v3, 3, v0 // (op 3276) 

v_lshrrev_b32 v0, 7, v1 // (op 3276) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3276) 


v_add_u32 v0, v5, v4 // (op 3276) 


v_lshrrev_b32 v3, 7, v1 // (op 3417) 
v_and_b32 v4, 15, v1 // (op 3417) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 3417) 


v_and_b32 v3, 48, v1 // (op 3417) 
v_add_u32 v4, v3, v5 // (op 3417) 


v_and_b32 v3, 48, v1 // (op 3600) 
v_and_b32 v5, 63, v1 // (op 3600) 
v_sub_u32 v6, v5, v3 // (op 3600) 

v_lshrrev_b32 v5, 1, v6 // (op 3600) 
v_add_lshl_u32 v7, v3, v5, 4 // (op 3600) 


v_and_b32 v3, 1, v6 // (op 3600) 

v_lshlrev_b32 v5, 3, v3 // (op 3600) 

v_lshrrev_b32 v3, 7, v1 // (op 3600) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 3600) 


v_add_u32 v3, v7, v6 // (op 3600) 


v_lshrrev_b32 v5, 7, v1 // (op 3741) 
v_and_b32 v6, 15, v1 // (op 3741) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 3741) 


v_and_b32 v5, 48, v1 // (op 3741) 
v_add_u32 v6, v5, v7 // (op 3741) 


v_and_b32 v5, 48, v1 // (op 3911) 
v_and_b32 v7, 63, v1 // (op 3911) 
v_sub_u32 v8, v7, v5 // (op 3911) 

v_lshrrev_b32 v7, 1, v8 // (op 3911) 
v_add_lshl_u32 v9, v5, v7, 4 // (op 3911) 


v_and_b32 v5, 1, v8 // (op 3911) 

v_lshlrev_b32 v7, 3, v5 // (op 3911) 

v_lshrrev_b32 v5, 7, v1 // (op 3911) 
v_lshl_add_u32 v8, v5, 4, v7 // (op 3911) 


v_add_u32 v5, v9, v8 // (op 3911) 


v_lshrrev_b32 v7, 7, v1 // (op 3979) 
v_and_b32 v8, 15, v1 // (op 3979) 
v_lshl_add_u32 v9, v7, 4, v8 // (op 3979) 


v_and_b32 v7, 48, v1 // (op 3979) 
v_add_u32 v8, v7, v9 // (op 3979) 


v_and_b32 v7, 48, v1 // (op 4047) 
v_and_b32 v9, 63, v1 // (op 4047) 
v_sub_u32 v10, v9, v7 // (op 4047) 

v_lshrrev_b32 v9, 1, v10 // (op 4047) 
v_add_lshl_u32 v11, v7, v9, 4 // (op 4047) 


v_and_b32 v7, 1, v10 // (op 4047) 

v_lshlrev_b32 v9, 3, v7 // (op 4047) 

v_lshrrev_b32 v7, 7, v1 // (op 4047) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 4047) 


v_add_u32 v7, v11, v10 // (op 4047) 


v_lshrrev_b32 v9, 7, v1 // (op 4115) 
v_and_b32 v10, 15, v1 // (op 4115) 
v_lshl_add_u32 v11, v9, 4, v10 // (op 4115) 


v_and_b32 v9, 48, v1 // (op 4115) 
v_add_u32 v10, v9, v11 // (op 4115) 


v_and_b32 v9, 48, v1 // (op 3346) 
v_and_b32 v11, 63, v1 // (op 3346) 
v_sub_u32 v12, v11, v9 // (op 3346) 

v_lshrrev_b32 v11, 1, v12 // (op 3346) 
v_add_lshl_u32 v13, v9, v11, 6 // (op 3346) 


v_and_b32 v9, 1, v12 // (op 3346) 

v_lshlrev_b32 v11, 3, v9 // (op 3346) 

v_lshrrev_b32 v9, 6, v1 // (op 3346) 
v_and_b32 v12, 1, v9 // (op 3346) 

v_lshl_add_u32 v9, v12, 4, v11 // (op 3346) 


v_add_u32 v11, v13, v9 // (op 3346) 


v_lshrrev_b32 v9, 6, v1 // (op 3488) 
v_and_b32 v12, 1, v9 // (op 3488) 

v_and_b32 v9, 15, v1 // (op 3488) 
v_lshl_add_u32 v13, v12, 4, v9 // (op 3488) 


v_and_b32 v9, 63, v1 // (op 3488) 
v_lshrrev_b32 v12, 4, v9 // (op 3488) 

v_lshl_add_u32 v9, v12, 6, v13 // (op 3488) 


v_and_b32 v12, 48, v1 // (op 3670) 
v_and_b32 v13, 63, v1 // (op 3670) 
v_sub_u32 v14, v13, v12 // (op 3670) 

v_lshrrev_b32 v13, 1, v14 // (op 3670) 
v_add_lshl_u32 v15, v12, v13, 6 // (op 3670) 


v_and_b32 v12, 1, v14 // (op 3670) 

v_lshlrev_b32 v13, 3, v12 // (op 3670) 

v_lshrrev_b32 v12, 6, v1 // (op 3670) 
v_and_b32 v14, 1, v12 // (op 3670) 

v_lshl_add_u32 v12, v14, 4, v13 // (op 3670) 


v_add_u32 v13, v15, v12 // (op 3670) 


v_lshrrev_b32 v12, 6, v1 // (op 3812) 
v_and_b32 v14, 1, v12 // (op 3812) 

v_and_b32 v12, 15, v1 // (op 3812) 
v_lshl_add_u32 v15, v14, 4, v12 // (op 3812) 


v_and_b32 v12, 63, v1 // (op 3812) 
v_lshrrev_b32 v14, 4, v12 // (op 3812) 

v_lshl_add_u32 v12, v14, 6, v15 // (op 3812) 


v_and_b32 v14, 48, v1 // (op 3955) 
v_and_b32 v15, 63, v1 // (op 3955) 
v_sub_u32 v16, v15, v14 // (op 3955) 

v_lshrrev_b32 v15, 1, v16 // (op 3955) 
v_add_lshl_u32 v17, v14, v15, 6 // (op 3955) 


v_and_b32 v14, 1, v16 // (op 3955) 

v_lshlrev_b32 v15, 3, v14 // (op 3955) 

v_lshrrev_b32 v14, 6, v1 // (op 3955) 
v_and_b32 v16, 1, v14 // (op 3955) 

v_lshl_add_u32 v14, v16, 4, v15 // (op 3955) 


v_add_u32 v15, v17, v14 // (op 3955) 


v_lshrrev_b32 v14, 6, v1 // (op 4008) 
v_and_b32 v16, 1, v14 // (op 4008) 

v_and_b32 v14, 15, v1 // (op 4008) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 4008) 


v_and_b32 v14, 63, v1 // (op 4008) 
v_lshrrev_b32 v16, 4, v14 // (op 4008) 

v_lshl_add_u32 v14, v16, 6, v17 // (op 4008) 


v_and_b32 v16, 48, v1 // (op 4091) 
v_and_b32 v17, 63, v1 // (op 4091) 
v_sub_u32 v18, v17, v16 // (op 4091) 

v_lshrrev_b32 v17, 1, v18 // (op 4091) 
v_add_lshl_u32 v19, v16, v17, 6 // (op 4091) 


v_and_b32 v16, 1, v18 // (op 4091) 

v_lshlrev_b32 v17, 3, v16 // (op 4091) 

v_lshrrev_b32 v16, 6, v1 // (op 4091) 
v_and_b32 v18, 1, v16 // (op 4091) 

v_lshl_add_u32 v16, v18, 4, v17 // (op 4091) 


v_add_u32 v17, v19, v16 // (op 4091) 


v_lshrrev_b32 v16, 6, v1 // (op 4144) 
v_and_b32 v18, 1, v16 // (op 4144) 

v_and_b32 v16, 15, v1 // (op 4144) 
v_lshl_add_u32 v19, v18, 4, v16 // (op 4144) 


v_and_b32 v16, 63, v1 // (op 4144) 
v_lshrrev_b32 v18, 4, v16 // (op 4144) 

v_lshl_add_u32 v16, v18, 6, v19 // (op 4144) 


v_mov_b32 v18, s6 // (op 2) Move value
v_mov_b32 v19, s7 // (op 43) Move value


s_mov_b32 s0, 0 // (op 435) call()
s_mov_b32 s1, 0 // (op 436) call()
s_mov_b32 s6, 0 // (op 437) call()
s_mov_b32 s7, 0 // (op 438) call()
s_mov_b32 s18, 0 // (op 439) call()
s_mov_b32 s19, 0 // (op 440) call()
s_mov_b32 s0, 0 // (op 335) call()

s_waitcnt lgkmcnt(0)// DEBUG: Wait for scalar queue
s_mul_i32 s52, s8, s16 // (op 334) 
s_cmp_gt_u32 s9, s52 // (op 334) 
s_mov_b32 s53, scc // (op 334) 
s_sub_u32 s54, s9, s52 // (op 334) 

s_cmp_eq_i32 s53, 0 // (op 334) 
s_cselect_b32 s52, 0, s54 // (op 334) 


s_cmp_lt_u32 s8, s52 // (op 334) 
s_mov_b32 s53, scc // (op 334) 
s_cmp_eq_i32 s53, 0 // (op 334) 
s_cselect_b32 s54, s52, s8 // (op 334) 


s_cmp_lt_u32 s0, s54 // (op 334) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_SKStreamTileLoop_334 // (op 334) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_SKStreamTileLoop_334 if false)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_SKStreamTileLoop_334:
v_lshrrev_b32 v20, 7, v1 // (op 3197) 
s_mul_i32 s52, s16, s8 // (op 3197) 
s_add_u32 s52, s52, s0 // (op 3197) 
v_and_b32 v21, 63, v1 // (op 3197) 
s_mul_hi_u32 s53, s52, s13 // (op 3197) 
v_lshrrev_b32 v22, 4, v21 // (op 3197) 
s_sub_u32 s54, s52, s53 // (op 3197) 
v_lshlrev_b32 v23, 2, v22 // (op 3197) 

v_lshl_add_u32 v22, v20, 4, v23 // (op 3197) 


s_lshr_b32 s55, s54, 1 // (op 3197) 

s_add_u32 s54, s55, s53 // (op 3197) 


v_lshrrev_b32 v20, 2, v21 // (op 3197) 

s_lshr_b32 s53, s54, s11 // (op 3197) 

v_and_b32 v21, 3, v20 // (op 3197) 

s_and_b32 s54, s11, 2147483648 // (op 3197) 
v_and_b32 v20, 3, v1 // (op 3197) 
s_cmp_eq_i32 s54, 0 // (op 3197) 
s_mov_b32 s55, scc // (op 3197) 

v_lshl_add_u32 v23, v21, 2, v20 // (op 3197) 


s_cmp_eq_i32 s55, 0 // (op 3197) 
s_cselect_b32 s54, s52, s53 // (op 3197) 



v_lshrrev_b32 v20, 6, v1 // (op 3197) 
s_mul_hi_u32 s52, s54, s12 // (op 3197) 
v_and_b32 v21, 1, v20 // (op 3197) 

s_sub_u32 s53, s54, s52 // (op 3197) 
v_lshl_add_u32 v20, v21, 4, v23 // (op 3197) 


s_lshr_b32 s55, s53, 1 // (op 3197) 

s_add_u32 s53, s55, s52 // (op 3197) 


s_lshr_b32 s52, s53, s10 // (op 3197) 

s_and_b32 s53, s10, 2147483648 // (op 3197) 
s_cmp_eq_i32 s53, 0 // (op 3197) 
s_mov_b32 s55, scc // (op 3197) 

s_cmp_eq_i32 s55, 0 // (op 3197) 
s_cselect_b32 s53, s54, s52 // (op 3197) 


s_mul_i32 s52, s53, s14 // (op 3197) 
v_lshl_add_u32 v21, s53, 4, v22 // (op 3197) 


s_sub_u32 s53, s54, s52 // (op 3197) 


v_lshl_add_u32 v22, s53, 6, v20 // (op 3197) 


v_mul_lo_u32 v20, v22, s4 // (op 3197) 

v_add_lshl_u32 v22, v21, v20, 1 // (op 3197) 


s_mul_i32 s53, s4, s23 // (op 3199) most significant: low of low * high
s_mul_i32 s54, s5, s22 // (op 3199) most significant: low of high * low
s_mul_i32 s52, s4, s22 // (op 3199) least significant
s_mul_hi_u32 s55, s4, s22 // (op 3199) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3199) most significant: sum
s_add_u32 s53, s53, s55 // (op 3199) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 3199) 

s_mov_b32 s56, s20 // (op 3199) reg expression
s_mov_b32 s57, s21 // (op 3199) reg expression
s_mov_b32 s58, s54 // (op 3199) reg expression
s_mov_b32 s59, 131072 // (op 3199) call()

v_and_b32 v20, 48, v1 // (op 3288) 
v_and_b32 v21, 63, v1 // (op 3288) 
v_sub_u32 v23, v21, v20 // (op 3288) 

v_lshrrev_b32 v21, 1, v23 // (op 3288) 
v_add_lshl_u32 v24, v20, v21, 4 // (op 3288) 


v_and_b32 v20, 1, v23 // (op 3288) 

v_lshlrev_b32 v21, 3, v20 // (op 3288) 

v_lshrrev_b32 v20, 7, v1 // (op 3288) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 3288) 


v_add_u32 v20, v24, v23 // (op 3288) 


v_and_b32 v21, 48, v1 // (op 3363) 
v_and_b32 v23, 63, v1 // (op 3363) 
v_sub_u32 v24, v23, v21 // (op 3363) 

v_lshrrev_b32 v23, 1, v24 // (op 3363) 
v_add_lshl_u32 v25, v21, v23, 6 // (op 3363) 


v_and_b32 v21, 1, v24 // (op 3363) 

v_lshlrev_b32 v23, 3, v21 // (op 3363) 

v_lshrrev_b32 v21, 6, v1 // (op 3363) 
v_and_b32 v24, 1, v21 // (op 3363) 

v_lshl_add_u32 v21, v24, 4, v23 // (op 3363) 


v_add_u32 v23, v25, v21 // (op 3363) 


v_lshrrev_b32 v21, 7, v1 // (op 3429) 
v_and_b32 v24, 15, v1 // (op 3429) 
v_lshl_add_u32 v25, v21, 4, v24 // (op 3429) 


v_and_b32 v21, 48, v1 // (op 3429) 
v_add_u32 v24, v21, v25 // (op 3429) 


v_lshrrev_b32 v21, 6, v1 // (op 3505) 
v_and_b32 v25, 1, v21 // (op 3505) 

v_and_b32 v21, 15, v1 // (op 3505) 
v_lshl_add_u32 v26, v25, 4, v21 // (op 3505) 


v_and_b32 v21, 63, v1 // (op 3505) 
v_lshrrev_b32 v25, 4, v21 // (op 3505) 

v_lshl_add_u32 v21, v25, 6, v26 // (op 3505) 


v_lshrrev_b32 v25, 7, v1 // (op 3523) 
s_mul_i32 s52, s16, s8 // (op 3523) 
s_add_u32 s52, s52, s0 // (op 3523) 
v_and_b32 v26, 63, v1 // (op 3523) 
s_mul_hi_u32 s53, s52, s13 // (op 3523) 
v_lshrrev_b32 v27, 4, v26 // (op 3523) 
s_sub_u32 s54, s52, s53 // (op 3523) 
v_lshlrev_b32 v28, 2, v27 // (op 3523) 

s_lshr_b32 s55, s54, 1 // (op 3523) 

v_lshl_add_u32 v27, v25, 4, v28 // (op 3523) 


s_add_u32 s54, s55, s53 // (op 3523) 


v_lshrrev_b32 v25, 2, v26 // (op 3523) 

s_lshr_b32 s53, s54, s11 // (op 3523) 

v_and_b32 v26, 3, v25 // (op 3523) 

s_and_b32 s54, s11, 2147483648 // (op 3523) 
v_and_b32 v25, 3, v1 // (op 3523) 
s_cmp_eq_i32 s54, 0 // (op 3523) 
s_mov_b32 s55, scc // (op 3523) 

v_lshl_add_u32 v28, v26, 2, v25 // (op 3523) 


s_cmp_eq_i32 s55, 0 // (op 3523) 
s_cselect_b32 s54, s52, s53 // (op 3523) 



v_lshrrev_b32 v25, 6, v1 // (op 3523) 
s_mul_hi_u32 s55, s54, s12 // (op 3523) 
v_and_b32 v26, 1, v25 // (op 3523) 

s_sub_u32 s52, s54, s55 // (op 3523) 
v_lshl_add_u32 v25, v26, 4, v28 // (op 3523) 


s_lshr_b32 s53, s52, 1 // (op 3523) 

s_add_u32 s52, s53, s55 // (op 3523) 


s_lshr_b32 s53, s52, s10 // (op 3523) 

s_and_b32 s52, s10, 2147483648 // (op 3523) 
s_cmp_eq_i32 s52, 0 // (op 3523) 
s_mov_b32 s55, scc // (op 3523) 

s_cmp_eq_i32 s55, 0 // (op 3523) 
s_cselect_b32 s52, s54, s53 // (op 3523) 


s_mul_i32 s53, s52, s14 // (op 3523) 
v_lshl_add_u32 v26, s52, 4, v27 // (op 3523) 


s_sub_u32 s52, s54, s53 // (op 3523) 


v_lshl_add_u32 v27, s52, 6, v25 // (op 3523) 


v_mul_lo_u32 v25, v27, s24 // (op 3523) 

v_add_lshl_u32 v27, v26, v25, 1 // (op 3523) 


s_mul_i32 s53, s24, s23 // (op 3525) most significant: low of low * high
s_mul_i32 s54, s25, s22 // (op 3525) most significant: low of high * low
s_mul_i32 s52, s24, s22 // (op 3525) least significant
s_mul_hi_u32 s55, s24, s22 // (op 3525) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3525) most significant: sum
s_add_u32 s53, s53, s55 // (op 3525) most significant: sum


s_lshl_b64 s[54:55], s[52:53], 1 // (op 3525) 

s_mov_b32 s60, s26 // (op 3525) reg expression
s_mov_b32 s61, s27 // (op 3525) reg expression
s_mov_b32 s62, s54 // (op 3525) reg expression
s_mov_b32 s63, 131072 // (op 3525) call()

v_lshrrev_b32 v25, 5, v1 // (op 3847) 
v_lshlrev_b32 v26, 2, v25 // (op 3847) 

v_lshl_add_u32 v25, s16, 4, v26 // (op 3847) 

v_and_b32 v26, -32, v1 // (op 3847) 
v_sub_u32 v28, v1, v26 // (op 3847) 

v_lshl_add_u32 v26, v25, 6, v28 // (op 3847) 


v_lshlrev_b32 v25, 2, v26 // (op 3847) 

s_lshl_b32 s52, s15, 12 // (op 3849) 
s_mov_b32 s64, s28 // (op 3849) reg expression
s_mov_b32 s65, s29 // (op 3849) reg expression
s_mov_b32 s66, s52 // (op 3849) reg expression
s_mov_b32 s67, 131072 // (op 3849) call()

s_mul_i32 s6, s8, s16 // (op 339) 
s_add_u32 s6, s6, s0 // (op 339) 
s_mul_hi_u32 s52, s6, s13 // (op 340) 
s_sub_u32 s53, s6, s52 // (op 340) 
s_lshr_b32 s54, s53, 1 // (op 340) 

s_add_u32 s53, s54, s52 // (op 340) 


s_lshr_b32 s52, s53, s11 // (op 340) 

s_and_b32 s53, s11, 2147483648 // (op 340) 
s_cmp_eq_i32 s53, 0 // (op 340) 
s_mov_b32 s54, scc // (op 340) 

s_cmp_eq_i32 s54, 0 // (op 340) 
s_cselect_b32 s53, s6, s52 // (op 340) 


s_ashr_i64 s[54:55], s[2:3], 8 // (op 340) 
s_mul_i32 s52, s53, s54 // (op 340) 


s_sub_u32 s18, s6, s52 // (op 340) 

s_ashr_i64 s[52:53], s[2:3], 8 // (op 341) 
s_cmp_gt_u32 s52, s18 // (op 341) 
s_mov_b32 s54, scc // (op 341) 
s_sub_u32 s55, s52, s18 // (op 341) 

s_cmp_eq_i32 s54, 0 // (op 341) 
s_cselect_b32 s52, 0, s55 // (op 341) 


s_cmp_gt_u32 s8, s0 // (op 341) 
s_mov_b32 s53, scc // (op 341) 
s_sub_u32 s54, s8, s0 // (op 341) 
s_cmp_eq_i32 s53, 0 // (op 341) 
s_cselect_b32 s55, 0, s54 // (op 341) 


s_cmp_lt_u32 s52, s55 // (op 341) 
s_mov_b32 s53, scc // (op 341) 
s_cmp_eq_i32 s53, 0 // (op 341) 
s_cselect_b32 s54, s55, s52 // (op 341) 



s_cmp_gt_u32 s9, s6 // (op 341) 
s_mov_b32 s55, scc // (op 341) 
s_sub_u32 s52, s9, s6 // (op 341) 
s_cmp_eq_i32 s55, 0 // (op 341) 
s_cselect_b32 s53, 0, s52 // (op 341) 


s_cmp_lt_u32 s54, s53 // (op 341) 
s_mov_b32 s52, scc // (op 341) 
s_cmp_eq_i32 s52, 0 // (op 341) 
s_cselect_b32 s7, s53, s54 // (op 341) 



s_add_u32 s52, s18, s7 // (op 342) 
s_sub_u32 s19, s52, 1 // (op 342) 

v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 1273) call()
v_accvgpr_write a5, 0.00000 // (op 1273) call()
v_accvgpr_write a6, 0.00000 // (op 1273) call()
v_accvgpr_write a7, 0.00000 // (op 1273) call()
s_mul_i32 s52, s16, s8 // (op 2769) 
s_add_u32 s52, s52, s0 // (op 2769) 
s_mul_hi_u32 s53, s52, s13 // (op 2769) 
s_sub_u32 s54, s52, s53 // (op 2769) 
s_lshr_b32 s55, s54, 1 // (op 2769) 

s_add_u32 s54, s55, s53 // (op 2769) 


s_lshr_b32 s53, s54, s11 // (op 2769) 

s_and_b32 s54, s11, 2147483648 // (op 2769) 
s_cmp_eq_i32 s54, 0 // (op 2769) 
s_mov_b32 s55, scc // (op 2769) 

s_cmp_eq_i32 s55, 0 // (op 2769) 
s_cselect_b32 s54, s52, s53 // (op 2769) 


s_mul_hi_u32 s53, s54, s12 // (op 2769) 
s_sub_u32 s55, s54, s53 // (op 2769) 
s_lshr_b32 s68, s55, 1 // (op 2769) 

s_add_u32 s55, s68, s53 // (op 2769) 


s_lshr_b32 s53, s55, s10 // (op 2769) 

s_and_b32 s55, s10, 2147483648 // (op 2769) 
s_cmp_eq_i32 s55, 0 // (op 2769) 
s_mov_b32 s68, scc // (op 2769) 

s_cmp_eq_i32 s68, 0 // (op 2769) 
s_cselect_b32 s55, s54, s53 // (op 2769) 


s_lshl_b32 s53, s55, 4 // (op 2769) 

s_mov_b32 s69, 0 // (op 2769) convert
s_mov_b32 s68, s53 // (op 2769) convert

s_ashr_i64 s[70:71], s[2:3], 8 // (op 2769) 
s_mul_i32 s53, s54, s70 // (op 2769) 


s_sub_u32 s54, s52, s53 // (op 2769) 


v_lshl_add_u32 v26, s54, 8, v1 // (op 2769) 

v_mul_lo_u32 v29, v26, s31 // (op 2769) most significant: low of low * high
v_mul_hi_u32 v30, v26, s30 // (op 2769) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 2769) least significant: low of low * low
v_add_u32 v29, v29, v30 // (op 2769) most significant: sum


v_mov_b32 v26, s68 // (op 2769) 
v_mov_b32 v30, s69 // (op 2769) 
v_add_co_u32 v32, s[52:53], v26, v28 // (op 2769) least significant half
v_addc_co_u32 v33, s[52:53], v30, v29, s[52:53] // (op 2769) most significant half





s_mul_i32 s53, s30, s3 // (op 2771) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2771) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2771) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2771) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2771) most significant: sum
s_add_u32 s53, s53, s55 // (op 2771) most significant: sum


s_mov_b32 s68, s32 // (op 2771) reg expression
s_mov_b32 s69, s33 // (op 2771) reg expression
s_mov_b32 s70, s52 // (op 2771) reg expression
s_mov_b32 s71, 131072 // (op 2771) call()

s_mul_i32 s52, s16, s8 // (op 2788) 
s_add_u32 s52, s52, s0 // (op 2788) 
s_mul_hi_u32 s53, s52, s13 // (op 2788) 
s_sub_u32 s54, s52, s53 // (op 2788) 
s_lshr_b32 s55, s54, 1 // (op 2788) 

s_add_u32 s54, s55, s53 // (op 2788) 


s_lshr_b32 s53, s54, s11 // (op 2788) 

s_and_b32 s54, s11, 2147483648 // (op 2788) 
s_cmp_eq_i32 s54, 0 // (op 2788) 
s_mov_b32 s55, scc // (op 2788) 

s_cmp_eq_i32 s55, 0 // (op 2788) 
s_cselect_b32 s54, s52, s53 // (op 2788) 


s_mul_hi_u32 s53, s54, s12 // (op 2788) 
s_sub_u32 s55, s54, s53 // (op 2788) 
s_lshr_b32 s72, s55, 1 // (op 2788) 

s_add_u32 s55, s72, s53 // (op 2788) 


s_lshr_b32 s53, s55, s10 // (op 2788) 

s_and_b32 s55, s10, 2147483648 // (op 2788) 
s_cmp_eq_i32 s55, 0 // (op 2788) 
s_mov_b32 s72, scc // (op 2788) 

s_cmp_eq_i32 s72, 0 // (op 2788) 
s_cselect_b32 s55, s54, s53 // (op 2788) 


s_lshl_b32 s53, s55, 4 // (op 2788) 

s_mov_b32 s73, 0 // (op 2788) convert
s_mov_b32 s72, s53 // (op 2788) convert

s_ashr_i64 s[74:75], s[2:3], 8 // (op 2788) 
s_mul_i32 s53, s54, s74 // (op 2788) 


s_sub_u32 s54, s52, s53 // (op 2788) 


v_lshl_add_u32 v26, s54, 8, v1 // (op 2788) 

v_mul_lo_u32 v29, v26, s31 // (op 2788) most significant: low of low * high
v_mul_hi_u32 v30, v26, s30 // (op 2788) most significant: high of low * low
v_mul_lo_u32 v28, v26, s30 // (op 2788) least significant: low of low * low
v_add_u32 v29, v29, v30 // (op 2788) most significant: sum


v_mov_b32 v26, s72 // (op 2788) 
v_mov_b32 v30, s73 // (op 2788) 
v_add_co_u32 v34, s[52:53], v26, v28 // (op 2788) least significant half
v_addc_co_u32 v35, s[52:53], v30, v29, s[52:53] // (op 2788) most significant half





s_mul_i32 s53, s30, s3 // (op 2790) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2790) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2790) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2790) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2790) most significant: sum
s_add_u32 s53, s53, s55 // (op 2790) most significant: sum


s_mov_b32 s68, s32 // (op 2790) reg expression
s_mov_b32 s69, s33 // (op 2790) reg expression
s_mov_b32 s70, s52 // (op 2790) reg expression
s_mov_b32 s71, 131072 // (op 2790) call()

v_and_b32 v26, -4, v1 // (op 2897) 
s_mul_i32 s52, s16, s8 // (op 2897) 
s_add_u32 s52, s52, s0 // (op 2897) 
v_sub_u32 v28, v1, v26 // (op 2897) 

s_mul_hi_u32 s53, s52, s13 // (op 2897) 
v_lshlrev_b32 v26, 4, v28 // (op 2897) 

s_sub_u32 s54, s52, s53 // (op 2897) 
v_lshrrev_b32 v28, 2, v1 // (op 2897) 
s_lshr_b32 s55, s54, 1 // (op 2897) 

s_add_u32 s54, s55, s53 // (op 2897) 


s_lshr_b32 s53, s54, s11 // (op 2897) 

s_and_b32 s54, s11, 2147483648 // (op 2897) 
s_cmp_eq_i32 s54, 0 // (op 2897) 
s_mov_b32 s55, scc // (op 2897) 

s_cmp_eq_i32 s55, 0 // (op 2897) 
s_cselect_b32 s54, s52, s53 // (op 2897) 


s_mul_hi_u32 s53, s54, s12 // (op 2897) 
s_sub_u32 s55, s54, s53 // (op 2897) 
s_lshr_b32 s72, s55, 1 // (op 2897) 

s_add_u32 s55, s72, s53 // (op 2897) 


s_lshr_b32 s53, s55, s10 // (op 2897) 

s_and_b32 s55, s10, 2147483648 // (op 2897) 
s_cmp_eq_i32 s55, 0 // (op 2897) 
s_mov_b32 s72, scc // (op 2897) 

s_cmp_eq_i32 s72, 0 // (op 2897) 
s_cselect_b32 s55, s54, s53 // (op 2897) 


s_mul_i32 s53, s55, s14 // (op 2897) 

s_sub_u32 s55, s54, s53 // (op 2897) 

v_lshl_add_u32 v29, s55, 6, v26 // (op 2897) 


s_ashr_i64 s[72:73], s[2:3], 8 // (op 2897) 
v_mov_b32 v31, 0 // (op 2897) convert
v_mov_b32 v30, v29 // (op 2897) convert

s_mul_i32 s53, s54, s72 // (op 2897) 


s_sub_u32 s54, s52, s53 // (op 2897) 


v_lshl_add_u32 v26, s54, 8, v28 // (op 2897) 


v_mul_lo_u32 v29, v26, s35 // (op 2897) most significant: low of low * high
v_mul_hi_u32 v36, v26, s34 // (op 2897) most significant: high of low * low
v_mul_lo_u32 v28, v26, s34 // (op 2897) least significant: low of low * low
v_add_u32 v29, v29, v36 // (op 2897) most significant: sum

v_add_co_u32 v28, s[52:53], v28, v30 // (op 2897) least significant half
v_addc_co_u32 v29, s[52:53], v29, v31, s[52:53] // (op 2897) most significant half



s_mul_i32 s53, s34, s3 // (op 2899) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2899) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2899) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2899) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2899) most significant: sum
s_add_u32 s53, s53, s55 // (op 2899) most significant: sum


s_mov_b32 s72, s36 // (op 2899) reg expression
s_mov_b32 s73, s37 // (op 2899) reg expression
s_mov_b32 s74, s52 // (op 2899) reg expression
s_mov_b32 s75, 131072 // (op 2899) call()

v_and_b32 v26, -4, v1 // (op 2916) 
s_mul_i32 s52, s16, s8 // (op 2916) 
s_add_u32 s52, s52, s0 // (op 2916) 
v_sub_u32 v30, v1, v26 // (op 2916) 

s_mul_hi_u32 s53, s52, s13 // (op 2916) 
v_lshlrev_b32 v26, 4, v30 // (op 2916) 

s_sub_u32 s54, s52, s53 // (op 2916) 
v_lshrrev_b32 v30, 2, v1 // (op 2916) 
s_lshr_b32 s55, s54, 1 // (op 2916) 

s_add_u32 s54, s55, s53 // (op 2916) 


s_lshr_b32 s53, s54, s11 // (op 2916) 

s_and_b32 s54, s11, 2147483648 // (op 2916) 
s_cmp_eq_i32 s54, 0 // (op 2916) 
s_mov_b32 s55, scc // (op 2916) 

s_cmp_eq_i32 s55, 0 // (op 2916) 
s_cselect_b32 s54, s52, s53 // (op 2916) 


s_mul_hi_u32 s53, s54, s12 // (op 2916) 
s_sub_u32 s55, s54, s53 // (op 2916) 
s_lshr_b32 s76, s55, 1 // (op 2916) 

s_add_u32 s55, s76, s53 // (op 2916) 


s_lshr_b32 s53, s55, s10 // (op 2916) 

s_and_b32 s55, s10, 2147483648 // (op 2916) 
s_cmp_eq_i32 s55, 0 // (op 2916) 
s_mov_b32 s76, scc // (op 2916) 

s_cmp_eq_i32 s76, 0 // (op 2916) 
s_cselect_b32 s55, s54, s53 // (op 2916) 


s_mul_i32 s53, s55, s14 // (op 2916) 

s_sub_u32 s55, s54, s53 // (op 2916) 

v_lshl_add_u32 v31, s55, 6, v26 // (op 2916) 


s_ashr_i64 s[76:77], s[2:3], 8 // (op 2916) 
v_mov_b32 v37, 0 // (op 2916) convert
v_mov_b32 v36, v31 // (op 2916) convert

s_mul_i32 s53, s54, s76 // (op 2916) 


s_sub_u32 s54, s52, s53 // (op 2916) 


v_lshl_add_u32 v26, s54, 8, v30 // (op 2916) 


v_mul_lo_u32 v31, v26, s35 // (op 2916) most significant: low of low * high
v_mul_hi_u32 v38, v26, s34 // (op 2916) most significant: high of low * low
v_mul_lo_u32 v30, v26, s34 // (op 2916) least significant: low of low * low
v_add_u32 v31, v31, v38 // (op 2916) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v36 // (op 2916) least significant half
v_addc_co_u32 v31, s[52:53], v31, v37, s[52:53] // (op 2916) most significant half



s_mul_i32 s53, s34, s3 // (op 2918) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2918) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2918) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2918) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2918) most significant: sum
s_add_u32 s53, s53, s55 // (op 2918) most significant: sum


s_mov_b32 s72, s36 // (op 2918) reg expression
s_mov_b32 s73, s37 // (op 2918) reg expression
s_mov_b32 s74, s52 // (op 2918) reg expression
s_mov_b32 s75, 131072 // (op 2918) call()

v_and_b32 v26, -16, v1 // (op 3025) 
s_mul_i32 s52, s16, s8 // (op 3025) 
s_add_u32 s52, s52, s0 // (op 3025) 
v_sub_u32 v36, v1, v26 // (op 3025) 

s_mul_hi_u32 s53, s52, s13 // (op 3025) 
v_lshrrev_b32 v26, 4, v1 // (op 3025) 
s_sub_u32 s54, s52, s53 // (op 3025) 
s_lshr_b32 s55, s54, 1 // (op 3025) 

s_add_u32 s54, s55, s53 // (op 3025) 


s_lshr_b32 s53, s54, s11 // (op 3025) 

s_and_b32 s54, s11, 2147483648 // (op 3025) 
s_cmp_eq_i32 s54, 0 // (op 3025) 
s_mov_b32 s55, scc // (op 3025) 

s_cmp_eq_i32 s55, 0 // (op 3025) 
s_cselect_b32 s54, s52, s53 // (op 3025) 


s_mul_hi_u32 s53, s54, s12 // (op 3025) 
s_sub_u32 s55, s54, s53 // (op 3025) 
s_lshr_b32 s76, s55, 1 // (op 3025) 

s_add_u32 s55, s76, s53 // (op 3025) 


s_lshr_b32 s53, s55, s10 // (op 3025) 

s_and_b32 s55, s10, 2147483648 // (op 3025) 
s_cmp_eq_i32 s55, 0 // (op 3025) 
s_mov_b32 s76, scc // (op 3025) 

s_cmp_eq_i32 s76, 0 // (op 3025) 
s_cselect_b32 s55, s54, s53 // (op 3025) 


v_lshl_add_u32 v37, s55, 4, v36 // (op 3025) 


s_ashr_i64 s[76:77], s[2:3], 8 // (op 3025) 
s_mul_i32 s53, s54, s76 // (op 3025) 


s_sub_u32 s54, s52, s53 // (op 3025) 


v_lshl_add_u32 v36, s54, 3, v26 // (op 3025) 


v_mul_lo_u32 v26, v36, s38 // (op 3025) 

v_add_u32 v36, v37, v26 // (op 3025) 


s_mul_i32 s53, s38, s43 // (op 3027) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 3027) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 3027) least significant
s_mul_hi_u32 s55, s38, s42 // (op 3027) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3027) most significant: sum
s_add_u32 s53, s53, s55 // (op 3027) most significant: sum


s_mov_b32 s76, s40 // (op 3027) reg expression
s_mov_b32 s77, s41 // (op 3027) reg expression
s_mov_b32 s78, s52 // (op 3027) reg expression
s_mov_b32 s79, 131072 // (op 3027) call()

v_and_b32 v26, -16, v1 // (op 3044) 
s_mul_i32 s52, s16, s8 // (op 3044) 
s_add_u32 s52, s52, s0 // (op 3044) 
v_sub_u32 v37, v1, v26 // (op 3044) 

s_mul_hi_u32 s53, s52, s13 // (op 3044) 
v_lshrrev_b32 v26, 4, v1 // (op 3044) 
s_sub_u32 s54, s52, s53 // (op 3044) 
s_lshr_b32 s55, s54, 1 // (op 3044) 

s_add_u32 s54, s55, s53 // (op 3044) 


s_lshr_b32 s53, s54, s11 // (op 3044) 

s_and_b32 s54, s11, 2147483648 // (op 3044) 
s_cmp_eq_i32 s54, 0 // (op 3044) 
s_mov_b32 s55, scc // (op 3044) 

s_cmp_eq_i32 s55, 0 // (op 3044) 
s_cselect_b32 s54, s52, s53 // (op 3044) 


s_mul_hi_u32 s53, s54, s12 // (op 3044) 
s_sub_u32 s55, s54, s53 // (op 3044) 
s_lshr_b32 s80, s55, 1 // (op 3044) 

s_add_u32 s55, s80, s53 // (op 3044) 


s_lshr_b32 s53, s55, s10 // (op 3044) 

s_and_b32 s55, s10, 2147483648 // (op 3044) 
s_cmp_eq_i32 s55, 0 // (op 3044) 
s_mov_b32 s80, scc // (op 3044) 

s_cmp_eq_i32 s80, 0 // (op 3044) 
s_cselect_b32 s55, s54, s53 // (op 3044) 


v_lshl_add_u32 v38, s55, 4, v37 // (op 3044) 


s_ashr_i64 s[80:81], s[2:3], 8 // (op 3044) 
s_mul_i32 s53, s54, s80 // (op 3044) 


s_sub_u32 s54, s52, s53 // (op 3044) 


v_lshl_add_u32 v37, s54, 3, v26 // (op 3044) 


v_mul_lo_u32 v26, v37, s38 // (op 3044) 

v_add_u32 v37, v38, v26 // (op 3044) 


s_mul_i32 s53, s38, s43 // (op 3046) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 3046) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 3046) least significant
s_mul_hi_u32 s55, s38, s42 // (op 3046) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3046) most significant: sum
s_add_u32 s53, s53, s55 // (op 3046) most significant: sum


s_mov_b32 s76, s40 // (op 3046) reg expression
s_mov_b32 s77, s41 // (op 3046) reg expression
s_mov_b32 s78, s52 // (op 3046) reg expression
s_mov_b32 s79, 131072 // (op 3046) call()

v_and_b32 v26, -16, v1 // (op 3153) 
s_mul_i32 s52, s16, s8 // (op 3153) 
s_add_u32 s52, s52, s0 // (op 3153) 
v_sub_u32 v38, v1, v26 // (op 3153) 

s_mul_hi_u32 s53, s52, s13 // (op 3153) 
v_lshlrev_b32 v26, 2, v38 // (op 3153) 

s_sub_u32 s54, s52, s53 // (op 3153) 
v_lshrrev_b32 v38, 4, v1 // (op 3153) 
s_lshr_b32 s55, s54, 1 // (op 3153) 

s_add_u32 s54, s55, s53 // (op 3153) 


s_lshr_b32 s53, s54, s11 // (op 3153) 

s_and_b32 s54, s11, 2147483648 // (op 3153) 
s_cmp_eq_i32 s54, 0 // (op 3153) 
s_mov_b32 s55, scc // (op 3153) 

s_cmp_eq_i32 s55, 0 // (op 3153) 
s_cselect_b32 s54, s52, s53 // (op 3153) 


s_mul_hi_u32 s53, s54, s12 // (op 3153) 
s_sub_u32 s55, s54, s53 // (op 3153) 
s_lshr_b32 s80, s55, 1 // (op 3153) 

s_add_u32 s55, s80, s53 // (op 3153) 


s_lshr_b32 s53, s55, s10 // (op 3153) 

s_and_b32 s55, s10, 2147483648 // (op 3153) 
s_cmp_eq_i32 s55, 0 // (op 3153) 
s_mov_b32 s80, scc // (op 3153) 

s_cmp_eq_i32 s80, 0 // (op 3153) 
s_cselect_b32 s55, s54, s53 // (op 3153) 


s_mul_i32 s53, s55, s14 // (op 3153) 

s_sub_u32 s55, s54, s53 // (op 3153) 

s_ashr_i64 s[80:81], s[2:3], 8 // (op 3153) 
v_lshl_add_u32 v39, s55, 6, v26 // (op 3153) 


s_mul_i32 s53, s54, s80 // (op 3153) 


s_sub_u32 s54, s52, s53 // (op 3153) 


v_lshl_add_u32 v26, s54, 3, v38 // (op 3153) 


v_mul_lo_u32 v38, v26, s44 // (op 3153) 
v_add_u32 v38, v38, v39 // (op 3153) 


s_mul_i32 s53, s44, s43 // (op 3155) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 3155) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 3155) least significant
s_mul_hi_u32 s55, s44, s42 // (op 3155) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3155) most significant: sum
s_add_u32 s53, s53, s55 // (op 3155) most significant: sum


s_mov_b32 s80, s46 // (op 3155) reg expression
s_mov_b32 s81, s47 // (op 3155) reg expression
s_mov_b32 s82, s52 // (op 3155) reg expression
s_mov_b32 s83, 131072 // (op 3155) call()

v_and_b32 v26, -16, v1 // (op 3172) 
s_mul_i32 s52, s16, s8 // (op 3172) 
s_add_u32 s52, s52, s0 // (op 3172) 
v_sub_u32 v39, v1, v26 // (op 3172) 

s_mul_hi_u32 s53, s52, s13 // (op 3172) 
v_lshlrev_b32 v26, 2, v39 // (op 3172) 

s_sub_u32 s54, s52, s53 // (op 3172) 
v_lshrrev_b32 v39, 4, v1 // (op 3172) 
s_lshr_b32 s55, s54, 1 // (op 3172) 

s_add_u32 s54, s55, s53 // (op 3172) 


s_lshr_b32 s53, s54, s11 // (op 3172) 

s_and_b32 s54, s11, 2147483648 // (op 3172) 
s_cmp_eq_i32 s54, 0 // (op 3172) 
s_mov_b32 s55, scc // (op 3172) 

s_cmp_eq_i32 s55, 0 // (op 3172) 
s_cselect_b32 s54, s52, s53 // (op 3172) 


s_mul_hi_u32 s53, s54, s12 // (op 3172) 
s_sub_u32 s55, s54, s53 // (op 3172) 
s_lshr_b32 s84, s55, 1 // (op 3172) 

s_add_u32 s55, s84, s53 // (op 3172) 


s_lshr_b32 s53, s55, s10 // (op 3172) 

s_and_b32 s55, s10, 2147483648 // (op 3172) 
s_cmp_eq_i32 s55, 0 // (op 3172) 
s_mov_b32 s84, scc // (op 3172) 

s_cmp_eq_i32 s84, 0 // (op 3172) 
s_cselect_b32 s55, s54, s53 // (op 3172) 


s_mul_i32 s53, s55, s14 // (op 3172) 

s_sub_u32 s55, s54, s53 // (op 3172) 

v_lshl_add_u32 v40, s55, 6, v26 // (op 3172) 


s_ashr_i64 s[84:85], s[2:3], 8 // (op 3172) 
s_mul_i32 s53, s54, s84 // (op 3172) 


s_sub_u32 s54, s52, s53 // (op 3172) 


v_lshl_add_u32 v26, s54, 3, v39 // (op 3172) 


v_mul_lo_u32 v39, v26, s44 // (op 3172) 
v_add_u32 v39, v39, v40 // (op 3172) 


s_mul_i32 s53, s44, s43 // (op 3174) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 3174) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 3174) least significant
s_mul_hi_u32 s55, s44, s42 // (op 3174) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3174) most significant: sum
s_add_u32 s53, s53, s55 // (op 3174) most significant: sum


s_mov_b32 s80, s46 // (op 3174) reg expression
s_mov_b32 s81, s47 // (op 3174) reg expression
s_mov_b32 s82, s52 // (op 3174) reg expression
s_mov_b32 s83, 131072 // (op 3174) call()

v_and_b32 v26, -4, v1 // (op 3329) 
v_sub_u32 v40, v1, v26 // (op 3329) 

v_lshlrev_b32 v26, 4, v40 // (op 3329) 

v_lshrrev_b32 v40, 2, v1 // (op 3329) 
v_lshl_add_u32 v41, v40, 6, v26 // (op 3329) 


s_nop 0
v_readfirstlane_b32 s52, v41 // (op 3329) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4311) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 2647) 
v_mov_b64 v[40:41], s[54:55] // (op 2647) convert


v_mov_b64 v[42:43], v[28:29] // (op 2647) reg expression
s_mov_b32 m0, s52 // (op 2647) reg expression
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2647) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2647) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2647) most significant half

s_add_u32 m0, m0, 2048 // (op 2647) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2647) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2647) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2647) most significant half

s_add_u32 m0, m0, 2048 // (op 2647) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2647) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2647) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2647) most significant half

s_add_u32 m0, m0, 2048 // (op 2647) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2647) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2647) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2647) most significant half

s_add_u32 m0, m0, 2048 // (op 2647) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2647) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2647) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2647) most significant half

s_add_u32 m0, m0, 2048 // (op 2647) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2647) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2647) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2647) most significant half

s_add_u32 m0, m0, 2048 // (op 2647) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2647) Load value direct to lds
v_add_co_u32 v42, s[54:55], v42, v40 // (op 2647) least significant half
v_addc_co_u32 v43, s[54:55], 0, v43, s[54:55] // (op 2647) most significant half

s_add_u32 m0, m0, 2048 // (op 2647) 
buffer_load_dwordx4 v42, s[72:75], 0 offen offset: 0 lds // (op 2647) Load value direct to lds





v_mov_b32 v26, v38 // (op 2406) reg expression

buffer_load_dword v40, v26, s[80:83], 0 offen // (op 2406) Load value

v_lshlrev_b32 v26, 4, v1 // (op 3264) 
s_nop 0
v_readfirstlane_b32 s52, v26 // (op 3264) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4292) 

s_lshl_b64 s[54:55], s[30:31], 7 // (op 2652) 
v_mov_b64 v[42:43], s[54:55] // (op 2652) convert


v_mov_b64 v[44:45], v[32:33] // (op 2652) reg expression
s_add_u32 s53, 16384, s52 // (op 2652) 
s_mov_b32 m0, s53 // (op 2652) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 2652) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 2652) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 2652) most significant half

s_add_u32 m0, m0, 2048 // (op 2652) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 2652) Load value direct to lds






v_mov_b32 v26, v36 // (op 2412) reg expression

buffer_load_ubyte v41, v26, s[76:79], 0 offen offset: 0 // (op 2412) Load value

v_and_b32 v26, -4, v1 // (op 4074) 
v_sub_u32 v42, v1, v26 // (op 4074) 

v_lshlrev_b32 v26, 4, v42 // (op 4074) 

v_lshrrev_b32 v42, 2, v1 // (op 4074) 
v_lshl_add_u32 v43, v42, 6, v26 // (op 4074) 


s_nop 0
v_readfirstlane_b32 s52, v43 // (op 4074) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4494) 

s_lshl_b64 s[54:55], s[34:35], 5 // (op 2657) 
v_mov_b64 v[42:43], s[54:55] // (op 2657) convert


s_lshl_b64 s[54:55], s[34:35], 8 // (op 2657) 
v_mov_b32 v26, s54 // (op 2657) 
v_mov_b32 v44, s55 // (op 2657) 
v_add_co_u32 v46, s[84:85], v28, v26 // (op 2657) least significant half
v_addc_co_u32 v47, s[84:85], v29, v44, s[84:85] // (op 2657) most significant half




s_add_u32 s53, 20480, s52 // (op 2657) 
s_mov_b32 m0, s53 // (op 2657) reg expression
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2657) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2657) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2657) most significant half

s_add_u32 m0, m0, 2048 // (op 2657) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2657) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2657) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2657) most significant half

s_add_u32 m0, m0, 2048 // (op 2657) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2657) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2657) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2657) most significant half

s_add_u32 m0, m0, 2048 // (op 2657) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2657) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2657) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2657) most significant half

s_add_u32 m0, m0, 2048 // (op 2657) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2657) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2657) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2657) most significant half

s_add_u32 m0, m0, 2048 // (op 2657) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2657) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2657) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2657) most significant half

s_add_u32 m0, m0, 2048 // (op 2657) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2657) Load value direct to lds
v_add_co_u32 v46, s[54:55], v46, v42 // (op 2657) least significant half
v_addc_co_u32 v47, s[54:55], 0, v47, s[54:55] // (op 2657) most significant half

s_add_u32 m0, m0, 2048 // (op 2657) 
buffer_load_dwordx4 v46, s[72:75], 0 offen offset: 0 lds // (op 2657) Load value direct to lds







s_lshl_b32 s52, s44, 3 // (op 2418) 
v_add_u32 v26, v38, s52 // (op 2418) 


buffer_load_dword v28, v26, s[80:83], 0 offen // (op 2418) Load value


v_lshlrev_b32 v26, 4, v1 // (op 4035) 
s_nop 0
v_readfirstlane_b32 s52, v26 // (op 4035) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4484) 

s_lshl_b64 s[54:55], s[30:31], 7 // (op 2659) 
v_mov_b64 v[42:43], s[54:55] // (op 2659) convert


s_lshl_b64 s[54:55], s[30:31], 8 // (op 2659) 
v_mov_b32 v26, s54 // (op 2659) 
v_mov_b32 v29, s55 // (op 2659) 
v_add_co_u32 v44, s[84:85], v32, v26 // (op 2659) least significant half
v_addc_co_u32 v45, s[84:85], v33, v29, s[84:85] // (op 2659) most significant half




s_add_u32 s53, 36864, s52 // (op 2659) 
s_mov_b32 m0, s53 // (op 2659) reg expression
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 2659) Load value direct to lds
v_add_co_u32 v44, s[54:55], v44, v42 // (op 2659) least significant half
v_addc_co_u32 v45, s[54:55], 0, v45, s[54:55] // (op 2659) most significant half

s_add_u32 m0, m0, 2048 // (op 2659) 
buffer_load_dwordx4 v44, s[68:71], 0 offen offset: 0 lds // (op 2659) Load value direct to lds







s_lshl_b32 s52, s38, 3 // (op 2424) 
v_add_u32 v26, v36, s52 // (op 2424) 


buffer_load_ubyte v29, v26, s[76:79], 0 offen offset: 0 // (op 2424) Load value


v_and_b32 v26, -16, v1 // (op 3471) 
v_sub_u32 v36, v1, v26 // (op 3471) 

v_lshlrev_b32 v26, 2, v36 // (op 3471) 

v_lshrrev_b32 v36, 4, v1 // (op 3471) 
v_lshl_add_u32 v38, v36, 6, v26 // (op 3471) 


s_barrier  // (op 4352) 



s_waitcnt vmcnt(15)
ds_write_b32 v38, v40 offset:40960 // (op 2427) Store local data 

v_and_b32 v26, -16, v1 // (op 3405) 
v_sub_u32 v36, v1, v26 // (op 3405) 

v_and_b32 v26, 4294967280, v1 // (op 3405) 
v_add_u32 v38, v26, v36 // (op 3405) 


s_barrier  // (op 4331) 



s_waitcnt vmcnt(12)
ds_write_b8 v38, v41 offset:41472 // (op 2429) Store local data 

s_barrier  // (op 4308) 



ds_read_b64_tr_b8 v[42:43], v23 // (op 2450) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v23 offset:512 // (op 2450) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v23 offset:4096 // (op 2450) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v23 offset:4608 // (op 2450) Transpose load local data 



ds_read_b64_tr_b8 v[50:51], v23 offset:8192 // (op 2460) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v23 offset:8704 // (op 2460) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v23 offset:12288 // (op 2460) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v23 offset:12800 // (op 2460) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4349) 


ds_read_u8 v26, v21 offset:40960 // (op 2476) Load local data 


ds_read_u8 v36, v21 offset:41216 // (op 2486) Load local data 



ds_read_b64_tr_b8 v[58:59], v23 offset:32 // (op 2455) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v23 offset:544 // (op 2455) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v23 offset:4128 // (op 2455) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v23 offset:4640 // (op 2455) Transpose load local data 



ds_read_b64_tr_b8 v[66:67], v23 offset:8224 // (op 2465) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v23 offset:8736 // (op 2465) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v23 offset:12320 // (op 2465) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v23 offset:12832 // (op 2465) Transpose load local data 


ds_read_u8 v38, v21 offset:40992 // (op 2481) Load local data 


ds_read_u8 v32, v21 offset:41248 // (op 2491) Load local data 
s_barrier  // (op 4289) 



ds_read_b64_tr_b8 v[74:75], v20 offset:16384 // (op 2443) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v20 offset:16512 // (op 2443) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v20 offset:17408 // (op 2443) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v20 offset:17536 // (op 2443) Transpose load local data 



ds_read_b64_tr_b8 v[82:83], v20 offset:18432 // (op 2446) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v20 offset:18560 // (op 2446) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v20 offset:19456 // (op 2446) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v20 offset:19584 // (op 2446) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4328) 


ds_read_u8 v33, v24 offset:41472 // (op 2469) Load local data 


ds_read_u8 v90, v24 offset:41536 // (op 2472) Load local data 
v_lshlrev_b32 v91, 4, v1 // (op 3236) 
s_nop 0
v_readfirstlane_b32 s52, v91 // (op 3236) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v91, -4, v1 // (op 3301) 
v_sub_u32 v92, v1, v91 // (op 3301) 

v_lshlrev_b32 v91, 4, v92 // (op 3301) 

v_lshrrev_b32 v92, 2, v1 // (op 3301) 
v_lshl_add_u32 v93, v92, 6, v91 // (op 3301) 


s_nop 0
v_readfirstlane_b32 s53, v93 // (op 3301) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v91, -16, v1 // (op 3376) 
v_sub_u32 v92, v1, v91 // (op 3376) 

v_and_b32 v91, 4294967280, v1 // (op 3376) 
v_add_u32 v93, v91, v92 // (op 3376) 


v_and_b32 v91, -16, v1 // (op 3442) 
v_sub_u32 v92, v1, v91 // (op 3442) 

v_lshlrev_b32 v91, 2, v92 // (op 3442) 

v_lshrrev_b32 v92, 4, v1 // (op 3442) 
v_lshl_add_u32 v94, v92, 6, v91 // (op 3442) 


v_lshlrev_b32 v91, 4, v1 // (op 4020) 
s_nop 0
v_readfirstlane_b32 s54, v91 // (op 4020) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v91, -4, v1 // (op 4059) 
v_sub_u32 v92, v1, v91 // (op 4059) 

v_lshlrev_b32 v91, 4, v92 // (op 4059) 

v_lshrrev_b32 v92, 2, v1 // (op 4059) 
v_lshl_add_u32 v95, v92, 6, v91 // (op 4059) 


s_nop 0
v_readfirstlane_b32 s55, v95 // (op 4059) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v91, -16, v1 // (op 4103) 
v_sub_u32 v92, v1, v91 // (op 4103) 

v_and_b32 v91, 4294967280, v1 // (op 4103) 
v_add_u32 v95, v91, v92 // (op 4103) 


v_and_b32 v91, -16, v1 // (op 4127) 
v_sub_u32 v92, v1, v91 // (op 4127) 

v_lshlrev_b32 v91, 2, v92 // (op 4127) 

v_lshrrev_b32 v92, 4, v1 // (op 4127) 
v_lshl_add_u32 v96, v92, 6, v91 // (op 4127) 


s_mov_b32 s1, 0 // (op 1275) call()
s_and_b32 s84, s7, 4294967294 // (op 1274) 
s_cmp_lt_u32 s1, s84 // (op 1274) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_1274 // (op 1274) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_1274 if false)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_1274:
s_add_u32 s1, s1, 2 // (op 1276) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:81], v[42:49], a[0:3], v33, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 4314) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:81], v[58:65], a[4:7], v33, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1322) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 2572) 
v_mov_b64 v[98:99], s[84:85] // (op 2572) convert


v_mov_b64 v[100:101], v[30:31] // (op 2572) 
s_lshl_b64 s[84:85], s[34:35], 8 // (op 2572) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 2572) 

v_mov_b32 v97, s86 // (op 2572) 
v_mov_b32 v91, s87 // (op 2572) 
v_add_co_u32 v102, s[84:85], v100, v97 // (op 2572) least significant half
v_addc_co_u32 v103, s[84:85], v101, v91, s[84:85] // (op 2572) most significant half




s_mov_b32 m0, s53 // (op 2572) reg expression
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2572) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2572) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2572) most significant half

s_add_u32 m0, m0, 2048 // (op 2572) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2572) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2572) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2572) most significant half

s_add_u32 m0, m0, 2048 // (op 2572) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2572) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2572) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2572) most significant half

s_add_u32 m0, m0, 2048 // (op 2572) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:89], v[50:57], a[0:3], v90, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 107) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2572) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:89], v[66:73], a[4:7], v90, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1331) 
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2572) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2572) most significant half

s_add_u32 m0, m0, 2048 // (op 2572) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2572) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2572) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2572) most significant half

s_add_u32 m0, m0, 2048 // (op 2572) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2572) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2572) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2572) most significant half

s_add_u32 m0, m0, 2048 // (op 2572) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2572) Load value direct to lds
v_add_co_u32 v102, s[84:85], v102, v98 // (op 2572) least significant half
v_addc_co_u32 v103, s[84:85], 0, v103, s[84:85] // (op 2572) most significant half

s_add_u32 m0, m0, 2048 // (op 2572) 
buffer_load_dwordx4 v102, s[72:75], 0 offen offset: 0 lds // (op 2572) Load value direct to lds




v_mov_b32 v91, v39 // (op 25) 
s_lshl_b32 s84, s44, 4 // (op 25) 
v_add_u32 v92, v91, s84 // (op 25) 


buffer_load_dword v40, v92, s[80:83], 0 offen // (op 25) Load value

s_barrier  // (op 4295) 

s_lshl_b64 s[84:85], s[30:31], 7 // (op 2567) 
v_mov_b64 v[98:99], s[84:85] // (op 2567) convert


v_mov_b64 v[102:103], v[34:35] // (op 2567) 
s_lshl_b64 s[84:85], s[30:31], 8 // (op 2567) 
s_lshl_b64 s[86:87], s[84:85], 1 // (op 2567) 

v_mov_b32 v92, s86 // (op 2567) 
v_mov_b32 v97, s87 // (op 2567) 
v_add_co_u32 v104, s[84:85], v102, v92 // (op 2567) least significant half
v_addc_co_u32 v105, s[84:85], v103, v97, s[84:85] // (op 2567) most significant half




s_add_u32 s84, 16384, s52 // (op 2567) 
s_nop 0
s_mov_b32 m0, s84 // (op 2567) reg expression
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 2567) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v98 // (op 2567) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 2567) most significant half

s_add_u32 m0, m0, 2048 // (op 2567) 
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 2567) Load value direct to lds





v_mov_b32 v92, v37 // (op 17) 
s_lshl_b32 s84, s38, 4 // (op 17) 
v_add_u32 v97, v92, s84 // (op 17) 


buffer_load_ubyte v41, v97, s[76:79], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 4510) 



s_waitcnt vmcnt(15)
ds_write_b32 v96, v28 offset:41600 // (op 1179) Store local data 
s_barrier  // (op 4504) 



s_waitcnt vmcnt(12)
ds_write_b8 v95, v29 offset:42112 // (op 1178) Store local data 
s_barrier  // (op 4490) 



ds_read_b64_tr_b8 v[42:43], v17 offset:20480 // (op 1162) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v17 offset:20992 // (op 1162) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v17 offset:24576 // (op 1162) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v17 offset:25088 // (op 1162) Transpose load local data 



ds_read_b64_tr_b8 v[104:105], v17 offset:28672 // (op 1171) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v17 offset:29184 // (op 1171) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v17 offset:32768 // (op 1171) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v17 offset:33280 // (op 1171) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4507) 


ds_read_u8 v26, v16 offset:41600 // (op 1166) Load local data 


ds_read_u8 v97, v16 offset:41856 // (op 1175) Load local data 



ds_read_b64_tr_b8 v[58:59], v17 offset:20512 // (op 1345) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v17 offset:21024 // (op 1345) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v17 offset:24608 // (op 1345) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v17 offset:25120 // (op 1345) Transpose load local data 



ds_read_b64_tr_b8 v[112:113], v17 offset:28704 // (op 1336) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v17 offset:29216 // (op 1336) Transpose load local data 
ds_read_b64_tr_b8 v[116:117], v17 offset:32800 // (op 1336) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v17 offset:33312 // (op 1336) Transpose load local data 


ds_read_u8 v38, v16 offset:41632 // (op 1349) Load local data 


ds_read_u8 v98, v16 offset:41888 // (op 1340) Load local data 
s_barrier  // (op 4481) 
s_barrier  // (op 4699) 



ds_read_b64_tr_b8 v[74:75], v7 offset:36864 // (op 1160) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v7 offset:36992 // (op 1160) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v7 offset:37888 // (op 1160) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v7 offset:38016 // (op 1160) Transpose load local data 



ds_read_b64_tr_b8 v[120:121], v7 offset:38912 // (op 1169) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v7 offset:39040 // (op 1169) Transpose load local data 
ds_read_b64_tr_b8 v[124:125], v7 offset:39936 // (op 1169) Transpose load local data 
ds_read_b64_tr_b8 v[126:127], v7 offset:40064 // (op 1169) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4500) 


ds_read_u8 v33, v10 offset:42112 // (op 1164) Load local data 


ds_read_u8 v90, v10 offset:42176 // (op 1173) Load local data 
s_barrier  // (op 4663) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:81], v[42:49], a[0:3], v33, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1167) 
s_barrier  // (op 4497) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:81], v[58:65], a[4:7], v33, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1332) 
s_lshl_b64 s[84:85], s[34:35], 5 // (op 2628) 
v_mov_b64 v[128:129], s[84:85] // (op 2628) convert


s_lshl_b64 s[84:85], s[34:35], 8 // (op 2628) 
s_mul_i32 s87, 3, s85 // (op 2628) most significant: low of low * high
s_mul_i32 s86, 3, s84 // (op 2628) least significant
s_mul_hi_u32 s88, 3, s84 // (op 2628) most significant: high of low * low
s_add_u32 s87, s87, s88 // (op 2628) most significant: sum


v_mov_b32 v99, s86 // (op 2628) 
v_mov_b32 v130, s87 // (op 2628) 
v_add_co_u32 v132, s[84:85], v100, v99 // (op 2628) least significant half
v_addc_co_u32 v133, s[84:85], v101, v130, s[84:85] // (op 2628) most significant half




s_add_u32 s84, 20480, s55 // (op 2628) 
s_nop 0
s_mov_b32 m0, s84 // (op 2628) reg expression
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2628) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2628) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2628) most significant half

s_add_u32 m0, m0, 2048 // (op 2628) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2628) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2628) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2628) most significant half

s_add_u32 m0, m0, 2048 // (op 2628) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2628) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2628) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2628) most significant half

s_add_u32 m0, m0, 2048 // (op 2628) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:127], v[104:111], a[0:3], v90, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1176) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2628) Load value direct to lds


v_add_co_u32 v132, s[86:87], v132, v128 // (op 2628) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:127], v[112:119], a[4:7], v90, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1341) 
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2628) most significant half




s_add_u32 m0, m0, 2048 // (op 2628) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2628) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2628) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2628) most significant half

s_add_u32 m0, m0, 2048 // (op 2628) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2628) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2628) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2628) most significant half

s_add_u32 m0, m0, 2048 // (op 2628) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2628) Load value direct to lds
v_add_co_u32 v132, s[86:87], v132, v128 // (op 2628) least significant half
v_addc_co_u32 v133, s[86:87], 0, v133, s[86:87] // (op 2628) most significant half

s_add_u32 m0, m0, 2048 // (op 2628) 
buffer_load_dwordx4 v132, s[72:75], 0 offen offset: 0 lds // (op 2628) Load value direct to lds






s_lshl_b32 s84, s44, 3 // (op 1182) 
s_mul_i32 s85, 3, s84 // (op 1182) 

v_add_u32 v97, v91, s85 // (op 1182) 


buffer_load_dword v28, v97, s[80:83], 0 offen // (op 1182) Load value

s_lshl_b64 s[84:85], s[34:35], 8 // (op 2908) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 2908) 

s_barrier  // (op 4487) 
s_lshl_b32 s84, s44, 4 // (op 3164) 

v_add_u32 v39, v39, s84 // (op 3164) 

v_mov_b32 v91, s86 // (op 2908) 
v_mov_b32 v97, s87 // (op 2908) 
s_lshl_b64 s[84:85], s[30:31], 7 // (op 2623) 
v_mov_b64 v[98:99], s[84:85] // (op 2623) convert


s_lshl_b64 s[84:85], s[30:31], 8 // (op 2623) 
s_mul_i32 s89, 3, s85 // (op 2623) most significant: low of low * high
s_mul_i32 s88, 3, s84 // (op 2623) least significant
s_mul_hi_u32 s90, 3, s84 // (op 2623) most significant: high of low * low
s_add_u32 s89, s89, s90 // (op 2623) most significant: sum


v_mov_b32 v100, s88 // (op 2623) 
v_mov_b32 v101, s89 // (op 2623) 
v_add_co_u32 v30, s[84:85], v30, v91 // (op 2908) least significant half
v_addc_co_u32 v31, s[84:85], v31, v97, s[84:85] // (op 2908) most significant half




v_add_co_u32 v104, s[84:85], v102, v100 // (op 2623) least significant half
v_addc_co_u32 v105, s[84:85], v103, v101, s[84:85] // (op 2623) most significant half




s_add_u32 s84, 36864, s54 // (op 2623) 
s_nop 0
s_mov_b32 m0, s84 // (op 2623) reg expression
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 2623) Load value direct to lds
v_add_co_u32 v104, s[86:87], v104, v98 // (op 2623) least significant half
v_addc_co_u32 v105, s[86:87], 0, v105, s[86:87] // (op 2623) most significant half

s_add_u32 m0, m0, 2048 // (op 2623) 
buffer_load_dwordx4 v104, s[68:71], 0 offen offset: 0 lds // (op 2623) Load value direct to lds






s_lshl_b32 s84, s38, 3 // (op 1181) 
s_mul_i32 s85, 3, s84 // (op 1181) 

v_add_u32 v91, v92, s85 // (op 1181) 


buffer_load_ubyte v29, v91, s[76:79], 0 offen offset: 0 // (op 1181) Load value

s_lshl_b64 s[84:85], s[30:31], 8 // (op 2780) 

s_lshl_b64 s[86:87], s[84:85], 1 // (op 2780) 

s_barrier  // (op 4355) 
s_lshl_b32 s84, s38, 4 // (op 3036) 



v_add_u32 v37, v37, s84 // (op 3036) 

v_mov_b32 v91, s86 // (op 2780) 
v_mov_b32 v92, s87 // (op 2780) 
v_add_co_u32 v34, s[84:85], v34, v91 // (op 2780) least significant half
v_addc_co_u32 v35, s[84:85], v35, v92, s[84:85] // (op 2780) most significant half




s_waitcnt vmcnt(15)
ds_write_b32 v94, v40 offset:40960 // (op 30) Store local data 
s_barrier  // (op 4334) 



s_waitcnt vmcnt(12)
ds_write_b8 v93, v41 offset:41472 // (op 21) Store local data 
s_barrier  // (op 4320) 



ds_read_b64_tr_b8 v[42:43], v11 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v11 offset:512 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v11 offset:4096 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v11 offset:4608 // (op 77) Transpose load local data 



ds_read_b64_tr_b8 v[50:51], v11 offset:8192 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v11 offset:8704 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v11 offset:12288 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v11 offset:12800 // (op 97) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4362) 


ds_read_u8 v26, v9 offset:40960 // (op 87) Load local data 


ds_read_u8 v36, v9 offset:41216 // (op 105) Load local data 



ds_read_b64_tr_b8 v[58:59], v11 offset:32 // (op 1317) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v11 offset:544 // (op 1317) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v11 offset:4128 // (op 1317) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v11 offset:4640 // (op 1317) Transpose load local data 



ds_read_b64_tr_b8 v[66:67], v11 offset:8224 // (op 1326) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v11 offset:8736 // (op 1326) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v11 offset:12320 // (op 1326) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v11 offset:12832 // (op 1326) Transpose load local data 


ds_read_u8 v38, v9 offset:40992 // (op 1321) Load local data 


ds_read_u8 v32, v9 offset:41248 // (op 1330) Load local data 
s_barrier  // (op 4301) 



ds_read_b64_tr_b8 v[74:75], v0 offset:16384 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:16512 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v0 offset:17408 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v0 offset:17536 // (op 72) Transpose load local data 



ds_read_b64_tr_b8 v[82:83], v0 offset:18432 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v0 offset:18560 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:19456 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v0 offset:19584 // (op 93) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4341) 


ds_read_u8 v33, v4 offset:41472 // (op 82) Load local data 


ds_read_u8 v90, v4 offset:41536 // (op 101) Load local data 
s_and_b32 s84, s7, 4294967294 // (op 1274) 
s_cmp_lt_u32 s1, s84 // (op 1274) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_1274 // (op 1274) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_1274 if true)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_1274:



















s_barrier  // (op 4681) 
s_and_b32 s52, s7, 4294967294 // (op 2696) 
s_add_u32 s53, s0, s52 // (op 2696) 

s_mul_i32 s52, s16, s8 // (op 2696) 
s_add_u32 s52, s52, s53 // (op 2696) 

s_mul_hi_u32 s53, s52, s13 // (op 2696) 
s_sub_u32 s54, s52, s53 // (op 2696) 
s_lshr_b32 s55, s54, 1 // (op 2696) 

s_add_u32 s54, s55, s53 // (op 2696) 


s_lshr_b32 s53, s54, s11 // (op 2696) 

s_and_b32 s54, s11, 2147483648 // (op 2696) 
s_cmp_eq_i32 s54, 0 // (op 2696) 
s_mov_b32 s55, scc // (op 2696) 

s_cmp_eq_i32 s55, 0 // (op 2696) 
s_cselect_b32 s54, s52, s53 // (op 2696) 


s_mul_hi_u32 s53, s54, s12 // (op 2696) 
s_sub_u32 s55, s54, s53 // (op 2696) 
s_lshr_b32 s84, s55, 1 // (op 2696) 

s_add_u32 s55, s84, s53 // (op 2696) 


s_lshr_b32 s53, s55, s10 // (op 2696) 

s_and_b32 s55, s10, 2147483648 // (op 2696) 
s_cmp_eq_i32 s55, 0 // (op 2696) 
s_mov_b32 s84, scc // (op 2696) 

s_cmp_eq_i32 s84, 0 // (op 2696) 
s_cselect_b32 s55, s54, s53 // (op 2696) 


s_lshl_b32 s53, s55, 4 // (op 2696) 

s_mov_b32 s85, 0 // (op 2696) convert
s_mov_b32 s84, s53 // (op 2696) convert

s_ashr_i64 s[86:87], s[2:3], 8 // (op 2696) 
s_mul_i32 s53, s54, s86 // (op 2696) 


s_sub_u32 s54, s52, s53 // (op 2696) 


v_lshl_add_u32 v26, s54, 8, v1 // (op 2696) 

v_mul_lo_u32 v31, v26, s31 // (op 2696) most significant: low of low * high
v_mul_hi_u32 v34, v26, s30 // (op 2696) most significant: high of low * low
v_mul_lo_u32 v30, v26, s30 // (op 2696) least significant: low of low * low
v_add_u32 v31, v31, v34 // (op 2696) most significant: sum


v_mov_b32 v26, s84 // (op 2696) 
v_mov_b32 v34, s85 // (op 2696) 
v_add_co_u32 v36, s[52:53], v26, v30 // (op 2696) least significant half
v_addc_co_u32 v37, s[52:53], v34, v31, s[52:53] // (op 2696) most significant half





s_mul_i32 s53, s30, s3 // (op 2698) most significant: low of low * high
s_mul_i32 s54, s31, s2 // (op 2698) most significant: low of high * low
s_mul_i32 s52, s30, s2 // (op 2698) least significant
s_mul_hi_u32 s55, s30, s2 // (op 2698) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2698) most significant: sum
s_add_u32 s53, s53, s55 // (op 2698) most significant: sum


s_mov_b32 s68, s32 // (op 2698) reg expression
s_mov_b32 s69, s33 // (op 2698) reg expression
s_mov_b32 s70, s52 // (op 2698) reg expression
s_mov_b32 s71, 131072 // (op 2698) call()

s_and_b32 s52, s7, 4294967294 // (op 2832) 
v_and_b32 v26, -4, v1 // (op 2832) 
s_add_u32 s53, s0, s52 // (op 2832) 

v_sub_u32 v30, v1, v26 // (op 2832) 

v_lshlrev_b32 v26, 4, v30 // (op 2832) 

s_mul_i32 s52, s16, s8 // (op 2832) 
s_add_u32 s52, s52, s53 // (op 2832) 

v_lshrrev_b32 v30, 2, v1 // (op 2832) 
s_mul_hi_u32 s53, s52, s13 // (op 2832) 
s_sub_u32 s54, s52, s53 // (op 2832) 
s_lshr_b32 s55, s54, 1 // (op 2832) 

s_add_u32 s54, s55, s53 // (op 2832) 


s_lshr_b32 s53, s54, s11 // (op 2832) 

s_and_b32 s54, s11, 2147483648 // (op 2832) 
s_cmp_eq_i32 s54, 0 // (op 2832) 
s_mov_b32 s55, scc // (op 2832) 

s_cmp_eq_i32 s55, 0 // (op 2832) 
s_cselect_b32 s54, s52, s53 // (op 2832) 


s_mul_hi_u32 s53, s54, s12 // (op 2832) 
s_sub_u32 s55, s54, s53 // (op 2832) 
s_lshr_b32 s84, s55, 1 // (op 2832) 

s_add_u32 s55, s84, s53 // (op 2832) 


s_lshr_b32 s53, s55, s10 // (op 2832) 

s_and_b32 s55, s10, 2147483648 // (op 2832) 
s_cmp_eq_i32 s55, 0 // (op 2832) 
s_mov_b32 s84, scc // (op 2832) 

s_cmp_eq_i32 s84, 0 // (op 2832) 
s_cselect_b32 s55, s54, s53 // (op 2832) 


s_mul_i32 s53, s55, s14 // (op 2832) 

s_sub_u32 s55, s54, s53 // (op 2832) 

v_lshl_add_u32 v31, s55, 6, v26 // (op 2832) 


s_ashr_i64 s[84:85], s[2:3], 8 // (op 2832) 
v_mov_b32 v35, 0 // (op 2832) convert
v_mov_b32 v34, v31 // (op 2832) convert

s_mul_i32 s53, s54, s84 // (op 2832) 


s_sub_u32 s54, s52, s53 // (op 2832) 


v_lshl_add_u32 v26, s54, 8, v30 // (op 2832) 


v_mul_lo_u32 v31, v26, s35 // (op 2832) most significant: low of low * high
v_mul_hi_u32 v39, v26, s34 // (op 2832) most significant: high of low * low
v_mul_lo_u32 v30, v26, s34 // (op 2832) least significant: low of low * low
v_add_u32 v31, v31, v39 // (op 2832) most significant: sum

v_add_co_u32 v30, s[52:53], v30, v34 // (op 2832) least significant half
v_addc_co_u32 v31, s[52:53], v31, v35, s[52:53] // (op 2832) most significant half



s_mul_i32 s53, s34, s3 // (op 2834) most significant: low of low * high
s_mul_i32 s54, s35, s2 // (op 2834) most significant: low of high * low
s_mul_i32 s52, s34, s2 // (op 2834) least significant
s_mul_hi_u32 s55, s34, s2 // (op 2834) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2834) most significant: sum
s_add_u32 s53, s53, s55 // (op 2834) most significant: sum


s_mov_b32 s72, s36 // (op 2834) reg expression
s_mov_b32 s73, s37 // (op 2834) reg expression
s_mov_b32 s74, s52 // (op 2834) reg expression
s_mov_b32 s75, 131072 // (op 2834) call()

s_and_b32 s52, s7, 4294967294 // (op 2960) 
v_and_b32 v26, -16, v1 // (op 2960) 
s_add_u32 s53, s0, s52 // (op 2960) 

v_sub_u32 v34, v1, v26 // (op 2960) 

v_lshrrev_b32 v26, 4, v1 // (op 2960) 
s_mul_i32 s52, s16, s8 // (op 2960) 
s_add_u32 s52, s52, s53 // (op 2960) 

s_mul_hi_u32 s53, s52, s13 // (op 2960) 
s_sub_u32 s54, s52, s53 // (op 2960) 
s_lshr_b32 s55, s54, 1 // (op 2960) 

s_add_u32 s54, s55, s53 // (op 2960) 


s_lshr_b32 s53, s54, s11 // (op 2960) 

s_and_b32 s54, s11, 2147483648 // (op 2960) 
s_cmp_eq_i32 s54, 0 // (op 2960) 
s_mov_b32 s55, scc // (op 2960) 

s_cmp_eq_i32 s55, 0 // (op 2960) 
s_cselect_b32 s54, s52, s53 // (op 2960) 


s_mul_hi_u32 s53, s54, s12 // (op 2960) 
s_sub_u32 s55, s54, s53 // (op 2960) 
s_lshr_b32 s84, s55, 1 // (op 2960) 

s_add_u32 s55, s84, s53 // (op 2960) 


s_lshr_b32 s53, s55, s10 // (op 2960) 

s_and_b32 s55, s10, 2147483648 // (op 2960) 
s_cmp_eq_i32 s55, 0 // (op 2960) 
s_mov_b32 s84, scc // (op 2960) 

s_cmp_eq_i32 s84, 0 // (op 2960) 
s_cselect_b32 s55, s54, s53 // (op 2960) 


v_lshl_add_u32 v35, s55, 4, v34 // (op 2960) 


s_ashr_i64 s[84:85], s[2:3], 8 // (op 2960) 
s_mul_i32 s53, s54, s84 // (op 2960) 


s_sub_u32 s54, s52, s53 // (op 2960) 


v_lshl_add_u32 v34, s54, 3, v26 // (op 2960) 


v_mul_lo_u32 v26, v34, s38 // (op 2960) 

v_add_u32 v34, v35, v26 // (op 2960) 


s_mul_i32 s53, s38, s43 // (op 2962) most significant: low of low * high
s_mul_i32 s54, s39, s42 // (op 2962) most significant: low of high * low
s_mul_i32 s52, s38, s42 // (op 2962) least significant
s_mul_hi_u32 s55, s38, s42 // (op 2962) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 2962) most significant: sum
s_add_u32 s53, s53, s55 // (op 2962) most significant: sum


s_mov_b32 s76, s40 // (op 2962) reg expression
s_mov_b32 s77, s41 // (op 2962) reg expression
s_mov_b32 s78, s52 // (op 2962) reg expression
s_mov_b32 s79, 131072 // (op 2962) call()

s_and_b32 s52, s7, 4294967294 // (op 3088) 
v_and_b32 v26, -16, v1 // (op 3088) 
s_add_u32 s53, s0, s52 // (op 3088) 

v_sub_u32 v35, v1, v26 // (op 3088) 

v_lshlrev_b32 v26, 2, v35 // (op 3088) 

s_mul_i32 s52, s16, s8 // (op 3088) 
s_add_u32 s52, s52, s53 // (op 3088) 

v_lshrrev_b32 v35, 4, v1 // (op 3088) 
s_mul_hi_u32 s53, s52, s13 // (op 3088) 
s_sub_u32 s54, s52, s53 // (op 3088) 
s_lshr_b32 s55, s54, 1 // (op 3088) 

s_add_u32 s54, s55, s53 // (op 3088) 


s_lshr_b32 s53, s54, s11 // (op 3088) 

s_and_b32 s54, s11, 2147483648 // (op 3088) 
s_cmp_eq_i32 s54, 0 // (op 3088) 
s_mov_b32 s55, scc // (op 3088) 

s_cmp_eq_i32 s55, 0 // (op 3088) 
s_cselect_b32 s54, s52, s53 // (op 3088) 


s_mul_hi_u32 s53, s54, s12 // (op 3088) 
s_sub_u32 s55, s54, s53 // (op 3088) 
s_lshr_b32 s84, s55, 1 // (op 3088) 

s_add_u32 s55, s84, s53 // (op 3088) 


s_lshr_b32 s53, s55, s10 // (op 3088) 

s_and_b32 s55, s10, 2147483648 // (op 3088) 
s_cmp_eq_i32 s55, 0 // (op 3088) 
s_mov_b32 s84, scc // (op 3088) 

s_cmp_eq_i32 s84, 0 // (op 3088) 
s_cselect_b32 s55, s54, s53 // (op 3088) 


s_mul_i32 s53, s55, s14 // (op 3088) 

s_sub_u32 s55, s54, s53 // (op 3088) 

v_lshl_add_u32 v39, s55, 6, v26 // (op 3088) 


s_ashr_i64 s[84:85], s[2:3], 8 // (op 3088) 
s_mul_i32 s53, s54, s84 // (op 3088) 


s_sub_u32 s54, s52, s53 // (op 3088) 


v_lshl_add_u32 v26, s54, 3, v35 // (op 3088) 


v_mul_lo_u32 v35, v26, s44 // (op 3088) 
v_add_u32 v35, v35, v39 // (op 3088) 


s_mul_i32 s53, s44, s43 // (op 3090) most significant: low of low * high
s_mul_i32 s54, s45, s42 // (op 3090) most significant: low of high * low
s_mul_i32 s52, s44, s42 // (op 3090) least significant
s_mul_hi_u32 s55, s44, s42 // (op 3090) most significant: high of low * low
s_add_u32 s53, s53, s54 // (op 3090) most significant: sum
s_add_u32 s53, s53, s55 // (op 3090) most significant: sum


s_mov_b32 s80, s46 // (op 3090) reg expression
s_mov_b32 s81, s47 // (op 3090) reg expression
s_mov_b32 s82, s52 // (op 3090) reg expression
s_mov_b32 s83, 131072 // (op 3090) call()

v_lshlrev_b32 v26, 4, v1 // (op 3249) 
s_nop 0
v_readfirstlane_b32 s52, v26 // (op 3249) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -4, v1 // (op 3314) 
v_sub_u32 v39, v1, v26 // (op 3314) 

v_lshlrev_b32 v26, 4, v39 // (op 3314) 

v_lshrrev_b32 v39, 2, v1 // (op 3314) 
v_lshl_add_u32 v40, v39, 6, v26 // (op 3314) 


s_nop 0
v_readfirstlane_b32 s53, v40 // (op 3314) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v26, -16, v1 // (op 3389) 
v_sub_u32 v39, v1, v26 // (op 3389) 

v_and_b32 v26, 4294967280, v1 // (op 3389) 
v_add_u32 v40, v26, v39 // (op 3389) 


v_and_b32 v26, -16, v1 // (op 3455) 
v_sub_u32 v39, v1, v26 // (op 3455) 

v_lshlrev_b32 v26, 2, v39 // (op 3455) 

v_lshrrev_b32 v39, 4, v1 // (op 3455) 
v_lshl_add_u32 v41, v39, 6, v26 // (op 3455) 


s_and_b32 s54, s7, 4294967294 // (op 1280) 
s_cmp_lt_u32 s54, s7 // (op 1279) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_1279 // (op 1279) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_1279 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_1279:
s_barrier  // (op 4298) 
s_add_u32 s54, s54, 1 // (op 1281) 

s_lshl_b64 s[84:85], s[30:31], 7 // (op 2610) 
v_mov_b64 v[28:29], s[84:85] // (op 2610) convert


v_mov_b64 v[32:33], v[36:37] // (op 2610) 
v_mov_b64 v[38:39], v[32:33] // (op 2610) reg expression
s_add_u32 s55, 16384, s52 // (op 2610) 
s_mov_b32 m0, s55 // (op 2610) reg expression
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 2610) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2610) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2610) most significant half

s_add_u32 m0, m0, 2048 // (op 2610) 
buffer_load_dwordx4 v38, s[68:71], 0 offen offset: 0 lds // (op 2610) Load value direct to lds




s_lshl_b64 s[84:85], s[30:31], 8 // (op 2686) 
v_mov_b32 v26, s84 // (op 2686) 
v_mov_b32 v28, s85 // (op 2686) 
v_add_co_u32 v36, s[86:87], v36, v26 // (op 2686) least significant half
v_addc_co_u32 v37, s[86:87], v37, v28, s[86:87] // (op 2686) most significant half




s_barrier  // (op 4317) 

s_lshl_b64 s[84:85], s[34:35], 5 // (op 2616) 
v_mov_b64 v[28:29], s[84:85] // (op 2616) convert


v_mov_b64 v[32:33], v[30:31] // (op 2616) 
v_mov_b64 v[38:39], v[32:33] // (op 2616) reg expression
s_mov_b32 m0, s53 // (op 2616) reg expression
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2616) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2616) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2616) most significant half

s_add_u32 m0, m0, 2048 // (op 2616) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2616) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2616) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2616) most significant half

s_add_u32 m0, m0, 2048 // (op 2616) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2616) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2616) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2616) most significant half

s_add_u32 m0, m0, 2048 // (op 2616) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2616) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2616) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2616) most significant half

s_add_u32 m0, m0, 2048 // (op 2616) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2616) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2616) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2616) most significant half

s_add_u32 m0, m0, 2048 // (op 2616) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2616) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2616) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2616) most significant half

s_add_u32 m0, m0, 2048 // (op 2616) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2616) Load value direct to lds
v_add_co_u32 v38, s[84:85], v38, v28 // (op 2616) least significant half
v_addc_co_u32 v39, s[84:85], 0, v39, s[84:85] // (op 2616) most significant half

s_add_u32 m0, m0, 2048 // (op 2616) 
buffer_load_dwordx4 v38, s[72:75], 0 offen offset: 0 lds // (op 2616) Load value direct to lds



s_lshl_b64 s[84:85], s[34:35], 8 // (op 2824) 
v_mov_b32 v26, s84 // (op 2824) 
v_mov_b32 v28, s85 // (op 2824) 
v_add_co_u32 v30, s[86:87], v30, v26 // (op 2824) least significant half
v_addc_co_u32 v31, s[86:87], v31, v28, s[86:87] // (op 2824) most significant half






v_mov_b32 v26, v34 // (op 1097) 
v_mov_b32 v28, v26 // (op 1097) reg expression

buffer_load_ubyte v29, v28, s[76:79], 0 offen offset: 0 // (op 1097) Load value


s_lshl_b32 s55, s38, 3 // (op 2952) 
v_add_u32 v34, v34, s55 // (op 2952) 



s_barrier  // (op 4337) 
v_mov_b32 v26, v35 // (op 1098) 



v_mov_b32 v28, v26 // (op 1098) reg expression

buffer_load_dword v32, v28, s[80:83], 0 offen // (op 1098) Load value


s_lshl_b32 s55, s44, 3 // (op 3080) 
v_add_u32 v35, v35, s55 // (op 3080) 

s_waitcnt vmcnt(1)
ds_write_b8 v40, v29 offset:41472 // (op 1094) Store local data 

s_barrier  // (op 4304) 



s_barrier  // (op 4358) 
ds_read_b64_tr_b8 v[42:43], v0 offset:16384 // (op 1076) Transpose load local data 



ds_read_b64_tr_b8 v[44:45], v0 offset:16512 // (op 1076) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:17408 // (op 1076) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v0 offset:17536 // (op 1076) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v41, v32 offset:40960 // (op 1095) Store local data 

s_barrier  // (op 4324) 



ds_read_b64_tr_b8 v[50:51], v11 // (op 1078) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v11 offset:512 // (op 1078) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v11 offset:4096 // (op 1078) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v11 offset:4608 // (op 1078) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4345) 


ds_read_u8 v26, v4 offset:41472 // (op 1080) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 4365) 


ds_read_u8 v28, v9 offset:40960 // (op 1082) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:57], a[0:3], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1083) 





ds_read_b64_tr_b8 v[50:51], v0 offset:18432 // (op 1085) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:18560 // (op 1085) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:19456 // (op 1085) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:19584 // (op 1085) Transpose load local data 



ds_read_b64_tr_b8 v[58:59], v11 offset:8192 // (op 1087) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v11 offset:8704 // (op 1087) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v11 offset:12288 // (op 1087) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v11 offset:12800 // (op 1087) Transpose load local data 


ds_read_u8 v28, v4 offset:41536 // (op 1089) Load local data 


ds_read_u8 v29, v9 offset:41216 // (op 1091) Load local data 



ds_read_b64_tr_b8 v[66:67], v11 offset:32 // (op 1287) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v11 offset:544 // (op 1287) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v11 offset:4128 // (op 1287) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v11 offset:4640 // (op 1287) Transpose load local data 


ds_read_u8 v32, v9 offset:40992 // (op 1291) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[66:73], a[4:7], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1292) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[58:65], a[0:3], v28, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1092) 
ds_read_b64_tr_b8 v[42:43], v11 offset:8224 // (op 1296) Transpose load local data 


ds_read_b64_tr_b8 v[44:45], v11 offset:8736 // (op 1296) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v11 offset:12320 // (op 1296) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v11 offset:12832 // (op 1296) Transpose load local data 


ds_read_u8 v26, v9 offset:41248 // (op 1300) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[42:49], a[4:7], v28, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1301) 




s_cmp_lt_u32 s54, s7 // (op 1279) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_1279 // (op 1279) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_1279 if true)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_1279:









s_barrier  // (op 4685) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_add_u32 s0, s0, s7 // (op 336) 
s_cmp_lg_u32 s18, 0 // (op 362) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Send_Tile_362 // (op 362) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Send_Tile_362



buffer_store_dword a0, v25, s[64:67], 0 offen sc1 // (op 359) Store value
buffer_store_dword a1, v25, s[64:67], 0 offen offset:256 sc1 // (op 359) Store value
buffer_store_dword a2, v25, s[64:67], 0 offen offset:512 sc1 // (op 359) Store value
buffer_store_dword a3, v25, s[64:67], 0 offen offset:768 sc1 // (op 359) Store value



buffer_store_dword a4, v25, s[64:67], 0 offen offset:128 sc1 // (op 1584) Store value
buffer_store_dword a5, v25, s[64:67], 0 offen offset:384 sc1 // (op 1584) Store value
buffer_store_dword a6, v25, s[64:67], 0 offen offset:640 sc1 // (op 1584) Store value
buffer_store_dword a7, v25, s[64:67], 0 offen offset:896 sc1 // (op 1584) Store value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_barrier  // (op 363) 
v_cmp_eq_i32 vcc, v1, 0 // (op 367) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Store_Flag_367 // (op 367) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Store_Flag_367
s_mov_b32 s52, 1 // (op 365) call()
s_mov_b32 s55, 0 // (op 366) convert
s_mov_b32 s54, s16 // (op 366) convert
s_lshl_b64 s[54:55], s[54:55], 2 // (op 366) 
s_add_u32 s84, s48, s54 // (op 366) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 366) End of Int64 add, unlocking SCC
s_store_dword s52, s[84:85], 0 glc // (op 366) Store scalar value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Store_Flag_367 // (op 367) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Store_Flag_367
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Store_Flag_367:
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Store_Flag_367:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Send_Tile_362 // (op 362) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Send_Tile_362
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Send_Tile_362:
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Send_Tile_362:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_cmp_eq_i32 s18, 0 // (op 378) 
s_mov_b32 s52, scc // (op 378) 
s_ashr_i64 s[54:55], s[2:3], 8 // (op 378) 
s_mov_b32 s53, s54 // (op 378) convert

s_sub_u32 s54, s53, 1 // (op 378) 

s_cmp_lt_u32 s19, s54 // (op 378) 
s_mov_b32 s53, scc // (op 378) 

s_and_b32 s54, s52, s53 // (op 378) 
s_cmp_lg_u32 s54, 0 // (op 378) 



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Receive_Tile_378 // (op 378) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Receive_Tile_378
s_mov_b32 s52, 0 // (op 344) call()
s_mul_i32 s53, s52, s8 // (op 343) 
s_ashr_i64 s[54:55], s[2:3], 8 // (op 343) 
s_mov_b32 s84, s54 // (op 343) convert

s_sub_u32 s54, s84, s19 // (op 343) 

s_sub_u32 s55, s54, 1 // (op 343) 

s_cmp_lt_u32 s53, s55 // (op 343) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_ReceiveTileLoop_343 // (op 343) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_ReceiveTileLoop_343 if false)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_ReceiveTileLoop_343:
s_add_u32 s53, s16, 1 // (op 3869) 
v_lshrrev_b32 v20, 5, v1 // (op 3869) 
s_add_u32 s54, s53, s52 // (op 3869) 
s_lshl_b32 s54, s54, 4 // (op 3869) 

v_lshlrev_b32 v21, 2, v20 // (op 3869) 

v_add_lshl_u32 v20, s54, v21, 6 // (op 3869) 


v_and_b32 v21, -32, v1 // (op 3869) 
v_sub_u32 v23, v1, v21 // (op 3869) 

v_add_lshl_u32 v21, v20, v23, 2 // (op 3869) 


s_lshl_b32 s53, s15, 12 // (op 3871) 
s_mov_b32 s64, s28 // (op 3871) reg expression
s_mov_b32 s65, s29 // (op 3871) reg expression
s_mov_b32 s66, s53 // (op 3871) reg expression
s_mov_b32 s67, 131072 // (op 3871) call()

RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_DoWhileTop_Global_sync_spin_loop_382:
s_add_u32 s53, s16, 1 // (op 380) 
s_add_u32 s54, s53, s52 // (op 380) Start of Int64 add, locking SCC
s_addc_u32 s55, 0, 0 // (op 380) End of Int64 add, unlocking SCC
s_lshl_b64 s[54:55], s[54:55], 2 // (op 380) 

s_add_u32 s84, s48, s54 // (op 380) Start of Int64 add, locking SCC
s_addc_u32 s85, s49, s55 // (op 380) End of Int64 add, unlocking SCC
s_load_dword s53, s[84:85], 0 glc // (op 380) Load scalar value


s_waitcnt lgkmcnt(0)
s_cmp_eq_i32 s53, 0 // (op 382) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_DoWhileTop_Global_sync_spin_loop_382 // (op 382) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_DoWhileTop_Global_sync_spin_loop_382 if true)
s_barrier  // (op 386) 
v_cmp_eq_i32 vcc, v1, 0 // (op 385) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_385 // (op 385) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_385
s_mov_b32 s53, 0 // (op 383) call()
s_add_u32 s54, s16, 1 // (op 384) 
s_add_u32 s84, s54, s52 // (op 384) Start of Int64 add, locking SCC
s_addc_u32 s85, 0, 0 // (op 384) End of Int64 add, unlocking SCC
s_lshl_b64 s[84:85], s[84:85], 2 // (op 384) 

s_add_u32 s54, s48, s84 // (op 384) Start of Int64 add, locking SCC
s_addc_u32 s55, s49, s85 // (op 384) End of Int64 add, unlocking SCC
s_store_dword s53, s[54:55], 0 glc // (op 384) Store scalar value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_385 // (op 385) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_385
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Wave0_Reset_Flag_385:
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Wave0_Reset_Flag_385:




buffer_load_dword v28, v21, s[64:67], 0 offen // (op 397) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:256 // (op 397) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:512 // (op 397) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:768 // (op 397) Load value
v_accvgpr_read v32, a0 // (op 400) reg expression
v_accvgpr_read v33, a1 // (op 400) reg expression
v_accvgpr_read v34, a2 // (op 400) reg expression
v_accvgpr_read v35, a3 // (op 400) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 402) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 402) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 402) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 402) 

v_accvgpr_write a0, v32 // (op 404) reg expression
v_accvgpr_write a1, v33 // (op 404) reg expression
v_accvgpr_write a2, v34 // (op 404) reg expression
v_accvgpr_write a3, v35 // (op 404) reg expression




buffer_load_dword v28, v21, s[64:67], 0 offen offset:128 // (op 1594) Load value
buffer_load_dword v29, v21, s[64:67], 0 offen offset:384 // (op 1594) Load value
buffer_load_dword v30, v21, s[64:67], 0 offen offset:640 // (op 1594) Load value
buffer_load_dword v31, v21, s[64:67], 0 offen offset:896 // (op 1594) Load value
v_accvgpr_read v32, a4 // (op 1595) reg expression
v_accvgpr_read v33, a5 // (op 1595) reg expression
v_accvgpr_read v34, a6 // (op 1595) reg expression
v_accvgpr_read v35, a7 // (op 1595) reg expression
s_waitcnt vmcnt(3)
v_add_f32 v32, v32, v28 // (op 1596) 
s_waitcnt vmcnt(2)
v_add_f32 v33, v33, v29 // (op 1596) 
s_waitcnt vmcnt(1)
v_add_f32 v34, v34, v30 // (op 1596) 
s_waitcnt vmcnt(0)
v_add_f32 v35, v35, v31 // (op 1596) 

v_accvgpr_write a4, v32 // (op 1597) reg expression
v_accvgpr_write a5, v33 // (op 1597) reg expression
v_accvgpr_write a6, v34 // (op 1597) reg expression
v_accvgpr_write a7, v35 // (op 1597) reg expression

s_add_u32 s52, s52, 1 // (op 345) 

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// Explicit WaitZero operation
s_mul_i32 s53, s52, s8 // (op 343) 
s_ashr_i64 s[54:55], s[2:3], 8 // (op 343) 
s_mov_b32 s84, s54 // (op 343) convert

s_sub_u32 s54, s84, s19 // (op 343) 

s_sub_u32 s55, s54, 1 // (op 343) 

s_cmp_lt_u32 s53, s55 // (op 343) 


s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_ReceiveTileLoop_343 // (op 343) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_ReceiveTileLoop_343 if true)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_ReceiveTileLoop_343:

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Receive_Tile_378 // (op 378) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Receive_Tile_378
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Receive_Tile_378:
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Receive_Tile_378:

s_cmp_eq_i32 s18, 0 // (op 421) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_421 // (op 421) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_421
v_cmp_eq_f32 vcc, 0.00000, v18 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_157
v_accvgpr_read v28, a0 // (op 50) 
v_accvgpr_read v29, a1 // (op 50) 
v_accvgpr_read v30, a2 // (op 50) 
v_accvgpr_read v31, a3 // (op 50) 
v_mul_f32 v32, v19, v28 // (op 50) 
v_mul_f32 v33, v19, v29 // (op 50) 
v_mul_f32 v34, v19, v30 // (op 50) 
v_mul_f32 v35, v19, v31 // (op 50) 

v_cvt_f16_f32 v28, v32 // (op 56) 
v_cvt_f16_f32 v29, v33 // (op 56) 
v_cvt_f16_f32 v30, v34 // (op 56) 
v_cvt_f16_f32 v31, v35 // (op 56) 


v_pack_b32_f16 v20, v28, v29 // (op 58) 
v_pack_b32_f16 v21, v30, v31 // (op 58) 

v_mov_b32 v23, v27 // (op 58) reg expression

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 58) Store value



v_accvgpr_read v28, a4 // (op 1613) 
v_accvgpr_read v29, a5 // (op 1613) 
v_accvgpr_read v30, a6 // (op 1613) 
v_accvgpr_read v31, a7 // (op 1613) 
v_mul_f32 v32, v19, v28 // (op 1613) 
v_mul_f32 v33, v19, v29 // (op 1613) 
v_mul_f32 v34, v19, v30 // (op 1613) 
v_mul_f32 v35, v19, v31 // (op 1613) 

v_cvt_f16_f32 v28, v32 // (op 1615) 
v_cvt_f16_f32 v29, v33 // (op 1615) 
v_cvt_f16_f32 v30, v34 // (op 1615) 
v_cvt_f16_f32 v31, v35 // (op 1615) 


v_pack_b32_f16 v20, v28, v29 // (op 1616) 
v_pack_b32_f16 v21, v30, v31 // (op 1616) 

s_lshl_b32 s52, s24, 6 // (op 1616) 
v_add_u32 v23, v27, s52 // (op 1616) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 1616) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_157:


v_mov_b32 v20, v22 // (op 170) reg expression

buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 170) Load value

v_accvgpr_read v28, a0 // (op 166) 
v_accvgpr_read v29, a1 // (op 166) 
v_accvgpr_read v30, a2 // (op 166) 
v_accvgpr_read v31, a3 // (op 166) 
v_mul_f32 v32, v19, v28 // (op 166) 
v_mul_f32 v33, v19, v29 // (op 166) 
v_mul_f32 v34, v19, v30 // (op 166) 
v_mul_f32 v35, v19, v31 // (op 166) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 171) 
v_lshrrev_b32 v21, 16, v24 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v28, v18, v20 // (op 171) 
v_mul_f32 v29, v18, v21 // (op 171) 
v_cvt_f32_f16 v20, v25 // (op 171) 
v_lshrrev_b32 v21, 16, v25 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v30, v18, v20 // (op 171) 
v_mul_f32 v31, v18, v21 // (op 171) 


v_add_f32 v36, v28, v32 // (op 167) 
v_add_f32 v37, v29, v33 // (op 167) 
v_add_f32 v38, v30, v34 // (op 167) 
v_add_f32 v39, v31, v35 // (op 167) 


v_cvt_f16_f32 v28, v36 // (op 168) 
v_cvt_f16_f32 v29, v37 // (op 168) 
v_cvt_f16_f32 v30, v38 // (op 168) 
v_cvt_f16_f32 v31, v39 // (op 168) 


v_pack_b32_f16 v20, v28, v29 // (op 169) 
v_pack_b32_f16 v21, v30, v31 // (op 169) 

v_mov_b32 v23, v27 // (op 169) reg expression

buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 169) Store value





s_lshl_b32 s52, s4, 6 // (op 1640) 
v_add_u32 v20, v22, s52 // (op 1640) 


buffer_load_dwordx2 v[24:25], v20, s[56:59], 0 offen // (op 1640) Load value

v_accvgpr_read v28, a4 // (op 1636) 
v_accvgpr_read v29, a5 // (op 1636) 
v_accvgpr_read v30, a6 // (op 1636) 
v_accvgpr_read v31, a7 // (op 1636) 
v_mul_f32 v32, v19, v28 // (op 1636) 
v_mul_f32 v33, v19, v29 // (op 1636) 
v_mul_f32 v34, v19, v30 // (op 1636) 
v_mul_f32 v35, v19, v31 // (op 1636) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v24 // (op 1641) 
v_lshrrev_b32 v21, 16, v24 // (op 1641) 
v_cvt_f32_f16 v21, v21 // (op 1641) 
v_mul_f32 v28, v18, v20 // (op 1641) 
v_mul_f32 v29, v18, v21 // (op 1641) 
v_cvt_f32_f16 v20, v25 // (op 1641) 
v_lshrrev_b32 v21, 16, v25 // (op 1641) 
v_cvt_f32_f16 v21, v21 // (op 1641) 
v_mul_f32 v30, v18, v20 // (op 1641) 
v_mul_f32 v31, v18, v21 // (op 1641) 


v_add_f32 v36, v28, v32 // (op 1637) 
v_add_f32 v37, v29, v33 // (op 1637) 
v_add_f32 v38, v30, v34 // (op 1637) 
v_add_f32 v39, v31, v35 // (op 1637) 


v_cvt_f16_f32 v28, v36 // (op 1638) 
v_cvt_f16_f32 v29, v37 // (op 1638) 
v_cvt_f16_f32 v30, v38 // (op 1638) 
v_cvt_f16_f32 v31, v39 // (op 1638) 


v_pack_b32_f16 v20, v28, v29 // (op 1639) 
v_pack_b32_f16 v21, v30, v31 // (op 1639) 

s_lshl_b32 s52, s24, 6 // (op 1639) 
v_add_u32 v23, v27, s52 // (op 1639) 


buffer_store_dwordx2 v[20:21], v23, s[60:63], 0 offen // (op 1639) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_157:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_421 // (op 421) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_421
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_Postmaccumulation_Condition_421:
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_Postmaccumulation_Condition_421:


s_mul_i32 s52, s8, s16 // (op 334) 
s_cmp_gt_u32 s9, s52 // (op 334) 
s_mov_b32 s53, scc // (op 334) 
s_sub_u32 s54, s9, s52 // (op 334) 

s_cmp_eq_i32 s53, 0 // (op 334) 
s_cselect_b32 s52, 0, s54 // (op 334) 


s_cmp_lt_u32 s8, s52 // (op 334) 
s_mov_b32 s53, scc // (op 334) 
s_cmp_eq_i32 s53, 0 // (op 334) 
s_cselect_b32 s54, s52, s8 // (op 334) 


s_cmp_lt_u32 s0, s54 // (op 334) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_SKStreamTileLoop_334 // (op 334) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_SKStreamTileLoop_334 if true)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_SKStreamTileLoop_334:

















s_mov_b32 s0, 0 // (op 475) call()
s_mov_b32 s1, 0 // (op 476) call()
s_mov_b32 s0, 0 // (op 490) call()
s_mul_i32 s15, s50, s16 // (op 489) 
s_cmp_gt_u32 s51, s15 // (op 489) 
s_mov_b32 s6, scc // (op 489) 
s_sub_u32 s7, s51, s15 // (op 489) 

s_cmp_eq_i32 s6, 0 // (op 489) 
s_cselect_b32 s8, 0, s7 // (op 489) 


s_cmp_lt_u32 s50, s8 // (op 489) 
s_mov_b32 s15, scc // (op 489) 
s_cmp_eq_i32 s15, 0 // (op 489) 
s_cselect_b32 s6, s8, s50 // (op 489) 


s_cmp_lt_u32 s0, s6 // (op 489) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_DPStreamTileLoop_489 // (op 489) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_DPStreamTileLoop_489 if false)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_DPStreamTileLoop_489:
v_lshrrev_b32 v0, 7, v1 // (op 3219) 
s_mul_i32 s15, s16, s50 // (op 3219) 
s_add_u32 s15, s15, s0 // (op 3219) 
v_and_b32 v4, 63, v1 // (op 3219) 
s_add_u32 s6, s15, s9 // (op 3219) 

v_lshrrev_b32 v7, 4, v4 // (op 3219) 
s_mul_hi_u32 s15, s6, s13 // (op 3219) 
v_lshlrev_b32 v9, 2, v7 // (op 3219) 

s_sub_u32 s7, s6, s15 // (op 3219) 
v_lshl_add_u32 v7, v0, 4, v9 // (op 3219) 


s_lshr_b32 s8, s7, 1 // (op 3219) 

v_lshrrev_b32 v0, 2, v4 // (op 3219) 

s_add_u32 s7, s8, s15 // (op 3219) 


v_and_b32 v4, 3, v0 // (op 3219) 

s_lshr_b32 s8, s7, s11 // (op 3219) 

v_and_b32 v0, 3, v1 // (op 3219) 
s_and_b32 s7, s11, 2147483648 // (op 3219) 
v_lshl_add_u32 v9, v4, 2, v0 // (op 3219) 


s_cmp_eq_i32 s7, 0 // (op 3219) 
s_mov_b32 s15, scc // (op 3219) 

v_lshrrev_b32 v0, 6, v1 // (op 3219) 
s_cmp_eq_i32 s15, 0 // (op 3219) 
s_cselect_b32 s7, s6, s8 // (op 3219) 



v_and_b32 v4, 1, v0 // (op 3219) 

s_mul_hi_u32 s6, s7, s12 // (op 3219) 
v_lshl_add_u32 v0, v4, 4, v9 // (op 3219) 


s_sub_u32 s8, s7, s6 // (op 3219) 
s_lshr_b32 s15, s8, 1 // (op 3219) 

s_add_u32 s8, s15, s6 // (op 3219) 


s_lshr_b32 s6, s8, s10 // (op 3219) 

s_and_b32 s8, s10, 2147483648 // (op 3219) 
s_cmp_eq_i32 s8, 0 // (op 3219) 
s_mov_b32 s15, scc // (op 3219) 

s_cmp_eq_i32 s15, 0 // (op 3219) 
s_cselect_b32 s8, s7, s6 // (op 3219) 


s_mul_i32 s6, s8, s14 // (op 3219) 
v_lshl_add_u32 v4, s8, 4, v7 // (op 3219) 


s_sub_u32 s8, s7, s6 // (op 3219) 


v_lshl_add_u32 v7, s8, 6, v0 // (op 3219) 


v_mul_lo_u32 v0, v7, s4 // (op 3219) 

v_add_lshl_u32 v7, v4, v0, 1 // (op 3219) 


s_mul_i32 s19, s4, s23 // (op 3221) most significant: low of low * high
s_mul_i32 s15, s5, s22 // (op 3221) most significant: low of high * low
s_mul_i32 s18, s4, s22 // (op 3221) least significant
s_mul_hi_u32 s6, s4, s22 // (op 3221) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3221) most significant: sum
s_add_u32 s19, s19, s6 // (op 3221) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 3221) 

s_mov_b32 s56, s20 // (op 3221) reg expression
s_mov_b32 s57, s21 // (op 3221) reg expression
s_mov_b32 s58, s28 // (op 3221) reg expression
s_mov_b32 s59, 131072 // (op 3221) call()

v_lshrrev_b32 v0, 7, v1 // (op 3543) 
s_mul_i32 s15, s16, s50 // (op 3543) 
s_add_u32 s15, s15, s0 // (op 3543) 
v_and_b32 v4, 63, v1 // (op 3543) 
s_add_u32 s6, s15, s9 // (op 3543) 

v_lshrrev_b32 v9, 4, v4 // (op 3543) 
s_mul_hi_u32 s15, s6, s13 // (op 3543) 
v_lshlrev_b32 v10, 2, v9 // (op 3543) 

s_sub_u32 s7, s6, s15 // (op 3543) 
v_lshl_add_u32 v9, v0, 4, v10 // (op 3543) 


s_lshr_b32 s8, s7, 1 // (op 3543) 

v_lshrrev_b32 v0, 2, v4 // (op 3543) 

s_add_u32 s7, s8, s15 // (op 3543) 


v_and_b32 v4, 3, v0 // (op 3543) 

s_lshr_b32 s8, s7, s11 // (op 3543) 

v_and_b32 v0, 3, v1 // (op 3543) 
s_and_b32 s7, s11, 2147483648 // (op 3543) 
v_lshl_add_u32 v10, v4, 2, v0 // (op 3543) 


s_cmp_eq_i32 s7, 0 // (op 3543) 
s_mov_b32 s15, scc // (op 3543) 

v_lshrrev_b32 v0, 6, v1 // (op 3543) 
s_cmp_eq_i32 s15, 0 // (op 3543) 
s_cselect_b32 s7, s6, s8 // (op 3543) 



v_and_b32 v4, 1, v0 // (op 3543) 

s_mul_hi_u32 s6, s7, s12 // (op 3543) 
v_lshl_add_u32 v0, v4, 4, v10 // (op 3543) 


s_sub_u32 s8, s7, s6 // (op 3543) 
s_lshr_b32 s15, s8, 1 // (op 3543) 

s_add_u32 s8, s15, s6 // (op 3543) 


s_lshr_b32 s6, s8, s10 // (op 3543) 

s_and_b32 s8, s10, 2147483648 // (op 3543) 
s_cmp_eq_i32 s8, 0 // (op 3543) 
s_mov_b32 s15, scc // (op 3543) 

s_cmp_eq_i32 s15, 0 // (op 3543) 
s_cselect_b32 s8, s7, s6 // (op 3543) 


s_mul_i32 s6, s8, s14 // (op 3543) 
v_lshl_add_u32 v4, s8, 4, v9 // (op 3543) 


s_sub_u32 s8, s7, s6 // (op 3543) 


v_lshl_add_u32 v9, s8, 6, v0 // (op 3543) 


v_mul_lo_u32 v0, v9, s24 // (op 3543) 

v_add_lshl_u32 v9, v4, v0, 1 // (op 3543) 


s_mul_i32 s19, s24, s23 // (op 3545) most significant: low of low * high
s_mul_i32 s15, s25, s22 // (op 3545) most significant: low of high * low
s_mul_i32 s18, s24, s22 // (op 3545) least significant
s_mul_hi_u32 s6, s24, s22 // (op 3545) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3545) most significant: sum
s_add_u32 s19, s19, s6 // (op 3545) most significant: sum


s_lshl_b64 s[28:29], s[18:19], 1 // (op 3545) 

s_mov_b32 s60, s26 // (op 3545) reg expression
s_mov_b32 s61, s27 // (op 3545) reg expression
s_mov_b32 s62, s28 // (op 3545) reg expression
s_mov_b32 s63, 131072 // (op 3545) call()

v_and_b32 v0, 48, v1 // (op 3612) 
v_and_b32 v4, 63, v1 // (op 3612) 
v_sub_u32 v10, v4, v0 // (op 3612) 

v_lshrrev_b32 v4, 1, v10 // (op 3612) 
v_add_lshl_u32 v11, v0, v4, 4 // (op 3612) 


v_and_b32 v0, 1, v10 // (op 3612) 

v_lshlrev_b32 v4, 3, v0 // (op 3612) 

v_lshrrev_b32 v0, 7, v1 // (op 3612) 
v_lshl_add_u32 v10, v0, 4, v4 // (op 3612) 


v_add_u32 v0, v11, v10 // (op 3612) 


v_and_b32 v4, 48, v1 // (op 3687) 
v_and_b32 v10, 63, v1 // (op 3687) 
v_sub_u32 v11, v10, v4 // (op 3687) 

v_lshrrev_b32 v10, 1, v11 // (op 3687) 
v_add_lshl_u32 v16, v4, v10, 6 // (op 3687) 


v_and_b32 v4, 1, v11 // (op 3687) 

v_lshlrev_b32 v17, 3, v4 // (op 3687) 

v_lshrrev_b32 v4, 6, v1 // (op 3687) 
v_and_b32 v10, 1, v4 // (op 3687) 

v_lshl_add_u32 v4, v10, 4, v17 // (op 3687) 


v_add_u32 v17, v16, v4 // (op 3687) 


v_lshrrev_b32 v4, 7, v1 // (op 3753) 
v_and_b32 v16, 15, v1 // (op 3753) 
v_lshl_add_u32 v10, v4, 4, v16 // (op 3753) 


v_and_b32 v4, 48, v1 // (op 3753) 
v_add_u32 v11, v4, v10 // (op 3753) 


v_lshrrev_b32 v4, 6, v1 // (op 3829) 
v_and_b32 v10, 1, v4 // (op 3829) 

v_and_b32 v4, 15, v1 // (op 3829) 
v_lshl_add_u32 v16, v10, 4, v4 // (op 3829) 


v_and_b32 v4, 63, v1 // (op 3829) 
v_lshrrev_b32 v10, 4, v4 // (op 3829) 

v_lshl_add_u32 v4, v10, 6, v16 // (op 3829) 


v_accvgpr_write a0, 0.00000 // (op 206) call()
v_accvgpr_write a1, 0.00000 // (op 206) call()
v_accvgpr_write a2, 0.00000 // (op 206) call()
v_accvgpr_write a3, 0.00000 // (op 206) call()
v_accvgpr_write a4, 0.00000 // (op 702) call()
v_accvgpr_write a5, 0.00000 // (op 702) call()
v_accvgpr_write a6, 0.00000 // (op 702) call()
v_accvgpr_write a7, 0.00000 // (op 702) call()
s_mul_i32 s15, s16, s50 // (op 2723) 
s_add_u32 s15, s15, s0 // (op 2723) 
s_add_u32 s6, s15, s9 // (op 2723) 

s_mul_hi_u32 s15, s6, s13 // (op 2723) 
s_sub_u32 s7, s6, s15 // (op 2723) 
s_lshr_b32 s8, s7, 1 // (op 2723) 

s_add_u32 s7, s8, s15 // (op 2723) 


s_lshr_b32 s8, s7, s11 // (op 2723) 

s_and_b32 s7, s11, 2147483648 // (op 2723) 
s_cmp_eq_i32 s7, 0 // (op 2723) 
s_mov_b32 s15, scc // (op 2723) 

s_cmp_eq_i32 s15, 0 // (op 2723) 
s_cselect_b32 s7, s6, s8 // (op 2723) 


s_mul_hi_u32 s8, s7, s12 // (op 2723) 
s_sub_u32 s15, s7, s8 // (op 2723) 
s_lshr_b32 s18, s15, 1 // (op 2723) 

s_add_u32 s15, s18, s8 // (op 2723) 


s_lshr_b32 s8, s15, s10 // (op 2723) 

s_and_b32 s15, s10, 2147483648 // (op 2723) 
s_cmp_eq_i32 s15, 0 // (op 2723) 
s_mov_b32 s18, scc // (op 2723) 

s_cmp_eq_i32 s18, 0 // (op 2723) 
s_cselect_b32 s15, s7, s8 // (op 2723) 


s_lshl_b32 s8, s15, 4 // (op 2723) 

s_mov_b32 s19, 0 // (op 2723) convert
s_mov_b32 s18, s8 // (op 2723) convert

s_ashr_i64 s[28:29], s[2:3], 8 // (op 2723) 
s_mul_i32 s8, s7, s28 // (op 2723) 


s_sub_u32 s7, s6, s8 // (op 2723) 


v_lshl_add_u32 v10, s7, 8, v1 // (op 2723) 

v_mul_lo_u32 v21, v10, s31 // (op 2723) most significant: low of low * high
v_mul_hi_u32 v16, v10, s30 // (op 2723) most significant: high of low * low
v_mul_lo_u32 v20, v10, s30 // (op 2723) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2723) most significant: sum


v_mov_b32 v10, s18 // (op 2723) 
v_mov_b32 v16, s19 // (op 2723) 
v_add_co_u32 v22, s[28:29], v10, v20 // (op 2723) least significant half
v_addc_co_u32 v23, s[28:29], v16, v21, s[28:29] // (op 2723) most significant half





s_mul_i32 s19, s30, s3 // (op 2725) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2725) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2725) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2725) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2725) most significant: sum
s_add_u32 s19, s19, s6 // (op 2725) most significant: sum


s_mov_b32 s68, s32 // (op 2725) reg expression
s_mov_b32 s69, s33 // (op 2725) reg expression
s_mov_b32 s70, s18 // (op 2725) reg expression
s_mov_b32 s71, 131072 // (op 2725) call()

s_mul_i32 s15, s16, s50 // (op 2742) 
s_add_u32 s15, s15, s0 // (op 2742) 
s_add_u32 s6, s15, s9 // (op 2742) 

s_mul_hi_u32 s15, s6, s13 // (op 2742) 
s_sub_u32 s7, s6, s15 // (op 2742) 
s_lshr_b32 s8, s7, 1 // (op 2742) 

s_add_u32 s7, s8, s15 // (op 2742) 


s_lshr_b32 s8, s7, s11 // (op 2742) 

s_and_b32 s7, s11, 2147483648 // (op 2742) 
s_cmp_eq_i32 s7, 0 // (op 2742) 
s_mov_b32 s15, scc // (op 2742) 

s_cmp_eq_i32 s15, 0 // (op 2742) 
s_cselect_b32 s7, s6, s8 // (op 2742) 


s_mul_hi_u32 s8, s7, s12 // (op 2742) 
s_sub_u32 s15, s7, s8 // (op 2742) 
s_lshr_b32 s18, s15, 1 // (op 2742) 

s_add_u32 s15, s18, s8 // (op 2742) 


s_lshr_b32 s8, s15, s10 // (op 2742) 

s_and_b32 s15, s10, 2147483648 // (op 2742) 
s_cmp_eq_i32 s15, 0 // (op 2742) 
s_mov_b32 s18, scc // (op 2742) 

s_cmp_eq_i32 s18, 0 // (op 2742) 
s_cselect_b32 s15, s7, s8 // (op 2742) 


s_lshl_b32 s8, s15, 4 // (op 2742) 

s_mov_b32 s19, 0 // (op 2742) convert
s_mov_b32 s18, s8 // (op 2742) convert

s_ashr_i64 s[28:29], s[2:3], 8 // (op 2742) 
s_mul_i32 s8, s7, s28 // (op 2742) 


s_sub_u32 s7, s6, s8 // (op 2742) 


v_lshl_add_u32 v10, s7, 8, v1 // (op 2742) 

v_mul_lo_u32 v21, v10, s31 // (op 2742) most significant: low of low * high
v_mul_hi_u32 v16, v10, s30 // (op 2742) most significant: high of low * low
v_mul_lo_u32 v20, v10, s30 // (op 2742) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2742) most significant: sum


v_mov_b32 v10, s18 // (op 2742) 
v_mov_b32 v16, s19 // (op 2742) 
v_add_co_u32 v24, s[28:29], v10, v20 // (op 2742) least significant half
v_addc_co_u32 v25, s[28:29], v16, v21, s[28:29] // (op 2742) most significant half





s_mul_i32 s19, s30, s3 // (op 2744) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2744) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2744) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2744) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2744) most significant: sum
s_add_u32 s19, s19, s6 // (op 2744) most significant: sum


s_mov_b32 s68, s32 // (op 2744) reg expression
s_mov_b32 s69, s33 // (op 2744) reg expression
s_mov_b32 s70, s18 // (op 2744) reg expression
s_mov_b32 s71, 131072 // (op 2744) call()

v_and_b32 v10, -4, v1 // (op 2855) 
s_mul_i32 s15, s16, s50 // (op 2855) 
s_add_u32 s15, s15, s0 // (op 2855) 
v_sub_u32 v16, v1, v10 // (op 2855) 

s_add_u32 s6, s15, s9 // (op 2855) 

v_lshlrev_b32 v10, 4, v16 // (op 2855) 

s_mul_hi_u32 s15, s6, s13 // (op 2855) 
v_lshrrev_b32 v16, 2, v1 // (op 2855) 
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


s_mul_i32 s8, s15, s14 // (op 2855) 

s_sub_u32 s15, s7, s8 // (op 2855) 

v_lshl_add_u32 v20, s15, 6, v10 // (op 2855) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 2855) 
v_mov_b32 v27, 0 // (op 2855) convert
v_mov_b32 v26, v20 // (op 2855) convert

s_mul_i32 s8, s7, s18 // (op 2855) 


s_sub_u32 s7, s6, s8 // (op 2855) 


v_lshl_add_u32 v10, s7, 8, v16 // (op 2855) 


v_mul_lo_u32 v21, v10, s35 // (op 2855) most significant: low of low * high
v_mul_hi_u32 v16, v10, s34 // (op 2855) most significant: high of low * low
v_mul_lo_u32 v20, v10, s34 // (op 2855) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2855) most significant: sum

v_add_co_u32 v20, s[18:19], v20, v26 // (op 2855) least significant half
v_addc_co_u32 v21, s[18:19], v21, v27, s[18:19] // (op 2855) most significant half



s_mul_i32 s19, s34, s3 // (op 2857) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2857) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2857) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2857) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2857) most significant: sum
s_add_u32 s19, s19, s6 // (op 2857) most significant: sum


s_mov_b32 s72, s36 // (op 2857) reg expression
s_mov_b32 s73, s37 // (op 2857) reg expression
s_mov_b32 s74, s18 // (op 2857) reg expression
s_mov_b32 s75, 131072 // (op 2857) call()

v_and_b32 v10, -4, v1 // (op 2874) 
s_mul_i32 s15, s16, s50 // (op 2874) 
s_add_u32 s15, s15, s0 // (op 2874) 
v_sub_u32 v16, v1, v10 // (op 2874) 

s_add_u32 s6, s15, s9 // (op 2874) 

v_lshlrev_b32 v10, 4, v16 // (op 2874) 

s_mul_hi_u32 s15, s6, s13 // (op 2874) 
v_lshrrev_b32 v16, 2, v1 // (op 2874) 
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


s_mul_i32 s8, s15, s14 // (op 2874) 

s_sub_u32 s15, s7, s8 // (op 2874) 

v_lshl_add_u32 v26, s15, 6, v10 // (op 2874) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 2874) 
v_mov_b32 v29, 0 // (op 2874) convert
v_mov_b32 v28, v26 // (op 2874) convert

s_mul_i32 s8, s7, s18 // (op 2874) 


s_sub_u32 s7, s6, s8 // (op 2874) 


v_lshl_add_u32 v10, s7, 8, v16 // (op 2874) 


v_mul_lo_u32 v27, v10, s35 // (op 2874) most significant: low of low * high
v_mul_hi_u32 v16, v10, s34 // (op 2874) most significant: high of low * low
v_mul_lo_u32 v26, v10, s34 // (op 2874) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 2874) most significant: sum

v_add_co_u32 v26, s[18:19], v26, v28 // (op 2874) least significant half
v_addc_co_u32 v27, s[18:19], v27, v29, s[18:19] // (op 2874) most significant half



s_mul_i32 s19, s34, s3 // (op 2876) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2876) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2876) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2876) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2876) most significant: sum
s_add_u32 s19, s19, s6 // (op 2876) most significant: sum


s_mov_b32 s72, s36 // (op 2876) reg expression
s_mov_b32 s73, s37 // (op 2876) reg expression
s_mov_b32 s74, s18 // (op 2876) reg expression
s_mov_b32 s75, 131072 // (op 2876) call()

v_and_b32 v10, -16, v1 // (op 2983) 
s_mul_i32 s15, s16, s50 // (op 2983) 
s_add_u32 s15, s15, s0 // (op 2983) 
v_sub_u32 v16, v1, v10 // (op 2983) 

s_add_u32 s6, s15, s9 // (op 2983) 

v_lshrrev_b32 v10, 4, v1 // (op 2983) 
s_mul_hi_u32 s15, s6, s13 // (op 2983) 
s_sub_u32 s7, s6, s15 // (op 2983) 
s_lshr_b32 s8, s7, 1 // (op 2983) 

s_add_u32 s7, s8, s15 // (op 2983) 


s_lshr_b32 s8, s7, s11 // (op 2983) 

s_and_b32 s7, s11, 2147483648 // (op 2983) 
s_cmp_eq_i32 s7, 0 // (op 2983) 
s_mov_b32 s15, scc // (op 2983) 

s_cmp_eq_i32 s15, 0 // (op 2983) 
s_cselect_b32 s7, s6, s8 // (op 2983) 


s_mul_hi_u32 s8, s7, s12 // (op 2983) 
s_sub_u32 s15, s7, s8 // (op 2983) 
s_lshr_b32 s18, s15, 1 // (op 2983) 

s_add_u32 s15, s18, s8 // (op 2983) 


s_lshr_b32 s8, s15, s10 // (op 2983) 

s_and_b32 s15, s10, 2147483648 // (op 2983) 
s_cmp_eq_i32 s15, 0 // (op 2983) 
s_mov_b32 s18, scc // (op 2983) 

s_cmp_eq_i32 s18, 0 // (op 2983) 
s_cselect_b32 s15, s7, s8 // (op 2983) 


v_lshl_add_u32 v28, s15, 4, v16 // (op 2983) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 2983) 
s_mul_i32 s8, s7, s18 // (op 2983) 


s_sub_u32 s7, s6, s8 // (op 2983) 


v_lshl_add_u32 v16, s7, 3, v10 // (op 2983) 


v_mul_lo_u32 v10, v16, s38 // (op 2983) 

v_add_u32 v16, v28, v10 // (op 2983) 


s_mul_i32 s19, s38, s43 // (op 2985) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2985) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2985) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2985) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2985) most significant: sum
s_add_u32 s19, s19, s6 // (op 2985) most significant: sum


s_mov_b32 s76, s40 // (op 2985) reg expression
s_mov_b32 s77, s41 // (op 2985) reg expression
s_mov_b32 s78, s18 // (op 2985) reg expression
s_mov_b32 s79, 131072 // (op 2985) call()

v_and_b32 v10, -16, v1 // (op 3002) 
s_mul_i32 s15, s16, s50 // (op 3002) 
s_add_u32 s15, s15, s0 // (op 3002) 
v_sub_u32 v28, v1, v10 // (op 3002) 

s_add_u32 s6, s15, s9 // (op 3002) 

v_lshrrev_b32 v10, 4, v1 // (op 3002) 
s_mul_hi_u32 s15, s6, s13 // (op 3002) 
s_sub_u32 s7, s6, s15 // (op 3002) 
s_lshr_b32 s8, s7, 1 // (op 3002) 

s_add_u32 s7, s8, s15 // (op 3002) 


s_lshr_b32 s8, s7, s11 // (op 3002) 

s_and_b32 s7, s11, 2147483648 // (op 3002) 
s_cmp_eq_i32 s7, 0 // (op 3002) 
s_mov_b32 s15, scc // (op 3002) 

s_cmp_eq_i32 s15, 0 // (op 3002) 
s_cselect_b32 s7, s6, s8 // (op 3002) 


s_mul_hi_u32 s8, s7, s12 // (op 3002) 
s_sub_u32 s15, s7, s8 // (op 3002) 
s_lshr_b32 s18, s15, 1 // (op 3002) 

s_add_u32 s15, s18, s8 // (op 3002) 


s_lshr_b32 s8, s15, s10 // (op 3002) 

s_and_b32 s15, s10, 2147483648 // (op 3002) 
s_cmp_eq_i32 s15, 0 // (op 3002) 
s_mov_b32 s18, scc // (op 3002) 

s_cmp_eq_i32 s18, 0 // (op 3002) 
s_cselect_b32 s15, s7, s8 // (op 3002) 


v_lshl_add_u32 v29, s15, 4, v28 // (op 3002) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 3002) 
s_mul_i32 s8, s7, s18 // (op 3002) 


s_sub_u32 s7, s6, s8 // (op 3002) 


v_lshl_add_u32 v28, s7, 3, v10 // (op 3002) 


v_mul_lo_u32 v10, v28, s38 // (op 3002) 

v_add_u32 v28, v29, v10 // (op 3002) 


s_mul_i32 s19, s38, s43 // (op 3004) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 3004) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 3004) least significant
s_mul_hi_u32 s6, s38, s42 // (op 3004) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3004) most significant: sum
s_add_u32 s19, s19, s6 // (op 3004) most significant: sum


s_mov_b32 s76, s40 // (op 3004) reg expression
s_mov_b32 s77, s41 // (op 3004) reg expression
s_mov_b32 s78, s18 // (op 3004) reg expression
s_mov_b32 s79, 131072 // (op 3004) call()

v_and_b32 v10, -16, v1 // (op 3111) 
s_mul_i32 s15, s16, s50 // (op 3111) 
s_add_u32 s15, s15, s0 // (op 3111) 
v_sub_u32 v29, v1, v10 // (op 3111) 

s_add_u32 s6, s15, s9 // (op 3111) 

v_lshlrev_b32 v10, 2, v29 // (op 3111) 

s_mul_hi_u32 s15, s6, s13 // (op 3111) 
v_lshrrev_b32 v29, 4, v1 // (op 3111) 
s_sub_u32 s7, s6, s15 // (op 3111) 
s_lshr_b32 s8, s7, 1 // (op 3111) 

s_add_u32 s7, s8, s15 // (op 3111) 


s_lshr_b32 s8, s7, s11 // (op 3111) 

s_and_b32 s7, s11, 2147483648 // (op 3111) 
s_cmp_eq_i32 s7, 0 // (op 3111) 
s_mov_b32 s15, scc // (op 3111) 

s_cmp_eq_i32 s15, 0 // (op 3111) 
s_cselect_b32 s7, s6, s8 // (op 3111) 


s_mul_hi_u32 s8, s7, s12 // (op 3111) 
s_sub_u32 s15, s7, s8 // (op 3111) 
s_lshr_b32 s18, s15, 1 // (op 3111) 

s_add_u32 s15, s18, s8 // (op 3111) 


s_lshr_b32 s8, s15, s10 // (op 3111) 

s_and_b32 s15, s10, 2147483648 // (op 3111) 
s_cmp_eq_i32 s15, 0 // (op 3111) 
s_mov_b32 s18, scc // (op 3111) 

s_cmp_eq_i32 s18, 0 // (op 3111) 
s_cselect_b32 s15, s7, s8 // (op 3111) 


s_mul_i32 s8, s15, s14 // (op 3111) 

s_sub_u32 s15, s7, s8 // (op 3111) 

v_lshl_add_u32 v30, s15, 6, v10 // (op 3111) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 3111) 
s_mul_i32 s8, s7, s18 // (op 3111) 


s_sub_u32 s7, s6, s8 // (op 3111) 


v_lshl_add_u32 v10, s7, 3, v29 // (op 3111) 


v_mul_lo_u32 v29, v10, s44 // (op 3111) 
v_add_u32 v29, v29, v30 // (op 3111) 


s_mul_i32 s19, s44, s43 // (op 3113) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 3113) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 3113) least significant
s_mul_hi_u32 s6, s44, s42 // (op 3113) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3113) most significant: sum
s_add_u32 s19, s19, s6 // (op 3113) most significant: sum


s_mov_b32 s80, s46 // (op 3113) reg expression
s_mov_b32 s81, s47 // (op 3113) reg expression
s_mov_b32 s82, s18 // (op 3113) reg expression
s_mov_b32 s83, 131072 // (op 3113) call()

v_and_b32 v10, -16, v1 // (op 3130) 
s_mul_i32 s15, s16, s50 // (op 3130) 
s_add_u32 s15, s15, s0 // (op 3130) 
v_sub_u32 v30, v1, v10 // (op 3130) 

s_add_u32 s6, s15, s9 // (op 3130) 

v_lshlrev_b32 v10, 2, v30 // (op 3130) 

s_mul_hi_u32 s15, s6, s13 // (op 3130) 
v_lshrrev_b32 v30, 4, v1 // (op 3130) 
s_sub_u32 s7, s6, s15 // (op 3130) 
s_lshr_b32 s8, s7, 1 // (op 3130) 

s_add_u32 s7, s8, s15 // (op 3130) 


s_lshr_b32 s8, s7, s11 // (op 3130) 

s_and_b32 s7, s11, 2147483648 // (op 3130) 
s_cmp_eq_i32 s7, 0 // (op 3130) 
s_mov_b32 s15, scc // (op 3130) 

s_cmp_eq_i32 s15, 0 // (op 3130) 
s_cselect_b32 s7, s6, s8 // (op 3130) 


s_mul_hi_u32 s8, s7, s12 // (op 3130) 
s_sub_u32 s15, s7, s8 // (op 3130) 
s_lshr_b32 s18, s15, 1 // (op 3130) 

s_add_u32 s15, s18, s8 // (op 3130) 


s_lshr_b32 s8, s15, s10 // (op 3130) 

s_and_b32 s15, s10, 2147483648 // (op 3130) 
s_cmp_eq_i32 s15, 0 // (op 3130) 
s_mov_b32 s18, scc // (op 3130) 

s_cmp_eq_i32 s18, 0 // (op 3130) 
s_cselect_b32 s15, s7, s8 // (op 3130) 


s_mul_i32 s8, s15, s14 // (op 3130) 

s_sub_u32 s15, s7, s8 // (op 3130) 

v_lshl_add_u32 v31, s15, 6, v10 // (op 3130) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 3130) 
s_mul_i32 s8, s7, s18 // (op 3130) 


s_sub_u32 s7, s6, s8 // (op 3130) 


v_lshl_add_u32 v10, s7, 3, v30 // (op 3130) 


v_mul_lo_u32 v30, v10, s44 // (op 3130) 
v_add_u32 v30, v30, v31 // (op 3130) 


s_mul_i32 s19, s44, s43 // (op 3132) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 3132) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 3132) least significant
s_mul_hi_u32 s6, s44, s42 // (op 3132) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3132) most significant: sum
s_add_u32 s19, s19, s6 // (op 3132) most significant: sum


s_mov_b32 s80, s46 // (op 3132) reg expression
s_mov_b32 s81, s47 // (op 3132) reg expression
s_mov_b32 s82, s18 // (op 3132) reg expression
s_mov_b32 s83, 131072 // (op 3132) call()

v_and_b32 v10, -4, v1 // (op 3653) 
v_sub_u32 v31, v1, v10 // (op 3653) 

v_lshlrev_b32 v10, 4, v31 // (op 3653) 

v_lshrrev_b32 v31, 2, v1 // (op 3653) 
v_lshl_add_u32 v32, v31, 6, v10 // (op 3653) 


s_nop 0
v_readfirstlane_b32 s15, v32 // (op 3653) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4391) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 2633) 
v_mov_b64 v[32:33], s[18:19] // (op 2633) convert


v_mov_b64 v[34:35], v[20:21] // (op 2633) reg expression
s_mov_b32 m0, s15 // (op 2633) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2633) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2633) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2633) most significant half

s_add_u32 m0, m0, 2048 // (op 2633) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2633) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2633) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2633) most significant half

s_add_u32 m0, m0, 2048 // (op 2633) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2633) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2633) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2633) most significant half

s_add_u32 m0, m0, 2048 // (op 2633) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2633) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2633) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2633) most significant half

s_add_u32 m0, m0, 2048 // (op 2633) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2633) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2633) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2633) most significant half

s_add_u32 m0, m0, 2048 // (op 2633) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2633) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2633) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2633) most significant half

s_add_u32 m0, m0, 2048 // (op 2633) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2633) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2633) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2633) most significant half

s_add_u32 m0, m0, 2048 // (op 2633) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2633) Load value direct to lds





v_mov_b32 v10, v29 // (op 2235) reg expression

buffer_load_dword v31, v10, s[80:83], 0 offen // (op 2235) Load value

v_lshlrev_b32 v10, 4, v1 // (op 3588) 
s_nop 0
v_readfirstlane_b32 s15, v10 // (op 3588) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4372) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 2638) 
v_mov_b64 v[32:33], s[18:19] // (op 2638) convert


v_mov_b64 v[34:35], v[22:23] // (op 2638) reg expression
s_add_u32 s6, 16384, s15 // (op 2638) 
s_mov_b32 m0, s6 // (op 2638) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 2638) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 2638) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2638) most significant half

s_add_u32 m0, m0, 2048 // (op 2638) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 2638) Load value direct to lds






v_mov_b32 v10, v16 // (op 2241) reg expression

buffer_load_ubyte v32, v10, s[76:79], 0 offen offset: 0 // (op 2241) Load value

v_and_b32 v10, -4, v1 // (op 3938) 
v_sub_u32 v33, v1, v10 // (op 3938) 

v_lshlrev_b32 v10, 4, v33 // (op 3938) 

v_lshrrev_b32 v33, 2, v1 // (op 3938) 
v_lshl_add_u32 v34, v33, 6, v10 // (op 3938) 


s_nop 0
v_readfirstlane_b32 s15, v34 // (op 3938) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4462) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 2643) 
v_mov_b64 v[34:35], s[18:19] // (op 2643) convert


s_lshl_b64 s[18:19], s[34:35], 8 // (op 2643) 
v_mov_b32 v10, s18 // (op 2643) 
v_mov_b32 v33, s19 // (op 2643) 
v_add_co_u32 v36, s[28:29], v20, v10 // (op 2643) least significant half
v_addc_co_u32 v37, s[28:29], v21, v33, s[28:29] // (op 2643) most significant half




s_add_u32 s6, 20480, s15 // (op 2643) 
s_mov_b32 m0, s6 // (op 2643) reg expression
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2643) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2643) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2643) most significant half

s_add_u32 m0, m0, 2048 // (op 2643) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2643) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2643) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2643) most significant half

s_add_u32 m0, m0, 2048 // (op 2643) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2643) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2643) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2643) most significant half

s_add_u32 m0, m0, 2048 // (op 2643) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2643) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2643) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2643) most significant half

s_add_u32 m0, m0, 2048 // (op 2643) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2643) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2643) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2643) most significant half

s_add_u32 m0, m0, 2048 // (op 2643) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2643) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2643) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2643) most significant half

s_add_u32 m0, m0, 2048 // (op 2643) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2643) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2643) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2643) most significant half

s_add_u32 m0, m0, 2048 // (op 2643) 
buffer_load_dwordx4 v36, s[72:75], 0 offen offset: 0 lds // (op 2643) Load value direct to lds







s_lshl_b32 s15, s44, 3 // (op 2247) 
v_add_u32 v10, v29, s15 // (op 2247) 


buffer_load_dword v20, v10, s[80:83], 0 offen // (op 2247) Load value


v_lshlrev_b32 v10, 4, v1 // (op 3899) 
s_nop 0
v_readfirstlane_b32 s15, v10 // (op 3899) Wait state hazard: VALU Write Readlane Hazard

s_barrier  // (op 4452) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 2645) 
v_mov_b64 v[34:35], s[18:19] // (op 2645) convert


s_lshl_b64 s[18:19], s[30:31], 8 // (op 2645) 
v_mov_b32 v10, s18 // (op 2645) 
v_mov_b32 v21, s19 // (op 2645) 
v_add_co_u32 v36, s[28:29], v22, v10 // (op 2645) least significant half
v_addc_co_u32 v37, s[28:29], v23, v21, s[28:29] // (op 2645) most significant half




s_add_u32 s6, 36864, s15 // (op 2645) 
s_mov_b32 m0, s6 // (op 2645) reg expression
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2645) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v34 // (op 2645) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 2645) most significant half

s_add_u32 m0, m0, 2048 // (op 2645) 
buffer_load_dwordx4 v36, s[68:71], 0 offen offset: 0 lds // (op 2645) Load value direct to lds







s_lshl_b32 s15, s38, 3 // (op 2253) 
v_add_u32 v10, v16, s15 // (op 2253) 


buffer_load_ubyte v29, v10, s[76:79], 0 offen offset: 0 // (op 2253) Load value


v_and_b32 v10, -16, v1 // (op 3795) 
v_sub_u32 v16, v1, v10 // (op 3795) 

v_lshlrev_b32 v10, 2, v16 // (op 3795) 

v_lshrrev_b32 v16, 4, v1 // (op 3795) 
v_lshl_add_u32 v21, v16, 6, v10 // (op 3795) 


s_barrier  // (op 4432) 



s_waitcnt vmcnt(15)
ds_write_b32 v21, v31 offset:40960 // (op 2256) Store local data 

v_and_b32 v10, -16, v1 // (op 3729) 
v_sub_u32 v16, v1, v10 // (op 3729) 

v_and_b32 v10, 4294967280, v1 // (op 3729) 
v_add_u32 v21, v10, v16 // (op 3729) 


s_barrier  // (op 4411) 



s_waitcnt vmcnt(12)
ds_write_b8 v21, v32 offset:41472 // (op 2258) Store local data 

s_barrier  // (op 4388) 



ds_read_b64_tr_b8 v[34:35], v17 // (op 2279) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v17 offset:512 // (op 2279) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v17 offset:4096 // (op 2279) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v17 offset:4608 // (op 2279) Transpose load local data 



ds_read_b64_tr_b8 v[42:43], v17 offset:8192 // (op 2289) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v17 offset:8704 // (op 2289) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v17 offset:12288 // (op 2289) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v17 offset:12800 // (op 2289) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4429) 


ds_read_u8 v10, v4 offset:40960 // (op 2305) Load local data 


ds_read_u8 v16, v4 offset:41216 // (op 2315) Load local data 



ds_read_b64_tr_b8 v[50:51], v17 offset:32 // (op 2284) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v17 offset:544 // (op 2284) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v17 offset:4128 // (op 2284) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v17 offset:4640 // (op 2284) Transpose load local data 



ds_read_b64_tr_b8 v[58:59], v17 offset:8224 // (op 2294) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v17 offset:8736 // (op 2294) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v17 offset:12320 // (op 2294) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v17 offset:12832 // (op 2294) Transpose load local data 


ds_read_u8 v33, v4 offset:40992 // (op 2310) Load local data 


ds_read_u8 v21, v4 offset:41248 // (op 2320) Load local data 
s_barrier  // (op 4369) 



ds_read_b64_tr_b8 v[66:67], v0 offset:16384 // (op 2272) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:16512 // (op 2272) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:17408 // (op 2272) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:17536 // (op 2272) Transpose load local data 



ds_read_b64_tr_b8 v[74:75], v0 offset:18432 // (op 2275) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:18560 // (op 2275) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v0 offset:19456 // (op 2275) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v0 offset:19584 // (op 2275) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4408) 


ds_read_u8 v22, v11 offset:41472 // (op 2298) Load local data 


ds_read_u8 v23, v11 offset:41536 // (op 2301) Load local data 
v_lshlrev_b32 v82, 4, v1 // (op 3560) 
s_nop 0
v_readfirstlane_b32 s15, v82 // (op 3560) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v82, -4, v1 // (op 3625) 
v_sub_u32 v83, v1, v82 // (op 3625) 

v_lshlrev_b32 v82, 4, v83 // (op 3625) 

v_lshrrev_b32 v83, 2, v1 // (op 3625) 
v_lshl_add_u32 v84, v83, 6, v82 // (op 3625) 


s_nop 0
v_readfirstlane_b32 s6, v84 // (op 3625) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v82, -16, v1 // (op 3700) 
v_sub_u32 v83, v1, v82 // (op 3700) 

v_and_b32 v82, 4294967280, v1 // (op 3700) 
v_add_u32 v84, v82, v83 // (op 3700) 


v_and_b32 v82, -16, v1 // (op 3766) 
v_sub_u32 v83, v1, v82 // (op 3766) 

v_lshlrev_b32 v82, 2, v83 // (op 3766) 

v_lshrrev_b32 v83, 4, v1 // (op 3766) 
v_lshl_add_u32 v85, v83, 6, v82 // (op 3766) 


v_lshlrev_b32 v82, 4, v1 // (op 3884) 
s_nop 0
v_readfirstlane_b32 s7, v82 // (op 3884) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v82, -4, v1 // (op 3923) 
v_sub_u32 v83, v1, v82 // (op 3923) 

v_lshlrev_b32 v82, 4, v83 // (op 3923) 

v_lshrrev_b32 v83, 2, v1 // (op 3923) 
v_lshl_add_u32 v86, v83, 6, v82 // (op 3923) 


s_nop 0
v_readfirstlane_b32 s8, v86 // (op 3923) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v82, -16, v1 // (op 3967) 
v_sub_u32 v83, v1, v82 // (op 3967) 

v_and_b32 v82, 4294967280, v1 // (op 3967) 
v_add_u32 v86, v82, v83 // (op 3967) 


v_and_b32 v82, -16, v1 // (op 3991) 
v_sub_u32 v83, v1, v82 // (op 3991) 

v_lshlrev_b32 v82, 2, v83 // (op 3991) 

v_lshrrev_b32 v83, 4, v1 // (op 3991) 
v_lshl_add_u32 v87, v83, 6, v82 // (op 3991) 


s_mov_b32 s1, 0 // (op 704) call()
s_ashr_i64 s[18:19], s[2:3], 8 // (op 703) 
s_mov_b32 s28, s18 // (op 703) convert

s_and_b32 s29, s28, 4294967294 // (op 703) 

s_cmp_lt_u32 s1, s29 // (op 703) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_703 // (op 703) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_703 if false)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_703:
s_add_u32 s1, s1, 2 // (op 705) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[34:41], a[0:3], v22, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 220) 
s_barrier  // (op 4394) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[50:57], a[4:7], v22, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 751) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 2582) 
v_mov_b64 v[82:83], s[18:19] // (op 2582) convert


v_mov_b64 v[88:89], v[26:27] // (op 2582) 
s_lshl_b64 s[18:19], s[34:35], 8 // (op 2582) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 2582) 

v_mov_b32 v90, s28 // (op 2582) 
v_mov_b32 v91, s29 // (op 2582) 
v_add_co_u32 v92, s[18:19], v88, v90 // (op 2582) least significant half
v_addc_co_u32 v93, s[18:19], v89, v91, s[18:19] // (op 2582) most significant half




s_mov_b32 m0, s6 // (op 2582) reg expression
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2582) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2582) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2582) most significant half

s_add_u32 m0, m0, 2048 // (op 2582) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2582) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2582) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2582) most significant half

s_add_u32 m0, m0, 2048 // (op 2582) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2582) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2582) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2582) most significant half

s_add_u32 m0, m0, 2048 // (op 2582) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:81], v[42:49], a[0:3], v23, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 229) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2582) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:81], v[58:65], a[4:7], v23, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 760) 
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2582) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2582) most significant half

s_add_u32 m0, m0, 2048 // (op 2582) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2582) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2582) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2582) most significant half

s_add_u32 m0, m0, 2048 // (op 2582) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2582) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2582) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2582) most significant half

s_add_u32 m0, m0, 2048 // (op 2582) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2582) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v82 // (op 2582) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 2582) most significant half

s_add_u32 m0, m0, 2048 // (op 2582) 
buffer_load_dwordx4 v92, s[72:75], 0 offen offset: 0 lds // (op 2582) Load value direct to lds




v_mov_b32 v82, v30 // (op 235) 
s_lshl_b32 s18, s44, 4 // (op 235) 
v_add_u32 v83, v82, s18 // (op 235) 


buffer_load_dword v31, v83, s[80:83], 0 offen // (op 235) Load value

s_barrier  // (op 4375) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 2577) 
v_mov_b64 v[90:91], s[18:19] // (op 2577) convert


v_mov_b64 v[92:93], v[24:25] // (op 2577) 
s_lshl_b64 s[18:19], s[30:31], 8 // (op 2577) 
s_lshl_b64 s[28:29], s[18:19], 1 // (op 2577) 

v_mov_b32 v83, s28 // (op 2577) 
v_mov_b32 v94, s29 // (op 2577) 
v_add_co_u32 v96, s[18:19], v92, v83 // (op 2577) least significant half
v_addc_co_u32 v97, s[18:19], v93, v94, s[18:19] // (op 2577) most significant half




s_add_u32 s18, 16384, s15 // (op 2577) 
s_nop 0
s_mov_b32 m0, s18 // (op 2577) reg expression
buffer_load_dwordx4 v96, s[68:71], 0 offen offset: 0 lds // (op 2577) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v90 // (op 2577) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 2577) most significant half

s_add_u32 m0, m0, 2048 // (op 2577) 
buffer_load_dwordx4 v96, s[68:71], 0 offen offset: 0 lds // (op 2577) Load value direct to lds





v_mov_b32 v83, v28 // (op 234) 
s_lshl_b32 s18, s38, 4 // (op 234) 
v_add_u32 v90, v83, s18 // (op 234) 


buffer_load_ubyte v32, v90, s[76:79], 0 offen offset: 0 // (op 234) Load value

s_barrier  // (op 4478) 



s_waitcnt vmcnt(15)
ds_write_b32 v87, v20 offset:41600 // (op 608) Store local data 
s_barrier  // (op 4472) 



s_waitcnt vmcnt(12)
ds_write_b8 v86, v29 offset:42112 // (op 607) Store local data 
s_barrier  // (op 4458) 



ds_read_b64_tr_b8 v[34:35], v15 offset:20480 // (op 591) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v15 offset:20992 // (op 591) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v15 offset:24576 // (op 591) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v15 offset:25088 // (op 591) Transpose load local data 



ds_read_b64_tr_b8 v[94:95], v15 offset:28672 // (op 600) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v15 offset:29184 // (op 600) Transpose load local data 
ds_read_b64_tr_b8 v[98:99], v15 offset:32768 // (op 600) Transpose load local data 
ds_read_b64_tr_b8 v[100:101], v15 offset:33280 // (op 600) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4475) 


ds_read_u8 v10, v14 offset:41600 // (op 595) Load local data 


ds_read_u8 v90, v14 offset:41856 // (op 604) Load local data 



ds_read_b64_tr_b8 v[50:51], v15 offset:20512 // (op 774) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v15 offset:21024 // (op 774) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v15 offset:24608 // (op 774) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v15 offset:25120 // (op 774) Transpose load local data 



ds_read_b64_tr_b8 v[102:103], v15 offset:28704 // (op 765) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v15 offset:29216 // (op 765) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v15 offset:32800 // (op 765) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v15 offset:33312 // (op 765) Transpose load local data 


ds_read_u8 v33, v14 offset:41632 // (op 778) Load local data 


ds_read_u8 v91, v14 offset:41888 // (op 769) Load local data 
s_barrier  // (op 4449) 
s_barrier  // (op 4623) 



ds_read_b64_tr_b8 v[66:67], v5 offset:36864 // (op 589) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v5 offset:36992 // (op 589) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v5 offset:37888 // (op 589) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v5 offset:38016 // (op 589) Transpose load local data 



ds_read_b64_tr_b8 v[110:111], v5 offset:38912 // (op 598) Transpose load local data 
ds_read_b64_tr_b8 v[112:113], v5 offset:39040 // (op 598) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v5 offset:39936 // (op 598) Transpose load local data 
ds_read_b64_tr_b8 v[116:117], v5 offset:40064 // (op 598) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4468) 


ds_read_u8 v22, v8 offset:42112 // (op 593) Load local data 


ds_read_u8 v23, v8 offset:42176 // (op 602) Load local data 
s_barrier  // (op 4587) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[34:41], a[0:3], v22, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 596) 
s_barrier  // (op 4465) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[50:57], a[4:7], v22, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 761) 
s_lshl_b64 s[18:19], s[34:35], 5 // (op 2605) 
v_mov_b64 v[118:119], s[18:19] // (op 2605) convert


s_lshl_b64 s[18:19], s[34:35], 8 // (op 2605) 
s_mul_i32 s29, 3, s19 // (op 2605) most significant: low of low * high
s_mul_i32 s28, 3, s18 // (op 2605) least significant
s_mul_hi_u32 s48, 3, s18 // (op 2605) most significant: high of low * low
s_add_u32 s29, s29, s48 // (op 2605) most significant: sum


v_mov_b32 v120, s28 // (op 2605) 
v_mov_b32 v121, s29 // (op 2605) 
v_add_co_u32 v122, s[18:19], v88, v120 // (op 2605) least significant half
v_addc_co_u32 v123, s[18:19], v89, v121, s[18:19] // (op 2605) most significant half




s_add_u32 s18, 20480, s8 // (op 2605) 
s_nop 0
s_mov_b32 m0, s18 // (op 2605) reg expression
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2605) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2605) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2605) most significant half

s_add_u32 m0, m0, 2048 // (op 2605) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2605) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2605) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2605) most significant half

s_add_u32 m0, m0, 2048 // (op 2605) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2605) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2605) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2605) most significant half

s_add_u32 m0, m0, 2048 // (op 2605) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[110:117], v[94:101], a[0:3], v23, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 605) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2605) Load value direct to lds


v_add_co_u32 v122, s[28:29], v122, v118 // (op 2605) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[110:117], v[102:109], a[4:7], v23, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 770) 
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2605) most significant half




s_add_u32 m0, m0, 2048 // (op 2605) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2605) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2605) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2605) most significant half

s_add_u32 m0, m0, 2048 // (op 2605) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2605) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2605) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2605) most significant half

s_add_u32 m0, m0, 2048 // (op 2605) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2605) Load value direct to lds
v_add_co_u32 v122, s[28:29], v122, v118 // (op 2605) least significant half
v_addc_co_u32 v123, s[28:29], 0, v123, s[28:29] // (op 2605) most significant half

s_add_u32 m0, m0, 2048 // (op 2605) 
buffer_load_dwordx4 v122, s[72:75], 0 offen offset: 0 lds // (op 2605) Load value direct to lds






s_lshl_b32 s18, s44, 3 // (op 611) 
s_mul_i32 s19, 3, s18 // (op 611) 

v_add_u32 v88, v82, s19 // (op 611) 


buffer_load_dword v20, v88, s[80:83], 0 offen // (op 611) Load value

s_lshl_b64 s[18:19], s[34:35], 8 // (op 2866) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 2866) 

s_barrier  // (op 4455) 
s_lshl_b32 s18, s44, 4 // (op 3122) 

v_add_u32 v30, v30, s18 // (op 3122) 

v_mov_b32 v82, s28 // (op 2866) 
v_mov_b32 v88, s29 // (op 2866) 
s_lshl_b64 s[18:19], s[30:31], 7 // (op 2600) 
v_mov_b64 v[90:91], s[18:19] // (op 2600) convert


s_lshl_b64 s[18:19], s[30:31], 8 // (op 2600) 
s_mul_i32 s49, 3, s19 // (op 2600) most significant: low of low * high
s_mul_i32 s48, 3, s18 // (op 2600) least significant
s_mul_hi_u32 s52, 3, s18 // (op 2600) most significant: high of low * low
s_add_u32 s49, s49, s52 // (op 2600) most significant: sum


v_mov_b32 v89, s48 // (op 2600) 
v_mov_b32 v94, s49 // (op 2600) 
v_add_co_u32 v26, s[18:19], v26, v82 // (op 2866) least significant half
v_addc_co_u32 v27, s[18:19], v27, v88, s[18:19] // (op 2866) most significant half




v_add_co_u32 v96, s[18:19], v92, v89 // (op 2600) least significant half
v_addc_co_u32 v97, s[18:19], v93, v94, s[18:19] // (op 2600) most significant half




s_add_u32 s18, 36864, s7 // (op 2600) 
s_nop 0
s_mov_b32 m0, s18 // (op 2600) reg expression
buffer_load_dwordx4 v96, s[68:71], 0 offen offset: 0 lds // (op 2600) Load value direct to lds
v_add_co_u32 v96, s[28:29], v96, v90 // (op 2600) least significant half
v_addc_co_u32 v97, s[28:29], 0, v97, s[28:29] // (op 2600) most significant half

s_add_u32 m0, m0, 2048 // (op 2600) 
buffer_load_dwordx4 v96, s[68:71], 0 offen offset: 0 lds // (op 2600) Load value direct to lds






s_lshl_b32 s18, s38, 3 // (op 610) 
s_mul_i32 s19, 3, s18 // (op 610) 

v_add_u32 v82, v83, s19 // (op 610) 


buffer_load_ubyte v29, v82, s[76:79], 0 offen offset: 0 // (op 610) Load value

s_lshl_b64 s[18:19], s[30:31], 8 // (op 2734) 

s_lshl_b64 s[28:29], s[18:19], 1 // (op 2734) 

s_barrier  // (op 4435) 
s_lshl_b32 s18, s38, 4 // (op 2994) 



v_add_u32 v28, v28, s18 // (op 2994) 

v_mov_b32 v82, s28 // (op 2734) 
v_mov_b32 v83, s29 // (op 2734) 
v_add_co_u32 v24, s[18:19], v24, v82 // (op 2734) least significant half
v_addc_co_u32 v25, s[18:19], v25, v83, s[18:19] // (op 2734) most significant half




s_waitcnt vmcnt(15)
ds_write_b32 v85, v31 offset:40960 // (op 232) Store local data 
s_barrier  // (op 4414) 



s_waitcnt vmcnt(12)
ds_write_b8 v84, v32 offset:41472 // (op 231) Store local data 
s_barrier  // (op 4400) 



ds_read_b64_tr_b8 v[34:35], v13 // (op 215) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v13 offset:512 // (op 215) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v13 offset:4096 // (op 215) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v13 offset:4608 // (op 215) Transpose load local data 



ds_read_b64_tr_b8 v[42:43], v13 offset:8192 // (op 224) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v13 offset:8704 // (op 224) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v13 offset:12288 // (op 224) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v13 offset:12800 // (op 224) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4442) 


ds_read_u8 v10, v12 offset:40960 // (op 219) Load local data 


ds_read_u8 v16, v12 offset:41216 // (op 228) Load local data 



ds_read_b64_tr_b8 v[50:51], v13 offset:32 // (op 746) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v13 offset:544 // (op 746) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v13 offset:4128 // (op 746) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v13 offset:4640 // (op 746) Transpose load local data 



ds_read_b64_tr_b8 v[58:59], v13 offset:8224 // (op 755) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v13 offset:8736 // (op 755) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v13 offset:12320 // (op 755) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v13 offset:12832 // (op 755) Transpose load local data 


ds_read_u8 v33, v12 offset:40992 // (op 750) Load local data 


ds_read_u8 v21, v12 offset:41248 // (op 759) Load local data 
s_barrier  // (op 4381) 



ds_read_b64_tr_b8 v[66:67], v3 offset:16384 // (op 213) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v3 offset:16512 // (op 213) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v3 offset:17408 // (op 213) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v3 offset:17536 // (op 213) Transpose load local data 



ds_read_b64_tr_b8 v[74:75], v3 offset:18432 // (op 222) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v3 offset:18560 // (op 222) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v3 offset:19456 // (op 222) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v3 offset:19584 // (op 222) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 4421) 


ds_read_u8 v22, v6 offset:41472 // (op 217) Load local data 


ds_read_u8 v23, v6 offset:41536 // (op 226) Load local data 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 703) 
s_mov_b32 s28, s18 // (op 703) convert

s_and_b32 s29, s28, 4294967294 // (op 703) 

s_cmp_lt_u32 s1, s29 // (op 703) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_703 // (op 703) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoop_703 if true)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_703:



















s_barrier  // (op 4605) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 2672) 
s_and_b32 s15, s18, 4294967294 // (op 2672) 
s_add_u32 s6, s0, s15 // (op 2672) 

s_mul_i32 s15, s16, s50 // (op 2672) 
s_add_u32 s15, s15, s6 // (op 2672) 

s_add_u32 s6, s15, s9 // (op 2672) 

s_mul_hi_u32 s15, s6, s13 // (op 2672) 
s_sub_u32 s7, s6, s15 // (op 2672) 
s_lshr_b32 s8, s7, 1 // (op 2672) 

s_add_u32 s7, s8, s15 // (op 2672) 


s_lshr_b32 s8, s7, s11 // (op 2672) 

s_and_b32 s7, s11, 2147483648 // (op 2672) 
s_cmp_eq_i32 s7, 0 // (op 2672) 
s_mov_b32 s15, scc // (op 2672) 

s_cmp_eq_i32 s15, 0 // (op 2672) 
s_cselect_b32 s7, s6, s8 // (op 2672) 


s_mul_hi_u32 s8, s7, s12 // (op 2672) 
s_sub_u32 s15, s7, s8 // (op 2672) 
s_lshr_b32 s28, s15, 1 // (op 2672) 

s_add_u32 s15, s28, s8 // (op 2672) 


s_lshr_b32 s8, s15, s10 // (op 2672) 

s_and_b32 s15, s10, 2147483648 // (op 2672) 
s_cmp_eq_i32 s15, 0 // (op 2672) 
s_mov_b32 s28, scc // (op 2672) 

s_cmp_eq_i32 s28, 0 // (op 2672) 
s_cselect_b32 s15, s7, s8 // (op 2672) 


s_lshl_b32 s8, s15, 4 // (op 2672) 

s_mov_b32 s29, 0 // (op 2672) convert
s_mov_b32 s28, s8 // (op 2672) convert

s_mul_i32 s8, s7, s18 // (op 2672) 


s_sub_u32 s7, s6, s8 // (op 2672) 


v_lshl_add_u32 v10, s7, 8, v1 // (op 2672) 

v_mul_lo_u32 v25, v10, s31 // (op 2672) most significant: low of low * high
v_mul_hi_u32 v16, v10, s30 // (op 2672) most significant: high of low * low
v_mul_lo_u32 v24, v10, s30 // (op 2672) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 2672) most significant: sum


v_mov_b32 v10, s28 // (op 2672) 
v_mov_b32 v16, s29 // (op 2672) 
v_add_co_u32 v26, s[18:19], v10, v24 // (op 2672) least significant half
v_addc_co_u32 v27, s[18:19], v16, v25, s[18:19] // (op 2672) most significant half





s_mul_i32 s19, s30, s3 // (op 2674) most significant: low of low * high
s_mul_i32 s15, s31, s2 // (op 2674) most significant: low of high * low
s_mul_i32 s18, s30, s2 // (op 2674) least significant
s_mul_hi_u32 s6, s30, s2 // (op 2674) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2674) most significant: sum
s_add_u32 s19, s19, s6 // (op 2674) most significant: sum


s_mov_b32 s68, s32 // (op 2674) reg expression
s_mov_b32 s69, s33 // (op 2674) reg expression
s_mov_b32 s70, s18 // (op 2674) reg expression
s_mov_b32 s71, 131072 // (op 2674) call()

v_and_b32 v10, -4, v1 // (op 2810) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 2810) 
v_sub_u32 v16, v1, v10 // (op 2810) 

s_and_b32 s15, s18, 4294967294 // (op 2810) 
v_lshlrev_b32 v10, 4, v16 // (op 2810) 

s_add_u32 s6, s0, s15 // (op 2810) 

v_lshrrev_b32 v16, 2, v1 // (op 2810) 
s_mul_i32 s15, s16, s50 // (op 2810) 
s_add_u32 s15, s15, s6 // (op 2810) 

s_add_u32 s6, s15, s9 // (op 2810) 

s_mul_hi_u32 s15, s6, s13 // (op 2810) 
s_sub_u32 s7, s6, s15 // (op 2810) 
s_lshr_b32 s8, s7, 1 // (op 2810) 

s_add_u32 s7, s8, s15 // (op 2810) 


s_lshr_b32 s8, s7, s11 // (op 2810) 

s_and_b32 s7, s11, 2147483648 // (op 2810) 
s_cmp_eq_i32 s7, 0 // (op 2810) 
s_mov_b32 s15, scc // (op 2810) 

s_cmp_eq_i32 s15, 0 // (op 2810) 
s_cselect_b32 s7, s6, s8 // (op 2810) 


s_mul_hi_u32 s8, s7, s12 // (op 2810) 
s_sub_u32 s15, s7, s8 // (op 2810) 
s_lshr_b32 s28, s15, 1 // (op 2810) 

s_add_u32 s15, s28, s8 // (op 2810) 


s_lshr_b32 s8, s15, s10 // (op 2810) 

s_and_b32 s15, s10, 2147483648 // (op 2810) 
s_cmp_eq_i32 s15, 0 // (op 2810) 
s_mov_b32 s28, scc // (op 2810) 

s_cmp_eq_i32 s28, 0 // (op 2810) 
s_cselect_b32 s15, s7, s8 // (op 2810) 


s_mul_i32 s8, s15, s14 // (op 2810) 

s_sub_u32 s15, s7, s8 // (op 2810) 

s_mul_i32 s8, s7, s18 // (op 2810) 


v_lshl_add_u32 v28, s15, 6, v10 // (op 2810) 


s_sub_u32 s7, s6, s8 // (op 2810) 


v_mov_b32 v25, 0 // (op 2810) convert
v_mov_b32 v24, v28 // (op 2810) convert

v_lshl_add_u32 v10, s7, 8, v16 // (op 2810) 


v_mul_lo_u32 v31, v10, s35 // (op 2810) most significant: low of low * high
v_mul_hi_u32 v16, v10, s34 // (op 2810) most significant: high of low * low
v_mul_lo_u32 v30, v10, s34 // (op 2810) least significant: low of low * low
v_add_u32 v31, v31, v16 // (op 2810) most significant: sum

v_add_co_u32 v30, s[18:19], v30, v24 // (op 2810) least significant half
v_addc_co_u32 v31, s[18:19], v31, v25, s[18:19] // (op 2810) most significant half



s_mul_i32 s19, s34, s3 // (op 2812) most significant: low of low * high
s_mul_i32 s15, s35, s2 // (op 2812) most significant: low of high * low
s_mul_i32 s18, s34, s2 // (op 2812) least significant
s_mul_hi_u32 s6, s34, s2 // (op 2812) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2812) most significant: sum
s_add_u32 s19, s19, s6 // (op 2812) most significant: sum


s_mov_b32 s72, s36 // (op 2812) reg expression
s_mov_b32 s73, s37 // (op 2812) reg expression
s_mov_b32 s74, s18 // (op 2812) reg expression
s_mov_b32 s75, 131072 // (op 2812) call()

v_and_b32 v10, -16, v1 // (op 2938) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 2938) 
v_sub_u32 v16, v1, v10 // (op 2938) 

s_and_b32 s15, s18, 4294967294 // (op 2938) 
v_lshrrev_b32 v10, 4, v1 // (op 2938) 
s_add_u32 s6, s0, s15 // (op 2938) 

s_mul_i32 s15, s16, s50 // (op 2938) 
s_add_u32 s15, s15, s6 // (op 2938) 

s_add_u32 s6, s15, s9 // (op 2938) 

s_mul_hi_u32 s15, s6, s13 // (op 2938) 
s_sub_u32 s7, s6, s15 // (op 2938) 
s_lshr_b32 s8, s7, 1 // (op 2938) 

s_add_u32 s7, s8, s15 // (op 2938) 


s_lshr_b32 s8, s7, s11 // (op 2938) 

s_and_b32 s7, s11, 2147483648 // (op 2938) 
s_cmp_eq_i32 s7, 0 // (op 2938) 
s_mov_b32 s15, scc // (op 2938) 

s_cmp_eq_i32 s15, 0 // (op 2938) 
s_cselect_b32 s7, s6, s8 // (op 2938) 


s_mul_hi_u32 s8, s7, s12 // (op 2938) 
s_sub_u32 s15, s7, s8 // (op 2938) 
s_lshr_b32 s28, s15, 1 // (op 2938) 

s_add_u32 s15, s28, s8 // (op 2938) 


s_lshr_b32 s8, s15, s10 // (op 2938) 

s_and_b32 s15, s10, 2147483648 // (op 2938) 
s_cmp_eq_i32 s15, 0 // (op 2938) 
s_mov_b32 s28, scc // (op 2938) 

s_cmp_eq_i32 s28, 0 // (op 2938) 
s_cselect_b32 s15, s7, s8 // (op 2938) 


s_mul_i32 s8, s7, s18 // (op 2938) 


v_lshl_add_u32 v28, s15, 4, v16 // (op 2938) 


s_sub_u32 s7, s6, s8 // (op 2938) 


v_lshl_add_u32 v16, s7, 3, v10 // (op 2938) 


v_mul_lo_u32 v10, v16, s38 // (op 2938) 

v_add_u32 v16, v28, v10 // (op 2938) 


s_mul_i32 s19, s38, s43 // (op 2940) most significant: low of low * high
s_mul_i32 s15, s39, s42 // (op 2940) most significant: low of high * low
s_mul_i32 s18, s38, s42 // (op 2940) least significant
s_mul_hi_u32 s6, s38, s42 // (op 2940) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 2940) most significant: sum
s_add_u32 s19, s19, s6 // (op 2940) most significant: sum


s_mov_b32 s76, s40 // (op 2940) reg expression
s_mov_b32 s77, s41 // (op 2940) reg expression
s_mov_b32 s78, s18 // (op 2940) reg expression
s_mov_b32 s79, 131072 // (op 2940) call()

v_and_b32 v10, -16, v1 // (op 3066) 
s_ashr_i64 s[18:19], s[2:3], 8 // (op 3066) 
v_sub_u32 v28, v1, v10 // (op 3066) 

s_and_b32 s15, s18, 4294967294 // (op 3066) 
v_lshlrev_b32 v10, 2, v28 // (op 3066) 

s_add_u32 s6, s0, s15 // (op 3066) 

v_lshrrev_b32 v28, 4, v1 // (op 3066) 
s_mul_i32 s15, s16, s50 // (op 3066) 
s_add_u32 s15, s15, s6 // (op 3066) 

s_add_u32 s6, s15, s9 // (op 3066) 

s_mul_hi_u32 s15, s6, s13 // (op 3066) 
s_sub_u32 s7, s6, s15 // (op 3066) 
s_lshr_b32 s8, s7, 1 // (op 3066) 

s_add_u32 s7, s8, s15 // (op 3066) 


s_lshr_b32 s8, s7, s11 // (op 3066) 

s_and_b32 s7, s11, 2147483648 // (op 3066) 
s_cmp_eq_i32 s7, 0 // (op 3066) 
s_mov_b32 s15, scc // (op 3066) 

s_cmp_eq_i32 s15, 0 // (op 3066) 
s_cselect_b32 s7, s6, s8 // (op 3066) 


s_mul_hi_u32 s8, s7, s12 // (op 3066) 
s_sub_u32 s15, s7, s8 // (op 3066) 
s_lshr_b32 s28, s15, 1 // (op 3066) 

s_add_u32 s15, s28, s8 // (op 3066) 


s_lshr_b32 s8, s15, s10 // (op 3066) 

s_and_b32 s15, s10, 2147483648 // (op 3066) 
s_cmp_eq_i32 s15, 0 // (op 3066) 
s_mov_b32 s28, scc // (op 3066) 

s_cmp_eq_i32 s28, 0 // (op 3066) 
s_cselect_b32 s15, s7, s8 // (op 3066) 


s_mul_i32 s8, s15, s14 // (op 3066) 

s_sub_u32 s15, s7, s8 // (op 3066) 

s_mul_i32 s8, s7, s18 // (op 3066) 


v_lshl_add_u32 v32, s15, 6, v10 // (op 3066) 


s_sub_u32 s7, s6, s8 // (op 3066) 


v_lshl_add_u32 v10, s7, 3, v28 // (op 3066) 


v_mul_lo_u32 v28, v10, s44 // (op 3066) 
v_add_u32 v28, v28, v32 // (op 3066) 


s_mul_i32 s19, s44, s43 // (op 3068) most significant: low of low * high
s_mul_i32 s15, s45, s42 // (op 3068) most significant: low of high * low
s_mul_i32 s18, s44, s42 // (op 3068) least significant
s_mul_hi_u32 s6, s44, s42 // (op 3068) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 3068) most significant: sum
s_add_u32 s19, s19, s6 // (op 3068) most significant: sum


s_mov_b32 s80, s46 // (op 3068) reg expression
s_mov_b32 s81, s47 // (op 3068) reg expression
s_mov_b32 s82, s18 // (op 3068) reg expression
s_mov_b32 s83, 131072 // (op 3068) call()

v_lshlrev_b32 v10, 4, v1 // (op 3573) 
s_nop 0
v_readfirstlane_b32 s15, v10 // (op 3573) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 3638) 
v_sub_u32 v32, v1, v10 // (op 3638) 

v_lshlrev_b32 v10, 4, v32 // (op 3638) 

v_lshrrev_b32 v32, 2, v1 // (op 3638) 
v_lshl_add_u32 v24, v32, 6, v10 // (op 3638) 


s_nop 0
v_readfirstlane_b32 s6, v24 // (op 3638) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -16, v1 // (op 3713) 
v_sub_u32 v32, v1, v10 // (op 3713) 

v_and_b32 v10, 4294967280, v1 // (op 3713) 
v_add_u32 v24, v10, v32 // (op 3713) 


v_and_b32 v10, -16, v1 // (op 3779) 
v_sub_u32 v25, v1, v10 // (op 3779) 

v_lshlrev_b32 v10, 2, v25 // (op 3779) 

v_lshrrev_b32 v25, 4, v1 // (op 3779) 
v_lshl_add_u32 v32, v25, 6, v10 // (op 3779) 


s_ashr_i64 s[18:19], s[2:3], 8 // (op 709) 
s_and_b32 s7, s18, 4294967294 // (op 709) 

s_ashr_i64 s[18:19], s[2:3], 8 // (op 708) 
s_mov_b32 s8, s18 // (op 708) convert

s_cmp_lt_u32 s7, s8 // (op 708) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_708 // (op 708) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_708 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_708:
s_barrier  // (op 4378) 
s_add_u32 s7, s7, 1 // (op 710) 

s_lshl_b64 s[18:19], s[30:31], 7 // (op 2587) 
v_mov_b64 v[20:21], s[18:19] // (op 2587) convert


v_mov_b64 v[22:23], v[26:27] // (op 2587) 
v_mov_b64 v[34:35], v[22:23] // (op 2587) reg expression
s_add_u32 s8, 16384, s15 // (op 2587) 
s_mov_b32 m0, s8 // (op 2587) reg expression
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 2587) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2587) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2587) most significant half

s_add_u32 m0, m0, 2048 // (op 2587) 
buffer_load_dwordx4 v34, s[68:71], 0 offen offset: 0 lds // (op 2587) Load value direct to lds




s_lshl_b64 s[18:19], s[30:31], 8 // (op 2662) 
v_mov_b32 v10, s18 // (op 2662) 
v_mov_b32 v25, s19 // (op 2662) 
v_add_co_u32 v26, s[28:29], v26, v10 // (op 2662) least significant half
v_addc_co_u32 v27, s[28:29], v27, v25, s[28:29] // (op 2662) most significant half




s_barrier  // (op 4397) 

s_lshl_b64 s[18:19], s[34:35], 5 // (op 2593) 
v_mov_b64 v[20:21], s[18:19] // (op 2593) convert


v_mov_b64 v[22:23], v[30:31] // (op 2593) 
v_mov_b64 v[34:35], v[22:23] // (op 2593) reg expression
s_mov_b32 m0, s6 // (op 2593) reg expression
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2593) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2593) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2593) most significant half

s_add_u32 m0, m0, 2048 // (op 2593) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2593) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2593) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2593) most significant half

s_add_u32 m0, m0, 2048 // (op 2593) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2593) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2593) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2593) most significant half

s_add_u32 m0, m0, 2048 // (op 2593) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2593) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2593) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2593) most significant half

s_add_u32 m0, m0, 2048 // (op 2593) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2593) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2593) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2593) most significant half

s_add_u32 m0, m0, 2048 // (op 2593) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2593) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2593) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2593) most significant half

s_add_u32 m0, m0, 2048 // (op 2593) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2593) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v20 // (op 2593) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 2593) most significant half

s_add_u32 m0, m0, 2048 // (op 2593) 
buffer_load_dwordx4 v34, s[72:75], 0 offen offset: 0 lds // (op 2593) Load value direct to lds



s_lshl_b64 s[18:19], s[34:35], 8 // (op 2802) 
v_mov_b32 v10, s18 // (op 2802) 
v_mov_b32 v25, s19 // (op 2802) 
v_add_co_u32 v30, s[28:29], v30, v10 // (op 2802) least significant half
v_addc_co_u32 v31, s[28:29], v31, v25, s[28:29] // (op 2802) most significant half






v_mov_b32 v10, v16 // (op 526) 
v_mov_b32 v25, v10 // (op 526) reg expression

buffer_load_ubyte v29, v25, s[76:79], 0 offen offset: 0 // (op 526) Load value


s_lshl_b32 s8, s38, 3 // (op 2930) 
v_add_u32 v16, v16, s8 // (op 2930) 



s_barrier  // (op 4417) 
v_mov_b32 v10, v28 // (op 527) 



v_mov_b32 v25, v10 // (op 527) reg expression

buffer_load_dword v20, v25, s[80:83], 0 offen // (op 527) Load value


s_lshl_b32 s8, s44, 3 // (op 3058) 
v_add_u32 v28, v28, s8 // (op 3058) 

s_waitcnt vmcnt(1)
ds_write_b8 v24, v29 offset:41472 // (op 523) Store local data 

s_barrier  // (op 4384) 



s_barrier  // (op 4438) 
ds_read_b64_tr_b8 v[34:35], v3 offset:16384 // (op 505) Transpose load local data 



ds_read_b64_tr_b8 v[36:37], v3 offset:16512 // (op 505) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:17408 // (op 505) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v3 offset:17536 // (op 505) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v32, v20 offset:40960 // (op 524) Store local data 

s_barrier  // (op 4404) 



ds_read_b64_tr_b8 v[42:43], v13 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v13 offset:512 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v13 offset:4096 // (op 507) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v13 offset:4608 // (op 507) Transpose load local data 
s_waitcnt lgkmcnt(9)
s_barrier  // (op 4425) 


ds_read_u8 v10, v6 offset:41472 // (op 509) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 4445) 


ds_read_u8 v25, v12 offset:40960 // (op 511) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:49], a[0:3], v10, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 512) 





ds_read_b64_tr_b8 v[42:43], v3 offset:18432 // (op 514) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:18560 // (op 514) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v3 offset:19456 // (op 514) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v3 offset:19584 // (op 514) Transpose load local data 



ds_read_b64_tr_b8 v[50:51], v13 offset:8192 // (op 516) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v13 offset:8704 // (op 516) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v13 offset:12288 // (op 516) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v13 offset:12800 // (op 516) Transpose load local data 


ds_read_u8 v25, v6 offset:41536 // (op 518) Load local data 


ds_read_u8 v29, v12 offset:41216 // (op 520) Load local data 



ds_read_b64_tr_b8 v[58:59], v13 offset:32 // (op 716) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v13 offset:544 // (op 716) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v13 offset:4128 // (op 716) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v13 offset:4640 // (op 716) Transpose load local data 


ds_read_u8 v33, v12 offset:40992 // (op 720) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:41], v[58:65], a[4:7], v10, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 721) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[50:57], a[0:3], v25, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 521) 
ds_read_b64_tr_b8 v[34:35], v13 offset:8224 // (op 725) Transpose load local data 


ds_read_b64_tr_b8 v[36:37], v13 offset:8736 // (op 725) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v13 offset:12320 // (op 725) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v13 offset:12832 // (op 725) Transpose load local data 


ds_read_u8 v10, v12 offset:41248 // (op 729) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[34:41], a[4:7], v25, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 730) 




s_ashr_i64 s[18:19], s[2:3], 8 // (op 708) 
s_mov_b32 s8, s18 // (op 708) convert

s_cmp_lt_u32 s7, s8 // (op 708) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_708 // (op 708) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_708 if true)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_708:









s_barrier  // (op 4609) 




v_cmp_eq_f32 vcc, 0.00000, v18 // (op 278) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_278 // (op 278) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_278
v_accvgpr_read v20, a0 // (op 286) 
v_accvgpr_read v21, a1 // (op 286) 
v_accvgpr_read v22, a2 // (op 286) 
v_accvgpr_read v23, a3 // (op 286) 
v_mul_f32 v24, v19, v20 // (op 286) 
v_mul_f32 v25, v19, v21 // (op 286) 
v_mul_f32 v26, v19, v22 // (op 286) 
v_mul_f32 v27, v19, v23 // (op 286) 

v_cvt_f16_f32 v20, v24 // (op 288) 
v_cvt_f16_f32 v21, v25 // (op 288) 
v_cvt_f16_f32 v22, v26 // (op 288) 
v_cvt_f16_f32 v23, v27 // (op 288) 


v_pack_b32_f16 v10, v20, v21 // (op 289) 
v_pack_b32_f16 v11, v22, v23 // (op 289) 

v_mov_b32 v0, v9 // (op 289) reg expression

buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 289) Store value



v_accvgpr_read v20, a4 // (op 1014) 
v_accvgpr_read v21, a5 // (op 1014) 
v_accvgpr_read v22, a6 // (op 1014) 
v_accvgpr_read v23, a7 // (op 1014) 
v_mul_f32 v24, v19, v20 // (op 1014) 
v_mul_f32 v25, v19, v21 // (op 1014) 
v_mul_f32 v26, v19, v22 // (op 1014) 
v_mul_f32 v27, v19, v23 // (op 1014) 

v_cvt_f16_f32 v20, v24 // (op 1016) 
v_cvt_f16_f32 v21, v25 // (op 1016) 
v_cvt_f16_f32 v22, v26 // (op 1016) 
v_cvt_f16_f32 v23, v27 // (op 1016) 


v_pack_b32_f16 v10, v20, v21 // (op 1017) 
v_pack_b32_f16 v11, v22, v23 // (op 1017) 

s_lshl_b32 s15, s24, 6 // (op 1017) 
v_add_u32 v0, v9, s15 // (op 1017) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1017) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_278 // (op 278) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_278
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_278:


v_mov_b32 v0, v7 // (op 303) reg expression

buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 303) Load value

v_accvgpr_read v20, a0 // (op 299) 
v_accvgpr_read v21, a1 // (op 299) 
v_accvgpr_read v22, a2 // (op 299) 
v_accvgpr_read v23, a3 // (op 299) 
v_mul_f32 v24, v19, v20 // (op 299) 
v_mul_f32 v25, v19, v21 // (op 299) 
v_mul_f32 v26, v19, v22 // (op 299) 
v_mul_f32 v27, v19, v23 // (op 299) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v10 // (op 304) 
v_lshrrev_b32 v17, 16, v10 // (op 304) 
v_cvt_f32_f16 v17, v17 // (op 304) 
v_mul_f32 v20, v18, v16 // (op 304) 
v_mul_f32 v21, v18, v17 // (op 304) 
v_cvt_f32_f16 v16, v11 // (op 304) 
v_lshrrev_b32 v17, 16, v11 // (op 304) 
v_cvt_f32_f16 v17, v17 // (op 304) 
v_mul_f32 v22, v18, v16 // (op 304) 
v_mul_f32 v23, v18, v17 // (op 304) 


v_add_f32 v28, v20, v24 // (op 300) 
v_add_f32 v29, v21, v25 // (op 300) 
v_add_f32 v30, v22, v26 // (op 300) 
v_add_f32 v31, v23, v27 // (op 300) 


v_cvt_f16_f32 v20, v28 // (op 301) 
v_cvt_f16_f32 v21, v29 // (op 301) 
v_cvt_f16_f32 v22, v30 // (op 301) 
v_cvt_f16_f32 v23, v31 // (op 301) 


v_pack_b32_f16 v10, v20, v21 // (op 302) 
v_pack_b32_f16 v11, v22, v23 // (op 302) 

v_mov_b32 v0, v9 // (op 302) reg expression

buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 302) Store value





s_lshl_b32 s15, s4, 6 // (op 1041) 
v_add_u32 v0, v7, s15 // (op 1041) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 1041) Load value

v_accvgpr_read v20, a4 // (op 1037) 
v_accvgpr_read v21, a5 // (op 1037) 
v_accvgpr_read v22, a6 // (op 1037) 
v_accvgpr_read v23, a7 // (op 1037) 
v_mul_f32 v24, v19, v20 // (op 1037) 
v_mul_f32 v25, v19, v21 // (op 1037) 
v_mul_f32 v26, v19, v22 // (op 1037) 
v_mul_f32 v27, v19, v23 // (op 1037) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v16, v10 // (op 1042) 
v_lshrrev_b32 v17, 16, v10 // (op 1042) 
v_cvt_f32_f16 v17, v17 // (op 1042) 
v_mul_f32 v20, v18, v16 // (op 1042) 
v_mul_f32 v21, v18, v17 // (op 1042) 
v_cvt_f32_f16 v16, v11 // (op 1042) 
v_lshrrev_b32 v17, 16, v11 // (op 1042) 
v_cvt_f32_f16 v17, v17 // (op 1042) 
v_mul_f32 v22, v18, v16 // (op 1042) 
v_mul_f32 v23, v18, v17 // (op 1042) 


v_add_f32 v28, v20, v24 // (op 1038) 
v_add_f32 v29, v21, v25 // (op 1038) 
v_add_f32 v30, v22, v26 // (op 1038) 
v_add_f32 v31, v23, v27 // (op 1038) 


v_cvt_f16_f32 v20, v28 // (op 1039) 
v_cvt_f16_f32 v21, v29 // (op 1039) 
v_cvt_f16_f32 v22, v30 // (op 1039) 
v_cvt_f16_f32 v23, v31 // (op 1039) 


v_pack_b32_f16 v10, v20, v21 // (op 1040) 
v_pack_b32_f16 v11, v22, v23 // (op 1040) 

s_lshl_b32 s15, s24, 6 // (op 1040) 
v_add_u32 v0, v9, s15 // (op 1040) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 1040) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_278:
s_add_u32 s0, s0, s1 // (op 491) 


s_mul_i32 s15, s50, s16 // (op 489) 
s_cmp_gt_u32 s51, s15 // (op 489) 
s_mov_b32 s6, scc // (op 489) 
s_sub_u32 s7, s51, s15 // (op 489) 

s_cmp_eq_i32 s6, 0 // (op 489) 
s_cselect_b32 s8, 0, s7 // (op 489) 


s_cmp_lt_u32 s50, s8 // (op 489) 
s_mov_b32 s15, scc // (op 489) 
s_cmp_eq_i32 s15, 0 // (op 489) 
s_cselect_b32 s6, s8, s50 // (op 489) 


s_cmp_lt_u32 s0, s6 // (op 489) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_DPStreamTileLoop_489 // (op 489) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopTop_DPStreamTileLoop_489 if true)
RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_ForLoopBottom_DPStreamTileLoop_489:











































s_endpgm  // End of RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2
.LRR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_end:
.size RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2, .LRR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2_end-RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2
  .amdhsa_next_free_vgpr 144
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 42240 // lds bytes
  .amdhsa_accum_offset 136
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
    DataTypeB: B8
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
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
    macrotile: [16, 64, 256]
    threads: [128, 1, 1]
    grid: [RocRollerStreamK, One, One]
    occupancy: 3
    workspaceType: RocRollerStreamK
    workspaceSizePerElemC: 4
    workspaceSizePerElemBias: 0
    unroll: 2
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2
    .symbol: RR_GEMM_NT_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_SK_WGT_16x64x256_UR_2.kd
    .kernarg_segment_size: 184
    .group_segment_fixed_size: 42240
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 91
    .vgpr_count: 134
    .agpr_count: 8
    .max_flat_workgroup_size: 128
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_size_1_9
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
      - .name: Tensor_2_size_1_19
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
      - .name: Tensor_2_stride_0_11
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_13
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_14
        .size: 8
        .offset: 144
        .value_kind: by_value
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

